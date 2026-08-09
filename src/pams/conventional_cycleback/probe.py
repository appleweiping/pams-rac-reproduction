"""Unified 256-optimizer-step mechanism probe for cycle-back TCC.

This is a bounded train337-only scientific probe, not the historical epoch-11
gate and not a full baseline training run.  Its artifact type and contract are
intentionally incompatible with the old gate.  A scientific PASS also
publishes the exact learned-L/AdamW/RNG seed, but grants no continuation
authority.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pams.conventional_cycleback.audit import _evaluate_embeddings
from pams.conventional_cycleback.authority import (
    validate_cycleback_launch_registry,
    validate_cycleback_pose_authority,
)
from pams.conventional_cycleback.config import (
    ConventionalCycleBackConfig,
    MechanismProbeThresholds,
    parse_conventional_cycleback_config,
)
from pams.conventional_cycleback.fullcontext_trainer import (
    FullContextTrainerContract,
    MechanismSeedPredecessorLineage,
    MechanismSeedSamplerState,
    build_fullcontext_adamw,
    build_fullcontext_objective,
    build_mechanism_seed_checkpoint_payload,
    capture_fullcontext_backend_state,
    capture_fullcontext_rng_state,
    configure_fullcontext_determinism,
    fullcontext_backend_state_sha256,
    fullcontext_rng_state_sha256,
    model_state_sha256,
    optimizer_state_sha256,
    preserve_fullcontext_diagnostic_state,
    unified2d_fullcontext_representation_contract,
    validate_mechanism_seed_checkpoint_payload,
)
from pams.conventional_cycleback.loss import ConventionalCycleBackLoss
from pams.conventional_cycleback.runtime import (
    PairEligibility,
    PairSegmentContexts,
    Unified2DSequence,
    build_encoder,
    cap_window_pairs,
    collate_to_device,
    encode_window_pairs,
    independently_permuted_pair_segment_contexts,
    independently_permuted_pair_segment_positions,
    joint_support_null_segment_contexts,
    load_identity_map,
    load_joint_mask_snapshot,
    load_pair_eligibility,
    load_segment_index,
    load_snapshot_sequences,
    load_stable_json_object,
    pair_segment_contexts,
    pairs_from_batch,
    ranked_sequences,
    require_real_optimizer_contexts,
    stable_file_bytes,
    stable_file_identity,
    zero_pair_segment_contexts,
)
from pams.conventional_cycleback.windows import (
    NativeWindowPairBatch,
    enumerate_native_window_pairs,
)
from pams.model import PAMSEncoder

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_CANONICAL_GEOMETRY_PARENT = Path(
    "/media/lenovo/data2/pams-rac/runs/pams-conventional-cycleback-geometry-v1"
)


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _validate_label_firewall(
    payload: Mapping[str, Any],
    *,
    expected_inputs: set[str],
) -> None:
    firewall = payload.get("label_firewall")
    if not isinstance(firewall, Mapping):
        raise ValueError("label firewall is missing")
    accepted = firewall.get("accepted_inputs")
    if not isinstance(accepted, list) or set(accepted) != expected_inputs:
        raise ValueError("label firewall accepted-input schema mismatch")
    for key in (
        "manifest_interface_supported",
        "media_interface_supported",
        "action_or_repetition_annotation_interface_supported",
        "development_pose_or_annotation_mounted",
        "sealed_pose_or_annotation_mounted",
    ):
        if firewall.get(key) is not False:
            raise ValueError(f"label firewall illegally sets {key}")
    if firewall.get("external_label_fields_accessed") != []:
        raise ValueError("label firewall reports external label access")
    if (
        firewall.get("video_id_handling")
        != "opaque_identifier_for_hash_order_and_same_video_equality_only"
        or firewall.get("video_id_tokens_parsed") is not False
    ):
        raise ValueError("label firewall does not treat video IDs as opaque")


def _device(value: str | torch.device | None) -> torch.device:
    if value is None or str(value).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(value)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return resolved


def _load_geometry_authorization(
    path: str | Path,
    *,
    expected_sha256: str,
    config: ConventionalCycleBackConfig,
    config_sha256: str,
    config_bytes: int,
    snapshot_sha256: str,
    snapshot_bytes: int,
    pose_cache_set_sha256: str,
    segment_index_sha256: str,
    segment_reset_policy_sha256: str,
    joint_mask_snapshot_sha256: str,
    joint_mask_set_sha256: str,
    identity_map_sha256: str,
    cycleback_pair_eligibility_sha256: str,
    pair_eligibility: PairEligibility,
    pose_input_authorization_sha256: str,
    pose_input_authority_run_receipt_sha256: str,
    representation_bindings: Mapping[str, Any],
    launch_registry_id: str,
    launch_authorization_sha256: str,
    launch_registry_receipt_sha256: str,
    source_git_sha: str,
    source_tree_sha: str,
    container_image_id: str,
) -> tuple[dict[str, Any], tuple[str, int]]:
    if not _SHA256.fullmatch(expected_sha256):
        raise ValueError("expected geometry-gate SHA-256 must be lowercase hexadecimal")
    payload, identity = load_stable_json_object(path, role="geometry-gate artifact")
    if identity[0] != expected_sha256:
        raise ValueError("geometry-gate bytes differ from the caller-pinned SHA-256")
    if payload.get("artifact_type") != config.geometry_audit.artifact_type:
        raise ValueError("wrong geometry-gate artifact type")
    if payload.get("schema_version") != 1 or payload.get("status") != "passed":
        raise ValueError("geometry-gate schema or status mismatch")
    if payload.get("namespace") != config.namespace:
        raise ValueError("geometry-gate namespace mismatch")
    if payload.get("protocol") != config.protocol:
        raise ValueError("geometry-gate protocol mismatch")
    if payload.get("candidate_id") != config.candidate_id:
        raise ValueError("geometry-gate candidate mismatch")
    if payload.get("classification") != config.classification:
        raise ValueError("geometry-gate classification mismatch")
    if payload.get("paper_table_claim_eligible") is not False:
        raise ValueError("geometry gate illegally claims paper-table authority")
    _validate_label_firewall(
        payload,
        expected_inputs={
            "cycleback_proxy_config",
            "sealed_cycleback_pose_input_authority",
        },
    )
    inputs = payload.get("inputs")
    gate = payload.get("gate")
    if not isinstance(inputs, dict) or not isinstance(gate, dict):
        raise ValueError("geometry-gate input or decision schema mismatch")
    expected_inputs = {
        "config_sha256": config_sha256,
        "config_bytes": config_bytes,
        "config_fingerprint": config.fingerprint,
        "pose_snapshot_sha256": snapshot_sha256,
        "pose_snapshot_bytes": snapshot_bytes,
        "pose_cache_set_sha256": pose_cache_set_sha256,
        "segment_index_sha256": segment_index_sha256,
        "segment_reset_policy_sha256": segment_reset_policy_sha256,
        "joint_mask_snapshot_sha256": joint_mask_snapshot_sha256,
        "joint_mask_set_sha256": joint_mask_set_sha256,
        "identity_map_sha256": identity_map_sha256,
        "cycleback_pair_eligibility_sha256": (
            cycleback_pair_eligibility_sha256
        ),
        "pair_start_grid_policy": pair_eligibility.start_grid_policy,
        "frozen_joint_support_thresholds": {
            "minimum_window_stable_action_joints": (
                pair_eligibility.minimum_window_stable_action_joints
            ),
            "minimum_window_joint_support_fraction": (
                pair_eligibility.minimum_window_joint_support_fraction
            ),
        },
        "pose_input_authorization_sha256": pose_input_authorization_sha256,
        "pose_input_authority_run_receipt_sha256": (
            pose_input_authority_run_receipt_sha256
        ),
        "representation_bindings": dict(representation_bindings),
        "launch_registry_id": launch_registry_id,
        "launch_authorization_sha256": launch_authorization_sha256,
        "launch_registry_receipt_sha256": launch_registry_receipt_sha256,
        "training_video_total": config.geometry_audit.expected_training_video_total,
        "source_git_sha": source_git_sha,
        "source_tree_sha": source_tree_sha,
        "container_image_id": container_image_id,
        "encoder_segment_context_batch_size": (
            config.geometry_audit.encoder_segment_context_batch_size
        ),
    }
    for name, expected in expected_inputs.items():
        if inputs.get(name) != expected:
            raise ValueError(f"geometry-gate binding mismatch: {name}")
    if (
        gate.get("overall_pass") is not True
        or gate.get("mechanism_probe_authorized") is not True
    ):
        raise ValueError("geometry gate did not authorize the 256-step mechanism probe")
    criteria = gate.get("criteria")
    if not isinstance(criteria, dict) or not criteria:
        raise ValueError("geometry gate criteria are missing")
    if any(
        not isinstance(item, dict) or item.get("passed") is not True
        for item in criteria.values()
    ):
        raise ValueError("geometry gate contains a failed or malformed criterion")
    for key in (
        "epoch11_encoder_continuation_authorized",
        "development_evaluation_authorized",
        "sealed_evaluation_authorized",
    ):
        if gate.get(key) is not False:
            raise ValueError(f"geometry gate illegally sets {key}")
    verification = payload.get("read_only_verification")
    expected_verification = {
        "config_unchanged": True,
        "pose_snapshot_unchanged": True,
        "pose_segment_index_unchanged": True,
        "joint_mask_snapshot_unchanged": True,
        "identity_map_unchanged": True,
        "pair_eligibility_unchanged": True,
        "all_pose_cache_bytes_bound_to_snapshot": True,
        "optimizer_steps": 0,
        "model_state_updated": False,
        "pose_cache_write_operations": 0,
    }
    if not isinstance(verification, Mapping) or any(
        verification.get(key) != expected
        for key, expected in expected_verification.items()
    ):
        raise ValueError("geometry gate read-only verification mismatch")
    return payload, identity


def _load_geometry_run_receipt(
    path: str | Path,
    *,
    expected_sha256: str,
    declared_host_root: str | Path,
    expected_gate_sha256: str,
    config: ConventionalCycleBackConfig,
    config_sha256: str,
    config_bytes: int,
    pose_authorization_sha256: str,
    pose_run_receipt_sha256: str,
    launch_registry_id: str,
    launch_authorization_sha256: str,
    launch_registry_receipt_sha256: str,
    source_git_sha: str,
    source_tree_sha: str,
    container_image_id: str,
) -> tuple[dict[str, Any], tuple[str, int]]:
    if not _SHA256.fullmatch(expected_sha256):
        raise ValueError("expected geometry run-receipt SHA-256 is invalid")
    receipt, identity = load_stable_json_object(path, role="geometry run receipt")
    if identity[0] != expected_sha256:
        raise ValueError("geometry run receipt differs from the outcome registry")
    declared = Path(declared_host_root)
    if (
        declared.parent != _CANONICAL_GEOMETRY_PARENT
        or not declared.name.startswith(f"{source_git_sha[:12]}-")
    ):
        raise ValueError("geometry run is outside the canonical namespace")
    mounted_root = Path(path).resolve(strict=True).parents[1]
    if mounted_root.is_symlink() or any(
        candidate.is_symlink() or candidate.lstat().st_mode & 0o222
        for candidate in (mounted_root, *mounted_root.rglob("*"))
    ):
        raise ValueError("geometry run root is not sealed or contains a symlink")
    if (mounted_root / "audit/failure.receipt.json").exists():
        raise ValueError("passed geometry run contains a failure receipt")
    if (
        receipt.get("artifact_type")
        != "pams_conventional_cycleback_geometry_run_receipt_v1"
        or receipt.get("schema_version") != 1
        or receipt.get("stage") != "geometry"
        or receipt.get("status") != "passed"
        or receipt.get("container_exit_code") != 0
        or receipt.get("candidate_id") != config.candidate_id
    ):
        raise ValueError("geometry run receipt status/schema mismatch")
    expected_lineage = {
        "source_git_sha": source_git_sha,
        "source_tree_sha": source_tree_sha,
        "container_image_id": container_image_id,
        "launch_registry_id": launch_registry_id,
        "launch_authorization_sha256": launch_authorization_sha256,
        "launch_registry_receipt_sha256": launch_registry_receipt_sha256,
        "geometry_gate_sha256": expected_gate_sha256,
    }
    if any(receipt.get(key) != value for key, value in expected_lineage.items()):
        raise ValueError("geometry run receipt lineage mismatch")
    if receipt.get("config") != {"sha256": config_sha256, "bytes": config_bytes}:
        raise ValueError("geometry run receipt config mismatch")
    predecessors = receipt.get("predecessors")
    if not isinstance(predecessors, list) or len(predecessors) != 1:
        raise ValueError("geometry run receipt predecessor schema mismatch")
    pose = predecessors[0]
    if (
        not isinstance(pose, Mapping)
        or pose.get("role") != "pose_input"
        or pose.get("output", {}).get("sha256") != pose_authorization_sha256
        or pose.get("receipt", {}).get("sha256") != pose_run_receipt_sha256
    ):
        raise ValueError("geometry run receipt pose predecessor mismatch")
    for field in (
        "scientific_authority_granted",
        "baseline_training_authorized",
        "full_training_authorized",
        "full_encoder_training_authorized",
        "epoch11_encoder_continuation_authorized",
        "development_evaluation_authorized",
        "sealed_evaluation_authorized",
    ):
        if receipt.get(field) is not False:
            raise ValueError(f"geometry run receipt illegally sets {field}")
    cleanup = receipt.get("container_cleanup")
    container_id = receipt.get("container_id")
    if (
        not isinstance(cleanup, Mapping)
        or cleanup.get("removed_by_exact_id") != container_id
        or cleanup.get("id_verified_before_remove") is not True
        or cleanup.get("remove_succeeded") is not True
        or cleanup.get("absence_verified") is not True
    ):
        raise ValueError("geometry run exact-ID cleanup is not proven")
    container_name = receipt.get("container_name")
    expected_name = re.compile(
        rf"^pams-cycleback-geometry-{source_git_sha[:12]}-"
        r"[a-z0-9][a-z0-9._-]{0,39}$"
    )
    if not isinstance(container_name, str) or not expected_name.fullmatch(container_name):
        raise ValueError("geometry receipt container name is unsafe")
    evidence = {
        "create_id_sha256": mounted_root / "audit" / f"{container_name}.create-id.txt",
        "docker_create_cidfile_sha256": (
            mounted_root / "audit" / f"{container_name}.docker-create.cid"
        ),
        "docker_create_stdout_sha256": (
            mounted_root / "audit" / f"{container_name}.docker-create.stdout"
        ),
        "docker_create_stderr_sha256": (
            mounted_root / "audit" / f"{container_name}.docker-create.stderr"
        ),
        "container_contract_sha256": mounted_root / "audit/container.contract.json",
        "pre_run_inspect_sha256": (
            mounted_root / "audit" / f"{container_name}.pre-run.inspect.json"
        ),
        "post_run_inspect_sha256": (
            mounted_root / "audit" / f"{container_name}.post-run.inspect.json"
        ),
        "source_export_pre_manifest_sha256": (
            mounted_root / "audit/source-export.pre.sha256"
        ),
        "source_export_post_manifest_sha256": (
            mounted_root / "audit/source-export.post.sha256"
        ),
        "pre_receipt_artifact_manifest_sha256": (
            mounted_root / "audit/pre-receipt-artifact.sha256"
        ),
        "stage_reservation_archive_sha256": mounted_root / "audit/stage.reservation.json",
    }
    for key, evidence_path in evidence.items():
        if stable_file_identity(evidence_path)[0] != receipt.get(key):
            raise ValueError(f"geometry receipt evidence hash mismatch: {key}")
    expected_raw_id = str(container_id).removeprefix("sha256:")
    for key in ("docker_create_cidfile_sha256", "docker_create_stdout_sha256"):
        try:
            observed_id = stable_file_bytes(evidence[key])[0].decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"geometry {key} is not ASCII") from exc
        if observed_id.removeprefix("sha256:") != expected_raw_id:
            raise ValueError(f"geometry {key} differs from the created container ID")
    if stable_file_bytes(evidence["docker_create_stderr_sha256"])[0]:
        raise ValueError("geometry Docker create unexpectedly wrote stderr")
    if (
        receipt.get("source_export_manifests_identical") is not True
        or receipt.get("source_export_pre_manifest_sha256")
        != receipt.get("source_export_post_manifest_sha256")
    ):
        raise ValueError("geometry source export changed during execution")
    return receipt, identity


def _eligible_sequences(
    sequences: Sequence[Unified2DSequence],
    config: ConventionalCycleBackConfig,
    *,
    segment_ranges_by_video: Mapping[str, Sequence[tuple[int, int]]],
    pair_eligibility: PairEligibility,
) -> tuple[Unified2DSequence, ...]:
    """Filter solely by preregistered native-window pose validity geometry."""

    window = config.window_pair
    eligible: list[Unified2DSequence] = []
    for sequence in sequences:
        batch = collate_to_device((sequence,), torch.device("cpu"))
        pairs = enumerate_native_window_pairs(
            batch.poses,
            batch.poses,
            batch.valid_mask,
            batch.joint_valid_mask,
            batch.lengths,
            batch.video_ids,
            (segment_ranges_by_video[sequence.video_id],),
            (
                pair_eligibility.base_valid_starts_by_variant[
                    config.candidate_id
                ][sequence.video_id],
            ),
            (
                pair_eligibility.starts_by_variant[config.candidate_id][
                    sequence.video_id
                ],
            ),
            window_length=window.length_frames,
            hop_frames=window.hop_frames,
            minimum_valid_frames_per_window=window.minimum_valid_frames_per_window,
            minimum_window_stable_action_joints=(
                pair_eligibility.minimum_window_stable_action_joints
            ),
            minimum_window_joint_support_fraction=(
                pair_eligibility.minimum_window_joint_support_fraction
            ),
        )
        if pairs.pair_count:
            eligible.append(sequence)
    if len(eligible) < config.mechanism_probe.video_batch_size:
        raise ValueError("too few geometry-eligible training videos for mechanism probe")
    return tuple(eligible)


def _cyclic_video_batch(
    sequences: Sequence[Unified2DSequence],
    *,
    step: int,
    batch_size: int,
) -> tuple[Unified2DSequence, ...]:
    start = (step * batch_size) % len(sequences)
    return tuple(sequences[(start + offset) % len(sequences)] for offset in range(batch_size))


def _update_mechanism_sampler_prefix(
    video_batch_digest: Any,
    pair_row_digest: Any,
    *,
    step: int,
    selected: Sequence[Unified2DSequence],
    pairs: NativeWindowPairBatch,
) -> None:
    video_batch_digest.update(
        _canonical_json_bytes(
            {
                "optimizer_step": step,
                "video_ids": [sequence.video_id for sequence in selected],
            }
        )
    )
    pair_row_digest.update(
        _canonical_json_bytes(
            {
                "optimizer_step": step,
                "video_ids": list(pairs.video_ids),
                "starts_a": pairs.starts_a.detach().cpu().tolist(),
                "starts_b": pairs.starts_b.detach().cpu().tolist(),
                "segment_starts": pairs.segment_starts.detach().cpu().tolist(),
                "segment_ends": pairs.segment_ends.detach().cpu().tolist(),
                "source_indices_a": (
                    pairs.source_indices_a.detach().cpu().tolist()
                ),
                "source_indices_b": (
                    pairs.source_indices_b.detach().cpu().tolist()
                ),
                "native_lengths": pairs.native_lengths.detach().cpu().tolist(),
            }
        )
    )


def _finite_parameter_gradients(model: PAMSEncoder) -> bool:
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    return bool(gradients) and all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )


def _median(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise RuntimeError("mechanism probe loss history is empty or non-finite")
    return float(np.median(array))


def _finite_positive_ratio(numerator: float, denominator: float) -> float | None:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or denominator <= 0.0
    ):
        return None
    return numerator / denominator


def _one_pair_per_video(pairs: NativeWindowPairBatch) -> Tensor:
    selected: list[int] = []
    seen: set[str] = set()
    for index, identifier in enumerate(pairs.video_ids):
        if identifier not in seen:
            seen.add(identifier)
            selected.append(index)
    if len(selected) < 2:
        raise ValueError("different-video control requires at least two videos")
    return torch.tensor(selected, dtype=torch.long, device=pairs.poses_a.device)


def _different_video_control(
    objective: ConventionalCycleBackLoss,
    pairs: NativeWindowPairBatch,
    embeddings_a: Tensor,
    embeddings_b: Tensor,
) -> dict[str, Any]:
    selected_indices = _one_pair_per_video(pairs)
    selected = pairs.select(selected_indices)
    selected_a = embeddings_a[selected_indices]
    selected_b = embeddings_b[selected_indices]
    same = objective.compute(
        selected_a,
        selected_b,
        selected.valid_a,
        selected.valid_b,
        selected.source_indices_a,
        selected.source_indices_b,
        selected.native_lengths,
        video_ids_a=selected.video_ids,
        video_ids_b=selected.video_ids,
    )
    permutation = torch.roll(
        torch.arange(selected.pair_count, device=selected_a.device),
        shifts=1,
    )
    mismatched_ids = tuple(selected.video_ids[int(index)] for index in permutation.cpu())
    different = objective.diagnostic_mismatched_video_control(
        selected_a,
        selected_b[permutation],
        selected.valid_a,
        selected.valid_b[permutation],
        selected.source_indices_a,
        selected.source_indices_b[permutation],
        selected.native_lengths,
        selected.native_lengths[permutation],
        video_ids_a=selected.video_ids,
        video_ids_b=mismatched_ids,
    )
    same_mse = float(
        0.5
        * (
            same.a_to_b_to_a.mean_squared_position_error
            + same.b_to_a_to_b.mean_squared_position_error
        ).detach()
    )
    different_mse = float(
        0.5
        * (
            different.a_to_b_to_a.mean_squared_position_error
            + different.b_to_a_to_b.mean_squared_position_error
        ).detach()
    )
    return {
        "control_pair_total": selected.pair_count,
        "same_video_symmetric_loss": float(same.total.detach()),
        "different_video_symmetric_loss": float(different.total.detach()),
        "same_video_symmetric_position_mse": same_mse,
        "different_video_symmetric_position_mse": different_mse,
        "different_minus_same_position_mse": different_mse - same_mse,
        "used_as_training_negative": False,
        "used_for_optimizer_update": False,
        "all_control_rows_are_different_video": all(
            first != second
            for first, second in zip(selected.video_ids, mismatched_ids, strict=True)
        ),
    }


def mechanism_probe_gate_decision(
    *,
    optimizer_steps: int,
    loss_relative_drop: float | None,
    final_temporal_rms_median: float | None,
    real_null_position_error_gap: float | None,
    final_valid_anchor_fraction: float | None,
    final_mask_flicker_to_real_position_error_ratio: float | None,
    final_torso_only_to_real_position_error_ratio: float | None,
    final_alternating_limb_dropout_to_real_position_error_ratio: float | None,
    final_permuted_pe_to_real_position_error_ratio: float | None,
    final_pe_off_to_real_position_error_ratio: float | None,
    thresholds: MechanismProbeThresholds,
) -> dict[str, Any]:
    def criterion(
        value: float | int | None,
        relation: str,
        threshold: float | int,
    ) -> dict[str, Any]:
        finite = value is not None and math.isfinite(float(value))
        if relation == "equal":
            passed = finite and value == threshold
        elif relation == "at_least":
            passed = finite and value >= threshold
        elif relation == "at_most":
            passed = finite and value <= threshold
        else:
            raise ValueError("unsupported mechanism criterion relation")
        return {
            "value": value,
            "relation": relation,
            "threshold": threshold,
            "passed": bool(passed),
        }

    criteria = {
        "optimizer_steps": criterion(optimizer_steps, "equal", 256),
        "loss_relative_drop": criterion(
            loss_relative_drop,
            "at_least",
            thresholds.minimum_loss_relative_drop,
        ),
        "final_temporal_rms_median": criterion(
            final_temporal_rms_median,
            "at_least",
            thresholds.minimum_final_temporal_rms_median,
        ),
        "real_null_position_error_gap": criterion(
            real_null_position_error_gap,
            "at_least",
            thresholds.minimum_real_null_position_error_gap,
        ),
        "final_valid_anchor_fraction": criterion(
            final_valid_anchor_fraction,
            "at_least",
            thresholds.minimum_final_valid_anchor_fraction,
        ),
        "final_mask_flicker_to_real_position_error_ratio": criterion(
            final_mask_flicker_to_real_position_error_ratio,
            "at_least",
            thresholds.minimum_final_mask_flicker_to_real_position_error_ratio,
        ),
        "final_torso_only_to_real_position_error_ratio": criterion(
            final_torso_only_to_real_position_error_ratio,
            "at_least",
            thresholds.minimum_final_torso_only_to_real_position_error_ratio,
        ),
        "final_alternating_limb_dropout_to_real_position_error_ratio": criterion(
            final_alternating_limb_dropout_to_real_position_error_ratio,
            "at_least",
            thresholds.minimum_final_alternating_limb_dropout_to_real_position_error_ratio,
        ),
        "final_permuted_pe_to_real_position_error_ratio_minimum": criterion(
            final_permuted_pe_to_real_position_error_ratio,
            "at_least",
            thresholds.minimum_final_permuted_pe_to_real_position_error_ratio,
        ),
        "final_permuted_pe_to_real_position_error_ratio_maximum": criterion(
            final_permuted_pe_to_real_position_error_ratio,
            "at_most",
            thresholds.maximum_final_permuted_pe_to_real_position_error_ratio,
        ),
        "final_pe_off_to_real_position_error_ratio_minimum": criterion(
            final_pe_off_to_real_position_error_ratio,
            "at_least",
            thresholds.minimum_final_pe_off_to_real_position_error_ratio,
        ),
        "final_pe_off_to_real_position_error_ratio_maximum": criterion(
            final_pe_off_to_real_position_error_ratio,
            "at_most",
            thresholds.maximum_final_pe_off_to_real_position_error_ratio,
        ),
    }
    passed = all(bool(item["passed"]) for item in criteria.values())
    return {
        "gate_contract": "cycleback_256_optimizer_steps_not_epoch11_gate",
        "thresholds_frozen_before_execution": True,
        "criteria": criteria,
        "overall_pass": passed,
        "cycleback_mechanism_supported": passed,
        "epoch11_encoder_continuation_authorized": False,
        "full_encoder_training_authorized": False,
        "development_evaluation_authorized": False,
        "sealed_evaluation_authorized": False,
    }


def validate_mechanism_probe_contract(
    payload: Mapping[str, Any],
    config: ConventionalCycleBackConfig,
) -> None:
    """Reject old or partial gate artifacts masquerading as this probe."""

    if payload.get("artifact_type") != config.mechanism_probe.artifact_type:
        raise ValueError("artifact is not the cycle-back 256-step mechanism probe")
    if payload.get("namespace") != config.namespace:
        raise ValueError("mechanism probe namespace mismatch")
    if payload.get("protocol") != config.protocol:
        raise ValueError("mechanism probe protocol mismatch")
    if payload.get("classification") != config.classification:
        raise ValueError("mechanism probe classification mismatch")
    if payload.get("candidate_id") != config.candidate_id:
        raise ValueError("mechanism probe candidate mismatch")
    if payload.get("paper_table_claim_eligible") is not False:
        raise ValueError("mechanism probe illegally claims paper-table authority")
    if payload.get("schema_version") != 1:
        raise ValueError("mechanism probe schema version mismatch")
    _validate_label_firewall(
        payload,
        expected_inputs={
            "cycleback_proxy_config",
            "passed_cycleback_geometry_gate",
            "sealed_cycleback_pose_input_authority",
        },
    )
    if payload.get("config_fingerprint") != config.fingerprint:
        raise ValueError("mechanism probe config fingerprint mismatch")
    trainer_contract = FullContextTrainerContract.from_candidate_config(config)
    training = payload.get("training")
    gate = payload.get("gate")
    if not isinstance(training, Mapping) or not isinstance(gate, Mapping):
        raise ValueError("mechanism probe training or gate schema mismatch")
    if training.get("optimizer_steps_completed") != 256:
        raise ValueError("mechanism probe did not complete exactly 256 optimizer steps")
    if (
        training.get("optimizer") != "AdamW"
        or training.get("optimizer_contract")
        != trainer_contract.to_dict()["optimizer"]
        or training.get("scheduler") != "none"
        or training.get("trainer_contract_fingerprint")
        != trainer_contract.fingerprint
    ):
        raise ValueError("mechanism probe optimizer/trainer contract mismatch")
    optimizer_state_sha256 = training.get("optimizer_state_sha256_at_step256")
    if not isinstance(optimizer_state_sha256, str) or not _SHA256.fullmatch(
        optimizer_state_sha256
    ):
        raise ValueError("mechanism probe optimizer state digest is invalid")
    sampler_payload = training.get("mechanism_sampler_state")
    if not isinstance(sampler_payload, Mapping) or set(sampler_payload) != {
        "policy",
        "eligible_video_total",
        "ordered_eligible_video_ids_sha256",
        "video_batch_size",
        "maximum_pairs_per_step",
        "completed_optimizer_steps",
        "consumed_video_batch_chain_sha256",
        "consumed_pair_row_chain_sha256",
        "next_cyclic_start_index",
        "mechanism_sampler_resume_allowed",
        "epoch1_sampler_requires_new_sealed_plan",
    }:
        raise ValueError("mechanism probe sampler state is missing")
    sampler_state = MechanismSeedSamplerState(**dict(sampler_payload))
    if (
        training.get("mechanism_sampler_state_sha256")
        != sampler_state.fingerprint
        or training.get("mechanism_consumed_video_batch_chain_sha256")
        != sampler_state.consumed_video_batch_chain_sha256
        or training.get("mechanism_consumed_pair_row_chain_sha256")
        != sampler_state.consumed_pair_row_chain_sha256
        or sampler_state.video_batch_size
        != trainer_contract.mechanism_video_batch_size
        or sampler_state.maximum_pairs_per_step
        != trainer_contract.mechanism_maximum_pairs_per_step
        or sampler_state.completed_optimizer_steps
        != trainer_contract.mechanism_optimizer_steps
    ):
        raise ValueError("mechanism probe sampler state differs from trainer contract")
    if training.get("objective") != "symmetric_variance_aware_cycleback_regression":
        raise ValueError("mechanism probe objective mismatch")
    if training.get("encoder_segment_context_batch_size") != (
        config.mechanism_probe.encoder_segment_context_batch_size
    ):
        raise ValueError("mechanism probe encoder context batch size mismatch")
    frozen_training_flags = {
        "same_video_window_pairs_only": True,
        "strictly_disjoint_source_index_sets": True,
        "all_window_frames_valid": True,
        "no_pair_crosses_segment_reset": True,
        "cross_video_positive_terms": False,
        "cross_video_negative_terms": False,
        "prototype_or_cluster_bank": False,
        "independent_skeleton_views_per_step": True,
        "full_authorized_stable_range_encoded_before_window_slice": True,
        "attention_never_crosses_authorized_stable_range": True,
        "training_inference_context_parity": True,
        "window_only_encoder_path_used": False,
        "deterministic_algorithms_enabled": True,
    }
    if any(training.get(key) is not expected for key, expected in frozen_training_flags.items()):
        raise ValueError("mechanism probe training-scope invariant mismatch")
    if (
        training.get("pair_policy") != config.window_pair.pair_policy
        or training.get("target_position_normalization")
        != config.objective.target_position_normalization
        or training.get("reset_count_aggregation_policy")
        != config.window_pair.reset_count_aggregation_policy
        or training.get("encoder_context_policy")
        != config.window_pair.encoder_context_policy
        or training.get("attention_boundary_policy")
        != config.window_pair.attention_boundary_policy
    ):
        raise ValueError("mechanism disjoint/window-local contract mismatch")
    if (
        training.get("gate_loss_source")
        != "fixed_label_free_evaluation_pair_set_step0_vs_step256"
    ):
        raise ValueError("mechanism gate loss is not based on the frozen evaluation set")
    fixed_evaluation = payload.get("fixed_evaluation")
    if not isinstance(fixed_evaluation, Mapping):
        raise ValueError("mechanism fixed evaluation is missing")
    for key in (
        "label_free",
        "same_native_pair_set_at_step0_and_step256",
        "same_augmented_view_bytes_at_step0_and_step256",
        "encoder_eval_mode",
    ):
        if fixed_evaluation.get(key) is not True:
            raise ValueError(f"mechanism fixed evaluation invariant failed: {key}")
    if fixed_evaluation.get("optimizer_updates_during_evaluation") != 0:
        raise ValueError("mechanism fixed evaluation updated the model")
    step0 = fixed_evaluation.get("step0")
    step256 = fixed_evaluation.get("step256")
    if not isinstance(step0, Mapping) or not isinstance(step256, Mapping):
        raise ValueError("mechanism fixed evaluation endpoints are missing")
    for key in (
        "pair_identity_sha256",
        "pair_payload_sha256",
        "pair_total",
        "unique_view_context_total",
        "reused_pair_side_context_references",
    ):
        if step0.get(key) != step256.get(key):
            raise ValueError(f"mechanism fixed evaluation endpoints differ: {key}")
    for endpoint, role in ((step0, "step0"), (step256, "step256")):
        for key in (
            "pair_identity_sha256",
            "pair_payload_sha256",
            "model_state_sha256",
        ):
            if not isinstance(endpoint.get(key), str) or not _SHA256.fullmatch(
                endpoint[key]
            ):
                raise ValueError(f"mechanism fixed evaluation {role} digest is invalid: {key}")
        if (
            not isinstance(endpoint.get("valid_anchor_total"), int)
            or endpoint["valid_anchor_total"] < 1
        ):
            raise ValueError(f"mechanism fixed evaluation {role} has no anchors")
    initial_loss = step0.get("symmetric_variance_aware_loss")
    final_loss = step256.get("symmetric_variance_aware_loss")
    if not isinstance(initial_loss, int | float) or not isinstance(
        final_loss, int | float
    ):
        raise ValueError("mechanism fixed evaluation loss is missing")
    expected_drop = None if initial_loss <= 0.0 else 1.0 - final_loss / initial_loss
    if fixed_evaluation.get("loss_relative_drop_used_by_gate") != expected_drop:
        raise ValueError("mechanism fixed evaluation loss-drop arithmetic mismatch")
    evaluation_scope = payload.get("evaluation_scope")
    if not isinstance(evaluation_scope, Mapping):
        raise ValueError("mechanism evaluation scope is missing")
    temporal_null = evaluation_scope.get("independent_temporal_null")
    if (
        not isinstance(temporal_null, Mapping)
        or temporal_null.get("side_seeds_are_distinct") is not True
        or temporal_null.get("selected_window_moved_fraction") != 1.0
        or temporal_null.get("selected_source_content_intersection_total") != 0
        or temporal_null.get("shared_selected_content_mapping") is not False
    ):
        raise ValueError("mechanism temporal-null contract mismatch")
    pe_null = evaluation_scope.get("independent_pe_null")
    if (
        not isinstance(pe_null, Mapping)
        or pe_null.get("side_seeds_are_distinct") is not True
        or pe_null.get("selected_moved_position_fraction") != 1.0
        or pe_null.get("selected_position_intersection_total") != 0
    ):
        raise ValueError("mechanism PE-null contract mismatch")
    joint_nulls = evaluation_scope.get("joint_support_nulls")
    if not isinstance(joint_nulls, Mapping) or set(joint_nulls) != {
        "mask_flicker",
        "torso_only",
        "alternating_limb_dropout",
    }:
        raise ValueError("mechanism joint-support null schema mismatch")
    for name, contract in joint_nulls.items():
        if (
            not isinstance(contract, Mapping)
            or contract.get("policy") != name
            or contract.get("label_free") is not True
            or contract.get("scope")
            != "full_authorized_stable_range_context"
            or contract.get("absolute_position_indices_unchanged") is not True
            or not isinstance(contract.get("dropped_fraction"), int | float)
            or not 0.0 < float(contract["dropped_fraction"]) <= 1.0
        ):
            raise ValueError(f"mechanism joint-support null mismatch: {name}")
    if gate.get("gate_contract") != config.mechanism_probe.gate_contract:
        raise ValueError("old epoch11 gate cannot satisfy the cycle-back probe contract")
    if gate.get("thresholds_frozen_before_execution") is not True:
        raise ValueError("mechanism probe thresholds were not frozen")
    criteria = gate.get("criteria")
    expected_criteria = {
        "optimizer_steps",
        "loss_relative_drop",
        "final_temporal_rms_median",
        "real_null_position_error_gap",
        "final_valid_anchor_fraction",
        "final_mask_flicker_to_real_position_error_ratio",
        "final_torso_only_to_real_position_error_ratio",
        "final_alternating_limb_dropout_to_real_position_error_ratio",
        "final_permuted_pe_to_real_position_error_ratio_minimum",
        "final_permuted_pe_to_real_position_error_ratio_maximum",
        "final_pe_off_to_real_position_error_ratio_minimum",
        "final_pe_off_to_real_position_error_ratio_maximum",
    }
    if not isinstance(criteria, Mapping) or set(criteria) != expected_criteria:
        raise ValueError("mechanism probe criterion schema mismatch")
    loss_criterion = criteria.get("loss_relative_drop")
    if (
        not isinstance(loss_criterion, Mapping)
        or loss_criterion.get("value") != expected_drop
        or training.get("fixed_evaluation_loss_relative_drop") != expected_drop
    ):
        raise ValueError("mechanism loss gate is not bound to the fixed evaluation")
    criterion_passes = []
    for item in criteria.values():
        if not isinstance(item, Mapping) or not isinstance(item.get("passed"), bool):
            raise ValueError("mechanism probe criterion is malformed")
        criterion_passes.append(bool(item["passed"]))
    overall = gate.get("overall_pass")
    if not isinstance(overall, bool) or overall != all(criterion_passes):
        raise ValueError("mechanism probe overall decision is inconsistent")
    if gate.get("cycleback_mechanism_supported") is not overall:
        raise ValueError("mechanism support flag differs from gate decision")
    expected_status = "passed" if overall else "rejected"
    if payload.get("status") != expected_status:
        raise ValueError("mechanism probe status differs from gate decision")
    checkpoint = payload.get("mechanism_seed_checkpoint")
    expected_checkpoint_keys = {
        "artifact_type",
        "produced",
        "publication_status",
        "relative_path",
        "sha256",
        "bytes",
        "captured_boundary_model_state_sha256",
        "captured_boundary_optimizer_state_sha256",
        "captured_boundary_rng_state_sha256",
        "captured_boundary_backend_state_sha256",
        "captured_sampler_state_sha256",
        "captured_view_state_sha256",
        "captured_consumed_video_batch_chain_sha256",
        "captured_consumed_pair_row_chain_sha256",
        "trainer_contract_fingerprint",
        "representation_contract_sha256",
        "seed_predecessor_lineage_fingerprint",
        "boundary_state_captured_before_diagnostics",
        "diagnostics_restored_exact_boundary",
        "checkpoint_payload_validated_before_publication",
        "retroactive_checkpoint_reconstruction_allowed",
        "rejected_checkpoint_written",
        "epoch11_train337_continuation_authorized",
        "direct_epoch150_start_authorized",
    }
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != expected_checkpoint_keys:
        raise ValueError("mechanism seed checkpoint binding schema mismatch")
    lineage = payload.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError("mechanism probe lineage is missing")
    if (
        checkpoint.get("artifact_type")
        != "pams_conventional_cycleback_mechanism_seed_checkpoint_v1"
        or checkpoint.get("produced") is not overall
        or checkpoint.get("captured_boundary_model_state_sha256")
        != training.get("final_model_state_sha256")
        or checkpoint.get("captured_boundary_optimizer_state_sha256")
        != optimizer_state_sha256
        or checkpoint.get("captured_sampler_state_sha256")
        != sampler_state.fingerprint
        or checkpoint.get("captured_view_state_sha256")
        != training.get("mechanism_view_state_sha256")
        or checkpoint.get("captured_consumed_video_batch_chain_sha256")
        != sampler_state.consumed_video_batch_chain_sha256
        or checkpoint.get("captured_consumed_pair_row_chain_sha256")
        != sampler_state.consumed_pair_row_chain_sha256
        or checkpoint.get("trainer_contract_fingerprint")
        != trainer_contract.fingerprint
        or checkpoint.get("representation_contract_sha256")
        != lineage.get("representation_contract_sha256")
        or checkpoint.get("seed_predecessor_lineage_fingerprint")
        != lineage.get("mechanism_seed_predecessor_lineage_fingerprint")
        or checkpoint.get("boundary_state_captured_before_diagnostics") is not True
        or checkpoint.get("diagnostics_restored_exact_boundary") is not True
        or checkpoint.get("checkpoint_payload_validated_before_publication") is not True
        or checkpoint.get("retroactive_checkpoint_reconstruction_allowed") is not False
        or checkpoint.get("rejected_checkpoint_written") is not False
        or checkpoint.get("epoch11_train337_continuation_authorized") is not False
        or checkpoint.get("direct_epoch150_start_authorized") is not False
    ):
        raise ValueError("mechanism seed checkpoint semantics mismatch")
    for key in (
        "captured_boundary_model_state_sha256",
        "captured_boundary_optimizer_state_sha256",
        "captured_boundary_rng_state_sha256",
        "captured_boundary_backend_state_sha256",
        "captured_sampler_state_sha256",
        "captured_view_state_sha256",
        "captured_consumed_video_batch_chain_sha256",
        "captured_consumed_pair_row_chain_sha256",
        "trainer_contract_fingerprint",
        "representation_contract_sha256",
        "seed_predecessor_lineage_fingerprint",
    ):
        value = checkpoint.get(key)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"mechanism seed checkpoint digest is invalid: {key}")
    if overall and (
        checkpoint.get("publication_status") != "passed_checkpoint_published"
        or checkpoint.get("relative_path") != "learned-encoder-L.pt"
        or not isinstance(checkpoint.get("sha256"), str)
        or not _SHA256.fullmatch(checkpoint["sha256"])
        or not isinstance(checkpoint.get("bytes"), int)
        or isinstance(checkpoint.get("bytes"), bool)
        or checkpoint["bytes"] < 1
    ):
        raise ValueError("passed mechanism lacks an exact seed checkpoint")
    if not overall and (
        checkpoint.get("publication_status") != "rejected_not_published"
        or checkpoint.get("relative_path") is not None
        or checkpoint.get("sha256") is not None
        or checkpoint.get("bytes") is not None
    ):
        raise ValueError("rejected mechanism claims a seed checkpoint")
    if gate.get("epoch11_encoder_continuation_authorized") is not False:
        raise ValueError("mechanism probe must not authorize epoch11 continuation")
    for key in (
        "full_encoder_training_authorized",
        "development_evaluation_authorized",
        "sealed_evaluation_authorized",
    ):
        if gate.get(key) is not False:
            raise ValueError(f"mechanism probe illegally sets {key}")
    for key in (
        "config_sha256",
        "geometry_gate_sha256",
        "geometry_run_receipt_sha256",
        "pose_snapshot_sha256",
        "pose_fingerprint",
        "pose_cache_set_sha256",
        "pose_input_authorization_sha256",
        "pose_input_authority_run_receipt_sha256",
        "launch_authorization_sha256",
        "launch_registry_receipt_sha256",
        "source_export_manifest_sha256",
        "representation_integration_outcome_sha256",
        "representation_contract_sha256",
        "mechanism_seed_predecessor_lineage_fingerprint",
    ):
        value = lineage.get(key)
        if not isinstance(value, str) or not _SHA256.fullmatch(value):
            raise ValueError(f"mechanism probe lineage digest is invalid: {key}")
    for key in (
        "config_bytes",
        "launch_authorization_bytes",
        "representation_integration_outcome_bytes",
    ):
        value = lineage.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"mechanism probe lineage byte count is invalid: {key}")
    if (
        lineage.get("representation_integration_outcome_sha256")
        != lineage.get("launch_authorization_sha256")
        or lineage.get("representation_integration_outcome_bytes")
        != lineage.get("launch_authorization_bytes")
    ):
        raise ValueError("mechanism representation integration binding mismatch")
    if not isinstance(lineage.get("source_git_sha"), str) or not _GIT_SHA.fullmatch(
        lineage["source_git_sha"]
    ):
        raise ValueError("mechanism probe source lineage is invalid")
    if not isinstance(lineage.get("container_image_id"), str) or not _IMAGE_ID.fullmatch(
        lineage["container_image_id"]
    ):
        raise ValueError("mechanism probe image lineage is invalid")
    authority = payload.get("authority_boundaries")
    if not isinstance(authority, Mapping) or any(
        authority.get(key) is not False
        for key in (
            "historical_epoch11_gate_satisfies_this_contract",
            "full_training_authorized",
            "development_evaluation_authorized",
            "sealed_evaluation_authorized",
        )
    ):
        raise ValueError("mechanism probe authority boundary mismatch")


def _final_conditions(
    encoder: PAMSEncoder,
    objective: ConventionalCycleBackLoss,
    config: ConventionalCycleBackConfig,
    *,
    device: torch.device,
    encoder_batch_size: int,
    fixed_pairs: NativeWindowPairBatch,
    fixed_contexts: PairSegmentContexts,
    fixed_evaluation_video_total: int,
    fixed_view_seeds: tuple[int, int],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, Any]]:
    pairs = fixed_pairs
    if pairs.pair_count < 2:
        raise ValueError("final mechanism evaluation requires at least two pairs")
    conditions: dict[str, dict[str, Any]] = {}
    encoder.eval()
    with torch.inference_mode():
        real_a, real_b = encode_window_pairs(
            encoder,
            pairs,
            fixed_contexts,
            batch_size=encoder_batch_size,
        )
        conditions["real"] = _evaluate_embeddings(objective, real_a, real_b, pairs)

        shuffled_contexts = independently_permuted_pair_segment_contexts(
            fixed_contexts,
            pairs,
            seed=config.seed,
        )
        shuffled_a, shuffled_b = encode_window_pairs(
            encoder,
            pairs,
            shuffled_contexts,
            batch_size=encoder_batch_size,
        )
        conditions["within_video_pose_shuffle"] = _evaluate_embeddings(
            objective,
            shuffled_a,
            shuffled_b,
            pairs,
        )

        zero_a, zero_b = encode_window_pairs(
            encoder,
            pairs,
            zero_pair_segment_contexts(fixed_contexts),
            batch_size=encoder_batch_size,
        )
        conditions["zero_pose"] = _evaluate_embeddings(
            objective,
            zero_a,
            zero_b,
            pairs,
        )

        joint_null_contracts: dict[str, Any] = {}
        for name, null_contexts in joint_support_null_segment_contexts(
            fixed_contexts
        ).items():
            null_a, null_b = encode_window_pairs(
                encoder,
                pairs,
                null_contexts,
                batch_size=encoder_batch_size,
            )
            conditions[name] = _evaluate_embeddings(
                objective,
                null_a,
                null_b,
                pairs,
            )
            joint_null_contracts[name] = dict(null_contexts.transform_contract)

        permuted_contexts = independently_permuted_pair_segment_positions(
            fixed_contexts,
            pairs,
            seed=config.seed,
        )
        permuted_a, permuted_b = encode_window_pairs(
            encoder,
            pairs,
            permuted_contexts,
            batch_size=encoder_batch_size,
        )
        conditions["permuted_pe"] = _evaluate_embeddings(
            objective,
            permuted_a,
            permuted_b,
            pairs,
        )

        pe_off = build_encoder(config, position_encoding_mode="none").to(device).eval()
        pe_off.load_state_dict(encoder.state_dict(), strict=True)
        pe_off_a, pe_off_b = encode_window_pairs(
            pe_off,
            pairs,
            fixed_contexts,
            batch_size=encoder_batch_size,
        )
        conditions["pe_off"] = _evaluate_embeddings(
            objective,
            pe_off_a,
            pe_off_b,
            pairs,
        )
        different_video = _different_video_control(
            objective,
            pairs,
            real_a,
            real_b,
        )
    return conditions, different_video, {
        "evaluation_training_video_total": fixed_evaluation_video_total,
        "evaluation_pair_total": pairs.pair_count,
        "independent_view_seeds": list(fixed_view_seeds),
        "disjoint_window_policy": (
            "adjacent_disjoint_same_video_windows_offset_by_window_length"
        ),
        "encoder_context": dict(fixed_contexts.transform_contract),
        "independent_temporal_null": dict(
            shuffled_contexts.transform_contract
        ),
        "independent_pe_null": dict(permuted_contexts.transform_contract),
        "joint_support_nulls": joint_null_contracts,
    }


def _fixed_evaluation_pairs(
    sequences: Sequence[Unified2DSequence],
    config: ConventionalCycleBackConfig,
    *,
    device: torch.device,
    segment_ranges_by_video: Mapping[str, Sequence[tuple[int, int]]],
    pair_eligibility: PairEligibility,
) -> tuple[NativeWindowPairBatch, PairSegmentContexts, int, tuple[int, int]]:
    evaluation = ranked_sequences(
        sequences,
        seed=config.seed,
        purpose="mechanism-fixed-evaluation",
    )[: config.mechanism_probe.evaluation_sample_videos]
    batch = collate_to_device(evaluation, device)
    pairs, sequence_views = pairs_from_batch(
        batch,
        config,
        step=1_000_000,
        segment_ranges_by_video=segment_ranges_by_video,
        pair_eligibility=pair_eligibility,
    )
    pairs = cap_window_pairs(
        pairs,
        maximum_pairs=config.mechanism_probe.evaluation_maximum_pairs,
        seed=config.seed,
        purpose="mechanism-final-pairs",
    )
    if pairs.pair_count < 2:
        raise ValueError("fixed mechanism evaluation requires at least two pairs")
    contexts = pair_segment_contexts(pairs, sequence_views)
    return pairs, contexts, len(evaluation), sequence_views.view_seeds


def _tensor_digest(digest: Any, value: Tensor) -> None:
    materialized = value.detach().cpu().contiguous()
    digest.update(str(materialized.dtype).encode())
    digest.update(json.dumps(list(materialized.shape)).encode())
    digest.update(materialized.numpy().tobytes())


def _fixed_pair_fingerprints(
    pairs: NativeWindowPairBatch,
    contexts: PairSegmentContexts,
) -> tuple[str, str]:
    identity = hashlib.sha256()
    identity.update(
        json.dumps(
            {
                "video_ids": list(pairs.video_ids),
                "starts_a": pairs.starts_a.detach().cpu().tolist(),
                "starts_b": pairs.starts_b.detach().cpu().tolist(),
                "segment_starts": pairs.segment_starts.detach().cpu().tolist(),
                "segment_ends": pairs.segment_ends.detach().cpu().tolist(),
                "native_lengths": pairs.native_lengths.detach().cpu().tolist(),
                "segment_context_lengths": [
                    value.shape[0] for value in contexts.poses_a
                ],
                "context_keys_a": list(contexts.context_keys_a),
                "context_keys_b": list(contexts.context_keys_b),
                "view_seeds": list(contexts.view_seeds),
                "context_contract": dict(contexts.transform_contract),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    )
    payload = hashlib.sha256(identity.digest())
    for tensor in (
        pairs.poses_a,
        pairs.poses_b,
        pairs.valid_a,
        pairs.valid_b,
        pairs.source_indices_a,
        pairs.source_indices_b,
    ):
        _tensor_digest(payload, tensor)
    for collection in (
        contexts.poses_a,
        contexts.poses_b,
        contexts.joint_valid_a,
        contexts.joint_valid_b,
        contexts.position_indices_a,
        contexts.position_indices_b,
    ):
        for tensor in collection:
            _tensor_digest(payload, tensor)
    return identity.hexdigest(), payload.hexdigest()


def _fixed_real_evaluation(
    encoder: PAMSEncoder,
    objective: ConventionalCycleBackLoss,
    pairs: NativeWindowPairBatch,
    contexts: PairSegmentContexts,
    *,
    encoder_batch_size: int,
) -> dict[str, Any]:
    was_training = encoder.training
    encoder.eval()
    with torch.inference_mode():
        embeddings_a, embeddings_b = encode_window_pairs(
            encoder,
            pairs,
            contexts,
            batch_size=encoder_batch_size,
        )
        metrics = _evaluate_embeddings(objective, embeddings_a, embeddings_b, pairs)
    encoder.train(was_training)
    identity_sha, payload_sha = _fixed_pair_fingerprints(pairs, contexts)
    return {
        "pair_identity_sha256": identity_sha,
        "pair_payload_sha256": payload_sha,
        "pair_total": pairs.pair_count,
        "unique_view_context_total": contexts.transform_contract[
            "unique_view_context_total"
        ],
        "reused_pair_side_context_references": contexts.transform_contract[
            "reused_pair_side_context_references"
        ],
        "valid_anchor_total": metrics["valid_anchor_total"],
        "possible_anchor_total": metrics["possible_anchor_total"],
        "symmetric_variance_aware_loss": metrics["symmetric_variance_aware_loss"],
        "symmetric_position_mse": metrics["symmetric_position_mse"],
        "model_state_sha256": model_state_sha256(encoder),
    }


def run_mechanism_probe(
    config_path: str | Path,
    geometry_gate_path: str | Path,
    expected_geometry_gate_sha256: str,
    geometry_run_receipt_path: str | Path,
    expected_geometry_run_receipt_sha256: str,
    declared_geometry_host_root: str | Path,
    pose_authority_root: str | Path,
    *,
    declared_pose_authority_host_root: str | Path,
    launch_registry_root: str | Path,
    declared_launch_registry_host_root: str | Path,
    source_root: str | Path,
    expected_pose_authorization_sha256: str,
    expected_pose_authority_run_receipt_sha256: str,
    source_git_sha: str,
    source_export_manifest_sha256: str,
    container_image_id: str,
    mechanism_output_path: str | Path,
    seed_checkpoint_output: str | Path,
    device: str | torch.device | None = None,
    encoder_batch_size: int | None = None,
) -> dict[str, Any]:
    """Execute exactly 256 label-free optimizer steps and evaluate controls."""

    if not _GIT_SHA.fullmatch(source_git_sha):
        raise ValueError("source_git_sha must be a full lowercase Git SHA")
    if not _SHA256.fullmatch(source_export_manifest_sha256):
        raise ValueError("source export manifest must be a lowercase SHA-256")
    if not _IMAGE_ID.fullmatch(container_image_id):
        raise ValueError("container_image_id must be an immutable image ID")
    output_path = Path(mechanism_output_path)
    seed_checkpoint_path = Path(seed_checkpoint_output)
    bundle_root = output_path.parent
    bundle_parent = bundle_root.parent
    if bundle_root.is_symlink() or bundle_root.exists():
        raise ValueError("mechanism probe bundle output must be absent")
    if (
        output_path.name != "mechanism-probe.json"
        or seed_checkpoint_path.name != "learned-encoder-L.pt"
        or bundle_root.name != "mechanism-bundle"
        or bundle_root != seed_checkpoint_path.parent
        or bundle_parent.is_symlink()
        or not bundle_parent.is_dir()
    ):
        raise ValueError("mechanism seed checkpoint output locator is not canonical")
    launch = validate_cycleback_launch_registry(
        launch_registry_root,
        declared_host_root=declared_launch_registry_host_root,
        source_root=source_root,
    )
    if source_git_sha != launch.source_revision:
        raise ValueError("source_git_sha differs from canonical launch registry")
    if container_image_id != launch.container_image_id:
        raise ValueError("container_image_id differs from canonical launch registry")
    launch_authorization_identity = stable_file_identity(launch.authorization_path)
    if launch_authorization_identity[0] != launch.authorization_sha256:
        raise RuntimeError("launch authorization changed after validation")
    config_bytes_payload, config_identity = stable_file_bytes(config_path)
    config_source = Path(config_path).resolve(strict=True)
    frozen_source = Path(source_root).resolve(strict=True)
    try:
        config_relative = config_source.relative_to(frozen_source).as_posix()
    except ValueError as exc:
        raise ValueError("cycle-back config lies outside the frozen source export") from exc
    config_registry = launch.configs.get(config_relative)
    if (
        not isinstance(config_registry, Mapping)
        or config_registry.get("sha256") != config_identity[0]
        or config_registry.get("bytes") != config_identity[1]
    ):
        raise ValueError("cycle-back config differs from canonical launch registry")
    config = parse_conventional_cycleback_config(config_bytes_payload)
    trainer_contract = FullContextTrainerContract.from_candidate_config(config)
    representation_contract = unified2d_fullcontext_representation_contract()
    frozen_batch_size = config.mechanism_probe.encoder_segment_context_batch_size
    if encoder_batch_size is not None and encoder_batch_size != frozen_batch_size:
        raise ValueError("encoder batch size differs from the fingerprinted config")
    authority = validate_cycleback_pose_authority(
        pose_authority_root,
        declared_host_root=declared_pose_authority_host_root,
        launch_authority=launch,
        expected_authorization_sha256=expected_pose_authorization_sha256,
        expected_run_receipt_sha256=expected_pose_authority_run_receipt_sha256,
    )
    pose_authorization_identity = stable_file_identity(authority.authorization_path)
    if pose_authorization_identity[0] != authority.authorization_sha256:
        raise RuntimeError("pose authorization changed after validation")
    _, geometry_receipt_identity = _load_geometry_run_receipt(
        geometry_run_receipt_path,
        expected_sha256=expected_geometry_run_receipt_sha256,
        declared_host_root=declared_geometry_host_root,
        expected_gate_sha256=expected_geometry_gate_sha256,
        config=config,
        config_sha256=config_identity[0],
        config_bytes=config_identity[1],
        pose_authorization_sha256=authority.authorization_sha256,
        pose_run_receipt_sha256=authority.run_receipt_sha256,
        launch_registry_id=launch.registry_id,
        launch_authorization_sha256=launch.authorization_sha256,
        launch_registry_receipt_sha256=launch.receipt_sha256,
        source_git_sha=source_git_sha,
        source_tree_sha=launch.source_tree_sha,
        container_image_id=container_image_id,
    )
    snapshot_identity = stable_file_identity(authority.snapshot_path)
    segment_index_identity = stable_file_identity(authority.segment_index_path)
    joint_snapshot_identity = stable_file_identity(authority.joint_mask_snapshot_path)
    identity_map_identity = stable_file_identity(authority.identity_map_path)
    pair_eligibility_identity = stable_file_identity(authority.pair_eligibility_path)
    snapshot = authority.snapshot
    joint_snapshot = load_joint_mask_snapshot(authority.joint_mask_snapshot_path)
    if joint_snapshot_identity[0] != authority.joint_mask_snapshot_sha256 or (
        joint_snapshot.fingerprint != authority.joint_mask_set_sha256
    ):
        raise ValueError("joint-mask snapshot differs from pose authority")
    identity_map, loaded_identity_map_identity = load_identity_map(
        authority.identity_map_path,
        expected_video_ids=[entry.video_id for entry in snapshot.entries],
        expected_sha256=authority.identity_map_sha256,
    )
    if loaded_identity_map_identity != identity_map_identity:
        raise RuntimeError("pose identity map changed while being loaded")
    pair_eligibility = load_pair_eligibility(
        authority.pair_eligibility_path,
        expected_sha256=authority.pair_eligibility_sha256,
        identity_map=identity_map,
    )
    if pair_eligibility.identity != pair_eligibility_identity:
        raise RuntimeError("pair-eligibility artifact changed while being loaded")
    segment_ranges_by_video, loaded_segment_identity = load_segment_index(
        authority.segment_index_path,
        expected_video_ids=[entry.video_id for entry in snapshot.entries],
        expected_sha256=authority.segment_index_sha256,
        expected_segment_reset_policy_sha256=(
            authority.segment_reset_policy_sha256
        ),
        identity_map=identity_map,
        pair_eligibility=pair_eligibility,
    )
    if loaded_segment_identity != segment_index_identity:
        raise RuntimeError("pose segment index changed while being loaded")
    _, geometry_identity = _load_geometry_authorization(
        geometry_gate_path,
        expected_sha256=expected_geometry_gate_sha256,
        config=config,
        config_sha256=config_identity[0],
        config_bytes=config_identity[1],
        snapshot_sha256=snapshot_identity[0],
        snapshot_bytes=snapshot_identity[1],
        pose_cache_set_sha256=snapshot.fingerprint,
        segment_index_sha256=authority.segment_index_sha256,
        segment_reset_policy_sha256=authority.segment_reset_policy_sha256,
        joint_mask_snapshot_sha256=authority.joint_mask_snapshot_sha256,
        joint_mask_set_sha256=authority.joint_mask_set_sha256,
        identity_map_sha256=authority.identity_map_sha256,
        cycleback_pair_eligibility_sha256=authority.pair_eligibility_sha256,
        pair_eligibility=pair_eligibility,
        pose_input_authorization_sha256=authority.authorization_sha256,
        pose_input_authority_run_receipt_sha256=authority.run_receipt_sha256,
        representation_bindings=authority.representation_bindings,
        launch_registry_id=launch.registry_id,
        launch_authorization_sha256=launch.authorization_sha256,
        launch_registry_receipt_sha256=launch.receipt_sha256,
        source_git_sha=source_git_sha,
        source_tree_sha=launch.source_tree_sha,
        container_image_id=container_image_id,
    )
    sequences = load_snapshot_sequences(
        authority.cache_dir,
        snapshot,
        authority.joint_mask_dir,
        joint_snapshot,
        expected_video_total=config.geometry_audit.expected_training_video_total,
    )
    if any(
        pair_eligibility.native_lengths[sequence.video_id] != sequence.num_frames
        for sequence in sequences
    ):
        raise ValueError("pair-eligibility native length differs from pose snapshot")
    eligible = ranked_sequences(
        _eligible_sequences(
            sequences,
            config,
            segment_ranges_by_video=segment_ranges_by_video,
            pair_eligibility=pair_eligibility,
        ),
        seed=config.seed,
        purpose="mechanism-training-order",
    )
    resolved_device = _device(device)
    configure_fullcontext_determinism()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    encoder = build_encoder(config).to(resolved_device).train()
    initial_state_sha256 = model_state_sha256(encoder)
    optimizer = build_fullcontext_adamw(encoder, trainer_contract)
    objective = build_fullcontext_objective(config, trainer_contract)
    (
        fixed_pairs,
        fixed_contexts,
        fixed_evaluation_video_total,
        fixed_view_seeds,
    ) = _fixed_evaluation_pairs(
        eligible,
        config,
        device=resolved_device,
        segment_ranges_by_video=segment_ranges_by_video,
        pair_eligibility=pair_eligibility,
    )
    with preserve_fullcontext_diagnostic_state(encoder):
        fixed_step0 = _fixed_real_evaluation(
            encoder,
            objective,
            fixed_pairs,
            fixed_contexts,
            encoder_batch_size=frozen_batch_size,
        )
    if fixed_step0["model_state_sha256"] != initial_state_sha256:
        raise RuntimeError("step-0 fixed evaluation changed encoder state")
    loss_history: list[float] = []
    anchor_history: list[float] = []
    pair_history: list[int] = []
    unique_context_history: list[int] = []
    reused_context_reference_history: list[int] = []
    video_batch_prefix = hashlib.sha256()
    pair_row_prefix = hashlib.sha256()
    for step in range(config.mechanism_probe.optimizer_steps):
        selected = _cyclic_video_batch(
            eligible,
            step=step,
            batch_size=config.mechanism_probe.video_batch_size,
        )
        batch = collate_to_device(selected, resolved_device)
        pairs, sequence_views = pairs_from_batch(
            batch,
            config,
            step=step + 1,
            segment_ranges_by_video=segment_ranges_by_video,
            pair_eligibility=pair_eligibility,
        )
        pairs = cap_window_pairs(
            pairs,
            maximum_pairs=config.mechanism_probe.maximum_pairs_per_step,
            seed=config.seed + step,
            purpose="mechanism-optimizer-pairs",
        )
        if pairs.pair_count < 2:
            raise RuntimeError("optimizer step has fewer than two eligible window pairs")
        _update_mechanism_sampler_prefix(
            video_batch_prefix,
            pair_row_prefix,
            step=step + 1,
            selected=selected,
            pairs=pairs,
        )
        contexts = pair_segment_contexts(pairs, sequence_views)
        unique_context_history.append(
            int(contexts.transform_contract["unique_view_context_total"])
        )
        reused_context_reference_history.append(
            int(
                contexts.transform_contract[
                    "reused_pair_side_context_references"
                ]
            )
        )
        require_real_optimizer_contexts(contexts)
        optimizer.zero_grad(set_to_none=True)
        embeddings_a, embeddings_b = encode_window_pairs(
            encoder,
            pairs,
            contexts,
            batch_size=frozen_batch_size,
        )
        output = objective.compute(
            embeddings_a,
            embeddings_b,
            pairs.valid_a,
            pairs.valid_b,
            pairs.source_indices_a,
            pairs.source_indices_b,
            pairs.native_lengths,
            video_ids_a=pairs.video_ids,
            video_ids_b=pairs.video_ids,
        )
        if output.valid_anchor_count < 1 or not torch.isfinite(output.total):
            raise RuntimeError("cycle-back optimizer step has no finite valid objective")
        output.total.backward()
        if not _finite_parameter_gradients(encoder):
            raise RuntimeError("cycle-back encoder gradients are missing or non-finite")
        optimizer.step()
        loss_history.append(float(output.total.detach()))
        possible = int(pairs.valid_a.sum() + pairs.valid_b.sum())
        if possible < 1:
            raise RuntimeError("optimizer step has no possible valid anchors")
        anchor_history.append(output.valid_anchor_count / possible)
        pair_history.append(pairs.pair_count)

    completed_steps = len(loss_history)
    if completed_steps != 256:
        raise RuntimeError("mechanism probe did not complete exactly 256 optimizer steps")
    final_state_sha256 = model_state_sha256(encoder)
    if final_state_sha256 == initial_state_sha256:
        raise RuntimeError("encoder state did not change across optimizer steps")
    sampler_state = MechanismSeedSamplerState(
        eligible_video_total=len(eligible),
        ordered_eligible_video_ids_sha256=hashlib.sha256(
            _canonical_json_bytes([sequence.video_id for sequence in eligible])
        ).hexdigest(),
        video_batch_size=config.mechanism_probe.video_batch_size,
        maximum_pairs_per_step=config.mechanism_probe.maximum_pairs_per_step,
        completed_optimizer_steps=completed_steps,
        consumed_video_batch_chain_sha256=video_batch_prefix.hexdigest(),
        consumed_pair_row_chain_sha256=pair_row_prefix.hexdigest(),
        next_cyclic_start_index=(
            completed_steps * config.mechanism_probe.video_batch_size
        )
        % len(eligible),
    )
    seed_predecessor = MechanismSeedPredecessorLineage(
        source_git_sha=source_git_sha,
        source_tree_sha256=source_export_manifest_sha256,
        container_image_id=container_image_id,
        config_sha256=config_identity[0],
        config_bytes=config_identity[1],
        representation_integration_outcome_sha256=(
            launch_authorization_identity[0]
        ),
        representation_integration_outcome_bytes=(
            launch_authorization_identity[1]
        ),
        representation_contract_sha256=representation_contract.fingerprint,
        representation_authorization_sha256=pose_authorization_identity[0],
        representation_authorization_bytes=pose_authorization_identity[1],
        mechanism_learned_model_state_sha256=final_state_sha256,
    )
    seed_checkpoint_payload = build_mechanism_seed_checkpoint_payload(
        encoder,
        optimizer,
        trainer_contract,
        seed_predecessor,
        sampler_state,
    )
    boundary_optimizer_sha256 = seed_checkpoint_payload["optimizer_state_sha256"]
    boundary_rng_sha256 = seed_checkpoint_payload["rng_state_sha256"]
    boundary_backend_sha256 = seed_checkpoint_payload["backend_state_sha256"]
    first_median = _median(loss_history[:16])
    final_median = _median(loss_history[-16:])
    training_history_relative_drop = (
        None if first_median <= 0.0 else 1.0 - final_median / first_median
    )
    with preserve_fullcontext_diagnostic_state(encoder) as diagnostic_boundary:
        fixed_step256 = _fixed_real_evaluation(
            encoder,
            objective,
            fixed_pairs,
            fixed_contexts,
            encoder_batch_size=frozen_batch_size,
        )
        if fixed_step256["model_state_sha256"] != final_state_sha256:
            raise RuntimeError("step-256 fixed evaluation changed encoder state")
        conditions, different_video, evaluation_scope = _final_conditions(
            encoder,
            objective,
            config,
            device=resolved_device,
            encoder_batch_size=frozen_batch_size,
            fixed_pairs=fixed_pairs,
            fixed_contexts=fixed_contexts,
            fixed_evaluation_video_total=fixed_evaluation_video_total,
            fixed_view_seeds=fixed_view_seeds,
        )
    if (
        diagnostic_boundary["model_state_sha256"] != final_state_sha256
        or diagnostic_boundary["rng_state_sha256"] != boundary_rng_sha256
        or diagnostic_boundary["backend_state_sha256"] != boundary_backend_sha256
        or model_state_sha256(encoder) != final_state_sha256
        or optimizer_state_sha256(optimizer) != boundary_optimizer_sha256
        or fullcontext_rng_state_sha256(capture_fullcontext_rng_state())
        != boundary_rng_sha256
        or fullcontext_backend_state_sha256(capture_fullcontext_backend_state())
        != boundary_backend_sha256
    ):
        raise RuntimeError("mechanism diagnostics changed the exact step-256 boundary")
    for key in (
        "pair_identity_sha256",
        "pair_payload_sha256",
        "pair_total",
        "unique_view_context_total",
        "reused_pair_side_context_references",
    ):
        if fixed_step0[key] != fixed_step256[key]:
            raise RuntimeError(f"fixed evaluation changed between step 0 and 256: {key}")
    fixed_initial_loss = float(fixed_step0["symmetric_variance_aware_loss"])
    fixed_final_loss = float(fixed_step256["symmetric_variance_aware_loss"])
    loss_relative_drop = (
        None
        if fixed_initial_loss <= 0.0
        else 1.0 - fixed_final_loss / fixed_initial_loss
    )

    real_error = float(conditions["real"]["symmetric_position_mse"])
    null_error = min(
        float(conditions["zero_pose"]["symmetric_position_mse"]),
        float(conditions["within_video_pose_shuffle"]["symmetric_position_mse"]),
        float(conditions["mask_flicker"]["symmetric_position_mse"]),
        float(conditions["torso_only"]["symmetric_position_mse"]),
        float(conditions["alternating_limb_dropout"]["symmetric_position_mse"]),
    )
    rms_median_raw = conditions["real"]["embedding_temporal_rms"]["median"]
    final_rms_median = None if rms_median_raw is None else float(rms_median_raw)
    real_null_gap = null_error - real_error
    different_video_gap = float(different_video["different_minus_same_position_mse"])
    final_anchor_fraction = float(conditions["real"]["valid_anchor_fraction"])
    permuted_pe_ratio = _finite_positive_ratio(
        float(conditions["permuted_pe"]["symmetric_position_mse"]),
        real_error,
    )
    pe_off_ratio = _finite_positive_ratio(
        float(conditions["pe_off"]["symmetric_position_mse"]),
        real_error,
    )
    mask_flicker_ratio = _finite_positive_ratio(
        float(conditions["mask_flicker"]["symmetric_position_mse"]),
        real_error,
    )
    torso_only_ratio = _finite_positive_ratio(
        float(conditions["torso_only"]["symmetric_position_mse"]),
        real_error,
    )
    alternating_limb_dropout_ratio = _finite_positive_ratio(
        float(conditions["alternating_limb_dropout"]["symmetric_position_mse"]),
        real_error,
    )
    decision = mechanism_probe_gate_decision(
        optimizer_steps=completed_steps,
        loss_relative_drop=loss_relative_drop,
        final_temporal_rms_median=final_rms_median,
        real_null_position_error_gap=real_null_gap,
        final_valid_anchor_fraction=final_anchor_fraction,
        final_mask_flicker_to_real_position_error_ratio=mask_flicker_ratio,
        final_torso_only_to_real_position_error_ratio=torso_only_ratio,
        final_alternating_limb_dropout_to_real_position_error_ratio=(
            alternating_limb_dropout_ratio
        ),
        final_permuted_pe_to_real_position_error_ratio=permuted_pe_ratio,
        final_pe_off_to_real_position_error_ratio=pe_off_ratio,
        thresholds=config.mechanism_probe.thresholds,
    )
    if stable_file_identity(config_path) != config_identity:
        raise RuntimeError("cycle-back config changed before checkpoint publication")
    if stable_file_identity(launch.authorization_path) != launch_authorization_identity:
        raise RuntimeError("launch authorization changed before checkpoint publication")
    if stable_file_identity(authority.authorization_path) != pose_authorization_identity:
        raise RuntimeError("pose authorization changed before checkpoint publication")
    if stable_file_identity(authority.snapshot_path) != snapshot_identity:
        raise RuntimeError("pose snapshot changed before checkpoint publication")
    if stable_file_identity(authority.segment_index_path) != segment_index_identity:
        raise RuntimeError("pose segment index changed before checkpoint publication")
    if stable_file_identity(authority.joint_mask_snapshot_path) != joint_snapshot_identity:
        raise RuntimeError("joint-mask snapshot changed before checkpoint publication")
    if stable_file_identity(authority.identity_map_path) != identity_map_identity:
        raise RuntimeError("pose identity map changed before checkpoint publication")
    if (
        stable_file_identity(authority.pair_eligibility_path)
        != pair_eligibility_identity
    ):
        raise RuntimeError("pair eligibility changed before checkpoint publication")
    if stable_file_identity(geometry_gate_path) != geometry_identity:
        raise RuntimeError("geometry gate changed before checkpoint publication")
    if stable_file_identity(geometry_run_receipt_path) != geometry_receipt_identity:
        raise RuntimeError("geometry receipt changed before checkpoint publication")
    checkpoint_identity: tuple[str, int] | None = None
    checkpoint_bytes: bytes | None = None
    if decision["overall_pass"]:
        checkpoint_buffer = io.BytesIO()
        torch.save(seed_checkpoint_payload, checkpoint_buffer)
        checkpoint_bytes = checkpoint_buffer.getvalue()
        reloaded_seed = torch.load(
            io.BytesIO(checkpoint_bytes),
            map_location=torch.device("cpu"),
            weights_only=False,
        )
        if not isinstance(reloaded_seed, Mapping):
            raise RuntimeError("serialized mechanism seed is not a mapping")
        validate_mechanism_seed_checkpoint_payload(
            reloaded_seed,
            trainer_contract,
            seed_predecessor,
        )
        checkpoint_identity = (
            hashlib.sha256(checkpoint_bytes).hexdigest(),
            len(checkpoint_bytes),
        )
    checkpoint_record = {
        "artifact_type": (
            "pams_conventional_cycleback_mechanism_seed_checkpoint_v1"
        ),
        "produced": checkpoint_identity is not None,
        "publication_status": (
            "passed_checkpoint_published"
            if checkpoint_identity is not None
            else "rejected_not_published"
        ),
        "relative_path": (
            seed_checkpoint_path.name if checkpoint_identity is not None else None
        ),
        "sha256": None if checkpoint_identity is None else checkpoint_identity[0],
        "bytes": None if checkpoint_identity is None else checkpoint_identity[1],
        "captured_boundary_model_state_sha256": final_state_sha256,
        "captured_boundary_optimizer_state_sha256": boundary_optimizer_sha256,
        "captured_boundary_rng_state_sha256": boundary_rng_sha256,
        "captured_boundary_backend_state_sha256": boundary_backend_sha256,
        "captured_sampler_state_sha256": sampler_state.fingerprint,
        "captured_view_state_sha256": seed_checkpoint_payload[
            "next_view_seed_state_sha256"
        ],
        "captured_consumed_video_batch_chain_sha256": (
            sampler_state.consumed_video_batch_chain_sha256
        ),
        "captured_consumed_pair_row_chain_sha256": (
            sampler_state.consumed_pair_row_chain_sha256
        ),
        "trainer_contract_fingerprint": trainer_contract.fingerprint,
        "representation_contract_sha256": representation_contract.fingerprint,
        "seed_predecessor_lineage_fingerprint": seed_predecessor.fingerprint,
        "boundary_state_captured_before_diagnostics": True,
        "diagnostics_restored_exact_boundary": True,
        "checkpoint_payload_validated_before_publication": True,
        "retroactive_checkpoint_reconstruction_allowed": False,
        "rejected_checkpoint_written": False,
        "epoch11_train337_continuation_authorized": False,
        "direct_epoch150_start_authorized": False,
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": config.mechanism_probe.artifact_type,
        "status": "passed" if decision["overall_pass"] else "rejected",
        "namespace": config.namespace,
        "protocol": config.protocol,
        "classification": config.classification,
        "paper_reference_status": config.paper_reference_status,
        "paper_table_claim_eligible": False,
        "candidate_id": config.candidate_id,
        "config_fingerprint": config.fingerprint,
        "mechanism_seed_checkpoint": checkpoint_record,
        "label_firewall": {
            "accepted_inputs": [
                "cycleback_proxy_config",
                "passed_cycleback_geometry_gate",
                "sealed_cycleback_pose_input_authority",
            ],
            "manifest_interface_supported": False,
            "media_interface_supported": False,
            "action_or_repetition_annotation_interface_supported": False,
            "development_pose_or_annotation_mounted": False,
            "sealed_pose_or_annotation_mounted": False,
            "external_label_fields_accessed": [],
            "video_id_handling": (
                "opaque_identifier_for_hash_order_and_same_video_equality_only"
            ),
            "video_id_tokens_parsed": False,
        },
        "lineage": {
            "config_sha256": config_identity[0],
            "config_bytes": config_identity[1],
            "geometry_gate_sha256": geometry_identity[0],
            "geometry_gate_bytes": geometry_identity[1],
            "geometry_run_receipt_sha256": geometry_receipt_identity[0],
            "geometry_run_receipt_bytes": geometry_receipt_identity[1],
            "pose_snapshot_sha256": snapshot_identity[0],
            "pose_snapshot_bytes": snapshot_identity[1],
            "pose_fingerprint": snapshot.pose_fingerprint,
            "pose_cache_set_sha256": snapshot.fingerprint,
            "pose_input_authorization_sha256": authority.authorization_sha256,
            "pose_input_authority_run_receipt_sha256": authority.run_receipt_sha256,
            "segment_index_sha256": authority.segment_index_sha256,
            "segment_reset_policy_sha256": authority.segment_reset_policy_sha256,
            "joint_mask_snapshot_sha256": authority.joint_mask_snapshot_sha256,
            "joint_mask_set_sha256": authority.joint_mask_set_sha256,
            "identity_map_sha256": authority.identity_map_sha256,
            "cycleback_pair_eligibility_sha256": (
                authority.pair_eligibility_sha256
            ),
            "pair_start_grid_policy": pair_eligibility.start_grid_policy,
            "frozen_joint_support_thresholds": {
                "minimum_window_stable_action_joints": (
                    pair_eligibility.minimum_window_stable_action_joints
                ),
                "minimum_window_joint_support_fraction": (
                    pair_eligibility.minimum_window_joint_support_fraction
                ),
            },
            "representation_bindings": dict(authority.representation_bindings),
            "launch_registry_id": launch.registry_id,
            "launch_authorization_sha256": launch.authorization_sha256,
            "launch_authorization_bytes": launch_authorization_identity[1],
            "launch_registry_receipt_sha256": launch.receipt_sha256,
            "source_export_manifest_sha256": source_export_manifest_sha256,
            "representation_integration_outcome_sha256": (
                launch_authorization_identity[0]
            ),
            "representation_integration_outcome_bytes": (
                launch_authorization_identity[1]
            ),
            "representation_contract_sha256": representation_contract.fingerprint,
            "mechanism_seed_predecessor_lineage_fingerprint": (
                seed_predecessor.fingerprint
            ),
            "training_video_total": len(sequences),
            "geometry_eligible_training_video_total": len(eligible),
            "source_git_sha": source_git_sha,
            "container_image_id": container_image_id,
        },
        "training": {
            "objective": config.objective.objective,
            "directions": config.objective.directions,
            "same_video_window_pairs_only": True,
            "pair_policy": config.window_pair.pair_policy,
            "strictly_disjoint_source_index_sets": True,
            "full_authorized_stable_range_encoded_before_window_slice": True,
            "attention_never_crosses_authorized_stable_range": True,
            "training_inference_context_parity": True,
            "window_only_encoder_path_used": False,
            "encoder_context_policy": config.window_pair.encoder_context_policy,
            "attention_boundary_policy": (
                config.window_pair.attention_boundary_policy
            ),
            "pair_starts_are_exact_representation_authorized_subset": True,
            "consumer_may_expand_pair_starts": False,
            "start_grid_policy": pair_eligibility.start_grid_policy,
            "all_window_frames_valid": True,
            "no_pair_crosses_segment_reset": True,
            "target_position_normalization": (
                config.objective.target_position_normalization
            ),
            "reset_count_aggregation_policy": (
                config.window_pair.reset_count_aggregation_policy
            ),
            "cross_video_positive_terms": False,
            "cross_video_negative_terms": False,
            "prototype_or_cluster_bank": False,
            "independent_skeleton_views_per_step": True,
            "optimizer": "AdamW",
            "optimizer_contract": trainer_contract.to_dict()["optimizer"],
            "optimizer_state_sha256_at_step256": boundary_optimizer_sha256,
            "mechanism_sampler_state": sampler_state.to_dict(),
            "mechanism_sampler_state_sha256": sampler_state.fingerprint,
            "mechanism_view_state_sha256": seed_checkpoint_payload[
                "next_view_seed_state_sha256"
            ],
            "mechanism_consumed_video_batch_chain_sha256": (
                sampler_state.consumed_video_batch_chain_sha256
            ),
            "mechanism_consumed_pair_row_chain_sha256": (
                sampler_state.consumed_pair_row_chain_sha256
            ),
            "scheduler": "none",
            "trainer_contract_fingerprint": trainer_contract.fingerprint,
            "optimizer_steps_completed": completed_steps,
            "encoder_segment_context_batch_size": frozen_batch_size,
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
            "initial_model_state_sha256": initial_state_sha256,
            "final_model_state_sha256": final_state_sha256,
            "final_model_state_role": "L_learned_cycleback_encoder",
            "readout_encoder_state_requirement": (
                "L_and_Epi_must_use_exact_final_state_config_source"
            ),
            "E0_untrained_state_may_substitute_for_L": False,
            "loss_first_16_median": first_median,
            "loss_final_16_median": final_median,
            "training_history_relative_drop_diagnostic_only": (
                training_history_relative_drop
            ),
            "gate_loss_source": "fixed_label_free_evaluation_pair_set_step0_vs_step256",
            "fixed_evaluation_loss_relative_drop": loss_relative_drop,
            "loss_minimum": min(loss_history),
            "loss_maximum": max(loss_history),
            "valid_anchor_fraction_minimum": min(anchor_history),
            "valid_anchor_fraction_median": _median(anchor_history),
            "window_pairs_per_step_minimum": min(pair_history),
            "window_pairs_per_step_maximum": max(pair_history),
            "unique_view_contexts_per_step_minimum": min(
                unique_context_history
            ),
            "unique_view_contexts_per_step_maximum": max(
                unique_context_history
            ),
            "reused_pair_side_context_references_per_step_minimum": min(
                reused_context_reference_history
            ),
            "reused_pair_side_context_references_per_step_maximum": max(
                reused_context_reference_history
            ),
            "same_view_same_range_encoded_once": True,
        },
        "fixed_evaluation": {
            "label_free": True,
            "same_native_pair_set_at_step0_and_step256": True,
            "same_augmented_view_bytes_at_step0_and_step256": True,
            "encoder_eval_mode": True,
            "optimizer_updates_during_evaluation": 0,
            "evaluation_training_video_total": fixed_evaluation_video_total,
            "independent_view_seeds": list(fixed_view_seeds),
            "step0": fixed_step0,
            "step256": fixed_step256,
            "loss_relative_drop_used_by_gate": loss_relative_drop,
        },
        "final_conditions": conditions,
        "pe_off_interpretation": {
            "scope": "trained_sinusoidal_candidate_with_position_input_disabled",
            "distribution_shift_warning": True,
            "separate_nope_candidate_claimed": False,
            "gate_role": "two_sided_shortcut_sensitivity_control_only",
            "ratio_interval_is_two_sided": True,
        },
        "different_video_negative_control": different_video,
        "evaluation_scope": evaluation_scope,
        "mechanism_metrics": {
            "final_temporal_rms_median": final_rms_median,
            "real_null_position_error_gap": real_null_gap,
            "different_video_position_error_gap_diagnostic_only": different_video_gap,
            "final_valid_anchor_fraction": final_anchor_fraction,
            "final_mask_flicker_to_real_position_error_ratio": mask_flicker_ratio,
            "final_torso_only_to_real_position_error_ratio": torso_only_ratio,
            "final_alternating_limb_dropout_to_real_position_error_ratio": (
                alternating_limb_dropout_ratio
            ),
            "final_permuted_pe_to_real_position_error_ratio": permuted_pe_ratio,
            "final_pe_off_to_real_position_error_ratio": pe_off_ratio,
        },
        "gate": decision,
        "authority_boundaries": {
            "historical_epoch11_gate_satisfies_this_contract": False,
            "full_training_authorized": False,
            "development_evaluation_authorized": False,
            "sealed_evaluation_authorized": False,
        },
    }
    validate_mechanism_probe_contract(payload, config)
    if stable_file_identity(config_path) != config_identity:
        raise RuntimeError("cycle-back config changed during mechanism probe")
    if stable_file_identity(launch.authorization_path) != launch_authorization_identity:
        raise RuntimeError("launch authorization changed during mechanism probe")
    if stable_file_identity(authority.authorization_path) != pose_authorization_identity:
        raise RuntimeError("pose authorization changed during mechanism probe")
    if stable_file_identity(authority.snapshot_path) != snapshot_identity:
        raise RuntimeError("pose snapshot changed during mechanism probe")
    if stable_file_identity(authority.segment_index_path) != segment_index_identity:
        raise RuntimeError("pose segment index changed during mechanism probe")
    if stable_file_identity(authority.joint_mask_snapshot_path) != joint_snapshot_identity:
        raise RuntimeError("joint-mask snapshot changed during mechanism probe")
    if stable_file_identity(authority.identity_map_path) != identity_map_identity:
        raise RuntimeError("pose identity map changed during mechanism probe")
    if (
        stable_file_identity(authority.pair_eligibility_path)
        != pair_eligibility_identity
    ):
        raise RuntimeError("pair-eligibility artifact changed during mechanism probe")
    if stable_file_identity(geometry_gate_path) != geometry_identity:
        raise RuntimeError("geometry-gate artifact changed during mechanism probe")
    if stable_file_identity(geometry_run_receipt_path) != geometry_receipt_identity:
        raise RuntimeError("geometry run receipt changed during mechanism probe")
    _publish_mechanism_bundle(
        output_path,
        seed_checkpoint_path,
        payload,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_identity=checkpoint_identity,
    )
    return payload


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(path, flags, 0o640)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    directory_flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    directory = os.open(path.parent, directory_flags)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _publish_mechanism_bundle(
    output_path: Path,
    seed_checkpoint_path: Path,
    payload: Mapping[str, Any],
    *,
    checkpoint_bytes: bytes | None,
    checkpoint_identity: tuple[str, int] | None,
) -> None:
    """Publish JSON and an optional PASS seed as one directory transaction."""

    bundle_root = output_path.parent
    parent = bundle_root.parent
    directory_flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    staging = Path(
        tempfile.mkdtemp(prefix=f".{bundle_root.name}.", suffix=".incomplete", dir=parent)
    )
    published = False
    try:
        staged_output = staging / output_path.name
        staged_checkpoint = staging / seed_checkpoint_path.name
        if checkpoint_bytes is None:
            if checkpoint_identity is not None:
                raise ValueError("absent mechanism seed has a file identity")
        else:
            if checkpoint_identity != (
                hashlib.sha256(checkpoint_bytes).hexdigest(),
                len(checkpoint_bytes),
            ):
                raise ValueError("mechanism seed bytes differ from prepared identity")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= int(getattr(os, "O_NOFOLLOW", 0))
            flags |= int(getattr(os, "O_CLOEXEC", 0))
            descriptor = os.open(staged_checkpoint, flags, 0o640)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    descriptor = -1
                    handle.write(checkpoint_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
        _write_new_json(staged_output, payload)
        if checkpoint_identity is not None and stable_file_identity(
            staged_checkpoint
        ) != checkpoint_identity:
            raise RuntimeError("staged mechanism seed changed before bundle publication")
        directory = os.open(staging, directory_flags)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if bundle_root.is_symlink() or bundle_root.exists():
            raise ValueError("mechanism bundle destination appeared before publication")
        os.rename(staging, bundle_root)
        published = True
        parent_descriptor = os.open(parent, directory_flags)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    except BaseException as error:
        cleanup = bundle_root if published else staging
        cleanup_failures: list[BaseException] = []
        try:
            if cleanup.exists():
                shutil.rmtree(cleanup)
        except BaseException as cleanup_error:
            cleanup_failures.append(cleanup_error)
        remaining_checkpoint = cleanup / seed_checkpoint_path.name
        try:
            if remaining_checkpoint.is_symlink():
                remaining_checkpoint.unlink()
            elif remaining_checkpoint.exists():
                remaining_checkpoint.chmod(0o600)
                remaining_checkpoint.unlink()
            if cleanup.exists():
                shutil.rmtree(cleanup)
        except BaseException as cleanup_error:
            cleanup_failures.append(cleanup_error)
        try:
            parent_descriptor = os.open(parent, directory_flags)
            try:
                os.fsync(parent_descriptor)
            finally:
                os.close(parent_descriptor)
        except BaseException as cleanup_error:
            cleanup_failures.append(cleanup_error)
        for cleanup_error in cleanup_failures:
            error.add_note(
                "mechanism bundle cleanup failure: "
                f"{type(cleanup_error).__name__}"
            )
        raise


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--launch-registry-root", type=Path, required=True)
    parser.add_argument("--declared-launch-registry-host-root", type=Path, required=True)
    parser.add_argument("--geometry-gate", type=Path, required=True)
    parser.add_argument("--expected-geometry-gate-sha256", required=True)
    parser.add_argument("--geometry-run-receipt", type=Path, required=True)
    parser.add_argument("--expected-geometry-run-receipt-sha256", required=True)
    parser.add_argument("--declared-geometry-host-root", type=Path, required=True)
    parser.add_argument("--pose-authority-root", type=Path, required=True)
    parser.add_argument("--declared-pose-authority-host-root", type=Path, required=True)
    parser.add_argument("--expected-pose-authorization-sha256", required=True)
    parser.add_argument("--expected-pose-authority-run-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-checkpoint-output", type=Path, required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--source-export-manifest-sha256", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encoder-batch-size", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    payload = run_mechanism_probe(
        arguments.config,
        arguments.geometry_gate,
        arguments.expected_geometry_gate_sha256,
        arguments.geometry_run_receipt,
        arguments.expected_geometry_run_receipt_sha256,
        arguments.declared_geometry_host_root,
        arguments.pose_authority_root,
        declared_pose_authority_host_root=arguments.declared_pose_authority_host_root,
        launch_registry_root=arguments.launch_registry_root,
        declared_launch_registry_host_root=(
            arguments.declared_launch_registry_host_root
        ),
        source_root=arguments.source_root,
        expected_pose_authorization_sha256=(
            arguments.expected_pose_authorization_sha256
        ),
        expected_pose_authority_run_receipt_sha256=(
            arguments.expected_pose_authority_run_receipt_sha256
        ),
        source_git_sha=arguments.source_git_sha,
        source_export_manifest_sha256=(
            arguments.source_export_manifest_sha256
        ),
        container_image_id=arguments.container_image_id,
        mechanism_output_path=arguments.output,
        seed_checkpoint_output=arguments.seed_checkpoint_output,
        device=arguments.device,
        encoder_batch_size=arguments.encoder_batch_size,
    )
    return 0 if payload["gate"]["overall_pass"] else 3
