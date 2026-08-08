"""Train337-only native-window geometry and PE/null audit for cycle-back TCC."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from pams.conventional_cycleback.authority import (
    validate_cycleback_launch_registry,
    validate_cycleback_pose_authority,
)
from pams.conventional_cycleback.config import (
    ConventionalCycleBackConfig,
    GeometryGateThresholds,
    parse_conventional_cycleback_config,
)
from pams.conventional_cycleback.loss import (
    ConventionalCycleBackLoss,
    CycleBackLossOutput,
)
from pams.conventional_cycleback.runtime import (
    PairEligibility,
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
    numeric_summary,
    pair_segment_contexts,
    pairs_from_batch,
    ranked_sequences,
    stable_file_bytes,
    stable_file_identity,
    temporal_rms_values,
    zero_pair_segment_contexts,
)
from pams.conventional_cycleback.windows import (
    NativeWindowPairBatch,
    enumerate_native_window_pairs,
    source_index_invariant_violations,
)

_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


class ScientificGeometryReject(RuntimeError):
    """Valid label-free input has insufficient geometry for null evaluation."""


def _device(value: str | torch.device | None) -> torch.device:
    if value is None or str(value).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(value)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return resolved


def _criterion(
    *,
    value: float | int | None,
    relation: str,
    threshold: float | int,
) -> dict[str, Any]:
    if relation == "at_least":
        passed = value is not None and math.isfinite(float(value)) and value >= threshold
    elif relation == "at_most":
        passed = value is not None and math.isfinite(float(value)) and value <= threshold
    else:
        raise ValueError(f"unsupported gate relation: {relation}")
    return {
        "value": value,
        "relation": relation,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _positive_ratio(numerator: float, denominator: float) -> float | None:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
        return None
    return numerator / denominator


def geometry_gate_decision(
    *,
    geometry: Mapping[str, Any],
    conditions: Mapping[str, Mapping[str, Any]],
    thresholds: GeometryGateThresholds,
) -> dict[str, Any]:
    """Apply only preregistered geometry and 0-step PE/null thresholds."""

    real_error = float(conditions["real"]["symmetric_position_mse"])
    ratios = {
        "zero_to_real_position_error": _positive_ratio(
            float(conditions["zero_pose"]["symmetric_position_mse"]),
            real_error,
        ),
        "shuffle_to_real_position_error": _positive_ratio(
            float(conditions["within_video_pose_shuffle"]["symmetric_position_mse"]),
            real_error,
        ),
        "mask_flicker_to_real_position_error": _positive_ratio(
            float(conditions["mask_flicker"]["symmetric_position_mse"]),
            real_error,
        ),
        "torso_only_to_real_position_error": _positive_ratio(
            float(conditions["torso_only"]["symmetric_position_mse"]),
            real_error,
        ),
        "alternating_limb_dropout_to_real_position_error": _positive_ratio(
            float(
                conditions["alternating_limb_dropout"]["symmetric_position_mse"]
            ),
            real_error,
        ),
        "permuted_pe_to_real_position_error": _positive_ratio(
            float(conditions["permuted_pe"]["symmetric_position_mse"]),
            real_error,
        ),
        "pe_off_to_real_position_error": _positive_ratio(
            float(conditions["pe_off"]["symmetric_position_mse"]),
            real_error,
        ),
    }
    criteria = {
        "eligible_video_fraction": _criterion(
            value=geometry["eligible_video_fraction"],
            relation="at_least",
            threshold=thresholds.minimum_eligible_video_fraction,
        ),
        "eligible_pair_fraction": _criterion(
            value=geometry["eligible_pair_fraction"],
            relation="at_least",
            threshold=thresholds.minimum_eligible_pair_fraction,
        ),
        "source_index_violation_total": _criterion(
            value=geometry["source_index_invariant_violations"]["total"],
            relation="at_most",
            threshold=thresholds.maximum_source_index_violation_total,
        ),
        "source_index_intersection_total": _criterion(
            value=geometry["source_index_invariant_violations"][
                "source_index_intersection_total"
            ],
            relation="at_most",
            threshold=0,
        ),
        "real_valid_anchor_fraction": _criterion(
            value=conditions["real"]["valid_anchor_fraction"],
            relation="at_least",
            threshold=thresholds.minimum_valid_anchor_fraction,
        ),
        "zero_to_real_position_error_ratio": _criterion(
            value=ratios["zero_to_real_position_error"],
            relation="at_least",
            threshold=thresholds.minimum_zero_to_real_position_error_ratio,
        ),
        "shuffle_to_real_position_error_ratio": _criterion(
            value=ratios["shuffle_to_real_position_error"],
            relation="at_least",
            threshold=thresholds.minimum_shuffle_to_real_position_error_ratio,
        ),
        "permuted_pe_to_real_position_error_ratio_minimum": _criterion(
            value=ratios["permuted_pe_to_real_position_error"],
            relation="at_least",
            threshold=thresholds.minimum_permuted_pe_to_real_position_error_ratio,
        ),
        "permuted_pe_to_real_position_error_ratio_maximum": _criterion(
            value=ratios["permuted_pe_to_real_position_error"],
            relation="at_most",
            threshold=thresholds.maximum_permuted_pe_to_real_position_error_ratio,
        ),
        "pe_off_to_real_position_error_ratio_minimum": _criterion(
            value=ratios["pe_off_to_real_position_error"],
            relation="at_least",
            threshold=thresholds.minimum_pe_off_to_real_position_error_ratio,
        ),
        "pe_off_to_real_position_error_ratio_maximum": _criterion(
            value=ratios["pe_off_to_real_position_error"],
            relation="at_most",
            threshold=thresholds.maximum_pe_off_to_real_position_error_ratio,
        ),
    }
    passed = all(bool(item["passed"]) for item in criteria.values())
    return {
        "thresholds_frozen_before_execution": True,
        "ratios": ratios,
        "criteria": criteria,
        "overall_pass": passed,
        "mechanism_probe_authorized": passed,
        "epoch11_encoder_continuation_authorized": False,
        "development_evaluation_authorized": False,
        "sealed_evaluation_authorized": False,
    }


def _loss_metrics(
    output: CycleBackLossOutput,
    embeddings_a: Tensor,
    embeddings_b: Tensor,
    pairs: NativeWindowPairBatch,
) -> dict[str, Any]:
    possible = int(pairs.valid_a.sum() + pairs.valid_b.sum())
    window_slots = 2 * pairs.pair_count * pairs.window_length
    symmetric_mse = 0.5 * (
        output.a_to_b_to_a.mean_squared_position_error
        + output.b_to_a_to_b.mean_squared_position_error
    )
    symmetric_variance = 0.5 * (
        output.a_to_b_to_a.mean_cycle_variance
        + output.b_to_a_to_b.mean_cycle_variance
    )
    return {
        "pair_total": pairs.pair_count,
        "symmetric_variance_aware_loss": float(output.total.detach()),
        "symmetric_position_mse": float(symmetric_mse.detach()),
        "symmetric_cycle_variance": float(symmetric_variance.detach()),
        "valid_anchor_total": output.valid_anchor_count,
        "possible_anchor_total": possible,
        "valid_anchor_fraction": (
            0.0 if possible == 0 else output.valid_anchor_count / possible
        ),
        "window_slot_total": window_slots,
        "valid_window_slot_fraction": (
            0.0 if window_slots == 0 else possible / window_slots
        ),
        "a_to_b_to_a_valid_anchor_total": output.a_to_b_to_a.valid_anchor_count,
        "b_to_a_to_b_valid_anchor_total": output.b_to_a_to_b.valid_anchor_count,
        "embedding_temporal_rms": numeric_summary(
            temporal_rms_values(embeddings_a, pairs.valid_a)
            + temporal_rms_values(embeddings_b, pairs.valid_b)
        ),
    }


def evaluate_cycleback_embeddings(
    objective: ConventionalCycleBackLoss,
    embeddings_a: Tensor,
    embeddings_b: Tensor,
    pairs: NativeWindowPairBatch,
) -> dict[str, Any]:
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
    return _loss_metrics(output, embeddings_a, embeddings_b, pairs)


def _evaluate_embeddings(
    objective: ConventionalCycleBackLoss,
    embeddings_a: Tensor,
    embeddings_b: Tensor,
    pairs: NativeWindowPairBatch,
) -> dict[str, Any]:
    """Backward-compatible private alias for existing audit/probe callers."""

    return evaluate_cycleback_embeddings(objective, embeddings_a, embeddings_b, pairs)


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _full_geometry(
    sequences: Sequence[Unified2DSequence],
    config: ConventionalCycleBackConfig,
    *,
    segment_ranges_by_video: Mapping[str, Sequence[tuple[int, int]]],
    pair_eligibility: PairEligibility,
) -> dict[str, Any]:
    window = config.window_pair
    available_total = 0
    raw_grid_total = 0
    eligible_total = 0
    eligible_videos = 0
    violations = {
        "start_offset": 0,
        "start_grid": 0,
        "a_start_identity": 0,
        "b_start_identity": 0,
        "a_contiguity": 0,
        "b_contiguity": 0,
        "source_index_intersection_total": 0,
        "segment_bounds": 0,
        "native_bounds": 0,
        "positional_capacity": 0,
        "total": 0,
    }
    valid_fractions: list[float] = []
    native_lengths: list[float] = []
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
        current = source_index_invariant_violations(
            pairs,
            expected_hop_frames=window.hop_frames,
        )
        for name in (
            "start_offset",
            "start_grid",
            "a_start_identity",
            "b_start_identity",
            "a_contiguity",
            "b_contiguity",
            "source_index_intersection_total",
            "segment_bounds",
            "native_bounds",
        ):
            violations[name] += current[name]
        if sequence.num_frames > config.encoder.maximum_native_length:
            violations["positional_capacity"] += 1
        available_total += pairs.available_pair_total
        raw_grid_total += pairs.raw_grid_pair_total
        eligible_total += pairs.pair_count
        eligible_videos += pairs.pair_count > 0
        valid_fractions.append(sequence.valid_fraction)
        native_lengths.append(float(sequence.num_frames))
    violations["total"] = sum(
        violations[name]
        for name in (
            "start_offset",
            "start_grid",
            "a_start_identity",
            "b_start_identity",
            "a_contiguity",
            "b_contiguity",
            "source_index_intersection_total",
            "segment_bounds",
            "native_bounds",
            "positional_capacity",
        )
    )
    video_total = len(sequences)
    return {
        "video_total": video_total,
        "eligible_video_total": eligible_videos,
        "eligible_video_fraction": 0.0 if not video_total else eligible_videos / video_total,
        "available_pair_total": available_total,
        "raw_grid_pair_total_diagnostic_only": raw_grid_total,
        "available_pair_denominator_policy": (
            "representation_authorized_same_segment_all_frame_valid_before_joint_support"
        ),
        "eligible_pair_total": eligible_total,
        "eligible_pair_fraction": (
            0.0 if not available_total else eligible_total / available_total
        ),
        "native_length_frames": numeric_summary(native_lengths),
        "valid_pose_fraction": numeric_summary(valid_fractions),
        "source_index_invariant_violations": violations,
    }


def _diagnostic_conditions(
    sequences: Sequence[Unified2DSequence],
    config: ConventionalCycleBackConfig,
    *,
    segment_ranges_by_video: Mapping[str, Sequence[tuple[int, int]]],
    pair_eligibility: PairEligibility,
    device: torch.device,
    encoder_batch_size: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    selected = ranked_sequences(
        sequences,
        seed=config.seed,
        purpose="geometry-pe-null-sample",
    )[: config.geometry_audit.diagnostic_sample_videos]
    batch = collate_to_device(selected, device)
    pairs, sequence_views = pairs_from_batch(
        batch,
        config,
        step=0,
        segment_ranges_by_video=segment_ranges_by_video,
        pair_eligibility=pair_eligibility,
    )
    pairs = cap_window_pairs(
        pairs,
        maximum_pairs=config.geometry_audit.maximum_probe_pairs,
        seed=config.seed,
        purpose="geometry-pe-null-pairs",
    )
    if pairs.pair_count < 2:
        raise ScientificGeometryReject(
            "geometry audit has fewer than two eligible sampled disjoint pairs"
        )
    contexts = pair_segment_contexts(pairs, sequence_views)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    encoder = build_encoder(config).to(device).eval()
    encoder_state_sha256 = _model_state_sha256(encoder)
    pe_off = build_encoder(config, position_encoding_mode="none").to(device).eval()
    pe_off.load_state_dict(encoder.state_dict(), strict=True)
    objective_config = config.objective
    objective = ConventionalCycleBackLoss(
        temperature=objective_config.temperature,
        variance_log_weight=objective_config.variance_log_weight,
        variance_floor=objective_config.variance_floor,
    )

    conditions: dict[str, dict[str, Any]] = {}
    with torch.inference_mode():
        real_a, real_b = encode_window_pairs(
            encoder,
            pairs,
            contexts,
            batch_size=encoder_batch_size,
        )
        conditions["real"] = _evaluate_embeddings(objective, real_a, real_b, pairs)

        shuffled_contexts = independently_permuted_pair_segment_contexts(
            contexts,
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
            zero_pair_segment_contexts(contexts),
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
            contexts
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
            contexts,
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

        pe_off_a, pe_off_b = encode_window_pairs(
            pe_off,
            pairs,
            contexts,
            batch_size=encoder_batch_size,
        )
        conditions["pe_off"] = _evaluate_embeddings(
            objective,
            pe_off_a,
            pe_off_b,
            pairs,
        )
    if _model_state_sha256(encoder) != encoder_state_sha256:
        raise RuntimeError("zero-step geometry audit changed encoder state")
    return conditions, {
        "sampled_training_video_total": len(selected),
        "sampled_pair_total": pairs.pair_count,
        "independent_view_seeds": list(sequence_views.view_seeds),
        "view_seed_streams_are_distinct": (
            sequence_views.view_seeds[0] != sequence_views.view_seeds[1]
        ),
        "encoder_context": dict(contexts.transform_contract),
        "disjoint_window_policy": (
            "adjacent_disjoint_same_video_windows_offset_by_window_length"
        ),
        "independent_temporal_null": dict(
            shuffled_contexts.transform_contract
        ),
        "independent_pe_null": dict(permuted_contexts.transform_contract),
        "joint_support_nulls": joint_null_contracts,
        "joint_support_null_ratios_are_report_only_at_random_initialization": True,
        "model_initialization_seed": config.seed,
        "encoder_training_steps": 0,
        "encoder_state_role": "E0_frozen_untrained_initialization",
        "encoder_state_sha256": encoder_state_sha256,
        "may_share_state_with_learned_readout_encoder": False,
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
    }


def run_geometry_audit(
    config_path: str | Path,
    pose_authority_root: str | Path,
    *,
    declared_pose_authority_host_root: str | Path,
    launch_registry_root: str | Path,
    declared_launch_registry_host_root: str | Path,
    source_root: str | Path,
    expected_pose_authorization_sha256: str,
    expected_pose_authority_run_receipt_sha256: str,
    source_git_sha: str,
    container_image_id: str,
    device: str | torch.device | None = None,
    encoder_batch_size: int | None = None,
) -> dict[str, Any]:
    """Run the complete read-only train337 geometry and 0-step null audit."""

    if not _GIT_SHA.fullmatch(source_git_sha):
        raise ValueError("source_git_sha must be a full lowercase Git SHA")
    if not _IMAGE_ID.fullmatch(container_image_id):
        raise ValueError("container_image_id must be an immutable image ID")
    launch = validate_cycleback_launch_registry(
        launch_registry_root,
        declared_host_root=declared_launch_registry_host_root,
        source_root=source_root,
    )
    if source_git_sha != launch.source_revision:
        raise ValueError("source_git_sha differs from canonical launch registry")
    if container_image_id != launch.container_image_id:
        raise ValueError("container_image_id differs from canonical launch registry")
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
    frozen_batch_size = config.geometry_audit.encoder_segment_context_batch_size
    if encoder_batch_size is not None and encoder_batch_size != frozen_batch_size:
        raise ValueError("encoder batch size differs from the fingerprinted config")
    authority = validate_cycleback_pose_authority(
        pose_authority_root,
        declared_host_root=declared_pose_authority_host_root,
        launch_authority=launch,
        expected_authorization_sha256=expected_pose_authorization_sha256,
        expected_run_receipt_sha256=expected_pose_authority_run_receipt_sha256,
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
    geometry = _full_geometry(
        sequences,
        config,
        segment_ranges_by_video=segment_ranges_by_video,
        pair_eligibility=pair_eligibility,
    )
    resolved_device = _device(device)
    try:
        conditions, diagnostic_scope = _diagnostic_conditions(
            sequences,
            config,
            segment_ranges_by_video=segment_ranges_by_video,
            pair_eligibility=pair_eligibility,
            device=resolved_device,
            encoder_batch_size=frozen_batch_size,
        )
        decision = geometry_gate_decision(
            geometry=geometry,
            conditions=conditions,
            thresholds=config.geometry_audit.thresholds,
        )
    except ScientificGeometryReject as exc:
        conditions = {"unavailable": {"reason": str(exc)}}
        diagnostic_scope = {
            "sampled_training_video_total": min(
                len(sequences), config.geometry_audit.diagnostic_sample_videos
            ),
            "sampled_pair_total": 0,
            "scientific_rejection": True,
            "reason": str(exc),
        }
        decision = {
            "thresholds_frozen_before_execution": True,
            "ratios": {},
            "criteria": {
                "null_controls_evaluable": {
                    "value": False,
                    "relation": "equal",
                    "threshold": True,
                    "passed": False,
                }
            },
            "overall_pass": False,
            "mechanism_probe_authorized": False,
            "epoch11_encoder_continuation_authorized": False,
            "development_evaluation_authorized": False,
            "sealed_evaluation_authorized": False,
        }
    if stable_file_identity(config_path) != config_identity:
        raise RuntimeError("cycle-back config changed during geometry audit")
    if stable_file_identity(authority.snapshot_path) != snapshot_identity:
        raise RuntimeError("pose snapshot changed during geometry audit")
    if stable_file_identity(authority.segment_index_path) != segment_index_identity:
        raise RuntimeError("pose segment index changed during geometry audit")
    if stable_file_identity(authority.joint_mask_snapshot_path) != joint_snapshot_identity:
        raise RuntimeError("joint-mask snapshot changed during geometry audit")
    if stable_file_identity(authority.identity_map_path) != identity_map_identity:
        raise RuntimeError("pose identity map changed during geometry audit")
    if (
        stable_file_identity(authority.pair_eligibility_path)
        != pair_eligibility_identity
    ):
        raise RuntimeError("pair-eligibility artifact changed during geometry audit")
    return {
        "schema_version": 1,
        "artifact_type": config.geometry_audit.artifact_type,
        "status": "passed" if decision["overall_pass"] else "rejected",
        "classification": config.classification,
        "paper_reference_status": config.paper_reference_status,
        "paper_table_claim_eligible": False,
        "namespace": config.namespace,
        "protocol": config.protocol,
        "candidate_id": config.candidate_id,
        "label_firewall": {
            "accepted_inputs": [
                "cycleback_proxy_config",
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
        "inputs": {
            "config_sha256": config_identity[0],
            "config_bytes": config_identity[1],
            "config_fingerprint": config.fingerprint,
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
            "launch_registry_receipt_sha256": launch.receipt_sha256,
            "training_video_total": len(sequences),
            "source_git_sha": source_git_sha,
            "source_tree_sha": launch.source_tree_sha,
            "container_image_id": container_image_id,
            "device_type": resolved_device.type,
            "encoder_segment_context_batch_size": frozen_batch_size,
        },
        "window_geometry": geometry,
        "pairing_contract": {
            "policy": config.window_pair.pair_policy,
            "a_interval": "[s,s+W)",
            "b_interval": "[s+W,s+2W)",
            "hop_controls_start_grid_only": True,
            "start_grid_policy": pair_eligibility.start_grid_policy,
            "pair_starts_are_exact_representation_authorized_subset": True,
            "consumer_may_expand_pair_starts": False,
            "source_index_intersection_total": geometry[
                "source_index_invariant_violations"
            ]["source_index_intersection_total"],
            "overlapping_pair_can_authorize_probe": False,
            "target_position_normalization": (
                config.objective.target_position_normalization
            ),
            "validity_policy": config.window_pair.validity_policy,
            "segment_policy": config.window_pair.segment_policy,
            "encoder_context_policy": config.window_pair.encoder_context_policy,
            "attention_boundary_policy": (
                config.window_pair.attention_boundary_policy
            ),
            "training_inference_context_parity": (
                config.window_pair.training_inference_context_parity
            ),
            "window_only_encoding_authorized": False,
            "reset_count_aggregation_policy": (
                config.window_pair.reset_count_aggregation_policy
            ),
        },
        "zero_step_diagnostic_scope": diagnostic_scope,
        "conditions": conditions,
        "pe_off_interpretation": {
            "scope": "same_initialized_encoder_with_position_input_disabled",
            "distribution_shift_warning": True,
            "separate_nope_candidate_claimed": False,
            "gate_role": "two_sided_shortcut_sensitivity_control_only",
            "ratio_interval_is_two_sided": True,
        },
        "gate": decision,
        "read_only_verification": {
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
        },
    }


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--launch-registry-root", type=Path, required=True)
    parser.add_argument("--declared-launch-registry-host-root", type=Path, required=True)
    parser.add_argument("--pose-authority-root", type=Path, required=True)
    parser.add_argument("--declared-pose-authority-host-root", type=Path, required=True)
    parser.add_argument("--expected-pose-authorization-sha256", required=True)
    parser.add_argument("--expected-pose-authority-run-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--encoder-batch-size", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    payload = run_geometry_audit(
        arguments.config,
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
        container_image_id=arguments.container_image_id,
        device=arguments.device,
        encoder_batch_size=arguments.encoder_batch_size,
    )
    _write_new_json(arguments.output, payload)
    print(
        json.dumps(
            {
                "artifact_type": payload["artifact_type"],
                "overall_pass": payload["gate"]["overall_pass"],
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["gate"]["overall_pass"] else 3
