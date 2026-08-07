"""Run the frozen, target-free train337 gate for dense-resampled PAMS readout v1.

The scientific input boundary is deliberately narrow: the exact policy, the
old training configuration, the two frozen checkpoints, one label-free
train337 sidecar/commitment pair, and its exact pose-cache set.  There is no
argument capable of accepting development/test identities, targets, counts,
or action labels.  Each checkpoint Period-Head action stream is generated once
and both decoders consume that same frozen in-memory stream.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import struct
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import yaml
from torch import Tensor

from pams.config import PAMSConfig, load_config
from pams.consensus import MultiExpertCounter
from pams.data import (
    PoseCacheEntryReceipt,
    PoseCacheSetSnapshot,
    load_pose_cache_set,
    load_pose_input_commitment,
    load_pose_input_manifest,
    validate_pose_input_binding,
)
from pams.diagnostics import _device, _identifier_commitment, _peek_checkpoint
from pams.losses import build_reference_relative_signal, masked_zscore
from pams.period import estimate_period_batch_direct_fft
from pams.reproducibility import clean_git_revision, hardware_fingerprint, sha256_json
from pams.run_manifest import CompletedRunReceipt
from pams.training import (
    _estimate_post_warmup_periods,
    collate_pose_sequences,
    load_model_checkpoint,
    validate_sshead_encoder_binding,
    validate_terminal_checkpoint,
)
from pams.types import PoseSequence

_ARTIFACT_TYPE = "pams_dense_timebase_v1_train337_gate"
_RECEIPT_TYPE = "pams_dense_timebase_v1_train337_gate_receipt"
_EXPECTED_CANDIDATE = "pams-reference-relative-dense-resampled-masked-dft-multi-v1"
_EXPECTED_POLICY_SHA256 = "828a6e64565274bfed1ff5c1822d98d1e3720bd0bea625d4c065ccc7aee63da4"
_EXPECTED_RECORDS = 337
_INFERENCE_BATCH_SIZE = 16
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

_EXPECTED_FROZEN_BASE = {
    "source_git_sha": "15cc1ec3c1d2c243ac3d01696207e96061e65515",
    "config_sha256": "eed60fd1c5795b3783f44d893f4bc4bd3764f4e8e546785218afcd7a37fba0f8",
    "training_config_fingerprint": (
        "32d086e1c2f76ee1c2beb76c984e0a6e6c60a05e1330c510fc9d0dde6b32054e"
    ),
    "pose_fingerprint": "f89cfc3e520cd3f45282ec06f42d5702f5e7c30c28ee6a3655ebfb5e58766047",
    "encoder_checkpoint_sha256": (
        "3f0591866b4c2691dc9b4bca5ddda75fac20dc55d767dcb3f24eacf31541b568"
    ),
    "encoder_progress_sha256": (
        "a5b77345c59d258ab13cc8fbe31fa31de0e803dbbb9af421fad5046ff29f59e3"
    ),
    "encoder_completion_receipt_sha256": (
        "a132815aadbdf7beebbe50255e314f00589c3c03691396cfc3d126b905572ff5"
    ),
    "sshead_checkpoint_sha256": (
        "4a37446c39ba4a73dfcfcfa569fa54800245bb14db82eeb353dae787e9742529"
    ),
    "sshead_progress_sha256": (
        "3b39d58633a431c60d8618f596158dc70af5632e263138ef3d523a00b3dd6325"
    ),
    "sshead_completion_receipt_sha256": (
        "18c974591cd7ff26d7b8283f498806f366256d4e7768215e117a440ea57b4e0f"
    ),
    "train337_inputs_sha256": (
        "f95df0050df21f05bbc9b42d0154470714dde279e04cb8b00d0212bf49057d16"
    ),
    "train337_commitment_sha256": (
        "85d41d2f59872e0900e9058481efbdf6bce91407d4c26bcde0a6547776454e53"
    ),
    "train337_pose_snapshot_file_sha256": (
        "214403d675d5483eedb831fd8f8a629a4635bf075271103a02a23fead513e5f2"
    ),
    "train337_pose_cache_set_sha256": (
        "03259ec8d914af5613785ba2eee3003d276b638bda5f0cc993e425188a4259bd"
    ),
}

_REQUIRED_THRESHOLDS = {
    "completed_records",
    "finite_records",
    "period_evidence_records_minimum",
    "collapsed_fraction_maximum",
    "invalid_payload_exact_invariance_fraction",
    "period_bin_closure_error_maximum",
    "fixed_internal_gap_period_bin_retention_minimum",
    "fixed_internal_gap_count_within_one_minimum",
    "two_x_resample_period_scaling_median_error_maximum",
    "two_x_resample_period_scaling_p90_error_maximum",
    "two_x_resample_count_equal_fraction_minimum",
    "compact_decoder_non_regression_required",
    "one_mask_robustness_metric_relative_improvement_minimum",
    "boundary_rate_increase_maximum",
    "coverage_strata_report_required",
    "all_selected_modes_multi",
}
_EXPECTED_SYNTHETIC_CHECKS = {
    "legacy_all_valid_exact_parity",
    "median_period_relative_error",
    "p95_period_relative_error",
    "harmonic_alias_fraction",
    "chain_nmae",
    "chain_obo",
    "clean_majority_fraction",
    "harmonic_majority_fraction",
    "resample_period_relative_error",
    "resample_count_equal_fraction",
    "invalid_payload_exact_invariance",
    "all_selected_modes_multi",
    "existing_576_counter_gate",
}
_EXPECTED_POLICY_METRIC_DEFINITIONS = {
    "stream_std": "population_std_over_original_valid_samples_valid_lt_2_is_zero",
    "collapsed": "stream_std_le_1e-6",
    "period_evidence": "direct_fft_confidence_gt_zero",
    "period_bin_closure": "abs_period_minus_timeline_over_rounded_bin_div_period",
    "invalid_payload": "alternating_plus_minus_1e6_at_original_invalid_positions",
    "invalid_payload_exact_fields": [
        "period",
        "period_confidence",
        "final_count",
        "expert_counts",
        "reference_count",
        "selected_expert",
        "selection_mode",
        "consensus_confidence",
    ],
    "internal_gap_interval": "half_open_floor_7L_over_16_to_ceil_9L_over_16",
    "internal_gap_affected": "at_least_one_original_valid_sample_removed",
    "dense_period_bin": "round_timeline_frames_over_period",
    "compact_period_bin": "round_valid_frames_over_period",
    "fixed_internal_gap_period_bin_retention": (
        "mean_gap_bin_equal_base_bin_on_affected_dual_evidence_rows"
    ),
    "fixed_internal_gap_count_within_one": (
        "mean_abs_final_gap_count_minus_base_count_le_one_on_all_affected_rows"
    ),
    "two_x_resample": "repeat_interleave_stream_and_mask_and_double_period_bounds",
    "two_x_resample_period_scaling_error": (
        "abs_resampled_period_over_twice_base_period_minus_one_on_dual_evidence_rows"
    ),
    "two_x_resample_count_equal": (
        "mean_resampled_final_count_equal_base_final_count_over_all_records"
    ),
    "mask_robustness_row_error": (
        "abs_log2_gap_period_over_base_period_plus_abs_gap_count_minus_base_count_over_"
        "max_1_base_count"
    ),
    "mask_robustness_metric": (
        "mean_row_error_on_shared_affected_dual_decoder_evidence_rows"
    ),
    "mask_robustness_relative_improvement": (
        "compact_minus_dense_over_max_compact_1e-12"
    ),
    "compact_decoder_non_regression": (
        "dense_mask_robustness_le_compact_plus_1e-12"
    ),
    "boundary_rate": (
        "fraction_selected_bins_at_either_configured_fft_band_edge_over_paired_base_and_"
        "gap_evidence_decodes"
    ),
    "boundary_rate_increase": "dense_boundary_rate_minus_compact_boundary_rate",
    "fail_closed": [
        "no_eligible_rows_for_a_required_metric",
        "compact_mask_robustness_metric_le_1e-12",
    ],
    "coverage_strata": {
        "empty": "coverage_eq_zero",
        "low": "coverage_gt_zero_and_lt_0.25",
        "medium": "coverage_ge_0.25_and_lt_0.75",
        "high": "coverage_ge_0.75_and_lt_one",
        "full": "coverage_eq_one",
    },
}


@dataclass(frozen=True, slots=True)
class _StreamRecord:
    video_id: str
    stream: Tensor
    valid_mask: Tensor

    @property
    def timeline_frames(self) -> int:
        return int(self.stream.numel())


def _encoded_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _stable_file_identity(path: str | Path) -> tuple[str, int]:
    """Hash one regular non-symlink file while detecting concurrent mutation."""

    source = Path(path)
    before = source.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"gate input must be a regular non-symlink file: {source}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        opened = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
            closed = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = source.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(closed, field)
        or getattr(closed, field) != getattr(after, field)
        for field in fields
    ):
        raise RuntimeError(f"gate input changed while it was read: {source}")
    if byte_count != after.st_size:
        raise RuntimeError(f"gate input byte count changed while reading: {source}")
    return digest.hexdigest(), byte_count


def _load_policy(path: str | Path) -> tuple[dict[str, Any], str, str]:
    source = Path(path)
    digest, _ = _stable_file_identity(source)
    if digest != _EXPECTED_POLICY_SHA256:
        raise ValueError("train337 gate requires the exact frozen dense-timebase v1 policy")
    raw = source.read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, dict):
        raise ValueError("dense-timebase policy root must be a mapping")
    policy = dict(parsed)
    if policy.get("schema_version") != 1:
        raise ValueError("dense-timebase policy requires schema_version=1")
    if policy.get("candidate_id") != _EXPECTED_CANDIDATE:
        raise ValueError("dense-timebase policy candidate_id drifted")
    if policy.get("table2_eligible") is not False:
        raise ValueError("dense-timebase candidate must remain Table-2 ineligible")
    if policy.get("frozen_base") != _EXPECTED_FROZEN_BASE:
        raise ValueError("dense-timebase frozen-base hashes or fingerprints drifted")

    readout = policy.get("readout")
    expected_readout = {
        "scope": "final_period_head_stream_fft_and_fallback_only",
        "direct_fft_timebase": "dense_resampled",
        "timeline": "full_resampled_clip",
        "timeline_frames": 256,
        "detrend_fit": "valid_samples_at_dense_indices",
        "invalid_sample_contribution": "zero_after_valid_only_affine_detrend",
        "interpolation_across_invalid_frames": False,
        "frequency_definition": "k_over_timeline_frames",
        "period_definition": "timeline_frames_over_k",
        "reference_count": "floor_timeline_frames_over_period",
        "reference_support": "full_timeline_label_free_extrapolation",
        "expert_support": "contiguous_valid_runs_only",
        "support_asymmetry_frozen": True,
        "consensus_expert_mode": "multi",
        "expert_parameters": "unchanged_from_frozen_training_config",
        "majority_first": True,
        "no_majority_fallback": "fft_reference_nearest",
        "checkpoint_retraining": False,
        "bootstrap_period_estimator": "unchanged",
        "bootstrap_valid_count_upper_bound": "unchanged",
        "sparse_mask_bootstrap_truncation": "known_coverage_gated_limitation",
    }
    if readout != expected_readout:
        raise ValueError("dense-timebase readout semantics drifted")

    boundary = policy.get("selection_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("dense-timebase policy is missing selection_boundary")
    if boundary.get("candidate_count") != 1 or boundary.get("parameter_sweep") is not False:
        raise ValueError("train337 gate permits exactly one unswept candidate")
    if boundary.get("train337_use") != "target_free_pass_fail_only":
        raise ValueError("train337 target-free use drifted")
    if boundary.get("permitted_label_source") != (
        "deterministic_synthetic_generation_truth_only"
    ):
        raise ValueError("train337 gate may not use dataset labels")

    gate = policy.get("train337_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("dense-timebase policy is missing train337_gate")
    thresholds = gate.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != _REQUIRED_THRESHOLDS:
        raise ValueError("train337 threshold names do not match frozen v1")
    if thresholds.get("completed_records") != 337 or thresholds.get("finite_records") != 337:
        raise ValueError("train337 completion thresholds drifted")
    if gate.get("metric_definitions") != _EXPECTED_POLICY_METRIC_DEFINITIONS:
        raise ValueError("train337 metric definitions drifted from the frozen formulas")

    authorization = policy.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("dense-timebase policy is missing authorization")
    expected_authorization = {
        "synthetic_must_pass_before_train337": True,
        "train337_must_pass_before_dev84": True,
        "dev84_prediction_attempt_budget": 1,
        "dev84_score_attempt_budget": 1,
        "dev84_success_nmae_maximum": 0.60,
        "dev84_success_obo_minimum": 0.30,
        "test105_attempt_budget": 0,
        "on_any_gate_failure": "stop_without_dev84",
        "on_dev84_failure": "archive_without_readout_switch",
    }
    if authorization != expected_authorization:
        raise ValueError("dense-timebase authorization boundary drifted")
    if _stable_file_identity(source)[0] != digest:
        raise RuntimeError("dense-timebase policy changed while it was validated")
    return policy, digest, sha256_json(policy)


def _load_strict_json_object(path: Path, *, document_name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field in {document_name}: {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {document_name}: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {document_name} JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{document_name} root must be an object")
    return payload


def _validate_frozen_pose_snapshot_file(
    path: Path,
    *,
    config: PAMSConfig,
) -> PoseCacheSetSnapshot:
    payload = _load_strict_json_object(
        path,
        document_name="frozen train337 pose-cache snapshot",
    )
    if set(payload) != {
        "schema_version",
        "pose_fingerprint",
        "fingerprint",
        "entry_count",
        "entries",
    }:
        raise ValueError("frozen train337 pose-cache snapshot schema drifted")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError("frozen train337 pose-cache snapshot entries must be a list")
    entries: list[PoseCacheEntryReceipt] = []
    for index, raw in enumerate(raw_entries):
        if not isinstance(raw, Mapping) or set(raw) != {"video_id", "cache_sha256", "bytes"}:
            raise ValueError(f"pose-cache snapshot entry schema drifted at index {index}")
        entries.append(
            PoseCacheEntryReceipt(
                video_id=raw["video_id"],
                cache_sha256=raw["cache_sha256"],
                bytes=raw["bytes"],
            )
        )
    snapshot = PoseCacheSetSnapshot(
        schema_version=payload["schema_version"],
        pose_fingerprint=payload["pose_fingerprint"],
        entries=tuple(entries),
    )
    if payload != snapshot.to_dict():
        raise ValueError("frozen train337 pose-cache snapshot canonical payload drifted")
    if len(snapshot.entries) != _EXPECTED_RECORDS:
        raise ValueError("frozen pose-cache snapshot must contain exactly 337 entries")
    if snapshot.pose_fingerprint != config.pose_fingerprint:
        raise ValueError("frozen pose-cache snapshot pose fingerprint differs from config")
    if snapshot.fingerprint != _EXPECTED_FROZEN_BASE["train337_pose_cache_set_sha256"]:
        raise ValueError("frozen pose-cache snapshot internal fingerprint drifted")
    return snapshot


def _validate_completion_receipt(
    receipt_path: Path,
    *,
    expected_stage: Literal["encoder", "sshead"],
    config: PAMSConfig,
    identities: Mapping[str, tuple[str, int]],
) -> dict[str, Any]:
    """Validate an exact frozen terminal-training receipt without opening other splits."""

    receipt = CompletedRunReceipt.model_validate(
        _load_strict_json_object(
            receipt_path,
            document_name=f"{expected_stage} completion receipt",
        )
    )
    if receipt.status != "completed" or receipt.receipt_type != "completed":
        raise ValueError(f"{expected_stage} training receipt is not completed")
    if receipt.started.protocol != config.protocol or receipt.started.seed != config.seed:
        raise ValueError(f"{expected_stage} training receipt protocol or seed drifted")
    if receipt.started.config_sha256 != config.fingerprint:
        raise ValueError(f"{expected_stage} training receipt config binding drifted")
    if receipt.started.git_sha != _EXPECTED_FROZEN_BASE["source_git_sha"]:
        raise ValueError(f"{expected_stage} training receipt source binding drifted")
    expected_epochs = (
        config.training.epochs if expected_stage == "encoder" else config.sshead.epochs
    )
    if receipt.metrics.get("completed_epochs") != expected_epochs:
        raise ValueError(f"{expected_stage} completion receipt is not terminal")
    artifacts = {artifact.role: artifact for artifact in receipt.artifacts}
    output_role = (
        "output_encoder_checkpoint"
        if expected_stage == "encoder"
        else "output_sshead_checkpoint"
    )
    checkpoint_key = f"{expected_stage}_checkpoint"
    progress_key = f"{expected_stage}_progress"
    required = {
        output_role,
        "progress_log",
        "input_config",
        "input_pose_cache_snapshot",
    }
    if expected_stage == "sshead":
        required.update({"input_encoder_checkpoint", "input_encoder_progress"})
    missing = sorted(required - set(artifacts))
    if missing:
        raise ValueError(
            f"{expected_stage} completion receipt is missing bindings: {missing}"
        )

    expected_live = {
        output_role: identities[checkpoint_key],
        "progress_log": identities[progress_key],
        "input_config": identities["training_config"],
        "input_pose_cache_snapshot": identities["train337_pose_snapshot"],
    }
    if expected_stage == "sshead":
        expected_live.update(
            {
                "input_encoder_checkpoint": identities["encoder_checkpoint"],
                "input_encoder_progress": identities["encoder_progress"],
            }
        )
    for role, (digest, byte_count) in expected_live.items():
        artifact = artifacts[role]
        if artifact.sha256 != digest or artifact.bytes != byte_count:
            raise ValueError(
                f"{expected_stage} completion receipt does not bind live {role} bytes"
            )
    if artifacts["input_pose_cache_snapshot"].sha256 != _EXPECTED_FROZEN_BASE[
        "train337_pose_snapshot_file_sha256"
    ]:
        raise ValueError(
            f"{expected_stage} completion receipt does not bind the frozen snapshot file"
        )
    return {
        "run_id": receipt.run_id,
        "status": receipt.status,
        "stage": expected_stage,
        "completed_epochs": expected_epochs,
        "started_git_sha": receipt.started.git_sha,
        "started_config_fingerprint": receipt.started.config_sha256,
        "started_dataset_sha256": receipt.started.dataset_sha256,
        "checkpoint_sha256": artifacts[output_role].sha256,
        "progress_sha256": artifacts["progress_log"].sha256,
    }


def _validate_synthetic_gate_pair(
    artifact_path: Path,
    receipt_path: Path,
    *,
    identities: Mapping[str, tuple[str, int]],
    policy_sha256: str,
    policy_semantic_sha256: str,
    synthetic_thresholds: Mapping[str, Any],
    runtime_source_git_sha: str,
) -> dict[str, Any]:
    artifact = _load_strict_json_object(
        artifact_path,
        document_name="synthetic dense-timebase gate artifact",
    )
    receipt = _load_strict_json_object(
        receipt_path,
        document_name="synthetic dense-timebase gate receipt",
    )
    authorization = artifact.get("authorization")
    source = artifact.get("source")
    checks = artifact.get("checks")
    if not isinstance(authorization, Mapping) or not isinstance(source, Mapping):
        raise ValueError("synthetic gate artifact is missing source or authorization")
    if (
        not isinstance(checks, Mapping)
        or set(checks) != _EXPECTED_SYNTHETIC_CHECKS
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("synthetic gate artifact does not contain an all-pass check set")
    expected_artifact = {
        "schema_version": 1,
        "artifact_type": "pams_dense_timebase_v1_synthetic_gate",
        "candidate_id": _EXPECTED_CANDIDATE,
        "status": "passed",
        "passed": True,
        "table2_eligible": False,
        "labels_accessed": False,
        "dataset_inputs_accessed": False,
        "checkpoint_inputs_accessed": False,
        "dev84_inputs_accessed": False,
        "test105_inputs_accessed": False,
    }
    for key, expected in expected_artifact.items():
        if artifact.get(key) != expected:
            raise ValueError(f"synthetic gate artifact field {key!r} is not authorized")
    if source.get("policy_sha256") != policy_sha256 or source.get(
        "policy_semantic_sha256"
    ) != policy_semantic_sha256:
        raise ValueError("synthetic gate artifact uses a different frozen policy")
    synthetic_runner_sha256 = identities["synthetic_gate_runner"][0]
    if source.get("runner_source_git_sha") != runtime_source_git_sha or source.get(
        "runner_sha256"
    ) != synthetic_runner_sha256:
        raise ValueError("synthetic artifact is not bound to the current source and runner")
    if artifact.get("thresholds") != dict(synthetic_thresholds):
        raise ValueError("synthetic artifact thresholds differ from the frozen policy")
    configuration = artifact.get("configuration")
    expected_configuration = {
        "direct_fft_timebase": "dense_resampled",
        "comparison_timebase": "compact_valid",
        "consensus_expert_mode": "multi",
        "candidate_count": 1,
        "parameter_sweep": False,
    }
    if not isinstance(configuration, Mapping) or any(
        configuration.get(key) != expected
        for key, expected in expected_configuration.items()
    ):
        raise ValueError("synthetic artifact decoder configuration drifted")
    expected_authorization = {
        "train337_gate_authorized": True,
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }
    if dict(authorization) != expected_authorization:
        raise ValueError("synthetic gate authorization boundary drifted")

    artifact_digest, artifact_bytes = identities["synthetic_gate_artifact"]
    expected_receipt = {
        "schema_version": 1,
        "artifact_type": "pams_dense_timebase_v1_synthetic_gate_receipt",
        "artifact_locator": artifact_path.name,
        "artifact_sha256": artifact_digest,
        "artifact_bytes": artifact_bytes,
        "artifact_status": "passed",
        "candidate_id": _EXPECTED_CANDIDATE,
        "policy_sha256": policy_sha256,
        "policy_semantic_sha256": policy_semantic_sha256,
        "labels_accessed": False,
        "dataset_inputs_accessed": False,
        "checkpoint_inputs_accessed": False,
        "train337_gate_authorized": True,
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }
    for key, expected in expected_receipt.items():
        if receipt.get(key) != expected:
            raise ValueError(f"synthetic gate receipt field {key!r} is not bound")
    if set(receipt) != set(expected_receipt) | {"runner_source_git_sha", "runner_sha256"}:
        raise ValueError("synthetic gate receipt schema drifted")
    if receipt.get("runner_source_git_sha") != runtime_source_git_sha or receipt.get(
        "runner_sha256"
    ) != synthetic_runner_sha256:
        raise ValueError("synthetic receipt is not bound to the current source and runner")
    if source.get("runner_source_git_sha") != receipt["runner_source_git_sha"] or source.get(
        "runner_sha256"
    ) != receipt["runner_sha256"]:
        raise ValueError("synthetic artifact and receipt runner identities differ")
    return {
        "artifact_sha256": artifact_digest,
        "receipt_sha256": identities["synthetic_gate_receipt"][0],
        "runner_sha256": synthetic_runner_sha256,
        "runner_source_git_sha": runtime_source_git_sha,
        "status": "passed",
        "train337_gate_authorized": True,
    }


def _validate_runtime_provenance(
    *,
    source_git_sha: str,
    container_image_id: str,
    container_environment_sha256: str,
) -> dict[str, str]:
    source = str(source_git_sha).strip()
    image = str(container_image_id).strip()
    environment = str(container_environment_sha256).strip().lower()
    if not _GIT_SHA_PATTERN.fullmatch(source):
        raise ValueError("source_git_sha must be a lowercase 40-character Git SHA")
    if not _IMAGE_ID_PATTERN.fullmatch(image):
        raise ValueError("container_image_id must be an immutable sha256:<digest> ID")
    if not _SHA256_PATTERN.fullmatch(environment):
        raise ValueError("container_environment_sha256 must be a SHA-256 digest")
    if clean_git_revision(Path.cwd()) != source:
        raise ValueError("runtime source Git SHA does not match the clean repository")
    expected_environment = {
        "PAMS_CONTAINER_SOURCE_REVISION": source,
        "PAMS_CONTAINER_IMAGE_ID": image,
        "PAMS_CONTAINER_ENVIRONMENT_SHA256": environment,
    }
    observed_environment = {key: os.environ.get(key) for key in expected_environment}
    if observed_environment != expected_environment:
        raise ValueError("runtime container provenance environment does not match arguments")
    return {
        "source_git_sha": source,
        "container_image_id": image,
        "container_environment_sha256": environment,
    }


def _counter_from_config(config: PAMSConfig) -> MultiExpertCounter:
    consensus = config.consensus
    counter = MultiExpertCounter(
        sigma_multipliers=consensus.sigma_multipliers,
        distance_multipliers=consensus.distance_multipliers,
        short_window_multiplier=consensus.short_window_multiplier,
        long_window_multiplier=consensus.long_window_multiplier,
        height_factor=consensus.height_factor,
        prominence_factor=consensus.prominence_factor,
        long_window_weight=consensus.long_window_weight,
        expert_mode=consensus.expert_mode,
    )
    if counter.expert_mode != "multi":
        raise ValueError("frozen train337 gate requires the original multi-expert vote")
    return counter


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _generate_period_head_streams_once(
    model: torch.nn.Module,
    sequences: Sequence[PoseSequence],
    config: PAMSConfig,
    *,
    device: torch.device,
) -> tuple[_StreamRecord, ...]:
    """Generate every original Period-Head stream exactly once."""

    if config.sshead.input_source != "projected_pose_reference_relative":
        raise ValueError("frozen dense-timebase v1 requires the reference-relative SSHead")
    typed_model = cast(Any, model)
    typed_model.to(device)
    typed_model.eval()
    records: list[_StreamRecord] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), _INFERENCE_BATCH_SIZE):
            items = sequences[start : start + _INFERENCE_BATCH_SIZE]
            batch = collate_pose_sequences(items).to(device)
            embeddings, projected_pose = typed_model.encoder.forward_with_pre_pe(
                batch.poses,
                batch.valid_mask,
            )
            periods, confidences, _ = _estimate_post_warmup_periods(
                config=config,
                embeddings=embeddings,
                projected_pose=projected_pose,
                valid_mask=batch.valid_mask,
            )
            reference = build_reference_relative_signal(
                projected_pose,
                periods,
                batch.valid_mask,
                period_confidence=confidences,
            )
            streams = typed_model.period_head(
                reference.head_inputs,
                valid_mask=batch.valid_mask,
            ).masked_fill(~batch.valid_mask, 0.0)
            streams = masked_zscore(
                streams,
                batch.valid_mask & reference.available.unsqueeze(1),
            )
            for index, sequence in enumerate(items):
                length = sequence.num_frames
                records.append(
                    _StreamRecord(
                        video_id=sequence.video_id,
                        stream=streams[index, :length].detach().to(device="cpu").contiguous(),
                        valid_mask=(
                            batch.valid_mask[index, :length]
                            .detach()
                            .to(device="cpu")
                            .contiguous()
                        ),
                    )
                )
    if len(records) != len(sequences):
        raise RuntimeError("Period-Head stream generation accounting mismatch")
    if [record.video_id for record in records] != [item.video_id for item in sequences]:
        raise RuntimeError("Period-Head stream order changed during generation")
    return tuple(records)


def _decode(
    stream: Tensor,
    mask: Tensor,
    *,
    minimum: int,
    maximum: int,
    timebase: Literal["compact_valid", "dense_resampled"],
    counter: MultiExpertCounter,
) -> dict[str, Any]:
    if stream.ndim != 1 or mask.shape != stream.shape or mask.dtype != torch.bool:
        raise ValueError("stream and boolean mask must have matching one-dimensional shape")
    timeline_frames = int(stream.numel())
    valid_frames = int(mask.sum())
    periods, confidences = estimate_period_batch_direct_fft(
        stream.unsqueeze(0),
        minimum=minimum,
        maximum=maximum,
        valid_mask=mask.unsqueeze(0),
        timebase=timebase,
        timeline_lengths=(
            torch.tensor([timeline_frames], dtype=torch.long)
            if timebase == "dense_resampled"
            else None
        ),
    )
    period = float(periods[0])
    period_confidence = float(confidences[0])
    clock_frames = timeline_frames if timebase == "dense_resampled" else valid_frames
    consensus = counter.count(
        stream,
        period,
        valid_mask=mask,
        period_confidence=period_confidence,
        reference_frames=(timeline_frames if timebase == "dense_resampled" else None),
    )
    selected_bin = int(round(clock_frames / period)) if clock_frames else 0
    return {
        "timebase": timebase,
        "timeline_frames": timeline_frames,
        "valid_frames": valid_frames,
        "clock_frames": clock_frames,
        "period_frames": period,
        "period_confidence": period_confidence,
        "period_evidence": period_confidence > 0.0,
        "selected_bin": selected_bin,
        "count": int(consensus.count),
        "expert_counts": [int(value) for value in consensus.expert_counts],
        "reference_count": int(consensus.reference_count),
        "selected_expert": consensus.selected_expert,
        "selection_mode": consensus.selection_mode,
        "consensus_confidence": float(consensus.confidence),
    }


def _internal_gap(mask: Tensor) -> tuple[Tensor, int, int, bool]:
    length = int(mask.numel())
    start = math.floor(7 * length / 16)
    stop = math.ceil(9 * length / 16)
    result = mask.clone()
    affected = bool(result[start:stop].any())
    result[start:stop] = False
    return result, start, stop, affected


def _pollute_invalid(stream: Tensor, mask: Tensor) -> Tensor:
    index = torch.arange(stream.numel(), device=stream.device)
    alternating = torch.where(
        index.remainder(2) == 0,
        stream.new_tensor(1_000_000.0),
        stream.new_tensor(-1_000_000.0),
    )
    return torch.where(mask, stream, alternating)


def _repeat_two_x(stream: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
    return stream.repeat_interleave(2), mask.repeat_interleave(2)


def _float_exact(left: float, right: float) -> bool:
    return struct.pack(">d", float(left)) == struct.pack(">d", float(right))


def _invalid_payload_exact(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    float_fields = ("period_frames", "period_confidence", "consensus_confidence")
    exact_fields = (
        "count",
        "expert_counts",
        "reference_count",
        "selected_expert",
        "selection_mode",
    )
    return all(_float_exact(left[field], right[field]) for field in float_fields) and all(
        left[field] == right[field] for field in exact_fields
    )


def _population_std(stream: Tensor, mask: Tensor) -> float | None:
    selected = stream[mask]
    if not bool(torch.isfinite(selected).all()):
        return None
    if selected.numel() < 2:
        return 0.0
    return float(selected.to(dtype=torch.float64).std(unbiased=False))


def _period_bin_closure(decode: Mapping[str, Any]) -> float | None:
    if not bool(decode["period_evidence"]):
        return None
    length = int(decode["timeline_frames"])
    period = float(decode["period_frames"])
    selected_bin = int(round(length / period))
    if selected_bin <= 0:
        raise RuntimeError("dense evidence decode produced a non-positive FFT bin")
    return abs(period - length / selected_bin) / max(period, 1e-12)


def _gap_robustness(base: Mapping[str, Any], gap: Mapping[str, Any]) -> float:
    return abs(math.log2(float(gap["period_frames"]) / float(base["period_frames"]))) + (
        abs(int(gap["count"]) - int(base["count"])) / max(1, int(base["count"]))
    )


def _allowed_bins(clock_frames: int, minimum: int, maximum: int) -> tuple[int, ...]:
    if clock_frames < 1:
        return ()
    upper_period = min(maximum, max(minimum, clock_frames - 1))
    frequencies = torch.fft.rfftfreq(clock_frames, d=1.0)
    allowed = (frequencies >= 1.0 / upper_period) & (frequencies <= 1.0 / minimum)
    if allowed.numel():
        allowed[0] = False
    return tuple(int(value) for value in torch.nonzero(allowed, as_tuple=False).flatten())


def _is_boundary_decode(
    decode: Mapping[str, Any],
    *,
    minimum: int,
    maximum: int,
) -> bool:
    if not bool(decode["period_evidence"]):
        raise ValueError("boundary classification requires period evidence")
    allowed = _allowed_bins(int(decode["clock_frames"]), minimum, maximum)
    if not allowed:
        raise RuntimeError("evidence decode has no allowed FFT bins")
    selected = int(decode["selected_bin"])
    if selected not in allowed:
        raise RuntimeError("selected FFT bin lies outside the decoder frequency band")
    return selected == allowed[0] or selected == allowed[-1]


def _finite_decode(decode: Mapping[str, Any]) -> bool:
    scalar_fields = (
        "timeline_frames",
        "valid_frames",
        "clock_frames",
        "period_frames",
        "period_confidence",
        "selected_bin",
        "count",
        "reference_count",
        "consensus_confidence",
    )
    values = [float(decode[field]) for field in scalar_fields]
    values.extend(float(value) for value in decode["expert_counts"])
    return bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())


def _fraction(values: Sequence[bool]) -> float | None:
    return None if not values else float(sum(values) / len(values))


def _mean(values: Sequence[float]) -> float | None:
    return None if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def _quantile(values: Sequence[float], quantile: float) -> float | None:
    return (
        None
        if not values
        else float(np.quantile(np.asarray(values, dtype=np.float64), quantile))
    )


def _coverage_stratum(coverage: float) -> str:
    if coverage == 0.0:
        return "empty"
    if 0.0 < coverage < 0.25:
        return "low"
    if 0.25 <= coverage < 0.75:
        return "medium"
    if 0.75 <= coverage < 1.0:
        return "high"
    if coverage == 1.0:
        return "full"
    raise ValueError("coverage must be finite and in [0, 1]")


def _summarize_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name in ("empty", "low", "medium", "high", "full"):
        selected = [row for row in rows if row["coverage_stratum"] == name]
        gap_rows = [
            row
            for row in selected
            if row["gap"]["dense"]["eligible"]
        ]
        affected_rows = [row for row in selected if row["gap"]["affected"]]
        report[name] = {
            "records": len(selected),
            "evidence_records": sum(
                bool(row["base"]["dense"]["period_evidence"]) for row in selected
            ),
            "evidence_fraction": _fraction(
                [bool(row["base"]["dense"]["period_evidence"]) for row in selected]
            ),
            "collapsed_records": sum(bool(row["collapsed"]) for row in selected),
            "collapsed_fraction": _fraction([bool(row["collapsed"]) for row in selected]),
            "base_period_median": _quantile(
                [float(row["base"]["dense"]["period_frames"]) for row in selected],
                0.5,
            ),
            "base_period_confidence_median": _quantile(
                [float(row["base"]["dense"]["period_confidence"]) for row in selected],
                0.5,
            ),
            "base_final_count_median": _quantile(
                [float(row["base"]["dense"]["count"]) for row in selected],
                0.5,
            ),
            "gap_eligible_records": len(gap_rows),
            "gap_affected_records": len(affected_rows),
            "gap_period_bin_retention": _fraction(
                [bool(row["gap"]["dense"]["period_bin_retained"]) for row in gap_rows]
            ),
            "gap_count_within_one": _fraction(
                [
                    bool(row["gap"]["dense"]["count_within_one"])
                    for row in affected_rows
                ]
            ),
        }
    return report


def _evaluate_checks(metrics: Mapping[str, Any], thresholds: Mapping[str, Any]) -> dict[str, bool]:
    def present(name: str) -> float | None:
        value = metrics.get(name)
        return None if value is None else float(value)

    closure = present("period_bin_closure_error_maximum")
    gap_retention = present("fixed_internal_gap_period_bin_retention")
    gap_count = present("fixed_internal_gap_count_within_one")
    scaling_median = present("two_x_resample_period_scaling_median_error")
    scaling_p90 = present("two_x_resample_period_scaling_p90_error")
    count_equal = present("two_x_resample_count_equal_fraction")
    improvement = present("mask_robustness_relative_improvement")
    boundary = present("boundary_rate_increase")
    return {
        "completed_records": int(metrics["completed_records"])
        == int(thresholds["completed_records"]),
        "finite_records": int(metrics["finite_records"])
        == int(thresholds["finite_records"]),
        "period_evidence_records": int(metrics["period_evidence_records"])
        >= int(thresholds["period_evidence_records_minimum"]),
        "collapsed_fraction": float(metrics["collapsed_fraction"])
        <= float(thresholds["collapsed_fraction_maximum"]),
        "invalid_payload_exact_invariance_fraction": float(
            metrics["invalid_payload_exact_invariance_fraction"]
        )
        >= float(thresholds["invalid_payload_exact_invariance_fraction"]),
        "period_bin_closure_error": closure is not None
        and closure <= float(thresholds["period_bin_closure_error_maximum"]),
        "fixed_internal_gap_period_bin_retention": gap_retention is not None
        and gap_retention
        >= float(thresholds["fixed_internal_gap_period_bin_retention_minimum"]),
        "fixed_internal_gap_count_within_one": gap_count is not None
        and gap_count >= float(thresholds["fixed_internal_gap_count_within_one_minimum"]),
        "two_x_resample_period_scaling_median_error": scaling_median is not None
        and scaling_median
        <= float(thresholds["two_x_resample_period_scaling_median_error_maximum"]),
        "two_x_resample_period_scaling_p90_error": scaling_p90 is not None
        and scaling_p90
        <= float(thresholds["two_x_resample_period_scaling_p90_error_maximum"]),
        "two_x_resample_count_equal_fraction": count_equal is not None
        and count_equal
        >= float(thresholds["two_x_resample_count_equal_fraction_minimum"]),
        "compact_decoder_non_regression": bool(metrics["compact_decoder_non_regression"])
        is bool(thresholds["compact_decoder_non_regression_required"]),
        "one_mask_robustness_metric_relative_improvement": improvement is not None
        and improvement
        >= float(thresholds["one_mask_robustness_metric_relative_improvement_minimum"]),
        "boundary_rate_increase": boundary is not None
        and boundary <= float(thresholds["boundary_rate_increase_maximum"]),
        "coverage_strata_report": bool(metrics["coverage_strata_reported"])
        is bool(thresholds["coverage_strata_report_required"]),
        "all_selected_modes_multi": bool(metrics["all_selected_modes_multi"])
        is bool(thresholds["all_selected_modes_multi"]),
    }


def _validate_exact_config(config: PAMSConfig, *, config_sha256: str) -> None:
    if config_sha256 != _EXPECTED_FROZEN_BASE["config_sha256"]:
        raise ValueError("train337 gate requires the exact frozen old training config")
    if config.fingerprint != _EXPECTED_FROZEN_BASE["training_config_fingerprint"]:
        raise ValueError("old training config fingerprint drifted")
    if config.pose_fingerprint != _EXPECTED_FROZEN_BASE["pose_fingerprint"]:
        raise ValueError("old training pose fingerprint drifted")
    if config.data.frames != 256:
        raise ValueError("dense-timebase v1 is frozen to 256 resampled frames")
    if config.period.direct_fft_timebase != "compact_valid":
        raise ValueError("training config must remain the pre-candidate compact default")
    if config.consensus.expert_mode != "multi":
        raise ValueError("training config must retain original multi-expert majority")
    if config.sshead.input_source != "projected_pose_reference_relative":
        raise ValueError("training config must use the frozen reference-relative SSHead")


def _validate_checkpoint_provenance(
    *,
    config: PAMSConfig,
    encoder_provenance: Any,
    sshead_provenance: Any,
    sidecar_video_ids: Sequence[str],
) -> None:
    encoder = encoder_provenance
    head = sshead_provenance
    if encoder.protocol != config.protocol or head.protocol != config.protocol:
        raise ValueError("checkpoint protocol differs from the frozen config")
    if encoder.pose_fingerprint != config.pose_fingerprint or head.pose_fingerprint != (
        config.pose_fingerprint
    ):
        raise ValueError("checkpoint pose fingerprint differs from the frozen config")
    if encoder.pose_cache_set_sha256 != _EXPECTED_FROZEN_BASE[
        "train337_pose_cache_set_sha256"
    ] or head.pose_cache_set_sha256 != _EXPECTED_FROZEN_BASE[
        "train337_pose_cache_set_sha256"
    ]:
        raise ValueError("checkpoint provenance does not bind the frozen train337 pose set")
    if encoder.source_git_sha != _EXPECTED_FROZEN_BASE["source_git_sha"] or (
        head.source_git_sha != _EXPECTED_FROZEN_BASE["source_git_sha"]
    ):
        raise ValueError("checkpoint algorithm source differs from the frozen base")
    identifiers = tuple(sorted(str(value) for value in sidecar_video_ids))
    if encoder.training_video_ids != identifiers or head.training_video_ids != identifiers:
        raise ValueError("checkpoint training IDs differ from the label-free train337 sidecar")
    if encoder.dataset_fingerprint != head.dataset_fingerprint:
        raise ValueError("encoder and SSHead dataset fingerprints differ")
    if encoder.container_image_id != head.container_image_id or (
        encoder.container_environment_sha256 != head.container_environment_sha256
    ):
        raise ValueError("encoder and SSHead training-container provenance differs")
    if encoder.upstream_encoder_checkpoint_sha256 is not None:
        raise ValueError("encoder checkpoint must not name an upstream encoder")
    if head.upstream_encoder_checkpoint_sha256 != _EXPECTED_FROZEN_BASE[
        "encoder_checkpoint_sha256"
    ]:
        raise ValueError("SSHead provenance does not bind the frozen encoder checkpoint")


def _definitions() -> dict[str, str]:
    return {
        "stream_collapse": (
            "For each original Period-Head action stream, population std is computed over "
            "original valid_mask samples with unbiased=False; valid<2 gives std=0; "
            "collapsed iff std<=1e-6."
        ),
        "period_evidence": "A decode has period evidence iff direct-FFT confidence>0.",
        "period_bin_closure": (
            "For each dense base evidence decode, L is original timeline length, "
            "k=round(L/T_hat)>0, and closure=abs(T_hat-L/k)/max(T_hat,1e-12); "
            "the metric is the maximum closure."
        ),
        "fixed_internal_gap": (
            "For timeline length L, [floor(7L/16),ceil(9L/16)) is forced invalid while "
            "the stream is unchanged; affected means the original mask has at least one "
            "valid sample in that interval. Each decoder compares its gap decode with its "
            "own original-mask base decode. Dense bin=round(L/T); compact "
            "bin=round(valid_count/T). Retention uses affected rows with period evidence in "
            "both base and gap; count-within-one uses all affected rows."
        ),
        "invalid_payload_invariance": (
            "Original-mask invalid positions are alternately replaced by +1e6/-1e6. Dense "
            "period, direct-FFT confidence, final count, expert counts, reference count, "
            "selected expert, selection mode, and consensus confidence must be byte/exact "
            "equal; the fraction denominator is all 337 records."
        ),
        "two_x_repeat_resample": (
            "Stream and mask use repeat_interleave(2), timeline is 2L, and period bounds are "
            "both multiplied by two. On rows with evidence in original and repeated dense "
            "decodes, scaling error=abs(T_2x/(2*T)-1) and its median/p90 are reported; exact "
            "final-count equality uses all 337 records."
        ),
        "mask_robustness": (
            "Core paired eligibility means affected and evidence in dense base+gap and "
            "compact base+gap. For each decoder and shared row, "
            "r=abs(log2(T_gap/T_base))+abs(C_gap-C_base)/max(1,C_base); metric=mean(r). "
            "relative_improvement=(compact_metric-dense_metric)/"
            "max(compact_metric,1e-12). non_regression iff dense_metric<=compact_metric+1e-12. "
            "No eligible rows or compact_metric<=1e-12 fails closed."
        ),
        "boundary_rate": (
            "Base and gap observations enter only when dense and compact both have period "
            "evidence, so boundary denominators are paired. Nclock=L for dense and valid_count "
            "for compact. Allowed bins use the same rfftfreq(Nclock) and config min/max band "
            "as the decoder. selected k=round(Nclock/T) is boundary iff it equals the allowed "
            "minimum or maximum k. increase=dense_rate-compact_rate."
        ),
        "coverage_strata": (
            "Original-mask coverage strata are empty=0, low=(0,.25), medium=[.25,.75), "
            "high=[.75,1), full=1. Each reports n/evidence/collapse/base period median/base "
            "confidence median/final count median and dense gap eligible/retention/"
            "count-within-one."
        ),
        "finite_records": (
            "A record is finite iff all original valid stream values and every scalar in its "
            "base, gap, polluted-invalid, and repeated-resample decode results are finite."
        ),
    }


def _evaluate_records(
    records: Sequence[_StreamRecord],
    *,
    config: PAMSConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    counter = _counter_from_config(config)
    minimum = config.period.minimum
    maximum = config.period.maximum
    rows: list[dict[str, Any]] = []
    all_modes: list[str] = []
    for record_index, record in enumerate(records):
        stream = record.stream
        mask = record.valid_mask
        length = record.timeline_frames
        if length != config.data.frames:
            raise ValueError("every frozen train337 pose sequence must contain 256 frames")

        base_dense = _decode(
            stream,
            mask,
            minimum=minimum,
            maximum=maximum,
            timebase="dense_resampled",
            counter=counter,
        )
        base_compact = _decode(
            stream,
            mask,
            minimum=minimum,
            maximum=maximum,
            timebase="compact_valid",
            counter=counter,
        )
        gap_mask, gap_start, gap_stop, affected = _internal_gap(mask)
        gap_dense = _decode(
            stream,
            gap_mask,
            minimum=minimum,
            maximum=maximum,
            timebase="dense_resampled",
            counter=counter,
        )
        gap_compact = _decode(
            stream,
            gap_mask,
            minimum=minimum,
            maximum=maximum,
            timebase="compact_valid",
            counter=counter,
        )
        polluted_dense = _decode(
            _pollute_invalid(stream, mask),
            mask,
            minimum=minimum,
            maximum=maximum,
            timebase="dense_resampled",
            counter=counter,
        )
        repeated_stream, repeated_mask = _repeat_two_x(stream, mask)
        repeated_dense = _decode(
            repeated_stream,
            repeated_mask,
            minimum=2 * minimum,
            maximum=2 * maximum,
            timebase="dense_resampled",
            counter=counter,
        )
        decodes = (
            base_dense,
            base_compact,
            gap_dense,
            gap_compact,
            polluted_dense,
            repeated_dense,
        )
        all_modes.extend(str(decode["selection_mode"]) for decode in decodes)
        population_std = _population_std(stream, mask)
        collapsed = population_std is None or population_std <= 1e-6
        dense_gap_eligible = affected and bool(base_dense["period_evidence"]) and bool(
            gap_dense["period_evidence"]
        )
        compact_gap_eligible = affected and bool(base_compact["period_evidence"]) and bool(
            gap_compact["period_evidence"]
        )
        resample_eligible = bool(base_dense["period_evidence"]) and bool(
            repeated_dense["period_evidence"]
        )
        coverage = int(mask.sum()) / length
        stream_finite = bool(torch.isfinite(stream[mask]).all())
        finite = stream_finite and all(_finite_decode(decode) for decode in decodes)
        row = {
            "record_index": record_index,
            "video_id_sha256": hashlib.sha256(record.video_id.encode("utf-8")).hexdigest(),
            "timeline_frames": length,
            "valid_frames": int(mask.sum()),
            "coverage": coverage,
            "coverage_stratum": _coverage_stratum(coverage),
            "stream_population_std": population_std,
            "collapsed": collapsed,
            "finite": finite,
            "base": {"dense": base_dense, "compact": base_compact},
            "period_bin_closure": _period_bin_closure(base_dense),
            "gap": {
                "start_inclusive": gap_start,
                "stop_exclusive": gap_stop,
                "affected": affected,
                "dense": {
                    "eligible": dense_gap_eligible,
                    "decode": gap_dense,
                    "period_bin_retained": (
                        gap_dense["selected_bin"] == base_dense["selected_bin"]
                        if dense_gap_eligible
                        else None
                    ),
                    "count_within_one": (
                        abs(int(gap_dense["count"]) - int(base_dense["count"])) <= 1
                        if affected
                        else None
                    ),
                    "robustness": (
                        _gap_robustness(base_dense, gap_dense)
                        if dense_gap_eligible
                        else None
                    ),
                },
                "compact": {
                    "eligible": compact_gap_eligible,
                    "decode": gap_compact,
                    "period_bin_retained": (
                        gap_compact["selected_bin"] == base_compact["selected_bin"]
                        if compact_gap_eligible
                        else None
                    ),
                    "count_within_one": (
                        abs(int(gap_compact["count"]) - int(base_compact["count"])) <= 1
                        if affected
                        else None
                    ),
                    "robustness": (
                        _gap_robustness(base_compact, gap_compact)
                        if compact_gap_eligible
                        else None
                    ),
                },
            },
            "invalid_payload": {
                "exact_invariance": _invalid_payload_exact(base_dense, polluted_dense),
                "decode": polluted_dense,
            },
            "two_x_repeat_resample": {
                "eligible": resample_eligible,
                "decode": repeated_dense,
                "period_scaling_error": (
                    abs(
                        float(repeated_dense["period_frames"])
                        / (2.0 * float(base_dense["period_frames"]))
                        - 1.0
                    )
                    if resample_eligible
                    else None
                ),
                "count_equal": (
                    int(repeated_dense["count"]) == int(base_dense["count"])
                ),
            },
        }
        rows.append(row)

    dense_gap_rows = [row for row in rows if row["gap"]["dense"]["eligible"]]
    compact_gap_rows = [row for row in rows if row["gap"]["compact"]["eligible"]]
    shared_gap_rows = [
        row
        for row in rows
        if row["gap"]["dense"]["eligible"]
        and row["gap"]["compact"]["eligible"]
    ]
    affected_gap_rows = [row for row in rows if row["gap"]["affected"]]
    repeated_rows = [row for row in rows if row["two_x_repeat_resample"]["eligible"]]
    dense_robustness_independent = _mean(
        [float(row["gap"]["dense"]["robustness"]) for row in dense_gap_rows]
    )
    compact_robustness_independent = _mean(
        [float(row["gap"]["compact"]["robustness"]) for row in compact_gap_rows]
    )
    dense_robustness = _mean(
        [float(row["gap"]["dense"]["robustness"]) for row in shared_gap_rows]
    )
    compact_robustness = _mean(
        [float(row["gap"]["compact"]["robustness"]) for row in shared_gap_rows]
    )
    robustness_well_defined = (
        dense_robustness is not None
        and compact_robustness is not None
        and compact_robustness > 1e-12
    )
    improvement = (
        (compact_robustness - dense_robustness) / max(compact_robustness, 1e-12)
        if robustness_well_defined
        else None
    )
    non_regression = bool(
        robustness_well_defined
        and dense_robustness <= compact_robustness + 1e-12
    )

    dense_boundary: list[bool] = []
    compact_boundary: list[bool] = []
    for row in rows:
        dense_base = row["base"]["dense"]
        compact_base = row["base"]["compact"]
        if dense_base["period_evidence"] and compact_base["period_evidence"]:
            dense_boundary.append(
                _is_boundary_decode(dense_base, minimum=minimum, maximum=maximum)
            )
            compact_boundary.append(
                _is_boundary_decode(compact_base, minimum=minimum, maximum=maximum)
            )
        dense_gap = row["gap"]["dense"]["decode"]
        compact_gap = row["gap"]["compact"]["decode"]
        if dense_gap["period_evidence"] and compact_gap["period_evidence"]:
            dense_boundary.append(
                _is_boundary_decode(dense_gap, minimum=minimum, maximum=maximum)
            )
            compact_boundary.append(
                _is_boundary_decode(compact_gap, minimum=minimum, maximum=maximum)
            )
    if len(dense_boundary) != len(compact_boundary):
        raise RuntimeError("paired boundary observation accounting mismatch")
    dense_boundary_rate = _fraction(dense_boundary)
    compact_boundary_rate = _fraction(compact_boundary)
    boundary_increase = (
        dense_boundary_rate - compact_boundary_rate
        if dense_boundary_rate is not None and compact_boundary_rate is not None
        else None
    )

    closures = [
        float(row["period_bin_closure"])
        for row in rows
        if row["period_bin_closure"] is not None
    ]
    scaling_errors = [
        float(row["two_x_repeat_resample"]["period_scaling_error"])
        for row in repeated_rows
    ]
    coverage = _summarize_coverage(rows)
    metrics = {
        "completed_records": len(rows),
        "finite_records": sum(bool(row["finite"]) for row in rows),
        "period_evidence_records": sum(
            bool(row["base"]["dense"]["period_evidence"]) for row in rows
        ),
        "collapsed_records": sum(bool(row["collapsed"]) for row in rows),
        "collapsed_fraction": float(np.mean([bool(row["collapsed"]) for row in rows])),
        "invalid_payload_exact_invariance_records": sum(
            bool(row["invalid_payload"]["exact_invariance"]) for row in rows
        ),
        "invalid_payload_exact_invariance_fraction": float(
            np.mean([bool(row["invalid_payload"]["exact_invariance"]) for row in rows])
        ),
        "period_bin_closure_evidence_records": len(closures),
        "period_bin_closure_error_maximum": max(closures) if closures else None,
        "fixed_internal_gap_dense_eligible_records": len(dense_gap_rows),
        "fixed_internal_gap_compact_eligible_records": len(compact_gap_rows),
        "fixed_internal_gap_shared_eligible_records": len(shared_gap_rows),
        "fixed_internal_gap_affected_records": len(affected_gap_rows),
        "fixed_internal_gap_period_bin_retention": _fraction(
            [bool(row["gap"]["dense"]["period_bin_retained"]) for row in dense_gap_rows]
        ),
        "fixed_internal_gap_count_within_one": _fraction(
            [
                bool(row["gap"]["dense"]["count_within_one"])
                for row in affected_gap_rows
            ]
        ),
        "fixed_internal_gap_compact_period_bin_retention": _fraction(
            [
                bool(row["gap"]["compact"]["period_bin_retained"])
                for row in compact_gap_rows
            ]
        ),
        "fixed_internal_gap_compact_count_within_one": _fraction(
            [
                bool(row["gap"]["compact"]["count_within_one"])
                for row in affected_gap_rows
            ]
        ),
        "two_x_resample_evidence_records": len(repeated_rows),
        "two_x_resample_period_scaling_median_error": _quantile(scaling_errors, 0.5),
        "two_x_resample_period_scaling_p90_error": _quantile(scaling_errors, 0.9),
        "two_x_resample_count_equal_fraction": _fraction(
            [bool(row["two_x_repeat_resample"]["count_equal"]) for row in rows]
        ),
        "dense_mask_robustness_metric": dense_robustness,
        "compact_mask_robustness_metric": compact_robustness,
        "dense_mask_robustness_metric_independent": dense_robustness_independent,
        "compact_mask_robustness_metric_independent": compact_robustness_independent,
        "mask_robustness_relative_improvement": improvement,
        "compact_decoder_non_regression": non_regression,
        "dense_boundary_observations": len(dense_boundary),
        "compact_boundary_observations": len(compact_boundary),
        "paired_boundary_observations": len(dense_boundary),
        "dense_boundary_rate": dense_boundary_rate,
        "compact_boundary_rate": compact_boundary_rate,
        "boundary_rate_increase": boundary_increase,
        "coverage_strata_reported": set(coverage)
        == {"empty", "low", "medium", "high", "full"},
        "all_selected_modes_multi": bool(all_modes)
        and all(mode == "multi" for mode in all_modes),
    }
    return {"metrics": metrics, "coverage_strata": coverage}, rows


def run_train337_gate(
    policy_path: str | Path,
    training_config_path: str | Path,
    sshead_checkpoint_path: str | Path,
    sshead_progress_path: str | Path,
    sshead_completion_receipt_path: str | Path,
    encoder_checkpoint_path: str | Path,
    encoder_progress_path: str | Path,
    encoder_completion_receipt_path: str | Path,
    train_sidecar_path: str | Path,
    train_commitment_path: str | Path,
    pose_cache_dir: str | Path,
    train_pose_snapshot_path: str | Path,
    synthetic_gate_artifact_path: str | Path,
    synthetic_gate_receipt_path: str | Path,
    *,
    device: str | torch.device | None,
    source_git_sha: str,
    container_image_id: str,
    container_environment_sha256: str,
) -> dict[str, Any]:
    """Execute the one frozen candidate using label-free train337 only."""

    runtime_provenance = _validate_runtime_provenance(
        source_git_sha=source_git_sha,
        container_image_id=container_image_id,
        container_environment_sha256=container_environment_sha256,
    )
    paths = {
        "policy": Path(policy_path),
        "training_config": Path(training_config_path),
        "sshead_checkpoint": Path(sshead_checkpoint_path),
        "sshead_progress": Path(sshead_progress_path),
        "sshead_completion_receipt": Path(sshead_completion_receipt_path),
        "encoder_checkpoint": Path(encoder_checkpoint_path),
        "encoder_progress": Path(encoder_progress_path),
        "encoder_completion_receipt": Path(encoder_completion_receipt_path),
        "train337_sidecar": Path(train_sidecar_path),
        "train337_commitment": Path(train_commitment_path),
        "train337_pose_snapshot": Path(train_pose_snapshot_path),
        "synthetic_gate_artifact": Path(synthetic_gate_artifact_path),
        "synthetic_gate_receipt": Path(synthetic_gate_receipt_path),
        "synthetic_gate_runner": Path(__file__).resolve().with_name(
            "run_pams_dense_timebase_v1_gate.py"
        ),
        "runner": Path(__file__).resolve(),
    }
    identities = {name: _stable_file_identity(path) for name, path in paths.items()}
    policy, policy_sha256, policy_semantic_sha256 = _load_policy(paths["policy"])
    base = policy["frozen_base"]
    exact_inputs = {
        "training_config": base["config_sha256"],
        "sshead_checkpoint": base["sshead_checkpoint_sha256"],
        "sshead_progress": base["sshead_progress_sha256"],
        "sshead_completion_receipt": base["sshead_completion_receipt_sha256"],
        "encoder_checkpoint": base["encoder_checkpoint_sha256"],
        "encoder_progress": base["encoder_progress_sha256"],
        "encoder_completion_receipt": base["encoder_completion_receipt_sha256"],
        "train337_sidecar": base["train337_inputs_sha256"],
        "train337_commitment": base["train337_commitment_sha256"],
        "train337_pose_snapshot": base["train337_pose_snapshot_file_sha256"],
    }
    observed_inputs = {name: identities[name][0] for name in exact_inputs}
    if observed_inputs != exact_inputs:
        raise ValueError(
            "train337 gate input identity mismatch: "
            + json.dumps(
                {"expected": exact_inputs, "observed": observed_inputs},
                sort_keys=True,
            )
        )

    config = load_config(paths["training_config"])
    _validate_exact_config(config, config_sha256=identities["training_config"][0])
    frozen_pose_snapshot = _validate_frozen_pose_snapshot_file(
        paths["train337_pose_snapshot"],
        config=config,
    )
    synthetic_binding = _validate_synthetic_gate_pair(
        paths["synthetic_gate_artifact"],
        paths["synthetic_gate_receipt"],
        identities=identities,
        policy_sha256=policy_sha256,
        policy_semantic_sha256=policy_semantic_sha256,
        synthetic_thresholds=policy["synthetic_gate"]["thresholds"],
        runtime_source_git_sha=runtime_provenance["source_git_sha"],
    )
    sidecar = load_pose_input_manifest(paths["train337_sidecar"], validate_exact=True)
    commitment = load_pose_input_commitment(paths["train337_commitment"])
    validate_pose_input_binding(
        sidecar,
        commitment,
        sidecar_sha256=identities["train337_sidecar"][0],
    )
    if sidecar.split != "train" or len(sidecar.records) != _EXPECTED_RECORDS:
        raise ValueError("gate requires the exact label-free UCFRep train337 sidecar")
    if tuple(entry.video_id for entry in frozen_pose_snapshot.entries) != tuple(
        sorted(record.video_id for record in sidecar.records)
    ):
        raise ValueError("frozen pose-cache snapshot IDs differ from train337 sidecar")

    encoder_stage, encoder_provenance = _peek_checkpoint(
        paths["encoder_checkpoint"], config
    )
    head_stage, head_provenance = _peek_checkpoint(paths["sshead_checkpoint"], config)
    if encoder_stage != "encoder" or head_stage != "sshead":
        raise ValueError("gate requires frozen encoder and SSHead checkpoint stages")
    _validate_checkpoint_provenance(
        config=config,
        encoder_provenance=encoder_provenance,
        sshead_provenance=head_provenance,
        sidecar_video_ids=[record.video_id for record in sidecar.records],
    )
    validate_sshead_encoder_binding(
        paths["sshead_checkpoint"],
        paths["encoder_checkpoint"],
    )
    validate_terminal_checkpoint(
        paths["encoder_checkpoint"],
        config,
        expected_stage="encoder",
        expected_provenance=encoder_provenance,
        progress_path=paths["encoder_progress"],
    )
    validate_terminal_checkpoint(
        paths["sshead_checkpoint"],
        config,
        expected_stage="sshead",
        expected_provenance=head_provenance,
        progress_path=paths["sshead_progress"],
    )
    encoder_completion = _validate_completion_receipt(
        paths["encoder_completion_receipt"],
        expected_stage="encoder",
        config=config,
        identities=identities,
    )
    sshead_completion = _validate_completion_receipt(
        paths["sshead_completion_receipt"],
        expected_stage="sshead",
        config=config,
        identities=identities,
    )
    if encoder_completion["started_dataset_sha256"] != (
        encoder_provenance.dataset_fingerprint
    ) or sshead_completion["started_dataset_sha256"] != head_provenance.dataset_fingerprint:
        raise ValueError("completion receipt dataset identity differs from checkpoint provenance")
    if encoder_completion["started_dataset_sha256"] != sshead_completion[
        "started_dataset_sha256"
    ]:
        raise ValueError("encoder and SSHead completion receipt dataset identities differ")

    sequences, pose_snapshot = load_pose_cache_set(
        sidecar.records,
        cache_dir=pose_cache_dir,
        pose_fingerprint=config.pose_fingerprint,
        materialize_sequences=True,
    )
    if len(sequences) != _EXPECTED_RECORDS:
        raise RuntimeError("pose-cache loader did not materialize all 337 records")
    if pose_snapshot.fingerprint != frozen_pose_snapshot.fingerprint or (
        pose_snapshot.fingerprint != base["train337_pose_cache_set_sha256"]
    ):
        raise ValueError("train337 pose-cache snapshot differs from the frozen policy")
    if pose_snapshot.fingerprint not in {
        encoder_provenance.pose_cache_set_sha256,
        head_provenance.pose_cache_set_sha256,
    } or encoder_provenance.pose_cache_set_sha256 != head_provenance.pose_cache_set_sha256:
        raise ValueError("pose-cache snapshot is not jointly checkpoint-bound")

    resolved_device = _device(device)
    model = load_model_checkpoint(
        paths["sshead_checkpoint"],
        config,
        device=resolved_device,
        expected_stage="sshead",
        expected_provenance=head_provenance,
    ).eval()
    state_before = _model_state_sha256(model)
    streams = _generate_period_head_streams_once(
        model,
        sequences,
        config,
        device=resolved_device,
    )
    evaluation, audit_rows = _evaluate_records(streams, config=config)
    state_after = _model_state_sha256(model)
    if state_before != state_after:
        raise RuntimeError("model state changed during the read-only train337 gate")

    _, final_pose_snapshot = load_pose_cache_set(
        sidecar.records,
        cache_dir=pose_cache_dir,
        pose_fingerprint=config.pose_fingerprint,
        materialize_sequences=False,
    )
    if final_pose_snapshot.fingerprint != pose_snapshot.fingerprint:
        raise RuntimeError("train337 pose-cache snapshot changed during the gate")
    for name, path in paths.items():
        if _stable_file_identity(path) != identities[name]:
            raise RuntimeError(f"gate input changed during execution: {name}")
    if clean_git_revision(Path.cwd()) != runtime_provenance["source_git_sha"]:
        raise RuntimeError("runtime source changed during train337 gate")

    thresholds = dict(policy["train337_gate"]["thresholds"])
    metrics = evaluation["metrics"]
    checks = _evaluate_checks(metrics, thresholds)
    passed = all(checks.values())
    hardware = hardware_fingerprint()
    return {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "candidate_id": policy["candidate_id"],
        "classification": policy["classification"],
        "status": "passed" if passed else "failed",
        "passed": passed,
        "table2_eligible": False,
        "labels_accessed": False,
        "counts_accessed": False,
        "actions_accessed": False,
        "dev84_inputs_accessed": False,
        "test105_inputs_accessed": False,
        "period_head_stream_generations_per_record": 1,
        "source": {
            "policy_sha256": policy_sha256,
            "policy_semantic_sha256": policy_semantic_sha256,
            "runner_sha256": identities["runner"][0],
            **runtime_provenance,
        },
        "inputs": {
            **{f"{name}_sha256": identity[0] for name, identity in identities.items()},
            **{f"{name}_bytes": identity[1] for name, identity in identities.items()},
            "training_config_fingerprint": config.fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
            "train337_sidecar_fingerprint": sidecar.fingerprint,
            "train337_commitment_fingerprint": commitment.fingerprint,
            "train337_identity_sha256": commitment.identity_sha256,
            "train337_video_ids_sha256": _identifier_commitment(
                [record.video_id for record in sidecar.records]
            ),
            "train337_pose_cache_set_sha256": pose_snapshot.fingerprint,
            "frozen_train337_pose_snapshot_file_sha256": base[
                "train337_pose_snapshot_file_sha256"
            ],
            "encoder_provenance": encoder_provenance.to_dict(),
            "sshead_provenance": head_provenance.to_dict(),
            "encoder_completion_binding": encoder_completion,
            "sshead_completion_binding": sshead_completion,
            "synthetic_gate_binding": synthetic_binding,
            "read_only_post_run_identity_verified": True,
        },
        "configuration": {
            "timeline_frames": config.data.frames,
            "candidate_direct_fft_timebase": "dense_resampled",
            "comparison_direct_fft_timebase": "compact_valid",
            "consensus_expert_mode": config.consensus.expert_mode,
            "period_minimum": config.period.minimum,
            "period_maximum": config.period.maximum,
            "two_x_period_minimum": 2 * config.period.minimum,
            "two_x_period_maximum": 2 * config.period.maximum,
            "inference_batch_size": _INFERENCE_BATCH_SIZE,
            "candidate_count": 1,
            "parameter_sweep": False,
            "checkpoint_retraining": False,
        },
        "policy_metric_definitions": policy["train337_gate"]["metric_definitions"],
        "metric_definitions": _definitions(),
        "thresholds": thresholds,
        "metrics": metrics,
        "checks": checks,
        "coverage_strata": evaluation["coverage_strata"],
        "audit_rows": audit_rows,
        "authorization": {
            "synthetic_gate_prerequisite_verified": True,
            "train337_gate_passed": passed,
            "dev84_prediction_authorized": passed,
            "dev84_scoring_authorized": False,
            "test105_evaluation_authorized": False,
        },
        "hardware": hardware,
        "hardware_sha256": sha256_json(hardware),
        "read_only_verification": {
            "all_file_inputs_unchanged": True,
            "train337_pose_snapshot_unchanged": True,
            "model_state_unchanged": True,
            "model_stream_generated_once_per_record": True,
            "checkpoint_retraining_performed": False,
            "optimizer_created": False,
            "pose_cache_write_operations": 0,
        },
    }


def _receipt_path(output: Path) -> Path:
    return Path(str(output) + ".receipt.json")


def _write_new_regular_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with suppress(OSError):
            path.unlink()
        raise


def _write_artifact_and_receipt(
    output: str | Path,
    payload: Mapping[str, Any],
) -> tuple[Path, str]:
    destination = Path(output)
    receipt_path = _receipt_path(destination)
    if destination.exists() or receipt_path.exists():
        raise FileExistsError("train337 report and receipt destinations must both be new")
    artifact = _encoded_json(payload)
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    receipt = {
        "schema_version": 1,
        "artifact_type": _RECEIPT_TYPE,
        "artifact_locator": destination.name,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": len(artifact),
        "artifact_status": payload["status"],
        "candidate_id": payload["candidate_id"],
        "policy_sha256": payload["source"]["policy_sha256"],
        "runner_source_git_sha": payload["source"]["source_git_sha"],
        "runner_sha256": payload["source"]["runner_sha256"],
        "train337_pose_cache_set_sha256": payload["inputs"][
            "train337_pose_cache_set_sha256"
        ],
        "train337_pose_snapshot_file_sha256": payload["inputs"][
            "train337_pose_snapshot_sha256"
        ],
        "synthetic_gate_artifact_sha256": payload["inputs"][
            "synthetic_gate_artifact_sha256"
        ],
        "synthetic_gate_receipt_sha256": payload["inputs"][
            "synthetic_gate_receipt_sha256"
        ],
        "labels_accessed": False,
        "counts_accessed": False,
        "actions_accessed": False,
        "dev84_inputs_accessed": False,
        "test105_inputs_accessed": False,
        "train337_gate_passed": payload["authorization"]["train337_gate_passed"],
        "dev84_prediction_authorized": payload["authorization"][
            "dev84_prediction_authorized"
        ],
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }
    _write_new_regular_file(destination, artifact)
    try:
        _write_new_regular_file(receipt_path, _encoded_json(receipt))
    except BaseException:
        if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == (
            artifact_sha256
        ):
            destination.unlink()
        raise
    return receipt_path, artifact_sha256


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--sshead-checkpoint", type=Path, required=True)
    parser.add_argument("--sshead-progress", type=Path, required=True)
    parser.add_argument("--sshead-completion-receipt", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-progress", type=Path, required=True)
    parser.add_argument("--encoder-completion-receipt", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--train-commitment", type=Path, required=True)
    parser.add_argument("--pose-cache", type=Path, required=True)
    parser.add_argument("--train-pose-snapshot", type=Path, required=True)
    parser.add_argument("--synthetic-gate-artifact", type=Path, required=True)
    parser.add_argument("--synthetic-gate-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--container-environment-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    payload = run_train337_gate(
        arguments.policy,
        arguments.training_config,
        arguments.sshead_checkpoint,
        arguments.sshead_progress,
        arguments.sshead_completion_receipt,
        arguments.encoder_checkpoint,
        arguments.encoder_progress,
        arguments.encoder_completion_receipt,
        arguments.train_sidecar,
        arguments.train_commitment,
        arguments.pose_cache,
        arguments.train_pose_snapshot,
        arguments.synthetic_gate_artifact,
        arguments.synthetic_gate_receipt,
        device=arguments.device,
        source_git_sha=arguments.source_git_sha,
        container_image_id=arguments.container_image_id,
        container_environment_sha256=arguments.container_environment_sha256,
    )
    receipt_path, artifact_sha256 = _write_artifact_and_receipt(
        arguments.output,
        payload,
    )
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "output": str(arguments.output),
                "receipt": str(receipt_path),
                "status": payload["status"],
                "train337_gate_passed": payload["authorization"][
                    "train337_gate_passed"
                ],
                "dev84_prediction_authorized": payload["authorization"][
                    "dev84_prediction_authorized"
                ],
                "dev84_scoring_authorized": False,
                "test105_evaluation_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
