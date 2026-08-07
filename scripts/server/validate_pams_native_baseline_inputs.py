#!/usr/bin/env python3
"""Validate the train337-only inputs for the native Table-2 proxy runner.

This preflight has no training, prediction, target, or label interface. Dev84
and test105 appear only as exact count-free identity sidecars required by the
formal CLI provenance contract. The only pose cache accepted here is an
externally preregistered native train337 recovery artifact with an exact,
passing baseline-authorization receipt. The rejected v4a recovery can never
satisfy this contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pams.config import PAMSConfig, load_config
from pams.data import (
    _reject_json_object_keys_before_deserialization,
    load_pose_cache_set,
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
)

_ARTIFACT_TYPE = "pams_native_table2_baseline_proxy_train337_preflight"
_CLASSIFICATION = "independently_inferred_proxy_not_author_table2_baseline"
_EXPECTED_PROTOCOL = "ucfrep_526"
_POSE_AUTHORIZATION_TYPE = "pams_native_pose_recovery_train337_authorization"
_POSE_AUTHORIZED_CONSUMER = "pams_native_table2_baseline_proxy_train337_encoder"
_REJECTED_POSE_RECOVERY_VERSIONS = frozenset(
    {
        "v4a",
        "official-segment-heavy-missing-retry-full-timeline-v4a",
    }
)
_VERSION_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")
_IMAGE_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_EXPECTED_RECORDS = {"train": 337, "dev": 84, "test": 105}
_EXPECTED_SIDECAR_SHA256 = {
    "train": "f95df0050df21f05bbc9b42d0154470714dde279e04cb8b00d0212bf49057d16",
    "dev": "74b6628679c3d4c9b48f82ef8cf7e3a678e0a8298a5cb245512af9912e4337ba",
    "test": "5de62008db8adc54e6d1ce1df55e25fa0c4a43815ccc1580a010bc1070abafff",
}
_EXPECTED_COMMITMENT_SHA256 = {
    "train": "85d41d2f59872e0900e9058481efbdf6bce91407d4c26bcde0a6547776454e53",
    "dev": "15083106499f6917c8c7c91f6e3985e4938eedbe370e1f40c281dd00884bcef0",
    "test": "634eb578de8b9d96aa33c128ff2cff8e4e4759304297d331f924ae2b354be6c0",
}
_FORBIDDEN_UPSTREAM_KEYS = frozenset(
    {
        "action",
        "actions",
        "count",
        "counts",
        "cycle_boundaries",
        "cycle_boundary",
        "label",
        "labels",
        "metrics_result",
        "nmae",
        "obo",
        "prediction",
        "predictions",
        "target",
        "targets",
    }
)
_POSE_GATE_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "protocol",
        "split",
        "passed",
        "bindings",
        "metrics",
        "criteria",
        "paired_rows",
        "paired_rows_sha256",
        "mount_audit",
    }
)
_POSE_GATE_METRIC_KEYS = frozenset(
    {
        "record_total",
        "v4_zero_video_total",
        "recovered_reference_zero_video_total",
        "reference_usable_to_v4_zero_video_total",
        "source_coverage_mean",
        "source_coverage_p10",
        "source_coverage_p25",
        "source_coverage_median",
        "final_below_pass0_video_total",
        "observed_at_most_8_video_total",
        "longest_run_fraction_p10",
        "longest_run_fraction_median",
        "pose_coordinate_interpolation_true_total",
        "official_segment_timeline_invariant_failure_total",
        "pass0_source_valid_count_mismatch_total",
        "pass0_preservation_failure_total",
        "pass0_shared_coordinate_max_abs_error",
        "native_timeline_invariant_failure_total",
    }
)
_POSE_GATE_CRITERION_KEYS = _POSE_GATE_METRIC_KEYS - {"record_total"}
_POSE_GATE_BINDING_KEYS = frozenset(
    {
        "gate_sha256",
        "sidecar_sha256",
        "commitment_sha256",
        "identity_sha256",
        "reference_ledger_sha256",
        "reference_pose_fingerprint",
        "reference_pose_cache_set_sha256",
        "v4_ledger_sha256",
        "v4_pose_fingerprint",
        "v4_pose_cache_set_sha256",
        "heavy_model_asset_sha256",
    }
)
_POSE_GATE_MOUNT_KEYS = frozenset(
    {
        "train_identity_mounted",
        "train_pose_caches_mounted",
        "source_videos_mounted",
        "dev84_mounted",
        "test105_mounted",
        "targets_mounted",
    }
)
_POSE_GATE_PAIRED_ROW_KEYS = frozenset(
    {
        "video_id_sha256",
        "source_frames",
        "reference_cached_valid_frames",
        "pass0_valid_frames",
        "recovered_valid_frames",
        "v4_cached_valid_frames",
        "final_valid_frames",
        "observed_span_frames",
        "source_coverage",
        "longest_run_fraction",
        "pass0_preserved",
    }
)
_POSE_RUN_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "source_revision",
        "container_image_id",
        "config_file_sha256",
        "config_fingerprint",
        "pose_fingerprint",
        "heavy_model_asset_sha256",
        "temporal_resampling",
        "scope",
        "paired_gate_exit_status",
        "paired_gate_sha256",
    }
)
_POSE_LEDGER_ROOT_KEYS = frozenset(
    {
        "schema_version",
        "artifact_type",
        "input_kind",
        "protocol",
        "split",
        "sidecar_sha256",
        "commitment_file_sha256",
        "full_train_identity_sha256",
        "identity_sha256",
        "selection_sha256",
        "pose_fingerprint",
        "pose_recovery",
        "successful_cache_snapshot",
        "selected",
        "completed",
        "extracted",
        "skipped",
        "failed",
        "recovery_version",
        "caches",
        "failures",
    }
)
_POSE_LEDGER_RECOVERY_KEYS = frozenset(
    {
        "schema_version",
        "preprocessing_revision",
        "recovery_mode",
        "heavy_model_id",
        "heavy_model_asset_sha256",
        "temporal_resampling",
        "pose_coordinate_interpolation",
    }
)
_POSE_SNAPSHOT_KEYS = frozenset(
    {"schema_version", "pose_fingerprint", "fingerprint", "entry_count", "entries"}
)
_POSE_SNAPSHOT_ENTRY_KEYS = frozenset({"video_id", "cache_sha256", "bytes"})
_POSE_LEDGER_CACHE_KEYS = frozenset(
    {
        "video_id",
        "video_path",
        "video_sha256",
        "annotation_sha256",
        "pose_fingerprint",
        "pose_model",
        "source_frames",
        "source_valid_frames",
        "selected_source_frames",
        "fps",
        "cached_frames",
        "cached_valid_frames",
        "clip_start_frame",
        "clip_end_frame",
        "decoded_clip_frames",
        "expected_clip_frames",
        "padded_tail_frames",
        "incomplete_clip_policy",
        "recovery_audit",
        "cache_path",
        "skipped",
    }
)
_POSE_LEDGER_RECOVERY_AUDIT_KEYS = frozenset(
    {
        "schema_version",
        "source_frames",
        "decoded_segment_frames",
        "expected_segment_frames",
        "padded_tail_frames",
        "pass0_valid_frames",
        "pass0_valid_mask_sha256",
        "recovered_valid_frames",
        "final_valid_frames",
        "final_valid_mask_sha256",
        "final_longest_valid_run",
        "observed_span_frames",
        "pass0_observations_preserved",
        "pass0_shared_coordinate_max_abs_error",
        "heavy_full_frame_attempted",
        "heavy_full_frame_detected",
        "roi_retry_eligible",
        "roi_retry_attempted",
        "roi_retry_detected",
        "heavy_model_id",
        "heavy_model_asset_sha256",
        "temporal_resampling",
        "pose_coordinate_interpolation",
        "recovery_mode",
        "heavy_video_frames_observed",
        "heavy_video_valid_frames",
        "heavy_video_pass0_overlap_valid_frames",
        "heavy_video_fill_candidates",
        "heavy_video_valid_mask_sha256",
        "heavy_video_candidate_total",
        "heavy_video_max_candidates_per_frame",
        "heavy_video_num_poses",
        "heavy_video_timestamp_sha256",
    }
)


class NativeBaselineInputError(RuntimeError):
    """Raised when a frozen proxy-training input violates its contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NativeBaselineInputError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: Any, role: str) -> str:
    text = str(value)
    _require(
        len(text) == 64 and all(character in "0123456789abcdef" for character in text),
        f"{role} must be lowercase SHA-256",
    )
    return text


def _reject_non_finite(value: str) -> None:
    raise NativeBaselineInputError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NativeBaselineInputError(f"duplicate JSON field is forbidden: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path, *, role: str) -> tuple[dict[str, Any], str]:
    source = path.resolve(strict=True)
    encoded = source.read_bytes()
    try:
        text = encoded.decode("utf-8")
        _reject_json_object_keys_before_deserialization(
            text,
            forbidden=_FORBIDDEN_UPSTREAM_KEYS,
            document_name=role,
        )
        payload = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_non_finite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NativeBaselineInputError(f"{role} must be strict UTF-8 JSON") from exc
    except ValueError as exc:
        raise NativeBaselineInputError(str(exc)) from exc
    _require(isinstance(payload, dict), f"{role} root must be an object")
    return payload, hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, role: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{role} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, Any],
    expected: set[str] | frozenset[str],
    role: str,
) -> None:
    actual = set(value)
    _require(
        actual == set(expected),
        f"{role} schema mismatch: missing={sorted(set(expected) - actual)}, "
        f"extra={sorted(actual - set(expected))}",
    )


def _validate_proxy_config(
    candidate: PAMSConfig,
    pose_recovery: PAMSConfig,
) -> None:
    """Require the preregistered inferred proxy semantics."""

    _require(candidate.protocol == _EXPECTED_PROTOCOL, "candidate protocol mismatch")
    _require(candidate.seed == 2026, "candidate seed must be 2026")
    _require(
        candidate.data == pose_recovery.data,
        "candidate data block differs from the authorized pose recovery",
    )
    _require(
        candidate.pose == pose_recovery.pose,
        "candidate pose block differs from the authorized pose recovery",
    )
    _require(
        bool(candidate.pose.preprocessing_revision.strip()),
        "candidate pose preprocessing revision is empty",
    )
    _require(
        candidate.pose.preprocessing_revision
        not in _REJECTED_POSE_RECOVERY_VERSIONS,
        "candidate pose preprocessing revision is formally rejected",
    )
    _require(
        candidate.pose.recovery is not None
        and candidate.pose.recovery.temporal_resampling == "none_native_timeline",
        "candidate must retain the authorized native timeline",
    )
    _require(
        candidate.model.position_encoding_mode == "sinusoidal",
        "Table-2 proxy must retain sinusoidal positional encoding",
    )
    _require(
        candidate.period.training_mode == "fixed_period_inferred",
        "Table-2 proxy must use an inferred fixed period",
    )
    _require(
        candidate.period.minimum == 4
        and candidate.period.maximum == 4096
        and candidate.period.maximum_mode == "half_timeline",
        "native period bounds must use [4, half_timeline] with ceiling 4096",
    )
    _require(
        candidate.period.direct_fft_timebase == "dense_resampled",
        "native direct FFT fallback must use the dense-resampled timebase",
    )
    _require(
        candidate.readout.action_curve_source == "embedding_recurrence_carrier",
        "native proxy must use the target-free recurrence-carrier readout",
    )
    _require(candidate.loss.scales == (1.0,), "proxy must use one TCC scale")
    _require(candidate.loss.temperature == 0.1, "proxy temperature must be 0.1")
    _require(
        (
            candidate.period.fixed_period_frames,
            candidate.loss.anchor_stride,
        )
        in {(16, 4), (16, 2), (24, 4)},
        "proxy period/stride pair is outside the frozen A/B/C matrix",
    )
    _require(
        candidate.loss.use_cross_cluster_negatives is False,
        "inferred cross-cluster prototype negatives must be disabled",
    )
    _require(
        candidate.loss.exclude_other_scale_positives_from_denominator is False,
        "single-scale proxy must retain the literal denominator",
    )
    _require(
        candidate.training.epochs == 150
        and candidate.training.effective_batch_size == 32,
        "proxy must use 150 epochs and physical batch 32",
    )
    augmentation = candidate.training.skeleton_augmentation
    _require(augmentation.enabled, "paper-disclosed skeleton augmentation must be enabled")
    _require(
        augmentation.rotation_degrees == (15.0, 15.0, 15.0)
        and augmentation.scale_range == (0.85, 1.15)
        and augmentation.jitter_std == 0.01,
        "inferred skeleton-augmentation magnitudes changed",
    )


def _validate_sidecar_pair(
    sidecar_path: Path,
    commitment_path: Path,
    *,
    split: str,
) -> dict[str, Any]:
    sidecar_sha256 = _sha256_file(sidecar_path.resolve(strict=True))
    commitment_sha256 = _sha256_file(commitment_path.resolve(strict=True))
    _require(
        sidecar_sha256 == _EXPECTED_SIDECAR_SHA256[split],
        f"official {split} identity sidecar SHA mismatch",
    )
    _require(
        commitment_sha256 == _EXPECTED_COMMITMENT_SHA256[split],
        f"official {split} identity commitment SHA mismatch",
    )
    manifest = load_pose_input_manifest(sidecar_path, validate_exact=True)
    commitment = load_pose_input_commitment(commitment_path)
    expected_split = "test" if split == "test" else split
    _require(manifest.protocol == _EXPECTED_PROTOCOL, f"{split} protocol mismatch")
    _require(manifest.split == expected_split, f"{split} sidecar split mismatch")
    _require(
        len(manifest.records) == _EXPECTED_RECORDS[split],
        f"{split} sidecar record count mismatch",
    )
    identity_sha256 = pose_input_identity_sha256(manifest.records)
    _require(commitment.protocol == manifest.protocol, f"{split} commitment protocol mismatch")
    _require(commitment.split == manifest.split, f"{split} commitment split mismatch")
    _require(
        commitment.record_total == len(manifest.records),
        f"{split} commitment record count mismatch",
    )
    _require(
        commitment.identity_sha256 == identity_sha256,
        f"{split} commitment identity mismatch",
    )
    _require(
        commitment.sidecar_sha256 == sidecar_sha256,
        f"{split} commitment sidecar SHA mismatch",
    )
    _require(
        commitment.sidecar_fingerprint == manifest.fingerprint,
        f"{split} commitment fingerprint mismatch",
    )
    return {
        "manifest": manifest,
        "sidecar_sha256": sidecar_sha256,
        "commitment_sha256": commitment_sha256,
        "identity_sha256": identity_sha256,
        "record_total": len(manifest.records),
    }


def _validate_pose_recovery_authorization(
    authorization: Mapping[str, Any],
    gate: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    expected_version: str,
    gate_sha256: str,
    run_receipt_sha256: str,
    config_sha256: str,
    config_fingerprint: str,
    pose_fingerprint: str,
    ledger_sha256: str,
    pose_cache_set_sha256: str,
) -> None:
    """Require an external, fail-closed native-pose authorization."""

    _require(
        _VERSION_PATTERN.fullmatch(expected_version) is not None,
        "pose-recovery version must be a lowercase safe identifier",
    )
    _require(
        expected_version not in _REJECTED_POSE_RECOVERY_VERSIONS,
        f"pose-recovery version is formally rejected: {expected_version}",
    )
    _require(
        "authorization_sha256" not in authorization,
        "pose authorization must not contain a self-reported SHA",
    )
    _require_exact_keys(
        authorization,
        {
            "schema_version",
            "artifact_type",
            "version",
            "authorized",
            "authorized_consumer",
            "classification",
            "protocol",
            "split",
            "record_total",
            "bindings",
            "scope",
        },
        "pose authorization",
    )
    _require(authorization.get("schema_version") == 1, "pose authorization schema mismatch")
    _require(
        authorization.get("artifact_type") == _POSE_AUTHORIZATION_TYPE,
        "unexpected pose authorization artifact type",
    )
    _require(authorization.get("version") == expected_version, "pose version mismatch")
    _require(authorization.get("authorized") is True, "pose recovery is not authorized")
    _require(
        authorization.get("authorized_consumer") == _POSE_AUTHORIZED_CONSUMER,
        "pose recovery is not authorized for this baseline runner",
    )
    _require(
        authorization.get("classification") == "trusted_native_pose_recovery_train337",
        "pose authorization classification mismatch",
    )
    _require(
        authorization.get("protocol") == _EXPECTED_PROTOCOL
        and authorization.get("split") == "train"
        and authorization.get("record_total") == 337,
        "pose authorization is not scoped to train337",
    )
    bindings = _mapping(authorization.get("bindings"), "pose authorization bindings")
    _require_exact_keys(
        bindings,
        {
            "source_revision",
            "container_image_id",
            "config_file_sha256",
            "config_fingerprint",
            "pose_fingerprint",
            "paired_gate_sha256",
            "run_receipt_sha256",
            "ledger_sha256",
            "pose_cache_set_sha256",
        },
        "pose authorization bindings",
    )
    expected_bindings = {
        "config_file_sha256": config_sha256,
        "config_fingerprint": config_fingerprint,
        "pose_fingerprint": pose_fingerprint,
        "paired_gate_sha256": gate_sha256,
        "run_receipt_sha256": run_receipt_sha256,
        "ledger_sha256": ledger_sha256,
        "pose_cache_set_sha256": pose_cache_set_sha256,
    }
    for field, expected in expected_bindings.items():
        _require(
            bindings.get(field) == expected,
            f"pose authorization binding mismatch: {field}",
        )
    source_revision = bindings.get("source_revision")
    container_image_id = bindings.get("container_image_id")
    _require(
        isinstance(source_revision, str)
        and _GIT_SHA_PATTERN.fullmatch(source_revision) is not None,
        "pose authorization source revision is invalid",
    )
    _require(
        isinstance(container_image_id, str)
        and _IMAGE_ID_PATTERN.fullmatch(container_image_id) is not None,
        "pose authorization container image ID is invalid",
    )
    scope = _mapping(authorization.get("scope"), "pose authorization scope")
    expected_scope = {
        "pose_timeline": "native",
        "pose_cache": "train337_only",
        "source_videos_mounted": False,
        "dev84_mounted": False,
        "test105_mounted": False,
        "targets_mounted": False,
        "network_mode": "none",
        "source_export_read_only": True,
    }
    _require_exact_keys(scope, set(expected_scope), "pose authorization scope")
    for field, expected in expected_scope.items():
        _require(scope.get(field) == expected, f"pose authorization scope mismatch: {field}")

    _require_exact_keys(gate, _POSE_GATE_ROOT_KEYS, "pose-recovery paired gate")
    _require(gate.get("schema_version") == 1, "pose-recovery gate schema mismatch")
    _require(
        gate.get("artifact_type")
        == f"pams_pose_recovery_{expected_version}_train337_paired_audit",
        "pose-recovery gate artifact type mismatch",
    )
    _require(gate.get("protocol") == _EXPECTED_PROTOCOL, "pose-recovery gate protocol mismatch")
    _require(gate.get("split") == "train", "pose-recovery gate must be train-only")
    _require(gate.get("passed") is True, "pose-recovery paired gate did not pass")
    gate_bindings = _mapping(gate.get("bindings"), "pose-recovery gate bindings")
    _require_exact_keys(gate_bindings, _POSE_GATE_BINDING_KEYS, "pose-recovery gate bindings")
    _require(
        gate_bindings.get("sidecar_sha256") == _EXPECTED_SIDECAR_SHA256["train"],
        "pose-recovery gate train sidecar mismatch",
    )
    _require(
        gate_bindings.get("commitment_sha256") == _EXPECTED_COMMITMENT_SHA256["train"],
        "pose-recovery gate train commitment mismatch",
    )
    for field in _POSE_GATE_BINDING_KEYS:
        _sha256(gate_bindings.get(field), f"pose-recovery gate binding {field}")
    _require(
        gate_bindings.get("v4_ledger_sha256") == ledger_sha256,
        "pose-recovery gate ledger binding mismatch",
    )
    _require(
        gate_bindings.get("v4_pose_fingerprint") == pose_fingerprint,
        "pose-recovery gate pose fingerprint mismatch",
    )
    _require(
        gate_bindings.get("v4_pose_cache_set_sha256") == pose_cache_set_sha256,
        "pose-recovery gate pose-cache set mismatch",
    )
    metrics = _mapping(gate.get("metrics"), "pose-recovery gate metrics")
    _require_exact_keys(metrics, _POSE_GATE_METRIC_KEYS, "pose-recovery gate metrics")
    _require(metrics.get("record_total") == 337, "pose-recovery gate is not train337")
    criteria = _mapping(gate.get("criteria"), "pose-recovery gate criteria")
    _require_exact_keys(criteria, _POSE_GATE_CRITERION_KEYS, "pose-recovery gate criteria")
    for name, raw_criterion in criteria.items():
        criterion = _mapping(raw_criterion, f"pose-recovery gate criterion {name}")
        _require_exact_keys(
            criterion,
            {"value", "relation", "threshold", "passed"},
            f"pose-recovery gate criterion {name}",
        )
        _require(
            criterion.get("relation") in {"at_least", "at_most"},
            f"pose-recovery gate criterion relation mismatch: {name}",
        )
        value = criterion.get("value")
        threshold = criterion.get("threshold")
        _require(
            not isinstance(value, bool)
            and isinstance(value, int | float)
            and math.isfinite(float(value))
            and not isinstance(threshold, bool)
            and isinstance(threshold, int | float)
            and math.isfinite(float(threshold)),
            f"pose-recovery gate criterion is non-numeric: {name}",
        )
        _require(
            value == metrics.get(name),
            f"pose-recovery gate criterion and metric differ: {name}",
        )
        expected_pass = (
            value >= threshold
            if criterion.get("relation") == "at_least"
            else value <= threshold
        )
        _require(
            criterion.get("passed") is expected_pass is True,
            f"pose-recovery gate criterion did not pass: {name}",
        )
    paired_rows = gate.get("paired_rows")
    _require(
        isinstance(paired_rows, list) and len(paired_rows) == 337,
        "pose-recovery gate paired rows must contain train337",
    )
    for index, raw_row in enumerate(paired_rows):
        row = _mapping(raw_row, f"pose-recovery gate paired row {index}")
        _require_exact_keys(
            row,
            _POSE_GATE_PAIRED_ROW_KEYS,
            f"pose-recovery gate paired row {index}",
        )
    _require(
        gate.get("paired_rows_sha256") == _canonical_sha256(paired_rows),
        "pose-recovery gate paired-row SHA mismatch",
    )
    mounts = _mapping(gate.get("mount_audit"), "pose-recovery gate mount audit")
    _require_exact_keys(mounts, _POSE_GATE_MOUNT_KEYS, "pose-recovery gate mount audit")
    _require(mounts.get("train_identity_mounted") is True, "train identity is absent")
    _require(mounts.get("train_pose_caches_mounted") is True, "train pose is absent")
    for field in (
        "source_videos_mounted",
        "dev84_mounted",
        "test105_mounted",
        "targets_mounted",
    ):
        _require(mounts.get(field) is False, f"pose gate violates firewall: {field}")

    _require_exact_keys(receipt, _POSE_RUN_RECEIPT_KEYS, "pose-recovery run receipt")
    _require(receipt.get("schema_version") == 1, "pose-recovery run receipt schema mismatch")
    _require(
        receipt.get("artifact_type")
        == f"pams_pose_recovery_{expected_version}_train337_run_receipt",
        "pose-recovery run receipt artifact type mismatch",
    )
    _require(
        receipt.get("source_revision") == source_revision,
        "pose-recovery run receipt source revision mismatch",
    )
    _require(
        receipt.get("container_image_id") == container_image_id,
        "pose-recovery run receipt image mismatch",
    )
    _require(
        receipt.get("config_file_sha256") == config_sha256,
        "pose-recovery run receipt config SHA mismatch",
    )
    _require(
        receipt.get("config_fingerprint") == config_fingerprint,
        "pose-recovery run receipt config fingerprint mismatch",
    )
    _require(
        receipt.get("pose_fingerprint") == pose_fingerprint,
        "pose-recovery run receipt pose fingerprint mismatch",
    )
    _require(
        receipt.get("heavy_model_asset_sha256")
        == gate_bindings.get("heavy_model_asset_sha256"),
        "pose-recovery run receipt heavy-model binding mismatch",
    )
    _require(
        receipt.get("temporal_resampling") == "none_native_timeline",
        "pose-recovery run receipt is not native timeline",
    )
    _require(
        receipt.get("scope") == "count-free-train337-pose-input-recovery-only",
        "pose-recovery run receipt scope mismatch",
    )
    _require(receipt.get("paired_gate_exit_status") == 0, "pose-recovery paired gate failed")
    _require(
        receipt.get("paired_gate_sha256") == gate_sha256,
        "pose-recovery run receipt does not bind the paired gate",
    )


def _validate_pose_recovery_ledger(
    ledger: Mapping[str, Any],
    *,
    expected_version: str,
    pose_recovery: PAMSConfig,
    pose_cache_set_sha256: str,
) -> None:
    """Validate the canonical v4c full-train ledger without schema slack."""

    _require_exact_keys(ledger, _POSE_LEDGER_ROOT_KEYS, "pose-recovery train337 ledger")
    _require(ledger.get("schema_version") == 2, "pose ledger schema mismatch")
    _require(
        ledger.get("artifact_type")
        == f"pams_pose_recovery_{expected_version}_train337_ledger",
        "pose ledger artifact type mismatch",
    )
    _require(
        ledger.get("input_kind") == "label_free_train337_hashed_full_set",
        "pose ledger input kind mismatch",
    )
    _require(ledger.get("protocol") == _EXPECTED_PROTOCOL, "pose ledger protocol mismatch")
    _require(ledger.get("split") == "train", "pose ledger split mismatch")
    _require(
        ledger.get("sidecar_sha256") == _EXPECTED_SIDECAR_SHA256["train"],
        "pose ledger train sidecar mismatch",
    )
    _require(
        ledger.get("commitment_file_sha256") == _EXPECTED_COMMITMENT_SHA256["train"],
        "pose ledger train commitment mismatch",
    )
    full_identity = _sha256(ledger.get("full_train_identity_sha256"), "full train identity")
    _require(
        ledger.get("identity_sha256") == full_identity,
        "pose ledger selected identity is not the full train identity",
    )
    _sha256(ledger.get("selection_sha256"), "pose ledger selection")
    _require(
        ledger.get("pose_fingerprint") == pose_recovery.pose_fingerprint,
        "pose ledger fingerprint mismatch",
    )
    _require(ledger.get("recovery_version") == expected_version, "pose ledger version mismatch")
    _require(
        ledger.get("selected") == 337
        and ledger.get("completed") == 337
        and ledger.get("extracted") == 337
        and ledger.get("skipped") == 0
        and ledger.get("failed") == 0,
        "pose ledger is not one complete fresh train337 extraction",
    )
    _require(ledger.get("failures") == [], "pose ledger contains failures")

    recovery = _mapping(ledger.get("pose_recovery"), "pose ledger recovery declaration")
    _require_exact_keys(recovery, _POSE_LEDGER_RECOVERY_KEYS, "pose ledger recovery declaration")
    expected_recovery = pose_recovery.pose.recovery
    _require(expected_recovery is not None, "pose recovery config has no recovery block")
    _require(recovery.get("schema_version") == 1, "pose recovery declaration schema mismatch")
    _require(
        recovery.get("preprocessing_revision")
        == pose_recovery.pose.preprocessing_revision,
        "pose recovery preprocessing revision mismatch",
    )
    _require(
        recovery.get("heavy_model_id") == expected_recovery.heavy_model_id
        and recovery.get("heavy_model_asset_sha256")
        == expected_recovery.heavy_model_asset_sha256
        and recovery.get("temporal_resampling")
        == expected_recovery.temporal_resampling
        and recovery.get("pose_coordinate_interpolation")
        == expected_recovery.pose_coordinate_interpolation,
        "pose ledger recovery configuration mismatch",
    )
    _require(
        isinstance(recovery.get("recovery_mode"), str)
        and bool(str(recovery.get("recovery_mode")).strip()),
        "pose ledger recovery mode is empty",
    )

    snapshot = _mapping(
        ledger.get("successful_cache_snapshot"),
        "pose ledger successful cache snapshot",
    )
    _require_exact_keys(snapshot, _POSE_SNAPSHOT_KEYS, "pose ledger successful cache snapshot")
    _require(snapshot.get("schema_version") == 1, "pose ledger snapshot schema mismatch")
    _require(
        snapshot.get("pose_fingerprint") == pose_recovery.pose_fingerprint,
        "pose ledger snapshot pose fingerprint mismatch",
    )
    _require(
        snapshot.get("fingerprint") == pose_cache_set_sha256,
        "pose ledger snapshot fingerprint mismatch",
    )
    snapshot_entries = snapshot.get("entries")
    _require(
        snapshot.get("entry_count") == 337
        and isinstance(snapshot_entries, list)
        and len(snapshot_entries) == 337,
        "pose ledger snapshot is not train337",
    )
    for index, raw_entry in enumerate(snapshot_entries):
        entry = _mapping(raw_entry, f"pose ledger snapshot entry {index}")
        _require_exact_keys(
            entry,
            _POSE_SNAPSHOT_ENTRY_KEYS,
            f"pose ledger snapshot entry {index}",
        )

    caches = ledger.get("caches")
    _require(isinstance(caches, list) and len(caches) == 337, "pose ledger caches are not train337")
    cache_ids: list[str] = []
    for index, raw_cache in enumerate(caches):
        cache = _mapping(raw_cache, f"pose ledger cache row {index}")
        _require_exact_keys(cache, _POSE_LEDGER_CACHE_KEYS, f"pose ledger cache row {index}")
        _require(
            cache.get("pose_fingerprint") == pose_recovery.pose_fingerprint,
            f"pose ledger cache fingerprint mismatch at row {index}",
        )
        _require(cache.get("skipped") is False, f"pose ledger cache row {index} was skipped")
        video_id = cache.get("video_id")
        _require(
            isinstance(video_id, str) and bool(video_id.strip()),
            f"pose ledger cache video ID is invalid at row {index}",
        )
        cache_ids.append(video_id)
        audit = _mapping(cache.get("recovery_audit"), f"pose ledger recovery audit {index}")
        _require_exact_keys(
            audit,
            _POSE_LEDGER_RECOVERY_AUDIT_KEYS,
            f"pose ledger recovery audit {index}",
        )
        _require(
            audit.get("recovery_mode") == recovery.get("recovery_mode"),
            f"pose ledger recovery mode mismatch at row {index}",
        )
    snapshot_ids = [str(item["video_id"]) for item in snapshot_entries]
    _require(len(set(cache_ids)) == 337, "pose ledger contains duplicate cache video IDs")
    _require(
        sorted(cache_ids) == sorted(snapshot_ids),
        "pose ledger caches and snapshot entries name different videos",
    )


def validate_native_baseline_inputs(
    *,
    candidate_config_path: Path,
    pose_recovery_version: str,
    pose_recovery_config_path: Path,
    train_sidecar_path: Path,
    train_commitment_path: Path,
    dev_sidecar_path: Path,
    dev_commitment_path: Path,
    test_identity_sidecar_path: Path,
    test_identity_commitment_path: Path,
    pose_recovery_authorization_path: Path,
    expected_pose_recovery_authorization_sha256: str,
    pose_recovery_gate_path: Path,
    expected_pose_recovery_gate_sha256: str,
    pose_recovery_run_receipt_path: Path,
    expected_pose_recovery_run_receipt_sha256: str,
    pose_recovery_ledger_path: Path,
    train_pose_cache_dir: Path,
) -> dict[str, Any]:
    """Return a target-free preflight artifact for one exact input set."""

    expected_authorization_sha256 = _sha256(
        expected_pose_recovery_authorization_sha256,
        "expected pose-recovery authorization",
    )
    expected_gate_sha256 = _sha256(
        expected_pose_recovery_gate_sha256,
        "expected pose-recovery gate",
    )
    expected_run_receipt_sha256 = _sha256(
        expected_pose_recovery_run_receipt_sha256,
        "expected pose-recovery run receipt",
    )
    candidate_config_file = candidate_config_path.resolve(strict=True)
    pose_recovery_config_file = pose_recovery_config_path.resolve(strict=True)
    candidate = load_config(candidate_config_file)
    pose_recovery = load_config(pose_recovery_config_file)
    _validate_proxy_config(candidate, pose_recovery)
    candidate_config_sha256 = _sha256_file(candidate_config_file)
    pose_recovery_config_sha256 = _sha256_file(pose_recovery_config_file)

    protocol = {
        "train": _validate_sidecar_pair(
            train_sidecar_path,
            train_commitment_path,
            split="train",
        ),
        "dev": _validate_sidecar_pair(
            dev_sidecar_path,
            dev_commitment_path,
            split="dev",
        ),
        "test": _validate_sidecar_pair(
            test_identity_sidecar_path,
            test_identity_commitment_path,
            split="test",
        ),
    }
    train_manifest = protocol["train"]["manifest"]
    cache_dir = train_pose_cache_dir.resolve(strict=True)
    _require(cache_dir.is_dir(), "authorized train pose cache is not a directory")
    entries = tuple(cache_dir.iterdir())
    _require(len(entries) == 337, "pose cache directory must contain 337 files")
    for entry in entries:
        _require(not entry.is_symlink(), "authorized pose cache contains a symlink")
        _require(entry.is_file() and entry.suffix == ".npz", "unexpected pose-cache entry")
    _, pose_snapshot = load_pose_cache_set(
        train_manifest.records,
        cache_dir=cache_dir,
        pose_fingerprint=candidate.pose_fingerprint,
        materialize_sequences=False,
    )
    _require(len(pose_snapshot.entries) == 337, "pose snapshot is not train337")

    authorization, authorization_sha256 = _load_json(
        pose_recovery_authorization_path,
        role="pose-recovery authorization",
    )
    _require(
        authorization_sha256 == expected_authorization_sha256,
        "pose-recovery authorization SHA mismatch",
    )
    gate, gate_sha256 = _load_json(
        pose_recovery_gate_path,
        role="pose-recovery paired gate",
    )
    _require(gate_sha256 == expected_gate_sha256, "pose-recovery gate SHA mismatch")
    run_receipt, run_receipt_sha256 = _load_json(
        pose_recovery_run_receipt_path,
        role="pose-recovery run receipt",
    )
    _require(
        run_receipt_sha256 == expected_run_receipt_sha256,
        "pose-recovery run receipt SHA mismatch",
    )
    ledger_path = pose_recovery_ledger_path.resolve(strict=True)
    pose_recovery_ledger_sha256 = _sha256_file(ledger_path)
    ledger, _ = _load_json(ledger_path, role="pose-recovery train337 ledger")
    _validate_pose_recovery_ledger(
        ledger,
        expected_version=pose_recovery_version,
        pose_recovery=pose_recovery,
        pose_cache_set_sha256=pose_snapshot.fingerprint,
    )
    _validate_pose_recovery_authorization(
        authorization,
        gate,
        run_receipt,
        expected_version=pose_recovery_version,
        gate_sha256=gate_sha256,
        run_receipt_sha256=run_receipt_sha256,
        config_sha256=pose_recovery_config_sha256,
        config_fingerprint=pose_recovery.fingerprint,
        pose_fingerprint=candidate.pose_fingerprint,
        ledger_sha256=pose_recovery_ledger_sha256,
        pose_cache_set_sha256=pose_snapshot.fingerprint,
    )

    sidecars = {
        split: {
            key: value
            for key, value in row.items()
            if key != "manifest"
        }
        for split, row in protocol.items()
    }
    candidate_pair = (
        candidate.period.fixed_period_frames,
        candidate.loss.anchor_stride,
    )
    candidate_id = {(16, 4): "A", (16, 2): "B", (24, 4): "C"}[candidate_pair]
    return {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "classification": _CLASSIFICATION,
        "paper_table2_value_claim_eligible": False,
        "protocol": _EXPECTED_PROTOCOL,
        "passed": True,
        "candidate": {
            "candidate_id": candidate_id,
            "config_file_sha256": candidate_config_sha256,
            "config_fingerprint": candidate.fingerprint,
            "pose_fingerprint": candidate.pose_fingerprint,
            "period_training_mode": candidate.period.training_mode,
            "fixed_period_frames": candidate.period.fixed_period_frames,
            "period_maximum_ceiling_frames": candidate.period.maximum,
            "period_maximum_mode": candidate.period.maximum_mode,
            "direct_fft_timebase": candidate.period.direct_fft_timebase,
            "action_curve_source": candidate.readout.action_curve_source,
            "tcc_scales": list(candidate.loss.scales),
            "anchor_stride": candidate.loss.anchor_stride,
            "use_cross_cluster_negatives": candidate.loss.use_cross_cluster_negatives,
            "other_video_frame_negatives_retained_by_loss": True,
            "encoder_epochs": candidate.training.epochs,
            "physical_batch_size": candidate.training.effective_batch_size,
        },
        "pose_recovery": {
            "version": pose_recovery_version,
            "authorization_sha256": authorization_sha256,
            "source_revision": _mapping(
                authorization.get("bindings"),
                "pose authorization bindings",
            )["source_revision"],
            "container_image_id": _mapping(
                authorization.get("bindings"),
                "pose authorization bindings",
            )["container_image_id"],
            "config_file_sha256": pose_recovery_config_sha256,
            "config_fingerprint": pose_recovery.fingerprint,
            "paired_gate_sha256": gate_sha256,
            "run_receipt_sha256": run_receipt_sha256,
            "ledger_sha256": pose_recovery_ledger_sha256,
            "pose_cache_set_sha256": pose_snapshot.fingerprint,
            "pose_cache_entry_total": len(pose_snapshot.entries),
            "native_timeline": True,
        },
        "identity_sidecars": sidecars,
        "mount_policy": {
            "train337_pose_cache": True,
            "dev84_pose_cache": False,
            "test105_pose_cache": False,
            "train_identity_sidecar": True,
            "dev_identity_sidecar_for_cli_provenance_only": True,
            "test_identity_sidecar_for_cli_provenance_only": True,
            "targets": False,
        },
        "authorization": {
            "encoder_training": "train337_only",
            "undisclosed_period_head_training": False,
            "sshead_training": False,
            "dev_prediction_authorized": False,
            "dev_scoring_authorized": False,
            "test_prediction_authorized": False,
            "test_scoring_authorized": False,
        },
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--pose-recovery-version", required=True)
    parser.add_argument("--pose-recovery-config", type=Path, required=True)
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--train-commitment", type=Path, required=True)
    parser.add_argument("--dev-identity-sidecar", type=Path, required=True)
    parser.add_argument("--dev-identity-commitment", type=Path, required=True)
    parser.add_argument("--test-identity-sidecar", type=Path, required=True)
    parser.add_argument("--test-identity-commitment", type=Path, required=True)
    parser.add_argument("--pose-recovery-authorization", type=Path, required=True)
    parser.add_argument(
        "--expected-pose-recovery-authorization-sha256",
        required=True,
    )
    parser.add_argument("--pose-recovery-paired-gate", type=Path, required=True)
    parser.add_argument("--expected-pose-recovery-paired-gate-sha256", required=True)
    parser.add_argument("--pose-recovery-run-receipt", type=Path, required=True)
    parser.add_argument("--expected-pose-recovery-run-receipt-sha256", required=True)
    parser.add_argument("--pose-recovery-ledger", type=Path, required=True)
    parser.add_argument("--train-pose-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    try:
        payload = validate_native_baseline_inputs(
            candidate_config_path=arguments.candidate_config,
            pose_recovery_version=arguments.pose_recovery_version,
            pose_recovery_config_path=arguments.pose_recovery_config,
            train_sidecar_path=arguments.train_sidecar,
            train_commitment_path=arguments.train_commitment,
            dev_sidecar_path=arguments.dev_identity_sidecar,
            dev_commitment_path=arguments.dev_identity_commitment,
            test_identity_sidecar_path=arguments.test_identity_sidecar,
            test_identity_commitment_path=arguments.test_identity_commitment,
            pose_recovery_authorization_path=(
                arguments.pose_recovery_authorization
            ),
            expected_pose_recovery_authorization_sha256=(
                arguments.expected_pose_recovery_authorization_sha256
            ),
            pose_recovery_gate_path=arguments.pose_recovery_paired_gate,
            expected_pose_recovery_gate_sha256=(
                arguments.expected_pose_recovery_paired_gate_sha256
            ),
            pose_recovery_run_receipt_path=arguments.pose_recovery_run_receipt,
            expected_pose_recovery_run_receipt_sha256=(
                arguments.expected_pose_recovery_run_receipt_sha256
            ),
            pose_recovery_ledger_path=arguments.pose_recovery_ledger,
            train_pose_cache_dir=arguments.train_pose_cache,
        )
        _write_exclusive(arguments.output, payload)
    except (OSError, TypeError, ValueError, NativeBaselineInputError) as exc:
        print(f"native baseline input preflight failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
