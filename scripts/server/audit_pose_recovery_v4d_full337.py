#!/usr/bin/env python3
"""Run the immutable label-free full337 pose, continuity, periodicity and risk gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pams.data import (
    load_pose_cache_set,
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
    validate_pose_input_binding,
)
from pams.period import estimate_period_from_embedding_velocity_vectors

from pose_recovery_v4d_full337_contract import (
    CONTAINER_IMAGE_ID,
    MODEL_ASSET_SHA256,
    TRAIN337_COMMITMENT_SHA256,
    TRAIN337_IDENTITY_SHA256,
    TRAIN337_SIDECAR_SHA256,
    V4A_POSE_FINGERPRINT,
    V4D_CONFIG_FILE_SHA256,
    V4D_CONFIG_FINGERPRINT,
    V4D_POSE_FINGERPRINT,
    Full337ContractError,
    canonical_sha256,
    finite,
    integer,
    load_strict_json,
    mapping,
    require,
    sha256_file,
    validate_full337_gate,
    validate_same39_authorization_v2,
    validate_v4a_inputs,
    write_json_exclusive,
)

FULL_LEDGER_KEYS = frozenset(
    {
        "artifact_type",
        "baseline_training_authorized",
        "caches",
        "commitment_file_sha256",
        "completed",
        "config_file_sha256",
        "config_fingerprint",
        "container_image_id",
        "extracted_fresh",
        "failed",
        "failures",
        "gate_sha256",
        "identity_sha256",
        "input_kind",
        "model_asset_sha256",
        "pose_fingerprint",
        "pose_recovery",
        "protocol",
        "reused_same39",
        "same39_authorization",
        "schema_version",
        "selected",
        "sidecar_sha256",
        "source_revision",
        "split",
        "successful_cache_snapshot",
        "training_authorization_requires",
        "v4a",
    }
)
FULL_CACHE_ROW_KEYS = frozenset(
    {
        "annotation_sha256",
        "base_v4a_cache_sha256",
        "cache_bytes",
        "cache_origin",
        "cache_path",
        "cache_sha256",
        "cached_frames",
        "cached_valid_frames",
        "clip_end_frame",
        "clip_start_frame",
        "decoded_clip_frames",
        "expected_clip_frames",
        "fps",
        "incomplete_clip_policy",
        "padded_tail_frames",
        "pose_fingerprint",
        "pose_model",
        "recovery_audit",
        "selected_source_frames",
        "skipped",
        "source_frames",
        "source_valid_frames",
        "video_id",
        "video_path",
        "video_sha256",
    }
)
RECOVERY_AUDIT_KEYS = frozenset(
    {
        "base_v4a_coordinate_sha256",
        "base_v4a_observations_preserved",
        "base_v4a_shared_coordinate_max_abs_error",
        "base_v4a_valid_frames",
        "base_v4a_valid_mask_sha256",
        "decoded_segment_frames",
        "decoder_fps_matches_v4a",
        "decoder_frame_bytes_sha256",
        "expected_segment_frames",
        "final_base_v4a_coordinate_sha256",
        "final_longest_valid_run",
        "final_valid_frames",
        "final_valid_mask_sha256",
        "heavy_full_frame_attempted",
        "heavy_full_frame_detected",
        "heavy_model_asset_sha256",
        "heavy_model_id",
        "keypointrcnn_association_coordinate_space",
        "keypointrcnn_box_score_threshold",
        "keypointrcnn_candidate_total",
        "keypointrcnn_coco_to_mediapipe_mapping",
        "keypointrcnn_fill_candidates",
        "keypointrcnn_fill_coordinate_space",
        "keypointrcnn_frames_attempted",
        "keypointrcnn_frames_observed",
        "keypointrcnn_frames_with_candidates",
        "keypointrcnn_input_scale_policy",
        "keypointrcnn_keypoint_logit_threshold",
        "keypointrcnn_max_candidates_per_frame",
        "keypointrcnn_maximum_candidates_per_frame",
        "keypointrcnn_minimum_confident_keypoints",
        "keypointrcnn_missing_frames_eligible",
        "keypointrcnn_missing_frames_with_candidates",
        "keypointrcnn_model_asset_sha256",
        "keypointrcnn_model_id",
        "keypointrcnn_parameter_count",
        "keypointrcnn_runtime_receipt",
        "keypointrcnn_state_dict_keys",
        "keypointrcnn_v4a_anchor_distance_maximum",
        "keypointrcnn_v4a_anchor_distance_mean",
        "keypointrcnn_v4a_anchor_distance_observed_maximum",
        "keypointrcnn_v4a_anchor_distance_sha256",
        "keypointrcnn_v4a_anchor_frames",
        "keypointrcnn_v4a_anchor_policy",
        "keypointrcnn_v4a_anchor_rejected_frames",
        "keypointrcnn_v4a_anchor_unusable_shape_frames",
        "keypointrcnn_z_coordinate_policy",
        "observed_span_frames",
        "padded_tail_frames",
        "pass0_observations_preserved",
        "pass0_shared_coordinate_max_abs_error",
        "pass0_valid_frames",
        "pass0_valid_mask_sha256",
        "pose_coordinate_interpolation",
        "recovered_valid_frames",
        "recovery_mode",
        "roi_retry_attempted",
        "roi_retry_detected",
        "roi_retry_eligible",
        "schema_version",
        "source_frames",
        "source_video_sha256_after",
        "source_video_sha256_before",
        "source_video_stat_stable",
        "temporal_resampling",
    }
)


def _percentile(values: Sequence[float], quantile: float) -> float:
    require(bool(values), "percentile requires at least one value")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile, method="linear"))


def _criterion(value: int | float, *, relation: str, threshold: int | float) -> dict[str, Any]:
    passed = value >= threshold if relation == "at_least" else value <= threshold
    return {
        "value": value,
        "relation": relation,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in np.asarray(mask, dtype=np.bool_):
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def _mask_transition_rate(mask: np.ndarray) -> float:
    values = np.asarray(mask, dtype=np.bool_)
    if values.size < 2:
        return 0.0
    return float(np.count_nonzero(values[1:] != values[:-1]) / (values.size - 1))


def _base_fill_boundary_jumps(
    final_xyz: np.ndarray,
    final_mask: np.ndarray,
    base_mask: np.ndarray,
) -> tuple[float, ...]:
    final_valid = np.asarray(final_mask, dtype=np.bool_)
    base_valid = np.asarray(base_mask, dtype=np.bool_)
    jumps: list[float] = []
    for index in range(1, final_valid.size):
        if not (final_valid[index - 1] and final_valid[index]):
            continue
        if base_valid[index - 1] == base_valid[index]:
            continue
        difference = np.asarray(final_xyz[index, :, :2] - final_xyz[index - 1, :, :2])
        jumps.append(float(np.sqrt(np.mean(np.square(difference, dtype=np.float64)))))
    return tuple(jumps)


def _period_evidence(xyz: np.ndarray, valid_mask: np.ndarray) -> tuple[float, float]:
    features = torch.from_numpy(np.asarray(xyz, dtype=np.float32).reshape(xyz.shape[0], -1))
    mask = torch.from_numpy(np.asarray(valid_mask, dtype=np.bool_))
    periods, confidences = estimate_period_from_embedding_velocity_vectors(
        features,
        minimum=4,
        maximum=128,
        valid_mask=mask,
    )
    return float(periods[0]), float(confidences[0])


def audit_full337(
    *,
    source_revision: str,
    container_image_id: str,
    expected_gate_sha256: str,
    expected_extraction_authorization_sha256: str,
    expected_same39_failure_receipt_sha256: str,
    gate_path: Path,
    train_input_path: Path,
    train_commitment_path: Path,
    v4a_cache_dir: Path,
    v4a_ledger_path: Path,
    v4a_paired_gate_path: Path,
    extraction_authorization_path: Path,
    same39_failure_receipt_path: Path,
    same39_audit_path: Path,
    same39_ledger_path: Path,
    same39_selection_path: Path,
    same39_cache_dir: Path,
    candidate_cache_dir: Path,
    candidate_ledger_path: Path,
) -> dict[str, Any]:
    require(
        os.environ.get("PAMS_CONTAINER_SOURCE_REVISION") == source_revision,
        "audit source-revision environment mismatch",
    )
    require(
        os.environ.get("PAMS_CONTAINER_IMAGE_ID") == container_image_id == CONTAINER_IMAGE_ID,
        "audit image-ID binding mismatch",
    )
    gate, gate_sha256 = validate_full337_gate(
        gate_path,
        expected_sha256=expected_gate_sha256,
    )
    manifest = load_pose_input_manifest(train_input_path, validate_exact=True)
    commitment = load_pose_input_commitment(train_commitment_path)
    sidecar_sha256 = sha256_file(train_input_path)
    commitment_sha256 = sha256_file(train_commitment_path)
    validate_pose_input_binding(manifest, commitment, sidecar_sha256=sidecar_sha256)
    require(manifest.split == "train" and len(manifest.records) == 337, "audit input is not train337")
    require(sidecar_sha256 == TRAIN337_SIDECAR_SHA256, "audit sidecar SHA mismatch")
    require(commitment_sha256 == TRAIN337_COMMITMENT_SHA256, "audit commitment SHA mismatch")
    require(
        pose_input_identity_sha256(manifest.records) == TRAIN337_IDENTITY_SHA256,
        "audit train337 identity mismatch",
    )
    v4a = validate_v4a_inputs(
        manifest.records,
        cache_dir=v4a_cache_dir,
        ledger_path=v4a_ledger_path,
        paired_gate_path=v4a_paired_gate_path,
    )
    same39 = validate_same39_authorization_v2(
        manifest.records,
        expected_source_revision=source_revision,
        expected_gate_sha256=gate_sha256,
        expected_authorization_receipt_sha256=expected_extraction_authorization_sha256,
        authorization_receipt_path=extraction_authorization_path,
        expected_failure_receipt_sha256=expected_same39_failure_receipt_sha256,
        failure_receipt_path=same39_failure_receipt_path,
        audit_path=same39_audit_path,
        ledger_path=same39_ledger_path,
        selection_path=same39_selection_path,
        cache_dir=same39_cache_dir,
    )
    ledger, ledger_sha256 = load_strict_json(candidate_ledger_path, role="full337 ledger")
    require(set(ledger) == FULL_LEDGER_KEYS, "full337 ledger schema mismatch")
    require(
        ledger.get("schema_version") == 1
        and ledger.get("artifact_type") == "pams_pose_recovery_v4d_train337_full_ledger"
        and ledger.get("source_revision") == source_revision
        and ledger.get("container_image_id") == container_image_id
        and ledger.get("gate_sha256") == gate_sha256
        and ledger.get("config_file_sha256") == V4D_CONFIG_FILE_SHA256
        and ledger.get("config_fingerprint") == V4D_CONFIG_FINGERPRINT
        and ledger.get("pose_fingerprint") == V4D_POSE_FINGERPRINT
        and ledger.get("model_asset_sha256") == MODEL_ASSET_SHA256
        and ledger.get("input_kind") == "label_free_train337_sidecar"
        and ledger.get("protocol") == "ucfrep_526"
        and ledger.get("split") == "train"
        and ledger.get("sidecar_sha256") == TRAIN337_SIDECAR_SHA256
        and ledger.get("commitment_file_sha256") == TRAIN337_COMMITMENT_SHA256
        and ledger.get("identity_sha256") == TRAIN337_IDENTITY_SHA256,
        "full337 ledger immutable binding mismatch",
    )
    require(
        ledger.get("selected") == ledger.get("completed") == 337
        and ledger.get("extracted_fresh") == 298
        and ledger.get("reused_same39") == 39
        and ledger.get("failed") == 0
        and ledger.get("failures") == []
        and ledger.get("baseline_training_authorized") is False,
        "full337 ledger is not an exactly-once successful extraction",
    )
    same_binding = mapping(ledger.get("same39_authorization"), "full ledger same39 binding")
    require(
        dict(same_binding)
        == {
            "authorization_protocol_version": 2,
            "authorization_scope": "full337_pose_extraction_only",
            "source_revision": source_revision,
            "authorization_receipt_sha256": same39.authorization_receipt_sha256,
            "failure_receipt_sha256": same39.evidence.failure_receipt_sha256,
            "audit_sha256": same39.evidence.audit_sha256,
            "ledger_sha256": same39.evidence.ledger_sha256,
            "selection_sha256": same39.evidence.selection_sha256,
            "pose_fingerprint": V4D_POSE_FINGERPRINT,
            "cache_set_sha256": same39.snapshot.fingerprint,
            "cache_entry_count": 39,
            "full337_pose_extraction_authorized": True,
            "baseline_training_authorized": False,
        },
        "full ledger same39 authorization mismatch",
    )
    candidate_sequences, candidate_snapshot = load_pose_cache_set(
        manifest.records,
        cache_dir=candidate_cache_dir.resolve(strict=True),
        pose_fingerprint=V4D_POSE_FINGERPRINT,
        materialize_sequences=True,
    )
    base_sequences, base_snapshot = load_pose_cache_set(
        manifest.records,
        cache_dir=v4a_cache_dir.resolve(strict=True),
        pose_fingerprint=V4A_POSE_FINGERPRINT,
        materialize_sequences=True,
    )
    require(base_snapshot.fingerprint == v4a.snapshot.fingerprint, "v4a changed during audit")
    require(
        ledger.get("successful_cache_snapshot") == candidate_snapshot.to_dict(),
        "full337 cache snapshot differs from exact bytes",
    )
    full_receipts = {entry.video_id: entry for entry in candidate_snapshot.entries}
    base_receipts = {entry.video_id: entry for entry in base_snapshot.entries}
    same39_receipts = {entry.video_id: entry for entry in same39.snapshot.entries}
    rows = ledger.get("caches")
    require(isinstance(rows, list) and len(rows) == 337, "full337 cache rows mismatch")
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for value in rows:
        row = mapping(value, "full337 cache row")
        require(set(row) == FULL_CACHE_ROW_KEYS, "full337 cache-row schema mismatch")
        identifier = str(row.get("video_id", ""))
        require(identifier and identifier not in rows_by_id, "duplicate full337 cache row")
        rows_by_id[identifier] = row
    require(set(rows_by_id) == {record.video_id for record in manifest.records}, "row identity mismatch")

    paired_rows = v4a.paired_gate.get("paired_rows")
    require(isinstance(paired_rows, list) and len(paired_rows) == 337, "paired rows missing")
    require(
        canonical_sha256(paired_rows) == v4a.paired_gate.get("paired_rows_sha256"),
        "paired rows canonical SHA mismatch",
    )
    paired_by_hash = {
        str(mapping(row, "paired row").get("video_id_sha256")): mapping(row, "paired row")
        for row in paired_rows
    }
    candidate_by_id = {sequence.video_id: sequence for sequence in candidate_sequences}
    base_by_id = {sequence.video_id: sequence for sequence in base_sequences}

    coverages: list[float] = []
    longest_fractions: list[float] = []
    transition_rates: list[float] = []
    boundary_jumps: list[float] = []
    period_confidences: list[float] = []
    periodic_track_flags: list[bool] = []
    contiguous_cycle_flags: list[bool] = []
    diagnostic_rows: list[dict[str, Any]] = []
    invariant_failures = 0
    bitwise_mismatches = 0
    same39_mismatches = 0
    nonfinite_metrics = 0
    recovered_reference_zero = 0
    reference_usable_to_zero = 0
    unanchored_filled = 0
    unanchored_filled_low_periodicity = 0
    for record in manifest.records:
        row = rows_by_id[record.video_id]
        final = candidate_by_id[record.video_id]
        base = base_by_id[record.video_id]
        receipt = full_receipts[record.video_id]
        base_receipt = base_receipts[record.video_id]
        recovery = mapping(row.get("recovery_audit"), "v4d recovery audit")
        if set(recovery) != RECOVERY_AUDIT_KEYS:
            invariant_failures += 1
        source_frames = final.num_frames
        final_valid = int(np.count_nonzero(final.valid_mask))
        base_valid = int(np.count_nonzero(base.valid_mask))
        longest = _longest_run(final.valid_mask)
        row_ok = (
            row.get("pose_fingerprint") == V4D_POSE_FINGERPRINT
            and row.get("video_sha256") == record.video_sha256
            and row.get("annotation_sha256") == record.annotation_sha256
            and row.get("clip_start_frame") == record.clip_start_frame
            and row.get("clip_end_frame") == record.clip_end_frame
            and integer(row.get("cached_frames"), "cached frames") == source_frames
            and integer(row.get("cached_valid_frames"), "cached valid") == final_valid
            and row.get("cache_sha256") == receipt.cache_sha256
            and integer(row.get("cache_bytes"), "cache bytes") == receipt.bytes
            and row.get("base_v4a_cache_sha256") == base_receipt.cache_sha256
            and recovery.get("source_frames") == source_frames
            and recovery.get("expected_segment_frames") == source_frames
            and recovery.get("final_valid_frames") == final_valid
            and recovery.get("base_v4a_valid_frames") == base_valid
            and recovery.get("final_longest_valid_run") == longest
            and recovery.get("base_v4a_observations_preserved") is True
            and finite(
                recovery.get("base_v4a_shared_coordinate_max_abs_error"),
                "base shared coordinate error",
            )
            == 0.0
            and recovery.get("keypointrcnn_model_asset_sha256") == MODEL_ASSET_SHA256
            and recovery.get("keypointrcnn_state_dict_keys") == 313
            and recovery.get("keypointrcnn_parameter_count") == 59137258
            and recovery.get("pose_coordinate_interpolation") is False
            and recovery.get("temporal_resampling") == "none_native_timeline"
            and recovery.get("decoder_fps_matches_v4a") is True
            and recovery.get("source_video_stat_stable") is True
            and recovery.get("source_video_sha256_before")
            == recovery.get("source_video_sha256_after")
            == record.video_sha256
        )
        if not row_ok:
            invariant_failures += 1
        if not (
            np.all(final.valid_mask[base.valid_mask])
            and np.array_equal(final.xyz[base.valid_mask], base.xyz[base.valid_mask])
        ):
            bitwise_mismatches += 1
        expected_origin = (
            "bytewise_reused_frozen_same39"
            if record.video_id in same39_receipts
            else "fresh_v4d_full337_remainder"
        )
        if row.get("cache_origin") != expected_origin:
            invariant_failures += 1
        if record.video_id in same39_receipts:
            pilot_receipt = same39_receipts[record.video_id]
            if (receipt.cache_sha256, receipt.bytes) != (
                pilot_receipt.cache_sha256,
                pilot_receipt.bytes,
            ):
                same39_mismatches += 1

        coverage = final_valid / source_frames
        longest_fraction = longest / source_frames
        transition_rate = _mask_transition_rate(final.valid_mask)
        raw_jumps = _base_fill_boundary_jumps(final.xyz, final.valid_mask, base.valid_mask)
        raw_period, raw_confidence = _period_evidence(final.xyz, final.valid_mask)
        for value in (
            coverage,
            longest_fraction,
            transition_rate,
            raw_period,
            raw_confidence,
            *raw_jumps,
        ):
            if not math.isfinite(float(value)):
                nonfinite_metrics += 1
        if raw_period <= 0.0 or not 0.0 <= raw_confidence <= 1.0:
            invariant_failures += 1
        if any(value < 0.0 for value in raw_jumps):
            invariant_failures += 1
        period = raw_period if math.isfinite(raw_period) and raw_period > 0.0 else 128.0
        confidence = (
            raw_confidence
            if math.isfinite(raw_confidence) and 0.0 <= raw_confidence <= 1.0
            else 0.0
        )
        jumps = tuple(value if math.isfinite(value) and value >= 0.0 else 1.0 for value in raw_jumps)
        periodic = confidence >= 0.03 and final_valid >= 2.0 * period
        contiguous = confidence >= 0.03 and longest >= period
        anchor_frames = integer(
            recovery.get("keypointrcnn_v4a_anchor_frames"),
            "anchor frames",
        )
        filled = integer(recovery.get("keypointrcnn_fill_candidates"), "fill candidates")
        unanchored = anchor_frames == 0 and filled > 0
        if unanchored:
            unanchored_filled += 1
            if not periodic:
                unanchored_filled_low_periodicity += 1
        video_hash = hashlib.sha256(record.video_id.encode("utf-8")).hexdigest()
        paired = paired_by_hash[video_hash]
        reference_valid = integer(
            paired.get("reference_cached_valid_frames"),
            "reference cached valid",
        )
        recovered_reference_zero += int(reference_valid == 0 and final_valid > 0)
        reference_usable_to_zero += int(reference_valid > 0 and final_valid == 0)
        coverages.append(coverage)
        longest_fractions.append(longest_fraction)
        transition_rates.append(transition_rate)
        boundary_jumps.extend(jumps)
        period_confidences.append(confidence)
        periodic_track_flags.append(periodic)
        contiguous_cycle_flags.append(contiguous)
        diagnostic_rows.append(
            {
                "video_id_sha256": video_hash,
                "source_frames": source_frames,
                "base_v4a_valid_frames": base_valid,
                "final_valid_frames": final_valid,
                "source_coverage": coverage,
                "longest_run_fraction": longest_fraction,
                "mask_transition_rate": transition_rate,
                "base_fill_boundary_jump_maximum": max(jumps, default=0.0),
                "period_frames": period,
                "period_confidence": confidence,
                "periodic_track": periodic,
                "contiguous_cycle_supported": contiguous,
                "v4a_anchor_frames": anchor_frames,
                "unanchored_filled": unanchored,
                "cache_origin": expected_origin,
            }
        )

    coverage_metrics: dict[str, Any] = {
        "record_total": 337,
        "v4_zero_video_total": sum(value == 0.0 for value in coverages),
        "recovered_reference_zero_video_total": recovered_reference_zero,
        "reference_usable_to_v4_zero_video_total": reference_usable_to_zero,
        "source_coverage_mean": float(np.mean(coverages)),
        "source_coverage_p10": _percentile(coverages, 0.10),
        "source_coverage_p25": _percentile(coverages, 0.25),
        "source_coverage_median": _percentile(coverages, 0.50),
        "observed_at_most_8_video_total": sum(
            row["final_valid_frames"] <= 8 for row in diagnostic_rows
        ),
        "longest_run_fraction_p10": _percentile(longest_fractions, 0.10),
        "longest_run_fraction_median": _percentile(longest_fractions, 0.50),
    }
    track_metrics: dict[str, Any] = {
        "mask_transition_rate_p90": _percentile(transition_rates, 0.90),
        "base_fill_boundary_shape_jump_p95": _percentile(boundary_jumps, 0.95)
        if boundary_jumps
        else 0.0,
        "base_fill_boundary_observation_total": len(boundary_jumps),
        "contiguous_cycle_supported_video_fraction": float(np.mean(contiguous_cycle_flags)),
    }
    periodicity_metrics: dict[str, Any] = {
        "period_confidence_p25": _percentile(period_confidences, 0.25),
        "period_confidence_median": _percentile(period_confidences, 0.50),
        "periodic_track_video_fraction": float(np.mean(periodic_track_flags)),
        "periodic_track_confidence_minimum": 0.03,
    }
    anchor_metrics: dict[str, Any] = {
        "unanchored_filled_video_total": unanchored_filled,
        "unanchored_filled_video_fraction": unanchored_filled / 337,
        "unanchored_filled_low_periodicity_video_total": unanchored_filled_low_periodicity,
    }
    integrity_metrics: dict[str, Any] = {
        "invariant_failure_total": invariant_failures,
        "v4a_bitwise_mismatch_total": bitwise_mismatches,
        "same39_bytewise_mismatch_total": same39_mismatches,
        "nonfinite_metric_total": nonfinite_metrics,
        "exactly_once_record_total": len(rows_by_id),
        "fresh_remainder_total": sum(
            row.get("cache_origin") == "fresh_v4d_full337_remainder" for row in rows_by_id.values()
        ),
        "reused_same39_total": sum(
            row.get("cache_origin") == "bytewise_reused_frozen_same39"
            for row in rows_by_id.values()
        ),
    }
    coverage_thresholds = mapping(gate.get("coverage_thresholds"), "coverage thresholds")
    coverage_criteria = {
        "v4_zero_video_total": _criterion(
            coverage_metrics["v4_zero_video_total"],
            relation="at_most",
            threshold=int(coverage_thresholds["v4_zero_video_maximum"]),
        ),
        "recovered_reference_zero_video_total": _criterion(
            recovered_reference_zero,
            relation="at_least",
            threshold=int(coverage_thresholds["recovered_reference_zero_video_minimum"]),
        ),
        "reference_usable_to_v4_zero_video_total": _criterion(
            reference_usable_to_zero,
            relation="at_most",
            threshold=int(coverage_thresholds["reference_usable_to_v4_zero_video_maximum"]),
        ),
        "source_coverage_mean": _criterion(
            coverage_metrics["source_coverage_mean"],
            relation="at_least",
            threshold=float(coverage_thresholds["source_coverage_mean_minimum"]),
        ),
        "source_coverage_p10": _criterion(
            coverage_metrics["source_coverage_p10"],
            relation="at_least",
            threshold=float(coverage_thresholds["source_coverage_p10_minimum"]),
        ),
        "source_coverage_p25": _criterion(
            coverage_metrics["source_coverage_p25"],
            relation="at_least",
            threshold=float(coverage_thresholds["source_coverage_p25_minimum"]),
        ),
        "source_coverage_median": _criterion(
            coverage_metrics["source_coverage_median"],
            relation="at_least",
            threshold=float(coverage_thresholds["source_coverage_median_minimum"]),
        ),
        "observed_at_most_8_video_total": _criterion(
            coverage_metrics["observed_at_most_8_video_total"],
            relation="at_most",
            threshold=int(coverage_thresholds["observed_at_most_8_video_maximum"]),
        ),
        "longest_run_fraction_p10": _criterion(
            coverage_metrics["longest_run_fraction_p10"],
            relation="at_least",
            threshold=float(coverage_thresholds["longest_run_fraction_p10_minimum"]),
        ),
        "longest_run_fraction_median": _criterion(
            coverage_metrics["longest_run_fraction_median"],
            relation="at_least",
            threshold=float(coverage_thresholds["longest_run_fraction_median_minimum"]),
        ),
    }
    track_thresholds = mapping(gate.get("track_quality_thresholds"), "track thresholds")
    track_criteria = {
        "mask_transition_rate_p90": _criterion(
            track_metrics["mask_transition_rate_p90"],
            relation="at_most",
            threshold=float(track_thresholds["mask_transition_rate_p90_maximum"]),
        ),
        "base_fill_boundary_shape_jump_p95": _criterion(
            track_metrics["base_fill_boundary_shape_jump_p95"],
            relation="at_most",
            threshold=float(track_thresholds["base_fill_boundary_shape_jump_p95_maximum"]),
        ),
        "contiguous_cycle_supported_video_fraction": _criterion(
            track_metrics["contiguous_cycle_supported_video_fraction"],
            relation="at_least",
            threshold=float(
                track_thresholds["contiguous_cycle_supported_video_fraction_minimum"]
            ),
        ),
    }
    periodicity_thresholds = mapping(gate.get("periodicity_thresholds"), "period thresholds")
    periodicity_criteria = {
        "period_confidence_p25": _criterion(
            periodicity_metrics["period_confidence_p25"],
            relation="at_least",
            threshold=float(periodicity_thresholds["period_confidence_p25_minimum"]),
        ),
        "period_confidence_median": _criterion(
            periodicity_metrics["period_confidence_median"],
            relation="at_least",
            threshold=float(periodicity_thresholds["period_confidence_median_minimum"]),
        ),
        "periodic_track_video_fraction": _criterion(
            periodicity_metrics["periodic_track_video_fraction"],
            relation="at_least",
            threshold=float(periodicity_thresholds["periodic_track_video_fraction_minimum"]),
        ),
    }
    anchor_thresholds = mapping(gate.get("anchor_risk_thresholds"), "anchor thresholds")
    anchor_criteria = {
        "unanchored_filled_video_fraction": _criterion(
            anchor_metrics["unanchored_filled_video_fraction"],
            relation="at_most",
            threshold=float(anchor_thresholds["unanchored_filled_video_fraction_maximum"]),
        ),
        "unanchored_filled_low_periodicity_video_total": _criterion(
            unanchored_filled_low_periodicity,
            relation="at_most",
            threshold=int(anchor_thresholds["unanchored_filled_low_periodicity_video_maximum"]),
        ),
    }
    integrity_thresholds = mapping(gate.get("integrity_thresholds"), "integrity thresholds")
    integrity_criteria = {
        "invariant_failure_total": _criterion(
            invariant_failures,
            relation="at_most",
            threshold=int(integrity_thresholds["invariant_failure_maximum"]),
        ),
        "v4a_bitwise_mismatch_total": _criterion(
            bitwise_mismatches,
            relation="at_most",
            threshold=int(integrity_thresholds["v4a_bitwise_mismatch_maximum"]),
        ),
        "same39_bytewise_mismatch_total": _criterion(
            same39_mismatches,
            relation="at_most",
            threshold=int(integrity_thresholds["same39_bytewise_mismatch_maximum"]),
        ),
        "nonfinite_metric_total": _criterion(
            nonfinite_metrics,
            relation="at_most",
            threshold=int(integrity_thresholds["nonfinite_metric_maximum"]),
        ),
        "exactly_once_record_total": _criterion(
            integrity_metrics["exactly_once_record_total"], relation="at_least", threshold=337
        ),
        "fresh_remainder_total": _criterion(
            integrity_metrics["fresh_remainder_total"], relation="at_least", threshold=298
        ),
        "reused_same39_total": _criterion(
            integrity_metrics["reused_same39_total"], relation="at_least", threshold=39
        ),
    }
    criteria_groups = (
        coverage_criteria,
        track_criteria,
        periodicity_criteria,
        anchor_criteria,
        integrity_criteria,
    )
    passed = all(item["passed"] for group in criteria_groups for item in group.values())
    return {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4d_train337_full_label_free_gate",
        "source_revision": source_revision,
        "container_image_id": container_image_id,
        "protocol": "ucfrep_526",
        "split": "train",
        "label_free": True,
        "passed": bool(passed),
        "baseline_training_authorized": bool(passed),
        "training_authorization_scope": "baseline-training-with-exact-full337-v4d-pose-cache",
        "extraction_only_authorization_accepted_as_training_authority": False,
        "bindings": {
            "gate_sha256": gate_sha256,
            "full337_ledger_sha256": ledger_sha256,
            "full337_pose_fingerprint": V4D_POSE_FINGERPRINT,
            "full337_cache_set_sha256": candidate_snapshot.fingerprint,
            "v2_extraction_authorization_sha256": same39.authorization_receipt_sha256,
            "same39_failure_receipt_sha256": same39.evidence.failure_receipt_sha256,
            "same39_audit_sha256": same39.evidence.audit_sha256,
            "same39_ledger_sha256": same39.evidence.ledger_sha256,
            "same39_selection_sha256": same39.evidence.selection_sha256,
            "same39_cache_set_sha256": same39.snapshot.fingerprint,
            "v4a_cache_set_sha256": v4a.snapshot.fingerprint,
            "train337_sidecar_sha256": sidecar_sha256,
            "train337_commitment_sha256": commitment_sha256,
            "train337_identity_sha256": TRAIN337_IDENTITY_SHA256,
            "config_file_sha256": V4D_CONFIG_FILE_SHA256,
            "config_fingerprint": V4D_CONFIG_FINGERPRINT,
            "model_asset_sha256": MODEL_ASSET_SHA256,
        },
        "coverage_metrics": coverage_metrics,
        "coverage_criteria": coverage_criteria,
        "track_quality_metrics": track_metrics,
        "track_quality_criteria": track_criteria,
        "periodicity_metrics": periodicity_metrics,
        "periodicity_criteria": periodicity_criteria,
        "anchor_risk_metrics": anchor_metrics,
        "anchor_risk_criteria": anchor_criteria,
        "integrity_metrics": integrity_metrics,
        "integrity_criteria": integrity_criteria,
        "diagnostic_rows_sha256": canonical_sha256(diagnostic_rows),
        "diagnostic_rows": diagnostic_rows,
        "mount_audit": {
            "train337_identity_mounted": True,
            "train337_candidate_caches_mounted": True,
            "train337_v4a_caches_mounted": True,
            "source_videos_mounted": False,
            "model_asset_mounted": False,
            "dev84_mounted": False,
            "test105_mounted": False,
            "targets_mounted": False,
        },
    }


def _decision_receipt(audit: Mapping[str, Any], *, audit_sha256: str) -> dict[str, Any]:
    passed = audit.get("passed") is True
    failed: list[str] = []
    for group_name in (
        "coverage_criteria",
        "track_quality_criteria",
        "periodicity_criteria",
        "anchor_risk_criteria",
        "integrity_criteria",
    ):
        group = mapping(audit.get(group_name), group_name)
        failed.extend(
            f"{group_name}.{name}"
            for name, item in group.items()
            if not bool(mapping(item, f"{group_name}.{name}").get("passed"))
        )
    return {
        "schema_version": 1,
        "artifact_type": (
            "pams_pose_recovery_v4d_full337_training_authorization"
            if passed
            else "pams_pose_recovery_v4d_full337_training_denial"
        ),
        "source_revision": audit.get("source_revision"),
        "container_image_id": audit.get("container_image_id"),
        "protocol": "ucfrep_526",
        "split": "train",
        "label_free": True,
        "authorization_scope": "baseline_training_input_pose_cache" if passed else "none",
        "passed": passed,
        "baseline_training_authorized": passed,
        "full337_pose_extraction_only_receipt_is_insufficient": True,
        "failed_criteria": sorted(failed),
        "bindings": {
            **dict(mapping(audit.get("bindings"), "audit bindings")),
            "full337_gate_audit_sha256": audit_sha256,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--extraction-authorization-sha256", required=True)
    parser.add_argument("--same39-failure-receipt-sha256", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-commitment", type=Path, required=True)
    parser.add_argument("--v4a-cache-dir", type=Path, required=True)
    parser.add_argument("--v4a-ledger", type=Path, required=True)
    parser.add_argument("--v4a-paired-gate", type=Path, required=True)
    parser.add_argument("--extraction-authorization", type=Path, required=True)
    parser.add_argument("--same39-failure-receipt", type=Path, required=True)
    parser.add_argument("--same39-audit", type=Path, required=True)
    parser.add_argument("--same39-ledger", type=Path, required=True)
    parser.add_argument("--same39-selection", type=Path, required=True)
    parser.add_argument("--same39-cache-dir", type=Path, required=True)
    parser.add_argument("--candidate-cache-dir", type=Path, required=True)
    parser.add_argument("--candidate-ledger", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--authorization-output", type=Path, required=True)
    parser.add_argument("--denial-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_full337(
            source_revision=args.source_revision,
            container_image_id=args.container_image_id,
            expected_gate_sha256=args.gate_sha256,
            expected_extraction_authorization_sha256=args.extraction_authorization_sha256,
            expected_same39_failure_receipt_sha256=args.same39_failure_receipt_sha256,
            gate_path=args.gate,
            train_input_path=args.train_input,
            train_commitment_path=args.train_commitment,
            v4a_cache_dir=args.v4a_cache_dir,
            v4a_ledger_path=args.v4a_ledger,
            v4a_paired_gate_path=args.v4a_paired_gate,
            extraction_authorization_path=args.extraction_authorization,
            same39_failure_receipt_path=args.same39_failure_receipt,
            same39_audit_path=args.same39_audit,
            same39_ledger_path=args.same39_ledger,
            same39_selection_path=args.same39_selection,
            same39_cache_dir=args.same39_cache_dir,
            candidate_cache_dir=args.candidate_cache_dir,
            candidate_ledger_path=args.candidate_ledger,
        )
        write_json_exclusive(args.audit_output, result)
        audit_sha256 = sha256_file(args.audit_output)
        decision = _decision_receipt(result, audit_sha256=audit_sha256)
        write_json_exclusive(
            args.authorization_output if result["passed"] else args.denial_output,
            decision,
        )
    except (OSError, TypeError, ValueError, Full337ContractError) as exc:
        print(f"v4d full337 gate failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "baseline_training_authorized": result["baseline_training_authorized"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
