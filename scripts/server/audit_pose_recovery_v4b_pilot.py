#!/usr/bin/env python3
"""Project the frozen train337 gate from a label-free v4b/v4c long-tail pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml


class PilotAuditError(RuntimeError):
    """Raised when a pilot artifact or monotonic projection is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotAuditError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, role: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotAuditError(f"{role} must be strict UTF-8 JSON") from exc
    _require(isinstance(value, Mapping), f"{role} root must be an object")
    return value


def _mapping(value: Any, role: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{role} must be an object")
    return value


def _integer(value: Any, role: str) -> int:
    _require(type(value) is int and value >= 0, f"{role} must be a non-negative integer")
    return int(value)


def _digest(value: Any, role: str) -> str:
    text = str(value)
    _require(
        len(text) == 64 and all(character in "0123456789abcdef" for character in text),
        f"{role} must be lowercase SHA-256",
    )
    return text


def _percentile(values: Sequence[float], quantile: float) -> float:
    _require(bool(values), "percentile requires values")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile, method="linear"))


def _criterion(value: int | float, *, relation: str, threshold: int | float) -> dict[str, Any]:
    passed = value >= threshold if relation == "at_least" else value <= threshold
    return {
        "value": value,
        "relation": relation,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _minimum_longest_run(valid_frames: int, source_frames: int) -> int:
    """Return the tight arrangement-free lower bound for a binary mask."""

    _require(0 <= valid_frames <= source_frames and source_frames > 0, "invalid mask counts")
    if valid_frames == 0:
        return 0
    invalid_frames = source_frames - valid_frames
    return math.ceil(valid_frames / (invalid_frames + 1))


def _project_pass0_only_row(
    original: Mapping[str, Any], recovery: Mapping[str, Any]
) -> dict[str, Any]:
    """Replace a v4a row by the strict nonpilot v4b lower bound."""

    source_frames = _integer(recovery.get("source_frames"), "v4a source frames")
    pass0_valid = _integer(recovery.get("pass0_valid_frames"), "v4a pass0 valid")
    minimum_longest = _minimum_longest_run(pass0_valid, source_frames)
    projected = dict(original)
    projected.update(
        {
            "source_frames": source_frames,
            "pass0_valid_frames": pass0_valid,
            "recovered_valid_frames": 0,
            "final_valid_frames": pass0_valid,
            "v4_cached_valid_frames": pass0_valid,
            "source_coverage": pass0_valid / source_frames,
            "longest_run_fraction": minimum_longest / source_frames,
        }
    )
    return projected


def _project_observed_v4b_row(
    lower_bound: Mapping[str, Any], recovery: Mapping[str, Any]
) -> dict[str, Any]:
    """Replace exactly one selected lower-bound row by its pilot observation."""

    source_frames = _integer(recovery.get("source_frames"), "v4b source frames")
    pass0_valid = _integer(recovery.get("pass0_valid_frames"), "v4b pass0 valid")
    final_valid = _integer(recovery.get("final_valid_frames"), "v4b final valid")
    recovered = _integer(recovery.get("recovered_valid_frames"), "v4b recovered")
    longest = _integer(recovery.get("final_longest_valid_run"), "v4b longest run")
    projected = dict(lower_bound)
    projected.update(
        {
            "source_frames": source_frames,
            "pass0_valid_frames": pass0_valid,
            "recovered_valid_frames": recovered,
            "final_valid_frames": final_valid,
            "observed_span_frames": _integer(recovery.get("observed_span_frames"), "observed span"),
            "v4_cached_valid_frames": final_valid,
            "source_coverage": final_valid / source_frames,
            "longest_run_fraction": longest / source_frames,
            "pass0_preserved": recovery.get("pass0_observations_preserved") is True,
        }
    )
    return projected


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def audit_pilot(
    *,
    gate_path: Path,
    selection_path: Path,
    v4a_ledger_path: Path,
    v4a_paired_gate_path: Path,
    v4b_ledger_path: Path,
) -> dict[str, Any]:
    """Return an exact monotonic lower-bound projection for full train337."""

    gate = yaml.safe_load(gate_path.resolve(strict=True).read_text(encoding="utf-8"))
    _require(isinstance(gate, Mapping), "pilot gate must be a mapping")
    artifact_type = gate.get("artifact_type")
    if artifact_type == "pams_pose_recovery_v4b_train337_long_tail_pilot_gate":
        recovery_version = "v4b"
        expected_recovery_mode = "full-timeline-video-fill-missing-v4b"
        pose_binding_key = "v4b_pose_fingerprint"
    elif artifact_type == "pams_pose_recovery_v4c_train337_long_tail_pilot_gate":
        recovery_version = "v4c"
        expected_recovery_mode = "tasks-video-multipose4-fill-missing-v4c"
        pose_binding_key = "v4c_pose_fingerprint"
    else:
        raise PilotAuditError("unexpected pilot gate artifact type")
    _require(gate.get("split") == "train", "pilot gate split must be train")
    _require(gate.get("expected_pilot_records") == 39, "pilot gate must freeze 39 records")
    bindings = _mapping(gate.get("bindings"), "pilot gate bindings")
    thresholds = _mapping(gate.get("thresholds"), "pilot gate thresholds")
    escalation_thresholds = (
        _mapping(
            gate.get("full_extraction_escalation_thresholds"),
            "full-extraction escalation thresholds",
        )
        if recovery_version == "v4c"
        else {}
    )

    selection = _load_json(selection_path, role="pilot selection")
    v4a_ledger = _load_json(v4a_ledger_path, role="v4a ledger")
    v4a_gate = _load_json(v4a_paired_gate_path, role="v4a paired gate")
    v4b_ledger = _load_json(v4b_ledger_path, role="v4b pilot ledger")
    _require(
        _sha256_file(v4a_ledger_path.resolve(strict=True))
        == _digest(bindings.get("v4a_ledger_sha256"), "frozen v4a ledger SHA"),
        "v4a ledger SHA mismatch",
    )
    _require(
        _sha256_file(v4a_paired_gate_path.resolve(strict=True))
        == _digest(bindings.get("v4a_paired_gate_sha256"), "frozen v4a paired gate SHA"),
        "v4a paired gate SHA mismatch",
    )
    _require(
        v4b_ledger.get("pose_fingerprint")
        == _digest(bindings.get(pose_binding_key), "candidate pose fingerprint"),
        "candidate pose fingerprint mismatch",
    )
    _require(
        v4b_ledger.get("selection_sha256") == _sha256_file(selection_path.resolve(strict=True)),
        "v4b ledger/selection mismatch",
    )
    _require(v4b_ledger.get("selected") == 39, "v4b pilot did not select 39 records")
    _require(v4b_ledger.get("completed") == 39, "v4b pilot did not complete 39 records")
    _require(v4b_ledger.get("failed") == 0, "v4b pilot contains failures")
    _require(v4b_ledger.get("failures") == [], "v4b pilot failure list is not empty")
    recovery_binding = _mapping(v4b_ledger.get("pose_recovery"), "candidate recovery binding")
    _require(
        recovery_binding.get("recovery_mode") == expected_recovery_mode,
        "candidate recovery mode mismatch",
    )
    expected_heavy_asset_sha256 = _digest(
        bindings.get("heavy_model_asset_sha256"), "candidate heavy asset SHA"
    )
    _require(
        recovery_binding.get("heavy_model_asset_sha256") == expected_heavy_asset_sha256,
        "candidate recovery heavy asset mismatch",
    )

    selection_rows = selection.get("rows")
    _require(
        isinstance(selection_rows, list) and len(selection_rows) == 39, "selection rows mismatch"
    )
    selected_hashes = {
        _digest(_mapping(row, "selection row").get("video_id_sha256"), "selected video hash")
        for row in selection_rows
    }
    _require(len(selected_hashes) == 39, "selection contains duplicate hashes")
    v4a_rows = v4a_ledger.get("caches")
    _require(isinstance(v4a_rows, list) and len(v4a_rows) == 337, "v4a ledger is not train337")
    v4a_by_hash: dict[str, Mapping[str, Any]] = {}
    for raw_row in v4a_rows:
        row = _mapping(raw_row, "v4a row")
        video_id_hash = hashlib.sha256(str(row.get("video_id", "")).encode("utf-8")).hexdigest()
        _require(video_id_hash not in v4a_by_hash, "duplicate v4a video hash")
        v4a_by_hash[video_id_hash] = row

    paired_rows = v4a_gate.get("paired_rows")
    _require(isinstance(paired_rows, list) and len(paired_rows) == 337, "paired rows mismatch")
    projected_by_hash: dict[str, dict[str, Any]] = {}
    original_by_hash: dict[str, dict[str, Any]] = {}
    for raw_row in paired_rows:
        row = dict(_mapping(raw_row, "v4a paired row"))
        video_id_hash = _digest(row.get("video_id_sha256"), "paired video hash")
        _require(video_id_hash not in projected_by_hash, "duplicate paired video hash")
        projected_by_hash[video_id_hash] = row
        original_by_hash[video_id_hash] = dict(row)
    _require(set(projected_by_hash) == set(v4a_by_hash), "v4a artifact identity sets differ")

    # The candidate run does not retain v4a static/ROI detections. Before replacing
    # pilot rows with observed candidate results, reduce every train337 row to the
    # only auditable lower bound: its locked pass0 count and the tight minimum
    # possible longest run for any binary mask with that count.
    for video_id_hash, projected in projected_by_hash.items():
        v4a_recovery = _mapping(
            v4a_by_hash[video_id_hash].get("recovery_audit"), "v4a recovery audit"
        )
        projected_by_hash[video_id_hash] = _project_pass0_only_row(projected, v4a_recovery)

    pilot_rows = v4b_ledger.get("caches")
    _require(isinstance(pilot_rows, list) and len(pilot_rows) == 39, "v4b pilot rows mismatch")
    pilot_hashes: set[str] = set()
    pilot_invariant_failures = 0
    pass0_mask_mismatches = 0
    delta_rows: list[dict[str, Any]] = []
    maximum_shared_error = 0.0
    pilot_pass0_valid: list[int] = []
    pilot_candidate_valid: list[int] = []
    pilot_v4a_valid: list[int] = []
    pilot_source_frames: list[int] = []
    pilot_candidate_longest_fractions: list[float] = []
    pilot_v4a_longest_fractions: list[float] = []
    for raw_row in pilot_rows:
        row = _mapping(raw_row, "v4b row")
        video_id_hash = hashlib.sha256(str(row.get("video_id", "")).encode("utf-8")).hexdigest()
        _require(video_id_hash not in pilot_hashes, "duplicate v4b pilot video hash")
        pilot_hashes.add(video_id_hash)
        _require(video_id_hash in selected_hashes, "v4b row is outside frozen pilot selection")
        recovery = _mapping(row.get("recovery_audit"), "v4b recovery audit")
        v4a_recovery = _mapping(
            v4a_by_hash[video_id_hash].get("recovery_audit"), "v4a recovery audit"
        )
        source_frames = _integer(recovery.get("source_frames"), "v4b source frames")
        expected_frames = _integer(recovery.get("expected_segment_frames"), "v4b expected frames")
        decoded_frames = _integer(recovery.get("decoded_segment_frames"), "v4b decoded frames")
        padded_frames = _integer(recovery.get("padded_tail_frames"), "v4b padded frames")
        pass0_valid = _integer(recovery.get("pass0_valid_frames"), "v4b pass0 valid")
        final_valid = _integer(recovery.get("final_valid_frames"), "v4b final valid")
        recovered = _integer(recovery.get("recovered_valid_frames"), "v4b recovered")
        longest = _integer(recovery.get("final_longest_valid_run"), "v4b longest run")
        heavy_observed = _integer(
            recovery.get("heavy_video_frames_observed"), "heavy video observed"
        )
        heavy_fill = _integer(recovery.get("heavy_video_fill_candidates"), "heavy fill candidates")
        heavy_asset_matches = (
            recovery.get("heavy_model_asset_sha256") == expected_heavy_asset_sha256
        )
        tasks_invariant_ok = True
        if recovery_version == "v4c":
            tasks_valid = _integer(recovery.get("heavy_video_valid_frames"), "Tasks valid frames")
            tasks_overlap = _integer(
                recovery.get("heavy_video_pass0_overlap_valid_frames"),
                "Tasks/pass0 overlap frames",
            )
            candidate_total = _integer(
                recovery.get("heavy_video_candidate_total"), "Tasks candidate total"
            )
            max_candidates = _integer(
                recovery.get("heavy_video_max_candidates_per_frame"),
                "Tasks maximum candidates per frame",
            )
            timestamp_sha256 = _digest(
                recovery.get("heavy_video_timestamp_sha256"),
                "Tasks timestamp SHA",
            )
            tasks_invariant_ok = (
                recovery.get("heavy_video_num_poses") == 4
                and tasks_valid <= decoded_frames
                and tasks_overlap <= min(tasks_valid, pass0_valid)
                and heavy_fill <= tasks_valid - tasks_overlap
                and tasks_valid <= candidate_total <= 4 * tasks_valid
                and max_candidates <= 4
                and bool(timestamp_sha256)
            )
        shared_error = float(recovery.get("pass0_shared_coordinate_max_abs_error"))
        _require(math.isfinite(shared_error), "shared-coordinate error must be finite")
        maximum_shared_error = max(maximum_shared_error, shared_error)
        invariant_ok = (
            source_frames == expected_frames
            and decoded_frames + padded_frames == expected_frames
            and heavy_observed == decoded_frames
            and final_valid == pass0_valid + recovered == pass0_valid + heavy_fill
            and final_valid <= source_frames
            and longest <= final_valid
            and recovery.get("recovery_mode") == expected_recovery_mode
            and heavy_asset_matches
            and tasks_invariant_ok
            and recovery.get("pass0_observations_preserved") is True
            and shared_error <= 1e-6
            and recovery.get("pose_coordinate_interpolation") is False
            and recovery.get("temporal_resampling") == "none_native_timeline"
            and _integer(recovery.get("roi_retry_attempted"), "ROI attempted") == 0
            and _integer(recovery.get("roi_retry_detected"), "ROI detected") == 0
        )
        if not invariant_ok:
            pilot_invariant_failures += 1
        if not (
            pass0_valid == _integer(v4a_recovery.get("pass0_valid_frames"), "v4a pass0 valid")
            and recovery.get("pass0_valid_mask_sha256")
            == v4a_recovery.get("pass0_valid_mask_sha256")
        ):
            pass0_mask_mismatches += 1
        projected = projected_by_hash[video_id_hash]
        original_v4a = original_by_hash[video_id_hash]
        v4a_final = _integer(original_v4a.get("final_valid_frames"), "v4a final valid")
        v4a_coverage = float(original_v4a.get("source_coverage"))
        v4a_longest_fraction = float(original_v4a.get("longest_run_fraction"))
        _require(final_valid >= pass0_valid, "v4b pilot reduced locked pass0 coverage")
        pilot_pass0_valid.append(pass0_valid)
        pilot_candidate_valid.append(final_valid)
        pilot_v4a_valid.append(v4a_final)
        pilot_source_frames.append(source_frames)
        pilot_candidate_longest_fractions.append(longest / source_frames)
        pilot_v4a_longest_fractions.append(v4a_longest_fraction)
        projected_by_hash[video_id_hash] = _project_observed_v4b_row(projected, recovery)
        delta_rows.append(
            {
                "video_id_sha256": video_id_hash,
                "v4a_final_valid_frames": v4a_final,
                f"{recovery_version}_final_valid_frames": final_valid,
                "valid_frame_gain": final_valid - v4a_final,
                "v4a_source_coverage": v4a_coverage,
                f"{recovery_version}_source_coverage": final_valid / source_frames,
                "v4a_longest_run_fraction": v4a_longest_fraction,
                f"{recovery_version}_longest_run_fraction": longest / source_frames,
            }
        )
    _require(pilot_hashes == selected_hashes, "pilot and selection identity sets differ")

    pilot_pass0_coverages = [
        valid / source for valid, source in zip(pilot_pass0_valid, pilot_source_frames, strict=True)
    ]
    pilot_candidate_coverages = [
        valid / source
        for valid, source in zip(pilot_candidate_valid, pilot_source_frames, strict=True)
    ]
    pilot_v4a_coverages = [
        valid / source for valid, source in zip(pilot_v4a_valid, pilot_source_frames, strict=True)
    ]
    pilot_metrics: dict[str, Any] = {
        "record_total": 39,
        "pass0_valid_frames_total": sum(pilot_pass0_valid),
        "v4a_final_valid_frames_total": sum(pilot_v4a_valid),
        "candidate_final_valid_frames_total": sum(pilot_candidate_valid),
        "recovered_valid_frames_total": sum(pilot_candidate_valid) - sum(pilot_pass0_valid),
        "pass0_zero_video_total": sum(value == 0 for value in pilot_pass0_valid),
        "candidate_zero_video_total": sum(value == 0 for value in pilot_candidate_valid),
        "pass0_zero_video_reduction": sum(value == 0 for value in pilot_pass0_valid)
        - sum(value == 0 for value in pilot_candidate_valid),
        "pass0_at_most_8_video_total": sum(value <= 8 for value in pilot_pass0_valid),
        "candidate_at_most_8_video_total": sum(value <= 8 for value in pilot_candidate_valid),
        "pass0_at_most_8_video_reduction": sum(value <= 8 for value in pilot_pass0_valid)
        - sum(value <= 8 for value in pilot_candidate_valid),
        "pass0_source_coverage_mean": float(np.mean(pilot_pass0_coverages)),
        "v4a_source_coverage_mean": float(np.mean(pilot_v4a_coverages)),
        "candidate_source_coverage_mean": float(np.mean(pilot_candidate_coverages)),
        "source_coverage_mean_gain_over_pass0": float(
            np.mean(pilot_candidate_coverages) - np.mean(pilot_pass0_coverages)
        ),
        "source_coverage_mean_gain_over_v4a": float(
            np.mean(pilot_candidate_coverages) - np.mean(pilot_v4a_coverages)
        ),
        "v4a_longest_run_fraction_mean": float(np.mean(pilot_v4a_longest_fractions)),
        "candidate_longest_run_fraction_mean": float(np.mean(pilot_candidate_longest_fractions)),
        "longest_run_fraction_mean_gain_over_v4a": float(
            np.mean(pilot_candidate_longest_fractions) - np.mean(pilot_v4a_longest_fractions)
        ),
        "candidate_longest_run_fraction_p25": _percentile(pilot_candidate_longest_fractions, 0.25),
    }
    escalation_criteria = (
        {
            "pilot_invariant_failure_total": _criterion(
                pilot_invariant_failures,
                relation="at_most",
                threshold=int(escalation_thresholds["pilot_invariant_failure_maximum"]),
            ),
            "pass0_mask_mismatch_total": _criterion(
                pass0_mask_mismatches,
                relation="at_most",
                threshold=int(escalation_thresholds["pass0_mask_mismatch_maximum"]),
            ),
            "recovered_valid_frames_total": _criterion(
                pilot_metrics["recovered_valid_frames_total"],
                relation="at_least",
                threshold=int(escalation_thresholds["recovered_valid_frames_total_minimum"]),
            ),
            "pass0_zero_video_reduction": _criterion(
                pilot_metrics["pass0_zero_video_reduction"],
                relation="at_least",
                threshold=int(escalation_thresholds["pass0_zero_video_reduction_minimum"]),
            ),
            "pass0_at_most_8_video_reduction": _criterion(
                pilot_metrics["pass0_at_most_8_video_reduction"],
                relation="at_least",
                threshold=int(escalation_thresholds["pass0_at_most_8_video_reduction_minimum"]),
            ),
            "source_coverage_mean_gain_over_pass0": _criterion(
                pilot_metrics["source_coverage_mean_gain_over_pass0"],
                relation="at_least",
                threshold=float(
                    escalation_thresholds["source_coverage_mean_gain_over_pass0_minimum"]
                ),
            ),
            "source_coverage_mean_gain_over_v4a": _criterion(
                pilot_metrics["source_coverage_mean_gain_over_v4a"],
                relation="at_least",
                threshold=float(
                    escalation_thresholds["source_coverage_mean_gain_over_v4a_minimum"]
                ),
            ),
            "longest_run_fraction_mean_gain_over_v4a": _criterion(
                pilot_metrics["longest_run_fraction_mean_gain_over_v4a"],
                relation="at_least",
                threshold=float(
                    escalation_thresholds["longest_run_fraction_mean_gain_over_v4a_minimum"]
                ),
            ),
            "candidate_longest_run_fraction_p25": _criterion(
                pilot_metrics["candidate_longest_run_fraction_p25"],
                relation="at_least",
                threshold=float(
                    escalation_thresholds["candidate_longest_run_fraction_p25_minimum"]
                ),
            ),
        }
        if recovery_version == "v4c"
        else {}
    )

    projected_rows = tuple(projected_by_hash.values())
    coverages = [float(row["source_coverage"]) for row in projected_rows]
    longest_fractions = [float(row["longest_run_fraction"]) for row in projected_rows]
    metrics: dict[str, Any] = {
        "record_total": 337,
        "v4_zero_video_total": sum(int(row["final_valid_frames"]) == 0 for row in projected_rows),
        "recovered_reference_zero_video_total": sum(
            int(row["reference_cached_valid_frames"]) == 0 and int(row["final_valid_frames"]) > 0
            for row in projected_rows
        ),
        "reference_usable_to_v4_zero_video_total": sum(
            int(row["reference_cached_valid_frames"]) > 0 and int(row["final_valid_frames"]) == 0
            for row in projected_rows
        ),
        "source_coverage_mean": float(np.mean(coverages)),
        "source_coverage_p10": _percentile(coverages, 0.10),
        "source_coverage_p25": _percentile(coverages, 0.25),
        "source_coverage_median": _percentile(coverages, 0.50),
        "observed_at_most_8_video_total": sum(
            int(row["final_valid_frames"]) <= 8 for row in projected_rows
        ),
        "longest_run_fraction_p10": _percentile(longest_fractions, 0.10),
        "longest_run_fraction_median": _percentile(longest_fractions, 0.50),
        "pass0_source_valid_count_mismatch_total": pass0_mask_mismatches,
        "final_below_pass0_video_total": 0,
        "pass0_preservation_failure_total": pilot_invariant_failures,
        "pass0_shared_coordinate_max_abs_error": maximum_shared_error,
        "pose_coordinate_interpolation_true_total": 0,
        "native_timeline_invariant_failure_total": pilot_invariant_failures,
        "official_segment_timeline_invariant_failure_total": pilot_invariant_failures,
    }
    criteria = {
        "v4_zero_video_total": _criterion(
            metrics["v4_zero_video_total"],
            relation="at_most",
            threshold=int(thresholds["v4_zero_video_maximum"]),
        ),
        "recovered_reference_zero_video_total": _criterion(
            metrics["recovered_reference_zero_video_total"],
            relation="at_least",
            threshold=int(thresholds["recovered_reference_zero_video_minimum"]),
        ),
        "reference_usable_to_v4_zero_video_total": _criterion(
            metrics["reference_usable_to_v4_zero_video_total"],
            relation="at_most",
            threshold=int(thresholds["reference_usable_to_v4_zero_video_maximum"]),
        ),
        "source_coverage_mean": _criterion(
            metrics["source_coverage_mean"],
            relation="at_least",
            threshold=float(thresholds["source_coverage_mean_minimum"]),
        ),
        "source_coverage_p10": _criterion(
            metrics["source_coverage_p10"],
            relation="at_least",
            threshold=float(thresholds["source_coverage_p10_minimum"]),
        ),
        "source_coverage_p25": _criterion(
            metrics["source_coverage_p25"],
            relation="at_least",
            threshold=float(thresholds["source_coverage_p25_minimum"]),
        ),
        "source_coverage_median": _criterion(
            metrics["source_coverage_median"],
            relation="at_least",
            threshold=float(thresholds["source_coverage_median_minimum"]),
        ),
        "observed_at_most_8_video_total": _criterion(
            metrics["observed_at_most_8_video_total"],
            relation="at_most",
            threshold=int(thresholds["observed_at_most_8_video_maximum"]),
        ),
        "longest_run_fraction_p10": _criterion(
            metrics["longest_run_fraction_p10"],
            relation="at_least",
            threshold=float(thresholds["longest_run_fraction_p10_minimum"]),
        ),
        "longest_run_fraction_median": _criterion(
            metrics["longest_run_fraction_median"],
            relation="at_least",
            threshold=float(thresholds["longest_run_fraction_median_minimum"]),
        ),
        "pass0_source_valid_count_mismatch_total": _criterion(
            metrics["pass0_source_valid_count_mismatch_total"],
            relation="at_most",
            threshold=0,
        ),
        "pass0_preservation_failure_total": _criterion(
            metrics["pass0_preservation_failure_total"], relation="at_most", threshold=0
        ),
        "pass0_shared_coordinate_max_abs_error": _criterion(
            metrics["pass0_shared_coordinate_max_abs_error"],
            relation="at_most",
            threshold=float(thresholds["pass0_shared_coordinate_max_abs_error_maximum"]),
        ),
        "native_timeline_invariant_failure_total": _criterion(
            metrics["native_timeline_invariant_failure_total"],
            relation="at_most",
            threshold=0,
        ),
    }
    return {
        "schema_version": 1,
        "artifact_type": (f"pams_pose_recovery_{recovery_version}_train337_projected_pilot_audit"),
        "protocol": "ucfrep_526",
        "split": "train",
        "label_free": True,
        "passed": all(bool(criterion["passed"]) for criterion in criteria.values()),
        "worth_full_extraction": bool(escalation_criteria)
        and all(bool(criterion["passed"]) for criterion in escalation_criteria.values()),
        "projection": {
            "kind": (
                f"strict_pass0_only_lower_bound_plus_observed_39_record_{recovery_version}_pilot"
            ),
            "nonpilot_behavior": "pass0_only_count_with_tight_arrangement_free_longest_run_bound",
            "justification": (
                f"{recovery_version} does not retain v4a static/ROI recovery; therefore "
                "nonpilot rows use only locked pass0 counts and no v4a recovered frames"
            ),
        },
        "bindings": {
            "gate_sha256": _sha256_file(gate_path.resolve(strict=True)),
            "selection_sha256": _sha256_file(selection_path.resolve(strict=True)),
            "v4a_ledger_sha256": _sha256_file(v4a_ledger_path.resolve(strict=True)),
            "v4a_paired_gate_sha256": _sha256_file(v4a_paired_gate_path.resolve(strict=True)),
            "candidate_version": recovery_version,
            "candidate_ledger_sha256": _sha256_file(v4b_ledger_path.resolve(strict=True)),
            "candidate_pose_fingerprint": v4b_ledger.get("pose_fingerprint"),
        },
        "pilot_invariants": {
            "record_total": 39,
            "failure_total": int(v4b_ledger.get("failed", -1)),
            "pass0_mask_mismatch_total": pass0_mask_mismatches,
            "invariant_failure_total": pilot_invariant_failures,
            "maximum_pass0_shared_coordinate_error": maximum_shared_error,
        },
        "projected_metrics": metrics,
        "criteria": criteria,
        "pilot_metrics": pilot_metrics,
        "full_extraction_escalation_criteria": escalation_criteria,
        "delta_rows": delta_rows,
        "mount_audit": {
            "train_pilot_ledger_mounted": True,
            "train_v4a_artifacts_mounted": True,
            "source_videos_mounted": False,
            "dev84_mounted": False,
            "test105_mounted": False,
            "targets_mounted": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--v4a-ledger", type=Path, required=True)
    parser.add_argument("--v4a-paired-gate", type=Path, required=True)
    parser.add_argument("--v4b-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_pilot(
            gate_path=args.gate,
            selection_path=args.selection,
            v4a_ledger_path=args.v4a_ledger,
            v4a_paired_gate_path=args.v4a_paired_gate,
            v4b_ledger_path=args.v4b_ledger,
        )
        _write_exclusive(args.output, result)
    except (OSError, TypeError, ValueError, PilotAuditError) as exc:
        print(f"v4 recovery pilot audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["projected_metrics"], sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
