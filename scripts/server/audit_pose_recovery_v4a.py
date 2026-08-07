#!/usr/bin/env python3
"""Run the frozen train337 official-segment reference/v4a pose-input gate.

The audit accepts only the train pose-input identity sidecar/commitment, the
two train pose-cache pools, their extraction ledgers, and a frozen gate YAML.
It has no argument for source videos, dev84, test105, annotations, or counts.
"""

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

from pams.data import (
    load_pose_cache_set,
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
)

_FORBIDDEN_KEYS = frozenset(
    {
        "action",
        "count",
        "ground_truth",
        "ground_truth_count",
        "gt",
        "label",
        "labels",
        "target",
        "targets",
        "test",
        "dev",
    }
)


class RecoveryAuditError(RuntimeError):
    """Raised when a frozen v4a input or invariant is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryAuditError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_non_finite(value: str) -> None:
    raise RecoveryAuditError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_or_privileged_fields(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise RecoveryAuditError(f"duplicate JSON field is forbidden: {key!r}")
        if key.casefold() in _FORBIDDEN_KEYS:
            raise RecoveryAuditError(f"privileged split/target field is forbidden: {key!r}")
        payload[key] = value
    return payload


def _load_label_free_json(path: Path, *, role: str) -> tuple[dict[str, Any], str]:
    source = path.resolve(strict=True)
    encoded = source.read_bytes()
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_or_privileged_fields,
            parse_constant=_reject_non_finite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecoveryAuditError(f"{role} must be strict UTF-8 JSON") from exc
    _require(isinstance(payload, dict), f"{role} root must be an object")
    return payload, hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any, role: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{role} must be an object")
    return value


def _integer(value: Any, role: str) -> int:
    _require(type(value) is int and value >= 0, f"{role} must be a non-negative integer")
    return int(value)


def _finite(value: Any, role: str) -> float:
    _require(type(value) in {int, float} and math.isfinite(float(value)), f"{role} must be finite")
    return float(value)


def _digest(value: Any, role: str) -> str:
    text = str(value)
    _require(
        len(text) == 64 and all(character in "0123456789abcdef" for character in text),
        f"{role} must be lowercase SHA-256",
    )
    return text


def _load_gate(path: Path) -> dict[str, Any]:
    source = path.resolve(strict=True)
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "gate root must be a mapping")
    _require(payload.get("schema_version") == 1, "gate schema_version must be 1")
    _require(
        payload.get("artifact_type") == "pams_pose_recovery_v4a_train337_gate",
        "unexpected gate artifact_type",
    )
    _require(payload.get("protocol") == "ucfrep_526", "gate protocol must be ucfrep_526")
    _require(payload.get("split") == "train", "gate split must be train")
    _require(payload.get("expected_records") == 337, "gate must freeze exactly train337")
    return payload


def _validate_ledger(
    payload: Mapping[str, Any],
    *,
    role: str,
    expected_records: int,
    expected_pose_fingerprint: str,
    expected_identity_sha256: str,
) -> tuple[Mapping[str, Any], ...]:
    _require(payload.get("schema_version") == 2, f"{role} ledger schema must be 2")
    _require(payload.get("input_kind") == "label_free_sidecar", f"{role} is not label-free")
    _require(payload.get("protocol") == "ucfrep_526", f"{role} protocol mismatch")
    _require(payload.get("split") == "train", f"{role} split mismatch")
    _require(payload.get("identity_sha256") == expected_identity_sha256, f"{role} identity mismatch")
    _require(payload.get("pose_fingerprint") == expected_pose_fingerprint, f"{role} pose fingerprint mismatch")
    for field in ("selected", "completed", "extracted"):
        _require(payload.get(field) == expected_records, f"{role} {field} mismatch")
    _require(payload.get("skipped") == 0, f"{role} must not contain skipped caches")
    _require(payload.get("failed") == 0, f"{role} contains extraction failures")
    _require(payload.get("failures") == [], f"{role} failure list is not empty")
    rows = payload.get("caches")
    _require(isinstance(rows, list) and len(rows) == expected_records, f"{role} cache rows mismatch")
    typed = tuple(_mapping(row, f"{role} cache row") for row in rows)
    identifiers = [str(row.get("video_id")) for row in typed]
    _require(len(set(identifiers)) == expected_records, f"{role} has duplicate video IDs")
    return typed


def _rows_by_id(rows: Sequence[Mapping[str, Any]], *, role: str) -> dict[str, Mapping[str, Any]]:
    result = {str(row["video_id"]): row for row in rows}
    _require(len(result) == len(rows), f"{role} has duplicate video IDs")
    return result


def _percentile(values: Sequence[float], quantile: float) -> float:
    _require(bool(values), "percentile requires at least one value")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile, method="linear"))


def _criterion(value: float | int, *, relation: str, threshold: float | int) -> dict[str, Any]:
    if relation == "at_least":
        passed = value >= threshold
    elif relation == "at_most":
        passed = value <= threshold
    else:
        raise ValueError(f"unknown gate relation: {relation}")
    return {
        "value": value,
        "relation": relation,
        "threshold": threshold,
        "passed": bool(passed),
    }


def audit_pose_recovery_v4a(
    *,
    gate_path: Path,
    train_input_path: Path,
    train_commitment_path: Path,
    reference_cache_dir: Path,
    reference_ledger_path: Path,
    v4_cache_dir: Path,
    v4_ledger_path: Path,
) -> dict[str, Any]:
    """Return a path-free paired gate artifact for one immutable train337 run."""

    gate = _load_gate(gate_path)
    bindings = _mapping(gate.get("bindings"), "gate bindings")
    thresholds = _mapping(gate.get("thresholds"), "gate thresholds")
    expected_records = int(gate["expected_records"])
    reference_pose_fingerprint = _digest(
        bindings.get("reference_pose_fingerprint"),
        "official-segment reference pose fingerprint",
    )
    v4_pose_fingerprint = _digest(bindings.get("v4_pose_fingerprint"), "v4 pose fingerprint")
    expected_asset_sha256 = _digest(
        bindings.get("heavy_model_asset_sha256"),
        "heavy model asset SHA-256",
    )
    expected_sidecar_sha256 = _digest(
        bindings.get("train_sidecar_sha256"),
        "frozen train337 sidecar SHA-256",
    )
    expected_commitment_sha256 = _digest(
        bindings.get("train_commitment_sha256"),
        "frozen train337 commitment SHA-256",
    )
    expected_identity_sha256 = _digest(
        bindings.get("train_identity_sha256"),
        "frozen train337 identity SHA-256",
    )
    expected_reference_ledger_sha256 = _digest(
        bindings.get("reference_ledger_sha256"),
        "frozen official-segment reference ledger SHA-256",
    )

    sidecar_payload, sidecar_sha256 = _load_label_free_json(
        train_input_path,
        role="train337 sidecar",
    )
    _require(sidecar_payload.get("split") == "train", "sidecar split must be train")
    manifest = load_pose_input_manifest(train_input_path, validate_exact=True)
    commitment = load_pose_input_commitment(train_commitment_path)
    identity_sha256 = pose_input_identity_sha256(manifest.records)
    commitment_sha256 = _sha256_file(train_commitment_path.resolve(strict=True))
    _require(sidecar_sha256 == expected_sidecar_sha256, "frozen sidecar SHA mismatch")
    _require(
        commitment_sha256 == expected_commitment_sha256,
        "frozen commitment SHA mismatch",
    )
    _require(identity_sha256 == expected_identity_sha256, "frozen identity mismatch")
    _require(len(manifest.records) == expected_records, "sidecar is not train337")
    _require(commitment.protocol == manifest.protocol, "commitment protocol mismatch")
    _require(commitment.split == manifest.split, "commitment split mismatch")
    _require(commitment.record_total == expected_records, "commitment record count mismatch")
    _require(commitment.identity_sha256 == identity_sha256, "commitment identity mismatch")
    _require(commitment.sidecar_sha256 == sidecar_sha256, "commitment sidecar SHA mismatch")
    _require(commitment.sidecar_fingerprint == manifest.fingerprint, "commitment fingerprint mismatch")

    reference_payload, reference_ledger_sha256 = _load_label_free_json(
        reference_ledger_path,
        role="official-segment reference ledger",
    )
    v4_payload, v4_ledger_sha256 = _load_label_free_json(v4_ledger_path, role="v4a ledger")
    _require(
        reference_ledger_sha256 == expected_reference_ledger_sha256,
        "frozen official-segment reference ledger SHA mismatch",
    )
    reference_rows = _validate_ledger(
        reference_payload,
        role="official-segment reference",
        expected_records=expected_records,
        expected_pose_fingerprint=reference_pose_fingerprint,
        expected_identity_sha256=identity_sha256,
    )
    v4_rows = _validate_ledger(
        v4_payload,
        role="v4a",
        expected_records=expected_records,
        expected_pose_fingerprint=v4_pose_fingerprint,
        expected_identity_sha256=identity_sha256,
    )
    recovery_binding = _mapping(v4_payload.get("pose_recovery"), "v4a recovery binding")
    _require(recovery_binding.get("schema_version") == 1, "v4a recovery schema mismatch")
    _require(
        recovery_binding.get("preprocessing_revision")
        == "official-segment-heavy-missing-retry-full-timeline-v4a",
        "v4a preprocessing revision mismatch",
    )
    _require(
        recovery_binding.get("heavy_model_asset_sha256") == expected_asset_sha256,
        "v4a heavy asset binding mismatch",
    )
    _require(
        recovery_binding.get("pose_coordinate_interpolation") is False,
        "v4a ledger permits pose-coordinate interpolation",
    )
    _require(
        recovery_binding.get("temporal_resampling") == "none_native_timeline",
        "v4a ledger does not preserve the native timeline",
    )

    reference_sequences, reference_snapshot = load_pose_cache_set(
        manifest.records,
        cache_dir=reference_cache_dir,
        pose_fingerprint=reference_pose_fingerprint,
    )
    v4_sequences, v4_snapshot = load_pose_cache_set(
        manifest.records,
        cache_dir=v4_cache_dir,
        pose_fingerprint=v4_pose_fingerprint,
    )
    _require(
        reference_snapshot.fingerprint
        == bindings.get("reference_pose_cache_set_sha256"),
        "official-segment reference cache-set fingerprint mismatch",
    )
    reference_by_id = _rows_by_id(reference_rows, role="official-segment reference")
    v4_by_id = _rows_by_id(v4_rows, role="v4a")
    expected_ids = {record.video_id for record in manifest.records}
    _require(
        set(reference_by_id) == expected_ids,
        "official-segment reference ledger identity set mismatch",
    )
    _require(set(v4_by_id) == expected_ids, "v4a ledger identity set mismatch")

    coverages: list[float] = []
    longest_run_fractions: list[float] = []
    v4_zero = 0
    recovered_reference_zero = 0
    reference_usable_to_v4_zero = 0
    observed_at_most_8 = 0
    pass0_count_mismatch = 0
    final_below_pass0 = 0
    preservation_failures = 0
    maximum_shared_error = 0.0
    coordinate_interpolation_true = 0
    native_timeline_failures = 0
    official_timeline_failures = 0
    audit_rows: list[dict[str, Any]] = []
    for record, reference_sequence, v4_sequence in zip(
        manifest.records,
        reference_sequences,
        v4_sequences,
        strict=True,
    ):
        reference_row = reference_by_id[record.video_id]
        v4_row = v4_by_id[record.video_id]
        recovery = _mapping(v4_row.get("recovery_audit"), "v4a recovery audit")
        source_frames = _integer(recovery.get("source_frames"), "source_frames")
        expected_segment_frames = _integer(
            recovery.get("expected_segment_frames"),
            "expected_segment_frames",
        )
        decoded_segment_frames = _integer(
            recovery.get("decoded_segment_frames"),
            "decoded_segment_frames",
        )
        padded_tail_frames = _integer(
            recovery.get("padded_tail_frames"),
            "padded_tail_frames",
        )
        pass0_valid = _integer(recovery.get("pass0_valid_frames"), "pass0_valid_frames")
        final_valid = _integer(recovery.get("final_valid_frames"), "final_valid_frames")
        recovered = _integer(recovery.get("recovered_valid_frames"), "recovered_valid_frames")
        observed_span = _integer(recovery.get("observed_span_frames"), "observed_span_frames")
        longest_run = _integer(recovery.get("final_longest_valid_run"), "final_longest_valid_run")
        shared_error = _finite(
            recovery.get("pass0_shared_coordinate_max_abs_error"),
            "pass0 shared coordinate error",
        )
        _require(source_frames > 0, "source_frames must be positive")
        expected_from_sidecar = int(record.clip_end_frame) - int(record.clip_start_frame)
        if not (
            source_frames == expected_segment_frames == expected_from_sidecar
            and decoded_segment_frames + padded_tail_frames == expected_segment_frames
            and decoded_segment_frames
            == _integer(v4_row.get("source_frames"), "v4a source_frames")
            == _integer(reference_row.get("source_frames"), "reference source_frames")
            and _integer(v4_row.get("selected_source_frames"), "v4a selected_source_frames")
            == expected_segment_frames
            and _integer(
                reference_row.get("selected_source_frames"),
                "reference selected_source_frames",
            )
            == expected_segment_frames
            and v4_row.get("clip_start_frame") == reference_row.get("clip_start_frame")
            == record.clip_start_frame
            and v4_row.get("clip_end_frame") == reference_row.get("clip_end_frame")
            == record.clip_end_frame
            and v4_row.get("annotation_sha256")
            == reference_row.get("annotation_sha256")
            == record.annotation_sha256
        ):
            official_timeline_failures += 1
        _require(final_valid == pass0_valid + recovered, "recovery count conservation failed")
        _require(final_valid <= source_frames, "final source-valid count exceeds source frames")
        _require(longest_run <= final_valid, "longest run exceeds final-valid count")
        _digest(recovery.get("pass0_valid_mask_sha256"), "pass0 mask SHA-256")
        _digest(recovery.get("final_valid_mask_sha256"), "final mask SHA-256")
        if pass0_valid != _integer(
            reference_row.get("source_valid_frames"),
            "reference source_valid_frames",
        ):
            pass0_count_mismatch += 1
        if final_valid < pass0_valid:
            final_below_pass0 += 1
        if recovery.get("pass0_observations_preserved") is not True:
            preservation_failures += 1
        maximum_shared_error = max(maximum_shared_error, shared_error)
        if recovery.get("pose_coordinate_interpolation") is not False:
            coordinate_interpolation_true += 1
        if recovery.get("temporal_resampling") != "none_native_timeline":
            native_timeline_failures += 1
        _require(
            recovery.get("heavy_model_asset_sha256") == expected_asset_sha256,
            "per-video heavy asset SHA-256 mismatch",
        )
        reference_cached_valid = int(np.count_nonzero(reference_sequence.valid_mask))
        v4_cached_valid = int(np.count_nonzero(v4_sequence.valid_mask))
        if final_valid == 0:
            v4_zero += 1
        if reference_cached_valid == 0 and v4_cached_valid > 0:
            recovered_reference_zero += 1
        if reference_cached_valid > 0 and v4_cached_valid == 0:
            reference_usable_to_v4_zero += 1
        if final_valid <= 8:
            observed_at_most_8 += 1
        cached_frames = _integer(v4_row.get("cached_frames"), "v4a cached_frames")
        final_mask_sha256 = hashlib.sha256(
            bytes(int(value) for value in v4_sequence.valid_mask)
        ).hexdigest()
        if not (
            v4_sequence.num_frames == expected_segment_frames
            and cached_frames == expected_segment_frames
            and v4_cached_valid == final_valid
            and final_mask_sha256 == recovery.get("final_valid_mask_sha256")
        ):
            native_timeline_failures += 1
        coverage = final_valid / source_frames
        longest_fraction = longest_run / source_frames
        coverages.append(coverage)
        longest_run_fractions.append(longest_fraction)
        audit_rows.append(
            {
                "video_id_sha256": hashlib.sha256(record.video_id.encode("utf-8")).hexdigest(),
                "source_frames": source_frames,
                "pass0_valid_frames": pass0_valid,
                "recovered_valid_frames": recovered,
                "final_valid_frames": final_valid,
                "reference_cached_valid_frames": reference_cached_valid,
                "v4_cached_valid_frames": v4_cached_valid,
                "source_coverage": coverage,
                "longest_run_fraction": longest_fraction,
                "pass0_preserved": recovery.get("pass0_observations_preserved") is True,
            }
        )

    metrics: dict[str, Any] = {
        "record_total": expected_records,
        "v4_zero_video_total": v4_zero,
        "recovered_reference_zero_video_total": recovered_reference_zero,
        "reference_usable_to_v4_zero_video_total": reference_usable_to_v4_zero,
        "source_coverage_mean": float(np.mean(coverages)),
        "source_coverage_p10": _percentile(coverages, 0.10),
        "source_coverage_p25": _percentile(coverages, 0.25),
        "source_coverage_median": _percentile(coverages, 0.50),
        "observed_at_most_8_video_total": observed_at_most_8,
        "longest_run_fraction_p10": _percentile(longest_run_fractions, 0.10),
        "longest_run_fraction_median": _percentile(longest_run_fractions, 0.50),
        "pass0_source_valid_count_mismatch_total": pass0_count_mismatch,
        "final_below_pass0_video_total": final_below_pass0,
        "pass0_preservation_failure_total": preservation_failures,
        "pass0_shared_coordinate_max_abs_error": maximum_shared_error,
        "pose_coordinate_interpolation_true_total": coordinate_interpolation_true,
        "native_timeline_invariant_failure_total": native_timeline_failures,
        "official_segment_timeline_invariant_failure_total": official_timeline_failures,
    }
    criteria = {
        "v4_zero_video_total": _criterion(
            v4_zero,
            relation="at_most",
            threshold=int(thresholds["v4_zero_video_maximum"]),
        ),
        "recovered_reference_zero_video_total": _criterion(
            recovered_reference_zero,
            relation="at_least",
            threshold=int(thresholds["recovered_reference_zero_video_minimum"]),
        ),
        "reference_usable_to_v4_zero_video_total": _criterion(
            reference_usable_to_v4_zero,
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
            observed_at_most_8,
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
            pass0_count_mismatch,
            relation="at_most",
            threshold=int(thresholds["pass0_source_valid_count_mismatch_maximum"]),
        ),
        "final_below_pass0_video_total": _criterion(
            final_below_pass0,
            relation="at_most",
            threshold=0,
        ),
        "pass0_preservation_failure_total": _criterion(
            preservation_failures,
            relation="at_most",
            threshold=0,
        ),
        "pass0_shared_coordinate_max_abs_error": _criterion(
            maximum_shared_error,
            relation="at_most",
            threshold=float(thresholds["pass0_shared_coordinate_max_abs_error_maximum"]),
        ),
        "pose_coordinate_interpolation_true_total": _criterion(
            coordinate_interpolation_true,
            relation="at_most",
            threshold=0,
        ),
        "native_timeline_invariant_failure_total": _criterion(
            native_timeline_failures,
            relation="at_most",
            threshold=0,
        ),
        "official_segment_timeline_invariant_failure_total": _criterion(
            official_timeline_failures,
            relation="at_most",
            threshold=0,
        ),
    }
    passed = all(bool(item["passed"]) for item in criteria.values())
    return {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4a_train337_paired_audit",
        "protocol": "ucfrep_526",
        "split": "train",
        "passed": passed,
        "bindings": {
            "gate_sha256": _sha256_file(gate_path.resolve(strict=True)),
            "sidecar_sha256": sidecar_sha256,
            "commitment_sha256": commitment_sha256,
            "identity_sha256": identity_sha256,
            "reference_pose_fingerprint": reference_pose_fingerprint,
            "reference_pose_cache_set_sha256": reference_snapshot.fingerprint,
            "reference_ledger_sha256": reference_ledger_sha256,
            "v4_pose_fingerprint": v4_pose_fingerprint,
            "v4_pose_cache_set_sha256": v4_snapshot.fingerprint,
            "v4_ledger_sha256": v4_ledger_sha256,
            "heavy_model_asset_sha256": expected_asset_sha256,
        },
        "metrics": metrics,
        "criteria": criteria,
        "paired_rows_sha256": _canonical_sha256(audit_rows),
        "paired_rows": audit_rows,
        "mount_audit": {
            "train_identity_mounted": True,
            "train_pose_caches_mounted": True,
            "source_videos_mounted": False,
            "dev84_mounted": False,
            "test105_mounted": False,
            "targets_mounted": False,
        },
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = "x"
    with target.open(flags, encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-commitment", type=Path, required=True)
    parser.add_argument("--reference-cache-dir", type=Path, required=True)
    parser.add_argument("--reference-ledger", type=Path, required=True)
    parser.add_argument("--v4-cache-dir", type=Path, required=True)
    parser.add_argument("--v4-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_pose_recovery_v4a(
            gate_path=args.gate,
            train_input_path=args.train_input,
            train_commitment_path=args.train_commitment,
            reference_cache_dir=args.reference_cache_dir,
            reference_ledger_path=args.reference_ledger,
            v4_cache_dir=args.v4_cache_dir,
            v4_ledger_path=args.v4_ledger,
        )
        _write_exclusive(args.output, result)
    except (OSError, TypeError, ValueError, RecoveryAuditError) as exc:
        print(f"v4a pose recovery audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
