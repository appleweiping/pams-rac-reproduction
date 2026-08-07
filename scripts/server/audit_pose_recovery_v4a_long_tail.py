#!/usr/bin/env python3
"""Explain v4a recovery failures from label-free train337 ledgers only.

The output deliberately hashes video identifiers and has no input for counts,
action classes, annotations, targets, dev84, test105, or source video pixels.
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


class LongTailAuditError(RuntimeError):
    """Raised when a label-free recovery ledger violates an invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LongTailAuditError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, role: str) -> Mapping[str, Any]:
    source = path.resolve(strict=True)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LongTailAuditError(f"{role} must be strict UTF-8 JSON") from exc
    _require(isinstance(value, Mapping), f"{role} root must be an object")
    return value


def _integer(value: Any, role: str) -> int:
    _require(type(value) is int and value >= 0, f"{role} must be a non-negative integer")
    return int(value)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _cohort_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summed_fields = (
        "expected_segment_frames",
        "decoded_segment_frames",
        "padded_tail_frames",
        "pass0_valid_frames",
        "pass0_missing_decoded_frames",
        "heavy_full_frame_detected",
        "post_full_missing_frames",
        "roi_retry_eligible",
        "roi_retry_attempted",
        "roi_retry_detected",
        "unrecovered_without_roi_path_frames",
        "roi_failed_frames",
        "final_unrecovered_decoded_frames",
        "final_valid_frames",
    )
    totals = {field: sum(int(row[field]) for row in rows) for field in summed_fields}
    recovered = totals["heavy_full_frame_detected"] + totals["roi_retry_detected"]
    missing = totals["pass0_missing_decoded_frames"]
    full_failed = totals["post_full_missing_frames"]
    unrecovered = totals["final_unrecovered_decoded_frames"]
    return {
        "video_total": len(rows),
        "frame_totals": totals,
        "rates": {
            "recovered_fraction_of_pass0_missing": _ratio(recovered, missing),
            "full_frame_recovered_fraction_of_pass0_missing": _ratio(
                totals["heavy_full_frame_detected"], missing
            ),
            "roi_attempted_fraction_of_post_full_missing": _ratio(
                totals["roi_retry_attempted"], full_failed
            ),
            "roi_recovered_fraction_of_roi_attempted": _ratio(
                totals["roi_retry_detected"], totals["roi_retry_attempted"]
            ),
            "unrecovered_without_roi_path_fraction": _ratio(
                totals["unrecovered_without_roi_path_frames"], unrecovered
            ),
        },
    }


def audit_long_tail(
    *,
    ledger_path: Path,
    paired_gate_path: Path,
    analyzed_source_revision: str,
) -> dict[str, Any]:
    """Return a target-free mechanism audit for a completed v4a extraction."""

    ledger = _load_json(ledger_path, role="v4a ledger")
    paired_gate = _load_json(paired_gate_path, role="v4a paired gate")
    _require(ledger.get("schema_version") == 2, "v4a ledger schema must be 2")
    _require(ledger.get("input_kind") == "label_free_sidecar", "ledger is not label-free")
    _require(ledger.get("protocol") == "ucfrep_526", "ledger protocol mismatch")
    _require(ledger.get("split") == "train", "ledger split must be train")
    _require(ledger.get("selected") == 337, "ledger must contain exactly train337")
    _require(ledger.get("failed") == 0, "ledger contains extraction failures")
    _require(
        paired_gate.get("artifact_type") == "pams_pose_recovery_v4a_train337_paired_audit",
        "unexpected paired gate artifact type",
    )
    _require(paired_gate.get("split") == "train", "paired gate split must be train")
    bindings = paired_gate.get("bindings")
    _require(isinstance(bindings, Mapping), "paired gate bindings must be an object")
    ledger_sha256 = _sha256_file(ledger_path.resolve(strict=True))
    _require(bindings.get("v4_ledger_sha256") == ledger_sha256, "paired gate/ledger mismatch")
    raw_rows = ledger.get("caches")
    _require(isinstance(raw_rows, list) and len(raw_rows) == 337, "ledger rows must be train337")

    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        _require(isinstance(raw_row, Mapping), "ledger cache row must be an object")
        recovery = raw_row.get("recovery_audit")
        _require(isinstance(recovery, Mapping), "cache row is missing recovery audit")
        video_id = str(raw_row.get("video_id", ""))
        _require(bool(video_id), "cache row video_id must be non-empty")
        expected = _integer(recovery.get("expected_segment_frames"), "expected frames")
        decoded = _integer(recovery.get("decoded_segment_frames"), "decoded frames")
        padded = _integer(recovery.get("padded_tail_frames"), "padded frames")
        pass0 = _integer(recovery.get("pass0_valid_frames"), "pass0 valid frames")
        full_attempted = _integer(
            recovery.get("heavy_full_frame_attempted"), "heavy full-frame attempted"
        )
        full_detected = _integer(
            recovery.get("heavy_full_frame_detected"), "heavy full-frame detected"
        )
        roi_eligible = _integer(recovery.get("roi_retry_eligible"), "ROI eligible")
        roi_attempted = _integer(recovery.get("roi_retry_attempted"), "ROI attempted")
        roi_detected = _integer(recovery.get("roi_retry_detected"), "ROI detected")
        final_valid = _integer(recovery.get("final_valid_frames"), "final valid frames")
        longest_run = _integer(
            recovery.get("final_longest_valid_run"), "longest valid run"
        )
        pass0_missing = decoded - pass0
        post_full_missing = full_attempted - full_detected
        without_roi_path = post_full_missing - roi_attempted
        roi_failed = roi_attempted - roi_detected
        final_unrecovered = decoded - final_valid
        _require(expected == decoded + padded, "decode/padding conservation failed")
        _require(pass0_missing == full_attempted, "full retry did not cover every decoded miss")
        _require(full_detected <= full_attempted, "full detections exceed attempts")
        _require(roi_detected <= roi_attempted <= roi_eligible, "ROI count ordering failed")
        _require(without_roi_path >= 0 and roi_failed >= 0, "derived missing counts are negative")
        _require(
            final_unrecovered == without_roi_path + roi_failed,
            "final missing-frame conservation failed",
        )
        _require(
            final_valid == pass0 + full_detected + roi_detected,
            "final valid-frame conservation failed",
        )
        _require(longest_run <= final_valid, "longest run exceeds final-valid frames")
        rows.append(
            {
                "video_id_sha256": hashlib.sha256(video_id.encode("utf-8")).hexdigest(),
                "expected_segment_frames": expected,
                "decoded_segment_frames": decoded,
                "padded_tail_frames": padded,
                "pass0_valid_frames": pass0,
                "pass0_missing_decoded_frames": pass0_missing,
                "heavy_full_frame_detected": full_detected,
                "post_full_missing_frames": post_full_missing,
                "roi_retry_eligible": roi_eligible,
                "roi_retry_attempted": roi_attempted,
                "roi_retry_detected": roi_detected,
                "unrecovered_without_roi_path_frames": without_roi_path,
                "roi_failed_frames": roi_failed,
                "final_unrecovered_decoded_frames": final_unrecovered,
                "final_valid_frames": final_valid,
                "final_longest_valid_run": longest_run,
                "source_coverage": _ratio(final_valid, expected),
                "longest_run_fraction": _ratio(longest_run, expected),
            }
        )

    bottom_decile_size = math.ceil(len(rows) * 0.10)
    bottom_decile = sorted(
        rows,
        key=lambda row: (
            float(row["source_coverage"]),
            int(row["final_valid_frames"]),
            str(row["video_id_sha256"]),
        ),
    )[:bottom_decile_size]
    zero = [row for row in rows if int(row["final_valid_frames"]) == 0]
    at_most_8 = [row for row in rows if int(row["final_valid_frames"]) <= 8]
    cohorts = {
        "all_train337": _cohort_summary(rows),
        "final_zero": _cohort_summary(zero),
        "final_at_most_8": _cohort_summary(at_most_8),
        "bottom_coverage_decile": _cohort_summary(bottom_decile),
    }
    return {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4a_train337_long_tail_mechanism_audit",
        "protocol": "ucfrep_526",
        "split": "train",
        "label_free": True,
        "bindings": {
            "analyzed_source_revision": analyzed_source_revision,
            "ledger_sha256": ledger_sha256,
            "paired_gate_sha256": _sha256_file(paired_gate_path.resolve(strict=True)),
            "pose_fingerprint": ledger.get("pose_fingerprint"),
            "identity_sha256": ledger.get("identity_sha256"),
        },
        "cohort_definitions": {
            "bottom_coverage_decile": "lowest ceil(0.10 * 337) rows by source coverage",
            "final_at_most_8": "final_valid_frames <= 8",
            "final_zero": "final_valid_frames == 0",
        },
        "cohorts": cohorts,
        "mechanism_finding": {
            "code": "static_heavy_and_short_bilateral_roi_leave_long_unanchored_gaps_unreached",
            "evidence": {
                "all_unrecovered_without_roi_path_fraction": cohorts["all_train337"]["rates"][
                    "unrecovered_without_roi_path_fraction"
                ],
                "bottom_decile_unrecovered_without_roi_path_fraction": cohorts[
                    "bottom_coverage_decile"
                ]["rates"]["unrecovered_without_roi_path_fraction"],
                "bottom_decile_recovered_fraction_of_pass0_missing": cohorts[
                    "bottom_coverage_decile"
                ]["rates"]["recovered_fraction_of_pass0_missing"],
            },
            "next_target_free_intervention": (
                "run the verified heavy model in VIDEO mode over the complete decoded timeline; "
                "use its observations only where the locked pass0 track is missing"
            ),
        },
        "rows": rows,
        "mount_audit": {
            "train_recovery_ledger_mounted": True,
            "train_paired_gate_mounted": True,
            "source_videos_mounted": False,
            "dev84_mounted": False,
            "test105_mounted": False,
            "targets_mounted": False,
        },
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--paired-gate", type=Path, required=True)
    parser.add_argument("--analyzed-source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_long_tail(
            ledger_path=args.ledger,
            paired_gate_path=args.paired_gate,
            analyzed_source_revision=str(args.analyzed_source_revision),
        )
        _write_exclusive(args.output, result)
    except (OSError, TypeError, ValueError, LongTailAuditError) as exc:
        print(f"v4a long-tail audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["mechanism_finding"], sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
