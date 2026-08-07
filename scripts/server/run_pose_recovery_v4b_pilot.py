#!/usr/bin/env python3
"""Extract the frozen train337 v4a long-tail union with v4b/v4c, without labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pams.config import load_config
from pams.data import (
    load_pose_cache_set,
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
    validate_pose_input_binding,
)
from pams.pose import (
    TASKS_MULTIPOSE_RECOVERY_PREPROCESSING_REVISION,
    VIDEO_RECOVERY_PREPROCESSING_REVISION,
    PoseExtractorConfig,
    PoseRecoveryExtractorConfig,
    extract_many_with_failures,
)


class PilotError(RuntimeError):
    """Raised when a pilot binding or target-free invariant is violated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotError(message)


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
        raise PilotError(f"{role} must be strict UTF-8 JSON") from exc
    _require(isinstance(value, Mapping), f"{role} root must be an object")
    return value


def _integer(value: Any, role: str) -> int:
    _require(type(value) is int and value >= 0, f"{role} must be a non-negative integer")
    return int(value)


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def run_pilot(
    *,
    config_path: Path,
    train_input_path: Path,
    train_commitment_path: Path,
    video_root: Path,
    heavy_model_asset_path: Path,
    v4a_ledger_path: Path,
    v4a_long_tail_audit_path: Path,
    cache_dir: Path,
    ledger_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    """Run the deterministic coverage/longest-P10 union pilot."""

    config = load_config(config_path.resolve(strict=True))
    if config.pose.preprocessing_revision == VIDEO_RECOVERY_PREPROCESSING_REVISION:
        recovery_version = "v4b"
        recovery_mode = "full-timeline-video-fill-missing-v4b"
    elif (
        config.pose.preprocessing_revision
        == TASKS_MULTIPOSE_RECOVERY_PREPROCESSING_REVISION
    ):
        recovery_version = "v4c"
        recovery_mode = "tasks-video-multipose4-fill-missing-v4c"
    else:
        raise PilotError("pilot config is not a supported v4b/v4c recovery")
    recovery = config.pose.recovery
    _require(recovery is not None, "pilot config has no recovery settings")
    manifest = load_pose_input_manifest(train_input_path, validate_exact=True)
    commitment = load_pose_input_commitment(train_commitment_path)
    sidecar_sha256 = _sha256_file(train_input_path.resolve(strict=True))
    validate_pose_input_binding(manifest, commitment, sidecar_sha256=sidecar_sha256)
    _require(manifest.split == "train" and len(manifest.records) == 337, "input is not train337")
    full_identity_sha256 = pose_input_identity_sha256(manifest.records)

    v4a_ledger = _load_json(v4a_ledger_path, role="v4a ledger")
    long_tail = _load_json(v4a_long_tail_audit_path, role="v4a long-tail audit")
    v4a_ledger_sha256 = _sha256_file(v4a_ledger_path.resolve(strict=True))
    _require(v4a_ledger.get("input_kind") == "label_free_sidecar", "v4a ledger is not label-free")
    _require(v4a_ledger.get("split") == "train", "v4a ledger split is not train")
    _require(v4a_ledger.get("identity_sha256") == full_identity_sha256, "v4a identity mismatch")
    _require(v4a_ledger.get("selected") == 337, "v4a ledger is not train337")
    _require(v4a_ledger.get("failed") == 0, "v4a ledger has extraction failures")
    long_tail_bindings = long_tail.get("bindings")
    _require(isinstance(long_tail_bindings, Mapping), "long-tail bindings are missing")
    _require(
        long_tail_bindings.get("ledger_sha256") == v4a_ledger_sha256,
        "long-tail audit does not bind the supplied v4a ledger",
    )
    raw_rows = v4a_ledger.get("caches")
    _require(isinstance(raw_rows, list) and len(raw_rows) == 337, "v4a rows are not train337")
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    ranked: list[tuple[str, float, float, int]] = []
    for raw_row in raw_rows:
        _require(isinstance(raw_row, Mapping), "v4a cache row must be an object")
        video_id = str(raw_row.get("video_id", ""))
        _require(video_id and video_id not in rows_by_id, "invalid or duplicate v4a video ID")
        audit = raw_row.get("recovery_audit")
        _require(isinstance(audit, Mapping), "v4a row is missing recovery audit")
        source_frames = _integer(audit.get("source_frames"), "v4a source frames")
        final_valid = _integer(audit.get("final_valid_frames"), "v4a final-valid frames")
        longest = _integer(audit.get("final_longest_valid_run"), "v4a longest run")
        _require(source_frames > 0 and final_valid <= source_frames, "invalid v4a coverage")
        _require(longest <= final_valid, "invalid v4a longest run")
        rows_by_id[video_id] = raw_row
        ranked.append(
            (video_id, final_valid / source_frames, longest / source_frames, final_valid)
        )
    record_ids = {record.video_id for record in manifest.records}
    _require(set(rows_by_id) == record_ids, "v4a ledger and train337 identity sets differ")

    decile_size = math.ceil(len(ranked) * 0.10)
    coverage_bottom = {
        item[0]
        for item in sorted(ranked, key=lambda item: (item[1], item[3], item[0]))[
            :decile_size
        ]
    }
    longest_bottom = {
        item[0]
        for item in sorted(ranked, key=lambda item: (item[2], item[3], item[0]))[
            :decile_size
        ]
    }
    at_most_8 = {item[0] for item in ranked if item[3] <= 8}
    selected_ids = coverage_bottom | longest_bottom | at_most_8
    selected_records = tuple(
        record for record in manifest.records if record.video_id in selected_ids
    )
    _require(len(selected_records) == len(selected_ids), "pilot selection lost records")
    _require(len(selected_records) == 39, "frozen v4a long-tail union must contain 39 records")
    selected_identity_sha256 = pose_input_identity_sha256(selected_records)
    rank_by_id = {item[0]: item for item in ranked}
    selection_rows = [
        {
            "video_id_sha256": hashlib.sha256(record.video_id.encode("utf-8")).hexdigest(),
            "coverage_bottom_decile": record.video_id in coverage_bottom,
            "longest_run_bottom_decile": record.video_id in longest_bottom,
            "final_valid_at_most_8": record.video_id in at_most_8,
            "v4a_source_coverage": rank_by_id[record.video_id][1],
            "v4a_longest_run_fraction": rank_by_id[record.video_id][2],
            "v4a_final_valid_frames": rank_by_id[record.video_id][3],
        }
        for record in selected_records
    ]
    selection_payload = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4b_train337_long_tail_pilot_selection",
        "protocol": manifest.protocol,
        "split": "train",
        "label_free": True,
        "selection_rule": (
            "union(coverage bottom ceil(10%*337), longest-run bottom ceil(10%*337), "
            "final_valid_frames<=8), ranked from frozen v4a recovery audit"
        ),
        "record_total": len(selected_records),
        "selected_identity_sha256": selected_identity_sha256,
        "bindings": {
            "train337_sidecar_sha256": sidecar_sha256,
            "train337_commitment_sha256": _sha256_file(
                train_commitment_path.resolve(strict=True)
            ),
            "train337_identity_sha256": full_identity_sha256,
            "v4a_ledger_sha256": v4a_ledger_sha256,
            "v4a_long_tail_audit_sha256": _sha256_file(
                v4a_long_tail_audit_path.resolve(strict=True)
            ),
        },
        "cohort_counts": {
            "coverage_bottom_decile": len(coverage_bottom),
            "longest_run_bottom_decile": len(longest_bottom),
            "final_valid_at_most_8": len(at_most_8),
            "union": len(selected_records),
        },
        "rows": selection_rows,
    }
    _write_exclusive(selection_path, selection_payload)

    root = video_root.resolve(strict=True)
    videos: list[tuple[str, Path, str | None, str | None, int | None, int | None]] = []
    for record in selected_records:
        resolved_video = (root / record.video_path).resolve(strict=True)
        try:
            resolved_video.relative_to(root)
        except ValueError:
            raise PilotError("pilot video locator escapes the frozen train337 root") from None
        videos.append(
            (
                record.video_id,
                resolved_video,
                record.video_sha256,
                record.annotation_sha256,
                record.clip_start_frame,
                record.clip_end_frame,
            )
        )
    runtime_recovery = PoseRecoveryExtractorConfig(
        heavy_model_id=recovery.heavy_model_id,
        heavy_model_asset_path=heavy_model_asset_path,
        heavy_model_asset_sha256=recovery.heavy_model_asset_sha256,
        temporal_resampling=recovery.temporal_resampling,
        model_complexity=recovery.model_complexity,
        static_image_mode=recovery.static_image_mode,
        smooth_landmarks=recovery.smooth_landmarks,
        min_detection_confidence=recovery.min_detection_confidence,
        min_tracking_confidence=recovery.min_tracking_confidence,
        full_frame_retry=recovery.full_frame_retry,
        roi_retry=recovery.roi_retry,
        roi_margin_fraction=recovery.roi_margin_fraction,
        roi_min_side_fraction=recovery.roi_min_side_fraction,
        association_cost=recovery.association_cost,
        association_center_weight=recovery.association_center_weight,
        association_log_scale_weight=recovery.association_log_scale_weight,
        dominant_track_strategy=recovery.dominant_track_strategy,
        maximum_gap_frames=recovery.maximum_gap_frames,
        maximum_gap_seconds=recovery.maximum_gap_seconds,
        pose_coordinate_interpolation=recovery.pose_coordinate_interpolation,
        recovery_mode=recovery_mode,
    )
    summaries, failures = extract_many_with_failures(
        tuple(videos),
        cache_dir=cache_dir,
        pose_fingerprint=config.pose_fingerprint,
        extractor_config=PoseExtractorConfig(
            target_frames=config.data.frames,
            preprocessing_revision=config.pose.preprocessing_revision,
            model_id=config.pose.model_id,
            model_complexity=config.pose.model_complexity,
            smooth_landmarks=config.pose.smooth_landmarks,
            min_detection_confidence=config.pose.min_detection_confidence,
            min_tracking_confidence=config.pose.min_tracking_confidence,
            crop_to_detected_span=config.pose.crop_to_detected_span,
            incomplete_clip_policy=config.pose.incomplete_clip_policy,
            recovery=runtime_recovery,
        ),
    )
    successful_ids = {summary.video_id for summary in summaries}
    successful_records = tuple(
        record for record in selected_records if record.video_id in successful_ids
    )
    cache_snapshot = None
    if successful_records:
        _, snapshot = load_pose_cache_set(
            successful_records,
            cache_dir=cache_dir,
            pose_fingerprint=config.pose_fingerprint,
            materialize_sequences=False,
        )
        cache_snapshot = snapshot.to_dict()
    ledger = {
        "schema_version": 2,
        "artifact_type": (
            f"pams_pose_recovery_{recovery_version}_train337_long_tail_pilot_ledger"
        ),
        "input_kind": "label_free_train337_hashed_long_tail_subset",
        "protocol": manifest.protocol,
        "split": "train",
        "sidecar_sha256": sidecar_sha256,
        "commitment_file_sha256": _sha256_file(
            train_commitment_path.resolve(strict=True)
        ),
        "full_train_identity_sha256": full_identity_sha256,
        "identity_sha256": selected_identity_sha256,
        "selection_sha256": _sha256_file(selection_path.resolve(strict=True)),
        "pose_fingerprint": config.pose_fingerprint,
        "pose_recovery": {
            "schema_version": 1,
            "preprocessing_revision": config.pose.preprocessing_revision,
            "recovery_mode": recovery_mode,
            "heavy_model_id": recovery.heavy_model_id,
            "heavy_model_asset_sha256": recovery.heavy_model_asset_sha256,
            "temporal_resampling": recovery.temporal_resampling,
            "pose_coordinate_interpolation": recovery.pose_coordinate_interpolation,
        },
        "successful_cache_snapshot": cache_snapshot,
        "selected": len(selected_records),
        "completed": len(summaries),
        "extracted": len(summaries),
        "skipped": 0,
        "failed": len(failures),
        "recovery_version": recovery_version,
        "caches": [summary.to_dict() for summary in summaries],
        "failures": [failure.to_dict() for failure in failures],
    }
    _write_exclusive(ledger_path, ledger)
    return {
        "selected": len(selected_records),
        "completed": len(summaries),
        "failed": len(failures),
        "pose_fingerprint": config.pose_fingerprint,
        "selection_sha256": ledger["selection_sha256"],
        "cache_set_sha256": (
            None if cache_snapshot is None else cache_snapshot["fingerprint"]
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-commitment", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--heavy-model-asset", type=Path, required=True)
    parser.add_argument("--v4a-ledger", type=Path, required=True)
    parser.add_argument("--v4a-long-tail-audit", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_pilot(
            config_path=args.config,
            train_input_path=args.train_input,
            train_commitment_path=args.train_commitment,
            video_root=args.video_root,
            heavy_model_asset_path=args.heavy_model_asset,
            v4a_ledger_path=args.v4a_ledger,
            v4a_long_tail_audit_path=args.v4a_long_tail_audit,
            cache_dir=args.cache_dir,
            ledger_path=args.ledger,
            selection_path=args.selection,
        )
    except (OSError, TypeError, ValueError, PilotError) as exc:
        print(f"v4 recovery pilot extraction failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
