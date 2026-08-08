#!/usr/bin/env python3
"""Replay same39 prefix60/suffix60 paths and measure their overlap20 agreement."""

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
from pose_recovery_v4d_full337_contract import (
    Full337ContractError,
    load_strict_json,
    require,
    sha256_file,
    write_json_exclusive,
)

from pams.config import load_config
from pams.data import load_pose_input_manifest, pose_input_identity_sha256
from pams.keypoint_single_source import (
    AssociationWeights,
    association_weights_from_settings,
    canonicalize_raw_detector_frame,
    effective_maximum_bridge_gap_frames,
    load_candidate_evidence_npz,
    load_raw_detector_evidence_npz,
    select_segmented_stable_actor_paths,
)


def _object(path: Path, role: str) -> Mapping[str, Any]:
    value, _ = load_strict_json(path, role=role)
    return value


def _replay_prefix_suffix(
    *,
    candidates: Sequence[Any],
    weights: AssociationWeights,
    prefix_stop: int,
    suffix_start: int,
    maximum_bridge_gap_frames: int,
) -> tuple[tuple[int | None, ...], tuple[int | None, ...]]:
    no_anchors = tuple(None for _ in candidates)
    prefix = select_segmented_stable_actor_paths(
        candidates[:prefix_stop],
        anchor_residuals=no_anchors[:prefix_stop],
        weights=weights,
        maximum_bridge_gap_frames=maximum_bridge_gap_frames,
        native_frame_offset=0,
    )
    suffix = select_segmented_stable_actor_paths(
        candidates[suffix_start:],
        anchor_residuals=no_anchors[suffix_start:],
        weights=weights,
        maximum_bridge_gap_frames=maximum_bridge_gap_frames,
        native_frame_offset=suffix_start,
    )
    return (
        prefix.selected_path.selected_indices,
        suffix.selected_path.selected_indices,
    )


def _overlap_agreement(
    prefix: tuple[int | None, ...],
    suffix: tuple[int | None, ...],
    *,
    prefix_stop: int,
    suffix_start: int,
) -> tuple[int, float]:
    comparable = [
        index
        for index in range(suffix_start, prefix_stop)
        if prefix[index] is not None and suffix[index - suffix_start] is not None
    ]
    require(comparable, "prefix/suffix overlap has no comparable observations")
    matches = sum(prefix[index] == suffix[index - suffix_start] for index in comparable)
    return len(comparable), matches / len(comparable)


def audit_same39_overlap(
    *,
    config_path: Path,
    train_input_path: Path,
    raw_ledger_path: Path,
    expected_raw_ledger_sha256: str,
    v4a_ledger_path: Path,
    independent_replay_receipt_path: Path,
    expected_independent_replay_receipt_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    require(sha256_file(raw_ledger_path) == expected_raw_ledger_sha256, "raw ledger SHA mismatch")
    config = load_config(config_path.resolve(strict=True))
    settings = config.pose.keypoint_single_source
    require(settings is not None, "v4e settings missing")
    primary_weights, secondary_weights = association_weights_from_settings(settings)
    manifest = load_pose_input_manifest(train_input_path, validate_exact=True)
    require(len(manifest.records) == 337, "expected canonical train337")
    raw = _object(raw_ledger_path, "same39 raw ledger")
    require(
        raw.get("config_file_sha256") == sha256_file(config_path)
        and raw.get("config_fingerprint") == config.fingerprint
        and raw.get("pose_fingerprint") == config.pose_fingerprint,
        "same39 raw ledger/config binding mismatch",
    )
    require(
        sha256_file(independent_replay_receipt_path)
        == expected_independent_replay_receipt_sha256,
        "independent replay receipt SHA mismatch",
    )
    replay_receipt = _object(independent_replay_receipt_path, "independent replay receipt")
    require(
        replay_receipt.get("artifact_type")
        == "pams_cycleback_unified_2d_pose_cache_validation_receipt_v1"
        and replay_receipt.get("overall_pass") is True
        and replay_receipt.get("entry_count") == 39,
        "independent same39 replay did not pass",
    )
    replay_bindings = replay_receipt.get("bindings")
    require(
        isinstance(replay_bindings, Mapping)
        and replay_bindings.get("raw_ledger_sha256")
        == expected_raw_ledger_sha256,
        "independent replay receipt does not bind this same39 ledger",
    )
    replay_rows = replay_receipt.get("replay_rows")
    require(isinstance(replay_rows, list) and len(replay_rows) == 39, "independent replay rows missing")
    replay_by_opaque = {
        str(row["video_id_sha256"]): row
        for row in replay_rows
        if isinstance(row, Mapping)
    }
    require(len(replay_by_opaque) == 39, "independent replay identities invalid")
    require(
        raw.get("artifact_type") == "pams_pose_recovery_v4e_same39_raw_ledger_v1"
        and raw.get("status") == "complete",
        "same39 raw ledger incomplete",
    )
    expected_output = raw_ledger_path.resolve(strict=True).parent / (
        "prefix-suffix-overlap.receipt.json"
    )
    require(
        output_path.parent.resolve(strict=True) / output_path.name == expected_output,
        "same39 overlap receipt locator mismatch",
    )
    raw_rows = raw.get("caches")
    require(isinstance(raw_rows, list) and len(raw_rows) == 39, "same39 rows mismatch")
    same_ids = {str(row["video_id"]) for row in raw_rows if isinstance(row, Mapping)}
    require(len(same_ids) == 39, "same39 identities invalid")
    artifact_root = raw_ledger_path.resolve(strict=True).parent.parent
    require(
        raw_ledger_path.resolve(strict=True)
        == artifact_root / "output/raw-ledger.json",
        "same39 raw ledger locator mismatch",
    )

    v4a = _object(v4a_ledger_path, "v4a ledger")
    require(sha256_file(v4a_ledger_path) == settings.base_pose_ledger_sha256, "v4a ledger SHA mismatch")
    v4a_rows = v4a.get("caches")
    require(isinstance(v4a_rows, list) and len(v4a_rows) == 337, "v4a rows mismatch")
    zero_ids = {
        str(row["video_id"])
        for row in v4a_rows
        if isinstance(row, Mapping)
        and str(row.get("video_id")) in same_ids
        and int((row.get("recovery_audit") or {}).get("final_valid_frames", -1)) == 0
    }
    require(len(zero_ids) == 11, "same39 must contain exact zero11")
    zero_records = tuple(record for record in manifest.records if record.video_id in zero_ids)
    require(
        pose_input_identity_sha256(zero_records) == settings.frozen_zero11_identity_sha256,
        "zero11 identity mismatch",
    )

    rows: list[dict[str, Any]] = []
    for raw_row in raw_rows:
        require(isinstance(raw_row, Mapping), "same39 row must be an object")
        video_id = str(raw_row["video_id"])
        opaque_id = hashlib.sha256(video_id.encode("utf-8")).hexdigest()
        evidence_root = artifact_root / "output/raw-evidence"
        path_locator = evidence_root / f"{opaque_id}.path.json"
        candidate_locator = evidence_root / f"{opaque_id}.candidates.npz"
        raw_detector_locator = evidence_root / f"{opaque_id}.raw-detector.npz"
        require(
            not path_locator.is_symlink()
            and not candidate_locator.is_symlink()
            and not raw_detector_locator.is_symlink(),
            "same39 raw evidence locator must not be a symlink",
        )
        path_artifact_path = path_locator.resolve(strict=True)
        candidate_path = candidate_locator.resolve(strict=True)
        raw_detector_path = raw_detector_locator.resolve(strict=True)
        require(
            raw_row.get("path_segment_evidence_path") == str(path_artifact_path)
            and raw_row.get("candidate_evidence_path") == str(candidate_path)
            and raw_row.get("raw_detector_evidence_path")
            == str(raw_detector_path),
            "same39 ledger raw evidence locator mismatch",
        )
        require(sha256_file(path_artifact_path) == raw_row["path_segment_evidence_sha256"], "path SHA mismatch")
        require(sha256_file(candidate_path) == raw_row["candidate_evidence_sha256"], "candidate SHA mismatch")
        require(
            sha256_file(raw_detector_path)
            == raw_row["raw_detector_evidence_sha256"],
            "raw detector SHA mismatch",
        )
        path_artifact = _object(path_artifact_path, "path artifact")
        evidence = path_artifact.get("raw_evidence")
        require(isinstance(evidence, Mapping), "raw path evidence missing")
        decoded = int(evidence["decoded_frames"])
        require(decoded > 0, "same39 overlap requires decoded frames")
        source_frames = int(evidence["source_frames"])
        bundle = load_candidate_evidence_npz(candidate_path)
        raw_bundle = load_raw_detector_evidence_npz(raw_detector_path)
        require(
            len(bundle.frame_offsets) == source_frames + 1
            and len(raw_bundle.frame_offsets) == source_frames + 1,
            "candidate/raw detector timeline mismatch",
        )
        candidates_list: list[np.ndarray | None] = []
        all_candidates_list: list[np.ndarray | None] = []
        for frame_index in range(source_frames):
            raw_start = int(raw_bundle.frame_offsets[frame_index])
            raw_stop = int(raw_bundle.frame_offsets[frame_index + 1])
            width = int(raw_bundle.frame_dimensions[frame_index, 0])
            height = int(raw_bundle.frame_dimensions[frame_index, 1])
            if frame_index >= decoded:
                require(
                    raw_start == raw_stop and width == 0 and height == 0,
                    "padded tail contains raw detector evidence",
                )
                candidates_list.append(None)
                all_candidates_list.append(None)
                continue
            require(width > 0 and height > 0, "decoded detector frame lacks dimensions")
            canonical = canonicalize_raw_detector_frame(
                labels=raw_bundle.labels[raw_start:raw_stop],
                scores=raw_bundle.scores[raw_start:raw_stop],
                boxes=raw_bundle.boxes[raw_start:raw_stop],
                keypoints=raw_bundle.keypoints[raw_start:raw_stop],
                keypoint_logits=raw_bundle.keypoint_logits[raw_start:raw_stop],
                image_width=width,
                image_height=height,
                settings=settings,
            )
            candidates_list.append(canonical.top_candidates)
            all_candidates_list.append(canonical.all_eligible_candidates)
        expected_offsets = np.zeros(source_frames + 1, dtype=np.int64)
        expected_rows: list[np.ndarray] = []
        for frame_index, frame_candidates in enumerate(all_candidates_list):
            if frame_candidates is not None:
                expected_rows.append(frame_candidates)
                expected_offsets[frame_index + 1] = (
                    expected_offsets[frame_index] + len(frame_candidates)
                )
            else:
                expected_offsets[frame_index + 1] = expected_offsets[frame_index]
        expected_packed = (
            np.ascontiguousarray(np.concatenate(expected_rows, axis=0), dtype=np.float32)
            if expected_rows
            else np.empty((0, 17, 6), dtype=np.float32)
        )
        require(
            np.array_equal(bundle.frame_offsets, expected_offsets)
            and np.array_equal(bundle.candidates, expected_packed),
            "raw detector replay differs from canonical candidate artifact",
        )
        candidates = tuple(candidates_list[:decoded])
        prefix_stop = int(math.ceil(0.60 * decoded))
        suffix_start = int(math.floor(0.40 * decoded))
        require(0 <= suffix_start < prefix_stop <= decoded, "prefix/suffix geometry invalid")
        gap = effective_maximum_bridge_gap_frames(
            fps=float(evidence["fps"]),
            maximum_seconds=settings.maximum_bridge_gap_seconds,
            frame_cap=settings.maximum_bridge_gap_frame_cap,
        )
        require(
            evidence.get("effective_maximum_bridge_gap_frames") == gap
            and evidence.get("primary_weights") == primary_weights.to_dict()
            and evidence.get("secondary_weights") == secondary_weights.to_dict(),
            "same39 path mechanics differ from bound config",
        )
        primary_prefix, primary_suffix = _replay_prefix_suffix(
            candidates=candidates,
            weights=primary_weights,
            prefix_stop=prefix_stop,
            suffix_start=suffix_start,
            maximum_bridge_gap_frames=gap,
        )
        secondary_prefix, secondary_suffix = _replay_prefix_suffix(
            candidates=candidates,
            weights=secondary_weights,
            prefix_stop=prefix_stop,
            suffix_start=suffix_start,
            maximum_bridge_gap_frames=gap,
        )
        primary_frames, primary_agreement = _overlap_agreement(
            primary_prefix,
            primary_suffix,
            prefix_stop=prefix_stop,
            suffix_start=suffix_start,
        )
        secondary_frames, secondary_agreement = _overlap_agreement(
            secondary_prefix,
            secondary_suffix,
            prefix_stop=prefix_stop,
            suffix_start=suffix_start,
        )
        replay_row = replay_by_opaque.get(opaque_id)
        require(replay_row is not None, "overlap/replay membership mismatch")
        rows.append(
            {
                "video_id_sha256": opaque_id,
                "decoded_frames": decoded,
                "prefix_stop": prefix_stop,
                "suffix_start": suffix_start,
                "overlap_observed_frames": min(primary_frames, secondary_frames),
                "prefix_suffix_primary_path_agreement": primary_agreement,
                "prefix_suffix_secondary_path_agreement": secondary_agreement,
                "v4a_zero_anchor": video_id in zero_ids,
                "representation_eligible": replay_row["representation_eligible"],
                "eligible_pair_start_count": replay_row["eligible_pair_start_count"],
                "eligible_pair_start_count_by_variant": replay_row[
                    "eligible_pair_start_count_by_variant"
                ],
                "valid_frame_fraction": replay_row["valid_frame_fraction"],
                "mean_active_joint_count": replay_row["mean_active_joint_count"],
                "mean_active_action_joint_count": replay_row[
                    "mean_active_action_joint_count"
                ],
                "candidate_evidence_sha256": raw_row["candidate_evidence_sha256"],
                "raw_detector_evidence_sha256": raw_row[
                    "raw_detector_evidence_sha256"
                ],
                "path_segment_evidence_sha256": raw_row["path_segment_evidence_sha256"],
            }
        )

    def group_summary(group_rows: list[dict[str, Any]]) -> dict[str, Any]:
        require(group_rows, "same39 stratum is empty")
        member_ids = sorted(str(row["video_id_sha256"]) for row in group_rows)
        return {
            "record_total": len(group_rows),
            "opaque_member_set_sha256": hashlib.sha256(
                json.dumps(
                    member_ids,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            "representation_eligible_total": sum(
                row["representation_eligible"] is True for row in group_rows
            ),
            "with_eligible_pair_total": sum(
                int(row["eligible_pair_start_count"]) > 0 for row in group_rows
            ),
            "minimum_primary_overlap_agreement": min(
                float(row["prefix_suffix_primary_path_agreement"]) for row in group_rows
            ),
            "minimum_secondary_overlap_agreement": min(
                float(row["prefix_suffix_secondary_path_agreement"]) for row in group_rows
            ),
            "mean_valid_frame_fraction": sum(
                float(row["valid_frame_fraction"]) for row in group_rows
            ) / len(group_rows),
            "mean_active_joint_count": sum(
                float(row["mean_active_joint_count"]) for row in group_rows
            ) / len(group_rows),
            "mean_active_action_joint_count": sum(
                float(row["mean_active_action_joint_count"]) for row in group_rows
            ) / len(group_rows),
        }

    zero_rows = [row for row in rows if row["v4a_zero_anchor"] is True]
    other_rows = [row for row in rows if row["v4a_zero_anchor"] is False]
    require(len(zero_rows) == 11 and len(other_rows) == 28, "same39 strata are not exact 11+28")
    result = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_same39_prefix_suffix_overlap_replay_v1",
        "entry_count": 39,
        "prefix_fraction": 0.60,
        "suffix_fraction": 0.60,
        "overlap_fraction": 0.20,
        "rows": rows,
        "strata": {
            "v4a_zero11": group_summary(zero_rows),
            "v4a_nonzero28": group_summary(other_rows),
        },
        "bindings": {
            "source_revision": raw["source_revision"],
            "container_image_id": raw["container_image_id"],
            "config_file_sha256": raw["config_file_sha256"],
            "config_fingerprint": raw["config_fingerprint"],
            "raw_ledger_sha256": expected_raw_ledger_sha256,
            "v4a_ledger_sha256": settings.base_pose_ledger_sha256,
            "zero11_identity_sha256": settings.frozen_zero11_identity_sha256,
            "zero11_record_total": 11,
            "independent_replay_receipt_sha256": (
                expected_independent_replay_receipt_sha256
            ),
        },
        "label_free": True,
        "cycleback_representation_input_authorized": False,
        "baseline_training_authorized": False,
    }
    write_json_exclusive(output_path, result)
    os.chmod(output_path, 0o444)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--raw-ledger", type=Path, required=True)
    parser.add_argument("--raw-ledger-sha256", required=True)
    parser.add_argument("--v4a-ledger", type=Path, required=True)
    parser.add_argument("--independent-replay-receipt", type=Path, required=True)
    parser.add_argument("--independent-replay-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_same39_overlap(
            config_path=args.config,
            train_input_path=args.train_input,
            raw_ledger_path=args.raw_ledger,
            expected_raw_ledger_sha256=args.raw_ledger_sha256,
            v4a_ledger_path=args.v4a_ledger,
            independent_replay_receipt_path=args.independent_replay_receipt,
            expected_independent_replay_receipt_sha256=(
                args.independent_replay_receipt_sha256
            ),
            output_path=args.output,
        )
    except (OSError, TypeError, ValueError, Full337ContractError, KeyError) as exc:
        print(f"v4e same39 overlap replay failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
