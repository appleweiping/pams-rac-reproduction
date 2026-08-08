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
    require,
    sha256_file,
    write_json_exclusive,
)

from pams.config import load_config
from pams.data import load_pose_input_manifest, pose_input_identity_sha256
from pams.keypoint_single_source import (
    AssociationWeights,
    load_candidate_evidence_npz,
    select_segmented_top2_viterbi_paths,
)


def _object(path: Path, role: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, Mapping), f"{role} must be an object")
    return value


def _replay_prefix_suffix(
    *,
    candidates: Sequence[Any],
    evidence: Mapping[str, Any],
    weights_key: str,
    prefix_stop: int,
    suffix_start: int,
    maximum_bridge_gap_frames: int,
) -> tuple[tuple[int | None, ...], tuple[int | None, ...]]:
    no_anchors = tuple(None for _ in candidates)
    weights = AssociationWeights(**dict(evidence[weights_key]))
    prefix, _ = select_segmented_top2_viterbi_paths(
        candidates[:prefix_stop],
        anchor_residuals=no_anchors[:prefix_stop],
        weights=weights,
        maximum_bridge_gap_frames=maximum_bridge_gap_frames,
    )
    suffix, _ = select_segmented_top2_viterbi_paths(
        candidates[suffix_start:],
        anchor_residuals=no_anchors[suffix_start:],
        weights=weights,
        maximum_bridge_gap_frames=maximum_bridge_gap_frames,
    )
    return prefix.selected_indices, suffix.selected_indices


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
    manifest = load_pose_input_manifest(train_input_path, validate_exact=True)
    require(len(manifest.records) == 337, "expected canonical train337")
    raw = _object(raw_ledger_path, "same39 raw ledger")
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
        path_artifact_path = Path(str(raw_row["path_segment_evidence_path"])).resolve(strict=True)
        candidate_path = Path(str(raw_row["candidate_evidence_path"])).resolve(strict=True)
        require(sha256_file(path_artifact_path) == raw_row["path_segment_evidence_sha256"], "path SHA mismatch")
        require(sha256_file(candidate_path) == raw_row["candidate_evidence_sha256"], "candidate SHA mismatch")
        path_artifact = _object(path_artifact_path, "path artifact")
        evidence = path_artifact.get("raw_evidence")
        require(isinstance(evidence, Mapping), "raw path evidence missing")
        decoded = int(evidence["decoded_frames"])
        require(decoded > 0, "same39 overlap requires decoded frames")
        bundle = load_candidate_evidence_npz(candidate_path)
        require(len(bundle.frame_offsets) == int(evidence["source_frames"]) + 1, "candidate timeline mismatch")
        candidates = tuple(
            (
                None
                if int(bundle.frame_offsets[index]) == int(bundle.frame_offsets[index + 1])
                else np.ascontiguousarray(
                    bundle.candidates[
                        int(bundle.frame_offsets[index]) : int(bundle.frame_offsets[index + 1])
                    ][:4],
                    dtype=np.float32,
                )
            )
            for index in range(decoded)
        )
        prefix_stop = int(math.ceil(0.60 * decoded))
        suffix_start = int(math.floor(0.40 * decoded))
        require(0 <= suffix_start < prefix_stop <= decoded, "prefix/suffix geometry invalid")
        gap = int(evidence["effective_maximum_bridge_gap_frames"])
        primary_prefix, primary_suffix = _replay_prefix_suffix(
            candidates=candidates,
            evidence=evidence,
            weights_key="primary_weights",
            prefix_stop=prefix_stop,
            suffix_start=suffix_start,
            maximum_bridge_gap_frames=gap,
        )
        secondary_prefix, secondary_suffix = _replay_prefix_suffix(
            candidates=candidates,
            evidence=evidence,
            weights_key="secondary_weights",
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
        video_id = str(raw_row["video_id"])
        opaque_id = hashlib.sha256(video_id.encode("utf-8")).hexdigest()
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
