#!/usr/bin/env python3
"""Validate v4e unified-2d caches without legacy min-max assumptions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
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

from pams.data import (
    load_pose_cache_with_receipt,
    load_pose_input_manifest,
    pose_cache_path,
)
from pams.keypoint_single_source import (
    CYCLEBACK_PAIR_VARIANT_ALIASES,
    CYCLEBACK_PAIR_VARIANTS,
    AssociationWeights,
    TrackStabilityThresholds,
    ViterbiPath,
    _local_ambiguity_gap_rows,
    assess_track_stability,
    body_centered_uniform_scale_xy,
    build_single_source_sequence,
    load_candidate_evidence_npz,
    reconstruct_canonical_frame_evidence,
    select_segmented_top2_viterbi_paths,
)

BODY_CENTER_HIPS = (11, 12)
RELIABLE_TORSO_JOINTS = (5, 6, 11, 12)
RELIABLE_ACTION_JOINTS = (7, 8, 9, 10, 13, 14, 15, 16)
NORMALIZATION_TOLERANCE = 1.0e-5


def _empty_pair_starts() -> dict[str, list[int]]:
    return {name: [] for name in CYCLEBACK_PAIR_VARIANTS}


def _read_object(path: Path, role: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, Mapping), f"{role} must be a JSON object")
    return value


def _load_joint_mask(path: Path, *, frames: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        require(
            set(archive.files) == {"schema_version", "joint_valid_mask"},
            "joint-mask NPZ schema mismatch",
        )
        schema = np.asarray(archive["schema_version"])
        mask = np.asarray(archive["joint_valid_mask"])
    require(schema.shape == () and schema.dtype == np.int64 and int(schema) == 1, "joint-mask schema mismatch")
    require(mask.dtype == np.bool_ and mask.shape == (frames, 17), "joint-mask shape mismatch")
    return np.ascontiguousarray(mask)


def validate_unified_2d_cache_set(
    *,
    train_input_path: Path,
    cache_dir: Path,
    snapshot_path: Path,
    raw_ledger_path: Path,
    segment_index_path: Path,
    segment_policy_path: Path,
    joint_mask_snapshot_path: Path,
    pair_eligibility_path: Path,
    identity_map_path: Path,
    expected_pose_fingerprint: str,
    expected_entry_count: int,
    authorization_path: Path,
    expected_authorization_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    manifest = load_pose_input_manifest(train_input_path, validate_exact=True)
    require(manifest.split == "train" and len(manifest.records) == 337, "expected canonical train337 source")
    require(expected_entry_count in {39, 337}, "validator supports only same39 or full337")
    require(
        sha256_file(authorization_path) == expected_authorization_sha256,
        "raw authorization SHA mismatch",
    )
    authorization = _read_object(authorization_path, "raw authorization")
    require(
        authorization.get("artifact_type")
        == "pams_pose_recovery_v4e_raw_extraction_authorization_v1"
        and authorization.get("status") == "authorized"
        and authorization.get("raw_extraction_authorized") is True,
        "raw authorization schema mismatch",
    )
    expected_scope = (
        "canonical_train337_same39_source_video_kprcnn_raw_evidence_only"
        if expected_entry_count == 39
        else "canonical_train337_source_video_kprcnn_raw_evidence_only"
    )
    require(authorization.get("authorization_scope") == expected_scope, "raw authorization scope mismatch")
    auth_bindings = authorization.get("bindings")
    require(isinstance(auth_bindings, Mapping), "raw authorization bindings missing")
    outcome_locator = Path(str(auth_bindings["canonical_outcome_locator"]))
    artifact_root = (
        outcome_locator / "pilot/same39" if expected_entry_count == 39 else outcome_locator
    )
    require(
        output_path.parent.resolve(strict=True) / output_path.name
        == artifact_root / "output/pose-cache-validation.receipt.json",
        "unified-2d validation receipt locator mismatch",
    )
    for sealed_input in (
        snapshot_path, raw_ledger_path, segment_index_path, segment_policy_path,
        joint_mask_snapshot_path, pair_eligibility_path, identity_map_path,
    ):
        require(
            stat.S_IMODE(sealed_input.resolve(strict=True).stat().st_mode) & 0o222 == 0,
            f"raw artifact is not sealed read-only: {sealed_input.name}",
        )
    snapshot = _read_object(snapshot_path, "cache snapshot")
    require(snapshot.get("schema_version") == 1, "snapshot schema mismatch")
    require(snapshot.get("entry_count") == expected_entry_count, "snapshot entry count mismatch")
    require(snapshot.get("pose_fingerprint") == expected_pose_fingerprint, "pose fingerprint mismatch")
    snapshot_entries = snapshot.get("entries")
    require(isinstance(snapshot_entries, list) and len(snapshot_entries) == expected_entry_count, "snapshot entries mismatch")
    snapshot_by_id = {
        str(entry["video_id"]): entry
        for entry in snapshot_entries
        if isinstance(entry, Mapping)
    }
    require(len(snapshot_by_id) == expected_entry_count, "snapshot has duplicate video IDs")

    ledger = _read_object(raw_ledger_path, "raw ledger")
    rows = ledger.get("caches")
    require(isinstance(rows, list) and len(rows) == expected_entry_count, "raw ledger row count mismatch")
    rows_by_id = {
        str(row["video_id"]): row for row in rows if isinstance(row, Mapping)
    }
    require(len(rows_by_id) == expected_entry_count, "raw ledger has duplicate video IDs")
    require(
        ledger.get("raw_extraction_authorization_sha256")
        == expected_authorization_sha256,
        "raw ledger authorization binding mismatch",
    )
    raw_thresholds = ledger.get("frozen_thresholds")
    require(isinstance(raw_thresholds, Mapping), "raw ledger thresholds missing")
    thresholds = TrackStabilityThresholds(**dict(raw_thresholds))
    segment_index_sha256 = sha256_file(segment_index_path)
    segment_policy_sha256 = sha256_file(segment_policy_path)
    require(
        ledger.get("segment_index_sha256") == segment_index_sha256
        and ledger.get("segment_reset_policy_sha256") == segment_policy_sha256,
        "ledger segment bindings mismatch",
    )
    segment_index = _read_object(segment_index_path, "segment index")
    require(
        segment_index.get("artifact_type") == "pams_pose_recovery_v4e_segment_index_v1"
        and segment_index.get("entry_count") == expected_entry_count,
        "segment index schema mismatch",
    )
    require(
        segment_index.get("segment_reset_policy_sha256") == segment_policy_sha256,
        "segment policy binding mismatch",
    )
    segment_rows = segment_index.get("entries")
    require(isinstance(segment_rows, list) and len(segment_rows) == expected_entry_count, "segment index rows mismatch")
    segment_by_id = {
        str(entry["video_id"]): entry
        for entry in segment_rows
        if isinstance(entry, Mapping)
    }
    require(len(segment_by_id) == expected_entry_count, "segment index duplicate identities")
    joint_mask_snapshot_sha256 = sha256_file(joint_mask_snapshot_path)
    joint_mask_snapshot = _read_object(joint_mask_snapshot_path, "joint-mask snapshot")
    require(
        joint_mask_snapshot.get("artifact_type")
        == "pams_pose_recovery_v4e_joint_mask_set_snapshot_v1"
        and joint_mask_snapshot.get("entry_count") == expected_entry_count,
        "joint-mask snapshot schema mismatch",
    )
    require(
        ledger.get("joint_mask_snapshot_sha256") == joint_mask_snapshot_sha256
        and ledger.get("joint_mask_set_sha256") == joint_mask_snapshot.get("fingerprint"),
        "joint-mask snapshot ledger binding mismatch",
    )
    joint_mask_entries = joint_mask_snapshot.get("entries")
    require(
        isinstance(joint_mask_entries, list) and len(joint_mask_entries) == expected_entry_count,
        "joint-mask snapshot entries mismatch",
    )
    joint_mask_by_id = {
        str(entry["video_id"]): entry
        for entry in joint_mask_entries
        if isinstance(entry, Mapping)
    }
    require(len(joint_mask_by_id) == expected_entry_count, "joint-mask snapshot duplicate identities")
    expected_joint_mask_fingerprint = hashlib.sha256(
        json.dumps(
            {"schema_version": 1, "entries": joint_mask_entries},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    require(
        joint_mask_snapshot.get("fingerprint") == expected_joint_mask_fingerprint,
        "joint-mask set fingerprint mismatch",
    )
    pair_eligibility_sha256 = sha256_file(pair_eligibility_path)
    pair_eligibility = _read_object(pair_eligibility_path, "pair eligibility")
    require(
        pair_eligibility.get("artifact_type")
        == "pams_pose_recovery_v4e_cycleback_pair_eligibility_v1"
        and pair_eligibility.get("entry_count") == expected_entry_count
        and pair_eligibility.get("consumer_may_expand_starts") is False
        and pair_eligibility.get("start_grid_policy")
        == "native_zero_based_start_mod_hop_equals_zero",
        "pair eligibility schema mismatch",
    )
    require(
        pair_eligibility.get("variant_geometry")
        == {
            name: {"window_frames": shape[0], "hop_frames": shape[1]}
            for name, shape in CYCLEBACK_PAIR_VARIANTS.items()
        }
        and pair_eligibility.get("frozen_variant_aliases")
        == CYCLEBACK_PAIR_VARIANT_ALIASES,
        "pair eligibility variant contract mismatch",
    )
    require(
        ledger.get("cycleback_pair_eligibility_sha256") == pair_eligibility_sha256,
        "pair eligibility ledger binding mismatch",
    )
    pair_rows = pair_eligibility.get("entries")
    require(isinstance(pair_rows, list) and len(pair_rows) == expected_entry_count, "pair eligibility rows mismatch")
    pair_by_id = {
        str(entry["video_id"]): entry for entry in pair_rows if isinstance(entry, Mapping)
    }
    require(len(pair_by_id) == expected_entry_count, "pair eligibility duplicate identities")
    identity_map_sha256 = sha256_file(identity_map_path)
    identity_map = _read_object(identity_map_path, "identity map")
    require(
        identity_map.get("artifact_type") == "pams_pose_recovery_v4e_identity_map_v1"
        and identity_map.get("entry_count") == expected_entry_count,
        "identity map schema mismatch",
    )
    require(ledger.get("identity_map_sha256") == identity_map_sha256, "identity map binding mismatch")
    identity_entries = identity_map.get("entries")
    require(isinstance(identity_entries, list) and len(identity_entries) == expected_entry_count, "identity map entries mismatch")
    canonical_pairs = {
        (record.video_id, hashlib.sha256(record.video_id.encode("utf-8")).hexdigest())
        for record in manifest.records
    }
    mapped_pairs = {
        (entry["video_id"], entry["video_id_sha256"])
        for entry in identity_entries
        if isinstance(entry, Mapping)
    }
    require(len(mapped_pairs) == expected_entry_count and mapped_pairs <= canonical_pairs, "identity map is not exact")

    selected_ids = {str(entry["video_id"]) for entry in identity_entries}
    selected_records = tuple(record for record in manifest.records if record.video_id in selected_ids)
    require(len(selected_records) == expected_entry_count, "identity map lost canonical records")
    selected_opaque = {
        hashlib.sha256(record.video_id.encode("utf-8")).hexdigest()
        for record in selected_records
    }
    if expected_entry_count == 39:
        authorized_opaque = authorization.get("selected_video_id_sha256")
        require(
            isinstance(authorized_opaque, list)
            and len(authorized_opaque) == 39
            and set(str(value) for value in authorized_opaque) == selected_opaque,
            "same39 identity map does not match authorization membership",
        )
    else:
        require(selected_opaque == {pair[1] for pair in canonical_pairs}, "full337 membership mismatch")
    require(set(rows_by_id) == selected_ids, "raw ledger real-ID membership mismatch")
    require(set(snapshot_by_id) == selected_ids, "pose snapshot real-ID membership mismatch")
    require(set(joint_mask_by_id) == selected_ids, "joint-mask real-ID membership mismatch")
    require(set(pair_by_id) == selected_opaque, "pair eligibility opaque membership mismatch")
    require(set(segment_by_id) == selected_opaque, "segment index opaque membership mismatch")

    checked = 0
    replay_rows: list[dict[str, Any]] = []
    for record in selected_records:
        row = rows_by_id.get(record.video_id)
        receipt_row = snapshot_by_id.get(record.video_id)
        joint_mask_receipt = joint_mask_by_id.get(record.video_id)
        opaque_id = hashlib.sha256(record.video_id.encode("utf-8")).hexdigest()
        pair_row = pair_by_id.get(opaque_id)
        segment_row = segment_by_id.get(opaque_id)
        require(
            row is not None
            and receipt_row is not None
            and joint_mask_receipt is not None
            and pair_row is not None,
            "cache identity set mismatch",
        )
        require(segment_row is not None, "segment identity set mismatch")
        cache_path = pose_cache_path(cache_dir, record.video_id).resolve(strict=True)
        sequence, metadata, receipt = load_pose_cache_with_receipt(
            cache_path,
            expected_video_sha256=record.video_sha256,
            expected_pose_fingerprint=expected_pose_fingerprint,
            expected_annotation_sha256=record.annotation_sha256,
            expected_clip_start_frame=record.clip_start_frame,
            expected_clip_end_frame=record.clip_end_frame,
            validate_clip_provenance=True,
        )
        require(
            receipt.cache_sha256 == receipt_row.get("cache_sha256")
            and receipt.bytes == receipt_row.get("bytes")
            and receipt.cache_sha256 == row.get("cache_sha256")
            and receipt.bytes == row.get("cache_bytes"),
            "cache SHA/byte receipt mismatch",
        )
        require(
            metadata.frames == record.clip_end_frame - record.clip_start_frame
            and sequence.num_frames == metadata.frames,
            "native timeline mismatch",
        )
        xyz = np.asarray(sequence.xyz)
        valid = np.asarray(sequence.valid_mask)
        require(xyz.dtype == np.float32 and xyz.shape == (sequence.num_frames, 33, 3), "cache tensor mismatch")
        require(valid.dtype == np.bool_ and valid.shape == (sequence.num_frames,), "cache mask mismatch")
        require(np.isfinite(xyz[valid]).all(), "valid cache coordinates are non-finite")
        require(np.all(xyz[~valid] == 0.0), "invalid cache frames are not exact zero")
        require(np.all(xyz[:, 17:, :] == 0.0), "padded joints 17:33 are not exact zero")
        require(np.all(xyz[:, :, 2] == 0.0), "z coordinates are not exact zero")

        candidate_path = Path(str(row["candidate_evidence_path"])).resolve(strict=True)
        path_path = Path(str(row["path_segment_evidence_path"])).resolve(strict=True)
        require(
            sha256_file(candidate_path) == row.get("candidate_evidence_sha256")
            and candidate_path.stat().st_size == row.get("candidate_evidence_bytes"),
            "candidate artifact receipt mismatch",
        )
        require(
            sha256_file(path_path) == row.get("path_segment_evidence_sha256")
            and path_path.stat().st_size == row.get("path_segment_evidence_bytes"),
            "path artifact receipt mismatch",
        )
        require(
            stat.S_IMODE(cache_path.stat().st_mode) & 0o222 == 0
            and stat.S_IMODE(candidate_path.stat().st_mode) & 0o222 == 0
            and stat.S_IMODE(path_path.stat().st_mode) & 0o222 == 0,
            "cache/candidate/path artifact is not sealed read-only",
        )
        bundle = load_candidate_evidence_npz(candidate_path)
        require(len(bundle.frame_offsets) == sequence.num_frames + 1, "candidate timeline mismatch")
        all_candidates = tuple(
            (
                None
                if int(bundle.frame_offsets[index]) == int(bundle.frame_offsets[index + 1])
                else np.ascontiguousarray(
                    bundle.candidates[
                        int(bundle.frame_offsets[index]) : int(bundle.frame_offsets[index + 1])
                    ],
                    dtype=np.float32,
                )
            )
            for index in range(sequence.num_frames)
        )
        candidates = tuple(
            None
            if frame_candidates is None
            else np.ascontiguousarray(frame_candidates[:4], dtype=np.float32)
            for frame_candidates in all_candidates
        )
        path_artifact = _read_object(path_path, "path evidence")
        raw_evidence = path_artifact.get("raw_evidence")
        recorded_decision = path_artifact.get("eligibility_decision")
        require(
            isinstance(raw_evidence, Mapping) and isinstance(recorded_decision, Mapping),
            "path evidence payload mismatch",
        )
        decoded_frames = int(raw_evidence["decoded_frames"])
        require(
            metadata.decoded_clip_frames == decoded_frames
            and metadata.expected_clip_frames == sequence.num_frames
            and metadata.padded_tail_frames == sequence.num_frames - decoded_frames
            and metadata.incomplete_clip_policy == "pad_invalid_tail"
            and not np.any(valid[decoded_frames:]),
            "decoded/padded tail metadata mismatch",
        )
        require(
            path_artifact.get("candidate_evidence_sha256") == row.get("candidate_evidence_sha256"),
            "path/candidate binding mismatch",
        )
        no_anchors = tuple(None for _ in candidates)
        primary_weights = AssociationWeights(**dict(raw_evidence["primary_weights"]))
        secondary_weights = AssociationWeights(**dict(raw_evidence["secondary_weights"]))
        effective_gap = int(raw_evidence["effective_maximum_bridge_gap_frames"])
        replay_primary, replay_segments = select_segmented_top2_viterbi_paths(
            candidates,
            anchor_residuals=no_anchors,
            weights=primary_weights,
            maximum_bridge_gap_frames=effective_gap,
        )
        replay_secondary, secondary_segments = select_segmented_top2_viterbi_paths(
            candidates,
            anchor_residuals=no_anchors,
            weights=secondary_weights,
            maximum_bridge_gap_frames=effective_gap,
        )
        require(replay_segments == secondary_segments, "replayed segment boundaries differ")
        frame_evidence = raw_evidence.get("frame_evidence")
        require(isinstance(frame_evidence, list) and len(frame_evidence) == sequence.num_frames, "frame evidence mismatch")
        for frame_index, (frame_row, frame_candidates) in enumerate(
            zip(frame_evidence, candidates, strict=True)
        ):
            candidate_rows = frame_row.get("candidates")
            require(isinstance(candidate_rows, list), "frame candidate rows missing")
            expected_count = 0 if frame_candidates is None else len(frame_candidates)
            all_frame_candidates = all_candidates[frame_index]
            all_count = 0 if all_frame_candidates is None else len(all_frame_candidates)
            detector_counts = frame_row.get("detector_filter_counts")
            if detector_counts is None:
                require(frame_index >= decoded_frames and all_count == 0, "decoded frame filter counts missing")
                continue
            require(isinstance(detector_counts, Mapping), "detector filter counts invalid")
            require(
                detector_counts.get("eligible_before_top4") == all_count
                and detector_counts.get("dropped_by_top4") == max(0, all_count - 4),
                "pre-top4 detector evidence mismatch",
            )
            require(
                frame_row.get("frame_index") == frame_index
                and frame_row.get("candidate_count") == expected_count
                and len(candidate_rows) == expected_count,
                "frame candidate count mismatch",
            )
            if frame_candidates is not None:
                for candidate_index, (candidate_row, candidate) in enumerate(
                    zip(candidate_rows, frame_candidates, strict=True)
                ):
                    require(candidate_row.get("candidate_index") == candidate_index, "candidate index mismatch")
                    require(
                        candidate_row.get("candidate_sha256")
                        == hashlib.sha256(
                            np.ascontiguousarray(candidate, dtype=np.float32).tobytes(order="C")
                        ).hexdigest(),
                        "candidate digest mismatch",
                    )
                    require(
                        candidate_row.get("raw_keypoint_logits")
                        == [float(value) for value in candidate[:, 4]]
                        and candidate_row.get("box_score") == float(candidate[0, 5]),
                        "candidate channel summary mismatch",
                    )
        require(
            tuple(frame["primary_index"] for frame in frame_evidence)
            == replay_primary.selected_indices
            and tuple(frame["secondary_index"] for frame in frame_evidence)
            == replay_secondary.selected_indices,
            "replayed Viterbi path mismatch",
        )
        for segment in replay_segments:
            start, stop = segment[0], segment[-1] + 1
            subpath = ViterbiPath(
                replay_primary.selected_indices[start:stop],
                None,
                None,
                len(segment),
                "diagnostic-replay",
            )
            replay_gap_rows = _local_ambiguity_gap_rows(
                candidates[start:stop],
                anchor_residuals=no_anchors[start:stop],
                weights=primary_weights,
                selected_path=subpath,
            )
            for relative_index, gap in replay_gap_rows:
                recorded_gap = frame_evidence[start + relative_index][
                    "ambiguous_max_marginal_gap"
                ]
                require(
                    recorded_gap is not None and abs(float(recorded_gap) - gap) <= 1.0e-9,
                    "replayed ambiguity gap mismatch",
                )
        replay_ambiguity_by_frame: dict[int, float] = {}
        for segment in replay_segments:
            start, stop = segment[0], segment[-1] + 1
            subpath = ViterbiPath(
                replay_primary.selected_indices[start:stop],
                None,
                None,
                len(segment),
                "diagnostic-replay",
            )
            replay_ambiguity_by_frame.update(
                {
                    start + relative_index: gap
                    for relative_index, gap in _local_ambiguity_gap_rows(
                        candidates[start:stop],
                        anchor_residuals=no_anchors[start:stop],
                        weights=primary_weights,
                        selected_path=subpath,
                    )
                }
            )
        replay_sequence = build_single_source_sequence(
            video_id=record.video_id,
            fps=sequence.fps,
            source_frames=sequence.num_frames,
            decoded_frames=decoded_frames,
            candidates=candidates,
            primary_path=replay_primary,
        )
        require(
            np.array_equal(replay_sequence.valid_mask, valid)
            and np.array_equal(replay_sequence.xyz, xyz),
            "candidate/path replay does not reproduce cache bytes",
        )
        detector_diagnostics = [
            frame.get("detector_filter_counts") for frame in frame_evidence
        ]
        replay_frame_evidence = reconstruct_canonical_frame_evidence(
            candidates=candidates,
            primary=replay_primary,
            secondary=replay_secondary,
            association_segments=replay_segments,
            sequence=replay_sequence,
            ambiguity_by_frame=replay_ambiguity_by_frame,
            maximum_bridge_gap_frames=effective_gap,
            detector_diagnostics=detector_diagnostics,
            keypoint_logit_threshold=2.0,
        )
        require(
            replay_frame_evidence == frame_evidence,
            "candidate/path bytes do not reproduce canonical frame evidence",
        )
        comparable = [
            index
            for index, (primary_index, secondary_index, frame_candidates) in enumerate(
                zip(
                    replay_primary.selected_indices,
                    replay_secondary.selected_indices,
                    candidates,
                    strict=True,
                )
            )
            if primary_index is not None
            and secondary_index is not None
            and frame_candidates is not None
            and len(frame_candidates) > 1
        ]
        replay_agreement = (
            sum(
                replay_primary.selected_indices[index]
                == replay_secondary.selected_indices[index]
                for index in comparable
            )
            / len(comparable)
            if comparable
            else (1.0 if replay_primary.observed_frames > 0 else 0.0)
        )
        valid_ranges: list[dict[str, int]] = []
        valid_start: int | None = None
        for frame_index, is_valid in enumerate(valid):
            if bool(is_valid) and valid_start is None:
                valid_start = frame_index
            elif not bool(is_valid) and valid_start is not None:
                valid_ranges.append(
                    {"start": valid_start, "stop": frame_index, "frames": frame_index - valid_start}
                )
                valid_start = None
        if valid_start is not None:
            valid_ranges.append(
                {"start": valid_start, "stop": len(valid), "frames": len(valid) - valid_start}
            )
        expected_aggregates = {
            "source_frames": sequence.num_frames,
            "decoded_frames": decoded_frames,
            "padded_tail_frames": sequence.num_frames - decoded_frames,
            "valid_frames": int(np.count_nonzero(valid)),
            "source_coverage": float(np.count_nonzero(valid) / sequence.num_frames),
            "candidate_observed_frames": sum(value is not None for value in candidates),
            "candidate_total": sum(0 if value is None else len(value) for value in candidates),
            "candidate_maximum_per_frame": max(
                (0 if value is None else len(value) for value in candidates), default=0
            ),
            "eligible_candidate_before_top4_total": sum(
                0 if value is None else len(value) for value in all_candidates
            ),
            "candidate_dropped_by_top4_total": sum(
                max(0, (0 if value is None else len(value)) - 4)
                for value in all_candidates
            ),
            "crowd_quarantined_frame_total": sum(
                value is not None and len(value) > 4 for value in all_candidates
            ),
            "selected_normalization_failure_frames": sum(
                index < decoded_frames
                and replay_primary.selected_indices[index] is not None
                and not bool(valid[index])
                for index in range(sequence.num_frames)
            ),
            "dual_path_comparable_frames": len(comparable),
            "dual_path_agreement": float(replay_agreement),
            "track_segment_count": len(replay_segments),
            "track_segment_ranges": [
                {"start": segment[0], "stop": segment[-1] + 1}
                for segment in replay_segments
            ],
            "trainable_segments": valid_ranges,
        }
        filter_eligible_normalization_failure_total = 0
        for frame_candidates in all_candidates[:decoded_frames]:
            if frame_candidates is None:
                continue
            for candidate in frame_candidates:
                try:
                    body_centered_uniform_scale_xy(
                        candidate[:, :2], visibility=candidate[:, 3]
                    )
                except ValueError:
                    filter_eligible_normalization_failure_total += 1
        expected_aggregates["filter_eligible_normalization_failure_total"] = (
            filter_eligible_normalization_failure_total
        )
        require(
            all(raw_evidence.get(key) == value for key, value in expected_aggregates.items()),
            "candidate/cache bytes do not reproduce raw aggregate evidence",
        )
        replay_raw_evidence = dict(raw_evidence)
        replay_raw_evidence["frame_evidence"] = replay_frame_evidence
        replay_raw_evidence["dual_path_agreement"] = replay_agreement
        replay_raw_evidence["source_frames"] = sequence.num_frames
        replay_decision = assess_track_stability(
            replay_raw_evidence,
            thresholds=thresholds,
            same_source_period_supported=False,
        )
        require(
            replay_decision == dict(recorded_decision)
            and replay_decision == row.get("eligibility_decision"),
            "eligibility replay mismatch",
        )
        require(
            pair_row.get("native_length") == sequence.num_frames
            and pair_row.get("representation_eligible") == replay_decision["eligible"]
            and pair_row.get("starts_by_variant")
            == (
                replay_decision["eligible_pair_starts_by_variant"]
                if replay_decision["eligible"]
                else _empty_pair_starts()
            ),
            "pair eligibility replay mismatch",
        )
        require(
            pair_row.get("base_valid_starts_by_variant")
            == (
                replay_decision["base_valid_pair_starts_by_variant"]
                if replay_decision["eligible"]
                else _empty_pair_starts()
            ),
            "base pair eligibility replay mismatch",
        )
        require(
            all(
                set(pair_row["starts_by_variant"][variant])
                <= set(pair_row["base_valid_starts_by_variant"][variant])
                for variant in CYCLEBACK_PAIR_VARIANTS
            ),
            "final pair starts are not a subset of base-valid starts",
        )
        require(
            pair_row["starts_by_variant"]["W16_H4_PE0"]
            == pair_row["starts_by_variant"]["W16_H4"]
            and pair_row["base_valid_starts_by_variant"]["W16_H4_PE0"]
            == pair_row["base_valid_starts_by_variant"]["W16_H4"],
            "PE0 pair-start alias diverged from W16_H4",
        )
        expected_resets = [
            segment["start"] for segment in raw_evidence["track_segment_ranges"][1:]
        ]
        require(
            segment_row.get("native_length") == sequence.num_frames
            and segment_row.get("association_resets") == expected_resets
            and segment_row.get("eligible_frame_ranges")
            == (
                replay_decision["track_stability_frame_ranges"]
                if replay_decision["eligible"]
                else []
            )
            and segment_row.get("eligible_pair_starts_by_candidate")
            == (
                replay_decision["eligible_pair_starts_by_variant"]
                if replay_decision["eligible"]
                else _empty_pair_starts()
            )
            and segment_row.get("representation_eligible") == replay_decision["eligible"],
            "segment index replay mismatch",
        )

        joint_mask_path = Path(str(row["joint_valid_mask_path"])).resolve(strict=True)
        require(
            stat.S_IMODE(joint_mask_path.stat().st_mode) & 0o222 == 0,
            "joint-mask artifact is not sealed read-only",
        )
        require(sha256_file(joint_mask_path) == row.get("joint_valid_mask_sha256"), "joint-mask SHA mismatch")
        require(joint_mask_path.stat().st_size == row.get("joint_valid_mask_bytes"), "joint-mask bytes mismatch")
        require(
            joint_mask_receipt.get("sidecar_sha256") == row.get("joint_valid_mask_sha256")
            and joint_mask_receipt.get("bytes") == row.get("joint_valid_mask_bytes"),
            "joint-mask snapshot receipt mismatch",
        )
        joint_mask = _load_joint_mask(joint_mask_path, frames=sequence.num_frames)
        replay_joint_mask = np.zeros_like(joint_mask)
        for frame_index, selected_index in enumerate(replay_primary.selected_indices):
            if selected_index is not None and bool(valid[frame_index]):
                frame_candidates = candidates[frame_index]
                require(frame_candidates is not None, "selected candidate frame missing")
                replay_joint_mask[frame_index] = (
                    frame_candidates[selected_index, :, 4] > 2.0
                )
        require(np.array_equal(joint_mask, replay_joint_mask), "joint mask replay mismatch")
        require(np.array_equal(np.any(joint_mask, axis=1), valid), "joint/frame masks disagree")
        require(np.all(xyz[:, :17, :2][~joint_mask] == 0.0), "weak joints are not exact zero")
        for frame_index in np.flatnonzero(valid):
            mask = joint_mask[frame_index]
            require(int(np.count_nonzero(mask)) >= 8, "active frame has fewer than 8 joints")
            require(
                all(bool(mask[index]) for index in RELIABLE_TORSO_JOINTS),
                "active frame lacks bilateral shoulders or hips",
            )
            require(
                int(np.count_nonzero(mask[list(RELIABLE_ACTION_JOINTS)])) >= 4,
                "active frame has fewer than four action joints",
            )
            require(bool(mask[BODY_CENTER_HIPS[0]]) and bool(mask[BODY_CENTER_HIPS[1]]), "active frame lacks hips")
            active = np.asarray(xyz[frame_index, :17, :2][mask], dtype=np.float64)
            hip_center = 0.5 * (
                np.asarray(xyz[frame_index, BODY_CENTER_HIPS[0], :2], dtype=np.float64)
                + np.asarray(xyz[frame_index, BODY_CENTER_HIPS[1], :2], dtype=np.float64)
            )
            require(float(np.linalg.norm(hip_center)) <= NORMALIZATION_TOLERANCE, "body center mismatch")
            rms = float(np.sqrt(np.mean(np.sum(np.square(active), axis=1))))
            require(math.isfinite(rms) and abs(rms - 1.0) <= NORMALIZATION_TOLERANCE, "uniform RMS mismatch")
        checked += 1
        replay_rows.append(
            {
                "video_id_sha256": opaque_id,
                "track_stability_decision_sha256": hashlib.sha256(
                    json.dumps(
                        replay_decision,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest(),
                "representation_eligible": replay_decision["eligible"],
                "eligible_pair_start_count": replay_decision["eligible_pair_start_count"],
                "eligible_pair_start_count_by_variant": {
                    name: len(starts)
                    for name, starts in replay_decision[
                        "eligible_pair_starts_by_variant"
                    ].items()
                },
                "valid_frame_fraction": float(np.count_nonzero(valid) / len(valid)),
                "mean_active_joint_count": float(
                    np.mean(np.count_nonzero(joint_mask[valid], axis=1))
                ) if np.any(valid) else 0.0,
                "mean_active_action_joint_count": float(
                    np.mean(
                        np.count_nonzero(
                            joint_mask[valid][:, list(RELIABLE_ACTION_JOINTS)], axis=1
                        )
                    )
                ) if np.any(valid) else 0.0,
            }
        )

    criteria = {
        "finite_valid_coordinates": True,
        "invalid_frames_exact_zero": True,
        "padded_joints_17_to_32_exact_zero": True,
        "z_coordinates_exact_zero": True,
        "active_coco17_body_centered": True,
        "active_coco17_uniform_rms_scale": True,
        "mask_and_native_timeline_closed": True,
        "cache_sha_and_bytes_closed": True,
    }
    receipt = {
        "schema_version": 1,
        "artifact_type": "pams_cycleback_unified_2d_pose_cache_validation_receipt_v1",
        "status": "passed",
        "overall_pass": True,
        "representation": "unified_2d",
        "representation_detail": {
            "coordinates": "body-centered-uniform-rms-scale-xy-z0-v1",
            "body_center_joints": [11, 12],
            "body_center_formula": "0.5*(left_hip_xy+right_hip_xy)",
            "scale_reference": "raw-logit>2 selected COCO17 joints",
            "scale_formula": "sqrt(mean(sum((xy-hip_center)^2,axis=xy),axis=reliable_joints))",
            "weak_joint_storage": "exact-zero",
            "tolerance": NORMALIZATION_TOLERANCE,
        },
        "normalization_contract": {
            "joint_valid_source": "raw_keypoint_logit_strictly_greater_than_2",
            "body_center": "coco17_hips_11_12_midpoint",
            "scale": "unweighted_rms_of_selected_reliable_xy_about_body_center",
            "xy_scale_shared": True,
            "validation_absolute_tolerance": NORMALIZATION_TOLERANCE,
            "masked_joint_value": "exact_zero",
            "z_value": "exact_zero",
            "padded_joints_17_to_32": "exact_zero",
            "joint_valid_mask_storage": "independent_sha256_bound_sidecar_npz_v1",
        },
        "entry_count": checked,
        "replay_rows": replay_rows,
        "criteria": criteria,
        "bindings": {
            "pose_fingerprint": expected_pose_fingerprint,
            "pose_cache_snapshot_sha256": sha256_file(snapshot_path),
            "pose_cache_set_sha256": snapshot["fingerprint"],
            "raw_ledger_sha256": sha256_file(raw_ledger_path),
            "raw_extraction_authorization_sha256": expected_authorization_sha256,
            "canonical_outcome_locator": str(outcome_locator),
            "segment_index_sha256": segment_index_sha256,
            "segment_reset_policy_sha256": segment_policy_sha256,
            "joint_mask_snapshot_sha256": joint_mask_snapshot_sha256,
            "joint_mask_set_sha256": joint_mask_snapshot["fingerprint"],
            "cycleback_pair_eligibility_sha256": pair_eligibility_sha256,
            "identity_map_sha256": identity_map_sha256,
        },
        "label_free": True,
        "cycleback_representation_input_authorized": False,
        "baseline_training_authorized": False,
    }
    write_json_exclusive(output_path, receipt)
    os.chmod(output_path, 0o444)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--raw-ledger", type=Path, required=True)
    parser.add_argument("--segment-index", type=Path, required=True)
    parser.add_argument("--segment-policy", type=Path, required=True)
    parser.add_argument("--joint-mask-snapshot", type=Path, required=True)
    parser.add_argument("--pair-eligibility", type=Path, required=True)
    parser.add_argument("--identity-map", type=Path, required=True)
    parser.add_argument("--pose-fingerprint", required=True)
    parser.add_argument("--expected-entry-count", type=int, choices=(39, 337), required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = validate_unified_2d_cache_set(
            train_input_path=args.train_input,
            cache_dir=args.cache_dir,
            snapshot_path=args.snapshot,
            raw_ledger_path=args.raw_ledger,
            segment_index_path=args.segment_index,
            segment_policy_path=args.segment_policy,
            joint_mask_snapshot_path=args.joint_mask_snapshot,
            pair_eligibility_path=args.pair_eligibility,
            identity_map_path=args.identity_map,
            expected_pose_fingerprint=args.pose_fingerprint,
            expected_entry_count=args.expected_entry_count,
            authorization_path=args.authorization,
            expected_authorization_sha256=args.authorization_sha256,
            output_path=args.output,
        )
    except (OSError, TypeError, ValueError, Full337ContractError) as exc:
        print(f"v4e unified-2d cache validation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
