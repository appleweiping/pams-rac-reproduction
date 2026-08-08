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
    load_strict_json,
    require,
    sha256_file,
    write_json_exclusive,
)

from pams.config import load_config
from pams.data import (
    PoseCacheEntryReceipt,
    PoseCacheSetSnapshot,
    load_pose_cache_with_receipt,
    load_pose_input_manifest,
    pose_cache_path,
)
from pams.keypoint_single_source import (
    CYCLEBACK_PAIR_VARIANT_ALIASES,
    CYCLEBACK_PAIR_VARIANTS,
    TrackStabilityThresholds,
    _candidate_evidence_sha256,
    _canonical_sha256,
    _strict_valid_segments_with_association,
    assess_track_stability,
    association_weights_from_settings,
    body_centered_uniform_scale_xy,
    build_single_source_sequence,
    canonicalize_raw_detector_frame,
    effective_maximum_bridge_gap_frames,
    load_candidate_evidence_npz,
    load_raw_detector_evidence_npz,
    reconstruct_canonical_frame_evidence,
    select_segmented_stable_actor_paths,
)

BODY_CENTER_HIPS = (11, 12)
RELIABLE_TORSO_JOINTS = (5, 6, 11, 12)
RELIABLE_ACTION_JOINTS = (7, 8, 9, 10, 13, 14, 15, 16)
NORMALIZATION_TOLERANCE = 1.0e-5


def _empty_pair_starts() -> dict[str, list[int]]:
    return {name: [] for name in CYCLEBACK_PAIR_VARIANTS}


def _read_object(path: Path, role: str) -> Mapping[str, Any]:
    value, _ = load_strict_json(path, role=role)
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
    config_path: Path,
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
    config_sha256 = sha256_file(config_path)
    config = load_config(config_path.resolve(strict=True))
    settings = config.pose.keypoint_single_source
    require(settings is not None, "v4e single-source config missing")
    require(
        auth_bindings.get("config_file_sha256") == config_sha256
        and auth_bindings.get("config_fingerprint") == config.fingerprint
        and auth_bindings.get("pose_fingerprint") == config.pose_fingerprint
        and expected_pose_fingerprint == config.pose_fingerprint,
        "raw authorization/config binding mismatch",
    )
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
    canonical_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=expected_pose_fingerprint,
        entries=tuple(
            PoseCacheEntryReceipt(
                video_id=str(entry["video_id"]),
                cache_sha256=str(entry["cache_sha256"]),
                bytes=int(entry["bytes"]),
            )
            for entry in snapshot_entries
            if isinstance(entry, Mapping)
        ),
    )
    require(
        snapshot == canonical_snapshot.to_dict(),
        "pose-cache snapshot is not canonical or fingerprint-closed",
    )
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
    require(
        raw_thresholds == authorization.get("frozen_thresholds"),
        "raw ledger thresholds differ from authorization",
    )
    thresholds = TrackStabilityThresholds(**dict(raw_thresholds))
    segment_index_sha256 = sha256_file(segment_index_path)
    segment_policy_sha256 = sha256_file(segment_policy_path)
    require(
        ledger.get("segment_index_sha256") == segment_index_sha256
        and ledger.get("segment_reset_policy_sha256") == segment_policy_sha256,
        "ledger segment bindings mismatch",
    )
    segment_policy = _read_object(segment_policy_path, "segment reset policy")
    require(
        set(segment_policy)
        == {
            "schema_version",
            "artifact_type",
            "association_bridge_rule",
            "maximum_bridge_gap_seconds",
            "maximum_bridge_gap_frame_cap",
            "reset_identity_semantics",
            "multi_segment_representation_eligibility",
            "cross_reset_count_aggregation",
            "trainable_segment_rule",
            "cycleback_pair_rule",
            "missing_frame_bridge_for_training",
            "label_free",
        }
        and segment_policy.get("schema_version") == 1
        and segment_policy.get("artifact_type")
        == "pams_pose_recovery_v4e_segment_reset_policy_v1"
        and segment_policy.get("association_bridge_rule")
        == "min(frame_cap,floor(seconds*fps))-zero-allowed-v1"
        and segment_policy.get("maximum_bridge_gap_seconds")
        == settings.maximum_bridge_gap_seconds
        and segment_policy.get("maximum_bridge_gap_frame_cap")
        == settings.maximum_bridge_gap_frame_cap
        and segment_policy.get("multi_segment_representation_eligibility")
        == "v1-quarantine-no-cross-reset-training-or-count-aggregation"
        and segment_policy.get("missing_frame_bridge_for_training") is False
        and segment_policy.get("label_free") is True,
        "segment reset policy differs from bound v4e config",
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
    require(
        joint_mask_entries
        == sorted(joint_mask_entries, key=lambda entry: str(entry["video_id"])),
        "joint-mask snapshot entries are not canonically ordered",
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
        and pair_eligibility.get("policy")
        == "representation_authorized_exact_2w_starts_only"
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
        pair_eligibility.get("alias_semantics")
        == "listed-starts-must-be-bytewise-equal-no-consumer-inference"
        and pair_eligibility.get("frozen_thresholds") == raw_thresholds
        and pair_eligibility.get("label_free") is True,
        "pair eligibility policy/threshold binding mismatch",
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
    require(
        identity_map.get("mapping_rule") == "utf8-video-id-sha256",
        "identity map rule mismatch",
    )
    require(ledger.get("identity_map_sha256") == identity_map_sha256, "identity map binding mismatch")
    identity_entries = identity_map.get("entries")
    require(isinstance(identity_entries, list) and len(identity_entries) == expected_entry_count, "identity map entries mismatch")
    require(
        identity_entries
        == sorted(identity_entries, key=lambda entry: str(entry["video_id"])),
        "identity-map entries are not canonically ordered",
    )
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
        require(
            set(pair_row)
            == {
                "video_id",
                "native_length",
                "representation_eligible",
                "starts_by_variant",
                "base_valid_starts_by_variant",
            }
            and set(segment_row)
            == {
                "video_id",
                "native_length",
                "association_resets",
                "eligible_frame_ranges",
                "eligible_pair_starts_by_candidate",
                "representation_eligible",
                "ineligibility_reason",
                "no_unreported_internal_reset",
            },
            "pair/segment row has missing or extra fields",
        )
        cache_path = pose_cache_path(cache_dir, record.video_id).resolve(strict=True)
        require(row.get("cache_path") == str(cache_path), "cache ledger path is not canonical")
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

        candidate_locator = (
            artifact_root / f"output/raw-evidence/{opaque_id}.candidates.npz"
        )
        raw_detector_locator = (
            artifact_root / f"output/raw-evidence/{opaque_id}.raw-detector.npz"
        )
        path_locator = artifact_root / f"output/raw-evidence/{opaque_id}.path.json"
        require(
            not candidate_locator.is_symlink()
            and not raw_detector_locator.is_symlink()
            and not path_locator.is_symlink(),
            "raw evidence locator must not be a symlink",
        )
        candidate_path = candidate_locator.resolve(strict=True)
        raw_detector_path = raw_detector_locator.resolve(strict=True)
        path_path = path_locator.resolve(strict=True)
        require(
            row.get("candidate_evidence_path") == str(candidate_path)
            and row.get("raw_detector_evidence_path") == str(raw_detector_path)
            and row.get("path_segment_evidence_path") == str(path_path),
            "raw evidence ledger path is not canonical",
        )
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
            sha256_file(raw_detector_path) == row.get("raw_detector_evidence_sha256")
            and raw_detector_path.stat().st_size
            == row.get("raw_detector_evidence_bytes"),
            "raw detector artifact receipt mismatch",
        )
        require(
            stat.S_IMODE(cache_path.stat().st_mode) & 0o222 == 0
            and stat.S_IMODE(candidate_path.stat().st_mode) & 0o222 == 0
            and stat.S_IMODE(raw_detector_path.stat().st_mode) & 0o222 == 0
            and stat.S_IMODE(path_path.stat().st_mode) & 0o222 == 0,
            "cache/raw/candidate/path artifact is not sealed read-only",
        )
        path_artifact = _read_object(path_path, "path evidence")
        require(
            set(path_artifact)
            == {
                "schema_version",
                "artifact_type",
                "video_id_sha256",
                "raw_evidence",
                "eligibility_decision",
                "candidate_evidence_sha256",
                "candidate_evidence_bytes",
                "raw_detector_evidence_sha256",
                "raw_detector_evidence_bytes",
                "joint_valid_mask_sha256",
                "joint_valid_mask_bytes",
            },
            "path evidence has missing or extra fields",
        )
        raw_evidence = path_artifact.get("raw_evidence")
        recorded_decision = path_artifact.get("eligibility_decision")
        require(
            isinstance(raw_evidence, Mapping) and isinstance(recorded_decision, Mapping),
            "path evidence payload mismatch",
        )
        require(metadata.decoded_clip_frames is not None, "decoded frame metadata missing")
        decoded_frames = int(metadata.decoded_clip_frames)
        require(
            raw_evidence.get("decoded_frames") == decoded_frames
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
        require(
            path_artifact.get("raw_detector_evidence_sha256")
            == row.get("raw_detector_evidence_sha256")
            and path_artifact.get("raw_detector_evidence_bytes")
            == row.get("raw_detector_evidence_bytes"),
            "path/raw-detector binding mismatch",
        )

        raw_bundle = load_raw_detector_evidence_npz(raw_detector_path)
        require(
            len(raw_bundle.frame_offsets) == sequence.num_frames + 1,
            "raw detector timeline mismatch",
        )
        replay_all_candidates: list[np.ndarray | None] = []
        replay_candidates: list[np.ndarray | None] = []
        replay_detector_diagnostics: list[dict[str, int] | None] = []
        replay_detector_digests: list[dict[str, str] | None] = []
        replay_detector_source_indices: list[tuple[int, ...] | None] = []
        for frame_index in range(sequence.num_frames):
            start = int(raw_bundle.frame_offsets[frame_index])
            stop = int(raw_bundle.frame_offsets[frame_index + 1])
            width = int(raw_bundle.frame_dimensions[frame_index, 0])
            height = int(raw_bundle.frame_dimensions[frame_index, 1])
            if frame_index >= decoded_frames:
                require(
                    start == stop and width == 0 and height == 0,
                    "padded tail contains raw detector evidence",
                )
                replay_all_candidates.append(None)
                replay_candidates.append(None)
                replay_detector_diagnostics.append(None)
                replay_detector_digests.append(None)
                replay_detector_source_indices.append(None)
                continue
            require(width > 0 and height > 0, "decoded raw detector frame lacks dimensions")
            canonical = canonicalize_raw_detector_frame(
                labels=raw_bundle.labels[start:stop],
                scores=raw_bundle.scores[start:stop],
                boxes=raw_bundle.boxes[start:stop],
                keypoints=raw_bundle.keypoints[start:stop],
                keypoint_logits=raw_bundle.keypoint_logits[start:stop],
                image_width=width,
                image_height=height,
                settings=settings,
            )
            replay_all_candidates.append(canonical.all_eligible_candidates)
            replay_candidates.append(canonical.top_candidates)
            replay_detector_diagnostics.append(dict(canonical.diagnostics))
            replay_detector_digests.append(
                {
                    "raw_output_sha256": canonical.raw_output_sha256,
                    "canonical_eligible_sha256": canonical.canonical_eligible_sha256,
                }
            )
            replay_detector_source_indices.append(canonical.canonical_source_indices)
        all_candidates = tuple(replay_all_candidates)
        candidates = tuple(replay_candidates)
        bundle = load_candidate_evidence_npz(candidate_path)
        require(len(bundle.frame_offsets) == sequence.num_frames + 1, "candidate timeline mismatch")
        expected_offsets = np.zeros(sequence.num_frames + 1, dtype=np.int64)
        expected_rows: list[np.ndarray] = []
        for frame_index, frame_candidates in enumerate(all_candidates):
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
            "raw detector bytes do not reproduce canonical candidate artifact",
        )
        no_anchors = tuple(None for _ in candidates)
        primary_weights, secondary_weights = association_weights_from_settings(
            settings
        )
        require(
            raw_evidence.get("primary_weights") == primary_weights.to_dict()
            and raw_evidence.get("secondary_weights") == secondary_weights.to_dict(),
            "recorded association weights differ from bound config",
        )
        effective_gap = effective_maximum_bridge_gap_frames(
            fps=sequence.fps,
            maximum_seconds=settings.maximum_bridge_gap_seconds,
            frame_cap=settings.maximum_bridge_gap_frame_cap,
        )
        require(
            raw_evidence.get("effective_maximum_bridge_gap_frames") == effective_gap,
            "recorded bridge gap differs from bound config/fps",
        )
        replay_primary_bank = select_segmented_stable_actor_paths(
            candidates,
            anchor_residuals=no_anchors,
            weights=primary_weights,
            maximum_bridge_gap_frames=effective_gap,
        )
        replay_secondary_bank = select_segmented_stable_actor_paths(
            candidates,
            anchor_residuals=no_anchors,
            weights=secondary_weights,
            maximum_bridge_gap_frames=effective_gap,
        )
        replay_primary = replay_primary_bank.selected_path
        replay_secondary = replay_secondary_bank.selected_path
        replay_segments = replay_primary_bank.association_segments
        require(
            replay_segments == replay_secondary_bank.association_segments,
            "replayed segment boundaries differ",
        )
        replay_actor_margins = {
            name: {str(start): float(value) for start, value in rows.items()}
            for name, rows in (
                replay_primary_bank.selected_pair_utility_margin_by_variant.items()
            )
        }
        require(
            raw_evidence.get("identity_association_motion_free") is True
            and raw_evidence.get("primary_identity_hypothesis_table_sha256")
            == replay_primary_bank.identity_hypothesis_table_sha256
            and raw_evidence.get("secondary_identity_hypothesis_table_sha256")
            == replay_secondary_bank.identity_hypothesis_table_sha256
            and raw_evidence.get("primary_stable_track_hypotheses")
            == list(replay_primary_bank.hypothesis_rows)
            and raw_evidence.get("secondary_stable_track_hypotheses")
            == list(replay_secondary_bank.hypothesis_rows)
            and raw_evidence.get("identity_assignment_unresolvable_frames")
            == list(replay_primary_bank.identity_unresolvable_frames)
            and raw_evidence.get("selected_actor_pair_utility_margin_by_variant")
            == replay_actor_margins,
            "recorded stable-track bank differs from raw/config replay",
        )
        frame_evidence = raw_evidence.get("frame_evidence")
        require(isinstance(frame_evidence, list) and len(frame_evidence) == sequence.num_frames, "frame evidence mismatch")
        for frame_index, (frame_row, frame_candidates) in enumerate(
            zip(frame_evidence, candidates, strict=True)
        ):
            require(isinstance(frame_row, Mapping), "frame evidence row must be an object")
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
                and detector_counts.get("dropped_by_top4")
                == max(0, all_count - settings.maximum_candidates_per_frame),
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
        replay_ambiguity_by_frame = dict(
            replay_primary_bank.local_identity_gap_by_frame
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
        replay_frame_evidence = reconstruct_canonical_frame_evidence(
            candidates=candidates,
            primary=replay_primary,
            secondary=replay_secondary,
            association_segments=replay_segments,
            sequence=replay_sequence,
            ambiguity_by_frame=replay_ambiguity_by_frame,
            maximum_bridge_gap_frames=effective_gap,
            detector_diagnostics=replay_detector_diagnostics,
            keypoint_logit_threshold=settings.keypoint_logit_threshold,
            detector_frame_digests=replay_detector_digests,
            detector_frame_source_indices=replay_detector_source_indices,
            identity_alternative_reachable_by_frame=(
                replay_primary_bank.identity_alternative_reachable_by_frame
            ),
            identity_unresolvable_frames=(
                replay_primary_bank.identity_unresolvable_frames
            ),
        )
        require(
            replay_frame_evidence == frame_evidence,
            "candidate/path bytes do not reproduce canonical frame evidence",
        )
        require(
            raw_evidence.get("frame_evidence_sha256")
            == _canonical_sha256(replay_frame_evidence)
            and raw_evidence.get("selected_top4_candidate_evidence_sha256")
            == _candidate_evidence_sha256(candidates),
            "frame or selected-top4 evidence digest mismatch",
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
        replay_ambiguity_values = tuple(replay_ambiguity_by_frame.values())
        replay_continuity_steps = tuple(
            float(frame["continuity_from_previous"]["center_step_normalized_per_frame"])
            for frame in replay_frame_evidence
            if isinstance(frame.get("continuity_from_previous"), Mapping)
        )

        def percentile(values: Sequence[float], quantile: float) -> float | None:
            return (
                None
                if not values
                else float(np.quantile(np.asarray(values, dtype=np.float64), quantile))
            )

        valid_ranges = [
            {"start": start, "stop": stop, "frames": stop - start}
            for start, stop in _strict_valid_segments_with_association(
                valid,
                replay_segments,
            )
        ]
        expected_aggregates = {
            "source_frames": sequence.num_frames,
            "fps": float(sequence.fps),
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
                value is not None
                and len(value) > settings.maximum_candidates_per_frame
                for value in all_candidates
            ),
            "detector_rejected_unreliable_torso_total": sum(
                diagnostics["rejected_unreliable_torso"]
                for diagnostics in replay_detector_diagnostics
                if diagnostics is not None
            ),
            "selected_normalization_failure_frames": sum(
                index < decoded_frames
                and replay_primary.selected_indices[index] is not None
                and not bool(valid[index])
                for index in range(sequence.num_frames)
            ),
            "dual_path_comparable_frames": len(comparable),
            "dual_path_agreement": float(replay_agreement),
            "normalization_valid_mask_sha256": hashlib.sha256(
                bytes(int(value) for value in valid)
            ).hexdigest(),
            "primary_path_score": replay_primary.score,
            "primary_runner_up_score": replay_primary.runner_up_score,
            "primary_global_margin_per_observed_frame": (
                replay_primary.margin_per_observed_frame
            ),
            "primary_runner_up_available": (
                replay_primary.runner_up_score is not None
            ),
            "primary_path_sha256": replay_primary.selected_path_sha256,
            "secondary_path_sha256": replay_secondary.selected_path_sha256,
            "local_ambiguous_frame_total": len(replay_ambiguity_values),
            "local_ambiguity_gap_p10": percentile(replay_ambiguity_values, 0.1),
            "local_ambiguity_gap_median": percentile(
                replay_ambiguity_values,
                0.5,
            ),
            "continuity_step_p95": percentile(replay_continuity_steps, 0.95),
            "continuity_step_maximum": max(replay_continuity_steps, default=None),
            "maximum_bridge_gap_seconds": settings.maximum_bridge_gap_seconds,
            "maximum_bridge_gap_frame_cap": settings.maximum_bridge_gap_frame_cap,
            "effective_maximum_bridge_gap_frames": effective_gap,
            "track_segment_count": len(replay_segments),
            "track_segment_ranges": [
                {"start": segment[0], "stop": segment[-1] + 1}
                for segment in replay_segments
            ],
            "trainable_segments": valid_ranges,
            "longest_trainable_segment_frames": max(
                (row["frames"] for row in valid_ranges),
                default=0,
            ),
            "longest_track_segment_fraction": (
                max((len(segment) for segment in replay_segments), default=0)
                / sequence.num_frames
            ),
            "longest_trainable_segment_fraction": (
                max((row["frames"] for row in valid_ranges), default=0)
                / sequence.num_frames
            ),
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
        require(
            raw_evidence.get("identity_hard_edge_policy")
            == "torso-center-scale-morphology-all-within-bound-config-envelope-v1"
            and raw_evidence.get("dominant_subject_unary_policy")
            == "log(torso-geometry-scale*mean-reliable-confidence)-kprcnn-only-v2"
            and raw_evidence.get("track_segment_scope")
            == "independent-pseudotracks-after-reset-no-cross-segment-identity-claim"
            and raw_evidence.get("trainable_segment_policy")
            == "half-open-contiguous-valid-runs-no-missing-frame-or-reset-crossing-v1"
            and raw_evidence.get("all_raw_detector_outputs_persisted_for_independent_replay")
            is True,
            "recorded v4e mechanism policy differs from frozen science core",
        )
        replay_raw_evidence = dict(raw_evidence)
        replay_raw_evidence["frame_evidence"] = replay_frame_evidence
        replay_raw_evidence["dual_path_agreement"] = replay_agreement
        replay_raw_evidence["source_frames"] = sequence.num_frames
        recorded_video_evidence_sha256 = replay_raw_evidence.pop(
            "video_evidence_sha256", None
        )
        require(
            recorded_video_evidence_sha256 == _canonical_sha256(replay_raw_evidence)
            and recorded_video_evidence_sha256
            == row.get("video_evidence_sha256"),
            "canonical video evidence digest mismatch",
        )
        replay_raw_evidence["video_evidence_sha256"] = (
            recorded_video_evidence_sha256
        )
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
        expected_resets = [int(segment[0]) for segment in replay_segments[1:]]
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
            and segment_row.get("representation_eligible") == replay_decision["eligible"]
            and segment_row.get("ineligibility_reason")
            == (
                None
                if replay_decision["eligible"]
                else ",".join(replay_decision["quarantine_reasons"])
            )
            and segment_row.get("no_unreported_internal_reset") is True,
            "segment index replay mismatch",
        )

        joint_mask_locator = (
            artifact_root / f"output/joint-mask/{opaque_id}.joint-mask.npz"
        )
        require(
            not joint_mask_locator.is_symlink(),
            "joint-mask locator must not be a symlink",
        )
        joint_mask_path = joint_mask_locator.resolve(strict=True)
        require(
            row.get("joint_valid_mask_path") == str(joint_mask_path),
            "joint-mask ledger path is not canonical",
        )
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
                "decoded_frames": decoded_frames,
                "fps": float(sequence.fps),
                "raw_detector_evidence_sha256": row[
                    "raw_detector_evidence_sha256"
                ],
                "candidate_evidence_sha256": row["candidate_evidence_sha256"],
                "path_segment_evidence_sha256": row[
                    "path_segment_evidence_sha256"
                ],
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
    parser.add_argument("--config", type=Path, required=True)
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
            config_path=args.config,
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
