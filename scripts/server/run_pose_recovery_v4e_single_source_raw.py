#!/usr/bin/env python3
"""Extract fresh single-source Keypoint R-CNN evidence for canonical train337."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pose_recovery_v4d_full337_contract import (
    MODEL_ASSET_SHA256,
    TRAIN337_COMMITMENT_SHA256,
    TRAIN337_IDENTITY_SHA256,
    TRAIN337_SIDECAR_SHA256,
    V4A_CACHE_SET_SHA256,
    V4A_LEDGER_SHA256,
    V4A_PAIRED_GATE_SHA256,
    V4A_POSE_FINGERPRINT,
    Full337ContractError,
    require,
    sha256_file,
    validate_v4a_inputs,
    write_json_exclusive,
)

from pams.config import load_config
from pams.data import (
    load_pose_cache,
    load_pose_cache_set,
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_cache_path,
    pose_input_identity_sha256,
    validate_pose_input_binding,
    write_pose_cache,
)
from pams.keypoint_recovery import KeypointRecoveryError, load_keypointrcnn_runtime
from pams.keypoint_single_source import (
    CYCLEBACK_PAIR_VARIANT_ALIASES,
    CYCLEBACK_PAIR_VARIANTS,
    V4E_PREPROCESSING_REVISION,
    KeypointSingleSourceError,
    TrackStabilityThresholds,
    assess_track_stability,
    extract_single_source_video,
    write_candidate_evidence_npz,
)

APPROVED_RAW_EXTRACTION_SECURE_LAUNCH_AUTHORITY_SHA256 = ""


def _empty_pair_starts() -> dict[str, list[int]]:
    return {name: [] for name in CYCLEBACK_PAIR_VARIANTS}


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, Mapping), f"JSON artifact must be an object: {path}")
    return value


def _validate_raw_authorization(
    path: Path,
    *,
    expected_sha256: str,
    source_revision: str,
    container_image_id: str,
    config_file_sha256: str,
    config_fingerprint: str,
    pose_fingerprint: str,
) -> Mapping[str, Any]:
    require(sha256_file(path) == expected_sha256, "raw authorization SHA mismatch")
    value = _read_json(path)
    require(
        value.get("artifact_type")
        == "pams_pose_recovery_v4e_raw_extraction_authorization_v1",
        "raw authorization type mismatch",
    )
    require(value.get("status") == "authorized", "raw extraction is not authorized")
    require(value.get("label_free") is True, "raw authorization must be label-free")
    require(value.get("raw_extraction_authorized") is True, "raw extraction not authorized")
    require(
        value.get("authorization_scope")
        in {
            "canonical_train337_same39_source_video_kprcnn_raw_evidence_only",
            "canonical_train337_source_video_kprcnn_raw_evidence_only",
        },
        "raw authorization scope mismatch",
    )
    for forbidden in (
        "cycleback_representation_input_authorized",
        "baseline_training_authorized",
        "scientific_training_authorized",
        "dev_evaluation_authorized",
        "sealed_test_authorized",
    ):
        require(value.get(forbidden) is False, f"raw authorization grants {forbidden}")
    require(value.get("source_revision") == source_revision, "authorization source mismatch")
    require(value.get("container_image_id") == container_image_id, "authorization image mismatch")
    require(value.get("v4d_denied_cache_consumed") is False, "authorization consumes v4d")
    bindings = value.get("bindings")
    require(isinstance(bindings, Mapping), "raw authorization bindings missing")
    outcome_locator = Path(str(bindings["canonical_outcome_locator"]))
    expected_name = (
        "v4e-same39-raw.authorization.json"
        if value.get("authorization_scope")
        == "canonical_train337_same39_source_video_kprcnn_raw_evidence_only"
        else "v4e-full337-raw.authorization.json"
    )
    require(
        path.resolve(strict=True) == outcome_locator / "authorization" / expected_name
        and stat.S_IMODE(path.stat().st_mode) & 0o222 == 0,
        "raw authorization canonical locator/sealing mismatch",
    )
    registry_path = outcome_locator / "attempt.reservation.json"
    require(
        registry_path.resolve(strict=True) == registry_path
        and sha256_file(registry_path)
        == bindings["canonical_registry_reservation_sha256"]
        and stat.S_IMODE(registry_path.stat().st_mode) & 0o222 == 0
        and bindings["canonical_registry_artifact_type"]
        == "pams_pose_recovery_v4e_canonical_outcome_reservation_v1",
        "canonical registry reservation binding mismatch",
    )
    require(bindings.get("config_file_sha256") == config_file_sha256, "config SHA binding mismatch")
    require(bindings.get("config_fingerprint") == config_fingerprint, "config binding mismatch")
    require(bindings.get("pose_fingerprint") == pose_fingerprint, "pose binding mismatch")
    require(bindings.get("model_asset_sha256") == MODEL_ASSET_SHA256, "asset binding mismatch")
    require(
        bindings.get("train337_sidecar_sha256") == TRAIN337_SIDECAR_SHA256,
        "sidecar binding mismatch",
    )
    require(
        bindings.get("train337_commitment_sha256") == TRAIN337_COMMITMENT_SHA256,
        "commitment binding mismatch",
    )
    require(
        bindings.get("train337_identity_sha256") == TRAIN337_IDENTITY_SHA256,
        "identity binding mismatch",
    )
    require(bindings.get("v4a_cache_set_sha256") == V4A_CACHE_SET_SHA256, "v4a set mismatch")
    return value


def run_raw_train337(
    *,
    source_revision: str,
    container_image_id: str,
    expected_authorization_sha256: str,
    authorization_path: Path,
    config_path: Path,
    train_input_path: Path,
    train_commitment_path: Path,
    video_root: Path,
    model_asset_path: Path,
    v4a_cache_dir: Path,
    v4a_ledger_path: Path,
    v4a_paired_gate_path: Path,
    output_cache_dir: Path,
    output_evidence_dir: Path,
    output_joint_mask_dir: Path,
    output_ledger_path: Path,
    output_snapshot_path: Path,
    output_segment_index_path: Path,
    output_segment_policy_path: Path,
    output_joint_mask_snapshot_path: Path,
    output_pair_eligibility_path: Path,
    output_identity_map_path: Path,
    same39_raw_ledger_path: Path | None,
    same39_cache_dir: Path | None,
    same39_evidence_dir: Path | None,
    same39_joint_mask_dir: Path | None,
) -> dict[str, Any]:
    """Run fresh extraction from canonical videos; never reuse v4d bytes."""

    require(
        len(APPROVED_RAW_EXTRACTION_SECURE_LAUNCH_AUTHORITY_SHA256) == 64,
        "v4e raw extraction is disabled pending monotonic secure authority",
    )

    require(
        os.environ.get("PAMS_CONTAINER_SOURCE_REVISION") == source_revision,
        "container source revision mismatch",
    )
    require(
        os.environ.get("PAMS_CONTAINER_IMAGE_ID") == container_image_id,
        "container image ID mismatch",
    )
    config_file_sha256 = sha256_file(config_path)
    config = load_config(config_path.resolve(strict=True))
    require(
        config.pose.preprocessing_revision == V4E_PREPROCESSING_REVISION,
        "raw runner config is not v4e",
    )
    settings = config.pose.keypoint_single_source
    require(settings is not None, "v4e settings missing")
    authorization = _validate_raw_authorization(
        authorization_path,
        expected_sha256=expected_authorization_sha256,
        source_revision=source_revision,
        container_image_id=container_image_id,
        config_file_sha256=config_file_sha256,
        config_fingerprint=config.fingerprint,
        pose_fingerprint=config.pose_fingerprint,
    )
    frozen_thresholds = authorization.get("frozen_thresholds")
    require(isinstance(frozen_thresholds, Mapping), "authorization thresholds missing")
    thresholds = TrackStabilityThresholds(**dict(frozen_thresholds))
    require(sha256_file(model_asset_path) == MODEL_ASSET_SHA256, "model asset SHA mismatch")

    manifest = load_pose_input_manifest(train_input_path, validate_exact=True)
    commitment = load_pose_input_commitment(train_commitment_path)
    sidecar_sha256 = sha256_file(train_input_path)
    commitment_sha256 = sha256_file(train_commitment_path)
    validate_pose_input_binding(manifest, commitment, sidecar_sha256=sidecar_sha256)
    require(manifest.split == "train" and len(manifest.records) == 337, "expected train337")
    require(sidecar_sha256 == TRAIN337_SIDECAR_SHA256, "sidecar SHA mismatch")
    require(commitment_sha256 == TRAIN337_COMMITMENT_SHA256, "commitment SHA mismatch")
    identity_sha256 = pose_input_identity_sha256(manifest.records)
    require(identity_sha256 == TRAIN337_IDENTITY_SHA256, "identity mismatch")
    v4a = validate_v4a_inputs(
        manifest.records,
        cache_dir=v4a_cache_dir,
        ledger_path=v4a_ledger_path,
        paired_gate_path=v4a_paired_gate_path,
    )
    require(v4a.snapshot.fingerprint == V4A_CACHE_SET_SHA256, "v4a cache set mismatch")

    scope = str(authorization["authorization_scope"])
    if scope == "canonical_train337_same39_source_video_kprcnn_raw_evidence_only":
        selected_hashes = authorization.get("selected_video_id_sha256")
        require(
            isinstance(selected_hashes, list)
            and len(selected_hashes) == 39
            and len(set(selected_hashes)) == 39,
            "same39 authorization selection mismatch",
        )
        selected_records = tuple(
            record
            for record in manifest.records
            if hashlib.sha256(record.video_id.encode("utf-8")).hexdigest()
            in set(selected_hashes)
        )
        require(len(selected_records) == 39, "same39 authorization lost records")
    else:
        selected_records = tuple(manifest.records)
    expected_total = len(selected_records)
    outcome_locator = Path(str(authorization["bindings"]["canonical_outcome_locator"]))
    artifact_root = (
        outcome_locator / "pilot/same39" if expected_total == 39 else outcome_locator
    )
    expected_outputs = {
        "cache": artifact_root / "pose-cache",
        "evidence": artifact_root / "output/raw-evidence",
        "joint_mask": artifact_root / "output/joint-mask",
        "ledger": artifact_root / "output/raw-ledger.json",
        "snapshot": artifact_root / "output/pose-cache-set.snapshot.json",
        "segment_index": artifact_root / "output/segment-index.json",
        "segment_policy": artifact_root / "output/segment-reset.policy.json",
        "joint_mask_snapshot": artifact_root / "output/joint-mask-set.snapshot.json",
        "pair_eligibility": artifact_root / "output/cycleback-pair-eligibility.json",
        "identity_map": artifact_root / "output/identity-map.json",
    }
    provided_outputs = {
        "cache": output_cache_dir,
        "evidence": output_evidence_dir,
        "joint_mask": output_joint_mask_dir,
        "ledger": output_ledger_path,
        "snapshot": output_snapshot_path,
        "segment_index": output_segment_index_path,
        "segment_policy": output_segment_policy_path,
        "joint_mask_snapshot": output_joint_mask_snapshot_path,
        "pair_eligibility": output_pair_eligibility_path,
        "identity_map": output_identity_map_path,
    }
    require(
        all(path.resolve(strict=False) == expected_outputs[name] for name, path in provided_outputs.items()),
        "raw extraction output locator contract mismatch",
    )
    reused_same39_rows: dict[str, Mapping[str, Any]] = {}
    same39_cache_root: Path | None = None
    same39_evidence_root: Path | None = None
    same39_joint_mask_root: Path | None = None
    if expected_total == 337:
        require(authorization.get("same39_bytewise_reuse_required") is True, "full337 must reuse same39")
        require(
            same39_raw_ledger_path is not None
            and same39_cache_dir is not None
            and same39_evidence_dir is not None
            and same39_joint_mask_dir is not None,
            "full337 same39 predecessor paths missing",
        )
        same39_artifact_root = outcome_locator / "pilot/same39"
        require(
            same39_raw_ledger_path.resolve(strict=True)
            == same39_artifact_root / "output/raw-ledger.json"
            and same39_cache_dir.resolve(strict=True)
            == same39_artifact_root / "pose-cache"
            and same39_evidence_dir.resolve(strict=True)
            == same39_artifact_root / "output/raw-evidence"
            and same39_joint_mask_dir.resolve(strict=True)
            == same39_artifact_root / "output/joint-mask",
            "full337 same39 predecessor locator mismatch",
        )
        require(
            sha256_file(same39_raw_ledger_path)
            == authorization["bindings"]["same39_raw_ledger_sha256"],
            "same39 predecessor ledger SHA mismatch",
        )
        same39_ledger = _read_json(same39_raw_ledger_path)
        same39_values = same39_ledger.get("caches")
        require(isinstance(same39_values, list) and len(same39_values) == 39, "same39 rows missing")
        reused_same39_rows = {
            str(value["video_id"]): value
            for value in same39_values
            if isinstance(value, Mapping)
        }
        require(len(reused_same39_rows) == 39, "same39 predecessor identity mismatch")
        same39_cache_root = same39_cache_dir.resolve(strict=True)
        same39_evidence_root = same39_evidence_dir.resolve(strict=True)
        same39_joint_mask_root = same39_joint_mask_dir.resolve(strict=True)
    runtime = load_keypointrcnn_runtime(model_asset_path, settings=settings)
    root = video_root.resolve(strict=True)
    base_root = v4a_cache_dir.resolve(strict=True)
    output_root = output_cache_dir.resolve(strict=True)
    evidence_root = output_evidence_dir.resolve(strict=True)
    joint_mask_root = output_joint_mask_dir.resolve(strict=True)
    base_receipts = {
        entry.video_id: (entry.cache_sha256, entry.bytes) for entry in v4a.snapshot.entries
    }
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for offset, record in enumerate(selected_records, start=1):
        try:
            if record.video_id in reused_same39_rows:
                source_row = reused_same39_rows[record.video_id]
                opaque_id = hashlib.sha256(record.video_id.encode("utf-8")).hexdigest()
                require(
                    same39_cache_root is not None
                    and same39_evidence_root is not None
                    and same39_joint_mask_root is not None,
                    "same39 predecessor roots unavailable",
                )
                source_cache = pose_cache_path(same39_cache_root, record.video_id).resolve(strict=True)
                target_cache = pose_cache_path(output_root, record.video_id)
                source_candidate = (same39_evidence_root / f"{opaque_id}.candidates.npz").resolve(strict=True)
                source_path = (same39_evidence_root / f"{opaque_id}.path.json").resolve(strict=True)
                source_joint = (same39_joint_mask_root / f"{opaque_id}.joint-mask.npz").resolve(strict=True)
                target_candidate = evidence_root / source_candidate.name
                target_path = evidence_root / source_path.name
                target_joint = joint_mask_root / source_joint.name
                for source_artifact, target_artifact, sha_key, bytes_key in (
                    (source_cache, target_cache, "cache_sha256", "cache_bytes"),
                    (source_candidate, target_candidate, "candidate_evidence_sha256", "candidate_evidence_bytes"),
                    (source_path, target_path, "path_segment_evidence_sha256", "path_segment_evidence_bytes"),
                    (source_joint, target_joint, "joint_valid_mask_sha256", "joint_valid_mask_bytes"),
                ):
                    require(sha256_file(source_artifact) == source_row[sha_key], "same39 artifact SHA mismatch")
                    require(source_artifact.stat().st_size == source_row[bytes_key], "same39 artifact bytes mismatch")
                    shutil.copyfile(source_artifact, target_artifact)
                    require(sha256_file(target_artifact) == source_row[sha_key], "same39 bytewise reuse failed")
                copied_row = dict(source_row)
                copied_row.update(
                    {
                        "cache_path": str(target_cache.resolve()),
                        "candidate_evidence_path": str(target_candidate.resolve()),
                        "path_segment_evidence_path": str(target_path.resolve()),
                        "joint_valid_mask_path": str(target_joint.resolve()),
                        "cache_origin": "bytewise-reused-sealed-v4e-same39",
                    }
                )
                rows.append(copied_row)
                print(json.dumps({"reused_same39": offset, "total": expected_total}, sort_keys=True), flush=True)
                continue
            video_path = (root / record.video_path).resolve(strict=True)
            video_path.relative_to(root)
            require(sha256_file(video_path) == record.video_sha256, "source video SHA mismatch")
            base_path = pose_cache_path(base_root, record.video_id).resolve(strict=True)
            base_sha, base_bytes = base_receipts[record.video_id]
            require(sha256_file(base_path) == base_sha, "v4a cache SHA mismatch")
            require(base_path.stat().st_size == base_bytes, "v4a cache byte count mismatch")
            base_sequence, base_metadata = load_pose_cache(
                base_path,
                expected_video_sha256=record.video_sha256,
                expected_pose_fingerprint=V4A_POSE_FINGERPRINT,
                expected_annotation_sha256=record.annotation_sha256,
                expected_clip_start_frame=record.clip_start_frame,
                expected_clip_end_frame=record.clip_end_frame,
                validate_clip_provenance=True,
            )
            require(
                base_metadata.decoded_clip_frames is not None
                and base_metadata.expected_clip_frames == base_sequence.num_frames
                and base_metadata.padded_tail_frames
                == base_sequence.num_frames - int(base_metadata.decoded_clip_frames),
                "sealed v4a decoded/padded timeline metadata is incomplete",
            )
            decoded_frames = int(base_metadata.decoded_clip_frames)
            sequence, evidence, candidate_bundle = extract_single_source_video(
                video_path,
                video_id=record.video_id,
                clip_start_frame=int(record.clip_start_frame),
                clip_end_frame=int(record.clip_end_frame),
                base_sequence=base_sequence,
                base_decoded_frames=decoded_frames,
                expected_video_sha256=record.video_sha256,
                runtime=runtime,
                settings=settings,
            )
            opaque_id = hashlib.sha256(record.video_id.encode("utf-8")).hexdigest()
            candidate_evidence_path = evidence_root / f"{opaque_id}.candidates.npz"
            joint_mask_path = joint_mask_root / f"{opaque_id}.joint-mask.npz"
            path_evidence_path = evidence_root / f"{opaque_id}.path.json"
            write_candidate_evidence_npz(candidate_evidence_path, candidate_bundle)
            joint_mask = np.zeros((sequence.num_frames, 17), dtype=np.bool_)
            for frame_row in evidence["frame_evidence"]:
                selected_index = frame_row["primary_index"]
                if selected_index is None or not frame_row["raw_normalization_eligible"]:
                    continue
                raw_logits = frame_row["candidates"][selected_index]["raw_keypoint_logits"]
                require(raw_logits is not None and len(raw_logits) == 17, "joint logits missing")
                joint_mask[frame_row["frame_index"]] = np.asarray(raw_logits) > settings.keypoint_logit_threshold
            with joint_mask_path.open("xb") as handle:
                np.savez_compressed(
                    handle,
                    schema_version=np.asarray(1, dtype=np.int64),
                    joint_valid_mask=joint_mask,
                )
            decision = assess_track_stability(
                evidence,
                thresholds=thresholds,
                same_source_period_supported=False,
            )
            path_payload = {
                "schema_version": 1,
                "artifact_type": "pams_pose_recovery_v4e_path_segment_evidence_v1",
                "video_id_sha256": opaque_id,
                "raw_evidence": evidence,
                "eligibility_decision": decision,
                "candidate_evidence_sha256": sha256_file(candidate_evidence_path),
                "candidate_evidence_bytes": candidate_evidence_path.stat().st_size,
                "joint_valid_mask_sha256": sha256_file(joint_mask_path),
                "joint_valid_mask_bytes": joint_mask_path.stat().st_size,
            }
            write_json_exclusive(path_evidence_path, path_payload)
            target = pose_cache_path(output_root, record.video_id)
            metadata = write_pose_cache(
                target,
                sequence,
                video_sha256=record.video_sha256,
                pose_fingerprint=config.pose_fingerprint,
                pose_model=settings.model_id,
                annotation_sha256=record.annotation_sha256,
                clip_start_frame=record.clip_start_frame,
                clip_end_frame=record.clip_end_frame,
                decoded_clip_frames=decoded_frames,
                expected_clip_frames=int(record.clip_end_frame) - int(record.clip_start_frame),
                padded_tail_frames=base_metadata.padded_tail_frames,
                incomplete_clip_policy=base_metadata.incomplete_clip_policy,
            )
            rows.append(
                {
                    "video_id": record.video_id,
                    "video_id_sha256": opaque_id,
                    "cache_path": str(target.resolve()),
                    "cache_sha256": sha256_file(target),
                    "cache_bytes": target.stat().st_size,
                    "video_sha256": record.video_sha256,
                    "pose_fingerprint": config.pose_fingerprint,
                    "cached_frames": sequence.num_frames,
                    "cached_valid_frames": int(np.count_nonzero(sequence.valid_mask)),
                    "annotation_sha256": metadata.annotation_sha256,
                    "clip_start_frame": metadata.clip_start_frame,
                    "clip_end_frame": metadata.clip_end_frame,
                    "base_v4a_cache_sha256": base_sha,
                    "candidate_evidence_path": str(candidate_evidence_path.resolve()),
                    "candidate_evidence_sha256": sha256_file(candidate_evidence_path),
                    "candidate_evidence_bytes": candidate_evidence_path.stat().st_size,
                    "joint_valid_mask_path": str(joint_mask_path.resolve()),
                    "joint_valid_mask_sha256": sha256_file(joint_mask_path),
                    "joint_valid_mask_bytes": joint_mask_path.stat().st_size,
                    "path_segment_evidence_path": str(path_evidence_path.resolve()),
                    "path_segment_evidence_sha256": sha256_file(path_evidence_path),
                    "path_segment_evidence_bytes": path_evidence_path.stat().st_size,
                    "video_evidence_sha256": evidence["video_evidence_sha256"],
                    "eligibility_decision": decision,
                    "association_reset_frames": [
                        segment["start"]
                        for segment in evidence["track_segment_ranges"][1:]
                    ],
                    "cache_origin": "fresh-canonical-source-kprcnn",
                }
            )
            print(json.dumps({"completed": offset, "total": expected_total}, sort_keys=True), flush=True)
        except (
            OSError,
            TypeError,
            ValueError,
            KeypointRecoveryError,
            KeypointSingleSourceError,
            Full337ContractError,
        ) as exc:
            failures.append(
                {
                    "video_id_sha256": hashlib.sha256(record.video_id.encode("utf-8")).hexdigest(),
                    "phase": "fresh-v4e-single-source-extraction",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            print(json.dumps({"failed": offset, "error_type": type(exc).__name__}), flush=True)

    snapshot_payload = None
    reused_same39_total = sum(
        row.get("cache_origin") == "bytewise-reused-sealed-v4e-same39" for row in rows
    )
    fresh_total = sum(
        row.get("cache_origin") == "fresh-canonical-source-kprcnn" for row in rows
    )
    if len(rows) == expected_total and not failures:
        if expected_total == 337:
            require(reused_same39_total == 39 and fresh_total == 298, "full337 must be exact 39+298")
        else:
            require(reused_same39_total == 0 and fresh_total == 39, "same39 must be fresh")
    segment_policy = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_segment_reset_policy_v1",
        "association_bridge_rule": "min(frame_cap,floor(seconds*fps))-zero-allowed-v1",
        "maximum_bridge_gap_seconds": settings.maximum_bridge_gap_seconds,
        "maximum_bridge_gap_frame_cap": settings.maximum_bridge_gap_frame_cap,
        "reset_identity_semantics": "each-association-segment-is-an-independent-pseudotrack",
        "cross_reset_count_aggregation": "undefined-requires-separate-preregistered-policy",
        "trainable_segment_rule": "exact-prevalidated-2W-pair-spans-only-v1",
        "cycleback_pair_rule": "consume-only-listed-start-stop-without-expansion",
        "missing_frame_bridge_for_training": False,
        "label_free": True,
    }
    write_json_exclusive(output_segment_policy_path, segment_policy)
    segment_policy_sha256 = sha256_file(output_segment_policy_path)
    segment_index = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_segment_index_v1",
        "segment_reset_policy_sha256": segment_policy_sha256,
        "entry_count": len(rows),
        "entries": [
            {
                "video_id": row["video_id_sha256"],
                "native_length": row["cached_frames"],
                "association_resets": row["association_reset_frames"],
                "eligible_frame_ranges": row["eligibility_decision"][
                    "track_stability_frame_ranges"
                ] if row["eligibility_decision"]["eligible"] else [],
                "eligible_pair_starts_by_candidate": (
                    row["eligibility_decision"]["eligible_pair_starts_by_variant"]
                    if row["eligibility_decision"]["eligible"]
                    else _empty_pair_starts()
                ),
                "representation_eligible": row["eligibility_decision"]["eligible"],
                "ineligibility_reason": (
                    None
                    if row["eligibility_decision"]["eligible"]
                    else ",".join(row["eligibility_decision"]["quarantine_reasons"])
                ),
                "no_unreported_internal_reset": True,
            }
            for row in rows
        ],
        "all_resets_explicit_and_invalid_or_segment_sidecar_complete": True,
    }
    write_json_exclusive(output_segment_index_path, segment_index)
    segment_index_sha256 = sha256_file(output_segment_index_path)
    joint_mask_entries = sorted(
        (
            {
                "video_id": row["video_id"],
                "sidecar_sha256": row["joint_valid_mask_sha256"],
                "bytes": row["joint_valid_mask_bytes"],
            }
            for row in rows
        ),
        key=lambda entry: entry["video_id"],
    )
    joint_mask_identity = {
        "schema_version": 1,
        "entries": joint_mask_entries,
    }
    joint_mask_snapshot = {
        **joint_mask_identity,
        "artifact_type": "pams_pose_recovery_v4e_joint_mask_set_snapshot_v1",
        "entry_count": len(joint_mask_entries),
        "fingerprint": hashlib.sha256(
            json.dumps(
                joint_mask_identity,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
    }
    write_json_exclusive(output_joint_mask_snapshot_path, joint_mask_snapshot)
    joint_mask_snapshot_sha256 = sha256_file(output_joint_mask_snapshot_path)
    pair_eligibility = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_cycleback_pair_eligibility_v1",
        "policy": "representation_authorized_exact_2w_starts_only",
        "start_grid_policy": "native_zero_based_start_mod_hop_equals_zero",
        "variant_geometry": {
            name: {"window_frames": shape[0], "hop_frames": shape[1]}
            for name, shape in CYCLEBACK_PAIR_VARIANTS.items()
        },
        "frozen_variant_aliases": dict(CYCLEBACK_PAIR_VARIANT_ALIASES),
        "alias_semantics": "listed-starts-must-be-bytewise-equal-no-consumer-inference",
        "entry_count": len(rows),
        "frozen_thresholds": dict(frozen_thresholds),
        "entries": [
            {
                "video_id": row["video_id_sha256"],
                "native_length": row["cached_frames"],
                "representation_eligible": row["eligibility_decision"]["eligible"],
                "starts_by_variant": (
                    row["eligibility_decision"]["eligible_pair_starts_by_variant"]
                    if row["eligibility_decision"]["eligible"]
                    else _empty_pair_starts()
                ),
                "base_valid_starts_by_variant": (
                    row["eligibility_decision"]["base_valid_pair_starts_by_variant"]
                    if row["eligibility_decision"]["eligible"]
                    else _empty_pair_starts()
                ),
            }
            for row in rows
        ],
        "consumer_may_expand_starts": False,
        "label_free": True,
    }
    write_json_exclusive(output_pair_eligibility_path, pair_eligibility)
    pair_eligibility_sha256 = sha256_file(output_pair_eligibility_path)
    identity_map = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_identity_map_v1",
        "entry_count": len(rows),
        "entries": sorted(
            (
                {"video_id": row["video_id"], "video_id_sha256": row["video_id_sha256"]}
                for row in rows
            ),
            key=lambda entry: entry["video_id"],
        ),
        "mapping_rule": "utf8-video-id-sha256",
    }
    write_json_exclusive(output_identity_map_path, identity_map)
    identity_map_sha256 = sha256_file(output_identity_map_path)
    if len(rows) == expected_total and not failures:
        _, snapshot = load_pose_cache_set(
            selected_records,
            cache_dir=output_root,
            pose_fingerprint=config.pose_fingerprint,
            materialize_sequences=False,
        )
        snapshot_payload = snapshot.to_dict()
        write_json_exclusive(output_snapshot_path, snapshot_payload)
    ledger: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": (
            "pams_pose_recovery_v4e_same39_raw_ledger_v1"
            if expected_total == 39
            else "pams_pose_recovery_v4e_train337_raw_ledger_v1"
        ),
        "status": "complete" if len(rows) == expected_total and not failures else "failed",
        "source_revision": source_revision,
        "container_image_id": container_image_id,
        "label_free": True,
        "input_kind": "canonical_label_free_train337_source_videos",
        "config_file_sha256": config_file_sha256,
        "config_fingerprint": config.fingerprint,
        "pose_fingerprint": config.pose_fingerprint,
        "frozen_thresholds": dict(frozen_thresholds),
        "raw_extraction_authorization_sha256": expected_authorization_sha256,
        "same39_gate_sha256": authorization["bindings"].get("same39_gate_sha256"),
        "synthetic_threshold_receipt_sha256": authorization["bindings"][
            "synthetic_threshold_receipt_sha256"
        ],
        "model_asset_sha256": MODEL_ASSET_SHA256,
        "sidecar_sha256": sidecar_sha256,
        "commitment_file_sha256": commitment_sha256,
        "identity_sha256": identity_sha256,
        "v4a": {
            "evidence_scope": "processed-shape-anchor-only-not-output-coordinates",
            "pose_fingerprint": V4A_POSE_FINGERPRINT,
            "ledger_sha256": V4A_LEDGER_SHA256,
            "paired_gate_sha256": V4A_PAIRED_GATE_SHA256,
            "cache_set_sha256": v4a.snapshot.fingerprint,
        },
        "v4d_denied_cache": {"consumed": False, "authority_accepted": False},
        "selected": expected_total,
        "completed": len(rows),
        "failed": len(failures),
        "reused_same39": reused_same39_total,
        "fresh_extracted": fresh_total,
        "successful_cache_snapshot": snapshot_payload,
        "segment_reset_policy_sha256": segment_policy_sha256,
        "segment_index_sha256": segment_index_sha256,
        "joint_mask_snapshot_sha256": joint_mask_snapshot_sha256,
        "joint_mask_set_sha256": joint_mask_snapshot["fingerprint"],
        "cycleback_pair_eligibility_sha256": pair_eligibility_sha256,
        "identity_map_sha256": identity_map_sha256,
        "caches": rows,
        "failures": failures,
        "raw_extraction_only": True,
        "cycleback_representation_input_authorized": False,
        "baseline_training_authorized": False,
    }
    write_json_exclusive(output_ledger_path, ledger)
    sealed_paths = [
        output_ledger_path,
        output_snapshot_path,
        output_segment_index_path,
        output_segment_policy_path,
        output_joint_mask_snapshot_path,
        output_pair_eligibility_path,
        output_identity_map_path,
    ]
    for row in rows:
        sealed_paths.extend(
            (
                Path(str(row["cache_path"])),
                Path(str(row["candidate_evidence_path"])),
                Path(str(row["path_segment_evidence_path"])),
                Path(str(row["joint_valid_mask_path"])),
            )
        )
    for artifact_path in sealed_paths:
        require(artifact_path.exists() and artifact_path.is_file(), "raw artifact missing before sealing")
        os.chmod(artifact_path, 0o444)
        require(
            stat.S_IMODE(artifact_path.stat().st_mode) & 0o222 == 0,
            "raw artifact retained write permission after sealing",
        )
    return {
        "selected": expected_total,
        "completed": len(rows),
        "failed": len(failures),
        "cache_set_sha256": (
            None if snapshot_payload is None else snapshot_payload["fingerprint"]
        ),
        "cycleback_representation_input_authorized": False,
        "baseline_training_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-commitment", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model-asset", type=Path, required=True)
    parser.add_argument("--v4a-cache-dir", type=Path, required=True)
    parser.add_argument("--v4a-ledger", type=Path, required=True)
    parser.add_argument("--v4a-paired-gate", type=Path, required=True)
    parser.add_argument("--output-cache-dir", type=Path, required=True)
    parser.add_argument("--output-evidence-dir", type=Path, required=True)
    parser.add_argument("--output-joint-mask-dir", type=Path, required=True)
    parser.add_argument("--output-ledger", type=Path, required=True)
    parser.add_argument("--output-snapshot", type=Path, required=True)
    parser.add_argument("--output-segment-index", type=Path, required=True)
    parser.add_argument("--output-segment-policy", type=Path, required=True)
    parser.add_argument("--output-joint-mask-snapshot", type=Path, required=True)
    parser.add_argument("--output-pair-eligibility", type=Path, required=True)
    parser.add_argument("--output-identity-map", type=Path, required=True)
    parser.add_argument("--same39-raw-ledger", type=Path)
    parser.add_argument("--same39-cache-dir", type=Path)
    parser.add_argument("--same39-evidence-dir", type=Path)
    parser.add_argument("--same39-joint-mask-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_raw_train337(
            source_revision=args.source_revision,
            container_image_id=args.container_image_id,
            expected_authorization_sha256=args.authorization_sha256,
            authorization_path=args.authorization,
            config_path=args.config,
            train_input_path=args.train_input,
            train_commitment_path=args.train_commitment,
            video_root=args.video_root,
            model_asset_path=args.model_asset,
            v4a_cache_dir=args.v4a_cache_dir,
            v4a_ledger_path=args.v4a_ledger,
            v4a_paired_gate_path=args.v4a_paired_gate,
            output_cache_dir=args.output_cache_dir,
            output_evidence_dir=args.output_evidence_dir,
            output_joint_mask_dir=args.output_joint_mask_dir,
            output_ledger_path=args.output_ledger,
            output_snapshot_path=args.output_snapshot,
            output_segment_index_path=args.output_segment_index,
            output_segment_policy_path=args.output_segment_policy,
            output_joint_mask_snapshot_path=args.output_joint_mask_snapshot,
            output_pair_eligibility_path=args.output_pair_eligibility,
            output_identity_map_path=args.output_identity_map,
            same39_raw_ledger_path=args.same39_raw_ledger,
            same39_cache_dir=args.same39_cache_dir,
            same39_evidence_dir=args.same39_evidence_dir,
            same39_joint_mask_dir=args.same39_joint_mask_dir,
        )
    except (
        OSError,
        TypeError,
        ValueError,
        KeypointRecoveryError,
        KeypointSingleSourceError,
        Full337ContractError,
    ) as exc:
        print(f"v4e raw extraction failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["failed"] == 0 and result["completed"] == result["selected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
