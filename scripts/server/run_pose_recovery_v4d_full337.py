#!/usr/bin/env python3
"""Extract exact train337 v4d poses after a frozen v2 compute authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pose_recovery_v4d_full337_contract import (
    CONTAINER_IMAGE_ID,
    MODEL_ASSET_SHA256,
    TRAIN337_COMMITMENT_SHA256,
    TRAIN337_IDENTITY_SHA256,
    TRAIN337_SIDECAR_SHA256,
    V4A_LEDGER_SHA256,
    V4A_PAIRED_GATE_SHA256,
    V4A_POSE_FINGERPRINT,
    V4D_CONFIG_FILE_SHA256,
    V4D_CONFIG_FINGERPRINT,
    V4D_POSE_FINGERPRINT,
    Full337ContractError,
    require,
    sha256_file,
    stable_file_bytes,
    validate_full337_gate,
    validate_same39_authorization_v2,
    validate_v4a_inputs,
    write_bytes_exclusive,
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
from pams.keypoint_recovery import (
    V4D_PREPROCESSING_REVISION,
    V4D_RECOVERY_MODE,
    KeypointRecoveryError,
    load_keypointrcnn_runtime,
    recover_v4a_locked_video,
)


def _base_receipts(snapshot: Any) -> dict[str, tuple[str, int]]:
    return {
        entry.video_id: (entry.cache_sha256, entry.bytes)
        for entry in snapshot.entries
    }


def _copied_candidate_row(
    *,
    record: Any,
    source_row: Mapping[str, Any],
    output_path: Path,
    expected_cache_sha256: str,
    expected_cache_bytes: int,
) -> dict[str, Any]:
    source = pose_cache_path(output_path.parent, record.video_id)
    require(source == output_path, "candidate output path mismatch")
    sequence, metadata = load_pose_cache(
        output_path,
        expected_video_sha256=record.video_sha256,
        expected_pose_fingerprint=V4D_POSE_FINGERPRINT,
        expected_annotation_sha256=record.annotation_sha256,
        expected_clip_start_frame=record.clip_start_frame,
        expected_clip_end_frame=record.clip_end_frame,
        validate_clip_provenance=True,
    )
    require(sha256_file(output_path) == expected_cache_sha256, "copied candidate SHA mismatch")
    require(output_path.stat().st_size == expected_cache_bytes, "copied candidate size mismatch")
    return {
        "video_id": record.video_id,
        "video_path": source_row.get("video_path"),
        "cache_path": str(output_path.resolve()),
        "cache_sha256": expected_cache_sha256,
        "cache_bytes": expected_cache_bytes,
        "cache_origin": "bytewise_reused_frozen_same39",
        "video_sha256": record.video_sha256,
        "pose_fingerprint": V4D_POSE_FINGERPRINT,
        "source_frames": source_row.get("source_frames"),
        "source_valid_frames": source_row.get("source_valid_frames"),
        "selected_source_frames": source_row.get("selected_source_frames"),
        "cached_frames": sequence.num_frames,
        "cached_valid_frames": int(np.count_nonzero(sequence.valid_mask)),
        "fps": sequence.fps,
        "pose_model": metadata.pose_model,
        "annotation_sha256": metadata.annotation_sha256,
        "clip_start_frame": metadata.clip_start_frame,
        "clip_end_frame": metadata.clip_end_frame,
        "expected_clip_frames": metadata.expected_clip_frames,
        "decoded_clip_frames": metadata.decoded_clip_frames,
        "padded_tail_frames": metadata.padded_tail_frames,
        "incomplete_clip_policy": metadata.incomplete_clip_policy,
        "skipped": False,
        "base_v4a_cache_sha256": source_row.get("base_v4a_cache_sha256"),
        "recovery_audit": source_row.get("recovery_audit"),
    }


def run_full337(
    *,
    source_revision: str,
    container_image_id: str,
    expected_gate_sha256: str,
    expected_extraction_authorization_sha256: str,
    expected_same39_failure_receipt_sha256: str,
    gate_path: Path,
    config_path: Path,
    train_input_path: Path,
    train_commitment_path: Path,
    video_root: Path,
    model_asset_path: Path,
    v4a_cache_dir: Path,
    v4a_ledger_path: Path,
    v4a_paired_gate_path: Path,
    extraction_authorization_path: Path,
    same39_failure_receipt_path: Path,
    same39_audit_path: Path,
    same39_ledger_path: Path,
    same39_selection_path: Path,
    same39_cache_dir: Path,
    output_cache_dir: Path,
    output_ledger_path: Path,
) -> dict[str, Any]:
    """Run the only extraction authorized by the exact same39 PASS chain."""

    require(
        len(source_revision) == 40
        and all(character in "0123456789abcdef" for character in source_revision),
        "source revision must be lowercase 40-hex",
    )
    require(container_image_id == CONTAINER_IMAGE_ID, "container image ID mismatch")
    require(
        os.environ.get("PAMS_CONTAINER_SOURCE_REVISION") == source_revision,
        "container source-revision environment mismatch",
    )
    require(
        os.environ.get("PAMS_CONTAINER_IMAGE_ID") == container_image_id,
        "container image-ID environment mismatch",
    )
    _, gate_sha256 = validate_full337_gate(
        gate_path,
        expected_sha256=expected_gate_sha256,
    )
    config_file_sha256 = sha256_file(config_path)
    require(config_file_sha256 == V4D_CONFIG_FILE_SHA256, "v4d config SHA mismatch")
    config = load_config(config_path.resolve(strict=True))
    require(config.fingerprint == V4D_CONFIG_FINGERPRINT, "v4d config fingerprint mismatch")
    require(config.pose_fingerprint == V4D_POSE_FINGERPRINT, "v4d pose fingerprint mismatch")
    require(
        config.pose.preprocessing_revision == V4D_PREPROCESSING_REVISION,
        "full337 config is not v4d",
    )
    settings = config.pose.keypoint_recovery
    require(settings is not None, "v4d keypoint-recovery settings are missing")
    require(sha256_file(model_asset_path) == MODEL_ASSET_SHA256, "model asset SHA mismatch")

    manifest = load_pose_input_manifest(train_input_path, validate_exact=True)
    commitment = load_pose_input_commitment(train_commitment_path)
    sidecar_sha256 = sha256_file(train_input_path)
    commitment_sha256 = sha256_file(train_commitment_path)
    validate_pose_input_binding(manifest, commitment, sidecar_sha256=sidecar_sha256)
    require(
        manifest.split == "train" and len(manifest.records) == 337,
        "full337 runner accepts only train337",
    )
    require(sidecar_sha256 == TRAIN337_SIDECAR_SHA256, "train337 sidecar SHA mismatch")
    require(commitment_sha256 == TRAIN337_COMMITMENT_SHA256, "train337 commitment SHA mismatch")
    identity_sha256 = pose_input_identity_sha256(manifest.records)
    require(identity_sha256 == TRAIN337_IDENTITY_SHA256, "train337 identity mismatch")

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
    same39_ids = {record.video_id for record in same39.records}
    require(len(same39_ids) == 39, "same39 authorization identity count mismatch")
    base_receipts = _base_receipts(v4a.snapshot)
    same39_receipts = _base_receipts(same39.snapshot)

    output_root = output_cache_dir.resolve(strict=True)
    base_root = v4a_cache_dir.resolve(strict=True)
    pilot_root = same39_cache_dir.resolve(strict=True)
    root = video_root.resolve(strict=True)
    summaries_by_id: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for record in same39.records:
        try:
            source_path = pose_cache_path(pilot_root, record.video_id).resolve(strict=True)
            expected_sha, expected_bytes = same39_receipts[record.video_id]
            payload = stable_file_bytes(source_path)
            require(hashlib.sha256(payload).hexdigest() == expected_sha, "same39 bytes changed")
            require(len(payload) == expected_bytes, "same39 byte count changed")
            target = pose_cache_path(output_root, record.video_id)
            write_bytes_exclusive(target, payload)
            summaries_by_id[record.video_id] = _copied_candidate_row(
                record=record,
                source_row=same39.rows_by_id[record.video_id],
                output_path=target,
                expected_cache_sha256=expected_sha,
                expected_cache_bytes=expected_bytes,
            )
        except (OSError, TypeError, ValueError, Full337ContractError) as exc:
            failures.append(
                {
                    "video_id_sha256": hashlib.sha256(record.video_id.encode("utf-8")).hexdigest(),
                    "phase": "same39-bytewise-reuse",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    require(not failures, "same39 bytewise reuse failed closed")

    runtime = load_keypointrcnn_runtime(model_asset_path, settings=settings)
    remaining = tuple(record for record in manifest.records if record.video_id not in same39_ids)
    require(len(remaining) == 298, "full337 remainder must contain exactly 298 records")
    for offset, record in enumerate(remaining, start=1):
        video_path = (root / record.video_path).resolve(strict=True)
        try:
            video_path.relative_to(root)
            require(sha256_file(video_path) == record.video_sha256, "source video SHA mismatch")
            base_path = pose_cache_path(base_root, record.video_id).resolve(strict=True)
            base_payload = stable_file_bytes(base_path)
            base_sha, base_bytes = base_receipts[record.video_id]
            require(hashlib.sha256(base_payload).hexdigest() == base_sha, "v4a bytes changed")
            require(len(base_payload) == base_bytes, "v4a byte count changed")
            base_sequence, base_metadata = load_pose_cache(
                base_path,
                expected_video_sha256=record.video_sha256,
                expected_pose_fingerprint=V4A_POSE_FINGERPRINT,
                expected_annotation_sha256=record.annotation_sha256,
                expected_clip_start_frame=record.clip_start_frame,
                expected_clip_end_frame=record.clip_end_frame,
                validate_clip_provenance=True,
            )
            base_row = v4a.rows_by_id[record.video_id]
            base_audit = base_row.get("recovery_audit")
            require(isinstance(base_audit, Mapping), "v4a recovery audit missing")
            decoded_frames = int(base_metadata.decoded_clip_frames or 0)
            final_sequence, recovery_audit = recover_v4a_locked_video(
                video_path,
                video_id=record.video_id,
                clip_start_frame=int(record.clip_start_frame),
                clip_end_frame=int(record.clip_end_frame),
                base_sequence=base_sequence,
                base_decoded_frames=decoded_frames,
                base_recovery_audit=base_audit,
                expected_video_sha256=record.video_sha256,
                runtime=runtime,
                settings=settings,
            )
            target = pose_cache_path(output_root, record.video_id)
            metadata = write_pose_cache(
                target,
                final_sequence,
                video_sha256=record.video_sha256,
                pose_fingerprint=V4D_POSE_FINGERPRINT,
                pose_model=settings.model_id,
                annotation_sha256=record.annotation_sha256,
                clip_start_frame=record.clip_start_frame,
                clip_end_frame=record.clip_end_frame,
                decoded_clip_frames=decoded_frames,
                expected_clip_frames=int(record.clip_end_frame) - int(record.clip_start_frame),
                padded_tail_frames=base_metadata.padded_tail_frames,
                incomplete_clip_policy=base_metadata.incomplete_clip_policy,
            )
            cache_sha = sha256_file(target)
            summaries_by_id[record.video_id] = {
                "video_id": record.video_id,
                "video_path": str(video_path),
                "cache_path": str(target.resolve()),
                "cache_sha256": cache_sha,
                "cache_bytes": target.stat().st_size,
                "cache_origin": "fresh_v4d_full337_remainder",
                "video_sha256": record.video_sha256,
                "pose_fingerprint": V4D_POSE_FINGERPRINT,
                "source_frames": base_row.get("source_frames"),
                "source_valid_frames": base_row.get("source_valid_frames"),
                "selected_source_frames": base_row.get("selected_source_frames"),
                "cached_frames": final_sequence.num_frames,
                "cached_valid_frames": int(np.count_nonzero(final_sequence.valid_mask)),
                "fps": final_sequence.fps,
                "pose_model": settings.model_id,
                "annotation_sha256": metadata.annotation_sha256,
                "clip_start_frame": metadata.clip_start_frame,
                "clip_end_frame": metadata.clip_end_frame,
                "expected_clip_frames": metadata.expected_clip_frames,
                "decoded_clip_frames": metadata.decoded_clip_frames,
                "padded_tail_frames": metadata.padded_tail_frames,
                "incomplete_clip_policy": metadata.incomplete_clip_policy,
                "skipped": False,
                "base_v4a_cache_sha256": base_sha,
                "recovery_audit": recovery_audit,
            }
            print(
                json.dumps(
                    {
                        "completed_remainder": offset,
                        "remaining_total": 298,
                        "full337_completed": len(summaries_by_id),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except (OSError, TypeError, ValueError, KeypointRecoveryError, Full337ContractError) as exc:
            failures.append(
                {
                    "video_id_sha256": hashlib.sha256(record.video_id.encode("utf-8")).hexdigest(),
                    "phase": "fresh-v4d-extraction",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            print(
                json.dumps(
                    {"failed_remainder": offset, "error_type": type(exc).__name__},
                    sort_keys=True,
                ),
                flush=True,
            )

    ordered_summaries = [
        summaries_by_id[record.video_id]
        for record in manifest.records
        if record.video_id in summaries_by_id
    ]
    cache_snapshot = None
    if len(ordered_summaries) == 337 and not failures:
        _, snapshot = load_pose_cache_set(
            manifest.records,
            cache_dir=output_root,
            pose_fingerprint=V4D_POSE_FINGERPRINT,
            materialize_sequences=False,
        )
        cache_snapshot = snapshot.to_dict()
    ledger = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4d_train337_full_ledger",
        "source_revision": source_revision,
        "container_image_id": container_image_id,
        "gate_sha256": gate_sha256,
        "config_file_sha256": config_file_sha256,
        "config_fingerprint": config.fingerprint,
        "pose_fingerprint": config.pose_fingerprint,
        "model_asset_sha256": MODEL_ASSET_SHA256,
        "input_kind": "label_free_train337_sidecar",
        "protocol": manifest.protocol,
        "split": "train",
        "sidecar_sha256": sidecar_sha256,
        "commitment_file_sha256": commitment_sha256,
        "identity_sha256": identity_sha256,
        "v4a": {
            "ledger_sha256": V4A_LEDGER_SHA256,
            "paired_gate_sha256": V4A_PAIRED_GATE_SHA256,
            "pose_fingerprint": V4A_POSE_FINGERPRINT,
            "cache_set_sha256": v4a.snapshot.fingerprint,
            "cache_entry_count": len(v4a.snapshot.entries),
        },
        "same39_authorization": {
            "authorization_protocol_version": 2,
            "authorization_scope": "full337_pose_extraction_only",
            "source_revision": same39.authorization_receipt.get("source_revision"),
            "authorization_receipt_sha256": same39.authorization_receipt_sha256,
            "failure_receipt_sha256": same39.evidence.failure_receipt_sha256,
            "audit_sha256": same39.evidence.audit_sha256,
            "ledger_sha256": same39.evidence.ledger_sha256,
            "selection_sha256": same39.evidence.selection_sha256,
            "pose_fingerprint": V4D_POSE_FINGERPRINT,
            "cache_set_sha256": same39.snapshot.fingerprint,
            "cache_entry_count": len(same39.snapshot.entries),
            "full337_pose_extraction_authorized": True,
            "baseline_training_authorized": False,
        },
        "pose_recovery": {
            "schema_version": 1,
            "preprocessing_revision": config.pose.preprocessing_revision,
            "recovery_mode": V4D_RECOVERY_MODE,
            "keypointrcnn_model_id": settings.model_id,
            "keypointrcnn_model_asset_sha256": settings.model_asset_sha256,
            "keypointrcnn_runtime_receipt": dict(runtime.runtime_receipt),
            "detector_observes_all_decoded_frames": settings.detector_observes_all_decoded_frames,
            "fill_missing_only": settings.fill_missing_only,
            "temporal_resampling": settings.temporal_resampling,
            "pose_coordinate_interpolation": settings.pose_coordinate_interpolation,
        },
        "selected": 337,
        "completed": len(ordered_summaries),
        "extracted_fresh": sum(
            row["cache_origin"] == "fresh_v4d_full337_remainder" for row in ordered_summaries
        ),
        "reused_same39": sum(
            row["cache_origin"] == "bytewise_reused_frozen_same39" for row in ordered_summaries
        ),
        "failed": len(failures),
        "successful_cache_snapshot": cache_snapshot,
        "caches": ordered_summaries,
        "failures": failures,
        "baseline_training_authorized": False,
        "training_authorization_requires": "full337-label-free-track-quality-periodicity-gate-pass",
    }
    write_json_exclusive(output_ledger_path, ledger)
    return {
        "selected": 337,
        "completed": len(ordered_summaries),
        "reused_same39": ledger["reused_same39"],
        "extracted_fresh": ledger["extracted_fresh"],
        "failed": len(failures),
        "cache_set_sha256": None if cache_snapshot is None else cache_snapshot["fingerprint"],
        "baseline_training_authorized": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--extraction-authorization-sha256", required=True)
    parser.add_argument("--same39-failure-receipt-sha256", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-commitment", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model-asset", type=Path, required=True)
    parser.add_argument("--v4a-cache-dir", type=Path, required=True)
    parser.add_argument("--v4a-ledger", type=Path, required=True)
    parser.add_argument("--v4a-paired-gate", type=Path, required=True)
    parser.add_argument("--extraction-authorization", type=Path, required=True)
    parser.add_argument("--same39-failure-receipt", type=Path, required=True)
    parser.add_argument("--same39-audit", type=Path, required=True)
    parser.add_argument("--same39-ledger", type=Path, required=True)
    parser.add_argument("--same39-selection", type=Path, required=True)
    parser.add_argument("--same39-cache-dir", type=Path, required=True)
    parser.add_argument("--output-cache-dir", type=Path, required=True)
    parser.add_argument("--output-ledger", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_full337(
            source_revision=args.source_revision,
            container_image_id=args.container_image_id,
            expected_gate_sha256=args.gate_sha256,
            expected_extraction_authorization_sha256=args.extraction_authorization_sha256,
            expected_same39_failure_receipt_sha256=args.same39_failure_receipt_sha256,
            gate_path=args.gate,
            config_path=args.config,
            train_input_path=args.train_input,
            train_commitment_path=args.train_commitment,
            video_root=args.video_root,
            model_asset_path=args.model_asset,
            v4a_cache_dir=args.v4a_cache_dir,
            v4a_ledger_path=args.v4a_ledger,
            v4a_paired_gate_path=args.v4a_paired_gate,
            extraction_authorization_path=args.extraction_authorization,
            same39_failure_receipt_path=args.same39_failure_receipt,
            same39_audit_path=args.same39_audit,
            same39_ledger_path=args.same39_ledger,
            same39_selection_path=args.same39_selection,
            same39_cache_dir=args.same39_cache_dir,
            output_cache_dir=args.output_cache_dir,
            output_ledger_path=args.output_ledger,
        )
    except (OSError, TypeError, ValueError, KeypointRecoveryError, Full337ContractError) as exc:
        print(f"v4d full337 extraction failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["failed"] == 0 and result["completed"] == 337 else 1


if __name__ == "__main__":
    raise SystemExit(main())
