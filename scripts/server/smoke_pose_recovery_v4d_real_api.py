#!/usr/bin/env python3
"""Run one label-free, batch-1 v4d real-API determinism smoke."""

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

from pams.config import load_config
from pams.data import (
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
    validate_pose_input_binding,
    write_pose_cache,
)
from pams.keypoint_recovery import (
    KeypointRecoveryError,
    _fill_xyz,
    _infer_batch,
    load_keypointrcnn_runtime,
)
from pams.pose import select_global_dominant_track, sha256_file
from pams.types import PoseSequence


class SmokeError(RuntimeError):
    pass


_FROZEN_VIDEO_ID_SHA256 = (
    "0c13cd62647c4e658294ea962e3008fc55eb3f7c59de26e99c721130556377f7"
)
_FROZEN_FRAME_OFFSET = 0


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_smoke(
    *,
    source_revision: str,
    container_image_id: str,
    config_path: Path,
    expected_config_file_sha256: str,
    expected_config_fingerprint: str,
    train_input_path: Path,
    train_commitment_path: Path,
    video_root: Path,
    video_id_sha256: str,
    frame_offset: int,
    model_asset_path: Path,
    cache_output_path: Path,
) -> dict[str, Any]:
    _require(len(source_revision) == 40, "source revision must be 40-hex")
    _require(os.environ.get("PAMS_CONTAINER_SOURCE_REVISION") == source_revision, "source mismatch")
    _require(os.environ.get("PAMS_CONTAINER_IMAGE_ID") == container_image_id, "image mismatch")
    _require(video_id_sha256 == _FROZEN_VIDEO_ID_SHA256, "smoke video identity is not frozen")
    _require(frame_offset == _FROZEN_FRAME_OFFSET, "smoke frame offset is not frozen")
    config_file_sha256 = sha256_file(config_path.resolve(strict=True))
    _require(config_file_sha256 == expected_config_file_sha256, "config SHA mismatch")
    config = load_config(config_path.resolve(strict=True))
    _require(config.fingerprint == expected_config_fingerprint, "config fingerprint mismatch")
    settings = config.pose.keypoint_recovery
    _require(settings is not None, "smoke requires v4d config")

    manifest = load_pose_input_manifest(train_input_path, validate_exact=True)
    commitment = load_pose_input_commitment(train_commitment_path)
    sidecar_sha256 = sha256_file(train_input_path.resolve(strict=True))
    commitment_sha256 = sha256_file(train_commitment_path.resolve(strict=True))
    validate_pose_input_binding(manifest, commitment, sidecar_sha256=sidecar_sha256)
    _require(manifest.split == "train" and len(manifest.records) == 337, "input is not train337")
    matches = tuple(
        record
        for record in manifest.records
        if hashlib.sha256(record.video_id.encode("utf-8")).hexdigest() == video_id_sha256
    )
    _require(len(matches) == 1, "frozen video hash does not select exactly one train record")
    record = matches[0]
    clip_frames = int(record.clip_end_frame) - int(record.clip_start_frame)
    _require(0 <= frame_offset < clip_frames, "frame offset is outside the official clip")
    video_path = (video_root.resolve(strict=True) / record.video_path).resolve(strict=True)
    video_path.relative_to(video_root.resolve(strict=True))
    video_stat_before = video_path.stat()
    video_sha256_before = sha256_file(video_path)
    _require(video_sha256_before == record.video_sha256, "video SHA mismatch")

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    _require(capture.isOpened(), "OpenCV could not open smoke video")
    absolute_frame = int(record.clip_start_frame) + frame_offset
    try:
        _require(capture.set(cv2.CAP_PROP_POS_FRAMES, absolute_frame), "frame seek failed")
        positioned = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
        _require(math.isfinite(positioned) and abs(positioned - absolute_frame) <= 0.5, "seek drift")
        ok, frame = capture.read()
        _require(ok and frame is not None, "smoke frame decode failed")
        positioned_after = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
        _require(
            math.isfinite(positioned_after) and abs(positioned_after - absolute_frame - 1) <= 0.5,
            "post-read frame drift",
        )
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        _require(math.isfinite(fps) and fps > 0.0, "invalid smoke FPS")
    finally:
        capture.release()
    video_stat_after = video_path.stat()
    video_sha256_after = sha256_file(video_path)
    _require(video_sha256_after == video_sha256_before, "video changed during smoke")
    _require(
        tuple(
            getattr(video_stat_before, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        )
        == tuple(
            getattr(video_stat_after, field)
            for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        ),
        "video stat changed during smoke",
    )

    runtime = load_keypointrcnn_runtime(model_asset_path, settings=settings)
    (candidates,) = _infer_batch(runtime, [frame], settings=settings)
    candidate_array = (
        np.empty((0, 33, 4), dtype=np.float32)
        if candidates is None
        else np.asarray(candidates, dtype=np.float32)
    )
    candidate_bytes = np.ascontiguousarray(candidate_array).tobytes(order="C")
    (selected,) = select_global_dominant_track(
        [candidates],
        center_weight=settings.association_center_weight,
        log_scale_weight=settings.association_log_scale_weight,
    )
    fill_xyz = (
        np.zeros((33, 3), dtype=np.float32)
        if selected is None
        else _fill_xyz(selected)
    )
    valid = selected is not None
    sequence = PoseSequence(
        video_id=video_id_sha256,
        fps=fps,
        xyz=fill_xyz[None, ...],
        valid_mask=np.asarray([valid], dtype=np.bool_),
    )
    write_pose_cache(
        cache_output_path,
        sequence,
        video_sha256=record.video_sha256,
        pose_fingerprint=config.pose_fingerprint,
        pose_model=settings.model_id,
    )
    runtime_receipt = dict(runtime.runtime_receipt)
    return {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4d_real_api_determinism_smoke",
        "scope": "label-free-train337-single-frame-batch1",
        "source_revision": source_revision,
        "container_image_id": container_image_id,
        "config_file_sha256": config_file_sha256,
        "config_fingerprint": config.fingerprint,
        "pose_fingerprint": config.pose_fingerprint,
        "train337_sidecar_sha256": sidecar_sha256,
        "train337_commitment_sha256": commitment_sha256,
        "train337_identity_sha256": pose_input_identity_sha256(manifest.records),
        "video_id_sha256": video_id_sha256,
        "video_sha256": video_sha256_before,
        "absolute_frame_index": absolute_frame,
        "frame_offset": frame_offset,
        "decoded_frame_sha256": hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest(),
        "inference_batch_size": 1,
        "candidate_count": int(candidate_array.shape[0]),
        "selected_candidate_present": valid,
        "candidate_bytes_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "selected_fill_bytes_sha256": hashlib.sha256(
            np.ascontiguousarray(fill_xyz).tobytes(order="C")
        ).hexdigest(),
        "final_cache_sha256": sha256_file(cache_output_path.resolve(strict=True)),
        "model_asset_sha256": runtime.asset_sha256,
        "runtime_receipt": runtime_receipt,
        "runtime_receipt_sha256": _canonical_sha256(runtime_receipt),
        "video_source_stat_stable": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-file-sha256", required=True)
    parser.add_argument("--config-fingerprint", required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-commitment", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--video-id-sha256", required=True)
    parser.add_argument("--frame-offset", type=int, required=True)
    parser.add_argument("--model-asset", type=Path, required=True)
    parser.add_argument("--cache-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_smoke(
            source_revision=args.source_revision,
            container_image_id=args.container_image_id,
            config_path=args.config,
            expected_config_file_sha256=args.config_file_sha256,
            expected_config_fingerprint=args.config_fingerprint,
            train_input_path=args.train_input,
            train_commitment_path=args.train_commitment,
            video_root=args.video_root,
            video_id_sha256=args.video_id_sha256,
            frame_offset=args.frame_offset,
            model_asset_path=args.model_asset,
            cache_output_path=args.cache_output,
        )
        target = args.receipt_output.resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except (OSError, TypeError, ValueError, KeypointRecoveryError, SmokeError) as exc:
        print(f"v4d real-API smoke failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
