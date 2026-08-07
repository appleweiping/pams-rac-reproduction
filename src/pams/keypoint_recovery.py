"""Torchvision Keypoint R-CNN recovery over immutable v4a pose caches."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pams.config import KeypointRCNNRecoveryConfig
from pams.data import per_frame_minmax
from pams.pose import select_global_dominant_track, sha256_file
from pams.types import PoseSequence

V4D_PREPROCESSING_REVISION = (
    "official-segment-v4a-locked-keypointrcnn-fill-missing-full-timeline-v4d"
)
V4D_RECOVERY_MODE = "v4a-locked-keypointrcnn-missing-fill-v4d"

# MediaPipe index -> COCO index. COCO has no mouth, finger, heel, or toe
# landmarks, so those locations use the explicitly frozen duplicate policy.
MEDIAPIPE33_TO_COCO17 = np.asarray(
    [
        0,
        1,
        1,
        1,
        2,
        2,
        2,
        3,
        4,
        0,
        0,
        5,
        6,
        7,
        8,
        9,
        10,
        9,
        10,
        9,
        10,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        15,
        16,
        15,
        16,
    ],
    dtype=np.int64,
)

# One non-duplicated MediaPipe landmark for each COCO keypoint, in COCO order.
COCO17_TO_MEDIAPIPE33 = np.asarray(
    [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28],
    dtype=np.int64,
)


class KeypointRecoveryError(RuntimeError):
    """Raised when v4d provenance or recovery invariants fail."""


@dataclass(frozen=True, slots=True)
class KeypointRCNNRuntime:
    """Loaded model plus the exact immutable asset identity."""

    model: Any
    torch: Any
    device: Any
    asset_path: Path
    asset_sha256: str
    state_dict_keys: int
    parameter_count: int
    runtime_receipt: Mapping[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KeypointRecoveryError(message)


def _as_numpy(value: Any, *, dtype: Any) -> NDArray[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def _canonical_candidate_order(
    candidates: Sequence[NDArray[np.float32]],
) -> NDArray[np.float32] | None:
    if not candidates:
        return None
    ordered = sorted(
        (np.asarray(candidate, dtype=np.float32) for candidate in candidates),
        key=lambda candidate: hashlib.sha256(candidate.tobytes(order="C")).digest(),
    )
    return np.stack(ordered, axis=0)


def keypointrcnn_candidates_from_output(
    output: Mapping[str, Any],
    *,
    frame_width: int,
    frame_height: int,
    settings: KeypointRCNNRecoveryConfig,
) -> NDArray[np.float32] | None:
    """Apply the preregistered COCO-person filter and 17-to-33 mapping."""

    if frame_width < 1 or frame_height < 1:
        raise ValueError("frame dimensions must be positive")
    required = {"labels", "scores", "keypoints", "keypoints_scores"}
    if not required.issubset(output):
        raise KeypointRecoveryError("Keypoint R-CNN output is missing required fields")
    labels = _as_numpy(output["labels"], dtype=np.int64)
    scores = _as_numpy(output["scores"], dtype=np.float32)
    keypoints = _as_numpy(output["keypoints"], dtype=np.float32)
    logits = _as_numpy(output["keypoints_scores"], dtype=np.float32)
    count = len(scores)
    _require(labels.shape == (count,), "detector labels shape mismatch")
    _require(keypoints.shape == (count, 17, 3), "detector keypoints shape mismatch")
    _require(logits.shape == (count, 17), "detector keypoint logits shape mismatch")
    _require(
        np.isfinite(scores).all() and np.isfinite(keypoints).all() and np.isfinite(logits).all(),
        "detector output contains non-finite values",
    )

    eligible: list[tuple[float, bytes, NDArray[np.float32]]] = []
    for index in range(count):
        box_score = float(scores[index])
        confident = int(np.count_nonzero(logits[index] > settings.keypoint_logit_threshold))
        if (
            int(labels[index]) != settings.person_label
            or box_score < settings.box_score_threshold
            or confident < settings.minimum_confident_keypoints
        ):
            continue
        xy = np.array(keypoints[index, :, :2], dtype=np.float32, copy=True)
        xy[:, 0] = np.clip(xy[:, 0] / float(frame_width), 0.0, 1.0)
        xy[:, 1] = np.clip(xy[:, 1] / float(frame_height), 0.0, 1.0)
        mapped_xy = xy[MEDIAPIPE33_TO_COCO17]
        xyz = np.zeros((33, 3), dtype=np.float32)
        xyz[:, :2] = mapped_xy
        mapped_logits = logits[index, MEDIAPIPE33_TO_COCO17]
        visibility = (1.0 / (1.0 + np.exp(-mapped_logits.astype(np.float64)))).astype(np.float32)
        visibility *= np.float32(box_score)
        candidate = np.concatenate((xyz, visibility[:, None]), axis=1)
        digest = hashlib.sha256(candidate.tobytes(order="C")).digest()
        eligible.append((box_score, digest, candidate))
    eligible.sort(key=lambda item: (-item[0], item[1]))
    retained = [item[2] for item in eligible[: settings.maximum_candidates_per_frame]]
    return _canonical_candidate_order(retained)


def load_keypointrcnn_runtime(
    asset_path: str | Path,
    *,
    settings: KeypointRCNNRecoveryConfig,
) -> KeypointRCNNRuntime:
    """Strictly load the frozen COCO_V1 state dict without network access."""

    source = Path(asset_path).resolve(strict=True)
    _require(source.name == settings.model_asset_filename, "model asset filename mismatch")
    source_stat_before = source.stat()
    _require(not os.access(source, os.W_OK), "model asset must be mounted read-only")
    received = sha256_file(source)
    _require(received == settings.model_asset_sha256, "Keypoint R-CNN asset SHA mismatch")
    import torch
    import torchvision
    from torch import nn
    from torchvision.models.detection import (
        KeypointRCNN_ResNet50_FPN_Weights,
        keypointrcnn_resnet50_fpn,
    )
    from torchvision.ops.misc import FrozenBatchNorm2d

    _require(torch.__version__ == settings.expected_torch_version, "torch version mismatch")
    _require(
        torchvision.__version__ == settings.expected_torchvision_version,
        "torchvision version mismatch",
    )
    _require(torch.version.cuda == settings.expected_cuda_version, "CUDA build mismatch")
    _require(
        torch.backends.cudnn.version() == settings.expected_cudnn_version,
        "cuDNN version mismatch",
    )
    _require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == settings.cublas_workspace_config,
        "CUBLAS_WORKSPACE_CONFIG mismatch",
    )
    torch_home = Path(os.environ.get("TORCH_HOME", "")).resolve()
    expected_cache_asset = (
        torch_home / "hub" / "checkpoints" / settings.model_asset_filename
    ).resolve(strict=True)
    _require(
        source == expected_cache_asset,
        "model asset must be the exact offline torchvision cache entry",
    )
    for variable in (
        "XDG_CACHE_HOME",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
    ):
        cache_path = Path(os.environ.get(variable, "")).resolve(strict=True)
        _require(
            cache_path == Path("/pams/cache").resolve()
            or Path("/pams/cache").resolve() in cache_path.parents,
            f"{variable} must use the ephemeral /pams/cache tmpfs",
        )
        _require(os.access(cache_path, os.W_OK), f"{variable} is not writable")

    _require(torch.cuda.is_available(), "v4d requires the preregistered CUDA device")
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    torch.use_deterministic_algorithms(settings.deterministic_algorithms)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = settings.allow_tf32
    torch.backends.cudnn.allow_tf32 = settings.allow_tf32
    torch.set_float32_matmul_precision("highest")
    properties = torch.cuda.get_device_properties(0)
    compute_capability = f"{properties.major}.{properties.minor}"
    _require(properties.name == settings.expected_gpu_name, "GPU model mismatch")
    _require(
        compute_capability == settings.expected_compute_capability,
        "GPU compute capability mismatch",
    )
    weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
    _require(
        int(weights.meta["num_params"]) == settings.expected_parameter_count,
        "torchvision COCO_V1 metadata parameter count mismatch",
    )
    model = keypointrcnn_resnet50_fpn(
        weights=weights,
        progress=False,
        min_size=settings.input_min_size,
        max_size=settings.input_max_size,
        box_nms_thresh=settings.box_nms_threshold,
        box_detections_per_img=settings.box_detections_per_image,
    )
    state_dict_keys = len(model.state_dict())
    _require(state_dict_keys == settings.expected_state_dict_keys, "state-dict key mismatch")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    _require(
        parameter_count == settings.expected_parameter_count,
        "Keypoint R-CNN parameter count mismatch",
    )
    frozen_batchnorm_count = sum(
        isinstance(module, FrozenBatchNorm2d) for module in model.modules()
    )
    batchnorm_count = sum(
        isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()
    )
    _require(frozen_batchnorm_count > 0, "official FrozenBatchNorm2d topology is absent")
    _require(batchnorm_count == 0, "unexpected trainable BatchNorm topology")
    _require(
        all(
            float(module.eps) == 0.0
            for module in model.modules()
            if isinstance(module, FrozenBatchNorm2d)
        ),
        "COCO_V1 FrozenBatchNorm epsilon was not overwritten to zero",
    )
    device = torch.device(settings.inference_device)
    model.eval().to(device)
    source_stat_after = source.stat()
    _require(sha256_file(source) == received, "Keypoint R-CNN asset changed while loading")
    _require(
        (
            source_stat_before.st_dev,
            source_stat_before.st_ino,
            source_stat_before.st_size,
            source_stat_before.st_mtime_ns,
        )
        == (
            source_stat_after.st_dev,
            source_stat_after.st_ino,
            source_stat_after.st_size,
            source_stat_after.st_mtime_ns,
        ),
        "Keypoint R-CNN asset stat identity changed while loading",
    )
    return KeypointRCNNRuntime(
        model=model,
        torch=torch,
        device=device,
        asset_path=source,
        asset_sha256=received,
        state_dict_keys=state_dict_keys,
        parameter_count=parameter_count,
        runtime_receipt={
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "gpu_name": properties.name,
            "gpu_compute_capability": compute_capability,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "frozen_batchnorm_modules": frozen_batchnorm_count,
            "batchnorm_modules": batchnorm_count,
            "frozen_batchnorm_eps": 0.0,
            "offline_weights_cache_path": str(source),
            "model_asset_read_only": True,
            "model_asset_stat_stable": True,
            "model_asset_bytes": source_stat_after.st_size,
        },
    )


def _infer_batch(
    runtime: KeypointRCNNRuntime,
    frames: Sequence[NDArray[np.uint8]],
    *,
    settings: KeypointRCNNRecoveryConfig,
) -> tuple[NDArray[np.float32] | None, ...]:
    tensors = []
    dimensions: list[tuple[int, int]] = []
    for frame in frames:
        _require(
            frame.ndim == 3 and frame.shape[2] == 3,
            "decoded frame must have BGR shape [H,W,3]",
        )
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        tensor = runtime.torch.from_numpy(rgb).permute(2, 0, 1)
        tensors.append(tensor.to(runtime.device, dtype=runtime.torch.float32).div_(255.0))
        dimensions.append((int(frame.shape[1]), int(frame.shape[0])))
    with runtime.torch.inference_mode():
        outputs = runtime.model(tensors)
    _require(len(outputs) == len(frames), "detector batch output count mismatch")
    return tuple(
        keypointrcnn_candidates_from_output(
            output,
            frame_width=width,
            frame_height=height,
            settings=settings,
        )
        for output, (width, height) in zip(outputs, dimensions, strict=True)
    )


def _mask_sha256(mask: NDArray[np.bool_]) -> str:
    return hashlib.sha256(bytes(int(value) for value in mask)).hexdigest()


def _coordinate_sha256(xyz: NDArray[np.float32], mask: NDArray[np.bool_]) -> str:
    return hashlib.sha256(np.ascontiguousarray(xyz[mask]).tobytes(order="C")).hexdigest()


def _longest_run(mask: NDArray[np.bool_]) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def _fill_xyz(candidate: NDArray[np.float32]) -> NDArray[np.float32]:
    """Convert one image-space association candidate to the frozen fill space."""

    xyz = np.asarray(candidate[:, :3], dtype=np.float32)
    return per_frame_minmax(xyz[None, ...])[0]


def _v4a_shape_anchor(
    candidates: NDArray[np.float32] | None,
    base_xyz: NDArray[np.float32],
    *,
    maximum_distance: float,
) -> tuple[NDArray[np.float32] | None, float | None]:
    """Select one image-space candidate using only normalized xy pose shape."""

    if candidates is None:
        return None, None
    array = np.asarray(candidates, dtype=np.float32)
    _require(array.ndim == 3 and array.shape[1:] == (33, 4), "candidate shape mismatch")

    def normalize_xy_shape(xy: NDArray[np.float32]) -> NDArray[np.float64] | None:
        points = np.asarray(xy[COCO17_TO_MEDIAPIPE33], dtype=np.float64)
        minimum = np.min(points, axis=0)
        span = np.max(points, axis=0) - minimum
        if not np.all(span > 1e-6):
            return None
        return (points - minimum) / span

    base_xy = normalize_xy_shape(np.asarray(base_xyz[:, :2], dtype=np.float32))
    if base_xy is None:
        return None, None
    normalized_candidates = [normalize_xy_shape(candidate[:, :2]) for candidate in array]
    distances = np.asarray(
        [
            (
                np.mean(
                    np.linalg.norm(
                        normalized - base_xy,
                        axis=1,
                    )
                )
                if normalized is not None
                else math.inf
            )
            for normalized in normalized_candidates
        ],
        dtype=np.float64,
    )
    if not np.isfinite(distances).any():
        return None, None
    best = int(np.argmin(distances))
    distance = float(distances[best])
    if distance > maximum_distance:
        return None, distance
    return array[best][None, ...], distance


def recover_v4a_locked_video(
    video_path: str | Path,
    *,
    video_id: str,
    clip_start_frame: int,
    clip_end_frame: int,
    base_sequence: PoseSequence,
    base_decoded_frames: int,
    base_recovery_audit: Mapping[str, Any],
    expected_video_sha256: str,
    runtime: KeypointRCNNRuntime,
    settings: KeypointRCNNRecoveryConfig,
) -> tuple[PoseSequence, dict[str, Any]]:
    """Fill only v4a-invalid decoded frames and retain every v4a value bitwise."""

    source = Path(video_path).resolve(strict=True)
    source_stat_before = source.stat()
    source_sha256_before = sha256_file(source)
    _require(source_sha256_before == expected_video_sha256, "source video SHA mismatch")
    expected_frames = int(clip_end_frame) - int(clip_start_frame)
    _require(expected_frames > 0, "v4d requires a non-empty official segment")
    _require(base_sequence.video_id == video_id, "v4a cache video identity mismatch")
    _require(base_sequence.num_frames == expected_frames, "v4a cache timeline mismatch")
    _require(
        0 <= base_decoded_frames <= expected_frames,
        "v4a decoded frame count is invalid",
    )
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise KeypointRecoveryError(f"OpenCV could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise KeypointRecoveryError("video reports invalid FPS")
    _require(
        math.isclose(fps, float(base_sequence.fps), rel_tol=0.0, abs_tol=1e-6),
        "v4d decoder FPS differs from v4a cache FPS",
    )
    if clip_start_frame:
        seek_ok = bool(capture.set(cv2.CAP_PROP_POS_FRAMES, int(clip_start_frame)))
        positioned = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
        if not seek_ok or (np.isfinite(positioned) and abs(positioned - clip_start_frame) > 0.5):
            capture.release()
            raise KeypointRecoveryError("OpenCV could not seek to official clip start")

    detector_candidates: list[NDArray[np.float32] | None] = [None] * expected_frames
    pending_frames: list[NDArray[np.uint8]] = []
    pending_indices: list[int] = []
    decoded_frame_digest = hashlib.sha256()

    def flush() -> None:
        nonlocal pending_frames, pending_indices
        if not pending_frames:
            return
        outputs = _infer_batch(runtime, pending_frames, settings=settings)
        for index, candidates in zip(pending_indices, outputs, strict=True):
            detector_candidates[index] = candidates
        pending_frames = []
        pending_indices = []

    decoded = 0
    try:
        while decoded < base_decoded_frames:
            positioned_before = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            expected_position_before = int(clip_start_frame) + decoded
            _require(
                np.isfinite(positioned_before)
                and abs(positioned_before - expected_position_before) <= 0.5,
                "decoder frame position drifted before read",
            )
            ok, frame = capture.read()
            if not ok:
                raise KeypointRecoveryError("v4d decode ended before frozen v4a decode")
            _require(frame is not None, "decoder returned an empty frame")
            decoded_frame_digest.update(np.asarray(frame.shape, dtype=np.int64).tobytes())
            decoded_frame_digest.update(np.ascontiguousarray(frame).tobytes(order="C"))
            positioned_after = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            _require(
                np.isfinite(positioned_after)
                and abs(positioned_after - (expected_position_before + 1)) <= 0.5,
                "decoder frame position drifted after read",
            )
            pending_frames.append(frame)
            pending_indices.append(decoded)
            if len(pending_frames) == settings.inference_batch_size:
                flush()
            decoded += 1
        flush()
    finally:
        capture.release()
    _require(decoded == base_decoded_frames, "v4d decoded frame count mismatch")

    source_stat_after = source.stat()
    source_sha256_after = sha256_file(source)
    _require(source_sha256_after == source_sha256_before, "source video changed during decode")
    _require(
        (
            source_stat_before.st_dev,
            source_stat_before.st_ino,
            source_stat_before.st_size,
            source_stat_before.st_mtime_ns,
        )
        == (
            source_stat_after.st_dev,
            source_stat_after.st_ino,
            source_stat_after.st_size,
            source_stat_after.st_mtime_ns,
        ),
        "source video stat identity changed during decode",
    )

    anchored: list[NDArray[np.float32] | None] = []
    anchor_distances: list[float] = []
    rejected_anchor_distances: list[float] = []
    for index, candidates in enumerate(detector_candidates):
        if bool(base_sequence.valid_mask[index]):
            anchor, distance = _v4a_shape_anchor(
                candidates,
                base_sequence.xyz[index],
                maximum_distance=settings.v4a_anchor_distance_maximum,
            )
            anchored.append(anchor)
            if distance is not None:
                (anchor_distances if anchor is not None else rejected_anchor_distances).append(
                    distance
                )
        else:
            anchored.append(candidates)
    selected = select_global_dominant_track(
        anchored,
        center_weight=settings.association_center_weight,
        log_scale_weight=settings.association_log_scale_weight,
    )
    final_xyz = np.array(base_sequence.xyz, dtype=np.float32, copy=True)
    final_mask = np.array(base_sequence.valid_mask, dtype=np.bool_, copy=True)
    for index, candidate in enumerate(selected):
        if not final_mask[index] and candidate is not None:
            final_xyz[index] = _fill_xyz(candidate)
            final_mask[index] = True
    base_mask = np.asarray(base_sequence.valid_mask, dtype=np.bool_)
    _require(
        np.array_equal(final_xyz[base_mask], base_sequence.xyz[base_mask]),
        "v4d changed a locked v4a coordinate",
    )
    _require(np.all(final_mask[base_mask]), "v4d removed a locked v4a observation")
    final_sequence = PoseSequence(
        video_id=video_id,
        fps=base_sequence.fps,
        xyz=final_xyz,
        valid_mask=final_mask,
    )
    candidate_counts = [
        0 if candidate is None else int(candidate.shape[0])
        for candidate in detector_candidates[:base_decoded_frames]
    ]
    detector_valid = sum(count > 0 for count in candidate_counts)
    base_valid = int(np.count_nonzero(base_mask))
    base_valid_decoded = int(np.count_nonzero(base_mask[:base_decoded_frames]))
    missing_candidate_frames = sum(
        count > 0 and not bool(base_mask[index]) for index, count in enumerate(candidate_counts)
    )
    anchor_frames = len(anchor_distances)
    base_frames_with_detector_candidates = sum(
        candidate is not None and bool(base_mask[index])
        for index, candidate in enumerate(detector_candidates[:base_decoded_frames])
    )
    anchor_unusable_frames = (
        base_frames_with_detector_candidates - anchor_frames - len(rejected_anchor_distances)
    )
    final_valid = int(np.count_nonzero(final_mask))
    pass0_valid = int(base_recovery_audit["pass0_valid_frames"])
    _require(final_valid >= base_valid >= pass0_valid, "v4d valid-count monotonicity failed")
    valid_indices = np.flatnonzero(final_mask)
    observed_span = int(valid_indices[-1] - valid_indices[0] + 1) if valid_indices.size else 0
    base_coordinate_sha = _coordinate_sha256(base_sequence.xyz, base_mask)
    final_base_coordinate_sha = _coordinate_sha256(final_xyz, base_mask)
    _require(base_coordinate_sha == final_base_coordinate_sha, "v4a coordinate digest changed")
    audit = dict(base_recovery_audit)
    audit.update(
        {
            "recovery_mode": V4D_RECOVERY_MODE,
            "recovered_valid_frames": final_valid - pass0_valid,
            "final_valid_frames": final_valid,
            "observed_span_frames": observed_span,
            "final_longest_valid_run": _longest_run(final_mask),
            "final_valid_mask_sha256": _mask_sha256(final_mask),
            "base_v4a_valid_frames": base_valid,
            "base_v4a_valid_mask_sha256": _mask_sha256(base_mask),
            "base_v4a_observations_preserved": True,
            "base_v4a_shared_coordinate_max_abs_error": 0.0,
            "base_v4a_coordinate_sha256": base_coordinate_sha,
            "final_base_v4a_coordinate_sha256": final_base_coordinate_sha,
            "keypointrcnn_frames_observed": decoded,
            "keypointrcnn_frames_attempted": decoded,
            "keypointrcnn_frames_with_candidates": detector_valid,
            "keypointrcnn_missing_frames_eligible": decoded - base_valid_decoded,
            "keypointrcnn_missing_frames_with_candidates": missing_candidate_frames,
            "keypointrcnn_v4a_anchor_frames": anchor_frames,
            "keypointrcnn_v4a_anchor_rejected_frames": len(rejected_anchor_distances),
            "keypointrcnn_v4a_anchor_unusable_shape_frames": anchor_unusable_frames,
            "keypointrcnn_v4a_anchor_distance_maximum": (settings.v4a_anchor_distance_maximum),
            "keypointrcnn_v4a_anchor_distance_mean": (
                float(np.mean(anchor_distances)) if anchor_distances else None
            ),
            "keypointrcnn_v4a_anchor_distance_observed_maximum": (
                max(anchor_distances) if anchor_distances else None
            ),
            "keypointrcnn_v4a_anchor_distance_sha256": hashlib.sha256(
                np.asarray(anchor_distances, dtype=np.float64).tobytes(order="C")
            ).hexdigest(),
            "keypointrcnn_candidate_total": sum(candidate_counts),
            "keypointrcnn_max_candidates_per_frame": max(candidate_counts, default=0),
            "keypointrcnn_fill_candidates": final_valid - base_valid,
            "keypointrcnn_model_id": settings.model_id,
            "keypointrcnn_model_asset_sha256": settings.model_asset_sha256,
            "keypointrcnn_box_score_threshold": settings.box_score_threshold,
            "keypointrcnn_keypoint_logit_threshold": settings.keypoint_logit_threshold,
            "keypointrcnn_minimum_confident_keypoints": settings.minimum_confident_keypoints,
            "keypointrcnn_maximum_candidates_per_frame": settings.maximum_candidates_per_frame,
            "keypointrcnn_coco_to_mediapipe_mapping": settings.coco_to_mediapipe_mapping,
            "keypointrcnn_z_coordinate_policy": settings.z_coordinate_policy,
            "keypointrcnn_input_scale_policy": settings.input_scale_policy,
            "keypointrcnn_association_coordinate_space": (settings.association_coordinate_space),
            "keypointrcnn_v4a_anchor_policy": settings.v4a_anchor_policy,
            "keypointrcnn_fill_coordinate_space": settings.fill_coordinate_space,
            "keypointrcnn_state_dict_keys": runtime.state_dict_keys,
            "keypointrcnn_parameter_count": runtime.parameter_count,
            "keypointrcnn_runtime_receipt": dict(runtime.runtime_receipt),
            "decoder_frame_bytes_sha256": decoded_frame_digest.hexdigest(),
            "decoder_fps_matches_v4a": True,
            "source_video_sha256_before": source_sha256_before,
            "source_video_sha256_after": source_sha256_after,
            "source_video_stat_stable": True,
        }
    )
    _require(
        sha256_file(runtime.asset_path) == runtime.asset_sha256,
        "Keypoint R-CNN asset changed during inference",
    )
    return final_sequence, audit
