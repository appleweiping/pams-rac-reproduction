"""Optional MediaPipe pose extraction with auditable cache provenance.

The heavy video dependencies are imported only when extraction is requested.
Importing :mod:`pams.pose`, running the CLI help, and executing unit tests
therefore do not require MediaPipe or OpenCV.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pams.data import (
    INCOMPLETE_CLIP_POLICIES,
    PoseCacheMetadata,
    load_pose_cache,
    longest_valid_span,
    normalize_clip_provenance,
    pose_cache_path,
    preprocess_native_pose_sequence,
    preprocess_pose_sequence,
    write_pose_cache,
)
from pams.types import PoseSequence

POSE_MODEL_ID = "mediapipe-pose-0.10.14"
DETECTED_SPAN_PREPROCESSING_REVISION = "detected-span-minmax-zero-span-invalid-v2"
LONGEST_TRACK_PREPROCESSING_REVISION = (
    "longest-contiguous-track-minmax-zero-span-invalid-v3"
)
OFFICIAL_SEGMENT_PREPROCESSING_REVISION = "official-segment-full-timeline-v1"
RECOVERY_PREPROCESSING_REVISION = (
    "official-segment-heavy-missing-retry-full-timeline-v4a"
)
VIDEO_RECOVERY_PREPROCESSING_REVISION = (
    "official-segment-heavy-video-fill-missing-full-timeline-v4b"
)
TASKS_MULTIPOSE_RECOVERY_PREPROCESSING_REVISION = (
    "official-segment-tasks-heavy-video-multipose4-fill-missing-full-timeline-v4c"
)
SUPPORTED_PREPROCESSING_REVISIONS = frozenset(
    {
        DETECTED_SPAN_PREPROCESSING_REVISION,
        LONGEST_TRACK_PREPROCESSING_REVISION,
        OFFICIAL_SEGMENT_PREPROCESSING_REVISION,
        RECOVERY_PREPROCESSING_REVISION,
        VIDEO_RECOVERY_PREPROCESSING_REVISION,
        TASKS_MULTIPOSE_RECOVERY_PREPROCESSING_REVISION,
    }
)


class PoseDependencyError(RuntimeError):
    """Raised when the optional pose-extraction dependencies are unavailable."""


class PoseExtractionError(RuntimeError):
    """Raised when a video cannot yield a usable pose sequence."""


@dataclass(frozen=True, slots=True)
class PoseRecoveryExtractorConfig:
    """Runtime realization of identity-bearing v4a/v4b recovery settings."""

    heavy_model_id: str
    heavy_model_asset_path: Path
    heavy_model_asset_sha256: str
    temporal_resampling: str = "none_native_timeline"
    model_complexity: int = 2
    static_image_mode: bool = True
    smooth_landmarks: bool = False
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    full_frame_retry: bool = True
    roi_retry: bool = True
    roi_margin_fraction: float = 0.20
    roi_min_side_fraction: float = 0.08
    association_cost: str = "normalized-center-l2-plus-log-scale-v1"
    association_center_weight: float = 1.0
    association_log_scale_weight: float = 0.25
    dominant_track_strategy: str = "global-viterbi-visible-extent-v1"
    maximum_gap_frames: int = 8
    maximum_gap_seconds: float = 0.25
    pose_coordinate_interpolation: bool = False
    recovery_mode: str = "missing-frame-static-plus-short-roi-v4a"

    def __post_init__(self) -> None:
        if not self.heavy_model_id.strip():
            raise ValueError("heavy_model_id must be non-empty")
        if self.model_complexity != 2:
            raise ValueError("v4 heavy retry requires model_complexity=2")
        if len(self.heavy_model_asset_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.heavy_model_asset_sha256
        ):
            raise ValueError("heavy_model_asset_sha256 must be lowercase SHA-256")
        if self.temporal_resampling != "none_native_timeline":
            raise ValueError("v4 requires temporal_resampling=none_native_timeline")
        if self.recovery_mode == "missing-frame-static-plus-short-roi-v4a":
            if not self.static_image_mode or self.smooth_landmarks:
                raise ValueError("v4a heavy retry must be static and unsmoothed")
            if not self.full_frame_retry or not self.roi_retry:
                raise ValueError("v4a requires both full-frame and ROI retries")
        elif self.recovery_mode == "full-timeline-video-fill-missing-v4b":
            if self.static_image_mode or not self.smooth_landmarks:
                raise ValueError("v4b heavy retry must be a smoothed VIDEO pass")
            if not self.full_frame_retry or self.roi_retry:
                raise ValueError("v4b requires full-timeline observation without ROI retry")
        elif self.recovery_mode == "tasks-video-multipose4-fill-missing-v4c":
            if self.static_image_mode or not self.smooth_landmarks:
                raise ValueError("v4c Tasks retry must be a VIDEO pass")
            if not self.full_frame_retry or self.roi_retry:
                raise ValueError("v4c requires full-timeline observation without ROI retry")
        else:
            raise ValueError("unsupported pose recovery mode")
        for name, value in (
            ("min_detection_confidence", self.min_detection_confidence),
            ("min_tracking_confidence", self.min_tracking_confidence),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not 0.0 <= self.roi_margin_fraction <= 1.0:
            raise ValueError("roi_margin_fraction must be in [0, 1]")
        if not 0.0 < self.roi_min_side_fraction <= 1.0:
            raise ValueError("roi_min_side_fraction must be in (0, 1]")
        if self.association_cost != "normalized-center-l2-plus-log-scale-v1":
            raise ValueError("unsupported recovery association cost")
        if self.dominant_track_strategy != "global-viterbi-visible-extent-v1":
            raise ValueError("unsupported dominant-track strategy")
        if self.association_center_weight < 0.0 or self.association_log_scale_weight < 0.0:
            raise ValueError("association weights must be non-negative")
        if self.association_center_weight == 0.0 and self.association_log_scale_weight == 0.0:
            raise ValueError("at least one association weight must be positive")
        if self.maximum_gap_frames < 1 or self.maximum_gap_seconds <= 0.0:
            raise ValueError("recovery gap bounds must be positive")
        if self.pose_coordinate_interpolation:
            raise ValueError("v4 forbids pose-coordinate interpolation")


@dataclass(frozen=True, slots=True)
class PoseExtractorConfig:
    """Frozen settings for the MediaPipe Pose video extractor."""

    target_frames: int = 256
    preprocessing_revision: str = DETECTED_SPAN_PREPROCESSING_REVISION
    model_id: str = POSE_MODEL_ID
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    smooth_landmarks: bool = True
    crop_to_detected_span: bool = True
    incomplete_clip_policy: str = "error"
    recovery: PoseRecoveryExtractorConfig | None = None

    def __post_init__(self) -> None:
        if self.target_frames < 1:
            raise ValueError("target_frames must be positive")
        if self.preprocessing_revision not in SUPPORTED_PREPROCESSING_REVISIONS:
            raise ValueError(
                "unsupported pose preprocessing revision: "
                f"{self.preprocessing_revision!r}"
            )
        if (
            self.preprocessing_revision == LONGEST_TRACK_PREPROCESSING_REVISION
            and not self.crop_to_detected_span
        ):
            raise ValueError(
                "longest-contiguous-track preprocessing requires "
                "crop_to_detected_span=true"
            )
        if (
            self.preprocessing_revision
            in {
                OFFICIAL_SEGMENT_PREPROCESSING_REVISION,
                RECOVERY_PREPROCESSING_REVISION,
                VIDEO_RECOVERY_PREPROCESSING_REVISION,
                TASKS_MULTIPOSE_RECOVERY_PREPROCESSING_REVISION,
            }
            and self.crop_to_detected_span
        ):
            raise ValueError(
                "official-segment-full-timeline preprocessing requires "
                "crop_to_detected_span=false"
            )
        if (
            self.preprocessing_revision
            not in {
                OFFICIAL_SEGMENT_PREPROCESSING_REVISION,
                RECOVERY_PREPROCESSING_REVISION,
                VIDEO_RECOVERY_PREPROCESSING_REVISION,
                TASKS_MULTIPOSE_RECOVERY_PREPROCESSING_REVISION,
            }
            and self.incomplete_clip_policy != "error"
        ):
            raise ValueError(
                "pad_invalid_tail is only valid for official-segment-full-timeline"
            )
        recovery_revision = self.preprocessing_revision in {
            RECOVERY_PREPROCESSING_REVISION,
            VIDEO_RECOVERY_PREPROCESSING_REVISION,
            TASKS_MULTIPOSE_RECOVERY_PREPROCESSING_REVISION,
        }
        if recovery_revision:
            if self.model_complexity != 1:
                raise ValueError("v4 pass0 must use model_complexity=1")
            if self.recovery is None:
                raise ValueError("v4 pose recovery requires recovery settings")
            expected_mode = (
                "missing-frame-static-plus-short-roi-v4a"
                if self.preprocessing_revision == RECOVERY_PREPROCESSING_REVISION
                else (
                    "full-timeline-video-fill-missing-v4b"
                    if self.preprocessing_revision
                    == VIDEO_RECOVERY_PREPROCESSING_REVISION
                    else "tasks-video-multipose4-fill-missing-v4c"
                )
            )
            if self.recovery.recovery_mode != expected_mode:
                raise ValueError("pose recovery mode does not match preprocessing revision")
        elif self.recovery is not None:
            raise ValueError("recovery settings are accepted only by v4a/v4b")
        if not self.model_id.strip():
            raise ValueError("model_id must be non-empty")
        if self.model_complexity not in {0, 1, 2}:
            raise ValueError("model_complexity must be 0, 1, or 2")
        for name, value in (
            ("min_detection_confidence", self.min_detection_confidence),
            ("min_tracking_confidence", self.min_tracking_confidence),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        policy = str(self.incomplete_clip_policy).strip()
        if policy not in INCOMPLETE_CLIP_POLICIES:
            raise ValueError(
                "incomplete_clip_policy must be 'error' or 'pad_invalid_tail'"
            )
        object.__setattr__(self, "incomplete_clip_policy", policy)


@dataclass(frozen=True, slots=True)
class PoseRecoveryAudit:
    """Per-video, label-free proof that v4 only filled pass0 misses."""

    source_frames: int
    expected_segment_frames: int
    decoded_segment_frames: int
    padded_tail_frames: int
    pass0_valid_frames: int
    heavy_full_frame_attempted: int
    heavy_full_frame_detected: int
    roi_retry_eligible: int
    roi_retry_attempted: int
    roi_retry_detected: int
    recovered_valid_frames: int
    final_valid_frames: int
    observed_span_frames: int
    final_longest_valid_run: int
    pass0_shared_coordinate_max_abs_error: float
    pass0_observations_preserved: bool
    pose_coordinate_interpolation: bool
    temporal_resampling: str
    heavy_model_id: str
    heavy_model_asset_sha256: str
    pass0_valid_mask_sha256: str
    final_valid_mask_sha256: str
    recovery_mode: str | None = None
    heavy_video_frames_observed: int | None = None
    heavy_video_valid_frames: int | None = None
    heavy_video_pass0_overlap_valid_frames: int | None = None
    heavy_video_fill_candidates: int | None = None
    heavy_video_valid_mask_sha256: str | None = None
    heavy_video_candidate_total: int | None = None
    heavy_video_max_candidates_per_frame: int | None = None
    heavy_video_num_poses: int | None = None
    heavy_video_timestamp_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": 1,
            "source_frames": self.source_frames,
            "expected_segment_frames": self.expected_segment_frames,
            "decoded_segment_frames": self.decoded_segment_frames,
            "padded_tail_frames": self.padded_tail_frames,
            "pass0_valid_frames": self.pass0_valid_frames,
            "heavy_full_frame_attempted": self.heavy_full_frame_attempted,
            "heavy_full_frame_detected": self.heavy_full_frame_detected,
            "roi_retry_eligible": self.roi_retry_eligible,
            "roi_retry_attempted": self.roi_retry_attempted,
            "roi_retry_detected": self.roi_retry_detected,
            "recovered_valid_frames": self.recovered_valid_frames,
            "final_valid_frames": self.final_valid_frames,
            "observed_span_frames": self.observed_span_frames,
            "final_longest_valid_run": self.final_longest_valid_run,
            "pass0_shared_coordinate_max_abs_error": (
                self.pass0_shared_coordinate_max_abs_error
            ),
            "pass0_observations_preserved": self.pass0_observations_preserved,
            "pose_coordinate_interpolation": self.pose_coordinate_interpolation,
            "temporal_resampling": self.temporal_resampling,
            "heavy_model_id": self.heavy_model_id,
            "heavy_model_asset_sha256": self.heavy_model_asset_sha256,
            "pass0_valid_mask_sha256": self.pass0_valid_mask_sha256,
            "final_valid_mask_sha256": self.final_valid_mask_sha256,
        }
        optional = {
            "recovery_mode": self.recovery_mode,
            "heavy_video_frames_observed": self.heavy_video_frames_observed,
            "heavy_video_valid_frames": self.heavy_video_valid_frames,
            "heavy_video_pass0_overlap_valid_frames": (
                self.heavy_video_pass0_overlap_valid_frames
            ),
            "heavy_video_fill_candidates": self.heavy_video_fill_candidates,
            "heavy_video_valid_mask_sha256": self.heavy_video_valid_mask_sha256,
            "heavy_video_candidate_total": self.heavy_video_candidate_total,
            "heavy_video_max_candidates_per_frame": (
                self.heavy_video_max_candidates_per_frame
            ),
            "heavy_video_num_poses": self.heavy_video_num_poses,
            "heavy_video_timestamp_sha256": self.heavy_video_timestamp_sha256,
        }
        payload.update({key: value for key, value in optional.items() if value is not None})
        return payload


@dataclass(frozen=True, slots=True)
class PoseExtractionSummary:
    """Small, JSON-compatible summary of one extracted cache."""

    video_id: str
    video_path: str
    cache_path: str
    video_sha256: str
    pose_fingerprint: str
    source_frames: int | None
    source_valid_frames: int | None
    selected_source_frames: int | None
    cached_frames: int
    cached_valid_frames: int
    fps: float
    pose_model: str
    annotation_sha256: str | None = None
    clip_start_frame: int | None = None
    clip_end_frame: int | None = None
    expected_clip_frames: int | None = None
    decoded_clip_frames: int | None = None
    padded_tail_frames: int | None = None
    incomplete_clip_policy: str | None = None
    skipped: bool = False
    recovery_audit: PoseRecoveryAudit | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "cache_path": self.cache_path,
            "video_sha256": self.video_sha256,
            "pose_fingerprint": self.pose_fingerprint,
            "source_frames": self.source_frames,
            "source_valid_frames": self.source_valid_frames,
            "selected_source_frames": self.selected_source_frames,
            "cached_frames": self.cached_frames,
            "cached_valid_frames": self.cached_valid_frames,
            "fps": self.fps,
            "pose_model": self.pose_model,
            "annotation_sha256": self.annotation_sha256,
            "clip_start_frame": self.clip_start_frame,
            "clip_end_frame": self.clip_end_frame,
            "expected_clip_frames": self.expected_clip_frames,
            "decoded_clip_frames": self.decoded_clip_frames,
            "padded_tail_frames": self.padded_tail_frames,
            "incomplete_clip_policy": self.incomplete_clip_policy,
            "skipped": self.skipped,
        }
        if self.recovery_audit is not None:
            payload["recovery_audit"] = self.recovery_audit.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class PoseExtractionFailure:
    """One explicit decode/extraction failure for the shared failure ledger."""

    video_id: str
    video_path: str
    error_type: str
    message: str
    annotation_sha256: str | None = None
    clip_start_frame: int | None = None
    clip_end_frame: int | None = None

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "error_type": self.error_type,
            "message": self.message,
            "annotation_sha256": self.annotation_sha256,
            "clip_start_frame": self.clip_start_frame,
            "clip_end_frame": self.clip_end_frame,
        }


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a local file."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_pose_dependencies() -> tuple[Any, Any]:
    """Load OpenCV and MediaPipe, translating import failures clearly."""

    try:
        import cv2  # type: ignore[import-not-found]
        import mediapipe as mp  # type: ignore[import-not-found,import-untyped]
    except (ImportError, ModuleNotFoundError) as exc:
        raise PoseDependencyError(
            "pose extraction requires optional dependencies; install with "
            '`python -m pip install -e ".[pose]"`'
        ) from exc
    if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "pose"):
        raise PoseDependencyError(
            "the installed mediapipe package does not expose solutions.pose; "
            "install the pinned pose extra"
        )
    return cv2, mp


def _candidate_score(candidate: NDArray[np.float64]) -> float:
    """Score a detected person by visible spatial extent.

    MediaPipe's legacy Pose API is single-person, but this helper accepts
    multi-person-shaped inputs so that any future detector still makes a
    deterministic dominant-subject choice. The largest well-visible body is
    selected; ties retain the earliest detector result.
    """

    xyz = candidate[:, :3]
    if not np.isfinite(xyz).all():
        return float("-inf")
    visibility = (
        np.clip(candidate[:, 3], 0.0, 1.0)
        if candidate.shape[1] >= 4
        else np.ones(candidate.shape[0], dtype=np.float64)
    )
    visible = visibility >= 0.1
    if not np.any(visible):
        return float("-inf")
    xy = xyz[visible, :2]
    extent = np.ptp(xy, axis=0)
    spatial_extent = max(float(extent[0] * extent[1]), float(np.linalg.norm(extent)), 1e-9)
    return spatial_extent * float(np.mean(visibility[visible]))


def select_dominant_pose(candidates: ArrayLike | None) -> NDArray[np.float32] | None:
    """Choose one deterministic 33-landmark subject from detector candidates.

    ``candidates`` may have shape ``[33, 3+]`` or ``[people, 33, 3+]``.
    Only XYZ is returned. Invalid candidates are ignored and an absent usable
    person returns ``None``.
    """

    if candidates is None:
        return None
    array = np.asarray(candidates, dtype=np.float64)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[1] != 33 or array.shape[2] < 3:
        raise ValueError("pose candidates must have shape [33, 3+] or [people, 33, 3+]")
    if array.shape[0] == 0:
        return None

    scores = np.asarray([_candidate_score(candidate) for candidate in array])
    index = int(np.argmax(scores))
    if not np.isfinite(scores[index]):
        return None
    return np.asarray(array[index, :, :3], dtype=np.float32)


def _candidate_array(candidates: ArrayLike | None) -> NDArray[np.float32]:
    """Canonicalize a frame's candidates while retaining visibility."""

    if candidates is None:
        return np.empty((0, 33, 4), dtype=np.float32)
    array = np.asarray(candidates, dtype=np.float32)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[1] != 33 or array.shape[2] < 3:
        raise ValueError("pose candidates must have shape [33, 3+] or [people, 33, 3+]")
    if array.shape[2] == 3:
        visibility = np.ones((*array.shape[:2], 1), dtype=np.float32)
        array = np.concatenate((array, visibility), axis=2)
    canonical = np.asarray(array[:, :, :4], dtype=np.float32)
    keep = np.asarray(
        [np.isfinite(_candidate_score(candidate)) for candidate in canonical],
        dtype=np.bool_,
    )
    return canonical[keep]


def _candidate_bbox(candidate: NDArray[np.float32]) -> tuple[float, float, float, float]:
    """Return an unclipped normalized bounding box for association."""

    visibility = np.clip(candidate[:, 3], 0.0, 1.0)
    visible = visibility >= 0.1
    if not np.any(visible):
        raise ValueError("candidate has no visible landmarks")
    xy = np.asarray(candidate[visible, :2], dtype=np.float64)
    return (
        float(np.min(xy[:, 0])),
        float(np.min(xy[:, 1])),
        float(np.max(xy[:, 0])),
        float(np.max(xy[:, 1])),
    )


def _expanded_bbox(
    candidate: NDArray[np.float32],
    *,
    margin_fraction: float,
    minimum_side_fraction: float,
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = _candidate_bbox(candidate)
    center_x = 0.5 * (x0 + x1)
    center_y = 0.5 * (y0 + y1)
    width = max(x1 - x0, minimum_side_fraction)
    height = max(y1 - y0, minimum_side_fraction)
    width *= 1.0 + 2.0 * margin_fraction
    height *= 1.0 + 2.0 * margin_fraction
    x0 = max(0.0, center_x - 0.5 * width)
    x1 = min(1.0, center_x + 0.5 * width)
    y0 = max(0.0, center_y - 0.5 * height)
    y1 = min(1.0, center_y + 0.5 * height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("candidate produced a degenerate ROI")
    return x0, y0, x1, y1


def _transition_cost(
    previous: NDArray[np.float32],
    current: NDArray[np.float32],
    *,
    frame_gap: int,
    center_weight: float,
    log_scale_weight: float,
) -> float:
    previous_box = _candidate_bbox(previous)
    current_box = _candidate_bbox(current)
    previous_center = np.asarray(
        (
            0.5 * (previous_box[0] + previous_box[2]),
            0.5 * (previous_box[1] + previous_box[3]),
        ),
        dtype=np.float64,
    )
    current_center = np.asarray(
        (
            0.5 * (current_box[0] + current_box[2]),
            0.5 * (current_box[1] + current_box[3]),
        ),
        dtype=np.float64,
    )
    previous_scale = max(
        math.hypot(previous_box[2] - previous_box[0], previous_box[3] - previous_box[1]),
        1e-6,
    )
    current_scale = max(
        math.hypot(current_box[2] - current_box[0], current_box[3] - current_box[1]),
        1e-6,
    )
    normalization = max(0.5 * (previous_scale + current_scale), 1e-6)
    elapsed = max(int(frame_gap), 1)
    center_cost = float(np.linalg.norm(current_center - previous_center)) / (
        normalization * elapsed
    )
    scale_cost = abs(math.log(current_scale / previous_scale)) / elapsed
    return center_weight * center_cost + log_scale_weight * scale_cost


def select_global_dominant_track(
    frame_candidates: Sequence[ArrayLike | None],
    *,
    center_weight: float = 1.0,
    log_scale_weight: float = 0.25,
) -> tuple[NDArray[np.float32] | None, ...]:
    """Select one deterministic subject with a whole-video Viterbi path.

    Every non-empty frame contributes exactly one node to the path. Missing
    frames remain missing. Candidate order is the final deterministic tie
    break, so equal inputs always produce byte-identical tracks.
    """

    if not frame_candidates:
        raise PoseExtractionError("video contains no decoded frames")
    if center_weight < 0.0 or log_scale_weight < 0.0:
        raise ValueError("association weights must be non-negative")
    if center_weight == 0.0 and log_scale_weight == 0.0:
        raise ValueError("at least one association weight must be positive")
    candidates = tuple(_candidate_array(item) for item in frame_candidates)
    observed = tuple(index for index, item in enumerate(candidates) if len(item))
    selected: list[NDArray[np.float32] | None] = [None] * len(candidates)
    if not observed:
        return tuple(selected)

    score_history: list[NDArray[np.float64]] = []
    parent_history: list[NDArray[np.int64]] = []
    first = candidates[observed[0]]
    first_scores = np.asarray(
        [math.log(max(_candidate_score(item), 1e-12)) for item in first],
        dtype=np.float64,
    )
    score_history.append(first_scores)
    parent_history.append(np.full(len(first), -1, dtype=np.int64))

    for observed_offset in range(1, len(observed)):
        previous_index = observed[observed_offset - 1]
        current_index = observed[observed_offset]
        previous_candidates = candidates[previous_index]
        current_candidates = candidates[current_index]
        previous_scores = score_history[-1]
        current_scores = np.empty(len(current_candidates), dtype=np.float64)
        parents = np.empty(len(current_candidates), dtype=np.int64)
        for current_offset, current in enumerate(current_candidates):
            transition_scores = np.asarray(
                [
                    previous_scores[previous_offset]
                    - _transition_cost(
                        previous,
                        current,
                        frame_gap=current_index - previous_index,
                        center_weight=center_weight,
                        log_scale_weight=log_scale_weight,
                    )
                    for previous_offset, previous in enumerate(previous_candidates)
                ],
                dtype=np.float64,
            )
            parent = int(np.argmax(transition_scores))
            parents[current_offset] = parent
            current_scores[current_offset] = transition_scores[parent] + math.log(
                max(_candidate_score(current), 1e-12)
            )
        score_history.append(current_scores)
        parent_history.append(parents)

    chosen = int(np.argmax(score_history[-1]))
    for observed_offset in range(len(observed) - 1, -1, -1):
        frame_index = observed[observed_offset]
        selected[frame_index] = np.asarray(
            candidates[frame_index][chosen],
            dtype=np.float32,
        )
        chosen = int(parent_history[observed_offset][chosen])
    return tuple(selected)


def _valid_mask_sha256(track: Sequence[NDArray[np.float32] | None]) -> str:
    mask = bytes(1 if candidate is not None else 0 for candidate in track)
    return hashlib.sha256(mask).hexdigest()


def _longest_true_run(mask: NDArray[np.bool_]) -> int:
    longest = 0
    current = 0
    for value in mask:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def _short_gap_rois(
    pass0_track: Sequence[NDArray[np.float32] | None],
    *,
    fps: float,
    maximum_gap_frames: int,
    maximum_gap_seconds: float,
    margin_fraction: float,
    minimum_side_fraction: float,
) -> dict[int, tuple[float, float, float, float]]:
    """Interpolate only ROI boxes across short, bilaterally anchored gaps."""

    maximum = min(
        int(maximum_gap_frames),
        int(math.floor(float(maximum_gap_seconds) * float(fps) + 1e-12)),
    )
    if maximum < 1:
        return {}
    rois: dict[int, tuple[float, float, float, float]] = {}
    index = 0
    while index < len(pass0_track):
        if pass0_track[index] is not None:
            index += 1
            continue
        start = index
        while index < len(pass0_track) and pass0_track[index] is None:
            index += 1
        stop = index
        gap = stop - start
        left_index = start - 1
        right_index = stop
        if gap > maximum or left_index < 0 or right_index >= len(pass0_track):
            continue
        left = pass0_track[left_index]
        right = pass0_track[right_index]
        if left is None or right is None:
            continue
        left_box = np.asarray(
            _expanded_bbox(
                left,
                margin_fraction=margin_fraction,
                minimum_side_fraction=minimum_side_fraction,
            ),
            dtype=np.float64,
        )
        right_box = np.asarray(
            _expanded_bbox(
                right,
                margin_fraction=margin_fraction,
                minimum_side_fraction=minimum_side_fraction,
            ),
            dtype=np.float64,
        )
        for offset, frame_index in enumerate(range(start, stop), start=1):
            fraction = offset / (gap + 1)
            box = left_box * (1.0 - fraction) + right_box * fraction
            rois[frame_index] = tuple(float(value) for value in box)
    return rois


def _remap_roi_candidate(
    candidate: NDArray[np.float32] | None,
    roi: tuple[float, float, float, float],
) -> NDArray[np.float32] | None:
    if candidate is None:
        return None
    output = np.array(candidate, dtype=np.float32, copy=True)
    x0, y0, x1, y1 = roi
    width = x1 - x0
    height = y1 - y0
    output[:, 0] = x0 + output[:, 0] * width
    output[:, 1] = y0 + output[:, 1] * height
    output[:, 2] *= width
    return output


def assemble_pose_sequence(
    frames: Sequence[ArrayLike | None],
    *,
    video_id: str,
    fps: float,
) -> PoseSequence:
    """Convert per-frame candidates to a zero-filled, frame-masked sequence."""

    if not frames:
        raise PoseExtractionError("video contains no decoded frames")
    xyz = np.zeros((len(frames), 33, 3), dtype=np.float32)
    valid = np.zeros(len(frames), dtype=np.bool_)
    for index, candidates in enumerate(frames):
        dominant = select_dominant_pose(candidates)
        if dominant is not None:
            xyz[index] = dominant
            valid[index] = True
    return PoseSequence(video_id=video_id, fps=fps, xyz=xyz, valid_mask=valid)


def trim_to_detected_span(sequence: PoseSequence) -> PoseSequence:
    """Trim only leading/trailing misses while preserving internal gaps."""

    valid_indices = np.flatnonzero(sequence.valid_mask)
    if not len(valid_indices):
        raise PoseExtractionError(f"no valid pose was detected in video {sequence.video_id!r}")
    span = slice(int(valid_indices[0]), int(valid_indices[-1]) + 1)
    return PoseSequence(
        video_id=sequence.video_id,
        fps=sequence.fps,
        xyz=sequence.xyz[span],
        valid_mask=sequence.valid_mask[span],
    )


def trim_to_longest_contiguous_track(sequence: PoseSequence) -> PoseSequence:
    """Select the earliest longest uninterrupted detected trajectory.

    MediaPipe's legacy video API emits at most one person per frame. Its
    deterministic main-subject trajectory is therefore the longest
    contiguous run of valid detections. Equal-length runs choose the earliest
    occurrence.
    """

    try:
        span = longest_valid_span(sequence.valid_mask)
    except ValueError:
        raise PoseExtractionError(
            f"no valid pose was detected in video {sequence.video_id!r}"
        ) from None
    selected_mask = sequence.valid_mask[span]
    if not np.all(selected_mask):
        raise RuntimeError("longest contiguous trajectory contains an invalid frame")
    return PoseSequence(
        video_id=sequence.video_id,
        fps=sequence.fps,
        xyz=sequence.xyz[span],
        valid_mask=selected_mask,
    )


def _select_preprocessing_span(
    sequence: PoseSequence,
    *,
    crop_to_detected_span: bool,
    preprocessing_revision: str,
) -> tuple[PoseSequence, int]:
    """Return the selected raw sequence and its detected-run length."""

    if preprocessing_revision not in SUPPORTED_PREPROCESSING_REVISIONS:
        raise ValueError(
            f"unsupported pose preprocessing revision: {preprocessing_revision!r}"
        )
    if (
        preprocessing_revision == LONGEST_TRACK_PREPROCESSING_REVISION
        and not crop_to_detected_span
    ):
        raise ValueError(
            "track preprocessing requires "
            "crop_to_detected_span=true"
        )
    if not np.any(sequence.valid_mask):
        return sequence, 0
    if not crop_to_detected_span:
        return sequence, sequence.num_frames
    if preprocessing_revision == LONGEST_TRACK_PREPROCESSING_REVISION:
        selected = trim_to_longest_contiguous_track(sequence)
        if selected.num_frames == 1:
            # A one-frame observation has no defined temporal duration. Keep
            # the original video duration but mark every frame invalid rather
            # than manufacturing a 256-frame static trajectory.
            invalid = PoseSequence(
                video_id=sequence.video_id,
                fps=sequence.fps,
                xyz=np.zeros_like(sequence.xyz, dtype=np.float32),
                valid_mask=np.zeros(sequence.num_frames, dtype=np.bool_),
            )
            return invalid, 1
        return selected, selected.num_frames
    selected = trim_to_detected_span(sequence)
    return selected, selected.num_frames


def preprocess_extracted_pose(
    sequence: PoseSequence,
    *,
    target_frames: int = 256,
    crop_to_detected_span: bool = True,
    preprocessing_revision: str = DETECTED_SPAN_PREPROCESSING_REVISION,
) -> PoseSequence:
    """Apply the frozen track selection, normalization, and uniform sampling."""

    selected, _ = _select_preprocessing_span(
        sequence,
        crop_to_detected_span=crop_to_detected_span,
        preprocessing_revision=preprocessing_revision,
    )
    return preprocess_pose_sequence(selected, target_frames=target_frames)


def _landmarks_from_result(result: Any) -> NDArray[np.float32] | None:
    landmarks = getattr(result, "pose_landmarks", None)
    if landmarks is None:
        return None
    values = getattr(landmarks, "landmark", None)
    if values is None:
        return None
    rows = tuple(values)
    if len(rows) != 33:
        return None
    candidate = np.asarray(
        [
            (
                float(landmark.x),
                float(landmark.y),
                float(landmark.z),
                float(getattr(landmark, "visibility", 1.0)),
            )
            for landmark in rows
        ],
        dtype=np.float32,
    )
    return candidate


def _verify_heavy_asset(
    recovery: PoseRecoveryExtractorConfig,
    mp: Any,
    *,
    expected_resource_path: Path | None = None,
) -> Path:
    """Fail closed unless the verified asset is already at MediaPipe's path."""

    supplied = recovery.heavy_model_asset_path.resolve(strict=True)
    if expected_resource_path is None:
        module_file = getattr(mp, "__file__", None)
        if not module_file:
            raise PoseDependencyError("cannot locate the installed MediaPipe package")
        expected_resource_path = (
            Path(module_file).resolve().parent
            / "modules"
            / "pose_landmark"
            / "pose_landmark_heavy.tflite"
        )
    expected = expected_resource_path.resolve(strict=True)
    if supplied != expected:
        raise PoseDependencyError(
            "the verified heavy asset must be bind-mounted directly at "
            f"MediaPipe's resource path: {expected}"
        )
    received = sha256_file(expected)
    if received != recovery.heavy_model_asset_sha256:
        raise PoseDependencyError(
            "heavy pose asset SHA-256 mismatch: "
            f"expected {recovery.heavy_model_asset_sha256}, received {received}"
        )
    return expected


def _verify_tasks_heavy_asset(recovery: PoseRecoveryExtractorConfig) -> Path:
    """Fail closed on the explicit official Tasks PoseLandmarker bundle."""

    supplied = recovery.heavy_model_asset_path.resolve(strict=True)
    if supplied.suffix.casefold() != ".task":
        raise PoseDependencyError("v4c Tasks recovery requires a .task model bundle")
    received = sha256_file(supplied)
    if received != recovery.heavy_model_asset_sha256:
        raise PoseDependencyError(
            "Tasks heavy pose asset SHA-256 mismatch: "
            f"expected {recovery.heavy_model_asset_sha256}, received {received}"
        )
    return supplied


def _tasks_candidates_from_result(result: Any) -> NDArray[np.float32] | None:
    """Convert every Tasks pose candidate to the shared [P,33,4] form."""

    poses = getattr(result, "pose_landmarks", None)
    if poses is None:
        return None
    candidates: list[NDArray[np.float32]] = []
    for pose in tuple(poses):
        landmarks = tuple(pose)
        if len(landmarks) != 33:
            continue
        candidate = np.asarray(
            [
                (
                    float(landmark.x),
                    float(landmark.y),
                    float(landmark.z),
                    float(
                        1.0
                        if getattr(landmark, "visibility", None) is None
                        else landmark.visibility
                    ),
                )
                for landmark in landmarks
            ],
            dtype=np.float32,
        )
        if len(_candidate_array(candidate)):
            candidates.append(candidate)
    if not candidates:
        return None
    return np.stack(candidates, axis=0)


def _canonicalize_tasks_candidate_order(
    candidates: ArrayLike | None,
) -> NDArray[np.float32] | None:
    """Make Tasks detector output-order permutations observationally neutral."""

    array = _candidate_array(candidates)
    if not len(array):
        return None
    order = sorted(
        range(len(array)),
        key=lambda index: hashlib.sha256(
            np.ascontiguousarray(array[index], dtype=np.float32).tobytes()
        ).digest(),
    )
    return np.asarray(array[order], dtype=np.float32)


def _strict_video_timestamp_ms(frame_offset: int, fps: float, previous: int) -> int:
    """Return a deterministic VIDEO timestamp that is strictly increasing."""

    if frame_offset < 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid frame offset or FPS for VIDEO timestamp")
    nominal = int(round(frame_offset * 1000.0 / fps))
    return max(nominal, previous + 1)


def _crop_normalized_roi(
    frame: NDArray[np.uint8],
    roi: tuple[float, float, float, float],
) -> tuple[NDArray[np.uint8], tuple[float, float, float, float]]:
    height, width = frame.shape[:2]
    if height < 1 or width < 1:
        raise PoseExtractionError("decoded video frame is empty")
    x0, y0, x1, y1 = roi
    left = max(0, min(width - 1, int(math.floor(x0 * width))))
    top = max(0, min(height - 1, int(math.floor(y0 * height))))
    right = max(left + 1, min(width, int(math.ceil(x1 * width))))
    bottom = max(top + 1, min(height, int(math.ceil(y1 * height))))
    crop = np.asarray(frame[top:bottom, left:right])
    actual = (left / width, top / height, right / width, bottom / height)
    return crop, actual


def _process_pose_frame(detector: Any, cv2: Any, frame: NDArray[np.uint8]) -> NDArray[np.float32] | None:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return _landmarks_from_result(detector.process(rgb))


def extract_pose_sequence_recovery(
    video_path: str | Path,
    *,
    video_id: str | None = None,
    config: PoseExtractorConfig,
    clip_start_frame: int | None = None,
    clip_end_frame: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> tuple[PoseSequence, int, int, int, PoseRecoveryAudit]:
    """Run v4a pass0 plus verified heavy retries on pass0-missing frames.

    Pose coordinates are never interpolated. A short missing run may receive
    an interpolated crop box only when both pass0 anchors exist. Pass0 poses
    are locked into the final global path and checked to an absolute tolerance
    of ``1e-6`` before any cache can be returned.
    """

    if config.preprocessing_revision == TASKS_MULTIPOSE_RECOVERY_PREPROCESSING_REVISION:
        return extract_pose_sequence_tasks_recovery(
            video_path,
            video_id=video_id,
            config=config,
            clip_start_frame=clip_start_frame,
            clip_end_frame=clip_end_frame,
            progress=progress,
        )
    if config.preprocessing_revision == VIDEO_RECOVERY_PREPROCESSING_REVISION:
        return extract_pose_sequence_video_recovery(
            video_path,
            video_id=video_id,
            config=config,
            clip_start_frame=clip_start_frame,
            clip_end_frame=clip_end_frame,
            progress=progress,
        )
    if config.preprocessing_revision != RECOVERY_PREPROCESSING_REVISION:
        raise ValueError("recovery extraction requires the v4a preprocessing revision")
    recovery = config.recovery
    if recovery is None:
        raise ValueError("v4a recovery settings are required")
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"video does not exist: {source}")
    identifier = str(video_id or source.stem).strip()
    if not identifier:
        raise ValueError("video_id must be non-empty")
    if (clip_start_frame is None) != (clip_end_frame is None):
        raise ValueError("clip_start_frame and clip_end_frame must be supplied together")
    if clip_start_frame is None or clip_end_frame is None:
        raise ValueError("v4a recovery requires an official-segment input clip")
    if isinstance(clip_start_frame, bool | np.bool_) or isinstance(
        clip_end_frame, bool | np.bool_
    ):
        raise TypeError("clip frame indices must be integers, not bool")
    clip_start_frame = int(clip_start_frame)
    clip_end_frame = int(clip_end_frame)
    if clip_start_frame < 0 or clip_end_frame <= clip_start_frame:
        raise ValueError("clip must be a non-empty 0-based half-open interval")
    expected_frames = clip_end_frame - clip_start_frame

    cv2, mp = _load_pose_dependencies()
    heavy_asset = _verify_heavy_asset(recovery, mp)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise PoseExtractionError(f"OpenCV could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise PoseExtractionError(f"video reports an invalid FPS: {source}")

    pass0_candidates: list[NDArray[np.float32] | None] = []
    retry_frames: dict[int, NDArray[np.uint8]] = {}
    decoded_frames = 0
    try:
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=config.model_complexity,
            smooth_landmarks=config.smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        ) as detector:
            if clip_start_frame:
                seek_ok = bool(capture.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame))
                if not seek_ok:
                    raise PoseExtractionError(
                        f"OpenCV could not seek to clip start frame {clip_start_frame} "
                        f"for {source}"
                    )
                positioned = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
                if np.isfinite(positioned) and abs(positioned - clip_start_frame) > 0.5:
                    raise PoseExtractionError(
                        f"OpenCV seek coverage mismatch for {source}: requested frame "
                        f"{clip_start_frame}, positioned at {positioned}"
                    )
            while decoded_frames < expected_frames:
                ok, frame = capture.read()
                if not ok:
                    missing_tail = expected_frames - decoded_frames
                    if config.incomplete_clip_policy == "error":
                        raise PoseExtractionError(
                            f"incomplete official clip decode for {identifier!r}: "
                            f"range=[{clip_start_frame},{clip_end_frame}), "
                            f"expected={expected_frames}, decoded={decoded_frames}, "
                            f"missing_tail={missing_tail}"
                        )
                    pass0_candidates.extend([None] * missing_tail)
                    break
                candidate = _process_pose_frame(detector, cv2, frame)
                frame_index = len(pass0_candidates)
                pass0_candidates.append(candidate)
                if not len(_candidate_array(candidate)):
                    retry_frames[frame_index] = np.array(frame, copy=True)
                decoded_frames += 1
                if progress is not None:
                    progress(decoded_frames)
    finally:
        capture.release()
    if len(pass0_candidates) != expected_frames:
        raise RuntimeError("v4a pass0 timeline does not equal the official segment length")

    pass0_track = select_global_dominant_track(
        pass0_candidates,
        center_weight=recovery.association_center_weight,
        log_scale_weight=recovery.association_log_scale_weight,
    )
    missing = tuple(index for index, candidate in enumerate(pass0_track) if candidate is None)
    retryable_missing = tuple(index for index in missing if index in retry_frames)
    if not set(retry_frames).issubset(missing):
        raise RuntimeError("pass0 retry-frame ledger contains a valid global-track frame")

    full_results: dict[int, NDArray[np.float32]] = {}
    roi_results: dict[int, NDArray[np.float32]] = {}
    all_gap_rois = _short_gap_rois(
        pass0_track,
        fps=fps,
        maximum_gap_frames=recovery.maximum_gap_frames,
        maximum_gap_seconds=recovery.maximum_gap_seconds,
        margin_fraction=recovery.roi_margin_fraction,
        minimum_side_fraction=recovery.roi_min_side_fraction,
    )
    gap_rois = {
        index: roi for index, roi in all_gap_rois.items() if index in retry_frames
    }
    roi_attempted = 0
    with mp.solutions.pose.Pose(
        static_image_mode=recovery.static_image_mode,
        model_complexity=recovery.model_complexity,
        smooth_landmarks=recovery.smooth_landmarks,
        enable_segmentation=False,
        min_detection_confidence=recovery.min_detection_confidence,
        min_tracking_confidence=recovery.min_tracking_confidence,
    ) as heavy_detector:
        for frame_index in retryable_missing:
            candidate = _process_pose_frame(
                heavy_detector,
                cv2,
                retry_frames[frame_index],
            )
            if len(_candidate_array(candidate)):
                assert candidate is not None
                full_results[frame_index] = candidate
        for frame_index in retryable_missing:
            if frame_index in full_results or frame_index not in gap_rois:
                continue
            roi_attempted += 1
            crop, actual_roi = _crop_normalized_roi(
                retry_frames[frame_index],
                gap_rois[frame_index],
            )
            candidate = _process_pose_frame(heavy_detector, cv2, crop)
            remapped = _remap_roi_candidate(candidate, actual_roi)
            if len(_candidate_array(remapped)):
                assert remapped is not None
                roi_results[frame_index] = remapped

    if sha256_file(heavy_asset) != recovery.heavy_model_asset_sha256:
        raise RuntimeError("heavy pose asset changed during extraction")
    combined: list[NDArray[np.float32] | None] = []
    for frame_index, pass0_candidate in enumerate(pass0_track):
        if pass0_candidate is not None:
            combined.append(pass0_candidate)
        elif frame_index in full_results:
            combined.append(full_results[frame_index])
        else:
            combined.append(roi_results.get(frame_index))
    final_track = select_global_dominant_track(
        combined,
        center_weight=recovery.association_center_weight,
        log_scale_weight=recovery.association_log_scale_weight,
    )
    shared_errors = [
        float(np.max(np.abs(final_track[index][:, :3] - candidate[:, :3])))
        for index, candidate in enumerate(pass0_track)
        if candidate is not None and final_track[index] is not None
    ]
    pass0_valid = sum(candidate is not None for candidate in pass0_track)
    final_valid = sum(candidate is not None for candidate in final_track)
    shared_max_error = max(shared_errors, default=0.0)
    preserved = len(shared_errors) == pass0_valid and shared_max_error <= 1e-6
    if not preserved or final_valid < pass0_valid:
        raise RuntimeError("v4a recovery altered or removed a pass0 observation")

    raw = assemble_pose_sequence(final_track, video_id=identifier, fps=fps)
    valid_indices = np.flatnonzero(raw.valid_mask)
    observed_span = (
        int(valid_indices[-1] - valid_indices[0] + 1) if valid_indices.size else 0
    )
    # The official annotation-bound segment is the complete timebase. Leading,
    # trailing, and internal misses stay in place as invalid frames.
    selected_source_frames = raw.num_frames
    processed = preprocess_native_pose_sequence(raw)
    if processed.num_frames != expected_frames or processed.fps != raw.fps:
        raise RuntimeError("v4a preprocessing changed the native official-segment timeline")
    audit = PoseRecoveryAudit(
        source_frames=raw.num_frames,
        expected_segment_frames=expected_frames,
        decoded_segment_frames=decoded_frames,
        padded_tail_frames=expected_frames - decoded_frames,
        pass0_valid_frames=pass0_valid,
        heavy_full_frame_attempted=len(retryable_missing),
        heavy_full_frame_detected=len(full_results),
        roi_retry_eligible=len(gap_rois),
        roi_retry_attempted=roi_attempted,
        roi_retry_detected=len(roi_results),
        recovered_valid_frames=final_valid - pass0_valid,
        final_valid_frames=final_valid,
        observed_span_frames=observed_span,
        final_longest_valid_run=_longest_true_run(raw.valid_mask),
        pass0_shared_coordinate_max_abs_error=shared_max_error,
        pass0_observations_preserved=preserved,
        pose_coordinate_interpolation=recovery.pose_coordinate_interpolation,
        temporal_resampling=recovery.temporal_resampling,
        heavy_model_id=recovery.heavy_model_id,
        heavy_model_asset_sha256=recovery.heavy_model_asset_sha256,
        pass0_valid_mask_sha256=_valid_mask_sha256(pass0_track),
        final_valid_mask_sha256=_valid_mask_sha256(final_track),
    )
    return (
        processed,
        decoded_frames,
        pass0_valid,
        selected_source_frames,
        audit,
    )


def extract_pose_sequence_video_recovery(
    video_path: str | Path,
    *,
    video_id: str | None = None,
    config: PoseExtractorConfig,
    clip_start_frame: int | None = None,
    clip_end_frame: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> tuple[PoseSequence, int, int, int, PoseRecoveryAudit]:
    """Run v4b pass0 plus a heavy VIDEO pass over the decoded timeline.

    The heavy detector observes every decoded frame in order so its tracking
    state is continuous. Its coordinates can fill only pass0-invalid frames;
    every pass0-valid coordinate is locked exactly. Early-EOF padding is never
    presented to either detector and remains an invalid native-timeline tail.
    """

    if config.preprocessing_revision != VIDEO_RECOVERY_PREPROCESSING_REVISION:
        raise ValueError("video recovery requires the v4b preprocessing revision")
    recovery = config.recovery
    if recovery is None:
        raise ValueError("v4b recovery settings are required")
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"video does not exist: {source}")
    identifier = str(video_id or source.stem).strip()
    if not identifier:
        raise ValueError("video_id must be non-empty")
    if (clip_start_frame is None) != (clip_end_frame is None):
        raise ValueError("clip_start_frame and clip_end_frame must be supplied together")
    if clip_start_frame is None or clip_end_frame is None:
        raise ValueError("v4b recovery requires an official-segment input clip")
    if isinstance(clip_start_frame, bool | np.bool_) or isinstance(
        clip_end_frame, bool | np.bool_
    ):
        raise TypeError("clip frame indices must be integers, not bool")
    clip_start_frame = int(clip_start_frame)
    clip_end_frame = int(clip_end_frame)
    if clip_start_frame < 0 or clip_end_frame <= clip_start_frame:
        raise ValueError("clip must be a non-empty 0-based half-open interval")
    expected_frames = clip_end_frame - clip_start_frame

    cv2, mp = _load_pose_dependencies()
    heavy_asset = _verify_heavy_asset(recovery, mp)

    def open_positioned_capture() -> tuple[Any, float]:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise PoseExtractionError(f"OpenCV could not open video: {source}")
        capture_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(capture_fps) or capture_fps <= 0:
            capture.release()
            raise PoseExtractionError(f"video reports an invalid FPS: {source}")
        if clip_start_frame:
            seek_ok = bool(capture.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame))
            if not seek_ok:
                capture.release()
                raise PoseExtractionError(
                    f"OpenCV could not seek to clip start frame {clip_start_frame} "
                    f"for {source}"
                )
            positioned = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if np.isfinite(positioned) and abs(positioned - clip_start_frame) > 0.5:
                capture.release()
                raise PoseExtractionError(
                    f"OpenCV seek coverage mismatch for {source}: requested frame "
                    f"{clip_start_frame}, positioned at {positioned}"
                )
        return capture, capture_fps

    pass0_candidates: list[NDArray[np.float32] | None] = []
    decoded_frames = 0
    capture, fps = open_positioned_capture()
    try:
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=config.model_complexity,
            smooth_landmarks=config.smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        ) as detector:
            while decoded_frames < expected_frames:
                ok, frame = capture.read()
                if not ok:
                    missing_tail = expected_frames - decoded_frames
                    if config.incomplete_clip_policy == "error":
                        raise PoseExtractionError(
                            f"incomplete official clip decode for {identifier!r}: "
                            f"range=[{clip_start_frame},{clip_end_frame}), "
                            f"expected={expected_frames}, decoded={decoded_frames}, "
                            f"missing_tail={missing_tail}"
                        )
                    pass0_candidates.extend([None] * missing_tail)
                    break
                pass0_candidates.append(_process_pose_frame(detector, cv2, frame))
                decoded_frames += 1
                if progress is not None:
                    progress(decoded_frames)
    finally:
        capture.release()
    if len(pass0_candidates) != expected_frames:
        raise RuntimeError("v4b pass0 timeline does not equal the official segment length")

    pass0_track = select_global_dominant_track(
        pass0_candidates,
        center_weight=recovery.association_center_weight,
        log_scale_weight=recovery.association_log_scale_weight,
    )
    heavy_candidates: list[NDArray[np.float32] | None] = []
    heavy_capture, heavy_fps = open_positioned_capture()
    if not math.isclose(heavy_fps, fps, rel_tol=0.0, abs_tol=1e-6):
        heavy_capture.release()
        raise PoseExtractionError("v4b decode passes reported different FPS values")
    try:
        with mp.solutions.pose.Pose(
            static_image_mode=recovery.static_image_mode,
            model_complexity=recovery.model_complexity,
            smooth_landmarks=recovery.smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=recovery.min_detection_confidence,
            min_tracking_confidence=recovery.min_tracking_confidence,
        ) as heavy_detector:
            for frame_index in range(decoded_frames):
                ok, frame = heavy_capture.read()
                if not ok:
                    raise PoseExtractionError(
                        "v4b heavy VIDEO decode diverged from pass0 at decoded frame "
                        f"{frame_index}"
                    )
                heavy_candidates.append(
                    _process_pose_frame(heavy_detector, cv2, frame)
                )
    finally:
        heavy_capture.release()
    heavy_candidates.extend([None] * (expected_frames - decoded_frames))
    if len(heavy_candidates) != expected_frames:
        raise RuntimeError("v4b heavy timeline does not equal the official segment length")

    if sha256_file(heavy_asset) != recovery.heavy_model_asset_sha256:
        raise RuntimeError("heavy pose asset changed during extraction")
    heavy_track = select_global_dominant_track(
        heavy_candidates,
        center_weight=recovery.association_center_weight,
        log_scale_weight=recovery.association_log_scale_weight,
    )
    combined = [
        pass0_candidate if pass0_candidate is not None else heavy_candidate
        for pass0_candidate, heavy_candidate in zip(
            pass0_track,
            heavy_track,
            strict=True,
        )
    ]
    final_track = select_global_dominant_track(
        combined,
        center_weight=recovery.association_center_weight,
        log_scale_weight=recovery.association_log_scale_weight,
    )
    shared_errors = [
        float(np.max(np.abs(final_track[index][:, :3] - candidate[:, :3])))
        for index, candidate in enumerate(pass0_track)
        if candidate is not None and final_track[index] is not None
    ]
    pass0_valid = sum(candidate is not None for candidate in pass0_track)
    heavy_valid = sum(candidate is not None for candidate in heavy_track[:decoded_frames])
    heavy_overlap = sum(
        pass0_candidate is not None and heavy_candidate is not None
        for pass0_candidate, heavy_candidate in zip(
            pass0_track[:decoded_frames],
            heavy_track[:decoded_frames],
            strict=True,
        )
    )
    fill_candidates = sum(
        pass0_candidate is None and heavy_candidate is not None
        for pass0_candidate, heavy_candidate in zip(
            pass0_track[:decoded_frames],
            heavy_track[:decoded_frames],
            strict=True,
        )
    )
    final_valid = sum(candidate is not None for candidate in final_track)
    shared_max_error = max(shared_errors, default=0.0)
    preserved = len(shared_errors) == pass0_valid and shared_max_error <= 1e-6
    if not preserved or final_valid < pass0_valid:
        raise RuntimeError("v4b recovery altered or removed a pass0 observation")
    if final_valid != pass0_valid + fill_candidates:
        raise RuntimeError("v4b recovery lost a heavy fill candidate")

    raw = assemble_pose_sequence(final_track, video_id=identifier, fps=fps)
    valid_indices = np.flatnonzero(raw.valid_mask)
    observed_span = (
        int(valid_indices[-1] - valid_indices[0] + 1) if valid_indices.size else 0
    )
    selected_source_frames = raw.num_frames
    processed = preprocess_native_pose_sequence(raw)
    if processed.num_frames != expected_frames or processed.fps != raw.fps:
        raise RuntimeError("v4b preprocessing changed the native official-segment timeline")
    audit = PoseRecoveryAudit(
        source_frames=raw.num_frames,
        expected_segment_frames=expected_frames,
        decoded_segment_frames=decoded_frames,
        padded_tail_frames=expected_frames - decoded_frames,
        pass0_valid_frames=pass0_valid,
        heavy_full_frame_attempted=decoded_frames,
        heavy_full_frame_detected=heavy_valid,
        roi_retry_eligible=0,
        roi_retry_attempted=0,
        roi_retry_detected=0,
        recovered_valid_frames=final_valid - pass0_valid,
        final_valid_frames=final_valid,
        observed_span_frames=observed_span,
        final_longest_valid_run=_longest_true_run(raw.valid_mask),
        pass0_shared_coordinate_max_abs_error=shared_max_error,
        pass0_observations_preserved=preserved,
        pose_coordinate_interpolation=recovery.pose_coordinate_interpolation,
        temporal_resampling=recovery.temporal_resampling,
        heavy_model_id=recovery.heavy_model_id,
        heavy_model_asset_sha256=recovery.heavy_model_asset_sha256,
        pass0_valid_mask_sha256=_valid_mask_sha256(pass0_track),
        final_valid_mask_sha256=_valid_mask_sha256(final_track),
        recovery_mode=recovery.recovery_mode,
        heavy_video_frames_observed=decoded_frames,
        heavy_video_valid_frames=heavy_valid,
        heavy_video_pass0_overlap_valid_frames=heavy_overlap,
        heavy_video_fill_candidates=fill_candidates,
        heavy_video_valid_mask_sha256=_valid_mask_sha256(heavy_track),
    )
    return processed, decoded_frames, pass0_valid, selected_source_frames, audit


def extract_pose_sequence_tasks_recovery(
    video_path: str | Path,
    *,
    video_id: str | None = None,
    config: PoseExtractorConfig,
    clip_start_frame: int | None = None,
    clip_end_frame: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> tuple[PoseSequence, int, int, int, PoseRecoveryAudit]:
    """Run v4c pass0 plus Tasks Heavy VIDEO multipose global association."""

    if config.preprocessing_revision != TASKS_MULTIPOSE_RECOVERY_PREPROCESSING_REVISION:
        raise ValueError("Tasks recovery requires the v4c preprocessing revision")
    recovery = config.recovery
    if recovery is None:
        raise ValueError("v4c recovery settings are required")
    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"video does not exist: {source}")
    identifier = str(video_id or source.stem).strip()
    if not identifier:
        raise ValueError("video_id must be non-empty")
    if (clip_start_frame is None) != (clip_end_frame is None):
        raise ValueError("clip_start_frame and clip_end_frame must be supplied together")
    if clip_start_frame is None or clip_end_frame is None:
        raise ValueError("v4c recovery requires an official-segment input clip")
    if isinstance(clip_start_frame, bool | np.bool_) or isinstance(
        clip_end_frame, bool | np.bool_
    ):
        raise TypeError("clip frame indices must be integers, not bool")
    clip_start_frame = int(clip_start_frame)
    clip_end_frame = int(clip_end_frame)
    if clip_start_frame < 0 or clip_end_frame <= clip_start_frame:
        raise ValueError("clip must be a non-empty 0-based half-open interval")
    expected_frames = clip_end_frame - clip_start_frame

    cv2, mp = _load_pose_dependencies()
    heavy_asset = _verify_tasks_heavy_asset(recovery)

    def open_positioned_capture() -> tuple[Any, float]:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise PoseExtractionError(f"OpenCV could not open video: {source}")
        capture_fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(capture_fps) or capture_fps <= 0:
            capture.release()
            raise PoseExtractionError(f"video reports an invalid FPS: {source}")
        if clip_start_frame:
            seek_ok = bool(capture.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame))
            if not seek_ok:
                capture.release()
                raise PoseExtractionError(
                    f"OpenCV could not seek to clip start frame {clip_start_frame} "
                    f"for {source}"
                )
            positioned = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if np.isfinite(positioned) and abs(positioned - clip_start_frame) > 0.5:
                capture.release()
                raise PoseExtractionError(
                    f"OpenCV seek coverage mismatch for {source}: requested frame "
                    f"{clip_start_frame}, positioned at {positioned}"
                )
        return capture, capture_fps

    pass0_candidates: list[NDArray[np.float32] | None] = []
    decoded_frames = 0
    capture, fps = open_positioned_capture()
    try:
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=config.model_complexity,
            smooth_landmarks=config.smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=config.min_detection_confidence,
            min_tracking_confidence=config.min_tracking_confidence,
        ) as detector:
            while decoded_frames < expected_frames:
                ok, frame = capture.read()
                if not ok:
                    missing_tail = expected_frames - decoded_frames
                    if config.incomplete_clip_policy == "error":
                        raise PoseExtractionError(
                            f"incomplete official clip decode for {identifier!r}: "
                            f"range=[{clip_start_frame},{clip_end_frame}), "
                            f"expected={expected_frames}, decoded={decoded_frames}, "
                            f"missing_tail={missing_tail}"
                        )
                    pass0_candidates.extend([None] * missing_tail)
                    break
                pass0_candidates.append(_process_pose_frame(detector, cv2, frame))
                decoded_frames += 1
                if progress is not None:
                    progress(decoded_frames)
    finally:
        capture.release()
    if len(pass0_candidates) != expected_frames:
        raise RuntimeError("v4c pass0 timeline does not equal the official segment length")
    pass0_track = select_global_dominant_track(
        pass0_candidates,
        center_weight=recovery.association_center_weight,
        log_scale_weight=recovery.association_log_scale_weight,
    )

    tasks_candidates: list[NDArray[np.float32] | None] = []
    timestamps: list[int] = []
    heavy_capture, heavy_fps = open_positioned_capture()
    if not math.isclose(heavy_fps, fps, rel_tol=0.0, abs_tol=1e-6):
        heavy_capture.release()
        raise PoseExtractionError("v4c decode passes reported different FPS values")
    try:
        base_options = mp.tasks.BaseOptions(model_asset_path=str(heavy_asset))
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=4,
            min_pose_detection_confidence=recovery.min_detection_confidence,
            min_pose_presence_confidence=recovery.min_detection_confidence,
            min_tracking_confidence=recovery.min_tracking_confidence,
            output_segmentation_masks=False,
        )
        with mp.tasks.vision.PoseLandmarker.create_from_options(options) as landmarker:
            previous_timestamp = -1
            for frame_offset in range(decoded_frames):
                ok, frame = heavy_capture.read()
                if not ok:
                    raise PoseExtractionError(
                        "v4c Tasks VIDEO decode diverged from pass0 at decoded frame "
                        f"{frame_offset}"
                    )
                timestamp = _strict_video_timestamp_ms(
                    frame_offset,
                    fps,
                    previous_timestamp,
                )
                previous_timestamp = timestamp
                timestamps.append(timestamp)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(image, timestamp)
                tasks_candidates.append(
                    _canonicalize_tasks_candidate_order(
                        _tasks_candidates_from_result(result)
                    )
                )
    finally:
        heavy_capture.release()
    tasks_candidates.extend([None] * (expected_frames - decoded_frames))
    if len(tasks_candidates) != expected_frames:
        raise RuntimeError("v4c Tasks timeline does not equal the official segment length")
    if len(timestamps) != decoded_frames or any(
        right <= left for left, right in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise RuntimeError("v4c Tasks VIDEO timestamps are not strictly increasing")
    if sha256_file(heavy_asset) != recovery.heavy_model_asset_sha256:
        raise RuntimeError("Tasks heavy pose asset changed during extraction")

    anchored_candidates: list[ArrayLike | None] = []
    for pass0_candidate, task_candidates in zip(
        pass0_track,
        tasks_candidates,
        strict=True,
    ):
        # The pass0 singleton is the only node allowed on a pass0-valid frame.
        # Tasks multipose nodes are exposed only at pass0 misses, so the one
        # global Viterbi path is anchored to the immutable original subject.
        anchored_candidates.append(
            pass0_candidate if pass0_candidate is not None else task_candidates
        )
    final_track = select_global_dominant_track(
        anchored_candidates,
        center_weight=recovery.association_center_weight,
        log_scale_weight=recovery.association_log_scale_weight,
    )
    shared_errors = [
        float(np.max(np.abs(final_track[index][:, :3] - candidate[:, :3])))
        for index, candidate in enumerate(pass0_track)
        if candidate is not None and final_track[index] is not None
    ]
    pass0_valid = sum(candidate is not None for candidate in pass0_track)
    task_candidate_counts = [
        len(_candidate_array(candidate)) for candidate in tasks_candidates[:decoded_frames]
    ]
    task_valid = sum(count > 0 for count in task_candidate_counts)
    task_overlap = sum(
        pass0_candidate is not None and count > 0
        for pass0_candidate, count in zip(
            pass0_track[:decoded_frames],
            task_candidate_counts,
            strict=True,
        )
    )
    fill_candidates = sum(
        pass0_candidate is None and final_candidate is not None
        for pass0_candidate, final_candidate in zip(
            pass0_track[:decoded_frames],
            final_track[:decoded_frames],
            strict=True,
        )
    )
    final_valid = sum(candidate is not None for candidate in final_track)
    shared_max_error = max(shared_errors, default=0.0)
    preserved = len(shared_errors) == pass0_valid and shared_max_error <= 1e-6
    if not preserved or final_valid < pass0_valid:
        raise RuntimeError("v4c recovery altered or removed a pass0 observation")
    if final_valid != pass0_valid + fill_candidates:
        raise RuntimeError("v4c recovery count conservation failed")

    raw = assemble_pose_sequence(final_track, video_id=identifier, fps=fps)
    valid_indices = np.flatnonzero(raw.valid_mask)
    observed_span = (
        int(valid_indices[-1] - valid_indices[0] + 1) if valid_indices.size else 0
    )
    selected_source_frames = raw.num_frames
    processed = preprocess_native_pose_sequence(raw)
    if processed.num_frames != expected_frames or processed.fps != raw.fps:
        raise RuntimeError("v4c preprocessing changed the native official-segment timeline")
    task_presence_track = tuple(
        None if not len(_candidate_array(candidate)) else _candidate_array(candidate)[0]
        for candidate in tasks_candidates
    )
    timestamp_sha256 = hashlib.sha256(
        json.dumps(timestamps, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    audit = PoseRecoveryAudit(
        source_frames=raw.num_frames,
        expected_segment_frames=expected_frames,
        decoded_segment_frames=decoded_frames,
        padded_tail_frames=expected_frames - decoded_frames,
        pass0_valid_frames=pass0_valid,
        heavy_full_frame_attempted=decoded_frames,
        heavy_full_frame_detected=task_valid,
        roi_retry_eligible=0,
        roi_retry_attempted=0,
        roi_retry_detected=0,
        recovered_valid_frames=final_valid - pass0_valid,
        final_valid_frames=final_valid,
        observed_span_frames=observed_span,
        final_longest_valid_run=_longest_true_run(raw.valid_mask),
        pass0_shared_coordinate_max_abs_error=shared_max_error,
        pass0_observations_preserved=preserved,
        pose_coordinate_interpolation=recovery.pose_coordinate_interpolation,
        temporal_resampling=recovery.temporal_resampling,
        heavy_model_id=recovery.heavy_model_id,
        heavy_model_asset_sha256=recovery.heavy_model_asset_sha256,
        pass0_valid_mask_sha256=_valid_mask_sha256(pass0_track),
        final_valid_mask_sha256=_valid_mask_sha256(final_track),
        recovery_mode=recovery.recovery_mode,
        heavy_video_frames_observed=decoded_frames,
        heavy_video_valid_frames=task_valid,
        heavy_video_pass0_overlap_valid_frames=task_overlap,
        heavy_video_fill_candidates=fill_candidates,
        heavy_video_valid_mask_sha256=_valid_mask_sha256(task_presence_track),
        heavy_video_candidate_total=sum(task_candidate_counts),
        heavy_video_max_candidates_per_frame=max(task_candidate_counts, default=0),
        heavy_video_num_poses=4,
        heavy_video_timestamp_sha256=timestamp_sha256,
    )
    return processed, decoded_frames, pass0_valid, selected_source_frames, audit


def extract_pose_sequence(
    video_path: str | Path,
    *,
    video_id: str | None = None,
    config: PoseExtractorConfig | None = None,
    clip_start_frame: int | None = None,
    clip_end_frame: int | None = None,
    progress: Callable[[int], None] | None = None,
) -> tuple[PoseSequence, int, int, int]:
    """Extract and preprocess one video with MediaPipe Pose.

    Returns ``(sequence, decoded_frames, valid_frames_before_track_crop,
    selected_source_frames)``. ``clip_start_frame`` and ``clip_end_frame``
    use 0-based half-open semantics. A ranged extraction preserves that whole
    timeline and never applies detected-span or longest-track trimming.
    Dependency loading happens only after the input path has been validated.
    """

    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"video does not exist: {source}")
    settings = config or PoseExtractorConfig()
    identifier = str(video_id or source.stem).strip()
    if not identifier:
        raise ValueError("video_id must be non-empty")
    if (clip_start_frame is None) != (clip_end_frame is None):
        raise ValueError("clip_start_frame and clip_end_frame must be supplied together")
    if clip_start_frame is not None:
        if isinstance(clip_start_frame, bool | np.bool_) or isinstance(
            clip_end_frame, bool | np.bool_
        ):
            raise TypeError("clip frame indices must be integers, not bool")
        clip_start_frame = int(clip_start_frame)
        clip_end_frame = int(clip_end_frame)  # type: ignore[arg-type]
        if clip_start_frame < 0 or clip_end_frame <= clip_start_frame:
            raise ValueError("clip must be a non-empty 0-based half-open interval")
    if settings.preprocessing_revision == OFFICIAL_SEGMENT_PREPROCESSING_REVISION:
        if clip_start_frame is None:
            raise ValueError(
                "official-segment-full-timeline preprocessing requires an input clip"
            )
    elif clip_start_frame is not None:
        raise ValueError(
            "annotation-bound clips require official-segment-full-timeline preprocessing"
        )

    cv2, mp = _load_pose_dependencies()
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise PoseExtractionError(f"OpenCV could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise PoseExtractionError(f"video reports an invalid FPS: {source}")

    frame_candidates: list[NDArray[np.float32] | None] = []
    decoded_frames = 0
    try:
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=settings.model_complexity,
            smooth_landmarks=settings.smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        ) as detector:
            if clip_start_frame is not None and clip_start_frame:
                seek_ok = bool(capture.set(cv2.CAP_PROP_POS_FRAMES, clip_start_frame))
                if not seek_ok:
                    raise PoseExtractionError(
                        f"OpenCV could not seek to clip start frame {clip_start_frame} "
                        f"for {source}"
                    )
                positioned = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
                if np.isfinite(positioned) and abs(positioned - clip_start_frame) > 0.5:
                    raise PoseExtractionError(
                        f"OpenCV seek coverage mismatch for {source}: requested frame "
                        f"{clip_start_frame}, positioned at {positioned}"
                    )
            expected_frames = (
                None
                if clip_start_frame is None
                else int(clip_end_frame) - clip_start_frame
            )
            while expected_frames is None or decoded_frames < expected_frames:
                ok, frame = capture.read()
                if not ok:
                    if expected_frames is not None:
                        missing = expected_frames - decoded_frames
                        if settings.incomplete_clip_policy == "error":
                            raise PoseExtractionError(
                                f"incomplete official clip decode for {identifier!r}: "
                                f"range=[{clip_start_frame},{clip_end_frame}), "
                                f"expected={expected_frames}, decoded={decoded_frames}, "
                                f"missing_tail={missing}"
                            )
                        frame_candidates.extend([None] * missing)
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.process(rgb)
                frame_candidates.append(_landmarks_from_result(result))
                decoded_frames += 1
                if progress is not None:
                    progress(decoded_frames)
    finally:
        capture.release()

    raw = assemble_pose_sequence(frame_candidates, video_id=identifier, fps=fps)
    valid_frames = int(np.count_nonzero(raw.valid_mask))
    if clip_start_frame is None:
        selected, selected_source_frames = _select_preprocessing_span(
            raw,
            crop_to_detected_span=settings.crop_to_detected_span,
            preprocessing_revision=settings.preprocessing_revision,
        )
    else:
        selected = raw
        selected_source_frames = raw.num_frames
    processed = preprocess_pose_sequence(
        selected,
        target_frames=settings.target_frames,
    )
    return processed, decoded_frames, valid_frames, selected_source_frames


def extract_pose_to_cache(
    video_path: str | Path,
    *,
    video_id: str,
    cache_dir: str | Path,
    pose_fingerprint: str,
    expected_video_sha256: str | None = None,
    annotation_sha256: str | None = None,
    clip_start_frame: int | None = None,
    clip_end_frame: int | None = None,
    extractor_config: PoseExtractorConfig | None = None,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> tuple[PoseExtractionSummary, PoseCacheMetadata]:
    """Extract one video and atomically write its provenance-checked pose cache."""

    if overwrite and skip_existing:
        raise ValueError("overwrite and skip_existing are mutually exclusive")
    source = Path(video_path)
    video_digest = sha256_file(source)
    if expected_video_sha256 is not None and video_digest.lower() != expected_video_sha256.lower():
        raise ValueError(
            f"video SHA-256 mismatch for {video_id!r}: "
            f"expected {expected_video_sha256}, received {video_digest}"
        )
    settings = extractor_config or PoseExtractorConfig()
    annotation, clip_start, clip_end = normalize_clip_provenance(
        annotation_sha256=annotation_sha256,
        clip_start_frame=clip_start_frame,
        clip_end_frame=clip_end_frame,
    )
    if settings.preprocessing_revision in {
        RECOVERY_PREPROCESSING_REVISION,
        VIDEO_RECOVERY_PREPROCESSING_REVISION,
        TASKS_MULTIPOSE_RECOVERY_PREPROCESSING_REVISION,
    } and skip_existing:
        raise ValueError(
            "v4 recovery forbids --skip-existing because its paired source audit "
            "must be produced in the same immutable extraction attempt"
        )
    target = pose_cache_path(cache_dir, video_id)
    if target.exists():
        if not skip_existing:
            if not overwrite:
                raise FileExistsError(f"pose cache already exists: {target}")
        else:
            sequence, metadata = load_pose_cache(
                target,
                expected_video_sha256=video_digest,
                expected_pose_fingerprint=pose_fingerprint,
                expected_annotation_sha256=annotation,
                expected_clip_start_frame=clip_start,
                expected_clip_end_frame=clip_end,
                validate_clip_provenance=True,
            )
            if sequence.video_id != video_id:
                raise ValueError(
                    f"pose cache video_id mismatch: expected {video_id!r}, "
                    f"received {sequence.video_id!r}"
                )
            if metadata.pose_model != settings.model_id:
                raise ValueError(
                    "pose cache model ID mismatch: "
                    f"expected {settings.model_id!r}, received {metadata.pose_model!r}"
                )
            if sequence.num_frames != settings.target_frames:
                raise ValueError(
                    "pose cache frame count mismatch: "
                    f"expected {settings.target_frames}, received {sequence.num_frames}"
                )
            summary = PoseExtractionSummary(
                video_id=video_id,
                video_path=str(source.resolve()),
                cache_path=str(target.resolve()),
                video_sha256=video_digest,
                pose_fingerprint=metadata.pose_fingerprint,
                source_frames=None,
                source_valid_frames=None,
                selected_source_frames=None,
                cached_frames=sequence.num_frames,
                cached_valid_frames=int(np.count_nonzero(sequence.valid_mask)),
                fps=sequence.fps,
                pose_model=metadata.pose_model,
                annotation_sha256=metadata.annotation_sha256,
                clip_start_frame=metadata.clip_start_frame,
                clip_end_frame=metadata.clip_end_frame,
                expected_clip_frames=metadata.expected_clip_frames,
                decoded_clip_frames=metadata.decoded_clip_frames,
                padded_tail_frames=metadata.padded_tail_frames,
                incomplete_clip_policy=metadata.incomplete_clip_policy,
                skipped=True,
            )
            return summary, metadata

    recovery_audit: PoseRecoveryAudit | None = None
    if settings.preprocessing_revision in {
        RECOVERY_PREPROCESSING_REVISION,
        VIDEO_RECOVERY_PREPROCESSING_REVISION,
        TASKS_MULTIPOSE_RECOVERY_PREPROCESSING_REVISION,
    }:
        (
            sequence,
            source_frames,
            source_valid,
            selected_source_frames,
            recovery_audit,
        ) = extract_pose_sequence_recovery(
            source,
            video_id=video_id,
            config=settings,
            clip_start_frame=clip_start,
            clip_end_frame=clip_end,
        )
    else:
        sequence, source_frames, source_valid, selected_source_frames = (
            extract_pose_sequence(
                source,
                video_id=video_id,
                config=settings,
                clip_start_frame=clip_start,
                clip_end_frame=clip_end,
            )
        )
    expected_clip_frames = None if clip_start is None else clip_end - clip_start
    padded_tail_frames = (
        None if expected_clip_frames is None else expected_clip_frames - source_frames
    )
    metadata = write_pose_cache(
        target,
        sequence,
        video_sha256=video_digest,
        pose_fingerprint=pose_fingerprint,
        pose_model=settings.model_id,
        annotation_sha256=annotation,
        clip_start_frame=clip_start,
        clip_end_frame=clip_end,
        decoded_clip_frames=(None if clip_start is None else source_frames),
        expected_clip_frames=expected_clip_frames,
        padded_tail_frames=padded_tail_frames,
        incomplete_clip_policy=(
            None if clip_start is None else settings.incomplete_clip_policy
        ),
        overwrite=overwrite,
    )
    summary = PoseExtractionSummary(
        video_id=video_id,
        video_path=str(source.resolve()),
        cache_path=str(target.resolve()),
        video_sha256=video_digest,
        pose_fingerprint=metadata.pose_fingerprint,
        source_frames=source_frames,
        source_valid_frames=source_valid,
        selected_source_frames=selected_source_frames,
        cached_frames=sequence.num_frames,
        cached_valid_frames=int(np.count_nonzero(sequence.valid_mask)),
        fps=sequence.fps,
        pose_model=settings.model_id,
        annotation_sha256=annotation,
        clip_start_frame=clip_start,
        clip_end_frame=clip_end,
        expected_clip_frames=expected_clip_frames,
        decoded_clip_frames=(None if clip_start is None else source_frames),
        padded_tail_frames=padded_tail_frames,
        incomplete_clip_policy=(
            None if clip_start is None else settings.incomplete_clip_policy
        ),
        recovery_audit=recovery_audit,
    )
    return summary, metadata


def _unpack_extraction_row(
    row: Sequence[Any],
) -> tuple[str, str | Path, str | None, str | None, int | None, int | None]:
    """Accept legacy three-field and official-segment six-field rows."""

    values = tuple(row)
    if len(values) == 3:
        video_id, video_path, video_sha256 = values
        return str(video_id), video_path, video_sha256, None, None, None
    if len(values) == 6:
        video_id, video_path, video_sha256, annotation_sha256, clip_start, clip_end = values
        return (
            str(video_id),
            video_path,
            video_sha256,
            annotation_sha256,
            clip_start,
            clip_end,
        )
    raise ValueError("pose extraction rows must contain 3 legacy or 6 official fields")


def extract_many_to_cache(
    videos: Iterable[Sequence[Any]],
    *,
    cache_dir: str | Path,
    pose_fingerprint: str,
    extractor_config: PoseExtractorConfig | None = None,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> tuple[PoseExtractionSummary, ...]:
    """Extract legacy or annotation-bound official rows sequentially."""

    if overwrite and skip_existing:
        raise ValueError("overwrite and skip_existing are mutually exclusive")
    summaries: list[PoseExtractionSummary] = []
    for row in videos:
        (
            video_id,
            video_path,
            expected_digest,
            annotation_sha256,
            clip_start,
            clip_end,
        ) = _unpack_extraction_row(row)
        summary, _ = extract_pose_to_cache(
            video_path,
            video_id=video_id,
            cache_dir=cache_dir,
            pose_fingerprint=pose_fingerprint,
            expected_video_sha256=expected_digest,
            annotation_sha256=annotation_sha256,
            clip_start_frame=clip_start,
            clip_end_frame=clip_end,
            extractor_config=extractor_config,
            overwrite=overwrite,
            skip_existing=skip_existing,
        )
        summaries.append(summary)
    return tuple(summaries)


def extract_many_with_failures(
    videos: Iterable[Sequence[Any]],
    *,
    cache_dir: str | Path,
    pose_fingerprint: str,
    extractor_config: PoseExtractorConfig | None = None,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> tuple[tuple[PoseExtractionSummary, ...], tuple[PoseExtractionFailure, ...]]:
    """Extract every row and return a complete, non-silent failure ledger."""

    if overwrite and skip_existing:
        raise ValueError("overwrite and skip_existing are mutually exclusive")
    summaries: list[PoseExtractionSummary] = []
    failures: list[PoseExtractionFailure] = []
    for row in videos:
        (
            video_id,
            video_path,
            expected_digest,
            annotation_sha256,
            clip_start,
            clip_end,
        ) = _unpack_extraction_row(row)
        try:
            summary, _ = extract_pose_to_cache(
                video_path,
                video_id=video_id,
                cache_dir=cache_dir,
                pose_fingerprint=pose_fingerprint,
                expected_video_sha256=expected_digest,
                annotation_sha256=annotation_sha256,
                clip_start_frame=clip_start,
                clip_end_frame=clip_end,
                extractor_config=extractor_config,
                overwrite=overwrite,
                skip_existing=skip_existing,
            )
            summaries.append(summary)
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(
                PoseExtractionFailure(
                    video_id=str(video_id),
                    video_path=str(video_path),
                    error_type=type(exc).__name__,
                    message=str(exc),
                    annotation_sha256=annotation_sha256,
                    clip_start_frame=clip_start,
                    clip_end_frame=clip_end,
                )
            )
    return tuple(summaries), tuple(failures)
