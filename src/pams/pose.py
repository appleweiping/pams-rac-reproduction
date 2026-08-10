"""Optional MediaPipe pose extraction with auditable cache provenance.

The heavy video dependencies are imported only when extraction is requested.
Importing :mod:`pams.pose`, running the CLI help, and executing unit tests
therefore do not require MediaPipe or OpenCV.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pams.data import (
    PoseCacheMetadata,
    load_pose_cache,
    longest_valid_span,
    pose_cache_path,
    preprocess_pose_sequence,
    write_pose_cache,
)
from pams.types import PoseSequence

POSE_MODEL_ID = "mediapipe-pose-0.10.14"
DETECTED_SPAN_PREPROCESSING_REVISION = "detected-span-minmax-zero-span-invalid-v2"
LONGEST_TRACK_PREPROCESSING_REVISION = (
    "longest-contiguous-track-minmax-zero-span-invalid-v3"
)
SUPPORTED_PREPROCESSING_REVISIONS = frozenset(
    {
        DETECTED_SPAN_PREPROCESSING_REVISION,
        LONGEST_TRACK_PREPROCESSING_REVISION,
    }
)


class PoseDependencyError(RuntimeError):
    """Raised when the optional pose-extraction dependencies are unavailable."""


class PoseExtractionError(RuntimeError):
    """Raised when a video cannot yield a usable pose sequence."""


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
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
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
            "skipped": self.skipped,
        }


@dataclass(frozen=True, slots=True)
class PoseExtractionFailure:
    """One explicit decode/extraction failure for the shared failure ledger."""

    video_id: str
    video_path: str
    error_type: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "error_type": self.error_type,
            "message": self.message,
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
            "longest-contiguous-track preprocessing requires "
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


def extract_pose_sequence(
    video_path: str | Path,
    *,
    video_id: str | None = None,
    config: PoseExtractorConfig | None = None,
    progress: Callable[[int], None] | None = None,
) -> tuple[PoseSequence, int, int, int]:
    """Extract and preprocess one video with MediaPipe Pose.

    Returns ``(sequence, decoded_frames, valid_frames_before_track_crop,
    selected_source_frames)``.
    Dependency loading happens only after the input path has been validated.
    """

    source = Path(video_path)
    if not source.is_file():
        raise FileNotFoundError(f"video does not exist: {source}")
    settings = config or PoseExtractorConfig()
    identifier = str(video_id or source.stem).strip()
    if not identifier:
        raise ValueError("video_id must be non-empty")

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
    try:
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=settings.model_complexity,
            smooth_landmarks=settings.smooth_landmarks,
            enable_segmentation=False,
            min_detection_confidence=settings.min_detection_confidence,
            min_tracking_confidence=settings.min_tracking_confidence,
        ) as detector:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = detector.process(rgb)
                frame_candidates.append(_landmarks_from_result(result))
                if progress is not None:
                    progress(len(frame_candidates))
    finally:
        capture.release()

    raw = assemble_pose_sequence(frame_candidates, video_id=identifier, fps=fps)
    valid_frames = int(np.count_nonzero(raw.valid_mask))
    selected, selected_source_frames = _select_preprocessing_span(
        raw,
        crop_to_detected_span=settings.crop_to_detected_span,
        preprocessing_revision=settings.preprocessing_revision,
    )
    processed = preprocess_pose_sequence(
        selected,
        target_frames=settings.target_frames,
    )
    return processed, raw.num_frames, valid_frames, selected_source_frames


def extract_pose_to_cache(
    video_path: str | Path,
    *,
    video_id: str,
    cache_dir: str | Path,
    pose_fingerprint: str,
    expected_video_sha256: str | None = None,
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
                skipped=True,
            )
            return summary, metadata

    sequence, source_frames, source_valid, selected_source_frames = extract_pose_sequence(
        source,
        video_id=video_id,
        config=settings,
    )
    metadata = write_pose_cache(
        target,
        sequence,
        video_sha256=video_digest,
        pose_fingerprint=pose_fingerprint,
        pose_model=settings.model_id,
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
    )
    return summary, metadata


def extract_many_to_cache(
    videos: Iterable[tuple[str, str | Path, str | None]],
    *,
    cache_dir: str | Path,
    pose_fingerprint: str,
    extractor_config: PoseExtractorConfig | None = None,
    overwrite: bool = False,
    skip_existing: bool = False,
) -> tuple[PoseExtractionSummary, ...]:
    """Extract ``(video_id, path, expected_sha256)`` rows sequentially."""

    if overwrite and skip_existing:
        raise ValueError("overwrite and skip_existing are mutually exclusive")
    summaries: list[PoseExtractionSummary] = []
    for video_id, video_path, expected_digest in videos:
        summary, _ = extract_pose_to_cache(
            video_path,
            video_id=video_id,
            cache_dir=cache_dir,
            pose_fingerprint=pose_fingerprint,
            expected_video_sha256=expected_digest,
            extractor_config=extractor_config,
            overwrite=overwrite,
            skip_existing=skip_existing,
        )
        summaries.append(summary)
    return tuple(summaries)


def extract_many_with_failures(
    videos: Iterable[tuple[str, str | Path, str | None]],
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
    for video_id, video_path, expected_digest in videos:
        try:
            summary, _ = extract_pose_to_cache(
                video_path,
                video_id=video_id,
                cache_dir=cache_dir,
                pose_fingerprint=pose_fingerprint,
                expected_video_sha256=expected_digest,
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
                )
            )
    return tuple(summaries), tuple(failures)
