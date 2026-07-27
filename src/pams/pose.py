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
    longest_valid_span,
    pose_cache_path,
    preprocess_pose_sequence,
    write_pose_cache,
)
from pams.types import PoseSequence

POSE_MODEL_ID = "mediapipe-pose-0.10.14"


class PoseDependencyError(RuntimeError):
    """Raised when the optional pose-extraction dependencies are unavailable."""


class PoseExtractionError(RuntimeError):
    """Raised when a video cannot yield a usable pose sequence."""


@dataclass(frozen=True, slots=True)
class PoseExtractorConfig:
    """Frozen settings for the MediaPipe Pose video extractor."""

    target_frames: int = 256
    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    smooth_landmarks: bool = True
    crop_to_longest_valid_span: bool = True

    def __post_init__(self) -> None:
        if self.target_frames < 1:
            raise ValueError("target_frames must be positive")
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
    source_frames: int
    source_valid_frames: int
    cached_frames: int
    cached_valid_frames: int
    fps: float
    pose_model: str = POSE_MODEL_ID

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_path": self.video_path,
            "cache_path": self.cache_path,
            "video_sha256": self.video_sha256,
            "source_frames": self.source_frames,
            "source_valid_frames": self.source_valid_frames,
            "cached_frames": self.cached_frames,
            "cached_valid_frames": self.cached_valid_frames,
            "fps": self.fps,
            "pose_model": self.pose_model,
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
        import mediapipe as mp  # type: ignore[import-not-found]
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


def trim_to_longest_valid_span(sequence: PoseSequence) -> PoseSequence:
    """Keep the earliest longest contiguous valid subject track."""

    try:
        span = longest_valid_span(sequence.valid_mask)
    except ValueError as exc:
        raise PoseExtractionError(
            f"no valid pose was detected in video {sequence.video_id!r}"
        ) from exc
    return PoseSequence(
        video_id=sequence.video_id,
        fps=sequence.fps,
        xyz=sequence.xyz[span],
        valid_mask=sequence.valid_mask[span],
    )


def preprocess_extracted_pose(
    sequence: PoseSequence,
    *,
    target_frames: int = 256,
    crop_to_longest_valid_span: bool = True,
) -> PoseSequence:
    """Apply the frozen track selection, normalization, and uniform sampling."""

    # The protocol retains a video when pose extraction finds no subject.  An
    # all-zero, all-invalid cache lets every method share the same denominator
    # and makes the failure visible in metadata rather than silently dropping
    # the sample.
    if not np.any(sequence.valid_mask):
        return preprocess_pose_sequence(sequence, target_frames=target_frames)
    selected = trim_to_longest_valid_span(sequence) if crop_to_longest_valid_span else sequence
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
) -> tuple[PoseSequence, int, int]:
    """Extract and preprocess one video with MediaPipe Pose.

    Returns ``(sequence, decoded_frames, valid_frames_before_track_crop)``.
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
    processed = preprocess_extracted_pose(
        raw,
        target_frames=settings.target_frames,
        crop_to_longest_valid_span=settings.crop_to_longest_valid_span,
    )
    return processed, raw.num_frames, valid_frames


def extract_pose_to_cache(
    video_path: str | Path,
    *,
    video_id: str,
    cache_dir: str | Path,
    config_sha256: str,
    expected_video_sha256: str | None = None,
    extractor_config: PoseExtractorConfig | None = None,
    overwrite: bool = False,
) -> tuple[PoseExtractionSummary, PoseCacheMetadata]:
    """Extract one video and atomically write its provenance-checked pose cache."""

    source = Path(video_path)
    video_digest = sha256_file(source)
    if expected_video_sha256 is not None and video_digest.lower() != expected_video_sha256.lower():
        raise ValueError(
            f"video SHA-256 mismatch for {video_id!r}: "
            f"expected {expected_video_sha256}, received {video_digest}"
        )
    settings = extractor_config or PoseExtractorConfig()
    sequence, source_frames, source_valid = extract_pose_sequence(
        source,
        video_id=video_id,
        config=settings,
    )
    target = pose_cache_path(cache_dir, video_id)
    metadata = write_pose_cache(
        target,
        sequence,
        video_sha256=video_digest,
        config_sha256=config_sha256,
        pose_model=POSE_MODEL_ID,
        overwrite=overwrite,
    )
    summary = PoseExtractionSummary(
        video_id=video_id,
        video_path=str(source.resolve()),
        cache_path=str(target.resolve()),
        video_sha256=video_digest,
        source_frames=source_frames,
        source_valid_frames=source_valid,
        cached_frames=sequence.num_frames,
        cached_valid_frames=int(np.count_nonzero(sequence.valid_mask)),
        fps=sequence.fps,
    )
    return summary, metadata


def extract_many_to_cache(
    videos: Iterable[tuple[str, str | Path, str | None]],
    *,
    cache_dir: str | Path,
    config_sha256: str,
    extractor_config: PoseExtractorConfig | None = None,
    overwrite: bool = False,
) -> tuple[PoseExtractionSummary, ...]:
    """Extract ``(video_id, path, expected_sha256)`` rows sequentially."""

    summaries: list[PoseExtractionSummary] = []
    for video_id, video_path, expected_digest in videos:
        summary, _ = extract_pose_to_cache(
            video_path,
            video_id=video_id,
            cache_dir=cache_dir,
            config_sha256=config_sha256,
            expected_video_sha256=expected_digest,
            extractor_config=extractor_config,
            overwrite=overwrite,
        )
        summaries.append(summary)
    return tuple(summaries)


def extract_many_with_failures(
    videos: Iterable[tuple[str, str | Path, str | None]],
    *,
    cache_dir: str | Path,
    config_sha256: str,
    extractor_config: PoseExtractorConfig | None = None,
    overwrite: bool = False,
) -> tuple[tuple[PoseExtractionSummary, ...], tuple[PoseExtractionFailure, ...]]:
    """Extract every row and return a complete, non-silent failure ledger."""

    summaries: list[PoseExtractionSummary] = []
    failures: list[PoseExtractionFailure] = []
    for video_id, video_path, expected_digest in videos:
        try:
            summary, _ = extract_pose_to_cache(
                video_path,
                video_id=video_id,
                cache_dir=cache_dir,
                config_sha256=config_sha256,
                expected_video_sha256=expected_digest,
                extractor_config=extractor_config,
                overwrite=overwrite,
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
