from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pams.data import load_pose_cache
from pams.pose import (
    DETECTED_SPAN_PREPROCESSING_REVISION,
    LONGEST_TRACK_PREPROCESSING_REVISION,
    OFFICIAL_SEGMENT_PREPROCESSING_REVISION,
    PoseDependencyError,
    PoseExtractorConfig,
    assemble_pose_sequence,
    extract_many_with_failures,
    extract_pose_sequence,
    extract_pose_to_cache,
    preprocess_extracted_pose,
    select_dominant_pose,
    trim_to_detected_span,
    trim_to_longest_contiguous_track,
)
from pams.types import PoseSequence


def _person(scale: float, *, visibility: float = 1.0) -> np.ndarray:
    values = np.zeros((33, 4), dtype=np.float32)
    values[:, 0] = np.linspace(-scale, scale, 33)
    values[:, 1] = np.linspace(0.0, scale, 33)
    values[:, 2] = 0.1
    values[:, 3] = visibility
    return values


def test_select_dominant_pose_is_deterministic_and_returns_xyz() -> None:
    small = _person(0.2)
    large = _person(1.0)
    selected = select_dominant_pose(np.stack((small, large)))
    assert selected is not None
    assert selected.shape == (33, 3)
    np.testing.assert_array_equal(selected, large[:, :3])

    invalid = large.copy()
    invalid[0, 0] = np.nan
    np.testing.assert_array_equal(
        select_dominant_pose(np.stack((invalid, small))),
        small[:, :3],
    )
    assert select_dominant_pose(None) is None


def test_assemble_sequence_uses_exact_zeros_and_mask() -> None:
    frames = [_person(1.0), None, _person(0.5)]
    sequence = assemble_pose_sequence(frames, video_id="clip", fps=25.0)
    np.testing.assert_array_equal(sequence.valid_mask, [True, False, True])
    assert np.all(sequence.xyz[1] == 0.0)
    assert sequence.xyz.dtype == np.float32


def test_detected_span_preserves_internal_missing_frames_and_preprocesses_256() -> None:
    xyz = np.arange(7 * 33 * 3, dtype=np.float32).reshape(7, 33, 3)
    sequence = PoseSequence(
        "track",
        30.0,
        xyz,
        np.asarray([False, True, True, False, True, True, False]),
    )
    trimmed = trim_to_detected_span(sequence)
    assert trimmed.num_frames == 5
    np.testing.assert_array_equal(trimmed.xyz, sequence.xyz[1:6])
    np.testing.assert_array_equal(trimmed.valid_mask, [True, True, False, True, True])

    output = preprocess_extracted_pose(
        sequence,
        preprocessing_revision=DETECTED_SPAN_PREPROCESSING_REVISION,
    )
    assert output.xyz.shape == (256, 33, 3)
    assert output.valid_mask.any()
    assert not output.valid_mask.all()
    assert np.all(output.xyz[~output.valid_mask] == 0.0)
    assert output.xyz.min() >= 0
    assert output.xyz.max() <= 1
    assert output.fps == pytest.approx(30.0 * 255 / 4)


def test_longest_contiguous_track_is_earliest_on_ties_and_resamples_all_valid() -> None:
    xyz = np.arange(10 * 33 * 3, dtype=np.float32).reshape(10, 33, 3)
    sequence = PoseSequence(
        "track",
        30.0,
        xyz,
        np.asarray(
            [False, True, True, True, False, True, True, True, False, True]
        ),
    )
    selected = trim_to_longest_contiguous_track(sequence)
    assert selected.num_frames == 3
    np.testing.assert_array_equal(selected.xyz, sequence.xyz[1:4])
    assert selected.valid_mask.all()

    output = preprocess_extracted_pose(
        sequence,
        preprocessing_revision=LONGEST_TRACK_PREPROCESSING_REVISION,
    )
    assert output.xyz.shape == (256, 33, 3)
    assert output.valid_mask.all()
    assert np.all(output.xyz >= 0.0)
    assert np.all(output.xyz <= 1.0)
    assert output.fps == pytest.approx(30.0 * 255 / 2)


def test_longest_track_revision_requires_crop_and_rejects_unknown_revision() -> None:
    with pytest.raises(ValueError, match="requires crop"):
        PoseExtractorConfig(
            preprocessing_revision=LONGEST_TRACK_PREPROCESSING_REVISION,
            crop_to_detected_span=False,
        )
    with pytest.raises(ValueError, match="unsupported pose preprocessing"):
        PoseExtractorConfig(preprocessing_revision="future")
    with pytest.raises(ValueError, match="requires crop_to_detected_span=false"):
        PoseExtractorConfig(
            preprocessing_revision=OFFICIAL_SEGMENT_PREPROCESSING_REVISION,
            crop_to_detected_span=True,
        )


def test_single_frame_longest_track_becomes_full_duration_invalid_cache() -> None:
    xyz = np.ones((9, 33, 3), dtype=np.float32)
    sequence = PoseSequence(
        "single-detection",
        30.0,
        xyz,
        np.asarray([False, False, True, False, False, False, False, False, False]),
    )
    output = preprocess_extracted_pose(
        sequence,
        preprocessing_revision=LONGEST_TRACK_PREPROCESSING_REVISION,
    )
    assert output.num_frames == 256
    assert not output.valid_mask.any()
    assert np.all(output.xyz == 0.0)
    assert output.fps == pytest.approx(30.0 * 255 / 8)


def test_preprocess_retains_video_without_any_pose_as_invalid_cache() -> None:
    sequence = PoseSequence(
        "empty",
        30.0,
        np.zeros((3, 33, 3), dtype=np.float32),
        np.zeros(3, dtype=bool),
    )
    output = preprocess_extracted_pose(sequence)
    assert output.num_frames == 256
    assert not output.valid_mask.any()
    assert np.all(output.xyz == 0.0)


def test_optional_dependency_error_names_install_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    from pams import pose

    original_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"cv2", "mediapipe"}:
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(PoseDependencyError, match=r"\.\[pose\]"):
        pose._load_pose_dependencies()


class _FakeCapture:
    def __init__(self, frames: list[int]) -> None:
        self.frames = frames
        self.position = 0

    def isOpened(self) -> bool:
        return True

    def get(self, field: int) -> float:
        if field == _FakeCV2.CAP_PROP_FPS:
            return 30.0
        if field == _FakeCV2.CAP_PROP_POS_FRAMES:
            return float(self.position)
        return 0.0

    def set(self, field: int, value: float) -> bool:
        if field != _FakeCV2.CAP_PROP_POS_FRAMES:
            return False
        self.position = int(value)
        return True

    def read(self) -> tuple[bool, int | None]:
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def release(self) -> None:
        return None


class _FakeCV2:
    CAP_PROP_POS_FRAMES = 1
    CAP_PROP_FPS = 5
    COLOR_BGR2RGB = 4

    def __init__(self, frames: list[int]) -> None:
        self.frames = frames

    def VideoCapture(self, _path: str) -> _FakeCapture:  # noqa: N802
        return _FakeCapture(self.frames)

    @staticmethod
    def cvtColor(frame: int, _conversion: int) -> int:  # noqa: N802
        return frame


class _FakeDetector:
    def __enter__(self) -> _FakeDetector:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    @staticmethod
    def process(frame: int) -> int:
        return frame


def _fake_pose_dependencies(frames: list[int]) -> tuple[_FakeCV2, SimpleNamespace]:
    detector = _FakeDetector()
    mediapipe = SimpleNamespace(
        solutions=SimpleNamespace(
            pose=SimpleNamespace(Pose=lambda **_kwargs: detector),
        )
    )
    return _FakeCV2(frames), mediapipe


def test_official_clip_decodes_only_range_and_preserves_missing_timeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import pose

    video = tmp_path / "clip.avi"
    video.write_bytes(b"fixture")
    monkeypatch.setattr(pose, "_load_pose_dependencies", lambda: _fake_pose_dependencies(list(range(8))))
    monkeypatch.setattr(
        pose,
        "_landmarks_from_result",
        lambda result: None if result == 3 else _person(1.0),
    )
    config = PoseExtractorConfig(
        target_frames=3,
        preprocessing_revision=OFFICIAL_SEGMENT_PREPROCESSING_REVISION,
        crop_to_detected_span=False,
    )

    sequence, decoded, valid, selected = extract_pose_sequence(
        video,
        config=config,
        clip_start_frame=2,
        clip_end_frame=5,
    )

    assert (decoded, valid, selected) == (3, 2, 3)
    np.testing.assert_array_equal(sequence.valid_mask, [True, False, True])


def test_official_clip_shortfall_is_failure_by_default_and_opt_in_padding_is_audited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import pose

    video = tmp_path / "short.avi"
    video.write_bytes(b"fixture")
    monkeypatch.setattr(pose, "_load_pose_dependencies", lambda: _fake_pose_dependencies(list(range(5))))
    monkeypatch.setattr(pose, "_landmarks_from_result", lambda _result: _person(1.0))
    strict = PoseExtractorConfig(
        target_frames=4,
        preprocessing_revision=OFFICIAL_SEGMENT_PREPROCESSING_REVISION,
        crop_to_detected_span=False,
    )

    summaries, failures = extract_many_with_failures(
        (("short", video, None, "a" * 64, 2, 6),),
        cache_dir=tmp_path / "strict-cache",
        pose_fingerprint="b" * 64,
        extractor_config=strict,
    )

    assert summaries == ()
    assert len(failures) == 1
    assert "expected=4, decoded=3, missing_tail=1" in failures[0].message
    assert (failures[0].clip_start_frame, failures[0].clip_end_frame) == (2, 6)

    padded = PoseExtractorConfig(
        target_frames=4,
        preprocessing_revision=OFFICIAL_SEGMENT_PREPROCESSING_REVISION,
        crop_to_detected_span=False,
        incomplete_clip_policy="pad_invalid_tail",
    )
    summary, metadata = extract_pose_to_cache(
        video,
        video_id="short",
        cache_dir=tmp_path / "padded-cache",
        pose_fingerprint="c" * 64,
        annotation_sha256="a" * 64,
        clip_start_frame=2,
        clip_end_frame=6,
        extractor_config=padded,
    )

    assert summary.decoded_clip_frames == 3
    assert summary.expected_clip_frames == 4
    assert summary.padded_tail_frames == 1
    assert metadata.schema_version == 3
    assert metadata.incomplete_clip_policy == "pad_invalid_tail"
    cached, _ = load_pose_cache(summary.cache_path)
    np.testing.assert_array_equal(cached.valid_mask, [True, True, True, False])


def test_extract_to_cache_verifies_video_and_config_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pams import pose

    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"not-a-real-video")
    sequence = PoseSequence(
        "fixture",
        30.0,
        np.ones((256, 33, 3), dtype=np.float32),
        np.ones(256, dtype=bool),
    )

    def fake_extract(
        *args: object, **kwargs: object
    ) -> tuple[PoseSequence, int, int, int]:
        return sequence, 300, 280, 250

    monkeypatch.setattr(pose, "extract_pose_sequence", fake_extract)
    summary, metadata = extract_pose_to_cache(
        video,
        video_id="fixture",
        cache_dir=tmp_path / "cache",
        pose_fingerprint="a" * 64,
    )
    loaded, cached_metadata = load_pose_cache(summary.cache_path)
    assert loaded.video_id == "fixture"
    assert metadata == cached_metadata
    assert summary.source_frames == 300
    assert summary.selected_source_frames == 250
    assert len(summary.video_sha256) == 64

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        extract_pose_to_cache(
            video,
            video_id="fixture",
            cache_dir=tmp_path / "other",
            pose_fingerprint="a" * 64,
            expected_video_sha256="b" * 64,
        )


def test_batch_extraction_returns_non_silent_failure_ledger(tmp_path: Path) -> None:
    summaries, failures = extract_many_with_failures(
        (
            ("missing-a", tmp_path / "a.mp4", None),
            ("missing-b", tmp_path / "b.mp4", None),
        ),
        cache_dir=tmp_path / "cache",
        pose_fingerprint="a" * 64,
    )
    assert summaries == ()
    assert [failure.video_id for failure in failures] == [
        "missing-a",
        "missing-b",
    ]
    assert all(failure.error_type == "FileNotFoundError" for failure in failures)


def test_skip_existing_requires_exact_video_and_pose_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import pose

    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"stable-video")
    sequence = PoseSequence(
        "fixture",
        30.0,
        np.ones((256, 33, 3), dtype=np.float32),
        np.ones(256, dtype=bool),
    )
    extraction_calls = 0

    def fake_extract(
        *args: object, **kwargs: object
    ) -> tuple[PoseSequence, int, int, int]:
        nonlocal extraction_calls
        extraction_calls += 1
        return sequence, 300, 280, 250

    monkeypatch.setattr(pose, "extract_pose_sequence", fake_extract)
    first, _ = extract_pose_to_cache(
        video,
        video_id="fixture",
        cache_dir=tmp_path / "cache",
        pose_fingerprint="a" * 64,
    )
    assert not first.skipped
    assert extraction_calls == 1

    summaries, failures = extract_many_with_failures(
        (("fixture", video, first.video_sha256),),
        cache_dir=tmp_path / "cache",
        pose_fingerprint="a" * 64,
        skip_existing=True,
    )
    assert failures == ()
    assert len(summaries) == 1
    assert summaries[0].skipped
    assert summaries[0].source_frames is None
    assert extraction_calls == 1

    summaries, failures = extract_many_with_failures(
        (("fixture", video, first.video_sha256),),
        cache_dir=tmp_path / "cache",
        pose_fingerprint="b" * 64,
        skip_existing=True,
    )
    assert summaries == ()
    assert len(failures) == 1
    assert "fingerprint mismatch" in failures[0].message
    assert extraction_calls == 1

    video.write_bytes(b"changed-video")
    summaries, failures = extract_many_with_failures(
        (("fixture", video, None),),
        cache_dir=tmp_path / "cache",
        pose_fingerprint="a" * 64,
        skip_existing=True,
    )
    assert summaries == ()
    assert len(failures) == 1
    assert "video SHA-256 mismatch" in failures[0].message
    assert extraction_calls == 1


def test_skip_existing_and_overwrite_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        extract_many_with_failures(
            (),
            cache_dir=tmp_path,
            pose_fingerprint="a" * 64,
            overwrite=True,
            skip_existing=True,
        )
