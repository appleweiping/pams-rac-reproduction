from __future__ import annotations

import builtins
from pathlib import Path

import numpy as np
import pytest

from pams.data import load_pose_cache
from pams.pose import (
    PoseDependencyError,
    assemble_pose_sequence,
    extract_many_with_failures,
    extract_pose_to_cache,
    preprocess_extracted_pose,
    select_dominant_pose,
    trim_to_longest_valid_span,
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


def test_longest_valid_track_prefers_earliest_and_preprocesses_256() -> None:
    xyz = np.arange(7 * 33 * 3, dtype=np.float32).reshape(7, 33, 3)
    sequence = PoseSequence(
        "track",
        30.0,
        xyz,
        np.asarray([False, True, True, False, True, True, False]),
    )
    trimmed = trim_to_longest_valid_span(sequence)
    assert trimmed.num_frames == 2
    np.testing.assert_array_equal(trimmed.xyz, xyz[1:3])

    output = preprocess_extracted_pose(sequence)
    assert output.xyz.shape == (256, 33, 3)
    assert output.valid_mask.all()
    assert output.xyz.min() >= 0
    assert output.xyz.max() <= 1
    assert output.fps == pytest.approx(30.0 * 255)


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

    def guarded_import(name: str, *args: object, **kwargs: object) -> object:
        if name in {"cv2", "mediapipe"}:
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(PoseDependencyError, match=r"\.\[pose\]"):
        pose._load_pose_dependencies()


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

    def fake_extract(*args: object, **kwargs: object) -> tuple[PoseSequence, int, int]:
        return sequence, 300, 280

    monkeypatch.setattr(pose, "extract_pose_sequence", fake_extract)
    summary, metadata = extract_pose_to_cache(
        video,
        video_id="fixture",
        cache_dir=tmp_path / "cache",
        config_sha256="a" * 64,
    )
    loaded, cached_metadata = load_pose_cache(summary.cache_path)
    assert loaded.video_id == "fixture"
    assert metadata == cached_metadata
    assert summary.source_frames == 300
    assert len(summary.video_sha256) == 64

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        extract_pose_to_cache(
            video,
            video_id="fixture",
            cache_dir=tmp_path / "other",
            config_sha256="a" * 64,
            expected_video_sha256="b" * 64,
        )


def test_batch_extraction_returns_non_silent_failure_ledger(tmp_path: Path) -> None:
    summaries, failures = extract_many_with_failures(
        (
            ("missing-a", tmp_path / "a.mp4", None),
            ("missing-b", tmp_path / "b.mp4", None),
        ),
        cache_dir=tmp_path / "cache",
        config_sha256="a" * 64,
    )
    assert summaries == ()
    assert [failure.video_id for failure in failures] == [
        "missing-a",
        "missing-b",
    ]
    assert all(failure.error_type == "FileNotFoundError" for failure in failures)
