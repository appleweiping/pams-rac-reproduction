from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pams.config import load_config
from pams.pose import (
    VIDEO_RECOVERY_PREPROCESSING_REVISION,
    PoseExtractorConfig,
    PoseRecoveryExtractorConfig,
    extract_pose_sequence_recovery,
)


def _candidate(center: float, *, z: float = 0.0) -> np.ndarray:
    offsets = np.linspace(-0.1, 0.1, 33, dtype=np.float32)
    candidate = np.empty((33, 4), dtype=np.float32)
    candidate[:, 0] = center + offsets
    candidate[:, 1] = 0.5 + offsets * 0.5
    candidate[:, 2] = z + offsets * 0.1
    candidate[:, 3] = 1.0
    return candidate


def _result(candidate: np.ndarray | None) -> Any:
    if candidate is None:
        return SimpleNamespace(pose_landmarks=None)
    landmarks = [
        SimpleNamespace(x=row[0], y=row[1], z=row[2], visibility=row[3])
        for row in candidate
    ]
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks))


def _recovery(asset: Path, digest: str) -> PoseRecoveryExtractorConfig:
    return PoseRecoveryExtractorConfig(
        heavy_model_id="mediapipe-pose-heavy-video-0.10.14",
        heavy_model_asset_path=asset,
        heavy_model_asset_sha256=digest,
        static_image_mode=False,
        smooth_landmarks=True,
        roi_retry=False,
        recovery_mode="full-timeline-video-fill-missing-v4b",
    )


def _asset(tmp_path: Path) -> tuple[Path, Path, str]:
    package = tmp_path / "mediapipe"
    module = package / "__init__.py"
    module.parent.mkdir()
    module.write_text("", encoding="utf-8")
    asset = package / "modules/pose_landmark/pose_landmark_heavy.tflite"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"verified-heavy-video")
    return module, asset, hashlib.sha256(asset.read_bytes()).hexdigest()


def test_v4b_config_is_distinct_and_preserves_frozen_v4a_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    v4a = load_config(root / "configs/experiments/pams_pose_recovery_v4a.yaml")
    v4b = load_config(root / "configs/experiments/pams_pose_recovery_v4b.yaml")

    assert v4a.fingerprint == "c4b2dfae47581750281a43cf8eae92e4ceab91d113af601a4155ad3e2dd2f99a"
    assert v4a.pose_fingerprint == (
        "817013890533cd19e6c791969c35c3699d56d0e6656768ce64199bc30ceebe9c"
    )
    assert v4b.pose.preprocessing_revision == VIDEO_RECOVERY_PREPROCESSING_REVISION
    assert v4b.pose.recovery is not None
    assert v4b.pose.recovery.static_image_mode is False
    assert v4b.pose.recovery.smooth_landmarks is True
    assert v4b.pose.recovery.full_frame_retry is True
    assert v4b.pose.recovery.roi_retry is False
    assert v4b.pose_fingerprint != v4a.pose_fingerprint


def test_v4b_rejects_static_or_roi_heavy_retry() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = load_config(
        root / "configs/experiments/pams_pose_recovery_v4b.yaml"
    ).model_dump(mode="json")
    payload["pose"]["recovery"]["static_image_mode"] = True
    with pytest.raises(ValueError, match="full-timeline VIDEO"):
        type(load_config(root / "configs/experiments/pams_pose_recovery_v4b.yaml"))(
            **payload
        )


def test_v4b_heavy_video_observes_full_timeline_but_only_fills_pass0_misses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import pose

    video = tmp_path / "source.mp4"
    video.write_bytes(b"fixture")
    module, asset, digest = _asset(tmp_path)
    pass0 = {
        3: _candidate(0.25, z=0.1),
        4: None,
        5: _candidate(0.35, z=0.2),
        6: None,
        7: None,
    }
    heavy = {
        3: _candidate(0.75, z=0.8),
        4: _candidate(0.30, z=0.3),
        5: _candidate(0.80, z=0.9),
        6: None,
        7: _candidate(0.40, z=0.4),
    }
    seen: dict[int, list[int]] = {1: [], 2: []}
    captures: list[Any] = []

    class FakeCapture:
        def __init__(self) -> None:
            self.offset = 0
            captures.append(self)

        def isOpened(self) -> bool:
            return True

        def set(self, prop: object, value: float) -> bool:
            assert prop == 1
            self.offset = int(value)
            return True

        def get(self, prop: object) -> float:
            return 25.0 if prop == 5 else float(self.offset)

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.offset >= 10:
                return False, None
            frame = np.full((8, 8, 3), self.offset, dtype=np.uint8)
            self.offset += 1
            return True, frame

        def release(self) -> None:
            return None

    class FakeDetector:
        def __init__(self, complexity: int) -> None:
            self.complexity = complexity

        def __enter__(self) -> FakeDetector:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def process(self, frame: np.ndarray) -> Any:
            source_index = int(frame[0, 0, 0])
            seen[self.complexity].append(source_index)
            values = pass0 if self.complexity == 1 else heavy
            return _result(values[source_index])

    class FakePoseFactory:
        def __call__(self, **kwargs: Any) -> FakeDetector:
            assert kwargs["static_image_mode"] is False
            if kwargs["model_complexity"] == 2:
                assert kwargs["smooth_landmarks"] is True
            return FakeDetector(int(kwargs["model_complexity"]))

    fake_cv2 = SimpleNamespace(
        CAP_PROP_POS_FRAMES=1,
        CAP_PROP_FPS=5,
        COLOR_BGR2RGB=7,
        VideoCapture=lambda _: FakeCapture(),
        cvtColor=lambda frame, _: frame,
    )
    fake_mp = SimpleNamespace(
        __file__=str(module),
        solutions=SimpleNamespace(pose=SimpleNamespace(Pose=FakePoseFactory())),
    )
    monkeypatch.setattr(pose, "_load_pose_dependencies", lambda: (fake_cv2, fake_mp))
    monkeypatch.setattr(pose, "preprocess_native_pose_sequence", lambda value: value)
    settings = PoseExtractorConfig(
        preprocessing_revision=VIDEO_RECOVERY_PREPROCESSING_REVISION,
        crop_to_detected_span=False,
        incomplete_clip_policy="pad_invalid_tail",
        recovery=_recovery(asset, digest),
    )

    sequence, decoded, pass0_valid, selected, audit = extract_pose_sequence_recovery(
        video,
        video_id="nonzero-v4b",
        config=settings,
        clip_start_frame=3,
        clip_end_frame=8,
    )

    assert len(captures) == 2
    assert seen[1] == [3, 4, 5, 6, 7]
    assert seen[2] == [3, 4, 5, 6, 7]
    assert decoded == selected == 5
    assert pass0_valid == 2
    assert sequence.valid_mask.tolist() == [True, True, True, False, True]
    np.testing.assert_array_equal(sequence.xyz[0], pass0[3][:, :3])
    np.testing.assert_array_equal(sequence.xyz[2], pass0[5][:, :3])
    np.testing.assert_array_equal(sequence.xyz[1], heavy[4][:, :3])
    np.testing.assert_array_equal(sequence.xyz[4], heavy[7][:, :3])
    assert audit.recovery_mode == "full-timeline-video-fill-missing-v4b"
    assert audit.heavy_video_frames_observed == 5
    assert audit.heavy_video_valid_frames == 4
    assert audit.heavy_video_pass0_overlap_valid_frames == 2
    assert audit.heavy_video_fill_candidates == 2
    assert audit.recovered_valid_frames == 2
    assert audit.final_valid_frames == 4
    assert audit.pass0_observations_preserved is True
    assert audit.pass0_shared_coordinate_max_abs_error == 0.0
    assert audit.roi_retry_attempted == 0


def test_v4b_never_sends_padded_tail_to_either_detector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import pose

    video = tmp_path / "short-source.mp4"
    video.write_bytes(b"fixture")
    module, asset, digest = _asset(tmp_path)
    seen: dict[int, list[int]] = {1: [], 2: []}

    class FakeCapture:
        def __init__(self) -> None:
            self.offset = 0

        def isOpened(self) -> bool:
            return True

        def set(self, _: object, value: float) -> bool:
            self.offset = int(value)
            return True

        def get(self, prop: object) -> float:
            return 24.0 if prop == 5 else float(self.offset)

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.offset >= 6:
                return False, None
            frame = np.full((8, 8, 3), self.offset, dtype=np.uint8)
            self.offset += 1
            return True, frame

        def release(self) -> None:
            return None

    class FakeDetector:
        def __init__(self, complexity: int) -> None:
            self.complexity = complexity

        def __enter__(self) -> FakeDetector:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def process(self, frame: np.ndarray) -> Any:
            source_index = int(frame[0, 0, 0])
            seen[self.complexity].append(source_index)
            return _result(_candidate(0.4))

    class FakePoseFactory:
        def __call__(self, **kwargs: Any) -> FakeDetector:
            return FakeDetector(int(kwargs["model_complexity"]))

    fake_cv2 = SimpleNamespace(
        CAP_PROP_POS_FRAMES=1,
        CAP_PROP_FPS=5,
        COLOR_BGR2RGB=7,
        VideoCapture=lambda _: FakeCapture(),
        cvtColor=lambda frame, _: frame,
    )
    fake_mp = SimpleNamespace(
        __file__=str(module),
        solutions=SimpleNamespace(pose=SimpleNamespace(Pose=FakePoseFactory())),
    )
    monkeypatch.setattr(pose, "_load_pose_dependencies", lambda: (fake_cv2, fake_mp))
    settings = PoseExtractorConfig(
        preprocessing_revision=VIDEO_RECOVERY_PREPROCESSING_REVISION,
        crop_to_detected_span=False,
        incomplete_clip_policy="pad_invalid_tail",
        recovery=_recovery(asset, digest),
    )

    sequence, decoded, pass0_valid, selected, audit = extract_pose_sequence_recovery(
        video,
        video_id="padded-v4b",
        config=settings,
        clip_start_frame=3,
        clip_end_frame=8,
    )

    assert seen == {1: [3, 4, 5], 2: [3, 4, 5]}
    assert decoded == pass0_valid == 3
    assert selected == sequence.num_frames == 5
    assert audit.heavy_video_frames_observed == 3
    assert audit.padded_tail_frames == 2
    assert sequence.valid_mask.tolist() == [True, True, True, False, False]
    assert np.count_nonzero(sequence.xyz[3:]) == 0


def test_v4b_pilot_projection_uses_pass0_only_nonpilot_lower_bound() -> None:
    from scripts.server.audit_pose_recovery_v4b_pilot import (
        _minimum_longest_run,
        _project_observed_v4b_row,
        _project_pass0_only_row,
    )

    assert _minimum_longest_run(0, 10) == 0
    assert _minimum_longest_run(5, 10) == 1
    assert _minimum_longest_run(8, 10) == 3
    assert _minimum_longest_run(10, 10) == 10
    original = {
        "video_id_sha256": "a" * 64,
        "reference_cached_valid_frames": 4,
        "final_valid_frames": 9,
        "source_coverage": 0.9,
        "longest_run_fraction": 0.9,
    }
    nonpilot = _project_pass0_only_row(
        original,
        {"source_frames": 10, "pass0_valid_frames": 5},
    )
    assert nonpilot["final_valid_frames"] == 5
    assert nonpilot["recovered_valid_frames"] == 0
    assert nonpilot["source_coverage"] == 0.5
    assert nonpilot["longest_run_fraction"] == 0.1
    pilot = _project_observed_v4b_row(
        nonpilot,
        {
            "source_frames": 10,
            "pass0_valid_frames": 5,
            "recovered_valid_frames": 3,
            "final_valid_frames": 8,
            "final_longest_valid_run": 6,
            "observed_span_frames": 9,
            "pass0_observations_preserved": True,
        },
    )
    assert pilot["final_valid_frames"] == 8
    assert pilot["recovered_valid_frames"] == 3
    assert pilot["source_coverage"] == 0.8
    assert pilot["longest_run_fraction"] == 0.6
    assert pilot["reference_cached_valid_frames"] == 4
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/server/audit_pose_recovery_v4b_pilot.py"
    ).read_text(encoding="utf-8")
    assert '"nonpilot_behavior": "pass0_only_count_with_tight_arrangement_free' in source
    assert '"nonpilot_behavior": "retain_v4a' not in source
