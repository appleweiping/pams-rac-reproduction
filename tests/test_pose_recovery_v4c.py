from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pams.config import load_config
from pams.pose import (
    TASKS_MULTIPOSE_RECOVERY_PREPROCESSING_REVISION,
    PoseExtractorConfig,
    PoseRecoveryExtractorConfig,
    extract_pose_sequence_recovery,
)


def _candidate(center: float, *, extent: float = 0.2) -> np.ndarray:
    offsets = np.linspace(-extent / 2.0, extent / 2.0, 33, dtype=np.float32)
    candidate = np.empty((33, 4), dtype=np.float32)
    candidate[:, 0] = center + offsets
    candidate[:, 1] = 0.5 + offsets
    candidate[:, 2] = offsets * 0.1
    candidate[:, 3] = 1.0
    return candidate


def _legacy_result(candidate: np.ndarray | None) -> Any:
    if candidate is None:
        return SimpleNamespace(pose_landmarks=None)
    landmarks = [
        SimpleNamespace(x=row[0], y=row[1], z=row[2], visibility=row[3])
        for row in candidate
    ]
    return SimpleNamespace(pose_landmarks=SimpleNamespace(landmark=landmarks))


def _tasks_result(candidates: list[np.ndarray]) -> Any:
    poses = [
        [
            SimpleNamespace(x=row[0], y=row[1], z=row[2], visibility=row[3])
            for row in candidate
        ]
        for candidate in candidates
    ]
    return SimpleNamespace(pose_landmarks=poses)


def _run_tasks_fixture(
    tmp_path: Path,
    monkeypatch: Any,
    *,
    identifier: str,
    pass0: dict[int, np.ndarray | None],
    tasks: dict[int, list[np.ndarray]],
    clip_start: int,
    clip_end: int,
    source_end: int,
    fps: float = 25.0,
    mutate_asset_during_detect: bool = False,
) -> tuple[Any, Any, list[int], list[int], Any]:
    from pams import pose

    video = tmp_path / f"{identifier}.mp4"
    video.write_bytes(b"fixture")
    asset = tmp_path / f"{identifier}.task"
    asset.write_bytes(b"official-tasks-heavy")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    pass0_seen: list[int] = []
    task_timestamps: list[int] = []
    created_options: list[Any] = []

    class FakeCapture:
        def __init__(self) -> None:
            self.offset = 0

        def isOpened(self) -> bool:
            return True

        def set(self, prop: object, value: float) -> bool:
            assert prop == 1
            self.offset = int(value)
            return True

        def get(self, prop: object) -> float:
            return fps if prop == 5 else float(self.offset)

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.offset >= source_end:
                return False, None
            frame = np.full((8, 8, 3), self.offset, dtype=np.uint8)
            self.offset += 1
            return True, frame

        def release(self) -> None:
            return None

    class FakeLegacyDetector:
        def __enter__(self) -> FakeLegacyDetector:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def process(self, frame: np.ndarray) -> Any:
            source_index = int(frame[0, 0, 0])
            pass0_seen.append(source_index)
            return _legacy_result(pass0[source_index])

    class FakePoseFactory:
        def __call__(self, **kwargs: Any) -> FakeLegacyDetector:
            assert kwargs["static_image_mode"] is False
            assert kwargs["model_complexity"] == 1
            return FakeLegacyDetector()

    class FakeLandmarker:
        def __enter__(self) -> FakeLandmarker:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def detect_for_video(self, image: Any, timestamp: int) -> Any:
            source_index = int(image.data[0, 0, 0])
            task_timestamps.append(timestamp)
            if mutate_asset_during_detect and len(task_timestamps) == 1:
                asset.write_bytes(b"mutated-during-inference")
            return _tasks_result(tasks[source_index])

    class FakePoseLandmarker:
        @staticmethod
        def create_from_options(options: Any) -> FakeLandmarker:
            created_options.append(options)
            return FakeLandmarker()

    fake_tasks_vision = SimpleNamespace(
        RunningMode=SimpleNamespace(VIDEO="VIDEO"),
        PoseLandmarkerOptions=lambda **kwargs: SimpleNamespace(**kwargs),
        PoseLandmarker=FakePoseLandmarker,
    )
    fake_mp = SimpleNamespace(
        solutions=SimpleNamespace(pose=SimpleNamespace(Pose=FakePoseFactory())),
        tasks=SimpleNamespace(
            BaseOptions=lambda **kwargs: SimpleNamespace(**kwargs),
            vision=fake_tasks_vision,
        ),
        Image=lambda **kwargs: SimpleNamespace(**kwargs),
        ImageFormat=SimpleNamespace(SRGB="SRGB"),
    )
    fake_cv2 = SimpleNamespace(
        CAP_PROP_POS_FRAMES=1,
        CAP_PROP_FPS=5,
        COLOR_BGR2RGB=7,
        VideoCapture=lambda _: FakeCapture(),
        cvtColor=lambda frame, _: frame,
    )
    monkeypatch.setattr(pose, "_load_pose_dependencies", lambda: (fake_cv2, fake_mp))
    monkeypatch.setattr(pose, "preprocess_native_pose_sequence", lambda value: value)
    recovery = PoseRecoveryExtractorConfig(
        heavy_model_id="mediapipe-tasks-pose-landmarker-heavy-num-poses-4",
        heavy_model_asset_path=asset,
        heavy_model_asset_sha256=digest,
        static_image_mode=False,
        smooth_landmarks=True,
        roi_retry=False,
        recovery_mode="tasks-video-multipose4-fill-missing-v4c",
    )
    settings = PoseExtractorConfig(
        preprocessing_revision=TASKS_MULTIPOSE_RECOVERY_PREPROCESSING_REVISION,
        crop_to_detected_span=False,
        incomplete_clip_policy="pad_invalid_tail",
        recovery=recovery,
    )
    sequence, decoded, pass0_valid, selected, audit = extract_pose_sequence_recovery(
        video,
        video_id=identifier,
        config=settings,
        clip_start_frame=clip_start,
        clip_end_frame=clip_end,
    )
    assert selected == clip_end - clip_start
    assert pass0_valid == sum(value is not None for value in pass0.values())
    assert len(created_options) == 1
    assert created_options[0].num_poses == 4
    assert created_options[0].running_mode == "VIDEO"
    return sequence, audit, pass0_seen, task_timestamps, created_options[0]


def test_v4c_config_freezes_official_tasks_bundle_and_multipose_revision() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/experiments/pams_pose_recovery_v4c.yaml")

    assert config.pose.preprocessing_revision == (
        TASKS_MULTIPOSE_RECOVERY_PREPROCESSING_REVISION
    )
    assert config.pose.recovery is not None
    assert config.pose.recovery.heavy_model_asset_sha256 == (
        "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b"
    )
    assert config.pose.recovery.static_image_mode is False
    assert config.pose.recovery.roi_retry is False


def test_v4c_candidate_order_is_invariant_and_pass0_anchors_subject(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    person_a = _candidate(0.25)
    person_b = _candidate(0.78)
    pass0 = {0: person_a, 1: None, 2: None, 3: None, 4: person_a}
    forward = {index: [person_a, person_b] for index in range(5)}
    reversed_order = {index: [person_b, person_a] for index in range(5)}

    first, first_audit, _, _, _ = _run_tasks_fixture(
        tmp_path,
        monkeypatch,
        identifier="forward",
        pass0=pass0,
        tasks=forward,
        clip_start=0,
        clip_end=5,
        source_end=5,
    )
    second, second_audit, _, _, _ = _run_tasks_fixture(
        tmp_path,
        monkeypatch,
        identifier="reversed",
        pass0=pass0,
        tasks=reversed_order,
        clip_start=0,
        clip_end=5,
        source_end=5,
    )

    np.testing.assert_array_equal(first.xyz, second.xyz)
    np.testing.assert_array_equal(first.valid_mask, second.valid_mask)
    for frame_index in range(5):
        np.testing.assert_array_equal(first.xyz[frame_index], person_a[:, :3])
    assert first_audit.pass0_observations_preserved is True
    assert first_audit.heavy_video_candidate_total == 10
    assert first_audit.heavy_video_max_candidates_per_frame == 2
    assert first_audit.final_valid_mask_sha256 == second_audit.final_valid_mask_sha256


def test_v4c_without_pass0_selects_one_global_dominant_tasks_track(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    small = _candidate(0.25, extent=0.1)
    large = _candidate(0.70, extent=0.35)
    pass0 = {index: None for index in range(4)}
    tasks = {index: [small, large] for index in range(4)}

    sequence, audit, _, _, _ = _run_tasks_fixture(
        tmp_path,
        monkeypatch,
        identifier="no-pass0",
        pass0=pass0,
        tasks=tasks,
        clip_start=0,
        clip_end=4,
        source_end=4,
    )

    assert sequence.valid_mask.all()
    for frame_index in range(4):
        np.testing.assert_array_equal(sequence.xyz[frame_index], large[:, :3])
    assert audit.pass0_valid_frames == 0
    assert audit.recovered_valid_frames == 4
    assert audit.final_longest_valid_run == 4


def test_v4c_timestamps_are_strict_and_eof_tail_is_never_observed(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    person = _candidate(0.4)
    pass0 = {3: None, 4: None, 5: None}
    tasks = {3: [person], 4: [person], 5: [person]}

    sequence, audit, pass0_seen, timestamps, _ = _run_tasks_fixture(
        tmp_path,
        monkeypatch,
        identifier="timestamps-eof",
        pass0=pass0,
        tasks=tasks,
        clip_start=3,
        clip_end=8,
        source_end=6,
        fps=24.0,
    )

    assert pass0_seen == [3, 4, 5]
    assert timestamps == [0, 42, 83]
    assert all(
        right > left for left, right in zip(timestamps, timestamps[1:], strict=False)
    )
    assert audit.heavy_video_frames_observed == 3
    assert audit.padded_tail_frames == 2
    assert audit.heavy_video_num_poses == 4
    assert sequence.valid_mask.tolist() == [True, True, True, False, False]
    assert np.count_nonzero(sequence.xyz[3:]) == 0


def test_v4c_fails_closed_if_tasks_asset_changes_during_inference(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    person = _candidate(0.4)
    pass0 = {0: None, 1: None}
    tasks = {0: [person], 1: [person]}

    with pytest.raises(RuntimeError, match="asset changed during extraction"):
        _run_tasks_fixture(
            tmp_path,
            monkeypatch,
            identifier="mutated-asset",
            pass0=pass0,
            tasks=tasks,
            clip_start=0,
            clip_end=2,
            source_end=2,
            mutate_asset_during_detect=True,
        )
