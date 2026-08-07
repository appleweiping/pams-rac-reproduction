from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from pams.config import load_config
from pams.keypoint_recovery import (
    COCO17_TO_MEDIAPIPE33,
    MEDIAPIPE33_TO_COCO17,
    KeypointRCNNRuntime,
    keypointrcnn_candidates_from_output,
    recover_v4a_locked_video,
)
from pams.types import PoseSequence


def _output(
    *,
    scores: list[float],
    labels: list[int] | None = None,
    confident: list[int] | None = None,
) -> dict[str, np.ndarray]:
    count = len(scores)
    keypoints = np.zeros((count, 17, 3), dtype=np.float32)
    logits = np.full((count, 17), -2.0, dtype=np.float32)
    for person in range(count):
        keypoints[person, :, 0] = np.linspace(10 + person, 90 + person, 17)
        keypoints[person, :, 1] = np.linspace(20 + person, 180 + person, 17)
        logits[person, : (confident or [17] * count)[person]] = 3.0
    return {
        "labels": np.asarray(labels or [1] * count, dtype=np.int64),
        "scores": np.asarray(scores, dtype=np.float32),
        "keypoints": keypoints,
        "keypoints_scores": logits,
    }


def test_v4d_config_is_frozen_and_preserves_old_fingerprints() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/experiments/pams_pose_recovery_v4d.yaml")
    settings = config.pose.keypoint_recovery

    assert settings is not None
    assert settings.model_asset_sha256 == (
        "fc266e953d2b302cdcbb9ae66f71f6b0d4649928bf02dc573961e361e4918926"
    )
    assert settings.box_score_threshold == 0.2
    assert settings.keypoint_logit_threshold == 2.0
    assert settings.minimum_confident_keypoints == 4
    assert settings.maximum_candidates_per_frame == 4
    assert settings.expected_parameter_count == 59137258
    assert settings.detector_observes_all_decoded_frames is True
    assert settings.fill_missing_only is True
    assert config.pose_fingerprint == (
        "4cc1f905cfb3484dccc1efc480e3fa59e4a106ffbe20528a85201a3fbd85b5d8"
    )
    assert load_config(
        root / "configs/experiments/pams_pose_recovery_v4c.yaml"
    ).pose_fingerprint == ("a2a875dd5b80e82fd2e01be21574c1c3ce96e720f31c0343996ae09e9b31f491")


def test_candidate_filter_mapping_and_output_permutation_are_deterministic() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = load_config(
        root / "configs/experiments/pams_pose_recovery_v4d.yaml"
    ).pose.keypoint_recovery
    assert settings is not None
    output = _output(
        scores=[0.91, 0.19, 0.82, 0.75, 0.70, 0.65],
        labels=[1, 1, 2, 1, 1, 1],
        confident=[17, 17, 17, 3, 4, 17],
    )
    first = keypointrcnn_candidates_from_output(
        output, frame_width=100, frame_height=200, settings=settings
    )
    permutation = np.asarray([5, 4, 3, 2, 1, 0])
    second = keypointrcnn_candidates_from_output(
        {key: value[permutation] for key, value in output.items()},
        frame_width=100,
        frame_height=200,
        settings=settings,
    )

    assert first is not None and second is not None
    assert first.shape == (3, 33, 4)
    assert np.array_equal(first, second)
    assert np.all(first[:, :, 2] == 0.0)
    assert np.array_equal(first[:, 1], first[:, 2])
    assert np.array_equal(first[:, 2], first[:, 3])
    assert np.array_equal(first[:, 9], first[:, 10])
    assert np.array_equal(first[:, 15], first[:, 17])
    assert np.array_equal(first[:, 27], first[:, 29])
    assert MEDIAPIPE33_TO_COCO17.shape == (33,)
    assert COCO17_TO_MEDIAPIPE33.shape == (17,)


def test_v4d_fills_only_decoded_base_misses_and_locks_v4a_bitwise(
    monkeypatch: Any, tmp_path: Path
) -> None:
    root = Path(__file__).resolve().parents[1]
    settings = load_config(
        root / "configs/experiments/pams_pose_recovery_v4d.yaml"
    ).pose.keypoint_recovery
    assert settings is not None
    frames = [np.full((8, 8, 3), value, dtype=np.uint8) for value in range(4)]

    class Capture:
        def __init__(self, _: str) -> None:
            self.index = 0

        def isOpened(self) -> bool:
            return True

        def get(self, prop: int) -> float:
            if prop == 5:
                return 24.0
            return float(self.index)

        def set(self, _: int, value: int) -> bool:
            self.index = int(value)
            return True

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.index >= len(frames):
                return False, None
            frame = frames[self.index]
            self.index += 1
            return True, frame

        def release(self) -> None:
            return None

    fake_cv2 = SimpleNamespace(
        VideoCapture=Capture,
        CAP_PROP_FPS=5,
        CAP_PROP_POS_FRAMES=1,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    candidate = np.zeros((1, 33, 4), dtype=np.float32)
    candidate[0, :, 0] = np.linspace(0.1, 0.9, 33)
    candidate[0, :, 1] = np.linspace(0.2, 0.8, 33)
    candidate[0, :, 3] = 0.9
    calls: list[int] = []

    def fake_infer(
        runtime: KeypointRCNNRuntime,
        batch: list[np.ndarray],
        *,
        settings: Any,
    ) -> tuple[np.ndarray | None, ...]:
        del runtime, settings
        calls.append(len(batch))
        return tuple(candidate.copy() for _ in batch)

    monkeypatch.setattr("pams.keypoint_recovery._infer_batch", fake_infer)
    asset = tmp_path / "weights.pth"
    asset.write_bytes(b"weights")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    runtime = KeypointRCNNRuntime(
        model=object(),
        torch=object(),
        device=object(),
        asset_path=asset,
        asset_sha256=digest,
        state_dict_keys=313,
        parameter_count=59137258,
        runtime_receipt={"fixture": True},
    )
    xyz = np.zeros((5, 33, 3), dtype=np.float32)
    xyz[0, :, 0] = np.linspace(0.0, 1.0, 33)
    xyz[0, :, 1] = 0.25
    xyz[3, :, 0] = np.linspace(1.0, 0.0, 33)
    xyz[3, :, 1] = 0.75
    mask = np.asarray([True, False, False, True, False])
    base = PoseSequence(video_id="fixture", fps=24.0, xyz=xyz, valid_mask=mask)
    base_audit = {
        "schema_version": 1,
        "source_frames": 5,
        "expected_segment_frames": 5,
        "decoded_segment_frames": 4,
        "padded_tail_frames": 1,
        "pass0_valid_frames": 1,
        "recovered_valid_frames": 1,
        "final_valid_frames": 2,
        "observed_span_frames": 4,
        "final_longest_valid_run": 1,
        "pass0_valid_mask_sha256": "0" * 64,
        "final_valid_mask_sha256": "1" * 64,
        "pass0_observations_preserved": True,
        "pass0_shared_coordinate_max_abs_error": 0.0,
        "pose_coordinate_interpolation": False,
        "temporal_resampling": "none_native_timeline",
    }
    video = tmp_path / "fixture.mp4"
    video.write_bytes(b"video")
    final, audit = recover_v4a_locked_video(
        video,
        video_id="fixture",
        clip_start_frame=0,
        clip_end_frame=5,
        base_sequence=base,
        base_decoded_frames=4,
        base_recovery_audit=base_audit,
        expected_video_sha256=hashlib.sha256(b"video").hexdigest(),
        runtime=runtime,
        settings=settings,
    )

    assert calls == [4]
    assert np.array_equal(final.xyz[mask], base.xyz[mask])
    assert final.valid_mask.tolist() == [True, True, True, True, False]
    assert audit["base_v4a_valid_frames"] == 2
    assert audit["keypointrcnn_frames_attempted"] == 4
    assert audit["keypointrcnn_missing_frames_eligible"] == 2
    assert audit["keypointrcnn_fill_candidates"] == 2
    assert audit["base_v4a_observations_preserved"] is True
    assert audit["base_v4a_coordinate_sha256"] == audit["final_base_v4a_coordinate_sha256"]
