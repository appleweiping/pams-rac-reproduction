from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from pams.config import PAMSConfig, load_config
from pams.period import estimate_period_batch_direct_fft
from pams.pose import (
    RECOVERY_PREPROCESSING_REVISION,
    PoseDependencyError,
    PoseExtractorConfig,
    PoseRecoveryExtractorConfig,
    _short_gap_rois,
    _verify_heavy_asset,
    extract_pose_sequence_recovery,
    select_global_dominant_track,
)
from pams.training import collate_pose_sequences
from pams.types import PoseSequence


def _candidate(
    center_x: float,
    *,
    center_y: float = 0.5,
    scale: float = 0.2,
) -> np.ndarray:
    values = np.zeros((33, 4), dtype=np.float32)
    values[:, 0] = center_x + np.linspace(-scale, scale, 33)
    values[:, 1] = center_y + np.linspace(-scale, scale, 33)
    values[:, 2] = np.linspace(-0.05, 0.05, 33)
    values[:, 3] = 1.0
    return values


def _result(candidate: np.ndarray | None) -> Any:
    if candidate is None:
        return SimpleNamespace(pose_landmarks=None)
    landmarks = [
        SimpleNamespace(x=row[0], y=row[1], z=row[2], visibility=row[3])
        for row in candidate
    ]
    return SimpleNamespace(
        pose_landmarks=SimpleNamespace(landmark=landmarks),
    )


def _recovery(asset: Path, digest: str) -> PoseRecoveryExtractorConfig:
    return PoseRecoveryExtractorConfig(
        heavy_model_id="mediapipe-pose-heavy-0.10.14",
        heavy_model_asset_path=asset,
        heavy_model_asset_sha256=digest,
    )


def test_v4a_config_freezes_asset_retry_roi_association_and_no_pose_interpolation() -> None:
    root = Path(__file__).resolve().parents[1]
    historical = load_config(root / "configs/experiments/pams_longest_contiguous_track_v8.yaml")
    candidate = load_config(root / "configs/experiments/pams_pose_recovery_v4a.yaml")
    assert historical.pose_fingerprint == (
        "3dd0388320095796f42aa82d904073b6478b073ed351394ef8d03c30798a0116"
    )
    assert candidate.pose_fingerprint != historical.pose_fingerprint
    recovery = candidate.pose.recovery
    assert recovery is not None
    assert recovery.heavy_model_asset_sha256 == (
        "59e42d71bcd44cbdbabc419f0ff76686595fd265419566bd4009ef703ea8e1fe"
    )
    assert recovery.model_complexity == 2
    assert recovery.temporal_resampling == "none_native_timeline"
    assert recovery.static_image_mode is True
    assert recovery.full_frame_retry is True
    assert recovery.roi_retry is True
    assert recovery.maximum_gap_frames == 8
    assert recovery.maximum_gap_seconds == 0.25
    assert recovery.pose_coordinate_interpolation is False
    identity = candidate.pose_canonical_dict()["pose"]["recovery"]
    assert identity == recovery.model_dump(mode="json")


def test_v4a_config_rejects_non_complexity1_pass0_and_non_v4_recovery() -> None:
    root = Path(__file__).resolve().parents[1]
    candidate = load_config(root / "configs/experiments/pams_pose_recovery_v4a.yaml")
    payload = candidate.model_dump(mode="json")
    payload["pose"]["model_complexity"] = 2
    with pytest.raises(ValueError, match="pass0 must use model_complexity=1"):
        PAMSConfig.model_validate(payload)
    payload = candidate.model_dump(mode="json")
    payload["pose"]["preprocessing_revision"] = (
        "detected-span-minmax-zero-span-invalid-v2"
    )
    payload["pose"]["incomplete_clip_policy"] = "error"
    with pytest.raises(ValueError, match="accepted only by the v4a"):
        PAMSConfig.model_validate(payload)


def test_global_track_is_deterministic_and_uses_temporal_association() -> None:
    person_a = [_candidate(0.25 + 0.01 * index, scale=0.22) for index in range(4)]
    person_b = [_candidate(0.78 - 0.01 * index, scale=0.20) for index in range(4)]
    frames = [np.stack((a, b)) for a, b in zip(person_a, person_b, strict=True)]
    first = select_global_dominant_track(frames)
    second = select_global_dominant_track(frames)
    for index, selected in enumerate(first):
        assert selected is not None
        np.testing.assert_array_equal(selected, person_a[index])
        np.testing.assert_array_equal(selected, second[index])


def test_short_gap_interpolates_only_rois_with_bilateral_pass0_anchors() -> None:
    anchor_a = _candidate(0.3)
    anchor_b = _candidate(0.7)
    track = (anchor_a, None, None, None, anchor_b, None, None)
    rois = _short_gap_rois(
        track,
        fps=20.0,
        maximum_gap_frames=8,
        maximum_gap_seconds=0.25,
        margin_fraction=0.2,
        minimum_side_fraction=0.08,
    )
    assert set(rois) == {1, 2, 3}
    assert rois[1][0] < rois[2][0] < rois[3][0]
    assert 5 not in rois and 6 not in rois

    too_long = (anchor_a, None, None, None, None, None, None, anchor_b)
    assert not _short_gap_rois(
        too_long,
        fps=20.0,
        maximum_gap_frames=8,
        maximum_gap_seconds=0.25,
        margin_fraction=0.2,
        minimum_side_fraction=0.08,
    )


def test_heavy_asset_must_match_sha_and_installed_resource_path(tmp_path: Path) -> None:
    package = tmp_path / "mediapipe"
    module = package / "__init__.py"
    module.parent.mkdir()
    module.write_text("", encoding="utf-8")
    asset = package / "modules/pose_landmark/pose_landmark_heavy.tflite"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"verified-heavy")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    mp = SimpleNamespace(__file__=str(module))
    assert _verify_heavy_asset(_recovery(asset, digest), mp) == asset.resolve()

    with pytest.raises(PoseDependencyError, match="SHA-256 mismatch"):
        _verify_heavy_asset(_recovery(asset, "0" * 64), mp)
    elsewhere = tmp_path / "elsewhere.tflite"
    elsewhere.write_bytes(asset.read_bytes())
    with pytest.raises(PoseDependencyError, match="bind-mounted directly"):
        _verify_heavy_asset(_recovery(elsewhere, digest), mp)


def test_recovery_locks_pass0_and_uses_roi_only_after_full_frame_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import pose

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fixture")
    package = tmp_path / "mediapipe"
    module = package / "__init__.py"
    module.parent.mkdir()
    module.write_text("", encoding="utf-8")
    asset = package / "modules/pose_landmark/pose_landmark_heavy.tflite"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"verified-heavy")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    left = _candidate(0.25)
    right = _candidate(0.65)
    recovered = _candidate(0.5)

    class FakeCapture:
        def __init__(self) -> None:
            self.frames = [np.full((100, 100, 3), index, dtype=np.uint8) for index in range(5)]
            self.offset = 0

        def isOpened(self) -> bool:
            return True

        def get(self, _: object) -> float:
            return 20.0

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.offset == len(self.frames):
                return False, None
            frame = self.frames[self.offset]
            self.offset += 1
            return True, frame

        def release(self) -> None:
            return None

    class FakeDetector:
        def __init__(self, *, static: bool) -> None:
            self.static = static
            self.offset = 0

        def __enter__(self) -> FakeDetector:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def process(self, frame: np.ndarray) -> Any:
            if not self.static:
                values = (left, None, None, None, right)
                result = _result(values[self.offset])
                self.offset += 1
                return result
            if frame.shape[:2] == (100, 100):
                return _result(None)
            return _result(recovered)

    class FakePoseFactory:
        def __call__(self, **kwargs: Any) -> FakeDetector:
            return FakeDetector(static=bool(kwargs["static_image_mode"]))

    fake_cv2 = SimpleNamespace(
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
    recovery = _recovery(asset, digest)
    settings = PoseExtractorConfig(
        preprocessing_revision=RECOVERY_PREPROCESSING_REVISION,
        crop_to_detected_span=False,
        incomplete_clip_policy="pad_invalid_tail",
        recovery=recovery,
    )
    sequence, source_frames, pass0_valid, selected_frames, audit = (
        extract_pose_sequence_recovery(
            video,
            video_id="clip",
            config=settings,
            clip_start_frame=0,
            clip_end_frame=5,
        )
    )
    assert source_frames == 5
    assert pass0_valid == 2
    assert selected_frames == 5
    assert audit.heavy_full_frame_attempted == 3
    assert audit.heavy_full_frame_detected == 0
    assert audit.roi_retry_eligible == 3
    assert audit.roi_retry_attempted == 3
    assert audit.roi_retry_detected == 3
    assert audit.recovered_valid_frames == 3
    assert audit.expected_segment_frames == 5
    assert audit.decoded_segment_frames == 5
    assert audit.padded_tail_frames == 0
    assert audit.pass0_observations_preserved is True
    assert audit.pass0_shared_coordinate_max_abs_error == 0.0
    assert audit.pose_coordinate_interpolation is False
    assert audit.temporal_resampling == "none_native_timeline"
    assert sequence.num_frames == 5
    assert sequence.fps == 20.0
    assert sequence.valid_mask.all()


def test_v4a_preserves_more_than_1000_native_frames_and_invalid_decode_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import pose

    video = tmp_path / "long-clip.mp4"
    video.write_bytes(b"fixture")
    package = tmp_path / "mediapipe"
    module = package / "__init__.py"
    module.parent.mkdir()
    module.write_text("", encoding="utf-8")
    asset = package / "modules/pose_landmark/pose_landmark_heavy.tflite"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"verified-heavy")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    candidate = _candidate(0.5)
    decoded_frames = 1001
    expected_frames = 1005

    class FakeCapture:
        def __init__(self) -> None:
            self.offset = 0

        def isOpened(self) -> bool:
            return True

        def get(self, _: object) -> float:
            return 24.0

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.offset == decoded_frames:
                return False, None
            frame = np.full((8, 8, 3), self.offset % 255, dtype=np.uint8)
            self.offset += 1
            return True, frame

        def release(self) -> None:
            return None

    class FakeDetector:
        def __init__(self, *, static: bool) -> None:
            self.static = static

        def __enter__(self) -> FakeDetector:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def process(self, _: np.ndarray) -> Any:
            if self.static:
                raise AssertionError("decode-tail padding must not invoke heavy retry")
            return _result(candidate)

    class FakePoseFactory:
        def __call__(self, **kwargs: Any) -> FakeDetector:
            return FakeDetector(static=bool(kwargs["static_image_mode"]))

    fake_cv2 = SimpleNamespace(
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
        preprocessing_revision=RECOVERY_PREPROCESSING_REVISION,
        crop_to_detected_span=False,
        incomplete_clip_policy="pad_invalid_tail",
        recovery=_recovery(asset, digest),
    )

    sequence, decoded, pass0_valid, selected, audit = extract_pose_sequence_recovery(
        video,
        video_id="long-clip",
        config=settings,
        clip_start_frame=0,
        clip_end_frame=expected_frames,
    )

    assert sequence.num_frames == expected_frames
    assert sequence.fps == 24.0
    assert decoded == decoded_frames
    assert pass0_valid == decoded_frames
    assert selected == expected_frames
    assert sequence.valid_mask[:decoded_frames].all()
    assert not sequence.valid_mask[decoded_frames:].any()
    assert np.count_nonzero(sequence.xyz[decoded_frames:]) == 0
    assert audit.padded_tail_frames == expected_frames - decoded_frames
    assert audit.temporal_resampling == "none_native_timeline"


def test_v4a_nonzero_clip_start_seeks_before_decoding_exact_half_open_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import pose

    video = tmp_path / "source-video.mp4"
    video.write_bytes(b"fixture")
    package = tmp_path / "mediapipe"
    module = package / "__init__.py"
    module.parent.mkdir()
    module.write_text("", encoding="utf-8")
    asset = package / "modules/pose_landmark/pose_landmark_heavy.tflite"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"verified-heavy")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    seen_source_indices: list[int] = []

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
            return 25.0 if prop == 5 else float(self.offset)

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.offset == 8:
                return False, None
            frame = np.full((8, 8, 3), self.offset, dtype=np.uint8)
            self.offset += 1
            return True, frame

        def release(self) -> None:
            return None

    class FakeDetector:
        def __init__(self, *, static: bool) -> None:
            self.static = static

        def __enter__(self) -> FakeDetector:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def process(self, frame: np.ndarray) -> Any:
            if self.static:
                raise AssertionError("all pass0 clip frames are valid")
            source_index = int(frame[0, 0, 0])
            seen_source_indices.append(source_index)
            return _result(_candidate(0.3 + source_index * 0.01))

    class FakePoseFactory:
        def __call__(self, **kwargs: Any) -> FakeDetector:
            return FakeDetector(static=bool(kwargs["static_image_mode"]))

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
        preprocessing_revision=RECOVERY_PREPROCESSING_REVISION,
        crop_to_detected_span=False,
        incomplete_clip_policy="pad_invalid_tail",
        recovery=_recovery(asset, digest),
    )

    sequence, decoded, pass0_valid, selected, audit = extract_pose_sequence_recovery(
        video,
        video_id="nonzero-start",
        config=settings,
        clip_start_frame=3,
        clip_end_frame=7,
    )

    assert seen_source_indices == [3, 4, 5, 6]
    assert sequence.num_frames == 4
    assert decoded == pass0_valid == selected == 4
    assert audit.expected_segment_frames == 4
    assert audit.padded_tail_frames == 0


def test_native_frame_indices_and_short_period_survive_batch_padding() -> None:
    native_frames = 1200
    longer_frames = 1237
    native_period = 6

    def sequence(identifier: str, frames: int) -> PoseSequence:
        time = np.arange(frames, dtype=np.float32)
        xyz = np.zeros((frames, 33, 3), dtype=np.float32)
        xyz[:, 0, 0] = np.sin(2.0 * np.pi * time / native_period)
        xyz[:, 1, 1] = time
        return PoseSequence(
            video_id=identifier,
            fps=30.0,
            xyz=xyz,
            valid_mask=np.ones(frames, dtype=np.bool_),
        )

    native = sequence("native", native_frames)
    longer = sequence("longer", longer_frames)
    batch = collate_pose_sequences((native, longer))

    assert batch.lengths.tolist() == [native_frames, longer_frames]
    assert batch.fps.tolist() == [30.0, 30.0]
    torch.testing.assert_close(
        batch.poses[0, :native_frames],
        torch.from_numpy(np.asarray(native.xyz).copy()),
    )
    assert not batch.valid_mask[0, native_frames:].any()
    assert torch.count_nonzero(batch.poses[0, native_frames:]) == 0
    assert batch.poses[0, 997, 1, 1].item() == 997.0

    unpadded_period, _ = estimate_period_batch_direct_fft(
        torch.from_numpy(np.asarray(native.xyz[:, 0, 0]).copy()),
        minimum=4,
        maximum=128,
    )
    padded_period, _ = estimate_period_batch_direct_fft(
        batch.poses[0, :, 0, 0],
        minimum=4,
        maximum=128,
        valid_mask=batch.valid_mask[0],
    )
    assert unpadded_period.item() == pytest.approx(native_period, rel=0.01)
    assert padded_period.item() == pytest.approx(unpadded_period.item())
    old_resampled_period = native_period * (256 - 1) / (native_frames - 1)
    assert old_resampled_period < 4


def test_audit_source_has_no_dev_test_or_target_cli_inputs(tmp_path: Path) -> None:
    from scripts.server.audit_pose_recovery_v4a import (
        RecoveryAuditError,
        _load_label_free_json,
    )

    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts/server/audit_pose_recovery_v4a.py").read_text(
        encoding="utf-8"
    )
    argument_lines = [line for line in source.splitlines() if "parser.add_argument" in line]
    joined = "\n".join(argument_lines)
    assert "--train-input" in joined
    assert "--train-commitment" in joined
    assert "--reference-cache-dir" in joined
    assert "--v4-cache-dir" in joined
    assert "--dev" not in joined
    assert "--test" not in joined
    assert "--target" not in joined

    path = tmp_path / "forbidden.json"
    path.write_text(json.dumps({"schema_version": 1, "target": 3}), encoding="utf-8")
    with pytest.raises(RecoveryAuditError, match="privileged"):
        _load_label_free_json(path, role="fixture")
