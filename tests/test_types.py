from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pams.types import CountResult, PoseSequence


def test_pose_sequence_canonicalizes_invalid_frames_and_is_immutable() -> None:
    xyz = np.ones((4, 33, 3), dtype=np.float64)
    xyz[2] = np.nan
    sequence = PoseSequence(
        video_id=" video-1 ",
        fps=30,
        xyz=xyz,
        valid_mask=np.asarray([True, True, False, True]),
    )

    assert sequence.video_id == "video-1"
    assert sequence.xyz.dtype == np.float32
    assert sequence.xyz.shape == (4, 33, 3)
    assert np.all(sequence.xyz[2] == 0)
    assert sequence.flattened().shape == (4, 99)
    assert sequence.valid_fraction == pytest.approx(0.75)
    with pytest.raises(ValueError):
        sequence.xyz.setflags(write=True)
    with pytest.raises(ValueError):
        sequence.xyz[0, 0, 0] = 7
    with pytest.raises(FrozenInstanceError):
        sequence.fps = 25  # type: ignore[misc]


@pytest.mark.parametrize(
    ("xyz", "mask", "message"),
    [
        (np.zeros((3, 17, 3)), np.ones(3), "shape"),
        (np.zeros((3, 33, 3)), np.ones(2), "valid_mask"),
        (
            np.full((3, 33, 3), np.nan),
            np.ones(3, dtype=bool),
            "finite",
        ),
    ],
)
def test_pose_sequence_rejects_invalid_inputs(
    xyz: np.ndarray, mask: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PoseSequence(video_id="bad", fps=30, xyz=xyz, valid_mask=mask)


def test_count_result_validates_and_serializes() -> None:
    result = CountResult(
        count=7,
        period_frames=12.5,
        expert_counts=(7, 8, 7),
        confidence=0.75,
        period_stream=np.linspace(0, 1, 16),
    )
    assert result.expert_counts == (7, 8, 7)
    assert result.period_stream.dtype == np.float32
    assert result.to_dict(include_stream=False)["count"] == 7
    with pytest.raises(ValueError):
        result.period_stream.setflags(write=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"count": -1},
        {"period_frames": 0},
        {"expert_counts": (1, 2)},
        {"confidence": 1.1},
        {"period_stream": np.asarray([np.nan])},
    ],
)
def test_count_result_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "count": 1,
        "period_frames": 8.0,
        "expert_counts": (1, 1, 1),
        "confidence": 0.5,
        "period_stream": np.ones(8),
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        CountResult(**values)  # type: ignore[arg-type]
