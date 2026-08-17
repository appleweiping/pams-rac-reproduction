from __future__ import annotations

import numpy as np
import pytest

from pams.temporac.preprocess import (
    BONES,
    PreprocessError,
    build_geometry_features,
    collapse_duplicate_clocks,
    mask_to_half_open_runs,
    preprocess_identity,
)


def _track(sample_count: int = 20) -> np.ndarray:
    base = np.stack(
        (
            np.linspace(-0.8, 0.8, 17),
            np.linspace(0.2, 1.0, 17),
        ),
        axis=1,
    )
    # Give hips and shoulders an unambiguous nonzero scale.
    base[11] = (-0.2, 0.0)
    base[12] = (0.2, 0.0)
    base[5] = (-0.3, 0.8)
    base[6] = (0.3, 0.8)
    motion = np.zeros((sample_count, 17, 3), dtype=np.float32)
    for sample in range(sample_count):
        phase = 2.0 * np.pi * sample / 8.0
        motion[sample, :, :2] = base
        motion[sample, 7:11, 0] += np.float32(0.15 * np.sin(phase))
        motion[sample, 13:17, 1] += np.float32(0.12 * np.cos(phase))
        motion[sample, :, 2] = np.float32(0.9)
    return motion


def test_duplicate_collapse_uses_one_scale_and_weighted_binary64_mean() -> None:
    motion = _track(4)
    duplicated = np.insert(motion, 2, motion[2], axis=0)
    duplicated[2, :, 0] -= np.float32(1e-4)
    duplicated[3, :, 0] += np.float32(1e-4)
    collapsed = collapse_duplicate_clocks(
        duplicated,
        np.ones(5, dtype=np.uint8),
        np.array([0, 1, 2, 2, 3], dtype=np.int64),
    )

    assert collapsed.motion.shape == (4, 17, 3)
    assert collapsed.clocks.tolist() == [0, 1, 2, 3]
    assert collapsed.scale > 0.0
    assert collapsed.motion.dtype == np.dtype("<f4")
    assert np.all(collapsed.motion[:, 11:13, :2].mean(axis=1) == 0.0)


def test_duplicate_conflict_and_noncontiguous_clock_fail_closed() -> None:
    motion = _track(4)
    conflict = np.insert(motion, 2, motion[2], axis=0)
    conflict[3, 7:11, 0] += np.float32(1.0)
    with pytest.raises(PreprocessError, match="DUPLICATE_CONFLICT"):
        collapse_duplicate_clocks(
            conflict,
            np.ones(5, dtype=np.uint8),
            np.array([0, 1, 2, 2, 3], dtype=np.int64),
        )

    with pytest.raises(PreprocessError, match="CLOCK_ORDER"):
        collapse_duplicate_clocks(
            motion,
            np.ones(4, dtype=np.uint8),
            np.array([0, 1, 0, 2], dtype=np.int64),
        )


def test_preprocess_trims_only_ends_and_builds_exact_channel_layout() -> None:
    motion = _track(20)
    frame_mask = np.ones(20, dtype=np.uint8)
    frame_mask[[0, 19]] = 0
    result = preprocess_identity(motion, frame_mask, np.arange(20, dtype=np.int64))

    assert result.motion.shape == (18, 17, 3)
    assert result.trim_bounds == (1, 19)
    assert result.teacher_input.shape == (18, 215)
    assert result.response_input.shape == (18, 269)
    assert result.teacher_input.dtype == np.dtype("<f4")
    assert result.response_input.dtype == np.dtype("<f4")
    assert result.run_bounds.tolist() == [[0, 17]]
    assert np.all(result.edge_lengths > 0.0)
    assert np.all(result.geometry_features.edge_coordinate_mask)
    # Lags are exact zero/false until their own within-run history exists.
    assert np.all(result.response_input[:4, 83:117] == 0.0)
    assert np.all(result.response_input[:4, 218:235] == 0.0)
    assert np.array_equal(
        result.response_input[4, 83:117], result.geometry_features.pose[0].astype(np.float32)
    )


def test_invalid_interior_frame_and_unsupported_clock_edge_abstain() -> None:
    motion = _track(20)
    frame_mask = np.ones(20, dtype=np.uint8)
    frame_mask[10] = 0
    with pytest.raises(PreprocessError, match="FRAME_SUPPORT"):
        preprocess_identity(motion, frame_mask, np.arange(20, dtype=np.int64))

    clocks = np.arange(20, dtype=np.int64)
    clocks[10:] += 4
    with pytest.raises(PreprocessError, match="EDGE_SUPPORT"):
        preprocess_identity(motion, np.ones(20, dtype=np.uint8), clocks)


def test_f4_masked_length_and_lag_state_reset_at_half_open_runs() -> None:
    normalized = _track(12)
    # Input is already treated as normalized by this lower-level fixture.
    joint_mask = np.ones((12, 17), dtype=np.uint8)
    features = build_geometry_features(
        normalized,
        joint_mask,
        np.array([[0, 5], [6, 11]], dtype=np.int32),
    )
    delta = features.geometry[1] - features.geometry[0]
    expected = np.sqrt(np.sum(delta * delta, dtype=np.float64))
    assert features.edge_lengths[0] == pytest.approx(expected, rel=1e-15, abs=1e-15)
    assert features.edge_lengths[5] == 0.0
    # Sample six starts the second run; no lag may cross edge five.
    assert np.all(features.response_input[6, 83:185] == 0.0)
    assert np.all(features.response_input[6, 218:269] == 0.0)
    assert len(BONES) == 16


def test_half_open_run_extraction_preserves_seams() -> None:
    runs = mask_to_half_open_runs(np.array([0, 1, 1, 0, 1, 0, 1, 1, 1, 0], dtype=np.uint8))
    assert runs.tolist() == [[1, 3], [4, 5], [6, 9]]
