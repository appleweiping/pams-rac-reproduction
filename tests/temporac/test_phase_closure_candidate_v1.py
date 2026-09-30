from __future__ import annotations

import numpy as np

from pams.temporac.phase_closure_candidate_v1 import close_phase_on_landmarks


def test_phase_closure_has_positive_unit_winding_and_exact_returns() -> None:
    angle = np.linspace(0.0, 5.0, 33)
    phase = np.column_stack((np.cos(angle), np.sin(angle)))
    landmarks = np.asarray([0, 16, 32], dtype=np.int32)
    closed = close_phase_on_landmarks(phase, landmarks)
    assert np.array_equal(closed[landmarks], np.asarray([[1.0, 0.0]] * 3))
    for left, right in zip(landmarks[:-1], landmarks[1:], strict=True):
        cross = (
            closed[left:right, 0] * closed[left + 1 : right + 1, 1]
            - closed[left:right, 1] * closed[left + 1 : right + 1, 0]
        )
        dot = np.sum(closed[left:right] * closed[left + 1 : right + 1], axis=1)
        increments = np.arctan2(cross, dot) / (2.0 * np.pi)
        assert np.all(increments > 0.0)
        np.testing.assert_allclose(np.sum(increments), 1.0, atol=1e-12)


def test_geometry_arc_closure_ignores_raw_phase_speed() -> None:
    phase = np.tile(np.asarray([[1.0, 0.0]]), (33, 1))
    geometry = np.zeros((33, 66), dtype=np.float64)
    geometry[:, 0] = np.linspace(0.0, 2.0, 33)
    landmarks = np.asarray([0, 16, 32], dtype=np.int32)
    closed = close_phase_on_landmarks(phase, landmarks, geometry=geometry)
    np.testing.assert_allclose(closed[8], [-1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(closed[24], [-1.0, 0.0], atol=1e-12)
