from __future__ import annotations

import numpy as np

from pams.temporac.certify_adaptive_v3 import periodic_integer_landmarks


def test_periodic_landmarks_recover_full_period_instead_of_half_return() -> None:
    time = np.arange(320, dtype=np.float64)
    phase = 2.0 * np.pi * time / 40.0
    geometry = np.zeros((320, 66), dtype=np.float64)
    geometry[:, 0] = np.sin(phase)
    geometry[:, 1] = np.cos(phase)
    geometry[:, 2] = 0.25 * np.sin(2.0 * phase)
    result = periodic_integer_landmarks(geometry, np.asarray([[0, 320]], dtype=np.int32))
    assert result.estimated_period == 40
    assert result.landmarks.size >= 7
    np.testing.assert_allclose(np.diff(result.landmarks), 40, atol=1)


def test_periodic_landmarks_abstain_on_aperiodic_drift() -> None:
    time = np.arange(320, dtype=np.float64)
    geometry = np.zeros((320, 66), dtype=np.float64)
    geometry[:, 0] = time
    result = periodic_integer_landmarks(geometry, np.asarray([[0, 320]], dtype=np.int32))
    assert result.landmarks.size == 0
