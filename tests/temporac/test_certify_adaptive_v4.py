from __future__ import annotations

import numpy as np

from pams.temporac.certify_adaptive_v4 import stable_pair_integer_landmarks


def test_stable_pair_returns_exactly_two_adjacent_cycles() -> None:
    time = np.arange(320, dtype=np.float64)
    phase = 2.0 * np.pi * time / 40.0
    geometry = np.zeros((320, 66), dtype=np.float64)
    geometry[:, 0] = np.sin(phase)
    geometry[:, 1] = np.cos(phase)
    result = stable_pair_integer_landmarks(
        geometry,
        np.asarray([[0, 320]], dtype=np.int32),
    )
    assert result.landmarks.size == 3
    np.testing.assert_allclose(np.diff(result.landmarks), 40, atol=1)
    assert result.pair_geometry_huber is not None
    assert result.pair_geometry_huber < 1e-12
