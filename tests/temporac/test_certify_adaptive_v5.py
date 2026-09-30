from __future__ import annotations

import numpy as np

from pams.temporac.certify_adaptive_v5 import stable_pair_integer_landmarks_v5


def test_v5_keeps_a_structurally_sampled_periodic_pair() -> None:
    time = np.arange(320, dtype=np.float64)
    phase = 2.0 * np.pi * time / 40.0
    geometry = np.zeros((320, 66), dtype=np.float64)
    geometry[:, 0] = np.sin(phase)
    geometry[:, 1] = np.cos(phase)
    result = stable_pair_integer_landmarks_v5(
        geometry,
        np.asarray([[0, 320]], dtype=np.int32),
    )
    assert result.landmarks.size == 3
    np.testing.assert_allclose(np.diff(result.landmarks), 40, atol=1)
