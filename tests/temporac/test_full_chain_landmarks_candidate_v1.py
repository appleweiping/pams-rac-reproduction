from __future__ import annotations

import numpy as np

from pams.temporac.full_chain_landmarks_candidate_v1 import full_chain_integer_landmarks


def test_full_chain_keeps_regular_complete_periods() -> None:
    time = np.arange(320, dtype=np.float64)
    phase = 2.0 * np.pi * time / 40.0
    geometry = np.zeros((320, 66), dtype=np.float64)
    geometry[:, 0] = np.sin(phase)
    geometry[:, 1] = np.cos(phase)
    result = full_chain_integer_landmarks(
        geometry,
        np.asarray([[0, 320]], dtype=np.int32),
    )
    assert result.landmarks.size >= 7
    assert result.retained_event_fraction == 1.0
    np.testing.assert_allclose(np.diff(result.landmarks), 40, atol=1)
