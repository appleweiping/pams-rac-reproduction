from __future__ import annotations

import numpy as np

from pams.temporac.dtw_pair_selector_candidate_v1 import (
    phase_from_dtw_pair,
    select_dtw_pair,
)


def test_dtw_pair_selects_and_builds_two_unit_windings() -> None:
    time = np.arange(320, dtype=np.float64)
    phase = 2.0 * np.pi * time / 40.0
    geometry = np.zeros((320, 66), dtype=np.float64)
    geometry[:, 0] = np.sin(phase)
    geometry[:, 1] = np.cos(phase)
    mask = np.ones_like(geometry)
    result = select_dtw_pair(
        geometry,
        geometry,
        mask,
        np.asarray([[0, 320]], dtype=np.int32),
    )
    assert result.landmarks.size == 3
    output = phase_from_dtw_pair(320, result)
    np.testing.assert_allclose(output[result.landmarks], [[1.0, 0.0]] * 3, atol=0.0)
