from __future__ import annotations

import numpy as np

from pams.temporac.multi_cycle_dtw_template_candidate_v2 import (
    multi_cycle_dtw_median_reconstruction,
)


def test_multi_cycle_dtw_reconstructs_repeated_sequence() -> None:
    time = np.arange(201, dtype=np.float64)
    phase = 2.0 * np.pi * time / 40.0
    geometry = np.column_stack((np.sin(phase), np.cos(phase)))
    mask = np.ones_like(geometry)
    landmarks = np.arange(0, 201, 40, dtype=np.int32)
    reconstruction, coordinates = multi_cycle_dtw_median_reconstruction(
        geometry,
        mask,
        geometry,
        landmarks,
    )
    assert len(coordinates) == 5
    np.testing.assert_allclose(reconstruction[1:200], geometry[1:200], atol=5e-4)

