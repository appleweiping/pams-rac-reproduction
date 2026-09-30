from __future__ import annotations

import numpy as np

from pams.temporac.dtw_phase_candidate_v1 import dtw_phase_coordinates


def test_dtw_coordinates_are_monotone_and_endpoint_bound() -> None:
    first_time = np.linspace(0.0, 1.0, 31)
    second_time = np.linspace(0.0, 1.0, 47) ** 1.5
    first = np.column_stack((np.sin(2 * np.pi * first_time), np.cos(2 * np.pi * first_time)))
    second = np.column_stack(
        (np.sin(2 * np.pi * second_time), np.cos(2 * np.pi * second_time))
    )
    left, right = dtw_phase_coordinates(first, second)
    assert left[0] == right[0] == 0.0
    assert left[-1] == right[-1] == 1.0
    assert np.all(np.diff(left) > 0.0)
    assert np.all(np.diff(right) > 0.0)
