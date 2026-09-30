from __future__ import annotations

import numpy as np

from pams.temporac.certify import certificate_track_from_x0, certify_target
from pams.temporac.certify_target_adaptive_v7 import certify_target_adaptive_v7
from pams.temporac.x0 import generate_view


def test_adaptive_v7_is_ledger_identical_on_x0() -> None:
    view = generate_view(23, 0, resampler="linear", offset=0)
    track = certificate_track_from_x0(view, teacher_sha256="0" * 64)
    expected = certify_target(track, "X0")
    actual = certify_target_adaptive_v7(track, "X0")
    assert actual.status == expected.status
    assert np.array_equal(actual.reasons, expected.reasons)
    assert np.array_equal(actual.landmarks, expected.landmarks)
    assert np.array_equal(actual.traversal_bounds, expected.traversal_bounds)
    assert np.array_equal(actual.pulse, expected.pulse)
    assert np.array_equal(actual.chi, expected.chi)
