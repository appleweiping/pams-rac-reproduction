from __future__ import annotations

import numpy as np

from pams.temporac.certify import integer_landmarks
from pams.temporac.certify_adaptive_v2 import (
    adaptive_integer_landmarks,
    hybrid_integer_landmarks,
)
from pams.temporac.preprocess import preprocess_identity
from pams.temporac.x0 import generate_view


def test_hybrid_landmarks_preserve_frozen_x0_contract() -> None:
    for source_id in (0, 7, 23, 31):
        view = generate_view(source_id, 0, resampler="linear", offset=0)
        prepared = preprocess_identity(
            view.motion,
            np.ones(view.motion.shape[0], dtype=np.uint8),
            view.clock,
        )
        sample_bounds = np.array(prepared.run_bounds, copy=True)
        sample_bounds[:, 1] += 1
        expected = integer_landmarks(
            prepared.geometry_features.geometry, sample_bounds
        )
        result = hybrid_integer_landmarks(
            prepared.geometry_features.geometry, sample_bounds
        )
        assert expected.size >= 3
        assert np.array_equal(result.landmarks, expected)


def test_static_track_does_not_create_fake_cycles() -> None:
    geometry = np.zeros((64, 66), dtype=np.float64)
    result = adaptive_integer_landmarks(geometry, np.asarray([[0, 64]], dtype=np.int32))
    assert result.landmarks.size == 0
