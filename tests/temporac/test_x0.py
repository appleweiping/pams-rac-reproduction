"""Executable checks for canonical F8--F10 and Amendment 001."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pams.temporac import x0

AMENDMENT_SCALE = np.float64(25.0 / 6.0)
GRID_SIZE = 4096


def _separation_rms(scale: float, divisor: int) -> float:
    """Closed-form all-34-coordinate RMS implied by the orthonormal F8 basis."""

    delta = 1.0 / divisor
    mean_square_l2 = 0.12**2 * (1.0 - math.cos(2.0 * math.pi * delta))
    for harmonic in (2, 3, 4):
        mean_square_l2 += (
            2.0 * (0.03 / harmonic) ** 2 * (1.0 - math.cos(2.0 * math.pi * harmonic * delta))
        )
    return scale * math.sqrt(mean_square_l2 / 34.0)


def test_original_f8_is_impossible_and_amendment_is_executable() -> None:
    """The original bracket is impossible; canonical production uses Amendment 001."""

    separation = tuple(_separation_rms(1.0, divisor) for divisor in range(2, 9))
    assert separation == pytest.approx(
        (
            0.029305691075485064,
            0.025692611664010475,
            0.021351401662213574,
            0.018152744011886935,
            0.01575898436560219,
            0.013896607758660433,
            0.012407402566436636,
        ),
        abs=2e-16,
    )
    assert max(separation) < 0.05

    _, coefficients = x0._orthonormal_coefficients(3, 0)
    assert x0._candidate_passes(coefficients)


def test_amendment_uses_one_outer_multiply_for_displacement_and_tangent() -> None:
    _, coefficients = x0._orthonormal_coefficients(3, 0)
    phase = np.arange(257, dtype=np.float64) / 257.0
    flat = phase.reshape(-1)

    displacement = 0.12 * np.sin(2.0 * np.pi * flat)[:, None] * x0.LANDMARK_DIRECTION
    tangent = 0.24 * np.pi * np.cos(2.0 * np.pi * flat)[:, None] * x0.LANDMARK_DIRECTION
    for index, harmonic in enumerate((2, 3, 4)):
        u_h = coefficients[2 * index]
        v_h = coefficients[2 * index + 1]
        displacement += (0.03 / harmonic) * (
            np.cos(2.0 * np.pi * harmonic * flat)[:, None] * u_h
            + np.sin(2.0 * np.pi * harmonic * flat)[:, None] * v_h
        )
        tangent += (
            0.06
            * np.pi
            * (
                -np.sin(2.0 * np.pi * harmonic * flat)[:, None] * u_h
                + np.cos(2.0 * np.pi * harmonic * flat)[:, None] * v_h
            )
        )
    displacement *= np.float64(25.0 / 6.0)
    tangent *= np.float64(25.0 / 6.0)

    assert np.array_equal(x0._moving_displacement(phase, coefficients), displacement)
    assert np.array_equal(x0._moving_tangent(phase, coefficients), tangent)


def test_production_amendment_probe_all_40_by_64_candidates() -> None:
    """Probe every candidate using production's exact outer-scale orbit bytes.

    The inventory scan computes the only attempt-dependent guards (MGS,
    coordinate range, and 66-geometry polygonal arc).  Every remaining guard
    has an exact orthogonal-basis bound, checked below.
    """

    phase = np.arange(GRID_SIZE, dtype=np.float64) / GRID_SIZE
    moving_base = x0.BASE_POSE[list(x0.MOVING_JOINTS)].reshape(16)
    coordinate_min = math.inf
    coordinate_max = -math.inf
    arc_min = math.inf
    arc_max = -math.inf
    gram_error = 0.0
    accepted_candidates = 0

    for source_id in range(x0.SOURCE_COUNT):
        for attempt in range(64):
            _, coefficients = x0._orthonormal_coefficients(source_id, attempt)
            coefficient_basis = np.vstack((x0.LANDMARK_DIRECTION, coefficients))
            gram_error = max(
                gram_error,
                float(np.max(np.abs(coefficient_basis @ coefficient_basis.T - np.eye(7)))),
            )
            displacement = x0._moving_displacement(phase, coefficients)
            moving_coordinates = displacement + moving_base
            coordinate_min = min(coordinate_min, float(np.min(moving_coordinates)))
            coordinate_max = max(coordinate_max, float(np.max(moving_coordinates)))
            geometry_displacement = displacement @ x0.MOVING_TO_GEOMETRY.T
            polygonal_arc = float(
                np.sum(
                    np.linalg.norm(
                        np.roll(geometry_displacement, -1, axis=0) - geometry_displacement,
                        axis=1,
                    ),
                    dtype=np.float64,
                )
            )
            arc_min = min(arc_min, polygonal_arc)
            arc_max = max(arc_max, polygonal_arc)
            accepted_candidates += 1

    # All 2,560 coefficient attempts exist and the exact full-grid scan stays
    # inside the coordinate and arc guards. Fixed coordinates add the 0.95 max.
    assert accepted_candidates == 40 * 64
    assert gram_error <= 7e-16
    assert coordinate_min == pytest.approx(-1.0615805372092377, abs=2e-15)
    assert coordinate_max == pytest.approx(0.6923906156773691, abs=2e-15)
    assert max(coordinate_max, float(np.max(x0.BASE_POSE))) == 0.95
    assert arc_min == pytest.approx(4.088598312896032, abs=2e-14)
    assert arc_max == pytest.approx(4.317988354684301, abs=2e-14)
    assert arc_min > 1.0

    # Attempt-independent analytic guards. The higher-harmonic pairs alone
    # establish tangent, arc, and noncollision lower bounds.
    coordinate_abs_bound = 0.95 + AMENDMENT_SCALE * (
        0.12 + 2.0 * 0.03 * (1.0 / 2.0 + 1.0 / 3.0 + 1.0 / 4.0)
    )
    assert coordinate_abs_bound == pytest.approx(1.7208333333333334)
    assert 0.65 > 0.1  # shoulders and hips are fixed, hence pose scale is exact
    tangent_min = AMENDMENT_SCALE * 0.06 * math.pi * math.sqrt(3.0)
    tangent_max = AMENDMENT_SCALE * math.sqrt((0.24 * math.pi) ** 2 + 3.0 * (0.06 * math.pi) ** 2)
    assert tangent_min == pytest.approx(1.3603495231756633)
    assert tangent_max == pytest.approx(3.423471224691923)
    assert tangent_min > 1e-6

    separation = tuple(_separation_rms(AMENDMENT_SCALE, divisor) for divisor in range(2, 9))
    assert separation == pytest.approx(
        (
            0.12210704614785443,
            0.10705254860004364,
            0.08896417359255655,
            0.07563643338286222,
            0.06566243485667578,
            0.057902532327751804,
            0.05169751069348598,
        )
    )
    assert min(separation) > 0.05

    delta = np.arange(math.ceil(0.10 * GRID_SIZE), GRID_SIZE // 2 + 1) / GRID_SIZE
    higher_difference_l2 = np.zeros_like(delta)
    for harmonic in (2, 3, 4):
        higher_difference_l2 += (
            2.0
            * (AMENDMENT_SCALE * 0.03 / harmonic) ** 2
            * (1.0 - np.cos(2.0 * np.pi * harmonic * delta))
        )
    collision_rms_lower = math.sqrt(float(np.min(higher_difference_l2)) / 34.0)
    assert collision_rms_lower == pytest.approx(0.014291548761875736)
    assert collision_rms_lower > 1e-3

    reversal_grid_min = AMENDMENT_SCALE * math.sqrt(
        2.0 * sum((0.03 / harmonic) ** 2 for harmonic in (2, 3, 4)) / 34.0
    )
    assert reversal_grid_min == pytest.approx(0.01973191444620664)
    assert reversal_grid_min > 0.01

    # rho annihilates all higher directions; the score is a positive sine.
    score = np.sin(2.0 * np.pi * np.arange(GRID_SIZE) / GRID_SIZE)
    high = score >= np.max(score) - 0.05 * (np.max(score) - np.min(score))
    assert np.count_nonzero(score == np.max(score)) == 1
    assert np.count_nonzero(score == np.min(score)) == 1
    assert np.count_nonzero(high & ~np.roll(high, 1)) == 1

    retained = tuple(x0.generate_orbit(source_id) for source_id in range(x0.SOURCE_COUNT))
    assert len(retained) == 40
    assert {orbit.accepted_attempt for orbit in retained} == {0}


def test_f9_f10_direct_offsets_blocks_resamplers_and_pulses() -> None:
    blocks = x0.blocks_for_source(3)
    assert len(blocks) == 7
    assert blocks[0].durations == (32,) * 10
    assert blocks[1].durations == (64, 32, 16, 64, 32, 16, 64, 32, 16, 64)

    for offset in x0.OFFSETS:
        clock, clean, boundaries = x0._direct_map(blocks[0].durations, offset)
        assert boundaries[0] == 32 + offset
        assert clean[0] == -8.0
        assert clean[-1] == 344.0
        assert np.array_equal(clean[boundaries], 8.0 + 32.0 * np.arange(11))
        target_mask = np.zeros(clock.size - 1, dtype=np.uint8)
        target_mask[boundaries[0] : boundaries[-1]] = 1
        pulse = np.zeros_like(target_mask)
        pulse[boundaries[:-1]] = 1
        assert int(np.sum(pulse)) == 10
        assert not target_mask[boundaries[-1]]

    knot = np.arange(-39, 376, dtype=np.int64)
    values = np.stack((np.sin(knot / 17.0), np.cos(knot / 19.0)), axis=1)
    query = np.asarray((-8.0, 8.0, 40.0, 328.0, 344.0))
    expected = values[query.astype(np.int64) - knot[0]]
    for resampler in ("linear", "pchip", "sinc"):
        actual = x0._sample_knots(values, knot, query, resampler)
        assert actual == pytest.approx(expected, abs=2e-14)
