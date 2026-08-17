from __future__ import annotations

import numpy as np
import pytest

from pams.temporac.cue import (
    UNIFORM_GATE,
    Window,
    compute_routing,
    compute_window_cue,
    window_schedule,
)
from pams.temporac.quadrature import (
    Subedge,
    arc_static_code,
    arc_static_code_from_subedges,
    shared_edge_quadrature,
)


def test_shared_quadrature_is_one_irregular_midpoint_trapezoid() -> None:
    clocks = np.array([0, 1, 3, 4, 7, 8], dtype=np.int64)
    mask = np.ones(5, dtype=np.uint8)
    quadrature = shared_edge_quadrature(clocks, mask)
    x = (clocks[:-1] + clocks[1:]) / 2.0
    trapezoid = np.array(
        [
            (x[1] - x[0]) / 2.0,
            (x[2] - x[0]) / 2.0,
            (x[3] - x[1]) / 2.0,
            (x[4] - x[2]) / 2.0,
            (x[4] - x[3]) / 2.0,
        ]
    )
    taper = np.sin(np.pi * (x - x[0]) / (x[-1] - x[0])) ** 2
    taper[[0, -1]] = 0.0
    assert np.array_equal(quadrature.midpoints, x)
    assert np.allclose(quadrature.weights, trapezoid * taper, rtol=0.0, atol=1e-15)
    assert quadrature.weights[[0, -1]].tolist() == [0.0, 0.0]


def test_f5_squared_linear_integral_is_subdivision_invariant() -> None:
    values = np.array([[0.0, 2.0], [1.0, -1.0], [3.0, 4.0], [2.0, 8.0]], dtype=np.float64)
    lengths = np.array([1.25, 0.75, 2.5], dtype=np.float64)
    whole = arc_static_code(values, lengths, np.array([[0, 3]], dtype=np.int32))
    subdivided = arc_static_code_from_subedges(
        values,
        lengths,
        [
            tuple(
                subedge
                for edge in range(3)
                for subedge in (Subedge(edge, 0.0, 0.5), Subedge(edge, 0.5, 1.0))
            )
        ],
    )
    tolerance = 2.0 * np.maximum(np.spacing(np.abs(whole)), np.finfo(np.float64).tiny)
    assert np.all(np.abs(whole - subdivided) <= tolerance)


def test_128_32_schedule_appends_terminal_right_aligned_window() -> None:
    windows = window_schedule(np.array([[0, 200]], dtype=np.int32))
    assert [(window.edge_start, window.edge_stop) for window in windows] == [
        (0, 127),
        (32, 159),
        (64, 191),
        (73, 200),
    ]
    short = window_schedule(np.array([[7, 20]], dtype=np.int32))
    assert [(window.edge_start, window.edge_stop) for window in short] == [(7, 20)]


def _periodic_geometry(sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    increments = np.resize(np.array([1, 2, 1, 1], dtype=np.int64), sample_count - 1)
    clocks = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(increments)))
    phase_offsets = np.linspace(0.0, 2.0 * np.pi, 66, endpoint=False)
    geometry = np.sin(2.0 * np.pi * clocks[:, None] / 16.0 + phase_offsets[None, :])
    return geometry, clocks


def test_irregular_clock_nudft_is_reliable_without_hard_period_selection() -> None:
    geometry, clocks = _periodic_geometry(128)
    cue = compute_window_cue(
        geometry,
        np.ones((127, 66), dtype=np.uint8),
        clocks,
        np.ones(127, dtype=np.uint8),
        Window(0, 0, 127),
    )
    assert cue.failure is None
    assert cue.valid_coordinate_count == 66
    assert cue.gamma >= 0.15
    assert cue.distribution.shape == (48,)
    assert np.sum(cue.distribution) == pytest.approx(1.0, rel=0.0, abs=1e-14)
    assert np.isfinite(cue.u)


def test_run_reference_and_all_five_same_response_interventions() -> None:
    geometry, clocks = _periodic_geometry(257)
    result = compute_routing(
        geometry,
        np.ones((256, 66), dtype=np.uint8),
        clocks,
        np.ones(256, dtype=np.uint8),
        np.ones(256, dtype=np.float64),
        np.array([[0, 256]], dtype=np.int32),
    )
    assert result.references[0].available
    assert result.references[0].reliable_windows >= 3
    for mode in ("local", "global", "uniform", "blocked", "shuffled"):
        gates = result.gates(mode)
        assert np.allclose(gates.sum(axis=1), 1.0, rtol=0.0, atol=1e-14)
    assert np.array_equal(result.blocked_gates, result.uniform_gates)
    assert np.array_equal(result.uniform_gates, np.tile(UNIFORM_GATE, (len(result.windows), 1)))
    assert np.all(result.global_gates == result.global_gates[0])
    reliable = [index for index, cue in enumerate(result.cues) if cue.reliable]
    assert np.array_equal(
        result.shuffled_gates[reliable], np.roll(result.local_gates[reliable], 1, axis=0)
    )
