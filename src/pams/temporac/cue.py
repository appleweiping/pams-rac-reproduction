"""Detached irregular-clock NUDFT cue and frozen routing interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pams.temporac.preprocess import validate_run_bounds
from pams.temporac.quadrature import (
    QuadratureError,
    neumaier_sum,
    neumaier_sum_array,
    shared_edge_quadrature,
)

Float64Array = NDArray[np.float64]
Int32Array = NDArray[np.int32]
BoolArray = NDArray[np.bool_]
RoutingMode = Literal["local", "global", "uniform", "blocked", "shuffled"]

WINDOW_SAMPLES: Final = 128
WINDOW_EDGES: Final = 127
WINDOW_STRIDE: Final = 32
PERIOD_COUNT: Final = 48
PERIODS: Final[Float64Array] = np.exp(
    np.log(np.float64(5.0))
    + np.arange(PERIOD_COUNT, dtype=np.float64)
    * np.log(np.float64(128.0 / 5.0))
    / np.float64(PERIOD_COUNT - 1)
)
FREQUENCIES: Final[Float64Array] = np.asarray(1.0 / PERIODS, dtype=np.float64)
LOG_FREQUENCIES: Final[Float64Array] = np.log(FREQUENCIES)
UNIFORM_SPECTRUM: Final[Float64Array] = np.full(
    PERIOD_COUNT, np.float64(1.0 / PERIOD_COUNT), dtype=np.float64
)
UNIFORM_U: Final = float(np.mean(LOG_FREQUENCIES, dtype=np.float64))
ROUTER_MU: Final[Float64Array] = np.asarray((-np.log(1.5), 0.0, np.log(1.5)), dtype=np.float64)
ROUTER_SIGMA: Final = np.float64(np.log(1.5) / 2.0)
UNIFORM_GATE: Final[Float64Array] = np.full(3, np.float64(1.0 / 3.0))

for _constant in (PERIODS, FREQUENCIES, LOG_FREQUENCIES, UNIFORM_SPECTRUM, ROUTER_MU, UNIFORM_GATE):
    _constant.setflags(write=False)


class CueError(ValueError):
    """Raised when cue inputs violate their package-private interface."""


@dataclass(frozen=True, slots=True)
class Window:
    """A half-open interval of edge slots inside one valid run."""

    run_index: int
    edge_start: int
    edge_stop: int

    def __post_init__(self) -> None:
        if self.run_index < 0 or not (0 <= self.edge_start < self.edge_stop):
            raise CueError("window must be a nonempty ordered edge interval")

    @property
    def sample_start(self) -> int:
        return self.edge_start

    @property
    def sample_stop(self) -> int:
        return self.edge_stop + 1

    @property
    def edge_count(self) -> int:
        return self.edge_stop - self.edge_start


@dataclass(frozen=True, slots=True)
class WindowCue:
    """Detached spectral cue receipt for one window."""

    window: Window
    spectrum: Float64Array
    distribution: Float64Array
    u: float
    gamma: float
    valid_coordinate_count: int
    valid_edge_count: int
    midpoint_span: float
    quadrature_weights: Float64Array
    failure: str | None

    def __post_init__(self) -> None:
        for value in (self.spectrum, self.distribution, self.quadrature_weights):
            value.setflags(write=False)

    @property
    def reliable(self) -> bool:
        return self.gamma > 0.0


@dataclass(frozen=True, slots=True)
class RunReference:
    run_index: int
    u: float
    available: bool
    reliable_windows: int
    total_weight: float


@dataclass(frozen=True, slots=True)
class RoutingResult:
    """All same-response v4 routing interventions on one identity."""

    windows: tuple[Window, ...]
    cues: tuple[WindowCue, ...]
    references: tuple[RunReference, ...]
    local_gates: Float64Array
    global_gates: Float64Array
    uniform_gates: Float64Array
    blocked_gates: Float64Array
    shuffled_gates: Float64Array
    coverage: Int32Array

    def __post_init__(self) -> None:
        for value in (
            self.local_gates,
            self.global_gates,
            self.uniform_gates,
            self.blocked_gates,
            self.shuffled_gates,
            self.coverage,
        ):
            value.setflags(write=False)

    def gates(self, mode: RoutingMode) -> Float64Array:
        """Return the frozen gate matrix for an inference intervention."""

        return {
            "local": self.local_gates,
            "global": self.global_gates,
            "uniform": self.uniform_gates,
            "blocked": self.blocked_gates,
            "shuffled": self.shuffled_gates,
        }[mode]


def window_schedule(run_bounds: ArrayLike) -> tuple[Window, ...]:
    """Create 128/32 windows plus the terminal right-aligned start."""

    raw = np.asarray(run_bounds)
    if raw.ndim != 2 or raw.shape[1:] != (2,) or raw.dtype.kind not in "iu":
        raise CueError("run_bounds must be an integer [R,2] array")
    if raw.size == 0:
        return ()
    edge_count = int(np.max(raw[:, 1]))
    bounds = validate_run_bounds(raw, edge_count)
    windows: list[Window] = []
    for run_index, (start, stop) in enumerate(bounds.tolist()):
        sample_count = stop - start + 1
        if sample_count <= WINDOW_SAMPLES:
            starts = [start]
        else:
            final_start = stop - WINDOW_EDGES
            starts = list(range(start, final_start + 1, WINDOW_STRIDE))
            if starts[-1] != final_start:
                starts.append(final_start)
        windows.extend(
            Window(run_index, window_start, min(window_start + WINDOW_EDGES, stop))
            for window_start in starts
        )
    return tuple(windows)


def _failure_cue(
    window: Window,
    edge_count: int,
    failure: str,
    *,
    valid_edges: int = 0,
    span: float = 0.0,
) -> WindowCue:
    return WindowCue(
        window=window,
        spectrum=np.zeros(PERIOD_COUNT, dtype=np.float64),
        distribution=UNIFORM_SPECTRUM.copy(),
        u=UNIFORM_U,
        gamma=0.0,
        valid_coordinate_count=0,
        valid_edge_count=valid_edges,
        midpoint_span=span,
        quadrature_weights=np.zeros(edge_count, dtype=np.float64),
        failure=failure,
    )


def _coordinate_spectrum(
    velocity: Float64Array,
    midpoint: Float64Array,
    weight: Float64Array,
) -> Float64Array | None:
    if velocity.size < 3:
        return None
    span = float(midpoint[-1] - midpoint[0])
    total_weight = neumaier_sum(weight)
    if not np.isfinite(span) or span <= 0.0 or not np.isfinite(total_weight) or total_weight <= 0.0:
        return None
    center = np.float64((midpoint[0] + midpoint[-1]) / 2.0)
    half_span = np.float64(span / 2.0)
    z = (midpoint - center) / half_span
    s0 = total_weight
    s1 = neumaier_sum(weight * z)
    s2 = neumaier_sum(weight * z * z)
    t0 = neumaier_sum(weight * velocity)
    t1 = neumaier_sum(weight * z * velocity)
    determinant = np.float64(s0 * s2 - s1 * s1)
    if (
        s0 <= 0.0
        or s2 <= 0.0
        or not np.isfinite(determinant)
        or determinant < np.float64(1e-12) * np.float64(s0) * np.float64(s2)
    ):
        return None
    alpha = np.float64((t0 * s2 - t1 * s1) / determinant)
    beta = np.float64((t1 * s0 - t0 * s1) / determinant)
    residual = velocity - alpha - beta * z
    power = neumaier_sum(weight * residual * residual)
    denominator = np.float64(s0 * power)
    if not np.isfinite(denominator) or denominator <= np.float64(1e-12):
        return None

    spectrum = np.empty(PERIOD_COUNT, dtype=np.float64)
    for frequency_index, frequency in enumerate(FREQUENCIES):
        angle = np.float64(2.0 * np.pi) * frequency * midpoint
        real = neumaier_sum(weight * residual * np.cos(angle))
        imaginary = neumaier_sum(-weight * residual * np.sin(angle))
        spectrum[frequency_index] = np.float64((real * real + imaginary * imaginary) / denominator)
    if not np.all(np.isfinite(spectrum)) or np.any(spectrum < 0.0):
        return None
    return spectrum


def compute_window_cue(
    geometry: ArrayLike,
    coordinate_edge_mask: ArrayLike,
    clocks: ArrayLike,
    edge_mask: ArrayLike,
    window: Window,
) -> WindowCue:
    """Compute F6 and reliability on one shared irregular-clock quadrature."""

    g = np.asarray(geometry, dtype=np.float64)
    coordinate_mask_raw = np.asarray(coordinate_edge_mask)
    q_raw = np.asarray(clocks)
    base_mask_raw = np.asarray(edge_mask)
    if g.ndim != 2 or g.shape[1] != 66 or g.shape[0] < 2:
        raise CueError("geometry must have shape [T,66]")
    edge_count = g.shape[0] - 1
    if coordinate_mask_raw.shape != (edge_count, 66) or coordinate_mask_raw.dtype.kind not in "bu":
        raise CueError("coordinate_edge_mask must be binary [E,66]")
    if not np.all((coordinate_mask_raw == 0) | (coordinate_mask_raw == 1)):
        raise CueError("coordinate_edge_mask must be binary")
    if q_raw.shape != (g.shape[0],) or q_raw.dtype.kind not in "iu":
        raise CueError("clocks must be integer shape [T]")
    if base_mask_raw.shape != (edge_count,) or base_mask_raw.dtype.kind not in "bu":
        raise CueError("edge_mask must be binary shape [E]")
    if not np.all((base_mask_raw == 0) | (base_mask_raw == 1)):
        raise CueError("edge_mask must be binary")
    if not (0 <= window.edge_start < window.edge_stop <= edge_count):
        raise CueError("window is outside the edge sequence")
    if not np.all(np.isfinite(g)):
        raise CueError("geometry must be finite")

    coordinate_mask = np.asarray(coordinate_mask_raw, dtype=np.bool_)
    base_mask = np.asarray(base_mask_raw, dtype=np.bool_)
    try:
        quadrature = shared_edge_quadrature(
            q_raw,
            base_mask,
            edge_start=window.edge_start,
            edge_stop=window.edge_stop,
        )
    except QuadratureError as error:
        valid_edges = int(np.count_nonzero(base_mask[window.edge_start : window.edge_stop]))
        return _failure_cue(window, edge_count, str(error), valid_edges=valid_edges)

    indices = quadrature.edge_indices
    valid_edge_count = int(indices.size)
    span = quadrature.span
    if valid_edge_count < 16 or span < 16.0:
        return _failure_cue(
            window,
            edge_count,
            "cue requires 16 base-valid edges and midpoint span 16",
            valid_edges=valid_edge_count,
            span=span,
        )
    clocks_f64 = np.asarray(q_raw, dtype=np.float64)
    clock_delta = np.diff(clocks_f64)
    geometry_delta = np.diff(g, axis=0)
    valid_spectra: list[Float64Array] = []
    for coordinate in range(66):
        subset = indices[coordinate_mask[indices, coordinate]]
        if subset.size < 3:
            continue
        weight = quadrature.weights[subset]
        midpoint = quadrature.midpoints[subset]
        if neumaier_sum(weight) <= 0.0 or float(midpoint[-1] - midpoint[0]) <= 0.0:
            continue
        velocity = geometry_delta[subset, coordinate] / clock_delta[subset]
        spectrum = _coordinate_spectrum(velocity, midpoint, weight)
        if spectrum is not None:
            valid_spectra.append(spectrum)

    valid_coordinate_count = len(valid_spectra)
    if valid_coordinate_count < 16:
        failed = _failure_cue(
            window,
            edge_count,
            "cue requires at least 16 valid coordinate spectra",
            valid_edges=valid_edge_count,
            span=span,
        )
        return WindowCue(
            window=failed.window,
            spectrum=failed.spectrum,
            distribution=failed.distribution,
            u=failed.u,
            gamma=failed.gamma,
            valid_coordinate_count=valid_coordinate_count,
            valid_edge_count=failed.valid_edge_count,
            midpoint_span=failed.midpoint_span,
            quadrature_weights=quadrature.weights.copy(),
            failure=failed.failure,
        )

    average_spectrum = neumaier_sum_array(np.stack(valid_spectra, axis=0)) / np.float64(
        valid_coordinate_count
    )
    stabilized = average_spectrum + np.float64(1e-8)
    normalizer = neumaier_sum(stabilized)
    if not np.isfinite(normalizer) or normalizer <= 0.0:
        return _failure_cue(
            window,
            edge_count,
            "spectral probability denominator is invalid",
            valid_edges=valid_edge_count,
            span=span,
        )
    distribution = stabilized / np.float64(normalizer)
    u = neumaier_sum(distribution * LOG_FREQUENCIES)
    entropy_term = neumaier_sum(distribution * np.log(distribution + np.float64(1e-8)))
    entropy_confidence = float(
        np.clip(1.0 + entropy_term / np.log(np.float64(PERIOD_COUNT)), 0.0, 1.0)
    )
    gamma = float(
        min(1.0, valid_edge_count / 32.0)
        * min(1.0, span / 128.0)
        * (valid_coordinate_count / 66.0)
        * entropy_confidence
    )
    if not np.isfinite(u) or not np.isfinite(gamma):
        return _failure_cue(
            window,
            edge_count,
            "cue statistic is nonfinite",
            valid_edges=valid_edge_count,
            span=span,
        )
    if gamma < 0.15:
        gamma = 0.0
    return WindowCue(
        window=window,
        spectrum=np.ascontiguousarray(average_spectrum),
        distribution=np.ascontiguousarray(distribution),
        u=u,
        gamma=gamma,
        valid_coordinate_count=valid_coordinate_count,
        valid_edge_count=valid_edge_count,
        midpoint_span=span,
        quadrature_weights=quadrature.weights.copy(),
        failure=None,
    )


def _coverage(windows: tuple[Window, ...], edge_count: int) -> Int32Array:
    coverage = np.zeros(edge_count, dtype=np.int32)
    for window in windows:
        coverage[window.edge_start : window.edge_stop] += 1
    return coverage


def _run_reference(
    run_index: int,
    cue_indices: list[int],
    cues: tuple[WindowCue, ...],
    edge_lengths: Float64Array,
    coverage: Int32Array,
) -> RunReference:
    weighted: list[tuple[float, int, float]] = []
    for index in cue_indices:
        cue = cues[index]
        if not cue.reliable:
            continue
        window_indices = np.arange(cue.window.edge_start, cue.window.edge_stop)
        positive_coverage = coverage[window_indices] > 0
        if not np.all(positive_coverage):
            raise CueError("window coverage must be positive inside every window")
        beta = cue.gamma * neumaier_sum(
            cue.quadrature_weights[window_indices]
            * edge_lengths[window_indices]
            / coverage[window_indices].astype(np.float64)
        )
        if np.isfinite(beta) and beta > 0.0:
            weighted.append((cue.u, cue.window.edge_start, beta))
    if len(weighted) < 3:
        return RunReference(run_index, UNIFORM_U, False, len(weighted), 0.0)
    weighted.sort(key=lambda item: (item[0], item[1]))
    total = neumaier_sum(item[2] for item in weighted)
    if not np.isfinite(total) or total <= 0.0:
        return RunReference(run_index, UNIFORM_U, False, len(weighted), 0.0)
    cumulative = np.float64(0.0)
    correction = np.float64(0.0)
    reference = weighted[-1][0]
    for u, _, weight in weighted:
        candidate = np.float64(cumulative + np.float64(weight))
        if abs(cumulative) >= abs(weight):
            correction = np.float64(correction + ((cumulative - candidate) + weight))
        else:
            correction = np.float64(correction + ((weight - candidate) + cumulative))
        cumulative = candidate
        if float(cumulative + correction) >= total / 2.0:
            reference = u
            break
    return RunReference(run_index, reference, True, len(weighted), total)


def _gate_for_cue(cue: WindowCue, reference: RunReference) -> Float64Array:
    if not reference.available or cue.gamma == 0.0:
        return UNIFORM_GATE.copy()
    relative = np.float64(cue.u - reference.u)
    score = -((relative - ROUTER_MU) ** 2) / (np.float64(2.0) * ROUTER_SIGMA * ROUTER_SIGMA)
    score -= np.max(score)
    exponent = np.exp(score)
    softmax = exponent / np.sum(exponent, dtype=np.float64)
    gate = np.float64(cue.gamma) * softmax + np.float64(1.0 - cue.gamma) * UNIFORM_GATE
    if not np.all(np.isfinite(gate)) or np.any(gate < 0.0):
        raise CueError("router gate is invalid")
    return np.asarray(gate, dtype=np.float64)


def compute_routing(
    geometry: ArrayLike,
    coordinate_edge_mask: ArrayLike,
    clocks: ArrayLike,
    edge_mask: ArrayLike,
    edge_lengths: ArrayLike,
    run_bounds: ArrayLike,
) -> RoutingResult:
    """Compute local/global/uniform/blocked/shuffled gates with per-run reset."""

    g = np.asarray(geometry, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != 66 or g.shape[0] < 2:
        raise CueError("geometry must have shape [T,66]")
    edge_count = g.shape[0] - 1
    bounds = validate_run_bounds(run_bounds, edge_count)
    lengths = np.asarray(edge_lengths, dtype=np.float64)
    if lengths.shape != (edge_count,) or not np.all(np.isfinite(lengths)):
        raise CueError("edge_lengths must be finite shape [E]")
    windows = window_schedule(bounds)
    cues = tuple(
        compute_window_cue(
            geometry,
            coordinate_edge_mask,
            clocks,
            edge_mask,
            window,
        )
        for window in windows
    )
    coverage = _coverage(windows, edge_count)
    references_list: list[RunReference] = []
    indices_by_run: list[list[int]] = []
    for run_index in range(bounds.shape[0]):
        indices = [index for index, window in enumerate(windows) if window.run_index == run_index]
        indices_by_run.append(indices)
        references_list.append(_run_reference(run_index, indices, cues, lengths, coverage))
    references = tuple(references_list)

    local = np.empty((len(windows), 3), dtype=np.float64)
    for index, cue in enumerate(cues):
        local[index] = _gate_for_cue(cue, references[cue.window.run_index])

    global_gates = np.empty_like(local)
    for run_index, (start, stop) in enumerate(bounds.tolist()):
        full_window = Window(run_index, start, stop)
        # The full-run cue is an intervention and not appended to the frozen
        # local window/reference receipt.
        full_cue = compute_window_cue(
            geometry,
            coordinate_edge_mask,
            clocks,
            edge_mask,
            full_window,
        )
        gate = _gate_for_cue(full_cue, references[run_index])
        global_gates[indices_by_run[run_index]] = gate

    uniform = np.tile(UNIFORM_GATE, (len(windows), 1))
    # Blocking the cue is definitionally the exact uniform byte path.
    blocked = uniform.copy()
    shuffled = local.copy()
    for indices in indices_by_run:
        reliable = [index for index in indices if cues[index].reliable]
        if reliable:
            shifted = np.roll(local[reliable], shift=1, axis=0)
            shuffled[reliable] = shifted
    if not np.array_equal(blocked, uniform):
        raise AssertionError("blocked and uniform routing must be bit-identical")
    return RoutingResult(
        windows=windows,
        cues=cues,
        references=references,
        local_gates=np.ascontiguousarray(local),
        global_gates=np.ascontiguousarray(global_gates),
        uniform_gates=np.ascontiguousarray(uniform),
        blocked_gates=np.ascontiguousarray(blocked),
        shuffled_gates=np.ascontiguousarray(shuffled),
        coverage=np.ascontiguousarray(coverage),
    )


__all__ = [
    "CueError",
    "FREQUENCIES",
    "PERIODS",
    "RoutingMode",
    "RoutingResult",
    "RunReference",
    "Window",
    "WindowCue",
    "compute_routing",
    "compute_window_cue",
    "window_schedule",
]
