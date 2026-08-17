"""Binary64 compensated quadrature primitives for TempoRAC.

This module is deliberately self-contained.  In particular, it does not use
the historical PAMS/WARP signal stack: the v4 contract defines one edge
quadrature and one ordered compensated summation rule of its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


class QuadratureError(ValueError):
    """Raised when a required v4 quadrature denominator is not admissible."""


@dataclass(frozen=True, slots=True)
class EdgeQuadrature:
    """One shared window-level edge quadrature.

    ``weights`` has the same length as the complete edge sequence and is zero
    outside ``edge_indices``.  Coordinate-specific consumers subset this
    array; they must never recompute trapezoid weights.
    """

    midpoints: Float64Array
    weights: Float64Array
    edge_indices: NDArray[np.int64]
    span: float

    def __post_init__(self) -> None:
        for value in (self.midpoints, self.weights, self.edge_indices):
            value.setflags(write=False)


@dataclass(frozen=True, slots=True)
class Subedge:
    """Positive fractional interval within a base edge."""

    edge: int
    start: float = 0.0
    stop: float = 1.0

    def __post_init__(self) -> None:
        if self.edge < 0:
            raise QuadratureError("subedge index must be nonnegative")
        if not (np.isfinite(self.start) and np.isfinite(self.stop)):
            raise QuadratureError("subedge endpoints must be finite")
        if not (0.0 <= self.start < self.stop <= 1.0):
            raise QuadratureError("subedge must satisfy 0 <= start < stop <= 1")


def neumaier_sum(values: Iterable[float]) -> float:
    """Sum scalars in input order using binary64 Neumaier compensation."""

    total = np.float64(0.0)
    correction = np.float64(0.0)
    for raw_value in values:
        value = np.float64(raw_value)
        candidate = np.float64(total + value)
        if abs(total) >= abs(value):
            correction = np.float64(correction + ((total - candidate) + value))
        else:
            correction = np.float64(correction + ((value - candidate) + total))
        total = candidate
    return float(np.float64(total + correction))


def neumaier_sum_array(values: ArrayLike) -> Float64Array:
    """Sum an ordered leading axis with elementwise Neumaier compensation."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim < 1:
        raise QuadratureError("array summation requires a leading axis")
    total = np.zeros(array.shape[1:], dtype=np.float64)
    correction = np.zeros_like(total)
    for value in array:
        candidate = total + value
        use_total = np.abs(total) >= np.abs(value)
        correction += np.where(
            use_total,
            (total - candidate) + value,
            (value - candidate) + total,
        )
        total = candidate
    return np.asarray(total + correction, dtype=np.float64)


def _as_binary_mask(mask: ArrayLike, *, length: int, name: str) -> BoolArray:
    raw = np.asarray(mask)
    if raw.shape != (length,):
        raise QuadratureError(f"{name} must have shape ({length},)")
    if raw.dtype.kind not in "bu":
        raise QuadratureError(f"{name} must be boolean or unsigned integer")
    if not np.all((raw == 0) | (raw == 1)):
        raise QuadratureError(f"{name} must be binary")
    return np.asarray(raw, dtype=np.bool_)


def edge_midpoints(clocks: ArrayLike) -> Float64Array:
    """Return physical midpoints for strictly increasing retained clocks."""

    raw = np.asarray(clocks)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise QuadratureError("clocks must be a one-dimensional integer array")
    if raw.size < 2:
        raise QuadratureError("at least two clocks are required")
    q = np.asarray(raw, dtype=np.float64)
    if not np.all(np.diff(q) > 0.0):
        raise QuadratureError("retained clocks must be strictly increasing")
    return np.asarray((q[:-1] + q[1:]) / np.float64(2.0), dtype=np.float64)


def shared_edge_quadrature(
    clocks: ArrayLike,
    edge_mask: ArrayLike,
    *,
    edge_start: int = 0,
    edge_stop: int | None = None,
) -> EdgeQuadrature:
    """Construct the sole v4 trapezoid--sine-squared edge quadrature.

    The trapezoid is formed once on the sorted base-valid midpoint list.  Its
    full-length weights can then be subset for a coordinate without changing
    either neighbor spacing or endpoint ownership.
    """

    midpoints = edge_midpoints(clocks)
    edge_count = int(midpoints.size)
    valid = _as_binary_mask(edge_mask, length=edge_count, name="edge_mask")
    stop = edge_count if edge_stop is None else edge_stop
    if not (0 <= edge_start < stop <= edge_count):
        raise QuadratureError("quadrature edge interval must be nonempty and in range")
    selected = np.flatnonzero(
        valid & (np.arange(edge_count) >= edge_start) & (np.arange(edge_count) < stop)
    )
    if selected.size < 3:
        raise QuadratureError("shared quadrature requires at least three midpoints")

    x = midpoints[selected]
    spacing = np.diff(x)
    if not np.all(np.isfinite(x)) or not np.all(spacing > 0.0):
        raise QuadratureError("shared quadrature midpoints must be finite and increasing")
    span = float(x[-1] - x[0])
    if not np.isfinite(span) or span <= 0.0:
        raise QuadratureError("shared quadrature requires positive midpoint span")

    trapezoid = np.empty(selected.size, dtype=np.float64)
    trapezoid[0] = spacing[0] / np.float64(2.0)
    trapezoid[-1] = spacing[-1] / np.float64(2.0)
    if selected.size > 2:
        trapezoid[1:-1] = (x[2:] - x[:-2]) / np.float64(2.0)
    phase = np.pi * (x - x[0]) / np.float64(span)
    taper = np.sin(phase) ** np.float64(2.0)
    # The mathematical endpoint weights are exactly zero.  Canonicalize them
    # rather than retaining libm's small nonzero sin(pi) residual.
    taper[0] = np.float64(0.0)
    taper[-1] = np.float64(0.0)
    selected_weights = trapezoid * taper
    total_weight = neumaier_sum(selected_weights)
    if (
        not np.all(np.isfinite(selected_weights))
        or not np.all(selected_weights >= 0.0)
        or not np.isfinite(total_weight)
        or total_weight <= 0.0
    ):
        raise QuadratureError("shared quadrature has a zero or nonfinite weight sum")

    weights = np.zeros(edge_count, dtype=np.float64)
    weights[selected] = selected_weights
    return EdgeQuadrature(
        midpoints=np.ascontiguousarray(midpoints),
        weights=np.ascontiguousarray(weights),
        edge_indices=np.ascontiguousarray(selected, dtype=np.int64),
        span=span,
    )


def arc_static_code_from_subedges(
    values: ArrayLike,
    edge_lengths: ArrayLike,
    traversals: Sequence[Sequence[Subedge]],
) -> Float64Array:
    """Evaluate F5 on explicit, ordered traversal subedges.

    ``values`` are the continuous endpoint channels (149 for the canonical
    teacher).  Fractional endpoints are obtained by exact linear evaluation,
    and fractional length is the matching fraction of the base edge length.
    """

    y = np.asarray(values, dtype=np.float64)
    lengths = np.asarray(edge_lengths, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] < 2:
        raise QuadratureError("values must have shape [samples, channels]")
    if lengths.shape != (y.shape[0] - 1,):
        raise QuadratureError("edge_lengths must have shape [samples-1]")
    if not np.all(np.isfinite(y)):
        raise QuadratureError("arc values must be finite")
    if not np.all(np.isfinite(lengths)) or not np.all(lengths > 0.0):
        raise QuadratureError("arc edge lengths must be finite and strictly positive")
    if not traversals:
        raise QuadratureError("at least one traversal is required")

    traversal_means: list[Float64Array] = []
    traversal_seconds: list[Float64Array] = []
    channel_count = y.shape[1]
    for traversal in traversals:
        if not traversal:
            raise QuadratureError("every traversal denominator must be positive")
        denominator_terms: list[float] = []
        mean_terms: list[Float64Array] = []
        second_terms: list[Float64Array] = []
        for subedge in traversal:
            if subedge.edge >= lengths.size:
                raise QuadratureError("subedge index is outside edge_lengths")
            fraction = np.float64(subedge.stop - subedge.start)
            length = np.float64(lengths[subedge.edge] * fraction)
            left = y[subedge.edge]
            delta = y[subedge.edge + 1] - left
            a = left + np.float64(subedge.start) * delta
            b = left + np.float64(subedge.stop) * delta
            denominator_terms.append(float(length))
            mean_terms.append(length * (a + b) / np.float64(2.0))
            second_terms.append(length * (a * a + a * b + b * b) / np.float64(3.0))
        denominator = neumaier_sum(denominator_terms)
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise QuadratureError("every traversal denominator must be positive")
        mean = neumaier_sum_array(np.stack(mean_terms, axis=0)) / np.float64(denominator)
        second = neumaier_sum_array(np.stack(second_terms, axis=0)) / np.float64(denominator)
        if mean.shape != (channel_count,) or second.shape != (channel_count,):
            raise AssertionError("internal static-code shape error")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(second)):
            raise QuadratureError("arc moments must be finite")
        traversal_means.append(mean)
        traversal_seconds.append(second)

    count = np.float64(len(traversal_means))
    mean = neumaier_sum_array(np.stack(traversal_means, axis=0)) / count
    second = neumaier_sum_array(np.stack(traversal_seconds, axis=0)) / count
    variance = np.maximum(second - mean * mean, np.float64(0.0))
    scale = np.sqrt(variance)
    result = np.concatenate((mean, scale)).astype(np.float64, copy=False)
    if not np.all(np.isfinite(result)):
        raise QuadratureError("static code must be finite")
    return np.ascontiguousarray(result)


def arc_static_code(
    values: ArrayLike,
    edge_lengths: ArrayLike,
    traversal_bounds: ArrayLike,
) -> Float64Array:
    """Evaluate F5 for integer half-open traversal edge bounds."""

    lengths = np.asarray(edge_lengths)
    bounds = np.asarray(traversal_bounds)
    if bounds.ndim != 2 or bounds.shape[1:] != (2,) or bounds.dtype.kind not in "iu":
        raise QuadratureError("traversal_bounds must be an integer [K,2] array")
    traversals: list[tuple[Subedge, ...]] = []
    previous_stop = -1
    for raw_start, raw_stop in bounds.tolist():
        start = int(raw_start)
        stop = int(raw_stop)
        if not (0 <= start < stop <= lengths.size):
            raise QuadratureError("traversal bounds must be nonempty and in range")
        if start < previous_stop:
            raise QuadratureError("traversal bounds must follow edge order")
        previous_stop = stop
        traversals.append(tuple(Subedge(edge) for edge in range(start, stop)))
    return arc_static_code_from_subedges(values, edge_lengths, traversals)


__all__ = [
    "EdgeQuadrature",
    "QuadratureError",
    "Subedge",
    "arc_static_code",
    "arc_static_code_from_subedges",
    "edge_midpoints",
    "neumaier_sum",
    "neumaier_sum_array",
    "shared_edge_quadrature",
]
