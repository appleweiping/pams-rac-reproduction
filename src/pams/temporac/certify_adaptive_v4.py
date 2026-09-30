"""Select the most geometrically consistent adjacent natural-cycle pair."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pams.temporac.certify_adaptive_v3 import periodic_integer_landmarks

MAX_PAIR_GEOMETRY_HUBER = 0.02


@dataclass(frozen=True, slots=True)
class StablePairLandmarkResult:
    landmarks: NDArray[np.int32]
    estimated_period: int | None
    lag_score: float | None
    pair_geometry_huber: float | None


def _arc_resample(
    geometry: NDArray[np.float64],
    left: int,
    right: int,
) -> NDArray[np.float64]:
    values = geometry[left : right + 1]
    edge = np.linalg.norm(np.diff(values, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(edge, dtype=np.float64)))
    if cumulative[-1] <= 0.0:
        return np.repeat(values[:1], 129, axis=0)
    cumulative /= cumulative[-1]
    grid = np.linspace(0.0, 1.0, 129, dtype=np.float64)
    return np.column_stack(
        [np.interp(grid, cumulative, values[:, channel]) for channel in range(66)]
    )


def _pair_huber(
    geometry: NDArray[np.float64],
    points: NDArray[np.int32],
) -> float:
    first = _arc_resample(geometry, int(points[0]), int(points[1]))
    second = _arc_resample(geometry, int(points[1]), int(points[2]))
    error = np.abs(first[1:-1] - second[1:-1])
    huber = np.where(error <= 0.05, 0.5 * np.square(error) / 0.05, error - 0.025)
    return float(np.mean(huber))


def stable_pair_integer_landmarks(
    geometry: NDArray[np.float64],
    run_bounds: NDArray[np.int32],
) -> StablePairLandmarkResult:
    """Return one count-blind adjacent pair or abstain on poor geometry agreement."""

    base = periodic_integer_landmarks(geometry, run_bounds)
    if base.landmarks.size < 3:
        return StablePairLandmarkResult(
            base.landmarks,
            base.estimated_period,
            base.lag_score,
            None,
        )
    candidates = [base.landmarks[index : index + 3] for index in range(base.landmarks.size - 2)]
    scored = [(_pair_huber(geometry, points), points) for points in candidates]
    score, selected = min(scored, key=lambda row: (row[0], int(row[1][0])))
    if score > MAX_PAIR_GEOMETRY_HUBER:
        selected = np.empty(0, dtype=np.int32)
    else:
        selected = np.array(selected, dtype=np.int32, copy=True)
    selected.flags.writeable = False
    return StablePairLandmarkResult(
        landmarks=selected,
        estimated_period=base.estimated_period,
        lag_score=base.lag_score,
        pair_geometry_huber=score,
    )


__all__ = ["StablePairLandmarkResult", "stable_pair_integer_landmarks"]
