"""Count-blind full-period landmark candidate based on global lag agreement."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pams.temporac.certify_adaptive_v2 import AdaptiveLandmarkError

MIN_PERIOD = 6
MAX_PERIOD = 160
MAX_RETURN_RMS = 0.075


@dataclass(frozen=True, slots=True)
class PeriodicLandmarkResult:
    landmarks: NDArray[np.int32]
    estimated_period: int | None
    lag_score: float | None
    maximum_return_rms: float | None
    interval_cv: float | None


def _lag_scores(run: NDArray[np.float64]) -> NDArray[np.float64]:
    maximum = min(MAX_PERIOD, (run.shape[0] - 1) // 2)
    scores = np.full(maximum + 1, np.inf, dtype=np.float64)
    for lag in range(MIN_PERIOD, maximum + 1):
        difference = run[lag:] - run[:-lag]
        per_row = np.linalg.norm(difference, axis=1) / np.sqrt(run.shape[1])
        scores[lag] = float(np.median(per_row))
    return scores


def _candidate_periods(scores: NDArray[np.float64]) -> list[int]:
    candidates = [
        lag
        for lag in range(MIN_PERIOD, scores.size)
        if scores[lag] <= scores[lag - 1]
        and (lag + 1 == scores.size or scores[lag] < scores[lag + 1])
    ]
    if not candidates:
        return []
    best = min(float(scores[lag]) for lag in candidates)
    return [lag for lag in candidates if float(scores[lag]) <= 1.25 * best + 1e-12]


def _return_chain(
    run: NDArray[np.float64],
    period: int,
) -> tuple[list[int], float, float] | None:
    best: tuple[tuple[float, float, int], list[int], float, float] | None = None
    radius = max(2, int(round(0.20 * period)))
    for reference in range(min(period, run.shape[0])):
        points = [reference]
        errors = [0.0]
        expected = reference + period
        while expected < run.shape[0]:
            left = max(points[-1] + max(3, int(round(0.55 * period))), expected - radius)
            right = min(run.shape[0], expected + radius + 1)
            if left >= right:
                break
            local = run[left:right] - run[reference]
            distance = np.linalg.norm(local, axis=1) / np.sqrt(run.shape[1])
            offset = int(np.argmin(distance))
            error = float(distance[offset])
            if error > MAX_RETURN_RMS:
                break
            point = left + offset
            points.append(point)
            errors.append(error)
            expected = point + period
        if len(points) < 3:
            continue
        intervals = np.diff(np.asarray(points, dtype=np.float64))
        interval_cv = float(np.std(intervals) / np.mean(intervals))
        maximum = max(errors)
        key = (-float(len(points)), maximum, reference)
        if best is None or key < best[0]:
            best = (key, points, maximum, interval_cv)
    if best is None:
        return None
    return best[1], best[2], best[3]


def periodic_integer_landmarks(
    geometry: NDArray[np.float64],
    run_bounds: NDArray[np.int32],
) -> PeriodicLandmarkResult:
    """Select stable full-period returns using global geometric autocorrelation."""

    values = np.asarray(geometry, dtype=np.float64)
    bounds = np.asarray(run_bounds, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != 66 or not np.isfinite(values).all():
        raise AdaptiveLandmarkError("geometry must be finite [T,66]")
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] != 1:
        raise AdaptiveLandmarkError("periodic candidate requires one [1,2] run")
    start, stop = (int(value) for value in bounds[0])
    if not 0 <= start < stop <= values.shape[0]:
        raise AdaptiveLandmarkError("run bound is out of range")
    run = values[start:stop]
    scores = _lag_scores(run)
    candidates = _candidate_periods(scores)
    best: tuple[tuple[float, float, float, int], list[int], int, float, float] | None = None
    for period in candidates:
        chain = _return_chain(run, period)
        if chain is None:
            continue
        points, maximum, interval_cv = chain
        key = (float(scores[period]), interval_cv, maximum, period)
        if best is None or key < best[0]:
            best = (key, points, period, maximum, interval_cv)
    if best is None:
        empty = np.empty(0, dtype=np.int32)
        empty.flags.writeable = False
        return PeriodicLandmarkResult(empty, None, None, None, None)
    key, points, period, maximum, interval_cv = best
    landmarks = np.asarray([start + point for point in points], dtype=np.int32)
    landmarks.flags.writeable = False
    return PeriodicLandmarkResult(
        landmarks=landmarks,
        estimated_period=period,
        lag_score=key[0],
        maximum_return_rms=maximum,
        interval_cv=interval_cv,
    )


__all__ = ["PeriodicLandmarkResult", "periodic_integer_landmarks"]
