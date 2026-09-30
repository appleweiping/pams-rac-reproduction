"""Complete-chain landmarks from every return-valid local lag minimum."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pams.temporac.certify_adaptive_v3 import (
    MAX_RETURN_RMS,
    MIN_PERIOD,
    _lag_scores,
    _return_chain,
)
from pams.temporac.full_chain_landmarks_candidate_v1 import (
    FullChainLandmarkResult,
    _maximum_legal_subsequence,
    _origin_subset,
)


def _all_return_valid_landmarks(
    geometry: NDArray[np.float64],
    run_bounds: NDArray[np.int32],
) -> NDArray[np.int32]:
    values = np.asarray(geometry, dtype=np.float64)
    bounds = np.asarray(run_bounds, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != 66 or not np.isfinite(values).all():
        raise ValueError("geometry must be finite [T,66]")
    if bounds.shape != (1, 2):
        raise ValueError("periodic candidate requires one [1,2] run")
    start, stop = (int(value) for value in bounds[0])
    if not 0 <= start < stop <= values.shape[0]:
        raise ValueError("run bound is out of range")
    run = values[start:stop]
    scores = _lag_scores(run)
    candidates: list[tuple[tuple[float, float, float, float, int], list[int]]] = []
    for period in range(MIN_PERIOD, scores.size):
        score = float(scores[period])
        if not (
            score <= MAX_RETURN_RMS
            and score <= float(scores[period - 1])
            and (period + 1 == scores.size or score < float(scores[period + 1]))
        ):
            continue
        chain = _return_chain(run, period)
        if chain is None:
            continue
        points, maximum, interval_cv = chain
        key = (-float(len(points)), score, interval_cv, maximum, period)
        candidates.append((key, points))
    if not candidates:
        empty = np.empty(0, dtype=np.int32)
        empty.flags.writeable = False
        return empty
    _key, points = min(candidates, key=lambda row: row[0])
    landmarks = np.asarray([start + point for point in points], dtype=np.int32)
    landmarks.flags.writeable = False
    return landmarks


def full_chain_integer_landmarks_v2(
    geometry: NDArray[np.float64],
    run_bounds: NDArray[np.int32],
) -> FullChainLandmarkResult:
    """Return the longest legal chain across all return-valid local lag minima."""

    base = _all_return_valid_landmarks(geometry, run_bounds)
    original_events = max(0, int(base.size) - 1)
    origin = _origin_subset(geometry, base) if base.size >= 3 else base
    selected = _maximum_legal_subsequence(geometry, origin)
    selected.flags.writeable = False
    retained_events = max(0, int(selected.size) - 1)
    fraction = retained_events / original_events if original_events else 0.0
    return FullChainLandmarkResult(
        landmarks=selected,
        base_landmarks=int(base.size),
        retained_event_fraction=fraction,
    )


__all__ = ["full_chain_integer_landmarks_v2"]
