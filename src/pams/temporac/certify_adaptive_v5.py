"""Stable-pair selector with train-frozen structural prefilters."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pams.temporac.certify_adaptive_v3 import periodic_integer_landmarks
from pams.temporac.certify_adaptive_v4 import (
    StablePairLandmarkResult,
    _pair_huber,
)

MAX_PAIR_GEOMETRY_HUBER = 0.025


def _structurally_valid(
    geometry: NDArray[np.float64],
    points: NDArray[np.int32],
) -> bool:
    starts = geometry[points[:-1]]
    median = np.median(starts, axis=0)
    if np.any(np.linalg.norm(starts - median, axis=1) / np.sqrt(66.0) > 0.05):
        return False
    for left_raw, right_raw in zip(points[:-1], points[1:], strict=True):
        left, right = int(left_raw), int(right_raw)
        arc = np.linalg.norm(
            geometry[left + 1 : right + 1] - geometry[left:right],
            axis=1,
        )
        if (
            arc.size < 6
            or np.count_nonzero(arc > 0.0) < 5
            or float(np.max(arc) / np.sum(arc)) >= 0.25
        ):
            return False
    return True


def stable_pair_integer_landmarks_v5(
    geometry: NDArray[np.float64],
    run_bounds: NDArray[np.int32],
) -> StablePairLandmarkResult:
    """Choose the best structurally valid pair under the train-frozen threshold."""

    base = periodic_integer_landmarks(geometry, run_bounds)
    if base.landmarks.size < 3:
        return StablePairLandmarkResult(
            base.landmarks,
            base.estimated_period,
            base.lag_score,
            None,
        )
    candidates = [
        base.landmarks[index : index + 3]
        for index in range(base.landmarks.size - 2)
        if _structurally_valid(geometry, base.landmarks[index : index + 3])
    ]
    if not candidates:
        selected = np.empty(0, dtype=np.int32)
        selected.flags.writeable = False
        return StablePairLandmarkResult(
            selected,
            base.estimated_period,
            base.lag_score,
            None,
        )
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


__all__ = ["stable_pair_integer_landmarks_v5"]
