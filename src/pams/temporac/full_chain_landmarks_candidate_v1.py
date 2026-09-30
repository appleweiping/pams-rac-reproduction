"""Count-blind maximum legal subsequence of natural periodic landmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pams.temporac.certify_adaptive_v3 import periodic_integer_landmarks


@dataclass(frozen=True, slots=True)
class FullChainLandmarkResult:
    landmarks: NDArray[np.int32]
    base_landmarks: int
    retained_event_fraction: float


def _origin_subset(
    geometry: NDArray[np.float64],
    points: NDArray[np.int32],
) -> NDArray[np.int32]:
    values = geometry[points]
    distance = np.linalg.norm(values[:, None] - values[None, :], axis=2) / np.sqrt(66.0)
    reference = min(
        range(points.size),
        key=lambda index: (
            -int(np.count_nonzero(distance[index] <= 0.05)),
            float(np.max(distance[index][distance[index] <= 0.05])),
            int(points[index]),
        ),
    )
    selected = np.array(points[distance[reference] <= 0.05], dtype=np.int32, copy=True)
    while selected.size >= 3:
        starts = geometry[selected[:-1]]
        median = np.median(starts, axis=0)
        rms = np.linalg.norm(starts - median, axis=1) / np.sqrt(66.0)
        if np.all(rms <= 0.05):
            break
        remove = int(np.argmax(rms))
        selected = np.delete(selected, remove)
    return selected


def _valid_edge(geometry: NDArray[np.float64], left: int, right: int) -> bool:
    arc = np.linalg.norm(geometry[left + 1 : right + 1] - geometry[left:right], axis=1)
    return bool(
        arc.size >= 6
        and np.count_nonzero(arc > 0.0) >= 5
        and float(np.max(arc) / np.sum(arc)) < 0.25
    )


def _maximum_legal_subsequence(
    geometry: NDArray[np.float64],
    points: NDArray[np.int32],
) -> NDArray[np.int32]:
    if points.size < 3:
        return np.empty(0, dtype=np.int32)
    length = np.ones(points.size, dtype=np.int32)
    span = np.zeros(points.size, dtype=np.int32)
    predecessor = np.full(points.size, -1, dtype=np.int32)
    for right_index in range(points.size):
        for left_index in range(right_index):
            if not _valid_edge(
                geometry,
                int(points[left_index]),
                int(points[right_index]),
            ):
                continue
            candidate_length = int(length[left_index]) + 1
            candidate_span = int(points[right_index] - points[0])
            if candidate_length > int(length[right_index]) or (
                candidate_length == int(length[right_index])
                and candidate_span > int(span[right_index])
            ):
                length[right_index] = candidate_length
                span[right_index] = candidate_span
                predecessor[right_index] = left_index
    endpoint = max(
        range(points.size),
        key=lambda index: (int(length[index]), int(span[index]), -int(points[index])),
    )
    if int(length[endpoint]) < 3:
        return np.empty(0, dtype=np.int32)
    indices: list[int] = []
    current = endpoint
    while current >= 0:
        indices.append(current)
        current = int(predecessor[current])
    indices.reverse()
    return np.asarray(points[indices], dtype=np.int32)


def full_chain_integer_landmarks(
    geometry: NDArray[np.float64],
    run_bounds: NDArray[np.int32],
) -> FullChainLandmarkResult:
    """Retain the maximum count-blind legal subsequence of full-period returns."""

    base = periodic_integer_landmarks(geometry, run_bounds).landmarks
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


__all__ = ["FullChainLandmarkResult", "full_chain_integer_landmarks"]
