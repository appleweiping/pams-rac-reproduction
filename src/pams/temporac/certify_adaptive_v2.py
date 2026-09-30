"""Additive adaptive-landmark candidate for natural TempoRAC tracks.

The frozen :mod:`pams.temporac.certify` module is not modified.  This module
only proposes a count-blind replacement for its natural-data landmark stage;
all later phase, topology, reconstruction, collision, and pulse checks remain
outside this first-stage candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pams.temporac.certify import integer_landmarks as frozen_integer_landmarks

POSE_RETURN_RMS = 0.05
MIN_EXCURSION_RMS = 0.05
MIN_RETURN_SEPARATION = 4


class AdaptiveLandmarkError(ValueError):
    """Raised when the adaptive-landmark input is malformed."""


@dataclass(frozen=True, slots=True)
class AdaptiveLandmarkResult:
    landmarks: NDArray[np.int32]
    reference_index: int | None
    maximum_return_rms: float | None
    minimum_excursion_rms: float | None


def _local_minima(distance: NDArray[np.float64], threshold: float) -> list[int]:
    candidates: list[int] = []
    for index in range(distance.size):
        left = distance[index - 1] if index > 0 else float("inf")
        right = distance[index + 1] if index + 1 < distance.size else float("inf")
        if distance[index] <= threshold and distance[index] <= left and distance[index] < right:
            candidates.append(index)
    return candidates


def _separated(candidates: list[int], distance: NDArray[np.float64]) -> list[int]:
    if not candidates:
        return []
    groups: list[list[int]] = [[candidates[0]]]
    for candidate in candidates[1:]:
        if candidate - groups[-1][-1] < MIN_RETURN_SEPARATION:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    return [min(group, key=lambda index: (float(distance[index]), index)) for group in groups]


def _excursion_filtered(
    candidates: list[int],
    distance: NDArray[np.float64],
) -> tuple[list[int], list[float]]:
    if len(candidates) < 2:
        return candidates, []
    retained = [candidates[0]]
    excursions: list[float] = []
    for candidate in candidates[1:]:
        left = retained[-1]
        if candidate <= left:
            continue
        excursion = float(np.max(distance[left : candidate + 1]))
        if excursion < MIN_EXCURSION_RMS:
            continue
        retained.append(candidate)
        excursions.append(excursion)
    return retained, excursions


def _pairwise_rms(run: NDArray[np.float64]) -> NDArray[np.float64]:
    squared_norm = np.einsum("ij,ij->i", run, run)
    squared = squared_norm[:, None] + squared_norm[None, :] - 2.0 * (run @ run.T)
    np.maximum(squared, 0.0, out=squared)
    np.sqrt(squared / float(run.shape[1]), out=squared)
    return squared


def adaptive_integer_landmarks(
    geometry: NDArray[np.float64],
    run_bounds: NDArray[np.int32],
) -> AdaptiveLandmarkResult:
    """Find count-blind repeated returns to one data-selected pose per run.

    Selection maximizes the number of separated, excursion-backed returns.
    Ties prefer the smallest maximum return error, then the smallest interval
    coefficient of variation, and finally the earliest reference index.
    """

    values = np.asarray(geometry, dtype=np.float64)
    bounds = np.asarray(run_bounds, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != 66 or not np.isfinite(values).all():
        raise AdaptiveLandmarkError("geometry must be finite [T,66]")
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] < 1:
        raise AdaptiveLandmarkError("run_bounds must be [R,2]")
    output: list[int] = []
    chosen_reference: int | None = None
    chosen_maximum: float | None = None
    chosen_excursion: float | None = None
    for start_raw, stop_raw in bounds:
        start, stop = int(start_raw), int(stop_raw)
        if not 0 <= start < stop <= values.shape[0]:
            raise AdaptiveLandmarkError("run bound is out of range")
        run = values[start:stop]
        pairwise_rms = _pairwise_rms(run)
        best: tuple[
            tuple[float, float, float, int],
            list[int],
            list[float],
            NDArray[np.float64],
        ] | None = None
        for reference_index in range(run.shape[0]):
            distance = pairwise_rms[reference_index]
            candidates = _separated(_local_minima(distance, POSE_RETURN_RMS), distance)
            candidates, excursions = _excursion_filtered(candidates, distance)
            if len(candidates) < 3:
                continue
            intervals = np.diff(np.asarray(candidates, dtype=np.float64))
            interval_cv = float(np.std(intervals) / np.mean(intervals)) if intervals.size else 0.0
            maximum = float(np.max(distance[candidates]))
            key = (-float(len(candidates)), maximum, interval_cv, reference_index)
            if best is None or key < best[0]:
                best = (key, candidates, excursions, distance)
        if best is None:
            continue
        _key, candidates, excursions, distance = best
        output.extend(start + index for index in candidates)
        reference_index = candidates[int(np.argmin(distance[candidates]))]
        chosen_reference = start + reference_index
        chosen_maximum = float(np.max(distance[candidates]))
        chosen_excursion = min(excursions) if excursions else None
    landmarks = np.asarray(output, dtype=np.int32)
    landmarks.flags.writeable = False
    return AdaptiveLandmarkResult(
        landmarks=landmarks,
        reference_index=chosen_reference,
        maximum_return_rms=chosen_maximum,
        minimum_excursion_rms=chosen_excursion,
    )


def hybrid_integer_landmarks(
    geometry: NDArray[np.float64],
    run_bounds: NDArray[np.int32],
) -> AdaptiveLandmarkResult:
    """Preserve a valid frozen result and use adaptive returns only as fallback."""

    frozen = frozen_integer_landmarks(geometry, run_bounds)
    if frozen.size >= 3:
        return AdaptiveLandmarkResult(
            landmarks=frozen,
            reference_index=None,
            maximum_return_rms=None,
            minimum_excursion_rms=None,
        )
    return adaptive_integer_landmarks(geometry, run_bounds)


__all__ = [
    "AdaptiveLandmarkError",
    "AdaptiveLandmarkResult",
    "adaptive_integer_landmarks",
    "hybrid_integer_landmarks",
]
