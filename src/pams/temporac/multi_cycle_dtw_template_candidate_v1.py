"""Complete-chain periodic template aligned to a geometry medoid with DTW."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pams.temporac.dtw_phase_candidate_v1 import dtw_phase_coordinates


def _arc_resample(values: NDArray[np.float64]) -> NDArray[np.float64]:
    edge = np.linalg.norm(np.diff(values, axis=0), axis=1)
    coordinate = np.concatenate(([0.0], np.cumsum(edge, dtype=np.float64)))
    coordinate /= coordinate[-1]
    grid = np.linspace(0.0, 1.0, 129, dtype=np.float64)
    return np.column_stack(
        [np.interp(grid, coordinate, values[:, channel]) for channel in range(values.shape[1])]
    )


def multi_cycle_dtw_reconstruction(
    target: NDArray[np.float64],
    mask: NDArray[np.float64],
    geometry: NDArray[np.float64],
    landmarks: NDArray[np.int32],
) -> tuple[NDArray[np.float64], tuple[NDArray[np.float64], ...]]:
    """Return a shared full-chain template and per-traversal monotone coordinates."""

    bounds = [
        (int(left), int(right))
        for left, right in zip(landmarks[:-1], landmarks[1:], strict=True)
    ]
    traversals = [geometry[left : right + 1] for left, right in bounds]
    resampled = np.stack([_arc_resample(values) for values in traversals])
    pairwise = np.linalg.norm(
        resampled[:, None] - resampled[None, :],
        axis=3,
    ).mean(axis=2) / np.sqrt(geometry.shape[1])
    medoid = min(
        range(len(traversals)),
        key=lambda index: (float(np.mean(pairwise[index])), index),
    )
    reference = traversals[medoid]
    reference_coordinates: list[NDArray[np.float64]] = []
    coordinates: list[NDArray[np.float64] | None] = [None] * len(traversals)
    for index, traversal in enumerate(traversals):
        if index == medoid:
            continue
        reference_coordinate, traversal_coordinate = dtw_phase_coordinates(
            reference,
            traversal,
        )
        reference_coordinates.append(reference_coordinate)
        coordinates[index] = traversal_coordinate
    if reference_coordinates:
        medoid_coordinate = np.mean(np.stack(reference_coordinates), axis=0)
    else:
        medoid_coordinate = np.linspace(0.0, 1.0, reference.shape[0], dtype=np.float64)
    medoid_coordinate[0], medoid_coordinate[-1] = 0.0, 1.0
    coordinates[medoid] = medoid_coordinate
    typed_coordinates = tuple(
        np.asarray(value, dtype=np.float64) for value in coordinates if value is not None
    )
    grid = np.linspace(0.0, 1.0, 513, dtype=np.float64)
    numerator = np.zeros((grid.size, target.shape[1]), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    for (left, right), coordinate in zip(bounds, typed_coordinates, strict=True):
        for channel in range(target.shape[1]):
            valid = mask[left : right + 1, channel] > 0
            if np.count_nonzero(valid) < 2:
                continue
            numerator[:, channel] += np.interp(
                grid,
                coordinate[valid],
                target[left : right + 1, channel][valid],
            )
            denominator[:, channel] += 1.0
    template = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    template[-1] = template[0] = 0.5 * (template[0] + template[-1])
    reconstruction = np.array(target, dtype=np.float64, copy=True)
    for (left, right), coordinate in zip(bounds, typed_coordinates, strict=True):
        for channel in range(target.shape[1]):
            reconstruction[left : right + 1, channel] = np.interp(
                coordinate,
                grid,
                template[:, channel],
            )
    return reconstruction, typed_coordinates


def phase_from_multi_cycle_coordinates(
    length: int,
    landmarks: NDArray[np.int32],
    coordinates: tuple[NDArray[np.float64], ...],
) -> NDArray[np.float64]:
    """Build one positive unit winding for every aligned complete-chain cycle."""

    if len(coordinates) != landmarks.size - 1:
        raise ValueError("coordinate count differs from landmark traversals")
    phase = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float64), (length, 1))
    for left_raw, right_raw, coordinate in zip(
        landmarks[:-1],
        landmarks[1:],
        coordinates,
        strict=True,
    ):
        left, right = int(left_raw), int(right_raw)
        increment = np.diff(coordinate)
        if (
            coordinate.shape != (right - left + 1,)
            or np.any(increment < 0.0)
            or np.any(increment >= 0.25)
            or np.count_nonzero(increment > 0.0) < 5
        ):
            raise ValueError("DTW coordinate violates phase topology")
        angle = 2.0 * np.pi * coordinate
        phase[left : right + 1, 0] = np.cos(angle)
        phase[left : right + 1, 1] = np.sin(angle)
    phase[landmarks, 0] = 1.0
    phase[landmarks, 1] = 0.0
    return np.asarray(phase, dtype="<f8")


__all__ = ["multi_cycle_dtw_reconstruction", "phase_from_multi_cycle_coordinates"]
