"""Robust complete-chain DTW template using a coordinate-wise cycle median."""

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


def multi_cycle_dtw_median_reconstruction(
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
    aligned_cycles: list[NDArray[np.float64]] = []
    for (left, right), coordinate in zip(bounds, typed_coordinates, strict=True):
        aligned = np.full((grid.size, target.shape[1]), np.nan, dtype=np.float64)
        for channel in range(target.shape[1]):
            valid = mask[left : right + 1, channel] > 0
            if np.count_nonzero(valid) < 2:
                continue
            aligned[:, channel] = np.interp(
                grid,
                coordinate[valid],
                target[left : right + 1, channel][valid],
            )
        aligned_cycles.append(aligned)
    template = np.nanmedian(
        np.stack(aligned_cycles),
        axis=0,
    )
    np.nan_to_num(template, copy=False)
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


__all__ = ["multi_cycle_dtw_median_reconstruction"]
