"""Count-blind monotone phase coordinates from pairwise geometry DTW."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def dtw_phase_coordinates(
    first: NDArray[np.float64],
    second: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Align two finite geometry traversals and return monotone unit coordinates."""

    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != right.shape[1]
        or min(left.shape[0], right.shape[0]) < 7
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
    ):
        raise ValueError("DTW traversals must be finite [T,D] with T>=7")
    difference = left[:, None, :] - right[None, :, :]
    cost = np.linalg.norm(difference, axis=2) / np.sqrt(left.shape[1])
    accumulated = np.full(cost.shape, np.inf, dtype=np.float64)
    back = np.full(cost.shape, -1, dtype=np.int8)
    accumulated[0, 0] = cost[0, 0]
    for i in range(left.shape[0]):
        for j in range(right.shape[0]):
            if i == 0 and j == 0:
                continue
            predecessors: list[tuple[float, int]] = []
            if i > 0 and j > 0:
                predecessors.append((float(accumulated[i - 1, j - 1]), 0))
            if i > 0:
                predecessors.append((float(accumulated[i - 1, j]), 1))
            if j > 0:
                predecessors.append((float(accumulated[i, j - 1]), 2))
            value, direction = min(predecessors, key=lambda row: (row[0], row[1]))
            accumulated[i, j] = cost[i, j] + value
            back[i, j] = direction
    path: list[tuple[int, int]] = []
    i, j = left.shape[0] - 1, right.shape[0] - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        direction = int(back[i, j])
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        elif direction == 2:
            j -= 1
        else:
            raise RuntimeError("DTW backtrace is incomplete")
    path.reverse()
    coordinate = np.linspace(0.0, 1.0, len(path), dtype=np.float64)
    first_sum = np.zeros(left.shape[0], dtype=np.float64)
    first_count = np.zeros(left.shape[0], dtype=np.float64)
    second_sum = np.zeros(right.shape[0], dtype=np.float64)
    second_count = np.zeros(right.shape[0], dtype=np.float64)
    for value, (left_index, right_index) in zip(coordinate, path, strict=True):
        first_sum[left_index] += value
        first_count[left_index] += 1.0
        second_sum[right_index] += value
        second_count[right_index] += 1.0
    first_coordinate = first_sum / first_count
    second_coordinate = second_sum / second_count
    first_coordinate[0], first_coordinate[-1] = 0.0, 1.0
    second_coordinate[0], second_coordinate[-1] = 0.0, 1.0
    return first_coordinate, second_coordinate


def dtw_periodic_reconstruction(
    target: NDArray[np.float64],
    mask: NDArray[np.float64],
    landmarks: NDArray[np.int32],
    first_coordinate: NDArray[np.float64],
    second_coordinate: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Average two DTW-aligned traversals and reconstruct both on a common grid."""

    left, middle, right = (int(value) for value in landmarks)
    coordinates = (first_coordinate, second_coordinate)
    bounds = ((left, middle), (middle, right))
    grid = np.linspace(0.0, 1.0, 513, dtype=np.float64)
    numerator = np.zeros((grid.size, target.shape[1]), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    for (start, stop), coordinate in zip(bounds, coordinates, strict=True):
        for channel in range(target.shape[1]):
            valid = mask[start : stop + 1, channel] > 0
            if np.count_nonzero(valid) < 2:
                continue
            numerator[:, channel] += np.interp(
                grid,
                coordinate[valid],
                target[start : stop + 1, channel][valid],
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
    for (start, stop), coordinate in zip(bounds, coordinates, strict=True):
        for channel in range(target.shape[1]):
            reconstruction[start : stop + 1, channel] = np.interp(
                coordinate,
                grid,
                template[:, channel],
            )
    return reconstruction


__all__ = ["dtw_periodic_reconstruction", "dtw_phase_coordinates"]
