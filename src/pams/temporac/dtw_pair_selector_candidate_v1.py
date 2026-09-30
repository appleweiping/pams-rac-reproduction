"""Label-free stable-pair selection using DTW-aligned source consistency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pams.temporac.certify_adaptive_v3 import periodic_integer_landmarks
from pams.temporac.certify_adaptive_v5 import _structurally_valid
from pams.temporac.dtw_phase_candidate_v1 import (
    dtw_periodic_reconstruction,
    dtw_phase_coordinates,
)

MAX_DTW_RECONSTRUCTION_HUBER = 0.018


@dataclass(frozen=True, slots=True)
class DTWPairResult:
    landmarks: NDArray[np.int32]
    first_coordinate: NDArray[np.float64]
    second_coordinate: NDArray[np.float64]
    reconstruction: NDArray[np.float64]
    maximum_huber: float | None


def _maximum_huber(
    target: NDArray[np.float64],
    reconstruction: NDArray[np.float64],
    mask: NDArray[np.float64],
    landmarks: NDArray[np.int32],
) -> float:
    values: list[float] = []
    for left_raw, right_raw in zip(landmarks[:-1], landmarks[1:], strict=True):
        left, right = int(left_raw), int(right_raw)
        error = np.abs(reconstruction[left + 1 : right] - target[left + 1 : right])
        local_mask = mask[left + 1 : right]
        huber = np.where(error <= 0.05, 0.5 * np.square(error) / 0.05, error - 0.025)
        values.append(float(np.sum(huber * local_mask) / np.sum(local_mask)))
    return max(values)


def _phase_valid(coordinate: NDArray[np.float64]) -> bool:
    increment = np.diff(coordinate)
    return bool(
        np.all(increment >= 0.0)
        and np.all(increment < 0.25)
        and np.count_nonzero(increment > 0.0) >= 5
    )


def select_dtw_pair(
    geometry: NDArray[np.float64],
    continuous: NDArray[np.float64],
    continuous_mask: NDArray[np.float64],
    run_bounds: NDArray[np.int32],
) -> DTWPairResult:
    """Select the best structurally valid DTW pair under a train-frozen margin."""

    base = periodic_integer_landmarks(geometry, run_bounds)
    empty = np.empty(0, dtype=np.float64)
    empty_landmarks = np.empty(0, dtype=np.int32)
    if base.landmarks.size < 3:
        return DTWPairResult(
            empty_landmarks,
            empty,
            empty,
            np.array(continuous, dtype=np.float64, copy=True),
            None,
        )
    candidates: list[DTWPairResult] = []
    for index in range(base.landmarks.size - 2):
        landmarks = base.landmarks[index : index + 3]
        if not _structurally_valid(geometry, landmarks):
            continue
        left, middle, right = (int(value) for value in landmarks)
        first_coordinate, second_coordinate = dtw_phase_coordinates(
            geometry[left : middle + 1],
            geometry[middle : right + 1],
        )
        if not _phase_valid(first_coordinate) or not _phase_valid(second_coordinate):
            continue
        reconstruction = dtw_periodic_reconstruction(
            continuous,
            continuous_mask,
            landmarks,
            first_coordinate,
            second_coordinate,
        )
        maximum = _maximum_huber(
            continuous,
            reconstruction,
            continuous_mask,
            landmarks,
        )
        candidates.append(
            DTWPairResult(
                np.array(landmarks, dtype=np.int32, copy=True),
                first_coordinate,
                second_coordinate,
                reconstruction,
                maximum,
            )
        )
    if not candidates:
        return DTWPairResult(
            empty_landmarks,
            empty,
            empty,
            np.array(continuous, dtype=np.float64, copy=True),
            None,
        )
    selected = min(
        candidates,
        key=lambda row: (
            float(row.maximum_huber),
            int(row.landmarks[0]),
        ),
    )
    if float(selected.maximum_huber) > MAX_DTW_RECONSTRUCTION_HUBER:
        return DTWPairResult(
            empty_landmarks,
            empty,
            empty,
            np.array(continuous, dtype=np.float64, copy=True),
            selected.maximum_huber,
        )
    return selected


def phase_from_dtw_pair(
    length: int,
    result: DTWPairResult,
) -> NDArray[np.float64]:
    """Build a unit-circle phase whose two traversals follow the DTW coordinates."""

    phase = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float64), (length, 1))
    if result.landmarks.size != 3:
        return phase
    left, middle, right = (int(value) for value in result.landmarks)
    for start, stop, coordinate in (
        (left, middle, result.first_coordinate),
        (middle, right, result.second_coordinate),
    ):
        angle = 2.0 * np.pi * coordinate
        phase[start : stop + 1, 0] = np.cos(angle)
        phase[start : stop + 1, 1] = np.sin(angle)
    phase[result.landmarks, 0] = 1.0
    phase[result.landmarks, 1] = 0.0
    return np.asarray(phase, dtype="<f8")


__all__ = ["DTWPairResult", "phase_from_dtw_pair", "select_dtw_pair"]
