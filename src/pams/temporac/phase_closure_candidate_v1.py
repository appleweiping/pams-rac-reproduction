"""Count-blind natural phase closure over predeclared geometry landmarks."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray


class PhaseClosureError(ValueError):
    """Raised when phase closure receives malformed arrays."""


def close_phase_on_landmarks(
    phase: NDArray[np.float64],
    landmarks: NDArray[np.int32],
    *,
    geometry: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Preserve predicted local speed while enforcing one positive winding."""

    values = np.asarray(phase, dtype=np.float64)
    points = np.asarray(landmarks, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise PhaseClosureError("phase must be finite [T,2]")
    if points.ndim != 1 or points.size < 3 or np.any(np.diff(points) <= 0):
        raise PhaseClosureError("landmarks must be a strictly increasing vector of size >=3")
    if int(points[0]) < 0 or int(points[-1]) >= values.shape[0]:
        raise PhaseClosureError("landmark is outside the phase array")
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norm < 1e-8):
        raise PhaseClosureError("phase normalization failed")
    unit = values / norm
    geometry_values: NDArray[np.float64] | None = None
    if geometry is not None:
        geometry_values = np.asarray(geometry, dtype=np.float64)
        if geometry_values.shape != (values.shape[0], 66) or not np.isfinite(
            geometry_values
        ).all():
            raise PhaseClosureError("geometry must be finite [T,66]")
    output = np.array(unit, copy=True)
    for left_raw, right_raw in zip(points[:-1], points[1:], strict=True):
        left, right = int(left_raw), int(right_raw)
        if right - left < 2:
            raise PhaseClosureError("phase traversal has fewer than two edges")
        if geometry_values is None:
            local_left = unit[left:right]
            local_right = unit[left + 1 : right + 1]
            cross = (
                local_left[:, 0] * local_right[:, 1]
                - local_left[:, 1] * local_right[:, 0]
            )
            dot = np.sum(local_left * local_right, axis=1)
            raw_increment = np.arctan2(cross, dot) / (2.0 * math.pi)
            scaled = np.clip(16.0 * raw_increment, -40.0, 40.0)
            positive = np.log1p(np.exp(scaled)) / 16.0 + 1e-6
        else:
            delta = geometry_values[left + 1 : right + 1] - geometry_values[left:right]
            positive = np.linalg.norm(delta, axis=1) / math.sqrt(66.0) + 1e-12
        cumulative = np.concatenate(([0.0], np.cumsum(positive, dtype=np.float64)))
        cumulative /= cumulative[-1]
        angle = 2.0 * math.pi * cumulative
        output[left : right + 1, 0] = np.cos(angle)
        output[left : right + 1, 1] = np.sin(angle)
    output[points, 0] = 1.0
    output[points, 1] = 0.0
    return np.asarray(output, dtype="<f8")


__all__ = ["PhaseClosureError", "close_phase_on_landmarks"]
