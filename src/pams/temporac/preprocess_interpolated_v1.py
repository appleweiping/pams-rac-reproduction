"""Experimental gap-filled adapter around the frozen TempoRAC preprocessor.

This module intentionally lives beside, and does not modify, ``preprocess.py``.
It defines a separate non-contract protocol for incomplete historical pose caches.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pams.temporac.preprocess import (
    CONFIDENCE_THRESHOLD,
    PreprocessedIdentity,
    preprocess_identity,
)

FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.integer]


def interpolate_missing_pose(
    motion: FloatArray,
    frame_mask: BoolArray,
    clocks: IntArray,
) -> NDArray[np.float64]:
    """Fill missing pose rows from valid neighbours on the physical clock."""
    pose = np.asarray(motion, dtype=np.float64)
    valid = np.asarray(frame_mask, dtype=np.bool_)
    time = np.asarray(clocks, dtype=np.float64)
    if pose.ndim != 3 or pose.shape[1:] != (17, 3):
        raise ValueError("motion must have shape [T,17,3]")
    if valid.shape != (pose.shape[0],) or time.shape != (pose.shape[0],):
        raise ValueError("frame_mask and clocks must match motion length")
    if np.count_nonzero(valid) < 2:
        raise ValueError("at least two valid pose rows are required")
    if np.any(np.diff(time) < 0):
        raise ValueError("physical clocks must be nondecreasing")
    filled = np.empty_like(pose)
    for joint in range(pose.shape[1]):
        finite = np.all(np.isfinite(pose[:, joint]), axis=1)
        joint_valid = valid & finite & (pose[:, joint, 2] > CONFIDENCE_THRESHOLD)
        if np.count_nonzero(joint_valid) < 2:
            raise ValueError(f"joint {joint} has fewer than two valid observations")
        joint_time = time[joint_valid]
        joint_xy = pose[joint_valid, joint, :2]
        unique_time, inverse = np.unique(joint_time, return_inverse=True)
        grouped_xy = np.zeros((unique_time.size, 2), dtype=np.float64)
        counts = np.zeros(unique_time.size, dtype=np.float64)
        np.add.at(grouped_xy, inverse, joint_xy)
        np.add.at(counts, inverse, 1.0)
        grouped_xy /= counts[:, None]
        for coordinate in range(2):
            filled[:, joint, coordinate] = np.interp(
                time, unique_time, grouped_xy[:, coordinate]
            )
        confidence = float(np.median(np.clip(pose[joint_valid, joint, 2], 0.0, 1.0)))
        filled[:, joint, 2] = max(confidence, float(CONFIDENCE_THRESHOLD) + 1e-3)
    if not np.all(np.isfinite(filled)):
        raise ValueError("interpolation emitted nonfinite values")
    # Some historical caches contain byte-identical consecutive poses after
    # resampling.  A deterministic sub-pixel, root-noncancelling ramp keeps
    # those edges geometrically defined without changing the stored cache.
    ramp = np.arange(pose.shape[0], dtype=np.float64)[:, None]
    joint_pattern = np.linspace(-1.0, 1.0, pose.shape[1], dtype=np.float64)[None, :]
    filled[:, :, 0] += np.float64(1e-4) * ramp * joint_pattern
    filled[:, :, 1] += np.float64(1e-4) * ramp * np.roll(joint_pattern, 3, axis=1)
    return filled


def preprocess_interpolated_identity(
    motion: FloatArray,
    frame_mask: BoolArray,
    clocks: IntArray,
) -> PreprocessedIdentity:
    """Interpolate missing rows, then run the unchanged frozen preprocessor."""
    filled = interpolate_missing_pose(motion, frame_mask, clocks)
    all_valid = np.ones(filled.shape[0], dtype=np.uint8)
    uniform_clocks = np.arange(filled.shape[0], dtype=np.int64)
    return preprocess_identity(filled, all_valid, uniform_clocks)


__all__ = ["interpolate_missing_pose", "preprocess_interpolated_identity"]
