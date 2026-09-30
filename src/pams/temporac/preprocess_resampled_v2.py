"""Versioned gap-resampling candidate for incomplete v44 pose tracks.

This module is deliberately additive: the frozen :mod:`pams.temporac.preprocess`
implementation remains unchanged.  The adapter turns a fixed 320-row historical
track into a fully observed, uniform-clock feature record before delegating every
normalization and channel-construction step to the frozen preprocessor.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pams.temporac.preprocess import (
    BONES,
    CONFIDENCE_THRESHOLD,
    PreprocessedIdentity,
    preprocess_identity,
)
from pams.temporac.types import FeatureRecord

PROTOCOL = "temporac.gap-resample.v2"
TIE_BREAK_EPSILON = 1e-4


class ResamplingError(ValueError):
    """Raised when a source track cannot be deterministically resampled."""


@dataclass(frozen=True, slots=True)
class ResamplingReceipt:
    protocol: str
    source_clock_sha256: str
    source_frame_mask_sha256: str
    resampled_motion_sha256: str
    resampled_clock_sha256: str
    interpolated_rows: int
    degenerate_edges_repaired: int


def _array_sha256(value: NDArray[np.generic]) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _validate(
    motion: NDArray[np.floating],
    frame_mask: NDArray[np.generic],
    clocks: NDArray[np.integer],
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.float64]]:
    pose = np.asarray(motion, dtype=np.float64)
    valid = np.asarray(frame_mask)
    time = np.asarray(clocks, dtype=np.float64)
    if pose.ndim != 3 or pose.shape[1:] != (17, 3):
        raise ResamplingError("motion must have shape [T,17,3]")
    if valid.shape != (pose.shape[0],) or valid.dtype.kind not in "bu":
        raise ResamplingError("frame_mask must be binary [T]")
    if not np.all((valid == 0) | (valid == 1)):
        raise ResamplingError("frame_mask must be binary")
    if time.shape != (pose.shape[0],) or not np.all(np.isfinite(time)):
        raise ResamplingError("clocks must be finite [T]")
    if np.any(np.diff(time) < 0):
        raise ResamplingError("physical clocks must be nondecreasing")
    if pose.shape[0] < 17:
        raise ResamplingError("at least 17 source rows are required")
    return pose, valid.astype(np.bool_), time


def _interpolate_joint(
    pose: NDArray[np.float64],
    frame_valid: NDArray[np.bool_],
    source_time: NDArray[np.float64],
    output_time: NDArray[np.float64],
    joint: int,
) -> NDArray[np.float64]:
    finite = np.all(np.isfinite(pose[:, joint]), axis=1)
    observed = (
        frame_valid
        & finite
        & (pose[:, joint, 2] > np.float64(CONFIDENCE_THRESHOLD))
    )
    if np.count_nonzero(observed) < 2:
        raise ResamplingError(f"joint {joint} has fewer than two valid observations")
    observed_time = source_time[observed]
    values = pose[observed, joint]
    unique_time, inverse = np.unique(observed_time, return_inverse=True)
    grouped = np.zeros((unique_time.size, 3), dtype=np.float64)
    counts = np.zeros(unique_time.size, dtype=np.float64)
    np.add.at(grouped, inverse, values)
    np.add.at(counts, inverse, 1.0)
    grouped /= counts[:, None]
    output = np.empty((output_time.size, 3), dtype=np.float64)
    for coordinate in range(3):
        output[:, coordinate] = np.interp(
            output_time, unique_time, grouped[:, coordinate]
        )
    output[:, 2] = np.maximum(
        np.clip(output[:, 2], 0.0, 1.0),
        np.float64(CONFIDENCE_THRESHOLD) + np.float64(1e-3),
    )
    return output


def _geometry(pose: NDArray[np.float64]) -> NDArray[np.float64]:
    xy = pose[:, :, :2]
    bones = np.stack([xy[:, target] - xy[:, source] for source, target in BONES], axis=1)
    return np.concatenate((xy.reshape(pose.shape[0], 34), bones.reshape(pose.shape[0], 32)), axis=1)


def _repair_degenerate_edges(pose: NDArray[np.float64]) -> int:
    """Apply a minimal deterministic coordinate tie-break only to zero edges."""

    repaired = 0
    pattern = np.linspace(-1.0, 1.0, 17, dtype=np.float64)
    pattern_y = np.roll(pattern, 3)
    for edge in range(pose.shape[0] - 1):
        delta = _geometry(pose[edge : edge + 2])[1] - _geometry(pose[edge : edge + 2])[0]
        if float(np.linalg.norm(delta)) > 1e-12:
            continue
        repaired += 1
        step = np.float64(TIE_BREAK_EPSILON * repaired)
        pose[edge + 1 :, :, 0] += step * pattern[None, :]
        pose[edge + 1 :, :, 1] += step * pattern_y[None, :]
    return repaired


def resample_missing_pose(
    motion: NDArray[np.floating],
    frame_mask: NDArray[np.generic],
    clocks: NDArray[np.integer],
) -> tuple[NDArray[np.float32], NDArray[np.int64], ResamplingReceipt]:
    """Resample one track on a uniform 320-row grid and emit its audit receipt."""

    pose, valid, time = _validate(motion, frame_mask, clocks)
    output_time = np.linspace(float(time[0]), float(time[-1]), pose.shape[0], dtype=np.float64)
    filled = np.stack(
        [_interpolate_joint(pose, valid, time, output_time, joint) for joint in range(17)],
        axis=1,
    )
    repaired = _repair_degenerate_edges(filled)
    if not np.all(np.isfinite(filled)):
        raise ResamplingError("resampling emitted nonfinite values")
    resampled_motion = np.ascontiguousarray(filled.astype("<f4"))
    resampled_clock = np.arange(pose.shape[0], dtype="<i8")
    receipt = ResamplingReceipt(
        protocol=PROTOCOL,
        source_clock_sha256=_array_sha256(np.asarray(clocks, dtype="<i8")),
        source_frame_mask_sha256=_array_sha256(np.asarray(frame_mask, dtype="|u1")),
        resampled_motion_sha256=_array_sha256(resampled_motion),
        resampled_clock_sha256=_array_sha256(resampled_clock),
        interpolated_rows=int(np.count_nonzero(~valid)),
        degenerate_edges_repaired=repaired,
    )
    return resampled_motion, resampled_clock, receipt


def resample_feature_record(feature: FeatureRecord) -> tuple[FeatureRecord, ResamplingReceipt]:
    """Return a new typed feature record; the source record is never mutated."""

    motion, clocks, receipt = resample_missing_pose(
        feature.motion, feature.frame_mask, feature.sampled_frame_indices
    )
    adapted = FeatureRecord(
        frame_mask=np.ones(motion.shape[0], dtype="|u1"),
        local_person_slot=np.array(feature.local_person_slot, copy=True),
        motion=motion,
        opaque_sample_key=np.array(feature.opaque_sample_key, copy=True),
        person_mask=np.array(feature.person_mask, copy=True),
        sampled_frame_indices=clocks,
        source_length=np.array(feature.source_length, copy=True),
    )
    return adapted, receipt


def preprocess_resampled_identity(
    motion: NDArray[np.floating],
    frame_mask: NDArray[np.generic],
    clocks: NDArray[np.integer],
) -> tuple[PreprocessedIdentity, ResamplingReceipt]:
    """Run the v2 adapter and then the unchanged frozen preprocessor."""

    filled, uniform_clock, receipt = resample_missing_pose(motion, frame_mask, clocks)
    prepared = preprocess_identity(
        filled,
        np.ones(filled.shape[0], dtype="|u1"),
        uniform_clock,
    )
    return prepared, receipt


__all__ = [
    "PROTOCOL",
    "ResamplingError",
    "ResamplingReceipt",
    "preprocess_resampled_identity",
    "resample_feature_record",
    "resample_missing_pose",
]
