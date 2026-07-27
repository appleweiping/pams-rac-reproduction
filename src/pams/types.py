"""Validated public value types used across the reproduction.

The arrays stored by these classes are copied into immutable byte-backed NumPy
views.  This is intentional: experiment inputs and outputs must not change
after they have been hashed or written to a run manifest.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float32]
BoolArray = NDArray[np.bool_]


def _immutable_array(
    value: ArrayLike,
    *,
    dtype: np.dtype[Any] | type[Any],
) -> NDArray[Any]:
    """Return a C-contiguous array that cannot be made writeable again."""

    contiguous = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    # A bytes object provides a genuinely read-only backing buffer.  Merely
    # calling setflags(write=False) on an owning array can be reversed later.
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return immutable.reshape(contiguous.shape)


def _finite_positive(value: float, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return result


def _non_negative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer, not bool")
    integer = int(value)
    if integer != value or integer < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return integer


@dataclass(frozen=True, slots=True, eq=False)
class PoseSequence:
    """One normalized or raw pose sequence.

    Parameters
    ----------
    video_id:
        Stable identifier from the dataset manifest.
    fps:
        Sampling rate represented by the sequence.
    xyz:
        Pose coordinates with shape ``[frames, 33, 3]``.
    valid_mask:
        Frame-level pose validity with shape ``[frames]``. Invalid frames are
        canonicalized to exact zeros on construction.
    """

    video_id: str
    fps: float
    xyz: FloatArray
    valid_mask: BoolArray

    def __post_init__(self) -> None:
        video_id = str(self.video_id).strip()
        if not video_id:
            raise ValueError("video_id must be a non-empty string")

        fps = _finite_positive(self.fps, "fps")
        xyz = np.asarray(self.xyz, dtype=np.float32)
        mask = np.asarray(self.valid_mask, dtype=np.bool_)

        if xyz.ndim != 3 or xyz.shape[1:] != (33, 3):
            raise ValueError(f"xyz must have shape [frames, 33, 3], received {tuple(xyz.shape)}")
        if xyz.shape[0] < 1:
            raise ValueError("xyz must contain at least one frame")
        if mask.shape != (xyz.shape[0],):
            raise ValueError(
                f"valid_mask must have shape ({xyz.shape[0]},), received {tuple(mask.shape)}"
            )
        if not np.isfinite(xyz[mask]).all():
            raise ValueError("coordinates in valid frames must all be finite")

        canonical = np.array(xyz, dtype=np.float32, copy=True, order="C")
        canonical[~mask] = 0.0
        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "fps", fps)
        object.__setattr__(self, "xyz", _immutable_array(canonical, dtype=np.float32))
        object.__setattr__(self, "valid_mask", _immutable_array(mask, dtype=np.bool_))

    @property
    def num_frames(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def valid_fraction(self) -> float:
        return float(np.mean(self.valid_mask))

    def flattened(self) -> FloatArray:
        """Return the immutable ``[frames, 99]`` encoder input view."""

        return self.xyz.reshape(self.num_frames, 99)

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-compatible metadata (without the potentially large arrays)."""

        return {
            "video_id": self.video_id,
            "fps": self.fps,
            "frames": self.num_frames,
            "keypoints": 33,
            "coordinates": 3,
            "valid_frames": int(np.sum(self.valid_mask)),
        }


@dataclass(frozen=True, slots=True, eq=False)
class CountResult:
    """Immutable output of a counting method."""

    count: int
    period_frames: float
    expert_counts: tuple[int, int, int]
    confidence: float
    period_stream: FloatArray

    def __post_init__(self) -> None:
        count = _non_negative_integer(self.count, "count")
        period = _finite_positive(self.period_frames, "period_frames")
        confidence = float(self.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be finite and in [0, 1]")

        if not isinstance(self.expert_counts, Sequence) or len(self.expert_counts) != 3:
            raise ValueError("expert_counts must contain exactly fast/medium/slow counts")
        experts = tuple(
            _non_negative_integer(value, f"expert_counts[{index}]")
            for index, value in enumerate(self.expert_counts)
        )
        stream = np.asarray(self.period_stream, dtype=np.float32)
        if stream.ndim != 1 or stream.size < 1:
            raise ValueError("period_stream must be a non-empty one-dimensional array")
        if not np.isfinite(stream).all():
            raise ValueError("period_stream must contain only finite values")

        object.__setattr__(self, "count", count)
        object.__setattr__(self, "period_frames", period)
        object.__setattr__(self, "expert_counts", experts)
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "period_stream", _immutable_array(stream, dtype=np.float32))

    def to_dict(self, *, include_stream: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "count": self.count,
            "period_frames": self.period_frames,
            "expert_counts": list(self.expert_counts),
            "confidence": self.confidence,
        }
        if include_stream:
            payload["period_stream"] = self.period_stream.tolist()
        return payload
