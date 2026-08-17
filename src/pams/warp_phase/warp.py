"""Deterministic WARP-PHASE time maps, interpolation, and signed overlaps."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

COORDINATE_DTYPE = torch.float32
CLOCK_DTYPE = torch.int64
QUERY_DTYPE = torch.float64
JOINT_COUNT = 17


@dataclass(frozen=True)
class WarpedTrack:
    """A warped single-identity tensor and all masks derived before zeroing."""

    pose: Tensor
    joint_mask: Tensor
    frame_mask: Tensor
    cell_mask: Tensor
    query: Tensor


@dataclass(frozen=True)
class OrientedIntegral:
    """Float64 clean-measure targets and their fail-closed validity mask."""

    value: Tensor
    valid: Tensor
    required_clean_cells: Tensor


def _require_tensor(
    name: str,
    value: Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | None = None,
) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must be on device {device}")


def _validate_source_clocks(source_clocks: Tensor) -> int:
    if not isinstance(source_clocks, Tensor):
        raise TypeError("source_clocks must be a torch.Tensor")
    if source_clocks.ndim != 1:
        raise ValueError("source_clocks must have shape [source_time]")
    if source_clocks.dtype != CLOCK_DTYPE:
        raise TypeError("source_clocks must have dtype torch.int64")
    source_time = source_clocks.numel()
    if source_time < 2:
        raise ValueError("source_clocks must contain at least two clocks")
    if not bool((source_clocks[1:] > source_clocks[:-1]).all()):
        raise ValueError("source_clocks must be strictly increasing and unique")
    return source_time


def identity_time_map(source_clocks: Tensor) -> Tensor:
    """Return the exact float64 identity query for int64 retained clocks."""

    _validate_source_clocks(source_clocks)
    return source_clocks.to(dtype=QUERY_DTYPE)


def three_segment_time_map(
    target_clocks: Tensor,
    *,
    target_start: float,
    target_end: float,
    source_start: float,
    source_end: float,
    breakpoints: Tensor,
    slopes: Tensor,
    reverse: bool = False,
) -> Tensor:
    """Evaluate one endpoint-normalized, continuous three-segment schedule.

    ``breakpoints`` are the two normalized internal positions and ``slopes``
    are the already endpoint-normalized final slopes.  A zero middle slope is
    the only pause representation.  Reversal swaps the global source
    endpoints and does not sign-canonicalize the schedule.
    """

    if not isinstance(target_clocks, Tensor):
        raise TypeError("target_clocks must be a torch.Tensor")
    if target_clocks.ndim != 1:
        raise ValueError("target_clocks must have shape [target_time]")
    if target_clocks.dtype != QUERY_DTYPE:
        raise TypeError("target_clocks must have dtype torch.float64")
    if target_clocks.numel() < 1:
        raise ValueError("target_clocks must not be empty")
    _require_tensor(
        "breakpoints",
        breakpoints,
        shape=(2,),
        dtype=QUERY_DTYPE,
        device=target_clocks.device,
    )
    _require_tensor(
        "slopes",
        slopes,
        shape=(3,),
        dtype=QUERY_DTYPE,
        device=target_clocks.device,
    )
    scalars = (target_start, target_end, source_start, source_end)
    if not all(math.isfinite(value) for value in scalars):
        raise ValueError("time-map endpoints must be finite")
    if target_end <= target_start or source_end <= source_start:
        raise ValueError("time-map endpoint ranges must be strictly positive")
    if not bool(torch.isfinite(breakpoints).all()) or not bool(torch.isfinite(slopes).all()):
        raise ValueError("breakpoints and slopes must be finite")
    b1, b2 = (float(value.item()) for value in breakpoints)
    if not 0.0 < b1 < b2 < 1.0:
        raise ValueError("breakpoints must satisfy 0 < b1 < b2 < 1")
    if bool((slopes < 0.0).any()):
        raise ValueError("final segment slopes must be nonnegative")
    normalization = b1 * slopes[0] + (b2 - b1) * slopes[1] + (1.0 - b2) * slopes[2]
    if not bool(torch.isclose(normalization, normalization.new_tensor(1.0), rtol=1e-12, atol=1e-12)):
        raise ValueError("final segment slopes must be endpoint-normalized")

    x = (target_clocks - target_start) / (target_end - target_start)
    first_end = slopes[0] * b1
    second_end = first_end + slopes[1] * (b2 - b1)
    normalized = torch.where(
        x < b1,
        slopes[0] * x,
        torch.where(
            x < b2,
            first_end + slopes[1] * (x - b1),
            second_end + slopes[2] * (x - b2),
        ),
    )
    forward = source_start + (source_end - source_start) * normalized
    if reverse:
        return source_end - (forward - source_start)
    return forward


def _source_frame_from_joints(joint_mask: Tensor) -> Tensor:
    hip_valid = joint_mask[..., 11] | joint_mask[..., 12]
    return (joint_mask.sum(dim=-1) >= 8) & hip_valid


def _support_cells(
    source_clocks_float: Tensor,
    source_cell_mask: Tensor,
    left: float,
    right: float,
) -> tuple[bool, int]:
    if not math.isfinite(left) or not math.isfinite(right):
        return False, 0
    source_first = float(source_clocks_float[0].item())
    source_last = float(source_clocks_float[-1].item())
    if left < source_first or left > source_last or right < source_first or right > source_last:
        return False, 0
    if left == right:
        return True, 0

    lower, upper = sorted((left, right))
    required = 0
    for index in range(source_cell_mask.numel()):
        cell_left = float(source_clocks_float[index].item())
        cell_right = float(source_clocks_float[index + 1].item())
        overlap = min(upper, cell_right) - max(lower, cell_left)
        if overlap > 0.0:
            required += 1
            if not bool(source_cell_mask[index]):
                return False, required
    return required > 0, required


def warp_track(
    source_pose: Tensor,
    source_clocks: Tensor,
    source_joint_mask: Tensor,
    source_frame_mask: Tensor,
    source_cell_mask: Tensor,
    target_query: Tensor,
) -> WarpedTrack:
    """Query a collapsed source track with the sole same-cell linear kernel."""

    source_time = _validate_source_clocks(source_clocks)
    device = source_clocks.device
    _require_tensor(
        "source_pose",
        source_pose,
        shape=(source_time, JOINT_COUNT, 3),
        dtype=COORDINATE_DTYPE,
        device=device,
    )
    _require_tensor(
        "source_joint_mask",
        source_joint_mask,
        shape=(source_time, JOINT_COUNT),
        dtype=torch.bool,
        device=device,
    )
    _require_tensor(
        "source_frame_mask",
        source_frame_mask,
        shape=(source_time,),
        dtype=torch.bool,
        device=device,
    )
    _require_tensor(
        "source_cell_mask",
        source_cell_mask,
        shape=(source_time - 1,),
        dtype=torch.bool,
        device=device,
    )
    if not isinstance(target_query, Tensor):
        raise TypeError("target_query must be a torch.Tensor")
    if target_query.ndim != 1:
        raise ValueError("target_query must have shape [target_time]")
    if target_query.dtype != QUERY_DTYPE:
        raise TypeError("target_query must have dtype torch.float64")
    if target_query.device != device:
        raise ValueError("target_query must be on the source device")
    if target_query.numel() < 2:
        raise ValueError("target_query must contain at least two clocks")
    if not bool(torch.isfinite(source_pose).all()):
        raise ValueError("source_pose must contain only finite values")
    derived_source_frame = _source_frame_from_joints(source_joint_mask)
    if not torch.equal(source_frame_mask, derived_source_frame):
        raise ValueError("source_frame_mask is inconsistent with the frozen joint rule")

    target_time = target_query.numel()
    target_pose = torch.zeros(
        (target_time, JOINT_COUNT, 3),
        dtype=COORDINATE_DTYPE,
        device=device,
    )
    target_joint = torch.zeros(
        (target_time, JOINT_COUNT),
        dtype=torch.bool,
        device=device,
    )
    clocks_float = source_clocks.to(dtype=QUERY_DTYPE)
    first = float(clocks_float[0].item())
    last = float(clocks_float[-1].item())

    for target_index in range(target_time):
        query = float(target_query[target_index].item())
        if not math.isfinite(query) or query < first or query > last:
            continue
        insertion = int(torch.searchsorted(clocks_float, target_query[target_index]).item())
        if insertion < source_time and query == float(clocks_float[insertion].item()):
            valid_joint = source_joint_mask[insertion]
            target_joint[target_index] = valid_joint
            target_pose[target_index] = source_pose[insertion].masked_fill(
                ~valid_joint.unsqueeze(-1),
                0.0,
            )
            continue
        if insertion <= 0 or insertion >= source_time:
            continue
        left_index = insertion - 1
        if not bool(source_cell_mask[left_index]):
            continue
        left_clock = clocks_float[left_index]
        right_clock = clocks_float[insertion]
        alpha = (target_query[target_index] - left_clock) / (right_clock - left_clock)
        valid_joint = source_joint_mask[left_index] & source_joint_mask[insertion]
        interpolated = (
            (1.0 - alpha) * source_pose[left_index].to(dtype=QUERY_DTYPE)
            + alpha * source_pose[insertion].to(dtype=QUERY_DTYPE)
        )
        target_joint[target_index] = valid_joint
        target_pose[target_index] = interpolated.to(dtype=COORDINATE_DTYPE).masked_fill(
            ~valid_joint.unsqueeze(-1),
            0.0,
        )

    target_frame = _source_frame_from_joints(target_joint)
    target_cell = torch.zeros((target_time - 1,), dtype=torch.bool, device=device)
    for target_index in range(target_time - 1):
        supported, _count = _support_cells(
            clocks_float,
            source_cell_mask,
            float(target_query[target_index].item()),
            float(target_query[target_index + 1].item()),
        )
        target_cell[target_index] = bool(
            supported and target_frame[target_index] and target_frame[target_index + 1]
        )

    return WarpedTrack(
        pose=target_pose,
        joint_mask=target_joint,
        frame_mask=target_frame,
        cell_mask=target_cell,
        query=target_query.clone(),
    )


def oriented_overlap_integral(
    source_clocks: Tensor,
    clean_increments: Tensor,
    clean_cell_mask: Tensor,
    target_query: Tensor,
    *,
    alias_margin: float = 0.05,
) -> OrientedIntegral:
    """Integrate a raw signed clean phase measure over oriented target cells.

    Any missing positive-length support, invalid required clean cell, or alias
    boundary invalidates the whole target interval and leaves an exact-zero
    stored target.  Pauses inside support are valid exact-zero integrals.
    """

    source_time = _validate_source_clocks(source_clocks)
    device = source_clocks.device
    if clean_increments.dtype not in {torch.float32, torch.float64}:
        raise TypeError("clean_increments must have dtype torch.float32 or torch.float64")
    if clean_increments.shape != (source_time - 1,):
        raise ValueError(
            "clean_increments must have shape "
            f"{(source_time - 1,)}, got {tuple(clean_increments.shape)}"
        )
    if clean_increments.device != device:
        raise ValueError("clean_increments must be on the source device")
    _require_tensor(
        "clean_cell_mask",
        clean_cell_mask,
        shape=(source_time - 1,),
        dtype=torch.bool,
        device=device,
    )
    if not isinstance(target_query, Tensor):
        raise TypeError("target_query must be a torch.Tensor")
    if target_query.ndim != 1 or target_query.numel() < 2:
        raise ValueError("target_query must have shape [target_time] with target_time >= 2")
    if target_query.dtype != QUERY_DTYPE:
        raise TypeError("target_query must have dtype torch.float64")
    if target_query.device != device:
        raise ValueError("target_query must be on the source device")
    if not bool(torch.isfinite(clean_increments).all()):
        raise ValueError("clean_increments must contain only finite values")
    if not math.isfinite(alias_margin) or not 0.0 <= alias_margin < 1.0:
        raise ValueError("alias_margin must be finite and lie in [0, 1)")

    clocks = source_clocks.to(dtype=QUERY_DTYPE)
    increments = clean_increments.detach().to(dtype=QUERY_DTYPE)
    rates = increments / (clocks[1:] - clocks[:-1])
    target_intervals = target_query.numel() - 1
    values = torch.zeros((target_intervals,), dtype=QUERY_DTYPE, device=device)
    valid = torch.zeros((target_intervals,), dtype=torch.bool, device=device)
    required_counts = torch.zeros((target_intervals,), dtype=torch.int64, device=device)
    alias_limit = math.pi * (1.0 - alias_margin)
    first = float(clocks[0].item())
    last = float(clocks[-1].item())

    for target_index in range(target_intervals):
        left = float(target_query[target_index].item())
        right = float(target_query[target_index + 1].item())
        if (
            not math.isfinite(left)
            or not math.isfinite(right)
            or left < first
            or left > last
            or right < first
            or right > last
        ):
            continue
        if left == right:
            valid[target_index] = True
            continue

        sign = 1.0 if right > left else -1.0
        lower, upper = sorted((left, right))
        integral = rates.new_zeros(())
        interval_valid = True
        required = 0
        for cell_index in range(source_time - 1):
            cell_left = float(clocks[cell_index].item())
            cell_right = float(clocks[cell_index + 1].item())
            overlap = min(upper, cell_right) - max(lower, cell_left)
            if overlap <= 0.0:
                continue
            required += 1
            if not bool(clean_cell_mask[cell_index]) or bool(
                torch.abs(increments[cell_index]) >= alias_limit
            ):
                interval_valid = False
                break
            integral = integral + sign * overlap * rates[cell_index]
        required_counts[target_index] = required
        if required == 0 or not interval_valid or bool(torch.abs(integral) >= alias_limit):
            continue
        values[target_index] = integral
        valid[target_index] = True

    return OrientedIntegral(
        value=values,
        valid=valid,
        required_clean_cells=required_counts,
    )
