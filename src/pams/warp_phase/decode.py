"""Strict positive-interval NOLA and one continuous decode per identity."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from pams.warp_phase.losses import phase_increments

WINDOW_SAMPLES = 64
WINDOW_INTERVALS = 63
WINDOW_STRIDE = 32
MINIMUM_INTERVAL_WEIGHT = 1e-3


@dataclass(frozen=True)
class DecodedTrack:
    """Primary continuous count and explicitly secondary half-even value."""

    continuous: float
    rounded_half_even: int
    interval_mass: Tensor
    denominator: Tensor
    starts: tuple[int, ...]


def positive_interval_weights(*, device: torch.device | None = None) -> Tensor:
    """Return the frozen 63 strictly positive float64 interval weights."""

    index = torch.arange(WINDOW_INTERVALS, dtype=torch.float64, device=device)
    weights = torch.sin(math.pi * (index + 0.5) / WINDOW_INTERVALS).square()
    return torch.clamp(weights, min=MINIMUM_INTERVAL_WEIGHT)


def window_starts(track_length: int) -> tuple[int, ...]:
    """Return stride-32 starts, appending the final window when absent."""

    if isinstance(track_length, bool) or not isinstance(track_length, int):
        raise TypeError("track_length must be an integer")
    if track_length < 2:
        raise ValueError("track_length must be at least two")
    if track_length <= WINDOW_SAMPLES:
        return (0,)
    final = track_length - WINDOW_SAMPLES
    starts = list(range(0, final + 1, WINDOW_STRIDE))
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def round_half_even(value: float) -> int:
    """Apply secondary-only round-to-nearest with ties to even."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("value must be a real scalar")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError("value must be finite and nonnegative")
    return int(round(numeric))


def decode_track(
    phase: Tensor,
    interval_present: Tensor,
    interval_gate: Tensor,
) -> DecodedTrack:
    """Decode one complete supplied identity exactly once.

    ``interval_present`` is the non-padding NOLA support ``v``.  The exact
    protocol gate ``g`` is supplied separately by ``interval_gate`` and may
    never create support.  Windows are read-only slices of the one full-track
    recurrence output.
    """

    if not isinstance(phase, Tensor):
        raise TypeError("phase must be a torch.Tensor")
    if phase.ndim != 2 or phase.shape[-1] != 2:
        raise ValueError("phase must have shape [track_time, 2]")
    if phase.dtype not in {torch.float32, torch.float64}:
        raise TypeError("phase must have dtype torch.float32 or torch.float64")
    track_length = phase.shape[0]
    if track_length < 2:
        raise ValueError("phase must contain at least two clocks")
    if not bool(torch.isfinite(phase).all()):
        raise ValueError("phase must contain only finite values")
    for name, mask in (
        ("interval_present", interval_present),
        ("interval_gate", interval_gate),
    ):
        if not isinstance(mask, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if mask.shape != (track_length - 1,):
            raise ValueError(f"{name} must have shape {(track_length - 1,)}")
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must have dtype torch.bool")
        if mask.device != phase.device:
            raise ValueError(f"{name} must be on the phase device")
    if bool((interval_gate & ~interval_present).any()):
        raise ValueError("interval_gate cannot enable a non-present interval")
    if not bool(interval_present.any()):
        raise ValueError("at least one non-padding interval must be present")
    if interval_present.numel() > 1 and bool(
        (~interval_present[:-1] & interval_present[1:]).any()
    ):
        raise ValueError("interval_present must be a contiguous non-padding prefix")

    nonpadding_intervals = int(interval_present.sum().item())
    nonpadding_length = nonpadding_intervals + 1
    starts = window_starts(nonpadding_length)
    weights = positive_interval_weights(device=phase.device)
    denominator = torch.zeros(
        (track_length - 1,),
        dtype=torch.float64,
        device=phase.device,
    )
    numerator = torch.zeros_like(denominator)
    increments = phase_increments(phase.to(dtype=torch.float64).unsqueeze(0)).squeeze(0)
    raw_mass = torch.abs(increments) / (2.0 * math.pi)

    for start in starts:
        interval_count = min(WINDOW_INTERVALS, nonpadding_intervals - start)
        stop = start + interval_count
        local_present = interval_present[start:stop]
        local_weight = weights[:interval_count]
        weighted_support = local_present.to(dtype=torch.float64) * local_weight
        denominator[start:stop] += weighted_support
        local_gate = interval_gate[start:stop].to(dtype=torch.float64)
        numerator[start:stop] += (
            weighted_support * local_gate * raw_mass[start:stop]
        )

    present_denominator = denominator[interval_present]
    if not bool(torch.isfinite(present_denominator).all()) or bool(
        (present_denominator < MINIMUM_INTERVAL_WEIGHT).any()
    ):
        raise ValueError("strict positive-interval NOLA denominator failed")
    interval_mass = torch.zeros_like(numerator)
    interval_mass[interval_present] = (
        numerator[interval_present] / present_denominator
    )
    if not bool(torch.isfinite(interval_mass).all()):
        raise ValueError("decoded interval mass must be finite")
    continuous = float(interval_mass.sum().item())
    if not math.isfinite(continuous) or continuous < 0.0:
        raise ValueError("decoded continuous count must be finite and nonnegative")
    return DecodedTrack(
        continuous=continuous,
        rounded_half_even=round_half_even(continuous),
        interval_mass=interval_mass,
        denominator=denominator,
        starts=starts,
    )
