"""The circular PAMS base loss and sole signed WARP-PHASE treatment loss."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from pams.warp_phase.warp import OrientedIntegral, oriented_overlap_integral

HUBER_BETA_RADIANS = 0.10
DEFAULT_ALIAS_MARGIN = 0.05
DEFAULT_MINIMUM_WINDOW_INTERVALS = 32
DEFAULT_MINIMUM_TRACK_COVERAGE = 0.80
WINDOW_SAMPLES = 64
WINDOW_STRIDE = 32


@dataclass(frozen=True)
class MatchedPhaseLoss:
    """The matched arm loss with no slot for an auxiliary objective."""

    total: Tensor
    pams: Tensor
    warp_integral: Tensor | None


def _require_floating(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype not in {torch.float32, torch.float64}:
        raise TypeError(f"{name} must have dtype torch.float32 or torch.float64")


def circular_wrap(angle: Tensor) -> Tensor:
    """Wrap finite angles to the contract range ``(-pi, pi]``."""

    _require_floating("angle", angle)
    if not bool(torch.isfinite(angle).all()):
        raise ValueError("angle must contain only finite values")
    pi = angle.new_tensor(math.pi)
    wrapped = torch.remainder(angle + pi, 2.0 * pi) - pi
    return torch.where(wrapped <= -pi, wrapped + 2.0 * pi, wrapped)


def circular_huber(residual: Tensor, *, beta: float = HUBER_BETA_RADIANS) -> Tensor:
    """Elementwise circular Huber penalty with a fixed radian transition."""

    _require_floating("residual", residual)
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    wrapped = circular_wrap(residual)
    absolute = torch.abs(wrapped)
    return torch.where(
        absolute < beta,
        0.5 * absolute.square() / beta,
        absolute - 0.5 * beta,
    )


def phase_increments(phase: Tensor) -> Tensor:
    """Return ``Arg(z[t+1] * conjugate(z[t]))`` for ``[B,T,2]`` phases."""

    _require_floating("phase", phase)
    if phase.ndim != 3 or phase.shape[-1] != 2:
        raise ValueError("phase must have shape [batch, time, 2]")
    if phase.shape[0] < 1 or phase.shape[1] < 2:
        raise ValueError("phase must contain a positive batch and at least two clocks")
    if not bool(torch.isfinite(phase).all()):
        raise ValueError("phase must contain only finite values")
    left = phase[:, :-1]
    right = phase[:, 1:]
    cross = right[..., 1] * left[..., 0] - right[..., 0] * left[..., 1]
    dot = right[..., 0] * left[..., 0] + right[..., 1] * left[..., 1]
    return torch.atan2(cross, dot)


def _validate_sequence_masks(
    phase: Tensor,
    frame_mask: Tensor,
    clock_mask: Tensor,
) -> tuple[int, int]:
    _require_floating("phase", phase)
    if phase.ndim != 3 or phase.shape[-1] != 2:
        raise ValueError("phase must have shape [batch, time, 2]")
    batch, time = phase.shape[:2]
    if batch < 1 or time < 2:
        raise ValueError("phase must contain a positive batch and at least two clocks")
    for name, mask in (("frame_mask", frame_mask), ("clock_mask", clock_mask)):
        if not isinstance(mask, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if mask.shape != (batch, time):
            raise ValueError(f"{name} must have shape {(batch, time)}")
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must have dtype torch.bool")
        if mask.device != phase.device:
            raise ValueError(f"{name} must be on the phase device")
    if bool((frame_mask & ~clock_mask).any()):
        raise ValueError("a padded clock cannot be frame-valid")
    if time > 1 and bool((~clock_mask[:, :-1] & clock_mask[:, 1:]).any()):
        raise ValueError("clock_mask must be a contiguous valid prefix")
    return batch, time


def _window_starts(length: int) -> tuple[int, ...]:
    if length < 2:
        raise ValueError("a track must contain at least two clocks")
    if length <= WINDOW_SAMPLES:
        return (0,)
    final = length - WINDOW_SAMPLES
    starts = list(range(0, final + 1, WINDOW_STRIDE))
    if starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def pams_correspondence_loss(
    phase: Tensor,
    source_clocks: Tensor,
    pseudo_period: Tensor,
    frame_mask: Tensor,
    clock_mask: Tensor,
    *,
    beta: float = HUBER_BETA_RADIANS,
) -> Tensor:
    """Compute the selector-period circular PAMS objective identity-first."""

    batch, time = _validate_sequence_masks(phase, frame_mask, clock_mask)
    if not isinstance(source_clocks, Tensor):
        raise TypeError("source_clocks must be a torch.Tensor")
    if source_clocks.shape != (batch, time):
        raise ValueError(f"source_clocks must have shape {(batch, time)}")
    if source_clocks.dtype != torch.int64:
        raise TypeError("source_clocks must have dtype torch.int64")
    if source_clocks.device != phase.device:
        raise ValueError("source_clocks must be on the phase device")
    if not isinstance(pseudo_period, Tensor):
        raise TypeError("pseudo_period must be a torch.Tensor")
    if pseudo_period.shape != (batch,):
        raise ValueError(f"pseudo_period must have shape {(batch,)}")
    if pseudo_period.dtype != torch.float64:
        raise TypeError("pseudo_period must have dtype torch.float64")
    if pseudo_period.device != phase.device:
        raise ValueError("pseudo_period must be on the phase device")
    if not bool(torch.isfinite(pseudo_period).all()) or bool((pseudo_period <= 0.0).any()):
        raise ValueError("pseudo_period must be finite and positive")

    losses: list[Tensor] = []
    for batch_index in range(batch):
        length = int(clock_mask[batch_index].sum().item())
        if length < 2:
            raise ValueError("every identity must contain at least two non-padding clocks")
        clocks = source_clocks[batch_index, :length]
        if not bool((clocks[1:] > clocks[:-1]).all()):
            raise ValueError("non-padding source clocks must be strictly increasing")
        valid = frame_mask[batch_index, :length]
        left_index, right_index = torch.triu_indices(
            length,
            length,
            offset=1,
            device=phase.device,
        )
        separations = clocks[right_index] - clocks[left_index]
        pair_valid = (
            valid[left_index]
            & valid[right_index]
            & (separations.to(dtype=torch.float64) <= pseudo_period[batch_index])
        )
        if not bool(pair_valid.any()):
            raise ValueError("every identity must contain at least one valid PAMS pair")
        left_phase = phase[batch_index, left_index[pair_valid]]
        right_phase = phase[batch_index, right_index[pair_valid]]
        cross = right_phase[:, 1] * left_phase[:, 0] - right_phase[:, 0] * left_phase[:, 1]
        dot = right_phase[:, 0] * left_phase[:, 0] + right_phase[:, 1] * left_phase[:, 1]
        observed = torch.atan2(cross, dot)
        target = (
            2.0
            * math.pi
            * separations[pair_valid].to(dtype=torch.float64)
            / pseudo_period[batch_index]
        ).to(dtype=observed.dtype)
        losses.append(circular_huber(observed - target, beta=beta).mean())
    return torch.stack(losses).mean()


def clean_phase_integral(
    clean_phase: Tensor,
    source_clocks: Tensor,
    clean_cell_mask: Tensor,
    target_query: Tensor,
    *,
    alias_margin: float = DEFAULT_ALIAS_MARGIN,
) -> OrientedIntegral:
    """Build one stop-gradient clean integral target from normalized phase."""

    _require_floating("clean_phase", clean_phase)
    if clean_phase.ndim != 2 or clean_phase.shape[-1] != 2:
        raise ValueError("clean_phase must have shape [source_time, 2]")
    if clean_phase.shape[0] < 2:
        raise ValueError("clean_phase must contain at least two clocks")
    increments = phase_increments(clean_phase.unsqueeze(0)).squeeze(0)
    return oriented_overlap_integral(
        source_clocks,
        increments,
        clean_cell_mask,
        target_query,
        alias_margin=alias_margin,
    )


def warp_integral_loss(
    warped_phase: Tensor,
    clean_integral: Tensor,
    integral_valid: Tensor,
    target_cell_mask: Tensor,
    clock_mask: Tensor,
    *,
    beta: float = HUBER_BETA_RADIANS,
    minimum_window_intervals: int = DEFAULT_MINIMUM_WINDOW_INTERVALS,
    minimum_track_coverage: float = DEFAULT_MINIMUM_TRACK_COVERAGE,
) -> Tensor:
    """Compute signed local warp-integral loss with fail-closed coverage."""

    _require_floating("warped_phase", warped_phase)
    if warped_phase.ndim != 3 or warped_phase.shape[-1] != 2:
        raise ValueError("warped_phase must have shape [batch, time, 2]")
    batch, time = warped_phase.shape[:2]
    if batch < 1 or time < 2:
        raise ValueError("warped_phase must contain a positive batch and at least two clocks")
    if not bool(torch.isfinite(warped_phase).all()):
        raise ValueError("warped_phase must contain only finite values")
    if not isinstance(clean_integral, Tensor):
        raise TypeError("clean_integral must be a torch.Tensor")
    if clean_integral.shape != (batch, time - 1):
        raise ValueError(f"clean_integral must have shape {(batch, time - 1)}")
    if clean_integral.dtype != torch.float64:
        raise TypeError("clean_integral must have dtype torch.float64")
    if clean_integral.device != warped_phase.device:
        raise ValueError("clean_integral must be on the warped_phase device")
    if not bool(torch.isfinite(clean_integral).all()):
        raise ValueError("clean_integral must contain only finite values")
    for name, mask, shape in (
        ("integral_valid", integral_valid, (batch, time - 1)),
        ("target_cell_mask", target_cell_mask, (batch, time - 1)),
        ("clock_mask", clock_mask, (batch, time)),
    ):
        if not isinstance(mask, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if mask.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must have dtype torch.bool")
        if mask.device != warped_phase.device:
            raise ValueError(f"{name} must be on the warped_phase device")
    if time > 1 and bool((~clock_mask[:, :-1] & clock_mask[:, 1:]).any()):
        raise ValueError("clock_mask must be a contiguous valid prefix")
    interval_present = clock_mask[:, :-1] & clock_mask[:, 1:]
    if bool((integral_valid & ~interval_present).any()):
        raise ValueError("a padded interval cannot contain a valid clean integral")
    if bool((target_cell_mask & ~interval_present).any()):
        raise ValueError("a padded interval cannot be target-cell-valid")
    if (
        isinstance(minimum_window_intervals, bool)
        or not isinstance(minimum_window_intervals, int)
        or minimum_window_intervals < 1
    ):
        raise ValueError("minimum_window_intervals must be a positive integer")
    if not math.isfinite(minimum_track_coverage) or not 0.0 <= minimum_track_coverage <= 1.0:
        raise ValueError("minimum_track_coverage must be finite and lie in [0, 1]")
    if bool((clean_integral[~integral_valid] != 0.0).any()):
        raise ValueError("invalid clean-integral entries must be exact zero")

    observed = phase_increments(warped_phase)
    track_losses: list[Tensor] = []
    for batch_index in range(batch):
        length = int(clock_mask[batch_index].sum().item())
        if length < 2:
            raise ValueError("every identity must contain at least two non-padding clocks")
        valid = (
            integral_valid[batch_index, : length - 1]
            & target_cell_mask[batch_index, : length - 1]
        )
        coverage = float(valid.sum().item()) / float(length - 1)
        if coverage < minimum_track_coverage:
            raise ValueError(
                "valid target-interval coverage is below the frozen track minimum"
            )
        residual = circular_wrap(
            observed[batch_index, : length - 1]
            - clean_integral[batch_index, : length - 1]
            .detach()
            .to(dtype=observed.dtype)
        )
        penalties = circular_huber(residual, beta=beta)
        window_losses: list[Tensor] = []
        for start in _window_starts(length):
            stop = min(start + WINDOW_SAMPLES - 1, length - 1)
            window_valid = valid[start:stop]
            valid_count = int(window_valid.sum().item())
            if valid_count < minimum_window_intervals:
                raise ValueError(
                    "a loss window contains fewer than the frozen minimum valid intervals"
                )
            window_losses.append(penalties[start:stop][window_valid].mean())
        track_losses.append(torch.stack(window_losses).mean())
    return torch.stack(track_losses).mean()


def matched_phase_objective(
    pams: Tensor,
    *,
    arm: Literal["control", "treatment"],
    warp_integral: Tensor | None = None,
) -> MatchedPhaseLoss:
    """Apply the protocol's unique treatment deletion and nothing else."""

    _require_floating("pams", pams)
    if pams.ndim != 0 or not bool(torch.isfinite(pams)):
        raise ValueError("pams must be one finite scalar tensor")
    if arm == "control":
        if warp_integral is not None:
            raise ValueError("control must not receive the warp-integral loss")
        return MatchedPhaseLoss(total=pams, pams=pams, warp_integral=None)
    if arm != "treatment":
        raise ValueError("arm must be 'control' or 'treatment'")
    if warp_integral is None:
        raise ValueError("treatment requires the sole warp-integral loss")
    _require_floating("warp_integral", warp_integral)
    if warp_integral.ndim != 0 or not bool(torch.isfinite(warp_integral)):
        raise ValueError("warp_integral must be one finite scalar tensor")
    if warp_integral.device != pams.device or warp_integral.dtype != pams.dtype:
        raise ValueError("pams and warp_integral must share dtype and device")
    return MatchedPhaseLoss(
        total=pams + warp_integral,
        pams=pams,
        warp_integral=warp_integral,
    )
