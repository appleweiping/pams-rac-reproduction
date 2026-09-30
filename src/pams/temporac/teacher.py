"""Primitive-orbit phase teacher for ``temporac.execution.v4``."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn


class TeacherContractError(ValueError):
    """Raised when a teacher input or selection violates the frozen contract."""


def _unit_l2(value: Tensor, *, epsilon: float = 1e-8) -> Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if bool(torch.any(~torch.isfinite(norm)).item()) or bool(torch.any(norm < epsilon).item()):
        raise TeacherContractError("teacher unit-L2 normalization failed")
    return value / norm


class StaticNetwork(nn.Module):
    """The exact 298->128->32 static-content network."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(298, 128), nn.GELU(), nn.Linear(128, 32))

    def forward(self, static_features: Tensor) -> Tensor:
        if static_features.shape[-1] != 298:
            raise TeacherContractError("static features must have width 298")
        return _unit_l2(self.layers(static_features))


class PhaseNetwork(nn.Module):
    """The exact memoryless 247->256->128->2 phase network."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(247, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(self, teacher_input: Tensor, static_code: Tensor) -> Tensor:
        if teacher_input.shape[-1] != 215 or static_code.shape[-1] != 32:
            raise TeacherContractError("phase network expects widths 215 and 32")
        expanded = static_code
        while expanded.ndim < teacher_input.ndim:
            expanded = expanded.unsqueeze(-2)
        expanded = expanded.expand(*teacher_input.shape[:-1], 32)
        return _unit_l2(self.layers(torch.cat((teacher_input, expanded), dim=-1)))


class ReconstructionNetwork(nn.Module):
    """No-bypass phase/static reconstruction network."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(34, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 149),
        )

    def forward(self, phase: Tensor, static_code: Tensor) -> Tensor:
        if phase.shape[-1] != 2 or static_code.shape[-1] != 32:
            raise TeacherContractError("reconstruction expects phase width 2 and static width 32")
        expanded = static_code
        while expanded.ndim < phase.ndim:
            expanded = expanded.unsqueeze(-2)
        expanded = expanded.expand(*phase.shape[:-1], 32)
        return self.layers(torch.cat((phase, expanded), dim=-1))


class TempoRACTeacher(nn.Module):
    """Static, phase, and no-bypass reconstruction modules as one graph."""

    def __init__(self) -> None:
        super().__init__()
        self.static = StaticNetwork()
        self.phase = PhaseNetwork()
        self.reconstruction = ReconstructionNetwork()

    def forward(
        self, teacher_input: Tensor, static_features: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        static_code = self.static(static_features)
        phase = self.phase(teacher_input, static_code)
        reconstruction = self.reconstruction(phase, static_code)
        return static_code, phase, reconstruction


def phase_increments(phase: Tensor) -> Tensor:
    """Return signed principal adjacent increments in cycles."""

    if phase.shape[-1] != 2 or phase.shape[-2] < 2:
        raise TeacherContractError("phase must end in [samples>=2,2]")
    left = phase[..., :-1, :]
    right = phase[..., 1:, :]
    cross = left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]
    dot = torch.sum(left * right, dim=-1)
    return torch.atan2(cross, dot) / (2.0 * math.pi)


def masked_huber(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    delta: float = 0.05,
) -> Tensor:
    """Mask-normalized Huber mean with a positive finite denominator."""

    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise TeacherContractError("prediction, target, and mask shapes must match")
    if delta <= 0.0:
        raise TeacherContractError("Huber delta must be positive")
    weights = mask.to(dtype=prediction.dtype)
    denominator = torch.sum(weights)
    if bool((~torch.isfinite(denominator) | (denominator <= 0)).item()):
        raise TeacherContractError("masked Huber denominator must be finite and positive")
    error = torch.abs(prediction - target)
    loss = torch.where(error <= delta, 0.5 * error.square() / delta, error - 0.5 * delta)
    result = torch.sum(weights * loss) / denominator
    if bool((~torch.isfinite(result)).item()):
        raise TeacherContractError("masked Huber result is nonfinite")
    return result


@dataclass(frozen=True, slots=True)
class TeacherLoss:
    total: Tensor
    reconstruction: Tensor
    correspondence: Tensor
    orientation: Tensor
    alias: Tensor


def teacher_objective(
    *,
    reconstruction: Tensor,
    reconstruction_target: Tensor,
    reconstruction_mask: Tensor,
    reference_phase: Tensor,
    view_phase: Tensor,
    observed_phase: Tensor,
    observed_edge_mask: Tensor | None = None,
) -> TeacherLoss:
    """Compute F11 for an already hierarchy-balanced unit batch.

    Callers perform the contract's fraction/traversal/block/source equal-vote
    reductions before stacking the unit tensors supplied here.
    """

    rec = masked_huber(reconstruction, reconstruction_target, reconstruction_mask)
    if reference_phase.shape != view_phase.shape or reference_phase.shape[-1] != 2:
        raise TeacherContractError("correspondence phases must have equal [...,2] shapes")
    correspondence_rows = 1.0 - torch.sum(reference_phase * view_phase, dim=-1)
    correspondence = torch.mean(correspondence_rows)
    increments = phase_increments(observed_phase)
    if observed_edge_mask is not None:
        if observed_edge_mask.shape != increments.shape:
            raise TeacherContractError("observed edge mask shape mismatch")
        selected = increments[observed_edge_mask.to(dtype=torch.bool)]
    else:
        selected = increments.reshape(-1)
    if selected.numel() == 0:
        raise TeacherContractError("teacher orientation denominator is zero")
    orientation = torch.mean(torch.relu(-selected))
    alias = torch.mean(torch.relu(torch.abs(selected) - 0.25))
    total = rec + correspondence + 0.25 * orientation + 0.25 * alias
    for name, value in (
        ("total", total),
        ("reconstruction", rec),
        ("correspondence", correspondence),
        ("orientation", orientation),
        ("alias", alias),
    ):
        if bool((~torch.isfinite(value)).item()):
            raise TeacherContractError(f"teacher {name} is nonfinite")
    return TeacherLoss(total, rec, correspondence, orientation, alias)


def arc_static_features(
    continuous: NDArray[np.float64],
    edge_lengths: NDArray[np.float64],
    traversal_bounds: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Compute the exact piecewise-linear 298-dimensional F5 static input."""

    values = np.asarray(continuous, dtype=np.float64)
    lengths = np.asarray(edge_lengths, dtype=np.float64)
    bounds = np.asarray(traversal_bounds, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != 149:
        raise TeacherContractError("continuous channels must have shape [T,149]")
    if lengths.shape != (values.shape[0] - 1,):
        raise TeacherContractError("edge_lengths must have shape [T-1]")
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] < 1:
        raise TeacherContractError("traversal bounds must have shape [K,2]")
    means: list[NDArray[np.float64]] = []
    seconds: list[NDArray[np.float64]] = []
    for left_raw, right_raw in bounds:
        left = int(left_raw)
        right = int(right_raw)
        if not 0 <= left < right <= lengths.size:
            raise TeacherContractError("invalid traversal bounds")
        weight = lengths[left:right]
        denominator = float(np.sum(weight, dtype=np.float64))
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise TeacherContractError("static-code arc denominator is invalid")
        endpoint_left = values[left:right]
        endpoint_right = values[left + 1 : right + 1]
        means.append(
            np.sum(weight[:, None] * (endpoint_left + endpoint_right) / 2.0, axis=0) / denominator
        )
        seconds.append(
            np.sum(
                weight[:, None]
                * (
                    np.square(endpoint_left)
                    + endpoint_left * endpoint_right
                    + np.square(endpoint_right)
                )
                / 3.0,
                axis=0,
            )
            / denominator
        )
    mean = np.mean(np.stack(means), axis=0, dtype=np.float64)
    second = np.mean(np.stack(seconds), axis=0, dtype=np.float64)
    spread = np.sqrt(np.maximum(second - np.square(mean), 0.0))
    return np.asarray(np.concatenate((mean, spread)), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class TeacherCheckpointScore:
    seed: int
    step: int
    tune_objective: float
    negative_edge_fraction: float
    tune_abstentions: int
    mean_masked_reconstruction: float
    payload: object | None = None

    def __post_init__(self) -> None:
        if self.seed not in (20260815, 20260816, 20260817):
            raise TeacherContractError("unexpected teacher seed")
        if self.step <= 0 or self.step % 500 != 0 or self.step > 20_000:
            raise TeacherContractError("teacher checkpoint step must be 500..20000 by 500")
        numeric = (
            self.tune_objective,
            self.negative_edge_fraction,
            self.mean_masked_reconstruction,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise TeacherContractError("teacher checkpoint score is nonfinite")
        if not 0.0 <= self.negative_edge_fraction <= 1.0 or self.tune_abstentions < 0:
            raise TeacherContractError("invalid teacher checkpoint diagnostic")


def select_teacher_checkpoint(
    checkpoints: Iterable[TeacherCheckpointScore],
) -> TeacherCheckpointScore:
    """Apply the exact within-seed then across-seed v4 selection precedence."""

    rows = tuple(checkpoints)
    selected_by_seed: list[TeacherCheckpointScore] = []
    for seed in (20260815, 20260816, 20260817):
        eligible = [row for row in rows if row.seed == seed and row.negative_edge_fraction <= 0.01]
        if not eligible:
            raise TeacherContractError(f"teacher seed {seed} has no eligible checkpoint")
        selected_by_seed.append(min(eligible, key=lambda row: (row.tune_objective, row.step)))
    return min(
        selected_by_seed,
        key=lambda row: (
            row.tune_abstentions,
            row.mean_masked_reconstruction,
            row.tune_objective,
            row.seed,
        ),
    )


def interpolate_phase_shortest_arc(left: Tensor, right: Tensor, fraction: Tensor) -> Tensor:
    """Interpolate unit phase pairs along their signed principal arc."""

    left_unit = _unit_l2(left)
    right_unit = _unit_l2(right)
    cross = left_unit[..., 0] * right_unit[..., 1] - left_unit[..., 1] * right_unit[..., 0]
    dot = torch.sum(left_unit * right_unit, dim=-1)
    angle = torch.atan2(cross, dot)
    base = torch.atan2(left_unit[..., 1], left_unit[..., 0])
    output_angle = base + fraction * angle
    return torch.stack((torch.cos(output_angle), torch.sin(output_angle)), dim=-1)


__all__ = [
    "PhaseNetwork",
    "ReconstructionNetwork",
    "StaticNetwork",
    "TeacherCheckpointScore",
    "TeacherContractError",
    "TeacherLoss",
    "TempoRACTeacher",
    "arc_static_features",
    "interpolate_phase_shortest_arc",
    "masked_huber",
    "phase_increments",
    "select_teacher_checkpoint",
    "teacher_objective",
]
