"""Balanced pulse-response objective and equal-unit reductions (F14)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor


class ObjectiveContractError(ValueError):
    """Raised when an F14 denominator or hierarchy is invalid."""


@dataclass(frozen=True, slots=True)
class ResponseFunctional:
    total: Tensor
    balanced_bce: Tensor
    positive_margin: Tensor
    negative_valley: Tensor


def _as_probability_rows(
    probability: Tensor,
    pulse: Tensor,
    weight: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if probability.shape != pulse.shape or probability.shape != weight.shape:
        raise ObjectiveContractError("probability, pulse, and weight shapes must match")
    if probability.numel() == 0:
        raise ObjectiveContractError("response functional cannot be empty")
    if bool(torch.any(~torch.isfinite(probability)).item()):
        raise ObjectiveContractError("response probability is nonfinite")
    if bool(torch.any((probability < 0.0) | (probability > 1.0)).item()):
        raise ObjectiveContractError("response probability must be in [0,1]")
    if bool(torch.any(~torch.isfinite(weight)).item()) or bool(torch.any(weight < 0).item()):
        raise ObjectiveContractError("response weight must be finite and nonnegative")
    binary = pulse.to(dtype=probability.dtype)
    if bool(torch.any((binary != 0.0) & (binary != 1.0)).item()):
        raise ObjectiveContractError("pulse target must be binary")
    return probability, binary, weight.to(dtype=probability.dtype)


def balanced_response_functional(
    probability: Tensor,
    pulse: Tensor,
    weight: Tensor,
) -> ResponseFunctional:
    """Compute F14's balanced BCE and positive/negative margin penalties."""

    x, p, w = _as_probability_rows(probability, pulse, weight)
    positive_weight = torch.sum(w * p)
    negative_weight = torch.sum(w * (1.0 - p))
    if bool(
        torch.any(
            torch.stack(
                (
                    ~torch.isfinite(positive_weight),
                    ~torch.isfinite(negative_weight),
                    positive_weight <= 0.0,
                    negative_weight <= 0.0,
                )
            )
        ).item()
    ):
        raise ObjectiveContractError("F14 requires positive finite W+ and W-")
    clipped = torch.clamp(x, min=1e-7, max=1.0 - 1e-7)
    positive_bce = torch.sum(w * p * torch.log(clipped)) / positive_weight
    negative_bce = torch.sum(w * (1.0 - p) * torch.log1p(-clipped)) / negative_weight
    bce = -0.5 * (positive_bce + negative_bce)
    positive_margin = torch.sum(w * p * torch.relu(0.75 - x).square()) / positive_weight
    negative_valley = torch.sum(w * (1.0 - p) * torch.relu(x - 0.25).square()) / negative_weight
    total = bce + 0.5 * positive_margin + 0.5 * negative_valley
    for value in (bce, positive_margin, negative_valley, total):
        if bool((~torch.isfinite(value)).item()):
            raise ObjectiveContractError("F14 emitted a nonfinite term")
    return ResponseFunctional(total, bce, positive_margin, negative_valley)


@dataclass(frozen=True, slots=True)
class UnitResponseLoss:
    total: Tensor
    branch: tuple[ResponseFunctional, ResponseFunctional, ResponseFunctional]
    fused: ResponseFunctional
    traversal_count: int


def traversal_unit_loss(
    branch_probability: Tensor,
    fused_probability: Tensor,
    pulse: Tensor,
    edge_mask: Tensor,
    chi: Tensor,
    responsibility: Tensor,
    traversal_bounds: Tensor | Sequence[Sequence[int]],
    *,
    capacity_control: bool = False,
) -> UnitResponseLoss:
    """Reduce F14 equally by traversal for one source view.

    ``branch_probability`` is ``[E,3]`` in slow/medium/fast order and
    ``responsibility`` has the same shape.  Capacity control deliberately
    omits the responsibility multiplier while retaining all other weights.
    """

    if branch_probability.ndim != 2 or branch_probability.shape[1] != 3:
        raise ObjectiveContractError("branch probabilities must have shape [E,3]")
    edge_count = branch_probability.shape[0]
    if fused_probability.shape != (edge_count,):
        raise ObjectiveContractError("fused probability must have shape [E]")
    for name, value in (
        ("pulse", pulse),
        ("edge_mask", edge_mask),
        ("chi", chi),
    ):
        if value.shape != (edge_count,):
            raise ObjectiveContractError(f"{name} must have shape [E]")
    if responsibility.shape != (edge_count, 3):
        raise ObjectiveContractError("responsibility must have shape [E,3]")
    bounds = torch.as_tensor(traversal_bounds, dtype=torch.int64, device=pulse.device)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] < 1:
        raise ObjectiveContractError("traversal bounds must have shape [K,2]")
    branch_by_traversal: list[list[ResponseFunctional]] = [[], [], []]
    fused_by_traversal: list[ResponseFunctional] = []
    base_weight = edge_mask.to(dtype=chi.dtype) * chi
    for left_tensor, right_tensor in bounds:
        left = int(left_tensor.item())
        right = int(right_tensor.item())
        if not 0 <= left < right <= edge_count:
            raise ObjectiveContractError("invalid traversal bounds")
        fused_by_traversal.append(
            balanced_response_functional(
                fused_probability[left:right], pulse[left:right], base_weight[left:right]
            )
        )
        for branch_index in range(3):
            branch_weight = base_weight[left:right]
            if not capacity_control:
                branch_weight = branch_weight * responsibility[left:right, branch_index]
            branch_by_traversal[branch_index].append(
                balanced_response_functional(
                    branch_probability[left:right, branch_index],
                    pulse[left:right],
                    branch_weight,
                )
            )

    def reduce_rows(rows: Sequence[ResponseFunctional]) -> ResponseFunctional:
        return ResponseFunctional(
            total=torch.mean(torch.stack([row.total for row in rows])),
            balanced_bce=torch.mean(torch.stack([row.balanced_bce for row in rows])),
            positive_margin=torch.mean(torch.stack([row.positive_margin for row in rows])),
            negative_valley=torch.mean(torch.stack([row.negative_valley for row in rows])),
        )

    branch = tuple(reduce_rows(rows) for rows in branch_by_traversal)
    branch_typed = (branch[0], branch[1], branch[2])
    fused = reduce_rows(fused_by_traversal)
    total = torch.mean(torch.stack([row.total for row in branch_typed])) + fused.total
    return UnitResponseLoss(total, branch_typed, fused, int(bounds.shape[0]))


def equal_block_reduction(block_losses: Iterable[UnitResponseLoss]) -> Tensor:
    """Give each nonempty X0 block one vote."""

    rows = tuple(block_losses)
    if not rows:
        raise ObjectiveContractError("block reduction cannot be empty")
    result = torch.mean(torch.stack([row.total for row in rows]))
    if bool((~torch.isfinite(result)).item()):
        raise ObjectiveContractError("block reduction is nonfinite")
    return result


def balanced_family_step_loss(x0_loss: Tensor, natural_loss: Tensor) -> Tensor:
    """Return the exact one-X0/one-natural 0.5/0.5 response step loss."""

    if x0_loss.numel() != 1 or natural_loss.numel() != 1:
        raise ObjectiveContractError("family losses must be scalar")
    if bool(torch.any(~torch.isfinite(torch.stack((x0_loss, natural_loss)))).item()):
        raise ObjectiveContractError("family loss is nonfinite")
    return 0.5 * x0_loss + 0.5 * natural_loss


def responsibility_fraction(
    responsibility: Tensor,
    edge_mask: Tensor,
    chi: Tensor,
) -> Tensor:
    """Return the three exact soft-responsibility support fractions."""

    if responsibility.ndim != 2 or responsibility.shape[1] != 3:
        raise ObjectiveContractError("responsibility must have shape [E,3]")
    if edge_mask.shape != responsibility.shape[:1] or chi.shape != edge_mask.shape:
        raise ObjectiveContractError("responsibility support shapes mismatch")
    weight = edge_mask.to(dtype=responsibility.dtype) * chi.to(dtype=responsibility.dtype)
    totals = torch.sum(weight[:, None] * responsibility, dim=0)
    denominator = torch.sum(totals)
    if bool((~torch.isfinite(denominator) | (denominator <= 0)).item()):
        raise ObjectiveContractError("responsibility denominator is invalid")
    return totals / denominator


__all__ = [
    "ObjectiveContractError",
    "ResponseFunctional",
    "UnitResponseLoss",
    "balanced_family_step_loss",
    "balanced_response_functional",
    "equal_block_reduction",
    "responsibility_fraction",
    "traversal_unit_loss",
]
