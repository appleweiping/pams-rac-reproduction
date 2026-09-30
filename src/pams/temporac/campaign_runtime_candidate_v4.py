"""Protocol-revision diagnostic with an X0-only analytic phase anchor.

This is deliberately not represented as frozen F11.  It tests whether the
unchanged teacher architecture can satisfy the certificate when the synthetic
one-winding gauge is made identifiable.  No natural label enters this loss.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from pams.temporac.campaign_runtime_candidate_v1 import (
    CampaignRuntimeError,
    _teacher_view,
)
from pams.temporac.campaign_runtime_candidate_v3 import teacher_source_loss_v3
from pams.temporac.teacher import TempoRACTeacher


def teacher_source_loss_phase_anchored_v4(
    model: TempoRACTeacher,
    source_id: int,
    *,
    device: torch.device,
    anchor_weight: float = 1.0,
) -> Tensor:
    """Add a direct unit-phase anchor available only on synthetic X0."""

    if anchor_weight <= 0.0:
        raise CampaignRuntimeError("phase anchor weight must be positive")
    base = teacher_source_loss_v3(model, source_id, device=device)
    views = [_teacher_view(source_id, block, device) for block in range(7)]
    lengths = [int(row.teacher_input.shape[0]) for row in views]
    maximum = max(lengths)
    teacher_batch = torch.zeros((7, maximum, 215), dtype=torch.float32, device=device)
    static_batch = torch.stack([row.static_features for row in views])
    for index, row in enumerate(views):
        teacher_batch[index, : lengths[index]] = row.teacher_input
    _static, phase_batch, _reconstruction = model(teacher_batch, static_batch)
    phase_losses: list[Tensor] = []
    for index, row in enumerate(views):
        length = lengths[index]
        target = torch.from_numpy(np.array(row.view.phase, dtype=np.float32, copy=True)).to(device)
        phase_losses.append(
            torch.mean(1.0 - torch.sum(phase_batch[index, :length] * target, dim=1))
        )
    total = base + anchor_weight * torch.mean(torch.stack(phase_losses))
    if not bool(torch.isfinite(total)):
        raise CampaignRuntimeError("phase-anchored teacher loss is nonfinite")
    return total


__all__ = ["teacher_source_loss_phase_anchored_v4"]
