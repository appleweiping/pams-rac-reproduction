"""Corrected additive TempoRAC teacher objective candidate.

Candidate v1 incorrectly queried correspondence at clean coordinates in
``[0,10]``.  X0 traversals occupy ``8 + 32 * [0,10]``.  This version also
performs equal traversal/block reductions for reconstruction and phase-edge
penalties, while leaving the frozen model modules unchanged.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from pams.temporac.campaign_runtime_candidate_v1 import (
    CampaignRuntimeError,
    _masked_huber,
    _phase_at_coordinate,
    _teacher_view,
)
from pams.temporac.teacher import TempoRACTeacher, phase_increments


def teacher_source_loss_v3(
    model: TempoRACTeacher,
    source_id: int,
    *,
    device: torch.device,
) -> Tensor:
    """Compute corrected seven-block X0 loss with full-cycle correspondence."""

    if not 0 <= source_id < 32:
        raise CampaignRuntimeError("teacher source must be train/tune X0")
    views = [_teacher_view(source_id, block, device) for block in range(7)]
    lengths = [int(row.teacher_input.shape[0]) for row in views]
    maximum = max(lengths)
    teacher_batch = torch.zeros((7, maximum, 215), dtype=torch.float32, device=device)
    static_batch = torch.stack([row.static_features for row in views])
    for index, row in enumerate(views):
        teacher_batch[index, : lengths[index]] = row.teacher_input
    _static, phase_batch, reconstruction_batch = model(teacher_batch, static_batch)

    phases: list[Tensor] = []
    block_reconstruction: list[Tensor] = []
    block_orientation: list[Tensor] = []
    block_alias: list[Tensor] = []
    for index, row in enumerate(views):
        length = lengths[index]
        phase = phase_batch[index, :length]
        reconstruction = reconstruction_batch[index, :length]
        phases.append(phase)
        traversal_reconstruction: list[Tensor] = []
        traversal_orientation: list[Tensor] = []
        traversal_alias: list[Tensor] = []
        for left_raw, right_raw in row.view.traversal_bounds:
            left, right = int(left_raw), int(right_raw)
            traversal_reconstruction.append(
                _masked_huber(
                    reconstruction[left:right],
                    row.reconstruction_target[left:right],
                    row.reconstruction_mask[left:right],
                )
            )
            increments = phase_increments(phase[left : right + 1])
            traversal_orientation.append(torch.mean(torch.relu(-increments)))
            traversal_alias.append(torch.mean(torch.relu(torch.abs(increments) - 0.25)))
        block_reconstruction.append(torch.mean(torch.stack(traversal_reconstruction)))
        block_orientation.append(torch.mean(torch.stack(traversal_orientation)))
        block_alias.append(torch.mean(torch.stack(traversal_alias)))

    fractions = (np.arange(128, dtype=np.float64) + 0.5) / 128.0
    queries = np.concatenate(
        [8.0 + 32.0 * (traversal + fractions) for traversal in range(10)]
    )
    reference = _phase_at_coordinate(phases[0], views[0].clean_coordinate, queries)
    correspondence = torch.mean(
        torch.stack(
            [
                torch.mean(
                    1.0
                    - torch.sum(
                        reference
                        * _phase_at_coordinate(
                            phases[index],
                            views[index].clean_coordinate,
                            queries,
                        ),
                        dim=1,
                    )
                )
                for index in range(1, 7)
            ]
        )
    )
    total = (
        torch.mean(torch.stack(block_reconstruction))
        + correspondence
        + 0.25 * torch.mean(torch.stack(block_orientation))
        + 0.25 * torch.mean(torch.stack(block_alias))
    )
    if not bool(torch.isfinite(total)):
        raise CampaignRuntimeError("corrected teacher source loss is nonfinite")
    return total


__all__ = ["teacher_source_loss_v3"]
