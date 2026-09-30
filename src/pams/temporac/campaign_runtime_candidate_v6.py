"""Certificate-aligned X0 teacher objective diagnostic.

This candidate fixes the discovered target-domain mismatch by training the
reconstruction head against the exact canonical continuous arrays and masks
that the frozen X0 certificate later evaluates.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from torch import Tensor

from pams.temporac.campaign_runtime_candidate_v1 import (
    CampaignRuntimeError,
    _masked_huber,
    _phase_at_coordinate,
    _teacher_view,
)
from pams.temporac.campaign_runtime_candidate_v5 import DirectionBlindTeacherAdapter
from pams.temporac.certify import certificate_track_from_x0
from pams.temporac.teacher import TempoRACTeacher, phase_increments


@lru_cache(maxsize=224)
def _certificate_target(
    source_id: int,
    block_index: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    row = _teacher_view(source_id, block_index, device)
    fixture = certificate_track_from_x0(row.view, teacher_sha256="0" * 64)
    target = torch.from_numpy(np.array(fixture.continuous, dtype=np.float32, copy=True)).to(device)
    mask = torch.from_numpy(np.array(fixture.continuous_mask, dtype=np.float32, copy=True)).to(
        device
    )
    return target, mask


def teacher_source_loss_certificate_aligned_v6(
    model: TempoRACTeacher,
    source_id: int,
    *,
    device: torch.device,
    anchor_weight: float = 0.1,
) -> Tensor:
    """Train against the exact X0 certificate target domain."""

    if not 0 <= source_id < 32:
        raise CampaignRuntimeError("teacher source must be train/tune X0")
    if anchor_weight <= 0.0:
        raise CampaignRuntimeError("phase anchor weight must be positive")
    views = [_teacher_view(source_id, block, device) for block in range(7)]
    targets_and_masks = [_certificate_target(source_id, block, device) for block in range(7)]
    lengths = [int(row.teacher_input.shape[0]) for row in views]
    maximum = max(lengths)
    teacher_batch = torch.zeros((7, maximum, 215), dtype=torch.float32, device=device)
    static_batch = torch.stack([row.static_features for row in views])
    for index, row in enumerate(views):
        teacher_batch[index, : lengths[index]] = row.teacher_input
    adapter = DirectionBlindTeacherAdapter(model)
    _static, phase_batch, reconstruction_batch = adapter(teacher_batch, static_batch)

    phases: list[Tensor] = []
    block_reconstruction: list[Tensor] = []
    block_orientation: list[Tensor] = []
    block_alias: list[Tensor] = []
    phase_anchor: list[Tensor] = []
    for index, row in enumerate(views):
        length = lengths[index]
        phase = phase_batch[index, :length]
        reconstruction = reconstruction_batch[index, :length]
        target, mask = targets_and_masks[index]
        phases.append(phase)
        analytic_phase = torch.from_numpy(
            np.array(row.view.phase, dtype=np.float32, copy=True)
        ).to(device)
        phase_anchor.append(
            torch.mean(1.0 - torch.sum(phase * analytic_phase, dim=1))
        )
        traversal_reconstruction: list[Tensor] = []
        traversal_orientation: list[Tensor] = []
        traversal_alias: list[Tensor] = []
        for left_raw, right_raw in row.view.traversal_bounds:
            left, right = int(left_raw), int(right_raw)
            traversal_reconstruction.append(
                _masked_huber(
                    reconstruction[left:right],
                    target[left:right],
                    mask[left:right],
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
        + anchor_weight * torch.mean(torch.stack(phase_anchor))
    )
    if not bool(torch.isfinite(total)):
        raise CampaignRuntimeError("certificate-aligned teacher loss is nonfinite")
    return total


__all__ = ["teacher_source_loss_certificate_aligned_v6"]
