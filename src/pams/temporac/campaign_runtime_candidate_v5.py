"""Direction-blind phase-input diagnostic for exact return consistency."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from pams.temporac.campaign_runtime_candidate_v4 import (
    teacher_source_loss_phase_anchored_v4,
)
from pams.temporac.teacher import TempoRACTeacher


class DirectionBlindTeacherAdapter(nn.Module):
    """Reuse an unchanged teacher while masking phase-direction channels."""

    def __init__(self, base: TempoRACTeacher) -> None:
        super().__init__()
        self.base = base

    def forward(
        self,
        teacher_input: Tensor,
        static_features: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        phase_input = teacher_input.clone()
        phase_input[..., 66:132] = 0.0
        phase_input[..., 182:215] = 0.0
        static_code = self.base.static(static_features)
        phase = self.base.phase(phase_input, static_code)
        reconstruction = self.base.reconstruction(phase, static_code)
        return static_code, phase, reconstruction


def teacher_source_loss_direction_blind_v5(
    model: TempoRACTeacher,
    source_id: int,
    *,
    device: torch.device,
    anchor_weight: float = 10.0,
) -> Tensor:
    """Run the phase-anchored diagnostic through a direction-blind adapter."""

    adapter = DirectionBlindTeacherAdapter(model)
    return teacher_source_loss_phase_anchored_v4(
        adapter,  # type: ignore[arg-type]
        source_id,
        device=device,
        anchor_weight=anchor_weight,
    )


__all__ = ["DirectionBlindTeacherAdapter", "teacher_source_loss_direction_blind_v5"]
