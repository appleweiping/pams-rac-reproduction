from __future__ import annotations

import torch

from pams.temporac.campaign_runtime_candidate_v5 import (
    teacher_source_loss_direction_blind_v5,
)
from pams.temporac.teacher import TempoRACTeacher


def test_direction_blind_phase_diagnostic_reaches_all_parameters() -> None:
    model = TempoRACTeacher()
    loss = teacher_source_loss_direction_blind_v5(
        model,
        0,
        device=torch.device("cpu"),
    )
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
