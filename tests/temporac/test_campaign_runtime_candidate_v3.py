from __future__ import annotations

import torch

from pams.temporac.campaign_runtime_candidate_v3 import teacher_source_loss_v3
from pams.temporac.teacher import TempoRACTeacher


def test_corrected_teacher_loss_is_finite_and_differentiable() -> None:
    model = TempoRACTeacher()
    loss = teacher_source_loss_v3(model, 0, device=torch.device("cpu"))
    assert bool(torch.isfinite(loss))
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
