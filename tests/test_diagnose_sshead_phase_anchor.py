from __future__ import annotations

import math

import torch

from scripts.diagnose_sshead_phase_anchor import (
    phase_anchor_loss,
    pose_band_phase_teacher,
)


def test_pose_band_phase_teacher_recovers_signed_fundamental() -> None:
    time = 64
    phase = torch.arange(time) * (2.0 * math.pi / 8.0)
    projected = torch.zeros((1, time, 4))
    projected[0, :, 0] = 2.0 * torch.sin(phase)
    projected[0, :, 1] = 0.2 * torch.sin(phase)
    valid = torch.ones((1, time), dtype=torch.bool)

    teacher, available = pose_band_phase_teacher(
        projected,
        valid,
        torch.tensor([8.0]),
        torch.tensor([1.0]),
    )

    assert available.tolist() == [True]
    correlation = torch.corrcoef(torch.stack((teacher[0], torch.sin(phase))))[0, 1]
    assert abs(float(correlation)) > 0.99
    assert torch.isclose(
        teacher[0].std(unbiased=False),
        torch.tensor(1.0),
        atol=1e-5,
    )


def test_phase_anchor_has_nonzero_gradient_at_constant_stream() -> None:
    time = 64
    phase = torch.arange(time) * (2.0 * math.pi / 8.0)
    teacher = torch.sin(phase).unsqueeze(0)
    teacher = teacher / teacher.std(unbiased=False)
    stream = torch.zeros((1, time), requires_grad=True)

    output = phase_anchor_loss(
        stream,
        teacher,
        torch.ones((1, time), dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        torch.ones(1),
    )
    gradient = torch.autograd.grad(output.loss, stream)[0]

    assert torch.isclose(output.loss.detach(), torch.tensor(1.0))
    assert float(gradient.norm()) > 0.0
    assert int(torch.count_nonzero(gradient)) == time


def test_phase_anchor_continuously_weights_confidence() -> None:
    teacher = torch.tensor(
        [
            [-1.0, 1.0],
            [-1.0, 1.0],
        ]
    )
    stream = torch.tensor(
        [
            [-1.0, 1.0],
            [1.0, -1.0],
        ],
        requires_grad=True,
    )
    output = phase_anchor_loss(
        stream,
        teacher,
        torch.ones((2, 2), dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        torch.tensor([1.0, 0.1]),
    )

    assert torch.isclose(output.loss, torch.tensor(0.4 / 1.1))
    assert output.contributing_samples == 2
    assert math.isclose(output.confidence_sum, 1.1, rel_tol=1e-6)
