from __future__ import annotations

import pytest
import torch

from pams.warp_phase.model import (
    EXPECTED_TRAINABLE_PARAMETERS,
    WarpPhaseModel,
    count_trainable_parameters,
)


def _model_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(17)
    pose = torch.randn(1, 4, 17, 3, dtype=torch.float32)
    joint = torch.ones(1, 4, 17, dtype=torch.bool)
    frame = torch.tensor([[True, False, True, False]])
    clock = torch.tensor([[True, True, True, False]])
    joint[:, 3] = False
    pose[:, 3] = 0.0
    return pose, joint, frame, clock


def test_frozen_architecture_has_exact_parameter_receipt() -> None:
    model = WarpPhaseModel()

    assert model.conv1.in_channels == 68
    assert model.conv1.out_channels == 128
    assert model.conv1.kernel_size == (5,)
    assert model.conv2.in_channels == 128
    assert model.conv2.out_channels == 128
    assert model.gru.input_size == 128
    assert model.gru.hidden_size == 128
    assert model.phase_head.in_features == 128
    assert model.phase_head.out_features == 2
    assert count_trainable_parameters(model) == EXPECTED_TRAINABLE_PARAMETERS
    assert sum(parameter.numel() for parameter in model.conv1.parameters()) == 43_648
    assert sum(parameter.numel() for parameter in model.conv2.parameters()) == 82_048
    assert sum(parameter.numel() for parameter in model.gru.parameters()) == 99_072
    assert sum(parameter.numel() for parameter in model.phase_head.parameters()) == 258


def test_invalid_state_holds_and_every_nonpadding_clock_commits_once() -> None:
    model = WarpPhaseModel()
    pose, joint, frame, clock = _model_inputs()

    output = model(pose, joint, frame, clock)

    assert output.phase.shape == (1, 4, 2)
    assert output.hidden.shape == (1, 4, 128)
    assert output.final_state.shape == (1, 128)
    assert output.commit_count.dtype == torch.int64
    assert output.commit_count.tolist() == [3]
    assert torch.equal(output.hidden[:, 1], output.hidden[:, 0])
    assert torch.equal(output.hidden[:, 3], output.hidden[:, 2])
    assert torch.equal(output.final_state, output.hidden[:, -1])
    assert torch.equal(output.phase[:, 1], torch.zeros_like(output.phase[:, 1]))
    assert torch.equal(output.phase[:, 3], torch.zeros_like(output.phase[:, 3]))
    valid_norms = torch.linalg.vector_norm(output.phase[frame], dim=-1)
    assert torch.allclose(valid_norms, torch.ones_like(valid_norms), atol=1e-6, rtol=0.0)


def test_invalid_joint_pose_values_cannot_enter_the_68_channels() -> None:
    model = WarpPhaseModel().eval()
    pose, joint, frame, clock = _model_inputs()
    joint[:, 0, 0] = False
    first = pose.clone()
    second = pose.clone()
    first[:, 0, 0] = 0.0
    second[:, 0, 0] = 10_000.0

    with torch.no_grad():
        first_output = model(first, joint, frame, clock)
        second_output = model(second, joint, frame, clock)

    assert torch.equal(first_output.phase, second_output.phase)
    assert torch.equal(first_output.hidden, second_output.hidden)


def test_invalid_frame_pose_and_joint_channels_cannot_affect_valid_outputs() -> None:
    torch.manual_seed(20270815)
    model = WarpPhaseModel().eval()
    pose = torch.randn(2, 9, 17, 3, dtype=torch.float32)
    joint = torch.ones(2, 9, 17, dtype=torch.bool)
    frame = torch.ones(2, 9, dtype=torch.bool)
    clock = torch.ones(2, 9, dtype=torch.bool)
    frame[0, 3] = False
    frame[1, 5] = False

    changed = pose.clone()
    changed[0, 3] = torch.linspace(-1.0e6, 1.0e6, 51).reshape(17, 3)
    changed[1, 5] = torch.linspace(1.0e6, -1.0e6, 51).reshape(17, 3)

    with torch.no_grad():
        baseline = model(pose, joint, frame, clock)
        perturbed = model(changed, joint, frame, clock)

    valid = frame & clock
    assert torch.equal(baseline.phase[valid], perturbed.phase[valid])
    assert torch.equal(baseline.hidden[valid], perturbed.hidden[valid])
    assert count_trainable_parameters(model) == EXPECTED_TRAINABLE_PARAMETERS


def test_model_shape_and_dtype_contract_fails_closed() -> None:
    model = WarpPhaseModel()
    pose, joint, frame, clock = _model_inputs()

    with pytest.raises(TypeError, match="torch.float32"):
        model(pose.double(), joint, frame, clock)
    with pytest.raises(TypeError, match="torch.bool"):
        model(pose, joint.to(torch.uint8), frame, clock)
    bad_frame = frame.clone()
    bad_frame[:, 3] = True
    with pytest.raises(ValueError, match="padded clock"):
        model(pose, joint, bad_frame, clock)
