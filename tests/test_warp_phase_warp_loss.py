from __future__ import annotations

import math

import pytest
import torch

from pams.warp_phase.losses import (
    clean_phase_integral,
    matched_phase_objective,
    warp_integral_loss,
)
from pams.warp_phase.warp import (
    identity_time_map,
    oriented_overlap_integral,
    three_segment_time_map,
    warp_track,
)


def _source_track() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    clocks = torch.tensor([0, 2, 4], dtype=torch.int64)
    pose = torch.zeros(3, 17, 3, dtype=torch.float32)
    for index in range(3):
        pose[index, :, 0] = float(index * 2)
        pose[index, :, 1] = float(index * 4)
        pose[index, :, 2] = 0.5 + 0.1 * index
    joint = torch.ones(3, 17, dtype=torch.bool)
    frame = torch.ones(3, dtype=torch.bool)
    cell = torch.ones(2, dtype=torch.bool)
    return pose, clocks, joint, frame, cell


def _phase(angles: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)


def test_identity_and_same_cell_warp_interpolation() -> None:
    pose, clocks, joint, frame, cell = _source_track()
    identity = warp_track(pose, clocks, joint, frame, cell, identity_time_map(clocks))

    assert torch.equal(identity.pose, pose)
    assert torch.equal(identity.joint_mask, joint)
    assert torch.equal(identity.frame_mask, frame)
    assert torch.equal(identity.cell_mask, cell)

    query = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    interpolated = warp_track(pose, clocks, joint, frame, cell, query)
    assert torch.equal(interpolated.pose[1], (pose[0] + pose[1]) / 2.0)
    assert interpolated.joint_mask[1].all()
    assert interpolated.frame_mask[1]
    assert interpolated.cell_mask.tolist() == [True, True]


def test_interpolation_and_target_cells_fail_closed_on_invalid_source_cell() -> None:
    pose, clocks, joint, frame, cell = _source_track()
    cell[0] = False
    query = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)

    warped = warp_track(pose, clocks, joint, frame, cell, query)

    assert not warped.joint_mask[1].any()
    assert torch.equal(warped.pose[1], torch.zeros_like(warped.pose[1]))
    assert warped.cell_mask.tolist() == [False, False]


def test_three_segment_map_has_exact_pause_and_global_reversal() -> None:
    target = torch.tensor([0.0, 2.5, 5.0, 7.5, 10.0], dtype=torch.float64)
    breakpoints = torch.tensor([0.25, 0.75], dtype=torch.float64)
    slopes = torch.tensor([2.0, 0.0, 2.0], dtype=torch.float64)

    forward = three_segment_time_map(
        target,
        target_start=0.0,
        target_end=10.0,
        source_start=0.0,
        source_end=10.0,
        breakpoints=breakpoints,
        slopes=slopes,
    )
    reversal = three_segment_time_map(
        target,
        target_start=0.0,
        target_end=10.0,
        source_start=0.0,
        source_end=10.0,
        breakpoints=breakpoints,
        slopes=slopes,
        reverse=True,
    )

    assert forward.tolist() == [0.0, 5.0, 5.0, 5.0, 10.0]
    assert torch.equal(reversal, 10.0 - forward)


def test_oriented_overlap_is_signed_with_pause_zero_and_no_partial_repair() -> None:
    clocks = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    increments = torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64)
    cells = torch.ones(3, dtype=torch.bool)
    query = torch.tensor([0.5, 2.5, 0.5, 0.5], dtype=torch.float64)

    integral = oriented_overlap_integral(clocks, increments, cells, query)

    assert integral.valid.tolist() == [True, True, True]
    assert integral.value.tolist() == pytest.approx([0.8, -0.8, 0.0])
    assert integral.required_clean_cells.tolist() == [3, 3, 0]

    cells[1] = False
    invalid = oriented_overlap_integral(
        clocks,
        increments,
        cells,
        torch.tensor([0.5, 2.5], dtype=torch.float64),
    )
    assert invalid.valid.tolist() == [False]
    assert invalid.value.tolist() == [0.0]


def test_clean_integral_is_stop_gradient_and_alias_fails_closed() -> None:
    angles = torch.tensor([0.0, 0.2, 0.6, 1.2], requires_grad=True)
    phase = _phase(angles)
    clocks = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
    cells = torch.ones(3, dtype=torch.bool)
    target = clean_phase_integral(
        phase,
        clocks,
        cells,
        torch.tensor([0.5, 2.5], dtype=torch.float64),
    )
    assert target.valid.tolist() == [True]
    assert target.value.tolist() == pytest.approx([0.8])
    assert not target.value.requires_grad

    alias = oriented_overlap_integral(
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([math.pi * 0.95], dtype=torch.float64),
        torch.ones(1, dtype=torch.bool),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )
    assert alias.valid.tolist() == [False]


def test_warp_integral_loss_matches_forward_reversal_and_pause() -> None:
    increments = 0.05
    forward_angles = torch.arange(65, dtype=torch.float32) * increments
    reverse_angles = -forward_angles
    pause_angles = torch.zeros_like(forward_angles)
    warped_phase = torch.stack(
        (_phase(forward_angles), _phase(reverse_angles), _phase(pause_angles))
    )
    integral = torch.stack(
        (
            torch.full((64,), increments, dtype=torch.float64),
            torch.full((64,), -increments, dtype=torch.float64),
            torch.zeros(64, dtype=torch.float64),
        )
    )
    valid = torch.ones(3, 64, dtype=torch.bool)
    clocks = torch.ones(3, 65, dtype=torch.bool)

    loss = warp_integral_loss(warped_phase, integral, valid, valid, clocks)

    assert loss.item() == pytest.approx(0.0, abs=1e-10)


def test_warp_loss_coverage_and_unique_treatment_deletion_fail_closed() -> None:
    angles = torch.arange(65, dtype=torch.float32) * 0.05
    phase = _phase(angles).unsqueeze(0)
    integral = torch.full((1, 64), 0.05, dtype=torch.float64)
    valid = torch.ones(1, 64, dtype=torch.bool)
    valid[:, :20] = False
    integral[:, :20] = 0.0
    clock = torch.ones(1, 65, dtype=torch.bool)
    with pytest.raises(ValueError, match="coverage"):
        warp_integral_loss(phase, integral, valid, torch.ones_like(valid), clock)

    pams = torch.tensor(1.25)
    warp = torch.tensor(0.75)
    control = matched_phase_objective(pams, arm="control")
    treatment = matched_phase_objective(pams, arm="treatment", warp_integral=warp)
    assert control.total.item() == pytest.approx(1.25)
    assert control.warp_integral is None
    assert treatment.total.item() == pytest.approx(2.0)
    with pytest.raises(ValueError, match="control"):
        matched_phase_objective(pams, arm="control", warp_integral=warp)
