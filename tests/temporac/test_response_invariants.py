"""Cached causal response, shortcut, capacity, and F14 invariants."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pams.temporac.objective import (
    balanced_family_step_loss,
    balanced_response_functional,
    equal_block_reduction,
    responsibility_fraction,
    traversal_unit_loss,
)
from pams.temporac.response import (
    CapacityControl,
    ExpertBranch,
    ResponseContractError,
    ShortcutAdapter,
    TempoRACResponse,
    no_pose_timestamp_vector,
    pose_shuffle_raw,
    shortcut_projection,
)


def test_single_expert_class_equal_shapes_and_receptive_fields() -> None:
    model = TempoRACResponse()
    assert all(type(branch) is ExpertBranch for branch in model.experts.values())
    assert tuple(model.experts[name].dilation for name in model.branch_order) == (4, 2, 1)
    assert tuple(model.experts[name].receptive_field for name in model.branch_order) == (33, 17, 9)
    shapes = model.branch_parameter_shapes()
    assert shapes[0] == shapes[1] == shapes[2]
    assert not any("gru" in type(module).__name__.lower() for module in model.modules())


def test_cached_logits_t_plus_one_run_reset_and_identity_isolation() -> None:
    torch.manual_seed(20260815)
    model = TempoRACResponse().eval()
    value = torch.randn(2, 60, 269)
    bounds = ((0, 20), (21, 59))
    with torch.no_grad():
        baseline = model.cached_responses(value, bounds)
        changed_run = value.clone()
        changed_run[:, :21] = torch.randn_like(changed_run[:, :21]) * 100.0
        after_run_change = model.cached_responses(changed_run, bounds)
        changed_identity = value.clone()
        changed_identity[0] = torch.randn_like(changed_identity[0]) * 100.0
        after_identity_change = model.cached_responses(changed_identity, bounds)

    assert baseline.sample_logits.shape == (2, 60, 3)
    assert baseline.edge_logits.shape == (2, 59, 3)
    assert torch.equal(baseline.edge_logits, baseline.sample_logits[:, 1:, :])
    assert torch.equal(baseline.sample_logits[:, 21:], after_run_change.sample_logits[:, 21:])
    assert torch.equal(
        baseline.edge_probabilities[:, 20:],
        after_run_change.edge_probabilities[:, 20:],
    )
    assert torch.equal(baseline.sample_logits[1], after_identity_change.sample_logits[1])
    assert torch.equal(baseline.edge_probabilities[1], after_identity_change.edge_probabilities[1])


def test_single_full_edge_run_matches_direct_logits_through_terminal_edge() -> None:
    torch.manual_seed(20260815)
    model = TempoRACResponse().eval()
    value = torch.randn(18, 269)
    with torch.no_grad():
        direct = model._run_logits(value.unsqueeze(0)).squeeze(0)
        implicit = model.cached_responses(value)
        explicit = model.cached_responses(value, ((0, 17),))

    torch.testing.assert_close(implicit.sample_logits, direct, rtol=0.0, atol=0.0)
    torch.testing.assert_close(explicit.sample_logits, direct, rtol=0.0, atol=0.0)
    torch.testing.assert_close(explicit.edge_logits, direct[1:], rtol=0.0, atol=0.0)
    torch.testing.assert_close(explicit.edge_logits[-1], direct[-1], rtol=0.0, atol=0.0)


def test_multiple_separated_edge_runs_match_direct_run_caches() -> None:
    torch.manual_seed(20260815)
    model = TempoRACResponse().eval()
    value = torch.randn(20, 269)
    bounds = ((1, 6), (8, 19))
    with torch.no_grad():
        cached = model.cached_responses(value, bounds)
        first = model._run_logits(value[None, 1:7]).squeeze(0)
        second = model._run_logits(value[None, 8:20]).squeeze(0)

    expected_samples = torch.zeros_like(cached.sample_logits)
    expected_samples[1:7] = first
    expected_samples[8:20] = second
    torch.testing.assert_close(cached.sample_logits, expected_samples, rtol=0.0, atol=0.0)
    torch.testing.assert_close(cached.edge_logits[1:6], first[1:], rtol=0.0, atol=0.0)
    torch.testing.assert_close(cached.edge_logits[8:19], second[1:], rtol=0.0, atol=0.0)


def test_response_boundary_rejects_adjacent_runs_non_float32_and_nonfinite() -> None:
    model = TempoRACResponse().eval()
    value = torch.zeros(10, 269, dtype=torch.float32)

    with pytest.raises(ResponseContractError, match="nonadjacent"):
        model.cached_responses(value, ((0, 4), (4, 9)))
    with pytest.raises(ResponseContractError, match="out of range"):
        model.cached_responses(value, ((0, 10),))
    with pytest.raises(ResponseContractError, match="dtype float32"):
        model.cached_responses(value.to(torch.float64))
    for invalid in (float("nan"), float("inf"), -float("inf")):
        changed = value.clone()
        changed[3, 7] = invalid
        with pytest.raises(ResponseContractError, match="finite"):
            model.cached_responses(changed)


def test_capacity_probability_mean_and_shortcut_determinism() -> None:
    torch.manual_seed(20260815)
    capacity = CapacityControl().eval()
    value = torch.randn(24, 269)
    with torch.no_grad():
        cached = capacity.cached_responses(value)
        fused = capacity.fused_response(value)
    torch.testing.assert_close(
        fused,
        torch.mean(cached.edge_probabilities, dim=-1),
        rtol=0.0,
        atol=0.0,
    )

    projection = shortcut_projection("no-pose-timestamp")
    assert projection.shape == (269, 20)
    assert not projection.flags.writeable
    assert np.linalg.norm(projection, axis=1) == pytest.approx(np.ones(269), abs=2e-7)
    adapter = ShortcutAdapter("no-pose-timestamp")
    raw = torch.randn(2, 7, 20)
    assert adapter(raw).shape == (2, 7, 269)

    response_input = np.arange(12 * 269, dtype=np.float32).reshape(12, 269)
    source_key = bytes(range(32))
    bounds = ((0, 5), (5, 12))
    shuffled = pose_shuffle_raw(response_input, source_key, bounds)
    assert np.array_equal(shuffled, pose_shuffle_raw(response_input, source_key, bounds))
    changed = response_input.copy()
    changed[:5] *= -3.0
    changed_shuffled = pose_shuffle_raw(changed, source_key, bounds)
    assert np.array_equal(shuffled[5:], changed_shuffled[5:])

    timestamp = no_pose_timestamp_vector(
        np.ones((12, 17), dtype=np.bool_),
        np.ones(12, dtype=np.bool_),
        np.linspace(0.0, 1.0, 12, dtype=np.float32),
        np.full(12, 2.0, dtype=np.float32),
    )
    assert timestamp.shape == (12, 20)
    assert timestamp.dtype == np.float32


def test_f14_balancing_equal_reductions_and_gradients() -> None:
    edge_count = 64
    logits = torch.zeros(edge_count, 3, requires_grad=True)
    branch_probability = torch.sigmoid(logits)
    fused_probability = torch.mean(branch_probability, dim=1)
    pulse = torch.zeros(edge_count)
    pulse[0] = 1.0
    pulse[32] = 1.0
    edge_mask = torch.ones(edge_count)
    chi = torch.full((edge_count,), 1.0 / 32.0)
    responsibility = torch.softmax(
        torch.stack(
            (
                torch.linspace(-1.0, 1.0, edge_count),
                torch.zeros(edge_count),
                torch.linspace(1.0, -1.0, edge_count),
            ),
            dim=1,
        ),
        dim=1,
    )
    bounds = ((0, 32), (32, 64))
    loss = traversal_unit_loss(
        branch_probability,
        fused_probability,
        pulse,
        edge_mask,
        chi,
        responsibility,
        bounds,
    )
    assert loss.traversal_count == 2
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert logits.grad is not None
    assert all(float(torch.linalg.vector_norm(logits.grad[:, index])) > 0.0 for index in range(3))

    fractions = responsibility_fraction(responsibility, edge_mask, chi)
    assert torch.sum(fractions).item() == pytest.approx(1.0, abs=2e-7)
    block_mean = equal_block_reduction((loss, loss))
    assert block_mean.item() == pytest.approx(loss.total.item())
    family = balanced_family_step_loss(block_mean, 2.0 * block_mean)
    assert family.item() == pytest.approx(1.5 * block_mean.item())

    ideal = torch.full((10,), 0.1)
    ideal[0] = 0.9
    functional = balanced_response_functional(
        ideal,
        torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        torch.ones(10),
    )
    assert functional.balanced_bce.item() == pytest.approx(-np.log(0.9), abs=2e-7)
    assert functional.positive_margin.item() == 0.0
    assert functional.negative_valley.item() == 0.0
