from __future__ import annotations

import pytest
import torch

from pams.temporac.contract import CANONICAL_JOB_NAMES
from pams.temporac.runtime import JOB_SPECS, checkpoint_steps
from pams.temporac.training import (
    RESOURCE_LEDGER,
    CampaignResourceLedger,
    four_fresh_graph_optimizer_step,
    make_adamw,
    validate_completed_job,
)
from pams.temporac.types import CheckpointRecord, ContractError, ResourceRecord


def test_exactly_four_fresh_backward_calls_and_one_optimizer_step() -> None:
    module = torch.nn.Linear(2, 1, dtype=torch.float32)
    job = JOB_SPECS[CANONICAL_JOB_NAMES[0]]
    optimizer = make_adamw(module, job)
    hook_calls = 0

    def count_hook(gradient: torch.Tensor) -> torch.Tensor:
        nonlocal hook_calls
        hook_calls += 1
        return gradient

    handle = module.weight.register_hook(count_hook)
    inputs = tuple(torch.tensor([[float(index), 1.0]]) for index in range(1, 5))
    targets = tuple(torch.tensor([[float(index) / 3.0]]) for index in range(1, 5))
    factories = tuple(
        lambda value=value, target=target: torch.mean((module(value) - target).square())
        for value, target in zip(inputs, targets, strict=True)
    )
    before = tuple(parameter.detach().clone() for parameter in module.parameters())
    result = four_fresh_graph_optimizer_step(
        module,
        optimizer,
        factories,
        job=job,
        completed_step=1,
    )
    handle.remove()

    assert result.backward_calls == 4
    assert result.optimizer_steps == 1
    assert hook_calls == 4
    assert result.learning_rate == pytest.approx(2e-6)
    assert result.mean_loss == pytest.approx(sum(result.loss_values) / 4.0)
    assert any(
        not torch.equal(old, new) for old, new in zip(before, module.parameters(), strict=True)
    )
    assert all(parameter.grad is None for parameter in module.parameters())


def test_reused_or_wrong_graph_inventory_fails_before_optimizer_step() -> None:
    module = torch.nn.Linear(2, 1, dtype=torch.float32)
    job = JOB_SPECS[CANONICAL_JOB_NAMES[0]]
    optimizer = make_adamw(module, job)
    shared = module(torch.ones(1, 2)).square().mean()
    before = tuple(parameter.detach().clone() for parameter in module.parameters())
    with pytest.raises(ContractError, match="reused"):
        four_fresh_graph_optimizer_step(
            module,
            optimizer,
            (lambda: shared,) * 4,
            job=job,
            completed_step=1,
        )
    assert all(torch.equal(old, new) for old, new in zip(before, module.parameters(), strict=True))
    assert all(parameter.grad is None for parameter in module.parameters())
    with pytest.raises(ContractError, match="exactly four"):
        four_fresh_graph_optimizer_step(
            module,
            optimizer,
            (lambda: module(torch.ones(1, 2)).mean(),) * 3,
            job=job,
            completed_step=1,
        )


def test_checkpoint_tie_order_and_fixed_resource_ledger() -> None:
    job = JOB_SPECS[CANONICAL_JOB_NAMES[0]]
    candidates = tuple(
        CheckpointRecord(
            completed_step=step,
            artifact_sha256=f"{index + 1:064x}",
            tune_objective=float(0.25 if step in {500, 1_000} else 1.0 + index),
            resource=ResourceRecord(
                completed_steps=step,
                gpu_seconds=float(step),
                max_cuda_bytes=1024,
                wall_seconds=float(step),
            ),
        )
        for index, step in enumerate(checkpoint_steps(job))
    )
    final_resource = ResourceRecord(
        completed_steps=10_000,
        gpu_seconds=2.5 * 3600.0,
        max_cuda_bytes=24 * 1024**3,
        wall_seconds=2.5 * 3600.0,
    )
    assert validate_completed_job(job.name, candidates, final_resource).completed_step == 500
    with pytest.raises(ContractError, match="missing"):
        validate_completed_job(job.name, candidates[:-1], final_resource)

    assert RESOURCE_LEDGER.training_a6000_hours == 78.0
    assert RESOURCE_LEDGER.allocated_a6000_hours == 93.0
    assert RESOURCE_LEDGER.campaign_ceiling_a6000_hours == 100.0
    assert not RESOURCE_LEDGER.launch_authorized
    with pytest.raises(ContractError, match="differs"):
        CampaignResourceLedger(unallocated_a6000_hours=8.0)
