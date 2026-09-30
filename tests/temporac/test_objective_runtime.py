from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from pams.temporac.contract import JOB_NAMES, SEEDS
from pams.temporac.runtime import (
    JOB_SPECS,
    RUNTIME_CONFIG,
    JobSpec,
    RngDomain,
    RuntimeConfig,
    checkpoint_steps,
    cycle_permutation,
    initialization_seed,
    initialize_module_cpu,
    learning_rate,
    select_checkpoint,
    torch_generator,
    validate_final_resource,
    validate_runtime_config,
)
from pams.temporac.types import CheckpointRecord, ContractError, ResourceRecord


def test_runtime_config_and_job_inventory_are_closed() -> None:
    assert RUNTIME_CONFIG.cpython == "3.12.4"
    assert RUNTIME_CONFIG.pytorch == "2.4.1+cu124"
    assert RUNTIME_CONFIG.cublas_workspace_config == ":4096:8"
    assert set(JOB_SPECS) == set(JOB_NAMES)
    assert len(JOB_SPECS) == 27
    assert sum(job.gpu_hour_ceiling for job in JOB_SPECS.values()) == 78.0
    assert validate_runtime_config(RUNTIME_CONFIG.as_dict()) is RUNTIME_CONFIG
    with pytest.raises(ContractError, match="unknown or missing"):
        validate_runtime_config({**RUNTIME_CONFIG.as_dict(), "extra": True})
    with pytest.raises(ContractError, match="differs"):
        RuntimeConfig(cpython="3.12.5")


def test_f22_learning_rate_and_checkpoint_schedule() -> None:
    assert learning_rate(1, total_steps=20_000, base_lr=3e-4) == pytest.approx(3e-7)
    assert learning_rate(1_000, total_steps=20_000, base_lr=3e-4) == pytest.approx(3e-4)
    assert learning_rate(20_000, total_steps=20_000, base_lr=3e-4) == pytest.approx(3e-5)
    assert len(checkpoint_steps(20_000)) == 40
    assert len(checkpoint_steps(10_000)) == 20
    with pytest.raises(ContractError):
        learning_rate(0, total_steps=20_000, base_lr=3e-4)


def test_rng_domains_are_deterministic_and_separate() -> None:
    job = "temporac.execution.v4/teacher/seed=20260815"
    first = cycle_permutation(
        RngDomain.TEACHER_SOURCE_CYCLE,
        job_name=job,
        seed=SEEDS[0],
        cycle=0,
        size=24,
    )
    second = cycle_permutation(
        RngDomain.TEACHER_SOURCE_CYCLE,
        job_name=job,
        seed=SEEDS[0],
        cycle=0,
        size=24,
    )
    other_job = "temporac.execution.v4/response/canonical/seed=20260815"
    other = cycle_permutation(
        RngDomain.RESPONSE_X0_CYCLE,
        job_name=other_job,
        seed=SEEDS[0],
        cycle=0,
        size=24,
    )
    assert first.dtype == np.dtype("<i8")
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    assert sorted(first.tolist()) == list(range(24))
    torch_left = torch_generator(
        RngDomain.TORCH_CPU, job_name=job, seed=SEEDS[0], fields=(b"batch",)
    )
    torch_right = torch_generator(
        RngDomain.TORCH_CPU, job_name=job, seed=SEEDS[0], fields=(b"batch",)
    )
    assert torch.equal(torch.rand(8, generator=torch_left), torch.rand(8, generator=torch_right))


def test_cpu_initialization_uses_one_isolated_generator_and_layernorm_constants() -> None:
    def model() -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.LayerNorm(3),
            torch.nn.Conv1d(1, 2, kernel_size=3),
        ).to(dtype=torch.float32, device="cpu")

    job = "temporac.execution.v4/response/canonical/seed=20260816"
    left = model()
    right = model()
    derived = initialize_module_cpu(left, job_name=job, seed=SEEDS[1])
    assert derived == initialization_seed(job, SEEDS[1])
    initialize_module_cpu(right, job_name=job, seed=SEEDS[1])
    assert all(
        torch.equal(left_value, right.state_dict()[name])
        for name, left_value in left.state_dict().items()
    )
    layer_norm = left[1]
    assert isinstance(layer_norm, torch.nn.LayerNorm)
    assert torch.equal(layer_norm.weight, torch.ones_like(layer_norm.weight))
    assert torch.equal(layer_norm.bias, torch.zeros_like(layer_norm.bias))


def test_checkpoint_selection_requires_complete_inventory_and_resource_caps() -> None:
    job = JOB_SPECS["temporac.execution.v4/teacher/seed=20260815"]
    candidates = tuple(
        CheckpointRecord(
            completed_step=step,
            artifact_sha256=f"{index + 1:064x}",
            tune_objective=float(0.5 if step in {500, 1_000} else 1.0 + index),
            resource=ResourceRecord(
                completed_steps=step,
                gpu_seconds=float(step),
                max_cuda_bytes=1024,
                wall_seconds=float(step),
            ),
        )
        for index, step in enumerate(checkpoint_steps(job))
    )
    assert select_checkpoint(job, candidates).completed_step == 500
    with pytest.raises(ContractError, match="missing"):
        select_checkpoint(job, candidates[:-1])
    final = ResourceRecord(
        completed_steps=20_000,
        gpu_seconds=float(6 * 3600),
        max_cuda_bytes=24 * 1024**3,
        wall_seconds=float(6 * 3600),
    )
    validate_final_resource(job, final)
    assert math.isfinite(final.gpu_seconds)


def test_job_spec_rejects_mutated_values() -> None:
    job = JOB_SPECS[JOB_NAMES[0]]
    with pytest.raises(ContractError, match="differs"):
        JobSpec(
            name=job.name,
            seed=job.seed,
            kind=job.kind,
            steps=job.steps - 1,
            base_lr=job.base_lr,
            checkpoint_interval=job.checkpoint_interval,
            checkpoint_count=job.checkpoint_count,
            gpu_hour_ceiling=job.gpu_hour_ceiling,
            cuda_byte_ceiling=job.cuda_byte_ceiling,
        )
