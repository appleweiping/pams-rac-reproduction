"""Fail-closed local training-step and fixed-inventory audit primitives.

This module does not launch jobs or open data.  It only validates already
materialized synthetic losses, optimizer state, checkpoints, and resource
records against ``temporac.execution.v4``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

from pams.temporac.contract import (
    ALLOCATED_A6000_HOURS,
    CAMPAIGN_A6000_HOUR_CEILING,
    HELDOUT_INFERENCE_A6000_HOURS,
    JOB_NAMES,
    MAX_CONCURRENCY,
    NATURAL_PREDICTION_A6000_HOURS,
    SEEDS,
    TRAINING_A6000_HOURS,
    UNALLOCATED_A6000_HOURS,
)
from pams.temporac.runtime import (
    JOB_SPECS,
    JobSpec,
    job_learning_rate,
    select_checkpoint,
    validate_final_resource,
)
from pams.temporac.types import CheckpointRecord, ContractError, ResourceRecord

LossFactory = Callable[[], Tensor]


@dataclass(frozen=True, slots=True)
class FreshGraphStepResult:
    """Audit witness for exactly four backward calls and one optimizer step."""

    completed_step: int
    learning_rate: float
    mean_loss: float
    loss_values: tuple[float, float, float, float]
    gradient_norm_before_clip: float
    backward_calls: int = 4
    optimizer_steps: int = 1

    def __post_init__(self) -> None:
        if self.completed_step <= 0 or self.backward_calls != 4 or self.optimizer_steps != 1:
            raise ContractError("fresh-graph step witness has an invalid call count")
        numeric = (
            self.learning_rate,
            self.mean_loss,
            self.gradient_norm_before_clip,
            *self.loss_values,
        )
        if any(not math.isfinite(value) for value in numeric):
            raise ContractError("fresh-graph step witness contains a nonfinite value")


def make_adamw(module: nn.Module, job: JobSpec) -> AdamW:
    """Create the exact all-parameter AdamW optimizer without taking a step."""

    parameters = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
    if not parameters:
        raise ContractError("training graph has no trainable parameter")
    if any(
        parameter.device.type != "cpu" or parameter.dtype != torch.float32
        for parameter in parameters
    ):
        raise ContractError("local optimizer construction requires float32 CPU parameters")
    return AdamW(
        parameters,
        lr=job_learning_rate(job, 1),
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        foreach=False,
        fused=False,
    )


def _validate_optimizer(module: nn.Module, optimizer: Optimizer) -> tuple[nn.Parameter, ...]:
    if not isinstance(optimizer, AdamW) or len(optimizer.param_groups) != 1:
        raise ContractError("v4 training requires one exact AdamW parameter group")
    group = optimizer.param_groups[0]
    if group["betas"] != (0.9, 0.999) or group["eps"] != 1e-8 or group["weight_decay"] != 1e-4:
        raise ContractError("AdamW hyperparameters differ from v4")
    expected = tuple(parameter for parameter in module.parameters() if parameter.requires_grad)
    observed = tuple(group["params"])
    if not expected or len(observed) != len(expected):
        raise ContractError("optimizer parameter inventory differs from the trainable graph")
    if {id(parameter) for parameter in observed} != {id(parameter) for parameter in expected}:
        raise ContractError("optimizer does not own every and only trainable parameter")
    if len({id(parameter) for parameter in observed}) != len(observed):
        raise ContractError("optimizer parameter inventory contains duplicates")
    return expected


def four_fresh_graph_optimizer_step(
    module: nn.Module,
    optimizer: Optimizer,
    loss_factories: Sequence[LossFactory],
    *,
    job: JobSpec,
    completed_step: int,
) -> FreshGraphStepResult:
    """Run four fresh scalar graphs, four backward calls, and one AdamW step.

    Each graph contributes one quarter of the accumulated gradient.  No graph
    is retained, and the global norm is clipped once to 1.0 immediately before
    the sole optimizer step.
    """

    if JOB_SPECS.get(job.name) != job:
        raise ContractError("training step requires an exact frozen JobSpec")
    if type(completed_step) is not int or not 1 <= completed_step <= job.steps:
        raise ContractError("completed step is outside the fixed job horizon")
    parameters = _validate_optimizer(module, optimizer)
    optimizer.zero_grad(set_to_none=True)
    factories = tuple(loss_factories)
    if len(factories) != 4 or any(not callable(factory) for factory in factories):
        raise ContractError("the local accumulation step requires exactly four loss factories")
    learning_rate = job_learning_rate(job, completed_step)
    optimizer.param_groups[0]["lr"] = learning_rate

    loss_tensors: list[Tensor] = []
    graph_nodes: list[object] = []
    loss_values: list[float] = []
    for factory in factories:
        loss = factory()
        if (
            not isinstance(loss, Tensor)
            or loss.ndim != 0
            or loss.dtype != torch.float32
            or loss.device.type != "cpu"
            or not loss.requires_grad
            or loss.grad_fn is None
            or not bool(torch.isfinite(loss).item())
        ):
            optimizer.zero_grad(set_to_none=True)
            raise ContractError("each fresh graph must emit one finite CPU float32 scalar")
        graph_node = loss.grad_fn
        if any(graph_node is previous for previous in graph_nodes):
            optimizer.zero_grad(set_to_none=True)
            raise ContractError("a loss factory reused an autograd graph")
        graph_nodes.append(graph_node)
        loss_tensors.append(loss)
        loss_values.append(float(loss.detach().item()))
        try:
            (loss / 4.0).backward()
        except RuntimeError as exc:
            optimizer.zero_grad(set_to_none=True)
            raise ContractError("a loss factory reused or invalidated an autograd graph") from exc

    gradients: list[Tensor] = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            optimizer.zero_grad(set_to_none=True)
            raise ContractError("fresh graphs did not reach every trainable parameter")
        gradients.append(gradient)
    if any(not bool(torch.all(torch.isfinite(gradient)).item()) for gradient in gradients):
        optimizer.zero_grad(set_to_none=True)
        raise ContractError("fresh graphs produced a nonfinite gradient")
    gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
    gradient_norm_value = float(gradient_norm.detach().item())
    if not math.isfinite(gradient_norm_value):
        optimizer.zero_grad(set_to_none=True)
        raise ContractError("global gradient norm is nonfinite")
    optimizer.step()
    if any(not bool(torch.all(torch.isfinite(parameter)).item()) for parameter in parameters):
        optimizer.zero_grad(set_to_none=True)
        raise ContractError("optimizer step produced a nonfinite parameter")
    values = tuple(loss_values)
    result = FreshGraphStepResult(
        completed_step=completed_step,
        learning_rate=learning_rate,
        mean_loss=sum(values) / 4.0,
        loss_values=(values[0], values[1], values[2], values[3]),
        gradient_norm_before_clip=gradient_norm_value,
    )
    optimizer.zero_grad(set_to_none=True)
    return result


@dataclass(frozen=True, slots=True)
class CampaignResourceLedger:
    """The immutable 93-hour allocation; it is not launch authorization."""

    training_a6000_hours: float = TRAINING_A6000_HOURS
    heldout_inference_a6000_hours: float = HELDOUT_INFERENCE_A6000_HOURS
    natural_prediction_a6000_hours: float = NATURAL_PREDICTION_A6000_HOURS
    allocated_a6000_hours: float = ALLOCATED_A6000_HOURS
    unallocated_a6000_hours: float = UNALLOCATED_A6000_HOURS
    campaign_ceiling_a6000_hours: float = CAMPAIGN_A6000_HOUR_CEILING
    max_concurrency: int = MAX_CONCURRENCY
    launch_authorized: bool = False

    def __post_init__(self) -> None:
        expected = (78.0, 3.0, 12.0, 93.0, 7.0, 100.0, 2, False)
        observed = (
            self.training_a6000_hours,
            self.heldout_inference_a6000_hours,
            self.natural_prediction_a6000_hours,
            self.allocated_a6000_hours,
            self.unallocated_a6000_hours,
            self.campaign_ceiling_a6000_hours,
            self.max_concurrency,
            self.launch_authorized,
        )
        if observed != expected:
            raise ContractError("campaign resource ledger differs from v4")
        if self.training_a6000_hours != sum(spec.gpu_hour_ceiling for spec in JOB_SPECS.values()):
            raise ContractError("job ceilings do not sum to the training ledger")
        if (
            self.training_a6000_hours
            + self.heldout_inference_a6000_hours
            + self.natural_prediction_a6000_hours
            != self.allocated_a6000_hours
            or self.allocated_a6000_hours + self.unallocated_a6000_hours
            != self.campaign_ceiling_a6000_hours
        ):
            raise ContractError("campaign resource classes do not sum exactly")


RESOURCE_LEDGER = CampaignResourceLedger()


def validate_completed_job(
    job_name: str,
    candidates: Sequence[CheckpointRecord],
    final_resource: ResourceRecord,
) -> CheckpointRecord:
    """Validate complete checkpoints/resources and apply tune-only tie order."""

    if job_name not in JOB_NAMES:
        raise ContractError("completed job name is outside the 27-job inventory")
    job = JOB_SPECS[job_name]
    if job.seed not in SEEDS:
        raise ContractError("completed job seed is outside the three-seed inventory")
    ordered = tuple(candidates)
    selected = select_checkpoint(job, ordered)
    validate_final_resource(job, final_resource)
    if ordered[-1].resource.completed_steps != final_resource.completed_steps:
        raise ContractError("final checkpoint and final resource step differ")
    return selected


__all__ = [
    "CampaignResourceLedger",
    "FreshGraphStepResult",
    "RESOURCE_LEDGER",
    "four_fresh_graph_optimizer_step",
    "make_adamw",
    "validate_completed_job",
]
