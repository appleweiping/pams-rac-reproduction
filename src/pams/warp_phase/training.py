"""Fresh-graph, four-step WARP-PHASE gradient accumulation primitives."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

from pams.warp_phase.model import (
    EXPECTED_TRAINABLE_PARAMETERS,
    WarpPhaseModel,
    count_trainable_parameters,
)

IDENTITIES_PER_FRESH_STEP = 8
FRESH_STEPS_PER_UPDATE = 4
IDENTITY_DRAWS_PER_UPDATE = IDENTITIES_PER_FRESH_STEP * FRESH_STEPS_PER_UPDATE
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
MAX_GRADIENT_NORM = 1.0
FULL_UPDATE_LIMIT = 20_000
TRAINING_AUTHORIZED = False


@dataclass(frozen=True)
class FreshStepLoss:
    """One identity-averaged scalar produced by one newly built graph."""

    loss: Tensor
    identity_count: int


class FreshStepFactory(Protocol):
    """Build one fresh complete-track graph and its sole matched objective."""

    def __call__(self, model: WarpPhaseModel) -> FreshStepLoss: ...


@dataclass(frozen=True)
class OptimizerUpdateReceipt:
    """Deterministic execution counts and finite scalar diagnostics."""

    unscaled_losses: tuple[float, float, float, float]
    gradient_norm_before_clip: float
    backward_calls: int
    optimizer_steps: int
    identity_draws: int


@dataclass(frozen=True)
class TrainingCommandReceipt:
    """Feature-only command preflight that can never authorize this pilot."""

    status: str
    checks: dict[str, bool]
    bindings: dict[str, str]
    blockers: tuple[str, ...]
    authorizes: tuple[str, ...]


def _require_sha256(value: str, *, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def training_command_preflight(
    *,
    features_root: Path,
    split: str,
    environment_lock_sha256: str,
    output_space_receipt_sha256: str,
) -> TrainingCommandReceipt:
    """Inspect only the seven-field feature split and return a blocked plan.

    The command accepts no privileged data path and imports only the feature
    reader.  It intentionally has no training launch transition: the frozen
    pilot configuration says ``training_authorized: false`` and Gate 0--3 pass
    receipts do not yet exist.
    """

    from pams.warp_phase.data import load_feature_split

    environment_hash = _require_sha256(
        environment_lock_sha256,
        name="environment_lock_sha256",
    )
    output_hash = _require_sha256(
        output_space_receipt_sha256,
        name="output_space_receipt_sha256",
    )
    shards = load_feature_split(Path(features_root), split)
    inventory = [
        {
            "local_person_slot": shard.local_person_slot,
            "opaque_sample_key": shard.opaque_sample_key,
            "shard_sha256": shard.shard_sha256,
        }
        for shard in shards
    ]
    if any(row["shard_sha256"] is None for row in inventory):
        raise ValueError("every loaded feature shard requires a verified SHA-256")
    encoded_inventory = json.dumps(
        inventory,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return TrainingCommandReceipt(
        status="BLOCKED",
        checks={
            "feature_only_split_verified": True,
            "environment_lock_verified_by_cli": True,
            "empty_output_space_verified_by_cli": True,
            "training_authorized": TRAINING_AUTHORIZED,
        },
        bindings={
            "environment_lock_sha256": environment_hash,
            "feature_identity_count": str(len(shards)),
            "feature_inventory_sha256": hashlib.sha256(encoded_inventory).hexdigest(),
            "features_root": str(Path(features_root).resolve(strict=True)),
            "output_space_receipt_sha256": output_hash,
            "split": split,
        },
        blockers=(
            "training_authorized_false",
            "gate0_gate1_gate2_gate3_pass_chain_missing",
            "end_to_end_fresh_step_factory_not_frozen",
        ),
        authorizes=(),
    )


def build_optimizer(model: WarpPhaseModel) -> AdamW:
    """Create the sole frozen AdamW optimizer for the phase learner."""

    if not isinstance(model, WarpPhaseModel):
        raise TypeError("model must be a WarpPhaseModel")
    actual = count_trainable_parameters(model)
    if actual != EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError(
            f"model must have exactly {EXPECTED_TRAINABLE_PARAMETERS} trainable parameters"
        )
    return AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def _validate_optimizer(model: nn.Module, optimizer: Optimizer) -> None:
    if not isinstance(optimizer, AdamW):
        raise TypeError("optimizer must be torch.optim.AdamW")
    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_parameters: list[Tensor] = []
    for group in optimizer.param_groups:
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        if learning_rate != LEARNING_RATE or weight_decay != WEIGHT_DECAY:
            raise ValueError("AdamW learning rate or weight decay differs from the frozen contract")
        parameters = group["params"]
        if not isinstance(parameters, list):
            raise TypeError("optimizer parameter groups must contain concrete parameter lists")
        for parameter in parameters:
            if not isinstance(parameter, Tensor):
                raise TypeError("optimizer parameters must be tensors")
            optimizer_parameters.append(parameter)
    if len({id(parameter) for parameter in optimizer_parameters}) != len(optimizer_parameters):
        raise ValueError("optimizer must not contain duplicate parameters")
    if {id(parameter) for parameter in optimizer_parameters} != {
        id(parameter) for parameter in model_parameters
    }:
        raise ValueError("optimizer parameters must exactly match the trainable model parameters")


def _validate_fresh_result(result: FreshStepLoss, model: WarpPhaseModel) -> Tensor:
    if not isinstance(result, FreshStepLoss):
        raise TypeError("each fresh-step factory must return FreshStepLoss")
    if (
        isinstance(result.identity_count, bool)
        or not isinstance(result.identity_count, int)
        or result.identity_count != IDENTITIES_PER_FRESH_STEP
    ):
        raise ValueError(
            f"each fresh step must contain exactly {IDENTITIES_PER_FRESH_STEP} complete identities"
        )
    loss = result.loss
    if not isinstance(loss, Tensor):
        raise TypeError("fresh-step loss must be a torch.Tensor")
    if loss.ndim != 0:
        raise ValueError("fresh-step loss must be one scalar")
    if loss.dtype not in {torch.float32, torch.float64}:
        raise TypeError("fresh-step loss must have dtype torch.float32 or torch.float64")
    if not bool(torch.isfinite(loss)):
        raise ValueError("fresh-step loss must be finite")
    if not loss.requires_grad or loss.grad_fn is None:
        raise ValueError("fresh-step loss must belong to a newly built autograd graph")
    parameter_devices = {parameter.device for parameter in model.parameters()}
    if len(parameter_devices) != 1 or loss.device not in parameter_devices:
        raise ValueError("fresh-step loss and model must share one device")
    return loss


def _assert_finite_gradients(model: nn.Module) -> None:
    found = False
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        found = True
        if not bool(torch.isfinite(parameter.grad).all()):
            raise FloatingPointError("a model gradient is non-finite")
    if not found:
        raise RuntimeError("fresh-step backward produced no parameter gradients")


def run_optimizer_update(
    model: WarpPhaseModel,
    optimizer: Optimizer,
    fresh_steps: Sequence[FreshStepFactory],
) -> OptimizerUpdateReceipt:
    """Run exactly four fresh graphs/backwards followed by one clipped update.

    A factory is invoked only when its step begins.  Its complete graph is
    backwarded once without ``retain_graph`` and is then dropped; only parameter
    gradients survive to the next factory.  No activation, recurrent state,
    decoded value, or graph cache is accepted by this interface.
    """

    if not isinstance(model, WarpPhaseModel):
        raise TypeError("model must be a WarpPhaseModel")
    if count_trainable_parameters(model) != EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError("model parameter count differs from the frozen contract")
    if not isinstance(fresh_steps, Sequence):
        raise TypeError("fresh_steps must be a sequence of factories")
    if len(fresh_steps) != FRESH_STEPS_PER_UPDATE:
        raise ValueError(
            f"each optimizer update requires exactly {FRESH_STEPS_PER_UPDATE} fresh steps"
        )
    _validate_optimizer(model, optimizer)

    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    backward_calls = 0
    try:
        for factory in fresh_steps:
            if not callable(factory):
                raise TypeError("each fresh step must be callable")
            result = factory(model)
            loss = _validate_fresh_result(result, model)
            losses.append(float(loss.detach().item()))
            scaled_loss = loss / FRESH_STEPS_PER_UPDATE
            scaled_loss.backward()
            backward_calls += 1
            _assert_finite_gradients(model)
            del scaled_loss, loss, result

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=MAX_GRADIENT_NORM,
        )
        gradient_norm_value = float(gradient_norm.item())
        if not math.isfinite(gradient_norm_value):
            raise FloatingPointError("accumulated gradient norm is non-finite")
        optimizer.step()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise
    optimizer.zero_grad(set_to_none=True)

    if len(losses) != FRESH_STEPS_PER_UPDATE or backward_calls != FRESH_STEPS_PER_UPDATE:
        raise RuntimeError("fresh-step accumulation receipt is incomplete")
    return OptimizerUpdateReceipt(
        unscaled_losses=(losses[0], losses[1], losses[2], losses[3]),
        gradient_norm_before_clip=gradient_norm_value,
        backward_calls=backward_calls,
        optimizer_steps=1,
        identity_draws=IDENTITY_DRAWS_PER_UPDATE,
    )
