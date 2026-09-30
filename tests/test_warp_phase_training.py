from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.optim import AdamW

from pams.warp_phase import packing
from pams.warp_phase.cli import train
from pams.warp_phase.model import WarpPhaseModel
from pams.warp_phase.training import (
    FRESH_STEPS_PER_UPDATE,
    IDENTITIES_PER_FRESH_STEP,
    LEARNING_RATE,
    WEIGHT_DECAY,
    FreshStepLoss,
    build_optimizer,
    run_optimizer_update,
    training_command_preflight,
)
from pams.warp_phase.types import FeatureShard


class CountingAdamW(AdamW):
    step_calls: int

    def __init__(self, parameters: list[torch.nn.Parameter]) -> None:
        super().__init__(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.step_calls = 0

    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override]
        self.step_calls += 1
        result = super().step(closure)
        if result is None:
            return None
        return float(result)


def _fresh_factory(seed: int, *, identities: int = IDENTITIES_PER_FRESH_STEP):
    def factory(model: WarpPhaseModel) -> FreshStepLoss:
        generator = torch.Generator().manual_seed(seed)
        pose = torch.randn(
            identities,
            3,
            17,
            3,
            dtype=torch.float32,
            generator=generator,
        )
        joint = torch.ones(identities, 3, 17, dtype=torch.bool)
        frame = torch.ones(identities, 3, dtype=torch.bool)
        clock = torch.ones(identities, 3, dtype=torch.bool)
        output = model(pose, joint, frame, clock)
        loss = output.hidden.square().mean()
        return FreshStepLoss(loss=loss, identity_count=identities)

    return factory


def test_four_fresh_backwards_make_one_clipped_adamw_update() -> None:
    torch.manual_seed(5)
    model = WarpPhaseModel()
    optimizer = CountingAdamW(list(model.parameters()))
    before = model.conv1.weight.detach().clone()
    factories = [_fresh_factory(index) for index in range(FRESH_STEPS_PER_UPDATE)]

    receipt = run_optimizer_update(model, optimizer, factories)

    assert receipt.backward_calls == 4
    assert receipt.optimizer_steps == 1
    assert receipt.identity_draws == 32
    assert len(receipt.unscaled_losses) == 4
    assert receipt.gradient_norm_before_clip >= 0.0
    assert optimizer.step_calls == 1
    assert not torch.equal(model.conv1.weight, before)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_build_optimizer_binds_frozen_hyperparameters_and_parameters() -> None:
    model = WarpPhaseModel()
    optimizer = build_optimizer(model)

    assert isinstance(optimizer, AdamW)
    assert optimizer.param_groups[0]["lr"] == LEARNING_RATE
    assert optimizer.param_groups[0]["weight_decay"] == WEIGHT_DECAY
    assert {
        id(parameter) for parameter in optimizer.param_groups[0]["params"]
    } == {id(parameter) for parameter in model.parameters()}


def test_accumulation_rejects_wrong_step_or_identity_count() -> None:
    model = WarpPhaseModel()
    optimizer = build_optimizer(model)
    with pytest.raises(ValueError, match="exactly 4"):
        run_optimizer_update(model, optimizer, [_fresh_factory(0)])

    factories = [_fresh_factory(index) for index in range(3)]
    factories.append(_fresh_factory(4, identities=7))
    with pytest.raises(ValueError, match="exactly 8"):
        run_optimizer_update(model, optimizer, factories)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_reusing_a_freed_graph_fails_before_any_optimizer_step() -> None:
    model = WarpPhaseModel()
    optimizer = CountingAdamW(list(model.parameters()))
    stale_result = _fresh_factory(9)(model)

    def stale_factory(_model: WarpPhaseModel) -> FreshStepLoss:
        return stale_result

    with pytest.raises(RuntimeError, match="backward through the graph a second time"):
        run_optimizer_update(model, optimizer, [stale_factory] * 4)
    assert optimizer.step_calls == 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_optimizer_type_and_frozen_values_fail_closed() -> None:
    model = WarpPhaseModel()
    wrong = AdamW(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    with pytest.raises(ValueError, match="differs"):
        run_optimizer_update(
            model,
            wrong,
            [_fresh_factory(index) for index in range(4)],
        )


def _feature_for_command() -> FeatureShard:
    motion = np.zeros((320, 17, 3), dtype="<f4")
    motion[:, :, 2] = np.float32(0.9)
    motion[:, 11, :2] = (0.0, 0.0)
    motion[:, 12, :2] = (2.0, 0.0)
    motion[:, 5, :2] = (0.0, 1.0)
    motion[:, 6, :2] = (2.0, 1.0)
    return FeatureShard(
        motion=motion,
        person_mask=True,
        frame_mask=np.ones(320, dtype="|u1"),
        sampled_frame_indices=np.arange(320, dtype="<i8"),
        source_length=320,
        opaque_sample_key="7" * 64,
        local_person_slot=3,
    )


def test_training_command_preflight_reads_only_feature_shards_and_stays_blocked(
    tmp_path: Path,
) -> None:
    layout = packing.create_pilot_layout(tmp_path / "run")
    packing.write_feature_shard(layout, "train", _feature_for_command())

    receipt = training_command_preflight(
        features_root=layout.features,
        split="train",
        environment_lock_sha256="a" * 64,
        output_space_receipt_sha256="b" * 64,
    )

    assert receipt.status == "BLOCKED"
    assert receipt.checks["feature_only_split_verified"]
    assert not receipt.checks["training_authorized"]
    assert receipt.bindings["feature_identity_count"] == "1"
    assert receipt.authorizes == ()


def test_training_interface_has_no_privileged_data_parameter_or_import() -> None:
    assert set(train.__annotations__) >= {
        "features_root",
        "output_root",
        "environment_lock",
        "output_space_receipt",
    }
    assert "vault" not in train.__annotations__
    assert "evaluator" not in train.__annotations__
    source = Path("src/pams/warp_phase/training.py").read_text(encoding="utf-8")
    assert "pams.warp_phase.evaluator" not in source
    assert "pams.warp_phase.packing" not in source
