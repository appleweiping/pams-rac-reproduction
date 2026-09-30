from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
import torch

from pams.baselines.countllm_recipe import (
    FROZEN_COUNTLLM_LITE_RECIPE,
    CountLLMLiteRecipeStatus,
    PeriodicityBridgeConfig,
    PeriodicityQueryBridge,
    StandardizedCountOutput,
    format_standardized_count_output,
    parse_standardized_count_output,
    validate_countllm_lite_environment,
)


def test_frozen_countllm_lite_recipe_matches_preregistered_reduction() -> None:
    recipe = FROZEN_COUNTLLM_LITE_RECIPE
    assert recipe.llm_name == "Vicuna-7B"
    assert recipe.frames_per_clip == 32
    assert recipe.periodicity_queries == 64
    assert recipe.periodicity_transformer_layers == 12
    assert recipe.lora_rank == 16
    assert recipe.quantization_bits == 4
    assert recipe.quantization_type == "nf4"
    assert recipe.micro_batch_size == 1
    assert recipe.gradient_accumulation_steps == 32
    assert recipe.effective_batch_size == 32
    assert not recipe.stage1_webvid_enabled
    assert recipe.stage2_epochs == recipe.stage3_epochs == 50
    assert recipe.enabled_stages == (
        "stage2_periodicity_alignment",
        "stage3_instruction_tuning",
    )
    with pytest.raises(FrozenInstanceError):
        recipe.frames_per_clip = 16  # type: ignore[misc]


def test_periodicity_query_bridge_shape_mask_and_gradients() -> None:
    torch.manual_seed(7)
    config = PeriodicityBridgeConfig(
        input_dim=6,
        model_dim=8,
        output_dim=12,
        num_queries=4,
        num_layers=2,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
    )
    bridge = PeriodicityQueryBridge(config)
    video_tokens = torch.randn(2, 7, 6, requires_grad=True)
    valid_mask = torch.tensor(
        [
            [True, True, True, True, True, False, False],
            [True, True, True, True, True, True, True],
        ]
    )

    output = bridge(video_tokens, valid_mask)
    assert output.shape == (2, 4, 12)
    output.square().mean().backward()
    assert video_tokens.grad is not None
    assert torch.isfinite(video_tokens.grad).all()
    assert bridge.periodicity_queries.weight.grad is not None


def test_periodicity_query_bridge_rejects_all_padding_sample() -> None:
    bridge = PeriodicityQueryBridge(
        PeriodicityBridgeConfig(
            input_dim=4,
            model_dim=8,
            output_dim=8,
            num_queries=2,
            num_layers=1,
            num_heads=2,
            feedforward_dim=16,
            dropout=0.0,
        )
    )
    with pytest.raises(ValueError, match="at least one valid"):
        bridge(
            torch.randn(2, 3, 4),
            torch.tensor([[True, True, False], [False, False, False]]),
        )


def test_standardized_output_round_trip_is_canonical_and_has_no_gt_input() -> None:
    text = format_standardized_count_output(
        37,
        incomplete_at_start=True,
        incomplete_at_end=False,
    )
    assert text == "[0037,1,0]"
    assert parse_standardized_count_output(text) == StandardizedCountOutput(
        count=37,
        incomplete_at_start=True,
        incomplete_at_end=False,
    )
    with pytest.raises(TypeError):
        format_standardized_count_output(
            37,
            incomplete_at_start=False,
            incomplete_at_end=False,
            ground_truth_count=37,  # type: ignore[call-arg]
        )


@pytest.mark.parametrize(
    "invalid",
    [
        "[37,1,0]",
        "[0037, 1, 0]",
        "[0037,2,0]",
        "[10000,1,0]",
        "answer: [0037,1,0]",
        "[0037,1,0]\n",
    ],
)
def test_standardized_output_parser_rejects_noncanonical_text(invalid: str) -> None:
    with pytest.raises(ValueError, match="canonical"):
        parse_standardized_count_output(invalid)


class _FakeCuda:
    def __init__(self, memories_gib: tuple[float, ...]) -> None:
        self._memories_gib = memories_gib

    def is_available(self) -> bool:
        return bool(self._memories_gib)

    def device_count(self) -> int:
        return len(self._memories_gib)

    def get_device_properties(self, index: int) -> SimpleNamespace:
        return SimpleNamespace(total_memory=int(self._memories_gib[index] * 1024**3))


def test_resource_validator_is_lazy_noncomparable_and_fails_honestly() -> None:
    requested_modules: list[str] = []

    def missing_bitsandbytes(name: str) -> object | None:
        requested_modules.append(name)
        return None if name == "bitsandbytes" else object()

    dependency_failure = validate_countllm_lite_environment(
        torch_module=SimpleNamespace(cuda=_FakeCuda((80.0,))),
        dependency_finder=missing_bitsandbytes,
    )
    assert requested_modules == ["transformers", "peft", "bitsandbytes"]
    assert dependency_failure.status is CountLLMLiteRecipeStatus.BLOCKED_DEPENDENCY
    assert not dependency_failure.environment_ready_for_recipe
    assert "bitsandbytes" in dependency_failure.explanation
    assert not dependency_failure.runnable_full_adapter
    assert not dependency_failure.comparable_to_reported_countllm

    resource_failure = validate_countllm_lite_environment(
        torch_module=SimpleNamespace(cuda=_FakeCuda((24.0,))),
        dependency_finder=lambda _name: object(),
    )
    assert resource_failure.status is CountLLMLiteRecipeStatus.BLOCKED_RESOURCE
    assert resource_failure.maximum_gpu_memory_gib == pytest.approx(24.0)
    assert "48.0 GiB is required" in resource_failure.explanation

    passing_gate = validate_countllm_lite_environment(
        torch_module=SimpleNamespace(cuda=_FakeCuda((80.0,))),
        dependency_finder=lambda _name: object(),
    )
    assert passing_gate.status is CountLLMLiteRecipeStatus.SCAFFOLD_ONLY
    assert passing_gate.environment_ready_for_recipe
    assert not passing_gate.runnable_full_adapter
    assert not passing_gate.comparable_to_reported_countllm
