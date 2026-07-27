"""Auditable CountLLM-Lite recipe components.

This module is intentionally *not* a full CountLLM baseline adapter.  The
authors did not release an implementation, and the reduced recipe omits the
WebVid-10M semantic-alignment stage used by the paper.  Consequently, results
from this scaffold are not comparable with reported CountLLM results.

The only executable model component here is a small, configurable
periodicity-query bridge for already-computed video tokens.  Importing this
module never downloads or loads Vicuna, a video encoder, or optional Hugging
Face dependencies.
"""

from __future__ import annotations

import importlib.util
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import Tensor, nn

COUNTLLM_LITE_MINIMUM_GPU_MEMORY_GIB = 48.0
_OPTIONAL_RUNTIME_DEPENDENCIES = ("transformers", "peft", "bitsandbytes")
_STANDARDIZED_OUTPUT_PATTERN = re.compile(r"\[(?P<count>[0-9]{4}),(?P<start>[01]),(?P<end>[01])\]")


@dataclass(frozen=True, slots=True)
class CountLLMLiteRecipeConfig:
    """Frozen, reduced-resource recipe agreed for the independent reproduction.

    These settings describe a proposed experiment, not settings disclosed by a
    runnable official implementation.  In particular, NF4 quantization and the
    omission of stage 1 are reproduction-specific resource compromises.
    """

    schema_version: int = 1
    llm_name: str = "Vicuna-7B"
    llm_model_id: str = "lmsys/vicuna-7b-v1.5"
    video_encoder_name: str = "VideoMAE"
    frames_per_clip: int = 32
    periodicity_queries: int = 64
    periodicity_transformer_layers: int = 12
    lora_rank: int = 16
    quantization_bits: int = 4
    quantization_type: str = "nf4"
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    stage1_webvid_enabled: bool = False
    stage2_epochs: int = 50
    stage3_epochs: int = 50
    minimum_gpu_memory_gib: float = COUNTLLM_LITE_MINIMUM_GPU_MEMORY_GIB

    def __post_init__(self) -> None:
        locked_values = {
            "llm_name": (self.llm_name, "Vicuna-7B"),
            "frames_per_clip": (self.frames_per_clip, 32),
            "lora_rank": (self.lora_rank, 16),
            "quantization_bits": (self.quantization_bits, 4),
            "quantization_type": (self.quantization_type, "nf4"),
            "micro_batch_size": (self.micro_batch_size, 1),
            "gradient_accumulation_steps": (self.gradient_accumulation_steps, 32),
            "stage1_webvid_enabled": (self.stage1_webvid_enabled, False),
            "stage2_epochs": (self.stage2_epochs, 50),
            "stage3_epochs": (self.stage3_epochs, 50),
        }
        for field_name, (actual, expected) in locked_values.items():
            if actual != expected:
                raise ValueError(
                    f"{field_name} is frozen to {expected!r} for the CountLLM-Lite recipe"
                )
        if self.periodicity_queries != 64 or self.periodicity_transformer_layers != 12:
            raise ValueError(
                "the recipe preserves the paper's 64 queries and 12-layer periodicity transformer"
            )
        if self.minimum_gpu_memory_gib < COUNTLLM_LITE_MINIMUM_GPU_MEMORY_GIB:
            raise ValueError("the CountLLM-Lite training gate cannot be lowered below 48 GiB")

    @property
    def effective_batch_size(self) -> int:
        """Effective batch size for one process before distributed scaling."""

        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def enabled_stages(self) -> tuple[str, ...]:
        """The only stages enabled by this deliberately reduced recipe."""

        return ("stage2_periodicity_alignment", "stage3_instruction_tuning")


FROZEN_COUNTLLM_LITE_RECIPE = CountLLMLiteRecipeConfig()


@dataclass(frozen=True, slots=True)
class PeriodicityBridgeConfig:
    """Architecture for a query transformer over precomputed video tokens."""

    input_dim: int = 768
    model_dim: int = 768
    output_dim: int = 4096
    num_queries: int = 64
    num_layers: int = 12
    num_heads: int = 12
    feedforward_dim: int = 3072
    dropout: float = 0.1

    def __post_init__(self) -> None:
        integer_fields = {
            "input_dim": self.input_dim,
            "model_dim": self.model_dim,
            "output_dim": self.output_dim,
            "num_queries": self.num_queries,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
        }
        for field_name, value in integer_fields.items():
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.model_dim % self.num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class PeriodicityQueryBridge(nn.Module):
    """Compress precomputed video tokens into LLM-width periodicity queries.

    This is a clean-room architectural scaffold inspired by the paper's
    high-level description.  It deliberately starts from video tokens and has
    no code path that constructs or downloads a video encoder or language
    model.
    """

    def __init__(self, config: PeriodicityBridgeConfig | None = None) -> None:
        super().__init__()
        config = config or PeriodicityBridgeConfig()
        self.config = config
        self.input_projection = nn.Linear(config.input_dim, config.model_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.periodicity_transformer = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.model_dim),
        )
        self.periodicity_queries = nn.Embedding(config.num_queries, config.model_dim)
        self.output_projection = nn.Linear(config.model_dim, config.output_dim)
        nn.init.normal_(self.periodicity_queries.weight, mean=0.0, std=0.02)

    def forward(
        self,
        video_tokens: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        """Return periodicity tokens shaped ``[batch, queries, output_dim]``.

        ``valid_mask`` uses ``True`` for valid video tokens.  Every sample must
        retain at least one token; silently attending to an all-padding sequence
        would otherwise create undefined attention values.
        """

        if video_tokens.ndim != 3:
            raise ValueError("video_tokens must have shape [batch, tokens, input_dim]")
        batch_size, token_count, input_dim = video_tokens.shape
        if batch_size == 0 or token_count == 0:
            raise ValueError("video_tokens must contain at least one batch item and token")
        if input_dim != self.config.input_dim:
            raise ValueError(f"expected input_dim={self.config.input_dim}, received {input_dim}")
        if not torch.is_floating_point(video_tokens):
            raise TypeError("video_tokens must be floating point")

        if valid_mask is None:
            valid_mask = torch.ones(
                (batch_size, token_count),
                dtype=torch.bool,
                device=video_tokens.device,
            )
        elif valid_mask.shape != (batch_size, token_count):
            raise ValueError("valid_mask must have shape [batch, tokens]")
        elif valid_mask.dtype is not torch.bool:
            raise TypeError("valid_mask must have boolean dtype")
        else:
            valid_mask = valid_mask.to(device=video_tokens.device)
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("every sample must contain at least one valid video token")

        memory = self.input_projection(video_tokens)
        memory = memory + _sinusoidal_positions(
            length=token_count,
            width=self.config.model_dim,
            device=memory.device,
            dtype=memory.dtype,
        )
        memory = memory.masked_fill(~valid_mask.unsqueeze(-1), 0.0)

        query_ids = torch.arange(
            self.config.num_queries,
            device=video_tokens.device,
        )
        queries = self.periodicity_queries(query_ids).unsqueeze(0).expand(batch_size, -1, -1)
        periodicity_embeddings = self.periodicity_transformer(
            tgt=queries,
            memory=memory,
            memory_key_padding_mask=~valid_mask,
        )
        return self.output_projection(periodicity_embeddings)


def _sinusoidal_positions(
    *,
    length: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    positions = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, width, 2, device=device, dtype=torch.float32)
        * (-math.log(10_000.0) / width)
    )
    encoding = torch.zeros((length, width), device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    odd_width = encoding[:, 1::2].shape[1]
    encoding[:, 1::2] = torch.cos(positions * frequencies[:odd_width])
    return encoding.to(dtype=dtype).unsqueeze(0)


@dataclass(frozen=True, slots=True)
class StandardizedCountOutput:
    """Canonical CountLLM clip result represented as ``[abcd,e,f]``."""

    count: int
    incomplete_at_start: bool
    incomplete_at_end: bool

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError("count must be an integer")
        if not 0 <= self.count <= 9999:
            raise ValueError("count must be in [0, 9999]")
        if not isinstance(self.incomplete_at_start, bool):
            raise TypeError("incomplete_at_start must be bool")
        if not isinstance(self.incomplete_at_end, bool):
            raise TypeError("incomplete_at_end must be bool")


def format_standardized_count_output(
    count: int,
    *,
    incomplete_at_start: bool,
    incomplete_at_end: bool,
) -> str:
    """Format a prediction without accepting any ground-truth argument."""

    output = StandardizedCountOutput(
        count=count,
        incomplete_at_start=incomplete_at_start,
        incomplete_at_end=incomplete_at_end,
    )
    return f"[{output.count:04d},{int(output.incomplete_at_start)},{int(output.incomplete_at_end)}]"


def parse_standardized_count_output(text: str) -> StandardizedCountOutput:
    """Parse only the canonical ``[abcd,e,f]`` output, with no surrounding text."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    match = _STANDARDIZED_OUTPUT_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError("expected canonical CountLLM output '[dddd,b,b]'")
    return StandardizedCountOutput(
        count=int(match.group("count")),
        incomplete_at_start=match.group("start") == "1",
        incomplete_at_end=match.group("end") == "1",
    )


class CountLLMLiteRecipeStatus(str, Enum):
    """Readiness of the recipe scaffold, never baseline accuracy status."""

    BLOCKED_DEPENDENCY = "blocked_dependency"
    BLOCKED_RESOURCE = "blocked_resource"
    SCAFFOLD_ONLY = "scaffold_only"


@dataclass(frozen=True, slots=True)
class OptionalDependencyCheck:
    """Availability discovered without importing an optional package."""

    name: str
    available: bool


@dataclass(frozen=True, slots=True)
class CountLLMLiteEnvironmentReport:
    """Dependency/resource gate with explicit comparability disclosure."""

    status: CountLLMLiteRecipeStatus
    dependencies: tuple[OptionalDependencyCheck, ...]
    cuda_available: bool
    device_count: int
    maximum_gpu_memory_gib: float
    environment_ready_for_recipe: bool
    runnable_full_adapter: bool
    comparable_to_reported_countllm: bool
    explanation: str


def validate_countllm_lite_environment(
    *,
    torch_module: Any | None = None,
    dependency_finder: Callable[[str], object | None] = importlib.util.find_spec,
    config: CountLLMLiteRecipeConfig = FROZEN_COUNTLLM_LITE_RECIPE,
) -> CountLLMLiteEnvironmentReport:
    """Validate optional packages and CUDA without loading models or weights.

    ``dependency_finder`` defaults to ``importlib.util.find_spec``: optional
    packages are located, never imported.  Passing fakes makes the resource
    decision fully unit-testable without CUDA.
    """

    dependencies = tuple(
        OptionalDependencyCheck(name=name, available=_module_is_available(name, dependency_finder))
        for name in _OPTIONAL_RUNTIME_DEPENDENCIES
    )
    missing_dependencies = tuple(item.name for item in dependencies if not item.available)
    runtime = torch if torch_module is None else torch_module
    cuda = getattr(runtime, "cuda", None)
    cuda_available = bool(cuda is not None and cuda.is_available())
    device_count = int(cuda.device_count()) if cuda is not None and cuda_available else 0
    memories: list[float] = []
    if cuda is not None and cuda_available:
        for device_index in range(device_count):
            properties = cuda.get_device_properties(device_index)
            memories.append(float(properties.total_memory) / (1024.0**3))
    maximum_memory = max(memories, default=0.0)

    if missing_dependencies:
        status = CountLLMLiteRecipeStatus.BLOCKED_DEPENDENCY
        ready = False
        reason = f"missing optional packages: {', '.join(missing_dependencies)}"
    elif not cuda_available:
        status = CountLLMLiteRecipeStatus.BLOCKED_RESOURCE
        ready = False
        reason = "CUDA is unavailable"
    elif maximum_memory < config.minimum_gpu_memory_gib:
        status = CountLLMLiteRecipeStatus.BLOCKED_RESOURCE
        ready = False
        reason = (
            f"maximum GPU memory is {maximum_memory:.1f} GiB; "
            f"{config.minimum_gpu_memory_gib:.1f} GiB is required"
        )
    else:
        status = CountLLMLiteRecipeStatus.SCAFFOLD_ONLY
        ready = True
        reason = "dependency and hardware checks passed"

    explanation = (
        f"{reason}. This is a non-comparable CountLLM-Lite recipe scaffold: "
        "stage 1/WebVid-10M is omitted, and no runnable full baseline adapter is provided."
    )
    return CountLLMLiteEnvironmentReport(
        status=status,
        dependencies=dependencies,
        cuda_available=cuda_available,
        device_count=device_count,
        maximum_gpu_memory_gib=maximum_memory,
        environment_ready_for_recipe=ready,
        runnable_full_adapter=False,
        comparable_to_reported_countllm=False,
        explanation=explanation,
    )


def _module_is_available(
    name: str,
    dependency_finder: Callable[[str], object | None],
) -> bool:
    try:
        return dependency_finder(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False
