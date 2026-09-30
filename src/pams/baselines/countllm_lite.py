"""Resource gate and explicit non-implementation for CountLLM-Lite.

The plan specifies a reduced training recipe, not a drop-in implementation of
the unpublished CountLLM code.  This module checks whether the host can even
attempt that recipe and always labels the current adapter honestly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pams.baselines.base import (
    BaselineSpec,
    BaselineStatus,
    BaselineUnavailableError,
    ImplementationKind,
    ProtocolRecord,
    ResourceRequirement,
)
from pams.types import CountResult, PoseSequence

MINIMUM_COUNTLLM_LITE_GPU_MEMORY_GIB = 48.0


@dataclass(frozen=True, slots=True)
class GPUResourceReport:
    """Serializable resource decision for manifests and reports."""

    cuda_available: bool
    device_count: int
    maximum_gpu_memory_gib: float
    device_names: tuple[str, ...]
    eligible_for_training: bool
    decision: str


def inspect_countllm_lite_resources(
    torch_module: Any | None = None,
    *,
    minimum_gpu_memory_gib: float = MINIMUM_COUNTLLM_LITE_GPU_MEMORY_GIB,
) -> GPUResourceReport:
    """Inspect CUDA without allocating a model or mutating device state."""

    if torch_module is None:
        try:
            import torch as torch_module
        except ImportError:
            return GPUResourceReport(
                cuda_available=False,
                device_count=0,
                maximum_gpu_memory_gib=0.0,
                device_names=(),
                eligible_for_training=False,
                decision="PyTorch is unavailable; CountLLM-Lite is blocked.",
            )

    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        return GPUResourceReport(
            cuda_available=False,
            device_count=0,
            maximum_gpu_memory_gib=0.0,
            device_names=(),
            eligible_for_training=False,
            decision="CUDA is unavailable; permit metadata/CPU smoke tests only.",
        )

    device_count = int(cuda.device_count())
    names: list[str] = []
    memories: list[float] = []
    for index in range(device_count):
        properties = cuda.get_device_properties(index)
        names.append(str(properties.name))
        total_memory = float(properties.total_memory) / (1024.0**3)
        memories.append(total_memory)

    maximum_memory = max(memories, default=0.0)
    eligible = device_count >= 1 and maximum_memory >= minimum_gpu_memory_gib
    if eligible:
        decision = (
            f"Resource gate passed: maximum GPU memory is {maximum_memory:.1f} GiB "
            f"(required {minimum_gpu_memory_gib:.1f} GiB)."
        )
    else:
        decision = (
            f"Resource gate failed: maximum GPU memory is {maximum_memory:.1f} GiB "
            f"(required {minimum_gpu_memory_gib:.1f} GiB); smoke tests only."
        )
    return GPUResourceReport(
        cuda_available=True,
        device_count=device_count,
        maximum_gpu_memory_gib=maximum_memory,
        device_names=tuple(names),
        eligible_for_training=eligible,
        decision=decision,
    )


def countllm_lite_spec(
    report: GPUResourceReport | None = None,
) -> BaselineSpec:
    """Build a host-specific registration record."""

    report = report or inspect_countllm_lite_resources()
    if report.eligible_for_training:
        status = BaselineStatus.BLOCKED_UNIMPLEMENTED
        reason = (
            "The hardware gate passes, but the clean-room Vicuna/video-encoder "
            "training adapter has not been implemented or validated."
        )
    elif report.cuda_available:
        status = BaselineStatus.SMOKE_ONLY
        reason = report.decision
    else:
        status = BaselineStatus.BLOCKED_RESOURCE
        reason = report.decision

    return BaselineSpec(
        key="countllm-lite",
        paper_name="CountLLM-Lite (reduced-resource, non-comparable recipe)",
        status=status,
        implementation=ImplementationKind.RESOURCE_GATED_RECIPE,
        protocols=(
            ProtocolRecord(
                key="ucfrep_526_countllm_lite",
                dataset="UCFRep-526",
                train_split="421 videos; stage 2/3 only",
                evaluation_split="105 held-out videos",
                modality="32 RGB frames + language model",
                source_faithful=False,
                fair_comparison=False,
                notes=(
                    "Omits WebVid-10M stage 1 and is never placed in the original "
                    "CountLLM same-condition result column."
                ),
            ),
        ),
        source_urls=("https://arxiv.org/abs/2503.17690",),
        config_path="configs/baselines/countllm_lite.yaml",
        blocked_reason=reason,
        resource_requirement=ResourceRequirement(
            minimum_gpu_memory_gib=MINIMUM_COUNTLLM_LITE_GPU_MEMORY_GIB,
            minimum_gpu_count=1,
            cuda_required=True,
            notes="Below this gate, only import/configuration smoke tests are permitted.",
        ),
    )


class CountLLMLiteAdapter:
    """Explicit guard against treating the reduced recipe as implemented."""

    def __init__(
        self,
        *,
        resource_inspector: Callable[[], GPUResourceReport] = inspect_countllm_lite_resources,
    ) -> None:
        self._spec = countllm_lite_spec(resource_inspector())

    @property
    def spec(self) -> BaselineSpec:
        return self._spec

    def predict(self, sample: PoseSequence) -> CountResult:
        del sample
        raise BaselineUnavailableError.from_spec(self.spec)
