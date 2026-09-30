"""Public contracts and audit records for clean-room baseline adapters.

Registration is deliberately separate from executability.  A paper method can
have a complete audit record while still being blocked because its clean-room
implementation, protocol, or required resources are unavailable.  Callers must
therefore inspect ``BaselineSpec.status`` (or use ``require_runnable``) before
constructing an adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from pams.types import CountResult, PoseSequence


class BaselineStatus(str, Enum):
    """Machine-readable execution status for a registered method."""

    READY = "ready"
    REFERENCE_ONLY = "reference_only"
    SMOKE_ONLY = "smoke_only"
    BLOCKED_UNIMPLEMENTED = "blocked_unimplemented"
    BLOCKED_PROTOCOL = "blocked_protocol"
    BLOCKED_RESOURCE = "blocked_resource"

    @property
    def runnable(self) -> bool:
        """Whether the status permits calling ``predict``."""

        return self in {self.READY, self.REFERENCE_ONLY}


class ImplementationKind(str, Enum):
    """Provenance of an implementation, never a claim about its accuracy."""

    CLEAN_ROOM = "clean_room"
    OFFICIAL_ASSET_WRAPPER = "official_asset_wrapper"
    DETERMINISTIC_REFERENCE = "deterministic_reference"
    RESOURCE_GATED_RECIPE = "resource_gated_recipe"
    NOT_IMPLEMENTED = "not_implemented"


@dataclass(frozen=True, slots=True)
class ProtocolRecord:
    """A dataset protocol against which a method may be discussed."""

    key: str
    dataset: str
    train_split: str
    evaluation_split: str
    modality: str
    source_faithful: bool
    fair_comparison: bool
    ground_truth_count_at_inference: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.ground_truth_count_at_inference and self.fair_comparison:
            raise ValueError("a protocol that reads ground-truth count at inference cannot be fair")


@dataclass(frozen=True, slots=True)
class ResourceRequirement:
    """Minimum resources declared by an adapter or training recipe."""

    minimum_gpu_memory_gib: float = 0.0
    minimum_gpu_count: int = 0
    cuda_required: bool = False
    notes: str = ""


@dataclass(frozen=True, slots=True)
class BaselineSpec:
    """Immutable registration and provenance record."""

    key: str
    paper_name: str
    status: BaselineStatus
    implementation: ImplementationKind
    protocols: tuple[ProtocolRecord, ...]
    source_urls: tuple[str, ...] = ()
    config_path: str | None = None
    blocked_reason: str | None = None
    resource_requirement: ResourceRequirement = ResourceRequirement()
    is_paper_baseline: bool = True

    def __post_init__(self) -> None:
        if not self.key or self.key != self.key.lower():
            raise ValueError("baseline keys must be non-empty lowercase strings")
        if not self.protocols:
            raise ValueError(f"{self.key} must declare at least one protocol")
        if not self.status.runnable and not self.blocked_reason:
            raise ValueError(f"{self.key} is blocked but has no blocked_reason")
        if self.status.runnable and self.implementation is ImplementationKind.NOT_IMPLEMENTED:
            raise ValueError(f"{self.key} cannot be runnable without an implementation")

    @property
    def runnable(self) -> bool:
        """Whether the registered implementation can currently predict."""

        return self.status.runnable

    def require_runnable(self) -> None:
        """Raise a detailed error rather than silently substituting a proxy."""

        if not self.runnable:
            raise BaselineUnavailableError.from_spec(self)


class BaselineUnavailableError(RuntimeError):
    """Raised when a registered method has no honest runnable adapter."""

    def __init__(
        self,
        key: str,
        status: BaselineStatus,
        reason: str,
    ) -> None:
        self.key = key
        self.status = status
        self.reason = reason
        super().__init__(f"baseline '{key}' is {status.value}: {reason}")

    @classmethod
    def from_spec(cls, spec: BaselineSpec) -> BaselineUnavailableError:
        return cls(
            key=spec.key,
            status=spec.status,
            reason=spec.blocked_reason or "no runnable adapter is registered",
        )


@runtime_checkable
class BaselineAdapter(Protocol):
    """Common prediction interface for all fair and diagnostic baselines."""

    @property
    def spec(self) -> BaselineSpec:
        """Return the immutable audit record for this adapter."""

        ...

    def predict(self, sample: PoseSequence) -> CountResult:
        """Predict without reading an evaluation count label."""

        ...
