"""Package-private typed records for the WARP-PHASE pilot.

These records intentionally do not inherit from, wrap, or replace any of the
historical PAMS data types.  Routing identifiers are represented as Python
metadata and are kept separate from model-facing arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

PilotSplit: TypeAlias = Literal["train", "val"]
Float32Array: TypeAlias = NDArray[np.float32]
UInt8Array: TypeAlias = NDArray[np.uint8]
Int64Array: TypeAlias = NDArray[np.int64]

FEATURE_FIELD_NAMES: tuple[str, ...] = (
    "frame_mask",
    "local_person_slot",
    "motion",
    "opaque_sample_key",
    "person_mask",
    "sampled_frame_indices",
    "source_length",
)


class WarpPhaseContractError(ValueError):
    """Base exception for a fail-closed WARP-PHASE contract violation."""


@dataclass(frozen=True, slots=True)
class SourceVerificationReceipt:
    """Identity of one source pickle verified before trusted unpickling."""

    split: PilotSplit
    path: Path
    relative_path: str
    sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class FeatureShard:
    """One complete supplied identity from a feature-only shard.

    ``opaque_sample_key`` and ``local_person_slot`` are routing metadata.  They
    must never be converted into model inputs.
    """

    motion: Float32Array
    person_mask: bool
    frame_mask: UInt8Array
    sampled_frame_indices: Int64Array
    source_length: int
    opaque_sample_key: str
    local_person_slot: int
    shard_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class EligibilityInput:
    """Label-free inputs used to decide whether one supplied slot is eligible."""

    feature: FeatureShard
    association_ambiguous: bool
    association_one_to_one: bool
    pose_coverage: float


@dataclass(frozen=True, slots=True)
class EligibilityDecision:
    """Deterministic count-blind eligibility receipt for one supplied slot."""

    opaque_sample_key: str
    local_person_slot: int
    association_ambiguous: bool
    association_one_to_one: bool
    pose_coverage: float
    eligible: bool
    reasons: tuple[str, ...]
    retained_distinct_clocks: int
    valid_adjacent_cells: int
    feature_frame_coverage: float
    conflict_clocks: tuple[int, ...]
    normalization_scale: float | None


@dataclass(frozen=True, slots=True)
class CollapsedTrack:
    """A duplicate-collapsed and COCO17-normalized complete identity."""

    motion: Float32Array
    joint_mask: UInt8Array
    frame_mask: UInt8Array
    cell_mask: UInt8Array
    sampled_frame_indices: Int64Array
    source_length: int
    opaque_sample_key: str
    local_person_slot: int
    conflict_clocks: tuple[int, ...]
    normalization_scale: float


@dataclass(frozen=True, slots=True)
class EligibilityAssessment:
    """Eligibility decision plus the usable track, when normalization succeeded."""

    decision: EligibilityDecision
    track: CollapsedTrack | None


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    """Right-padded model arrays with routing metadata kept out-of-tensor."""

    motion: Float32Array
    person_mask: UInt8Array
    frame_mask: UInt8Array
    joint_mask: UInt8Array
    cell_mask: UInt8Array
    padding_mask: UInt8Array
    sampled_frame_indices: Int64Array
    source_length: Int64Array
    opaque_sample_keys: tuple[str, ...]
    local_person_slots: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EvaluatorVaultRecord:
    """Privileged packer-side record; no reader is exposed by the data module."""

    opaque_sample_key: str
    local_person_slot: int
    source_length: int
    count: int
    periods: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class ArrayMemberReceipt:
    """Schema and digest for one deterministic NPY member."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class ArtifactReceipt:
    """Digest and detached audit receipt for one packed artifact."""

    role: str
    path: Path
    relative_path: str
    sha256: str
    byte_count: int
    receipt_path: Path
    receipt_sha256: str
    members: tuple[ArrayMemberReceipt, ...] = ()


@dataclass(frozen=True, slots=True)
class PackedIdentityReceipt:
    """Result of the label-lazy per-identity packing boundary."""

    eligibility: EligibilityDecision
    audit: ArtifactReceipt
    feature: ArtifactReceipt | None
    vault: ArtifactReceipt | None
