"""Trusted, physically separated feature/vault packing primitives.

There is intentionally no deserializer in this module.  A future separately
authorized caller must verify a source file first, deserialize it in the
trusted boundary, and pass a strictly typed in-memory source object here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from pams.temporac.contract import CONTRACT_SHA256, FEATURE_RECEIPT_SCHEMA, ReasonCode
from pams.temporac.hashio import (
    canonical_json_bytes,
    feature_npz_bytes,
    opaque_sample_key,
    read_regular_file,
    sha256_bytes,
    source_binding_sha256,
    write_bytes_exclusive,
)
from pams.temporac.receipts import member_payload, receipt_bytes
from pams.temporac.types import ContractError, FeatureRecord

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_SOURCE_PARTS = frozenset({"test", "sealed", "heldout", "results", "output", "server"})
_FORBIDDEN_SOURCE_MARKERS = ("v4x", "v46", "official-complete", "precommit", "selection")
SOURCE_SHA256 = MappingProxyType(
    {
        "train": "c96fc1dfa233ec4bf1af19e7480ee8c721f4cddee9f121185f4e494c7244e1eb",
        "val": "4064ef24dc2970e9b07de8adddf1ecd4932aecb90453a67b4f90d7bd35d10251",
    }
)

ObjectID: TypeAlias = str | bytes | int


class PackerError(ContractError):
    """A trusted-source, join, feature, vault, or manifest contract failure."""


@dataclass(frozen=True, slots=True)
class VerifiedSourceFile:
    split: str
    path: Path
    sha256: str
    bytes: int


def verify_source_file(
    path: Path,
    *,
    split: str,
    allowlisted_path: Path,
    expected_sha256: str,
    expected_bytes: int,
) -> VerifiedSourceFile:
    """Verify text allowlist, regular-file status, size, and hash in that order."""

    if split not in {"train", "val"}:
        raise PackerError("trusted source split must be train or val")
    candidate = Path(path)
    parts = tuple(part.casefold() for part in candidate.parts)
    if any(part in _FORBIDDEN_SOURCE_PARTS for part in parts) or any(
        marker in part for part in parts for marker in _FORBIDDEN_SOURCE_MARKERS
    ):
        raise PackerError("trusted source path is in a forbidden split/result class")
    if candidate.absolute() != Path(allowlisted_path).absolute():
        raise PackerError("trusted source path is not the exact allowlisted path")
    if expected_sha256 != SOURCE_SHA256[split]:
        raise PackerError("trusted source digest is not the frozen v44 split digest")
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise PackerError("trusted source expected byte count must be positive")
    payload = read_regular_file(candidate, max_bytes=expected_bytes)
    if len(payload) != expected_bytes:
        raise PackerError("trusted source byte count differs from the allowlist")
    observed = sha256_bytes(payload)
    if observed != expected_sha256:
        raise PackerError("trusted source digest differs from the allowlist")
    return VerifiedSourceFile(split, candidate, observed, len(payload))


@dataclass(frozen=True, slots=True)
class TrustedSourceObject:
    """The audited multi-person v44 shape plus privileged join identifiers."""

    motion: NDArray[np.float32]
    person_mask: NDArray[np.bool_]
    frame_mask: NDArray[np.bool_]
    sampled_frame_indices: NDArray[np.int64]
    source_length: int
    person_object_ids: tuple[ObjectID, ...]

    def __post_init__(self) -> None:
        motion = np.asarray(self.motion)
        people = np.asarray(self.person_mask)
        frames = np.asarray(self.frame_mask)
        clocks = np.asarray(self.sampled_frame_indices)
        if motion.dtype.str != "<f4" or motion.ndim != 4 or motion.shape[1:] != (320, 17, 3):
            raise PackerError("trusted motion must be <f4[P,320,17,3]")
        person_count = motion.shape[0]
        if people.dtype != np.dtype(np.bool_) or people.shape != (person_count,):
            raise PackerError("trusted person_mask must be bool[P]")
        if frames.dtype != np.dtype(np.bool_) or frames.shape != (person_count, 320):
            raise PackerError("trusted frame_mask must be bool[P,320]")
        if clocks.dtype.str != "<i8" or clocks.shape != (320,) or np.any(np.diff(clocks) < 0):
            raise PackerError("trusted sampled_frame_indices must be nondecreasing <i8[320]")
        if type(self.source_length) is not int or self.source_length <= 1:
            raise PackerError("trusted source_length must be an integer greater than one")
        if len(self.person_object_ids) != person_count or any(
            not isinstance(value, (str, bytes, int)) or isinstance(value, bool)
            for value in self.person_object_ids
        ):
            raise PackerError("person_object_ids must contain one scalar ID per source slot")
        for name, array in (
            ("motion", motion),
            ("person_mask", people),
            ("frame_mask", frames),
            ("sampled_frame_indices", clocks),
        ):
            copied = np.array(array, order="C", copy=True)
            copied.flags.writeable = False
            object.__setattr__(self, name, copied)


@dataclass(frozen=True, slots=True)
class PrivilegedAnnotation:
    """Evaluator-only values keyed by the trusted raw person object ID."""

    count_gt: int
    periods: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if type(self.count_gt) is not int or self.count_gt <= 0:
            raise PackerError("vault count_gt must be a positive integer")
        normalized: list[tuple[int, int]] = []
        for period in self.periods:
            if (
                not isinstance(period, tuple)
                or len(period) != 2
                or type(period[0]) is not int
                or type(period[1]) is not int
            ):
                raise PackerError("vault periods must be integer endpoint pairs")
            normalized.append(period)
        object.__setattr__(self, "periods", tuple(normalized))


@dataclass(frozen=True, slots=True)
class VaultRow:
    """One evaluator-only raw row; raw source IDs never enter this record."""

    opaque_key_hex: str
    slot: int
    source_length: int
    count_gt: int
    periods: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        if _SHA256_RE.fullmatch(self.opaque_key_hex) is None:
            raise PackerError("vault opaque key must be lowercase SHA-256 hex")
        if type(self.slot) is not int or self.slot < 0:
            raise PackerError("vault slot must be nonnegative")
        if type(self.source_length) is not int or self.source_length <= 1:
            raise PackerError("vault source_length must be greater than one")
        if type(self.count_gt) is not int or self.count_gt <= 0:
            raise PackerError("vault count_gt must be positive")
        normalized: list[tuple[int, int]] = []
        for period in self.periods:
            if (
                not isinstance(period, tuple)
                or len(period) != 2
                or type(period[0]) is not int
                or type(period[1]) is not int
            ):
                raise PackerError("vault periods must be integer endpoint pairs")
            start, end = period
            if not (0 <= start < end <= self.source_length):
                raise PackerError("vault period does not satisfy 0<=s<e<=source_length")
            normalized.append(period)
        object.__setattr__(self, "periods", tuple(normalized))

    @property
    def identity(self) -> tuple[str, int]:
        return self.opaque_key_hex, self.slot

    def as_dict(self) -> dict[str, object]:
        return {
            "count_gt": self.count_gt,
            "opaque_key_hex": self.opaque_key_hex,
            "periods": [list(period) for period in self.periods],
            "slot": self.slot,
            "source_length": self.source_length,
        }


@dataclass(frozen=True, slots=True)
class PackedIdentity:
    feature: FeatureRecord
    vault: VaultRow
    source_binding_sha256: str

    def __post_init__(self) -> None:
        if self.feature.opaque_key_bytes.hex() != self.vault.opaque_key_hex:
            raise PackerError("feature/vault opaque keys differ")
        if self.feature.slot != self.vault.slot:
            raise PackerError("feature/vault slots differ")
        if int(self.feature.source_length[0]) != self.vault.source_length:
            raise PackerError("feature/vault source lengths differ")
        if _SHA256_RE.fullmatch(self.source_binding_sha256) is None:
            raise PackerError("source binding must be lowercase SHA-256")


def _canonical_slot_motion(
    motion: NDArray[np.float32], frame_mask: NDArray[np.bool_]
) -> NDArray[np.float32]:
    finite = np.isfinite(motion).all(axis=2)
    clipped_confidence = np.clip(motion[:, :, 2].astype(np.float64), 0.0, 1.0)
    valid = frame_mask[:, None] & finite & (clipped_confidence > 0.20)
    canonical = np.zeros((320, 17, 3), dtype="<f4")
    canonical[:, :, 0] = np.where(valid, motion[:, :, 0], np.float32(0.0))
    canonical[:, :, 1] = np.where(valid, motion[:, :, 1], np.float32(0.0))
    canonical[:, :, 2] = np.where(valid, clipped_confidence, 0.0).astype("<f4")
    if not np.isfinite(canonical).all() or np.signbit(canonical[~valid]).any():
        raise PackerError("invalid joints were not canonicalized to finite +0.0")
    return canonical


def pack_source_object(
    source: TrustedSourceObject,
    *,
    split: str,
    source_pickle_sha256: str,
    object_ordinal: int,
    annotations: Mapping[ObjectID, PrivilegedAnnotation],
) -> tuple[PackedIdentity, ...]:
    """Split one trusted object without renumbering its valid person slots."""

    if split not in {"train", "val"} or _SHA256_RE.fullmatch(source_pickle_sha256) is None:
        raise PackerError("packing requires a frozen split and source digest")
    if type(object_ordinal) is not int or object_ordinal < 0:
        raise PackerError("source object ordinal must be nonnegative")
    valid_slots = tuple(int(slot) for slot in np.flatnonzero(source.person_mask))
    valid_ids = tuple(source.person_object_ids[slot] for slot in valid_slots)
    if len(valid_ids) != len(set(valid_ids)):
        raise PackerError("many-to-one person_object_ids are forbidden")
    if set(annotations) != set(valid_ids):
        raise PackerError("source-to-vault annotation join is missing, extra, or non-total")
    opaque = opaque_sample_key(split, source_pickle_sha256, object_ordinal)
    packed: list[PackedIdentity] = []
    for slot, object_id in zip(valid_slots, valid_ids, strict=True):
        annotation = annotations[object_id]
        feature = FeatureRecord(
            frame_mask=np.asarray(source.frame_mask[slot], dtype="|u1"),
            local_person_slot=np.asarray([slot], dtype="<i8"),
            motion=_canonical_slot_motion(source.motion[slot], source.frame_mask[slot]),
            opaque_sample_key=np.frombuffer(opaque, dtype="|u1").copy(),
            person_mask=np.asarray([1], dtype="|u1"),
            sampled_frame_indices=np.asarray(source.sampled_frame_indices, dtype="<i8"),
            source_length=np.asarray([source.source_length], dtype="<i8"),
        )
        vault = VaultRow(
            opaque_key_hex=opaque.hex(),
            slot=slot,
            source_length=source.source_length,
            count_gt=annotation.count_gt,
            periods=annotation.periods,
        )
        packed.append(
            PackedIdentity(
                feature=feature,
                vault=vault,
                source_binding_sha256=source_binding_sha256(
                    split, source_pickle_sha256, object_ordinal, slot
                ),
            )
        )
    return tuple(packed)


def feature_artifact_and_receipt(
    packed: PackedIdentity,
) -> tuple[bytes, bytes, dict[str, object]]:
    """Build exact feature NPZ and detached closed-schema receipt bytes."""

    artifact, members = feature_npz_bytes(packed.feature)
    payload: dict[str, object] = {
        "artifact_bytes": len(artifact),
        "artifact_sha256": sha256_bytes(artifact),
        "contract_sha256": CONTRACT_SHA256,
        "members": member_payload(members),
        "opaque_key_hex": packed.feature.opaque_key_bytes.hex(),
        "schema": FEATURE_RECEIPT_SCHEMA,
        "slot": packed.feature.slot,
        "source_binding_sha256": packed.source_binding_sha256,
    }
    return artifact, receipt_bytes(payload, expected_schema=FEATURE_RECEIPT_SCHEMA), payload


def vault_bytes(rows: Sequence[VaultRow]) -> bytes:
    ordered = sorted(rows, key=lambda row: (bytes.fromhex(row.opaque_key_hex), row.slot))
    identities = [row.identity for row in ordered]
    if len(identities) != len(set(identities)):
        raise PackerError("vault contains a duplicate opaque-key/slot identity")
    return canonical_json_bytes([row.as_dict() for row in ordered])


def publish_packed_identity(
    packed: PackedIdentity,
    *,
    feature_path: Path,
    receipt_path: Path,
    vault_path: Path,
) -> tuple[str, str, str]:
    """Exclusively publish physically separate feature, receipt, and vault bytes."""

    expected_name = f"{packed.feature.opaque_key_bytes.hex()}.{packed.feature.slot}.npz"
    if feature_path.name != expected_name:
        raise PackerError("feature output filename does not bind opaque key and slot")
    if len({feature_path.parent, receipt_path.parent, vault_path.parent}) != 3:
        raise PackerError("feature, receipt, and vault roots must be physically distinct")
    artifact, receipt, _ = feature_artifact_and_receipt(packed)
    return (
        write_bytes_exclusive(feature_path, artifact, mode=0o444),
        write_bytes_exclusive(receipt_path, receipt, mode=0o444),
        write_bytes_exclusive(vault_path, canonical_json_bytes(packed.vault.as_dict()), mode=0o400),
    )


@dataclass(frozen=True, slots=True)
class PopulationRow:
    split: str
    opaque_key_hex: str
    slot: int
    component_key_hex: str
    source_binding_sha256: str
    feature_receipt_sha256_or_null: str | None
    eligible: bool
    reason_codes: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.split not in {"train", "val"}:
            raise PackerError("population split must be train or val")
        for name in ("opaque_key_hex", "component_key_hex", "source_binding_sha256"):
            if _SHA256_RE.fullmatch(cast(str, getattr(self, name))) is None:
                raise PackerError(f"population {name} must be lowercase SHA-256")
        if (
            self.feature_receipt_sha256_or_null is not None
            and _SHA256_RE.fullmatch(self.feature_receipt_sha256_or_null) is None
        ):
            raise PackerError("population feature receipt must be null or lowercase SHA-256")
        if type(self.slot) is not int or self.slot < 0 or type(self.eligible) is not bool:
            raise PackerError("population slot/eligible types are invalid")
        reasons = tuple(int(code) for code in self.reason_codes)
        if reasons != tuple(sorted(set(reasons))) or any(
            code not in ReasonCode._value2member_map_ for code in reasons
        ):
            raise PackerError("population reason codes must be known, unique, and ascending")
        if self.eligible == bool(reasons):
            raise PackerError("eligible rows have no reasons; ineligible rows have reasons")
        object.__setattr__(self, "reason_codes", reasons)

    def as_dict(self) -> dict[str, object]:
        return {
            "component_key_hex": self.component_key_hex,
            "eligible": self.eligible,
            "feature_receipt_sha256_or_null": self.feature_receipt_sha256_or_null,
            "opaque_key_hex": self.opaque_key_hex,
            "reason_codes": list(self.reason_codes),
            "slot": self.slot,
            "source_binding_sha256": self.source_binding_sha256,
            "split": self.split,
        }


def population_manifest_bytes(
    rows: Sequence[PopulationRow], *, enforce_frozen_totals: bool = True
) -> bytes:
    ordered = sorted(
        rows,
        key=lambda row: (
            0 if row.split == "train" else 1,
            bytes.fromhex(row.opaque_key_hex),
            row.slot,
        ),
    )
    if tuple(rows) != tuple(ordered):
        raise PackerError("population rows must already be in frozen bytewise order")
    identities = [(row.split, row.opaque_key_hex, row.slot) for row in ordered]
    if len(identities) != len(set(identities)):
        raise PackerError("population manifest has a duplicate identity")
    if enforce_frozen_totals:
        train = sum(row.split == "train" for row in ordered)
        val = sum(row.split == "val" for row in ordered)
        if (train, val) != (268, 134):
            raise PackerError("population manifest must contain exactly 268 train and 134 val rows")
    return canonical_json_bytes([row.as_dict() for row in ordered])
