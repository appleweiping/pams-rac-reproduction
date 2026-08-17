"""Learner-side feature-only I/O and split firewall for TempoRAC.

This module has no trusted-packer, vault, evaluator, pickle, or historical
PAMS dependency.  Paths are rejected from their text before any filesystem
operation, and archive structure is inspected before arrays are returned.
"""

from __future__ import annotations

import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from pams.temporac.contract import FEATURE_MEMBER_NAMES, FEATURE_MEMBER_SCHEMA
from pams.temporac.hashio import read_deterministic_npz
from pams.temporac.types import ContractError, FeatureRecord, GenericArray

_FEATURE_FILENAME = re.compile(r"(?P<key>[0-9a-f]{64})\.(?P<slot>0|[1-9][0-9]*)\.npz\Z")
_FORBIDDEN_PATH_PARTS = frozenset(
    {"test", "sealed", "heldout", "results", "output", "server", "vault", "audit"}
)
_FORBIDDEN_PATH_MARKERS = ("v4x", "v46", "official-complete", "precommit", "selection")
_PRIVILEGED_KEYS = frozenset(
    {
        "association",
        "bbox",
        "bounding_box",
        "boundaries",
        "boundary",
        "classification",
        "count",
        "count_gt",
        "coverage",
        "cycle_boundary",
        "density",
        "density_gt",
        "evaluator",
        "metric",
        "object_id",
        "object_ids",
        "path",
        "paths",
        "period",
        "periodicity",
        "periods",
        "person_object_ids",
        "provenance",
        "source_id",
        "source_path",
        "vault",
        "video_id",
        "video_name",
    }
)


class FeatureIOError(ContractError):
    """A learner-side path, key, archive, or routing firewall violation."""


def reject_forbidden_path(path: Path) -> Path:
    """Reject forbidden path classes before ``stat``, open, hash, or load."""

    candidate = Path(path)
    raw_parts = tuple(part.casefold() for part in candidate.parts)
    if any(part in _FORBIDDEN_PATH_PARTS for part in raw_parts) or any(
        marker in part for part in raw_parts for marker in _FORBIDDEN_PATH_MARKERS
    ):
        raise FeatureIOError("path belongs to a forbidden evaluator/test/result class")
    return candidate


def reject_privileged_keys(payload: object) -> None:
    """Recursively reject evaluator/vault keys without interpreting values."""

    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if not isinstance(key, str):
                raise FeatureIOError("learner mapping keys must be strings")
            if key.casefold() in _PRIVILEGED_KEYS:
                raise FeatureIOError(f"privileged learner key is forbidden: {key}")
            reject_privileged_keys(value)
    elif isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for value in payload:
            reject_privileged_keys(value)


@dataclass(frozen=True, slots=True)
class RoutedFeature:
    """Feature payload plus routing metadata that is never a model tensor."""

    feature: FeatureRecord

    @property
    def identity(self) -> tuple[bytes, int]:
        return self.feature.opaque_key_bytes, self.feature.slot

    def preprocessing_arrays(self) -> Mapping[str, GenericArray]:
        """Return the five allowed preprocessing arrays, excluding key and slot."""

        return MappingProxyType(
            {
                "frame_mask": self.feature.frame_mask,
                "motion": self.feature.motion,
                "person_mask": self.feature.person_mask,
                "sampled_frame_indices": self.feature.sampled_frame_indices,
                "source_length": self.feature.source_length,
            }
        )


def _record_from_arrays(arrays: Mapping[str, GenericArray]) -> FeatureRecord:
    if tuple(arrays) != FEATURE_MEMBER_NAMES:
        raise FeatureIOError("feature arrays are not in the exact seven-member order")
    return FeatureRecord(**cast(Any, dict(arrays)))


def load_feature_archive(path: Path, *, enforce_filename: bool = True) -> RoutedFeature:
    """Load one canonical seven-member feature artifact and no other format."""

    candidate = reject_forbidden_path(path)
    match = _FEATURE_FILENAME.fullmatch(candidate.name)
    if enforce_filename and match is None:
        raise FeatureIOError("feature filename must be opaque-key.slot.npz")
    arrays, _ = read_deterministic_npz(candidate, schema=FEATURE_MEMBER_SCHEMA)
    record = _record_from_arrays(arrays)
    if (
        enforce_filename
        and match is not None
        and (
            match.group("key") != record.opaque_key_bytes.hex()
            or int(match.group("slot")) != record.slot
        )
    ):
        raise FeatureIOError("feature filename does not match its opaque routing identity")
    return RoutedFeature(record)


def load_feature_split(features_root: Path, split: str) -> tuple[RoutedFeature, ...]:
    """Load a bytewise-ordered train/val directory rooted exactly at ``features``."""

    if split not in {"train", "val"}:
        raise FeatureIOError("feature split must be exact ASCII train or val")
    root = reject_forbidden_path(features_root)
    if root.name != "features":
        raise FeatureIOError("learner root basename must be exactly features")
    split_root = root / split
    for candidate, role in ((root, "feature root"), (split_root, "feature split")):
        try:
            metadata = candidate.lstat()
        except FileNotFoundError as exc:
            raise FeatureIOError(f"{role} does not exist") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise FeatureIOError(f"{role} must be a non-symlink directory")
    entries = sorted(split_root.iterdir(), key=lambda value: value.name.encode("utf-8"))
    if not entries:
        raise FeatureIOError("feature split must not be empty")
    if any(_FEATURE_FILENAME.fullmatch(entry.name) is None for entry in entries):
        raise FeatureIOError("feature split contains a noncanonical filename")
    routed = tuple(load_feature_archive(entry) for entry in entries)
    identities = tuple(item.identity for item in routed)
    if len(identities) != len(set(identities)):
        raise FeatureIOError("feature split contains a duplicate routing identity")
    return routed


def validate_feature_mapping_keys(payload: Mapping[str, object]) -> None:
    """Fail before array construction on missing, extra, or privileged fields."""

    reject_privileged_keys(payload)
    if tuple(sorted(payload, key=lambda value: value.encode("ascii"))) != FEATURE_MEMBER_NAMES:
        raise FeatureIOError("feature mapping does not match the exact seven-field whitelist")
