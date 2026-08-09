"""Shared label-free runtime helpers for cycle-back audits and probes."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor

from pams.config import SkeletonAugmentationConfig
from pams.conventional_cycleback.config import ConventionalCycleBackConfig
from pams.conventional_cycleback.windows import (
    NativeWindowPairBatch,
    enumerate_native_window_pairs,
)
from pams.data import (
    PoseCacheEntryReceipt,
    PoseCacheMetadata,
    PoseCacheSetSnapshot,
    pose_cache_path,
)
from pams.model import PAMSEncoder
from pams.training import PoseBatch, collate_pose_sequences
from pams.types import PoseSequence

_FORBIDDEN_SNAPSHOT_KEYS = frozenset(
    {
        "action",
        "actions",
        "count",
        "counts",
        "label",
        "labels",
        "target",
        "targets",
        "dev",
        "development",
        "test",
    }
)


@dataclass(frozen=True, slots=True)
class Unified2DSequence:
    """One v4e pose sequence plus its explicit COCO17 joint-valid mask."""

    pose: PoseSequence
    joint_valid_mask: np.ndarray

    def __post_init__(self) -> None:
        mask = np.asarray(self.joint_valid_mask)
        if mask.dtype != np.bool_ or mask.shape != (self.pose.num_frames, 17):
            raise ValueError("joint_valid_mask must be bool [frames,17]")
        if mask[self.pose.valid_mask == 0].any():
            raise ValueError("invalid frames cannot contain valid joints")
        immutable = np.frombuffer(
            np.ascontiguousarray(mask).tobytes(order="C"),
            dtype=np.bool_,
        ).reshape(mask.shape)
        object.__setattr__(self, "joint_valid_mask", immutable)

    @property
    def video_id(self) -> str:
        return self.pose.video_id

    @property
    def num_frames(self) -> int:
        return self.pose.num_frames

    @property
    def valid_mask(self) -> np.ndarray:
        return self.pose.valid_mask

    @property
    def valid_fraction(self) -> float:
        return self.pose.valid_fraction


@dataclass(frozen=True, slots=True)
class Unified2DBatch:
    """Padded pose batch with the independently authorized joint mask."""

    pose_batch: PoseBatch
    joint_valid_mask: Tensor

    def __post_init__(self) -> None:
        if (
            self.joint_valid_mask.dtype is not torch.bool
            or self.joint_valid_mask.shape
            != (*self.pose_batch.valid_mask.shape, 17)
            or self.joint_valid_mask.device != self.pose_batch.poses.device
        ):
            raise ValueError("joint-valid batch must be bool [batch,time,17]")
        if bool(
            (
                self.joint_valid_mask
                & ~self.pose_batch.valid_mask.unsqueeze(-1)
            ).any()
        ):
            raise ValueError("invalid frames cannot carry valid joints")

    @property
    def video_ids(self) -> tuple[str, ...]:
        return self.pose_batch.video_ids

    @property
    def poses(self) -> Tensor:
        return self.pose_batch.poses

    @property
    def valid_mask(self) -> Tensor:
        return self.pose_batch.valid_mask

    @property
    def lengths(self) -> Tensor:
        return self.pose_batch.lengths


@dataclass(frozen=True, slots=True)
class AugmentedSequenceViews:
    """Two full native-timeline views before any window embedding is sliced."""

    video_ids: tuple[str, ...]
    poses_a: Tensor
    poses_b: Tensor
    valid_mask: Tensor
    joint_valid_mask: Tensor
    lengths: Tensor
    view_seeds: tuple[int, int]

    def __post_init__(self) -> None:
        if self.poses_a.ndim != 4 or self.poses_a.shape[2:] != (33, 3):
            raise ValueError("augmented sequence views must be [batch,time,33,3]")
        if self.poses_b.shape != self.poses_a.shape:
            raise ValueError("augmented sequence views must have identical shapes")
        batch, time = self.poses_a.shape[:2]
        if len(self.video_ids) != batch or len(set(self.video_ids)) != batch:
            raise ValueError("augmented sequence video identities must be unique")
        if self.valid_mask.shape != (batch, time) or (
            self.valid_mask.dtype is not torch.bool
        ):
            raise ValueError("augmented sequence validity must be bool [batch,time]")
        if (
            self.joint_valid_mask.ndim != 3
            or self.joint_valid_mask.shape[:2] != (batch, time)
            or self.joint_valid_mask.shape[2] < 1
            or self.joint_valid_mask.dtype is not torch.bool
        ):
            raise ValueError(
                "augmented sequence joint validity must be bool "
                "[batch,time,support_channels]"
            )
        if self.lengths.shape != (batch,) or self.lengths.dtype != torch.long:
            raise ValueError("augmented sequence lengths must be int64 [batch]")
        tensors = (
            self.poses_b,
            self.valid_mask,
            self.joint_valid_mask,
            self.lengths,
        )
        if any(value.device != self.poses_a.device for value in tensors):
            raise ValueError("augmented sequence tensors must share one device")
        if self.view_seeds[0] == self.view_seeds[1]:
            raise ValueError("augmented sequence view seeds must be distinct")


PairContextObjectiveRole = Literal[
    "real_optimizer",
    "diagnostic_zero",
    "diagnostic_temporal_permutation",
    "diagnostic_pe_permutation",
    "diagnostic_joint_mask_flicker",
    "diagnostic_joint_torso_only",
    "diagnostic_joint_alternating_limb_dropout",
]


@dataclass(frozen=True, slots=True)
class PairSegmentContexts:
    """Per-pair full authorized stable-range contexts used by the encoder."""

    video_ids: tuple[str, ...]
    context_keys_a: tuple[str, ...]
    context_keys_b: tuple[str, ...]
    poses_a: tuple[Tensor, ...]
    poses_b: tuple[Tensor, ...]
    joint_valid_a: tuple[Tensor, ...]
    joint_valid_b: tuple[Tensor, ...]
    position_indices_a: tuple[Tensor, ...]
    position_indices_b: tuple[Tensor, ...]
    view_seeds: tuple[int, int]
    objective_role: PairContextObjectiveRole
    transform_contract: Mapping[str, Any]

    def __post_init__(self) -> None:
        pair_count = len(self.video_ids)
        collections = (
            self.context_keys_a,
            self.context_keys_b,
            self.poses_a,
            self.poses_b,
            self.joint_valid_a,
            self.joint_valid_b,
            self.position_indices_a,
            self.position_indices_b,
        )
        if any(len(values) != pair_count for values in collections):
            raise ValueError("pair-segment context collections must match pair count")
        if any(not key for key in self.context_keys_a + self.context_keys_b):
            raise ValueError("pair-segment context keys must be non-empty")
        support_channels: int | None = None
        for index in range(pair_count):
            first = self.poses_a[index]
            second = self.poses_b[index]
            if first.ndim != 3 or first.shape[1:] != (33, 3) or second.shape != first.shape:
                raise ValueError("pair-segment poses must be [segment,33,3]")
            length = first.shape[0]
            for mask in (self.joint_valid_a[index], self.joint_valid_b[index]):
                if (
                    mask.ndim != 2
                    or mask.shape[0] != length
                    or mask.shape[1] < 1
                    or mask.dtype is not torch.bool
                ):
                    raise ValueError(
                        "pair-segment joint mask must be bool "
                        "[segment,support_channels]"
                    )
                if support_channels is None:
                    support_channels = int(mask.shape[1])
                elif int(mask.shape[1]) != support_channels:
                    raise ValueError(
                        "pair-segment contexts mix support-channel schemas"
                    )
            for positions in (
                self.position_indices_a[index],
                self.position_indices_b[index],
            ):
                if positions.shape != (length,) or positions.dtype != torch.long:
                    raise ValueError("pair-segment positions must be int64 [segment]")
            tensors = (
                second,
                self.joint_valid_a[index],
                self.joint_valid_b[index],
                self.position_indices_a[index],
                self.position_indices_b[index],
            )
            if any(value.device != first.device for value in tensors):
                raise ValueError("pair-segment context tensors must share one device")
        if self.view_seeds[0] == self.view_seeds[1]:
            raise ValueError("pair-segment view seeds must be distinct")
        allowed_roles = {
            "real_optimizer",
            "diagnostic_zero",
            "diagnostic_temporal_permutation",
            "diagnostic_pe_permutation",
            "diagnostic_joint_mask_flicker",
            "diagnostic_joint_torso_only",
            "diagnostic_joint_alternating_limb_dropout",
        }
        if self.objective_role not in allowed_roles:
            raise ValueError("pair-segment context objective role is invalid")


@dataclass(frozen=True, slots=True)
class JointMaskEntryReceipt:
    video_id: str
    sidecar_sha256: str
    bytes: int

    def __post_init__(self) -> None:
        if not self.video_id or len(self.sidecar_sha256) != 64 or self.bytes < 1:
            raise ValueError("invalid joint-mask entry receipt")
        if any(character not in "0123456789abcdef" for character in self.sidecar_sha256):
            raise ValueError("invalid joint-mask SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "sidecar_sha256": self.sidecar_sha256,
            "bytes": self.bytes,
        }


@dataclass(frozen=True, slots=True)
class JointMaskSetSnapshot:
    entries: tuple[JointMaskEntryReceipt, ...]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.entries, key=lambda item: item.video_id))
        if not ordered or len({item.video_id for item in ordered}) != len(ordered):
            raise ValueError("joint-mask snapshot identities must be unique")
        object.__setattr__(self, "entries", ordered)

    @property
    def fingerprint(self) -> str:
        return canonical_json_sha256(
            {
                "schema_version": 1,
                "artifact_type": "pams_cycleback_unified_2d_joint_mask_set_snapshot_v1",
                "entries": [entry.to_dict() for entry in self.entries],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_type": "pams_cycleback_unified_2d_joint_mask_set_snapshot_v1",
            "entry_count": len(self.entries),
            "fingerprint": self.fingerprint,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def joint_mask_path(root: str | Path, video_id: str) -> Path:
    digest = hashlib.sha256(video_id.encode("utf-8")).hexdigest()
    return Path(root) / f"{digest}.joint-mask.npz"


def load_joint_mask_snapshot(path: str | Path) -> JointMaskSetSnapshot:
    payload, _ = load_stable_json_object(path, role="joint-mask snapshot")
    if set(payload) != {
        "schema_version",
        "artifact_type",
        "entry_count",
        "fingerprint",
        "entries",
    } or (
        payload["schema_version"] != 1
        or payload["artifact_type"]
        != "pams_cycleback_unified_2d_joint_mask_set_snapshot_v1"
        or not isinstance(payload["entries"], list)
    ):
        raise ValueError("joint-mask snapshot schema mismatch")
    entries: list[JointMaskEntryReceipt] = []
    for raw in payload["entries"]:
        if not isinstance(raw, dict) or set(raw) != {
            "video_id",
            "sidecar_sha256",
            "bytes",
        }:
            raise ValueError("joint-mask snapshot entry schema mismatch")
        entries.append(JointMaskEntryReceipt(**raw))
    snapshot = JointMaskSetSnapshot(entries=tuple(entries))
    if payload["entry_count"] != len(snapshot.entries) or (
        payload["fingerprint"] != snapshot.fingerprint
    ):
        raise ValueError("joint-mask snapshot count/fingerprint mismatch")
    return snapshot


def load_joint_mask_sidecar(
    path: str | Path,
    *,
    expected: JointMaskEntryReceipt,
    native_length: int,
) -> np.ndarray:
    encoded, identity = stable_file_bytes(path)
    if identity != (expected.sidecar_sha256, expected.bytes):
        raise ValueError("joint-mask sidecar bytes differ from snapshot")
    with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
        if set(archive.files) != {"schema_version", "joint_valid_mask"}:
            raise ValueError("joint-mask sidecar archive schema mismatch")
        schema = np.asarray(archive["schema_version"])
        mask = np.asarray(archive["joint_valid_mask"])
    if schema.shape != () or schema.dtype != np.int64 or int(schema) != 1:
        raise ValueError("joint-mask sidecar schema version mismatch")
    if mask.dtype != np.bool_ or mask.shape != (native_length, 17):
        raise ValueError("joint-mask sidecar must be bool [native_length,17]")
    return mask


@dataclass(frozen=True, slots=True)
class PairEligibility:
    start_grid_policy: str
    frozen_thresholds: Mapping[str, float | int]
    minimum_window_stable_action_joints: int
    minimum_window_joint_support_fraction: float
    native_lengths: Mapping[str, int]
    representation_eligible: Mapping[str, bool]
    base_valid_starts_by_variant: Mapping[str, Mapping[str, tuple[int, ...]]]
    starts_by_variant: Mapping[str, Mapping[str, tuple[int, ...]]]
    identity: tuple[str, int]


def load_identity_map(
    path: str | Path,
    *,
    expected_video_ids: Sequence[str],
    expected_sha256: str,
) -> tuple[dict[str, str], tuple[str, int]]:
    payload, identity = load_stable_json_object(path, role="v4e identity map")
    if identity[0] != expected_sha256:
        raise ValueError("v4e identity map differs from representation authority")
    if set(payload) != {
        "schema_version",
        "artifact_type",
        "entry_count",
        "entries",
        "mapping_rule",
    } or (
        payload["schema_version"] != 1
        or payload["artifact_type"] != "pams_pose_recovery_v4e_identity_map_v1"
        or payload["mapping_rule"] != "utf8-video-id-sha256"
        or not isinstance(payload["entries"], list)
        or payload["entry_count"] != len(payload["entries"])
    ):
        raise ValueError("v4e identity-map schema mismatch")
    mapping: dict[str, str] = {}
    opaque_seen: set[str] = set()
    for raw in payload["entries"]:
        if not isinstance(raw, dict) or set(raw) != {"video_id", "video_id_sha256"}:
            raise ValueError("v4e identity-map entry schema mismatch")
        identifier = raw["video_id"]
        opaque = raw["video_id_sha256"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in mapping
            or not isinstance(opaque, str)
            or len(opaque) != 64
            or any(character not in "0123456789abcdef" for character in opaque)
            or opaque in opaque_seen
            or hashlib.sha256(identifier.encode("utf-8")).hexdigest() != opaque
        ):
            raise ValueError("v4e identity-map entry identity mismatch")
        mapping[identifier] = opaque
        opaque_seen.add(opaque)
    if set(mapping) != set(expected_video_ids):
        raise ValueError("v4e identity-map membership differs from pose snapshot")
    return mapping, identity


def load_pair_eligibility(
    path: str | Path,
    *,
    expected_sha256: str,
    identity_map: Mapping[str, str],
) -> PairEligibility:
    payload, identity = load_stable_json_object(path, role="cycleback pair eligibility")
    if identity[0] != expected_sha256:
        raise ValueError("pair eligibility differs from representation authority")
    expected_root = {
        "schema_version",
        "artifact_type",
        "policy",
        "start_grid_policy",
        "variant_geometry",
        "frozen_variant_aliases",
        "alias_semantics",
        "entry_count",
        "frozen_thresholds",
        "consumer_may_expand_starts",
        "label_free",
        "entries",
    }
    if set(payload) != expected_root or (
        payload["schema_version"] != 1
        or payload["artifact_type"]
        != "pams_pose_recovery_v4e_cycleback_pair_eligibility_v1"
        or payload["policy"] != "representation_authorized_exact_2w_starts_only"
        or payload["start_grid_policy"]
        != "native_zero_based_start_mod_hop_equals_zero"
        or payload["variant_geometry"]
        != {
            "W16_H2": {"window_frames": 16, "hop_frames": 2},
            "W16_H4": {"window_frames": 16, "hop_frames": 4},
            "W16_H4_PE0": {"window_frames": 16, "hop_frames": 4},
            "W24_H4": {"window_frames": 24, "hop_frames": 4},
        }
        or payload["frozen_variant_aliases"] != {"W16_H4_PE0": "W16_H4"}
        or payload["alias_semantics"]
        != "listed-starts-must-be-bytewise-equal-no-consumer-inference"
        or payload["consumer_may_expand_starts"] is not False
        or payload["label_free"] is not True
        or not isinstance(payload["entries"], list)
        or payload["entry_count"] != len(payload["entries"])
    ):
        raise ValueError("cycleback pair-eligibility schema mismatch")
    thresholds = payload["frozen_thresholds"]
    expected_threshold_keys = {
        "maximum_candidate_window_frames",
        "maximum_frame_center_step",
        "maximum_frame_log_scale_step",
        "maximum_frame_joint_mask_flicker_fraction",
        "minimum_dual_path_agreement",
        "minimum_frame_local_ambiguity_gap",
        "minimum_longest_trainable_segment_fraction",
        "minimum_longest_trainable_segment_frames",
        "minimum_source_coverage",
        "minimum_window_joint_support_fraction",
        "minimum_window_stable_action_joints",
    }
    if not isinstance(thresholds, dict) or set(thresholds) != expected_threshold_keys:
        raise ValueError("cycleback pair-eligibility threshold schema mismatch")
    minimum_joints = thresholds["minimum_window_stable_action_joints"]
    minimum_fraction = thresholds["minimum_window_joint_support_fraction"]
    if (
        isinstance(minimum_joints, bool)
        or not isinstance(minimum_joints, int)
        or not 1 <= minimum_joints <= 8
        or isinstance(minimum_fraction, bool)
        or not isinstance(minimum_fraction, float)
        or not 0.0 < minimum_fraction <= 1.0
        or thresholds["maximum_candidate_window_frames"] != 24
        or thresholds["minimum_longest_trainable_segment_frames"] != 48
        or any(
            isinstance(thresholds[key], bool)
            or not isinstance(thresholds[key], float)
            or not math.isfinite(thresholds[key])
            for key in expected_threshold_keys
            if key
            not in {
                "maximum_candidate_window_frames",
                "minimum_longest_trainable_segment_frames",
                "minimum_window_stable_action_joints",
            }
        )
        or any(
            not 0.0 <= thresholds[key] <= 1.0
            for key in {
                "maximum_frame_joint_mask_flicker_fraction",
                "minimum_dual_path_agreement",
                "minimum_longest_trainable_segment_fraction",
                "minimum_source_coverage",
                "minimum_window_joint_support_fraction",
            }
        )
        or any(
            thresholds[key] < 0.0
            for key in {
                "maximum_frame_center_step",
                "maximum_frame_log_scale_step",
                "minimum_frame_local_ambiguity_gap",
            }
        )
    ):
        raise ValueError("cycleback pair-eligibility threshold value mismatch")
    real_by_opaque = {opaque: real for real, opaque in identity_map.items()}
    starts_by_variant: dict[str, dict[str, tuple[int, ...]]] = {
        variant: {}
        for variant in ("W16_H2", "W16_H4", "W16_H4_PE0", "W24_H4")
    }
    variant_geometry = {
        "W16_H2": (16, 2),
        "W16_H4": (16, 4),
        "W16_H4_PE0": (16, 4),
        "W24_H4": (24, 4),
    }
    base_valid_starts_by_variant: dict[str, dict[str, tuple[int, ...]]] = {
        variant: {} for variant in starts_by_variant
    }
    native_lengths: dict[str, int] = {}
    representation_eligible: dict[str, bool] = {}
    for raw in payload["entries"]:
        if not isinstance(raw, dict) or set(raw) != {
            "video_id",
            "native_length",
            "representation_eligible",
            "base_valid_starts_by_variant",
            "starts_by_variant",
        }:
            raise ValueError("cycleback pair-eligibility entry schema mismatch")
        opaque = raw["video_id"]
        real = real_by_opaque.get(opaque)
        native_length = raw["native_length"]
        eligible = raw["representation_eligible"]
        base_variants = raw["base_valid_starts_by_variant"]
        variants = raw["starts_by_variant"]
        if (
            real is None
            or real in native_lengths
            or isinstance(native_length, bool)
            or not isinstance(native_length, int)
            or native_length < 1
            or not isinstance(eligible, bool)
            or not isinstance(base_variants, dict)
            or set(base_variants) != set(starts_by_variant)
            or not isinstance(variants, dict)
            or set(variants) != set(starts_by_variant)
        ):
            raise ValueError("cycleback pair-eligibility entry identity mismatch")
        native_lengths[real] = native_length
        representation_eligible[real] = eligible
        any_starts = False
        for variant in starts_by_variant:
            base_values = base_variants[variant]
            values = variants[variant]
            if (
                not isinstance(base_values, list)
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in base_values
                )
                or base_values != sorted(set(base_values))
                or not isinstance(values, list)
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in values
                )
                or values != sorted(set(values))
            ):
                raise ValueError("cycleback pair starts must be sorted unique integers")
            base_starts = tuple(base_values)
            starts = tuple(values)
            window, hop = variant_geometry[variant]
            maximum_start = native_length - 2 * window
            if any(
                start < 0 or start > maximum_start or start % hop
                for start in base_starts
            ):
                raise ValueError(
                    "base-valid pair start violates frozen native-zero grid geometry"
                )
            if not set(starts).issubset(base_starts):
                raise ValueError(
                    "eligible pair starts are not a subset of base-valid starts"
                )
            base_valid_starts_by_variant[variant][real] = base_starts
            starts_by_variant[variant][real] = starts
            any_starts = any_starts or bool(starts)
        if not eligible and (
            any_starts
            or any(
                base_valid_starts_by_variant[variant][real]
                for variant in base_valid_starts_by_variant
            )
        ):
            raise ValueError("ineligible representation entry contains pair starts")
        for collection in (base_valid_starts_by_variant, starts_by_variant):
            if collection["W16_H4_PE0"][real] != collection["W16_H4"][real]:
                raise ValueError("PE0 alias starts differ from explicit W16_H4 starts")
    if set(native_lengths) != set(identity_map):
        raise ValueError("pair-eligibility membership differs from identity map")
    return PairEligibility(
        start_grid_policy=payload["start_grid_policy"],
        frozen_thresholds=dict(thresholds),
        minimum_window_stable_action_joints=minimum_joints,
        minimum_window_joint_support_fraction=minimum_fraction,
        native_lengths=native_lengths,
        representation_eligible=representation_eligible,
        base_valid_starts_by_variant=base_valid_starts_by_variant,
        starts_by_variant=starts_by_variant,
        identity=identity,
    )


def json_loads_no_duplicates(encoded: bytes | str, *, role: str) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{role} contains duplicate JSON field {key!r}")
            result[key] = value
        return result

    return json.loads(encoded, object_pairs_hook=reject)


def load_stable_json_object(
    path: str | Path,
    *,
    role: str,
) -> tuple[dict[str, Any], tuple[str, int]]:
    encoded, identity = stable_file_bytes(path)
    try:
        value = json_loads_no_duplicates(encoded, role=role)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{role} must be a JSON object")
    return value, identity


def stable_file_identity(path: str | Path) -> tuple[str, int]:
    """Hash a stable regular non-symlink file."""

    encoded, identity = stable_file_bytes(path)
    if len(encoded) != identity[1]:
        raise RuntimeError("stable file byte count mismatch")
    return identity


def stable_file_bytes(path: str | Path) -> tuple[bytes, tuple[str, int]]:
    """Read/hash one stable O_NOFOLLOW descriptor exactly once."""

    source = Path(path)
    before = source.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"input must be a regular non-symlink file: {source}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    descriptor = os.open(source, flags)
    digest = hashlib.sha256()
    collected = bytearray()
    byte_count = 0
    try:
        opened = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                collected.extend(chunk)
                byte_count += len(chunk)
            closed = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = source.lstat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(closed, field)
        or getattr(closed, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise RuntimeError(f"input changed while being hashed: {source}")
    if byte_count != after.st_size:
        raise RuntimeError(f"input size changed while being hashed: {source}")
    return bytes(collected), (digest.hexdigest(), byte_count)


def _reject_privileged_snapshot_keys(encoded: str) -> None:
    """Reject privileged JSON keys before decoding any values."""

    index = 0
    while index < len(encoded):
        if encoded[index] != '"':
            index += 1
            continue
        start = index
        index += 1
        escaped = False
        while index < len(encoded):
            character = encoded[index]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                index += 1
                break
            index += 1
        else:
            return
        cursor = index
        while cursor < len(encoded) and encoded[cursor] in " \t\r\n":
            cursor += 1
        if cursor >= len(encoded) or encoded[cursor] != ":":
            continue
        try:
            key = json.loads(encoded[start:index])
        except json.JSONDecodeError:
            return
        if isinstance(key, str) and key.lower() in _FORBIDDEN_SNAPSHOT_KEYS:
            raise ValueError(f"pose snapshot contains privileged field {key!r}")


def load_pose_snapshot(path: str | Path) -> PoseCacheSetSnapshot:
    """Load an exact cache-only snapshot with no label-bearing surface."""

    encoded_bytes, _ = stable_file_bytes(path)
    try:
        encoded = encoded_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("pose snapshot is not valid UTF-8") from exc
    _reject_privileged_snapshot_keys(encoded)
    payload = json_loads_no_duplicates(encoded, role="pose snapshot")
    expected = {
        "schema_version",
        "pose_fingerprint",
        "fingerprint",
        "entry_count",
        "entries",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("pose snapshot schema mismatch")
    if payload["schema_version"] != 1 or not isinstance(payload["entries"], list):
        raise ValueError("pose snapshot schema version or entries mismatch")
    entries: list[PoseCacheEntryReceipt] = []
    for raw in payload["entries"]:
        if not isinstance(raw, dict) or set(raw) != {"video_id", "cache_sha256", "bytes"}:
            raise ValueError("pose snapshot entry schema mismatch")
        entries.append(PoseCacheEntryReceipt(**raw))
    snapshot = PoseCacheSetSnapshot(
        schema_version=payload["schema_version"],
        pose_fingerprint=payload["pose_fingerprint"],
        entries=tuple(entries),
    )
    if payload["entry_count"] != len(snapshot.entries):
        raise ValueError("pose snapshot entry count mismatch")
    if payload["fingerprint"] != snapshot.fingerprint:
        raise ValueError("pose snapshot fingerprint mismatch")
    return snapshot


def load_segment_index(
    path: str | Path,
    *,
    expected_video_ids: Sequence[str],
    expected_sha256: str,
    expected_segment_reset_policy_sha256: str,
    identity_map: Mapping[str, str],
    pair_eligibility: PairEligibility,
) -> tuple[dict[str, tuple[tuple[int, int], ...]], tuple[str, int]]:
    """Load the label-free reset/segment sidecar bound by representation PASS."""

    payload, identity = load_stable_json_object(path, role="pose segment index")
    if identity[0] != expected_sha256:
        raise ValueError("pose segment index differs from representation authority")
    expected_fields = {
        "schema_version",
        "artifact_type",
        "segment_reset_policy_sha256",
        "entry_count",
        "entries",
        "all_resets_explicit_and_invalid_or_segment_sidecar_complete",
    }
    if set(payload) != expected_fields:
        raise ValueError("pose segment index schema mismatch")
    if (
        payload["schema_version"] != 1
        or payload["artifact_type"] != "pams_pose_recovery_v4e_segment_index_v1"
        or payload["segment_reset_policy_sha256"]
        != expected_segment_reset_policy_sha256
        or payload["all_resets_explicit_and_invalid_or_segment_sidecar_complete"]
        is not True
    ):
        raise ValueError("pose segment index representation scope mismatch")
    rows = payload["entries"]
    if not isinstance(rows, list) or payload["entry_count"] != len(rows):
        raise ValueError("pose segment index entry count mismatch")
    expected_ids = tuple(expected_video_ids)
    if set(identity_map) != set(expected_ids):
        raise ValueError("identity map differs from expected snapshot membership")
    real_by_opaque = {opaque: real for real, opaque in identity_map.items()}
    result: dict[str, tuple[tuple[int, int], ...]] = {}
    for raw in rows:
        if not isinstance(raw, dict) or set(raw) != {
            "video_id",
            "native_length",
            "association_resets",
            "eligible_frame_ranges",
            "eligible_pair_starts_by_candidate",
            "representation_eligible",
            "ineligibility_reason",
            "no_unreported_internal_reset",
        }:
            raise ValueError("pose segment index entry schema mismatch")
        identifier = real_by_opaque.get(raw["video_id"])
        native_length = raw["native_length"]
        if (
            identifier is None
            or identifier in result
            or isinstance(native_length, bool)
            or not isinstance(native_length, int)
            or native_length < 1
            or not isinstance(raw["representation_eligible"], bool)
            or raw["no_unreported_internal_reset"] is not True
        ):
            raise ValueError("pose segment index entry identity is invalid")
        if native_length != pair_eligibility.native_lengths.get(identifier):
            raise ValueError("segment native length differs from pair eligibility")
        if (
            raw["representation_eligible"]
            != pair_eligibility.representation_eligible.get(identifier)
        ):
            raise ValueError(
                "segment eligibility differs from cycleback pair eligibility"
            )
        raw_segments = raw["eligible_frame_ranges"]
        if not isinstance(raw_segments, list):
            raise ValueError("pose segment index segments must be a list")
        eligible = raw["representation_eligible"]
        reason = raw["ineligibility_reason"]
        if eligible:
            if not raw_segments or reason is not None:
                raise ValueError("eligible pose entry requires segments and no reason")
        elif (
            raw_segments
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError(
                "ineligible pose entry requires no consumable ranges and a reason"
            )
        segments: list[tuple[int, int]] = []
        previous_end = 0
        for item in raw_segments:
            if (
                not isinstance(item, dict)
                or set(item) != {"start", "stop", "frames"}
                or any(
                    isinstance(item[key], bool) or not isinstance(item[key], int)
                    for key in ("start", "stop", "frames")
                )
            ):
                raise ValueError("pose segment range schema mismatch")
            start, end = item["start"], item["stop"]
            if not (0 <= start < end <= native_length) or (
                segments and start < previous_end
            ) or item["frames"] != end - start:
                raise ValueError("pose segments must be ordered, disjoint, and in bounds")
            segments.append((start, end))
            previous_end = end
        reset_frames = raw["association_resets"]
        if (
            not isinstance(reset_frames, list)
            or reset_frames != sorted(set(reset_frames))
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 < value < native_length
                for value in reset_frames
            )
        ):
            raise ValueError("association reset schema mismatch")
        if any(
            start < reset < end
            for start, end in segments
            for reset in reset_frames
        ):
            raise ValueError("eligible frame range crosses an association reset")
        starts_by_candidate = raw["eligible_pair_starts_by_candidate"]
        if (
            not isinstance(starts_by_candidate, dict)
            or set(starts_by_candidate)
            != set(pair_eligibility.starts_by_variant)
        ):
            raise ValueError("segment candidate-start schema mismatch")
        for variant, starts in starts_by_candidate.items():
            if starts != list(
                pair_eligibility.starts_by_variant[variant][identifier]
            ):
                raise ValueError(
                    "segment candidate starts differ from pair eligibility"
                )
        result[identifier] = tuple(segments) if eligible else ()
    if set(result) != set(expected_ids):
        raise ValueError("pose segment index order/membership differs from snapshot")
    return {identifier: result[identifier] for identifier in expected_ids}, identity


def augment_unified_2d_pose_batch(
    poses: Tensor,
    valid_mask: Tensor,
    joint_valid_mask: Tensor,
    config: SkeletonAugmentationConfig,
    *,
    generator: torch.Generator,
) -> Tensor:
    """Augment only COCO17 XY while preserving the v4e padded representation."""

    if poses.ndim != 4 or poses.shape[2:] != (33, 3):
        raise ValueError("unified-2D poses must have shape [batch,time,33,3]")
    if joint_valid_mask.shape != (*poses.shape[:2], 17) or (
        joint_valid_mask.dtype is not torch.bool
    ):
        raise ValueError("unified-2D joint mask must be bool [batch,time,17]")
    if not bool((poses[:, :, 17:, :] == 0).all()):
        raise ValueError("unified-2D padded joints must be exact zero before augmentation")
    if not bool((poses[:, :, :17, 2] == 0).all()):
        raise ValueError("unified-2D z coordinates must be exact zero before augmentation")
    if tuple(config.rotation_degrees[:2]) != (0.0, 0.0):
        raise ValueError("unified-2D augmentation permits only in-plane z rotation")
    if bool((joint_valid_mask & ~valid_mask.unsqueeze(-1)).any()):
        raise ValueError("invalid frames cannot carry valid joints")
    if not bool((poses[:, :, :17, :][~joint_valid_mask] == 0).all()):
        raise ValueError("masked COCO17 joints must be zero before augmentation")
    batch_size = poses.shape[0]
    radians = float(config.rotation_degrees[2]) * math.pi / 180.0
    angles = (
        2.0
        * torch.rand(
            (batch_size,),
            dtype=poses.dtype,
            device=poses.device,
            generator=generator,
        )
        - 1.0
    ) * radians
    cosine = torch.cos(angles).view(batch_size, 1, 1)
    sine = torch.sin(angles).view(batch_size, 1, 1)
    active_xy = poses[:, :, :17, :2]
    rotated_x = cosine * active_xy[..., 0] - sine * active_xy[..., 1]
    rotated_y = sine * active_xy[..., 0] + cosine * active_xy[..., 1]
    rotated = torch.stack((rotated_x, rotated_y), dim=-1)
    scale_minimum, scale_maximum = config.scale_range
    scales = scale_minimum + (scale_maximum - scale_minimum) * torch.rand(
        (batch_size, 1, 1, 1),
        dtype=poses.dtype,
        device=poses.device,
        generator=generator,
    )
    active_xy = rotated * scales
    if config.jitter_std > 0.0:
        jitter = torch.randn(
            active_xy.shape,
            dtype=poses.dtype,
            device=poses.device,
            generator=generator,
        )
        active_xy = active_xy + config.jitter_std * jitter
    active_xy = active_xy.masked_fill(~joint_valid_mask.unsqueeze(-1), 0.0)
    result = torch.zeros_like(poses)
    result[:, :, :17, :2] = active_xy
    if not bool((result[:, :, 17:, :] == 0).all()) or not bool(
        (result[:, :, :, 2] == 0).all()
    ):
        raise RuntimeError("unified-2D augmentation changed padded joints or z")
    return result


def load_snapshot_sequences(
    pose_cache_dir: str | Path,
    snapshot: PoseCacheSetSnapshot,
    joint_mask_dir: str | Path,
    joint_mask_snapshot: JointMaskSetSnapshot,
    *,
    expected_video_total: int,
) -> tuple[Unified2DSequence, ...]:
    """Read and re-hash every cache byte stream bound by a snapshot."""

    if len(snapshot.entries) != expected_video_total:
        raise ValueError(
            f"expected {expected_video_total} training pose caches, "
            f"received {len(snapshot.entries)}"
        )
    cache_root = Path(pose_cache_dir)
    if cache_root.is_symlink() or not cache_root.is_dir():
        raise ValueError("pose cache root must be a non-symlink directory")
    if len(joint_mask_snapshot.entries) != expected_video_total or tuple(
        entry.video_id for entry in joint_mask_snapshot.entries
    ) != tuple(entry.video_id for entry in snapshot.entries):
        raise ValueError("joint-mask snapshot membership differs from pose snapshot")
    joint_root = Path(joint_mask_dir)
    if joint_root.is_symlink() or not joint_root.is_dir():
        raise ValueError("joint-mask root must be a non-symlink directory")
    joint_by_id = {entry.video_id: entry for entry in joint_mask_snapshot.entries}
    sequences: list[Unified2DSequence] = []
    observed_receipts: list[PoseCacheEntryReceipt] = []
    for expected in snapshot.entries:
        cache_path = pose_cache_path(cache_root, expected.video_id)
        if cache_path.is_symlink() or not cache_path.is_file():
            raise ValueError("pose cache entry must be a regular non-symlink file")
        pose, _, receipt = load_unified_2d_pose_cache_with_receipt(
            cache_path,
            expected_pose_fingerprint=snapshot.pose_fingerprint,
        )
        if receipt != expected or pose.video_id != expected.video_id:
            raise ValueError("pose cache bytes differ from the declared train337 snapshot")
        joint_mask = load_joint_mask_sidecar(
            joint_mask_path(joint_root, expected.video_id),
            expected=joint_by_id[expected.video_id],
            native_length=pose.num_frames,
        )
        sequence = validate_unified_2d_sequence(pose, joint_mask)
        sequences.append(sequence)
        observed_receipts.append(receipt)
    observed = PoseCacheSetSnapshot(
        pose_fingerprint=snapshot.pose_fingerprint,
        entries=tuple(observed_receipts),
    )
    if observed.fingerprint != snapshot.fingerprint:
        raise RuntimeError("pose cache set changed during snapshot verification")
    return tuple(sequences)


def load_unified_2d_pose_cache_with_receipt(
    path: str | Path,
    *,
    expected_pose_fingerprint: str,
) -> tuple[PoseSequence, PoseCacheMetadata, PoseCacheEntryReceipt]:
    """Read one v4e cache and recheck representation invariants from raw bytes."""

    encoded, (cache_sha256, byte_count) = stable_file_bytes(path)
    with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
        if set(archive.files) != {"xyz", "valid_mask", "metadata"}:
            raise ValueError("unified-2D pose cache archive schema mismatch")
        raw_metadata = archive["metadata"]
        if raw_metadata.ndim != 0:
            raise ValueError("unified-2D pose cache metadata must be scalar JSON")
        metadata_payload = json_loads_no_duplicates(
            str(raw_metadata.item()),
            role="unified-2D pose cache metadata",
        )
        if not isinstance(metadata_payload, dict):
            raise ValueError("unified-2D pose cache metadata must be an object")
        try:
            metadata = PoseCacheMetadata(**metadata_payload)
        except TypeError as exc:
            raise ValueError("unified-2D pose cache metadata schema mismatch") from exc
        xyz = np.asarray(archive["xyz"])
        valid_mask = np.asarray(archive["valid_mask"])
    if xyz.dtype != np.float32 or xyz.ndim != 3 or xyz.shape[1:] != (33, 3):
        raise ValueError("unified-2D xyz must be float32 [frames,33,3]")
    if valid_mask.dtype != np.bool_ or valid_mask.shape != (xyz.shape[0],):
        raise ValueError("unified-2D valid_mask must be bool [frames]")
    if metadata.frames != xyz.shape[0]:
        raise ValueError("unified-2D metadata frame count mismatch")
    if metadata.pose_fingerprint != expected_pose_fingerprint:
        raise ValueError("unified-2D pose fingerprint mismatch")
    if not np.isfinite(xyz[valid_mask]).all():
        raise ValueError("unified-2D valid coordinates must be finite")
    if not np.equal(xyz[~valid_mask], np.float32(0.0)).all():
        raise ValueError("unified-2D invalid frames must be exact zero in raw cache")
    if not np.equal(xyz[:, 17:, :], np.float32(0.0)).all():
        raise ValueError("unified-2D padded joints 17:33 must remain exact zero")
    if not np.equal(xyz[:, :17, 2], np.float32(0.0)).all():
        raise ValueError("unified-2D active-joint z coordinates must remain exact zero")
    sequence = PoseSequence(
        video_id=metadata.video_id,
        fps=metadata.fps,
        xyz=xyz,
        valid_mask=valid_mask,
    )
    return (
        sequence,
        metadata,
        PoseCacheEntryReceipt(
            video_id=sequence.video_id,
            cache_sha256=cache_sha256,
            bytes=byte_count,
        ),
    )


def validate_unified_2d_sequence(
    sequence: PoseSequence,
    joint_valid_mask: np.ndarray,
) -> Unified2DSequence:
    """Close body-center/RMS and weak-joint invariants using the mask sidecar."""

    mask = np.asarray(joint_valid_mask)
    if mask.dtype != np.bool_ or mask.shape != (sequence.num_frames, 17):
        raise ValueError("joint-valid sidecar must be bool [frames,17]")
    if mask[~sequence.valid_mask].any():
        raise ValueError("invalid frames cannot carry valid COCO17 joints")
    if not np.equal(sequence.xyz[:, :17, :][~mask], np.float32(0.0)).all():
        raise ValueError("masked COCO17 joints must remain exact zero")
    tolerance = np.float32(1e-5)
    for frame_index in np.flatnonzero(sequence.valid_mask):
        selected = mask[frame_index]
        if not selected[11] or not selected[12] or int(selected.sum()) < 2:
            raise ValueError("unified-2D valid frame requires both COCO hip anchors")
        frame_xy = sequence.xyz[frame_index, :17, :2]
        hip_center = (frame_xy[11] + frame_xy[12]) * np.float32(0.5)
        if np.max(np.abs(hip_center)) > tolerance:
            raise ValueError("unified-2D hip midpoint is not body-centered")
        rms = np.sqrt(np.mean(np.sum(np.square(frame_xy[selected]), axis=1)))
        if not np.isfinite(rms) or abs(float(rms) - 1.0) > float(tolerance):
            raise ValueError("unified-2D reliable joints are not unit RMS scaled")
    return Unified2DSequence(pose=sequence, joint_valid_mask=mask)


def ranked_sequences(
    sequences: Sequence[Unified2DSequence],
    *,
    seed: int,
    purpose: str,
) -> tuple[Unified2DSequence, ...]:
    """Return a deterministic hash ordering independent of filesystem order."""

    prefix = f"pams-conventional-cycleback-v1\0{seed}\0{purpose}\0".encode()
    return tuple(
        sorted(
            sequences,
            key=lambda sequence: (
                hashlib.sha256(prefix + sequence.video_id.encode()).digest(),
                sequence.video_id,
            ),
        )
    )


def build_encoder(
    config: ConventionalCycleBackConfig,
    *,
    position_encoding_mode: str = "sinusoidal",
) -> PAMSEncoder:
    model = config.encoder
    if position_encoding_mode not in {"sinusoidal", "none"}:
        raise ValueError("position encoding mode must be sinusoidal or none")
    return PAMSEncoder(
        input_dim=model.input_dim,
        model_dim=model.model_dim,
        embedding_dim=model.embedding_dim,
        num_layers=model.layers,
        num_heads=model.heads,
        feedforward_dim=model.feedforward_dim,
        dropout=model.dropout,
        max_length=model.maximum_native_length,
        norm_first=model.norm_first,
        input_projection_scale=model.input_projection_scale,
        position_encoding_mode=position_encoding_mode,  # type: ignore[arg-type]
    )


def _view_seed(*, base_seed: int, step: int, view: str) -> int:
    if view not in {"a", "b"}:
        raise ValueError("view must be a or b")
    encoded = f"pams-conventional-cycleback-view-v1:{base_seed}:{step}:{view}".encode()
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)


def independent_view_seeds(*, base_seed: int, step: int) -> tuple[int, int]:
    """Return the exact deterministic augmentation seeds for one optimizer step."""

    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("base seed must be a non-negative integer")
    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise ValueError("augmentation step must be a non-negative integer")
    seeds = (
        _view_seed(base_seed=base_seed, step=step, view="a"),
        _view_seed(base_seed=base_seed, step=step, view="b"),
    )
    if seeds[0] == seeds[1]:
        raise RuntimeError("independent augmentation seeds collided")
    return seeds


def independent_augmented_views(
    batch: Unified2DBatch,
    config: ConventionalCycleBackConfig,
    *,
    step: int,
) -> tuple[Tensor, Tensor, tuple[int, int]]:
    """Create deterministic, non-shared augmentation streams for A and B."""

    if step < 0:
        raise ValueError("augmentation step must be non-negative")
    seeds = independent_view_seeds(base_seed=config.seed, step=step)
    generators: list[torch.Generator] = []
    for seed in seeds:
        generator = torch.Generator(device=batch.poses.device)
        generator.manual_seed(seed)
        generators.append(generator)
    augmentation = config.views.augmentation
    return (
        augment_unified_2d_pose_batch(
            batch.poses,
            batch.valid_mask,
            batch.joint_valid_mask,
            augmentation,
            generator=generators[0],
        ),
        augment_unified_2d_pose_batch(
            batch.poses,
            batch.valid_mask,
            batch.joint_valid_mask,
            augmentation,
            generator=generators[1],
        ),
        seeds,
    )


def independently_permuted_pair_poses(
    pairs: NativeWindowPairBatch,
    *,
    seed: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Break temporal order independently within disjoint A/B content sets."""

    outputs = [pairs.poses_a.detach().clone(), pairs.poses_b.detach().clone()]
    provenances = [
        pairs.source_indices_a.detach().clone(),
        pairs.source_indices_b.detach().clone(),
    ]
    side_seeds: list[int] = []
    moved_positions = 0
    eligible_positions = 0
    for side_index, (side, valid, starts) in enumerate(
        (
            ("a", pairs.valid_a, pairs.starts_a),
            ("b", pairs.valid_b, pairs.starts_b),
        )
    ):
        side_digest = hashlib.sha256(
            f"cycleback-independent-temporal-null-v1\0{seed}\0{side}".encode()
        ).digest()
        side_seed = int.from_bytes(side_digest[:8], "big") & ((1 << 63) - 1)
        side_seeds.append(side_seed)
        for row, identifier in enumerate(pairs.video_ids):
            positions = torch.nonzero(valid[row].detach().cpu(), as_tuple=False).flatten()
            if positions.numel() < 2:
                continue
            row_digest = hashlib.sha256(
                (
                    f"cycleback-independent-temporal-null-row-v1\0{side_seed}\0"
                    f"{identifier}\0{int(starts[row])}"
                ).encode()
            ).digest()
            shift = 1 + int.from_bytes(row_digest[:8], "big") % (positions.numel() - 1)
            permutation = torch.roll(torch.arange(positions.numel()), shifts=shift)
            if bool((permutation == torch.arange(positions.numel())).any()):
                raise RuntimeError("temporal null permutation is not a derangement")
            moved_positions += positions.numel()
            eligible_positions += positions.numel()
            destination = positions.to(device=outputs[side_index].device)
            source = positions[permutation].to(device=outputs[side_index].device)
            original_poses = outputs[side_index][row].clone()
            original_provenance = provenances[side_index][row].clone()
            outputs[side_index][row, destination] = original_poses[source]
            provenances[side_index][row, destination] = original_provenance[source]
    if side_seeds[0] == side_seeds[1]:
        raise RuntimeError("independent A/B temporal permutation seeds collided")
    intersection_total = sum(
        len(set(row_a[valid_a].tolist()).intersection(row_b[valid_b].tolist()))
        for row_a, row_b, valid_a, valid_b in zip(
            provenances[0].detach().cpu(),
            provenances[1].detach().cpu(),
            pairs.valid_a.detach().cpu(),
            pairs.valid_b.detach().cpu(),
            strict=True,
        )
    )
    if intersection_total != 0:
        raise RuntimeError("independent temporal null created a shared content mapping")
    moved_fraction = 1.0 if eligible_positions else 0.0
    if eligible_positions < 1 or moved_positions != eligible_positions:
        raise RuntimeError("temporal null did not move every eligible frame")
    return outputs[0], outputs[1], {
        "policy": "independent_within_disjoint_window_temporal_permutations",
        "side_seeds": side_seeds,
        "side_seeds_are_distinct": True,
        "source_content_intersection_total": intersection_total,
        "shared_content_mapping": False,
        "eligible_frame_positions": eligible_positions,
        "moved_frame_positions": moved_positions,
        "moved_frame_fraction": moved_fraction,
        "derangement_per_valid_window_side": True,
    }


def joint_support_null_poses(
    pairs: NativeWindowPairBatch,
) -> dict[str, tuple[Tensor, Tensor, dict[str, Any]]]:
    """Build three deterministic label-free joint-support stress controls."""

    if pairs.joint_valid_a.shape[2] != 17:
        raise ValueError("COCO17 joint-support nulls require exactly 17 channels")
    results: dict[str, tuple[Tensor, Tensor, dict[str, Any]]] = {}
    torso = torch.zeros(17, dtype=torch.bool, device=pairs.poses_a.device)
    torso[torch.tensor([5, 6, 11, 12], device=torso.device)] = True
    left = torch.zeros_like(torso)
    left[torch.tensor([5, 7, 9, 11, 13, 15], device=torso.device)] = True
    right = torch.zeros_like(torso)
    right[torch.tensor([6, 8, 10, 12, 14, 16], device=torso.device)] = True
    joint_index = torch.arange(17, device=pairs.poses_a.device).view(1, 1, 17)
    for name in ("mask_flicker", "torso_only", "alternating_limb_dropout"):
        outputs: list[Tensor] = []
        dropped = 0
        eligible = 0
        for poses, support, source_indices in (
            (pairs.poses_a, pairs.joint_valid_a, pairs.source_indices_a),
            (pairs.poses_b, pairs.joint_valid_b, pairs.source_indices_b),
        ):
            if name == "mask_flicker":
                keep = (source_indices.unsqueeze(-1) + joint_index).remainder(3) != 0
            elif name == "torso_only":
                keep = torso.view(1, 1, 17).expand_as(support)
            else:
                even = source_indices.remainder(2).eq(0).unsqueeze(-1)
                keep = torch.where(
                    even,
                    ~left.view(1, 1, 17),
                    ~right.view(1, 1, 17),
                )
            effective = support & keep
            eligible += int(support.sum())
            dropped += int((support & ~effective).sum())
            controlled = poses.detach().clone()
            controlled[:, :, :17, :] = controlled[:, :, :17, :].masked_fill(
                ~effective.unsqueeze(-1),
                0.0,
            )
            outputs.append(controlled)
        if eligible < 1 or dropped < 1:
            raise RuntimeError(f"{name} joint-support null did not drop any support")
        results[name] = (
            outputs[0],
            outputs[1],
            {
                "policy": name,
                "label_free": True,
                "source": "authorized_joint_valid_mask_and_opaque_source_index_only",
                "eligible_joint_positions": eligible,
                "dropped_joint_positions": dropped,
                "dropped_fraction": dropped / eligible,
                "frame_valid_mask_unchanged": True,
                "source_indices_unchanged": True,
            },
        )
    return results


def pairs_from_batch(
    batch: Unified2DBatch,
    config: ConventionalCycleBackConfig,
    *,
    step: int,
    segment_ranges_by_video: Mapping[str, Sequence[tuple[int, int]]],
    pair_eligibility: PairEligibility,
) -> tuple[NativeWindowPairBatch, AugmentedSequenceViews]:
    view_a, view_b, seeds = independent_augmented_views(batch, config, step=step)
    window = config.window_pair
    pairs = enumerate_native_window_pairs(
        view_a,
        view_b,
        batch.valid_mask,
        batch.joint_valid_mask,
        batch.lengths,
        batch.video_ids,
        tuple(segment_ranges_by_video[identifier] for identifier in batch.video_ids),
        tuple(
            pair_eligibility.base_valid_starts_by_variant[config.candidate_id][
                identifier
            ]
            for identifier in batch.video_ids
        ),
        tuple(
            pair_eligibility.starts_by_variant[config.candidate_id][identifier]
            for identifier in batch.video_ids
        ),
        window_length=window.length_frames,
        hop_frames=window.hop_frames,
        minimum_valid_frames_per_window=window.minimum_valid_frames_per_window,
        minimum_window_stable_action_joints=(
            pair_eligibility.minimum_window_stable_action_joints
        ),
        minimum_window_joint_support_fraction=(
            pair_eligibility.minimum_window_joint_support_fraction
        ),
    )
    return pairs, AugmentedSequenceViews(
        video_ids=batch.video_ids,
        poses_a=view_a,
        poses_b=view_b,
        valid_mask=batch.valid_mask,
        joint_valid_mask=batch.joint_valid_mask,
        lengths=batch.lengths,
        view_seeds=seeds,
    )


def validate_exact_authorized_pair_rows(
    pairs: NativeWindowPairBatch,
    *,
    ordered_video_ids: Sequence[str],
    native_lengths: Mapping[str, int],
    authorized_ranges_by_video: Mapping[str, Sequence[tuple[int, int]]],
    base_valid_starts_by_video: Mapping[str, Sequence[int]],
    eligible_starts_by_video: Mapping[str, Sequence[int]],
    window_frames: int,
    hop_frames: int,
) -> str:
    """Replay every emitted pair row against the sealed representation geometry.

    Counts and unordered membership are insufficient here: swapping two rows or
    changing a segment boundary leaves both unchanged while altering dropout and
    optimizer order.  This validator therefore binds the exact per-video order,
    starts, absolute source indices, native length, containing stable range, and
    all three per-video count layers.  The returned digest is safe to include in
    step evidence and checkpoint prefix state.
    """

    identifiers = tuple(ordered_video_ids)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("ordered pair replay video IDs must be non-empty unique")
    expected_membership = set(identifiers)
    for role, values in (
        ("native lengths", native_lengths),
        ("authorized ranges", authorized_ranges_by_video),
        ("base-valid starts", base_valid_starts_by_video),
        ("eligible starts", eligible_starts_by_video),
    ):
        if set(values) != expected_membership:
            raise ValueError(f"pair replay {role} membership mismatch")
    for role, value in (
        ("window frames", window_frames),
        ("hop frames", hop_frames),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"pair replay {role} must be a positive integer")
    if pairs.window_length != window_frames:
        raise RuntimeError("runtime pair window differs from authorized geometry")

    expected_rows: list[dict[str, Any]] = []
    expected_raw_counts: list[int] = []
    expected_base_counts: list[int] = []
    expected_eligible_counts: list[int] = []
    for source_index, video_id in enumerate(identifiers):
        native_length = native_lengths[video_id]
        if (
            isinstance(native_length, bool)
            or not isinstance(native_length, int)
            or native_length < 1
        ):
            raise ValueError("pair replay native length is invalid")
        ranges = tuple(authorized_ranges_by_video[video_id])
        base_starts = tuple(base_valid_starts_by_video[video_id])
        starts = tuple(eligible_starts_by_video[video_id])
        previous_stop = 0
        for range_index, value in enumerate(ranges):
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or any(
                    isinstance(item, bool) or not isinstance(item, int)
                    for item in value
                )
            ):
                raise ValueError("pair replay stable ranges are malformed")
            range_start, range_stop = value
            if (
                not 0 <= range_start < range_stop <= native_length
                or (range_index and range_start < previous_stop)
            ):
                raise ValueError("pair replay stable ranges are not canonical")
            previous_stop = range_stop
        if base_starts != tuple(sorted(set(base_starts))) or starts != tuple(
            sorted(set(starts))
        ):
            raise ValueError("pair replay starts must be sorted unique")
        if any(
            isinstance(start, bool) or not isinstance(start, int)
            for start in (*base_starts, *starts)
        ):
            raise ValueError("pair replay starts must be integer frame indices")
        if not set(starts).issubset(set(base_starts)):
            raise ValueError("pair replay eligible starts exceed base-valid starts")
        maximum_start = native_length - 2 * window_frames
        raw_count = 0 if maximum_start < 0 else maximum_start // hop_frames + 1
        expected_raw_counts.append(raw_count)
        expected_base_counts.append(len(base_starts))
        expected_eligible_counts.append(len(starts))
        for start in base_starts:
            containing = tuple(
                (range_start, range_stop)
                for range_start, range_stop in ranges
                if range_start <= start
                and start + 2 * window_frames <= range_stop
            )
            if (
                start < 0
                or start % hop_frames
                or len(containing) != 1
            ):
                raise ValueError(
                    "base-valid pair start violates the sealed stable-range grid"
                )
        for start in starts:
            containing = tuple(
                (range_start, range_stop)
                for range_start, range_stop in ranges
                if range_start <= start
                and start + 2 * window_frames <= range_stop
            )
            if len(containing) != 1:
                raise RuntimeError(
                    "authorized pair does not belong to exactly one stable range"
                )
            range_start, range_stop = containing[0]
            expected_rows.append(
                {
                    "video_id": video_id,
                    "source_video_index": source_index,
                    "native_length": native_length,
                    "start_a": start,
                    "start_b": start + window_frames,
                    "source_indices_a": list(range(start, start + window_frames)),
                    "source_indices_b": list(
                        range(start + window_frames, start + 2 * window_frames)
                    ),
                    "segment_start": range_start,
                    "segment_end": range_stop,
                }
            )

    if pairs.raw_grid_pair_counts != tuple(expected_raw_counts):
        raise RuntimeError("runtime raw-grid pair counts differ from authority")
    if pairs.available_pair_counts != tuple(expected_base_counts):
        raise RuntimeError("runtime base-valid pair counts differ from authority")
    if pairs.eligible_pair_counts != tuple(expected_eligible_counts):
        raise RuntimeError("runtime eligible pair counts differ from authority")
    if pairs.pair_count != len(expected_rows):
        raise RuntimeError("runtime pair row total differs from authority")

    observed_rows: list[dict[str, Any]] = []
    for row, video_id in enumerate(pairs.video_ids):
        observed_rows.append(
            {
                "video_id": video_id,
                "source_video_index": int(pairs.source_video_indices[row]),
                "native_length": int(pairs.native_lengths[row]),
                "start_a": int(pairs.starts_a[row]),
                "start_b": int(pairs.starts_b[row]),
                "source_indices_a": pairs.source_indices_a[row]
                .detach()
                .cpu()
                .tolist(),
                "source_indices_b": pairs.source_indices_b[row]
                .detach()
                .cpu()
                .tolist(),
                "segment_start": int(pairs.segment_starts[row]),
                "segment_end": int(pairs.segment_ends[row]),
            }
        )
    if observed_rows != expected_rows:
        raise RuntimeError(
            "runtime pair order/source indices/range lineage differ from authority"
        )
    combined_source_indices = torch.cat(
        (pairs.source_indices_a, pairs.source_indices_b),
        dim=1,
    )
    sorted_source_indices = combined_source_indices.sort(dim=1).values
    if pairs.pair_count and (
        not bool(pairs.valid_a.all())
        or not bool(pairs.valid_b.all())
        or bool(
            sorted_source_indices[:, 1:]
            .eq(sorted_source_indices[:, :-1])
            .any()
        )
    ):
        raise RuntimeError("runtime pair rows are invalid or share source indices")
    return canonical_json_sha256(
        {
            "schema_version": 1,
            "policy": "exact_authorized_pair_rows_in_optimizer_order",
            "window_frames": window_frames,
            "hop_frames": hop_frames,
            "rows": expected_rows,
            "raw_grid_pair_counts": expected_raw_counts,
            "base_valid_pair_counts": expected_base_counts,
            "eligible_pair_counts": expected_eligible_counts,
        }
    )


def cap_window_pairs(
    pairs: NativeWindowPairBatch,
    *,
    maximum_pairs: int,
    seed: int,
    purpose: str,
) -> NativeWindowPairBatch:
    """Select a stable bounded pair set without consulting any labels.

    The first pass retains the best-ranked pair from each represented video.
    Remaining capacity is filled by the same global hash order.  This avoids a
    long sequence monopolizing a bounded probe while retaining deterministic,
    target-free selection.
    """

    if maximum_pairs < 1:
        raise ValueError("maximum_pairs must be positive")
    if pairs.pair_count <= maximum_pairs:
        return pairs
    ranked = sorted(
        range(pairs.pair_count),
        key=lambda index: (
            hashlib.sha256(
                (
                    f"cycleback-pair-cap-v1\0{seed}\0{purpose}\0"
                    f"{pairs.video_ids[index]}\0{int(pairs.starts_a[index])}"
                ).encode()
            ).digest(),
            index,
        ),
    )
    first_per_video: list[int] = []
    seen: set[str] = set()
    for index in ranked:
        identifier = pairs.video_ids[index]
        if identifier not in seen:
            seen.add(identifier)
            first_per_video.append(index)
    selected = first_per_video[:maximum_pairs]
    selected_set = set(selected)
    if len(selected) < maximum_pairs:
        selected.extend(
            index
            for index in ranked
            if index not in selected_set
        )
        selected = selected[:maximum_pairs]
    indices = torch.tensor(
        selected,
        dtype=torch.long,
        device=pairs.poses_a.device,
    )
    return pairs.select(indices)


def permuted_valid_position_indices(
    source_indices: Tensor,
    valid_mask: Tensor,
    video_ids: Sequence[str],
    starts: Tensor,
    *,
    seed: int,
    side: str,
) -> Tensor:
    """Permute absolute PE rows only among valid positions in each window."""

    if source_indices.shape != valid_mask.shape or source_indices.dtype != torch.long:
        raise ValueError("source indices and valid mask must share [pairs, window]")
    if len(video_ids) != source_indices.shape[0] or starts.shape != (source_indices.shape[0],):
        raise ValueError("position-permutation identity inputs must match pair count")
    side_digest = hashlib.sha256(
        f"cycleback-independent-pe-null-side-v1\0{seed}\0{side}".encode()
    ).digest()
    side_seed = int.from_bytes(side_digest[:8], "big") & ((1 << 63) - 1)
    result = source_indices.detach().clone()
    for row, identifier in enumerate(video_ids):
        positions = torch.nonzero(valid_mask[row].detach().cpu(), as_tuple=False).flatten()
        if positions.numel() < 2:
            continue
        digest = hashlib.sha256(
            (
                f"cycleback-pe-permutation-v2\0{side_seed}\0{side}\0"
                f"{identifier}\0{int(starts[row])}"
            ).encode()
        ).digest()
        shift = 1 + int.from_bytes(digest[:8], "big") % (positions.numel() - 1)
        permutation = torch.roll(torch.arange(positions.numel()), shifts=shift)
        if bool((permutation == torch.arange(positions.numel())).any()):
            raise RuntimeError("PE null permutation is not a derangement")
        destination = positions.to(device=result.device)
        source = positions[permutation].to(device=result.device)
        result[row, destination] = source_indices[row, source]
    return result


def independently_permuted_pair_position_indices(
    pairs: NativeWindowPairBatch,
    *,
    seed: int,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """Derange valid absolute PE rows independently on both disjoint sides."""

    permuted_a = permuted_valid_position_indices(
        pairs.source_indices_a,
        pairs.valid_a,
        pairs.video_ids,
        pairs.starts_a,
        seed=seed,
        side="a",
    )
    permuted_b = permuted_valid_position_indices(
        pairs.source_indices_b,
        pairs.valid_b,
        pairs.video_ids,
        pairs.starts_b,
        seed=seed,
        side="b",
    )
    side_seeds = [
        int.from_bytes(
            hashlib.sha256(
                f"cycleback-independent-pe-null-side-v1\0{seed}\0{side}".encode()
            ).digest()[:8],
            "big",
        )
        & ((1 << 63) - 1)
        for side in ("a", "b")
    ]
    if side_seeds[0] == side_seeds[1]:
        raise RuntimeError("independent A/B PE-null seeds collided")
    eligible = int(pairs.valid_a.sum() + pairs.valid_b.sum())
    moved = int(
        ((permuted_a != pairs.source_indices_a) & pairs.valid_a).sum()
        + ((permuted_b != pairs.source_indices_b) & pairs.valid_b).sum()
    )
    if eligible < 1 or moved != eligible:
        raise RuntimeError("PE null did not move every valid position row")
    intersection = sum(
        len(set(row_a[valid_a].tolist()).intersection(row_b[valid_b].tolist()))
        for row_a, row_b, valid_a, valid_b in zip(
            permuted_a.detach().cpu(),
            permuted_b.detach().cpu(),
            pairs.valid_a.detach().cpu(),
            pairs.valid_b.detach().cpu(),
            strict=True,
        )
    )
    if intersection:
        raise RuntimeError("PE null created an A/B source-index intersection")
    return permuted_a, permuted_b, {
        "policy": "independent_valid_position_derangements_within_disjoint_sides",
        "side_seeds": side_seeds,
        "side_seeds_are_distinct": True,
        "derangement_per_valid_window_side": True,
        "eligible_position_rows": eligible,
        "moved_position_rows": moved,
        "moved_position_fraction": 1.0,
        "source_index_intersection_total": 0,
    }


def pair_segment_contexts(
    pairs: NativeWindowPairBatch,
    views: AugmentedSequenceViews,
) -> PairSegmentContexts:
    """Materialize the full all-valid stable range for every selected pair."""

    poses_a: list[Tensor] = []
    poses_b: list[Tensor] = []
    joints_a: list[Tensor] = []
    joints_b: list[Tensor] = []
    positions_a: list[Tensor] = []
    positions_b: list[Tensor] = []
    keys_a: list[str] = []
    keys_b: list[str] = []
    for row in range(pairs.pair_count):
        source = int(pairs.source_video_indices[row])
        if source < 0 or source >= len(views.video_ids):
            raise ValueError("pair source video lies outside augmented sequence views")
        if pairs.video_ids[row] != views.video_ids[source]:
            raise ValueError("pair video identity differs from augmented sequence view")
        start = int(pairs.segment_starts[row])
        stop = int(pairs.segment_ends[row])
        if not 0 <= start < stop <= int(views.lengths[source]):
            raise ValueError("pair stable-range context lies outside native timeline")
        context_valid = views.valid_mask[source, start:stop]
        if context_valid.numel() != stop - start or not bool(context_valid.all()):
            raise ValueError("authorized stable-range encoder context is not all-valid")
        poses_a.append(views.poses_a[source, start:stop])
        poses_b.append(views.poses_b[source, start:stop])
        joint_context = views.joint_valid_mask[source, start:stop]
        joints_a.append(joint_context)
        joints_b.append(joint_context)
        positions = torch.arange(
            start,
            stop,
            dtype=torch.long,
            device=views.poses_a.device,
        )
        positions_a.append(positions)
        positions_b.append(positions.clone())
        opaque = hashlib.sha256(pairs.video_ids[row].encode()).hexdigest()
        keys_a.append(f"real:a:{opaque}:{start}:{stop}")
        keys_b.append(f"real:b:{opaque}:{start}:{stop}")
    unique_context_total = len(set(keys_a)) + len(set(keys_b))
    return PairSegmentContexts(
        video_ids=pairs.video_ids,
        context_keys_a=tuple(keys_a),
        context_keys_b=tuple(keys_b),
        poses_a=tuple(poses_a),
        poses_b=tuple(poses_b),
        joint_valid_a=tuple(joints_a),
        joint_valid_b=tuple(joints_b),
        position_indices_a=tuple(positions_a),
        position_indices_b=tuple(positions_b),
        view_seeds=views.view_seeds,
        objective_role="real_optimizer",
        transform_contract={
            "policy": "full_authorized_all_valid_stable_range_then_slice_embeddings",
            "attention_context_boundary": "eligible_frame_range_start_stop",
            "association_reset_is_not_a_wider_attention_context": True,
            "absolute_native_position_indices": True,
            "window_only_encoder_path_used": False,
            "pair_side_context_references": 2 * pairs.pair_count,
            "unique_view_context_total": unique_context_total,
            "reused_pair_side_context_references": (
                2 * pairs.pair_count - unique_context_total
            ),
            "same_view_same_range_encoded_once": True,
        },
    )


def zero_pair_segment_contexts(
    contexts: PairSegmentContexts,
) -> PairSegmentContexts:
    return replace(
        contexts,
        objective_role="diagnostic_zero",
        context_keys_a=tuple(f"zero:{key}" for key in contexts.context_keys_a),
        context_keys_b=tuple(f"zero:{key}" for key in contexts.context_keys_b),
        poses_a=tuple(torch.zeros_like(value) for value in contexts.poses_a),
        poses_b=tuple(torch.zeros_like(value) for value in contexts.poses_b),
        transform_contract={
            "policy": "zero_pose_over_full_authorized_stable_range_context",
            "position_indices_unchanged": True,
            "joint_masks_unchanged": True,
        },
    )


def _context_partitions(
    pairs: NativeWindowPairBatch,
    row: int,
) -> tuple[tuple[int, int], ...]:
    segment_start = int(pairs.segment_starts[row])
    segment_stop = int(pairs.segment_ends[row])
    start_a = int(pairs.starts_a[row]) - segment_start
    start_b = int(pairs.starts_b[row]) - segment_start
    stop_b = start_b + pairs.window_length
    boundaries = (0, start_a, start_b, stop_b, segment_stop - segment_start)
    if not 0 <= start_a < start_b < stop_b <= boundaries[-1]:
        raise ValueError("pair windows do not partition their stable-range context")
    return tuple(
        (start, stop)
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True)
        if stop > start
    )


def _partition_derangement(
    length: int,
    *,
    seed: int,
    identity: str,
) -> Tensor:
    if length < 2:
        return torch.arange(length, dtype=torch.long)
    digest = hashlib.sha256(f"{seed}\0{identity}".encode()).digest()
    shift = 1 + int.from_bytes(digest[:8], "big") % (length - 1)
    return torch.roll(torch.arange(length, dtype=torch.long), shifts=shift)


def independently_permuted_pair_segment_contexts(
    contexts: PairSegmentContexts,
    pairs: NativeWindowPairBatch,
    *,
    seed: int,
) -> PairSegmentContexts:
    """Derange context partitions independently while keeping A/B content disjoint."""

    side_seeds = tuple(
        int.from_bytes(
            hashlib.sha256(
                f"cycleback-segment-temporal-null-v1\0{seed}\0{side}".encode()
            ).digest()[:8],
            "big",
        )
        & ((1 << 63) - 1)
        for side in ("a", "b")
    )
    if side_seeds[0] == side_seeds[1]:
        raise RuntimeError("segment temporal-null side seeds collided")
    output_poses = [[], []]
    output_joints = [[], []]
    selected_moved = 0
    context_moved = 0
    context_positions = 0
    intersection_total = 0
    for row in range(pairs.pair_count):
        partitions = _context_partitions(pairs, row)
        provenances: list[Tensor] = []
        for side_index, (poses, joints) in enumerate(
            (
                (contexts.poses_a[row], contexts.joint_valid_a[row]),
                (contexts.poses_b[row], contexts.joint_valid_b[row]),
            )
        ):
            controlled_poses = poses.clone()
            controlled_joints = joints.clone()
            provenance = torch.arange(
                int(pairs.segment_starts[row]),
                int(pairs.segment_ends[row]),
                dtype=torch.long,
                device=poses.device,
            )
            for part_index, (start, stop) in enumerate(partitions):
                permutation = _partition_derangement(
                    stop - start,
                    seed=side_seeds[side_index],
                    identity=(
                        f"{pairs.video_ids[row]}:{int(pairs.starts_a[row])}:"
                        f"{side_index}:{part_index}"
                    ),
                ).to(device=poses.device)
                destination = torch.arange(start, stop, device=poses.device)
                source = start + permutation
                original_poses = controlled_poses.clone()
                original_joints = controlled_joints.clone()
                original_provenance = provenance.clone()
                controlled_poses[destination] = original_poses[source]
                controlled_joints[destination] = original_joints[source]
                provenance[destination] = original_provenance[source]
                context_moved += int((source != destination).sum())
                context_positions += stop - start
            output_poses[side_index].append(controlled_poses)
            output_joints[side_index].append(controlled_joints)
            provenances.append(provenance)
        local_a = pairs.source_indices_a[row] - pairs.segment_starts[row]
        local_b = pairs.source_indices_b[row] - pairs.segment_starts[row]
        selected_a = provenances[0][local_a]
        selected_b = provenances[1][local_b]
        selected_moved += int(
            (selected_a != pairs.source_indices_a[row]).sum()
            + (selected_b != pairs.source_indices_b[row]).sum()
        )
        intersection_total += len(
            set(selected_a.detach().cpu().tolist()).intersection(
                selected_b.detach().cpu().tolist()
            )
        )
    selected_total = 2 * pairs.pair_count * pairs.window_length
    if selected_total < 1 or selected_moved != selected_total:
        raise RuntimeError("segment temporal null did not move every selected frame")
    if intersection_total:
        raise RuntimeError("segment temporal null created shared A/B window content")
    return replace(
        contexts,
        objective_role="diagnostic_temporal_permutation",
        context_keys_a=tuple(
            f"temporal-pair:{row}:a:{key}"
            for row, key in enumerate(contexts.context_keys_a)
        ),
        context_keys_b=tuple(
            f"temporal-pair:{row}:b:{key}"
            for row, key in enumerate(contexts.context_keys_b)
        ),
        poses_a=tuple(output_poses[0]),
        poses_b=tuple(output_poses[1]),
        joint_valid_a=tuple(output_joints[0]),
        joint_valid_b=tuple(output_joints[1]),
        transform_contract={
            "policy": "independent_partition_derangement_in_full_stable_range_context",
            "side_seeds": list(side_seeds),
            "side_seeds_are_distinct": True,
            "selected_window_positions": selected_total,
            "selected_window_moved_positions": selected_moved,
            "selected_window_moved_fraction": 1.0,
            "context_positions": context_positions,
            "context_moved_positions": context_moved,
            "context_moved_fraction": (
                0.0 if not context_positions else context_moved / context_positions
            ),
            "selected_source_content_intersection_total": intersection_total,
            "shared_selected_content_mapping": False,
            "diagnostic_pair_specific_contexts": True,
            "authorized_for_real_optimizer": False,
        },
    )


def independently_permuted_pair_segment_positions(
    contexts: PairSegmentContexts,
    pairs: NativeWindowPairBatch,
    *,
    seed: int,
) -> PairSegmentContexts:
    """Derange absolute PE rows independently within stable-range partitions."""

    side_seeds = tuple(
        int.from_bytes(
            hashlib.sha256(
                f"cycleback-segment-pe-null-v1\0{seed}\0{side}".encode()
            ).digest()[:8],
            "big",
        )
        & ((1 << 63) - 1)
        for side in ("a", "b")
    )
    if side_seeds[0] == side_seeds[1]:
        raise RuntimeError("segment PE-null side seeds collided")
    outputs = [[], []]
    selected_moved = 0
    intersection_total = 0
    for row in range(pairs.pair_count):
        partitions = _context_partitions(pairs, row)
        for side_index, canonical in enumerate(
            (contexts.position_indices_a[row], contexts.position_indices_b[row])
        ):
            controlled = canonical.clone()
            for part_index, (start, stop) in enumerate(partitions):
                permutation = _partition_derangement(
                    stop - start,
                    seed=side_seeds[side_index],
                    identity=(
                        f"{pairs.video_ids[row]}:{int(pairs.starts_a[row])}:"
                        f"{side_index}:{part_index}"
                    ),
                ).to(device=canonical.device)
                destination = torch.arange(start, stop, device=canonical.device)
                original = controlled.clone()
                controlled[destination] = original[start + permutation]
            outputs[side_index].append(controlled)
        local_a = pairs.source_indices_a[row] - pairs.segment_starts[row]
        local_b = pairs.source_indices_b[row] - pairs.segment_starts[row]
        selected_a = outputs[0][-1][local_a]
        selected_b = outputs[1][-1][local_b]
        selected_moved += int(
            (selected_a != pairs.source_indices_a[row]).sum()
            + (selected_b != pairs.source_indices_b[row]).sum()
        )
        intersection_total += len(
            set(selected_a.detach().cpu().tolist()).intersection(
                selected_b.detach().cpu().tolist()
            )
        )
    selected_total = 2 * pairs.pair_count * pairs.window_length
    if selected_total < 1 or selected_moved != selected_total:
        raise RuntimeError("segment PE null did not move every selected position")
    if intersection_total:
        raise RuntimeError("segment PE null created an A/B position intersection")
    return replace(
        contexts,
        objective_role="diagnostic_pe_permutation",
        context_keys_a=tuple(
            f"pe-pair:{row}:a:{key}"
            for row, key in enumerate(contexts.context_keys_a)
        ),
        context_keys_b=tuple(
            f"pe-pair:{row}:b:{key}"
            for row, key in enumerate(contexts.context_keys_b)
        ),
        position_indices_a=tuple(outputs[0]),
        position_indices_b=tuple(outputs[1]),
        transform_contract={
            "policy": "independent_partition_derangement_of_absolute_segment_pe",
            "side_seeds": list(side_seeds),
            "side_seeds_are_distinct": True,
            "selected_position_rows": selected_total,
            "selected_moved_position_rows": selected_moved,
            "selected_moved_position_fraction": 1.0,
            "selected_position_intersection_total": intersection_total,
            "diagnostic_pair_specific_contexts": True,
            "authorized_for_real_optimizer": False,
        },
    )


def joint_support_null_segment_contexts(
    contexts: PairSegmentContexts,
) -> dict[str, PairSegmentContexts]:
    """Apply deterministic joint-support nulls over full stable-range contexts."""

    if any(
        mask.shape[1] != 17
        for mask in (*contexts.joint_valid_a, *contexts.joint_valid_b)
    ):
        raise ValueError("COCO17 joint-support nulls require exactly 17 channels")
    device = contexts.poses_a[0].device if contexts.poses_a else torch.device("cpu")
    torso = torch.zeros(17, dtype=torch.bool, device=device)
    torso[torch.tensor([5, 6, 11, 12], device=device)] = True
    left = torch.zeros_like(torso)
    left[torch.tensor([5, 7, 9, 11, 13, 15], device=device)] = True
    right = torch.zeros_like(torso)
    right[torch.tensor([6, 8, 10, 12, 14, 16], device=device)] = True
    results: dict[str, PairSegmentContexts] = {}
    for name in ("mask_flicker", "torso_only", "alternating_limb_dropout"):
        output_poses = [[], []]
        output_joints = [[], []]
        eligible = 0
        dropped = 0
        for side_index, (poses_collection, joints_collection, positions_collection) in enumerate(
            (
                (contexts.poses_a, contexts.joint_valid_a, contexts.position_indices_a),
                (contexts.poses_b, contexts.joint_valid_b, contexts.position_indices_b),
            )
        ):
            for poses, support, positions in zip(
                poses_collection,
                joints_collection,
                positions_collection,
                strict=True,
            ):
                joint_index = torch.arange(17, device=poses.device).view(1, 17)
                if name == "mask_flicker":
                    keep = (positions.unsqueeze(-1) + joint_index).remainder(3) != 0
                elif name == "torso_only":
                    keep = torso.view(1, 17).expand_as(support)
                else:
                    even = positions.remainder(2).eq(0).unsqueeze(-1)
                    keep = torch.where(
                        even,
                        ~left.view(1, 17),
                        ~right.view(1, 17),
                    )
                effective = support & keep
                controlled = poses.clone()
                controlled[:, :17, :] = controlled[:, :17, :].masked_fill(
                    ~effective.unsqueeze(-1),
                    0.0,
                )
                eligible += int(support.sum())
                dropped += int((support & ~effective).sum())
                output_poses[side_index].append(controlled)
                output_joints[side_index].append(effective)
        if eligible < 1 or dropped < 1:
            raise RuntimeError(f"{name} segment-context null dropped no support")
        results[name] = replace(
            contexts,
            objective_role=f"diagnostic_joint_{name}",
            context_keys_a=tuple(
                f"joint-{name}:{key}" for key in contexts.context_keys_a
            ),
            context_keys_b=tuple(
                f"joint-{name}:{key}" for key in contexts.context_keys_b
            ),
            poses_a=tuple(output_poses[0]),
            poses_b=tuple(output_poses[1]),
            joint_valid_a=tuple(output_joints[0]),
            joint_valid_b=tuple(output_joints[1]),
            transform_contract={
                "policy": name,
                "scope": "full_authorized_stable_range_context",
                "label_free": True,
                "eligible_joint_positions": eligible,
                "dropped_joint_positions": dropped,
                "dropped_fraction": dropped / eligible,
                "absolute_position_indices_unchanged": True,
            },
        )
    return results


def _unique_segment_context_indices(
    keys: tuple[str, ...],
    poses: tuple[Tensor, ...],
    positions: tuple[Tensor, ...],
    joints: tuple[Tensor, ...],
) -> dict[str, int]:
    observed: dict[str, int] = {}
    for index, key in enumerate(keys):
        previous = observed.get(key)
        if previous is None:
            observed[key] = index
            continue
        if (
            not torch.equal(poses[previous], poses[index])
            or not torch.equal(positions[previous], positions[index])
            or not torch.equal(joints[previous], joints[index])
        ):
            raise ValueError(
                "duplicate segment-context key has different pose/PE/joint bytes"
            )
    return observed


def _segment_encoding_batches(
    keys: tuple[str, ...],
    poses: tuple[Tensor, ...],
    positions: tuple[Tensor, ...],
    joints: tuple[Tensor, ...],
    *,
    batch_size: int,
) -> tuple[tuple[str, ...], ...]:
    if batch_size < 1:
        raise ValueError("encoder segment-context batch size must be positive")
    observed = _unique_segment_context_indices(keys, poses, positions, joints)
    ordered = sorted(
        observed,
        key=lambda key: (poses[observed[key]].shape[0], key),
    )
    batches: list[tuple[str, ...]] = []
    start = 0
    while start < len(ordered):
        length = poses[observed[ordered[start]]].shape[0]
        stop = start
        while (
            stop < len(ordered)
            and poses[observed[ordered[stop]]].shape[0] == length
        ):
            stop += 1
        for chunk_start in range(start, stop, batch_size):
            batches.append(tuple(ordered[chunk_start : min(stop, chunk_start + batch_size)]))
        start = stop
    return tuple(batches)


def context_encoding_plan(
    contexts: PairSegmentContexts,
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Return the deterministic exact-length full-context encoder plan."""

    side_batches: dict[str, tuple[tuple[str, ...], ...]] = {}
    lengths: dict[str, int] = {}
    for side, keys, poses, positions, joints in (
        (
            "a",
            contexts.context_keys_a,
            contexts.poses_a,
            contexts.position_indices_a,
            contexts.joint_valid_a,
        ),
        (
            "b",
            contexts.context_keys_b,
            contexts.poses_b,
            contexts.position_indices_b,
            contexts.joint_valid_b,
        ),
    ):
        batches = _segment_encoding_batches(
            keys,
            poses,
            positions,
            joints,
            batch_size=batch_size,
        )
        side_batches[side] = batches
        observed = _unique_segment_context_indices(keys, poses, positions, joints)
        lengths.update({key: poses[index].shape[0] for key, index in observed.items()})
    payload = {
        "schema_version": 1,
        "policy": (
            "side_a_then_side_b_exact_context_length_bucket_key_order_"
            "fixed_batch_size"
        ),
        "batch_size": batch_size,
        "side_batches": {
            side: [list(batch) for batch in side_batches[side]]
            for side in ("a", "b")
        },
        "context_lengths": {key: lengths[key] for key in sorted(lengths)},
        "mixed_context_lengths_within_batch": False,
        "window_only_encoder_path_used": False,
    }
    payload["fingerprint"] = canonical_json_sha256(payload)
    return payload


def _encode_segment_collection(
    encoder: PAMSEncoder,
    keys: tuple[str, ...],
    poses: tuple[Tensor, ...],
    positions: tuple[Tensor, ...],
    joints: tuple[Tensor, ...],
    *,
    batches: Sequence[Sequence[str]],
) -> tuple[Tensor, ...]:
    observed = _unique_segment_context_indices(keys, poses, positions, joints)
    outputs: list[Tensor] = []
    output_keys: list[str] = []
    for batch in batches:
        indices = [observed[key] for key in batch]
        pose_chunk = tuple(poses[index] for index in indices)
        position_chunk = tuple(positions[index] for index in indices)
        lengths = {value.shape[0] for value in pose_chunk}
        if len(lengths) != 1:
            raise RuntimeError("encoder plan mixed different stable-range lengths")
        length = lengths.pop()
        stacked = torch.stack(pose_chunk)
        valid = torch.ones(
            (len(pose_chunk), length),
            dtype=torch.bool,
            device=stacked.device,
        )
        stacked_positions = torch.stack(position_chunk)
        encoded = encoder(stacked, valid, position_indices=stacked_positions)
        outputs.extend(encoded[row] for row in range(len(pose_chunk)))
        output_keys.extend(batch)
    encoded_by_key = dict(zip(output_keys, outputs, strict=True))
    if set(encoded_by_key) != set(observed):
        raise RuntimeError("encoder plan did not cover every unique stable-range context")
    return tuple(encoded_by_key[key] for key in keys)


def _validate_context_key_joint_identity(
    keys: tuple[str, ...],
    joints: tuple[Tensor, ...],
) -> None:
    observed: dict[str, Tensor] = {}
    for key, mask in zip(keys, joints, strict=True):
        previous = observed.get(key)
        if previous is not None and not torch.equal(previous, mask):
            raise ValueError("duplicate segment-context key has different joint-mask bytes")
        observed[key] = mask


def require_real_optimizer_contexts(contexts: PairSegmentContexts) -> None:
    """Fail closed before backward if a diagnostic context reaches optimization."""

    if contexts.objective_role != "real_optimizer":
        raise ValueError(
            "diagnostic pair-segment contexts cannot enter the real optimizer"
        )
    if (
        contexts.transform_contract.get("same_view_same_range_encoded_once") is not True
        or contexts.transform_contract.get("window_only_encoder_path_used") is not False
    ):
        raise ValueError("real optimizer context contract is incomplete")


def encode_window_pairs(
    encoder: PAMSEncoder,
    pairs: NativeWindowPairBatch,
    contexts: PairSegmentContexts,
    *,
    batch_size: int,
) -> tuple[Tensor, Tensor]:
    """Encode full stable ranges, then slice the two disjoint embedding windows."""

    if batch_size < 1:
        raise ValueError("encoder segment-context batch size must be positive")
    if contexts.video_ids != pairs.video_ids:
        raise ValueError("pair-segment contexts differ from selected pair identities")
    if pairs.pair_count == 0:
        empty_shape = (0, pairs.window_length, encoder.embedding_dim)
        return (
            pairs.poses_a.new_zeros(empty_shape),
            pairs.poses_b.new_zeros(empty_shape),
        )
    plan = context_encoding_plan(contexts, batch_size=batch_size)
    side_batches = plan["side_batches"]
    encoded_a = _encode_segment_collection(
        encoder,
        contexts.context_keys_a,
        contexts.poses_a,
        contexts.position_indices_a,
        contexts.joint_valid_a,
        batches=side_batches["a"],
    )
    encoded_b = _encode_segment_collection(
        encoder,
        contexts.context_keys_b,
        contexts.poses_b,
        contexts.position_indices_b,
        contexts.joint_valid_b,
        batches=side_batches["b"],
    )
    selected_a: list[Tensor] = []
    selected_b: list[Tensor] = []
    for row in range(pairs.pair_count):
        local_a = pairs.source_indices_a[row] - pairs.segment_starts[row]
        local_b = pairs.source_indices_b[row] - pairs.segment_starts[row]
        if (
            int(local_a.min()) < 0
            or int(local_b.min()) < 0
            or int(local_a.max()) >= encoded_a[row].shape[0]
            or int(local_b.max()) >= encoded_b[row].shape[0]
        ):
            raise ValueError("pair source indices lie outside encoded stable range")
        selected_a.append(encoded_a[row][local_a])
        selected_b.append(encoded_b[row][local_b])
    return torch.stack(selected_a), torch.stack(selected_b)


def temporal_rms_values(embeddings: Tensor, valid_mask: Tensor) -> list[float]:
    """Per-window centered embedding RMS over valid rows."""

    if embeddings.ndim != 3 or valid_mask.shape != embeddings.shape[:2]:
        raise ValueError("temporal RMS expects embeddings and matching mask")
    values: list[float] = []
    for stream, valid in zip(embeddings, valid_mask, strict=True):
        selected = stream[valid]
        if selected.shape[0] < 2:
            continue
        centered = selected - selected.mean(dim=0, keepdim=True)
        values.append(float(centered.square().sum(dim=1).mean().sqrt().detach()))
    return values


def numeric_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "observations": 0,
            "minimum": None,
            "p10": None,
            "median": None,
            "mean": None,
            "p90": None,
            "maximum": None,
        }
    if not np.isfinite(array).all():
        raise RuntimeError("summary received a non-finite value")
    return {
        "observations": int(array.size),
        "minimum": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def collate_to_device(
    sequences: Sequence[Unified2DSequence],
    device: torch.device,
) -> Unified2DBatch:
    if not sequences:
        raise ValueError("cannot collate an empty unified-2D sequence batch")
    pose_batch = collate_pose_sequences([item.pose for item in sequences]).to(device)
    maximum_length = int(pose_batch.poses.shape[1])
    joint_mask = torch.zeros(
        (len(sequences), maximum_length, 17),
        dtype=torch.bool,
        device=device,
    )
    for index, sequence in enumerate(sequences):
        joint_mask[index, : sequence.num_frames] = torch.from_numpy(
            np.asarray(sequence.joint_valid_mask)
        ).to(device=device)
    return Unified2DBatch(
        pose_batch=pose_batch,
        joint_valid_mask=joint_mask,
    )


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
