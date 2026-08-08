"""Fail-closed science core for full-context cycle-back continuation.

This module defines the deterministic training, sampler, checkpoint, and
label-free gate contracts needed after the bounded 256-step mechanism probe.
It deliberately does not expose a command-line entry point.  The current
mechanism probe records only a model-state digest and therefore cannot satisfy
the exact learned-state predecessor required here.  A future independently
sealed integration outcome must provide the checkpoint bytes before any formal
epoch-11 or epoch-150 continuation can be launched.

The implementation remains an independently inferred conventional cycle-back
proxy.  PAMS names a conventional TCC baseline, but does not disclose this
objective, epoch sampler, optimizer schedule, or checkpoint protocol.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import random
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

from pams.conventional_cycleback.audit import evaluate_cycleback_embeddings
from pams.conventional_cycleback.config import ConventionalCycleBackConfig
from pams.conventional_cycleback.loss import ConventionalCycleBackLoss
from pams.conventional_cycleback.runtime import (
    PairEligibility,
    PairSegmentContexts,
    Unified2DSequence,
    build_encoder,
    collate_to_device,
    encode_window_pairs,
    independent_view_seeds,
    independently_permuted_pair_segment_contexts,
    independently_permuted_pair_segment_positions,
    joint_support_null_segment_contexts,
    pair_segment_contexts,
    pairs_from_batch,
    require_real_optimizer_contexts,
    stable_file_bytes,
    temporal_rms_values,
    zero_pair_segment_contexts,
)
from pams.conventional_cycleback.windows import NativeWindowPairBatch
from pams.model import PAMSEncoder

CandidateId = Literal["W16_H4", "W16_H2", "W24_H4"]
TrainingPhase = Literal["epoch11", "epoch150"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_CANDIDATE_GEOMETRY: dict[str, tuple[int, int]] = {
    "W16_H4": (16, 4),
    "W16_H2": (16, 2),
    "W24_H4": (24, 4),
}
_MECHANISM_OPTIMIZER_STEPS = 256
_EPOCH11_TARGET = 11
_EPOCH150_TARGET = 150


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_integer(value: Any, *, role: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{role} must be an integer >= {minimum}")
    return value


def _require_sha256(value: str, *, role: str) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{role} must be a lowercase SHA-256")


def _require_git_sha(value: str, *, role: str) -> None:
    if not isinstance(value, str) or not _GIT_SHA.fullmatch(value):
        raise ValueError(f"{role} must be a full lowercase Git SHA")


def _require_image_id(value: str, *, role: str) -> None:
    if not isinstance(value, str) or not _IMAGE_ID.fullmatch(value):
        raise ValueError(f"{role} must be an immutable sha256 image ID")


@dataclass(frozen=True, slots=True)
class FullContextRepresentationContract:
    """Typed adapter boundary; the trainer does not assume COCO17 semantics."""

    representation_family: str
    coordinate_contract: str
    pose_joint_count: Literal[33]
    support_channel_count: int
    support_channel_to_pose_joint_indices: tuple[int, ...]
    augmentation_adapter: str
    diagnostic_adapter: str
    stable_range_policy: str
    pair_authority_policy: Literal[
        "sealed_representation_exact_starts_no_consumer_expansion"
    ] = "sealed_representation_exact_starts_no_consumer_expansion"
    label_free: Literal[True] = True

    def __post_init__(self) -> None:
        for role, value in (
            ("representation family", self.representation_family),
            ("coordinate contract", self.coordinate_contract),
            ("augmentation adapter", self.augmentation_adapter),
            ("diagnostic adapter", self.diagnostic_adapter),
            ("stable range policy", self.stable_range_policy),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{role} must be a non-empty frozen identifier")
        if self.pose_joint_count != 33:
            raise ValueError("PAMS encoder input remains exactly 33x3")
        _require_integer(
            self.support_channel_count,
            role="joint-support channel count",
            minimum=1,
        )
        mapping = self.support_channel_to_pose_joint_indices
        if (
            len(mapping) != self.support_channel_count
            or len(set(mapping)) != len(mapping)
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < self.pose_joint_count
                for index in mapping
            )
        ):
            raise ValueError("support-channel to pose-joint mapping is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_type": "pams_cycleback_fullcontext_representation_contract_v1",
            "representation_family": self.representation_family,
            "coordinate_contract": self.coordinate_contract,
            "pose_joint_count": self.pose_joint_count,
            "support_channel_count": self.support_channel_count,
            "support_channel_to_pose_joint_indices": list(
                self.support_channel_to_pose_joint_indices
            ),
            "augmentation_adapter": self.augmentation_adapter,
            "diagnostic_adapter": self.diagnostic_adapter,
            "stable_range_policy": self.stable_range_policy,
            "pair_authority_policy": self.pair_authority_policy,
            "label_free": self.label_free,
            "action_or_repetition_labels_consumed": False,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class FullContextTrainerContract:
    """Frozen inferred training choices shared by all three candidates."""

    candidate_id: CandidateId
    candidate_config_fingerprint: str
    window_frames: int
    hop_frames: int
    seed: int
    maximum_native_length: int = 4096
    training_video_total: int = 337
    optimizer_name: Literal["AdamW"] = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    scheduler: Literal["none"] = "none"
    mechanism_optimizer_transition: Literal[
        "continue_exact_step256_adamw_state_no_reset"
    ] = "continue_exact_step256_adamw_state_no_reset"
    maximum_videos_per_batch: int = 8
    maximum_context_frames_per_batch: int = 2048
    encoder_context_batch_size: int = 8
    mechanism_optimizer_steps: int = _MECHANISM_OPTIMIZER_STEPS
    epoch11_target: int = _EPOCH11_TARGET
    epoch150_target: int = _EPOCH150_TARGET
    classification: Literal[
        "independently_inferred_proxy_not_author_implementation"
    ] = "independently_inferred_proxy_not_author_implementation"
    paper_table_claim_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        expected = _CANDIDATE_GEOMETRY.get(self.candidate_id)
        if expected is None or (self.window_frames, self.hop_frames) != expected:
            raise ValueError("full-context candidate geometry mismatch")
        _require_sha256(
            self.candidate_config_fingerprint,
            role="candidate config fingerprint",
        )
        _require_integer(self.seed, role="trainer seed")
        _require_integer(
            self.maximum_native_length,
            role="maximum native length",
            minimum=2 * self.window_frames,
        )
        if self.training_video_total != 337:
            raise ValueError("full-context continuation is train337-only")
        if self.optimizer_name != "AdamW":
            raise ValueError("full-context optimizer must be AdamW")
        if self.learning_rate != 1e-4 or self.weight_decay != 1e-4:
            raise ValueError("full-context AdamW hyperparameters are frozen")
        if self.scheduler != "none":
            raise ValueError("undisclosed scheduler must remain explicitly disabled")
        if (
            self.mechanism_optimizer_transition
            != "continue_exact_step256_adamw_state_no_reset"
        ):
            raise ValueError("mechanism optimizer transition must preserve exact state")
        for role, value in (
            ("maximum videos per batch", self.maximum_videos_per_batch),
            (
                "maximum context frames per batch",
                self.maximum_context_frames_per_batch,
            ),
            ("encoder context batch size", self.encoder_context_batch_size),
        ):
            _require_integer(value, role=role, minimum=1)
        if (
            self.mechanism_optimizer_steps != _MECHANISM_OPTIMIZER_STEPS
            or self.epoch11_target != _EPOCH11_TARGET
            or self.epoch150_target != _EPOCH150_TARGET
        ):
            raise ValueError("full-context stage boundaries are frozen")

    @classmethod
    def from_candidate_config(
        cls,
        config: ConventionalCycleBackConfig,
    ) -> FullContextTrainerContract:
        if (
            config.mechanism_probe.learning_rate != 1e-4
            or config.mechanism_probe.weight_decay != 1e-4
        ):
            raise ValueError("mechanism optimizer differs from continuation AdamW")
        return cls(
            candidate_id=config.candidate_id,
            candidate_config_fingerprint=config.fingerprint,
            window_frames=config.window_pair.length_frames,
            hop_frames=config.window_pair.hop_frames,
            seed=config.seed,
            maximum_native_length=config.encoder.maximum_native_length,
            encoder_context_batch_size=(
                config.mechanism_probe.encoder_segment_context_batch_size
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_type": "pams_cycleback_fullcontext_trainer_contract_v1",
            "candidate_id": self.candidate_id,
            "candidate_config_fingerprint": self.candidate_config_fingerprint,
            "window_frames": self.window_frames,
            "hop_frames": self.hop_frames,
            "seed": self.seed,
            "maximum_native_length": self.maximum_native_length,
            "training_video_total": self.training_video_total,
            "optimizer": {
                "name": self.optimizer_name,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "amsgrad": False,
                "maximize": False,
            },
            "scheduler": self.scheduler,
            "mechanism_optimizer_transition": self.mechanism_optimizer_transition,
            "maximum_videos_per_batch": self.maximum_videos_per_batch,
            "maximum_context_frames_per_batch": (
                self.maximum_context_frames_per_batch
            ),
            "encoder_context_batch_size": self.encoder_context_batch_size,
            "mechanism_optimizer_steps": self.mechanism_optimizer_steps,
            "epoch11_target": self.epoch11_target,
            "epoch150_target": self.epoch150_target,
            "classification": self.classification,
            "paper_table_claim_eligible": self.paper_table_claim_eligible,
            "context_policy": (
                "encode_each_unique_video_stable_range_augmentation_view_once_"
                "then_gather_disjoint_windows_by_absolute_source_index"
            ),
            "training_inference_context_parity": True,
            "window_only_encoder_path_used": False,
            "null_contexts_authorized_for_optimizer": False,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class FullContextLineage:
    """Exact immutable predecessor identities required by every checkpoint."""

    source_git_sha: str
    source_tree_sha256: str
    container_image_id: str
    config_sha256: str
    config_bytes: int
    representation_integration_outcome_sha256: str
    representation_integration_outcome_bytes: int
    representation_contract_sha256: str
    representation_authorization_sha256: str
    representation_authorization_bytes: int
    mechanism_outcome_sha256: str
    mechanism_outcome_bytes: int
    mechanism_run_receipt_sha256: str
    mechanism_run_receipt_bytes: int
    mechanism_seed_checkpoint_sha256: str
    mechanism_seed_checkpoint_bytes: int
    mechanism_learned_model_state_sha256: str

    def __post_init__(self) -> None:
        _require_git_sha(self.source_git_sha, role="source revision")
        _require_image_id(self.container_image_id, role="container image")
        for role, value in (
            ("source tree", self.source_tree_sha256),
            ("config", self.config_sha256),
            (
                "representation integration outcome",
                self.representation_integration_outcome_sha256,
            ),
            ("representation contract", self.representation_contract_sha256),
            (
                "representation authorization",
                self.representation_authorization_sha256,
            ),
            ("mechanism outcome", self.mechanism_outcome_sha256),
            ("mechanism run receipt", self.mechanism_run_receipt_sha256),
            ("mechanism seed checkpoint", self.mechanism_seed_checkpoint_sha256),
            ("mechanism learned model state", self.mechanism_learned_model_state_sha256),
        ):
            _require_sha256(value, role=role)
        for role, value in (
            ("config bytes", self.config_bytes),
            (
                "representation integration outcome bytes",
                self.representation_integration_outcome_bytes,
            ),
            (
                "representation authorization bytes",
                self.representation_authorization_bytes,
            ),
            ("mechanism outcome bytes", self.mechanism_outcome_bytes),
            ("mechanism run receipt bytes", self.mechanism_run_receipt_bytes),
            (
                "mechanism seed checkpoint bytes",
                self.mechanism_seed_checkpoint_bytes,
            ),
        ):
            _require_integer(value, role=role, minimum=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_git_sha": self.source_git_sha,
            "source_tree_sha256": self.source_tree_sha256,
            "container_image_id": self.container_image_id,
            "config_sha256": self.config_sha256,
            "config_bytes": self.config_bytes,
            "representation_integration_outcome_sha256": (
                self.representation_integration_outcome_sha256
            ),
            "representation_integration_outcome_bytes": (
                self.representation_integration_outcome_bytes
            ),
            "representation_contract_sha256": self.representation_contract_sha256,
            "representation_authorization_sha256": (
                self.representation_authorization_sha256
            ),
            "representation_authorization_bytes": (
                self.representation_authorization_bytes
            ),
            "mechanism_outcome_sha256": self.mechanism_outcome_sha256,
            "mechanism_outcome_bytes": self.mechanism_outcome_bytes,
            "mechanism_run_receipt_sha256": self.mechanism_run_receipt_sha256,
            "mechanism_run_receipt_bytes": self.mechanism_run_receipt_bytes,
            "mechanism_seed_checkpoint_sha256": (
                self.mechanism_seed_checkpoint_sha256
            ),
            "mechanism_seed_checkpoint_bytes": self.mechanism_seed_checkpoint_bytes,
            "mechanism_learned_model_state_sha256": (
                self.mechanism_learned_model_state_sha256
            ),
        }

    def mechanism_seed_predecessor_dict(self) -> dict[str, Any]:
        """Return the acyclic lineage embedded inside the seed checkpoint."""

        payload = self.to_dict()
        for key in (
            "mechanism_outcome_sha256",
            "mechanism_outcome_bytes",
            "mechanism_run_receipt_sha256",
            "mechanism_run_receipt_bytes",
            "mechanism_seed_checkpoint_sha256",
            "mechanism_seed_checkpoint_bytes",
        ):
            del payload[key]
        return payload

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())


def validate_representation_contract_lineage(
    representation: FullContextRepresentationContract,
    lineage: FullContextLineage,
) -> None:
    if representation.fingerprint != lineage.representation_contract_sha256:
        raise ValueError("typed representation contract differs from frozen lineage")


@dataclass(frozen=True, slots=True)
class FullContextTrainingUnit:
    """One opaque train337 video and its exact consumed stable ranges."""

    video_id: str
    native_length: int
    authorized_ranges: tuple[tuple[int, int], ...]
    authorized_pair_starts: tuple[int, ...]
    window_frames: int
    hop_frames: int

    def __post_init__(self) -> None:
        if not isinstance(self.video_id, str) or not self.video_id:
            raise ValueError("training unit video ID must be a non-empty opaque string")
        _require_integer(self.native_length, role="native length", minimum=1)
        if (self.window_frames, self.hop_frames) not in set(
            _CANDIDATE_GEOMETRY.values()
        ):
            raise ValueError("training unit candidate geometry is unsupported")
        previous_stop = 0
        for index, (start, stop) in enumerate(self.authorized_ranges):
            _require_integer(start, role="stable-range start")
            _require_integer(stop, role="stable-range stop", minimum=1)
            if not 0 <= start < stop <= self.native_length:
                raise ValueError("stable range lies outside the native timeline")
            if index and start < previous_stop:
                raise ValueError("stable ranges must be ordered and disjoint")
            previous_stop = stop
        if not self.authorized_ranges:
            raise ValueError("training unit requires a consumed stable range")
        starts = self.authorized_pair_starts
        if not starts or starts != tuple(sorted(set(starts))):
            raise ValueError("authorized pair starts must be non-empty sorted unique")
        for start in starts:
            _require_integer(start, role="authorized pair start")
            if start % self.hop_frames:
                raise ValueError("authorized pair start violates native hop grid")
            containing = [
                (range_start, range_stop)
                for range_start, range_stop in self.authorized_ranges
                if range_start <= start
                and start + 2 * self.window_frames <= range_stop
            ]
            if len(containing) != 1:
                raise ValueError(
                    "authorized disjoint pair must lie in exactly one stable range"
                )

    @property
    def context_frames(self) -> int:
        return sum(stop - start for start, stop in self.authorized_ranges)

    @property
    def longest_range_frames(self) -> int:
        return max(stop - start for start, stop in self.authorized_ranges)

    @property
    def identity_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "native_length": self.native_length,
            "authorized_ranges": [list(value) for value in self.authorized_ranges],
            "authorized_pair_starts": list(self.authorized_pair_starts),
            "window_frames": self.window_frames,
            "hop_frames": self.hop_frames,
            "context_frames": self.context_frames,
            "longest_range_frames": self.longest_range_frames,
            "video_id_handling": "opaque_hash_order_and_equality_only",
        }


def build_fullcontext_training_units(
    segment_ranges_by_video: Mapping[str, Sequence[tuple[int, int]]],
    pair_eligibility: PairEligibility,
    contract: FullContextTrainerContract,
) -> tuple[FullContextTrainingUnit, ...]:
    """Derive consumed units without expanding representation-authorized starts."""

    expected_ids = set(pair_eligibility.native_lengths)
    if set(segment_ranges_by_video) != expected_ids:
        raise ValueError("segment membership differs from pair eligibility")
    starts_by_video = pair_eligibility.starts_by_variant[contract.candidate_id]
    units: list[FullContextTrainingUnit] = []
    for video_id in sorted(expected_ids):
        eligible = pair_eligibility.representation_eligible[video_id]
        starts = tuple(starts_by_video[video_id])
        if not eligible:
            if starts or segment_ranges_by_video[video_id]:
                raise ValueError("ineligible representation exposes consumable geometry")
            continue
        if not starts:
            continue
        source_ranges = tuple(segment_ranges_by_video[video_id])
        used_ranges = tuple(
            (start, stop)
            for start, stop in source_ranges
            if any(
                start <= pair_start
                and pair_start + 2 * contract.window_frames <= stop
                for pair_start in starts
            )
        )
        unit = FullContextTrainingUnit(
            video_id=video_id,
            native_length=pair_eligibility.native_lengths[video_id],
            authorized_ranges=used_ranges,
            authorized_pair_starts=starts,
            window_frames=contract.window_frames,
            hop_frames=contract.hop_frames,
        )
        if unit.longest_range_frames > contract.maximum_native_length:
            raise ValueError(
                "authorized stable range exceeds encoder positional capacity"
            )
        units.append(unit)
    if not units:
        raise ValueError("no representation-authorized training units")
    return tuple(units)


@dataclass(frozen=True, slots=True)
class LengthBucketBatch:
    video_ids: tuple[str, ...]
    context_frames: int
    longest_range_frames: int
    bucket_upper_bound: int

    def __post_init__(self) -> None:
        if not self.video_ids or len(set(self.video_ids)) != len(self.video_ids):
            raise ValueError("length-bucket batch video IDs must be non-empty unique")
        for role, value in (
            ("batch context frames", self.context_frames),
            ("batch longest range", self.longest_range_frames),
            ("bucket upper bound", self.bucket_upper_bound),
        ):
            _require_integer(value, role=role, minimum=1)
        if self.longest_range_frames > self.bucket_upper_bound:
            raise ValueError("batch range exceeds its length bucket")

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_ids": list(self.video_ids),
            "context_frames": self.context_frames,
            "longest_range_frames": self.longest_range_frames,
            "bucket_upper_bound": self.bucket_upper_bound,
        }


@dataclass(frozen=True, slots=True)
class LengthBucketPlan:
    epoch: int
    candidate_id: CandidateId
    seed: int
    trainer_contract_fingerprint: str
    representation_lineage_fingerprint: str
    units: tuple[FullContextTrainingUnit, ...]
    batches: tuple[LengthBucketBatch, ...]
    maximum_videos_per_batch: int
    maximum_context_frames_per_batch: int

    def __post_init__(self) -> None:
        _require_integer(self.epoch, role="sampler epoch", minimum=1)
        _require_integer(self.seed, role="sampler seed")
        if self.candidate_id not in _CANDIDATE_GEOMETRY:
            raise ValueError("sampler plan candidate is unsupported")
        _require_sha256(
            self.trainer_contract_fingerprint,
            role="trainer contract fingerprint",
        )
        _require_sha256(
            self.representation_lineage_fingerprint,
            role="representation lineage fingerprint",
        )
        for role, value in (
            ("maximum videos per batch", self.maximum_videos_per_batch),
            (
                "maximum context frames per batch",
                self.maximum_context_frames_per_batch,
            ),
        ):
            _require_integer(value, role=role, minimum=1)
        unit_ids = tuple(unit.video_id for unit in self.units)
        if not unit_ids or len(set(unit_ids)) != len(unit_ids):
            raise ValueError("sampler plan units must have unique video IDs")
        observed = tuple(
            video_id for batch in self.batches for video_id in batch.video_ids
        )
        if len(observed) != len(unit_ids) or set(observed) != set(unit_ids):
            raise ValueError("sampler batches must cover every unit exactly once")
        unit_by_id = {unit.video_id: unit for unit in self.units}
        for batch in self.batches:
            if len(batch.video_ids) > self.maximum_videos_per_batch:
                raise ValueError("sampler batch exceeds its video limit")
            expected_frames = sum(
                unit_by_id[video_id].context_frames
                for video_id in batch.video_ids
            )
            expected_longest = max(
                unit_by_id[video_id].longest_range_frames
                for video_id in batch.video_ids
            )
            if (
                batch.context_frames != expected_frames
                or batch.longest_range_frames != expected_longest
            ):
                raise ValueError("sampler batch workload differs from unit geometry")
            if any(
                _length_bucket_upper_bound(unit_by_id[video_id].longest_range_frames)
                != batch.bucket_upper_bound
                for video_id in batch.video_ids
            ):
                raise ValueError("sampler batch mixes different length buckets")
            if (
                batch.context_frames > self.maximum_context_frames_per_batch
                and len(batch.video_ids) != 1
            ):
                raise ValueError("only one long-video unit may exceed the frame limit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_type": "pams_cycleback_length_bucket_plan_v1",
            "epoch": self.epoch,
            "candidate_id": self.candidate_id,
            "seed": self.seed,
            "trainer_contract_fingerprint": self.trainer_contract_fingerprint,
            "representation_lineage_fingerprint": (
                self.representation_lineage_fingerprint
            ),
            "maximum_videos_per_batch": self.maximum_videos_per_batch,
            "maximum_context_frames_per_batch": (
                self.maximum_context_frames_per_batch
            ),
            "unit_total": len(self.units),
            "batch_total": len(self.batches),
            "units": [unit.to_dict() for unit in self.units],
            "batches": [batch.to_dict() for batch in self.batches],
            "ordering_policy": (
                "power_of_two_longest_stable_range_bucket_then_label_free_hash_"
                "within_bucket_and_hash_ordered_batches"
            ),
            "video_id_handling": "opaque_hash_order_and_equality_only",
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())


def _length_bucket_upper_bound(length: int) -> int:
    _require_integer(length, role="stable-range length", minimum=1)
    return 1 << (length - 1).bit_length()


def _opaque_rank(*, seed: int, epoch: int, purpose: str, identity: str) -> bytes:
    return hashlib.sha256(
        (
            f"pams-cycleback-fullcontext-v1\0{seed}\0{epoch}\0{purpose}\0"
            f"{identity}"
        ).encode()
    ).digest()


def build_length_bucket_plan(
    units: Sequence[FullContextTrainingUnit],
    contract: FullContextTrainerContract,
    lineage: FullContextLineage,
    *,
    epoch: int,
) -> LengthBucketPlan:
    """Build a deterministic, label-free, stable-range length-bucket plan."""

    _require_integer(epoch, role="sampler epoch", minimum=1)
    ordered_units = tuple(sorted(units, key=lambda unit: unit.video_id))
    if not ordered_units or len({unit.video_id for unit in ordered_units}) != len(
        ordered_units
    ):
        raise ValueError("sampler units must be non-empty with unique video IDs")
    if any(
        (unit.window_frames, unit.hop_frames)
        != (contract.window_frames, contract.hop_frames)
        for unit in ordered_units
    ):
        raise ValueError("sampler unit geometry differs from trainer candidate")
    if any(
        unit.longest_range_frames > contract.maximum_native_length
        for unit in ordered_units
    ):
        raise ValueError("sampler stable range exceeds encoder positional capacity")
    buckets: dict[int, list[FullContextTrainingUnit]] = {}
    for unit in ordered_units:
        upper = _length_bucket_upper_bound(unit.longest_range_frames)
        buckets.setdefault(upper, []).append(unit)
    batches: list[LengthBucketBatch] = []
    for upper in sorted(buckets):
        ranked = sorted(
            buckets[upper],
            key=lambda unit: (
                _opaque_rank(
                    seed=contract.seed,
                    epoch=epoch,
                    purpose=f"unit-bucket-{upper}",
                    identity=unit.video_id,
                ),
                unit.video_id,
            ),
        )
        current: list[FullContextTrainingUnit] = []
        current_frames = 0
        for unit in ranked:
            would_exceed_videos = len(current) >= contract.maximum_videos_per_batch
            would_exceed_frames = bool(current) and (
                current_frames + unit.context_frames
                > contract.maximum_context_frames_per_batch
            )
            if would_exceed_videos or would_exceed_frames:
                batches.append(
                    LengthBucketBatch(
                        video_ids=tuple(item.video_id for item in current),
                        context_frames=current_frames,
                        longest_range_frames=max(
                            item.longest_range_frames for item in current
                        ),
                        bucket_upper_bound=upper,
                    )
                )
                current = []
                current_frames = 0
            current.append(unit)
            current_frames += unit.context_frames
        if current:
            batches.append(
                LengthBucketBatch(
                    video_ids=tuple(item.video_id for item in current),
                    context_frames=current_frames,
                    longest_range_frames=max(
                        item.longest_range_frames for item in current
                    ),
                    bucket_upper_bound=upper,
                )
            )
    batches.sort(
        key=lambda batch: (
            _opaque_rank(
                seed=contract.seed,
                epoch=epoch,
                purpose="batch-order",
                identity="\0".join(batch.video_ids),
            ),
            batch.bucket_upper_bound,
            batch.video_ids,
        )
    )
    return LengthBucketPlan(
        epoch=epoch,
        candidate_id=contract.candidate_id,
        seed=contract.seed,
        trainer_contract_fingerprint=contract.fingerprint,
        representation_lineage_fingerprint=lineage.fingerprint,
        units=ordered_units,
        batches=tuple(batches),
        maximum_videos_per_batch=contract.maximum_videos_per_batch,
        maximum_context_frames_per_batch=(
            contract.maximum_context_frames_per_batch
        ),
    )


def validate_train337_plan(
    plan: LengthBucketPlan,
    contract: FullContextTrainerContract,
    pair_eligibility: PairEligibility,
) -> None:
    """Bind the eligible plan to an exact 337-member training representation."""

    if len(pair_eligibility.native_lengths) != contract.training_video_total:
        raise ValueError("formal full-context representation must contain train337")
    if plan.candidate_id != contract.candidate_id:
        raise ValueError("sampler candidate differs from trainer contract")
    if plan.trainer_contract_fingerprint != contract.fingerprint:
        raise ValueError("sampler plan differs from trainer contract")
    expected_units = {
        video_id
        for video_id in pair_eligibility.native_lengths
        if pair_eligibility.representation_eligible[video_id]
        and pair_eligibility.starts_by_variant[contract.candidate_id][video_id]
    }
    if {unit.video_id for unit in plan.units} != expected_units:
        raise ValueError("sampler units differ from the authorized train337 subset")


def build_fullcontext_adamw(
    model: nn.Module,
    contract: FullContextTrainerContract,
) -> AdamW:
    """Construct the exact scheduler-free optimizer used by this proxy."""

    return AdamW(
        model.parameters(),
        lr=contract.learning_rate,
        weight_decay=contract.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


def validate_fullcontext_optimizer(
    optimizer: Optimizer,
    contract: FullContextTrainerContract,
) -> None:
    if type(optimizer) is not AdamW:
        raise ValueError("full-context optimizer must be exact torch.optim.AdamW")
    if not optimizer.param_groups:
        raise ValueError("full-context optimizer has no parameter groups")
    for group in optimizer.param_groups:
        if (
            group.get("lr") != contract.learning_rate
            or group.get("weight_decay") != contract.weight_decay
            or tuple(group.get("betas", ())) != (0.9, 0.999)
            or group.get("eps") != 1e-8
            or group.get("amsgrad") is not False
            or group.get("maximize") is not False
        ):
            raise ValueError("full-context AdamW parameter group differs from contract")


def _validate_adamw_state_step(
    optimizer_state: Mapping[str, Any],
    *,
    expected_step: int,
) -> None:
    _require_integer(expected_step, role="expected AdamW step")
    if set(optimizer_state) != {"state", "param_groups"}:
        raise ValueError("AdamW checkpoint state schema mismatch")
    state = optimizer_state["state"]
    param_groups = optimizer_state["param_groups"]
    if (
        not isinstance(state, Mapping)
        or not state
        or not isinstance(param_groups, list)
        or any(
            not isinstance(group, Mapping) or not isinstance(group.get("params"), list)
            for group in param_groups
        )
    ):
        raise ValueError("AdamW checkpoint must contain learned moment state")
    parameter_total = sum(len(group["params"]) for group in param_groups)
    if len(state) != parameter_total:
        raise ValueError("AdamW checkpoint lacks state for an encoder parameter")
    observed: list[int] = []
    for parameter_state in state.values():
        if not isinstance(parameter_state, Mapping) or "step" not in parameter_state:
            raise ValueError("AdamW parameter state is missing its step")
        step = parameter_state["step"]
        if isinstance(step, Tensor):
            if step.numel() != 1 or not bool(torch.isfinite(step).all()):
                raise ValueError("AdamW tensor step must be one finite scalar")
            numeric = float(step.detach().cpu())
        elif isinstance(step, int | float) and not isinstance(step, bool):
            numeric = float(step)
        else:
            raise ValueError("AdamW step has an unsupported type")
        if not numeric.is_integer():
            raise ValueError("AdamW step must be an exact integer")
        observed.append(int(numeric))
    if not observed or any(value != expected_step for value in observed):
        raise ValueError("AdamW state does not match the global optimizer step")


@dataclass(frozen=True, slots=True)
class FullContextOptimizerStep:
    global_optimizer_step: int
    sampler_epoch: int
    sampler_batch_index: int
    video_ids: tuple[str, ...]
    pair_total: int
    unique_view_context_total: int
    reused_pair_side_context_references: int
    view_seeds: tuple[int, int]
    loss: float
    valid_anchor_total: int
    possible_anchor_total: int
    model_state_sha256: str

    def __post_init__(self) -> None:
        for role, value in (
            ("global optimizer step", self.global_optimizer_step),
            ("sampler epoch", self.sampler_epoch),
            ("sampler batch index", self.sampler_batch_index),
            ("pair total", self.pair_total),
            ("unique context total", self.unique_view_context_total),
            (
                "reused pair context references",
                self.reused_pair_side_context_references,
            ),
            ("valid anchor total", self.valid_anchor_total),
            ("possible anchor total", self.possible_anchor_total),
        ):
            _require_integer(value, role=role)
        if not self.video_ids or len(set(self.video_ids)) != len(self.video_ids):
            raise ValueError("optimizer-step video IDs must be non-empty unique")
        if self.view_seeds[0] == self.view_seeds[1]:
            raise ValueError("optimizer-step views require distinct seeds")
        if not math.isfinite(self.loss):
            raise ValueError("optimizer-step loss must be finite")
        _require_sha256(self.model_state_sha256, role="optimizer-step model state")


def model_state_sha256(model_or_state: nn.Module | Mapping[str, Tensor]) -> str:
    state = (
        model_or_state.state_dict()
        if isinstance(model_or_state, nn.Module)
        else model_or_state
    )
    if not isinstance(state, Mapping) or not state:
        raise ValueError("model state must be a non-empty mapping")
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise ValueError("model state must map string names to tensors")
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("model state contains a non-finite tensor")
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json_bytes(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _finite_parameter_gradients(model: nn.Module) -> bool:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    return bool(gradients) and all(
        gradient is not None and bool(torch.isfinite(gradient).all())
        for gradient in gradients
    )


def optimize_real_pair_contexts(
    encoder: PAMSEncoder,
    optimizer: Optimizer,
    objective: ConventionalCycleBackLoss,
    pairs: NativeWindowPairBatch,
    contexts: PairSegmentContexts,
    contract: FullContextTrainerContract,
    *,
    expected_prior_optimizer_step: int | None = None,
) -> tuple[Tensor, int, int]:
    """Apply one update, rejecting every typed diagnostic/null context first."""

    require_real_optimizer_contexts(contexts)
    validate_fullcontext_optimizer(optimizer, contract)
    if expected_prior_optimizer_step is None:
        raise ValueError("real optimizer step requires its exact prior step count")
    _validate_adamw_state_step(
        optimizer.state_dict(),
        expected_step=expected_prior_optimizer_step,
    )
    if pairs.pair_count < 1:
        raise ValueError("full-context optimizer step requires a window pair")
    optimizer.zero_grad(set_to_none=True)
    embeddings_a, embeddings_b = encode_window_pairs(
        encoder,
        pairs,
        contexts,
        batch_size=contract.encoder_context_batch_size,
    )
    output = objective.compute(
        embeddings_a,
        embeddings_b,
        pairs.valid_a,
        pairs.valid_b,
        pairs.source_indices_a,
        pairs.source_indices_b,
        pairs.native_lengths,
        video_ids_a=pairs.video_ids,
        video_ids_b=pairs.video_ids,
    )
    if output.valid_anchor_count < 1 or not bool(torch.isfinite(output.total)):
        raise RuntimeError("full-context optimizer objective is empty or non-finite")
    output.total.backward()
    if not _finite_parameter_gradients(encoder):
        raise RuntimeError("full-context optimizer gradients are missing or non-finite")
    optimizer.step()
    possible = int(pairs.valid_a.sum() + pairs.valid_b.sum())
    if possible < 1:
        raise RuntimeError("full-context optimizer has no possible valid anchors")
    return output.total.detach(), output.valid_anchor_count, possible


def run_unified2d_fullcontext_optimizer_step(
    encoder: PAMSEncoder,
    optimizer: Optimizer,
    objective: ConventionalCycleBackLoss,
    sequences_by_video: Mapping[str, Unified2DSequence],
    segment_ranges_by_video: Mapping[str, Sequence[tuple[int, int]]],
    pair_eligibility: PairEligibility,
    candidate_config: ConventionalCycleBackConfig,
    contract: FullContextTrainerContract,
    representation: FullContextRepresentationContract,
    lineage: FullContextLineage,
    plan: LengthBucketPlan,
    *,
    batch_index: int,
    global_optimizer_step: int,
) -> FullContextOptimizerStep:
    """Run the explicit v4e adapter into the representation-neutral optimizer."""

    validate_representation_contract_lineage(representation, lineage)
    if (
        representation.representation_family != "v4e_unified2d_coco17_padded33"
        or representation.support_channel_to_pose_joint_indices != tuple(range(17))
        or representation.diagnostic_adapter
        != "unified2d_coco17_joint_nulls_v1"
    ):
        raise ValueError("unified2d adapter received a different representation type")
    if contract.candidate_config_fingerprint != candidate_config.fingerprint:
        raise ValueError("candidate config differs from full-context contract")
    if plan.trainer_contract_fingerprint != contract.fingerprint:
        raise ValueError("sampler plan differs from full-context contract")
    if plan.representation_lineage_fingerprint != lineage.fingerprint:
        raise ValueError("sampler plan differs from representation/mechanism lineage")
    if not 0 <= batch_index < len(plan.batches):
        raise ValueError("sampler batch cursor lies outside the plan")
    _require_integer(
        global_optimizer_step,
        role="global optimizer step",
        minimum=_MECHANISM_OPTIMIZER_STEPS + 1,
    )
    selected_ids = plan.batches[batch_index].video_ids
    if any(video_id not in sequences_by_video for video_id in selected_ids):
        raise ValueError("sampler plan references an unavailable pose sequence")
    selected = tuple(sequences_by_video[video_id] for video_id in selected_ids)
    if tuple(sequence.video_id for sequence in selected) != selected_ids:
        raise ValueError("pose sequence identity differs from sampler plan")
    device = next(encoder.parameters()).device
    batch = collate_to_device(selected, device)
    pairs, views = pairs_from_batch(
        batch,
        candidate_config,
        step=global_optimizer_step,
        segment_ranges_by_video=segment_ranges_by_video,
        pair_eligibility=pair_eligibility,
    )
    unit_by_id = {unit.video_id: unit for unit in plan.units}
    expected_pair_total = sum(
        len(unit_by_id[video_id].authorized_pair_starts)
        for video_id in selected_ids
    )
    if pairs.pair_count != expected_pair_total or set(pairs.video_ids) != set(
        selected_ids
    ):
        raise RuntimeError("runtime pair set differs from the frozen sampler plan")
    contexts = pair_segment_contexts(pairs, views)
    if (
        contexts.transform_contract.get("same_view_same_range_encoded_once") is not True
        or contexts.transform_contract.get("window_only_encoder_path_used") is not False
        or contexts.transform_contract.get("attention_context_boundary")
        != "eligible_frame_range_start_stop"
    ):
        raise RuntimeError("full-context encode-once contract is incomplete")
    expected_view_seeds = independent_view_seeds(
        base_seed=contract.seed,
        step=global_optimizer_step,
    )
    if views.view_seeds != expected_view_seeds:
        raise RuntimeError("runtime view seeds differ from the checkpointable policy")
    encoder.train()
    loss, valid_anchors, possible_anchors = optimize_real_pair_contexts(
        encoder,
        optimizer,
        objective,
        pairs,
        contexts,
        contract,
        expected_prior_optimizer_step=global_optimizer_step - 1,
    )
    return FullContextOptimizerStep(
        global_optimizer_step=global_optimizer_step,
        sampler_epoch=plan.epoch,
        sampler_batch_index=batch_index,
        video_ids=selected_ids,
        pair_total=pairs.pair_count,
        unique_view_context_total=int(
            contexts.transform_contract["unique_view_context_total"]
        ),
        reused_pair_side_context_references=int(
            contexts.transform_contract["reused_pair_side_context_references"]
        ),
        view_seeds=views.view_seeds,
        loss=float(loss),
        valid_anchor_total=valid_anchors,
        possible_anchor_total=possible_anchors,
        model_state_sha256=model_state_sha256(encoder),
    )


@dataclass(frozen=True, slots=True)
class FullContextProgress:
    phase: TrainingPhase
    target_epoch: int
    completed_epochs: int
    continuation_optimizer_steps: int
    active_plan: LengthBucketPlan | None
    sampler_cursor: int
    epoch11_prefix_checkpoint_sha256: str | None = None
    epoch11_prefix_checkpoint_bytes: int | None = None

    def __post_init__(self) -> None:
        expected_target = {
            "epoch11": _EPOCH11_TARGET,
            "epoch150": _EPOCH150_TARGET,
        }.get(self.phase)
        if expected_target is None or self.target_epoch != expected_target:
            raise ValueError("full-context phase target mismatch")
        minimum_completed = 0 if self.phase == "epoch11" else _EPOCH11_TARGET
        if not minimum_completed <= self.completed_epochs <= self.target_epoch:
            raise ValueError("completed epoch count lies outside the phase")
        _require_integer(
            self.continuation_optimizer_steps,
            role="continuation optimizer steps",
        )
        _require_integer(self.sampler_cursor, role="sampler cursor")
        if self.phase == "epoch11":
            if (
                self.epoch11_prefix_checkpoint_sha256 is not None
                or self.epoch11_prefix_checkpoint_bytes is not None
            ):
                raise ValueError("epoch11 phase cannot name its own prefix checkpoint")
        else:
            if self.epoch11_prefix_checkpoint_sha256 is None:
                raise ValueError("epoch150 phase requires the exact epoch11 prefix")
            _require_sha256(
                self.epoch11_prefix_checkpoint_sha256,
                role="epoch11 prefix checkpoint",
            )
            _require_integer(
                self.epoch11_prefix_checkpoint_bytes,
                role="epoch11 prefix checkpoint bytes",
                minimum=1,
            )
        if self.completed_epochs == self.target_epoch:
            if self.active_plan is not None or self.sampler_cursor != 0:
                raise ValueError("completed phase cannot retain an active sampler plan")
        else:
            if self.active_plan is None:
                raise ValueError("incomplete phase requires an active sampler plan")
            if self.active_plan.epoch != self.completed_epochs + 1:
                raise ValueError("active sampler plan epoch is not the next epoch")
            if not 0 <= self.sampler_cursor < len(self.active_plan.batches):
                raise ValueError("sampler cursor lies outside the active plan")

    @property
    def global_optimizer_step(self) -> int:
        return _MECHANISM_OPTIMIZER_STEPS + self.continuation_optimizer_steps

    @property
    def next_augmentation_step(self) -> int:
        return self.global_optimizer_step + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "target_epoch": self.target_epoch,
            "completed_epochs": self.completed_epochs,
            "continuation_optimizer_steps": self.continuation_optimizer_steps,
            "global_optimizer_step": self.global_optimizer_step,
            "active_plan": (
                None if self.active_plan is None else self.active_plan.to_dict()
            ),
            "active_plan_fingerprint": (
                None if self.active_plan is None else self.active_plan.fingerprint
            ),
            "sampler_cursor": self.sampler_cursor,
            "epoch11_prefix_checkpoint_sha256": (
                self.epoch11_prefix_checkpoint_sha256
            ),
            "epoch11_prefix_checkpoint_bytes": (
                self.epoch11_prefix_checkpoint_bytes
            ),
        }


def initial_epoch11_progress(plan: LengthBucketPlan) -> FullContextProgress:
    if plan.epoch != 1:
        raise ValueError("epoch11 continuation must begin with epoch-one sampler plan")
    return FullContextProgress(
        phase="epoch11",
        target_epoch=_EPOCH11_TARGET,
        completed_epochs=0,
        continuation_optimizer_steps=0,
        active_plan=plan,
        sampler_cursor=0,
    )


def initial_epoch150_progress(
    plan: LengthBucketPlan,
    *,
    epoch11_checkpoint_sha256: str,
    epoch11_checkpoint_bytes: int,
    epoch11_continuation_optimizer_steps: int,
) -> FullContextProgress:
    if plan.epoch != _EPOCH11_TARGET + 1:
        raise ValueError("epoch150 continuation must begin with epoch-twelve plan")
    return FullContextProgress(
        phase="epoch150",
        target_epoch=_EPOCH150_TARGET,
        completed_epochs=_EPOCH11_TARGET,
        continuation_optimizer_steps=epoch11_continuation_optimizer_steps,
        active_plan=plan,
        sampler_cursor=0,
        epoch11_prefix_checkpoint_sha256=epoch11_checkpoint_sha256,
        epoch11_prefix_checkpoint_bytes=epoch11_checkpoint_bytes,
    )


def advance_fullcontext_progress(
    progress: FullContextProgress,
    *,
    next_epoch_plan: LengthBucketPlan | None = None,
) -> FullContextProgress:
    """Advance exactly one completed optimizer step without losing cursor state."""

    if progress.active_plan is None:
        raise ValueError("completed full-context phase cannot advance")
    next_cursor = progress.sampler_cursor + 1
    next_steps = progress.continuation_optimizer_steps + 1
    if next_cursor < len(progress.active_plan.batches):
        if next_epoch_plan is not None:
            raise ValueError("next epoch plan supplied before the current epoch ended")
        return FullContextProgress(
            phase=progress.phase,
            target_epoch=progress.target_epoch,
            completed_epochs=progress.completed_epochs,
            continuation_optimizer_steps=next_steps,
            active_plan=progress.active_plan,
            sampler_cursor=next_cursor,
            epoch11_prefix_checkpoint_sha256=(
                progress.epoch11_prefix_checkpoint_sha256
            ),
            epoch11_prefix_checkpoint_bytes=(
                progress.epoch11_prefix_checkpoint_bytes
            ),
        )
    completed = progress.completed_epochs + 1
    if completed == progress.target_epoch:
        if next_epoch_plan is not None:
            raise ValueError("completed phase cannot accept another sampler plan")
        return FullContextProgress(
            phase=progress.phase,
            target_epoch=progress.target_epoch,
            completed_epochs=completed,
            continuation_optimizer_steps=next_steps,
            active_plan=None,
            sampler_cursor=0,
            epoch11_prefix_checkpoint_sha256=(
                progress.epoch11_prefix_checkpoint_sha256
            ),
            epoch11_prefix_checkpoint_bytes=(
                progress.epoch11_prefix_checkpoint_bytes
            ),
        )
    if next_epoch_plan is None or next_epoch_plan.epoch != completed + 1:
        raise ValueError("epoch boundary requires the exact next sampler plan")
    if (
        next_epoch_plan.trainer_contract_fingerprint
        != progress.active_plan.trainer_contract_fingerprint
        or next_epoch_plan.representation_lineage_fingerprint
        != progress.active_plan.representation_lineage_fingerprint
        or tuple(unit.identity_sha256 for unit in next_epoch_plan.units)
        != tuple(unit.identity_sha256 for unit in progress.active_plan.units)
    ):
        raise ValueError("next sampler plan changes contract, lineage, or units")
    return FullContextProgress(
        phase=progress.phase,
        target_epoch=progress.target_epoch,
        completed_epochs=completed,
        continuation_optimizer_steps=next_steps,
        active_plan=next_epoch_plan,
        sampler_cursor=0,
        epoch11_prefix_checkpoint_sha256=(
            progress.epoch11_prefix_checkpoint_sha256
        ),
        epoch11_prefix_checkpoint_bytes=progress.epoch11_prefix_checkpoint_bytes,
    )


def capture_fullcontext_rng_state() -> dict[str, Any]:
    """Capture every RNG stream that can affect a continuation step."""

    cuda_states: tuple[Tensor, ...] = ()
    if torch.cuda.is_available():
        cuda_states = tuple(
            state.detach().cpu().clone() for state in torch.cuda.get_rng_state_all()
        )
    return {
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().detach().cpu().clone(),
        "torch_cuda_all": cuda_states,
        "torch_cuda_device_count": torch.cuda.device_count(),
    }


def _rng_state_sha256(state: Mapping[str, Any]) -> str:
    if set(state) != {
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda_all",
        "torch_cuda_device_count",
    }:
        raise ValueError("full-context RNG state schema mismatch")
    torch_cpu = state["torch_cpu"]
    cuda_states = state["torch_cuda_all"]
    device_count = state["torch_cuda_device_count"]
    if not isinstance(torch_cpu, Tensor) or torch_cpu.dtype != torch.uint8:
        raise ValueError("CPU RNG state must be a uint8 tensor")
    if not isinstance(cuda_states, Sequence) or any(
        not isinstance(value, Tensor) or value.dtype != torch.uint8
        for value in cuda_states
    ):
        raise ValueError("CUDA RNG state must be a sequence of uint8 tensors")
    _require_integer(device_count, role="CUDA RNG device count")
    if len(cuda_states) != device_count:
        raise ValueError("CUDA RNG state count differs from captured device count")
    python_state = state["python"]
    numpy_state = state["numpy"]
    if not isinstance(python_state, tuple) or not isinstance(numpy_state, tuple):
        raise ValueError("Python and NumPy RNG states must be tuples")
    digest = hashlib.sha256()
    digest.update(repr(python_state).encode("utf-8"))
    if len(numpy_state) != 5 or not isinstance(numpy_state[1], np.ndarray):
        raise ValueError("NumPy RNG state schema mismatch")
    digest.update(str(numpy_state[0]).encode("utf-8"))
    digest.update(np.ascontiguousarray(numpy_state[1]).tobytes(order="C"))
    digest.update(_canonical_json_bytes([numpy_state[2], numpy_state[3], numpy_state[4]]))
    for tensor in (torch_cpu, *cuda_states):
        materialized = tensor.detach().cpu().contiguous()
        digest.update(_canonical_json_bytes(list(materialized.shape)))
        digest.update(materialized.numpy().tobytes(order="C"))
    digest.update(str(device_count).encode("ascii"))
    return digest.hexdigest()


def restore_fullcontext_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a previously validated complete RNG state."""

    _rng_state_sha256(state)
    expected_devices = int(state["torch_cuda_device_count"])
    if torch.cuda.device_count() != expected_devices:
        raise ValueError("CUDA device count differs from the checkpoint")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].detach().cpu())
    if expected_devices:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint requires CUDA RNG streams")
        torch.cuda.set_rng_state_all(
            [value.detach().cpu() for value in state["torch_cuda_all"]]
        )


def _clone_model_state(model: nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _view_seed_state(
    contract: FullContextTrainerContract,
    progress: FullContextProgress,
) -> dict[str, Any]:
    seeds = independent_view_seeds(
        base_seed=contract.seed,
        step=progress.next_augmentation_step,
    )
    digest = hashlib.sha256()
    for step in range(
        _MECHANISM_OPTIMIZER_STEPS + 1,
        progress.global_optimizer_step + 1,
    ):
        consumed = independent_view_seeds(base_seed=contract.seed, step=step)
        digest.update(_canonical_json_bytes([step, consumed[0], consumed[1]]))
    return {
        "policy": "pams-conventional-cycleback-view-v1",
        "base_seed": contract.seed,
        "first_continuation_augmentation_step": (
            _MECHANISM_OPTIMIZER_STEPS + 1
        ),
        "last_consumed_augmentation_step": progress.global_optimizer_step,
        "consumed_view_seed_chain_sha256": digest.hexdigest(),
        "next_augmentation_step": progress.next_augmentation_step,
        "next_view_seeds": list(seeds),
        "independent_views": True,
    }


def _authority_boundaries() -> dict[str, bool]:
    return {
        "epoch11_train337_continuation_authorized": False,
        "epoch150_train337_continuation_authorized": False,
        "development_evaluation_authorized": False,
        "sealed_evaluation_authorized": False,
        "readout_authorized": False,
    }


def build_fullcontext_checkpoint_payload(
    model: nn.Module,
    optimizer: Optimizer,
    contract: FullContextTrainerContract,
    lineage: FullContextLineage,
    progress: FullContextProgress,
) -> dict[str, Any]:
    """Snapshot a complete continuation state; the checkpoint grants no authority."""

    validate_fullcontext_optimizer(optimizer, contract)
    _validate_adamw_state_step(
        optimizer.state_dict(),
        expected_step=progress.global_optimizer_step,
    )
    if progress.active_plan is not None and (
        progress.active_plan.trainer_contract_fingerprint != contract.fingerprint
        or progress.active_plan.representation_lineage_fingerprint
        != lineage.fingerprint
    ):
        raise ValueError("active sampler plan differs from checkpoint lineage")
    model_state = _clone_model_state(model)
    rng_state = capture_fullcontext_rng_state()
    return {
        "schema_version": 1,
        "artifact_type": "pams_cycleback_fullcontext_checkpoint_v1",
        "classification": contract.classification,
        "candidate_id": contract.candidate_id,
        "trainer_contract": contract.to_dict(),
        "trainer_contract_fingerprint": contract.fingerprint,
        "lineage": lineage.to_dict(),
        "lineage_fingerprint": lineage.fingerprint,
        "progress": progress.to_dict(),
        "model_state": model_state,
        "model_state_sha256": model_state_sha256(model_state),
        "optimizer_name": "AdamW",
        "optimizer_state": copy.deepcopy(optimizer.state_dict()),
        "scheduler": "none",
        "scheduler_state": None,
        "rng_state": rng_state,
        "rng_state_sha256": _rng_state_sha256(rng_state),
        "view_seed_state": _view_seed_state(contract, progress),
        "determinism_contract": {
            "length_bucket_plan_and_cursor_checkpointed": True,
            "python_numpy_torch_cpu_all_cuda_rng_checkpointed": True,
            "view_seed_policy_and_next_seeds_checkpointed": True,
            "same_view_same_range_encoded_once": True,
            "exact_length_context_buckets": True,
            "training_inference_context_parity": True,
            "scheduler": "none",
        },
        "authority_boundaries": _authority_boundaries(),
    }


def _parse_training_unit(payload: Mapping[str, Any]) -> FullContextTrainingUnit:
    expected = {
        "video_id",
        "native_length",
        "authorized_ranges",
        "authorized_pair_starts",
        "window_frames",
        "hop_frames",
        "context_frames",
        "longest_range_frames",
        "video_id_handling",
    }
    if set(payload) != expected:
        raise ValueError("checkpoint sampler unit schema mismatch")
    ranges = payload["authorized_ranges"]
    starts = payload["authorized_pair_starts"]
    if (
        not isinstance(ranges, list)
        or any(
            not isinstance(value, list)
            or len(value) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
            for value in ranges
        )
        or not isinstance(starts, list)
    ):
        raise ValueError("checkpoint sampler unit geometry schema mismatch")
    unit = FullContextTrainingUnit(
        video_id=payload["video_id"],
        native_length=payload["native_length"],
        authorized_ranges=tuple((value[0], value[1]) for value in ranges),
        authorized_pair_starts=tuple(starts),
        window_frames=payload["window_frames"],
        hop_frames=payload["hop_frames"],
    )
    if unit.to_dict() != dict(payload):
        raise ValueError("checkpoint sampler unit derived fields mismatch")
    return unit


def _parse_length_bucket_plan(
    payload: Mapping[str, Any],
    *,
    expected_fingerprint: str,
) -> LengthBucketPlan:
    expected = {
        "schema_version",
        "artifact_type",
        "epoch",
        "candidate_id",
        "seed",
        "trainer_contract_fingerprint",
        "representation_lineage_fingerprint",
        "maximum_videos_per_batch",
        "maximum_context_frames_per_batch",
        "unit_total",
        "batch_total",
        "units",
        "batches",
        "ordering_policy",
        "video_id_handling",
    }
    if set(payload) != expected or (
        payload["schema_version"] != 1
        or payload["artifact_type"] != "pams_cycleback_length_bucket_plan_v1"
        or payload["ordering_policy"]
        != (
            "power_of_two_longest_stable_range_bucket_then_label_free_hash_"
            "within_bucket_and_hash_ordered_batches"
        )
        or payload["video_id_handling"] != "opaque_hash_order_and_equality_only"
        or not isinstance(payload["units"], list)
        or not isinstance(payload["batches"], list)
    ):
        raise ValueError("checkpoint sampler plan schema mismatch")
    units = tuple(_parse_training_unit(value) for value in payload["units"])
    batches: list[LengthBucketBatch] = []
    for value in payload["batches"]:
        if not isinstance(value, Mapping) or set(value) != {
            "video_ids",
            "context_frames",
            "longest_range_frames",
            "bucket_upper_bound",
        } or not isinstance(value["video_ids"], list):
            raise ValueError("checkpoint length-bucket batch schema mismatch")
        batches.append(
            LengthBucketBatch(
                video_ids=tuple(value["video_ids"]),
                context_frames=value["context_frames"],
                longest_range_frames=value["longest_range_frames"],
                bucket_upper_bound=value["bucket_upper_bound"],
            )
        )
    plan = LengthBucketPlan(
        epoch=payload["epoch"],
        candidate_id=payload["candidate_id"],
        seed=payload["seed"],
        trainer_contract_fingerprint=payload["trainer_contract_fingerprint"],
        representation_lineage_fingerprint=payload[
            "representation_lineage_fingerprint"
        ],
        units=units,
        batches=tuple(batches),
        maximum_videos_per_batch=payload["maximum_videos_per_batch"],
        maximum_context_frames_per_batch=payload[
            "maximum_context_frames_per_batch"
        ],
    )
    if (
        payload["unit_total"] != len(units)
        or payload["batch_total"] != len(batches)
        or plan.to_dict() != dict(payload)
        or plan.fingerprint != expected_fingerprint
    ):
        raise ValueError("checkpoint sampler plan fingerprint/count mismatch")
    return plan


def _parse_progress(payload: Mapping[str, Any]) -> FullContextProgress:
    expected = {
        "phase",
        "target_epoch",
        "completed_epochs",
        "continuation_optimizer_steps",
        "global_optimizer_step",
        "active_plan",
        "active_plan_fingerprint",
        "sampler_cursor",
        "epoch11_prefix_checkpoint_sha256",
        "epoch11_prefix_checkpoint_bytes",
    }
    if set(payload) != expected:
        raise ValueError("checkpoint progress schema mismatch")
    active_payload = payload["active_plan"]
    active_fingerprint = payload["active_plan_fingerprint"]
    if active_payload is None:
        if active_fingerprint is not None:
            raise ValueError("absent sampler plan cannot have a fingerprint")
        active_plan = None
    else:
        if not isinstance(active_payload, Mapping) or not isinstance(
            active_fingerprint, str
        ):
            raise ValueError("active sampler plan schema mismatch")
        active_plan = _parse_length_bucket_plan(
            active_payload,
            expected_fingerprint=active_fingerprint,
        )
    progress = FullContextProgress(
        phase=payload["phase"],
        target_epoch=payload["target_epoch"],
        completed_epochs=payload["completed_epochs"],
        continuation_optimizer_steps=payload["continuation_optimizer_steps"],
        active_plan=active_plan,
        sampler_cursor=payload["sampler_cursor"],
        epoch11_prefix_checkpoint_sha256=payload[
            "epoch11_prefix_checkpoint_sha256"
        ],
        epoch11_prefix_checkpoint_bytes=payload[
            "epoch11_prefix_checkpoint_bytes"
        ],
    )
    if progress.to_dict() != dict(payload):
        raise ValueError("checkpoint progress derived fields mismatch")
    return progress


def validate_fullcontext_checkpoint_payload(
    payload: Mapping[str, Any],
    contract: FullContextTrainerContract,
    lineage: FullContextLineage,
    *,
    expected_phase: TrainingPhase,
) -> FullContextProgress:
    expected = {
        "schema_version",
        "artifact_type",
        "classification",
        "candidate_id",
        "trainer_contract",
        "trainer_contract_fingerprint",
        "lineage",
        "lineage_fingerprint",
        "progress",
        "model_state",
        "model_state_sha256",
        "optimizer_name",
        "optimizer_state",
        "scheduler",
        "scheduler_state",
        "rng_state",
        "rng_state_sha256",
        "view_seed_state",
        "determinism_contract",
        "authority_boundaries",
    }
    if set(payload) != expected or (
        payload["schema_version"] != 1
        or payload["artifact_type"] != "pams_cycleback_fullcontext_checkpoint_v1"
        or payload["classification"] != contract.classification
        or payload["candidate_id"] != contract.candidate_id
        or payload["trainer_contract"] != contract.to_dict()
        or payload["trainer_contract_fingerprint"] != contract.fingerprint
        or payload["lineage"] != lineage.to_dict()
        or payload["lineage_fingerprint"] != lineage.fingerprint
    ):
        raise ValueError("full-context checkpoint contract or lineage mismatch")
    progress_payload = payload["progress"]
    if not isinstance(progress_payload, Mapping):
        raise ValueError("full-context checkpoint progress must be an object")
    progress = _parse_progress(progress_payload)
    if progress.phase != expected_phase:
        raise ValueError("full-context checkpoint phase mismatch")
    if progress.active_plan is not None and (
        progress.active_plan.trainer_contract_fingerprint != contract.fingerprint
        or progress.active_plan.representation_lineage_fingerprint
        != lineage.fingerprint
    ):
        raise ValueError("checkpoint sampler plan lineage mismatch")
    model_state = payload["model_state"]
    if not isinstance(model_state, Mapping) or (
        model_state_sha256(model_state) != payload["model_state_sha256"]
    ):
        raise ValueError("full-context checkpoint model-state digest mismatch")
    if (
        payload["optimizer_name"] != "AdamW"
        or not isinstance(payload["optimizer_state"], Mapping)
        or payload["scheduler"] != "none"
        or payload["scheduler_state"] is not None
    ):
        raise ValueError("full-context checkpoint optimizer/scheduler mismatch")
    _validate_adamw_state_step(
        payload["optimizer_state"],
        expected_step=progress.global_optimizer_step,
    )
    rng_state = payload["rng_state"]
    if not isinstance(rng_state, Mapping) or (
        _rng_state_sha256(rng_state) != payload["rng_state_sha256"]
    ):
        raise ValueError("full-context checkpoint RNG-state digest mismatch")
    if payload["view_seed_state"] != _view_seed_state(contract, progress):
        raise ValueError("full-context checkpoint next view seeds mismatch")
    if payload["determinism_contract"] != {
        "length_bucket_plan_and_cursor_checkpointed": True,
        "python_numpy_torch_cpu_all_cuda_rng_checkpointed": True,
        "view_seed_policy_and_next_seeds_checkpointed": True,
        "same_view_same_range_encoded_once": True,
        "exact_length_context_buckets": True,
        "training_inference_context_parity": True,
        "scheduler": "none",
    }:
        raise ValueError("full-context checkpoint determinism contract mismatch")
    if payload["authority_boundaries"] != _authority_boundaries():
        raise ValueError("full-context checkpoint illegally grants downstream authority")
    return progress


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_save_fullcontext_checkpoint(
    payload: Mapping[str, Any],
    path: str | Path,
) -> tuple[str, int]:
    """Atomically replace one checkpoint and return its exact byte identity."""

    destination = Path(path)
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("checkpoint parent must be an existing non-symlink directory")
    if destination.is_symlink():
        raise ValueError("checkpoint destination cannot be a symlink")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=parent,
            prefix=f".{destination.name}.",
            suffix=".incomplete",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
        _fsync_directory(parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
            _fsync_directory(parent)
    return stable_file_bytes(destination)[1]


def load_fullcontext_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    contract: FullContextTrainerContract,
    lineage: FullContextLineage,
    expected_phase: TrainingPhase,
    model: nn.Module,
    optimizer: Optimizer,
) -> FullContextProgress:
    """Verify exact bytes and semantics before restoring model, optimizer, and RNG."""

    _require_sha256(expected_sha256, role="expected checkpoint")
    _require_integer(expected_bytes, role="expected checkpoint bytes", minimum=1)
    encoded, identity = stable_file_bytes(path)
    if identity != (expected_sha256, expected_bytes):
        raise ValueError("full-context checkpoint bytes differ from authority")
    value = torch.load(
        io.BytesIO(encoded),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if not isinstance(value, Mapping):
        raise ValueError("full-context checkpoint root must be a mapping")
    progress = validate_fullcontext_checkpoint_payload(
        value,
        contract,
        lineage,
        expected_phase=expected_phase,
    )
    validate_fullcontext_optimizer(optimizer, contract)
    model.load_state_dict(value["model_state"], strict=True)
    optimizer.load_state_dict(value["optimizer_state"])
    validate_fullcontext_optimizer(optimizer, contract)
    _validate_adamw_state_step(
        optimizer.state_dict(),
        expected_step=progress.global_optimizer_step,
    )
    restore_fullcontext_rng_state(value["rng_state"])
    if model_state_sha256(model) != value["model_state_sha256"]:
        raise RuntimeError("restored full-context model state differs from checkpoint")
    return progress


def initialize_epoch150_from_exact_epoch11_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    contract: FullContextTrainerContract,
    lineage: FullContextLineage,
    epoch12_plan: LengthBucketPlan,
    model: nn.Module,
    optimizer: Optimizer,
) -> FullContextProgress:
    """Restore a completed epoch11 prefix and create only the epoch12 cursor."""

    prefix = load_fullcontext_checkpoint(
        path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
        contract=contract,
        lineage=lineage,
        expected_phase="epoch11",
        model=model,
        optimizer=optimizer,
    )
    if (
        prefix.completed_epochs != _EPOCH11_TARGET
        or prefix.target_epoch != _EPOCH11_TARGET
        or prefix.active_plan is not None
        or prefix.sampler_cursor != 0
    ):
        raise ValueError("epoch150 requires an exactly completed epoch11 prefix")
    if (
        epoch12_plan.trainer_contract_fingerprint != contract.fingerprint
        or epoch12_plan.representation_lineage_fingerprint != lineage.fingerprint
    ):
        raise ValueError("epoch12 sampler plan differs from the exact epoch11 lineage")
    return initial_epoch150_progress(
        epoch12_plan,
        epoch11_checkpoint_sha256=expected_sha256,
        epoch11_checkpoint_bytes=expected_bytes,
        epoch11_continuation_optimizer_steps=(
            prefix.continuation_optimizer_steps
        ),
    )


def validate_mechanism_seed_checkpoint_payload(
    payload: Mapping[str, Any],
    contract: FullContextTrainerContract,
    lineage: FullContextLineage,
) -> None:
    """Validate the future exact L/optimizer/RNG seed required for epoch one.

    No currently produced mechanism artifact satisfies this schema.  In
    particular, a JSON-only ``final_model_state_sha256`` is not a checkpoint
    and cannot be upgraded or reconstructed after the run.
    """

    expected = {
        "schema_version",
        "artifact_type",
        "status",
        "candidate_id",
        "trainer_contract_fingerprint",
        "lineage",
        "lineage_fingerprint",
        "completed_optimizer_steps",
        "completed_epochs",
        "model_state",
        "model_state_sha256",
        "model_state_role",
        "optimizer_name",
        "optimizer_state",
        "scheduler",
        "scheduler_state",
        "rng_state",
        "rng_state_sha256",
        "next_view_seed_state",
        "authority_boundaries",
    }
    if set(payload) != expected or (
        payload["schema_version"] != 1
        or payload["artifact_type"]
        != "pams_conventional_cycleback_mechanism_seed_checkpoint_v1"
        or payload["status"] != "passed"
        or payload["candidate_id"] != contract.candidate_id
        or payload["trainer_contract_fingerprint"] != contract.fingerprint
        or payload["lineage"] != lineage.mechanism_seed_predecessor_dict()
        or payload["lineage_fingerprint"]
        != _canonical_sha256(lineage.mechanism_seed_predecessor_dict())
        or payload["completed_optimizer_steps"] != _MECHANISM_OPTIMIZER_STEPS
        or payload["completed_epochs"] != 0
        or payload["model_state_role"] != "L_learned_cycleback_encoder"
        or payload["optimizer_name"] != "AdamW"
        or payload["scheduler"] != "none"
        or payload["scheduler_state"] is not None
    ):
        raise ValueError("mechanism seed checkpoint contract or lineage mismatch")
    state = payload["model_state"]
    if not isinstance(state, Mapping) or (
        model_state_sha256(state) != payload["model_state_sha256"]
        or payload["model_state_sha256"]
        != lineage.mechanism_learned_model_state_sha256
    ):
        raise ValueError("mechanism learned-L state digest mismatch")
    if not isinstance(payload["optimizer_state"], Mapping):
        raise ValueError("mechanism seed checkpoint lacks optimizer state")
    _validate_adamw_state_step(
        payload["optimizer_state"],
        expected_step=_MECHANISM_OPTIMIZER_STEPS,
    )
    rng_state = payload["rng_state"]
    if not isinstance(rng_state, Mapping) or (
        _rng_state_sha256(rng_state) != payload["rng_state_sha256"]
    ):
        raise ValueError("mechanism seed checkpoint RNG-state digest mismatch")
    mechanism_view_digest = hashlib.sha256()
    for step in range(1, _MECHANISM_OPTIMIZER_STEPS + 1):
        consumed = independent_view_seeds(base_seed=contract.seed, step=step)
        mechanism_view_digest.update(
            _canonical_json_bytes([step, consumed[0], consumed[1]])
        )
    expected_view_state = {
        "policy": "pams-conventional-cycleback-view-v1",
        "base_seed": contract.seed,
        "first_optimizer_augmentation_step": 1,
        "last_consumed_augmentation_step": _MECHANISM_OPTIMIZER_STEPS,
        "consumed_view_seed_chain_sha256": mechanism_view_digest.hexdigest(),
        "next_augmentation_step": _MECHANISM_OPTIMIZER_STEPS + 1,
        "next_view_seeds": list(
            independent_view_seeds(
                base_seed=contract.seed,
                step=_MECHANISM_OPTIMIZER_STEPS + 1,
            )
        ),
        "independent_views": True,
    }
    if payload["next_view_seed_state"] != expected_view_state:
        raise ValueError("mechanism seed next-view state mismatch")
    if payload["authority_boundaries"] != _authority_boundaries():
        raise ValueError("mechanism checkpoint may not self-authorize continuation")


def load_mechanism_seed_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    contract: FullContextTrainerContract,
    lineage: FullContextLineage,
    model: nn.Module,
    optimizer: Optimizer,
) -> None:
    """Restore exact learned L only after byte and semantic verification."""

    if (
        expected_sha256 != lineage.mechanism_seed_checkpoint_sha256
        or expected_bytes != lineage.mechanism_seed_checkpoint_bytes
    ):
        raise ValueError("mechanism seed identity differs from frozen lineage")
    encoded, identity = stable_file_bytes(path)
    if identity != (expected_sha256, expected_bytes):
        raise ValueError("mechanism seed checkpoint bytes differ from authority")
    value = torch.load(
        io.BytesIO(encoded),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if not isinstance(value, Mapping):
        raise ValueError("mechanism seed checkpoint root must be a mapping")
    validate_mechanism_seed_checkpoint_payload(value, contract, lineage)
    validate_fullcontext_optimizer(optimizer, contract)
    model.load_state_dict(value["model_state"], strict=True)
    optimizer.load_state_dict(value["optimizer_state"])
    validate_fullcontext_optimizer(optimizer, contract)
    _validate_adamw_state_step(
        optimizer.state_dict(),
        expected_step=_MECHANISM_OPTIMIZER_STEPS,
    )
    restore_fullcontext_rng_state(value["rng_state"])
    if model_state_sha256(model) != lineage.mechanism_learned_model_state_sha256:
        raise RuntimeError("restored mechanism seed is not exact learned L")


def pair_context_fingerprints(
    pairs: NativeWindowPairBatch,
    contexts: PairSegmentContexts,
) -> tuple[str, str]:
    """Bind a fixed evaluation pair identity and its exact input tensor bytes."""

    identity = hashlib.sha256(
        _canonical_json_bytes(
            {
                "video_ids": list(pairs.video_ids),
                "starts_a": pairs.starts_a.detach().cpu().tolist(),
                "starts_b": pairs.starts_b.detach().cpu().tolist(),
                "segment_starts": pairs.segment_starts.detach().cpu().tolist(),
                "segment_ends": pairs.segment_ends.detach().cpu().tolist(),
                "source_indices_a": pairs.source_indices_a.detach().cpu().tolist(),
                "source_indices_b": pairs.source_indices_b.detach().cpu().tolist(),
                "native_lengths": pairs.native_lengths.detach().cpu().tolist(),
                "context_keys_a": list(contexts.context_keys_a),
                "context_keys_b": list(contexts.context_keys_b),
                "view_seeds": list(contexts.view_seeds),
                "context_contract": dict(contexts.transform_contract),
            }
        )
    ).hexdigest()
    payload = hashlib.sha256(identity.encode("ascii"))
    for collection in (
        contexts.poses_a,
        contexts.poses_b,
        contexts.joint_valid_a,
        contexts.joint_valid_b,
        contexts.position_indices_a,
        contexts.position_indices_b,
    ):
        for tensor in collection:
            materialized = tensor.detach().cpu().contiguous()
            payload.update(str(materialized.dtype).encode("ascii"))
            payload.update(_canonical_json_bytes(list(materialized.shape)))
            payload.update(materialized.numpy().tobytes(order="C"))
    return identity, payload.hexdigest()


def _finite_ratio(numerator: float, denominator: float) -> float | None:
    if (
        not math.isfinite(numerator)
        or not math.isfinite(denominator)
        or denominator <= 0.0
    ):
        return None
    return numerator / denominator


def evaluate_unified2d_epoch11_label_free_controls(
    encoder: PAMSEncoder,
    objective: ConventionalCycleBackLoss,
    pairs: NativeWindowPairBatch,
    contexts: PairSegmentContexts,
    candidate_config: ConventionalCycleBackConfig,
    contract: FullContextTrainerContract,
    representation: FullContextRepresentationContract,
    lineage: FullContextLineage,
    *,
    expected_pair_identity_sha256: str,
    expected_pair_payload_sha256: str,
) -> dict[str, Any]:
    """Evaluate the typed v4e null adapter; diagnostic contexts never backpropagate."""

    validate_representation_contract_lineage(representation, lineage)
    if (
        representation.representation_family != "v4e_unified2d_coco17_padded33"
        or representation.diagnostic_adapter
        != "unified2d_coco17_joint_nulls_v1"
    ):
        raise ValueError("unified2d diagnostics received another representation type")
    require_real_optimizer_contexts(contexts)
    observed_identity, observed_payload = pair_context_fingerprints(pairs, contexts)
    if (
        observed_identity != expected_pair_identity_sha256
        or observed_payload != expected_pair_payload_sha256
    ):
        raise ValueError("epoch11 evaluation pairs differ from the sealed fixed set")
    was_training = encoder.training
    encoder.eval()
    conditions: dict[str, dict[str, Any]] = {}
    null_contracts: dict[str, Any] = {}
    with torch.inference_mode():
        real_a, real_b = encode_window_pairs(
            encoder,
            pairs,
            contexts,
            batch_size=contract.encoder_context_batch_size,
        )
        conditions["real"] = evaluate_cycleback_embeddings(
            objective,
            real_a,
            real_b,
            pairs,
        )
        conditions["real"]["embedding_temporal_rms"] = {
            "values": temporal_rms_values(real_a, pairs.valid_a)
            + temporal_rms_values(real_b, pairs.valid_b)
        }
        diagnostic_contexts: dict[str, PairSegmentContexts] = {
            "zero_pose": zero_pair_segment_contexts(contexts),
            "within_video_pose_shuffle": (
                independently_permuted_pair_segment_contexts(
                    contexts,
                    pairs,
                    seed=contract.seed,
                )
            ),
            **joint_support_null_segment_contexts(contexts),
        }
        for name, controlled in diagnostic_contexts.items():
            if not controlled.objective_role.startswith("diagnostic_"):
                raise RuntimeError("label-free null lacks a diagnostic context role")
            controlled_a, controlled_b = encode_window_pairs(
                encoder,
                pairs,
                controlled,
                batch_size=contract.encoder_context_batch_size,
            )
            conditions[name] = evaluate_cycleback_embeddings(
                objective,
                controlled_a,
                controlled_b,
                pairs,
            )
            null_contracts[name] = dict(controlled.transform_contract)
        permuted = independently_permuted_pair_segment_positions(
            contexts,
            pairs,
            seed=contract.seed,
        )
        if not permuted.objective_role.startswith("diagnostic_"):
            raise RuntimeError("PE permutation lacks a diagnostic context role")
        permuted_a, permuted_b = encode_window_pairs(
            encoder,
            pairs,
            permuted,
            batch_size=contract.encoder_context_batch_size,
        )
        conditions["permuted_pe"] = evaluate_cycleback_embeddings(
            objective,
            permuted_a,
            permuted_b,
            pairs,
        )
        pe_off = build_encoder(candidate_config, position_encoding_mode="none").to(
            next(encoder.parameters()).device
        )
        pe_off.load_state_dict(encoder.state_dict(), strict=True)
        pe_off.eval()
        pe_off_a, pe_off_b = encode_window_pairs(
            pe_off,
            pairs,
            contexts,
            batch_size=contract.encoder_context_batch_size,
        )
        conditions["pe_off"] = evaluate_cycleback_embeddings(
            objective,
            pe_off_a,
            pe_off_b,
            pairs,
        )
    encoder.train(was_training)
    real_error = float(conditions["real"]["symmetric_position_mse"])
    ratios = {
        name: _finite_ratio(
            float(conditions[name]["symmetric_position_mse"]),
            real_error,
        )
        for name in (
            "zero_pose",
            "within_video_pose_shuffle",
            "mask_flicker",
            "torso_only",
            "alternating_limb_dropout",
            "permuted_pe",
            "pe_off",
        )
    }
    return {
        "schema_version": 1,
        "artifact_type": "pams_cycleback_epoch11_label_free_controls_v1",
        "candidate_id": contract.candidate_id,
        "label_free": True,
        "encoder_eval_mode": True,
        "optimizer_updates_during_evaluation": 0,
        "pair_identity_sha256": observed_identity,
        "pair_payload_sha256": observed_payload,
        "pair_total": pairs.pair_count,
        "view_seeds": list(contexts.view_seeds),
        "conditions": conditions,
        "ratios": ratios,
        "null_context_contracts": null_contracts,
        "pe_controls": {
            "permuted_pe_role": "diagnostic_two_sided_sensitivity_control",
            "pe_off_role": "diagnostic_two_sided_distribution_shift_control",
            "ratios_must_be_bounded_on_both_sides": True,
        },
        "authority_boundaries": _authority_boundaries(),
    }


@dataclass(frozen=True, slots=True)
class Epoch11GateThresholds:
    minimum_real_null_position_error_gap: float
    minimum_valid_anchor_fraction: float
    minimum_temporal_rms_median: float
    minimum_null_to_real_ratio: float
    minimum_pe_ratio: float
    maximum_pe_ratio: float

    def __post_init__(self) -> None:
        values = (
            self.minimum_real_null_position_error_gap,
            self.minimum_valid_anchor_fraction,
            self.minimum_temporal_rms_median,
            self.minimum_null_to_real_ratio,
            self.minimum_pe_ratio,
            self.maximum_pe_ratio,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("epoch11 gate thresholds must be finite")
        if (
            self.minimum_real_null_position_error_gap < 0.0
            or not 0.0 <= self.minimum_valid_anchor_fraction <= 1.0
            or self.minimum_temporal_rms_median < 0.0
            or self.minimum_null_to_real_ratio <= 0.0
            or self.minimum_pe_ratio <= 0.0
            or self.maximum_pe_ratio < self.minimum_pe_ratio
        ):
            raise ValueError("epoch11 gate thresholds are invalid")

    def to_dict(self) -> dict[str, float | int | str | bool]:
        return {
            "schema_version": 1,
            "artifact_type": "pams_cycleback_epoch11_gate_thresholds_v1",
            "minimum_real_null_position_error_gap": (
                self.minimum_real_null_position_error_gap
            ),
            "minimum_valid_anchor_fraction": self.minimum_valid_anchor_fraction,
            "minimum_temporal_rms_median": self.minimum_temporal_rms_median,
            "minimum_null_to_real_ratio": self.minimum_null_to_real_ratio,
            "minimum_pe_ratio": self.minimum_pe_ratio,
            "maximum_pe_ratio": self.maximum_pe_ratio,
            "pe_ratio_bounds_are_two_sided": True,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())


def _criterion(value: float | None, *, relation: str, threshold: float) -> dict[str, Any]:
    passed = False
    if value is not None and math.isfinite(value):
        if relation == "at_least":
            passed = value >= threshold
        elif relation == "at_most":
            passed = value <= threshold
        else:
            raise ValueError("unsupported gate relation")
    return {
        "value": value,
        "relation": relation,
        "threshold": threshold,
        "passed": passed,
    }


def epoch11_gate_decision(
    controls: Mapping[str, Any],
    thresholds: Epoch11GateThresholds,
) -> dict[str, Any]:
    """Return scientific eligibility only; a separate authority must launch 150."""

    if (
        controls.get("artifact_type")
        != "pams_cycleback_epoch11_label_free_controls_v1"
        or controls.get("label_free") is not True
        or controls.get("optimizer_updates_during_evaluation") != 0
        or controls.get("authority_boundaries") != _authority_boundaries()
    ):
        raise ValueError("epoch11 controls contract mismatch")
    conditions = controls.get("conditions")
    ratios = controls.get("ratios")
    if not isinstance(conditions, Mapping) or not isinstance(ratios, Mapping):
        raise ValueError("epoch11 control metrics are missing")
    required_conditions = {
        "real",
        "zero_pose",
        "within_video_pose_shuffle",
        "mask_flicker",
        "torso_only",
        "alternating_limb_dropout",
        "permuted_pe",
        "pe_off",
    }
    if set(conditions) != required_conditions or set(ratios) != (
        required_conditions - {"real"}
    ):
        raise ValueError("epoch11 control condition set mismatch")
    real = conditions["real"]
    if not isinstance(real, Mapping):
        raise ValueError("epoch11 real-control metrics are invalid")
    real_error = float(real["symmetric_position_mse"])
    null_errors = [
        float(conditions[name]["symmetric_position_mse"])
        for name in (
            "zero_pose",
            "within_video_pose_shuffle",
            "mask_flicker",
            "torso_only",
            "alternating_limb_dropout",
        )
    ]
    rms = real.get("embedding_temporal_rms")
    rms_values = rms.get("values") if isinstance(rms, Mapping) else None
    rms_median = (
        None
        if not isinstance(rms_values, list) or not rms_values
        else float(np.median(np.asarray(rms_values, dtype=np.float64)))
    )
    criteria = {
        "real_null_position_error_gap": _criterion(
            min(null_errors) - real_error,
            relation="at_least",
            threshold=thresholds.minimum_real_null_position_error_gap,
        ),
        "valid_anchor_fraction": _criterion(
            float(real["valid_anchor_fraction"]),
            relation="at_least",
            threshold=thresholds.minimum_valid_anchor_fraction,
        ),
        "temporal_rms_median": _criterion(
            rms_median,
            relation="at_least",
            threshold=thresholds.minimum_temporal_rms_median,
        ),
    }
    for name in (
        "zero_pose",
        "within_video_pose_shuffle",
        "mask_flicker",
        "torso_only",
        "alternating_limb_dropout",
    ):
        criteria[f"{name}_to_real_ratio"] = _criterion(
            ratios[name],
            relation="at_least",
            threshold=thresholds.minimum_null_to_real_ratio,
        )
    for name in ("permuted_pe", "pe_off"):
        criteria[f"{name}_to_real_ratio_minimum"] = _criterion(
            ratios[name],
            relation="at_least",
            threshold=thresholds.minimum_pe_ratio,
        )
        criteria[f"{name}_to_real_ratio_maximum"] = _criterion(
            ratios[name],
            relation="at_most",
            threshold=thresholds.maximum_pe_ratio,
        )
    passed = all(value["passed"] for value in criteria.values())
    return {
        "schema_version": 1,
        "artifact_type": "pams_cycleback_epoch11_scientific_gate_decision_v1",
        "thresholds_frozen_before_execution": True,
        "threshold_provenance": (
            "future_independent_synthetic_preregistration_required"
        ),
        "thresholds": thresholds.to_dict(),
        "thresholds_sha256": thresholds.fingerprint,
        "criteria": criteria,
        "overall_pass": passed,
        "scientifically_eligible_for_epoch150_train337": passed,
        "epoch150_train337_continuation_authorized": False,
        "development_evaluation_authorized": False,
        "sealed_evaluation_authorized": False,
        "readout_authorized": False,
        "authorization_required_after_scientific_pass": True,
    }
