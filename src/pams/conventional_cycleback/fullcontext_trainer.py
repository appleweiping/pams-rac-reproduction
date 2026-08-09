"""Fail-closed science core for full-context cycle-back continuation.

This module defines the deterministic training, sampler, checkpoint, and
label-free gate contracts needed after the bounded 256-step mechanism probe.
It deliberately does not expose a command-line entry point.  A newly executed
probe may publish an exact learned-state seed, but that seed grants no training
authority: a future independently sealed integration outcome must bind its
bytes before any formal epoch-11 or epoch-150 continuation can be launched.

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
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
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
    build_encoder,
    encode_window_pairs,
    independent_view_seeds,
    independently_permuted_pair_segment_contexts,
    independently_permuted_pair_segment_positions,
    joint_support_null_segment_contexts,
    require_real_optimizer_contexts,
    stable_file_bytes,
    temporal_rms_values,
    validate_exact_authorized_pair_rows,
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
_FULLCONTEXT_RUNTIME_ATTESTATION = object()
_FULLCONTEXT_PARSED_ATTESTATION = object()


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


def unified2d_fullcontext_representation_contract() -> FullContextRepresentationContract:
    """Return the exact v4e representation contract used by this probe.

    The continuation core remains typed and representation-generic.  This
    helper exists only because the current mechanism producer consumes the
    authorized unified-2D adapter and must embed those exact semantics.
    """

    return FullContextRepresentationContract(
        representation_family="v4e_unified2d_coco17_padded33",
        coordinate_contract="body-centered-uniform-rms-scale-xy-z0-v1",
        pose_joint_count=33,
        support_channel_count=17,
        support_channel_to_pose_joint_indices=tuple(range(17)),
        augmentation_adapter="unified2d_masked_xy_inplane_v1",
        diagnostic_adapter="unified2d_coco17_joint_nulls_v1",
        stable_range_policy="v4e_exact_all_valid_track_stability_ranges_v1",
    )


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
    objective_temperature: float = 0.1
    objective_variance_log_weight: float = 0.001
    objective_variance_floor: float = 1e-6
    mechanism_optimizer_transition: Literal[
        "continue_exact_step256_adamw_state_no_reset"
    ] = "continue_exact_step256_adamw_state_no_reset"
    maximum_videos_per_batch: int = 8
    maximum_context_frames_per_batch: int = 2048
    encoder_context_batch_size: int = 8
    mechanism_optimizer_steps: int = _MECHANISM_OPTIMIZER_STEPS
    mechanism_video_batch_size: int = 8
    mechanism_maximum_pairs_per_step: int = 32
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
            self.objective_temperature != 0.1
            or self.objective_variance_log_weight != 0.001
            or self.objective_variance_floor != 1e-6
        ):
            raise ValueError("full-context objective equation differs from candidate v1")
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
        if self.mechanism_video_batch_size != self.maximum_videos_per_batch:
            raise ValueError("mechanism and continuation video-batch contracts differ")
        _require_integer(
            self.mechanism_maximum_pairs_per_step,
            role="mechanism maximum pairs per step",
            minimum=2,
        )

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
            objective_temperature=config.objective.temperature,
            objective_variance_log_weight=(
                config.objective.variance_log_weight
            ),
            objective_variance_floor=config.objective.variance_floor,
            encoder_context_batch_size=(
                config.mechanism_probe.encoder_segment_context_batch_size
            ),
            mechanism_optimizer_steps=config.mechanism_probe.optimizer_steps,
            mechanism_video_batch_size=config.mechanism_probe.video_batch_size,
            mechanism_maximum_pairs_per_step=(
                config.mechanism_probe.maximum_pairs_per_step
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
                "foreach": False,
                "capturable": False,
                "differentiable": False,
                "fused": False,
            },
            "scheduler": self.scheduler,
            "objective": {
                "name": "symmetric_variance_aware_cycleback_regression",
                "temperature": self.objective_temperature,
                "variance_log_weight": self.objective_variance_log_weight,
                "variance_floor": self.objective_variance_floor,
                "directions": "a_to_b_to_a_and_b_to_a_to_b",
                "normalization": "window_local_absolute_source_offset_div_w_minus_one",
            },
            "mechanism_optimizer_transition": self.mechanism_optimizer_transition,
            "maximum_videos_per_batch": self.maximum_videos_per_batch,
            "maximum_context_frames_per_batch": (
                self.maximum_context_frames_per_batch
            ),
            "encoder_context_batch_size": self.encoder_context_batch_size,
            "mechanism_optimizer_steps": self.mechanism_optimizer_steps,
            "mechanism_sampler": {
                "policy": "ranked_sequences_then_cyclic_video_batch_v1",
                "video_batch_size": self.mechanism_video_batch_size,
                "maximum_pairs_per_step": self.mechanism_maximum_pairs_per_step,
                "completed_optimizer_steps": self.mechanism_optimizer_steps,
                "resume_after_gate": False,
            },
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


def validate_fullcontext_candidate_config(
    config: ConventionalCycleBackConfig,
    contract: FullContextTrainerContract,
) -> None:
    if (
        config.fingerprint != contract.candidate_config_fingerprint
        or config.candidate_id != contract.candidate_id
        or config.window_pair.length_frames != contract.window_frames
        or config.window_pair.hop_frames != contract.hop_frames
        or config.seed != contract.seed
        or config.encoder.maximum_native_length != contract.maximum_native_length
        or config.objective.objective
        != "symmetric_variance_aware_cycleback_regression"
        or config.objective.directions != "a_to_b_to_a_and_b_to_a_to_b"
        or config.objective.temperature != contract.objective_temperature
        or config.objective.variance_log_weight
        != contract.objective_variance_log_weight
        or config.objective.variance_floor != contract.objective_variance_floor
    ):
        raise ValueError("candidate config/objective differs from trainer contract")


def build_fullcontext_objective(
    config: ConventionalCycleBackConfig,
    contract: FullContextTrainerContract,
) -> ConventionalCycleBackLoss:
    """Build the only objective accepted by continuation and its gates."""

    validate_fullcontext_candidate_config(config, contract)
    return ConventionalCycleBackLoss(
        temperature=contract.objective_temperature,
        variance_log_weight=contract.objective_variance_log_weight,
        variance_floor=contract.objective_variance_floor,
    )


def validate_fullcontext_objective(
    objective: ConventionalCycleBackLoss,
    config: ConventionalCycleBackConfig,
    contract: FullContextTrainerContract,
) -> None:
    validate_fullcontext_candidate_config(config, contract)
    if type(objective) is not ConventionalCycleBackLoss or (
        objective.temperature != contract.objective_temperature
        or objective.variance_log_weight != contract.objective_variance_log_weight
        or objective.variance_floor != contract.objective_variance_floor
    ):
        raise ValueError("cycle-back objective parameters differ from frozen config")


@dataclass(frozen=True, slots=True)
class MechanismSeedPredecessorLineage:
    """Acyclic inputs embedded in a freshly produced learned-L seed.

    The mechanism output, run receipt, and checkpoint identity do not exist
    when the checkpoint bytes are serialized.  They are intentionally absent
    here and must be bound later by the sealed mechanism outcome.
    """

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
            (
                "mechanism learned model state",
                self.mechanism_learned_model_state_sha256,
            ),
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
            "mechanism_learned_model_state_sha256": (
                self.mechanism_learned_model_state_sha256
            ),
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class MechanismSeedSamplerState:
    """Exact bounded-probe sample prefix captured with learned L."""

    eligible_video_total: int
    ordered_eligible_video_ids_sha256: str
    video_batch_size: int
    maximum_pairs_per_step: int
    completed_optimizer_steps: int
    consumed_video_batch_chain_sha256: str
    consumed_pair_row_chain_sha256: str
    next_cyclic_start_index: int
    policy: Literal[
        "ranked_sequences_then_cyclic_video_batch_v1"
    ] = "ranked_sequences_then_cyclic_video_batch_v1"
    mechanism_sampler_resume_allowed: Literal[False] = False
    epoch1_sampler_requires_new_sealed_plan: Literal[True] = True

    def __post_init__(self) -> None:
        if (
            self.policy != "ranked_sequences_then_cyclic_video_batch_v1"
            or self.mechanism_sampler_resume_allowed is not False
            or self.epoch1_sampler_requires_new_sealed_plan is not True
        ):
            raise ValueError("mechanism sampler transition policy mismatch")
        _require_integer(
            self.eligible_video_total,
            role="mechanism eligible video total",
            minimum=1,
        )
        _require_integer(
            self.video_batch_size,
            role="mechanism video batch size",
            minimum=1,
        )
        _require_integer(
            self.maximum_pairs_per_step,
            role="mechanism maximum pairs per step",
            minimum=2,
        )
        if self.video_batch_size > self.eligible_video_total:
            raise ValueError("mechanism video batch exceeds eligible dataset")
        if self.completed_optimizer_steps != _MECHANISM_OPTIMIZER_STEPS:
            raise ValueError("mechanism sampler prefix must end at exact step 256")
        _require_integer(
            self.next_cyclic_start_index,
            role="mechanism next cyclic index",
        )
        expected_next = (
            self.completed_optimizer_steps * self.video_batch_size
        ) % self.eligible_video_total
        if self.next_cyclic_start_index != expected_next:
            raise ValueError("mechanism next cyclic index differs from exact prefix")
        for role, value in (
            ("ordered eligible video IDs", self.ordered_eligible_video_ids_sha256),
            ("consumed video batches", self.consumed_video_batch_chain_sha256),
            ("consumed pair rows", self.consumed_pair_row_chain_sha256),
        ):
            _require_sha256(value, role=role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "eligible_video_total": self.eligible_video_total,
            "ordered_eligible_video_ids_sha256": (
                self.ordered_eligible_video_ids_sha256
            ),
            "video_batch_size": self.video_batch_size,
            "maximum_pairs_per_step": self.maximum_pairs_per_step,
            "completed_optimizer_steps": self.completed_optimizer_steps,
            "consumed_video_batch_chain_sha256": (
                self.consumed_video_batch_chain_sha256
            ),
            "consumed_pair_row_chain_sha256": self.consumed_pair_row_chain_sha256,
            "next_cyclic_start_index": self.next_cyclic_start_index,
            "mechanism_sampler_resume_allowed": (
                self.mechanism_sampler_resume_allowed
            ),
            "epoch1_sampler_requires_new_sealed_plan": (
                self.epoch1_sampler_requires_new_sealed_plan
            ),
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
    mechanism_optimizer_state_sha256: str
    mechanism_rng_state_sha256: str
    mechanism_backend_state_sha256: str
    mechanism_sampler_state_sha256: str
    mechanism_view_state_sha256: str
    mechanism_consumed_video_batch_chain_sha256: str
    mechanism_consumed_pair_row_chain_sha256: str

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
            ("mechanism optimizer state", self.mechanism_optimizer_state_sha256),
            ("mechanism RNG state", self.mechanism_rng_state_sha256),
            ("mechanism backend state", self.mechanism_backend_state_sha256),
            ("mechanism sampler state", self.mechanism_sampler_state_sha256),
            ("mechanism view state", self.mechanism_view_state_sha256),
            (
                "mechanism consumed video-batch chain",
                self.mechanism_consumed_video_batch_chain_sha256,
            ),
            (
                "mechanism consumed pair-row chain",
                self.mechanism_consumed_pair_row_chain_sha256,
            ),
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
            "mechanism_optimizer_state_sha256": (
                self.mechanism_optimizer_state_sha256
            ),
            "mechanism_rng_state_sha256": self.mechanism_rng_state_sha256,
            "mechanism_backend_state_sha256": self.mechanism_backend_state_sha256,
            "mechanism_sampler_state_sha256": self.mechanism_sampler_state_sha256,
            "mechanism_view_state_sha256": self.mechanism_view_state_sha256,
            "mechanism_consumed_video_batch_chain_sha256": (
                self.mechanism_consumed_video_batch_chain_sha256
            ),
            "mechanism_consumed_pair_row_chain_sha256": (
                self.mechanism_consumed_pair_row_chain_sha256
            ),
        }

    def mechanism_seed_predecessor(self) -> MechanismSeedPredecessorLineage:
        """Derive the acyclic checkpoint predecessor from final lineage."""

        return MechanismSeedPredecessorLineage(
            source_git_sha=self.source_git_sha,
            source_tree_sha256=self.source_tree_sha256,
            container_image_id=self.container_image_id,
            config_sha256=self.config_sha256,
            config_bytes=self.config_bytes,
            representation_integration_outcome_sha256=(
                self.representation_integration_outcome_sha256
            ),
            representation_integration_outcome_bytes=(
                self.representation_integration_outcome_bytes
            ),
            representation_contract_sha256=self.representation_contract_sha256,
            representation_authorization_sha256=(
                self.representation_authorization_sha256
            ),
            representation_authorization_bytes=(
                self.representation_authorization_bytes
            ),
            mechanism_learned_model_state_sha256=(
                self.mechanism_learned_model_state_sha256
            ),
        )

    def mechanism_seed_predecessor_dict(self) -> dict[str, Any]:
        """Return the acyclic lineage embedded inside the seed checkpoint."""

        return self.mechanism_seed_predecessor().to_dict()

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
    expected_group_keys = {
        "params",
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
    }
    if len(optimizer.param_groups) != 1:
        raise ValueError("full-context optimizer requires exactly one parameter group")
    for group in optimizer.param_groups:
        params = group.get("params")
        if (
            set(group) != expected_group_keys
            or not isinstance(params, list)
            or not params
            or len({id(parameter) for parameter in params}) != len(params)
            or group.get("lr") != contract.learning_rate
            or group.get("weight_decay") != contract.weight_decay
            or tuple(group.get("betas", ())) != (0.9, 0.999)
            or group.get("eps") != 1e-8
            or group.get("amsgrad") is not False
            or group.get("maximize") is not False
            or group.get("foreach") is not False
            or group.get("capturable") is not False
            or group.get("differentiable") is not False
            or group.get("fused") is not False
        ):
            raise ValueError("full-context AdamW parameter group differs from contract")
    parameters = tuple(optimizer.param_groups[0]["params"])
    if optimizer.state:
        if set(optimizer.state) != set(parameters):
            raise ValueError("full-context AdamW runtime state is incomplete")
        for parameter in parameters:
            state = optimizer.state[parameter]
            if not isinstance(state, Mapping) or set(state) != {
                "step",
                "exp_avg",
                "exp_avg_sq",
            }:
                raise ValueError("full-context AdamW runtime state schema mismatch")
            step = state["step"]
            first = state["exp_avg"]
            second = state["exp_avg_sq"]
            if (
                not isinstance(step, Tensor)
                or step.numel() != 1
                or not bool(torch.isfinite(step).all())
                or not isinstance(first, Tensor)
                or not isinstance(second, Tensor)
                or first.shape != parameter.shape
                or second.shape != parameter.shape
                or first.dtype != parameter.dtype
                or second.dtype != parameter.dtype
                or first.device != parameter.device
                or second.device != parameter.device
                or not bool(torch.isfinite(first).all())
                or not bool(torch.isfinite(second).all())
            ):
                raise ValueError("full-context AdamW runtime tensor state mismatch")


def _validate_optimizer_model_binding(
    optimizer: Optimizer,
    model: nn.Module,
) -> None:
    observed = tuple(
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    )
    expected = tuple(model.parameters())
    if len(observed) != len(expected) or any(
        left is not right for left, right in zip(observed, expected, strict=True)
    ):
        raise ValueError("full-context optimizer is not bound to the exact encoder")


def _validate_adamw_state_contract(
    optimizer_state: Mapping[str, Any],
    contract: FullContextTrainerContract,
) -> None:
    if set(optimizer_state) != {"state", "param_groups"}:
        raise ValueError("AdamW checkpoint state schema mismatch")
    groups = optimizer_state["param_groups"]
    expected_group_keys = {
        "params",
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "amsgrad",
        "maximize",
        "foreach",
        "capturable",
        "differentiable",
        "fused",
    }
    if (
        not isinstance(groups, list)
        or len(groups) != 1
        or not isinstance(groups[0], Mapping)
        or set(groups[0]) != expected_group_keys
    ):
        raise ValueError("AdamW checkpoint parameter-group schema mismatch")
    group = groups[0]
    params = group["params"]
    betas = group["betas"]
    if (
        not isinstance(params, list)
        or not params
        or any(isinstance(value, bool) or not isinstance(value, int) for value in params)
        or len(set(params)) != len(params)
        or group["lr"] != contract.learning_rate
        or group["weight_decay"] != contract.weight_decay
        or not isinstance(betas, list | tuple)
        or tuple(betas) != (0.9, 0.999)
        or group["eps"] != 1e-8
        or group["amsgrad"] is not False
        or group["maximize"] is not False
        or group["foreach"] is not False
        or group["capturable"] is not False
        or group["differentiable"] is not False
        or group["fused"] is not False
    ):
        raise ValueError("AdamW checkpoint hyperparameters differ from contract")


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
    parameter_ids = [
        parameter_id
        for group in param_groups
        for parameter_id in group["params"]
    ]
    if (
        len(state) != parameter_total
        or len(set(parameter_ids)) != parameter_total
        or set(state) != set(parameter_ids)
    ):
        raise ValueError("AdamW checkpoint lacks state for an encoder parameter")
    observed: list[int] = []
    for parameter_state in state.values():
        if not isinstance(parameter_state, Mapping) or set(parameter_state) != {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }:
            raise ValueError("AdamW parameter state schema mismatch")
        for moment_name in ("exp_avg", "exp_avg_sq"):
            moment = parameter_state[moment_name]
            if not isinstance(moment, Tensor) or not bool(torch.isfinite(moment).all()):
                raise ValueError("AdamW moment state must be a finite tensor")
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
    optimizer_state_sha256: str
    rng_state_sha256: str
    dataset_fingerprint: str
    active_plan_fingerprint: str
    exact_pair_rows_sha256: str
    representation_contract_sha256: str
    representation_lineage_fingerprint: str
    progress_before_sha256: str
    progress_after_sha256: str

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
        for role, value in (
            ("optimizer-step model state", self.model_state_sha256),
            ("optimizer-step optimizer state", self.optimizer_state_sha256),
            ("optimizer-step RNG state", self.rng_state_sha256),
            ("optimizer-step dataset", self.dataset_fingerprint),
            ("optimizer-step active plan", self.active_plan_fingerprint),
            ("optimizer-step exact pair rows", self.exact_pair_rows_sha256),
            (
                "optimizer-step representation contract",
                self.representation_contract_sha256,
            ),
            (
                "optimizer-step representation lineage",
                self.representation_lineage_fingerprint,
            ),
            ("optimizer-step progress before", self.progress_before_sha256),
            ("optimizer-step progress after", self.progress_after_sha256),
        ):
            _require_sha256(value, role=role)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_type": "pams_cycleback_fullcontext_optimizer_step_v1",
            "global_optimizer_step": self.global_optimizer_step,
            "sampler_epoch": self.sampler_epoch,
            "sampler_batch_index": self.sampler_batch_index,
            "video_ids": list(self.video_ids),
            "pair_total": self.pair_total,
            "unique_view_context_total": self.unique_view_context_total,
            "reused_pair_side_context_references": (
                self.reused_pair_side_context_references
            ),
            "view_seeds": list(self.view_seeds),
            "loss": self.loss,
            "valid_anchor_total": self.valid_anchor_total,
            "possible_anchor_total": self.possible_anchor_total,
            "model_state_sha256": self.model_state_sha256,
            "optimizer_state_sha256": self.optimizer_state_sha256,
            "rng_state_sha256": self.rng_state_sha256,
            "dataset_fingerprint": self.dataset_fingerprint,
            "active_plan_fingerprint": self.active_plan_fingerprint,
            "exact_pair_rows_sha256": self.exact_pair_rows_sha256,
            "representation_contract_sha256": (
                self.representation_contract_sha256
            ),
            "representation_lineage_fingerprint": (
                self.representation_lineage_fingerprint
            ),
            "progress_before_sha256": self.progress_before_sha256,
            "progress_after_sha256": self.progress_after_sha256,
            "null_context_used_by_optimizer": False,
            "window_only_encoder_path_used": False,
        }


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


def _update_canonical_state_digest(digest: Any, value: Any) -> None:
    if isinstance(value, Tensor):
        tensor = value.detach().cpu().contiguous()
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError("optimizer state contains a non-finite tensor")
        digest.update(b"tensor\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_json_bytes(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes(order="C"))
    elif isinstance(value, Mapping):
        digest.update(b"mapping\0")
        keys = sorted(value, key=lambda item: (type(item).__name__, repr(item)))
        for key in keys:
            _update_canonical_state_digest(digest, key)
            _update_canonical_state_digest(digest, value[key])
    elif isinstance(value, tuple):
        digest.update(b"tuple\0")
        for item in value:
            _update_canonical_state_digest(digest, item)
    elif isinstance(value, list):
        digest.update(b"list\0")
        for item in value:
            _update_canonical_state_digest(digest, item)
    elif value is None:
        digest.update(b"none\0")
    elif isinstance(value, bool):
        digest.update(b"bool\0true" if value else b"bool\0false")
    elif isinstance(value, int):
        digest.update(b"int\0")
        digest.update(str(value).encode("ascii"))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("optimizer state contains a non-finite scalar")
        digest.update(b"float\0")
        digest.update(value.hex().encode("ascii"))
    elif isinstance(value, str):
        digest.update(b"str\0")
        digest.update(value.encode("utf-8"))
    else:
        raise ValueError(f"optimizer state contains unsupported {type(value).__name__}")
    digest.update(b"\0end\0")


def optimizer_state_sha256(
    optimizer_or_state: Optimizer | Mapping[str, Any],
) -> str:
    state = (
        optimizer_or_state.state_dict()
        if isinstance(optimizer_or_state, Optimizer)
        else optimizer_or_state
    )
    if not isinstance(state, Mapping) or set(state) != {"state", "param_groups"}:
        raise ValueError("optimizer state digest requires an exact state_dict")
    digest = hashlib.sha256()
    _update_canonical_state_digest(digest, state)
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


def _optimize_real_pair_contexts(
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


def training_units_fingerprint(
    units: Sequence[FullContextTrainingUnit],
) -> str:
    normalized = tuple(units)
    if not normalized or len({unit.video_id for unit in normalized}) != len(normalized):
        raise ValueError("training dataset units must be non-empty unique")
    if normalized != tuple(sorted(normalized, key=lambda unit: unit.video_id)):
        raise ValueError("training dataset units must use canonical video-ID order")
    return _canonical_sha256(
        {
            "schema_version": 1,
            "policy": "exact_authorized_units_canonical_opaque_video_order",
            "units": [unit.to_dict() for unit in normalized],
        }
    )


@dataclass(frozen=True, slots=True)
class MechanismSeedLoadReceipt:
    checkpoint_sha256: str
    checkpoint_bytes: int
    learned_model_state_sha256: str
    optimizer_state_sha256: str
    optimizer_step: Literal[256]
    mechanism_sampler_state_sha256: str
    mechanism_view_state_sha256: str
    mechanism_consumed_video_batch_chain_sha256: str
    mechanism_consumed_pair_row_chain_sha256: str
    rng_state_sha256: str
    backend_state_sha256: str
    trainer_contract_fingerprint: str
    representation_contract_sha256: str
    representation_lineage_fingerprint: str
    loaded_as_epoch1_start: Literal[True] = True
    direct_epoch150_start_authorized: Literal[False] = False

    def __post_init__(self) -> None:
        for role, value in (
            ("mechanism seed checkpoint", self.checkpoint_sha256),
            ("mechanism learned model", self.learned_model_state_sha256),
            ("mechanism optimizer state", self.optimizer_state_sha256),
            ("mechanism sampler state", self.mechanism_sampler_state_sha256),
            ("mechanism view state", self.mechanism_view_state_sha256),
            (
                "mechanism consumed video-batch chain",
                self.mechanism_consumed_video_batch_chain_sha256,
            ),
            (
                "mechanism consumed pair-row chain",
                self.mechanism_consumed_pair_row_chain_sha256,
            ),
            ("mechanism RNG state", self.rng_state_sha256),
            ("mechanism backend state", self.backend_state_sha256),
            ("mechanism trainer contract", self.trainer_contract_fingerprint),
            ("mechanism representation contract", self.representation_contract_sha256),
            ("mechanism representation lineage", self.representation_lineage_fingerprint),
        ):
            _require_sha256(value, role=role)
        _require_integer(
            self.checkpoint_bytes,
            role="mechanism seed checkpoint bytes",
            minimum=1,
        )
        if self.optimizer_step != _MECHANISM_OPTIMIZER_STEPS:
            raise ValueError("learned-L receipt must be the exact step-256 state")
        if (
            self.loaded_as_epoch1_start is not True
            or self.direct_epoch150_start_authorized is not False
        ):
            raise ValueError("learned-L receipt has invalid continuation authority")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_type": "pams_cycleback_mechanism_seed_load_receipt_v1",
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_bytes": self.checkpoint_bytes,
            "learned_model_state_sha256": self.learned_model_state_sha256,
            "optimizer_state_sha256": self.optimizer_state_sha256,
            "optimizer_step": self.optimizer_step,
            "mechanism_sampler_state_sha256": (
                self.mechanism_sampler_state_sha256
            ),
            "mechanism_view_state_sha256": self.mechanism_view_state_sha256,
            "mechanism_consumed_video_batch_chain_sha256": (
                self.mechanism_consumed_video_batch_chain_sha256
            ),
            "mechanism_consumed_pair_row_chain_sha256": (
                self.mechanism_consumed_pair_row_chain_sha256
            ),
            "rng_state_sha256": self.rng_state_sha256,
            "backend_state_sha256": self.backend_state_sha256,
            "trainer_contract_fingerprint": self.trainer_contract_fingerprint,
            "representation_contract_sha256": self.representation_contract_sha256,
            "representation_lineage_fingerprint": (
                self.representation_lineage_fingerprint
            ),
            "loaded_as_epoch1_start": self.loaded_as_epoch1_start,
            "direct_epoch150_start_authorized": (
                self.direct_epoch150_start_authorized
            ),
            "checkpoint_self_authorizes_training": False,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class CompletedEpochPlan:
    epoch: int
    plan_fingerprint: str
    batch_total: int
    cumulative_batch_total: int

    def __post_init__(self) -> None:
        _require_integer(self.epoch, role="completed plan epoch", minimum=1)
        _require_sha256(self.plan_fingerprint, role="completed plan fingerprint")
        _require_integer(self.batch_total, role="completed plan batch total", minimum=1)
        _require_integer(
            self.cumulative_batch_total,
            role="completed cumulative batch total",
            minimum=self.batch_total,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "plan_fingerprint": self.plan_fingerprint,
            "batch_total": self.batch_total,
            "cumulative_batch_total": self.cumulative_batch_total,
        }


@dataclass(frozen=True, slots=True)
class Epoch150TransitionBindings:
    candidate_id: CandidateId
    epoch11_checkpoint_sha256: str
    epoch11_checkpoint_bytes: int
    epoch11_model_state_sha256: str
    epoch11_optimizer_state_sha256: str
    epoch11_progress_sha256: str
    trainer_contract_fingerprint: str
    representation_contract_sha256: str
    representation_lineage_fingerprint: str
    fixed_pair_identity_sha256: str
    fixed_pair_payload_sha256: str
    epoch11_gate_outcome_sha256: str
    epoch11_gate_outcome_bytes: int
    threshold_receipt_sha256: str
    threshold_receipt_bytes: int
    epoch150_launch_authorization_sha256: str
    epoch150_launch_authorization_bytes: int
    scientific_gate_passed: Literal[True] = True
    epoch150_train337_continuation_authorized: Literal[True] = True
    development_evaluation_authorized: Literal[False] = False
    sealed_evaluation_authorized: Literal[False] = False
    readout_authorized: Literal[False] = False

    def __post_init__(self) -> None:
        if self.candidate_id not in _CANDIDATE_GEOMETRY:
            raise ValueError("epoch150 transition candidate is unsupported")
        for role, value in (
            ("epoch11 checkpoint", self.epoch11_checkpoint_sha256),
            ("epoch11 model", self.epoch11_model_state_sha256),
            ("epoch11 optimizer", self.epoch11_optimizer_state_sha256),
            ("epoch11 progress", self.epoch11_progress_sha256),
            ("trainer contract", self.trainer_contract_fingerprint),
            ("representation contract", self.representation_contract_sha256),
            ("representation lineage", self.representation_lineage_fingerprint),
            ("fixed pair identity", self.fixed_pair_identity_sha256),
            ("fixed pair payload", self.fixed_pair_payload_sha256),
            ("epoch11 gate outcome", self.epoch11_gate_outcome_sha256),
            ("epoch11 threshold receipt", self.threshold_receipt_sha256),
            ("epoch150 launch authorization", self.epoch150_launch_authorization_sha256),
        ):
            _require_sha256(value, role=role)
        for role, value in (
            ("epoch11 checkpoint bytes", self.epoch11_checkpoint_bytes),
            ("epoch11 gate outcome bytes", self.epoch11_gate_outcome_bytes),
            ("epoch11 threshold receipt bytes", self.threshold_receipt_bytes),
            (
                "epoch150 launch authorization bytes",
                self.epoch150_launch_authorization_bytes,
            ),
        ):
            _require_integer(value, role=role, minimum=1)
        if (
            self.scientific_gate_passed is not True
            or self.epoch150_train337_continuation_authorized is not True
            or self.development_evaluation_authorized is not False
            or self.sealed_evaluation_authorized is not False
            or self.readout_authorized is not False
        ):
            raise ValueError("epoch150 transition authority flags are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_type": "pams_cycleback_epoch150_transition_bindings_v1",
            "candidate_id": self.candidate_id,
            "epoch11_checkpoint_sha256": self.epoch11_checkpoint_sha256,
            "epoch11_checkpoint_bytes": self.epoch11_checkpoint_bytes,
            "epoch11_model_state_sha256": self.epoch11_model_state_sha256,
            "epoch11_optimizer_state_sha256": (
                self.epoch11_optimizer_state_sha256
            ),
            "epoch11_progress_sha256": self.epoch11_progress_sha256,
            "trainer_contract_fingerprint": self.trainer_contract_fingerprint,
            "representation_contract_sha256": self.representation_contract_sha256,
            "representation_lineage_fingerprint": (
                self.representation_lineage_fingerprint
            ),
            "fixed_pair_identity_sha256": self.fixed_pair_identity_sha256,
            "fixed_pair_payload_sha256": self.fixed_pair_payload_sha256,
            "epoch11_gate_outcome_sha256": self.epoch11_gate_outcome_sha256,
            "epoch11_gate_outcome_bytes": self.epoch11_gate_outcome_bytes,
            "threshold_receipt_sha256": self.threshold_receipt_sha256,
            "threshold_receipt_bytes": self.threshold_receipt_bytes,
            "epoch150_launch_authorization_sha256": (
                self.epoch150_launch_authorization_sha256
            ),
            "epoch150_launch_authorization_bytes": (
                self.epoch150_launch_authorization_bytes
            ),
            "scientific_gate_passed": self.scientific_gate_passed,
            "epoch150_train337_continuation_authorized": (
                self.epoch150_train337_continuation_authorized
            ),
            "development_evaluation_authorized": (
                self.development_evaluation_authorized
            ),
            "sealed_evaluation_authorized": self.sealed_evaluation_authorized,
            "readout_authorized": self.readout_authorized,
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class FullContextProgress:
    phase: TrainingPhase
    target_epoch: int
    completed_epochs: int
    continuation_optimizer_steps: int
    units: tuple[FullContextTrainingUnit, ...]
    dataset_fingerprint: str
    mechanism_seed_load_receipt: MechanismSeedLoadReceipt
    current_model_state_sha256: str
    current_optimizer_state_sha256: str
    current_rng_state_sha256: str
    completed_plan_prefix: tuple[CompletedEpochPlan, ...]
    active_plan: LengthBucketPlan | None
    sampler_cursor: int
    epoch11_prefix_checkpoint_sha256: str | None = None
    epoch11_prefix_checkpoint_bytes: int | None = None
    epoch150_transition: Epoch150TransitionBindings | None = None
    _runtime_attestation: object = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        expected_target = {
            "epoch11": _EPOCH11_TARGET,
            "epoch150": _EPOCH150_TARGET,
        }.get(self.phase)
        if expected_target is None or self.target_epoch != expected_target:
            raise ValueError("full-context phase target mismatch")
        if self.dataset_fingerprint != training_units_fingerprint(self.units):
            raise ValueError("full-context dataset fingerprint mismatch")
        _require_sha256(self.current_model_state_sha256, role="current model state")
        _require_sha256(
            self.current_optimizer_state_sha256,
            role="current optimizer state",
        )
        _require_sha256(self.current_rng_state_sha256, role="current RNG state")
        if (
            self._runtime_attestation is not _FULLCONTEXT_RUNTIME_ATTESTATION
            and self._runtime_attestation is not _FULLCONTEXT_PARSED_ATTESTATION
        ):
            raise ValueError(
                "full-context progress must originate from a validated checkpoint load"
            )
        minimum_completed = 0 if self.phase == "epoch11" else _EPOCH11_TARGET
        if not minimum_completed <= self.completed_epochs <= self.target_epoch:
            raise ValueError("completed epoch count lies outside the phase")
        if len(self.completed_plan_prefix) != self.completed_epochs:
            raise ValueError("completed epoch count differs from plan prefix")
        cumulative = 0
        for expected_epoch, record in enumerate(self.completed_plan_prefix, start=1):
            cumulative += record.batch_total
            if (
                record.epoch != expected_epoch
                or record.cumulative_batch_total != cumulative
            ):
                raise ValueError("completed sampler plan prefix is non-canonical")
        _require_integer(self.sampler_cursor, role="sampler cursor")
        expected_steps = cumulative + self.sampler_cursor
        if self.continuation_optimizer_steps != expected_steps:
            raise ValueError("optimizer steps differ from plan-prefix/cursor total")
        if self.phase == "epoch11":
            if (
                self.epoch11_prefix_checkpoint_sha256 is not None
                or self.epoch11_prefix_checkpoint_bytes is not None
                or self.epoch150_transition is not None
            ):
                raise ValueError("epoch11 phase cannot carry epoch150 authority")
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
            if self.epoch150_transition is None or (
                self.epoch150_transition.epoch11_checkpoint_sha256
                != self.epoch11_prefix_checkpoint_sha256
                or self.epoch150_transition.epoch11_checkpoint_bytes
                != self.epoch11_prefix_checkpoint_bytes
            ):
                raise ValueError("epoch150 transition differs from epoch11 prefix")
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
            if tuple(self.active_plan.units) != self.units:
                raise ValueError("active sampler plan changes permanent dataset units")

    @property
    def global_optimizer_step(self) -> int:
        return _MECHANISM_OPTIMIZER_STEPS + self.continuation_optimizer_steps

    @property
    def next_augmentation_step(self) -> int:
        return self.global_optimizer_step + 1

    @property
    def plan_dataset_prefix_sha256(self) -> str:
        return _canonical_sha256(
            {
                "mechanism_seed_load_receipt_sha256": (
                    self.mechanism_seed_load_receipt.fingerprint
                ),
                "dataset_fingerprint": self.dataset_fingerprint,
                "completed_plan_prefix": [
                    record.to_dict() for record in self.completed_plan_prefix
                ],
                "active_plan_fingerprint": (
                    None if self.active_plan is None else self.active_plan.fingerprint
                ),
                "sampler_cursor": self.sampler_cursor,
                "continuation_optimizer_steps": self.continuation_optimizer_steps,
                "current_model_state_sha256": self.current_model_state_sha256,
                "current_optimizer_state_sha256": (
                    self.current_optimizer_state_sha256
                ),
                "current_rng_state_sha256": self.current_rng_state_sha256,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "target_epoch": self.target_epoch,
            "completed_epochs": self.completed_epochs,
            "continuation_optimizer_steps": self.continuation_optimizer_steps,
            "global_optimizer_step": self.global_optimizer_step,
            "units": [unit.to_dict() for unit in self.units],
            "dataset_fingerprint": self.dataset_fingerprint,
            "mechanism_seed_load_receipt": (
                self.mechanism_seed_load_receipt.to_dict()
            ),
            "mechanism_seed_load_receipt_sha256": (
                self.mechanism_seed_load_receipt.fingerprint
            ),
            "current_model_state_sha256": self.current_model_state_sha256,
            "current_optimizer_state_sha256": (
                self.current_optimizer_state_sha256
            ),
            "current_rng_state_sha256": self.current_rng_state_sha256,
            "completed_plan_prefix": [
                record.to_dict() for record in self.completed_plan_prefix
            ],
            "active_plan": (
                None if self.active_plan is None else self.active_plan.to_dict()
            ),
            "active_plan_fingerprint": (
                None if self.active_plan is None else self.active_plan.fingerprint
            ),
            "sampler_cursor": self.sampler_cursor,
            "plan_dataset_prefix_sha256": self.plan_dataset_prefix_sha256,
            "epoch11_prefix_checkpoint_sha256": (
                self.epoch11_prefix_checkpoint_sha256
            ),
            "epoch11_prefix_checkpoint_bytes": (
                self.epoch11_prefix_checkpoint_bytes
            ),
            "epoch150_transition": (
                None
                if self.epoch150_transition is None
                else self.epoch150_transition.to_dict()
            ),
            "epoch150_transition_sha256": (
                None
                if self.epoch150_transition is None
                else self.epoch150_transition.fingerprint
            ),
        }

    @property
    def fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict())


def _require_live_fullcontext_progress(progress: FullContextProgress) -> None:
    if progress._runtime_attestation is not _FULLCONTEXT_RUNTIME_ATTESTATION:
        raise ValueError(
            "full-context progress is semantic evidence, not a loaded runtime state"
        )


def validate_fullcontext_progress_schedule(
    progress: FullContextProgress,
    contract: FullContextTrainerContract,
    lineage: FullContextLineage,
) -> None:
    receipt = progress.mechanism_seed_load_receipt
    if (
        receipt.checkpoint_sha256 != lineage.mechanism_seed_checkpoint_sha256
        or receipt.checkpoint_bytes != lineage.mechanism_seed_checkpoint_bytes
        or receipt.learned_model_state_sha256
        != lineage.mechanism_learned_model_state_sha256
        or receipt.optimizer_state_sha256
        != lineage.mechanism_optimizer_state_sha256
        or receipt.mechanism_sampler_state_sha256
        != lineage.mechanism_sampler_state_sha256
        or receipt.mechanism_view_state_sha256
        != lineage.mechanism_view_state_sha256
        or receipt.mechanism_consumed_video_batch_chain_sha256
        != lineage.mechanism_consumed_video_batch_chain_sha256
        or receipt.mechanism_consumed_pair_row_chain_sha256
        != lineage.mechanism_consumed_pair_row_chain_sha256
        or receipt.rng_state_sha256 != lineage.mechanism_rng_state_sha256
        or receipt.backend_state_sha256 != lineage.mechanism_backend_state_sha256
        or receipt.trainer_contract_fingerprint != contract.fingerprint
        or receipt.representation_contract_sha256
        != lineage.representation_contract_sha256
        or receipt.representation_lineage_fingerprint != lineage.fingerprint
    ):
        raise ValueError("learned-L receipt differs from continuation lineage")
    if progress.continuation_optimizer_steps == 0 and (
        progress.current_model_state_sha256
        != progress.mechanism_seed_load_receipt.learned_model_state_sha256
        or progress.current_optimizer_state_sha256
        != progress.mechanism_seed_load_receipt.optimizer_state_sha256
        or progress.current_rng_state_sha256
        != progress.mechanism_seed_load_receipt.rng_state_sha256
    ):
        raise ValueError("epoch-one progress differs from exact learned-L state")
    if (
        progress.phase == "epoch150"
        and progress.completed_epochs == _EPOCH11_TARGET
        and progress.sampler_cursor == 0
    ):
        transition = progress.epoch150_transition
        if transition is None or (
            transition.epoch11_model_state_sha256
            != progress.current_model_state_sha256
            or transition.epoch11_optimizer_state_sha256
            != progress.current_optimizer_state_sha256
        ):
            raise ValueError("epoch150 start differs from exact epoch11 state")
    for record in progress.completed_plan_prefix:
        expected = build_length_bucket_plan(
            progress.units,
            contract,
            lineage,
            epoch=record.epoch,
        )
        if (
            record.plan_fingerprint != expected.fingerprint
            or record.batch_total != len(expected.batches)
        ):
            raise ValueError("completed sampler plan prefix differs from replay")
    if progress.active_plan is not None:
        expected_active = build_length_bucket_plan(
            progress.units,
            contract,
            lineage,
            epoch=progress.completed_epochs + 1,
        )
        if progress.active_plan.to_dict() != expected_active.to_dict():
            raise ValueError("active sampler plan differs from deterministic replay")


def _initial_epoch11_progress_from_loaded_seed(
    plan: LengthBucketPlan,
    *,
    mechanism_seed_load_receipt: MechanismSeedLoadReceipt,
) -> FullContextProgress:
    if plan.epoch != 1:
        raise ValueError("epoch11 continuation must begin with epoch-one sampler plan")
    if (
        mechanism_seed_load_receipt.trainer_contract_fingerprint
        != plan.trainer_contract_fingerprint
        or mechanism_seed_load_receipt.representation_lineage_fingerprint
        != plan.representation_lineage_fingerprint
    ):
        raise ValueError("epoch-one plan differs from learned-L load receipt")
    return FullContextProgress(
        phase="epoch11",
        target_epoch=_EPOCH11_TARGET,
        completed_epochs=0,
        continuation_optimizer_steps=0,
        units=plan.units,
        dataset_fingerprint=training_units_fingerprint(plan.units),
        mechanism_seed_load_receipt=mechanism_seed_load_receipt,
        current_model_state_sha256=(
            mechanism_seed_load_receipt.learned_model_state_sha256
        ),
        current_optimizer_state_sha256=(
            mechanism_seed_load_receipt.optimizer_state_sha256
        ),
        current_rng_state_sha256=mechanism_seed_load_receipt.rng_state_sha256,
        completed_plan_prefix=(),
        active_plan=plan,
        sampler_cursor=0,
        _runtime_attestation=_FULLCONTEXT_RUNTIME_ATTESTATION,
    )


def _initial_epoch150_progress_from_loaded_epoch11(
    plan: LengthBucketPlan,
    *,
    completed_epoch11_progress: FullContextProgress,
    transition: Epoch150TransitionBindings,
) -> FullContextProgress:
    _require_live_fullcontext_progress(completed_epoch11_progress)
    if plan.epoch != _EPOCH11_TARGET + 1:
        raise ValueError("epoch150 continuation must begin with epoch-twelve plan")
    if (
        completed_epoch11_progress.phase != "epoch11"
        or completed_epoch11_progress.completed_epochs != _EPOCH11_TARGET
        or completed_epoch11_progress.active_plan is not None
        or tuple(plan.units) != completed_epoch11_progress.units
        or transition.epoch11_progress_sha256
        != completed_epoch11_progress.fingerprint
        or transition.epoch11_model_state_sha256
        != completed_epoch11_progress.current_model_state_sha256
        or transition.epoch11_optimizer_state_sha256
        != completed_epoch11_progress.current_optimizer_state_sha256
    ):
        raise ValueError("epoch150 does not extend the exact completed epoch11 state")
    return FullContextProgress(
        phase="epoch150",
        target_epoch=_EPOCH150_TARGET,
        completed_epochs=_EPOCH11_TARGET,
        continuation_optimizer_steps=(
            completed_epoch11_progress.continuation_optimizer_steps
        ),
        units=completed_epoch11_progress.units,
        dataset_fingerprint=completed_epoch11_progress.dataset_fingerprint,
        mechanism_seed_load_receipt=(
            completed_epoch11_progress.mechanism_seed_load_receipt
        ),
        current_model_state_sha256=(
            completed_epoch11_progress.current_model_state_sha256
        ),
        current_optimizer_state_sha256=(
            completed_epoch11_progress.current_optimizer_state_sha256
        ),
        current_rng_state_sha256=(
            completed_epoch11_progress.current_rng_state_sha256
        ),
        completed_plan_prefix=completed_epoch11_progress.completed_plan_prefix,
        active_plan=plan,
        sampler_cursor=0,
        epoch11_prefix_checkpoint_sha256=transition.epoch11_checkpoint_sha256,
        epoch11_prefix_checkpoint_bytes=transition.epoch11_checkpoint_bytes,
        epoch150_transition=transition,
        _runtime_attestation=_FULLCONTEXT_RUNTIME_ATTESTATION,
    )


def _advance_fullcontext_progress(
    progress: FullContextProgress,
    contract: FullContextTrainerContract,
    lineage: FullContextLineage,
    *,
    next_model_state_sha256: str,
    next_optimizer_state_sha256: str,
    next_rng_state_sha256: str,
) -> FullContextProgress:
    """Advance exactly the active cursor; next plans are replayed internally."""

    _require_live_fullcontext_progress(progress)
    validate_fullcontext_progress_schedule(progress, contract, lineage)
    _require_sha256(next_model_state_sha256, role="advanced model state")
    _require_sha256(next_optimizer_state_sha256, role="advanced optimizer state")
    _require_sha256(next_rng_state_sha256, role="advanced RNG state")
    if (
        next_model_state_sha256 == progress.current_model_state_sha256
        and next_optimizer_state_sha256 == progress.current_optimizer_state_sha256
    ):
        raise ValueError("optimizer progress cannot advance without state change")
    if progress.active_plan is None:
        raise ValueError("completed full-context phase cannot advance")
    next_cursor = progress.sampler_cursor + 1
    next_steps = progress.continuation_optimizer_steps + 1
    if next_cursor < len(progress.active_plan.batches):
        return replace(
            progress,
            continuation_optimizer_steps=next_steps,
            current_model_state_sha256=next_model_state_sha256,
            current_optimizer_state_sha256=next_optimizer_state_sha256,
            current_rng_state_sha256=next_rng_state_sha256,
            sampler_cursor=next_cursor,
        )
    previous_cumulative = (
        0
        if not progress.completed_plan_prefix
        else progress.completed_plan_prefix[-1].cumulative_batch_total
    )
    completed_record = CompletedEpochPlan(
        epoch=progress.active_plan.epoch,
        plan_fingerprint=progress.active_plan.fingerprint,
        batch_total=len(progress.active_plan.batches),
        cumulative_batch_total=(
            previous_cumulative + len(progress.active_plan.batches)
        ),
    )
    completed = progress.completed_epochs + 1
    prefix = (*progress.completed_plan_prefix, completed_record)
    if completed == progress.target_epoch:
        return replace(
            progress,
            completed_epochs=completed,
            continuation_optimizer_steps=next_steps,
            current_model_state_sha256=next_model_state_sha256,
            current_optimizer_state_sha256=next_optimizer_state_sha256,
            current_rng_state_sha256=next_rng_state_sha256,
            completed_plan_prefix=prefix,
            active_plan=None,
            sampler_cursor=0,
        )
    next_plan = build_length_bucket_plan(
        progress.units,
        contract,
        lineage,
        epoch=completed + 1,
    )
    return replace(
        progress,
        completed_epochs=completed,
        continuation_optimizer_steps=next_steps,
        current_model_state_sha256=next_model_state_sha256,
        current_optimizer_state_sha256=next_optimizer_state_sha256,
        current_rng_state_sha256=next_rng_state_sha256,
        completed_plan_prefix=prefix,
        active_plan=next_plan,
        sampler_cursor=0,
    )


@dataclass(frozen=True, slots=True)
class FullContextStepResult:
    evidence: FullContextOptimizerStep
    progress: FullContextProgress

    def __post_init__(self) -> None:
        if (
            self.evidence.global_optimizer_step
            != self.progress.global_optimizer_step
            or self.evidence.progress_after_sha256 != self.progress.fingerprint
        ):
            raise ValueError("optimizer evidence does not bind the advanced progress")


def _validate_prepared_real_context_rows(
    pairs: NativeWindowPairBatch,
    contexts: PairSegmentContexts,
    *,
    support_channel_count: int,
) -> str:
    require_real_optimizer_contexts(contexts)
    expected_contract_keys = {
        "policy",
        "attention_context_boundary",
        "association_reset_is_not_a_wider_attention_context",
        "absolute_native_position_indices",
        "window_only_encoder_path_used",
        "pair_side_context_references",
        "unique_view_context_total",
        "reused_pair_side_context_references",
        "same_view_same_range_encoded_once",
    }
    unique_keys = len(set(contexts.context_keys_a)) + len(
        set(contexts.context_keys_b)
    )
    expected_contract = {
        "policy": "full_authorized_all_valid_stable_range_then_slice_embeddings",
        "attention_context_boundary": "eligible_frame_range_start_stop",
        "association_reset_is_not_a_wider_attention_context": True,
        "absolute_native_position_indices": True,
        "window_only_encoder_path_used": False,
        "pair_side_context_references": 2 * pairs.pair_count,
        "unique_view_context_total": unique_keys,
        "reused_pair_side_context_references": 2 * pairs.pair_count - unique_keys,
        "same_view_same_range_encoded_once": True,
    }
    if (
        contexts.video_ids != pairs.video_ids
        or set(contexts.transform_contract) != expected_contract_keys
        or dict(contexts.transform_contract) != expected_contract
    ):
        raise ValueError("prepared real contexts violate the exact encode-once contract")
    for row, video_id in enumerate(pairs.video_ids):
        start = int(pairs.segment_starts[row])
        stop = int(pairs.segment_ends[row])
        expected_positions = torch.arange(
            start,
            stop,
            dtype=torch.long,
            device=pairs.poses_a.device,
        )
        opaque = hashlib.sha256(video_id.encode("utf-8")).hexdigest()
        if (
            contexts.poses_a[row].shape[0] != stop - start
            or contexts.poses_b[row].shape[0] != stop - start
            or contexts.poses_a[row].device != pairs.poses_a.device
            or contexts.poses_b[row].device != pairs.poses_b.device
            or contexts.context_keys_a[row] != f"real:a:{opaque}:{start}:{stop}"
            or contexts.context_keys_b[row] != f"real:b:{opaque}:{start}:{stop}"
            or not torch.equal(contexts.position_indices_a[row], expected_positions)
            or not torch.equal(contexts.position_indices_b[row], expected_positions)
        ):
            raise ValueError("prepared context identity/absolute positions are not exact")
        local_a = pairs.source_indices_a[row] - start
        local_b = pairs.source_indices_b[row] - start
        if (
            contexts.joint_valid_a[row].shape[1] != support_channel_count
            or contexts.joint_valid_b[row].shape[1] != support_channel_count
            or not torch.equal(contexts.poses_a[row][local_a], pairs.poses_a[row])
            or not torch.equal(contexts.poses_b[row][local_b], pairs.poses_b[row])
            or not torch.equal(
                contexts.joint_valid_a[row][local_a],
                pairs.joint_valid_a[row],
            )
            or not torch.equal(
                contexts.joint_valid_b[row][local_b],
                pairs.joint_valid_b[row],
            )
        ):
            raise ValueError("prepared pair windows differ from their shared contexts")
    identity_sha256, payload_sha256 = pair_context_fingerprints(pairs, contexts)
    return _canonical_sha256(
        {
            "schema_version": 1,
            "pair_context_identity_sha256": identity_sha256,
            "pair_context_payload_sha256": payload_sha256,
            "support_channel_count": support_channel_count,
        }
    )


def run_fullcontext_optimizer_step(
    encoder: PAMSEncoder,
    optimizer: Optimizer,
    pairs: NativeWindowPairBatch,
    contexts: PairSegmentContexts,
    authorized_ranges_by_video: Mapping[str, Sequence[tuple[int, int]]],
    pair_eligibility: PairEligibility,
    candidate_config: ConventionalCycleBackConfig,
    contract: FullContextTrainerContract,
    representation: FullContextRepresentationContract,
    lineage: FullContextLineage,
    progress: FullContextProgress,
) -> FullContextStepResult:
    """Execute the sole optimizer step API from the active progress cursor.

    An independently authorized representation adapter constructs ``pairs``
    and ``contexts``.  This core binds their exact geometry, support-channel
    schema, view seeds, real-only role, and full-range encode-once semantics;
    it never assumes COCO17 or MediaPipe33 support semantics itself.
    """

    _require_live_fullcontext_progress(progress)
    validate_representation_contract_lineage(representation, lineage)
    validate_fullcontext_candidate_config(candidate_config, contract)
    validate_fullcontext_progress_schedule(progress, contract, lineage)
    validate_frozen_fullcontext_backend_state(capture_fullcontext_backend_state())
    plan = progress.active_plan
    if plan is None:
        raise ValueError("completed progress cannot run another optimizer step")
    validate_train337_plan(plan, contract, pair_eligibility)
    validate_fullcontext_optimizer(optimizer, contract)
    _validate_optimizer_model_binding(optimizer, encoder)
    observed_model_state_sha256 = model_state_sha256(encoder)
    observed_optimizer_state_sha256 = optimizer_state_sha256(optimizer)
    observed_rng_state_sha256 = _rng_state_sha256(capture_fullcontext_rng_state())
    if (
        observed_model_state_sha256 != progress.current_model_state_sha256
        or observed_optimizer_state_sha256
        != progress.current_optimizer_state_sha256
        or observed_rng_state_sha256 != progress.current_rng_state_sha256
    ):
        raise ValueError("model/optimizer/RNG bytes differ from active progress")
    _validate_adamw_state_step(
        optimizer.state_dict(),
        expected_step=progress.global_optimizer_step,
    )
    batch_index = progress.sampler_cursor
    global_optimizer_step = progress.next_augmentation_step
    selected_ids = plan.batches[batch_index].video_ids
    selected_ranges = {
        video_id: tuple(authorized_ranges_by_video[video_id])
        for video_id in selected_ids
    }
    unit_by_id = {unit.video_id: unit for unit in progress.units}
    if any(
        selected_ranges[video_id] != unit_by_id[video_id].authorized_ranges
        for video_id in selected_ids
    ):
        raise ValueError("prepared stable ranges differ from permanent training units")
    support_channels = representation.support_channel_count
    if (
        pairs.joint_valid_a.shape[2] != support_channels
        or pairs.joint_valid_b.shape[2] != support_channels
        or any(
            mask.shape[1] != support_channels
            for mask in (
                *contexts.joint_valid_a,
                *contexts.joint_valid_b,
            )
        )
    ):
        raise ValueError("prepared batch differs from typed support-channel schema")
    exact_pair_rows_sha256 = validate_exact_authorized_pair_rows(
        pairs,
        ordered_video_ids=selected_ids,
        native_lengths={
            video_id: pair_eligibility.native_lengths[video_id]
            for video_id in selected_ids
        },
        authorized_ranges_by_video=selected_ranges,
        base_valid_starts_by_video={
            video_id: pair_eligibility.base_valid_starts_by_variant[
                contract.candidate_id
            ][video_id]
            for video_id in selected_ids
        },
        eligible_starts_by_video={
            video_id: unit_by_id[video_id].authorized_pair_starts
            for video_id in selected_ids
        },
        window_frames=contract.window_frames,
        hop_frames=contract.hop_frames,
    )
    for row, video_id in enumerate(pairs.video_ids):
        emitted_range = (
            int(pairs.segment_starts[row]),
            int(pairs.segment_ends[row]),
        )
        if emitted_range not in unit_by_id[video_id].authorized_ranges:
            raise RuntimeError("runtime pair references an unconsumed stable range")
    exact_context_rows_sha256 = _validate_prepared_real_context_rows(
        pairs,
        contexts,
        support_channel_count=support_channels,
    )
    exact_pair_rows_sha256 = _canonical_sha256(
        {
            "authorized_pair_rows_sha256": exact_pair_rows_sha256,
            "prepared_context_rows_sha256": exact_context_rows_sha256,
            "representation_contract_sha256": representation.fingerprint,
        }
    )
    expected_view_seeds = independent_view_seeds(
        base_seed=contract.seed,
        step=global_optimizer_step,
    )
    if contexts.view_seeds != expected_view_seeds:
        raise RuntimeError("runtime view seeds differ from checkpointable policy")
    objective = build_fullcontext_objective(candidate_config, contract)
    progress_before_sha256 = progress.fingerprint
    encoder.train()
    loss, valid_anchors, possible_anchors = _optimize_real_pair_contexts(
        encoder,
        optimizer,
        objective,
        pairs,
        contexts,
        contract,
        expected_prior_optimizer_step=progress.global_optimizer_step,
    )
    next_model_state_sha256 = model_state_sha256(encoder)
    next_optimizer_state_sha256 = optimizer_state_sha256(optimizer)
    next_rng_state_sha256 = _rng_state_sha256(capture_fullcontext_rng_state())
    advanced = _advance_fullcontext_progress(
        progress,
        contract,
        lineage,
        next_model_state_sha256=next_model_state_sha256,
        next_optimizer_state_sha256=next_optimizer_state_sha256,
        next_rng_state_sha256=next_rng_state_sha256,
    )
    _validate_adamw_state_step(
        optimizer.state_dict(),
        expected_step=advanced.global_optimizer_step,
    )
    evidence = FullContextOptimizerStep(
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
        view_seeds=contexts.view_seeds,
        loss=float(loss),
        valid_anchor_total=valid_anchors,
        possible_anchor_total=possible_anchors,
        model_state_sha256=next_model_state_sha256,
        optimizer_state_sha256=next_optimizer_state_sha256,
        rng_state_sha256=next_rng_state_sha256,
        dataset_fingerprint=progress.dataset_fingerprint,
        active_plan_fingerprint=plan.fingerprint,
        exact_pair_rows_sha256=exact_pair_rows_sha256,
        representation_contract_sha256=representation.fingerprint,
        representation_lineage_fingerprint=lineage.fingerprint,
        progress_before_sha256=progress_before_sha256,
        progress_after_sha256=advanced.fingerprint,
    )
    return FullContextStepResult(evidence=evidence, progress=advanced)


def capture_fullcontext_backend_state() -> dict[str, Any]:
    """Capture every backend switch that can change continuation numerics."""

    return {
        "torch_deterministic_algorithms": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_deterministic_debug_mode": torch.get_deterministic_debug_mode(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "sdpa_mem_efficient_enabled": (
            torch.backends.cuda.mem_efficient_sdp_enabled()
        ),
        "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_device_count": torch.cuda.device_count(),
    }


def _backend_state_sha256(state: Mapping[str, Any]) -> str:
    expected = {
        "torch_deterministic_algorithms",
        "torch_deterministic_debug_mode",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "cuda_matmul_allow_tf32",
        "cudnn_allow_tf32",
        "sdpa_flash_enabled",
        "sdpa_mem_efficient_enabled",
        "sdpa_math_enabled",
        "cublas_workspace_config",
        "cuda_device_count",
    }
    if set(state) != expected:
        raise ValueError("full-context backend-state schema mismatch")
    boolean_fields = expected - {
        "torch_deterministic_debug_mode",
        "cublas_workspace_config",
        "cuda_device_count",
    }
    if any(not isinstance(state[key], bool) for key in boolean_fields):
        raise ValueError("full-context backend flags must be booleans")
    _require_integer(
        state["torch_deterministic_debug_mode"],
        role="torch deterministic debug mode",
    )
    _require_integer(state["cuda_device_count"], role="backend CUDA device count")
    workspace = state["cublas_workspace_config"]
    if workspace is not None and not isinstance(workspace, str):
        raise ValueError("CUBLAS workspace contract must be a string or null")
    return _canonical_sha256(dict(state))


def fullcontext_backend_state_sha256(state: Mapping[str, Any]) -> str:
    """Return the validated canonical backend-state digest."""

    return _backend_state_sha256(state)


def validate_frozen_fullcontext_backend_state(state: Mapping[str, Any]) -> None:
    _backend_state_sha256(state)
    if (
        state["torch_deterministic_algorithms"] is not True
        or state["cudnn_benchmark"] is not False
        or state["cudnn_deterministic"] is not True
        or state["cuda_matmul_allow_tf32"] is not False
        or state["cudnn_allow_tf32"] is not False
        or state["sdpa_flash_enabled"] is not False
        or state["sdpa_mem_efficient_enabled"] is not False
        or state["sdpa_math_enabled"] is not True
    ):
        raise ValueError("full-context deterministic backend contract is not active")
    if state["cuda_device_count"] and (
        state["cublas_workspace_config"] != ":4096:8"
    ):
        raise ValueError("CUDA continuation lacks the frozen CUBLAS workspace")


def configure_fullcontext_determinism() -> dict[str, Any]:
    """Apply the frozen backend policy, refusing late CUBLAS configuration."""

    if torch.cuda.device_count() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != (
        ":4096:8"
    ):
        raise ValueError("CUBLAS_WORKSPACE_CONFIG must be frozen before process start")
    torch.use_deterministic_algorithms(True)
    torch.set_deterministic_debug_mode("error")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    state = capture_fullcontext_backend_state()
    validate_frozen_fullcontext_backend_state(state)
    return state


def restore_fullcontext_backend_state(state: Mapping[str, Any]) -> None:
    """Restore a captured backend state exactly, including the environment key."""

    _backend_state_sha256(state)
    workspace = state["cublas_workspace_config"]
    if workspace is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    torch.use_deterministic_algorithms(bool(state["torch_deterministic_algorithms"]))
    torch.set_deterministic_debug_mode(int(state["torch_deterministic_debug_mode"]))
    torch.backends.cudnn.benchmark = bool(state["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(state["cudnn_deterministic"])
    torch.backends.cuda.matmul.allow_tf32 = bool(state["cuda_matmul_allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(state["cudnn_allow_tf32"])
    torch.backends.cuda.enable_flash_sdp(bool(state["sdpa_flash_enabled"]))
    torch.backends.cuda.enable_mem_efficient_sdp(
        bool(state["sdpa_mem_efficient_enabled"])
    )
    torch.backends.cuda.enable_math_sdp(bool(state["sdpa_math_enabled"]))
    if capture_fullcontext_backend_state() != dict(state):
        raise RuntimeError("full-context backend state did not restore exactly")


@contextmanager
def preserve_fullcontext_diagnostic_state(
    encoder: nn.Module,
) -> Any:
    """Restore model mode, RNG streams, and backend flags after diagnostics."""

    rng_state = capture_fullcontext_rng_state()
    rng_sha256 = _rng_state_sha256(rng_state)
    backend_state = capture_fullcontext_backend_state()
    backend_sha256 = _backend_state_sha256(backend_state)
    model_sha256 = model_state_sha256(encoder)
    was_training = encoder.training
    try:
        yield {
            "rng_state_sha256": rng_sha256,
            "backend_state_sha256": backend_sha256,
            "model_state_sha256": model_sha256,
        }
    finally:
        encoder.train(was_training)
        restore_fullcontext_backend_state(backend_state)
        restore_fullcontext_rng_state(rng_state)
        if (
            _rng_state_sha256(capture_fullcontext_rng_state()) != rng_sha256
            or _backend_state_sha256(capture_fullcontext_backend_state())
            != backend_sha256
            or model_state_sha256(encoder) != model_sha256
        ):
            raise RuntimeError("diagnostic evaluation changed a restored state")


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


def fullcontext_rng_state_sha256(state: Mapping[str, Any]) -> str:
    """Return the validated canonical Python/NumPy/Torch RNG digest."""

    return _rng_state_sha256(state)


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

    _require_live_fullcontext_progress(progress)
    validate_fullcontext_progress_schedule(progress, contract, lineage)
    validate_fullcontext_optimizer(optimizer, contract)
    _validate_optimizer_model_binding(optimizer, model)
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
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    if (
        model_state_sha256(model_state) != progress.current_model_state_sha256
        or optimizer_state_sha256(optimizer_state)
        != progress.current_optimizer_state_sha256
    ):
        raise ValueError("checkpoint model/optimizer differ from progress state")
    rng_state = capture_fullcontext_rng_state()
    if _rng_state_sha256(rng_state) != progress.current_rng_state_sha256:
        raise ValueError("checkpoint RNG differs from progress state")
    backend_state = capture_fullcontext_backend_state()
    validate_frozen_fullcontext_backend_state(backend_state)
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
        "optimizer_state": optimizer_state,
        "optimizer_state_sha256": optimizer_state_sha256(optimizer_state),
        "scheduler": "none",
        "scheduler_state": None,
        "rng_state": rng_state,
        "rng_state_sha256": _rng_state_sha256(rng_state),
        "backend_state": backend_state,
        "backend_state_sha256": _backend_state_sha256(backend_state),
        "view_seed_state": _view_seed_state(contract, progress),
        "determinism_contract": {
            "length_bucket_plan_and_cursor_checkpointed": True,
            "python_numpy_torch_cpu_all_cuda_rng_checkpointed": True,
            "torch_cudnn_tf32_sdpa_cublas_backend_state_checkpointed": True,
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


def _parse_mechanism_seed_load_receipt(
    payload: Mapping[str, Any],
    *,
    expected_fingerprint: str,
) -> MechanismSeedLoadReceipt:
    expected = {
        "schema_version",
        "artifact_type",
        "checkpoint_sha256",
        "checkpoint_bytes",
        "learned_model_state_sha256",
        "optimizer_state_sha256",
        "optimizer_step",
        "mechanism_sampler_state_sha256",
        "mechanism_view_state_sha256",
        "mechanism_consumed_video_batch_chain_sha256",
        "mechanism_consumed_pair_row_chain_sha256",
        "rng_state_sha256",
        "backend_state_sha256",
        "trainer_contract_fingerprint",
        "representation_contract_sha256",
        "representation_lineage_fingerprint",
        "loaded_as_epoch1_start",
        "direct_epoch150_start_authorized",
        "checkpoint_self_authorizes_training",
    }
    if set(payload) != expected or (
        payload["schema_version"] != 1
        or payload["artifact_type"]
        != "pams_cycleback_mechanism_seed_load_receipt_v1"
        or payload["loaded_as_epoch1_start"] is not True
        or payload["direct_epoch150_start_authorized"] is not False
        or payload["checkpoint_self_authorizes_training"] is not False
    ):
        raise ValueError("mechanism seed load receipt schema mismatch")
    receipt = MechanismSeedLoadReceipt(
        checkpoint_sha256=payload["checkpoint_sha256"],
        checkpoint_bytes=payload["checkpoint_bytes"],
        learned_model_state_sha256=payload["learned_model_state_sha256"],
        optimizer_state_sha256=payload["optimizer_state_sha256"],
        optimizer_step=payload["optimizer_step"],
        mechanism_sampler_state_sha256=payload[
            "mechanism_sampler_state_sha256"
        ],
        mechanism_view_state_sha256=payload["mechanism_view_state_sha256"],
        mechanism_consumed_video_batch_chain_sha256=payload[
            "mechanism_consumed_video_batch_chain_sha256"
        ],
        mechanism_consumed_pair_row_chain_sha256=payload[
            "mechanism_consumed_pair_row_chain_sha256"
        ],
        rng_state_sha256=payload["rng_state_sha256"],
        backend_state_sha256=payload["backend_state_sha256"],
        trainer_contract_fingerprint=payload["trainer_contract_fingerprint"],
        representation_contract_sha256=payload["representation_contract_sha256"],
        representation_lineage_fingerprint=payload[
            "representation_lineage_fingerprint"
        ],
    )
    if receipt.to_dict() != dict(payload) or receipt.fingerprint != expected_fingerprint:
        raise ValueError("mechanism seed load receipt fingerprint mismatch")
    return receipt


def _parse_epoch150_transition(
    payload: Mapping[str, Any],
    *,
    expected_fingerprint: str,
) -> Epoch150TransitionBindings:
    expected = {
        "schema_version",
        "artifact_type",
        "candidate_id",
        "epoch11_checkpoint_sha256",
        "epoch11_checkpoint_bytes",
        "epoch11_model_state_sha256",
        "epoch11_optimizer_state_sha256",
        "epoch11_progress_sha256",
        "trainer_contract_fingerprint",
        "representation_contract_sha256",
        "representation_lineage_fingerprint",
        "fixed_pair_identity_sha256",
        "fixed_pair_payload_sha256",
        "epoch11_gate_outcome_sha256",
        "epoch11_gate_outcome_bytes",
        "threshold_receipt_sha256",
        "threshold_receipt_bytes",
        "epoch150_launch_authorization_sha256",
        "epoch150_launch_authorization_bytes",
        "scientific_gate_passed",
        "epoch150_train337_continuation_authorized",
        "development_evaluation_authorized",
        "sealed_evaluation_authorized",
        "readout_authorized",
    }
    if set(payload) != expected or (
        payload["schema_version"] != 1
        or payload["artifact_type"]
        != "pams_cycleback_epoch150_transition_bindings_v1"
        or payload["scientific_gate_passed"] is not True
        or payload["epoch150_train337_continuation_authorized"] is not True
        or payload["development_evaluation_authorized"] is not False
        or payload["sealed_evaluation_authorized"] is not False
        or payload["readout_authorized"] is not False
    ):
        raise ValueError("epoch150 transition binding schema mismatch")
    transition = Epoch150TransitionBindings(
        candidate_id=payload["candidate_id"],
        epoch11_checkpoint_sha256=payload["epoch11_checkpoint_sha256"],
        epoch11_checkpoint_bytes=payload["epoch11_checkpoint_bytes"],
        epoch11_model_state_sha256=payload["epoch11_model_state_sha256"],
        epoch11_optimizer_state_sha256=payload[
            "epoch11_optimizer_state_sha256"
        ],
        epoch11_progress_sha256=payload["epoch11_progress_sha256"],
        trainer_contract_fingerprint=payload["trainer_contract_fingerprint"],
        representation_contract_sha256=payload["representation_contract_sha256"],
        representation_lineage_fingerprint=payload[
            "representation_lineage_fingerprint"
        ],
        fixed_pair_identity_sha256=payload["fixed_pair_identity_sha256"],
        fixed_pair_payload_sha256=payload["fixed_pair_payload_sha256"],
        epoch11_gate_outcome_sha256=payload["epoch11_gate_outcome_sha256"],
        epoch11_gate_outcome_bytes=payload["epoch11_gate_outcome_bytes"],
        threshold_receipt_sha256=payload["threshold_receipt_sha256"],
        threshold_receipt_bytes=payload["threshold_receipt_bytes"],
        epoch150_launch_authorization_sha256=payload[
            "epoch150_launch_authorization_sha256"
        ],
        epoch150_launch_authorization_bytes=payload[
            "epoch150_launch_authorization_bytes"
        ],
    )
    if transition.to_dict() != dict(payload) or transition.fingerprint != (
        expected_fingerprint
    ):
        raise ValueError("epoch150 transition binding fingerprint mismatch")
    return transition


def _parse_progress(payload: Mapping[str, Any]) -> FullContextProgress:
    expected = {
        "phase",
        "target_epoch",
        "completed_epochs",
        "continuation_optimizer_steps",
        "global_optimizer_step",
        "units",
        "dataset_fingerprint",
        "mechanism_seed_load_receipt",
        "mechanism_seed_load_receipt_sha256",
        "current_model_state_sha256",
        "current_optimizer_state_sha256",
        "current_rng_state_sha256",
        "completed_plan_prefix",
        "active_plan",
        "active_plan_fingerprint",
        "sampler_cursor",
        "plan_dataset_prefix_sha256",
        "epoch11_prefix_checkpoint_sha256",
        "epoch11_prefix_checkpoint_bytes",
        "epoch150_transition",
        "epoch150_transition_sha256",
    }
    if set(payload) != expected:
        raise ValueError("checkpoint progress schema mismatch")
    raw_units = payload["units"]
    raw_receipt = payload["mechanism_seed_load_receipt"]
    raw_prefix = payload["completed_plan_prefix"]
    if (
        not isinstance(raw_units, list)
        or not isinstance(raw_receipt, Mapping)
        or not isinstance(payload["mechanism_seed_load_receipt_sha256"], str)
        or not isinstance(raw_prefix, list)
    ):
        raise ValueError("checkpoint permanent progress lineage is malformed")
    units = tuple(_parse_training_unit(item) for item in raw_units)
    receipt = _parse_mechanism_seed_load_receipt(
        raw_receipt,
        expected_fingerprint=payload["mechanism_seed_load_receipt_sha256"],
    )
    completed_prefix: list[CompletedEpochPlan] = []
    for raw in raw_prefix:
        if not isinstance(raw, Mapping) or set(raw) != {
            "epoch",
            "plan_fingerprint",
            "batch_total",
            "cumulative_batch_total",
        }:
            raise ValueError("checkpoint completed plan-prefix schema mismatch")
        completed_prefix.append(
            CompletedEpochPlan(
                epoch=raw["epoch"],
                plan_fingerprint=raw["plan_fingerprint"],
                batch_total=raw["batch_total"],
                cumulative_batch_total=raw["cumulative_batch_total"],
            )
        )
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
    raw_transition = payload["epoch150_transition"]
    transition_fingerprint = payload["epoch150_transition_sha256"]
    if raw_transition is None:
        if transition_fingerprint is not None:
            raise ValueError("absent epoch150 transition has a fingerprint")
        transition = None
    else:
        if not isinstance(raw_transition, Mapping) or not isinstance(
            transition_fingerprint, str
        ):
            raise ValueError("epoch150 transition schema mismatch")
        transition = _parse_epoch150_transition(
            raw_transition,
            expected_fingerprint=transition_fingerprint,
        )
    progress = FullContextProgress(
        phase=payload["phase"],
        target_epoch=payload["target_epoch"],
        completed_epochs=payload["completed_epochs"],
        continuation_optimizer_steps=payload["continuation_optimizer_steps"],
        units=units,
        dataset_fingerprint=payload["dataset_fingerprint"],
        mechanism_seed_load_receipt=receipt,
        current_model_state_sha256=payload["current_model_state_sha256"],
        current_optimizer_state_sha256=payload[
            "current_optimizer_state_sha256"
        ],
        current_rng_state_sha256=payload["current_rng_state_sha256"],
        completed_plan_prefix=tuple(completed_prefix),
        active_plan=active_plan,
        sampler_cursor=payload["sampler_cursor"],
        epoch11_prefix_checkpoint_sha256=payload[
            "epoch11_prefix_checkpoint_sha256"
        ],
        epoch11_prefix_checkpoint_bytes=payload[
            "epoch11_prefix_checkpoint_bytes"
        ],
        epoch150_transition=transition,
        _runtime_attestation=_FULLCONTEXT_PARSED_ATTESTATION,
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
        "optimizer_state_sha256",
        "scheduler",
        "scheduler_state",
        "rng_state",
        "rng_state_sha256",
        "backend_state",
        "backend_state_sha256",
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
    validate_fullcontext_progress_schedule(progress, contract, lineage)
    if progress.active_plan is not None and (
        progress.active_plan.trainer_contract_fingerprint != contract.fingerprint
        or progress.active_plan.representation_lineage_fingerprint
        != lineage.fingerprint
    ):
        raise ValueError("checkpoint sampler plan lineage mismatch")
    model_state = payload["model_state"]
    if not isinstance(model_state, Mapping) or (
        model_state_sha256(model_state) != payload["model_state_sha256"]
        or payload["model_state_sha256"] != progress.current_model_state_sha256
    ):
        raise ValueError("full-context checkpoint model-state digest mismatch")
    if (
        payload["optimizer_name"] != "AdamW"
        or not isinstance(payload["optimizer_state"], Mapping)
        or payload["scheduler"] != "none"
        or payload["scheduler_state"] is not None
    ):
        raise ValueError("full-context checkpoint optimizer/scheduler mismatch")
    _validate_adamw_state_contract(payload["optimizer_state"], contract)
    _validate_adamw_state_step(
        payload["optimizer_state"],
        expected_step=progress.global_optimizer_step,
    )
    observed_optimizer_sha256 = optimizer_state_sha256(payload["optimizer_state"])
    if (
        observed_optimizer_sha256 != payload["optimizer_state_sha256"]
        or observed_optimizer_sha256 != progress.current_optimizer_state_sha256
    ):
        raise ValueError("checkpoint optimizer bytes differ from progress state")
    rng_state = payload["rng_state"]
    if not isinstance(rng_state, Mapping) or (
        _rng_state_sha256(rng_state) != payload["rng_state_sha256"]
        or payload["rng_state_sha256"] != progress.current_rng_state_sha256
    ):
        raise ValueError("full-context checkpoint RNG-state digest mismatch")
    backend_state = payload["backend_state"]
    if not isinstance(backend_state, Mapping) or (
        _backend_state_sha256(backend_state) != payload["backend_state_sha256"]
    ):
        raise ValueError("full-context checkpoint backend-state digest mismatch")
    validate_frozen_fullcontext_backend_state(backend_state)
    if payload["view_seed_state"] != _view_seed_state(contract, progress):
        raise ValueError("full-context checkpoint next view seeds mismatch")
    if payload["determinism_contract"] != {
        "length_bucket_plan_and_cursor_checkpointed": True,
        "python_numpy_torch_cpu_all_cuda_rng_checkpointed": True,
        "torch_cudnn_tf32_sdpa_cublas_backend_state_checkpointed": True,
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


def atomic_save_new_mechanism_seed_checkpoint(
    payload: Mapping[str, Any],
    path: str | Path,
) -> tuple[str, int]:
    """Publish one seed without any overwrite or post-check TOCTOU window."""

    destination = Path(path)
    parent = destination.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError("checkpoint parent must be an existing non-symlink directory")
    if destination.is_symlink() or destination.exists():
        raise ValueError("mechanism seed checkpoint destination must be absent")
    temporary: Path | None = None
    linked = False
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
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError as exc:
            raise ValueError(
                "mechanism seed checkpoint destination appeared during publication"
            ) from exc
        linked = True
        temporary.unlink()
        temporary = None
        _fsync_directory(parent)
        return stable_file_bytes(destination)[1]
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
            _fsync_directory(parent)
        if not linked and destination.is_symlink():
            raise ValueError("mechanism seed checkpoint destination became a symlink")


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
    current_backend = capture_fullcontext_backend_state()
    validate_frozen_fullcontext_backend_state(current_backend)
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
    if dict(value["backend_state"]) != current_backend:
        raise ValueError("checkpoint backend differs from active process contract")
    validate_fullcontext_optimizer(optimizer, contract)
    _validate_optimizer_model_binding(optimizer, model)
    model.load_state_dict(value["model_state"], strict=True)
    optimizer.load_state_dict(value["optimizer_state"])
    validate_fullcontext_optimizer(optimizer, contract)
    _validate_optimizer_model_binding(optimizer, model)
    _validate_adamw_state_step(
        optimizer.state_dict(),
        expected_step=progress.global_optimizer_step,
    )
    restore_fullcontext_backend_state(value["backend_state"])
    restore_fullcontext_rng_state(value["rng_state"])
    if model_state_sha256(model) != value["model_state_sha256"]:
        raise RuntimeError("restored full-context model state differs from checkpoint")
    if optimizer_state_sha256(optimizer) != progress.current_optimizer_state_sha256:
        raise RuntimeError("restored optimizer state differs from checkpoint progress")
    if _rng_state_sha256(capture_fullcontext_rng_state()) != value["rng_state_sha256"]:
        raise RuntimeError("restored full-context RNG differs from checkpoint")
    if capture_fullcontext_backend_state() != dict(value["backend_state"]):
        raise RuntimeError("restored full-context backend differs from checkpoint")
    return replace(
        progress,
        _runtime_attestation=_FULLCONTEXT_RUNTIME_ATTESTATION,
    )


def initialize_epoch150_from_exact_epoch11_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    contract: FullContextTrainerContract,
    lineage: FullContextLineage,
    epoch12_plan: LengthBucketPlan,
    transition: Epoch150TransitionBindings,
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
    validate_fullcontext_progress_schedule(prefix, contract, lineage)
    expected_epoch11_steps = sum(
        len(
            build_length_bucket_plan(
                prefix.units,
                contract,
                lineage,
                epoch=epoch,
            ).batches
        )
        for epoch in range(1, _EPOCH11_TARGET + 1)
    )
    if prefix.continuation_optimizer_steps != expected_epoch11_steps:
        raise ValueError("epoch11 optimizer steps differ from epochs 1 through 11")
    if (
        epoch12_plan.trainer_contract_fingerprint != contract.fingerprint
        or epoch12_plan.representation_lineage_fingerprint != lineage.fingerprint
    ):
        raise ValueError("epoch12 sampler plan differs from the exact epoch11 lineage")
    expected_epoch12 = build_length_bucket_plan(
        prefix.units,
        contract,
        lineage,
        epoch=_EPOCH11_TARGET + 1,
    )
    if epoch12_plan.to_dict() != expected_epoch12.to_dict():
        raise ValueError("epoch12 plan is not the deterministic continuation prefix")
    if (
        transition.candidate_id != contract.candidate_id
        or transition.epoch11_checkpoint_sha256 != expected_sha256
        or transition.epoch11_checkpoint_bytes != expected_bytes
        or transition.epoch11_model_state_sha256 != model_state_sha256(model)
        or transition.epoch11_optimizer_state_sha256
        != optimizer_state_sha256(optimizer)
        or transition.epoch11_progress_sha256 != prefix.fingerprint
        or transition.trainer_contract_fingerprint != contract.fingerprint
        or transition.representation_contract_sha256
        != lineage.representation_contract_sha256
        or transition.representation_lineage_fingerprint != lineage.fingerprint
    ):
        raise ValueError("epoch150 gate/threshold/launch lineage differs from epoch11")
    return _initial_epoch150_progress_from_loaded_epoch11(
        epoch12_plan,
        completed_epoch11_progress=prefix,
        transition=transition,
    )


def _mechanism_seed_predecessor(
    lineage: FullContextLineage | MechanismSeedPredecessorLineage,
) -> MechanismSeedPredecessorLineage:
    if isinstance(lineage, FullContextLineage):
        return lineage.mechanism_seed_predecessor()
    if isinstance(lineage, MechanismSeedPredecessorLineage):
        return lineage
    raise TypeError("mechanism seed lineage has an unsupported type")


def _mechanism_seed_view_state(
    contract: FullContextTrainerContract,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    for step in range(1, _MECHANISM_OPTIMIZER_STEPS + 1):
        consumed = independent_view_seeds(base_seed=contract.seed, step=step)
        digest.update(_canonical_json_bytes([step, consumed[0], consumed[1]]))
    return {
        "policy": "pams-conventional-cycleback-view-v1",
        "base_seed": contract.seed,
        "first_optimizer_augmentation_step": 1,
        "last_consumed_augmentation_step": _MECHANISM_OPTIMIZER_STEPS,
        "consumed_view_seed_chain_sha256": digest.hexdigest(),
        "next_augmentation_step": _MECHANISM_OPTIMIZER_STEPS + 1,
        "next_view_seeds": list(
            independent_view_seeds(
                base_seed=contract.seed,
                step=_MECHANISM_OPTIMIZER_STEPS + 1,
            )
        ),
        "independent_views": True,
    }


def build_mechanism_seed_checkpoint_payload(
    model: nn.Module,
    optimizer: Optimizer,
    contract: FullContextTrainerContract,
    lineage: MechanismSeedPredecessorLineage,
    sampler_state: MechanismSeedSamplerState,
) -> dict[str, Any]:
    """Capture the exact step-256 boundary before any diagnostic evaluation."""

    validate_fullcontext_optimizer(optimizer, contract)
    _validate_optimizer_model_binding(optimizer, model)
    if not model.training:
        raise ValueError("mechanism learned-L must be captured in training mode")
    optimizer_state = copy.deepcopy(optimizer.state_dict())
    _validate_adamw_state_contract(optimizer_state, contract)
    _validate_adamw_state_step(
        optimizer_state,
        expected_step=_MECHANISM_OPTIMIZER_STEPS,
    )
    model_state = _clone_model_state(model)
    observed_model_sha256 = model_state_sha256(model_state)
    if observed_model_sha256 != lineage.mechanism_learned_model_state_sha256:
        raise ValueError("mechanism seed learned-L digest differs from predecessor")
    rng_state = capture_fullcontext_rng_state()
    backend_state = capture_fullcontext_backend_state()
    validate_frozen_fullcontext_backend_state(backend_state)
    if (
        sampler_state.video_batch_size != contract.mechanism_video_batch_size
        or sampler_state.maximum_pairs_per_step
        != contract.mechanism_maximum_pairs_per_step
        or sampler_state.completed_optimizer_steps
        != contract.mechanism_optimizer_steps
    ):
        raise ValueError("mechanism sampler state differs from trainer contract")
    next_view_seed_state = _mechanism_seed_view_state(contract)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": (
            "pams_conventional_cycleback_mechanism_seed_checkpoint_v1"
        ),
        "status": "passed",
        "candidate_id": contract.candidate_id,
        "trainer_contract": contract.to_dict(),
        "trainer_contract_fingerprint": contract.fingerprint,
        "representation_contract_sha256": (
            lineage.representation_contract_sha256
        ),
        "lineage": lineage.to_dict(),
        "lineage_fingerprint": lineage.fingerprint,
        "completed_optimizer_steps": _MECHANISM_OPTIMIZER_STEPS,
        "completed_epochs": 0,
        "sampler_state": sampler_state.to_dict(),
        "sampler_state_sha256": sampler_state.fingerprint,
        "model_state": model_state,
        "model_state_sha256": observed_model_sha256,
        "model_state_role": "L_learned_cycleback_encoder",
        "model_training_mode": True,
        "optimizer_name": "AdamW",
        "optimizer_state": optimizer_state,
        "optimizer_state_sha256": optimizer_state_sha256(optimizer_state),
        "scheduler": "none",
        "scheduler_state": None,
        "rng_state": rng_state,
        "rng_state_sha256": _rng_state_sha256(rng_state),
        "backend_state": backend_state,
        "backend_state_sha256": _backend_state_sha256(backend_state),
        "next_view_seed_state": next_view_seed_state,
        "next_view_seed_state_sha256": _canonical_sha256(next_view_seed_state),
        "capture_contract": {
            "captured_in_training_process_at_exact_step256": True,
            "captured_before_diagnostic_evaluation": True,
            "retroactive_reconstruction_allowed": False,
            "json_only_model_digest_may_substitute": False,
        },
        "epoch1_start_contract": {
            "loaded_state_is_epoch1_start": True,
            "optimizer_state_continues_exact_step256": True,
            "optimizer_reset_allowed": False,
            "direct_epoch150_start_authorized": False,
        },
        "authority_boundaries": _authority_boundaries(),
    }
    validate_mechanism_seed_checkpoint_payload(payload, contract, lineage)
    return payload


def validate_mechanism_seed_checkpoint_payload(
    payload: Mapping[str, Any],
    contract: FullContextTrainerContract,
    lineage: FullContextLineage | MechanismSeedPredecessorLineage,
) -> None:
    """Validate an exact L/optimizer/RNG seed required for epoch one.

    A JSON-only ``final_model_state_sha256`` remains insufficient and cannot
    be upgraded or reconstructed after the run.
    """

    predecessor = _mechanism_seed_predecessor(lineage)

    expected = {
        "schema_version",
        "artifact_type",
        "status",
        "candidate_id",
        "trainer_contract",
        "trainer_contract_fingerprint",
        "representation_contract_sha256",
        "lineage",
        "lineage_fingerprint",
        "completed_optimizer_steps",
        "completed_epochs",
        "sampler_state",
        "sampler_state_sha256",
        "model_state",
        "model_state_sha256",
        "model_state_role",
        "model_training_mode",
        "optimizer_name",
        "optimizer_state",
        "optimizer_state_sha256",
        "scheduler",
        "scheduler_state",
        "rng_state",
        "rng_state_sha256",
        "backend_state",
        "backend_state_sha256",
        "next_view_seed_state",
        "next_view_seed_state_sha256",
        "capture_contract",
        "epoch1_start_contract",
        "authority_boundaries",
    }
    if set(payload) != expected or (
        payload["schema_version"] != 1
        or payload["artifact_type"]
        != "pams_conventional_cycleback_mechanism_seed_checkpoint_v1"
        or payload["status"] != "passed"
        or payload["candidate_id"] != contract.candidate_id
        or payload["trainer_contract"] != contract.to_dict()
        or payload["trainer_contract_fingerprint"] != contract.fingerprint
        or payload["representation_contract_sha256"]
        != predecessor.representation_contract_sha256
        or payload["lineage"] != predecessor.to_dict()
        or payload["lineage_fingerprint"]
        != predecessor.fingerprint
        or payload["completed_optimizer_steps"] != _MECHANISM_OPTIMIZER_STEPS
        or payload["completed_epochs"] != 0
        or payload["model_state_role"] != "L_learned_cycleback_encoder"
        or payload["model_training_mode"] is not True
        or payload["optimizer_name"] != "AdamW"
        or payload["scheduler"] != "none"
        or payload["scheduler_state"] is not None
    ):
        raise ValueError("mechanism seed checkpoint contract or lineage mismatch")
    sampler_state = payload["sampler_state"]
    if not isinstance(sampler_state, Mapping) or set(sampler_state) != {
        "policy",
        "eligible_video_total",
        "ordered_eligible_video_ids_sha256",
        "video_batch_size",
        "maximum_pairs_per_step",
        "completed_optimizer_steps",
        "consumed_video_batch_chain_sha256",
        "consumed_pair_row_chain_sha256",
        "next_cyclic_start_index",
        "mechanism_sampler_resume_allowed",
        "epoch1_sampler_requires_new_sealed_plan",
    }:
        raise ValueError("mechanism seed sampler-state schema mismatch")
    parsed_sampler = MechanismSeedSamplerState(**dict(sampler_state))
    if (
        parsed_sampler.fingerprint != payload["sampler_state_sha256"]
        or parsed_sampler.video_batch_size != contract.mechanism_video_batch_size
        or parsed_sampler.maximum_pairs_per_step
        != contract.mechanism_maximum_pairs_per_step
        or parsed_sampler.completed_optimizer_steps
        != contract.mechanism_optimizer_steps
    ):
        raise ValueError("mechanism seed sampler state differs from contract")
    state = payload["model_state"]
    if not isinstance(state, Mapping) or (
        model_state_sha256(state) != payload["model_state_sha256"]
        or payload["model_state_sha256"]
        != predecessor.mechanism_learned_model_state_sha256
    ):
        raise ValueError("mechanism learned-L state digest mismatch")
    if not isinstance(payload["optimizer_state"], Mapping):
        raise ValueError("mechanism seed checkpoint lacks optimizer state")
    _validate_adamw_state_contract(payload["optimizer_state"], contract)
    _validate_adamw_state_step(
        payload["optimizer_state"],
        expected_step=_MECHANISM_OPTIMIZER_STEPS,
    )
    if optimizer_state_sha256(payload["optimizer_state"]) != payload[
        "optimizer_state_sha256"
    ]:
        raise ValueError("mechanism optimizer-state digest mismatch")
    rng_state = payload["rng_state"]
    if not isinstance(rng_state, Mapping) or (
        _rng_state_sha256(rng_state) != payload["rng_state_sha256"]
    ):
        raise ValueError("mechanism seed checkpoint RNG-state digest mismatch")
    backend_state = payload["backend_state"]
    if not isinstance(backend_state, Mapping) or (
        _backend_state_sha256(backend_state) != payload["backend_state_sha256"]
    ):
        raise ValueError("mechanism seed checkpoint backend-state digest mismatch")
    validate_frozen_fullcontext_backend_state(backend_state)
    if (
        payload["next_view_seed_state"] != _mechanism_seed_view_state(contract)
        or payload["next_view_seed_state_sha256"]
        != _canonical_sha256(payload["next_view_seed_state"])
    ):
        raise ValueError("mechanism seed next-view state mismatch")
    if isinstance(lineage, FullContextLineage) and (
        payload["optimizer_state_sha256"]
        != lineage.mechanism_optimizer_state_sha256
        or payload["rng_state_sha256"] != lineage.mechanism_rng_state_sha256
        or payload["backend_state_sha256"]
        != lineage.mechanism_backend_state_sha256
        or payload["sampler_state_sha256"]
        != lineage.mechanism_sampler_state_sha256
        or payload["next_view_seed_state_sha256"]
        != lineage.mechanism_view_state_sha256
        or parsed_sampler.consumed_video_batch_chain_sha256
        != lineage.mechanism_consumed_video_batch_chain_sha256
        or parsed_sampler.consumed_pair_row_chain_sha256
        != lineage.mechanism_consumed_pair_row_chain_sha256
    ):
        raise ValueError("mechanism seed state closure differs from final lineage")
    if payload["capture_contract"] != {
        "captured_in_training_process_at_exact_step256": True,
        "captured_before_diagnostic_evaluation": True,
        "retroactive_reconstruction_allowed": False,
        "json_only_model_digest_may_substitute": False,
    }:
        raise ValueError("mechanism seed capture provenance mismatch")
    if payload["epoch1_start_contract"] != {
        "loaded_state_is_epoch1_start": True,
        "optimizer_state_continues_exact_step256": True,
        "optimizer_reset_allowed": False,
        "direct_epoch150_start_authorized": False,
    }:
        raise ValueError("mechanism seed epoch-one start contract mismatch")
    if payload["authority_boundaries"] != _authority_boundaries():
        raise ValueError("mechanism checkpoint may not self-authorize continuation")


def _load_mechanism_seed_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    contract: FullContextTrainerContract,
    lineage: FullContextLineage,
    model: nn.Module,
    optimizer: Optimizer,
) -> MechanismSeedLoadReceipt:
    """Restore exact learned L only after byte and semantic verification."""

    if (
        expected_sha256 != lineage.mechanism_seed_checkpoint_sha256
        or expected_bytes != lineage.mechanism_seed_checkpoint_bytes
    ):
        raise ValueError("mechanism seed identity differs from frozen lineage")
    current_backend = capture_fullcontext_backend_state()
    validate_frozen_fullcontext_backend_state(current_backend)
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
    if dict(value["backend_state"]) != current_backend:
        raise ValueError("mechanism seed backend differs from active process contract")
    if not model.training:
        raise ValueError("mechanism seed must be loaded into an epoch-one training model")
    validate_fullcontext_optimizer(optimizer, contract)
    _validate_optimizer_model_binding(optimizer, model)
    model.load_state_dict(value["model_state"], strict=True)
    optimizer.load_state_dict(value["optimizer_state"])
    validate_fullcontext_optimizer(optimizer, contract)
    _validate_optimizer_model_binding(optimizer, model)
    _validate_adamw_state_step(
        optimizer.state_dict(),
        expected_step=_MECHANISM_OPTIMIZER_STEPS,
    )
    restore_fullcontext_backend_state(value["backend_state"])
    restore_fullcontext_rng_state(value["rng_state"])
    if (
        model_state_sha256(model)
        != lineage.mechanism_learned_model_state_sha256
        or optimizer_state_sha256(optimizer) != value["optimizer_state_sha256"]
        or _rng_state_sha256(capture_fullcontext_rng_state())
        != value["rng_state_sha256"]
        or _backend_state_sha256(capture_fullcontext_backend_state())
        != value["backend_state_sha256"]
    ):
        raise RuntimeError("restored mechanism seed differs from exact learned L state")
    receipt = MechanismSeedLoadReceipt(
        checkpoint_sha256=expected_sha256,
        checkpoint_bytes=expected_bytes,
        learned_model_state_sha256=lineage.mechanism_learned_model_state_sha256,
        optimizer_state_sha256=value["optimizer_state_sha256"],
        optimizer_step=_MECHANISM_OPTIMIZER_STEPS,
        mechanism_sampler_state_sha256=value["sampler_state_sha256"],
        mechanism_view_state_sha256=value["next_view_seed_state_sha256"],
        mechanism_consumed_video_batch_chain_sha256=value["sampler_state"][
            "consumed_video_batch_chain_sha256"
        ],
        mechanism_consumed_pair_row_chain_sha256=value["sampler_state"][
            "consumed_pair_row_chain_sha256"
        ],
        rng_state_sha256=value["rng_state_sha256"],
        backend_state_sha256=value["backend_state_sha256"],
        trainer_contract_fingerprint=contract.fingerprint,
        representation_contract_sha256=lineage.representation_contract_sha256,
        representation_lineage_fingerprint=lineage.fingerprint,
    )
    if not receipt.loaded_as_epoch1_start:
        raise RuntimeError("mechanism learned-L was not loaded as the epoch-one start")
    return receipt


def initialize_epoch11_from_exact_mechanism_seed_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    contract: FullContextTrainerContract,
    lineage: FullContextLineage,
    epoch1_plan: LengthBucketPlan,
    model: nn.Module,
    optimizer: Optimizer,
) -> FullContextProgress:
    """The sole public mechanism-load to epoch-one progress transition."""

    receipt = _load_mechanism_seed_checkpoint(
        path,
        expected_sha256=expected_sha256,
        expected_bytes=expected_bytes,
        contract=contract,
        lineage=lineage,
        model=model,
        optimizer=optimizer,
    )
    progress = _initial_epoch11_progress_from_loaded_seed(
        epoch1_plan,
        mechanism_seed_load_receipt=receipt,
    )
    validate_fullcontext_progress_schedule(progress, contract, lineage)
    if (
        model_state_sha256(model) != progress.current_model_state_sha256
        or optimizer_state_sha256(optimizer)
        != progress.current_optimizer_state_sha256
        or _rng_state_sha256(capture_fullcontext_rng_state())
        != progress.current_rng_state_sha256
    ):
        raise RuntimeError("epoch-one runtime differs from the loaded mechanism seed")
    return progress


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
    pairs: NativeWindowPairBatch,
    contexts: PairSegmentContexts,
    candidate_config: ConventionalCycleBackConfig,
    contract: FullContextTrainerContract,
    representation: FullContextRepresentationContract,
    lineage: FullContextLineage,
    progress: FullContextProgress,
    *,
    epoch11_checkpoint_sha256: str,
    epoch11_checkpoint_bytes: int,
    expected_epoch11_model_state_sha256: str,
    expected_epoch11_optimizer_state_sha256: str,
    expected_pair_identity_sha256: str,
    expected_pair_payload_sha256: str,
) -> dict[str, Any]:
    """Evaluate the typed v4e null adapter; diagnostic contexts never backpropagate."""

    validate_representation_contract_lineage(representation, lineage)
    validate_fullcontext_candidate_config(candidate_config, contract)
    validate_fullcontext_progress_schedule(progress, contract, lineage)
    _require_sha256(epoch11_checkpoint_sha256, role="epoch11 checkpoint")
    _require_integer(
        epoch11_checkpoint_bytes,
        role="epoch11 checkpoint bytes",
        minimum=1,
    )
    _require_sha256(
        expected_epoch11_model_state_sha256,
        role="expected epoch11 model state",
    )
    _require_sha256(
        expected_epoch11_optimizer_state_sha256,
        role="expected epoch11 optimizer state",
    )
    if (
        progress.phase != "epoch11"
        or progress.completed_epochs != _EPOCH11_TARGET
        or progress.active_plan is not None
        or progress.sampler_cursor != 0
        or progress.current_model_state_sha256
        != expected_epoch11_model_state_sha256
        or progress.current_optimizer_state_sha256
        != expected_epoch11_optimizer_state_sha256
        or model_state_sha256(encoder) != expected_epoch11_model_state_sha256
    ):
        raise ValueError("epoch11 controls require the exact completed checkpoint state")
    if (
        representation.representation_family != "v4e_unified2d_coco17_padded33"
        or representation.support_channel_count != 17
        or representation.support_channel_to_pose_joint_indices != tuple(range(17))
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
    objective = build_fullcontext_objective(candidate_config, contract)
    validate_frozen_fullcontext_backend_state(capture_fullcontext_backend_state())
    conditions: dict[str, dict[str, Any]] = {}
    null_contracts: dict[str, Any] = {}
    with preserve_fullcontext_diagnostic_state(encoder) as boundary, torch.inference_mode():
        encoder.eval()
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
        "bindings": {
            "epoch11_checkpoint_sha256": epoch11_checkpoint_sha256,
            "epoch11_checkpoint_bytes": epoch11_checkpoint_bytes,
            "epoch11_model_state_sha256": expected_epoch11_model_state_sha256,
            "epoch11_optimizer_state_sha256": (
                expected_epoch11_optimizer_state_sha256
            ),
            "epoch11_progress_sha256": progress.fingerprint,
            "plan_dataset_prefix_sha256": progress.plan_dataset_prefix_sha256,
            "trainer_contract_fingerprint": contract.fingerprint,
            "representation_contract_sha256": representation.fingerprint,
            "representation_lineage_fingerprint": lineage.fingerprint,
            "pair_identity_sha256": observed_identity,
            "pair_payload_sha256": observed_payload,
            "pre_gate_rng_state_sha256": boundary["rng_state_sha256"],
            "pre_gate_backend_state_sha256": boundary["backend_state_sha256"],
            "pre_gate_model_state_sha256": boundary["model_state_sha256"],
            "rng_backend_and_model_restored_by_finally": True,
        },
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
    *,
    threshold_receipt_sha256: str,
    threshold_receipt_bytes: int,
) -> dict[str, Any]:
    """Return scientific eligibility only; a separate authority must launch 150."""

    _require_sha256(threshold_receipt_sha256, role="epoch11 threshold receipt")
    _require_integer(
        threshold_receipt_bytes,
        role="epoch11 threshold receipt bytes",
        minimum=1,
    )
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
    bindings = controls.get("bindings")
    expected_binding_keys = {
        "epoch11_checkpoint_sha256",
        "epoch11_checkpoint_bytes",
        "epoch11_model_state_sha256",
        "epoch11_optimizer_state_sha256",
        "epoch11_progress_sha256",
        "plan_dataset_prefix_sha256",
        "trainer_contract_fingerprint",
        "representation_contract_sha256",
        "representation_lineage_fingerprint",
        "pair_identity_sha256",
        "pair_payload_sha256",
        "pre_gate_rng_state_sha256",
        "pre_gate_backend_state_sha256",
        "pre_gate_model_state_sha256",
        "rng_backend_and_model_restored_by_finally",
    }
    if (
        not isinstance(conditions, Mapping)
        or not isinstance(ratios, Mapping)
        or not isinstance(bindings, Mapping)
        or set(bindings) != expected_binding_keys
        or bindings.get("rng_backend_and_model_restored_by_finally") is not True
    ):
        raise ValueError("epoch11 control metrics are missing")
    for key in expected_binding_keys - {
        "epoch11_checkpoint_bytes",
        "rng_backend_and_model_restored_by_finally",
    }:
        _require_sha256(bindings[key], role=f"epoch11 control binding {key}")
    _require_integer(
        bindings["epoch11_checkpoint_bytes"],
        role="epoch11 control checkpoint bytes",
        minimum=1,
    )
    if (
        bindings["pair_identity_sha256"] != controls.get("pair_identity_sha256")
        or bindings["pair_payload_sha256"] != controls.get("pair_payload_sha256")
        or bindings["epoch11_model_state_sha256"]
        != bindings["pre_gate_model_state_sha256"]
    ):
        raise ValueError("epoch11 controls are not internally bound")
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
        "bindings": {
            **dict(bindings),
            "controls_sha256": _canonical_sha256(dict(controls)),
            "threshold_receipt_sha256": threshold_receipt_sha256,
            "threshold_receipt_bytes": threshold_receipt_bytes,
        },
        "criteria": criteria,
        "overall_pass": passed,
        "scientifically_eligible_for_epoch150_train337": passed,
        "epoch150_train337_continuation_authorized": False,
        "development_evaluation_authorized": False,
        "sealed_evaluation_authorized": False,
        "readout_authorized": False,
        "authorization_required_after_scientific_pass": True,
    }


def build_epoch150_transition_bindings(
    *,
    gate_outcome: Mapping[str, Any],
    gate_outcome_identity: tuple[str, int],
    threshold_receipt: Mapping[str, Any],
    threshold_receipt_identity: tuple[str, int],
    launch_authorization: Mapping[str, Any],
    launch_authorization_identity: tuple[str, int],
    epoch11_checkpoint_identity: tuple[str, int],
    epoch11_model_state_sha256: str,
    epoch11_progress: FullContextProgress,
    contract: FullContextTrainerContract,
    representation: FullContextRepresentationContract,
    lineage: FullContextLineage,
) -> Epoch150TransitionBindings:
    """Validate exact gate/threshold/launch semantics before epoch 12 exists.

    The future secure adapter must additionally pin all three file identities.
    This science layer deliberately grants no authority from paths or caller
    values on its own.
    """

    for role, identity in (
        ("epoch11 gate outcome", gate_outcome_identity),
        ("epoch11 threshold receipt", threshold_receipt_identity),
        ("epoch150 launch authorization", launch_authorization_identity),
        ("epoch11 checkpoint", epoch11_checkpoint_identity),
    ):
        _require_sha256(identity[0], role=role)
        _require_integer(identity[1], role=f"{role} bytes", minimum=1)
    _require_sha256(epoch11_model_state_sha256, role="epoch11 model state")
    validate_representation_contract_lineage(representation, lineage)
    validate_fullcontext_progress_schedule(epoch11_progress, contract, lineage)
    if (
        epoch11_progress.phase != "epoch11"
        or epoch11_progress.completed_epochs != _EPOCH11_TARGET
        or epoch11_progress.active_plan is not None
        or epoch11_progress.current_model_state_sha256
        != epoch11_model_state_sha256
    ):
        raise ValueError("epoch150 transition requires completed epoch11 progress")
    gate_bindings = gate_outcome.get("bindings")
    if (
        gate_outcome.get("artifact_type")
        != "pams_cycleback_epoch11_scientific_gate_decision_v1"
        or gate_outcome.get("overall_pass") is not True
        or gate_outcome.get("scientifically_eligible_for_epoch150_train337")
        is not True
        or gate_outcome.get("epoch150_train337_continuation_authorized")
        is not False
        or gate_outcome.get("development_evaluation_authorized") is not False
        or gate_outcome.get("sealed_evaluation_authorized") is not False
        or gate_outcome.get("readout_authorized") is not False
        or not isinstance(gate_bindings, Mapping)
    ):
        raise ValueError("epoch11 scientific gate outcome is not an exact PASS")
    expected_gate_links = {
        "epoch11_checkpoint_sha256": epoch11_checkpoint_identity[0],
        "epoch11_checkpoint_bytes": epoch11_checkpoint_identity[1],
        "epoch11_model_state_sha256": epoch11_model_state_sha256,
        "epoch11_optimizer_state_sha256": (
            epoch11_progress.current_optimizer_state_sha256
        ),
        "epoch11_progress_sha256": epoch11_progress.fingerprint,
        "trainer_contract_fingerprint": contract.fingerprint,
        "representation_contract_sha256": representation.fingerprint,
        "representation_lineage_fingerprint": lineage.fingerprint,
        "threshold_receipt_sha256": threshold_receipt_identity[0],
        "threshold_receipt_bytes": threshold_receipt_identity[1],
    }
    if any(gate_bindings.get(key) != value for key, value in expected_gate_links.items()):
        raise ValueError("epoch11 gate outcome lineage differs from exact checkpoint")
    threshold_expected = {
        "schema_version",
        "artifact_type",
        "status",
        "candidate_id",
        "thresholds_sha256",
        "thresholds_frozen_before_execution",
        "label_free",
        "epoch150_train337_continuation_authorized",
        "development_evaluation_authorized",
        "sealed_evaluation_authorized",
    }
    if set(threshold_receipt) != threshold_expected or (
        threshold_receipt["schema_version"] != 1
        or threshold_receipt["artifact_type"]
        != "pams_cycleback_epoch11_threshold_preregistration_receipt_v1"
        or threshold_receipt["status"] != "frozen"
        or threshold_receipt["candidate_id"] != contract.candidate_id
        or threshold_receipt["thresholds_sha256"]
        != gate_outcome.get("thresholds_sha256")
        or threshold_receipt["thresholds_frozen_before_execution"] is not True
        or threshold_receipt["label_free"] is not True
        or threshold_receipt["epoch150_train337_continuation_authorized"]
        is not False
        or threshold_receipt["development_evaluation_authorized"] is not False
        or threshold_receipt["sealed_evaluation_authorized"] is not False
    ):
        raise ValueError("epoch11 threshold preregistration receipt mismatch")
    launch_expected = {
        "schema_version",
        "artifact_type",
        "status",
        "candidate_id",
        "bindings",
        "label_free",
        "epoch150_train337_continuation_authorized",
        "development_evaluation_authorized",
        "sealed_evaluation_authorized",
        "readout_authorized",
    }
    launch_bindings = launch_authorization.get("bindings")
    if set(launch_authorization) != launch_expected or (
        launch_authorization["schema_version"] != 1
        or launch_authorization["artifact_type"]
        != "pams_cycleback_epoch150_launch_authorization_v1"
        or launch_authorization["status"] != "authorized"
        or launch_authorization["candidate_id"] != contract.candidate_id
        or launch_authorization["label_free"] is not True
        or launch_authorization["epoch150_train337_continuation_authorized"]
        is not True
        or launch_authorization["development_evaluation_authorized"] is not False
        or launch_authorization["sealed_evaluation_authorized"] is not False
        or launch_authorization["readout_authorized"] is not False
        or not isinstance(launch_bindings, Mapping)
        or launch_bindings
        != {
            "epoch11_checkpoint_sha256": epoch11_checkpoint_identity[0],
            "epoch11_checkpoint_bytes": epoch11_checkpoint_identity[1],
            "epoch11_model_state_sha256": epoch11_model_state_sha256,
            "epoch11_optimizer_state_sha256": (
                epoch11_progress.current_optimizer_state_sha256
            ),
            "epoch11_progress_sha256": epoch11_progress.fingerprint,
            "epoch11_gate_outcome_sha256": gate_outcome_identity[0],
            "epoch11_gate_outcome_bytes": gate_outcome_identity[1],
            "threshold_receipt_sha256": threshold_receipt_identity[0],
            "threshold_receipt_bytes": threshold_receipt_identity[1],
            "trainer_contract_fingerprint": contract.fingerprint,
            "representation_contract_sha256": representation.fingerprint,
            "representation_lineage_fingerprint": lineage.fingerprint,
        }
    ):
        raise ValueError("epoch150 launch authorization lineage mismatch")
    return Epoch150TransitionBindings(
        candidate_id=contract.candidate_id,
        epoch11_checkpoint_sha256=epoch11_checkpoint_identity[0],
        epoch11_checkpoint_bytes=epoch11_checkpoint_identity[1],
        epoch11_model_state_sha256=epoch11_model_state_sha256,
        epoch11_optimizer_state_sha256=(
            epoch11_progress.current_optimizer_state_sha256
        ),
        epoch11_progress_sha256=epoch11_progress.fingerprint,
        trainer_contract_fingerprint=contract.fingerprint,
        representation_contract_sha256=representation.fingerprint,
        representation_lineage_fingerprint=lineage.fingerprint,
        fixed_pair_identity_sha256=gate_bindings["pair_identity_sha256"],
        fixed_pair_payload_sha256=gate_bindings["pair_payload_sha256"],
        epoch11_gate_outcome_sha256=gate_outcome_identity[0],
        epoch11_gate_outcome_bytes=gate_outcome_identity[1],
        threshold_receipt_sha256=threshold_receipt_identity[0],
        threshold_receipt_bytes=threshold_receipt_identity[1],
        epoch150_launch_authorization_sha256=launch_authorization_identity[0],
        epoch150_launch_authorization_bytes=launch_authorization_identity[1],
    )
