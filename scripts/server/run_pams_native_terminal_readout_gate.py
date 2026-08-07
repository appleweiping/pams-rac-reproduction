"""Run the frozen train337-only native terminal readout gate.

The command has no dataset-manifest, media, annotation, action-class,
repetition-target, development, sealed-evaluation, prediction, scoring, or
training interface.  It binds one completed epoch-150 encoder to its exact
progress log, completion receipt, source-export receipt, experiment config,
epoch-11 continuation gate, and checkpoint-bound train337 pose cache.

All per-video estimates remain in memory.  The artifact contains aggregate
statistics and cryptographic commitments only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F

from pams.config import PAMSConfig
from pams.consensus import MultiExpertCounter
from pams.data import PoseCacheSetSnapshot, uniform_resample
from pams.diagnostics import (
    _device,
    _identifier_commitment,
    _load_training_poses,
    _peek_checkpoint,
    _stable_file_sha256,
    _summary,
)
from pams.period import estimate_period_from_embedding_velocity_vectors
from pams.recurrence_carrier import (
    RecurrenceCarrierBatch,
    build_recurrence_carrier_curves,
    estimate_recurrence_carrier_curves,
)
from pams.reproducibility import (
    clean_git_revision,
    hardware_fingerprint,
    sha256_json,
)
from pams.run_manifest import CompletedRunReceipt
from pams.training import (
    collate_pose_sequences,
    load_model_checkpoint,
    validate_terminal_checkpoint,
)
from pams.types import PoseSequence
from scripts.server import run_pams_native_epoch11_train_gate as _epoch11

_GATE_SCHEMA_VERSION = 1
_ARTIFACT_SCHEMA_VERSION = 1
_EXPECTED_ARTIFACT_TYPE = "pams_native_terminal_readout_train337_gate_v1"
_EXPECTED_RECEIPT_TYPE = "pams_native_terminal_readout_train337_gate_receipt_v1"
_EXPECTED_EPOCH11_ARTIFACT_TYPE = "pams_native_epoch11_train_mechanism_gate_v1"
_EXPECTED_EPOCH11_RECEIPT_TYPE = (
    "pams_native_epoch11_train_mechanism_gate_receipt_v1"
)
_CANDIDATE_ORDER = ("A", "B", "C")
_FORBIDDEN_FIELD_NAMES = _epoch11._FORBIDDEN_FIELD_NAMES
_FORBIDDEN_PATH_TOKEN = _epoch11._FORBIDDEN_PATH_TOKEN

_SPECIFICATION_KEYS = {
    "schema_version",
    "artifact_type",
    "receipt_type",
    "classification",
    "frozen_before_candidate",
    "expected_protocol",
    "expected_seed",
    "expected_training_video_total",
    "expected_completed_epochs",
    "minimum_lag_pair_total",
    "near_collapse_rms",
    "time_scale_factors",
    "candidate_profiles",
    "candidate_invariants",
    "thresholds",
    "non_authorizing_metrics",
}
_PROFILE_KEYS = {
    "fixed_training_period_frames",
    "anchor_stride",
    "required_prior_scientific_rejections",
}
_INVARIANT_KEYS = {
    "model_input_dim",
    "model_dim",
    "embedding_dim",
    "transformer_layers",
    "attention_heads",
    "feedforward_dim",
    "position_encoding_mode",
    "norm_first",
    "period_minimum",
    "period_maximum_safety_ceiling",
    "period_maximum_mode",
    "period_training_mode",
    "direct_fft_timebase",
    "loss_scales",
    "use_cross_cluster_negatives",
    "exclude_other_scale_positives_from_denominator",
    "encoder_epochs",
    "effective_batch_size",
    "skeleton_augmentation_enabled",
    "expert_mode",
    "action_curve_source",
}
_THRESHOLD_KEYS = {
    "near_collapsed_share_maximum",
    "lag_eligible_share_minimum",
    "real_cycle_margin_median_minimum",
    "real_minus_strongest_null_median_minimum",
    "real_beats_all_nulls_share_minimum",
    "period_boundary_share_maximum_exclusive",
    "period_mode_share_maximum_exclusive",
    "period_time_scale_eligible_share_minimum",
    "period_time_scale_median_relative_error_maximum",
    "carrier_eligible_available_share_minimum",
    "embedding_shuffle_harmonic_median_ratio_maximum",
    "embedding_shuffle_gate_energy_median_ratio_maximum",
    "zero_pose_positive_period_confidence_share_maximum",
    "zero_pose_carrier_available_share_maximum",
    "zero_pose_positive_support_share_maximum",
    "static_pose_positive_period_confidence_share_maximum",
    "static_pose_carrier_available_share_maximum",
    "static_pose_positive_support_share_maximum",
    "expert_majority_share_minimum",
    "peak_total_zero_share_maximum_exclusive",
    "peak_total_mode_share_maximum_exclusive",
    "selected_vs_active_reference_gap_median_maximum",
    "time_scale_peak_total_eligible_share_minimum",
    "time_scale_peak_total_exact_share_minimum",
    "time_scale_peak_total_within_one_share_minimum",
}
_NON_AUTHORIZING_METRICS = {
    "absolute_period_confidence",
    "training_period_lock",
    "absolute_harmonic_energy_fraction",
    "raw_or_normalized_curve_standard_deviation",
    "exact_active_support_magnitude",
    "split_half_period_agreement",
    "positional_index_permutation_sensitivity",
    "full_timeline_reference_gap",
    "absolute_peak_total_distribution",
    "terminal_loss_drop",
    "cross_candidate_metric_ranking",
}
_OUTPUT_KEYS = {
    "schema_version",
    "artifact_type",
    "status",
    "classification",
    "protocol",
    "seed",
    "candidate",
    "label_firewall",
    "inputs",
    "algorithm",
    "thresholds",
    "aggregates",
    "hard_invariants",
    "gate",
    "scientific_caveats",
    "hardware",
    "hardware_sha256",
    "runtime",
    "runtime_sha256",
    "read_only_verification",
}
_RECEIPT_KEYS = {
    "schema_version",
    "artifact_type",
    "artifact_locator",
    "artifact_sha256",
    "artifact_bytes",
    "artifact_status",
    "candidate_id",
    "overall_pass",
    "eligible_for_single_frozen_dev84_protocol_build",
    "next_candidate_training_authorized",
    "encoder_checkpoint_sha256",
    "encoder_progress_sha256",
    "encoder_completion_receipt_sha256",
    "encoder_started_receipt_sha256",
    "source_export_receipt_sha256",
    "experiment_config_sha256",
    "gate_specification_sha256",
    "pose_snapshot_sha256",
    "pose_cache_set_sha256",
    "epoch11_gate_artifact_sha256",
    "epoch11_gate_receipt_sha256",
    "source_git_sha",
    "code_files_sha256_commitment",
    "prior_scientific_rejections_sha256",
    "aggregate_only",
    "dev84_pose_or_scoring_authorized",
    "test105_evaluation_authorized",
}
_CANDIDATE_KEYS = {
    "id",
    "fixed_training_period_frames_provenance_only",
    "anchor_stride",
    "first_pass_only",
    "prior_scientific_rejections",
    "cross_candidate_metric_ranking_forbidden",
}
_GATE_DECISION_KEYS = {
    "thresholds_frozen_before_candidate_a",
    "criteria",
    "criterion_total",
    "all_hard_criteria_pass",
    "overall_pass",
    "eligible_for_single_frozen_dev84_protocol_build",
    "dev84_identity_media_pose_or_scoring_authorized",
    "test105_evaluation_authorized",
    "next_candidate_training_authorized",
    "all_candidates_exhausted",
}
_CRITERION_TO_THRESHOLD = {
    "near_collapsed_share": "near_collapsed_share_maximum",
    "lag_eligible_share": "lag_eligible_share_minimum",
    "real_cycle_margin_median": "real_cycle_margin_median_minimum",
    "real_minus_strongest_null_median": (
        "real_minus_strongest_null_median_minimum"
    ),
    "real_beats_all_nulls_share": "real_beats_all_nulls_share_minimum",
    "period_boundary_share": "period_boundary_share_maximum_exclusive",
    "period_mode_share": "period_mode_share_maximum_exclusive",
    "period_time_scale_eligible_share": (
        "period_time_scale_eligible_share_minimum"
    ),
    "period_time_scale_median_relative_error": (
        "period_time_scale_median_relative_error_maximum"
    ),
    "carrier_eligible_available_share": (
        "carrier_eligible_available_share_minimum"
    ),
    "embedding_shuffle_harmonic_median_ratio": (
        "embedding_shuffle_harmonic_median_ratio_maximum"
    ),
    "embedding_shuffle_gate_energy_median_ratio": (
        "embedding_shuffle_gate_energy_median_ratio_maximum"
    ),
    "zero_pose_positive_period_confidence_share": (
        "zero_pose_positive_period_confidence_share_maximum"
    ),
    "zero_pose_carrier_available_share": (
        "zero_pose_carrier_available_share_maximum"
    ),
    "zero_pose_positive_support_share": (
        "zero_pose_positive_support_share_maximum"
    ),
    "static_pose_positive_period_confidence_share": (
        "static_pose_positive_period_confidence_share_maximum"
    ),
    "static_pose_carrier_available_share": (
        "static_pose_carrier_available_share_maximum"
    ),
    "static_pose_positive_support_share": (
        "static_pose_positive_support_share_maximum"
    ),
    "expert_majority_share": "expert_majority_share_minimum",
    "peak_total_zero_share": "peak_total_zero_share_maximum_exclusive",
    "peak_total_mode_share": "peak_total_mode_share_maximum_exclusive",
    "selected_vs_active_reference_gap_median": (
        "selected_vs_active_reference_gap_median_maximum"
    ),
    "time_scale_peak_total_eligible_share": (
        "time_scale_peak_total_eligible_share_minimum"
    ),
    "time_scale_peak_total_exact_share": (
        "time_scale_peak_total_exact_share_minimum"
    ),
    "time_scale_peak_total_within_one_share": (
        "time_scale_peak_total_within_one_share_minimum"
    ),
}
_HARD_INVARIANT_KEYS = {
    "selected_peak_total_is_one_of_three_experts",
    "eligible_readouts_majority_first_then_fft_nearest_verified",
    "active_mask_subset_of_valid_mask",
    "curve_zero_outside_active_mask",
    "score_zero_outside_valid_mask",
    "finite_readout_outputs",
    "per_video_predictions_persisted",
}
_AGGREGATE_KEYS = {
    "representation",
    "period",
    "readout",
    "peak_total",
    "time_scale",
    "embedding_shuffle_fixed_real_period",
    "zero_pose",
    "static_pose",
    "per_video_rows_persisted",
}
_READ_ONLY_VERIFICATION_KEYS = {
    "all_file_inputs_unchanged",
    "train337_pose_cache_set_unchanged",
    "model_state_sha256_before",
    "model_state_sha256_after",
    "model_or_optimizer_state_updated",
    "training_steps_executed",
    "pose_cache_write_operations",
    "per_video_prediction_write_operations",
}


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate YAML field is forbidden: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class CandidateProfile:
    fixed_training_period_frames: int
    anchor_stride: int
    required_prior_scientific_rejections: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GateSpecification:
    artifact_type: str
    receipt_type: str
    classification: str
    frozen_before_candidate: str
    expected_protocol: str
    expected_seed: int
    expected_training_video_total: int
    expected_completed_epochs: int
    minimum_lag_pair_total: int
    near_collapse_rms: float
    time_scale_factors: tuple[float, ...]
    candidate_profiles: Mapping[str, CandidateProfile]
    candidate_invariants: Mapping[str, Any]
    thresholds: Mapping[str, float]
    non_authorizing_metrics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReadoutSample:
    video_id: str
    timeline_length: int
    valid_total: int
    period_upper_bound: int
    period_frames: float
    period_confidence: float
    period_lag_eligible: bool
    carrier_eligible: bool
    raw_carrier_available: bool
    harmonic_energy_fraction: float
    recurrence_gate_energy: float
    active_support_fraction: float
    normalized_curve_rms: float
    raw_curve_rms: float
    selected_peak_total: int
    active_reference_peak_total: int
    full_timeline_reference_peak_total: int
    expert_peak_totals: tuple[int, int, int]
    selected_expert: str
    selection_rule: str


@dataclass(frozen=True, slots=True)
class RepresentationSample:
    temporal_rms: float
    lag_eligible: bool
    real_cycle_margin: float | None
    pose_shuffle_cycle_margin: float | None
    embedding_shuffle_cycle_margin: float | None
    zero_pose_cycle_margin: float | None
    static_pose_cycle_margin: float | None
    real_minus_strongest_null: float | None
    real_beats_all_nulls: bool | None


@dataclass(frozen=True, slots=True)
class ShuffleControlSample:
    real_carrier_eligible: bool
    real_harmonic_energy_fraction: float
    shuffled_harmonic_energy_fraction: float
    real_recurrence_gate_energy: float
    shuffled_recurrence_gate_energy: float


@dataclass(frozen=True, slots=True)
class PoseControlSample:
    positive_period_confidence: bool
    carrier_eligible: bool
    positive_support: bool


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _walk_mapping_keys(value: Any) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_walk_mapping_keys(item))
    elif isinstance(value, list | tuple):
        for item in value:
            keys.extend(_walk_mapping_keys(item))
    return tuple(keys)


def _reject_forbidden_mapping_keys(value: Any, *, document: str) -> None:
    for key in _walk_mapping_keys(value):
        if key.strip().lower() in _FORBIDDEN_FIELD_NAMES:
            raise ValueError(
                f"{document} contains forbidden privileged field {key!r}"
            )


def _load_unique_yaml(path: Path, *, document: str) -> dict[str, Any]:
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid {document} YAML") from exc
    if not isinstance(raw, dict) or any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{document} root must be a string-keyed mapping")
    _reject_forbidden_mapping_keys(raw, document=document)
    return raw


def _strict_candidate_invariants(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _INVARIANT_KEYS:
        raise ValueError("candidate invariant schema mismatch")
    integer_keys = {
        "model_input_dim",
        "model_dim",
        "embedding_dim",
        "transformer_layers",
        "attention_heads",
        "feedforward_dim",
        "period_minimum",
        "period_maximum_safety_ceiling",
        "encoder_epochs",
        "effective_batch_size",
    }
    boolean_keys = {
        "norm_first",
        "use_cross_cluster_negatives",
        "exclude_other_scale_positives_from_denominator",
        "skeleton_augmentation_enabled",
    }
    string_keys = _INVARIANT_KEYS - integer_keys - boolean_keys - {"loss_scales"}
    for key in integer_keys:
        if type(value[key]) is not int or value[key] < 1:
            raise TypeError(f"candidate_invariants.{key} must be a positive integer")
    for key in boolean_keys:
        if type(value[key]) is not bool:
            raise TypeError(f"candidate_invariants.{key} must be boolean")
    for key in string_keys:
        if type(value[key]) is not str or not value[key].strip():
            raise TypeError(f"candidate_invariants.{key} must be non-empty text")
    loss_scales = value["loss_scales"]
    if (
        not isinstance(loss_scales, list)
        or len(loss_scales) != 1
        or type(loss_scales[0]) is not float
        or not math.isfinite(loss_scales[0])
        or loss_scales[0] != 1.0
    ):
        raise TypeError("candidate_invariants.loss_scales must be exactly [1.0]")
    return dict(value)


def load_gate_specification(path: str | Path) -> GateSpecification:
    source = Path(path)
    _reject_privileged_path(source, role="gate specification")
    raw = _load_unique_yaml(source, document="gate specification")
    if set(raw) != _SPECIFICATION_KEYS:
        raise ValueError("terminal gate specification schema mismatch")
    if raw["schema_version"] != _GATE_SCHEMA_VERSION:
        raise ValueError("unsupported terminal gate specification schema")
    if raw["artifact_type"] != _EXPECTED_ARTIFACT_TYPE:
        raise ValueError("terminal gate artifact type changed")
    if raw["receipt_type"] != _EXPECTED_RECEIPT_TYPE:
        raise ValueError("terminal gate receipt type changed")
    if raw["frozen_before_candidate"] != "A":
        raise ValueError("terminal gate must be frozen before candidate A")

    profiles_raw = raw["candidate_profiles"]
    if not isinstance(profiles_raw, dict) or tuple(profiles_raw) != _CANDIDATE_ORDER:
        raise ValueError("candidate profile order must be exactly A, B, C")
    expected_predecessors = {"A": (), "B": ("A",), "C": ("A", "B")}
    profiles: dict[str, CandidateProfile] = {}
    for candidate_id in _CANDIDATE_ORDER:
        item = profiles_raw[candidate_id]
        if not isinstance(item, dict) or set(item) != _PROFILE_KEYS:
            raise ValueError(f"candidate {candidate_id} profile schema mismatch")
        predecessors = item["required_prior_scientific_rejections"]
        if not isinstance(predecessors, list) or tuple(predecessors) != expected_predecessors[
            candidate_id
        ]:
            raise ValueError(f"candidate {candidate_id} predecessor order changed")
        profiles[candidate_id] = CandidateProfile(
            fixed_training_period_frames=_positive_integer(
                item["fixed_training_period_frames"],
                name=f"candidate_profiles.{candidate_id}.fixed_training_period_frames",
            ),
            anchor_stride=_positive_integer(
                item["anchor_stride"],
                name=f"candidate_profiles.{candidate_id}.anchor_stride",
            ),
            required_prior_scientific_rejections=tuple(predecessors),
        )
    expected_profile_values = {"A": (16, 4), "B": (16, 2), "C": (24, 4)}
    if {
        name: (profile.fixed_training_period_frames, profile.anchor_stride)
        for name, profile in profiles.items()
    } != expected_profile_values:
        raise ValueError("candidate A/B/C scientific profiles changed")

    invariants = _strict_candidate_invariants(raw["candidate_invariants"])

    thresholds_raw = raw["thresholds"]
    if not isinstance(thresholds_raw, dict) or set(thresholds_raw) != _THRESHOLD_KEYS:
        raise ValueError("terminal gate threshold schema mismatch")
    thresholds = {
        key: _finite_number(value, name=f"thresholds.{key}")
        for key, value in thresholds_raw.items()
    }
    if any(value < 0.0 for value in thresholds.values()):
        raise ValueError("terminal gate thresholds must be non-negative")
    fraction_keys = {
        key
        for key in _THRESHOLD_KEYS
        if "share" in key or "ratio" in key or "relative_error" in key
    }
    if any(not 0.0 <= thresholds[key] <= 1.0 for key in fraction_keys):
        raise ValueError("share, ratio, and relative-error thresholds must lie in [0, 1]")

    factors = raw["time_scale_factors"]
    if not isinstance(factors, list):
        raise ValueError("time_scale_factors must be a list")
    normalized_factors = tuple(
        _finite_number(value, name="time_scale_factors") for value in factors
    )
    if normalized_factors != (0.75, 1.25):
        raise ValueError("terminal time-scale factors must be exactly 0.75 and 1.25")

    non_authorizing = raw["non_authorizing_metrics"]
    if (
        not isinstance(non_authorizing, list)
        or len(non_authorizing) != len(set(non_authorizing))
        or set(non_authorizing) != _NON_AUTHORIZING_METRICS
    ):
        raise ValueError("non-authorizing metric declaration changed")
    near_collapse = _finite_number(raw["near_collapse_rms"], name="near_collapse_rms")
    if near_collapse <= 0.0:
        raise ValueError("near_collapse_rms must be positive")
    return GateSpecification(
        artifact_type=str(raw["artifact_type"]),
        receipt_type=str(raw["receipt_type"]),
        classification=str(raw["classification"]).strip(),
        frozen_before_candidate=str(raw["frozen_before_candidate"]),
        expected_protocol=str(raw["expected_protocol"]).strip(),
        expected_seed=_positive_integer(raw["expected_seed"], name="expected_seed"),
        expected_training_video_total=_positive_integer(
            raw["expected_training_video_total"],
            name="expected_training_video_total",
        ),
        expected_completed_epochs=_positive_integer(
            raw["expected_completed_epochs"],
            name="expected_completed_epochs",
        ),
        minimum_lag_pair_total=_positive_integer(
            raw["minimum_lag_pair_total"],
            name="minimum_lag_pair_total",
        ),
        near_collapse_rms=near_collapse,
        time_scale_factors=normalized_factors,
        candidate_profiles=profiles,
        candidate_invariants=dict(invariants),
        thresholds=thresholds,
        non_authorizing_metrics=tuple(str(value) for value in non_authorizing),
    )


def _reject_privileged_path(path: Path, *, role: str) -> None:
    normalized = str(path).replace("\\", "/")
    if _FORBIDDEN_PATH_TOKEN.search(normalized):
        raise ValueError(f"{role} path contains a forbidden privileged token")


def _strict_json(path: Path, *, document: str) -> dict[str, Any]:
    _reject_privileged_path(path, role=document)
    return _epoch11._strict_json(path.read_text(encoding="utf-8"), document=document)


def _load_candidate_config(path: Path) -> PAMSConfig:
    return PAMSConfig.model_validate(
        _load_unique_yaml(path, document="candidate experiment config")
    )


def _validate_candidate_config(
    config: PAMSConfig,
    specification: GateSpecification,
    *,
    candidate_id: str,
) -> CandidateProfile:
    if candidate_id not in specification.candidate_profiles:
        raise ValueError("candidate_id must be one of A, B, C")
    profile = specification.candidate_profiles[candidate_id]
    actual = {
        "model_input_dim": config.model.input_dim,
        "model_dim": config.model.model_dim,
        "embedding_dim": config.model.embedding_dim,
        "transformer_layers": config.model.layers,
        "attention_heads": config.model.heads,
        "feedforward_dim": config.model.feedforward_dim,
        "position_encoding_mode": config.model.position_encoding_mode,
        "norm_first": config.model.norm_first,
        "period_minimum": config.period.minimum,
        "period_maximum_safety_ceiling": config.period.maximum,
        "period_maximum_mode": config.period.maximum_mode,
        "period_training_mode": config.period.training_mode,
        "direct_fft_timebase": config.period.direct_fft_timebase,
        "loss_scales": list(config.loss.scales),
        "use_cross_cluster_negatives": config.loss.use_cross_cluster_negatives,
        "exclude_other_scale_positives_from_denominator": (
            config.loss.exclude_other_scale_positives_from_denominator
        ),
        "encoder_epochs": config.training.epochs,
        "effective_batch_size": config.training.effective_batch_size,
        "skeleton_augmentation_enabled": config.training.skeleton_augmentation.enabled,
        "expert_mode": config.consensus.expert_mode,
        "action_curve_source": config.readout.action_curve_source,
    }
    expected = dict(specification.candidate_invariants)
    if actual != expected:
        raise ValueError(
            "candidate invariant mismatch: "
            + json.dumps({"actual": actual, "expected": expected}, sort_keys=True)
        )
    if config.protocol != specification.expected_protocol:
        raise ValueError("candidate protocol differs from gate specification")
    if config.seed != specification.expected_seed:
        raise ValueError("candidate seed differs from gate specification")
    if config.training.epochs != specification.expected_completed_epochs:
        raise ValueError("candidate terminal epoch differs from gate specification")
    if config.period.fixed_period_frames != profile.fixed_training_period_frames:
        raise ValueError("candidate fixed training period differs from frozen profile")
    if config.loss.anchor_stride != profile.anchor_stride:
        raise ValueError("candidate anchor stride differs from frozen profile")
    return profile


def _validated_counter(config: PAMSConfig) -> MultiExpertCounter:
    consensus = config.consensus
    if consensus.expert_mode != "multi":
        raise ValueError("terminal readout gate requires all three experts")
    return MultiExpertCounter(
        sigma_multipliers=consensus.sigma_multipliers,
        distance_multipliers=consensus.distance_multipliers,
        short_window_multiplier=consensus.short_window_multiplier,
        long_window_multiplier=consensus.long_window_multiplier,
        height_factor=consensus.height_factor,
        prominence_factor=consensus.prominence_factor,
        long_window_weight=consensus.long_window_weight,
        expert_mode=consensus.expert_mode,
    )


def _selection_rule(
    expert_peak_totals: tuple[int, int, int],
    *,
    selected_peak_total: int,
    active_reference_peak_total: int,
) -> str:
    frequencies = Counter(expert_peak_totals)
    majority_total, votes = max(
        frequencies.items(),
        key=lambda item: (
            item[1],
            -abs(item[0] - active_reference_peak_total),
            -item[0],
        ),
    )
    if votes >= 2:
        if selected_peak_total != majority_total:
            raise RuntimeError("three-expert consensus did not select its majority")
        return "majority_first"
    minimum_gap = min(
        abs(value - active_reference_peak_total) for value in expert_peak_totals
    )
    if (
        selected_peak_total not in expert_peak_totals
        or abs(selected_peak_total - active_reference_peak_total) != minimum_gap
    ):
        raise RuntimeError("three-expert consensus did not select an FFT-nearest expert")
    return "fft_nearest_fallback"


def _finite_float(value: Tensor | float, *, role: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"terminal readout produced non-finite {role}")
    return result


def _period_upper_bound(
    timeline_length: int,
    *,
    minimum: int,
    maximum: int,
    maximum_mode: str,
) -> int:
    if maximum_mode == "half_timeline":
        return min(maximum, max(minimum, timeline_length // 2))
    if maximum_mode == "fixed":
        return min(maximum, max(minimum, timeline_length - 1))
    raise ValueError("unsupported maximum period mode")


def _fractional_lag_similarity(
    embeddings: Tensor,
    valid_mask: Tensor,
    *,
    timeline_length: int,
    lag: float,
    minimum_pairs: int,
) -> float | None:
    if embeddings.ndim != 2 or valid_mask.ndim != 1:
        raise ValueError("fractional lag inputs must have [time,dimension] and [time]")
    if not math.isfinite(lag) or lag <= 0.0:
        raise ValueError("fractional lag must be finite and positive")
    if timeline_length < 1 or timeline_length > embeddings.shape[0]:
        raise ValueError("fractional lag timeline length is invalid")
    valid = valid_mask[:timeline_length].to(dtype=torch.bool)
    states = F.normalize(
        embeddings[:timeline_length].detach().to(dtype=torch.float32),
        p=2,
        dim=1,
        eps=1e-12,
    )
    positions = torch.arange(
        timeline_length, dtype=states.dtype, device=states.device
    )
    shifted = positions + float(lag)
    inside = shifted <= float(timeline_length - 1)
    safe = shifted.clamp(0.0, float(timeline_length - 1))
    lower = torch.floor(safe).to(dtype=torch.long)
    upper = torch.ceil(safe).to(dtype=torch.long)
    fraction = safe - lower.to(dtype=safe.dtype)
    exact = upper == lower
    source_valid = valid[lower] & (exact | valid[upper])
    pair_valid = valid & inside & source_valid
    if int(pair_valid.sum()) < minimum_pairs:
        return None
    interpolated = (
        states[lower] * (1.0 - fraction).unsqueeze(1)
        + states[upper] * fraction.unsqueeze(1)
    )
    interpolated = F.normalize(interpolated, p=2, dim=1, eps=1e-12)
    similarities = (states * interpolated).sum(dim=1).clamp(-1.0, 1.0)
    return float(similarities[pair_valid].mean())


def _fractional_cycle_margin(
    embeddings: Tensor,
    valid_mask: Tensor,
    *,
    timeline_length: int,
    period: float,
    minimum_pairs: int,
) -> float | None:
    values = tuple(
        _fractional_lag_similarity(
            embeddings,
            valid_mask,
            timeline_length=timeline_length,
            lag=multiplier * period,
            minimum_pairs=minimum_pairs,
        )
        for multiplier in (0.5, 1.0, 1.5)
    )
    if any(value is None for value in values):
        return None
    half, full, three_half = values
    if half is None or full is None or three_half is None:
        raise RuntimeError("fractional cycle margin eligibility changed unexpectedly")
    return full - 0.5 * (half + three_half)


def _readout_samples_from_batch(
    *,
    video_ids: Sequence[str],
    embeddings: Tensor,
    valid_mask: Tensor,
    timeline_lengths: Tensor,
    readout: RecurrenceCarrierBatch,
    config: PAMSConfig,
    minimum_pairs: int,
    counter: MultiExpertCounter | None = None,
) -> tuple[ReadoutSample, ...]:
    if embeddings.ndim != 3 or valid_mask.shape != embeddings.shape[:2]:
        raise ValueError("readout batch requires [batch,time,dimension] embeddings")
    if timeline_lengths.shape != embeddings.shape[:1] or len(video_ids) != embeddings.shape[0]:
        raise ValueError("readout batch identity or length shape mismatch")
    counter = _validated_counter(config) if counter is None else counter
    samples: list[ReadoutSample] = []
    for index, video_id in enumerate(video_ids):
        length = int(timeline_lengths[index])
        valid = valid_mask[index, :length].to(dtype=torch.bool)
        active = readout.active_masks[index, :length]
        curve = readout.curves[index, :length]
        score = readout.recurrence_scores[index, :length]
        gate = readout.recurrence_gates[index, :length]
        if bool(active.logical_and(~valid).any()):
            raise RuntimeError("recurrence active mask escaped the valid pose mask")
        if not bool(torch.isfinite(curve).all()):
            raise RuntimeError("recurrence carrier contains a non-finite value")
        if bool((curve.masked_select(~active).abs() > 0.0).any()):
            raise RuntimeError("recurrence carrier is nonzero outside its active mask")
        if bool((score.masked_select(~valid).abs() > 0.0).any()):
            raise RuntimeError("recurrence score is nonzero outside valid frames")
        if bool((gate.masked_select(~active).abs() > 0.0).any()):
            raise RuntimeError("recurrence gate is nonzero outside its active mask")
        period = _finite_float(readout.periods[index], role="period")
        confidence = _finite_float(
            readout.period_confidences[index], role="period confidence"
        )
        structural = _fractional_cycle_margin(
            embeddings[index],
            valid_mask[index],
            timeline_length=length,
            period=period,
            minimum_pairs=minimum_pairs,
        ) is not None
        raw_available = bool(readout.available[index])
        carrier_eligible = confidence > 0.0 and structural and raw_available
        active_reference = int(readout.active_reference_counts[index])
        if carrier_eligible:
            result = counter.count(
                curve,
                period_frames=period,
                valid_mask=active,
                period_confidence=confidence,
                reference_count_override=active_reference,
            )
            if result.selection_mode != "multi":
                raise RuntimeError("terminal readout bypassed three-expert consensus")
            expert_peak_totals = tuple(int(value) for value in result.expert_counts)
            selected_peak_total = int(result.count)
            selected_expert = str(result.selected_expert)
            active_reference = int(result.reference_count)
            rule = _selection_rule(
                expert_peak_totals,
                selected_peak_total=selected_peak_total,
                active_reference_peak_total=active_reference,
            )
            if selected_peak_total not in expert_peak_totals:
                raise RuntimeError("selected peak total is not one of the three experts")
        else:
            # Ineligible samples remain in aggregate coverage and zero-share
            # denominators, but their action curves are never passed to the
            # peak detector.  This prevents confidence-zero carriers from
            # creating apparently valid per-video outputs.
            expert_peak_totals = (0, 0, 0)
            selected_peak_total = 0
            selected_expert = "medium"
            rule = "ineligible_no_peak_readout"
        samples.append(
            ReadoutSample(
                video_id=str(video_id),
                timeline_length=length,
                valid_total=int(valid.sum()),
                period_upper_bound=_period_upper_bound(
                    length,
                    minimum=config.period.minimum,
                    maximum=config.period.maximum,
                    maximum_mode=config.period.maximum_mode,
                ),
                period_frames=period,
                period_confidence=confidence,
                period_lag_eligible=confidence > 0.0 and structural,
                carrier_eligible=carrier_eligible,
                raw_carrier_available=raw_available,
                harmonic_energy_fraction=_finite_float(
                    readout.harmonic_energy_fractions[index], role="harmonic fraction"
                ),
                recurrence_gate_energy=_finite_float(
                    readout.recurrence_gate_energies[index], role="gate energy"
                ),
                active_support_fraction=_finite_float(
                    readout.active_support_fractions[index], role="active support"
                ),
                normalized_curve_rms=_finite_float(
                    readout.curve_standard_deviations[index], role="curve RMS"
                ),
                raw_curve_rms=_finite_float(
                    readout.raw_curve_standard_deviations[index], role="raw curve RMS"
                ),
                selected_peak_total=selected_peak_total,
                active_reference_peak_total=active_reference,
                full_timeline_reference_peak_total=int(
                    readout.full_timeline_reference_counts[index]
                ),
                expert_peak_totals=expert_peak_totals,
                selected_expert=selected_expert,
                selection_rule=rule,
            )
        )
    return tuple(samples)


def readout_samples_from_embeddings(
    *,
    video_ids: Sequence[str],
    embeddings: Tensor,
    valid_mask: Tensor,
    timeline_lengths: Tensor,
    config: PAMSConfig,
    minimum_pairs: int = 8,
) -> tuple[ReadoutSample, ...]:
    """Run the frozen adaptive period/carrier/expert readout on one batch."""

    readout = estimate_recurrence_carrier_curves(
        embeddings.detach(),
        minimum_period=config.period.minimum,
        maximum_period=config.period.maximum,
        valid_mask=valid_mask,
        timeline_lengths=timeline_lengths,
        maximum_mode=config.period.maximum_mode,
    )
    return _readout_samples_from_batch(
        video_ids=video_ids,
        embeddings=embeddings,
        valid_mask=valid_mask,
        timeline_lengths=timeline_lengths,
        readout=readout,
        config=config,
        minimum_pairs=minimum_pairs,
    )


def representation_samples_from_embeddings(
    *,
    real_embeddings: Tensor,
    pose_shuffle_embeddings: Tensor,
    embedding_shuffle_embeddings: Tensor,
    zero_pose_embeddings: Tensor,
    static_pose_embeddings: Tensor,
    valid_mask: Tensor,
    timeline_lengths: Tensor,
    periods: Tensor,
    minimum_pairs: int,
) -> tuple[RepresentationSample, ...]:
    expected = real_embeddings.shape
    if real_embeddings.ndim != 3 or any(
        value.shape != expected
        for value in (
            pose_shuffle_embeddings,
            embedding_shuffle_embeddings,
            zero_pose_embeddings,
            static_pose_embeddings,
        )
    ):
        raise ValueError("representation views must share [batch,time,dimension]")
    if valid_mask.shape != expected[:2] or timeline_lengths.shape != expected[:1]:
        raise ValueError("representation masks or lengths are incompatible")
    if periods.shape != expected[:1]:
        raise ValueError("representation periods must match the batch")
    samples: list[RepresentationSample] = []
    for index in range(expected[0]):
        length = int(timeline_lengths[index])
        period = float(periods[index])
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError("representation period must be finite and positive")
        margins = tuple(
            _fractional_cycle_margin(
                values[index],
                valid_mask[index],
                timeline_length=length,
                period=period,
                minimum_pairs=minimum_pairs,
            )
            for values in (
                real_embeddings,
                pose_shuffle_embeddings,
                embedding_shuffle_embeddings,
                zero_pose_embeddings,
                static_pose_embeddings,
            )
        )
        real, pose_shuffle, embedding_shuffle, zero_pose, static_pose = margins
        eligible = all(value is not None for value in margins)
        separation: float | None = None
        beats: bool | None = None
        if eligible:
            if any(value is None for value in margins):
                raise RuntimeError("representation null eligibility changed unexpectedly")
            if (
                real is None
                or pose_shuffle is None
                or embedding_shuffle is None
                or zero_pose is None
                or static_pose is None
            ):
                raise RuntimeError("representation null margin is unexpectedly absent")
            separation = real - max(
                pose_shuffle, embedding_shuffle, zero_pose, static_pose
            )
            beats = separation > 0.0
        samples.append(
            RepresentationSample(
                temporal_rms=_epoch11._temporal_rms(
                    real_embeddings[index], valid_mask[index], length
                ),
                lag_eligible=eligible,
                real_cycle_margin=real,
                pose_shuffle_cycle_margin=pose_shuffle,
                embedding_shuffle_cycle_margin=embedding_shuffle,
                zero_pose_cycle_margin=zero_pose,
                static_pose_cycle_margin=static_pose,
                real_minus_strongest_null=separation,
                real_beats_all_nulls=beats,
            )
        )
    return tuple(samples)


def _deterministically_shuffle_valid_embeddings(
    embeddings: Tensor,
    valid_mask: Tensor,
    video_ids: Sequence[str],
) -> Tensor:
    if embeddings.ndim != 3 or valid_mask.shape != embeddings.shape[:2]:
        raise ValueError("embedding shuffle inputs have incompatible shapes")
    if len(video_ids) != embeddings.shape[0]:
        raise ValueError("embedding shuffle video IDs must match the batch")
    shuffled = embeddings.detach().clone()
    for index, video_id in enumerate(video_ids):
        valid_indices = torch.nonzero(
            valid_mask[index].detach().cpu(), as_tuple=False
        ).flatten()
        if valid_indices.numel() < 2:
            continue
        digest = hashlib.sha256(
            f"pams-native-terminal-embedding-shuffle-v1\0{video_id}".encode()
        ).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))
        permutation = torch.randperm(valid_indices.numel(), generator=generator)
        destination = valid_indices.to(device=embeddings.device)
        source = valid_indices[permutation].to(device=embeddings.device)
        shuffled[index, destination] = embeddings[index, source]
    return shuffled


def _static_pose_batch(poses: Tensor, valid_mask: Tensor) -> Tensor:
    if poses.ndim != 4 or valid_mask.shape != poses.shape[:2]:
        raise ValueError("static pose inputs have incompatible shapes")
    static = torch.zeros_like(poses)
    for index in range(poses.shape[0]):
        valid_indices = torch.nonzero(
            valid_mask[index].detach().cpu(), as_tuple=False
        ).flatten()
        if valid_indices.numel() == 0:
            continue
        source = int(valid_indices[valid_indices.numel() // 2])
        destination = valid_indices.to(device=poses.device)
        static[index, destination] = poses[index, source]
    return static


def _pose_control_samples(
    *,
    embeddings: Tensor,
    readout: RecurrenceCarrierBatch,
    valid_mask: Tensor,
    timeline_lengths: Tensor,
    minimum_pairs: int,
) -> tuple[PoseControlSample, ...]:
    rows: list[PoseControlSample] = []
    for index in range(readout.periods.shape[0]):
        length = int(timeline_lengths[index])
        period = _finite_float(readout.periods[index], role="control period")
        confidence = _finite_float(
            readout.period_confidences[index], role="control period confidence"
        )
        structural = _fractional_cycle_margin(
            embeddings[index],
            valid_mask[index],
            timeline_length=length,
            period=period,
            minimum_pairs=minimum_pairs,
        ) is not None
        support = _finite_float(
            readout.active_support_fractions[index], role="control active support"
        )
        rows.append(
            PoseControlSample(
                positive_period_confidence=confidence > 0.0,
                carrier_eligible=(
                    confidence > 0.0 and structural and bool(readout.available[index])
                ),
                positive_support=support > 0.0,
            )
        )
    return tuple(rows)


def _encode_baseline_and_controls(
    model: nn.Module,
    sequences: Sequence[PoseSequence],
    *,
    config: PAMSConfig,
    minimum_pairs: int,
    device: torch.device,
    batch_size: int,
) -> tuple[
    tuple[ReadoutSample, ...],
    tuple[RepresentationSample, ...],
    tuple[ShuffleControlSample, ...],
    tuple[PoseControlSample, ...],
    tuple[PoseControlSample, ...],
]:
    counter = _validated_counter(config)
    baseline_rows: list[ReadoutSample] = []
    representation_rows: list[RepresentationSample] = []
    shuffle_rows: list[ShuffleControlSample] = []
    zero_rows: list[PoseControlSample] = []
    static_rows: list[PoseControlSample] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            pose_shuffle = _epoch11._shuffle_valid_pose_rows(
                batch.poses, batch.valid_mask, batch.video_ids
            )
            static_pose = _static_pose_batch(batch.poses, batch.valid_mask)
            real_embeddings = model.encoder(batch.poses, batch.valid_mask)
            pose_shuffle_embeddings = model.encoder(pose_shuffle, batch.valid_mask)
            zero_pose_embeddings = model.encoder(
                torch.zeros_like(batch.poses), batch.valid_mask
            )
            static_pose_embeddings = model.encoder(static_pose, batch.valid_mask)
            real_readout = estimate_recurrence_carrier_curves(
                real_embeddings,
                minimum_period=config.period.minimum,
                maximum_period=config.period.maximum,
                valid_mask=batch.valid_mask,
                timeline_lengths=batch.lengths,
                maximum_mode=config.period.maximum_mode,
            )
            baseline = _readout_samples_from_batch(
                video_ids=batch.video_ids,
                embeddings=real_embeddings,
                valid_mask=batch.valid_mask,
                timeline_lengths=batch.lengths,
                readout=real_readout,
                config=config,
                minimum_pairs=minimum_pairs,
                counter=counter,
            )
            baseline_rows.extend(baseline)
            shuffled_embeddings = _deterministically_shuffle_valid_embeddings(
                real_embeddings, batch.valid_mask, batch.video_ids
            )
            representation_rows.extend(
                representation_samples_from_embeddings(
                    real_embeddings=real_embeddings,
                    pose_shuffle_embeddings=pose_shuffle_embeddings,
                    embedding_shuffle_embeddings=shuffled_embeddings,
                    zero_pose_embeddings=zero_pose_embeddings,
                    static_pose_embeddings=static_pose_embeddings,
                    valid_mask=batch.valid_mask,
                    timeline_lengths=batch.lengths,
                    periods=real_readout.periods,
                    minimum_pairs=minimum_pairs,
                )
            )
            shuffled_readout = build_recurrence_carrier_curves(
                shuffled_embeddings,
                real_readout.periods,
                batch.valid_mask,
                timeline_lengths=batch.lengths,
                period_confidences=real_readout.period_confidences,
            )
            for index, source in enumerate(baseline):
                shuffle_rows.append(
                    ShuffleControlSample(
                        real_carrier_eligible=source.carrier_eligible,
                        real_harmonic_energy_fraction=source.harmonic_energy_fraction,
                        shuffled_harmonic_energy_fraction=_finite_float(
                            shuffled_readout.harmonic_energy_fractions[index],
                            role="shuffled harmonic fraction",
                        ),
                        real_recurrence_gate_energy=source.recurrence_gate_energy,
                        shuffled_recurrence_gate_energy=_finite_float(
                            shuffled_readout.recurrence_gate_energies[index],
                            role="shuffled gate energy",
                        ),
                    )
                )

            zero_readout = estimate_recurrence_carrier_curves(
                zero_pose_embeddings,
                minimum_period=config.period.minimum,
                maximum_period=config.period.maximum,
                valid_mask=batch.valid_mask,
                timeline_lengths=batch.lengths,
                maximum_mode=config.period.maximum_mode,
            )
            static_readout = estimate_recurrence_carrier_curves(
                static_pose_embeddings,
                minimum_period=config.period.minimum,
                maximum_period=config.period.maximum,
                valid_mask=batch.valid_mask,
                timeline_lengths=batch.lengths,
                maximum_mode=config.period.maximum_mode,
            )
            zero_rows.extend(
                _pose_control_samples(
                    embeddings=zero_pose_embeddings,
                    readout=zero_readout,
                    valid_mask=batch.valid_mask,
                    timeline_lengths=batch.lengths,
                    minimum_pairs=minimum_pairs,
                )
            )
            static_rows.extend(
                _pose_control_samples(
                    embeddings=static_pose_embeddings,
                    readout=static_readout,
                    valid_mask=batch.valid_mask,
                    timeline_lengths=batch.lengths,
                    minimum_pairs=minimum_pairs,
                )
            )
    return (
        tuple(baseline_rows),
        tuple(representation_rows),
        tuple(shuffle_rows),
        tuple(zero_rows),
        tuple(static_rows),
    )


def _encode_readout_only(
    model: nn.Module,
    sequences: Sequence[PoseSequence],
    *,
    config: PAMSConfig,
    minimum_pairs: int,
    device: torch.device,
    batch_size: int,
) -> tuple[ReadoutSample, ...]:
    rows: list[ReadoutSample] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            embeddings = model.encoder(batch.poses, batch.valid_mask)
            rows.extend(
                readout_samples_from_embeddings(
                    video_ids=batch.video_ids,
                    embeddings=embeddings,
                    valid_mask=batch.valid_mask,
                    timeline_lengths=batch.lengths,
                    config=config,
                    minimum_pairs=minimum_pairs,
                )
            )
    return tuple(rows)


def _resample_for_time_scale(sequence: PoseSequence, factor: float) -> PoseSequence:
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("time-scale factor must be finite and positive")
    target_frames = int(round(sequence.num_frames * factor))
    if target_frames < 2:
        raise ValueError("time-scale factor produces fewer than two frames")
    xyz, valid = uniform_resample(
        sequence.xyz, sequence.valid_mask, target_frames=target_frames
    )
    return PoseSequence(
        video_id=f"{sequence.video_id}::time-scale-{format(factor, 'g')}",
        fps=sequence.fps * factor,
        xyz=xyz,
        valid_mask=valid,
    )


def time_scale_aggregates_from_samples(
    baseline: Sequence[ReadoutSample],
    scaled_by_factor: Mapping[float, Sequence[ReadoutSample]],
    *,
    factors: Sequence[float],
    minimum_period: int,
) -> dict[str, Any]:
    period_errors: list[float] = []
    peak_differences: list[int] = []
    comparison_total = 0
    for factor in factors:
        scaled = scaled_by_factor[float(factor)]
        if len(scaled) != len(baseline):
            raise ValueError("scaled readout total differs from baseline")
        for source, candidate in zip(baseline, scaled, strict=True):
            comparison_total += 1
            expected_period = source.period_frames * float(factor)
            period_eligible = (
                source.period_lag_eligible
                and candidate.period_lag_eligible
                and minimum_period <= expected_period <= candidate.period_upper_bound
            )
            if period_eligible:
                error = abs(candidate.period_frames - expected_period) / expected_period
                if not math.isfinite(error):
                    raise RuntimeError("time-scale period error is non-finite")
                period_errors.append(error)
            if source.carrier_eligible and candidate.carrier_eligible:
                peak_differences.append(
                    abs(candidate.selected_peak_total - source.selected_peak_total)
                )
    exact_total = sum(value == 0 for value in peak_differences)
    within_one_total = sum(value <= 1 for value in peak_differences)
    return {
        "factors": [float(value) for value in factors],
        "period": {
            "candidate_comparison_total": comparison_total,
            "eligible_comparison_total": len(period_errors),
            "eligible_comparison_share": (
                len(period_errors) / comparison_total if comparison_total else 0.0
            ),
            "relative_error": _summary(period_errors),
        },
        "peak_total": {
            "candidate_comparison_total": comparison_total,
            "eligible_comparison_total": len(peak_differences),
            "eligible_comparison_share": (
                len(peak_differences) / comparison_total if comparison_total else 0.0
            ),
            "absolute_difference": _summary(peak_differences),
            "exact_share": (
                exact_total / len(peak_differences) if peak_differences else 0.0
            ),
            "within_one_share": (
                within_one_total / len(peak_differences) if peak_differences else 0.0
            ),
        },
        "per_video_rows_persisted": False,
    }


def _time_scale_consistency(
    model: nn.Module,
    sequences: Sequence[PoseSequence],
    baseline: Sequence[ReadoutSample],
    *,
    config: PAMSConfig,
    specification: GateSpecification,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    scaled: dict[float, tuple[ReadoutSample, ...]] = {}
    position_encoding = getattr(getattr(model, "encoder", None), "position_encoding", None)
    encoding = getattr(position_encoding, "encoding", None)
    if not isinstance(encoding, Tensor) or encoding.ndim != 2:
        raise RuntimeError("encoder positional capacity is unavailable")
    positional_capacity = int(encoding.shape[0])
    for factor in specification.time_scale_factors:
        scaled_sequences = tuple(
            _resample_for_time_scale(sequence, factor) for sequence in sequences
        )
        if any(sequence.num_frames > positional_capacity for sequence in scaled_sequences):
            raise ValueError(
                "time-scaled train337 sequence exceeds frozen positional capacity"
            )
        scaled[float(factor)] = _encode_readout_only(
            model,
            scaled_sequences,
            config=config,
            minimum_pairs=specification.minimum_lag_pair_total,
            device=device,
            batch_size=batch_size,
        )
    return time_scale_aggregates_from_samples(
        baseline,
        scaled,
        factors=specification.time_scale_factors,
        minimum_period=config.period.minimum,
    )


def _median(summary: Mapping[str, Any]) -> float | None:
    value = summary.get("median")
    return float(value) if isinstance(value, int | float) else None


def _representation_distribution(
    samples: Sequence[RepresentationSample], *, near_collapse_rms: float
) -> dict[str, Any]:
    if not samples:
        raise ValueError("representation distribution requires samples")
    eligible = [sample for sample in samples if sample.lag_eligible]
    return {
        "record_total": len(samples),
        "temporal_rms": _summary([sample.temporal_rms for sample in samples]),
        "near_collapsed_share": float(
            np.mean([sample.temporal_rms <= near_collapse_rms for sample in samples])
        ),
        "lag_eligible_total": len(eligible),
        "lag_eligible_share": len(eligible) / len(samples),
        "real_cycle_margin": _summary(
            [
                sample.real_cycle_margin
                for sample in eligible
                if sample.real_cycle_margin is not None
            ]
        ),
        "pose_shuffle_cycle_margin": _summary(
            [
                sample.pose_shuffle_cycle_margin
                for sample in eligible
                if sample.pose_shuffle_cycle_margin is not None
            ]
        ),
        "embedding_shuffle_cycle_margin": _summary(
            [
                sample.embedding_shuffle_cycle_margin
                for sample in eligible
                if sample.embedding_shuffle_cycle_margin is not None
            ]
        ),
        "zero_pose_cycle_margin": _summary(
            [
                sample.zero_pose_cycle_margin
                for sample in eligible
                if sample.zero_pose_cycle_margin is not None
            ]
        ),
        "static_pose_cycle_margin": _summary(
            [
                sample.static_pose_cycle_margin
                for sample in eligible
                if sample.static_pose_cycle_margin is not None
            ]
        ),
        "real_minus_strongest_null": _summary(
            [
                sample.real_minus_strongest_null
                for sample in eligible
                if sample.real_minus_strongest_null is not None
            ]
        ),
        "real_beats_all_nulls_share": (
            float(
                np.mean(
                    [
                        bool(sample.real_beats_all_nulls)
                        for sample in eligible
                        if sample.real_beats_all_nulls is not None
                    ]
                )
            )
            if eligible
            else 0.0
        ),
    }


def _period_distribution(
    samples: Sequence[ReadoutSample], *, minimum_period: int
) -> dict[str, Any]:
    if not samples:
        raise ValueError("period distribution requires samples")
    periods = np.asarray([sample.period_frames for sample in samples], dtype=np.float64)
    boundary = np.asarray(
        [
            sample.period_frames <= float(minimum_period) + 1e-6
            or sample.period_frames >= sample.period_upper_bound - 1e-6
            for sample in samples
        ],
        dtype=np.bool_,
    )
    keys = [format(float(value), ".9g") for value in periods]
    frequencies = Counter(keys)
    mode_key, mode_total = min(
        frequencies.items(), key=lambda item: (-item[1], item[0])
    )
    return {
        "record_total": len(samples),
        "per_sample_upper_bound_used": True,
        "boundary_share": float(np.mean(boundary)),
        "minimum_boundary_share": float(
            np.mean(periods <= float(minimum_period) + 1e-6)
        ),
        "upper_boundary_share": float(
            np.mean(
                [
                    sample.period_frames >= sample.period_upper_bound - 1e-6
                    for sample in samples
                ]
            )
        ),
        "mode_period_frames": float(mode_key),
        "mode_frequency": mode_total,
        "mode_share": mode_total / len(samples),
        "unique_period_total": len(frequencies),
        "positive_confidence_share": float(
            np.mean([sample.period_confidence > 0.0 for sample in samples])
        ),
        "period_frames": _summary(periods),
        "period_confidence_non_authorizing": _summary(
            [sample.period_confidence for sample in samples]
        ),
        "period_histogram_aggregate_only": dict(sorted(frequencies.items())),
    }


def _readout_distribution(samples: Sequence[ReadoutSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("readout distribution requires samples")
    eligible = [sample for sample in samples if sample.carrier_eligible]
    majority = [sample.selection_rule == "majority_first" for sample in eligible]
    return {
        "record_total": len(samples),
        "carrier_eligible_available_total": len(eligible),
        "carrier_eligible_available_share": len(eligible) / len(samples),
        "raw_carrier_available_share_non_authorizing": float(
            np.mean([sample.raw_carrier_available for sample in samples])
        ),
        "expert_majority_share": float(np.mean(majority)) if majority else 0.0,
        "expert_fft_nearest_fallback_share": (
            float(np.mean(np.logical_not(majority))) if majority else 0.0
        ),
        "selected_vs_active_reference_absolute_gap": _summary(
            [
                abs(
                    sample.selected_peak_total
                    - sample.active_reference_peak_total
                )
                for sample in eligible
            ]
        ),
        "normalized_curve_rms_non_authorizing": _summary(
            [sample.normalized_curve_rms for sample in samples]
        ),
        "raw_curve_rms_non_authorizing": _summary(
            [sample.raw_curve_rms for sample in samples]
        ),
        "harmonic_energy_fraction_non_authorizing": _summary(
            [sample.harmonic_energy_fraction for sample in samples]
        ),
        "active_support_fraction_non_authorizing": _summary(
            [sample.active_support_fraction for sample in samples]
        ),
        "recurrence_gate_energy_non_authorizing": _summary(
            [sample.recurrence_gate_energy for sample in samples]
        ),
    }


def _peak_total_distribution(samples: Sequence[ReadoutSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("peak-total distribution requires samples")
    values = np.asarray(
        [sample.selected_peak_total for sample in samples], dtype=np.int64
    )
    frequencies = Counter(int(value) for value in values)
    mode_value, mode_total = min(
        frequencies.items(), key=lambda item: (-item[1], item[0])
    )
    return {
        "record_total": len(samples),
        "zero_share": float(np.mean(values == 0)),
        "mode_peak_total": mode_value,
        "mode_frequency": mode_total,
        "mode_share": mode_total / len(samples),
        "unique_peak_total": len(frequencies),
        "absolute_distribution_non_authorizing": _summary(values),
        "histogram_aggregate_only": {
            str(key): value for key, value in sorted(frequencies.items())
        },
    }


def _ratio(numerator: Sequence[float], denominator: Sequence[float]) -> float | None:
    numerator_median = _median(_summary(numerator))
    denominator_median = _median(_summary(denominator))
    if numerator_median is None or denominator_median is None or denominator_median <= 1e-12:
        return None
    return numerator_median / denominator_median


def _shuffle_distribution(samples: Sequence[ShuffleControlSample]) -> dict[str, Any]:
    eligible = [sample for sample in samples if sample.real_carrier_eligible]
    real_harmonic = [sample.real_harmonic_energy_fraction for sample in eligible]
    shuffled_harmonic = [sample.shuffled_harmonic_energy_fraction for sample in eligible]
    real_gate = [sample.real_recurrence_gate_energy for sample in eligible]
    shuffled_gate = [sample.shuffled_recurrence_gate_energy for sample in eligible]
    return {
        "real_eligibility_sets_denominator": True,
        "real_eligible_total": len(eligible),
        "real_harmonic_energy_fraction": _summary(real_harmonic),
        "shuffled_harmonic_energy_fraction": _summary(shuffled_harmonic),
        "shuffled_to_real_harmonic_median_ratio": _ratio(
            shuffled_harmonic, real_harmonic
        ),
        "real_recurrence_gate_energy": _summary(real_gate),
        "shuffled_recurrence_gate_energy": _summary(shuffled_gate),
        "shuffled_to_real_gate_energy_median_ratio": _ratio(
            shuffled_gate, real_gate
        ),
    }


def _pose_control_distribution(samples: Sequence[PoseControlSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("pose control distribution requires samples")
    return {
        "record_total": len(samples),
        "positive_period_confidence_share": float(
            np.mean([sample.positive_period_confidence for sample in samples])
        ),
        "carrier_available_share": float(
            np.mean([sample.carrier_eligible for sample in samples])
        ),
        "positive_support_share": float(
            np.mean([sample.positive_support for sample in samples])
        ),
    }


def _criterion(value: float | None, *, operator: str, threshold: float) -> dict[str, Any]:
    passed = False
    if value is not None and math.isfinite(value):
        if operator == ">=":
            passed = value >= threshold
        elif operator == "<=":
            passed = value <= threshold
        elif operator == "<":
            passed = value < threshold
        else:
            raise ValueError("unsupported terminal gate operator")
    return {"value": value, "operator": operator, "threshold": threshold, "pass": passed}


def gate_decision(
    *,
    candidate_id: str,
    aggregates: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    representation = aggregates["representation"]
    period = aggregates["period"]
    readout = aggregates["readout"]
    peak_total = aggregates["peak_total"]
    scale = aggregates["time_scale"]
    shuffle = aggregates["embedding_shuffle_fixed_real_period"]
    zero = aggregates["zero_pose"]
    static = aggregates["static_pose"]
    criteria = {
        "near_collapsed_share": _criterion(
            float(representation["near_collapsed_share"]),
            operator="<=",
            threshold=thresholds["near_collapsed_share_maximum"],
        ),
        "lag_eligible_share": _criterion(
            float(representation["lag_eligible_share"]),
            operator=">=",
            threshold=thresholds["lag_eligible_share_minimum"],
        ),
        "real_cycle_margin_median": _criterion(
            _median(representation["real_cycle_margin"]),
            operator=">=",
            threshold=thresholds["real_cycle_margin_median_minimum"],
        ),
        "real_minus_strongest_null_median": _criterion(
            _median(representation["real_minus_strongest_null"]),
            operator=">=",
            threshold=thresholds["real_minus_strongest_null_median_minimum"],
        ),
        "real_beats_all_nulls_share": _criterion(
            float(representation["real_beats_all_nulls_share"]),
            operator=">=",
            threshold=thresholds["real_beats_all_nulls_share_minimum"],
        ),
        "period_boundary_share": _criterion(
            float(period["boundary_share"]),
            operator="<",
            threshold=thresholds["period_boundary_share_maximum_exclusive"],
        ),
        "period_mode_share": _criterion(
            float(period["mode_share"]),
            operator="<",
            threshold=thresholds["period_mode_share_maximum_exclusive"],
        ),
        "period_time_scale_eligible_share": _criterion(
            float(scale["period"]["eligible_comparison_share"]),
            operator=">=",
            threshold=thresholds["period_time_scale_eligible_share_minimum"],
        ),
        "period_time_scale_median_relative_error": _criterion(
            _median(scale["period"]["relative_error"]),
            operator="<=",
            threshold=thresholds[
                "period_time_scale_median_relative_error_maximum"
            ],
        ),
        "carrier_eligible_available_share": _criterion(
            float(readout["carrier_eligible_available_share"]),
            operator=">=",
            threshold=thresholds["carrier_eligible_available_share_minimum"],
        ),
        "embedding_shuffle_harmonic_median_ratio": _criterion(
            shuffle["shuffled_to_real_harmonic_median_ratio"],
            operator="<=",
            threshold=thresholds[
                "embedding_shuffle_harmonic_median_ratio_maximum"
            ],
        ),
        "embedding_shuffle_gate_energy_median_ratio": _criterion(
            shuffle["shuffled_to_real_gate_energy_median_ratio"],
            operator="<=",
            threshold=thresholds[
                "embedding_shuffle_gate_energy_median_ratio_maximum"
            ],
        ),
        "zero_pose_positive_period_confidence_share": _criterion(
            float(zero["positive_period_confidence_share"]),
            operator="<=",
            threshold=thresholds[
                "zero_pose_positive_period_confidence_share_maximum"
            ],
        ),
        "zero_pose_carrier_available_share": _criterion(
            float(zero["carrier_available_share"]),
            operator="<=",
            threshold=thresholds["zero_pose_carrier_available_share_maximum"],
        ),
        "zero_pose_positive_support_share": _criterion(
            float(zero["positive_support_share"]),
            operator="<=",
            threshold=thresholds["zero_pose_positive_support_share_maximum"],
        ),
        "static_pose_positive_period_confidence_share": _criterion(
            float(static["positive_period_confidence_share"]),
            operator="<=",
            threshold=thresholds[
                "static_pose_positive_period_confidence_share_maximum"
            ],
        ),
        "static_pose_carrier_available_share": _criterion(
            float(static["carrier_available_share"]),
            operator="<=",
            threshold=thresholds["static_pose_carrier_available_share_maximum"],
        ),
        "static_pose_positive_support_share": _criterion(
            float(static["positive_support_share"]),
            operator="<=",
            threshold=thresholds["static_pose_positive_support_share_maximum"],
        ),
        "expert_majority_share": _criterion(
            float(readout["expert_majority_share"]),
            operator=">=",
            threshold=thresholds["expert_majority_share_minimum"],
        ),
        "peak_total_zero_share": _criterion(
            float(peak_total["zero_share"]),
            operator="<",
            threshold=thresholds["peak_total_zero_share_maximum_exclusive"],
        ),
        "peak_total_mode_share": _criterion(
            float(peak_total["mode_share"]),
            operator="<",
            threshold=thresholds["peak_total_mode_share_maximum_exclusive"],
        ),
        "selected_vs_active_reference_gap_median": _criterion(
            _median(readout["selected_vs_active_reference_absolute_gap"]),
            operator="<=",
            threshold=thresholds[
                "selected_vs_active_reference_gap_median_maximum"
            ],
        ),
        "time_scale_peak_total_eligible_share": _criterion(
            float(scale["peak_total"]["eligible_comparison_share"]),
            operator=">=",
            threshold=thresholds[
                "time_scale_peak_total_eligible_share_minimum"
            ],
        ),
        "time_scale_peak_total_exact_share": _criterion(
            float(scale["peak_total"]["exact_share"]),
            operator=">=",
            threshold=thresholds["time_scale_peak_total_exact_share_minimum"],
        ),
        "time_scale_peak_total_within_one_share": _criterion(
            float(scale["peak_total"]["within_one_share"]),
            operator=">=",
            threshold=thresholds[
                "time_scale_peak_total_within_one_share_minimum"
            ],
        ),
    }
    # Criterion names deliberately omit threshold suffixes.  The one-to-one
    # total check catches accidental threshold removal without coupling names.
    if len(criteria) != len(_THRESHOLD_KEYS):
        raise RuntimeError("terminal criterion total differs from frozen thresholds")
    passed = all(bool(item["pass"]) for item in criteria.values())
    next_candidate = None
    if not passed and candidate_id != _CANDIDATE_ORDER[-1]:
        next_candidate = _CANDIDATE_ORDER[_CANDIDATE_ORDER.index(candidate_id) + 1]
    return {
        "thresholds_frozen_before_candidate_a": True,
        "criteria": criteria,
        "criterion_total": len(criteria),
        "all_hard_criteria_pass": passed,
        "overall_pass": passed,
        "eligible_for_single_frozen_dev84_protocol_build": passed,
        "dev84_identity_media_pose_or_scoring_authorized": False,
        "test105_evaluation_authorized": False,
        "next_candidate_training_authorized": next_candidate,
        "all_candidates_exhausted": bool(not passed and candidate_id == "C"),
    }


def _validate_source_receipt_binding(path: Path, *, sha256: str) -> None:
    environment_path = os.environ.get("PAMS_SOURCE_EXPORT_RECEIPT", "").strip()
    environment_sha256 = os.environ.get(
        "PAMS_SOURCE_EXPORT_RECEIPT_SHA256", ""
    ).strip()
    if not environment_path or not environment_sha256:
        raise RuntimeError("source-export receipt environment is incomplete")
    if Path(environment_path).resolve(strict=True) != path.resolve(strict=True):
        raise RuntimeError("source-receipt argument differs from runtime binding")
    if sha256 != environment_sha256:
        raise RuntimeError("source-receipt bytes differ from runtime binding")


def _validate_encoder_completion_receipt(
    path: Path,
    *,
    expected_source_git_sha: str,
    specification: GateSpecification,
    config: PAMSConfig,
    identities: Mapping[str, tuple[str, int]],
    epoch11_artifact: Mapping[str, Any],
    epoch11_receipt: Mapping[str, Any],
) -> tuple[CompletedRunReceipt, Path, tuple[str, int]]:
    receipt = CompletedRunReceipt.model_validate(
        _strict_json(path, document="encoder completion receipt")
    )
    if receipt.schema_version != 3 or receipt.status != "completed":
        raise ValueError("encoder completion receipt must be schema-v3 completed")
    if receipt.started.git_sha != expected_source_git_sha:
        raise ValueError("encoder completion receipt source revision mismatch")
    if receipt.started.config_sha256 != config.fingerprint:
        raise ValueError("encoder completion receipt config fingerprint mismatch")
    if receipt.started.protocol != specification.expected_protocol:
        raise ValueError("encoder completion receipt protocol mismatch")
    if receipt.started.seed != specification.expected_seed:
        raise ValueError("encoder completion receipt seed mismatch")
    if receipt.metrics.get("completed_epochs") != specification.expected_completed_epochs:
        raise ValueError("encoder completion receipt terminal epoch mismatch")
    artifacts = {artifact.role: artifact for artifact in receipt.artifacts}
    required = {
        "input_config",
        "input_pose_cache_snapshot",
        "input_resume_checkpoint",
        "input_resume_progress",
        "output_encoder_checkpoint",
        "progress_log",
    }
    if not required <= set(artifacts):
        raise ValueError(
            "encoder completion receipt is missing required artifact roles: "
            + repr(sorted(required - set(artifacts)))
        )
    epoch11_inputs = epoch11_artifact["inputs"]
    expected_artifacts = {
        "input_config": identities["experiment_config"],
        "input_pose_cache_snapshot": identities["pose_snapshot"],
        "input_resume_checkpoint": (
            epoch11_receipt["encoder_checkpoint_sha256"],
            epoch11_inputs["encoder_checkpoint_bytes"],
        ),
        "input_resume_progress": (
            epoch11_receipt["encoder_progress_sha256"],
            epoch11_inputs["encoder_progress_bytes"],
        ),
        "output_encoder_checkpoint": identities["encoder_checkpoint"],
        "progress_log": identities["encoder_progress"],
    }
    for role, expected_identity in expected_artifacts.items():
        actual_identity = (artifacts[role].sha256, artifacts[role].bytes)
        if actual_identity != expected_identity:
            raise ValueError(f"encoder completion receipt {role} identity mismatch")
    started_path = path.with_name(f"{receipt.run_id}.started.json")
    started_identity = _stable_file_sha256(started_path)
    started_payload = _strict_json(started_path, document="encoder started receipt")
    if started_payload != receipt.started.model_dump(mode="json"):
        raise ValueError("encoder started receipt bytes differ from embedded receipt")
    if started_identity[0] != receipt.start_manifest_sha256:
        raise ValueError("encoder started receipt SHA-256 mismatch")
    return receipt, started_path, started_identity


def _validate_completion_provenance(
    receipt: CompletedRunReceipt,
    provenance: Any,
) -> None:
    if receipt.started.dataset_sha256 != provenance.dataset_fingerprint:
        raise ValueError("encoder completion receipt dataset fingerprint mismatch")
    container = receipt.started.hardware.get("container")
    if not isinstance(container, Mapping):
        raise ValueError("encoder completion receipt lacks formal container provenance")
    expected_container = {
        "image_id": provenance.container_image_id,
        "environment_sha256": provenance.container_environment_sha256,
        "source_revision": provenance.source_git_sha,
    }
    for key, expected in expected_container.items():
        if container.get(key) != expected:
            raise ValueError(f"encoder completion receipt container {key} mismatch")


def _require_exact_mapping(
    value: Any,
    expected_keys: set[str],
    *,
    role: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError(f"{role} schema mismatch")
    return value


def _validate_summary_mapping(value: Any, *, role: str) -> None:
    summary = _require_exact_mapping(
        value,
        {
            "observations",
            "minimum",
            "p10",
            "median",
            "mean",
            "p90",
            "maximum",
            "standard_deviation",
        },
        role=role,
    )
    observations = summary["observations"]
    if type(observations) is not int or observations < 0:
        raise ValueError(f"{role}.observations must be a non-negative integer")
    for key in (
        "minimum",
        "p10",
        "median",
        "mean",
        "p90",
        "maximum",
        "standard_deviation",
    ):
        item = summary[key]
        if observations == 0:
            if item is not None:
                raise ValueError(f"{role}.{key} must be null without observations")
        elif (
            isinstance(item, bool)
            or not isinstance(item, int | float)
            or not math.isfinite(float(item))
        ):
            raise ValueError(f"{role}.{key} must be finite")


def _validate_epoch11_nested_artifact(
    artifact: Mapping[str, Any],
    *,
    specification: GateSpecification,
    config: PAMSConfig,
) -> None:
    label_firewall = _require_exact_mapping(
        artifact["label_firewall"],
        {
            "accepted_scientific_inputs",
            "manifest_interface_supported",
            "media_interface_supported",
            "external_label_fields_accessed",
            "privileged_interface_fields_and_paths_rejected",
            "training_interface_supported",
        },
        role="epoch11 label firewall",
    )
    if (
        label_firewall["manifest_interface_supported"] is not False
        or label_firewall["media_interface_supported"] is not False
        or label_firewall["external_label_fields_accessed"] != []
        or label_firewall["privileged_interface_fields_and_paths_rejected"] is not True
        or label_firewall["training_interface_supported"] is not False
    ):
        raise ValueError("epoch11 label firewall is not fail-closed")

    schedule = _require_exact_mapping(
        artifact["schedule"],
        {
            "completed_epochs",
            "period_source",
            "loss_first_three_median",
            "loss_final_three_median",
            "loss_relative_drop",
            "fixed_period_evidence_fraction_mean",
            "optimizer_steps_total",
            "prototype_bank_negative_tallies_all_zero",
            "progress_exactly_matches_checkpoint_history",
        },
        role="epoch11 schedule",
    )
    if (
        schedule["completed_epochs"] != 11
        or schedule["period_source"] != "fixed_period_inferred"
        or type(schedule["optimizer_steps_total"]) is not int
        or schedule["optimizer_steps_total"] < 11
        or schedule["prototype_bank_negative_tallies_all_zero"] is not True
        or schedule["progress_exactly_matches_checkpoint_history"] is not True
    ):
        raise ValueError("epoch11 schedule does not prove the frozen resume boundary")
    for key in (
        "loss_first_three_median",
        "loss_final_three_median",
        "loss_relative_drop",
        "fixed_period_evidence_fraction_mean",
    ):
        _finite_number(schedule[key], name=f"epoch11 schedule.{key}")

    representation = _require_exact_mapping(
        artifact["representation"],
        {
            "record_total",
            "temporal_rms",
            "near_collapse_rms",
            "near_collapsed_fraction",
            "lag_eligible_total",
            "lag_eligible_fraction",
            "real_cycle_margin",
            "shuffled_cycle_margin",
            "zero_cycle_margin",
            "stronger_null_separation",
            "real_beats_both_nulls_fraction",
            "fixed_period_carrier_advisory",
        },
        role="epoch11 representation",
    )
    if representation["record_total"] != specification.expected_training_video_total:
        raise ValueError("epoch11 representation does not contain train337")
    if representation["near_collapse_rms"] != specification.near_collapse_rms:
        raise ValueError("epoch11 collapse threshold differs from terminal preregistration")
    if (
        type(representation["lag_eligible_total"]) is not int
        or not 0 <= representation["lag_eligible_total"] <= representation["record_total"]
    ):
        raise ValueError("epoch11 lag eligibility total is invalid")
    for key in (
        "near_collapsed_fraction",
        "lag_eligible_fraction",
        "real_beats_both_nulls_fraction",
    ):
        value = _finite_number(representation[key], name=f"epoch11 representation.{key}")
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"epoch11 representation.{key} must lie in [0, 1]")
    for key in (
        "temporal_rms",
        "real_cycle_margin",
        "shuffled_cycle_margin",
        "zero_cycle_margin",
        "stronger_null_separation",
    ):
        _validate_summary_mapping(
            representation[key], role=f"epoch11 representation.{key}"
        )
    advisory = _require_exact_mapping(
        representation["fixed_period_carrier_advisory"],
        {
            "authorization_role",
            "real_available_fraction",
            "real_support",
            "real_gate_energy",
            "shuffled_available_fraction",
            "shuffled_support",
            "shuffled_gate_energy",
            "zero_available_fraction",
            "zero_support",
            "zero_gate_energy",
            "shuffled_to_real_gate_energy_median_ratio",
        },
        role="epoch11 fixed-period carrier advisory",
    )
    if advisory["authorization_role"] != "diagnostic_only":
        raise ValueError("epoch11 carrier advisory became authorizing")
    for key in (
        "real_support",
        "real_gate_energy",
        "shuffled_support",
        "shuffled_gate_energy",
        "zero_support",
        "zero_gate_energy",
    ):
        _validate_summary_mapping(
            advisory[key], role=f"epoch11 carrier advisory.{key}"
        )

    gate = _require_exact_mapping(
        artifact["gate"],
        {
            "thresholds_frozen_before_native_candidate_training",
            "criteria",
            "all_core_criteria_pass",
            "encoder_continuation_authorized",
            "prediction_or_scoring_authorized",
        },
        role="epoch11 gate decision",
    )
    criterion_names = {
        "loss_relative_drop",
        "fixed_period_evidence_fraction",
        "near_collapsed_fraction",
        "lag_eligible_fraction",
        "real_cycle_margin_median",
        "stronger_null_separation_median",
        "real_beats_both_nulls_fraction",
    }
    criteria = _require_exact_mapping(
        gate["criteria"], criterion_names, role="epoch11 gate criteria"
    )
    for name, criterion in criteria.items():
        row = _require_exact_mapping(
            criterion,
            {"value", "operator", "threshold", "pass"},
            role=f"epoch11 criterion {name}",
        )
        if row["pass"] is not True or row["operator"] not in {">=", "<="}:
            raise ValueError(f"epoch11 criterion {name} did not pass canonically")
        _finite_number(row["value"], name=f"epoch11 criterion {name}.value")
        _finite_number(row["threshold"], name=f"epoch11 criterion {name}.threshold")
    if (
        gate["thresholds_frozen_before_native_candidate_training"] is not True
        or gate["all_core_criteria_pass"] is not True
        or gate["encoder_continuation_authorized"] is not True
        or gate["prediction_or_scoring_authorized"] is not False
    ):
        raise ValueError("epoch11 gate decision is not a passing continuation receipt")

    scientific_scope = _require_exact_mapping(
        artifact["scientific_scope"],
        {
            "fixed_training_period_frames",
            "dense_frame_indices_preserved",
            "invalid_slots_never_compacted",
            "real_shuffled_and_zero_views_share_one_valid_mask",
            "fixed_period_carrier_is_advisory_at_epoch11",
            "adaptive_time_scale_criterion_used",
        },
        role="epoch11 scientific scope",
    )
    if scientific_scope != {
        "fixed_training_period_frames": config.period.fixed_period_frames,
        "dense_frame_indices_preserved": True,
        "invalid_slots_never_compacted": True,
        "real_shuffled_and_zero_views_share_one_valid_mask": True,
        "fixed_period_carrier_is_advisory_at_epoch11": True,
        "adaptive_time_scale_criterion_used": False,
    }:
        raise ValueError("epoch11 scientific scope differs from the frozen candidate")

    read_only = _require_exact_mapping(
        artifact["read_only_verification"],
        {
            "all_file_inputs_unchanged",
            "pose_cache_set_unchanged",
            "model_state_sha256_before",
            "model_state_sha256_after",
            "model_or_optimizer_state_updated",
            "training_steps_executed",
            "pose_cache_write_operations",
        },
        role="epoch11 read-only verification",
    )
    if (
        read_only["all_file_inputs_unchanged"] is not True
        or read_only["pose_cache_set_unchanged"] is not True
        or read_only["model_state_sha256_before"]
        != read_only["model_state_sha256_after"]
        or read_only["model_or_optimizer_state_updated"] is not False
        or read_only["training_steps_executed"] != 0
        or read_only["pose_cache_write_operations"] != 0
    ):
        raise ValueError("epoch11 artifact is not read-only and immutable")


def _validate_epoch11_gate_pair(
    artifact_path: Path,
    receipt_path: Path,
    *,
    config: PAMSConfig,
    specification: GateSpecification,
    config_sha256: str,
    pose_snapshot_sha256: str,
    pose_cache_set_sha256: str,
    source_git_sha: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = _strict_json(artifact_path, document="epoch11 gate artifact")
    expected_artifact_keys = {
        "schema_version",
        "artifact_type",
        "status",
        "classification",
        "protocol",
        "seed",
        "label_firewall",
        "inputs",
        "schedule",
        "representation",
        "gate",
        "scientific_scope",
        "read_only_verification",
    }
    if set(artifact) != expected_artifact_keys:
        raise ValueError("epoch11 gate artifact schema mismatch")
    if (
        artifact["schema_version"] != 1
        or artifact["artifact_type"] != _EXPECTED_EPOCH11_ARTIFACT_TYPE
        or artifact["status"] != "encoder_continuation_authorized"
        or artifact["protocol"] != specification.expected_protocol
        or artifact["seed"] != specification.expected_seed
    ):
        raise ValueError("epoch11 gate artifact does not authorize this protocol")
    gate = artifact["gate"]
    if (
        not isinstance(gate, dict)
        or gate.get("encoder_continuation_authorized") is not True
        or gate.get("prediction_or_scoring_authorized") is not False
        or gate.get("all_core_criteria_pass") is not True
    ):
        raise ValueError("epoch11 gate did not pass its frozen mechanism criteria")
    _validate_epoch11_nested_artifact(
        artifact,
        specification=specification,
        config=config,
    )
    inputs = artifact["inputs"]
    inputs = _require_exact_mapping(
        inputs,
        {
            "encoder_checkpoint_sha256",
            "encoder_progress_sha256",
            "experiment_config_sha256",
            "gate_specification_sha256",
            "pose_snapshot_sha256",
            "encoder_checkpoint_bytes",
            "encoder_progress_bytes",
            "experiment_config_bytes",
            "gate_specification_bytes",
            "pose_snapshot_bytes",
            "config_fingerprint",
            "pose_fingerprint",
            "pose_cache_set_sha256",
            "training_video_total",
            "training_video_ids_sha256",
            "source_git_sha",
            "source_receipt_covered_paths",
            "container_image_id",
            "container_environment_sha256",
        },
        role="epoch11 gate inputs",
    )
    for role in ("encoder_checkpoint", "encoder_progress"):
        if not isinstance(inputs.get(f"{role}_sha256"), str):
            raise ValueError(f"epoch11 gate lacks {role} SHA-256")
        if (
            isinstance(inputs.get(f"{role}_bytes"), bool)
            or not isinstance(inputs.get(f"{role}_bytes"), int)
            or inputs[f"{role}_bytes"] < 1
        ):
            raise ValueError(f"epoch11 gate lacks canonical {role} byte total")
    expected_inputs = {
        "experiment_config_sha256": config_sha256,
        "pose_snapshot_sha256": pose_snapshot_sha256,
        "config_fingerprint": config.fingerprint,
        "pose_fingerprint": config.pose_fingerprint,
        "pose_cache_set_sha256": pose_cache_set_sha256,
        "training_video_total": specification.expected_training_video_total,
        "source_git_sha": source_git_sha,
    }
    for key, expected in expected_inputs.items():
        if inputs.get(key) != expected:
            raise ValueError(f"epoch11 gate input binding mismatch: {key}")

    receipt = _strict_json(receipt_path, document="epoch11 gate receipt")
    expected_receipt_keys = {
        "schema_version",
        "artifact_type",
        "artifact_sha256",
        "artifact_bytes",
        "artifact_status",
        "encoder_continuation_authorized",
        "encoder_checkpoint_sha256",
        "encoder_progress_sha256",
        "pose_cache_set_sha256",
        "gate_specification_sha256",
        "source_git_sha",
    }
    if set(receipt) != expected_receipt_keys:
        raise ValueError("epoch11 gate receipt schema mismatch")
    artifact_identity = _stable_file_sha256(artifact_path)
    if (
        receipt["schema_version"] != 1
        or receipt["artifact_type"] != _EXPECTED_EPOCH11_RECEIPT_TYPE
        or receipt["artifact_sha256"] != artifact_identity[0]
        or receipt["artifact_bytes"] != artifact_identity[1]
        or receipt["artifact_status"] != artifact["status"]
        or receipt["encoder_continuation_authorized"] is not True
        or receipt["encoder_checkpoint_sha256"]
        != inputs["encoder_checkpoint_sha256"]
        or receipt["encoder_progress_sha256"] != inputs["encoder_progress_sha256"]
        or receipt["pose_cache_set_sha256"] != pose_cache_set_sha256
        or receipt["gate_specification_sha256"]
        != inputs["gate_specification_sha256"]
        or receipt["source_git_sha"] != source_git_sha
    ):
        raise ValueError("epoch11 gate receipt does not bind the passing artifact")
    return artifact, receipt


def _validate_prior_rejection_pair(
    artifact_path: Path,
    receipt_path: Path,
    *,
    expected_candidate_id: str,
    prior_pairs: Sequence[Mapping[str, Any]],
    gate_specification_sha256: str,
    pose_cache_set_sha256: str,
    source_git_sha: str,
) -> dict[str, Any]:
    artifact_identity = _stable_file_sha256(artifact_path)
    receipt_identity = _stable_file_sha256(receipt_path)
    artifact = _strict_json(artifact_path, document="prior scientific rejection artifact")
    if set(artifact) != _OUTPUT_KEYS:
        raise ValueError("prior scientific rejection artifact schema mismatch")
    if (
        artifact["schema_version"] != _ARTIFACT_SCHEMA_VERSION
        or artifact["artifact_type"] != _EXPECTED_ARTIFACT_TYPE
        or artifact["status"] != "scientific_rejection"
        or artifact["candidate"].get("id") != expected_candidate_id
        or artifact["gate"].get("overall_pass") is not False
        or artifact["gate"].get(
            "eligible_for_single_frozen_dev84_protocol_build"
        )
        is not False
        or artifact["inputs"].get("gate_specification_sha256")
        != gate_specification_sha256
        or artifact["inputs"].get("pose_cache_set_sha256")
        != pose_cache_set_sha256
        or artifact["inputs"].get("source_git_sha") != source_git_sha
    ):
        raise ValueError("prior artifact is not the required scientific rejection")
    declared_prior = artifact["candidate"].get("prior_scientific_rejections")
    if declared_prior != list(prior_pairs):
        raise ValueError("prior scientific rejection chain is not prefix-complete")
    receipt = _strict_json(receipt_path, document="prior scientific rejection receipt")
    if set(receipt) != _RECEIPT_KEYS:
        raise ValueError("prior scientific rejection receipt schema mismatch")
    next_candidate = (
        _CANDIDATE_ORDER[_CANDIDATE_ORDER.index(expected_candidate_id) + 1]
        if expected_candidate_id != "C"
        else None
    )
    if (
        receipt["schema_version"] != 1
        or receipt["artifact_type"] != _EXPECTED_RECEIPT_TYPE
        or receipt["artifact_locator"] != artifact_path.name
        or receipt["artifact_sha256"] != artifact_identity[0]
        or receipt["artifact_bytes"] != artifact_identity[1]
        or receipt["artifact_status"] != "scientific_rejection"
        or receipt["candidate_id"] != expected_candidate_id
        or receipt["overall_pass"] is not False
        or receipt["eligible_for_single_frozen_dev84_protocol_build"] is not False
        or receipt["next_candidate_training_authorized"] != next_candidate
        or receipt["gate_specification_sha256"] != gate_specification_sha256
        or receipt["pose_cache_set_sha256"] != pose_cache_set_sha256
        or receipt["source_git_sha"] != source_git_sha
        or receipt["prior_scientific_rejections_sha256"]
        != sha256_json(declared_prior)
        or receipt["aggregate_only"] is not True
        or receipt["dev84_pose_or_scoring_authorized"] is not False
        or receipt["test105_evaluation_authorized"] is not False
    ):
        raise ValueError("prior scientific rejection receipt binding mismatch")
    receipt_to_input = {
        "encoder_checkpoint_sha256": "encoder_checkpoint_sha256",
        "encoder_progress_sha256": "encoder_progress_sha256",
        "encoder_completion_receipt_sha256": "encoder_completion_receipt_sha256",
        "encoder_started_receipt_sha256": "encoder_started_receipt_sha256",
        "source_export_receipt_sha256": "source_export_receipt_sha256",
        "experiment_config_sha256": "experiment_config_sha256",
        "gate_specification_sha256": "gate_specification_sha256",
        "pose_snapshot_sha256": "pose_snapshot_sha256",
        "epoch11_gate_artifact_sha256": "epoch11_gate_artifact_sha256",
        "epoch11_gate_receipt_sha256": "epoch11_gate_receipt_sha256",
        "code_files_sha256_commitment": "code_files_sha256_commitment",
    }
    if any(
        receipt[receipt_key] != artifact["inputs"].get(input_key)
        for receipt_key, input_key in receipt_to_input.items()
    ):
        raise ValueError("prior receipt differs from its artifact input commitments")
    return {
        "candidate_id": expected_candidate_id,
        "artifact_sha256": artifact_identity[0],
        "receipt_sha256": receipt_identity[0],
    }


def _validate_predecessor_chain(
    *,
    candidate_id: str,
    profile: CandidateProfile,
    artifact_paths: Sequence[Path],
    receipt_paths: Sequence[Path],
    gate_specification_sha256: str,
    pose_cache_set_sha256: str,
    source_git_sha: str,
) -> tuple[dict[str, Any], ...]:
    required = profile.required_prior_scientific_rejections
    if len(artifact_paths) != len(receipt_paths) or len(artifact_paths) != len(required):
        raise ValueError(
            f"candidate {candidate_id} requires exactly {len(required)} prior "
            "scientific-rejection artifact/receipt pairs"
        )
    chain: list[dict[str, Any]] = []
    for expected, artifact_path, receipt_path in zip(
        required, artifact_paths, receipt_paths, strict=True
    ):
        chain.append(
            _validate_prior_rejection_pair(
                artifact_path,
                receipt_path,
                expected_candidate_id=expected,
                prior_pairs=chain,
                gate_specification_sha256=gate_specification_sha256,
                pose_cache_set_sha256=pose_cache_set_sha256,
                source_git_sha=source_git_sha,
            )
        )
    return tuple(chain)


def _terminal_schedule_metadata(
    checkpoint: Path,
    progress: Path,
    *,
    config: PAMSConfig,
    specification: GateSpecification,
) -> dict[str, Any]:
    compatibility = _epoch11.GateSpecification(
        artifact_type="terminal-schedule-only",
        classification="terminal-schedule-only",
        expected_protocol=specification.expected_protocol,
        expected_seed=specification.expected_seed,
        expected_training_video_total=specification.expected_training_video_total,
        expected_completed_epochs=specification.expected_completed_epochs,
        minimum_lag_pair_total=specification.minimum_lag_pair_total,
        near_collapse_rms=specification.near_collapse_rms,
        thresholds={},
    )
    schedule, _ = _epoch11._checkpoint_epoch_metadata(
        checkpoint, progress, config=config, specification=compatibility
    )
    return schedule


def _require_unchanged(
    paths: Mapping[str, Path], identities: Mapping[str, tuple[str, int]]
) -> None:
    current = {name: _stable_file_sha256(path) for name, path in paths.items()}
    if current != dict(identities):
        raise RuntimeError("terminal gate file input changed during evaluation")


def _preflight_progress_jsonl(path: Path) -> None:
    """Reject privileged, duplicate, malformed, or non-finite rows before torch.load."""

    encoded = path.read_text(encoding="utf-8")
    _epoch11._reject_json_object_keys_before_deserialization(
        encoded,
        forbidden=_FORBIDDEN_FIELD_NAMES,
        document_name="encoder progress",
    )
    lines = encoded.splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("encoder progress must contain only non-empty JSONL rows")
    for line_number, line in enumerate(lines, start=1):
        _epoch11._strict_json(
            line,
            document=f"encoder progress row {line_number}",
        )


def run_gate(
    encoder_checkpoint_path: str | Path,
    encoder_progress_path: str | Path,
    encoder_completion_receipt_path: str | Path,
    source_receipt_path: str | Path,
    config_path: str | Path,
    gate_specification_path: str | Path,
    epoch11_gate_artifact_path: str | Path,
    epoch11_gate_receipt_path: str | Path,
    pose_cache_dir: str | Path,
    pose_snapshot_path: str | Path,
    *,
    candidate_id: str,
    prior_rejection_artifact_paths: Sequence[str | Path] = (),
    prior_rejection_receipt_paths: Sequence[str | Path] = (),
    device: str | torch.device | None = None,
    batch_size: int = 16,
) -> dict[str, Any]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
        raise ValueError("batch_size must be an integer in [1, 32]")
    if candidate_id not in _CANDIDATE_ORDER:
        raise ValueError("candidate_id must be one of A, B, C")
    prior_artifacts = tuple(Path(value) for value in prior_rejection_artifact_paths)
    prior_receipts = tuple(Path(value) for value in prior_rejection_receipt_paths)
    if len(prior_artifacts) != len(prior_receipts):
        raise ValueError(
            "prior scientific-rejection artifact and receipt totals must match"
        )
    paths: dict[str, Path] = {
        "encoder_checkpoint": Path(encoder_checkpoint_path),
        "encoder_progress": Path(encoder_progress_path),
        "encoder_completion_receipt": Path(encoder_completion_receipt_path),
        "source_export_receipt": Path(source_receipt_path),
        "experiment_config": Path(config_path),
        "gate_specification": Path(gate_specification_path),
        "epoch11_gate_artifact": Path(epoch11_gate_artifact_path),
        "epoch11_gate_receipt": Path(epoch11_gate_receipt_path),
        "pose_snapshot": Path(pose_snapshot_path),
        "gate_runner": Path(__file__),
        "period_module": Path(
            estimate_period_from_embedding_velocity_vectors.__code__.co_filename
        ),
        "recurrence_carrier_module": Path(build_recurrence_carrier_curves.__code__.co_filename),
        "consensus_module": Path(MultiExpertCounter.__module__.replace(".", "/") + ".py"),
        "epoch11_gate_dependency": Path(_epoch11.__file__),
    }
    source_root = Path.cwd().resolve(strict=True)
    paths["consensus_module"] = source_root / "src" / paths["consensus_module"]
    for index, value in enumerate(prior_artifacts):
        paths[f"prior_rejection_artifact_{index}"] = value
    for index, value in enumerate(prior_receipts):
        paths[f"prior_rejection_receipt_{index}"] = value
    for role, path in {**paths, "pose_cache_directory": Path(pose_cache_dir)}.items():
        _reject_privileged_path(path, role=role)

    source_covered = _epoch11._require_source_tree_membership(
        source_root,
        {
            "gate_runner": paths["gate_runner"],
            "experiment_config": paths["experiment_config"],
            "gate_specification": paths["gate_specification"],
            "period_module": paths["period_module"],
            "recurrence_carrier_module": paths["recurrence_carrier_module"],
            "consensus_module": paths["consensus_module"],
            "epoch11_gate_dependency": paths["epoch11_gate_dependency"],
        },
    )
    identities = {name: _stable_file_sha256(path) for name, path in paths.items()}

    # All text containers and the progress log are syntax/firewall checked
    # before the first trusted torch checkpoint is deserialized.
    _strict_json(paths["source_export_receipt"], document="source export receipt")
    completion_preflight = _strict_json(
        paths["encoder_completion_receipt"],
        document="encoder completion receipt",
    )
    run_id = completion_preflight.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("encoder completion receipt lacks a run_id")
    started_preflight_path = paths["encoder_completion_receipt"].with_name(
        f"{run_id}.started.json"
    )
    _reject_privileged_path(started_preflight_path, role="encoder started receipt")
    paths["encoder_started_receipt"] = started_preflight_path
    identities["encoder_started_receipt"] = _stable_file_sha256(
        started_preflight_path
    )
    _strict_json(started_preflight_path, document="encoder started receipt")
    _strict_json(paths["epoch11_gate_artifact"], document="epoch11 gate artifact")
    _strict_json(paths["epoch11_gate_receipt"], document="epoch11 gate receipt")
    _strict_json(paths["pose_snapshot"], document="pose snapshot")
    for index in range(len(prior_artifacts)):
        _strict_json(
            paths[f"prior_rejection_artifact_{index}"],
            document="prior scientific rejection artifact",
        )
        _strict_json(
            paths[f"prior_rejection_receipt_{index}"],
            document="prior scientific rejection receipt",
        )
    _preflight_progress_jsonl(paths["encoder_progress"])

    _validate_source_receipt_binding(
        paths["source_export_receipt"],
        sha256=identities["source_export_receipt"][0],
    )
    source_git_sha = clean_git_revision(source_root)
    specification = load_gate_specification(paths["gate_specification"])
    config = _load_candidate_config(paths["experiment_config"])
    profile = _validate_candidate_config(
        config, specification, candidate_id=candidate_id
    )
    declared_snapshot = _epoch11._load_snapshot(paths["pose_snapshot"])
    if _strict_json(paths["pose_snapshot"], document="pose snapshot") != declared_snapshot.to_dict():
        raise ValueError("pose snapshot is not in canonical order or schema form")
    if len(declared_snapshot.entries) != specification.expected_training_video_total:
        raise ValueError("pose snapshot does not contain exactly train337")
    if declared_snapshot.pose_fingerprint != config.pose_fingerprint:
        raise ValueError("pose snapshot fingerprint differs from the candidate config")
    epoch11_artifact, epoch11_receipt = _validate_epoch11_gate_pair(
        paths["epoch11_gate_artifact"],
        paths["epoch11_gate_receipt"],
        config=config,
        specification=specification,
        config_sha256=identities["experiment_config"][0],
        pose_snapshot_sha256=identities["pose_snapshot"][0],
        pose_cache_set_sha256=declared_snapshot.fingerprint,
        source_git_sha=source_git_sha,
    )
    completion, started_path, started_identity = _validate_encoder_completion_receipt(
        paths["encoder_completion_receipt"],
        expected_source_git_sha=source_git_sha,
        specification=specification,
        config=config,
        identities=identities,
        epoch11_artifact=epoch11_artifact,
        epoch11_receipt=epoch11_receipt,
    )
    if (
        started_path != paths["encoder_started_receipt"]
        or started_identity != identities["encoder_started_receipt"]
    ):
        raise RuntimeError("encoder started receipt identity changed after preflight")

    stage, provenance = _peek_checkpoint(paths["encoder_checkpoint"], config)
    if stage != "encoder":
        raise ValueError("terminal gate requires an encoder checkpoint")
    if len(provenance.training_video_ids) != specification.expected_training_video_total:
        raise ValueError("checkpoint provenance does not bind exactly train337")
    if provenance.upstream_encoder_checkpoint_sha256 is not None:
        raise ValueError("encoder checkpoint unexpectedly names an upstream encoder")
    if provenance.source_git_sha != source_git_sha:
        raise ValueError("checkpoint source revision differs from gate source")
    runtime_image = os.environ.get("PAMS_CONTAINER_IMAGE_ID", "").strip()
    runtime_environment = os.environ.get(
        "PAMS_CONTAINER_ENVIRONMENT_SHA256", ""
    ).strip()
    runtime_source = os.environ.get("PAMS_CONTAINER_SOURCE_REVISION", "").strip()
    if (
        runtime_image != provenance.container_image_id
        or runtime_environment != provenance.container_environment_sha256
        or runtime_source != provenance.source_git_sha
    ):
        raise ValueError("runtime container identity differs from checkpoint provenance")
    _validate_completion_provenance(completion, provenance)
    validate_terminal_checkpoint(
        paths["encoder_checkpoint"],
        config,
        expected_stage="encoder",
        expected_provenance=provenance,
        progress_path=paths["encoder_progress"],
    )
    schedule = _terminal_schedule_metadata(
        paths["encoder_checkpoint"],
        paths["encoder_progress"],
        config=config,
        specification=specification,
    )

    sequences, selected_receipts, full_receipts, selected_ids = _load_training_poses(
        pose_cache_dir=Path(pose_cache_dir),
        provenance=provenance,
        config=config,
        sample_size=0,
        seed=config.seed,
    )
    if len(sequences) != specification.expected_training_video_total:
        raise ValueError("pose loader did not return the complete train337 set")
    selected_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint, entries=selected_receipts
    )
    full_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint, entries=full_receipts
    )
    if selected_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("selected and complete train337 pose snapshots differ")
    if full_snapshot.fingerprint != declared_snapshot.fingerprint:
        raise ValueError("declared train337 pose snapshot differs from cache bytes")
    if full_snapshot.fingerprint != provenance.pose_cache_set_sha256:
        raise ValueError("train337 pose cache differs from checkpoint provenance")

    prior_chain = _validate_predecessor_chain(
        candidate_id=candidate_id,
        profile=profile,
        artifact_paths=prior_artifacts,
        receipt_paths=prior_receipts,
        gate_specification_sha256=identities["gate_specification"][0],
        pose_cache_set_sha256=full_snapshot.fingerprint,
        source_git_sha=source_git_sha,
    )

    resolved_device = _device(device)
    model = load_model_checkpoint(
        paths["encoder_checkpoint"],
        config,
        device=resolved_device,
        expected_stage="encoder",
        expected_provenance=provenance,
    ).eval()
    model_state_before = _epoch11._model_state_sha256(model)
    baseline, representation, shuffle, zero_pose, static_pose = (
        _encode_baseline_and_controls(
            model,
            sequences,
            config=config,
            minimum_pairs=specification.minimum_lag_pair_total,
            device=resolved_device,
            batch_size=batch_size,
        )
    )
    time_scale = _time_scale_consistency(
        model,
        sequences,
        baseline,
        config=config,
        specification=specification,
        device=resolved_device,
        batch_size=batch_size,
    )
    model_state_after = _epoch11._model_state_sha256(model)
    if model_state_after != model_state_before:
        raise RuntimeError("model state changed during the read-only terminal gate")

    aggregates = {
        "representation": _representation_distribution(
            representation, near_collapse_rms=specification.near_collapse_rms
        ),
        "period": _period_distribution(
            baseline, minimum_period=config.period.minimum
        ),
        "readout": _readout_distribution(baseline),
        "peak_total": _peak_total_distribution(baseline),
        "time_scale": time_scale,
        "embedding_shuffle_fixed_real_period": _shuffle_distribution(shuffle),
        "zero_pose": _pose_control_distribution(zero_pose),
        "static_pose": _pose_control_distribution(static_pose),
        "per_video_rows_persisted": False,
    }
    decision = gate_decision(
        candidate_id=candidate_id,
        aggregates=aggregates,
        thresholds=specification.thresholds,
    )

    _, _, final_receipts, final_ids = _load_training_poses(
        pose_cache_dir=Path(pose_cache_dir),
        provenance=provenance,
        config=config,
        sample_size=2,
        seed=config.seed,
    )
    final_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint, entries=final_receipts
    )
    if final_ids != selected_ids[:2] or final_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("train337 pose cache changed during the terminal gate")
    _require_unchanged(paths, identities)
    if clean_git_revision(source_root) != source_git_sha:
        raise RuntimeError("terminal gate source changed during evaluation")

    hardware = hardware_fingerprint()
    runtime = {
        "device_type": resolved_device.type,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "source_git_sha": source_git_sha,
        "container_image_id": runtime_image,
        "container_environment_sha256": runtime_environment,
    }
    code_roles = {
        "gate_runner",
        "period_module",
        "recurrence_carrier_module",
        "consensus_module",
        "epoch11_gate_dependency",
    }
    code_hashes = {name: identities[name][0] for name in sorted(code_roles)}
    hard_invariants = {
        "selected_peak_total_is_one_of_three_experts": True,
        "eligible_readouts_majority_first_then_fft_nearest_verified": True,
        "active_mask_subset_of_valid_mask": True,
        "curve_zero_outside_active_mask": True,
        "score_zero_outside_valid_mask": True,
        "finite_readout_outputs": True,
        "per_video_predictions_persisted": False,
    }
    payload = {
        "schema_version": _ARTIFACT_SCHEMA_VERSION,
        "artifact_type": specification.artifact_type,
        "status": (
            "terminal_readout_eligible" if decision["overall_pass"] else "scientific_rejection"
        ),
        "classification": specification.classification,
        "protocol": config.protocol,
        "seed": config.seed,
        "candidate": {
            "id": candidate_id,
            "fixed_training_period_frames_provenance_only": (
                profile.fixed_training_period_frames
            ),
            "anchor_stride": profile.anchor_stride,
            "first_pass_only": True,
            "prior_scientific_rejections": list(prior_chain),
            "cross_candidate_metric_ranking_forbidden": True,
        },
        "label_firewall": {
            "accepted_scientific_inputs": [
                "exact_terminal_encoder_checkpoint_and_progress",
                "schema_v3_encoder_completion_and_started_receipts",
                "exact_source_export_receipt",
                "exact_candidate_config_and_frozen_gate_specification",
                "passing_epoch11_gate_artifact_and_receipt",
                "checkpoint_bound_train337_pose_cache_and_snapshot",
                "required_prior_scientific_rejection_receipts",
            ],
            "manifest_interface_supported": False,
            "media_interface_supported": False,
            "development_identity_media_pose_or_target_interface_supported": False,
            "sealed_evaluation_identity_media_pose_or_target_interface_supported": False,
            "action_class_interface_supported": False,
            "repetition_annotation_interface_supported": False,
            "external_label_fields_accessed": [],
            "training_interface_supported": False,
        },
        "inputs": {
            **{f"{name}_sha256": value[0] for name, value in identities.items()},
            **{f"{name}_bytes": value[1] for name, value in identities.items()},
            "config_fingerprint": config.fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
            "pose_cache_set_sha256": full_snapshot.fingerprint,
            "training_video_total": len(selected_ids),
            "training_video_ids_sha256": _identifier_commitment(selected_ids),
            "source_git_sha": source_git_sha,
            "source_receipt_covered_paths": source_covered,
            "checkpoint_provenance": {
                "protocol": provenance.protocol,
                "dataset_fingerprint": provenance.dataset_fingerprint,
                "training_video_total": len(provenance.training_video_ids),
                "training_video_ids_sha256": _identifier_commitment(
                    provenance.training_video_ids
                ),
                "pose_fingerprint": provenance.pose_fingerprint,
                "pose_cache_set_sha256": provenance.pose_cache_set_sha256,
                "source_git_sha": provenance.source_git_sha,
                "container_image_id": provenance.container_image_id,
                "container_environment_sha256": (
                    provenance.container_environment_sha256
                ),
                "upstream_encoder_checkpoint_sha256": (
                    provenance.upstream_encoder_checkpoint_sha256
                ),
            },
            "checkpoint_schedule": schedule,
            "encoder_completion_run_id_sha256": hashlib.sha256(
                completion.run_id.encode("utf-8")
            ).hexdigest(),
            "code_files_sha256": code_hashes,
            "code_files_sha256_commitment": sha256_json(code_hashes),
            "read_only_post_run_identity_verified": True,
        },
        "algorithm": {
            "period_estimator": "full_vector_embedding_velocity_acf",
            "per_sample_period_upper_bound": "min(4096, floor(timeline_length/2))",
            "period_confidence_part_of_carrier_eligibility": True,
            "carrier_eligibility": (
                "positive period confidence AND structural lag eligibility AND "
                "recurrence carrier availability"
            ),
            "pose_time_shuffle": "deterministic valid-row shuffle then full re-encode",
            "embedding_time_shuffle": (
                "deterministic valid-row shuffle with the real estimated period fixed"
            ),
            "zero_pose": "zero coordinates with original valid mask and positional encoding",
            "static_pose": (
                "middle valid real pose repeated over valid frames with positional encoding"
            ),
            "cycle_margin": "sim(P)-0.5*(sim(P/2)+sim(1.5P))",
            "time_scale_factors": list(specification.time_scale_factors),
            "peak_readout": "three experts; majority first, then active-reference nearest",
            "full_timeline_reference_authorizing": False,
            "fixed_training_period_used_at_inference": False,
        },
        "thresholds": dict(specification.thresholds),
        "aggregates": aggregates,
        "hard_invariants": hard_invariants,
        "gate": decision,
        "scientific_caveats": [
            "The native recurrence-carrier readout is independently inferred, not author-disclosed.",
            "This train337 gate tests label-free mechanism consistency, not accuracy.",
            "Passing authorizes only one separately frozen dev84 protocol build.",
            "No development or sealed-evaluation identity, pose, prediction, or score is authorized here.",
            "Absolute confidence, training-period proximity, curve RMS, and cross-candidate ranking are non-authorizing.",
        ],
        "hardware": hardware,
        "hardware_sha256": sha256_json(hardware),
        "runtime": runtime,
        "runtime_sha256": sha256_json(runtime),
        "read_only_verification": {
            "all_file_inputs_unchanged": True,
            "train337_pose_cache_set_unchanged": True,
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after": model_state_after,
            "model_or_optimizer_state_updated": False,
            "training_steps_executed": 0,
            "pose_cache_write_operations": 0,
            "per_video_prediction_write_operations": 0,
        },
    }
    if set(payload) != _OUTPUT_KEYS:
        raise RuntimeError("terminal gate output schema drifted")
    _reject_forbidden_mapping_keys(payload, document="terminal gate output")
    del model, epoch11_artifact
    gc.collect()
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def _encoded_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _receipt_path(output: Path) -> Path:
    return Path(str(output) + ".receipt.json")


def write_gate_artifact(
    output: Path, payload: Mapping[str, Any]
) -> tuple[Path, str]:
    _reject_privileged_path(output, role="terminal gate output")
    receipt_path = _receipt_path(output)
    if output.exists() or receipt_path.exists():
        raise FileExistsError("terminal gate artifact and receipt must both be new")
    if set(payload) != _OUTPUT_KEYS:
        raise ValueError("terminal gate payload schema mismatch")
    _reject_forbidden_mapping_keys(payload, document="terminal gate payload")
    encoded = _encoded_json(payload)
    artifact_sha256 = hashlib.sha256(encoded).hexdigest()
    _write_new(output, encoded)
    prior_chain = payload["candidate"]["prior_scientific_rejections"]
    receipt = {
        "schema_version": 1,
        "artifact_type": _EXPECTED_RECEIPT_TYPE,
        "artifact_locator": output.name,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": len(encoded),
        "artifact_status": payload["status"],
        "candidate_id": payload["candidate"]["id"],
        "overall_pass": payload["gate"]["overall_pass"],
        "eligible_for_single_frozen_dev84_protocol_build": payload["gate"][
            "eligible_for_single_frozen_dev84_protocol_build"
        ],
        "next_candidate_training_authorized": payload["gate"][
            "next_candidate_training_authorized"
        ],
        "encoder_checkpoint_sha256": payload["inputs"]["encoder_checkpoint_sha256"],
        "encoder_progress_sha256": payload["inputs"]["encoder_progress_sha256"],
        "encoder_completion_receipt_sha256": payload["inputs"][
            "encoder_completion_receipt_sha256"
        ],
        "encoder_started_receipt_sha256": payload["inputs"][
            "encoder_started_receipt_sha256"
        ],
        "source_export_receipt_sha256": payload["inputs"][
            "source_export_receipt_sha256"
        ],
        "experiment_config_sha256": payload["inputs"]["experiment_config_sha256"],
        "gate_specification_sha256": payload["inputs"]["gate_specification_sha256"],
        "pose_snapshot_sha256": payload["inputs"]["pose_snapshot_sha256"],
        "pose_cache_set_sha256": payload["inputs"]["pose_cache_set_sha256"],
        "epoch11_gate_artifact_sha256": payload["inputs"][
            "epoch11_gate_artifact_sha256"
        ],
        "epoch11_gate_receipt_sha256": payload["inputs"][
            "epoch11_gate_receipt_sha256"
        ],
        "source_git_sha": payload["inputs"]["source_git_sha"],
        "code_files_sha256_commitment": payload["inputs"][
            "code_files_sha256_commitment"
        ],
        "prior_scientific_rejections_sha256": sha256_json(prior_chain),
        "aggregate_only": True,
        "dev84_pose_or_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }
    if set(receipt) != _RECEIPT_KEYS:
        raise RuntimeError("terminal gate receipt schema drifted")
    _write_new(receipt_path, _encoded_json(receipt))
    return receipt_path, artifact_sha256


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-progress", type=Path, required=True)
    parser.add_argument("--encoder-completion-receipt", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate-specification", type=Path, required=True)
    parser.add_argument("--epoch11-gate-artifact", type=Path, required=True)
    parser.add_argument("--epoch11-gate-receipt", type=Path, required=True)
    parser.add_argument("--pose-cache-dir", type=Path, required=True)
    parser.add_argument("--pose-snapshot", type=Path, required=True)
    parser.add_argument("--candidate-id", choices=_CANDIDATE_ORDER, required=True)
    parser.add_argument(
        "--prior-rejection-artifact", type=Path, action="append", default=[]
    )
    parser.add_argument(
        "--prior-rejection-receipt", type=Path, action="append", default=[]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    _reject_privileged_path(arguments.output, role="terminal gate output")
    if arguments.output.exists() or _receipt_path(arguments.output).exists():
        raise FileExistsError("terminal gate output and receipt must both be new")
    payload = run_gate(
        arguments.encoder_checkpoint,
        arguments.encoder_progress,
        arguments.encoder_completion_receipt,
        arguments.source_receipt,
        arguments.config,
        arguments.gate_specification,
        arguments.epoch11_gate_artifact,
        arguments.epoch11_gate_receipt,
        arguments.pose_cache_dir,
        arguments.pose_snapshot,
        candidate_id=arguments.candidate_id,
        prior_rejection_artifact_paths=arguments.prior_rejection_artifact,
        prior_rejection_receipt_paths=arguments.prior_rejection_receipt,
        device=arguments.device,
        batch_size=arguments.batch_size,
    )
    receipt_path, artifact_sha256 = write_gate_artifact(arguments.output, payload)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "output": str(arguments.output),
                "receipt": str(receipt_path),
                "candidate_id": payload["candidate"]["id"],
                "overall_pass": payload["gate"]["overall_pass"],
                "eligible_for_single_frozen_dev84_protocol_build": payload["gate"][
                    "eligible_for_single_frozen_dev84_protocol_build"
                ],
                "next_candidate_training_authorized": payload["gate"][
                    "next_candidate_training_authorized"
                ],
                "dev84_pose_or_scoring_authorized": False,
                "test105_evaluation_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0 if payload["gate"]["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
