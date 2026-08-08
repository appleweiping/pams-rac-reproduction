"""Strict, fingerprinted schema for the conventional cycle-back TCC proxy.

The schema is intentionally not an extension of :class:`pams.config.PAMSConfig`.
That separation prevents a new experimental objective from silently changing
the serialization or fingerprints of historical PAMS configurations.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pams.config import SkeletonAugmentationConfig


class _NoDuplicateSafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _NoDuplicateSafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"cycle-back config contains duplicate YAML field {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_NoDuplicateSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(
        allow_inf_nan=False,
        extra="forbid",
        frozen=True,
        protected_namespaces=(),
    )


class NativeWindowPairConfig(_StrictModel):
    """Two strictly disjoint windows on one unresampled timeline."""

    length_frames: int = Field(ge=4, strict=True)
    hop_frames: int = Field(ge=1, strict=True)
    start_policy: Literal[
        "representation_authorized_native_zero_based_hop_grid"
    ] = "representation_authorized_native_zero_based_hop_grid"
    pair_policy: Literal[
        "adjacent_disjoint_same_video_windows_offset_by_window_length"
    ] = "adjacent_disjoint_same_video_windows_offset_by_window_length"
    minimum_valid_frames_per_window: int = Field(default=4, ge=2, strict=True)
    joint_support_policy: Literal[
        "representation_authorized_exact_2w_starts_and_frozen_support_tuple_only"
    ] = "representation_authorized_exact_2w_starts_and_frozen_support_tuple_only"
    validity_policy: Literal[
        "all_frames_valid_in_each_window"
    ] = "all_frames_valid_in_each_window"
    segment_policy: Literal[
        "both_windows_inside_one_authorized_contiguous_segment"
    ] = "both_windows_inside_one_authorized_contiguous_segment"
    temporal_resampling: Literal["none_native_timeline"] = "none_native_timeline"
    encoder_context_policy: Literal[
        "full_authorized_all_valid_stable_range_per_view_then_slice_embeddings"
    ] = "full_authorized_all_valid_stable_range_per_view_then_slice_embeddings"
    attention_boundary_policy: Literal[
        "eligible_frame_range_start_stop_never_wider_association_segment"
    ] = "eligible_frame_range_start_stop_never_wider_association_segment"
    training_inference_context_parity: Literal[True] = True
    source_index_convention: Literal[
        "zero_based_native_frame_index"
    ] = "zero_based_native_frame_index"
    bounded_pair_selection: Literal[
        "one_pair_per_video_then_label_free_hash_fill"
    ] = "one_pair_per_video_then_label_free_hash_fill"
    reset_count_aggregation_policy: Literal[
        "segment_counts_reported_separately_no_default_cross_reset_sum"
    ] = "segment_counts_reported_separately_no_default_cross_reset_sum"

    @model_validator(mode="after")
    def validate_geometry(self) -> NativeWindowPairConfig:
        if self.hop_frames >= self.length_frames:
            raise ValueError("window hop must be smaller than window length")
        if self.minimum_valid_frames_per_window != self.length_frames:
            raise ValueError(
                "cycle-back v1 requires every frame in each window to be valid"
            )
        return self


class CycleBackObjectiveConfig(_StrictModel):
    objective: Literal[
        "symmetric_variance_aware_cycleback_regression"
    ] = "symmetric_variance_aware_cycleback_regression"
    directions: Literal["a_to_b_to_a_and_b_to_a_to_b"] = (
        "a_to_b_to_a_and_b_to_a_to_b"
    )
    temperature: float = Field(default=0.1, gt=0.0)
    variance_log_weight: float = Field(default=0.001, ge=0.0)
    variance_floor: float = Field(default=1e-6, gt=0.0)
    embedding_normalization: Literal["l2"] = "l2"
    target_position_normalization: Literal[
        "absolute_source_index_minus_window_start_divided_by_window_length_minus_one"
    ] = "absolute_source_index_minus_window_start_divided_by_window_length_minus_one"
    candidate_scope: Literal[
        "valid_frames_in_same_video_paired_window_only"
    ] = "valid_frames_in_same_video_paired_window_only"
    cross_video_positive_terms: Literal[False] = False
    cross_video_negative_terms: Literal[False] = False
    prototype_or_cluster_bank: Literal[False] = False

    @model_validator(mode="after")
    def validate_frozen_equation(self) -> CycleBackObjectiveConfig:
        if not math.isclose(self.temperature, 0.1, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("cycle-back v1 freezes temperature at 0.1")
        if not math.isclose(
            self.variance_log_weight,
            0.001,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("cycle-back v1 freezes variance_log_weight at 0.001")
        return self


class EncoderConfig(_StrictModel):
    input_dim: Literal[99] = 99
    model_dim: int = Field(default=512, ge=4, strict=True)
    embedding_dim: int = Field(default=512, ge=2, strict=True)
    layers: int = Field(default=4, ge=1, strict=True)
    heads: int = Field(default=16, ge=1, strict=True)
    feedforward_dim: int = Field(default=2048, ge=4, strict=True)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    norm_first: Literal[False] = False
    input_projection_scale: Literal["none"] = "none"
    position_encoding_mode: Literal["sinusoidal"] = "sinusoidal"
    maximum_native_length: int = Field(default=4096, ge=32, strict=True)

    @model_validator(mode="after")
    def validate_attention_shape(self) -> EncoderConfig:
        if self.model_dim % self.heads:
            raise ValueError("encoder model_dim must be divisible by heads")
        return self


class IndependentViewConfig(_StrictModel):
    view_policy: Literal[
        "two_independently_seeded_skeleton_augmentations"
    ] = "two_independently_seeded_skeleton_augmentations"
    shared_random_generator: Literal[False] = False
    representation: Literal[
        "coco17_body_centered_uniform_rms_scale_xy_z0_v1_padded_to_33"
    ] = "coco17_body_centered_uniform_rms_scale_xy_z0_v1_padded_to_33"
    active_joint_count: Literal[17] = 17
    padded_joint_count: Literal[16] = 16
    augmentation_joint_scope: Literal[
        "active_coco17_only"
    ] = "active_coco17_only"
    rotation_scope: Literal["in_plane_z_axis_only"] = "in_plane_z_axis_only"
    z_coordinate_policy: Literal["exact_zero_before_and_after"] = (
        "exact_zero_before_and_after"
    )
    padded_joint_policy: Literal["indices_17_to_32_exact_zero"] = (
        "indices_17_to_32_exact_zero"
    )
    augmentation: SkeletonAugmentationConfig

    @model_validator(mode="after")
    def validate_enabled_augmentation(self) -> IndependentViewConfig:
        if not self.augmentation.enabled:
            raise ValueError("cycle-back training requires enabled independent views")
        if tuple(self.augmentation.rotation_degrees[:2]) != (0.0, 0.0):
            raise ValueError("unified-2D cycle-back augmentation permits only z-axis rotation")
        return self


class GeometryGateThresholds(_StrictModel):
    minimum_eligible_video_fraction: float = Field(default=0.90, ge=0.0, le=1.0)
    minimum_eligible_pair_fraction: float = Field(default=0.50, ge=0.0, le=1.0)
    minimum_valid_anchor_fraction: float = Field(default=0.50, ge=0.0, le=1.0)
    maximum_source_index_violation_total: Literal[0] = 0
    minimum_zero_to_real_position_error_ratio: float = Field(default=1.0, gt=0.0)
    minimum_shuffle_to_real_position_error_ratio: float = Field(default=1.0, gt=0.0)
    minimum_permuted_pe_to_real_position_error_ratio: float = Field(
        default=0.80,
        gt=0.0,
    )
    maximum_permuted_pe_to_real_position_error_ratio: float = Field(
        default=1.25,
        gt=0.0,
    )
    minimum_pe_off_to_real_position_error_ratio: float = Field(default=0.80, gt=0.0)
    maximum_pe_off_to_real_position_error_ratio: float = Field(default=1.25, gt=0.0)


class GeometryAuditConfig(_StrictModel):
    artifact_type: Literal[
        "pams_conventional_cycleback_train337_geometry_pe_null_gate_v1"
    ] = "pams_conventional_cycleback_train337_geometry_pe_null_gate_v1"
    expected_training_video_total: Literal[337] = 337
    diagnostic_sample_videos: int = Field(default=32, ge=2, le=337, strict=True)
    maximum_probe_pairs: int = Field(default=256, ge=2, strict=True)
    encoder_segment_context_batch_size: Literal[8] = 8
    thresholds_frozen_before_execution: Literal[True] = True
    thresholds: GeometryGateThresholds = GeometryGateThresholds()


class MechanismProbeThresholds(_StrictModel):
    minimum_loss_relative_drop: float = Field(default=0.10, ge=0.0, le=1.0)
    minimum_final_temporal_rms_median: float = Field(default=0.001, ge=0.0)
    minimum_real_null_position_error_gap: float = Field(default=0.02, ge=0.0)
    minimum_final_mask_flicker_to_real_position_error_ratio: float = Field(
        default=1.0,
        gt=0.0,
    )
    minimum_final_torso_only_to_real_position_error_ratio: float = Field(
        default=1.0,
        gt=0.0,
    )
    minimum_final_alternating_limb_dropout_to_real_position_error_ratio: float = Field(
        default=1.0,
        gt=0.0,
    )
    minimum_final_valid_anchor_fraction: float = Field(default=0.50, ge=0.0, le=1.0)
    minimum_final_permuted_pe_to_real_position_error_ratio: float = Field(
        default=0.80,
        gt=0.0,
    )
    maximum_final_permuted_pe_to_real_position_error_ratio: float = Field(
        default=1.25,
        gt=0.0,
    )
    minimum_final_pe_off_to_real_position_error_ratio: float = Field(
        default=0.80,
        gt=0.0,
    )
    maximum_final_pe_off_to_real_position_error_ratio: float = Field(
        default=1.25,
        gt=0.0,
    )


class MechanismProbeConfig(_StrictModel):
    artifact_type: Literal[
        "pams_conventional_cycleback_256_step_mechanism_probe_v1"
    ] = "pams_conventional_cycleback_256_step_mechanism_probe_v1"
    gate_contract: Literal[
        "cycleback_256_optimizer_steps_not_epoch11_gate"
    ] = "cycleback_256_optimizer_steps_not_epoch11_gate"
    optimizer_steps: Literal[256] = 256
    video_batch_size: int = Field(default=8, ge=2, strict=True)
    maximum_pairs_per_step: int = Field(default=32, ge=2, strict=True)
    evaluation_sample_videos: int = Field(default=32, ge=2, le=337, strict=True)
    evaluation_maximum_pairs: int = Field(default=256, ge=2, strict=True)
    encoder_segment_context_batch_size: Literal[8] = 8
    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    thresholds_frozen_before_execution: Literal[True] = True
    threshold_provenance: Literal[
        "independent_synthetic_cycleback_preregistration_required_before_launch"
    ] = "independent_synthetic_cycleback_preregistration_required_before_launch"
    thresholds: MechanismProbeThresholds = MechanismProbeThresholds()


class ConventionalCycleBackConfig(_StrictModel):
    """Complete identity of one independently inferred cycle-back candidate."""

    schema_version: Literal[1] = 1
    namespace: Literal[
        "pams.conventional_cycleback_tcc.v1"
    ] = "pams.conventional_cycleback_tcc.v1"
    classification: Literal[
        "independently_inferred_proxy_not_author_implementation"
    ] = "independently_inferred_proxy_not_author_implementation"
    paper_reference_status: Literal[
        "baseline_named_but_cycleback_equation_and_sampling_undisclosed"
    ] = "baseline_named_but_cycleback_equation_and_sampling_undisclosed"
    paper_table_claim_eligible: Literal[False] = False
    protocol: Literal["ucfrep_526_train337_only"] = "ucfrep_526_train337_only"
    generation_failure_policy: Literal[
        "all_three_sinusoidal_candidates_rejected_terminates_generation_pe_off_diagnostic_only"
    ] = (
        "all_three_sinusoidal_candidates_rejected_terminates_generation_pe_off_diagnostic_only"
    )
    candidate_id: Literal["W16_H4", "W16_H2", "W24_H4"]
    seed: int = Field(default=2026, ge=0, strict=True)
    window_pair: NativeWindowPairConfig
    objective: CycleBackObjectiveConfig = CycleBackObjectiveConfig()
    encoder: EncoderConfig = EncoderConfig()
    views: IndependentViewConfig
    geometry_audit: GeometryAuditConfig = GeometryAuditConfig()
    mechanism_probe: MechanismProbeConfig = MechanismProbeConfig()

    @model_validator(mode="after")
    def validate_candidate_matrix(self) -> ConventionalCycleBackConfig:
        expected = {
            "W16_H4": (16, 4),
            "W16_H2": (16, 2),
            "W24_H4": (24, 4),
        }[self.candidate_id]
        observed = (
            self.window_pair.length_frames,
            self.window_pair.hop_frames,
        )
        if observed != expected:
            raise ValueError(
                f"candidate {self.candidate_id} requires window/hop {expected}, "
                f"received {observed}"
            )
        if self.window_pair.length_frames > self.encoder.maximum_native_length:
            raise ValueError("window length exceeds encoder positional capacity")
        return self

    @property
    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def candidate_invariant_fingerprint(self) -> str:
        """Fingerprint after removing only the preregistered window choice."""

        payload = self.model_dump(mode="json")
        payload["candidate_id"] = "WINDOW_CANDIDATE"
        payload["window_pair"]["length_frames"] = 0
        payload["window_pair"]["hop_frames"] = 0
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @field_validator("seed")
    @classmethod
    def reject_boolean_seed(cls, value: int) -> int:
        if isinstance(value, bool):
            raise TypeError("seed must be an integer, not bool")
        return value


def load_conventional_cycleback_config(
    path: str | Path,
) -> ConventionalCycleBackConfig:
    """Load exactly one strict cycle-back namespace document."""

    source = Path(path)
    return parse_conventional_cycleback_config(source.read_bytes())


def parse_conventional_cycleback_config(
    encoded: bytes | str,
) -> ConventionalCycleBackConfig:
    """Parse one already-stabilized config buffer with duplicate-key rejection."""

    if isinstance(encoded, bytes):
        try:
            source = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("cycle-back configuration is not UTF-8") from exc
    else:
        source = encoded
    payload = yaml.load(source, Loader=_NoDuplicateSafeLoader)
    if not isinstance(payload, dict):
        raise ValueError("cycle-back configuration root must be a mapping")
    return ConventionalCycleBackConfig.model_validate(payload)
