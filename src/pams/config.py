"""Strict configuration models for the reproduction.

All paper-inferred values live in YAML and are validated here.  Unknown keys
are rejected so that a typo cannot silently change an experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Base class that rejects undeclared configuration fields."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        protected_namespaces=(),
    )


class DataConfig(StrictModel):
    frames: int = Field(default=256, ge=16)
    keypoints: int = Field(default=33, ge=1)
    coordinates: int = Field(default=3, ge=1)
    normalization: Literal[
        "per_frame_minmax",
        "body_centered_uniform_scale",
    ] = "per_frame_minmax"
    missing_value: float = Field(default=0.0, strict=True)

    @field_validator("missing_value")
    @classmethod
    def validate_missing_value(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("data.missing_value must be exactly 0.0")
        return value


class PoseRecoveryConfig(StrictModel):
    """Identity-bearing settings for the opt-in v4 recovery pass.

    The heavy model file is supplied separately at runtime because filesystem
    locations are host-specific.  Its content digest, model identity, retry
    policy, and every association/ROI choice are nevertheless frozen here and
    therefore included in :attr:`PAMSConfig.pose_fingerprint`.
    """

    heavy_model_id: str = "mediapipe-pose-heavy-0.10.14"
    model_complexity: Literal[2] = 2
    heavy_model_asset_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )
    temporal_resampling: Literal["none_native_timeline"] = "none_native_timeline"
    static_image_mode: bool = True
    smooth_landmarks: bool = False
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    full_frame_retry: bool = True
    roi_retry: bool = True
    roi_margin_fraction: float = Field(default=0.20, ge=0.0, le=1.0)
    roi_min_side_fraction: float = Field(default=0.08, gt=0.0, le=1.0)
    association_cost: Literal["normalized-center-l2-plus-log-scale-v1"] = (
        "normalized-center-l2-plus-log-scale-v1"
    )
    association_center_weight: float = Field(default=1.0, ge=0.0)
    association_log_scale_weight: float = Field(default=0.25, ge=0.0)
    dominant_track_strategy: Literal["global-viterbi-visible-extent-v1"] = (
        "global-viterbi-visible-extent-v1"
    )
    maximum_gap_frames: int = Field(default=8, ge=1)
    maximum_gap_seconds: float = Field(default=0.25, gt=0.0)
    pose_coordinate_interpolation: Literal[False] = False

    @model_validator(mode="after")
    def validate_non_degenerate_association(self) -> PoseRecoveryConfig:
        if self.association_center_weight == 0.0 and self.association_log_scale_weight == 0.0:
            raise ValueError("pose recovery association requires a positive cost weight")
        return self

    @field_validator("heavy_model_id")
    @classmethod
    def validate_heavy_model_id(cls, value: str) -> str:
        model_id = value.strip()
        if not model_id:
            raise ValueError("pose.recovery.heavy_model_id must be non-empty")
        return model_id


class KeypointRCNNRecoveryConfig(StrictModel):
    """Frozen identity for v4a-locked torchvision keypoint recovery."""

    model_id: str = "torchvision-keypointrcnn-resnet50-fpn-coco-v1-0.20.1"
    model_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_asset_filename: Literal["keypointrcnn_resnet50_fpn_coco-fc266e95.pth"] = (
        "keypointrcnn_resnet50_fpn_coco-fc266e95.pth"
    )
    model_topology: Literal["torchvision-coco-v1-frozen-batchnorm-overwrite-eps-zero-v1"] = (
        "torchvision-coco-v1-frozen-batchnorm-overwrite-eps-zero-v1"
    )
    expected_state_dict_keys: Literal[313] = 313
    expected_parameter_count: Literal[59137258] = 59137258
    expected_torch_version: Literal["2.5.1+cu124"] = "2.5.1+cu124"
    expected_torchvision_version: Literal["0.20.1+cu124"] = "0.20.1+cu124"
    expected_cuda_version: Literal["12.4"] = "12.4"
    expected_cudnn_version: Literal[90100] = 90100
    expected_gpu_name: Literal["NVIDIA RTX A6000"] = "NVIDIA RTX A6000"
    expected_compute_capability: Literal["8.6"] = "8.6"
    base_pose_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_pose_cache_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_pose_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_paired_gate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_same39_selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_same39_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_zero11_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_preprocessing_revision: Literal[
        "official-segment-heavy-missing-retry-full-timeline-v4a"
    ] = "official-segment-heavy-missing-retry-full-timeline-v4a"
    temporal_resampling: Literal["none_native_timeline"] = "none_native_timeline"
    detector_observes_all_decoded_frames: Literal[True] = True
    fill_missing_only: Literal[True] = True
    base_observations_bitwise_locked: Literal[True] = True
    box_score_threshold: Literal[0.2] = 0.2
    keypoint_logit_threshold: Literal[2.0] = 2.0
    minimum_confident_keypoints: Literal[4] = 4
    maximum_candidates_per_frame: Literal[4] = 4
    person_label: Literal[1] = 1
    coco_keypoints: Literal[17] = 17
    coco_to_mediapipe_mapping: Literal["duplicate-eyes-wrists-ankles-nose-mouth-v1"] = (
        "duplicate-eyes-wrists-ankles-nose-mouth-v1"
    )
    z_coordinate_policy: Literal["zero"] = "zero"
    visibility_policy: Literal["sigmoid-keypoint-logit-times-box-score-v1"] = (
        "sigmoid-keypoint-logit-times-box-score-v1"
    )
    input_scale_policy: Literal["torchvision-fpn-min800-max1333-v1"] = (
        "torchvision-fpn-min800-max1333-v1"
    )
    input_min_size: Literal[800] = 800
    input_max_size: Literal[1333] = 1333
    box_nms_threshold: Literal[0.5] = 0.5
    box_detections_per_image: Literal[100] = 100
    inference_batch_size: Literal[4] = 4
    inference_device: Literal["cuda:0"] = "cuda:0"
    deterministic_algorithms: Literal[True] = True
    allow_tf32: Literal[False] = False
    cublas_workspace_config: Literal[":4096:8"] = ":4096:8"
    runtime_cache_policy: Literal["ephemeral-exec-tmpfs-user1000-v1"] = (
        "ephemeral-exec-tmpfs-user1000-v1"
    )
    association_cost: Literal["normalized-center-l2-plus-log-scale-v1"] = (
        "normalized-center-l2-plus-log-scale-v1"
    )
    association_coordinate_space: Literal["frame-normalized-absolute-xy-v1"] = (
        "frame-normalized-absolute-xy-v1"
    )
    v4a_anchor_policy: Literal["nearest-shared-coco17-xy-minmax-singleton-threshold-v1"] = (
        "nearest-shared-coco17-xy-minmax-singleton-threshold-v1"
    )
    v4a_anchor_distance_maximum: Literal[0.25] = 0.25
    fill_coordinate_space: Literal["selected-candidate-per-frame-minmax-xyz-v1"] = (
        "selected-candidate-per-frame-minmax-xyz-v1"
    )
    association_center_weight: float = Field(default=1.0, ge=0.0)
    association_log_scale_weight: float = Field(default=0.25, ge=0.0)
    dominant_track_strategy: Literal["global-viterbi-visible-extent-v1"] = (
        "global-viterbi-visible-extent-v1"
    )
    pose_coordinate_interpolation: Literal[False] = False

    @model_validator(mode="after")
    def validate_non_degenerate_association(self) -> KeypointRCNNRecoveryConfig:
        if self.association_center_weight == 0.0 and self.association_log_scale_weight == 0.0:
            raise ValueError("keypoint recovery association requires a positive cost weight")
        return self


class KeypointRCNNSingleSourceConfig(StrictModel):
    """Frozen mechanics for the v4e single-detector representation.

    This deliberately reuses only the immutable Keypoint R-CNN runtime
    identity from v4d.  It does not inherit v4d's mixed-source cache semantics:
    every valid output frame comes from one Keypoint R-CNN path, while the
    MediaPipe v4a sequence contributes an isolated post-selection shape
    diagnostic only and cannot influence the selected KPRCNN path.
    """

    model_id: Literal["torchvision-keypointrcnn-resnet50-fpn-coco-v1-0.20.1"] = (
        "torchvision-keypointrcnn-resnet50-fpn-coco-v1-0.20.1"
    )
    model_asset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_asset_filename: Literal["keypointrcnn_resnet50_fpn_coco-fc266e95.pth"] = (
        "keypointrcnn_resnet50_fpn_coco-fc266e95.pth"
    )
    model_topology: Literal["torchvision-coco-v1-frozen-batchnorm-overwrite-eps-zero-v1"] = (
        "torchvision-coco-v1-frozen-batchnorm-overwrite-eps-zero-v1"
    )
    expected_state_dict_keys: Literal[313] = 313
    expected_parameter_count: Literal[59137258] = 59137258
    expected_torch_version: Literal["2.5.1+cu124"] = "2.5.1+cu124"
    expected_torchvision_version: Literal["0.20.1+cu124"] = "0.20.1+cu124"
    expected_cuda_version: Literal["12.4"] = "12.4"
    expected_cudnn_version: Literal[90100] = 90100
    expected_gpu_name: Literal["NVIDIA RTX A6000"] = "NVIDIA RTX A6000"
    expected_compute_capability: Literal["8.6"] = "8.6"
    base_pose_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_pose_cache_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_pose_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_paired_gate_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_same39_selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_same39_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_zero11_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    base_preprocessing_revision: Literal[
        "official-segment-heavy-missing-retry-full-timeline-v4a"
    ] = "official-segment-heavy-missing-retry-full-timeline-v4a"
    temporal_resampling: Literal["none_native_timeline"] = "none_native_timeline"
    detector_observes_all_decoded_frames: Literal[True] = True
    box_score_threshold: Literal[0.2] = 0.2
    keypoint_logit_threshold: Literal[2.0] = 2.0
    minimum_confident_keypoints: Literal[8] = 8
    minimum_confident_action_keypoints: Literal[4] = 4
    maximum_candidates_per_frame: Literal[4] = 4
    person_label: Literal[1] = 1
    coco_keypoints: Literal[17] = 17
    visibility_policy: Literal["sigmoid-logit-times-box-score-torso-reliable-v1"] = (
        "sigmoid-logit-times-box-score-torso-reliable-v1"
    )
    reliable_visibility_minimum: Literal[0.1] = 0.1
    reliable_torso_policy: Literal["bilateral-shoulders-and-hips-required-v1"] = (
        "bilateral-shoulders-and-hips-required-v1"
    )
    input_scale_policy: Literal["torchvision-fpn-min800-max1333-v1"] = (
        "torchvision-fpn-min800-max1333-v1"
    )
    input_min_size: Literal[800] = 800
    input_max_size: Literal[1333] = 1333
    box_nms_threshold: Literal[0.5] = 0.5
    box_detections_per_image: Literal[100] = 100
    inference_batch_size: Literal[4] = 4
    inference_device: Literal["cuda:0"] = "cuda:0"
    deterministic_algorithms: Literal[True] = True
    allow_tf32: Literal[False] = False
    cublas_workspace_config: Literal[":4096:8"] = ":4096:8"
    runtime_cache_policy: Literal["ephemeral-exec-tmpfs-user1000-v1"] = (
        "ephemeral-exec-tmpfs-user1000-v1"
    )
    single_source_full_track: Literal[True] = True
    v4d_candidate_cache_consumed: Literal[False] = False
    training_joint_space: Literal["coco17-first17-zero-pad33-v1"] = (
        "coco17-first17-zero-pad33-v1"
    )
    training_coordinate_space: Literal["body-centered-uniform-rms-scale-xy-z0-v1"] = (
        "body-centered-uniform-rms-scale-xy-z0-v1"
    )
    anchor_coordinate_space: Literal["shared-coco17-similarity-procrustes-xy-v1"] = (
        "shared-coco17-similarity-procrustes-xy-v1"
    )
    identity_anchor_source: Literal["mediapipe-v4a-shape-diagnostic-only-not-viterbi"] = (
        "mediapipe-v4a-shape-diagnostic-only-not-viterbi"
    )
    primary_center_weight: Literal[1.0] = 1.0
    primary_log_scale_weight: Literal[0.25] = 0.25
    # Shape means a same-source, torso-proportion signature only; it never
    # consumes processed-v4a coordinates or penalizes articulated limb motion.
    primary_shape_weight: Literal[0.5] = 0.5
    primary_anchor_weight: Literal[0.0] = 0.0
    secondary_center_weight: Literal[0.75] = 0.75
    secondary_log_scale_weight: Literal[0.5] = 0.5
    secondary_shape_weight: Literal[0.25] = 0.25
    secondary_anchor_weight: Literal[0.0] = 0.0
    # These identity-only hard-edge limits are frozen before same39/full337.
    # Articulated action motion is prohibited from this association objective.
    identity_maximum_center_step: Literal[0.12] = 0.12
    identity_maximum_log_scale_step: Literal[0.12] = 0.12
    identity_maximum_morphology_step: Literal[0.02] = 0.02
    # This physical-time rule predates any full337 v4e observation.  The
    # effective association bridge is min(cap, floor(seconds * fps)); a zero
    # result is valid and means that no missing frame may be bridged.
    maximum_bridge_gap_seconds: Literal[0.25] = 0.25
    maximum_bridge_gap_frame_cap: Literal[8] = 8
    global_track_strategy: Literal[
        "seeded-stable-track-bank-motion-free-identity-then-2w-actor-utility-v4"
    ] = (
        "seeded-stable-track-bank-motion-free-identity-then-2w-actor-utility-v4"
    )
    second_path_policy: Literal["independent-frozen-weight-stable-track-bank-v2"] = (
        "independent-frozen-weight-stable-track-bank-v2"
    )
    raw_extraction_authorizes_training: Literal[False] = False

    @model_validator(mode="after")
    def validate_dual_association_weights(self) -> KeypointRCNNSingleSourceConfig:
        primary = (
            self.primary_center_weight,
            self.primary_log_scale_weight,
            self.primary_shape_weight,
            self.primary_anchor_weight,
        )
        secondary = (
            self.secondary_center_weight,
            self.secondary_log_scale_weight,
            self.secondary_shape_weight,
            self.secondary_anchor_weight,
        )
        if not any(primary) or not any(secondary):
            raise ValueError("v4e requires two non-degenerate association weight sets")
        if primary == secondary:
            raise ValueError("v4e secondary association weights must differ from primary")
        return self


class PoseConfig(StrictModel):
    """Frozen MediaPipe extractor settings included in pose-cache identity."""

    preprocessing_revision: Literal[
        "detected-span-minmax-zero-span-invalid-v2",
        "longest-contiguous-track-minmax-zero-span-invalid-v3",
        "official-segment-full-timeline-v1",
        "official-segment-heavy-missing-retry-full-timeline-v4a",
        "official-segment-heavy-video-fill-missing-full-timeline-v4b",
        "official-segment-tasks-heavy-video-multipose4-fill-missing-full-timeline-v4c",
        "official-segment-v4a-locked-keypointrcnn-fill-missing-full-timeline-v4d",
        "official-segment-keypointrcnn-stable-track-bank-coco17-full-timeline-v4e",
    ] = "detected-span-minmax-zero-span-invalid-v2"
    model_id: str = "mediapipe-pose-0.10.14"
    model_complexity: int = Field(default=1, ge=0, le=2)
    smooth_landmarks: bool = True
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    crop_to_detected_span: bool = True
    incomplete_clip_policy: Literal["error", "pad_invalid_tail"] = "error"
    recovery: PoseRecoveryConfig | None = None
    keypoint_recovery: KeypointRCNNRecoveryConfig | None = None
    keypoint_single_source: KeypointRCNNSingleSourceConfig | None = None

    @model_validator(mode="after")
    def validate_track_policy(self) -> PoseConfig:
        if (
            self.preprocessing_revision == "longest-contiguous-track-minmax-zero-span-invalid-v3"
            and not self.crop_to_detected_span
        ):
            raise ValueError(
                "longest-contiguous-track preprocessing requires crop_to_detected_span=true"
            )
        if (
            self.preprocessing_revision
            in {
                "official-segment-full-timeline-v1",
                "official-segment-heavy-missing-retry-full-timeline-v4a",
                "official-segment-heavy-video-fill-missing-full-timeline-v4b",
                "official-segment-tasks-heavy-video-multipose4-fill-missing-full-timeline-v4c",
                "official-segment-v4a-locked-keypointrcnn-fill-missing-full-timeline-v4d",
                "official-segment-keypointrcnn-stable-track-bank-coco17-full-timeline-v4e",
            }
            and self.crop_to_detected_span
        ):
            raise ValueError(
                "official-segment-full-timeline preprocessing requires crop_to_detected_span=false"
            )
        if (
            self.preprocessing_revision
            not in {
                "official-segment-full-timeline-v1",
                "official-segment-heavy-missing-retry-full-timeline-v4a",
                "official-segment-heavy-video-fill-missing-full-timeline-v4b",
                "official-segment-tasks-heavy-video-multipose4-fill-missing-full-timeline-v4c",
                "official-segment-v4a-locked-keypointrcnn-fill-missing-full-timeline-v4d",
                "official-segment-keypointrcnn-stable-track-bank-coco17-full-timeline-v4e",
            }
            and self.incomplete_clip_policy != "error"
        ):
            raise ValueError("pad_invalid_tail is only valid for official-segment-full-timeline")
        v4a_revision = (
            self.preprocessing_revision == "official-segment-heavy-missing-retry-full-timeline-v4a"
        )
        v4b_revision = (
            self.preprocessing_revision
            == "official-segment-heavy-video-fill-missing-full-timeline-v4b"
        )
        v4c_revision = (
            self.preprocessing_revision
            == "official-segment-tasks-heavy-video-multipose4-fill-missing-full-timeline-v4c"
        )
        v4d_revision = (
            self.preprocessing_revision
            == "official-segment-v4a-locked-keypointrcnn-fill-missing-full-timeline-v4d"
        )
        v4e_revision = (
            self.preprocessing_revision
            == "official-segment-keypointrcnn-stable-track-bank-coco17-full-timeline-v4e"
        )
        recovery_revision = v4a_revision or v4b_revision or v4c_revision
        if recovery_revision:
            if self.model_complexity != 1:
                raise ValueError("v4 pass0 must use model_complexity=1")
            if self.recovery is None:
                raise ValueError("v4 pose recovery requires pose.recovery settings")
            if v4a_revision and not (
                self.recovery.static_image_mode
                and not self.recovery.smooth_landmarks
                and self.recovery.full_frame_retry
                and self.recovery.roi_retry
            ):
                raise ValueError(
                    "v4a recovery requires static unsmoothed full-frame and ROI retries"
                )
            if (v4b_revision or v4c_revision) and not (
                not self.recovery.static_image_mode
                and self.recovery.smooth_landmarks
                and self.recovery.full_frame_retry
                and not self.recovery.roi_retry
            ):
                raise ValueError(
                    "v4b/v4c recovery requires a smoothed full-timeline VIDEO pass without ROI"
                )
        elif self.recovery is not None:
            raise ValueError(
                "pose.recovery is accepted only by the v4a/v4b/v4c preprocessing revisions"
            )
        if v4d_revision:
            if self.model_complexity != 1:
                raise ValueError("v4d base pose must use model_complexity=1")
            if self.keypoint_recovery is None:
                raise ValueError("v4d requires pose.keypoint_recovery settings")
        elif self.keypoint_recovery is not None:
            raise ValueError("pose.keypoint_recovery is accepted only by v4d")
        if v4e_revision:
            if self.model_complexity != 1:
                raise ValueError("v4e anchor pose must use model_complexity=1")
            if self.keypoint_single_source is None:
                raise ValueError("v4e requires pose.keypoint_single_source settings")
        elif self.keypoint_single_source is not None:
            raise ValueError("pose.keypoint_single_source is accepted only by v4e")
        return self

    @field_validator("model_id")
    @classmethod
    def validate_model_id(cls, value: str) -> str:
        model_id = value.strip()
        if not model_id:
            raise ValueError("pose.model_id must be non-empty")
        return model_id


class ModelConfig(StrictModel):
    input_dim: int = 99
    model_dim: int = 512
    input_projection_scale: Literal["none", "sqrt_model_dim"] = "none"
    position_encoding_mode: Literal["sinusoidal", "none"] = "sinusoidal"
    embedding_dim: int = 512
    layers: int = 4
    heads: int = 16
    feedforward_dim: int = 2048
    dropout: float = 0.1
    norm_first: bool = False
    period_head_hidden_dim: int = 128

    @model_validator(mode="after")
    def validate_attention_shape(self) -> ModelConfig:
        if self.model_dim % self.heads:
            raise ValueError("model_dim must be divisible by heads")
        return self


class PeriodConfig(StrictModel):
    minimum: int = Field(default=4, ge=2)
    maximum: int = Field(default=128, ge=3)
    pose_energy_epochs: int = Field(default=10, ge=0)
    training_mode: Literal[
        "adaptive",
        "fixed_period_inferred",
    ] = "adaptive"
    fixed_period_frames: int = Field(default=16, ge=2)
    post_warmup_source: Literal[
        "embedding_velocity_coordinate",
        "embedding_velocity_vector_acf",
        "projected_pose_velocity_vector_acf",
    ] = "embedding_velocity_coordinate"

    @model_validator(mode="after")
    def validate_period_bounds(self) -> PeriodConfig:
        if self.maximum <= self.minimum:
            raise ValueError("period.maximum must be greater than period.minimum")
        if (
            self.training_mode == "fixed_period_inferred"
            and not self.minimum <= self.fixed_period_frames <= self.maximum
        ):
            raise ValueError(
                "period.fixed_period_frames must lie within [minimum, maximum] "
                "for fixed_period_inferred"
            )
        return self


class LossConfig(StrictModel):
    scales: tuple[float, ...] = (0.5, 1.0, 1.5)
    temperature: float = Field(default=0.1, gt=0)
    kmeans_clusters: int = Field(default=8, ge=2)
    kmeans_refresh_epochs: int = Field(default=5, ge=1)
    exclude_other_scale_positives_from_denominator: bool = False


class SkeletonAugmentationConfig(StrictModel):
    """Opt-in, target-free skeleton augmentation used only for encoder training."""

    enabled: bool = Field(default=False, strict=True)
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_range: tuple[float, float] = (1.0, 1.0)
    jitter_std: float = Field(default=0.0, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_transform_ranges(self) -> SkeletonAugmentationConfig:
        if any(
            not math.isfinite(value) or value < 0.0 or value > 180.0
            for value in self.rotation_degrees
        ):
            raise ValueError("rotation_degrees must contain three finite values in [0, 180]")
        scale_minimum, scale_maximum = self.scale_range
        if (
            not math.isfinite(scale_minimum)
            or not math.isfinite(scale_maximum)
            or scale_minimum <= 0.0
            or scale_maximum < scale_minimum
            or scale_maximum > 10.0
        ):
            raise ValueError("scale_range must be finite, positive, ordered, and at most 10")
        neutral = (
            self.rotation_degrees == (0.0, 0.0, 0.0)
            and self.scale_range == (1.0, 1.0)
            and self.jitter_std == 0.0
        )
        if self.enabled and neutral:
            raise ValueError(
                "enabled skeleton augmentation requires at least one non-neutral transform"
            )
        if not self.enabled and not neutral:
            raise ValueError(
                "disabled skeleton augmentation must keep all transform values neutral"
            )
        return self


class TrainingConfig(StrictModel):
    epochs: int = Field(default=150, ge=1)
    effective_batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    scheduler_factor: float = Field(default=0.5, gt=0, lt=1)
    scheduler_patience: int = Field(default=8, ge=0)
    minimum_learning_rate: float = Field(default=1e-6, gt=0)
    position_permutation_consistency_weight: float = Field(default=0.0, ge=0)
    skeleton_augmentation: SkeletonAugmentationConfig = SkeletonAugmentationConfig()


class SSHeadConfig(StrictModel):
    epochs: int = Field(default=30, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    architecture: Literal[
        "pointwise_mlp",
        "temporal_conv",
    ] = "pointwise_mlp"
    input_source: Literal[
        "encoder_embedding",
        "projected_pose_pre_pe",
        "projected_pose_reference_relative",
    ] = "encoder_embedding"
    period_confidence_mode: Literal[
        "nonzero_gate",
        "normalized_weight",
    ] = "nonzero_gate"
    cycle_weight: float = Field(default=1.0, ge=0)
    spectral_weight: float = Field(default=1.0, ge=0)
    variance_weight: float = Field(default=0.1, ge=0)
    smoothness_weight: float = Field(default=0.01, ge=0)


class ConsensusConfig(StrictModel):
    expert_mode: Literal["multi", "medium_only"] = "multi"
    sigma_multipliers: tuple[float, float, float] = (0.05, 0.12, 0.15)
    distance_multipliers: tuple[float, float, float] = (0.5, 0.8, 1.2)
    short_window_multiplier: float = Field(default=0.5, gt=0)
    long_window_multiplier: float = Field(default=2.0, gt=0)
    height_factor: float = Field(default=0.6, ge=0)
    prominence_factor: float = Field(default=0.25, ge=0)
    long_window_weight: float = Field(default=0.6, ge=0, le=1)


class PAMSConfig(StrictModel):
    schema_version: int = 1
    protocol: str = "ucfrep_526"
    seed: int = 2026
    data: DataConfig = DataConfig()
    pose: PoseConfig = PoseConfig()
    model: ModelConfig = ModelConfig()
    period: PeriodConfig = PeriodConfig()
    loss: LossConfig = LossConfig()
    training: TrainingConfig = TrainingConfig()
    sshead: SSHeadConfig = SSHeadConfig()
    consensus: ConsensusConfig = ConsensusConfig()

    @model_validator(mode="after")
    def validate_input_dimensions(self) -> PAMSConfig:
        expected = self.data.keypoints * self.data.coordinates
        if self.model.input_dim != expected:
            raise ValueError(
                f"model.input_dim={self.model.input_dim} does not match "
                f"keypoints*coordinates={expected}"
            )
        if self.model.embedding_dim != self.model.model_dim:
            raise ValueError("the disclosed architecture requires embedding_dim == model_dim")
        if (
            self.model.position_encoding_mode == "none"
            and self.training.position_permutation_consistency_weight != 0.0
        ):
            raise ValueError(
                "position-permutation consistency requires model.position_encoding_mode=sinusoidal"
            )
        return self

    def canonical_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible representation used for hashing."""

        payload = self.model_dump(mode="json")
        pose = payload["pose"]
        if pose["incomplete_clip_policy"] == "error":
            # Preserve historical fingerprints. The opt-in padding policy is
            # extraction-affecting and remains in every new method identity.
            pose.pop("incomplete_clip_policy")
        if pose["recovery"] is None:
            # Preserve every pre-v4a whole-experiment fingerprint.
            pose.pop("recovery")
        if pose["keypoint_recovery"] is None:
            # Preserve every pre-v4d whole-experiment fingerprint.
            pose.pop("keypoint_recovery")
        if pose["keypoint_single_source"] is None:
            # Preserve every pre-v4e fingerprint.  The v4e block is always
            # explicit and identity-bearing when the new representation is used.
            pose.pop("keypoint_single_source")
        model = payload["model"]
        if model["input_projection_scale"] == "none":
            # ``none`` is the historical behavior.  Omitting only this default
            # from the hash payload keeps existing configs/checkpoints bound to
            # their original fingerprint, while every opt-in scale remains an
            # explicit, fingerprint-changing experiment choice.
            model.pop("input_projection_scale")
        if model["position_encoding_mode"] == "sinusoidal":
            # Preserve every historical config/checkpoint fingerprint.  The
            # opt-in ``none`` route is an independently inferred anti-shortcut
            # candidate and therefore remains in the method identity.
            model.pop("position_encoding_mode")
        period = payload["period"]
        if period["post_warmup_source"] == "embedding_velocity_coordinate":
            # Preserve every historical f99 config/checkpoint fingerprint.
            # Only the opt-in inferred vector-ACF route changes method
            # identity; the explicit default names the existing implementation.
            period.pop("post_warmup_source")
        if period["training_mode"] == "adaptive":
            # The configurable fixed-period route was added after the formal
            # adaptive runs.  Omitting both default fields keeps every
            # historical config/checkpoint fingerprint byte-for-byte stable.
            # Opting into the independently inferred fixed-period baseline
            # retains both fields in the method identity.
            period.pop("training_mode")
            period.pop("fixed_period_frames")
        loss = payload["loss"]
        if not loss["exclude_other_scale_positives_from_denominator"]:
            # Preserve historical fingerprints for the literal denominator.
            # The opt-in inferred union repair is identity-changing.
            loss.pop("exclude_other_scale_positives_from_denominator")
        training = payload["training"]
        if training["position_permutation_consistency_weight"] == 0.0:
            # Preserve all historical config/checkpoint identities.  A
            # positive value enables the independently inferred PE-nuisance
            # consistency objective and therefore remains in the method
            # identity.
            training.pop("position_permutation_consistency_weight")
        if not training["skeleton_augmentation"]["enabled"]:
            # Preserve all historical config/checkpoint identities. Disabled
            # augmentation is strictly required to be neutral by validation;
            # the opt-in training transform remains in the method identity.
            training.pop("skeleton_augmentation")
        sshead = payload["sshead"]
        if sshead["architecture"] == "pointwise_mlp":
            # Preserve all historical SSHead config/checkpoint identities.
            # The local temporal-convolution head is an explicit inferred
            # architecture and remains in the canonical payload.
            sshead.pop("architecture")
        if sshead["input_source"] == "encoder_embedding":
            # Preserve historical config/checkpoint identities.  The opt-in
            # pre-PE and reference-relative routes are independently inferred
            # repairs and therefore remain in the canonical payload.
            sshead.pop("input_source")
        if sshead["period_confidence_mode"] == "nonzero_gate":
            # Preserve historical config/checkpoint identities.  Continuous
            # normalized confidence weighting is an opt-in inferred repair.
            sshead.pop("period_confidence_mode")
        consensus = payload["consensus"]
        if consensus["expert_mode"] == "multi":
            # Preserve every historical checkpoint/config fingerprint.  The
            # opt-in ``medium_only`` value is an inferred inference ablation
            # and therefore changes the effective inference identity.
            consensus.pop("expert_mode")
        return payload

    def nonseed_canonical_dict(self) -> dict[str, Any]:
        """Return the experiment specification with only the seed removed."""

        payload = self.canonical_dict()
        payload.pop("seed")
        return payload

    def pose_canonical_dict(self) -> dict[str, Any]:
        """Return only preprocessing/extractor state used by pose caches."""

        pose = self.pose.model_dump(mode="json")
        if pose["incomplete_clip_policy"] == "error":
            pose.pop("incomplete_clip_policy")
        if pose["recovery"] is None:
            # Keep every historical v2/v3/official-segment fingerprint byte-for-byte
            # stable. Recovery is opt-in and identity-changing.
            pose.pop("recovery")
        if pose["keypoint_recovery"] is None:
            # Keep every historical pre-v4d pose fingerprint stable.
            pose.pop("keypoint_recovery")
        if pose["keypoint_single_source"] is None:
            # Keep every historical pre-v4e pose fingerprint stable.
            pose.pop("keypoint_single_source")
        return {
            "data": self.data.model_dump(mode="json"),
            "pose": pose,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()

    @property
    def nonseed_fingerprint(self) -> str:
        """Hash the frozen method specification shared by preregistered seeds."""

        payload = json.dumps(
            self.nonseed_canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def pose_fingerprint(self) -> str:
        """Hash pose identity without seeds, models, or training hyperparameters."""

        payload = json.dumps(
            self.pose_canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def load_config(path: str | Path) -> PAMSConfig:
    """Load and strictly validate a YAML configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    return PAMSConfig.model_validate(raw)
