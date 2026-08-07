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
    normalization: Literal["per_frame_minmax"] = "per_frame_minmax"
    missing_value: float = Field(default=0.0, strict=True)

    @field_validator("missing_value")
    @classmethod
    def validate_missing_value(cls, value: float) -> float:
        if value != 0.0:
            raise ValueError("data.missing_value must be exactly 0.0")
        return value


class PoseConfig(StrictModel):
    """Frozen MediaPipe extractor settings included in pose-cache identity."""

    preprocessing_revision: Literal[
        "detected-span-minmax-zero-span-invalid-v2",
        "longest-contiguous-track-minmax-zero-span-invalid-v3",
        "official-segment-full-timeline-v1",
    ] = "detected-span-minmax-zero-span-invalid-v2"
    model_id: str = "mediapipe-pose-0.10.14"
    model_complexity: int = Field(default=1, ge=0, le=2)
    smooth_landmarks: bool = True
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    crop_to_detected_span: bool = True
    incomplete_clip_policy: Literal["error", "pad_invalid_tail"] = "error"

    @model_validator(mode="after")
    def validate_track_policy(self) -> PoseConfig:
        if (
            self.preprocessing_revision
            == "longest-contiguous-track-minmax-zero-span-invalid-v3"
            and not self.crop_to_detected_span
        ):
            raise ValueError(
                "longest-contiguous-track preprocessing requires "
                "crop_to_detected_span=true"
            )
        if (
            self.preprocessing_revision == "official-segment-full-timeline-v1"
            and self.crop_to_detected_span
        ):
            raise ValueError(
                "official-segment-full-timeline preprocessing requires "
                "crop_to_detected_span=false"
            )
        if (
            self.preprocessing_revision != "official-segment-full-timeline-v1"
            and self.incomplete_clip_policy != "error"
        ):
            raise ValueError(
                "pad_invalid_tail is only valid for official-segment-full-timeline"
            )
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
    direct_fft_timebase: Literal[
        "compact_valid",
        "dense_resampled",
    ] = "compact_valid"

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
            raise ValueError(
                "rotation_degrees must contain three finite values in [0, 180]"
            )
        scale_minimum, scale_maximum = self.scale_range
        if (
            not math.isfinite(scale_minimum)
            or not math.isfinite(scale_maximum)
            or scale_minimum <= 0.0
            or scale_maximum < scale_minimum
            or scale_maximum > 10.0
        ):
            raise ValueError(
                "scale_range must be finite, positive, ordered, and at most 10"
            )
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
    shape_normalization: Literal[
        "raw",
        "masked_rms",
    ] = "raw"
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
                "position-permutation consistency requires "
                "model.position_encoding_mode=sinusoidal"
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
        if period["direct_fft_timebase"] == "compact_valid":
            # Preserve every historical config/checkpoint fingerprint.  The
            # opt-in dense-resampled inference clock is an independently
            # inferred mask/timebase repair and remains in the method identity.
            period.pop("direct_fft_timebase")
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
        if sshead["shape_normalization"] == "raw":
            # Preserve every historical config/checkpoint fingerprint.  The
            # opt-in masked-RMS route changes the inferred SSHead objective
            # and therefore remains in the method identity.
            sshead.pop("shape_normalization")
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
