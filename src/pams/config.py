"""Strict configuration models for the reproduction.

All paper-inferred values live in YAML and are validated here.  Unknown keys
are rejected so that a typo cannot silently change an experiment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base class that rejects undeclared configuration fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(StrictModel):
    frames: int = Field(default=256, ge=16)
    keypoints: int = Field(default=33, ge=1)
    coordinates: int = Field(default=3, ge=1)
    normalization: str = "per_frame_minmax"
    missing_value: float = 0.0


class ModelConfig(StrictModel):
    input_dim: int = 99
    model_dim: int = 512
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

    @model_validator(mode="after")
    def validate_period_bounds(self) -> PeriodConfig:
        if self.maximum <= self.minimum:
            raise ValueError("period.maximum must be greater than period.minimum")
        return self


class LossConfig(StrictModel):
    scales: tuple[float, ...] = (0.5, 1.0, 1.5)
    temperature: float = Field(default=0.1, gt=0)
    correspondence_tolerance: float = Field(default=0.1, ge=0, le=0.5)
    kmeans_clusters: int = Field(default=8, ge=2)
    kmeans_refresh_epochs: int = Field(default=5, ge=1)


class TrainingConfig(StrictModel):
    epochs: int = Field(default=150, ge=1)
    effective_batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    scheduler_factor: float = Field(default=0.5, gt=0, lt=1)
    scheduler_patience: int = Field(default=8, ge=0)
    minimum_learning_rate: float = Field(default=1e-6, gt=0)


class SSHeadConfig(StrictModel):
    epochs: int = Field(default=30, ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    cycle_weight: float = Field(default=1.0, ge=0)
    spectral_weight: float = Field(default=1.0, ge=0)
    variance_weight: float = Field(default=0.1, ge=0)
    smoothness_weight: float = Field(default=0.01, ge=0)


class ConsensusConfig(StrictModel):
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
        return self

    def canonical_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible representation used for hashing."""

        return self.model_dump(mode="json")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(payload).hexdigest()


def load_config(path: str | Path) -> PAMSConfig:
    """Load and strictly validate a YAML configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    return PAMSConfig.model_validate(raw)
