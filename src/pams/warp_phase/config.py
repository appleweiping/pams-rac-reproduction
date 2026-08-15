"""Frozen, independent configuration schema for WARP-PHASE pilot v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar, Literal

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    frozen_values: ClassVar[dict[str, object]] = {}

    @model_validator(mode="after")
    def validate_frozen_values(self) -> _StrictModel:
        for field, expected in self.frozen_values.items():
            if getattr(self, field) != expected:
                raise ValueError(f"{field} is frozen at {expected!r}")
        return self


class SourceConfig(_StrictModel):
    train_path: Literal[
        "data/counting_multirep_skeleton_pose_expanded_v44_len320/train.pkl"
    ]
    train_sha256: Literal[
        "c96fc1dfa233ec4bf1af19e7480ee8c721f4cddee9f121185f4e494c7244e1eb"
    ]
    val_path: Literal[
        "data/counting_multirep_skeleton_pose_expanded_v44_len320/val.pkl"
    ]
    val_sha256: Literal[
        "4064ef24dc2970e9b07de8adddf1ecd4932aecb90453a67b4f90d7bd35d10251"
    ]
    allowed_splits: tuple[Literal["train"], Literal["val"]]


class StorageConfig(_StrictModel):
    run_root: Literal["data/warp_phase_pilot_v1"]
    features_root: Literal["features"]
    vault_root: Literal["vault"]
    audit_root: Literal["audit"]


class DataContractConfig(_StrictModel):
    source_samples: Literal[320]
    joints: Literal[17]
    pose_channels: Literal[3]
    confidence_threshold: float
    minimum_pose_coverage: float
    minimum_distinct_clocks: Literal[64]
    minimum_valid_adjacent_cells: Literal[63]
    minimum_feature_frame_coverage: float
    minimum_valid_joints_per_frame: Literal[8]
    duplicate_xy_tolerance: float
    duplicate_confidence_tolerance: float
    ambiguous_train_slots: Literal[13]
    ambiguous_val_slots: Literal[8]
    frozen_values = {
        "confidence_threshold": 0.2,
        "minimum_pose_coverage": 0.8,
        "minimum_feature_frame_coverage": 0.6,
        "duplicate_xy_tolerance": 1e-5,
        "duplicate_confidence_tolerance": 1e-6,
    }


class SelectorConfig(_StrictModel):
    rng_seed: Literal[20270815]
    parent_clocks: Literal[128]
    parent_stride: Literal[64]
    rejection_clocks: Literal[64]
    fft_samples: Literal[256]
    minimum_period: Literal[4]
    maximum_period: Literal[128]
    minimum_acf_samples: Literal[16]
    minimum_acf: float
    weak_jitter_std: float
    weak_jitter_clip: float
    weak_dropout: float
    rejection_relative_tolerance: float
    harmonic_margin: float
    frozen_values = {
        "minimum_acf": 0.25,
        "weak_jitter_std": 0.01,
        "weak_jitter_clip": 0.03,
        "weak_dropout": 0.1,
        "rejection_relative_tolerance": 0.1,
        "harmonic_margin": 0.1,
    }


class WarpConfig(_StrictModel):
    segments: Literal[3]
    first_breakpoint_range: tuple[float, float]
    second_breakpoint_range: tuple[float, float]
    raw_slope_range: tuple[float, float]
    accepted_slope_range: tuple[float, float]
    pause_probability: float
    maximum_schedule_attempts: Literal[128]
    alias_margin_pi: float
    minimum_valid_intervals_per_window: Literal[32]
    minimum_track_interval_coverage: float
    frozen_values = {
        "first_breakpoint_range": (0.2, 0.4),
        "second_breakpoint_range": (0.6, 0.8),
        "raw_slope_range": (0.5, 1.5),
        "accepted_slope_range": (0.4, 2.0),
        "pause_probability": 0.2,
        "alias_margin_pi": 0.05,
        "minimum_track_interval_coverage": 0.8,
    }


class ModelConfig(_StrictModel):
    input_channels: Literal[68]
    convolution_channels: Literal[128]
    recurrent_size: Literal[128]
    phase_dimensions: Literal[2]
    trainable_parameters: Literal[225026]


class TrainingConfig(_StrictModel):
    seeds: tuple[Literal[20270815], Literal[20270816], Literal[20270817]]
    identities_per_step: Literal[8]
    accumulation_steps: Literal[4]
    updates: Literal[20000]
    learning_rate: float
    weight_decay: float
    gradient_clip: float
    window_samples: Literal[64]
    window_stride: Literal[32]
    frozen_values = {
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "gradient_clip": 1.0,
    }


class DecodeConfig(_StrictModel):
    window_samples: Literal[64]
    stride: Literal[32]
    minimum_interval_weight: float
    rounding: Literal["half_even"]
    frozen_values = {"minimum_interval_weight": 0.001}


class GateConfig(_StrictModel):
    schema_fixture_bytes: Literal[617]
    schema_fixture_sha256: Literal[
        "31f81549f9c5dbedc6ddb278e868793246826a234de90c06c169c71387428275"
    ]
    fixture_count: Literal[1000]
    symmetric_fixture_count: Literal[250]
    minimum_fixture_agreement: float
    maximum_half_selections: Literal[20]
    maximum_double_selections: Literal[20]
    maximum_symmetric_half_selections: Literal[12]
    maximum_symmetric_double_selections: Literal[12]
    frozen_values = {"minimum_fixture_agreement": 0.95}


class WarpPhaseConfig(_StrictModel):
    """Complete frozen identity of the prospective pilot."""

    schema_version: Literal[1]
    pilot_id: Literal["warp_phase_pilot_v1"]
    protocol: Literal["multirep_v44_partial_gt_bbox_supplied_track"]
    evidence_status: Literal["zero_eligible_results"]
    training_authorized: Literal[False]
    source: SourceConfig
    storage: StorageConfig
    data: DataContractConfig
    selector: SelectorConfig
    warp: WarpConfig
    model: ModelConfig
    training: TrainingConfig
    decode: DecodeConfig
    gates: GateConfig

    def canonical_dict(self) -> dict[str, Any]:
        """Return the recursively JSON-compatible configuration identity."""

        return self.model_dump(mode="json")

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_warp_phase_config(path: str | Path) -> WarpPhaseConfig:
    """Load a WARP-PHASE YAML file with no implicit or extra fields."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"WARP-PHASE configuration must be a mapping: {config_path}")
    return WarpPhaseConfig.model_validate(payload)
