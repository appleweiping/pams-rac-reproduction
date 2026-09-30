"""Strict dev-only boundary for projected-position ACF counting.

``PAMS-PositionACFDirect-inferred-v1`` is a post-hoc representation
diagnostic.  It reuses the exact completed v8 encoder but estimates frequency
from the projected *positions* before positional encoding, rather than from
the projected velocity used by ``PAMS-TeacherPeriodDirect``.  It is not an
author-disclosed readout, is permanently paper-table ineligible, and cannot
authorize test105.

Prediction accepts only count-free protocol identities.  Scoring validates
the complete prediction, receipt, exact-v8 identity, source identity, and
canonical dev84 identity before it first opens the separate dev target file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from pydantic import Field, model_validator

import pams.pams_dev as strict_dev
import pams.period as period_module
from pams.config import PAMSConfig, StrictModel, load_config
from pams.data import (
    PoseInputManifest,
    UnlabeledVideoRecord,
    load_dev_target_manifest,
    load_pose_cache_set,
    pose_input_identity_sha256,
)
from pams.evaluation import PredictionRecord, evaluate_predictions
from pams.metrics import round_count
from pams.period import estimate_period_from_projected_pose
from pams.reproducibility import (
    clean_git_revision,
    hardware_fingerprint,
    sha256_file,
    sha256_json,
)
from pams.training import collate_pose_sequences
from pams.types import CountResult, PoseSequence

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

_METHOD_KEY: Literal["pams-position-acf-direct-inferred-v1"] = (
    "pams-position-acf-direct-inferred-v1"
)
_CLASSIFICATION: Literal["PAMS-PositionACFDirect-inferred post-hoc diagnostic"] = (
    "PAMS-PositionACFDirect-inferred post-hoc diagnostic"
)
_ROUNDING: Literal["nearest_integer_half_up"] = "nearest_integer_half_up"
_PREDICTION_NAME: Literal["predictions.json"] = "predictions.json"
_PREDICTION_RECEIPT_NAME = "prediction.receipt.json"
_POSE_SNAPSHOT_NAME = "dev-pose-cache-snapshot.json"
_EVALUATION_NAME = "evaluation.json"
_EVALUATION_RECEIPT_NAME = "evaluation.receipt.json"
_PREDICTION_BATCH_SIZE: Literal[32] = 32
_ALLOWED_BINS = tuple(range(2, 65))
_HARMONIC_CANDIDATES = tuple(range(2, 8))
_HARMONIC_RELATIVE_TOLERANCE = 0.20
_READOUT_CONFIG_FILE_SHA256 = (
    "9c4db6210ecd82d0aa5e2a2437b4e8132eb993bff4afe808ad0db8772f9d7923"
)
_READOUT_CONFIG_FINGERPRINT = (
    "861e43963d45e4277d8daf93de7d556ebf63f1ab531bf4ded7077d6d20fd3f91"
)

_V8_CONFIG_FILE_SHA256 = "eb4072a195608757cd2542855b33fb000c987f96c94b83a1e448ab78f97e9374"
_V8_CONFIG_FINGERPRINT = "eaf9e2ce6047c4a13c13daaef10af1ae288542ad94ff9fb513d19e225912adf2"
_V8_POSE_FINGERPRINT = "3dd0388320095796f42aa82d904073b6478b073ed351394ef8d03c30798a0116"
_V8_ENCODER_SHA256 = "6817fe468d76c3fa2182dd831695d6fe3d6da208a2b7f8dfa90cffa6872e8053"
_V8_ENCODER_PROGRESS_SHA256 = "aadc8f9bf067b3489efcd80db43c42cf09dba3477c2976bbc16571ff31684622"
_V8_ENCODER_COMPLETION_RECEIPT_SHA256 = (
    "f4815e1961ba883c23ef479ddcdef44aea5cad8a50d54e17b8e6a4cccc613f3d"
)
_V8_CHECKPOINT_SOURCE_GIT_SHA = "07192debb7f5748c53a1c6d4f0c0d3f16228e229"
_V8_CHECKPOINT_CONTAINER_IMAGE_ID = (
    "sha256:a6d10131c321c323c00bb85408cb8333503ecbc5d780d5059a73152d54df6005"
)
_V8_CHECKPOINT_CONTAINER_ENVIRONMENT_SHA256 = (
    "1ccfd265d829e5160b8af39156193f63622ac73eaefec00b0f9dab40106c3508"
)
_V8_PROTOCOL_IDENTITY_SHA256 = "7aaaea4fdd0620cc7abb4f307cf63a9506448540e3d3d920acf70957f6b8b97f"
_V8_TRAINING_IDENTITY_SHA256 = "177e3a70e6fb272bac7c976e9700f70355bc30adb3990dc1276485c3e0cbddd6"
_V8_DEV_IDENTITY_SHA256 = "bdf944d2aa13e22c07383163794e8edc3198f0c8fd8355b6a6d0255566b3cb5b"
_V8_TRAINING_POSE_CACHE_SET_SHA256 = (
    "f32d718ae922120778f535a6a4f467edba79cafeba90373b3a6a55bf11be6ee2"
)
_V8_DEV_POSE_CACHE_SET_SHA256 = "681b0390ff252df9cf5b1e7c41bfcb6ecb8a0c5b4ef341087cacddc1bf5a3de1"
_V8_DEV_POSE_CACHE_SNAPSHOT_SHA256 = (
    "14c290d86097b572cd544853f9d9c8bde379a3e53d7b637b22b66c6fbe1bb417"
)
_V8_INPUT_SHA256 = {
    "train_inputs_sha256": "e238d307b4ed37b2f6c19eefdf7fa8812ec3d231cd2a7091be38e1c279f04207",
    "train_commitment_sha256": (
        "ce39b1c9038bb1506354de010e46a23053b307b852222332f13df3f79d799044"
    ),
    "dev_inputs_sha256": "f21c49569805e4aea326977f6d7ce2964f503fad129a17e95214594364ca4ae7",
    "dev_commitment_sha256": (
        "a2bb2db86abef64e16a8fe1ac196857ac8e77cc9cd461c568b2193401a8cd169"
    ),
    "test_identity_inputs_sha256": (
        "9eb2a057246b77fc7a1121ba4c9d27bf55177ad2fcd5dbb87237396a4a6015ca"
    ),
    "test_identity_commitment_sha256": (
        "232d3db7ce09018716594a1f1b332eec47b37410a8d3068e662b026ee2dc9c66"
    ),
}


class PositionACFReadoutConfig(StrictModel):
    """Exact semantic identity of the independently frozen readout."""

    schema_version: Literal[1]
    key: Literal["pams-projected-position-acf-direct-v1"]
    classification: Literal[
        "inferred frozen-v8 projected-position vector-ACF diagnostic"
    ]
    eligible_for_paper_table: Literal[False]
    test105_evaluation_authorized: Literal[False]
    frozen_inputs: dict[str, Any]
    readout: dict[str, Any]
    predev_gates: dict[str, Any]
    dev_stop_gate: dict[str, Any]
    mount_policy: dict[str, Any]

    @model_validator(mode="after")
    def validate_frozen_semantics(self) -> PositionACFReadoutConfig:
        if self.fingerprint != _READOUT_CONFIG_FINGERPRINT:
            raise ValueError(
                "position-acf readout configuration semantics differ from "
                "the frozen v1 specification"
            )
        return self

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


def load_position_acf_readout_config(
    path: str | Path,
) -> PositionACFReadoutConfig:
    source = Path(path)
    source_sha256 = sha256_file(source)
    if source_sha256 != _READOUT_CONFIG_FILE_SHA256:
        raise ValueError(
            "position-acf readout configuration bytes differ from the frozen "
            "v1 file"
        )
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("position-acf readout configuration root must be a mapping")
    config = PositionACFReadoutConfig.model_validate(payload)
    if sha256_file(source) != source_sha256:
        raise RuntimeError("position-acf readout configuration changed while loading")
    return config


def _canonical_sha256(value: str, name: str) -> str:
    digest = str(value)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return digest


def _canonical_git_sha(value: str, name: str) -> str:
    revision = str(value)
    if _GIT_SHA_PATTERN.fullmatch(revision) is None:
        raise ValueError(f"{name} must be a lowercase 40-character Git SHA")
    return revision


def _canonical_image_id(value: str, name: str) -> str:
    image_id = str(value)
    if _IMAGE_ID_PATTERN.fullmatch(image_id) is None:
        raise ValueError(f"{name} must be an immutable sha256:<digest> image ID")
    return image_id


def _code_file_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    return {
        "cli": sha256_file(directory / "cli.py"),
        "config": sha256_file(directory / "config.py"),
        "data": sha256_file(directory / "data.py"),
        "evaluation": sha256_file(directory / "evaluation.py"),
        "metrics": sha256_file(directory / "metrics.py"),
        "model": sha256_file(directory / "model.py"),
        "pams_dev": sha256_file(directory / "pams_dev.py"),
        "period": sha256_file(directory / "period.py"),
        "position_acf_dev": sha256_file(Path(__file__)),
        "reproducibility": sha256_file(directory / "reproducibility.py"),
        "training": sha256_file(directory / "training.py"),
        "types": sha256_file(directory / "types.py"),
    }


def estimate_period_from_projected_position(
    projected_pose: torch.Tensor,
    *,
    minimum: int,
    maximum: int,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Late-bind the independently inferred projected-position estimator."""

    return period_module.estimate_period_from_projected_position(
        projected_pose,
        minimum=minimum,
        maximum=maximum,
        valid_mask=valid_mask,
    )


def projected_position_vector_acf_diagnostics(
    projected_pose: torch.Tensor,
    *,
    minimum: int,
    maximum: int,
    valid_mask: torch.Tensor,
) -> tuple[Any, ...]:
    """Late-bind the immutable 63-bin spectrum diagnostic."""

    return period_module.projected_position_vector_acf_diagnostics(
        projected_pose,
        minimum=minimum,
        maximum=maximum,
        valid_mask=valid_mask,
    )


def _require_exact_v8_bindings(
    *,
    config_path: Path,
    config: PAMSConfig,
    checkpoint_path: Path,
    progress_path: Path,
    completion_receipt_path: Path,
    input_hashes: dict[str, str],
) -> None:
    observed = {
        "config file": (sha256_file(config_path), _V8_CONFIG_FILE_SHA256),
        "config fingerprint": (config.fingerprint, _V8_CONFIG_FINGERPRINT),
        "pose fingerprint": (config.pose_fingerprint, _V8_POSE_FINGERPRINT),
        "encoder checkpoint": (sha256_file(checkpoint_path), _V8_ENCODER_SHA256),
        "encoder progress": (
            sha256_file(progress_path),
            _V8_ENCODER_PROGRESS_SHA256,
        ),
        "encoder completion receipt": (
            sha256_file(completion_receipt_path),
            _V8_ENCODER_COMPLETION_RECEIPT_SHA256,
        ),
    }
    observed.update(
        {
            f"protocol input {name}": (input_hashes[name], expected)
            for name, expected in _V8_INPUT_SHA256.items()
        }
    )
    mismatches = [
        f"{name}: observed={actual}, expected={expected}"
        for name, (actual, expected) in observed.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError(
            "position-acf diagnostic accepts only the exact frozen v8 encoder "
            "package; " + "; ".join(mismatches)
        )


class PositionACFSpectrumBin(StrictModel):
    """One of the 63 fixed frequency bins allowed by periods 4--128."""

    bin_index: int = Field(ge=2, le=64)
    period_frames: float = Field(gt=0)
    power_share: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_bin(self) -> PositionACFSpectrumBin:
        if not math.isclose(
            self.period_frames,
            256.0 / self.bin_index,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("spectrum period_frames must equal 256/bin_index")
        return self


class TeacherBinRelation(StrictModel):
    """Relationship between the old velocity teacher and position ACF bins."""

    teacher_period_frames: float = Field(gt=0)
    teacher_confidence: float = Field(ge=0, le=1)
    teacher_selected_bin: int | None = Field(default=None, ge=2, le=64)
    position_selected_bin: int | None = Field(default=None, ge=2, le=64)
    teacher_over_position_bin_ratio: float | None = Field(default=None, gt=0)
    nearest_harmonic_2_to_7: int | None = Field(default=None, ge=2, le=7)
    nearest_harmonic_relative_error: float | None = Field(default=None, ge=0)
    within_20_percent_of_harmonic: bool = False

    @model_validator(mode="after")
    def validate_relation(self) -> TeacherBinRelation:
        evidence = (
            self.teacher_selected_bin is not None
            and self.position_selected_bin is not None
        )
        derived = (
            self.teacher_over_position_bin_ratio,
            self.nearest_harmonic_2_to_7,
            self.nearest_harmonic_relative_error,
        )
        if not evidence:
            if any(value is not None for value in derived):
                raise ValueError("bin relation requires both selected bins")
            if self.within_20_percent_of_harmonic:
                raise ValueError("missing bins cannot form a harmonic relation")
            return self
        assert self.teacher_selected_bin is not None
        assert self.position_selected_bin is not None
        ratio = self.teacher_selected_bin / self.position_selected_bin
        nearest = min(_HARMONIC_CANDIDATES, key=lambda value: abs(ratio - value))
        relative_error = abs(ratio - nearest) / nearest
        if self.teacher_over_position_bin_ratio != ratio:
            raise ValueError("teacher/position bin ratio mismatch")
        if self.nearest_harmonic_2_to_7 != nearest:
            raise ValueError("nearest 2--7 harmonic mismatch")
        if self.nearest_harmonic_relative_error != relative_error:
            raise ValueError("harmonic relative error mismatch")
        if self.within_20_percent_of_harmonic != (
            relative_error <= _HARMONIC_RELATIVE_TOLERANCE
        ):
            raise ValueError("harmonic threshold flag mismatch")
        return self


class PositionACFPredictionRow(StrictModel):
    """One target-free position-ACF prediction and complete spectrum."""

    video_id: str
    video_sha256: str
    raw_count: float = Field(ge=0)
    rounded_count: int = Field(ge=0)
    period_frames: float = Field(gt=0)
    valid_frames: int = Field(ge=0, le=256)
    expert_counts: tuple[int, int, int]
    confidence: float = Field(ge=0, le=1)
    position_selected_bin: int | None = Field(default=None, ge=2, le=64)
    allowed_spectrum: tuple[PositionACFSpectrumBin, ...]
    teacher_bin_relation: TeacherBinRelation
    period_stream: tuple[float, ...]

    @model_validator(mode="after")
    def validate_row(self) -> PositionACFPredictionRow:
        if not self.video_id.strip():
            raise ValueError("prediction video_id must be non-empty")
        _canonical_sha256(self.video_sha256, "prediction video_sha256")
        if not math.isfinite(self.raw_count):
            raise ValueError("raw_count must be finite")
        expected_raw = (
            0.0
            if self.confidence <= 0.0 or self.valid_frames < 2
            else float(self.valid_frames - 1) / self.period_frames
        )
        if self.raw_count != expected_raw:
            raise ValueError("raw_count does not match the frozen position-ACF formula")
        if self.rounded_count != round_count(self.raw_count):
            raise ValueError("rounded_count must use nearest-integer half-up rounding")
        if self.expert_counts != (self.rounded_count,) * 3:
            raise ValueError("expert_counts must repeat rounded_count")
        if len(self.period_stream) != 256:
            raise ValueError("period_stream must contain exactly 256 frames")
        if any(value != self.period_frames for value in self.period_stream):
            raise ValueError("period_stream must repeat period_frames exactly")
        if tuple(item.bin_index for item in self.allowed_spectrum) != _ALLOWED_BINS:
            raise ValueError("allowed_spectrum must contain bins 2 through 64")
        power_sum = math.fsum(item.power_share for item in self.allowed_spectrum)
        if self.confidence > 0:
            if self.position_selected_bin is None:
                raise ValueError("positive confidence requires a selected bin")
            selected = self.allowed_spectrum[self.position_selected_bin - 2]
            if selected.period_frames != self.period_frames:
                raise ValueError("selected spectrum period mismatch")
            if selected.power_share != self.confidence:
                raise ValueError("selected spectrum power share mismatch")
            if not math.isclose(power_sum, 1.0, rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError("positive spectrum power shares must sum to one")
        else:
            if self.position_selected_bin is not None:
                raise ValueError("zero confidence forbids a selected bin")
            if power_sum != 0.0:
                raise ValueError("zero-confidence spectrum must have zero power")
        if (
            self.teacher_bin_relation.position_selected_bin
            != self.position_selected_bin
        ):
            raise ValueError("teacher relation position bin mismatch")
        self.to_count_result()
        return self

    def to_count_result(self) -> CountResult:
        return CountResult(
            count=self.rounded_count,
            period_frames=self.period_frames,
            expert_counts=self.expert_counts,
            confidence=self.confidence,
            period_stream=np.asarray(self.period_stream, dtype=np.float32),
        )


class PositionACFPredictionArtifact(StrictModel):
    schema_version: Literal[1] = 1
    artifact_type: Literal["pams_position_acf_direct_dev_predictions"]
    classification: Literal[
        "PAMS-PositionACFDirect-inferred post-hoc diagnostic"
    ]
    eligible_for_paper_table: Literal[False] = False
    test_evaluation_authorized: Literal[False] = False
    protocol: Literal["ucfrep_526"]
    split: Literal["dev"]
    method_key: Literal["pams-position-acf-direct-inferred-v1"]
    rounding: Literal["nearest_integer_half_up"]
    prediction_batch_size: Literal[32] = 32
    spectrum_allowed_bins: Literal[63] = 63
    record_total: Literal[84]
    config_file_sha256: str
    config_fingerprint: str
    readout_config_file_sha256: str
    readout_config_fingerprint: str
    pose_fingerprint: str
    protocol_identity_sha256: str
    training_identity_sha256: str
    train_inputs_sha256: str
    train_commitment_sha256: str
    dev_inputs_sha256: str
    dev_commitment_sha256: str
    test_identity_inputs_sha256: str
    test_identity_commitment_sha256: str
    dev_identity_sha256: str
    training_pose_cache_set_sha256: str
    dev_pose_cache_set_sha256: str
    dev_pose_cache_snapshot_sha256: str
    checkpoint_sha256: str
    checkpoint_progress_sha256: str
    checkpoint_completion_receipt_sha256: str
    checkpoint_source_git_sha: str
    checkpoint_container_image_id: str
    checkpoint_container_environment_sha256: str
    prediction_source_git_sha: str
    prediction_container_image_id: str
    prediction_container_environment_sha256: str
    prediction_code_files_sha256: dict[str, str]
    prediction_code_sha256: str
    records: tuple[PositionACFPredictionRow, ...]

    @model_validator(mode="after")
    def validate_artifact(self) -> PositionACFPredictionArtifact:
        for field in (
            "config_file_sha256",
            "config_fingerprint",
            "readout_config_file_sha256",
            "readout_config_fingerprint",
            "pose_fingerprint",
            "protocol_identity_sha256",
            "training_identity_sha256",
            "train_inputs_sha256",
            "train_commitment_sha256",
            "dev_inputs_sha256",
            "dev_commitment_sha256",
            "test_identity_inputs_sha256",
            "test_identity_commitment_sha256",
            "dev_identity_sha256",
            "training_pose_cache_set_sha256",
            "dev_pose_cache_set_sha256",
            "dev_pose_cache_snapshot_sha256",
            "checkpoint_sha256",
            "checkpoint_progress_sha256",
            "checkpoint_completion_receipt_sha256",
            "checkpoint_container_environment_sha256",
            "prediction_container_environment_sha256",
            "prediction_code_sha256",
        ):
            _canonical_sha256(getattr(self, field), field)
        _canonical_git_sha(self.checkpoint_source_git_sha, "checkpoint_source_git_sha")
        _canonical_git_sha(self.prediction_source_git_sha, "prediction_source_git_sha")
        _canonical_image_id(
            self.checkpoint_container_image_id,
            "checkpoint_container_image_id",
        )
        _canonical_image_id(
            self.prediction_container_image_id,
            "prediction_container_image_id",
        )
        for name, digest in self.prediction_code_files_sha256.items():
            _canonical_sha256(digest, f"prediction_code_files_sha256[{name}]")
        if set(self.prediction_code_files_sha256) != set(_code_file_hashes()):
            raise ValueError("prediction code-file set is incomplete")
        if sha256_json(self.prediction_code_files_sha256) != self.prediction_code_sha256:
            raise ValueError("prediction_code_sha256 does not bind code files")
        if len(self.records) != self.record_total:
            raise ValueError("prediction record_total mismatch")
        identifiers = [row.video_id for row in self.records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("prediction video_id values must be unique")
        return self


class PositionACFPredictionReceipt(StrictModel):
    schema_version: Literal[1] = 1
    artifact_type: Literal["pams_position_acf_direct_dev_prediction_receipt"]
    test_evaluation_authorized: Literal[False] = False
    protocol: Literal["ucfrep_526"]
    split: Literal["dev"]
    method_key: Literal["pams-position-acf-direct-inferred-v1"]
    prediction_batch_size: Literal[32] = 32
    spectrum_allowed_bins: Literal[63] = 63
    prediction_file: Literal["predictions.json"]
    prediction_sha256: str
    prediction_bytes: int = Field(ge=1)
    config_fingerprint: str
    readout_config_file_sha256: str
    readout_config_fingerprint: str
    dev_identity_sha256: str
    checkpoint_sha256: str
    checkpoint_progress_sha256: str
    checkpoint_completion_receipt_sha256: str
    training_pose_cache_set_sha256: str
    dev_pose_cache_set_sha256: str
    dev_pose_cache_snapshot_sha256: str
    prediction_source_git_sha: str
    prediction_container_image_id: str
    prediction_container_environment_sha256: str
    prediction_code_sha256: str
    command: tuple[str, ...]
    hardware: dict[str, Any]

    @model_validator(mode="after")
    def validate_receipt(self) -> PositionACFPredictionReceipt:
        for field in (
            "prediction_sha256",
            "config_fingerprint",
            "readout_config_file_sha256",
            "readout_config_fingerprint",
            "dev_identity_sha256",
            "checkpoint_sha256",
            "checkpoint_progress_sha256",
            "checkpoint_completion_receipt_sha256",
            "training_pose_cache_set_sha256",
            "dev_pose_cache_set_sha256",
            "dev_pose_cache_snapshot_sha256",
            "prediction_container_environment_sha256",
            "prediction_code_sha256",
        ):
            _canonical_sha256(getattr(self, field), field)
        _canonical_git_sha(self.prediction_source_git_sha, "prediction_source_git_sha")
        _canonical_image_id(
            self.prediction_container_image_id,
            "prediction_container_image_id",
        )
        if not self.command or any(not item for item in self.command):
            raise ValueError("prediction command must contain non-empty strings")
        json.dumps(self.hardware, allow_nan=False)
        return self


def _teacher_relation(
    *,
    teacher_period: float,
    teacher_confidence: float,
    position_selected_bin: int | None,
) -> TeacherBinRelation:
    teacher_bin = (
        None
        if teacher_confidence <= 0
        else min(max(int(round(256.0 / teacher_period)), 2), 64)
    )
    if teacher_bin is None or position_selected_bin is None:
        return TeacherBinRelation(
            teacher_period_frames=teacher_period,
            teacher_confidence=teacher_confidence,
            teacher_selected_bin=teacher_bin,
            position_selected_bin=position_selected_bin,
        )
    ratio = teacher_bin / position_selected_bin
    nearest = min(_HARMONIC_CANDIDATES, key=lambda value: abs(ratio - value))
    relative_error = abs(ratio - nearest) / nearest
    return TeacherBinRelation(
        teacher_period_frames=teacher_period,
        teacher_confidence=teacher_confidence,
        teacher_selected_bin=teacher_bin,
        position_selected_bin=position_selected_bin,
        teacher_over_position_bin_ratio=ratio,
        nearest_harmonic_2_to_7=nearest,
        nearest_harmonic_relative_error=relative_error,
        within_20_percent_of_harmonic=(
            relative_error <= _HARMONIC_RELATIVE_TOLERANCE
        ),
    )


def _spectrum_bins(diagnostic: Any) -> tuple[PositionACFSpectrumBin, ...]:
    allowed = tuple(int(value) for value in diagnostic.allowed_bins)
    frequencies = tuple(float(value) for value in diagnostic.frequencies)
    periods = tuple(float(value) for value in diagnostic.periods)
    power_shares = tuple(float(value) for value in diagnostic.power_shares)
    if allowed != _ALLOWED_BINS:
        raise ValueError("position spectrum must expose exactly bins 2 through 64")
    if not (
        len(frequencies)
        == len(periods)
        == len(power_shares)
        == len(_ALLOWED_BINS)
    ):
        raise ValueError("position spectrum diagnostic length mismatch")
    for bin_index, frequency in zip(allowed, frequencies, strict=True):
        if not math.isclose(
            frequency,
            bin_index / 256.0,
            rel_tol=1e-6,
            abs_tol=1e-8,
        ):
            raise ValueError("position spectrum frequency/bin mismatch")
    return tuple(
        PositionACFSpectrumBin(
            bin_index=bin_index,
            period_frames=period,
            power_share=share,
        )
        for bin_index, period, share in zip(
            allowed,
            periods,
            power_shares,
            strict=True,
        )
    )


def predict_position_acf_direct_batches(
    model: Any,
    sequences: Sequence[PoseSequence],
    config: PAMSConfig,
) -> tuple[PositionACFPredictionRow, ...]:
    """Predict in the formal fixed batches 32, with a final short batch."""

    items = tuple(sequences)
    if not items:
        raise ValueError("position-acf prediction requires at least one sequence")
    if len({item.video_id for item in items}) != len(items):
        raise ValueError("position-acf sequence video_id values must be unique")
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    model.eval()
    rows: list[PositionACFPredictionRow] = []
    with torch.inference_mode():
        for start in range(0, len(items), _PREDICTION_BATCH_SIZE):
            batch_sequences = items[start : start + _PREDICTION_BATCH_SIZE]
            batch = collate_pose_sequences(batch_sequences).to(model_device)
            _, projected = model.encoder.forward_with_pre_pe(
                batch.poses,
                batch.valid_mask,
            )
            periods, confidences = estimate_period_from_projected_position(
                projected,
                minimum=config.period.minimum,
                maximum=config.period.maximum,
                valid_mask=batch.valid_mask,
            )
            diagnostics = projected_position_vector_acf_diagnostics(
                projected,
                minimum=config.period.minimum,
                maximum=config.period.maximum,
                valid_mask=batch.valid_mask,
            )
            teacher_periods, teacher_confidences = (
                estimate_period_from_projected_pose(
                    projected,
                    minimum=config.period.minimum,
                    maximum=config.period.maximum,
                    valid_mask=batch.valid_mask,
                )
            )
            if len(diagnostics) != len(batch.video_ids):
                raise RuntimeError("position spectrum diagnostic batch mismatch")
            valid_counts = batch.valid_mask.sum(dim=1)
            for index, video_id in enumerate(batch.video_ids):
                period = float(periods[index])
                confidence = float(confidences[index])
                valid_count = int(valid_counts[index])
                diagnostic = diagnostics[index]
                if int(diagnostic.valid_length) != valid_count:
                    raise RuntimeError("position diagnostic valid-length mismatch")
                selected_bin = (
                    None
                    if diagnostic.selected_bin is None
                    else int(diagnostic.selected_bin)
                )
                if not math.isclose(
                    float(diagnostic.selected_period),
                    period,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                ):
                    raise RuntimeError("position estimator/diagnostic period mismatch")
                if not math.isclose(
                    float(diagnostic.confidence),
                    confidence,
                    rel_tol=1e-6,
                    abs_tol=1e-7,
                ):
                    raise RuntimeError("position estimator/diagnostic confidence mismatch")
                raw_count = (
                    0.0
                    if confidence <= 0.0 or valid_count < 2
                    else float(valid_count - 1) / period
                )
                rounded_count = round_count(raw_count)
                teacher_period = float(teacher_periods[index])
                teacher_confidence = float(teacher_confidences[index])
                rows.append(
                    PositionACFPredictionRow(
                        video_id=video_id,
                        video_sha256="0" * 64,
                        raw_count=raw_count,
                        rounded_count=rounded_count,
                        period_frames=period,
                        valid_frames=valid_count,
                        expert_counts=(rounded_count,) * 3,
                        confidence=confidence,
                        position_selected_bin=selected_bin,
                        allowed_spectrum=_spectrum_bins(diagnostic),
                        teacher_bin_relation=_teacher_relation(
                            teacher_period=teacher_period,
                            teacher_confidence=teacher_confidence,
                            position_selected_bin=selected_bin,
                        ),
                        period_stream=(period,) * 256,
                    )
                )
    return tuple(rows)


def _require_exact_v8_artifact_bindings(
    artifact: PositionACFPredictionArtifact,
) -> None:
    expected = {
        "config_file_sha256": _V8_CONFIG_FILE_SHA256,
        "config_fingerprint": _V8_CONFIG_FINGERPRINT,
        "readout_config_file_sha256": _READOUT_CONFIG_FILE_SHA256,
        "readout_config_fingerprint": _READOUT_CONFIG_FINGERPRINT,
        "pose_fingerprint": _V8_POSE_FINGERPRINT,
        "protocol_identity_sha256": _V8_PROTOCOL_IDENTITY_SHA256,
        "training_identity_sha256": _V8_TRAINING_IDENTITY_SHA256,
        **_V8_INPUT_SHA256,
        "dev_identity_sha256": _V8_DEV_IDENTITY_SHA256,
        "training_pose_cache_set_sha256": _V8_TRAINING_POSE_CACHE_SET_SHA256,
        "dev_pose_cache_set_sha256": _V8_DEV_POSE_CACHE_SET_SHA256,
        "dev_pose_cache_snapshot_sha256": _V8_DEV_POSE_CACHE_SNAPSHOT_SHA256,
        "checkpoint_sha256": _V8_ENCODER_SHA256,
        "checkpoint_progress_sha256": _V8_ENCODER_PROGRESS_SHA256,
        "checkpoint_completion_receipt_sha256": (
            _V8_ENCODER_COMPLETION_RECEIPT_SHA256
        ),
        "checkpoint_source_git_sha": _V8_CHECKPOINT_SOURCE_GIT_SHA,
        "checkpoint_container_image_id": _V8_CHECKPOINT_CONTAINER_IMAGE_ID,
        "checkpoint_container_environment_sha256": (
            _V8_CHECKPOINT_CONTAINER_ENVIRONMENT_SHA256
        ),
    }
    mismatches = [
        f"{field}: observed={getattr(artifact, field)}, expected={value}"
        for field, value in expected.items()
        if getattr(artifact, field) != value
    ]
    if mismatches:
        raise ValueError(
            "position-acf artifact does not bind the exact frozen v8 run; "
            + "; ".join(mismatches)
        )


def run_dev_prediction(
    *,
    checkpoint_path: str | Path,
    checkpoint_progress_path: str | Path,
    checkpoint_completion_receipt_path: str | Path,
    train_inputs_path: str | Path,
    train_commitment_path: str | Path,
    dev_inputs_path: str | Path,
    dev_commitment_path: str | Path,
    test_identity_inputs_path: str | Path,
    test_identity_commitment_path: str | Path,
    pose_cache_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    readout_config_path: str | Path,
    device: str | None,
    repository_root: str | Path,
    command: Sequence[str],
) -> dict[str, Any]:
    """Freeze 84 position-ACF dev predictions without accepting targets."""

    checkpoint = Path(checkpoint_path)
    progress = Path(checkpoint_progress_path)
    completion_receipt = Path(checkpoint_completion_receipt_path)
    train_inputs = Path(train_inputs_path)
    train_commitment = Path(train_commitment_path)
    dev_inputs = Path(dev_inputs_path)
    dev_commitment = Path(dev_commitment_path)
    test_inputs = Path(test_identity_inputs_path)
    test_commitment = Path(test_identity_commitment_path)
    cache_dir = Path(pose_cache_dir)
    destination = Path(output_dir)
    config_file = Path(config_path)
    readout_config_file = Path(readout_config_path)
    repository = Path(repository_root)
    output_paths = {
        "pose_snapshot": destination / "inputs" / _POSE_SNAPSHOT_NAME,
        "predictions": destination / _PREDICTION_NAME,
        "receipt": destination / _PREDICTION_RECEIPT_NAME,
    }
    collisions = [str(path) for path in output_paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(
            f"refusing to overwrite position-acf dev artifacts: {collisions}"
        )

    prediction_source_git_sha = clean_git_revision(repository)
    prediction_image_id, prediction_environment_sha256 = (
        strict_dev._runtime_container_identity(prediction_source_git_sha)
    )
    code_files_sha256 = _code_file_hashes()
    code_sha256 = sha256_json(code_files_sha256)
    config_file_sha256 = sha256_file(config_file)
    config = load_config(config_file)
    readout_config_file_sha256 = sha256_file(readout_config_file)
    readout_config = load_position_acf_readout_config(readout_config_file)
    if config.protocol != "ucfrep_526":
        raise ValueError("position-acf dev prediction requires protocol ucfrep_526")
    inputs, input_hashes = strict_dev._load_protocol_inputs(
        train_inputs_path=train_inputs,
        train_commitment_path=train_commitment,
        dev_inputs_path=dev_inputs,
        dev_commitment_path=dev_commitment,
        test_identity_inputs_path=test_inputs,
        test_identity_commitment_path=test_commitment,
    )
    _require_exact_v8_bindings(
        config_path=config_file,
        config=config,
        checkpoint_path=checkpoint,
        progress_path=progress,
        completion_receipt_path=completion_receipt,
        input_hashes=input_hashes,
    )
    training_records = inputs.training_records(include_dev=False)
    _, training_pose_snapshot = load_pose_cache_set(
        training_records,
        cache_dir=cache_dir,
        pose_fingerprint=config.pose_fingerprint,
        materialize_sequences=False,
    )
    input_paths = {
        "train_inputs_sha256": train_inputs,
        "train_commitment_sha256": train_commitment,
        "dev_inputs_sha256": dev_inputs,
        "dev_commitment_sha256": dev_commitment,
        "test_identity_inputs_sha256": test_inputs,
        "test_identity_commitment_sha256": test_commitment,
    }
    training_binding = strict_dev._validate_completed_training_receipt(
        completion_receipt,
        expected_stage="encoder",
        checkpoint_path=checkpoint,
        progress_path=progress,
        config_path=config_file,
        config=config,
        inputs=inputs,
        input_paths=input_paths,
        input_hashes=input_hashes,
        training_pose_snapshot=training_pose_snapshot,
        upstream_encoder_checkpoint_path=None,
        upstream_encoder_progress_path=None,
    )
    model, checkpoint_provenance, checkpoint_hashes = (
        strict_dev._validated_checkpoint_model(
            checkpoint_path=checkpoint,
            checkpoint_progress_path=progress,
            variant="literal",
            checkpoint_training_binding=training_binding,
            upstream_encoder_checkpoint_path=None,
            upstream_encoder_progress_path=None,
            upstream_encoder_training_binding=None,
            config=config,
            inputs=inputs,
            training_pose_snapshot=training_pose_snapshot,
            device=device,
        )
    )
    dev_records = inputs.records_for("dev")
    sequences, dev_pose_snapshot = load_pose_cache_set(
        dev_records,
        cache_dir=cache_dir,
        pose_fingerprint=config.pose_fingerprint,
    )
    expected_ids = tuple(record.video_id for record in dev_records)
    if tuple(sequence.video_id for sequence in sequences) != expected_ids:
        raise RuntimeError("dev pose-cache order does not match committed inputs")
    predicted = predict_position_acf_direct_batches(model, sequences, config)
    video_hashes = {
        record.video_id: record.video_sha256 or "" for record in dev_records
    }
    rows = tuple(
        PositionACFPredictionRow.model_validate(
            {
                **row.model_dump(mode="json"),
                "video_sha256": video_hashes[row.video_id],
            }
        )
        for row in predicted
    )
    if tuple(row.video_id for row in rows) != expected_ids:
        raise RuntimeError("prediction order does not match committed dev inputs")

    source_paths = {
        "config_file_sha256": config_file,
        "readout_config_file_sha256": readout_config_file,
        "train_inputs_sha256": train_inputs,
        "train_commitment_sha256": train_commitment,
        "dev_inputs_sha256": dev_inputs,
        "dev_commitment_sha256": dev_commitment,
        "test_identity_inputs_sha256": test_inputs,
        "test_identity_commitment_sha256": test_commitment,
        "checkpoint_sha256": checkpoint,
        "checkpoint_progress_sha256": progress,
        "checkpoint_completion_receipt_sha256": completion_receipt,
    }
    for role, path in source_paths.items():
        expected_sha256 = (
            config_file_sha256
            if role == "config_file_sha256"
            else (
                readout_config_file_sha256
                if role == "readout_config_file_sha256"
                else input_hashes.get(role, checkpoint_hashes.get(role))
            )
        )
        if expected_sha256 is None or sha256_file(path) != expected_sha256:
            raise RuntimeError(f"{role} changed during position-acf prediction")
    if _code_file_hashes() != code_files_sha256:
        raise RuntimeError("position-acf source changed during prediction")
    if clean_git_revision(repository) != prediction_source_git_sha:
        raise RuntimeError("position-acf prediction Git revision changed")

    pose_snapshot_payload = dev_pose_snapshot.to_dict()
    pose_snapshot_sha256 = hashlib.sha256(
        strict_dev._encoded_json(pose_snapshot_payload)
    ).hexdigest()
    artifact = PositionACFPredictionArtifact(
        artifact_type="pams_position_acf_direct_dev_predictions",
        classification=_CLASSIFICATION,
        protocol="ucfrep_526",
        split="dev",
        method_key=_METHOD_KEY,
        rounding=_ROUNDING,
        record_total=84,
        config_file_sha256=config_file_sha256,
        config_fingerprint=config.fingerprint,
        readout_config_file_sha256=readout_config_file_sha256,
        readout_config_fingerprint=readout_config.fingerprint,
        pose_fingerprint=config.pose_fingerprint,
        protocol_identity_sha256=inputs.fingerprint,
        training_identity_sha256=inputs.training_fingerprint(include_dev=False),
        train_inputs_sha256=input_hashes["train_inputs_sha256"],
        train_commitment_sha256=input_hashes["train_commitment_sha256"],
        dev_inputs_sha256=input_hashes["dev_inputs_sha256"],
        dev_commitment_sha256=input_hashes["dev_commitment_sha256"],
        test_identity_inputs_sha256=input_hashes["test_identity_inputs_sha256"],
        test_identity_commitment_sha256=input_hashes[
            "test_identity_commitment_sha256"
        ],
        dev_identity_sha256=pose_input_identity_sha256(dev_records),
        training_pose_cache_set_sha256=training_pose_snapshot.fingerprint,
        dev_pose_cache_set_sha256=dev_pose_snapshot.fingerprint,
        dev_pose_cache_snapshot_sha256=pose_snapshot_sha256,
        checkpoint_sha256=str(checkpoint_hashes["checkpoint_sha256"]),
        checkpoint_progress_sha256=str(
            checkpoint_hashes["checkpoint_progress_sha256"]
        ),
        checkpoint_completion_receipt_sha256=str(
            checkpoint_hashes["checkpoint_completion_receipt_sha256"]
        ),
        checkpoint_source_git_sha=checkpoint_provenance.source_git_sha,
        checkpoint_container_image_id=checkpoint_provenance.container_image_id
        or "",
        checkpoint_container_environment_sha256=(
            checkpoint_provenance.container_environment_sha256 or ""
        ),
        prediction_source_git_sha=prediction_source_git_sha,
        prediction_container_image_id=prediction_image_id,
        prediction_container_environment_sha256=prediction_environment_sha256,
        prediction_code_files_sha256=code_files_sha256,
        prediction_code_sha256=code_sha256,
        records=rows,
    )
    _require_exact_v8_artifact_bindings(artifact)
    prediction_payload = artifact.model_dump(mode="json")
    prediction_encoded = strict_dev._encoded_json(prediction_payload)
    prediction_sha256 = hashlib.sha256(prediction_encoded).hexdigest()
    receipt = PositionACFPredictionReceipt(
        artifact_type="pams_position_acf_direct_dev_prediction_receipt",
        test_evaluation_authorized=False,
        protocol="ucfrep_526",
        split="dev",
        method_key=_METHOD_KEY,
        prediction_file=_PREDICTION_NAME,
        prediction_sha256=prediction_sha256,
        prediction_bytes=len(prediction_encoded),
        config_fingerprint=config.fingerprint,
        readout_config_file_sha256=readout_config_file_sha256,
        readout_config_fingerprint=readout_config.fingerprint,
        dev_identity_sha256=artifact.dev_identity_sha256,
        checkpoint_sha256=artifact.checkpoint_sha256,
        checkpoint_progress_sha256=artifact.checkpoint_progress_sha256,
        checkpoint_completion_receipt_sha256=(
            artifact.checkpoint_completion_receipt_sha256
        ),
        training_pose_cache_set_sha256=training_pose_snapshot.fingerprint,
        dev_pose_cache_set_sha256=dev_pose_snapshot.fingerprint,
        dev_pose_cache_snapshot_sha256=pose_snapshot_sha256,
        prediction_source_git_sha=prediction_source_git_sha,
        prediction_container_image_id=prediction_image_id,
        prediction_container_environment_sha256=prediction_environment_sha256,
        prediction_code_sha256=code_sha256,
        command=tuple(str(item) for item in command),
        hardware=hardware_fingerprint(),
    )
    written = strict_dev._write_json_bundle(
        (
            (output_paths["pose_snapshot"], pose_snapshot_payload),
            (output_paths["predictions"], prediction_payload),
            (output_paths["receipt"], receipt.model_dump(mode="json")),
        )
    )
    return {
        "classification": _CLASSIFICATION,
        "eligible_for_paper_table": False,
        "test_evaluation_authorized": False,
        "split": "dev",
        "method_key": _METHOD_KEY,
        "record_total": 84,
        "spectrum_allowed_bins": 63,
        "readout_config_file_sha256": readout_config_file_sha256,
        "readout_config_fingerprint": readout_config.fingerprint,
        "predictions_path": str(output_paths["predictions"].resolve()),
        "predictions_sha256": prediction_sha256,
        "prediction_receipt_path": str(output_paths["receipt"].resolve()),
        "prediction_receipt_sha256": written[output_paths["receipt"]],
        "pose_cache_snapshot_path": str(output_paths["pose_snapshot"].resolve()),
        "pose_cache_snapshot_sha256": pose_snapshot_sha256,
        "checkpoint_sha256": artifact.checkpoint_sha256,
        "checkpoint_progress_sha256": artifact.checkpoint_progress_sha256,
        "checkpoint_completion_receipt_sha256": (
            artifact.checkpoint_completion_receipt_sha256
        ),
        "prediction_source_git_sha": prediction_source_git_sha,
        "prediction_code_sha256": code_sha256,
    }


def _load_prediction_artifact(path: Path) -> PositionACFPredictionArtifact:
    return PositionACFPredictionArtifact.model_validate(
        strict_dev._load_json_object(
            path,
            document_name="position-acf dev predictions",
        )
    )


def _load_prediction_receipt(path: Path) -> PositionACFPredictionReceipt:
    return PositionACFPredictionReceipt.model_validate(
        strict_dev._load_json_object(
            path,
            document_name="position-acf dev prediction receipt",
        )
    )


def _validate_receipt_binding(
    artifact: PositionACFPredictionArtifact,
    receipt: PositionACFPredictionReceipt,
    *,
    prediction_path: Path,
    prediction_sha256: str,
) -> None:
    if prediction_path.name != receipt.prediction_file:
        raise ValueError("prediction filename does not match its receipt")
    if prediction_sha256 != receipt.prediction_sha256:
        raise ValueError("prediction SHA-256 does not match its receipt")
    if prediction_path.stat().st_size != receipt.prediction_bytes:
        raise ValueError("prediction byte count does not match its receipt")
    field_pairs = {
        "protocol": (artifact.protocol, receipt.protocol),
        "split": (artifact.split, receipt.split),
        "method_key": (artifact.method_key, receipt.method_key),
        "test_evaluation_authorized": (
            artifact.test_evaluation_authorized,
            receipt.test_evaluation_authorized,
        ),
        "prediction_batch_size": (
            artifact.prediction_batch_size,
            receipt.prediction_batch_size,
        ),
        "spectrum_allowed_bins": (
            artifact.spectrum_allowed_bins,
            receipt.spectrum_allowed_bins,
        ),
        "config_fingerprint": (
            artifact.config_fingerprint,
            receipt.config_fingerprint,
        ),
        "readout_config_file_sha256": (
            artifact.readout_config_file_sha256,
            receipt.readout_config_file_sha256,
        ),
        "readout_config_fingerprint": (
            artifact.readout_config_fingerprint,
            receipt.readout_config_fingerprint,
        ),
        "dev_identity_sha256": (
            artifact.dev_identity_sha256,
            receipt.dev_identity_sha256,
        ),
        "checkpoint_sha256": (
            artifact.checkpoint_sha256,
            receipt.checkpoint_sha256,
        ),
        "checkpoint_progress_sha256": (
            artifact.checkpoint_progress_sha256,
            receipt.checkpoint_progress_sha256,
        ),
        "checkpoint_completion_receipt_sha256": (
            artifact.checkpoint_completion_receipt_sha256,
            receipt.checkpoint_completion_receipt_sha256,
        ),
        "training_pose_cache_set_sha256": (
            artifact.training_pose_cache_set_sha256,
            receipt.training_pose_cache_set_sha256,
        ),
        "dev_pose_cache_set_sha256": (
            artifact.dev_pose_cache_set_sha256,
            receipt.dev_pose_cache_set_sha256,
        ),
        "dev_pose_cache_snapshot_sha256": (
            artifact.dev_pose_cache_snapshot_sha256,
            receipt.dev_pose_cache_snapshot_sha256,
        ),
        "prediction_source_git_sha": (
            artifact.prediction_source_git_sha,
            receipt.prediction_source_git_sha,
        ),
        "prediction_container_image_id": (
            artifact.prediction_container_image_id,
            receipt.prediction_container_image_id,
        ),
        "prediction_container_environment_sha256": (
            artifact.prediction_container_environment_sha256,
            receipt.prediction_container_environment_sha256,
        ),
        "prediction_code_sha256": (
            artifact.prediction_code_sha256,
            receipt.prediction_code_sha256,
        ),
    }
    mismatches = [
        field for field, values in field_pairs.items() if values[0] != values[1]
    ]
    if mismatches:
        raise ValueError(f"prediction receipt metadata mismatch: {mismatches}")


def _validate_canonical_dev_identity(
    artifact: PositionACFPredictionArtifact,
) -> None:
    records = tuple(
        UnlabeledVideoRecord(
            video_id=row.video_id,
            video_path=f"frozen-dev/{row.video_id}.avi",
            video_sha256=row.video_sha256,
        )
        for row in artifact.records
    )
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split="dev",
        records=records,
    )
    manifest.validate_exact_membership()
    if pose_input_identity_sha256(records) != artifact.dev_identity_sha256:
        raise ValueError("prediction rows do not bind the declared dev identity")


def _reject_test_target_locator(path: Path) -> None:
    if "test" in path.name.casefold():
        raise ValueError("position-acf scorer permanently rejects test targets")


def score_dev_predictions(
    *,
    predictions_path: str | Path,
    prediction_receipt_path: str | Path,
    dev_targets_path: str | Path,
    output_dir: str | Path,
    readout_config_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Score an already-frozen artifact at the only label-bearing boundary."""

    predictions_source = Path(predictions_path)
    receipt_source = Path(prediction_receipt_path)
    targets_source = Path(dev_targets_path)
    readout_config_source = Path(readout_config_path)
    destination = Path(output_dir)
    repository = Path(repository_root)
    evaluation_path = destination / _EVALUATION_NAME
    evaluation_receipt_path = destination / _EVALUATION_RECEIPT_NAME
    collisions = [
        str(path)
        for path in (evaluation_path, evaluation_receipt_path)
        if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            f"refusing to overwrite position-acf score artifacts: {collisions}"
        )

    # Do not stat, hash, or deserialize the target path above this boundary.
    readout_config_file_sha256 = sha256_file(readout_config_source)
    readout_config = load_position_acf_readout_config(readout_config_source)
    receipt_sha256 = sha256_file(receipt_source)
    receipt = _load_prediction_receipt(receipt_source)
    prediction_sha256 = sha256_file(predictions_source)
    artifact = _load_prediction_artifact(predictions_source)
    _validate_receipt_binding(
        artifact,
        receipt,
        prediction_path=predictions_source,
        prediction_sha256=prediction_sha256,
    )
    _require_exact_v8_artifact_bindings(artifact)
    if artifact.readout_config_file_sha256 != readout_config_file_sha256:
        raise ValueError("scorer readout configuration file differs from prediction")
    if artifact.readout_config_fingerprint != readout_config.fingerprint:
        raise ValueError("scorer readout configuration differs from prediction")
    _validate_canonical_dev_identity(artifact)
    scoring_source_git_sha = clean_git_revision(repository)
    if scoring_source_git_sha != artifact.prediction_source_git_sha:
        raise ValueError("scorer source revision differs from prediction source revision")
    scoring_code_files_sha256 = _code_file_hashes()
    scoring_code_sha256 = sha256_json(scoring_code_files_sha256)
    if scoring_code_files_sha256 != artifact.prediction_code_files_sha256:
        raise ValueError("scorer source files differ from prediction source files")
    if sha256_file(predictions_source) != prediction_sha256:
        raise RuntimeError("prediction artifact changed while it was validated")
    if sha256_file(receipt_source) != receipt_sha256:
        raise RuntimeError("prediction receipt changed while it was validated")
    if sha256_file(readout_config_source) != readout_config_file_sha256:
        raise RuntimeError("readout configuration changed while it was validated")
    _reject_test_target_locator(targets_source)

    # First permitted target access.
    targets_sha256 = sha256_file(targets_source)
    targets = load_dev_target_manifest(targets_source)
    target_ids = tuple(record.video_id for record in targets.records)
    prediction_ids = tuple(record.video_id for record in artifact.records)
    if target_ids != prediction_ids:
        raise ValueError("dev targets do not exactly match frozen prediction order")
    if sha256_file(targets_source) != targets_sha256:
        raise RuntimeError("dev-target manifest changed while it was loaded")

    prediction_records = tuple(
        PredictionRecord(video_id=row.video_id, result=row.to_count_result())
        for row in artifact.records
    )
    result = evaluate_predictions(
        prediction_records,
        {record.video_id: record.count for record in targets.records},
        actions={record.video_id: record.action for record in targets.records},
        bootstrap_samples=10_000,
        bootstrap_seed=2026,
        confidence_level=0.95,
    )
    if sha256_file(predictions_source) != prediction_sha256:
        raise RuntimeError("prediction artifact changed during scoring")
    if sha256_file(receipt_source) != receipt_sha256:
        raise RuntimeError("prediction receipt changed during scoring")
    if sha256_file(targets_source) != targets_sha256:
        raise RuntimeError("dev-target manifest changed during scoring")
    if sha256_file(readout_config_source) != readout_config_file_sha256:
        raise RuntimeError("readout configuration changed during scoring")
    if _code_file_hashes() != scoring_code_files_sha256:
        raise RuntimeError("position-acf scoring source changed during scoring")
    if clean_git_revision(repository) != scoring_source_git_sha:
        raise RuntimeError("position-acf scoring Git revision changed")

    evaluation_payload = {
        "schema_version": 1,
        "artifact_type": "pams_position_acf_direct_dev_evaluation",
        "classification": _CLASSIFICATION,
        "eligible_for_paper_table": False,
        "test_evaluation_authorized": False,
        "protocol": artifact.protocol,
        "split": "dev",
        "method_key": artifact.method_key,
        "rounding": artifact.rounding,
        "prediction_batch_size": artifact.prediction_batch_size,
        "spectrum_allowed_bins": artifact.spectrum_allowed_bins,
        "bootstrap_pairing": "paired_prediction_target_rows",
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "dev_targets_sha256": targets_sha256,
        "config_file_sha256": artifact.config_file_sha256,
        "config_fingerprint": artifact.config_fingerprint,
        "readout_config_file_sha256": artifact.readout_config_file_sha256,
        "readout_config_fingerprint": artifact.readout_config_fingerprint,
        "dev_identity_sha256": artifact.dev_identity_sha256,
        "checkpoint_sha256": artifact.checkpoint_sha256,
        "checkpoint_progress_sha256": artifact.checkpoint_progress_sha256,
        "checkpoint_completion_receipt_sha256": (
            artifact.checkpoint_completion_receipt_sha256
        ),
        "checkpoint_source_git_sha": artifact.checkpoint_source_git_sha,
        "prediction_source_git_sha": artifact.prediction_source_git_sha,
        "prediction_code_files_sha256": artifact.prediction_code_files_sha256,
        "prediction_code_sha256": artifact.prediction_code_sha256,
        "scoring_source_git_sha": scoring_source_git_sha,
        "scoring_code_files_sha256": scoring_code_files_sha256,
        "scoring_code_sha256": scoring_code_sha256,
        "report": result.report.to_dict(),
        "predictions": [
            {
                "video_id": row.video_id,
                "raw_count": row.raw_count,
                "rounded_count": row.rounded_count,
                "period_frames": row.period_frames,
                "valid_frames": row.valid_frames,
                "confidence": row.confidence,
                "position_selected_bin": row.position_selected_bin,
                "teacher_bin_relation": row.teacher_bin_relation.model_dump(
                    mode="json"
                ),
            }
            for row in artifact.records
        ],
    }
    evaluation_encoded = strict_dev._encoded_json(evaluation_payload)
    evaluation_sha256 = hashlib.sha256(evaluation_encoded).hexdigest()
    evaluation_receipt = {
        "schema_version": 1,
        "artifact_type": "pams_position_acf_direct_dev_evaluation_receipt",
        "protocol": artifact.protocol,
        "split": "dev",
        "method_key": artifact.method_key,
        "prediction_batch_size": artifact.prediction_batch_size,
        "spectrum_allowed_bins": artifact.spectrum_allowed_bins,
        "evaluation_file": _EVALUATION_NAME,
        "evaluation_sha256": evaluation_sha256,
        "evaluation_bytes": len(evaluation_encoded),
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "dev_targets_sha256": targets_sha256,
        "readout_config_file_sha256": artifact.readout_config_file_sha256,
        "readout_config_fingerprint": artifact.readout_config_fingerprint,
        "dev_identity_sha256": artifact.dev_identity_sha256,
        "checkpoint_sha256": artifact.checkpoint_sha256,
        "checkpoint_progress_sha256": artifact.checkpoint_progress_sha256,
        "checkpoint_completion_receipt_sha256": (
            artifact.checkpoint_completion_receipt_sha256
        ),
        "prediction_source_git_sha": artifact.prediction_source_git_sha,
        "prediction_code_sha256": artifact.prediction_code_sha256,
        "scoring_source_git_sha": scoring_source_git_sha,
        "scoring_code_sha256": scoring_code_sha256,
        "eligible_for_paper_table": False,
        "test_evaluation_authorized": False,
    }
    written = strict_dev._write_json_bundle(
        (
            (evaluation_path, evaluation_payload),
            (evaluation_receipt_path, evaluation_receipt),
        )
    )
    compact_metrics = result.report.to_dict()
    compact_metrics.pop("per_video")
    return {
        "classification": _CLASSIFICATION,
        "eligible_for_paper_table": False,
        "test_evaluation_authorized": False,
        "split": "dev",
        "method_key": artifact.method_key,
        "evaluation_path": str(evaluation_path.resolve()),
        "evaluation_sha256": evaluation_sha256,
        "evaluation_receipt_path": str(evaluation_receipt_path.resolve()),
        "evaluation_receipt_sha256": written[evaluation_receipt_path],
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "dev_targets_sha256": targets_sha256,
        "readout_config_file_sha256": artifact.readout_config_file_sha256,
        "readout_config_fingerprint": artifact.readout_config_fingerprint,
        "metrics": compact_metrics,
    }
