"""Strict dev-only runner for the frozen-v8 direct teacher-period diagnostic.

This module intentionally does not implement a new trainable head.  It loads
the exact completed v8 encoder, exposes the already-existing pre-position-
encoding projection, and applies the same projected-pose vector-ACF period
teacher used by the inferred training protocol.  The resulting method is an
independently inferred, post-hoc representation diagnostic.  It is never
eligible for a paper table and can never authorize opening the sealed test.

Prediction and scoring are separate processes.  ``run_dev_prediction`` has no
target argument.  ``score_dev_predictions`` validates the frozen prediction
bytes and their receipt before it first opens the dev-target manifest.
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
from pydantic import Field, model_validator

import pams.pams_dev as strict_dev
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

_METHOD_KEY: Literal["pams-teacher-period-direct-inferred-v1"] = (
    "pams-teacher-period-direct-inferred-v1"
)
_CLASSIFICATION: Literal["PAMS-TeacherPeriodDirect-inferred post-hoc diagnostic"] = (
    "PAMS-TeacherPeriodDirect-inferred post-hoc diagnostic"
)
_ROUNDING: Literal["nearest_integer_half_up"] = "nearest_integer_half_up"
_PREDICTION_NAME: Literal["predictions.json"] = "predictions.json"
_PREDICTION_RECEIPT_NAME = "prediction.receipt.json"
_POSE_SNAPSHOT_NAME = "dev-pose-cache-snapshot.json"
_EVALUATION_NAME = "evaluation.json"
_EVALUATION_RECEIPT_NAME = "evaluation.receipt.json"
_PREDICTION_BATCH_SIZE: Literal[32] = 32

# These immutable identities make this command a diagnostic of one completed
# encoder, not a generic hyperparameter or checkpoint search interface.
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
    "train_inputs_sha256": ("e238d307b4ed37b2f6c19eefdf7fa8812ec3d231cd2a7091be38e1c279f04207"),
    "train_commitment_sha256": ("ce39b1c9038bb1506354de010e46a23053b307b852222332f13df3f79d799044"),
    "dev_inputs_sha256": ("f21c49569805e4aea326977f6d7ce2964f503fad129a17e95214594364ca4ae7"),
    "dev_commitment_sha256": ("a2bb2db86abef64e16a8fe1ac196857ac8e77cc9cd461c568b2193401a8cd169"),
    "test_identity_inputs_sha256": (
        "9eb2a057246b77fc7a1121ba4c9d27bf55177ad2fcd5dbb87237396a4a6015ca"
    ),
    "test_identity_commitment_sha256": (
        "232d3db7ce09018716594a1f1b332eec47b37410a8d3068e662b026ee2dc9c66"
    ),
}


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
        "reproducibility": sha256_file(directory / "reproducibility.py"),
        "teacher_period_dev": sha256_file(Path(__file__)),
        "training": sha256_file(directory / "training.py"),
        "types": sha256_file(directory / "types.py"),
    }


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
            "teacher-period diagnostic accepts only the exact frozen v8 encoder "
            "package; " + "; ".join(mismatches)
        )


class TeacherPeriodPredictionRow(StrictModel):
    """One target-free direct-period prediction."""

    video_id: str
    video_sha256: str
    raw_count: float = Field(ge=0)
    rounded_count: int = Field(ge=0)
    period_frames: float = Field(gt=0)
    valid_frames: int = Field(ge=0, le=256)
    expert_counts: tuple[int, int, int]
    confidence: float = Field(ge=0, le=1)
    period_stream: tuple[float, ...]

    @model_validator(mode="after")
    def validate_row(self) -> TeacherPeriodPredictionRow:
        if not self.video_id.strip():
            raise ValueError("prediction video_id must be non-empty")
        _canonical_sha256(self.video_sha256, "prediction video_sha256")
        if not math.isfinite(self.raw_count):
            raise ValueError("raw_count must be finite")
        expected_raw_count = (
            0.0
            if self.confidence <= 0.0 or self.valid_frames < 2
            else float(self.valid_frames - 1) / self.period_frames
        )
        if self.raw_count != expected_raw_count:
            raise ValueError("raw_count does not match the frozen direct-period formula")
        if self.rounded_count != round_count(self.raw_count):
            raise ValueError("rounded_count must use nearest-integer half-up rounding")
        if self.expert_counts != (self.rounded_count,) * 3:
            raise ValueError("direct-period expert counts must repeat rounded_count")
        if len(self.period_stream) != 256:
            raise ValueError("direct-period stream must contain exactly 256 frames")
        if any(value != self.period_frames for value in self.period_stream):
            raise ValueError("direct-period stream must repeat period_frames exactly")
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


class TeacherPeriodPredictionArtifact(StrictModel):
    """Strict target-free artifact for the frozen-v8 diagnostic."""

    schema_version: Literal[1] = 1
    artifact_type: Literal["pams_teacher_period_direct_dev_predictions"]
    classification: Literal["PAMS-TeacherPeriodDirect-inferred post-hoc diagnostic"]
    eligible_for_paper_table: Literal[False] = False
    test_evaluation_authorized: Literal[False] = False
    protocol: Literal["ucfrep_526"]
    split: Literal["dev"]
    method_key: Literal["pams-teacher-period-direct-inferred-v1"]
    rounding: Literal["nearest_integer_half_up"]
    prediction_batch_size: Literal[32] = 32
    record_total: Literal[84]
    config_file_sha256: str
    config_fingerprint: str
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
    records: tuple[TeacherPeriodPredictionRow, ...]

    @model_validator(mode="after")
    def validate_artifact(self) -> TeacherPeriodPredictionArtifact:
        for field in (
            "config_file_sha256",
            "config_fingerprint",
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


class TeacherPeriodPredictionReceipt(StrictModel):
    """Independent commitment to already-frozen prediction bytes."""

    schema_version: Literal[1] = 1
    artifact_type: Literal["pams_teacher_period_direct_dev_prediction_receipt"]
    protocol: Literal["ucfrep_526"]
    split: Literal["dev"]
    method_key: Literal["pams-teacher-period-direct-inferred-v1"]
    prediction_batch_size: Literal[32] = 32
    prediction_file: Literal["predictions.json"]
    prediction_sha256: str
    prediction_bytes: int = Field(ge=1)
    config_fingerprint: str
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
    def validate_receipt(self) -> TeacherPeriodPredictionReceipt:
        for field in (
            "prediction_sha256",
            "config_fingerprint",
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


def predict_teacher_period_direct(
    model: Any,
    sequence: PoseSequence,
    config: PAMSConfig,
) -> TeacherPeriodPredictionRow:
    """Apply the direct pre-PE period teacher to one target-free sequence."""

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    inputs = torch.from_numpy(
        np.array(sequence.flattened(), dtype=np.float32, copy=True)
    ).unsqueeze(0)
    valid = torch.from_numpy(np.array(sequence.valid_mask, dtype=np.bool_, copy=True)).unsqueeze(0)
    inputs = inputs.to(model_device)
    valid = valid.to(model_device)
    model.eval()
    with torch.inference_mode():
        _, projected = model.encoder.forward_with_pre_pe(inputs, valid)
        periods, confidences = estimate_period_from_projected_pose(
            projected,
            minimum=config.period.minimum,
            maximum=config.period.maximum,
            valid_mask=valid,
        )
    period = float(periods[0].detach().cpu())
    confidence = float(confidences[0].detach().cpu())
    valid_count = int(valid.sum().detach().cpu())
    raw_count = 0.0 if confidence <= 0.0 or valid_count < 2 else float(valid_count - 1) / period
    rounded_count = round_count(raw_count)
    result = CountResult(
        count=rounded_count,
        period_frames=period,
        expert_counts=(rounded_count,) * 3,
        confidence=confidence,
        period_stream=np.full(sequence.num_frames, period, dtype=np.float32),
    )
    return TeacherPeriodPredictionRow(
        video_id=sequence.video_id,
        video_sha256="0" * 64,
        raw_count=raw_count,
        rounded_count=result.count,
        period_frames=result.period_frames,
        valid_frames=valid_count,
        expert_counts=result.expert_counts,
        confidence=result.confidence,
        period_stream=tuple(float(value) for value in result.period_stream),
    )


def predict_teacher_period_direct_batches(
    model: Any,
    sequences: Sequence[PoseSequence],
    config: PAMSConfig,
) -> tuple[TeacherPeriodPredictionRow, ...]:
    """Match the frozen server diagnostic's fixed 32-sequence CUDA batches."""

    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    model.eval()
    rows: list[TeacherPeriodPredictionRow] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), _PREDICTION_BATCH_SIZE):
            batch_sequences = sequences[start : start + _PREDICTION_BATCH_SIZE]
            batch = collate_pose_sequences(batch_sequences).to(model_device)
            _, projected = model.encoder.forward_with_pre_pe(
                batch.poses,
                batch.valid_mask,
            )
            periods, confidences = estimate_period_from_projected_pose(
                projected,
                minimum=config.period.minimum,
                maximum=config.period.maximum,
                valid_mask=batch.valid_mask,
            )
            valid_counts = batch.valid_mask.sum(dim=1)
            for index, video_id in enumerate(batch.video_ids):
                period = float(periods[index])
                confidence = float(confidences[index])
                valid_count = int(valid_counts[index])
                raw_count = (
                    0.0 if confidence <= 0.0 or valid_count < 2 else float(valid_count - 1) / period
                )
                rounded_count = round_count(raw_count)
                rows.append(
                    TeacherPeriodPredictionRow(
                        video_id=video_id,
                        video_sha256="0" * 64,
                        raw_count=raw_count,
                        rounded_count=rounded_count,
                        period_frames=period,
                        valid_frames=valid_count,
                        expert_counts=(rounded_count,) * 3,
                        confidence=confidence,
                        period_stream=tuple(period for _ in range(256)),
                    )
                )
    return tuple(rows)


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
    device: str | None,
    repository_root: str | Path,
    command: Sequence[str],
) -> dict[str, Any]:
    """Freeze exactly 84 direct-period dev predictions without a target input."""

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
    repository = Path(repository_root)
    output_paths = {
        "pose_snapshot": destination / "inputs" / _POSE_SNAPSHOT_NAME,
        "predictions": destination / _PREDICTION_NAME,
        "receipt": destination / _PREDICTION_RECEIPT_NAME,
    }
    collisions = [str(path) for path in output_paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite teacher-period dev artifacts: {collisions}")

    prediction_source_git_sha = clean_git_revision(repository)
    prediction_image_id, prediction_environment_sha256 = strict_dev._runtime_container_identity(
        prediction_source_git_sha
    )
    code_files_sha256 = _code_file_hashes()
    code_sha256 = sha256_json(code_files_sha256)
    config_file_sha256 = sha256_file(config_file)
    config = load_config(config_file)
    if config.protocol != "ucfrep_526":
        raise ValueError("teacher-period dev prediction requires protocol ucfrep_526")
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
    model, checkpoint_provenance, checkpoint_hashes = strict_dev._validated_checkpoint_model(
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
    dev_records = inputs.records_for("dev")
    sequences, dev_pose_snapshot = load_pose_cache_set(
        dev_records,
        cache_dir=cache_dir,
        pose_fingerprint=config.pose_fingerprint,
    )
    expected_ids = tuple(record.video_id for record in dev_records)
    if tuple(sequence.video_id for sequence in sequences) != expected_ids:
        raise RuntimeError("dev pose-cache order does not match committed inputs")

    video_hashes = {record.video_id: record.video_sha256 or "" for record in dev_records}
    rows = [
        row.model_copy(update={"video_sha256": video_hashes[row.video_id]})
        for row in predict_teacher_period_direct_batches(model, sequences, config)
    ]
    if tuple(row.video_id for row in rows) != expected_ids:
        raise RuntimeError("prediction order does not match committed dev inputs")

    source_paths = {
        "config_file_sha256": config_file,
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
            else input_hashes.get(role, checkpoint_hashes.get(role))
        )
        if expected_sha256 is None or sha256_file(path) != expected_sha256:
            raise RuntimeError(f"{role} changed during teacher-period prediction")
    if _code_file_hashes() != code_files_sha256:
        raise RuntimeError("teacher-period source changed during prediction")
    if clean_git_revision(repository) != prediction_source_git_sha:
        raise RuntimeError("teacher-period prediction Git revision changed")

    pose_snapshot_payload = dev_pose_snapshot.to_dict()
    pose_snapshot_sha256 = hashlib.sha256(
        strict_dev._encoded_json(pose_snapshot_payload)
    ).hexdigest()
    artifact = TeacherPeriodPredictionArtifact(
        artifact_type="pams_teacher_period_direct_dev_predictions",
        classification=_CLASSIFICATION,
        protocol="ucfrep_526",
        split="dev",
        method_key=_METHOD_KEY,
        rounding=_ROUNDING,
        record_total=84,
        config_file_sha256=config_file_sha256,
        config_fingerprint=config.fingerprint,
        pose_fingerprint=config.pose_fingerprint,
        protocol_identity_sha256=inputs.fingerprint,
        training_identity_sha256=inputs.training_fingerprint(include_dev=False),
        train_inputs_sha256=input_hashes["train_inputs_sha256"],
        train_commitment_sha256=input_hashes["train_commitment_sha256"],
        dev_inputs_sha256=input_hashes["dev_inputs_sha256"],
        dev_commitment_sha256=input_hashes["dev_commitment_sha256"],
        test_identity_inputs_sha256=input_hashes["test_identity_inputs_sha256"],
        test_identity_commitment_sha256=input_hashes["test_identity_commitment_sha256"],
        dev_identity_sha256=pose_input_identity_sha256(dev_records),
        training_pose_cache_set_sha256=training_pose_snapshot.fingerprint,
        dev_pose_cache_set_sha256=dev_pose_snapshot.fingerprint,
        dev_pose_cache_snapshot_sha256=pose_snapshot_sha256,
        checkpoint_sha256=str(checkpoint_hashes["checkpoint_sha256"]),
        checkpoint_progress_sha256=str(checkpoint_hashes["checkpoint_progress_sha256"]),
        checkpoint_completion_receipt_sha256=str(
            checkpoint_hashes["checkpoint_completion_receipt_sha256"]
        ),
        checkpoint_source_git_sha=checkpoint_provenance.source_git_sha,
        checkpoint_container_image_id=checkpoint_provenance.container_image_id or "",
        checkpoint_container_environment_sha256=(
            checkpoint_provenance.container_environment_sha256 or ""
        ),
        prediction_source_git_sha=prediction_source_git_sha,
        prediction_container_image_id=prediction_image_id,
        prediction_container_environment_sha256=prediction_environment_sha256,
        prediction_code_files_sha256=code_files_sha256,
        prediction_code_sha256=code_sha256,
        records=tuple(rows),
    )
    _require_exact_v8_artifact_bindings(artifact)
    prediction_payload = artifact.model_dump(mode="json")
    prediction_encoded = strict_dev._encoded_json(prediction_payload)
    prediction_sha256 = hashlib.sha256(prediction_encoded).hexdigest()
    receipt = TeacherPeriodPredictionReceipt(
        artifact_type="pams_teacher_period_direct_dev_prediction_receipt",
        protocol="ucfrep_526",
        split="dev",
        method_key=_METHOD_KEY,
        prediction_file=_PREDICTION_NAME,
        prediction_sha256=prediction_sha256,
        prediction_bytes=len(prediction_encoded),
        config_fingerprint=config.fingerprint,
        dev_identity_sha256=artifact.dev_identity_sha256,
        checkpoint_sha256=artifact.checkpoint_sha256,
        checkpoint_progress_sha256=artifact.checkpoint_progress_sha256,
        checkpoint_completion_receipt_sha256=(artifact.checkpoint_completion_receipt_sha256),
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
        "predictions_path": str(output_paths["predictions"].resolve()),
        "predictions_sha256": prediction_sha256,
        "prediction_receipt_path": str(output_paths["receipt"].resolve()),
        "prediction_receipt_sha256": written[output_paths["receipt"]],
        "pose_cache_snapshot_path": str(output_paths["pose_snapshot"].resolve()),
        "pose_cache_snapshot_sha256": pose_snapshot_sha256,
        "checkpoint_sha256": artifact.checkpoint_sha256,
        "checkpoint_progress_sha256": artifact.checkpoint_progress_sha256,
        "checkpoint_completion_receipt_sha256": (artifact.checkpoint_completion_receipt_sha256),
        "prediction_source_git_sha": prediction_source_git_sha,
        "prediction_code_sha256": code_sha256,
    }


def _load_prediction_artifact(path: Path) -> TeacherPeriodPredictionArtifact:
    return TeacherPeriodPredictionArtifact.model_validate(
        strict_dev._load_json_object(
            path,
            document_name="teacher-period dev predictions",
        )
    )


def _load_prediction_receipt(path: Path) -> TeacherPeriodPredictionReceipt:
    return TeacherPeriodPredictionReceipt.model_validate(
        strict_dev._load_json_object(
            path,
            document_name="teacher-period dev prediction receipt",
        )
    )


def _require_exact_v8_artifact_bindings(
    artifact: TeacherPeriodPredictionArtifact,
) -> None:
    """Reject a self-consistent receipt unless it binds the one frozen v8 run."""

    expected = {
        "config_file_sha256": _V8_CONFIG_FILE_SHA256,
        "config_fingerprint": _V8_CONFIG_FINGERPRINT,
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
        "checkpoint_completion_receipt_sha256": (_V8_ENCODER_COMPLETION_RECEIPT_SHA256),
        "checkpoint_source_git_sha": _V8_CHECKPOINT_SOURCE_GIT_SHA,
        "checkpoint_container_image_id": _V8_CHECKPOINT_CONTAINER_IMAGE_ID,
        "checkpoint_container_environment_sha256": (_V8_CHECKPOINT_CONTAINER_ENVIRONMENT_SHA256),
    }
    mismatches = [
        f"{field}: observed={getattr(artifact, field)}, expected={value}"
        for field, value in expected.items()
        if getattr(artifact, field) != value
    ]
    if mismatches:
        raise ValueError(
            "teacher-period artifact does not bind the exact frozen v8 run; "
            + "; ".join(mismatches)
        )


def _validate_receipt_binding(
    artifact: TeacherPeriodPredictionArtifact,
    receipt: TeacherPeriodPredictionReceipt,
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
        "prediction_batch_size": (
            artifact.prediction_batch_size,
            receipt.prediction_batch_size,
        ),
        "config_fingerprint": (
            artifact.config_fingerprint,
            receipt.config_fingerprint,
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
    mismatches = [field for field, values in field_pairs.items() if values[0] != values[1]]
    if mismatches:
        raise ValueError(f"prediction receipt metadata mismatch: {mismatches}")


def _validate_canonical_dev_identity(
    artifact: TeacherPeriodPredictionArtifact,
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


def score_dev_predictions(
    *,
    predictions_path: str | Path,
    prediction_receipt_path: str | Path,
    dev_targets_path: str | Path,
    output_dir: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Score frozen diagnostic predictions at the only label-bearing boundary."""

    predictions_source = Path(predictions_path)
    receipt_source = Path(prediction_receipt_path)
    targets_source = Path(dev_targets_path)
    destination = Path(output_dir)
    repository = Path(repository_root)
    evaluation_path = destination / _EVALUATION_NAME
    evaluation_receipt_path = destination / _EVALUATION_RECEIPT_NAME
    collisions = [str(path) for path in (evaluation_path, evaluation_receipt_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite teacher-period score artifacts: {collisions}")

    # Do not stat, hash, or deserialize the target path above this boundary.
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

    # First permitted access to labels.
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
    if _code_file_hashes() != scoring_code_files_sha256:
        raise RuntimeError("teacher-period scoring source changed during scoring")
    if clean_git_revision(repository) != scoring_source_git_sha:
        raise RuntimeError("teacher-period scoring Git revision changed")

    evaluation_payload = {
        "schema_version": 1,
        "artifact_type": "pams_teacher_period_direct_dev_evaluation",
        "classification": _CLASSIFICATION,
        "eligible_for_paper_table": False,
        "test_evaluation_authorized": False,
        "protocol": artifact.protocol,
        "split": "dev",
        "method_key": artifact.method_key,
        "rounding": artifact.rounding,
        "prediction_batch_size": artifact.prediction_batch_size,
        "bootstrap_pairing": "paired_prediction_target_rows",
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "dev_targets_sha256": targets_sha256,
        "config_file_sha256": artifact.config_file_sha256,
        "config_fingerprint": artifact.config_fingerprint,
        "dev_identity_sha256": artifact.dev_identity_sha256,
        "checkpoint_sha256": artifact.checkpoint_sha256,
        "checkpoint_progress_sha256": artifact.checkpoint_progress_sha256,
        "checkpoint_completion_receipt_sha256": (artifact.checkpoint_completion_receipt_sha256),
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
            }
            for row in artifact.records
        ],
    }
    evaluation_encoded = strict_dev._encoded_json(evaluation_payload)
    evaluation_sha256 = hashlib.sha256(evaluation_encoded).hexdigest()
    evaluation_receipt = {
        "schema_version": 1,
        "artifact_type": "pams_teacher_period_direct_dev_evaluation_receipt",
        "protocol": artifact.protocol,
        "split": "dev",
        "method_key": artifact.method_key,
        "prediction_batch_size": artifact.prediction_batch_size,
        "evaluation_file": _EVALUATION_NAME,
        "evaluation_sha256": evaluation_sha256,
        "evaluation_bytes": len(evaluation_encoded),
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "dev_targets_sha256": targets_sha256,
        "dev_identity_sha256": artifact.dev_identity_sha256,
        "checkpoint_sha256": artifact.checkpoint_sha256,
        "checkpoint_progress_sha256": artifact.checkpoint_progress_sha256,
        "checkpoint_completion_receipt_sha256": (artifact.checkpoint_completion_receipt_sha256),
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
        "metrics": compact_metrics,
    }
