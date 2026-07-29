"""Target-isolated UCFRep dev stress prediction and scoring."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from pams.config import StrictModel, load_config
from pams.data import (
    load_dev_target_manifest,
    load_pose_cache_set,
    pose_input_identity_sha256,
)
from pams.evaluation import PredictionRecord, evaluate_predictions, predict_sequences
from pams.metrics import paired_bootstrap_difference
from pams.pams_dev import (
    PAMSDevPredictionRow,
    PAMSDevVariant,
    PAMSInferenceExpertMode,
    _canonical_git_sha,
    _canonical_image_id,
    _canonical_sha256,
    _encoded_json,
    _inference_config,
    _load_bound_inputs,
    _load_json_object,
    _load_prediction_artifact,
    _load_prediction_receipt,
    _prediction_row,
    _runtime_container_identity,
    _validate_canonical_dev_identity,
    _validate_receipt_binding,
    _write_json_bundle,
)
from pams.reproducibility import (
    clean_git_revision,
    hardware_fingerprint,
    sha256_file,
    sha256_json,
)
from pams.stress import apply_stress_condition, load_stress_protocol
from pams.training import load_model_checkpoint
from pams.types import CountResult

_PREDICTIONS_NAME: Literal["stress-predictions.json"] = "stress-predictions.json"
_PREDICTION_RECEIPT_NAME = "stress-prediction.receipt.json"
_EVALUATION_NAME = "stress-evaluation.json"
_EVALUATION_RECEIPT_NAME = "stress-evaluation.receipt.json"
_CLASSIFICATION: Literal["PAMS dev pose-cache stress diagnostic"] = (
    "PAMS dev pose-cache stress diagnostic"
)


class StressPredictionRow(StrictModel):
    """One target-free prediction for one video and one perturbation."""

    video_id: str
    video_sha256: str
    count: int = Field(ge=0)
    period_frames: float = Field(gt=0)
    expert_counts: tuple[int, int, int]
    selection_mode: PAMSInferenceExpertMode
    selected_expert: Literal["fast", "medium", "slow"] | None
    confidence: float = Field(ge=0, le=1)
    period_stream: tuple[float, ...]
    perturbation_algorithm: str
    perturbation_parameters: dict[str, Any]
    transformed_pose_sha256: str

    @model_validator(mode="after")
    def validate_row(self) -> StressPredictionRow:
        if not self.video_id.strip() or not self.perturbation_algorithm.strip():
            raise ValueError("stress row identifiers must be non-empty")
        _canonical_sha256(self.video_sha256, "stress row video_sha256")
        _canonical_sha256(
            self.transformed_pose_sha256,
            "stress row transformed_pose_sha256",
        )
        json.dumps(self.perturbation_parameters, sort_keys=True, allow_nan=False)
        self.to_count_result()
        if self.selection_mode == "medium_only":
            if self.selected_expert != "medium" or self.count != self.expert_counts[1]:
                raise ValueError("medium_only stress row must select the medium expert")
        elif self.selected_expert is not None:
            raise ValueError("multi stress rows cannot infer one selected expert")
        return self

    def to_count_result(self) -> CountResult:
        return CountResult(
            count=self.count,
            period_frames=self.period_frames,
            expert_counts=self.expert_counts,
            confidence=self.confidence,
            period_stream=np.asarray(self.period_stream, dtype=np.float32),
        )


class StressConditionPredictions(StrictModel):
    """All 84 dev predictions under one condition."""

    condition_id: str
    perturbation_kind: str
    configured_parameters: dict[str, Any]
    record_total: Literal[84]
    records: tuple[StressPredictionRow, ...]

    @model_validator(mode="after")
    def validate_condition(self) -> StressConditionPredictions:
        if not self.condition_id.strip() or not self.perturbation_kind.strip():
            raise ValueError("condition identifiers must be non-empty")
        if len(self.records) != self.record_total:
            raise ValueError("stress condition record_total mismatch")
        identifiers = [row.video_id for row in self.records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("stress condition video IDs must be unique")
        return self


class StressPredictionArtifact(StrictModel):
    """Immutable target-free output of the full stress suite."""

    schema_version: Literal[1] = 1
    artifact_type: Literal["pams_checkpoint_dev_stress_predictions"]
    classification: Literal["PAMS dev pose-cache stress diagnostic"]
    table_eligible: Literal[False] = False
    protocol: Literal["ucfrep_526"]
    split: Literal["dev"]
    target_free_prediction: Literal[True] = True
    tune_on_stress_results: Literal[False] = False
    variant: PAMSDevVariant
    method_key: str
    consensus_expert_mode: PAMSInferenceExpertMode
    record_total_per_condition: Literal[84]
    condition_total: int = Field(ge=1)
    stress_key: Literal["pams_stress_v1"]
    stress_seed: int = Field(ge=0)
    stress_config_sha256: str
    stress_code_files_sha256: dict[str, str]
    stress_code_sha256: str
    source_clean_predictions_sha256: str
    source_clean_prediction_receipt_sha256: str
    source_config_file_sha256: str
    source_training_config_fingerprint: str
    source_inference_config_fingerprint: str
    source_protocol_identity_sha256: str
    source_training_identity_sha256: str
    source_dev_identity_sha256: str
    source_dev_inputs_sha256: str
    source_dev_commitment_sha256: str
    source_dev_pose_cache_set_sha256: str
    source_checkpoint_sha256: str
    source_checkpoint_progress_sha256: str
    source_checkpoint_completion_receipt_sha256: str
    source_upstream_encoder_checkpoint_sha256: str | None
    source_upstream_encoder_progress_sha256: str | None
    source_upstream_encoder_completion_receipt_sha256: str | None
    prediction_source_git_sha: str
    prediction_container_image_id: str
    prediction_container_environment_sha256: str
    clean_replay_matches_source: Literal[True]
    conditions: tuple[StressConditionPredictions, ...]

    @model_validator(mode="after")
    def validate_artifact(self) -> StressPredictionArtifact:
        hashes = (
            self.stress_config_sha256,
            self.stress_code_sha256,
            self.source_clean_predictions_sha256,
            self.source_clean_prediction_receipt_sha256,
            self.source_config_file_sha256,
            self.source_training_config_fingerprint,
            self.source_inference_config_fingerprint,
            self.source_protocol_identity_sha256,
            self.source_training_identity_sha256,
            self.source_dev_identity_sha256,
            self.source_dev_inputs_sha256,
            self.source_dev_commitment_sha256,
            self.source_dev_pose_cache_set_sha256,
            self.source_checkpoint_sha256,
            self.source_checkpoint_progress_sha256,
            self.source_checkpoint_completion_receipt_sha256,
            self.prediction_container_environment_sha256,
        )
        for index, digest in enumerate(hashes):
            _canonical_sha256(digest, f"stress artifact hash {index}")
        for index, optional_digest in enumerate(
            (
                self.source_upstream_encoder_checkpoint_sha256,
                self.source_upstream_encoder_progress_sha256,
                self.source_upstream_encoder_completion_receipt_sha256,
            )
        ):
            if optional_digest is not None:
                _canonical_sha256(
                    optional_digest,
                    f"stress artifact upstream hash {index}",
                )
        _canonical_git_sha(self.prediction_source_git_sha, "prediction_source_git_sha")
        _canonical_image_id(
            self.prediction_container_image_id,
            "prediction_container_image_id",
        )
        if set(self.stress_code_files_sha256) != {
            "config",
            "consensus",
            "data",
            "evaluation",
            "pams_dev",
            "stress",
            "stress_dev",
            "training",
        }:
            raise ValueError("stress source hash set is incomplete")
        for name, digest in self.stress_code_files_sha256.items():
            _canonical_sha256(digest, f"stress_code_files_sha256[{name}]")
        if sha256_json(self.stress_code_files_sha256) != self.stress_code_sha256:
            raise ValueError("stress_code_sha256 does not bind source hashes")
        if len(self.conditions) != self.condition_total:
            raise ValueError("condition_total mismatch")
        condition_ids = [condition.condition_id for condition in self.conditions]
        if not condition_ids or condition_ids[0] != "clean":
            raise ValueError("clean must be the first stress condition")
        if len(set(condition_ids)) != len(condition_ids):
            raise ValueError("stress condition IDs must be unique")
        reference_ids = tuple(row.video_id for row in self.conditions[0].records)
        if any(
            tuple(row.video_id for row in condition.records) != reference_ids
            for condition in self.conditions
        ):
            raise ValueError("all stress conditions must preserve canonical dev order")
        return self


class StressPredictionReceipt(StrictModel):
    """Independent byte receipt for the stress prediction artifact."""

    schema_version: Literal[1] = 1
    artifact_type: Literal["pams_checkpoint_dev_stress_prediction_receipt"]
    protocol: Literal["ucfrep_526"]
    split: Literal["dev"]
    prediction_file: Literal["stress-predictions.json"]
    prediction_sha256: str
    prediction_bytes: int = Field(ge=1)
    stress_config_sha256: str
    stress_code_sha256: str
    source_clean_predictions_sha256: str
    source_clean_prediction_receipt_sha256: str
    source_checkpoint_sha256: str
    prediction_source_git_sha: str
    prediction_container_image_id: str
    prediction_container_environment_sha256: str
    command: tuple[str, ...]
    hardware: dict[str, Any]

    @model_validator(mode="after")
    def validate_receipt(self) -> StressPredictionReceipt:
        for index, digest in enumerate(
            (
                self.prediction_sha256,
                self.stress_config_sha256,
                self.stress_code_sha256,
                self.source_clean_predictions_sha256,
                self.source_clean_prediction_receipt_sha256,
                self.source_checkpoint_sha256,
                self.prediction_container_environment_sha256,
            )
        ):
            _canonical_sha256(digest, f"stress receipt hash {index}")
        _canonical_git_sha(self.prediction_source_git_sha, "prediction_source_git_sha")
        _canonical_image_id(
            self.prediction_container_image_id,
            "prediction_container_image_id",
        )
        if not self.command or any(not item for item in self.command):
            raise ValueError("stress prediction command must contain non-empty strings")
        json.dumps(self.hardware, sort_keys=True, allow_nan=False)
        return self


def _stress_code_file_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    return {
        "config": sha256_file(directory / "config.py"),
        "consensus": sha256_file(directory / "consensus.py"),
        "data": sha256_file(directory / "data.py"),
        "evaluation": sha256_file(directory / "evaluation.py"),
        "pams_dev": sha256_file(directory / "pams_dev.py"),
        "stress": sha256_file(directory / "stress.py"),
        "stress_dev": sha256_file(Path(__file__)),
        "training": sha256_file(directory / "training.py"),
    }


def _load_stress_artifact(path: Path) -> StressPredictionArtifact:
    return StressPredictionArtifact.model_validate(
        _load_json_object(path, document_name="PAMS dev stress predictions")
    )


def _load_stress_receipt(path: Path) -> StressPredictionReceipt:
    return StressPredictionReceipt.model_validate(
        _load_json_object(path, document_name="PAMS dev stress prediction receipt")
    )


def _stress_row(
    base: PAMSDevPredictionRow,
    *,
    algorithm: str,
    parameters: dict[str, Any],
    transformed_pose_sha256: str,
) -> StressPredictionRow:
    return StressPredictionRow(
        **base.model_dump(mode="python"),
        perturbation_algorithm=algorithm,
        perturbation_parameters=parameters,
        transformed_pose_sha256=transformed_pose_sha256,
    )


def _count_result_equal(left: PAMSDevPredictionRow, right: StressPredictionRow) -> bool:
    return (
        left.video_id == right.video_id
        and left.count == right.count
        and left.period_frames == right.period_frames
        and left.expert_counts == right.expert_counts
        and left.selection_mode == right.selection_mode
        and left.selected_expert == right.selected_expert
        and left.confidence == right.confidence
        and left.period_stream == right.period_stream
    )


def run_pams_dev_stress_prediction(
    *,
    source_clean_predictions_path: str | Path,
    source_clean_prediction_receipt_path: str | Path,
    checkpoint_path: str | Path,
    dev_inputs_path: str | Path,
    dev_commitment_path: str | Path,
    pose_cache_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    stress_config_path: str | Path,
    device: str | None,
    repository_root: str | Path,
    command: Sequence[str],
) -> dict[str, Any]:
    """Run every configured stress condition without accepting target labels."""

    clean_path = Path(source_clean_predictions_path)
    clean_receipt_path = Path(source_clean_prediction_receipt_path)
    checkpoint = Path(checkpoint_path)
    dev_inputs = Path(dev_inputs_path)
    dev_commitment = Path(dev_commitment_path)
    cache_dir = Path(pose_cache_dir)
    destination = Path(output_dir)
    config_file = Path(config_path)
    stress_config = Path(stress_config_path)
    repository = Path(repository_root)
    prediction_path = destination / _PREDICTIONS_NAME
    receipt_path = destination / _PREDICTION_RECEIPT_NAME
    collisions = [str(path) for path in (prediction_path, receipt_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite stress artifacts: {collisions}")

    source_git_sha = clean_git_revision(repository)
    image_id, environment_sha256 = _runtime_container_identity(source_git_sha)
    code_files_sha256 = _stress_code_file_hashes()
    code_sha256 = sha256_json(code_files_sha256)
    protocol = load_stress_protocol(stress_config)
    config_file_sha256 = sha256_file(config_file)
    clean_prediction_sha256 = sha256_file(clean_path)
    clean_receipt_sha256 = sha256_file(clean_receipt_path)
    clean_artifact = _load_prediction_artifact(clean_path)
    clean_receipt = _load_prediction_receipt(clean_receipt_path)
    _validate_receipt_binding(
        clean_artifact,
        clean_receipt,
        prediction_path=clean_path,
        prediction_sha256=clean_prediction_sha256,
    )
    _validate_canonical_dev_identity(clean_artifact)
    if clean_artifact.consensus_expert_mode != "multi":
        raise ValueError("stress suite is frozen to the default multi-expert protocol")

    training_config = load_config(config_file)
    inference_config = _inference_config(
        training_config,
        clean_artifact.consensus_expert_mode,
    )
    if config_file_sha256 != clean_artifact.config_file_sha256:
        raise ValueError("config bytes differ from the source clean prediction")
    if training_config.fingerprint != clean_artifact.training_config_fingerprint:
        raise ValueError("training config fingerprint differs from source clean prediction")
    if inference_config.fingerprint != clean_artifact.config_fingerprint:
        raise ValueError("inference config fingerprint differs from source clean prediction")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != clean_artifact.checkpoint_sha256:
        raise ValueError("checkpoint bytes differ from source clean prediction")

    dev_manifest, dev_inputs_sha256, dev_commitment_sha256 = _load_bound_inputs(
        dev_inputs,
        dev_commitment,
        expected_split="dev",
    )
    if dev_inputs_sha256 != clean_artifact.dev_inputs_sha256:
        raise ValueError("dev input bytes differ from source clean prediction")
    if dev_commitment_sha256 != clean_artifact.dev_commitment_sha256:
        raise ValueError("dev commitment differs from source clean prediction")
    dev_records = dev_manifest.records
    if pose_input_identity_sha256(dev_records) != clean_artifact.dev_identity_sha256:
        raise ValueError("dev identity differs from source clean prediction")
    sequences, pose_snapshot = load_pose_cache_set(
        dev_records,
        cache_dir=cache_dir,
        pose_fingerprint=training_config.pose_fingerprint,
    )
    if pose_snapshot.fingerprint != clean_artifact.dev_pose_cache_set_sha256:
        raise ValueError("dev pose-cache set differs from source clean prediction")
    expected_ids = tuple(record.video_id for record in dev_records)
    if tuple(sequence.video_id for sequence in sequences) != expected_ids:
        raise RuntimeError("dev pose-cache order differs from the committed manifest")

    expected_stage: Literal["encoder", "sshead"] = (
        "encoder" if clean_artifact.variant == "literal" else "sshead"
    )
    model = load_model_checkpoint(
        checkpoint,
        training_config,
        device=device,
        expected_stage=expected_stage,
    )
    video_hashes = {record.video_id: record.video_sha256 or "" for record in dev_records}
    condition_outputs: list[StressConditionPredictions] = []
    for condition in protocol.conditions:
        transformed_items = []
        audits = {}
        for sequence in sequences:
            transformed, audit = apply_stress_condition(
                sequence,
                condition,
                seed=protocol.seed,
            )
            transformed_items.append(transformed)
            audits[sequence.video_id] = audit
        predictions = predict_sequences(
            model,
            tuple(transformed_items),
            inference_config,
            device=device,
        )
        if tuple(prediction.video_id for prediction in predictions) != expected_ids:
            raise RuntimeError("stress prediction order differs from canonical dev order")
        rows = []
        for prediction in predictions:
            base = _prediction_row(
                prediction,
                video_sha256=video_hashes[prediction.video_id],
                selection_mode=clean_artifact.consensus_expert_mode,
            )
            audit = audits[prediction.video_id]
            rows.append(
                _stress_row(
                    base,
                    algorithm=audit.algorithm,
                    parameters=audit.parameters,
                    transformed_pose_sha256=audit.transformed_pose_sha256,
                )
            )
        condition_outputs.append(
            StressConditionPredictions(
                condition_id=condition.condition_id,
                perturbation_kind=condition.kind,
                configured_parameters=condition.parameters,
                record_total=84,
                records=tuple(rows),
            )
        )

    source_clean_rows = clean_artifact.records
    replay_clean_rows = condition_outputs[0].records
    if len(source_clean_rows) != len(replay_clean_rows) or any(
        not _count_result_equal(source, replay)
        for source, replay in zip(source_clean_rows, replay_clean_rows, strict=True)
    ):
        raise RuntimeError("clean stress replay differs from frozen source prediction")

    source_files = (
        (clean_path, clean_prediction_sha256, "source clean predictions"),
        (clean_receipt_path, clean_receipt_sha256, "source clean receipt"),
        (checkpoint, checkpoint_sha256, "checkpoint"),
        (config_file, config_file_sha256, "config"),
        (stress_config, protocol.config_sha256, "stress config"),
        (dev_inputs, dev_inputs_sha256, "dev inputs"),
        (dev_commitment, dev_commitment_sha256, "dev commitment"),
    )
    for path, expected, name in source_files:
        if sha256_file(path) != expected:
            raise RuntimeError(f"{name} changed during stress prediction")
    if _stress_code_file_hashes() != code_files_sha256:
        raise RuntimeError("stress prediction source code changed during prediction")
    if clean_git_revision(repository) != source_git_sha:
        raise RuntimeError("stress prediction Git revision changed during prediction")

    artifact = StressPredictionArtifact(
        artifact_type="pams_checkpoint_dev_stress_predictions",
        classification=_CLASSIFICATION,
        protocol="ucfrep_526",
        split="dev",
        variant=clean_artifact.variant,
        method_key=clean_artifact.method_key,
        consensus_expert_mode=clean_artifact.consensus_expert_mode,
        record_total_per_condition=84,
        condition_total=len(condition_outputs),
        stress_key="pams_stress_v1",
        stress_seed=protocol.seed,
        stress_config_sha256=protocol.config_sha256,
        stress_code_files_sha256=code_files_sha256,
        stress_code_sha256=code_sha256,
        source_clean_predictions_sha256=clean_prediction_sha256,
        source_clean_prediction_receipt_sha256=clean_receipt_sha256,
        source_config_file_sha256=config_file_sha256,
        source_training_config_fingerprint=clean_artifact.training_config_fingerprint,
        source_inference_config_fingerprint=clean_artifact.config_fingerprint,
        source_protocol_identity_sha256=clean_artifact.protocol_identity_sha256,
        source_training_identity_sha256=clean_artifact.training_identity_sha256,
        source_dev_identity_sha256=clean_artifact.dev_identity_sha256,
        source_dev_inputs_sha256=dev_inputs_sha256,
        source_dev_commitment_sha256=dev_commitment_sha256,
        source_dev_pose_cache_set_sha256=pose_snapshot.fingerprint,
        source_checkpoint_sha256=checkpoint_sha256,
        source_checkpoint_progress_sha256=clean_artifact.checkpoint_progress_sha256,
        source_checkpoint_completion_receipt_sha256=(
            clean_artifact.checkpoint_completion_receipt_sha256
        ),
        source_upstream_encoder_checkpoint_sha256=(
            clean_artifact.upstream_encoder_checkpoint_sha256
        ),
        source_upstream_encoder_progress_sha256=(
            clean_artifact.upstream_encoder_progress_sha256
        ),
        source_upstream_encoder_completion_receipt_sha256=(
            clean_artifact.upstream_encoder_completion_receipt_sha256
        ),
        prediction_source_git_sha=source_git_sha,
        prediction_container_image_id=image_id,
        prediction_container_environment_sha256=environment_sha256,
        clean_replay_matches_source=True,
        conditions=tuple(condition_outputs),
    )
    prediction_payload = artifact.model_dump(mode="json")
    encoded = _encoded_json(prediction_payload)
    prediction_sha256 = hashlib.sha256(encoded).hexdigest()
    receipt = StressPredictionReceipt(
        artifact_type="pams_checkpoint_dev_stress_prediction_receipt",
        protocol="ucfrep_526",
        split="dev",
        prediction_file=_PREDICTIONS_NAME,
        prediction_sha256=prediction_sha256,
        prediction_bytes=len(encoded),
        stress_config_sha256=protocol.config_sha256,
        stress_code_sha256=code_sha256,
        source_clean_predictions_sha256=clean_prediction_sha256,
        source_clean_prediction_receipt_sha256=clean_receipt_sha256,
        source_checkpoint_sha256=checkpoint_sha256,
        prediction_source_git_sha=source_git_sha,
        prediction_container_image_id=image_id,
        prediction_container_environment_sha256=environment_sha256,
        command=tuple(str(item) for item in command),
        hardware=hardware_fingerprint(),
    )
    written = _write_json_bundle(
        (
            (prediction_path, prediction_payload),
            (receipt_path, receipt.model_dump(mode="json")),
        )
    )
    return {
        "classification": _CLASSIFICATION,
        "table_eligible": False,
        "target_free_prediction": True,
        "split": "dev",
        "condition_total": len(condition_outputs),
        "record_total_per_condition": 84,
        "stress_config_sha256": protocol.config_sha256,
        "source_checkpoint_sha256": checkpoint_sha256,
        "clean_replay_matches_source": True,
        "predictions_path": str(prediction_path.resolve()),
        "predictions_sha256": prediction_sha256,
        "prediction_receipt_path": str(receipt_path.resolve()),
        "prediction_receipt_sha256": written[receipt_path],
    }


def _validate_stress_receipt_binding(
    artifact: StressPredictionArtifact,
    receipt: StressPredictionReceipt,
    *,
    prediction_path: Path,
    prediction_sha256: str,
) -> None:
    if prediction_path.name != receipt.prediction_file:
        raise ValueError("stress prediction filename differs from receipt")
    if prediction_sha256 != receipt.prediction_sha256:
        raise ValueError("stress prediction SHA-256 differs from receipt")
    if prediction_path.stat().st_size != receipt.prediction_bytes:
        raise ValueError("stress prediction byte count differs from receipt")
    pairs = (
        (artifact.stress_config_sha256, receipt.stress_config_sha256),
        (artifact.stress_code_sha256, receipt.stress_code_sha256),
        (
            artifact.source_clean_predictions_sha256,
            receipt.source_clean_predictions_sha256,
        ),
        (
            artifact.source_clean_prediction_receipt_sha256,
            receipt.source_clean_prediction_receipt_sha256,
        ),
        (artifact.source_checkpoint_sha256, receipt.source_checkpoint_sha256),
        (artifact.prediction_source_git_sha, receipt.prediction_source_git_sha),
        (
            artifact.prediction_container_image_id,
            receipt.prediction_container_image_id,
        ),
        (
            artifact.prediction_container_environment_sha256,
            receipt.prediction_container_environment_sha256,
        ),
    )
    if any(left != right for left, right in pairs):
        raise ValueError("stress prediction metadata differs from receipt")


def _paired_payload(
    condition_values: list[int],
    clean_values: list[int],
    targets: list[int],
    *,
    metric: Literal["nmae", "obo"],
) -> dict[str, Any]:
    result = paired_bootstrap_difference(
        condition_values,
        clean_values,
        targets,
        metric=metric,
        samples=10_000,
        seed=2026,
        confidence_level=0.95,
    )
    return {
        "metric": result.metric,
        "difference_condition_minus_clean": result.difference,
        "confidence_interval": result.confidence_interval.to_dict(),
        "samples": result.samples,
        "seed": result.seed,
    }


def score_pams_dev_stress_predictions(
    *,
    predictions_path: str | Path,
    prediction_receipt_path: str | Path,
    stress_config_path: str | Path,
    dev_targets_path: str | Path,
    output_dir: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Score frozen stress predictions; targets enter only after all target-free checks."""

    prediction_source = Path(predictions_path)
    receipt_source = Path(prediction_receipt_path)
    stress_config = Path(stress_config_path)
    targets_source = Path(dev_targets_path)
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
        raise FileExistsError(f"refusing to overwrite stress score artifacts: {collisions}")

    # Target-free validation boundary: do not stat or open dev_targets_path above
    # the explicit marker below.
    prediction_sha256 = sha256_file(prediction_source)
    receipt_sha256 = sha256_file(receipt_source)
    artifact = _load_stress_artifact(prediction_source)
    receipt = _load_stress_receipt(receipt_source)
    _validate_stress_receipt_binding(
        artifact,
        receipt,
        prediction_path=prediction_source,
        prediction_sha256=prediction_sha256,
    )
    protocol = load_stress_protocol(stress_config)
    if protocol.config_sha256 != artifact.stress_config_sha256:
        raise ValueError("stress config differs from frozen prediction")
    expected_conditions = tuple(condition.condition_id for condition in protocol.conditions)
    observed_conditions = tuple(condition.condition_id for condition in artifact.conditions)
    if expected_conditions != observed_conditions:
        raise ValueError("stress conditions differ from frozen config expansion")
    for configured, observed in zip(
        protocol.conditions,
        artifact.conditions,
        strict=True,
    ):
        if (
            configured.kind != observed.perturbation_kind
            or configured.parameters != observed.configured_parameters
        ):
            raise ValueError(
                f"stress condition metadata differs from config: {configured.condition_id}"
            )
    scoring_source_git_sha = clean_git_revision(repository)
    if scoring_source_git_sha != artifact.prediction_source_git_sha:
        raise ValueError("stress scorer revision differs from prediction revision")
    scoring_code_files = _stress_code_file_hashes()
    if scoring_code_files != artifact.stress_code_files_sha256:
        raise ValueError("stress scorer source differs from prediction source")

    # First label-bearing operation.
    targets_sha256 = sha256_file(targets_source)
    targets = load_dev_target_manifest(targets_source)
    target_ids = tuple(record.video_id for record in targets.records)
    if target_ids != tuple(row.video_id for row in artifact.conditions[0].records):
        raise ValueError("dev targets do not match stress prediction order")
    target_values = [record.count for record in targets.records]
    target_map = {record.video_id: record.count for record in targets.records}
    action_map = {record.video_id: record.action for record in targets.records}

    reports: list[dict[str, Any]] = []
    clean_values = [row.count for row in artifact.conditions[0].records]
    for condition in artifact.conditions:
        records = tuple(
            PredictionRecord(video_id=row.video_id, result=row.to_count_result())
            for row in condition.records
        )
        result = evaluate_predictions(
            records,
            target_map,
            actions=action_map,
            bootstrap_samples=10_000,
            bootstrap_seed=2026,
            confidence_level=0.95,
        )
        values = [row.count for row in condition.records]
        comparisons = None
        if condition.condition_id != "clean":
            metrics: tuple[Literal["nmae", "obo"], ...] = ("nmae", "obo")
            comparisons = {
                metric: _paired_payload(
                    values,
                    clean_values,
                    target_values,
                    metric=metric,
                )
                for metric in metrics
            }
        reports.append(
            {
                "condition_id": condition.condition_id,
                "perturbation_kind": condition.perturbation_kind,
                "configured_parameters": condition.configured_parameters,
                "report": result.report.to_dict(),
                "paired_vs_clean": comparisons,
                "predictions": [
                    {
                        "video_id": row.video_id,
                        "count": row.count,
                        "period_frames": row.period_frames,
                        "expert_counts": list(row.expert_counts),
                        "confidence": row.confidence,
                        "perturbation_algorithm": row.perturbation_algorithm,
                        "perturbation_parameters": row.perturbation_parameters,
                        "transformed_pose_sha256": row.transformed_pose_sha256,
                    }
                    for row in condition.records
                ],
            }
        )

    if sha256_file(prediction_source) != prediction_sha256:
        raise RuntimeError("stress predictions changed during scoring")
    if sha256_file(receipt_source) != receipt_sha256:
        raise RuntimeError("stress prediction receipt changed during scoring")
    if sha256_file(stress_config) != protocol.config_sha256:
        raise RuntimeError("stress config changed during scoring")
    if sha256_file(targets_source) != targets_sha256:
        raise RuntimeError("dev targets changed during scoring")
    if _stress_code_file_hashes() != scoring_code_files:
        raise RuntimeError("stress scoring source changed during scoring")
    if clean_git_revision(repository) != scoring_source_git_sha:
        raise RuntimeError("stress scoring Git revision changed during scoring")

    evaluation_payload = {
        "schema_version": 1,
        "artifact_type": "pams_checkpoint_dev_stress_evaluation",
        "classification": _CLASSIFICATION,
        "table_eligible": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "condition_total": artifact.condition_total,
        "record_total_per_condition": 84,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 2026,
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "stress_config_sha256": protocol.config_sha256,
        "dev_targets_sha256": targets_sha256,
        "source_checkpoint_sha256": artifact.source_checkpoint_sha256,
        "prediction_source_git_sha": artifact.prediction_source_git_sha,
        "prediction_code_sha256": artifact.stress_code_sha256,
        "scoring_source_git_sha": scoring_source_git_sha,
        "scoring_code_sha256": sha256_json(scoring_code_files),
        "conditions": reports,
    }
    evaluation_encoded = _encoded_json(evaluation_payload)
    evaluation_sha256 = hashlib.sha256(evaluation_encoded).hexdigest()
    evaluation_receipt = {
        "schema_version": 1,
        "artifact_type": "pams_checkpoint_dev_stress_evaluation_receipt",
        "protocol": "ucfrep_526",
        "split": "dev",
        "evaluation_file": _EVALUATION_NAME,
        "evaluation_sha256": evaluation_sha256,
        "evaluation_bytes": len(evaluation_encoded),
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "stress_config_sha256": protocol.config_sha256,
        "dev_targets_sha256": targets_sha256,
        "source_checkpoint_sha256": artifact.source_checkpoint_sha256,
        "prediction_source_git_sha": artifact.prediction_source_git_sha,
        "prediction_code_sha256": artifact.stress_code_sha256,
        "scoring_source_git_sha": scoring_source_git_sha,
        "scoring_code_sha256": sha256_json(scoring_code_files),
    }
    written = _write_json_bundle(
        (
            (evaluation_path, evaluation_payload),
            (evaluation_receipt_path, evaluation_receipt),
        )
    )
    compact = {
        row["condition_id"]: {
            "nmae": row["report"]["nmae"],
            "obo": row["report"]["obo"],
            "mae": row["report"]["mae"],
            "rmse": row["report"]["rmse"],
        }
        for row in reports
    }
    return {
        "classification": _CLASSIFICATION,
        "table_eligible": False,
        "split": "dev",
        "condition_total": artifact.condition_total,
        "metrics": compact,
        "evaluation_path": str(evaluation_path.resolve()),
        "evaluation_sha256": evaluation_sha256,
        "evaluation_receipt_path": str(evaluation_receipt_path.resolve()),
        "evaluation_receipt_sha256": written[evaluation_receipt_path],
    }
