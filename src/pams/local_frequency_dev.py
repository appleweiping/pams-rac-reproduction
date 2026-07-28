"""Strict two-process dev runner for the synthetic-frozen frequency readout.

``run_dev_prediction`` has no target argument and accepts exactly one canonical
84-video dev pose-input sidecar plus its commitment.  It durably freezes
predictions before ``score_dev_predictions`` is allowed to deserialize the
separate dev-target manifest.  Neither operation accepts a test identity,
test target, or general labeled dataset manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from pams.config import StrictModel, load_config
from pams.data import (
    PoseInputManifest,
    load_dev_target_manifest,
    load_pose_cache_set,
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
    validate_pose_input_binding,
)
from pams.evaluation import PredictionRecord, evaluate_predictions
from pams.local_frequency import (
    LocalFrequencyReadout,
    SyntheticFreezeReport,
    load_local_frequency_config,
    normalized_file_sha256,
    replay_synthetic_freeze,
)
from pams.reproducibility import (
    clean_git_revision,
    durable_mkdir,
    fsync_directory,
    sha256_file,
    sha256_json,
)
from pams.types import CountResult

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_METHOD_KEY: Literal["pams-local-frequency-synthetic-v1"] = (
    "pams-local-frequency-synthetic-v1"
)
_CLASSIFICATION: Literal["inferred synthetic-frozen label-free readout"] = (
    "inferred synthetic-frozen label-free readout"
)
_PREDICTION_NAME: Literal["predictions.json"] = "predictions.json"
_PREDICTION_RECEIPT_NAME = "prediction.receipt.json"
_POSE_SNAPSHOT_NAME = "dev-pose-cache-snapshot.json"
_EVALUATION_NAME = "evaluation.json"
_EVALUATION_RECEIPT_NAME = "evaluation.receipt.json"
_SYNTHETIC_REPLAY_NAME = "synthetic-freeze-replay.json"
_SYNTHETIC_REPLAY_RECEIPT_NAME = "synthetic-freeze-replay.receipt.json"


def _canonical_sha256(value: str, name: str) -> str:
    digest = str(value)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return digest


def _canonical_git_sha(value: str) -> str:
    revision = str(value)
    if _GIT_SHA_PATTERN.fullmatch(revision) is None:
        raise ValueError("source_git_sha must be a lowercase 40-character Git SHA")
    return revision


def _reject_duplicate_fields(
    pairs: list[tuple[str, Any]],
    *,
    document_name: str,
) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON field in {document_name}: {key!r}")
        parsed[key] = value
    return parsed


def _load_json_object(path: Path, *, document_name: str) -> dict[str, Any]:
    if path.suffix.lower() != ".json":
        raise ValueError(f"{document_name} path must end in .json")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=lambda pairs: _reject_duplicate_fields(
                pairs,
                document_name=document_name,
            ),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {document_name}: {value}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {document_name} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{document_name} root must be an object")
    return payload


def _write_json_new(path: Path, payload: Any) -> str:
    """Durably create one JSON artifact without an overwrite race."""

    durable_mkdir(path.parent)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(path.parent)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    return hashlib.sha256(encoded).hexdigest()


class DevPredictionRow(StrictModel):
    """One label-free dev prediction plus its committed video identity."""

    video_id: str
    video_sha256: str
    count: int = Field(ge=0)
    period_frames: float = Field(gt=0)
    expert_counts: tuple[int, int, int]
    confidence: float = Field(ge=0, le=1)
    period_stream: tuple[float, ...]

    @model_validator(mode="after")
    def validate_row(self) -> DevPredictionRow:
        if not self.video_id.strip():
            raise ValueError("prediction video_id must be non-empty")
        _canonical_sha256(self.video_sha256, "prediction video_sha256")
        # Reuse the public result contract for integer, finiteness, and stream
        # validation instead of maintaining a second numerical policy.
        self.to_count_result()
        return self

    def to_count_result(self) -> CountResult:
        return CountResult(
            count=self.count,
            period_frames=self.period_frames,
            expert_counts=self.expert_counts,
            confidence=self.confidence,
            period_stream=np.asarray(self.period_stream, dtype=np.float32),
        )


class DevPredictionArtifact(StrictModel):
    """Strict target-free artifact emitted by the prediction process."""

    schema_version: Literal[1] = 1
    artifact_type: Literal["local_frequency_dev_predictions"]
    classification: Literal["inferred synthetic-frozen label-free readout"]
    eligible_for_paper_table: Literal[False] = False
    protocol: Literal["ucfrep_526"]
    split: Literal["dev"]
    method_key: Literal["pams-local-frequency-synthetic-v1"]
    record_total: Literal[84]
    config_file_sha256: str
    config_fingerprint: str
    selection_scope_sha256: str
    pose_config_file_sha256: str
    pose_fingerprint: str
    dev_inputs_sha256: str
    dev_commitment_sha256: str
    dev_identity_sha256: str
    dev_sidecar_fingerprint: str
    pose_cache_set_sha256: str
    pose_cache_snapshot_sha256: str
    source_git_sha: str
    code_files_sha256: dict[str, str]
    code_sha256: str
    records: tuple[DevPredictionRow, ...]

    @model_validator(mode="after")
    def validate_artifact(self) -> DevPredictionArtifact:
        for field in (
            "config_file_sha256",
            "config_fingerprint",
            "selection_scope_sha256",
            "pose_config_file_sha256",
            "pose_fingerprint",
            "dev_inputs_sha256",
            "dev_commitment_sha256",
            "dev_identity_sha256",
            "dev_sidecar_fingerprint",
            "pose_cache_set_sha256",
            "pose_cache_snapshot_sha256",
            "code_sha256",
        ):
            _canonical_sha256(getattr(self, field), field)
        _canonical_git_sha(self.source_git_sha)
        if set(self.code_files_sha256) != {"local_frequency", "local_frequency_dev"}:
            raise ValueError("code_files_sha256 must bind both readout and dev runner")
        for name, digest in self.code_files_sha256.items():
            _canonical_sha256(digest, f"code_files_sha256[{name}]")
        if sha256_json(self.code_files_sha256) != self.code_sha256:
            raise ValueError("code_sha256 does not bind code_files_sha256")
        if len(self.records) != self.record_total:
            raise ValueError("prediction record_total mismatch")
        identifiers = [record.video_id for record in self.records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("prediction video_id values must be unique")
        return self


class DevPredictionReceipt(StrictModel):
    """Independent commitment to the already-written prediction bytes."""

    schema_version: Literal[1] = 1
    artifact_type: Literal["local_frequency_dev_prediction_receipt"]
    protocol: Literal["ucfrep_526"]
    split: Literal["dev"]
    method_key: Literal["pams-local-frequency-synthetic-v1"]
    prediction_file: Literal["predictions.json"]
    prediction_sha256: str
    prediction_bytes: int = Field(ge=1)
    config_fingerprint: str
    dev_identity_sha256: str
    pose_cache_set_sha256: str
    source_git_sha: str
    code_sha256: str

    @model_validator(mode="after")
    def validate_receipt(self) -> DevPredictionReceipt:
        for field in (
            "prediction_sha256",
            "config_fingerprint",
            "dev_identity_sha256",
            "pose_cache_set_sha256",
            "code_sha256",
        ):
            _canonical_sha256(getattr(self, field), field)
        _canonical_git_sha(self.source_git_sha)
        return self


def _code_file_hashes() -> dict[str, str]:
    return {
        "local_frequency": sha256_file(Path(__file__).with_name("local_frequency.py")),
        "local_frequency_dev": sha256_file(Path(__file__)),
    }


def _load_bound_dev_inputs(
    sidecar_path: Path,
    commitment_path: Path,
) -> tuple[PoseInputManifest, str, str]:
    sidecar_sha256 = sha256_file(sidecar_path)
    commitment_sha256 = sha256_file(commitment_path)
    sidecar = load_pose_input_manifest(sidecar_path, validate_exact=True)
    commitment = load_pose_input_commitment(commitment_path)
    validate_pose_input_binding(
        sidecar,
        commitment,
        sidecar_sha256=sidecar_sha256,
    )
    if sidecar.protocol != "ucfrep_526" or sidecar.split != "dev":
        raise ValueError("local-frequency prediction requires the canonical ucfrep_526 dev sidecar")
    if len(sidecar.records) != 84:
        raise ValueError("local-frequency dev prediction requires exactly 84 records")
    if sha256_file(sidecar_path) != sidecar_sha256:
        raise RuntimeError("dev pose-input sidecar changed while it was being loaded")
    if sha256_file(commitment_path) != commitment_sha256:
        raise RuntimeError("dev pose-input commitment changed while it was being loaded")
    return sidecar, sidecar_sha256, commitment_sha256


def _prediction_row(
    prediction: PredictionRecord,
    *,
    video_sha256: str,
) -> DevPredictionRow:
    result = prediction.result
    return DevPredictionRow(
        video_id=prediction.video_id,
        video_sha256=video_sha256,
        count=result.count,
        period_frames=result.period_frames,
        expert_counts=result.expert_counts,
        confidence=result.confidence,
        period_stream=tuple(float(value) for value in result.period_stream),
    )


def _synthetic_replay_observed(report: SyntheticFreezeReport) -> dict[str, Any]:
    selected = report.selected_score
    return {
        "selected_candidate": report.selected_candidate,
        "sample_count": len(report.sample_ids),
        "mean_absolute_error": selected.mean_absolute_error,
        "normalized_mean_absolute_error": selected.normalized_mean_absolute_error,
        "exact_rate": selected.exact_rate,
        "obo": selected.obo,
    }


def run_synthetic_freeze_replay(
    *,
    config_path: str | Path,
    output_dir: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Replay the bounded synthetic selector and issue a success-only receipt.

    There is deliberately no external sequence, prediction, manifest, or
    target argument.  A mismatch raises before either success artifact is
    created.
    """

    frequency_config_path = Path(config_path)
    destination = Path(output_dir)
    repository = Path(repository_root)
    replay_path = destination / _SYNTHETIC_REPLAY_NAME
    receipt_path = destination / _SYNTHETIC_REPLAY_RECEIPT_NAME
    collisions = [str(path) for path in (replay_path, receipt_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite synthetic replay artifacts: {collisions}")

    source_git_sha = clean_git_revision(repository)
    config_file_sha256 = sha256_file(frequency_config_path)
    code_files_sha256 = _code_file_hashes()
    code_sha256 = sha256_json(code_files_sha256)
    config = load_local_frequency_config(
        frequency_config_path,
        repository_root=repository,
    )
    scope_path = repository / config.selection.scope_config
    scope_sha256 = normalized_file_sha256(scope_path)
    report = replay_synthetic_freeze(config, repository_root=repository)
    observed = _synthetic_replay_observed(report)
    expected = config.selection.expected_replay.model_dump(mode="json")
    if observed != expected:
        raise ValueError(
            "synthetic freeze replay does not exactly match expected_replay; "
            f"expected={expected}, observed={observed}"
        )

    if sha256_file(frequency_config_path) != config_file_sha256:
        raise RuntimeError("local-frequency config changed during synthetic replay")
    if normalized_file_sha256(scope_path) != scope_sha256:
        raise RuntimeError("synthetic stress scope changed during replay")
    if _code_file_hashes() != code_files_sha256:
        raise RuntimeError("local-frequency source code changed during synthetic replay")

    payload = {
        "schema_version": 1,
        "artifact_type": "local_frequency_synthetic_freeze_replay",
        "classification": _CLASSIFICATION,
        "eligible_for_paper_table": False,
        "method_key": _METHOD_KEY,
        "status": "verified",
        "expected_replay_match": True,
        "config_file_sha256": config_file_sha256,
        "config_fingerprint": config.fingerprint,
        "selection_scope_path": config.selection.scope_config,
        "selection_scope_sha256": scope_sha256,
        "source_git_sha": source_git_sha,
        "code_files_sha256": code_files_sha256,
        "code_sha256": code_sha256,
        "expected_replay": expected,
        "observed_replay": observed,
        "selection_report": report.to_dict(),
    }
    replay_sha256 = _write_json_new(replay_path, payload)
    receipt = {
        "schema_version": 1,
        "artifact_type": "local_frequency_synthetic_freeze_replay_receipt",
        "method_key": _METHOD_KEY,
        "status": "verified",
        "expected_replay_match": True,
        "replay_file": _SYNTHETIC_REPLAY_NAME,
        "replay_sha256": replay_sha256,
        "replay_bytes": replay_path.stat().st_size,
        "config_file_sha256": config_file_sha256,
        "config_fingerprint": config.fingerprint,
        "selection_scope_sha256": scope_sha256,
        "source_git_sha": source_git_sha,
        "code_sha256": code_sha256,
    }
    receipt_sha256 = _write_json_new(receipt_path, receipt)
    return {
        "classification": _CLASSIFICATION,
        "eligible_for_paper_table": False,
        "status": "verified",
        "expected_replay_match": True,
        "replay_path": str(replay_path.resolve()),
        "replay_sha256": replay_sha256,
        "replay_receipt_path": str(receipt_path.resolve()),
        "replay_receipt_sha256": receipt_sha256,
        "config_file_sha256": config_file_sha256,
        "config_fingerprint": config.fingerprint,
        "selection_scope_sha256": scope_sha256,
        "source_git_sha": source_git_sha,
        "code_sha256": code_sha256,
        "observed_replay": observed,
    }


def run_dev_prediction(
    *,
    dev_inputs_path: str | Path,
    dev_commitment_path: str | Path,
    pose_cache_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
    pose_config_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Freeze 84 dev predictions without opening a label-bearing file."""

    dev_inputs = Path(dev_inputs_path)
    dev_commitment = Path(dev_commitment_path)
    cache_dir = Path(pose_cache_dir)
    destination = Path(output_dir)
    frequency_config_path = Path(config_path)
    preprocessing_config_path = Path(pose_config_path)
    repository = Path(repository_root)

    output_paths = {
        "pose_snapshot": destination / "inputs" / _POSE_SNAPSHOT_NAME,
        "predictions": destination / _PREDICTION_NAME,
        "receipt": destination / _PREDICTION_RECEIPT_NAME,
    }
    collisions = [str(path) for path in output_paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite dev prediction artifacts: {collisions}")

    source_git_sha = clean_git_revision(repository)
    code_files_sha256 = _code_file_hashes()
    code_sha256 = sha256_json(code_files_sha256)
    config_file_sha256 = sha256_file(frequency_config_path)
    pose_config_file_sha256 = sha256_file(preprocessing_config_path)
    config = load_local_frequency_config(
        frequency_config_path,
        repository_root=repository,
    )
    pose_config = load_config(preprocessing_config_path)
    if pose_config.protocol != "ucfrep_526":
        raise ValueError("pose preprocessing config must use protocol ucfrep_526")
    sidecar, dev_inputs_sha256, dev_commitment_sha256 = _load_bound_dev_inputs(
        dev_inputs,
        dev_commitment,
    )

    sequences, pose_snapshot = load_pose_cache_set(
        sidecar.records,
        cache_dir=cache_dir,
        pose_fingerprint=pose_config.pose_fingerprint,
    )
    expected_ids = tuple(record.video_id for record in sidecar.records)
    if tuple(sequence.video_id for sequence in sequences) != expected_ids:
        raise RuntimeError("pose-cache order does not match the committed dev sidecar")
    pose_snapshot_sha256 = _write_json_new(
        output_paths["pose_snapshot"],
        pose_snapshot.to_dict(),
    )

    readout = LocalFrequencyReadout(config)
    predictions = tuple(
        PredictionRecord(video_id=sequence.video_id, result=readout.predict(sequence))
        for sequence in sequences
    )
    if tuple(record.video_id for record in predictions) != expected_ids:
        raise RuntimeError("prediction order does not match the committed dev sidecar")

    # Recheck every source after computation and before freezing predictions.
    if sha256_file(frequency_config_path) != config_file_sha256:
        raise RuntimeError("local-frequency config changed during prediction")
    if sha256_file(preprocessing_config_path) != pose_config_file_sha256:
        raise RuntimeError("pose preprocessing config changed during prediction")
    if sha256_file(dev_inputs) != dev_inputs_sha256:
        raise RuntimeError("dev pose-input sidecar changed during prediction")
    if sha256_file(dev_commitment) != dev_commitment_sha256:
        raise RuntimeError("dev pose-input commitment changed during prediction")
    if _code_file_hashes() != code_files_sha256:
        raise RuntimeError("local-frequency source code changed during prediction")

    video_hashes = {record.video_id: record.video_sha256 for record in sidecar.records}
    rows = tuple(
        _prediction_row(
            prediction,
            video_sha256=video_hashes[prediction.video_id] or "",
        )
        for prediction in predictions
    )
    artifact = DevPredictionArtifact(
        artifact_type="local_frequency_dev_predictions",
        classification=_CLASSIFICATION,
        protocol="ucfrep_526",
        split="dev",
        method_key=_METHOD_KEY,
        record_total=84,
        config_file_sha256=config_file_sha256,
        config_fingerprint=config.fingerprint,
        selection_scope_sha256=config.selection.scope_config_sha256,
        pose_config_file_sha256=pose_config_file_sha256,
        pose_fingerprint=pose_config.pose_fingerprint,
        dev_inputs_sha256=dev_inputs_sha256,
        dev_commitment_sha256=dev_commitment_sha256,
        dev_identity_sha256=pose_input_identity_sha256(sidecar.records),
        dev_sidecar_fingerprint=sidecar.fingerprint,
        pose_cache_set_sha256=pose_snapshot.fingerprint,
        pose_cache_snapshot_sha256=pose_snapshot_sha256,
        source_git_sha=source_git_sha,
        code_files_sha256=code_files_sha256,
        code_sha256=code_sha256,
        records=rows,
    )
    prediction_sha256 = _write_json_new(
        output_paths["predictions"],
        artifact.model_dump(mode="json"),
    )
    receipt = DevPredictionReceipt(
        artifact_type="local_frequency_dev_prediction_receipt",
        protocol="ucfrep_526",
        split="dev",
        method_key=_METHOD_KEY,
        prediction_file=_PREDICTION_NAME,
        prediction_sha256=prediction_sha256,
        prediction_bytes=output_paths["predictions"].stat().st_size,
        config_fingerprint=config.fingerprint,
        dev_identity_sha256=artifact.dev_identity_sha256,
        pose_cache_set_sha256=pose_snapshot.fingerprint,
        source_git_sha=source_git_sha,
        code_sha256=code_sha256,
    )
    receipt_sha256 = _write_json_new(
        output_paths["receipt"],
        receipt.model_dump(mode="json"),
    )
    return {
        "classification": _CLASSIFICATION,
        "eligible_for_paper_table": False,
        "split": "dev",
        "record_total": 84,
        "predictions_path": str(output_paths["predictions"].resolve()),
        "predictions_sha256": prediction_sha256,
        "prediction_receipt_path": str(output_paths["receipt"].resolve()),
        "prediction_receipt_sha256": receipt_sha256,
        "pose_cache_snapshot_path": str(output_paths["pose_snapshot"].resolve()),
        "pose_cache_snapshot_sha256": pose_snapshot_sha256,
        "config_fingerprint": config.fingerprint,
        "dev_identity_sha256": artifact.dev_identity_sha256,
        "pose_cache_set_sha256": pose_snapshot.fingerprint,
        "source_git_sha": source_git_sha,
        "code_sha256": code_sha256,
    }


def _load_prediction_receipt(path: Path) -> DevPredictionReceipt:
    return DevPredictionReceipt.model_validate(
        _load_json_object(path, document_name="local-frequency prediction receipt")
    )


def _load_prediction_artifact(path: Path) -> DevPredictionArtifact:
    return DevPredictionArtifact.model_validate(
        _load_json_object(path, document_name="local-frequency dev predictions")
    )


def _validate_receipt_binding(
    artifact: DevPredictionArtifact,
    receipt: DevPredictionReceipt,
    *,
    prediction_path: Path,
    prediction_sha256: str,
) -> None:
    if prediction_path.name != receipt.prediction_file:
        raise ValueError("prediction filename does not match its frozen receipt")
    if prediction_sha256 != receipt.prediction_sha256:
        raise ValueError("prediction SHA-256 does not match its frozen receipt")
    if prediction_path.stat().st_size != receipt.prediction_bytes:
        raise ValueError("prediction byte count does not match its frozen receipt")
    pairs = {
        "protocol": (artifact.protocol, receipt.protocol),
        "split": (artifact.split, receipt.split),
        "method_key": (artifact.method_key, receipt.method_key),
        "config_fingerprint": (artifact.config_fingerprint, receipt.config_fingerprint),
        "dev_identity_sha256": (artifact.dev_identity_sha256, receipt.dev_identity_sha256),
        "pose_cache_set_sha256": (
            artifact.pose_cache_set_sha256,
            receipt.pose_cache_set_sha256,
        ),
        "source_git_sha": (artifact.source_git_sha, receipt.source_git_sha),
        "code_sha256": (artifact.code_sha256, receipt.code_sha256),
    }
    mismatches = [name for name, values in pairs.items() if values[0] != values[1]]
    if mismatches:
        raise ValueError(f"prediction receipt metadata mismatch: {mismatches}")


def score_dev_predictions(
    *,
    predictions_path: str | Path,
    prediction_receipt_path: str | Path,
    dev_targets_path: str | Path,
    output_dir: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Score an already-frozen prediction artifact at the dev-only boundary."""

    predictions_source = Path(predictions_path)
    receipt_source = Path(prediction_receipt_path)
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
        raise FileExistsError(f"refusing to overwrite dev score artifacts: {collisions}")

    # The target path is intentionally untouched until all prediction bytes
    # and their independent receipt have been loaded and cross-validated.
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
    if sha256_file(predictions_source) != prediction_sha256:
        raise RuntimeError("prediction artifact changed while it was being loaded")
    if sha256_file(receipt_source) != receipt_sha256:
        raise RuntimeError("prediction receipt changed while it was being loaded")

    scoring_source_git_sha = clean_git_revision(repository)
    scoring_code_files_sha256 = _code_file_hashes()
    scoring_code_sha256 = sha256_json(scoring_code_files_sha256)
    targets_sha256 = sha256_file(targets_source)
    targets = load_dev_target_manifest(targets_source)
    target_ids = tuple(record.video_id for record in targets.records)
    prediction_ids = tuple(record.video_id for record in artifact.records)
    if target_ids != prediction_ids:
        raise ValueError("dev targets do not exactly match frozen prediction order/identity")
    if sha256_file(targets_source) != targets_sha256:
        raise RuntimeError("dev-target manifest changed while it was being loaded")

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
        raise RuntimeError("local-frequency scoring code changed during scoring")

    evaluation_payload = {
        "schema_version": 1,
        "artifact_type": "local_frequency_dev_evaluation",
        "classification": _CLASSIFICATION,
        "eligible_for_paper_table": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "method_key": _METHOD_KEY,
        "bootstrap_pairing": "paired_prediction_target_rows",
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "dev_targets_sha256": targets_sha256,
        "config_file_sha256": artifact.config_file_sha256,
        "config_fingerprint": artifact.config_fingerprint,
        "selection_scope_sha256": artifact.selection_scope_sha256,
        "pose_config_file_sha256": artifact.pose_config_file_sha256,
        "pose_fingerprint": artifact.pose_fingerprint,
        "dev_inputs_sha256": artifact.dev_inputs_sha256,
        "dev_commitment_sha256": artifact.dev_commitment_sha256,
        "dev_identity_sha256": artifact.dev_identity_sha256,
        "pose_cache_set_sha256": artifact.pose_cache_set_sha256,
        "pose_cache_snapshot_sha256": artifact.pose_cache_snapshot_sha256,
        "prediction_source_git_sha": artifact.source_git_sha,
        "prediction_code_files_sha256": artifact.code_files_sha256,
        "prediction_code_sha256": artifact.code_sha256,
        "scoring_source_git_sha": scoring_source_git_sha,
        "scoring_code_files_sha256": scoring_code_files_sha256,
        "scoring_code_sha256": scoring_code_sha256,
        "report": result.report.to_dict(),
        "predictions": [
            {
                "video_id": row.video_id,
                "count": row.count,
                "period_frames": row.period_frames,
                "expert_counts": list(row.expert_counts),
                "confidence": row.confidence,
            }
            for row in artifact.records
        ],
    }
    evaluation_sha256 = _write_json_new(evaluation_path, evaluation_payload)
    evaluation_receipt = {
        "schema_version": 1,
        "artifact_type": "local_frequency_dev_evaluation_receipt",
        "protocol": "ucfrep_526",
        "split": "dev",
        "method_key": _METHOD_KEY,
        "evaluation_file": _EVALUATION_NAME,
        "evaluation_sha256": evaluation_sha256,
        "evaluation_bytes": evaluation_path.stat().st_size,
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "dev_targets_sha256": targets_sha256,
        "config_fingerprint": artifact.config_fingerprint,
        "dev_identity_sha256": artifact.dev_identity_sha256,
        "prediction_source_git_sha": artifact.source_git_sha,
        "prediction_code_sha256": artifact.code_sha256,
        "scoring_source_git_sha": scoring_source_git_sha,
        "scoring_code_sha256": scoring_code_sha256,
    }
    evaluation_receipt_sha256 = _write_json_new(
        evaluation_receipt_path,
        evaluation_receipt,
    )
    compact_metrics = result.report.to_dict()
    compact_metrics.pop("per_video")
    return {
        "classification": _CLASSIFICATION,
        "eligible_for_paper_table": False,
        "split": "dev",
        "evaluation_path": str(evaluation_path.resolve()),
        "evaluation_sha256": evaluation_sha256,
        "evaluation_receipt_path": str(evaluation_receipt_path.resolve()),
        "evaluation_receipt_sha256": evaluation_receipt_sha256,
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "dev_targets_sha256": targets_sha256,
        "config_fingerprint": artifact.config_fingerprint,
        "dev_identity_sha256": artifact.dev_identity_sha256,
        "prediction_source_git_sha": artifact.source_git_sha,
        "prediction_code_sha256": artifact.code_sha256,
        "scoring_source_git_sha": scoring_source_git_sha,
        "scoring_code_sha256": scoring_code_sha256,
        "metrics": compact_metrics,
    }
