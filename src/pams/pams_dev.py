"""Two-process, target-isolated dev evaluation for PAMS checkpoints.

Prediction accepts only count/action-free protocol sidecars.  It validates a
terminal checkpoint against the exact 337-video training pose-cache identity,
freezes 84 ordered dev predictions, and writes an independent byte receipt.
Scoring validates that artifact and receipt completely before it first opens
the separate dev-target manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field, model_validator

from pams.config import PAMSConfig, StrictModel, load_config
from pams.data import (
    LabelFreeProtocolInputs,
    PoseCacheSetSnapshot,
    PoseInputManifest,
    UnlabeledVideoRecord,
    load_dev_target_manifest,
    load_pose_cache_set,
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
    validate_pose_input_binding,
)
from pams.evaluation import PredictionRecord, evaluate_predictions, predict_sequences
from pams.reproducibility import (
    clean_git_revision,
    durable_mkdir,
    fsync_directory,
    hardware_fingerprint,
    sha256_file,
    sha256_json,
)
from pams.run_manifest import (
    CompletedRunReceipt,
    RunManifest,
    resolve_artifact_path,
    validate_artifact_receipt,
)
from pams.training import (
    CheckpointProvenance,
    load_model_checkpoint,
    validate_sshead_encoder_binding,
    validate_terminal_checkpoint,
)
from pams.types import CountResult

PAMSDevVariant = Literal["literal", "sshead"]
PAMSInferenceExpertMode = Literal["multi", "medium_only"]
PAMSDirectFFTTimebase = Literal["compact_valid", "dense_resampled"]
PAMSSelectedExpert = Literal["fast", "medium", "slow"]
PAMSAblationStatus = Literal[
    "default multi-expert",
    "inferred single-expert ablation",
]
PAMSTimebaseStatus = Literal[
    "legacy compact-valid inference clock",
    "inferred dense-resampled mask-aware clock",
]

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_PREDICTION_NAME: Literal["predictions.json"] = "predictions.json"
_PREDICTION_RECEIPT_NAME = "prediction.receipt.json"
_POSE_SNAPSHOT_NAME = "dev-pose-cache-snapshot.json"
_EVALUATION_NAME = "evaluation.json"
_EVALUATION_RECEIPT_NAME = "evaluation.receipt.json"
_CLASSIFICATION: Literal["PAMS dev representation diagnostic"] = (
    "PAMS dev representation diagnostic"
)


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


def _encoded_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class _CreatedArtifact:
    device: int
    inode: int
    byte_count: int
    sha256: str


def _stable_artifact_identity(path: Path) -> _CreatedArtifact:
    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"created artifact must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
        closed = os.fstat(handle.fileno())
    after = path.lstat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(closed, field)
        or getattr(closed, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise RuntimeError(f"created artifact changed while hashing: {path}")
    if byte_count != after.st_size:
        raise RuntimeError(f"created artifact byte count changed while hashing: {path}")
    return _CreatedArtifact(
        device=after.st_dev,
        inode=after.st_ino,
        byte_count=byte_count,
        sha256=digest.hexdigest(),
    )


def _remove_artifact_if_exact(path: Path, expected: _CreatedArtifact) -> bool:
    """Remove only bytes and inode created by this transaction."""

    try:
        observed = _stable_artifact_identity(path)
        immediately_before = path.lstat()
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return False
    if observed != expected:
        return False
    if (
        immediately_before.st_dev != expected.device
        or immediately_before.st_ino != expected.inode
        or immediately_before.st_size != expected.byte_count
    ):
        return False
    try:
        path.unlink()
    except OSError:
        return False
    fsync_directory(path.parent)
    return True


def _remove_partial_artifact_if_owned(
    path: Path,
    *,
    device: int,
    inode: int,
    encoded: bytes,
) -> bool:
    """Remove an interrupted exclusive write only if it is our exact prefix."""

    try:
        observed = _stable_artifact_identity(path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return False
    if observed.device != device or observed.inode != inode:
        return False
    if observed.byte_count > len(encoded):
        return False
    expected_prefix = encoded[: observed.byte_count]
    if hashlib.sha256(expected_prefix).hexdigest() != observed.sha256:
        return False
    return _remove_artifact_if_exact(path, observed)


def _write_json_new(path: Path, payload: Any) -> str:
    """Durably create one JSON artifact and clean an owned partial write."""

    durable_mkdir(path.parent)
    encoded = _encoded_json(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    created = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(path.parent)
    except Exception as exc:
        if descriptor >= 0:
            os.close(descriptor)
        if not _remove_partial_artifact_if_owned(
            path,
            device=created.st_dev,
            inode=created.st_ino,
            encoded=encoded,
        ):
            raise RuntimeError(
                f"JSON write failed and its partial artifact could not be removed: {path}"
            ) from exc
        raise
    return hashlib.sha256(encoded).hexdigest()


def _write_json_bundle(entries: Sequence[tuple[Path, Any]]) -> dict[Path, str]:
    """Write an all-or-nothing artifact bundle with exact-file rollback."""

    created: list[tuple[Path, _CreatedArtifact]] = []
    digests: dict[Path, str] = {}
    try:
        for path, payload in entries:
            digest = _write_json_new(path, payload)
            identity = _stable_artifact_identity(path)
            if identity.sha256 != digest:
                raise RuntimeError(f"created JSON hash differs from encoded bytes: {path}")
            created.append((path, identity))
            digests[path] = digest
    except Exception as exc:
        unsafe = [
            str(path)
            for path, identity in reversed(created)
            if not _remove_artifact_if_exact(path, identity)
        ]
        if unsafe:
            raise RuntimeError(
                "artifact-bundle write failed and exact rollback was unsafe for "
                f"{unsafe}"
            ) from exc
        raise
    return digests


def _method_key(variant: PAMSDevVariant) -> str:
    return "pams-literal" if variant == "literal" else "pams-sshead-inferred"


def _inference_config(
    training_config: PAMSConfig,
    expert_mode: PAMSInferenceExpertMode | None,
    direct_fft_timebase: PAMSDirectFFTTimebase | None = None,
) -> PAMSConfig:
    """Derive inference-only readout choices without changing training identity."""

    resolved_mode = (
        training_config.consensus.expert_mode
        if expert_mode is None
        else expert_mode
    )
    if resolved_mode not in {"multi", "medium_only"}:
        raise ValueError("expert_mode must be 'multi' or 'medium_only'")
    consensus = training_config.consensus.model_copy(
        update={"expert_mode": resolved_mode}
    )
    resolved_timebase = (
        training_config.period.direct_fft_timebase
        if direct_fft_timebase is None
        else direct_fft_timebase
    )
    if resolved_timebase not in {"compact_valid", "dense_resampled"}:
        raise ValueError(
            "direct_fft_timebase must be 'compact_valid' or 'dense_resampled'"
        )
    period = training_config.period.model_copy(
        update={"direct_fft_timebase": resolved_timebase}
    )
    return training_config.model_copy(
        update={"consensus": consensus, "period": period}
    )


def _ablation_status(
    expert_mode: PAMSInferenceExpertMode,
) -> Literal["default multi-expert", "inferred single-expert ablation"]:
    return (
        "default multi-expert"
        if expert_mode == "multi"
        else "inferred single-expert ablation"
    )


def _timebase_status(
    direct_fft_timebase: PAMSDirectFFTTimebase,
) -> PAMSTimebaseStatus:
    if direct_fft_timebase == "compact_valid":
        return "legacy compact-valid inference clock"
    return "inferred dense-resampled mask-aware clock"


class PAMSDevPredictionRow(StrictModel):
    """One target-free PAMS prediction bound to source-video identity."""

    video_id: str
    video_sha256: str
    count: int = Field(ge=0)
    period_frames: float = Field(gt=0)
    expert_counts: tuple[int, int, int]
    selection_mode: PAMSInferenceExpertMode
    selected_expert: PAMSSelectedExpert | None
    confidence: float = Field(ge=0, le=1)
    period_stream: tuple[float, ...]

    @model_validator(mode="after")
    def validate_row(self) -> PAMSDevPredictionRow:
        if not self.video_id.strip():
            raise ValueError("prediction video_id must be non-empty")
        _canonical_sha256(self.video_sha256, "prediction video_sha256")
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


class PAMSDevPredictionArtifact(StrictModel):
    """Strict target-free artifact emitted by checkpoint prediction."""

    schema_version: Literal[2, 3] = 3
    artifact_type: Literal["pams_checkpoint_dev_predictions"]
    classification: Literal["PAMS dev representation diagnostic"]
    table2_eligible: Literal[False] = False
    protocol: Literal["ucfrep_526"]
    split: Literal["dev"]
    variant: PAMSDevVariant
    method_key: str
    record_total: Literal[84]
    config_file_sha256: str
    training_config_fingerprint: str
    config_fingerprint: str
    consensus_expert_mode: PAMSInferenceExpertMode
    ablation_status: PAMSAblationStatus
    direct_fft_timebase: PAMSDirectFFTTimebase = "compact_valid"
    timebase_status: PAMSTimebaseStatus = "legacy compact-valid inference clock"
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
    upstream_encoder_checkpoint_sha256: str | None
    upstream_encoder_progress_sha256: str | None
    upstream_encoder_completion_receipt_sha256: str | None
    checkpoint_source_git_sha: str
    checkpoint_container_image_id: str
    checkpoint_container_environment_sha256: str
    prediction_source_git_sha: str
    prediction_container_image_id: str
    prediction_container_environment_sha256: str
    prediction_code_files_sha256: dict[str, str]
    prediction_code_sha256: str
    records: tuple[PAMSDevPredictionRow, ...]

    @model_validator(mode="after")
    def validate_artifact(self) -> PAMSDevPredictionArtifact:
        for field in (
            "config_file_sha256",
            "training_config_fingerprint",
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
        optional_hashes = (
            self.upstream_encoder_checkpoint_sha256,
            self.upstream_encoder_progress_sha256,
            self.upstream_encoder_completion_receipt_sha256,
        )
        for index, digest in enumerate(optional_hashes):
            if digest is not None:
                _canonical_sha256(digest, f"upstream hash {index}")
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
        if self.method_key != _method_key(self.variant):
            raise ValueError("method_key does not match checkpoint variant")
        expected_status = (
            "default multi-expert"
            if self.consensus_expert_mode == "multi"
            else "inferred single-expert ablation"
        )
        if self.ablation_status != expected_status:
            raise ValueError("ablation_status does not match consensus_expert_mode")
        expected_timebase_status = _timebase_status(self.direct_fft_timebase)
        if self.timebase_status != expected_timebase_status:
            raise ValueError("timebase_status does not match direct_fft_timebase")
        if any(
            row.selection_mode != self.consensus_expert_mode
            for row in self.records
        ):
            raise ValueError("prediction row selection_mode mismatch")
        if self.consensus_expert_mode == "medium_only" and any(
            row.selected_expert != "medium" for row in self.records
        ):
            raise ValueError("medium_only predictions must select the medium expert")
        if self.consensus_expert_mode == "multi" and any(
            row.selected_expert is not None for row in self.records
        ):
            raise ValueError(
                "multi predictions cannot infer a selected expert from compact counts"
            )
        if self.variant == "literal" and any(value is not None for value in optional_hashes):
            raise ValueError("literal predictions cannot name an upstream encoder")
        if self.variant == "sshead" and any(value is None for value in optional_hashes):
            raise ValueError("SSHead predictions require all upstream encoder hashes")
        legacy_code_files = {
            "config",
            "consensus",
            "data",
            "evaluation",
            "pams_dev",
            "training",
        }
        expected_code_files = legacy_code_files | {"period"}
        observed_code_files = set(self.prediction_code_files_sha256)
        if self.schema_version == 2:
            if (
                self.direct_fft_timebase != "compact_valid"
                or self.timebase_status != "legacy compact-valid inference clock"
            ):
                raise ValueError("schema v2 only supports the legacy compact timebase")
            if observed_code_files != legacy_code_files:
                raise ValueError("schema v2 source set must match the legacy contract")
        elif observed_code_files != expected_code_files:
            raise ValueError("schema v3 predictions must bind period.py")
        for name, digest in self.prediction_code_files_sha256.items():
            _canonical_sha256(digest, f"prediction_code_files_sha256[{name}]")
        if sha256_json(self.prediction_code_files_sha256) != self.prediction_code_sha256:
            raise ValueError("prediction_code_sha256 does not bind source-file hashes")
        if len(self.records) != self.record_total:
            raise ValueError("prediction record_total mismatch")
        identifiers = [record.video_id for record in self.records]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("prediction video_id values must be unique")
        return self


class PAMSDevPredictionReceipt(StrictModel):
    """Independent commitment to already-written PAMS prediction bytes."""

    schema_version: Literal[2, 3] = 3
    artifact_type: Literal["pams_checkpoint_dev_prediction_receipt"]
    protocol: Literal["ucfrep_526"]
    split: Literal["dev"]
    variant: PAMSDevVariant
    method_key: str
    prediction_file: Literal["predictions.json"]
    prediction_sha256: str
    prediction_bytes: int = Field(ge=1)
    training_config_fingerprint: str
    config_fingerprint: str
    consensus_expert_mode: PAMSInferenceExpertMode
    ablation_status: PAMSAblationStatus
    direct_fft_timebase: PAMSDirectFFTTimebase = "compact_valid"
    timebase_status: PAMSTimebaseStatus = "legacy compact-valid inference clock"
    protocol_identity_sha256: str
    training_identity_sha256: str
    dev_identity_sha256: str
    checkpoint_sha256: str
    checkpoint_progress_sha256: str
    checkpoint_completion_receipt_sha256: str
    upstream_encoder_checkpoint_sha256: str | None
    upstream_encoder_progress_sha256: str | None
    upstream_encoder_completion_receipt_sha256: str | None
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
    def validate_receipt(self) -> PAMSDevPredictionReceipt:
        for field in (
            "prediction_sha256",
            "training_config_fingerprint",
            "config_fingerprint",
            "protocol_identity_sha256",
            "training_identity_sha256",
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
        for index, digest in enumerate(
            (
                self.upstream_encoder_checkpoint_sha256,
                self.upstream_encoder_progress_sha256,
                self.upstream_encoder_completion_receipt_sha256,
            )
        ):
            if digest is not None:
                _canonical_sha256(digest, f"receipt upstream hash {index}")
        _canonical_git_sha(self.prediction_source_git_sha, "prediction_source_git_sha")
        _canonical_image_id(
            self.prediction_container_image_id,
            "prediction_container_image_id",
        )
        if self.method_key != _method_key(self.variant):
            raise ValueError("receipt method_key does not match checkpoint variant")
        expected_status = (
            "default multi-expert"
            if self.consensus_expert_mode == "multi"
            else "inferred single-expert ablation"
        )
        if self.ablation_status != expected_status:
            raise ValueError("receipt ablation_status does not match expert mode")
        expected_timebase_status = _timebase_status(self.direct_fft_timebase)
        if self.timebase_status != expected_timebase_status:
            raise ValueError("receipt timebase_status does not match direct FFT timebase")
        if self.schema_version == 2 and self.direct_fft_timebase != "compact_valid":
            raise ValueError("schema v2 receipt only supports the legacy compact timebase")
        optional_hashes = (
            self.upstream_encoder_checkpoint_sha256,
            self.upstream_encoder_progress_sha256,
            self.upstream_encoder_completion_receipt_sha256,
        )
        if self.variant == "literal" and any(value is not None for value in optional_hashes):
            raise ValueError("literal receipt cannot name an upstream encoder")
        if self.variant == "sshead" and any(value is None for value in optional_hashes):
            raise ValueError("SSHead receipt requires all upstream encoder hashes")
        if not self.command or any(not item for item in self.command):
            raise ValueError("prediction receipt command must contain non-empty strings")
        json.dumps(self.hardware, allow_nan=False)
        return self


def _code_file_hashes() -> dict[str, str]:
    directory = Path(__file__).parent
    return {
        "config": sha256_file(directory / "config.py"),
        "consensus": sha256_file(directory / "consensus.py"),
        "data": sha256_file(directory / "data.py"),
        "evaluation": sha256_file(directory / "evaluation.py"),
        "pams_dev": sha256_file(Path(__file__)),
        "period": sha256_file(directory / "period.py"),
        "training": sha256_file(directory / "training.py"),
    }


def _load_bound_inputs(
    sidecar_path: Path,
    commitment_path: Path,
    *,
    expected_split: Literal["train", "dev", "test"],
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
    if sidecar.protocol != "ucfrep_526" or sidecar.split != expected_split:
        raise ValueError(f"expected canonical ucfrep_526 {expected_split} pose inputs")
    if sha256_file(sidecar_path) != sidecar_sha256:
        raise RuntimeError(f"{expected_split} pose-input sidecar changed while loading")
    if sha256_file(commitment_path) != commitment_sha256:
        raise RuntimeError(f"{expected_split} pose-input commitment changed while loading")
    return sidecar, sidecar_sha256, commitment_sha256


def _load_protocol_inputs(
    *,
    train_inputs_path: Path,
    train_commitment_path: Path,
    dev_inputs_path: Path,
    dev_commitment_path: Path,
    test_identity_inputs_path: Path,
    test_identity_commitment_path: Path,
) -> tuple[LabelFreeProtocolInputs, dict[str, str]]:
    train, train_sha256, train_commitment_sha256 = _load_bound_inputs(
        train_inputs_path,
        train_commitment_path,
        expected_split="train",
    )
    dev, dev_sha256, dev_commitment_sha256 = _load_bound_inputs(
        dev_inputs_path,
        dev_commitment_path,
        expected_split="dev",
    )
    test, test_sha256, test_commitment_sha256 = _load_bound_inputs(
        test_identity_inputs_path,
        test_identity_commitment_path,
        expected_split="test",
    )
    protocol_inputs = LabelFreeProtocolInputs(
        protocol="ucfrep_526",
        train=train,
        dev=dev,
        test=test,
    )
    return protocol_inputs, {
        "train_inputs_sha256": train_sha256,
        "train_commitment_sha256": train_commitment_sha256,
        "dev_inputs_sha256": dev_sha256,
        "dev_commitment_sha256": dev_commitment_sha256,
        "test_identity_inputs_sha256": test_sha256,
        "test_identity_commitment_sha256": test_commitment_sha256,
    }


def _runtime_container_identity(source_git_sha: str) -> tuple[str, str]:
    image_id = os.environ.get("PAMS_CONTAINER_IMAGE_ID", "").strip()
    environment_sha256 = os.environ.get(
        "PAMS_CONTAINER_ENVIRONMENT_SHA256",
        "",
    ).strip()
    source_revision = os.environ.get("PAMS_CONTAINER_SOURCE_REVISION", "").strip()
    if not all((image_id, environment_sha256, source_revision)):
        raise RuntimeError(
            "formal PAMS dev prediction requires container image, environment, "
            "and source-revision identity"
        )
    _canonical_image_id(image_id, "prediction container image")
    _canonical_sha256(environment_sha256, "prediction container environment")
    _canonical_git_sha(source_revision, "prediction container source revision")
    if source_revision != source_git_sha:
        raise RuntimeError("prediction container revision does not match clean Git")
    return image_id, environment_sha256


@dataclass(frozen=True, slots=True)
class _CompletedTrainingBinding:
    receipt_sha256: str
    source_git_sha: str
    container_image_id: str
    container_environment_sha256: str


def _validate_completed_training_receipt(
    receipt_path: Path,
    *,
    expected_stage: Literal["encoder", "sshead"],
    checkpoint_path: Path,
    progress_path: Path,
    config_path: Path,
    config: PAMSConfig,
    inputs: LabelFreeProtocolInputs,
    input_paths: Mapping[str, Path],
    input_hashes: Mapping[str, str],
    training_pose_snapshot: PoseCacheSetSnapshot,
    upstream_encoder_checkpoint_path: Path | None,
    upstream_encoder_progress_path: Path | None,
) -> _CompletedTrainingBinding:
    """Validate the independent training receipt before trusting checkpoint provenance."""

    receipt_sha256 = sha256_file(receipt_path)
    receipt = CompletedRunReceipt.model_validate(
        _load_json_object(receipt_path, document_name="completed training receipt")
    )
    sibling_started_path = receipt_path.with_name(
        f"{receipt.run_id}.started.json"
    )
    if not sibling_started_path.is_file():
        raise FileNotFoundError(
            "completed training receipt is missing its sibling started receipt"
        )
    if sha256_file(sibling_started_path) != receipt.start_manifest_sha256:
        raise ValueError("completed training receipt does not bind sibling started bytes")
    sibling_started = RunManifest.model_validate(
        _load_json_object(
            sibling_started_path,
            document_name="started training receipt",
        )
    )
    if sibling_started != receipt.started:
        raise ValueError("embedded and sibling started training receipts differ")
    if receipt.started.protocol != inputs.protocol:
        raise ValueError("training receipt protocol does not match label-free inputs")
    if receipt.started.seed != config.seed:
        raise ValueError("training receipt seed does not match config")
    if receipt.started.config_sha256 != config.fingerprint:
        raise ValueError("training receipt config fingerprint does not match")
    if (
        receipt.started.dataset_sha256
        != inputs.training_fingerprint(include_dev=False)
    ):
        raise ValueError("training receipt dataset identity is not train337-only")
    expected_epochs = (
        config.training.epochs
        if expected_stage == "encoder"
        else config.sshead.epochs
    )
    if receipt.metrics.get("completed_epochs") != expected_epochs:
        raise ValueError("training receipt is not terminal for the expected stage")

    artifacts = {artifact.role: artifact for artifact in receipt.artifacts}
    output_role = (
        "output_encoder_checkpoint"
        if expected_stage == "encoder"
        else "output_sshead_checkpoint"
    )
    required_roles = {
        output_role,
        "progress_log",
        "input_config",
        "input_dataset_manifest",
        "input_pose_cache_snapshot",
        "input_train_pose_inputs",
        "input_train_pose_input_commitment",
        "input_dev_pose_inputs",
        "input_dev_pose_input_commitment",
        "input_test_identity_pose_inputs",
        "input_test_identity_pose_input_commitment",
    }
    if expected_stage == "sshead":
        required_roles.update(
            {"input_encoder_checkpoint", "input_encoder_progress"}
        )
    missing_roles = sorted(required_roles - set(artifacts))
    if missing_roles:
        raise ValueError(
            f"completed training receipt is missing roles: {missing_roles}"
        )
    for role in sorted(required_roles):
        artifact_path = resolve_artifact_path(receipt_path, artifacts[role])
        validate_artifact_receipt(artifact_path, artifacts[role])

    live_paths = {
        output_role: checkpoint_path,
        "progress_log": progress_path,
        "input_config": config_path,
        "input_dataset_manifest": input_paths["train_inputs_sha256"],
        "input_train_pose_inputs": input_paths["train_inputs_sha256"],
        "input_train_pose_input_commitment": input_paths[
            "train_commitment_sha256"
        ],
        "input_dev_pose_inputs": input_paths["dev_inputs_sha256"],
        "input_dev_pose_input_commitment": input_paths[
            "dev_commitment_sha256"
        ],
        "input_test_identity_pose_inputs": input_paths[
            "test_identity_inputs_sha256"
        ],
        "input_test_identity_pose_input_commitment": input_paths[
            "test_identity_commitment_sha256"
        ],
    }
    if expected_stage == "sshead":
        if upstream_encoder_checkpoint_path is None:
            raise ValueError("SSHead receipt validation requires upstream encoder")
        if upstream_encoder_progress_path is None:
            raise ValueError("SSHead receipt validation requires encoder progress")
        live_paths["input_encoder_checkpoint"] = upstream_encoder_checkpoint_path
        live_paths["input_encoder_progress"] = upstream_encoder_progress_path
    for role, path in live_paths.items():
        validate_artifact_receipt(path, artifacts[role])
    if artifacts["input_config"].sha256 != sha256_file(config_path):
        raise ValueError("training receipt does not bind the supplied config file")
    for role, expected_sha256 in input_hashes.items():
        receipt_role = {
            "train_inputs_sha256": "input_train_pose_inputs",
            "train_commitment_sha256": "input_train_pose_input_commitment",
            "dev_inputs_sha256": "input_dev_pose_inputs",
            "dev_commitment_sha256": "input_dev_pose_input_commitment",
            "test_identity_inputs_sha256": "input_test_identity_pose_inputs",
            "test_identity_commitment_sha256": (
                "input_test_identity_pose_input_commitment"
            ),
        }[role]
        if artifacts[receipt_role].sha256 != expected_sha256:
            raise ValueError(
                f"training receipt does not bind supplied protocol input {role}"
            )
    snapshot_path = resolve_artifact_path(
        receipt_path,
        artifacts["input_pose_cache_snapshot"],
    )
    snapshot_payload = _load_json_object(
        snapshot_path,
        document_name="training pose-cache snapshot",
    )
    if snapshot_payload.get("fingerprint") != training_pose_snapshot.fingerprint:
        raise ValueError("training receipt pose-cache snapshot differs from train337")

    container = receipt.started.hardware.get("container")
    if not isinstance(container, Mapping):
        raise ValueError("training receipt is missing formal container identity")
    image_id = _canonical_image_id(
        str(container.get("image_id", "")),
        "training receipt container image",
    )
    environment_sha256 = _canonical_sha256(
        str(container.get("environment_sha256", "")),
        "training receipt container environment",
    )
    source_revision = _canonical_git_sha(
        receipt.started.git_sha,
        "training receipt source revision",
    )
    if str(container.get("source_revision", "")) != source_revision:
        raise ValueError("training receipt Git and container revisions differ")
    if sha256_file(receipt_path) != receipt_sha256:
        raise RuntimeError("completed training receipt changed during validation")
    return _CompletedTrainingBinding(
        receipt_sha256=receipt_sha256,
        source_git_sha=source_revision,
        container_image_id=image_id,
        container_environment_sha256=environment_sha256,
    )


def _expected_provenance(
    training_binding: _CompletedTrainingBinding,
    *,
    config: PAMSConfig,
    inputs: LabelFreeProtocolInputs,
    training_pose_snapshot: PoseCacheSetSnapshot,
    upstream_encoder_checkpoint_sha256: str | None,
) -> CheckpointProvenance:
    return CheckpointProvenance(
        protocol=inputs.protocol,
        dataset_fingerprint=inputs.training_fingerprint(include_dev=False),
        training_video_ids=tuple(
            record.video_id for record in inputs.training_records(include_dev=False)
        ),
        pose_fingerprint=config.pose_fingerprint,
        pose_cache_set_sha256=training_pose_snapshot.fingerprint,
        source_git_sha=training_binding.source_git_sha,
        container_image_id=training_binding.container_image_id,
        container_environment_sha256=(
            training_binding.container_environment_sha256
        ),
        upstream_encoder_checkpoint_sha256=upstream_encoder_checkpoint_sha256,
    )


def _validated_checkpoint_model(
    *,
    checkpoint_path: Path,
    checkpoint_progress_path: Path,
    variant: PAMSDevVariant,
    checkpoint_training_binding: _CompletedTrainingBinding,
    upstream_encoder_checkpoint_path: Path | None,
    upstream_encoder_progress_path: Path | None,
    upstream_encoder_training_binding: _CompletedTrainingBinding | None,
    config: PAMSConfig,
    inputs: LabelFreeProtocolInputs,
    training_pose_snapshot: PoseCacheSetSnapshot,
    device: str | None,
) -> tuple[Any, CheckpointProvenance, dict[str, str | None]]:
    checkpoint_sha256 = sha256_file(checkpoint_path)
    checkpoint_progress_sha256 = sha256_file(checkpoint_progress_path)
    expected_stage: Literal["encoder", "sshead"] = (
        "encoder" if variant == "literal" else "sshead"
    )
    upstream_checkpoint_sha256: str | None = None
    upstream_progress_sha256: str | None = None
    if variant == "literal":
        if upstream_encoder_checkpoint_path is not None:
            raise ValueError("literal prediction forbids an upstream encoder checkpoint")
        if upstream_encoder_progress_path is not None:
            raise ValueError("literal prediction forbids upstream encoder progress")
        if upstream_encoder_training_binding is not None:
            raise ValueError("literal prediction forbids an upstream encoder receipt")
    else:
        if upstream_encoder_checkpoint_path is None:
            raise ValueError("SSHead prediction requires an upstream encoder checkpoint")
        if upstream_encoder_progress_path is None:
            raise ValueError("SSHead prediction requires upstream encoder progress")
        if upstream_encoder_training_binding is None:
            raise ValueError("SSHead prediction requires the encoder completed receipt")
        upstream_checkpoint_sha256 = sha256_file(upstream_encoder_checkpoint_path)
        upstream_progress_sha256 = sha256_file(upstream_encoder_progress_path)
        encoder_expected = _expected_provenance(
            upstream_encoder_training_binding,
            config=config,
            inputs=inputs,
            training_pose_snapshot=training_pose_snapshot,
            upstream_encoder_checkpoint_sha256=None,
        )
        validate_terminal_checkpoint(
            upstream_encoder_checkpoint_path,
            config,
            expected_stage="encoder",
            expected_provenance=encoder_expected,
            progress_path=upstream_encoder_progress_path,
        )
        if (
            checkpoint_training_binding.source_git_sha
            != upstream_encoder_training_binding.source_git_sha
            or checkpoint_training_binding.container_image_id
            != upstream_encoder_training_binding.container_image_id
            or checkpoint_training_binding.container_environment_sha256
            != upstream_encoder_training_binding.container_environment_sha256
        ):
            raise ValueError("SSHead and upstream encoder training environments differ")
    expected = _expected_provenance(
        checkpoint_training_binding,
        config=config,
        inputs=inputs,
        training_pose_snapshot=training_pose_snapshot,
        upstream_encoder_checkpoint_sha256=upstream_checkpoint_sha256,
    )
    validate_terminal_checkpoint(
        checkpoint_path,
        config,
        expected_stage=expected_stage,
        expected_provenance=expected,
        progress_path=checkpoint_progress_path,
    )
    if variant == "sshead":
        assert upstream_encoder_checkpoint_path is not None
        validate_sshead_encoder_binding(
            checkpoint_path,
            upstream_encoder_checkpoint_path,
        )
    model = load_model_checkpoint(
        checkpoint_path,
        config,
        device=device,
        expected_stage=expected_stage,
        expected_provenance=expected,
    )
    if sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("checkpoint changed during terminal validation")
    if sha256_file(checkpoint_progress_path) != checkpoint_progress_sha256:
        raise RuntimeError("checkpoint progress changed during terminal validation")
    if (
        upstream_encoder_checkpoint_path is not None
        and sha256_file(upstream_encoder_checkpoint_path) != upstream_checkpoint_sha256
    ):
        raise RuntimeError("upstream encoder checkpoint changed during validation")
    if (
        upstream_encoder_progress_path is not None
        and sha256_file(upstream_encoder_progress_path) != upstream_progress_sha256
    ):
        raise RuntimeError("upstream encoder progress changed during validation")
    return model, expected, {
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_progress_sha256": checkpoint_progress_sha256,
        "checkpoint_completion_receipt_sha256": (
            checkpoint_training_binding.receipt_sha256
        ),
        "upstream_encoder_checkpoint_sha256": upstream_checkpoint_sha256,
        "upstream_encoder_progress_sha256": upstream_progress_sha256,
        "upstream_encoder_completion_receipt_sha256": (
            None
            if upstream_encoder_training_binding is None
            else upstream_encoder_training_binding.receipt_sha256
        ),
    }


def _prediction_row(
    prediction: PredictionRecord,
    *,
    video_sha256: str,
    selection_mode: PAMSInferenceExpertMode,
) -> PAMSDevPredictionRow:
    result = prediction.result
    selected_expert: PAMSSelectedExpert | None
    if selection_mode == "medium_only":
        selected_expert = "medium"
        if result.count != result.expert_counts[1]:
            raise ValueError("medium_only prediction did not return the medium expert count")
    else:
        # ``CountResult`` intentionally does not expose the full
        # ``ConsensusResult``.  Reconstructing an expert from matching counts
        # would be ambiguous when multiple experts agree, so v2 records no
        # per-row expert for the multi-expert path.
        selected_expert = None
    return PAMSDevPredictionRow(
        video_id=prediction.video_id,
        video_sha256=video_sha256,
        count=result.count,
        period_frames=result.period_frames,
        expert_counts=result.expert_counts,
        selection_mode=selection_mode,
        selected_expert=selected_expert,
        confidence=result.confidence,
        period_stream=tuple(float(value) for value in result.period_stream),
    )


def run_pams_dev_prediction(
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
    variant: PAMSDevVariant,
    expert_mode: PAMSInferenceExpertMode | None = None,
    direct_fft_timebase: PAMSDirectFFTTimebase | None = None,
    upstream_encoder_checkpoint_path: str | Path | None = None,
    upstream_encoder_progress_path: str | Path | None = None,
    upstream_encoder_completion_receipt_path: str | Path | None = None,
    device: str | None = None,
    repository_root: str | Path,
    command: Sequence[str],
) -> dict[str, Any]:
    """Freeze 84 PAMS dev predictions without accepting any target input."""

    checkpoint = Path(checkpoint_path)
    checkpoint_progress = Path(checkpoint_progress_path)
    checkpoint_completion_receipt = Path(checkpoint_completion_receipt_path)
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
    upstream_checkpoint = (
        None
        if upstream_encoder_checkpoint_path is None
        else Path(upstream_encoder_checkpoint_path)
    )
    upstream_progress = (
        None
        if upstream_encoder_progress_path is None
        else Path(upstream_encoder_progress_path)
    )
    upstream_completion_receipt = (
        None
        if upstream_encoder_completion_receipt_path is None
        else Path(upstream_encoder_completion_receipt_path)
    )
    normalized_variant = str(variant).strip().lower()
    if normalized_variant not in {"literal", "sshead"}:
        raise ValueError("variant must be 'literal' or 'sshead'")
    typed_variant: PAMSDevVariant = normalized_variant  # type: ignore[assignment]

    output_paths = {
        "pose_snapshot": destination / "inputs" / _POSE_SNAPSHOT_NAME,
        "predictions": destination / _PREDICTION_NAME,
        "receipt": destination / _PREDICTION_RECEIPT_NAME,
    }
    collisions = [str(path) for path in output_paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite PAMS dev artifacts: {collisions}")

    prediction_source_git_sha = clean_git_revision(repository)
    prediction_image_id, prediction_environment_sha256 = _runtime_container_identity(
        prediction_source_git_sha
    )
    code_files_sha256 = _code_file_hashes()
    code_sha256 = sha256_json(code_files_sha256)
    config_file_sha256 = sha256_file(config_file)
    training_config = load_config(config_file)
    inference_config = _inference_config(
        training_config,
        expert_mode,
        direct_fft_timebase,
    )
    if training_config.protocol != "ucfrep_526":
        raise ValueError("PAMS dev prediction requires protocol ucfrep_526")
    inputs, input_hashes = _load_protocol_inputs(
        train_inputs_path=train_inputs,
        train_commitment_path=train_commitment,
        dev_inputs_path=dev_inputs,
        dev_commitment_path=dev_commitment,
        test_identity_inputs_path=test_inputs,
        test_identity_commitment_path=test_commitment,
    )
    training_records = inputs.training_records(include_dev=False)
    _, training_pose_snapshot = load_pose_cache_set(
        training_records,
        cache_dir=cache_dir,
        pose_fingerprint=training_config.pose_fingerprint,
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
    expected_stage: Literal["encoder", "sshead"] = (
        "encoder" if typed_variant == "literal" else "sshead"
    )
    checkpoint_training_binding = _validate_completed_training_receipt(
        checkpoint_completion_receipt,
        expected_stage=expected_stage,
        checkpoint_path=checkpoint,
        progress_path=checkpoint_progress,
        config_path=config_file,
        config=training_config,
        inputs=inputs,
        input_paths=input_paths,
        input_hashes=input_hashes,
        training_pose_snapshot=training_pose_snapshot,
        upstream_encoder_checkpoint_path=upstream_checkpoint,
        upstream_encoder_progress_path=upstream_progress,
    )
    upstream_encoder_training_binding: _CompletedTrainingBinding | None = None
    if typed_variant == "sshead":
        if upstream_checkpoint is None:
            raise ValueError("SSHead prediction requires an upstream encoder checkpoint")
        if upstream_progress is None:
            raise ValueError("SSHead prediction requires upstream encoder progress")
        if upstream_completion_receipt is None:
            raise ValueError(
                "SSHead prediction requires the upstream encoder completed receipt"
            )
        upstream_encoder_training_binding = _validate_completed_training_receipt(
            upstream_completion_receipt,
            expected_stage="encoder",
            checkpoint_path=upstream_checkpoint,
            progress_path=upstream_progress,
            config_path=config_file,
            config=training_config,
            inputs=inputs,
            input_paths=input_paths,
            input_hashes=input_hashes,
            training_pose_snapshot=training_pose_snapshot,
            upstream_encoder_checkpoint_path=None,
            upstream_encoder_progress_path=None,
        )
    elif upstream_completion_receipt is not None:
        raise ValueError("literal prediction forbids an upstream encoder receipt")
    model, checkpoint_provenance, checkpoint_hashes = _validated_checkpoint_model(
        checkpoint_path=checkpoint,
        checkpoint_progress_path=checkpoint_progress,
        variant=typed_variant,
        checkpoint_training_binding=checkpoint_training_binding,
        upstream_encoder_checkpoint_path=upstream_checkpoint,
        upstream_encoder_progress_path=upstream_progress,
        upstream_encoder_training_binding=upstream_encoder_training_binding,
        config=training_config,
        inputs=inputs,
        training_pose_snapshot=training_pose_snapshot,
        device=device,
    )
    dev_records = inputs.records_for("dev")
    sequences, dev_pose_snapshot = load_pose_cache_set(
        dev_records,
        cache_dir=cache_dir,
        pose_fingerprint=training_config.pose_fingerprint,
    )
    expected_ids = tuple(record.video_id for record in dev_records)
    if tuple(sequence.video_id for sequence in sequences) != expected_ids:
        raise RuntimeError("dev pose-cache order does not match committed inputs")
    predictions = predict_sequences(
        model,
        sequences,
        inference_config,
        device=device,
    )
    if tuple(record.video_id for record in predictions) != expected_ids:
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
        "checkpoint_progress_sha256": checkpoint_progress,
        "checkpoint_completion_receipt_sha256": (
            checkpoint_completion_receipt
        ),
    }
    for role, path in source_paths.items():
        expected_sha256 = (
            config_file_sha256
            if role == "config_file_sha256"
            else input_hashes.get(role, checkpoint_hashes.get(role))
        )
        if expected_sha256 is None or sha256_file(path) != expected_sha256:
            raise RuntimeError(f"{role} changed during PAMS dev prediction")
    if upstream_checkpoint is not None and (
        sha256_file(upstream_checkpoint)
        != checkpoint_hashes["upstream_encoder_checkpoint_sha256"]
    ):
        raise RuntimeError("upstream encoder checkpoint changed during prediction")
    if upstream_progress is not None and (
        sha256_file(upstream_progress)
        != checkpoint_hashes["upstream_encoder_progress_sha256"]
    ):
        raise RuntimeError("upstream encoder progress changed during prediction")
    if upstream_completion_receipt is not None and (
        sha256_file(upstream_completion_receipt)
        != checkpoint_hashes["upstream_encoder_completion_receipt_sha256"]
    ):
        raise RuntimeError("upstream encoder completed receipt changed during prediction")
    if _code_file_hashes() != code_files_sha256:
        raise RuntimeError("PAMS dev prediction source code changed during prediction")
    if clean_git_revision(repository) != prediction_source_git_sha:
        raise RuntimeError("PAMS dev prediction Git revision changed during prediction")

    pose_snapshot_payload = dev_pose_snapshot.to_dict()
    pose_snapshot_sha256 = hashlib.sha256(
        _encoded_json(pose_snapshot_payload)
    ).hexdigest()
    video_hashes = {
        record.video_id: record.video_sha256 or "" for record in dev_records
    }
    rows = tuple(
        _prediction_row(
            prediction,
            video_sha256=video_hashes[prediction.video_id],
            selection_mode=inference_config.consensus.expert_mode,
        )
        for prediction in predictions
    )
    artifact = PAMSDevPredictionArtifact(
        artifact_type="pams_checkpoint_dev_predictions",
        classification=_CLASSIFICATION,
        protocol="ucfrep_526",
        split="dev",
        variant=typed_variant,
        method_key=_method_key(typed_variant),
        record_total=84,
        config_file_sha256=config_file_sha256,
        training_config_fingerprint=training_config.fingerprint,
        config_fingerprint=inference_config.fingerprint,
        consensus_expert_mode=inference_config.consensus.expert_mode,
        ablation_status=_ablation_status(inference_config.consensus.expert_mode),
        direct_fft_timebase=inference_config.period.direct_fft_timebase,
        timebase_status=_timebase_status(
            inference_config.period.direct_fft_timebase
        ),
        pose_fingerprint=training_config.pose_fingerprint,
        protocol_identity_sha256=inputs.fingerprint,
        training_identity_sha256=inputs.training_fingerprint(include_dev=False),
        train_inputs_sha256=input_hashes["train_inputs_sha256"],
        train_commitment_sha256=input_hashes["train_commitment_sha256"],
        dev_inputs_sha256=input_hashes["dev_inputs_sha256"],
        dev_commitment_sha256=input_hashes["dev_commitment_sha256"],
        test_identity_inputs_sha256=input_hashes[
            "test_identity_inputs_sha256"
        ],
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
        upstream_encoder_checkpoint_sha256=(
            checkpoint_hashes["upstream_encoder_checkpoint_sha256"]
        ),
        upstream_encoder_progress_sha256=(
            checkpoint_hashes["upstream_encoder_progress_sha256"]
        ),
        upstream_encoder_completion_receipt_sha256=(
            checkpoint_hashes["upstream_encoder_completion_receipt_sha256"]
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
        records=rows,
    )
    prediction_payload = artifact.model_dump(mode="json")
    prediction_encoded = _encoded_json(prediction_payload)
    prediction_sha256 = hashlib.sha256(prediction_encoded).hexdigest()
    receipt = PAMSDevPredictionReceipt(
        artifact_type="pams_checkpoint_dev_prediction_receipt",
        protocol="ucfrep_526",
        split="dev",
        variant=typed_variant,
        method_key=_method_key(typed_variant),
        prediction_file=_PREDICTION_NAME,
        prediction_sha256=prediction_sha256,
        prediction_bytes=len(prediction_encoded),
        training_config_fingerprint=training_config.fingerprint,
        config_fingerprint=inference_config.fingerprint,
        consensus_expert_mode=inference_config.consensus.expert_mode,
        ablation_status=_ablation_status(inference_config.consensus.expert_mode),
        direct_fft_timebase=inference_config.period.direct_fft_timebase,
        timebase_status=_timebase_status(
            inference_config.period.direct_fft_timebase
        ),
        protocol_identity_sha256=inputs.fingerprint,
        training_identity_sha256=inputs.training_fingerprint(include_dev=False),
        dev_identity_sha256=artifact.dev_identity_sha256,
        checkpoint_sha256=str(checkpoint_hashes["checkpoint_sha256"]),
        checkpoint_progress_sha256=str(
            checkpoint_hashes["checkpoint_progress_sha256"]
        ),
        checkpoint_completion_receipt_sha256=str(
            checkpoint_hashes["checkpoint_completion_receipt_sha256"]
        ),
        upstream_encoder_checkpoint_sha256=(
            checkpoint_hashes["upstream_encoder_checkpoint_sha256"]
        ),
        upstream_encoder_progress_sha256=(
            checkpoint_hashes["upstream_encoder_progress_sha256"]
        ),
        upstream_encoder_completion_receipt_sha256=(
            checkpoint_hashes["upstream_encoder_completion_receipt_sha256"]
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
    written = _write_json_bundle(
        (
            (output_paths["pose_snapshot"], pose_snapshot_payload),
            (output_paths["predictions"], prediction_payload),
            (output_paths["receipt"], receipt.model_dump(mode="json")),
        )
    )
    receipt_sha256 = written[output_paths["receipt"]]
    return {
        "classification": _CLASSIFICATION,
        "table2_eligible": False,
        "split": "dev",
        "variant": typed_variant,
        "method_key": _method_key(typed_variant),
        "consensus_expert_mode": inference_config.consensus.expert_mode,
        "ablation_status": _ablation_status(inference_config.consensus.expert_mode),
        "direct_fft_timebase": inference_config.period.direct_fft_timebase,
        "timebase_status": _timebase_status(
            inference_config.period.direct_fft_timebase
        ),
        "training_config_fingerprint": training_config.fingerprint,
        "config_fingerprint": inference_config.fingerprint,
        "record_total": 84,
        "predictions_path": str(output_paths["predictions"].resolve()),
        "predictions_sha256": prediction_sha256,
        "prediction_receipt_path": str(output_paths["receipt"].resolve()),
        "prediction_receipt_sha256": receipt_sha256,
        "pose_cache_snapshot_path": str(output_paths["pose_snapshot"].resolve()),
        "pose_cache_snapshot_sha256": pose_snapshot_sha256,
        "checkpoint_sha256": checkpoint_hashes["checkpoint_sha256"],
        "checkpoint_progress_sha256": checkpoint_hashes[
            "checkpoint_progress_sha256"
        ],
        "checkpoint_completion_receipt_sha256": checkpoint_hashes[
            "checkpoint_completion_receipt_sha256"
        ],
        "upstream_encoder_checkpoint_sha256": checkpoint_hashes[
            "upstream_encoder_checkpoint_sha256"
        ],
        "upstream_encoder_progress_sha256": checkpoint_hashes[
            "upstream_encoder_progress_sha256"
        ],
        "upstream_encoder_completion_receipt_sha256": checkpoint_hashes[
            "upstream_encoder_completion_receipt_sha256"
        ],
        "training_pose_cache_set_sha256": training_pose_snapshot.fingerprint,
        "dev_pose_cache_set_sha256": dev_pose_snapshot.fingerprint,
        "prediction_source_git_sha": prediction_source_git_sha,
        "prediction_code_sha256": code_sha256,
    }


def _load_prediction_artifact(path: Path) -> PAMSDevPredictionArtifact:
    return PAMSDevPredictionArtifact.model_validate(
        _load_json_object(path, document_name="PAMS dev predictions")
    )


def _load_prediction_receipt(path: Path) -> PAMSDevPredictionReceipt:
    return PAMSDevPredictionReceipt.model_validate(
        _load_json_object(path, document_name="PAMS dev prediction receipt")
    )


def _validate_receipt_binding(
    artifact: PAMSDevPredictionArtifact,
    receipt: PAMSDevPredictionReceipt,
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
        "schema_version": (artifact.schema_version, receipt.schema_version),
        "protocol": (artifact.protocol, receipt.protocol),
        "split": (artifact.split, receipt.split),
        "variant": (artifact.variant, receipt.variant),
        "method_key": (artifact.method_key, receipt.method_key),
        "training_config_fingerprint": (
            artifact.training_config_fingerprint,
            receipt.training_config_fingerprint,
        ),
        "config_fingerprint": (
            artifact.config_fingerprint,
            receipt.config_fingerprint,
        ),
        "consensus_expert_mode": (
            artifact.consensus_expert_mode,
            receipt.consensus_expert_mode,
        ),
        "ablation_status": (
            artifact.ablation_status,
            receipt.ablation_status,
        ),
        "direct_fft_timebase": (
            artifact.direct_fft_timebase,
            receipt.direct_fft_timebase,
        ),
        "timebase_status": (
            artifact.timebase_status,
            receipt.timebase_status,
        ),
        "protocol_identity_sha256": (
            artifact.protocol_identity_sha256,
            receipt.protocol_identity_sha256,
        ),
        "training_identity_sha256": (
            artifact.training_identity_sha256,
            receipt.training_identity_sha256,
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
        "upstream_encoder_checkpoint_sha256": (
            artifact.upstream_encoder_checkpoint_sha256,
            receipt.upstream_encoder_checkpoint_sha256,
        ),
        "upstream_encoder_progress_sha256": (
            artifact.upstream_encoder_progress_sha256,
            receipt.upstream_encoder_progress_sha256,
        ),
        "upstream_encoder_completion_receipt_sha256": (
            artifact.upstream_encoder_completion_receipt_sha256,
            receipt.upstream_encoder_completion_receipt_sha256,
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


def _validate_canonical_dev_identity(artifact: PAMSDevPredictionArtifact) -> None:
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


def score_pams_dev_predictions(
    *,
    predictions_path: str | Path,
    prediction_receipt_path: str | Path,
    dev_targets_path: str | Path,
    output_dir: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Score frozen PAMS predictions after the target-isolation firewall."""

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
        raise FileExistsError(f"refusing to overwrite PAMS dev score artifacts: {collisions}")

    # Do not stat, hash, or deserialize targets above this boundary.
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

    # This is the first operation permitted to open the label-bearing file.
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
        raise RuntimeError("PAMS dev scoring source changed during scoring")
    if clean_git_revision(repository) != scoring_source_git_sha:
        raise RuntimeError("PAMS dev scoring Git revision changed during scoring")

    evaluation_payload = {
        "schema_version": 1,
        "artifact_type": "pams_checkpoint_dev_evaluation",
        "classification": artifact.classification,
        "table2_eligible": False,
        "protocol": artifact.protocol,
        "split": "dev",
        "variant": artifact.variant,
        "method_key": artifact.method_key,
        "consensus_expert_mode": artifact.consensus_expert_mode,
        "ablation_status": artifact.ablation_status,
        "direct_fft_timebase": artifact.direct_fft_timebase,
        "timebase_status": artifact.timebase_status,
        "bootstrap_pairing": "paired_prediction_target_rows",
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "dev_targets_sha256": targets_sha256,
        "config_file_sha256": artifact.config_file_sha256,
        "training_config_fingerprint": artifact.training_config_fingerprint,
        "config_fingerprint": artifact.config_fingerprint,
        "protocol_identity_sha256": artifact.protocol_identity_sha256,
        "training_identity_sha256": artifact.training_identity_sha256,
        "dev_identity_sha256": artifact.dev_identity_sha256,
        "checkpoint_sha256": artifact.checkpoint_sha256,
        "checkpoint_progress_sha256": artifact.checkpoint_progress_sha256,
        "checkpoint_completion_receipt_sha256": (
            artifact.checkpoint_completion_receipt_sha256
        ),
        "upstream_encoder_checkpoint_sha256": (
            artifact.upstream_encoder_checkpoint_sha256
        ),
        "upstream_encoder_progress_sha256": (
            artifact.upstream_encoder_progress_sha256
        ),
        "upstream_encoder_completion_receipt_sha256": (
            artifact.upstream_encoder_completion_receipt_sha256
        ),
        "training_pose_cache_set_sha256": (
            artifact.training_pose_cache_set_sha256
        ),
        "dev_pose_cache_set_sha256": artifact.dev_pose_cache_set_sha256,
        "dev_pose_cache_snapshot_sha256": (
            artifact.dev_pose_cache_snapshot_sha256
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
                "count": row.count,
                "period_frames": row.period_frames,
                "expert_counts": list(row.expert_counts),
                "selection_mode": row.selection_mode,
                "selected_expert": row.selected_expert,
                "confidence": row.confidence,
            }
            for row in artifact.records
        ],
    }
    evaluation_encoded = _encoded_json(evaluation_payload)
    evaluation_sha256 = hashlib.sha256(evaluation_encoded).hexdigest()
    evaluation_receipt = {
        "schema_version": 1,
        "artifact_type": "pams_checkpoint_dev_evaluation_receipt",
        "protocol": artifact.protocol,
        "split": "dev",
        "variant": artifact.variant,
        "method_key": artifact.method_key,
        "evaluation_file": _EVALUATION_NAME,
        "evaluation_sha256": evaluation_sha256,
        "evaluation_bytes": len(evaluation_encoded),
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "dev_targets_sha256": targets_sha256,
        "training_config_fingerprint": artifact.training_config_fingerprint,
        "config_fingerprint": artifact.config_fingerprint,
        "consensus_expert_mode": artifact.consensus_expert_mode,
        "ablation_status": artifact.ablation_status,
        "direct_fft_timebase": artifact.direct_fft_timebase,
        "timebase_status": artifact.timebase_status,
        "dev_identity_sha256": artifact.dev_identity_sha256,
        "checkpoint_sha256": artifact.checkpoint_sha256,
        "checkpoint_progress_sha256": artifact.checkpoint_progress_sha256,
        "checkpoint_completion_receipt_sha256": (
            artifact.checkpoint_completion_receipt_sha256
        ),
        "upstream_encoder_checkpoint_sha256": (
            artifact.upstream_encoder_checkpoint_sha256
        ),
        "upstream_encoder_progress_sha256": (
            artifact.upstream_encoder_progress_sha256
        ),
        "upstream_encoder_completion_receipt_sha256": (
            artifact.upstream_encoder_completion_receipt_sha256
        ),
        "dev_pose_cache_snapshot_sha256": (
            artifact.dev_pose_cache_snapshot_sha256
        ),
        "prediction_source_git_sha": artifact.prediction_source_git_sha,
        "prediction_code_sha256": artifact.prediction_code_sha256,
        "scoring_source_git_sha": scoring_source_git_sha,
        "scoring_code_sha256": scoring_code_sha256,
    }
    written = _write_json_bundle(
        (
            (evaluation_path, evaluation_payload),
            (evaluation_receipt_path, evaluation_receipt),
        )
    )
    evaluation_receipt_sha256 = written[evaluation_receipt_path]
    compact_metrics = result.report.to_dict()
    compact_metrics.pop("per_video")
    return {
        "classification": artifact.classification,
        "table2_eligible": False,
        "split": "dev",
        "variant": artifact.variant,
        "method_key": artifact.method_key,
        "consensus_expert_mode": artifact.consensus_expert_mode,
        "ablation_status": artifact.ablation_status,
        "direct_fft_timebase": artifact.direct_fft_timebase,
        "timebase_status": artifact.timebase_status,
        "evaluation_path": str(evaluation_path.resolve()),
        "evaluation_sha256": evaluation_sha256,
        "evaluation_receipt_path": str(evaluation_receipt_path.resolve()),
        "evaluation_receipt_sha256": evaluation_receipt_sha256,
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "dev_targets_sha256": targets_sha256,
        "prediction_source_git_sha": artifact.prediction_source_git_sha,
        "prediction_code_sha256": artifact.prediction_code_sha256,
        "scoring_source_git_sha": scoring_source_git_sha,
        "scoring_code_sha256": scoring_code_sha256,
        "metrics": compact_metrics,
    }
