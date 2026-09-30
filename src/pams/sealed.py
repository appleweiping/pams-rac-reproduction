"""Atomic one-attempt registry for sealed-test evaluation.

The registry is intentionally independent of output directories.  Its
canonical key contains only protocol, the full labelled dataset identity,
method, and experiment seed.  Changing a checkpoint, configuration, Git
revision, or output path therefore cannot create another attempt for the same
pre-registered method/seed pair.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from pams.reproducibility import durable_mkdir, fsync_directory

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


class SealedAttemptAlreadyReservedError(FileExistsError):
    """Raised when a logical sealed-test attempt has already been reserved."""


def _validated_non_empty(value: str, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _validated_sha256(value: str, name: str) -> str:
    digest = str(value).strip()
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(f"{name} must be 64 lowercase hexadecimal characters")
    return digest


def _validated_git_sha(value: str) -> str:
    revision = str(value).strip()
    if not _GIT_SHA_PATTERN.fullmatch(revision):
        raise ValueError("git_sha must be 40 lowercase hexadecimal characters")
    return revision


def _validated_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("experiment_seed must be an integer")
    return value


def _validated_utc_timestamp(value: str) -> str:
    timestamp = str(value).strip()
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError as exc:
        raise ValueError("created_at_utc must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at_utc must include a UTC offset")
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("created_at_utc must use UTC")
    return timestamp


def _canonical_attempt_key(
    *,
    protocol: str,
    full_dataset_sha256: str,
    method_id: str,
    experiment_seed: int,
) -> dict[str, str | int]:
    return {
        "protocol": _validated_non_empty(protocol, "protocol"),
        "full_dataset_sha256": _validated_sha256(
            full_dataset_sha256,
            "full_dataset_sha256",
        ),
        "method_id": _validated_non_empty(method_id, "method_id"),
        "experiment_seed": _validated_seed(experiment_seed),
    }


def sealed_attempt_id(
    *,
    protocol: str,
    full_dataset_sha256: str,
    method_id: str,
    experiment_seed: int,
) -> str:
    """Return the stable SHA-256 identifier for one logical test attempt."""

    key = _canonical_attempt_key(
        protocol=protocol,
        full_dataset_sha256=full_dataset_sha256,
        method_id=method_id,
        experiment_seed=experiment_seed,
    )
    encoded = json.dumps(
        key,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class SealedTestAttemptReceipt(BaseModel):
    """Immutable receipt proving reservation of one sealed-test attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    status: Literal["reserved"] = "reserved"
    attempt_id: str
    created_at_utc: str
    protocol: str
    full_dataset_sha256: str
    config_sha256: str
    method_id: str
    experiment_seed: int
    input_artifact_role: Literal["checkpoint", "predictions"]
    input_artifact_sha256: str
    git_sha: str

    @field_validator("attempt_id")
    @classmethod
    def validate_attempt_id(cls, value: str) -> str:
        return _validated_sha256(value, "attempt_id")

    @field_validator("full_dataset_sha256", "config_sha256", "input_artifact_sha256")
    @classmethod
    def validate_sha256_fields(cls, value: str, info: Any) -> str:
        return _validated_sha256(value, info.field_name)

    @field_validator("git_sha")
    @classmethod
    def validate_git_revision(cls, value: str) -> str:
        return _validated_git_sha(value)

    @field_validator("created_at_utc")
    @classmethod
    def validate_timestamp(cls, value: str) -> str:
        return _validated_utc_timestamp(value)

    @field_validator("protocol", "method_id")
    @classmethod
    def validate_non_empty_fields(cls, value: str, info: Any) -> str:
        return _validated_non_empty(value, info.field_name)

    @field_validator("experiment_seed")
    @classmethod
    def validate_seed(cls, value: int) -> int:
        return _validated_seed(value)

    @model_validator(mode="after")
    def validate_attempt_identity(self) -> SealedTestAttemptReceipt:
        expected = sealed_attempt_id(
            protocol=self.protocol,
            full_dataset_sha256=self.full_dataset_sha256,
            method_id=self.method_id,
            experiment_seed=self.experiment_seed,
        )
        if self.attempt_id != expected:
            raise ValueError("attempt_id does not match the canonical sealed-attempt key")
        return self

    def canonical_json_bytes(self) -> bytes:
        """Return the stable, finite JSON representation written to registry."""

        payload = self.model_dump(mode="json")
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")


def create_sealed_test_attempt_receipt(
    *,
    protocol: str,
    full_dataset_sha256: str,
    config_sha256: str,
    method_id: str,
    experiment_seed: int,
    input_artifact_role: Literal["checkpoint", "predictions"],
    input_artifact_sha256: str,
    git_sha: str,
    created_at_utc: str | None = None,
) -> SealedTestAttemptReceipt:
    """Build and validate a receipt without changing registry state."""

    timestamp = created_at_utc or datetime.now(timezone.utc).isoformat()
    attempt_id = sealed_attempt_id(
        protocol=protocol,
        full_dataset_sha256=full_dataset_sha256,
        method_id=method_id,
        experiment_seed=experiment_seed,
    )
    return SealedTestAttemptReceipt(
        attempt_id=attempt_id,
        created_at_utc=timestamp,
        protocol=protocol,
        full_dataset_sha256=full_dataset_sha256,
        config_sha256=config_sha256,
        method_id=method_id,
        experiment_seed=experiment_seed,
        input_artifact_role=input_artifact_role,
        input_artifact_sha256=input_artifact_sha256,
        git_sha=git_sha,
    )


def sealed_attempt_path(
    registry_dir: str | Path,
    attempt_id: str,
) -> Path:
    """Return the deterministic registry path for a validated attempt ID."""

    identifier = _validated_sha256(attempt_id, "attempt_id")
    return Path(registry_dir) / f"{identifier}.json"


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _cleanup_failed_reservation(path: Path, created_identity: os.stat_result) -> None:
    """Remove only the directory entry created by the failed reservation."""

    try:
        current = path.stat()
    except FileNotFoundError:
        return
    if _same_file_identity(current, created_identity):
        path.unlink(missing_ok=True)
        fsync_directory(path.parent)


def reserve_sealed_test_attempt(
    registry_dir: str | Path,
    *,
    protocol: str,
    full_dataset_sha256: str,
    config_sha256: str,
    method_id: str,
    experiment_seed: int,
    input_artifact_role: Literal["checkpoint", "predictions"],
    input_artifact_sha256: str,
    git_sha: str,
    created_at_utc: str | None = None,
) -> tuple[SealedTestAttemptReceipt, Path]:
    """Atomically reserve exactly one sealed-test attempt.

    The caller must use one shared, durable ``registry_dir`` for the entire
    protocol.  ``O_CREAT | O_EXCL`` makes concurrent reservations of the same
    logical key filesystem-atomic: exactly one caller can create its receipt.
    """

    receipt = create_sealed_test_attempt_receipt(
        protocol=protocol,
        full_dataset_sha256=full_dataset_sha256,
        config_sha256=config_sha256,
        method_id=method_id,
        experiment_seed=experiment_seed,
        input_artifact_role=input_artifact_role,
        input_artifact_sha256=input_artifact_sha256,
        git_sha=git_sha,
        created_at_utc=created_at_utc,
    )
    destination = Path(registry_dir)
    durable_mkdir(destination)
    path = sealed_attempt_path(destination, receipt.attempt_id)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise SealedAttemptAlreadyReservedError(
            "sealed-test attempt already reserved for "
            f"protocol={receipt.protocol!r}, dataset={receipt.full_dataset_sha256}, "
            f"method={receipt.method_id!r}, seed={receipt.experiment_seed}; "
            f"registry receipt: {path}"
        ) from exc

    created_identity = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(receipt.canonical_json_bytes())
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        _cleanup_failed_reservation(path, created_identity)
        raise
    fsync_directory(destination)
    return receipt, path


def load_sealed_test_attempt_receipt(
    path: str | Path,
) -> SealedTestAttemptReceipt:
    """Load an existing registry receipt with strict JSON/schema validation."""

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field in sealed-attempt receipt: {key}")
            result[key] = value
        return result

    def reject_non_finite_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in sealed-attempt receipt: {value}")

    source = Path(path)
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid sealed-attempt receipt JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("sealed-attempt receipt JSON root must be an object")
    return SealedTestAttemptReceipt.model_validate(payload)
