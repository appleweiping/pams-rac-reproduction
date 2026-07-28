"""Immutable, two-stage experiment receipts."""

from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from pams.reproducibility import (
    durable_mkdir,
    fsync_directory,
    git_revision,
    hardware_fingerprint,
    sha256_json,
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")


def _validated_sha256(value: str, name: str) -> str:
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a 64-character lowercase hexadecimal SHA-256")
    return value


def _validated_utc_timestamp(value: str, name: str) -> str:
    timestamp = str(value).strip()
    parseable_timestamp = f"{timestamp[:-1]}+00:00" if timestamp.endswith("Z") else timestamp
    try:
        parsed = datetime.fromisoformat(parseable_timestamp)
    except ValueError as exc:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must include a UTC offset")
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")
    return timestamp


def _validate_json_value(value: Any, name: str) -> Any:
    try:
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite JSON-compatible data") from exc
    return value


class RunManifest(BaseModel):
    """Immutable started receipt for one training or evaluation attempt.

    Schema-v1 documents remain readable for compatibility. New receipts are
    schema v2 and use ``status="started"``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1, 2] = 2
    receipt_type: Literal["started"] = "started"
    run_id: str
    created_at_utc: str
    command: list[str]
    git_sha: str
    config_sha256: str
    dataset_sha256: str
    seed: int
    protocol: str
    status: Literal["created", "started"] = "started"
    hardware: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

    @field_validator("run_id", "protocol")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("value must be non-empty")
        return normalized

    @field_validator("created_at_utc")
    @classmethod
    def validate_created_at(cls, value: str) -> str:
        return _validated_utc_timestamp(value, "created_at_utc")

    @field_validator("git_sha")
    @classmethod
    def validate_git_sha(cls, value: str) -> str:
        revision = str(value).strip()
        if revision != "uncommitted" and not _GIT_SHA_PATTERN.fullmatch(revision):
            raise ValueError("git_sha must be 40 lowercase hex characters or 'uncommitted'")
        return revision

    @field_validator("config_sha256", "dataset_sha256")
    @classmethod
    def validate_sha256_fields(cls, value: str, info: Any) -> str:
        return _validated_sha256(value, info.field_name)

    @field_validator("command")
    @classmethod
    def validate_command(cls, value: list[str]) -> list[str]:
        if not value or any(not isinstance(item, str) or not item for item in value):
            raise ValueError("command must contain non-empty strings")
        return value

    @field_validator("hardware", "notes")
    @classmethod
    def validate_json_fields(cls, value: Any, info: Any) -> Any:
        return _validate_json_value(value, info.field_name)

    @model_validator(mode="after")
    def validate_schema_status(self) -> RunManifest:
        if self.schema_version == 1 and self.status != "created":
            raise ValueError("schema-v1 manifests require status='created'")
        if self.schema_version == 2 and self.status != "started":
            raise ValueError("schema-v2 manifests require status='started'")
        return self

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


class LegacyArtifactReceipt(BaseModel):
    """Schema-v2 artifact identity containing a host-specific absolute path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    path: str
    sha256: str
    bytes: int = Field(ge=0)

    @field_validator("role", "path")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("artifact role and path must be non-empty")
        return normalized

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _validated_sha256(value, "artifact sha256")


def _validated_relative_locator(value: str) -> str:
    locator = str(value).strip()
    if not locator:
        raise ValueError("artifact locator must be non-empty")
    if "\x00" in locator or "\\" in locator:
        raise ValueError("artifact locator must be a portable POSIX relative path")
    if locator.startswith("/") or _WINDOWS_DRIVE_PATTERN.match(locator):
        raise ValueError("artifact locator must be relative to the completed receipt")
    canonical = posixpath.normpath(locator)
    if canonical != locator or canonical == ".":
        raise ValueError("artifact locator must be a canonical relative path")
    return locator


class ArtifactReceipt(BaseModel):
    """Portable content identity for one material input or output artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str
    locator: str
    sha256: str
    bytes: int = Field(ge=0)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("artifact role must be non-empty")
        return normalized

    @field_validator("locator")
    @classmethod
    def validate_locator(cls, value: str) -> str:
        return _validated_relative_locator(value)

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _validated_sha256(value, "artifact sha256")


ArtifactReceiptRecord = ArtifactReceipt | LegacyArtifactReceipt


class CompletedRunReceipt(BaseModel):
    """Immutable completion receipt linked to an already-written start."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2, 3] = 3
    receipt_type: Literal["completed"] = "completed"
    run_id: str
    status: Literal["completed"] = "completed"
    finished_at: str
    start_manifest_sha256: str
    started: RunManifest
    artifacts: tuple[ArtifactReceiptRecord, ...] = ()
    metrics: dict[str, Any] = Field(default_factory=dict)

    @field_validator("finished_at")
    @classmethod
    def validate_finished_at(cls, value: str) -> str:
        return _validated_utc_timestamp(value, "finished_at")

    @field_validator("start_manifest_sha256")
    @classmethod
    def validate_start_sha256(cls, value: str) -> str:
        return _validated_sha256(value, "start_manifest_sha256")

    @field_validator("metrics")
    @classmethod
    def validate_metrics(cls, value: dict[str, Any]) -> dict[str, Any]:
        return _validate_json_value(value, "metrics")

    @model_validator(mode="after")
    def validate_started_identity(self) -> CompletedRunReceipt:
        if self.started.schema_version != 2 or self.started.status != "started":
            raise ValueError("completed receipts require a schema-v2 started receipt")
        if self.run_id != self.started.run_id:
            raise ValueError("completed receipt run_id must match its started receipt")
        started_at = datetime.fromisoformat(self.started.created_at_utc)
        finished_at = datetime.fromisoformat(self.finished_at)
        if finished_at < started_at:
            raise ValueError("finished_at cannot precede created_at_utc")
        roles = [artifact.role for artifact in self.artifacts]
        if len(set(roles)) != len(roles):
            raise ValueError("artifact roles must be unique within a completion receipt")
        if self.schema_version == 2 and any(
            not isinstance(artifact, LegacyArtifactReceipt) for artifact in self.artifacts
        ):
            raise ValueError("schema-v2 completed receipts require path-based artifacts")
        if self.schema_version == 3 and any(
            not isinstance(artifact, ArtifactReceipt) for artifact in self.artifacts
        ):
            raise ValueError("schema-v3 completed receipts require receipt-relative locators")
        return self

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.model_dump(mode="json"))


def create_run_manifest(
    *,
    command: list[str],
    config_sha256: str,
    dataset_sha256: str,
    seed: int,
    protocol: str,
    cwd: str | Path | None = None,
    notes: list[str] | None = None,
) -> RunManifest:
    """Create a schema-v2 started receipt while preserving the original API."""

    now = datetime.now(timezone.utc)
    git_sha = git_revision(cwd)
    identity = {
        "command": command,
        "config": config_sha256,
        "dataset": dataset_sha256,
        "seed": seed,
        "protocol": protocol,
        "git": git_sha,
        "time": now.isoformat(),
    }
    run_id = f"{now:%Y%m%dT%H%M%SZ}-{sha256_json(identity)[:12]}"
    return RunManifest(
        run_id=run_id,
        created_at_utc=now.isoformat(),
        command=command,
        git_sha=git_sha,
        config_sha256=config_sha256,
        dataset_sha256=dataset_sha256,
        seed=seed,
        protocol=protocol,
        hardware=hardware_fingerprint(),
        notes=notes or [],
    )


def _artifact_identity(path: str | Path) -> tuple[str, int]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"artifact does not exist or is not a file: {source}")
    before = source.stat()
    digest = hashlib.sha256()
    byte_count = 0
    with source.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
        closed = os.fstat(handle.fileno())
    after = source.stat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(closed, field)
        or getattr(closed, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise RuntimeError(f"artifact changed while it was being hashed: {source}")
    if byte_count != after.st_size:
        raise RuntimeError(f"artifact byte count changed while hashing: {source}")
    return digest.hexdigest(), byte_count


def _receipt_relative_locator(path: Path, receipt_directory: Path) -> str:
    try:
        relative = os.path.relpath(
            path.resolve(),
            start=receipt_directory.resolve(),
        )
    except ValueError as exc:
        raise ValueError(
            "artifact cannot be represented relative to the completed receipt; "
            "place the receipt and artifact on the same filesystem volume"
        ) from exc
    locator = relative.replace("\\", "/")
    return _validated_relative_locator(locator)


def _artifact_receipt(
    role: str,
    path: str | Path,
    *,
    receipt_directory: Path,
) -> ArtifactReceipt:
    source = Path(path)
    digest, byte_count = _artifact_identity(source)
    return ArtifactReceipt(
        role=role,
        locator=_receipt_relative_locator(source, receipt_directory),
        sha256=digest,
        bytes=byte_count,
    )


def resolve_artifact_path(
    completed_receipt_path: str | Path,
    artifact: ArtifactReceiptRecord,
    *,
    remapped_path: str | Path | None = None,
) -> Path:
    """Resolve an artifact without using the process working directory.

    Explicit remaps must be host-absolute. Schema-v3 locators are interpreted
    relative to the directory containing the completed receipt and confined
    to its package root. Legacy schema-v2 host paths always require an
    explicit role remap.
    """

    if remapped_path is not None:
        remapped = Path(remapped_path).expanduser()
        if not remapped.is_absolute():
            raise ValueError(
                f"artifact remap for role {artifact.role!r} must use an absolute path"
            )
        return remapped.resolve(strict=False)
    if isinstance(artifact, ArtifactReceipt):
        parts = PurePosixPath(artifact.locator).parts
        receipt_path = Path(completed_receipt_path).resolve(strict=False)
        receipt_directory = receipt_path.parent
        artifact_root = (
            receipt_directory.parent
            if receipt_directory.name == "manifests"
            else receipt_directory
        )
        candidate = receipt_directory.joinpath(*parts).resolve(strict=False)
        try:
            candidate.relative_to(artifact_root)
        except ValueError as exc:
            raise ValueError(
                f"artifact locator for role {artifact.role!r} escapes the trusted "
                "receipt artifact root; provide an explicit absolute role remap"
            ) from exc
        return candidate

    raise ValueError(
        f"legacy artifact path for role {artifact.role!r} is host-specific and "
        "requires an explicit absolute role remap"
    )


def validate_artifact_receipt(
    path: str | Path,
    artifact: ArtifactReceiptRecord,
) -> None:
    """Re-read one artifact stably and verify both byte count and SHA-256."""

    source = Path(path)
    try:
        observed_size = source.stat().st_size
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"completed receipt is missing artifact for role {artifact.role!r}"
        ) from exc
    if observed_size != artifact.bytes:
        raise ValueError(
            f"artifact size does not match receipt for role {artifact.role!r}"
        )
    try:
        digest, byte_count = _artifact_identity(source)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"completed receipt is missing artifact for role {artifact.role!r}"
        ) from exc
    if byte_count != artifact.bytes:
        raise ValueError(
            f"artifact size does not match receipt for role {artifact.role!r}"
        )
    if digest != artifact.sha256:
        raise ValueError(
            f"artifact SHA-256 does not match receipt for role {artifact.role!r}"
        )


def create_completed_receipt(
    started_manifest_path: str | Path,
    *,
    artifacts: Mapping[str, str | Path] | None = None,
    expected_artifact_sha256: Mapping[str, str] | None = None,
    metrics: Mapping[str, Any] | None = None,
    finished_at: str | None = None,
) -> CompletedRunReceipt:
    """Build a completion receipt from a durable started file and artifacts.

    When ``expected_artifact_sha256`` is supplied, every artifact is re-hashed
    at the terminal boundary and must match the digest captured when the
    command actually consumed or produced those bytes.  This prevents a file
    replacement during a long run from making the completion receipt attest
    to different bytes than the computation used.
    """

    start_path = Path(started_manifest_path)
    if not start_path.is_file():
        raise FileNotFoundError(f"started manifest does not exist: {start_path}")
    before = start_path.stat()
    with start_path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        start_payload = handle.read()
        closed = os.fstat(handle.fileno())
    after = start_path.stat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(closed, field)
        or getattr(closed, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise RuntimeError(f"started manifest changed while it was being read: {start_path}")
    started = RunManifest.model_validate_json(start_payload)
    if started.schema_version != 2 or started.status != "started":
        raise ValueError("completion requires a schema-v2 started manifest")
    artifact_paths = dict(artifacts or {})
    expected_digests = None if expected_artifact_sha256 is None else dict(expected_artifact_sha256)
    if expected_digests is not None:
        if set(expected_digests) != set(artifact_paths):
            raise ValueError("expected artifact SHA-256 roles must exactly match artifact roles")
        for role, digest in expected_digests.items():
            _validated_sha256(digest, f"expected SHA-256 for artifact role {role!r}")

    receipts: list[ArtifactReceipt] = []
    for role, artifact_path in sorted(artifact_paths.items()):
        receipt = _artifact_receipt(
            role,
            artifact_path,
            receipt_directory=start_path.resolve().parent,
        )
        if expected_digests is not None and receipt.sha256 != expected_digests[role]:
            raise RuntimeError(
                f"artifact SHA-256 changed before completion for role {role!r}: "
                f"expected {expected_digests[role]}, observed {receipt.sha256}"
            )
        receipts.append(receipt)
    artifact_receipts = tuple(receipts)
    return CompletedRunReceipt(
        run_id=started.run_id,
        finished_at=finished_at or datetime.now(timezone.utc).isoformat(),
        start_manifest_sha256=hashlib.sha256(start_payload).hexdigest(),
        started=started,
        artifacts=artifact_receipts,
        metrics=dict(metrics or {}),
    )


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def write_manifest_exclusive(
    manifest: RunManifest | CompletedRunReceipt,
    path: str | Path,
) -> Path:
    """Durably create one receipt without any check-then-replace race."""

    target = Path(path)
    durable_mkdir(target.parent)
    payload = (
        json.dumps(
            manifest.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(target, flags, 0o644)
    owned_identity = os.fstat(descriptor)
    try:
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            current = target.stat()
            if _same_file_identity(current, owned_identity):
                target.unlink()
                fsync_directory(target.parent)
        except OSError:
            pass
        raise
    fsync_directory(target.parent)
    return target
