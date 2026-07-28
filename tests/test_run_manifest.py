import hashlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import pams.run_manifest as run_manifest_module
from pams.run_manifest import (
    ArtifactReceipt,
    CompletedRunReceipt,
    LegacyArtifactReceipt,
    RunManifest,
    create_completed_receipt,
    create_run_manifest,
    resolve_artifact_path,
    validate_artifact_receipt,
    write_manifest_exclusive,
)


def _create_started(tmp_path: Path) -> RunManifest:
    return create_run_manifest(
        command=["pams", "train"],
        config_sha256="a" * 64,
        dataset_sha256="b" * 64,
        seed=2026,
        protocol="synthetic",
        cwd=tmp_path,
    )


def test_started_manifest_is_v2_durable_and_write_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _create_started(tmp_path)
    assert manifest.schema_version == 2
    assert manifest.receipt_type == "started"
    assert manifest.status == "started"

    path = tmp_path / "manifest.json"
    open_flags: list[int] = []
    synced_descriptors: list[int] = []
    real_open = os.open
    real_fsync = os.fsync

    def tracked_open(path_value: Any, flags: int, mode: int = 0o777) -> int:
        open_flags.append(flags)
        return real_open(path_value, flags, mode)

    def tracked_fsync(descriptor: int) -> None:
        synced_descriptors.append(descriptor)
        real_fsync(descriptor)

    monkeypatch.setattr(run_manifest_module.os, "open", tracked_open)
    monkeypatch.setattr(run_manifest_module.os, "fsync", tracked_fsync)
    write_manifest_exclusive(manifest, path)
    original = path.read_bytes()

    assert open_flags
    assert open_flags[0] & os.O_CREAT
    assert open_flags[0] & os.O_EXCL
    assert synced_descriptors
    assert RunManifest.model_validate_json(original) == manifest

    with pytest.raises(FileExistsError):
        write_manifest_exclusive(manifest, path)
    assert path.read_bytes() == original


def test_started_manifest_records_validated_container_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PAMS_CONTAINER_IMAGE_ID", f"sha256:{'a' * 64}")
    monkeypatch.setenv("PAMS_CONTAINER_ENVIRONMENT_SHA256", "b" * 64)
    monkeypatch.setenv("PAMS_CONTAINER_SOURCE_REVISION", "c" * 40)
    manifest = _create_started(tmp_path)
    assert manifest.hardware["container"] == {
        "image_id": f"sha256:{'a' * 64}",
        "environment_sha256": "b" * 64,
        "source_revision": "c" * 40,
    }

    monkeypatch.delenv("PAMS_CONTAINER_SOURCE_REVISION")
    with pytest.raises(RuntimeError, match="incomplete"):
        _create_started(tmp_path)


def test_completed_receipt_binds_start_identity_artifacts_and_metrics(tmp_path: Path) -> None:
    started = _create_started(tmp_path)
    started_path = write_manifest_exclusive(started, tmp_path / "started.json")
    checkpoint = tmp_path / "checkpoint.pt"
    predictions = tmp_path / "predictions.json"
    checkpoint.write_bytes(b"model-weights")
    predictions.write_text('{"count": 7}\n', encoding="utf-8")
    finished_at = (
        datetime.fromisoformat(started.created_at_utc) + timedelta(seconds=1)
    ).isoformat()

    completed = create_completed_receipt(
        started_path,
        artifacts={"predictions": predictions, "checkpoint": checkpoint},
        metrics={"nmae": 0.2, "per_seed": {"2026": 0.19}},
        finished_at=finished_at,
    )

    assert completed.schema_version == 3
    assert completed.receipt_type == "completed"
    assert completed.status == "completed"
    assert completed.run_id == started.run_id
    assert completed.started == started
    assert completed.finished_at == finished_at
    assert completed.start_manifest_sha256 == hashlib.sha256(started_path.read_bytes()).hexdigest()
    assert completed.metrics == {"nmae": 0.2, "per_seed": {"2026": 0.19}}
    assert [artifact.role for artifact in completed.artifacts] == ["checkpoint", "predictions"]

    artifacts = {artifact.role: artifact for artifact in completed.artifacts}
    assert isinstance(artifacts["checkpoint"], ArtifactReceipt)
    assert artifacts["checkpoint"].locator == "checkpoint.pt"
    assert artifacts["checkpoint"].bytes == len(b"model-weights")
    assert artifacts["checkpoint"].sha256 == hashlib.sha256(b"model-weights").hexdigest()
    assert isinstance(artifacts["predictions"], ArtifactReceipt)
    assert artifacts["predictions"].locator == "predictions.json"
    assert artifacts["predictions"].bytes == len(predictions.read_bytes())
    assert artifacts["predictions"].sha256 == hashlib.sha256(predictions.read_bytes()).hexdigest()

    completed_path = write_manifest_exclusive(completed, tmp_path / "completed.json")
    original = completed_path.read_bytes()
    assert CompletedRunReceipt.model_validate_json(original) == completed
    with pytest.raises(FileExistsError):
        write_manifest_exclusive(completed, completed_path)
    assert completed_path.read_bytes() == original


def test_schema_v3_artifact_locator_survives_moving_the_receipt_tree(
    tmp_path: Path,
) -> None:
    original_root = tmp_path / "original"
    manifests = original_root / "manifests"
    manifests.mkdir(parents=True)
    started = _create_started(original_root)
    started_path = write_manifest_exclusive(
        started,
        manifests / f"{started.run_id}.started.json",
    )
    artifact = original_root / "artifacts" / "checkpoint.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"portable-checkpoint")
    completed = create_completed_receipt(
        started_path,
        artifacts={"checkpoint": artifact},
    )
    completed_path = write_manifest_exclusive(
        completed,
        manifests / f"{started.run_id}.completed.json",
    )
    record = completed.artifacts[0]
    assert isinstance(record, ArtifactReceipt)
    assert record.locator == "../artifacts/checkpoint.pt"

    moved_root = tmp_path / "moved"
    original_root.rename(moved_root)
    moved_receipt = moved_root / "manifests" / completed_path.name
    moved_record = CompletedRunReceipt.model_validate_json(
        moved_receipt.read_bytes()
    ).artifacts[0]
    resolved = resolve_artifact_path(moved_receipt, moved_record)

    assert resolved == moved_root / "artifacts" / "checkpoint.pt"
    validate_artifact_receipt(resolved, moved_record)


def test_schema_v2_absolute_path_requires_safe_migration_remap(
    tmp_path: Path,
) -> None:
    started = _create_started(tmp_path)
    artifact = tmp_path / "legacy.bin"
    payload = b"legacy-artifact"
    artifact.write_bytes(payload)
    legacy = CompletedRunReceipt(
        schema_version=2,
        run_id=started.run_id,
        finished_at=started.created_at_utc,
        start_manifest_sha256="a" * 64,
        started=started,
        artifacts=(
            LegacyArtifactReceipt(
                role="fixture",
                path=str(artifact.resolve()),
                sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload),
            ),
        ),
    )
    parsed = CompletedRunReceipt.model_validate_json(
        legacy.model_dump_json()
    )
    record = parsed.artifacts[0]
    assert isinstance(record, LegacyArtifactReceipt)
    with pytest.raises(ValueError, match="explicit absolute role remap"):
        resolve_artifact_path(tmp_path / "legacy.completed.json", record)
    original_remap = resolve_artifact_path(
        tmp_path / "legacy.completed.json",
        record,
        remapped_path=artifact.resolve(),
    )
    validate_artifact_receipt(original_remap, record)

    migrated = tmp_path / "migrated" / "legacy.bin"
    migrated.parent.mkdir()
    artifact.rename(migrated)
    with pytest.raises(FileNotFoundError, match="missing artifact"):
        validate_artifact_receipt(
            original_remap,
            record,
        )
    remapped = resolve_artifact_path(
        tmp_path / "legacy.completed.json",
        record,
        remapped_path=migrated.resolve(),
    )
    validate_artifact_receipt(remapped, record)
    with pytest.raises(ValueError, match="absolute path"):
        resolve_artifact_path(
            tmp_path / "legacy.completed.json",
            record,
            remapped_path=Path("relative.bin"),
        )


@pytest.mark.parametrize(
    ("field", "digest"),
    [
        ("config_sha256", "A" * 64),
        ("config_sha256", "a" * 63),
        ("config_sha256", f" {'a' * 64}"),
        ("dataset_sha256", "g" * 64),
        ("dataset_sha256", "b" * 65),
    ],
)
def test_started_manifest_rejects_noncanonical_sha256(
    tmp_path: Path,
    field: str,
    digest: str,
) -> None:
    kwargs: dict[str, Any] = {
        "command": ["pams", "train"],
        "config_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "seed": 42,
        "protocol": "synthetic",
        "cwd": tmp_path,
    }
    kwargs[field] = digest

    with pytest.raises(ValidationError, match=field):
        create_run_manifest(**kwargs)


@pytest.mark.parametrize("digest", ["A" * 64, "0" * 63, "z" * 64, f"{'0' * 64}\n"])
def test_artifact_and_start_links_require_canonical_sha256(
    tmp_path: Path,
    digest: str,
) -> None:
    with pytest.raises(ValidationError, match="SHA-256"):
        ArtifactReceipt(role="checkpoint", locator="../model.pt", sha256=digest, bytes=1)
    started = _create_started(tmp_path)
    with pytest.raises(ValidationError, match="start_manifest_sha256"):
        CompletedRunReceipt(
            run_id=started.run_id,
            finished_at=started.created_at_utc,
            start_manifest_sha256=digest,
            started=started,
        )


@pytest.mark.parametrize(
    "locator",
    [
        "",
        ".",
        "/absolute/model.pt",
        "C:/absolute/model.pt",
        "nested\\model.pt",
        "nested//model.pt",
        "nested/./model.pt",
    ],
)
def test_schema_v3_artifact_locator_must_be_canonical_and_relative(
    locator: str,
) -> None:
    with pytest.raises(ValidationError, match="locator"):
        ArtifactReceipt(
            role="checkpoint",
            locator=locator,
            sha256="a" * 64,
            bytes=1,
        )


def test_completion_helper_requires_durable_v2_start_and_finite_metrics(tmp_path: Path) -> None:
    started = _create_started(tmp_path)
    unwritten_path = tmp_path / "missing-started.json"
    with pytest.raises(FileNotFoundError):
        create_completed_receipt(unwritten_path)

    started_path = write_manifest_exclusive(started, tmp_path / "started.json")
    with pytest.raises(ValidationError, match="metrics"):
        create_completed_receipt(started_path, metrics={"loss": float("nan")})

    legacy = started.model_copy(update={"schema_version": 1, "status": "created"})
    legacy_path = write_manifest_exclusive(legacy, tmp_path / "legacy.json")
    with pytest.raises(ValueError, match="schema-v2"):
        create_completed_receipt(legacy_path)


def test_completion_requires_the_exact_artifact_bytes_consumed(tmp_path: Path) -> None:
    started = _create_started(tmp_path)
    started_path = write_manifest_exclusive(started, tmp_path / "started.json")
    artifact = tmp_path / "predictions.json"
    consumed = b'{"count": 4}\n'
    artifact.write_bytes(consumed)
    expected = hashlib.sha256(consumed).hexdigest()

    completed = create_completed_receipt(
        started_path,
        artifacts={"predictions": artifact},
        expected_artifact_sha256={"predictions": expected},
    )
    assert completed.artifacts[0].sha256 == expected

    artifact.write_bytes(b'{"count": 400}\n')
    with pytest.raises(RuntimeError, match="changed before completion"):
        create_completed_receipt(
            started_path,
            artifacts={"predictions": artifact},
            expected_artifact_sha256={"predictions": expected},
        )
    with pytest.raises(ValueError, match="roles must exactly match"):
        create_completed_receipt(
            started_path,
            artifacts={"predictions": artifact},
            expected_artifact_sha256={"checkpoint": "a" * 64},
        )


def test_failed_write_never_overwrites_or_leaves_partial_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _create_started(tmp_path)
    occupied = tmp_path / "occupied.json"
    occupied.write_bytes(b"owner-data")

    with pytest.raises(FileExistsError):
        write_manifest_exclusive(manifest, occupied)
    assert occupied.read_bytes() == b"owner-data"

    def fail_fsync(descriptor: int) -> None:
        raise OSError(f"simulated fsync failure for descriptor {descriptor}")

    monkeypatch.setattr(run_manifest_module.os, "fsync", fail_fsync)
    partial = tmp_path / "partial.json"
    with pytest.raises(OSError, match="simulated fsync failure"):
        write_manifest_exclusive(manifest, partial)
    assert not partial.exists()
