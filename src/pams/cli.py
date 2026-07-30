"""Command-line entry point for the auditable PAMS reproduction."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import typer
from pydantic import ValidationError

from pams.baselines import BaselineUnavailableError, create_baseline, list_baselines
from pams.config import PAMSConfig, load_config
from pams.data import (
    DevTargetManifest,
    DevTargetRecord,
    LabelFreeProtocolInputs,
    PoseCacheSetSnapshot,
    PoseInputCommitment,
    PoseInputManifest,
    TrainingPoseDataset,
    UCFRepManifest,
    UCFRepRecord,
    UnlabeledVideoRecord,
    assert_split_disjoint,
    load_dev_target_manifest,
    load_pose_cache,
    load_pose_cache_set,
    load_pose_input_commitment,
    load_pose_input_manifest,
    load_ucfrep_manifest,
    pose_input_identity_sha256,
    save_ucfrep_manifest,
    split_ucfrep_pose_train_dev,
    split_ucfrep_train_dev,
    validate_canonical_split_membership,
    validate_pose_input_binding,
)
from pams.metrics import compute_count_metrics
from pams.reproducibility import durable_mkdir, fsync_directory
from pams.synthetic import SyntheticSpec, generate_synthetic_sample, synthetic_stress_suite

app = typer.Typer(
    name="pams",
    help="Independent, auditable PAMS repetition-counting reproduction.",
    no_args_is_help=True,
    add_completion=False,
)
config_app = typer.Typer(help="Validate frozen experiment configuration.", no_args_is_help=True)
data_app = typer.Typer(help="Validate and split UCFRep manifests.", no_args_is_help=True)
pose_app = typer.Typer(help="Extract and cache MediaPipe 33x3 pose.", no_args_is_help=True)
train_app = typer.Typer(help="Train the encoder or inferred SSHead.", no_args_is_help=True)
evaluate_app = typer.Typer(
    help="Evaluate a checkpoint on sealed cached poses.", no_args_is_help=True
)
baseline_app = typer.Typer(
    help="Inspect baselines or run the explicit spectral proxy.", no_args_is_help=True
)
report_app = typer.Typer(help="Compute frozen RAC metrics from predictions.", no_args_is_help=True)
synthetic_app = typer.Typer(help="Run deterministic synthetic diagnostics.", no_args_is_help=True)
diagnostic_app = typer.Typer(
    help="Run explicitly inferred, label-free implementation diagnostics.",
    no_args_is_help=True,
)
local_frequency_app = typer.Typer(
    help="Run the synthetic-frozen readout through a strict dev-only firewall.",
    no_args_is_help=True,
)
teacher_period_app = typer.Typer(
    help="Run the frozen-v8 direct teacher-period diagnostic on dev84.",
    no_args_is_help=True,
)
stress_app = typer.Typer(
    help="Run deterministic pose-cache stress tests through a dev-only firewall.",
    no_args_is_help=True,
)

_FROZEN_PAMS_CHECKPOINT_METHODS_BY_PROTOCOL: dict[
    str,
    dict[str, Literal["literal", "sshead"]],
] = {
    "ucfrep_526": {
        "pams-literal": "literal",
        "pams-sshead": "sshead",
    },
}
_FROZEN_SEALED_REPORT_METHOD_IDS_BY_PROTOCOL: dict[str, frozenset[str]] = {
    # Baseline names move into this set only after a runnable adapter,
    # method-specific provenance sidecar, and parity gate exist.  An arbitrary
    # prediction JSON must never create an audited paper-table row.
    "ucfrep_526": frozenset(),
}
_FROZEN_PAMS_NONSEED_FINGERPRINT = (
    "2c995b374bc8cf97df3568d745dabd94f111aa54224c86ece51468cbe419b47b"
)
_FROZEN_PAMS_SEEDS = frozenset({42, 2026, 3407})
_FORMAL_AUDIT_MODES = frozenset({"formal", "sealed"})


def _formal_label_firewall_required() -> bool:
    return os.environ.get("PAMS_AUDIT_MODE", "").strip().lower() in _FORMAL_AUDIT_MODES


def _require_formal_label_free_inputs(
    *,
    label_free_inputs: bool,
    operation: str,
) -> None:
    """Fail before any label-bearing manifest loader can be reached."""

    if _formal_label_firewall_required() and not label_free_inputs:
        raise ValueError(
            f"{operation} in formal/sealed audit mode requires --label-free-inputs; "
            "a full count/action manifest is forbidden at this process boundary"
        )


def _normalize_checkpoint_variant(value: str) -> Literal["literal", "sshead"]:
    normalized = value.strip().lower()
    if normalized not in {"literal", "sshead"}:
        raise ValueError("checkpoint variant must be 'literal' or 'sshead'")
    return cast(Literal["literal", "sshead"], normalized)


def _validate_frozen_pams_test_config(config: PAMSConfig) -> None:
    if config.seed not in _FROZEN_PAMS_SEEDS:
        raise ValueError(
            "sealed UCFRep evaluation requires a preregistered seed: 42, 2026, or 3407"
        )
    if config.nonseed_fingerprint != _FROZEN_PAMS_NONSEED_FINGERPRINT:
        raise ValueError(
            "sealed PAMS method configuration differs from the frozen "
            "non-seed specification in configs/pams.yaml"
        )


def _normalize_frozen_method_id(
    protocol: str,
    method_id: str,
    *,
    checkpoint_variant: Literal["literal", "sshead"] | None = None,
) -> str:
    """Validate one protocol-scoped, preregistered sealed-test identity."""

    normalized = method_id.strip().lower()
    registry = (
        _FROZEN_SEALED_REPORT_METHOD_IDS_BY_PROTOCOL
        if checkpoint_variant is None
        else {
            key: frozenset(value)
            for key, value in _FROZEN_PAMS_CHECKPOINT_METHODS_BY_PROTOCOL.items()
        }
    )
    allowed = registry.get(protocol)
    if allowed is None:
        raise ValueError(f"no frozen sealed-test method registry for protocol {protocol!r}")
    if normalized not in allowed:
        raise ValueError(
            "sealed test method_id is not in the frozen "
            f"{protocol} source-audit registry: {normalized!r}"
        )
    if checkpoint_variant is not None:
        expected_variant = _FROZEN_PAMS_CHECKPOINT_METHODS_BY_PROTOCOL[protocol][normalized]
        if expected_variant != checkpoint_variant:
            raise ValueError(
                f"sealed method_id {normalized!r} requires checkpoint variant "
                f"{expected_variant!r}, not {checkpoint_variant!r}"
            )
    return normalized


app.add_typer(config_app, name="config")
app.add_typer(data_app, name="data")
app.add_typer(pose_app, name="pose")
app.add_typer(train_app, name="train")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(baseline_app, name="baseline")
app.add_typer(report_app, name="report")
app.add_typer(synthetic_app, name="synthetic")
app.add_typer(diagnostic_app, name="diagnostic")
app.add_typer(local_frequency_app, name="local-frequency")
app.add_typer(teacher_period_app, name="teacher-period")
app.add_typer(stress_app, name="stress")


def _emit(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))


def _abort(message: str, *, code: int = 2) -> None:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(code)


def _reject_duplicate_run_receipt_fields(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in pairs:
        if key in parsed:
            raise ValueError(f"duplicate JSON field in run receipt: {key}")
        parsed[key] = value
    return parsed


def _write_json_exclusive(path: Path, payload: Any, *, overwrite: bool = False) -> None:
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
    if overwrite:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            fsync_directory(path.parent)
        except Exception:
            if descriptor >= 0:
                os.close(descriptor)
            if temporary.exists():
                temporary.unlink()
                fsync_directory(path.parent)
            raise
        return

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite existing file: {path}") from exc
    created_identity = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            current = path.stat()
            if (
                current.st_dev == created_identity.st_dev
                and current.st_ino == created_identity.st_ino
            ):
                path.unlink()
                fsync_directory(path.parent)
        except OSError:
            pass
        raise
    fsync_directory(path.parent)


def _copy_file_exclusive(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> None:
    """Copy one immutable input without overwriting either lineage."""

    if source.resolve() == destination.resolve():
        raise ValueError("resume input and output paths must be different")
    before = source.stat()
    if not source.is_file():
        raise FileNotFoundError(f"resume input is not a regular file: {source}")
    durable_mkdir(destination.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(destination, flags, 0o644)
    created_identity = os.fstat(descriptor)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with source.open("rb") as input_handle:
            opened = os.fstat(input_handle.fileno())
            with os.fdopen(descriptor, "wb") as output_handle:
                descriptor = -1
                while chunk := input_handle.read(1024 * 1024):
                    digest.update(chunk)
                    byte_count += len(chunk)
                    output_handle.write(chunk)
                closed = os.fstat(input_handle.fileno())
                output_handle.flush()
                os.fsync(output_handle.fileno())
        after = source.stat()
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(
            getattr(before, field) != getattr(opened, field)
            or getattr(opened, field) != getattr(closed, field)
            or getattr(closed, field) != getattr(after, field)
            for field in stable_fields
        ):
            raise RuntimeError(f"resume input changed while it was copied: {source}")
        if byte_count != before.st_size or digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"resume input hash changed before copy: {source}")
        fsync_directory(destination.parent)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            current = destination.stat()
            if (
                current.st_dev == created_identity.st_dev
                and current.st_ino == created_identity.st_ino
            ):
                destination.unlink()
                fsync_directory(destination.parent)
        except OSError:
            pass
        raise


def _validate_reusable_receipt_snapshot(
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
) -> None:
    """Accept an already-published snapshot only when its identity is exact."""

    before = destination.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise FileExistsError(
            "receipt artifact snapshot already exists but is not a regular "
            f"non-symlink file: {destination}"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(destination, flags)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise FileExistsError(f"receipt artifact snapshot is not a regular file: {destination}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
            closed = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = destination.lstat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(closed, field)
        or getattr(closed, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise RuntimeError(
            f"receipt artifact snapshot changed while it was verified: {destination}"
        )
    if (
        byte_count != expected_bytes
        or after.st_size != expected_bytes
        or digest.hexdigest() != expected_sha256
    ):
        raise FileExistsError(
            f"receipt artifact snapshot already exists with different bytes: {destination}"
        )


def _copy_receipt_snapshot_atomic(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> None:
    """Copy through a same-directory temporary and publish without replacement."""

    durable_mkdir(destination.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        before = source.stat()
        if not source.is_file():
            raise FileNotFoundError(f"receipt artifact source is not a regular file: {source}")
        with source.open("rb") as input_handle:
            opened = os.fstat(input_handle.fileno())
            with os.fdopen(descriptor, "wb") as output_handle:
                descriptor = -1
                while chunk := input_handle.read(1024 * 1024):
                    digest.update(chunk)
                    byte_count += len(chunk)
                    output_handle.write(chunk)
                closed = os.fstat(input_handle.fileno())
                output_handle.flush()
                os.fsync(output_handle.fileno())
        after = source.stat()
        stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        if any(
            getattr(before, field) != getattr(opened, field)
            or getattr(opened, field) != getattr(closed, field)
            or getattr(closed, field) != getattr(after, field)
            for field in stable_fields
        ):
            raise RuntimeError(f"receipt artifact changed while it was copied: {source}")
        observed_sha256 = digest.hexdigest()
        if byte_count != before.st_size or observed_sha256 != expected_sha256:
            raise RuntimeError(f"receipt artifact hash changed before snapshot: {source}")
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            _validate_reusable_receipt_snapshot(
                destination,
                expected_sha256=expected_sha256,
                expected_bytes=byte_count,
            )
        else:
            fsync_directory(destination.parent)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        else:
            fsync_directory(destination.parent)


def _durable_receipt_snapshot_directory(path: Path, *, package_root: Path) -> None:
    """Create one snapshot directory while rejecting symlinked components."""

    try:
        relative = path.relative_to(package_root)
    except ValueError as exc:
        raise ValueError("receipt artifact snapshot directory escapes package root") from exc
    current = package_root
    for part in relative.parts:
        current /= part
        try:
            existing = current.lstat()
        except FileNotFoundError:
            durable_mkdir(current)
            existing = current.lstat()
        if stat.S_ISLNK(existing.st_mode) or not stat.S_ISDIR(existing.st_mode):
            raise ValueError(
                "receipt artifact snapshot path contains a non-directory or "
                f"symlink component: {current}"
            )
    try:
        path.resolve().relative_to(package_root)
    except ValueError as exc:
        raise ValueError("receipt artifact snapshot directory escapes package root") from exc


def _materialize_portable_receipt_artifacts(
    started_manifest_path: Path,
    *,
    artifacts: dict[str, Path],
    expected_artifact_sha256: dict[str, str],
) -> dict[str, Path]:
    """Snapshot external artifacts into the self-contained run package.

    Schema-v3 locators are confined to the package rooted above ``manifests``.
    Training inputs, upstream checkpoints, and the canonical sealed-attempt
    registry intentionally live outside that tree while a command runs. At the
    terminal boundary, copy only those external files while enforcing the
    SHA-256 values captured when they were consumed.
    """

    if set(artifacts) != set(expected_artifact_sha256):
        raise ValueError("expected artifact SHA-256 roles must exactly match artifact roles")
    receipt_directory = started_manifest_path.resolve().parent
    package_root = (
        receipt_directory.parent if receipt_directory.name == "manifests" else receipt_directory
    )
    started_receipt_sha256 = _sha256_file(started_manifest_path)
    snapshot_directory = package_root / "inputs" / "receipt-artifacts" / started_receipt_sha256
    portable: dict[str, Path] = {}
    for role, raw_path in sorted(artifacts.items()):
        source = raw_path.resolve()
        try:
            source.relative_to(package_root)
        except ValueError:
            _durable_receipt_snapshot_directory(
                snapshot_directory,
                package_root=package_root,
            )
            role_digest = hashlib.sha256(role.encode("utf-8")).hexdigest()
            destination = snapshot_directory / f"{role_digest}.artifact"
            _copy_receipt_snapshot_atomic(
                source,
                destination,
                expected_sha256=expected_artifact_sha256[role],
            )
            portable[role] = destination
        else:
            portable[role] = source
    return portable


def _load_manifest(path: Path, *, protocol: str | None, exact: bool) -> UCFRepManifest:
    return load_ucfrep_manifest(path, protocol=protocol, validate_exact=exact)


@dataclass(frozen=True, slots=True)
class _LabelFreeInputBundle:
    inputs: LabelFreeProtocolInputs
    artifacts: dict[str, Path]
    artifact_sha256: dict[str, str]


def _load_bound_pose_inputs(
    sidecar_path: Path,
    commitment_path: Path,
    *,
    expected_split: str,
) -> tuple[PoseInputManifest, str, str]:
    """Load and byte-bind one count/action-free split without TOCTOU ambiguity."""

    sidecar_sha256 = _sha256_file(sidecar_path)
    commitment_sha256 = _sha256_file(commitment_path)
    sidecar = load_pose_input_manifest(sidecar_path, validate_exact=True)
    commitment = load_pose_input_commitment(commitment_path)
    validate_pose_input_binding(
        sidecar,
        commitment,
        sidecar_sha256=sidecar_sha256,
    )
    if sidecar.split != expected_split:
        raise ValueError(f"expected {expected_split!r} pose inputs, received {sidecar.split!r}")
    if _sha256_file(sidecar_path) != sidecar_sha256:
        raise RuntimeError("pose-input sidecar changed while it was being loaded")
    if _sha256_file(commitment_path) != commitment_sha256:
        raise RuntimeError("pose-input commitment changed while it was being loaded")
    return sidecar, sidecar_sha256, commitment_sha256


def _load_label_free_protocol_inputs(
    train_inputs_path: Path,
    *,
    train_commitment_path: Path | None,
    dev_inputs_path: Path | None,
    dev_commitment_path: Path | None,
    test_identity_inputs_path: Path | None,
    test_identity_commitment_path: Path | None,
) -> _LabelFreeInputBundle:
    """Assemble the frozen 337/84/105 identity using label-free files only."""

    required = {
        "--input-commitment": train_commitment_path,
        "--dev-inputs": dev_inputs_path,
        "--dev-input-commitment": dev_commitment_path,
        "--test-identity-inputs": test_identity_inputs_path,
        "--test-identity-commitment": test_identity_commitment_path,
    }
    missing = sorted(option for option, path in required.items() if path is None)
    if missing:
        raise ValueError(
            "--label-free-inputs requires the frozen 337/84/105 identity sidecars; "
            f"missing {', '.join(missing)}"
        )
    assert train_commitment_path is not None
    assert dev_inputs_path is not None
    assert dev_commitment_path is not None
    assert test_identity_inputs_path is not None
    assert test_identity_commitment_path is not None
    train, train_sha256, train_commitment_sha256 = _load_bound_pose_inputs(
        train_inputs_path,
        train_commitment_path,
        expected_split="train",
    )
    dev, dev_sha256, dev_commitment_sha256 = _load_bound_pose_inputs(
        dev_inputs_path,
        dev_commitment_path,
        expected_split="dev",
    )
    test, test_sha256, test_commitment_sha256 = _load_bound_pose_inputs(
        test_identity_inputs_path,
        test_identity_commitment_path,
        expected_split="test",
    )
    inputs = LabelFreeProtocolInputs(
        protocol=train.protocol,
        train=train,
        dev=dev,
        test=test,
    )
    artifacts = {
        "input_train_pose_inputs": train_inputs_path,
        "input_train_pose_input_commitment": train_commitment_path,
        "input_dev_pose_inputs": dev_inputs_path,
        "input_dev_pose_input_commitment": dev_commitment_path,
        "input_test_identity_pose_inputs": test_identity_inputs_path,
        "input_test_identity_pose_input_commitment": test_identity_commitment_path,
    }
    artifact_sha256 = {
        "input_train_pose_inputs": train_sha256,
        "input_train_pose_input_commitment": train_commitment_sha256,
        "input_dev_pose_inputs": dev_sha256,
        "input_dev_pose_input_commitment": dev_commitment_sha256,
        "input_test_identity_pose_inputs": test_sha256,
        "input_test_identity_pose_input_commitment": test_commitment_sha256,
    }
    return _LabelFreeInputBundle(
        inputs=inputs,
        artifacts=artifacts,
        artifact_sha256=artifact_sha256,
    )


def _validate_experiment_split(manifest: UCFRepManifest) -> None:
    """Validate official or frozen train/dev/test sizes without relabelling rows."""

    assert_split_disjoint(manifest.records)
    counts = manifest.split_counts
    permitted = {
        "ucfrep_526": (
            {"train": 421, "test": 105},
            {"train": 337, "dev": 84, "test": 105},
        ),
        "ucfrep_pose_110": (
            {"train": 89, "test": 21},
            {"train": 71, "dev": 18, "test": 21},
        ),
    }
    if counts not in permitted[manifest.protocol]:
        raise ValueError(
            f"{manifest.protocol} has invalid experiment split counts {counts}; "
            f"expected one of {permitted[manifest.protocol]}"
        )
    validate_canonical_split_membership(manifest)


def _assert_protocol_match(
    config: PAMSConfig,
    manifest: UCFRepManifest | LabelFreeProtocolInputs,
) -> None:
    if config.protocol != manifest.protocol:
        raise ValueError(
            f"config/manifest protocol mismatch: {config.protocol!r} != {manifest.protocol!r}"
        )


def _training_video_ids(
    manifest: UCFRepManifest | LabelFreeProtocolInputs,
    *,
    include_dev: bool,
) -> tuple[str, ...]:
    records = manifest.training_records(include_dev=include_dev)
    return tuple(sorted(record.video_id for record in records))


def _validate_final_training_pool(
    manifest: UCFRepManifest | LabelFreeProtocolInputs,
    *,
    include_dev: bool,
) -> None:
    identifiers = _training_video_ids(manifest, include_dev=include_dev)
    if manifest.protocol == "ucfrep_526" and len(identifiers) != 421:
        raise ValueError(
            "sealed ucfrep_526 evaluation requires a checkpoint trained on "
            "all 421 official training videos; use --include-dev with a "
            "337/84/105 manifest"
        )


def _make_checkpoint_provenance(
    manifest: UCFRepManifest | LabelFreeProtocolInputs,
    config: PAMSConfig,
    *,
    include_dev: bool,
    pose_cache_set_sha256: str,
    source_git_sha: str,
    container_image_id: str | None,
    container_environment_sha256: str | None,
    upstream_encoder_checkpoint_sha256: str | None = None,
) -> Any:
    from pams.training import CheckpointProvenance

    return CheckpointProvenance(
        protocol=manifest.protocol,
        dataset_fingerprint=manifest.training_fingerprint(include_dev=include_dev),
        training_video_ids=_training_video_ids(manifest, include_dev=include_dev),
        pose_fingerprint=config.pose_fingerprint,
        pose_cache_set_sha256=pose_cache_set_sha256,
        source_git_sha=source_git_sha,
        container_image_id=container_image_id,
        container_environment_sha256=container_environment_sha256,
        upstream_encoder_checkpoint_sha256=upstream_encoder_checkpoint_sha256,
    )


def _container_checkpoint_identity(
    source_git_sha: str,
    *,
    required: bool = False,
) -> tuple[str | None, str | None]:
    image_id = os.environ.get("PAMS_CONTAINER_IMAGE_ID", "").strip()
    environment_sha256 = os.environ.get(
        "PAMS_CONTAINER_ENVIRONMENT_SHA256",
        "",
    ).strip()
    container_revision = os.environ.get("PAMS_CONTAINER_SOURCE_REVISION", "").strip()
    supplied = (image_id, environment_sha256, container_revision)
    audit_mode = os.environ.get("PAMS_AUDIT_MODE", "").strip().lower()
    if (required or audit_mode in {"formal", "sealed"}) and not all(supplied):
        raise RuntimeError(
            "formal checkpoint provenance requires container image ID, "
            "environment SHA-256, and source revision"
        )
    if not any(supplied):
        return None, None
    if not all(supplied):
        raise RuntimeError("container checkpoint provenance environment is incomplete")
    if container_revision != source_git_sha:
        raise RuntimeError("container source revision does not match the clean Git checkout")
    return image_id, environment_sha256


def _sha256_file(path: Path) -> str:
    from pams.reproducibility import sha256_file

    return sha256_file(path)


def _validate_cli_checkpoint_destination(path: Path, *, resume: bool) -> None:
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
    elif path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {path}")


def _validate_cli_progress_destination(path: Path, *, resume: bool) -> None:
    if resume:
        if path.exists() and not path.is_file():
            raise ValueError(f"progress log is not a regular file: {path}")
    elif path.exists():
        raise FileExistsError(f"refusing to overwrite existing progress log: {path}")


def _cached_sequences(
    manifest: UCFRepManifest | LabelFreeProtocolInputs,
    *,
    split: str,
    cache_dir: Path,
    pose_fingerprint: str,
) -> tuple[tuple[Any, ...], list[str], PoseCacheSetSnapshot]:
    records = manifest.records_for(split)
    if not records:
        raise ValueError(f"manifest contains no {split!r} records")
    for record in records:
        if record.video_sha256 is None:
            raise ValueError(f"cached evaluation requires video_sha256 for {record.video_id!r}")
    unlabeled = tuple(
        UnlabeledVideoRecord(
            video_id=record.video_id,
            video_path=record.video_path,
            video_sha256=record.video_sha256,
        )
        for record in records
    )
    sequences, snapshot = load_pose_cache_set(
        unlabeled,
        cache_dir=cache_dir,
        pose_fingerprint=pose_fingerprint,
    )
    return sequences, [record.video_id for record in records], snapshot


def _training_pose_cache_snapshot(
    manifest: UCFRepManifest | LabelFreeProtocolInputs,
    *,
    include_dev: bool,
    cache_dir: Path,
    pose_fingerprint: str,
) -> PoseCacheSetSnapshot:
    records = manifest.training_records(include_dev=include_dev)
    dataset = TrainingPoseDataset(
        records,
        cache_dir=cache_dir,
        pose_fingerprint=pose_fingerprint,
    )
    return dataset.cache_snapshot()


def _persist_pose_cache_snapshot(
    path: Path,
    snapshot: PoseCacheSetSnapshot,
    *,
    resume: bool = False,
    overwrite: bool = False,
) -> str:
    """Write once, or validate the exact existing snapshot during resume."""

    if resume and overwrite:
        raise ValueError("pose-cache snapshot resume and overwrite are mutually exclusive")
    payload = snapshot.to_dict()
    if resume:
        if not path.is_file():
            raise FileNotFoundError(f"resume requires pose-cache snapshot: {path}")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("pose-cache snapshot differs from the resumed run")
    else:
        _write_json_exclusive(path, payload, overwrite=overwrite)
    return _sha256_file(path)


def _create_cli_run_manifest(
    *,
    output_dir: Path,
    command: list[str],
    config: PAMSConfig,
    dataset_sha256: str,
    notes: list[str] | None = None,
) -> Path:
    from pams.run_manifest import create_run_manifest, write_manifest_exclusive

    manifest = create_run_manifest(
        command=command,
        config_sha256=config.fingerprint,
        dataset_sha256=dataset_sha256,
        seed=config.seed,
        protocol=config.protocol,
        cwd=Path.cwd(),
        notes=notes,
    )
    return write_manifest_exclusive(
        manifest,
        output_dir / "manifests" / f"{manifest.run_id}.started.json",
    )


def _complete_cli_run_manifest(
    started_manifest_path: Path,
    *,
    artifacts: dict[str, Path],
    expected_artifact_sha256: dict[str, str],
    metrics: dict[str, Any],
) -> tuple[Path, str]:
    """Write the immutable terminal receipt and return its path and hash."""

    from pams.run_manifest import create_completed_receipt, write_manifest_exclusive

    portable_artifacts = _materialize_portable_receipt_artifacts(
        started_manifest_path,
        artifacts=artifacts,
        expected_artifact_sha256=expected_artifact_sha256,
    )
    completed = create_completed_receipt(
        started_manifest_path,
        artifacts=portable_artifacts,
        expected_artifact_sha256=expected_artifact_sha256,
        metrics=metrics,
    )
    completed_path = write_manifest_exclusive(
        completed,
        started_manifest_path.parent / f"{completed.run_id}.completed.json",
    )
    return completed_path, _sha256_file(completed_path)


def _device_or_none(device: str) -> str | None:
    return None if device.strip().lower() == "auto" else device


def _reserve_cli_sealed_attempt(
    registry_dir: Path,
    *,
    protocol: str,
    full_dataset_sha256: str,
    config_sha256: str,
    method_id: str,
    experiment_seed: int,
    input_artifact_role: Literal["checkpoint", "predictions"],
    input_artifact_sha256: str,
) -> tuple[Path, str]:
    """Consume one preregistered test attempt before labels are evaluated."""

    from pams.reproducibility import clean_git_revision
    from pams.sealed import reserve_sealed_test_attempt

    if experiment_seed not in {42, 2026, 3407}:
        raise ValueError(
            "sealed UCFRep evaluation requires a preregistered seed: 42, 2026, or 3407"
        )
    revision = clean_git_revision(Path.cwd())
    _, receipt_path = reserve_sealed_test_attempt(
        registry_dir,
        protocol=protocol,
        full_dataset_sha256=full_dataset_sha256,
        config_sha256=config_sha256,
        method_id=method_id,
        experiment_seed=experiment_seed,
        input_artifact_role=input_artifact_role,
        input_artifact_sha256=input_artifact_sha256,
        git_sha=revision,
    )
    return receipt_path, _sha256_file(receipt_path)


def _canonical_sealed_registry(
    requested: Path | None,
    *,
    protocol: str,
) -> Path:
    """Resolve the single registry fixed by the formal container."""

    run_root_value = os.environ.get("PAMS_RUN_ROOT", "").strip()
    if not run_root_value:
        raise ValueError(
            "PAMS_RUN_ROOT is required for sealed test evaluation so the "
            "attempt registry cannot be changed between invocations"
        )
    run_root = Path(run_root_value).expanduser().resolve(strict=False)
    expected = (run_root / "sealed-test-attempts" / protocol).resolve(strict=False)
    if requested is not None and requested.expanduser().resolve(strict=False) != expected:
        raise ValueError(
            f"--sealed-attempt-registry must equal the canonical shared path {expected}"
        )
    return expected


def _load_checkpoint_model(
    path: Path,
    config: PAMSConfig,
    *,
    expected_stage: Literal["encoder", "sshead"] | None = None,
    expected_provenance: Any = None,
    device: str = "auto",
) -> Any:
    """Load a training checkpoint without constructing optimizer state."""

    from pams.training import load_model_checkpoint

    return load_model_checkpoint(
        path,
        config,
        device=_device_or_none(device),
        expected_stage=expected_stage,
        expected_provenance=expected_provenance,
    )


@config_app.command("validate")
def config_validate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Strictly validate YAML and print its canonical fingerprint."""

    try:
        config = load_config(path)
    except (OSError, ValueError, ValidationError) as exc:
        _abort(str(exc))
    _emit(
        {
            "valid": True,
            "schema_version": config.schema_version,
            "protocol": config.protocol,
            "seed": config.seed,
            "fingerprint": config.fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
        }
    )


@data_app.command("validate")
@data_app.command("validate-manifest", hidden=True)
def data_validate(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    protocol: Annotated[str | None, typer.Option("--protocol")] = None,
    exact: Annotated[bool, typer.Option("--exact/--allow-partial")] = True,
) -> None:
    """Validate a UCFRep CSV/JSON manifest and split invariants."""

    try:
        manifest = _load_manifest(path, protocol=protocol, exact=exact)
        if not exact:
            assert_split_disjoint(manifest.records)
    except (OSError, ValueError) as exc:
        _abort(str(exc))
    _emit(
        {
            "valid": True,
            "protocol": manifest.protocol,
            "records": len(manifest.records),
            "split_counts": manifest.split_counts,
            "fingerprint": manifest.fingerprint,
            "exact_official_split": exact,
        }
    )


@data_app.command("prepare-ucfrep")
def data_prepare_ucfrep(
    video_root: Annotated[
        Path,
        typer.Argument(
            file_okay=False,
            help="Directory containing the separately obtained UCF101 AVI files.",
        ),
    ],
    output: Annotated[Path, typer.Option("--output", "-o", dir_okay=False)],
    annotation_archive: Annotated[
        Path,
        typer.Option(
            "--annotations",
            dir_okay=False,
            help="Cached official annotation ZIP; downloaded if absent.",
        ),
    ] = Path("data/annotations/ucf526_annotations.zip"),
    split_dir: Annotated[
        Path,
        typer.Option(
            "--split-dir",
            file_okay=False,
            help="Destination for label-free 421/105 and 337/84 ID lists.",
        ),
    ] = Path("data/splits"),
    hash_existing_videos: Annotated[
        bool,
        typer.Option(
            "--hash-videos/--no-hash-videos",
            help="Include SHA-256 for each resolved video (enabled by default).",
        ),
    ] = True,
    annotation_only: Annotated[
        bool,
        typer.Option(
            "--annotation-only",
            help=(
                "Explicitly allow missing videos. This is for annotation inspection "
                "only and is not a training-ready manifest."
            ),
        ),
    ] = False,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Build a strict, uniquely resolved 421/105 UCFRep manifest."""

    try:
        from pams.ucfrep import (
            OFFICIAL_ANNOTATION_SHA256,
            build_official_manifest,
            download_official_annotations,
            materialize_standard_split_ids,
        )

        if output.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing file: {output}")
        archive = download_official_annotations(annotation_archive)
        manifest = build_official_manifest(
            archive,
            video_root=video_root,
            hash_existing_videos=hash_existing_videos,
            allow_missing_videos=annotation_only,
        )
        save_ucfrep_manifest(manifest, output)
        split_paths = materialize_standard_split_ids(
            manifest,
            split_dir,
            seed=2026,
            overwrite=overwrite,
        )
    except (OSError, ValueError) as exc:
        _abort(str(exc))
    _emit(
        {
            "output": str(output.resolve()),
            "protocol": manifest.protocol,
            "split_counts": manifest.split_counts,
            "manifest_fingerprint": manifest.fingerprint,
            "annotation_sha256": OFFICIAL_ANNOTATION_SHA256,
            "strict_video_files": not annotation_only,
            "annotation_only": annotation_only,
            "video_hashes_included": hash_existing_videos,
            "split_id_files": {name: str(path.resolve()) for name, path in split_paths.items()},
        }
    )


@data_app.command("pose-inputs")
def data_pose_inputs(
    manifest_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o", dir_okay=False)],
    split: Annotated[str, typer.Option("--split")] = "test",
    video_root: Annotated[
        Path | None,
        typer.Option(
            "--video-root",
            exists=True,
            file_okay=False,
            readable=True,
            help="Root used to encode every source video as a portable relative locator.",
        ),
    ] = None,
    commitment_output: Annotated[
        Path | None,
        typer.Option(
            "--commitment-output",
            dir_okay=False,
            help="Independent label-free receipt; defaults beside --output.",
        ),
    ] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Compile one exact count-free/action-field-free pose input manifest."""

    try:
        if output.suffix.lower() != ".json":
            raise ValueError("pose-input output path must end in .json")
        commitment_path = commitment_output or output.with_name(f"{output.stem}.commitment.json")
        if commitment_path.suffix.lower() != ".json":
            raise ValueError("pose-input commitment output path must end in .json")
        resolved_manifest_path = manifest_path.resolve(strict=True)
        resolved_output = output.resolve(strict=False)
        resolved_commitment = commitment_path.resolve(strict=False)
        if resolved_output == resolved_manifest_path or (
            output.exists() and os.path.samefile(output, manifest_path)
        ):
            raise ValueError("pose-input output must not overwrite the source manifest")
        if resolved_commitment in {resolved_manifest_path, resolved_output} or (
            commitment_path.exists()
            and (
                os.path.samefile(commitment_path, manifest_path)
                or (output.exists() and os.path.samefile(commitment_path, output))
            )
        ):
            raise ValueError(
                "pose-input commitment must be distinct from the source manifest and sidecar"
            )
        if output.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing file: {output}")
        if commitment_path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing file: {commitment_path}")
        source_manifest_file_sha256 = _sha256_file(manifest_path)
        manifest = load_ucfrep_manifest(manifest_path, validate_exact=False)
        _validate_experiment_split(manifest)
        selected_split = split.strip().lower()
        if selected_split not in {"train", "dev", "test"}:
            raise ValueError("pose-input split must be train, dev, or test")
        records = manifest.records_for(selected_split)
        if not records:
            raise ValueError(f"manifest contains no {selected_split!r} records")
        source_root = manifest_path.resolve().parent
        resolved_video_root = (
            video_root.resolve(strict=True) if video_root is not None else source_root
        )
        unlabeled_records: list[UnlabeledVideoRecord] = []
        for record in records:
            if record.video_sha256 is None:
                raise ValueError(f"pose input {record.video_id!r} is missing source-video SHA-256")
            raw_video_path = Path(record.video_path)
            resolved_video_path = (
                raw_video_path if raw_video_path.is_absolute() else source_root / raw_video_path
            ).resolve(strict=True)
            try:
                portable_path = resolved_video_path.relative_to(resolved_video_root).as_posix()
            except ValueError:
                raise ValueError(
                    f"pose input {record.video_id!r} is outside --video-root; "
                    "choose a common portable video root"
                ) from None
            if resolved_output == resolved_video_path or (
                output.exists() and os.path.samefile(output, resolved_video_path)
            ):
                raise ValueError(f"pose-input output aliases selected video {record.video_id!r}")
            if resolved_commitment == resolved_video_path or (
                commitment_path.exists() and os.path.samefile(commitment_path, resolved_video_path)
            ):
                raise ValueError(
                    f"pose-input commitment aliases selected video {record.video_id!r}"
                )
            unlabeled_records.append(
                UnlabeledVideoRecord(
                    video_id=record.video_id,
                    video_path=portable_path,
                    video_sha256=record.video_sha256,
                )
            )
        pose_inputs = PoseInputManifest(
            protocol=manifest.protocol,
            split=selected_split,
            records=tuple(unlabeled_records),
        )
        pose_inputs.validate_exact_membership()
        _write_json_exclusive(
            output,
            pose_inputs.to_dict(),
            overwrite=overwrite,
        )
        persisted = load_pose_input_manifest(output, validate_exact=True)
        if persisted != pose_inputs:
            raise RuntimeError("persisted pose-input manifest changed during serialization")
        output_sha256 = _sha256_file(output)
        commitment = PoseInputCommitment(
            protocol=pose_inputs.protocol,
            split=pose_inputs.split,
            record_total=len(pose_inputs.records),
            identity_sha256=pose_input_identity_sha256(pose_inputs.records),
            sidecar_sha256=output_sha256,
            sidecar_fingerprint=pose_inputs.fingerprint,
        )
        _write_json_exclusive(
            commitment_path,
            commitment.to_dict(),
            overwrite=overwrite,
        )
        persisted_commitment = load_pose_input_commitment(commitment_path)
        if persisted_commitment != commitment:
            raise RuntimeError("persisted pose-input commitment changed during serialization")
        if _sha256_file(manifest_path) != source_manifest_file_sha256:
            raise RuntimeError("source dataset manifest changed while pose inputs were compiled")
        commitment_sha256 = _sha256_file(commitment_path)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(
        {
            "output": str(output.resolve()),
            "output_sha256": output_sha256,
            "commitment": str(commitment_path.resolve()),
            "commitment_sha256": commitment_sha256,
            "identity_sha256": commitment.identity_sha256,
            "manifest_type": pose_inputs.manifest_type,
            "protocol": pose_inputs.protocol,
            "split": pose_inputs.split,
            "records": len(pose_inputs.records),
            "fingerprint": pose_inputs.fingerprint,
            "contains_count_field": False,
            "contains_action_field": False,
        }
    )


@data_app.command("dev-targets")
def data_dev_targets(
    manifest_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o", dir_okay=False)],
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Privileged migration: isolate only the frozen 84 dev labels."""

    try:
        if output.suffix.lower() != ".json":
            raise ValueError("dev-target output path must end in .json")
        if output.resolve(strict=False) == manifest_path.resolve(strict=True):
            raise ValueError("dev-target output must not overwrite its source manifest")
        source_sha256 = _sha256_file(manifest_path)
        manifest = load_ucfrep_manifest(manifest_path, validate_exact=False)
        _validate_experiment_split(manifest)
        if manifest.protocol != "ucfrep_526" or manifest.split_counts != {
            "train": 337,
            "dev": 84,
            "test": 105,
        }:
            raise ValueError(
                "dev-target migration requires the frozen ucfrep_526 337/84/105 manifest"
            )
        dev_records = manifest.records_for("dev")
        targets = DevTargetManifest(
            protocol=manifest.protocol,
            records=tuple(
                DevTargetRecord(
                    video_id=record.video_id,
                    action=record.action,
                    count=record.count,
                )
                for record in dev_records
            ),
            source_annotation_sha256=manifest.sealed_dataset_fingerprint,
        )
        _write_json_exclusive(output, targets.to_dict(), overwrite=overwrite)
        persisted = load_dev_target_manifest(output)
        if persisted != targets:
            raise RuntimeError("persisted dev-target manifest changed during serialization")
        if _sha256_file(manifest_path) != source_sha256:
            raise RuntimeError("source dataset manifest changed during dev-target migration")
        output_sha256 = _sha256_file(output)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(
        {
            "output": str(output.resolve()),
            "output_sha256": output_sha256,
            "protocol": targets.protocol,
            "split": targets.split,
            "records": len(targets.records),
            "fingerprint": targets.fingerprint,
            "source_annotation_sha256": targets.source_annotation_sha256,
            "contains_test_targets": False,
        }
    )


@data_app.command("validate-run")
def data_validate_run(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    artifact_remap: Annotated[
        list[str] | None,
        typer.Option(
            "--artifact-remap",
            metavar="ROLE=ABSOLUTE_PATH",
            help=(
                "Override one receipt artifact locator by role. Repeat for multiple "
                "migrated artifacts; paths must be absolute."
            ),
        ),
    ] = None,
) -> None:
    """Validate an immutable started or completed run receipt."""

    try:
        from pams.run_manifest import (
            CompletedRunReceipt,
            RunManifest,
            resolve_artifact_path,
            validate_artifact_receipt,
        )

        raw = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_run_receipt_fields,
        )
        if not isinstance(raw, dict):
            raise ValueError("run receipt JSON root must be an object")
        remaps: dict[str, Path] = {}
        for specification in artifact_remap or ():
            role, separator, raw_path = specification.partition("=")
            normalized_role = role.strip()
            normalized_path = raw_path.strip()
            if not separator or not normalized_role or not normalized_path:
                raise ValueError("artifact remaps must use the form ROLE=ABSOLUTE_PATH")
            if normalized_role in remaps:
                raise ValueError(f"duplicate artifact remap for role {normalized_role!r}")
            candidate = Path(normalized_path).expanduser()
            if not candidate.is_absolute():
                raise ValueError(
                    f"artifact remap for role {normalized_role!r} must use an absolute path"
                )
            remaps[normalized_role] = candidate
        receipt_type = raw.get("receipt_type", "started")
        if receipt_type == "started":
            if remaps:
                raise ValueError("artifact remaps are only valid for completed receipts")
            receipt: RunManifest | CompletedRunReceipt = RunManifest.model_validate(raw)
        elif receipt_type == "completed":
            receipt = CompletedRunReceipt.model_validate(raw)
            sibling_start = path.with_name(f"{receipt.run_id}.started.json")
            if not sibling_start.is_file():
                raise FileNotFoundError(
                    f"completed receipt is missing sibling started receipt: {sibling_start}"
                )
            if _sha256_file(sibling_start) != receipt.start_manifest_sha256:
                raise ValueError("completed receipt does not match its sibling started receipt")
            sibling = RunManifest.model_validate_json(sibling_start.read_text(encoding="utf-8"))
            if sibling != receipt.started:
                raise ValueError("embedded and sibling started receipts differ")
            artifact_roles = {artifact.role for artifact in receipt.artifacts}
            unknown_remaps = sorted(set(remaps) - artifact_roles)
            if unknown_remaps:
                raise ValueError(
                    f"artifact remap roles are not present in the receipt: {unknown_remaps}"
                )
            for artifact in receipt.artifacts:
                artifact_path = resolve_artifact_path(
                    path,
                    artifact,
                    remapped_path=remaps.get(artifact.role),
                )
                validate_artifact_receipt(artifact_path, artifact)
        else:
            raise ValueError(f"unknown run receipt_type: {receipt_type!r}")
    except (OSError, ValidationError, ValueError) as exc:
        _abort(str(exc))
    _emit(
        {
            "valid": True,
            "receipt_type": receipt.receipt_type,
            "run_id": receipt.run_id,
            "fingerprint": receipt.fingerprint,
        }
    )


@data_app.command("split")
def data_split(
    path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path, typer.Option("--output", "-o", dir_okay=False)],
    seed: Annotated[int, typer.Option("--seed")] = 2026,
    protocol: Annotated[str | None, typer.Option("--protocol")] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Create the frozen 337/84 or 71/18 development split."""

    try:
        manifest = _load_manifest(path, protocol=protocol, exact=True)
        if seed != 2026:
            raise ValueError(
                "the preregistered development split is frozen at seed 2026; "
                "noncanonical seeds are not accepted by the experiment CLI"
            )
        if output.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing file: {output}")
        official_train = manifest.records_for("train")
        if manifest.protocol == "ucfrep_526":
            train, dev = split_ucfrep_train_dev(official_train, seed=seed)
        else:
            train, dev = split_ucfrep_pose_train_dev(official_train, seed=seed)
        split_manifest = UCFRepManifest(
            protocol=manifest.protocol,
            records=(*train, *dev, *manifest.records_for("test")),
        )
        _validate_experiment_split(split_manifest)
        save_ucfrep_manifest(split_manifest, output)
    except (OSError, ValueError) as exc:
        _abort(str(exc))
    _emit(
        {
            "output": str(output.resolve()),
            "protocol": split_manifest.protocol,
            "seed": seed,
            "split_counts": split_manifest.split_counts,
            "fingerprint": split_manifest.fingerprint,
        }
    )


@pose_app.command("extract")
def pose_extract(
    manifest_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    cache_dir: Annotated[Path, typer.Argument(file_okay=False)],
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/pams.yaml"
    ),
    split: Annotated[str, typer.Option("--split")] = "all",
    label_free_manifest: Annotated[
        bool,
        typer.Option(
            "--label-free-manifest",
            help=(
                "Require a pams data pose-inputs JSON file; the pose process then "
                "never parses count or action fields."
            ),
        ),
    ] = False,
    video_root: Annotated[
        Path | None,
        typer.Option(
            "--video-root",
            exists=True,
            file_okay=False,
            readable=True,
            help="Remap portable label-free video locators beneath this root.",
        ),
    ] = None,
    input_commitment: Annotated[
        Path | None,
        typer.Option(
            "--input-commitment",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Independent pose-input receipt; defaults beside the sidecar.",
        ),
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", min=1)] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    skip_existing: Annotated[
        bool,
        typer.Option(
            "--skip-existing",
            help=(
                "Resume by skipping only caches whose video hash and pose "
                "fingerprint match exactly."
            ),
        ),
    ] = False,
    failure_ledger: Annotated[
        Path | None,
        typer.Option(
            "--failure-ledger",
            dir_okay=False,
            help="JSON failure ledger; defaults to CACHE_DIR/failures.json.",
        ),
    ] = None,
) -> None:
    """Extract pose caches for a manifest, loading MediaPipe only on demand."""

    try:
        from pams.pose import PoseExtractorConfig, extract_many_with_failures

        config = load_config(config_path)
        if overwrite and skip_existing:
            raise ValueError("--overwrite and --skip-existing are mutually exclusive")
        if label_free_manifest:
            pose_inputs = load_pose_input_manifest(manifest_path, validate_exact=True)
            commitment_path = input_commitment or manifest_path.with_name(
                f"{manifest_path.stem}.commitment.json"
            )
            commitment = load_pose_input_commitment(commitment_path)
            sidecar_sha256 = _sha256_file(manifest_path)
            input_file_sha256 = sidecar_sha256
            identity_sha256 = pose_input_identity_sha256(pose_inputs.records)
            if (
                commitment.protocol != pose_inputs.protocol
                or commitment.split != pose_inputs.split
                or commitment.record_total != len(pose_inputs.records)
                or commitment.identity_sha256 != identity_sha256
                or commitment.sidecar_sha256 != sidecar_sha256
                or commitment.sidecar_fingerprint != pose_inputs.fingerprint
            ):
                raise ValueError(
                    "pose-input commitment does not bind this exact label-free sidecar"
                )
            if config.protocol != pose_inputs.protocol:
                raise ValueError(
                    f"configuration protocol {config.protocol!r} does not match "
                    f"pose-input protocol {pose_inputs.protocol!r}"
                )
            requested_split = split.strip().lower()
            if requested_split not in {"all", pose_inputs.split}:
                raise ValueError(
                    f"pose-input manifest is scoped to {pose_inputs.split!r}, not {split!r}"
                )
            records: tuple[UnlabeledVideoRecord | Any, ...] = pose_inputs.records
            input_manifest_fingerprint = pose_inputs.fingerprint
            input_kind = "label_free_sidecar"
            input_split = pose_inputs.split
            sidecar_fingerprint: str | None = pose_inputs.fingerprint
            commitment_file_sha256: str | None = _sha256_file(commitment_path)
            commitment_fingerprint: str | None = commitment.fingerprint
        else:
            if video_root is not None or input_commitment is not None:
                raise ValueError(
                    "--video-root and --input-commitment require --label-free-manifest"
                )
            manifest = load_ucfrep_manifest(manifest_path, validate_exact=False)
            input_file_sha256 = _sha256_file(manifest_path)
            _assert_protocol_match(config, manifest)
            _validate_experiment_split(manifest)
            records = manifest.records if split == "all" else manifest.records_for(split)
            input_manifest_fingerprint = manifest.fingerprint
            input_kind = "labelled_manifest"
            input_split = split.strip().lower()
            if input_split == "all":
                input_split = "all"
            sidecar_sha256 = None
            sidecar_fingerprint = None
            commitment_file_sha256 = None
            commitment_fingerprint = None
        if limit is not None:
            records = records[:limit]
        if not records:
            raise ValueError(f"no records selected by split {split!r}")
        root = (
            video_root.resolve(strict=True)
            if label_free_manifest and video_root is not None
            else manifest_path.resolve().parent
        )
        video_rows: list[tuple[str, Path, str | None]] = []
        for record in records:
            locator = Path(record.video_path)
            resolved_video = (locator if locator.is_absolute() else root / locator).resolve(
                strict=True
            )
            if label_free_manifest:
                try:
                    resolved_video.relative_to(root)
                except ValueError:
                    raise ValueError(
                        f"pose-input locator for {record.video_id!r} escapes --video-root"
                    ) from None
            video_rows.append((record.video_id, resolved_video, record.video_sha256))
        videos = tuple(video_rows)
        summaries, failures = extract_many_with_failures(
            videos,
            cache_dir=cache_dir,
            pose_fingerprint=config.pose_fingerprint,
            extractor_config=PoseExtractorConfig(
                target_frames=config.data.frames,
                preprocessing_revision=config.pose.preprocessing_revision,
                model_id=config.pose.model_id,
                model_complexity=config.pose.model_complexity,
                smooth_landmarks=config.pose.smooth_landmarks,
                min_detection_confidence=config.pose.min_detection_confidence,
                min_tracking_confidence=config.pose.min_tracking_confidence,
                crop_to_detected_span=config.pose.crop_to_detected_span,
            ),
            overwrite=overwrite,
            skip_existing=skip_existing,
        )
        skipped = sum(summary.skipped for summary in summaries)
        successful_ids = {summary.video_id for summary in summaries}
        successful_records = tuple(
            record for record in records if record.video_id in successful_ids
        )
        if successful_records:
            _, successful_snapshot = load_pose_cache_set(
                successful_records,
                cache_dir=cache_dir,
                pose_fingerprint=config.pose_fingerprint,
                materialize_sequences=False,
            )
            successful_cache_snapshot: dict[str, Any] | None = successful_snapshot.to_dict()
        else:
            successful_cache_snapshot = None
        selected_identity_sha256 = pose_input_identity_sha256(
            tuple(
                UnlabeledVideoRecord(
                    video_id=record.video_id,
                    video_path=str(record.video_path),
                    video_sha256=record.video_sha256,
                )
                for record in records
            )
        )
        if _sha256_file(manifest_path) != input_file_sha256:
            raise RuntimeError("input manifest changed during pose extraction")
        if label_free_manifest and (_sha256_file(commitment_path) != commitment_file_sha256):
            raise RuntimeError("pose-input commitment changed during pose extraction")
        ledger_path = failure_ledger or cache_dir / "failures.json"
        _write_json_exclusive(
            ledger_path,
            {
                "schema_version": 2,
                "input_kind": input_kind,
                "protocol": config.protocol,
                "split": input_split,
                "input_file_sha256": input_file_sha256,
                "input_fingerprint": input_manifest_fingerprint,
                "sidecar_sha256": sidecar_sha256,
                "sidecar_fingerprint": sidecar_fingerprint,
                "commitment_file_sha256": commitment_file_sha256,
                "commitment_fingerprint": commitment_fingerprint,
                "identity_sha256": selected_identity_sha256,
                "pose_fingerprint": config.pose_fingerprint,
                "successful_cache_snapshot": successful_cache_snapshot,
                "selected": len(records),
                "completed": len(summaries),
                "extracted": len(summaries) - skipped,
                "skipped": skipped,
                "failed": len(failures),
                "caches": [summary.to_dict() for summary in summaries],
                "failures": [failure.to_dict() for failure in failures],
            },
            # A resume rewrites only the ledger. Individual pose caches remain
            # immutable unless --overwrite was explicitly selected.
            overwrite=overwrite or skip_existing,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(
        {
            "pose_model": config.pose.model_id,
            "pose_fingerprint": config.pose_fingerprint,
            "label_free_manifest": label_free_manifest,
            "input_manifest_fingerprint": input_manifest_fingerprint,
            "identity_sha256": selected_identity_sha256,
            "successful_cache_snapshot": (
                None
                if successful_cache_snapshot is None
                else successful_cache_snapshot["fingerprint"]
            ),
            "selected": len(records),
            "completed": len(summaries),
            "extracted": len(summaries) - skipped,
            "skipped": skipped,
            "failed": len(failures),
            "complete": not failures,
            "failure_ledger": str(ledger_path.resolve()),
            "caches": [summary.to_dict() for summary in summaries],
        }
    )
    if failures:
        raise typer.Exit(1)


def _training_inputs(
    manifest_path: Path,
    cache_dir: Path,
    config: PAMSConfig,
    *,
    include_dev: bool,
) -> tuple[UCFRepManifest, tuple[Any, ...], PoseCacheSetSnapshot]:
    manifest = load_ucfrep_manifest(manifest_path, validate_exact=False)
    _assert_protocol_match(config, manifest)
    _validate_experiment_split(manifest)
    records = manifest.training_records(include_dev=include_dev)
    dataset = TrainingPoseDataset(
        records,
        cache_dir=cache_dir,
        pose_fingerprint=config.pose_fingerprint,
    )
    sequences, pose_snapshot = dataset.materialize_snapshot()
    return manifest, sequences, pose_snapshot


def _label_free_training_inputs(
    manifest_path: Path,
    cache_dir: Path,
    config: PAMSConfig,
    *,
    include_dev: bool,
    input_commitment: Path | None,
    dev_inputs: Path | None,
    dev_input_commitment: Path | None,
    test_identity_inputs: Path | None,
    test_identity_commitment: Path | None,
) -> tuple[
    LabelFreeProtocolInputs,
    tuple[Any, ...],
    PoseCacheSetSnapshot,
    _LabelFreeInputBundle,
]:
    bundle = _load_label_free_protocol_inputs(
        manifest_path,
        train_commitment_path=input_commitment,
        dev_inputs_path=dev_inputs,
        dev_commitment_path=dev_input_commitment,
        test_identity_inputs_path=test_identity_inputs,
        test_identity_commitment_path=test_identity_commitment,
    )
    _assert_protocol_match(config, bundle.inputs)
    dataset = TrainingPoseDataset(
        bundle.inputs.training_records(include_dev=include_dev),
        cache_dir=cache_dir,
        pose_fingerprint=config.pose_fingerprint,
    )
    sequences, pose_snapshot = dataset.materialize_snapshot()
    return bundle.inputs, sequences, pose_snapshot, bundle


@train_app.command("encoder")
def train_encoder_command(
    manifest_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    cache_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/pams.yaml"
    ),
    device: Annotated[str, typer.Option("--device")] = "auto",
    epochs: Annotated[int | None, typer.Option("--epochs", min=1)] = None,
    microbatch_size: Annotated[int | None, typer.Option("--microbatch-size", min=1)] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    resume_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--resume-checkpoint",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Immutable prior encoder checkpoint copied into this new output run.",
        ),
    ] = None,
    resume_progress: Annotated[
        Path | None,
        typer.Option(
            "--resume-progress",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Immutable prior encoder progress log copied into this new output run.",
        ),
    ] = None,
    include_dev: Annotated[bool, typer.Option("--include-dev")] = False,
    label_free_inputs: Annotated[
        bool,
        typer.Option(
            "--label-free-inputs",
            help="Treat MANIFEST_PATH as the strict count/action-free train sidecar.",
        ),
    ] = False,
    input_commitment: Annotated[
        Path | None,
        typer.Option("--input-commitment", exists=True, dir_okay=False, readable=True),
    ] = None,
    dev_inputs: Annotated[
        Path | None,
        typer.Option("--dev-inputs", exists=True, dir_okay=False, readable=True),
    ] = None,
    dev_input_commitment: Annotated[
        Path | None,
        typer.Option("--dev-input-commitment", exists=True, dir_okay=False, readable=True),
    ] = None,
    test_identity_inputs: Annotated[
        Path | None,
        typer.Option("--test-identity-inputs", exists=True, dir_okay=False, readable=True),
    ] = None,
    test_identity_commitment: Annotated[
        Path | None,
        typer.Option(
            "--test-identity-commitment",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
) -> None:
    """Train the PAMS encoder from label-free cached poses."""

    try:
        from pams.training import train_encoder

        _require_formal_label_free_inputs(
            label_free_inputs=label_free_inputs,
            operation="encoder training",
        )
        config_file_sha256 = _sha256_file(config_path)
        dataset_manifest_sha256 = _sha256_file(manifest_path)
        config = load_config(config_path)
        if resume and (resume_checkpoint is None or resume_progress is None):
            raise ValueError("--resume requires both --resume-checkpoint and --resume-progress")
        if not resume and (resume_checkpoint is not None or resume_progress is not None):
            raise ValueError("resume input options require --resume")
        label_free_bundle: _LabelFreeInputBundle | None = None
        manifest: UCFRepManifest | LabelFreeProtocolInputs
        if label_free_inputs:
            manifest, sequences, pose_snapshot, label_free_bundle = _label_free_training_inputs(
                manifest_path,
                cache_dir,
                config,
                include_dev=include_dev,
                input_commitment=input_commitment,
                dev_inputs=dev_inputs,
                dev_input_commitment=dev_input_commitment,
                test_identity_inputs=test_identity_inputs,
                test_identity_commitment=test_identity_commitment,
            )
        else:
            unexpected_label_free_options = {
                "--input-commitment": input_commitment,
                "--dev-inputs": dev_inputs,
                "--dev-input-commitment": dev_input_commitment,
                "--test-identity-inputs": test_identity_inputs,
                "--test-identity-commitment": test_identity_commitment,
            }
            supplied = sorted(
                option
                for option, value in unexpected_label_free_options.items()
                if value is not None
            )
            if supplied:
                raise ValueError(f"{', '.join(supplied)} require --label-free-inputs")
            manifest, sequences, pose_snapshot = _training_inputs(
                manifest_path, cache_dir, config, include_dev=include_dev
            )
        if _sha256_file(config_path) != config_file_sha256:
            raise RuntimeError("configuration changed while it was being loaded")
        if _sha256_file(manifest_path) != dataset_manifest_sha256:
            raise RuntimeError("dataset manifest changed while it was being loaded")
        if label_free_bundle is not None:
            for role, path in label_free_bundle.artifacts.items():
                if _sha256_file(path) != label_free_bundle.artifact_sha256[role]:
                    raise RuntimeError(
                        f"label-free protocol artifact changed while loading: {role}"
                    )
        from pams.reproducibility import clean_git_revision

        source_git_sha = clean_git_revision(Path.cwd())
        container_image_id, container_environment_sha256 = _container_checkpoint_identity(
            source_git_sha
        )
        provenance = _make_checkpoint_provenance(
            manifest,
            config,
            include_dev=include_dev,
            pose_cache_set_sha256=pose_snapshot.fingerprint,
            source_git_sha=source_git_sha,
            container_image_id=container_image_id,
            container_environment_sha256=container_environment_sha256,
        )
        checkpoint_path = output_dir / "encoder.pt"
        progress_path = output_dir / "logs" / "encoder.jsonl"
        pose_snapshot_path = output_dir / "inputs" / "training-pose-cache-snapshot.json"
        _validate_cli_checkpoint_destination(checkpoint_path, resume=False)
        _validate_cli_progress_destination(progress_path, resume=False)
        resume_checkpoint_sha256 = (
            None if resume_checkpoint is None else _sha256_file(resume_checkpoint)
        )
        resume_progress_sha256 = None if resume_progress is None else _sha256_file(resume_progress)
        durable_mkdir(output_dir)
        pose_snapshot_sha256 = _persist_pose_cache_snapshot(
            pose_snapshot_path,
            pose_snapshot,
            resume=False,
        )
        started_manifest_path = _create_cli_run_manifest(
            output_dir=output_dir,
            command=list(sys.argv),
            config=config,
            dataset_sha256=provenance.dataset_fingerprint,
            notes=[
                "count and action labels are not exposed to the training dataset",
                (
                    "formal 337/84/105 label-free identity sidecars"
                    if label_free_bundle is not None
                    else "legacy labeled-manifest compatibility path"
                ),
                f"pose cache set: {pose_snapshot.fingerprint}",
                (
                    "fresh run"
                    if not resume
                    else "resume lineage uses immutable copied checkpoint/progress inputs"
                ),
            ],
        )
        if resume:
            assert resume_checkpoint is not None
            assert resume_progress is not None
            assert resume_checkpoint_sha256 is not None
            assert resume_progress_sha256 is not None
            _copy_file_exclusive(
                resume_checkpoint,
                checkpoint_path,
                expected_sha256=resume_checkpoint_sha256,
            )
            _copy_file_exclusive(
                resume_progress,
                progress_path,
                expected_sha256=resume_progress_sha256,
            )
        result = train_encoder(
            sequences,
            config,
            device=_device_or_none(device),
            microbatch_size=microbatch_size,
            checkpoint_path=checkpoint_path,
            progress_path=progress_path,
            resume=resume,
            stop_after_epoch=epochs,
            provenance=provenance,
            allow_negative_shortfall=False,
        )
        if (
            resume_checkpoint is not None
            and _sha256_file(resume_checkpoint) != resume_checkpoint_sha256
        ):
            raise RuntimeError("resume encoder checkpoint changed during training")
        if resume_progress is not None and _sha256_file(resume_progress) != resume_progress_sha256:
            raise RuntimeError("resume encoder progress changed during training")
        final_epoch = asdict(result.history[-1]) if result.history else None
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        progress_sha256 = _sha256_file(progress_path)
        completion_artifacts = {
            "input_config": config_path,
            "input_dataset_manifest": manifest_path,
            "input_pose_cache_snapshot": pose_snapshot_path,
            "output_encoder_checkpoint": checkpoint_path,
            "progress_log": progress_path,
        }
        completion_expected_sha256 = {
            "input_config": config_file_sha256,
            "input_dataset_manifest": dataset_manifest_sha256,
            "input_pose_cache_snapshot": pose_snapshot_sha256,
            "output_encoder_checkpoint": checkpoint_sha256,
            "progress_log": progress_sha256,
        }
        if label_free_bundle is not None:
            completion_artifacts.update(label_free_bundle.artifacts)
            completion_expected_sha256.update(label_free_bundle.artifact_sha256)
        if resume_checkpoint is not None:
            assert resume_checkpoint_sha256 is not None
            completion_artifacts["input_resume_checkpoint"] = resume_checkpoint
            completion_expected_sha256["input_resume_checkpoint"] = resume_checkpoint_sha256
        if resume_progress is not None:
            assert resume_progress_sha256 is not None
            completion_artifacts["input_resume_progress"] = resume_progress
            completion_expected_sha256["input_resume_progress"] = resume_progress_sha256
        completed_manifest_path, completed_manifest_sha256 = _complete_cli_run_manifest(
            started_manifest_path,
            artifacts=completion_artifacts,
            expected_artifact_sha256=completion_expected_sha256,
            metrics={
                "completed_epochs": result.completed_epochs,
                "final_epoch": final_epoch,
            },
        )
    except ImportError as exc:
        _abort(f"encoder training module is unavailable: {exc}")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(
        {
            "status": "completed",
            "stage": "encoder",
            "completed_epochs": result.completed_epochs,
            "checkpoint_path": (
                None if result.checkpoint_path is None else str(result.checkpoint_path.resolve())
            ),
            "checkpoint_sha256": checkpoint_sha256,
            "progress_path": str(progress_path.resolve()),
            "progress_sha256": progress_sha256,
            "pose_cache_set_sha256": pose_snapshot.fingerprint,
            "pose_cache_snapshot_path": str(pose_snapshot_path.resolve()),
            "pose_cache_snapshot_sha256": pose_snapshot_sha256,
            "started_manifest_path": str(started_manifest_path.resolve()),
            "completed_manifest_path": str(completed_manifest_path.resolve()),
            "completed_manifest_sha256": completed_manifest_sha256,
            "cluster_assignments": dict(result.cluster_assignments),
            "history": [asdict(row) for row in result.history],
        }
    )


@train_app.command("sshead")
def train_sshead_command(
    encoder_checkpoint: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    manifest_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    cache_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    encoder_progress: Annotated[
        Path,
        typer.Option(
            "--encoder-progress",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Exact terminal progress log for the frozen upstream encoder.",
        ),
    ],
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/pams.yaml"
    ),
    device: Annotated[str, typer.Option("--device")] = "auto",
    epochs: Annotated[int | None, typer.Option("--epochs", min=1)] = None,
    microbatch_size: Annotated[int | None, typer.Option("--microbatch-size", min=1)] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    resume_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--resume-checkpoint",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Immutable prior SSHead checkpoint copied into this new output run.",
        ),
    ] = None,
    resume_progress: Annotated[
        Path | None,
        typer.Option(
            "--resume-progress",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Immutable prior SSHead progress log copied into this new output run.",
        ),
    ] = None,
    include_dev: Annotated[bool, typer.Option("--include-dev")] = False,
    label_free_inputs: Annotated[
        bool,
        typer.Option(
            "--label-free-inputs",
            help="Treat MANIFEST_PATH as the strict count/action-free train sidecar.",
        ),
    ] = False,
    input_commitment: Annotated[
        Path | None,
        typer.Option("--input-commitment", exists=True, dir_okay=False, readable=True),
    ] = None,
    dev_inputs: Annotated[
        Path | None,
        typer.Option("--dev-inputs", exists=True, dir_okay=False, readable=True),
    ] = None,
    dev_input_commitment: Annotated[
        Path | None,
        typer.Option("--dev-input-commitment", exists=True, dir_okay=False, readable=True),
    ] = None,
    test_identity_inputs: Annotated[
        Path | None,
        typer.Option("--test-identity-inputs", exists=True, dir_okay=False, readable=True),
    ] = None,
    test_identity_commitment: Annotated[
        Path | None,
        typer.Option(
            "--test-identity-commitment",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
) -> None:
    """Train the inferred self-supervised head with a frozen encoder."""

    try:
        from pams.training import train_sshead, validate_terminal_checkpoint

        _require_formal_label_free_inputs(
            label_free_inputs=label_free_inputs,
            operation="SSHead training",
        )
        config_file_sha256 = _sha256_file(config_path)
        dataset_manifest_sha256 = _sha256_file(manifest_path)
        config = load_config(config_path)
        if resume and (resume_checkpoint is None or resume_progress is None):
            raise ValueError("--resume requires both --resume-checkpoint and --resume-progress")
        if not resume and (resume_checkpoint is not None or resume_progress is not None):
            raise ValueError("resume input options require --resume")
        label_free_bundle: _LabelFreeInputBundle | None = None
        manifest: UCFRepManifest | LabelFreeProtocolInputs
        if label_free_inputs:
            manifest, sequences, pose_snapshot, label_free_bundle = _label_free_training_inputs(
                manifest_path,
                cache_dir,
                config,
                include_dev=include_dev,
                input_commitment=input_commitment,
                dev_inputs=dev_inputs,
                dev_input_commitment=dev_input_commitment,
                test_identity_inputs=test_identity_inputs,
                test_identity_commitment=test_identity_commitment,
            )
        else:
            unexpected_label_free_options = {
                "--input-commitment": input_commitment,
                "--dev-inputs": dev_inputs,
                "--dev-input-commitment": dev_input_commitment,
                "--test-identity-inputs": test_identity_inputs,
                "--test-identity-commitment": test_identity_commitment,
            }
            supplied = sorted(
                option
                for option, value in unexpected_label_free_options.items()
                if value is not None
            )
            if supplied:
                raise ValueError(f"{', '.join(supplied)} require --label-free-inputs")
            manifest, sequences, pose_snapshot = _training_inputs(
                manifest_path, cache_dir, config, include_dev=include_dev
            )
        if _sha256_file(config_path) != config_file_sha256:
            raise RuntimeError("configuration changed while it was being loaded")
        if _sha256_file(manifest_path) != dataset_manifest_sha256:
            raise RuntimeError("dataset manifest changed while it was being loaded")
        if label_free_bundle is not None:
            for role, path in label_free_bundle.artifacts.items():
                if _sha256_file(path) != label_free_bundle.artifact_sha256[role]:
                    raise RuntimeError(
                        f"label-free protocol artifact changed while loading: {role}"
                    )
        from pams.reproducibility import clean_git_revision

        source_git_sha = clean_git_revision(Path.cwd())
        container_image_id, container_environment_sha256 = _container_checkpoint_identity(
            source_git_sha
        )
        encoder_provenance = _make_checkpoint_provenance(
            manifest,
            config,
            include_dev=include_dev,
            pose_cache_set_sha256=pose_snapshot.fingerprint,
            source_git_sha=source_git_sha,
            container_image_id=container_image_id,
            container_environment_sha256=container_environment_sha256,
        )
        encoder_checkpoint_sha256 = _sha256_file(encoder_checkpoint)
        encoder_progress_sha256 = _sha256_file(encoder_progress)
        provenance = _make_checkpoint_provenance(
            manifest,
            config,
            include_dev=include_dev,
            pose_cache_set_sha256=pose_snapshot.fingerprint,
            source_git_sha=source_git_sha,
            container_image_id=container_image_id,
            container_environment_sha256=container_environment_sha256,
            upstream_encoder_checkpoint_sha256=encoder_checkpoint_sha256,
        )
        validate_terminal_checkpoint(
            encoder_checkpoint,
            config,
            expected_stage="encoder",
            expected_provenance=encoder_provenance,
            progress_path=encoder_progress,
        )
        if _sha256_file(encoder_checkpoint) != encoder_checkpoint_sha256:
            raise RuntimeError("encoder checkpoint changed during terminal validation")
        if _sha256_file(encoder_progress) != encoder_progress_sha256:
            raise RuntimeError("encoder progress log changed during terminal validation")
        checkpoint_path = output_dir / "sshead.pt"
        progress_path = output_dir / "logs" / "sshead.jsonl"
        pose_snapshot_path = output_dir / "inputs" / "training-pose-cache-snapshot.json"
        _validate_cli_checkpoint_destination(checkpoint_path, resume=False)
        _validate_cli_progress_destination(progress_path, resume=False)
        resume_checkpoint_sha256 = (
            None if resume_checkpoint is None else _sha256_file(resume_checkpoint)
        )
        resume_progress_sha256 = None if resume_progress is None else _sha256_file(resume_progress)
        durable_mkdir(output_dir)
        pose_snapshot_sha256 = _persist_pose_cache_snapshot(
            pose_snapshot_path,
            pose_snapshot,
            resume=False,
        )
        started_manifest_path = _create_cli_run_manifest(
            output_dir=output_dir,
            command=list(sys.argv),
            config=config,
            dataset_sha256=provenance.dataset_fingerprint,
            notes=[
                "PAMS-SSHead is an inferred reproduction completion, not author-disclosed",
                (
                    "formal 337/84/105 label-free identity sidecars"
                    if label_free_bundle is not None
                    else "legacy labeled-manifest compatibility path"
                ),
                f"pose cache set: {pose_snapshot.fingerprint}",
                (
                    "fresh run"
                    if not resume
                    else "resume lineage uses immutable copied checkpoint/progress inputs"
                ),
            ],
        )
        model = _load_checkpoint_model(
            encoder_checkpoint,
            config,
            expected_stage="encoder",
            expected_provenance=encoder_provenance,
            device=device,
        )
        if _sha256_file(encoder_checkpoint) != encoder_checkpoint_sha256:
            raise RuntimeError("encoder checkpoint changed while it was being loaded")
        if resume:
            assert resume_checkpoint is not None
            assert resume_progress is not None
            assert resume_checkpoint_sha256 is not None
            assert resume_progress_sha256 is not None
            _copy_file_exclusive(
                resume_checkpoint,
                checkpoint_path,
                expected_sha256=resume_checkpoint_sha256,
            )
            _copy_file_exclusive(
                resume_progress,
                progress_path,
                expected_sha256=resume_progress_sha256,
            )
        result = train_sshead(
            sequences,
            config,
            model=model,
            device=_device_or_none(device),
            microbatch_size=microbatch_size,
            checkpoint_path=checkpoint_path,
            progress_path=progress_path,
            resume=resume,
            stop_after_epoch=epochs,
            provenance=provenance,
        )
        if (
            resume_checkpoint is not None
            and _sha256_file(resume_checkpoint) != resume_checkpoint_sha256
        ):
            raise RuntimeError("resume SSHead checkpoint changed during training")
        if resume_progress is not None and _sha256_file(resume_progress) != resume_progress_sha256:
            raise RuntimeError("resume SSHead progress changed during training")
        final_epoch = asdict(result.history[-1]) if result.history else None
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        progress_sha256 = _sha256_file(progress_path)
        completion_artifacts = {
            "input_config": config_path,
            "input_dataset_manifest": manifest_path,
            "input_encoder_checkpoint": encoder_checkpoint,
            "input_encoder_progress": encoder_progress,
            "input_pose_cache_snapshot": pose_snapshot_path,
            "output_sshead_checkpoint": checkpoint_path,
            "progress_log": progress_path,
        }
        completion_expected_sha256 = {
            "input_config": config_file_sha256,
            "input_dataset_manifest": dataset_manifest_sha256,
            "input_encoder_checkpoint": encoder_checkpoint_sha256,
            "input_encoder_progress": encoder_progress_sha256,
            "input_pose_cache_snapshot": pose_snapshot_sha256,
            "output_sshead_checkpoint": checkpoint_sha256,
            "progress_log": progress_sha256,
        }
        if label_free_bundle is not None:
            completion_artifacts.update(label_free_bundle.artifacts)
            completion_expected_sha256.update(label_free_bundle.artifact_sha256)
        if resume_checkpoint is not None:
            assert resume_checkpoint_sha256 is not None
            completion_artifacts["input_resume_checkpoint"] = resume_checkpoint
            completion_expected_sha256["input_resume_checkpoint"] = resume_checkpoint_sha256
        if resume_progress is not None:
            assert resume_progress_sha256 is not None
            completion_artifacts["input_resume_progress"] = resume_progress
            completion_expected_sha256["input_resume_progress"] = resume_progress_sha256
        completed_manifest_path, completed_manifest_sha256 = _complete_cli_run_manifest(
            started_manifest_path,
            artifacts=completion_artifacts,
            expected_artifact_sha256=completion_expected_sha256,
            metrics={
                "completed_epochs": result.completed_epochs,
                "final_epoch": final_epoch,
            },
        )
    except ImportError as exc:
        _abort(f"SSHead training module is unavailable: {exc}")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(
        {
            "status": "completed",
            "stage": "sshead",
            "inferred_reproduction_completion": True,
            "completed_epochs": result.completed_epochs,
            "checkpoint_path": (
                None if result.checkpoint_path is None else str(result.checkpoint_path.resolve())
            ),
            "checkpoint_sha256": checkpoint_sha256,
            "upstream_encoder_checkpoint_sha256": encoder_checkpoint_sha256,
            "upstream_encoder_progress_sha256": encoder_progress_sha256,
            "progress_path": str(progress_path.resolve()),
            "progress_sha256": progress_sha256,
            "pose_cache_set_sha256": pose_snapshot.fingerprint,
            "pose_cache_snapshot_path": str(pose_snapshot_path.resolve()),
            "pose_cache_snapshot_sha256": pose_snapshot_sha256,
            "started_manifest_path": str(started_manifest_path.resolve()),
            "completed_manifest_path": str(completed_manifest_path.resolve()),
            "completed_manifest_sha256": completed_manifest_sha256,
            "history": [asdict(row) for row in result.history],
        }
    )


@evaluate_app.command("checkpoint")
def evaluate_checkpoint_command(
    checkpoint_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    manifest_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    cache_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    variant: Annotated[
        str,
        typer.Option(
            "--variant",
            help="literal requires an encoder checkpoint; sshead requires an SSHead checkpoint.",
        ),
    ],
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/pams.yaml"
    ),
    split: Annotated[str, typer.Option("--split")] = "test",
    device: Annotated[str, typer.Option("--device")] = "auto",
    include_dev: Annotated[
        bool,
        typer.Option(
            "--include-dev",
            help="Expect a checkpoint trained on the manifest's train and dev splits.",
        ),
    ] = False,
    checkpoint_progress: Annotated[
        Path | None,
        typer.Option(
            "--checkpoint-progress",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Terminal progress log for the evaluated checkpoint; required for test.",
        ),
    ] = None,
    upstream_encoder_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--upstream-encoder-checkpoint",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Required for sshead provenance validation; forbidden for literal.",
        ),
    ] = None,
    upstream_encoder_progress: Annotated[
        Path | None,
        typer.Option(
            "--upstream-encoder-progress",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Terminal encoder progress log; required for the SSHead variant.",
        ),
    ] = None,
    method_id: Annotated[
        str | None,
        typer.Option(
            "--method-id",
            help=(
                "Preregistered protocol-scoped experiment identity; "
                "required for a sealed test pass."
            ),
        ),
    ] = None,
    sealed_attempt_registry: Annotated[
        Path | None,
        typer.Option(
            "--sealed-attempt-registry",
            file_okay=False,
            help=(
                "Shared durable registry required for test evaluation; "
                "the method/seed attempt is consumed before targets are used."
            ),
        ),
    ] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
    label_free_inputs: Annotated[
        bool,
        typer.Option(
            "--label-free-inputs",
            help="Use strict 337/84/105 identity sidecars for dev evaluation.",
        ),
    ] = False,
    input_commitment: Annotated[
        Path | None,
        typer.Option("--input-commitment", exists=True, dir_okay=False, readable=True),
    ] = None,
    dev_inputs: Annotated[
        Path | None,
        typer.Option("--dev-inputs", exists=True, dir_okay=False, readable=True),
    ] = None,
    dev_input_commitment: Annotated[
        Path | None,
        typer.Option("--dev-input-commitment", exists=True, dir_okay=False, readable=True),
    ] = None,
    test_identity_inputs: Annotated[
        Path | None,
        typer.Option("--test-identity-inputs", exists=True, dir_okay=False, readable=True),
    ] = None,
    test_identity_commitment: Annotated[
        Path | None,
        typer.Option(
            "--test-identity-commitment",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    dev_targets: Annotated[
        Path | None,
        typer.Option(
            "--dev-targets",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Strict 84-row dev-only target file; required with --label-free-inputs.",
        ),
    ] = None,
) -> None:
    """Evaluate a stage-bound variant; labels are passed only to evaluation."""

    try:
        from pams.evaluation import evaluate_predictions, predict_sequences
        from pams.training import (
            validate_sshead_encoder_binding,
            validate_terminal_checkpoint,
        )

        evaluation_split = split.strip().lower()
        if evaluation_split not in {"dev", "test"}:
            raise ValueError("checkpoint evaluation split must be 'dev' or 'test'")
        if evaluation_split == "dev":
            _require_formal_label_free_inputs(
                label_free_inputs=label_free_inputs,
                operation="dev checkpoint evaluation",
            )
        if label_free_inputs and evaluation_split != "dev":
            raise ValueError(
                "--label-free-inputs is the dev-only firewall path; "
                "sealed test evaluation retains its dedicated label-unsealing protocol"
            )
        config_file_sha256 = _sha256_file(config_path)
        dataset_manifest_sha256 = _sha256_file(manifest_path)
        config = load_config(config_path)
        label_free_bundle: _LabelFreeInputBundle | None = None
        dev_target_manifest: DevTargetManifest | None = None
        dev_targets_sha256: str | None = None
        manifest: UCFRepManifest | LabelFreeProtocolInputs
        if label_free_inputs:
            label_free_bundle = _load_label_free_protocol_inputs(
                manifest_path,
                train_commitment_path=input_commitment,
                dev_inputs_path=dev_inputs,
                dev_commitment_path=dev_input_commitment,
                test_identity_inputs_path=test_identity_inputs,
                test_identity_commitment_path=test_identity_commitment,
            )
            manifest = label_free_bundle.inputs
            if dev_targets is None:
                raise ValueError("--dev-targets is required with --label-free-inputs")
            dev_targets_sha256 = _sha256_file(dev_targets)
            dev_target_manifest = load_dev_target_manifest(dev_targets)
            if dev_target_manifest.protocol != manifest.protocol:
                raise ValueError("dev-target/protocol identity mismatch")
            dev_ids = tuple(record.video_id for record in manifest.records_for("dev"))
            target_ids = tuple(record.video_id for record in dev_target_manifest.records)
            if target_ids != dev_ids:
                raise ValueError(
                    "dev-target order/identity does not exactly match the dev pose inputs"
                )
            if _sha256_file(dev_targets) != dev_targets_sha256:
                raise RuntimeError("dev-target manifest changed while it was being loaded")
        else:
            unexpected_label_free_options = {
                "--input-commitment": input_commitment,
                "--dev-inputs": dev_inputs,
                "--dev-input-commitment": dev_input_commitment,
                "--test-identity-inputs": test_identity_inputs,
                "--test-identity-commitment": test_identity_commitment,
                "--dev-targets": dev_targets,
            }
            supplied = sorted(
                option
                for option, value in unexpected_label_free_options.items()
                if value is not None
            )
            if supplied:
                raise ValueError(f"{', '.join(supplied)} require --label-free-inputs")
            manifest = load_ucfrep_manifest(manifest_path, validate_exact=False)
        if _sha256_file(config_path) != config_file_sha256:
            raise RuntimeError("configuration changed while it was being loaded")
        if _sha256_file(manifest_path) != dataset_manifest_sha256:
            raise RuntimeError("dataset manifest changed while it was being loaded")
        if label_free_bundle is not None:
            for role, path in label_free_bundle.artifacts.items():
                if _sha256_file(path) != label_free_bundle.artifact_sha256[role]:
                    raise RuntimeError(
                        f"label-free protocol artifact changed while loading: {role}"
                    )
        _assert_protocol_match(config, manifest)
        if not label_free_inputs:
            _validate_experiment_split(cast(UCFRepManifest, manifest))
        checkpoint_variant = _normalize_checkpoint_variant(variant)
        if evaluation_split == "test" and manifest.protocol != "ucfrep_526":
            raise ValueError(
                "sealed evaluation is disabled for ucfrep_pose_110 until its "
                "official 89/21 ID/count digest is frozen"
            )
        if checkpoint_variant == "literal":
            if upstream_encoder_checkpoint is not None:
                raise ValueError("--upstream-encoder-checkpoint is forbidden for --variant literal")
            if upstream_encoder_progress is not None:
                raise ValueError("--upstream-encoder-progress is forbidden for --variant literal")
            expected_stage: Literal["encoder", "sshead"] = "encoder"
            upstream_encoder_sha256 = None
            upstream_encoder_progress_sha256 = None
        else:
            if upstream_encoder_checkpoint is None:
                raise ValueError("--upstream-encoder-checkpoint is required for --variant sshead")
            if upstream_encoder_progress is None:
                raise ValueError("--upstream-encoder-progress is required for --variant sshead")
            expected_stage = "sshead"
            upstream_encoder_sha256 = _sha256_file(upstream_encoder_checkpoint)
            upstream_encoder_progress_sha256 = _sha256_file(upstream_encoder_progress)
        if evaluation_split == "test":
            if overwrite:
                raise ValueError("--overwrite is forbidden for sealed test evaluation")
            if checkpoint_progress is None:
                raise ValueError("sealed test evaluation requires --checkpoint-progress")
            _validate_final_training_pool(
                manifest,
                include_dev=include_dev,
            )
            if method_id is None:
                raise ValueError("sealed test checkpoint evaluation requires --method-id")
            sealed_method_id = _normalize_frozen_method_id(
                manifest.protocol,
                method_id,
                checkpoint_variant=checkpoint_variant,
            )
            _validate_frozen_pams_test_config(config)
        elif method_id is None:
            sealed_method_id = f"pams-{checkpoint_variant}"
        else:
            sealed_method_id = _normalize_frozen_method_id(
                manifest.protocol,
                method_id,
                checkpoint_variant=checkpoint_variant,
            )
        from pams.reproducibility import clean_git_revision

        source_git_sha = clean_git_revision(Path.cwd())
        container_image_id, container_environment_sha256 = _container_checkpoint_identity(
            source_git_sha,
            required=evaluation_split == "test",
        )
        training_pose_snapshot = _training_pose_cache_snapshot(
            manifest,
            include_dev=include_dev,
            cache_dir=cache_dir,
            pose_fingerprint=config.pose_fingerprint,
        )
        provenance = _make_checkpoint_provenance(
            manifest,
            config,
            include_dev=include_dev,
            pose_cache_set_sha256=training_pose_snapshot.fingerprint,
            source_git_sha=source_git_sha,
            container_image_id=container_image_id,
            container_environment_sha256=container_environment_sha256,
            upstream_encoder_checkpoint_sha256=upstream_encoder_sha256,
        )
        checkpoint_sha256 = _sha256_file(checkpoint_path)
        checkpoint_progress_sha256 = (
            None if checkpoint_progress is None else _sha256_file(checkpoint_progress)
        )
        encoder_provenance = (
            None
            if checkpoint_variant == "literal"
            else _make_checkpoint_provenance(
                manifest,
                config,
                include_dev=include_dev,
                pose_cache_set_sha256=training_pose_snapshot.fingerprint,
                source_git_sha=source_git_sha,
                container_image_id=container_image_id,
                container_environment_sha256=container_environment_sha256,
            )
        )
        if evaluation_split == "test":
            assert checkpoint_progress is not None
            validate_terminal_checkpoint(
                checkpoint_path,
                config,
                expected_stage=expected_stage,
                expected_provenance=provenance,
                progress_path=checkpoint_progress,
            )
            if checkpoint_variant == "sshead":
                assert upstream_encoder_checkpoint is not None
                assert upstream_encoder_progress is not None
                assert encoder_provenance is not None
                validate_terminal_checkpoint(
                    upstream_encoder_checkpoint,
                    config,
                    expected_stage="encoder",
                    expected_provenance=encoder_provenance,
                    progress_path=upstream_encoder_progress,
                )
                validate_sshead_encoder_binding(
                    checkpoint_path,
                    upstream_encoder_checkpoint,
                )
            if _sha256_file(checkpoint_path) != checkpoint_sha256:
                raise RuntimeError("checkpoint changed during terminal validation")
            if _sha256_file(checkpoint_progress) != checkpoint_progress_sha256:
                raise RuntimeError("checkpoint progress log changed during terminal validation")
            if (
                upstream_encoder_checkpoint is not None
                and _sha256_file(upstream_encoder_checkpoint) != upstream_encoder_sha256
            ):
                raise RuntimeError("upstream encoder changed during terminal validation")
            if (
                upstream_encoder_progress is not None
                and _sha256_file(upstream_encoder_progress) != upstream_encoder_progress_sha256
            ):
                raise RuntimeError("upstream encoder progress changed during terminal validation")
        if evaluation_split == "dev" and sealed_attempt_registry is not None:
            raise ValueError("--sealed-attempt-registry is only valid for split=test")
        durable_mkdir(output_dir)
        evaluation_path = output_dir / "evaluation.json"
        if evaluation_path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing file: {evaluation_path}")
        sequences, identifiers, evaluation_pose_snapshot = _cached_sequences(
            manifest,
            split=evaluation_split,
            cache_dir=cache_dir,
            pose_fingerprint=config.pose_fingerprint,
        )
        training_pose_snapshot_path = output_dir / "inputs" / "training-pose-cache-snapshot.json"
        evaluation_pose_snapshot_path = (
            output_dir / "inputs" / f"{evaluation_split}-pose-cache-snapshot.json"
        )
        training_pose_snapshot_sha256 = _persist_pose_cache_snapshot(
            training_pose_snapshot_path,
            training_pose_snapshot,
            overwrite=overwrite,
        )
        evaluation_pose_snapshot_sha256 = _persist_pose_cache_snapshot(
            evaluation_pose_snapshot_path,
            evaluation_pose_snapshot,
            overwrite=overwrite,
        )
        started_manifest_path = _create_cli_run_manifest(
            output_dir=output_dir,
            command=list(sys.argv),
            config=config,
            dataset_sha256=manifest.fingerprint,
            notes=[
                f"evaluation split: {evaluation_split}",
                (
                    "formal dev-only targets plus 337/84/105 label-free identity sidecars"
                    if label_free_bundle is not None
                    else "legacy labeled-manifest compatibility path"
                ),
                f"variant: {checkpoint_variant}",
                f"method_id: {sealed_method_id}",
                f"checkpoint stage: {expected_stage}",
                f"training pose cache set: {training_pose_snapshot.fingerprint}",
                f"{evaluation_split} pose cache set: {evaluation_pose_snapshot.fingerprint}",
            ],
        )
        model = _load_checkpoint_model(
            checkpoint_path,
            config,
            expected_stage=expected_stage,
            expected_provenance=provenance,
            device=device,
        )
        if _sha256_file(checkpoint_path) != checkpoint_sha256:
            raise RuntimeError("checkpoint changed while it was being loaded")
        if (
            upstream_encoder_checkpoint is not None
            and _sha256_file(upstream_encoder_checkpoint) != upstream_encoder_sha256
        ):
            raise RuntimeError("upstream encoder checkpoint changed during preflight")
        if (
            checkpoint_progress is not None
            and _sha256_file(checkpoint_progress) != checkpoint_progress_sha256
        ):
            raise RuntimeError("checkpoint progress log changed during preflight")
        if (
            upstream_encoder_progress is not None
            and _sha256_file(upstream_encoder_progress) != upstream_encoder_progress_sha256
        ):
            raise RuntimeError("upstream encoder progress changed during preflight")
        predictions = predict_sequences(
            model,
            sequences,
            config,
            device=_device_or_none(device),
        )
        if tuple(record.video_id for record in predictions) != tuple(identifiers):
            raise RuntimeError("label-free prediction order does not match cached video IDs")

        sealed_attempt_path: Path | None = None
        sealed_attempt_sha256: str | None = None
        if evaluation_split == "test":
            labeled_manifest = cast(UCFRepManifest, manifest)
            if _sha256_file(checkpoint_path) != checkpoint_sha256:
                raise RuntimeError("checkpoint changed during label-free prediction")
            if _sha256_file(config_path) != config_file_sha256:
                raise RuntimeError("configuration changed during label-free prediction")
            if _sha256_file(manifest_path) != dataset_manifest_sha256:
                raise RuntimeError("dataset manifest changed during label-free prediction")
            if (
                upstream_encoder_checkpoint is not None
                and _sha256_file(upstream_encoder_checkpoint) != upstream_encoder_sha256
            ):
                raise RuntimeError(
                    "upstream encoder checkpoint changed during label-free prediction"
                )
            if (
                checkpoint_progress is not None
                and _sha256_file(checkpoint_progress) != checkpoint_progress_sha256
            ):
                raise RuntimeError("checkpoint progress log changed during label-free prediction")
            if (
                upstream_encoder_progress is not None
                and _sha256_file(upstream_encoder_progress) != upstream_encoder_progress_sha256
            ):
                raise RuntimeError("upstream encoder progress changed during label-free prediction")
            registry = _canonical_sealed_registry(
                sealed_attempt_registry,
                protocol=manifest.protocol,
            )
            sealed_attempt_path, sealed_attempt_sha256 = _reserve_cli_sealed_attempt(
                registry,
                protocol=manifest.protocol,
                full_dataset_sha256=labeled_manifest.sealed_dataset_fingerprint,
                config_sha256=config.fingerprint,
                method_id=sealed_method_id,
                experiment_seed=config.seed,
                input_artifact_role="checkpoint",
                input_artifact_sha256=checkpoint_sha256,
            )

        evaluation_records: tuple[UCFRepRecord | DevTargetRecord, ...]
        if dev_target_manifest is None:
            evaluation_records = cast(UCFRepManifest, manifest).records_for(evaluation_split)
        else:
            evaluation_records = dev_target_manifest.records
        if tuple(record.video_id for record in evaluation_records) != tuple(identifiers):
            raise RuntimeError("evaluation records changed after label-free prediction")
        ground_truths = {record.video_id: record.count for record in evaluation_records}
        action_mapping = {record.video_id: record.action for record in evaluation_records}
        result = evaluate_predictions(
            predictions,
            ground_truths,
            actions=action_mapping,
            bootstrap_samples=10_000,
            bootstrap_seed=2026,
        )
        payload = {
            **result.to_dict(include_streams=True),
            "method_id": sealed_method_id,
            "variant": checkpoint_variant,
            "stage": expected_stage,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_progress_sha256": checkpoint_progress_sha256,
            "dataset_fingerprint": manifest.fingerprint,
            "sealed_dataset_fingerprint": (
                cast(UCFRepManifest, manifest).sealed_dataset_fingerprint
                if evaluation_split == "test"
                else None
            ),
            "config_fingerprint": config.fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
            "training_pose_cache_set_sha256": training_pose_snapshot.fingerprint,
            "evaluation_pose_cache_set_sha256": evaluation_pose_snapshot.fingerprint,
            "split": evaluation_split,
        }
        _write_json_exclusive(
            evaluation_path,
            payload,
            overwrite=overwrite,
        )
        evaluation_sha256 = _sha256_file(evaluation_path)
        report_metrics = result.report.to_dict()
        report_metrics.pop("per_video", None)
        completion_artifacts = {
            "input_checkpoint": checkpoint_path,
            "input_config": config_path,
            "input_dataset_manifest": manifest_path,
            "input_training_pose_cache_snapshot": training_pose_snapshot_path,
            "input_evaluation_pose_cache_snapshot": evaluation_pose_snapshot_path,
            "evaluation": evaluation_path,
        }
        completion_expected_sha256 = {
            "input_checkpoint": checkpoint_sha256,
            "input_config": config_file_sha256,
            "input_dataset_manifest": dataset_manifest_sha256,
            "input_training_pose_cache_snapshot": training_pose_snapshot_sha256,
            "input_evaluation_pose_cache_snapshot": evaluation_pose_snapshot_sha256,
            "evaluation": evaluation_sha256,
        }
        if label_free_bundle is not None:
            completion_artifacts.update(label_free_bundle.artifacts)
            completion_expected_sha256.update(label_free_bundle.artifact_sha256)
            assert dev_targets is not None
            assert dev_targets_sha256 is not None
            completion_artifacts["input_dev_targets"] = dev_targets
            completion_expected_sha256["input_dev_targets"] = dev_targets_sha256
        if checkpoint_progress is not None:
            assert checkpoint_progress_sha256 is not None
            completion_artifacts["input_checkpoint_progress"] = checkpoint_progress
            completion_expected_sha256["input_checkpoint_progress"] = checkpoint_progress_sha256
        if upstream_encoder_checkpoint is not None:
            assert upstream_encoder_sha256 is not None
            completion_artifacts["input_upstream_encoder_checkpoint"] = upstream_encoder_checkpoint
            completion_expected_sha256["input_upstream_encoder_checkpoint"] = (
                upstream_encoder_sha256
            )
        if upstream_encoder_progress is not None:
            assert upstream_encoder_progress_sha256 is not None
            completion_artifacts["input_upstream_encoder_progress"] = upstream_encoder_progress
            completion_expected_sha256["input_upstream_encoder_progress"] = (
                upstream_encoder_progress_sha256
            )
        if sealed_attempt_path is not None:
            assert sealed_attempt_sha256 is not None
            completion_artifacts["sealed_attempt_receipt"] = sealed_attempt_path
            completion_expected_sha256["sealed_attempt_receipt"] = sealed_attempt_sha256
        completed_manifest_path, completed_manifest_sha256 = _complete_cli_run_manifest(
            started_manifest_path,
            artifacts=completion_artifacts,
            expected_artifact_sha256=completion_expected_sha256,
            metrics=report_metrics,
        )
        payload.update(
            {
                "started_manifest_path": str(started_manifest_path.resolve()),
                "completed_manifest_path": str(completed_manifest_path.resolve()),
                "completed_manifest_sha256": completed_manifest_sha256,
                "sealed_attempt_path": (
                    None if sealed_attempt_path is None else str(sealed_attempt_path.resolve())
                ),
                "sealed_attempt_sha256": sealed_attempt_sha256,
                "training_pose_cache_snapshot_path": str(training_pose_snapshot_path.resolve()),
                "training_pose_cache_snapshot_sha256": training_pose_snapshot_sha256,
                "evaluation_pose_cache_snapshot_path": str(evaluation_pose_snapshot_path.resolve()),
                "evaluation_pose_cache_snapshot_sha256": evaluation_pose_snapshot_sha256,
            }
        )
    except ImportError as exc:
        _abort(f"checkpoint evaluation module is unavailable: {exc}")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)


@evaluate_app.command("dev-predict")
def evaluate_dev_predict_command(
    checkpoint_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    train_inputs_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    cache_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    checkpoint_progress_path: Annotated[
        Path,
        typer.Option(
            "--checkpoint-progress",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    checkpoint_completion_receipt_path: Annotated[
        Path,
        typer.Option(
            "--checkpoint-completion-receipt",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Independent completed training receipt for the checkpoint.",
        ),
    ],
    train_commitment_path: Annotated[
        Path,
        typer.Option(
            "--input-commitment",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    dev_inputs_path: Annotated[
        Path,
        typer.Option("--dev-inputs", exists=True, dir_okay=False, readable=True),
    ],
    dev_commitment_path: Annotated[
        Path,
        typer.Option(
            "--dev-input-commitment",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    test_identity_inputs_path: Annotated[
        Path,
        typer.Option(
            "--test-identity-inputs",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    test_identity_commitment_path: Annotated[
        Path,
        typer.Option(
            "--test-identity-commitment",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    variant: Annotated[
        str,
        typer.Option(
            "--variant",
            help="literal requires an encoder; sshead requires the inferred head.",
        ),
    ],
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/pams.yaml"),
    expert_mode: Annotated[
        str | None,
        typer.Option(
            "--expert-mode",
            help=(
                "Inference-only consensus override: multi or medium_only. "
                "medium_only is an inferred single-expert ablation."
            ),
        ),
    ] = None,
    device: Annotated[str, typer.Option("--device")] = "auto",
    upstream_encoder_checkpoint: Annotated[
        Path | None,
        typer.Option(
            "--upstream-encoder-checkpoint",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    upstream_encoder_progress: Annotated[
        Path | None,
        typer.Option(
            "--upstream-encoder-progress",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    upstream_encoder_completion_receipt: Annotated[
        Path | None,
        typer.Option(
            "--upstream-encoder-completion-receipt",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Completed encoder receipt required by the SSHead variant.",
        ),
    ] = None,
) -> None:
    """Freeze canonical PAMS dev predictions without accepting targets."""

    try:
        from pams.pams_dev import run_pams_dev_prediction

        normalized_variant = variant.strip().lower()
        if normalized_variant not in {"literal", "sshead"}:
            raise ValueError("--variant must be 'literal' or 'sshead'")
        normalized_expert_mode = None if expert_mode is None else expert_mode.strip().lower()
        if normalized_expert_mode not in {None, "multi", "medium_only"}:
            raise ValueError("--expert-mode must be 'multi' or 'medium_only'")
        payload = run_pams_dev_prediction(
            checkpoint_path=checkpoint_path,
            checkpoint_progress_path=checkpoint_progress_path,
            checkpoint_completion_receipt_path=checkpoint_completion_receipt_path,
            train_inputs_path=train_inputs_path,
            train_commitment_path=train_commitment_path,
            dev_inputs_path=dev_inputs_path,
            dev_commitment_path=dev_commitment_path,
            test_identity_inputs_path=test_identity_inputs_path,
            test_identity_commitment_path=test_identity_commitment_path,
            pose_cache_dir=cache_dir,
            output_dir=output_dir,
            config_path=config_path,
            variant=cast(Literal["literal", "sshead"], normalized_variant),
            expert_mode=cast(
                Literal["multi", "medium_only"] | None,
                normalized_expert_mode,
            ),
            upstream_encoder_checkpoint_path=upstream_encoder_checkpoint,
            upstream_encoder_progress_path=upstream_encoder_progress,
            upstream_encoder_completion_receipt_path=(upstream_encoder_completion_receipt),
            device=device,
            repository_root=Path.cwd(),
            command=list(sys.argv),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)


@evaluate_app.command("dev-score")
def evaluate_dev_score_command(
    predictions_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    prediction_receipt_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    dev_targets_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
) -> None:
    """Score frozen PAMS dev predictions at the only label-bearing boundary."""

    try:
        from pams.pams_dev import score_pams_dev_predictions

        payload = score_pams_dev_predictions(
            predictions_path=predictions_path,
            prediction_receipt_path=prediction_receipt_path,
            dev_targets_path=dev_targets_path,
            output_dir=output_dir,
            repository_root=Path.cwd(),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)


@stress_app.command("dev-predict")
def stress_dev_predict_command(
    source_clean_predictions_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    source_clean_prediction_receipt_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    checkpoint_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    dev_inputs_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    cache_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    dev_commitment_path: Annotated[
        Path,
        typer.Option(
            "--dev-input-commitment",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/pams.yaml"),
    stress_config_path: Annotated[
        Path,
        typer.Option(
            "--stress-config",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("configs/stress.yaml"),
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    """Freeze all target-free UCFRep dev pose-cache stress predictions."""

    try:
        from pams.stress_dev import run_pams_dev_stress_prediction

        payload = run_pams_dev_stress_prediction(
            source_clean_predictions_path=source_clean_predictions_path,
            source_clean_prediction_receipt_path=(source_clean_prediction_receipt_path),
            checkpoint_path=checkpoint_path,
            dev_inputs_path=dev_inputs_path,
            dev_commitment_path=dev_commitment_path,
            pose_cache_dir=cache_dir,
            output_dir=output_dir,
            config_path=config_path,
            stress_config_path=stress_config_path,
            device=device,
            repository_root=Path.cwd(),
            command=list(sys.argv),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)


@stress_app.command("dev-score")
def stress_dev_score_command(
    predictions_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    prediction_receipt_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    dev_targets_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    stress_config_path: Annotated[
        Path,
        typer.Option(
            "--stress-config",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("configs/stress.yaml"),
) -> None:
    """Score frozen stress predictions at the only label-bearing boundary."""

    try:
        from pams.stress_dev import score_pams_dev_stress_predictions

        payload = score_pams_dev_stress_predictions(
            predictions_path=predictions_path,
            prediction_receipt_path=prediction_receipt_path,
            stress_config_path=stress_config_path,
            dev_targets_path=dev_targets_path,
            output_dir=output_dir,
            repository_root=Path.cwd(),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)


@local_frequency_app.command("synthetic-replay")
def local_frequency_synthetic_replay_command(
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/readouts/local_frequency_synthetic_v1.yaml"),
) -> None:
    """Record the bounded synthetic-only parameter freeze and its hashes."""

    try:
        from pams.local_frequency_dev import run_synthetic_freeze_replay

        payload = run_synthetic_freeze_replay(
            config_path=config_path,
            output_dir=output_dir,
            repository_root=Path.cwd(),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)


@local_frequency_app.command("dev-predict")
def local_frequency_dev_predict_command(
    dev_inputs_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    dev_commitment_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    pose_cache_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/readouts/local_frequency_synthetic_v1.yaml"),
    pose_config_path: Annotated[
        Path,
        typer.Option("--pose-config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/pams.yaml"),
) -> None:
    """Freeze canonical dev predictions without accepting any target file."""

    try:
        from pams.local_frequency_dev import run_dev_prediction

        payload = run_dev_prediction(
            dev_inputs_path=dev_inputs_path,
            dev_commitment_path=dev_commitment_path,
            pose_cache_dir=pose_cache_dir,
            output_dir=output_dir,
            config_path=config_path,
            pose_config_path=pose_config_path,
            repository_root=Path.cwd(),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)


@local_frequency_app.command("dev-score")
def local_frequency_dev_score_command(
    predictions_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    prediction_receipt_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    dev_targets_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
) -> None:
    """Score an already-frozen artifact against the strict dev-only targets."""

    try:
        from pams.local_frequency_dev import score_dev_predictions

        payload = score_dev_predictions(
            predictions_path=predictions_path,
            prediction_receipt_path=prediction_receipt_path,
            dev_targets_path=dev_targets_path,
            output_dir=output_dir,
            repository_root=Path.cwd(),
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)


@teacher_period_app.command("dev-predict")
def teacher_period_dev_predict_command(
    checkpoint_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    train_inputs_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    pose_cache_dir: Annotated[
        Path,
        typer.Argument(exists=True, file_okay=False, readable=True),
    ],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    checkpoint_progress_path: Annotated[
        Path,
        typer.Option(
            "--checkpoint-progress",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    checkpoint_completion_receipt_path: Annotated[
        Path,
        typer.Option(
            "--checkpoint-completion-receipt",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    train_commitment_path: Annotated[
        Path,
        typer.Option(
            "--input-commitment",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    dev_inputs_path: Annotated[
        Path,
        typer.Option("--dev-inputs", exists=True, dir_okay=False, readable=True),
    ],
    dev_commitment_path: Annotated[
        Path,
        typer.Option(
            "--dev-input-commitment",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    test_identity_inputs_path: Annotated[
        Path,
        typer.Option(
            "--test-identity-inputs",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    test_identity_commitment_path: Annotated[
        Path,
        typer.Option(
            "--test-identity-commitment",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/experiments/pams_longest_contiguous_track_v8.yaml"),
    device: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    """Freeze target-free dev84 predictions from the exact v8 encoder."""

    try:
        from pams.teacher_period_dev import run_dev_prediction

        payload = run_dev_prediction(
            checkpoint_path=checkpoint_path,
            checkpoint_progress_path=checkpoint_progress_path,
            checkpoint_completion_receipt_path=(checkpoint_completion_receipt_path),
            train_inputs_path=train_inputs_path,
            train_commitment_path=train_commitment_path,
            dev_inputs_path=dev_inputs_path,
            dev_commitment_path=dev_commitment_path,
            test_identity_inputs_path=test_identity_inputs_path,
            test_identity_commitment_path=test_identity_commitment_path,
            pose_cache_dir=pose_cache_dir,
            output_dir=output_dir,
            config_path=config_path,
            device=None if device.strip().lower() == "auto" else device,
            repository_root=Path.cwd(),
            command=list(sys.argv),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)


@teacher_period_app.command("dev-score")
def teacher_period_dev_score_command(
    predictions_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    prediction_receipt_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    dev_targets_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
) -> None:
    """Score frozen diagnostic predictions at the label-bearing boundary."""

    try:
        from pams.teacher_period_dev import score_dev_predictions

        payload = score_dev_predictions(
            predictions_path=predictions_path,
            prediction_receipt_path=prediction_receipt_path,
            dev_targets_path=dev_targets_path,
            output_dir=output_dir,
            repository_root=Path.cwd(),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)


@diagnostic_app.command("encoder-shortcut-inferred")
def encoder_shortcut_inferred_diagnostic(
    checkpoint_path: Annotated[
        Path,
        typer.Argument(exists=True, dir_okay=False, readable=True),
    ],
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ],
    pose_cache_dir: Annotated[
        Path,
        typer.Option("--pose-cache-dir", exists=True, file_okay=False, readable=True),
    ],
    sample_size: Annotated[
        int,
        typer.Option(
            "--sample-size",
            min=0,
            help="Seeded training-video sample; 0 selects the full checkpoint-bound set.",
        ),
    ] = 64,
    seed: Annotated[int, typer.Option("--seed")] = 2026,
    device: Annotated[str, typer.Option("--device")] = "auto",
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 8,
    output: Annotated[Path | None, typer.Option("--output", "-o", dir_okay=False)] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Run the read-only, label-free inferred encoder shortcut audit."""

    try:
        from pams.diagnostics import run_encoder_shortcut_diagnostic

        config = load_config(config_path)
        payload = run_encoder_shortcut_diagnostic(
            checkpoint_path,
            config,
            config_path=config_path,
            pose_cache_dir=pose_cache_dir,
            sample_size=sample_size,
            seed=seed,
            device=device,
            batch_size=batch_size,
        )
        if output is not None:
            _write_json_exclusive(output, payload, overwrite=overwrite)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)


@baseline_app.command("list")
def baseline_list(
    paper_only: Annotated[bool, typer.Option("--paper-only")] = False,
) -> None:
    """List registered status without equating registration with executability."""

    specs = list_baselines(include_references=not paper_only)
    _emit(
        {
            "baselines": [
                {
                    "key": spec.key,
                    "paper_name": spec.paper_name,
                    "status": spec.status.value,
                    "runnable": spec.runnable,
                    "implementation": spec.implementation.value,
                    "paper_baseline": spec.is_paper_baseline,
                    "blocked_reason": spec.blocked_reason,
                }
                for spec in specs
            ]
        }
    )


@baseline_app.command("predict")
def baseline_predict(
    name: Annotated[str, typer.Argument()],
    pose_cache: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path | None, typer.Option("--output", "-o", dir_okay=False)] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Run only the explicit diagnostic spectral proxy on one cached sequence."""

    try:
        if name.strip().lower().replace("_", "-") != "spectral-proxy":
            # Registry construction supplies the precise blocked reason and
            # cannot silently return the diagnostic proxy.
            create_baseline(name)
            raise ValueError(
                "the CLI permits prediction only for spectral-proxy until a "
                "paper baseline has passed its parity and protocol gates"
            )
        adapter = create_baseline("spectral-proxy")
        sequence, metadata = load_pose_cache(pose_cache)
        result = adapter.predict(sequence)
        payload = {
            "method": adapter.spec.key,
            "diagnostic_only": True,
            "eligible_for_paper_table": False,
            "video_id": sequence.video_id,
            "pose_cache_pose_fingerprint": metadata.pose_fingerprint,
            **result.to_dict(),
        }
        if output is not None:
            _write_json_exclusive(output, payload, overwrite=overwrite)
    except (BaselineUnavailableError, KeyError, OSError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)


def _read_predictions(path: Path) -> dict[str, float]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or any("video_id" not in row or "prediction" not in row for row in rows):
            raise ValueError("prediction CSV requires video_id and prediction columns")
        pairs = [(str(row["video_id"]), float(row["prediction"])) for row in rows]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "predictions" in payload:
            payload = payload["predictions"]
        if isinstance(payload, dict):
            pairs = [(str(key), float(value)) for key, value in payload.items()]
        elif isinstance(payload, list):
            pairs = []
            for row in payload:
                if not isinstance(row, dict) or "video_id" not in row:
                    raise ValueError("prediction JSON rows require video_id")
                value_fields = [field for field in ("prediction", "count") if field in row]
                if len(value_fields) != 1:
                    raise ValueError(
                        "prediction JSON rows require exactly one of prediction or count"
                    )
                pairs.append((str(row["video_id"]), float(row[value_fields[0]])))
        else:
            raise ValueError("prediction JSON must be a mapping or list of rows")
    else:
        raise ValueError("predictions must be .json or .csv")
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("prediction video_id values must be unique")
    values = list(result.values())
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("predictions must contain only finite non-negative values")
    return result


@report_app.command("metrics")
def report_metrics(
    predictions_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    manifest_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
    output: Annotated[Path | None, typer.Option("--output", "-o", dir_okay=False)] = None,
    split: Annotated[str, typer.Option("--split")] = "test",
    bootstrap_samples: Annotated[int, typer.Option("--bootstrap-samples", min=0)] = 10_000,
    bootstrap_seed: Annotated[int, typer.Option("--bootstrap-seed")] = 2026,
    sealed_attempt_registry: Annotated[
        Path | None,
        typer.Option("--sealed-attempt-registry", file_okay=False),
    ] = None,
    method_id: Annotated[str | None, typer.Option("--method-id")] = None,
    experiment_seed: Annotated[int | None, typer.Option("--experiment-seed")] = None,
    experiment_config: Annotated[
        Path | None,
        typer.Option("--experiment-config", exists=True, dir_okay=False, readable=True),
    ] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Compute NMAE, raw MAE, RMSE, OBO, exact, and paired bootstrap CIs."""

    try:
        dataset_manifest_sha256 = _sha256_file(manifest_path)
        manifest = load_ucfrep_manifest(manifest_path, validate_exact=False)
        if _sha256_file(manifest_path) != dataset_manifest_sha256:
            raise RuntimeError("dataset manifest changed while it was being loaded")
        _validate_experiment_split(manifest)
        evaluation_split = split.strip().lower()
        if evaluation_split not in {"dev", "test"}:
            raise ValueError("metric report split must be 'dev' or 'test'")
        if evaluation_split == "test":
            if manifest.protocol != "ucfrep_526":
                raise ValueError(
                    "sealed metrics are disabled for ucfrep_pose_110 until its "
                    "official 89/21 ID/count digest is frozen"
                )
            if output is None:
                raise ValueError("sealed test metrics require an immutable --output artifact")
            if overwrite:
                raise ValueError("--overwrite is forbidden for sealed test metrics")
            if bootstrap_samples != 10_000:
                raise ValueError(
                    "sealed test metrics require exactly 10,000 paired bootstrap samples"
                )
            if bootstrap_seed != 2026:
                raise ValueError("sealed test metrics require bootstrap seed 2026")
        if output is not None and output.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing file: {output}")
        sealed_arguments = (
            sealed_attempt_registry,
            method_id,
            experiment_seed,
            experiment_config,
        )
        sealed_attempt_path: Path | None = None
        sealed_attempt_sha256: str | None = None
        started_manifest_path: Path | None = None
        completed_manifest_path: Path | None = None
        completed_manifest_sha256: str | None = None
        predictions_sha256 = _sha256_file(predictions_path)
        records = manifest.records_for(evaluation_split)
        predictions = _read_predictions(predictions_path)
        if _sha256_file(predictions_path) != predictions_sha256:
            raise RuntimeError("prediction artifact changed while it was being parsed")
        expected = {record.video_id for record in records}
        supplied = set(predictions)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(
                f"prediction IDs do not match {evaluation_split}: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        if evaluation_split == "test":
            if any(
                argument is None for argument in (method_id, experiment_seed, experiment_config)
            ):
                raise ValueError(
                    "test metrics require --method-id, --experiment-seed, and --experiment-config"
                )
            assert method_id is not None
            assert experiment_seed is not None
            assert experiment_config is not None
            normalized_method_id = _normalize_frozen_method_id(
                manifest.protocol,
                method_id,
            )
            registry = _canonical_sealed_registry(
                sealed_attempt_registry,
                protocol=manifest.protocol,
            )
            from pams.reproducibility import clean_git_revision
            from pams.run_manifest import create_run_manifest, write_manifest_exclusive

            experiment_config_sha256 = _sha256_file(experiment_config)
            clean_git_revision(Path.cwd())
            if _sha256_file(predictions_path) != predictions_sha256:
                raise RuntimeError("prediction artifact changed before sealed reservation")
            if _sha256_file(manifest_path) != dataset_manifest_sha256:
                raise RuntimeError("dataset manifest changed before sealed reservation")
            if _sha256_file(experiment_config) != experiment_config_sha256:
                raise RuntimeError("experiment configuration changed before sealed reservation")
            started = create_run_manifest(
                command=list(sys.argv),
                config_sha256=experiment_config_sha256,
                dataset_sha256=manifest.fingerprint,
                seed=experiment_seed,
                protocol=manifest.protocol,
                cwd=Path.cwd(),
                notes=[
                    f"sealed metrics method: {normalized_method_id}",
                    f"evaluation split: {evaluation_split}",
                ],
            )
            assert output is not None
            started_manifest_path = write_manifest_exclusive(
                started,
                output.parent / "manifests" / f"{started.run_id}.started.json",
            )
            sealed_attempt_path, sealed_attempt_sha256 = _reserve_cli_sealed_attempt(
                registry,
                protocol=manifest.protocol,
                full_dataset_sha256=manifest.sealed_dataset_fingerprint,
                config_sha256=experiment_config_sha256,
                method_id=normalized_method_id,
                experiment_seed=experiment_seed,
                input_artifact_role="predictions",
                input_artifact_sha256=predictions_sha256,
            )
        elif any(argument is not None for argument in sealed_arguments):
            raise ValueError("sealed-attempt options are only valid for split=test")
        report = compute_count_metrics(
            [predictions[record.video_id] for record in records],
            [record.count for record in records],
            video_ids=[record.video_id for record in records],
            actions=[record.action for record in records],
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
        payload = {
            "protocol": manifest.protocol,
            "split": evaluation_split,
            "dataset_sha256": manifest.fingerprint,
            "sealed_dataset_sha256": (
                manifest.sealed_dataset_fingerprint if evaluation_split == "test" else None
            ),
            "predictions_sha256": predictions_sha256,
            "sealed_attempt_path": (
                None if sealed_attempt_path is None else str(sealed_attempt_path.resolve())
            ),
            "sealed_attempt_sha256": sealed_attempt_sha256,
            "started_manifest_path": (
                None if started_manifest_path is None else str(started_manifest_path.resolve())
            ),
            **report.to_dict(),
        }
        if output is not None:
            _write_json_exclusive(output, payload, overwrite=overwrite)
        if evaluation_split == "test":
            assert output is not None
            assert experiment_config is not None
            assert sealed_attempt_path is not None
            assert started_manifest_path is not None
            assert sealed_attempt_sha256 is not None
            metrics_output_sha256 = _sha256_file(output)
            report_metrics_payload = report.to_dict()
            report_metrics_payload.pop("per_video", None)
            completed_manifest_path, completed_manifest_sha256 = _complete_cli_run_manifest(
                started_manifest_path,
                artifacts={
                    "experiment_config": experiment_config,
                    "input_dataset_manifest": manifest_path,
                    "input_predictions": predictions_path,
                    "sealed_attempt_receipt": sealed_attempt_path,
                    "metrics_output": output,
                },
                expected_artifact_sha256={
                    "experiment_config": experiment_config_sha256,
                    "input_dataset_manifest": dataset_manifest_sha256,
                    "input_predictions": predictions_sha256,
                    "sealed_attempt_receipt": sealed_attempt_sha256,
                    "metrics_output": metrics_output_sha256,
                },
                metrics=report_metrics_payload,
            )
            payload.update(
                {
                    "completed_manifest_path": str(completed_manifest_path.resolve()),
                    "completed_manifest_sha256": completed_manifest_sha256,
                }
            )
    except (OSError, RuntimeError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)


def _parse_counts(value: str) -> tuple[int, ...]:
    try:
        counts = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("counts must be comma-separated integers") from exc
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("counts must contain unique comma-separated integers")
    if any(count < 2 or count > 40 for count in counts):
        raise ValueError("synthetic counts must be in [2, 40]")
    return counts


@synthetic_app.command("evaluate")
def synthetic_evaluate(
    config_path: Annotated[Path, typer.Option("--config", exists=True, dir_okay=False)] = Path(
        "configs/pams.yaml"
    ),
    counts: Annotated[str, typer.Option("--counts")] = ",".join(
        str(count) for count in range(2, 41)
    ),
    stress: Annotated[bool, typer.Option("--stress")] = False,
    bootstrap_samples: Annotated[int, typer.Option("--bootstrap-samples", min=0)] = 0,
    output: Annotated[Path | None, typer.Option("--output", "-o", dir_okay=False)] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Evaluate the explicitly named spectral proxy on deterministic synthetic pose."""

    try:
        from pams.baselines.proxy import SpectralProxyAdapter

        config = load_config(config_path)
        selected_counts = _parse_counts(counts)
        adapter = SpectralProxyAdapter(
            minimum_period=config.period.minimum,
            maximum_period=config.period.maximum,
        )
        samples = [
            generate_synthetic_sample(
                SyntheticSpec(
                    video_id=f"count-{count:02d}",
                    frames=config.data.frames,
                    count=float(count),
                    seed=config.seed + count,
                )
            )
            for count in selected_counts
        ]
        if stress:
            samples.extend(
                synthetic_stress_suite(
                    count=8.0,
                    frames=config.data.frames,
                    seed=config.seed,
                ).values()
            )
        predictions = [adapter.predict(sample.sequence).count for sample in samples]
        targets = [int(round(sample.target_count)) for sample in samples]
        identifiers = [sample.sequence.video_id for sample in samples]
        report = compute_count_metrics(
            predictions,
            targets,
            video_ids=identifiers,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=config.seed,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        _abort(str(exc))
    payload = {
        "method": "spectral-proxy",
        "diagnostic_only": True,
        "eligible_for_paper_table": False,
        "full_count_sweep": selected_counts == tuple(range(2, 41)),
        "stress_enabled": stress,
        "config_sha256": config.fingerprint,
        **report.to_dict(),
    }
    if output is not None:
        try:
            _write_json_exclusive(output, payload, overwrite=overwrite)
        except OSError as exc:
            _abort(str(exc))
    _emit(payload)


@synthetic_app.command("acceptance")
def synthetic_acceptance(
    output_dir: Annotated[Path, typer.Argument(file_okay=False)],
    config_path: Annotated[
        Path,
        typer.Option("--config", exists=True, dir_okay=False, readable=True),
    ] = Path("configs/pams.yaml"),
    stress_config_path: Annotated[
        Path,
        typer.Option(
            "--stress-config",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = Path("configs/stress.yaml"),
) -> None:
    """Publish the two 576-case gates and full config-derived pose suite."""

    try:
        from pams.synthetic_audit import run_synthetic_acceptance

        payload = run_synthetic_acceptance(
            output_dir=output_dir,
            config_path=config_path,
            stress_config_path=stress_config_path,
            repository_root=Path.cwd(),
            command=list(sys.argv),
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _abort(str(exc))
    _emit(payload)
    if not bool(payload["passed"]):
        raise typer.Exit(1)


def _publish_safety_report(
    report: dict[str, Any],
    *,
    output: Path | None,
    overwrite: bool,
) -> None:
    if output is not None:
        try:
            _write_json_exclusive(output, report, overwrite=overwrite)
        except OSError as exc:
            _abort(str(exc))
    _emit(report)
    if not bool(report.get("passed")):
        raise typer.Exit(1)


@synthetic_app.command("counter-gate")
def synthetic_counter_gate(
    output: Annotated[Path | None, typer.Option("--output", "-o", dir_okay=False)] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Test only multi-expert counting with exact synthetic periods."""

    from pams.safety import run_counter_sign_phase_gate

    report = run_counter_sign_phase_gate()
    _publish_safety_report(report, output=output, overwrite=overwrite)


@synthetic_app.command("period-counter-gate")
def synthetic_period_counter_gate(
    output: Annotated[Path | None, typer.Option("--output", "-o", dir_okay=False)] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Test FFT period estimation followed by multi-expert counting."""

    from pams.safety import run_period_counter_sign_phase_gate

    report = run_period_counter_sign_phase_gate()
    _publish_safety_report(report, output=output, overwrite=overwrite)


@synthetic_app.command("sshead-collapse")
def synthetic_sshead_collapse(
    output: Annotated[Path | None, typer.Option("--output", "-o", dir_okay=False)] = None,
    overwrite: Annotated[bool, typer.Option("--overwrite")] = False,
) -> None:
    """Expose exact and near-collapse behavior of the inferred SSHead loss."""

    from pams.safety import run_sshead_collapse_diagnostic

    report = run_sshead_collapse_diagnostic()
    _publish_safety_report(report, output=output, overwrite=overwrite)


if __name__ == "__main__":
    app()
