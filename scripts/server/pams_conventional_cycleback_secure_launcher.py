#!/usr/bin/env python3
"""Fail-closed host launcher for all conventional cycle-back stages."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "pams_conventional_cycleback_wrapper_contract",
    Path(__file__).with_name("pams_conventional_cycleback_wrapper_contract.py"),
)
if _CONTRACT_SPEC is None or _CONTRACT_SPEC.loader is None:
    raise RuntimeError("cycleback wrapper contract module cannot be loaded")
_CONTRACT_MODULE = importlib.util.module_from_spec(_CONTRACT_SPEC)
_CONTRACT_SPEC.loader.exec_module(_CONTRACT_MODULE)
validate_inspect = _CONTRACT_MODULE.validate_inspect

CANONICAL_ROOT = Path("/media/lenovo/data2/pams-rac")
SAFE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
GIT_BINARY = "/usr/bin/git"
DOCKER_BINARY = "/usr/bin/docker"
LAUNCH_REGISTRY_ROOT = (
    CANONICAL_ROOT
    / "authorizations/pams-conventional-cycleback-launch-v1"
)
OUTCOME_REGISTRY_ROOT = (
    CANONICAL_ROOT
    / "authorizations/pams-conventional-cycleback-outcomes-v1"
)
GPU_LOCK_ROOT = CANONICAL_ROOT / ".pams-cycleback-gpu-locks-v1"
ATTEMPT_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,39}$")
CONTAINER_ID_PATTERN = re.compile(r"^(?:sha256:)?([0-9a-f]{64})$")
CONFIGS = {
    "W16_H4": "configs/conventional_cycleback/w16_hop4_v1.yaml",
    "W16_H2": "configs/conventional_cycleback/w16_hop2_v1.yaml",
    "W24_H4": "configs/conventional_cycleback/w24_hop4_v1.yaml",
}
STAGE_METADATA = {
    "adapter": {
        "run_parent": "runs/pams-conventional-cycleback-pose-input-authorization-v1",
        "container_prefix": "pams-cycleback-pose-auth",
        "memory": "32g",
        "memory_bytes": 32 * 1024**3,
        "cpus": "8",
        "nano_cpus": 8_000_000_000,
        "gpu": False,
        "allowed_exit_codes": [0],
    },
    "geometry": {
        "run_parent": "runs/pams-conventional-cycleback-geometry-v1",
        "container_prefix": "pams-cycleback-geometry",
        "memory": "64g",
        "memory_bytes": 64 * 1024**3,
        "cpus": "8",
        "nano_cpus": 8_000_000_000,
        "gpu": True,
        "allowed_exit_codes": [0, 3],
    },
    "mechanism": {
        "run_parent": "runs/pams-conventional-cycleback-mechanism-v1",
        "container_prefix": "pams-cycleback-mechanism",
        "memory": "96g",
        "memory_bytes": 96 * 1024**3,
        "cpus": "8",
        "nano_cpus": 8_000_000_000,
        "gpu": True,
        "allowed_exit_codes": [0, 3],
    },
}


@dataclass(frozen=True, slots=True)
class FileIdentity:
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class Predecessor:
    role: str
    root: Path
    output_path: Path
    output_identity: FileIdentity
    receipt_path: Path
    receipt_identity: FileIdentity
    tree_sha256: str
    tree_bytes: int
    tree_files: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "root": os.fspath(self.root),
            "output": {
                "path": os.fspath(self.output_path),
                **self.output_identity.to_dict(),
            },
            "receipt": {
                "path": os.fspath(self.receipt_path),
                **self.receipt_identity.to_dict(),
            },
            "tree_sha256": self.tree_sha256,
            "tree_bytes": self.tree_bytes,
            "tree_files": self.tree_files,
        }


class LaunchFailure(RuntimeError):
    """Infrastructure or contract failure, distinct from scientific reject."""


def _stable_bytes(path: Path, *, role: str) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise LaunchFailure(f"{role} must be a regular non-symlink file")
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            encoded = handle.read()
            closed = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = path.lstat()
    for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns"):
        if not (
            getattr(before, field)
            == getattr(opened, field)
            == getattr(closed, field)
            == getattr(after, field)
        ):
            raise LaunchFailure(f"{role} changed while being read")
    return encoded


def _identity(path: Path, *, role: str) -> FileIdentity:
    encoded = _stable_bytes(path, role=role)
    return FileIdentity(hashlib.sha256(encoded).hexdigest(), len(encoded))


def _json_no_duplicates(encoded: bytes, *, role: str) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LaunchFailure(f"{role} contains duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(encoded, object_pairs_hook=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LaunchFailure(f"{role} is not valid JSON") from exc


def _read_json(path: Path, *, role: str) -> tuple[Any, FileIdentity]:
    encoded = _stable_bytes(path, role=role)
    return (
        _json_no_duplicates(encoded, role=role),
        FileIdentity(hashlib.sha256(encoded).hexdigest(), len(encoded)),
    )


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any], *, mode: int = 0o440) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    stdout_path: Path | None = None,
) -> subprocess.CompletedProcess[bytes]:
    environment = {
        "PATH": SAFE_PATH,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    if stdout_path is None:
        completed = subprocess.run(
            list(arguments),
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
        )
    else:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= int(getattr(os, "O_NOFOLLOW", 0))
        flags |= int(getattr(os, "O_CLOEXEC", 0))
        descriptor = os.open(stdout_path, flags, 0o440)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = -1
                completed = subprocess.run(
                    list(arguments),
                    cwd=cwd,
                    env=environment,
                    check=False,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            if descriptor >= 0:
                os.close(descriptor)
    if check and completed.returncode != 0:
        raise LaunchFailure(
            f"command failed ({arguments[0]}), exit={completed.returncode}"
        )
    return completed


def _git_text(repository: Path, *arguments: str) -> str:
    completed = _run([GIT_BINARY, "-C", os.fspath(repository), *arguments])
    try:
        return completed.stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise LaunchFailure("git identity output is not ASCII") from exc


def _validate_attempt(value: str) -> str:
    if not ATTEMPT_PATTERN.fullmatch(value) or ".." in value:
        raise LaunchFailure("attempt ID is not a safe immutable slug")
    return value


def _validate_directory(path: Path, *, role: str, sealed: bool = False) -> Path:
    resolved = path.resolve(strict=True)
    details = path.lstat()
    if not stat.S_ISDIR(details.st_mode) or stat.S_ISLNK(details.st_mode):
        raise LaunchFailure(f"{role} must be a non-symlink directory")
    for candidate in (resolved, *resolved.rglob("*")):
        current = candidate.lstat()
        if stat.S_ISLNK(current.st_mode):
            raise LaunchFailure(f"{role} contains a symlink")
        if not (stat.S_ISDIR(current.st_mode) or stat.S_ISREG(current.st_mode)):
            raise LaunchFailure(f"{role} contains a special filesystem object")
        if sealed and current.st_mode & 0o222:
            raise LaunchFailure(f"{role} contains a writable path")
    return resolved


def _tree_identity(root: Path, *, role: str) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    total_bytes = 0
    total_files = 0
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        encoded = _stable_bytes(path, role=f"{role} {relative}")
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(len(encoded)).encode("ascii"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(encoded).digest())
        total_bytes += len(encoded)
        total_files += 1
    return digest.hexdigest(), total_bytes, total_files


def _write_source_manifest(root: Path, output: Path) -> FileIdentity:
    rows: list[str] = []
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        identity = _identity(path, role=f"source export {relative}")
        rows.append(f"{identity.sha256} {identity.bytes} {relative}\n")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(output, flags, 0o440)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            handle.writelines(rows)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(output.parent)
    return _identity(output, role="source export manifest")


def _archive_source(repository: Path, revision: str, output: Path) -> None:
    archive = _run(
        [GIT_BINARY, "-C", os.fspath(repository), "archive", revision]
    ).stdout
    output.mkdir(mode=0o750)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as bundle:
        for member in bundle.getmembers():
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts:
                raise LaunchFailure("git archive contains an unsafe path")
            destination = output.joinpath(*pure.parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise LaunchFailure("git archive contains a non-file entry")
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = bundle.extractfile(member)
            if source is None:
                raise LaunchFailure("git archive file cannot be read")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= int(getattr(os, "O_NOFOLLOW", 0))
            flags |= int(getattr(os, "O_CLOEXEC", 0))
            descriptor = os.open(destination, flags, 0o440)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    descriptor = -1
                    shutil.copyfileobj(source, handle, length=1024 * 1024)
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
    for candidate in sorted(output.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        candidate.chmod(0o555 if candidate.is_dir() else 0o444)
    output.chmod(0o555)


def _load_launch_bootstrap(repository: Path) -> tuple[Any, Mapping[str, Any]]:
    authorization_path = LAUNCH_REGISTRY_ROOT / "launch.authorization.json"
    authorization, _ = _read_json(authorization_path, role="launch authorization bootstrap")
    if not isinstance(authorization, Mapping):
        raise LaunchFailure("launch authorization bootstrap is not an object")
    identity = authorization.get("identity")
    if not isinstance(identity, Mapping):
        raise LaunchFailure("launch authorization identity is missing")
    revision = identity.get("source_revision")
    tree = identity.get("source_tree_sha")
    if _git_text(repository, "rev-parse", "HEAD^{commit}") != revision:
        raise LaunchFailure("current checkout is not the preregistered source revision")
    if _git_text(repository, "rev-parse", "HEAD^{tree}") != tree:
        raise LaunchFailure("current checkout tree is not the preregistered source tree")
    if _run(
        [
            GIT_BINARY,
            "-C",
            os.fspath(repository),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ]
    ).stdout:
        raise LaunchFailure("current checkout is not clean")
    sys.path.insert(0, os.fspath(repository / "src"))
    from pams.conventional_cycleback.authority import (
        validate_cycleback_launch_registry,
    )

    launch = validate_cycleback_launch_registry(
        LAUNCH_REGISTRY_ROOT,
        declared_host_root=LAUNCH_REGISTRY_ROOT,
        source_root=repository,
    )
    return launch, identity


def _outcome_registry_path(role: str, candidate_id: str | None) -> Path:
    if role == "pose_input":
        return OUTCOME_REGISTRY_ROOT / "pose-input/outcome.json"
    if role == "geometry" and candidate_id in CONFIGS:
        return OUTCOME_REGISTRY_ROOT / f"geometry/{candidate_id}.outcome.json"
    raise LaunchFailure("invalid predecessor outcome registry role")


def _outcome_slot_lock_path(stage: str, candidate_id: str | None) -> Path:
    suffix = "global" if candidate_id is None else candidate_id.lower()
    return OUTCOME_REGISTRY_ROOT / f"locks/{stage}-{suffix}.lock"


def _load_predecessor(
    *,
    role: str,
    candidate_id: str | None,
    launch: Any,
) -> Predecessor:
    registry_path = _outcome_registry_path(role, candidate_id)
    record, _ = _read_json(registry_path, role=f"{role} outcome registry")
    if not isinstance(record, Mapping):
        raise LaunchFailure(f"{role} outcome registry is not an object")
    expected_type = f"pams_conventional_cycleback_{role}_outcome_registry_v1"
    if (
        record.get("schema_version") != 1
        or record.get("artifact_type") != expected_type
        or record.get("status") != "passed"
        or record.get("launch_registry_id") != launch.registry_id
        or record.get("source_revision") != launch.source_revision
        or record.get("container_image_id") != launch.container_image_id
    ):
        raise LaunchFailure(f"{role} outcome registry lineage mismatch")
    if record.get("candidate_id") != candidate_id:
        raise LaunchFailure(f"{role} outcome registry candidate mismatch")
    root_value = record.get("root")
    if not isinstance(root_value, str):
        raise LaunchFailure(f"{role} outcome root is missing")
    root = _validate_directory(Path(root_value), role=f"{role} predecessor", sealed=True)
    expected_parent = (
        CANONICAL_ROOT
        / (
            "runs/pams-conventional-cycleback-pose-input-authorization-v1"
            if role == "pose_input"
            else "runs/pams-conventional-cycleback-geometry-v1"
        )
    )
    if root.parent != expected_parent or not root.name.startswith(
        f"{launch.source_revision[:12]}-"
    ):
        raise LaunchFailure(f"{role} predecessor is outside the canonical namespace")
    output_relative = record.get("output_relative")
    receipt_relative = record.get("receipt_relative")
    if not isinstance(output_relative, str) or not isinstance(receipt_relative, str):
        raise LaunchFailure(f"{role} predecessor locators are missing")
    for relative in (output_relative, receipt_relative):
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts:
            raise LaunchFailure(f"{role} predecessor locator is unsafe")
    output = root.joinpath(*PurePosixPath(output_relative).parts)
    receipt = root.joinpath(*PurePosixPath(receipt_relative).parts)
    output_identity = _identity(output, role=f"{role} predecessor output")
    receipt_identity = _identity(receipt, role=f"{role} predecessor receipt")
    if record.get("output") != output_identity.to_dict():
        raise LaunchFailure(f"{role} predecessor output bytes differ from registry")
    if record.get("receipt") != receipt_identity.to_dict():
        raise LaunchFailure(f"{role} predecessor receipt bytes differ from registry")
    tree_sha, tree_bytes, tree_files = _tree_identity(root, role=f"{role} predecessor")
    if record.get("tree") != {
        "sha256": tree_sha,
        "bytes": tree_bytes,
        "files": tree_files,
    }:
        raise LaunchFailure(f"{role} predecessor tree differs from registry")
    predecessor = Predecessor(
        role=role,
        root=root,
        output_path=output,
        output_identity=output_identity,
        receipt_path=receipt,
        receipt_identity=receipt_identity,
        tree_sha256=tree_sha,
        tree_bytes=tree_bytes,
        tree_files=tree_files,
    )
    if role == "pose_input":
        from pams.conventional_cycleback.authority import (
            validate_cycleback_pose_authority,
        )

        validated = validate_cycleback_pose_authority(
            root,
            declared_host_root=root,
            launch_authority=launch,
            expected_authorization_sha256=output_identity.sha256,
            expected_run_receipt_sha256=receipt_identity.sha256,
        )
        if (
            validated.authorization_path != output
            or validated.run_receipt_path != receipt
        ):
            raise LaunchFailure("pose-input predecessor semantic paths mismatch")
    return predecessor


def _revalidate_predecessor(predecessor: Predecessor) -> None:
    if _identity(
        predecessor.output_path, role=f"{predecessor.role} predecessor output post-run"
    ) != predecessor.output_identity:
        raise LaunchFailure(f"{predecessor.role} predecessor output changed during launch")
    if _identity(
        predecessor.receipt_path,
        role=f"{predecessor.role} predecessor receipt post-run",
    ) != predecessor.receipt_identity:
        raise LaunchFailure(f"{predecessor.role} predecessor receipt changed during launch")
    tree = _tree_identity(predecessor.root, role=f"{predecessor.role} predecessor post-run")
    if tree != (
        predecessor.tree_sha256,
        predecessor.tree_bytes,
        predecessor.tree_files,
    ):
        raise LaunchFailure(f"{predecessor.role} predecessor tree changed during launch")


def _open_lock(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    parent_details = path.parent.lstat()
    if not stat.S_ISDIR(parent_details.st_mode) or stat.S_ISLNK(parent_details.st_mode):
        raise LaunchFailure("canonical lock root is not a safe directory")
    flags = os.O_RDWR | os.O_CREAT
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(path, flags, 0o600)
    opened = os.fstat(descriptor)
    after = path.lstat()
    if not stat.S_ISREG(opened.st_mode) or (
        opened.st_dev,
        opened.st_ino,
    ) != (after.st_dev, after.st_ino):
        os.close(descriptor)
        raise LaunchFailure("canonical lock file identity is unsafe")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise LaunchFailure(f"lock is already held: {path.name}") from exc
    return descriptor


def _quarantine_stale(
    *,
    run_parent: Path,
    run_key: str,
    reservation_path: Path,
    staging: Path,
) -> None:
    stale = [path for path in (reservation_path, staging) if path.exists()]
    if not stale:
        return
    quarantine_parent = run_parent / "quarantine"
    quarantine_parent.mkdir(parents=True, exist_ok=True)
    quarantine = Path(
        tempfile.mkdtemp(prefix=f"{run_key}.stale.", dir=os.fspath(quarantine_parent))
    )
    for path in stale:
        if path.is_symlink():
            raise LaunchFailure("stale run state is a symlink")
        os.rename(path, quarantine / path.name)
    _fsync_directory(quarantine_parent)
    _fsync_directory(run_parent)


def _reserve_stage(
    *,
    stage: str,
    attempt_id: str,
    candidate_id: str | None,
    launch: Any,
    config_identity: Mapping[str, Any] | None,
    predecessors: Sequence[Predecessor],
) -> tuple[Path, Path, Path, FileIdentity, int]:
    metadata = STAGE_METADATA[stage]
    run_parent = CANONICAL_ROOT / str(metadata["run_parent"])
    run_parent.mkdir(parents=True, exist_ok=True)
    if run_parent.is_symlink():
        raise LaunchFailure("run parent is a symlink")
    run_key = f"{launch.source_revision[:12]}-{attempt_id}"
    final_root = run_parent / run_key
    staging = run_parent / f".incomplete-{run_key}"
    reservation_path = run_parent / f"{run_key}.reservation.json"
    lock_path = run_parent / f".{run_key}.launch.lock"
    lock_descriptor = _open_lock(lock_path)
    try:
        if final_root.exists():
            raise LaunchFailure("final run outcome already exists")
        unsafe_failure = final_root.with_name(f"{final_root.name}.failure.receipt.json")
        if unsafe_failure.exists():
            failure, _ = _read_json(
                unsafe_failure,
                role="prior unverified-cleanup failure receipt",
            )
            if not isinstance(failure, Mapping) or not any(
                failure.get(key) is True
                for key in (
                    "staging_publication_forbidden_while_container_may_exist",
                    "staging_publication_forbidden_after_primary_write_failure",
                )
            ):
                raise LaunchFailure("prior run-specific fallback receipt is malformed")
            raise LaunchFailure(
                "prior run publication is unsafe; manual recovery is required"
            )
        _quarantine_stale(
            run_parent=run_parent,
            run_key=run_key,
            reservation_path=reservation_path,
            staging=staging,
        )
        reservation = {
            "schema_version": 1,
            "artifact_type": "pams_conventional_cycleback_stage_reservation_v1",
            "stage": stage,
            "attempt_id": attempt_id,
            "candidate_id": candidate_id,
            "run_key": run_key,
            "final_root": os.fspath(final_root),
            "staging_root": os.fspath(staging),
            "launch_registry_id": launch.registry_id,
            "launch_authorization_sha256": launch.authorization_sha256,
            "launch_registry_receipt_sha256": launch.receipt_sha256,
            "source_revision": launch.source_revision,
            "source_tree_sha": launch.source_tree_sha,
            "container_image_id": launch.container_image_id,
            "config": config_identity,
            "predecessors": [item.to_dict() for item in predecessors],
            "process_id": os.getpid(),
            "authority_granted": False,
        }
        _write_json_exclusive(reservation_path, reservation)
        reservation_identity = _identity(reservation_path, role="stage reservation")
        staging.mkdir(mode=0o750)
        _fsync_directory(run_parent)
    except BaseException:
        os.close(lock_descriptor)
        raise
    return final_root, staging, reservation_path, reservation_identity, lock_descriptor


def _image_environment(image_id: str) -> dict[str, str]:
    completed = _run([DOCKER_BINARY, "image", "inspect", image_id])
    payload = _json_no_duplicates(completed.stdout, role="Docker image inspect")
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], Mapping):
        raise LaunchFailure("Docker image inspect schema mismatch")
    item = payload[0]
    if item.get("Id") != image_id:
        raise LaunchFailure("Docker image ID differs from launch registry")
    config = item.get("Config")
    if not isinstance(config, Mapping):
        raise LaunchFailure("Docker image config is missing")
    values = config.get("Env") or []
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise LaunchFailure("Docker image environment is malformed")
    result: dict[str, str] = {}
    for encoded in values:
        key, separator, value = encoded.partition("=")
        if not separator or not key or key in result:
            raise LaunchFailure("Docker image environment has a duplicate/malformed key")
        result[key] = value
    return result


def _stage_command(
    *,
    stage: str,
    candidate_id: str | None,
    config_relative: str | None,
    launch: Any,
    predecessors: Sequence[Predecessor],
) -> tuple[list[str], list[dict[str, Any]], str]:
    mounts: list[dict[str, Any]] = [
        {
            "destination": "/workspace",
            "type": "bind",
            "source": "SOURCE_EXPORT",
            "rw": False,
            "propagation": "rprivate",
        },
        {
            "destination": "/pams/launch-registry",
            "type": "bind",
            "source": os.fspath(LAUNCH_REGISTRY_ROOT),
            "rw": False,
            "propagation": "rprivate",
        },
        {
            "destination": "/pams/output",
            "type": "bind",
            "source": "OUTPUT_ROOT",
            "rw": True,
            "propagation": "rprivate",
        },
    ]
    if stage == "adapter":
        representation = Path(launch.representation_root)
        mounts.append(
            {
                "destination": "/pams/representation",
                "type": "bind",
                "source": os.fspath(representation),
                "rw": False,
                "propagation": "rprivate",
            }
        )
        command = [
            "-I",
            "scripts/server/run_pams_conventional_cycleback_pose_input_authorization.py",
            "--representation-root",
            "/pams/representation",
            "--declared-representation-host-root",
            os.fspath(representation),
            "--launch-registry-root",
            "/pams/launch-registry",
            "--declared-launch-registry-host-root",
            os.fspath(LAUNCH_REGISTRY_ROOT),
            "--source-root",
            "/workspace",
            "--output-root",
            "/pams/output",
        ]
        return command, mounts, "artifact/authorization/cycleback-pose-input.authorization.json"
    pose = next((item for item in predecessors if item.role == "pose_input"), None)
    if pose is None or candidate_id not in CONFIGS or config_relative is None:
        raise LaunchFailure("stage lacks its frozen pose/config predecessor")
    mounts.append(
        {
            "destination": "/pams/pose-authority",
            "type": "bind",
            "source": os.fspath(pose.root),
            "rw": False,
            "propagation": "rprivate",
        }
    )
    common = [
        "--config",
        f"/workspace/{config_relative}",
        "--source-root",
        "/workspace",
        "--launch-registry-root",
        "/pams/launch-registry",
        "--declared-launch-registry-host-root",
        os.fspath(LAUNCH_REGISTRY_ROOT),
        "--pose-authority-root",
        "/pams/pose-authority",
        "--declared-pose-authority-host-root",
        os.fspath(pose.root),
        "--expected-pose-authorization-sha256",
        pose.output_identity.sha256,
        "--expected-pose-authority-run-receipt-sha256",
        pose.receipt_identity.sha256,
        "--source-git-sha",
        launch.source_revision,
        "--container-image-id",
        launch.container_image_id,
        "--device",
        "cuda:0",
        "--encoder-batch-size",
        "8",
    ]
    if stage == "geometry":
        command = [
            "-I",
            "scripts/server/run_pams_conventional_cycleback_geometry_gate.py",
            *common,
            "--output",
            "/pams/output/geometry-gate.json",
        ]
        return command, mounts, "output/geometry-gate.json"
    geometry = next((item for item in predecessors if item.role == "geometry"), None)
    if geometry is None:
        raise LaunchFailure("mechanism stage lacks a passed geometry predecessor")
    mounts.append(
        {
            "destination": "/pams/geometry",
            "type": "bind",
            "source": os.fspath(geometry.root),
            "rw": False,
            "propagation": "rprivate",
        }
    )
    command = [
        "-I",
        "scripts/server/run_pams_conventional_cycleback_mechanism_probe.py",
        *common,
        "--geometry-gate",
        "/pams/geometry/output/geometry-gate.json",
        "--expected-geometry-gate-sha256",
        geometry.output_identity.sha256,
        "--geometry-run-receipt",
        "/pams/geometry/audit/run.receipt.json",
        "--expected-geometry-run-receipt-sha256",
        geometry.receipt_identity.sha256,
        "--declared-geometry-host-root",
        os.fspath(geometry.root),
        "--output",
        "/pams/output/mechanism-probe.json",
    ]
    return command, mounts, "output/mechanism-probe.json"


def _container_spec(
    *,
    stage: str,
    container_name: str,
    launch: Any,
    command: Sequence[str],
    mounts: Sequence[Mapping[str, Any]],
    image_environment: Mapping[str, str],
    gpu_device: str | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    required = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PAMS_CONTAINER_SOURCE_REVISION": launch.source_revision,
        "PAMS_CONTAINER_IMAGE_ID": launch.container_image_id,
    }
    if STAGE_METADATA[stage]["gpu"]:
        required.update(
            {
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "CUDA_VISIBLE_DEVICES": "0",
            }
        )
    expected_environment = {**dict(image_environment), **required}
    tmpfs = {
        "/tmp": "rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=1777,size=4g"
    }
    if stage == "mechanism":
        tmpfs["/pams/cache"] = (
            "rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=1777,size=4g"
        )
    metadata = STAGE_METADATA[stage]
    host_exact = {
        "NetworkMode": "none",
        "ReadonlyRootfs": True,
        "CapDrop": ["ALL"],
        "SecurityOpt": ["no-new-privileges:true"],
        "PidsLimit": 4096,
        "Memory": metadata["memory_bytes"],
        "NanoCpus": metadata["nano_cpus"],
        "IpcMode": "private",
        "PidMode": "",
        "UsernsMode": "",
        "Tmpfs": tmpfs,
        "AutoRemove": False,
    }
    return (
        {
            "schema_version": 2,
            "stage": stage,
            "container_name": container_name,
            "image_id": launch.container_image_id,
            "entrypoint": ["python"],
            "path": "python",
            "args": list(command),
            "user": "1000:1000",
            "working_dir": "/workspace",
            "expected_environment": expected_environment,
            "forbidden_environment": [
                "PAMS_DEV_ROOT",
                "PAMS_TEST_ROOT",
                "PAMS_TARGETS",
                "PYTHONOPTIMIZE",
                "PYTHONINSPECT",
                "PYTHONSTARTUP",
            ],
            "gpu_device": gpu_device,
            "allowed_exit_codes": metadata["allowed_exit_codes"],
            "host_config_exact": host_exact,
            "mounts": [dict(item) for item in mounts],
        },
        required,
    )


def _docker_create_arguments(
    *,
    stage: str,
    container_name: str,
    cidfile_path: Path,
    launch: Any,
    command: Sequence[str],
    mounts: Sequence[Mapping[str, Any]],
    required_environment: Mapping[str, str],
    gpu_device: str | None,
) -> list[str]:
    metadata = STAGE_METADATA[stage]
    arguments = [
        DOCKER_BINARY,
        "create",
        "--name",
        container_name,
        "--cidfile",
        os.fspath(cidfile_path),
        "--entrypoint",
        "python",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--pids-limit",
        "4096",
        "--memory",
        str(metadata["memory"]),
        "--cpus",
        str(metadata["cpus"]),
        "--restart",
        "no",
        "--ipc",
        "private",
        "--user",
        "1000:1000",
        "--workdir",
        "/workspace",
    ]
    if metadata["gpu"]:
        if gpu_device is None:
            raise LaunchFailure("GPU stage lacks a device reservation")
        arguments.extend(["--gpus", f"device={gpu_device}"])
    for key, value in sorted(required_environment.items()):
        arguments.extend(["--env", f"{key}={value}"])
    arguments.extend(
        [
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=1777,size=4g",
        ]
    )
    if stage == "mechanism":
        arguments.extend(
            [
                "--tmpfs",
                "/pams/cache:rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=1777,size=4g",
            ]
        )
    for mount in mounts:
        encoded = (
            f"type=bind,src={mount['source']},dst={mount['destination']},"
            "bind-propagation=rprivate"
        )
        if not mount["rw"]:
            encoded += ",readonly"
        arguments.extend(["--mount", encoded])
    arguments.extend([launch.container_image_id, *command])
    return arguments


def _normalized_container_id(encoded: bytes, *, role: str) -> str:
    try:
        value = encoded.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise LaunchFailure(f"{role} is not ASCII") from exc
    match = CONTAINER_ID_PATTERN.fullmatch(value)
    if match is None:
        raise LaunchFailure(f"{role} is malformed")
    return f"sha256:{match.group(1)}"


def _inspect_container(container_id: str) -> bytes:
    completed = _run([DOCKER_BINARY, "inspect", container_id])
    payload = _json_no_duplicates(completed.stdout, role="Docker inspect")
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], Mapping):
        raise LaunchFailure("Docker inspect root is malformed")
    if payload[0].get("Id") != container_id:
        raise LaunchFailure("Docker inspect returned a different container ID")
    return completed.stdout


def _write_bytes_exclusive(path: Path, encoded: bytes, *, mode: int = 0o440) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)


def _cleanup_exact_container(container_id: str | None) -> dict[str, Any]:
    if container_id is None:
        return {
            "create_succeeded": False,
            "id_verified_before_remove": False,
            "remove_succeeded": False,
            "absence_verified": False,
        }
    before = _run([DOCKER_BINARY, "inspect", container_id], check=False)
    if before.returncode != 0:
        raise LaunchFailure("created container disappeared before exact-ID cleanup")
    payload = _json_no_duplicates(before.stdout, role="cleanup Docker inspect")
    if (
        not isinstance(payload, list)
        or len(payload) != 1
        or not isinstance(payload[0], Mapping)
        or payload[0].get("Id") != container_id
    ):
        raise LaunchFailure("cleanup Docker inspect does not match the created ID")
    removed = _run([DOCKER_BINARY, "rm", "-f", container_id], check=False)
    if removed.returncode != 0:
        raise LaunchFailure("exact-ID Docker removal failed")
    if _normalized_container_id(removed.stdout, role="Docker removal ID") != container_id:
        raise LaunchFailure("Docker removal returned a different container ID")
    after = _run([DOCKER_BINARY, "inspect", container_id], check=False)
    absence_message = after.stderr.decode("utf-8", errors="replace").lower()
    if (
        after.returncode != 1
        or after.stdout.strip()
        or not any(
            marker in absence_message
            for marker in ("no such object", "no such container")
        )
    ):
        raise LaunchFailure("exact-ID container absence is not proven after removal")
    return {
        "create_succeeded": True,
        "id_verified_before_remove": True,
        "removed_by_exact_id": container_id,
        "remove_succeeded": True,
        "absence_verified": True,
    }


def _artifact_manifest(root: Path, output: Path) -> FileIdentity:
    rows: list[str] = []
    for path in sorted(
        (
            candidate
            for candidate in root.rglob("*")
            if candidate.is_file() and candidate != output
        ),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        identity = _identity(path, role=f"pre-receipt artifact {relative}")
        rows.append(f"{identity.sha256} {identity.bytes} {relative}\n")
    _write_bytes_exclusive(output, "".join(rows).encode("utf-8"))
    return _identity(output, role="pre-receipt artifact manifest")


def _seal_tree(root: Path) -> None:
    for candidate in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        candidate.chmod(0o555 if candidate.is_dir() else 0o444)
    root.chmod(0o555)
    for candidate in (root, *root.rglob("*")):
        if candidate.lstat().st_mode & 0o222:
            raise LaunchFailure("run tree is not fully sealed")


def _publish_tree(staging: Path, final_root: Path) -> None:
    if final_root.exists():
        raise LaunchFailure("final run root appeared before atomic publication")
    os.rename(staging, final_root)
    _fsync_directory(final_root.parent)


def _publish_outcome_registry(
    *,
    role: str,
    candidate_id: str | None,
    final_root: Path,
    output_relative: str,
    launch: Any,
) -> None:
    output = final_root.joinpath(*PurePosixPath(output_relative).parts)
    receipt = final_root / "audit/run.receipt.json"
    output_identity = _identity(output, role=f"{role} published output")
    receipt_identity = _identity(receipt, role=f"{role} published receipt")
    tree_sha, tree_bytes, tree_files = _tree_identity(final_root, role=f"{role} final tree")
    record = {
        "schema_version": 1,
        "artifact_type": f"pams_conventional_cycleback_{role}_outcome_registry_v1",
        "status": "passed",
        "candidate_id": candidate_id,
        "launch_registry_id": launch.registry_id,
        "source_revision": launch.source_revision,
        "container_image_id": launch.container_image_id,
        "root": os.fspath(final_root),
        "output_relative": output_relative,
        "receipt_relative": "audit/run.receipt.json",
        "output": output_identity.to_dict(),
        "receipt": receipt_identity.to_dict(),
        "tree": {
            "sha256": tree_sha,
            "bytes": tree_bytes,
            "files": tree_files,
        },
        "broader_authority_granted": False,
    }
    registry_path = _outcome_registry_path(role, candidate_id)
    _write_json_exclusive(registry_path, record, mode=0o444)


def _representation_predecessor(launch: Any) -> Predecessor:
    root = _validate_directory(
        Path(launch.representation_root),
        role="unified representation predecessor",
        sealed=True,
    )
    output = root / "gate-output/cycleback-input.authorization.json"
    receipt = root / "audit/run.receipt.json"
    output_identity = _identity(output, role="representation authorization")
    receipt_identity = _identity(receipt, role="representation receipt")
    if (
        output_identity.sha256
        != launch.representation_bindings["representation_authorization_sha256"]
        or receipt_identity.sha256
        != launch.representation_bindings["representation_run_receipt_sha256"]
    ):
        raise LaunchFailure("representation predecessor bytes differ from launch registry")
    tree_sha, tree_bytes, tree_files = _tree_identity(root, role="representation predecessor")
    return Predecessor(
        role="representation",
        root=root,
        output_path=output,
        output_identity=output_identity,
        receipt_path=receipt,
        receipt_identity=receipt_identity,
        tree_sha256=tree_sha,
        tree_bytes=tree_bytes,
        tree_files=tree_files,
    )


def _replace_mount_placeholders(
    mounts: Sequence[Mapping[str, Any]],
    *,
    source_export: Path,
    output_root: Path,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for original in mounts:
        row = dict(original)
        if row["source"] == "SOURCE_EXPORT":
            row["source"] = os.fspath(source_export)
        elif row["source"] == "OUTPUT_ROOT":
            row["source"] = os.fspath(output_root)
        result.append(row)
    return result


def _parse_post_exit(encoded: bytes, *, container_id: str) -> int:
    payload = _json_no_duplicates(encoded, role="post-run Docker inspect")
    if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], Mapping):
        raise LaunchFailure("post-run Docker inspect schema mismatch")
    if payload[0].get("Id") != container_id:
        raise LaunchFailure("post-run Docker inspect ID mismatch")
    state = payload[0].get("State")
    if not isinstance(state, Mapping) or not isinstance(state.get("ExitCode"), int):
        raise LaunchFailure("post-run Docker exit code is missing")
    return int(state["ExitCode"])


def _validate_stage_output(
    *,
    stage: str,
    output_path: Path,
    exit_code: int,
) -> tuple[Mapping[str, Any], FileIdentity, str]:
    payload, identity = _read_json(output_path, role=f"{stage} stage output")
    if not isinstance(payload, Mapping):
        raise LaunchFailure(f"{stage} stage output is not an object")
    if stage == "adapter":
        if exit_code != 0 or payload.get("cycleback_pose_input_authorized") is not True:
            raise LaunchFailure("pose-input adapter did not produce an authorization")
        return payload, identity, "passed"
    expected_status = "passed" if exit_code == 0 else "rejected"
    gate = payload.get("gate")
    if (
        payload.get("status") != expected_status
        or not isinstance(gate, Mapping)
        or gate.get("overall_pass") is not (exit_code == 0)
    ):
        raise LaunchFailure(f"{stage} output status differs from container exit")
    return payload, identity, expected_status


def _stage_receipt(
    *,
    stage: str,
    status: str,
    candidate_id: str | None,
    launch: Any,
    reservation_path: Path,
    reservation_identity: FileIdentity,
    config_relative: str | None,
    config_identity: Mapping[str, Any] | None,
    predecessors: Sequence[Predecessor],
    container_name: str,
    container_id: str,
    exit_code: int,
    cleanup: Mapping[str, Any],
    output_path: Path,
    output_identity: FileIdentity,
    output_payload: Mapping[str, Any],
    audit_root: Path,
    source_pre: FileIdentity,
    source_post: FileIdentity,
    artifact_manifest: FileIdentity,
) -> dict[str, Any]:
    artifact_type = {
        "adapter": "pams_conventional_cycleback_pose_input_authorization_run_receipt_v1",
        "geometry": "pams_conventional_cycleback_geometry_run_receipt_v1",
        "mechanism": "pams_conventional_cycleback_mechanism_run_receipt_v1",
    }[stage]
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": artifact_type,
        "stage": stage,
        "status": status,
        "candidate_id": candidate_id,
        "source_git_sha": launch.source_revision,
        "source_tree_sha": launch.source_tree_sha,
        "container_image_id": launch.container_image_id,
        "container_name": container_name,
        "container_id": container_id,
        "container_exit_code": exit_code,
        "container_cleanup": dict(cleanup),
        "stage_reservation_path": os.fspath(reservation_path),
        "stage_reservation_sha256": reservation_identity.sha256,
        "stage_reservation_bytes": reservation_identity.bytes,
        "stage_reservation_archive_sha256": _identity(
            audit_root / "stage.reservation.json", role="stage reservation archive"
        ).sha256,
        "launch_registry_id": launch.registry_id,
        "launch_authorization_sha256": launch.authorization_sha256,
        "launch_registry_receipt_sha256": launch.receipt_sha256,
        "config_relative": config_relative,
        "config": config_identity,
        "predecessors": [item.to_dict() for item in predecessors],
        "output_relative": output_path.relative_to(audit_root.parent).as_posix(),
        "output_sha256": output_identity.sha256,
        "output_bytes": output_identity.bytes,
        "create_id_sha256": _identity(
            audit_root / f"{container_name}.create-id.txt", role="create ID"
        ).sha256,
        "docker_create_cidfile_sha256": _identity(
            audit_root / f"{container_name}.docker-create.cid",
            role="Docker create CID file",
        ).sha256,
        "docker_create_stdout_sha256": _identity(
            audit_root / f"{container_name}.docker-create.stdout",
            role="Docker create stdout",
        ).sha256,
        "docker_create_stderr_sha256": _identity(
            audit_root / f"{container_name}.docker-create.stderr",
            role="Docker create stderr",
        ).sha256,
        "container_contract_sha256": _identity(
            audit_root / "container.contract.json", role="container contract"
        ).sha256,
        "pre_run_inspect_sha256": _identity(
            audit_root / f"{container_name}.pre-run.inspect.json",
            role="pre-run inspect",
        ).sha256,
        "post_run_inspect_sha256": _identity(
            audit_root / f"{container_name}.post-run.inspect.json",
            role="post-run inspect",
        ).sha256,
        "source_export_pre_manifest_sha256": source_pre.sha256,
        "source_export_post_manifest_sha256": source_post.sha256,
        "source_export_manifests_identical": source_pre == source_post,
        "pre_receipt_artifact_manifest_sha256": artifact_manifest.sha256,
        "pre_receipt_artifact_manifest_bytes": artifact_manifest.bytes,
        "scientific_authority_granted": False,
        "baseline_training_authorized": False,
        "full_training_authorized": False,
        "full_encoder_training_authorized": False,
        "epoch11_encoder_continuation_authorized": False,
        "development_evaluation_authorized": False,
        "sealed_evaluation_authorized": False,
    }
    if stage == "adapter":
        snapshot_path = output_path.parents[1] / "output/pose-cache-set.snapshot.json"
        snapshot_payload, snapshot_identity = _read_json(
            snapshot_path, role="adapter pose snapshot"
        )
        if not isinstance(snapshot_payload, Mapping):
            raise LaunchFailure("adapter pose snapshot is malformed")
        segment_identity = _identity(
            output_path.parents[1] / "output/segment-index.json",
            role="adapter segment index",
        )
        joint_snapshot_payload, joint_snapshot_identity = _read_json(
            output_path.parents[1] / "output/joint-mask-set.snapshot.json",
            role="adapter joint-mask snapshot",
        )
        if not isinstance(joint_snapshot_payload, Mapping):
            raise LaunchFailure("adapter joint-mask snapshot is malformed")
        identity_map_identity = _identity(
            output_path.parents[1] / "output/identity-map.json",
            role="adapter identity map",
        )
        pair_eligibility_identity = _identity(
            output_path.parents[1] / "output/cycleback-pair-eligibility.json",
            role="adapter pair eligibility",
        )
        receipt.update(
            pose_input_authorization_sha256=output_identity.sha256,
            pose_snapshot_sha256=snapshot_identity.sha256,
            pose_cache_set_sha256=snapshot_payload.get("fingerprint"),
            pose_cache_entry_count=snapshot_payload.get("entry_count"),
            segment_index_sha256=segment_identity.sha256,
            joint_mask_snapshot_sha256=joint_snapshot_identity.sha256,
            joint_mask_set_sha256=joint_snapshot_payload.get("fingerprint"),
            identity_map_sha256=identity_map_identity.sha256,
            cycleback_pair_eligibility_sha256=pair_eligibility_identity.sha256,
            representation_bindings=output_payload.get("bindings"),
        )
    elif stage == "geometry":
        receipt.update(
            geometry_gate_sha256=output_identity.sha256,
            mechanism_probe_authorized=(
                status == "passed"
                and output_payload.get("gate", {}).get("mechanism_probe_authorized") is True
            ),
            representation_bindings=output_payload.get("inputs", {}).get(
                "representation_bindings"
            ),
        )
    else:
        receipt.update(
            mechanism_probe_sha256=output_identity.sha256,
            cycleback_mechanism_supported=(
                status == "passed"
                and output_payload.get("gate", {}).get("cycleback_mechanism_supported")
                is True
            ),
            representation_bindings=output_payload.get("lineage", {}).get(
                "representation_bindings"
            ),
        )
    return receipt


def _failure_receipt(
    *,
    stage: str,
    phase: str,
    candidate_id: str | None,
    launch: Any | None,
    reservation_path: Path | None,
    reservation_identity: FileIdentity | None,
    config_relative: str | None,
    config_identity: Mapping[str, Any] | None,
    predecessors: Sequence[Predecessor],
    container_name: str | None,
    container_id: str | None,
    cleanup: Mapping[str, Any],
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_type": f"pams_conventional_cycleback_{stage}_failure_receipt_v1",
        "status": "failed",
        "stage": stage,
        "failed_phase": phase,
        "error_type": type(error).__name__,
        "candidate_id": candidate_id,
        "source_git_sha": None if launch is None else launch.source_revision,
        "source_tree_sha": None if launch is None else launch.source_tree_sha,
        "container_image_id": None if launch is None else launch.container_image_id,
        "launch_registry_id": None if launch is None else launch.registry_id,
        "launch_authorization_sha256": (
            None if launch is None else launch.authorization_sha256
        ),
        "launch_registry_receipt_sha256": (
            None if launch is None else launch.receipt_sha256
        ),
        "stage_reservation_path": (
            None if reservation_path is None else os.fspath(reservation_path)
        ),
        "stage_reservation": (
            None if reservation_identity is None else reservation_identity.to_dict()
        ),
        "config_relative": config_relative,
        "config": config_identity,
        "predecessors": [item.to_dict() for item in predecessors],
        "container_name": container_name,
        "container_id": container_id,
        "container_cleanup": dict(cleanup),
        "scientific_authority_granted": False,
        "baseline_training_authorized": False,
        "full_training_authorized": False,
        "full_encoder_training_authorized": False,
        "epoch11_encoder_continuation_authorized": False,
        "development_evaluation_authorized": False,
        "sealed_evaluation_authorized": False,
        "failure_receipt_write_policy": "o_excl_primary_then_o_excl_parent_fallback",
    }


def _write_failure_and_publish(
    *,
    stage: str,
    staging: Path | None,
    final_root: Path | None,
    reservation_path: Path | None,
    payload: Mapping[str, Any],
    container_absence_verified: bool,
) -> None:
    if not container_absence_verified:
        fallback_parent = (
            final_root.parent
            if final_root is not None
            else CANONICAL_ROOT / str(STAGE_METADATA[stage]["run_parent"])
        )
        fallback_parent.mkdir(parents=True, exist_ok=True)
        fallback_name = (
            f"{final_root.name}.failure.receipt.json"
            if final_root is not None
            else f"unreserved-{os.getpid()}.failure.receipt.json"
        )
        _write_json_exclusive(
            fallback_parent / fallback_name,
            {
                **dict(payload),
                "fallback": True,
                "staging_publication_forbidden_while_container_may_exist": True,
            },
            mode=0o444,
        )
        return
    if staging is not None and staging.exists() and final_root is not None:
        audit = staging / "audit"
        audit.mkdir(parents=True, exist_ok=True)
        manifest = audit / "pre-receipt-artifact.sha256"
        manifest_identity = (
            _identity(manifest, role="failure pre-receipt manifest")
            if manifest.exists()
            else _artifact_manifest(staging, manifest)
        )
        bound_payload = {
            **dict(payload),
            "pre_receipt_artifact_manifest_sha256": manifest_identity.sha256,
            "pre_receipt_artifact_manifest_bytes": manifest_identity.bytes,
        }
        primary = audit / "failure.receipt.json"
        try:
            _write_json_exclusive(primary, bound_payload)
        except OSError:
            fallback = final_root.with_name(f"{final_root.name}.failure.receipt.json")
            _write_json_exclusive(
                fallback,
                {
                    **bound_payload,
                    "fallback": True,
                    "staging_publication_forbidden_after_primary_write_failure": True,
                },
                mode=0o444,
            )
            return
        _seal_tree(staging)
        _publish_tree(staging, final_root)
        if reservation_path is not None and reservation_path.exists():
            reservation_path.chmod(0o444)
        return
    fallback_parent = (
        CANONICAL_ROOT / str(STAGE_METADATA[stage]["run_parent"])
    )
    fallback_parent.mkdir(parents=True, exist_ok=True)
    fallback = fallback_parent / f"unreserved-{os.getpid()}.failure.receipt.json"
    _write_json_exclusive(fallback, {**dict(payload), "fallback": True}, mode=0o444)


def launch_stage(
    *,
    stage: str,
    attempt_id: str,
    candidate_id: str | None,
    gpu_device: str | None,
) -> tuple[Path, int]:
    repository = Path(__file__).resolve().parents[2]
    phase = "launch-registry"
    launch: Any | None = None
    predecessors: list[Predecessor] = []
    config_relative: str | None = None
    config_identity: Mapping[str, Any] | None = None
    final_root: Path | None = None
    staging: Path | None = None
    reservation_path: Path | None = None
    reservation_identity: FileIdentity | None = None
    lock_descriptor: int | None = None
    stage_lock_descriptor: int | None = None
    outcome_lock_descriptor: int | None = None
    container_name: str | None = None
    container_id: str | None = None
    container_create_attempted = False
    cleanup: dict[str, Any] = {
        "create_succeeded": False,
        "id_verified_before_remove": False,
        "remove_succeeded": False,
        "absence_verified": False,
    }
    try:
        launch, _ = _load_launch_bootstrap(repository)
        phase = "stage-inputs"
        if stage == "adapter":
            if candidate_id is not None:
                raise LaunchFailure("adapter stage forbids a candidate ID")
            predecessors.append(_representation_predecessor(launch))
        else:
            if candidate_id not in CONFIGS:
                raise LaunchFailure("geometry/mechanism require a frozen candidate ID")
            config_relative = CONFIGS[candidate_id]
            config_identity = dict(launch.configs[config_relative])
            predecessors.append(
                _load_predecessor(role="pose_input", candidate_id=None, launch=launch)
            )
            if stage == "mechanism":
                predecessors.append(
                    _load_predecessor(
                        role="geometry",
                        candidate_id=candidate_id,
                        launch=launch,
                    )
                )
        outcome_lock_descriptor = _open_lock(
            _outcome_slot_lock_path(stage, candidate_id)
        )
        if stage == "adapter" and _outcome_registry_path(
            "pose_input", None
        ).exists():
            raise LaunchFailure("canonical pose-input outcome slot is already occupied")
        if stage == "geometry" and _outcome_registry_path(
            "geometry", candidate_id
        ).exists():
            raise LaunchFailure("canonical geometry outcome slot is already occupied")
        if STAGE_METADATA[stage]["gpu"]:
            if gpu_device is None or not gpu_device.isdigit():
                raise LaunchFailure("GPU device must be a decimal device ID")
            lock_descriptor = _open_lock(GPU_LOCK_ROOT / f"gpu{gpu_device}.lock")
        elif gpu_device is not None:
            raise LaunchFailure("CPU adapter stage forbids a GPU device")

        phase = "stage-reservation"
        (
            final_root,
            staging,
            reservation_path,
            reservation_identity,
            stage_lock_descriptor,
        ) = _reserve_stage(
            stage=stage,
            attempt_id=attempt_id,
            candidate_id=candidate_id,
            launch=launch,
            config_identity=config_identity,
            predecessors=predecessors,
        )
        if stage_lock_descriptor is not None:
            os.set_inheritable(stage_lock_descriptor, False)

        source_export = staging / "source"
        audit_root = staging / "audit"
        logs_root = staging / "logs"
        output_root = staging / ("artifact" if stage == "adapter" else "output")
        audit_root.mkdir()
        logs_root.mkdir()
        output_root.mkdir()
        _write_bytes_exclusive(
            audit_root / "stage.reservation.json",
            _stable_bytes(reservation_path, role="stage reservation archive source"),
        )
        phase = "source-export"
        _archive_source(repository, launch.source_revision, source_export)
        if _git_text(repository, "rev-parse", f"{launch.source_revision}^{{tree}}") != (
            launch.source_tree_sha
        ):
            raise LaunchFailure("source tree changed before container creation")
        source_pre = _write_source_manifest(
            source_export, audit_root / "source-export.pre.sha256"
        )
        if config_relative is not None:
            frozen_config = _identity(
                source_export / config_relative,
                role="frozen candidate config",
            )
            if frozen_config.to_dict() != config_identity:
                raise LaunchFailure("frozen candidate config differs from launch registry")

        phase = "container-contract"
        command, raw_mounts, output_relative = _stage_command(
            stage=stage,
            candidate_id=candidate_id,
            config_relative=config_relative,
            launch=launch,
            predecessors=predecessors,
        )
        mounts = _replace_mount_placeholders(
            raw_mounts,
            source_export=source_export,
            output_root=output_root,
        )
        container_name = (
            f"{STAGE_METADATA[stage]['container_prefix']}-"
            f"{launch.source_revision[:12]}-{attempt_id}"
        )
        image_environment = _image_environment(launch.container_image_id)
        spec, required_environment = _container_spec(
            stage=stage,
            container_name=container_name,
            launch=launch,
            command=command,
            mounts=mounts,
            image_environment=image_environment,
            gpu_device=gpu_device,
        )
        contract_path = audit_root / "container.contract.json"
        _write_json_exclusive(contract_path, spec)
        raw_cid_path = audit_root / f"{container_name}.docker-create.cid"
        create_stdout_path = audit_root / f"{container_name}.docker-create.stdout"
        create_stderr_path = audit_root / f"{container_name}.docker-create.stderr"
        create_id_path = audit_root / f"{container_name}.create-id.txt"
        pre_path = audit_root / f"{container_name}.pre-run.inspect.json"
        post_path = audit_root / f"{container_name}.post-run.inspect.json"
        log_path = logs_root / f"{container_name}.log"

        phase = "container-create"
        container_create_attempted = True
        create = _run(
            _docker_create_arguments(
                stage=stage,
                container_name=container_name,
                cidfile_path=raw_cid_path,
                launch=launch,
                command=command,
                mounts=mounts,
                required_environment=required_environment,
                gpu_device=gpu_device,
            ),
            check=False,
        )
        stdout_container_id: str | None = None
        cidfile_container_id: str | None = None
        with contextlib.suppress(LaunchFailure):
            stdout_container_id = _normalized_container_id(
                create.stdout,
                role="Docker create stdout ID",
            )
        if raw_cid_path.exists():
            cidfile_container_id = _normalized_container_id(
                _stable_bytes(raw_cid_path, role="Docker create CID file"),
                role="Docker create CID file",
            )
        recovered_container_id = cidfile_container_id or stdout_container_id
        if recovered_container_id is not None:
            container_id = recovered_container_id
            cleanup["create_succeeded"] = True
        _write_bytes_exclusive(create_stdout_path, create.stdout)
        _write_bytes_exclusive(create_stderr_path, create.stderr)
        if create.returncode != 0:
            raise LaunchFailure("Docker create failed")
        if (
            stdout_container_id is None
            or cidfile_container_id is None
            or stdout_container_id != cidfile_container_id
        ):
            raise LaunchFailure("Docker create ID sources are missing or inconsistent")
        raw_cid_path.chmod(0o440)
        _write_bytes_exclusive(create_id_path, f"{container_id}\n".encode("ascii"))
        pre_encoded = _inspect_container(container_id)
        _write_bytes_exclusive(pre_path, pre_encoded)
        validate_inspect(
            inspect_path=pre_path,
            create_id_path=create_id_path,
            spec_path=contract_path,
            phase="pre",
            expected_exit_code=None,
        )

        phase = "container-execution"
        attached = _run(
            [DOCKER_BINARY, "start", "--attach", container_id],
            check=False,
            stdout_path=log_path,
        )
        post_encoded = _inspect_container(container_id)
        _write_bytes_exclusive(post_path, post_encoded)
        exit_code = _parse_post_exit(post_encoded, container_id=container_id)
        if attached.returncode != exit_code:
            raise LaunchFailure("Docker attach exit differs from container state")
        validate_inspect(
            inspect_path=post_path,
            create_id_path=create_id_path,
            spec_path=contract_path,
            phase="post",
            expected_exit_code=exit_code,
        )
        if exit_code not in STAGE_METADATA[stage]["allowed_exit_codes"]:
            raise LaunchFailure("container exit is outside the stage contract")

        phase = "exact-container-cleanup"
        cleanup = _cleanup_exact_container(container_id)
        container_removed = container_id
        container_id = None
        for predecessor in predecessors:
            _revalidate_predecessor(predecessor)
        phase = "source-post-verification"
        source_post = _write_source_manifest(
            source_export, audit_root / "source-export.post.sha256"
        )
        if source_pre != source_post:
            raise LaunchFailure("frozen source export changed during execution")

        phase = "stage-output"
        output_path = staging.joinpath(*PurePosixPath(output_relative).parts)
        output_payload, output_identity, status = _validate_stage_output(
            stage=stage,
            output_path=output_path,
            exit_code=exit_code,
        )
        manifest = _artifact_manifest(
            staging, audit_root / "pre-receipt-artifact.sha256"
        )
        receipt = _stage_receipt(
            stage=stage,
            status=status,
            candidate_id=candidate_id,
            launch=launch,
            reservation_path=reservation_path,
            reservation_identity=reservation_identity,
            config_relative=config_relative,
            config_identity=config_identity,
            predecessors=predecessors,
            container_name=container_name,
            container_id=container_removed,
            exit_code=exit_code,
            cleanup=cleanup,
            output_path=output_path,
            output_identity=output_identity,
            output_payload=output_payload,
            audit_root=audit_root,
            source_pre=source_pre,
            source_post=source_post,
            artifact_manifest=manifest,
        )
        _write_json_exclusive(audit_root / "run.receipt.json", receipt)
        phase = "sealed-atomic-publication"
        _seal_tree(staging)
        _publish_tree(staging, final_root)
        reservation_path.chmod(0o444)
        if status == "passed" and stage in {"adapter", "geometry"}:
            _publish_outcome_registry(
                role="pose_input" if stage == "adapter" else "geometry",
                candidate_id=candidate_id,
                final_root=final_root,
                output_relative=output_relative,
                launch=launch,
            )
        return final_root, exit_code
    except BaseException as exc:
        if not isinstance(exc, Exception | KeyboardInterrupt):
            raise
        if container_id is not None:
            try:
                cleanup = _cleanup_exact_container(container_id)
                container_id = None
            except Exception as cleanup_error:
                cleanup = {
                    **cleanup,
                    "cleanup_error_type": type(cleanup_error).__name__,
                }
        failure = _failure_receipt(
            stage=stage,
            phase=phase,
            candidate_id=candidate_id,
            launch=launch,
            reservation_path=reservation_path,
            reservation_identity=reservation_identity,
            config_relative=config_relative,
            config_identity=config_identity,
            predecessors=predecessors,
            container_name=container_name,
            container_id=container_id,
            cleanup=cleanup,
            error=exc,
        )
        try:
            _write_failure_and_publish(
                stage=stage,
                staging=staging,
                final_root=final_root,
                reservation_path=reservation_path,
                payload=failure,
                container_absence_verified=(
                    not container_create_attempted
                    or cleanup.get("absence_verified") is True
                ),
            )
        except Exception as failure_error:
            print(
                f"cycleback failure evidence publication failed: {type(failure_error).__name__}",
                file=sys.stderr,
            )
        raise LaunchFailure(f"{stage} launch failed during {phase}: {type(exc).__name__}") from exc
    finally:
        if lock_descriptor is not None:
            os.close(lock_descriptor)
        if stage_lock_descriptor is not None:
            os.close(stage_lock_descriptor)
        if outcome_lock_descriptor is not None:
            os.close(outcome_lock_descriptor)


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_METADATA), required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--candidate-id", choices=tuple(CONFIGS))
    arguments = parser.parse_args(argv)
    arguments.attempt_id = _validate_attempt(arguments.attempt_id)
    return arguments


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = parse_arguments(argv)
        root, status = launch_stage(
            stage=arguments.stage,
            attempt_id=arguments.attempt_id,
            candidate_id=arguments.candidate_id,
            gpu_device="1" if arguments.stage != "adapter" else None,
        )
    except (OSError, TypeError, ValueError, LaunchFailure) as exc:
        print(f"cycleback secure launcher failed: {exc}", file=sys.stderr)
        return 2
    print(root)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
