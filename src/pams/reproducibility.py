"""Reproducibility, hashing, and hardware helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import stat
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch without mutating unrelated environment state."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        if torch.cuda.is_available():
            # TransformerEncoder may otherwise select fused SDPA kernels that
            # PyTorch explicitly marks non-deterministic on CUDA.
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def fsync_directory(path: str | Path) -> None:
    """Persist directory-entry changes on POSIX filesystems.

    Windows does not expose a portable directory ``fsync`` through
    :mod:`os`.  Formal experiments run on Linux; keeping the call a no-op on
    Windows also makes local development behave predictably.
    """

    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_mkdir(path: str | Path) -> Path:
    """Create a directory hierarchy and persist each new parent entry."""

    target = Path(path)
    missing: list[Path] = []
    cursor = target
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    if cursor.exists() and not cursor.is_dir():
        raise NotADirectoryError(f"directory ancestor is not a directory: {cursor}")
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise
        else:
            fsync_directory(directory.parent)
    if not target.is_dir():
        raise NotADirectoryError(f"path is not a directory: {target}")
    return target


def git_revision(cwd: str | Path | None = None) -> str:
    """Return the checked-out commit or ``uncommitted`` outside a repository."""

    receipt_value = os.environ.get("PAMS_SOURCE_EXPORT_RECEIPT", "").strip()
    receipt_sha256 = os.environ.get(
        "PAMS_SOURCE_EXPORT_RECEIPT_SHA256",
        "",
    ).strip()
    if bool(receipt_value) != bool(receipt_sha256):
        raise RuntimeError("source-export receipt environment is incomplete")
    if receipt_value:
        return _clean_source_export_revision(
            cwd,
            receipt_value=receipt_value,
            expected_receipt_sha256=receipt_sha256,
        )
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "uncommitted"
    return result.stdout.strip()


def _stable_regular_file_digest(path: Path) -> tuple[str, int]:
    """Hash one regular file while rejecting replacement or mutation."""

    if path.is_symlink():
        raise RuntimeError(f"source export contains a symlink: {path}")
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"source export entry is not a regular file: {path}")
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
        closed = os.fstat(handle.fileno())
    after = path.stat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(closed, field)
        or getattr(closed, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise RuntimeError(f"source export changed while hashing: {path}")
    if size != after.st_size:
        raise RuntimeError(f"source export byte count changed while hashing: {path}")
    return digest.hexdigest(), size


def _clean_source_export_revision(
    cwd: str | Path | None,
    *,
    receipt_value: str,
    expected_receipt_sha256: str,
) -> str:
    """Validate an exact, Git-object-free source export."""

    if not _SHA256_PATTERN.fullmatch(expected_receipt_sha256):
        raise RuntimeError("source-export receipt SHA-256 is invalid")
    root = Path.cwd() if cwd is None else Path(cwd)
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise RuntimeError("source-export root is not a directory")
    receipt_path = Path(receipt_value)
    if not receipt_path.is_absolute():
        receipt_path = root / receipt_path
    receipt_path = receipt_path.resolve(strict=True)
    receipt_sha256, _ = _stable_regular_file_digest(receipt_path)
    if receipt_sha256 != expected_receipt_sha256:
        raise RuntimeError("source-export receipt SHA-256 mismatch")
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("source-export receipt is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "source_revision",
        "root",
        "files",
    }:
        raise RuntimeError("source-export receipt has an invalid schema")
    if payload["schema_version"] != 1 or payload["root"] != ".":
        raise RuntimeError("source-export receipt version or root is unsupported")
    source_revision = payload["source_revision"]
    if not isinstance(source_revision, str) or not _GIT_SHA_PATTERN.fullmatch(
        source_revision
    ):
        raise RuntimeError("source-export revision is not a Git SHA")
    container_revision = os.environ.get(
        "PAMS_CONTAINER_SOURCE_REVISION",
        "",
    ).strip()
    if source_revision != container_revision:
        raise RuntimeError(
            "source-export revision does not match container source revision"
        )
    raw_files = payload["files"]
    if not isinstance(raw_files, list) or not raw_files:
        raise RuntimeError("source-export receipt requires at least one file")

    expected: dict[str, tuple[str, int]] = {}
    ordered_paths: list[str] = []
    for raw in raw_files:
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256", "bytes"}:
            raise RuntimeError("source-export file receipt has an invalid schema")
        relative = raw["path"]
        digest = raw["sha256"]
        byte_count = raw["bytes"]
        if not isinstance(relative, str) or "\\" in relative:
            raise RuntimeError("source-export path must be a POSIX relative path")
        posix = PurePosixPath(relative)
        if (
            not relative
            or posix.is_absolute()
            or any(part in {"", ".", ".."} for part in posix.parts)
        ):
            raise RuntimeError("source-export path is unsafe")
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise RuntimeError("source-export file SHA-256 is invalid")
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
        ):
            raise RuntimeError("source-export file byte count is invalid")
        if relative in expected:
            raise RuntimeError("source-export receipt contains duplicate paths")
        expected[relative] = (digest, byte_count)
        ordered_paths.append(relative)
    if ordered_paths != sorted(ordered_paths):
        raise RuntimeError("source-export receipt paths are not sorted")

    actual_paths: set[str] = set()
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise RuntimeError(f"source export contains a symlink: {entry}")
        if entry.is_dir():
            continue
        if not entry.is_file():
            raise RuntimeError(f"source export contains a non-file entry: {entry}")
        resolved = entry.resolve(strict=True)
        if resolved == receipt_path:
            continue
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError:
            raise RuntimeError("source-export entry escapes its root") from None
        actual_paths.add(relative)
    if actual_paths != set(expected):
        missing = sorted(set(expected) - actual_paths)
        extra = sorted(actual_paths - set(expected))
        raise RuntimeError(
            "source-export file set mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    for relative in ordered_paths:
        expected_digest, expected_bytes = expected[relative]
        source = root.joinpath(*PurePosixPath(relative).parts)
        actual_digest, actual_bytes = _stable_regular_file_digest(source)
        if actual_digest != expected_digest or actual_bytes != expected_bytes:
            raise RuntimeError(f"source-export file mismatch: {relative}")
    final_receipt_sha256, _ = _stable_regular_file_digest(receipt_path)
    if final_receipt_sha256 != expected_receipt_sha256:
        raise RuntimeError("source-export receipt changed during validation")
    return source_revision


def clean_git_revision(cwd: str | Path | None = None) -> str:
    """Return HEAD only when tracked and untracked source state is clean."""

    receipt_value = os.environ.get("PAMS_SOURCE_EXPORT_RECEIPT", "").strip()
    receipt_sha256 = os.environ.get(
        "PAMS_SOURCE_EXPORT_RECEIPT_SHA256",
        "",
    ).strip()
    if bool(receipt_value) != bool(receipt_sha256):
        raise RuntimeError("source-export receipt environment is incomplete")
    if receipt_value:
        return _clean_source_export_revision(
            cwd,
            receipt_value=receipt_value,
            expected_receipt_sha256=receipt_sha256,
        )

    revision = git_revision(cwd)
    if revision == "uncommitted":
        raise RuntimeError("a committed Git checkout is required")
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("unable to audit Git worktree cleanliness") from exc
    if result.stdout.strip():
        raise RuntimeError(
            "sealed evaluation requires a clean Git worktree; "
            "commit or remove tracked/untracked source changes first"
        )
    return revision


def _nvidia_driver_inventory() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"status": "no_cuda_device", "versions": []}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {"status": "unavailable", "versions": []}
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(versions) != torch.cuda.device_count():
        return {"status": "unavailable", "versions": []}
    return {"status": "available", "versions": versions}


def hardware_fingerprint() -> dict[str, Any]:
    """Collect non-secret runtime details needed to interpret measurements."""

    gpus: list[dict[str, Any]] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory_bytes": props.total_memory,
                    "capability": f"{props.major}.{props.minor}",
                }
            )
    fingerprint: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nvidia_driver": _nvidia_driver_inventory(),
        "gpus": gpus,
    }
    image_id = os.environ.get("PAMS_CONTAINER_IMAGE_ID", "").strip()
    environment_sha256 = os.environ.get(
        "PAMS_CONTAINER_ENVIRONMENT_SHA256",
        "",
    ).strip()
    source_revision = os.environ.get("PAMS_CONTAINER_SOURCE_REVISION", "").strip()
    supplied = (image_id, environment_sha256, source_revision)
    if any(supplied):
        if not all(supplied):
            raise RuntimeError("container provenance environment is incomplete")
        if not _IMAGE_ID_PATTERN.fullmatch(image_id):
            raise RuntimeError("PAMS_CONTAINER_IMAGE_ID is not an immutable SHA-256 image ID")
        if not _SHA256_PATTERN.fullmatch(environment_sha256):
            raise RuntimeError("container environment fingerprint is not a SHA-256")
        if not _GIT_SHA_PATTERN.fullmatch(source_revision):
            raise RuntimeError("container source revision is not a Git SHA")
        fingerprint["container"] = {
            "image_id": image_id,
            "environment_sha256": environment_sha256,
            "source_revision": source_revision,
        }
    return fingerprint
