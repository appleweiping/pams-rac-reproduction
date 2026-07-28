"""Reproducibility, hashing, and hardware helpers."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import re
import subprocess
from pathlib import Path
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
        torch.use_deterministic_algorithms(True, warn_only=True)
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


def clean_git_revision(cwd: str | Path | None = None) -> str:
    """Return HEAD only when tracked and untracked source state is clean."""

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
