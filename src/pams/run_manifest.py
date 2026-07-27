"""Immutable experiment manifests."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from pams.reproducibility import git_revision, hardware_fingerprint, sha256_json


class RunManifest(BaseModel):
    """Machine-readable provenance record for one training or evaluation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = 1
    run_id: str
    created_at_utc: str
    command: list[str]
    git_sha: str
    config_sha256: str
    dataset_sha256: str
    seed: int
    protocol: str
    status: str = "created"
    hardware: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)

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


def write_manifest_exclusive(manifest: RunManifest, path: str | Path) -> Path:
    """Write once, refusing to overwrite an existing provenance record."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable manifest: {target}")
    payload = json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=target.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.write("\n")
    try:
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target
