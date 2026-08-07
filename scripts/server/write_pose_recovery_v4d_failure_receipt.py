#!/usr/bin/env python3
"""Write a crash-tolerant, exclusive v4d wrapper failure receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


class FailureReceiptError(RuntimeError):
    """Raised when neither the primary nor fallback receipt can be created."""


def _sha256_if_readable(path: Path) -> str | None:
    try:
        if not path.is_file():
            return None
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _inspect_evidence(path: Path) -> dict[str, Any]:
    evidence: dict[str, Any] = {"status": "missing"}
    try:
        if not path.is_file():
            return evidence
        payload = path.read_bytes()
    except OSError as exc:
        evidence["status"] = "unreadable"
        evidence["error_type"] = type(exc).__name__
        return evidence
    evidence["sha256"] = hashlib.sha256(payload).hexdigest()
    evidence["bytes"] = len(payload)
    if not payload:
        evidence["status"] = "empty"
        return evidence
    try:
        value = json.loads(payload)
        if not isinstance(value, list) or not value or not isinstance(value[0], dict):
            raise TypeError("inspect root must contain an object")
        row = value[0]
        image = row.get("Image")
        state = row.get("State")
        if not isinstance(image, str) or not isinstance(state, dict):
            raise TypeError("inspect object lacks Image or State")
    except (IndexError, json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
        evidence["status"] = "corrupt"
        evidence["error_type"] = type(exc).__name__
        return evidence
    evidence["status"] = "valid"
    evidence["image_id"] = image
    evidence["oom_killed"] = bool(state.get("OOMKilled"))
    return evidence


def _write_json_o_excl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    descriptor = os.open(path, flags, 0o640)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def write_failure_receipt(
    *,
    kind: str,
    run_root: Path,
    fallback_output: Path,
    source_revision: str,
    container_image_id: str,
    failed_phase: str,
    exit_status: int,
    gpu_uuid: str = "unavailable",
    gpu_memory_used_mib: int = -1,
    gpu_utilization_percent: int = -1,
) -> Path:
    root = run_root.resolve()
    if kind == "authorization":
        inspect_paths = (
            "audit/container.inspect.pre.json",
            "audit/container.inspect.post.json",
        )
        artifact_paths = (
            *inspect_paths,
            "authorization/extraction.authorization.json",
            "audit/run.receipt.json",
        )
        artifact_type = (
            "pams_pose_recovery_v4d_full337_extraction_authorization_v2_failure"
        )
    elif kind == "full337":
        inspect_paths = (
            "audit/extract.inspect.pre.json",
            "audit/extract.inspect.post.json",
            "audit/gate.inspect.pre.json",
            "audit/gate.inspect.post.json",
        )
        artifact_paths = (
            *inspect_paths,
            "ledgers/train337.json",
            "gate-output/full337-gate.json",
            "gate-output/training.authorization.json",
            "gate-output/training.denial.json",
            "audit/run.receipt.json",
        )
        artifact_type = "pams_pose_recovery_v4d_full337_failure_receipt"
    else:
        raise FailureReceiptError(f"unsupported failure receipt kind: {kind}")

    inspect_evidence = {
        relative: _inspect_evidence(root / relative) for relative in inspect_paths
    }
    artifacts = {
        relative: digest
        for relative in artifact_paths
        if (digest := _sha256_if_readable(root / relative)) is not None
    }
    observed_images = sorted(
        {
            str(evidence["image_id"])
            for evidence in inspect_evidence.values()
            if evidence.get("status") == "valid"
        }
    )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "artifact_type": artifact_type,
        "source_revision": source_revision,
        "container_image_id": container_image_id,
        "failed_phase": failed_phase,
        "exit_status": exit_status,
        "baseline_training_authorized": False,
        "artifacts": artifacts,
        "inspect_evidence": inspect_evidence,
        "observed_container_image_ids": observed_images,
        "failure_receipt_write_policy": "o_excl_primary_then_o_excl_parent_fallback",
        "run_root_sealing_required": True,
        "completed_run_receipt_present": "audit/run.receipt.json" in artifacts,
        "sealing_failure_after_completed_run_receipt": (
            failed_phase == "run-root-sealing" and "audit/run.receipt.json" in artifacts
        ),
    }
    if kind == "full337":
        extract_post = inspect_evidence["audit/extract.inspect.post.json"]
        payload.update(
            {
                "oom_killed": bool(extract_post.get("oom_killed", False)),
                "oom_evidence_available": extract_post.get("status") == "valid",
                "physical_gpu_index": 1,
                "gpu_uuid": gpu_uuid,
                "gpu_memory_used_mib_preflight": gpu_memory_used_mib,
                "gpu_utilization_percent_preflight": gpu_utilization_percent,
            }
        )

    failures: list[str] = []
    targets = (
        (root / "audit/failure.receipt.json", "run_root_audit"),
        (fallback_output.resolve(), "run_parent_fallback"),
    )
    for target, location in targets:
        candidate = {**payload, "receipt_location_scope": location}
        try:
            _write_json_o_excl(target, candidate)
        except OSError as exc:
            failures.append(f"{target}: {type(exc).__name__}")
            continue
        return target
    raise FailureReceiptError("; ".join(failures))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("authorization", "full337"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--fallback-output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--failed-phase", required=True)
    parser.add_argument("--exit-status", type=int, required=True)
    parser.add_argument("--gpu-uuid", default="unavailable")
    parser.add_argument("--gpu-memory-used-mib", type=int, default=-1)
    parser.add_argument("--gpu-utilization-percent", type=int, default=-1)
    args = parser.parse_args(argv)
    try:
        output = write_failure_receipt(
            kind=args.kind,
            run_root=args.run_root,
            fallback_output=args.fallback_output,
            source_revision=args.source_revision,
            container_image_id=args.container_image_id,
            failed_phase=args.failed_phase,
            exit_status=args.exit_status,
            gpu_uuid=args.gpu_uuid,
            gpu_memory_used_mib=args.gpu_memory_used_mib,
            gpu_utilization_percent=args.gpu_utilization_percent,
        )
    except (FailureReceiptError, OSError, ValueError) as exc:
        print(f"v4d failure receipt could not be written: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
