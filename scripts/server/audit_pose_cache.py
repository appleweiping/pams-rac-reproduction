#!/usr/bin/env python3
"""Audit one exact, label-safe UCFRep pose-cache split.

The audit intentionally emits only aggregate, path-free JSON on stdout. Error
details are written to stderr by :func:`main`. Raw source detection counts are
not stored in pose-cache schema v2 or identity-ledger schema v2, so this script
reports that limitation instead of reconstructing or guessing those values.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pams.config import PAMSConfig, load_config
from pams.data import (
    PoseCacheEntryReceipt,
    PoseCacheSetSnapshot,
    UnlabeledVideoRecord,
    load_pose_cache_set,
    load_pose_cache_with_receipt,
    load_pose_input_commitment,
    load_pose_input_manifest,
    load_ucfrep_manifest,
    pose_cache_path,
    pose_input_identity_sha256,
)
from pams.reproducibility import sha256_file

_EXPECTED_SPLIT_SIZES = {"train": 421, "test": 105}
_TARGET_FRAMES = 256
_LEDGER_FIELDS = frozenset(
    {
        "schema_version",
        "input_kind",
        "protocol",
        "split",
        "input_file_sha256",
        "input_fingerprint",
        "sidecar_sha256",
        "sidecar_fingerprint",
        "commitment_file_sha256",
        "commitment_fingerprint",
        "identity_sha256",
        "pose_fingerprint",
        "successful_cache_snapshot",
        "selected",
        "completed",
        "extracted",
        "skipped",
        "failed",
        "failures",
    }
)
_FAILURE_FIELDS = frozenset({"video_id", "video_path", "error_type", "message"})
_PRIVILEGED_LABEL_KEYS = frozenset(
    {
        "source_manifest_file_sha256",
        "source_manifest_fingerprint",
        "sealed_dataset_fingerprint",
        "count",
        "action",
    }
)
_STABLE_STAT_FIELDS = ("st_dev", "st_ino", "st_size", "st_mtime_ns")


class AuditError(RuntimeError):
    """Raised when a cache-pool audit invariant fails."""


@dataclass(frozen=True, slots=True)
class CacheAuditEntry:
    """Path-free result of auditing one source video and pose cache."""

    receipt: PoseCacheEntryReceipt
    source_bytes: int
    source_inode: tuple[int, int]
    cached_frames: int
    cached_valid_frames: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _assert_label_free_payload(payload: Any) -> None:
    """Reject privileged label-derived keys anywhere in a label-free artifact."""

    if isinstance(payload, Mapping):
        leaked = _PRIVILEGED_LABEL_KEYS.intersection(payload)
        _require(not leaked, f"label-free payload contains privileged keys: {sorted(leaked)}")
        for value in payload.values():
            _assert_label_free_payload(value)
    elif isinstance(payload, list | tuple):
        for value in payload:
            _assert_label_free_payload(value)


def _reject_duplicate_json_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise AuditError(f"duplicate JSON field in identity ledger: {key!r}")
        payload[key] = value
    return payload


def _load_identity_ledger(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_fields,
        )
    except json.JSONDecodeError as exc:
        raise AuditError("identity ledger is not valid JSON") from exc
    _require(isinstance(payload, dict), "identity ledger root must be an object")
    return payload


def _ledger_integer(payload: Mapping[str, Any], field: str) -> int:
    value = payload[field]
    _require(
        type(value) is int and value >= 0,
        f"identity ledger field {field!r} must be a non-negative integer",
    )
    return value


def _validate_identity_ledger(
    payload: Mapping[str, Any],
    *,
    expected_size: int = 421,
    expected_bindings: Mapping[str, Any] | None = None,
    expected_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, int]:
    """Validate a clean, full-split ``--skip-existing`` identity ledger."""

    _require(
        set(payload) == _LEDGER_FIELDS,
        "identity ledger fields do not match schema_version=2",
    )
    counts = {
        field: _ledger_integer(payload, field)
        for field in (
            "schema_version",
            "selected",
            "completed",
            "extracted",
            "skipped",
            "failed",
        )
    }
    failures = payload["failures"]
    _require(isinstance(failures, list), "identity ledger failures must be a list")
    for index, failure in enumerate(failures):
        _require(isinstance(failure, dict), f"identity ledger failure {index} must be an object")
        _require(
            set(failure) == _FAILURE_FIELDS,
            f"identity ledger failure {index} fields do not match schema",
        )
        for field in _FAILURE_FIELDS:
            value = failure[field]
            _require(
                isinstance(value, str) and bool(value.strip()),
                f"identity ledger failure {index} field {field!r} must be non-empty",
            )

    _require(
        counts["schema_version"] == 2,
        "identity ledger schema_version must be 2; legacy ledgers are unbound diagnostics",
    )
    for field in (
        "input_kind",
        "protocol",
        "split",
        "input_file_sha256",
        "input_fingerprint",
        "identity_sha256",
        "pose_fingerprint",
    ):
        value = payload[field]
        _require(
            isinstance(value, str) and bool(value.strip()),
            f"identity ledger field {field!r} must be non-empty",
        )
    for field in (
        "input_file_sha256",
        "input_fingerprint",
        "identity_sha256",
        "pose_fingerprint",
    ):
        value = payload[field]
        _require(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"identity ledger field {field!r} must be a lowercase SHA-256",
        )
    input_kind = payload["input_kind"]
    _require(
        input_kind in {"label_free_sidecar", "labelled_manifest"},
        "identity ledger input_kind is unsupported",
    )
    sidecar_fields = (
        "sidecar_sha256",
        "sidecar_fingerprint",
        "commitment_file_sha256",
        "commitment_fingerprint",
    )
    if input_kind == "label_free_sidecar":
        for field in sidecar_fields:
            value = payload[field]
            _require(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value),
                f"label-free identity ledger field {field!r} must be a lowercase SHA-256",
            )
    else:
        _require(
            all(payload[field] is None for field in sidecar_fields),
            "labelled identity ledger must not claim sidecar commitment fields",
        )
    snapshot = payload["successful_cache_snapshot"]
    _require(
        isinstance(snapshot, dict),
        "clean identity ledger successful_cache_snapshot must be an object",
    )
    _require(
        counts["completed"] + counts["failed"] == counts["selected"],
        "identity ledger invariant completed + failed == selected failed",
    )
    _require(
        counts["extracted"] + counts["skipped"] == counts["completed"],
        "identity ledger invariant extracted + skipped == completed failed",
    )
    _require(
        counts["failed"] == len(failures),
        "identity ledger failed count does not match failures length",
    )
    _require(
        counts["selected"] == expected_size,
        f"identity ledger must select exactly {expected_size} videos",
    )
    _require(
        (
            counts["completed"],
            counts["extracted"],
            counts["skipped"],
            counts["failed"],
        )
        == (expected_size, 0, expected_size, 0)
        and not failures,
        "identity ledger is not a clean full-pool skip-existing validation",
    )
    if expected_bindings is not None:
        for field, expected in expected_bindings.items():
            _require(
                payload.get(field) == expected,
                f"identity ledger binding mismatch for {field!r}",
            )
    if expected_snapshot is not None:
        _require(
            snapshot == expected_snapshot,
            "identity ledger successful cache snapshot does not match audited caches",
        )
    return counts


def _stable_stat_matches(*snapshots: os.stat_result) -> bool:
    first = snapshots[0]
    return all(
        all(getattr(snapshot, field) == getattr(first, field) for field in _STABLE_STAT_FIELDS)
        for snapshot in snapshots[1:]
    )


def _stable_file_sha256(path: Path) -> tuple[str, int, tuple[int, int]]:
    """Hash a regular file and reject replacement or mutation during the read."""

    before = path.stat()
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        _require(stat.S_ISREG(opened.st_mode), "source video must be a regular file")
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
        closed = os.fstat(handle.fileno())
    after = path.stat()
    _require(
        _stable_stat_matches(before, opened, closed, after),
        "source video changed while it was being hashed",
    )
    _require(total == after.st_size, "source video byte count changed while hashing")
    return digest.hexdigest(), total, (after.st_dev, after.st_ino)


def _stable_file_bytes(path: Path) -> bytes:
    """Read a small regular file and reject replacement or mutation."""

    before = path.stat()
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        _require(stat.S_ISREG(opened.st_mode), "pose cache must be a regular file")
        payload = handle.read()
        closed = os.fstat(handle.fileno())
    after = path.stat()
    _require(
        _stable_stat_matches(before, opened, closed, after),
        "pose cache changed while it was being read",
    )
    _require(len(payload) == after.st_size, "pose cache byte count changed while reading")
    return payload


def _resolve_manifest_video_path(
    manifest_dir: Path,
    record: UnlabeledVideoRecord,
    *,
    portable_only: bool = False,
) -> Path:
    source = Path(record.video_path)
    if portable_only:
        _require(not source.is_absolute() and not source.anchor, "video locator must be relative")
    resolved = (source if source.is_absolute() else manifest_dir / source).resolve(strict=True)
    if portable_only:
        try:
            resolved.relative_to(manifest_dir)
        except ValueError as exc:
            raise AuditError("video locator escapes the selected video root") from exc
    return resolved


def _validate_raw_npz(
    payload: bytes,
    *,
    expected_frames: int,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            _require(
                set(archive.files) == {"xyz", "valid_mask", "metadata"},
                "pose cache NPZ members do not match schema v2",
            )
            xyz = np.asarray(archive["xyz"])
            valid_mask = np.asarray(archive["valid_mask"])
    except (OSError, ValueError) as exc:
        raise AuditError("pose cache is not a valid schema-v2 NPZ") from exc

    _require(xyz.dtype == np.float32, "pose cache xyz dtype must be float32")
    _require(valid_mask.dtype == np.bool_, "pose cache valid_mask dtype must be bool")
    _require(
        xyz.shape == (expected_frames, 33, 3),
        "pose cache xyz shape does not match the frozen configuration",
    )
    _require(
        valid_mask.shape == (expected_frames,),
        "pose cache valid_mask shape does not match the frozen configuration",
    )
    _require(bool(np.isfinite(xyz).all()), "pose cache contains non-finite raw coordinates")
    _require(
        bool(np.all(xyz[~valid_mask] == 0.0)),
        "pose cache has non-zero coordinates in an invalid frame",
    )
    if bool(valid_mask.any()):
        valid_xyz = xyz[valid_mask]
        _require(
            float(valid_xyz.min()) >= -2e-6 and float(valid_xyz.max()) <= 1.0 + 2e-6,
            "pose cache valid coordinates are outside the normalized [0, 1] range",
        )
        spans = np.ptp(valid_xyz, axis=(1, 2))
        _require(
            bool(np.all(spans > 1e-8)),
            "pose cache marks a zero-span normalized frame as valid",
        )
    return xyz, valid_mask


def _audit_cache_entry(
    record: UnlabeledVideoRecord,
    *,
    manifest_dir: Path,
    cache_dir: Path,
    config: PAMSConfig,
    portable_only: bool = False,
) -> CacheAuditEntry:
    """Validate one source/cache pair against manifest and pose identities."""

    _require(record.video_sha256 is not None, "pose-input record is missing video SHA-256")
    source = _resolve_manifest_video_path(
        manifest_dir,
        record,
        portable_only=portable_only,
    )
    source_sha256, source_bytes, source_inode = _stable_file_sha256(source)
    _require(source_sha256 == record.video_sha256, "source video SHA-256 mismatch")

    cache = pose_cache_path(cache_dir, record.video_id)
    _require(cache.exists(), "expected pose cache is missing")
    _require(not cache.is_symlink(), "pose cache must not be a symbolic link")
    payload = _stable_file_bytes(cache)
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    xyz, valid_mask = _validate_raw_npz(payload, expected_frames=config.data.frames)
    try:
        sequence, metadata, receipt = load_pose_cache_with_receipt(
            cache,
            expected_video_sha256=record.video_sha256,
            expected_pose_fingerprint=config.pose_fingerprint,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise AuditError("pose cache metadata or content identity validation failed") from exc

    _require(receipt.cache_sha256 == payload_sha256, "pose cache changed between audit reads")
    _require(receipt.bytes == len(payload), "pose cache receipt byte count mismatch")
    _require(
        metadata.video_id == sequence.video_id == record.video_id,
        "pose cache video_id does not match its manifest record",
    )
    _require(metadata.schema_version == 2, "pose cache metadata schema_version must be 2")
    _require(metadata.pose_model == config.pose.model_id, "pose cache model ID mismatch")
    _require(
        (metadata.frames, metadata.keypoints, metadata.coordinates)
        == (config.data.frames, config.data.keypoints, config.data.coordinates),
        "pose cache metadata shape does not match configuration",
    )
    _require(metadata.fps == sequence.fps, "pose cache FPS metadata mismatch")
    _require(
        np.array_equal(valid_mask, sequence.valid_mask),
        "raw and validated pose cache masks differ",
    )
    _require(np.array_equal(xyz, sequence.xyz), "raw and validated pose cache coordinates differ")
    return CacheAuditEntry(
        receipt=receipt,
        source_bytes=source_bytes,
        source_inode=source_inode,
        cached_frames=sequence.num_frames,
        cached_valid_frames=int(np.count_nonzero(sequence.valid_mask)),
    )


def _expected_cache_paths(cache_dir: Path, video_ids: Sequence[str]) -> set[Path]:
    paths = {pose_cache_path(cache_dir, video_id) for video_id in video_ids}
    _require(
        len(paths) == len(video_ids),
        "multiple video IDs map to the same pose-cache filename",
    )
    return paths


def _validate_cache_scope(
    cache_dir: Path,
    expected: set[Path],
    *,
    allow_extra: bool = False,
) -> tuple[int, int]:
    """Reject missing caches and, unless scoped auditing is requested, extras."""

    actual = {
        path
        for path in cache_dir.rglob("*")
        if path.suffix.lower() == ".npz" and (path.is_file() or path.is_symlink())
    }
    missing = expected - actual
    extra = actual - expected
    _require(not missing, f"pose-cache pool is missing {len(missing)} expected files")
    if not allow_extra:
        _require(not extra, f"pose-cache pool contains {len(extra)} extra NPZ files")
    return len(missing), len(extra)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 1 and array.size > 0, "distribution requires at least one value")
    quantiles = np.quantile(array, (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0))
    names = ("min", "p05", "p25", "p50", "p75", "p95", "max")
    return {name: float(value) for name, value in zip(names, quantiles, strict=True)}


def audit_pose_cache(
    manifest_path: Path,
    cache_dir: Path,
    config_path: Path,
    identity_ledger_path: Path,
    *,
    split: str = "train",
    label_free_manifest: bool = False,
    allow_extra_caches: bool = False,
    video_root: Path | None = None,
    input_commitment_path: Path | None = None,
) -> dict[str, Any]:
    """Audit and summarize one exact UCFRep pose-cache split."""

    manifest_path = manifest_path.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    identity_ledger_path = identity_ledger_path.resolve(strict=True)
    selected_split = split.strip().lower()
    _require(
        selected_split in _EXPECTED_SPLIT_SIZES,
        "pose-cache audit split must be train or test",
    )
    expected_size = _EXPECTED_SPLIT_SIZES[selected_split]
    manifest_file_sha256 = sha256_file(manifest_path)
    config_file_sha256 = sha256_file(config_path)
    identity_ledger_sha256 = sha256_file(identity_ledger_path)
    config = load_config(config_path)
    _require(config.protocol == "ucfrep_526", "configuration protocol must be ucfrep_526")
    _require(
        (
            config.data.frames,
            config.data.keypoints,
            config.data.coordinates,
        )
        == (_TARGET_FRAMES, 33, 3),
        "configuration must use the frozen 256x33x3 pose shape",
    )
    if label_free_manifest:
        pose_inputs = load_pose_input_manifest(manifest_path, validate_exact=True)
        commitment_path = (
            input_commitment_path.resolve(strict=True)
            if input_commitment_path is not None
            else manifest_path.with_name(f"{manifest_path.stem}.commitment.json").resolve(
                strict=True
            )
        )
        commitment = load_pose_input_commitment(commitment_path)
        commitment_file_sha256 = sha256_file(commitment_path)
        identity_sha256 = pose_input_identity_sha256(pose_inputs.records)
        _require(
            (
                commitment.protocol,
                commitment.split,
                commitment.record_total,
                commitment.identity_sha256,
                commitment.sidecar_sha256,
                commitment.sidecar_fingerprint,
            )
            == (
                pose_inputs.protocol,
                pose_inputs.split,
                len(pose_inputs.records),
                identity_sha256,
                manifest_file_sha256,
                pose_inputs.fingerprint,
            ),
            "pose-input commitment does not bind this exact label-free sidecar",
        )
        _require(
            pose_inputs.protocol == config.protocol,
            "pose-input/config protocol mismatch",
        )
        _require(
            pose_inputs.split == selected_split,
            "pose-input manifest split does not match the requested audit split",
        )
        records = pose_inputs.records
        manifest_fingerprint = pose_inputs.fingerprint
        training_fingerprint: str | None = None
        privileged_provenance: dict[str, str] = {}
        input_kind = "label_free_sidecar"
        sidecar_sha256: str | None = manifest_file_sha256
        sidecar_fingerprint: str | None = pose_inputs.fingerprint
        commitment_fingerprint: str | None = commitment.fingerprint
    else:
        _require(
            video_root is None and input_commitment_path is None,
            "--video-root and --input-commitment require --label-free-manifest",
        )
        _require(
            selected_split == "train",
            "test cache audit requires --label-free-manifest",
        )
        manifest = load_ucfrep_manifest(manifest_path, validate_exact=True)
        _require(manifest.protocol == config.protocol, "manifest/config protocol mismatch")
        _require(
            manifest.split_counts == {"train": 421, "test": 105},
            "manifest must be the canonical 421/105 UCFRep manifest",
        )
        records = manifest.training_records()
        manifest_fingerprint = manifest.fingerprint
        sealed_dataset_fingerprint = manifest.sealed_dataset_fingerprint
        source_manifest_file_sha256 = manifest_file_sha256
        source_manifest_fingerprint = manifest.fingerprint
        training_fingerprint = manifest.training_fingerprint()
        privileged_provenance = {
            "source_manifest_file_sha256": source_manifest_file_sha256,
            "source_manifest_fingerprint": source_manifest_fingerprint,
            "sealed_dataset_fingerprint": sealed_dataset_fingerprint,
        }
        identity_sha256 = pose_input_identity_sha256(
            tuple(
                UnlabeledVideoRecord(
                    video_id=record.video_id,
                    video_path=record.video_path,
                    video_sha256=record.video_sha256,
                )
                for record in records
            )
        )
        input_kind = "labelled_manifest"
        sidecar_sha256 = None
        sidecar_fingerprint = None
        commitment_file_sha256 = None
        commitment_fingerprint = None
    _require(
        len(records) == expected_size,
        f"{selected_split} pose-input manifest must contain {expected_size} rows",
    )
    video_ids = [record.video_id for record in records]
    _require(
        len(set(video_ids)) == expected_size,
        f"manifest {selected_split} video IDs must be unique",
    )
    video_sha256_values = [record.video_sha256 for record in records]
    _require(
        all(value is not None for value in video_sha256_values),
        f"every {selected_split} source video must have a manifest SHA-256",
    )
    _require(
        len(set(video_sha256_values)) == expected_size,
        f"{selected_split} source-video SHA-256 values must be unique",
    )

    ledger = _load_identity_ledger(identity_ledger_path)
    resolved_cache_dir = cache_dir.resolve(strict=True)
    _require(resolved_cache_dir.is_dir(), "pose-cache path must be a directory")
    expected_paths = _expected_cache_paths(resolved_cache_dir, video_ids)
    missing_pose_caches, extra_pose_caches = _validate_cache_scope(
        resolved_cache_dir,
        expected_paths,
        allow_extra=allow_extra_caches,
    )

    resolved_video_root = (
        video_root.resolve(strict=True)
        if label_free_manifest and video_root is not None
        else manifest_path.parent
    )
    entries = tuple(
        _audit_cache_entry(
            record,
            manifest_dir=resolved_video_root,
            cache_dir=resolved_cache_dir,
            config=config,
            portable_only=label_free_manifest,
        )
        for record in records
    )
    source_inodes = {entry.source_inode for entry in entries}
    _require(
        len(source_inodes) == expected_size,
        "multiple manifest records resolve to the same source-video inode",
    )
    cache_sha256_values = {entry.receipt.cache_sha256 for entry in entries}
    _require(
        len(cache_sha256_values) == expected_size,
        "multiple pose caches have the same byte-stream SHA-256",
    )
    receipts = tuple(entry.receipt for entry in entries)
    snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=receipts,
    )
    _, api_snapshot = load_pose_cache_set(
        records,
        cache_dir=resolved_cache_dir,
        pose_fingerprint=config.pose_fingerprint,
        materialize_sequences=False,
    )
    _require(
        snapshot.fingerprint == api_snapshot.fingerprint,
        "pose-cache snapshot disagrees with the shared cache-set loader",
    )
    ledger_counts = _validate_identity_ledger(
        ledger,
        expected_size=expected_size,
        expected_bindings={
            "input_kind": input_kind,
            "protocol": config.protocol,
            "split": selected_split,
            "input_file_sha256": manifest_file_sha256,
            "input_fingerprint": manifest_fingerprint,
            "sidecar_sha256": sidecar_sha256,
            "sidecar_fingerprint": sidecar_fingerprint,
            "commitment_file_sha256": commitment_file_sha256,
            "commitment_fingerprint": commitment_fingerprint,
            "identity_sha256": identity_sha256,
            "pose_fingerprint": config.pose_fingerprint,
        },
        expected_snapshot=snapshot.to_dict(),
    )
    _require(
        sha256_file(manifest_path) == manifest_file_sha256,
        "manifest changed during the audit",
    )
    _require(
        sha256_file(config_path) == config_file_sha256,
        "configuration changed during the audit",
    )
    _require(
        sha256_file(identity_ledger_path) == identity_ledger_sha256,
        "identity ledger changed during the audit",
    )
    if label_free_manifest:
        _require(
            sha256_file(commitment_path) == commitment_file_sha256,
            "pose-input commitment changed during the audit",
        )

    valid_counts = [entry.cached_valid_frames for entry in entries]
    valid_fractions = [value / config.data.frames for value in valid_counts]
    total_cached_frames = expected_size * config.data.frames
    total_cached_valid_frames = sum(valid_counts)
    sorted_ids = "\n".join(sorted(video_ids)) + "\n"
    execution_identity = {
        field: value
        for field, value in (
            ("source_git_sha", os.environ.get("PAMS_CONTAINER_SOURCE_REVISION")),
            ("container_image_id", os.environ.get("PAMS_CONTAINER_IMAGE_ID")),
            (
                "environment_sha256",
                os.environ.get("PAMS_CONTAINER_ENVIRONMENT_SHA256"),
            ),
            ("audit_mode", os.environ.get("PAMS_AUDIT_MODE")),
        )
        if value
    }
    provenance: dict[str, Any] = {
        "manifest_file_sha256": manifest_file_sha256,
        "manifest_fingerprint": manifest_fingerprint,
        f"{selected_split}_id_list_sha256": hashlib.sha256(
            sorted_ids.encode("utf-8")
        ).hexdigest(),
        "config_file_sha256": config_file_sha256,
        "pose_fingerprint": config.pose_fingerprint,
        "pose_model": config.pose.model_id,
        "pose_cache_set_sha256": snapshot.fingerprint,
        "identity_ledger_sha256": identity_ledger_sha256,
    }
    provenance.update(privileged_provenance)
    if training_fingerprint is not None:
        provenance["training_fingerprint"] = training_fingerprint
    result = {
        "schema_version": 1,
        "status": "passed",
        "scope": {
            "protocol": "ucfrep_526",
            "split": selected_split,
            "label_free_manifest": label_free_manifest,
            "expected_records": expected_size,
            "allow_extra_pose_caches": allow_extra_caches,
            "missing_pose_caches": missing_pose_caches,
            "extra_pose_caches": extra_pose_caches,
        },
        "provenance": provenance,
        "source_videos": {
            "files": expected_size,
            "unique_inodes": len(source_inodes),
            "unique_sha256": len(set(video_sha256_values)),
            "bytes": sum(entry.source_bytes for entry in entries),
        },
        "pose_caches": {
            "files": len(entries),
            "unique_sha256": len(cache_sha256_values),
            "bytes": sum(entry.receipt.bytes for entry in entries),
            "frames_total": total_cached_frames,
            "valid_frames_total": total_cached_valid_frames,
            "valid_fraction_micro": total_cached_valid_frames / total_cached_frames,
            "valid_fraction_per_video": _distribution(valid_fractions),
            "all_invalid_videos": sum(value == 0 for value in valid_counts),
        },
        "source_detection_frames": {
            "available": False,
            "coverage_videos": 0,
            "reason": "not_stored_in_pose_cache_schema_v2_or_identity_ledger_schema_v2",
        },
        "identity_ledger": {
            "schema_version": ledger_counts["schema_version"],
            **{
                field: ledger_counts[field]
                for field in ("selected", "completed", "extracted", "skipped", "failed")
            },
            "binds_manifest_or_pose_fingerprint": True,
            "binds_sidecar_and_commitment": label_free_manifest,
            "binds_successful_cache_snapshot": True,
            "historical_scope": "current_invocation_only",
        },
        "execution_identity": execution_identity,
    }
    if label_free_manifest:
        _assert_label_free_payload(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit one exact, label-safe UCFRep pose-cache split."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("identity_ledger", type=Path)
    parser.add_argument("--split", choices=tuple(_EXPECTED_SPLIT_SIZES), default="train")
    parser.add_argument(
        "--label-free-manifest",
        action="store_true",
        help="Load a pams data pose-inputs JSON file; required for split=test.",
    )
    parser.add_argument(
        "--allow-extra-caches",
        action="store_true",
        help="Audit the selected split inside a combined cache directory.",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        help="Remap portable label-free video locators beneath this root.",
    )
    parser.add_argument(
        "--input-commitment",
        type=Path,
        help="Independent pose-input receipt; defaults beside the sidecar.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = audit_pose_cache(
            args.manifest,
            args.cache_dir,
            args.config,
            args.identity_ledger,
            split=args.split,
            label_free_manifest=args.label_free_manifest,
            allow_extra_caches=args.allow_extra_caches,
            video_root=args.video_root,
            input_commitment_path=args.input_commitment,
        )
    except (AuditError, OSError, RuntimeError, ValueError) as exc:
        print(f"pose-cache audit failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            summary,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
