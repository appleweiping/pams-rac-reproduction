#!/usr/bin/env python3
"""Compare the frozen UCFRep v2 and v3 pose caches without loading labels.

Only the preregistered train337/dev84 pose-input sidecars, pose-cache files,
and the two v3 extraction ledgers are accepted.  Test identities, action
classes, counts, target files, and source videos are neither arguments nor
inputs to this audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pams.data import (
    PoseCacheEntryReceipt,
    PoseCacheSetSnapshot,
    PoseInputManifest,
    UnlabeledVideoRecord,
    load_pose_cache_with_receipt,
    load_pose_input_manifest,
    pose_cache_path,
    pose_input_identity_sha256,
)

_EXPECTED_SPLIT_SIZES = {"train": 337, "dev": 84}
_EXPECTED_TOTAL = 421
_EXPECTED_CACHED_FRAMES = 256
_EXTREMELY_SHORT_MAX_FRAMES = 4
_FROZEN_V2_POSE_FINGERPRINT = (
    "8ecb6c1384e1d6a088762322ed90bb6aa631683318e0ab7ed21c19541e5f6656"
)
_FROZEN_V3_POSE_FINGERPRINT = (
    "3dd0388320095796f42aa82d904073b6478b073ed351394ef8d03c30798a0116"
)
_FORBIDDEN_KEYS = frozenset(
    {
        "action",
        "count",
        "ground_truth",
        "ground_truth_count",
        "gt",
        "label",
        "labels",
        "target",
        "targets",
    }
)
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
        "caches",
        "failures",
    }
)
_CACHE_SUMMARY_FIELDS = frozenset(
    {
        "video_id",
        "video_path",
        "cache_path",
        "video_sha256",
        "pose_fingerprint",
        "source_frames",
        "source_valid_frames",
        "selected_source_frames",
        "cached_frames",
        "cached_valid_frames",
        "fps",
        "pose_model",
        "skipped",
    }
)


class ComparisonAuditError(RuntimeError):
    """Raised when a label-free comparison invariant fails."""


@dataclass(frozen=True, slots=True)
class CacheObservation:
    """One validated pose-cache receipt and its frame-validity summary."""

    video_id: str
    receipt: PoseCacheEntryReceipt
    valid_frames: int
    total_frames: int
    pose_model: str

    @property
    def valid_rate(self) -> float:
        return self.valid_frames / self.total_frames


@dataclass(frozen=True, slots=True)
class V3SourceObservation:
    """Label-free extraction diagnostics copied from one validated v3 ledger."""

    video_id: str
    source_frames: int
    source_valid_frames: int
    selected_source_frames: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonAuditError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_non_finite(value: str) -> None:
    raise ComparisonAuditError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_or_privileged_fields(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ComparisonAuditError(f"duplicate JSON field is forbidden: {key!r}")
        if key.casefold() in _FORBIDDEN_KEYS:
            raise ComparisonAuditError(
                f"label-free audit input contains forbidden field {key!r}"
            )
        payload[key] = value
    return payload


def _load_label_free_json(path: Path, *, document: str) -> tuple[dict[str, Any], str]:
    """Load one strict JSON object while rejecting label/target fields."""

    source = path.resolve(strict=True)
    _require(source.is_file(), f"{document} must be a regular file")
    try:
        encoded = source.read_bytes()
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_or_privileged_fields,
            parse_constant=_reject_non_finite,
        )
    except UnicodeDecodeError as exc:
        raise ComparisonAuditError(f"{document} must be UTF-8 JSON") from exc
    except json.JSONDecodeError as exc:
        raise ComparisonAuditError(f"{document} is not valid JSON") from exc
    _require(isinstance(payload, dict), f"{document} root must be an object")
    return payload, hashlib.sha256(encoded).hexdigest()


def _load_sidecar(path: Path, *, split: str) -> tuple[PoseInputManifest, str]:
    """Load a preregistered sidecar and preserve its exact file digest."""

    _, digest = _load_label_free_json(path, document=f"{split} sidecar")
    try:
        manifest = load_pose_input_manifest(path, validate_exact=True)
    except (OSError, TypeError, ValueError) as exc:
        raise ComparisonAuditError(
            f"{split} sidecar failed exact label-free validation"
        ) from exc
    _require(manifest.protocol == "ucfrep_526", f"{split} protocol must be ucfrep_526")
    _require(manifest.split == split, f"{split} sidecar declares a different split")
    _require(
        len(manifest.records) == _EXPECTED_SPLIT_SIZES[split],
        f"{split} sidecar must contain exactly {_EXPECTED_SPLIT_SIZES[split]} records",
    )
    _require(
        _sha256_file(path.resolve(strict=True)) == digest,
        f"{split} sidecar changed while it was being validated",
    )
    return manifest, digest


def _validate_sidecar_pair(
    train: PoseInputManifest,
    dev: PoseInputManifest,
) -> tuple[UnlabeledVideoRecord, ...]:
    train_ids = {record.video_id for record in train.records}
    dev_ids = {record.video_id for record in dev.records}
    _require(not train_ids.intersection(dev_ids), "train and dev sidecars overlap")
    combined = tuple(train.records) + tuple(dev.records)
    _require(len(combined) == _EXPECTED_TOTAL, "train/dev union must contain 421 records")
    _require(
        len({record.video_id for record in combined}) == _EXPECTED_TOTAL,
        "train/dev union has duplicate video IDs",
    )
    _require(
        len({record.video_sha256 for record in combined}) == _EXPECTED_TOTAL,
        "train/dev union has missing or duplicate source-video SHA-256 values",
    )
    return combined


def _non_negative_integer(value: Any, *, field: str) -> int:
    _require(
        type(value) is int and value >= 0,
        f"{field} must be a non-negative integer",
    )
    return value


def _positive_integer(value: Any, *, field: str) -> int:
    result = _non_negative_integer(value, field=field)
    _require(result > 0, f"{field} must be positive")
    return result


def _lowercase_sha256(value: Any, *, field: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{field} must be a lowercase SHA-256",
    )
    return value


def _load_cache_observations(
    records: Sequence[UnlabeledVideoRecord],
    *,
    cache_root: Path,
    expected_pose_fingerprint: str,
    cache_revision: str,
) -> tuple[CacheObservation, ...]:
    resolved_root = cache_root.resolve(strict=True)
    _require(resolved_root.is_dir(), f"{cache_revision} cache root must be a directory")
    observations: list[CacheObservation] = []
    for record in records:
        _require(
            record.video_sha256 is not None,
            f"{cache_revision} record {record.video_id!r} lacks a video SHA-256",
        )
        cache_path = pose_cache_path(resolved_root, record.video_id)
        _require(
            cache_path.exists() and cache_path.is_file(),
            f"{cache_revision} cache is missing for {record.video_id!r}",
        )
        _require(
            not cache_path.is_symlink(),
            f"{cache_revision} cache must not be a symlink for {record.video_id!r}",
        )
        try:
            sequence, metadata, receipt = load_pose_cache_with_receipt(
                cache_path,
                expected_video_sha256=record.video_sha256,
                expected_pose_fingerprint=expected_pose_fingerprint,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ComparisonAuditError(
                f"{cache_revision} cache validation failed for {record.video_id!r}"
            ) from exc
        _require(
            sequence.video_id == metadata.video_id == record.video_id,
            f"{cache_revision} cache video ID mismatch for {record.video_id!r}",
        )
        _require(
            sequence.num_frames == metadata.frames == _EXPECTED_CACHED_FRAMES,
            f"{cache_revision} cache must contain 256 frames for {record.video_id!r}",
        )
        observations.append(
            CacheObservation(
                video_id=record.video_id,
                receipt=receipt,
                valid_frames=int(np.count_nonzero(sequence.valid_mask)),
                total_frames=sequence.num_frames,
                pose_model=metadata.pose_model,
            )
        )
    return tuple(observations)


def _snapshot(
    observations: Sequence[CacheObservation],
    *,
    pose_fingerprint: str,
) -> PoseCacheSetSnapshot:
    return PoseCacheSetSnapshot(
        pose_fingerprint=pose_fingerprint,
        entries=tuple(observation.receipt for observation in observations),
    )


def _validate_v3_ledger(
    path: Path,
    *,
    split: str,
    sidecar: PoseInputManifest,
    sidecar_sha256: str,
    cache_observations: Sequence[CacheObservation],
    expected_pose_fingerprint: str,
) -> tuple[V3SourceObservation, ...]:
    payload, ledger_sha256 = _load_label_free_json(path, document=f"{split} v3 ledger")
    _require(set(payload) == _LEDGER_FIELDS, f"{split} v3 ledger fields mismatch")
    expected_size = _EXPECTED_SPLIT_SIZES[split]
    for field in ("schema_version", "selected", "completed", "extracted", "skipped", "failed"):
        _non_negative_integer(payload[field], field=f"{split} ledger {field}")
    _require(payload["schema_version"] == 2, f"{split} ledger schema_version must be 2")
    _require(payload["input_kind"] == "label_free_sidecar", f"{split} ledger is not label-free")
    _require(payload["protocol"] == "ucfrep_526", f"{split} ledger protocol mismatch")
    _require(payload["split"] == split, f"{split} ledger split mismatch")
    _require(
        payload["input_file_sha256"] == sidecar_sha256
        and payload["sidecar_sha256"] == sidecar_sha256,
        f"{split} ledger does not bind the exact sidecar bytes",
    )
    _require(
        payload["input_fingerprint"] == sidecar.fingerprint
        and payload["sidecar_fingerprint"] == sidecar.fingerprint,
        f"{split} ledger sidecar fingerprint mismatch",
    )
    _lowercase_sha256(
        payload["commitment_file_sha256"],
        field=f"{split} ledger commitment_file_sha256",
    )
    _lowercase_sha256(
        payload["commitment_fingerprint"],
        field=f"{split} ledger commitment_fingerprint",
    )
    _require(
        payload["identity_sha256"] == pose_input_identity_sha256(sidecar.records),
        f"{split} ledger identity SHA-256 mismatch",
    )
    _require(
        payload["pose_fingerprint"] == expected_pose_fingerprint,
        f"{split} ledger v3 pose fingerprint mismatch",
    )
    _require(
        payload["selected"] == payload["completed"] == expected_size,
        f"{split} ledger is incomplete",
    )
    _require(
        payload["extracted"] + payload["skipped"] == expected_size,
        f"{split} ledger extracted/skipped invariant failed",
    )
    _require(payload["failed"] == 0, f"{split} ledger contains extraction failures")
    _require(payload["failures"] == [], f"{split} ledger failure list is not empty")

    summaries = payload["caches"]
    _require(isinstance(summaries, list), f"{split} ledger caches must be a list")
    _require(len(summaries) == expected_size, f"{split} ledger cache count mismatch")
    record_by_id = {record.video_id: record for record in sidecar.records}
    cache_by_id = {observation.video_id: observation for observation in cache_observations}
    _require(
        len(cache_by_id) == expected_size,
        f"{split} audited v3 cache identities are not unique",
    )
    sources: list[V3SourceObservation] = []
    seen: set[str] = set()
    for index, summary in enumerate(summaries):
        _require(
            isinstance(summary, dict),
            f"{split} ledger cache summary {index} must be an object",
        )
        _require(
            set(summary) == _CACHE_SUMMARY_FIELDS,
            f"{split} ledger cache summary {index} fields mismatch",
        )
        video_id = summary["video_id"]
        _require(
            isinstance(video_id, str) and video_id in record_by_id,
            f"{split} ledger cache summary {index} has an unknown video ID",
        )
        _require(video_id not in seen, f"{split} ledger repeats video ID {video_id!r}")
        seen.add(video_id)
        record = record_by_id[video_id]
        cache = cache_by_id[video_id]
        _require(
            summary["video_sha256"] == record.video_sha256,
            f"{split} ledger video SHA-256 mismatch for {video_id!r}",
        )
        _require(
            summary["pose_fingerprint"] == expected_pose_fingerprint,
            f"{split} ledger pose fingerprint mismatch for {video_id!r}",
        )
        _require(
            isinstance(summary["video_path"], str)
            and bool(summary["video_path"].strip())
            and isinstance(summary["cache_path"], str)
            and bool(summary["cache_path"].strip()),
            f"{split} ledger paths must be non-empty for {video_id!r}",
        )
        _require(
            type(summary["skipped"]) is bool,
            f"{split} ledger skipped must be boolean for {video_id!r}",
        )
        _require(
            summary["skipped"] is False,
            f"{split} v3 source diagnostics are unavailable for skipped cache {video_id!r}",
        )
        source_frames = _positive_integer(
            summary["source_frames"],
            field=f"{split}/{video_id} source_frames",
        )
        source_valid_frames = _non_negative_integer(
            summary["source_valid_frames"],
            field=f"{split}/{video_id} source_valid_frames",
        )
        selected_source_frames = _non_negative_integer(
            summary["selected_source_frames"],
            field=f"{split}/{video_id} selected_source_frames",
        )
        _require(
            source_valid_frames <= source_frames,
            f"{split} source valid frames exceed decoded frames for {video_id!r}",
        )
        _require(
            selected_source_frames <= source_valid_frames,
            f"{split} selected track exceeds valid detections for {video_id!r}",
        )
        _require(
            (source_valid_frames == 0) == (selected_source_frames == 0),
            f"{split} zero-detection/selected-track invariant failed for {video_id!r}",
        )
        _require(
            summary["cached_frames"] == cache.total_frames,
            f"{split} ledger cached frame count mismatch for {video_id!r}",
        )
        _require(
            summary["cached_valid_frames"] == cache.valid_frames,
            f"{split} ledger cached valid-frame count mismatch for {video_id!r}",
        )
        _require(
            summary["pose_model"] == cache.pose_model,
            f"{split} ledger pose model mismatch for {video_id!r}",
        )
        fps = summary["fps"]
        _require(
            isinstance(fps, int | float)
            and not isinstance(fps, bool)
            and math.isfinite(float(fps))
            and float(fps) > 0,
            f"{split} ledger FPS must be finite and positive for {video_id!r}",
        )
        _require(
            selected_source_frames != 1 or cache.valid_frames == 0,
            f"{split} one-frame track was not converted to all-invalid for {video_id!r}",
        )
        sources.append(
            V3SourceObservation(
                video_id=video_id,
                source_frames=source_frames,
                source_valid_frames=source_valid_frames,
                selected_source_frames=selected_source_frames,
            )
        )
    _require(seen == set(record_by_id), f"{split} ledger identity set mismatch")

    snapshot = _snapshot(
        cache_observations,
        pose_fingerprint=expected_pose_fingerprint,
    )
    _require(
        payload["successful_cache_snapshot"] == snapshot.to_dict(),
        f"{split} ledger successful cache snapshot mismatch",
    )
    _require(
        _sha256_file(path.resolve(strict=True)) == ledger_sha256,
        f"{split} v3 ledger changed while it was being validated",
    )
    return tuple(sources)


def _distribution(values: Sequence[int | float]) -> dict[str, int | float]:
    _require(bool(values), "distribution requires at least one value")
    array = np.asarray(values, dtype=np.float64)
    quantiles = np.quantile(array, (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0))
    return {
        "count": int(array.size),
        "min": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "p50": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "max": float(quantiles[6]),
        "mean": float(np.mean(array)),
        "std_population": float(np.std(array)),
    }


def _snapshot_summary(
    observations: Sequence[CacheObservation],
    *,
    pose_fingerprint: str,
) -> dict[str, Any]:
    snapshot = _snapshot(observations, pose_fingerprint=pose_fingerprint)
    payload = snapshot.to_dict()
    return {
        "pose_fingerprint": pose_fingerprint,
        "cache_set_sha256": snapshot.fingerprint,
        "cache_snapshot_sha256": _canonical_sha256(payload),
        "entry_count": len(observations),
    }


def _summarize_scope(
    *,
    v2: Sequence[CacheObservation],
    v3: Sequence[CacheObservation],
    v3_sources: Sequence[V3SourceObservation],
    v2_pose_fingerprint: str,
    v3_pose_fingerprint: str,
) -> dict[str, Any]:
    v2_by_id = {observation.video_id: observation for observation in v2}
    v3_by_id = {observation.video_id: observation for observation in v3}
    source_by_id = {observation.video_id: observation for observation in v3_sources}
    _require(
        set(v2_by_id) == set(v3_by_id) == set(source_by_id),
        "v2, v3, and source-diagnostic identity sets differ",
    )
    video_ids = sorted(v2_by_id)
    v2_all_invalid = [
        video_id for video_id in video_ids if v2_by_id[video_id].valid_frames == 0
    ]
    v3_all_invalid = [
        video_id for video_id in video_ids if v3_by_id[video_id].valid_frames == 0
    ]
    extremely_short = [
        video_id
        for video_id in video_ids
        if 1
        <= source_by_id[video_id].selected_source_frames
        <= _EXTREMELY_SHORT_MAX_FRAMES
    ]
    single_frame_converted = [
        video_id
        for video_id in video_ids
        if source_by_id[video_id].selected_source_frames == 1
        and v3_by_id[video_id].valid_frames == 0
    ]
    return {
        "sample_count": len(video_ids),
        "v2": {
            **_snapshot_summary(v2, pose_fingerprint=v2_pose_fingerprint),
            "all_invalid_count": len(v2_all_invalid),
            "valid_frame_rate_distribution": _distribution(
                [v2_by_id[video_id].valid_rate for video_id in video_ids]
            ),
        },
        "v3": {
            **_snapshot_summary(v3, pose_fingerprint=v3_pose_fingerprint),
            "all_invalid_count": len(v3_all_invalid),
            "valid_frame_rate_distribution": _distribution(
                [v3_by_id[video_id].valid_rate for video_id in video_ids]
            ),
            "source_frames_distribution": _distribution(
                [source_by_id[video_id].source_frames for video_id in video_ids]
            ),
            "source_valid_frames_distribution": _distribution(
                [source_by_id[video_id].source_valid_frames for video_id in video_ids]
            ),
            "selected_source_frames_distribution": _distribution(
                [source_by_id[video_id].selected_source_frames for video_id in video_ids]
            ),
            "single_frame_converted_to_all_invalid_count": len(
                single_frame_converted
            ),
        },
        "video_ids": {
            "v2_all_invalid": v2_all_invalid,
            "v3_all_invalid": v3_all_invalid,
            "v3_extremely_short": extremely_short,
            "v3_single_frame_converted_to_all_invalid": single_frame_converted,
        },
    }


def audit_pose_cache_v2_v3(
    *,
    train_sidecar: Path,
    dev_sidecar: Path,
    v2_cache_root: Path,
    v3_train_ledger: Path,
    v3_dev_ledger: Path,
    v3_cache_root: Path,
    v2_pose_fingerprint: str = _FROZEN_V2_POSE_FINGERPRINT,
    v3_pose_fingerprint: str = _FROZEN_V3_POSE_FINGERPRINT,
) -> dict[str, Any]:
    """Run the strict train337/dev84 v2-v3 label-free comparison."""

    _lowercase_sha256(v2_pose_fingerprint, field="v2_pose_fingerprint")
    _lowercase_sha256(v3_pose_fingerprint, field="v3_pose_fingerprint")
    _require(
        v2_pose_fingerprint != v3_pose_fingerprint,
        "v2 and v3 pose fingerprints must differ",
    )
    train, train_sidecar_sha256 = _load_sidecar(train_sidecar, split="train")
    dev, dev_sidecar_sha256 = _load_sidecar(dev_sidecar, split="dev")
    combined_records = _validate_sidecar_pair(train, dev)

    split_inputs = {
        "train": (train, train_sidecar_sha256, v3_train_ledger),
        "dev": (dev, dev_sidecar_sha256, v3_dev_ledger),
    }
    by_split: dict[
        str,
        tuple[
            tuple[CacheObservation, ...],
            tuple[CacheObservation, ...],
            tuple[V3SourceObservation, ...],
        ],
    ] = {}
    for split, (sidecar, sidecar_sha256, ledger) in split_inputs.items():
        v2 = _load_cache_observations(
            sidecar.records,
            cache_root=v2_cache_root,
            expected_pose_fingerprint=v2_pose_fingerprint,
            cache_revision="v2",
        )
        v3 = _load_cache_observations(
            sidecar.records,
            cache_root=v3_cache_root,
            expected_pose_fingerprint=v3_pose_fingerprint,
            cache_revision="v3",
        )
        sources = _validate_v3_ledger(
            ledger,
            split=split,
            sidecar=sidecar,
            sidecar_sha256=sidecar_sha256,
            cache_observations=v3,
            expected_pose_fingerprint=v3_pose_fingerprint,
        )
        by_split[split] = (v2, v3, sources)

    all_v2 = by_split["train"][0] + by_split["dev"][0]
    all_v3 = by_split["train"][1] + by_split["dev"][1]
    all_sources = by_split["train"][2] + by_split["dev"][2]
    _require(
        {observation.video_id for observation in all_v2}
        == {record.video_id for record in combined_records},
        "audited cache union differs from the sidecar union",
    )
    pose_models = {
        observation.pose_model for observation in (*all_v2, *all_v3)
    }
    _require(len(pose_models) == 1, "v2 and v3 caches use different pose models")

    result = {
        "schema_version": 1,
        "status": "passed",
        "audit_class": "label-free-v2-v3-pose-cache-comparison",
        "scope": {
            "protocol": "ucfrep_526",
            "splits": ["train337", "dev84"],
            "sample_count": _EXPECTED_TOTAL,
            "test105_accessed": False,
            "source_videos_accessed": False,
            "labels_or_targets_accessed": False,
        },
        "definitions": {
            "valid_frame_rate": "cached_valid_frames / 256",
            "all_invalid": "cached_valid_frames == 0",
            "extremely_short": (
                "1 <= v3 selected_source_frames <= "
                f"{_EXTREMELY_SHORT_MAX_FRAMES}"
            ),
            "single_frame_converted_to_all_invalid": (
                "v3 selected_source_frames == 1 and cached_valid_frames == 0"
            ),
        },
        "inputs": {
            "train_sidecar_sha256": train_sidecar_sha256,
            "train_sidecar_fingerprint": train.fingerprint,
            "train_identity_sha256": pose_input_identity_sha256(train.records),
            "dev_sidecar_sha256": dev_sidecar_sha256,
            "dev_sidecar_fingerprint": dev.fingerprint,
            "dev_identity_sha256": pose_input_identity_sha256(dev.records),
            "combined_identity_sha256": pose_input_identity_sha256(combined_records),
            "pose_model": next(iter(pose_models)),
        },
        "splits": {
            split: _summarize_scope(
                v2=by_split[split][0],
                v3=by_split[split][1],
                v3_sources=by_split[split][2],
                v2_pose_fingerprint=v2_pose_fingerprint,
                v3_pose_fingerprint=v3_pose_fingerprint,
            )
            for split in ("train", "dev")
        },
        "total": _summarize_scope(
            v2=all_v2,
            v3=all_v3,
            v3_sources=all_sources,
            v2_pose_fingerprint=v2_pose_fingerprint,
            v3_pose_fingerprint=v3_pose_fingerprint,
        ),
    }
    _require(
        result["total"]["sample_count"] == _EXPECTED_TOTAL,
        "final audit total is not 421",
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen UCFRep train337/dev84 v2 and v3 pose caches "
            "without loading counts, actions, targets, source videos, or test105."
        )
    )
    parser.add_argument("--train-sidecar", type=Path, required=True)
    parser.add_argument("--dev-sidecar", type=Path, required=True)
    parser.add_argument("--v2-cache-root", type=Path, required=True)
    parser.add_argument("--v3-train-ledger", type=Path, required=True)
    parser.add_argument("--v3-dev-ledger", type=Path, required=True)
    parser.add_argument("--v3-cache-root", type=Path, required=True)
    parser.add_argument(
        "--v2-pose-fingerprint",
        default=_FROZEN_V2_POSE_FINGERPRINT,
        help="Expected historical detected-span-v2 pose fingerprint.",
    )
    parser.add_argument(
        "--v3-pose-fingerprint",
        default=_FROZEN_V3_POSE_FINGERPRINT,
        help="Expected longest-contiguous-track-v3 pose fingerprint.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = audit_pose_cache_v2_v3(
            train_sidecar=arguments.train_sidecar,
            dev_sidecar=arguments.dev_sidecar,
            v2_cache_root=arguments.v2_cache_root,
            v3_train_ledger=arguments.v3_train_ledger,
            v3_dev_ledger=arguments.v3_dev_ledger,
            v3_cache_root=arguments.v3_cache_root,
            v2_pose_fingerprint=arguments.v2_pose_fingerprint,
            v3_pose_fingerprint=arguments.v3_pose_fingerprint,
        )
    except (ComparisonAuditError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"v2-v3 pose-cache audit failed: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            result,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
