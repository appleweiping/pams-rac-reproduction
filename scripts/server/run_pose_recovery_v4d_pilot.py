#!/usr/bin/env python3
"""Run the frozen v4d zero11 or same39 label-free pose-recovery pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pams.config import load_config
from pams.data import (
    load_pose_cache,
    load_pose_cache_set,
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_cache_path,
    pose_input_identity_sha256,
    validate_pose_input_binding,
    write_pose_cache,
)
from pams.keypoint_recovery import (
    V4D_PREPROCESSING_REVISION,
    V4D_RECOVERY_MODE,
    KeypointRecoveryError,
    load_keypointrcnn_runtime,
    recover_v4a_locked_video,
)
from pams.pose import sha256_file

_TRAIN337_SIDECAR_SHA256 = "f95df0050df21f05bbc9b42d0154470714dde279e04cb8b00d0212bf49057d16"
_TRAIN337_COMMITMENT_SHA256 = "85d41d2f59872e0900e9058481efbdf6bce91407d4c26bcde0a6547776454e53"
_TRAIN337_IDENTITY_SHA256 = "88367535e296213377c366ed69abbd1e808c38049d79bb948091688e4c5b7b3f"
_V4A_LONG_TAIL_AUDIT_SHA256 = "d2eeb9d0f5ef75e6fc92f87357d530061c91326c8b97dfd5345a600deab1f398"
_FORBIDDEN_KEYS = frozenset(
    {
        "action",
        "actions",
        "count",
        "counts",
        "cycle_boundaries",
        "cycle_boundary",
        "ground_truth",
        "ground_truth_count",
        "gt",
        "gt_count",
        "label",
        "labels",
        "metrics_result",
        "nmae",
        "obo",
        "prediction",
        "predictions",
        "target",
        "targets",
        "test",
        "dev",
    }
)
_V4A_LEDGER_KEYS = frozenset(
    {
        "caches",
        "commitment_file_sha256",
        "commitment_fingerprint",
        "completed",
        "extracted",
        "failed",
        "failures",
        "identity_sha256",
        "incomplete_clip_policy",
        "input_file_sha256",
        "input_fingerprint",
        "input_kind",
        "pose_fingerprint",
        "pose_recovery",
        "protocol",
        "schema_version",
        "selected",
        "sidecar_fingerprint",
        "sidecar_sha256",
        "skipped",
        "split",
        "successful_cache_snapshot",
    }
)
_FROZEN_SELECTION_KEYS = frozenset(
    {
        "artifact_type",
        "bindings",
        "cohort_counts",
        "frozen_union_record_total",
        "label_free",
        "protocol",
        "record_total",
        "rows",
        "schema_version",
        "selected_identity_sha256",
        "selection_rule",
        "split",
    }
)
_V4A_CACHE_ROW_KEYS = frozenset(
    {
        "annotation_sha256",
        "cache_path",
        "cached_frames",
        "cached_valid_frames",
        "clip_end_frame",
        "clip_start_frame",
        "decoded_clip_frames",
        "expected_clip_frames",
        "fps",
        "incomplete_clip_policy",
        "padded_tail_frames",
        "pose_fingerprint",
        "pose_model",
        "recovery_audit",
        "selected_source_frames",
        "skipped",
        "source_frames",
        "source_valid_frames",
        "video_id",
        "video_path",
        "video_sha256",
    }
)
_V4A_RECOVERY_KEYS = frozenset(
    {
        "decoded_segment_frames",
        "expected_segment_frames",
        "final_longest_valid_run",
        "final_valid_frames",
        "final_valid_mask_sha256",
        "heavy_full_frame_attempted",
        "heavy_full_frame_detected",
        "heavy_model_asset_sha256",
        "heavy_model_id",
        "observed_span_frames",
        "padded_tail_frames",
        "pass0_observations_preserved",
        "pass0_shared_coordinate_max_abs_error",
        "pass0_valid_frames",
        "pass0_valid_mask_sha256",
        "pose_coordinate_interpolation",
        "recovered_valid_frames",
        "roi_retry_attempted",
        "roi_retry_detected",
        "roi_retry_eligible",
        "schema_version",
        "source_frames",
        "temporal_resampling",
    }
)
_FROZEN_SELECTION_ROW_KEYS = frozenset(
    {
        "coverage_bottom_decile",
        "final_valid_at_most_8",
        "longest_run_bottom_decile",
        "v4a_final_valid_frames",
        "v4a_longest_run_fraction",
        "v4a_source_coverage",
        "video_id_sha256",
    }
)


class PilotError(RuntimeError):
    """Raised when a v4d pilot binding or cohort invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotError(message)


def _reject_non_finite(value: str) -> None:
    raise PilotError(f"non-finite JSON constant is forbidden: {value}")


def _reject_forbidden_keys(encoded: str, *, role: str) -> None:
    """Reject sensitive object keys before any corresponding value is decoded."""

    index = 0
    while index < len(encoded):
        if encoded[index] != '"':
            index += 1
            continue
        start = index
        index += 1
        escaped = False
        while index < len(encoded):
            character = encoded[index]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                index += 1
                break
            index += 1
        else:
            return
        cursor = index
        while cursor < len(encoded) and encoded[cursor] in " \t\r\n":
            cursor += 1
        if cursor >= len(encoded) or encoded[cursor] != ":":
            continue
        try:
            key = json.loads(encoded[start:index])
        except json.JSONDecodeError:
            return
        if isinstance(key, str) and key.casefold() in _FORBIDDEN_KEYS:
            raise PilotError(f"{role} contains forbidden sensitive field {key!r}")


def _reject_duplicate_or_privileged_fields(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise PilotError(f"duplicate JSON field is forbidden: {key!r}")
        if key.casefold() in _FORBIDDEN_KEYS:
            raise PilotError(f"privileged split/target field is forbidden: {key!r}")
        payload[key] = value
    return payload


def _load_json(path: Path, *, role: str) -> tuple[dict[str, Any], str]:
    source = path.resolve(strict=True)
    encoded = source.read_bytes()
    try:
        encoded_text = encoded.decode("utf-8")
        _reject_forbidden_keys(encoded_text, role=role)
        value = json.loads(
            encoded_text,
            object_pairs_hook=_reject_duplicate_or_privileged_fields,
            parse_constant=_reject_non_finite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotError(f"{role} must be strict UTF-8 JSON") from exc
    _require(isinstance(value, dict), f"{role} root must be an object")
    return value, hashlib.sha256(encoded).hexdigest()


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _integer(value: Any, role: str) -> int:
    _require(type(value) is int and value >= 0, f"{role} must be a non-negative integer")
    return int(value)


def run_pilot(
    *,
    cohort: str,
    config_path: Path,
    train_input_path: Path,
    train_commitment_path: Path,
    video_root: Path,
    model_asset_path: Path,
    v4a_cache_dir: Path,
    v4a_ledger_path: Path,
    v4a_paired_gate_path: Path,
    frozen_selection_path: Path,
    cache_dir: Path,
    ledger_path: Path,
    selection_path: Path,
) -> dict[str, Any]:
    """Extract one frozen v4d pilot cohort."""

    _require(cohort in {"zero11", "same39"}, "unsupported v4d pilot cohort")
    config = load_config(config_path.resolve(strict=True))
    _require(
        config.pose.preprocessing_revision == V4D_PREPROCESSING_REVISION,
        "pilot config is not v4d",
    )
    settings = config.pose.keypoint_recovery
    _require(settings is not None, "v4d config has no keypoint recovery settings")
    manifest = load_pose_input_manifest(train_input_path, validate_exact=True)
    commitment = load_pose_input_commitment(train_commitment_path)
    sidecar_sha256 = sha256_file(train_input_path.resolve(strict=True))
    commitment_sha256 = sha256_file(train_commitment_path.resolve(strict=True))
    _require(sidecar_sha256 == _TRAIN337_SIDECAR_SHA256, "train337 sidecar SHA mismatch")
    _require(
        commitment_sha256 == _TRAIN337_COMMITMENT_SHA256,
        "train337 commitment SHA mismatch",
    )
    validate_pose_input_binding(manifest, commitment, sidecar_sha256=sidecar_sha256)
    _require(manifest.split == "train" and len(manifest.records) == 337, "input is not train337")
    full_identity_sha256 = pose_input_identity_sha256(manifest.records)
    _require(
        full_identity_sha256 == _TRAIN337_IDENTITY_SHA256,
        "train337 identity mismatch",
    )

    v4a_ledger, v4a_ledger_sha256 = _load_json(v4a_ledger_path, role="v4a ledger")
    _require(set(v4a_ledger) == _V4A_LEDGER_KEYS, "v4a ledger root schema mismatch")
    _require(
        v4a_ledger_sha256 == settings.base_pose_ledger_sha256,
        "v4a ledger SHA mismatch",
    )
    v4a_paired_gate, v4a_paired_gate_sha256 = _load_json(
        v4a_paired_gate_path,
        role="v4a paired gate",
    )
    _require(
        v4a_paired_gate_sha256 == settings.base_paired_gate_sha256,
        "v4a paired gate SHA mismatch",
    )
    _require(
        v4a_paired_gate.get("artifact_type") == "pams_pose_recovery_v4a_train337_paired_audit",
        "v4a paired gate artifact mismatch",
    )
    paired_bindings = v4a_paired_gate.get("bindings")
    _require(isinstance(paired_bindings, Mapping), "v4a paired gate bindings missing")
    _require(
        paired_bindings.get("v4_ledger_sha256") == settings.base_pose_ledger_sha256,
        "v4a paired gate does not bind the frozen ledger",
    )
    _require(v4a_ledger.get("schema_version") == 2, "v4a ledger schema mismatch")
    _require(v4a_ledger.get("input_kind") == "label_free_sidecar", "v4a ledger is not label-free")
    _require(v4a_ledger.get("split") == "train", "v4a ledger split mismatch")
    _require(v4a_ledger.get("identity_sha256") == full_identity_sha256, "v4a identity mismatch")
    _require(
        v4a_ledger.get("sidecar_sha256") == _TRAIN337_SIDECAR_SHA256,
        "v4a sidecar binding mismatch",
    )
    _require(
        v4a_ledger.get("commitment_file_sha256") == _TRAIN337_COMMITMENT_SHA256,
        "v4a commitment binding mismatch",
    )
    _require(v4a_ledger.get("selected") == 337, "v4a ledger is not train337")
    _require(v4a_ledger.get("failed") == 0, "v4a ledger contains failures")
    _require(
        v4a_ledger.get("pose_fingerprint") == settings.base_pose_fingerprint,
        "v4a pose fingerprint mismatch",
    )
    v4a_recovery_binding = v4a_ledger.get("pose_recovery")
    _require(isinstance(v4a_recovery_binding, Mapping), "v4a recovery binding missing")
    _require(
        set(v4a_recovery_binding)
        == {
            "heavy_model_asset_sha256",
            "heavy_model_id",
            "pose_coordinate_interpolation",
            "preprocessing_revision",
            "schema_version",
            "temporal_resampling",
        },
        "v4a recovery binding schema mismatch",
    )
    _require(
        v4a_recovery_binding.get("preprocessing_revision") == settings.base_preprocessing_revision,
        "v4a preprocessing revision mismatch",
    )
    raw_v4a_rows = v4a_ledger.get("caches")
    _require(isinstance(raw_v4a_rows, list) and len(raw_v4a_rows) == 337, "v4a rows mismatch")
    v4a_by_id: dict[str, Mapping[str, Any]] = {}
    for value in raw_v4a_rows:
        row = value if isinstance(value, Mapping) else None
        _require(row is not None, "v4a cache row must be an object")
        _require(set(row) == _V4A_CACHE_ROW_KEYS, "v4a cache row schema mismatch")
        recovery_audit = row.get("recovery_audit")
        _require(isinstance(recovery_audit, Mapping), "v4a recovery audit missing")
        _require(
            set(recovery_audit) == _V4A_RECOVERY_KEYS,
            "v4a recovery audit schema mismatch",
        )
        video_id = str(row.get("video_id", ""))
        _require(video_id and video_id not in v4a_by_id, "invalid or duplicate v4a video ID")
        v4a_by_id[video_id] = row
    _require(
        set(v4a_by_id) == {record.video_id for record in manifest.records},
        "v4a and train337 identity sets differ",
    )
    _, base_snapshot = load_pose_cache_set(
        manifest.records,
        cache_dir=v4a_cache_dir.resolve(strict=True),
        pose_fingerprint=settings.base_pose_fingerprint,
        materialize_sequences=False,
    )
    _require(
        base_snapshot.fingerprint == settings.base_pose_cache_set_sha256,
        "v4a cache-set fingerprint mismatch",
    )

    frozen_selection, frozen_selection_sha256 = _load_json(
        frozen_selection_path,
        role="frozen same39 selection",
    )
    _require(
        set(frozen_selection) == _FROZEN_SELECTION_KEYS,
        "frozen selection root schema mismatch",
    )
    _require(
        frozen_selection_sha256 == settings.frozen_same39_selection_sha256,
        "frozen selection SHA mismatch",
    )
    _require(frozen_selection.get("schema_version") == 1, "selection schema mismatch")
    _require(
        frozen_selection.get("artifact_type")
        == "pams_pose_recovery_v4c_train337_long_tail_pilot_selection",
        "selection artifact mismatch",
    )
    _require(
        frozen_selection.get("protocol") == "ucfrep_526"
        and frozen_selection.get("split") == "train"
        and frozen_selection.get("label_free") is True,
        "selection protocol mismatch",
    )
    _require(
        frozen_selection.get("selected_identity_sha256") == settings.frozen_same39_identity_sha256,
        "same39 identity binding mismatch",
    )
    selection_bindings = frozen_selection.get("bindings")
    _require(isinstance(selection_bindings, Mapping), "selection bindings missing")
    _require(
        set(selection_bindings)
        == {
            "train337_commitment_sha256",
            "train337_identity_sha256",
            "train337_sidecar_sha256",
            "v4a_ledger_sha256",
            "v4a_long_tail_audit_sha256",
        },
        "selection binding schema mismatch",
    )
    _require(
        dict(selection_bindings)
        == {
            "train337_commitment_sha256": _TRAIN337_COMMITMENT_SHA256,
            "train337_identity_sha256": _TRAIN337_IDENTITY_SHA256,
            "train337_sidecar_sha256": _TRAIN337_SIDECAR_SHA256,
            "v4a_ledger_sha256": settings.base_pose_ledger_sha256,
            "v4a_long_tail_audit_sha256": _V4A_LONG_TAIL_AUDIT_SHA256,
        },
        "selection bindings mismatch",
    )
    _require(frozen_selection.get("record_total") == 39, "frozen selection is not same39")
    frozen_rows = frozen_selection.get("rows")
    _require(isinstance(frozen_rows, list) and len(frozen_rows) == 39, "same39 rows mismatch")
    _require(
        all(
            isinstance(row, Mapping) and set(row) == _FROZEN_SELECTION_ROW_KEYS
            for row in frozen_rows
        ),
        "same39 row schema mismatch",
    )
    selected_hashes = {
        str(row.get("video_id_sha256")) for row in frozen_rows if isinstance(row, Mapping)
    }
    _require(len(selected_hashes) == 39, "same39 selection hashes are invalid")
    same39_records = tuple(
        record
        for record in manifest.records
        if hashlib.sha256(record.video_id.encode("utf-8")).hexdigest() in selected_hashes
    )
    _require(len(same39_records) == 39, "same39 selection lost records")
    zero11_records = tuple(
        record
        for record in same39_records
        if _integer(
            (v4a_by_id[record.video_id].get("recovery_audit") or {}).get("final_valid_frames"),
            "v4a final-valid frames",
        )
        == 0
    )
    _require(len(zero11_records) == 11, "frozen same39 must contain exactly zero11")
    selected_records = zero11_records if cohort == "zero11" else same39_records
    selected_identity_sha256 = pose_input_identity_sha256(selected_records)
    expected_selected_identity = (
        settings.frozen_zero11_identity_sha256
        if cohort == "zero11"
        else settings.frozen_same39_identity_sha256
    )
    _require(
        selected_identity_sha256 == expected_selected_identity,
        f"{cohort} selected identity mismatch",
    )
    selection_payload = {
        "schema_version": 1,
        "artifact_type": f"pams_pose_recovery_v4d_train337_{cohort}_pilot_selection",
        "protocol": manifest.protocol,
        "split": "train",
        "label_free": True,
        "selection_rule": (
            "v4a_final_valid_frames==0 within frozen same39"
            if cohort == "zero11"
            else "exact frozen v4c same39 long-tail selection"
        ),
        "record_total": len(selected_records),
        "selected_identity_sha256": selected_identity_sha256,
        "bindings": {
            "train337_sidecar_sha256": sidecar_sha256,
            "train337_commitment_sha256": sha256_file(train_commitment_path.resolve(strict=True)),
            "train337_identity_sha256": full_identity_sha256,
            "v4a_ledger_sha256": v4a_ledger_sha256,
            "v4a_paired_gate_sha256": v4a_paired_gate_sha256,
            "v4a_pose_cache_set_sha256": base_snapshot.fingerprint,
            "frozen_same39_selection_sha256": frozen_selection_sha256,
        },
        "rows": [
            {
                "video_id_sha256": hashlib.sha256(record.video_id.encode("utf-8")).hexdigest(),
                "v4a_final_valid_frames": _integer(
                    (v4a_by_id[record.video_id].get("recovery_audit") or {}).get(
                        "final_valid_frames"
                    ),
                    "v4a final-valid frames",
                ),
            }
            for record in selected_records
        ],
    }
    _write_exclusive(selection_path, selection_payload)

    runtime = load_keypointrcnn_runtime(model_asset_path, settings=settings)
    root = video_root.resolve(strict=True)
    base_root = v4a_cache_dir.resolve(strict=True)
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    successful_records = []
    for offset, record in enumerate(selected_records, start=1):
        video_path = (root / record.video_path).resolve(strict=True)
        try:
            video_path.relative_to(root)
            _require(
                sha256_file(video_path) == record.video_sha256,
                "source video SHA mismatch",
            )
            base_path = pose_cache_path(base_root, record.video_id).resolve(strict=True)
            base_sequence, base_metadata = load_pose_cache(
                base_path,
                expected_video_sha256=record.video_sha256,
                expected_pose_fingerprint=settings.base_pose_fingerprint,
                expected_annotation_sha256=record.annotation_sha256,
                expected_clip_start_frame=record.clip_start_frame,
                expected_clip_end_frame=record.clip_end_frame,
                validate_clip_provenance=True,
            )
            base_row = v4a_by_id[record.video_id]
            base_audit = base_row.get("recovery_audit")
            _require(isinstance(base_audit, Mapping), "v4a row lacks recovery audit")
            decoded_frames = _integer(
                base_metadata.decoded_clip_frames,
                "v4a decoded clip frames",
            )
            final_sequence, recovery_audit = recover_v4a_locked_video(
                video_path,
                video_id=record.video_id,
                clip_start_frame=int(record.clip_start_frame),
                clip_end_frame=int(record.clip_end_frame),
                base_sequence=base_sequence,
                base_decoded_frames=decoded_frames,
                base_recovery_audit=base_audit,
                expected_video_sha256=record.video_sha256,
                runtime=runtime,
                settings=settings,
            )
            target = pose_cache_path(cache_dir, record.video_id)
            metadata = write_pose_cache(
                target,
                final_sequence,
                video_sha256=record.video_sha256,
                pose_fingerprint=config.pose_fingerprint,
                pose_model=settings.model_id,
                annotation_sha256=record.annotation_sha256,
                clip_start_frame=record.clip_start_frame,
                clip_end_frame=record.clip_end_frame,
                decoded_clip_frames=decoded_frames,
                expected_clip_frames=int(record.clip_end_frame) - int(record.clip_start_frame),
                padded_tail_frames=base_metadata.padded_tail_frames,
                incomplete_clip_policy=base_metadata.incomplete_clip_policy,
            )
            summaries.append(
                {
                    "video_id": record.video_id,
                    "video_path": str(video_path),
                    "cache_path": str(target.resolve()),
                    "video_sha256": record.video_sha256,
                    "pose_fingerprint": config.pose_fingerprint,
                    "source_frames": base_row.get("source_frames"),
                    "source_valid_frames": base_row.get("source_valid_frames"),
                    "selected_source_frames": base_row.get("selected_source_frames"),
                    "cached_frames": final_sequence.num_frames,
                    "cached_valid_frames": int(np.count_nonzero(final_sequence.valid_mask)),
                    "fps": final_sequence.fps,
                    "pose_model": settings.model_id,
                    "annotation_sha256": metadata.annotation_sha256,
                    "clip_start_frame": metadata.clip_start_frame,
                    "clip_end_frame": metadata.clip_end_frame,
                    "expected_clip_frames": metadata.expected_clip_frames,
                    "decoded_clip_frames": metadata.decoded_clip_frames,
                    "padded_tail_frames": metadata.padded_tail_frames,
                    "incomplete_clip_policy": metadata.incomplete_clip_policy,
                    "skipped": False,
                    "base_v4a_cache_sha256": sha256_file(base_path),
                    "recovery_audit": recovery_audit,
                }
            )
            successful_records.append(record)
            print(
                json.dumps(
                    {
                        "cohort": cohort,
                        "completed": offset,
                        "selected": len(selected_records),
                        "cached_valid_frames": summaries[-1]["cached_valid_frames"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except (OSError, TypeError, ValueError, KeypointRecoveryError, PilotError) as exc:
            failures.append(
                {
                    "video_id_sha256": hashlib.sha256(record.video_id.encode("utf-8")).hexdigest(),
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            print(
                json.dumps(
                    {"cohort": cohort, "failed": offset, "error": str(exc)},
                    sort_keys=True,
                ),
                flush=True,
            )

    cache_snapshot = None
    if successful_records:
        _, snapshot = load_pose_cache_set(
            tuple(successful_records),
            cache_dir=cache_dir,
            pose_fingerprint=config.pose_fingerprint,
            materialize_sequences=False,
        )
        cache_snapshot = snapshot.to_dict()
    ledger = {
        "schema_version": 2,
        "artifact_type": f"pams_pose_recovery_v4d_train337_{cohort}_pilot_ledger",
        "input_kind": "label_free_train337_hashed_pilot_subset",
        "protocol": manifest.protocol,
        "split": "train",
        "sidecar_sha256": sidecar_sha256,
        "commitment_file_sha256": sha256_file(train_commitment_path.resolve(strict=True)),
        "full_train_identity_sha256": full_identity_sha256,
        "identity_sha256": selected_identity_sha256,
        "selection_sha256": sha256_file(selection_path.resolve(strict=True)),
        "pose_fingerprint": config.pose_fingerprint,
        "pose_recovery": {
            "schema_version": 1,
            "preprocessing_revision": config.pose.preprocessing_revision,
            "recovery_mode": V4D_RECOVERY_MODE,
            "base_pose_fingerprint": settings.base_pose_fingerprint,
            "base_pose_cache_set_sha256": settings.base_pose_cache_set_sha256,
            "keypointrcnn_model_id": settings.model_id,
            "keypointrcnn_model_asset_sha256": settings.model_asset_sha256,
            "keypointrcnn_model_topology": settings.model_topology,
            "keypointrcnn_runtime_receipt": dict(runtime.runtime_receipt),
            "detector_observes_all_decoded_frames": (settings.detector_observes_all_decoded_frames),
            "fill_missing_only": settings.fill_missing_only,
            "temporal_resampling": settings.temporal_resampling,
            "pose_coordinate_interpolation": settings.pose_coordinate_interpolation,
        },
        "successful_cache_snapshot": cache_snapshot,
        "selected": len(selected_records),
        "completed": len(summaries),
        "extracted": len(summaries),
        "skipped": 0,
        "failed": len(failures),
        "recovery_version": "v4d",
        "cohort": cohort,
        "caches": summaries,
        "failures": failures,
    }
    _write_exclusive(ledger_path, ledger)
    return {
        "cohort": cohort,
        "selected": len(selected_records),
        "completed": len(summaries),
        "failed": len(failures),
        "pose_fingerprint": config.pose_fingerprint,
        "selection_sha256": ledger["selection_sha256"],
        "cache_set_sha256": None if cache_snapshot is None else cache_snapshot["fingerprint"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("zero11", "same39"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-commitment", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--model-asset", type=Path, required=True)
    parser.add_argument("--v4a-cache-dir", type=Path, required=True)
    parser.add_argument("--v4a-ledger", type=Path, required=True)
    parser.add_argument("--v4a-paired-gate", type=Path, required=True)
    parser.add_argument("--frozen-selection", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_pilot(
            cohort=args.cohort,
            config_path=args.config,
            train_input_path=args.train_input,
            train_commitment_path=args.train_commitment,
            video_root=args.video_root,
            model_asset_path=args.model_asset,
            v4a_cache_dir=args.v4a_cache_dir,
            v4a_ledger_path=args.v4a_ledger,
            v4a_paired_gate_path=args.v4a_paired_gate,
            frozen_selection_path=args.frozen_selection,
            cache_dir=args.cache_dir,
            ledger_path=args.ledger,
            selection_path=args.selection,
        )
    except (OSError, TypeError, ValueError, KeypointRecoveryError, PilotError) as exc:
        print(f"v4d pilot failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
