#!/usr/bin/env python3
"""Audit frozen v4d zero11/same39 pilots without count or action labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from pams.data import (
    load_pose_cache_set,
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
    validate_pose_input_binding,
)
from pams.keypoint_recovery import V4D_RECOVERY_MODE

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


class PilotAuditError(RuntimeError):
    """Raised when a v4d pilot artifact violates a frozen invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotAuditError(message)


def _reject_forbidden_keys(encoded: str, *, role: str) -> None:
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
            raise PilotAuditError(f"{role} contains forbidden sensitive field {key!r}")


def _reject_non_finite(value: str) -> None:
    raise PilotAuditError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise PilotAuditError(f"duplicate JSON field is forbidden: {key!r}")
        payload[key] = value
    return payload


def _load_json(path: Path, *, role: str) -> dict[str, Any]:
    raw = path.resolve(strict=True).read_bytes()
    try:
        encoded = raw.decode("utf-8")
        _reject_forbidden_keys(encoded, role=role)
        value = json.loads(
            encoded,
            object_pairs_hook=_reject_duplicate_fields,
            parse_constant=_reject_non_finite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PilotAuditError(f"{role} must be strict UTF-8 JSON") from exc
    _require(isinstance(value, dict), f"{role} root must be an object")
    return value


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    payload: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in payload:
            raise PilotAuditError(f"duplicate YAML field is forbidden: {key!r}")
        payload[key] = loader.construct_object(value_node, deep=deep)
    return payload


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _mapping(value: Any, role: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{role} must be an object")
    return value


def _integer(value: Any, role: str) -> int:
    _require(type(value) is int and value >= 0, f"{role} must be a non-negative integer")
    return int(value)


def _finite(value: Any, role: str) -> float:
    _require(type(value) in {int, float} and math.isfinite(float(value)), f"{role} must be finite")
    return float(value)


def _digest(value: Any, role: str) -> str:
    text = str(value)
    _require(
        len(text) == 64 and all(character in "0123456789abcdef" for character in text),
        f"{role} must be lowercase SHA-256",
    )
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: Sequence[float], quantile: float) -> float:
    _require(bool(values), "percentile requires values")
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile, method="linear"))


def _criterion(value: int | float, *, relation: str, threshold: int | float) -> dict[str, Any]:
    passed = value >= threshold if relation == "at_least" else value <= threshold
    return {
        "value": value,
        "relation": relation,
        "threshold": threshold,
        "passed": bool(passed),
    }


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def audit_pilot(
    *,
    cohort: str,
    gate_path: Path,
    train_input_path: Path,
    train_commitment_path: Path,
    selection_path: Path,
    v4a_cache_dir: Path,
    v4a_ledger_path: Path,
    v4a_paired_gate_path: Path,
    candidate_cache_dir: Path,
    candidate_ledger_path: Path,
) -> dict[str, Any]:
    """Return actual cohort metrics and the v4a-retaining full projection."""

    _require(cohort in {"zero11", "same39"}, "unsupported v4d cohort")
    gate = yaml.load(
        gate_path.resolve(strict=True).read_text(encoding="utf-8"),
        Loader=_UniqueKeyLoader,
    )
    _require(isinstance(gate, Mapping), "v4d gate must be a mapping")
    _require(
        set(gate)
        == {
            "artifact_type",
            "bindings",
            "expected_same39_records",
            "expected_zero11_records",
            "full337_thresholds",
            "protocol",
            "same39_thresholds",
            "schema_version",
            "split",
            "zero11_thresholds",
        },
        "v4d gate root schema mismatch",
    )
    _require(gate.get("schema_version") == 1, "v4d gate schema mismatch")
    _require(
        gate.get("artifact_type") == "pams_pose_recovery_v4d_train337_pilot_gate",
        "unexpected v4d gate artifact type",
    )
    bindings = _mapping(gate.get("bindings"), "v4d gate bindings")
    zero_thresholds = _mapping(gate.get("zero11_thresholds"), "zero11 thresholds")
    same_thresholds = _mapping(gate.get("same39_thresholds"), "same39 thresholds")
    full_thresholds = _mapping(gate.get("full337_thresholds"), "full337 thresholds")
    expected_records = 11 if cohort == "zero11" else 39
    _require(
        gate.get("expected_zero11_records") == 11 and gate.get("expected_same39_records") == 39,
        "v4d gate cohort sizes mismatch",
    )

    manifest = load_pose_input_manifest(train_input_path, validate_exact=True)
    commitment = load_pose_input_commitment(train_commitment_path)
    sidecar_sha256 = _sha256_file(train_input_path.resolve(strict=True))
    commitment_sha256 = _sha256_file(train_commitment_path.resolve(strict=True))
    validate_pose_input_binding(manifest, commitment, sidecar_sha256=sidecar_sha256)
    _require(manifest.split == "train" and len(manifest.records) == 337, "input is not train337")
    _require(
        sidecar_sha256 == _digest(bindings.get("train337_sidecar_sha256"), "sidecar SHA"),
        "train337 sidecar SHA mismatch",
    )
    _require(
        commitment_sha256 == _digest(bindings.get("train337_commitment_sha256"), "commitment SHA"),
        "train337 commitment SHA mismatch",
    )
    _require(
        pose_input_identity_sha256(manifest.records)
        == _digest(bindings.get("train337_identity_sha256"), "train337 identity"),
        "train337 identity mismatch",
    )

    _require(
        _sha256_file(v4a_ledger_path.resolve(strict=True))
        == _digest(bindings.get("v4a_ledger_sha256"), "v4a ledger SHA"),
        "v4a ledger SHA mismatch",
    )
    _require(
        _sha256_file(v4a_paired_gate_path.resolve(strict=True))
        == _digest(bindings.get("v4a_paired_gate_sha256"), "v4a paired gate SHA"),
        "v4a paired gate SHA mismatch",
    )
    selection = _load_json(selection_path, role="v4d selection")
    v4a_ledger = _load_json(v4a_ledger_path, role="v4a ledger")
    v4a_gate = _load_json(v4a_paired_gate_path, role="v4a paired gate")
    candidate = _load_json(candidate_ledger_path, role="v4d candidate ledger")
    _require(
        set(candidate)
        == {
            "artifact_type",
            "caches",
            "cohort",
            "commitment_file_sha256",
            "completed",
            "extracted",
            "failed",
            "failures",
            "full_train_identity_sha256",
            "identity_sha256",
            "input_kind",
            "pose_fingerprint",
            "pose_recovery",
            "protocol",
            "recovery_version",
            "schema_version",
            "selected",
            "selection_sha256",
            "sidecar_sha256",
            "skipped",
            "split",
            "successful_cache_snapshot",
        },
        "candidate ledger root schema mismatch",
    )
    _require(selection.get("record_total") == expected_records, "selection record count mismatch")
    _require(candidate.get("cohort") == cohort, "candidate cohort mismatch")
    _require(candidate.get("selected") == expected_records, "candidate selected count mismatch")
    _require(candidate.get("completed") == expected_records, "candidate completion mismatch")
    _require(candidate.get("failed") == 0 and candidate.get("failures") == [], "candidate failed")
    _require(
        candidate.get("pose_fingerprint")
        == _digest(bindings.get("v4d_pose_fingerprint"), "v4d pose fingerprint"),
        "candidate pose fingerprint mismatch",
    )
    _require(
        candidate.get("selection_sha256") == _sha256_file(selection_path.resolve(strict=True)),
        "candidate selection binding mismatch",
    )
    recovery_binding = _mapping(candidate.get("pose_recovery"), "candidate recovery binding")
    _require(
        recovery_binding.get("recovery_mode") == V4D_RECOVERY_MODE,
        "candidate recovery mode mismatch",
    )
    expected_asset_sha = _digest(
        bindings.get("keypointrcnn_model_asset_sha256"), "Keypoint R-CNN asset SHA"
    )
    _require(
        recovery_binding.get("keypointrcnn_model_asset_sha256") == expected_asset_sha,
        "candidate model asset mismatch",
    )

    selection_rows = selection.get("rows")
    _require(
        isinstance(selection_rows, list) and len(selection_rows) == expected_records,
        "selection rows mismatch",
    )
    selected_hashes = {
        _digest(_mapping(row, "selection row").get("video_id_sha256"), "selected video hash")
        for row in selection_rows
    }
    _require(len(selected_hashes) == expected_records, "selection has duplicate identities")
    selected_records = tuple(
        record
        for record in manifest.records
        if hashlib.sha256(record.video_id.encode("utf-8")).hexdigest() in selected_hashes
    )
    _require(len(selected_records) == expected_records, "selection does not map to train337")
    candidate_pose_fingerprint = _digest(
        bindings.get("v4d_pose_fingerprint"),
        "v4d pose fingerprint",
    )
    candidate_sequences, candidate_snapshot = load_pose_cache_set(
        selected_records,
        cache_dir=candidate_cache_dir.resolve(strict=True),
        pose_fingerprint=candidate_pose_fingerprint,
        materialize_sequences=True,
    )
    base_pose_fingerprint = _digest(
        bindings.get("v4a_pose_fingerprint"),
        "v4a pose fingerprint",
    )
    base_sequences, base_snapshot = load_pose_cache_set(
        selected_records,
        cache_dir=v4a_cache_dir.resolve(strict=True),
        pose_fingerprint=base_pose_fingerprint,
        materialize_sequences=True,
    )
    _, base_full_snapshot = load_pose_cache_set(
        manifest.records,
        cache_dir=v4a_cache_dir.resolve(strict=True),
        pose_fingerprint=base_pose_fingerprint,
        materialize_sequences=False,
    )
    _require(
        v4a_ledger.get("successful_cache_snapshot") == base_full_snapshot.to_dict(),
        "v4a cache snapshot differs from the frozen ledger",
    )
    _require(
        base_full_snapshot.fingerprint
        == _digest(bindings.get("v4a_pose_cache_set_sha256"), "v4a cache-set SHA"),
        "v4a cache-set fingerprint mismatch",
    )
    _require(
        candidate.get("successful_cache_snapshot") == candidate_snapshot.to_dict(),
        "candidate cache snapshot differs from actual cache bytes",
    )
    candidate_sequence_by_hash = {
        hashlib.sha256(sequence.video_id.encode("utf-8")).hexdigest(): sequence
        for sequence in candidate_sequences
    }
    base_sequence_by_hash = {
        hashlib.sha256(sequence.video_id.encode("utf-8")).hexdigest(): sequence
        for sequence in base_sequences
    }
    base_cache_sha_by_hash = {
        hashlib.sha256(entry.video_id.encode("utf-8")).hexdigest(): entry.cache_sha256
        for entry in base_snapshot.entries
    }
    v4a_rows = v4a_ledger.get("caches")
    _require(isinstance(v4a_rows, list) and len(v4a_rows) == 337, "v4a ledger is not train337")
    v4a_by_hash: dict[str, Mapping[str, Any]] = {}
    for value in v4a_rows:
        row = _mapping(value, "v4a row")
        video_hash = hashlib.sha256(str(row.get("video_id", "")).encode("utf-8")).hexdigest()
        _require(video_hash not in v4a_by_hash, "duplicate v4a video hash")
        v4a_by_hash[video_hash] = row
    paired_rows = v4a_gate.get("paired_rows")
    _require(isinstance(paired_rows, list) and len(paired_rows) == 337, "v4a paired rows mismatch")
    projected_by_hash = {
        _digest(_mapping(row, "paired row").get("video_id_sha256"), "paired video hash"): dict(
            _mapping(row, "paired row")
        )
        for row in paired_rows
    }
    _require(set(projected_by_hash) == set(v4a_by_hash), "v4a artifact identity sets differ")

    candidate_rows = candidate.get("caches")
    _require(
        isinstance(candidate_rows, list) and len(candidate_rows) == expected_records,
        "candidate rows mismatch",
    )
    seen: set[str] = set()
    invariant_failures = 0
    base_mask_mismatches = 0
    base_valid_values: list[int] = []
    final_valid_values: list[int] = []
    source_values: list[int] = []
    base_longest_values: list[int] = []
    final_longest_values: list[int] = []
    candidate_total = 0
    attempted_total = 0
    detected_total = 0
    delta_rows: list[dict[str, Any]] = []
    for value in candidate_rows:
        row = _mapping(value, "candidate row")
        video_hash = hashlib.sha256(str(row.get("video_id", "")).encode("utf-8")).hexdigest()
        _require(
            video_hash in selected_hashes and video_hash not in seen, "candidate identity mismatch"
        )
        seen.add(video_hash)
        recovery = _mapping(row.get("recovery_audit"), "candidate recovery audit")
        base_row = v4a_by_hash[video_hash]
        base_recovery = _mapping(base_row.get("recovery_audit"), "v4a recovery audit")
        runtime_receipt = _mapping(
            recovery.get("keypointrcnn_runtime_receipt"),
            "KPRCNN runtime receipt",
        )
        runtime_ok = (
            runtime_receipt.get("torch_version") == "2.5.1+cu124"
            and runtime_receipt.get("torchvision_version") == "0.20.1+cu124"
            and runtime_receipt.get("cuda_version") == "12.4"
            and runtime_receipt.get("cudnn_version") == 90100
            and runtime_receipt.get("gpu_name") == "NVIDIA RTX A6000"
            and runtime_receipt.get("gpu_compute_capability") == "8.6"
            and runtime_receipt.get("deterministic_algorithms") is True
            and runtime_receipt.get("cublas_workspace_config") == ":4096:8"
            and runtime_receipt.get("allow_tf32") is False
            and runtime_receipt.get("cudnn_allow_tf32") is False
            and runtime_receipt.get("frozen_batchnorm_modules", 0) > 0
            and runtime_receipt.get("batchnorm_modules") == 0
            and runtime_receipt.get("frozen_batchnorm_eps") == 0.0
        )
        source_frames = _integer(recovery.get("source_frames"), "source frames")
        decoded_frames = _integer(recovery.get("decoded_segment_frames"), "decoded frames")
        padded_frames = _integer(recovery.get("padded_tail_frames"), "padded frames")
        pass0_valid = _integer(recovery.get("pass0_valid_frames"), "pass0 valid frames")
        base_valid = _integer(recovery.get("base_v4a_valid_frames"), "base v4a valid frames")
        final_valid = _integer(recovery.get("final_valid_frames"), "final valid frames")
        fill = _integer(recovery.get("keypointrcnn_fill_candidates"), "KPRCNN fills")
        attempted = _integer(recovery.get("keypointrcnn_frames_attempted"), "KPRCNN attempts")
        observed = _integer(recovery.get("keypointrcnn_frames_observed"), "KPRCNN observations")
        detected = _integer(
            recovery.get("keypointrcnn_frames_with_candidates"), "KPRCNN detected frames"
        )
        missing_eligible = _integer(
            recovery.get("keypointrcnn_missing_frames_eligible"),
            "KPRCNN missing-frame eligibility",
        )
        missing_detected = _integer(
            recovery.get("keypointrcnn_missing_frames_with_candidates"),
            "KPRCNN missing frames with candidates",
        )
        anchor_frames = _integer(
            recovery.get("keypointrcnn_v4a_anchor_frames"),
            "KPRCNN v4a anchor frames",
        )
        rejected_anchors = _integer(
            recovery.get("keypointrcnn_v4a_anchor_rejected_frames"),
            "KPRCNN rejected v4a anchors",
        )
        unusable_anchors = _integer(
            recovery.get("keypointrcnn_v4a_anchor_unusable_shape_frames"),
            "KPRCNN unusable v4a anchors",
        )
        candidates = _integer(recovery.get("keypointrcnn_candidate_total"), "KPRCNN candidates")
        maximum_candidates = _integer(
            recovery.get("keypointrcnn_max_candidates_per_frame"),
            "KPRCNN maximum candidates",
        )
        final_longest = _integer(recovery.get("final_longest_valid_run"), "final longest run")
        base_longest = _integer(base_recovery.get("final_longest_valid_run"), "v4a longest run")
        base_matches = base_valid == _integer(
            base_recovery.get("final_valid_frames"), "v4a final valid"
        ) and recovery.get("base_v4a_valid_mask_sha256") == base_recovery.get(
            "final_valid_mask_sha256"
        )
        candidate_sequence = candidate_sequence_by_hash[video_hash]
        base_sequence = base_sequence_by_hash[video_hash]
        candidate_mask = np.asarray(candidate_sequence.valid_mask, dtype=np.bool_)
        base_mask = np.asarray(base_sequence.valid_mask, dtype=np.bool_)
        actual_mask_sha256 = hashlib.sha256(
            bytes(int(value) for value in candidate_mask)
        ).hexdigest()
        base_coordinate_bytes = np.ascontiguousarray(base_sequence.xyz[base_mask]).tobytes(
            order="C"
        )
        candidate_base_coordinate_bytes = np.ascontiguousarray(
            candidate_sequence.xyz[base_mask]
        ).tobytes(order="C")
        base_coordinate_sha256 = hashlib.sha256(base_coordinate_bytes).hexdigest()
        candidate_base_coordinate_sha256 = hashlib.sha256(
            candidate_base_coordinate_bytes
        ).hexdigest()
        cache_matches = (
            candidate_sequence.num_frames == source_frames
            and base_sequence.num_frames == source_frames
            and candidate_sequence.fps == base_sequence.fps
            and int(np.count_nonzero(candidate_mask)) == final_valid
            and actual_mask_sha256 == recovery.get("final_valid_mask_sha256")
            and np.all(candidate_mask[base_mask])
            and candidate_base_coordinate_bytes == base_coordinate_bytes
            and base_coordinate_sha256
            == recovery.get("base_v4a_coordinate_sha256")
            == recovery.get("final_base_v4a_coordinate_sha256")
            == candidate_base_coordinate_sha256
            and row.get("base_v4a_cache_sha256") == base_cache_sha_by_hash[video_hash]
        )
        if not base_matches:
            base_mask_mismatches += 1
        invariant_ok = (
            source_frames == _integer(recovery.get("expected_segment_frames"), "expected frames")
            and decoded_frames + padded_frames == source_frames
            and pass0_valid == _integer(base_recovery.get("pass0_valid_frames"), "v4a pass0")
            and recovery.get("pass0_valid_mask_sha256")
            == base_recovery.get("pass0_valid_mask_sha256")
            and final_valid == base_valid + fill
            and final_valid >= base_valid >= pass0_valid
            and base_valid <= decoded_frames
            and observed == attempted == decoded_frames
            and missing_eligible == decoded_frames - base_valid
            and fill <= missing_detected <= missing_eligible
            and detected <= observed
            and anchor_frames + rejected_anchors + unusable_anchors == detected - missing_detected
            and detected <= candidates <= 4 * detected
            and maximum_candidates <= 4
            and final_longest <= final_valid
            and recovery.get("base_v4a_observations_preserved") is True
            and _finite(
                recovery.get("base_v4a_shared_coordinate_max_abs_error"),
                "base coordinate error",
            )
            == 0.0
            and recovery.get("base_v4a_coordinate_sha256")
            == recovery.get("final_base_v4a_coordinate_sha256")
            and recovery.get("keypointrcnn_model_asset_sha256") == expected_asset_sha
            and recovery.get("keypointrcnn_box_score_threshold") == 0.2
            and recovery.get("keypointrcnn_keypoint_logit_threshold") == 2.0
            and recovery.get("keypointrcnn_minimum_confident_keypoints") == 4
            and recovery.get("keypointrcnn_maximum_candidates_per_frame") == 4
            and recovery.get("keypointrcnn_state_dict_keys") == 313
            and recovery.get("keypointrcnn_parameter_count") == 59137258
            and recovery.get("keypointrcnn_association_coordinate_space")
            == "frame-normalized-absolute-xy-v1"
            and recovery.get("keypointrcnn_v4a_anchor_policy")
            == "nearest-shared-coco17-xy-minmax-singleton-threshold-v1"
            and recovery.get("keypointrcnn_fill_coordinate_space")
            == "selected-candidate-per-frame-minmax-xyz-v1"
            and recovery.get("keypointrcnn_v4a_anchor_distance_maximum") == 0.25
            and runtime_ok
            and recovery.get("decoder_fps_matches_v4a") is True
            and recovery.get("source_video_stat_stable") is True
            and recovery.get("source_video_sha256_before")
            == recovery.get("source_video_sha256_after")
            == row.get("video_sha256")
            and recovery.get("pose_coordinate_interpolation") is False
            and recovery.get("temporal_resampling") == "none_native_timeline"
            and cache_matches
        )
        if not invariant_ok:
            invariant_failures += 1
        base_valid_values.append(base_valid)
        final_valid_values.append(final_valid)
        source_values.append(source_frames)
        base_longest_values.append(base_longest)
        final_longest_values.append(final_longest)
        candidate_total += candidates
        attempted_total += attempted
        detected_total += detected
        projected = projected_by_hash[video_hash]
        projected.update(
            {
                "pass0_valid_frames": pass0_valid,
                "recovered_valid_frames": final_valid - pass0_valid,
                "final_valid_frames": final_valid,
                "observed_span_frames": _integer(
                    recovery.get("observed_span_frames"), "observed span"
                ),
                "v4_cached_valid_frames": final_valid,
                "source_coverage": final_valid / source_frames,
                "longest_run_fraction": final_longest / source_frames,
                "pass0_preserved": recovery.get("pass0_observations_preserved") is True,
            }
        )
        delta_rows.append(
            {
                "video_id_sha256": video_hash,
                "source_frames": source_frames,
                "v4a_final_valid_frames": base_valid,
                "v4d_final_valid_frames": final_valid,
                "valid_frame_gain": fill,
                "v4a_longest_run_fraction": base_longest / source_frames,
                "v4d_longest_run_fraction": final_longest / source_frames,
            }
        )
    _require(seen == selected_hashes, "candidate and selection identity sets differ")

    base_coverages = [
        valid / source for valid, source in zip(base_valid_values, source_values, strict=True)
    ]
    final_coverages = [
        valid / source for valid, source in zip(final_valid_values, source_values, strict=True)
    ]
    base_longest_fractions = [
        value / source for value, source in zip(base_longest_values, source_values, strict=True)
    ]
    final_longest_fractions = [
        value / source for value, source in zip(final_longest_values, source_values, strict=True)
    ]
    actual_metrics: dict[str, Any] = {
        "record_total": expected_records,
        "base_v4a_valid_frames_total": sum(base_valid_values),
        "candidate_final_valid_frames_total": sum(final_valid_values),
        "keypointrcnn_fill_frames_total": sum(final_valid_values) - sum(base_valid_values),
        "base_v4a_zero_video_total": sum(value == 0 for value in base_valid_values),
        "candidate_zero_video_total": sum(value == 0 for value in final_valid_values),
        "recovered_base_zero_video_total": sum(
            base == 0 and final > 0
            for base, final in zip(base_valid_values, final_valid_values, strict=True)
        ),
        "base_v4a_at_most_8_video_total": sum(value <= 8 for value in base_valid_values),
        "candidate_at_most_8_video_total": sum(value <= 8 for value in final_valid_values),
        "at_most_8_video_reduction": sum(value <= 8 for value in base_valid_values)
        - sum(value <= 8 for value in final_valid_values),
        "base_v4a_source_coverage_mean": float(np.mean(base_coverages)),
        "candidate_source_coverage_mean": float(np.mean(final_coverages)),
        "source_coverage_mean_gain_over_v4a": float(
            np.mean(final_coverages) - np.mean(base_coverages)
        ),
        "base_v4a_longest_run_fraction_mean": float(np.mean(base_longest_fractions)),
        "candidate_longest_run_fraction_mean": float(np.mean(final_longest_fractions)),
        "longest_run_fraction_mean_gain_over_v4a": float(
            np.mean(final_longest_fractions) - np.mean(base_longest_fractions)
        ),
        "candidate_longest_run_fraction_p25": _percentile(final_longest_fractions, 0.25),
        "keypointrcnn_frames_attempted_total": attempted_total,
        "keypointrcnn_frames_with_candidates_total": detected_total,
        "keypointrcnn_candidate_total": candidate_total,
        "base_v4a_mask_mismatch_total": base_mask_mismatches,
        "invariant_failure_total": invariant_failures,
    }
    zero_criteria = {
        "candidate_zero_video_total": _criterion(
            actual_metrics["candidate_zero_video_total"],
            relation="at_most",
            threshold=int(zero_thresholds["candidate_zero_video_maximum"]),
        ),
        "recovered_base_zero_video_total": _criterion(
            actual_metrics["recovered_base_zero_video_total"],
            relation="at_least",
            threshold=int(zero_thresholds["recovered_base_zero_video_minimum"]),
        ),
        "candidate_at_most_8_video_total": _criterion(
            actual_metrics["candidate_at_most_8_video_total"],
            relation="at_most",
            threshold=int(zero_thresholds["candidate_at_most_8_video_maximum"]),
        ),
        "keypointrcnn_fill_frames_total": _criterion(
            actual_metrics["keypointrcnn_fill_frames_total"],
            relation="at_least",
            threshold=int(zero_thresholds["keypointrcnn_fill_frames_minimum"]),
        ),
        "source_coverage_mean_gain_over_v4a": _criterion(
            actual_metrics["source_coverage_mean_gain_over_v4a"],
            relation="at_least",
            threshold=float(zero_thresholds["source_coverage_mean_gain_minimum"]),
        ),
        "base_v4a_mask_mismatch_total": _criterion(
            base_mask_mismatches, relation="at_most", threshold=0
        ),
        "invariant_failure_total": _criterion(invariant_failures, relation="at_most", threshold=0),
    }
    same_criteria = {
        "candidate_zero_video_total": _criterion(
            actual_metrics["candidate_zero_video_total"],
            relation="at_most",
            threshold=int(same_thresholds["candidate_zero_video_maximum"]),
        ),
        "candidate_at_most_8_video_total": _criterion(
            actual_metrics["candidate_at_most_8_video_total"],
            relation="at_most",
            threshold=int(same_thresholds["candidate_at_most_8_video_maximum"]),
        ),
        "keypointrcnn_fill_frames_total": _criterion(
            actual_metrics["keypointrcnn_fill_frames_total"],
            relation="at_least",
            threshold=int(same_thresholds["keypointrcnn_fill_frames_minimum"]),
        ),
        "source_coverage_mean_gain_over_v4a": _criterion(
            actual_metrics["source_coverage_mean_gain_over_v4a"],
            relation="at_least",
            threshold=float(same_thresholds["source_coverage_mean_gain_minimum"]),
        ),
        "longest_run_fraction_mean_gain_over_v4a": _criterion(
            actual_metrics["longest_run_fraction_mean_gain_over_v4a"],
            relation="at_least",
            threshold=float(same_thresholds["longest_run_fraction_mean_gain_minimum"]),
        ),
        "candidate_longest_run_fraction_p25": _criterion(
            actual_metrics["candidate_longest_run_fraction_p25"],
            relation="at_least",
            threshold=float(same_thresholds["candidate_longest_run_fraction_p25_minimum"]),
        ),
        "base_v4a_mask_mismatch_total": _criterion(
            base_mask_mismatches, relation="at_most", threshold=0
        ),
        "invariant_failure_total": _criterion(invariant_failures, relation="at_most", threshold=0),
    }

    projected_rows = tuple(projected_by_hash.values())
    projected_coverages = [float(row["source_coverage"]) for row in projected_rows]
    projected_longest = [float(row["longest_run_fraction"]) for row in projected_rows]
    projected_metrics: dict[str, Any] = {
        "record_total": 337,
        "v4_zero_video_total": sum(int(row["final_valid_frames"]) == 0 for row in projected_rows),
        "recovered_reference_zero_video_total": sum(
            int(row["reference_cached_valid_frames"]) == 0 and int(row["final_valid_frames"]) > 0
            for row in projected_rows
        ),
        "reference_usable_to_v4_zero_video_total": sum(
            int(row["reference_cached_valid_frames"]) > 0 and int(row["final_valid_frames"]) == 0
            for row in projected_rows
        ),
        "source_coverage_mean": float(np.mean(projected_coverages)),
        "source_coverage_p10": _percentile(projected_coverages, 0.10),
        "source_coverage_p25": _percentile(projected_coverages, 0.25),
        "source_coverage_median": _percentile(projected_coverages, 0.50),
        "observed_at_most_8_video_total": sum(
            int(row["final_valid_frames"]) <= 8 for row in projected_rows
        ),
        "longest_run_fraction_p10": _percentile(projected_longest, 0.10),
        "longest_run_fraction_median": _percentile(projected_longest, 0.50),
        "pass0_source_valid_count_mismatch_total": 0,
        "pass0_preservation_failure_total": invariant_failures,
        "pass0_shared_coordinate_max_abs_error": 0.0,
        "native_timeline_invariant_failure_total": invariant_failures,
    }
    projected_criteria = {
        "v4_zero_video_total": _criterion(
            projected_metrics["v4_zero_video_total"],
            relation="at_most",
            threshold=int(full_thresholds["v4_zero_video_maximum"]),
        ),
        "recovered_reference_zero_video_total": _criterion(
            projected_metrics["recovered_reference_zero_video_total"],
            relation="at_least",
            threshold=int(full_thresholds["recovered_reference_zero_video_minimum"]),
        ),
        "reference_usable_to_v4_zero_video_total": _criterion(
            projected_metrics["reference_usable_to_v4_zero_video_total"],
            relation="at_most",
            threshold=int(full_thresholds["reference_usable_to_v4_zero_video_maximum"]),
        ),
        "source_coverage_mean": _criterion(
            projected_metrics["source_coverage_mean"],
            relation="at_least",
            threshold=float(full_thresholds["source_coverage_mean_minimum"]),
        ),
        "source_coverage_p10": _criterion(
            projected_metrics["source_coverage_p10"],
            relation="at_least",
            threshold=float(full_thresholds["source_coverage_p10_minimum"]),
        ),
        "source_coverage_p25": _criterion(
            projected_metrics["source_coverage_p25"],
            relation="at_least",
            threshold=float(full_thresholds["source_coverage_p25_minimum"]),
        ),
        "source_coverage_median": _criterion(
            projected_metrics["source_coverage_median"],
            relation="at_least",
            threshold=float(full_thresholds["source_coverage_median_minimum"]),
        ),
        "observed_at_most_8_video_total": _criterion(
            projected_metrics["observed_at_most_8_video_total"],
            relation="at_most",
            threshold=int(full_thresholds["observed_at_most_8_video_maximum"]),
        ),
        "longest_run_fraction_p10": _criterion(
            projected_metrics["longest_run_fraction_p10"],
            relation="at_least",
            threshold=float(full_thresholds["longest_run_fraction_p10_minimum"]),
        ),
        "longest_run_fraction_median": _criterion(
            projected_metrics["longest_run_fraction_median"],
            relation="at_least",
            threshold=float(full_thresholds["longest_run_fraction_median_minimum"]),
        ),
        "base_v4a_mask_mismatch_total": _criterion(
            base_mask_mismatches, relation="at_most", threshold=0
        ),
        "invariant_failure_total": _criterion(invariant_failures, relation="at_most", threshold=0),
    }
    zero_passed = all(item["passed"] for item in zero_criteria.values())
    same_cost_passed = all(item["passed"] for item in same_criteria.values())
    projected_passed = all(item["passed"] for item in projected_criteria.values())
    passed = zero_passed if cohort == "zero11" else same_cost_passed and projected_passed
    return {
        "schema_version": 1,
        "artifact_type": f"pams_pose_recovery_v4d_train337_{cohort}_pilot_audit",
        "protocol": "ucfrep_526",
        "split": "train",
        "label_free": True,
        "passed": bool(passed),
        "zero11_passed": bool(zero_passed),
        "same39_cost_gate_passed": bool(same_cost_passed),
        "projected_full337_gate_passed": bool(projected_passed),
        "projection": {
            "kind": "retain_exact_v4a_nonpilot_plus_observed_v4d_pilot",
            "justification": (
                "v4d consumes and bitwise retains v4a caches, so every nonpilot row "
                "uses its exact frozen v4a observation rather than a pass0-only bound"
            ),
        },
        "bindings": {
            "gate_sha256": _sha256_file(gate_path.resolve(strict=True)),
            "selection_sha256": _sha256_file(selection_path.resolve(strict=True)),
            "v4a_ledger_sha256": _sha256_file(v4a_ledger_path.resolve(strict=True)),
            "v4a_paired_gate_sha256": _sha256_file(v4a_paired_gate_path.resolve(strict=True)),
            "candidate_ledger_sha256": _sha256_file(candidate_ledger_path.resolve(strict=True)),
            "candidate_pose_fingerprint": candidate.get("pose_fingerprint"),
            "candidate_cache_set_sha256": candidate_snapshot.fingerprint,
        },
        "actual_metrics": actual_metrics,
        "zero11_criteria": zero_criteria,
        "same39_criteria": same_criteria,
        "projected_metrics": projected_metrics,
        "projected_criteria": projected_criteria,
        "delta_rows": delta_rows,
        "mount_audit": {
            "train_candidate_ledger_mounted": True,
            "train_candidate_caches_mounted": True,
            "train_v4a_artifacts_mounted": True,
            "source_videos_mounted": False,
            "dev84_mounted": False,
            "test105_mounted": False,
            "targets_mounted": False,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("zero11", "same39"), required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-commitment", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--v4a-cache-dir", type=Path, required=True)
    parser.add_argument("--v4a-ledger", type=Path, required=True)
    parser.add_argument("--v4a-paired-gate", type=Path, required=True)
    parser.add_argument("--candidate-cache-dir", type=Path, required=True)
    parser.add_argument("--candidate-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = audit_pilot(
            cohort=args.cohort,
            gate_path=args.gate,
            train_input_path=args.train_input,
            train_commitment_path=args.train_commitment,
            selection_path=args.selection,
            v4a_cache_dir=args.v4a_cache_dir,
            v4a_ledger_path=args.v4a_ledger,
            v4a_paired_gate_path=args.v4a_paired_gate,
            candidate_cache_dir=args.candidate_cache_dir,
            candidate_ledger_path=args.candidate_ledger,
        )
        _write_exclusive(args.output, result)
    except (OSError, TypeError, ValueError, PilotAuditError) as exc:
        print(f"v4d pilot audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["actual_metrics"], sort_keys=True, allow_nan=False))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
