#!/usr/bin/env python3
"""Strict, label-free contracts shared by the v4d full337 runner and gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from pams.data import (
    PoseCacheSetSnapshot,
    UnlabeledVideoRecord,
    load_pose_cache_set,
    pose_input_identity_sha256,
)

PARENT_V4D_SOURCE_REVISION = "081c3c1383953c7f2e30f69b2e7bcabeaa2503f9"
CONTAINER_IMAGE_ID = "sha256:0a4d42c2d9911f147a17860e4e15095746b21c4e618e4fc1b20b62dd443c5898"
TRAIN337_SIDECAR_SHA256 = "f95df0050df21f05bbc9b42d0154470714dde279e04cb8b00d0212bf49057d16"
TRAIN337_COMMITMENT_SHA256 = "85d41d2f59872e0900e9058481efbdf6bce91407d4c26bcde0a6547776454e53"
TRAIN337_IDENTITY_SHA256 = "88367535e296213377c366ed69abbd1e808c38049d79bb948091688e4c5b7b3f"
V4A_LEDGER_SHA256 = "4cbba0d0f678cfdbd2c99752bbb55a8ceeb3aaa0b0b994bb6195b55cd0bc6018"
V4A_PAIRED_GATE_SHA256 = "faf086277f4d0a0687a266b64edc6ae45a2ca0ab4b39cc8fc4da644350c571e5"
V4A_POSE_FINGERPRINT = "817013890533cd19e6c791969c35c3699d56d0e6656768ce64199bc30ceebe9c"
V4A_CACHE_SET_SHA256 = "22629d0ddc5d892bb53bcef78c887da4ff9e054f5a17bd5e7ea4fbc43a03c662"
V4D_CONFIG_FILE_SHA256 = "9933126d16735d42108203f512168e0eb438434f954fbcc969c5db622f5b6073"
V4D_CONFIG_FINGERPRINT = "587ad8a427e6d387a23b142e57cfac4a8bd49ffed93176b477000aeb2c2a5125"
V4D_POSE_FINGERPRINT = "4cc1f905cfb3484dccc1efc480e3fa59e4a106ffbe20528a85201a3fbd85b5d8"
MODEL_ASSET_SHA256 = "fc266e953d2b302cdcbb9ae66f71f6b0d4649928bf02dc573961e361e4918926"
SAME39_IDENTITY_SHA256 = "afc8c7a10dfc624f1635e792e49a8c08180ca9f32fab29ff24e3972a0718f554"
SAME39_FAILURE_RECEIPT_SHA256 = "91c702663e6be37bec59cbf558ada047435c0e8f4c4c8efd951f88d505039a00"
SAME39_AUDIT_SHA256 = "d25a9fb6da457677cc2c33e79725cedfc36ed9b5555e762878569d3f43e754f5"
SAME39_LEDGER_SHA256 = "2c89a66c3203e73db0a6597a7e6cb0e0912c81866795499d444ea3d66c2db7fc"
SAME39_SELECTION_SHA256 = "c3a7d1113c4be28e0a9c06e1d59eeea31383a266e3400d7559fd0fab605123d3"
SAME39_CACHE_SET_SHA256 = "ef4a2e341128598fbc94a6c883ef797f780a2cd0ae8100b937d3e679c9f2f7d1"

FORBIDDEN_KEYS = frozenset(
    {
        "action",
        "action_label",
        "action_labels",
        "actions",
        "class",
        "class_name",
        "classes",
        "count",
        "count_label",
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
V4A_LEDGER_KEYS = frozenset(
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
V4A_CACHE_ROW_KEYS = frozenset(
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
PILOT_LEDGER_KEYS = frozenset(
    {
        "artifact_type",
        "caches",
        "cohort",
        "commitment_file_sha256",
        "completed",
        "config_file_sha256",
        "config_fingerprint",
        "container_image_id",
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
        "source_revision",
        "split",
        "successful_cache_snapshot",
    }
)
PILOT_CACHE_ROW_KEYS = frozenset(
    {
        "annotation_sha256",
        "base_v4a_cache_sha256",
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
PILOT_SELECTION_KEYS = frozenset(
    {
        "artifact_type",
        "bindings",
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
PILOT_SELECTION_ROW_KEYS = frozenset({"video_id_sha256", "v4a_final_valid_frames"})
PILOT_FAILURE_RECEIPT_KEYS = frozenset(
    {
        "artifact_type",
        "artifacts",
        "batch_fallback_used",
        "cohort",
        "container_image_id",
        "exit_status",
        "failed_phase",
        "gpu_memory_used_mib_preflight",
        "gpu_utilization_percent_preflight",
        "gpu_uuid",
        "oom_killed",
        "physical_gpu_index",
        "schema_version",
        "source_revision",
    }
)
EXTRACTION_AUTHORIZATION_KEYS = frozenset(
    {
        "artifact_type",
        "authorization_basis",
        "authorization_protocol_version",
        "authorization_scope",
        "baseline_training_authorized",
        "bindings",
        "container_image_id",
        "full337_pose_extraction_authorized",
        "label_free",
        "protocol",
        "schema_version",
        "source_revision",
        "split",
        "training_authorization_requires",
        "training_runner_must_reject",
    }
)
PILOT_AUDIT_KEYS = frozenset(
    {
        "actual_metrics",
        "artifact_type",
        "baseline_training_authorized",
        "bindings",
        "config_file_sha256",
        "config_fingerprint",
        "container_image_id",
        "delta_rows",
        "full337_pose_extraction_authorized",
        "label_free",
        "mount_audit",
        "passed",
        "projected_criteria",
        "projected_full337_gate_passed",
        "projected_metrics",
        "projection",
        "protocol",
        "same39_cost_gate_passed",
        "same39_criteria",
        "schema_version",
        "source_revision",
        "split",
        "track_identity_risk",
        "training_authorization_requires",
        "zero11_criteria",
        "zero11_passed",
    }
)


class Full337ContractError(RuntimeError):
    """Raised when an immutable full337 input or authorization is invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Full337ContractError(message)


def digest(value: Any, role: str) -> str:
    text = str(value)
    require(
        len(text) == 64 and all(character in "0123456789abcdef" for character in text),
        f"{role} must be lowercase SHA-256",
    )
    return text


def integer(value: Any, role: str) -> int:
    require(type(value) is int and value >= 0, f"{role} must be a non-negative integer")
    return int(value)


def finite(value: Any, role: str) -> float:
    require(
        type(value) in {int, float} and math.isfinite(float(value)),
        f"{role} must be finite",
    )
    return float(value)


def sha256_file(path: Path) -> str:
    source = path.resolve(strict=True)
    hasher = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_file_bytes(path: Path) -> bytes:
    source = path.resolve(strict=True)
    before = source.stat()
    require(source.is_file(), f"not a regular file: {source}")
    with source.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        payload = handle.read()
        closed = os.fstat(handle.fileno())
    after = source.stat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    require(
        all(
            getattr(before, field)
            == getattr(opened, field)
            == getattr(closed, field)
            == getattr(after, field)
            for field in fields
        ),
        f"file changed while reading: {source}",
    )
    require(len(payload) == after.st_size, f"file size changed while reading: {source}")
    return payload


def write_bytes_exclusive(path: Path, payload: bytes) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        if isinstance(key, str) and key.casefold() in FORBIDDEN_KEYS:
            raise Full337ContractError(f"{role} contains forbidden sensitive field {key!r}")


def _reject_non_finite(value: str) -> None:
    raise Full337ContractError(f"non-finite JSON constant is forbidden: {value}")


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON field is forbidden: {key!r}")
        require(key.casefold() not in FORBIDDEN_KEYS, f"forbidden sensitive field: {key!r}")
        result[key] = value
    return result


def load_strict_json(path: Path, *, role: str) -> tuple[dict[str, Any], str]:
    payload = stable_file_bytes(path)
    try:
        encoded = payload.decode("utf-8")
        _reject_forbidden_keys(encoded, role=role)
        value = json.loads(
            encoded,
            object_pairs_hook=_unique_pairs,
            parse_constant=_reject_non_finite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Full337ContractError(f"{role} must be strict UTF-8 JSON") from exc
    require(isinstance(value, dict), f"{role} root must be an object")
    return value, hashlib.sha256(payload).hexdigest()


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.nodes.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        require(key not in result, f"duplicate YAML field is forbidden: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_gate(path: Path) -> tuple[dict[str, Any], str]:
    payload = stable_file_bytes(path)
    try:
        value = yaml.load(payload.decode("utf-8"), Loader=_UniqueKeyLoader)
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise Full337ContractError("full337 gate must be strict UTF-8 YAML") from exc
    require(isinstance(value, dict), "full337 gate root must be a mapping")
    return value, hashlib.sha256(payload).hexdigest()


def validate_full337_gate(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], str]:
    gate, gate_sha256 = load_gate(path)
    require(
        gate_sha256 == digest(expected_sha256, "full337 gate SHA"),
        "full337 gate SHA mismatch",
    )
    require(
        set(gate)
        == {
            "artifact_type",
            "anchor_risk_thresholds",
            "bindings",
            "coverage_thresholds",
            "expected_records",
            "integrity_thresholds",
            "measurement_protocol",
            "periodicity_thresholds",
            "preregistration",
            "protocol",
            "schema_version",
            "split",
            "track_quality_thresholds",
        },
        "full337 gate root schema mismatch",
    )
    require(
        gate.get("schema_version") == 1
        and gate.get("artifact_type") == "pams_pose_recovery_v4d_train337_full_gate"
        and gate.get("protocol") == "ucfrep_526"
        and gate.get("split") == "train"
        and gate.get("expected_records") == 337,
        "full337 gate protocol mismatch",
    )
    require(
        dict(mapping(gate.get("preregistration"), "full337 preregistration"))
        == {
            "frozen_before_full337_outputs": True,
            "output_independent_thresholds": True,
            "same39_authorizes_extraction_only": True,
            "full337_pass_required_for_baseline_training": True,
        },
        "full337 preregistration mismatch",
    )
    require(
        dict(mapping(gate.get("measurement_protocol"), "full337 measurement protocol"))
        == {
            "periodicity_coordinates": "xy_only",
            "periodicity_signal": "precomputed_pose_velocity_vectors",
            "eligible_velocity_pair_policy": (
                "adjacent_both_valid_and_same_extractor_origin_only"
            ),
            "excluded_velocity_pairs": ["invalid_gap", "base_to_fill", "fill_to_base"],
            "minimum_anchor_frames": 2,
            "minimum_anchor_fraction_of_final_valid_frames": 0.02,
        },
        "full337 measurement protocol changed",
    )
    require(
        dict(mapping(gate.get("bindings"), "full337 bindings"))
        == {
            "container_image_id": CONTAINER_IMAGE_ID,
            "keypointrcnn_model_asset_sha256": MODEL_ASSET_SHA256,
            "same39_audit_sha256": SAME39_AUDIT_SHA256,
            "same39_candidate_cache_set_sha256": SAME39_CACHE_SET_SHA256,
            "same39_failure_receipt_sha256": SAME39_FAILURE_RECEIPT_SHA256,
            "same39_ledger_sha256": SAME39_LEDGER_SHA256,
            "same39_selection_sha256": SAME39_SELECTION_SHA256,
            "train337_commitment_sha256": TRAIN337_COMMITMENT_SHA256,
            "train337_identity_sha256": TRAIN337_IDENTITY_SHA256,
            "train337_sidecar_sha256": TRAIN337_SIDECAR_SHA256,
            "v2_extraction_authorization_receipt_sha256_must_be_caller_pinned": True,
            "v4a_ledger_sha256": V4A_LEDGER_SHA256,
            "v4a_paired_gate_sha256": V4A_PAIRED_GATE_SHA256,
            "v4a_pose_cache_set_sha256": V4A_CACHE_SET_SHA256,
            "v4a_pose_fingerprint": V4A_POSE_FINGERPRINT,
            "v4d_config_file_sha256": V4D_CONFIG_FILE_SHA256,
            "v4d_config_fingerprint": V4D_CONFIG_FINGERPRINT,
            "v4d_parent_source_revision": PARENT_V4D_SOURCE_REVISION,
            "v4d_pose_fingerprint": V4D_POSE_FINGERPRINT,
        },
        "full337 immutable bindings mismatch",
    )
    require(
        dict(mapping(gate.get("coverage_thresholds"), "coverage thresholds"))
        == {
            "v4_zero_video_maximum": 8,
            "recovered_reference_zero_video_minimum": 8,
            "reference_usable_to_v4_zero_video_maximum": 0,
            "source_coverage_mean_minimum": 0.80,
            "source_coverage_p10_minimum": 0.25,
            "source_coverage_p25_minimum": 0.60,
            "source_coverage_median_minimum": 0.97,
            "observed_at_most_8_video_maximum": 4,
            "longest_run_fraction_p10_minimum": 0.15,
            "longest_run_fraction_median_minimum": 0.80,
        },
        "coverage thresholds changed",
    )
    require(
        dict(mapping(gate.get("track_quality_thresholds"), "track thresholds"))
        == {
            "mask_transition_rate_p90_maximum": 0.20,
            "base_fill_boundary_shape_jump_p95_maximum": 0.45,
            "contiguous_cycle_supported_video_fraction_minimum": 0.70,
        },
        "track-quality thresholds changed",
    )
    require(
        dict(mapping(gate.get("periodicity_thresholds"), "periodicity thresholds"))
        == {
            "period_confidence_p25_minimum": 0.03,
            "period_confidence_median_minimum": 0.06,
            "periodic_track_video_fraction_minimum": 0.70,
            "periodic_track_confidence_minimum": 0.03,
            "eligible_velocity_fraction_p10_minimum": 0.50,
            "period_velocity_pair_count_p10_minimum": 8,
        },
        "periodicity thresholds changed",
    )
    require(
        dict(mapping(gate.get("anchor_risk_thresholds"), "anchor-risk thresholds"))
        == {
            "weakly_anchored_filled_video_fraction_maximum": 0.05,
            "weakly_anchored_filled_low_periodicity_video_maximum": 4,
        },
        "anchor-risk thresholds changed",
    )
    require(
        dict(mapping(gate.get("integrity_thresholds"), "integrity thresholds"))
        == {
            "invariant_failure_maximum": 0,
            "v4a_bitwise_mismatch_maximum": 0,
            "same39_bytewise_mismatch_maximum": 0,
            "nonfinite_metric_maximum": 0,
        },
        "integrity thresholds changed",
    )
    return gate, gate_sha256


def mapping(value: Any, role: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{role} must be an object")
    return value


@dataclass(frozen=True, slots=True)
class V4AInputs:
    ledger: Mapping[str, Any]
    paired_gate: Mapping[str, Any]
    snapshot: PoseCacheSetSnapshot
    rows_by_id: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class Same39PilotEvidence:
    failure_receipt: Mapping[str, Any]
    audit: Mapping[str, Any]
    ledger: Mapping[str, Any]
    selection: Mapping[str, Any]
    records: tuple[UnlabeledVideoRecord, ...]
    snapshot: PoseCacheSetSnapshot
    rows_by_id: Mapping[str, Mapping[str, Any]]
    failure_receipt_sha256: str
    audit_sha256: str
    ledger_sha256: str
    selection_sha256: str


@dataclass(frozen=True, slots=True)
class Same39Authorization:
    authorization_receipt: Mapping[str, Any]
    evidence: Same39PilotEvidence
    authorization_receipt_sha256: str

    @property
    def audit(self) -> Mapping[str, Any]:
        return self.evidence.audit

    @property
    def ledger(self) -> Mapping[str, Any]:
        return self.evidence.ledger

    @property
    def selection(self) -> Mapping[str, Any]:
        return self.evidence.selection

    @property
    def records(self) -> tuple[UnlabeledVideoRecord, ...]:
        return self.evidence.records

    @property
    def snapshot(self) -> PoseCacheSetSnapshot:
        return self.evidence.snapshot

    @property
    def rows_by_id(self) -> Mapping[str, Mapping[str, Any]]:
        return self.evidence.rows_by_id


def validate_v4a_inputs(
    records: Sequence[UnlabeledVideoRecord],
    *,
    cache_dir: Path,
    ledger_path: Path,
    paired_gate_path: Path,
) -> V4AInputs:
    source = tuple(records)
    require(len(source) == 337, "v4a validation requires exact train337")
    ledger, ledger_sha = load_strict_json(ledger_path, role="v4a ledger")
    paired, paired_sha = load_strict_json(paired_gate_path, role="v4a paired gate")
    require(ledger_sha == V4A_LEDGER_SHA256, "v4a ledger SHA mismatch")
    require(paired_sha == V4A_PAIRED_GATE_SHA256, "v4a paired gate SHA mismatch")
    require(set(ledger) == V4A_LEDGER_KEYS, "v4a ledger root schema mismatch")
    require(
        ledger.get("schema_version") == 2
        and ledger.get("protocol") == "ucfrep_526"
        and ledger.get("split") == "train"
        and ledger.get("input_kind") == "label_free_sidecar",
        "v4a ledger protocol mismatch",
    )
    require(
        ledger.get("selected") == ledger.get("completed") == ledger.get("extracted") == 337
        and ledger.get("failed") == ledger.get("skipped") == 0,
        "v4a ledger is not a successful train337 extraction",
    )
    require(ledger.get("identity_sha256") == TRAIN337_IDENTITY_SHA256, "v4a identity mismatch")
    require(ledger.get("sidecar_sha256") == TRAIN337_SIDECAR_SHA256, "v4a sidecar mismatch")
    require(
        ledger.get("commitment_file_sha256") == TRAIN337_COMMITMENT_SHA256,
        "v4a commitment mismatch",
    )
    require(ledger.get("pose_fingerprint") == V4A_POSE_FINGERPRINT, "v4a pose mismatch")
    require(
        set(paired)
        == {
            "artifact_type",
            "bindings",
            "criteria",
            "metrics",
            "mount_audit",
            "paired_rows",
            "paired_rows_sha256",
            "passed",
            "protocol",
            "schema_version",
            "split",
        },
        "v4a paired-gate schema mismatch",
    )
    require(
        paired.get("schema_version") == 1
        and paired.get("artifact_type") == "pams_pose_recovery_v4a_train337_paired_audit"
        and paired.get("protocol") == "ucfrep_526"
        and paired.get("split") == "train",
        "v4a paired-gate protocol mismatch",
    )
    paired_bindings = mapping(paired.get("bindings"), "v4a paired bindings")
    require(paired_bindings.get("v4_ledger_sha256") == V4A_LEDGER_SHA256, "paired ledger mismatch")
    require(
        paired_bindings.get("v4_pose_cache_set_sha256") == V4A_CACHE_SET_SHA256,
        "paired cache-set mismatch",
    )
    rows = ledger.get("caches")
    require(isinstance(rows, list) and len(rows) == 337, "v4a cache rows mismatch")
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for value in rows:
        row = mapping(value, "v4a cache row")
        require(set(row) == V4A_CACHE_ROW_KEYS, "v4a cache-row schema mismatch")
        identifier = str(row.get("video_id", ""))
        require(identifier and identifier not in rows_by_id, "duplicate v4a video ID")
        rows_by_id[identifier] = row
    require(set(rows_by_id) == {record.video_id for record in source}, "v4a identity set mismatch")
    _, snapshot = load_pose_cache_set(
        source,
        cache_dir=cache_dir.resolve(strict=True),
        pose_fingerprint=V4A_POSE_FINGERPRINT,
        materialize_sequences=False,
    )
    require(snapshot.fingerprint == V4A_CACHE_SET_SHA256, "v4a cache bytes changed")
    require(ledger.get("successful_cache_snapshot") == snapshot.to_dict(), "v4a snapshot mismatch")
    return V4AInputs(
        ledger=ledger,
        paired_gate=paired,
        snapshot=snapshot,
        rows_by_id=rows_by_id,
    )


def validate_same39_failed_pilot(
    records: Sequence[UnlabeledVideoRecord],
    *,
    expected_failure_receipt_sha256: str,
    failure_receipt_path: Path,
    audit_path: Path,
    ledger_path: Path,
    selection_path: Path,
    cache_dir: Path,
) -> Same39PilotEvidence:
    source = tuple(records)
    require(len(source) == 337, "same39 evidence requires exact train337")
    expected_failure = digest(
        expected_failure_receipt_sha256,
        "caller-pinned same39 failure receipt",
    )
    require(
        expected_failure == SAME39_FAILURE_RECEIPT_SHA256,
        "same39 failure receipt is not the committed frozen artifact",
    )
    failure, failure_sha = load_strict_json(
        failure_receipt_path,
        role="same39 failure receipt",
    )
    audit, audit_sha = load_strict_json(audit_path, role="same39 audit")
    ledger, ledger_sha = load_strict_json(ledger_path, role="same39 ledger")
    selection, selection_sha = load_strict_json(selection_path, role="same39 selection")
    require(failure_sha == expected_failure, "same39 failure receipt SHA mismatch")
    require(audit_sha == SAME39_AUDIT_SHA256, "same39 audit is not the committed artifact")
    require(ledger_sha == SAME39_LEDGER_SHA256, "same39 ledger is not the committed artifact")
    require(
        selection_sha == SAME39_SELECTION_SHA256,
        "same39 selection is not the committed artifact",
    )
    require(set(failure) == PILOT_FAILURE_RECEIPT_KEYS, "same39 failure-receipt schema mismatch")
    require(set(audit) == PILOT_AUDIT_KEYS, "same39 audit schema mismatch")
    require(set(ledger) == PILOT_LEDGER_KEYS, "same39 ledger schema mismatch")
    require(set(selection) == PILOT_SELECTION_KEYS, "same39 selection schema mismatch")
    require(
        failure.get("schema_version") == 1
        and failure.get("artifact_type") == "pams_pose_recovery_v4d_pilot_failure_receipt"
        and failure.get("source_revision") == PARENT_V4D_SOURCE_REVISION
        and failure.get("container_image_id") == CONTAINER_IMAGE_ID
        and failure.get("cohort") == "same39"
        and failure.get("failed_phase") == "audit"
        and failure.get("exit_status") == 2
        and failure.get("oom_killed") is False
        and failure.get("batch_fallback_used") is False
        and failure.get("physical_gpu_index") == 1,
        "same39 failure receipt is not the expected scientific gate rejection",
    )
    failure_artifacts = mapping(failure.get("artifacts"), "same39 failure artifacts")
    require(
        set(failure_artifacts)
        == {
            "audit/audit.inspect.post.json",
            "audit/audit.inspect.pre.json",
            "audit/extract.inspect.post.json",
            "audit/extract.inspect.pre.json",
            "audit/selection.json",
            "ledgers/same39.json",
        },
        "same39 failure artifact schema mismatch",
    )
    require(
        failure_artifacts.get("ledgers/same39.json") == ledger_sha
        and failure_artifacts.get("audit/selection.json") == selection_sha,
        "same39 failure artifact hash chain mismatch",
    )
    for role, value in failure_artifacts.items():
        digest(value, f"same39 failure artifact {role}")
    require(
        audit.get("schema_version") == 1
        and audit.get("artifact_type") == "pams_pose_recovery_v4d_train337_same39_pilot_audit"
        and audit.get("source_revision") == PARENT_V4D_SOURCE_REVISION
        and audit.get("container_image_id") == CONTAINER_IMAGE_ID
        and audit.get("protocol") == "ucfrep_526"
        and audit.get("split") == "train"
        and audit.get("label_free") is True
        and audit.get("passed") is False
        and audit.get("zero11_passed") is True
        and audit.get("same39_cost_gate_passed") is True
        and audit.get("full337_pose_extraction_authorized") is False
        and audit.get("baseline_training_authorized") is False,
        "same39 audit is not the frozen failed-pilot evidence required by v2",
    )
    actual = mapping(audit.get("actual_metrics"), "same39 actual metrics")
    require(
        set(actual)
        == {
            "at_most_8_video_reduction",
            "base_v4a_at_most_8_video_total",
            "base_v4a_longest_run_fraction_mean",
            "base_v4a_mask_mismatch_total",
            "base_v4a_source_coverage_mean",
            "base_v4a_valid_frames_total",
            "base_v4a_zero_video_total",
            "candidate_at_most_8_video_total",
            "candidate_final_valid_frames_total",
            "candidate_longest_run_fraction_mean",
            "candidate_longest_run_fraction_p25",
            "candidate_source_coverage_mean",
            "candidate_zero_video_total",
            "invariant_failure_total",
            "keypointrcnn_candidate_total",
            "keypointrcnn_fill_frames_total",
            "keypointrcnn_frames_attempted_total",
            "keypointrcnn_frames_with_candidates_total",
            "longest_run_fraction_mean_gain_over_v4a",
            "record_total",
            "recovered_base_zero_video_total",
            "source_coverage_mean_gain_over_v4a",
        },
        "same39 actual-metric schema mismatch",
    )
    require(actual.get("record_total") == 39, "same39 actual record count mismatch")
    same39_thresholds: dict[str, tuple[str, float]] = {
        "base_v4a_mask_mismatch_total": ("at_most", 0.0),
        "candidate_at_most_8_video_total": ("at_most", 4.0),
        "candidate_longest_run_fraction_p25": ("at_least", 0.10),
        "candidate_zero_video_total": ("at_most", 8.0),
        "invariant_failure_total": ("at_most", 0.0),
        "keypointrcnn_fill_frames_total": ("at_least", 390.0),
        "longest_run_fraction_mean_gain_over_v4a": ("at_least", 0.05),
        "source_coverage_mean_gain_over_v4a": ("at_least", 0.10),
    }
    zero11_thresholds: dict[str, tuple[str, float]] = {
        "base_v4a_mask_mismatch_total": ("at_most", 0.0),
        "candidate_at_most_8_video_total": ("at_most", 5.0),
        "candidate_zero_video_total": ("at_most", 5.0),
        "invariant_failure_total": ("at_most", 0.0),
        "keypointrcnn_fill_frames_total": ("at_least", 60.0),
        "recovered_base_zero_video_total": ("at_least", 6.0),
        "source_coverage_mean_gain_over_v4a": ("at_least", 0.03),
    }
    for role, expected in (
        ("same39_criteria", same39_thresholds),
        ("zero11_criteria", zero11_thresholds),
    ):
        criteria = mapping(audit.get(role), role)
        require(set(criteria) == set(expected), f"{role} schema mismatch")
        for name, (relation, threshold) in expected.items():
            criterion = mapping(criteria.get(name), f"{role}.{name}")
            require(
                set(criterion) == {"passed", "relation", "threshold", "value"},
                f"{role}.{name} schema mismatch",
            )
            require(
                criterion.get("passed") is True
                and criterion.get("relation") == relation
                and finite(criterion.get("threshold"), f"{role}.{name}.threshold") == threshold
                and finite(criterion.get("value"), f"{role}.{name}.value")
                == finite(actual.get(name), f"actual_metrics.{name}"),
                f"{role}.{name} did not pass its frozen measured criterion",
            )
    audit_bindings = mapping(audit.get("bindings"), "same39 audit bindings")
    require(audit_bindings.get("candidate_ledger_sha256") == ledger_sha, "audit ledger mismatch")
    require(audit_bindings.get("selection_sha256") == selection_sha, "audit selection mismatch")
    require(
        audit_bindings.get("candidate_pose_fingerprint") == V4D_POSE_FINGERPRINT,
        "audit pose fingerprint mismatch",
    )
    require(
        ledger.get("schema_version") == 2
        and ledger.get("artifact_type") == "pams_pose_recovery_v4d_train337_same39_pilot_ledger"
        and ledger.get("source_revision") == PARENT_V4D_SOURCE_REVISION
        and ledger.get("container_image_id") == CONTAINER_IMAGE_ID
        and ledger.get("protocol") == "ucfrep_526"
        and ledger.get("split") == "train"
        and ledger.get("cohort") == "same39"
        and ledger.get("input_kind") == "label_free_train337_hashed_pilot_subset"
        and ledger.get("recovery_version") == "v4d",
        "same39 ledger protocol mismatch",
    )
    require(
        ledger.get("selected") == ledger.get("completed") == ledger.get("extracted") == 39
        and ledger.get("failed") == ledger.get("skipped") == 0
        and ledger.get("failures") == [],
        "same39 ledger is incomplete",
    )
    require(
        ledger.get("sidecar_sha256") == TRAIN337_SIDECAR_SHA256
        and ledger.get("commitment_file_sha256") == TRAIN337_COMMITMENT_SHA256
        and ledger.get("full_train_identity_sha256") == TRAIN337_IDENTITY_SHA256
        and ledger.get("identity_sha256") == SAME39_IDENTITY_SHA256
        and ledger.get("config_file_sha256") == V4D_CONFIG_FILE_SHA256
        and ledger.get("config_fingerprint") == V4D_CONFIG_FINGERPRINT
        and ledger.get("pose_fingerprint") == V4D_POSE_FINGERPRINT
        and ledger.get("selection_sha256") == selection_sha,
        "same39 ledger binding mismatch",
    )
    require(
        selection.get("schema_version") == 1
        and selection.get("artifact_type")
        == "pams_pose_recovery_v4d_train337_same39_pilot_selection"
        and selection.get("protocol") == "ucfrep_526"
        and selection.get("split") == "train"
        and selection.get("label_free") is True
        and selection.get("record_total") == 39
        and selection.get("selected_identity_sha256") == SAME39_IDENTITY_SHA256,
        "same39 selection mismatch",
    )
    selection_rows = selection.get("rows")
    require(isinstance(selection_rows, list) and len(selection_rows) == 39, "same39 rows mismatch")
    selected_hashes: set[str] = set()
    for value in selection_rows:
        row = mapping(value, "same39 selection row")
        require(set(row) == PILOT_SELECTION_ROW_KEYS, "same39 selection-row schema mismatch")
        selected_hashes.add(digest(row.get("video_id_sha256"), "same39 video hash"))
    require(len(selected_hashes) == 39, "same39 selection contains duplicate identities")
    selected_records = tuple(
        record
        for record in source
        if hashlib.sha256(record.video_id.encode("utf-8")).hexdigest() in selected_hashes
    )
    require(len(selected_records) == 39, "same39 selection is not a train337 subset")
    require(
        pose_input_identity_sha256(selected_records) == SAME39_IDENTITY_SHA256,
        "same39 selected identity mismatch",
    )
    rows = ledger.get("caches")
    require(isinstance(rows, list) and len(rows) == 39, "same39 cache rows mismatch")
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    for value in rows:
        row = mapping(value, "same39 cache row")
        require(set(row) == PILOT_CACHE_ROW_KEYS, "same39 cache-row schema mismatch")
        identifier = str(row.get("video_id", ""))
        require(identifier and identifier not in rows_by_id, "duplicate same39 video ID")
        rows_by_id[identifier] = row
    require(set(rows_by_id) == {record.video_id for record in selected_records}, "same39 row IDs mismatch")
    _, snapshot = load_pose_cache_set(
        selected_records,
        cache_dir=cache_dir.resolve(strict=True),
        pose_fingerprint=V4D_POSE_FINGERPRINT,
        materialize_sequences=False,
    )
    require(ledger.get("successful_cache_snapshot") == snapshot.to_dict(), "same39 cache bytes changed")
    require(
        snapshot.fingerprint == SAME39_CACHE_SET_SHA256,
        "same39 candidate cache set is not the committed byte set",
    )
    require(
        audit_bindings.get("candidate_cache_set_sha256") == snapshot.fingerprint,
        "same39 audit cache-set mismatch",
    )
    return Same39PilotEvidence(
        failure_receipt=failure,
        audit=audit,
        ledger=ledger,
        selection=selection,
        records=selected_records,
        snapshot=snapshot,
        rows_by_id=rows_by_id,
        failure_receipt_sha256=failure_sha,
        audit_sha256=audit_sha,
        ledger_sha256=ledger_sha,
        selection_sha256=selection_sha,
    )


def validate_same39_authorization_v2(
    records: Sequence[UnlabeledVideoRecord],
    *,
    expected_source_revision: str,
    expected_gate_sha256: str,
    expected_authorization_receipt_sha256: str,
    authorization_receipt_path: Path,
    expected_failure_receipt_sha256: str,
    failure_receipt_path: Path,
    audit_path: Path,
    ledger_path: Path,
    selection_path: Path,
    cache_dir: Path,
) -> Same39Authorization:
    evidence = validate_same39_failed_pilot(
        records,
        expected_failure_receipt_sha256=expected_failure_receipt_sha256,
        failure_receipt_path=failure_receipt_path,
        audit_path=audit_path,
        ledger_path=ledger_path,
        selection_path=selection_path,
        cache_dir=cache_dir,
    )
    receipt, receipt_sha = load_strict_json(
        authorization_receipt_path,
        role="v2 extraction-only authorization receipt",
    )
    require(
        receipt_sha
        == digest(
            expected_authorization_receipt_sha256,
            "caller-pinned v2 authorization receipt",
        ),
        "v2 authorization receipt SHA mismatch",
    )
    require(set(receipt) == EXTRACTION_AUTHORIZATION_KEYS, "v2 authorization schema mismatch")
    require(
        receipt.get("schema_version") == 1
        and receipt.get("artifact_type")
        == "pams_pose_recovery_v4d_full337_extraction_authorization_v2"
        and receipt.get("authorization_protocol_version") == 2
        and receipt.get("authorization_scope") == "full337_pose_extraction_only"
        and receipt.get("protocol") == "ucfrep_526"
        and receipt.get("split") == "train"
        and receipt.get("label_free") is True
        and receipt.get("source_revision") == expected_source_revision
        and receipt.get("container_image_id") == CONTAINER_IMAGE_ID
        and receipt.get("full337_pose_extraction_authorized") is True
        and receipt.get("baseline_training_authorized") is False
        and receipt.get("training_runner_must_reject") is True
        and receipt.get("training_authorization_requires")
        == "unchanged-full337-label-free-track-quality-periodicity-gate-pass",
        "v2 receipt is not extraction-only authorization",
    )
    basis = mapping(receipt.get("authorization_basis"), "v2 authorization basis")
    require(
        dict(basis)
        == {
            "actual_criteria_names": sorted(
                {
                    "base_v4a_mask_mismatch_total",
                    "candidate_at_most_8_video_total",
                    "candidate_longest_run_fraction_p25",
                    "candidate_zero_video_total",
                    "invariant_failure_total",
                    "keypointrcnn_fill_frames_total",
                    "longest_run_fraction_mean_gain_over_v4a",
                    "recovered_base_zero_video_total",
                    "source_coverage_mean_gain_over_v4a",
                }
            ),
            "old_full337_pose_extraction_authorized": False,
            "old_pilot_passed": False,
            "projection_consulted_for_decision": False,
            "same39_cost_gate_passed": True,
            "zero11_passed": True,
        },
        "v2 authorization basis mismatch",
    )
    bindings = mapping(receipt.get("bindings"), "v2 authorization bindings")
    require(
        dict(bindings)
        == {
            "container_image_id": CONTAINER_IMAGE_ID,
            "full337_gate_sha256": digest(expected_gate_sha256, "full337 gate SHA"),
            "keypointrcnn_model_asset_sha256": MODEL_ASSET_SHA256,
            "same39_audit_sha256": evidence.audit_sha256,
            "same39_candidate_cache_set_sha256": evidence.snapshot.fingerprint,
            "same39_failure_receipt_sha256": evidence.failure_receipt_sha256,
            "same39_identity_sha256": SAME39_IDENTITY_SHA256,
            "same39_ledger_sha256": evidence.ledger_sha256,
            "same39_selection_sha256": evidence.selection_sha256,
            "same39_source_revision": PARENT_V4D_SOURCE_REVISION,
            "train337_commitment_sha256": TRAIN337_COMMITMENT_SHA256,
            "train337_identity_sha256": TRAIN337_IDENTITY_SHA256,
            "train337_sidecar_sha256": TRAIN337_SIDECAR_SHA256,
            "v4a_ledger_sha256": V4A_LEDGER_SHA256,
            "v4a_paired_gate_sha256": V4A_PAIRED_GATE_SHA256,
            "v4a_pose_cache_set_sha256": V4A_CACHE_SET_SHA256,
            "v4a_pose_fingerprint": V4A_POSE_FINGERPRINT,
            "v4d_config_file_sha256": V4D_CONFIG_FILE_SHA256,
            "v4d_config_fingerprint": V4D_CONFIG_FINGERPRINT,
            "v4d_pose_fingerprint": V4D_POSE_FINGERPRINT,
        },
        "v2 authorization binding mismatch",
    )
    return Same39Authorization(
        authorization_receipt=receipt,
        evidence=evidence,
        authorization_receipt_sha256=receipt_sha,
    )
