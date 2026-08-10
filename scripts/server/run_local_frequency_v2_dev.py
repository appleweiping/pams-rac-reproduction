"""Strictly isolated UCFRep dev84 runner for local-frequency v2.

The ``predict`` boundary accepts only the frozen target-free selector
artifact/source, the canonical count-free dev84 pose sidecar/cache, and two
new output paths.  The ``score`` boundary accepts only frozen
predictions/receipt, the canonical dev target manifest, and two new output
paths.  Test105 is permanently unauthorized.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from pams.data import (
    PoseInputManifest,
    UnlabeledVideoRecord,
    load_dev_target_manifest,
    load_pose_cache_set,
    load_pose_input_manifest,
    pose_input_identity_sha256,
)
from pams.metrics import compute_count_metrics
from pams.types import PoseSequence

_METHOD_ID = "pams-local-frequency-v2-target-free-selected-v1"
_CLASSIFICATION = "exploratory-derived target-free-selected dev diagnostic; paper-table ineligible"
_PREDICTION_ARTIFACT_TYPE = "pams_local_frequency_v2_dev_predictions"
_PREDICTION_RECEIPT_TYPE = "pams_local_frequency_v2_dev_prediction_receipt"
_EVALUATION_ARTIFACT_TYPE = "pams_local_frequency_v2_dev_evaluation"
_EVALUATION_RECEIPT_TYPE = "pams_local_frequency_v2_dev_evaluation_receipt"
_SELECTOR_ARTIFACT_TYPE = "pams_local_frequency_v2_target_free_selector"
_SELECTOR_CLASSIFICATION = "exploratory-derived target-free selector; paper-table ineligible"
_SELECTOR_SOURCE_RELATIVE_PATH = "scripts/server/run_local_frequency_v2_target_free_selector.py"
_RUNNER_SOURCE_RELATIVE_PATH = "scripts/server/run_local_frequency_v2_dev.py"
_APPROVED_SELECTOR_SOURCE_GIT_SHA = "8e3f33bb9e68ccaf19fe46773888c013d2de3951"
_APPROVED_SELECTOR_SOURCE_SHA256 = (
    "52d9cc2dc026773e7fb45fcc8328edc28779823215c5cd2a1649e263ce0dd8e1"
)
_FROZEN_SELECTOR_ARTIFACT_SHA256 = (
    "0bcad7d0fe270f1e71db465bd2e0f4ac861dfc1167b60283603d4624786e18db"
)
_FROZEN_SELECTOR_ARTIFACT_BYTES = 1_050_892
_FROZEN_SELECTED_CANDIDATE = "xyz.trim0.norm.c1.mean.scale_min"
_FROZEN_SELECTED_RANK_KEY = (
    0,
    0.0,
    0.0,
    0.030882947198275862,
    0.14661458333333333,
    1.4201388888888888,
    -11,
    _FROZEN_SELECTED_CANDIDATE,
)
_FROZEN_SELECTED_PARAMETERS = {
    "count_bounds": [2, 40],
    "feature": "xyz",
    "minimum_cycles_per_window": 1.0,
    "normalized_dimensions": True,
    "scale_reducer": "scale_min",
    "statistic": "mean",
    "trim_fraction_each_tail": 0.0,
    "window_frames": [64, 96, 128, 192, 256],
}
_FROZEN_SELECTOR_SOURCE_ENVIRONMENT = {
    "python_implementation": "cpython",
    "python_version": "3.11.10",
    "numpy_version": "1.26.4",
    "scipy_version": "1.14.1",
}
_FROZEN_POSE_FINGERPRINT = "3dd0388320095796f42aa82d904073b6478b073ed351394ef8d03c30798a0116"
_FROZEN_TRAIN337_SIDECAR_SHA256 = "e238d307b4ed37b2f6c19eefdf7fa8812ec3d231cd2a7091be38e1c279f04207"
_FROZEN_TRAIN337_MANIFEST_FINGERPRINT = (
    "e3d6c979f5d160e37199726bff773a803cc77a0357079d046462b58da6454662"
)
_FROZEN_TRAIN337_IDENTITY_SHA256 = (
    "88367535e296213377c366ed69abbd1e808c38049d79bb948091688e4c5b7b3f"
)
_FROZEN_DEV_INPUTS_SHA256 = "f21c49569805e4aea326977f6d7ce2964f503fad129a17e95214594364ca4ae7"
_FROZEN_DEV_IDENTITY_SHA256 = "bdf944d2aa13e22c07383163794e8edc3198f0c8fd8355b6a6d0255566b3cb5b"
_FROZEN_DEV_TARGETS_SHA256 = "1ab3bc010ccb5d869af7662586594c973e53841838643e1e9b8cc1787b0efab6"
_DEV_RECORD_TOTAL = 84
_CANDIDATE_TOTAL = 512
_COUNT_MINIMUM = 2
_COUNT_MAXIMUM = 40
_DEV_PREDICTION_WORKER_TOTAL = 1
_BOOTSTRAP_SAMPLES = 10_000
_BOOTSTRAP_SEED = 2026
_REQUIRED_NMAE_AT_MOST = 0.228
_REQUIRED_OBO_AT_LEAST = 0.666
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_TRIMS = (0.0, 0.05, 0.075, 0.1)
_FEATURE_NAMES = ("xyz", "centered")
_NORMALIZED_DIMENSIONS = (False, True)
_MINIMUM_CYCLES = (1.0, 1.5)
_WINDOWS = (64, 96, 128, 192, 256)
_STATISTICS = ("mean", "median")
_SCALE_REDUCERS = ("scale_median", "scale_min", "scale_max")
_TRANSFORM_NAMES = (
    "reverse",
    "rotation_negative_15",
    "rotation_positive_15",
    "scale_1_15",
    "translation_positive_0_1",
    "occlusion_20pct_10joints",
    "noise_std_0_02",
    "speed_warp_0_5_to_2_0",
)
_LEXICOGRAPHIC_OBJECTIVE = (
    "degenerate_candidate_flag",
    "synthetic_count_sweep_nmae",
    "synthetic_stress_nmae",
    "train_transform_relative_disagreement",
    "train_duplicate_time_relative_scale_error",
    "prediction_boundary_and_mode_penalty",
    "negative_distinct_training_count_total",
    "candidate_key",
)
_SELECTOR_ROOT_FIELDS = {
    "schema_version",
    "artifact_type",
    "classification",
    "eligible_for_paper_table",
    "selection_status",
    "candidate_total",
    "selected_candidate",
    "selected_parameters",
    "selected_rank_key",
    "objective",
    "frozen_protocol",
    "label_firewall",
    "source",
    "inputs",
    "selected_predictions",
    "candidate_audits",
}
_PREDICTION_ROOT_FIELDS = {
    "schema_version",
    "artifact_type",
    "method_id",
    "classification",
    "eligible_for_paper_table",
    "diagnostic_only",
    "protocol",
    "split",
    "record_total",
    "selected_candidate",
    "selected_parameters",
    "selected_parameters_sha256",
    "selector",
    "dev_input",
    "dev_pose_cache_set",
    "dev_pose_cache_set_sha256",
    "source",
    "records",
    "label_firewall",
    "mount_audit",
    "test105_evaluation_authorized",
}
_PREDICTION_RECEIPT_FIELDS = {
    "schema_version",
    "artifact_type",
    "method_id",
    "protocol",
    "split",
    "record_total",
    "prediction_file",
    "prediction_bytes",
    "prediction_sha256",
    "runner_source_sha256",
    "selector_artifact_sha256",
    "selector_source_sha256",
    "selected_candidate",
    "selected_parameters_sha256",
    "dev_inputs_sha256",
    "dev_identity_sha256",
    "dev_pose_cache_set_sha256",
    "label_firewall",
    "test105_evaluation_authorized",
}


@dataclass(frozen=True, slots=True)
class SelectorBinding:
    """Validated target-free selector inputs used by prediction."""

    artifact_sha256: str
    artifact_bytes: int
    source_sha256: str
    selected_candidate: str
    selected_parameters: Mapping[str, Any]
    selected_parameters_sha256: str
    module: ModuleType


@dataclass(frozen=True, slots=True)
class FrozenPredictions:
    """Validated score-side prediction pair."""

    payload: Mapping[str, Any]
    prediction_sha256: str
    prediction_bytes: int
    receipt_sha256: str
    runner_source_sha256: str


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _require_int(
    value: Any,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return value


def _require_finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _assert_exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    field: str,
) -> None:
    supplied = set(payload)
    if supplied != expected:
        raise ValueError(
            f"{field} fields mismatch; "
            f"missing={sorted(expected - supplied)}, "
            f"unknown={sorted(supplied - expected)}"
        )


def _assert_finite_json(value: Any, field: str = "$") -> None:
    if value is None or isinstance(value, str | bool | int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{field} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite_json(item, f"{field}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} contains a non-string object key")
            _assert_finite_json(item, f"{field}.{key}")
        return
    raise ValueError(f"{field} contains unsupported JSON value {type(value).__name__}")


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _read_stable_bytes(path: Path) -> bytes:
    source = path.resolve()
    before = source.stat()
    with source.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        payload = handle.read()
        closed = os.fstat(handle.fileno())
    after = source.stat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, name) != getattr(opened, name)
        or getattr(opened, name) != getattr(closed, name)
        or getattr(closed, name) != getattr(after, name)
        for name in stable_fields
    ):
        raise RuntimeError(f"input changed while being read: {source}")
    if len(payload) != after.st_size:
        raise RuntimeError(f"input byte count changed while being read: {source}")
    return payload


def _sha256_stable_file(path: Path) -> tuple[str, int]:
    payload = _read_stable_bytes(path)
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _strict_json_bytes(payload: bytes, field: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{field} contains duplicate field {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{field} contains non-finite JSON constant {value}")

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not valid finite UTF-8 JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{field} root must be an object")
    _assert_finite_json(parsed, field)
    return parsed


def _load_strict_json(path: Path, field: str) -> tuple[Mapping[str, Any], str, int]:
    raw = _read_stable_bytes(path)
    return _strict_json_bytes(raw, field), hashlib.sha256(raw).hexdigest(), len(raw)


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> tuple[str, int]:
    _assert_finite_json(payload)
    encoded = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    target = path.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as handle:
        handle.write(encoded)
    return hashlib.sha256(encoded).hexdigest(), len(encoded)


def _validate_new_output_pair(first: Path, second: Path, field: str) -> None:
    outputs = (first.resolve(), second.resolve())
    if outputs[0] == outputs[1]:
        raise ValueError(f"{field} output paths must be distinct")
    collisions = [str(path) for path in outputs if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite {field} output: {collisions}")


def _candidate_methods() -> tuple[str, ...]:
    methods: list[str] = []
    for trim in _TRIMS:
        for feature_name in _FEATURE_NAMES:
            for normalized in _NORMALIZED_DIMENSIONS:
                normalization = "norm" if normalized else "sum"
                for minimum_cycles in _MINIMUM_CYCLES:
                    for window in _WINDOWS:
                        for statistic in _STATISTICS:
                            methods.append(
                                f"{feature_name}.trim{trim:g}.w{window}."
                                f"{normalization}.c{minimum_cycles:g}.{statistic}"
                            )
                    for statistic in _STATISTICS:
                        for reducer in _SCALE_REDUCERS:
                            methods.append(
                                f"{feature_name}.trim{trim:g}.{normalization}."
                                f"c{minimum_cycles:g}.{statistic}.{reducer}"
                            )
    result = tuple(methods)
    if len(result) != _CANDIDATE_TOTAL or len(set(result)) != _CANDIDATE_TOTAL:
        raise AssertionError("local-frequency v2 grid must contain 512 candidates")
    return result


def _candidate_parameters(candidate_key: str) -> dict[str, Any]:
    match = re.fullmatch(
        r"(xyz|centered)\.trim(0|0\.05|0\.075|0\.1)"
        r"(?:\.w(64|96|128|192|256))?\.(sum|norm)"
        r"\.c(1|1\.5)\.(mean|median)"
        r"(?:\.(scale_median|scale_min|scale_max))?",
        candidate_key,
    )
    if match is None:
        raise ValueError("selected candidate is outside the frozen 512-grid")
    (
        feature_name,
        trim,
        window,
        normalization,
        minimum_cycles,
        statistic,
        scale_reducer,
    ) = match.groups()
    if (window is None) == (scale_reducer is None):
        raise ValueError("candidate must select exactly one window or scale reducer")
    result: dict[str, Any] = {
        "feature": feature_name,
        "trim_fraction_each_tail": float(trim),
        "normalized_dimensions": normalization == "norm",
        "minimum_cycles_per_window": float(minimum_cycles),
        "statistic": statistic,
        "count_bounds": [_COUNT_MINIMUM, _COUNT_MAXIMUM],
    }
    if window is None:
        result["window_frames"] = list(_WINDOWS)
        result["scale_reducer"] = scale_reducer
    else:
        result["window_frames"] = int(window)
        result["scale_reducer"] = None
    return result


def _load_approved_selector_module(selector_source: Path) -> tuple[ModuleType, str]:
    source_sha256, _ = _sha256_stable_file(selector_source)
    if source_sha256 != _APPROVED_SELECTOR_SOURCE_SHA256:
        raise ValueError("selector source SHA-256 is not the approved frozen source")
    module_name = "scripts.server.run_local_frequency_v2_target_free_selector"
    specification = importlib.util.spec_from_file_location(
        module_name,
        selector_source.resolve(),
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("could not construct the approved selector module")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if _sha256_stable_file(selector_source)[0] != source_sha256:
        raise RuntimeError("selector source changed while being imported")
    expected_constants = {
        "_ARTIFACT_TYPE": _SELECTOR_ARTIFACT_TYPE,
        "_CLASSIFICATION": _SELECTOR_CLASSIFICATION,
        "_SOURCE_RELATIVE_PATH": _SELECTOR_SOURCE_RELATIVE_PATH,
        "_FROZEN_POSE_FINGERPRINT": _FROZEN_POSE_FINGERPRINT,
        "_FROZEN_TRAIN337_SIDECAR_SHA256": _FROZEN_TRAIN337_SIDECAR_SHA256,
        "_FROZEN_TRAIN337_MANIFEST_FINGERPRINT": (_FROZEN_TRAIN337_MANIFEST_FINGERPRINT),
        "_FROZEN_TRAIN337_IDENTITY_SHA256": _FROZEN_TRAIN337_IDENTITY_SHA256,
    }
    for name, expected in expected_constants.items():
        if getattr(module, name, None) != expected:
            raise ValueError(f"approved selector source constant mismatch: {name}")
    if tuple(module._candidate_methods()) != _candidate_methods():
        raise ValueError("approved selector source grid differs from dev runner")
    return module, source_sha256


def _validate_pose_cache_snapshot(
    snapshot: Any,
    *,
    expected_pose_fingerprint: str,
    expected_video_ids: Sequence[str],
) -> str:
    if not isinstance(snapshot, Mapping):
        raise ValueError("pose-cache snapshot must be an object")
    _assert_exact_fields(
        snapshot,
        {
            "schema_version",
            "pose_fingerprint",
            "fingerprint",
            "entry_count",
            "entries",
        },
        "pose-cache snapshot",
    )
    if snapshot["schema_version"] != 1:
        raise ValueError("pose-cache snapshot schema_version must be 1")
    if snapshot["pose_fingerprint"] != expected_pose_fingerprint:
        raise ValueError("pose-cache snapshot fingerprint family mismatch")
    entries = snapshot["entries"]
    if not isinstance(entries, list):
        raise ValueError("pose-cache snapshot entries must be a list")
    expected_ids = tuple(expected_video_ids)
    if _require_int(snapshot["entry_count"], "pose-cache entry_count") != len(entries):
        raise ValueError("pose-cache entry_count does not match entries")
    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError("pose-cache entries must be objects")
        _assert_exact_fields(
            entry,
            {"video_id", "cache_sha256", "bytes"},
            f"pose-cache entries[{index}]",
        )
        video_id = entry["video_id"]
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("pose-cache entry video_id must be non-empty")
        normalized.append(
            {
                "video_id": video_id,
                "cache_sha256": _require_sha256(
                    entry["cache_sha256"],
                    "pose-cache entry cache_sha256",
                ),
                "bytes": _require_int(
                    entry["bytes"],
                    "pose-cache entry bytes",
                    minimum=1,
                ),
            }
        )
    identifiers = tuple(entry["video_id"] for entry in normalized)
    if identifiers != tuple(sorted(identifiers)) or len(set(identifiers)) != len(identifiers):
        raise ValueError("pose-cache snapshot entries must have unique sorted IDs")
    if set(identifiers) != set(expected_ids) or len(identifiers) != len(expected_ids):
        raise ValueError("pose-cache snapshot identities differ from expected inputs")
    computed = _sha256_json(
        {
            "schema_version": 1,
            "pose_fingerprint": expected_pose_fingerprint,
            "entries": normalized,
        }
    )
    if snapshot["fingerprint"] != computed:
        raise ValueError("pose-cache snapshot canonical fingerprint mismatch")
    return computed


def _validate_selector_candidate_audits(
    audits: Any,
    *,
    selected_candidate: str,
    selected_parameters: Mapping[str, Any],
    selected_rank_key: Any,
) -> None:
    if not isinstance(audits, list) or len(audits) != _CANDIDATE_TOTAL:
        raise ValueError("selector must contain exactly 512 candidate audits")
    methods = set(_candidate_methods())
    observed: set[str] = set()
    rank_keys: list[tuple[Any, ...]] = []
    for offset, audit in enumerate(audits, start=1):
        if not isinstance(audit, Mapping):
            raise ValueError("selector candidate audits must be objects")
        _assert_exact_fields(
            audit,
            {
                "rank",
                "candidate_key",
                "parameters",
                "rank_key",
                "degenerate_candidate_flag",
                "synthetic_count_sweep",
                "synthetic_stress",
                "unlabeled_train_consistency",
            },
            f"candidate_audits[{offset - 1}]",
        )
        if audit["rank"] != offset:
            raise ValueError("selector candidate ranks must be contiguous and ordered")
        candidate_key = audit["candidate_key"]
        if not isinstance(candidate_key, str) or candidate_key not in methods:
            raise ValueError("selector candidate audit contains an unknown candidate")
        if candidate_key in observed:
            raise ValueError("selector candidate audits contain a duplicate candidate")
        observed.add(candidate_key)
        if audit["parameters"] != _candidate_parameters(candidate_key):
            raise ValueError("selector candidate parameters do not match its key")
        if not isinstance(audit["degenerate_candidate_flag"], bool):
            raise ValueError("selector degenerate_candidate_flag must be boolean")
        synthetic_sweep = audit["synthetic_count_sweep"]
        synthetic_stress = audit["synthetic_stress"]
        consistency = audit["unlabeled_train_consistency"]
        if not all(
            isinstance(value, Mapping) for value in (synthetic_sweep, synthetic_stress, consistency)
        ):
            raise ValueError("selector candidate metric groups must be objects")
        expected_rank = (
            int(bool(audit["degenerate_candidate_flag"])),
            _require_finite_number(
                synthetic_sweep.get("nmae"),
                "synthetic count-sweep NMAE",
            ),
            _require_finite_number(
                synthetic_stress.get("nmae"),
                "synthetic stress NMAE",
            ),
            _require_finite_number(
                consistency.get("transform_relative_disagreement"),
                "train transform disagreement",
            ),
            _require_finite_number(
                consistency.get("duplicate_time_relative_scale_error"),
                "train duplicate-time error",
            ),
            _require_finite_number(
                consistency.get("prediction_boundary_and_mode_penalty"),
                "training degeneracy penalty",
            ),
            -_require_int(
                consistency.get("distinct_prediction_total"),
                "distinct training predictions",
                minimum=1,
            ),
            candidate_key,
        )
        rank_key = audit["rank_key"]
        if not isinstance(rank_key, list) or tuple(rank_key) != expected_rank:
            raise ValueError("selector candidate rank_key cannot be reconstructed")
        rank_keys.append(expected_rank)
    if observed != methods:
        raise ValueError("selector candidate audits do not cover the frozen grid")
    if rank_keys != sorted(rank_keys):
        raise ValueError("selector candidate audits are not lexicographically ranked")
    first = audits[0]
    if (
        first["candidate_key"] != selected_candidate
        or first["parameters"] != selected_parameters
        or first["rank_key"] != selected_rank_key
    ):
        raise ValueError("selector selected candidate is not rank one")


def _validate_selector_artifact(
    selector_artifact: Path,
    selector_source: Path,
) -> SelectorBinding:
    artifact_raw = _read_stable_bytes(selector_artifact)
    artifact_sha256 = hashlib.sha256(artifact_raw).hexdigest()
    artifact_bytes = len(artifact_raw)
    if (
        artifact_sha256 != _FROZEN_SELECTOR_ARTIFACT_SHA256
        or artifact_bytes != _FROZEN_SELECTOR_ARTIFACT_BYTES
    ):
        raise ValueError(
            "selector artifact bytes do not match the frozen target-free selection run"
        )
    module, source_sha256 = _load_approved_selector_module(selector_source)
    payload = _strict_json_bytes(artifact_raw, "selector artifact")
    _assert_exact_fields(payload, _SELECTOR_ROOT_FIELDS, "selector artifact")
    if (
        payload["schema_version"] != 1
        or payload["artifact_type"] != _SELECTOR_ARTIFACT_TYPE
        or payload["classification"] != _SELECTOR_CLASSIFICATION
        or payload["eligible_for_paper_table"] is not False
        or payload["selection_status"] != "exploratory-derived_ineligible"
        or payload["candidate_total"] != _CANDIDATE_TOTAL
    ):
        raise ValueError("selector artifact identity/classification mismatch")
    source = payload["source"]
    if not isinstance(source, Mapping):
        raise ValueError("selector source metadata must be an object")
    if (
        source.get("relative_path") != _SELECTOR_SOURCE_RELATIVE_PATH
        or source.get("sha256") != source_sha256
        or any(
            source.get(name) != expected
            for name, expected in _FROZEN_SELECTOR_SOURCE_ENVIRONMENT.items()
        )
    ):
        raise ValueError("selector artifact is not bound to the approved source")
    objective = payload["objective"]
    if not isinstance(objective, Mapping) or objective.get("type") != (
        "strict_lexicographic_minimum"
    ):
        raise ValueError("selector objective is not the frozen strict ordering")
    if objective.get("lexicographic_order") != list(_LEXICOGRAPHIC_OBJECTIVE):
        raise ValueError("selector objective order differs from the frozen order")
    if objective.get("candidate_key_final_tie_break") is not True:
        raise ValueError("selector objective must retain the candidate-key tie break")
    frozen = payload["frozen_protocol"]
    if not isinstance(frozen, Mapping):
        raise ValueError("selector frozen_protocol must be an object")
    expected_grid = {
        "trims": list(_TRIMS),
        "features": list(_FEATURE_NAMES),
        "normalized_dimensions": list(_NORMALIZED_DIMENSIONS),
        "minimum_cycles": list(_MINIMUM_CYCLES),
        "windows": list(_WINDOWS),
        "statistics": list(_STATISTICS),
        "scale_reducers": list(_SCALE_REDUCERS),
    }
    if (
        frozen.get("grid") != expected_grid
        or frozen.get("train_hash_sample_total") != 64
        or frozen.get("transforms") != list(_TRANSFORM_NAMES)
        or frozen.get("synthetic_count_range_inclusive") != [_COUNT_MINIMUM, _COUNT_MAXIMUM]
    ):
        raise ValueError("selector frozen 512-grid/train protocol mismatch")
    label_firewall = payload["label_firewall"]
    if not isinstance(label_firewall, Mapping) or any(
        label_firewall.get(name) is not False
        for name in (
            "development_inputs_loaded",
            "development_targets_loaded",
            "test_inputs_loaded",
            "test_targets_loaded",
            "dataset_count_labels_loaded",
            "dataset_action_labels_loaded",
        )
    ):
        raise ValueError("selector artifact label firewall is not target-free")
    inputs = payload["inputs"]
    if not isinstance(inputs, Mapping):
        raise ValueError("selector inputs must be an object")
    expected_train_binding = {
        "train337_sidecar_sha256": _FROZEN_TRAIN337_SIDECAR_SHA256,
        "frozen_train337_sidecar_sha256": _FROZEN_TRAIN337_SIDECAR_SHA256,
        "train337_sidecar_fingerprint": _FROZEN_TRAIN337_MANIFEST_FINGERPRINT,
        "frozen_train337_sidecar_fingerprint": (_FROZEN_TRAIN337_MANIFEST_FINGERPRINT),
        "train337_identity_sha256": _FROZEN_TRAIN337_IDENTITY_SHA256,
        "frozen_train337_identity_sha256": _FROZEN_TRAIN337_IDENTITY_SHA256,
        "pose_fingerprint": _FROZEN_POSE_FINGERPRINT,
    }
    if any(inputs.get(name) != value for name, value in expected_train_binding.items()):
        raise ValueError("selector artifact canonical train337 binding mismatch")
    selected_train_ids = inputs.get("selected_train_video_ids")
    if (
        not isinstance(selected_train_ids, list)
        or len(selected_train_ids) != 64
        or len(set(selected_train_ids)) != 64
        or any(
            not isinstance(identifier, str) or not identifier for identifier in selected_train_ids
        )
    ):
        raise ValueError("selector artifact must bind 64 unique training identities")
    snapshot_fingerprint = _validate_pose_cache_snapshot(
        inputs.get("selected_pose_cache_set"),
        expected_pose_fingerprint=_FROZEN_POSE_FINGERPRINT,
        expected_video_ids=selected_train_ids,
    )
    if inputs.get("selected_pose_cache_set_sha256") != snapshot_fingerprint:
        raise ValueError("selector training pose-cache set hash mismatch")
    _require_sha256(
        inputs.get("selected_train_identity_sha256"),
        "selector selected_train_identity_sha256",
    )
    transforms = inputs.get("transformed_pose_sha256")
    if not isinstance(transforms, Mapping) or set(transforms) != set(selected_train_ids):
        raise ValueError("selector transformed-pose binding identities mismatch")
    for video_id, transform in transforms.items():
        if not isinstance(transform, Mapping):
            raise ValueError("selector transformed-pose records must be objects")
        _assert_exact_fields(
            transform,
            {
                "original_pose_sha256",
                "transforms",
                "duplicate_time_pose_sha256",
            },
            f"selector transformed pose {video_id}",
        )
        _require_sha256(
            transform["original_pose_sha256"],
            "selector original pose SHA-256",
        )
        _require_sha256(
            transform["duplicate_time_pose_sha256"],
            "selector duplicate pose SHA-256",
        )
        transform_hashes = transform["transforms"]
        if not isinstance(transform_hashes, Mapping) or set(transform_hashes) != set(
            _TRANSFORM_NAMES
        ):
            raise ValueError("selector transformed-pose transform set mismatch")
        for digest in transform_hashes.values():
            _require_sha256(digest, "selector transformed pose SHA-256")
    selected_candidate = payload["selected_candidate"]
    if selected_candidate != _FROZEN_SELECTED_CANDIDATE:
        raise ValueError("selector selected_candidate differs from the frozen result")
    if payload["selected_rank_key"] != list(_FROZEN_SELECTED_RANK_KEY):
        raise ValueError("selector selected_rank_key differs from the frozen result")
    selected_parameters = _candidate_parameters(selected_candidate)
    if (
        selected_parameters != _FROZEN_SELECTED_PARAMETERS
        or payload["selected_parameters"] != _FROZEN_SELECTED_PARAMETERS
    ):
        raise ValueError("selector selected_parameters do not match selected_candidate")
    _validate_selector_candidate_audits(
        payload["candidate_audits"],
        selected_candidate=selected_candidate,
        selected_parameters=selected_parameters,
        selected_rank_key=payload["selected_rank_key"],
    )
    selected_rank_key = payload["selected_rank_key"]
    selected_audit = payload["candidate_audits"][0]
    if (
        not isinstance(selected_rank_key, list)
        or not selected_rank_key
        or isinstance(selected_rank_key[0], bool)
        or _require_finite_number(
            selected_rank_key[0],
            "selector selected_rank_key degeneracy flag",
        )
        != 0.0
        or selected_audit["degenerate_candidate_flag"] is not False
    ):
        raise ValueError("degenerate selected candidate is not authorized for dev84")
    if _sha256_stable_file(selector_source)[0] != source_sha256:
        raise RuntimeError("selector source changed while its artifact was validated")
    if _sha256_stable_file(selector_artifact)[0] != artifact_sha256:
        raise RuntimeError("selector artifact changed while it was validated")
    return SelectorBinding(
        artifact_sha256=artifact_sha256,
        artifact_bytes=artifact_bytes,
        source_sha256=source_sha256,
        selected_candidate=selected_candidate,
        selected_parameters=selected_parameters,
        selected_parameters_sha256=_sha256_json(selected_parameters),
        module=module,
    )


def _validate_canonical_dev_manifest(
    manifest: PoseInputManifest,
    *,
    sidecar_sha256: str,
) -> str:
    if sidecar_sha256 != _FROZEN_DEV_INPUTS_SHA256:
        raise ValueError("predict requires the exact canonical dev.inputs bytes")
    if (
        manifest.protocol != "ucfrep_526"
        or manifest.split != "dev"
        or len(manifest.records) != _DEV_RECORD_TOTAL
    ):
        raise ValueError("predict requires the exact UCFRep dev84 pose sidecar")
    identity_sha256 = pose_input_identity_sha256(manifest.records)
    if identity_sha256 != _FROZEN_DEV_IDENTITY_SHA256:
        raise ValueError("dev84 pose-input identity commitment mismatch")
    return identity_sha256


def _predict_selected_candidate(
    selector_module: ModuleType,
    sequences: Sequence[PoseSequence],
    selected_candidate: str,
) -> tuple[int, ...]:
    items = tuple(sequences)
    if len(items) != _DEV_RECORD_TOTAL:
        raise ValueError("prediction requires exactly 84 pose sequences")
    # The approved selector is loaded from an exact source hash at runtime.
    # Its functions therefore do not have a normally importable module
    # identity for ProcessPool pickling.  Dev84 is small enough to run the
    # identical frozen grid sequentially, which also removes that platform
    # dependency from the formal prediction boundary.
    predicted = selector_module._predict_many(
        items,
        worker_total=_DEV_PREDICTION_WORKER_TOTAL,
    )
    identifiers = tuple(sequence.video_id for sequence in items)
    if tuple(predicted) != identifiers:
        raise RuntimeError("local-frequency prediction changed the dev input order")
    result: list[int] = []
    for video_id in identifiers:
        row = predicted[video_id]
        if set(row) != set(_candidate_methods()):
            raise RuntimeError("local-frequency prediction did not evaluate the 512-grid")
        value = _require_int(
            row[selected_candidate],
            f"prediction for {video_id}",
            minimum=_COUNT_MINIMUM,
            maximum=_COUNT_MAXIMUM,
        )
        result.append(value)
    return tuple(result)


def run_predict(
    *,
    selector_artifact: Path,
    selector_source: Path,
    dev_inputs: Path,
    dev_pose_cache_dir: Path,
    predictions_output: Path,
    prediction_receipt_output: Path,
) -> dict[str, Any]:
    """Create one immutable, target-free dev84 prediction/receipt pair."""

    _validate_new_output_pair(
        predictions_output,
        prediction_receipt_output,
        "prediction",
    )
    runner_source = Path(__file__).resolve()
    runner_source_sha256, _ = _sha256_stable_file(runner_source)
    binding = _validate_selector_artifact(selector_artifact, selector_source)
    dev_inputs_sha256, _ = _sha256_stable_file(dev_inputs)
    if dev_inputs_sha256 != _FROZEN_DEV_INPUTS_SHA256:
        raise ValueError("predict requires the exact canonical dev.inputs bytes")
    manifest = load_pose_input_manifest(dev_inputs, validate_exact=True)
    dev_identity_sha256 = _validate_canonical_dev_manifest(
        manifest,
        sidecar_sha256=dev_inputs_sha256,
    )
    sequences, pose_snapshot = load_pose_cache_set(
        manifest.records,
        cache_dir=dev_pose_cache_dir,
        pose_fingerprint=_FROZEN_POSE_FINGERPRINT,
    )
    if tuple(sequence.video_id for sequence in sequences) != tuple(
        record.video_id for record in manifest.records
    ):
        raise RuntimeError("dev pose-cache order differs from canonical dev.inputs")
    predictions = _predict_selected_candidate(
        binding.module,
        sequences,
        binding.selected_candidate,
    )
    pose_snapshot_dict = pose_snapshot.to_dict()
    pose_snapshot_sha256 = _validate_pose_cache_snapshot(
        pose_snapshot_dict,
        expected_pose_fingerprint=_FROZEN_POSE_FINGERPRINT,
        expected_video_ids=[record.video_id for record in manifest.records],
    )
    cache_sha_by_id = {
        entry["video_id"]: entry["cache_sha256"] for entry in pose_snapshot_dict["entries"]
    }
    records = [
        {
            "video_id": record.video_id,
            "video_sha256": record.video_sha256,
            "pose_cache_sha256": cache_sha_by_id[record.video_id],
            "frames": sequence.num_frames,
            "valid_frames": int(np.sum(sequence.valid_mask)),
            "prediction": prediction,
        }
        for record, sequence, prediction in zip(
            manifest.records,
            sequences,
            predictions,
            strict=True,
        )
    ]
    stable_checks = {
        "runner_source": (
            runner_source,
            runner_source_sha256,
        ),
        "selector_source": (
            selector_source,
            binding.source_sha256,
        ),
        "selector_artifact": (
            selector_artifact,
            binding.artifact_sha256,
        ),
        "dev_inputs": (
            dev_inputs,
            dev_inputs_sha256,
        ),
    }
    for name, (path, expected) in stable_checks.items():
        if _sha256_stable_file(path)[0] != expected:
            raise RuntimeError(f"{name} changed during target-free prediction")
    prediction_payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": _PREDICTION_ARTIFACT_TYPE,
        "method_id": _METHOD_ID,
        "classification": _CLASSIFICATION,
        "eligible_for_paper_table": False,
        "diagnostic_only": True,
        "protocol": "ucfrep_526",
        "split": "dev",
        "record_total": _DEV_RECORD_TOTAL,
        "selected_candidate": binding.selected_candidate,
        "selected_parameters": dict(binding.selected_parameters),
        "selected_parameters_sha256": binding.selected_parameters_sha256,
        "selector": {
            "artifact_sha256": binding.artifact_sha256,
            "artifact_bytes": binding.artifact_bytes,
            "source_sha256": binding.source_sha256,
            "source_git_sha": _APPROVED_SELECTOR_SOURCE_GIT_SHA,
            "candidate_total": _CANDIDATE_TOTAL,
            "canonical_train337_sidecar_sha256": (_FROZEN_TRAIN337_SIDECAR_SHA256),
            "canonical_train337_manifest_fingerprint": (_FROZEN_TRAIN337_MANIFEST_FINGERPRINT),
            "canonical_train337_identity_sha256": (_FROZEN_TRAIN337_IDENTITY_SHA256),
        },
        "dev_input": {
            "sidecar_sha256": dev_inputs_sha256,
            "manifest_fingerprint": manifest.fingerprint,
            "identity_sha256": dev_identity_sha256,
            "record_total": len(manifest.records),
        },
        "dev_pose_cache_set": pose_snapshot_dict,
        "dev_pose_cache_set_sha256": pose_snapshot_sha256,
        "source": {
            "relative_path": _RUNNER_SOURCE_RELATIVE_PATH,
            "sha256": runner_source_sha256,
        },
        "records": records,
        "label_firewall": {
            "prediction_loaded_action_labels": False,
            "prediction_loaded_count_labels": False,
            "prediction_loaded_dev_targets": False,
            "prediction_loaded_test_identity": False,
            "prediction_loaded_test_pose": False,
            "prediction_loaded_test_targets": False,
            "gt_count_or_action_oracle_used": False,
        },
        "mount_audit": {
            "network": "none",
            "selector_artifact_mounted": True,
            "selector_source_mounted": True,
            "dev_inputs_mounted": True,
            "dev_pose_mounted": True,
            "dev_targets_mounted": False,
            "test_identity_mounted": False,
            "test_pose_mounted": False,
            "test_targets_mounted": False,
        },
        "test105_evaluation_authorized": False,
    }
    prediction_sha256, prediction_bytes = _write_json_exclusive(
        predictions_output,
        prediction_payload,
    )
    receipt_payload = {
        "schema_version": 1,
        "artifact_type": _PREDICTION_RECEIPT_TYPE,
        "method_id": _METHOD_ID,
        "protocol": "ucfrep_526",
        "split": "dev",
        "record_total": _DEV_RECORD_TOTAL,
        "prediction_file": predictions_output.name,
        "prediction_bytes": prediction_bytes,
        "prediction_sha256": prediction_sha256,
        "runner_source_sha256": runner_source_sha256,
        "selector_artifact_sha256": binding.artifact_sha256,
        "selector_source_sha256": binding.source_sha256,
        "selected_candidate": binding.selected_candidate,
        "selected_parameters_sha256": binding.selected_parameters_sha256,
        "dev_inputs_sha256": dev_inputs_sha256,
        "dev_identity_sha256": dev_identity_sha256,
        "dev_pose_cache_set_sha256": pose_snapshot_sha256,
        "label_firewall": prediction_payload["label_firewall"],
        "test105_evaluation_authorized": False,
    }
    receipt_sha256, receipt_bytes = _write_json_exclusive(
        prediction_receipt_output,
        receipt_payload,
    )
    return {
        "method_id": _METHOD_ID,
        "prediction_output": str(predictions_output),
        "prediction_sha256": prediction_sha256,
        "prediction_bytes": prediction_bytes,
        "prediction_receipt_output": str(prediction_receipt_output),
        "prediction_receipt_sha256": receipt_sha256,
        "prediction_receipt_bytes": receipt_bytes,
        "record_total": _DEV_RECORD_TOTAL,
        "selected_candidate": binding.selected_candidate,
        "test105_evaluation_authorized": False,
    }


def _prediction_identity_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    synthetic_records = tuple(
        UnlabeledVideoRecord(
            video_id=str(record["video_id"]),
            video_path=f"identity/{index:03d}.mp4",
            video_sha256=str(record["video_sha256"]),
        )
        for index, record in enumerate(records)
    )
    return pose_input_identity_sha256(synthetic_records)


def _validate_frozen_predictions(
    predictions: Path,
    prediction_receipt: Path,
) -> FrozenPredictions:
    runner_source_sha256, _ = _sha256_stable_file(Path(__file__).resolve())
    payload, prediction_sha256, prediction_bytes = _load_strict_json(
        predictions,
        "predictions",
    )
    receipt, receipt_sha256, _ = _load_strict_json(
        prediction_receipt,
        "prediction receipt",
    )
    _assert_exact_fields(payload, _PREDICTION_ROOT_FIELDS, "predictions")
    _assert_exact_fields(
        receipt,
        _PREDICTION_RECEIPT_FIELDS,
        "prediction receipt",
    )
    if (
        payload["schema_version"] != 1
        or payload["artifact_type"] != _PREDICTION_ARTIFACT_TYPE
        or payload["method_id"] != _METHOD_ID
        or payload["classification"] != _CLASSIFICATION
        or payload["eligible_for_paper_table"] is not False
        or payload["diagnostic_only"] is not True
        or payload["protocol"] != "ucfrep_526"
        or payload["split"] != "dev"
        or payload["record_total"] != _DEV_RECORD_TOTAL
        or payload["test105_evaluation_authorized"] is not False
    ):
        raise ValueError("prediction artifact identity/classification mismatch")
    if (
        receipt["schema_version"] != 1
        or receipt["artifact_type"] != _PREDICTION_RECEIPT_TYPE
        or receipt["method_id"] != _METHOD_ID
        or receipt["protocol"] != "ucfrep_526"
        or receipt["split"] != "dev"
        or receipt["record_total"] != _DEV_RECORD_TOTAL
        or receipt["prediction_file"] != predictions.name
        or receipt["prediction_bytes"] != prediction_bytes
        or receipt["prediction_sha256"] != prediction_sha256
        or receipt["test105_evaluation_authorized"] is not False
    ):
        raise ValueError("prediction receipt does not bind the frozen prediction")
    source = payload["source"]
    if (
        not isinstance(source, Mapping)
        or source.get("relative_path") != _RUNNER_SOURCE_RELATIVE_PATH
        or source.get("sha256") != runner_source_sha256
        or receipt["runner_source_sha256"] != runner_source_sha256
    ):
        raise ValueError("predictions were not produced by this frozen runner source")
    selector = payload["selector"]
    if not isinstance(selector, Mapping):
        raise ValueError("prediction selector binding must be an object")
    expected_selector = {
        "artifact_sha256": _FROZEN_SELECTOR_ARTIFACT_SHA256,
        "artifact_bytes": _FROZEN_SELECTOR_ARTIFACT_BYTES,
        "source_sha256": _APPROVED_SELECTOR_SOURCE_SHA256,
        "source_git_sha": _APPROVED_SELECTOR_SOURCE_GIT_SHA,
        "candidate_total": _CANDIDATE_TOTAL,
        "canonical_train337_sidecar_sha256": _FROZEN_TRAIN337_SIDECAR_SHA256,
        "canonical_train337_manifest_fingerprint": (_FROZEN_TRAIN337_MANIFEST_FINGERPRINT),
        "canonical_train337_identity_sha256": _FROZEN_TRAIN337_IDENTITY_SHA256,
    }
    if any(selector.get(name) != expected for name, expected in expected_selector.items()):
        raise ValueError("prediction selector binding differs from the frozen selector")
    selector_artifact_sha256 = _require_sha256(
        selector.get("artifact_sha256"),
        "prediction selector artifact_sha256",
    )
    if (
        receipt["selector_artifact_sha256"] != selector_artifact_sha256
        or receipt["selector_source_sha256"] != _APPROVED_SELECTOR_SOURCE_SHA256
    ):
        raise ValueError("prediction receipt selector binding mismatch")
    selected_candidate = payload["selected_candidate"]
    if selected_candidate != _FROZEN_SELECTED_CANDIDATE:
        raise ValueError("prediction selected_candidate differs from the frozen result")
    selected_parameters = _candidate_parameters(selected_candidate)
    selected_parameters_sha256 = _sha256_json(selected_parameters)
    if (
        payload["selected_parameters"] != selected_parameters
        or payload["selected_parameters_sha256"] != selected_parameters_sha256
        or receipt["selected_candidate"] != selected_candidate
        or receipt["selected_parameters_sha256"] != selected_parameters_sha256
    ):
        raise ValueError("prediction selected-candidate binding mismatch")
    dev_input = payload["dev_input"]
    if not isinstance(dev_input, Mapping):
        raise ValueError("prediction dev_input must be an object")
    _assert_exact_fields(
        dev_input,
        {
            "sidecar_sha256",
            "manifest_fingerprint",
            "identity_sha256",
            "record_total",
        },
        "prediction dev_input",
    )
    _require_sha256(
        dev_input["manifest_fingerprint"],
        "prediction dev manifest_fingerprint",
    )
    if (
        dev_input["sidecar_sha256"] != _FROZEN_DEV_INPUTS_SHA256
        or dev_input["identity_sha256"] != _FROZEN_DEV_IDENTITY_SHA256
        or dev_input["record_total"] != _DEV_RECORD_TOTAL
        or receipt["dev_inputs_sha256"] != _FROZEN_DEV_INPUTS_SHA256
        or receipt["dev_identity_sha256"] != _FROZEN_DEV_IDENTITY_SHA256
    ):
        raise ValueError("prediction dev84 input binding mismatch")
    rows = payload["records"]
    if not isinstance(rows, list) or len(rows) != _DEV_RECORD_TOTAL:
        raise ValueError("predictions must contain exactly 84 records")
    identifiers: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("prediction records must be objects")
        _assert_exact_fields(
            row,
            {
                "video_id",
                "video_sha256",
                "pose_cache_sha256",
                "frames",
                "valid_frames",
                "prediction",
            },
            f"prediction records[{index}]",
        )
        video_id = row["video_id"]
        if not isinstance(video_id, str) or not video_id:
            raise ValueError("prediction record video_id must be non-empty")
        identifiers.append(video_id)
        _require_sha256(row["video_sha256"], "prediction video_sha256")
        _require_sha256(row["pose_cache_sha256"], "prediction pose_cache_sha256")
        frames = _require_int(row["frames"], "prediction frames", minimum=1)
        _require_int(
            row["valid_frames"],
            "prediction valid_frames",
            minimum=0,
            maximum=frames,
        )
        _require_int(
            row["prediction"],
            "prediction",
            minimum=_COUNT_MINIMUM,
            maximum=_COUNT_MAXIMUM,
        )
    if len(set(identifiers)) != _DEV_RECORD_TOTAL:
        raise ValueError("prediction video_id values must be unique")
    if _prediction_identity_sha256(rows) != _FROZEN_DEV_IDENTITY_SHA256:
        raise ValueError("prediction row identity does not match canonical dev84")
    snapshot_sha256 = _validate_pose_cache_snapshot(
        payload["dev_pose_cache_set"],
        expected_pose_fingerprint=_FROZEN_POSE_FINGERPRINT,
        expected_video_ids=identifiers,
    )
    if (
        payload["dev_pose_cache_set_sha256"] != snapshot_sha256
        or receipt["dev_pose_cache_set_sha256"] != snapshot_sha256
    ):
        raise ValueError("prediction pose-cache snapshot binding mismatch")
    cache_sha_by_id = {
        entry["video_id"]: entry["cache_sha256"]
        for entry in payload["dev_pose_cache_set"]["entries"]
    }
    if any(row["pose_cache_sha256"] != cache_sha_by_id[row["video_id"]] for row in rows):
        raise ValueError("prediction records do not bind their pose-cache bytes")
    expected_label_firewall = {
        "prediction_loaded_action_labels": False,
        "prediction_loaded_count_labels": False,
        "prediction_loaded_dev_targets": False,
        "prediction_loaded_test_identity": False,
        "prediction_loaded_test_pose": False,
        "prediction_loaded_test_targets": False,
        "gt_count_or_action_oracle_used": False,
    }
    if (
        payload["label_firewall"] != expected_label_firewall
        or receipt["label_firewall"] != expected_label_firewall
    ):
        raise ValueError("prediction label firewall mismatch")
    expected_mount_audit = {
        "network": "none",
        "selector_artifact_mounted": True,
        "selector_source_mounted": True,
        "dev_inputs_mounted": True,
        "dev_pose_mounted": True,
        "dev_targets_mounted": False,
        "test_identity_mounted": False,
        "test_pose_mounted": False,
        "test_targets_mounted": False,
    }
    if payload["mount_audit"] != expected_mount_audit:
        raise ValueError("prediction mount audit mismatch")
    if _sha256_stable_file(predictions) != (prediction_sha256, prediction_bytes):
        raise RuntimeError("prediction artifact changed while being validated")
    if _sha256_stable_file(prediction_receipt)[0] != receipt_sha256:
        raise RuntimeError("prediction receipt changed while being validated")
    return FrozenPredictions(
        payload=payload,
        prediction_sha256=prediction_sha256,
        prediction_bytes=prediction_bytes,
        receipt_sha256=receipt_sha256,
        runner_source_sha256=runner_source_sha256,
    )


def run_score(
    *,
    predictions: Path,
    prediction_receipt: Path,
    dev_targets: Path,
    evaluation_output: Path,
    evaluation_receipt_output: Path,
) -> dict[str, Any]:
    """Score one immutable prediction pair at the sole label-bearing boundary."""

    _validate_new_output_pair(
        evaluation_output,
        evaluation_receipt_output,
        "evaluation",
    )
    frozen = _validate_frozen_predictions(predictions, prediction_receipt)

    # This is the first permitted target access.  Prediction validation above
    # does not stat, hash, mount, or deserialize a target artifact.
    dev_targets_sha256, _ = _sha256_stable_file(dev_targets)
    if dev_targets_sha256 != _FROZEN_DEV_TARGETS_SHA256:
        raise ValueError("score requires the exact canonical dev.targets bytes")
    targets = load_dev_target_manifest(dev_targets)
    rows = frozen.payload["records"]
    prediction_ids = tuple(str(row["video_id"]) for row in rows)
    target_ids = tuple(record.video_id for record in targets.records)
    if prediction_ids != target_ids:
        raise ValueError("dev target identity/order differs from frozen predictions")
    report = compute_count_metrics(
        [int(row["prediction"]) for row in rows],
        [record.count for record in targets.records],
        video_ids=prediction_ids,
        actions=[record.action for record in targets.records],
        bootstrap_samples=_BOOTSTRAP_SAMPLES,
        bootstrap_seed=_BOOTSTRAP_SEED,
        confidence_level=0.95,
    )
    if _sha256_stable_file(predictions) != (
        frozen.prediction_sha256,
        frozen.prediction_bytes,
    ):
        raise RuntimeError("predictions changed during scoring")
    if _sha256_stable_file(prediction_receipt)[0] != frozen.receipt_sha256:
        raise RuntimeError("prediction receipt changed during scoring")
    if _sha256_stable_file(dev_targets)[0] != dev_targets_sha256:
        raise RuntimeError("dev targets changed during scoring")
    if _sha256_stable_file(Path(__file__).resolve())[0] != (frozen.runner_source_sha256):
        raise RuntimeError("runner source changed during scoring")
    joint_pass = report.nmae <= _REQUIRED_NMAE_AT_MOST and report.obo >= _REQUIRED_OBO_AT_LEAST
    evaluation_payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": _EVALUATION_ARTIFACT_TYPE,
        "method_id": _METHOD_ID,
        "classification": _CLASSIFICATION,
        "eligible_for_paper_table": False,
        "diagnostic_only": True,
        "protocol": "ucfrep_526",
        "split": "dev",
        "prediction_sha256": frozen.prediction_sha256,
        "prediction_receipt_sha256": frozen.receipt_sha256,
        "dev_targets_sha256": dev_targets_sha256,
        "bootstrap_pairing": "paired_prediction_target_rows",
        "metrics": report.to_dict(),
        "acceptance_gate": {
            "required_nmae_at_most": _REQUIRED_NMAE_AT_MOST,
            "required_obo_at_least": _REQUIRED_OBO_AT_LEAST,
            "nmae_pass": report.nmae <= _REQUIRED_NMAE_AT_MOST,
            "obo_pass": report.obo >= _REQUIRED_OBO_AT_LEAST,
            "joint_pass": joint_pass,
            "verified_reproduction": False,
        },
        "source": {
            "relative_path": _RUNNER_SOURCE_RELATIVE_PATH,
            "sha256": frozen.runner_source_sha256,
        },
        "mount_audit": {
            "network": "none",
            "predictions_mounted": True,
            "prediction_receipt_mounted": True,
            "dev_targets_mounted": True,
            "selector_artifact_mounted": False,
            "selector_source_mounted": False,
            "dev_inputs_mounted": False,
            "dev_pose_mounted": False,
            "test_identity_mounted": False,
            "test_pose_mounted": False,
            "test_targets_mounted": False,
        },
        "test105_evaluation_authorized": False,
    }
    evaluation_sha256, evaluation_bytes = _write_json_exclusive(
        evaluation_output,
        evaluation_payload,
    )
    evaluation_receipt = {
        "schema_version": 1,
        "artifact_type": _EVALUATION_RECEIPT_TYPE,
        "method_id": _METHOD_ID,
        "protocol": "ucfrep_526",
        "split": "dev",
        "evaluation_file": evaluation_output.name,
        "evaluation_bytes": evaluation_bytes,
        "evaluation_sha256": evaluation_sha256,
        "prediction_sha256": frozen.prediction_sha256,
        "prediction_receipt_sha256": frozen.receipt_sha256,
        "dev_targets_sha256": dev_targets_sha256,
        "runner_source_sha256": frozen.runner_source_sha256,
        "bootstrap_samples": _BOOTSTRAP_SAMPLES,
        "bootstrap_seed": _BOOTSTRAP_SEED,
        "test105_evaluation_authorized": False,
    }
    evaluation_receipt_sha256, evaluation_receipt_bytes = _write_json_exclusive(
        evaluation_receipt_output,
        evaluation_receipt,
    )
    return {
        "method_id": _METHOD_ID,
        "evaluation_output": str(evaluation_output),
        "evaluation_sha256": evaluation_sha256,
        "evaluation_bytes": evaluation_bytes,
        "evaluation_receipt_output": str(evaluation_receipt_output),
        "evaluation_receipt_sha256": evaluation_receipt_sha256,
        "evaluation_receipt_bytes": evaluation_receipt_bytes,
        "prediction_sha256": frozen.prediction_sha256,
        "dev_targets_sha256": dev_targets_sha256,
        "metrics": {
            "nmae": report.nmae,
            "mae": report.mae,
            "rmse": report.rmse,
            "obo": report.obo,
            "exact": report.exact,
        },
        "acceptance_joint_pass": joint_pass,
        "verified_reproduction": False,
        "test105_evaluation_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--selector-artifact", type=Path, required=True)
    predict.add_argument("--selector-source", type=Path, required=True)
    predict.add_argument("--dev-inputs", type=Path, required=True)
    predict.add_argument("--dev-pose-cache-dir", type=Path, required=True)
    predict.add_argument("--predictions-output", type=Path, required=True)
    predict.add_argument(
        "--prediction-receipt-output",
        type=Path,
        required=True,
    )
    predict.set_defaults(handler=run_predict)

    score = subparsers.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--prediction-receipt", type=Path, required=True)
    score.add_argument("--dev-targets", type=Path, required=True)
    score.add_argument("--evaluation-output", type=Path, required=True)
    score.add_argument(
        "--evaluation-receipt-output",
        type=Path,
        required=True,
    )
    score.set_defaults(handler=run_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    handler = arguments.handler
    kwargs = vars(arguments).copy()
    kwargs.pop("command")
    kwargs.pop("handler")
    payload = handler(**kwargs)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
