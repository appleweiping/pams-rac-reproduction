"""Sealed 84-dev scorer for the PoseRAC-v1 official-checkpoint diagnostic.

The prediction JSON, its receipt, all source/checkpoint/cache/config/runner
provenance, and the complete 84-video label-free identity are validated before
``dev.targets.json`` is opened for the first time.  Test paths are rejected
lexically and no test loader exists in this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pams.baselines.poserac_v1_official_runner import (
    ACTION_NAMES,
    CHECKPOINT_SPEC,
    CLASSIFICATION,
    COMPATIBILITY_IMAGE_DIGEST,
    FROZEN_POSERAC_V1_CONFIG,
    METHOD_ID,
    OFFICIAL_ACTION_CSV_SHA256,
    OFFICIAL_CONFIG_SHA256,
    OFFICIAL_EVAL_PY_SHA256,
    OFFICIAL_LICENSE_SHA256,
    OFFICIAL_MODEL_PY_SHA256,
    OFFICIAL_PRE_TEST_PY_SHA256,
    OFFICIAL_SOURCE_COMMIT,
    OFFICIAL_SOURCE_REPOSITORY,
    OFFICIAL_SOURCE_TREE_BYTES,
    OFFICIAL_SOURCE_TREE_FILES,
    OFFICIAL_SOURCE_TREE_SHA256,
    RUNNER_CODE_RELATIVE_PATH,
    SOURCE_ARCHIVE_SPEC,
)
from pams.baselines.repnet_official_runner import (
    _stable_file_digest,
    durable_mkdir,
    fsync_directory,
    sha256_json,
)
from pams.data import (
    PoseCacheEntryReceipt,
    PoseCacheSetSnapshot,
    PoseInputManifest,
    load_dev_target_manifest,
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
    validate_pose_input_binding,
)

EVALUATION_CLASSIFICATION = (
    "PoseRAC-v1 official checkpoint / PAMS 256-frame pose-cache / "
    "inferred oracle-free dynamic-range channel / local 84-dev diagnostic"
)
CANONICAL_DEV_ID_SHA256 = (
    "2199294a1d22da6c67d2fdaaafb4bebaa11e92a5bb3accd4670975815001f2c6"
)
SCORING_CODE_RELATIVE_PATH = "src/pams/baselines/poserac_v1_official_score.py"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
FORBIDDEN_KEYS = frozenset(
    {
        "action",
        "count",
        "ground_truth",
        "ground_truth_count",
        "gt",
        "gt_count",
        "label",
        "labels",
        "target",
        "targets",
    }
)


class PoseRACV1OfficialScoreError(RuntimeError):
    """Prediction integrity or two-process firewall violation."""


@dataclass(frozen=True, slots=True)
class ValidatedPredictions:
    prediction_sha256: str
    prediction_bytes: int
    receipt_sha256: str
    video_ids: tuple[str, ...]
    raw_counts: tuple[float, ...]
    rounded_counts: tuple[int, ...]
    input_binding: Mapping[str, Any]
    pose_cache_set_sha256: str
    pose_fingerprint: str
    runtime_versions: Mapping[str, str]
    runner_git_sha: str
    runner_code_sha256: str


@dataclass(frozen=True, slots=True)
class ValidatedLabelFreeInputs:
    manifest: PoseInputManifest
    sidecar_path: Path
    commitment_path: Path
    sidecar_digest: Any
    commitment_digest: Any
    input_binding: Mapping[str, Any]

    def assert_unchanged(self) -> None:
        if _stable_file_digest(self.sidecar_path) != self.sidecar_digest:
            raise PoseRACV1OfficialScoreError(
                "label-free sidecar changed during PoseRAC-v1 scoring"
            )
        if _stable_file_digest(self.commitment_path) != self.commitment_digest:
            raise PoseRACV1OfficialScoreError(
                "label-free commitment changed during PoseRAC-v1 scoring"
            )


def _load_label_free_scoring_inputs(
    sidecar_path: Path,
    commitment_path: Path,
) -> ValidatedLabelFreeInputs:
    sidecar_digest = _stable_file_digest(sidecar_path)
    commitment_digest = _stable_file_digest(commitment_path)
    manifest = load_pose_input_manifest(sidecar_path, validate_exact=True)
    commitment = load_pose_input_commitment(commitment_path)
    validate_pose_input_binding(
        manifest,
        commitment,
        sidecar_sha256=sidecar_digest.sha256,
    )
    if (
        manifest.protocol != "ucfrep_526"
        or manifest.split != "dev"
        or len(manifest.records) != 84
    ):
        raise ValueError("PoseRAC-v1 scoring requires the exact 84-video dev sidecar")
    if _stable_file_digest(sidecar_path) != sidecar_digest:
        raise PoseRACV1OfficialScoreError(
            "label-free sidecar changed while it was validated"
        )
    if _stable_file_digest(commitment_path) != commitment_digest:
        raise PoseRACV1OfficialScoreError(
            "label-free commitment changed while it was validated"
        )
    return ValidatedLabelFreeInputs(
        manifest=manifest,
        sidecar_path=sidecar_path,
        commitment_path=commitment_path,
        sidecar_digest=sidecar_digest,
        commitment_digest=commitment_digest,
        input_binding={
            "protocol": manifest.protocol,
            "split": manifest.split,
            "full_sidecar_sample_count": len(manifest.records),
            "sidecar_sha256": sidecar_digest.sha256,
            "commitment_sha256": commitment_digest.sha256,
            "sidecar_fingerprint": manifest.fingerprint,
            "identity_sha256": pose_input_identity_sha256(manifest.records),
        },
    )


def _strict_object(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    supplied = set(value)
    if supplied != expected:
        raise ValueError(
            f"{name} fields mismatch; missing={sorted(expected - supplied)}, "
            f"unknown={sorted(supplied - expected)}"
        )
    return value


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _finite(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ValueError(f"{field} is outside its finite range")
    return number


def _integer(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN_KEYS:
                raise ValueError(
                    f"PoseRAC-v1 prediction contains forbidden label field {key!r}"
                )
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)


def _load_json_strict(path: Path, expected_sha256: str) -> Any:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PoseRACV1OfficialScoreError(
            f"file changed between hashing and parsing: {path}"
        )

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON: {path}") from exc


def _canonical_dev_id_sha256(video_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(video_ids)) + "\n").encode("utf-8")
    ).hexdigest()


def _validate_pose_snapshot(
    raw: Any,
    manifest: PoseInputManifest,
) -> tuple[str, str, dict[str, PoseCacheEntryReceipt]]:
    snapshot = _strict_object(
        raw,
        {
            "schema_version",
            "pose_fingerprint",
            "fingerprint",
            "entry_count",
            "entries",
        },
        "PoseRAC-v1 pose-cache snapshot",
    )
    pose_fingerprint = _sha256(
        snapshot["pose_fingerprint"],
        "input.pose_cache_snapshot.pose_fingerprint",
    )
    entries_raw = snapshot["entries"]
    if not isinstance(entries_raw, list) or len(entries_raw) != 84:
        raise ValueError("PoseRAC-v1 pose snapshot requires exactly 84 entries")
    entries: list[PoseCacheEntryReceipt] = []
    for index, raw_entry in enumerate(entries_raw):
        entry = _strict_object(
            raw_entry,
            {"video_id", "cache_sha256", "bytes"},
            f"PoseRAC-v1 pose snapshot entry {index}",
        )
        video_id = entry["video_id"]
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"pose snapshot entry {index} has invalid video_id")
        entries.append(
            PoseCacheEntryReceipt(
                video_id=video_id,
                cache_sha256=_sha256(
                    entry["cache_sha256"],
                    f"pose snapshot entry {index} cache_sha256",
                ),
                bytes=_integer(
                    entry["bytes"],
                    f"pose snapshot entry {index} bytes",
                    minimum=1,
                ),
            )
        )
    reconstructed = PoseCacheSetSnapshot(
        pose_fingerprint=pose_fingerprint,
        entries=tuple(entries),
    )
    manifest_ids = {record.video_id for record in manifest.records}
    entry_ids = {entry.video_id for entry in reconstructed.entries}
    if entry_ids != manifest_ids:
        raise ValueError("pose snapshot identities differ from the dev sidecar")
    if (
        snapshot["schema_version"] != 1
        or snapshot["entry_count"] != 84
        or snapshot["fingerprint"] != reconstructed.fingerprint
        or snapshot != reconstructed.to_dict()
    ):
        raise ValueError("PoseRAC-v1 pose-cache snapshot fingerprint mismatch")
    return (
        pose_fingerprint,
        reconstructed.fingerprint,
        {entry.video_id: entry for entry in reconstructed.entries},
    )


def _validate_predictions(
    prediction_path: Path,
    receipt_path: Path,
    label_free_inputs: ValidatedLabelFreeInputs,
    *,
    expected_runner_git_sha: str,
    expected_runner_code_sha256: str,
) -> ValidatedPredictions:
    prediction_digest = _stable_file_digest(prediction_path)
    receipt_digest = _stable_file_digest(receipt_path)
    prediction = _load_json_strict(prediction_path, prediction_digest.sha256)
    receipt = _load_json_strict(receipt_path, receipt_digest.sha256)
    _reject_forbidden_keys(prediction)
    _reject_forbidden_keys(receipt)

    root = _strict_object(
        prediction,
        {
            "schema_version",
            "method_id",
            "classification",
            "method_version",
            "distinct_from",
            "eligible_for_original_poserac_v1_cell",
            "eligible_for_original_pams_poserac_cell",
            "ineligibility_reasons",
            "labels_loaded",
            "scoring_performed",
            "ground_truth_count_channel_oracle",
            "runner_provenance",
            "official_source",
            "input",
            "selection",
            "assets",
            "config",
            "runtime_versions",
            "restore_audit",
            "prediction_total",
            "failure_total",
            "predictions",
        },
        "PoseRAC-v1 prediction",
    )
    expected_reasons = [
        "upstream evaluation GT-count channel oracle is disabled",
        "channel selection is independently inferred",
        "PAMS caches are uniformly resampled to 256 rather than original-frame poses",
        "standard UCFRep-526 dev is not the released UCFRep-pose-110 test protocol",
    ]
    if (
        root["schema_version"] != 1
        or root["method_id"] != METHOD_ID
        or root["classification"] != CLASSIFICATION
        or root["method_version"] != "PoseRAC-v1-2023"
        or root["distinct_from"] != "PoseRAC-ICONIP24"
        or root["eligible_for_original_poserac_v1_cell"] is not False
        or root["eligible_for_original_pams_poserac_cell"] is not False
        or root["ineligibility_reasons"] != expected_reasons
        or root["labels_loaded"] is not False
        or root["scoring_performed"] is not False
        or root["ground_truth_count_channel_oracle"] is not False
    ):
        raise ValueError("PoseRAC-v1 prediction method or label-boundary mismatch")

    provenance = _strict_object(
        root["runner_provenance"],
        {"source_git_sha", "runner_code_sha256", "compatibility_image_digest"},
        "PoseRAC-v1 runner provenance",
    )
    if provenance != {
        "source_git_sha": expected_runner_git_sha,
        "runner_code_sha256": expected_runner_code_sha256,
        "compatibility_image_digest": COMPATIBILITY_IMAGE_DIGEST,
    }:
        raise ValueError("PoseRAC-v1 prediction runner provenance mismatch")

    source = _strict_object(
        root["official_source"],
        {
            "repository",
            "commit",
            "archive",
            "tree_files",
            "tree_bytes",
            "tree_sha256",
            "model_py_sha256",
            "eval_py_sha256",
            "pre_test_py_sha256",
            "all_action_csv_sha256",
            "config_sha256",
            "license",
            "license_sha256",
        },
        "PoseRAC-v1 official source",
    )
    if source != {
        "repository": OFFICIAL_SOURCE_REPOSITORY,
        "commit": OFFICIAL_SOURCE_COMMIT,
        "archive": SOURCE_ARCHIVE_SPEC.to_dict(),
        "tree_files": OFFICIAL_SOURCE_TREE_FILES,
        "tree_bytes": OFFICIAL_SOURCE_TREE_BYTES,
        "tree_sha256": OFFICIAL_SOURCE_TREE_SHA256,
        "model_py_sha256": OFFICIAL_MODEL_PY_SHA256,
        "eval_py_sha256": OFFICIAL_EVAL_PY_SHA256,
        "pre_test_py_sha256": OFFICIAL_PRE_TEST_PY_SHA256,
        "all_action_csv_sha256": OFFICIAL_ACTION_CSV_SHA256,
        "config_sha256": OFFICIAL_CONFIG_SHA256,
        "license": "MIT",
        "license_sha256": OFFICIAL_LICENSE_SHA256,
    }:
        raise ValueError("PoseRAC-v1 official source binding mismatch")

    input_root = _strict_object(
        root["input"],
        {
            "protocol",
            "split",
            "full_sidecar_sample_count",
            "sidecar_sha256",
            "commitment_sha256",
            "sidecar_fingerprint",
            "identity_sha256",
            "pose_cache_snapshot",
        },
        "PoseRAC-v1 input binding",
    )
    input_binding = {
        key: value for key, value in input_root.items() if key != "pose_cache_snapshot"
    }
    if input_binding != label_free_inputs.input_binding:
        raise ValueError(
            "PoseRAC-v1 prediction input binding differs from the supplied sidecar"
        )
    pose_fingerprint, pose_cache_set_sha256, cache_entries = _validate_pose_snapshot(
        input_root["pose_cache_snapshot"],
        label_free_inputs.manifest,
    )

    selection = _strict_object(
        root["selection"],
        {
            "selected_count",
            "selected_video_ids",
            "selected_video_ids_sha256",
            "rule",
            "rule_provenance",
        },
        "PoseRAC-v1 selection",
    )
    selected_ids = selection["selected_video_ids"]
    if (
        selection["selected_count"] != 84
        or not isinstance(selected_ids, list)
        or len(selected_ids) != 84
        or any(not isinstance(value, str) or not value for value in selected_ids)
        or len(set(selected_ids)) != 84
    ):
        raise ValueError("PoseRAC-v1 scoring requires 84 unique selected IDs")
    video_ids = tuple(selected_ids)
    manifest_ids = tuple(
        record.video_id for record in label_free_inputs.manifest.records
    )
    if video_ids != manifest_ids:
        raise ValueError("PoseRAC-v1 selection order differs from the dev sidecar")
    if _canonical_dev_id_sha256(video_ids) != CANONICAL_DEV_ID_SHA256:
        raise ValueError("PoseRAC-v1 selected identities are not canonical dev84")
    if (
        selection["selected_video_ids_sha256"] != sha256_json(list(video_ids))
        or selection["rule"] != FROZEN_POSERAC_V1_CONFIG.channel_selection
        or selection["rule_provenance"] != "inferred"
    ):
        raise ValueError("PoseRAC-v1 selection provenance mismatch")

    assets = _strict_object(
        root["assets"],
        {"checkpoint", "third_party_assets_redistributed"},
        "PoseRAC-v1 assets",
    )
    if (
        assets["checkpoint"] != CHECKPOINT_SPEC.to_dict()
        or assets["third_party_assets_redistributed"] is not False
    ):
        raise ValueError("PoseRAC-v1 checkpoint identity mismatch")

    config = _strict_object(
        root["config"],
        {"sha256", "values"},
        "PoseRAC-v1 config",
    )
    if (
        config["values"] != FROZEN_POSERAC_V1_CONFIG.to_dict()
        or config["sha256"] != FROZEN_POSERAC_V1_CONFIG.fingerprint
    ):
        raise ValueError("PoseRAC-v1 configuration is not frozen")

    runtime = root["runtime_versions"]
    runtime_fields = {
        "python",
        "torch",
        "numpy",
        "cuda_runtime",
        "device",
        "gpu_name",
    }
    if (
        not isinstance(runtime, dict)
        or set(runtime) != runtime_fields
        or any(not isinstance(value, str) or not value for value in runtime.values())
    ):
        raise ValueError("PoseRAC-v1 runtime provenance is incomplete")

    restore = _strict_object(
        root["restore_audit"],
        {
            "checkpoint_format",
            "checkpoint_key_count",
            "model_key_count",
            "loaded_key_count",
            "missing_keys",
            "unexpected_keys",
            "strict_key_coverage",
            "model_parameter_count",
            "fc1_weight_shape",
        },
        "PoseRAC-v1 checkpoint restore audit",
    )
    if restore != {
        "checkpoint_format": "plain_ordered_tensor_state_dict",
        "checkpoint_key_count": 74,
        "model_key_count": 74,
        "loaded_key_count": 74,
        "missing_keys": [],
        "unexpected_keys": [],
        "strict_key_coverage": True,
        "model_parameter_count": 2_686_682,
        "fc1_weight_shape": [8, 99],
    }:
        raise ValueError("PoseRAC-v1 checkpoint restore coverage mismatch")
    if root["prediction_total"] != 84 or root["failure_total"] != 0:
        raise ValueError("PoseRAC-v1 dev prediction is incomplete")

    rows = root["predictions"]
    if not isinstance(rows, list) or len(rows) != 84:
        raise ValueError("PoseRAC-v1 dev prediction requires exactly 84 rows")
    row_fields = {
        "video_id",
        "pose_cache_sha256",
        "pose_cache_bytes",
        "raw_count",
        "rounded_count",
        "selected_channel",
        "selected_action",
        "channel_selection_uses_count_label",
        "channel_counts",
        "channel_dynamic_ranges",
        "probability_min",
        "probability_max",
        "frames",
        "valid_frames",
    }
    raw_counts: list[float] = []
    rounded_counts: list[int] = []
    observed_ids: list[str] = []
    for index, raw_row in enumerate(rows):
        row = _strict_object(raw_row, row_fields, f"PoseRAC-v1 row {index}")
        video_id = row["video_id"]
        if not isinstance(video_id, str) or video_id != video_ids[index]:
            raise ValueError(f"PoseRAC-v1 row {index} video identity/order mismatch")
        cache_entry = cache_entries[video_id]
        if (
            _sha256(
                row["pose_cache_sha256"],
                f"rows[{index}].pose_cache_sha256",
            )
            != cache_entry.cache_sha256
            or _integer(
                row["pose_cache_bytes"],
                f"rows[{index}].pose_cache_bytes",
                minimum=1,
            )
            != cache_entry.bytes
        ):
            raise ValueError(f"PoseRAC-v1 row {index} pose-cache receipt mismatch")
        raw_count = _finite(
            row["raw_count"],
            f"rows[{index}].raw_count",
            minimum=0.0,
        )
        rounded_count = _integer(
            row["rounded_count"],
            f"rows[{index}].rounded_count",
        )
        if raw_count != float(rounded_count):
            raise ValueError(f"PoseRAC-v1 row {index} raw/rounded count mismatch")
        selected_channel = _integer(
            row["selected_channel"],
            f"rows[{index}].selected_channel",
        )
        if selected_channel >= len(ACTION_NAMES):
            raise ValueError(f"PoseRAC-v1 row {index} channel is outside [0, 7]")
        if (
            row["selected_action"] != ACTION_NAMES[selected_channel]
            or row["channel_selection_uses_count_label"] is not False
        ):
            raise ValueError(f"PoseRAC-v1 row {index} channel provenance mismatch")
        channel_counts_raw = row["channel_counts"]
        if (
            not isinstance(channel_counts_raw, list)
            or len(channel_counts_raw) != len(ACTION_NAMES)
        ):
            raise ValueError(f"PoseRAC-v1 row {index} channel_counts mismatch")
        channel_counts = tuple(
            _integer(value, f"rows[{index}].channel_counts[{channel}]")
            for channel, value in enumerate(channel_counts_raw)
        )
        if channel_counts[selected_channel] != rounded_count:
            raise ValueError(f"PoseRAC-v1 row {index} selected count mismatch")
        ranges_raw = row["channel_dynamic_ranges"]
        if not isinstance(ranges_raw, list) or len(ranges_raw) != len(ACTION_NAMES):
            raise ValueError(f"PoseRAC-v1 row {index} dynamic ranges mismatch")
        ranges = tuple(
            _finite(
                value,
                f"rows[{index}].channel_dynamic_ranges[{channel}]",
                minimum=0.0,
            )
            for channel, value in enumerate(ranges_raw)
        )
        if selected_channel != max(range(len(ranges)), key=ranges.__getitem__):
            raise ValueError(
                f"PoseRAC-v1 row {index} violates first maximum-range selection"
            )
        probability_min = _finite(
            row["probability_min"],
            f"rows[{index}].probability_min",
            minimum=0.0,
        )
        probability_max = _finite(
            row["probability_max"],
            f"rows[{index}].probability_max",
            minimum=0.0,
        )
        if probability_max > 1.0 or probability_min > probability_max:
            raise ValueError(f"PoseRAC-v1 row {index} probability bounds mismatch")
        if (
            row["frames"] != FROZEN_POSERAC_V1_CONFIG.expected_frames
            or isinstance(row["valid_frames"], bool)
            or not isinstance(row["valid_frames"], int)
            or not 0 <= row["valid_frames"] <= row["frames"]
        ):
            raise ValueError(f"PoseRAC-v1 row {index} frame metadata mismatch")
        observed_ids.append(video_id)
        raw_counts.append(raw_count)
        rounded_counts.append(rounded_count)
    if tuple(observed_ids) != video_ids:
        raise ValueError("PoseRAC-v1 row order differs from selection")

    receipt_root = _strict_object(
        receipt,
        {
            "schema_version",
            "artifact_type",
            "method_id",
            "prediction_file",
            "prediction_sha256",
            "prediction_bytes",
            "prediction_total",
            "source_commit",
            "source_archive_sha256",
            "source_tree_sha256",
            "checkpoint_sha256",
            "input_sidecar_sha256",
            "input_commitment_sha256",
            "input_identity_sha256",
            "pose_cache_set_sha256",
            "config_sha256",
            "compatibility_image_digest",
            "runner_source_git_sha",
            "runner_code_sha256",
            "labels_loaded",
            "scoring_performed",
            "ground_truth_count_channel_oracle",
        },
        "PoseRAC-v1 prediction receipt",
    )
    expected_receipt = {
        "schema_version": 1,
        "artifact_type": "poserac_v1_official_predictions_receipt",
        "method_id": METHOD_ID,
        "prediction_file": prediction_path.name,
        "prediction_sha256": prediction_digest.sha256,
        "prediction_bytes": prediction_digest.byte_count,
        "prediction_total": 84,
        "source_commit": OFFICIAL_SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_ARCHIVE_SPEC.sha256,
        "source_tree_sha256": OFFICIAL_SOURCE_TREE_SHA256,
        "checkpoint_sha256": CHECKPOINT_SPEC.sha256,
        "input_sidecar_sha256": input_binding["sidecar_sha256"],
        "input_commitment_sha256": input_binding["commitment_sha256"],
        "input_identity_sha256": input_binding["identity_sha256"],
        "pose_cache_set_sha256": pose_cache_set_sha256,
        "config_sha256": FROZEN_POSERAC_V1_CONFIG.fingerprint,
        "compatibility_image_digest": COMPATIBILITY_IMAGE_DIGEST,
        "runner_source_git_sha": expected_runner_git_sha,
        "runner_code_sha256": expected_runner_code_sha256,
        "labels_loaded": False,
        "scoring_performed": False,
        "ground_truth_count_channel_oracle": False,
    }
    if receipt_root != expected_receipt:
        raise ValueError("PoseRAC-v1 receipt does not bind the prediction")
    if _stable_file_digest(prediction_path) != prediction_digest:
        raise PoseRACV1OfficialScoreError("prediction changed during validation")
    if _stable_file_digest(receipt_path) != receipt_digest:
        raise PoseRACV1OfficialScoreError("prediction receipt changed during validation")
    return ValidatedPredictions(
        prediction_sha256=prediction_digest.sha256,
        prediction_bytes=prediction_digest.byte_count,
        receipt_sha256=receipt_digest.sha256,
        video_ids=video_ids,
        raw_counts=tuple(raw_counts),
        rounded_counts=tuple(rounded_counts),
        input_binding=dict(input_binding),
        pose_cache_set_sha256=pose_cache_set_sha256,
        pose_fingerprint=pose_fingerprint,
        runtime_versions=dict(runtime),
        runner_git_sha=expected_runner_git_sha,
        runner_code_sha256=expected_runner_code_sha256,
    )


def compute_poserac_v1_metrics(
    raw: Sequence[float],
    rounded: Sequence[int],
    targets: Sequence[int],
) -> dict[str, float]:
    if not raw or not (len(raw) == len(rounded) == len(targets)):
        raise ValueError("metric inputs must have the same nonzero length")
    if any(target <= 0 for target in targets):
        raise ValueError("metric targets must be positive")
    rounded_errors = [
        abs(prediction - target)
        for prediction, target in zip(rounded, targets, strict=True)
    ]
    raw_errors = [
        abs(prediction - target)
        for prediction, target in zip(raw, targets, strict=True)
    ]
    total = len(targets)
    return {
        "nmae_rounded": sum(
            error / target
            for error, target in zip(rounded_errors, targets, strict=True)
        )
        / total,
        "mae_raw_prediction": sum(raw_errors) / total,
        "rmse_raw_prediction": math.sqrt(
            sum(
                (prediction - target) ** 2
                for prediction, target in zip(raw, targets, strict=True)
            )
            / total
        ),
        "mae_rounded": sum(rounded_errors) / total,
        "rmse_rounded": math.sqrt(
            sum(error**2 for error in rounded_errors) / total
        ),
        "obo_rounded": sum(error <= 1 for error in rounded_errors) / total,
        "exact_rounded": sum(error == 0 for error in rounded_errors) / total,
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


def paired_bootstrap(
    raw: Sequence[float],
    rounded: Sequence[int],
    targets: Sequence[int],
    *,
    samples: int = 10_000,
    seed: int = 2026,
) -> dict[str, dict[str, float]]:
    if not raw or not (len(raw) == len(rounded) == len(targets)):
        raise ValueError("bootstrap inputs must have the same nonzero length")
    if samples < 1:
        raise ValueError("bootstrap samples must be positive")
    generator = random.Random(seed)
    values: dict[str, list[float]] = {
        key: [] for key in compute_poserac_v1_metrics(raw, rounded, targets)
    }
    for _ in range(samples):
        indices = [generator.randrange(len(targets)) for _ in targets]
        metrics = compute_poserac_v1_metrics(
            [raw[index] for index in indices],
            [rounded[index] for index in indices],
            [targets[index] for index in indices],
        )
        for key, value in metrics.items():
            values[key].append(value)
    return {
        key: {
            "low": _percentile(metric_values, 0.025),
            "high": _percentile(metric_values, 0.975),
        }
        for key, metric_values in values.items()
    }


def _clean_git_revision(repository_root: Path) -> str:
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise PoseRACV1OfficialScoreError(
            "unable to audit scoring Git checkout"
        ) from exc
    if GIT_SHA_PATTERN.fullmatch(revision) is None or status:
        raise PoseRACV1OfficialScoreError(
            "sealed PoseRAC-v1 scoring requires a clean full-SHA Git checkout"
        )
    return revision


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite PoseRAC-v1 score artifact: {path}"
        ) from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)
    return hashlib.sha256(encoded).hexdigest()


def _remove_artifact_if_exact(path: Path, expected: Any) -> bool:
    try:
        observed = _stable_file_digest(path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return False
    if observed != expected:
        return False
    try:
        path.unlink()
    except OSError:
        return False
    fsync_directory(path.parent)
    return True


def _validate_dev_target_locator(path: str | Path) -> Path:
    """Reject any non-canonical or test-looking target locator before file I/O."""

    raw = Path(path)
    if raw.name != "dev.targets.json":
        raise ValueError("PoseRAC-v1 scorer accepts only a file named dev.targets.json")
    return raw.expanduser().resolve(strict=False)


def score_poserac_v1_dev(
    *,
    predictions_path: str | Path,
    prediction_receipt_path: str | Path,
    sidecar_path: str | Path,
    commitment_path: str | Path,
    dev_targets_path: str | Path,
    output_dir: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Validate every label-free byte/provenance binding, then score dev84."""

    # This lexical gate occurs before stat/read/resolve of the supplied target.
    target_file = _validate_dev_target_locator(dev_targets_path)
    prediction_file = Path(predictions_path).expanduser().resolve(strict=True)
    prediction_receipt = Path(prediction_receipt_path).expanduser().resolve(strict=True)
    sidecar_file = Path(sidecar_path).expanduser().resolve(strict=True)
    commitment_file = Path(commitment_path).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    repository = Path(repository_root).expanduser().resolve(strict=True)
    durable_mkdir(destination)
    evaluation_path = destination / "evaluation.json"
    receipt_path = destination / "evaluation.receipt.json"
    if evaluation_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite PoseRAC-v1 score artifacts")

    label_free_inputs = _load_label_free_scoring_inputs(
        sidecar_file,
        commitment_file,
    )
    scoring_git_sha = _clean_git_revision(repository)
    scoring_code = (repository / SCORING_CODE_RELATIVE_PATH).resolve(strict=True)
    if scoring_code != Path(__file__).resolve(strict=True):
        raise PoseRACV1OfficialScoreError("executed scorer is outside repository_root")
    scoring_code_digest = _stable_file_digest(scoring_code)
    runner_code = (repository / RUNNER_CODE_RELATIVE_PATH).resolve(strict=True)
    expected_runner = Path(__file__).with_name(
        "poserac_v1_official_runner.py"
    ).resolve(strict=True)
    if runner_code != expected_runner:
        raise PoseRACV1OfficialScoreError("bound runner is outside repository_root")
    runner_code_digest = _stable_file_digest(runner_code)
    predictions = _validate_predictions(
        prediction_file,
        prediction_receipt,
        label_free_inputs,
        expected_runner_git_sha=scoring_git_sha,
        expected_runner_code_sha256=runner_code_digest.sha256,
    )

    # First target-file I/O occurs only after all validation above succeeds.
    target_digest = _stable_file_digest(target_file)
    targets = load_dev_target_manifest(target_file)
    target_ids = tuple(record.video_id for record in targets.records)
    if target_ids != predictions.video_ids:
        raise ValueError("PoseRAC-v1 prediction/target IDs or order differ")
    target_counts = tuple(record.count for record in targets.records)
    metrics = compute_poserac_v1_metrics(
        predictions.raw_counts,
        predictions.rounded_counts,
        target_counts,
    )
    intervals = paired_bootstrap(
        predictions.raw_counts,
        predictions.rounded_counts,
        target_counts,
        samples=10_000,
        seed=2026,
    )
    if _stable_file_digest(target_file) != target_digest:
        raise PoseRACV1OfficialScoreError("dev targets changed during scoring")
    if _stable_file_digest(prediction_file).sha256 != predictions.prediction_sha256:
        raise PoseRACV1OfficialScoreError("prediction changed during scoring")
    if _stable_file_digest(prediction_receipt).sha256 != predictions.receipt_sha256:
        raise PoseRACV1OfficialScoreError("prediction receipt changed during scoring")
    if _stable_file_digest(scoring_code) != scoring_code_digest:
        raise PoseRACV1OfficialScoreError("scoring code changed during scoring")
    if _stable_file_digest(runner_code) != runner_code_digest:
        raise PoseRACV1OfficialScoreError("runner code changed during scoring")
    if _clean_git_revision(repository) != scoring_git_sha:
        raise PoseRACV1OfficialScoreError("scoring Git revision changed during scoring")
    label_free_inputs.assert_unchanged()

    evaluation = {
        "schema_version": 1,
        "artifact_type": "poserac_v1_official_dev_evaluation",
        "classification": EVALUATION_CLASSIFICATION,
        "method_version": "PoseRAC-v1-2023",
        "distinct_from": "PoseRAC-ICONIP24",
        "eligible_for_original_poserac_v1_cell": False,
        "eligible_for_original_pams_poserac_cell": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "prediction_total": 84,
        "sealed_test_status": "untouched",
        "ground_truth_count_channel_oracle": False,
        "channel_selection": {
            "rule": FROZEN_POSERAC_V1_CONFIG.channel_selection,
            "provenance": "inferred",
            "validated_before_targets": True,
        },
        "prediction_boundary": {
            "labels_loaded": False,
            "scoring_performed": False,
        },
        "metrics": metrics,
        "bootstrap": {
            "samples": 10_000,
            "seed": 2026,
            "confidence_level": 0.95,
            "interval_method": "percentile",
            "pairing": "paired_prediction_target_rows",
            "confidence_intervals": intervals,
        },
        "provenance": {
            "source_commit": OFFICIAL_SOURCE_COMMIT,
            "source_archive_sha256": SOURCE_ARCHIVE_SPEC.sha256,
            "source_tree_sha256": OFFICIAL_SOURCE_TREE_SHA256,
            "checkpoint_sha256": CHECKPOINT_SPEC.sha256,
            "input_manifest_sha256": predictions.input_binding["sidecar_sha256"],
            "input_commitment_sha256": predictions.input_binding["commitment_sha256"],
            "input_identity_sha256": predictions.input_binding["identity_sha256"],
            "pose_fingerprint": predictions.pose_fingerprint,
            "pose_cache_set_sha256": predictions.pose_cache_set_sha256,
            "config_sha256": FROZEN_POSERAC_V1_CONFIG.fingerprint,
            "compatibility_image_digest": COMPATIBILITY_IMAGE_DIGEST,
            "prediction_sha256": predictions.prediction_sha256,
            "prediction_receipt_sha256": predictions.receipt_sha256,
            "dev_targets_sha256": target_digest.sha256,
            "scoring_code_sha256": scoring_code_digest.sha256,
            "scoring_source_git_sha": scoring_git_sha,
            "runner_code_sha256": runner_code_digest.sha256,
            "runner_source_git_sha": scoring_git_sha,
        },
        "runtime_versions": dict(predictions.runtime_versions),
        "per_video": [
            {
                "video_id": video_id,
                "raw_count": raw,
                "rounded_count": rounded,
                "target": target,
                "absolute_error": abs(rounded - target),
                "normalized_absolute_error": abs(rounded - target) / target,
            }
            for video_id, raw, rounded, target in zip(
                predictions.video_ids,
                predictions.raw_counts,
                predictions.rounded_counts,
                target_counts,
                strict=True,
            )
        ],
    }
    evaluation_sha256 = _write_json_exclusive(evaluation_path, evaluation)
    receipt = {
        "schema_version": 1,
        "artifact_type": "poserac_v1_official_dev_evaluation_receipt",
        "classification": EVALUATION_CLASSIFICATION,
        "eligible_for_original_poserac_v1_cell": False,
        "eligible_for_original_pams_poserac_cell": False,
        "evaluation_file": evaluation_path.name,
        "evaluation_sha256": evaluation_sha256,
        "evaluation_bytes": evaluation_path.stat().st_size,
        "prediction_sha256": predictions.prediction_sha256,
        "prediction_receipt_sha256": predictions.receipt_sha256,
        "input_manifest_sha256": predictions.input_binding["sidecar_sha256"],
        "input_commitment_sha256": predictions.input_binding["commitment_sha256"],
        "input_identity_sha256": predictions.input_binding["identity_sha256"],
        "pose_fingerprint": predictions.pose_fingerprint,
        "pose_cache_set_sha256": predictions.pose_cache_set_sha256,
        "dev_targets_file": target_file.name,
        "dev_targets_sha256": target_digest.sha256,
        "dev_targets_bytes": target_digest.byte_count,
        "scoring_code_sha256": scoring_code_digest.sha256,
        "scoring_source_git_sha": scoring_git_sha,
        "runner_code_sha256": runner_code_digest.sha256,
        "runner_source_git_sha": scoring_git_sha,
        "compatibility_image_digest": COMPATIBILITY_IMAGE_DIGEST,
        "ground_truth_count_channel_oracle": False,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 2026,
        "sealed_test_status": "untouched",
    }
    created_evaluation = _stable_file_digest(evaluation_path)
    try:
        receipt_sha256 = _write_json_exclusive(receipt_path, receipt)
    except Exception as exc:
        if not _remove_artifact_if_exact(evaluation_path, created_evaluation):
            raise RuntimeError(
                "PoseRAC-v1 receipt write failed and the exact orphan evaluation "
                "could not be removed safely"
            ) from exc
        raise
    return {
        "classification": EVALUATION_CLASSIFICATION,
        "eligible_for_original_poserac_v1_cell": False,
        "eligible_for_original_pams_poserac_cell": False,
        "metrics": metrics,
        "confidence_intervals": intervals,
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": evaluation_sha256,
        "evaluation_receipt_path": str(receipt_path),
        "evaluation_receipt_sha256": receipt_sha256,
        "sealed_test_status": "untouched",
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pams.baselines.poserac_v1_official_score",
        description=(
            "Score a fully validated PoseRAC-v1 label-free prediction artifact "
            "against canonical dev.targets.json."
        ),
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-receipt", type=Path, required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--dev-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(None if argv is None else list(argv))
    try:
        result = score_poserac_v1_dev(
            predictions_path=arguments.predictions,
            prediction_receipt_path=arguments.prediction_receipt,
            sidecar_path=arguments.sidecar,
            commitment_path=arguments.commitment,
            dev_targets_path=arguments.dev_targets,
            output_dir=arguments.output_dir,
            repository_root=arguments.repository_root,
        )
    except (OSError, PoseRACV1OfficialScoreError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "EVALUATION_CLASSIFICATION",
    "ValidatedPredictions",
    "compute_poserac_v1_metrics",
    "main",
    "paired_bootstrap",
    "score_poserac_v1_dev",
]


if __name__ == "__main__":
    raise SystemExit(main())
