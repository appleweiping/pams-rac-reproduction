"""Sealed dev scorer for frozen official RepNet prediction artifacts.

Prediction bytes are hashed and fully validated before this module opens the
only permitted label-bearing input, :class:`pams.data.DevTargetManifest`.
This keeps source inference and label-aware scoring in separate processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pams.baselines.repnet_official_runner import (
    FROZEN_REPNET_OFFICIAL_CONFIG,
    OFFICIAL_CHECKPOINT_PREFIX,
    OFFICIAL_CKPT70_FILES,
    OFFICIAL_NOTEBOOK_PATH,
    OFFICIAL_NOTEBOOK_SHA256,
    OFFICIAL_SOURCE_COMMIT,
    OFFICIAL_SOURCE_REPOSITORY,
)
from pams.data import load_dev_target_manifest
from pams.metrics import compute_count_metrics, round_count

_METHOD_ID = "repnet-official-current-ckpt70"
_CLASSIFICATION = (
    "official-current-ckpt70 / independent label-free UCFRep evaluation"
)
_EVALUATION_CLASSIFICATION = (
    "official-current-ckpt70 / independent sealed UCFRep dev evaluation"
)
_EVALUATION_NAME = "evaluation.json"
_RECEIPT_NAME = "evaluation.receipt.json"
_SCORING_CODE_RELATIVE_PATH = "src/pams/baselines/repnet_official_score.py"
_CANONICAL_DEV_ID_SHA256 = (
    "2199294a1d22da6c67d2fdaaafb4bebaa11e92a5bb3accd4670975815001f2c6"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_FORBIDDEN_PREDICTION_KEYS = frozenset(
    {
        "action",
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


class RepNetOfficialScoreError(RuntimeError):
    """An integrity or sealed-boundary scoring failure."""


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    byte_count: int
    modified_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileIdentity:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            byte_count=value.st_size,
            modified_ns=value.st_mtime_ns,
        )


@dataclass(frozen=True, slots=True)
class _FileDigest:
    sha256: str
    byte_count: int
    identity: _FileIdentity


@dataclass(frozen=True, slots=True)
class _ValidatedPredictions:
    digest: _FileDigest
    video_ids: tuple[str, ...]
    rounded_counts: tuple[int, ...]
    raw_counts: tuple[float, ...]
    checkpoint_sha256: str
    input_binding: Mapping[str, Any]
    config_sha256: str
    source_binding: Mapping[str, Any]


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_mkdir(path: Path) -> None:
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    if cursor.exists() and not cursor.is_dir():
        raise NotADirectoryError(f"directory ancestor is not a directory: {cursor}")
    for directory in reversed(missing):
        directory.mkdir()
        _fsync_directory(directory.parent)


def _stable_file_digest(path: Path) -> _FileDigest:
    try:
        before = path.lstat()
    except OSError as exc:
        raise RepNetOfficialScoreError(f"unable to open required file: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RepNetOfficialScoreError(
            f"required input must be a regular non-symlink file: {path}"
        )
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as handle:
            opened = os.fstat(handle.fileno())
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
            closed = os.fstat(handle.fileno())
        after = path.lstat()
    except OSError as exc:
        raise RepNetOfficialScoreError(f"required file changed while read: {path}") from exc
    identities = tuple(
        _FileIdentity.from_stat(value) for value in (before, opened, closed, after)
    )
    if len(set(identities)) != 1 or byte_count != before.st_size:
        raise RepNetOfficialScoreError(f"required file changed while hashed: {path}")
    return _FileDigest(
        sha256=digest.hexdigest(),
        byte_count=byte_count,
        identity=identities[0],
    )


def _assert_digest_unchanged(
    path: Path,
    expected: _FileDigest,
    *,
    role: str,
) -> None:
    observed = _stable_file_digest(path)
    if observed != expected:
        raise RepNetOfficialScoreError(f"{role} changed during sealed scoring")


def _canonical_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _strict_object(
    value: Any,
    *,
    expected: set[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    supplied = set(value)
    if supplied != expected:
        raise ValueError(
            f"{name} fields mismatch; "
            f"missing={sorted(expected - supplied)}, "
            f"unknown={sorted(supplied - expected)}"
        )
    return value


def _strict_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _strict_float(
    value: Any,
    *,
    field: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < minimum:
        raise ValueError(f"{field} is outside its finite range")
    if maximum is not None and numeric > maximum:
        raise ValueError(f"{field} is outside its finite range")
    return numeric


def _reject_forbidden_keys(encoded: str) -> None:
    """Reject label keys from raw JSON before their values are deserialized."""

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
        if isinstance(key, str) and key.lower() in _FORBIDDEN_PREDICTION_KEYS:
            raise ValueError(
                f"RepNet prediction artifact contains forbidden label field {key!r}"
            )


def _load_prediction_json(path: Path, digest: _FileDigest) -> dict[str, Any]:
    try:
        raw_bytes = path.read_bytes()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("RepNet prediction artifact is not valid UTF-8 JSON") from exc
    if (
        len(raw_bytes) != digest.byte_count
        or hashlib.sha256(raw_bytes).hexdigest() != digest.sha256
    ):
        raise RepNetOfficialScoreError(
            "RepNet prediction artifact changed between hashing and parsing"
        )
    try:
        encoded = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("RepNet prediction artifact is not valid UTF-8 JSON") from exc
    _reject_forbidden_keys(encoded)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate prediction JSON field: {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            encoded,
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite prediction JSON constant: {value}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid RepNet prediction JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("RepNet prediction artifact root must be an object")
    return payload


def _canonical_dev_id_sha256(video_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(video_ids)) + "\n").encode("utf-8")
    ).hexdigest()


def _validate_prediction_payload(
    payload: dict[str, Any],
    digest: _FileDigest,
) -> _ValidatedPredictions:
    root = _strict_object(
        payload,
        expected={
            "schema_version",
            "method_id",
            "classification",
            "eligible_for_original_pams_repnet_cell",
            "source",
            "input",
            "selection",
            "checkpoint",
            "config",
            "runtime_versions",
            "predictions",
        },
        name="RepNet prediction artifact",
    )
    if root["schema_version"] != 1:
        raise ValueError("RepNet prediction schema_version must be 1")
    if root["method_id"] != _METHOD_ID:
        raise ValueError("RepNet prediction method_id is not frozen")
    if root["classification"] != _CLASSIFICATION:
        raise ValueError("RepNet prediction classification is not frozen")
    if root["eligible_for_original_pams_repnet_cell"] is not False:
        raise ValueError("RepNet prediction cannot claim the original PAMS table cell")

    source = _strict_object(
        root["source"],
        expected={"repository", "commit", "notebook_path", "notebook_sha256"},
        name="RepNet prediction source",
    )
    expected_source = {
        "repository": OFFICIAL_SOURCE_REPOSITORY,
        "commit": OFFICIAL_SOURCE_COMMIT,
        "notebook_path": OFFICIAL_NOTEBOOK_PATH,
        "notebook_sha256": OFFICIAL_NOTEBOOK_SHA256,
    }
    if source != expected_source:
        raise ValueError("RepNet prediction source binding is not the frozen official source")

    input_binding = _strict_object(
        root["input"],
        expected={
            "protocol",
            "split",
            "sample_count",
            "sidecar_sha256",
            "commitment_sha256",
            "sidecar_fingerprint",
            "identity_sha256",
        },
        name="RepNet prediction input binding",
    )
    if (
        input_binding["protocol"] != "ucfrep_526"
        or input_binding["split"] != "dev"
        or input_binding["sample_count"] != 84
    ):
        raise ValueError("RepNet prediction input must be the complete 84-video dev split")
    for field in (
        "sidecar_sha256",
        "commitment_sha256",
        "sidecar_fingerprint",
        "identity_sha256",
    ):
        _canonical_sha256(input_binding[field], field=f"input.{field}")

    selection = _strict_object(
        root["selection"],
        expected={
            "selected_count",
            "selected_video_ids",
            "selected_video_ids_sha256",
        },
        name="RepNet prediction selection",
    )
    if selection["selected_count"] != 84:
        raise ValueError("RepNet dev scoring requires selection.selected_count=84")
    raw_ids = selection["selected_video_ids"]
    if not isinstance(raw_ids, list) or len(raw_ids) != 84:
        raise ValueError("RepNet dev scoring requires 84 selected_video_ids")
    if any(not isinstance(value, str) or not value.strip() for value in raw_ids):
        raise ValueError("selected_video_ids must contain non-empty strings")
    video_ids = tuple(raw_ids)
    if len(set(video_ids)) != 84:
        raise ValueError("selected_video_ids must be unique")
    if _canonical_dev_id_sha256(video_ids) != _CANONICAL_DEV_ID_SHA256:
        raise ValueError("selected_video_ids are not the frozen UCFRep dev identities")
    selection_sha256 = _canonical_sha256(
        selection["selected_video_ids_sha256"],
        field="selection.selected_video_ids_sha256",
    )
    if selection_sha256 != _sha256_json(list(video_ids)):
        raise ValueError("selection SHA-256 does not bind selected_video_ids")

    checkpoint = _strict_object(
        root["checkpoint"],
        expected={"prefix", "aggregate_sha256", "files"},
        name="RepNet prediction checkpoint",
    )
    if checkpoint["prefix"] != OFFICIAL_CHECKPOINT_PREFIX:
        raise ValueError("RepNet prediction checkpoint prefix is not ckpt-70")
    raw_files = checkpoint["files"]
    if not isinstance(raw_files, list) or len(raw_files) != len(OFFICIAL_CKPT70_FILES):
        raise ValueError("RepNet checkpoint must contain all three official objects")
    verified_files: list[dict[str, Any]] = []
    for index, (raw_file, expected) in enumerate(
        zip(raw_files, OFFICIAL_CKPT70_FILES, strict=True)
    ):
        item = _strict_object(
            raw_file,
            expected={"name", "bytes", "published_md5", "sha256"},
            name=f"RepNet checkpoint file {index}",
        )
        if (
            item["name"] != expected.name
            or item["bytes"] != expected.byte_count
            or item["published_md5"] != expected.md5
        ):
            raise ValueError(f"RepNet checkpoint file {index} identity mismatch")
        _canonical_sha256(item["sha256"], field=f"checkpoint.files[{index}].sha256")
        verified_files.append(dict(item))
    checkpoint_sha256 = _canonical_sha256(
        checkpoint["aggregate_sha256"],
        field="checkpoint.aggregate_sha256",
    )
    expected_checkpoint_sha256 = _sha256_json(
        {
            "checkpoint_prefix": OFFICIAL_CHECKPOINT_PREFIX,
            "files": verified_files,
        }
    )
    if checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError("checkpoint aggregate SHA-256 does not bind checkpoint files")

    config = _strict_object(
        root["config"],
        expected={"sha256", "values"},
        name="RepNet prediction config",
    )
    config_sha256 = _canonical_sha256(config["sha256"], field="config.sha256")
    if (
        config["values"] != FROZEN_REPNET_OFFICIAL_CONFIG.to_dict()
        or config_sha256 != FROZEN_REPNET_OFFICIAL_CONFIG.fingerprint
    ):
        raise ValueError("RepNet prediction config is not the frozen official config")

    runtime = root["runtime_versions"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "numpy",
        "opencv",
        "scipy",
        "tensorflow",
    }:
        raise ValueError("RepNet runtime_versions fields are incomplete")
    if any(not isinstance(value, str) or not value for value in runtime.values()):
        raise ValueError("RepNet runtime_versions values must be non-empty strings")

    rows = root["predictions"]
    if not isinstance(rows, list) or len(rows) != 84:
        raise ValueError("RepNet dev scoring requires exactly 84 predictions")
    rounded_counts: list[int] = []
    raw_counts: list[float] = []
    prediction_ids: list[str] = []
    prediction_fields = {
        "video_id",
        "video_locator",
        "expected_video_sha256",
        "observed_video_sha256",
        "raw_count",
        "rounded_count",
        "chosen_stride",
        "confidence",
        "decoded_frames",
        "decode_status",
        "failure_reason",
    }
    for index, raw_row in enumerate(rows):
        row = _strict_object(
            raw_row,
            expected=prediction_fields,
            name=f"RepNet prediction row {index}",
        )
        video_id = row["video_id"]
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"prediction row {index} has invalid video_id")
        prediction_ids.append(video_id)
        locator = row["video_locator"]
        if not isinstance(locator, str) or "\\" in locator:
            raise ValueError(f"prediction row {index} has invalid video_locator")
        locator_path = PurePosixPath(locator)
        if locator_path.is_absolute() or any(
            part in {"", ".", ".."} for part in locator_path.parts
        ):
            raise ValueError(f"prediction row {index} has unsafe video_locator")
        expected_video_sha256 = _canonical_sha256(
            row["expected_video_sha256"],
            field=f"predictions[{index}].expected_video_sha256",
        )
        observed_video_sha256 = _canonical_sha256(
            row["observed_video_sha256"],
            field=f"predictions[{index}].observed_video_sha256",
        )
        if observed_video_sha256 != expected_video_sha256:
            raise ValueError(f"prediction row {index} source-video hash mismatch")
        if row["decode_status"] != "ok" or row["failure_reason"] is not None:
            raise ValueError(f"prediction row {index} was not decoded successfully")
        raw_count = _strict_float(
            row["raw_count"],
            field=f"predictions[{index}].raw_count",
            minimum=0.0,
        )
        rounded_count = _strict_int(
            row["rounded_count"],
            field=f"predictions[{index}].rounded_count",
            minimum=0,
        )
        if rounded_count != round_count(raw_count):
            raise ValueError(f"prediction row {index} rounded_count is inconsistent")
        stride = _strict_int(
            row["chosen_stride"],
            field=f"predictions[{index}].chosen_stride",
            minimum=1,
        )
        if stride not in FROZEN_REPNET_OFFICIAL_CONFIG.strides:
            raise ValueError(f"prediction row {index} has an invalid stride")
        _strict_float(
            row["confidence"],
            field=f"predictions[{index}].confidence",
            minimum=0.0,
            maximum=1.0,
        )
        _strict_int(
            row["decoded_frames"],
            field=f"predictions[{index}].decoded_frames",
            minimum=1,
        )
        raw_counts.append(raw_count)
        rounded_counts.append(rounded_count)
    if tuple(prediction_ids) != video_ids:
        raise ValueError("prediction IDs/order do not exactly match the frozen selection")

    return _ValidatedPredictions(
        digest=digest,
        video_ids=video_ids,
        rounded_counts=tuple(rounded_counts),
        raw_counts=tuple(raw_counts),
        checkpoint_sha256=checkpoint_sha256,
        input_binding=dict(input_binding),
        config_sha256=config_sha256,
        source_binding=dict(source),
    )


def _clean_git_revision(repository_root: Path) -> str:
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status_result = subprocess.run(
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
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RepNetOfficialScoreError("unable to audit scoring Git checkout") from exc
    if _GIT_SHA.fullmatch(revision) is None:
        raise RepNetOfficialScoreError("scoring Git revision is not a full SHA")
    if status_result.stdout.strip():
        raise RepNetOfficialScoreError(
            "sealed RepNet scoring requires a clean Git checkout"
        )
    return revision


def _scoring_code_path(repository_root: Path) -> Path:
    expected = (repository_root / _SCORING_CODE_RELATIVE_PATH).resolve(strict=True)
    actual = Path(__file__).resolve(strict=True)
    if expected != actual:
        raise RepNetOfficialScoreError(
            "executed scorer does not match repository_root scoring source"
        )
    return actual


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> _FileDigest:
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
        raise FileExistsError(f"refusing to overwrite scoring artifact: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(path.parent)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    return _stable_file_digest(path)


def score_repnet_dev(
    *,
    predictions_path: str | Path,
    dev_targets_path: str | Path,
    output_dir: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Score a complete, frozen official RepNet dev prediction ledger."""

    predictions_source = Path(predictions_path).expanduser().resolve(strict=True)
    targets_source = Path(dev_targets_path).expanduser().resolve(strict=False)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    repository = Path(repository_root).expanduser().resolve(strict=True)
    _durable_mkdir(destination)
    evaluation_path = destination / _EVALUATION_NAME
    receipt_path = destination / _RECEIPT_NAME
    collisions = [
        str(path) for path in (evaluation_path, receipt_path) if path.exists()
    ]
    if collisions:
        raise FileExistsError(
            f"refusing to overwrite RepNet dev score artifacts: {collisions}"
        )

    # No target path is opened before this entire block succeeds.
    prediction_digest = _stable_file_digest(predictions_source)
    prediction_payload = _load_prediction_json(
        predictions_source,
        prediction_digest,
    )
    predictions = _validate_prediction_payload(
        prediction_payload,
        prediction_digest,
    )
    _assert_digest_unchanged(
        predictions_source,
        prediction_digest,
        role="RepNet prediction artifact",
    )

    scoring_git_sha = _clean_git_revision(repository)
    scoring_code_path = _scoring_code_path(repository)
    scoring_code_digest = _stable_file_digest(scoring_code_path)

    # Labels enter only here, through the strict canonical dev loader.
    targets_digest = _stable_file_digest(targets_source)
    targets = load_dev_target_manifest(targets_source)
    target_ids = tuple(record.video_id for record in targets.records)
    if target_ids != predictions.video_ids:
        raise ValueError(
            "dev target IDs/order do not exactly match frozen RepNet predictions"
        )
    _assert_digest_unchanged(
        targets_source,
        targets_digest,
        role="canonical dev targets",
    )

    report = compute_count_metrics(
        predictions.rounded_counts,
        [record.count for record in targets.records],
        video_ids=predictions.video_ids,
        actions=[record.action for record in targets.records],
        bootstrap_samples=10_000,
        bootstrap_seed=2026,
        confidence_level=0.95,
    )

    _assert_digest_unchanged(
        predictions_source,
        prediction_digest,
        role="RepNet prediction artifact",
    )
    _assert_digest_unchanged(
        targets_source,
        targets_digest,
        role="canonical dev targets",
    )
    _assert_digest_unchanged(
        scoring_code_path,
        scoring_code_digest,
        role="RepNet scoring code",
    )
    if _clean_git_revision(repository) != scoring_git_sha:
        raise RepNetOfficialScoreError("scoring Git revision changed during scoring")

    evaluation_payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "repnet_official_dev_evaluation",
        "classification": _EVALUATION_CLASSIFICATION,
        "eligible_for_original_pams_repnet_cell": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "method_id": _METHOD_ID,
        "bootstrap_pairing": "paired_prediction_target_rows",
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 2026,
        "prediction_sha256": prediction_digest.sha256,
        "dev_targets_sha256": targets_digest.sha256,
        "checkpoint_aggregate_sha256": predictions.checkpoint_sha256,
        "input_binding": dict(predictions.input_binding),
        "config_sha256": predictions.config_sha256,
        "source_binding": dict(predictions.source_binding),
        "scoring_source_git_sha": scoring_git_sha,
        "scoring_code_sha256": scoring_code_digest.sha256,
        "report": report.to_dict(),
        "predictions": [
            {
                "video_id": video_id,
                "raw_count": raw_count,
                "rounded_count": rounded_count,
            }
            for video_id, raw_count, rounded_count in zip(
                predictions.video_ids,
                predictions.raw_counts,
                predictions.rounded_counts,
                strict=True,
            )
        ],
    }
    evaluation_digest = _write_json_exclusive(
        evaluation_path,
        evaluation_payload,
    )
    receipt_payload = {
        "schema_version": 1,
        "artifact_type": "repnet_official_dev_evaluation_receipt",
        "protocol": "ucfrep_526",
        "split": "dev",
        "method_id": _METHOD_ID,
        "evaluation_file": _EVALUATION_NAME,
        "evaluation_sha256": evaluation_digest.sha256,
        "evaluation_bytes": evaluation_digest.byte_count,
        "prediction_file": predictions_source.name,
        "prediction_sha256": prediction_digest.sha256,
        "prediction_bytes": prediction_digest.byte_count,
        "dev_targets_file": targets_source.name,
        "dev_targets_sha256": targets_digest.sha256,
        "dev_targets_bytes": targets_digest.byte_count,
        "checkpoint_aggregate_sha256": predictions.checkpoint_sha256,
        "input_sidecar_sha256": predictions.input_binding["sidecar_sha256"],
        "input_commitment_sha256": predictions.input_binding["commitment_sha256"],
        "input_identity_sha256": predictions.input_binding["identity_sha256"],
        "config_sha256": predictions.config_sha256,
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "official_notebook_sha256": OFFICIAL_NOTEBOOK_SHA256,
        "scoring_code_file": _SCORING_CODE_RELATIVE_PATH,
        "scoring_code_sha256": scoring_code_digest.sha256,
        "scoring_source_git_sha": scoring_git_sha,
        "bootstrap_samples": 10_000,
        "bootstrap_seed": 2026,
    }
    receipt_digest = _write_json_exclusive(receipt_path, receipt_payload)
    compact_report = report.to_dict()
    compact_report.pop("per_video")
    return {
        "classification": _EVALUATION_CLASSIFICATION,
        "eligible_for_original_pams_repnet_cell": False,
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": evaluation_digest.sha256,
        "evaluation_receipt_path": str(receipt_path),
        "evaluation_receipt_sha256": receipt_digest.sha256,
        "prediction_sha256": prediction_digest.sha256,
        "dev_targets_sha256": targets_digest.sha256,
        "scoring_code_sha256": scoring_code_digest.sha256,
        "scoring_source_git_sha": scoring_git_sha,
        "metrics": compact_report,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pams.baselines.repnet_official_score",
        description="Strictly score a frozen official RepNet artifact on UCFRep dev.",
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--dev-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(None if argv is None else list(argv))
    try:
        result = score_repnet_dev(
            predictions_path=arguments.predictions,
            dev_targets_path=arguments.dev_targets,
            output_dir=arguments.output_dir,
            repository_root=arguments.repository_root,
        )
    except (OSError, RepNetOfficialScoreError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


__all__ = [
    "RepNetOfficialScoreError",
    "main",
    "score_repnet_dev",
]


if __name__ == "__main__":
    raise SystemExit(main())
