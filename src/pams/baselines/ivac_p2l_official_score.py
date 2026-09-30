"""Second-process scorer for strict IVAC-P2L prediction artifacts.

Prediction bytes, receipt bytes, and every label-free sidecar identity are
validated before this module first opens the development target file.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from pams.baselines.ivac_p2l_official_runner import (
    BACKBONE_SPEC,
    CHECKPOINT_SPEC,
    CLASSIFICATION,
    COMPATIBILITY_IMAGE_DIGEST,
    FROZEN_IVAC_P2L_OFFICIAL_CONFIG,
    METHOD_ID,
    OFFICIAL_SOURCE_COMMIT,
    OFFICIAL_SOURCE_TREE_SHA256,
    PROTOCOL_ASSUMPTIONS,
    RUNNER_CODE_RELATIVE_PATH,
    SOURCE_ARCHIVE_SPEC,
    _remove_artifact_if_exact,
    _write_json_exclusive,
    official_source_dict,
    sha256_json,
)
from pams.baselines.repnet_official_runner import _stable_file_digest
from pams.data import (
    PoseInputManifest,
    load_dev_target_manifest,
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
    validate_pose_input_binding,
)
from pams.metrics import round_count

EVALUATION_CLASSIFICATION = (
    "IVAC-P2L official RepCount-A checkpoint / modern compatibility / "
    "local 84-dev sanity"
)
CANONICAL_DEV_ID_SHA256 = (
    "2199294a1d22da6c67d2fdaaafb4bebaa11e92a5bb3accd4670975815001f2c6"
)
SCORING_CODE_RELATIVE_PATH = "src/pams/baselines/ivac_p2l_official_score.py"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
FORBIDDEN_KEYS = frozenset(
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


class IVACP2LOfficialScoreError(RuntimeError):
    """Prediction integrity or two-process firewall violation."""


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
            raise IVACP2LOfficialScoreError(
                "label-free sidecar changed during IVAC-P2L scoring"
            )
        if _stable_file_digest(self.commitment_path) != self.commitment_digest:
            raise IVACP2LOfficialScoreError(
                "label-free commitment changed during IVAC-P2L scoring"
            )


@dataclass(frozen=True, slots=True)
class ValidatedPredictions:
    prediction_sha256: str
    prediction_bytes: int
    receipt_sha256: str
    video_ids: tuple[str, ...]
    raw_counts: tuple[float, ...]
    rounded_counts: tuple[int, ...]
    input_binding: Mapping[str, Any]
    runtime_versions: Mapping[str, str]
    runner_git_sha: str
    runner_code_sha256: str


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
        raise ValueError("IVAC-P2L scoring requires the exact 84-video dev sidecar")
    if _stable_file_digest(sidecar_path) != sidecar_digest:
        raise IVACP2LOfficialScoreError(
            "label-free sidecar changed while it was validated"
        )
    if _stable_file_digest(commitment_path) != commitment_digest:
        raise IVACP2LOfficialScoreError(
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
            "sample_count": len(manifest.records),
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


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in FORBIDDEN_KEYS:
                raise ValueError(
                    f"IVAC-P2L prediction contains forbidden label field {key!r}"
                )
            _reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden_keys(child)


def _load_json_strict(path: Path, expected_sha256: str) -> Any:
    raw = path.read_bytes()
    if __import__("hashlib").sha256(raw).hexdigest() != expected_sha256:
        raise IVACP2LOfficialScoreError(
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
    return __import__("hashlib").sha256(
        ("\n".join(sorted(video_ids)) + "\n").encode("utf-8")
    ).hexdigest()


def _validate_prediction(
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
            "eligible_for_original_pams_ivac_p2l_cell",
            "labels_loaded",
            "scoring_performed",
            "runner_provenance",
            "source",
            "input",
            "selection",
            "assets",
            "config",
            "protocol_assumptions",
            "runtime_versions",
            "restore_audit",
            "decode_ok_total",
            "failure_total",
            "failures",
            "predictions",
        },
        "IVAC-P2L prediction",
    )
    if (
        root["schema_version"] != 1
        or root["method_id"] != METHOD_ID
        or root["classification"] != CLASSIFICATION
    ):
        raise ValueError("IVAC-P2L prediction method/schema mismatch")
    if (
        root["eligible_for_original_pams_ivac_p2l_cell"] is not False
        or root["labels_loaded"] is not False
        or root["scoring_performed"] is not False
    ):
        raise ValueError("IVAC-P2L prediction crossed the label boundary")
    runner_provenance = _strict_object(
        root["runner_provenance"],
        {
            "source_git_sha",
            "runner_code_sha256",
            "compatibility_image_digest",
        },
        "IVAC-P2L runner provenance",
    )
    if runner_provenance != {
        "source_git_sha": expected_runner_git_sha,
        "runner_code_sha256": expected_runner_code_sha256,
        "compatibility_image_digest": COMPATIBILITY_IMAGE_DIGEST,
    }:
        raise ValueError(
            "IVAC-P2L prediction was not produced by this clean runner revision"
        )
    if root["source"] != official_source_dict():
        raise ValueError("IVAC-P2L source binding is not frozen")

    input_binding = _strict_object(
        root["input"],
        {
            "protocol",
            "split",
            "sample_count",
            "sidecar_sha256",
            "commitment_sha256",
            "sidecar_fingerprint",
            "identity_sha256",
        },
        "IVAC-P2L input binding",
    )
    if input_binding != label_free_inputs.input_binding:
        raise ValueError(
            "IVAC-P2L prediction does not bind the supplied sidecar/commitment"
        )

    selection = _strict_object(
        root["selection"],
        {"selected_count", "selected_video_ids", "selected_video_ids_sha256"},
        "IVAC-P2L selection",
    )
    ids = selection["selected_video_ids"]
    if (
        selection["selected_count"] != 84
        or not isinstance(ids, list)
        or len(ids) != 84
        or any(not isinstance(value, str) or not value for value in ids)
        or len(set(ids)) != 84
    ):
        raise ValueError("IVAC-P2L scoring requires 84 unique selected IDs")
    video_ids = tuple(ids)
    manifest_records = label_free_inputs.manifest.records
    if video_ids != tuple(record.video_id for record in manifest_records):
        raise ValueError("IVAC-P2L selection/order differs from the sidecar")
    if _canonical_dev_id_sha256(video_ids) != CANONICAL_DEV_ID_SHA256:
        raise ValueError("IVAC-P2L IDs are not the frozen dev identities")
    if selection["selected_video_ids_sha256"] != sha256_json(list(video_ids)):
        raise ValueError("IVAC-P2L selection digest mismatch")

    assets = _strict_object(
        root["assets"],
        {"backbone", "checkpoint", "third_party_assets_redistributed"},
        "IVAC-P2L assets",
    )
    if (
        assets["backbone"] != BACKBONE_SPEC.public_dict()
        or assets["checkpoint"] != CHECKPOINT_SPEC.public_dict()
        or assets["third_party_assets_redistributed"] is not False
    ):
        raise ValueError("IVAC-P2L asset identity mismatch")
    config = _strict_object(root["config"], {"sha256", "values"}, "IVAC-P2L config")
    if (
        config["values"] != FROZEN_IVAC_P2L_OFFICIAL_CONFIG.to_dict()
        or config["sha256"] != FROZEN_IVAC_P2L_OFFICIAL_CONFIG.fingerprint
    ):
        raise ValueError("IVAC-P2L config is not frozen")
    if root["protocol_assumptions"] != list(PROTOCOL_ASSUMPTIONS):
        raise ValueError("IVAC-P2L protocol assumptions differ")

    runtime = root["runtime_versions"]
    runtime_keys = {
        "python",
        "torch",
        "cuda_runtime",
        "torchvision",
        "mmcv",
        "opencv",
        "numpy",
        "gpu_name",
    }
    if (
        not isinstance(runtime, dict)
        or set(runtime) != runtime_keys
        or any(not isinstance(value, str) or not value for value in runtime.values())
    ):
        raise ValueError("IVAC-P2L runtime provenance is incomplete")
    restore = _strict_object(
        root["restore_audit"],
        {
            "checkpoint_epoch",
            "state_dict_keys",
            "loaded_key_count",
            "missing_keys",
            "unexpected_keys",
            "strict_key_coverage",
        },
        "IVAC-P2L restore audit",
    )
    if restore != {
        "checkpoint_epoch": 67,
        "state_dict_keys": 230,
        "loaded_key_count": 230,
        "missing_keys": [],
        "unexpected_keys": [],
        "strict_key_coverage": True,
    }:
        raise ValueError("IVAC-P2L checkpoint restore audit mismatch")
    if (
        root["decode_ok_total"] != 84
        or root["failure_total"] != 0
        or root["failures"] != []
    ):
        raise ValueError("IVAC-P2L dev prediction contains failures")

    rows = root["predictions"]
    if not isinstance(rows, list) or len(rows) != 84:
        raise ValueError("IVAC-P2L dev prediction requires exactly 84 rows")
    row_fields = {
        "video_id",
        "video_locator",
        "expected_video_sha256",
        "observed_video_sha256",
        "raw_count",
        "rounded_count",
        "decoded_frame_count",
        "sampled_frame_count",
        "density_min",
        "density_max",
        "density_mean",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "decode_status",
        "failure_reason",
    }
    prediction_ids: list[str] = []
    raw_counts: list[float] = []
    rounded_counts: list[int] = []
    for index, (raw_row, manifest_record) in enumerate(
        zip(rows, manifest_records, strict=True)
    ):
        row = _strict_object(raw_row, row_fields, f"IVAC-P2L row {index}")
        video_id = row["video_id"]
        locator = row["video_locator"]
        if not isinstance(video_id, str) or not video_id:
            raise ValueError(f"IVAC-P2L row {index} has invalid video_id")
        if not isinstance(locator, str) or "\\" in locator:
            raise ValueError(f"IVAC-P2L row {index} has invalid locator")
        locator_path = PurePosixPath(locator)
        if locator_path.is_absolute() or any(
            part in {"", ".", ".."} for part in locator_path.parts
        ):
            raise ValueError(f"IVAC-P2L row {index} has unsafe locator")
        expected_video_sha = _sha256(
            row["expected_video_sha256"],
            f"rows[{index}].expected_video_sha256",
        )
        if (
            video_id != manifest_record.video_id
            or locator != manifest_record.video_path
            or expected_video_sha != manifest_record.video_sha256
        ):
            raise ValueError(
                f"IVAC-P2L row {index} locator/video SHA differs from sidecar"
            )
        if (
            _sha256(
                row["observed_video_sha256"],
                f"rows[{index}].observed_video_sha256",
            )
            != expected_video_sha
        ):
            raise ValueError(f"IVAC-P2L row {index} observed video SHA mismatch")
        if row["decode_status"] != "ok" or row["failure_reason"] is not None:
            raise ValueError(f"IVAC-P2L row {index} is not successful")
        raw_count = _finite(row["raw_count"], f"rows[{index}].raw_count", minimum=0)
        rounded_count = row["rounded_count"]
        if (
            isinstance(rounded_count, bool)
            or not isinstance(rounded_count, int)
            or rounded_count != round_count(raw_count)
        ):
            raise ValueError(f"IVAC-P2L row {index} rounded count mismatch")
        if (
            isinstance(row["decoded_frame_count"], bool)
            or not isinstance(row["decoded_frame_count"], int)
            or row["decoded_frame_count"] < 1
            or row["sampled_frame_count"] != 64
        ):
            raise ValueError(f"IVAC-P2L row {index} frame metadata mismatch")
        for field in ("density_min", "density_max", "density_mean"):
            _finite(row[field], f"rows[{index}].{field}")
        for field in ("peak_allocated_mib", "peak_reserved_mib"):
            _finite(row[field], f"rows[{index}].{field}", minimum=0)
        prediction_ids.append(video_id)
        raw_counts.append(raw_count)
        rounded_counts.append(rounded_count)
    if tuple(prediction_ids) != video_ids:
        raise ValueError("IVAC-P2L row order differs from selection")

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
            "decode_ok_total",
            "failure_total",
            "source_commit",
            "source_archive_sha256",
            "source_tree_sha256",
            "backbone_sha256",
            "checkpoint_sha256",
            "input_sidecar_sha256",
            "input_commitment_sha256",
            "input_identity_sha256",
            "config_sha256",
            "compatibility_image_digest",
            "runner_source_git_sha",
            "runner_code_sha256",
            "labels_loaded",
            "scoring_performed",
        },
        "IVAC-P2L prediction receipt",
    )
    expected_receipt = {
        "schema_version": 1,
        "artifact_type": "ivac_p2l_official_prediction_receipt",
        "method_id": METHOD_ID,
        "prediction_file": prediction_path.name,
        "prediction_sha256": prediction_digest.sha256,
        "prediction_bytes": prediction_digest.byte_count,
        "prediction_total": 84,
        "decode_ok_total": 84,
        "failure_total": 0,
        "source_commit": OFFICIAL_SOURCE_COMMIT,
        "source_archive_sha256": SOURCE_ARCHIVE_SPEC.sha256,
        "source_tree_sha256": OFFICIAL_SOURCE_TREE_SHA256,
        "backbone_sha256": BACKBONE_SPEC.sha256,
        "checkpoint_sha256": CHECKPOINT_SPEC.sha256,
        "input_sidecar_sha256": input_binding["sidecar_sha256"],
        "input_commitment_sha256": input_binding["commitment_sha256"],
        "input_identity_sha256": input_binding["identity_sha256"],
        "config_sha256": FROZEN_IVAC_P2L_OFFICIAL_CONFIG.fingerprint,
        "compatibility_image_digest": COMPATIBILITY_IMAGE_DIGEST,
        "runner_source_git_sha": expected_runner_git_sha,
        "runner_code_sha256": expected_runner_code_sha256,
        "labels_loaded": False,
        "scoring_performed": False,
    }
    if receipt_root != expected_receipt:
        raise ValueError("IVAC-P2L receipt does not bind the prediction")
    if _stable_file_digest(prediction_path) != prediction_digest:
        raise IVACP2LOfficialScoreError("prediction changed during validation")
    if _stable_file_digest(receipt_path) != receipt_digest:
        raise IVACP2LOfficialScoreError(
            "prediction receipt changed during validation"
        )
    label_free_inputs.assert_unchanged()
    return ValidatedPredictions(
        prediction_sha256=prediction_digest.sha256,
        prediction_bytes=prediction_digest.byte_count,
        receipt_sha256=receipt_digest.sha256,
        video_ids=video_ids,
        raw_counts=tuple(raw_counts),
        rounded_counts=tuple(rounded_counts),
        input_binding=dict(input_binding),
        runtime_versions=dict(runtime),
        runner_git_sha=expected_runner_git_sha,
        runner_code_sha256=expected_runner_code_sha256,
    )


def compute_ivac_p2l_metrics(
    raw: Sequence[float],
    rounded: Sequence[int],
    targets: Sequence[int],
) -> dict[str, float]:
    if not raw or not (len(raw) == len(rounded) == len(targets)):
        raise ValueError("metric inputs must have the same nonzero length")
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
    return float(ordered[lower] * (1 - fraction) + ordered[upper] * fraction)


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
    generator = random.Random(seed)
    values: dict[str, list[float]] = {
        key: [] for key in compute_ivac_p2l_metrics(raw, rounded, targets)
    }
    for _ in range(samples):
        indices = [generator.randrange(len(targets)) for _ in targets]
        metrics = compute_ivac_p2l_metrics(
            [raw[index] for index in indices],
            [rounded[index] for index in indices],
            [targets[index] for index in indices],
        )
        for key, value in metrics.items():
            values[key].append(value)
    return {
        key: {
            "low": _percentile(metric, 0.025),
            "high": _percentile(metric, 0.975),
        }
        for key, metric in values.items()
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
        raise IVACP2LOfficialScoreError(
            "unable to audit scoring Git checkout"
        ) from exc
    if GIT_SHA_PATTERN.fullmatch(revision) is None or status:
        raise IVACP2LOfficialScoreError(
            "sealed IVAC-P2L scoring requires a clean full-SHA Git checkout"
        )
    return revision


def score_ivac_p2l_dev(
    *,
    predictions_path: str | Path,
    prediction_receipt_path: str | Path,
    sidecar_path: str | Path,
    commitment_path: str | Path,
    dev_targets_path: str | Path,
    output_dir: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Validate all label-free bindings, then first open and score targets."""

    prediction_file = Path(predictions_path).expanduser().resolve(strict=True)
    prediction_receipt = (
        Path(prediction_receipt_path).expanduser().resolve(strict=True)
    )
    sidecar_file = Path(sidecar_path).expanduser().resolve(strict=True)
    commitment_file = Path(commitment_path).expanduser().resolve(strict=True)
    target_file = Path(dev_targets_path).expanduser().resolve(strict=False)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    repository = Path(repository_root).expanduser().resolve(strict=True)
    destination.mkdir(parents=True, exist_ok=True)
    evaluation_path = destination / "evaluation.json"
    receipt_path = destination / "evaluation.receipt.json"
    if evaluation_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite IVAC-P2L score artifacts")

    # Nothing above or in this validation block reads target_file.
    label_free_inputs = _load_label_free_scoring_inputs(
        sidecar_file,
        commitment_file,
    )
    scoring_git_sha = _clean_git_revision(repository)
    scoring_code = (repository / SCORING_CODE_RELATIVE_PATH).resolve(strict=True)
    if scoring_code != Path(__file__).resolve(strict=True):
        raise IVACP2LOfficialScoreError("executed scorer is outside repository_root")
    scoring_code_digest = _stable_file_digest(scoring_code)
    runner_code = (repository / RUNNER_CODE_RELATIVE_PATH).resolve(strict=True)
    expected_runner_module = Path(__file__).with_name(
        "ivac_p2l_official_runner.py"
    ).resolve(strict=True)
    if runner_code != expected_runner_module:
        raise IVACP2LOfficialScoreError(
            "bound IVAC-P2L runner is outside repository_root"
        )
    runner_code_digest = _stable_file_digest(runner_code)
    predictions = _validate_prediction(
        prediction_file,
        prediction_receipt,
        label_free_inputs,
        expected_runner_git_sha=scoring_git_sha,
        expected_runner_code_sha256=runner_code_digest.sha256,
    )

    # This full digest is the first target-file read.
    target_digest = _stable_file_digest(target_file)
    targets = load_dev_target_manifest(target_file)
    target_ids = tuple(row.video_id for row in targets.records)
    if target_ids != predictions.video_ids:
        raise ValueError("IVAC-P2L prediction/target IDs or order differ")
    target_counts = tuple(row.count for row in targets.records)
    metrics = compute_ivac_p2l_metrics(
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
        raise IVACP2LOfficialScoreError("dev targets changed during scoring")
    if _stable_file_digest(prediction_file).sha256 != predictions.prediction_sha256:
        raise IVACP2LOfficialScoreError("prediction changed during scoring")
    if (
        _stable_file_digest(prediction_receipt).sha256
        != predictions.receipt_sha256
    ):
        raise IVACP2LOfficialScoreError(
            "prediction receipt changed during scoring"
        )
    if _stable_file_digest(scoring_code) != scoring_code_digest:
        raise IVACP2LOfficialScoreError("scoring code changed during scoring")
    if _stable_file_digest(runner_code) != runner_code_digest:
        raise IVACP2LOfficialScoreError("IVAC-P2L runner code changed during scoring")
    if _clean_git_revision(repository) != scoring_git_sha:
        raise IVACP2LOfficialScoreError("scoring Git revision changed during scoring")
    label_free_inputs.assert_unchanged()

    evaluation = {
        "schema_version": 1,
        "artifact_type": "ivac_p2l_official_dev_evaluation",
        "classification": EVALUATION_CLASSIFICATION,
        "eligible_for_original_paper_table": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "prediction_total": 84,
        "sealed_test_status": "untouched",
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
            "backbone_sha256": BACKBONE_SPEC.sha256,
            "checkpoint_sha256": CHECKPOINT_SPEC.sha256,
            "input_manifest_sha256": predictions.input_binding["sidecar_sha256"],
            "input_commitment_sha256": predictions.input_binding[
                "commitment_sha256"
            ],
            "input_identity_sha256": predictions.input_binding["identity_sha256"],
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
        "artifact_type": "ivac_p2l_official_dev_evaluation_receipt",
        "classification": EVALUATION_CLASSIFICATION,
        "eligible_for_original_paper_table": False,
        "evaluation_file": evaluation_path.name,
        "evaluation_sha256": evaluation_sha256,
        "evaluation_bytes": evaluation_path.stat().st_size,
        "prediction_sha256": predictions.prediction_sha256,
        "prediction_receipt_sha256": predictions.receipt_sha256,
        "input_manifest_sha256": predictions.input_binding["sidecar_sha256"],
        "input_commitment_sha256": predictions.input_binding["commitment_sha256"],
        "input_identity_sha256": predictions.input_binding["identity_sha256"],
        "dev_targets_sha256": target_digest.sha256,
        "scoring_code_sha256": scoring_code_digest.sha256,
        "scoring_source_git_sha": scoring_git_sha,
        "runner_code_sha256": runner_code_digest.sha256,
        "runner_source_git_sha": scoring_git_sha,
        "compatibility_image_digest": COMPATIBILITY_IMAGE_DIGEST,
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
                "IVAC-P2L score receipt write failed and the exact orphan "
                "evaluation could not be removed safely"
            ) from exc
        raise
    return {
        "classification": EVALUATION_CLASSIFICATION,
        "eligible_for_original_paper_table": False,
        "metrics": metrics,
        "confidence_intervals": intervals,
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": evaluation_sha256,
        "evaluation_receipt_path": str(receipt_path),
        "evaluation_receipt_sha256": receipt_sha256,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pams.baselines.ivac_p2l_official_score",
        description="Score a validated IVAC-P2L prediction/receipt pair.",
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
        result = score_ivac_p2l_dev(
            predictions_path=arguments.predictions,
            prediction_receipt_path=arguments.prediction_receipt,
            sidecar_path=arguments.sidecar,
            commitment_path=arguments.commitment,
            dev_targets_path=arguments.dev_targets,
            output_dir=arguments.output_dir,
            repository_root=arguments.repository_root,
        )
    except (OSError, IVACP2LOfficialScoreError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
