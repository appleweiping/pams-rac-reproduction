#!/usr/bin/env python3
"""Aggregate the frozen 42/2026/3407 PAMS temporal-conv v7 dev84 results.

The seed-3407 follow-up was run after the two prespecified seeds had already
failed, so this artifact is permanently classified as a post-hoc variance
follow-up.  The paired bootstrap resamples the same video indices for all
three seeds and averages the three per-seed metrics in each replicate.

The historical seed-42/2026 evaluation format does not carry source or target
hashes.  Required caller bindings are therefore recorded, but never presented
as verified for a legacy input.  The strict dev-score format is checked
directly against those bindings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

_EXPECTED_SEEDS = (42, 2026, 3407)
_EXPECTED_LEGACY_METHOD = "pams-sshead"
_EXPECTED_STRICT_METHOD = "pams-sshead-inferred"
_EXPECTED_VARIANT = "sshead"
_CLASSIFICATION = "post-hoc-three-seed-variance-follow-up"
_BOOTSTRAP_SAMPLES = 10_000
_BOOTSTRAP_SEED = 2026
_CONFIDENCE_LEVEL = 0.95
_METRIC_NAMES = ("nmae", "mae", "rmse", "obo", "exact")
_SHA256_LENGTH = 64
_GIT_SHA_LENGTH = 40


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hex(value: str, *, length: int, field: str) -> str:
    if len(value) != length or value != value.lower():
        raise ValueError(f"{field} must be a lowercase {length}-character hexadecimal value")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{field} must be hexadecimal") from error
    return value


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    created = False
    try:
        with path.open("xb") as handle:
            created = True
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise


def _parse_seed_path(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator or not path_text:
        raise argparse.ArgumentTypeError("evaluation must use SEED=PATH")
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("evaluation seed must be an integer") from error
    return seed, Path(path_text)


def _parse_seed_sha256(value: str) -> tuple[int, str]:
    seed_text, separator, digest = value.partition("=")
    if not separator or not digest:
        raise argparse.ArgumentTypeError("config fingerprint must use SEED=SHA256")
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("config seed must be an integer") from error
    try:
        canonical = _canonical_hex(
            digest,
            length=_SHA256_LENGTH,
            field=f"config fingerprint for seed {seed}",
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return seed, canonical


def _close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)


def _schema_flavor(payload: dict[str, Any], *, seed: int) -> Literal["legacy", "strict"]:
    if payload.get("method_id") == _EXPECTED_LEGACY_METHOD:
        if "method_key" in payload:
            raise ValueError(f"seed {seed}: evaluation mixes legacy and strict method fields")
        return "legacy"
    if payload.get("method_key") == _EXPECTED_STRICT_METHOD:
        if payload.get("artifact_type") != "pams_checkpoint_dev_evaluation":
            raise ValueError(f"seed {seed}: unexpected strict artifact type")
        if "method_id" in payload:
            raise ValueError(f"seed {seed}: evaluation mixes legacy and strict method fields")
        return "strict"
    raise ValueError(f"seed {seed}: evaluation is not a supported PAMS-SSHead schema")


def _validate_predictions(
    *,
    seed: int,
    payload: dict[str, Any],
    per_video: list[dict[str, Any]],
) -> None:
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 84:
        raise ValueError(f"seed {seed}: expected 84 top-level predictions")
    prediction_ids = [str(row["video_id"]) for row in predictions]
    report_ids = [str(row["video_id"]) for row in per_video]
    if prediction_ids != report_ids:
        raise ValueError(f"seed {seed}: prediction and report order differ")
    for index, (prediction, report_row) in enumerate(zip(predictions, per_video, strict=True)):
        count = prediction.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"seed {seed}: prediction {index} has an invalid count")
        if not _close(float(report_row["prediction"]), float(count)):
            raise ValueError(f"seed {seed}: prediction {index} differs from its report row")
        rounded = report_row["rounded_prediction"]
        if isinstance(rounded, bool) or not isinstance(rounded, int) or rounded != count:
            raise ValueError(f"seed {seed}: rounded prediction {index} differs from count")


def _load_evaluation(
    seed: int,
    path: Path,
    *,
    source_git_sha: str,
    dev_targets_sha256: str,
    config_fingerprint: str,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"seed {seed}: evaluation root must be an object")
    flavor = _schema_flavor(payload, seed=seed)
    if payload.get("variant") != _EXPECTED_VARIANT:
        raise ValueError(f"seed {seed}: unexpected variant")
    if payload.get("split") != "dev":
        raise ValueError(f"seed {seed}: only dev evaluations are allowed")
    if flavor == "legacy" and payload.get("stage") != "sshead":
        raise ValueError(f"seed {seed}: legacy stage must be sshead")
    if flavor == "strict":
        if payload.get("protocol") != "ucfrep_526":
            raise ValueError(f"seed {seed}: unexpected strict protocol")
        if payload.get("table2_eligible") is not False:
            raise ValueError(f"seed {seed}: strict evaluation must be Table-2 ineligible")
        for field in (
            "checkpoint_source_git_sha",
            "prediction_source_git_sha",
            "scoring_source_git_sha",
        ):
            if payload.get(field) != source_git_sha:
                raise ValueError(f"seed {seed}: {field} differs from the caller binding")
        if payload.get("dev_targets_sha256") != dev_targets_sha256:
            raise ValueError(f"seed {seed}: dev target binding changed")

    observed_config = payload.get("config_fingerprint")
    if observed_config != config_fingerprint:
        raise ValueError(f"seed {seed}: config fingerprint differs from the caller binding")
    _canonical_hex(
        str(observed_config),
        length=_SHA256_LENGTH,
        field=f"seed {seed} config_fingerprint",
    )

    report = payload.get("report")
    if not isinstance(report, dict):
        raise TypeError(f"seed {seed}: report object is required")
    if (
        report.get("sample_count") != 84
        or report.get("bootstrap_samples") != _BOOTSTRAP_SAMPLES
        or report.get("bootstrap_seed") != _BOOTSTRAP_SEED
    ):
        raise ValueError(f"seed {seed}: frozen dev84/bootstrap protocol changed")
    per_video = report.get("per_video")
    if not isinstance(per_video, list) or len(per_video) != 84:
        raise ValueError(f"seed {seed}: expected 84 per-video rows")
    identifiers = [str(row["video_id"]) for row in per_video]
    if len(set(identifiers)) != 84:
        raise ValueError(f"seed {seed}: video identifiers are not unique")
    actions = tuple(str(row["action"]) for row in per_video)
    if any(not action for action in actions):
        raise ValueError(f"seed {seed}: empty action label")

    _validate_predictions(seed=seed, payload=payload, per_video=per_video)
    absolute = np.asarray([row["absolute_error"] for row in per_video], dtype=np.float64)
    target = np.asarray([row["target"] for row in per_video], dtype=np.float64)
    normalized = np.asarray(
        [row["normalized_absolute_error"] for row in per_video],
        dtype=np.float64,
    )
    rounded = np.asarray([row["rounded_prediction"] for row in per_video], dtype=np.float64)
    within_one_values = [row["within_one"] for row in per_video]
    exact_values = [row["exact"] for row in per_video]
    if not all(isinstance(value, bool) for value in within_one_values + exact_values):
        raise TypeError(f"seed {seed}: exact/within_one values must be booleans")
    within_one = np.asarray(within_one_values, dtype=np.float64)
    exact = np.asarray(exact_values, dtype=np.float64)
    if (
        not np.isfinite(absolute).all()
        or not np.isfinite(target).all()
        or not np.isfinite(normalized).all()
        or not np.isfinite(rounded).all()
        or np.any(target <= 0)
        or np.any(absolute < 0)
    ):
        raise ValueError(f"seed {seed}: non-finite or invalid per-video values")
    recomputed_absolute = np.abs(rounded - target)
    if not np.array_equal(absolute, recomputed_absolute):
        raise ValueError(f"seed {seed}: absolute errors do not recompute")
    if not np.allclose(normalized, absolute / target, rtol=0.0, atol=1e-12):
        raise ValueError(f"seed {seed}: normalized errors do not recompute")
    if not np.array_equal(within_one, absolute <= 1):
        raise ValueError(f"seed {seed}: within_one flags do not recompute")
    if not np.array_equal(exact, absolute == 0):
        raise ValueError(f"seed {seed}: exact flags do not recompute")
    recomputed = {
        "nmae": float(normalized.mean()),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(absolute).mean())),
        "obo": float(within_one.mean()),
        "exact": float(exact.mean()),
    }
    for name, value in recomputed.items():
        if name not in report or not _close(float(report[name]), value):
            raise ValueError(f"seed {seed}: {name} does not recompute")

    config_file_sha256 = payload.get("config_file_sha256")
    if config_file_sha256 is not None:
        _canonical_hex(
            str(config_file_sha256),
            length=_SHA256_LENGTH,
            field=f"seed {seed} config_file_sha256",
        )
    return {
        "seed": seed,
        "path": resolved,
        "sha256": _sha256(resolved),
        "payload": payload,
        "schema_flavor": flavor,
        "report": report,
        "video_ids": tuple(identifiers),
        "actions": actions,
        "targets": target,
        "absolute": absolute,
        "normalized": normalized,
        "within_one": within_one,
        "exact": exact,
        "config_file_sha256": config_file_sha256,
        "source_binding_evidence": (
            "verified_against_checkpoint_prediction_and_scoring_source_fields"
            if flavor == "strict"
            else "unverifiable_from_legacy_evaluation"
        ),
        "target_binding_evidence": (
            "verified_against_dev_targets_sha256"
            if flavor == "strict"
            else "unverifiable_from_legacy_evaluation"
        ),
    }


def _paired_bootstrap(
    evaluations: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    seed_count = len(evaluations)
    sample_count = len(evaluations[0]["video_ids"])
    distributions = {name: np.empty(_BOOTSTRAP_SAMPLES, dtype=np.float64) for name in _METRIC_NAMES}
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    chunk_size = 512
    for start in range(0, _BOOTSTRAP_SAMPLES, chunk_size):
        stop = min(start + chunk_size, _BOOTSTRAP_SAMPLES)
        indices = rng.integers(0, sample_count, size=(stop - start, sample_count))
        chunk_values: dict[str, list[NDArray[np.float64]]] = {name: [] for name in _METRIC_NAMES}
        for evaluation in evaluations:
            chunk_values["nmae"].append(evaluation["normalized"][indices].mean(axis=1))
            chunk_values["mae"].append(evaluation["absolute"][indices].mean(axis=1))
            chunk_values["rmse"].append(
                np.sqrt(np.square(evaluation["absolute"])[indices].mean(axis=1))
            )
            chunk_values["obo"].append(evaluation["within_one"][indices].mean(axis=1))
            chunk_values["exact"].append(evaluation["exact"][indices].mean(axis=1))
        for name in _METRIC_NAMES:
            stacked = np.stack(chunk_values[name], axis=0)
            if stacked.shape != (seed_count, stop - start):
                raise AssertionError("bootstrap shape changed")
            distributions[name][start:stop] = stacked.mean(axis=0)

    tail = (1.0 - _CONFIDENCE_LEVEL) / 2.0
    return {
        name: {
            "low": float(np.quantile(values, tail)),
            "high": float(np.quantile(values, 1.0 - tail)),
            "level": _CONFIDENCE_LEVEL,
        }
        for name, values in distributions.items()
    }


def aggregate_evaluations(
    specifications: list[tuple[int, Path]],
    *,
    source_git_sha: str,
    dev_targets_sha256: str,
    config_fingerprints: dict[int, str],
) -> dict[str, Any]:
    source_git_sha = _canonical_hex(
        source_git_sha,
        length=_GIT_SHA_LENGTH,
        field="source_git_sha",
    )
    dev_targets_sha256 = _canonical_hex(
        dev_targets_sha256,
        length=_SHA256_LENGTH,
        field="dev_targets_sha256",
    )
    supplied_seeds = tuple(seed for seed, _ in specifications)
    if len(set(supplied_seeds)) != len(supplied_seeds):
        raise ValueError("each seed may be supplied only once")
    if set(supplied_seeds) != set(_EXPECTED_SEEDS):
        raise ValueError(f"evaluations must cover exactly seeds {_EXPECTED_SEEDS}")
    if set(config_fingerprints) != set(_EXPECTED_SEEDS):
        raise ValueError(f"config fingerprints must cover exactly seeds {_EXPECTED_SEEDS}")
    canonical_configs = {
        seed: _canonical_hex(
            digest,
            length=_SHA256_LENGTH,
            field=f"config fingerprint for seed {seed}",
        )
        for seed, digest in config_fingerprints.items()
    }
    evaluations = [
        _load_evaluation(
            seed,
            path,
            source_git_sha=source_git_sha,
            dev_targets_sha256=dev_targets_sha256,
            config_fingerprint=canonical_configs[seed],
        )
        for seed, path in sorted(specifications, key=lambda item: item[0])
    ]
    reference = evaluations[0]
    for evaluation in evaluations[1:]:
        if (
            evaluation["video_ids"] != reference["video_ids"]
            or evaluation["actions"] != reference["actions"]
            or not np.array_equal(evaluation["targets"], reference["targets"])
        ):
            raise ValueError("per-video membership, actions, or targets differ across seeds")

    intervals = _paired_bootstrap(evaluations)
    metrics: dict[str, Any] = {}
    for name in _METRIC_NAMES:
        seed_values = [float(row["report"][name]) for row in evaluations]
        metrics[name] = {
            "mean": statistics.fmean(seed_values),
            "sample_sd": statistics.stdev(seed_values),
            "paired_bootstrap_95_percent": intervals[name],
            "per_seed": {str(row["seed"]): float(row["report"][name]) for row in evaluations},
        }
    numerically_passing_seeds = sum(
        row["report"]["nmae"] <= 0.228 and row["report"]["obo"] >= 0.666 for row in evaluations
    )
    strict_seeds = [row["seed"] for row in evaluations if row["schema_flavor"] == "strict"]
    legacy_seeds = [row["seed"] for row in evaluations if row["schema_flavor"] == "legacy"]

    return {
        "schema_version": 1,
        "artifact_type": "pams_sshead_temporal_conv_v7_three_seed_dev84_aggregate",
        "classification": _CLASSIFICATION,
        "method_id": _EXPECTED_LEGACY_METHOD,
        "normalized_strict_method_key": _EXPECTED_STRICT_METHOD,
        "variant": _EXPECTED_VARIANT,
        "protocol": "ucfrep_526_train337_dev84",
        "split": "dev",
        "sample_count_per_seed": 84,
        "seeds": list(_EXPECTED_SEEDS),
        "input_schema_coverage": {
            "legacy_seeds": legacy_seeds,
            "strict_dev_score_seeds": strict_seeds,
        },
        "caller_audit_bindings": {
            "source_git_sha": source_git_sha,
            "dev_targets_sha256": dev_targets_sha256,
            "config_fingerprints": {str(seed): canonical_configs[seed] for seed in _EXPECTED_SEEDS},
            "legacy_boundary": (
                "Legacy evaluations verify config_fingerprint and per-video target "
                "membership, but do not carry source_git_sha or dev_targets_sha256. "
                "Their source/target bindings remain caller assertions."
            ),
        },
        "input_evaluations": [
            {
                "seed": row["seed"],
                "schema_flavor": row["schema_flavor"],
                "sha256": row["sha256"],
                "config_fingerprint": row["payload"]["config_fingerprint"],
                "config_file_sha256": row["config_file_sha256"],
                "checkpoint_sha256": row["payload"].get("checkpoint_sha256"),
                "prediction_sha256": row["payload"].get("prediction_sha256"),
                "source_binding_evidence": row["source_binding_evidence"],
                "target_binding_evidence": row["target_binding_evidence"],
            }
            for row in evaluations
        ],
        "metrics_after_half_up_rounding": metrics,
        "bootstrap": {
            "samples": _BOOTSTRAP_SAMPLES,
            "seed": _BOOTSTRAP_SEED,
            "confidence_level": _CONFIDENCE_LEVEL,
            "method": (
                "paired percentile over identical video indices; same indices for all "
                "seeds; mean of per-seed metrics in each replicate"
            ),
        },
        "frozen_gate_reference_only": {
            "required_nmae_at_most": 0.228,
            "required_obo_at_least": 0.666,
            "required_individual_seed_passes": 2,
            "mean_nmae_pass": metrics["nmae"]["mean"] <= 0.228,
            "mean_obo_pass": metrics["obo"]["mean"] >= 0.666,
            "individual_seeds_jointly_passing": numerically_passing_seeds,
            "numerical_gate_pass": (
                metrics["nmae"]["mean"] <= 0.228
                and metrics["obo"]["mean"] >= 0.666
                and numerically_passing_seeds >= 2
            ),
            "claim_override": (
                "Post-hoc execution makes this aggregate ineligible for the verified "
                "claim regardless of its numerical values."
            ),
        },
        "claim_boundary": {
            "post_hoc": True,
            "eligible_for_original_paper_table": False,
            "eligible_for_verified_pams_reproduction_claim": False,
            "eligible_for_preregistered_three_seed_claim": False,
            "development_only": True,
            "test105_access": False,
        },
        "aggregator": {
            "path": "scripts/aggregate_pams_v7_seed_evaluations.py",
            "sha256": _sha256(Path(__file__).resolve(strict=True)),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        action="append",
        type=_parse_seed_path,
        required=True,
        help="frozen evaluation as SEED=PATH; supply exactly 42, 2026, and 3407",
    )
    parser.add_argument(
        "--config-fingerprint",
        action="append",
        type=_parse_seed_sha256,
        required=True,
        help="caller-audited PAMS config fingerprint as SEED=SHA256",
    )
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--dev-targets-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    config_bindings = dict(arguments.config_fingerprint)
    if len(config_bindings) != len(arguments.config_fingerprint):
        raise ValueError("each config seed may be supplied only once")
    payload = aggregate_evaluations(
        arguments.evaluation,
        source_git_sha=arguments.source_git_sha,
        dev_targets_sha256=arguments.dev_targets_sha256,
        config_fingerprints=config_bindings,
    )
    _write_json_exclusive(arguments.output.resolve(strict=False), payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
