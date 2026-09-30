#!/usr/bin/env python3
"""Aggregate the frozen 42/2026/3407 JTSPS dev84 evaluations.

The bootstrap resamples the same video indices for all seeds.  Each replicate
first computes one metric per seed and then averages those three metrics, so
the interval matches the reported mean-across-seeds estimand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

_EXPECTED_SEEDS = (42, 2026, 3407)
_EXPECTED_METHOD = "jtsps-count-only"
_EXPECTED_STATUS = "inferred-clean-room-not-source-parity"
_EXPECTED_DEV_TARGET_SHA256 = (
    "1ab3bc010ccb5d869af7662586594c973e53841838643e1e9b8cc1787b0efab6"
)
_BOOTSTRAP_SAMPLES = 10_000
_BOOTSTRAP_SEED = 2026
_CONFIDENCE_LEVEL = 0.95
_METRIC_NAMES = ("nmae", "mae", "rmse", "obo", "exact")
_RAW_METRIC_NAMES = ("nmae", "mae", "rmse")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {path}")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _parse_evaluation_specification(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator or not path_text:
        raise argparse.ArgumentTypeError("evaluation must use SEED=PATH")
    try:
        seed = int(seed_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("evaluation seed must be an integer") from error
    return seed, Path(path_text)


def _close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12)


def _load_evaluation(seed: int, path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("method") != _EXPECTED_METHOD:
        raise ValueError(f"seed {seed}: unexpected method")
    if payload.get("status") != _EXPECTED_STATUS:
        raise ValueError(f"seed {seed}: unexpected status")
    if payload.get("dev_targets_sha256") != _EXPECTED_DEV_TARGET_SHA256:
        raise ValueError(f"seed {seed}: dev target binding changed")
    metrics = payload.get("metrics")
    raw_metrics = payload.get("raw_metrics")
    if not isinstance(metrics, dict) or not isinstance(raw_metrics, dict):
        raise TypeError(f"seed {seed}: metrics objects are required")
    if (
        metrics.get("sample_count") != 84
        or metrics.get("bootstrap_samples") != _BOOTSTRAP_SAMPLES
        or metrics.get("bootstrap_seed") != _BOOTSTRAP_SEED
    ):
        raise ValueError(f"seed {seed}: frozen dev84/bootstrap protocol changed")
    per_video = metrics.get("per_video")
    if not isinstance(per_video, list) or len(per_video) != 84:
        raise ValueError(f"seed {seed}: expected 84 per-video rows")
    identifiers = [str(row["video_id"]) for row in per_video]
    if len(set(identifiers)) != 84:
        raise ValueError(f"seed {seed}: video identifiers are not unique")

    absolute = np.asarray([row["absolute_error"] for row in per_video], dtype=np.float64)
    target = np.asarray([row["target"] for row in per_video], dtype=np.float64)
    normalized = np.asarray(
        [row["normalized_absolute_error"] for row in per_video], dtype=np.float64
    )
    within_one = np.asarray([row["within_one"] for row in per_video], dtype=np.float64)
    exact = np.asarray([row["exact"] for row in per_video], dtype=np.float64)
    if (
        not np.isfinite(absolute).all()
        or not np.isfinite(target).all()
        or not np.isfinite(normalized).all()
        or np.any(target <= 0)
        or np.any(absolute < 0)
    ):
        raise ValueError(f"seed {seed}: non-finite or invalid per-video values")
    if not np.allclose(normalized, absolute / target, rtol=0.0, atol=1e-12):
        raise ValueError(f"seed {seed}: normalized errors do not recompute")
    recomputed = {
        "nmae": float(normalized.mean()),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.square(absolute).mean())),
        "obo": float(within_one.mean()),
        "exact": float(exact.mean()),
    }
    for name, value in recomputed.items():
        if not _close(float(metrics[name]), value):
            raise ValueError(f"seed {seed}: {name} does not recompute")
    for name in _RAW_METRIC_NAMES:
        if not math.isfinite(float(raw_metrics[name])):
            raise ValueError(f"seed {seed}: raw {name} is non-finite")
    return {
        "seed": seed,
        "path": resolved,
        "sha256": _sha256(resolved),
        "payload": payload,
        "video_ids": tuple(identifiers),
        "actions": tuple(row.get("action") for row in per_video),
        "targets": target,
        "absolute": absolute,
        "normalized": normalized,
        "within_one": within_one,
        "exact": exact,
    }


def _paired_bootstrap(
    evaluations: list[dict[str, Any]],
) -> dict[str, dict[str, float]]:
    seed_count = len(evaluations)
    sample_count = len(evaluations[0]["video_ids"])
    distributions = {
        name: np.empty(_BOOTSTRAP_SAMPLES, dtype=np.float64)
        for name in _METRIC_NAMES
    }
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    chunk_size = 512
    for start in range(0, _BOOTSTRAP_SAMPLES, chunk_size):
        stop = min(start + chunk_size, _BOOTSTRAP_SAMPLES)
        indices = rng.integers(0, sample_count, size=(stop - start, sample_count))
        chunk_values: dict[str, list[NDArray[np.float64]]] = {
            name: [] for name in _METRIC_NAMES
        }
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


def aggregate_evaluations(specifications: list[tuple[int, Path]]) -> dict[str, Any]:
    supplied_seeds = tuple(seed for seed, _ in specifications)
    if len(set(supplied_seeds)) != len(supplied_seeds):
        raise ValueError("each seed may be supplied only once")
    if set(supplied_seeds) != set(_EXPECTED_SEEDS):
        raise ValueError(f"evaluations must cover exactly seeds {_EXPECTED_SEEDS}")
    evaluations = [
        _load_evaluation(seed, path)
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
        seed_values = [float(row["payload"]["metrics"][name]) for row in evaluations]
        metrics[name] = {
            "mean": statistics.fmean(seed_values),
            "sample_sd": statistics.stdev(seed_values),
            "paired_bootstrap_95_percent": intervals[name],
            "per_seed": {
                str(row["seed"]): float(row["payload"]["metrics"][name])
                for row in evaluations
            },
        }
    raw_metrics: dict[str, Any] = {}
    for name in _RAW_METRIC_NAMES:
        seed_values = [float(row["payload"]["raw_metrics"][name]) for row in evaluations]
        raw_metrics[name] = {
            "mean": statistics.fmean(seed_values),
            "sample_sd": statistics.stdev(seed_values),
            "per_seed": {
                str(row["seed"]): float(row["payload"]["raw_metrics"][name])
                for row in evaluations
            },
        }

    return {
        "schema_version": 1,
        "artifact_type": "jtsps_count_only_three_seed_dev84_aggregate",
        "method": _EXPECTED_METHOD,
        "status": _EXPECTED_STATUS,
        "classification": "inferred clean-room baseline; not source parity",
        "protocol": "ucfrep_526_train337_dev84",
        "split": "dev",
        "sample_count_per_seed": 84,
        "seeds": list(_EXPECTED_SEEDS),
        "input_evaluations": [
            {
                "seed": row["seed"],
                "sha256": row["sha256"],
                "prediction_sha256": row["payload"]["prediction_sha256"],
            }
            for row in evaluations
        ],
        "metrics_after_half_up_rounding": metrics,
        "raw_metrics": raw_metrics,
        "bootstrap": {
            "samples": _BOOTSTRAP_SAMPLES,
            "seed": _BOOTSTRAP_SEED,
            "confidence_level": _CONFIDENCE_LEVEL,
            "method": (
                "paired percentile over video indices; same indices for all seeds; "
                "mean of per-seed metrics in each replicate"
            ),
        },
        "claim_boundary": {
            "eligible_for_original_paper_table": False,
            "eligible_for_verified_pams_reproduction_claim": False,
            "source_protocol_uniquely_verifiable": False,
            "development_only": True,
            "test105_access": False,
        },
        "frozen_gate_reference_only": {
            "required_nmae_at_most": 0.228,
            "required_obo_at_least": 0.666,
            "mean_nmae_pass": metrics["nmae"]["mean"] <= 0.228,
            "mean_obo_pass": metrics["obo"]["mean"] >= 0.666,
            "individual_seeds_jointly_passing": sum(
                row["payload"]["metrics"]["nmae"] <= 0.228
                and row["payload"]["metrics"]["obo"] >= 0.666
                for row in evaluations
            ),
            "note": "The gate is defined for PAMS verification, not baseline certification.",
        },
        "aggregator": {
            "path": "scripts/aggregate_jtsps_seed_evaluations.py",
            "sha256": _sha256(Path(__file__).resolve(strict=True)),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        action="append",
        type=_parse_evaluation_specification,
        required=True,
        help="frozen evaluation as SEED=PATH; supply exactly 42, 2026, and 3407",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    payload = aggregate_evaluations(arguments.evaluation)
    _write_json_exclusive(arguments.output.resolve(strict=False), payload)
    print(json.dumps(payload, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
