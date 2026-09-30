"""Strict paired-video comparison of frozen UCFRep dev84 evaluations.

The command consumes already-scored evaluation artifacts.  It never accepts a
dataset manifest, pose cache, video, or test artifact.  Repeated baseline runs
with the same group name are treated as seeds of one method; their metrics are
averaged at the method level, never flattened into independent observations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

DEV84_SIZE = 84
DEV84_TARGETS_SHA256 = "1ab3bc010ccb5d869af7662586594c973e53841838643e1e9b8cc1787b0efab6"
DEV84_VIDEO_ID_SHA256 = "2199294a1d22da6c67d2fdaaafb4bebaa11e92a5bb3accd4670975815001f2c6"
DEV84_TARGET_ROWS_SHA256 = "850541977de481223b2376c33aff2ccae3c872df848798d5de7b4c631f722a27"
METRICS = ("nmae", "mae", "rmse", "obo", "exact")
LABEL_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class InputSpec:
    group: str
    run: str
    path: Path
    row_path: str


@dataclass(frozen=True)
class LoadedRun:
    spec: InputSpec
    artifact_sha256: str
    method_id: str | None
    prediction_sha256: str | None
    rows: dict[str, dict[str, Any]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_test_path(path: Path) -> None:
    if "test105" in str(path).lower():
        raise ValueError(f"test105 paths are prohibited: {path.name}")


def _parse_spec(raw: str, *, candidate: bool) -> InputSpec:
    try:
        label, source = raw.split("=", 1)
        path_text, row_path = source.rsplit("::", 1)
    except ValueError as error:
        form = "NAME=FILE::DOT.PATH" if candidate else "GROUP/RUN=FILE::DOT.PATH"
        raise ValueError(f"invalid input spec; expected {form}") from error
    if candidate:
        group, run = label, "candidate"
    else:
        try:
            group, run = label.split("/", 1)
        except ValueError as error:
            raise ValueError("baseline label must be GROUP/RUN") from error
    if not LABEL_RE.fullmatch(group) or not LABEL_RE.fullmatch(run):
        raise ValueError("group and run labels may contain only letters, digits, ._-")
    if not row_path or any(not part for part in row_path.split(".")):
        raise ValueError("per-video JSON path must be a non-empty dotted path")
    path = Path(path_text).resolve()
    _reject_test_path(path)
    if not path.is_file():
        raise ValueError(f"evaluation artifact does not exist: {path.name}")
    return InputSpec(group=group, run=run, path=path, row_path=row_path)


def _extract(document: dict[str, Any], dotted_path: str) -> Any:
    value: Any = document
    for part in dotted_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise ValueError(f"missing JSON path {dotted_path!r}")
        value = value[part]
    return value


def _validated_row(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("each per-video row must be an object")
    required = {
        "video_id",
        "target",
        "prediction",
        "rounded_prediction",
        "absolute_error",
        "normalized_absolute_error",
        "within_one",
        "exact",
    }
    if not required.issubset(raw):
        raise ValueError(f"per-video row is missing {sorted(required - set(raw))}")
    video_id = str(raw["video_id"])
    if not video_id:
        raise ValueError("video_id must be non-empty")
    integer_fields: dict[str, int] = {}
    for field in ("target", "rounded_prediction", "absolute_error"):
        value = raw[field]
        if isinstance(value, bool):
            raise ValueError(f"{field} must be an integer for {video_id}")
        numeric = float(value)
        integer = int(numeric)
        if not np.isfinite(numeric) or numeric != integer:
            raise ValueError(f"{field} must be an integer for {video_id}")
        integer_fields[field] = integer
    target = integer_fields["target"]
    rounded = integer_fields["rounded_prediction"]
    absolute = integer_fields["absolute_error"]
    if target <= 0 or rounded < 0 or absolute != abs(rounded - target):
        raise ValueError(f"inconsistent count fields for {video_id}")
    normalized = float(raw["normalized_absolute_error"])
    if not np.isclose(normalized, absolute / target, rtol=0.0, atol=1e-12):
        raise ValueError(f"inconsistent normalized error for {video_id}")
    if not isinstance(raw["within_one"], bool) or not isinstance(raw["exact"], bool):
        raise ValueError(f"indicator fields must be booleans for {video_id}")
    within_one = raw["within_one"]
    exact = raw["exact"]
    if within_one != (absolute <= 1) or exact != (absolute == 0):
        raise ValueError(f"inconsistent indicator fields for {video_id}")
    prediction = float(raw["prediction"])
    if not np.isfinite(prediction):
        raise ValueError(f"non-finite prediction for {video_id}")
    return {
        "video_id": video_id,
        "target": target,
        "prediction": prediction,
        "rounded_prediction": rounded,
        "absolute_error": absolute,
        "normalized_absolute_error": normalized,
        "within_one": within_one,
        "exact": exact,
    }


def _load_run(spec: InputSpec, *, expected_targets_sha256: str) -> LoadedRun:
    document = json.loads(spec.path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"{spec.run}: evaluation root must be an object")
    if document.get("split") not in (None, "dev"):
        raise ValueError(f"{spec.run}: only split='dev' is accepted")
    if document.get("protocol") not in (None, "ucfrep_526"):
        raise ValueError(f"{spec.run}: only protocol='ucfrep_526' is accepted")
    if document.get("dev_targets_sha256") != expected_targets_sha256:
        raise ValueError(f"{spec.run}: dev target digest mismatch")
    raw_rows = _extract(document, spec.row_path)
    if not isinstance(raw_rows, list) or len(raw_rows) != DEV84_SIZE:
        raise ValueError(f"{spec.run}: expected exactly {DEV84_SIZE} per-video rows")
    rows: dict[str, dict[str, Any]] = {}
    for raw in raw_rows:
        row = _validated_row(raw)
        if row["video_id"] in rows:
            raise ValueError(f"{spec.run}: duplicate video_id {row['video_id']}")
        rows[row["video_id"]] = row
    prediction_sha = document.get("prediction_sha256")
    if prediction_sha is not None and not SHA256_RE.fullmatch(str(prediction_sha)):
        raise ValueError(f"{spec.run}: invalid prediction_sha256")
    method_id = document.get("method_id") or document.get("method_key")
    return LoadedRun(
        spec=spec,
        artifact_sha256=_sha256(spec.path),
        method_id=None if method_id is None else str(method_id),
        prediction_sha256=None if prediction_sha is None else str(prediction_sha),
        rows=rows,
    )


def _identity_sha256(video_ids: list[str]) -> str:
    payload = ("\n".join(sorted(video_ids)) + "\n").encode()
    return hashlib.sha256(payload).hexdigest()


def _target_rows_sha256(rows: dict[str, dict[str, Any]]) -> str:
    pairs = [[video_id, rows[video_id]["target"]] for video_id in sorted(rows)]
    payload = json.dumps(pairs, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _components(run: LoadedRun, video_ids: list[str]) -> dict[str, NDArray[np.float64]]:
    rows = [run.rows[video_id] for video_id in video_ids]
    absolute = np.asarray([row["absolute_error"] for row in rows], dtype=np.float64)
    return {
        "nmae": np.asarray([row["normalized_absolute_error"] for row in rows], dtype=np.float64),
        "mae": absolute,
        "squared_error": np.square(absolute),
        "obo": np.asarray([row["within_one"] for row in rows], dtype=np.float64),
        "exact": np.asarray([row["exact"] for row in rows], dtype=np.float64),
    }


def _metric(
    components: dict[str, NDArray[np.float64]],
    metric: str,
    indices: NDArray[np.int64] | None = None,
) -> float | NDArray[np.float64]:
    if metric == "rmse":
        values = components["squared_error"]
        return (
            float(np.sqrt(values.mean())) if indices is None else np.sqrt(values[indices].mean(1))
        )
    values = components[metric]
    return float(values.mean()) if indices is None else values[indices].mean(1)


def compare(
    *,
    candidate_spec: InputSpec,
    baseline_specs: list[InputSpec],
    source_git_sha: str,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 2026,
    expected_targets_sha256: str = DEV84_TARGETS_SHA256,
) -> dict[str, Any]:
    if not GIT_SHA_RE.fullmatch(source_git_sha):
        raise ValueError("source_git_sha must be a lowercase 40-character Git SHA")
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    if not SHA256_RE.fullmatch(expected_targets_sha256):
        raise ValueError("expected_targets_sha256 must be a lowercase SHA-256")
    candidate = _load_run(candidate_spec, expected_targets_sha256=expected_targets_sha256)
    baselines = [
        _load_run(spec, expected_targets_sha256=expected_targets_sha256) for spec in baseline_specs
    ]
    if not baselines:
        raise ValueError("at least one baseline evaluation is required")
    grouped: dict[str, list[LoadedRun]] = defaultdict(list)
    for run in baselines:
        if run.spec.group == candidate.spec.group:
            raise ValueError("baseline group must differ from candidate name")
        if any(existing.spec.run == run.spec.run for existing in grouped[run.spec.group]):
            raise ValueError(f"duplicate baseline run label {run.spec.group}/{run.spec.run}")
        grouped[run.spec.group].append(run)

    video_ids = sorted(candidate.rows)
    targets = [candidate.rows[video_id]["target"] for video_id in video_ids]
    for run in baselines:
        if sorted(run.rows) != video_ids:
            raise ValueError(f"{run.spec.group}/{run.spec.run}: dev84 identity mismatch")
        if [run.rows[video_id]["target"] for video_id in video_ids] != targets:
            raise ValueError(f"{run.spec.group}/{run.spec.run}: per-video target mismatch")
    video_id_sha256 = _identity_sha256(video_ids)
    target_rows_sha256 = _target_rows_sha256(candidate.rows)
    if video_id_sha256 != DEV84_VIDEO_ID_SHA256:
        raise ValueError("canonical dev84 video identity digest mismatch")
    if target_rows_sha256 != DEV84_TARGET_ROWS_SHA256:
        raise ValueError("canonical dev84 per-video target digest mismatch")

    candidate_components = _components(candidate, video_ids)
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, DEV84_SIZE, size=(bootstrap_samples, DEV84_SIZE), dtype=np.int64)
    comparisons: dict[str, Any] = {}
    paired_rows: list[dict[str, Any]] = [
        {
            "video_id": video_id,
            "target": candidate.rows[video_id]["target"],
            "candidate": {
                key: candidate.rows[video_id][key]
                for key in (
                    "rounded_prediction",
                    "absolute_error",
                    "normalized_absolute_error",
                    "within_one",
                    "exact",
                )
            },
            "baselines": {},
        }
        for video_id in video_ids
    ]
    for group in sorted(grouped):
        runs = sorted(grouped[group], key=lambda item: item.spec.run)
        run_components = [_components(run, video_ids) for run in runs]
        metrics: dict[str, Any] = {}
        for metric in METRICS:
            candidate_point = float(_metric(candidate_components, metric))
            baseline_point = float(
                np.mean([float(_metric(component, metric)) for component in run_components])
            )
            candidate_samples = np.asarray(
                _metric(candidate_components, metric, indices), dtype=np.float64
            )
            baseline_samples = np.mean(
                [
                    np.asarray(_metric(component, metric, indices), dtype=np.float64)
                    for component in run_components
                ],
                axis=0,
            )
            difference_samples = candidate_samples - baseline_samples
            low, high = np.quantile(difference_samples, [0.025, 0.975])
            metrics[metric] = {
                "candidate": candidate_point,
                "baseline_seed_mean": baseline_point,
                "difference_candidate_minus_baseline": candidate_point - baseline_point,
                "paired_bootstrap_95_percent": {
                    "low": float(low),
                    "high": float(high),
                    "samples": bootstrap_samples,
                    "seed": bootstrap_seed,
                },
            }
        comparisons[group] = {
            "baseline_run_total": len(runs),
            "baseline_run_labels": [run.spec.run for run in runs],
            "metrics": metrics,
        }
        for index, video_id in enumerate(video_ids):
            paired_rows[index]["baselines"][group] = [
                {
                    "run": run.spec.run,
                    **{
                        key: run.rows[video_id][key]
                        for key in (
                            "rounded_prediction",
                            "absolute_error",
                            "normalized_absolute_error",
                            "within_one",
                            "exact",
                        )
                    },
                }
                for run in runs
            ]

    def provenance(run: LoadedRun) -> dict[str, Any]:
        return {
            "group": run.spec.group,
            "run": run.spec.run,
            "artifact_file": run.spec.path.name,
            "artifact_sha256": run.artifact_sha256,
            "per_video_json_path": run.spec.row_path,
            "method_id": run.method_id,
            "prediction_sha256": run.prediction_sha256,
        }

    return {
        "schema_version": 1,
        "artifact_type": "ucfrep_dev84_paired_method_comparison",
        "protocol": "ucfrep_526",
        "split": "dev",
        "classification": "development-only frozen-prediction comparison",
        "source": {
            "git_sha": source_git_sha,
            "path": "scripts/compare_dev84_evaluations.py",
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "label_firewall": {
            "accepts_evaluation_artifacts_only": True,
            "dataset_manifest_accepted": False,
            "pose_or_video_accepted": False,
            "test105_path_rejected": True,
            "test105_accessed": False,
        },
        "paired_identity": {
            "sample_count": DEV84_SIZE,
            "video_ids_unique": True,
            "video_id_sha256": video_id_sha256,
            "target_rows_sha256": target_rows_sha256,
            "dev_targets_sha256": expected_targets_sha256,
        },
        "estimand": {
            "difference": "candidate_minus_baseline",
            "baseline_multi_run_policy": "metric per run, then arithmetic mean across runs",
            "resampling_unit": "paired video",
            "multi_run_rows_not_treated_as_independent": True,
            "confidence_level": 0.95,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "candidate": provenance(candidate),
        "baseline_runs": [provenance(run) for run in baselines],
        "comparisons": comparisons,
        "paired_rows": paired_rows,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, help="NAME=FILE::DOT.PATH")
    parser.add_argument(
        "--baseline",
        action="append",
        required=True,
        help="GROUP/RUN=FILE::DOT.PATH; repeat a GROUP for multiple seeds",
    )
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    parser.add_argument("--expected-targets-sha256", default=DEV84_TARGETS_SHA256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _reject_test_path(args.output.resolve())
    result = compare(
        candidate_spec=_parse_spec(args.candidate, candidate=True),
        baseline_specs=[_parse_spec(raw, candidate=False) for raw in args.baseline],
        source_git_sha=args.source_git_sha,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        expected_targets_sha256=args.expected_targets_sha256,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
