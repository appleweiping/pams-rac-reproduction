from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "compare_dev84_evaluations.py"
SPEC = importlib.util.spec_from_file_location("compare_dev84_evaluations", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
compare: ModuleType = MODULE
SYNTHETIC_ROWS = {f"video-{index:03d}": {"target": 10} for index in range(compare.DEV84_SIZE)}
compare.DEV84_VIDEO_ID_SHA256 = compare._identity_sha256(list(SYNTHETIC_ROWS))
compare.DEV84_TARGET_ROWS_SHA256 = compare._target_rows_sha256(SYNTHETIC_ROWS)


def _artifact(path: Path, errors: list[int], *, method: str) -> Path:
    rows = []
    for index, error in enumerate(errors):
        target = 10
        rounded = target + error
        absolute = abs(error)
        rows.append(
            {
                "video_id": f"video-{index:03d}",
                "prediction": float(rounded),
                "rounded_prediction": rounded,
                "target": target,
                "absolute_error": absolute,
                "normalized_absolute_error": absolute / target,
                "within_one": absolute <= 1,
                "exact": absolute == 0,
            }
        )
    payload = {
        "protocol": "ucfrep_526",
        "split": "dev",
        "dev_targets_sha256": compare.DEV84_TARGETS_SHA256,
        "method_id": method,
        "metrics": {"per_video": rows},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _candidate(path: Path) -> object:
    return compare._parse_spec(f"candidate={path}::metrics.per_video", candidate=True)


def _baseline(group: str, run: str, path: Path) -> object:
    return compare._parse_spec(f"{group}/{run}={path}::metrics.per_video", candidate=False)


def test_multi_seed_estimand_and_paired_bootstrap_are_deterministic(tmp_path: Path) -> None:
    candidate_path = _artifact(tmp_path / "candidate.json", [0] * 84, method="candidate")
    seed_a = _artifact(tmp_path / "seed-a.json", [1] * 84, method="baseline")
    seed_b = _artifact(tmp_path / "seed-b.json", [2] * 84, method="baseline")
    kwargs = {
        "candidate_spec": _candidate(candidate_path),
        "baseline_specs": [
            _baseline("baseline", "a", seed_a),
            _baseline("baseline", "b", seed_b),
        ],
        "source_git_sha": "a" * 40,
        "bootstrap_samples": 50,
        "bootstrap_seed": 7,
    }
    first = compare.compare(**kwargs)
    second = compare.compare(**kwargs)
    assert first == second
    metrics = first["comparisons"]["baseline"]["metrics"]
    assert metrics["nmae"]["candidate"] == 0.0
    assert metrics["nmae"]["baseline_seed_mean"] == pytest.approx(0.15)
    assert metrics["nmae"]["difference_candidate_minus_baseline"] == pytest.approx(-0.15)
    assert metrics["rmse"]["baseline_seed_mean"] == pytest.approx(1.5)
    assert metrics["rmse"]["difference_candidate_minus_baseline"] == pytest.approx(-1.5)
    assert len(first["paired_rows"]) == 84
    assert first["estimand"]["multi_run_rows_not_treated_as_independent"] is True
    assert all(len(row["baselines"]["baseline"]) == 2 for row in first["paired_rows"])


def test_identity_and_target_mismatch_fail_closed(tmp_path: Path) -> None:
    candidate_path = _artifact(tmp_path / "candidate.json", [0] * 84, method="candidate")
    baseline_path = _artifact(tmp_path / "baseline.json", [0] * 84, method="baseline")
    document = json.loads(baseline_path.read_text(encoding="utf-8"))
    document["metrics"]["per_video"][0]["video_id"] = "different"
    baseline_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="identity mismatch"):
        compare.compare(
            candidate_spec=_candidate(candidate_path),
            baseline_specs=[_baseline("baseline", "one", baseline_path)],
            source_git_sha="b" * 40,
            bootstrap_samples=5,
        )


def test_row_components_are_revalidated(tmp_path: Path) -> None:
    candidate_path = _artifact(tmp_path / "candidate.json", [0] * 84, method="candidate")
    baseline_path = _artifact(tmp_path / "baseline.json", [0] * 84, method="baseline")
    document = json.loads(baseline_path.read_text(encoding="utf-8"))
    document["metrics"]["per_video"][2]["within_one"] = False
    baseline_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="indicator"):
        compare.compare(
            candidate_spec=_candidate(candidate_path),
            baseline_specs=[_baseline("baseline", "one", baseline_path)],
            source_git_sha="c" * 40,
            bootstrap_samples=5,
        )


def test_test105_paths_are_rejected(tmp_path: Path) -> None:
    prohibited = _artifact(tmp_path / "test105-evaluation.json", [0] * 84, method="bad")
    with pytest.raises(ValueError, match="test105"):
        _candidate(prohibited)


def test_bootstrap_matches_direct_paired_resampling(tmp_path: Path) -> None:
    candidate_errors = [index % 4 for index in range(84)]
    baseline_errors = [(index + 1) % 5 for index in range(84)]
    candidate_path = _artifact(tmp_path / "candidate.json", candidate_errors, method="candidate")
    baseline_path = _artifact(tmp_path / "baseline.json", baseline_errors, method="baseline")
    result = compare.compare(
        candidate_spec=_candidate(candidate_path),
        baseline_specs=[_baseline("baseline", "one", baseline_path)],
        source_git_sha="d" * 40,
        bootstrap_samples=100,
        bootstrap_seed=2026,
    )
    rng = np.random.default_rng(2026)
    indices = rng.integers(0, 84, size=(100, 84), dtype=np.int64)
    candidate = np.asarray(candidate_errors, dtype=float) / 10.0
    baseline = np.asarray(baseline_errors, dtype=float) / 10.0
    differences = candidate[indices].mean(1) - baseline[indices].mean(1)
    low, high = np.quantile(differences, [0.025, 0.975])
    interval = result["comparisons"]["baseline"]["metrics"]["nmae"]["paired_bootstrap_95_percent"]
    assert interval["low"] == pytest.approx(low)
    assert interval["high"] == pytest.approx(high)
