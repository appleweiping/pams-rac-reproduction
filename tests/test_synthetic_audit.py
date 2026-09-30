from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pams.cli import app
from pams.config import load_config
from pams.synthetic_audit import (
    _pose_suite,
    build_synthetic_audit_cases,
    load_synthetic_audit_policy,
    run_synthetic_acceptance,
)


def _root() -> Path:
    return Path(__file__).parents[1]


def test_policy_expands_exact_config_derived_55_case_matrix() -> None:
    policy = load_synthetic_audit_policy(_root() / "configs/stress.yaml")
    cases = build_synthetic_audit_cases(policy)

    assert policy.expected_case_count == 55
    assert len(cases) == 55
    assert [case.spec.count for case in cases[:39]] == [
        float(value) for value in range(2, 41)
    ]
    assert {
        case.category: sum(item.category == case.category for item in cases)
        for case in cases
    } == {
        "count_sweep": 39,
        "constant_speed": 3,
        "linear_speed": 2,
        "pause": 2,
        "gaussian_noise": 4,
        "missing_joints": 1,
        "second_harmonic": 4,
    }
    descending = next(
        case
        for case in cases
        if case.configured_parameters == {"start": 2.0, "stop": 0.5}
    )
    assert descending.spec.speed_range == (2.0, 0.5)
    harmonic = next(
        case
        for case in cases
        if case.configured_parameters.get("second_harmonic_amplitude") == 1.0
    )
    assert harmonic.spec.harmonics == (1.0, 2.0)


def test_pose_suite_records_every_input_hash_and_category_metric() -> None:
    root = _root()
    policy = load_synthetic_audit_policy(root / "configs/stress.yaml")
    config = load_config(root / "configs/pams.yaml")
    report = _pose_suite(policy, config)

    assert len(report["cases"]) == 55
    assert report["metrics"]["sample_count"] == 55
    assert set(report["category_metrics"]) == {
        "count_sweep",
        "constant_speed",
        "linear_speed",
        "pause",
        "gaussian_noise",
        "missing_joints",
        "second_harmonic",
    }
    assert all(len(row["input_sha256"]) == 64 for row in report["cases"])
    assert all(
        set(row["input_component_sha256"])
        == {"xyz", "valid_mask", "phase_radians", "joint_visibility"}
        for row in report["cases"]
    )
    assert report["metrics"]["max_generation_cycle_error"] <= 1e-12
    assert report["thresholds"] == {
        "max_absolute_error": 1.0,
        "obo": 1.0,
        "nmae": 0.08,
    }


def test_acceptance_writes_bound_artifact_and_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pams.synthetic_audit as audit_module

    monkeypatch.setattr(
        audit_module,
        "clean_git_revision",
        lambda repository: "a" * 40,
    )
    monkeypatch.setattr(
        audit_module,
        "hardware_fingerprint",
        lambda: {"platform": "test"},
    )
    root = _root()
    result = run_synthetic_acceptance(
        output_dir=tmp_path / "run",
        config_path=root / "configs/pams.yaml",
        stress_config_path=root / "configs/stress.yaml",
        repository_root=root,
        command=["pams", "synthetic", "acceptance"],
    )

    artifact_path = Path(result["artifact_path"])
    receipt_path = Path(result["receipt_path"])
    artifact_bytes = artifact_path.read_bytes()
    artifact = json.loads(artifact_bytes)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(artifact_bytes).hexdigest() == result["artifact_sha256"]
    assert receipt["artifact_sha256"] == result["artifact_sha256"]
    assert receipt["artifact_bytes"] == len(artifact_bytes)
    assert artifact["sealed_test_accessed"] is False
    assert artifact["counter_576"]["metrics"]["case_count"] == 576
    assert artifact["period_counter_576"]["metrics"]["case_count"] == 576
    assert artifact["pose_suite"]["metrics"]["sample_count"] == 55


def test_synthetic_acceptance_cli_propagates_gate_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import pams.synthetic_audit as audit_module

    observed: dict[str, object] = {}

    def fake_run(**kwargs):
        observed.update(kwargs)
        return {
            "classification": "synthetic component acceptance diagnostic",
            "passed": False,
        }

    monkeypatch.setattr(audit_module, "run_synthetic_acceptance", fake_run)
    root = _root()
    result = CliRunner().invoke(
        app,
        [
            "synthetic",
            "acceptance",
            str(tmp_path / "run"),
            "--config",
            str(root / "configs/pams.yaml"),
            "--stress-config",
            str(root / "configs/stress.yaml"),
        ],
    )
    assert result.exit_code == 1
    assert json.loads(result.stdout)["passed"] is False
    assert observed["output_dir"] == tmp_path / "run"
    assert observed["config_path"] == root / "configs/pams.yaml"
    assert observed["stress_config_path"] == root / "configs/stress.yaml"
