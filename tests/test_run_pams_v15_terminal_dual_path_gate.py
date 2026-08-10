from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest

from pams.config import load_config
from scripts.server import run_pams_v14_dual_path_counterfactual as v14
from scripts.server import run_pams_v15_terminal_dual_path_gate as runner


def _passing_inputs() -> dict[str, object]:
    return {
        "projected_distribution": {
            "boundary_share": 0.24,
            "mode_share": 0.24,
        },
        "projected_time_scale": {
            "candidate_comparison_total": 10,
            "eligible_comparison_total": 5,
            "relative_error": {"median": 0.15},
        },
        "post_pe_distribution": {
            "boundary_share": 0.24,
            "mode_share": 0.24,
        },
        "post_pe_time_scale": {
            "candidate_comparison_total": 10,
            "eligible_comparison_total": 5,
            "relative_error": {"median": 0.15},
        },
        "cross_path": {
            "candidate_comparison_total": 10,
            "eligible_comparison_total": 5,
            "relative_error": {"median": 0.25},
        },
    }


def _artifact_payload() -> dict[str, object]:
    digest = "a" * 64
    return {
        "status": "sshead_training_authorized",
        "gate": {"sshead_training_authorized": True},
        "inputs": {
            "encoder_checkpoint_sha256": digest,
            "encoder_progress_sha256": digest,
            "config_sha256": digest,
            "source_export_receipt_sha256": digest,
            "train337_pose_cache_set_sha256": digest,
            "checkpoint_algorithm_source_git_sha": "b" * 40,
            "gate_code_source_git_sha": "c" * 40,
            "code_files_sha256_commitment": digest,
        },
        "hardware_sha256": digest,
        "runtime_sha256": digest,
    }


def test_cli_surface_is_read_only_train337_encoder_only() -> None:
    parsed = runner._parse_arguments(
        [
            "--encoder-checkpoint",
            "encoder.pt",
            "--encoder-progress",
            "encoder.jsonl",
            "--config",
            "v15.yaml",
            "--pose-cache-dir",
            "train337-pose",
            "--source-receipt",
            "source-export.receipt.json",
            "--output",
            "terminal-gate.json",
        ]
    )

    assert set(vars(parsed)) == {
        "encoder_checkpoint",
        "encoder_progress",
        "config",
        "pose_cache_dir",
        "source_receipt",
        "output",
        "device",
        "batch_size",
    }
    assert set(inspect.signature(runner.run_terminal_gate).parameters) == {
        "encoder_checkpoint_path",
        "encoder_progress_path",
        "config_path",
        "pose_cache_dir",
        "source_receipt_path",
        "device",
        "batch_size",
    }
    parser_source = inspect.getsource(runner._parse_arguments)
    for forbidden in (
        "--manifest",
        "--targets",
        "--action",
        "--count",
        "--dev",
        "--test",
        "--train",
        "--sshead",
    ):
        assert f'"{forbidden}"' not in parser_source
        assert f"'{forbidden}'" not in parser_source


def test_exact_v15_config_identity_accepts_only_frozen_file() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/experiments/pams_projected_teacher_v15.yaml"
    config = load_config(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    runner._validate_exact_v15_config(config, config_sha256=digest)
    assert digest == runner._EXPECTED_CONFIG_SHA256
    assert config.fingerprint == runner._EXPECTED_CONFIG_FINGERPRINT
    assert config.pose_fingerprint == runner._EXPECTED_POSE_FINGERPRINT

    with pytest.raises(ValueError, match="exact frozen v15"):
        runner._validate_exact_v15_config(
            config,
            config_sha256="0" * 64,
        )


def test_all_ten_criteria_are_required_for_sshead_training() -> None:
    passing = _passing_inputs()
    decision = runner._gate_decision(**passing)

    assert len(decision["criteria"]) == 10
    assert decision["all_ten_diagnostic_criteria_pass"] is True
    assert decision["sshead_training_authorized"] is True
    assert decision["dev84_prediction_authorized"] is False
    assert decision["dev84_scoring_authorized"] is False
    assert decision["test105_evaluation_authorized"] is False

    failures: tuple[tuple[tuple[str, ...], object], ...] = (
        (("projected_distribution", "boundary_share"), 0.25),
        (("projected_distribution", "mode_share"), 0.25),
        (("projected_time_scale", "eligible_comparison_total"), 4),
        (("projected_time_scale", "relative_error", "median"), 0.151),
        (("post_pe_distribution", "boundary_share"), 0.25),
        (("post_pe_distribution", "mode_share"), 0.25),
        (("post_pe_time_scale", "eligible_comparison_total"), 4),
        (("post_pe_time_scale", "relative_error", "median"), 0.151),
        (("cross_path", "eligible_comparison_total"), 4),
        (("cross_path", "relative_error", "median"), 0.251),
    )
    for path, value in failures:
        inputs = deepcopy(passing)
        destination = inputs
        for field in path[:-1]:
            destination = destination[field]  # type: ignore[assignment,index]
        destination[path[-1]] = value  # type: ignore[index]
        rejected = runner._gate_decision(**inputs)
        assert rejected["sshead_training_authorized"] is False
        assert rejected["all_ten_diagnostic_criteria_pass"] is False


def test_gate_reuses_v14_measurements_and_thresholds_without_mutation() -> None:
    assert runner._v14._encode_dual_paths is v14._encode_dual_paths
    assert runner._v14._distribution is v14._distribution
    assert runner._v14._time_scale_consistency is v14._time_scale_consistency
    assert runner._v14._cross_path_comparison is v14._cross_path_comparison
    assert runner._v14._THRESHOLDS == v14._THRESHOLDS

    v14_decision = v14._gate_decision(**_passing_inputs())
    assert v14_decision["v15_encoder_training_authorized"] is True
    assert "sshead_training_authorized" not in v14_decision


def test_artifact_and_receipt_are_exclusive_and_hash_bound(
    tmp_path: Path,
) -> None:
    output = tmp_path / "terminal-gate.json"
    payload = _artifact_payload()

    receipt_path, digest = runner._write_artifact_and_receipt(output, payload)

    artifact = output.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert digest == hashlib.sha256(artifact).hexdigest()
    assert receipt["artifact_sha256"] == digest
    assert receipt["artifact_bytes"] == len(artifact)
    assert receipt["sshead_training_authorized"] is True
    assert receipt["source_export_receipt_sha256"] == "a" * 64
    with pytest.raises(FileExistsError, match="overwrite"):
        runner._write_artifact_and_receipt(output, payload)
