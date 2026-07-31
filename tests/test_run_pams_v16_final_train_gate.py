from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest

from pams.config import load_config
from scripts.server import run_pams_v14_final_train_gate as v14
from scripts.server import run_pams_v16_final_train_gate as runner


def _passing_inputs() -> dict[str, object]:
    return {
        "encoder_distribution": {"boundary_share": 0.24, "mode_share": 0.24},
        "encoder_time_scale": {
            "candidate_comparison_total": 10,
            "eligible_comparison_total": 5,
            "relative_error": {"median": 0.15},
        },
        "sshead_distribution": {
            "boundary_share": 0.24,
            "mode_share": 0.24,
            "period_stream_std": {"median": 0.05},
        },
        "sshead_correlation": {
            "eligible_pair_fraction": 0.90,
            "absolute_correlation": {"median": 0.89},
        },
        "sshead_time_scale": {
            "candidate_comparison_total": 10,
            "eligible_comparison_total": 5,
            "relative_error": {"median": 0.15},
        },
        "sshead_history": {"zero_grad_steps_total": 0},
        "head_parameter_change": {"delta_l2": 0.1},
    }


def _artifact_payload() -> dict[str, object]:
    digest = "a" * 64
    return {
        "status": "isolated_dev84_prediction_authorized",
        "gate": {
            "overall_pass": True,
            "isolated_dev84_prediction_authorized": True,
        },
        "inputs": {
            "encoder_checkpoint_sha256": digest,
            "encoder_progress_sha256": digest,
            "sshead_checkpoint_sha256": digest,
            "sshead_progress_sha256": digest,
            "config_sha256": digest,
            "train337_pose_cache_set_sha256": digest,
            "checkpoint_algorithm_source_git_sha": "b" * 40,
            "gate_code_source_git_sha": "c" * 40,
            "code_files_sha256_commitment": digest,
        },
        "hardware_sha256": digest,
        "runtime_sha256": digest,
    }


def test_cli_has_only_train337_artifacts_and_no_label_or_dev_test_input() -> None:
    parsed = runner._parse_arguments(
        [
            "--encoder-checkpoint",
            "encoder.pt",
            "--encoder-progress",
            "encoder.jsonl",
            "--sshead-checkpoint",
            "sshead.pt",
            "--sshead-progress",
            "sshead.jsonl",
            "--config",
            "v16.yaml",
            "--pose-cache-dir",
            "train337-pose",
            "--output",
            "final-gate.json",
        ]
    )
    assert set(vars(parsed)) == {
        "encoder_checkpoint",
        "encoder_progress",
        "sshead_checkpoint",
        "sshead_progress",
        "config",
        "pose_cache_dir",
        "output",
        "device",
        "batch_size",
    }
    assert set(inspect.signature(runner.run_final_gate).parameters) == {
        "encoder_checkpoint_path",
        "encoder_progress_path",
        "sshead_checkpoint_path",
        "sshead_progress_path",
        "config_path",
        "pose_cache_dir",
        "device",
        "batch_size",
    }
    source = inspect.getsource(runner._parse_arguments)
    for forbidden in (
        "--manifest",
        "--source-receipt",
        "--targets",
        "--action",
        "--count",
        "--label",
        "--dev",
        "--test",
    ):
        assert f'"{forbidden}"' not in source
        assert f"'{forbidden}'" not in source


def test_exact_v16_config_and_all_four_checkpoint_artifacts_are_bound() -> None:
    root = Path(__file__).resolve().parents[1]
    config_path = root / "configs/experiments/pams_noabs_projected_teacher_v16.yaml"
    config = load_config(config_path)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    runner._validate_exact_v16_config(config, config_sha256=config_sha256)
    assert config.model.position_encoding_mode == "none"
    assert config.training.epochs == 150
    assert config.sshead.epochs == 30

    identities = {
        "encoder_checkpoint": (
            "491fd5df67fb4d1fc7669019e4f78fac7b5c681dfa94a4cc7a469da2f73f5196",
            1,
        ),
        "encoder_progress": (
            "047c6af919843cd5d1133033ae07314b8ddd9426d878e4b6519181b94eb13ec7",
            2,
        ),
        "sshead_checkpoint": (
            "bad56ddf7b53561733cfa3d4786df6ba8d286e9b02cd656335bdc212b9fc7b74",
            3,
        ),
        "sshead_progress": (
            "0e820d03b6b4d86b2341074bbcb0d9793839a2bf2bbb4388f44343e397cd5401",
            4,
        ),
        "config": (config_sha256, 5),
    }
    runner._validate_exact_inputs(identities)
    wrong = deepcopy(identities)
    wrong["sshead_checkpoint"] = ("0" * 64, 3)
    with pytest.raises(ValueError, match="exact frozen train-only"):
        runner._validate_exact_inputs(wrong)


def test_all_thirteen_unchanged_criteria_are_required_for_prediction_only() -> None:
    passing = _passing_inputs()
    decision = runner._gate_decision(**passing)

    assert len(decision["criteria"]) == 13
    assert decision["overall_pass"] is True
    assert decision["isolated_dev84_prediction_authorized"] is True
    assert decision["dev84_prediction_authorized"] is True
    assert decision["dev84_scoring_authorized"] is False
    assert decision["test105_evaluation_authorized"] is False

    failures: tuple[tuple[tuple[str, ...], object], ...] = (
        (("encoder_distribution", "boundary_share"), 0.25),
        (("encoder_distribution", "mode_share"), 0.25),
        (("encoder_time_scale", "eligible_comparison_total"), 4),
        (("encoder_time_scale", "relative_error", "median"), 0.151),
        (("sshead_distribution", "period_stream_std", "median"), 0.049),
        (("sshead_distribution", "boundary_share"), 0.25),
        (("sshead_distribution", "mode_share"), 0.25),
        (("sshead_correlation", "eligible_pair_fraction"), 0.899),
        (("sshead_correlation", "absolute_correlation", "median"), 0.90),
        (("sshead_time_scale", "eligible_comparison_total"), 4),
        (("sshead_time_scale", "relative_error", "median"), 0.151),
        (("sshead_history", "zero_grad_steps_total"), 1),
        (("head_parameter_change", "delta_l2"), 0.0),
    )
    for path, value in failures:
        inputs = deepcopy(passing)
        destination = inputs
        for field in path[:-1]:
            destination = destination[field]  # type: ignore[assignment,index]
        destination[path[-1]] = value  # type: ignore[index]
        rejected = runner._gate_decision(**inputs)
        assert rejected["overall_pass"] is False
        assert rejected["isolated_dev84_prediction_authorized"] is False
        assert rejected["dev84_scoring_authorized"] is False
        assert rejected["test105_evaluation_authorized"] is False


def test_scientific_functions_and_thirteen_thresholds_are_exact_v14_reuse() -> None:
    assert runner._THRESHOLDS is v14._THRESHOLDS
    assert runner._v14._encode_training_sequences is v14._encode_training_sequences
    assert runner._v14._training_distribution is v14._training_distribution
    assert runner._v14._time_scale_consistency is v14._time_scale_consistency
    assert runner._v14._encode_head_sequences is v14._encode_head_sequences
    assert runner._v14._centered_cross_video_correlation is (v14._centered_cross_video_correlation)
    assert runner._v14._head_time_scale_consistency is (v14._head_time_scale_consistency)
    assert runner._v14._head_parameter_change is v14._head_parameter_change
    assert runner._v14._sshead_history_audit is v14._sshead_history_audit


def test_artifact_receipt_is_exclusive_hash_bound_and_scope_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "final-gate.json"
    payload = _artifact_payload()

    receipt_path, digest = runner._write_artifact_and_receipt(output, payload)

    artifact = output.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert digest == hashlib.sha256(artifact).hexdigest()
    assert receipt["artifact_sha256"] == digest
    assert receipt["artifact_bytes"] == len(artifact)
    assert receipt["isolated_dev84_prediction_authorized"] is True
    assert receipt["dev84_scoring_authorized"] is False
    assert receipt["test105_evaluation_authorized"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        runner._write_artifact_and_receipt(output, payload)
