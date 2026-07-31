from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from pams.config import load_config
from scripts.server import run_pams_v14_dual_path_counterfactual as v14
from scripts.server import run_pams_v16_epoch11_dual_path_gate as epoch11
from scripts.server import run_pams_v16_terminal_dual_path_gate as runner


def _passing_inputs() -> dict[str, object]:
    return {
        "projected_distribution": {"boundary_share": 0.24, "mode_share": 0.24},
        "projected_time_scale": {
            "candidate_comparison_total": 10,
            "eligible_comparison_total": 5,
            "relative_error": {"median": 0.15},
        },
        "post_transformer_distribution": {
            "boundary_share": 0.24,
            "mode_share": 0.24,
        },
        "post_transformer_time_scale": {
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


def _write_schedule_checkpoint(
    path: Path,
    *,
    config_fingerprint: str,
) -> None:
    sources = ["pose"] * 10 + ["projected_pose_velocity_vector_acf"] * 140
    torch.save(
        {
            "stage": "encoder",
            "config_fingerprint": config_fingerprint,
            "completed_epochs": 150,
            "history": [
                {"epoch": epoch, "period_source": source}
                for epoch, source in enumerate(sources, start=1)
            ],
        },
        path,
    )


def _artifact_payload() -> dict[str, object]:
    digest = "a" * 64
    return {
        "status": "train337_sshead_training_authorized",
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


def test_cli_surface_is_read_only_train337_terminal_encoder_only() -> None:
    parsed = runner._parse_arguments(
        [
            "--encoder-checkpoint",
            "encoder.pt",
            "--encoder-progress",
            "encoder.jsonl",
            "--config",
            "v16.yaml",
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
        "--label",
        "--dev",
        "--test",
        "--train",
        "--sshead",
        "--resume",
    ):
        assert f'"{forbidden}"' not in parser_source
        assert f"'{forbidden}'" not in parser_source
    assert "validate_terminal_checkpoint(" in inspect.getsource(runner.run_terminal_gate)


def test_exact_v16_config_identity_is_shared_with_epoch11_gate() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/experiments/pams_noabs_projected_teacher_v16.yaml"
    config = load_config(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    runner._validate_exact_v16_config(config, config_sha256=digest)
    assert runner._EXPECTED_CONFIG_SHA256 == epoch11._EXPECTED_CONFIG_SHA256
    assert runner._EXPECTED_CONFIG_FINGERPRINT == (epoch11._EXPECTED_CONFIG_FINGERPRINT)
    assert runner._EXPECTED_NONSEED_FINGERPRINT == (epoch11._EXPECTED_NONSEED_FINGERPRINT)
    assert runner._EXPECTED_POSE_FINGERPRINT == epoch11._EXPECTED_POSE_FINGERPRINT
    assert digest == runner._EXPECTED_CONFIG_SHA256
    assert config.training.epochs == 150
    assert config.period.pose_energy_epochs == 10
    assert config.model.position_encoding_mode == "none"
    assert config.period.post_warmup_source == ("projected_pose_velocity_vector_acf")

    with pytest.raises(ValueError, match="exact frozen v16"):
        runner._validate_exact_v16_config(config, config_sha256="0" * 64)


def test_terminal_schedule_requires_150_epochs_and_exact_teacher_sources(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/experiments/pams_noabs_projected_teacher_v16.yaml")
    checkpoint = tmp_path / "encoder.pt"
    _write_schedule_checkpoint(
        checkpoint,
        config_fingerprint=config.fingerprint,
    )

    metadata = runner._terminal_schedule_metadata(checkpoint, config=config)

    assert metadata["completed_epochs"] == 150
    assert metadata["pose_proxy_epoch_total"] == 10
    assert metadata["projected_teacher_epoch_total"] == 140
    assert metadata["first_projected_teacher_epoch"] == 11
    assert metadata["last_projected_teacher_epoch"] == 150
    assert metadata["terminal_checkpoint_validation_called"] is True

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["history"][10]["period_source"] = "pose"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="ten pose epochs"):
        runner._terminal_schedule_metadata(checkpoint, config=config)


def test_checkpoint_and_progress_hash_sentinels_fail_closed_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = {
        "encoder_checkpoint": (
            "491fd5df67fb4d1fc7669019e4f78fac7b5c681dfa94a4cc7a469da2f73f5196",
            123,
        ),
        "encoder_progress": (
            "047c6af919843cd5d1133033ae07314b8ddd9426d878e4b6519181b94eb13ec7",
            456,
        ),
    }
    assert identities["encoder_checkpoint"][0] == runner._EXPECTED_ENCODER_CHECKPOINT_SHA256
    assert identities["encoder_progress"][0] == runner._EXPECTED_ENCODER_PROGRESS_SHA256
    runner._validate_exact_terminal_artifact_identities(identities)

    with monkeypatch.context() as sentinel:
        sentinel.setattr(
            runner,
            "_EXPECTED_ENCODER_CHECKPOINT_SHA256",
            "__FREEZE_AFTER_V16_EPOCH150__",
        )
        sentinel.setattr(
            runner,
            "_EXPECTED_ENCODER_PROGRESS_SHA256",
            "__FREEZE_AFTER_V16_EPOCH150__",
        )
        with pytest.raises(RuntimeError, match="have not been frozen"):
            runner._validate_exact_terminal_artifact_identities(identities)

    wrong = deepcopy(identities)
    wrong["encoder_checkpoint"] = ("0" * 64, 123)
    with pytest.raises(ValueError, match="exact frozen v16 artifacts"):
        runner._validate_exact_terminal_artifact_identities(wrong)

    run_source = inspect.getsource(runner.run_terminal_gate)
    identity_position = run_source.index("_validate_exact_terminal_artifact_identities")
    checkpoint_load_positions = [
        position
        for token in (
            "_peek_checkpoint(",
            "validate_terminal_checkpoint(",
            "_terminal_schedule_metadata(",
            "load_model_checkpoint(",
        )
        if (position := run_source.index(token)) >= 0
    ]
    assert identity_position < min(checkpoint_load_positions)


def test_all_ten_criteria_authorize_only_train337_sshead() -> None:
    passing = _passing_inputs()
    decision = runner._gate_decision(**passing)

    assert len(decision["criteria"]) == 10
    assert decision["all_ten_diagnostic_criteria_pass"] is True
    assert decision["sshead_training_authorized"] is True
    assert decision["authorized_scope"] == {
        "protocol": "ucfrep_526",
        "seed": 2026,
        "training_video_total": 337,
        "stage": "sshead",
        "encoder_state": "frozen_exact_terminal_v16",
        "maximum_training_epochs": 30,
        "label_or_target_input": False,
    }
    assert decision["encoder_training_authorized"] is False
    assert decision["dev84_prediction_authorized"] is False
    assert decision["dev84_scoring_authorized"] is False
    assert decision["test105_evaluation_authorized"] is False

    failures: tuple[tuple[tuple[str, ...], object], ...] = (
        (("projected_distribution", "boundary_share"), 0.25),
        (("projected_distribution", "mode_share"), 0.25),
        (("projected_time_scale", "eligible_comparison_total"), 4),
        (("projected_time_scale", "relative_error", "median"), 0.151),
        (("post_transformer_distribution", "boundary_share"), 0.25),
        (("post_transformer_distribution", "mode_share"), 0.25),
        (("post_transformer_time_scale", "eligible_comparison_total"), 4),
        (("post_transformer_time_scale", "relative_error", "median"), 0.151),
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
        assert rejected["authorized_scope"] is None
        assert rejected["dev84_prediction_authorized"] is False
        assert rejected["dev84_scoring_authorized"] is False
        assert rejected["test105_evaluation_authorized"] is False


def test_terminal_gate_reuses_v14_measurements_factors_and_thresholds() -> None:
    assert runner._v14._encode_dual_paths is v14._encode_dual_paths
    assert runner._v14._distribution is v14._distribution
    assert runner._v14._time_scale_consistency is v14._time_scale_consistency
    assert runner._v14._cross_path_comparison is v14._cross_path_comparison
    assert runner._v14._TIME_SCALE_FACTORS == (0.5, 0.75)
    assert runner._v14._THRESHOLDS == v14._THRESHOLDS


def test_artifact_and_receipt_are_exclusive_hash_bound_and_scope_closed(
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
    assert receipt["encoder_training_authorized"] is False
    assert receipt["dev84_prediction_authorized"] is False
    assert receipt["dev84_scoring_authorized"] is False
    assert receipt["test105_evaluation_authorized"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        runner._write_artifact_and_receipt(output, payload)
