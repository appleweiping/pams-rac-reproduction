from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from pams.config import load_config
from pams.training import EncoderEpochStats, _progress_row
from scripts.server import run_pams_v14_dual_path_counterfactual as v14
from scripts.server import run_pams_v16_epoch11_dual_path_gate as runner


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


def _epoch_stats(epoch: int, source: str) -> EncoderEpochStats:
    return EncoderEpochStats(
        epoch=epoch,
        loss=1.0,
        learning_rate=0.0001,
        period_source=source,  # type: ignore[arg-type]
        period_confidence_mean=0.5,
        period_valid_fraction=1.0,
        optimizer_steps=1,
        clusters_refreshed=False,
        cross_cluster_requested=1,
        cross_cluster_actual=1,
        cross_cluster_shortfall=0,
        position_permutation_consistency=0.0,
    )


def _write_epoch11_pair(tmp_path: Path) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/experiments/pams_noabs_projected_teacher_v16.yaml")
    sources = ["pose"] * 10 + ["projected_pose_velocity_vector_acf"]
    history = [_epoch_stats(epoch, source) for epoch, source in enumerate(sources, 1)]
    serialized_history: list[dict[str, object]] = []
    for statistics in history:
        row = {
            field: getattr(statistics, field)
            for field in statistics.__dataclass_fields__
            if field != "position_permutation_consistency"
        }
        serialized_history.append(row)
    checkpoint = tmp_path / "encoder.pt"
    torch.save(
        {
            "schema_version": 5,
            "stage": "encoder",
            "config_fingerprint": config.fingerprint,
            "provenance": {},
            "completed_epochs": 11,
            "model_state": {},
            "optimizer_state": {},
            "scheduler_state": {},
            "history": serialized_history,
            "cluster_assignments": {},
            "prototype_bank": {},
            "rng_state": {},
        },
        checkpoint,
    )
    progress = tmp_path / "encoder.jsonl"
    progress.write_text(
        "".join(
            json.dumps(
                _progress_row(stage="encoder", stats=statistics, config=config),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for statistics in history
        ),
        encoding="utf-8",
    )
    return checkpoint, progress


def _artifact_payload() -> dict[str, object]:
    digest = "a" * 64
    return {
        "status": "encoder_epochs12_150_continuation_authorized",
        "gate": {"encoder_epochs12_150_continuation_authorized": True},
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


def test_cli_surface_is_read_only_and_has_no_training_or_label_interface() -> None:
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
            "epoch11-gate.json",
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
    assert set(inspect.signature(runner.run_epoch11_gate).parameters) == {
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
        "--resume",
    ):
        assert f'"{forbidden}"' not in parser_source
        assert f"'{forbidden}'" not in parser_source
    assert "validate_terminal_checkpoint(" not in inspect.getsource(runner.run_epoch11_gate)


def test_exact_v16_config_identity_accepts_only_frozen_file() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/experiments/pams_noabs_projected_teacher_v16.yaml"
    config = load_config(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    runner._validate_exact_v16_config(config, config_sha256=digest)
    assert digest == runner._EXPECTED_CONFIG_SHA256
    assert config.fingerprint == runner._EXPECTED_CONFIG_FINGERPRINT
    assert config.nonseed_fingerprint == runner._EXPECTED_NONSEED_FINGERPRINT
    assert config.pose_fingerprint == runner._EXPECTED_POSE_FINGERPRINT
    assert config.model.position_encoding_mode == "none"

    with pytest.raises(ValueError, match="exact frozen v16"):
        runner._validate_exact_v16_config(config, config_sha256="0" * 64)


def test_epoch11_validation_requires_exact_schedule_and_progress(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/experiments/pams_noabs_projected_teacher_v16.yaml")
    checkpoint, progress = _write_epoch11_pair(tmp_path)

    metadata = runner._checkpoint_epoch11_metadata(
        checkpoint,
        progress,
        config=config,
    )
    assert metadata["completed_epochs"] == 11
    assert metadata["period_sources"] == [
        "pose",
        "pose",
        "pose",
        "pose",
        "pose",
        "pose",
        "pose",
        "pose",
        "pose",
        "pose",
        "projected_pose_velocity_vector_acf",
    ]
    assert metadata["terminal_checkpoint_validation_called"] is False

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["history"][-1]["period_source"] = "pose"
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="ten pose epochs"):
        runner._checkpoint_epoch11_metadata(checkpoint, progress, config=config)


def test_checkpoint_and_progress_hashes_must_be_frozen_and_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identities = {
        "encoder_checkpoint": (
            "7ee1617fcac65261222a8c37c90977e92580b73c017a3d87465e794b205d7b31",
            123,
        ),
        "encoder_progress": (
            "c3782c40484916aaec97cd4b853ab6638398b00aa55ae7a3d69c2f3b862fa9ab",
            456,
        ),
    }
    runner._validate_exact_epoch11_artifact_identities(identities)

    monkeypatch.setattr(
        runner,
        "_EXPECTED_ENCODER_CHECKPOINT_SHA256",
        "__FREEZE_AFTER_V16_EPOCH11__",
    )
    with pytest.raises(RuntimeError, match="have not been frozen"):
        runner._validate_exact_epoch11_artifact_identities(identities)

    monkeypatch.setattr(
        runner,
        "_EXPECTED_ENCODER_CHECKPOINT_SHA256",
        identities["encoder_checkpoint"][0],
    )
    runner._validate_exact_epoch11_artifact_identities(identities)

    wrong = deepcopy(identities)
    wrong["encoder_progress"] = ("0" * 64, 456)
    with pytest.raises(ValueError, match="exact frozen v16 artifacts"):
        runner._validate_exact_epoch11_artifact_identities(wrong)


def test_all_ten_criteria_authorize_only_encoder_continuation() -> None:
    passing = _passing_inputs()
    decision = runner._gate_decision(**passing)

    assert len(decision["criteria"]) == 10
    assert decision["all_ten_diagnostic_criteria_pass"] is True
    assert decision["encoder_epochs12_150_continuation_authorized"] is True
    assert decision["sshead_training_authorized"] is False
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
        assert rejected["encoder_epochs12_150_continuation_authorized"] is False
        assert rejected["sshead_training_authorized"] is False


def test_gate_reuses_v14_measurements_factors_and_thresholds() -> None:
    assert runner._v14._encode_dual_paths is v14._encode_dual_paths
    assert runner._v14._distribution is v14._distribution
    assert runner._v14._time_scale_consistency is v14._time_scale_consistency
    assert runner._v14._cross_path_comparison is v14._cross_path_comparison
    assert runner._v14._TIME_SCALE_FACTORS == (0.5, 0.75)
    assert runner._v14._THRESHOLDS == v14._THRESHOLDS


def test_artifact_and_receipt_are_exclusive_and_hash_bound(tmp_path: Path) -> None:
    output = tmp_path / "epoch11-gate.json"
    payload = _artifact_payload()

    receipt_path, digest = runner._write_artifact_and_receipt(output, payload)

    artifact = output.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert digest == hashlib.sha256(artifact).hexdigest()
    assert receipt["artifact_sha256"] == digest
    assert receipt["artifact_bytes"] == len(artifact)
    assert receipt["encoder_epochs12_150_continuation_authorized"] is True
    assert receipt["sshead_training_authorized"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        runner._write_artifact_and_receipt(output, payload)
