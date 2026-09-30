from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.server import run_pams_v14_dual_path_counterfactual as runner


def _sample(
    video_id: str,
    period: float,
    confidence: float = 1.0,
) -> runner._PeriodSample:
    return runner._PeriodSample(
        video_id=video_id,
        period_frames=period,
        confidence=confidence,
    )


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
        "status": "projected_teacher_passed",
        "gate": {"v15_encoder_training_authorized": True},
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
            "v14.yaml",
            "--pose-cache-dir",
            "train337-pose",
            "--source-receipt",
            "source-export.receipt.json",
            "--output",
            "counterfactual.json",
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
    assert set(inspect.signature(runner.run_counterfactual).parameters) == {
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


def test_cross_path_comparison_uses_only_common_positive_confidence() -> None:
    result = runner._cross_path_comparison(
        (
            _sample("a", 10.0),
            _sample("b", 20.0),
            _sample("c", 30.0, confidence=0.0),
        ),
        (
            _sample("a", 12.0),
            _sample("b", 10.0, confidence=0.0),
            _sample("c", 33.0),
        ),
    )

    assert result["candidate_comparison_total"] == 3
    assert result["eligible_comparison_total"] == 1
    assert result["eligible_comparison_fraction"] == pytest.approx(1 / 3)
    assert result["relative_error"]["median"] == pytest.approx(0.2)
    assert result["rows"][0]["relative_error"] == pytest.approx(0.2)
    assert result["rows"][1]["relative_error"] is None


def test_dual_paths_call_distinct_estimators_without_swapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = torch.full((2, 8, 4), 11.0)
    projected = torch.full((2, 8, 4), 22.0)
    valid = torch.ones((2, 8), dtype=torch.bool)
    calls: list[tuple[str, torch.Tensor]] = []

    def post_estimator(
        features: torch.Tensor,
        minimum: int,
        maximum: int,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (minimum, maximum) == (4, 128)
        assert valid_mask is valid
        calls.append(("post_pe", features))
        return torch.tensor([10.0, 11.0]), torch.tensor([0.1, 0.2])

    def projected_estimator(
        features: torch.Tensor,
        minimum: int,
        maximum: int,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert (minimum, maximum) == (4, 128)
        assert valid_mask is valid
        calls.append(("projected_pre_pe", features))
        return torch.tensor([20.0, 21.0]), torch.tensor([0.3, 0.4])

    monkeypatch.setattr(
        runner,
        "estimate_period_from_embedding_velocity_vectors",
        post_estimator,
    )
    monkeypatch.setattr(
        runner,
        "estimate_period_from_projected_pose",
        projected_estimator,
    )
    result = runner._estimate_dual_features(
        embeddings,
        projected,
        valid,
        config=SimpleNamespace(
            period=SimpleNamespace(minimum=4, maximum=128),
        ),
    )

    assert calls == [("post_pe", embeddings), ("projected_pre_pe", projected)]
    assert result["post_pe"][0].tolist() == [10.0, 11.0]
    assert result["projected_pre_pe"][0].tolist() == [20.0, 21.0]


def test_distribution_reports_boundary_and_mode_without_targets() -> None:
    rows = (
        _sample("a", 4.0),
        _sample("b", 8.0),
        _sample("c", 8.0),
        _sample("d", 128.0, confidence=0.0),
    )
    result = runner._distribution(rows, minimum=4, maximum=128)

    assert result["record_total"] == 4
    assert result["boundary_share"] == 0.5
    assert result["mode_period_frames"] == 8.0
    assert result["mode_share"] == 0.5
    assert result["zero_confidence_share"] == 0.25


def test_all_ten_criteria_are_reported_but_only_projected_four_authorize_v15() -> None:
    passing = _passing_inputs()
    decision = runner._gate_decision(**passing)

    assert len(decision["criteria"]) == 10
    assert decision["all_ten_diagnostic_criteria_pass"] is True
    assert decision["v15_encoder_training_authorized"] is True
    assert decision["dev84_prediction_authorized"] is False
    assert decision["dev84_scoring_authorized"] is False
    assert decision["test105_evaluation_authorized"] is False

    causal_baseline_failure = deepcopy(passing)
    causal_baseline_failure["post_pe_distribution"]["mode_share"] = 0.95
    causal_baseline_failure["cross_path"]["relative_error"]["median"] = 0.9
    diagnostic = runner._gate_decision(**causal_baseline_failure)
    assert diagnostic["all_ten_diagnostic_criteria_pass"] is False
    assert diagnostic["criteria"]["post_pe_mode_share"]["pass"] is False
    assert diagnostic["criteria"]["cross_path_median_relative_error"]["pass"] is False
    assert diagnostic["v15_encoder_training_authorized"] is True

    projected_failure = deepcopy(passing)
    projected_failure["projected_distribution"]["mode_share"] = 0.25
    rejected = runner._gate_decision(**projected_failure)
    assert rejected["criteria"]["projected_mode_share"]["pass"] is False
    assert rejected["v15_encoder_training_authorized"] is False


def test_each_projected_teacher_criterion_is_required_for_v15_authorization() -> None:
    failures: tuple[tuple[tuple[str, ...], object], ...] = (
        (("projected_distribution", "boundary_share"), 0.25),
        (("projected_distribution", "mode_share"), 0.25),
        (("projected_time_scale", "eligible_comparison_total"), 4),
        (("projected_time_scale", "relative_error", "median"), 0.151),
    )
    for path, value in failures:
        inputs = deepcopy(_passing_inputs())
        destination = inputs
        for field in path[:-1]:
            destination = destination[field]  # type: ignore[assignment,index]
        destination[path[-1]] = value  # type: ignore[index]
        assert runner._gate_decision(**inputs)["v15_encoder_training_authorized"] is False


def test_exact_v14_identity_rejects_semantically_similar_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Config:
        fingerprint = runner._EXPECTED_CONFIG_FINGERPRINT
        pose_fingerprint = runner._EXPECTED_POSE_FINGERPRINT

    monkeypatch.setattr(runner, "_validate_v14_config", lambda config: None)
    runner._validate_exact_v14_config(
        Config(),  # type: ignore[arg-type]
        config_sha256=runner._EXPECTED_CONFIG_SHA256,
    )
    with pytest.raises(ValueError, match="exact frozen v14"):
        runner._validate_exact_v14_config(
            Config(),  # type: ignore[arg-type]
            config_sha256="0" * 64,
        )


def test_source_receipt_argument_must_match_runtime_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = tmp_path / "source.receipt.json"
    receipt.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    monkeypatch.setenv("PAMS_SOURCE_EXPORT_RECEIPT", str(receipt))
    monkeypatch.setenv("PAMS_SOURCE_EXPORT_RECEIPT_SHA256", digest)

    runner._validate_source_receipt_argument(receipt, receipt_sha256=digest)
    with pytest.raises(RuntimeError, match="bytes differ"):
        runner._validate_source_receipt_argument(
            receipt,
            receipt_sha256="0" * 64,
        )


def test_artifact_and_receipt_are_exclusive_and_hash_bound(
    tmp_path: Path,
) -> None:
    output = tmp_path / "counterfactual.json"
    payload = _artifact_payload()

    receipt_path, digest = runner._write_artifact_and_receipt(output, payload)

    artifact = output.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert digest == hashlib.sha256(artifact).hexdigest()
    assert receipt["artifact_sha256"] == digest
    assert receipt["artifact_bytes"] == len(artifact)
    assert receipt["source_export_receipt_sha256"] == "a" * 64
    assert receipt["v15_encoder_training_authorized"] is True
    with pytest.raises(FileExistsError, match="overwrite"):
        runner._write_artifact_and_receipt(output, payload)


def test_dual_path_names_and_frozen_factors_are_explicit() -> None:
    source = inspect.getsource(runner._encode_dual_paths)
    assert "forward_with_pre_pe" in source
    estimator_source = inspect.getsource(runner._estimate_dual_features)
    assert "estimate_period_from_embedding_velocity_vectors(" in estimator_source
    assert "estimate_period_from_projected_pose(" in estimator_source
    assert runner._TIME_SCALE_FACTORS == (0.50, 0.75)
    assert np.isclose(
        runner._THRESHOLDS["cross_path_median_relative_error_maximum"],
        0.25,
    )
