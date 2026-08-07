from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from pams.config import load_config
from scripts.server import run_pams_v16_embedding_curve_train337_gate as runner


def _summary(median: float) -> dict[str, float]:
    return {
        "minimum": median,
        "p10": median,
        "median": median,
        "mean": median,
        "p90": median,
        "maximum": median,
    }


def _passing_inputs() -> dict[str, object]:
    return {
        "period_distribution": {
            "boundary_share": 0.24,
            "mode_share": 0.24,
        },
        "readout_distribution": {
            "action_curve_std": _summary(0.05),
            "algorithm1_majority_share": 0.60,
            "selected_count_fft_reference_absolute_gap": _summary(1.0),
        },
        "count_distribution": {
            "zero_share": 0.24,
            "mode_share": 0.24,
        },
        "time_scale": {
            "period": {
                "eligible_comparison_fraction": 0.50,
                "relative_error": _summary(0.15),
            },
            "count": {
                "exact_fraction": 0.70,
                "within_one_fraction": 0.90,
            },
        },
        "corruption_controls": {
            "time_shuffle": {
                "shuffled_to_baseline_median_ratio": 0.75,
            },
            "zero_pose": {
                "positive_period_confidence_share": 0.05,
                "curve_available_share": 0.05,
            },
        },
    }


def test_thresholds_are_exact_and_frozen_before_any_run() -> None:
    assert dict(runner._THRESHOLDS) == {
        "upstream_period_boundary_share_maximum_exclusive": 0.25,
        "upstream_period_mode_share_maximum_exclusive": 0.25,
        "upstream_period_time_scale_eligible_fraction_minimum": 0.50,
        "upstream_period_time_scale_median_relative_error_maximum": 0.15,
        "action_curve_std_median_minimum": 0.05,
        "algorithm1_majority_share_minimum": 0.60,
        "count_time_scale_exact_fraction_minimum": 0.70,
        "count_time_scale_off_by_one_fraction_minimum": 0.90,
        "selected_count_fft_gap_median_maximum": 1.0,
        "count_zero_share_maximum_exclusive": 0.25,
        "count_mode_share_maximum_exclusive": 0.25,
        "time_shuffle_harmonic_median_ratio_maximum": 0.75,
        "zero_pose_positive_period_confidence_share_maximum": 0.05,
        "zero_pose_curve_available_share_maximum": 0.05,
    }
    with pytest.raises(TypeError):
        runner._THRESHOLDS["action_curve_std_median_minimum"] = 0.0  # type: ignore[index]


def test_all_fourteen_criteria_are_required_and_test105_stays_closed() -> None:
    passing = _passing_inputs()
    decision = runner._gate_decision(**passing)

    assert len(decision["criteria"]) == 14
    assert decision["all_fourteen_criteria_pass"] is True
    assert decision["overall_pass"] is True
    assert decision["isolated_dev84_prediction_authorized"] is True
    assert decision["dev84_scoring_authorized"] is False
    assert decision["test105_evaluation_authorized"] is False

    failures: tuple[tuple[tuple[str, ...], float], ...] = (
        (("period_distribution", "boundary_share"), 0.25),
        (("period_distribution", "mode_share"), 0.25),
        (("time_scale", "period", "eligible_comparison_fraction"), 0.49),
        (("time_scale", "period", "relative_error", "median"), 0.151),
        (("readout_distribution", "action_curve_std", "median"), 0.049),
        (("readout_distribution", "algorithm1_majority_share"), 0.59),
        (("time_scale", "count", "exact_fraction"), 0.69),
        (("time_scale", "count", "within_one_fraction"), 0.89),
        (
            (
                "readout_distribution",
                "selected_count_fft_reference_absolute_gap",
                "median",
            ),
            1.01,
        ),
        (("count_distribution", "zero_share"), 0.25),
        (("count_distribution", "mode_share"), 0.25),
        (
            (
                "corruption_controls",
                "time_shuffle",
                "shuffled_to_baseline_median_ratio",
            ),
            0.751,
        ),
        (
            (
                "corruption_controls",
                "zero_pose",
                "positive_period_confidence_share",
            ),
            0.051,
        ),
        (
            ("corruption_controls", "zero_pose", "curve_available_share"),
            0.051,
        ),
    )
    for path, value in failures:
        inputs = deepcopy(passing)
        destination = inputs
        for field in path[:-1]:
            destination = destination[field]  # type: ignore[assignment,index]
        destination[path[-1]] = value  # type: ignore[index]
        rejected = runner._gate_decision(**inputs)
        assert rejected["overall_pass"] is False, path
        assert rejected["isolated_dev84_prediction_authorized"] is False
        assert rejected["dev84_scoring_authorized"] is False
        assert rejected["test105_evaluation_authorized"] is False


def test_gate_cli_is_train337_read_only_and_has_no_dev_or_test_surface() -> None:
    parsed = runner._parse_arguments(
        [
            "--encoder-checkpoint",
            "encoder.pt",
            "--encoder-progress",
            "encoder.jsonl",
            "--upstream-config",
            "v16.yaml",
            "--candidate-config",
            "embedding-curve.yaml",
            "--pose-cache-dir",
            "train337-pose",
            "--source-receipt",
            "source-export.receipt.json",
            "--output",
            "gate.json",
        ]
    )

    assert set(vars(parsed)) == {
        "encoder_checkpoint",
        "encoder_progress",
        "upstream_config",
        "candidate_config",
        "pose_cache_dir",
        "source_receipt",
        "output",
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
    run_source = inspect.getsource(runner.run_train337_gate)
    assert "validate_terminal_checkpoint(" in run_source
    assert "load_model_checkpoint(" in run_source
    assert "expected_provenance=provenance" in run_source
    assert "upstream_config" in run_source
    assert "candidate_config" in run_source


def test_exact_candidate_config_is_single_field_checkpoint_compatible() -> None:
    root = Path(__file__).resolve().parents[1]
    upstream_path = (
        root / "configs" / "experiments" / "pams_noabs_projected_teacher_v16.yaml"
    )
    candidate_path = (
        root
        / "configs"
        / "experiments"
        / "pams_noabs_projected_teacher_v16_embedding_curve_v1.yaml"
    )
    upstream = load_config(upstream_path)
    candidate = load_config(candidate_path)
    digest = hashlib.sha256(candidate_path.read_bytes()).hexdigest()

    runner._validate_candidate_config(
        upstream,
        candidate,
        candidate_config_sha256=digest,
    )
    assert digest == runner._EXPECTED_CANDIDATE_CONFIG_SHA256
    assert upstream.model == candidate.model
    assert upstream.pose_fingerprint == candidate.pose_fingerprint

    changed = candidate.model_dump()
    changed["period"]["minimum"] = 5
    with pytest.raises(ValueError, match="may differ only"):
        runner._validate_candidate_config(
            upstream,
            type(candidate).model_validate(changed),
            candidate_config_sha256=digest,
        )


def test_valid_embedding_shuffle_is_deterministic_and_preserves_dense_holes() -> None:
    values = torch.arange(48, dtype=torch.float32).reshape(1, 16, 3)
    mask = torch.ones((1, 16), dtype=torch.bool)
    mask[0, [3, 9]] = False
    values[0, ~mask[0]] = -999.0

    first = runner._deterministically_shuffle_valid_embeddings(
        values,
        mask,
        ("video-a",),
    )
    second = runner._deterministically_shuffle_valid_embeddings(
        values,
        mask,
        ("video-a",),
    )

    assert torch.equal(first, second)
    assert torch.equal(first[0, ~mask[0]], values[0, ~mask[0]])
    expected_rows = sorted(tuple(row.tolist()) for row in values[0, mask[0]])
    actual_rows = sorted(tuple(row.tolist()) for row in first[0, mask[0]])
    assert actual_rows == expected_rows
    assert not torch.equal(first[0, mask[0]], values[0, mask[0]])


def test_algorithm1_selection_never_synthesizes_a_count() -> None:
    assert runner._selection_rule(
        (5, 5, 7),
        selected_count=5,
        reference_count=6,
    ) == "majority_first"
    assert runner._selection_rule(
        (4, 6, 8),
        selected_count=6,
        reference_count=7,
    ) == "fft_nearest_fallback"
    with pytest.raises(RuntimeError, match="FFT-nearest"):
        runner._selection_rule(
            (4, 6, 8),
            selected_count=7,
            reference_count=7,
        )


def test_artifact_receipt_is_exclusive_hash_bound_and_test_closed(
    tmp_path: Path,
) -> None:
    digest = "a" * 64
    payload = {
        "status": "isolated_dev84_prediction_authorized",
        "gate": {
            "overall_pass": True,
            "isolated_dev84_prediction_authorized": True,
        },
        "inputs": {
            "encoder_checkpoint_sha256": digest,
            "encoder_progress_sha256": digest,
            "upstream_config_sha256": digest,
            "candidate_config_sha256": digest,
            "source_export_receipt_sha256": digest,
            "train337_pose_cache_set_sha256": digest,
            "candidate_algorithm_source_git_sha": "b" * 40,
            "code_files_sha256_commitment": digest,
        },
        "hardware_sha256": digest,
        "runtime_sha256": digest,
    }
    output = tmp_path / "gate.json"

    receipt_path, artifact_digest = runner._write_artifact_and_receipt(
        output,
        payload,
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert artifact_digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt["artifact_sha256"] == artifact_digest
    assert receipt["isolated_dev84_prediction_authorized"] is True
    assert receipt["dev84_scoring_authorized"] is False
    assert receipt["test105_evaluation_authorized"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        runner._write_artifact_and_receipt(output, payload)
