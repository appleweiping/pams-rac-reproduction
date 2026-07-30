from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Literal

import numpy as np
import pytest
import torch

from pams.config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    PAMSConfig,
    PeriodConfig,
    TrainingConfig,
)
from pams.training import build_pams_model

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/server/run_pe_permutation_predev_gate.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_pe_permutation_predev_gate",
        RUNNER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_config(
    *,
    position_encoding_mode: Literal["sinusoidal", "none"] = "sinusoidal",
    consistency_weight: float = 1.0,
) -> PAMSConfig:
    return PAMSConfig(
        seed=2026,
        data=DataConfig(frames=16),
        model=ModelConfig(
            input_dim=99,
            model_dim=8,
            embedding_dim=8,
            layers=1,
            heads=2,
            feedforward_dim=16,
            dropout=0.0,
            period_head_hidden_dim=4,
            position_encoding_mode=position_encoding_mode,
        ),
        period=PeriodConfig(minimum=4, maximum=8, pose_energy_epochs=1),
        loss=LossConfig(
            scales=(0.5, 1.0, 1.5),
            temperature=0.1,
            kmeans_clusters=2,
            kmeans_refresh_epochs=1,
        ),
        training=TrainingConfig(
            epochs=1,
            effective_batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            scheduler_factor=0.5,
            scheduler_patience=1,
            minimum_learning_rate=1e-6,
            position_permutation_consistency_weight=consistency_weight,
        ),
    )


def test_parser_has_only_target_free_input_surface() -> None:
    runner = _load_runner()
    parsed = runner._parse_arguments(
        [
            "--checkpoint",
            "encoder.pt",
            "--config",
            "config.yaml",
            "--pose-cache-dir",
            "pose-cache",
            "--output",
            "predev.json",
        ]
    )

    assert set(vars(parsed)) == {
        "checkpoint",
        "config",
        "pose_cache_dir",
        "output",
        "sample_size",
        "seed",
        "device",
        "batch_size",
    }
    signature = inspect.signature(runner.run_predev_gate)
    assert set(signature.parameters) == {
        "checkpoint_path",
        "config_path",
        "pose_cache_dir",
        "sample_size",
        "seed",
        "device",
        "batch_size",
    }


def test_valid_frame_permutation_is_deterministic_independent_and_mask_safe() -> None:
    runner = _load_runner()
    valid = torch.tensor(
        [
            [True, True, False, True, True, False],
            [True, False, True, True, False, True],
        ]
    )
    first = runner._independent_valid_frame_permutations(
        valid,
        ("video-a", "video-b"),
        seed=2026,
    )
    second = runner._independent_valid_frame_permutations(
        valid,
        ("video-a", "video-b"),
        seed=2026,
    )
    reordered = runner._independent_valid_frame_permutations(
        valid.flip(0),
        ("video-b", "video-a"),
        seed=2026,
    ).flip(0)

    assert torch.equal(first, second)
    assert torch.equal(first, reordered)
    canonical = torch.arange(valid.shape[1]).expand_as(first)
    assert torch.equal(first[~valid], canonical[~valid])
    for row in range(valid.shape[0]):
        positions = canonical[row][valid[row]]
        assert torch.equal(
            first[row][valid[row]].sort().values,
            positions.sort().values,
        )


def test_synthetic_period_sequence_is_exact_normalized_and_deterministic() -> None:
    runner = _load_runner()
    first = runner._synthetic_period_sequence(8, frames=32, seed=2026)
    second = runner._synthetic_period_sequence(8, frames=32, seed=2026)

    assert np.array_equal(first.xyz, second.xyz)
    assert np.array_equal(first.valid_mask, second.valid_mask)
    assert np.allclose(first.xyz[:24], first.xyz[8:32], atol=1e-6)
    assert float(first.xyz.min()) >= 0.0
    assert float(first.xyz.max()) <= 1.0


def test_permutation_consistency_reports_valid_frame_and_video_medians() -> None:
    runner = _load_runner()
    config = _tiny_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = build_pams_model(config)
    sequences = tuple(
        runner._synthetic_period_sequence(period, frames=16, seed=2026) for period in (4, 8)
    )

    payload = runner._permutation_consistency(
        model,
        sequences,
        device=torch.device("cpu"),
        batch_size=2,
        seed=2026,
    )

    assert payload["sampled_video_total"] == 2
    assert payload["position_encoding_mode"] == "sinusoidal"
    assert payload["valid_frame_cosine"]["observations"] == 32
    assert payload["per_video_median_cosine"]["observations"] == 2
    assert -1.0 <= payload["valid_frame_cosine"]["median"] <= 1.0


def test_no_absolute_pe_permutation_gate_is_exactly_invariant() -> None:
    runner = _load_runner()
    config = _tiny_config(
        position_encoding_mode="none",
        consistency_weight=0.0,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = build_pams_model(config)
    sequences = tuple(
        runner._synthetic_period_sequence(period, frames=16, seed=2026)
        for period in (4, 8)
    )

    payload = runner._permutation_consistency(
        model,
        sequences,
        device=torch.device("cpu"),
        batch_size=2,
        seed=2026,
    )

    assert payload["position_encoding_mode"] == "none"
    assert payload["valid_frame_cosine"]["median"] == pytest.approx(
        1.0,
        abs=1e-6,
    )
    assert payload["per_video_median_cosine"]["minimum"] == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_candidate_validation_accepts_both_preregistered_modes_only() -> None:
    runner = _load_runner()

    assert runner._validated_candidate_mode(_tiny_config()) == (
        "sinusoidal",
        1.0,
    )
    assert runner._validated_candidate_mode(
        _tiny_config(
            position_encoding_mode="none",
            consistency_weight=0.0,
        )
    ) == ("none", 0.0)
    with pytest.raises(ValueError, match="either sinusoidal"):
        runner._validated_candidate_mode(
            _tiny_config(consistency_weight=0.0)
        )


def test_frozen_gate_thresholds_require_every_criterion() -> None:
    runner = _load_runner()
    passing = runner._gate_decision(
        frame_index_r2=0.10,
        zero_pose_confidence=0.10,
        random_pose_confidence=0.10,
        training_top_bin_share=0.25,
        permutation_median_cosine=0.95,
        synthetic_median_relative_error=0.10,
    )

    assert passing["overall_pass"] is True
    assert passing["dev84_prediction_authorized"] is True
    assert passing["dev84_scoring_authorized"] is False
    assert passing["test105_evaluation_authorized"] is False
    assert all(item["pass"] for item in passing["criteria"].values())

    failing = runner._gate_decision(
        frame_index_r2=None,
        zero_pose_confidence=0.100001,
        random_pose_confidence=0.10,
        training_top_bin_share=0.25,
        permutation_median_cosine=0.95,
        synthetic_median_relative_error=0.10,
    )
    assert failing["overall_pass"] is False
    assert failing["dev84_prediction_authorized"] is False
    assert failing["criteria"]["frame_index_r2"]["pass"] is False
    assert failing["criteria"]["zero_pose_period_confidence"]["pass"] is False


def test_immutable_output_rejects_overwrite(tmp_path: Path) -> None:
    runner = _load_runner()
    output = tmp_path / "predev.json"
    payload = {"finite": 1.0}
    runner._write_new_json(output, payload)

    assert output.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError):
        runner._write_new_json(output, payload)
