from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

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
from pams.period import (
    embedding_velocity_harmonic_fundamental_diagnostics,
    estimate_harmonic_fundamental_from_embeddings,
    estimate_period_from_embeddings,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/server/run_noabspe_harmonic_predev_gate.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_noabspe_harmonic_predev_gate",
        RUNNER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _noabs_config() -> PAMSConfig:
    return PAMSConfig(
        seed=2026,
        data=DataConfig(frames=256),
        model=ModelConfig(
            input_dim=99,
            model_dim=8,
            embedding_dim=8,
            layers=1,
            heads=2,
            feedforward_dim=16,
            dropout=0.0,
            period_head_hidden_dim=4,
            position_encoding_mode="none",
        ),
        period=PeriodConfig(
            minimum=4,
            maximum=128,
            pose_energy_epochs=1,
        ),
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
            position_permutation_consistency_weight=0.0,
        ),
    )


def _harmonic_embedding(
    *,
    period: int,
    dominant_order: int,
    frames: int = 256,
) -> torch.Tensor:
    time = torch.arange(frames, dtype=torch.float32)
    weak_fundamental = 0.04 * torch.sin(
        2.0 * torch.pi * time / period
    )
    dominant_harmonic = torch.sin(
        2.0 * torch.pi * dominant_order * time / period
    )
    return (weak_fundamental + dominant_harmonic)[None, :, None]


@pytest.mark.parametrize("dominant_order", [3, 5, 7])
def test_dominant_high_harmonic_recovers_fundamental(
    dominant_order: int,
) -> None:
    embeddings = _harmonic_embedding(
        period=64,
        dominant_order=dominant_order,
    )

    period, confidence = estimate_harmonic_fundamental_from_embeddings(
        embeddings,
        minimum=4,
        maximum=128,
    )

    assert float(period[0]) == pytest.approx(64.0)
    assert 0.0 < float(confidence[0]) <= 1.0


@pytest.mark.parametrize("expected", [4, 8, 16, 32, 64, 128])
def test_pure_fundamental_grid_is_not_promoted_to_subharmonic(
    expected: int,
) -> None:
    time = torch.arange(256, dtype=torch.float32)
    embeddings = torch.sin(
        2.0 * torch.pi * time / expected
    )[None, :, None]

    period, confidence = estimate_harmonic_fundamental_from_embeddings(
        embeddings,
        minimum=4,
        maximum=128,
    )

    assert float(period[0]) == pytest.approx(float(expected))
    assert 0.0 <= float(confidence[0]) <= 1.0


def test_zero_and_white_noise_confidence_are_conservative_and_finite() -> None:
    zero = torch.zeros((1, 256, 8), dtype=torch.float32)
    zero_period, zero_confidence = (
        estimate_harmonic_fundamental_from_embeddings(
            zero,
            minimum=4,
            maximum=128,
        )
    )
    assert float(zero_period[0]) == 4.0
    assert float(zero_confidence[0]) == 0.0

    noise_confidences: list[float] = []
    for seed in range(8):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        white_noise = torch.randn(
            (1, 256, 8),
            generator=generator,
        )
        period, confidence = estimate_harmonic_fundamental_from_embeddings(
            white_noise,
            minimum=4,
            maximum=128,
        )
        assert torch.isfinite(period).all()
        assert torch.isfinite(confidence).all()
        assert 0.0 <= float(confidence[0]) <= 1.0
        noise_confidences.append(float(confidence[0]))
    assert sum(noise_confidences) / len(noise_confidences) <= 0.10


def test_mask_blocks_invalid_values_and_preserves_harmonic_fundamental() -> None:
    clean = _harmonic_embedding(period=64, dominant_order=7)
    valid = torch.ones((1, 256), dtype=torch.bool)
    valid[:, 70:96] = False
    corrupted = clean.clone()
    corrupted[:, 70:96] = 1e20

    clean_result = estimate_harmonic_fundamental_from_embeddings(
        clean,
        minimum=4,
        maximum=128,
        valid_mask=valid,
    )
    corrupted_result = estimate_harmonic_fundamental_from_embeddings(
        corrupted,
        minimum=4,
        maximum=128,
        valid_mask=valid,
    )

    assert float(clean_result[0][0]) == pytest.approx(64.0)
    assert torch.equal(clean_result[0], corrupted_result[0])
    assert torch.equal(clean_result[1], corrupted_result[1])
    with pytest.raises(ValueError, match="valid_mask"):
        estimate_harmonic_fundamental_from_embeddings(
            clean,
            minimum=4,
            maximum=128,
            valid_mask=torch.ones(256, dtype=torch.bool),
        )


def test_diagnostic_records_normalized_eight_harmonic_evidence() -> None:
    diagnostic = embedding_velocity_harmonic_fundamental_diagnostics(
        _harmonic_embedding(period=64, dominant_order=7),
        minimum=4,
        maximum=128,
        maximum_harmonic=8,
    )[0]

    assert diagnostic.selected_period == pytest.approx(64.0)
    assert diagnostic.selected_bin == 4
    assert diagnostic.prewhitened_bin == 28
    assert len(diagnostic.selected_harmonic_bins) == 8
    assert diagnostic.selected_harmonic_bins[6] == 28
    assert diagnostic.overtone_noise_floor > 0.0
    assert 0.0 <= diagnostic.confidence <= 1.0


def test_legacy_embedding_estimator_remains_stable_across_platforms() -> None:
    time = torch.arange(64, dtype=torch.float32)
    embeddings = torch.stack(
        (
            torch.sin(2.0 * torch.pi * time / 16.0),
            0.3 * torch.cos(2.0 * torch.pi * time / 9.0),
            time / 64.0,
        ),
        dim=-1,
    )[None]
    valid = torch.ones((1, 64), dtype=torch.bool)
    valid[:, 13:17] = False

    period, confidence = estimate_period_from_embeddings(
        embeddings,
        minimum=4,
        maximum=32,
        valid_mask=valid,
    )

    assert int(period.view(torch.int32)[0]) == 1_098_907_648
    torch.testing.assert_close(
        confidence,
        torch.tensor([0.6535627841949463], dtype=torch.float32),
        rtol=0.0,
        atol=2.0 * torch.finfo(torch.float32).eps,
    )


def test_runner_cli_has_only_target_free_input_surface_and_frozen_gates() -> None:
    runner = _load_runner()
    parsed = runner._parse_arguments(
        [
            "--checkpoint",
            "encoder.pt",
            "--config",
            "noabs.yaml",
            "--pose-cache-dir",
            "train-pose-cache",
            "--output",
            "harmonic-predev.json",
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
    assert set(inspect.signature(runner.run_predev_gate).parameters) == {
        "checkpoint_path",
        "config_path",
        "pose_cache_dir",
        "sample_size",
        "seed",
        "device",
        "batch_size",
    }
    assert runner._ESTIMATOR_CONFIG["maximum_harmonic"] == 8
    assert runner._THRESHOLDS == runner._FROZEN_GATE._THRESHOLDS
    runner._validated_noabspe_candidate(_noabs_config())


def test_runner_rejects_non_noabs_candidate() -> None:
    runner = _load_runner()
    payload = _noabs_config().model_dump()
    payload["model"]["position_encoding_mode"] = "sinusoidal"
    config = PAMSConfig.model_validate(payload)

    with pytest.raises(ValueError, match="position_encoding_mode=none"):
        runner._validated_noabspe_candidate(config)
