from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path
from types import ModuleType

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
    load_config,
)
from pams.period import estimate_period_from_projected_pose
from pams.training import build_pams_model, collate_pose_sequences

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "server" / "run_noabspe_projected_null_predev_gate.py"
V11_CONFIG = ROOT / "configs" / "experiments" / "pams_no_absolute_pe_v11.yaml"
SEED42_V11_CONFIG = ROOT / "configs" / "experiments" / "pams_no_absolute_pe_seed42_v11.yaml"
V12_CONFIG = ROOT / "configs" / "experiments" / "pams_no_absolute_pe_seed3407_v12.yaml"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_noabspe_projected_null_predev_gate",
        RUNNER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_config() -> PAMSConfig:
    return PAMSConfig(
        seed=3407,
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
            position_encoding_mode="none",
        ),
        period=PeriodConfig(
            minimum=4,
            maximum=8,
            pose_energy_epochs=1,
            post_warmup_source="projected_pose_velocity_vector_acf",
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


def test_v12_config_is_seed_only_replica_and_preserves_v11_identity() -> None:
    runner = _load_runner()
    v11 = load_config(V11_CONFIG)
    seed42_v11 = load_config(SEED42_V11_CONFIG)
    v12 = load_config(V12_CONFIG)

    assert v11.fingerprint == ("d7a81b4acc44b6148574553803c896d5183817200d7cac5b2621fdb7a87f8614")
    assert seed42_v11.fingerprint == (
        "6a177379f48a2233d5e53df1d7f7c8799f81ba45fa322a55c9b3e58774aeabf0"
    )
    assert v12.fingerprint == runner._REQUIRED_CONFIG_FINGERPRINT
    assert v12.seed == 3407
    assert v12.nonseed_canonical_dict() == v11.nonseed_canonical_dict()
    assert v12.nonseed_fingerprint == v11.nonseed_fingerprint
    assert v12.pose_fingerprint == v11.pose_fingerprint
    runner._validate_candidate_config(v12)
    with pytest.raises(ValueError, match="exact frozen v12"):
        runner._validate_candidate_config(v11)


def test_frozen_constants_and_target_free_cli_surface() -> None:
    runner = _load_runner()
    parsed = runner._parse_arguments(
        [
            "--checkpoint",
            "encoder.pt",
            "--config",
            "config.yaml",
            "--train-pose-cache-dir",
            "train-pose",
            "--output",
            "predev.json",
        ]
    )

    assert vars(parsed) == {
        "checkpoint": Path("encoder.pt"),
        "config": Path("config.yaml"),
        "train_pose_cache_dir": Path("train-pose"),
        "output": Path("predev.json"),
    }
    assert set(inspect.signature(runner.run_predev_gate).parameters) == {
        "checkpoint_path",
        "config_path",
        "train_pose_cache_dir",
    }
    assert runner._NULL_SEED == 73_400_711
    assert runner._NULL_SAMPLE_TOTAL == 2_048
    assert runner._SIGNIFICANCE_ALPHA == 0.01
    assert runner._REQUIRED_SEED == 3407
    assert runner._REQUIRED_DEVICE_TYPE == "cuda"
    assert runner._DIAGNOSTIC_SEED == 2026
    assert runner._SAMPLED_TRAINING_VIDEO_TOTAL == 64
    assert runner._BATCH_SIZE == 32


def test_empirical_null_calibration_has_plus_one_correction() -> None:
    runner = _load_runner()
    null = np.asarray([0.1, 0.2, 0.3, 0.4], dtype="<f4")
    assert "period" not in inspect.signature(runner._calibrate_peak_share).parameters

    middle = runner._calibrate_peak_share(0.25, null, alpha=0.75)
    tied = runner._calibrate_peak_share(0.30, null, alpha=0.75)
    above = runner._calibrate_peak_share(0.50, null, alpha=0.75)

    assert middle["null_exceedance_total"] == 2
    assert middle["empirical_one_sided_p"] == pytest.approx(3 / 5)
    assert middle["calibrated_periodicity_confidence"] == pytest.approx(0.2)
    assert tied["null_exceedance_total"] == 2
    assert above["null_exceedance_total"] == 0
    assert above["empirical_one_sided_p"] == pytest.approx(1 / 5)
    assert above["calibrated_periodicity_confidence"] == pytest.approx(1 - (1 / 5) / 0.75)
    assert runner._calibrate_peak_share(0.0, null)["calibrated_periodicity_confidence"] == 0.0


def test_projected_wrapper_preserves_direct_period_argmax() -> None:
    runner = _load_runner()
    config = _tiny_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3407)
        model = build_pams_model(config).eval()
    sequence = runner._synthetic_period_sequence(8, frames=16)
    batch = collate_pose_sequences((sequence,))

    with torch.inference_mode():
        wrapped_period, wrapped_raw = runner._projected_period_estimate(
            model,
            batch.poses,
            batch.valid_mask,
            config,
        )
        _, projected = model.encoder.forward_with_pre_pe(
            batch.poses,
            batch.valid_mask,
        )
        direct_period, direct_raw = estimate_period_from_projected_pose(
            projected,
            minimum=config.period.minimum,
            maximum=config.period.maximum,
            valid_mask=batch.valid_mask,
        )

    assert torch.equal(wrapped_period, direct_period)
    assert torch.equal(wrapped_raw, direct_raw)


def test_white_noise_null_is_seeded_bounded_and_numerically_batch_stable() -> None:
    runner = _load_runner()
    config = _tiny_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3407)
        model = build_pams_model(config).eval()

    first = runner._build_white_noise_null(
        model,
        config,
        device=torch.device("cpu"),
        sample_total=7,
        seed=73_400_711,
        batch_size=2,
    )
    repeat = runner._build_white_noise_null(
        model,
        config,
        device=torch.device("cpu"),
        sample_total=7,
        seed=73_400_711,
        batch_size=2,
    )
    alternate_batch = runner._build_white_noise_null(
        model,
        config,
        device=torch.device("cpu"),
        sample_total=7,
        seed=73_400_711,
        batch_size=3,
    )

    assert first.dtype == np.dtype("<f4")
    assert np.array_equal(first, repeat)
    assert np.allclose(first, alternate_batch, rtol=0.0, atol=1e-6)
    assert np.isfinite(first).all()
    assert np.logical_and(first >= 0.0, first <= 1.0).all()


def test_synthetic_generator_is_exact_and_normalized() -> None:
    runner = _load_runner()
    first = runner._synthetic_period_sequence(8, frames=32)
    second = runner._synthetic_period_sequence(8, frames=32)

    assert np.array_equal(first.xyz, second.xyz)
    assert np.array_equal(first.valid_mask, second.valid_mask)
    assert np.allclose(first.xyz[:24], first.xyz[8:32], atol=1e-6)
    assert float(first.xyz.min()) >= 0.0
    assert float(first.xyz.max()) <= 1.0


def test_noabs_position_invariance_is_exact() -> None:
    runner = _load_runner()
    config = _tiny_config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3407)
        model = build_pams_model(config).eval()
    sequences = tuple(runner._synthetic_period_sequence(period, frames=16) for period in (4, 8))

    report = runner._noabs_position_invariance(
        model,
        sequences,
        device=torch.device("cpu"),
        batch_size=2,
    )

    assert report["position_encoding_mode"] == "none"
    assert report["valid_frame_cosine"]["median"] == pytest.approx(
        1.0,
        abs=1e-6,
    )


def test_six_gate_decision_requires_every_frozen_criterion() -> None:
    runner = _load_runner()
    passing = runner._gate_decision(
        frame_index_r2=0.10,
        zero_pose_confidence=0.10,
        random_pose_confidence=0.10,
        training_top_bin_share=0.25,
        position_invariance_median_cosine=0.95,
        synthetic_median_relative_error=0.10,
    )

    assert passing["overall_pass"] is True
    assert passing["dev84_prediction_authorized"] is True
    assert passing["dev84_scoring_authorized"] is False
    assert passing["test105_evaluation_authorized"] is False
    assert all(row["pass"] for row in passing["criteria"].values())

    failing = runner._gate_decision(
        frame_index_r2=0.10,
        zero_pose_confidence=0.10,
        random_pose_confidence=0.100001,
        training_top_bin_share=0.25,
        position_invariance_median_cosine=0.95,
        synthetic_median_relative_error=0.10,
    )
    assert failing["overall_pass"] is False
    assert failing["dev84_prediction_authorized"] is False
    assert failing["criteria"]["random_pose_period_confidence"]["pass"] is False


def test_immutable_output_rejects_overwrite(tmp_path: Path) -> None:
    runner = _load_runner()
    output = tmp_path / "predev.json"
    runner._write_new_json(output, {"finite": 1.0})

    assert output.read_text(encoding="utf-8").endswith("\n")
    with pytest.raises(FileExistsError):
        runner._write_new_json(output, {"finite": 1.0})
