from __future__ import annotations

import copy
import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pams.config import (
    ConsensusConfig,
    DataConfig,
    LossConfig,
    ModelConfig,
    PAMSConfig,
    PeriodConfig,
    TrainingConfig,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/server/run_noabspe_projected_null_dev.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_noabspe_projected_null_dev",
        RUNNER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> PAMSConfig:
    return PAMSConfig(
        seed=3407,
        data=DataConfig(frames=32),
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
            maximum=16,
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
        consensus=ConsensusConfig(),
    )


def _passed_payload(
    runner: ModuleType,
    config: PAMSConfig,
    *,
    null_sha256: str = "c" * 64,
) -> dict[str, object]:
    criteria = {
        name: {
            "value": 1.0 if operator == ">=" else 0.0,
            "operator": operator,
            "threshold": threshold,
            "pass": True,
        }
        for name, (operator, threshold) in (runner._EXPECTED_PREDEV_CRITERIA.items())
    }
    return {
        "artifact_type": "pams_noabspe_projected_null_predev_gate_v12",
        "classification": "exploratory-derived target-free candidate protocol",
        "disclosed_by_pams_authors": False,
        "eligible_for_paper_table": False,
        "protocol_freeze": copy.deepcopy(runner._EXPECTED_PROTOCOL_FREEZE),
        "method": copy.deepcopy(runner._EXPECTED_PREDEV_METHOD),
        "inputs": {
            "checkpoint_sha256": "a" * 64,
            "checkpoint_bytes": 123,
            "checkpoint_stage": "encoder",
            "config_sha256": "b" * 64,
            "config_bytes": 456,
            "config_fingerprint": config.fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
            "diagnostic_seed": 2026,
            "device_type": "cuda",
            "full_checkpoint_pose_cache_set_verified": True,
        },
        "label_firewall": {
            "dataset_manifest_argument_supported": False,
            "action_or_count_label_argument_supported": False,
            "development_input_mounted": False,
            "development_pose_mounted": False,
            "development_labels_mounted": False,
            "sealed_test_input_mounted": False,
            "sealed_test_labels_mounted": False,
            "label_fields_accessed": [],
        },
        "null_distribution": {
            "sample_total": 2_048,
            "seed": 73_400_711,
            "generation_batch_size": 32,
            "probe_random_seed_is_disjoint": True,
            "alpha": 0.01,
            "dtype": "little-endian float32",
            "sha256": null_sha256,
        },
        "gate": {
            "thresholds_frozen_before_seed3407_training": True,
            "overall_pass": True,
            "dev84_prediction_authorized": True,
            "dev84_scoring_authorized": False,
            "test105_evaluation_authorized": False,
            "criteria": criteria,
        },
        "synthetic_period_recovery": {
            "periods": [4, 8, 16, 32, 64, 128],
            "prediction_period_unchanged_by_calibration": True,
            "truth_source": "deterministic in-run synthetic generation only",
            "rows": [
                {
                    "expected_period_frames": period,
                    "predicted_period_frames": float(period),
                    "relative_error": 0.0,
                    "raw_peak_share": 1.0,
                    "null_exceedance_total": 0,
                    "empirical_one_sided_p": 1 / 2_049,
                    "calibrated_periodicity_confidence": 0.95,
                }
                for period in (4, 8, 16, 32, 64, 128)
            ],
        },
        "read_only_verification": {
            "checkpoint_sha256_unchanged": True,
            "config_sha256_unchanged": True,
            "model_or_training_state_updated": False,
            "pose_cache_write_operations": 0,
        },
    }


class _ProjectedEncoder(nn.Module):
    position_encoding_mode = "none"

    def forward_with_pre_pe(
        self,
        poses: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        projected = poses.flatten(start_dim=2)
        projected = projected.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return F.normalize(projected, p=2, dim=-1), projected


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.encoder = _ProjectedEncoder()


def test_cli_separates_target_free_prediction_from_scoring() -> None:
    runner = _load_runner()
    parser = runner.build_parser()
    prediction = parser.parse_args(
        [
            "predict",
            "--checkpoint",
            "encoder.pt",
            "--config",
            "config.yaml",
            "--passed-predev",
            "passed.json",
            "--dev-inputs",
            "dev.inputs.json",
            "--dev-commitment",
            "dev.commitment.json",
            "--pose-cache-dir",
            "pose",
            "--output-dir",
            "output",
        ]
    )
    scoring = parser.parse_args(
        [
            "score",
            "--predictions",
            "predictions.json",
            "--prediction-receipt",
            "receipt.json",
            "--dev-targets",
            "targets.json",
            "--output-dir",
            "score",
        ]
    )

    assert set(vars(prediction)) == {
        "command",
        "checkpoint",
        "config",
        "passed_predev",
        "dev_inputs",
        "dev_commitment",
        "pose_cache_dir",
        "output_dir",
        "handler",
    }
    assert "dev_targets" not in vars(prediction)
    assert set(vars(scoring)) == {
        "command",
        "predictions",
        "prediction_receipt",
        "dev_targets",
        "output_dir",
        "handler",
    }
    assert "checkpoint" not in vars(scoring)
    assert "pose_cache_dir" not in vars(scoring)
    source = inspect.getsource(runner.run_score)
    assert source.index("_validate_frozen_predictions") < source.index("arguments.dev_targets")


def test_passed_predev_requires_exact_six_of_six_and_null_binding() -> None:
    runner = _load_runner()
    config = _config()
    payload = _passed_payload(runner, config)
    assert (
        runner._validate_passed_predev_payload(
            payload,
            checkpoint_sha256="a" * 64,
            checkpoint_bytes=123,
            config_sha256="b" * 64,
            config_bytes=456,
            config=config,
        )
        == "c" * 64
    )

    unauthorized = copy.deepcopy(payload)
    unauthorized["gate"]["dev84_prediction_authorized"] = False
    with pytest.raises(ValueError, match="authorization mismatch"):
        runner._validate_passed_predev_payload(
            unauthorized,
            checkpoint_sha256="a" * 64,
            checkpoint_bytes=123,
            config_sha256="b" * 64,
            config_bytes=456,
            config=config,
        )

    drifted = copy.deepcopy(payload)
    drifted["protocol_freeze"]["null_sample_total"] = 2_047
    with pytest.raises(ValueError, match="protocol constants"):
        runner._validate_passed_predev_payload(
            drifted,
            checkpoint_sha256="a" * 64,
            checkpoint_bytes=123,
            config_sha256="b" * 64,
            config_bytes=456,
            config=config,
        )

    null = np.zeros(2_048, dtype="<f4")
    replay_payload = copy.deepcopy(payload)
    replay_payload["null_distribution"]["sha256"] = hashlib.sha256(
        null.tobytes(order="C")
    ).hexdigest()
    assert runner._validate_replayed_null_sha(replay_payload, null)
    changed = null.copy()
    changed[0] = 1.0
    with pytest.raises(ValueError, match="null SHA-256"):
        runner._validate_replayed_null_sha(replay_payload, changed)


def test_batch_path_uses_projected_pose_period_and_frozen_counter() -> None:
    runner = _load_runner()
    config = _config()
    sequence = runner.predev._synthetic_period_sequence(8, frames=32)
    null = np.zeros(2_048, dtype="<f4")
    null_sha256 = hashlib.sha256(null.tobytes(order="C")).hexdigest()

    rows = runner.predict_projected_null_batches(
        _Model(),
        (sequence,),
        config,
        null=null,
        null_sha256=null_sha256,
        video_sha256={sequence.video_id: "a" * 64},
        pose_cache_sha256={sequence.video_id: "b" * 64},
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["period_evidence"]["period_input"] == "projected_pose_pre_pe"
    assert row["period_evidence"]["period_estimator"] == ("estimate_period_from_projected_pose")
    assert row["period_evidence"]["counting_stream"] == (
        "temporal_component(projected_pose_pre_pe)"
    )
    assert row["period_evidence"]["null_sha256"] == null_sha256
    assert len(row["period_stream"]) == 32
    assert len(row["expert_counts"]) == 3
    assert 0.0 <= row["confidence"] <= 1.0


def test_predeclared_projected_stream_stress_reports_frozen_threshold_outcome() -> None:
    runner = _load_runner()
    report = runner.run_projected_stream_synthetic_stress()

    scenario_passes: list[bool] = []
    for scenario in runner._PLANNED_STRESS_SCENARIOS:
        scenario_report = report["reports"][scenario]
        metrics = scenario_report["metrics"]
        assert metrics["sample_total"] == 39
        threshold_pass = (
            metrics["nmae"] <= runner._STRESS_NMAE_MAXIMUM
            and metrics["obo"] >= runner._STRESS_OBO_MINIMUM
        )
        assert scenario_report["threshold_pass"] is threshold_pass
        scenario_passes.append(threshold_pass)
    # Peak detection at the frozen 0.95 OBO boundary is numerically
    # platform-sensitive (37/39 on the designated Linux image versus 38/39
    # on the development Windows build).  Preserve the threshold and surface
    # the measured outcome instead of weakening it or asserting a false
    # cross-platform pass.
    assert report["planned_overall_pass"] is all(scenario_passes)

    # This unplanned harsher diagnostic is intentionally disclosed as a
    # limitation; it must never be silently reclassified as a passing stress.
    assert report["extended_full_frame_invalid_gap_pass"] is False
    invalid = report["reports"]["full_frame_invalid_gap"]["metrics"]
    assert invalid["nmae"] == pytest.approx(0.2578248795524859)
    assert invalid["obo"] == pytest.approx(4 / 39)
