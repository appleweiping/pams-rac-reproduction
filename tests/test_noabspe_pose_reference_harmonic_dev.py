from __future__ import annotations

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
RUNNER = ROOT / "scripts/server/run_noabspe_pose_reference_harmonic_dev.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_noabspe_pose_reference_harmonic_dev",
        RUNNER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config() -> PAMSConfig:
    return PAMSConfig(
        seed=2026,
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
        period=PeriodConfig(minimum=4, maximum=16, pose_energy_epochs=1),
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
        ),
        consensus=ConsensusConfig(),
    )


class _PoseEncoder(nn.Module):
    position_encoding_mode = "none"

    def forward(
        self,
        poses: torch.Tensor,
        valid_mask: torch.Tensor,
        *,
        position_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del position_indices
        flattened = poses.flatten(start_dim=2)
        encoded = F.normalize(flattened, p=2, dim=-1)
        return encoded.masked_fill(~valid_mask.unsqueeze(-1), 0.0)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.encoder = _PoseEncoder()


def test_predict_cli_has_only_label_free_dev_inputs() -> None:
    runner = _load_runner()
    parser = runner.build_parser()
    arguments = parser.parse_args(
        [
            "predict",
            "--checkpoint",
            "encoder.pt",
            "--config",
            "config.yaml",
            "--passed-predev",
            "passed-predev.json",
            "--dev-inputs",
            "dev.inputs.json",
            "--dev-commitment",
            "dev.inputs.commitment.json",
            "--pose-cache-dir",
            "pose-dev84",
            "--output-dir",
            "output",
        ]
    )

    assert set(vars(arguments)) == {
        "command",
        "checkpoint",
        "config",
        "passed_predev",
        "dev_inputs",
        "dev_commitment",
        "pose_cache_dir",
        "output_dir",
        "device",
        "handler",
    }
    predict = parser._subparsers._group_actions[0].choices["predict"]
    options = {
        option
        for action in predict._actions
        for option in action.option_strings
    }
    assert "--dev-targets" not in options
    assert "--test-inputs" not in options
    assert "--test-targets" not in options


def test_passed_predev_must_bind_exact_inputs_semantics_and_authorization() -> None:
    runner = _load_runner()
    config = _config()
    criteria = {
        name: {
            "value": 1.0 if operator == ">=" else 0.0,
            "operator": operator,
            "threshold": threshold,
            "pass": True,
        }
        for name, (operator, threshold) in (
            runner._EXPECTED_PREDEV_CRITERIA.items()
        )
    }
    payload = {
        "artifact_type": "pams_noabspe_pose_reference_harmonic_predev_gate",
        "classification": "inferred target-free predev diagnostic",
        "disclosed_by_pams_authors": False,
        "eligible_for_paper_table": False,
        "readout": runner._EXPECTED_PREDEV_READOUT,
        "inputs": {
            "checkpoint_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "config_fingerprint": config.fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
            "position_encoding_mode": "none",
            "checkpoint_stage": "encoder",
        },
        "gate": {
            "thresholds_frozen_before_checkpoint_evaluation": True,
            "overall_pass": True,
            "dev84_prediction_authorized": True,
            "dev84_scoring_authorized": False,
            "test105_evaluation_authorized": False,
            "criteria": criteria,
        },
        "synthetic_period_recovery": {
            "embedding_estimator": "legacy_embedding_velocity_fft",
            "pose_reference_estimator": "raw_pose_energy_fft",
            "maximum_harmonic": 8,
            "periods": [4, 8, 16, 32, 64, 128],
            "rows": [
                {
                    "expected_period_frames": period,
                    "confidence_rule": (
                        "embedding_confidence_times_pose_energy_confidence"
                    ),
                    "selected_harmonic_factor": 1,
                    "selected_period_frames": float(period),
                    "relative_error": 0.0,
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
    runner._validate_passed_predev_payload(
        payload,
        checkpoint_sha256="a" * 64,
        config_sha256="b" * 64,
        config=config,
    )

    payload["gate"]["dev84_prediction_authorized"] = False
    with pytest.raises(ValueError, match="authorization mismatch"):
        runner._validate_passed_predev_payload(
            payload,
            checkpoint_sha256="a" * 64,
            config_sha256="b" * 64,
            config=config,
        )

    payload["gate"]["dev84_prediction_authorized"] = True
    payload["gate"]["criteria"]["training_period_top_bin_share"]["threshold"] = 1.0
    with pytest.raises(ValueError, match="criterion is malformed"):
        runner._validate_passed_predev_payload(
            payload,
            checkpoint_sha256="a" * 64,
            config_sha256="b" * 64,
            config=config,
        )

    payload["gate"]["criteria"]["training_period_top_bin_share"]["threshold"] = 0.25
    payload["gate"]["criteria"]["training_period_top_bin_share"]["value"] = 0.5
    with pytest.raises(ValueError, match="criterion value failed"):
        runner._validate_passed_predev_payload(
            payload,
            checkpoint_sha256="a" * 64,
            config_sha256="b" * 64,
            config=config,
        )


def test_current_predev_source_is_an_explicit_prediction_receipt_binding() -> None:
    runner = _load_runner()
    source = inspect.getsource(runner.run_predict)

    assert (
        '"predev_source_sha256": source_hashes["predev_readout"]'
        in source
    )
    assert runner._source_hashes()["predev_readout"] == (
        "65028e744588a828509763d1ca1374d07251033df80cc944c9c8a9b3c447f2b2"
    )


def test_frozen_counter_rejects_parameter_drift() -> None:
    runner = _load_runner()
    runner._validated_counter(_config())
    changed = _config().model_copy(
        update={
            "consensus": _config().consensus.model_copy(
                update={"height_factor": 0.61}
            )
        }
    )
    with pytest.raises(ValueError, match="frozen multi-expert defaults"):
        runner._validated_counter(changed)


def test_batch_prediction_returns_count_result_and_both_period_evidence_paths() -> None:
    runner = _load_runner()
    sequence = runner.predev._synthetic_period_sequence(
        8,
        frames=32,
        seed=2026,
    )
    records = runner.predict_pose_reference_harmonic_batches(
        _Model(),
        (sequence,),
        _config(),
        video_sha256={sequence.video_id: "a" * 64},
        pose_cache_sha256={sequence.video_id: "b" * 64},
    )

    assert len(records) == 1
    row = records[0]
    assert isinstance(row["count"], int)
    assert len(row["expert_counts"]) == 3
    assert len(row["period_stream"]) == 32
    assert np.isfinite(row["period_stream"]).all()
    evidence = row["period_evidence"]
    assert evidence["pose_energy_period_frames"] == pytest.approx(8.0)
    assert evidence["selected_period_frames"] == pytest.approx(8.0)
    assert evidence["selected_period_confidence"] == pytest.approx(
        evidence["embedding_period_confidence"]
        * evidence["pose_energy_period_confidence"]
    )


def test_score_validates_frozen_prediction_before_first_target_access() -> None:
    runner = _load_runner()
    source = inspect.getsource(runner.run_score)
    validation = source.index("_validate_frozen_predictions(")
    first_target = source.index("targets_path = arguments.dev_targets.resolve()")
    target_hash = source.index("_sha256_file(targets_path)")

    assert validation < first_target < target_hash
    assert "load_dev_target_manifest(targets_path)" in source[target_hash:]


def test_prediction_writer_is_exclusive(tmp_path: Path) -> None:
    runner = _load_runner()
    output = tmp_path / "predictions.json"
    first = runner._write_json_exclusive(output, {"finite": 1.0})

    assert len(first) == 64
    assert output.read_bytes().endswith(b"\n")
    with pytest.raises(FileExistsError):
        runner._write_json_exclusive(output, {"finite": 1.0})
