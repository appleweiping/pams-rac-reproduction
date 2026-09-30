from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/server/run_pams_v14_predev_gate.py"


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_pams_v14_predev_gate",
        RUNNER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sample(
    runner: ModuleType,
    video_id: str,
    period: float,
    *,
    confidence: float = 1.0,
    stream_std: float = 1.0,
) -> object:
    return runner._EncodedTrainingSample(
        video_id=video_id,
        valid_frames=256,
        period_frames=period,
        confidence=confidence,
        period_stream=(period - stream_std, period + stream_std),
        confidence_stream=(1.0, 1.0),
        period_stream_std=stream_std,
    )


def _artifact_payload() -> dict[str, object]:
    digest = "a" * 64
    return {
        "status": "passed",
        "inputs": {
            "checkpoint_sha256": digest,
            "config_sha256": digest,
            "train337_pose_cache_set_sha256": digest,
            "runner_sha256": digest,
        },
        "hardware_sha256": digest,
        "runtime_sha256": digest,
    }


def test_cli_and_run_surface_accept_only_target_free_scientific_inputs() -> None:
    runner = _load_runner()
    parsed = runner._parse_arguments(
        [
            "--checkpoint",
            "encoder.pt",
            "--config",
            "v14.yaml",
            "--pose-cache-dir",
            "train337-pose",
            "--output",
            "predev.json",
        ]
    )

    assert set(vars(parsed)) == {
        "checkpoint",
        "config",
        "pose_cache_dir",
        "output",
        "device",
        "batch_size",
    }
    assert set(inspect.signature(runner.run_predev_gate).parameters) == {
        "checkpoint_path",
        "config_path",
        "pose_cache_dir",
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
    ):
        assert f'"{forbidden}"' not in parser_source
        assert f"'{forbidden}'" not in parser_source


def test_distribution_metrics_detect_boundaries_mode_and_stream_collapse() -> None:
    runner = _load_runner()
    collapsed = tuple(
        _sample(
            runner,
            f"v{index}",
            4.0,
            confidence=0.0,
            stream_std=0.0,
        )
        for index in range(8)
    )
    metrics = runner._training_distribution(
        collapsed,
        minimum=4,
        maximum=128,
    )

    assert metrics["boundary_share"] == 1.0
    assert metrics["minimum_period_share"] == 1.0
    assert metrics["maximum_period_share"] == 0.0
    assert metrics["mode_share"] == 1.0
    assert metrics["zero_confidence_share"] == 1.0
    assert metrics["period_stream_std"]["median"] == 0.0

    diverse = tuple(
        _sample(runner, f"d{index}", float(12 + index), stream_std=0.5) for index in range(8)
    )
    diverse_metrics = runner._training_distribution(
        diverse,
        minimum=4,
        maximum=128,
    )
    assert diverse_metrics["boundary_share"] == 0.0
    assert diverse_metrics["mode_share"] == pytest.approx(1 / 8)
    assert diverse_metrics["period_stream_std"]["median"] == 0.5


def test_frozen_gate_requires_all_seven_target_free_criteria() -> None:
    runner = _load_runner()
    passing = {
        "boundary_share": 0.24,
        "mode_share": 0.24,
        "period_stream_std_median": 0.05,
        "time_scale_median_relative_error": 0.15,
        "synthetic_count_median_relative_error": 0.05,
        "static_maximum_confidence": 0.05,
        "harmonic_fundamental_fraction": 1.0,
    }
    decision = runner._gate_decision(**passing)
    assert decision["overall_pass"] is True
    assert decision["dev84_prediction_authorized"] is True
    assert decision["dev84_scoring_authorized"] is False
    assert decision["test105_evaluation_authorized"] is False

    for field in passing:
        failing = dict(passing)
        if field in {"boundary_share", "mode_share"}:
            failing[field] = 0.25
        elif field == "period_stream_std_median":
            failing[field] = 0.049
        elif field == "harmonic_fundamental_fraction":
            failing[field] = 0.94
        else:
            failing[field] = float(failing[field]) + 0.001
        failed = runner._gate_decision(**failing)
        assert failed["overall_pass"] is False
        assert (
            failed["criteria"][
                {
                    "boundary_share": "training_boundary_share",
                    "mode_share": "training_mode_share",
                    "period_stream_std_median": ("training_period_stream_std_median"),
                    "time_scale_median_relative_error": (
                        "training_time_scale_median_relative_error"
                    ),
                    "synthetic_count_median_relative_error": (
                        "synthetic_count_median_relative_error"
                    ),
                    "static_maximum_confidence": ("synthetic_static_maximum_confidence"),
                    "harmonic_fundamental_fraction": ("synthetic_harmonic_fundamental_fraction"),
                }[field]
            ]["pass"]
            is False
        )


def test_window_schedule_and_time_scaling_are_frozen() -> None:
    runner = _load_runner()

    assert runner._window_starts(256) == (0, 32, 64, 96, 128)
    assert runner._TIME_SCALE_FACTORS == (0.50, 0.75)
    assert tuple(range(2, 41)) == runner._SYNTHETIC_COUNTS
    assert tuple(range(2, 8)) == runner._HARMONIC_ORDERS
    with pytest.raises(ValueError, match="at least 128"):
        runner._window_starts(127)


def test_artifact_and_receipt_are_exclusive_and_hash_bound(
    tmp_path: Path,
) -> None:
    runner = _load_runner()
    output = tmp_path / "predev.json"
    payload = _artifact_payload()

    receipt_path, digest = runner._write_artifact_and_receipt(
        output,
        payload,
    )

    artifact_bytes = output.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert digest == hashlib.sha256(artifact_bytes).hexdigest()
    assert receipt["artifact_sha256"] == digest
    assert receipt["artifact_bytes"] == len(artifact_bytes)
    assert receipt["hardware_sha256"] == payload["hardware_sha256"]
    assert receipt["runtime_sha256"] == payload["runtime_sha256"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner._write_artifact_and_receipt(output, payload)


def test_source_declares_hashes_read_only_recheck_and_no_label_loader() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert '"hardware_sha256": sha256_json(hardware)' in source
    assert '"runtime_sha256": sha256_json(runtime)' in source
    assert '"classification": "inferred anti-collapse diagnostic"' in source
    assert "not the paper's untrained Period Head" in source
    assert "genuinely stable per-video period" in source
    assert "train337_pose_cache_set_sha256_unchanged" in source
    assert source.count("_load_training_poses(") == 2
    assert "_stable_file_sha256(checkpoint)" in source
    assert "_stable_file_sha256(configuration)" in source
    assert "load_dev_target_manifest" not in source
    assert "load_target" not in source
    assert 'test105_evaluation_authorized": False' in source
