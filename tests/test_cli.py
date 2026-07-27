from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from pams.cli import app
from pams.data import write_pose_cache
from pams.types import PoseSequence

runner = CliRunner()
REPOSITORY = Path(__file__).parents[1]


def _json_output(value: str) -> dict[str, object]:
    return json.loads(value)


def _pose_cache(tmp_path: Path) -> Path:
    frames = 64
    phase = np.arange(frames, dtype=np.float32) * 2.0 * np.pi / 8.0
    xyz = np.zeros((frames, 33, 3), dtype=np.float32)
    xyz[:, :, 0] = np.sin(phase)[:, None]
    xyz[:, :, 1] = np.cos(phase)[:, None]
    sequence = PoseSequence(
        "periodic",
        30.0,
        xyz,
        np.ones(frames, dtype=bool),
    )
    path = tmp_path / "pose.npz"
    write_pose_cache(
        path,
        sequence,
        video_sha256="a" * 64,
        config_sha256="b" * 64,
    )
    return path


def test_root_help_has_only_frozen_command_groups() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    expected = {
        "config",
        "data",
        "pose",
        "train",
        "evaluate",
        "baseline",
        "report",
        "synthetic",
    }
    for command in expected:
        assert command in result.stdout


def test_config_validate_outputs_fingerprint() -> None:
    result = runner.invoke(
        app,
        ["config", "validate", str(REPOSITORY / "configs" / "pams.yaml")],
    )
    assert result.exit_code == 0, result.output
    payload = _json_output(result.stdout)
    assert payload["valid"] is True
    assert payload["protocol"] == "ucfrep_526"
    assert len(str(payload["fingerprint"])) == 64


def test_spectral_proxy_is_explicitly_diagnostic(tmp_path: Path) -> None:
    cache = _pose_cache(tmp_path)
    result = runner.invoke(app, ["baseline", "predict", "spectral-proxy", str(cache)])
    assert result.exit_code == 0, result.output
    payload = _json_output(result.stdout)
    assert payload["method"] == "spectral-proxy"
    assert payload["diagnostic_only"] is True
    assert payload["eligible_for_paper_table"] is False


def test_blocked_baseline_never_falls_back_to_proxy(tmp_path: Path) -> None:
    cache = _pose_cache(tmp_path)
    result = runner.invoke(app, ["baseline", "predict", "repnet", str(cache)])
    assert result.exit_code != 0
    assert "blocked_unimplemented" in result.output
    assert '"method": "spectral-proxy"' not in result.output


def test_synthetic_evaluation_requires_no_pose_dependency() -> None:
    result = runner.invoke(
        app,
        [
            "synthetic",
            "evaluate",
            "--config",
            str(REPOSITORY / "configs" / "pams.yaml"),
            "--counts",
            "2,4",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _json_output(result.stdout)
    assert payload["method"] == "spectral-proxy"
    assert payload["sample_count"] == 2
    assert payload["diagnostic_only"] is True
