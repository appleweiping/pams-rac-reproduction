from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from pams.period import estimate_period_from_embeddings, estimate_period_from_pose
from pams.training import collate_pose_sequences

ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT / "scripts/server/run_noabspe_pose_reference_harmonic_predev.py"
)


def _load_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_noabspe_pose_reference_harmonic_predev",
        RUNNER,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("factor", [3, 5, 7])
def test_pose_reference_recovers_common_embedding_harmonics(factor: int) -> None:
    runner = _load_runner()
    expected = 84.0
    payload = runner._pose_reference_harmonic_resolution(
        expected / factor,
        0.40,
        expected,
        0.50,
    )

    assert payload["selected_harmonic_factor"] == factor
    assert payload["selected_period_frames"] == pytest.approx(expected)
    assert payload["selected_period_confidence"] == pytest.approx(0.20)


def test_candidate_bounds_and_deterministic_tie_break_are_explicit() -> None:
    runner = _load_runner()
    bounded = runner._pose_reference_harmonic_resolution(
        1.0,
        1.0,
        7.0,
        1.0,
        minimum=4,
        maximum=7,
        maximum_harmonic=8,
    )
    assert [row["harmonic_factor"] for row in bounded["candidates"]] == [
        4,
        5,
        6,
        7,
    ]
    assert bounded["selected_harmonic_factor"] == 7

    upper_boundary = runner._pose_reference_harmonic_resolution(
        128.000003 / 3.0,
        1.0,
        128.0,
        1.0,
    )
    assert upper_boundary["selected_harmonic_factor"] == 3
    assert upper_boundary["selected_period_frames"] == 128.0
    selected = upper_boundary["candidates"][-1]
    assert selected["raw_harmonic_period_frames"] > 128.0

    # 20 and 40 are exactly equidistant in log space around sqrt(800).
    # Relative distance then resolves toward 20; the final smaller-h rule
    # remains explicitly present as the last deterministic key.
    tied = runner._pose_reference_harmonic_resolution(
        20.0,
        1.0,
        math.sqrt(20.0 * 40.0),
        1.0,
    )
    assert tied["selected_harmonic_factor"] == 1
    assert "smaller harmonic factor" in tied["candidate_ranking"]


def test_confidence_is_exact_product_and_zero_pose_suppresses_output() -> None:
    runner = _load_runner()
    payload = runner._pose_reference_harmonic_resolution(
        8.0,
        0.25,
        24.0,
        0.40,
    )
    zero = runner._pose_reference_harmonic_resolution(
        8.0,
        0.90,
        24.0,
        0.0,
    )

    assert payload["selected_period_confidence"] == pytest.approx(0.10)
    assert zero["selected_period_confidence"] == 0.0


def test_raw_pose_energy_recovers_all_frozen_synthetic_periods_exactly() -> None:
    runner = _load_runner()
    expected = (4, 8, 16, 32, 64, 128)
    sequences = tuple(
        runner._synthetic_period_sequence(period, frames=256, seed=2026)
        for period in expected
    )
    batch = collate_pose_sequences(sequences)
    periods, confidences = estimate_period_from_pose(
        batch.poses,
        minimum=4,
        maximum=128,
        valid_mask=batch.valid_mask,
    )

    assert periods.tolist() == list(expected)
    assert torch.all(confidences > 0.0)


def test_white_random_pose_has_low_product_confidence() -> None:
    runner = _load_runner()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2026)
    poses = torch.rand((1, 256, 33, 3), generator=generator)
    valid = torch.ones((1, 256), dtype=torch.bool)
    # A deterministic no-PE identity projection is sufficient to exercise
    # both estimators without a checkpoint or any dataset input.
    embeddings = poses.flatten(start_dim=2)
    embedding_period, embedding_confidence = estimate_period_from_embeddings(
        embeddings,
        minimum=4,
        maximum=128,
        valid_mask=valid,
    )
    pose_period, pose_confidence = estimate_period_from_pose(
        poses,
        minimum=4,
        maximum=128,
        valid_mask=valid,
    )
    payload = runner._pose_reference_harmonic_resolution(
        float(embedding_period[0]),
        float(embedding_confidence[0]),
        float(pose_period[0]),
        float(pose_confidence[0]),
    )

    assert payload["selected_period_confidence"] < 0.10


def test_cli_and_callable_have_no_manifest_or_target_surface() -> None:
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
    assert set(inspect.signature(runner.run_predev_gate).parameters) == {
        "checkpoint_path",
        "config_path",
        "pose_cache_dir",
        "sample_size",
        "seed",
        "device",
        "batch_size",
    }
    source = RUNNER.read_text(encoding="utf-8").lower()
    assert "--manifest" not in source
    assert "--targets" not in source
    assert "load_dev_target_manifest" not in source
