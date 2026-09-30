from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

SCRIPT = Path(__file__).parents[1] / "scripts/server/explore_lag_acf_dev_candidates.py"


def _module():
    spec = importlib.util.spec_from_file_location("explore_lag_acf_dev_candidates", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diagnostic():
    return {
        "positive_peak_lags": [8, 16, 32],
        "positive_peak_heights": [0.50, 0.95, 0.90],
        "positive_peak_prominences": [0.40, 0.70, 0.80],
        "acf_lag_zero_to_128": [
            1.0 if index == 0 else 0.95 if index == 16 else 0.90 if index == 32 else 0.0
            for index in range(129)
        ],
    }


def test_metrics_use_half_up_rounded_nmae_and_obo() -> None:
    module = _module()

    metrics = module._metrics([1.5, 4.49, 8.5], [2, 5, 10])

    assert metrics["nmae_rounded"] == (0 / 2 + 1 / 5 + 1 / 10) / 3
    assert metrics["obo_rounded"] == 1.0
    assert metrics["exact_rounded"] == 1 / 3


def test_peak_selection_rules_are_deterministic() -> None:
    module = _module()
    diagnostic = _diagnostic()

    assert (
        module._select_period(
            diagnostic,
            family="near-small",
            parameter=0.90,
        )
        == 16.0
    )
    assert (
        module._select_period(
            diagnostic,
            family="near-large",
            parameter=0.90,
        )
        == 32.0
    )
    assert (
        module._select_period(
            diagnostic,
            family="height-power",
            parameter=0.0,
        )
        == 16.0
    )


def test_strategy_grid_covers_all_representations_and_is_bounded() -> None:
    module = _module()
    representations = ["pose-centered", "embedding-centered"]

    strategies = module._strategy_grid(representations)

    assert 1 < len(strategies) < 500
    assert {strategy[0] for strategy in strategies} == set(representations)
    assert len({_module()._strategy_name(strategy) for strategy in strategies}) == len(
        strategies
    )


def test_representation_diagnostics_recovers_a_simple_period() -> None:
    module = _module()
    time = torch.arange(256, dtype=torch.float32)
    signal = torch.stack(
        (
            torch.sin(2.0 * torch.pi * time / 32.0),
            torch.cos(2.0 * torch.pi * time / 32.0),
        ),
        dim=-1,
    ).unsqueeze(0)
    mask = torch.ones((1, 256), dtype=torch.bool)

    rows = module._representation_diagnostics(
        signal,
        mask,
        detrend=False,
        minimum=4,
        maximum=128,
    )

    assert len(rows) == 1
    assert 32 in rows[0]["positive_peak_lags"]
    assert len(rows[0]["acf_lag_zero_to_128"]) == 129


def test_prediction_and_scoring_claim_boundaries_are_explicit() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert '"dev_targets_mounted": False' in source
    assert '"sealed_test_assets_mounted": False' in source
    assert '"development_labels_selected_strategy": True' in source
    assert '"may_authorize_sealed_test": False' in source
    assert '"pose_mounted": False' in source
    assert '"checkpoint_mounted": False' in source
