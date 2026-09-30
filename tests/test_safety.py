from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from pams.safety import (
    COUNTER_GATE_COUNTS,
    COUNTER_GATE_GAUSSIAN_WIDTH_CYCLES,
    COUNTER_GATE_PHASES,
    COUNTER_GATE_SECOND_HARMONIC_WEIGHT,
    COUNTER_GATE_SHAPES,
    COUNTER_GATE_SIGNS,
    COUNTER_GATE_THRESHOLDS,
    SSHEAD_NEAR_COLLAPSE_GRAD_RMS_LIMIT,
    run_counter_sign_phase_gate,
    run_period_counter_sign_phase_gate,
    run_sshead_collapse_diagnostic,
)


@dataclass(frozen=True)
class _Prediction:
    count: int


class _PerfectCounter:
    def count(self, period_stream: np.ndarray, period_frames: float) -> _Prediction:
        del period_stream
        return _Prediction(round(256.0 / period_frames))


class _SignBiasedCounter(_PerfectCounter):
    def __init__(self) -> None:
        self.call_index = 0

    def count(self, period_stream: np.ndarray, period_frames: float) -> _Prediction:
        target = super().count(period_stream, period_frames).count
        # The frozen gate calls -1 first and +1 second for every pair.
        prediction = target if self.call_index % 2 == 0 else target + 2
        self.call_index += 1
        return _Prediction(prediction)


def test_counter_sign_phase_gate_freezes_full_matrix_and_thresholds() -> None:
    report = run_counter_sign_phase_gate(_PerfectCounter())

    assert report["passed"] is True
    assert report["scope"] == "multi_expert_counter_component_only"
    assert report["period_source"] == "exact_synthetic_period"
    assert report["end_to_end_pams"] is False
    assert report["metrics"]["case_count"] == (
        len(COUNTER_GATE_COUNTS)
        * COUNTER_GATE_PHASES
        * len(COUNTER_GATE_SIGNS)
        * len(COUNTER_GATE_SHAPES)
    )
    assert report["thresholds"] == {
        "max_absolute_error": 1.0,
        "obo": 1.0,
        "nmae": 0.08,
        "shape_nmae": 0.10,
        "phase_range": 1.0,
        "sign_pair_abs_delta": 1.0,
        "mean_sign_pair_abs_delta": 0.90,
        "absolute_signed_bias": 0.25,
    }
    assert report["thresholds"] == dict(COUNTER_GATE_THRESHOLDS)
    assert report["configuration"]["second_harmonic_weight"] == 0.5
    assert COUNTER_GATE_SECOND_HARMONIC_WEIGHT == 0.5
    assert report["configuration"]["circular_gaussian_width_cycles"] == 0.07
    assert COUNTER_GATE_GAUSSIAN_WIDTH_CYCLES == 0.07
    assert all(row["period_frames"] == 256.0 / row["target_count"] for row in report["details"])
    assert len(report["details"]) == 576
    assert json.loads(json.dumps(report, sort_keys=True)) == report


def test_counter_sign_phase_gate_rejects_a_systematic_bias() -> None:
    report = run_counter_sign_phase_gate(_SignBiasedCounter())

    assert report["passed"] is False
    assert report["checks"]["max_absolute_error"] is False
    assert report["checks"]["obo"] is False
    assert report["checks"]["absolute_signed_bias"] is False
    assert report["metrics"]["sign_pair_signed_bias"] == 2.0
    assert report["metrics"]["prediction_signed_error_mean"] == 1.0


def test_period_counter_gate_estimates_period_without_oracle_and_passes() -> None:
    report = run_period_counter_sign_phase_gate()

    assert report["passed"] is True
    assert report["period_source"] == "estimated_from_stream"
    assert report["end_to_end_pams"] is False
    assert report["metrics"]["case_count"] == 576
    assert report["metrics"]["max_period_relative_error"] <= 1e-4
    assert report["metrics"]["min_period_confidence"] > 0.0
    assert all("period_frames" not in row for row in report["details"])
    assert all("estimated_period_frames" in row for row in report["details"])


def test_sshead_collapse_diagnostic_reports_full_grid_and_current_failure() -> None:
    report = run_sshead_collapse_diagnostic()
    exact_cases = [case for case in report["cases"] if case["kind"] == "constant"]
    near_cases = [case for case in report["cases"] if case["kind"] == "epsilon_sinusoid"]

    assert report["diagnosis"]["case_count"] == 160
    assert report["configuration"]["fixed_gap"] == [96, 128]
    assert len(exact_cases) == 32
    assert len(near_cases) == 128
    assert report["diagnosis"]["exact_dead_point_count"] == 32
    assert all(case["stream_std"] == 0.0 for case in exact_cases)
    assert all(case["loss"]["total"] > 0.0 for case in exact_cases)
    assert all(case["gradient_rms"] == 0.0 for case in exact_cases)
    assert all(case["gradient_nonzero_count"] == 0 for case in exact_cases)

    expected_grid = {
        (constant, period, mask, epsilon)
        for constant in (-1.0, 0.0, 1.0, 7.0)
        for period in (4, 16, 64, 128)
        for mask in ("full", "fixed_gap")
        for epsilon in (1e-6, 1e-4, 1e-3, 1e-2)
    }
    observed_grid = {
        (
            case["constant_value"],
            case["period_frames"],
            case["mask_mode"],
            case["epsilon"],
        )
        for case in near_cases
    }
    assert observed_grid == expected_grid
    assert {
        case["valid_frame_count"] for case in report["cases"] if case["mask_mode"] == "fixed_gap"
    } == {224}
    assert all(
        case["gradient_safe"] == (case["gradient_rms"] <= SSHEAD_NEAR_COLLAPSE_GRAD_RMS_LIMIT)
        for case in near_cases
    )
    assert report["passed"] is False
    assert report["checks"]["no_exact_constant_dead_points"] is False
    assert report["checks"]["all_finite"] is True
    assert json.loads(json.dumps(report, sort_keys=True)) == report
