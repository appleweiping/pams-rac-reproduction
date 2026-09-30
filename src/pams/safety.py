"""Deterministic, label-free component diagnostics for inferred PAMS logic.

These diagnostics are intentionally separate from training and evaluation.
They do not tune a model or consult dataset labels. The counter gate freezes
the synthetic matrix used to isolate phase/sign sensitivity when the exact
synthetic period is supplied; it does not exercise period estimation, the
encoder, or the head. The SSHead diagnostic exposes exact- and near-collapse
pathologies of the inferred loss.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol, cast

import numpy as np
import torch

from pams.consensus import MultiExpertCounter
from pams.losses import SSHeadLoss

COUNTER_GATE_LENGTH = 256
COUNTER_GATE_COUNTS = (2, 5, 10, 20, 30, 40)
COUNTER_GATE_PHASES = 16
COUNTER_GATE_SIGNS = (-1, 1)
COUNTER_GATE_SHAPES = ("sin", "second_harmonic", "circular_gaussian")
COUNTER_GATE_SECOND_HARMONIC_WEIGHT = 0.5
COUNTER_GATE_GAUSSIAN_WIDTH_CYCLES = 0.07
COUNTER_GATE_GAUSSIAN_WIDTH_RADIANS = 2.0 * math.pi * COUNTER_GATE_GAUSSIAN_WIDTH_CYCLES

COUNTER_GATE_THRESHOLDS: Mapping[str, float] = {
    "max_absolute_error": 1.0,
    "obo": 1.0,
    "nmae": 0.08,
    "shape_nmae": 0.10,
    "phase_range": 1.0,
    "sign_pair_abs_delta": 1.0,
    "mean_sign_pair_abs_delta": 0.90,
    "absolute_signed_bias": 0.25,
}

SSHEAD_DIAGNOSTIC_LENGTH = 256
SSHEAD_DIAGNOSTIC_PERIODS = (4, 16, 64, 128)
SSHEAD_CONSTANT_VALUES = (-1.0, 0.0, 1.0, 7.0)
SSHEAD_NEAR_COLLAPSE_EPSILONS = (1e-6, 1e-4, 1e-3, 1e-2)
SSHEAD_FIXED_GAP = (96, 128)
SSHEAD_MASK_MODES = ("full", "fixed_gap")
SSHEAD_NEAR_COLLAPSE_GRAD_RMS_LIMIT = 10.0
SSHEAD_DEAD_GRAD_RMS_TOLERANCE = 0.0


class _CountOutput(Protocol):
    @property
    def count(self) -> int: ...


class CounterLike(Protocol):
    """Structural interface accepted by :func:`run_counter_sign_phase_gate`."""

    def count(self, period_stream: Any, period_frames: float) -> _CountOutput: ...


class _EstimatedPeriodCounter:
    """Adapter that replaces the exact-period input with the FFT estimate."""

    def __init__(self, counter: MultiExpertCounter) -> None:
        self.counter = counter
        self.estimates: list[dict[str, float]] = []

    def count(self, period_stream: Any, period_frames: float) -> _CountOutput:
        del period_frames
        from pams.period import estimate_period

        stream = np.asarray(period_stream, dtype=np.float64)
        estimate = estimate_period(
            torch.from_numpy(stream),
            minimum=4,
            maximum=128,
        )
        self.estimates.append(
            {
                "period_frames": estimate.period,
                "confidence": estimate.confidence,
            }
        )
        return self.counter.count(
            stream,
            estimate.period,
            period_confidence=estimate.confidence,
        )


def _counter_stream(
    shape: str,
    angular_phase: np.ndarray[Any, np.dtype[np.float64]],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    if shape == "sin":
        return np.sin(angular_phase)
    if shape == "second_harmonic":
        return np.sin(angular_phase) + COUNTER_GATE_SECOND_HARMONIC_WEIGHT * np.sin(
            2.0 * angular_phase + 0.3
        )
    if shape == "circular_gaussian":
        circular_distance = np.angle(np.exp(1j * angular_phase))
        return np.exp(-0.5 * np.square(circular_distance / COUNTER_GATE_GAUSSIAN_WIDTH_RADIANS))
    raise ValueError(f"unknown counter-gate shape: {shape}")


def run_counter_sign_phase_gate(
    counter: CounterLike | None = None,
) -> dict[str, Any]:
    """Run the frozen 576-case counter-only sign/phase matrix.

    Every length-256 stream contains exactly ``N`` periods and the supplied
    period is strictly ``256 / N``. Supplying that exact synthetic period
    deliberately isolates :class:`MultiExpertCounter`; this result is not an
    end-to-end PAMS inference claim. Inputs are never altered to make a
    currently failing component gate pass.
    """

    selected_counter = cast(
        CounterLike,
        counter if counter is not None else MultiExpertCounter(),
    )
    frame_positions = np.arange(COUNTER_GATE_LENGTH, dtype=np.float64) / float(COUNTER_GATE_LENGTH)
    details: list[dict[str, Any]] = []

    for shape in COUNTER_GATE_SHAPES:
        for target_count in COUNTER_GATE_COUNTS:
            period_frames = COUNTER_GATE_LENGTH / target_count
            for phase_index in range(COUNTER_GATE_PHASES):
                phase_radians = 2.0 * math.pi * phase_index / COUNTER_GATE_PHASES
                angular_phase = 2.0 * math.pi * target_count * frame_positions + phase_radians
                unsigned_stream = _counter_stream(shape, angular_phase)
                for sign in COUNTER_GATE_SIGNS:
                    result = selected_counter.count(sign * unsigned_stream, period_frames)
                    prediction = int(result.count)
                    signed_error = prediction - target_count
                    details.append(
                        {
                            "shape": shape,
                            "target_count": target_count,
                            "phase_index": phase_index,
                            "phase_radians": phase_radians,
                            "sign": sign,
                            "period_frames": period_frames,
                            "prediction": prediction,
                            "signed_error": signed_error,
                            "absolute_error": abs(signed_error),
                            "normalized_absolute_error": abs(signed_error) / target_count,
                            "within_one": abs(signed_error) <= 1,
                        }
                    )

    absolute_errors = np.asarray(
        [row["absolute_error"] for row in details],
        dtype=np.float64,
    )
    normalized_errors = np.asarray(
        [row["normalized_absolute_error"] for row in details],
        dtype=np.float64,
    )
    signed_errors = np.asarray(
        [row["signed_error"] for row in details],
        dtype=np.float64,
    )
    shape_nmae = {
        shape: float(
            np.mean([row["normalized_absolute_error"] for row in details if row["shape"] == shape])
        )
        for shape in COUNTER_GATE_SHAPES
    }

    phase_ranges: list[int] = []
    sign_pair_deltas: list[int] = []
    sign_pair_signed_deltas: list[int] = []
    for shape in COUNTER_GATE_SHAPES:
        for target_count in COUNTER_GATE_COUNTS:
            for sign in COUNTER_GATE_SIGNS:
                phase_predictions = [
                    row["prediction"]
                    for row in details
                    if row["shape"] == shape
                    and row["target_count"] == target_count
                    and row["sign"] == sign
                ]
                phase_ranges.append(max(phase_predictions) - min(phase_predictions))
            for phase_index in range(COUNTER_GATE_PHASES):
                sign_predictions = [
                    row["prediction"]
                    for row in details
                    if row["shape"] == shape
                    and row["target_count"] == target_count
                    and row["phase_index"] == phase_index
                ]
                sign_pair_deltas.append(abs(sign_predictions[0] - sign_predictions[1]))
                # COUNTER_GATE_SIGNS is frozen as (-1, +1), so this is
                # explicitly pred(+1) minus pred(-1).
                sign_pair_signed_deltas.append(sign_predictions[1] - sign_predictions[0])

    metrics: dict[str, Any] = {
        "case_count": len(details),
        "max_absolute_error": float(np.max(absolute_errors)),
        "obo": float(np.mean(absolute_errors <= 1.0)),
        "nmae": float(np.mean(normalized_errors)),
        "shape_nmae": shape_nmae,
        "max_phase_range": float(max(phase_ranges)),
        "max_sign_pair_abs_delta": float(max(sign_pair_deltas)),
        "mean_sign_pair_abs_delta": float(np.mean(sign_pair_deltas)),
        "sign_pair_signed_bias": float(np.mean(sign_pair_signed_deltas)),
        "prediction_signed_error_mean": float(np.mean(signed_errors)),
    }
    checks = {
        "max_absolute_error": (
            metrics["max_absolute_error"] <= COUNTER_GATE_THRESHOLDS["max_absolute_error"]
        ),
        "obo": metrics["obo"] >= COUNTER_GATE_THRESHOLDS["obo"],
        "nmae": metrics["nmae"] <= COUNTER_GATE_THRESHOLDS["nmae"],
        "shape_nmae": all(
            value <= COUNTER_GATE_THRESHOLDS["shape_nmae"] for value in shape_nmae.values()
        ),
        "phase_range": (metrics["max_phase_range"] <= COUNTER_GATE_THRESHOLDS["phase_range"]),
        "sign_pair_abs_delta": (
            metrics["max_sign_pair_abs_delta"] <= COUNTER_GATE_THRESHOLDS["sign_pair_abs_delta"]
        ),
        "mean_sign_pair_abs_delta": (
            metrics["mean_sign_pair_abs_delta"]
            <= COUNTER_GATE_THRESHOLDS["mean_sign_pair_abs_delta"]
        ),
        "absolute_signed_bias": (
            abs(metrics["sign_pair_signed_bias"]) <= COUNTER_GATE_THRESHOLDS["absolute_signed_bias"]
        ),
    }
    return {
        "schema_version": 1,
        "gate": "counter-sign-phase",
        "scope": "multi_expert_counter_component_only",
        "period_source": "exact_synthetic_period",
        "end_to_end_pams": False,
        "configuration": {
            "length": COUNTER_GATE_LENGTH,
            "counts": list(COUNTER_GATE_COUNTS),
            "phase_count": COUNTER_GATE_PHASES,
            "signs": list(COUNTER_GATE_SIGNS),
            "shapes": list(COUNTER_GATE_SHAPES),
            "second_harmonic_weight": COUNTER_GATE_SECOND_HARMONIC_WEIGHT,
            "circular_gaussian_width_cycles": COUNTER_GATE_GAUSSIAN_WIDTH_CYCLES,
            "circular_gaussian_width_radians": COUNTER_GATE_GAUSSIAN_WIDTH_RADIANS,
        },
        "thresholds": dict(COUNTER_GATE_THRESHOLDS),
        "metrics": metrics,
        "checks": checks,
        "passed": all(checks.values()),
        "details": details,
    }


def run_period_counter_sign_phase_gate() -> dict[str, Any]:
    """Run FFT/autocorrelation period estimation plus multi-expert counting.

    This remains a deterministic component-chain gate, not an end-to-end
    learned-model test: synthetic streams bypass the encoder and Period Head.
    Unlike :func:`run_counter_sign_phase_gate`, no exact period is supplied to
    the code under test.
    """

    adapter = _EstimatedPeriodCounter(MultiExpertCounter())
    report = run_counter_sign_phase_gate(adapter)
    period_relative_errors: list[float] = []
    confidences: list[float] = []
    for row, estimate in zip(report["details"], adapter.estimates, strict=True):
        exact_period = float(row["period_frames"])
        estimated_period = float(estimate["period_frames"])
        confidence = float(estimate["confidence"])
        row["exact_period_frames"] = exact_period
        row["estimated_period_frames"] = estimated_period
        row["period_confidence"] = confidence
        row["period_relative_error"] = abs(estimated_period - exact_period) / exact_period
        del row["period_frames"]
        period_relative_errors.append(float(row["period_relative_error"]))
        confidences.append(confidence)

    maximum_period_relative_error = max(period_relative_errors)
    minimum_period_confidence = min(confidences)
    report["schema_version"] = 1
    report["gate"] = "period-counter-sign-phase"
    report["scope"] = "fft_autocorrelation_and_multi_expert_components"
    report["period_source"] = "estimated_from_stream"
    report["end_to_end_pams"] = False
    report["thresholds"]["max_period_relative_error"] = 1e-4
    report["thresholds"]["min_period_confidence"] = 0.0
    report["metrics"]["max_period_relative_error"] = maximum_period_relative_error
    report["metrics"]["min_period_confidence"] = minimum_period_confidence
    report["checks"]["max_period_relative_error"] = maximum_period_relative_error <= 1e-4
    report["checks"]["min_period_confidence"] = minimum_period_confidence > 0.0
    report["passed"] = all(bool(value) for value in report["checks"].values())
    return report


def _sshead_case(
    *,
    kind: str,
    constant_value: float,
    period_frames: int,
    mask_mode: str,
    epsilon: float,
    objective: SSHeadLoss,
) -> dict[str, Any]:
    time = torch.arange(SSHEAD_DIAGNOSTIC_LENGTH, dtype=torch.float64)
    values = torch.full(
        (SSHEAD_DIAGNOSTIC_LENGTH,),
        constant_value,
        dtype=torch.float64,
    )
    if epsilon:
        values = values + epsilon * torch.sin(2.0 * math.pi * time / period_frames)
    valid_mask = torch.ones(SSHEAD_DIAGNOSTIC_LENGTH, dtype=torch.bool)
    if mask_mode == "fixed_gap":
        gap_start, gap_stop = SSHEAD_FIXED_GAP
        valid_mask[gap_start:gap_stop] = False
    elif mask_mode != "full":
        raise ValueError(f"unknown SSHead mask mode: {mask_mode}")

    stream = values.detach().requires_grad_(True)
    output = objective.compute(
        stream,
        torch.tensor(float(period_frames), dtype=torch.float64),
        valid_mask,
    )
    output.total.backward()
    if stream.grad is None:
        raise RuntimeError("SSHead diagnostic did not produce a stream gradient")
    gradient = stream.grad.detach()
    return {
        "kind": kind,
        "constant_value": constant_value,
        "period_frames": period_frames,
        "mask_mode": mask_mode,
        "epsilon": epsilon,
        "valid_frame_count": int(valid_mask.sum()),
        "stream_std": float(stream.detach()[valid_mask].std(unbiased=False)),
        "loss": {
            "total": float(output.total.detach()),
            "cycle": float(output.cycle.detach()),
            "spectral": float(output.spectral.detach()),
            "variance": float(output.variance.detach()),
            "smoothness": float(output.smoothness.detach()),
        },
        "gradient_rms": float(torch.sqrt(torch.mean(gradient.square()))),
        "gradient_max_abs": float(gradient.abs().max()),
        "gradient_nonzero_count": int(torch.count_nonzero(gradient)),
        "all_finite": bool(
            torch.isfinite(output.total).item() and torch.isfinite(gradient).all().item()
        ),
    }


def run_sshead_collapse_diagnostic(
    objective: SSHeadLoss | None = None,
) -> dict[str, Any]:
    """Run the frozen 160-case exact/near-collapse diagnostic.

    This is not a claim that the current objective is safe. Exact constant
    streams are dead points when they retain positive loss but have zero
    gradient. Near-constant cases fail individually when gradient RMS exceeds
    the pre-registered limit of 10.
    """

    selected_objective = objective if objective is not None else SSHeadLoss()
    cases: list[dict[str, Any]] = []
    for constant_value in SSHEAD_CONSTANT_VALUES:
        for period_frames in SSHEAD_DIAGNOSTIC_PERIODS:
            for mask_mode in SSHEAD_MASK_MODES:
                exact = _sshead_case(
                    kind="constant",
                    constant_value=constant_value,
                    period_frames=period_frames,
                    mask_mode=mask_mode,
                    epsilon=0.0,
                    objective=selected_objective,
                )
                exact["dead_point"] = bool(
                    exact["stream_std"] == 0.0
                    and exact["loss"]["total"] > 0.0
                    and exact["gradient_rms"] <= SSHEAD_DEAD_GRAD_RMS_TOLERANCE
                )
                exact["gradient_safe"] = True
                cases.append(exact)
                for epsilon in SSHEAD_NEAR_COLLAPSE_EPSILONS:
                    near = _sshead_case(
                        kind="epsilon_sinusoid",
                        constant_value=constant_value,
                        period_frames=period_frames,
                        mask_mode=mask_mode,
                        epsilon=epsilon,
                        objective=selected_objective,
                    )
                    near["dead_point"] = False
                    near["gradient_safe"] = bool(
                        near["gradient_rms"] <= SSHEAD_NEAR_COLLAPSE_GRAD_RMS_LIMIT
                    )
                    cases.append(near)

    exact_cases = [case for case in cases if case["kind"] == "constant"]
    near_cases = [case for case in cases if case["kind"] == "epsilon_sinusoid"]
    exact_dead_point_count = sum(bool(case["dead_point"]) for case in exact_cases)
    near_collapse_unsafe_count = sum(not bool(case["gradient_safe"]) for case in near_cases)
    all_finite = all(bool(case["all_finite"]) for case in cases)
    checks = {
        "no_exact_constant_dead_points": exact_dead_point_count == 0,
        "all_near_collapse_gradients_safe": near_collapse_unsafe_count == 0,
        "all_finite": all_finite,
    }
    return {
        "schema_version": 1,
        "gate": "sshead-collapse",
        "configuration": {
            "length": SSHEAD_DIAGNOSTIC_LENGTH,
            "periods": list(SSHEAD_DIAGNOSTIC_PERIODS),
            "constant_values": list(SSHEAD_CONSTANT_VALUES),
            "epsilons": list(SSHEAD_NEAR_COLLAPSE_EPSILONS),
            "mask_modes": list(SSHEAD_MASK_MODES),
            "fixed_gap": list(SSHEAD_FIXED_GAP),
        },
        "thresholds": {
            "near_collapse_gradient_rms_max": SSHEAD_NEAR_COLLAPSE_GRAD_RMS_LIMIT,
            "dead_gradient_rms_tolerance": SSHEAD_DEAD_GRAD_RMS_TOLERANCE,
        },
        "diagnosis": {
            "case_count": len(cases),
            "exact_case_count": len(exact_cases),
            "near_collapse_case_count": len(near_cases),
            "exact_dead_point_count": exact_dead_point_count,
            "near_collapse_unsafe_count": near_collapse_unsafe_count,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "cases": cases,
    }
