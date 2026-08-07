"""Recurrence-gated, embedding-conditioned carrier action curves.

This inference-only readout keeps the frozen v16 encoder and its existing
full-vector embedding-velocity ACF period estimate.  Unlike the v1 scalar
projection, it does not expose an unfiltered projection of the complete
embedding trajectory to the peak experts.  The frozen period supplies a
stable analytic carrier, while real embeddings determine both its phase and
the continuous temporal spans in which the carrier is observable.

Local repeatness is a contrast between recurrence at one period and the two
off-phase lags at half and one-and-a-half periods.  Fractional dense-frame
lags are interpolated without compacting or bridging invalid frames.  Feature
weights use phase agreement across alternating target-period cycles, so the
same chance Fourier coefficient is not both selected and accepted as its own
recurrence evidence.  Every non-constant feature retains a uniform base
weight; cross-cycle agreement supplies only a bounded uplift.  A
constant/static trajectory therefore has zero contrast, while a genuinely
periodic trajectory has positive contrast.  The returned active mask is the
only mask used by both the three peak experts and the analytic reference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from pams.embedding_curve import (
    _dominant_harmonic_direction,
    _validated_inputs,
)
from pams.period import estimate_period_from_embedding_velocity_vectors

if TYPE_CHECKING:
    from pams.config import PAMSConfig


# Frozen before any train337 recurrence-carrier execution.  These constants
# are scale-free and are covered by synthetic, shuffle, and null tests.
RECURRENCE_SMOOTH_SIGMA_PERIODS = 0.25
RECURRENCE_ON_CONTRAST = 0.12
RECURRENCE_OFF_CONTRAST = 0.04
RECURRENCE_MINIMUM_SPAN_PERIODS = 1.5
RECURRENCE_SPAN_PADDING_PERIODS = 0.25
RECURRENCE_AMPLITUDE_FLOOR = 0.75


@dataclass(frozen=True, slots=True)
class RecurrenceCarrierBatch:
    """Auditable outputs of the recurrence-gated carrier readout."""

    curves: Tensor
    active_masks: Tensor
    recurrence_scores: Tensor
    recurrence_gates: Tensor
    periods: Tensor
    period_confidences: Tensor
    curve_standard_deviations: Tensor
    raw_curve_standard_deviations: Tensor
    harmonic_energy_fractions: Tensor
    active_support_fractions: Tensor
    recurrence_gate_energies: Tensor
    phase_offsets: Tensor
    active_reference_counts: Tensor
    full_timeline_reference_counts: Tensor
    available: Tensor

    def __post_init__(self) -> None:
        if self.curves.ndim != 2:
            raise ValueError("curves must have shape [batch, time]")
        batch, time = self.curves.shape
        for name in ("active_masks", "recurrence_scores", "recurrence_gates"):
            value = getattr(self, name)
            if value.shape != (batch, time):
                raise ValueError(f"{name} must match curves")
        if self.active_masks.dtype != torch.bool:
            raise TypeError("active_masks must be boolean")
        for name in (
            "periods",
            "period_confidences",
            "curve_standard_deviations",
            "raw_curve_standard_deviations",
            "harmonic_energy_fractions",
            "active_support_fractions",
            "recurrence_gate_energies",
            "phase_offsets",
            "active_reference_counts",
            "full_timeline_reference_counts",
            "available",
        ):
            if getattr(self, name).shape != (batch,):
                raise ValueError(f"{name} must have shape [batch]")
        if self.available.dtype != torch.bool:
            raise TypeError("available must be boolean")
        if self.active_reference_counts.dtype != torch.long:
            raise TypeError("active_reference_counts must use torch.long")
        if self.full_timeline_reference_counts.dtype != torch.long:
            raise TypeError("full_timeline_reference_counts must use torch.long")


def validate_recurrence_carrier_compatibility(
    upstream_config: PAMSConfig,
    candidate_config: PAMSConfig,
) -> None:
    """Require exact v16 semantics except for the readout marker."""

    if upstream_config.readout.action_curve_source != "learned_period_head":
        raise ValueError("upstream encoder config must use the historical readout")
    if candidate_config.readout.action_curve_source != "embedding_recurrence_carrier":
        raise ValueError("candidate config must opt into embedding_recurrence_carrier")
    upstream = upstream_config.model_dump(mode="json")
    candidate = candidate_config.model_dump(mode="json")
    upstream_source = upstream.pop("readout")["action_curve_source"]
    candidate_source = candidate.pop("readout")["action_curve_source"]
    if (
        upstream_source != "learned_period_head"
        or candidate_source != "embedding_recurrence_carrier"
    ):
        raise AssertionError("validated action-curve sources changed unexpectedly")
    if upstream != candidate:
        raise ValueError(
            "upstream and candidate configs may differ only in "
            "readout.action_curve_source"
        )


def _valid_runs(mask: Tensor) -> tuple[tuple[int, int], ...]:
    indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
    if not indices:
        return ()
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    previous = start
    for raw in indices[1:]:
        index = int(raw)
        if index != previous + 1:
            runs.append((start, previous + 1))
            start = index
        previous = index
    runs.append((start, previous + 1))
    return tuple(runs)


def _fractional_similarity(
    states: Tensor,
    valid: Tensor,
    lag: float,
) -> tuple[Tensor, Tensor]:
    """Return symmetric fractional-lag cosine similarity and availability."""

    if states.ndim != 2 or valid.shape != states.shape[:1]:
        raise ValueError("fractional similarity inputs have incompatible shapes")
    if not math.isfinite(lag) or lag <= 0.0:
        raise ValueError("fractional similarity lag must be finite and positive")
    time = states.shape[0]
    positions = torch.arange(time, dtype=states.dtype, device=states.device)
    accumulated = torch.zeros(time, dtype=states.dtype, device=states.device)
    weights = torch.zeros_like(accumulated)
    for direction in (-1.0, 1.0):
        shifted = positions + direction * lag
        inside = (shifted >= 0.0) & (shifted <= float(time - 1))
        safe = shifted.clamp(0.0, float(time - 1))
        lower = torch.floor(safe).to(dtype=torch.long)
        upper = torch.ceil(safe).to(dtype=torch.long)
        fraction = safe - lower.to(dtype=safe.dtype)
        exact = upper == lower
        source_valid = valid[lower] & (exact | valid[upper])
        pair_valid = valid & inside & source_valid
        interpolated = (
            states[lower] * (1.0 - fraction).unsqueeze(1)
            + states[upper] * fraction.unsqueeze(1)
        )
        interpolated = F.normalize(interpolated, p=2, dim=1, eps=1e-12)
        similarity = (states * interpolated).sum(dim=1).clamp(-1.0, 1.0)
        accumulated = accumulated + similarity * pair_valid.to(states.dtype)
        weights = weights + pair_valid.to(states.dtype)
    available = weights > 0.0
    values = torch.where(
        available,
        accumulated / weights.clamp_min(1.0),
        torch.zeros_like(accumulated),
    )
    return values, available


def _gaussian_kernel(
    sigma: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    sigma = max(float(sigma), 1e-6)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-0.5 * (offsets / sigma).square())
    return kernel / kernel.sum().clamp_min(1e-12)


def _smooth_available_runs(
    values: Tensor,
    available: Tensor,
    valid: Tensor,
    *,
    sigma: float,
) -> Tensor:
    """Mask-normalized Gaussian smoothing without crossing invalid holes."""

    smoothed = torch.zeros_like(values)
    kernel = _gaussian_kernel(sigma, dtype=values.dtype, device=values.device)
    padding = kernel.numel() // 2
    for start, stop in _valid_runs(valid):
        run_values = values[start:stop]
        run_available = available[start:stop].to(values.dtype)
        numerator = F.conv1d(
            (run_values * run_available).reshape(1, 1, -1),
            kernel.reshape(1, 1, -1),
            padding=padding,
        ).reshape(-1)
        denominator = F.conv1d(
            run_available.reshape(1, 1, -1),
            kernel.reshape(1, 1, -1),
            padding=padding,
        ).reshape(-1)
        smoothed[start:stop] = torch.where(
            denominator > 1e-6,
            numerator / denominator.clamp_min(1e-6),
            torch.zeros_like(numerator),
        )
    return smoothed


def _active_mask_from_score(score: Tensor, valid: Tensor, period: float) -> Tensor:
    """Apply frozen hysteresis, minimum duration, and in-run padding."""

    active = torch.zeros_like(valid)
    minimum = max(3, int(math.ceil(RECURRENCE_MINIMUM_SPAN_PERIODS * period)))
    padding = max(0, int(round(RECURRENCE_SPAN_PADDING_PERIODS * period)))
    for run_start, run_stop in _valid_runs(valid):
        low = score[run_start:run_stop] >= RECURRENCE_OFF_CONTRAST
        high = score[run_start:run_stop] >= RECURRENCE_ON_CONTRAST
        for local_start, local_stop in _valid_runs(low):
            if local_stop - local_start < minimum or not bool(
                high[local_start:local_stop].any()
            ):
                continue
            start = max(run_start, run_start + local_start - padding)
            stop = min(run_stop, run_start + local_stop + padding)
            active[start:stop] = True
    return active & valid


def _analytic_peak_reference(carrier: Tensor, active: Tensor) -> int:
    """Count deterministic carrier maxima strictly inside active spans."""

    total = 0
    for start, stop in _valid_runs(active):
        working = carrier[start:stop]
        if working.numel() < 3:
            continue
        # The asymmetric tie break counts a two-sample plateau once.
        peaks = (working[1:-1] >= working[:-2]) & (working[1:-1] > working[2:])
        total += int(peaks.sum())
    return total


def _align_active_spans_to_carrier_troughs(
    active: Tensor,
    carrier: Tensor,
    period: float,
) -> Tensor:
    """Shrink recurrence spans to carrier troughs for boundary-stable peaks."""

    aligned = torch.zeros_like(active)
    search = max(2, int(math.ceil(0.75 * period)))
    minimum = max(3, int(math.ceil(period)))
    for start, stop in _valid_runs(active):
        if stop - start < minimum:
            continue
        left_stop = min(stop, start + search)
        right_start = max(start, stop - search)
        aligned_start = start + int(torch.argmin(carrier[start:left_stop]))
        aligned_stop_index = right_start + int(
            torch.argmin(carrier[right_start:stop])
        )
        aligned_stop = aligned_stop_index + 1
        if aligned_stop - aligned_start >= minimum:
            aligned[aligned_start:aligned_stop] = True
    return aligned


def _masked_harmonic_coefficients(
    centered: Tensor,
    mask: Tensor,
    period: float,
) -> Tensor | None:
    """Return fixed-axis target-harmonic coefficients on one temporal fold."""

    if centered.ndim != 2 or mask.shape != centered.shape[:1]:
        raise ValueError("harmonic coefficient inputs have incompatible shapes")
    selected_total = int(mask.sum())
    if selected_total < 3:
        return None
    time = centered.shape[0]
    dense_time = torch.arange(
        time,
        dtype=centered.dtype,
        device=centered.device,
    )
    omega = (2.0 * math.pi) / period
    cosine = torch.cos(omega * dense_time)
    sine = torch.sin(omega * dense_time)
    mask_values = mask.to(centered.dtype)
    cosine = (cosine - cosine[mask].mean()) * mask_values
    sine = (sine - sine[mask].mean()) * mask_values
    cosine_norm = cosine.norm()
    sine_norm = sine.norm()
    if float(cosine_norm) <= 1e-12 or float(sine_norm) <= 1e-12:
        return None
    safe = torch.where(mask.unsqueeze(1), centered, torch.zeros_like(centered))
    fold_mean = safe.sum(dim=0) / float(selected_total)
    fold_centered = torch.where(
        mask.unsqueeze(1),
        centered - fold_mean,
        torch.zeros_like(centered),
    )
    return torch.stack(
        (
            (cosine / cosine_norm) @ fold_centered,
            (sine / sine_norm) @ fold_centered,
        ),
        dim=0,
    )


def _cross_cycle_harmonic_feature_weights(
    centered: Tensor,
    valid: Tensor,
    period: float,
) -> Tensor:
    """Give only a bounded uplift to phase-stable target-frequency features.

    Global target-frequency coefficients are data-dependent.  Reusing their
    unbounded magnitudes as recurrence weights lets a shuffled trajectory's
    largest chance coefficient select the very coordinate on which it is
    subsequently judged.  Here coefficients are estimated independently on
    alternating target-period cycles.  Only positive phase agreement between
    the two folds earns an uplift, and the all-feature unit baseline remains.
    Consequently, recurrence is still evaluated in the full diagonal-whitened
    representation rather than in a chance-selected low-dimensional slice.
    """

    if centered.ndim != 2 or valid.shape != centered.shape[:1]:
        raise ValueError("cross-cycle support inputs have incompatible shapes")
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("cross-cycle support period must be finite and positive")
    dimension = centered.shape[1]
    uniform = torch.ones(
        dimension,
        dtype=centered.dtype,
        device=centered.device,
    )
    dense_time = torch.arange(
        centered.shape[0],
        dtype=centered.dtype,
        device=centered.device,
    )
    cycle_index = torch.floor(dense_time / period).to(dtype=torch.long)
    first_mask = valid & ((cycle_index % 2) == 0)
    second_mask = valid & ((cycle_index % 2) == 1)
    first = _masked_harmonic_coefficients(centered, first_mask, period)
    second = _masked_harmonic_coefficients(centered, second_mask, period)
    if first is None or second is None:
        return uniform
    coherent_energy = (first * second).sum(dim=0).clamp_min(0.0)
    maximum = coherent_energy.max()
    if not bool(torch.isfinite(maximum)) or float(maximum) <= 1e-12:
        return uniform
    # Similarity normalizes states after weighting.  sqrt(1 + support) gives
    # a parameter-free [1, sqrt(2)] uplift and cannot collapse the effective
    # feature dimension to a single chance Fourier coordinate.
    normalized = (coherent_energy / maximum).clamp(0.0, 1.0)
    return torch.sqrt(1.0 + normalized)


def _recurrence_score(
    centered: Tensor,
    valid: Tensor,
    period: float,
    harmonic_feature_weights: Tensor,
) -> Tensor:
    valid_total = int(valid.sum())
    feature_rms = (
        centered.square().sum(dim=0) / float(max(valid_total, 1))
    ).sqrt()
    whitened = torch.where(
        feature_rms.unsqueeze(0) > 1e-6,
        centered / feature_rms.clamp_min(1e-6).unsqueeze(0),
        torch.zeros_like(centered),
    )
    if harmonic_feature_weights.shape != centered.shape[1:]:
        raise ValueError("harmonic feature weights must match embedding dimension")
    if (
        not bool(torch.isfinite(harmonic_feature_weights).all())
        or bool((harmonic_feature_weights <= 0.0).any())
    ):
        raise ValueError("harmonic feature weights must be finite and positive")
    states = F.normalize(
        whitened * harmonic_feature_weights.unsqueeze(0),
        p=2,
        dim=1,
        eps=1e-12,
    )
    nonzero = states.square().sum(dim=1) > 1e-8
    state_valid = valid & nonzero
    positive, positive_available = _fractional_similarity(states, state_valid, period)
    half, half_available = _fractional_similarity(states, state_valid, 0.5 * period)
    three_half, three_half_available = _fractional_similarity(
        states,
        state_valid,
        1.5 * period,
    )
    negative_weights = half_available.to(states.dtype) + three_half_available.to(
        states.dtype
    )
    negative = torch.where(
        negative_weights > 0.0,
        (
            half * half_available.to(states.dtype)
            + three_half * three_half_available.to(states.dtype)
        )
        / negative_weights.clamp_min(1.0),
        torch.zeros_like(half),
    )
    available = positive_available & (negative_weights > 0.0)
    contrast = torch.where(available, positive - negative, torch.zeros_like(positive))
    return _smooth_available_runs(
        contrast,
        available,
        valid,
        sigma=RECURRENCE_SMOOTH_SIGMA_PERIODS * period,
    )


def build_recurrence_carrier_curves(
    embeddings: Tensor,
    periods: Tensor,
    valid_mask: Tensor | None = None,
    *,
    timeline_lengths: Tensor | None = None,
    period_confidences: Tensor | None = None,
) -> RecurrenceCarrierBatch:
    """Build locally gated carriers using only frozen embeddings and period."""

    values, period_values, valid, lengths, _ = _validated_inputs(
        embeddings,
        periods,
        valid_mask,
        timeline_lengths,
    )
    if period_confidences is None:
        confidence_values = torch.zeros_like(period_values)
    else:
        confidence_values = (
            period_confidences.reshape(1)
            if period_confidences.ndim == 0
            else period_confidences
        )
        if confidence_values.shape != period_values.shape:
            raise ValueError("period_confidences must match periods")
        confidence_values = confidence_values.to(
            device=period_values.device,
            dtype=period_values.dtype,
        )
        if (
            not bool(torch.isfinite(confidence_values).all())
            or bool((confidence_values < 0.0).any())
            or bool((confidence_values > 1.0).any())
        ):
            raise ValueError("period_confidences must be finite and in [0, 1]")
    work = values.detach().to(dtype=torch.float32)
    batch, padded_time, dimension = work.shape
    curves = torch.zeros((batch, padded_time), dtype=work.dtype, device=work.device)
    active_masks = torch.zeros(
        (batch, padded_time), dtype=torch.bool, device=work.device
    )
    recurrence_scores = torch.zeros_like(curves)
    recurrence_gates = torch.zeros_like(curves)
    normalized_stds = torch.zeros(batch, dtype=work.dtype, device=work.device)
    raw_stds = torch.zeros_like(normalized_stds)
    harmonic_fractions = torch.zeros_like(normalized_stds)
    support_fractions = torch.zeros_like(normalized_stds)
    gate_energies = torch.zeros_like(normalized_stds)
    phase_offsets = torch.zeros_like(normalized_stds)
    active_references = torch.zeros(batch, dtype=torch.long, device=work.device)
    full_references = torch.zeros_like(active_references)
    available = torch.zeros(batch, dtype=torch.bool, device=work.device)

    for index in range(batch):
        length = int(lengths[index])
        sample_valid = valid[index, :length]
        valid_total = int(sample_valid.sum())
        period = float(period_values[index])
        full_references[index] = int(math.floor(length / period))
        if valid_total < 3:
            continue
        sample = work[index, :length]
        if not bool(torch.isfinite(sample[sample_valid]).all()):
            raise ValueError("valid embeddings must contain only finite values")
        safe = torch.where(sample_valid.unsqueeze(1), sample, torch.zeros_like(sample))
        mean = safe.sum(dim=0) / float(valid_total)
        centered = torch.where(
            sample_valid.unsqueeze(1),
            sample - mean,
            torch.zeros_like(sample),
        )

        dense_time = torch.arange(length, dtype=work.dtype, device=work.device)
        omega = (2.0 * math.pi) / period_values[index]
        cosine = torch.cos(omega * dense_time)
        sine = torch.sin(omega * dense_time)
        mask_values = sample_valid.to(work.dtype)
        cosine_basis = (cosine - cosine[sample_valid].mean()) * mask_values
        sine_basis = (sine - sine[sample_valid].mean()) * mask_values
        cosine_norm = cosine_basis.norm()
        if float(cosine_norm) <= 1e-12:
            continue
        cosine_basis = cosine_basis / cosine_norm
        sine_basis = sine_basis - torch.dot(sine_basis, cosine_basis) * cosine_basis
        sine_norm = sine_basis.norm()
        if float(sine_norm) <= 1e-12:
            continue
        sine_basis = sine_basis / sine_norm
        coefficients = torch.stack(
            (cosine_basis @ centered, sine_basis @ centered),
            dim=0,
        )
        direction, direction_available = _dominant_harmonic_direction(coefficients)
        if not direction_available or direction.shape != (dimension,):
            continue
        projection = centered @ direction
        phase_cosine = torch.dot(projection, cosine * mask_values)
        phase_sine = torch.dot(projection, sine * mask_values)
        phase = torch.atan2(-phase_sine, phase_cosine)
        carrier = torch.cos(omega * dense_time + phase)

        total_energy = centered.square().sum()
        harmonic_energy = coefficients.square().sum()
        harmonic_fractions[index] = (
            harmonic_energy / total_energy.clamp_min(1e-12)
        ).clamp(0.0, 1.0)
        phase_offsets[index] = phase
        harmonic_feature_weights = _cross_cycle_harmonic_feature_weights(
            centered,
            sample_valid,
            period,
        )
        score = _recurrence_score(
            centered,
            sample_valid,
            period,
            harmonic_feature_weights,
        )
        active = _active_mask_from_score(score, sample_valid, period)
        active = _align_active_spans_to_carrier_troughs(active, carrier, period)
        if not bool(active.any()):
            recurrence_scores[index, :length] = score
            continue
        gate = (score.clamp(0.0, 2.0) / 2.0).masked_fill(~active, 0.0)
        amplitude = RECURRENCE_AMPLITUDE_FLOOR + (
            1.0 - RECURRENCE_AMPLITUDE_FLOOR
        ) * gate
        raw_curve = (carrier * amplitude).masked_fill(~active, 0.0)
        raw_curve = (
            raw_curve - raw_curve[active].mean()
        ).masked_fill(~active, 0.0)
        raw_std = raw_curve[active].square().mean().sqrt()
        if not bool(torch.isfinite(raw_std)) or float(raw_std) <= 1e-12:
            continue
        curve = (raw_curve / raw_std).masked_fill(~active, 0.0)

        curves[index, :length] = curve
        active_masks[index, :length] = active
        recurrence_scores[index, :length] = score
        recurrence_gates[index, :length] = gate
        normalized_stds[index] = curve[active].square().mean().sqrt()
        raw_stds[index] = raw_std
        support_fractions[index] = active.sum().to(work.dtype) / float(valid_total)
        gate_energies[index] = gate[sample_valid].square().mean()
        active_references[index] = _analytic_peak_reference(carrier, active)
        available[index] = True

    curves = curves.masked_fill(~active_masks, 0.0)
    recurrence_scores = recurrence_scores.masked_fill(~valid, 0.0)
    recurrence_gates = recurrence_gates.masked_fill(~active_masks, 0.0)
    return RecurrenceCarrierBatch(
        curves=curves,
        active_masks=active_masks,
        recurrence_scores=recurrence_scores,
        recurrence_gates=recurrence_gates,
        periods=period_values,
        period_confidences=confidence_values,
        curve_standard_deviations=normalized_stds,
        raw_curve_standard_deviations=raw_stds,
        harmonic_energy_fractions=harmonic_fractions,
        active_support_fractions=support_fractions,
        recurrence_gate_energies=gate_energies,
        phase_offsets=phase_offsets,
        active_reference_counts=active_references,
        full_timeline_reference_counts=full_references,
        available=available,
    )


def estimate_recurrence_carrier_curves(
    embeddings: Tensor,
    *,
    minimum_period: int,
    maximum_period: int,
    valid_mask: Tensor | None = None,
    timeline_lengths: Tensor | None = None,
) -> RecurrenceCarrierBatch:
    """Estimate the frozen ACF period and its recurrence-gated carrier."""

    periods, confidences = estimate_period_from_embedding_velocity_vectors(
        embeddings.detach(),
        minimum=minimum_period,
        maximum=maximum_period,
        valid_mask=valid_mask,
    )
    return build_recurrence_carrier_curves(
        embeddings.detach(),
        periods,
        valid_mask,
        timeline_lengths=timeline_lengths,
        period_confidences=confidences,
    )
