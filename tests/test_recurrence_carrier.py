from __future__ import annotations

import math

import pytest
import torch

from pams.config import PAMSConfig
from pams.consensus import MultiExpertCounter
from pams.recurrence_carrier import (
    _cross_cycle_harmonic_feature_weights,
    build_recurrence_carrier_curves,
    validate_recurrence_carrier_compatibility,
)


def _localized_embeddings(
    *,
    frames: int = 160,
    period: float = 16.0,
    active_start: int = 32,
    active_stop: int = 128,
    phase: float = 0.0,
) -> torch.Tensor:
    time = torch.arange(frames, dtype=torch.float32)
    values = torch.zeros((frames, 8), dtype=torch.float32)
    values[:, 6] = 1.0
    values[:active_start, 4] = 1.5
    values[active_stop:, 5] = -1.5
    active = (time >= active_start) & (time < active_stop)
    angle = 2.0 * math.pi * time / period + phase
    values[active, 0] = torch.cos(angle[active])
    values[active, 1] = torch.sin(angle[active])
    values[active, 2] = 0.5 * torch.cos(2.0 * angle[active] + 0.2)
    values[active, 3] = 0.5 * torch.sin(2.0 * angle[active] + 0.2)
    values[:, 7] = 0.002 * time
    return values


def _build(
    embeddings: torch.Tensor,
    *,
    period: float,
    mask: torch.Tensor | None = None,
):
    if mask is None:
        mask = torch.ones(embeddings.shape[0], dtype=torch.bool)
    return build_recurrence_carrier_curves(
        embeddings,
        torch.tensor([period]),
        mask,
        timeline_lengths=torch.tensor([embeddings.shape[0]]),
        period_confidences=torch.tensor([0.8]),
    )


def test_recurrence_carrier_compatibility_allows_only_readout_marker() -> None:
    upstream = PAMSConfig()
    payload = upstream.model_dump()
    payload["readout"]["action_curve_source"] = "embedding_recurrence_carrier"
    candidate = PAMSConfig.model_validate(payload)

    validate_recurrence_carrier_compatibility(upstream, candidate)

    changed = candidate.model_dump()
    changed["consensus"]["height_factor"] = 0.7
    with pytest.raises(ValueError, match="may differ only"):
        validate_recurrence_carrier_compatibility(
            upstream,
            PAMSConfig.model_validate(changed),
        )


def test_local_recurrence_suppresses_static_context_and_modulates_carrier() -> None:
    embeddings = _localized_embeddings()
    result = _build(embeddings, period=16.0)
    active = result.active_masks[0]

    assert result.available.tolist() == [True]
    assert active[40:120].float().mean().item() > 0.8
    assert torch.cat((active[:20], active[140:])).float().mean().item() < 0.2
    assert torch.count_nonzero(result.curves[0, ~active]) == 0
    assert result.curve_standard_deviations.item() == pytest.approx(1.0)
    assert 0.0 < result.active_support_fractions.item() < 1.0
    assert result.recurrence_gate_energies.item() > 0.0

    shifted = _build(
        _localized_embeddings(active_start=16, active_stop=112),
        period=16.0,
    )
    assert not torch.equal(shifted.active_masks, result.active_masks)
    assert not torch.equal(shifted.curves, result.curves)


def test_quadrature_phase_shift_preserves_active_reference_and_expert_count() -> None:
    base = _build(_localized_embeddings(phase=0.0), period=16.0)
    shifted = _build(_localized_embeddings(phase=0.7), period=16.0)
    counter = MultiExpertCounter()

    outputs = []
    for readout in (base, shifted):
        output = counter.count(
            readout.curves[0],
            period_frames=16.0,
            valid_mask=readout.active_masks[0],
            period_confidence=0.8,
            reference_count_override=int(readout.active_reference_counts[0]),
        )
        outputs.append(output)
        assert abs(output.count - output.reference_count) <= 1
        assert len(set(output.expert_counts)) <= 2
    assert abs(outputs[0].count - outputs[1].count) <= 1


def test_off_frequency_amplitude_does_not_dominate_recurrence_carrier() -> None:
    embeddings = _localized_embeddings()
    time = torch.arange(embeddings.shape[0], dtype=torch.float32)
    distractor = torch.stack(
        (
            100.0 * torch.cos(2.0 * math.pi * time / 5.0),
            80.0 * torch.sin(2.0 * math.pi * time / 7.0),
        ),
        dim=1,
    )
    polluted = torch.cat((embeddings, distractor), dim=1)
    clean = _build(embeddings, period=16.0)
    noisy = _build(polluted, period=16.0)

    assert noisy.available.tolist() == [True]
    assert abs(
        int(noisy.active_reference_counts[0])
        - int(clean.active_reference_counts[0])
    ) <= 1


def test_shuffle_and_null_do_not_retain_an_unconditioned_carrier() -> None:
    embeddings = _localized_embeddings()
    baseline = _build(embeddings, period=16.0)
    generator = torch.Generator().manual_seed(2026)
    shuffled = _build(
        embeddings[torch.randperm(embeddings.shape[0], generator=generator)],
        period=16.0,
    )
    zero = _build(torch.zeros_like(embeddings), period=16.0)

    assert shuffled.active_support_fractions.item() < (
        0.75 * baseline.active_support_fractions.item()
    )
    assert shuffled.recurrence_gate_energies.item() < (
        0.75 * baseline.recurrence_gate_energies.item()
    )
    assert zero.available.tolist() == [False]
    assert zero.active_support_fractions.tolist() == [0.0]
    assert torch.count_nonzero(zero.curves) == 0


def test_cross_cycle_support_only_uplifts_phase_stable_features() -> None:
    frames = 192
    period = 12.0
    time = torch.arange(frames, dtype=torch.float32)
    angle = 2.0 * math.pi * time / period
    cycle = torch.floor(time / period).to(dtype=torch.long)
    alternating_sign = torch.where(
        (cycle % 2) == 0,
        torch.ones_like(time),
        -torch.ones_like(time),
    )
    centered = torch.stack(
        (
            torch.cos(angle),
            alternating_sign * torch.cos(angle),
            torch.sin(2.0 * math.pi * time / 7.0),
        ),
        dim=1,
    )
    weights = _cross_cycle_harmonic_feature_weights(
        centered,
        torch.ones(frames, dtype=torch.bool),
        period,
    )

    assert weights.shape == (3,)
    assert weights[0].item() == pytest.approx(math.sqrt(2.0), rel=1e-4)
    assert weights[1].item() == pytest.approx(1.0, abs=1e-4)
    assert 1.0 <= weights[2].item() < weights[0].item()


def test_unavailable_span_still_reports_observed_harmonic_energy() -> None:
    frames = 160
    period = 16.0
    embeddings = _localized_embeddings(
        frames=frames,
        period=period,
        active_start=64,
        active_stop=80,
    )
    result = _build(embeddings, period=period)

    assert result.available.tolist() == [False]
    assert result.active_support_fractions.tolist() == [0.0]
    assert result.harmonic_energy_fractions.item() > 0.0
    assert torch.count_nonzero(result.curves) == 0


def test_short_period_distributed_recurrence_is_scale_consistent() -> None:
    counter = MultiExpertCounter()
    counts: list[int] = []
    energies: list[float] = []
    for factor in (1.0, 0.75, 0.5):
        frames = int(round(192 * factor))
        period = 6.0 * factor
        time = torch.arange(frames, dtype=torch.float32)
        angle = 2.0 * math.pi * time / period
        phases = torch.linspace(0.0, math.pi, 24)
        periodic = torch.stack(
            [0.05 * torch.cos(angle + phase) for phase in phases],
            dim=1,
        )
        distractors = torch.stack(
            [
                torch.sin(2.0 * math.pi * time / (7.0 + index))
                for index in range(8)
            ],
            dim=1,
        )
        embeddings = torch.cat((periodic, distractors), dim=1)
        readout = _build(embeddings, period=period)
        output = counter.count(
            readout.curves[0],
            period_frames=period,
            valid_mask=readout.active_masks[0],
            period_confidence=0.8,
            reference_count_override=int(readout.active_reference_counts[0]),
        )
        assert readout.available.tolist() == [True]
        counts.append(output.count)
        energies.append(float(readout.recurrence_gate_energies[0]))

    assert max(counts) - min(counts) <= 1
    assert min(energies) > 0.0


def test_time_scaling_preserves_count_and_invalid_holes_are_not_bridged() -> None:
    counter = MultiExpertCounter()
    counts: list[int] = []
    for factor in (1.0, 0.75, 0.5):
        frames = int(round(160 * factor))
        period = 16.0 * factor
        start = int(round(32 * factor))
        stop = int(round(128 * factor))
        embeddings = _localized_embeddings(
            frames=frames,
            period=period,
            active_start=start,
            active_stop=stop,
        )
        mask = torch.ones(frames, dtype=torch.bool)
        readout = _build(embeddings, period=period, mask=mask)
        output = counter.count(
            readout.curves[0],
            period_frames=period,
            valid_mask=readout.active_masks[0],
            period_confidence=0.8,
            reference_count_override=int(readout.active_reference_counts[0]),
        )
        assert abs(output.count - output.reference_count) <= 1
        counts.append(output.count)
    assert max(counts) - min(counts) <= 1

    embeddings = _localized_embeddings()
    hole_mask = torch.ones(embeddings.shape[0], dtype=torch.bool)
    hole_mask[72:80] = False
    with_hole = _build(embeddings, period=16.0, mask=hole_mask)
    assert not bool(with_hole.active_masks[0, 72:80].any())
    assert torch.count_nonzero(with_hole.curves[0, 72:80]) == 0


def test_reference_override_is_explicit_and_does_not_change_experts() -> None:
    time = torch.arange(96, dtype=torch.float32)
    curve = torch.cos(2.0 * math.pi * time / 16.0)
    counter = MultiExpertCounter()
    ordinary = counter.count(curve, 16.0)
    overridden = counter.count(curve, 16.0, reference_count_override=2)

    assert overridden.reference_count == 2
    assert overridden.expert_counts == ordinary.expert_counts
    with pytest.raises(ValueError, match="mutually exclusive"):
        counter.count(
            curve,
            16.0,
            reference_frames=96,
            reference_count_override=2,
        )
