from __future__ import annotations

import math

import pytest
import torch

from pams.config import PAMSConfig
from pams.embedding_curve import (
    _dominant_harmonic_direction,
    build_frequency_matched_action_curves,
    estimate_embedding_action_curves,
    validate_embedding_action_curve_compatibility,
)
from pams.period import estimate_period_from_embedding_velocity_vectors


def _periodic_embeddings(*, frames: int = 96, period: float = 16.0) -> torch.Tensor:
    time = torch.arange(frames, dtype=torch.float32)
    return torch.stack(
        (
            2.0 * torch.cos(2.0 * math.pi * time / period),
            torch.sin(2.0 * math.pi * time / period),
            0.2 * torch.cos(4.0 * math.pi * time / period + 0.3),
            0.01 * time / frames,
        ),
        dim=1,
    )


def _candidate_config(upstream: PAMSConfig) -> PAMSConfig:
    payload = upstream.model_dump()
    payload["readout"]["action_curve_source"] = (
        "embedding_frequency_projection"
    )
    return PAMSConfig.model_validate(payload)


def test_embedding_curve_compatibility_allows_only_readout_marker() -> None:
    upstream = PAMSConfig()
    candidate = _candidate_config(upstream)

    validate_embedding_action_curve_compatibility(upstream, candidate)

    changed = candidate.model_dump()
    changed["consensus"]["height_factor"] = 0.7
    with pytest.raises(ValueError, match="may differ only"):
        validate_embedding_action_curve_compatibility(
            upstream,
            PAMSConfig.model_validate(changed),
        )


def test_frequency_projection_is_deterministic_and_retains_dense_hole_indices() -> None:
    embeddings = _periodic_embeddings()
    mask = torch.ones(embeddings.shape[0], dtype=torch.bool)
    mask[19:31] = False
    mask[55] = False
    corrupted = embeddings.clone()
    corrupted[~mask] = 1_000_000.0

    clean = build_frequency_matched_action_curves(
        embeddings,
        torch.tensor([16.0]),
        mask,
        timeline_lengths=torch.tensor([embeddings.shape[0]]),
    )
    repeated = build_frequency_matched_action_curves(
        embeddings,
        torch.tensor([16.0]),
        mask,
        timeline_lengths=torch.tensor([embeddings.shape[0]]),
    )
    polluted = build_frequency_matched_action_curves(
        corrupted,
        torch.tensor([16.0]),
        mask,
        timeline_lengths=torch.tensor([embeddings.shape[0]]),
    )

    assert clean.available.tolist() == [True]
    assert torch.equal(repeated.curves, clean.curves)
    assert torch.equal(polluted.curves, clean.curves)
    assert torch.count_nonzero(clean.curves[0, ~mask]) == 0
    assert clean.curve_standard_deviations.item() == pytest.approx(1.0)

    dense_wave = embeddings[:, 0]
    dense_wave = dense_wave - dense_wave[mask].mean()
    dense_wave = dense_wave / dense_wave[mask].square().mean().sqrt()
    compact_time = torch.arange(int(mask.sum()), dtype=torch.float32)
    compact_wave = torch.cos(2.0 * math.pi * compact_time / 16.0)
    dense_correlation = torch.abs(
        torch.corrcoef(torch.stack((clean.curves[0, mask], dense_wave[mask])))[0, 1]
    )
    compact_correlation = torch.abs(
        torch.corrcoef(torch.stack((clean.curves[0, mask], compact_wave)))[0, 1]
    )
    assert dense_correlation > 0.99
    assert dense_correlation > compact_correlation + 0.1


def test_frequency_projection_has_explicit_sign_and_tie_breaks() -> None:
    tied = torch.eye(2, dtype=torch.float32)
    direction, available = _dominant_harmonic_direction(tied)
    negated_direction, negated_available = _dominant_harmonic_direction(-tied)

    assert available is True
    assert negated_available is True
    assert torch.equal(direction, torch.tensor([1.0, 0.0]))
    assert torch.equal(negated_direction, torch.tensor([1.0, 0.0]))
    assert torch.equal(
        _dominant_harmonic_direction(tied)[0],
        direction,
    )


def test_estimated_curve_keeps_acf_confidence_and_zero_input_degrades() -> None:
    embeddings = _periodic_embeddings(frames=128, period=16.0)
    mask = torch.ones(embeddings.shape[0], dtype=torch.bool)
    periodic = estimate_embedding_action_curves(
        embeddings,
        minimum_period=4,
        maximum_period=64,
        valid_mask=mask,
        timeline_lengths=torch.tensor([embeddings.shape[0]]),
    )
    expected_period, expected_confidence = (
        estimate_period_from_embedding_velocity_vectors(
            embeddings,
            minimum=4,
            maximum=64,
            valid_mask=mask,
        )
    )
    zero = estimate_embedding_action_curves(
        torch.zeros_like(embeddings),
        minimum_period=4,
        maximum_period=64,
        valid_mask=mask,
        timeline_lengths=torch.tensor([embeddings.shape[0]]),
    )

    assert torch.equal(periodic.periods, expected_period)
    assert torch.equal(periodic.period_confidences, expected_confidence)
    assert periodic.periods.item() == pytest.approx(16.0)
    assert 0.0 < periodic.period_confidences.item() <= 1.0
    assert periodic.available.tolist() == [True]
    assert zero.period_confidences.tolist() == [0.0]
    assert zero.available.tolist() == [False]
    assert torch.count_nonzero(zero.curves) == 0


def test_frequency_projection_rejects_valid_padding_and_ignores_internal_nan() -> None:
    embeddings = _periodic_embeddings(frames=64)
    padded = torch.cat((embeddings, torch.zeros(8, embeddings.shape[1])), dim=0)
    mask = torch.zeros(72, dtype=torch.bool)
    mask[:64] = True
    mask[17] = False
    padded[17] = torch.nan

    result = build_frequency_matched_action_curves(
        padded,
        torch.tensor([16.0]),
        mask,
        timeline_lengths=torch.tensor([64]),
    )
    assert result.available.tolist() == [True]
    assert torch.count_nonzero(result.curves[0, ~mask]) == 0

    invalid_mask = mask.clone()
    invalid_mask[70] = True
    with pytest.raises(ValueError, match="padded frames"):
        build_frequency_matched_action_curves(
            padded,
            torch.tensor([16.0]),
            invalid_mask,
            timeline_lengths=torch.tensor([64]),
        )
