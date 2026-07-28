import math

import pytest
import torch

from pams.model import SinusoidalPositionalEncoding
from pams.period import (
    autocorrelation_fft,
    embedding_energy,
    estimate_period,
    estimate_period_batch,
    estimate_period_from_embeddings,
    estimate_period_from_pose,
)


def _sine(period: int, length: int) -> torch.Tensor:
    time = torch.arange(length, dtype=torch.float32)
    return torch.sin(2.0 * math.pi * time / period)


def test_fft_autocorrelation_recovers_known_period() -> None:
    signal = _sine(period=16, length=256)
    autocorrelation = autocorrelation_fft(signal)
    assert autocorrelation.shape == (256,)
    assert torch.isclose(autocorrelation[0], torch.tensor(1.0), atol=1e-6)
    estimate = estimate_period(signal, minimum=4, maximum=64)
    assert estimate.period == 16
    assert 0.0 < estimate.confidence <= 1.0
    assert estimate_period(signal.half(), minimum=4, maximum=64).period == 16


def test_fft_period_preserves_fractional_dominant_bin() -> None:
    time = torch.arange(256, dtype=torch.float32)
    forty_cycles = torch.sin(2.0 * math.pi * 40.0 * time / 256.0)

    estimate = estimate_period(forty_cycles, minimum=4, maximum=128)

    assert estimate.period == pytest.approx(6.4)
    assert estimate.frequency == pytest.approx(40.0 / 256.0)


def test_period_estimator_respects_mask_and_bounds() -> None:
    signal = torch.cat((_sine(20, 200), torch.randn(56) * 20.0))
    mask = torch.zeros(256, dtype=torch.bool)
    mask[:200] = True
    periods, confidence = estimate_period_batch(
        signal,
        minimum=8,
        maximum=40,
        valid_mask=mask,
    )
    assert periods.item() == pytest.approx(20.0, rel=0.02)
    assert confidence.shape == (1,)

    constant = estimate_period(torch.ones(24), minimum=4, maximum=12)
    assert 4 <= constant.period <= 12
    assert constant.confidence == 0.0


def test_pose_and_embedding_proxies_preserve_cycle_phase() -> None:
    base = _sine(12, 120)
    pose = torch.stack(
        (
            base,
            torch.cos(torch.arange(120) * (2.0 * math.pi / 12)),
            0.5 * base,
        ),
        dim=-1,
    ).reshape(120, 1, 3)
    pose_periods, _ = estimate_period_from_pose(pose, minimum=4, maximum=30)
    assert pose_periods.tolist() == [12]

    embeddings = torch.stack((base, torch.roll(base, 3), -base), dim=-1)
    proxy = embedding_energy(embeddings)
    assert proxy.shape == (120,)
    embedding_periods, _ = estimate_period_from_embeddings(embeddings, minimum=4, maximum=30)
    assert embedding_periods.tolist() == [12]


def test_batched_embedding_estimation() -> None:
    first = torch.stack((_sine(10, 100), torch.roll(_sine(10, 100), 2)), dim=-1)
    second = torch.stack((_sine(25, 100), torch.roll(_sine(25, 100), 5)), dim=-1)
    periods, confidence = estimate_period_from_embeddings(
        torch.stack((first, second)),
        minimum=5,
        maximum=40,
    )
    assert periods.tolist() == [10, 25]
    assert torch.all(confidence > 0)


@pytest.mark.parametrize("period", [5, 8, 12, 20, 32, 64])
def test_embedding_velocity_recovers_multiple_sine_periods(period: int) -> None:
    base = _sine(period, 256)
    embeddings = torch.stack((base, torch.roll(base, 2), -0.5 * base), dim=-1)
    periods, confidence = estimate_period_from_embeddings(
        embeddings,
        minimum=4,
        maximum=128,
    )
    assert periods.item() == pytest.approx(float(period), rel=0.04)
    assert confidence.item() > 0


def test_embedding_velocity_zero_and_noise_do_not_mimic_clean_periodicity() -> None:
    zero_period, zero_confidence = estimate_period_from_embeddings(
        torch.zeros(256, 16),
        minimum=4,
        maximum=128,
    )
    assert zero_period.tolist() == [128]
    assert zero_confidence.tolist() == [0.0]

    generator = torch.Generator().manual_seed(2026)
    noise = torch.randn(256, 16, generator=generator)
    _, noise_confidence = estimate_period_from_embeddings(
        noise,
        minimum=4,
        maximum=128,
    )
    clean = torch.stack((_sine(16, 256), torch.roll(_sine(16, 256), 4)), dim=-1)
    clean_period, clean_confidence = estimate_period_from_embeddings(
        clean,
        minimum=4,
        maximum=128,
    )
    assert clean_period.tolist() == [16]
    assert clean_confidence.item() > noise_confidence.item()


def test_embedding_velocity_is_mask_aware_and_does_not_cross_gaps() -> None:
    embeddings = torch.zeros(32, 2)
    embeddings[:, 0] = torch.arange(32)
    embeddings[17:, 0] += 10_000
    mask = torch.ones(32, dtype=torch.bool)
    mask[16] = False

    proxy = embedding_energy(embeddings, mask)

    assert proxy[0] == 0
    assert proxy[16] == 0
    assert proxy[17] == 0
    assert proxy[15] == pytest.approx(proxy[18])
    assert proxy.abs().max() < 1e-6


@pytest.mark.parametrize("period", [8, 16, 32, 64])
def test_embedding_velocity_resists_absolute_position_encoding(period: int) -> None:
    time = torch.arange(256, dtype=torch.float32)
    action = 2.0 * torch.sin(2.0 * math.pi * time / period)
    sinusoidal_position = SinusoidalPositionalEncoding(32).encoding[:256] * 0.05
    embeddings = torch.cat(
        (
            action[:, None],
            sinusoidal_position,
            (100.0 * time)[:, None],
        ),
        dim=-1,
    )

    periods, confidence = estimate_period_from_embeddings(
        embeddings,
        minimum=4,
        maximum=128,
    )

    assert periods.tolist() == [period]
    assert confidence.item() > 0
