import math

import torch

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
    assert periods.tolist() == [20]
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
