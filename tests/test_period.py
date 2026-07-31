import math

import pytest
import torch

import pams.period as period_module
from pams.model import SinusoidalPositionalEncoding
from pams.period import (
    autocorrelation_fft,
    embedding_energy,
    estimate_period,
    estimate_period_batch,
    estimate_period_from_detrended_projected_position,
    estimate_period_from_embedding_velocity_vectors,
    estimate_period_from_embeddings,
    estimate_period_from_pose,
    estimate_period_from_projected_pose,
    estimate_period_from_projected_position,
    estimate_period_from_projected_position_lag_velocity_fallback,
    estimate_period_from_vectors,
    linear_detrend_projected_position,
    projected_position_detrended_vector_acf_diagnostics,
    projected_position_lag_velocity_fallback_diagnostics,
    projected_position_vector_acf_diagnostics,
    vector_autocorrelation_fft,
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


@pytest.mark.parametrize("period", [4, 5, 8, 16, 32, 64, 128])
def test_vector_acf_recovers_signed_periods_across_frozen_bounds(period: int) -> None:
    time = torch.arange(512, dtype=torch.float64)
    angle = 2.0 * math.pi * time / period
    vectors = torch.stack(
        (
            torch.sin(angle),
            torch.cos(angle),
            0.4 * torch.sin(angle + 0.37),
        ),
        dim=-1,
    )

    autocorrelation = vector_autocorrelation_fft(vectors)
    periods, confidence = estimate_period_from_vectors(
        vectors,
        minimum=4,
        maximum=128,
    )

    assert periods.item() == pytest.approx(float(period), rel=0.011)
    assert confidence.item() > 0
    if period >= 8:
        assert autocorrelation[period // 2].item() < 0
        assert autocorrelation[period].item() > 0


def test_vector_acf_zero_and_constant_inputs_have_no_period_evidence() -> None:
    for sequence in (
        torch.zeros(256, 7),
        torch.full((256, 7), 3.5),
    ):
        autocorrelation = vector_autocorrelation_fft(sequence)
        _, confidence = estimate_period_from_vectors(
            sequence,
            minimum=4,
            maximum=128,
        )

        assert torch.count_nonzero(autocorrelation) == 0
        assert confidence.tolist() == [0.0]


def test_vector_acf_is_mask_aware_and_ignores_invalid_corruption() -> None:
    time = torch.arange(256, dtype=torch.float32)
    angle = 2.0 * math.pi * time / 20.0
    clean = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    corrupted = clean.clone()
    mask = torch.ones(256, dtype=torch.bool)
    mask[80:120] = False
    corrupted[~mask] = torch.randn_like(corrupted[~mask]) * 100_000.0

    clean_period, clean_confidence = estimate_period_from_vectors(
        clean,
        minimum=4,
        maximum=64,
        valid_mask=mask,
    )
    corrupted_period, corrupted_confidence = estimate_period_from_vectors(
        corrupted,
        minimum=4,
        maximum=64,
        valid_mask=mask,
    )

    assert torch.equal(clean_period, corrupted_period)
    assert torch.equal(clean_confidence, corrupted_confidence)
    assert corrupted_period.item() == pytest.approx(20.0, rel=0.03)


def test_vector_acf_period_and_confidence_are_orthogonal_basis_invariant() -> None:
    generator = torch.Generator().manual_seed(3407)
    time = torch.arange(256, dtype=torch.float64)
    angle = 2.0 * math.pi * time / 16.0
    sequence = torch.stack(
        (
            torch.sin(angle),
            torch.cos(angle),
            0.7 * torch.sin(angle + 0.2),
            0.3 * torch.cos(angle - 0.4),
        ),
        dim=-1,
    )
    orthogonal, _ = torch.linalg.qr(
        torch.randn(4, 4, dtype=torch.float64, generator=generator)
    )

    base_period, base_confidence = estimate_period_from_vectors(
        sequence,
        minimum=4,
        maximum=128,
    )
    rotated_period, rotated_confidence = estimate_period_from_vectors(
        sequence @ orthogonal,
        minimum=4,
        maximum=128,
    )

    assert torch.equal(base_period, rotated_period)
    assert torch.allclose(base_confidence, rotated_confidence, rtol=1e-12, atol=1e-12)


def test_embedding_velocity_vector_acf_is_orthogonal_basis_invariant() -> None:
    generator = torch.Generator().manual_seed(2026)
    time = torch.arange(256, dtype=torch.float32)
    angle = 2.0 * math.pi * time / 20.0
    embeddings = torch.stack(
        (
            torch.sin(angle),
            torch.cos(angle),
            0.7 * torch.sin(angle + 0.2),
            0.3 * torch.cos(angle - 0.4),
        ),
        dim=-1,
    )
    orthogonal, _ = torch.linalg.qr(
        torch.randn(4, 4, dtype=embeddings.dtype, generator=generator)
    )

    base_period, base_confidence = (
        estimate_period_from_embedding_velocity_vectors(
            embeddings,
            minimum=4,
            maximum=128,
        )
    )
    rotated_period, rotated_confidence = (
        estimate_period_from_embedding_velocity_vectors(
            embeddings @ orthogonal,
            minimum=4,
            maximum=128,
        )
    )

    assert torch.equal(base_period, rotated_period)
    assert torch.allclose(base_confidence, rotated_confidence, rtol=1e-5, atol=1e-6)


def test_embedding_velocity_vector_acf_recovers_synthetic_counts_two_to_forty() -> None:
    time = torch.arange(256, dtype=torch.float32)
    embeddings = []
    expected_periods = []
    for count in range(2, 41):
        angle = 2.0 * math.pi * count * time / 256.0
        embeddings.append(
            torch.stack(
                (
                    torch.sin(angle),
                    torch.cos(angle),
                    0.7 * torch.sin(angle + 0.3),
                ),
                dim=-1,
            )
        )
        expected_periods.append(256.0 / count)

    periods, confidence = estimate_period_from_embedding_velocity_vectors(
        torch.stack(embeddings),
        minimum=4,
        maximum=128,
    )
    expected = torch.tensor(expected_periods, dtype=periods.dtype)
    relative_error = (periods - expected).abs() / expected

    assert torch.all(relative_error <= 0.01)
    assert torch.all((periods >= 4.0) & (periods <= 128.0))
    assert torch.all(torch.isfinite(periods))
    assert torch.all(torch.isfinite(confidence))
    assert torch.all(confidence > 0.0)


def test_embedding_velocity_vector_acf_static_input_has_zero_confidence() -> None:
    periods, confidence = estimate_period_from_embedding_velocity_vectors(
        torch.full((2, 256, 8), 3.5),
        minimum=4,
        maximum=128,
    )

    assert periods.tolist() == [128.0, 128.0]
    assert confidence.tolist() == [0.0, 0.0]
    assert torch.all(torch.isfinite(periods))
    assert torch.all(torch.isfinite(confidence))


def test_embedding_velocity_vector_acf_is_mask_aware_and_stop_gradient() -> None:
    time = torch.arange(256, dtype=torch.float32)
    angle = 2.0 * math.pi * time / 32.0
    clean = torch.stack(
        (
            torch.sin(angle),
            torch.cos(angle),
            0.5 * torch.sin(angle + 0.7),
        ),
        dim=-1,
    )
    mask = torch.ones(256, dtype=torch.bool)
    mask[80:120] = False
    corrupted = clean.clone()
    corrupted[~mask] = float("nan")
    corrupted.requires_grad_(True)

    clean_period, clean_confidence = (
        estimate_period_from_embedding_velocity_vectors(
            clean,
            minimum=4,
            maximum=128,
            valid_mask=mask,
        )
    )
    corrupted_period, corrupted_confidence = (
        estimate_period_from_embedding_velocity_vectors(
            corrupted,
            minimum=4,
            maximum=128,
            valid_mask=mask,
        )
    )

    assert torch.equal(clean_period, corrupted_period)
    assert torch.equal(clean_confidence, corrupted_confidence)
    assert clean_period.tolist() == [32.0]
    assert torch.all(torch.isfinite(corrupted_period))
    assert torch.all(torch.isfinite(corrupted_confidence))
    assert not corrupted_period.requires_grad
    assert not corrupted_confidence.requires_grad
    assert corrupted.grad is None


def test_projected_pose_velocity_vector_acf_preserves_fundamental_period() -> None:
    time = torch.arange(256, dtype=torch.float32)
    angle = 2.0 * math.pi * time / 32.0
    projected_pose = torch.stack(
        (
            torch.sin(angle),
            torch.cos(angle),
            torch.sin(angle + 0.5),
        ),
        dim=-1,
    )

    periods, confidence = estimate_period_from_projected_pose(
        projected_pose,
        minimum=4,
        maximum=128,
    )

    assert periods.tolist() == [32]
    assert confidence.item() > 0


@pytest.mark.parametrize("harmonic", range(2, 8))
def test_projected_position_acf_rejects_velocity_amplified_harmonics(
    harmonic: int,
) -> None:
    time = torch.arange(256, dtype=torch.float64)
    fundamental_period = 64
    fundamental_angle = 2.0 * math.pi * time / fundamental_period
    harmonic_angle = harmonic * fundamental_angle + 0.37
    projected_pose = torch.stack(
        (
            torch.sin(fundamental_angle) + 0.75 * torch.sin(harmonic_angle),
            torch.cos(fundamental_angle) + 0.75 * torch.cos(harmonic_angle),
        ),
        dim=-1,
    )

    position_period, position_confidence = estimate_period_from_projected_position(
        projected_pose,
        minimum=4,
        maximum=128,
    )
    velocity_period, velocity_confidence = estimate_period_from_projected_pose(
        projected_pose,
        minimum=4,
        maximum=128,
    )

    assert position_period.item() == pytest.approx(float(fundamental_period))
    assert velocity_period.item() == pytest.approx(fundamental_period / harmonic)
    assert position_confidence.item() > 0
    assert velocity_confidence.item() > 0


def test_projected_position_acf_is_constant_bias_invariant() -> None:
    time = torch.arange(256, dtype=torch.float64)
    angle = 2.0 * math.pi * time / 32.0
    projected_pose = torch.stack(
        (
            torch.sin(angle),
            torch.cos(angle),
            0.4 * torch.sin(angle + 0.2),
        ),
        dim=-1,
    )
    bias = torch.tensor((1000.0, -700.0, 53.0), dtype=projected_pose.dtype)

    base_period, base_confidence = estimate_period_from_projected_position(projected_pose)
    biased_period, biased_confidence = estimate_period_from_projected_position(
        projected_pose + bias
    )

    assert torch.equal(base_period, biased_period)
    assert torch.allclose(base_confidence, biased_confidence, rtol=1e-10, atol=1e-12)


def test_projected_position_acf_is_orthogonal_basis_invariant() -> None:
    generator = torch.Generator().manual_seed(2026)
    time = torch.arange(256, dtype=torch.float64)
    angle = 2.0 * math.pi * time / 20.0
    projected_pose = torch.stack(
        (
            torch.sin(angle),
            torch.cos(angle),
            0.7 * torch.sin(angle + 0.3),
            0.2 * torch.cos(angle - 0.4),
        ),
        dim=-1,
    )
    orthogonal, _ = torch.linalg.qr(
        torch.randn(4, 4, dtype=projected_pose.dtype, generator=generator)
    )

    base_period, base_confidence = estimate_period_from_projected_position(projected_pose)
    rotated_period, rotated_confidence = estimate_period_from_projected_position(
        projected_pose @ orthogonal
    )

    assert torch.equal(base_period, rotated_period)
    assert torch.allclose(base_confidence, rotated_confidence, rtol=1e-12, atol=1e-12)


def test_projected_position_acf_mask_ignores_invalid_corruption() -> None:
    generator = torch.Generator().manual_seed(3407)
    time = torch.arange(256, dtype=torch.float32)
    angle = 2.0 * math.pi * time / 32.0
    clean = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    corrupted = clean.clone()
    mask = torch.ones(256, dtype=torch.bool)
    mask[192:] = False
    corrupted[~mask] = (
        torch.randn(
            corrupted[~mask].shape,
            dtype=corrupted.dtype,
            generator=generator,
        )
        * 100_000.0
    )

    clean_period, clean_confidence = estimate_period_from_projected_position(
        clean,
        minimum=4,
        maximum=128,
        valid_mask=mask,
    )
    corrupted_period, corrupted_confidence = estimate_period_from_projected_position(
        corrupted,
        minimum=4,
        maximum=128,
        valid_mask=mask,
    )

    assert torch.equal(clean_period, corrupted_period)
    assert torch.equal(clean_confidence, corrupted_confidence)
    assert clean_period.item() == 32.0


def test_projected_position_acf_zero_constant_and_all_invalid_have_no_evidence() -> None:
    all_valid = torch.ones(256, dtype=torch.bool)
    all_invalid = torch.zeros(256, dtype=torch.bool)
    generator = torch.Generator().manual_seed(42)
    inputs = (
        (torch.zeros(256, 3), all_valid),
        (torch.full((256, 3), 17.0), all_valid),
        (torch.randn(256, 3, generator=generator), all_invalid),
    )

    for projected_pose, mask in inputs:
        _, confidence = estimate_period_from_projected_position(
            projected_pose,
            valid_mask=mask,
        )
        diagnostic = projected_position_vector_acf_diagnostics(
            projected_pose,
            valid_mask=mask,
        )[0]

        assert confidence.tolist() == [0.0]
        assert diagnostic.confidence == 0.0
        assert diagnostic.selected_bin is None
        assert diagnostic.allowed_bins == tuple(range(2, 65))
        assert len(diagnostic.power_shares) == 63
        assert not any(diagnostic.power_shares)


def test_projected_position_spectrum_exposes_exact_allowed_band_and_argmax() -> None:
    time = torch.arange(256, dtype=torch.float64)
    angle = 2.0 * math.pi * time / 16.0
    projected_pose = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)

    period, confidence = estimate_period_from_projected_position(
        projected_pose,
        minimum=4,
        maximum=128,
    )
    diagnostic = projected_position_vector_acf_diagnostics(
        projected_pose,
        minimum=4,
        maximum=128,
    )[0]

    assert diagnostic.valid_length == 256
    assert diagnostic.allowed_bins == tuple(range(2, 65))
    assert len(diagnostic.allowed_bins) == 63
    assert len(diagnostic.frequencies) == 63
    assert len(diagnostic.periods) == 63
    assert len(diagnostic.power_shares) == 63
    assert sum(diagnostic.power_shares) == pytest.approx(1.0)
    selected_offset = max(
        range(len(diagnostic.power_shares)),
        key=diagnostic.power_shares.__getitem__,
    )
    assert diagnostic.selected_bin == diagnostic.allowed_bins[selected_offset] == 16
    assert diagnostic.selected_period == period.item() == 16.0
    assert diagnostic.confidence == pytest.approx(confidence.item())
    assert diagnostic.confidence == pytest.approx(diagnostic.power_shares[selected_offset])


def test_projected_position_acf_supports_batched_masks_and_diagnostics() -> None:
    time = torch.arange(256, dtype=torch.float32)
    sequences = torch.stack(
        tuple(
            torch.stack(
                (
                    torch.sin(2.0 * math.pi * time / period),
                    torch.cos(2.0 * math.pi * time / period),
                ),
                dim=-1,
            )
            for period in (16.0, 32.0)
        )
    )
    mask = torch.ones((2, 256), dtype=torch.bool)

    periods, confidence = estimate_period_from_projected_position(
        sequences,
        valid_mask=mask,
    )
    diagnostics = projected_position_vector_acf_diagnostics(
        sequences,
        valid_mask=mask,
    )

    assert periods.tolist() == [16.0, 32.0]
    assert torch.all(confidence > 0)
    assert len(diagnostics) == 2
    assert tuple(diagnostic.selected_bin for diagnostic in diagnostics) == (16, 8)
    assert all(diagnostic.allowed_bins == tuple(range(2, 65)) for diagnostic in diagnostics)


def test_detrended_projected_position_is_invariant_to_affine_drift() -> None:
    time = torch.arange(256, dtype=torch.float64)
    angle = 2.0 * math.pi * time / 32.0
    clean = torch.stack(
        (
            torch.sin(angle),
            torch.cos(angle),
            0.4 * torch.sin(angle + 0.31),
        ),
        dim=-1,
    )
    offset = torch.tensor((12.0, -7.0, 3.5), dtype=clean.dtype)
    slope = torch.tensor((0.03, -0.05, 0.08), dtype=clean.dtype)
    drifted = clean + offset + time.unsqueeze(-1) * slope

    clean_residual = linear_detrend_projected_position(clean)
    drifted_residual = linear_detrend_projected_position(drifted)
    clean_period, clean_confidence = (
        estimate_period_from_detrended_projected_position(clean)
    )
    drifted_period, drifted_confidence = (
        estimate_period_from_detrended_projected_position(drifted)
    )

    assert torch.allclose(clean_residual, drifted_residual, rtol=1e-12, atol=1e-12)
    assert clean_period.tolist() == drifted_period.tolist() == [32.0]
    assert torch.allclose(
        clean_confidence,
        drifted_confidence,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("harmonic", range(2, 8))
def test_detrended_projected_position_preserves_k2_to_k7_fundamental(
    harmonic: int,
) -> None:
    time = torch.arange(256, dtype=torch.float64)
    fundamental_angle = 2.0 * math.pi * time / 64.0
    harmonic_angle = harmonic * fundamental_angle + 0.37
    periodic = torch.stack(
        (
            torch.sin(fundamental_angle) + 0.75 * torch.sin(harmonic_angle),
            torch.cos(fundamental_angle) + 0.75 * torch.cos(harmonic_angle),
        ),
        dim=-1,
    )
    offset = torch.tensor((20.0, -11.0), dtype=periodic.dtype)
    slope = torch.tensor((0.04, -0.07), dtype=periodic.dtype)
    projected_pose = periodic + offset + time.unsqueeze(-1) * slope

    period, confidence = estimate_period_from_detrended_projected_position(
        projected_pose,
        minimum=4,
        maximum=128,
    )
    diagnostic = projected_position_detrended_vector_acf_diagnostics(
        projected_pose,
        minimum=4,
        maximum=128,
    )[0]

    assert period.tolist() == [64.0]
    assert confidence.item() > 0.0
    assert diagnostic.selected_bin == 4
    assert diagnostic.allowed_bins == tuple(range(2, 65))


def test_detrended_projected_position_mask_ignores_invalid_pollution() -> None:
    generator = torch.Generator().manual_seed(3407)
    time = torch.arange(256, dtype=torch.float64)
    angle = 2.0 * math.pi * time / 32.0
    clean = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    clean = clean + torch.tensor((5.0, -8.0)) + time.unsqueeze(-1) * torch.tensor(
        (0.02, -0.03)
    )
    mask = torch.ones(256, dtype=torch.bool)
    mask[48:77] = False
    mask[190:213] = False
    corrupted = clean.clone()
    corrupted[~mask] = (
        torch.randn(
            corrupted[~mask].shape,
            dtype=corrupted.dtype,
            generator=generator,
        )
        * 1e12
    )

    clean_residual = linear_detrend_projected_position(clean, mask)
    corrupted_residual = linear_detrend_projected_position(corrupted, mask)
    clean_period, clean_confidence = (
        estimate_period_from_detrended_projected_position(
            clean,
            valid_mask=mask,
        )
    )
    corrupted_period, corrupted_confidence = (
        estimate_period_from_detrended_projected_position(
            corrupted,
            valid_mask=mask,
        )
    )

    assert torch.equal(clean_residual, corrupted_residual)
    assert torch.count_nonzero(corrupted_residual[~mask]) == 0
    assert torch.equal(clean_period, corrupted_period)
    assert torch.equal(clean_confidence, corrupted_confidence)
    assert clean_period.tolist() == [32.0]


def test_detrended_projected_position_no_evidence_contract() -> None:
    time = torch.arange(256, dtype=torch.float64)
    all_valid = torch.ones(256, dtype=torch.bool)
    all_invalid = torch.zeros(256, dtype=torch.bool)
    affine = torch.stack((3.0 + 0.2 * time, -7.0 - 0.05 * time), dim=-1)
    float32_time = time.float()
    float32_affine = torch.stack(
        (3.0 + 0.2 * float32_time, -7.0 - 0.05 * float32_time),
        dim=-1,
    )
    inputs = (
        (torch.zeros(256, 2, dtype=torch.float64), all_valid, 128.0),
        (torch.full((256, 2), 11.0, dtype=torch.float64), all_valid, 128.0),
        (affine, all_valid, 128.0),
        (float32_affine, all_valid, 128.0),
        (torch.randn(256, 2, dtype=torch.float64), all_invalid, 4.0),
    )

    for projected_pose, mask, fallback_period in inputs:
        period, confidence = estimate_period_from_detrended_projected_position(
            projected_pose,
            valid_mask=mask,
        )
        diagnostic = projected_position_detrended_vector_acf_diagnostics(
            projected_pose,
            valid_mask=mask,
        )[0]

        assert period.tolist() == [fallback_period]
        assert confidence.tolist() == [0.0]
        assert diagnostic.selected_bin is None
        assert diagnostic.confidence == 0.0
        assert diagnostic.allowed_bins == tuple(range(2, 65))
        assert not any(diagnostic.power_shares)


def test_detrended_projected_position_supports_batches_and_masks() -> None:
    time = torch.arange(256, dtype=torch.float32)
    sequences = torch.stack(
        tuple(
            torch.stack(
                (
                    torch.sin(2.0 * math.pi * time / period),
                    torch.cos(2.0 * math.pi * time / period),
                ),
                dim=-1,
            )
            + time.unsqueeze(-1) * torch.tensor((0.01, -0.02))
            for period in (16.0, 32.0)
        )
    )
    mask = torch.ones((2, 256), dtype=torch.bool)
    mask[0, 220:] = False
    mask[1, 96:112] = False

    periods, confidence = estimate_period_from_detrended_projected_position(
        sequences,
        valid_mask=mask,
    )
    diagnostics = projected_position_detrended_vector_acf_diagnostics(
        sequences,
        valid_mask=mask,
    )
    residual = linear_detrend_projected_position(sequences, mask)

    assert periods.tolist() == [16.0, 32.0]
    assert torch.all(confidence > 0.0)
    assert len(diagnostics) == 2
    assert tuple(item.valid_length for item in diagnostics) == (220, 240)
    assert tuple(item.selected_bin for item in diagnostics) == (16, 8)
    assert torch.count_nonzero(residual[~mask]) == 0


@pytest.mark.parametrize("drift", [0.02, 1.0])
@pytest.mark.parametrize("harmonic", range(2, 8))
def test_lag_velocity_fallback_preserves_drifted_fundamental(
    harmonic: int,
    drift: float,
) -> None:
    time = torch.arange(256, dtype=torch.float64)
    fundamental_angle = 2.0 * math.pi * time / 64.0
    harmonic_angle = harmonic * fundamental_angle + 0.37
    projected_pose = torch.stack(
        (
            torch.sin(fundamental_angle) + 0.75 * torch.sin(harmonic_angle),
            torch.cos(fundamental_angle) + 0.75 * torch.cos(harmonic_angle),
        ),
        dim=-1,
    )
    projected_pose = projected_pose + drift * time.unsqueeze(-1)

    period, confidence = (
        estimate_period_from_projected_position_lag_velocity_fallback(
            projected_pose
        )
    )
    diagnostic = projected_position_lag_velocity_fallback_diagnostics(
        projected_pose
    )[0]

    assert period.tolist() == [64.0]
    assert confidence.item() > 0.0
    assert diagnostic.selected_lag == 64
    assert diagnostic.selected_period == 64.0
    assert diagnostic.selection_source == "detrended-position-lag-acf"
    assert diagnostic.searched_lags == tuple(range(4, 129))
    assert diagnostic.eligible_peak_lags[0] == 64
    assert diagnostic.near_best_height_threshold == pytest.approx(
        0.90 * max(diagnostic.positive_peak_heights)
    )
    assert diagnostic.positive_peak_heights[
        diagnostic.positive_peak_lags.index(64)
    ] == pytest.approx(diagnostic.confidence)


def test_lag_velocity_near_best_rule_avoids_count_two_to_forty_multiples() -> None:
    time = torch.arange(256, dtype=torch.float64)
    counts = torch.arange(2, 41, dtype=torch.float64)
    angles = 2.0 * math.pi * counts[:, None] * time[None, :] / 256.0
    sequences = torch.stack((torch.sin(angles), torch.cos(angles)), dim=-1)
    sequences = sequences + time[None, :, None]

    periods, confidence = (
        estimate_period_from_projected_position_lag_velocity_fallback(sequences)
    )
    expected_periods = 256.0 / counts
    relative_error = (periods - expected_periods).abs() / expected_periods
    diagnostics = projected_position_lag_velocity_fallback_diagnostics(sequences)

    assert torch.all(confidence > 0.0)
    assert torch.all(relative_error <= 0.10)
    assert relative_error.max().item() == pytest.approx(0.06640625)
    assert all(item.eligible_peak_lags for item in diagnostics)
    assert tuple(item.selected_lag for item in diagnostics) == tuple(
        item.eligible_peak_lags[0] for item in diagnostics
    )


@pytest.mark.parametrize(
    ("early_height", "expected_lag"),
    [(0.90, 12), (0.899, 24)],
)
def test_lag_velocity_near_best_threshold_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
    early_height: float,
    expected_lag: int,
) -> None:
    def controlled_autocorrelation(
        sequence: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del valid_mask
        batch = 1 if sequence.ndim == 2 else sequence.shape[0]
        result = sequence.new_zeros((batch, sequence.shape[-2]))
        result[:, 0] = 1.0
        result[:, 12] = early_height
        result[:, 24] = 1.0
        return result[0] if sequence.ndim == 2 else result

    monkeypatch.setattr(
        period_module,
        "vector_autocorrelation_fft",
        controlled_autocorrelation,
    )
    sequence = torch.stack(
        (
            torch.arange(256, dtype=torch.float64),
            -torch.arange(256, dtype=torch.float64),
        ),
        dim=-1,
    )

    diagnostic = projected_position_lag_velocity_fallback_diagnostics(sequence)[0]

    assert diagnostic.near_best_height_threshold == pytest.approx(0.90)
    assert diagnostic.selected_lag == expected_lag
    assert diagnostic.eligible_peak_lags == (
        (12, 24) if early_height == 0.90 else (24,)
    )
    assert diagnostic.confidence == pytest.approx(
        early_height if expected_lag == 12 else 1.0
    )


def test_lag_velocity_fallback_uses_raw_velocity_only_without_position_peak() -> None:
    time = torch.arange(256, dtype=torch.float32)
    angle = 2.0 * math.pi * time / 16.0
    projected_pose = torch.stack(
        (
            100.0 * time + 0.01 * torch.sin(angle),
            -70.0 * time + 0.01 * torch.cos(angle),
        ),
        dim=-1,
    )

    period, confidence = (
        estimate_period_from_projected_position_lag_velocity_fallback(
            projected_pose
        )
    )
    diagnostic = projected_position_lag_velocity_fallback_diagnostics(
        projected_pose
    )[0]

    assert period.tolist() == [16.0]
    assert confidence.item() > 0.0
    assert diagnostic.positive_peak_lags == ()
    assert diagnostic.selected_lag is None
    assert diagnostic.selection_source == "projected-velocity-spectrum-fallback"
    assert diagnostic.selected_period == diagnostic.fallback_period == 16.0
    assert diagnostic.confidence == pytest.approx(diagnostic.fallback_confidence)


def test_lag_velocity_fallback_mask_ignores_invalid_pollution() -> None:
    generator = torch.Generator().manual_seed(3407)
    time = torch.arange(256, dtype=torch.float64)
    angle = 2.0 * math.pi * time / 32.0
    clean = torch.stack((torch.sin(angle), torch.cos(angle)), dim=-1)
    clean = clean + 0.02 * time.unsqueeze(-1)
    mask = torch.ones(256, dtype=torch.bool)
    mask[96:112] = False
    corrupted = clean.clone()
    corrupted[~mask] = (
        torch.randn(
            corrupted[~mask].shape,
            dtype=corrupted.dtype,
            generator=generator,
        )
        * 1e12
    )

    clean_period, clean_confidence = (
        estimate_period_from_projected_position_lag_velocity_fallback(
            clean,
            valid_mask=mask,
        )
    )
    corrupted_period, corrupted_confidence = (
        estimate_period_from_projected_position_lag_velocity_fallback(
            corrupted,
            valid_mask=mask,
        )
    )
    clean_diagnostic = projected_position_lag_velocity_fallback_diagnostics(
        clean,
        valid_mask=mask,
    )[0]
    corrupted_diagnostic = projected_position_lag_velocity_fallback_diagnostics(
        corrupted,
        valid_mask=mask,
    )[0]

    assert clean_period.tolist() == corrupted_period.tolist() == [32.0]
    assert torch.equal(clean_confidence, corrupted_confidence)
    assert clean_diagnostic == corrupted_diagnostic
    assert clean_diagnostic.valid_length == int(mask.sum())
    assert clean_diagnostic.searched_lags[-1] == int(mask.sum()) // 2


def test_lag_velocity_fallback_supports_batches_and_no_evidence() -> None:
    time = torch.arange(256, dtype=torch.float32)
    sequences = torch.stack(
        tuple(
            torch.stack(
                (
                    torch.sin(2.0 * math.pi * time / period),
                    torch.cos(2.0 * math.pi * time / period),
                ),
                dim=-1,
            )
            + 0.02 * time.unsqueeze(-1)
            for period in (16.0, 32.0)
        )
    )
    mask = torch.ones((2, 256), dtype=torch.bool)
    mask[0, 220:] = False
    mask[1, 96:112] = False

    periods, confidence = (
        estimate_period_from_projected_position_lag_velocity_fallback(
            sequences,
            valid_mask=mask,
        )
    )
    diagnostics = projected_position_lag_velocity_fallback_diagnostics(
        sequences,
        valid_mask=mask,
    )

    assert periods.tolist() == [16.0, 32.0]
    assert torch.all(confidence > 0.0)
    assert len(diagnostics) == 2
    assert tuple(item.selected_lag for item in diagnostics) == (16, 32)
    assert tuple(item.valid_length for item in diagnostics) == (220, 240)

    all_valid = torch.ones(256, dtype=torch.bool)
    all_invalid = torch.zeros(256, dtype=torch.bool)
    no_evidence_inputs = (
        (torch.zeros(256, 2), all_valid),
        (torch.full((256, 2), 17.0), all_valid),
        (torch.randn(256, 2, generator=torch.Generator().manual_seed(42)), all_invalid),
    )
    for sequence, sequence_mask in no_evidence_inputs:
        period, no_evidence_confidence = (
            estimate_period_from_projected_position_lag_velocity_fallback(
                sequence,
                valid_mask=sequence_mask,
            )
        )
        diagnostic = projected_position_lag_velocity_fallback_diagnostics(
            sequence,
            valid_mask=sequence_mask,
        )[0]
        assert period.tolist() == [4.0]
        assert no_evidence_confidence.tolist() == [0.0]
        assert diagnostic.selected_lag is None
        assert diagnostic.selection_source == "no-evidence"
        assert diagnostic.positive_peak_lags == ()
