"""Period estimation from pose or embedding sequences.

PAMS states that the period is obtained through an FFT over an autocorrelation
signal, but does not publish implementation details.  The routines here are a
deterministic, mask-aware implementation of that description.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from scipy.signal import find_peaks
from torch import Tensor


@dataclass(frozen=True)
class PeriodEstimate:
    """One sequence's bounded period estimate and diagnostic information."""

    period: float
    confidence: float
    frequency: float
    valid_length: int


@dataclass(frozen=True)
class ProjectedPositionSpectrumDiagnostic:
    """Auditable allowed-band evidence for one projected-position sequence."""

    valid_length: int
    allowed_bins: tuple[int, ...]
    frequencies: tuple[float, ...]
    periods: tuple[float, ...]
    power_shares: tuple[float, ...]
    selected_bin: int | None
    selected_period: float
    confidence: float


@dataclass(frozen=True)
class ProjectedPositionLagVelocityFallbackDiagnostic:
    """Auditable lag-ACF peak selection with velocity-spectrum fallback."""

    valid_length: int
    searched_lags: tuple[int, ...]
    autocorrelation_values: tuple[float, ...]
    positive_peak_lags: tuple[int, ...]
    positive_peak_heights: tuple[float, ...]
    positive_peak_prominences: tuple[float, ...]
    near_best_height_threshold: float
    eligible_peak_lags: tuple[int, ...]
    eligible_peak_heights: tuple[float, ...]
    selected_lag: int | None
    selected_period: float
    confidence: float
    selection_source: str
    fallback_period: float
    fallback_confidence: float


@dataclass(frozen=True)
class HarmonicFundamentalDiagnostic:
    """Auditable evidence for an inferred embedding-velocity fundamental."""

    valid_length: int
    fft_length: int
    candidate_bins: tuple[int, ...]
    candidate_periods: tuple[float, ...]
    candidate_scores: tuple[float, ...]
    selected_bin: int | None
    selected_period: float
    prewhitened_bin: int | None
    prewhitened_residual_power_fraction: float
    selected_harmonic_bins: tuple[int, ...]
    selected_harmonic_power_shares: tuple[float, ...]
    candidate_distribution_concentration: float
    family_spectral_concentration: float
    uniform_family_expectation: float
    overtone_noise_floor: float
    confidence: float


def _as_batch_signal(signal: Tensor) -> tuple[Tensor, bool]:
    if signal.ndim == 1:
        return signal.unsqueeze(0), True
    if signal.ndim != 2:
        raise ValueError("signal must have shape [time] or [batch, time]")
    return signal, False


def _as_batch_vectors(sequence: Tensor) -> tuple[Tensor, bool]:
    if sequence.ndim == 2:
        if sequence.shape[-1] < 1:
            raise ValueError("vector sequence feature dimension must be positive")
        return sequence.unsqueeze(0), True
    if sequence.ndim != 3:
        raise ValueError(
            "vector sequence must have shape [time, features] or "
            "[batch, time, features]"
        )
    if sequence.shape[-1] < 1:
        raise ValueError("vector sequence feature dimension must be positive")
    return sequence, False


def _validated_mask(signal: Tensor, valid_mask: Tensor | None) -> Tensor:
    if valid_mask is None:
        return torch.ones_like(signal, dtype=torch.bool)
    if valid_mask.shape != signal.shape:
        raise ValueError(
            f"valid_mask must have shape {tuple(signal.shape)}, got {tuple(valid_mask.shape)}"
        )
    return valid_mask.to(device=signal.device, dtype=torch.bool)


def temporal_component(
    sequence: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Extract a signed one-dimensional activity proxy.

    The highest-variance signed coordinate preserves cycle phase, unlike a
    squared velocity magnitude which can halve the apparent period.  Selection
    is performed on detached values because period selection is intentionally
    non-gradient.  Inputs may be ``[time, features]`` or
    ``[batch, time, ...]``.
    """

    if sequence.ndim < 2:
        raise ValueError("sequence must have a time and feature dimension")
    unbatched = sequence.ndim == 2
    values = sequence.unsqueeze(0) if unbatched else sequence
    if values.ndim < 3:
        raise ValueError("sequence must have shape [time, features] or [batch, time, ...]")
    batch, time = values.shape[:2]
    flat = values.detach().to(dtype=torch.float32).reshape(batch, time, -1)

    if valid_mask is None:
        mask = torch.ones((batch, time), dtype=torch.bool, device=flat.device)
    else:
        expected = (time,) if unbatched else (batch, time)
        if valid_mask.shape != expected:
            raise ValueError(
                f"valid_mask must have shape {expected}, got {tuple(valid_mask.shape)}"
            )
        mask = valid_mask.unsqueeze(0) if unbatched else valid_mask
        mask = mask.to(device=flat.device, dtype=torch.bool)

    weights = mask.to(flat.dtype).unsqueeze(-1)
    counts = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    means = (flat * weights).sum(dim=1, keepdim=True) / counts
    centered = torch.nan_to_num((flat - means) * weights)
    coordinate_energy = centered.square().sum(dim=1)
    coordinates = coordinate_energy.argmax(dim=1)
    gather_indices = coordinates.view(batch, 1, 1).expand(batch, time, 1)
    result = centered.gather(dim=2, index=gather_indices).squeeze(-1)
    usable = (mask.sum(dim=1) >= 2) & (coordinate_energy.max(dim=1).values > 1e-12)
    result = result.masked_fill(~mask, 0.0)
    result = result * usable.to(result.dtype).unsqueeze(-1)
    return result[0] if unbatched else result


def pose_energy(poses: Tensor, valid_mask: Tensor | None = None) -> Tensor:
    """Return the deterministic pose-activity proxy used during warm-up."""

    if poses.ndim not in (3, 4):
        raise ValueError("poses must have shape [time, K, C] or [batch, time, K, C]")
    if poses.ndim == 3:
        return temporal_component(poses.flatten(start_dim=1), valid_mask)
    return temporal_component(poses.flatten(start_dim=2), valid_mask)


def embedding_energy(
    embeddings: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Return a stop-gradient signed embedding-velocity proxy.

    PAMS' frozen protocol uses temporal embedding velocity after the pose
    warm-up.  A signed coordinate is retained rather than an L2 speed: speed
    folds a one-dimensional sinusoid onto its absolute derivative and can
    therefore halve the recovered period.  Velocity samples are valid only
    when both adjacent embedding frames are valid, so a missing interval can
    never create a synthetic jump.
    """

    if embeddings.ndim not in (2, 3):
        raise ValueError("embeddings must have shape [time, dim] or [batch, time, dim]")
    velocities, velocity_valid = _embedding_velocity(embeddings, valid_mask)
    return temporal_component(velocities, velocity_valid)


def _embedding_velocity(
    embeddings: Tensor,
    valid_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Return detached first differences and their pairwise-valid mask."""

    unbatched = embeddings.ndim == 2
    values = embeddings.unsqueeze(0) if unbatched else embeddings
    batch, time, _ = values.shape
    if valid_mask is None:
        valid = torch.ones((batch, time), dtype=torch.bool, device=values.device)
    else:
        expected = (time,) if unbatched else (batch, time)
        if valid_mask.shape != expected:
            raise ValueError(
                f"valid_mask must have shape {expected}, got {tuple(valid_mask.shape)}"
            )
        valid = valid_mask.unsqueeze(0) if unbatched else valid_mask
        valid = valid.to(device=values.device, dtype=torch.bool)

    detached = values.detach().to(dtype=torch.float32)
    velocities = torch.zeros_like(detached)
    velocity_valid = torch.zeros((batch, time), dtype=torch.bool, device=values.device)
    if time > 1:
        adjacent_valid = valid[:, 1:] & valid[:, :-1]
        velocities[:, 1:] = (detached[:, 1:] - detached[:, :-1]) * adjacent_valid.unsqueeze(-1)
        velocity_valid[:, 1:] = adjacent_valid
    if unbatched:
        return velocities[0], velocity_valid[0]
    return velocities, velocity_valid


def autocorrelation_fft(
    signal: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Compute a mask-normalized, non-circular autocorrelation using FFT."""

    batched, unbatched = _as_batch_signal(signal)
    if batched.dtype not in (torch.float32, torch.float64):
        batched = batched.float()
    batched = torch.nan_to_num(batched)
    mask = _validated_mask(
        batched, valid_mask.unsqueeze(0) if unbatched and valid_mask is not None else valid_mask
    )
    weights = mask.to(dtype=batched.dtype)
    count = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (batched * weights).sum(dim=1, keepdim=True) / count
    centered = (batched - mean) * weights

    time = batched.shape[1]
    fft_length = 1 << max(1, (2 * time - 1).bit_length())
    spectrum = torch.fft.rfft(centered, n=fft_length, dim=1)
    numerator = torch.fft.irfft(spectrum.conj() * spectrum, n=fft_length, dim=1)[:, :time]
    mask_spectrum = torch.fft.rfft(weights, n=fft_length, dim=1)
    pair_count = torch.fft.irfft(
        mask_spectrum.conj() * mask_spectrum,
        n=fft_length,
        dim=1,
    )[:, :time]
    autocorrelation = numerator / pair_count.clamp_min(1.0)
    lag_zero = autocorrelation[:, :1].abs()
    autocorrelation = torch.where(
        lag_zero > 1e-12,
        autocorrelation / lag_zero.clamp_min(1e-12),
        torch.zeros_like(autocorrelation),
    )
    return autocorrelation[0] if unbatched else autocorrelation


def vector_autocorrelation_fft(
    sequence: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Compute an orthogonal-basis-invariant vector autocorrelation.

    The input is centered per feature over valid frames.  Every lag then uses
    the sum of signed cross-time dot products over all feature dimensions,
    computed as a zero-padded FFT correlation and divided by the exact number
    of valid frame pairs.  Unlike selecting one high-variance coordinate, this
    quantity is unchanged by an orthogonal change of feature basis.

    This is an independently inferred diagnostic primitive; it is not claimed
    as an author-disclosed PAMS component.
    """

    batched, unbatched = _as_batch_vectors(sequence)
    if batched.dtype not in (torch.float32, torch.float64):
        batched = batched.float()
    batched = torch.nan_to_num(batched.detach())
    batch, time, _ = batched.shape
    if valid_mask is None:
        mask = torch.ones((batch, time), dtype=torch.bool, device=batched.device)
    else:
        expected = (time,) if unbatched else (batch, time)
        if valid_mask.shape != expected:
            raise ValueError(
                f"valid_mask must have shape {expected}, got {tuple(valid_mask.shape)}"
            )
        mask = valid_mask.unsqueeze(0) if unbatched else valid_mask
        mask = mask.to(device=batched.device, dtype=torch.bool)

    weights = mask.to(dtype=batched.dtype)
    feature_weights = weights.unsqueeze(-1)
    counts = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    means = (batched * feature_weights).sum(dim=1, keepdim=True) / counts.unsqueeze(-1)
    centered = torch.nan_to_num((batched - means) * feature_weights)

    fft_length = 1 << max(1, (2 * time - 1).bit_length())
    spectrum = torch.fft.rfft(centered, n=fft_length, dim=1)
    # Parseval/Wiener-Khinchin over every feature at once.  Summed squared
    # spectral magnitudes are exactly the FFT representation of cross-time
    # vector dot products.
    cross_power = (spectrum.conj() * spectrum).sum(dim=-1)
    numerator = torch.fft.irfft(cross_power, n=fft_length, dim=1)[:, :time]
    mask_spectrum = torch.fft.rfft(weights, n=fft_length, dim=1)
    pair_count = torch.fft.irfft(
        mask_spectrum.conj() * mask_spectrum,
        n=fft_length,
        dim=1,
    )[:, :time]
    autocorrelation = numerator / pair_count.clamp_min(1.0)
    lag_zero = autocorrelation[:, :1].abs()
    autocorrelation = torch.where(
        lag_zero > 1e-12,
        autocorrelation / lag_zero.clamp_min(1e-12),
        torch.zeros_like(autocorrelation),
    )
    return autocorrelation[0] if unbatched else autocorrelation


def estimate_period_batch(
    signal: Tensor,
    minimum: int = 4,
    maximum: int = 128,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Estimate integer periods through the spectrum of autocorrelation.

    Returns:
        Integer periods and confidence scores, both shaped ``[batch]``.  The
        confidence is the selected bin's share of non-DC power in the allowed
        frequency band.
    """

    if minimum < 2:
        raise ValueError("minimum period must be at least 2")
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")
    batched, unbatched = _as_batch_signal(signal)
    if batched.dtype not in (torch.float32, torch.float64):
        batched = batched.float()
    mask = _validated_mask(
        batched,
        valid_mask.unsqueeze(0) if unbatched and valid_mask is not None else valid_mask,
    )
    autocorrelation = autocorrelation_fft(batched, mask)
    if autocorrelation.ndim == 1:
        autocorrelation = autocorrelation.unsqueeze(0)

    periods: list[float] = []
    confidences: list[Tensor] = []
    for sample_ac, sample_signal, sample_mask in zip(autocorrelation, batched, mask, strict=True):
        valid_length = int(sample_mask.sum())
        upper_period = min(maximum, max(minimum, valid_length - 1))
        if valid_length < minimum * 2:
            periods.append(float(minimum))
            confidences.append(sample_signal.new_tensor(0.0))
            continue

        # Windowing reduces leakage when the video contains a partial cycle.
        window = torch.hann_window(
            sample_ac.numel(),
            periodic=False,
            dtype=sample_ac.dtype,
            device=sample_ac.device,
        )
        power = torch.fft.rfft(sample_ac * window).abs().square()
        frequencies = torch.fft.rfftfreq(
            sample_ac.numel(),
            d=1.0,
            device=sample_ac.device,
        )
        allowed = (frequencies >= 1.0 / upper_period) & (frequencies <= 1.0 / minimum)
        allowed[0] = False
        band = power.masked_fill(~allowed, 0.0)
        total = band.sum()
        if not torch.isfinite(total) or float(total) <= 1e-12:
            periods.append(float(upper_period))
            confidences.append(sample_signal.new_tensor(0.0))
            continue

        index = int(torch.argmax(band))
        frequency = index / sample_ac.numel()
        # Preserve the FFT bin's fractional period (for example, 256/40 =
        # 6.4 frames). TCC/SSHead round only where integer indexing is
        # unavoidable, while inference smoothing and reference counting use
        # the disclosed dominant period without premature quantization.
        estimate = min(max(1.0 / frequency, float(minimum)), float(upper_period))
        periods.append(estimate)
        confidences.append((band[index] / total.clamp_min(1e-12)).clamp(0.0, 1.0))

    period_tensor = torch.tensor(periods, dtype=batched.dtype, device=batched.device)
    confidence_tensor = torch.stack(confidences).to(device=batched.device)
    if unbatched:
        return period_tensor[:1], confidence_tensor[:1]
    return period_tensor, confidence_tensor


def estimate_period_batch_direct_fft(
    signal: Tensor,
    minimum: int = 4,
    maximum: int = 128,
    valid_mask: Tensor | None = None,
    *,
    timebase: Literal["compact_valid", "dense_resampled"] = "compact_valid",
    timeline_lengths: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Estimate a dominant period directly from the observed action curve.

    PAMS Algorithm 1 applies an FFT to the period stream itself.  The legacy
    reproduction instead transformed the stream autocorrelation and then
    windowed that lag signal, which is a different decoder and can bias a weak
    stream toward the longest allowed period.  This implementation follows the
    disclosed inference order while making the usual, explicit signal-processing
    choices: affine detrending, one Hann window, and a bounded non-DC
    frequency band.

    ``compact_valid`` is the historical behavior: invalid samples are removed
    before detrending and the FFT, so the returned period is measured in valid
    samples.  ``dense_resampled`` keeps the resampled frame clock.  It fits the
    affine trend using only valid observations at their original dense indices,
    contributes exact zeros at invalid locations, and performs the FFT over
    each sample's explicit ``timeline_lengths`` extent.  Its returned period is
    therefore measured in dense resampled frames.
    """

    if minimum < 2:
        raise ValueError("minimum period must be at least 2")
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")
    if timebase not in {"compact_valid", "dense_resampled"}:
        raise ValueError(
            "timebase must be 'compact_valid' or 'dense_resampled'"
        )
    batched, unbatched = _as_batch_signal(signal)
    if batched.dtype not in (torch.float32, torch.float64):
        batched = batched.float()
    mask = _validated_mask(
        batched,
        (
            valid_mask.unsqueeze(0)
            if unbatched and valid_mask is not None
            else valid_mask
        ),
    )
    if timebase == "compact_valid":
        if timeline_lengths is not None:
            raise ValueError(
                "timeline_lengths is only valid with timebase='dense_resampled'"
            )
        lengths: Tensor | None = None
    else:
        if timeline_lengths is None:
            raise ValueError(
                "timeline_lengths is required with timebase='dense_resampled'"
            )
        if not isinstance(timeline_lengths, Tensor):
            raise TypeError("timeline_lengths must be a torch.Tensor")
        if timeline_lengths.shape != (batched.shape[0],):
            raise ValueError(
                "timeline_lengths must have shape "
                f"({batched.shape[0]},), got {tuple(timeline_lengths.shape)}"
            )
        if timeline_lengths.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("timeline_lengths must have an integer dtype")
        lengths = timeline_lengths.to(device=batched.device, dtype=torch.long)
        if bool(torch.any(lengths < 1)) or bool(
            torch.any(lengths > batched.shape[1])
        ):
            raise ValueError(
                "timeline_lengths values must lie within the signal time dimension"
            )
        frame_indices = torch.arange(batched.shape[1], device=batched.device)
        outside_timeline = frame_indices.unsqueeze(0) >= lengths.unsqueeze(1)
        if bool(torch.any(mask & outside_timeline)):
            raise ValueError(
                "valid_mask cannot mark samples beyond timeline_lengths as valid"
            )

    periods: list[float] = []
    confidences: list[Tensor] = []
    for sample_index, (sample_signal, sample_mask) in enumerate(
        zip(batched, mask, strict=True)
    ):
        if timebase == "compact_valid":
            selected = sample_signal[sample_mask]
            valid_length = int(selected.numel())
            fft_length = valid_length
            trend_time = torch.arange(
                valid_length,
                dtype=selected.dtype,
                device=selected.device,
            )
        else:
            assert lengths is not None
            fft_length = int(lengths[sample_index])
            timeline_mask = sample_mask[:fft_length]
            timeline_signal = sample_signal[:fft_length]
            valid_length = int(timeline_mask.sum())
            selected = timeline_signal[timeline_mask]
            trend_time = torch.arange(
                fft_length,
                dtype=selected.dtype,
                device=selected.device,
            )[timeline_mask]

        upper_period = min(maximum, max(minimum, fft_length - 1))
        if valid_length < minimum * 2:
            periods.append(float(minimum))
            confidences.append(sample_signal.new_tensor(0.0))
            continue
        if timebase == "dense_resampled" and not bool(
            torch.isfinite(selected).all()
        ):
            raise ValueError("valid dense-resampled signal samples must be finite")

        centered_time = trend_time - trend_time.mean()
        centered = selected - selected.mean()
        slope = (centered_time * centered).sum() / centered_time.square().sum().clamp_min(
            1e-12
        )
        detrended_valid = centered - slope * centered_time
        if float(detrended_valid.square().mean()) <= 1e-12:
            periods.append(float(upper_period))
            confidences.append(sample_signal.new_tensor(0.0))
            continue

        if timebase == "compact_valid":
            detrended = detrended_valid
        else:
            detrended = torch.zeros(
                fft_length,
                dtype=detrended_valid.dtype,
                device=detrended_valid.device,
            )
            detrended[timeline_mask] = detrended_valid

        window = torch.hann_window(
            fft_length,
            periodic=False,
            dtype=detrended.dtype,
            device=detrended.device,
        )
        power = torch.fft.rfft(detrended * window).abs().square()
        frequencies = torch.fft.rfftfreq(
            fft_length,
            d=1.0,
            device=detrended.device,
        )
        allowed = (frequencies >= 1.0 / upper_period) & (
            frequencies <= 1.0 / minimum
        )
        allowed[0] = False
        band = power.masked_fill(~allowed, 0.0)
        total = band.sum()
        if not torch.isfinite(total) or float(total) <= 1e-12:
            periods.append(float(upper_period))
            confidences.append(sample_signal.new_tensor(0.0))
            continue

        index = int(torch.argmax(band))
        frequency = index / fft_length
        estimate = min(
            max(1.0 / frequency, float(minimum)),
            float(upper_period),
        )
        periods.append(estimate)
        confidences.append((band[index] / total).clamp(0.0, 1.0))

    period_tensor = torch.tensor(
        periods,
        dtype=batched.dtype,
        device=batched.device,
    )
    confidence_tensor = torch.stack(confidences).to(device=batched.device)
    if unbatched:
        return period_tensor[:1], confidence_tensor[:1]
    return period_tensor, confidence_tensor


def estimate_period_from_vectors(
    sequence: Tensor,
    minimum: int = 4,
    maximum: int = 128,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Estimate periods from the full signed vector autocorrelation.

    This follows the same bounded ACF-spectrum rule as
    :func:`estimate_period_batch`, but its evidence is the cross-dimensional
    dot-product sum rather than a selected scalar coordinate.  Zero or
    constant valid inputs have exactly zero confidence.
    """

    if minimum < 2:
        raise ValueError("minimum period must be at least 2")
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")
    batched, unbatched = _as_batch_vectors(sequence)
    if batched.dtype not in (torch.float32, torch.float64):
        batched = batched.float()
    batch, time, _ = batched.shape
    if valid_mask is None:
        mask = torch.ones((batch, time), dtype=torch.bool, device=batched.device)
    else:
        expected = (time,) if unbatched else (batch, time)
        if valid_mask.shape != expected:
            raise ValueError(
                f"valid_mask must have shape {expected}, got {tuple(valid_mask.shape)}"
            )
        mask = valid_mask.unsqueeze(0) if unbatched else valid_mask
        mask = mask.to(device=batched.device, dtype=torch.bool)

    autocorrelation = vector_autocorrelation_fft(batched, mask)
    if autocorrelation.ndim == 1:
        autocorrelation = autocorrelation.unsqueeze(0)

    periods: list[float] = []
    confidences: list[Tensor] = []
    for sample_ac, sample_mask in zip(autocorrelation, mask, strict=True):
        valid_length = int(sample_mask.sum())
        upper_period = min(maximum, max(minimum, valid_length - 1))
        if valid_length < minimum * 2:
            periods.append(float(minimum))
            confidences.append(sample_ac.new_tensor(0.0))
            continue

        window = torch.hann_window(
            sample_ac.numel(),
            periodic=False,
            dtype=sample_ac.dtype,
            device=sample_ac.device,
        )
        power = torch.fft.rfft(sample_ac * window).abs().square()
        frequencies = torch.fft.rfftfreq(
            sample_ac.numel(),
            d=1.0,
            device=sample_ac.device,
        )
        allowed = (frequencies >= 1.0 / upper_period) & (
            frequencies <= 1.0 / minimum
        )
        allowed[0] = False
        band = power.masked_fill(~allowed, 0.0)
        total = band.sum()
        if not torch.isfinite(total) or float(total) <= 1e-12:
            periods.append(float(upper_period))
            confidences.append(sample_ac.new_tensor(0.0))
            continue

        index = int(torch.argmax(band))
        frequency = index / sample_ac.numel()
        estimate = min(
            max(1.0 / frequency, float(minimum)),
            float(upper_period),
        )
        periods.append(estimate)
        confidences.append((band[index] / total.clamp_min(1e-12)).clamp(0.0, 1.0))

    period_tensor = torch.tensor(
        periods,
        dtype=batched.dtype,
        device=batched.device,
    )
    confidence_tensor = torch.stack(confidences).to(device=batched.device)
    if unbatched:
        return period_tensor[:1], confidence_tensor[:1]
    return period_tensor, confidence_tensor


def estimate_period(
    signal: Tensor,
    minimum: int = 4,
    maximum: int = 128,
    valid_mask: Tensor | None = None,
) -> PeriodEstimate:
    """Return a scalar diagnostic estimate for one activity signal."""

    if signal.ndim != 1:
        raise ValueError("estimate_period expects a one-dimensional signal")
    periods, confidences = estimate_period_batch(
        signal,
        minimum=minimum,
        maximum=maximum,
        valid_mask=valid_mask,
    )
    mask = (
        torch.ones_like(signal, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.to(dtype=torch.bool)
    )
    period = float(periods[0])
    return PeriodEstimate(
        period=period,
        confidence=float(confidences[0]),
        frequency=1.0 / period,
        valid_length=int(mask.sum()),
    )


def estimate_period_from_pose(
    poses: Tensor,
    minimum: int = 4,
    maximum: int = 128,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Estimate periods from the warm-up pose proxy."""

    energy = pose_energy(poses, valid_mask)
    return estimate_period_batch(energy, minimum, maximum, valid_mask)


def estimate_period_from_embeddings(
    embeddings: Tensor,
    minimum: int = 4,
    maximum: int = 128,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Estimate periods from stop-gradient encoder embeddings."""

    velocities, velocity_valid = _embedding_velocity(embeddings, valid_mask)
    energy = temporal_component(velocities, velocity_valid)
    return estimate_period_batch(energy, minimum, maximum, velocity_valid)


def estimate_period_from_embedding_velocity_vectors(
    embeddings: Tensor,
    minimum: int = 4,
    maximum: int = 128,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Estimate periods from the full stop-gradient embedding velocity.

    Unlike :func:`estimate_period_from_embeddings`, this route never selects
    or otherwise reduces the embedding velocity to one scalar coordinate.
    It reuses the pairwise-valid first differences from
    :func:`_embedding_velocity` and the orthogonal-basis-invariant vector ACF
    spectrum from :func:`estimate_period_from_vectors`.  Consequently,
    missing-frame boundaries cannot introduce synthetic motion, invalid
    values do not enter the estimate, and neither returned tensor retains an
    autograd path to the encoder embeddings.
    """

    velocities, velocity_valid = _embedding_velocity(embeddings, valid_mask)
    return estimate_period_from_vectors(
        velocities,
        minimum=minimum,
        maximum=maximum,
        valid_mask=velocity_valid,
    )


def embedding_velocity_harmonic_fundamental_diagnostics(
    embeddings: Tensor,
    minimum: int = 4,
    maximum: int = 128,
    valid_mask: Tensor | None = None,
    *,
    maximum_harmonic: int = 8,
    fundamental_only_weight: float = 0.001,
    local_bin_radius: int = 0,
) -> tuple[HarmonicFundamentalDiagnostic, ...]:
    """Infer a fundamental from full embedding-velocity harmonic evidence.

    This is an independently inferred readout, not an author-disclosed PAMS
    component.  It deliberately leaves :func:`estimate_period_from_embeddings`
    unchanged.

    For every bounded fundamental-frequency candidate, the selected-bin power
    is multiplied with the mean power over all *available* overtones up to
    ``maximum_harmonic``.  Taking the square root gives an HPS-like
    fundamental/overtone consensus.  A small fundamental-only term preserves
    a deterministic pure-sinusoid fallback.  Averaging the overtone power by
    its available count prevents long-period candidates from winning merely
    because more harmonics fit below Nyquist.

    Confidence is the product of two bounded quantities: normalized
    candidate-distribution concentration (Herfindahl excess over uniform) and
    the selected harmonic family's spectral-power excess over its uniform
    bin-count expectation.  Exact zero input therefore has zero confidence,
    while unstructured white noise is not assigned the confidence of a
    concentrated periodic spectrum.
    """

    if embeddings.ndim not in (2, 3):
        raise ValueError(
            "embeddings must have shape [time, dim] or [batch, time, dim]"
        )
    if minimum < 2:
        raise ValueError("minimum period must be at least 2")
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")
    if isinstance(maximum_harmonic, bool) or maximum_harmonic < 2:
        raise ValueError("maximum_harmonic must be an integer of at least 2")
    if not isinstance(maximum_harmonic, int):
        raise TypeError("maximum_harmonic must be an integer")
    if (
        not isinstance(fundamental_only_weight, int | float)
        or isinstance(fundamental_only_weight, bool)
        or not torch.isfinite(torch.tensor(float(fundamental_only_weight)))
        or not 0.0 <= float(fundamental_only_weight) <= 1.0
    ):
        raise ValueError("fundamental_only_weight must be finite and in [0, 1]")
    if (
        isinstance(local_bin_radius, bool)
        or not isinstance(local_bin_radius, int)
        or local_bin_radius < 0
    ):
        raise ValueError("local_bin_radius must be a non-negative integer")

    batched, unbatched = _as_batch_vectors(embeddings)
    if batched.dtype not in (torch.float32, torch.float64):
        batched = batched.float()
    values = torch.nan_to_num(
        batched.detach(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    batch, time, _ = values.shape
    if valid_mask is None:
        mask = torch.ones((batch, time), dtype=torch.bool, device=values.device)
    else:
        expected = (time,) if unbatched else (batch, time)
        if valid_mask.shape != expected:
            raise ValueError(
                f"valid_mask must have shape {expected}, got {tuple(valid_mask.shape)}"
            )
        mask = valid_mask.unsqueeze(0) if unbatched else valid_mask
        mask = mask.to(device=values.device, dtype=torch.bool)

    velocities, velocity_valid = _embedding_velocity(values, mask)
    if velocities.ndim == 2:
        velocities = velocities.unsqueeze(0)
        velocity_valid = velocity_valid.unsqueeze(0)
    diagnostics: list[HarmonicFundamentalDiagnostic] = []
    epsilon = torch.finfo(values.dtype).eps

    for sample_velocity, sample_mask in zip(
        velocities,
        velocity_valid,
        strict=True,
    ):
        valid_length = int(sample_mask.sum())
        upper_period = min(maximum, max(minimum, valid_length - 1))
        empty = HarmonicFundamentalDiagnostic(
            valid_length=valid_length,
            fft_length=time,
            candidate_bins=(),
            candidate_periods=(),
            candidate_scores=(),
            selected_bin=None,
            selected_period=float(minimum),
            prewhitened_bin=None,
            prewhitened_residual_power_fraction=0.0,
            selected_harmonic_bins=(),
            selected_harmonic_power_shares=(),
            candidate_distribution_concentration=0.0,
            family_spectral_concentration=0.0,
            uniform_family_expectation=0.0,
            overtone_noise_floor=0.0,
            confidence=0.0,
        )
        if valid_length < minimum * 2 or time < minimum * 2:
            diagnostics.append(empty)
            continue

        weights = sample_mask.to(dtype=values.dtype)
        count = weights.sum().clamp_min(1.0)
        mean = (sample_velocity * weights.unsqueeze(-1)).sum(dim=0) / count
        centered = torch.nan_to_num(
            (sample_velocity - mean) * weights.unsqueeze(-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        window = torch.hann_window(
            time,
            periodic=False,
            dtype=values.dtype,
            device=values.device,
        )
        spectrum = torch.fft.rfft(centered * window.unsqueeze(-1), dim=0)
        power = torch.nan_to_num(
            spectrum.abs().square().sum(dim=-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        if power.numel() > 0:
            power[0] = 0.0
        total_power = power.sum()
        if not bool(torch.isfinite(total_power)) or float(total_power) <= float(epsilon):
            diagnostics.append(empty)
            continue
        power_share = power / total_power.clamp_min(epsilon)
        nyquist_bin = power_share.numel() - 1
        non_dc_bin_total = max(power_share.numel() - 1, 1)
        overtone_noise_floor = 0.10 / non_dc_bin_total
        dominant_bin = int(torch.argmax(power_share[1:])) + 1

        # Remove the single dominant sinusoid by mask-weighted least squares.
        # The residual spectrum exposes a weak fundamental that a 3rd/5th/7th
        # harmonic can otherwise hide.  The original spectrum is retained for
        # overtone evidence and for the pure-sinusoid fallback.
        frame_index = torch.arange(
            time,
            dtype=values.dtype,
            device=values.device,
        )
        phase = (2.0 * math.pi * dominant_bin / time) * frame_index
        design = torch.stack(
            (
                torch.cos(phase),
                torch.sin(phase),
                torch.ones_like(phase),
            ),
            dim=-1,
        )
        weighted_design = design * weights.unsqueeze(-1)
        gram = design.transpose(0, 1) @ weighted_design
        regularizer = epsilon * gram.diagonal().abs().max().clamp_min(1.0)
        gram = gram + regularizer * torch.eye(
            3,
            dtype=values.dtype,
            device=values.device,
        )
        coefficients = torch.linalg.solve(
            gram,
            weighted_design.transpose(0, 1) @ centered,
        )
        residual = (
            centered - design @ coefficients
        ) * weights.unsqueeze(-1)
        residual_spectrum = torch.fft.rfft(
            residual * window.unsqueeze(-1),
            dim=0,
        )
        residual_power = torch.nan_to_num(
            residual_spectrum.abs().square().sum(dim=-1),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        residual_power[0] = 0.0
        residual_total = residual_power.sum()
        residual_fraction = float(
            (residual_total / total_power.clamp_min(epsilon)).clamp(0.0, 1.0)
        )
        if (
            bool(torch.isfinite(residual_total))
            and float(residual_total) > float(epsilon)
            and residual_fraction >= 1e-6
        ):
            residual_share = residual_power / residual_total.clamp_min(epsilon)
        else:
            residual_share = torch.zeros_like(power_share)
        minimum_bin = max(1, math.ceil(time / float(upper_period)))
        maximum_bin = min(nyquist_bin, math.floor(time / float(minimum)))
        candidate_bins = tuple(range(minimum_bin, maximum_bin + 1))
        if not candidate_bins:
            diagnostics.append(empty)
            continue

        candidate_scores: list[Tensor] = []
        candidate_peak_bins: list[tuple[int, ...]] = []
        candidate_peak_shares: list[tuple[Tensor, ...]] = []
        for candidate_bin in candidate_bins:
            harmonic_bins: list[int] = []
            harmonic_shares: list[Tensor] = []
            available_harmonic = min(
                maximum_harmonic,
                nyquist_bin // candidate_bin,
            )
            for order in range(1, available_harmonic + 1):
                center_bin = order * candidate_bin
                lower = max(1, center_bin - local_bin_radius)
                upper = min(nyquist_bin, center_bin + local_bin_radius)
                local = power_share[lower : upper + 1]
                relative_peak = int(torch.argmax(local))
                peak_bin = lower + relative_peak
                harmonic_bins.append(peak_bin)
                harmonic_shares.append(power_share[peak_bin])
            original_fundamental_share = harmonic_shares[0]
            residual_lower = max(1, candidate_bin - local_bin_radius)
            residual_upper = min(nyquist_bin, candidate_bin + local_bin_radius)
            residual_fundamental_share = residual_share[
                residual_lower : residual_upper + 1
            ].max()
            fundamental_share = torch.maximum(
                original_fundamental_share,
                residual_fundamental_share,
            )
            if len(harmonic_shares) > 1:
                # The mean, rather than a raw sum, is the explicit
                # available-harmonic normalization. A fixed tenth of the
                # uniform-bin expectation is subtracted from every overtone:
                # this prevents numerical leakage beside one dominant peak
                # from masquerading as a second harmonic, without erasing a
                # weak candidate fundamental.
                overtone_mean = (
                    torch.stack(harmonic_shares[1:])
                    .sub(overtone_noise_floor)
                    .clamp_min(0.0)
                    .mean()
                )
            else:
                overtone_mean = fundamental_share.new_tensor(0.0)
            consensus = torch.sqrt(
                fundamental_share.clamp_min(0.0)
                * overtone_mean.clamp_min(0.0)
            )
            score = consensus + (
                float(fundamental_only_weight) * fundamental_share
            )
            candidate_scores.append(
                torch.nan_to_num(
                    score,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            )
            candidate_peak_bins.append(tuple(harmonic_bins))
            candidate_peak_shares.append(tuple(harmonic_shares))

        score_tensor = torch.stack(candidate_scores).clamp_min(0.0)
        score_total = score_tensor.sum()
        if not bool(torch.isfinite(score_total)) or float(score_total) <= float(epsilon):
            diagnostics.append(empty)
            continue
        maximum_score = float(score_tensor.max())
        # A deterministic shorter-period tie break is used only for numerically
        # equal scores; genuine lower-frequency harmonic consensus dominates
        # the small fundamental-only fallback by score.
        score_tolerance = max(float(epsilon) * max(maximum_score, 1.0) * 8.0, 1e-15)
        tied = [
            index
            for index, score in enumerate(score_tensor.detach().cpu().tolist())
            if maximum_score - float(score) <= score_tolerance
        ]
        selected_index = max(tied, key=lambda index: candidate_bins[index])
        selected_bin = candidate_bins[selected_index]
        selected_period = min(
            max(time / float(selected_bin), float(minimum)),
            float(upper_period),
        )

        probabilities = score_tensor / score_total.clamp_min(epsilon)
        candidate_total = probabilities.numel()
        if candidate_total > 1:
            uniform_candidate = 1.0 / candidate_total
            hhi = probabilities.square().sum()
            hhi_excess = (
                (hhi - uniform_candidate) / (1.0 - uniform_candidate)
            ).clamp(0.0, 1.0)
            candidate_concentration = torch.sqrt(hhi_excess)
        else:
            candidate_concentration = probabilities.new_tensor(1.0)

        selected_bins = candidate_peak_bins[selected_index]
        selected_shares = candidate_peak_shares[selected_index]
        unique_selected_bins = tuple(dict.fromkeys(selected_bins))
        family_share = power_share[
            torch.tensor(
                unique_selected_bins,
                dtype=torch.long,
                device=power_share.device,
            )
        ].sum().clamp(0.0, 1.0)
        uniform_family = min(
            len(unique_selected_bins) / non_dc_bin_total,
            1.0,
        )
        if uniform_family < 1.0:
            family_excess = (
                (family_share - uniform_family) / (1.0 - uniform_family)
            ).clamp(0.0, 1.0)
        else:
            family_excess = family_share.new_tensor(0.0)
        confidence = (candidate_concentration * family_excess).clamp(0.0, 1.0)

        diagnostics.append(
            HarmonicFundamentalDiagnostic(
                valid_length=valid_length,
                fft_length=time,
                candidate_bins=candidate_bins,
                candidate_periods=tuple(
                    min(
                        max(time / float(candidate_bin), float(minimum)),
                        float(upper_period),
                    )
                    for candidate_bin in candidate_bins
                ),
                candidate_scores=tuple(
                    float(value) for value in score_tensor.detach().cpu()
                ),
                selected_bin=selected_bin,
                selected_period=selected_period,
                prewhitened_bin=dominant_bin,
                prewhitened_residual_power_fraction=residual_fraction,
                selected_harmonic_bins=selected_bins,
                selected_harmonic_power_shares=tuple(
                    float(value) for value in selected_shares
                ),
                candidate_distribution_concentration=float(
                    candidate_concentration
                ),
                family_spectral_concentration=float(family_share),
                uniform_family_expectation=uniform_family,
                overtone_noise_floor=overtone_noise_floor,
                confidence=float(confidence),
            )
        )
    return tuple(diagnostics)


def estimate_harmonic_fundamental_from_embeddings(
    embeddings: Tensor,
    minimum: int = 4,
    maximum: int = 128,
    valid_mask: Tensor | None = None,
    *,
    maximum_harmonic: int = 8,
    fundamental_only_weight: float = 0.001,
    local_bin_radius: int = 0,
) -> tuple[Tensor, Tensor]:
    """Return the opt-in inferred embedding-velocity harmonic fundamental."""

    batched, unbatched = _as_batch_vectors(embeddings)
    if batched.dtype not in (torch.float32, torch.float64):
        batched = batched.float()
    diagnostics = embedding_velocity_harmonic_fundamental_diagnostics(
        embeddings,
        minimum=minimum,
        maximum=maximum,
        valid_mask=valid_mask,
        maximum_harmonic=maximum_harmonic,
        fundamental_only_weight=fundamental_only_weight,
        local_bin_radius=local_bin_radius,
    )
    periods = torch.tensor(
        [diagnostic.selected_period for diagnostic in diagnostics],
        dtype=batched.dtype,
        device=batched.device,
    )
    confidences = torch.tensor(
        [diagnostic.confidence for diagnostic in diagnostics],
        dtype=batched.dtype,
        device=batched.device,
    )
    if unbatched:
        return periods[:1], confidences[:1]
    return periods, confidences


def estimate_period_from_projected_pose(
    projected_pose: Tensor,
    minimum: int = 4,
    maximum: int = 128,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Estimate period from pre-position-encoding projected-pose velocity.

    The linear projection is intentionally sampled before positional encoding
    and the Transformer.  First differences remove a constant projection bias,
    and the full vector ACF avoids the historical selected-coordinate shortcut.
    This route is inferred and must remain opt-in.
    """

    velocities, velocity_valid = _embedding_velocity(projected_pose, valid_mask)
    return estimate_period_from_vectors(
        velocities,
        minimum=minimum,
        maximum=maximum,
        valid_mask=velocity_valid,
    )


def projected_position_vector_acf_diagnostics(
    projected_pose: Tensor,
    minimum: int = 4,
    maximum: int = 128,
    valid_mask: Tensor | None = None,
) -> tuple[ProjectedPositionSpectrumDiagnostic, ...]:
    """Return mask-aware vector-ACF spectrum evidence without differencing.

    The projected coordinates are centered inside
    :func:`vector_autocorrelation_fft`, so a constant projection bias cannot
    affect the result.  Keeping positions instead of first differences avoids
    the latter's deterministic high-frequency gain.  This independently
    inferred diagnostic is intentionally separate from the frozen projected
    velocity route.

    One diagnostic is returned per batch item, including for an unbatched
    input.  ``allowed_bins`` contains the absolute rFFT bin indices in the
    valid period band, and ``power_shares`` is normalized over exactly those
    bins.  Inputs without usable evidence have zero shares, zero confidence,
    and no selected bin.
    """

    if minimum < 2:
        raise ValueError("minimum period must be at least 2")
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")
    batched, unbatched = _as_batch_vectors(projected_pose)
    if batched.dtype not in (torch.float32, torch.float64):
        batched = batched.float()
    batch, time, _ = batched.shape
    if valid_mask is None:
        mask = torch.ones((batch, time), dtype=torch.bool, device=batched.device)
    else:
        expected = (time,) if unbatched else (batch, time)
        if valid_mask.shape != expected:
            raise ValueError(
                f"valid_mask must have shape {expected}, got {tuple(valid_mask.shape)}"
            )
        mask = valid_mask.unsqueeze(0) if unbatched else valid_mask
        mask = mask.to(device=batched.device, dtype=torch.bool)

    autocorrelation = vector_autocorrelation_fft(batched, mask)
    if autocorrelation.ndim == 1:
        autocorrelation = autocorrelation.unsqueeze(0)

    diagnostics: list[ProjectedPositionSpectrumDiagnostic] = []
    for sample_ac, sample_mask in zip(autocorrelation, mask, strict=True):
        valid_length = int(sample_mask.sum())
        frequencies = torch.fft.rfftfreq(
            sample_ac.numel(),
            d=1.0,
            dtype=sample_ac.dtype,
            device=sample_ac.device,
        )
        allowed = (frequencies >= 1.0 / maximum) & (frequencies <= 1.0 / minimum)
        allowed[0] = False
        allowed_indices = torch.nonzero(allowed, as_tuple=False).flatten()
        allowed_frequencies = frequencies[allowed_indices]
        allowed_periods = allowed_frequencies.reciprocal()
        zero_shares = torch.zeros_like(allowed_frequencies)

        selected_bin: int | None = None
        selected_period = float(minimum)
        confidence = 0.0
        shares = zero_shares
        if valid_length >= minimum * 2:
            window = torch.hann_window(
                sample_ac.numel(),
                periodic=False,
                dtype=sample_ac.dtype,
                device=sample_ac.device,
            )
            power = torch.fft.rfft(sample_ac * window).abs().square()
            band = power.masked_fill(~allowed, 0.0)
            total = band.sum()
            if torch.isfinite(total) and float(total) > 1e-12:
                selected_bin = int(torch.argmax(band))
                frequency = selected_bin / sample_ac.numel()
                selected_period = min(
                    max(1.0 / frequency, float(minimum)),
                    float(maximum),
                )
                shares = band[allowed_indices] / total.clamp_min(1e-12)
                confidence = float(band[selected_bin] / total.clamp_min(1e-12))
            else:
                selected_period = float(maximum)

        diagnostics.append(
            ProjectedPositionSpectrumDiagnostic(
                valid_length=valid_length,
                allowed_bins=tuple(int(value) for value in allowed_indices.detach().cpu()),
                frequencies=tuple(float(value) for value in allowed_frequencies.detach().cpu()),
                periods=tuple(float(value) for value in allowed_periods.detach().cpu()),
                power_shares=tuple(float(value) for value in shares.detach().cpu()),
                selected_bin=selected_bin,
                selected_period=selected_period,
                confidence=confidence,
            )
        )
    return tuple(diagnostics)


def estimate_period_from_projected_position(
    projected_pose: Tensor,
    minimum: int = 4,
    maximum: int = 128,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Estimate projected-position periods through a full vector ACF.

    This opt-in inferred route deliberately does not call
    :func:`_embedding_velocity`.  Its output shape and dtype follow the
    existing projected-pose estimator while its auditable spectrum is exposed
    separately by :func:`projected_position_vector_acf_diagnostics`.
    """

    batched, unbatched = _as_batch_vectors(projected_pose)
    if batched.dtype not in (torch.float32, torch.float64):
        batched = batched.float()
    diagnostics = projected_position_vector_acf_diagnostics(
        projected_pose,
        minimum=minimum,
        maximum=maximum,
        valid_mask=valid_mask,
    )
    periods = torch.tensor(
        [diagnostic.selected_period for diagnostic in diagnostics],
        dtype=batched.dtype,
        device=batched.device,
    )
    confidences = torch.tensor(
        [diagnostic.confidence for diagnostic in diagnostics],
        dtype=batched.dtype,
        device=batched.device,
    )
    if unbatched:
        return periods[:1], confidences[:1]
    return periods, confidences


def linear_detrend_projected_position(
    projected_pose: Tensor,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Remove a per-feature affine time trend using valid frames only.

    Each batch item and projected feature receives its own least-squares
    intercept and slope.  Frame indices retain their original locations when
    the mask contains gaps; invalid values never enter the fit and are exact
    zero in the returned residual.  Inputs with fewer than two valid frame
    locations reduce to intercept-only fitting and therefore contain no
    residual evidence.

    This is an independently inferred, non-gradient diagnostic transform.  It
    is deliberately separate from both raw projected-position ACF and the
    frozen projected-velocity route.
    """

    batched, unbatched = _as_batch_vectors(projected_pose)
    if batched.dtype not in (torch.float32, torch.float64):
        batched = batched.float()
    values = torch.nan_to_num(batched.detach())
    batch, time, _ = values.shape
    if valid_mask is None:
        mask = torch.ones((batch, time), dtype=torch.bool, device=values.device)
    else:
        expected = (time,) if unbatched else (batch, time)
        if valid_mask.shape != expected:
            raise ValueError(
                f"valid_mask must have shape {expected}, got {tuple(valid_mask.shape)}"
            )
        mask = valid_mask.unsqueeze(0) if unbatched else valid_mask
        mask = mask.to(device=values.device, dtype=torch.bool)

    weights = mask.to(dtype=values.dtype)
    feature_weights = weights.unsqueeze(-1)
    valid_values = values.masked_fill(~mask.unsqueeze(-1), 0.0)
    counts = weights.sum(dim=1, keepdim=True)
    safe_counts = counts.clamp_min(1.0)
    frame_index = torch.arange(
        time,
        dtype=values.dtype,
        device=values.device,
    ).view(1, time)
    mean_index = (frame_index * weights).sum(dim=1, keepdim=True) / safe_counts
    centered_index = (frame_index - mean_index) * weights
    mean_value = valid_values.sum(dim=1, keepdim=True) / safe_counts.unsqueeze(-1)
    denominator = centered_index.square().sum(dim=1)
    numerator = (
        centered_index.unsqueeze(-1) * (valid_values - mean_value)
    ).sum(dim=1)
    epsilon = torch.finfo(values.dtype).eps
    slope = numerator / denominator.clamp_min(epsilon).unsqueeze(-1)
    usable_slope = (counts.squeeze(1) >= 2) & (denominator > epsilon)
    slope = slope * usable_slope.to(values.dtype).unsqueeze(-1)
    fitted = mean_value + (
        (frame_index - mean_index).unsqueeze(-1) * slope.unsqueeze(1)
    )
    residual = (values - fitted) * feature_weights
    residual = torch.nan_to_num(residual).masked_fill(
        ~mask.unsqueeze(-1),
        0.0,
    )
    # A mathematically affine float32 sequence can retain a few ulps of
    # quantization residue after least squares.  Treat only that
    # representation-scale floor as exact zero so it cannot become a
    # high-confidence spectrum after lag-zero normalization.
    feature_scale = valid_values.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    numerical_floor = 32.0 * torch.finfo(values.dtype).eps * feature_scale
    residual = residual.masked_fill(residual.abs() <= numerical_floor, 0.0)
    return residual[0] if unbatched else residual


def projected_position_detrended_vector_acf_diagnostics(
    projected_pose: Tensor,
    minimum: int = 4,
    maximum: int = 128,
    valid_mask: Tensor | None = None,
) -> tuple[ProjectedPositionSpectrumDiagnostic, ...]:
    """Return the standard position-ACF diagnostic after linear detrending.

    The transform changes only the evidence supplied to the existing
    mask-aware vector ACF.  Hann windowing, the fixed 4--128 period band,
    confidence definition, no-evidence behavior, and diagnostic fields are
    exactly those of :func:`projected_position_vector_acf_diagnostics`.
    """

    detrended = linear_detrend_projected_position(
        projected_pose,
        valid_mask=valid_mask,
    )
    return projected_position_vector_acf_diagnostics(
        detrended,
        minimum=minimum,
        maximum=maximum,
        valid_mask=valid_mask,
    )


def estimate_period_from_detrended_projected_position(
    projected_pose: Tensor,
    minimum: int = 4,
    maximum: int = 128,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Estimate period from linearly detrended projected positions."""

    batched, unbatched = _as_batch_vectors(projected_pose)
    if batched.dtype not in (torch.float32, torch.float64):
        batched = batched.float()
    diagnostics = projected_position_detrended_vector_acf_diagnostics(
        projected_pose,
        minimum=minimum,
        maximum=maximum,
        valid_mask=valid_mask,
    )
    periods = torch.tensor(
        [diagnostic.selected_period for diagnostic in diagnostics],
        dtype=batched.dtype,
        device=batched.device,
    )
    confidences = torch.tensor(
        [diagnostic.confidence for diagnostic in diagnostics],
        dtype=batched.dtype,
        device=batched.device,
    )
    if unbatched:
        return periods[:1], confidences[:1]
    return periods, confidences


def projected_position_lag_velocity_fallback_diagnostics(
    projected_pose: Tensor,
    minimum: int = 4,
    maximum: int = 128,
    valid_mask: Tensor | None = None,
) -> tuple[ProjectedPositionLagVelocityFallbackDiagnostic, ...]:
    """Select a detrended position-ACF lag, with velocity fallback.

    For each sample, projected positions are linearly detrended per feature
    using valid frames only.  After computing the mask-normalized vector
    autocorrelation, :func:`scipy.signal.find_peaks` is applied to the
    complete ACF with ``distance=minimum`` and zero prominence.  Its peaks are
    then filtered to
    lags ``minimum..min(maximum, valid_length // 2)`` and strictly positive
    heights.  Among peaks whose height is at least 90% of the greatest
    positive height, the smallest lag wins.  This fixed near-best rule avoids
    selecting a repeated multiple merely because finite-sample ACF height
    increases slightly at a later cycle.

    If no positive local peak exists, the existing raw projected-position
    pairwise-valid velocity spectrum is used.  A non-positive fallback
    confidence is treated as no evidence and returns ``minimum`` with zero
    confidence.  This is an independently inferred, target-free readout.
    """

    if minimum < 2:
        raise ValueError("minimum period must be at least 2")
    if maximum <= minimum:
        raise ValueError("maximum must be greater than minimum")
    batched, unbatched = _as_batch_vectors(projected_pose)
    if batched.dtype not in (torch.float32, torch.float64):
        batched = batched.float()
    batch, time, _ = batched.shape
    if valid_mask is None:
        mask = torch.ones((batch, time), dtype=torch.bool, device=batched.device)
    else:
        expected = (time,) if unbatched else (batch, time)
        if valid_mask.shape != expected:
            raise ValueError(
                f"valid_mask must have shape {expected}, got {tuple(valid_mask.shape)}"
            )
        mask = valid_mask.unsqueeze(0) if unbatched else valid_mask
        mask = mask.to(device=batched.device, dtype=torch.bool)

    detrended = linear_detrend_projected_position(
        batched,
        valid_mask=mask,
    )
    autocorrelation = vector_autocorrelation_fft(detrended, mask)
    if autocorrelation.ndim == 1:
        autocorrelation = autocorrelation.unsqueeze(0)
    fallback_periods, fallback_confidences = estimate_period_from_projected_pose(
        batched,
        minimum=minimum,
        maximum=maximum,
        valid_mask=mask,
    )

    diagnostics: list[ProjectedPositionLagVelocityFallbackDiagnostic] = []
    for sample_ac, sample_mask, fallback_period_tensor, fallback_confidence_tensor in zip(
        autocorrelation,
        mask,
        fallback_periods,
        fallback_confidences,
        strict=True,
    ):
        valid_length = int(sample_mask.sum())
        upper_lag = min(maximum, valid_length // 2)
        searched_lags = tuple(range(minimum, upper_lag + 1))
        searched_values = tuple(
            float(value)
            for value in sample_ac[minimum : upper_lag + 1].detach().cpu()
        )
        peaks, properties = find_peaks(
            sample_ac.detach().cpu().numpy(),
            distance=minimum,
            prominence=0.0,
        )
        candidates = tuple(
            (
                int(lag),
                float(sample_ac[int(lag)]),
                float(prominence),
            )
            for lag, prominence in zip(
                peaks,
                properties["prominences"],
                strict=True,
            )
            if minimum <= int(lag) <= upper_lag and float(sample_ac[int(lag)]) > 0.0
        )
        positive_peak_lags = tuple(item[0] for item in candidates)
        positive_peak_heights = tuple(item[1] for item in candidates)
        positive_peak_prominences = tuple(item[2] for item in candidates)

        fallback_period = float(fallback_period_tensor)
        fallback_confidence = float(fallback_confidence_tensor)
        selected_lag: int | None = None
        selected_period = float(minimum)
        confidence = 0.0
        selection_source = "no-evidence"
        near_best_height_threshold = 0.0
        eligible_candidates: tuple[tuple[int, float, float], ...] = ()
        if candidates:
            maximum_height = max(item[1] for item in candidates)
            near_best_height_threshold = 0.90 * maximum_height
            eligible_candidates = tuple(
                item
                for item in candidates
                if item[1] >= near_best_height_threshold
            )
            selected_lag = eligible_candidates[0][0]
            selected_period = float(selected_lag)
            confidence = min(max(eligible_candidates[0][1], 0.0), 1.0)
            selection_source = "detrended-position-lag-acf"
        elif fallback_confidence > 0.0:
            selected_period = fallback_period
            confidence = fallback_confidence
            selection_source = "projected-velocity-spectrum-fallback"

        diagnostics.append(
            ProjectedPositionLagVelocityFallbackDiagnostic(
                valid_length=valid_length,
                searched_lags=searched_lags,
                autocorrelation_values=searched_values,
                positive_peak_lags=positive_peak_lags,
                positive_peak_heights=positive_peak_heights,
                positive_peak_prominences=positive_peak_prominences,
                near_best_height_threshold=near_best_height_threshold,
                eligible_peak_lags=tuple(item[0] for item in eligible_candidates),
                eligible_peak_heights=tuple(item[1] for item in eligible_candidates),
                selected_lag=selected_lag,
                selected_period=selected_period,
                confidence=confidence,
                selection_source=selection_source,
                fallback_period=fallback_period,
                fallback_confidence=fallback_confidence,
            )
        )
    return tuple(diagnostics)


def estimate_period_from_projected_position_lag_velocity_fallback(
    projected_pose: Tensor,
    minimum: int = 4,
    maximum: int = 128,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Estimate periods from the target-free lag-peak hybrid readout."""

    batched, unbatched = _as_batch_vectors(projected_pose)
    if batched.dtype not in (torch.float32, torch.float64):
        batched = batched.float()
    diagnostics = projected_position_lag_velocity_fallback_diagnostics(
        projected_pose,
        minimum=minimum,
        maximum=maximum,
        valid_mask=valid_mask,
    )
    periods = torch.tensor(
        [diagnostic.selected_period for diagnostic in diagnostics],
        dtype=batched.dtype,
        device=batched.device,
    )
    confidences = torch.tensor(
        [diagnostic.confidence for diagnostic in diagnostics],
        dtype=batched.dtype,
        device=batched.device,
    )
    if unbatched:
        return periods[:1], confidences[:1]
    return periods, confidences
