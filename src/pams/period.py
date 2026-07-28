"""Period estimation from pose or embedding sequences.

PAMS states that the period is obtained through an FFT over an autocorrelation
signal, but does not publish implementation details.  The routines here are a
deterministic, mask-aware implementation of that description.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PeriodEstimate:
    """One sequence's bounded period estimate and diagnostic information."""

    period: float
    confidence: float
    frequency: float
    valid_length: int


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
