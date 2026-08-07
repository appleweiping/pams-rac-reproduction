"""Deterministic target-frequency action curves from frozen embeddings.

The PAMS paper discloses a two-layer scalar Period Head but not a training
target for that head.  This module is an independently inferred, target-free
readout.  It first obtains the global period from the existing full-vector
embedding-velocity ACF estimator.  For that frequency it then finds, per
video, the embedding-space direction with maximum sine/cosine coefficient
energy and projects the complete frozen embedding trajectory onto that
direction.

The Fourier basis uses original dense frame indices.  Invalid frames are
excluded from every statistic and are exact zero in the returned curve;
internal holes are never compacted or bridged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from pams.period import estimate_period_from_embedding_velocity_vectors

if TYPE_CHECKING:
    from pams.config import PAMSConfig


@dataclass(frozen=True, slots=True)
class EmbeddingActionCurveBatch:
    """Auditable batched output of the deterministic embedding readout."""

    curves: Tensor
    periods: Tensor
    period_confidences: Tensor
    curve_standard_deviations: Tensor
    raw_curve_standard_deviations: Tensor
    harmonic_energy_fractions: Tensor
    available: Tensor

    def __post_init__(self) -> None:
        if self.curves.ndim != 2:
            raise ValueError("curves must have shape [batch, time]")
        batch = self.curves.shape[0]
        for name in (
            "periods",
            "period_confidences",
            "curve_standard_deviations",
            "raw_curve_standard_deviations",
            "harmonic_energy_fractions",
            "available",
        ):
            value = getattr(self, name)
            if value.shape != (batch,):
                raise ValueError(f"{name} must have shape [batch]")
        if self.available.dtype != torch.bool:
            raise TypeError("available must be boolean")


def validate_embedding_action_curve_compatibility(
    upstream_config: PAMSConfig,
    candidate_config: PAMSConfig,
) -> None:
    """Require exact v16 semantics except for the inference readout marker."""

    if upstream_config.readout.action_curve_source != "learned_period_head":
        raise ValueError("upstream encoder config must use the historical readout")
    if (
        candidate_config.readout.action_curve_source
        != "embedding_frequency_projection"
    ):
        raise ValueError(
            "candidate config must opt into embedding_frequency_projection"
        )
    upstream = upstream_config.model_dump(mode="json")
    candidate = candidate_config.model_dump(mode="json")
    upstream_source = upstream.pop("readout")["action_curve_source"]
    candidate_source = candidate.pop("readout")["action_curve_source"]
    if (
        upstream_source != "learned_period_head"
        or candidate_source != "embedding_frequency_projection"
    ):
        raise AssertionError("validated action-curve sources changed unexpectedly")
    if upstream != candidate:
        raise ValueError(
            "upstream and candidate configs may differ only in "
            "readout.action_curve_source"
        )


def _validated_inputs(
    embeddings: Tensor,
    periods: Tensor,
    valid_mask: Tensor | None,
    timeline_lengths: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor, bool]:
    unbatched = embeddings.ndim == 2
    values = embeddings.unsqueeze(0) if unbatched else embeddings
    if values.ndim != 3 or values.shape[-1] < 1:
        raise ValueError(
            "embeddings must have shape [time, dimension] or "
            "[batch, time, dimension]"
        )
    batch, time, _ = values.shape
    if time < 1:
        raise ValueError("embeddings must contain at least one frame")
    period_values = periods.reshape(1) if periods.ndim == 0 else periods
    if period_values.shape != (batch,):
        raise ValueError(f"periods must have shape {(batch,)}")
    period_values = period_values.to(device=values.device, dtype=torch.float32)
    if not bool(torch.isfinite(period_values).all()) or bool((period_values <= 0).any()):
        raise ValueError("periods must be finite and positive")

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

    if timeline_lengths is None:
        lengths = torch.full(
            (batch,),
            time,
            dtype=torch.long,
            device=values.device,
        )
    else:
        lengths = timeline_lengths.reshape(1) if timeline_lengths.ndim == 0 else timeline_lengths
        if lengths.shape != (batch,):
            raise ValueError(f"timeline_lengths must have shape {(batch,)}")
        if lengths.dtype == torch.bool or lengths.is_floating_point():
            raise TypeError("timeline_lengths must use an integer dtype")
        lengths = lengths.to(device=values.device, dtype=torch.long)
        if bool((lengths < 1).any()) or bool((lengths > time).any()):
            raise ValueError("timeline_lengths must lie in [1, padded_time]")
    positions = torch.arange(time, device=values.device).unsqueeze(0)
    inside_timeline = positions < lengths.unsqueeze(1)
    if bool((valid & ~inside_timeline).any()):
        raise ValueError("valid_mask cannot mark padded frames as valid")
    return values, period_values, valid, lengths, unbatched


def _dominant_harmonic_direction(coefficients: Tensor) -> tuple[Tensor, bool]:
    """Return the deterministic top right-singular direction of ``[2, D]``."""

    gram = coefficients @ coefficients.transpose(0, 1)
    g00 = gram[0, 0]
    g01 = gram[0, 1]
    g11 = gram[1, 1]
    total = g00 + g11
    if not bool(torch.isfinite(total)) or float(total) <= 1e-12:
        return torch.zeros(
            coefficients.shape[1],
            dtype=coefficients.dtype,
            device=coefficients.device,
        ), False

    # Closed-form 2x2 eigensystem avoids an opaque SVD sign convention.  The
    # diagonal/tied branch has an explicit cosine-first tie break.
    if float(torch.abs(g01)) <= 1e-12:
        harmonic_direction = coefficients.new_tensor(
            [1.0, 0.0] if float(g00) >= float(g11) else [0.0, 1.0]
        )
    else:
        discriminant = torch.sqrt((g00 - g11).square() + 4.0 * g01.square())
        eigenvalue = 0.5 * (g00 + g11 + discriminant)
        first = torch.stack((g01, eigenvalue - g00))
        second = torch.stack((eigenvalue - g11, g01))
        harmonic_direction = (
            first
            if float(first.square().sum()) >= float(second.square().sum())
            else second
        )
        harmonic_direction = harmonic_direction / harmonic_direction.norm().clamp_min(1e-12)

    feature_direction = coefficients.transpose(0, 1) @ harmonic_direction
    feature_norm = feature_direction.norm()
    if not bool(torch.isfinite(feature_norm)) or float(feature_norm) <= 1e-12:
        return torch.zeros_like(feature_direction), False
    feature_direction = feature_direction / feature_norm
    anchor = int(torch.argmax(torch.abs(feature_direction)))
    if float(feature_direction[anchor]) < 0.0:
        feature_direction = -feature_direction
    return feature_direction, True


def build_frequency_matched_action_curves(
    embeddings: Tensor,
    periods: Tensor,
    valid_mask: Tensor | None = None,
    *,
    timeline_lengths: Tensor | None = None,
    period_confidences: Tensor | None = None,
) -> EmbeddingActionCurveBatch:
    """Project frozen embeddings onto a target-frequency PCA direction.

    The target sine/cosine pair is centered and Gram--Schmidt orthonormalized
    over valid dense-frame positions.  Its two per-feature Fourier coefficient
    vectors define a rank-two covariance; the leading feature direction is
    selected with a deterministic closed-form eigensystem.  The full centered
    embedding trajectory (not a synthesized sine wave) is projected onto that
    direction and valid-frame RMS-normalized before peak counting.
    """

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
    if not values.is_floating_point():
        values = values.float()
    work = values.detach().to(dtype=torch.float32)
    batch, padded_time, dimension = work.shape
    curves = torch.zeros((batch, padded_time), dtype=work.dtype, device=work.device)
    normalized_stds = torch.zeros((batch,), dtype=work.dtype, device=work.device)
    raw_stds = torch.zeros_like(normalized_stds)
    energy_fractions = torch.zeros_like(normalized_stds)
    available = torch.zeros((batch,), dtype=torch.bool, device=work.device)

    for sample_index in range(batch):
        length = int(lengths[sample_index])
        sample_valid = valid[sample_index, :length]
        valid_total = int(sample_valid.sum())
        if valid_total < 3:
            continue
        sample = work[sample_index, :length]
        if not bool(torch.isfinite(sample[sample_valid]).all()):
            raise ValueError("valid embeddings must contain only finite values")
        safe_sample = torch.where(
            sample_valid.unsqueeze(1),
            sample,
            torch.zeros_like(sample),
        )
        mean = safe_sample.sum(dim=0) / float(valid_total)
        centered = torch.where(
            sample_valid.unsqueeze(1),
            sample - mean,
            torch.zeros_like(sample),
        )

        dense_time = torch.arange(length, dtype=work.dtype, device=work.device)
        angular_frequency = (2.0 * math.pi) / period_values[sample_index]
        cosine = torch.cos(angular_frequency * dense_time)
        sine = torch.sin(angular_frequency * dense_time)
        mask_values = sample_valid.to(dtype=work.dtype)
        cosine = (cosine - cosine[sample_valid].mean()) * mask_values
        sine = (sine - sine[sample_valid].mean()) * mask_values
        cosine_norm = cosine.norm()
        if float(cosine_norm) <= 1e-12:
            continue
        cosine = cosine / cosine_norm
        sine = sine - torch.dot(sine, cosine) * cosine
        sine_norm = sine.norm()
        if float(sine_norm) <= 1e-12:
            continue
        sine = sine / sine_norm
        coefficients = torch.stack(
            (cosine @ centered, sine @ centered),
            dim=0,
        )
        direction, direction_available = _dominant_harmonic_direction(coefficients)
        if not direction_available or direction.shape != (dimension,):
            continue

        raw_curve = (centered @ direction).masked_fill(~sample_valid, 0.0)
        raw_mean = raw_curve[sample_valid].mean()
        raw_curve = (raw_curve - raw_mean) * mask_values
        raw_std = raw_curve[sample_valid].square().mean().sqrt()
        if not bool(torch.isfinite(raw_std)) or float(raw_std) <= 1e-12:
            continue
        curve = (raw_curve / raw_std).masked_fill(~sample_valid, 0.0)
        curve_std = curve[sample_valid].square().mean().sqrt()
        total_energy = centered.square().sum()
        harmonic_energy = coefficients.square().sum()

        curves[sample_index, :length] = curve
        raw_stds[sample_index] = raw_std
        normalized_stds[sample_index] = curve_std
        energy_fractions[sample_index] = (
            harmonic_energy / total_energy.clamp_min(1e-12)
        ).clamp(0.0, 1.0)
        available[sample_index] = True

    curves = curves.masked_fill(~valid, 0.0)
    return EmbeddingActionCurveBatch(
        curves=curves,
        periods=period_values,
        period_confidences=confidence_values,
        curve_standard_deviations=normalized_stds,
        raw_curve_standard_deviations=raw_stds,
        harmonic_energy_fractions=energy_fractions,
        available=available,
    )


def estimate_embedding_action_curves(
    embeddings: Tensor,
    *,
    minimum_period: int,
    maximum_period: int,
    valid_mask: Tensor | None = None,
    timeline_lengths: Tensor | None = None,
) -> EmbeddingActionCurveBatch:
    """Estimate a full-vector ACF period and its matched embedding curve."""

    periods, confidences = estimate_period_from_embedding_velocity_vectors(
        embeddings.detach(),
        minimum=minimum_period,
        maximum=maximum_period,
        valid_mask=valid_mask,
    )
    result = build_frequency_matched_action_curves(
        embeddings.detach(),
        periods,
        valid_mask,
        timeline_lengths=timeline_lengths,
        period_confidences=confidences,
    )
    return result
