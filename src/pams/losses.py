"""Self-supervised objectives for the independent PAMS reproduction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _check_embeddings(embeddings: Tensor) -> tuple[int, int, int]:
    if embeddings.ndim != 3:
        raise ValueError("embeddings must have shape [batch, time, dimension]")
    return embeddings.shape[0], embeddings.shape[1], embeddings.shape[2]


def _valid_mask(embeddings: Tensor, valid_mask: Tensor | None) -> Tensor:
    batch, time, _ = _check_embeddings(embeddings)
    if valid_mask is None:
        return torch.ones((batch, time), dtype=torch.bool, device=embeddings.device)
    if valid_mask.shape != (batch, time):
        raise ValueError(
            f"valid_mask must have shape {(batch, time)}, got {tuple(valid_mask.shape)}"
        )
    return valid_mask.to(device=embeddings.device, dtype=torch.bool)


def periodic_correspondence_indices(
    embeddings: Tensor,
    periods: Tensor,
    *,
    scale: float = 1.0,
    correspondence_tolerance: float = 0.1,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Select the most similar past/future cyclic correspondence.

    The disclosed method varies a period-adaptive window but does not publish
    exact index construction.  Here the expected displacement remains one
    period while ``scale * period`` determines the search-window width.  The
    returned tensor is ``[batch, time, 2]`` for past and future indices, with
    ``-1`` denoting no valid candidate.
    """

    batch, time, _ = _check_embeddings(embeddings)
    if periods.shape not in {(batch,), (batch, 1)}:
        raise ValueError(f"periods must have shape [{batch}]")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if not 0.0 <= correspondence_tolerance <= 0.5:
        raise ValueError("correspondence_tolerance must be in [0, 0.5]")
    valid = _valid_mask(embeddings, valid_mask)
    normalized = F.normalize(embeddings.detach(), dim=-1, eps=1e-12)
    similarities = torch.einsum("btd,bsd->bts", normalized, normalized)
    return _correspondences_from_similarities(
        similarities,
        periods.reshape(batch),
        scale=scale,
        correspondence_tolerance=correspondence_tolerance,
        valid=valid,
    )


def _correspondences_from_similarities(
    similarities: Tensor,
    periods: Tensor,
    *,
    scale: float,
    correspondence_tolerance: float,
    valid: Tensor,
) -> Tensor:
    batch, time, candidate_time = similarities.shape
    if candidate_time != time:
        raise ValueError("similarities must be square over time")
    indices = torch.arange(time, device=similarities.device)
    offsets = indices.view(1, 1, time) - indices.view(1, time, 1)
    rounded_periods = periods.round().clamp_min(1).to(dtype=torch.long)
    windows = (rounded_periods.float() * scale).round().clamp_min(1)
    radii = (windows * correspondence_tolerance).round().to(dtype=torch.long)
    result: list[Tensor] = []
    finite_floor = torch.finfo(similarities.dtype).min

    for direction in (-1, 1):
        centers = (direction * rounded_periods).view(batch, 1, 1)
        allowed = (offsets - centers).abs() <= radii.view(batch, 1, 1)
        allowed = allowed & valid.unsqueeze(1) & valid.unsqueeze(2)
        scores = similarities.masked_fill(~allowed, finite_floor)
        best = scores.argmax(dim=-1)
        exists = allowed.any(dim=-1)
        result.append(best.masked_fill(~exists, -1))
    return torch.stack(result, dim=-1)


@dataclass(frozen=True)
class TCCLossOutput:
    """Detailed PAMS-TCC loss values for logging and tests."""

    total: Tensor
    scale_losses: tuple[Tensor, ...]
    valid_anchor_counts: tuple[int, ...]


class PAMSTCCLoss(nn.Module):
    """Multi-scale, multi-positive InfoNCE with period-adaptive positives."""

    def __init__(
        self,
        scales: tuple[float, ...] = (0.5, 1.0, 1.5),
        temperature: float = 0.1,
        correspondence_tolerance: float = 0.1,
    ) -> None:
        super().__init__()
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError("scales must contain positive values")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= correspondence_tolerance <= 0.5:
            raise ValueError("correspondence_tolerance must be in [0, 0.5]")
        self.scales = tuple(float(scale) for scale in scales)
        self.temperature = float(temperature)
        self.correspondence_tolerance = float(correspondence_tolerance)

    def compute(
        self,
        embeddings: Tensor,
        periods: Tensor,
        valid_mask: Tensor | None = None,
        *,
        cluster_labels: Tensor | None = None,
        use_cross_cluster_negatives: bool = True,
    ) -> TCCLossOutput:
        batch, time, dimension = _check_embeddings(embeddings)
        valid = _valid_mask(embeddings, valid_mask)
        if periods.shape not in {(batch,), (batch, 1)}:
            raise ValueError(f"periods must have shape [{batch}]")
        periods = periods.reshape(batch)
        if cluster_labels is not None:
            if cluster_labels.shape != (batch,):
                raise ValueError(f"cluster_labels must have shape [{batch}]")
            cluster_labels = cluster_labels.to(device=embeddings.device)

        normalized = F.normalize(embeddings, dim=-1, eps=1e-12)
        within_logits = torch.einsum("btd,bsd->bts", normalized, normalized) / self.temperature
        valid_counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
        prototypes = (normalized * valid.unsqueeze(-1)).sum(dim=1) / valid_counts.to(
            normalized.dtype
        )
        prototypes = F.normalize(prototypes, dim=-1, eps=1e-12)
        prototype_valid = valid.any(dim=1)
        cross_video_logits = torch.einsum("btd,cd->btc", normalized, prototypes) / self.temperature
        video_indices = torch.arange(batch, device=embeddings.device)
        other_video_mask = video_indices.view(1, 1, batch) != video_indices.view(batch, 1, 1)
        other_video_mask = (
            other_video_mask & valid.unsqueeze(-1) & prototype_valid.view(1, 1, batch)
        )
        negative_infinity = torch.finfo(embeddings.dtype).min
        scale_losses: list[Tensor] = []
        anchor_counts: list[int] = []

        for scale in self.scales:
            correspondences = _correspondences_from_similarities(
                within_logits.detach(),
                periods,
                scale=scale,
                correspondence_tolerance=self.correspondence_tolerance,
                valid=valid,
            )
            positive_mask = torch.zeros(
                (batch, time, time),
                dtype=torch.bool,
                device=embeddings.device,
            )
            if time > 1:
                adjacent = torch.arange(time - 1, device=embeddings.device)
                positive_mask[:, adjacent, adjacent + 1] = True
                positive_mask[:, adjacent + 1, adjacent] = True
            for direction_slot in range(2):
                selected = correspondences[:, :, direction_slot]
                has_selected = selected >= 0
                batch_grid, time_grid = torch.nonzero(has_selected, as_tuple=True)
                positive_mask[
                    batch_grid,
                    time_grid,
                    selected[batch_grid, time_grid],
                ] = True
            positive_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
            identity = torch.eye(time, dtype=torch.bool, device=embeddings.device).unsqueeze(0)
            within_candidate_mask = valid.unsqueeze(1) & valid.unsqueeze(2) & ~identity
            positive_mask &= within_candidate_mask
            positive_counts = positive_mask.sum(dim=-1)
            valid_anchors = valid & (positive_counts > 0)

            positive_logits = within_logits.masked_fill(~positive_mask, negative_infinity)
            denominator_parts = [
                within_logits.masked_fill(~within_candidate_mask, negative_infinity),
                cross_video_logits.masked_fill(~other_video_mask, negative_infinity),
            ]

            if use_cross_cluster_negatives and cluster_labels is not None and batch > 1:
                different_cluster = cluster_labels.view(batch, 1) != cluster_labels.view(1, batch)
                cross_cluster_mask = other_video_mask & different_cluster.view(batch, 1, batch)
                cross_cluster_logits = cross_video_logits.masked_fill(
                    ~cross_cluster_mask, negative_infinity
                )
                maximum_positive_count = int(positive_counts.max())
                pool_size = min(maximum_positive_count, batch)
                if pool_size:
                    hard_cross = torch.topk(cross_cluster_logits, k=pool_size, dim=-1).values
                    pool_indices = torch.arange(pool_size, device=embeddings.device).view(
                        1, 1, pool_size
                    )
                    hard_cross = hard_cross.masked_fill(
                        pool_indices >= positive_counts.unsqueeze(-1),
                        negative_infinity,
                    )
                    denominator_parts.append(hard_cross)

            numerator = torch.logsumexp(positive_logits, dim=-1)
            denominator = torch.logsumexp(torch.cat(denominator_parts, dim=-1), dim=-1)
            losses = denominator - numerator
            if valid_anchors.any():
                scale_loss = losses[valid_anchors].mean()
            else:
                scale_loss = embeddings.sum() * 0.0
            scale_losses.append(scale_loss)
            anchor_counts.append(int(valid_anchors.sum()))

        total = torch.stack(scale_losses).mean()
        return TCCLossOutput(
            total=total,
            scale_losses=tuple(scale_losses),
            valid_anchor_counts=tuple(anchor_counts),
        )

    def forward(
        self,
        embeddings: Tensor,
        periods: Tensor,
        valid_mask: Tensor | None = None,
        *,
        cluster_labels: Tensor | None = None,
        use_cross_cluster_negatives: bool = True,
    ) -> Tensor:
        return self.compute(
            embeddings,
            periods,
            valid_mask,
            cluster_labels=cluster_labels,
            use_cross_cluster_negatives=use_cross_cluster_negatives,
        ).total


@dataclass(frozen=True)
class SSHeadLossOutput:
    """Weighted SSHead total and its four unweighted components."""

    total: Tensor
    cycle: Tensor
    spectral: Tensor
    variance: Tensor
    smoothness: Tensor


class SSHeadLoss(nn.Module):
    """Independent self-supervised completion for the undisclosed head loss."""

    def __init__(
        self,
        cycle_weight: float = 1.0,
        spectral_weight: float = 1.0,
        variance_weight: float = 0.1,
        smoothness_weight: float = 0.01,
    ) -> None:
        super().__init__()
        weights = (
            cycle_weight,
            spectral_weight,
            variance_weight,
            smoothness_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("SSHead weights must be non-negative")
        self.cycle_weight = float(cycle_weight)
        self.spectral_weight = float(spectral_weight)
        self.variance_weight = float(variance_weight)
        self.smoothness_weight = float(smoothness_weight)

    def compute(
        self,
        period_stream: Tensor,
        periods: Tensor,
        valid_mask: Tensor | None = None,
    ) -> SSHeadLossOutput:
        if period_stream.ndim == 1:
            stream = period_stream.unsqueeze(0)
            unbatched = True
        elif period_stream.ndim == 2:
            stream = period_stream
            unbatched = False
        else:
            raise ValueError("period_stream must have shape [time] or [batch, time]")
        batch, time = stream.shape
        if periods.ndim == 0:
            periods = periods.expand(batch)
        if periods.shape not in {(batch,), (batch, 1)}:
            raise ValueError(f"periods must have shape [{batch}]")
        periods = periods.reshape(batch).to(device=stream.device)
        if valid_mask is None:
            valid = torch.ones((batch, time), dtype=torch.bool, device=stream.device)
        else:
            expected = (time,) if unbatched else (batch, time)
            if valid_mask.shape != expected:
                raise ValueError(
                    f"valid_mask must have shape {expected}, got {tuple(valid_mask.shape)}"
                )
            valid = valid_mask.unsqueeze(0) if unbatched else valid_mask
            valid = valid.to(device=stream.device, dtype=torch.bool)

        cycle_losses: list[Tensor] = []
        spectral_losses: list[Tensor] = []
        variance_losses: list[Tensor] = []
        smoothness_losses: list[Tensor] = []
        zero = stream.sum() * 0.0

        for values, sample_period, sample_valid in zip(stream, periods, valid, strict=True):
            period = max(1, int(round(float(sample_period.detach()))))
            if period < time:
                pair_valid = sample_valid[:-period] & sample_valid[period:]
                if pair_valid.any():
                    difference = values[:-period] - values[period:]
                    cycle_losses.append(difference[pair_valid].square().mean())

            count = sample_valid.sum().clamp_min(1)
            mean = (values * sample_valid).sum() / count
            centered = (values - mean) * sample_valid
            spectral_values = (
                centered.float() if centered.dtype in (torch.float16, torch.bfloat16) else centered
            )
            power = torch.fft.rfft(spectral_values).abs().square()
            if power.numel() > 1:
                power = power.clone()
                power[0] = 0.0
                target_bin = int(round(time / period))
                target_bin = min(max(target_bin, 1), power.numel() - 1)
                low = max(1, target_bin - 1)
                high = min(power.numel(), target_bin + 2)
                target_power = power[low:high].sum()
                spectral_losses.append(1.0 - target_power / power.sum().clamp_min(1e-12))

            selected = values[sample_valid]
            if selected.numel() >= 2:
                standard_deviation = selected.std(unbiased=False)
            else:
                standard_deviation = selected.new_tensor(0.0)
            variance_losses.append(F.relu(1.0 - standard_deviation))

            if time >= 3:
                triplet_valid = sample_valid[:-2] & sample_valid[1:-1] & sample_valid[2:]
                if triplet_valid.any():
                    second_difference = values[2:] - 2.0 * values[1:-1] + values[:-2]
                    smoothness_losses.append(second_difference[triplet_valid].square().mean())

        def mean_or_zero(items: list[Tensor]) -> Tensor:
            return torch.stack(items).mean() if items else zero

        cycle = mean_or_zero(cycle_losses)
        spectral = mean_or_zero(spectral_losses)
        variance = mean_or_zero(variance_losses)
        smoothness = mean_or_zero(smoothness_losses)
        total = (
            self.cycle_weight * cycle
            + self.spectral_weight * spectral
            + self.variance_weight * variance
            + self.smoothness_weight * smoothness
        )
        return SSHeadLossOutput(
            total=total,
            cycle=cycle,
            spectral=spectral,
            variance=variance,
            smoothness=smoothness,
        )

    def forward(
        self,
        period_stream: Tensor,
        periods: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        return self.compute(period_stream, periods, valid_mask).total
