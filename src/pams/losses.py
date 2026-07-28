"""Self-supervised objectives for the independent PAMS reproduction."""

from __future__ import annotations

from collections.abc import Sequence
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
    valid = _valid_mask(embeddings, valid_mask)
    normalized = F.normalize(embeddings.detach(), dim=-1, eps=1e-12)
    similarities = torch.einsum("btd,bsd->bts", normalized, normalized)
    return _correspondences_from_similarities(
        similarities,
        periods.reshape(batch),
        scale=scale,
        valid=valid,
    )


def _correspondences_from_similarities(
    similarities: Tensor,
    periods: Tensor,
    *,
    scale: float,
    valid: Tensor,
) -> Tensor:
    batch, time, candidate_time = similarities.shape
    if candidate_time != time:
        raise ValueError("similarities must be square over time")
    indices = torch.arange(time, device=similarities.device)
    offsets = indices.view(1, 1, time) - indices.view(1, time, 1)
    rounded_periods = periods.round().clamp_min(1).to(dtype=torch.long)
    # The paper discloses W_k = round(scale * T) but not whether W_k is a
    # radius or a full span around t +/- T. We infer a symmetric full-span
    # interpretation, with a one-frame minimum radius. Unlike the previous
    # extra 0.1 tolerance, this does not collapse all scales at short periods.
    radii = (rounded_periods.float() * scale / 2.0).round().clamp_min(1).to(dtype=torch.long)
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
    cross_cluster_requested_counts: tuple[int, ...]
    cross_cluster_actual_counts: tuple[int, ...]
    cross_cluster_shortfall_counts: tuple[int, ...]
    cross_cluster_requested_per_anchor: tuple[Tensor, ...]
    cross_cluster_actual_per_anchor: tuple[Tensor, ...]
    cross_cluster_shortfall_per_anchor: tuple[Tensor, ...]


class PAMSTCCLoss(nn.Module):
    """Multi-scale, multi-positive InfoNCE with period-adaptive positives."""

    def __init__(
        self,
        scales: tuple[float, ...] = (0.5, 1.0, 1.5),
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError("scales must contain positive values")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.scales = tuple(float(scale) for scale in scales)
        self.temperature = float(temperature)

    def compute(
        self,
        embeddings: Tensor,
        periods: Tensor,
        valid_mask: Tensor | None = None,
        *,
        period_confidence: Tensor | None = None,
        cluster_labels: Tensor | None = None,
        video_ids: Sequence[str] | None = None,
        bank_features: Tensor | None = None,
        bank_cluster_labels: Tensor | None = None,
        bank_video_ids: Sequence[str] | None = None,
        use_cross_cluster_negatives: bool = True,
    ) -> TCCLossOutput:
        batch, time, dimension = _check_embeddings(embeddings)
        valid = _valid_mask(embeddings, valid_mask)
        if periods.shape not in {(batch,), (batch, 1)}:
            raise ValueError(f"periods must have shape [{batch}]")
        periods = periods.reshape(batch)
        if period_confidence is None:
            period_evidence = torch.ones(batch, dtype=torch.bool, device=embeddings.device)
        else:
            if period_confidence.shape not in {(batch,), (batch, 1)}:
                raise ValueError(f"period_confidence must have shape [{batch}]")
            confidence = period_confidence.reshape(batch).to(device=embeddings.device)
            if (
                not torch.isfinite(confidence).all()
                or (confidence < 0).any()
                or (confidence > 1).any()
            ):
                raise ValueError("period_confidence must be finite and in [0, 1]")
            period_evidence = confidence != 0
        if cluster_labels is not None:
            if cluster_labels.shape != (batch,):
                raise ValueError(f"cluster_labels must have shape [{batch}]")
            cluster_labels = cluster_labels.to(device=embeddings.device)

        bank_arguments = (bank_features, bank_cluster_labels, bank_video_ids)
        supplied_bank_arguments = sum(argument is not None for argument in bank_arguments)
        if supplied_bank_arguments not in {0, len(bank_arguments)}:
            raise ValueError(
                "bank_features, bank_cluster_labels, and bank_video_ids must be supplied together"
            )
        bank_logits: Tensor | None = None
        bank_candidate_mask: Tensor | None = None
        if supplied_bank_arguments:
            assert bank_features is not None
            assert bank_cluster_labels is not None
            assert bank_video_ids is not None
            if cluster_labels is None:
                raise ValueError("cluster_labels are required with a prototype bank")
            if video_ids is None or len(video_ids) != batch:
                raise ValueError(f"video_ids must contain {batch} current-batch identifiers")
            current_ids = tuple(str(identifier) for identifier in video_ids)
            stored_ids = tuple(str(identifier) for identifier in bank_video_ids)
            if len(set(current_ids)) != batch:
                raise ValueError("video_ids must be unique")
            if len(set(stored_ids)) != len(stored_ids):
                raise ValueError("bank_video_ids must be unique")
            if bank_features.ndim != 2 or bank_features.shape[1] != dimension:
                raise ValueError(f"bank_features must have shape [bank, {dimension}]")
            bank_size = bank_features.shape[0]
            if bank_size != len(stored_ids):
                raise ValueError("bank_features and bank_video_ids must have equal length")
            if bank_cluster_labels.shape != (bank_size,):
                raise ValueError(f"bank_cluster_labels must have shape [{bank_size}]")
            stored_clusters = bank_cluster_labels.to(
                device=embeddings.device,
                dtype=cluster_labels.dtype,
            )
            excluded_ids = set(current_ids)
            outside_current_batch = torch.tensor(
                [identifier not in excluded_ids for identifier in stored_ids],
                dtype=torch.bool,
                device=embeddings.device,
            )
            bank_feature_valid = (
                bank_features.detach().to(device=embeddings.device).abs().sum(dim=-1) > 1e-12
            )
            bank_candidate_mask = (
                cluster_labels.view(batch, 1) != stored_clusters.view(1, bank_size)
            ) & outside_current_batch.view(1, bank_size)
            bank_candidate_mask &= bank_feature_valid.view(1, bank_size)

        normalized = F.normalize(embeddings, dim=-1, eps=1e-12)
        within_logits = torch.einsum("btd,bsd->bts", normalized, normalized) / self.temperature
        valid_counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
        prototypes = (normalized * valid.unsqueeze(-1)).sum(dim=1) / valid_counts.to(
            normalized.dtype
        )
        prototypes = F.normalize(prototypes, dim=-1, eps=1e-12)
        prototype_valid = valid.any(dim=1)
        cross_video_logits = torch.einsum("btd,cd->btc", normalized, prototypes) / self.temperature
        if supplied_bank_arguments:
            assert bank_features is not None
            normalized_bank = F.normalize(
                bank_features.detach().to(
                    device=embeddings.device,
                    dtype=embeddings.dtype,
                ),
                dim=-1,
                eps=1e-12,
            )
            bank_logits = (
                torch.einsum("btd,nd->btn", normalized, normalized_bank) / self.temperature
            )
        video_indices = torch.arange(batch, device=embeddings.device)
        other_video_mask = video_indices.view(1, 1, batch) != video_indices.view(batch, 1, 1)
        other_video_mask = (
            other_video_mask & valid.unsqueeze(-1) & prototype_valid.view(1, 1, batch)
        )
        negative_infinity = float("-inf")
        scale_losses: list[Tensor] = []
        anchor_counts: list[int] = []
        requested_counts_by_scale: list[Tensor] = []
        actual_counts_by_scale: list[Tensor] = []
        shortfall_counts_by_scale: list[Tensor] = []

        for scale in self.scales:
            correspondences = _correspondences_from_similarities(
                within_logits.detach(),
                periods,
                scale=scale,
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
                has_selected = (selected >= 0) & period_evidence.unsqueeze(1)
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

            denominator_parts = [
                within_logits.masked_fill(~within_candidate_mask, negative_infinity),
                cross_video_logits.masked_fill(~other_video_mask, negative_infinity),
            ]

            requested_counts = torch.zeros_like(positive_counts)
            actual_counts = torch.zeros_like(positive_counts)
            if use_cross_cluster_negatives and cluster_labels is not None:
                requested_counts = torch.where(
                    valid_anchors,
                    positive_counts,
                    torch.zeros_like(positive_counts),
                )
                if bank_logits is not None and bank_candidate_mask is not None:
                    available_counts = bank_candidate_mask.sum(dim=-1, keepdim=True)
                    actual_counts = torch.minimum(requested_counts, available_counts)
                    maximum_actual_count = int(actual_counts.max())
                    if maximum_actual_count:
                        candidate_scores = bank_logits.masked_fill(
                            ~bank_candidate_mask.unsqueeze(1),
                            negative_infinity,
                        )
                        hard_cross = torch.topk(
                            candidate_scores,
                            k=maximum_actual_count,
                            dim=-1,
                        ).values
                        slots = torch.arange(
                            maximum_actual_count,
                            device=embeddings.device,
                        ).view(1, 1, maximum_actual_count)
                        selected_slots = slots < actual_counts.unsqueeze(-1)
                        denominator_parts.append(
                            hard_cross.masked_fill(~selected_slots, negative_infinity)
                        )
            shortfall_counts = requested_counts - actual_counts
            requested_counts_by_scale.append(requested_counts.detach())
            actual_counts_by_scale.append(actual_counts.detach())
            shortfall_counts_by_scale.append(shortfall_counts.detach())

            mean_positive_logit = within_logits.masked_fill(~positive_mask, 0.0).sum(
                dim=-1
            ) / positive_counts.clamp_min(1).to(within_logits.dtype)
            denominator = torch.logsumexp(torch.cat(denominator_parts, dim=-1), dim=-1)
            # Equation (1) averages one log-softmax term per positive.  A
            # logsumexp positive numerator would instead reward satisfying
            # only the easiest positive and is not the disclosed objective.
            losses = denominator - mean_positive_logit
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
            cross_cluster_requested_counts=tuple(
                int(counts.sum()) for counts in requested_counts_by_scale
            ),
            cross_cluster_actual_counts=tuple(
                int(counts.sum()) for counts in actual_counts_by_scale
            ),
            cross_cluster_shortfall_counts=tuple(
                int(counts.sum()) for counts in shortfall_counts_by_scale
            ),
            cross_cluster_requested_per_anchor=tuple(requested_counts_by_scale),
            cross_cluster_actual_per_anchor=tuple(actual_counts_by_scale),
            cross_cluster_shortfall_per_anchor=tuple(shortfall_counts_by_scale),
        )

    def forward(
        self,
        embeddings: Tensor,
        periods: Tensor,
        valid_mask: Tensor | None = None,
        *,
        period_confidence: Tensor | None = None,
        cluster_labels: Tensor | None = None,
        video_ids: Sequence[str] | None = None,
        bank_features: Tensor | None = None,
        bank_cluster_labels: Tensor | None = None,
        bank_video_ids: Sequence[str] | None = None,
        use_cross_cluster_negatives: bool = True,
    ) -> Tensor:
        return self.compute(
            embeddings,
            periods,
            valid_mask,
            period_confidence=period_confidence,
            cluster_labels=cluster_labels,
            video_ids=video_ids,
            bank_features=bank_features,
            bank_cluster_labels=bank_cluster_labels,
            bank_video_ids=bank_video_ids,
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
        *,
        period_confidence: Tensor | None = None,
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
        if period_confidence is None:
            confidences = torch.ones(batch, dtype=stream.dtype, device=stream.device)
        else:
            if period_confidence.ndim == 0:
                period_confidence = period_confidence.expand(batch)
            if period_confidence.shape not in {(batch,), (batch, 1)}:
                raise ValueError(f"period_confidence must have shape [{batch}]")
            confidences = period_confidence.reshape(batch).to(device=stream.device)
            if (
                not torch.isfinite(confidences).all()
                or (confidences < 0).any()
                or (confidences > 1).any()
            ):
                raise ValueError("period_confidence must be finite and in [0, 1]")
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

        for values, sample_period, sample_valid, sample_confidence in zip(
            stream,
            periods,
            valid,
            confidences,
            strict=True,
        ):
            if bool(sample_confidence != 0):
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
                    centered.float()
                    if centered.dtype in (torch.float16, torch.bfloat16)
                    else centered
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
        *,
        period_confidence: Tensor | None = None,
    ) -> Tensor:
        return self.compute(
            period_stream,
            periods,
            valid_mask,
            period_confidence=period_confidence,
        ).total
