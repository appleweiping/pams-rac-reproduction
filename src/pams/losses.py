"""Self-supervised objectives for the independent PAMS reproduction."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def masked_zscore(
    values: Tensor,
    valid_mask: Tensor,
    *,
    epsilon: float = 1e-4,
) -> Tensor:
    """Standardize each sample over valid time rows only.

    ``values`` may be scalar ``[batch, time]`` streams or vector-valued
    ``[batch, time, feature]`` streams.  Invalid rows are never read into the
    moments and are returned as exact zero.  Adding ``epsilon`` to the
    variance keeps the transformation continuous at a constant stream, so a
    downstream non-constant regression target still supplies a gradient.
    """

    if values.ndim not in {2, 3}:
        raise ValueError("values must have shape [batch, time] or [batch, time, feature]")
    if valid_mask.shape != values.shape[:2]:
        raise ValueError("valid_mask must match values batch and time dimensions")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    valid = valid_mask.to(device=values.device, dtype=torch.bool)
    expanded_valid = valid
    if values.ndim == 3:
        expanded_valid = expanded_valid.unsqueeze(-1)
    safe_values = torch.where(expanded_valid, values, torch.zeros_like(values))
    counts = valid.sum(dim=1).clamp_min(1).to(dtype=values.dtype)
    if values.ndim == 3:
        counts = counts.unsqueeze(-1)
    means = safe_values.sum(dim=1) / counts
    centered = torch.where(
        expanded_valid,
        values - means.unsqueeze(1),
        torch.zeros_like(values),
    )
    variances = centered.square().sum(dim=1) / counts
    denominator = (variances + epsilon).sqrt().unsqueeze(1)
    standardized = centered / denominator
    return standardized.masked_fill(~expanded_valid, 0.0)


@dataclass(frozen=True)
class ReferenceRelativeSignal:
    """Target-free per-video reference representation for the inferred head."""

    head_inputs: Tensor
    teacher: Tensor
    prototypes: Tensor
    available: Tensor


def _reference_anchor_index(
    projected_pose: Tensor,
    valid_mask: Tensor,
    period: int,
) -> int:
    """Choose a dynamic, cycle-stable phase anchor without labels.

    The score is mean half-period squared distance minus mean full-period
    squared distance, which is equivalent to full-period similarity minus
    half-period similarity under negative squared-distance similarity.  Thus
    a recurring action extremum outranks a stationary segment.  Ties are
    deterministic and favor the earliest valid frame.
    """

    if projected_pose.ndim != 2:
        raise ValueError("projected_pose must have shape [time, feature]")
    time, feature = projected_pose.shape
    if feature < 1 or valid_mask.shape != (time,):
        raise ValueError("valid_mask must match projected_pose time")
    if period < 1:
        raise ValueError("period must be positive")
    valid = valid_mask.to(device=projected_pose.device, dtype=torch.bool)
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if not valid_indices.numel():
        raise ValueError("reference anchor requires at least one valid frame")

    def neighbour_distance(offset: int) -> tuple[Tensor, Tensor]:
        sums = projected_pose.new_zeros(time)
        counts = projected_pose.new_zeros(time)
        if offset >= time:
            return sums, counts
        pair_valid = valid[:-offset] & valid[offset:]
        distances = (
            projected_pose[:-offset] - projected_pose[offset:]
        ).square().mean(dim=-1)
        weighted = torch.where(pair_valid, distances, torch.zeros_like(distances))
        pair_counts = pair_valid.to(dtype=projected_pose.dtype)
        sums[:-offset] += weighted
        sums[offset:] += weighted
        counts[:-offset] += pair_counts
        counts[offset:] += pair_counts
        return sums, counts

    full_sum, full_count = neighbour_distance(period)
    half_sum, half_count = neighbour_distance(max(1, int(round(period / 2.0))))
    eligible = valid & (full_count > 0) & (half_count > 0)
    if not bool(eligible.any()):
        return int(valid_indices[0])
    score = half_sum / half_count.clamp_min(1.0) - full_sum / full_count.clamp_min(1.0)
    score = score.masked_fill(~eligible, float("-inf"))
    return int(score.argmax())


def build_reference_relative_signal(
    projected_pose: Tensor,
    periods: Tensor,
    valid_mask: Tensor,
    *,
    period_confidence: Tensor,
    epsilon: float = 1e-4,
) -> ReferenceRelativeSignal:
    """Build a phase-anchored squared-distance input and scalar teacher.

    This is an independently inferred closure for the paper's undisclosed
    Period Head objective.  For each video, full-period similarity minus
    half-period similarity chooses a deterministic, dynamic phase anchor.
    Valid projected-pose rows one rounded target-free period apart are
    averaged into a reference prototype.  Per-coordinate squared distances
    to that prototype form the head input; the negative feature mean forms
    the scalar regression teacher, so each reference-pose recurrence is a
    peak for the downstream peak counter.  Both are mask-aware z-scores,
    making the representation invariant to a global affine scale/offset of
    the frozen projected pose apart from the explicit near-zero-variance
    stabilizer.

    At least two same-phase observations and non-zero period confidence are
    required.  A missing/constant reference signal is represented by exact
    zeros with ``available=False`` rather than fabricated periodic evidence.
    """

    if projected_pose.ndim != 3:
        raise ValueError("projected_pose must have shape [batch, time, feature]")
    batch, time, feature = projected_pose.shape
    if feature < 1:
        raise ValueError("projected_pose feature dimension must be positive")
    if valid_mask.shape != (batch, time):
        raise ValueError("valid_mask must have shape [batch, time]")
    if periods.shape not in {(batch,), (batch, 1)}:
        raise ValueError(f"periods must have shape [{batch}]")
    if period_confidence.shape not in {(batch,), (batch, 1)}:
        raise ValueError(f"period_confidence must have shape [{batch}]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    valid = valid_mask.to(device=projected_pose.device, dtype=torch.bool)
    sample_periods = periods.reshape(batch).to(device=projected_pose.device)
    confidences = period_confidence.reshape(batch).to(device=projected_pose.device)
    if not bool(torch.isfinite(sample_periods).all()) or bool((sample_periods <= 0).any()):
        raise ValueError("periods must be finite and positive")
    if (
        not bool(torch.isfinite(confidences).all())
        or bool((confidences < 0).any())
        or bool((confidences > 1).any())
    ):
        raise ValueError("period_confidence must be finite and in [0, 1]")

    raw_inputs = projected_pose.new_zeros(projected_pose.shape)
    raw_teacher = projected_pose.new_zeros((batch, time))
    prototypes = projected_pose.new_zeros((batch, feature))
    available = torch.zeros(batch, dtype=torch.bool, device=projected_pose.device)
    frame_indices = torch.arange(time, device=projected_pose.device)

    for sample_index in range(batch):
        sample_valid = valid[sample_index]
        if not bool(confidences[sample_index] > 0) or int(sample_valid.sum()) < 2:
            continue
        period = max(1, int(round(float(sample_periods[sample_index].detach()))))
        anchor = _reference_anchor_index(
            projected_pose[sample_index],
            sample_valid,
            period,
        )
        same_phase = sample_valid & (
            torch.remainder(frame_indices - anchor, period) == 0
        )
        if int(same_phase.sum()) < 2:
            continue
        prototype = projected_pose[sample_index, same_phase].mean(dim=0)
        squared_distance = (projected_pose[sample_index] - prototype).square()
        squared_distance = squared_distance.masked_fill(
            ~sample_valid.unsqueeze(-1),
            0.0,
        )
        raw_inputs[sample_index] = squared_distance
        raw_teacher[sample_index] = -squared_distance.mean(dim=-1)
        prototypes[sample_index] = prototype
        available[sample_index] = True

    normalization_mask = valid & available.unsqueeze(1)
    head_inputs = masked_zscore(
        raw_inputs,
        normalization_mask,
        epsilon=epsilon,
    )
    teacher = masked_zscore(
        raw_teacher,
        normalization_mask,
        epsilon=epsilon,
    )
    # A geometrically constant distance curve carries no phase target.  Keep
    # this decision detached because the projected encoder is frozen here.
    teacher_energy = teacher.detach().abs().sum(dim=1)
    available = available & (teacher_energy > epsilon)
    availability_mask = available.unsqueeze(1)
    head_inputs = head_inputs.masked_fill(
        ~availability_mask.unsqueeze(-1),
        0.0,
    )
    teacher = teacher.masked_fill(~availability_mask, 0.0)
    prototypes = prototypes.masked_fill(~available.unsqueeze(-1), 0.0)
    return ReferenceRelativeSignal(
        head_inputs=head_inputs,
        teacher=teacher,
        prototypes=prototypes,
        available=available,
    )


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
        exclude_other_scale_positives_from_denominator: bool = False,
    ) -> None:
        super().__init__()
        if not scales or any(scale <= 0 for scale in scales):
            raise ValueError("scales must contain positive values")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.scales = tuple(float(scale) for scale in scales)
        self.temperature = float(temperature)
        self.exclude_other_scale_positives_from_denominator = (
            exclude_other_scale_positives_from_denominator
        )

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
        # The paper names frames from other videos as negatives.  Keep every
        # valid frame in the physical batch instead of collapsing each video
        # into one pooled prototype: pooling changes both the candidate count
        # and the contrastive geometry.
        flattened_frames = normalized.reshape(batch * time, dimension)
        cross_video_logits = (
            torch.einsum("btd,nd->btn", normalized, flattened_frames)
            / self.temperature
        )
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
        candidate_video_indices = video_indices.repeat_interleave(time)
        other_video_mask = (
            (
                candidate_video_indices.view(1, 1, batch * time)
                != video_indices.view(batch, 1, 1)
            )
            & valid.unsqueeze(-1)
            & valid.reshape(1, 1, batch * time)
        )
        negative_infinity = float("-inf")
        scale_losses: list[Tensor] = []
        anchor_counts: list[int] = []
        requested_counts_by_scale: list[Tensor] = []
        actual_counts_by_scale: list[Tensor] = []
        shortfall_counts_by_scale: list[Tensor] = []

        identity = torch.eye(time, dtype=torch.bool, device=embeddings.device).unsqueeze(0)
        within_candidate_mask = valid.unsqueeze(1) & valid.unsqueeze(2) & ~identity

        def positive_mask_for_scale(scale: float) -> Tensor:
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
            positive_mask &= within_candidate_mask
            return positive_mask

        positive_masks: tuple[Tensor, ...] | None = None
        all_scale_positive_mask: Tensor | None = None
        if self.exclude_other_scale_positives_from_denominator:
            positive_masks = tuple(positive_mask_for_scale(scale) for scale in self.scales)
            all_scale_positive_mask = torch.stack(positive_masks, dim=0).any(dim=0)

        for scale_index, scale in enumerate(self.scales):
            if positive_masks is None:
                positive_mask = positive_mask_for_scale(scale)
            else:
                positive_mask = positive_masks[scale_index]
            positive_counts = positive_mask.sum(dim=-1)
            valid_anchors = valid & (positive_counts > 0)

            scale_within_candidate_mask = within_candidate_mask
            if all_scale_positive_mask is not None:
                other_scale_only_positives = all_scale_positive_mask & ~positive_mask
                scale_within_candidate_mask = (
                    within_candidate_mask & ~other_scale_only_positives
                )
            denominator_parts = [
                within_logits.masked_fill(~scale_within_candidate_mask, negative_infinity),
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
            # Reduce each candidate family before combining them.  This is
            # exactly the same partition function as concatenation, while it
            # avoids materializing a second [batch, time, batch*time] tensor
            # for every PAMS scale.
            family_log_partitions = tuple(
                torch.logsumexp(part, dim=-1) for part in denominator_parts
            )
            denominator = torch.logsumexp(
                torch.stack(family_log_partitions, dim=-1),
                dim=-1,
            )
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
    """Weighted SSHead total and legacy/reference-relative components."""

    total: Tensor
    cycle: Tensor
    spectral: Tensor
    variance: Tensor
    smoothness: Tensor
    reference: Tensor
    fundamental: Tensor
    lag: Tensor
    low_frequency: Tensor


class SSHeadLoss(nn.Module):
    """Independent self-supervised completion for the undisclosed head loss."""

    def __init__(
        self,
        cycle_weight: float = 1.0,
        spectral_weight: float = 1.0,
        variance_weight: float = 0.1,
        smoothness_weight: float = 0.01,
        *,
        confidence_weighted_period_losses: bool = False,
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
        self.confidence_weighted_period_losses = bool(
            confidence_weighted_period_losses
        )

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
        cycle_confidences: list[Tensor] = []
        spectral_losses: list[Tensor] = []
        spectral_confidences: list[Tensor] = []
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
                        cycle_confidences.append(sample_confidence)

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
                    spectral_losses.append(
                        1.0 - target_power / power.sum().clamp_min(1e-12)
                    )
                    spectral_confidences.append(sample_confidence)

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

        def period_mean_or_zero(
            items: list[Tensor],
            item_confidences: list[Tensor],
        ) -> Tensor:
            if not items:
                return zero
            if not self.confidence_weighted_period_losses:
                return torch.stack(items).mean()
            weights = torch.stack(item_confidences).to(
                device=stream.device,
                dtype=stream.dtype,
            )
            values = torch.stack(items)
            return (values * weights).sum() / weights.sum().clamp_min(1e-12)

        cycle = period_mean_or_zero(cycle_losses, cycle_confidences)
        spectral = period_mean_or_zero(spectral_losses, spectral_confidences)
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
            reference=zero,
            fundamental=zero,
            lag=zero,
            low_frequency=zero,
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


class ReferenceRelativeSSHeadLoss(nn.Module):
    """Identifiable target-free loss for reference-relative head inputs.

    This opt-in inferred objective deliberately does not reuse the legacy
    anti-variance term.  Regression to the pose-derived reference curve gives
    a constant output a non-zero gradient, while the remaining terms enforce
    the target-free period's fundamental, its time-domain lag recurrence,
    low-frequency rejection, and local smoothness.  Every component consumes
    the same continuous mask-aware z-score of the raw head output that is used
    by reference-relative inference.
    """

    def __init__(
        self,
        reference_weight: float = 1.0,
        fundamental_weight: float = 1.0,
        lag_weight: float = 1.0,
        low_frequency_weight: float = 0.1,
        smoothness_weight: float = 0.01,
        *,
        confidence_weighted_losses: bool = False,
    ) -> None:
        super().__init__()
        weights = (
            reference_weight,
            fundamental_weight,
            lag_weight,
            low_frequency_weight,
            smoothness_weight,
        )
        if any(weight < 0 for weight in weights):
            raise ValueError("reference-relative SSHead weights must be non-negative")
        self.reference_weight = float(reference_weight)
        self.fundamental_weight = float(fundamental_weight)
        self.lag_weight = float(lag_weight)
        self.low_frequency_weight = float(low_frequency_weight)
        self.smoothness_weight = float(smoothness_weight)
        self.confidence_weighted_losses = bool(confidence_weighted_losses)

    def compute(
        self,
        period_stream: Tensor,
        reference_teacher: Tensor,
        periods: Tensor,
        valid_mask: Tensor,
        *,
        period_confidence: Tensor,
        teacher_available: Tensor,
    ) -> SSHeadLossOutput:
        if period_stream.ndim != 2:
            raise ValueError("period_stream must have shape [batch, time]")
        batch, time = period_stream.shape
        if reference_teacher.shape != (batch, time):
            raise ValueError("reference_teacher must match period_stream")
        if valid_mask.shape != (batch, time):
            raise ValueError("valid_mask must match period_stream")
        if periods.shape not in {(batch,), (batch, 1)}:
            raise ValueError(f"periods must have shape [{batch}]")
        if period_confidence.shape not in {(batch,), (batch, 1)}:
            raise ValueError(f"period_confidence must have shape [{batch}]")
        if teacher_available.shape != (batch,):
            raise ValueError(f"teacher_available must have shape [{batch}]")

        stream = period_stream
        valid = valid_mask.to(device=stream.device, dtype=torch.bool)
        sample_periods = periods.reshape(batch).to(device=stream.device)
        confidences = period_confidence.reshape(batch).to(
            device=stream.device,
            dtype=stream.dtype,
        )
        available = teacher_available.to(device=stream.device, dtype=torch.bool)
        if not bool(torch.isfinite(sample_periods).all()) or bool(
            (sample_periods <= 0).any()
        ):
            raise ValueError("periods must be finite and positive")
        if (
            not bool(torch.isfinite(confidences).all())
            or bool((confidences < 0).any())
            or bool((confidences > 1).any())
        ):
            raise ValueError("period_confidence must be finite and in [0, 1]")
        usable_samples = available & (confidences > 0)
        normalization_mask = valid & usable_samples.unsqueeze(1)
        normalized_stream = masked_zscore(
            stream,
            normalization_mask,
        )
        normalized_teacher = masked_zscore(
            reference_teacher.to(device=stream.device, dtype=stream.dtype),
            normalization_mask,
        )

        references: list[Tensor] = []
        fundamentals: list[Tensor] = []
        lags: list[Tensor] = []
        low_frequencies: list[Tensor] = []
        smoothnesses: list[Tensor] = []
        weights: list[Tensor] = []
        zero = stream.sum() * 0.0

        for values, target, sample_period, sample_valid, confidence, usable in zip(
            normalized_stream,
            normalized_teacher,
            sample_periods,
            valid,
            confidences,
            available,
            strict=True,
        ):
            if not bool(usable) or not bool(confidence > 0) or int(sample_valid.sum()) < 2:
                continue
            selected = values[sample_valid]
            references.append(
                (selected - target[sample_valid]).square().mean()
            )

            period = max(1, int(round(float(sample_period.detach()))))
            if period < time:
                pair_valid = sample_valid[:-period] & sample_valid[period:]
                if pair_valid.any():
                    difference = values[:-period] - values[period:]
                    lags.append(difference[pair_valid].square().mean())
                else:
                    lags.append(zero)
            else:
                lags.append(zero)

            count = sample_valid.sum().to(dtype=values.dtype)
            sample_mean = torch.where(
                sample_valid,
                values,
                torch.zeros_like(values),
            ).sum() / count
            centered = torch.where(
                sample_valid,
                values - sample_mean,
                torch.zeros_like(values),
            )
            spectral_values = (
                centered.float()
                if centered.dtype in (torch.float16, torch.bfloat16)
                else centered
            )
            power = torch.fft.rfft(spectral_values).abs().square()
            if power.numel() > 1:
                power = power.clone()
                power[0] = 0.0
                total_power = power.sum().clamp_min(1e-12)
                target_bin = int(round(time / period))
                target_bin = min(max(target_bin, 1), power.numel() - 1)
                fundamentals.append(1.0 - power[target_bin] / total_power)
                low_frequencies.append(power[1:target_bin].sum() / total_power)
            else:
                fundamentals.append(zero)
                low_frequencies.append(zero)

            if time >= 3:
                triplet_valid = (
                    sample_valid[:-2]
                    & sample_valid[1:-1]
                    & sample_valid[2:]
                )
                if triplet_valid.any():
                    second_difference = (
                        values[2:] - 2.0 * values[1:-1] + values[:-2]
                    )
                    smoothnesses.append(
                        second_difference[triplet_valid].square().mean()
                    )
                else:
                    smoothnesses.append(zero)
            else:
                smoothnesses.append(zero)
            weights.append(confidence.detach())

        def reduce(items: list[Tensor]) -> Tensor:
            if not items:
                return zero
            stacked = torch.stack(items)
            if not self.confidence_weighted_losses:
                return stacked.mean()
            sample_weights = torch.stack(weights).to(
                device=stream.device,
                dtype=stream.dtype,
            )
            return (
                stacked * sample_weights
            ).sum() / sample_weights.sum().clamp_min(1e-12)

        reference = reduce(references)
        fundamental = reduce(fundamentals)
        lag = reduce(lags)
        low_frequency = reduce(low_frequencies)
        smoothness = reduce(smoothnesses)
        total = (
            self.reference_weight * reference
            + self.fundamental_weight * fundamental
            + self.lag_weight * lag
            + self.low_frequency_weight * low_frequency
            + self.smoothness_weight * smoothness
        )
        return SSHeadLossOutput(
            total=total,
            cycle=zero,
            spectral=zero,
            variance=zero,
            smoothness=smoothness,
            reference=reference,
            fundamental=fundamental,
            lag=lag,
            low_frequency=low_frequency,
        )

    def forward(
        self,
        period_stream: Tensor,
        reference_teacher: Tensor,
        periods: Tensor,
        valid_mask: Tensor,
        *,
        period_confidence: Tensor,
        teacher_available: Tensor,
    ) -> Tensor:
        return self.compute(
            period_stream,
            reference_teacher,
            periods,
            valid_mask,
            period_confidence=period_confidence,
            teacher_available=teacher_available,
        ).total
