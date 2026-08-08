"""Symmetric variance-aware cycle-back regression on paired native windows."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class CycleBackDirectionOutput:
    """One directional soft-nearest-neighbor cycle."""

    loss: Tensor
    valid_anchor_count: int
    candidate_pair_count: int
    mean_squared_position_error: Tensor
    mean_cycle_variance: Tensor
    predicted_normalized_positions: Tensor
    target_normalized_positions: Tensor
    valid_anchor_mask: Tensor


@dataclass(frozen=True, slots=True)
class CycleBackLossOutput:
    """Symmetric A->B->A and B->A->B cycle-back values."""

    total: Tensor
    a_to_b_to_a: CycleBackDirectionOutput
    b_to_a_to_b: CycleBackDirectionOutput

    @property
    def valid_anchor_count(self) -> int:
        return (
            self.a_to_b_to_a.valid_anchor_count
            + self.b_to_a_to_b.valid_anchor_count
        )


def _validate_embeddings_and_geometry(
    embeddings_a: Tensor,
    embeddings_b: Tensor,
    valid_a: Tensor,
    valid_b: Tensor,
    source_indices_a: Tensor,
    source_indices_b: Tensor,
    native_lengths_a: Tensor,
    native_lengths_b: Tensor,
) -> tuple[int, int, int]:
    if embeddings_a.ndim != 3:
        raise ValueError("cycle-back embeddings must have shape [pairs, window, dim]")
    if embeddings_b.shape != embeddings_a.shape:
        raise ValueError("paired embedding windows must have identical shapes")
    if not embeddings_a.is_floating_point() or not embeddings_b.is_floating_point():
        raise TypeError("cycle-back embeddings must use a floating-point dtype")
    if embeddings_b.dtype != embeddings_a.dtype:
        raise TypeError("paired cycle-back embeddings must use the same dtype")
    pairs, window, dimension = embeddings_a.shape
    if dimension < 1:
        raise ValueError("embedding dimension must be positive")
    for name, value in (("valid_a", valid_a), ("valid_b", valid_b)):
        if value.shape != (pairs, window) or value.dtype is not torch.bool:
            raise ValueError(f"{name} must be boolean [pairs, window]")
    for name, value in (
        ("source_indices_a", source_indices_a),
        ("source_indices_b", source_indices_b),
    ):
        if value.shape != (pairs, window) or value.dtype != torch.long:
            raise ValueError(f"{name} must be int64 [pairs, window]")
    for name, value in (
        ("native_lengths_a", native_lengths_a),
        ("native_lengths_b", native_lengths_b),
    ):
        if value.shape != (pairs,) or value.dtype != torch.long:
            raise ValueError(f"{name} must be int64 [pairs]")
        if value.numel() and int(value.min()) < 2:
            raise ValueError("native lengths must be at least two frames")
    tensors = (
        embeddings_b,
        valid_a,
        valid_b,
        source_indices_a,
        source_indices_b,
        native_lengths_a,
        native_lengths_b,
    )
    if any(value.device != embeddings_a.device for value in tensors):
        raise ValueError("all cycle-back tensors must be on the same device")
    for indices, lengths in (
        (source_indices_a, native_lengths_a),
        (source_indices_b, native_lengths_b),
    ):
        if indices.numel() and (
            int(indices.min()) < 0
            or bool((indices >= lengths.unsqueeze(1)).any())
        ):
            raise ValueError("source indices must lie inside each native timeline")
        if indices.shape[1] > 1 and bool((indices[:, 1:] - indices[:, :-1] != 1).any()):
            raise ValueError("source indices must preserve the contiguous native window grid")
    for embeddings, valid in (
        (embeddings_a, valid_a),
        (embeddings_b, valid_b),
    ):
        if valid.any() and not torch.isfinite(embeddings[valid]).all():
            raise ValueError("valid cycle-back embeddings must be finite")
    return pairs, window, dimension


def _masked_softmax(logits: Tensor, candidate_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Softmax with exact zeros when a row has no valid candidate."""

    if candidate_mask.shape != logits.shape:
        raise ValueError("candidate mask must match logits")
    if not logits.is_floating_point():
        raise TypeError("masked softmax logits must be floating point")
    has_candidate = candidate_mask.any(dim=-1)
    finite_floor = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(~candidate_mask, finite_floor)
    weights = torch.softmax(masked, dim=-1)
    weights = weights.masked_fill(~candidate_mask, 0.0)
    weights = torch.where(
        has_candidate.unsqueeze(-1),
        weights,
        torch.zeros_like(weights),
    )
    return weights, has_candidate


def _normalized_window_positions(
    source_indices: Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    window_length = source_indices.shape[1]
    if window_length < 2:
        raise ValueError("cycle-back window must contain at least two positions")
    offsets = (source_indices - source_indices[:, :1]).to(device=device, dtype=dtype)
    return offsets / float(window_length - 1)


def variance_aware_cycleback_direction(
    query_embeddings: Tensor,
    candidate_embeddings: Tensor,
    query_valid: Tensor,
    candidate_valid: Tensor,
    query_source_indices: Tensor,
    query_native_lengths: Tensor,
    *,
    temperature: float,
    variance_log_weight: float,
    variance_floor: float,
) -> CycleBackDirectionOutput:
    """Compute query->candidate->query regression for every valid query frame."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if not math.isfinite(variance_log_weight) or variance_log_weight < 0.0:
        raise ValueError("variance_log_weight must be non-negative")
    if not math.isfinite(variance_floor) or variance_floor <= 0.0:
        raise ValueError("variance_floor must be positive")
    normalized_query = F.normalize(
        query_embeddings.masked_fill(~query_valid.unsqueeze(-1), 0.0),
        dim=-1,
        eps=1e-12,
    )
    normalized_candidate = F.normalize(
        candidate_embeddings.masked_fill(~candidate_valid.unsqueeze(-1), 0.0),
        dim=-1,
        eps=1e-12,
    )
    forward_logits = torch.einsum(
        "btd,bsd->bts",
        normalized_query,
        normalized_candidate,
    ) / temperature
    forward_candidates = candidate_valid.unsqueeze(1).expand_as(forward_logits)
    forward_weights, has_forward_candidate = _masked_softmax(
        forward_logits,
        forward_candidates,
    )
    soft_candidate = torch.einsum(
        "bts,bsd->btd",
        forward_weights,
        normalized_candidate,
    )

    backward_logits = torch.einsum(
        "btd,bsd->bts",
        soft_candidate,
        normalized_query,
    ) / temperature
    backward_candidates = query_valid.unsqueeze(1).expand_as(backward_logits)
    backward_weights, has_backward_candidate = _masked_softmax(
        backward_logits,
        backward_candidates,
    )

    positions = _normalized_window_positions(
        query_source_indices,
        dtype=query_embeddings.dtype,
        device=query_embeddings.device,
    )
    predicted = torch.einsum("bts,bs->bt", backward_weights, positions)
    centered = positions.unsqueeze(1) - predicted.unsqueeze(-1)
    variance = torch.einsum(
        "bts,bts->bt",
        backward_weights,
        centered.square(),
    ).clamp_min(variance_floor)
    squared_error = (positions - predicted).square()
    per_anchor = squared_error / variance + variance_log_weight * torch.log(variance)

    query_candidate_count = query_valid.sum(dim=1)
    candidate_candidate_count = candidate_valid.sum(dim=1)
    eligible_pair = (query_candidate_count >= 2) & (candidate_candidate_count >= 1)
    valid_anchor = (
        query_valid
        & has_forward_candidate
        & has_backward_candidate
        & eligible_pair.unsqueeze(1)
    )
    zero = query_embeddings.sum() * 0.0
    if valid_anchor.any():
        loss = per_anchor[valid_anchor].mean()
        mean_error = squared_error[valid_anchor].mean()
        mean_variance = variance[valid_anchor].mean()
    else:
        loss = zero
        mean_error = zero
        mean_variance = zero
    return CycleBackDirectionOutput(
        loss=loss,
        valid_anchor_count=int(valid_anchor.sum()),
        candidate_pair_count=int(eligible_pair.sum()),
        mean_squared_position_error=mean_error,
        mean_cycle_variance=mean_variance,
        predicted_normalized_positions=predicted,
        target_normalized_positions=positions,
        valid_anchor_mask=valid_anchor,
    )


class ConventionalCycleBackLoss(nn.Module):
    """Symmetric same-video cycle-back objective with no batch negatives.

    Each batch row is evaluated independently.  No tensor is flattened across
    videos, and no prototype/cluster bank is accepted by this API.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.1,
        variance_log_weight: float = 0.001,
        variance_floor: float = 1e-6,
    ) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not math.isfinite(variance_log_weight) or variance_log_weight < 0.0:
            raise ValueError("variance_log_weight must be non-negative")
        if not math.isfinite(variance_floor) or variance_floor <= 0.0:
            raise ValueError("variance_floor must be positive")
        self.temperature = float(temperature)
        self.variance_log_weight = float(variance_log_weight)
        self.variance_floor = float(variance_floor)

    def _compute(
        self,
        embeddings_a: Tensor,
        embeddings_b: Tensor,
        valid_a: Tensor,
        valid_b: Tensor,
        source_indices_a: Tensor,
        source_indices_b: Tensor,
        native_lengths_a: Tensor,
        native_lengths_b: Tensor,
    ) -> CycleBackLossOutput:
        _validate_embeddings_and_geometry(
            embeddings_a,
            embeddings_b,
            valid_a,
            valid_b,
            source_indices_a,
            source_indices_b,
            native_lengths_a,
            native_lengths_b,
        )
        forward = variance_aware_cycleback_direction(
            embeddings_a,
            embeddings_b,
            valid_a,
            valid_b,
            source_indices_a,
            native_lengths_a,
            temperature=self.temperature,
            variance_log_weight=self.variance_log_weight,
            variance_floor=self.variance_floor,
        )
        reverse = variance_aware_cycleback_direction(
            embeddings_b,
            embeddings_a,
            valid_b,
            valid_a,
            source_indices_b,
            native_lengths_b,
            temperature=self.temperature,
            variance_log_weight=self.variance_log_weight,
            variance_floor=self.variance_floor,
        )
        total = 0.5 * (forward.loss + reverse.loss)
        return CycleBackLossOutput(
            total=total,
            a_to_b_to_a=forward,
            b_to_a_to_b=reverse,
        )

    @staticmethod
    def _validate_same_video_ids(
        video_ids_a: Sequence[str],
        video_ids_b: Sequence[str],
        *,
        pair_count: int,
    ) -> None:
        first = tuple(str(value).strip() for value in video_ids_a)
        second = tuple(str(value).strip() for value in video_ids_b)
        if len(first) != pair_count or len(second) != pair_count:
            raise ValueError("cycle-back video IDs must match pair count")
        if any(not value for value in first + second):
            raise ValueError("cycle-back video IDs must be non-empty")
        if first != second:
            raise ValueError(
                "training cycle-back pairs must come from the same video; "
                "different-video inputs are diagnostic-only"
            )

    @staticmethod
    def _validate_disjoint_source_indices(
        source_indices_a: Tensor,
        source_indices_b: Tensor,
        valid_a: Tensor,
        valid_b: Tensor,
    ) -> None:
        for row_a, row_b, mask_a, mask_b in zip(
            source_indices_a.detach().cpu(),
            source_indices_b.detach().cpu(),
            valid_a.detach().cpu(),
            valid_b.detach().cpu(),
            strict=True,
        ):
            if set(row_a[mask_a].tolist()).intersection(row_b[mask_b].tolist()):
                raise ValueError(
                    "training cycle-back windows must have disjoint source frames"
                )

    def compute(
        self,
        embeddings_a: Tensor,
        embeddings_b: Tensor,
        valid_a: Tensor,
        valid_b: Tensor,
        source_indices_a: Tensor,
        source_indices_b: Tensor,
        native_lengths: Tensor,
        *,
        video_ids_a: Sequence[str],
        video_ids_b: Sequence[str],
    ) -> CycleBackLossOutput:
        pair_count = int(embeddings_a.shape[0]) if embeddings_a.ndim else 0
        self._validate_same_video_ids(
            video_ids_a,
            video_ids_b,
            pair_count=pair_count,
        )
        self._validate_disjoint_source_indices(
            source_indices_a,
            source_indices_b,
            valid_a,
            valid_b,
        )
        return self._compute(
            embeddings_a,
            embeddings_b,
            valid_a,
            valid_b,
            source_indices_a,
            source_indices_b,
            native_lengths,
            native_lengths,
        )

    def diagnostic_mismatched_video_control(
        self,
        embeddings_a: Tensor,
        embeddings_b: Tensor,
        valid_a: Tensor,
        valid_b: Tensor,
        source_indices_a: Tensor,
        source_indices_b: Tensor,
        native_lengths_a: Tensor,
        native_lengths_b: Tensor,
        *,
        video_ids_a: Sequence[str],
        video_ids_b: Sequence[str],
    ) -> CycleBackLossOutput:
        """Evaluate, but never optimize, an explicit different-video control."""

        first = tuple(str(value).strip() for value in video_ids_a)
        second = tuple(str(value).strip() for value in video_ids_b)
        pair_count = int(embeddings_a.shape[0]) if embeddings_a.ndim else 0
        if len(first) != pair_count or len(second) != pair_count:
            raise ValueError("diagnostic video IDs must match pair count")
        if any(not value for value in first + second):
            raise ValueError("diagnostic video IDs must be non-empty")
        if any(a == b for a, b in zip(first, second, strict=True)):
            raise ValueError("different-video control contains a same-video row")
        return self._compute(
            embeddings_a,
            embeddings_b,
            valid_a,
            valid_b,
            source_indices_a,
            source_indices_b,
            native_lengths_a,
            native_lengths_b,
        )

    def forward(
        self,
        embeddings_a: Tensor,
        embeddings_b: Tensor,
        valid_a: Tensor,
        valid_b: Tensor,
        source_indices_a: Tensor,
        source_indices_b: Tensor,
        native_lengths: Tensor,
        *,
        video_ids_a: Sequence[str],
        video_ids_b: Sequence[str],
    ) -> Tensor:
        return self.compute(
            embeddings_a,
            embeddings_b,
            valid_a,
            valid_b,
            source_indices_a,
            source_indices_b,
            native_lengths,
            video_ids_a=video_ids_a,
            video_ids_b=video_ids_b,
        ).total
