"""Run the frozen train337-only PAMS-v15 teacher-identifiability probe.

This diagnostic tests whether the inferred broad, similarity-maximizing PAMS
correspondence windows make the projected-pose period teacher non-identifiable.
It accepts only the exact v15 terminal encoder artifacts, their checkpoint-bound
train337 pose cache, and the exact source-export receipt.  It has no interface
for labels, development data, sealed-test data, training, or optimizer state.

The primary unit of analysis is one positive-confidence training video.  The
probe compares the current selector with deterministic teacher interventions
and with a mandatory-center intervention that *adds* the strict indices
``t +/- round(T)`` when those exact frames are valid.  The latter never searches
for a nearby frame.  All model parameters and checkpoint bytes remain unchanged.

PyTorch checkpoints are pickle containers.  Use only checkpoints produced by
this repository in the trusted experiment environment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

import pams.losses as _losses
from pams.config import PAMSConfig, load_config
from pams.data import PoseCacheSetSnapshot
from pams.diagnostics import (
    _device,
    _identifier_commitment,
    _load_training_poses,
    _peek_checkpoint,
    _summary,
)
from pams.period import (
    estimate_period_from_embedding_velocity_vectors,
    estimate_period_from_projected_pose,
)
from pams.reproducibility import clean_git_revision, hardware_fingerprint, sha256_json
from pams.training import (
    VideoPrototypeBank,
    collate_pose_sequences,
    load_model_checkpoint,
    validate_terminal_checkpoint,
)
from pams.types import PoseSequence

try:
    from scripts.server import run_pams_v15_terminal_dual_path_gate as _v15
except ModuleNotFoundError:
    # Formal runs may mount the audited gate files in /gate-code while the
    # checkpoint-producing source remains mounted read-only at /workspace.
    import run_pams_v15_terminal_dual_path_gate as _v15  # type: ignore[no-redef]

_v14 = _v15._v14

_ARTIFACT_TYPE = "pams_v15_teacher_identifiability_probe"
_RECEIPT_TYPE = "pams_v15_teacher_identifiability_probe_receipt"
_CLASSIFICATION = "inferred target-free read-only teacher-identifiability probe"
_EXPECTED_TRAINING_VIDEOS = 337
_EXPECTED_POSITIVE_CONFIDENCE_VIDEOS = 321
_EXPECTED_ENCODER_CHECKPOINT_SHA256 = (
    "aa1750154e0ba36d41188cc023bb382db83db9f5c69b1867f310cdc636a48e19"
)
_EXPECTED_ENCODER_PROGRESS_SHA256 = (
    "70226a5f01a3d9e8b688fde53736c397a135c242d8e15ce429a36615552029c4"
)
_BOOTSTRAP_RESAMPLES = 10_000
_BOOTSTRAP_SEED = 2026
_MAXIMUM_ANCHORS = 64
_ANCHOR_BUDGETS = (32, 64)
_HARD_CROSS_CLUSTER_PROTOTYPES = 4
_HARMONIC_FRAMES = 8
_HARMONIC_TOLERANCE_FRAMES = 1
_EPSILON = 1e-6

# Frozen before this runner is evaluated on the v15 checkpoint.  These values
# intentionally authorize at most a ten-epoch, train337-only mandatory-center
# continuation.  They can never authorize SSHead, dev84, or test105 access.
_THRESHOLDS: dict[str, float | int] = {
    "permutable_positive_confidence_videos_minimum": 300,
    "paired_eligible_fraction_minimum": 0.50,
    "gradient_zero_fraction_maximum": 0.01,
    "w8_all_scale_opportunity_fraction_minimum": 0.75,
    "s8_video_median_minimum": 0.75,
    "s8_bootstrap_lower_95_minimum": 0.65,
    "r8_video_median_maximum": 0.02,
    "r8_bootstrap_upper_95_maximum": 0.05,
    "aperm_identifiable_video_median_minimum": 0.05,
    "aperm_identifiable_bootstrap_lower_95_minimum": 0.02,
    "aperm_blind_video_median_maximum": 0.02,
    "aperm_blind_bootstrap_upper_95_maximum": 0.05,
    "gperm_identifiable_video_median_maximum": 0.90,
    "gperm_identifiable_bootstrap_upper_95_maximum": 0.95,
    "gperm_blind_video_median_minimum": 0.95,
    "gperm_blind_bootstrap_lower_95_minimum": 0.90,
    "center_direction_eligibility_fraction_minimum": 0.50,
    "center_nonzero_gradient_fraction_minimum": 0.99,
    "gperm_center_video_median_maximum": 0.90,
    "gperm_center_bootstrap_upper_95_maximum": 0.95,
    "gperm_current_minus_center_video_median_minimum": 0.05,
    "gperm_current_minus_center_bootstrap_lower_95_exclusive": 0.0,
    "convergence_w8_absolute_difference_maximum": 0.02,
    "convergence_s8_absolute_difference_maximum": 0.02,
    "convergence_relative_nll_advantage_difference_maximum": 0.01,
    "convergence_gradient_cosine_difference_maximum": 0.03,
}


@dataclass(frozen=True, slots=True)
class _EncodedTrain337:
    video_ids: tuple[str, ...]
    embeddings: Tensor
    valid_mask: Tensor
    projected_periods: Tensor
    projected_confidences: Tensor
    post_pe_periods: Tensor
    post_pe_confidences: Tensor


def _midpoint_quantile_anchor_indices(
    valid_mask: Tensor,
    *,
    maximum: int = _MAXIMUM_ANCHORS,
) -> Tensor:
    """Return deterministic nested midpoint-quantile anchors.

    The returned row is the 64-anchor schedule (or every valid frame if fewer
    than 64 exist).  Budget 32 uses even schedule positions and the recorded
    16-anchor audit subset uses every fourth schedule position.
    """

    if valid_mask.ndim != 1:
        raise ValueError("valid_mask must be one-dimensional")
    if maximum < 1:
        raise ValueError("maximum must be positive")
    valid = torch.nonzero(
        valid_mask.detach().to(device="cpu", dtype=torch.bool),
        as_tuple=False,
    ).flatten()
    count = min(maximum, int(valid.numel()))
    if not count:
        return torch.empty(0, dtype=torch.long)
    # floor((k+1/2)*N/K) is the midpoint of each equal-mass quantile cell.
    slots = torch.arange(count, dtype=torch.float64)
    ranks = torch.floor((slots + 0.5) * valid.numel() / count).to(torch.long)
    selected = valid[ranks.clamp_max(valid.numel() - 1)]
    if int(torch.unique(selected).numel()) != count:
        raise RuntimeError("midpoint-quantile anchor construction produced duplicates")
    return selected


def _budget_schedule_positions(anchor_total: int, budget: int) -> Tensor:
    if anchor_total < 0:
        raise ValueError("anchor_total must be non-negative")
    if budget not in {16, 32, 64}:
        raise ValueError("budget must be one of 16, 32, or 64")
    stride = 64 // budget
    return torch.arange(0, anchor_total, stride, dtype=torch.long)


def _teacher_permutation(
    periods: Tensor,
    confidences: Tensor,
    valid_counts: Tensor,
    video_ids: Sequence[str],
) -> tuple[Tensor, Tensor, Tensor, dict[str, Any]]:
    """Cyclically permute paired teacher evidence inside exact-length strata."""

    if periods.ndim != 1 or confidences.shape != periods.shape:
        raise ValueError("periods and confidences must be equal one-dimensional tensors")
    if valid_counts.shape != periods.shape:
        raise ValueError("valid_counts must match periods")
    if len(video_ids) != periods.numel() or len(set(video_ids)) != len(video_ids):
        raise ValueError("video_ids must be unique and match periods")
    if not torch.isfinite(periods).all() or not torch.isfinite(confidences).all():
        raise ValueError("teacher values must be finite")

    permuted_periods = periods.detach().clone()
    permuted_confidences = confidences.detach().clone()
    permutable = torch.zeros_like(confidences, dtype=torch.bool)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, (confidence, count) in enumerate(
        zip(confidences.tolist(), valid_counts.tolist(), strict=True)
    ):
        if float(confidence) > 0.0:
            groups[int(count)].append(index)

    rows: list[dict[str, Any]] = []
    singleton_total = 0
    for valid_count in sorted(groups):
        members = sorted(groups[valid_count], key=lambda index: str(video_ids[index]))
        if len(members) < 2:
            singleton_total += len(members)
            rows.append(
                {
                    "valid_count": valid_count,
                    "positive_confidence_video_total": len(members),
                    "cyclic_shift": 0,
                    "permutable": False,
                }
            )
            continue
        donors = members[1:] + members[:1]
        receiver_index = torch.tensor(members, dtype=torch.long)
        donor_index = torch.tensor(donors, dtype=torch.long)
        permuted_periods[receiver_index] = periods[donor_index]
        permuted_confidences[receiver_index] = confidences[donor_index]
        permutable[receiver_index] = True
        rows.append(
            {
                "valid_count": valid_count,
                "positive_confidence_video_total": len(members),
                "cyclic_shift": 1,
                "permutable": True,
            }
        )

    original_pairs = Counter(
        (float(period), float(confidence))
        for period, confidence, selected in zip(
            periods.tolist(),
            confidences.tolist(),
            permutable.tolist(),
            strict=True,
        )
        if selected
    )
    permuted_pairs = Counter(
        (float(period), float(confidence))
        for period, confidence, selected in zip(
            permuted_periods.tolist(),
            permuted_confidences.tolist(),
            permutable.tolist(),
            strict=True,
        )
        if selected
    )
    if original_pairs != permuted_pairs:
        raise RuntimeError("teacher permutation did not preserve paired evidence")
    return (
        permuted_periods,
        permuted_confidences,
        permutable,
        {
            "algorithm": (
                "Positive-confidence videos are partitioned by exact valid-frame "
                "count, sorted lexicographically by video_id, and assigned the next "
                "(period, confidence) pair under a nonzero cyclic shift. Singleton "
                "strata are excluded from paired permutation statistics."
            ),
            "strata": rows,
            "positive_confidence_video_total": int((confidences > 0).sum()),
            "permutable_positive_confidence_video_total": int(permutable.sum()),
            "singleton_positive_confidence_video_total": singleton_total,
            "paired_period_confidence_multiset_preserved": True,
        },
    )


def _strict_center_indices(period: float, valid_mask: Tensor) -> Tensor:
    """Return strict past/future ``t +/- round(T)`` indices or ``-1``."""

    if valid_mask.ndim != 1:
        raise ValueError("valid_mask must be one-dimensional")
    time = valid_mask.numel()
    rounded = max(1, int(torch.round(torch.tensor(float(period))).item()))
    anchors = torch.arange(time, device=valid_mask.device)
    result: list[Tensor] = []
    valid = valid_mask.to(dtype=torch.bool)
    for direction in (-1, 1):
        candidate = anchors + direction * rounded
        in_bounds = (candidate >= 0) & (candidate < time) & valid
        safe = candidate.clamp(0, max(time - 1, 0))
        if time:
            in_bounds &= valid[safe]
        result.append(candidate.masked_fill(~in_bounds, -1))
    return torch.stack(result, dim=-1)


def _near_positive_multiple_of_eight(lags: Tensor) -> Tensor:
    if lags.numel() == 0:
        return torch.empty_like(lags, dtype=torch.bool)
    integer_lags = lags.to(dtype=torch.long)
    remainder = torch.remainder(integer_lags, _HARMONIC_FRAMES)
    return (integer_lags > 0) & (
        (remainder <= _HARMONIC_TOLERANCE_FRAMES)
        | (remainder >= _HARMONIC_FRAMES - _HARMONIC_TOLERANCE_FRAMES)
    )


def _encode_train337(
    model: Any,
    sequences: Sequence[PoseSequence],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> _EncodedTrain337:
    video_ids: list[str] = []
    embeddings: list[Tensor] = []
    valid_masks: list[Tensor] = []
    projected_periods: list[Tensor] = []
    projected_confidences: list[Tensor] = []
    post_periods: list[Tensor] = []
    post_confidences: list[Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            post_pe, projected = model.encoder.forward_with_pre_pe(
                batch.poses,
                batch.valid_mask,
            )
            teacher_period, teacher_confidence = estimate_period_from_projected_pose(
                projected,
                minimum=config.period.minimum,
                maximum=config.period.maximum,
                valid_mask=batch.valid_mask,
            )
            post_period, post_confidence = estimate_period_from_embedding_velocity_vectors(
                post_pe,
                minimum=config.period.minimum,
                maximum=config.period.maximum,
                valid_mask=batch.valid_mask,
            )
            video_ids.extend(batch.video_ids)
            embeddings.append(post_pe.detach().to(device="cpu", dtype=torch.float32))
            valid_masks.append(batch.valid_mask.detach().to(device="cpu", dtype=torch.bool))
            projected_periods.append(teacher_period.detach().to(device="cpu"))
            projected_confidences.append(teacher_confidence.detach().to(device="cpu"))
            post_periods.append(post_period.detach().to(device="cpu"))
            post_confidences.append(post_confidence.detach().to(device="cpu"))
    return _EncodedTrain337(
        video_ids=tuple(video_ids),
        embeddings=torch.cat(embeddings),
        valid_mask=torch.cat(valid_masks),
        projected_periods=torch.cat(projected_periods).float(),
        projected_confidences=torch.cat(projected_confidences).float(),
        post_pe_periods=torch.cat(post_periods).float(),
        post_pe_confidences=torch.cat(post_confidences).float(),
    )


def _period_distribution(periods: Tensor, confidences: Tensor) -> dict[str, Any]:
    period_values = [float(value) for value in periods.tolist()]
    confidence_values = [float(value) for value in confidences.tolist()]
    frequencies = Counter(format(value, ".9g") for value in period_values)
    mode_key, mode_frequency = min(
        frequencies.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return {
        "record_total": len(period_values),
        "positive_confidence_total": sum(value > 0.0 for value in confidence_values),
        "period_frames": _summary(period_values),
        "confidence": _summary(confidence_values),
        "mode_period_frames": float(mode_key),
        "mode_frequency": mode_frequency,
        "mode_share": mode_frequency / len(period_values),
        "unique_period_total": len(frequencies),
        "period_histogram": dict(sorted(frequencies.items())),
    }


def _cross_period_summary(
    reference_periods: Tensor,
    reference_confidences: Tensor,
    candidate_periods: Tensor,
    candidate_confidences: Tensor,
) -> dict[str, Any]:
    eligible = (reference_confidences > 0) & (candidate_confidences > 0)
    errors = (candidate_periods[eligible] - reference_periods[eligible]).abs() / reference_periods[
        eligible
    ].clamp_min(_EPSILON)
    return {
        "eligible_video_total": int(eligible.sum()),
        "eligible_video_fraction": float(eligible.float().mean()),
        "relative_error": _summary(errors.tolist()),
    }


def _load_prototype_bank(checkpoint_path: Path) -> VideoPrototypeBank:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("encoder checkpoint payload must be a mapping")
    raw_bank = payload.get("prototype_bank")
    if not isinstance(raw_bank, Mapping):
        raise ValueError("terminal v15 checkpoint is missing its prototype bank")
    return VideoPrototypeBank.from_checkpoint(raw_bank)


def _geometry_opportunity(
    periods: Tensor,
    confidences: Tensor,
    video_ids: Sequence[str],
    *,
    scales: Sequence[float],
    time: int,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    scale_hits = {format(float(scale), "g"): 0 for scale in scales}
    positive_total = 0
    all_scale_hits = 0
    for video_id, period, confidence in zip(
        video_ids,
        periods.tolist(),
        confidences.tolist(),
        strict=True,
    ):
        if float(confidence) <= 0.0:
            continue
        positive_total += 1
        rounded = max(1, int(torch.round(torch.tensor(float(period))).item()))
        opportunities: dict[str, bool] = {}
        for scale in scales:
            radius = max(1, int(round(rounded * float(scale) / 2.0)))
            hit = any(
                abs(multiple - rounded) <= radius
                for multiple in range(_HARMONIC_FRAMES, time, _HARMONIC_FRAMES)
            )
            key = format(float(scale), "g")
            opportunities[key] = hit
            scale_hits[key] += int(hit)
        all_hit = all(opportunities.values())
        all_scale_hits += int(all_hit)
        rows.append(
            {
                "video_id_sha256": _identifier_commitment((video_id,)),
                "teacher_period_frames": float(period),
                "teacher_confidence": float(confidence),
                "opportunity_by_scale": opportunities,
                "all_scales_contain_positive_8_multiple": all_hit,
            }
        )
    if not positive_total:
        raise RuntimeError("geometry opportunity requires positive-confidence videos")
    return {
        "algorithm": (
            "For each positive-confidence video and scale, test whether the "
            "integer lag interval [round(T)-round(scale*T/2), "
            "round(T)+round(scale*T/2)] contains a positive multiple of eight "
            "below the 256-frame sequence length. This is geometry-only and "
            "does not inspect labels or embedding similarities."
        ),
        "positive_confidence_video_total": positive_total,
        "per_scale_opportunity_fraction": {
            key: hits / positive_total for key, hits in scale_hits.items()
        },
        "all_scale_opportunity_video_total": all_scale_hits,
        "all_scale_opportunity_fraction": all_scale_hits / positive_total,
        "rows": rows,
    }


def _selector_indices(
    similarities: Tensor,
    *,
    period: float,
    scale: float,
    valid_mask: Tensor,
) -> Tensor:
    selected = _losses._correspondences_from_similarities(
        similarities.unsqueeze(0),
        torch.tensor([period], dtype=similarities.dtype, device=similarities.device),
        scale=float(scale),
        valid=valid_mask.unsqueeze(0),
    )
    return selected[0]


def _denominator_and_row_gradients(
    video_embeddings: Tensor,
    valid_mask: Tensor,
    anchors: Tensor,
    *,
    flat_embeddings: Tensor,
    flat_owner: Tensor,
    owner_index: int,
    bank_features: Tensor,
    bank_clusters: Tensor,
    owner_cluster: int,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return common NLL denominator and its independent query-row gradients."""

    query = video_embeddings.index_select(0, anchors).detach().clone().requires_grad_(True)
    normalized_query = F.normalize(query, dim=-1, eps=1e-12)
    normalized_video = F.normalize(video_embeddings.detach(), dim=-1, eps=1e-12)
    normalized_flat = flat_embeddings.detach()
    normalized_bank = bank_features.detach()

    within = normalized_query @ normalized_video.transpose(0, 1) / temperature
    within_allowed = valid_mask.to(dtype=torch.bool).unsqueeze(0).expand_as(within).clone()
    within_allowed.scatter_(1, anchors.unsqueeze(1), False)
    within_partition = torch.logsumexp(
        within.masked_fill(~within_allowed, float("-inf")),
        dim=-1,
    )

    all_frames = normalized_query @ normalized_flat.transpose(0, 1) / temperature
    other_allowed = (flat_owner != owner_index).unsqueeze(0)
    other_partition = torch.logsumexp(
        all_frames.masked_fill(~other_allowed, float("-inf")),
        dim=-1,
    )

    bank_logits = normalized_query @ normalized_bank.transpose(0, 1) / temperature
    bank_allowed = bank_clusters != int(owner_cluster)
    available = int(bank_allowed.sum())
    if available < _HARD_CROSS_CLUSTER_PROTOTYPES:
        raise RuntimeError("fewer than four cross-cluster prototypes are available")
    hard_values = torch.topk(
        bank_logits.masked_fill(~bank_allowed.unsqueeze(0), float("-inf")),
        k=_HARD_CROSS_CLUSTER_PROTOTYPES,
        dim=-1,
    ).values
    bank_partition = torch.logsumexp(hard_values, dim=-1)

    denominator = torch.logsumexp(
        torch.stack((within_partition, other_partition, bank_partition), dim=-1),
        dim=-1,
    )
    row_gradients = torch.autograd.grad(denominator.sum(), query)[0].detach()
    return (
        denominator.detach(),
        row_gradients,
        normalized_query.detach(),
    )


def _projection_gradient(
    raw_query: Tensor,
    normalized_query: Tensor,
    target: Tensor,
    *,
    temperature: float,
) -> Tensor:
    """Gradient of ``q_normalized dot target / temperature`` wrt raw query."""

    norms = raw_query.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    parallel = (normalized_query * target).sum(dim=-1, keepdim=True)
    return (target - normalized_query * parallel) / (norms * temperature)


def _video_arm_measurement(
    video_embeddings: Tensor,
    valid_mask: Tensor,
    anchors64: Tensor,
    *,
    period: float,
    confidence: float,
    scales: Sequence[float],
    temperature: float,
    denominator: Tensor,
    denominator_row_gradients: Tensor,
    normalized_query: Tensor,
    budget: int,
    mandatory_center: bool,
    canonical_selections: Mapping[str, Tensor] | None,
) -> tuple[dict[str, Any], Tensor, dict[str, Tensor]]:
    """Measure one arm for one video using the shared frozen denominator."""

    schedule_positions = _budget_schedule_positions(int(anchors64.numel()), budget)
    if schedule_positions.numel() == 0 or confidence <= 0.0:
        dimension = video_embeddings.shape[-1]
        return (
            {
                "eligible_cell_total": 0,
                "loss": None,
                "gradient_norm": 0.0,
                "selected_correspondence_total": 0,
                "near_8_multiple_fraction": None,
                "median_relative_teacher_deviation": None,
                "exact_canonical_selected_index_agreement": None,
                "center_direction_eligibility_fraction": None,
                "per_scale": {},
            },
            torch.zeros(
                (anchors64.numel(), dimension),
                dtype=video_embeddings.dtype,
                device=video_embeddings.device,
            ),
            {},
        )

    anchor_indices = anchors64.to(device=video_embeddings.device)
    selected_anchor_indices = anchor_indices.index_select(
        0,
        schedule_positions.to(device=anchor_indices.device),
    )
    full = F.normalize(video_embeddings.detach(), dim=-1, eps=1e-12)
    similarities = full @ full.transpose(0, 1)
    strict_center = _strict_center_indices(period, valid_mask)

    target_accumulator = torch.zeros_like(normalized_query)
    denominator_counts = torch.zeros(
        anchors64.numel(),
        dtype=video_embeddings.dtype,
        device=video_embeddings.device,
    )
    all_lags: list[int] = []
    all_relative_deviations: list[float] = []
    agreement_numerator = 0
    agreement_denominator = 0
    center_eligible = 0
    center_possible = 0
    per_scale: dict[str, Any] = {}
    selections: dict[str, Tensor] = {}
    scale_loss_parts: dict[str, tuple[float, int]] = {}

    anchor_to_slot = {int(anchor): slot for slot, anchor in enumerate(anchor_indices.tolist())}
    selected_slots = {int(slot) for slot in schedule_positions.tolist()}
    for scale in scales:
        scale_key = format(float(scale), "g")
        selected = _selector_indices(
            similarities,
            period=period,
            scale=float(scale),
            valid_mask=valid_mask,
        )
        selections[scale_key] = selected.detach().to(device="cpu")
        canonical = None if canonical_selections is None else canonical_selections[scale_key]
        scale_loss_numerator = 0.0
        scale_cell_total = 0
        scale_lags: list[int] = []
        scale_relative: list[float] = []
        scale_agree_numerator = 0
        scale_agree_denominator = 0

        for anchor in selected_anchor_indices.tolist():
            slot = anchor_to_slot[int(anchor)]
            if slot not in selected_slots:
                raise RuntimeError("anchor schedule lookup is inconsistent")
            for direction_slot in range(2):
                current_index = int(selected[int(anchor), direction_slot])
                positives: list[int] = []
                if current_index >= 0:
                    positives.append(current_index)
                exact_index = int(strict_center[int(anchor), direction_slot])
                if mandatory_center:
                    center_possible += 1
                    if exact_index >= 0:
                        center_eligible += 1
                        if exact_index not in positives:
                            positives.append(exact_index)
                if not positives:
                    continue

                candidate_vectors = full[
                    torch.tensor(positives, dtype=torch.long, device=full.device)
                ]
                mean_candidate = candidate_vectors.mean(dim=0)
                query = normalized_query[slot]
                positive_logit = float((query * mean_candidate).sum() / temperature)
                loss_value = float(denominator[slot]) - positive_logit
                scale_loss_numerator += loss_value
                scale_cell_total += 1
                denominator_counts[slot] += 1.0
                target_accumulator[slot] += mean_candidate

                for positive_index in positives:
                    lag = abs(int(positive_index) - int(anchor))
                    scale_lags.append(lag)
                    all_lags.append(lag)
                    deviation = abs(lag - float(period)) / max(float(period), _EPSILON)
                    scale_relative.append(deviation)
                    all_relative_deviations.append(deviation)

                if canonical is not None and current_index >= 0:
                    canonical_index = int(canonical[int(anchor), direction_slot])
                    if canonical_index >= 0:
                        scale_agree_denominator += 1
                        agreement_denominator += 1
                        equal = int(current_index == canonical_index)
                        scale_agree_numerator += equal
                        agreement_numerator += equal

        if not scale_cell_total:
            raise RuntimeError(
                f"positive-confidence video has no eligible cells at scale={scale_key}"
            )
        scale_loss = scale_loss_numerator / scale_cell_total
        scale_loss_parts[scale_key] = (scale_loss_numerator, scale_cell_total)
        lag_tensor = torch.tensor(scale_lags, dtype=torch.long)
        near = _near_positive_multiple_of_eight(lag_tensor)
        per_scale[scale_key] = {
            "eligible_cell_total": scale_cell_total,
            "loss": scale_loss,
            "selected_correspondence_total": len(scale_lags),
            "selected_abs_lag": _summary(scale_lags),
            "selected_abs_lag_histogram": {
                str(key): value for key, value in sorted(Counter(scale_lags).items())
            },
            "median_relative_teacher_deviation": float(np.median(scale_relative)),
            "near_8_multiple_fraction": float(near.float().mean()),
            "exact_canonical_selected_index_agreement": (
                scale_agree_numerator / scale_agree_denominator if scale_agree_denominator else None
            ),
        }

    eligible_cell_total = int(denominator_counts.sum().item())
    if not eligible_cell_total:
        raise RuntimeError("positive-confidence video has no eligible correspondence cells")
    weights = denominator_counts / float(eligible_cell_total)
    target = target_accumulator / float(eligible_cell_total)
    raw_query = video_embeddings.index_select(0, anchor_indices).detach()
    positive_gradient = _projection_gradient(
        raw_query,
        normalized_query,
        target,
        temperature=temperature,
    )
    gradient = denominator_row_gradients * weights.unsqueeze(-1) - positive_gradient
    total_loss_numerator = sum(value[0] for value in scale_loss_parts.values())
    total_loss_cells = sum(value[1] for value in scale_loss_parts.values())
    total_loss = total_loss_numerator / total_loss_cells
    all_lag_tensor = torch.tensor(all_lags, dtype=torch.long)
    all_near = _near_positive_multiple_of_eight(all_lag_tensor)
    return (
        {
            "eligible_cell_total": eligible_cell_total,
            "loss": total_loss,
            "gradient_norm": float(gradient.norm()),
            "selected_correspondence_total": len(all_lags),
            "selected_abs_lag": _summary(all_lags),
            "selected_abs_lag_histogram": {
                str(key): value for key, value in sorted(Counter(all_lags).items())
            },
            "near_8_multiple_fraction": float(all_near.float().mean()),
            "median_relative_teacher_deviation": float(np.median(all_relative_deviations)),
            "exact_canonical_selected_index_agreement": (
                agreement_numerator / agreement_denominator if agreement_denominator else None
            ),
            "center_direction_eligibility_fraction": (
                center_eligible / center_possible if center_possible else None
            ),
            "per_scale": per_scale,
        },
        gradient.detach(),
        selections,
    )


def _cosine(first: Tensor, second: Tensor) -> float | None:
    first_norm = float(first.norm())
    second_norm = float(second.norm())
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return None
    return float(F.cosine_similarity(first.flatten(), second.flatten(), dim=0))


def _bootstrap_summary(
    values: Sequence[float],
    *,
    statistic: Literal["median", "mean"] = "median",
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "observations": 0,
            statistic: None,
            "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": _BOOTSTRAP_SEED,
            "bootstrap_lower_95": None,
            "bootstrap_upper_95": None,
        }
    if not np.isfinite(array).all():
        raise RuntimeError("bootstrap inputs contain non-finite values")
    point = float(np.median(array) if statistic == "median" else np.mean(array))
    generator = np.random.default_rng(_BOOTSTRAP_SEED)
    bootstrapped = np.empty(_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    chunk = 512
    for start in range(0, _BOOTSTRAP_RESAMPLES, chunk):
        size = min(chunk, _BOOTSTRAP_RESAMPLES - start)
        indices = generator.integers(0, array.size, size=(size, array.size))
        samples = array[indices]
        bootstrapped[start : start + size] = (
            np.median(samples, axis=1) if statistic == "median" else np.mean(samples, axis=1)
        )
    return {
        "observations": int(array.size),
        statistic: point,
        "bootstrap_resamples": _BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": _BOOTSTRAP_SEED,
        "bootstrap_lower_95": float(np.quantile(bootstrapped, 0.025)),
        "bootstrap_upper_95": float(np.quantile(bootstrapped, 0.975)),
    }


def _relative_advantage(candidate: float, canonical: float) -> float:
    return (candidate - canonical) / max(canonical, _EPSILON)


def _aggregate_arm(
    rows: Sequence[Mapping[str, Any]],
    arm_name: str,
) -> dict[str, Any]:
    selected = [row["arms"][arm_name] for row in rows]
    positive = [value for value in selected if value["loss"] is not None]
    scale_keys = sorted(
        {scale for value in positive for scale in value["per_scale"]},
        key=float,
    )
    per_scale: dict[str, Any] = {}
    for scale in scale_keys:
        scale_values = [value["per_scale"][scale] for value in positive]
        pooled_histogram: Counter[int] = Counter()
        for value in scale_values:
            pooled_histogram.update(
                {int(key): int(count) for key, count in value["selected_abs_lag_histogram"].items()}
            )
        per_scale[scale] = {
            "video_loss": _summary([float(value["loss"]) for value in scale_values]),
            "selected_correspondence_total": sum(
                int(value["selected_correspondence_total"]) for value in scale_values
            ),
            "video_near_8_multiple_fraction": _summary(
                [float(value["near_8_multiple_fraction"]) for value in scale_values]
            ),
            "video_median_relative_teacher_deviation": _summary(
                [float(value["median_relative_teacher_deviation"]) for value in scale_values]
            ),
            "pooled_selected_abs_lag_histogram": {
                str(key): count for key, count in sorted(pooled_histogram.items())
            },
        }
    return {
        "positive_confidence_video_total": len(positive),
        "video_loss": _summary([float(value["loss"]) for value in positive]),
        "video_gradient_norm": _summary([float(value["gradient_norm"]) for value in positive]),
        "gradient_zero_fraction": (
            sum(float(value["gradient_norm"]) <= 1e-12 for value in positive) / len(positive)
            if positive
            else None
        ),
        "video_near_8_multiple_fraction": _summary(
            [float(value["near_8_multiple_fraction"]) for value in positive]
        ),
        "video_median_relative_teacher_deviation": _summary(
            [float(value["median_relative_teacher_deviation"]) for value in positive]
        ),
        "video_exact_canonical_selected_index_agreement": _summary(
            [
                float(value["exact_canonical_selected_index_agreement"])
                for value in positive
                if value["exact_canonical_selected_index_agreement"] is not None
            ]
        ),
        "center_direction_eligibility_fraction": (
            float(
                np.mean(
                    [
                        value["center_direction_eligibility_fraction"]
                        for value in positive
                        if value["center_direction_eligibility_fraction"] is not None
                    ]
                )
            )
            if any(value["center_direction_eligibility_fraction"] is not None for value in positive)
            else None
        ),
        "per_scale": per_scale,
    }


def _metric_classification(
    *,
    geometry: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    permutable_mask: Sequence[bool],
    arm_aggregates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    eligible_rows = [
        row
        for row, selected in zip(rows, permutable_mask, strict=True)
        if selected
        and row["arms"]["canonical"]["loss"] is not None
        and row["arms"]["teacher_permutation"]["loss"] is not None
    ]
    r8_values = [
        _relative_advantage(
            float(row["arms"]["constant8"]["loss"]),
            float(row["arms"]["canonical"]["loss"]),
        )
        for row in rows
        if row["arms"]["canonical"]["loss"] is not None
        and row["arms"]["constant8"]["loss"] is not None
    ]
    aperm_values = [
        _relative_advantage(
            float(row["arms"]["teacher_permutation"]["loss"]),
            float(row["arms"]["canonical"]["loss"]),
        )
        for row in eligible_rows
    ]
    gperm_values = [
        float(row["comparisons"]["canonical_vs_teacher_permutation_gradient_cosine"])
        for row in eligible_rows
        if row["comparisons"]["canonical_vs_teacher_permutation_gradient_cosine"] is not None
    ]
    gcenter_values = [
        float(row["comparisons"]["mandatory_center_canonical_vs_permuted_gradient_cosine"])
        for row in eligible_rows
        if row["comparisons"]["mandatory_center_canonical_vs_permuted_gradient_cosine"] is not None
    ]
    gradient_delta_values = [
        float(row["comparisons"]["gperm_current_minus_mandatory_center"])
        for row in eligible_rows
        if row["comparisons"]["gperm_current_minus_mandatory_center"] is not None
    ]
    s8 = _bootstrap_summary(
        [
            float(row["arms"]["canonical"]["near_8_multiple_fraction"])
            for row in rows
            if row["arms"]["canonical"]["loss"] is not None
        ]
    )
    r8 = _bootstrap_summary(r8_values)
    aperm = _bootstrap_summary(aperm_values)
    gperm = _bootstrap_summary(gperm_values)
    gcenter = _bootstrap_summary(gcenter_values)
    gradient_delta = _bootstrap_summary(gradient_delta_values)
    w8 = float(geometry["all_scale_opportunity_fraction"])

    alias_confirmed = (
        w8 >= float(_THRESHOLDS["w8_all_scale_opportunity_fraction_minimum"])
        and float(s8["median"]) >= float(_THRESHOLDS["s8_video_median_minimum"])
        and float(s8["bootstrap_lower_95"]) >= float(_THRESHOLDS["s8_bootstrap_lower_95_minimum"])
        and float(r8["median"]) <= float(_THRESHOLDS["r8_video_median_maximum"])
        and float(r8["bootstrap_upper_95"]) <= float(_THRESHOLDS["r8_bootstrap_upper_95_maximum"])
    )
    loss_identifiable = float(aperm["median"]) >= float(
        _THRESHOLDS["aperm_identifiable_video_median_minimum"]
    ) and float(aperm["bootstrap_lower_95"]) >= float(
        _THRESHOLDS["aperm_identifiable_bootstrap_lower_95_minimum"]
    )
    loss_blind = float(aperm["median"]) <= float(
        _THRESHOLDS["aperm_blind_video_median_maximum"]
    ) and float(aperm["bootstrap_upper_95"]) <= float(
        _THRESHOLDS["aperm_blind_bootstrap_upper_95_maximum"]
    )
    gradient_identifiable = float(gperm["median"]) <= float(
        _THRESHOLDS["gperm_identifiable_video_median_maximum"]
    ) and float(gperm["bootstrap_upper_95"]) <= float(
        _THRESHOLDS["gperm_identifiable_bootstrap_upper_95_maximum"]
    )
    gradient_blind = float(gperm["median"]) >= float(
        _THRESHOLDS["gperm_blind_video_median_minimum"]
    ) and float(gperm["bootstrap_lower_95"]) >= float(
        _THRESHOLDS["gperm_blind_bootstrap_lower_95_minimum"]
    )
    if loss_identifiable and gradient_identifiable:
        current_teacher_state = "identifiable"
    elif loss_blind and gradient_blind:
        current_teacher_state = "teacher_blind"
    else:
        current_teacher_state = "indeterminate"

    center_canonical = arm_aggregates["mandatory_center_canonical"]
    center_permuted = arm_aggregates["mandatory_center_teacher_permutation"]
    center_nonzero = min(
        1.0 - float(center_canonical["gradient_zero_fraction"]),
        1.0 - float(center_permuted["gradient_zero_fraction"]),
    )
    center_viable = (
        float(center_canonical["center_direction_eligibility_fraction"])
        >= float(_THRESHOLDS["center_direction_eligibility_fraction_minimum"])
        and center_nonzero >= float(_THRESHOLDS["center_nonzero_gradient_fraction_minimum"])
        and float(gcenter["median"]) <= float(_THRESHOLDS["gperm_center_video_median_maximum"])
        and float(gcenter["bootstrap_upper_95"])
        <= float(_THRESHOLDS["gperm_center_bootstrap_upper_95_maximum"])
        and float(gradient_delta["median"])
        >= float(_THRESHOLDS["gperm_current_minus_center_video_median_minimum"])
        and float(gradient_delta["bootstrap_lower_95"])
        > float(_THRESHOLDS["gperm_current_minus_center_bootstrap_lower_95_exclusive"])
    )
    gradient_zero_fraction = max(
        float(arm_aggregates["canonical"]["gradient_zero_fraction"]),
        float(arm_aggregates["teacher_permutation"]["gradient_zero_fraction"]),
    )
    paired_fraction = len(eligible_rows) / max(
        int(geometry["positive_confidence_video_total"]),
        1,
    )
    return {
        "w8_all_scale_opportunity_fraction": w8,
        "w8_all_scale_opportunity_bootstrap": _bootstrap_summary(
            [float(row["all_scales_contain_positive_8_multiple"]) for row in geometry["rows"]],
            statistic="mean",
        ),
        "s8_canonical_selected_near_8_multiple": s8,
        "r8_constant8_relative_nll_advantage": r8,
        "aperm_teacher_permutation_relative_nll_advantage": aperm,
        "gperm_current_gradient_cosine": gperm,
        "gperm_mandatory_center_gradient_cosine": gcenter,
        "gperm_current_minus_mandatory_center": gradient_delta,
        "paired_eligible_video_total": len(eligible_rows),
        "paired_eligible_fraction": paired_fraction,
        "gradient_zero_fraction": gradient_zero_fraction,
        "center_nonzero_gradient_fraction": center_nonzero,
        "alias_confirmed": alias_confirmed,
        "current_teacher_state": current_teacher_state,
        "mandatory_center_signal_viable": center_viable,
    }


def _convergence_decision(
    budget32: Mapping[str, Any],
    budget64: Mapping[str, Any],
) -> dict[str, Any]:
    differences = {
        "w8": abs(
            float(budget32["w8_all_scale_opportunity_fraction"])
            - float(budget64["w8_all_scale_opportunity_fraction"])
        ),
        "s8": abs(
            float(budget32["s8_canonical_selected_near_8_multiple"]["median"])
            - float(budget64["s8_canonical_selected_near_8_multiple"]["median"])
        ),
        "r8": abs(
            float(budget32["r8_constant8_relative_nll_advantage"]["median"])
            - float(budget64["r8_constant8_relative_nll_advantage"]["median"])
        ),
        "aperm": abs(
            float(budget32["aperm_teacher_permutation_relative_nll_advantage"]["median"])
            - float(budget64["aperm_teacher_permutation_relative_nll_advantage"]["median"])
        ),
        "gperm_current": abs(
            float(budget32["gperm_current_gradient_cosine"]["median"])
            - float(budget64["gperm_current_gradient_cosine"]["median"])
        ),
        "gperm_center": abs(
            float(budget32["gperm_mandatory_center_gradient_cosine"]["median"])
            - float(budget64["gperm_mandatory_center_gradient_cosine"]["median"])
        ),
    }
    classification_fields = (
        "alias_confirmed",
        "current_teacher_state",
        "mandatory_center_signal_viable",
    )
    classification_unchanged = all(
        budget32[field] == budget64[field] for field in classification_fields
    )
    criteria = {
        "w8_absolute_difference": differences["w8"]
        <= float(_THRESHOLDS["convergence_w8_absolute_difference_maximum"]),
        "s8_absolute_difference": differences["s8"]
        <= float(_THRESHOLDS["convergence_s8_absolute_difference_maximum"]),
        "r8_relative_nll_advantage_difference": differences["r8"]
        <= float(_THRESHOLDS["convergence_relative_nll_advantage_difference_maximum"]),
        "aperm_relative_nll_advantage_difference": differences["aperm"]
        <= float(_THRESHOLDS["convergence_relative_nll_advantage_difference_maximum"]),
        "gperm_current_gradient_cosine_difference": differences["gperm_current"]
        <= float(_THRESHOLDS["convergence_gradient_cosine_difference_maximum"]),
        "gperm_center_gradient_cosine_difference": differences["gperm_center"]
        <= float(_THRESHOLDS["convergence_gradient_cosine_difference_maximum"]),
        "classification_unchanged": classification_unchanged,
    }
    return {
        "differences": differences,
        "classification_fields": list(classification_fields),
        "classification_unchanged": classification_unchanged,
        "criteria": criteria,
        "pass": all(criteria.values()),
        "failure_disposition": (
            "If convergence fails, this artifact remains valid but authorizes "
            "nothing; an all-anchor train337 rerun may be preregistered separately."
        ),
    }


def _operational_decision(
    *,
    train_video_total: int,
    positive_confidence_total: int,
    permutable_total: int,
    budget_results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    paired_minimum = min(
        float(result["classification"]["paired_eligible_fraction"])
        for result in budget_results.values()
    )
    gradient_zero_maximum = max(
        float(result["classification"]["gradient_zero_fraction"])
        for result in budget_results.values()
    )
    criteria = {
        "exact_train337": train_video_total == _EXPECTED_TRAINING_VIDEOS,
        "exact_positive_confidence_321": (
            positive_confidence_total == _EXPECTED_POSITIVE_CONFIDENCE_VIDEOS
        ),
        "permutable_positive_confidence_videos": permutable_total
        >= int(_THRESHOLDS["permutable_positive_confidence_videos_minimum"]),
        "paired_eligible_fraction": paired_minimum
        >= float(_THRESHOLDS["paired_eligible_fraction_minimum"]),
        "gradient_zero_fraction": gradient_zero_maximum
        <= float(_THRESHOLDS["gradient_zero_fraction_maximum"]),
        "finite_statistics": True,
        "evaluation_mode_only": True,
        "training_steps_zero": True,
        "checkpoint_and_model_state_unchanged": True,
        "development_or_test_inputs_absent": True,
    }
    return {
        "criteria": criteria,
        "paired_eligible_fraction_minimum_observed": paired_minimum,
        "gradient_zero_fraction_maximum_observed": gradient_zero_maximum,
        "pass": all(criteria.values()),
    }


def _final_decision(
    *,
    operational: Mapping[str, Any],
    convergence: Mapping[str, Any],
    primary: Mapping[str, Any],
) -> dict[str, Any]:
    scientific_branch = bool(primary["mandatory_center_signal_viable"]) and (
        bool(primary["alias_confirmed"])
        or (
            not bool(primary["alias_confirmed"])
            and primary["current_teacher_state"] == "teacher_blind"
        )
    )
    authorized = bool(operational["pass"]) and bool(convergence["pass"]) and scientific_branch
    return {
        "truth_table": (
            "Authorize only when operational and 32/64 convergence pass, "
            "mandatory-center signal is viable, and either alias is confirmed "
            "or alias is not confirmed while the current selector is teacher-blind."
        ),
        "scientific_branch_pass": scientific_branch,
        "v16_short_continuation_authorized": authorized,
        "authorized_scope": (
            "mandatory-center, train337-only, seed2026, at most 10 encoder epochs"
            if authorized
            else None
        ),
        "sshead_training_authorized": False,
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }


def _model_state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _physical_contrastive_groups(
    video_ids: Sequence[str],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...], dict[str, Any]]:
    """Freeze physical groups of 32 while giving every train video one anchor turn."""

    if len(video_ids) != _EXPECTED_TRAINING_VIDEOS:
        raise ValueError("physical grouping requires exactly 337 video IDs")
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("physical grouping requires unique video IDs")
    groups: list[tuple[int, ...]] = []
    group_by_anchor = [-1] * len(video_ids)
    full_anchor_end = (len(video_ids) // 32) * 32
    for start in range(0, full_anchor_end, 32):
        members = tuple(range(start, start + 32))
        group_index = len(groups)
        groups.append(members)
        for index in members:
            group_by_anchor[index] = group_index
    final_anchors = tuple(range(full_anchor_end, len(video_ids)))
    final_context = tuple(range(0, 32 - len(final_anchors)))
    final_members = final_anchors + final_context
    if len(final_members) != 32:
        raise RuntimeError("final physical group is not exactly 32 videos")
    final_group_index = len(groups)
    groups.append(final_members)
    for index in final_anchors:
        group_by_anchor[index] = final_group_index
    if any(index < 0 for index in group_by_anchor):
        raise RuntimeError("a train337 video has no physical anchor group")
    if sorted(
        index
        for group_index, group in enumerate(groups)
        for index in group
        if group_by_anchor[index] == group_index
    ) != list(range(len(video_ids))):
        raise RuntimeError("physical grouping does not assign every anchor exactly once")
    structured_ids = [[str(video_ids[index]) for index in group] for group in groups]
    audit = {
        "algorithm": (
            "Checkpoint-bound order is partitioned into ten disjoint physical "
            "groups of 32. The final 17 videos are the anchor videos in group "
            "eleven and the first 15 videos are fixed negative context only. "
            "Every one of 337 videos is an anchor in exactly one group."
        ),
        "physical_group_total": len(groups),
        "physical_group_sizes": [len(group) for group in groups],
        "anchor_assignment_total": len(video_ids),
        "final_group_anchor_video_total": len(final_anchors),
        "final_group_context_video_total": len(final_context),
        "grouping_sha256": sha256_json(structured_ids),
        "final_anchor_ids_sha256": _identifier_commitment(
            tuple(video_ids[index] for index in final_anchors)
        ),
        "final_context_ids_sha256": _identifier_commitment(
            tuple(video_ids[index] for index in final_context)
        ),
    }
    return tuple(groups), tuple(group_by_anchor), audit


def _measure_budgets(
    encoded: _EncodedTrain337,
    *,
    config: PAMSConfig,
    bank: VideoPrototypeBank,
    permuted_periods: Tensor,
    permuted_confidences: Tensor,
    permutable: Tensor,
    device: torch.device,
    geometry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    embeddings = encoded.embeddings.to(device)
    valid = encoded.valid_mask.to(device)
    physical_groups, group_by_anchor, grouping_audit = _physical_contrastive_groups(
        encoded.video_ids
    )
    bank_features = F.normalize(bank.features.to(device), dim=-1, eps=1e-12)
    bank_clusters = bank.cluster_labels.to(device)
    bank_assignments = bank.assignments

    arm_periods = {
        "canonical": encoded.projected_periods,
        "teacher_permutation": permuted_periods,
        "half_teacher": encoded.projected_periods.div(2).clamp(
            config.period.minimum,
            config.period.maximum,
        ),
        "third_teacher": encoded.projected_periods.div(3).clamp(
            config.period.minimum,
            config.period.maximum,
        ),
        "constant8": torch.full_like(
            encoded.projected_periods,
            float(_HARMONIC_FRAMES),
        ),
        "mandatory_center_canonical": encoded.projected_periods,
        "mandatory_center_teacher_permutation": permuted_periods,
    }
    arm_confidences = {
        name: (
            permuted_confidences if "teacher_permutation" in name else encoded.projected_confidences
        )
        for name in arm_periods
    }
    mandatory = {
        "mandatory_center_canonical",
        "mandatory_center_teacher_permutation",
    }
    rows_by_budget: dict[int, list[dict[str, Any]]] = {budget: [] for budget in _ANCHOR_BUDGETS}

    active_group_index = -1
    flat = torch.empty(0, config.model.embedding_dim, device=device)
    flat_owner = torch.empty(0, dtype=torch.long, device=device)
    for video_index, video_id in enumerate(encoded.video_ids):
        group_index = group_by_anchor[video_index]
        if group_index != active_group_index:
            members = physical_groups[group_index]
            group_rows: list[Tensor] = []
            group_owners: list[Tensor] = []
            for member in members:
                selected = valid[member]
                group_rows.append(embeddings[member][selected])
                group_owners.append(
                    torch.full(
                        (int(selected.sum()),),
                        member,
                        dtype=torch.long,
                        device=device,
                    )
                )
            flat = F.normalize(torch.cat(group_rows), dim=-1, eps=1e-12)
            flat_owner = torch.cat(group_owners)
            active_group_index = group_index
        confidence = float(encoded.projected_confidences[video_index])
        anchor_cpu = _midpoint_quantile_anchor_indices(
            encoded.valid_mask[video_index],
        )
        row_base = {
            "video_id_sha256": _identifier_commitment((video_id,)),
            "valid_frame_total": int(encoded.valid_mask[video_index].sum()),
            "positive_teacher_confidence": confidence > 0.0,
            "teacher_permutable": bool(permutable[video_index]),
            "anchor64_total": int(anchor_cpu.numel()),
            "anchor32_total": int(_budget_schedule_positions(int(anchor_cpu.numel()), 32).numel()),
            "anchor16_audit_total": int(
                _budget_schedule_positions(int(anchor_cpu.numel()), 16).numel()
            ),
        }
        if confidence <= 0.0:
            for budget in _ANCHOR_BUDGETS:
                rows_by_budget[budget].append(
                    {
                        **row_base,
                        "arms": {},
                        "comparisons": {},
                    }
                )
            continue

        video_embeddings = embeddings[video_index]
        video_valid = valid[video_index]
        anchors = anchor_cpu.to(device)
        denominator, denominator_grad, normalized_query = _denominator_and_row_gradients(
            video_embeddings,
            video_valid,
            anchors,
            flat_embeddings=flat,
            flat_owner=flat_owner,
            owner_index=video_index,
            bank_features=bank_features,
            bank_clusters=bank_clusters,
            owner_cluster=bank_assignments[video_id],
            temperature=config.loss.temperature,
        )
        for budget in _ANCHOR_BUDGETS:
            measurements: dict[str, Any] = {}
            gradients: dict[str, Tensor] = {}
            canonical_selections: dict[str, Tensor] | None = None
            for arm_name in (
                "canonical",
                "teacher_permutation",
                "half_teacher",
                "third_teacher",
                "constant8",
                "mandatory_center_canonical",
                "mandatory_center_teacher_permutation",
            ):
                measurement, gradient, selections = _video_arm_measurement(
                    video_embeddings,
                    video_valid,
                    anchors,
                    period=float(arm_periods[arm_name][video_index]),
                    confidence=float(arm_confidences[arm_name][video_index]),
                    scales=config.loss.scales,
                    temperature=config.loss.temperature,
                    denominator=denominator,
                    denominator_row_gradients=denominator_grad,
                    normalized_query=normalized_query,
                    budget=budget,
                    mandatory_center=arm_name in mandatory,
                    canonical_selections=canonical_selections,
                )
                measurements[arm_name] = measurement
                gradients[arm_name] = gradient
                if arm_name == "canonical":
                    canonical_selections = selections
            gperm = _cosine(
                gradients["canonical"],
                gradients["teacher_permutation"],
            )
            gcenter = _cosine(
                gradients["mandatory_center_canonical"],
                gradients["mandatory_center_teacher_permutation"],
            )
            comparisons = {
                "canonical_vs_teacher_permutation_gradient_cosine": gperm,
                "mandatory_center_canonical_vs_permuted_gradient_cosine": gcenter,
                "gperm_current_minus_mandatory_center": (
                    gperm - gcenter if gperm is not None and gcenter is not None else None
                ),
            }
            rows_by_budget[budget].append(
                {
                    **row_base,
                    "arms": measurements,
                    "comparisons": comparisons,
                }
            )
        del denominator, denominator_grad, normalized_query

    results: dict[str, Any] = {}
    mask_values = [bool(value) for value in permutable.tolist()]
    for budget in _ANCHOR_BUDGETS:
        rows = rows_by_budget[budget]
        positive_rows = [row for row in rows if row["positive_teacher_confidence"]]
        arm_names = tuple(positive_rows[0]["arms"])
        aggregates = {arm_name: _aggregate_arm(positive_rows, arm_name) for arm_name in arm_names}
        classification = _metric_classification(
            geometry=geometry,
            rows=positive_rows,
            permutable_mask=[
                selected
                for selected, confidence in zip(
                    mask_values,
                    encoded.projected_confidences.tolist(),
                    strict=True,
                )
                if float(confidence) > 0.0
            ],
            arm_aggregates=aggregates,
        )
        results[str(budget)] = {
            "anchor_budget": budget,
            "anchor_algorithm": (
                "64 midpoint-quantile valid-frame anchors; budget 32 is the "
                "nested even-k subset. Every fourth k is also recorded as an "
                "unused 16-anchor audit subset."
            ),
            "loss_algorithm": (
                "Each video is averaged uniformly over eligible "
                "scale x direction x anchor cells. The common denominator is "
                "all valid nonself frames from the same video, all valid frames "
                "from the other 31 videos in its frozen physical group of 32, "
                "and exactly four hardest cross-cluster checkpoint prototypes. "
                "Adjacent t+/-1 positives are excluded. Gradients are query-local "
                "with every candidate embedding detached, making one comparable "
                "gradient vector per video."
            ),
            "arms": aggregates,
            "classification": classification,
            "video_rows": rows,
        }
    return results, grouping_audit


def run_teacher_identifiability_probe(
    encoder_checkpoint_path: str | Path,
    encoder_progress_path: str | Path,
    config_path: str | Path,
    pose_cache_dir: str | Path,
    source_receipt_path: str | Path,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate the exact terminal v15 encoder on checkpoint-bound train337."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or batch_size > 32
    ):
        raise ValueError("batch_size must be an integer in [1, 32]")
    paths = {
        "encoder_checkpoint": Path(encoder_checkpoint_path),
        "encoder_progress": Path(encoder_progress_path),
        "config": Path(config_path),
        "source_export_receipt": Path(source_receipt_path),
        "teacher_identifiability_probe_runner": Path(__file__),
        "v15_terminal_gate_dependency": Path(_v15.__file__),
        "v14_counterfactual_dependency": Path(_v14.__file__),
        "losses_dependency": Path(_losses.__file__),
    }
    identities = _v15._input_identities(paths)
    exact_terminal_identities = {
        "encoder_checkpoint": _EXPECTED_ENCODER_CHECKPOINT_SHA256,
        "encoder_progress": _EXPECTED_ENCODER_PROGRESS_SHA256,
    }
    for name, expected_sha256 in exact_terminal_identities.items():
        if identities[name][0] != expected_sha256:
            raise ValueError(
                "teacher-identifiability probe requires the exact prior v15 "
                f"terminal {name.replace('_', ' ')} SHA-256"
            )
    _v14._validate_source_receipt_argument(
        paths["source_export_receipt"],
        receipt_sha256=identities["source_export_receipt"][0],
    )
    gate_code_source_git_sha = _v14._gate_code_source_revision()
    config = load_config(paths["config"])
    _v15._validate_exact_v15_config(
        config,
        config_sha256=identities["config"][0],
    )
    stage, provenance = _peek_checkpoint(paths["encoder_checkpoint"], config)
    if stage != "encoder":
        raise ValueError("teacher-identifiability probe requires an encoder checkpoint")
    if len(provenance.training_video_ids) != _EXPECTED_TRAINING_VIDEOS:
        raise ValueError("teacher-identifiability probe requires exactly train337")
    if provenance.upstream_encoder_checkpoint_sha256 is not None:
        raise ValueError("encoder provenance unexpectedly names an upstream checkpoint")
    if provenance.container_image_id is None or provenance.container_environment_sha256 is None:
        raise ValueError("formal container provenance is required")
    validate_terminal_checkpoint(
        paths["encoder_checkpoint"],
        config,
        expected_stage="encoder",
        expected_provenance=provenance,
        progress_path=paths["encoder_progress"],
    )
    algorithm_source_git_sha = clean_git_revision(Path.cwd())
    runtime_container = _v14._validate_runtime_binding(
        provenance,
        source_git_sha=algorithm_source_git_sha,
    )
    resolved_device = _device(device)
    sequences, selected_receipts, full_receipts, selected_ids = _load_training_poses(
        pose_cache_dir=Path(pose_cache_dir),
        provenance=provenance,
        config=config,
        sample_size=0,
        seed=config.seed,
    )
    if len(sequences) != _EXPECTED_TRAINING_VIDEOS:
        raise RuntimeError("checkpoint-bound pose loader did not return train337")
    selected_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=selected_receipts,
    )
    full_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=full_receipts,
    )
    if selected_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("full train337 selection and pose snapshots differ")
    if full_snapshot.fingerprint != provenance.pose_cache_set_sha256:
        raise ValueError("train337 pose-cache set differs from checkpoint provenance")

    model = load_model_checkpoint(
        paths["encoder_checkpoint"],
        config,
        device=resolved_device,
        expected_stage="encoder",
        expected_provenance=provenance,
    ).eval()
    model_state_before = _model_state_sha256(model)
    encoded = _encode_train337(
        model,
        sequences,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    if encoded.video_ids != tuple(selected_ids):
        raise RuntimeError("encoded train337 order differs from checkpoint-bound order")
    valid_counts = encoded.valid_mask.sum(dim=1).to(dtype=torch.long)
    (
        permuted_periods,
        permuted_confidences,
        permutable,
        permutation_audit,
    ) = _teacher_permutation(
        encoded.projected_periods,
        encoded.projected_confidences,
        valid_counts,
        encoded.video_ids,
    )
    geometry = _geometry_opportunity(
        encoded.projected_periods,
        encoded.projected_confidences,
        encoded.video_ids,
        scales=config.loss.scales,
        time=config.data.frames,
    )
    bank = _load_prototype_bank(paths["encoder_checkpoint"])
    measurements, physical_batching = _measure_budgets(
        encoded,
        config=config,
        bank=bank,
        permuted_periods=permuted_periods,
        permuted_confidences=permuted_confidences,
        permutable=permutable,
        device=resolved_device,
        geometry=geometry,
    )
    convergence = _convergence_decision(
        measurements["32"]["classification"],
        measurements["64"]["classification"],
    )
    positive_total = int((encoded.projected_confidences > 0).sum())
    operational = _operational_decision(
        train_video_total=len(encoded.video_ids),
        positive_confidence_total=positive_total,
        permutable_total=int(permutable.sum()),
        budget_results=measurements,
    )
    decision = _final_decision(
        operational=operational,
        convergence=convergence,
        primary=measurements["64"]["classification"],
    )
    model_state_after = _model_state_sha256(model)
    if model_state_after != model_state_before:
        raise RuntimeError("model state changed during read-only diagnostic")

    _, _, final_receipts, final_ids = _load_training_poses(
        pose_cache_dir=Path(pose_cache_dir),
        provenance=provenance,
        config=config,
        sample_size=2,
        seed=config.seed,
    )
    final_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=final_receipts,
    )
    if final_ids != selected_ids[:2]:
        raise RuntimeError("train337 pose selection changed during diagnostic")
    if final_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("train337 pose cache changed during diagnostic")
    _v14._require_unchanged(paths, identities)
    if clean_git_revision(Path.cwd()) != algorithm_source_git_sha:
        raise RuntimeError("checkpoint algorithm source changed during diagnostic")

    hardware = hardware_fingerprint()
    runtime = _v14._runtime_provenance(
        runner_sha256=identities["teacher_identifiability_probe_runner"][0],
        device=resolved_device,
    )
    expected_container = {
        "PAMS_CONTAINER_IMAGE_ID": runtime_container["image_id"],
        "PAMS_CONTAINER_ENVIRONMENT_SHA256": runtime_container["environment_sha256"],
        "PAMS_CONTAINER_SOURCE_REVISION": runtime_container["source_revision"],
    }
    if runtime["container"] != expected_container:
        raise RuntimeError("runtime provenance changed while it was recorded")
    runtime["algorithm_source_git_sha"] = algorithm_source_git_sha
    runtime["gate_code_source_git_sha"] = gate_code_source_git_sha
    code_names = (
        "teacher_identifiability_probe_runner",
        "v15_terminal_gate_dependency",
        "v14_counterfactual_dependency",
        "losses_dependency",
    )
    code_files_sha256 = {name: identities[name][0] for name in code_names}
    path_context = {
        "governing_role": (
            "The projected path supplies the canonical teacher and the post-PE "
            "embeddings supply similarities, losses, and gradients. Their global "
            "period distributions are recorded only as path context."
        ),
        "canonical_post_pe": _period_distribution(
            encoded.post_pe_periods,
            encoded.post_pe_confidences,
        ),
        "projected_pre_pe_teacher": _period_distribution(
            encoded.projected_periods,
            encoded.projected_confidences,
        ),
        "projected_teacher_vs_canonical_post_pe": _cross_period_summary(
            encoded.projected_periods,
            encoded.projected_confidences,
            encoded.post_pe_periods,
            encoded.post_pe_confidences,
        ),
    }
    payload = {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "status": (
            "v16_short_continuation_authorized"
            if decision["v16_short_continuation_authorized"]
            else "v16_short_continuation_rejected"
        ),
        "classification": _CLASSIFICATION,
        "disclosed_by_pams_authors": False,
        "eligible_for_paper_table": False,
        "protocol": config.protocol,
        "seed": config.seed,
        "thresholds_frozen_before_checkpoint_probe": dict(_THRESHOLDS),
        "label_firewall": {
            "accepted_scientific_inputs": [
                "exact_v15_experiment_config",
                "terminal_v15_encoder_checkpoint_and_progress",
                "checkpoint_bound_train337_pose_cache",
                "exact_checkpoint_algorithm_source_export_receipt",
            ],
            "dataset_manifest_argument_supported": False,
            "development_identity_media_pose_or_target_argument_supported": False,
            "sealed_test_identity_media_pose_or_target_argument_supported": False,
            "action_class_argument_supported": False,
            "repetition_count_argument_supported": False,
            "external_label_fields_accessed": [],
            "training_interface_supported": False,
        },
        "inputs": {
            **{f"{name}_sha256": identity[0] for name, identity in identities.items()},
            **{f"{name}_bytes": identity[1] for name, identity in identities.items()},
            "config_fingerprint": config.fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
            "encoder_provenance": provenance.to_dict(),
            "train337_video_total": len(selected_ids),
            "train337_video_ids_sha256": _identifier_commitment(selected_ids),
            "train337_pose_cache_set_sha256": full_snapshot.fingerprint,
            "positive_confidence_video_total": positive_total,
            "checkpoint_algorithm_source_git_sha": algorithm_source_git_sha,
            "gate_code_source_git_sha": gate_code_source_git_sha,
            "code_files_sha256": code_files_sha256,
            "code_files_sha256_commitment": sha256_json(code_files_sha256),
            "read_only_post_run_identity_verified": True,
        },
        "teacher_permutation": permutation_audit,
        "physical_contrastive_batching": physical_batching,
        "geometry_only_w8": geometry,
        "anchor_budget_measurements": measurements,
        "frozen_path_context": path_context,
        "convergence": convergence,
        "operational_gate": operational,
        "decision": decision,
        "scientific_caveats": [
            (
                "The selector, interventions, loss, gradients, thresholds, and "
                "authorization truth table are independently inferred and are "
                "not disclosed by the PAMS authors."
            ),
            (
                "Query-local gradients detach all candidate embeddings so that "
                "the statistical unit is one video; they are causal diagnostics, "
                "not a replacement for a full training gradient."
            ),
            (
                "Any pass authorizes only a separately audited ten-epoch "
                "mandatory-center train337 continuation. SSHead, dev84, and "
                "sealed test105 remain unauthorized."
            ),
        ],
        "hardware": hardware,
        "hardware_sha256": sha256_json(hardware),
        "runtime": runtime,
        "runtime_sha256": sha256_json(runtime),
        "read_only_verification": {
            "all_file_inputs_unchanged": True,
            "train337_pose_cache_set_sha256_unchanged": True,
            "checkpoint_algorithm_source_git_sha_unchanged": True,
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after": model_state_after,
            "model_or_optimizer_state_updated": False,
            "training_steps_executed": 0,
            "pose_cache_write_operations": 0,
        },
    }
    del model, encoded, bank
    gc.collect()
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def _receipt_path(output: Path) -> Path:
    return Path(str(output) + ".receipt.json")


def _write_artifact_and_receipt(
    output: Path,
    payload: Mapping[str, Any],
) -> tuple[Path, str]:
    receipt_path = _receipt_path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite probe artifact: {output}")
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite probe receipt: {receipt_path}")
    artifact = _v14._encoded_json(payload)
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    _v14._write_new_regular_file(output, artifact)
    receipt = {
        "schema_version": 1,
        "artifact_type": _RECEIPT_TYPE,
        "artifact_locator": output.name,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": len(artifact),
        "artifact_status": payload["status"],
        "v16_short_continuation_authorized": payload["decision"][
            "v16_short_continuation_authorized"
        ],
        "sshead_training_authorized": False,
        "dev84_prediction_authorized": False,
        "test105_evaluation_authorized": False,
        "encoder_checkpoint_sha256": payload["inputs"]["encoder_checkpoint_sha256"],
        "encoder_progress_sha256": payload["inputs"]["encoder_progress_sha256"],
        "config_sha256": payload["inputs"]["config_sha256"],
        "source_export_receipt_sha256": payload["inputs"]["source_export_receipt_sha256"],
        "train337_pose_cache_set_sha256": payload["inputs"]["train337_pose_cache_set_sha256"],
        "checkpoint_algorithm_source_git_sha": payload["inputs"][
            "checkpoint_algorithm_source_git_sha"
        ],
        "gate_code_source_git_sha": payload["inputs"]["gate_code_source_git_sha"],
        "code_files_sha256_commitment": payload["inputs"]["code_files_sha256_commitment"],
        "hardware_sha256": payload["hardware_sha256"],
        "runtime_sha256": payload["runtime_sha256"],
    }
    _v14._write_new_regular_file(receipt_path, _v14._encoded_json(receipt))
    return receipt_path, artifact_sha256


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-progress", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pose-cache-dir", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    receipt_path = _receipt_path(arguments.output)
    if arguments.output.exists() or receipt_path.exists():
        raise FileExistsError("probe output and receipt destinations must both be new")
    payload = run_teacher_identifiability_probe(
        arguments.encoder_checkpoint,
        arguments.encoder_progress,
        arguments.config,
        arguments.pose_cache_dir,
        arguments.source_receipt,
        device=arguments.device,
        batch_size=arguments.batch_size,
    )
    receipt_path, artifact_sha256 = _write_artifact_and_receipt(
        arguments.output,
        payload,
    )
    authorized = payload["decision"]["v16_short_continuation_authorized"]
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "receipt": receipt_path.name,
                "status": payload["status"],
                "v16_short_continuation_authorized": authorized,
                "sshead_training_authorized": False,
                "dev84_prediction_authorized": False,
                "test105_evaluation_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0 if authorized else 3


if __name__ == "__main__":
    raise SystemExit(main())
