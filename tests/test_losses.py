import math

import pytest
import torch

import pams.losses as losses_module
from pams.losses import (
    PAMSTCCLoss,
    ReferenceRelativeSSHeadLoss,
    SSHeadLoss,
    _reference_anchor_index,
    build_reference_relative_signal,
    masked_zscore,
    periodic_correspondence_indices,
)


def _periodic_embeddings(batch: int = 2, time: int = 24, period: int = 6) -> torch.Tensor:
    positions = torch.arange(time, dtype=torch.float32)
    features = torch.stack(
        (
            torch.sin(2.0 * math.pi * positions / period),
            torch.cos(2.0 * math.pi * positions / period),
            torch.sin(4.0 * math.pi * positions / period),
            torch.cos(4.0 * math.pi * positions / period),
        ),
        dim=-1,
    )
    return features.unsqueeze(0).repeat(batch, 1, 1)


def _assert_tcc_outputs_identical(
    left: losses_module.TCCLossOutput,
    right: losses_module.TCCLossOutput,
) -> None:
    assert torch.equal(left.total, right.total)
    assert len(left.scale_losses) == len(right.scale_losses)
    assert all(
        torch.equal(left_loss, right_loss)
        for left_loss, right_loss in zip(left.scale_losses, right.scale_losses, strict=True)
    )
    assert left.valid_anchor_counts == right.valid_anchor_counts
    assert left.cross_cluster_requested_counts == right.cross_cluster_requested_counts
    assert left.cross_cluster_actual_counts == right.cross_cluster_actual_counts
    assert left.cross_cluster_shortfall_counts == right.cross_cluster_shortfall_counts
    for left_values, right_values in zip(
        (
            left.cross_cluster_requested_per_anchor,
            left.cross_cluster_actual_per_anchor,
            left.cross_cluster_shortfall_per_anchor,
        ),
        (
            right.cross_cluster_requested_per_anchor,
            right.cross_cluster_actual_per_anchor,
            right.cross_cluster_shortfall_per_anchor,
        ),
        strict=True,
    ):
        assert all(
            torch.equal(left_value, right_value)
            for left_value, right_value in zip(left_values, right_values, strict=True)
        )


def test_periodic_correspondence_selects_exact_cycle_match() -> None:
    embeddings = _periodic_embeddings(batch=1)
    indices = periodic_correspondence_indices(
        embeddings,
        torch.tensor([6]),
        scale=1.0,
    )
    assert indices.shape == (1, 24, 2)
    assert indices[0, 10].tolist() == [4, 16]
    assert indices[0, 2, 0] == -1
    assert indices[0, 2, 1] == 8


def test_shortest_period_uses_three_distinct_scale_windows() -> None:
    embeddings = torch.zeros(1, 20, 4)
    embeddings[..., 1] = 1.0
    embeddings[0, 10] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    embeddings[0, 5] = torch.tensor([0.3, 1.0, 0.0, 0.0])
    embeddings[0, 4] = torch.tensor([0.6, 1.0, 0.0, 0.0])
    embeddings[0, 3] = torch.tensor([0.9, 1.0, 0.0, 0.0])

    selected = [
        int(
            periodic_correspondence_indices(
                embeddings,
                torch.tensor([4]),
                scale=scale,
            )[0, 10, 0]
        )
        for scale in (0.5, 1.0, 1.5)
    ]

    assert selected == [5, 4, 3]


def test_multiscale_tcc_is_finite_differentiable_and_mask_safe() -> None:
    embeddings = _periodic_embeddings().requires_grad_()
    valid = torch.ones(2, 24, dtype=torch.bool)
    valid[0, -3:] = False
    objective = PAMSTCCLoss(scales=(0.5, 1.0, 1.5), temperature=0.1)
    details = objective.compute(
        embeddings,
        torch.tensor([6, 6]),
        valid,
        cluster_labels=torch.tensor([0, 1]),
    )
    assert torch.isfinite(details.total)
    assert len(details.scale_losses) == 3
    assert all(count > 0 for count in details.valid_anchor_counts)
    details.total.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()

    missing = torch.zeros(2, 24, dtype=torch.bool)
    empty_loss = objective(embeddings.detach(), torch.tensor([6, 6]), missing)
    assert empty_loss == 0


def test_other_video_negatives_are_valid_frames_not_pooled_prototypes() -> None:
    embeddings = torch.tensor(
        [
            [[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]],
            [[1.0, 1.0], [-1.0, 1.0], [20.0, 0.0]],
        ],
        requires_grad=True,
    )
    valid = torch.tensor([[True, True, True], [True, True, False]])
    objective = PAMSTCCLoss(scales=(1.0,), temperature=1.0)
    details = objective.compute(
        embeddings,
        torch.tensor([2.0, 2.0]),
        valid,
        period_confidence=torch.zeros(2),
        use_cross_cluster_negatives=False,
    )

    normalized = torch.nn.functional.normalize(embeddings, dim=-1, eps=1e-12)
    within_logits = torch.einsum("btd,bsd->bts", normalized, normalized)
    identity = torch.eye(3, dtype=torch.bool).unsqueeze(0)
    within_mask = valid.unsqueeze(1) & valid.unsqueeze(2) & ~identity

    frame_logits = torch.einsum("btd,csd->btcs", normalized, normalized).flatten(2)
    video_indices = torch.arange(2)
    candidate_video_indices = video_indices.repeat_interleave(3)
    frame_mask = (
        (
            candidate_video_indices.view(1, 1, 6)
            != video_indices.view(2, 1, 1)
        )
        & valid.unsqueeze(-1)
        & valid.reshape(1, 1, 6)
    )

    positive_mask = torch.zeros((2, 3, 3), dtype=torch.bool)
    adjacent = torch.arange(2)
    positive_mask[:, adjacent, adjacent + 1] = True
    positive_mask[:, adjacent + 1, adjacent] = True
    positive_mask &= within_mask
    positive_counts = positive_mask.sum(dim=-1)
    valid_anchors = valid & (positive_counts > 0)
    mean_positive = within_logits.masked_fill(~positive_mask, 0.0).sum(
        dim=-1
    ) / positive_counts.clamp_min(1)
    full_frame_denominator = torch.logsumexp(
        torch.cat(
            (
                within_logits.masked_fill(~within_mask, float("-inf")),
                frame_logits.masked_fill(~frame_mask, float("-inf")),
            ),
            dim=-1,
        ),
        dim=-1,
    )
    expected = (full_frame_denominator - mean_positive)[valid_anchors].mean()
    assert torch.allclose(details.total, expected)

    valid_counts = valid.sum(dim=1, keepdim=True).clamp_min(1)
    prototypes = (normalized * valid.unsqueeze(-1)).sum(dim=1) / valid_counts
    prototypes = torch.nn.functional.normalize(prototypes, dim=-1, eps=1e-12)
    prototype_logits = torch.einsum("btd,cd->btc", normalized, prototypes)
    prototype_mask = (
        (video_indices.view(1, 1, 2) != video_indices.view(2, 1, 1))
        & valid.unsqueeze(-1)
        & valid.any(dim=1).view(1, 1, 2)
    )
    prototype_denominator = torch.logsumexp(
        torch.cat(
            (
                within_logits.masked_fill(~within_mask, float("-inf")),
                prototype_logits.masked_fill(~prototype_mask, float("-inf")),
            ),
            dim=-1,
        ),
        dim=-1,
    )
    old_pooled_loss = (prototype_denominator - mean_positive)[valid_anchors].mean()
    assert not torch.allclose(details.total, old_pooled_loss)

    details.total.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert torch.count_nonzero(embeddings.grad[1, 2]) == 0


def test_cross_scale_positive_exclusion_is_default_off_and_bitwise_equivalent() -> None:
    base = _periodic_embeddings()
    default_embeddings = base.clone().requires_grad_()
    explicit_embeddings = base.clone().requires_grad_()
    arguments = {
        "periods": torch.tensor([6, 6]),
        "cluster_labels": torch.tensor([0, 1]),
        "use_cross_cluster_negatives": False,
    }

    default = PAMSTCCLoss(scales=(0.5, 1.0, 1.5)).compute(
        default_embeddings,
        **arguments,
    )
    explicit_disabled = PAMSTCCLoss(
        scales=(0.5, 1.0, 1.5),
        exclude_other_scale_positives_from_denominator=False,
    ).compute(
        explicit_embeddings,
        **arguments,
    )

    _assert_tcc_outputs_identical(default, explicit_disabled)
    default.total.backward()
    explicit_disabled.total.backward()
    assert torch.equal(default_embeddings.grad, explicit_embeddings.grad)


def test_cross_scale_positive_exclusion_is_a_noop_for_one_scale() -> None:
    base = _periodic_embeddings()
    default_embeddings = base.clone().requires_grad_()
    enabled_embeddings = base.clone().requires_grad_()

    default = PAMSTCCLoss(scales=(1.0,)).compute(
        default_embeddings,
        torch.tensor([6, 6]),
        use_cross_cluster_negatives=False,
    )
    enabled = PAMSTCCLoss(
        scales=(1.0,),
        exclude_other_scale_positives_from_denominator=True,
    ).compute(
        enabled_embeddings,
        torch.tensor([6, 6]),
        use_cross_cluster_negatives=False,
    )

    _assert_tcc_outputs_identical(default, enabled)
    default.total.backward()
    enabled.total.backward()
    assert torch.equal(default_embeddings.grad, enabled_embeddings.grad)


def test_cross_scale_positive_exclusion_removes_only_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def constructed_correspondences(
        similarities: torch.Tensor,
        periods: torch.Tensor,
        *,
        scale: float,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        del periods, valid
        result = torch.full(
            (similarities.shape[0], similarities.shape[1], 2),
            -1,
            dtype=torch.long,
            device=similarities.device,
        )
        result[0, 2, 0] = {0.5: 0, 1.5: 4}[scale]
        return result

    monkeypatch.setattr(
        losses_module,
        "_correspondences_from_similarities",
        constructed_correspondences,
    )
    embeddings = torch.tensor(
        [
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.6, 0.8],
                [0.2, 0.8],
                [0.0, 1.0],
            ]
        ],
        requires_grad=True,
    )
    objective = PAMSTCCLoss(
        scales=(0.5, 1.5),
        temperature=1.0,
        exclude_other_scale_positives_from_denominator=True,
    )
    details = objective.compute(
        embeddings,
        torch.tensor([2.0]),
        use_cross_cluster_negatives=False,
    )

    normalized = torch.nn.functional.normalize(embeddings, dim=-1, eps=1e-12)
    within_logits = torch.einsum("btd,bsd->bts", normalized, normalized)
    candidate_mask = ~torch.eye(5, dtype=torch.bool).unsqueeze(0)
    adjacency = torch.zeros_like(candidate_mask)
    adjacent = torch.arange(4)
    adjacency[:, adjacent, adjacent + 1] = True
    adjacency[:, adjacent + 1, adjacent] = True
    positive_masks = (adjacency.clone(), adjacency.clone())
    positive_masks[0][0, 2, 0] = True
    positive_masks[1][0, 2, 4] = True
    positive_union = positive_masks[0] | positive_masks[1]

    def expected_scale_loss(
        positive_mask: torch.Tensor,
        denominator_mask: torch.Tensor,
    ) -> torch.Tensor:
        positive_counts = positive_mask.sum(dim=-1)
        mean_positive = within_logits.masked_fill(~positive_mask, 0.0).sum(
            dim=-1
        ) / positive_counts.clamp_min(1).to(within_logits.dtype)
        cross_video_logits = torch.full(
            (1, 5, 1),
            float("-inf"),
            dtype=within_logits.dtype,
        )
        denominator = torch.logsumexp(
            torch.cat(
                [
                    within_logits.masked_fill(~denominator_mask, float("-inf")),
                    cross_video_logits,
                ],
                dim=-1,
            ),
            dim=-1,
        )
        valid_anchors = positive_counts > 0
        return (denominator - mean_positive)[valid_anchors].mean()

    expected_losses: list[torch.Tensor] = []
    for scale_index, positive_mask in enumerate(positive_masks):
        other_scale_only = positive_union & ~positive_mask
        denominator_mask = candidate_mask & ~other_scale_only
        other_periodic_positive = 4 if scale_index == 0 else 0
        current_periodic_positive = 0 if scale_index == 0 else 4
        assert not denominator_mask[0, 2, other_periodic_positive]
        assert denominator_mask[0, 2, current_periodic_positive]
        assert denominator_mask[0, 2, 1]
        assert denominator_mask[0, 2, 3]
        expected_losses.append(expected_scale_loss(positive_mask, denominator_mask))

        current_positive_removed = denominator_mask.clone()
        current_positive_removed[0, 2, current_periodic_positive] = False
        assert not torch.allclose(
            details.scale_losses[scale_index],
            expected_scale_loss(positive_mask, current_positive_removed),
        )

    assert all(
        torch.allclose(actual, expected)
        for actual, expected in zip(details.scale_losses, expected_losses, strict=True)
    )
    assert torch.isfinite(details.total)
    details.total.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()


def test_cross_cluster_pool_changes_denominator_without_instability() -> None:
    embeddings = _periodic_embeddings(batch=3)
    objective = PAMSTCCLoss(scales=(1.0,))
    without = objective(
        embeddings,
        torch.tensor([6, 6, 6]),
        cluster_labels=torch.tensor([0, 1, 1]),
        use_cross_cluster_negatives=False,
    )
    bank = torch.randn(8, embeddings.shape[-1])
    details = objective.compute(
        embeddings,
        torch.tensor([6, 6, 6]),
        cluster_labels=torch.tensor([0, 1, 1]),
        video_ids=("batch-a", "batch-b", "batch-c"),
        bank_features=bank,
        bank_cluster_labels=torch.tensor([1, 1, 1, 1, 0, 0, 0, 0]),
        bank_video_ids=tuple(f"bank-{index}" for index in range(len(bank))),
        use_cross_cluster_negatives=True,
    )
    assert torch.isfinite(details.total)
    assert details.total >= without
    requested = details.cross_cluster_requested_per_anchor[0]
    actual = details.cross_cluster_actual_per_anchor[0]
    shortfall = details.cross_cluster_shortfall_per_anchor[0]
    assert torch.equal(actual, requested)
    assert torch.count_nonzero(shortfall) == 0
    assert details.cross_cluster_requested_counts == (int(requested.sum()),)
    assert details.cross_cluster_actual_counts == (int(actual.sum()),)


def test_period_evidence_gate_keeps_only_local_tcc_positives() -> None:
    embeddings = _periodic_embeddings(batch=1)
    objective = PAMSTCCLoss(scales=(1.0,))
    bank = torch.randn(8, embeddings.shape[-1])
    bank_clusters = torch.ones(8, dtype=torch.long)
    bank_ids = tuple(f"bank-{index}" for index in range(8))
    with_period = objective.compute(
        embeddings,
        torch.tensor([6]),
        period_confidence=torch.tensor([1.0]),
        cluster_labels=torch.tensor([0]),
        video_ids=("current",),
        bank_features=bank,
        bank_cluster_labels=bank_clusters,
        bank_video_ids=bank_ids,
    )
    local_only = objective.compute(
        embeddings,
        torch.tensor([6]),
        period_confidence=torch.tensor([0.0]),
        cluster_labels=torch.tensor([0]),
        video_ids=("current",),
        bank_features=bank,
        bank_cluster_labels=bank_clusters,
        bank_video_ids=bank_ids,
    )

    requested_period = with_period.cross_cluster_requested_per_anchor[0]
    requested_local = local_only.cross_cluster_requested_per_anchor[0]
    assert int(requested_local.sum()) > 0
    assert int(requested_local.sum()) < int(requested_period.sum())
    assert int(requested_local.max()) <= 2
    assert torch.equal(
        local_only.cross_cluster_actual_per_anchor[0],
        requested_local,
    )


def test_cross_cluster_bank_reports_exact_distinct_shortfall() -> None:
    embeddings = _periodic_embeddings(batch=1)
    objective = PAMSTCCLoss(scales=(1.0,))
    bank = torch.randn(3, embeddings.shape[-1])
    details = objective.compute(
        embeddings,
        torch.tensor([6]),
        period_confidence=torch.tensor([1.0]),
        cluster_labels=torch.tensor([0]),
        video_ids=("current",),
        bank_features=bank,
        bank_cluster_labels=torch.tensor([1, 1, 1]),
        bank_video_ids=("eligible-a", "eligible-b", "current"),
    )
    requested = details.cross_cluster_requested_per_anchor[0]
    actual = details.cross_cluster_actual_per_anchor[0]
    shortfall = details.cross_cluster_shortfall_per_anchor[0]

    assert torch.equal(actual, requested.clamp_max(2))
    assert torch.equal(shortfall, requested - actual)
    assert int(shortfall.sum()) > 0
    assert torch.isfinite(details.total)


def test_sshead_weighting_and_anti_collapse() -> None:
    time = torch.arange(64, dtype=torch.float32)
    stream = (2.0 * torch.sin(2.0 * math.pi * time / 8)).requires_grad_()
    objective = SSHeadLoss(
        cycle_weight=1.0,
        spectral_weight=1.0,
        variance_weight=0.1,
        smoothness_weight=0.01,
    )
    details = objective.compute(stream, torch.tensor(8))
    expected = details.cycle + details.spectral + 0.1 * details.variance + 0.01 * details.smoothness
    assert torch.allclose(details.total, expected)
    assert details.cycle < 1e-5
    assert details.spectral < 0.05
    details.total.backward()
    assert stream.grad is not None
    assert torch.isfinite(stream.grad).all()

    collapsed = objective.compute(torch.zeros(64), torch.tensor(8))
    assert collapsed.variance == 1
    assert collapsed.spectral == 1
    assert collapsed.total > details.total


def test_sshead_ignores_invalid_cycle_pairs() -> None:
    stream = torch.randn(2, 20, requires_grad=True)
    mask = torch.ones(2, 20, dtype=torch.bool)
    mask[0, :8] = False
    mask[1] = False
    loss = SSHeadLoss()(stream, torch.tensor([5, 5]), mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert stream.grad is not None


def test_sshead_zero_period_evidence_skips_only_cycle_and_spectral() -> None:
    stream = (torch.arange(32, dtype=torch.float32) % 2).requires_grad_()
    details = SSHeadLoss().compute(
        stream,
        torch.tensor(8),
        period_confidence=torch.tensor(0.0),
    )
    assert details.cycle == 0
    assert details.spectral == 0
    assert details.variance > 0
    assert details.smoothness > 0
    assert details.total == 0.1 * details.variance + 0.01 * details.smoothness
    details.total.backward()
    assert stream.grad is not None


def test_sshead_can_normalize_period_losses_by_continuous_confidence() -> None:
    time = torch.arange(64, dtype=torch.float32)
    streams = torch.stack(
        (
            torch.sin(2.0 * math.pi * time / 8.0),
            torch.sin(2.0 * math.pi * time / 16.0),
        )
    ).requires_grad_()
    periods = torch.tensor([8.0, 8.0])
    confidences = torch.tensor([1.0, 0.1])
    unweighted = SSHeadLoss().compute(
        streams,
        periods,
        period_confidence=confidences,
    )
    weighted_objective = SSHeadLoss(
        confidence_weighted_period_losses=True,
    )
    weighted = weighted_objective.compute(
        streams,
        periods,
        period_confidence=confidences,
    )
    individual = [
        SSHeadLoss().compute(
            streams[index],
            periods[index],
            period_confidence=confidences[index],
        )
        for index in range(2)
    ]

    expected_spectral = (
        individual[0].spectral + 0.1 * individual[1].spectral
    ) / 1.1
    expected_cycle = (
        individual[0].cycle + 0.1 * individual[1].cycle
    ) / 1.1
    assert torch.allclose(weighted.spectral, expected_spectral)
    assert torch.allclose(weighted.cycle, expected_cycle)
    assert weighted.spectral < unweighted.spectral
    assert weighted.cycle < unweighted.cycle
    # Variance and smoothness are deliberately unchanged by this single
    # inferred repair.
    assert torch.equal(weighted.variance, unweighted.variance)
    assert torch.equal(weighted.smoothness, unweighted.smoothness)
    weighted.total.backward()
    assert streams.grad is not None
    assert torch.isfinite(streams.grad).all()

    equal_confidence = torch.full((2,), 0.25)
    equal_weighted = weighted_objective.compute(
        streams.detach(),
        periods,
        period_confidence=equal_confidence,
    )
    equal_unweighted = SSHeadLoss().compute(
        streams.detach(),
        periods,
        period_confidence=equal_confidence,
    )
    assert torch.allclose(equal_weighted.cycle, equal_unweighted.cycle)
    assert torch.allclose(equal_weighted.spectral, equal_unweighted.spectral)


def _circular_projected_pose(*, batch: int = 1, time: int = 32, period: int = 8) -> torch.Tensor:
    phase = 2.0 * math.pi * torch.arange(time, dtype=torch.float32) / period
    sample = torch.stack(
        (
            torch.cos(phase),
            torch.sin(phase),
            0.4 * torch.cos(phase + 0.3),
        ),
        dim=-1,
    )
    return sample.unsqueeze(0).repeat(batch, 1, 1)


def test_masked_zscore_is_continuous_at_a_constant_stream() -> None:
    time = 32
    stream = torch.zeros((1, time), requires_grad=True)
    valid = torch.ones((1, time), dtype=torch.bool)
    target = torch.sin(2.0 * math.pi * torch.arange(time) / 8.0).unsqueeze(0)

    standardized = masked_zscore(stream, valid)
    loss = (standardized - target).square().mean()
    gradient = torch.autograd.grad(loss, stream)[0]

    assert torch.equal(standardized.detach(), torch.zeros_like(standardized))
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.0


def test_reference_relative_signal_is_affine_invariant_and_peaks_at_reference() -> None:
    projected = _circular_projected_pose()
    valid = torch.ones((1, 32), dtype=torch.bool)
    periods = torch.tensor([8.0])
    confidence = torch.ones(1)

    base = build_reference_relative_signal(
        projected,
        periods,
        valid,
        period_confidence=confidence,
    )
    transformed = build_reference_relative_signal(
        3.5 * projected - 2.25,
        periods,
        valid,
        period_confidence=confidence,
    )

    assert base.available.tolist() == [True]
    assert transformed.available.tolist() == [True]
    assert torch.allclose(base.head_inputs, transformed.head_inputs, atol=3e-3, rtol=3e-3)
    assert torch.allclose(base.teacher, transformed.teacher, atol=3e-3, rtol=3e-3)
    anchor = _reference_anchor_index(projected[0], valid[0], period=8)
    same_phase = torch.arange(32).remainder(8) == anchor % 8
    assert torch.all(base.teacher[0, same_phase] >= base.teacher[0, ~same_phase].max())
    assert torch.allclose(
        transformed.prototypes[0],
        3.5 * base.prototypes[0] - 2.25,
    )


def test_reference_anchor_rejects_an_early_stationary_phase() -> None:
    projected = _circular_projected_pose()[0]
    projected[[0, 4, 8]] = projected[0]
    valid = torch.ones(32, dtype=torch.bool)

    anchor = _reference_anchor_index(projected, valid, period=8)

    assert anchor != 0
    full_neighbor = anchor + 8 if anchor + 8 < len(projected) else anchor - 8
    half_neighbor = anchor + 4 if anchor + 4 < len(projected) else anchor - 4
    full_distance = (projected[anchor] - projected[full_neighbor]).square().mean()
    half_distance = (projected[anchor] - projected[half_neighbor]).square().mean()
    assert half_distance > full_distance


def test_reference_relative_signal_ignores_mask_gaps_and_zero_confidence() -> None:
    projected = _circular_projected_pose(batch=2)
    valid = torch.ones((2, 32), dtype=torch.bool)
    valid[:, [3, 11, 19]] = False
    corrupted = projected.clone()
    corrupted[~valid] = 1_000_000.0
    periods = torch.tensor([8.0, 8.0])
    confidence = torch.tensor([1.0, 0.0])

    clean = build_reference_relative_signal(
        projected,
        periods,
        valid,
        period_confidence=confidence,
    )
    polluted = build_reference_relative_signal(
        corrupted,
        periods,
        valid,
        period_confidence=confidence,
    )

    assert clean.available.tolist() == [True, False]
    assert torch.allclose(clean.head_inputs[0], polluted.head_inputs[0])
    assert torch.allclose(clean.teacher[0], polluted.teacher[0])
    assert torch.count_nonzero(polluted.head_inputs[0, ~valid[0]]) == 0
    assert torch.count_nonzero(polluted.teacher[0, ~valid[0]]) == 0
    assert torch.count_nonzero(polluted.head_inputs[1]) == 0
    assert torch.count_nonzero(polluted.teacher[1]) == 0
    assert torch.count_nonzero(polluted.prototypes[1]) == 0


def test_reference_regression_has_nonzero_gradient_at_constant_output() -> None:
    time = 64
    phase = 2.0 * math.pi * torch.arange(time, dtype=torch.float32) / 8.0
    teacher = torch.sin(phase).unsqueeze(0)
    stream = torch.zeros((1, time), requires_grad=True)
    objective = ReferenceRelativeSSHeadLoss(
        fundamental_weight=0.0,
        lag_weight=0.0,
        low_frequency_weight=0.0,
        smoothness_weight=0.0,
    )

    details = objective.compute(
        stream,
        teacher,
        torch.tensor([8.0]),
        torch.ones((1, time), dtype=torch.bool),
        period_confidence=torch.ones(1),
        teacher_available=torch.ones(1, dtype=torch.bool),
    )
    gradient = torch.autograd.grad(details.total, stream)[0]

    assert details.reference > 0
    assert torch.isfinite(gradient).all()
    assert float(gradient.norm()) > 0.0
    assert int(torch.count_nonzero(gradient)) == time


def test_reference_objective_prefers_target_fundamental_over_low_frequency() -> None:
    time = 64
    indices = torch.arange(time, dtype=torch.float32)
    teacher = torch.sin(2.0 * math.pi * indices / 8.0).unsqueeze(0)
    target = teacher.clone()
    low_frequency = torch.sin(2.0 * math.pi * indices / 32.0).unsqueeze(0)
    objective = ReferenceRelativeSSHeadLoss()
    arguments = {
        "reference_teacher": teacher,
        "periods": torch.tensor([8.0]),
        "valid_mask": torch.ones((1, time), dtype=torch.bool),
        "period_confidence": torch.ones(1),
        "teacher_available": torch.ones(1, dtype=torch.bool),
    }

    target_details = objective.compute(target, **arguments)
    low_details = objective.compute(low_frequency, **arguments)

    assert target_details.fundamental < low_details.fundamental
    assert target_details.low_frequency < low_details.low_frequency
    assert target_details.lag < low_details.lag
    assert target_details.total < low_details.total


def test_reference_objective_is_output_scale_and_offset_invariant() -> None:
    time = 64
    indices = torch.arange(time, dtype=torch.float32)
    teacher = torch.sin(2.0 * math.pi * indices / 8.0).unsqueeze(0)
    stream = (
        teacher + 0.2 * torch.sin(4.0 * math.pi * indices / 8.0).unsqueeze(0)
    )
    objective = ReferenceRelativeSSHeadLoss()
    arguments = {
        "reference_teacher": teacher,
        "periods": torch.tensor([8.0]),
        "valid_mask": torch.ones((1, time), dtype=torch.bool),
        "period_confidence": torch.ones(1),
        "teacher_available": torch.ones(1, dtype=torch.bool),
    }

    base = objective.compute(stream, **arguments)
    transformed = objective.compute(4.0 * stream - 7.0, **arguments)

    for field in (
        "total",
        "reference",
        "fundamental",
        "lag",
        "low_frequency",
        "smoothness",
    ):
        assert torch.allclose(
            getattr(base, field),
            getattr(transformed, field),
            atol=2e-4,
            rtol=2e-4,
        )


def test_reference_objective_zero_confidence_disables_every_component() -> None:
    stream = torch.randn(1, 32, requires_grad=True)
    teacher = torch.randn(1, 32)
    details = ReferenceRelativeSSHeadLoss().compute(
        stream,
        teacher,
        torch.tensor([8.0]),
        torch.ones((1, 32), dtype=torch.bool),
        period_confidence=torch.zeros(1),
        teacher_available=torch.ones(1, dtype=torch.bool),
    )

    assert details.total == 0
    assert details.reference == 0
    assert details.fundamental == 0
    assert details.lag == 0
    assert details.low_frequency == 0
    assert details.smoothness == 0
    details.total.backward()
    assert torch.equal(stream.grad, torch.zeros_like(stream.grad))


def test_reference_objective_ignores_values_inside_mask_gaps() -> None:
    time = 64
    indices = torch.arange(time, dtype=torch.float32)
    stream = torch.sin(2.0 * math.pi * indices / 8.0).unsqueeze(0)
    teacher = stream.clone()
    valid = torch.ones((1, time), dtype=torch.bool)
    valid[0, [5, 6, 21, 37]] = False
    polluted_stream = stream.clone()
    polluted_teacher = teacher.clone()
    polluted_stream[~valid] = 1_000_000.0
    polluted_teacher[~valid] = -1_000_000.0
    objective = ReferenceRelativeSSHeadLoss()
    arguments = {
        "periods": torch.tensor([8.0]),
        "valid_mask": valid,
        "period_confidence": torch.ones(1),
        "teacher_available": torch.ones(1, dtype=torch.bool),
    }

    clean = objective.compute(stream, teacher, **arguments)
    polluted = objective.compute(
        polluted_stream,
        polluted_teacher,
        **arguments,
    )

    for field in (
        "total",
        "reference",
        "fundamental",
        "lag",
        "low_frequency",
        "smoothness",
    ):
        assert torch.equal(getattr(clean, field), getattr(polluted, field))


@torch.no_grad()
def test_period_confidence_rejects_non_finite_or_out_of_range_values() -> None:
    embeddings = _periodic_embeddings(batch=1)
    stream = torch.zeros(32)
    for invalid in (-0.1, 1.1, float("nan"), float("inf")):
        confidence = torch.tensor([invalid])
        with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
            PAMSTCCLoss(scales=(1.0,)).compute(
                embeddings,
                torch.tensor([6]),
                period_confidence=confidence,
            )
        with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
            SSHeadLoss().compute(
                stream,
                torch.tensor(8),
                period_confidence=confidence,
            )
