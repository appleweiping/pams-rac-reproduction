import math

import pytest
import torch

import pams.losses as losses_module
from pams.losses import PAMSTCCLoss, SSHeadLoss, periodic_correspondence_indices


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
