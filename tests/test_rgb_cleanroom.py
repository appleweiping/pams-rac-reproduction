from collections.abc import Callable

import pytest
import torch
from torch import nn

from pams.baselines.rgb_cleanroom import (
    ESCountsStyleRegressor,
    IVACP2LStyleCounter,
    RepNetStyleCounter,
    TransRACStyleRegressor,
    temporal_self_similarity,
)


def _features(batch: int = 2, time: int = 8, dimension: int = 6) -> torch.Tensor:
    generator = torch.Generator().manual_seed(2026)
    return torch.randn(batch, time, dimension, generator=generator)


def _mask() -> torch.Tensor:
    return torch.tensor(
        [
            [True, True, True, True, True, False, False, False],
            [False, False, False, False, False, False, False, False],
        ]
    )


def _assert_nonnegative_masked_density(
    density: torch.Tensor,
    count: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    assert density.shape == mask.shape
    assert count.shape == (mask.shape[0],)
    assert torch.isfinite(density).all()
    assert torch.isfinite(count).all()
    assert torch.all(density >= 0)
    assert torch.count_nonzero(density[~mask]) == 0
    assert count[1] == 0
    assert torch.allclose(count, density.sum(dim=1))


def test_temporal_self_similarity_is_cosine_and_mask_safe() -> None:
    features = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [5.0, 5.0]]])
    mask = torch.tensor([[True, True, True, False]])
    similarity = temporal_self_similarity(features, mask)
    expected = torch.tensor(
        [
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ]
    )
    assert torch.allclose(similarity, expected)


@pytest.mark.parametrize(
    "features,mask,error",
    [
        (torch.ones(4, 6), None, ValueError),
        (torch.ones(1, 4, 6, dtype=torch.int64), None, TypeError),
        (torch.ones(1, 4, 6), torch.ones(1, 3, dtype=torch.bool), ValueError),
        (torch.ones(1, 4, 6), torch.ones(1, 4), TypeError),
    ],
)
def test_temporal_self_similarity_validates_inputs(
    features: torch.Tensor,
    mask: torch.Tensor | None,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        temporal_self_similarity(features, mask)


def test_repnet_style_heads_are_structured_differentiable_and_mask_safe() -> None:
    model = RepNetStyleCounter(
        input_dim=6,
        model_dim=12,
        num_layers=1,
        num_heads=3,
        feedforward_dim=24,
        dropout=0.0,
        period_bins=(2, 4, 8),
        max_length=16,
    )
    features = _features().requires_grad_()
    mask = _mask()
    output = model(features, mask)

    assert output.temporal_similarity.shape == (2, 8, 8)
    assert output.period_logits.shape == (2, 8, 3)
    assert output.periodicity_logits.shape == (2, 8)
    assert output.within_period_logits.shape == (2, 8)
    assert output.period_length_logits is output.period_logits
    assert output.periodicity is output.within_period_probability
    assert output.period_length_distribution.shape == (2, 8, 3)
    assert torch.allclose(
        output.period_length_distribution[0, :5].sum(dim=-1),
        torch.ones(5),
    )
    assert torch.count_nonzero(output.period_length_distribution[~mask]) == 0
    _assert_nonnegative_masked_density(output.density, output.count, mask)

    output.count.sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_transrac_style_density_is_nonnegative_and_mask_safe() -> None:
    model = TransRACStyleRegressor(
        input_dim=6,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
        max_length=16,
    )
    features = _features().requires_grad_()
    mask = _mask()
    output = model(features, mask)
    assert output.hidden.shape == (2, 8, 8)
    _assert_nonnegative_masked_density(output.density, output.count, mask)
    output.count.sum().backward()
    assert features.grad is not None


def test_escounts_style_cross_attention_respects_both_masks() -> None:
    model = ESCountsStyleRegressor(
        input_dim=6,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
        max_length=16,
    )
    query = _features().requires_grad_()
    exemplars = _features(time=4).requires_grad_()
    query_mask = _mask()
    exemplar_mask = torch.tensor(
        [
            [True, True, False, False],
            [False, False, False, False],
        ]
    )
    output = model(query, exemplars, query_mask, exemplar_mask)

    assert output.cross_attention.shape == (2, 8, 4)
    assert torch.count_nonzero(output.cross_attention[:, :, 2:]) == 0
    assert torch.count_nonzero(output.cross_attention[1]) == 0
    _assert_nonnegative_masked_density(output.density, output.count, query_mask)
    output.count.sum().backward()
    assert query.grad is not None
    assert exemplars.grad is not None


def test_escounts_requires_matching_batches_and_zeroes_missing_exemplars() -> None:
    model = ESCountsStyleRegressor(
        input_dim=6,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
    )
    with pytest.raises(ValueError, match="batch size"):
        model(_features(batch=2), _features(batch=1))

    query = _features(batch=1)
    exemplars = _features(batch=1, time=3)
    output = model(
        query,
        exemplars,
        torch.ones(1, 8, dtype=torch.bool),
        torch.zeros(1, 3, dtype=torch.bool),
    )
    assert torch.count_nonzero(output.density) == 0
    assert output.count.item() == 0


def test_ivac_p2l_distribution_produces_expected_period_and_count() -> None:
    model = IVACP2LStyleCounter(
        input_dim=6,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
        period_bins=(2, 5, 9),
        max_length=16,
    )
    features = _features().requires_grad_()
    mask = _mask()
    output = model(features, mask)

    assert output.period_length_logits.shape == (2, 8, 3)
    assert output.period_length_distribution.shape == (2, 8, 3)
    assert output.expected_period.shape == (2, 8)
    assert torch.all(output.expected_period[mask] >= 2)
    assert torch.all(output.expected_period[mask] <= 9)
    assert torch.count_nonzero(output.expected_period[~mask]) == 0
    _assert_nonnegative_masked_density(output.density, output.count, mask)
    output.count.sum().backward()
    assert features.grad is not None


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RepNetStyleCounter(
            input_dim=6,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            feedforward_dim=16,
            period_bins=(2, 4),
        ),
        lambda: TransRACStyleRegressor(
            input_dim=6,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            feedforward_dim=16,
        ),
        lambda: IVACP2LStyleCounter(
            input_dim=6,
            model_dim=8,
            num_layers=1,
            num_heads=2,
            feedforward_dim=16,
            period_bins=(2, 4),
        ),
    ],
)
def test_sequence_models_reject_wrong_feature_and_mask_shapes(
    factory: Callable[[], nn.Module],
) -> None:
    model = factory()
    with pytest.raises(ValueError, match="feature dimension"):
        model(torch.ones(1, 4, 5))
    with pytest.raises(ValueError, match="valid_mask"):
        model(torch.ones(1, 4, 6), torch.ones(1, 3, dtype=torch.bool))


def test_period_bins_must_be_strictly_increasing() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        IVACP2LStyleCounter(input_dim=6, period_bins=(4, 2, 4))
