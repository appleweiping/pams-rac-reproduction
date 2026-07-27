import math

import torch

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


def test_periodic_correspondence_selects_exact_cycle_match() -> None:
    embeddings = _periodic_embeddings(batch=1)
    indices = periodic_correspondence_indices(
        embeddings,
        torch.tensor([6]),
        scale=1.0,
        correspondence_tolerance=0.1,
    )
    assert indices.shape == (1, 24, 2)
    assert indices[0, 10].tolist() == [4, 16]
    assert indices[0, 2, 0] == -1
    assert indices[0, 2, 1] == 8


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


def test_cross_cluster_pool_changes_denominator_without_instability() -> None:
    embeddings = _periodic_embeddings(batch=3)
    objective = PAMSTCCLoss(scales=(1.0,))
    without = objective(
        embeddings,
        torch.tensor([6, 6, 6]),
        cluster_labels=torch.tensor([0, 1, 1]),
        use_cross_cluster_negatives=False,
    )
    with_cross = objective(
        embeddings,
        torch.tensor([6, 6, 6]),
        cluster_labels=torch.tensor([0, 1, 1]),
        use_cross_cluster_negatives=True,
    )
    assert torch.isfinite(with_cross)
    assert with_cross >= without


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
