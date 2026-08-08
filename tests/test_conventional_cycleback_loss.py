from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from pams.conventional_cycleback.loss import ConventionalCycleBackLoss


def _objective() -> ConventionalCycleBackLoss:
    return ConventionalCycleBackLoss(
        temperature=0.1,
        variance_log_weight=0.001,
        variance_floor=1e-6,
    )


def _compute(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    valid_a: torch.Tensor,
    valid_b: torch.Tensor,
    source_a: torch.Tensor,
    source_b: torch.Tensor,
    lengths: torch.Tensor,
):
    identifiers = tuple(f"video-{index}" for index in range(embeddings_a.shape[0]))
    return _objective().compute(
        embeddings_a,
        embeddings_b,
        valid_a,
        valid_b,
        source_a,
        source_b,
        lengths,
        video_ids_a=identifiers,
        video_ids_b=identifiers,
    )


def test_variance_aware_direction_matches_explicit_cycleback_equation() -> None:
    embeddings = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    valid = torch.ones((1, 3), dtype=torch.bool)
    source_a = torch.tensor([[10, 11, 12]], dtype=torch.long)
    source_b = torch.tensor([[13, 14, 15]], dtype=torch.long)
    lengths = torch.tensor([20], dtype=torch.long)

    output = _compute(
        embeddings,
        embeddings,
        valid,
        valid,
        source_a,
        source_b,
        lengths,
    )

    normalized = F.normalize(embeddings, dim=-1)
    forward = torch.softmax(torch.einsum("btd,bsd->bts", normalized, normalized) / 0.1, dim=-1)
    soft_candidate = torch.einsum("bts,bsd->btd", forward, normalized)
    backward = torch.softmax(
        torch.einsum("btd,bsd->bts", soft_candidate, normalized) / 0.1,
        dim=-1,
    )
    positions = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float64)
    predicted = torch.einsum("bts,bs->bt", backward, positions)
    variance = torch.einsum(
        "bts,bts->bt",
        backward,
        (positions.unsqueeze(1) - predicted.unsqueeze(-1)).square(),
    ).clamp_min(1e-6)
    expected = (
        (positions - predicted).square() / variance
        + 0.001 * torch.log(variance)
    ).mean()

    assert output.total == pytest.approx(expected)
    assert output.a_to_b_to_a.predicted_normalized_positions == pytest.approx(predicted)
    assert output.a_to_b_to_a.target_normalized_positions == pytest.approx(positions)
    assert output.valid_anchor_count == 6


def test_invalid_padding_is_exactly_loss_invariant() -> None:
    generator = torch.Generator().manual_seed(2026)
    base_a = torch.randn((1, 4, 4), generator=generator)
    base_b = torch.randn((1, 4, 4), generator=generator)
    valid = torch.tensor([[True, True, True, False]])
    source_a = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    source_b = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
    lengths = torch.tensor([8], dtype=torch.long)
    expected = _compute(base_a, base_b, valid, valid, source_a, source_b, lengths)

    padded_a = base_a.clone()
    padded_b = base_b.clone()
    padded_a[:, -1] = 1e20
    padded_b[:, -1] = -1e20
    observed = _compute(
        padded_a,
        padded_b,
        valid,
        valid,
        source_a,
        source_b,
        lengths,
    )

    assert observed.total == pytest.approx(expected.total, abs=1e-12)
    assert observed.valid_anchor_count == expected.valid_anchor_count


def test_symmetric_objective_swaps_directional_outputs() -> None:
    generator = torch.Generator().manual_seed(3407)
    first = torch.randn((2, 5, 8), generator=generator)
    second = torch.randn((2, 5, 8), generator=generator)
    valid = torch.tensor(
        [[True, True, True, True, False], [True, True, True, True, True]]
    )
    source_a = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
    source_b = torch.tensor([[5, 6, 7, 8, 9], [6, 7, 8, 9, 10]])
    lengths = torch.tensor([12, 12])

    forward = _compute(first, second, valid, valid, source_a, source_b, lengths)
    reverse = _compute(second, first, valid, valid, source_b, source_a, lengths)

    assert forward.total == pytest.approx(reverse.total)
    assert forward.a_to_b_to_a.loss == pytest.approx(reverse.b_to_a_to_b.loss)
    assert forward.b_to_a_to_b.loss == pytest.approx(reverse.a_to_b_to_a.loss)


def test_targets_are_window_local_and_independent_of_native_length() -> None:
    embeddings = torch.eye(4).unsqueeze(0)
    valid = torch.ones((1, 4), dtype=torch.bool)
    source_a = torch.tensor([[10, 11, 12, 13]])
    source_b = torch.tensor([[14, 15, 16, 17]])
    lengths = torch.tensor([101])

    output = _compute(
        embeddings,
        embeddings,
        valid,
        valid,
        source_a,
        source_b,
        lengths,
    )

    assert output.a_to_b_to_a.target_normalized_positions[0] == pytest.approx(
        torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    )

    longer = _compute(
        embeddings,
        embeddings,
        valid,
        valid,
        source_a,
        source_b,
        torch.tensor([10_001]),
    )
    assert longer.total == pytest.approx(output.total)
    assert longer.a_to_b_to_a.target_normalized_positions == pytest.approx(
        output.a_to_b_to_a.target_normalized_positions
    )


def test_training_api_rejects_overlapping_source_frames() -> None:
    embeddings = torch.eye(4).unsqueeze(0)
    valid = torch.ones((1, 4), dtype=torch.bool)

    with pytest.raises(ValueError, match="disjoint source frames"):
        _compute(
            embeddings,
            embeddings,
            valid,
            valid,
            torch.tensor([[0, 1, 2, 3]]),
            torch.tensor([[3, 4, 5, 6]]),
            torch.tensor([7]),
        )


def test_window_local_position_normalization_preserves_float64_precision() -> None:
    embeddings = torch.eye(3, dtype=torch.float64).unsqueeze(0)
    valid = torch.ones((1, 3), dtype=torch.bool)
    source_a = torch.tensor([[50_000_001, 50_000_002, 50_000_003]])
    source_b = torch.tensor([[50_000_004, 50_000_005, 50_000_006]])
    lengths = torch.tensor([100_000_001])

    output = _compute(
        embeddings,
        embeddings,
        valid,
        valid,
        source_a,
        source_b,
        lengths,
    )

    expected = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float64)
    assert torch.equal(output.a_to_b_to_a.target_normalized_positions, expected)


def test_invalid_holes_preserve_absolute_window_time_spacing() -> None:
    embeddings = torch.eye(4).unsqueeze(0)
    valid = torch.tensor([[True, False, True, True]])
    source_a = torch.tensor([[30, 31, 32, 33]])
    source_b = torch.tensor([[34, 35, 36, 37]])
    output = _compute(
        embeddings,
        embeddings,
        valid,
        valid,
        source_a,
        source_b,
        torch.tensor([100]),
    )

    assert output.a_to_b_to_a.target_normalized_positions[0] == pytest.approx(
        torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    )
    assert output.a_to_b_to_a.valid_anchor_mask[0].tolist() == [True, False, True, True]


def test_batch_rows_are_mathematically_isolated() -> None:
    generator = torch.Generator().manual_seed(144)
    first = torch.randn((2, 5, 7), generator=generator)
    second = torch.randn((2, 5, 7), generator=generator)
    valid = torch.ones((2, 5), dtype=torch.bool)
    source_a = torch.arange(5).repeat(2, 1)
    source_b = torch.arange(5, 10).repeat(2, 1)
    lengths = torch.tensor([10, 10])
    reference = _compute(first, second, valid, valid, source_a, source_b, lengths)

    changed_first = first.clone()
    changed_second = second.clone()
    changed_first[1] = 1e6 * torch.randn((5, 7), generator=generator)
    changed_second[1] = -1e6 * torch.randn((5, 7), generator=generator)
    changed = _compute(
        changed_first,
        changed_second,
        valid,
        valid,
        source_a,
        source_b,
        lengths,
    )

    assert changed.a_to_b_to_a.predicted_normalized_positions[0] == pytest.approx(
        reference.a_to_b_to_a.predicted_normalized_positions[0]
    )
    assert changed.b_to_a_to_b.predicted_normalized_positions[0] == pytest.approx(
        reference.b_to_a_to_b.predicted_normalized_positions[0]
    )


def test_valid_nonfinite_embedding_is_rejected() -> None:
    embeddings = torch.eye(3).unsqueeze(0)
    embeddings[0, 1] = float("nan")
    valid = torch.ones((1, 3), dtype=torch.bool)
    source_a = torch.arange(3).unsqueeze(0)
    source_b = torch.arange(3, 6).unsqueeze(0)

    with pytest.raises(ValueError, match="finite"):
        _compute(
            embeddings,
            torch.eye(3).unsqueeze(0),
            valid,
            valid,
            source_a,
            source_b,
            torch.tensor([6]),
        )


def test_training_api_rejects_different_video_pairs() -> None:
    embeddings = torch.eye(3).unsqueeze(0)
    valid = torch.ones((1, 3), dtype=torch.bool)
    source_a = torch.arange(3).unsqueeze(0)
    source_b = torch.arange(3, 6).unsqueeze(0)
    lengths = torch.tensor([6])

    with pytest.raises(ValueError, match="same video"):
        _objective().compute(
            embeddings,
            embeddings,
            valid,
            valid,
            source_a,
            source_b,
            lengths,
            video_ids_a=("video-a",),
            video_ids_b=("video-b",),
        )


def test_different_video_control_is_diagnostic_only_and_requires_mismatch() -> None:
    embeddings = torch.eye(3).repeat(2, 1, 1)
    valid = torch.ones((2, 3), dtype=torch.bool)
    source = torch.arange(3).repeat(2, 1)
    lengths = torch.tensor([3, 3])

    output = _objective().diagnostic_mismatched_video_control(
        embeddings,
        embeddings.flip(0),
        valid,
        valid,
        source,
        source,
        lengths,
        lengths,
        video_ids_a=("a", "b"),
        video_ids_b=("b", "a"),
    )
    assert torch.isfinite(output.total)
    with pytest.raises(ValueError, match="same-video row"):
        _objective().diagnostic_mismatched_video_control(
            embeddings,
            embeddings,
            valid,
            valid,
            source,
            source,
            lengths,
            lengths,
            video_ids_a=("a", "b"),
            video_ids_b=("a", "b"),
        )


def test_loss_has_finite_encoder_gradients() -> None:
    generator = torch.Generator().manual_seed(7)
    embeddings_a = torch.randn((2, 4, 6), generator=generator, requires_grad=True)
    embeddings_b = torch.randn((2, 4, 6), generator=generator, requires_grad=True)
    valid = torch.ones((2, 4), dtype=torch.bool)
    source_a = torch.arange(4).repeat(2, 1)
    source_b = torch.arange(4, 8).repeat(2, 1)
    lengths = torch.tensor([8, 8])

    output = _compute(
        embeddings_a,
        embeddings_b,
        valid,
        valid,
        source_a,
        source_b,
        lengths,
    )
    output.total.backward()

    assert embeddings_a.grad is not None and torch.isfinite(embeddings_a.grad).all()
    assert embeddings_b.grad is not None and torch.isfinite(embeddings_b.grad).all()
