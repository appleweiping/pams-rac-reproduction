import numpy as np
import pytest
import torch

from pams.consensus import MultiExpertCounter, dynamic_threshold, vote_expert_counts


def _peak_stream(length: int, peaks: list[int], width: float = 1.5) -> np.ndarray:
    time = np.arange(length)
    result = np.zeros(length, dtype=np.float64)
    for peak in peaks:
        result += np.exp(-0.5 * ((time - peak) / width) ** 2)
    return result


def test_three_experts_count_clean_periodic_peaks() -> None:
    stream = _peak_stream(120, [10, 30, 50, 70, 90, 110])
    result = MultiExpertCounter().count(stream, period_frames=20)
    assert result.count == 6
    assert result.reference_count == 6
    assert result.expert_counts[:2] == (6, 6)
    assert result.expert_counts[2] < 6
    assert result.selected_expert == "medium"
    assert result.confidence == 2 / 3
    assert all(len(expert.threshold) == 120 for expert in result.experts)
    compact = result.to_count_result()
    assert compact.count == result.count
    assert compact.expert_counts == result.expert_counts


def test_vote_uses_majority_then_reference_with_medium_tie_break() -> None:
    assert vote_expert_counts((5, 5, 7), reference_count=6) == (5, 1, 2 / 3)
    count, selected, confidence = vote_expert_counts((4, 8, 10), reference_count=7)
    assert (count, selected) == (8, 1)
    assert 0 < confidence < 1 / 3

    count, selected, _ = vote_expert_counts((6, 8, 4), reference_count=6)
    assert (count, selected) == (6, 0)


def test_medium_only_returns_medium_without_vote_or_fft_fallback() -> None:
    stream = _peak_stream(120, [10, 30, 50, 70, 90, 110])
    counter = MultiExpertCounter(expert_mode="medium_only")
    result = counter.count(stream, period_frames=11, period_confidence=0.4)

    assert result.selection_mode == "medium_only"
    assert result.selected_expert == "medium"
    assert result.count == result.expert_counts[1]
    assert result.confidence == pytest.approx(0.4)

    with pytest.raises(ValueError, match="expert_mode"):
        MultiExpertCounter(expert_mode="oracle")  # type: ignore[arg-type]


def test_dynamic_threshold_and_invalid_frames_are_safe() -> None:
    stream = _peak_stream(80, [10, 30, 50, 70])
    threshold = dynamic_threshold(stream, period_frames=20)
    assert threshold.shape == stream.shape
    assert np.isfinite(threshold).all()

    mask = torch.ones(80, dtype=torch.bool)
    mask[35:45] = False
    result = MultiExpertCounter().count(
        torch.tensor(stream),
        period_frames=20,
        valid_mask=mask,
    )
    assert 0 <= result.count <= 4
    assert np.isinf(result.experts[0].threshold[35:45]).all()


def test_constant_and_fully_invalid_streams_return_zero() -> None:
    counter = MultiExpertCounter()
    constant = counter.count(np.ones(64), period_frames=16)
    assert constant.count == 0
    assert constant.expert_counts == (0, 0, 0)

    invalid = counter.count(
        np.random.default_rng(2).normal(size=64),
        period_frames=16,
        valid_mask=np.zeros(64, dtype=bool),
    )
    assert invalid.count == 0
    assert invalid.selected_expert == "medium"
    assert invalid.confidence == 0.0

    bfloat_stream = torch.ones(64, dtype=torch.bfloat16)
    assert counter.count(bfloat_stream, period_frames=16).count == 0


def test_period_confidence_scales_vote_confidence_and_rejects_invalid_values() -> None:
    counter = MultiExpertCounter()
    stream = np.ones(64)
    assert counter.count(stream, period_frames=16).confidence == 1.0
    assert counter.count(stream, period_frames=16, period_confidence=0.0).confidence == 0.0
    assert counter.count(stream, period_frames=16, period_confidence=0.25).confidence == 0.25

    for invalid in (-0.1, 1.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="period_confidence"):
            counter.count(stream, period_frames=16, period_confidence=invalid)


def test_valid_runs_are_filtered_independently_of_gap_content_and_length() -> None:
    counter = MultiExpertCounter()
    run = _peak_stream(60, [10, 30, 50])

    def separated(gap_length: int, gap_value: float) -> tuple[object, np.ndarray]:
        stream = np.concatenate((run, np.full(gap_length, gap_value), run))
        mask = np.ones(len(stream), dtype=bool)
        mask[60 : 60 + gap_length] = False
        return counter.count(stream, period_frames=20, valid_mask=mask), mask

    short_zero, short_mask = separated(7, 0.0)
    short_corrupt, _ = separated(7, 1e9)
    long_corrupt, long_mask = separated(41, -1e9)

    assert short_zero.expert_counts == short_corrupt.expert_counts
    assert short_zero.count == short_corrupt.count == long_corrupt.count
    assert short_zero.expert_counts == long_corrupt.expert_counts
    assert all(np.isinf(expert.threshold[~short_mask]).all() for expert in short_corrupt.experts)
    assert all(np.isinf(expert.threshold[~long_mask]).all() for expert in long_corrupt.experts)


def test_explicit_reference_frames_use_dense_timeline_without_changing_experts() -> None:
    counter = MultiExpertCounter()
    stream = _peak_stream(80, [10, 30, 50, 70])
    mask = np.ones(80, dtype=bool)
    mask[30:50] = False

    compact = counter.count(stream, period_frames=20, valid_mask=mask)
    dense = counter.count(
        stream,
        period_frames=20,
        valid_mask=mask,
        reference_frames=len(stream),
    )

    assert compact.reference_count == 3
    assert dense.reference_count == 4
    assert dense.expert_counts == compact.expert_counts
    assert tuple(expert.peaks for expert in dense.experts) == tuple(
        expert.peaks for expert in compact.experts
    )

    fully_invalid = counter.count(
        stream,
        period_frames=20,
        valid_mask=np.zeros_like(mask),
        reference_frames=len(stream),
    )
    assert fully_invalid.count == 0
    assert fully_invalid.reference_count == 0

    with pytest.raises(ValueError, match="cannot exceed"):
        counter.count(
            stream,
            period_frames=20,
            valid_mask=mask,
            reference_frames=len(stream) + 1,
        )
    with pytest.raises(ValueError, match="positive integer"):
        counter.count(stream, period_frames=20, valid_mask=mask, reference_frames=0)
    with pytest.raises(ValueError, match="beyond reference_frames"):
        counter.count(
            stream,
            period_frames=20,
            valid_mask=mask,
            reference_frames=40,
        )
