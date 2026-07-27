import numpy as np
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
    assert np.isinf(result.experts[0].threshold[35:45]).sum() == 0


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
