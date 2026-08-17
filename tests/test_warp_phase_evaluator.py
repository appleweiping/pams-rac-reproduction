from __future__ import annotations

import math

import numpy as np
import pytest

from pams.warp_phase.evaluator import (
    OFFICIAL_MULTIREP_EVALUATOR_BLOB_SHA,
    OFFICIAL_MULTIREP_EVALUATOR_COMMIT,
    OFFICIAL_MULTIREP_EVALUATOR_SHA256,
    OFFICIAL_PERIOD_TIOU_THRESHOLDS,
    EvaluationError,
    classify_harmonic,
    evaluate_counts,
    evaluate_harmonic_gate,
    evaluate_k4,
    evaluate_supplied_track_period_ap,
    paired_component_bootstrap,
    period_median,
)


def _reference_tiou(first: tuple[float, float], second: tuple[float, float]) -> float:
    overlap = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    return overlap / ((first[1] - first[0]) + (second[1] - second[0]) - overlap)


def _independent_period_ap(
    truth: dict[str, list[tuple[int, int]]],
    prediction: dict[str, list[tuple[float, float, float]]],
) -> list[float]:
    """Slow declarative oracle; intentionally structured unlike the source port."""

    gt = [
        (track_id, (float(start), float(end)))
        for track_id, intervals in truth.items()
        for start, end in intervals
    ]
    proposals = [
        (track_id, (start, end), score)
        for track_id in truth
        for start, end, score in prediction.get(track_id, [])
    ]
    ranked = sorted(range(len(proposals)), key=lambda index: proposals[index][2], reverse=True)
    output: list[float] = []
    for threshold in OFFICIAL_PERIOD_TIOU_THRESHOLDS:
        used: set[int] = set()
        is_true_positive: list[bool] = []
        for prediction_index in ranked:
            track_id, interval, _ = proposals[prediction_index]
            eligible = [
                (_reference_tiou(interval, gt_interval), gt_index)
                for gt_index, (gt_track, gt_interval) in enumerate(gt)
                if gt_track == track_id and gt_index not in used
            ]
            best = max(eligible, default=(-1.0, -1))
            matched = best[0] >= threshold
            is_true_positive.append(matched)
            if matched:
                used.add(best[1])

        if not is_true_positive:
            output.append(0.0)
            continue
        cumulative_true = np.cumsum(np.asarray(is_true_positive, dtype=np.float64))
        ranks = np.arange(1, len(is_true_positive) + 1, dtype=np.float64)
        precision = cumulative_true / ranks
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        output.append(
            sum(
                float(envelope[index]) / len(gt)
                for index, matched in enumerate(is_true_positive)
                if matched
            )
            * 100.0
        )
    return output


def test_video_first_count_metric_and_half_even_obo() -> None:
    truth = {"a": {"0": 2.0, "1": 4.0}, "b": {"0": 5.0}}
    prediction = {"a": {"0": 3.0, "1": 2.0}, "b": {"0": 7.0}}
    result = evaluate_counts(truth, prediction)
    assert result.per_video_normalized_mae == {"a": 0.5, "b": 0.4}
    assert result.normalized_video_first_mae == 0.45
    assert result.avg_obo == 0.25


def test_period_median_uses_half_open_source_boundaries() -> None:
    assert period_median([[0, 4], [4, 9], [9, 15]], 15) == 5.0
    with pytest.raises(EvaluationError):
        period_median([[0, 5], [4, 9]], 10)
    with pytest.raises(EvaluationError):
        period_median([], 10)


def test_period_ap_keeps_the_pinned_official_source_receipt() -> None:
    assert OFFICIAL_MULTIREP_EVALUATOR_COMMIT == "d15c767d82dac2f7a5df9fde1d7f6b5a4f491062"
    assert OFFICIAL_MULTIREP_EVALUATOR_BLOB_SHA == "e5c188aa032957fa628062df645a7e1fe8d59b6d"
    assert OFFICIAL_MULTIREP_EVALUATOR_SHA256 == (
        "c10a3a56704d8bdf1ba55a2681570dc77371089ecb73c71c56102d26eca1386c"
    )


def test_period_ap_perfect_and_empty_predictions_use_percentage_scale() -> None:
    truth = {"track": [(0, 10), (10, 20)]}
    perfect = evaluate_supplied_track_period_ap(
        truth,
        {"track": [(0.0, 10.0, 0.9), (10.0, 20.0, 0.8)]},
    )
    assert perfect.period_map == 100.0
    assert perfect.ap50 == 100.0
    assert perfect.ap75 == 100.0
    assert perfect.scale == "[0,100]"

    empty = evaluate_supplied_track_period_ap(truth, {})
    assert empty.period_map == 0.0
    assert empty.ap50 == 0.0
    assert empty.ap75 == 0.0


def test_period_ap_duplicate_and_unmatched_proposals_follow_official_locking() -> None:
    truth = {"track": [(0, 10)]}
    duplicate = evaluate_supplied_track_period_ap(
        truth,
        {"track": [(0.0, 10.0, 0.9), (0.0, 10.0, 0.8)]},
    )
    # The lower-scored duplicate is an FP, but it arrives after full recall and
    # therefore does not lower interpolated AP.
    assert duplicate.period_map == 100.0

    false_before_true = evaluate_supplied_track_period_ap(
        truth,
        {"track": [(40.0, 50.0, 0.9), (0.0, 10.0, 0.8)]},
    )
    assert false_before_true.period_map == 50.0


def test_period_ap_threshold_equality_is_inclusive() -> None:
    result = evaluate_supplied_track_period_ap(
        {"track": [(0, 10)]},
        {"track": [(0.0, 5.0, 1.0)]},
    )
    assert result.ap50 == 100.0
    assert result.ap75 == 0.0
    assert result.period_map == 10.0


def test_period_ap_exact_score_ties_preserve_official_argsort_reverse_order() -> None:
    truth = {"track": [(0, 10), (20, 30)]}
    true_then_false = evaluate_supplied_track_period_ap(
        truth,
        {"track": [(0.0, 10.0, 0.5), (40.0, 50.0, 0.5)]},
    )
    false_then_true = evaluate_supplied_track_period_ap(
        truth,
        {"track": [(40.0, 50.0, 0.5), (0.0, 10.0, 0.5)]},
    )
    assert true_then_false.period_map == 25.0
    assert false_then_true.period_map == 50.0


def test_period_ap_exact_tiou_tie_prefers_the_later_candidate() -> None:
    result = evaluate_supplied_track_period_ap(
        {"track": [(0, 10), (10, 20)]},
        {"track": [(0.0, 20.0, 0.9), (10.0, 20.0, 0.8)]},
    )
    # At 0.5 the first proposal ties both GT periods. Official reversed
    # argsort visits the later period first, leaving the second proposal an FP.
    assert result.ap50 == 50.0
    assert result.ap75 == 25.0
    assert result.period_map == pytest.approx(27.5)


def test_period_ap_random_small_cases_match_an_independent_oracle() -> None:
    # The official repository contains no Period-AP fixture at the pinned
    # commit, so use deterministic random parity against a separately written
    # oracle. Unique scores and nonidentical GT intervals avoid relying on the
    # dedicated tie test above.
    rng = np.random.Generator(np.random.PCG64(81723))
    for _ in range(24):
        truth: dict[str, list[tuple[int, int]]] = {}
        prediction: dict[str, list[tuple[float, float, float]]] = {}
        score = 1.0
        for track_index in range(3):
            track_id = f"track-{track_index}"
            offset = 100 * track_index + 20
            truth[track_id] = [(offset, offset + 10), (offset + 20, offset + 33)]
            proposals: list[tuple[float, float, float]] = []
            for start, end in truth[track_id]:
                left_jitter = int(rng.integers(-5, 6))
                right_jitter = int(rng.integers(-5, 6))
                proposal_start = float(start + left_jitter)
                proposal_end = float(max(start + left_jitter + 1, end + right_jitter))
                proposals.append((proposal_start, proposal_end, score))
                score -= 0.013
            proposals.append((float(offset + 60), float(offset + 70), score))
            score -= 0.013
            prediction[track_id] = proposals

        result = evaluate_supplied_track_period_ap(truth, prediction)
        expected = _independent_period_ap(truth, prediction)
        actual = [result.ap_by_tiou[threshold] for threshold in OFFICIAL_PERIOD_TIOU_THRESHOLDS]
        assert actual == pytest.approx(expected)
        assert result.period_map == pytest.approx(float(np.mean(expected)))


def test_period_ap_fails_closed_outside_the_supplied_track_contract() -> None:
    with pytest.raises(EvaluationError):
        evaluate_supplied_track_period_ap({}, {})
    with pytest.raises(EvaluationError):
        evaluate_supplied_track_period_ap({"track": [(0, 10)]}, {"unknown": []})
    with pytest.raises(EvaluationError):
        evaluate_supplied_track_period_ap(
            {"track": [(0, 10)]},
            {"track": [(0.0, 10.0, float("nan"))]},
        )
    with pytest.raises(EvaluationError):
        evaluate_supplied_track_period_ap(
            {"track": [(0, 10), (9, 12)]},
            {},
        )


def test_harmonic_gate_is_rejection_only() -> None:
    assert classify_harmonic(20.0, 20.0) == "fundamental"
    assert classify_harmonic(10.0, 20.0) == "half"
    assert classify_harmonic(40.0, 20.0) == "double"
    result = evaluate_harmonic_gate(
        {"video": {str(index): (20.0, 20.0) for index in range(10)}}
    )
    assert result.passed
    assert result.fractions["fundamental"] == 1.0


def test_k4_uses_only_frozen_ratios() -> None:
    result = evaluate_k4(
        control=0.4,
        treatment=0.2,
        clock=0.38,
        mask=0.37,
        control_shuffle=0.41,
        treatment_shuffle=0.39,
        control_unseen=0.42,
        treatment_unseen=0.24,
    )
    assert result.passed
    assert math.isclose(result.q_clock, 0.1)
    with pytest.raises(EvaluationError):
        evaluate_k4(
            control=0.2,
            treatment=0.2,
            clock=0.2,
            mask=0.2,
            control_shuffle=0.2,
            treatment_shuffle=0.2,
            control_unseen=0.2,
            treatment_unseen=0.2,
        )


def test_component_bootstrap_requires_all_nine_components() -> None:
    truth = {f"v{index}": {"p": 10.0} for index in range(9)}
    control = {f"v{index}": {"p": 12.0} for index in range(9)}
    treatment = {f"v{index}": {"p": 11.0} for index in range(9)}
    controls = {seed: control for seed in (20270815, 20270816, 20270817)}
    treatments = {seed: treatment for seed in (20270815, 20270816, 20270817)}
    endpoint = paired_component_bootstrap(
        truth,
        controls,
        treatments,
        {f"v{index}": f"c{index}" for index in range(9)},
    )
    assert endpoint.passed
    assert endpoint.draws == 10_000
    assert endpoint.relative_improvement == pytest.approx(0.5)
