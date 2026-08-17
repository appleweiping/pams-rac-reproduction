import numpy as np
import pytest

from pams.metrics import (
    compute_count_metrics,
    paired_bootstrap_difference,
    round_count,
    round_counts,
)


def test_rounding_is_nearest_integer_half_up() -> None:
    assert round_count(2.5) == 3
    assert round_count(2.499999) == 2
    np.testing.assert_array_equal(
        round_counts([0.0, 0.5, 1.5, 2.49, 2.5]),
        np.asarray([0, 1, 2, 2, 3]),
    )


def test_count_metrics_use_rounded_predictions() -> None:
    report = compute_count_metrics(
        [1.49, 2.5, 5.6],
        [1, 3, 5],
        video_ids=["a", "b", "c"],
        actions=["jump", "jump", "pushup"],
        bootstrap_samples=0,
    )
    assert report.nmae == pytest.approx(1 / 15)
    assert report.mae == pytest.approx(1 / 3)
    assert report.rmse == pytest.approx(np.sqrt(1 / 3))
    assert report.obo == 1.0
    assert report.exact == pytest.approx(2 / 3)
    assert report.per_video[1].rounded_prediction == 3
    assert report.per_video[2].normalized_absolute_error == pytest.approx(0.2)
    assert report.to_dict()["per_video"][0]["video_id"] == "a"


def test_bootstrap_is_paired_deterministic_and_defaults_to_10k() -> None:
    predictions = [4, 7, 8, 13, 20]
    targets = [5, 7, 10, 12, 20]
    first = compute_count_metrics(predictions, targets, bootstrap_samples=10_000, bootstrap_seed=42)
    second = compute_count_metrics(
        predictions, targets, bootstrap_samples=10_000, bootstrap_seed=42
    )
    assert first.bootstrap_samples == 10_000
    assert first.confidence_intervals == second.confidence_intervals
    assert first.confidence_intervals["nmae"].low <= first.nmae
    assert first.confidence_intervals["nmae"].high >= first.nmae


def test_paired_bootstrap_difference_has_expected_direction() -> None:
    target = np.arange(2, 22)
    exact = target.astype(float)
    worse = target + 3
    result = paired_bootstrap_difference(
        exact,
        worse,
        target,
        metric="nmae",
        samples=2_000,
        seed=7,
    )
    assert result.difference < 0
    assert result.confidence_interval.high < 0


@pytest.mark.parametrize(
    ("prediction", "target"),
    [
        ([1, -1], [1, 2]),
        ([1, np.nan], [1, 2]),
        ([1, 2], [0, 2]),
        ([1], [1, 2]),
    ],
)
def test_metrics_reject_invalid_vectors(prediction: list[float], target: list[float]) -> None:
    with pytest.raises(ValueError):
        compute_count_metrics(prediction, target, bootstrap_samples=0)


def test_metrics_reject_duplicate_video_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        compute_count_metrics(
            [1, 2],
            [1, 2],
            video_ids=["same", "same"],
            bootstrap_samples=0,
        )
