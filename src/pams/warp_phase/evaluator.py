"""Evaluator-only metrics for the WARP-PHASE supplied-track pilot.

This module is intentionally not imported by the training path.  It is the only
new package module allowed to consume the physically separated count/period
vault after predictions and receipts have been frozen.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


class EvaluationError(ValueError):
    """Raised when an evaluator contract would otherwise be weakened."""


NestedCounts = Mapping[str, Mapping[str, float]]

# Primary-source receipt for the small faithful port below.  The repository's
# Apache-2.0 source is intentionally not vendored into this project.
OFFICIAL_MULTIREP_EVALUATOR_COMMIT = "d15c767d82dac2f7a5df9fde1d7f6b5a4f491062"
OFFICIAL_MULTIREP_EVALUATOR_PATH = "mmdet/datasets/multirep_eval_api.py"
OFFICIAL_MULTIREP_EVALUATOR_BLOB_SHA = "e5c188aa032957fa628062df645a7e1fe8d59b6d"
OFFICIAL_MULTIREP_EVALUATOR_SHA256 = (
    "c10a3a56704d8bdf1ba55a2681570dc77371089ecb73c71c56102d26eca1386c"
)
OFFICIAL_PERIOD_TIOU_THRESHOLDS = tuple(
    float(value) for value in np.linspace(0.5, 0.95, 10)
)

NestedPeriods = Mapping[str, Sequence[Sequence[int]]]
NestedPeriodPredictions = Mapping[str, Sequence[Sequence[float]]]


@dataclass(frozen=True)
class CountEvaluation:
    """Continuous primary and half-even secondary supplied-track metrics."""

    normalized_video_first_mae: float
    avg_obo: float
    per_video_normalized_mae: dict[str, float]
    per_video_obo: dict[str, float]


@dataclass(frozen=True)
class PairedEndpoint:
    """Three-seed primary endpoint and the frozen component bootstrap CI."""

    control_mean: float
    treatment_mean: float
    absolute_effect: float
    relative_improvement: float
    ci_lower: float
    ci_upper: float
    passed: bool
    draws: int


@dataclass(frozen=True)
class HarmonicGateResult:
    """Person-first then video-mean harmonic fractions."""

    fractions: dict[str, float]
    passed: bool


@dataclass(frozen=True)
class K4Decision:
    """The four and only four frozen shortcut statistics."""

    q_clock: float
    q_mask: float
    p_shuffle: float
    p_unseen: float
    passed: bool


@dataclass(frozen=True)
class PeriodEvaluation:
    """Official-faithful supplied-track Period AP, reported in percentage points."""

    period_map: float
    ap50: float
    ap75: float
    ap_by_tiou: dict[float, float]
    scale: str = "[0,100]"


def _validated_count_keys(ground_truth: NestedCounts, prediction: NestedCounts) -> tuple[str, ...]:
    videos = tuple(sorted(ground_truth))
    if not videos or tuple(sorted(prediction)) != videos:
        raise EvaluationError("prediction and ground truth must have identical nonempty videos")
    for video in videos:
        people = tuple(sorted(ground_truth[video]))
        if not people or tuple(sorted(prediction[video])) != people:
            raise EvaluationError(f"person keys do not match for video {video!r}")
        for person in people:
            truth = float(ground_truth[video][person])
            estimate = float(prediction[video][person])
            if not math.isfinite(truth) or truth <= 0.0:
                raise EvaluationError("all evaluator counts must be finite and positive")
            if not math.isfinite(estimate):
                raise EvaluationError("all continuous predictions must be finite")
    return videos


def evaluate_counts(ground_truth: NestedCounts, prediction: NestedCounts) -> CountEvaluation:
    """Compute the frozen continuous video-first AvgMAE and secondary AvgOBO."""

    videos = _validated_count_keys(ground_truth, prediction)
    video_mae: dict[str, float] = {}
    video_obo: dict[str, float] = {}
    for video in videos:
        people = tuple(sorted(ground_truth[video]))
        normalized_errors: list[float] = []
        obo_values: list[float] = []
        for person in people:
            truth = float(ground_truth[video][person])
            estimate = float(prediction[video][person])
            normalized_errors.append(abs(estimate - truth) / truth)
            rounded = float(np.rint(np.float64(estimate)))
            obo_values.append(float(abs(rounded - truth) <= 1.0))
        video_mae[video] = float(np.mean(np.asarray(normalized_errors, dtype=np.float64)))
        video_obo[video] = float(np.mean(np.asarray(obo_values, dtype=np.float64)))
    return CountEvaluation(
        normalized_video_first_mae=float(
            np.mean(np.asarray(list(video_mae.values()), dtype=np.float64))
        ),
        avg_obo=float(np.mean(np.asarray(list(video_obo.values()), dtype=np.float64))),
        per_video_normalized_mae=video_mae,
        per_video_obo=video_obo,
    )


def _weighted_video_mean(values: Mapping[str, float], multiplicities: Mapping[str, int]) -> float:
    numerator = 0.0
    denominator = 0
    for video, value in values.items():
        multiplicity = int(multiplicities.get(video, 0))
        if multiplicity < 0:
            raise EvaluationError("component multiplicities cannot be negative")
        numerator += multiplicity * float(value)
        denominator += multiplicity
    if denominator <= 0:
        raise EvaluationError("bootstrap draw contains no videos")
    return numerator / denominator


def paired_component_bootstrap(
    ground_truth: NestedCounts,
    control_by_seed: Mapping[int, NestedCounts],
    treatment_by_seed: Mapping[int, NestedCounts],
    video_component: Mapping[str, str],
    *,
    seeds: Sequence[int] = (20270815, 20270816, 20270817),
    draws: int = 10_000,
    bootstrap_seed: int = 20270815,
) -> PairedEndpoint:
    """Apply the frozen all-nine-component paired bootstrap without clipping."""

    frozen_seeds = tuple(int(seed) for seed in seeds)
    if frozen_seeds != (20270815, 20270816, 20270817):
        raise EvaluationError("primary endpoint requires the three frozen seeds in order")
    if draws != 10_000 or bootstrap_seed != 20270815:
        raise EvaluationError("primary endpoint requires 10,000 PCG64(20270815) draws")
    if set(control_by_seed) != set(frozen_seeds) or set(treatment_by_seed) != set(frozen_seeds):
        raise EvaluationError("both arms must contain exactly the three frozen seeds")

    controls: dict[int, CountEvaluation] = {}
    treatments: dict[int, CountEvaluation] = {}
    for seed in frozen_seeds:
        controls[seed] = evaluate_counts(ground_truth, control_by_seed[seed])
        treatments[seed] = evaluate_counts(ground_truth, treatment_by_seed[seed])
    control_mean = float(
        np.mean(
            np.asarray(
                [controls[seed].normalized_video_first_mae for seed in frozen_seeds],
                dtype=np.float64,
            )
        )
    )
    treatment_mean = float(
        np.mean(
            np.asarray(
                [treatments[seed].normalized_video_first_mae for seed in frozen_seeds],
                dtype=np.float64,
            )
        )
    )
    if not math.isfinite(control_mean) or control_mean == 0.0:
        raise EvaluationError("zero or non-finite control mean fails K1")
    effect = control_mean - treatment_mean
    relative = effect / control_mean

    videos = tuple(sorted(ground_truth))
    if set(video_component) != set(videos):
        raise EvaluationError("every and only evaluated video must have one source component")
    components = tuple(sorted(set(video_component.values())))
    if len(components) != 9:
        raise EvaluationError("the frozen bootstrap requires exactly nine nonempty components")
    component_videos = {
        component: tuple(video for video in videos if video_component[video] == component)
        for component in components
    }
    if any(not member_videos for member_videos in component_videos.values()):
        raise EvaluationError("every frozen component must remain nonempty")

    rng = np.random.Generator(np.random.PCG64(bootstrap_seed))
    bootstrap_effect = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        selected_indices = rng.integers(0, len(components), size=len(components), dtype=np.int64)
        component_counts = np.bincount(selected_indices, minlength=len(components))
        multiplicities: dict[str, int] = {}
        for index, component in enumerate(components):
            for video in component_videos[component]:
                multiplicities[video] = int(component_counts[index])
        per_seed_effect: list[float] = []
        for seed in frozen_seeds:
            control_draw = _weighted_video_mean(
                controls[seed].per_video_normalized_mae,
                multiplicities,
            )
            treatment_draw = _weighted_video_mean(
                treatments[seed].per_video_normalized_mae,
                multiplicities,
            )
            per_seed_effect.append(control_draw - treatment_draw)
        bootstrap_effect[draw] = float(np.mean(np.asarray(per_seed_effect, dtype=np.float64)))
    lower, upper = np.quantile(
        bootstrap_effect,
        np.asarray([0.025, 0.975], dtype=np.float64),
        method="linear",
    )
    passed = bool(relative >= 0.05 and float(lower) > 0.0)
    return PairedEndpoint(
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        absolute_effect=effect,
        relative_improvement=relative,
        ci_lower=float(lower),
        ci_upper=float(upper),
        passed=passed,
        draws=draws,
    )


def _validated_period_rows(
    ground_truth: NestedPeriods,
    prediction: NestedPeriodPredictions,
) -> tuple[list[tuple[str, float, float]], list[tuple[str, float, float, float]]]:
    if not ground_truth:
        raise EvaluationError("Period AP requires at least one supplied track")
    unknown_tracks = set(prediction).difference(ground_truth)
    if unknown_tracks:
        raise EvaluationError("predictions contain identities outside the supplied GT tracks")

    gt_rows: list[tuple[str, float, float]] = []
    prediction_rows: list[tuple[str, float, float, float]] = []
    for track_id, intervals in ground_truth.items():
        if not isinstance(track_id, str) or not track_id:
            raise EvaluationError("supplied track identities must be nonempty strings")
        if not intervals:
            raise EvaluationError("every supplied track must contain a GT period")
        previous_end = -1
        for interval in intervals:
            if isinstance(interval, (str, bytes)) or len(interval) != 2:
                raise EvaluationError("each GT period must contain exactly [start,end]")
            start, end = interval
            if (
                isinstance(start, bool)
                or isinstance(end, bool)
                or not isinstance(start, int)
                or not isinstance(end, int)
            ):
                raise EvaluationError("GT period boundaries must be Python integers")
            if not 0 <= start < end:
                raise EvaluationError("GT periods must be positive half-open source intervals")
            if start < previous_end:
                raise EvaluationError("GT periods must be sorted and non-overlapping")
            previous_end = end
            gt_rows.append((track_id, float(start), float(end)))

        # Flatten in GT track order, as the official adapter does after spatial
        # instance matching.  This order is observable when scores are tied.
        for proposal in prediction.get(track_id, ()):
            if isinstance(proposal, (str, bytes)) or len(proposal) != 3:
                raise EvaluationError("each period prediction must be [start,end,score]")
            start_value, end_value, score_value = proposal
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating))
                for value in proposal
            ):
                raise EvaluationError("period predictions must contain real numbers")
            prediction_start = float(start_value)
            prediction_end = float(end_value)
            score = float(score_value)
            if not all(
                math.isfinite(value) for value in (prediction_start, prediction_end, score)
            ):
                raise EvaluationError("period predictions must be finite")
            if not 0.0 <= prediction_start < prediction_end:
                raise EvaluationError("predictions must be positive half-open source intervals")
            prediction_rows.append((track_id, prediction_start, prediction_end, score))
    return gt_rows, prediction_rows


def _temporal_iou(
    prediction: tuple[float, float],
    candidates: Sequence[tuple[float, float]],
) -> np.ndarray:
    candidate_array = np.asarray(candidates, dtype=np.float64)
    intersection = np.minimum(prediction[1], candidate_array[:, 1]) - np.maximum(
        prediction[0], candidate_array[:, 0]
    )
    intersection = np.clip(intersection, 0.0, None)
    union = (
        candidate_array[:, 1]
        - candidate_array[:, 0]
        + prediction[1]
        - prediction[0]
        - intersection
    )
    return intersection / union


def _interpolated_ap(precision: np.ndarray, recall: np.ndarray) -> float:
    if precision.size == 0:
        return 0.0
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_increments = np.diff(np.concatenate((np.asarray([0.0]), recall)))
    return float(np.sum(recall_increments * precision_envelope))


def evaluate_supplied_track_period_ap(
    ground_truth: NestedPeriods,
    prediction: NestedPeriodPredictions,
) -> PeriodEvaluation:
    """Evaluate periods after supplied identities have already been paired.

    This is a narrow adapter around the temporal portion of the official
    MultiCounter+ evaluator pinned by the constants above.  The official
    end-to-end path first performs spatiotemporal person-detection matching at
    instance IoU 0.5, drops unmatched person instances, and then assigns a
    shared ``gt_ID`` to the two period lists.  Here the mapping key supplies
    that identity, so unknown prediction identities fail closed; this function
    does not claim parity for detection, tracking, area/category selection, or
    the official unmatched-person filtering.

    Within each supplied identity, temporal behavior is faithful: predictions
    are globally ordered with the official NumPy default ``argsort()[::-1]``;
    candidate GT periods use the same tIoU ordering; equality passes a
    threshold; each GT can be locked once per threshold; duplicate or
    otherwise unmatched proposals are false positives; and missed GT periods
    remain in the recall denominator.  Input row order therefore deliberately
    resolves exact score and tIoU ties exactly as in the pinned implementation.
    The official AP values are fractions; the returned Period-mAP, AP50, AP75,
    and per-threshold AP values are multiplied by 100 onto the frozen [0,100]
    reporting scale.
    """

    gt_rows, prediction_rows = _validated_period_rows(ground_truth, prediction)
    thresholds = np.asarray(OFFICIAL_PERIOD_TIOU_THRESHOLDS, dtype=np.float64)
    true_positive = np.zeros((len(thresholds), len(prediction_rows)), dtype=np.float64)
    false_positive = np.zeros_like(true_positive)
    locked_gt = np.full((len(thresholds), len(gt_rows)), -1, dtype=np.int64)

    gt_by_track: dict[str, list[int]] = {}
    for gt_index, (track_id, _, _) in enumerate(gt_rows):
        gt_by_track.setdefault(track_id, []).append(gt_index)

    scores = np.asarray([row[3] for row in prediction_rows], dtype=np.float64)
    prediction_order = scores.argsort()[::-1]
    for rank, prediction_index in enumerate(prediction_order):
        track_id, start, end, _ = prediction_rows[int(prediction_index)]
        gt_indices = gt_by_track[track_id]
        candidates = [(gt_rows[index][1], gt_rows[index][2]) for index in gt_indices]
        overlaps = _temporal_iou((start, end), candidates)
        candidate_order = overlaps.argsort()[::-1]
        for threshold_index, threshold in enumerate(thresholds):
            for candidate_position in candidate_order:
                position = int(candidate_position)
                if overlaps[position] < threshold:
                    break
                gt_index = gt_indices[position]
                if locked_gt[threshold_index, gt_index] >= 0:
                    continue
                true_positive[threshold_index, rank] = 1.0
                locked_gt[threshold_index, gt_index] = rank
                break
            if true_positive[threshold_index, rank] == 0.0:
                false_positive[threshold_index, rank] = 1.0

    true_positive = np.cumsum(true_positive, axis=1)
    false_positive = np.cumsum(false_positive, axis=1)
    recall = true_positive / float(len(gt_rows))
    precision = true_positive / (true_positive + false_positive)
    ap_fraction = np.asarray(
        [
            _interpolated_ap(precision[index], recall[index])
            for index in range(len(thresholds))
        ],
        dtype=np.float64,
    )
    ap_percent = ap_fraction * 100.0
    per_threshold = {
        float(threshold): float(ap_percent[index])
        for index, threshold in enumerate(thresholds)
    }
    return PeriodEvaluation(
        period_map=float(np.mean(ap_fraction) * 100.0),
        ap50=float(ap_percent[0]),
        ap75=float(ap_percent[5]),
        ap_by_tiou=per_threshold,
    )


def period_median(periods: Sequence[Sequence[int]], source_length: int) -> float:
    """Validate raw integer source-boundary periods and return the sole P_eval."""

    if isinstance(source_length, bool) or not isinstance(source_length, int) or source_length <= 0:
        raise EvaluationError("source_length must be a positive integer")
    if not periods:
        raise EvaluationError("period list must be nonempty")
    durations: list[int] = []
    previous_end = -1
    for entry in periods:
        if len(entry) != 2:
            raise EvaluationError("each period must contain exactly [start,end]")
        start, end = entry
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise EvaluationError("period boundaries must be Python integers")
        if not (0 <= start < end <= source_length):
            raise EvaluationError("period boundaries must satisfy 0 <= start < end <= L")
        if start < previous_end:
            raise EvaluationError("periods must be sorted and non-overlapping")
        previous_end = end
        durations.append(end - start)
    return float(np.median(np.asarray(durations, dtype=np.float64)))


def classify_harmonic(selector_period: float, evaluator_period: float) -> str:
    """Classify a frozen selector period without correction or fallback."""

    if not math.isfinite(selector_period) or selector_period <= 0.0:
        raise EvaluationError("selector period must be positive and finite")
    if not math.isfinite(evaluator_period) or evaluator_period <= 0.0:
        raise EvaluationError("evaluator period must be positive and finite")
    matches = [
        label
        for harmonic, label in ((0.5, "half"), (1.0, "fundamental"), (2.0, "double"))
        if abs(selector_period / (harmonic * evaluator_period) - 1.0) <= 0.10
    ]
    if len(matches) > 1:
        raise EvaluationError("harmonic classification is not unique")
    return matches[0] if matches else "off_grid"


def evaluate_harmonic_gate(
    video_people: Mapping[str, Mapping[str, tuple[float, float]]],
) -> HarmonicGateResult:
    """Compute person fractions within video, then unweighted video means."""

    if not video_people:
        raise EvaluationError("harmonic gate requires at least one video")
    labels = ("half", "fundamental", "double", "off_grid")
    per_video: dict[str, dict[str, float]] = {}
    for video, people in sorted(video_people.items()):
        if not people:
            raise EvaluationError(f"harmonic gate video {video!r} has no people")
        counts = {label: 0 for label in labels}
        for selector_period, evaluator_period in people.values():
            counts[classify_harmonic(float(selector_period), float(evaluator_period))] += 1
        per_video[video] = {
            label: counts[label] / len(people)
            for label in labels
        }
    fractions = {
        label: float(
            np.mean(np.asarray([row[label] for row in per_video.values()], dtype=np.float64))
        )
        for label in labels
    }
    passed = bool(
        fractions["fundamental"] >= 0.90
        and fractions["half"] <= 0.05
        and fractions["double"] <= 0.05
        and fractions["off_grid"] <= 0.05
    )
    return HarmonicGateResult(fractions=fractions, passed=passed)


def evaluate_k4(
    *,
    control: float,
    treatment: float,
    clock: float,
    mask: float,
    control_shuffle: float,
    treatment_shuffle: float,
    control_unseen: float,
    treatment_unseen: float,
) -> K4Decision:
    """Apply the exact four frozen K4 ratios, with no clipping or epsilon."""

    values = (
        control,
        treatment,
        clock,
        mask,
        control_shuffle,
        treatment_shuffle,
        control_unseen,
        treatment_unseen,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise EvaluationError("all K4 inputs must be finite")
    gain = float(control) - float(treatment)
    if gain <= 0.0:
        raise EvaluationError("Stage B is unreachable when the primary gain is not positive")
    q_clock = (float(control) - float(clock)) / gain
    q_mask = (float(control) - float(mask)) / gain
    p_shuffle = (float(control_shuffle) - float(treatment_shuffle)) / gain
    p_unseen = (float(control_unseen) - float(treatment_unseen)) / gain
    passed = bool(
        q_clock <= 0.20
        and q_mask <= 0.20
        and p_shuffle <= 0.20
        and p_unseen >= 0.80
    )
    return K4Decision(
        q_clock=q_clock,
        q_mask=q_mask,
        p_shuffle=p_shuffle,
        p_unseen=p_unseen,
        passed=passed,
    )
