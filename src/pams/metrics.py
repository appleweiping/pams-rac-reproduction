"""Counting metrics with auditable rounding and paired bootstrap intervals."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

MetricFunction = Callable[[NDArray[np.int64], NDArray[np.int64]], float]


def round_counts(values: ArrayLike) -> NDArray[np.int64]:
    """Round non-negative counts to nearest integer with ties rounded upward.

    NumPy's ``rint`` uses ties-to-even, which can silently vary from counting
    implementations based on ``floor(x + 0.5)``.  The latter is explicitly
    frozen here and used by every metric.
    """

    predictions = np.asarray(values, dtype=np.float64)
    if not np.isfinite(predictions).all():
        raise ValueError("predictions must contain only finite values")
    if np.any(predictions < 0):
        raise ValueError("predictions must be non-negative")
    return np.floor(predictions + 0.5).astype(np.int64)


def round_count(value: float) -> int:
    """Scalar form of :func:`round_counts`."""

    rounded = round_counts(np.asarray([value], dtype=np.float64))
    return int(rounded[0])


def _validated_targets(values: ArrayLike) -> NDArray[np.int64]:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size < 1:
        raise ValueError("targets must be non-empty and one-dimensional")
    numeric = np.asarray(values, dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("targets must contain only finite values")
    integer = numeric.astype(np.int64)
    if np.any(numeric != integer) or np.any(integer <= 0):
        raise ValueError("targets must be positive integers")
    return integer


def _nmae(prediction: NDArray[np.int64], target: NDArray[np.int64]) -> float:
    return float(np.mean(np.abs(prediction - target) / target))


def _mae(prediction: NDArray[np.int64], target: NDArray[np.int64]) -> float:
    return float(np.mean(np.abs(prediction - target)))


def _rmse(prediction: NDArray[np.int64], target: NDArray[np.int64]) -> float:
    return float(np.sqrt(np.mean(np.square(prediction - target, dtype=np.float64))))


def _obo(prediction: NDArray[np.int64], target: NDArray[np.int64]) -> float:
    return float(np.mean(np.abs(prediction - target) <= 1))


def _exact(prediction: NDArray[np.int64], target: NDArray[np.int64]) -> float:
    return float(np.mean(prediction == target))


_METRICS: Mapping[str, MetricFunction] = MappingProxyType(
    {
        "nmae": _nmae,
        "mae": _mae,
        "rmse": _rmse,
        "obo": _obo,
        "exact": _exact,
    }
)


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """A percentile confidence interval."""

    low: float
    high: float
    level: float = 0.95

    def __post_init__(self) -> None:
        if not all(np.isfinite(value) for value in (self.low, self.high, self.level)):
            raise ValueError("confidence interval values must be finite")
        if self.low > self.high:
            raise ValueError("confidence interval low must not exceed high")
        if not 0.0 < self.level < 1.0:
            raise ValueError("confidence interval level must be in (0, 1)")

    def to_dict(self) -> dict[str, float]:
        return {"low": self.low, "high": self.high, "level": self.level}


@dataclass(frozen=True, slots=True)
class PerVideoMetric:
    video_id: str
    prediction: float
    rounded_prediction: int
    target: int
    absolute_error: int
    normalized_absolute_error: float
    within_one: bool
    exact: bool
    action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "action": self.action,
            "prediction": self.prediction,
            "rounded_prediction": self.rounded_prediction,
            "target": self.target,
            "absolute_error": self.absolute_error,
            "normalized_absolute_error": self.normalized_absolute_error,
            "within_one": self.within_one,
            "exact": self.exact,
        }


@dataclass(frozen=True, slots=True)
class MetricReport:
    """Aggregate and per-video results for one model."""

    sample_count: int
    nmae: float
    mae: float
    rmse: float
    obo: float
    exact: float
    per_video: tuple[PerVideoMetric, ...]
    confidence_intervals: Mapping[str, ConfidenceInterval]
    bootstrap_samples: int
    bootstrap_seed: int

    def __post_init__(self) -> None:
        if self.sample_count != len(self.per_video) or self.sample_count < 1:
            raise ValueError("sample_count must equal the non-empty per_video length")
        required = set(_METRICS)
        supplied = set(self.confidence_intervals)
        if self.bootstrap_samples > 0 and supplied != required:
            raise ValueError(
                f"confidence_intervals must contain {sorted(required)}, got {sorted(supplied)}"
            )
        if self.bootstrap_samples == 0 and supplied:
            raise ValueError("confidence_intervals must be empty when bootstrap is disabled")
        object.__setattr__(
            self,
            "confidence_intervals",
            MappingProxyType(dict(self.confidence_intervals)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "nmae": self.nmae,
            "mae": self.mae,
            "rmse": self.rmse,
            "obo": self.obo,
            "exact": self.exact,
            "bootstrap_samples": self.bootstrap_samples,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_intervals": {
                name: interval.to_dict() for name, interval in self.confidence_intervals.items()
            },
            "per_video": [row.to_dict() for row in self.per_video],
        }


@dataclass(frozen=True, slots=True)
class PairedBootstrapResult:
    """Paired model-A minus model-B bootstrap comparison."""

    metric: str
    difference: float
    confidence_interval: ConfidenceInterval
    samples: int
    seed: int


def _bootstrap_metric_values(
    prediction: NDArray[np.int64],
    target: NDArray[np.int64],
    *,
    samples: int,
    seed: int,
    chunk_size: int = 512,
) -> dict[str, NDArray[np.float64]]:
    if samples < 1:
        raise ValueError("samples must be at least one")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least one")

    rng = np.random.default_rng(seed)
    count = target.size
    output = {name: np.empty(samples, dtype=np.float64) for name in _METRICS}
    absolute = np.abs(prediction - target)
    normalized = absolute / target
    squared = np.square(prediction - target, dtype=np.float64)
    within_one = absolute <= 1
    exact = absolute == 0

    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(0, count, size=(stop - start, count))
        output["nmae"][start:stop] = normalized[indices].mean(axis=1)
        output["mae"][start:stop] = absolute[indices].mean(axis=1)
        output["rmse"][start:stop] = np.sqrt(squared[indices].mean(axis=1))
        output["obo"][start:stop] = within_one[indices].mean(axis=1)
        output["exact"][start:stop] = exact[indices].mean(axis=1)
    return output


def _percentile_interval(values: NDArray[np.float64], *, level: float) -> ConfidenceInterval:
    if not 0.0 < level < 1.0:
        raise ValueError("confidence level must be in (0, 1)")
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return ConfidenceInterval(low=float(low), high=float(high), level=level)


def compute_count_metrics(
    predictions: ArrayLike,
    targets: ArrayLike,
    *,
    video_ids: Sequence[str] | None = None,
    actions: Sequence[str | None] | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 2026,
    confidence_level: float = 0.95,
) -> MetricReport:
    """Compute the frozen RAC evaluation protocol.

    All errors are computed after nearest-integer, half-up rounding.  The
    bootstrap resamples paired ``(prediction, target)`` rows, never the two
    vectors independently.
    """

    raw_predictions = np.asarray(predictions, dtype=np.float64)
    if raw_predictions.ndim != 1 or raw_predictions.size < 1:
        raise ValueError("predictions must be a non-empty one-dimensional array")
    rounded = round_counts(raw_predictions)
    target = _validated_targets(targets)
    if rounded.shape != target.shape:
        raise ValueError("predictions and targets must have the same shape")

    count = target.size
    ids = tuple(str(index) for index in range(count)) if video_ids is None else tuple(video_ids)
    if len(ids) != count or any(not str(video_id).strip() for video_id in ids):
        raise ValueError("video_ids must contain one non-empty identifier per prediction")
    if len(set(ids)) != len(ids):
        raise ValueError("video_ids must be unique")
    action_values: tuple[str | None, ...] = (
        (None,) * count if actions is None else tuple(actions)
    )
    if len(action_values) != count:
        raise ValueError("actions must contain one value per prediction")

    absolute = np.abs(rounded - target)
    rows = tuple(
        PerVideoMetric(
            video_id=str(ids[index]),
            action=(None if action_values[index] is None else str(action_values[index])),
            prediction=float(raw_predictions[index]),
            rounded_prediction=int(rounded[index]),
            target=int(target[index]),
            absolute_error=int(absolute[index]),
            normalized_absolute_error=float(absolute[index] / target[index]),
            within_one=bool(absolute[index] <= 1),
            exact=bool(absolute[index] == 0),
        )
        for index in range(count)
    )

    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be non-negative")
    intervals: dict[str, ConfidenceInterval] = {}
    if bootstrap_samples:
        distributions = _bootstrap_metric_values(
            rounded,
            target,
            samples=bootstrap_samples,
            seed=bootstrap_seed,
        )
        intervals = {
            name: _percentile_interval(values, level=confidence_level)
            for name, values in distributions.items()
        }

    return MetricReport(
        sample_count=count,
        nmae=_nmae(rounded, target),
        mae=_mae(rounded, target),
        rmse=_rmse(rounded, target),
        obo=_obo(rounded, target),
        exact=_exact(rounded, target),
        per_video=rows,
        confidence_intervals=intervals,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )


def evaluate_counts(
    predictions: ArrayLike,
    targets: ArrayLike,
    **kwargs: Any,
) -> MetricReport:
    """Compatibility alias for the public evaluation operation."""

    return compute_count_metrics(predictions, targets, **kwargs)


def paired_bootstrap_difference(
    predictions_a: ArrayLike,
    predictions_b: ArrayLike,
    targets: ArrayLike,
    *,
    metric: str = "nmae",
    samples: int = 10_000,
    seed: int = 2026,
    confidence_level: float = 0.95,
    chunk_size: int = 512,
) -> PairedBootstrapResult:
    """Estimate a paired confidence interval for ``metric(A) - metric(B)``."""

    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}; expected one of {sorted(_METRICS)}")
    prediction_a = round_counts(predictions_a)
    prediction_b = round_counts(predictions_b)
    target = _validated_targets(targets)
    if prediction_a.shape != target.shape or prediction_b.shape != target.shape:
        raise ValueError("both prediction vectors must match targets")
    if samples < 1:
        raise ValueError("samples must be at least one")
    if chunk_size < 1:
        raise ValueError("chunk_size must be at least one")

    rng = np.random.default_rng(seed)
    differences = np.empty(samples, dtype=np.float64)
    function = _METRICS[metric]
    count = target.size
    for start in range(0, samples, chunk_size):
        stop = min(start + chunk_size, samples)
        indices = rng.integers(0, count, size=(stop - start, count))
        for offset, sampled_indices in enumerate(indices):
            differences[start + offset] = function(
                prediction_a[sampled_indices], target[sampled_indices]
            ) - function(prediction_b[sampled_indices], target[sampled_indices])

    return PairedBootstrapResult(
        metric=metric,
        difference=function(prediction_a, target) - function(prediction_b, target),
        confidence_interval=_percentile_interval(differences, level=confidence_level),
        samples=samples,
        seed=seed,
    )
