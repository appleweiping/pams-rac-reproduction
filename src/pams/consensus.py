"""Deterministic three-expert peak consensus for repetition counting."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks
from torch import Tensor

if TYPE_CHECKING:
    from pams.types import CountResult

ExpertName = Literal["fast", "medium", "slow"]
ExpertMode = Literal["multi", "medium_only"]


@dataclass(frozen=True)
class ExpertConfig:
    """Smoothing and minimum-distance multipliers for one peak expert."""

    name: ExpertName
    sigma_multiplier: float
    distance_multiplier: float

    def __post_init__(self) -> None:
        if self.sigma_multiplier <= 0 or self.distance_multiplier <= 0:
            raise ValueError("expert multipliers must be positive")


@dataclass(frozen=True)
class ExpertResult:
    """Auditable output of an individual peak expert."""

    name: ExpertName
    count: int
    peaks: tuple[int, ...]
    smoothed: NDArray[np.float64]
    threshold: NDArray[np.float64]
    prominence: float
    minimum_distance: int


@dataclass(frozen=True)
class ConsensusResult:
    """Final count plus all information used to choose it."""

    count: int
    period_frames: float
    reference_count: int
    expert_counts: tuple[int, int, int]
    selected_expert: ExpertName
    selection_mode: ExpertMode
    confidence: float
    period_stream: tuple[float, ...]
    experts: tuple[ExpertResult, ExpertResult, ExpertResult]

    def to_count_result(self) -> CountResult:
        """Convert to the package's compact public ``CountResult`` type.

        The local import keeps this numerical module usable in isolation.
        """

        from pams.types import CountResult

        return CountResult(
            count=self.count,
            period_frames=self.period_frames,
            expert_counts=self.expert_counts,
            confidence=self.confidence,
            period_stream=np.asarray(self.period_stream, dtype=np.float32),
        )


DEFAULT_EXPERTS = (
    ExpertConfig("fast", sigma_multiplier=0.05, distance_multiplier=0.5),
    ExpertConfig("medium", sigma_multiplier=0.12, distance_multiplier=0.8),
    ExpertConfig("slow", sigma_multiplier=0.15, distance_multiplier=1.2),
)


def _to_numpy_1d(values: Tensor | NDArray[np.floating] | list[float]) -> NDArray[np.float64]:
    array = (
        values.detach().cpu().double().numpy() if isinstance(values, Tensor) else np.asarray(values)
    )
    if array.ndim != 1:
        raise ValueError("period_stream must be one-dimensional")
    return np.nan_to_num(array.astype(np.float64, copy=True))


def _rolling_statistics(
    values: NDArray[np.float64],
    window: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    size = max(1, min(int(window), len(values)))
    mean = uniform_filter1d(values, size=size, mode="nearest")
    square_mean = uniform_filter1d(values * values, size=size, mode="nearest")
    variance = np.maximum(square_mean - mean * mean, 0.0)
    return mean, np.sqrt(variance)


def dynamic_threshold(
    values: NDArray[np.float64],
    period_frames: float,
    *,
    short_window_multiplier: float = 0.5,
    long_window_multiplier: float = 2.0,
    height_factor: float = 0.6,
    long_window_weight: float = 0.6,
) -> NDArray[np.float64]:
    """Compute the disclosed short/long-window adaptive peak threshold."""

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if period_frames <= 0:
        raise ValueError("period_frames must be positive")
    if not 0.0 <= long_window_weight <= 1.0:
        raise ValueError("long_window_weight must be in [0, 1]")
    if short_window_multiplier <= 0 or long_window_multiplier <= 0:
        raise ValueError("window multipliers must be positive")

    short_window = max(1, int(round(short_window_multiplier * period_frames)))
    long_window = max(1, int(round(long_window_multiplier * period_frames)))
    short_mean, short_std = _rolling_statistics(values, short_window)
    long_mean, long_std = _rolling_statistics(values, long_window)
    short_threshold = short_mean + height_factor * short_std
    long_threshold = long_mean + height_factor * long_std
    return long_window_weight * long_threshold + (1.0 - long_window_weight) * short_threshold


def vote_expert_counts(
    counts: tuple[int, int, int],
    reference_count: int,
) -> tuple[int, int, float]:
    """Return ``(count, selected_index, confidence)`` deterministically.

    Medium is preferred when several experts are equally eligible, followed
    by fast and slow.  A majority vote has confidence ``votes / 3``; the
    fallback confidence is penalized by distance from the FFT reference.
    """

    frequencies = Counter(counts)
    majority_count, votes = max(
        frequencies.items(),
        key=lambda item: (item[1], -abs(item[0] - reference_count), -item[0]),
    )
    priority = (1, 0, 2)
    if votes >= 2:
        selected = next(index for index in priority if counts[index] == majority_count)
        return majority_count, selected, votes / 3.0

    best_distance = min(abs(value - reference_count) for value in counts)
    selected = next(
        index for index in priority if abs(counts[index] - reference_count) == best_distance
    )
    confidence = (1.0 / 3.0) / (1.0 + best_distance)
    return counts[selected], selected, confidence


class MultiExpertCounter:
    """Gaussian smoothing, adaptive thresholds, and three-expert voting."""

    def __init__(
        self,
        experts: tuple[ExpertConfig, ExpertConfig, ExpertConfig] | None = None,
        *,
        sigma_multipliers: tuple[float, float, float] = (0.05, 0.12, 0.15),
        distance_multipliers: tuple[float, float, float] = (0.5, 0.8, 1.2),
        short_window_multiplier: float = 0.5,
        long_window_multiplier: float = 2.0,
        height_factor: float = 0.6,
        prominence_factor: float = 0.25,
        long_window_weight: float = 0.6,
        expert_mode: ExpertMode = "multi",
    ) -> None:
        if experts is None:
            if len(sigma_multipliers) != 3 or len(distance_multipliers) != 3:
                raise ValueError("sigma and distance multipliers must have length three")
            experts = (
                ExpertConfig("fast", sigma_multipliers[0], distance_multipliers[0]),
                ExpertConfig("medium", sigma_multipliers[1], distance_multipliers[1]),
                ExpertConfig("slow", sigma_multipliers[2], distance_multipliers[2]),
            )
        if len(experts) != 3:
            raise ValueError("exactly three experts are required")
        if tuple(expert.name for expert in experts) != ("fast", "medium", "slow"):
            raise ValueError("experts must be ordered fast, medium, slow")
        if height_factor < 0 or prominence_factor < 0:
            raise ValueError("height and prominence factors must be non-negative")
        if expert_mode not in {"multi", "medium_only"}:
            raise ValueError("expert_mode must be 'multi' or 'medium_only'")
        self.experts = experts
        self.short_window_multiplier = short_window_multiplier
        self.long_window_multiplier = long_window_multiplier
        self.height_factor = height_factor
        self.prominence_factor = prominence_factor
        self.long_window_weight = long_window_weight
        self.expert_mode = expert_mode

    def count(
        self,
        period_stream: Tensor | NDArray[np.floating] | list[float],
        period_frames: float,
        valid_mask: Tensor | NDArray[np.bool_] | list[bool] | None = None,
        *,
        period_confidence: float = 1.0,
        reference_frames: int | None = None,
        reference_count_override: int | None = None,
    ) -> ConsensusResult:
        """Count peaks without consulting an action label or ground-truth count.

        ``reference_frames`` only changes the label-free FFT fallback reference;
        it never changes any expert's smoothing, peak detection, or count.
        ``reference_count_override`` is reserved for independently inferred
        readouts whose analytic reference and experts share an identical
        target-free active mask.  It is mutually exclusive with
        ``reference_frames`` and never changes an expert result.
        """

        if period_frames <= 0 or not np.isfinite(period_frames):
            raise ValueError("period_frames must be finite and positive")
        if not np.isfinite(period_confidence) or not 0.0 <= period_confidence <= 1.0:
            raise ValueError("period_confidence must be finite and in [0, 1]")
        if reference_frames is not None:
            if (
                isinstance(reference_frames, bool | np.bool_)
                or not isinstance(reference_frames, int | np.integer)
                or reference_frames < 1
            ):
                raise ValueError("reference_frames must be a positive integer or None")
            reference_frames = int(reference_frames)
        if reference_count_override is not None:
            if reference_frames is not None:
                raise ValueError(
                    "reference_count_override and reference_frames are mutually exclusive"
                )
            if (
                isinstance(reference_count_override, bool | np.bool_)
                or not isinstance(reference_count_override, int | np.integer)
                or reference_count_override < 0
            ):
                raise ValueError(
                    "reference_count_override must be a non-negative integer or None"
                )
            reference_count_override = int(reference_count_override)
        original = _to_numpy_1d(period_stream)
        if original.size == 0:
            raise ValueError("period_stream must contain at least one frame")
        if reference_frames is not None and reference_frames > original.size:
            raise ValueError("reference_frames cannot exceed the period stream length")
        if valid_mask is None:
            valid = np.ones(len(original), dtype=bool)
        elif isinstance(valid_mask, Tensor):
            valid = valid_mask.detach().cpu().numpy().astype(bool)
        else:
            valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != original.shape:
            raise ValueError("valid_mask must match period_stream")
        if (
            reference_frames is not None
            and reference_frames < original.size
            and bool(valid[reference_frames:].any())
        ):
            raise ValueError(
                "valid_mask cannot mark samples beyond reference_frames as valid"
            )

        valid_indices = np.flatnonzero(valid)
        if not len(valid_indices):
            empty_experts = tuple(
                ExpertResult(
                    name=expert.name,
                    count=0,
                    peaks=(),
                    smoothed=np.zeros_like(original),
                    threshold=np.full_like(original, np.inf),
                    prominence=0.0,
                    minimum_distance=max(1, int(round(expert.distance_multiplier * period_frames))),
                )
                for expert in self.experts
            )
            return ConsensusResult(
                count=0,
                period_frames=float(period_frames),
                reference_count=0,
                expert_counts=(0, 0, 0),
                selected_expert="medium",
                selection_mode=self.expert_mode,
                confidence=0.0,
                period_stream=tuple(float(value) for value in original),
                experts=empty_experts,  # type: ignore[arg-type]
            )

        run_starts = valid_indices[np.concatenate((np.asarray([True]), np.diff(valid_indices) > 1))]
        run_stops = (
            valid_indices[np.concatenate((np.diff(valid_indices) > 1, np.asarray([True])))] + 1
        )
        valid_runs = tuple(
            (int(start), int(stop)) for start, stop in zip(run_starts, run_stops, strict=True)
        )

        expert_results: list[ExpertResult] = []
        for expert in self.experts:
            sigma = max(1e-6, expert.sigma_multiplier * period_frames)
            minimum_distance = max(1, int(round(expert.distance_multiplier * period_frames)))
            smoothed = np.zeros_like(original)
            threshold = np.full_like(original, np.inf)
            all_peaks: list[int] = []
            run_prominences: list[float] = []
            for start, stop in valid_runs:
                working = original[start:stop]
                smoothed_working = gaussian_filter1d(working, sigma=sigma, mode="nearest")
                threshold_working = dynamic_threshold(
                    smoothed_working,
                    period_frames,
                    short_window_multiplier=self.short_window_multiplier,
                    long_window_multiplier=self.long_window_multiplier,
                    height_factor=self.height_factor,
                    long_window_weight=self.long_window_weight,
                )
                signal_range = float(np.ptp(smoothed_working))
                prominence = self.prominence_factor * signal_range
                peaks, _ = find_peaks(
                    smoothed_working,
                    height=threshold_working,
                    prominence=prominence,
                    distance=minimum_distance,
                )
                all_peaks.extend(int(peak + start) for peak in peaks)
                run_prominences.append(prominence)
                smoothed[start:stop] = smoothed_working
                threshold[start:stop] = threshold_working
            expert_results.append(
                ExpertResult(
                    name=expert.name,
                    count=len(all_peaks),
                    peaks=tuple(all_peaks),
                    smoothed=smoothed,
                    threshold=threshold,
                    prominence=max(run_prominences, default=0.0),
                    minimum_distance=minimum_distance,
                )
            )

        counts = (
            expert_results[0].count,
            expert_results[1].count,
            expert_results[2].count,
        )
        reference_length = valid.sum() if reference_frames is None else reference_frames
        reference_count = (
            int(np.floor(reference_length / period_frames))
            if reference_count_override is None
            else reference_count_override
        )
        if self.expert_mode == "medium_only":
            count = counts[1]
            selected_index = 1
            vote_confidence = 1.0
        else:
            count, selected_index, vote_confidence = vote_expert_counts(
                counts,
                reference_count,
            )
        return ConsensusResult(
            count=count,
            period_frames=float(period_frames),
            reference_count=reference_count,
            expert_counts=counts,
            selected_expert=self.experts[selected_index].name,
            selection_mode=self.expert_mode,
            confidence=float(vote_confidence * period_confidence),
            period_stream=tuple(float(value) for value in original),
            experts=(expert_results[0], expert_results[1], expert_results[2]),
        )
