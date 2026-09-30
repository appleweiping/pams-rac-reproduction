"""Synthetic-frozen, label-free local-frequency readout.

This module is an independently inferred diagnostic completion.  It is not
author-disclosed PAMS logic and is never eligible for a paper-comparison
table.  Runtime prediction accepts only :class:`~pams.types.PoseSequence`;
the sole parameter-selection helper constructs its own deterministic
synthetic samples and therefore cannot consume UCFRep labels.

For this adapter, ``CountResult.expert_counts`` records the rounded arithmetic
mean, median, and confidence-weighted mean of the local frequency ridge.  Its
``period_stream`` records that ridge in cycles per frame, interpolated to the
input frame grid.  These are inferred readout diagnostics, not the disclosed
fast/medium/slow peak experts or the learned Period Head stream.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import Field, model_validator

from pams.config import StrictModel
from pams.evaluation import PredictionRecord
from pams.metrics import round_count
from pams.synthetic import generate_count_sweep, synthetic_stress_suite
from pams.types import CountResult, PoseSequence

Representation = Literal["xyz", "frame_centered"]
Aggregation = Literal["mean", "median", "confidence_weighted"]

_AGGREGATIONS: tuple[Aggregation, Aggregation, Aggregation] = (
    "mean",
    "median",
    "confidence_weighted",
)


class CandidateGrid(StrictModel):
    """Small, auditable candidate grid searched only on generated samples."""

    representations: tuple[Representation, ...]
    minimum_window_frames: tuple[int, ...]
    aggregations: tuple[Aggregation, ...]

    @model_validator(mode="after")
    def validate_grid(self) -> CandidateGrid:
        for name, values in (
            ("representations", self.representations),
            ("minimum_window_frames", self.minimum_window_frames),
            ("aggregations", self.aggregations),
        ):
            if not values:
                raise ValueError(f"selection.candidate_grid.{name} must be non-empty")
            if len(values) != len(set(values)):
                raise ValueError(f"selection.candidate_grid.{name} must be unique")
        if any(value < 8 for value in self.minimum_window_frames):
            raise ValueError("candidate minimum_window_frames must be at least eight")
        return self


class ExpectedSyntheticReplay(StrictModel):
    """Frozen integer-metric outcome of the synthetic-only search."""

    selected_candidate: str
    sample_count: int = Field(ge=1)
    mean_absolute_error: float = Field(ge=0)
    normalized_mean_absolute_error: float = Field(ge=0)
    exact_rate: float = Field(ge=0, le=1)
    obo: float = Field(ge=0, le=1)


class SyntheticSelectionProtocol(StrictModel):
    """Exact generators and label firewall used for parameter selection."""

    label_source: Literal["synthetic_generation_truth_only"]
    prohibited_label_sources: tuple[str, ...]
    scope_config: Literal["configs/stress.yaml"]
    scope_config_sha256: str
    count_sweep_minimum: Literal[2] = 2
    count_sweep_maximum: Literal[40] = 40
    frames: Literal[256] = 256
    count_sweep_seed: Literal[2026] = 2026
    stress_count: Literal[8] = 8
    stress_seed: Literal[2026] = 2026
    excluded_duplicate_stress_cases: tuple[Literal["clean"], ...] = ("clean",)
    objective_order: tuple[
        Literal["mean_absolute_error"],
        Literal["negative_exact_rate"],
        Literal["negative_obo"],
        Literal["normalized_mean_absolute_error"],
        Literal["candidate_key"],
    ]
    candidate_grid: CandidateGrid
    expected_replay: ExpectedSyntheticReplay

    @model_validator(mode="after")
    def validate_selection_firewall(self) -> SyntheticSelectionProtocol:
        digest = self.scope_config_sha256.lower()
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("selection.scope_config_sha256 must be a lowercase SHA-256")
        if not self.prohibited_label_sources:
            raise ValueError("selection.prohibited_label_sources must be explicit")
        return self


class LocalFrequencyParameters(StrictModel):
    """Frozen numerical choices used during label-free prediction."""

    representation: Representation
    minimum_period: int = Field(ge=2)
    maximum_period: int = Field(ge=3)
    adaptive_window_cycles: float = Field(gt=0)
    minimum_window_frames: int = Field(ge=8)
    window_step_fraction: float = Field(gt=0, le=1)
    minimum_step_frames: int = Field(ge=1)
    minimum_fft_size: int = Field(ge=16)
    harmonic_weight: float = Field(ge=0, le=1)
    aggregation: Aggregation
    duration_convention: Literal["frame_intervals_T_minus_1"]

    @model_validator(mode="after")
    def validate_numerics(self) -> LocalFrequencyParameters:
        if self.maximum_period <= self.minimum_period:
            raise ValueError("readout.maximum_period must exceed minimum_period")
        if self.minimum_fft_size & (self.minimum_fft_size - 1):
            raise ValueError("readout.minimum_fft_size must be a power of two")
        return self


class LocalFrequencyConfig(StrictModel):
    """Strict identity for the inferred local-frequency readout."""

    schema_version: Literal[1] = 1
    key: Literal["pams-local-frequency-synthetic-v1"]
    classification: Literal["inferred synthetic-frozen label-free readout"]
    eligible_for_paper_table: Literal[False] = False
    selection: SyntheticSelectionProtocol
    readout: LocalFrequencyParameters

    @model_validator(mode="after")
    def validate_selected_candidate(self) -> LocalFrequencyConfig:
        grid = self.selection.candidate_grid
        if self.readout.representation not in grid.representations:
            raise ValueError("selected representation is absent from the candidate grid")
        if self.readout.minimum_window_frames not in grid.minimum_window_frames:
            raise ValueError("selected minimum window is absent from the candidate grid")
        if self.readout.aggregation not in grid.aggregations:
            raise ValueError("selected aggregation is absent from the candidate grid")
        if self.selected_candidate != self.selection.expected_replay.selected_candidate:
            raise ValueError("readout parameters do not match expected_replay.selected_candidate")
        return self

    @property
    def selected_candidate(self) -> str:
        """Return the stable key used by the synthetic selection ranking."""

        return _candidate_key(
            self.readout.representation,
            self.readout.minimum_window_frames,
            self.readout.aggregation,
        )

    @property
    def fingerprint(self) -> str:
        """Hash the strict semantic config independently of YAML formatting."""

        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _find_repository_root(config_path: Path, scope_config: str) -> Path:
    for candidate_root in (config_path.parent, *config_path.parents):
        if (candidate_root / scope_config).is_file():
            return candidate_root
    raise ValueError(
        f"cannot locate selection scope config {scope_config!r} above {config_path}"
    )


def verify_selection_scope(
    config: LocalFrequencyConfig,
    repository_root: str | Path,
) -> Path:
    """Verify the frozen stress-policy bytes before selection or prediction."""

    root = Path(repository_root)
    scope_path = root / config.selection.scope_config
    if not scope_path.is_file():
        raise ValueError(f"selection scope config does not exist: {scope_path}")
    actual = normalized_file_sha256(scope_path)
    expected = config.selection.scope_config_sha256
    if actual != expected:
        raise ValueError(
            "selection scope config SHA-256 mismatch: "
            f"expected={expected}, actual={actual}, path={scope_path}"
        )
    return scope_path


def load_local_frequency_config(
    path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> LocalFrequencyConfig:
    """Load strict parameters and verify their synthetic stress-policy hash."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    config = LocalFrequencyConfig.model_validate(raw)
    root = (
        Path(repository_root)
        if repository_root is not None
        else _find_repository_root(config_path.resolve(), config.selection.scope_config)
    )
    verify_selection_scope(config, root)
    return config


def normalized_file_sha256(path: str | Path) -> str:
    """Hash UTF-8 text with platform-independent LF newlines."""

    text = Path(path).read_text(encoding="utf-8")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _SpectralPeak:
    frequency: float
    confidence: float


def _next_fft_size(minimum: int, length: int) -> int:
    required = max(minimum, length)
    return 1 << (required - 1).bit_length()


def _prepare_signal(
    sequence: PoseSequence,
    representation: Representation,
) -> NDArray[np.float64]:
    xyz = np.asarray(sequence.xyz, dtype=np.float64).copy()
    valid = np.asarray(sequence.valid_mask, dtype=np.bool_)
    valid_indices = np.flatnonzero(valid)
    if 1 < valid_indices.size < sequence.num_frames:
        missing_indices = np.flatnonzero(~valid)
        flattened = xyz.reshape(sequence.num_frames, -1)
        for coordinate in range(flattened.shape[1]):
            flattened[missing_indices, coordinate] = np.interp(
                missing_indices,
                valid_indices,
                flattened[valid_indices, coordinate],
            )
    elif valid_indices.size <= 1:
        xyz.fill(0.0)

    if representation == "frame_centered":
        xyz -= xyz.mean(axis=1, keepdims=True)
    return xyz.reshape(sequence.num_frames, -1)


def _spectral_peak(
    signal: NDArray[np.float64],
    parameters: LocalFrequencyParameters,
) -> _SpectralPeak:
    centered = signal - signal.mean(axis=0, keepdims=True)
    if not np.any(np.ptp(centered, axis=0) > 1e-12):
        return _SpectralPeak(frequency=0.0, confidence=0.0)

    windowed = centered * np.hanning(centered.shape[0])[:, None]
    fft_size = _next_fft_size(parameters.minimum_fft_size, centered.shape[0])
    spectrum = np.fft.rfft(windowed, n=fft_size, axis=0)
    power = np.sum(np.abs(spectrum) ** 2, axis=1)
    frequencies = np.fft.rfftfreq(fft_size)
    allowed = (frequencies >= 1.0 / parameters.maximum_period) & (
        frequencies <= 1.0 / parameters.minimum_period
    )
    indices = np.flatnonzero(allowed)
    if not indices.size:
        return _SpectralPeak(frequency=0.0, confidence=0.0)

    weight = parameters.harmonic_weight
    score = (
        power
        + weight * np.interp(2.0 * frequencies, frequencies, power, left=0.0, right=0.0)
        + weight**2 * np.interp(
            3.0 * frequencies,
            frequencies,
            power,
            left=0.0,
            right=0.0,
        )
    )
    band_total = float(np.sum(score[indices]))
    if not np.isfinite(band_total) or band_total <= np.finfo(np.float64).tiny:
        return _SpectralPeak(frequency=0.0, confidence=0.0)

    peak_index = int(indices[np.argmax(score[indices])])
    offset = 0.0
    if 0 < peak_index < score.size - 1:
        neighborhood = np.log(score[peak_index - 1 : peak_index + 2] + 1e-30)
        denominator = float(neighborhood[0] - 2.0 * neighborhood[1] + neighborhood[2])
        if abs(denominator) > 1e-12:
            offset = float(
                np.clip(
                    0.5 * (neighborhood[0] - neighborhood[2]) / denominator,
                    -0.5,
                    0.5,
                )
            )
    frequency = float(
        np.clip(
            (peak_index + offset) / fft_size,
            1.0 / parameters.maximum_period,
            1.0 / parameters.minimum_period,
        )
    )
    confidence = float(np.clip(score[peak_index] / band_total, 0.0, 1.0))
    return _SpectralPeak(frequency=frequency, confidence=confidence)


def _aggregate_frequency(
    frequencies: NDArray[np.float64],
    confidences: NDArray[np.float64],
    aggregation: Aggregation,
) -> float:
    if aggregation == "mean":
        return float(np.mean(frequencies))
    if aggregation == "median":
        return float(np.median(frequencies))
    return float(np.average(frequencies, weights=confidences + 1e-6))


def _zero_result(sequence: PoseSequence, maximum_period: int) -> CountResult:
    return CountResult(
        count=0,
        period_frames=float(maximum_period),
        expert_counts=(0, 0, 0),
        confidence=0.0,
        period_stream=np.zeros(sequence.num_frames, dtype=np.float32),
    )


class LocalFrequencyReadout:
    """Infer repetitions from a local harmonic spectral ridge without labels."""

    def __init__(self, config: LocalFrequencyConfig) -> None:
        if not isinstance(config, LocalFrequencyConfig):
            raise TypeError("config must be a LocalFrequencyConfig")
        self.config = config

    def predict(self, sample: PoseSequence) -> CountResult:
        """Return one label-free count using the synthetic-frozen parameters."""

        if not isinstance(sample, PoseSequence):
            raise TypeError("sample must be a PoseSequence")
        parameters = self.config.readout
        if sample.num_frames < 2 * parameters.minimum_period:
            return _zero_result(sample, parameters.maximum_period)

        signal = _prepare_signal(sample, parameters.representation)
        global_peak = _spectral_peak(signal, parameters)
        if global_peak.frequency <= 0.0:
            return _zero_result(sample, parameters.maximum_period)

        window_frames = int(round(parameters.adaptive_window_cycles / global_peak.frequency))
        window_frames = min(
            max(window_frames, parameters.minimum_window_frames),
            sample.num_frames,
        )
        if window_frames % 2:
            window_frames -= 1
        if window_frames < 2 * parameters.minimum_period:
            return _zero_result(sample, parameters.maximum_period)

        starts: tuple[int, ...]
        if window_frames >= sample.num_frames:
            starts = (0,)
            window_frames = sample.num_frames
        else:
            step = max(
                parameters.minimum_step_frames,
                int(window_frames * parameters.window_step_fraction),
            )
            mutable_starts = list(range(0, sample.num_frames - window_frames + 1, step))
            terminal = sample.num_frames - window_frames
            if mutable_starts[-1] != terminal:
                mutable_starts.append(terminal)
            starts = tuple(mutable_starts)

        peaks = tuple(
            _spectral_peak(signal[start : start + window_frames], parameters)
            for start in starts
        )
        frequencies = np.asarray([peak.frequency for peak in peaks], dtype=np.float64)
        confidences = np.asarray([peak.confidence for peak in peaks], dtype=np.float64)
        if not np.any(frequencies > 0.0):
            return _zero_result(sample, parameters.maximum_period)

        duration = float(sample.num_frames - 1)
        expert_counts = (
            round_count(_aggregate_frequency(frequencies, confidences, "mean") * duration),
            round_count(_aggregate_frequency(frequencies, confidences, "median") * duration),
            round_count(
                _aggregate_frequency(
                    frequencies,
                    confidences,
                    "confidence_weighted",
                )
                * duration
            ),
        )
        selected_frequency = _aggregate_frequency(
            frequencies,
            confidences,
            parameters.aggregation,
        )
        if selected_frequency <= 0.0:
            return _zero_result(sample, parameters.maximum_period)
        selected_count = round_count(selected_frequency * duration)

        centers = np.asarray(
            [start + (window_frames - 1) / 2.0 for start in starts],
            dtype=np.float64,
        )
        frame_grid = np.arange(sample.num_frames, dtype=np.float64)
        ridge = np.interp(frame_grid, centers, frequencies)
        selected_expert = _AGGREGATIONS.index(parameters.aggregation)
        agreement = expert_counts.count(expert_counts[selected_expert]) / len(expert_counts)
        confidence = float(np.clip(np.mean(confidences) * agreement, 0.0, 1.0))
        return CountResult(
            count=selected_count,
            period_frames=float(1.0 / selected_frequency),
            expert_counts=expert_counts,
            confidence=confidence,
            period_stream=ridge.astype(np.float32),
        )


def predict_local_frequency_sequences(
    readout: LocalFrequencyReadout,
    sequences: Sequence[PoseSequence],
) -> tuple[PredictionRecord, ...]:
    """Create standard prediction records for the sealed evaluator boundary."""

    items = tuple(sequences)
    if not items:
        raise ValueError("prediction requires at least one PoseSequence")
    identifiers = tuple(item.video_id for item in items)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("prediction video_id values must be unique")
    return tuple(
        PredictionRecord(video_id=item.video_id, result=readout.predict(item)) for item in items
    )


def _candidate_key(
    representation: Representation,
    minimum_window_frames: int,
    aggregation: Aggregation,
) -> str:
    return f"{representation}.min{minimum_window_frames:03d}.{aggregation}"


@dataclass(frozen=True, slots=True)
class SyntheticCandidateScore:
    """One candidate's metrics on generated truth only."""

    candidate: str
    mean_absolute_error: float
    normalized_mean_absolute_error: float
    exact_rate: float
    obo: float

    @property
    def rank(self) -> tuple[float, float, float, float, str]:
        return (
            self.mean_absolute_error,
            -self.exact_rate,
            -self.obo,
            self.normalized_mean_absolute_error,
            self.candidate,
        )


@dataclass(frozen=True, slots=True)
class SyntheticFreezeReport:
    """Replay result proving the frozen choice uses no dataset labels."""

    selected_candidate: str
    sample_ids: tuple[str, ...]
    scores: tuple[SyntheticCandidateScore, ...]

    @property
    def selected_score(self) -> SyntheticCandidateScore:
        return min(self.scores, key=lambda score: score.rank)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classification": "synthetic-only parameter freeze",
            "label_source": "synthetic_generation_truth_only",
            "selected_candidate": self.selected_candidate,
            "sample_ids": list(self.sample_ids),
            "scores": [
                {
                    "candidate": score.candidate,
                    "mean_absolute_error": score.mean_absolute_error,
                    "normalized_mean_absolute_error": score.normalized_mean_absolute_error,
                    "exact_rate": score.exact_rate,
                    "obo": score.obo,
                }
                for score in self.scores
            ],
        }


def replay_synthetic_freeze(
    config: LocalFrequencyConfig,
    *,
    repository_root: str | Path,
) -> SyntheticFreezeReport:
    """Re-run the bounded search using internally generated targets only.

    The function deliberately has no ``sequences`` or ``targets`` argument.
    It cannot be redirected to a labeled dataset by a caller.
    """

    verify_selection_scope(config, repository_root)
    selection = config.selection
    count_samples = generate_count_sweep(
        minimum=selection.count_sweep_minimum,
        maximum=selection.count_sweep_maximum,
        frames=selection.frames,
        seed=selection.count_sweep_seed,
    )
    stress_samples = synthetic_stress_suite(
        count=float(selection.stress_count),
        frames=selection.frames,
        seed=selection.stress_seed,
    )
    frozen_samples = tuple(
        (sample.sequence.video_id, sample.sequence, float(sample.target_count))
        for sample in count_samples
    ) + tuple(
        (f"stress:{name}", sample.sequence, float(sample.target_count))
        for name, sample in stress_samples.items()
        if name not in selection.excluded_duplicate_stress_cases
    )

    scores: list[SyntheticCandidateScore] = []
    grid = selection.candidate_grid
    for representation in grid.representations:
        for minimum_window_frames in grid.minimum_window_frames:
            # One spectral pass emits all three aggregation experts.  Reusing
            # that tuple keeps replay bounded without changing any candidate.
            parameters = config.readout.model_copy(
                update={
                    "representation": representation,
                    "minimum_window_frames": minimum_window_frames,
                }
            )
            candidate_config = config.model_copy(update={"readout": parameters})
            readout = LocalFrequencyReadout(candidate_config)
            expert_predictions = tuple(
                readout.predict(sequence).expert_counts for _, sequence, _ in frozen_samples
            )
            targets = np.asarray(
                [target for _, _, target in frozen_samples],
                dtype=np.float64,
            )
            for aggregation in grid.aggregations:
                expert_index = _AGGREGATIONS.index(aggregation)
                predictions = np.asarray(
                    [counts[expert_index] for counts in expert_predictions],
                    dtype=np.float64,
                )
                errors = np.abs(predictions - targets)
                scores.append(
                    SyntheticCandidateScore(
                        candidate=_candidate_key(
                            representation,
                            minimum_window_frames,
                            aggregation,
                        ),
                        mean_absolute_error=float(np.mean(errors)),
                        normalized_mean_absolute_error=float(np.mean(errors / targets)),
                        exact_rate=float(np.mean(errors == 0.0)),
                        obo=float(np.mean(errors <= 1.0)),
                    )
                )
    ordered_scores = tuple(sorted(scores, key=lambda score: score.candidate))
    selected = min(ordered_scores, key=lambda score: score.rank)
    return SyntheticFreezeReport(
        selected_candidate=selected.candidate,
        sample_ids=tuple(identifier for identifier, _, _ in frozen_samples),
        scores=ordered_scores,
    )
