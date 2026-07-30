"""Isolated dev84 diagnostic for target-free pose spectral consensus.

This is an independently inferred exploratory readout.  It is not disclosed
PAMS logic, is permanently ineligible for a paper-comparison table, and must
not be used on the sealed UCFRep test105 split.

The ``predict`` command accepts only the committed dev84 identity sidecar and
pose cache.  It forms a robust consensus across signed coordinates, joints,
and fixed overlapping windows.  Every candidate period ``T`` is evaluated
together with its ``T/2`` and ``2T`` harmonic family, with signed lag
recurrence used to reject half-cycle aliases.  The resulting prediction bytes
are frozen before the separate ``score`` command is allowed to open dev
targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pams.data import (
    load_dev_target_manifest,
    load_pose_cache_set,
    load_pose_input_commitment,
    load_pose_input_manifest,
    validate_pose_input_binding,
)
from pams.metrics import compute_count_metrics, round_count
from pams.types import PoseSequence

METHOD_ID = "pose-spectral-consensus-harmonic-v1"
CLASSIFICATION = "inferred exploratory target-free multi-joint multi-window spectral consensus"
MINIMUM_PERIOD = 4
MAXIMUM_PERIOD = 128
WINDOW_LENGTHS = (256, 192, 128, 96, 64)
WINDOW_STEP_FRACTION = 0.5
MINIMUM_JOINT_COVERAGE = 0.5
MINIMUM_OBSERVATIONS = 4
MINIMUM_ACTIVE_JOINTS = 3
FFT_SIZE = 1024
SECOND_HARMONIC_WEIGHT = 0.75
SUBHARMONIC_PENALTY = 0.25
HARMONIC_MATCH_TOLERANCE = 0.15
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2026
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class WindowSpectrum:
    """One target-free window's normalized candidate-period scores."""

    length: int
    start: int
    active_joints: int
    periods: NDArray[np.float64]
    scores: NDArray[np.float64]
    best_period: float
    best_score: float


@dataclass(frozen=True, slots=True)
class SpectralConsensusEstimate:
    """Compact prediction and harmonic-family diagnostics for one video."""

    period_frames: float
    raw_count: float
    rounded_count: int
    confidence: float
    selection_source: str
    valid_frames: int
    total_frames: int
    active_joint_count: int
    window_count: int
    scale_count: int
    expert_counts: tuple[int, int, int]
    harmonic_family_periods: tuple[float, float, float]
    harmonic_family_scores: tuple[float, float, float]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _reject_duplicate_fields(
    pairs: list[tuple[str, Any]],
    *,
    document: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON field in {document}: {key!r}")
        output[key] = value
    return output


def _load_json_object(path: Path, *, document: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=lambda pairs: _reject_duplicate_fields(
                pairs,
                document=document,
            ),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite constant in {document}: {value}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {document}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{document} must be a JSON object")
    return payload


def _canonical_git_sha(value: str) -> str:
    revision = str(value).strip().lower()
    if _GIT_SHA_RE.fullmatch(revision) is None:
        raise ValueError("source_git_sha must be a lowercase 40-character Git SHA")
    return revision


def _canonical_sha256(value: str, name: str) -> str:
    digest = str(value).strip().lower()
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return digest


def _algorithm_parameters() -> dict[str, Any]:
    return {
        "minimum_period": MINIMUM_PERIOD,
        "maximum_period": MAXIMUM_PERIOD,
        "window_lengths": list(WINDOW_LENGTHS),
        "window_step_fraction": WINDOW_STEP_FRACTION,
        "minimum_joint_coverage": MINIMUM_JOINT_COVERAGE,
        "minimum_observations": MINIMUM_OBSERVATIONS,
        "minimum_active_joints": MINIMUM_ACTIVE_JOINTS,
        "fft_size": FFT_SIZE,
        "second_harmonic_weight": SECOND_HARMONIC_WEIGHT,
        "subharmonic_penalty": SUBHARMONIC_PENALTY,
        "harmonic_family": ["T/2", "T", "2T"],
        "harmonic_match_tolerance": HARMONIC_MATCH_TOLERANCE,
        "duration_convention": "sampled_frame_intervals_T_minus_1",
    }


def _source_hashes(repository: Path, runner: Path) -> dict[str, str]:
    files = {
        "runner": runner,
        "pams.data": repository / "src/pams/data.py",
        "pams.metrics": repository / "src/pams/metrics.py",
        "pams.types": repository / "src/pams/types.py",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing source files: {missing}")
    return {name: _sha256_file(path) for name, path in files.items()}


def _runtime(container_image_id: str) -> dict[str, Any]:
    affinity = getattr(os, "sched_getaffinity", None)
    visible_cpu_count = len(affinity(0)) if callable(affinity) else (os.cpu_count() or 0)
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "visible_cpu_count": visible_cpu_count,
        "container_image_id": container_image_id,
        "network_required": False,
        "inference_device": "cpu",
    }


def _window_slices(frame_count: int) -> tuple[tuple[int, int], ...]:
    """Return deterministic overlapping windows, including the full sequence."""

    if frame_count < 2 * MINIMUM_PERIOD:
        return ()
    lengths = tuple(
        dict.fromkeys(
            min(frame_count, length)
            for length in WINDOW_LENGTHS
            if min(frame_count, length) >= 2 * MINIMUM_PERIOD
        )
    )
    windows: list[tuple[int, int]] = []
    for length in lengths:
        if length == frame_count:
            starts = [0]
        else:
            step = max(1, int(round(length * WINDOW_STEP_FRACTION)))
            starts = list(range(0, frame_count - length + 1, step))
            terminal = frame_count - length
            if starts[-1] != terminal:
                starts.append(terminal)
        windows.extend((start, length) for start in starts)
    return tuple(windows)


def _interpolate_pose(
    sequence: PoseSequence,
) -> tuple[NDArray[np.float64], NDArray[np.bool_], int]:
    """Interpolate sufficiently observed joints and remove camera translation."""

    xyz = np.asarray(sequence.xyz, dtype=np.float64)
    valid = np.asarray(sequence.valid_mask, dtype=np.bool_)
    if xyz.ndim != 3 or xyz.shape[1:] != (33, 3):
        raise ValueError(f"expected pose [T,33,3], got {xyz.shape}")
    if valid.shape != (xyz.shape[0],):
        raise ValueError(f"expected valid mask [{xyz.shape[0]}], got {valid.shape}")
    valid_count = int(np.count_nonzero(valid))
    if valid_count < MINIMUM_OBSERVATIONS:
        return (
            np.empty((xyz.shape[0], 0, 3), dtype=np.float64),
            np.empty((xyz.shape[0], 0), dtype=np.bool_),
            valid_count,
        )

    observed = np.any(np.abs(xyz) > np.finfo(np.float32).eps, axis=2)
    observed &= valid[:, None]
    counts = observed.sum(axis=0)
    coverage = counts / float(valid_count)
    selected = np.flatnonzero(
        (counts >= MINIMUM_OBSERVATIONS) & (coverage >= MINIMUM_JOINT_COVERAGE)
    )
    if selected.size < MINIMUM_ACTIVE_JOINTS:
        return (
            np.empty((xyz.shape[0], 0, 3), dtype=np.float64),
            np.empty((xyz.shape[0], 0), dtype=np.bool_),
            valid_count,
        )

    values = xyz[:, selected].copy()
    selected_observed = observed[:, selected]
    frame_grid = np.arange(xyz.shape[0], dtype=np.float64)
    for local_joint in range(selected.size):
        known = np.flatnonzero(selected_observed[:, local_joint])
        for coordinate in range(3):
            values[:, local_joint, coordinate] = np.interp(
                frame_grid,
                known,
                values[known, local_joint, coordinate],
            )

    # The pose cache is already framewise scale normalized.  Median body
    # centering removes camera translation without allowing one bad joint to
    # determine the reference.
    body_center = np.median(values, axis=1, keepdims=True)
    values -= body_center
    return values, selected_observed, valid_count


def _linear_detrend(values: NDArray[np.float64]) -> NDArray[np.float64]:
    centered = values - values.mean(axis=0, keepdims=True)
    time = np.linspace(-1.0, 1.0, values.shape[0], dtype=np.float64)
    denominator = float(np.dot(time, time))
    slopes = np.tensordot(time, centered, axes=(0, 0)) / denominator
    return centered - time[:, None, None] * slopes[None, :, :]


def _interpolated_power(
    frequencies: NDArray[np.float64],
    normalized_power: NDArray[np.float64],
    query: NDArray[np.float64],
) -> NDArray[np.float64]:
    output = np.empty((normalized_power.shape[0], query.size), dtype=np.float64)
    for joint in range(normalized_power.shape[0]):
        output[joint] = np.interp(
            query,
            frequencies,
            normalized_power[joint],
            left=0.0,
            right=0.0,
        )
    return output


def _lag_recurrence(
    values: NDArray[np.float64],
    periods: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return non-negative signed recurrence for every joint and period."""

    joint_count = values.shape[1]
    output = np.zeros((joint_count, periods.size), dtype=np.float64)
    for period_index, period in enumerate(periods):
        lag = int(round(float(period)))
        if lag < 1 or values.shape[0] - lag < max(4, lag // 2):
            continue
        left = values[:-lag]
        right = values[lag:]
        numerator = np.sum(left * right, axis=(0, 2))
        denominator = np.sqrt(np.sum(left * left, axis=(0, 2)) * np.sum(right * right, axis=(0, 2)))
        correlation = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > np.finfo(np.float64).eps,
        )
        output[:, period_index] = np.clip(correlation, 0.0, 1.0)
    return output


def _window_spectrum(
    values: NDArray[np.float64],
    observed: NDArray[np.bool_],
    *,
    start: int,
    length: int,
) -> WindowSpectrum | None:
    """Score integer periods for one window using joint and harmonic consensus."""

    window = _linear_detrend(values[start : start + length])
    local_coverage = observed[start : start + length].mean(axis=0)
    variance = np.mean(window * window, axis=(0, 2))
    positive_variance = variance[variance > np.finfo(np.float64).eps]
    if positive_variance.size == 0:
        return None
    variance_floor = max(
        np.finfo(np.float64).eps,
        float(np.median(positive_variance)) * 1e-4,
    )
    active = (local_coverage >= MINIMUM_JOINT_COVERAGE) & (variance >= variance_floor)
    if int(np.count_nonzero(active)) < MINIMUM_ACTIVE_JOINTS:
        return None
    window = window[:, active]

    tapered = window * np.hanning(length)[:, None, None]
    spectrum = np.fft.rfft(tapered, n=FFT_SIZE, axis=0)
    joint_power = np.sum(np.abs(spectrum) ** 2, axis=2).T
    frequencies = np.fft.rfftfreq(FFT_SIZE)
    band = (frequencies >= 1.0 / MAXIMUM_PERIOD) & (frequencies <= 1.0 / MINIMUM_PERIOD)
    totals = joint_power[:, band].sum(axis=1)
    useful = totals > np.finfo(np.float64).tiny
    if int(np.count_nonzero(useful)) < MINIMUM_ACTIVE_JOINTS:
        return None
    joint_power = joint_power[useful]
    window = window[:, useful]
    normalized = joint_power / totals[useful, None]

    maximum_for_window = min(MAXIMUM_PERIOD, length // 2)
    if maximum_for_window < MINIMUM_PERIOD:
        return None
    periods = np.arange(
        MINIMUM_PERIOD,
        maximum_for_window + 1,
        dtype=np.float64,
    )
    fundamental = _interpolated_power(frequencies, normalized, 1.0 / periods)
    second_harmonic = _interpolated_power(
        frequencies,
        normalized,
        2.0 / periods,
    )
    subharmonic = _interpolated_power(
        frequencies,
        normalized,
        0.5 / periods,
    )

    # A true fundamental can be weak when a pose traverses two similar
    # extrema per action.  The second harmonic therefore supports T.  Power
    # at 2T is instead a warning that T may be a half-cycle alias.
    family_power = np.maximum(
        fundamental + SECOND_HARMONIC_WEIGHT * second_harmonic - SUBHARMONIC_PENALTY * subharmonic,
        0.0,
    )
    family_total = fundamental + second_harmonic + subharmonic
    fundamental_share = np.divide(
        fundamental,
        family_total,
        out=np.zeros_like(fundamental),
        where=family_total > np.finfo(np.float64).tiny,
    )
    recurrence = _lag_recurrence(window, periods)
    joint_scores = family_power * (0.5 + 0.5 * fundamental_share) * (0.2 + 0.8 * recurrence)
    joint_maximum = joint_scores.max(axis=1, keepdims=True)
    joint_scores = np.divide(
        joint_scores,
        joint_maximum,
        out=np.zeros_like(joint_scores),
        where=joint_maximum > np.finfo(np.float64).tiny,
    )

    top_joint_count = max(
        MINIMUM_ACTIVE_JOINTS,
        int(math.ceil(joint_scores.shape[0] * 0.25)),
    )
    ordered = np.sort(joint_scores, axis=0)
    robust_top_mean = ordered[-top_joint_count:].mean(axis=0)
    joint_agreement = np.mean(joint_scores >= 0.8, axis=0)
    scores = 0.7 * robust_top_mean + 0.3 * joint_agreement
    maximum_score = float(np.max(scores, initial=0.0))
    if not np.isfinite(maximum_score) or maximum_score <= 0.0:
        return None
    scores /= maximum_score
    best_index = int(np.argmax(scores))
    return WindowSpectrum(
        length=length,
        start=start,
        active_joints=int(joint_scores.shape[0]),
        periods=periods,
        scores=scores,
        best_period=float(periods[best_index]),
        best_score=float(scores[best_index]),
    )


def _relative_match(left: float, right: float) -> bool:
    return abs(left / right - 1.0) <= HARMONIC_MATCH_TOLERANCE


def _score_at_period(window: WindowSpectrum, period: float) -> float:
    if period < window.periods[0] or period > window.periods[-1]:
        return 0.0
    return float(np.interp(period, window.periods, window.scores))


def _best_period_from_windows(
    windows: tuple[WindowSpectrum, ...],
    *,
    allowed_lengths: set[int] | None = None,
) -> float:
    selected = (
        windows
        if allowed_lengths is None
        else tuple(window for window in windows if window.length in allowed_lengths)
    )
    if not selected:
        return float(MAXIMUM_PERIOD)
    candidates = np.arange(MINIMUM_PERIOD, MAXIMUM_PERIOD + 1, dtype=np.float64)
    scores = np.zeros_like(candidates)
    weights = np.zeros_like(candidates)
    for window in selected:
        eligible = candidates <= window.periods[-1]
        scores[eligible] += np.interp(
            candidates[eligible],
            window.periods,
            window.scores,
        )
        weights[eligible] += 1.0
    scores = np.divide(scores, weights, out=np.zeros_like(scores), where=weights > 0.0)
    return float(candidates[int(np.argmax(scores))])


def estimate_pose_spectral_consensus(
    sequence: PoseSequence,
) -> SpectralConsensusEstimate:
    """Estimate count without labels using multi-joint/multi-window consensus."""

    values, observed, valid_count = _interpolate_pose(sequence)
    total_frames = sequence.num_frames
    if values.shape[1] < MINIMUM_ACTIVE_JOINTS or total_frames < 2 * MINIMUM_PERIOD:
        return SpectralConsensusEstimate(
            period_frames=float(MAXIMUM_PERIOD),
            raw_count=0.0,
            rounded_count=0,
            confidence=0.0,
            selection_source="no-evidence",
            valid_frames=valid_count,
            total_frames=total_frames,
            active_joint_count=int(values.shape[1]),
            window_count=0,
            scale_count=0,
            expert_counts=(0, 0, 0),
            harmonic_family_periods=(64.0, 128.0, 128.0),
            harmonic_family_scores=(0.0, 0.0, 0.0),
        )

    window_results = tuple(
        result
        for start, length in _window_slices(total_frames)
        if (
            result := _window_spectrum(
                values,
                observed,
                start=start,
                length=length,
            )
        )
        is not None
    )
    if not window_results:
        return SpectralConsensusEstimate(
            period_frames=float(MAXIMUM_PERIOD),
            raw_count=0.0,
            rounded_count=0,
            confidence=0.0,
            selection_source="no-evidence",
            valid_frames=valid_count,
            total_frames=total_frames,
            active_joint_count=int(values.shape[1]),
            window_count=0,
            scale_count=0,
            expert_counts=(0, 0, 0),
            harmonic_family_periods=(64.0, 128.0, 128.0),
            harmonic_family_scores=(0.0, 0.0, 0.0),
        )

    candidates = np.arange(MINIMUM_PERIOD, MAXIMUM_PERIOD + 1, dtype=np.float64)
    scale_values = sorted({window.length for window in window_results}, reverse=True)
    scale_scores: list[NDArray[np.float64]] = []
    scale_votes: list[NDArray[np.float64]] = []
    for length in scale_values:
        scale_windows = tuple(window for window in window_results if window.length == length)
        scores = np.zeros_like(candidates)
        weights = np.zeros_like(candidates)
        votes = np.zeros_like(candidates)
        for window in scale_windows:
            eligible = candidates <= window.periods[-1]
            scores[eligible] += np.interp(
                candidates[eligible],
                window.periods,
                window.scores,
            )
            weights[eligible] += 1.0
            for candidate_index, candidate in enumerate(candidates):
                if not eligible[candidate_index]:
                    continue
                if _relative_match(window.best_period, float(candidate)):
                    votes[candidate_index] += 1.0
                elif _relative_match(window.best_period, float(candidate) / 2.0):
                    votes[candidate_index] += 0.5
                elif _relative_match(window.best_period, float(candidate) * 2.0):
                    votes[candidate_index] += 0.25
        scores = np.divide(
            scores,
            weights,
            out=np.zeros_like(scores),
            where=weights > 0.0,
        )
        votes /= float(len(scale_windows))
        scale_scores.append(scores)
        scale_votes.append(votes)

    stacked_scores = np.stack(scale_scores)
    stacked_votes = np.stack(scale_votes)
    eligible_scales = stacked_scores > 0.0
    scale_weights = np.asarray(
        [4.0 if length == total_frames else 1.0 for length in scale_values],
        dtype=np.float64,
    )[:, None]
    weighted_scores = np.sum(stacked_scores * scale_weights, axis=0)
    weighted_votes = np.sum(stacked_votes * scale_weights, axis=0)
    denominators = np.sum(eligible_scales * scale_weights, axis=0)
    base_consensus = np.divide(
        weighted_scores,
        denominators,
        out=np.zeros_like(weighted_scores),
        where=denominators > 0.0,
    )
    harmonic_vote = np.divide(
        weighted_votes,
        denominators,
        out=np.zeros_like(weighted_votes),
        where=denominators > 0.0,
    )
    # Short windows cannot resolve slow actions and often vote for T/2.  The
    # full-sequence spectrum therefore anchors the family, while window votes
    # serve only as a bounded consistency term.
    final_scores = 0.95 * base_consensus + 0.05 * harmonic_vote
    if not np.any(final_scores > 0.0):
        raise RuntimeError("non-empty windows produced no consensus score")
    global_period = _best_period_from_windows(
        window_results,
        allowed_lengths={total_frames},
    )
    long_period = _best_period_from_windows(
        window_results,
        allowed_lengths={length for length in scale_values if length >= 128},
    )
    all_period = _best_period_from_windows(window_results)
    # Each expert already uses the full T/2,T,2T joint/window score.  Their
    # median prevents short-window half-cycle votes from overruling both
    # global and long-window evidence.
    selected_period = float(np.median((global_period, long_period, all_period)))
    selected_index = int(np.argmin(np.abs(candidates - selected_period)))

    excluded = np.abs(np.log(candidates / selected_period)) <= math.log(1.15)
    alternatives = final_scores[~excluded]
    runner_up = float(np.max(alternatives, initial=0.0))
    selected_score = float(final_scores[selected_index])
    margin = max(0.0, selected_score - runner_up) / selected_score if selected_score > 0.0 else 0.0
    best_by_scale = [float(candidates[int(np.argmax(scores))]) for scores in scale_scores]
    scale_agreement = float(
        np.mean(
            [
                _relative_match(period, selected_period)
                or _relative_match(period, selected_period / 2.0)
                or _relative_match(period, selected_period * 2.0)
                for period in best_by_scale
            ]
        )
    )
    confidence = float(
        np.clip(
            0.45 * margin + 0.35 * scale_agreement + 0.20 * min(selected_score, 1.0),
            0.0,
            1.0,
        )
    )

    duration = float(total_frames - 1)
    raw_count = duration / selected_period
    rounded = round_count(raw_count)
    expert_counts = tuple(
        round_count(duration / period) for period in (global_period, long_period, all_period)
    )
    family_periods = (
        max(float(MINIMUM_PERIOD), selected_period / 2.0),
        selected_period,
        min(float(MAXIMUM_PERIOD), selected_period * 2.0),
    )
    family_scores = tuple(
        float(
            np.interp(
                period,
                candidates,
                final_scores,
                left=0.0,
                right=0.0,
            )
        )
        for period in family_periods
    )
    return SpectralConsensusEstimate(
        period_frames=selected_period,
        raw_count=raw_count,
        rounded_count=rounded,
        confidence=confidence,
        selection_source="multi-joint-multi-window-harmonic-consensus",
        valid_frames=valid_count,
        total_frames=total_frames,
        active_joint_count=int(values.shape[1]),
        window_count=len(window_results),
        scale_count=len(scale_values),
        expert_counts=expert_counts,
        harmonic_family_periods=family_periods,
        harmonic_family_scores=family_scores,
    )


def _load_bound_dev_input(
    sidecar_path: Path,
    commitment_path: Path,
) -> tuple[Any, Any, str, str]:
    sidecar_sha256 = _sha256_file(sidecar_path)
    commitment_sha256 = _sha256_file(commitment_path)
    sidecar = load_pose_input_manifest(sidecar_path, validate_exact=True)
    commitment = load_pose_input_commitment(commitment_path)
    validate_pose_input_binding(
        sidecar,
        commitment,
        sidecar_sha256=sidecar_sha256,
    )
    if sidecar.protocol != "ucfrep_526" or sidecar.split != "dev":
        raise ValueError("prediction requires the canonical ucfrep_526 dev sidecar")
    if len(sidecar.records) != 84:
        raise ValueError("prediction requires exactly 84 dev identities")
    if _sha256_file(sidecar_path) != sidecar_sha256:
        raise RuntimeError("dev sidecar changed while it was loaded")
    if _sha256_file(commitment_path) != commitment_sha256:
        raise RuntimeError("dev commitment changed while it was loaded")
    return sidecar, commitment, sidecar_sha256, commitment_sha256


def run_predict(arguments: argparse.Namespace) -> None:
    """Freeze target-free dev84 predictions; this boundary has no target path."""

    source_sha = _canonical_git_sha(arguments.source_git_sha)
    pose_fingerprint = _canonical_sha256(
        arguments.pose_fingerprint,
        "pose_fingerprint",
    )
    repository = arguments.repository.resolve()
    runner = arguments.runner.resolve()
    output_dir = arguments.output_dir.resolve()
    predictions_path = output_dir / "predictions.json"
    receipt_path = output_dir / "prediction.receipt.json"
    collisions = [str(path) for path in (predictions_path, receipt_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite prediction artifacts: {collisions}")

    source_hashes = _source_hashes(repository, runner)
    source_code_sha256 = _sha256_json(source_hashes)
    parameters = _algorithm_parameters()
    parameter_sha256 = _sha256_json(parameters)
    sidecar, commitment, sidecar_sha256, commitment_sha256 = _load_bound_dev_input(
        arguments.dev_inputs,
        arguments.dev_commitment,
    )
    sequences, pose_snapshot = load_pose_cache_set(
        sidecar.records,
        cache_dir=arguments.pose_cache_dir,
        pose_fingerprint=pose_fingerprint,
    )
    expected_ids = tuple(record.video_id for record in sidecar.records)
    if tuple(sequence.video_id for sequence in sequences) != expected_ids:
        raise RuntimeError("pose-cache order differs from committed dev sidecar")
    cache_receipts = {entry.video_id: entry for entry in pose_snapshot.entries}

    records: list[dict[str, Any]] = []
    for sequence in sequences:
        estimate = estimate_pose_spectral_consensus(sequence)
        record = asdict(estimate)
        record.update(
            {
                "video_id": sequence.video_id,
                "video_sha256": next(
                    item.video_sha256
                    for item in sidecar.records
                    if item.video_id == sequence.video_id
                ),
                "pose_cache_sha256": cache_receipts[sequence.video_id].cache_sha256,
            }
        )
        record["expert_counts"] = list(estimate.expert_counts)
        record["harmonic_family_periods"] = list(estimate.harmonic_family_periods)
        record["harmonic_family_scores"] = list(estimate.harmonic_family_scores)
        records.append(record)
    if tuple(record["video_id"] for record in records) != expected_ids:
        raise RuntimeError("prediction order differs from committed dev sidecar")

    # Recheck every target-free input and source before freezing bytes.
    if _sha256_file(arguments.dev_inputs) != sidecar_sha256:
        raise RuntimeError("dev sidecar changed during prediction")
    if _sha256_file(arguments.dev_commitment) != commitment_sha256:
        raise RuntimeError("dev commitment changed during prediction")
    if _source_hashes(repository, runner) != source_hashes:
        raise RuntimeError("prediction source changed during prediction")

    payload = {
        "schema_version": 1,
        "artifact_type": "pose_spectral_consensus_dev_predictions",
        "created_at_utc": _utc_now(),
        "method_id": METHOD_ID,
        "classification": CLASSIFICATION,
        "diagnostic_only": True,
        "eligible_for_paper_table": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "record_total": 84,
        "source_git_sha": source_sha,
        "source_files_sha256": source_hashes,
        "source_code_sha256": source_code_sha256,
        "algorithm_parameters": parameters,
        "algorithm_parameters_sha256": parameter_sha256,
        "dev_inputs_sha256": sidecar_sha256,
        "dev_commitment_sha256": commitment_sha256,
        "dev_identity_sha256": commitment.identity_sha256,
        "pose_fingerprint": pose_fingerprint,
        "pose_cache_set_sha256": pose_snapshot.fingerprint,
        "runtime": _runtime(arguments.container_image_id),
        "mount_audit": {
            "network": "none",
            "dev_identity_mounted": True,
            "dev_pose_mounted": True,
            "dev_targets_mounted": False,
            "test_identity_mounted": False,
            "test_pose_mounted": False,
            "test_targets_mounted": False,
        },
        "records": records,
    }
    _write_json_exclusive(predictions_path, payload)
    prediction_sha256 = _sha256_file(predictions_path)
    receipt = {
        "schema_version": 1,
        "artifact_type": "pose_spectral_consensus_prediction_receipt",
        "created_at_utc": _utc_now(),
        "method_id": METHOD_ID,
        "protocol": "ucfrep_526",
        "split": "dev",
        "prediction_file": predictions_path.name,
        "prediction_sha256": prediction_sha256,
        "prediction_bytes": predictions_path.stat().st_size,
        "record_total": 84,
        "source_git_sha": source_sha,
        "source_code_sha256": source_code_sha256,
        "algorithm_parameters_sha256": parameter_sha256,
        "dev_inputs_sha256": sidecar_sha256,
        "dev_commitment_sha256": commitment_sha256,
        "dev_identity_sha256": commitment.identity_sha256,
        "pose_cache_set_sha256": pose_snapshot.fingerprint,
        "label_firewall": {
            "prediction_loaded_dev_targets": False,
            "prediction_loaded_test_identity": False,
            "prediction_loaded_test_targets": False,
            "gt_count_or_action_oracle_used": False,
        },
    }
    _write_json_exclusive(receipt_path, receipt)


def _validate_frozen_predictions(
    predictions_path: Path,
    receipt_path: Path,
    *,
    repository: Path,
    runner: Path,
    source_sha: str,
) -> tuple[dict[str, Any], str, str]:
    prediction_sha256 = _sha256_file(predictions_path)
    receipt_sha256 = _sha256_file(receipt_path)
    predictions = _load_json_object(predictions_path, document="predictions")
    receipt = _load_json_object(receipt_path, document="prediction receipt")
    if predictions.get("artifact_type") != "pose_spectral_consensus_dev_predictions":
        raise ValueError("unexpected prediction artifact_type")
    if receipt.get("artifact_type") != "pose_spectral_consensus_prediction_receipt":
        raise ValueError("unexpected prediction receipt artifact_type")
    if predictions.get("method_id") != METHOD_ID or receipt.get("method_id") != METHOD_ID:
        raise ValueError("prediction method identity mismatch")
    if predictions.get("protocol") != "ucfrep_526" or predictions.get("split") != "dev":
        raise ValueError("prediction protocol/split mismatch")
    if predictions.get("source_git_sha") != source_sha:
        raise ValueError("prediction source revision mismatch")
    if receipt.get("prediction_sha256") != prediction_sha256:
        raise ValueError("prediction receipt SHA-256 mismatch")
    if receipt.get("prediction_bytes") != predictions_path.stat().st_size:
        raise ValueError("prediction receipt byte count mismatch")
    if receipt.get("source_code_sha256") != predictions.get("source_code_sha256"):
        raise ValueError("prediction receipt source binding mismatch")
    if receipt.get("algorithm_parameters_sha256") != predictions.get("algorithm_parameters_sha256"):
        raise ValueError("prediction receipt parameter binding mismatch")
    expected_hashes = _source_hashes(repository, runner)
    if predictions.get("source_files_sha256") != expected_hashes:
        raise ValueError("scoring source differs from prediction source")
    if predictions.get("source_code_sha256") != _sha256_json(expected_hashes):
        raise ValueError("prediction aggregate source hash mismatch")
    if predictions.get("algorithm_parameters") != _algorithm_parameters():
        raise ValueError("prediction algorithm parameters drifted")
    if predictions.get("algorithm_parameters_sha256") != _sha256_json(_algorithm_parameters()):
        raise ValueError("prediction algorithm parameter hash mismatch")

    rows = predictions.get("records")
    if not isinstance(rows, list) or len(rows) != 84:
        raise ValueError("prediction artifact must contain exactly 84 records")
    identifiers: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("prediction rows must be JSON objects")
        if "target" in row or "action" in row:
            raise ValueError("prediction rows must remain target-free")
        video_id = str(row.get("video_id", ""))
        if not video_id:
            raise ValueError("prediction video_id must be non-empty")
        identifiers.append(video_id)
        for field in (
            "period_frames",
            "raw_count",
            "confidence",
        ):
            value = float(row.get(field, math.nan))
            if not math.isfinite(value):
                raise ValueError(f"non-finite prediction field: {field}")
        if int(row.get("rounded_count", -1)) != round_count(float(row["raw_count"])):
            raise ValueError("rounded_count does not match raw_count")
    if len(set(identifiers)) != 84:
        raise ValueError("prediction video_id values must be unique")
    if _sha256_file(predictions_path) != prediction_sha256:
        raise RuntimeError("predictions changed while being validated")
    if _sha256_file(receipt_path) != receipt_sha256:
        raise RuntimeError("prediction receipt changed while being validated")
    return predictions, prediction_sha256, receipt_sha256


def run_score(arguments: argparse.Namespace) -> None:
    """Score already-frozen predictions; only this process may open dev labels."""

    source_sha = _canonical_git_sha(arguments.source_git_sha)
    repository = arguments.repository.resolve()
    runner = arguments.runner.resolve()
    output_dir = arguments.output_dir.resolve()
    evaluation_path = output_dir / "evaluation.json"
    evaluation_receipt_path = output_dir / "evaluation.receipt.json"
    collisions = [str(path) for path in (evaluation_path, evaluation_receipt_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite evaluation artifacts: {collisions}")

    # No operation above or inside this validator stats or opens a label file.
    predictions, prediction_sha256, receipt_sha256 = _validate_frozen_predictions(
        arguments.predictions,
        arguments.prediction_receipt,
        repository=repository,
        runner=runner,
        source_sha=source_sha,
    )

    dev_targets_sha256 = _sha256_file(arguments.dev_targets)
    targets = load_dev_target_manifest(arguments.dev_targets)
    rows = predictions["records"]
    prediction_ids = tuple(str(row["video_id"]) for row in rows)
    target_ids = tuple(record.video_id for record in targets.records)
    if prediction_ids != target_ids:
        raise ValueError("dev target order/identity differs from frozen predictions")
    if _sha256_file(arguments.dev_targets) != dev_targets_sha256:
        raise RuntimeError("dev targets changed while being loaded")

    report = compute_count_metrics(
        [float(row["raw_count"]) for row in rows],
        [record.count for record in targets.records],
        video_ids=prediction_ids,
        actions=[record.action for record in targets.records],
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        confidence_level=0.95,
    )
    if _sha256_file(arguments.predictions) != prediction_sha256:
        raise RuntimeError("predictions changed during scoring")
    if _sha256_file(arguments.prediction_receipt) != receipt_sha256:
        raise RuntimeError("prediction receipt changed during scoring")
    if _sha256_file(arguments.dev_targets) != dev_targets_sha256:
        raise RuntimeError("dev targets changed during scoring")

    evaluation = {
        "schema_version": 1,
        "artifact_type": "pose_spectral_consensus_dev_evaluation",
        "created_at_utc": _utc_now(),
        "method_id": METHOD_ID,
        "classification": CLASSIFICATION,
        "diagnostic_only": True,
        "eligible_for_paper_table": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "source_git_sha": source_sha,
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "dev_targets_sha256": dev_targets_sha256,
        "bootstrap_pairing": "paired_prediction_target_rows",
        "metrics": report.to_dict(),
        "acceptance_gate": {
            "required_nmae_at_most": 0.228,
            "required_obo_at_least": 0.666,
            "nmae_pass": report.nmae <= 0.228,
            "obo_pass": report.obo >= 0.666,
            "joint_pass": report.nmae <= 0.228 and report.obo >= 0.666,
            "verified_reproduction": False,
        },
        "mount_audit": {
            "network": "none",
            "prediction_mounted": True,
            "dev_targets_mounted": True,
            "dev_pose_mounted": False,
            "test_identity_mounted": False,
            "test_pose_mounted": False,
            "test_targets_mounted": False,
        },
    }
    _write_json_exclusive(evaluation_path, evaluation)
    _write_json_exclusive(
        evaluation_receipt_path,
        {
            "schema_version": 1,
            "artifact_type": "pose_spectral_consensus_evaluation_receipt",
            "created_at_utc": _utc_now(),
            "evaluation_sha256": _sha256_file(evaluation_path),
            "evaluation_bytes": evaluation_path.stat().st_size,
            "prediction_sha256": prediction_sha256,
            "prediction_receipt_sha256": receipt_sha256,
            "dev_targets_sha256": dev_targets_sha256,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    )


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict")
    _add_source_arguments(predict)
    predict.add_argument("--dev-inputs", type=Path, required=True)
    predict.add_argument("--dev-commitment", type=Path, required=True)
    predict.add_argument("--pose-cache-dir", type=Path, required=True)
    predict.add_argument("--pose-fingerprint", required=True)
    predict.add_argument("--container-image-id", required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.set_defaults(handler=run_predict)

    score = subparsers.add_parser("score")
    _add_source_arguments(score)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--prediction-receipt", type=Path, required=True)
    score.add_argument("--dev-targets", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.set_defaults(handler=run_score)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    arguments.handler(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
