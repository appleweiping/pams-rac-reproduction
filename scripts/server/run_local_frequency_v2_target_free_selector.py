"""Target-free selector for the exploratory local-frequency v2 readout.

This runner formalizes the 512-candidate grid first explored in
``tmp/explore_local_frequency_readouts.py``.  It is an independently inferred
diagnostic and is permanently ineligible for a paper-comparison table.

The only dataset input is the exact, count-free UCFRep train337 pose sidecar.
Candidate selection uses deterministic synthetic truth plus prediction
consistency under transformations of 64 hash-selected training pose streams.
No development/test identity, action class, or repetition count is accepted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.signal import detrend

from pams.data import (
    PoseInputManifest,
    load_pose_cache_set,
    load_pose_input_manifest,
    pose_input_identity_sha256,
)
from pams.stress import pose_sequence_sha256
from pams.synthetic import generate_count_sweep, synthetic_stress_suite
from pams.types import PoseSequence

_ARTIFACT_TYPE = "pams_local_frequency_v2_target_free_selector"
_CLASSIFICATION = "exploratory-derived target-free selector; paper-table ineligible"
_SOURCE_RELATIVE_PATH = "scripts/server/run_local_frequency_v2_target_free_selector.py"
_FROZEN_POSE_FINGERPRINT = (
    "3dd0388320095796f42aa82d904073b6478b073ed351394ef8d03c30798a0116"
)
_FROZEN_TRAIN337_SIDECAR_SHA256 = (
    "e238d307b4ed37b2f6c19eefdf7fa8812ec3d231cd2a7091be38e1c279f04207"
)
_FROZEN_TRAIN337_MANIFEST_FINGERPRINT = (
    "e3d6c979f5d160e37199726bff773a803cc77a0357079d046462b58da6454662"
)
_FROZEN_TRAIN337_IDENTITY_SHA256 = (
    "88367535e296213377c366ed69abbd1e808c38049d79bb948091688e4c5b7b3f"
)
_TRAIN_SAMPLE_TOTAL = 64
_TRAIN_SAMPLE_DOMAIN = b"pams-local-frequency-v2-train64\0"
_TRANSFORM_SEED = 34_071_126
_WORKER_TOTAL = 8
_COUNT_MINIMUM = 2
_COUNT_MAXIMUM = 40
_SYNTHETIC_SEED = 2026
_SYNTHETIC_STRESS_COUNT = 8
_ROTATION_DEGREES = (-15.0, 15.0)
_SCALE_MULTIPLIER = 1.15
_TRANSLATION_DELTA = 0.1
_OCCLUSION_TIME_FRACTION = 0.20
_OCCLUDED_JOINT_TOTAL = 10
_NOISE_STANDARD_DEVIATION = 0.02
_SPEED_WARP_ENDPOINTS = (0.5, 2.0)
_DEGENERATE_BOUNDARY_SHARE_LIMIT = 0.75
_DEGENERATE_MODE_SHARE_LIMIT = 0.75
_DEGENERATE_MINIMUM_DISTINCT_COUNTS = 3

_TRIMS = (0.0, 0.05, 0.075, 0.1)
_FEATURE_NAMES = ("xyz", "centered")
_NORMALIZED_DIMENSIONS = (False, True)
_MINIMUM_CYCLES = (1.0, 1.5)
_WINDOWS = (64, 96, 128, 192, 256)
_STATISTICS = ("mean", "median")
_SCALE_REDUCERS = ("scale_median", "scale_min", "scale_max")
_TRANSFORM_NAMES = (
    "reverse",
    "rotation_negative_15",
    "rotation_positive_15",
    "scale_1_15",
    "translation_positive_0_1",
    "occlusion_20pct_10joints",
    "noise_std_0_02",
    "speed_warp_0_5_to_2_0",
)
_LEXICOGRAPHIC_OBJECTIVE = (
    "degenerate_candidate_flag",
    "synthetic_count_sweep_nmae",
    "synthetic_stress_nmae",
    "train_transform_relative_disagreement",
    "train_duplicate_time_relative_scale_error",
    "prediction_boundary_and_mode_penalty",
    "negative_distinct_training_count_total",
    "candidate_key",
)

@dataclass(frozen=True, slots=True)
class CandidateAudit:
    """Complete aggregate audit and stable selection key for one candidate."""

    candidate_key: str
    parameters: Mapping[str, Any]
    synthetic_count_sweep_nmae: float
    synthetic_count_sweep_mae: float
    synthetic_count_sweep_exact_rate: float
    synthetic_count_sweep_obo: float
    synthetic_stress_nmae: float
    synthetic_stress_mae: float
    synthetic_stress_exact_rate: float
    synthetic_stress_obo: float
    train_transform_relative_disagreement: float
    train_transform_exact_agreement: float
    train_duplicate_time_relative_scale_error: float
    train_duplicate_time_exact_scale_rate: float
    training_prediction_boundary_share: float
    training_prediction_mode: int
    training_prediction_mode_share: float
    distinct_training_count_total: int
    prediction_boundary_and_mode_penalty: float
    degenerate_candidate_flag: int
    synthetic_count_sweep_sample_total: int
    synthetic_stress_sample_total: int
    train_transform_pair_total: int
    train_duplicate_pair_total: int

    @property
    def rank_key(self) -> tuple[int | float | str, ...]:
        """Return the exact frozen lexicographic selection key."""

        return (
            self.degenerate_candidate_flag,
            self.synthetic_count_sweep_nmae,
            self.synthetic_stress_nmae,
            self.train_transform_relative_disagreement,
            self.train_duplicate_time_relative_scale_error,
            self.prediction_boundary_and_mode_penalty,
            -self.distinct_training_count_total,
            self.candidate_key,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key,
            "parameters": dict(self.parameters),
            "rank_key": list(self.rank_key),
            "degenerate_candidate_flag": bool(self.degenerate_candidate_flag),
            "synthetic_count_sweep": {
                "sample_total": self.synthetic_count_sweep_sample_total,
                "nmae": self.synthetic_count_sweep_nmae,
                "mae": self.synthetic_count_sweep_mae,
                "exact_rate": self.synthetic_count_sweep_exact_rate,
                "obo": self.synthetic_count_sweep_obo,
            },
            "synthetic_stress": {
                "sample_total": self.synthetic_stress_sample_total,
                "nmae": self.synthetic_stress_nmae,
                "mae": self.synthetic_stress_mae,
                "exact_rate": self.synthetic_stress_exact_rate,
                "obo": self.synthetic_stress_obo,
            },
            "unlabeled_train_consistency": {
                "transform_pair_total": self.train_transform_pair_total,
                "transform_relative_disagreement": (
                    self.train_transform_relative_disagreement
                ),
                "transform_exact_agreement": self.train_transform_exact_agreement,
                "duplicate_time_pair_total": self.train_duplicate_pair_total,
                "duplicate_time_expected": "min(40, 2 * original_prediction)",
                "duplicate_time_relative_scale_error": (
                    self.train_duplicate_time_relative_scale_error
                ),
                "duplicate_time_exact_scale_rate": (
                    self.train_duplicate_time_exact_scale_rate
                ),
                "prediction_boundary_share": self.training_prediction_boundary_share,
                "prediction_mode": self.training_prediction_mode,
                "prediction_mode_share": self.training_prediction_mode_share,
                "distinct_prediction_total": self.distinct_training_count_total,
                "prediction_boundary_and_mode_penalty": (
                    self.prediction_boundary_and_mode_penalty
                ),
            },
        }


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_stable_bytes(path: Path) -> bytes:
    before = path.stat()
    payload = path.read_bytes()
    after = path.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or len(payload) != after.st_size
    ):
        raise RuntimeError(f"input changed while it was being read: {path}")
    return payload


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical finite JSON with exclusive-create semantics."""

    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)


def _candidate_methods() -> tuple[str, ...]:
    methods: list[str] = []
    for trim in _TRIMS:
        for feature_name in _FEATURE_NAMES:
            for normalized in _NORMALIZED_DIMENSIONS:
                normalization = "norm" if normalized else "sum"
                for minimum_cycles in _MINIMUM_CYCLES:
                    for window in _WINDOWS:
                        for statistic in _STATISTICS:
                            methods.append(
                                f"{feature_name}.trim{trim:g}.w{window}."
                                f"{normalization}.c{minimum_cycles:g}.{statistic}"
                            )
                    for statistic in _STATISTICS:
                        for reducer in _SCALE_REDUCERS:
                            methods.append(
                                f"{feature_name}.trim{trim:g}.{normalization}."
                                f"c{minimum_cycles:g}.{statistic}.{reducer}"
                            )
    result = tuple(methods)
    if len(result) != 512 or len(set(result)) != 512:
        raise AssertionError("local-frequency v2 grid must contain 512 unique candidates")
    return result


def _candidate_parameters(candidate_key: str) -> dict[str, Any]:
    match = re.fullmatch(
        r"(xyz|centered)\.trim(0|0\.05|0\.075|0\.1)"
        r"(?:\.w(64|96|128|192|256))?\.(sum|norm)"
        r"\.c(1|1\.5)\.(mean|median)"
        r"(?:\.(scale_median|scale_min|scale_max))?",
        candidate_key,
    )
    if match is None:
        raise ValueError(f"candidate key is outside the frozen grid: {candidate_key!r}")
    (
        feature_name,
        trim,
        window,
        normalization,
        minimum_cycles,
        statistic,
        scale_reducer,
    ) = match.groups()
    if (window is None) == (scale_reducer is None):
        raise ValueError("candidate key must select one window or one scale reducer")
    parameters: dict[str, Any] = {
        "feature": feature_name,
        "trim_fraction_each_tail": float(trim),
        "normalized_dimensions": normalization == "norm",
        "minimum_cycles_per_window": float(minimum_cycles),
        "statistic": statistic,
        "count_bounds": [_COUNT_MINIMUM, _COUNT_MAXIMUM],
    }
    if window is None:
        parameters["window_frames"] = list(_WINDOWS)
        parameters["scale_reducer"] = scale_reducer
    else:
        parameters["window_frames"] = int(window)
        parameters["scale_reducer"] = None
    return parameters


def _interpolate(features: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(features, dtype=np.float64))
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return np.zeros_like(values)
    grid = np.arange(values.shape[0])
    return np.stack(
        [
            np.interp(grid, indices, values[indices, dimension])
            for dimension in range(values.shape[1])
        ],
        axis=1,
    )


def _activity_crop(
    xyz: np.ndarray,
    valid: np.ndarray,
    trim: float,
) -> tuple[int, int]:
    if trim <= 0:
        return 0, xyz.shape[0]
    flat = _interpolate(xyz.reshape(xyz.shape[0], -1), valid)
    velocity = np.diff(flat, axis=0, prepend=flat[:1]).reshape(
        flat.shape[0],
        33,
        3,
    )
    energy = np.median(np.linalg.norm(velocity, axis=-1), axis=1)
    energy = gaussian_filter1d(np.maximum(energy, 0.0), sigma=2.0)
    total = float(energy.sum())
    if total <= 1e-12:
        return 0, xyz.shape[0]
    cumulative = np.cumsum(energy) / total
    start = max(0, int(np.searchsorted(cumulative, trim)) - 2)
    stop = min(
        xyz.shape[0],
        int(np.searchsorted(cumulative, 1.0 - trim, side="right")) + 2,
    )
    return (start, stop) if stop - start >= 24 else (0, xyz.shape[0])


def _features(xyz: np.ndarray) -> dict[str, np.ndarray]:
    center = np.mean(xyz[:, (11, 12, 23, 24)], axis=1, keepdims=True)
    centered = (xyz - center).reshape(xyz.shape[0], -1)
    return {
        "xyz": xyz.reshape(xyz.shape[0], -1),
        "centered": centered,
    }


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    cumulative = np.cumsum(weights[order])
    return float(ordered[np.searchsorted(cumulative, cumulative[-1] / 2.0)])


def _local_frequency(
    features: np.ndarray,
    *,
    requested_window: int,
    normalized_dimensions: bool,
    minimum_cycles: float,
) -> tuple[float, float]:
    length = features.shape[0]
    window = min(requested_window, length)
    if window < 16:
        return 2.0 / max(length, 1), 2.0 / max(length, 1)
    hop = max(4, window // 4)
    starts = list(range(0, max(1, length - window + 1), hop))
    final_start = max(0, length - window)
    if not starts or starts[-1] != final_start:
        starts.append(final_start)
    frequencies: list[float] = []
    confidences: list[float] = []
    for start in starts:
        values = detrend(features[start : start + window], axis=0, type="linear")
        values *= np.hanning(window)[:, None]
        n_fft = 4 * window
        power = np.abs(np.fft.rfft(values, n=n_fft, axis=0)) ** 2
        frequency_axis = np.fft.rfftfreq(n_fft)
        allowed = (
            (frequency_axis >= max(1.0 / 128.0, minimum_cycles / window))
            & (frequency_axis <= 1.0 / 4.0)
        )
        band = power[allowed]
        informative = np.sum(band, axis=0) > 1e-12
        if not np.any(informative):
            continue
        band = band[:, informative]
        if normalized_dimensions:
            band = band / np.maximum(
                np.sum(band, axis=0, keepdims=True),
                1e-12,
            )
        aggregate = np.mean(band, axis=1)
        allowed_frequencies = frequency_axis[allowed]
        peak = int(np.argmax(aggregate))
        frequencies.append(float(allowed_frequencies[peak]))
        confidences.append(
            float(aggregate[peak] / max(float(np.sum(aggregate)), 1e-12))
        )
    if not frequencies:
        return 2.0 / max(length, 1), 2.0 / max(length, 1)
    frequency_values = np.asarray(frequencies)
    confidence_values = np.maximum(np.asarray(confidences), 1e-6)
    mean_frequency = float(
        np.sum(frequency_values * confidence_values) / np.sum(confidence_values)
    )
    median_frequency = _weighted_median(frequency_values, confidence_values)
    return mean_frequency, median_frequency


def _round_bounded_count(frequency: float, frames: int) -> int:
    return int(
        np.clip(
            math.floor(frequency * frames + 0.5),
            _COUNT_MINIMUM,
            _COUNT_MAXIMUM,
        )
    )


def _predict_all_candidates(sequence: PoseSequence) -> dict[str, int]:
    """Evaluate the exact exploratory 512-grid on one pose stream."""

    xyz = np.asarray(sequence.xyz, dtype=np.float64)
    valid = np.asarray(sequence.valid_mask, dtype=np.bool_)
    feature_sets = _features(xyz)
    predictions: dict[str, int] = {}
    for trim in _TRIMS:
        start, stop = _activity_crop(xyz, valid, trim)
        for feature_name, values in feature_sets.items():
            cropped = _interpolate(values, valid)[start:stop]
            for normalized in _NORMALIZED_DIMENSIONS:
                normalization = "norm" if normalized else "sum"
                for minimum_cycles in _MINIMUM_CYCLES:
                    scale_counts: dict[str, list[int]] = {
                        "mean": [],
                        "median": [],
                    }
                    for window in _WINDOWS:
                        mean_frequency, median_frequency = _local_frequency(
                            cropped,
                            requested_window=window,
                            normalized_dimensions=normalized,
                            minimum_cycles=minimum_cycles,
                        )
                        for statistic, frequency in (
                            ("mean", mean_frequency),
                            ("median", median_frequency),
                        ):
                            count = _round_bounded_count(frequency, len(cropped))
                            method = (
                                f"{feature_name}.trim{trim:g}.w{window}."
                                f"{normalization}.c{minimum_cycles:g}.{statistic}"
                            )
                            predictions[method] = count
                            scale_counts[statistic].append(count)
                    for statistic, counts in scale_counts.items():
                        for reducer, value in (
                            (
                                "scale_median",
                                int(math.floor(float(np.median(counts)) + 0.5)),
                            ),
                            ("scale_min", int(np.min(counts))),
                            ("scale_max", int(np.max(counts))),
                        ):
                            method = (
                                f"{feature_name}.trim{trim:g}.{normalization}."
                                f"c{minimum_cycles:g}.{statistic}.{reducer}"
                            )
                            predictions[method] = int(
                                np.clip(value, _COUNT_MINIMUM, _COUNT_MAXIMUM)
                            )
    if tuple(sorted(predictions)) != tuple(sorted(_candidate_methods())):
        raise AssertionError("candidate prediction keys differ from the frozen grid")
    return predictions


def _transform_rng(video_id: str, transform_name: str) -> np.random.Generator:
    encoded = f"{_TRANSFORM_SEED}\0{video_id}\0{transform_name}".encode()
    seed = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "little")
    return np.random.default_rng(seed)


def _new_sequence(
    source: PoseSequence,
    name: str,
    xyz: np.ndarray,
    valid_mask: np.ndarray | None = None,
) -> PoseSequence:
    return PoseSequence(
        video_id=f"{source.video_id}::{name}",
        fps=source.fps,
        xyz=xyz,
        valid_mask=source.valid_mask if valid_mask is None else valid_mask,
    )


def _interpolate_pose_time(
    source: PoseSequence,
    positions: np.ndarray,
    name: str,
) -> PoseSequence:
    clipped = np.clip(np.asarray(positions, dtype=np.float64), 0, source.num_frames - 1)
    left = np.floor(clipped).astype(np.int64)
    right = np.minimum(left + 1, source.num_frames - 1)
    weight = (clipped - left).astype(np.float32)
    xyz = (
        source.xyz[left] * (1.0 - weight[:, None, None])
        + source.xyz[right] * weight[:, None, None]
    )
    exact = left == right
    valid = (source.valid_mask[left] & source.valid_mask[right]) | (
        exact & source.valid_mask[left]
    )
    return _new_sequence(source, name, xyz, valid)


def _training_transforms(sequence: PoseSequence) -> dict[str, PoseSequence]:
    """Build the eight frozen, count-preserving label-free transformations."""

    xyz = np.asarray(sequence.xyz, dtype=np.float32)
    transforms: dict[str, PoseSequence] = {
        "reverse": _new_sequence(
            sequence,
            "reverse",
            xyz[::-1],
            sequence.valid_mask[::-1],
        )
    }
    center = np.mean(xyz, axis=1, keepdims=True, dtype=np.float64).astype(np.float32)
    centered = xyz - center
    for degrees, name in zip(
        _ROTATION_DEGREES,
        ("rotation_negative_15", "rotation_positive_15"),
        strict=True,
    ):
        radians = math.radians(degrees)
        cosine = math.cos(radians)
        sine = math.sin(radians)
        rotated = centered.copy()
        rotated[..., 0] = cosine * centered[..., 0] - sine * centered[..., 1]
        rotated[..., 1] = sine * centered[..., 0] + cosine * centered[..., 1]
        transforms[name] = _new_sequence(sequence, name, rotated + center)
    transforms["scale_1_15"] = _new_sequence(
        sequence,
        "scale_1_15",
        center + _SCALE_MULTIPLIER * centered,
    )
    translated = xyz.copy()
    translated[sequence.valid_mask] += _TRANSLATION_DELTA
    transforms["translation_positive_0_1"] = _new_sequence(
        sequence,
        "translation_positive_0_1",
        translated,
    )

    occluded = xyz.copy()
    time_count = max(
        1,
        int(math.floor(sequence.num_frames * _OCCLUSION_TIME_FRACTION + 0.5)),
    )
    occlusion_rng = _transform_rng(sequence.video_id, "occlusion_20pct_10joints")
    start = int(occlusion_rng.integers(0, sequence.num_frames - time_count + 1))
    joints = np.sort(
        occlusion_rng.choice(
            xyz.shape[1],
            size=_OCCLUDED_JOINT_TOTAL,
            replace=False,
        )
    )
    occluded[start : start + time_count, joints, :] = 0.0
    transforms["occlusion_20pct_10joints"] = _new_sequence(
        sequence,
        "occlusion_20pct_10joints",
        occluded,
    )

    noise_rng = _transform_rng(sequence.video_id, "noise_std_0_02")
    noisy = xyz.copy()
    noisy[sequence.valid_mask] += noise_rng.normal(
        0.0,
        _NOISE_STANDARD_DEVIATION,
        size=noisy[sequence.valid_mask].shape,
    ).astype(np.float32)
    transforms["noise_std_0_02"] = _new_sequence(
        sequence,
        "noise_std_0_02",
        noisy,
    )

    speed = np.linspace(
        _SPEED_WARP_ENDPOINTS[0],
        _SPEED_WARP_ENDPOINTS[1],
        sequence.num_frames,
        dtype=np.float64,
    )
    cumulative = np.concatenate(
        (np.asarray([0.0]), np.cumsum(0.5 * (speed[:-1] + speed[1:])))
    )
    positions = cumulative / cumulative[-1] * (sequence.num_frames - 1)
    transforms["speed_warp_0_5_to_2_0"] = _interpolate_pose_time(
        sequence,
        positions,
        "speed_warp_0_5_to_2_0",
    )
    if tuple(transforms) != _TRANSFORM_NAMES:
        raise AssertionError("training transforms differ from the frozen order")
    return transforms


def _duplicate_time(sequence: PoseSequence) -> PoseSequence:
    """Concatenate a pose stream with itself; expected count is exactly 2x."""

    return _new_sequence(
        sequence,
        "duplicate_time_2x",
        np.concatenate((sequence.xyz, sequence.xyz), axis=0),
        np.concatenate((sequence.valid_mask, sequence.valid_mask), axis=0),
    )


def _hash_select_training_records(
    manifest: PoseInputManifest,
    *,
    sample_total: int = _TRAIN_SAMPLE_TOTAL,
) -> tuple[Any, ...]:
    if (
        manifest.protocol != "ucfrep_526"
        or manifest.split != "train"
        or len(manifest.records) != 337
    ):
        raise ValueError("selector requires the exact canonical UCFRep train337 sidecar")
    ranked = sorted(
        manifest.records,
        key=lambda record: (
            hashlib.sha256(_TRAIN_SAMPLE_DOMAIN + record.video_id.encode()).hexdigest(),
            record.video_id,
        ),
    )
    if sample_total < 1 or sample_total > len(ranked):
        raise ValueError("training sample_total must lie in [1, 337]")
    return tuple(ranked[:sample_total])


def _validate_canonical_train337_binding(
    *,
    sidecar_sha256: str,
    manifest: PoseInputManifest | None = None,
) -> None:
    """Bind selection to the one preregistered train337 sidecar byte stream."""

    if sidecar_sha256 != _FROZEN_TRAIN337_SIDECAR_SHA256:
        raise ValueError("selector requires the exact frozen train337 sidecar bytes")
    if manifest is None:
        return
    if manifest.fingerprint != _FROZEN_TRAIN337_MANIFEST_FINGERPRINT:
        raise ValueError("selector train337 manifest fingerprint mismatch")
    if pose_input_identity_sha256(manifest.records) != _FROZEN_TRAIN337_IDENTITY_SHA256:
        raise ValueError("selector train337 identity commitment mismatch")


def _predict_worker(sequence: PoseSequence) -> tuple[str, dict[str, int]]:
    return sequence.video_id, _predict_all_candidates(sequence)


def _predict_many(
    sequences: Sequence[PoseSequence],
    *,
    worker_total: int = _WORKER_TOTAL,
) -> dict[str, dict[str, int]]:
    items = tuple(sequences)
    identifiers = tuple(sequence.video_id for sequence in items)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("prediction work items require unique video_id values")
    if not items:
        return {}
    if worker_total > 1 and os.name != "nt":
        with ProcessPoolExecutor(max_workers=worker_total) as executor:
            rows = tuple(executor.map(_predict_worker, items, chunksize=1))
    else:
        rows = tuple(_predict_worker(sequence) for sequence in items)
    if tuple(identifier for identifier, _ in rows) != identifiers:
        raise RuntimeError("parallel prediction changed the frozen input order")
    return dict(rows)


def _metric_summary(
    predictions: np.ndarray,
    truth: np.ndarray,
) -> dict[str, float | int]:
    predicted = np.asarray(predictions, dtype=np.int64)
    expected = np.asarray(truth, dtype=np.float64)
    if predicted.shape != expected.shape or predicted.ndim != 1 or not predicted.size:
        raise ValueError("metric vectors must be non-empty and shape-aligned")
    absolute = np.abs(predicted - expected)
    return {
        "sample_total": int(predicted.size),
        "nmae": float(np.mean(absolute / expected)),
        "mae": float(np.mean(absolute)),
        "exact_rate": float(np.mean(absolute == 0)),
        "obo": float(np.mean(absolute <= 1)),
    }


def _score_candidate(
    candidate_key: str,
    *,
    count_sweep_predictions: Sequence[int],
    count_sweep_truth: Sequence[float],
    stress_predictions: Sequence[int],
    stress_truth: Sequence[float],
    original_predictions: Sequence[int],
    transformed_predictions: Sequence[Sequence[int]],
    duplicate_predictions: Sequence[int],
) -> CandidateAudit:
    """Score one candidate without accepting a dataset label vector."""

    original = np.asarray(original_predictions, dtype=np.int64)
    transformed = np.asarray(transformed_predictions, dtype=np.int64)
    duplicate = np.asarray(duplicate_predictions, dtype=np.int64)
    if original.ndim != 1 or not original.size:
        raise ValueError("original predictions must be a non-empty vector")
    if transformed.shape != (original.size, len(_TRANSFORM_NAMES)):
        raise ValueError("transformed predictions must cover every frozen transform")
    if duplicate.shape != original.shape:
        raise ValueError("duplicate predictions must align with originals")

    count_sweep = _metric_summary(
        np.asarray(count_sweep_predictions),
        np.asarray(count_sweep_truth),
    )
    stress = _metric_summary(
        np.asarray(stress_predictions),
        np.asarray(stress_truth),
    )
    transform_difference = np.abs(transformed - original[:, None])
    transform_relative = transform_difference / np.maximum(original[:, None], 1)
    duplicate_expected = np.minimum(_COUNT_MAXIMUM, 2 * original)
    duplicate_difference = np.abs(duplicate - duplicate_expected)
    duplicate_relative = duplicate_difference / np.maximum(duplicate_expected, 1)

    training_values = np.concatenate((original[:, None], transformed), axis=1).ravel()
    frequencies = Counter(int(value) for value in training_values)
    mode, mode_total = min(
        frequencies.items(),
        key=lambda item: (-item[1], item[0]),
    )
    boundary_share = float(
        np.mean(
            (training_values == _COUNT_MINIMUM)
            | (training_values == _COUNT_MAXIMUM)
        )
    )
    mode_share = mode_total / training_values.size
    distinct_total = len(frequencies)
    degenerate = int(
        boundary_share >= _DEGENERATE_BOUNDARY_SHARE_LIMIT
        or mode_share >= _DEGENERATE_MODE_SHARE_LIMIT
        or distinct_total < _DEGENERATE_MINIMUM_DISTINCT_COUNTS
    )
    return CandidateAudit(
        candidate_key=candidate_key,
        parameters=_candidate_parameters(candidate_key),
        synthetic_count_sweep_nmae=float(count_sweep["nmae"]),
        synthetic_count_sweep_mae=float(count_sweep["mae"]),
        synthetic_count_sweep_exact_rate=float(count_sweep["exact_rate"]),
        synthetic_count_sweep_obo=float(count_sweep["obo"]),
        synthetic_stress_nmae=float(stress["nmae"]),
        synthetic_stress_mae=float(stress["mae"]),
        synthetic_stress_exact_rate=float(stress["exact_rate"]),
        synthetic_stress_obo=float(stress["obo"]),
        train_transform_relative_disagreement=float(np.mean(transform_relative)),
        train_transform_exact_agreement=float(np.mean(transform_difference == 0)),
        train_duplicate_time_relative_scale_error=float(np.mean(duplicate_relative)),
        train_duplicate_time_exact_scale_rate=float(np.mean(duplicate_difference == 0)),
        training_prediction_boundary_share=boundary_share,
        training_prediction_mode=mode,
        training_prediction_mode_share=mode_share,
        distinct_training_count_total=distinct_total,
        prediction_boundary_and_mode_penalty=boundary_share + mode_share,
        degenerate_candidate_flag=degenerate,
        synthetic_count_sweep_sample_total=int(count_sweep["sample_total"]),
        synthetic_stress_sample_total=int(stress["sample_total"]),
        train_transform_pair_total=int(transform_difference.size),
        train_duplicate_pair_total=int(duplicate_difference.size),
    )


def _evaluate_candidates(
    *,
    training_sequences: Sequence[PoseSequence],
    predictor: Callable[
        [Sequence[PoseSequence]],
        Mapping[str, Mapping[str, int]],
    ]
    | None = None,
) -> tuple[
    tuple[CandidateAudit, ...],
    dict[str, Any],
    dict[str, Any],
]:
    methods = _candidate_methods()
    count_sweep = generate_count_sweep(
        minimum=_COUNT_MINIMUM,
        maximum=_COUNT_MAXIMUM,
        frames=256,
        seed=_SYNTHETIC_SEED,
    )
    stress_suite = synthetic_stress_suite(
        count=float(_SYNTHETIC_STRESS_COUNT),
        frames=256,
        seed=_SYNTHETIC_SEED,
    )
    stress_items = tuple(
        (name, sample)
        for name, sample in stress_suite.items()
        if name != "clean"
    )
    training_items = tuple(training_sequences)
    transformed_by_id = {
        sequence.video_id: _training_transforms(sequence) for sequence in training_items
    }
    duplicate_by_id = {
        sequence.video_id: _duplicate_time(sequence) for sequence in training_items
    }
    work_items = (
        tuple(sample.sequence for sample in count_sweep)
        + tuple(sample.sequence for _, sample in stress_items)
        + training_items
        + tuple(
            transformed
            for sequence in training_items
            for transformed in transformed_by_id[sequence.video_id].values()
        )
        + tuple(duplicate_by_id[sequence.video_id] for sequence in training_items)
    )
    if predictor is None:
        prediction_rows = _predict_many(work_items)
    else:
        prediction_rows = dict(predictor(work_items))
    if tuple(prediction_rows) != tuple(item.video_id for item in work_items):
        raise RuntimeError("prediction rows differ from the frozen work-item order")
    for row in prediction_rows.values():
        if set(row) != set(methods):
            raise RuntimeError("prediction row differs from the 512-candidate grid")

    count_truth = tuple(sample.target_count for sample in count_sweep)
    stress_truth = tuple(sample.target_count for _, sample in stress_items)
    audits: list[CandidateAudit] = []
    for method in methods:
        originals = [
            prediction_rows[sequence.video_id][method] for sequence in training_items
        ]
        transforms = [
            [
                prediction_rows[transformed.video_id][method]
                for transformed in transformed_by_id[sequence.video_id].values()
            ]
            for sequence in training_items
        ]
        duplicates = [
            prediction_rows[duplicate_by_id[sequence.video_id].video_id][method]
            for sequence in training_items
        ]
        audits.append(
            _score_candidate(
                method,
                count_sweep_predictions=[
                    prediction_rows[sample.sequence.video_id][method]
                    for sample in count_sweep
                ],
                count_sweep_truth=count_truth,
                stress_predictions=[
                    prediction_rows[sample.sequence.video_id][method]
                    for _, sample in stress_items
                ],
                stress_truth=stress_truth,
                original_predictions=originals,
                transformed_predictions=transforms,
                duplicate_predictions=duplicates,
            )
        )
    ranked = tuple(sorted(audits, key=lambda audit: audit.rank_key))
    selected = ranked[0]
    selected_predictions = {
        "synthetic_count_sweep": [
            {
                "sample_id": sample.sequence.video_id,
                "synthetic_target_count": sample.target_count,
                "prediction": prediction_rows[sample.sequence.video_id][
                    selected.candidate_key
                ],
            }
            for sample in count_sweep
        ],
        "synthetic_stress": [
            {
                "sample_id": name,
                "synthetic_target_count": sample.target_count,
                "prediction": prediction_rows[sample.sequence.video_id][
                    selected.candidate_key
                ],
            }
            for name, sample in stress_items
        ],
        "unlabeled_train": [
            {
                "video_id": sequence.video_id,
                "original_prediction": prediction_rows[sequence.video_id][
                    selected.candidate_key
                ],
                "transform_predictions": {
                    name: prediction_rows[transformed.video_id][selected.candidate_key]
                    for name, transformed in transformed_by_id[sequence.video_id].items()
                },
                "duplicate_time_prediction": prediction_rows[
                    duplicate_by_id[sequence.video_id].video_id
                ][selected.candidate_key],
            }
            for sequence in training_items
        ],
    }
    transformation_hashes = {
        sequence.video_id: {
            "original_pose_sha256": pose_sequence_sha256(sequence),
            "transforms": {
                name: pose_sequence_sha256(transformed)
                for name, transformed in transformed_by_id[sequence.video_id].items()
            },
            "duplicate_time_pose_sha256": pose_sequence_sha256(
                duplicate_by_id[sequence.video_id]
            ),
        }
        for sequence in training_items
    }
    return ranked, selected_predictions, transformation_hashes


def run_selector(
    *,
    train_inputs: Path,
    train_pose_cache_dir: Path,
    output: Path,
) -> dict[str, Any]:
    """Run the complete selector and write one immutable target-free artifact."""

    if output.exists():
        raise FileExistsError(f"refusing to overwrite selector output: {output}")
    source_path = Path(__file__).resolve()
    source_bytes = _read_stable_bytes(source_path)
    source_sha256 = _sha256_bytes(source_bytes)
    sidecar_bytes = _read_stable_bytes(train_inputs)
    sidecar_sha256 = _sha256_bytes(sidecar_bytes)
    _validate_canonical_train337_binding(sidecar_sha256=sidecar_sha256)
    manifest = load_pose_input_manifest(train_inputs, validate_exact=True)
    _validate_canonical_train337_binding(
        sidecar_sha256=sidecar_sha256,
        manifest=manifest,
    )
    selected_records = _hash_select_training_records(manifest)
    training_sequences, cache_snapshot = load_pose_cache_set(
        selected_records,
        cache_dir=train_pose_cache_dir,
        pose_fingerprint=_FROZEN_POSE_FINGERPRINT,
    )
    if tuple(sequence.video_id for sequence in training_sequences) != tuple(
        record.video_id for record in selected_records
    ):
        raise RuntimeError("pose-cache order differs from the hash-selected sidecar order")

    ranked, selected_predictions, transformation_hashes = _evaluate_candidates(
        training_sequences=training_sequences,
    )
    selected = ranked[0]
    if _sha256_bytes(_read_stable_bytes(source_path)) != source_sha256:
        raise RuntimeError("selector source changed during execution")
    if _sha256_bytes(_read_stable_bytes(train_inputs)) != sidecar_sha256:
        raise RuntimeError("train337 sidecar changed during execution")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "classification": _CLASSIFICATION,
        "eligible_for_paper_table": False,
        "selection_status": "exploratory-derived_ineligible",
        "candidate_total": len(ranked),
        "selected_candidate": selected.candidate_key,
        "selected_parameters": dict(selected.parameters),
        "selected_rank_key": list(selected.rank_key),
        "objective": {
            "type": "strict_lexicographic_minimum",
            "lexicographic_order": list(_LEXICOGRAPHIC_OBJECTIVE),
            "candidate_key_final_tie_break": True,
            "sources": [
                "deterministic_internal_synthetic_truth",
                "unlabeled_train337_prediction_consistency",
            ],
            "degeneracy_gate": {
                "boundary_share_limit_inclusive": _DEGENERATE_BOUNDARY_SHARE_LIMIT,
                "mode_share_limit_inclusive": _DEGENERATE_MODE_SHARE_LIMIT,
                "minimum_distinct_predictions": _DEGENERATE_MINIMUM_DISTINCT_COUNTS,
                "boundary_values": [_COUNT_MINIMUM, _COUNT_MAXIMUM],
                "continuous_penalty": "boundary_share + mode_share",
            },
        },
        "frozen_protocol": {
            "source_exploration": "tmp/explore_local_frequency_readouts.py",
            "grid": {
                "trims": list(_TRIMS),
                "features": list(_FEATURE_NAMES),
                "normalized_dimensions": list(_NORMALIZED_DIMENSIONS),
                "minimum_cycles": list(_MINIMUM_CYCLES),
                "windows": list(_WINDOWS),
                "statistics": list(_STATISTICS),
                "scale_reducers": list(_SCALE_REDUCERS),
            },
            "train_hash_sample_total": _TRAIN_SAMPLE_TOTAL,
            "train_hash_domain_hex": _TRAIN_SAMPLE_DOMAIN.hex(),
            "transforms": list(_TRANSFORM_NAMES),
            "transform_seed": _TRANSFORM_SEED,
            "rotation_degrees": list(_ROTATION_DEGREES),
            "scale_multiplier": _SCALE_MULTIPLIER,
            "translation_delta_all_axes": _TRANSLATION_DELTA,
            "occlusion_time_fraction": _OCCLUSION_TIME_FRACTION,
            "occluded_joint_total": _OCCLUDED_JOINT_TOTAL,
            "noise_standard_deviation": _NOISE_STANDARD_DEVIATION,
            "speed_warp_endpoints": list(_SPEED_WARP_ENDPOINTS),
            "duplicate_time_operation": "concatenate_sequence_with_itself",
            "duplicate_time_expected_prediction": "min(40, 2 * original_prediction)",
            "synthetic_count_range_inclusive": [_COUNT_MINIMUM, _COUNT_MAXIMUM],
            "synthetic_seed": _SYNTHETIC_SEED,
            "synthetic_stress_count": _SYNTHETIC_STRESS_COUNT,
            "worker_total": _WORKER_TOTAL,
        },
        "label_firewall": {
            "cli_accepts_only_train337_sidecar_pose_cache_and_output": True,
            "pose_sidecar_rejects_count_and_action_fields_before_deserialization": True,
            "development_inputs_loaded": False,
            "development_targets_loaded": False,
            "test_inputs_loaded": False,
            "test_targets_loaded": False,
            "dataset_count_labels_loaded": False,
            "dataset_action_labels_loaded": False,
            "synthetic_truth_used": True,
            "real_pose_objective_uses_predictions_only": True,
        },
        "source": {
            "relative_path": _SOURCE_RELATIVE_PATH,
            "sha256": source_sha256,
            "python_implementation": os.sys.implementation.name,
            "python_version": os.sys.version.split()[0],
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
        },
        "inputs": {
            "train337_sidecar_sha256": sidecar_sha256,
            "frozen_train337_sidecar_sha256": _FROZEN_TRAIN337_SIDECAR_SHA256,
            "train337_sidecar_fingerprint": manifest.fingerprint,
            "frozen_train337_sidecar_fingerprint": (
                _FROZEN_TRAIN337_MANIFEST_FINGERPRINT
            ),
            "train337_identity_sha256": pose_input_identity_sha256(manifest.records),
            "frozen_train337_identity_sha256": _FROZEN_TRAIN337_IDENTITY_SHA256,
            "pose_fingerprint": _FROZEN_POSE_FINGERPRINT,
            "selected_train_video_ids": [
                record.video_id for record in selected_records
            ],
            "selected_train_identity_sha256": pose_input_identity_sha256(
                selected_records
            ),
            "selected_pose_cache_set": cache_snapshot.to_dict(),
            "selected_pose_cache_set_sha256": cache_snapshot.fingerprint,
            "transformed_pose_sha256": transformation_hashes,
        },
        "selected_predictions": selected_predictions,
        "candidate_audits": [
            {"rank": rank, **audit.to_dict()}
            for rank, audit in enumerate(ranked, start=1)
        ],
    }
    _write_new_json(output, payload)
    return payload


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select one exploratory local-frequency v2 candidate from synthetic "
            "truth and unlabeled canonical UCFRep train337 consistency only."
        )
    )
    parser.add_argument("--train-inputs", type=Path, required=True)
    parser.add_argument("--train-pose-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    arguments = _parse_arguments(argv)
    payload = run_selector(
        train_inputs=arguments.train_inputs,
        train_pose_cache_dir=arguments.train_pose_cache_dir,
        output=arguments.output,
    )
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "selected_candidate": payload["selected_candidate"],
                "selected_rank_key": payload["selected_rank_key"],
                "candidate_total": payload["candidate_total"],
                "paper_table_eligible": payload["eligible_for_paper_table"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
