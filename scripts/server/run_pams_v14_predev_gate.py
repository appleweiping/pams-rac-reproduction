"""Run the frozen, target-free PAMS-v14 epoch-11 pre-development gate.

The only scientific inputs accepted by this program are the v14 experiment
configuration, its bound epoch-11 encoder checkpoint, and the checkpoint's
337 training pose-cache files.  Development/test identities, media, poses,
action classes, and repetition targets are deliberately outside the API.
Synthetic truth is generated inside this file and is used only for primitive
diagnostics.

This gate is an independently inferred reproduction diagnostic.  Passing it
can authorize a separate, one-shot dev84 prediction run; it never authorizes
dev84 scoring by this process or any sealed-test access.

PyTorch checkpoints are pickle containers.  This runner must only be used
with checkpoints produced by this repository in the trusted experiment
environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import stat
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pams.config import PAMSConfig, load_config
from pams.data import PoseCacheSetSnapshot, per_frame_minmax, uniform_resample
from pams.diagnostics import (
    _device,
    _identifier_commitment,
    _load_training_poses,
    _peek_checkpoint,
    _stable_file_sha256,
    _summary,
)
from pams.model import PAMSModel
from pams.period import estimate_period_from_embedding_velocity_vectors
from pams.reproducibility import hardware_fingerprint, sha256_json
from pams.training import collate_pose_sequences, load_model_checkpoint
from pams.types import PoseSequence

_ARTIFACT_TYPE = "pams_v14_epoch11_target_free_predev_gate"
_RECEIPT_TYPE = "pams_v14_epoch11_target_free_predev_gate_receipt"
_CLASSIFICATION = "inferred target-free pre-development diagnostic"
_EXPECTED_TRAINING_VIDEOS = 337
_EXPECTED_EPOCH = 11
_STREAM_WINDOW_FRAMES = 128
_STREAM_STRIDE_FRAMES = 32
_TIME_SCALE_FACTORS = (0.50, 0.75)
_SYNTHETIC_COUNTS = tuple(range(2, 41))
_HARMONIC_ORDERS = tuple(range(2, 8))
_HARMONIC_FUNDAMENTAL_PERIOD = 64.0
_THRESHOLDS: dict[str, float] = {
    "training_boundary_share_maximum_exclusive": 0.25,
    "training_mode_share_maximum_exclusive": 0.25,
    "training_period_stream_std_median_minimum": 0.05,
    "training_time_scale_median_relative_error_maximum": 0.15,
    "synthetic_count_median_relative_error_maximum": 0.05,
    "synthetic_static_confidence_maximum": 0.05,
    "synthetic_harmonic_fundamental_fraction_minimum": 0.95,
}


@dataclass(frozen=True, slots=True)
class _EncodedTrainingSample:
    """One label-free global and windowed encoder-period result."""

    video_id: str
    valid_frames: int
    period_frames: float
    confidence: float
    period_stream: tuple[float, ...]
    confidence_stream: tuple[float, ...]
    period_stream_std: float


def _finite_float(value: Tensor | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("period diagnostic produced a non-finite value")
    return result


def _period_estimate(
    embeddings: Tensor,
    valid_mask: Tensor,
    *,
    config: PAMSConfig,
) -> tuple[Tensor, Tensor]:
    return estimate_period_from_embedding_velocity_vectors(
        embeddings,
        minimum=config.period.minimum,
        maximum=config.period.maximum,
        valid_mask=valid_mask,
    )


def _window_starts(time: int) -> tuple[int, ...]:
    if time < _STREAM_WINDOW_FRAMES:
        raise ValueError(
            f"period-stream diagnostic requires at least {_STREAM_WINDOW_FRAMES} frames"
        )
    starts = list(
        range(
            0,
            time - _STREAM_WINDOW_FRAMES + 1,
            _STREAM_STRIDE_FRAMES,
        )
    )
    tail = time - _STREAM_WINDOW_FRAMES
    if not starts or starts[-1] != tail:
        starts.append(tail)
    return tuple(starts)


def _encode_training_sequences(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> tuple[_EncodedTrainingSample, ...]:
    """Encode train337 and construct a fixed sliding-window period stream."""

    samples: list[_EncodedTrainingSample] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            embeddings = model.encoder(batch.poses, batch.valid_mask)
            periods, confidences = _period_estimate(
                embeddings,
                batch.valid_mask,
                config=config,
            )
            starts = _window_starts(embeddings.shape[1])
            window_embeddings = torch.cat(
                [
                    embeddings[
                        :,
                        offset : offset + _STREAM_WINDOW_FRAMES,
                    ]
                    for offset in starts
                ],
                dim=0,
            )
            window_masks = torch.cat(
                [
                    batch.valid_mask[
                        :,
                        offset : offset + _STREAM_WINDOW_FRAMES,
                    ]
                    for offset in starts
                ],
                dim=0,
            )
            window_periods, window_confidences = _period_estimate(
                window_embeddings,
                window_masks,
                config=config,
            )
            # Concatenation is window-major: [window0/all B, window1/all B, ...].
            window_periods = window_periods.reshape(len(starts), batch.batch_size)
            window_confidences = window_confidences.reshape(
                len(starts),
                batch.batch_size,
            )
            for index, video_id in enumerate(batch.video_ids):
                stream = tuple(
                    _finite_float(value) for value in window_periods[:, index].detach().cpu()
                )
                confidence_stream = tuple(
                    _finite_float(value) for value in window_confidences[:, index].detach().cpu()
                )
                positive_periods = [
                    period
                    for period, confidence in zip(
                        stream,
                        confidence_stream,
                        strict=True,
                    )
                    if confidence > 0.0
                ]
                stream_std = float(np.std(positive_periods)) if len(positive_periods) >= 2 else 0.0
                samples.append(
                    _EncodedTrainingSample(
                        video_id=video_id,
                        valid_frames=int(batch.valid_mask[index].sum()),
                        period_frames=_finite_float(periods[index]),
                        confidence=_finite_float(confidences[index]),
                        period_stream=stream,
                        confidence_stream=confidence_stream,
                        period_stream_std=stream_std,
                    )
                )
    return tuple(samples)


def _encode_global_periods(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> tuple[tuple[str, float, float], ...]:
    results: list[tuple[str, float, float]] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            embeddings = model.encoder(batch.poses, batch.valid_mask)
            periods, confidences = _period_estimate(
                embeddings,
                batch.valid_mask,
                config=config,
            )
            results.extend(
                (
                    video_id,
                    _finite_float(period),
                    _finite_float(confidence),
                )
                for video_id, period, confidence in zip(
                    batch.video_ids,
                    periods,
                    confidences,
                    strict=True,
                )
            )
    return tuple(results)


def _stable_period_key(period: float) -> str:
    return format(float(period), ".9g")


def _training_distribution(
    samples: Sequence[_EncodedTrainingSample],
    *,
    minimum: int,
    maximum: int,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("training distribution requires at least one sample")
    periods = np.asarray(
        [sample.period_frames for sample in samples],
        dtype=np.float64,
    )
    confidences = np.asarray(
        [sample.confidence for sample in samples],
        dtype=np.float64,
    )
    if not np.isfinite(periods).all() or not np.isfinite(confidences).all():
        raise RuntimeError("training distribution contains non-finite values")
    minimum_share = float(np.mean(periods <= float(minimum) + 1e-6))
    maximum_share = float(np.mean(periods >= float(maximum) - 1e-6))
    boundary_share = float(
        np.mean((periods <= float(minimum) + 1e-6) | (periods >= float(maximum) - 1e-6))
    )
    frequencies = Counter(_stable_period_key(value) for value in periods)
    mode_key, mode_total = min(
        frequencies.items(),
        key=lambda item: (-item[1], item[0]),
    )
    stream_stds = np.asarray(
        [sample.period_stream_std for sample in samples],
        dtype=np.float64,
    )
    return {
        "record_total": len(samples),
        "minimum_period_share": minimum_share,
        "maximum_period_share": maximum_share,
        "boundary_share": boundary_share,
        "mode_period_frames": float(mode_key),
        "mode_frequency": mode_total,
        "mode_share": mode_total / len(samples),
        "unique_period_total": len(frequencies),
        "zero_confidence_share": float(np.mean(confidences <= 0.0)),
        "period_frames": _summary(periods),
        "confidence": _summary(confidences),
        "period_stream_std": _summary(stream_stds),
        "period_histogram": dict(sorted(frequencies.items())),
        "period_histogram_key_format": "finite_float_9_significant_digits",
    }


def _resample_for_time_scale(
    sequence: PoseSequence,
    factor: float,
) -> PoseSequence:
    target_frames = int(round(sequence.num_frames * factor))
    if target_frames < 2:
        raise ValueError("time-scale factor produces fewer than two frames")
    xyz, valid = uniform_resample(
        sequence.xyz,
        sequence.valid_mask,
        target_frames=target_frames,
    )
    return PoseSequence(
        video_id=f"{sequence.video_id}::time-scale-{format(factor, 'g')}",
        fps=sequence.fps * factor,
        xyz=xyz,
        valid_mask=valid,
    )


def _time_scale_consistency(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    baseline: Sequence[_EncodedTrainingSample],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if [item.video_id for item in baseline] != [item.video_id for item in sequences]:
        raise ValueError("time-scale baseline does not match training sequence order")
    all_errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for factor in _TIME_SCALE_FACTORS:
        scaled = tuple(_resample_for_time_scale(sequence, factor) for sequence in sequences)
        predictions = _encode_global_periods(
            model,
            scaled,
            config=config,
            device=device,
            batch_size=batch_size,
        )
        for source, prediction in zip(baseline, predictions, strict=True):
            _, scaled_period, scaled_confidence = prediction
            expected = source.period_frames * factor
            eligible = (
                source.confidence > 0.0
                and scaled_confidence > 0.0
                and config.period.minimum <= expected <= config.period.maximum
            )
            relative_error = abs(scaled_period - expected) / expected if eligible else None
            if relative_error is not None:
                if not math.isfinite(relative_error):
                    raise RuntimeError("time-scale diagnostic contains a non-finite error")
                all_errors.append(relative_error)
            rows.append(
                {
                    "video_id": source.video_id,
                    "time_scale_factor": factor,
                    "baseline_period_frames": source.period_frames,
                    "baseline_confidence": source.confidence,
                    "expected_scaled_period_frames": expected,
                    "scaled_period_frames": scaled_period,
                    "scaled_confidence": scaled_confidence,
                    "eligible": eligible,
                    "relative_error": relative_error,
                }
            )
    return {
        "algorithm": (
            "Uniformly resample every checkpoint-bound training pose to "
            "round(T*factor) frames for factors 0.5 and 0.75, rerun the same "
            "encoder/vector-ACF estimator, and compare P_scaled with factor*P. "
            "Only positive-confidence estimates whose expected scaled period "
            "lies inside the frozen 4--128 range enter the error summary."
        ),
        "factors": list(_TIME_SCALE_FACTORS),
        "candidate_comparison_total": len(rows),
        "eligible_comparison_total": len(all_errors),
        "relative_error": _summary(all_errors),
        "rows": rows,
    }


def _synthetic_pose(
    *,
    video_id: str,
    frames: int,
    fundamental_period: float,
    harmonic_order: int | None = None,
) -> PoseSequence:
    time = np.arange(frames, dtype=np.float64)
    phase = 2.0 * np.pi * time / fundamental_period
    joint_phase = np.linspace(0.0, 2.0 * np.pi, 33, endpoint=False)
    joint_scale = np.linspace(0.7, 1.3, 33)
    fundamental = np.sin(phase[:, None] + joint_phase[None, :])
    cosine = np.cos(phase[:, None] - 0.5 * joint_phase[None, :])
    if harmonic_order is not None:
        harmonic_phase = harmonic_order * phase[:, None] + joint_phase[None, :] + 0.37
        fundamental = 0.25 * fundamental + 0.75 * np.sin(harmonic_phase)
        cosine = 0.25 * cosine + 0.75 * np.cos(harmonic_phase)
    xyz = np.zeros((frames, 33, 3), dtype=np.float64)
    xyz[:, :, 0] = 0.50 + 0.20 * joint_scale[None, :] * fundamental
    xyz[:, :, 1] = 0.45 + 0.16 * joint_scale[None, :] * cosine
    xyz[:, :, 2] = 0.40 + 0.10 * joint_scale[None, :] * np.sin(
        phase[:, None] + 0.25 * joint_phase[None, :]
    )
    valid = np.ones(frames, dtype=np.bool_)
    return PoseSequence(
        video_id=video_id,
        fps=30.0,
        xyz=per_frame_minmax(xyz, valid),
        valid_mask=valid,
    )


def _synthetic_count_gate(
    model: PAMSModel,
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    duration = float(config.data.frames - 1)
    sequences = tuple(
        _synthetic_pose(
            video_id=f"synthetic-count-{count:02d}",
            frames=config.data.frames,
            fundamental_period=duration / count,
        )
        for count in _SYNTHETIC_COUNTS
    )
    predictions = _encode_global_periods(
        model,
        sequences,
        config=config,
        device=device,
        batch_size=batch_size,
    )
    errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for count, (_, period, confidence) in zip(
        _SYNTHETIC_COUNTS,
        predictions,
        strict=True,
    ):
        expected = duration / count
        error = abs(period - expected) / expected
        errors.append(error)
        rows.append(
            {
                "synthetic_count": count,
                "expected_period_frames": expected,
                "predicted_period_frames": period,
                "confidence": confidence,
                "relative_error": error,
            }
        )
    return {
        "truth_source": "deterministic in-run synthetic generation only",
        "counts": list(_SYNTHETIC_COUNTS),
        "record_total": len(rows),
        "relative_error": _summary(errors),
        "rows": rows,
    }


def _synthetic_static_gate(
    model: PAMSModel,
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    shape = (
        config.data.frames,
        config.data.keypoints,
        config.data.coordinates,
    )
    valid = np.ones(config.data.frames, dtype=np.bool_)
    zeros = np.zeros(shape, dtype=np.float32)
    template = np.linspace(
        0.0,
        1.0,
        num=config.data.keypoints * config.data.coordinates,
        dtype=np.float32,
    ).reshape(config.data.keypoints, config.data.coordinates)
    repeated = np.repeat(
        template[None, :, :],
        config.data.frames,
        axis=0,
    )
    sequences = (
        PoseSequence(
            video_id="synthetic-static-zero",
            fps=30.0,
            xyz=zeros,
            valid_mask=valid,
        ),
        PoseSequence(
            video_id="synthetic-static-template",
            fps=30.0,
            xyz=repeated,
            valid_mask=valid,
        ),
    )
    predictions = _encode_global_periods(
        model,
        sequences,
        config=config,
        device=device,
        batch_size=batch_size,
    )
    rows = [
        {
            "probe": identifier,
            "period_frames": period,
            "confidence": confidence,
        }
        for identifier, period, confidence in predictions
    ]
    return {
        "truth_source": "deterministic in-run static generation only",
        "record_total": len(rows),
        "maximum_confidence": max(row["confidence"] for row in rows),
        "rows": rows,
    }


def _synthetic_harmonic_gate(
    model: PAMSModel,
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    sequences = tuple(
        _synthetic_pose(
            video_id=f"synthetic-harmonic-{order}",
            frames=config.data.frames,
            fundamental_period=_HARMONIC_FUNDAMENTAL_PERIOD,
            harmonic_order=order,
        )
        for order in _HARMONIC_ORDERS
    )
    predictions = _encode_global_periods(
        model,
        sequences,
        config=config,
        device=device,
        batch_size=batch_size,
    )
    rows: list[dict[str, Any]] = []
    recovered = 0
    for order, (_, period, confidence) in zip(
        _HARMONIC_ORDERS,
        predictions,
        strict=True,
    ):
        relative_error = abs(period - _HARMONIC_FUNDAMENTAL_PERIOD) / _HARMONIC_FUNDAMENTAL_PERIOD
        selected = relative_error <= 0.05 and confidence > 0.0
        recovered += int(selected)
        rows.append(
            {
                "dominant_harmonic_order": order,
                "expected_fundamental_period_frames": (_HARMONIC_FUNDAMENTAL_PERIOD),
                "predicted_period_frames": period,
                "confidence": confidence,
                "relative_error": relative_error,
                "fundamental_selected_within_5_percent": selected,
            }
        )
    return {
        "truth_source": "deterministic in-run synthetic generation only",
        "harmonic_orders": list(_HARMONIC_ORDERS),
        "fundamental_period_frames": _HARMONIC_FUNDAMENTAL_PERIOD,
        "fundamental_amplitude": 0.25,
        "dominant_harmonic_amplitude": 0.75,
        "record_total": len(rows),
        "fundamental_selection_fraction": recovered / len(rows),
        "rows": rows,
    }


def _criterion(
    value: float | None,
    *,
    operator: str,
    threshold: float,
) -> dict[str, float | str | bool | None]:
    finite = value is not None and math.isfinite(value)
    if operator == "<":
        passed = finite and value < threshold
    elif operator == "<=":
        passed = finite and value <= threshold
    elif operator == ">=":
        passed = finite and value >= threshold
    else:
        raise ValueError(f"unsupported gate operator: {operator}")
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "pass": bool(passed),
    }


def _gate_decision(
    *,
    boundary_share: float,
    mode_share: float,
    period_stream_std_median: float | None,
    time_scale_median_relative_error: float | None,
    synthetic_count_median_relative_error: float | None,
    static_maximum_confidence: float,
    harmonic_fundamental_fraction: float,
) -> dict[str, Any]:
    criteria = {
        "training_boundary_share": _criterion(
            boundary_share,
            operator="<",
            threshold=_THRESHOLDS["training_boundary_share_maximum_exclusive"],
        ),
        "training_mode_share": _criterion(
            mode_share,
            operator="<",
            threshold=_THRESHOLDS["training_mode_share_maximum_exclusive"],
        ),
        "training_period_stream_std_median": _criterion(
            period_stream_std_median,
            operator=">=",
            threshold=_THRESHOLDS["training_period_stream_std_median_minimum"],
        ),
        "training_time_scale_median_relative_error": _criterion(
            time_scale_median_relative_error,
            operator="<=",
            threshold=_THRESHOLDS["training_time_scale_median_relative_error_maximum"],
        ),
        "synthetic_count_median_relative_error": _criterion(
            synthetic_count_median_relative_error,
            operator="<=",
            threshold=_THRESHOLDS["synthetic_count_median_relative_error_maximum"],
        ),
        "synthetic_static_maximum_confidence": _criterion(
            static_maximum_confidence,
            operator="<=",
            threshold=_THRESHOLDS["synthetic_static_confidence_maximum"],
        ),
        "synthetic_harmonic_fundamental_fraction": _criterion(
            harmonic_fundamental_fraction,
            operator=">=",
            threshold=_THRESHOLDS["synthetic_harmonic_fundamental_fraction_minimum"],
        ),
    }
    passed = all(bool(item["pass"]) for item in criteria.values())
    return {
        "thresholds_frozen_before_epoch11_checkpoint_evaluation": True,
        "criteria": criteria,
        "overall_pass": passed,
        "dev84_prediction_authorized": passed,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }


def _validate_v14_config(config: PAMSConfig) -> None:
    expected = {
        "protocol": "ucfrep_526",
        "seed": 2026,
        "frames": 256,
        "period_minimum": 4,
        "period_maximum": 128,
        "pose_energy_epochs": 10,
        "post_warmup_source": "embedding_velocity_vector_acf",
        "model_dim": 512,
        "embedding_dim": 512,
        "layers": 4,
        "heads": 16,
        "feedforward_dim": 2048,
        "position_encoding_mode": "sinusoidal",
        "effective_batch_size": 32,
        "loss_scales": (0.5, 1.0, 1.5),
    }
    actual = {
        "protocol": config.protocol,
        "seed": config.seed,
        "frames": config.data.frames,
        "period_minimum": config.period.minimum,
        "period_maximum": config.period.maximum,
        "pose_energy_epochs": config.period.pose_energy_epochs,
        "post_warmup_source": config.period.post_warmup_source,
        "model_dim": config.model.model_dim,
        "embedding_dim": config.model.embedding_dim,
        "layers": config.model.layers,
        "heads": config.model.heads,
        "feedforward_dim": config.model.feedforward_dim,
        "position_encoding_mode": config.model.position_encoding_mode,
        "effective_batch_size": config.training.effective_batch_size,
        "loss_scales": config.loss.scales,
    }
    drift = {
        name: {"expected": expected[name], "actual": value}
        for name, value in actual.items()
        if value != expected[name]
    }
    augmentation = config.training.skeleton_augmentation
    if not augmentation.enabled:
        drift["skeleton_augmentation.enabled"] = {
            "expected": True,
            "actual": False,
        }
    if augmentation.rotation_degrees != (15.0, 15.0, 15.0):
        drift["skeleton_augmentation.rotation_degrees"] = {
            "expected": (15.0, 15.0, 15.0),
            "actual": augmentation.rotation_degrees,
        }
    if augmentation.scale_range != (0.85, 1.15):
        drift["skeleton_augmentation.scale_range"] = {
            "expected": (0.85, 1.15),
            "actual": augmentation.scale_range,
        }
    if augmentation.jitter_std != 0.01:
        drift["skeleton_augmentation.jitter_std"] = {
            "expected": 0.01,
            "actual": augmentation.jitter_std,
        }
    if drift:
        raise ValueError(
            "v14 predev gate configuration drift: " + json.dumps(drift, sort_keys=True)
        )


def _checkpoint_epoch11_metadata(
    checkpoint: Path,
    *,
    config: PAMSConfig,
) -> dict[str, Any]:
    payload = torch.load(
        checkpoint,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    completed = payload.get("completed_epochs")
    if completed != _EXPECTED_EPOCH:
        raise ValueError("v14 epoch-11 predev gate requires completed_epochs=11")
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != _EXPECTED_EPOCH:
        raise ValueError("checkpoint epoch history must contain exactly 11 rows")
    sources: list[str] = []
    for expected_epoch, row in enumerate(history, start=1):
        if not isinstance(row, Mapping) or row.get("epoch") != expected_epoch:
            raise ValueError("checkpoint epoch history must be consecutive from one")
        source = row.get("period_source")
        if not isinstance(source, str):
            raise ValueError("checkpoint history period_source is missing")
        sources.append(source)
    expected_sources = ["pose"] * config.period.pose_energy_epochs + [
        "embedding_velocity_vector_acf"
    ]
    if sources != expected_sources:
        raise ValueError(
            "checkpoint does not contain ten pose-proxy epochs followed by "
            "one embedding-velocity-vector-ACF epoch"
        )
    return {
        "completed_epochs": completed,
        "period_sources": sources,
        "first_post_warmup_epoch": _EXPECTED_EPOCH,
    }


def _runtime_provenance(
    *,
    runner_sha256: str,
    device: torch.device,
) -> dict[str, Any]:
    container = {
        name: os.environ.get(name)
        for name in (
            "PAMS_CONTAINER_IMAGE_ID",
            "PAMS_CONTAINER_ENVIRONMENT_SHA256",
            "PAMS_CONTAINER_SOURCE_REVISION",
        )
    }
    supplied = [value for value in container.values() if value]
    if supplied and len(supplied) != len(container):
        raise RuntimeError("container runtime provenance is incomplete")
    return {
        "runner_sha256": runner_sha256,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable_name": Path(sys.executable).name,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device": str(device),
        "torch_deterministic_algorithms_enabled": (torch.are_deterministic_algorithms_enabled()),
        "container": container if supplied else None,
    }


def run_predev_gate(
    checkpoint_path: str | Path,
    config_path: str | Path,
    pose_cache_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate the exact epoch-11 checkpoint without any external targets."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or batch_size > 32
    ):
        raise ValueError("batch_size must be an integer in [1, 32]")
    checkpoint = Path(checkpoint_path)
    configuration = Path(config_path)
    cache = Path(pose_cache_dir)
    runner = Path(__file__)
    checkpoint_identity = _stable_file_sha256(checkpoint)
    config_identity = _stable_file_sha256(configuration)
    runner_identity = _stable_file_sha256(runner)
    config = load_config(configuration)
    _validate_v14_config(config)
    stage, provenance = _peek_checkpoint(checkpoint, config)
    if stage != "encoder":
        raise ValueError("v14 epoch-11 predev gate requires an encoder checkpoint")
    if len(provenance.training_video_ids) != _EXPECTED_TRAINING_VIDEOS:
        raise ValueError(
            "v14 epoch-11 predev gate requires exactly 337 checkpoint-bound training videos"
        )
    checkpoint_metadata = _checkpoint_epoch11_metadata(
        checkpoint,
        config=config,
    )
    resolved_device = _device(device)
    model = load_model_checkpoint(
        checkpoint,
        config,
        device=resolved_device,
        expected_stage="encoder",
        expected_provenance=provenance,
    ).eval()
    sequences, sample_receipts, full_receipts, selected_ids = _load_training_poses(
        pose_cache_dir=cache,
        provenance=provenance,
        config=config,
        sample_size=0,
        seed=config.seed,
    )
    if len(sequences) != _EXPECTED_TRAINING_VIDEOS:
        raise RuntimeError("checkpoint-bound pose loader did not return train337")
    if any(sequence.num_frames != config.data.frames for sequence in sequences):
        raise ValueError("every v14 training pose must contain exactly 256 frames")
    sample_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=sample_receipts,
    )
    full_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=full_receipts,
    )
    if sample_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("full train337 selection and pose snapshots differ")
    if full_snapshot.fingerprint != provenance.pose_cache_set_sha256:
        raise ValueError("full train337 pose-cache set does not match checkpoint provenance")

    count_synthetic = _synthetic_count_gate(
        model,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    static_synthetic = _synthetic_static_gate(
        model,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    harmonic_synthetic = _synthetic_harmonic_gate(
        model,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    encoded = _encode_training_sequences(
        model,
        sequences,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    distribution = _training_distribution(
        encoded,
        minimum=config.period.minimum,
        maximum=config.period.maximum,
    )
    time_scale = _time_scale_consistency(
        model,
        sequences,
        encoded,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    decision = _gate_decision(
        boundary_share=float(distribution["boundary_share"]),
        mode_share=float(distribution["mode_share"]),
        period_stream_std_median=distribution["period_stream_std"]["median"],
        time_scale_median_relative_error=time_scale["relative_error"]["median"],
        synthetic_count_median_relative_error=(count_synthetic["relative_error"]["median"]),
        static_maximum_confidence=float(static_synthetic["maximum_confidence"]),
        harmonic_fundamental_fraction=float(harmonic_synthetic["fundamental_selection_fraction"]),
    )

    # Re-read and re-hash all scientific inputs after inference.  This catches
    # replacement or mutation during a long GPU diagnostic.
    _, _, final_receipts, final_ids = _load_training_poses(
        pose_cache_dir=cache,
        provenance=provenance,
        config=config,
        sample_size=2,
        seed=config.seed,
    )
    final_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=final_receipts,
    )
    if final_ids != selected_ids[:2]:
        raise RuntimeError("training pose selection changed during predev gate")
    if final_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("training pose cache changed during predev gate")
    if _stable_file_sha256(checkpoint) != checkpoint_identity:
        raise RuntimeError("checkpoint changed during predev gate")
    if _stable_file_sha256(configuration) != config_identity:
        raise RuntimeError("configuration changed during predev gate")
    if _stable_file_sha256(runner) != runner_identity:
        raise RuntimeError("gate runner changed during predev gate")

    hardware = hardware_fingerprint()
    runtime = _runtime_provenance(
        runner_sha256=runner_identity[0],
        device=resolved_device,
    )
    training_rows = [
        {
            "video_id": sample.video_id,
            "valid_frames": sample.valid_frames,
            "period_frames": sample.period_frames,
            "confidence": sample.confidence,
            "period_stream": list(sample.period_stream),
            "confidence_stream": list(sample.confidence_stream),
            "period_stream_std": sample.period_stream_std,
        }
        for sample in encoded
    ]
    return {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "status": "passed" if decision["overall_pass"] else "failed",
        "classification": _CLASSIFICATION,
        "disclosed_by_pams_authors": False,
        "eligible_for_paper_table": False,
        "protocol": config.protocol,
        "seed": config.seed,
        "label_firewall": {
            "accepted_scientific_inputs": [
                "v14_experiment_config",
                "bound_epoch11_encoder_checkpoint",
                "checkpoint_bound_train337_pose_cache",
            ],
            "external_synthetic_truth_input_supported": False,
            "dataset_manifest_argument_supported": False,
            "action_class_argument_supported": False,
            "repetition_target_argument_supported": False,
            "development_identity_or_pose_argument_supported": False,
            "sealed_test_identity_media_pose_or_target_argument_supported": (False),
            "external_label_fields_accessed": [],
        },
        "inputs": {
            "checkpoint_sha256": checkpoint_identity[0],
            "checkpoint_bytes": checkpoint_identity[1],
            "checkpoint_stage": stage,
            "checkpoint_metadata": checkpoint_metadata,
            "checkpoint_provenance": provenance.to_dict(),
            "config_sha256": config_identity[0],
            "config_bytes": config_identity[1],
            "config_fingerprint": config.fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
            "train337_video_total": len(selected_ids),
            "train337_video_ids_sha256": _identifier_commitment(selected_ids),
            "train337_pose_cache_set_sha256": full_snapshot.fingerprint,
            "runner_sha256": runner_identity[0],
            "runner_bytes": runner_identity[1],
            "read_only_post_run_identity_verified": True,
        },
        "estimator": {
            "name": "embedding_velocity_vector_acf",
            "input": "L2-normalized encoder embeddings",
            "gradient": "inference_mode_stop_gradient",
            "period_bounds_frames": [
                config.period.minimum,
                config.period.maximum,
            ],
            "period_stream_window_frames": _STREAM_WINDOW_FRAMES,
            "period_stream_stride_frames": _STREAM_STRIDE_FRAMES,
            "period_stream_interpretation": {
                "classification": "inferred anti-collapse diagnostic",
                "source": (
                    "five overlapping encoder/vector-ACF window estimates; "
                    "not the paper's untrained Period Head and not SSHead output"
                ),
                "positive_confidence_windows_only": True,
                "fewer_than_two_positive_windows_std": 0.0,
                "scientific_caveat": (
                    "A genuinely stable per-video period can have near-zero "
                    "windowed standard deviation and therefore fail this "
                    "anti-collapse gate. Passing detects variation, not temporal "
                    "calibration or counting accuracy."
                ),
            },
        },
        "synthetic_primitives": {
            "count_2_to_40": count_synthetic,
            "static_pose": static_synthetic,
            "dominant_harmonic": harmonic_synthetic,
        },
        "train337_target_free": {
            "distribution": distribution,
            "time_scale_consistency": time_scale,
            "rows": training_rows,
        },
        "gate": decision,
        "hardware": hardware,
        "hardware_sha256": sha256_json(hardware),
        "runtime": runtime,
        "runtime_sha256": sha256_json(runtime),
        "read_only_verification": {
            "checkpoint_sha256_unchanged": True,
            "config_sha256_unchanged": True,
            "runner_sha256_unchanged": True,
            "train337_pose_cache_set_sha256_unchanged": True,
            "model_or_optimizer_state_updated": False,
            "pose_cache_write_operations": 0,
        },
    }


def _encoded_json(payload: Mapping[str, Any]) -> bytes:
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def _write_new_regular_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise RuntimeError("new gate artifact is not a regular file")


def _receipt_path(output: Path) -> Path:
    return Path(str(output) + ".receipt.json")


def _write_artifact_and_receipt(
    output: Path,
    payload: Mapping[str, Any],
) -> tuple[Path, str]:
    receipt_path = _receipt_path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite gate artifact: {output}")
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite gate artifact receipt: {receipt_path}")
    artifact = _encoded_json(payload)
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    _write_new_regular_file(output, artifact)
    receipt = {
        "schema_version": 1,
        "artifact_type": _RECEIPT_TYPE,
        "artifact_locator": output.name,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": len(artifact),
        "artifact_status": payload["status"],
        "checkpoint_sha256": payload["inputs"]["checkpoint_sha256"],
        "config_sha256": payload["inputs"]["config_sha256"],
        "train337_pose_cache_set_sha256": payload["inputs"]["train337_pose_cache_set_sha256"],
        "runner_sha256": payload["inputs"]["runner_sha256"],
        "hardware_sha256": payload["hardware_sha256"],
        "runtime_sha256": payload["runtime_sha256"],
    }
    _write_new_regular_file(receipt_path, _encoded_json(receipt))
    return receipt_path, artifact_sha256


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pose-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    receipt = _receipt_path(arguments.output)
    if arguments.output.exists() or receipt.exists():
        raise FileExistsError("predev output and receipt destinations must both be new")
    payload = run_predev_gate(
        arguments.checkpoint,
        arguments.config,
        arguments.pose_cache_dir,
        device=arguments.device,
        batch_size=arguments.batch_size,
    )
    receipt, artifact_sha256 = _write_artifact_and_receipt(
        arguments.output,
        payload,
    )
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "output": str(arguments.output),
                "overall_pass": payload["gate"]["overall_pass"],
                "receipt": str(receipt),
            },
            sort_keys=True,
        )
    )
    return 0 if payload["gate"]["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
