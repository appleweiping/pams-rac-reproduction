"""Run the frozen train337-only terminal gate for PAMS-v14.

This independently inferred diagnostic accepts only the v14 configuration,
terminal encoder/SSHead checkpoints and progress logs, and the pose caches
named by checkpoint provenance.  It has no interface for development/test
data, action classes, repetition counts, or external synthetic truth.

The gate detects several failure modes without claiming counting accuracy:
encoder period collapse, a flat or shared-template SSHead stream, failure of
time-scale equivariance, a head that never received a gradient step, and a
head that is byte-identical to its deterministic random initialization.

Training and inference algorithms run from the checkpoint-bound, exact source
export.  This later audit runner and its frozen epoch-11 helper may be mounted
read-only from a separate ``/gate-code`` directory.  The two source identities
are deliberately distinct: ``PAMS_CONTAINER_SOURCE_REVISION`` binds the
checkpoint algorithm source, while required
``PAMS_GATE_CODE_SOURCE_REVISION`` maps the separately hashed gate files to
their public Git commit.  Invoke either as a module from the repository root
or as a script with both gate files on ``PYTHONPATH``:

    python -m scripts.server.run_pams_v14_final_train_gate ...

PyTorch checkpoints are pickle containers.  Use only checkpoints produced by
this repository in the trusted experiment environment.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pams.config import PAMSConfig, load_config
from pams.data import PoseCacheSetSnapshot
from pams.diagnostics import (
    _device,
    _identifier_commitment,
    _load_training_poses,
    _peek_checkpoint,
    _stable_file_sha256,
    _summary,
)
from pams.model import PAMSModel
from pams.period import estimate_period_batch
from pams.reproducibility import (
    clean_git_revision,
    hardware_fingerprint,
    sha256_json,
)
from pams.training import (
    CheckpointProvenance,
    collate_pose_sequences,
    load_model_checkpoint,
    validate_sshead_encoder_binding,
    validate_terminal_checkpoint,
)
from pams.types import PoseSequence

try:
    from scripts.server import run_pams_v14_predev_gate as _epoch11_gate
except ModuleNotFoundError:
    # Formal continuation experiments mount both audited gate files in a
    # separate read-only directory while /workspace remains the exact source
    # export that produced the checkpoints.
    import run_pams_v14_predev_gate as _epoch11_gate  # type: ignore[no-redef]

_TIME_SCALE_FACTORS = _epoch11_gate._TIME_SCALE_FACTORS
_criterion = _epoch11_gate._criterion
_encode_training_sequences = _epoch11_gate._encode_training_sequences
_encoded_json = _epoch11_gate._encoded_json
_resample_for_time_scale = _epoch11_gate._resample_for_time_scale
_runtime_provenance = _epoch11_gate._runtime_provenance
_time_scale_consistency = _epoch11_gate._time_scale_consistency
_training_distribution = _epoch11_gate._training_distribution
_validate_v14_config = _epoch11_gate._validate_v14_config
_write_new_regular_file = _epoch11_gate._write_new_regular_file

_ARTIFACT_TYPE = "pams_v14_terminal_train337_target_free_gate"
_RECEIPT_TYPE = "pams_v14_terminal_train337_target_free_gate_receipt"
_CLASSIFICATION = "inferred target-free terminal training diagnostic"
_EXPECTED_TRAINING_VIDEOS = 337
_MINIMUM_CORRELATION_OVERLAP = 8
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_THRESHOLDS: dict[str, float] = {
    "encoder_boundary_share_maximum_exclusive": 0.25,
    "encoder_mode_share_maximum_exclusive": 0.25,
    "encoder_time_scale_eligible_fraction_minimum": 0.50,
    "encoder_time_scale_median_relative_error_maximum": 0.15,
    "sshead_stream_std_median_minimum": 0.05,
    "sshead_boundary_share_maximum_exclusive": 0.25,
    "sshead_mode_share_maximum_exclusive": 0.25,
    "sshead_correlation_pair_coverage_minimum": 0.90,
    "sshead_median_absolute_correlation_maximum_exclusive": 0.90,
    "sshead_time_scale_eligible_fraction_minimum": 0.50,
    "sshead_time_scale_median_relative_error_maximum": 0.15,
    "sshead_zero_grad_steps_maximum": 0.0,
    "sshead_head_parameter_delta_l2_minimum_exclusive": 0.0,
}


@dataclass(frozen=True, slots=True)
class _HeadSample:
    """One target-free scalar-head stream and its FFT period."""

    video_id: str
    valid_frames: int
    period_frames: float
    confidence: float
    period_stream: tuple[float, ...]
    valid_mask: tuple[bool, ...]
    period_stream_std: float


def _finite_float(value: Tensor | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("SSHead diagnostic produced a non-finite value")
    return result


def _shared_provenance_fields(provenance: CheckpointProvenance) -> dict[str, Any]:
    return {
        "protocol": provenance.protocol,
        "dataset_fingerprint": provenance.dataset_fingerprint,
        "training_video_ids": provenance.training_video_ids,
        "pose_fingerprint": provenance.pose_fingerprint,
        "pose_cache_set_sha256": provenance.pose_cache_set_sha256,
        "source_git_sha": provenance.source_git_sha,
        "container_image_id": provenance.container_image_id,
        "container_environment_sha256": provenance.container_environment_sha256,
    }


def _validate_stage_bindings(
    encoder: CheckpointProvenance,
    sshead: CheckpointProvenance,
    *,
    encoder_checkpoint_sha256: str,
) -> None:
    """Require both terminal stages to name one exact formal training context."""

    if len(encoder.training_video_ids) != _EXPECTED_TRAINING_VIDEOS:
        raise ValueError("v14 terminal gate requires exactly 337 training videos")
    if _shared_provenance_fields(encoder) != _shared_provenance_fields(sshead):
        raise ValueError("encoder and SSHead checkpoint provenance differ")
    if sshead.upstream_encoder_checkpoint_sha256 != encoder_checkpoint_sha256:
        raise ValueError("SSHead provenance does not bind the supplied encoder bytes")
    if encoder.upstream_encoder_checkpoint_sha256 is not None:
        raise ValueError("encoder provenance unexpectedly names an upstream checkpoint")
    if encoder.container_image_id is None or encoder.container_environment_sha256 is None:
        raise ValueError("terminal gate requires formal container-bound checkpoints")


def _validate_runtime_binding(
    provenance: CheckpointProvenance,
    *,
    source_git_sha: str,
) -> dict[str, str]:
    """Bind gate execution to the same immutable source and environment."""

    values = {
        "image_id": os.environ.get("PAMS_CONTAINER_IMAGE_ID", ""),
        "environment_sha256": os.environ.get(
            "PAMS_CONTAINER_ENVIRONMENT_SHA256",
            "",
        ),
        "source_revision": os.environ.get("PAMS_CONTAINER_SOURCE_REVISION", ""),
    }
    if any(not value for value in values.values()):
        raise RuntimeError("formal terminal gate container provenance is incomplete")
    expected = {
        "image_id": provenance.container_image_id,
        "environment_sha256": provenance.container_environment_sha256,
        "source_revision": provenance.source_git_sha,
    }
    if values != expected:
        raise RuntimeError("terminal gate runtime differs from checkpoint provenance")
    if source_git_sha != provenance.source_git_sha:
        raise RuntimeError("clean source revision differs from checkpoint provenance")
    return values


def _gate_code_source_revision() -> str:
    """Return the public revision of the separately mounted gate code."""

    revision = os.environ.get("PAMS_GATE_CODE_SOURCE_REVISION", "").strip()
    if not _GIT_SHA_PATTERN.fullmatch(revision):
        raise RuntimeError(
            "PAMS_GATE_CODE_SOURCE_REVISION must be a 40-character lowercase Git SHA"
        )
    return revision


def _encode_head_sequences(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> tuple[_HeadSample, ...]:
    samples: list[_HeadSample] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            _, streams = model.forward_with_head_source(
                batch.poses,
                batch.valid_mask,
                head_input_source=config.sshead.input_source,
            )
            periods, confidences = estimate_period_batch(
                streams,
                minimum=config.period.minimum,
                maximum=config.period.maximum,
                valid_mask=batch.valid_mask,
            )
            for index, video_id in enumerate(batch.video_ids):
                stream = streams[index].detach().float().cpu()
                valid = batch.valid_mask[index].detach().cpu()
                selected = stream[valid]
                standard_deviation = (
                    float(selected.std(unbiased=False)) if selected.numel() >= 2 else 0.0
                )
                samples.append(
                    _HeadSample(
                        video_id=video_id,
                        valid_frames=int(valid.sum()),
                        period_frames=_finite_float(periods[index]),
                        confidence=_finite_float(confidences[index]),
                        period_stream=tuple(_finite_float(value) for value in stream),
                        valid_mask=tuple(bool(value) for value in valid),
                        period_stream_std=standard_deviation,
                    )
                )
    return tuple(samples)


def _centered_cross_video_correlation(
    samples: Sequence[_HeadSample],
) -> dict[str, Any]:
    """Measure shared scalar templates over pairwise valid-frame intersections."""

    if len(samples) < 2:
        raise ValueError("cross-video correlation requires at least two streams")
    total_pairs = len(samples) * (len(samples) - 1) // 2
    signed_correlations: list[float] = []
    absolute_correlations: list[float] = []
    overlap_frames: list[float] = []
    for left_index, left in enumerate(samples[:-1]):
        left_stream = np.asarray(left.period_stream, dtype=np.float64)
        left_valid = np.asarray(left.valid_mask, dtype=np.bool_)
        for right in samples[left_index + 1 :]:
            right_stream = np.asarray(right.period_stream, dtype=np.float64)
            right_valid = np.asarray(right.valid_mask, dtype=np.bool_)
            if left_stream.shape != right_stream.shape:
                raise ValueError("SSHead streams do not share one temporal shape")
            common = left_valid & right_valid
            overlap = int(np.count_nonzero(common))
            if overlap < _MINIMUM_CORRELATION_OVERLAP:
                continue
            left_values = left_stream[common]
            right_values = right_stream[common]
            left_values = left_values - left_values.mean()
            right_values = right_values - right_values.mean()
            denominator = float(np.linalg.norm(left_values) * np.linalg.norm(right_values))
            if denominator <= 1e-12:
                continue
            correlation = float(np.dot(left_values, right_values) / denominator)
            correlation = min(max(correlation, -1.0), 1.0)
            signed_correlations.append(correlation)
            absolute_correlations.append(abs(correlation))
            overlap_frames.append(float(overlap))
    eligible_pairs = len(signed_correlations)
    return {
        "algorithm": (
            "For every unordered train337 video pair, intersect valid masks, "
            "require at least eight shared frames, center each scalar stream "
            "over that intersection, and compute Pearson correlation. Pairs "
            "with a zero centered norm are ineligible. The gate uses absolute "
            "correlation so a sign-flipped shared template cannot pass."
        ),
        "minimum_shared_valid_frames": _MINIMUM_CORRELATION_OVERLAP,
        "candidate_pair_total": total_pairs,
        "eligible_pair_total": eligible_pairs,
        "eligible_pair_fraction": (eligible_pairs / total_pairs if total_pairs else 0.0),
        "shared_valid_frames": _summary(overlap_frames),
        "signed_correlation": _summary(signed_correlations),
        "absolute_correlation": _summary(absolute_correlations),
    }


def _head_time_scale_consistency(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    baseline: Sequence[_HeadSample],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Test scalar-head period equivariance using the frozen epoch-11 transforms."""

    if [sample.video_id for sample in baseline] != [sequence.video_id for sequence in sequences]:
        raise ValueError("SSHead time-scale baseline order differs from train337")
    errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for factor in _TIME_SCALE_FACTORS:
        scaled_sequences = tuple(
            _resample_for_time_scale(sequence, factor) for sequence in sequences
        )
        scaled_samples = _encode_head_sequences(
            model,
            scaled_sequences,
            config=config,
            device=device,
            batch_size=batch_size,
        )
        for source, scaled in zip(baseline, scaled_samples, strict=True):
            expected = source.period_frames * factor
            eligible = (
                source.confidence > 0.0
                and scaled.confidence > 0.0
                and config.period.minimum <= expected <= config.period.maximum
            )
            relative_error = abs(scaled.period_frames - expected) / expected if eligible else None
            if relative_error is not None:
                if not math.isfinite(relative_error):
                    raise RuntimeError("SSHead time-scale error is non-finite")
                errors.append(relative_error)
            rows.append(
                {
                    "video_id": source.video_id,
                    "time_scale_factor": factor,
                    "baseline_period_frames": source.period_frames,
                    "baseline_confidence": source.confidence,
                    "expected_scaled_period_frames": expected,
                    "scaled_period_frames": scaled.period_frames,
                    "scaled_confidence": scaled.confidence,
                    "eligible": eligible,
                    "relative_error": relative_error,
                }
            )
    return {
        "algorithm": (
            "Uniformly resample each train337 pose sequence to round(T*factor) "
            "frames for factors 0.5 and 0.75, run the frozen encoder and "
            "trained scalar head, estimate the head-stream FFT period, and "
            "compare P_scaled with factor*P. Both confidences must be positive "
            "and the expected period must remain inside 4--128."
        ),
        "factors": list(_TIME_SCALE_FACTORS),
        "candidate_comparison_total": len(rows),
        "eligible_comparison_total": len(errors),
        "eligible_comparison_fraction": (len(errors) / len(rows) if rows else 0.0),
        "relative_error": _summary(errors),
        "rows": rows,
    }


def _head_parameter_change(
    encoder_model: PAMSModel,
    sshead_model: PAMSModel,
) -> dict[str, Any]:
    initial = encoder_model.period_head.state_dict()
    final = sshead_model.period_head.state_dict()
    if set(initial) != set(final):
        raise ValueError("initial and trained period-head tensor keys differ")
    initial_squared = 0.0
    final_squared = 0.0
    delta_squared = 0.0
    changed_tensors = 0
    for name in sorted(initial):
        before = initial[name].detach().to(device="cpu", dtype=torch.float64)
        after = final[name].detach().to(device="cpu", dtype=torch.float64)
        if before.shape != after.shape:
            raise ValueError(f"period-head tensor shape changed: {name}")
        delta = after - before
        initial_squared += float(before.square().sum())
        final_squared += float(after.square().sum())
        delta_squared += float(delta.square().sum())
        changed_tensors += int(not torch.equal(before, after))
    return {
        "initial_l2": math.sqrt(initial_squared),
        "final_l2": math.sqrt(final_squared),
        "delta_l2": math.sqrt(delta_squared),
        "changed_tensor_total": changed_tensors,
        "tensor_total": len(initial),
        "interpretation": (
            "The encoder checkpoint contains the deterministic untouched "
            "random Period Head; the terminal validator proves that invariant "
            "before this difference is measured."
        ),
    }


def _sshead_history_audit(checkpoint: Path) -> dict[str, Any]:
    payload = torch.load(
        checkpoint,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if not isinstance(payload, Mapping) or payload.get("stage") != "sshead":
        raise ValueError("SSHead history audit requires an SSHead checkpoint")
    history = payload.get("history")
    if not isinstance(history, list):
        raise ValueError("SSHead checkpoint history must be a list")
    zero_grad_steps = [row.get("zero_grad_steps") for row in history if isinstance(row, Mapping)]
    if len(zero_grad_steps) != len(history) or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in zero_grad_steps
    ):
        raise ValueError("SSHead history zero_grad_steps is malformed")
    return {
        "completed_epochs": len(history),
        "zero_grad_steps_by_epoch": zero_grad_steps,
        "zero_grad_steps_total": sum(zero_grad_steps),
        "final_epoch": dict(history[-1]) if history else None,
    }


def _eligible_fraction(payload: Mapping[str, Any]) -> float:
    candidates = int(payload["candidate_comparison_total"])
    eligible = int(payload["eligible_comparison_total"])
    return eligible / candidates if candidates else 0.0


def _metric_median(payload: Mapping[str, Any]) -> float | None:
    value = payload.get("median")
    return float(value) if isinstance(value, int | float) else None


def _strictly_positive_criterion(value: float) -> dict[str, float | str | bool]:
    return {
        "value": value,
        "operator": ">",
        "threshold": 0.0,
        "pass": math.isfinite(value) and value > 0.0,
    }


def _gate_decision(
    *,
    encoder_distribution: Mapping[str, Any],
    encoder_time_scale: Mapping[str, Any],
    sshead_distribution: Mapping[str, Any],
    sshead_correlation: Mapping[str, Any],
    sshead_time_scale: Mapping[str, Any],
    sshead_history: Mapping[str, Any],
    head_parameter_change: Mapping[str, Any],
) -> dict[str, Any]:
    criteria = {
        "encoder_boundary_share": _criterion(
            float(encoder_distribution["boundary_share"]),
            operator="<",
            threshold=_THRESHOLDS["encoder_boundary_share_maximum_exclusive"],
        ),
        "encoder_mode_share": _criterion(
            float(encoder_distribution["mode_share"]),
            operator="<",
            threshold=_THRESHOLDS["encoder_mode_share_maximum_exclusive"],
        ),
        "encoder_time_scale_eligible_fraction": _criterion(
            _eligible_fraction(encoder_time_scale),
            operator=">=",
            threshold=_THRESHOLDS["encoder_time_scale_eligible_fraction_minimum"],
        ),
        "encoder_time_scale_median_relative_error": _criterion(
            _metric_median(encoder_time_scale["relative_error"]),
            operator="<=",
            threshold=_THRESHOLDS["encoder_time_scale_median_relative_error_maximum"],
        ),
        "sshead_stream_std_median": _criterion(
            _metric_median(sshead_distribution["period_stream_std"]),
            operator=">=",
            threshold=_THRESHOLDS["sshead_stream_std_median_minimum"],
        ),
        "sshead_boundary_share": _criterion(
            float(sshead_distribution["boundary_share"]),
            operator="<",
            threshold=_THRESHOLDS["sshead_boundary_share_maximum_exclusive"],
        ),
        "sshead_mode_share": _criterion(
            float(sshead_distribution["mode_share"]),
            operator="<",
            threshold=_THRESHOLDS["sshead_mode_share_maximum_exclusive"],
        ),
        "sshead_centered_cross_video_pair_coverage": _criterion(
            float(sshead_correlation["eligible_pair_fraction"]),
            operator=">=",
            threshold=_THRESHOLDS["sshead_correlation_pair_coverage_minimum"],
        ),
        "sshead_centered_cross_video_median_absolute_correlation": _criterion(
            _metric_median(sshead_correlation["absolute_correlation"]),
            operator="<",
            threshold=_THRESHOLDS["sshead_median_absolute_correlation_maximum_exclusive"],
        ),
        "sshead_time_scale_eligible_fraction": _criterion(
            _eligible_fraction(sshead_time_scale),
            operator=">=",
            threshold=_THRESHOLDS["sshead_time_scale_eligible_fraction_minimum"],
        ),
        "sshead_time_scale_median_relative_error": _criterion(
            _metric_median(sshead_time_scale["relative_error"]),
            operator="<=",
            threshold=_THRESHOLDS["sshead_time_scale_median_relative_error_maximum"],
        ),
        "sshead_zero_grad_steps_total": _criterion(
            float(sshead_history["zero_grad_steps_total"]),
            operator="<=",
            threshold=_THRESHOLDS["sshead_zero_grad_steps_maximum"],
        ),
        "sshead_head_parameter_delta_l2": _strictly_positive_criterion(
            float(head_parameter_change["delta_l2"])
        ),
    }
    passed = all(bool(criterion["pass"]) for criterion in criteria.values())
    return {
        "thresholds_frozen_before_terminal_checkpoint_evaluation": True,
        "criteria": criteria,
        "overall_pass": passed,
        "dev84_prediction_authorized": passed,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }


def _input_identities(paths: Mapping[str, Path]) -> dict[str, tuple[str, int]]:
    return {name: _stable_file_sha256(path) for name, path in paths.items()}


def _require_unchanged(
    paths: Mapping[str, Path],
    identities: Mapping[str, tuple[str, int]],
) -> None:
    for name, path in paths.items():
        if _stable_file_sha256(path) != identities[name]:
            raise RuntimeError(f"terminal gate input changed during inference: {name}")


def run_final_gate(
    encoder_checkpoint_path: str | Path,
    encoder_progress_path: str | Path,
    sshead_checkpoint_path: str | Path,
    sshead_progress_path: str | Path,
    config_path: str | Path,
    pose_cache_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate terminal encoder150 and SSHead30 using train337 only."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or batch_size > 32
    ):
        raise ValueError("batch_size must be an integer in [1, 32]")
    paths = {
        "encoder_checkpoint": Path(encoder_checkpoint_path),
        "encoder_progress": Path(encoder_progress_path),
        "sshead_checkpoint": Path(sshead_checkpoint_path),
        "sshead_progress": Path(sshead_progress_path),
        "config": Path(config_path),
        "final_gate_runner": Path(__file__),
        "epoch11_gate_dependency": Path(_epoch11_gate.__file__),
    }
    identities = _input_identities(paths)
    gate_code_source_git_sha = _gate_code_source_revision()
    config = load_config(paths["config"])
    _validate_v14_config(config)
    encoder_stage, encoder_provenance = _peek_checkpoint(
        paths["encoder_checkpoint"],
        config,
    )
    sshead_stage, sshead_provenance = _peek_checkpoint(
        paths["sshead_checkpoint"],
        config,
    )
    if encoder_stage != "encoder" or sshead_stage != "sshead":
        raise ValueError("terminal gate requires encoder then SSHead checkpoints")
    _validate_stage_bindings(
        encoder_provenance,
        sshead_provenance,
        encoder_checkpoint_sha256=identities["encoder_checkpoint"][0],
    )
    validate_terminal_checkpoint(
        paths["encoder_checkpoint"],
        config,
        expected_stage="encoder",
        expected_provenance=encoder_provenance,
        progress_path=paths["encoder_progress"],
    )
    validate_terminal_checkpoint(
        paths["sshead_checkpoint"],
        config,
        expected_stage="sshead",
        expected_provenance=sshead_provenance,
        progress_path=paths["sshead_progress"],
    )
    validate_sshead_encoder_binding(
        paths["sshead_checkpoint"],
        paths["encoder_checkpoint"],
    )
    algorithm_source_git_sha = clean_git_revision(Path.cwd())
    runtime_container = _validate_runtime_binding(
        encoder_provenance,
        source_git_sha=algorithm_source_git_sha,
    )
    resolved_device = _device(device)

    sequences, selected_receipts, full_receipts, selected_ids = _load_training_poses(
        pose_cache_dir=Path(pose_cache_dir),
        provenance=encoder_provenance,
        config=config,
        sample_size=0,
        seed=config.seed,
    )
    if len(sequences) != _EXPECTED_TRAINING_VIDEOS:
        raise RuntimeError("checkpoint-bound pose loader did not return train337")
    if any(sequence.num_frames != config.data.frames for sequence in sequences):
        raise ValueError("every v14 training pose must contain exactly 256 frames")
    selected_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=selected_receipts,
    )
    full_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=full_receipts,
    )
    if selected_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("full train337 selection and pose snapshots differ")
    if full_snapshot.fingerprint != encoder_provenance.pose_cache_set_sha256:
        raise ValueError("train337 pose-cache set differs from checkpoint provenance")
    if full_snapshot.fingerprint != sshead_provenance.pose_cache_set_sha256:
        raise ValueError("train337 pose-cache set differs between checkpoint stages")

    encoder_model = load_model_checkpoint(
        paths["encoder_checkpoint"],
        config,
        device=resolved_device,
        expected_stage="encoder",
        expected_provenance=encoder_provenance,
    ).eval()
    encoder_samples = _encode_training_sequences(
        encoder_model,
        sequences,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    encoder_distribution = _training_distribution(
        encoder_samples,
        minimum=config.period.minimum,
        maximum=config.period.maximum,
    )
    encoder_time_scale = _time_scale_consistency(
        encoder_model,
        sequences,
        encoder_samples,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )

    sshead_model = load_model_checkpoint(
        paths["sshead_checkpoint"],
        config,
        device=resolved_device,
        expected_stage="sshead",
        expected_provenance=sshead_provenance,
    ).eval()
    head_parameter_change = _head_parameter_change(
        encoder_model,
        sshead_model,
    )
    sshead_samples = _encode_head_sequences(
        sshead_model,
        sequences,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    sshead_distribution = _training_distribution(
        sshead_samples,
        minimum=config.period.minimum,
        maximum=config.period.maximum,
    )
    sshead_correlation = _centered_cross_video_correlation(sshead_samples)
    sshead_time_scale = _head_time_scale_consistency(
        sshead_model,
        sequences,
        sshead_samples,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    sshead_history = _sshead_history_audit(paths["sshead_checkpoint"])
    decision = _gate_decision(
        encoder_distribution=encoder_distribution,
        encoder_time_scale=encoder_time_scale,
        sshead_distribution=sshead_distribution,
        sshead_correlation=sshead_correlation,
        sshead_time_scale=sshead_time_scale,
        sshead_history=sshead_history,
        head_parameter_change=head_parameter_change,
    )

    _, _, final_receipts, final_ids = _load_training_poses(
        pose_cache_dir=Path(pose_cache_dir),
        provenance=encoder_provenance,
        config=config,
        sample_size=2,
        seed=config.seed,
    )
    final_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=final_receipts,
    )
    if final_ids != selected_ids[:2]:
        raise RuntimeError("train337 pose selection changed during terminal gate")
    if final_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("train337 pose cache changed during terminal gate")
    _require_unchanged(paths, identities)
    if clean_git_revision(Path.cwd()) != algorithm_source_git_sha:
        raise RuntimeError("source revision changed during terminal gate")

    hardware = hardware_fingerprint()
    runtime = _runtime_provenance(
        runner_sha256=identities["final_gate_runner"][0],
        device=resolved_device,
    )
    if runtime["container"] != {
        "PAMS_CONTAINER_IMAGE_ID": runtime_container["image_id"],
        "PAMS_CONTAINER_ENVIRONMENT_SHA256": (runtime_container["environment_sha256"]),
        "PAMS_CONTAINER_SOURCE_REVISION": runtime_container["source_revision"],
    }:
        raise RuntimeError("runtime provenance changed while it was recorded")
    runtime["algorithm_source_git_sha"] = algorithm_source_git_sha
    runtime["gate_code_source_git_sha"] = gate_code_source_git_sha
    runtime["source_identity_model"] = (
        "checkpoint algorithm source and external gate-code source are independently bound"
    )
    code_files_sha256 = {
        name: identities[name][0] for name in ("final_gate_runner", "epoch11_gate_dependency")
    }
    payload = {
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
                "terminal_encoder_checkpoint_and_progress",
                "terminal_sshead_checkpoint_and_progress",
                "checkpoint_bound_train337_pose_cache",
            ],
            "dataset_manifest_argument_supported": False,
            "development_identity_media_pose_or_target_argument_supported": False,
            "sealed_test_identity_media_pose_or_target_argument_supported": False,
            "action_class_argument_supported": False,
            "repetition_count_argument_supported": False,
            "external_label_fields_accessed": [],
        },
        "inputs": {
            **{f"{name}_sha256": identity[0] for name, identity in identities.items()},
            **{f"{name}_bytes": identity[1] for name, identity in identities.items()},
            "config_fingerprint": config.fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
            "encoder_provenance": encoder_provenance.to_dict(),
            "sshead_provenance": sshead_provenance.to_dict(),
            "train337_video_total": len(selected_ids),
            "train337_video_ids_sha256": _identifier_commitment(selected_ids),
            "train337_pose_cache_set_sha256": full_snapshot.fingerprint,
            "checkpoint_algorithm_source_git_sha": algorithm_source_git_sha,
            "gate_code_source_git_sha": gate_code_source_git_sha,
            "code_files_sha256": code_files_sha256,
            "code_files_sha256_commitment": sha256_json(code_files_sha256),
            "read_only_post_run_identity_verified": True,
        },
        "encoder150": {
            "estimator": "embedding_velocity_vector_acf",
            "distribution": encoder_distribution,
            "time_scale_consistency": encoder_time_scale,
            "window_period_std_interpretation": {
                "used_as_gate_criterion": False,
                "reason": (
                    "A genuinely stable per-video period can have near-zero "
                    "windowed period standard deviation; requiring instability "
                    "would have the wrong scientific direction."
                ),
            },
        },
        "sshead30": {
            "classification": (
                "independently inferred SSHead completion; loss and gradient "
                "path were not disclosed by the PAMS authors"
            ),
            "distribution": sshead_distribution,
            "centered_cross_video_correlation": sshead_correlation,
            "time_scale_consistency": sshead_time_scale,
            "history": sshead_history,
            "head_parameter_change": head_parameter_change,
            "stream_std_interpretation": {
                "threshold": _THRESHOLDS["sshead_stream_std_median_minimum"],
                "scale_dependent": True,
                "reason": (
                    "The 0.05 threshold is only a weak non-flat guard. Scalar "
                    "head amplitude is not calibrated and can be rescaled, so "
                    "this metric does not establish period or count accuracy."
                ),
            },
        },
        "gate": decision,
        "scientific_caveats": [
            (
                "All criteria are independently inferred anti-collapse or "
                "self-consistency diagnostics, not author-disclosed validation."
            ),
            (
                "Boundary and mode diversity can be affected by the discrete "
                "FFT estimator and the unlabeled training distribution."
            ),
            (
                "Cross-video correlation can penalize genuinely synchronized "
                "normalized actions; it primarily detects a shared positional template."
            ),
            (
                "Time-scale consistency is equivariance, not ground-truth "
                "period or repetition-count accuracy."
            ),
            (
                "Passing can authorize one isolated dev84 prediction process, "
                "but this process cannot score dev84 and never authorizes test105."
            ),
        ],
        "hardware": hardware,
        "hardware_sha256": sha256_json(hardware),
        "runtime": runtime,
        "runtime_sha256": sha256_json(runtime),
        "read_only_verification": {
            "all_file_inputs_unchanged": True,
            "train337_pose_cache_set_sha256_unchanged": True,
            "checkpoint_algorithm_source_git_sha_unchanged": True,
            "model_or_optimizer_state_updated": False,
            "pose_cache_write_operations": 0,
        },
    }
    del encoder_model, sshead_model
    gc.collect()
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


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
        "encoder_checkpoint_sha256": payload["inputs"]["encoder_checkpoint_sha256"],
        "encoder_progress_sha256": payload["inputs"]["encoder_progress_sha256"],
        "sshead_checkpoint_sha256": payload["inputs"]["sshead_checkpoint_sha256"],
        "sshead_progress_sha256": payload["inputs"]["sshead_progress_sha256"],
        "config_sha256": payload["inputs"]["config_sha256"],
        "train337_pose_cache_set_sha256": payload["inputs"]["train337_pose_cache_set_sha256"],
        "checkpoint_algorithm_source_git_sha": payload["inputs"][
            "checkpoint_algorithm_source_git_sha"
        ],
        "gate_code_source_git_sha": payload["inputs"]["gate_code_source_git_sha"],
        "code_files_sha256_commitment": payload["inputs"]["code_files_sha256_commitment"],
        "hardware_sha256": payload["hardware_sha256"],
        "runtime_sha256": payload["runtime_sha256"],
    }
    _write_new_regular_file(receipt_path, _encoded_json(receipt))
    return receipt_path, artifact_sha256


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-progress", type=Path, required=True)
    parser.add_argument("--sshead-checkpoint", type=Path, required=True)
    parser.add_argument("--sshead-progress", type=Path, required=True)
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
        raise FileExistsError("terminal gate output and receipt destinations must both be new")
    payload = run_final_gate(
        arguments.encoder_checkpoint,
        arguments.encoder_progress,
        arguments.sshead_checkpoint,
        arguments.sshead_progress,
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
