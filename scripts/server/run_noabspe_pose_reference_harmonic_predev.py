"""Target-free predev gate for the inferred NoAbsPE pose-reference readout.

The frozen encoder's legacy embedding-velocity estimator can select an
overtone of the physical pose cycle.  This independently inferred diagnostic
uses the raw pose-energy period only to choose an integer harmonic multiplier
for that legacy embedding period.  It never consumes actions, repetition
counts, development inputs, or sealed-test inputs.

Accepted inputs are deliberately limited to one encoder checkpoint, its exact
configuration, the checkpoint-bound training pose-cache directory, and a new
output path.  Passing this diagnostic can authorize a separate dev84
prediction run; it can never authorize scoring or sealed-test access.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from pams.config import PAMSConfig, load_config
from pams.data import PoseCacheSetSnapshot, per_frame_minmax
from pams.diagnostics import (
    _device,
    _frame_index_probe,
    _identifier_commitment,
    _load_training_poses,
    _peek_checkpoint,
    _period_histogram,
    _stable_file_sha256,
    _summary,
)
from pams.model import PAMSModel
from pams.period import estimate_period_from_embeddings, estimate_period_from_pose
from pams.training import collate_pose_sequences, load_model_checkpoint
from pams.types import PoseSequence

_ARTIFACT_TYPE = "pams_noabspe_pose_reference_harmonic_predev_gate"
_CLASSIFICATION = "inferred target-free predev diagnostic"
_SYNTHETIC_PERIODS = (4, 8, 16, 32, 64, 128)
_MAXIMUM_HARMONIC = 8
_THRESHOLDS = {
    "frame_index_r2_maximum": 0.10,
    "zero_pose_period_confidence_maximum": 0.10,
    "random_pose_period_confidence_maximum": 0.10,
    "training_period_top_bin_share_maximum": 0.25,
    "canonical_permuted_embedding_median_cosine_minimum": 0.95,
    "synthetic_period_median_relative_error_maximum": 0.10,
}


@dataclass(frozen=True, slots=True)
class _ReadoutSample:
    """One target-free training sample and both period-evidence paths."""

    video_id: str
    embeddings: Tensor
    valid_mask: Tensor
    period: float
    confidence: float
    embedding_period: float
    embedding_confidence: float
    pose_period: float
    pose_confidence: float
    harmonic_factor: int
    candidates: tuple[dict[str, float | int], ...]


def _finite_probability(value: float, *, name: str) -> float:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return value


def _pose_reference_harmonic_resolution(
    embedding_period: float,
    embedding_confidence: float,
    pose_period: float,
    pose_confidence: float,
    *,
    minimum: int = 4,
    maximum: int = 128,
    maximum_harmonic: int = _MAXIMUM_HARMONIC,
) -> dict[str, Any]:
    """Choose ``h * embedding_period`` nearest the pose-energy reference.

    Candidate ranking is lexicographic: absolute log-ratio distance, relative
    distance, then the smaller harmonic factor.  The first term treats equal
    multiplicative under- and over-estimates symmetrically; the second and
    third terms make every residual tie deterministic.
    """

    if minimum < 2 or maximum <= minimum:
        raise ValueError("period bounds must satisfy 2 <= minimum < maximum")
    if isinstance(maximum_harmonic, bool) or maximum_harmonic < 1:
        raise ValueError("maximum_harmonic must be a positive integer")
    for name, value in (
        ("embedding_period", embedding_period),
        ("pose_period", pose_period),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    embedding_confidence = _finite_probability(
        embedding_confidence,
        name="embedding_confidence",
    )
    pose_confidence = _finite_probability(
        pose_confidence,
        name="pose_confidence",
    )

    candidates: list[dict[str, float | int]] = []
    boundary_tolerance = max(1e-12, 1e-6 * maximum)
    for harmonic_factor in range(1, maximum_harmonic + 1):
        raw_candidate_period = harmonic_factor * embedding_period
        if (
            raw_candidate_period < minimum - boundary_tolerance
            or raw_candidate_period > maximum + boundary_tolerance
        ):
            continue
        candidate_period = min(
            max(raw_candidate_period, float(minimum)),
            float(maximum),
        )
        candidates.append(
            {
                "harmonic_factor": harmonic_factor,
                "raw_harmonic_period_frames": raw_candidate_period,
                "period_frames": candidate_period,
                "absolute_log_ratio_distance": abs(
                    math.log(candidate_period / pose_period)
                ),
                "relative_distance_to_pose": (
                    abs(candidate_period - pose_period) / pose_period
                ),
            }
        )
    if not candidates:
        raise ValueError("embedding period has no bounded harmonic candidate")

    selected = min(
        candidates,
        key=lambda item: (
            round(float(item["absolute_log_ratio_distance"]), 12),
            round(float(item["relative_distance_to_pose"]), 12),
            int(item["harmonic_factor"]),
        ),
    )
    confidence = embedding_confidence * pose_confidence
    return {
        "embedding_period_frames": embedding_period,
        "embedding_period_confidence": embedding_confidence,
        "pose_energy_period_frames": pose_period,
        "pose_energy_period_confidence": pose_confidence,
        "selected_harmonic_factor": int(selected["harmonic_factor"]),
        "selected_period_frames": float(selected["period_frames"]),
        "confidence_rule": "embedding_confidence_times_pose_energy_confidence",
        "selected_period_confidence": confidence,
        "candidate_ranking": (
            "minimum absolute log-ratio distance to pose-energy period; "
            "distances rounded to 12 decimals for numerical ties; then minimum "
            "relative distance; then smaller harmonic factor"
        ),
        "bound_tolerance_frames": boundary_tolerance,
        "candidates": candidates,
    }


def _readout_batch(
    model: PAMSModel,
    poses: Tensor,
    valid_mask: Tensor,
    *,
    config: PAMSConfig,
) -> tuple[Tensor, tuple[dict[str, Any], ...]]:
    """Evaluate the frozen encoder and resolve every batch row."""

    with torch.inference_mode():
        embeddings = model.encoder(poses, valid_mask)
        embedding_periods, embedding_confidences = estimate_period_from_embeddings(
            embeddings,
            minimum=config.period.minimum,
            maximum=config.period.maximum,
            valid_mask=valid_mask,
        )
        pose_periods, pose_confidences = estimate_period_from_pose(
            poses,
            minimum=config.period.minimum,
            maximum=config.period.maximum,
            valid_mask=valid_mask,
        )
    rows = tuple(
        _pose_reference_harmonic_resolution(
            float(embedding_period),
            float(embedding_confidence),
            float(pose_period),
            float(pose_confidence),
            minimum=config.period.minimum,
            maximum=config.period.maximum,
        )
        for embedding_period, embedding_confidence, pose_period, pose_confidence in zip(
            embedding_periods,
            embedding_confidences,
            pose_periods,
            pose_confidences,
            strict=True,
        )
    )
    return embeddings, rows


def _encode_training_samples(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> tuple[_ReadoutSample, ...]:
    """Encode sampled checkpoint-bound training poses without labels."""

    encoded: list[_ReadoutSample] = []
    model.eval()
    for start in range(0, len(sequences), batch_size):
        batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
        embeddings, rows = _readout_batch(
            model,
            batch.poses,
            batch.valid_mask,
            config=config,
        )
        for index, (identifier, row) in enumerate(
            zip(batch.video_ids, rows, strict=True)
        ):
            length = int(batch.lengths[index])
            encoded.append(
                _ReadoutSample(
                    video_id=identifier,
                    embeddings=embeddings[index, :length].detach().float().cpu(),
                    valid_mask=batch.valid_mask[index, :length].detach().cpu(),
                    period=float(row["selected_period_frames"]),
                    confidence=float(row["selected_period_confidence"]),
                    embedding_period=float(row["embedding_period_frames"]),
                    embedding_confidence=float(row["embedding_period_confidence"]),
                    pose_period=float(row["pose_energy_period_frames"]),
                    pose_confidence=float(row["pose_energy_period_confidence"]),
                    harmonic_factor=int(row["selected_harmonic_factor"]),
                    candidates=tuple(row["candidates"]),
                )
            )
    return tuple(encoded)


def _probe(
    model: PAMSModel,
    poses: Tensor,
    valid_mask: Tensor,
    *,
    config: PAMSConfig,
) -> dict[str, Any]:
    _, rows = _readout_batch(model, poses, valid_mask, config=config)
    if len(rows) != 1:
        raise ValueError("probe expects exactly one batch row")
    return rows[0]


def _zero_random_period_probes(
    model: PAMSModel,
    *,
    config: PAMSConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Run zero and seeded white-noise controls through both evidence paths."""

    frames = config.data.frames
    shape = (1, frames, config.data.keypoints, config.data.coordinates)
    valid = torch.ones((1, frames), dtype=torch.bool, device=device)
    zero = torch.zeros(shape, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1_041_729)
    random_pose = torch.rand(shape, generator=generator, dtype=torch.float32).to(device)
    return {
        "classification": _CLASSIFICATION,
        "algorithm": (
            "All-zero and seeded IID U[0,1] poses are evaluated by the frozen "
            "NoAbsPE encoder. The legacy embedding-velocity estimate and raw "
            "pose-energy estimate are retained separately; selected confidence "
            "is their product."
        ),
        "frames": frames,
        "zero_pose": _probe(model, zero, valid, config=config),
        "seeded_uniform_random_pose": _probe(
            model,
            random_pose,
            valid,
            config=config,
        ),
    }


def _noabs_position_index_invariance(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Verify that arbitrary position indices are ignored in NoAbsPE mode."""

    frame_cosines: list[float] = []
    video_medians: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            canonical = model.encoder(batch.poses, batch.valid_mask)
            time = batch.valid_mask.shape[1]
            reversed_indices = (
                torch.arange(time - 1, -1, -1, device=device)
                .expand(batch.valid_mask.shape[0], time)
                .clone()
            )
            permuted = model.encoder(
                batch.poses,
                batch.valid_mask,
                position_indices=reversed_indices,
            )
            cosine = F.cosine_similarity(canonical, permuted, dim=-1)
            for sample_index in range(cosine.shape[0]):
                values = (
                    cosine[sample_index][batch.valid_mask[sample_index]]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                if values.size:
                    if not np.isfinite(values).all():
                        raise RuntimeError("position-index invariance is non-finite")
                    frame_cosines.extend(values.tolist())
                    video_medians.append(float(np.median(values)))
    if not frame_cosines:
        raise RuntimeError("no valid frames available for NoAbsPE invariance")
    return {
        "classification": _CLASSIFICATION,
        "position_encoding_mode": model.encoder.position_encoding_mode,
        "algorithm": (
            "Canonical and reverse position-index tensors are supplied to the "
            "same eval-mode NoAbsPE encoder. Position indices are intentionally "
            "ignored, so valid-frame embeddings must be invariant."
        ),
        "sampled_video_total": len(sequences),
        "valid_frame_cosine": _summary(frame_cosines),
        "per_video_median_cosine": _summary(video_medians),
    }


def _synthetic_period_sequence(
    period_frames: int,
    *,
    frames: int,
    seed: int,
) -> PoseSequence:
    """Generate a deterministic exact-period, MediaPipe-shaped pose sequence."""

    if period_frames < 2 or period_frames > frames:
        raise ValueError("synthetic period must lie in [2, frames]")
    time = np.arange(frames, dtype=np.float64)
    phase = 2.0 * np.pi * time / float(period_frames)
    joint_phase = np.linspace(0.0, 2.0 * np.pi, 33, endpoint=False)
    joint_scale = np.linspace(0.65, 1.35, 33)
    rng = np.random.default_rng(seed + period_frames * 7_919)
    fixed_offset = rng.uniform(-0.03, 0.03, size=(33, 3))
    xyz = np.zeros((frames, 33, 3), dtype=np.float64)
    xyz[:, :, 0] = (
        0.50
        + fixed_offset[None, :, 0]
        + 0.22
        * joint_scale[None, :]
        * np.sin(phase[:, None] + joint_phase[None, :])
    )
    xyz[:, :, 1] = (
        0.45
        + fixed_offset[None, :, 1]
        + 0.17
        * joint_scale[None, :]
        * np.cos(phase[:, None] - 0.5 * joint_phase[None, :])
    )
    xyz[:, :, 2] = (
        0.40
        + fixed_offset[None, :, 2]
        + 0.11
        * joint_scale[None, :]
        * np.sin(phase[:, None] + 0.25 * joint_phase[None, :])
    )
    valid = np.ones(frames, dtype=np.bool_)
    return PoseSequence(
        video_id=f"synthetic-period-{period_frames:03d}",
        fps=30.0,
        xyz=per_frame_minmax(xyz, valid),
        valid_mask=valid,
    )


def _synthetic_period_recovery(
    model: PAMSModel,
    config: PAMSConfig,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Run the pose-reference harmonic readout on frozen exact-period probes."""

    periods = tuple(
        period
        for period in _SYNTHETIC_PERIODS
        if config.period.minimum <= period <= config.period.maximum
        and period <= config.data.frames
    )
    if len(periods) < 3:
        raise ValueError("configuration supports fewer than three synthetic periods")
    sequences = tuple(
        _synthetic_period_sequence(period, frames=config.data.frames, seed=seed)
        for period in periods
    )
    batch = collate_pose_sequences(sequences).to(device)
    _, evidence = _readout_batch(
        model,
        batch.poses,
        batch.valid_mask,
        config=config,
    )
    rows: list[dict[str, Any]] = []
    relative_errors: list[float] = []
    raw_pose_relative_errors: list[float] = []
    for expected, row in zip(periods, evidence, strict=True):
        relative_error = (
            abs(float(row["selected_period_frames"]) - expected) / expected
        )
        raw_pose_relative_error = (
            abs(float(row["pose_energy_period_frames"]) - expected) / expected
        )
        relative_errors.append(relative_error)
        raw_pose_relative_errors.append(raw_pose_relative_error)
        rows.append(
            {
                "expected_period_frames": expected,
                **row,
                "relative_error": relative_error,
                "pose_energy_relative_error": raw_pose_relative_error,
            }
        )
    return {
        "classification": _CLASSIFICATION,
        "truth_source": "deterministic in-run synthetic generation only",
        "periods": list(periods),
        "frames": config.data.frames,
        "normalization": "per_frame_minmax",
        "embedding_estimator": "legacy_embedding_velocity_fft",
        "pose_reference_estimator": "raw_pose_energy_fft",
        "maximum_harmonic": _MAXIMUM_HARMONIC,
        "rows": rows,
        "relative_error": _summary(relative_errors),
        "pose_energy_relative_error": _summary(raw_pose_relative_errors),
    }


def _criterion(
    *,
    value: float | None,
    threshold: float,
    operator: str,
) -> dict[str, float | str | bool | None]:
    finite = value is not None and math.isfinite(value)
    if operator == "<=":
        passed = finite and value <= threshold
    elif operator == ">=":
        passed = finite and value >= threshold
    else:
        raise ValueError(f"unsupported criterion operator: {operator}")
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "pass": bool(passed),
    }


def _gate_decision(
    *,
    frame_index_r2: float | None,
    zero_pose_confidence: float,
    random_pose_confidence: float,
    training_top_bin_share: float,
    permutation_median_cosine: float,
    synthetic_median_relative_error: float,
) -> dict[str, Any]:
    """Apply the same six frozen target-free anti-shortcut thresholds."""

    criteria = {
        "frame_index_r2": _criterion(
            value=frame_index_r2,
            threshold=_THRESHOLDS["frame_index_r2_maximum"],
            operator="<=",
        ),
        "zero_pose_period_confidence": _criterion(
            value=zero_pose_confidence,
            threshold=_THRESHOLDS["zero_pose_period_confidence_maximum"],
            operator="<=",
        ),
        "random_pose_period_confidence": _criterion(
            value=random_pose_confidence,
            threshold=_THRESHOLDS["random_pose_period_confidence_maximum"],
            operator="<=",
        ),
        "training_period_top_bin_share": _criterion(
            value=training_top_bin_share,
            threshold=_THRESHOLDS["training_period_top_bin_share_maximum"],
            operator="<=",
        ),
        "canonical_permuted_embedding_median_cosine": _criterion(
            value=permutation_median_cosine,
            threshold=_THRESHOLDS[
                "canonical_permuted_embedding_median_cosine_minimum"
            ],
            operator=">=",
        ),
        "synthetic_period_median_relative_error": _criterion(
            value=synthetic_median_relative_error,
            threshold=_THRESHOLDS[
                "synthetic_period_median_relative_error_maximum"
            ],
            operator="<=",
        ),
    }
    passed = all(bool(item["pass"]) for item in criteria.values())
    return {
        "thresholds_frozen_before_checkpoint_evaluation": True,
        "criteria": criteria,
        "overall_pass": passed,
        "dev84_prediction_authorized": passed,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }


def _validate_noabspe_config(config: PAMSConfig) -> None:
    if config.protocol != "ucfrep_526":
        raise ValueError("NoAbsPE predev gate requires protocol=ucfrep_526")
    if config.data.frames != 256:
        raise ValueError("NoAbsPE predev gate requires exactly 256 frames")
    if (config.period.minimum, config.period.maximum) != (4, 128):
        raise ValueError("NoAbsPE predev gate requires period range 4--128")
    if config.model.position_encoding_mode != "none":
        raise ValueError("NoAbsPE predev gate requires position_encoding_mode=none")
    if config.training.position_permutation_consistency_weight != 0.0:
        raise ValueError("NoAbsPE predev gate requires PE consistency weight zero")


def run_predev_gate(
    checkpoint_path: str | Path,
    config_path: str | Path,
    pose_cache_dir: str | Path,
    *,
    sample_size: int = 64,
    seed: int = 2026,
    device: str | torch.device | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run all read-only checks on checkpoint-bound training poses only."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    checkpoint = Path(checkpoint_path)
    configuration = Path(config_path)
    checkpoint_sha256, checkpoint_bytes = _stable_file_sha256(checkpoint)
    config_sha256, config_bytes = _stable_file_sha256(configuration)
    config = load_config(configuration)
    _validate_noabspe_config(config)
    stage, provenance = _peek_checkpoint(checkpoint, config)
    if stage != "encoder":
        raise ValueError("NoAbsPE predev gate requires an encoder checkpoint")
    resolved_device = _device(device)
    model = load_model_checkpoint(
        checkpoint,
        config,
        device=resolved_device,
        expected_stage="encoder",
        expected_provenance=provenance,
    )
    model.eval()
    sequences, sample_receipts, full_receipts, selected_ids = _load_training_poses(
        pose_cache_dir=Path(pose_cache_dir),
        provenance=provenance,
        config=config,
        sample_size=sample_size,
        seed=seed,
    )
    sample_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=sample_receipts,
    )
    full_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=full_receipts,
    )
    if full_snapshot.fingerprint != provenance.pose_cache_set_sha256:
        raise ValueError("full training pose-cache set does not match checkpoint provenance")

    encoded = _encode_training_samples(
        model,
        sequences,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    histogram = _period_histogram(
        encoded,
        minimum=config.period.minimum,
        maximum=config.period.maximum,
    )
    frame_probe = _frame_index_probe(encoded, seed=seed)
    probes = _zero_random_period_probes(
        model,
        config=config,
        device=resolved_device,
        seed=seed,
    )
    invariance = _noabs_position_index_invariance(
        model,
        sequences,
        device=resolved_device,
        batch_size=batch_size,
    )
    synthetic = _synthetic_period_recovery(
        model,
        config,
        device=resolved_device,
        seed=seed,
    )
    raw_r2 = frame_probe["r2"]
    frame_r2 = None if raw_r2 is None else float(raw_r2)
    decision = _gate_decision(
        frame_index_r2=frame_r2,
        zero_pose_confidence=float(
            probes["zero_pose"]["selected_period_confidence"]
        ),
        random_pose_confidence=float(
            probes["seeded_uniform_random_pose"]["selected_period_confidence"]
        ),
        training_top_bin_share=float(histogram["top_bin_share"]),
        permutation_median_cosine=float(
            invariance["valid_frame_cosine"]["median"]
        ),
        synthetic_median_relative_error=float(
            synthetic["relative_error"]["median"]
        ),
    )

    checkpoint_after = _stable_file_sha256(checkpoint)
    config_after = _stable_file_sha256(configuration)
    if checkpoint_after != (checkpoint_sha256, checkpoint_bytes):
        raise RuntimeError("checkpoint changed during predev gate")
    if config_after != (config_sha256, config_bytes):
        raise RuntimeError("configuration changed during predev gate")

    factor_counts = Counter(sample.harmonic_factor for sample in encoded)
    sample_rows = [
        {
            "video_id_sha256": _identifier_commitment((sample.video_id,)),
            "embedding_period_frames": sample.embedding_period,
            "embedding_period_confidence": sample.embedding_confidence,
            "pose_energy_period_frames": sample.pose_period,
            "pose_energy_period_confidence": sample.pose_confidence,
            "selected_harmonic_factor": sample.harmonic_factor,
            "selected_period_frames": sample.period,
            "selected_period_confidence": sample.confidence,
            "candidates": sample.candidates,
        }
        for sample in encoded
    ]
    return {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "classification": _CLASSIFICATION,
        "disclosed_by_pams_authors": False,
        "eligible_for_paper_table": False,
        "readout": {
            "embedding_evidence": "legacy_embedding_velocity_fft",
            "pose_reference_evidence": "raw_pose_energy_fft",
            "candidate_periods": "h * embedding_period for h=1..8 within [4,128]",
            "selection": (
                "minimum absolute log-ratio distance to pose-energy period; "
                "relative-distance and smaller-factor tie breaks"
            ),
            "confidence": "embedding_confidence * pose_energy_confidence",
            "legacy_estimator_modified": False,
        },
        "label_firewall": {
            "accepted_inputs": [
                "encoder_checkpoint",
                "experiment_config",
                "checkpoint_bound_training_pose_cache",
            ],
            "dataset_manifest_argument_supported": False,
            "action_or_count_label_argument_supported": False,
            "development_input_mounted": False,
            "development_pose_mounted": False,
            "development_labels_mounted": False,
            "sealed_test_input_mounted": False,
            "sealed_test_labels_mounted": False,
            "label_fields_accessed": [],
        },
        "inputs": {
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_stage": stage,
            "config_sha256": config_sha256,
            "config_bytes": config_bytes,
            "config_fingerprint": config.fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
            "position_encoding_mode": config.model.position_encoding_mode,
            "checkpoint_training_video_total": len(provenance.training_video_ids),
            "sampled_training_video_total": len(selected_ids),
            "sampled_training_video_ids_sha256": _identifier_commitment(selected_ids),
            "sample_pose_cache_set_sha256": sample_snapshot.fingerprint,
            "full_pose_cache_set_sha256": full_snapshot.fingerprint,
            "full_checkpoint_pose_cache_set_verified": True,
            "diagnostic_seed": seed,
            "device_type": resolved_device.type,
        },
        "training_pose_reference_harmonic_periods": {
            "sample_period_frames": _summary([sample.period for sample in encoded]),
            "period_confidence": _summary(
                [sample.confidence for sample in encoded]
            ),
            "legacy_embedding_period_frames": _summary(
                [sample.embedding_period for sample in encoded]
            ),
            "legacy_embedding_period_confidence": _summary(
                [sample.embedding_confidence for sample in encoded]
            ),
            "pose_energy_period_frames": _summary(
                [sample.pose_period for sample in encoded]
            ),
            "pose_energy_period_confidence": _summary(
                [sample.pose_confidence for sample in encoded]
            ),
            "harmonic_factor_histogram": {
                str(factor): factor_counts.get(factor, 0)
                for factor in range(1, _MAXIMUM_HARMONIC + 1)
            },
            "selected_period_histogram": histogram,
            "sample_evidence": sample_rows,
        },
        "zero_and_random_pose_period_probes": probes,
        "frame_index_linear_probe": frame_probe,
        "canonical_vs_permuted_pe_embeddings": invariance,
        "synthetic_period_recovery": synthetic,
        "gate": decision,
        "read_only_verification": {
            "checkpoint_sha256_unchanged": True,
            "config_sha256_unchanged": True,
            "model_or_training_state_updated": False,
            "pose_cache_write_operations": 0,
        },
    }


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pose-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    payload = run_predev_gate(
        arguments.checkpoint,
        arguments.config,
        arguments.pose_cache_dir,
        sample_size=arguments.sample_size,
        seed=arguments.seed,
        device=arguments.device,
        batch_size=arguments.batch_size,
    )
    _write_new_json(arguments.output, payload)
    print(
        json.dumps(
            {
                "artifact_type": payload["artifact_type"],
                "overall_pass": payload["gate"]["overall_pass"],
                "output": str(arguments.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
