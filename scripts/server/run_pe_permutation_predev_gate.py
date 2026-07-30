"""Strict target-free predev gate for inferred PE-shortcut encoder candidates.

This runner accepts only an encoder checkpoint, its exact configuration, and
the checkpoint-bound training pose-cache directory.  It has no dataset
manifest, action, count-label, development, or test input surface.  Synthetic
period truth is generated inside this file and is used only for a component
diagnostic.

The PE-permutation consistency and no-absolute-PE experiments are independently
inferred and were not disclosed by the PAMS authors. Passing this gate can
authorize a separate development-only prediction run; it can never authorize
sealed-test access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
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
    _encode_training_samples,
    _frame_index_probe,
    _identifier_commitment,
    _load_training_poses,
    _peek_checkpoint,
    _period_histogram,
    _stable_file_sha256,
    _summary,
    _zero_random_period_probes,
)
from pams.model import PAMSModel
from pams.period import estimate_period_from_embeddings
from pams.training import collate_pose_sequences, load_model_checkpoint
from pams.types import PoseSequence

_ARTIFACT_TYPE = "pams_pe_permutation_consistency_predev_gate"
_CLASSIFICATION = "inferred target-free predev diagnostic"
_SYNTHETIC_PERIODS = (4, 8, 16, 32, 64, 128)
_THRESHOLDS = {
    "frame_index_r2_maximum": 0.10,
    "zero_pose_period_confidence_maximum": 0.10,
    "random_pose_period_confidence_maximum": 0.10,
    "training_period_top_bin_share_maximum": 0.25,
    "canonical_permuted_embedding_median_cosine_minimum": 0.95,
    "synthetic_period_median_relative_error_maximum": 0.10,
}


def _stable_sample_seed(seed: int, video_id: str) -> int:
    """Derive a batching-independent CPU permutation seed."""

    payload = f"{_ARTIFACT_TYPE}\0{seed}\0{video_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _independent_valid_frame_permutations(
    valid_mask: Tensor,
    video_ids: Sequence[str],
    *,
    seed: int,
) -> Tensor:
    """Permute valid PE rows independently while leaving invalid rows canonical."""

    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must have shape [batch, time]")
    if len(video_ids) != valid_mask.shape[0]:
        raise ValueError("video_ids must contain one identifier per batch row")
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("video_ids must be unique inside a batch")
    valid = valid_mask.detach().to(device="cpu", dtype=torch.bool)
    batch, time = valid.shape
    indices = torch.arange(time, dtype=torch.long).expand(batch, time).clone()
    for sample_index, video_id in enumerate(video_ids):
        positions = torch.nonzero(valid[sample_index], as_tuple=False).flatten()
        if positions.numel() < 2:
            continue
        generator = torch.Generator(device="cpu")
        generator.manual_seed(_stable_sample_seed(seed, str(video_id)))
        order = torch.randperm(positions.numel(), generator=generator)
        indices[sample_index, positions] = positions[order]
    return indices.to(device=valid_mask.device)


def _permutation_consistency(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    """Compare canonical and independently PE-permuted valid embeddings."""

    position_encoding_mode = model.encoder.position_encoding_mode
    frame_cosines: list[float] = []
    video_medians: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            canonical = model.encoder(batch.poses, batch.valid_mask)
            position_indices = _independent_valid_frame_permutations(
                batch.valid_mask,
                batch.video_ids,
                seed=seed,
            )
            permuted = model.encoder(
                batch.poses,
                batch.valid_mask,
                position_indices=position_indices,
            )
            cosine = F.cosine_similarity(canonical, permuted, dim=-1)
            for sample_index in range(cosine.shape[0]):
                selected = (
                    cosine[sample_index][batch.valid_mask[sample_index]]
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )
                if selected.size:
                    if not np.isfinite(selected).all():
                        raise RuntimeError("PE-permutation cosine contains a non-finite value")
                    frame_cosines.extend(selected.tolist())
                    video_medians.append(float(np.median(selected)))
    if not frame_cosines:
        raise RuntimeError("no valid training frames were available for PE consistency")
    if position_encoding_mode == "none":
        algorithm = (
            "The encoder disables absolute positional encoding, so the same "
            "stable valid-frame index shuffles are supplied as a strict "
            "invariance control and intentionally ignored by the encoder. "
            "Cosine similarity is measured only on valid output rows."
        )
    else:
        algorithm = (
            "For every checkpoint-bound sampled training video, valid PE row "
            "indices are independently shuffled by a stable SHA-256-derived "
            "seed. Invalid and padded rows retain canonical indices. The frozen "
            "eval-mode encoder processes canonical and shuffled views, and "
            "cosine similarity is measured only on valid output rows."
        )
    return {
        "classification": _CLASSIFICATION,
        "position_encoding_mode": position_encoding_mode,
        "algorithm": algorithm,
        "permutation_seed": seed,
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
    """Generate one exact integer-period, MediaPipe-shaped normalized sequence."""

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
        + 0.22 * joint_scale[None, :] * np.sin(phase[:, None] + joint_phase[None, :])
    )
    xyz[:, :, 1] = (
        0.45
        + fixed_offset[None, :, 1]
        + 0.17 * joint_scale[None, :] * np.cos(phase[:, None] - 0.5 * joint_phase[None, :])
    )
    xyz[:, :, 2] = (
        0.40
        + fixed_offset[None, :, 2]
        + 0.11 * joint_scale[None, :] * np.sin(phase[:, None] + 0.25 * joint_phase[None, :])
    )
    valid = np.ones(frames, dtype=np.bool_)
    normalized = per_frame_minmax(xyz, valid)
    return PoseSequence(
        video_id=f"synthetic-period-{period_frames:03d}",
        fps=30.0,
        xyz=normalized,
        valid_mask=valid,
    )


def _synthetic_period_recovery(
    model: PAMSModel,
    config: PAMSConfig,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    """Measure encoder period recovery on fixed label-free synthetic inputs."""

    periods = tuple(
        period
        for period in _SYNTHETIC_PERIODS
        if config.period.minimum <= period <= config.period.maximum and period <= config.data.frames
    )
    if len(periods) < 3:
        raise ValueError("configuration supports fewer than three frozen synthetic periods")
    sequences = tuple(
        _synthetic_period_sequence(period, frames=config.data.frames, seed=seed)
        for period in periods
    )
    batch = collate_pose_sequences(sequences).to(device)
    model.eval()
    with torch.inference_mode():
        embeddings = model.encoder(batch.poses, batch.valid_mask)
        predictions, confidences = estimate_period_from_embeddings(
            embeddings,
            minimum=config.period.minimum,
            maximum=config.period.maximum,
            valid_mask=batch.valid_mask,
        )
    rows: list[dict[str, float | int]] = []
    relative_errors: list[float] = []
    for expected, predicted, confidence in zip(
        periods,
        predictions.detach().float().cpu().tolist(),
        confidences.detach().float().cpu().tolist(),
        strict=True,
    ):
        error = abs(float(predicted) - float(expected)) / float(expected)
        if not math.isfinite(error) or not math.isfinite(float(confidence)):
            raise RuntimeError("synthetic period diagnostic contains a non-finite value")
        relative_errors.append(error)
        rows.append(
            {
                "expected_period_frames": expected,
                "predicted_period_frames": float(predicted),
                "period_confidence": float(confidence),
                "relative_error": error,
            }
        )
    return {
        "classification": _CLASSIFICATION,
        "truth_source": "deterministic in-run synthetic generation only",
        "periods": list(periods),
        "frames": config.data.frames,
        "normalization": "per_frame_minmax",
        "estimator": "embedding_velocity_fft",
        "rows": rows,
        "relative_error": _summary(relative_errors),
    }


def _criterion(
    *,
    value: float | None,
    threshold: float,
    operator: str,
) -> dict[str, float | str | bool | None]:
    """Build one finite, explicit frozen-threshold decision."""

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
    """Apply the preregistered target-free PE-shortcut thresholds."""

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
            threshold=_THRESHOLDS["canonical_permuted_embedding_median_cosine_minimum"],
            operator=">=",
        ),
        "synthetic_period_median_relative_error": _criterion(
            value=synthetic_median_relative_error,
            threshold=_THRESHOLDS["synthetic_period_median_relative_error_maximum"],
            operator="<=",
        ),
    }
    return {
        "thresholds_frozen_before_checkpoint_evaluation": True,
        "criteria": criteria,
        "overall_pass": all(bool(item["pass"]) for item in criteria.values()),
        "dev84_prediction_authorized": all(bool(item["pass"]) for item in criteria.values()),
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }


def _validated_candidate_mode(config: PAMSConfig) -> tuple[str, float]:
    """Accept only either trained PE consistency or the no-absolute-PE control."""

    mode = config.model.position_encoding_mode
    consistency_weight = config.training.position_permutation_consistency_weight
    if not math.isfinite(consistency_weight):
        raise ValueError("predev gate requires a finite consistency weight")
    if mode == "sinusoidal" and consistency_weight > 0.0:
        return mode, consistency_weight
    if mode == "none" and consistency_weight == 0.0:
        return mode, consistency_weight
    raise ValueError(
        "predev gate requires either sinusoidal position encoding with a "
        "positive consistency weight or no position encoding with zero "
        "consistency weight"
    )


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
    """Run all read-only predev checks without accepting any label-bearing input."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    checkpoint = Path(checkpoint_path)
    configuration = Path(config_path)
    cache = Path(pose_cache_dir)
    checkpoint_sha256, checkpoint_bytes = _stable_file_sha256(checkpoint)
    config_sha256, config_bytes = _stable_file_sha256(configuration)
    config = load_config(configuration)
    if config.protocol != "ucfrep_526":
        raise ValueError("PE-permutation predev gate requires protocol=ucfrep_526")
    if config.data.frames != 256:
        raise ValueError("PE-permutation predev gate requires exactly 256 frames")
    if (config.period.minimum, config.period.maximum) != (4, 128):
        raise ValueError("PE-permutation predev gate requires period range 4--128")
    position_encoding_mode, consistency_weight = _validated_candidate_mode(config)

    stage, provenance = _peek_checkpoint(checkpoint, config)
    if stage != "encoder":
        raise ValueError("PE-permutation predev gate requires an encoder checkpoint")
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
        pose_cache_dir=cache,
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
    permutation = _permutation_consistency(
        model,
        sequences,
        device=resolved_device,
        batch_size=batch_size,
        seed=seed,
    )
    synthetic = _synthetic_period_recovery(
        model,
        config,
        device=resolved_device,
        seed=seed,
    )

    zero_confidence = float(probes["zero_pose"]["embedding_period_confidence"])
    random_confidence = float(probes["seeded_uniform_random_pose"]["embedding_period_confidence"])
    top_share = float(histogram["top_bin_share"])
    permutation_median = float(permutation["valid_frame_cosine"]["median"])
    synthetic_median = float(synthetic["relative_error"]["median"])
    raw_r2 = frame_probe["r2"]
    frame_r2 = None if raw_r2 is None else float(raw_r2)
    decision = _gate_decision(
        frame_index_r2=frame_r2,
        zero_pose_confidence=zero_confidence,
        random_pose_confidence=random_confidence,
        training_top_bin_share=top_share,
        permutation_median_cosine=permutation_median,
        synthetic_median_relative_error=synthetic_median,
    )

    checkpoint_after = _stable_file_sha256(checkpoint)
    config_after = _stable_file_sha256(configuration)
    if checkpoint_after != (checkpoint_sha256, checkpoint_bytes):
        raise RuntimeError("checkpoint changed during predev gate")
    if config_after != (config_sha256, config_bytes):
        raise RuntimeError("configuration changed during predev gate")

    return {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "classification": _CLASSIFICATION,
        "disclosed_by_pams_authors": False,
        "eligible_for_paper_table": False,
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
            "position_encoding_mode": position_encoding_mode,
            "position_permutation_consistency_weight": consistency_weight,
            "checkpoint_training_video_total": len(provenance.training_video_ids),
            "sampled_training_video_total": len(selected_ids),
            "sampled_training_video_ids_sha256": _identifier_commitment(selected_ids),
            "sample_pose_cache_set_sha256": sample_snapshot.fingerprint,
            "full_pose_cache_set_sha256": full_snapshot.fingerprint,
            "full_checkpoint_pose_cache_set_verified": True,
            "diagnostic_seed": seed,
            "device_type": resolved_device.type,
        },
        "training_embedding_periods": {
            "sample_period_frames": _summary([sample.period for sample in encoded]),
            "period_confidence": _summary([sample.confidence for sample in encoded]),
            "histogram": histogram,
        },
        "zero_and_random_pose_period_probes": probes,
        "frame_index_linear_probe": frame_probe,
        "canonical_vs_permuted_pe_embeddings": permutation,
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
