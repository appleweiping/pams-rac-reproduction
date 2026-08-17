"""Frozen target-free predev gate for the NoAbsPE seed-3407 v12 candidate.

This protocol was derived from exploratory diagnostics after the v11
checkpoints had failed their original predev gate.  It is not disclosed by the
PAMS authors and must never be used to rewrite the v11 result.  The protocol is
frozen before training seed 3407.

The period argmax remains the projected-pose velocity vector-ACF estimate.
Only its peak-share confidence is calibrated against a fixed temporal
white-noise null.  The runner accepts no dataset manifest, development input,
test input, action, or count-label path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
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
    _stable_file_sha256,
    _summary,
)
from pams.model import PAMSModel
from pams.period import estimate_period_from_projected_pose
from pams.training import collate_pose_sequences, load_model_checkpoint
from pams.types import PoseSequence

_ARTIFACT_TYPE = "pams_noabspe_projected_null_predev_gate_v12"
_CLASSIFICATION = "exploratory-derived target-free candidate protocol"
_REQUIRED_CONFIG_FINGERPRINT = "e64029a8c1ae1bf258a71eabe03fd35cd9cd3f18bdf0c7e4ae17657cf9c6a13e"
_REQUIRED_SEED = 3407
_REQUIRED_DEVICE_TYPE = "cuda"
_DIAGNOSTIC_SEED = 2026
_NULL_SEED = 73_400_711
_NULL_SAMPLE_TOTAL = 2_048
_SIGNIFICANCE_ALPHA = 0.01
_SAMPLED_TRAINING_VIDEO_TOTAL = 64
_BATCH_SIZE = 32
_SYNTHETIC_PERIODS = (4, 8, 16, 32, 64, 128)
_THRESHOLDS = {
    "frame_index_r2_maximum": 0.10,
    "zero_pose_period_confidence_maximum": 0.10,
    "random_pose_period_confidence_maximum": 0.10,
    "training_period_top_bin_share_maximum": 0.25,
    "position_invariance_median_cosine_minimum": 0.95,
    "synthetic_period_median_relative_error_maximum": 0.10,
}


def _validate_candidate_config(config: PAMSConfig) -> None:
    """Accept only the exact seed-3407 v12 semantic configuration."""

    if config.fingerprint != _REQUIRED_CONFIG_FINGERPRINT:
        raise ValueError("predev gate requires the exact frozen v12 config")
    if config.protocol != "ucfrep_526" or config.seed != _REQUIRED_SEED:
        raise ValueError("predev gate requires ucfrep_526 seed 3407")
    if config.model.position_encoding_mode != "none":
        raise ValueError("predev gate requires position_encoding_mode=none")
    if config.training.position_permutation_consistency_weight != 0.0:
        raise ValueError("NoAbsPE must leave PE consistency disabled")
    if config.data.frames != 256:
        raise ValueError("predev gate requires exactly 256 frames")
    if (config.period.minimum, config.period.maximum) != (4, 128):
        raise ValueError("predev gate requires period range 4--128")
    if config.period.post_warmup_source != "projected_pose_velocity_vector_acf":
        raise ValueError("predev gate requires projected-pose vector-ACF training")


def _synthetic_period_sequence(period_frames: int, *, frames: int) -> PoseSequence:
    """Generate the exact deterministic sequence used by the frozen PE gate."""

    if period_frames < 2 or period_frames > frames:
        raise ValueError("synthetic period must lie in [2, frames]")
    time = np.arange(frames, dtype=np.float64)
    phase = 2.0 * np.pi * time / float(period_frames)
    joint_phase = np.linspace(0.0, 2.0 * np.pi, 33, endpoint=False)
    joint_scale = np.linspace(0.65, 1.35, 33)
    rng = np.random.default_rng(_DIAGNOSTIC_SEED + period_frames * 7_919)
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
    return PoseSequence(
        video_id=f"synthetic-period-{period_frames:03d}",
        fps=30.0,
        xyz=per_frame_minmax(xyz, valid),
        valid_mask=valid,
    )


def _projected_period_estimate(
    model: PAMSModel,
    poses: Tensor,
    valid_mask: Tensor,
    config: PAMSConfig,
) -> tuple[Tensor, Tensor]:
    """Return the frozen period argmax and its uncalibrated peak share."""

    _, projected_pose = model.encoder.forward_with_pre_pe(poses, valid_mask)
    return estimate_period_from_projected_pose(
        projected_pose,
        minimum=config.period.minimum,
        maximum=config.period.maximum,
        valid_mask=valid_mask,
    )


def _build_white_noise_null(
    model: PAMSModel,
    config: PAMSConfig,
    *,
    device: torch.device,
    sample_total: int = _NULL_SAMPLE_TOTAL,
    seed: int = _NULL_SEED,
    batch_size: int = _BATCH_SIZE,
) -> np.ndarray:
    """Build a deterministic IID U[0,1] temporal white-pose null."""

    if sample_total < 1:
        raise ValueError("null sample_total must be positive")
    if batch_size < 1:
        raise ValueError("null batch_size must be positive")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    statistics: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, sample_total, batch_size):
            current = min(batch_size, sample_total - start)
            poses = torch.rand(
                (
                    current,
                    config.data.frames,
                    config.data.keypoints,
                    config.data.coordinates,
                ),
                generator=generator,
                dtype=torch.float32,
            ).to(device)
            valid = torch.ones(
                (current, config.data.frames),
                dtype=torch.bool,
                device=device,
            )
            _, raw_confidence = _projected_period_estimate(
                model,
                poses,
                valid,
                config,
            )
            statistics.extend(raw_confidence.detach().float().cpu().tolist())
    null = np.asarray(statistics, dtype="<f4")
    if null.size != sample_total or not np.isfinite(null).all():
        raise RuntimeError("white-noise null contains invalid values")
    return null


def _calibrate_peak_share(
    raw_peak_share: float,
    null: np.ndarray,
    *,
    alpha: float = _SIGNIFICANCE_ALPHA,
) -> dict[str, float | int]:
    """Map one raw statistic to a +1 empirical p and bounded evidence score."""

    if not math.isfinite(raw_peak_share) or raw_peak_share < 0.0 or raw_peak_share > 1.0:
        raise ValueError("raw_peak_share must be finite and in [0, 1]")
    if null.ndim != 1 or not null.size or not np.isfinite(null).all():
        raise ValueError("null must be a non-empty finite vector")
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be finite and in (0, 1)")
    exceedances = int(np.count_nonzero(null >= np.float32(raw_peak_share)))
    empirical_p = (1.0 + exceedances) / (1.0 + null.size)
    confidence = max(0.0, 1.0 - empirical_p / alpha)
    return {
        "raw_peak_share": raw_peak_share,
        "null_exceedance_total": exceedances,
        "empirical_one_sided_p": empirical_p,
        "calibrated_periodicity_confidence": confidence,
    }


def _synthetic_period_recovery(
    model: PAMSModel,
    config: PAMSConfig,
    *,
    device: torch.device,
    null: np.ndarray,
) -> dict[str, Any]:
    sequences = tuple(
        _synthetic_period_sequence(period, frames=config.data.frames)
        for period in _SYNTHETIC_PERIODS
    )
    batch = collate_pose_sequences(sequences).to(device)
    model.eval()
    with torch.inference_mode():
        periods, raw_confidences = _projected_period_estimate(
            model,
            batch.poses,
            batch.valid_mask,
            config,
        )
    rows: list[dict[str, float | int]] = []
    relative_errors: list[float] = []
    for expected, predicted, raw_confidence in zip(
        _SYNTHETIC_PERIODS,
        periods.detach().float().cpu().tolist(),
        raw_confidences.detach().float().cpu().tolist(),
        strict=True,
    ):
        error = abs(float(predicted) - expected) / expected
        relative_errors.append(error)
        rows.append(
            {
                "expected_period_frames": expected,
                "predicted_period_frames": float(predicted),
                "relative_error": error,
                **_calibrate_peak_share(float(raw_confidence), null),
            }
        )
    return {
        "classification": _CLASSIFICATION,
        "truth_source": "deterministic in-run synthetic generation only",
        "periods": list(_SYNTHETIC_PERIODS),
        "prediction_period_unchanged_by_calibration": True,
        "rows": rows,
        "relative_error": _summary(relative_errors),
    }


def _zero_random_probes(
    model: PAMSModel,
    config: PAMSConfig,
    *,
    device: torch.device,
    null: np.ndarray,
) -> dict[str, Any]:
    frames = config.data.frames
    shape = (
        1,
        frames,
        config.data.keypoints,
        config.data.coordinates,
    )
    valid = torch.ones((1, frames), dtype=torch.bool, device=device)
    zero = torch.zeros(shape, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_DIAGNOSTIC_SEED + 1_041_729)
    random_pose = torch.rand(
        shape,
        generator=generator,
        dtype=torch.float32,
    ).to(device)

    def evaluate(poses: Tensor) -> dict[str, float | int]:
        with torch.inference_mode():
            period, raw_confidence = _projected_period_estimate(
                model,
                poses,
                valid,
                config,
            )
        return {
            "period_frames": float(period[0]),
            **_calibrate_peak_share(float(raw_confidence[0]), null),
        }

    return {
        "classification": _CLASSIFICATION,
        "frames": frames,
        "probe_random_seed": _DIAGNOSTIC_SEED + 1_041_729,
        "probe_random_seed_is_disjoint_from_null_seed": True,
        "zero_pose": evaluate(zero),
        "seeded_uniform_random_pose": evaluate(random_pose),
    }


def _training_projected_periods(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    config: PAMSConfig,
    *,
    device: torch.device,
    batch_size: int = _BATCH_SIZE,
) -> dict[str, Any]:
    periods: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            estimates, _ = _projected_period_estimate(
                model,
                batch.poses,
                batch.valid_mask,
                config,
            )
            periods.extend(estimates.detach().float().cpu().tolist())
    rounded = [
        min(
            max(math.floor(period + 0.5), config.period.minimum),
            config.period.maximum,
        )
        for period in periods
    ]
    frequencies = Counter(rounded)
    top_period, top_frequency = min(
        frequencies.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return {
        "classification": _CLASSIFICATION,
        "prediction_period_unchanged_by_calibration": True,
        "sample_period_frames": _summary(periods),
        "histogram": {
            "rounding": "floor(period_frames + 0.5)",
            "nonzero_bin_frequencies": {
                str(period): frequencies[period] for period in sorted(frequencies)
            },
            "top_bin_period_frames": int(top_period),
            "top_bin_frequency": int(top_frequency),
            "top_bin_share": top_frequency / len(periods),
        },
    }


def _noabs_position_invariance(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    *,
    device: torch.device,
    batch_size: int = _BATCH_SIZE,
) -> dict[str, Any]:
    """Verify that changing valid PE indices cannot affect NoAbsPE outputs."""

    frame_cosines: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            canonical = model.encoder(batch.poses, batch.valid_mask)
            batch_total, time = batch.valid_mask.shape
            position_indices = (
                torch.arange(
                    time,
                    device=device,
                )
                .expand(batch_total, time)
                .clone()
            )
            for row in range(batch_total):
                positions = torch.nonzero(
                    batch.valid_mask[row],
                    as_tuple=False,
                ).flatten()
                position_indices[row, positions] = positions.flip(0)
            changed = model.encoder(
                batch.poses,
                batch.valid_mask,
                position_indices=position_indices,
            )
            cosine = F.cosine_similarity(canonical, changed, dim=-1)
            selected = cosine[batch.valid_mask].detach().float().cpu().numpy()
            frame_cosines.extend(selected.tolist())
    if not frame_cosines:
        raise RuntimeError("position invariance requires valid training frames")
    return {
        "classification": _CLASSIFICATION,
        "position_encoding_mode": model.encoder.position_encoding_mode,
        "intervention": "reverse valid position_indices independently per video",
        "valid_frame_cosine": _summary(frame_cosines),
    }


def _criterion(
    value: float | None,
    *,
    threshold: float,
    operator: str,
) -> dict[str, float | str | bool | None]:
    finite = value is not None and math.isfinite(value)
    if operator == "<=":
        passed = finite and value <= threshold
    elif operator == ">=":
        passed = finite and value >= threshold
    else:
        raise ValueError("unsupported gate operator")
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
    position_invariance_median_cosine: float,
    synthetic_median_relative_error: float,
) -> dict[str, Any]:
    criteria = {
        "frame_index_r2": _criterion(
            frame_index_r2,
            threshold=_THRESHOLDS["frame_index_r2_maximum"],
            operator="<=",
        ),
        "zero_pose_period_confidence": _criterion(
            zero_pose_confidence,
            threshold=_THRESHOLDS["zero_pose_period_confidence_maximum"],
            operator="<=",
        ),
        "random_pose_period_confidence": _criterion(
            random_pose_confidence,
            threshold=_THRESHOLDS["random_pose_period_confidence_maximum"],
            operator="<=",
        ),
        "training_period_top_bin_share": _criterion(
            training_top_bin_share,
            threshold=_THRESHOLDS["training_period_top_bin_share_maximum"],
            operator="<=",
        ),
        "position_invariance_median_cosine": _criterion(
            position_invariance_median_cosine,
            threshold=_THRESHOLDS["position_invariance_median_cosine_minimum"],
            operator=">=",
        ),
        "synthetic_period_median_relative_error": _criterion(
            synthetic_median_relative_error,
            threshold=_THRESHOLDS["synthetic_period_median_relative_error_maximum"],
            operator="<=",
        ),
    }
    overall = all(bool(row["pass"]) for row in criteria.values())
    return {
        "thresholds_frozen_before_seed3407_training": True,
        "criteria": criteria,
        "overall_pass": overall,
        "dev84_prediction_authorized": overall,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }


def run_predev_gate(
    checkpoint_path: str | Path,
    config_path: str | Path,
    train_pose_cache_dir: str | Path,
) -> dict[str, Any]:
    """Run the read-only v12 gate without accepting any labeled input."""

    checkpoint = Path(checkpoint_path)
    configuration = Path(config_path)
    pose_cache = Path(train_pose_cache_dir)
    checkpoint_before = _stable_file_sha256(checkpoint)
    config_before = _stable_file_sha256(configuration)
    config = load_config(configuration)
    _validate_candidate_config(config)
    stage, provenance = _peek_checkpoint(checkpoint, config)
    if stage != "encoder":
        raise ValueError("v12 predev gate requires an encoder checkpoint")

    device = _device(_REQUIRED_DEVICE_TYPE)
    model = load_model_checkpoint(
        checkpoint,
        config,
        device=device,
        expected_stage="encoder",
        expected_provenance=provenance,
    ).eval()
    sequences, sample_receipts, full_receipts, selected_ids = _load_training_poses(
        pose_cache_dir=pose_cache,
        provenance=provenance,
        config=config,
        sample_size=_SAMPLED_TRAINING_VIDEO_TOTAL,
        seed=_DIAGNOSTIC_SEED,
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
        raise ValueError("full training pose-cache set does not match checkpoint")

    null = _build_white_noise_null(
        model,
        config,
        device=device,
    )
    synthetic = _synthetic_period_recovery(
        model,
        config,
        device=device,
        null=null,
    )
    nuisance = _zero_random_probes(
        model,
        config,
        device=device,
        null=null,
    )
    training_periods = _training_projected_periods(
        model,
        sequences,
        config,
        device=device,
    )
    invariance = _noabs_position_invariance(
        model,
        sequences,
        device=device,
    )
    encoded = _encode_training_samples(
        model,
        sequences,
        config=config,
        device=device,
        batch_size=_BATCH_SIZE,
    )
    frame_probe = _frame_index_probe(
        encoded,
        seed=_DIAGNOSTIC_SEED,
    )
    raw_r2 = frame_probe["r2"]
    frame_r2 = None if raw_r2 is None else float(raw_r2)
    decision = _gate_decision(
        frame_index_r2=frame_r2,
        zero_pose_confidence=float(nuisance["zero_pose"]["calibrated_periodicity_confidence"]),
        random_pose_confidence=float(
            nuisance["seeded_uniform_random_pose"]["calibrated_periodicity_confidence"]
        ),
        training_top_bin_share=float(training_periods["histogram"]["top_bin_share"]),
        position_invariance_median_cosine=float(invariance["valid_frame_cosine"]["median"]),
        synthetic_median_relative_error=float(synthetic["relative_error"]["median"]),
    )

    checkpoint_after = _stable_file_sha256(checkpoint)
    config_after = _stable_file_sha256(configuration)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("checkpoint changed during predev gate")
    if config_after != config_before:
        raise RuntimeError("configuration changed during predev gate")
    null_bytes = null.tobytes(order="C")
    return {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "classification": _CLASSIFICATION,
        "disclosed_by_pams_authors": False,
        "eligible_for_paper_table": False,
        "protocol_freeze": {
            "derived_from_post_v11_exploration": True,
            "must_not_rewrite_v11_results": True,
            "frozen_before_seed3407_training": True,
            "candidate_config_fingerprint": _REQUIRED_CONFIG_FINGERPRINT,
            "null_seed": _NULL_SEED,
            "null_sample_total": _NULL_SAMPLE_TOTAL,
            "significance_alpha": _SIGNIFICANCE_ALPHA,
            "white_noise_batch_size": _BATCH_SIZE,
            "diagnostic_seed": _DIAGNOSTIC_SEED,
            "sampled_training_video_total": _SAMPLED_TRAINING_VIDEO_TOTAL,
            "required_device_type": _REQUIRED_DEVICE_TYPE,
        },
        "method": {
            "period_input": "encoder projected_pose_pre_pe",
            "period_estimator": "projected-pose velocity vector-ACF spectrum",
            "period_prediction_changed_by_calibration": False,
            "raw_statistic": ("selected bounded-band vector-ACF spectral power share"),
            "null": "fixed seeded IID U[0,1] temporal white pose noise",
            "empirical_p": ("(1 + count(null_stat >= observed_stat)) / (N + 1)"),
            "confidence": "max(0, 1 - empirical_p / alpha)",
        },
        "null_distribution": {
            "sample_total": _NULL_SAMPLE_TOTAL,
            "seed": _NULL_SEED,
            "generation_batch_size": _BATCH_SIZE,
            "probe_random_seed_is_disjoint": True,
            "alpha": _SIGNIFICANCE_ALPHA,
            "dtype": "little-endian float32",
            "sha256": hashlib.sha256(null_bytes).hexdigest(),
            "summary": _summary(null.astype(np.float64)),
            "quantiles": {
                "q900": float(np.quantile(null, 0.900)),
                "q950": float(np.quantile(null, 0.950)),
                "q990": float(np.quantile(null, 0.990)),
                "q995": float(np.quantile(null, 0.995)),
                "q999": float(np.quantile(null, 0.999)),
            },
        },
        "inputs": {
            "checkpoint_sha256": checkpoint_before[0],
            "checkpoint_bytes": checkpoint_before[1],
            "checkpoint_stage": stage,
            "config_sha256": config_before[0],
            "config_bytes": config_before[1],
            "config_fingerprint": config.fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
            "checkpoint_training_video_total": len(provenance.training_video_ids),
            "sampled_training_video_total": len(selected_ids),
            "sampled_training_video_ids_sha256": _identifier_commitment(selected_ids),
            "sample_pose_cache_set_sha256": sample_snapshot.fingerprint,
            "full_pose_cache_set_sha256": full_snapshot.fingerprint,
            "full_checkpoint_pose_cache_set_verified": True,
            "diagnostic_seed": _DIAGNOSTIC_SEED,
            "device_type": device.type,
        },
        "label_firewall": {
            "accepted_inputs": [
                "encoder_checkpoint",
                "exact_v12_experiment_config",
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
        "synthetic_period_recovery": synthetic,
        "zero_and_random_pose_period_probes": nuisance,
        "training_projected_pose_periods": training_periods,
        "position_index_invariance": invariance,
        "frame_index_linear_probe": frame_probe,
        "gate": decision,
        "read_only_verification": {
            "checkpoint_sha256_unchanged": True,
            "config_sha256_unchanged": True,
            "model_or_training_state_updated": False,
            "pose_cache_write_operations": 0,
        },
    }


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
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


def _parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--train-pose-cache-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    payload = run_predev_gate(
        arguments.checkpoint,
        arguments.config,
        arguments.train_pose_cache_dir,
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
