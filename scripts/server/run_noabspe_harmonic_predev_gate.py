"""Target-free NoAbsPE harmonic-fundamental predev gate.

This independently inferred runner accepts only an encoder checkpoint, its
exact configuration, the checkpoint-bound training pose cache, and a new
output path. It has no development/test sample, dataset-manifest, action, or
count-label input. Passing all six frozen gates can authorize a separate
development prediction; it never authorizes sealed-test access.

The checkpoint loader follows the repository's current PyTorch checkpoint
format and must only be used with checkpoints produced by this repository;
PyTorch pickle checkpoints from untrusted sources are outside this CLI's
security boundary.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import Tensor

from pams.config import PAMSConfig, load_config
from pams.data import PoseCacheSetSnapshot
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
from pams.period import (
    HarmonicFundamentalDiagnostic,
    embedding_velocity_harmonic_fundamental_diagnostics,
)
from pams.training import collate_pose_sequences, load_model_checkpoint
from pams.types import PoseSequence

_ARTIFACT_TYPE = "pams_noabspe_harmonic_fundamental_predev_gate_v1"
_CLASSIFICATION = "inferred target-free NoAbsPE inference-repair diagnostic"
_MAXIMUM_HARMONIC = 8
_FUNDAMENTAL_ONLY_WEIGHT = 0.001
_LOCAL_BIN_RADIUS = 0
_ESTIMATOR_CONFIG: dict[str, Any] = {
    "id": "embedding-velocity-harmonic-fundamental-inferred-v1",
    "disclosed_by_pams_authors": False,
    "input": "L2-normalized encoder embeddings",
    "temporal_transform": "pairwise-valid first difference",
    "feature_aggregation": "sum of full-vector rFFT power",
    "window": "Hann periodic=False",
    "period_bounds_frames": [4, 128],
    "maximum_harmonic": _MAXIMUM_HARMONIC,
    "candidate_score": (
        "sqrt(max(original,residual-prewhitened)-fundamental-power * "
        "mean(max(overtone-power-0.1/uniform-bin-count,0))) + "
        "0.001*fundamental-power"
    ),
    "available_harmonic_normalization": "mean over available orders 2..H",
    "dominant_peak_prewhitening": (
        "mask-weighted least-squares cosine+sine+intercept at the strongest bin"
    ),
    "local_bin_radius": _LOCAL_BIN_RADIUS,
    "exact_score_tie_break": "shorter period only",
    "confidence": (
        "sqrt(normalized Herfindahl excess of candidate scores) multiplied by "
        "selected-family spectral-power excess over uniform bin expectation"
    ),
    "no_evidence": "minimum period with exactly zero confidence",
}


def _load_frozen_gate_module() -> ModuleType:
    """Load the preregistered six-gate implementation from the sibling runner."""

    source = Path(__file__).with_name("run_pe_permutation_predev_gate.py")
    spec = importlib.util.spec_from_file_location(
        "_pams_frozen_pe_predev_gate",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen predev gate module: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_FROZEN_GATE = _load_frozen_gate_module()
_THRESHOLDS = dict(_FROZEN_GATE._THRESHOLDS)


@dataclass(frozen=True, slots=True)
class _HarmonicEncodedSample:
    """One label-free encoder result with the opt-in harmonic readout."""

    video_id: str
    embeddings: Tensor
    valid_mask: Tensor
    period: float
    confidence: float
    diagnostic: HarmonicFundamentalDiagnostic


def _estimate(
    embeddings: Tensor,
    valid_mask: Tensor,
    *,
    config: PAMSConfig,
) -> tuple[HarmonicFundamentalDiagnostic, ...]:
    return embedding_velocity_harmonic_fundamental_diagnostics(
        embeddings,
        minimum=config.period.minimum,
        maximum=config.period.maximum,
        valid_mask=valid_mask,
        maximum_harmonic=_MAXIMUM_HARMONIC,
        fundamental_only_weight=_FUNDAMENTAL_ONLY_WEIGHT,
        local_bin_radius=_LOCAL_BIN_RADIUS,
    )


def _encode_training_samples(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> tuple[_HarmonicEncodedSample, ...]:
    encoded: list[_HarmonicEncodedSample] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(
                sequences[start : start + batch_size]
            ).to(device)
            embeddings = model.encoder(batch.poses, batch.valid_mask)
            diagnostics = _estimate(
                embeddings,
                batch.valid_mask,
                config=config,
            )
            for index, (identifier, diagnostic) in enumerate(
                zip(batch.video_ids, diagnostics, strict=True)
            ):
                length = int(batch.lengths[index])
                encoded.append(
                    _HarmonicEncodedSample(
                        video_id=identifier,
                        embeddings=embeddings[index, :length]
                        .detach()
                        .float()
                        .cpu(),
                        valid_mask=batch.valid_mask[index, :length]
                        .detach()
                        .cpu(),
                        period=diagnostic.selected_period,
                        confidence=diagnostic.confidence,
                        diagnostic=diagnostic,
                    )
                )
    return tuple(encoded)


def _period_probe(
    model: PAMSModel,
    poses: Tensor,
    *,
    config: PAMSConfig,
    valid_mask: Tensor,
) -> dict[str, Any]:
    with torch.inference_mode():
        embeddings = model.encoder(poses, valid_mask)
        diagnostic = _estimate(
            embeddings,
            valid_mask,
            config=config,
        )[0]
    return {
        "embedding_period_frames": diagnostic.selected_period,
        "embedding_period_confidence": diagnostic.confidence,
        "harmonic_diagnostic": asdict(diagnostic),
    }


def _zero_random_period_probes(
    model: PAMSModel,
    *,
    config: PAMSConfig,
    device: torch.device,
    seed: int,
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
    generator.manual_seed(seed + 1_041_729)
    random_pose = torch.rand(
        shape,
        generator=generator,
        dtype=torch.float32,
    ).to(device)
    return {
        "classification": _CLASSIFICATION,
        "algorithm": (
            "The frozen encoder is evaluated on all-zero and seeded IID U[0,1] "
            "pose inputs. Both periods and confidences use only the inferred "
            "harmonic-fundamental estimator recorded in estimator_config."
        ),
        "frames": frames,
        "zero_pose": _period_probe(
            model,
            zero,
            config=config,
            valid_mask=valid,
        ),
        "seeded_uniform_random_pose": _period_probe(
            model,
            random_pose,
            config=config,
            valid_mask=valid,
        ),
    }


def _synthetic_period_recovery(
    model: PAMSModel,
    config: PAMSConfig,
    *,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    periods = tuple(
        period
        for period in _FROZEN_GATE._SYNTHETIC_PERIODS
        if config.period.minimum <= period <= config.period.maximum
        and period <= config.data.frames
    )
    if len(periods) < 3:
        raise ValueError(
            "configuration supports fewer than three frozen synthetic periods"
        )
    sequences = tuple(
        _FROZEN_GATE._synthetic_period_sequence(
            period,
            frames=config.data.frames,
            seed=seed,
        )
        for period in periods
    )
    batch = collate_pose_sequences(sequences).to(device)
    model.eval()
    with torch.inference_mode():
        embeddings = model.encoder(batch.poses, batch.valid_mask)
        diagnostics = _estimate(
            embeddings,
            batch.valid_mask,
            config=config,
        )

    rows: list[dict[str, Any]] = []
    relative_errors: list[float] = []
    for expected, diagnostic in zip(periods, diagnostics, strict=True):
        error = (
            abs(diagnostic.selected_period - float(expected))
            / float(expected)
        )
        if not math.isfinite(error) or not math.isfinite(diagnostic.confidence):
            raise RuntimeError(
                "synthetic period diagnostic contains a non-finite value"
            )
        relative_errors.append(error)
        rows.append(
            {
                "expected_period_frames": expected,
                "predicted_period_frames": diagnostic.selected_period,
                "period_confidence": diagnostic.confidence,
                "relative_error": error,
                "harmonic_diagnostic": asdict(diagnostic),
            }
        )
    return {
        "classification": _CLASSIFICATION,
        "truth_source": "deterministic in-run synthetic generation only",
        "periods": list(periods),
        "frames": config.data.frames,
        "normalization": "per_frame_minmax",
        "estimator": _ESTIMATOR_CONFIG["id"],
        "rows": rows,
        "relative_error": _summary(relative_errors),
    }


def _validated_noabspe_candidate(config: PAMSConfig) -> None:
    if config.model.position_encoding_mode != "none":
        raise ValueError(
            "NoAbsPE harmonic predev gate requires position_encoding_mode=none"
        )
    if config.training.position_permutation_consistency_weight != 0.0:
        raise ValueError(
            "NoAbsPE harmonic predev gate requires zero PE consistency weight"
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
    """Run all read-only harmonic checks without label-bearing inputs."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")

    checkpoint = Path(checkpoint_path)
    configuration = Path(config_path)
    cache = Path(pose_cache_dir)
    checkpoint_sha256, checkpoint_bytes = _stable_file_sha256(checkpoint)
    config_sha256, config_bytes = _stable_file_sha256(configuration)
    config = load_config(configuration)
    if config.protocol != "ucfrep_526":
        raise ValueError(
            "NoAbsPE harmonic predev gate requires protocol=ucfrep_526"
        )
    if config.data.frames != 256:
        raise ValueError(
            "NoAbsPE harmonic predev gate requires exactly 256 frames"
        )
    if (config.period.minimum, config.period.maximum) != (4, 128):
        raise ValueError(
            "NoAbsPE harmonic predev gate requires period range 4--128"
        )
    _validated_noabspe_candidate(config)

    stage, provenance = _peek_checkpoint(checkpoint, config)
    if stage != "encoder":
        raise ValueError(
            "NoAbsPE harmonic predev gate requires an encoder checkpoint"
        )
    resolved_device = _device(device)
    model = load_model_checkpoint(
        checkpoint,
        config,
        device=resolved_device,
        expected_stage="encoder",
        expected_provenance=provenance,
    )
    model.eval()
    sequences, sample_receipts, full_receipts, selected_ids = (
        _load_training_poses(
            pose_cache_dir=cache,
            provenance=provenance,
            config=config,
            sample_size=sample_size,
            seed=seed,
        )
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
        raise ValueError(
            "full training pose-cache set does not match checkpoint provenance"
        )

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
    permutation = _FROZEN_GATE._permutation_consistency(
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

    zero_confidence = float(
        probes["zero_pose"]["embedding_period_confidence"]
    )
    random_confidence = float(
        probes["seeded_uniform_random_pose"][
            "embedding_period_confidence"
        ]
    )
    top_share = float(histogram["top_bin_share"])
    permutation_median = float(
        permutation["valid_frame_cosine"]["median"]
    )
    synthetic_median = float(synthetic["relative_error"]["median"])
    raw_r2 = frame_probe["r2"]
    frame_r2 = None if raw_r2 is None else float(raw_r2)
    decision = _FROZEN_GATE._gate_decision(
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

    diagnostics = tuple(sample.diagnostic for sample in encoded)
    return {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "classification": _CLASSIFICATION,
        "disclosed_by_pams_authors": False,
        "eligible_for_paper_table": False,
        "estimator_config": dict(_ESTIMATOR_CONFIG),
        "frozen_gate_thresholds": dict(_THRESHOLDS),
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
            "position_permutation_consistency_weight": (
                config.training.position_permutation_consistency_weight
            ),
            "checkpoint_training_video_total": len(
                provenance.training_video_ids
            ),
            "sampled_training_video_total": len(selected_ids),
            "sampled_training_video_ids_sha256": _identifier_commitment(
                selected_ids
            ),
            "sample_pose_cache_set_sha256": sample_snapshot.fingerprint,
            "full_pose_cache_set_sha256": full_snapshot.fingerprint,
            "full_checkpoint_pose_cache_set_verified": True,
            "diagnostic_seed": seed,
            "device_type": resolved_device.type,
        },
        "training_embedding_periods": {
            "sample_period_frames": _summary(
                [sample.period for sample in encoded]
            ),
            "period_confidence": _summary(
                [sample.confidence for sample in encoded]
            ),
            "histogram": histogram,
            "prewhitened_residual_power_fraction": _summary(
                [
                    diagnostic.prewhitened_residual_power_fraction
                    for diagnostic in diagnostics
                ]
            ),
            "candidate_distribution_concentration": _summary(
                [
                    diagnostic.candidate_distribution_concentration
                    for diagnostic in diagnostics
                ]
            ),
            "family_spectral_concentration": _summary(
                [
                    diagnostic.family_spectral_concentration
                    for diagnostic in diagnostics
                ]
            ),
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


def _parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Trusted repository-generated encoder checkpoint only.",
    )
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
