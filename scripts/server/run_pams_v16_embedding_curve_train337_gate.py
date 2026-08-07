"""Gate the deterministic v16 embedding action curve on train337 only.

This read-only, target-free runner accepts only the exact terminal PAMS-v16
encoder/checkpoint pair, its original config, the single-field inference
candidate config, the checkpoint-bound train337 pose cache, and the exact
source-export receipt.  It has no manifest, label, count, action, development,
or test input surface.

The thresholds below are preregistered in source before any candidate run.
Passing can authorize one isolated label-free dev84 prediction, never scoring
or sealed test105 access.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from torch import Tensor

from pams.config import PAMSConfig, load_config
from pams.consensus import MultiExpertCounter
from pams.data import PoseCacheSetSnapshot
from pams.diagnostics import (
    _device,
    _identifier_commitment,
    _load_training_poses,
    _peek_checkpoint,
    _stable_file_sha256,
    _summary,
)
from pams.embedding_curve import (
    build_frequency_matched_action_curves,
    estimate_embedding_action_curves,
    validate_embedding_action_curve_compatibility,
)
from pams.reproducibility import clean_git_revision, hardware_fingerprint, sha256_json
from pams.training import (
    collate_pose_sequences,
    load_model_checkpoint,
    validate_terminal_checkpoint,
)
from pams.types import PoseSequence

try:
    from scripts.server import run_pams_v14_dual_path_counterfactual as _v14
    from scripts.server import run_pams_v16_terminal_dual_path_gate as _v16
except ModuleNotFoundError:
    import run_pams_v14_dual_path_counterfactual as _v14  # type: ignore[no-redef]
    import run_pams_v16_terminal_dual_path_gate as _v16  # type: ignore[no-redef]


_ARTIFACT_TYPE = "pams_v16_embedding_curve_train337_gate"
_RECEIPT_TYPE = "pams_v16_embedding_curve_train337_gate_receipt"
_CLASSIFICATION = "inferred target-free deterministic embedding action curve"
_EXPECTED_TRAINING_VIDEOS = 337
_TIME_SCALE_FACTORS = (0.50, 0.75)
_EXPECTED_CANDIDATE_CONFIG_SHA256 = (
    "48cb4cbe6fe80d9113ec5307fa55496a4be9fa01f548a3b0a6ab2c4348133e05"
)

# Frozen before the first train337 candidate execution.  The first four are
# the unchanged post-Transformer v16 period gate.  Every remaining threshold
# was requested and fixed before observing this readout's result.
_THRESHOLDS: Mapping[str, float] = MappingProxyType(
    {
        "upstream_period_boundary_share_maximum_exclusive": 0.25,
        "upstream_period_mode_share_maximum_exclusive": 0.25,
        "upstream_period_time_scale_eligible_fraction_minimum": 0.50,
        "upstream_period_time_scale_median_relative_error_maximum": 0.15,
        "action_curve_std_median_minimum": 0.05,
        "algorithm1_majority_share_minimum": 0.60,
        "count_time_scale_exact_fraction_minimum": 0.70,
        "count_time_scale_off_by_one_fraction_minimum": 0.90,
        "selected_count_fft_gap_median_maximum": 1.0,
        "count_zero_share_maximum_exclusive": 0.25,
        "count_mode_share_maximum_exclusive": 0.25,
        "time_shuffle_harmonic_median_ratio_maximum": 0.75,
        "zero_pose_positive_period_confidence_share_maximum": 0.05,
        "zero_pose_curve_available_share_maximum": 0.05,
    }
)


@dataclass(frozen=True, slots=True)
class _CandidateSample:
    video_id: str
    period_frames: float
    period_confidence: float
    action_curve_std: float
    raw_action_curve_std: float
    harmonic_energy_fraction: float
    curve_available: bool
    count: int
    reference_count: int
    expert_counts: tuple[int, int, int]
    selected_expert: str
    selection_rule: str

    @property
    def confidence(self) -> float:
        """Expose the established period-distribution protocol field name."""

        return self.period_confidence


def _finite_float(value: Tensor | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("embedding action-curve gate produced a non-finite value")
    return result


def _validated_counter(config: PAMSConfig) -> MultiExpertCounter:
    consensus = config.consensus
    if consensus.expert_mode != "multi":
        raise ValueError("embedding action-curve candidate requires all three experts")
    return MultiExpertCounter(
        sigma_multipliers=consensus.sigma_multipliers,
        distance_multipliers=consensus.distance_multipliers,
        short_window_multiplier=consensus.short_window_multiplier,
        long_window_multiplier=consensus.long_window_multiplier,
        height_factor=consensus.height_factor,
        prominence_factor=consensus.prominence_factor,
        long_window_weight=consensus.long_window_weight,
        expert_mode=consensus.expert_mode,
    )


def _selection_rule(
    expert_counts: tuple[int, int, int],
    *,
    selected_count: int,
    reference_count: int,
) -> str:
    frequencies = Counter(expert_counts)
    majority_count, votes = max(
        frequencies.items(),
        key=lambda item: (item[1], -abs(item[0] - reference_count), -item[0]),
    )
    if votes >= 2:
        if selected_count != majority_count:
            raise RuntimeError("Algorithm 1 did not select the expert majority")
        return "majority_first"
    minimum_gap = min(abs(value - reference_count) for value in expert_counts)
    if (
        selected_count not in expert_counts
        or abs(selected_count - reference_count) != minimum_gap
    ):
        raise RuntimeError("Algorithm 1 did not select an FFT-nearest expert")
    return "fft_nearest_fallback"


def _encode_candidates(
    model: torch.nn.Module,
    sequences: Sequence[PoseSequence],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> tuple[_CandidateSample, ...]:
    """Run the exact inference readout and unchanged three-expert counter."""

    counter = _validated_counter(config)
    samples: list[_CandidateSample] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            embeddings = model.encoder(batch.poses, batch.valid_mask)
            readout = estimate_embedding_action_curves(
                embeddings,
                minimum_period=config.period.minimum,
                maximum_period=config.period.maximum,
                valid_mask=batch.valid_mask,
                timeline_lengths=batch.lengths,
            )
            for index, video_id in enumerate(batch.video_ids):
                length = int(batch.lengths[index])
                curve = readout.curves[index, :length]
                mask = batch.valid_mask[index, :length]
                result = counter.count(
                    curve,
                    period_frames=_finite_float(readout.periods[index]),
                    valid_mask=mask,
                    period_confidence=_finite_float(
                        readout.period_confidences[index]
                    ),
                    reference_frames=length,
                )
                if result.selection_mode != "multi":
                    raise RuntimeError("candidate bypassed three-expert consensus")
                rule = _selection_rule(
                    result.expert_counts,
                    selected_count=result.count,
                    reference_count=result.reference_count,
                )
                samples.append(
                    _CandidateSample(
                        video_id=video_id,
                        period_frames=_finite_float(readout.periods[index]),
                        period_confidence=_finite_float(
                            readout.period_confidences[index]
                        ),
                        action_curve_std=_finite_float(
                            readout.curve_standard_deviations[index]
                        ),
                        raw_action_curve_std=_finite_float(
                            readout.raw_curve_standard_deviations[index]
                        ),
                        harmonic_energy_fraction=_finite_float(
                            readout.harmonic_energy_fractions[index]
                        ),
                        curve_available=bool(readout.available[index]),
                        count=result.count,
                        reference_count=result.reference_count,
                        expert_counts=result.expert_counts,
                        selected_expert=result.selected_expert,
                        selection_rule=rule,
                    )
                )
    return tuple(samples)


def _period_distribution(
    samples: Sequence[_CandidateSample],
    *,
    config: PAMSConfig,
) -> dict[str, Any]:
    # The v14 function is deliberately reused so boundary/mode semantics are
    # byte-for-byte the established upstream period gate semantics.
    return _v14._distribution(
        samples,
        minimum=config.period.minimum,
        maximum=config.period.maximum,
    )


def _readout_distribution(samples: Sequence[_CandidateSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("readout distribution requires at least one sample")
    curve_stds = np.asarray(
        [sample.action_curve_std for sample in samples], dtype=np.float64
    )
    raw_stds = np.asarray(
        [sample.raw_action_curve_std for sample in samples], dtype=np.float64
    )
    harmonic_fractions = np.asarray(
        [sample.harmonic_energy_fraction for sample in samples], dtype=np.float64
    )
    gaps = np.asarray(
        [abs(sample.count - sample.reference_count) for sample in samples],
        dtype=np.float64,
    )
    majority = np.asarray(
        [sample.selection_rule == "majority_first" for sample in samples],
        dtype=np.bool_,
    )
    available = np.asarray(
        [sample.curve_available for sample in samples], dtype=np.bool_
    )
    return {
        "record_total": len(samples),
        "action_curve_std": _summary(curve_stds),
        "raw_action_curve_std": _summary(raw_stds),
        "harmonic_energy_fraction": _summary(harmonic_fractions),
        "curve_available_share": float(np.mean(available)),
        "algorithm1_majority_share": float(np.mean(majority)),
        "algorithm1_fft_nearest_fallback_share": float(np.mean(~majority)),
        "selected_count_fft_reference_absolute_gap": _summary(gaps),
        "selected_count_is_one_of_three_experts": all(
            sample.count in sample.expert_counts for sample in samples
        ),
        "majority_first_then_fft_nearest_verified": True,
    }


def _count_distribution(samples: Sequence[_CandidateSample]) -> dict[str, Any]:
    if not samples:
        raise ValueError("count distribution requires at least one sample")
    counts = np.asarray([sample.count for sample in samples], dtype=np.int64)
    frequencies = Counter(int(value) for value in counts)
    mode_count, mode_total = min(
        frequencies.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return {
        "record_total": len(samples),
        "zero_share": float(np.mean(counts == 0)),
        "mode_count": mode_count,
        "mode_frequency": mode_total,
        "mode_share": mode_total / len(samples),
        "unique_count_total": len(frequencies),
        "count": _summary(counts.astype(np.float64)),
        "histogram": {str(key): value for key, value in sorted(frequencies.items())},
    }


def _time_scale_consistency(
    model: torch.nn.Module,
    sequences: Sequence[PoseSequence],
    baseline: Sequence[_CandidateSample],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if [sample.video_id for sample in baseline] != [item.video_id for item in sequences]:
        raise ValueError("baseline candidate order differs from train337")
    period_errors: list[float] = []
    period_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []
    for factor in _TIME_SCALE_FACTORS:
        scaled_sequences = tuple(
            _v14._resample_for_time_scale(sequence, factor)
            for sequence in sequences
        )
        scaled = _encode_candidates(
            model,
            scaled_sequences,
            config=config,
            device=device,
            batch_size=batch_size,
        )
        for source, candidate in zip(baseline, scaled, strict=True):
            expected_period = source.period_frames * factor
            eligible = (
                source.period_confidence > 0.0
                and candidate.period_confidence > 0.0
                and config.period.minimum
                <= expected_period
                <= config.period.maximum
            )
            relative_error = (
                abs(candidate.period_frames - expected_period) / expected_period
                if eligible
                else None
            )
            if relative_error is not None:
                if not math.isfinite(relative_error):
                    raise RuntimeError("period scale error is non-finite")
                period_errors.append(relative_error)
            period_rows.append(
                {
                    "video_id": source.video_id,
                    "time_scale_factor": factor,
                    "baseline_period_frames": source.period_frames,
                    "baseline_confidence": source.period_confidence,
                    "expected_scaled_period_frames": expected_period,
                    "scaled_period_frames": candidate.period_frames,
                    "scaled_confidence": candidate.period_confidence,
                    "eligible": eligible,
                    "relative_error": relative_error,
                }
            )
            difference = abs(candidate.count - source.count)
            count_rows.append(
                {
                    "video_id": source.video_id,
                    "time_scale_factor": factor,
                    "baseline_count": source.count,
                    "scaled_count": candidate.count,
                    "absolute_difference": difference,
                    "exact": difference == 0,
                    "within_one": difference <= 1,
                }
            )
    candidate_total = len(period_rows)
    exact_total = sum(bool(row["exact"]) for row in count_rows)
    within_one_total = sum(bool(row["within_one"]) for row in count_rows)
    return {
        "factors": list(_TIME_SCALE_FACTORS),
        "period": {
            "candidate_comparison_total": candidate_total,
            "eligible_comparison_total": len(period_errors),
            "eligible_comparison_fraction": (
                len(period_errors) / candidate_total if candidate_total else 0.0
            ),
            "relative_error": _summary(period_errors),
            "rows": period_rows,
        },
        "count": {
            "comparison_total": len(count_rows),
            "exact_total": exact_total,
            "exact_fraction": exact_total / len(count_rows) if count_rows else 0.0,
            "within_one_total": within_one_total,
            "within_one_fraction": (
                within_one_total / len(count_rows) if count_rows else 0.0
            ),
            "rows": count_rows,
        },
    }


def _deterministically_shuffle_valid_embeddings(
    embeddings: Tensor,
    valid_mask: Tensor,
    video_ids: Sequence[str],
) -> Tensor:
    """Destroy temporal order without changing values, masks, or dense slots."""

    if embeddings.ndim != 3 or valid_mask.shape != embeddings.shape[:2]:
        raise ValueError("shuffle inputs must have [batch, time, dimension] and [batch, time]")
    if len(video_ids) != embeddings.shape[0]:
        raise ValueError("shuffle video IDs must match the embedding batch")
    shuffled = embeddings.detach().clone()
    for index, video_id in enumerate(video_ids):
        valid_indices = torch.flatnonzero(valid_mask[index].detach().cpu())
        if valid_indices.numel() < 2:
            continue
        digest = hashlib.sha256(
            f"pams-embedding-time-shuffle-v1\0{video_id}".encode("utf-8")
        ).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))
        permutation = torch.randperm(valid_indices.numel(), generator=generator)
        source_indices = valid_indices[permutation].to(device=embeddings.device)
        destination_indices = valid_indices.to(device=embeddings.device)
        shuffled[index, destination_indices] = embeddings[index, source_indices]
    return shuffled


def _corruption_controls(
    model: torch.nn.Module,
    sequences: Sequence[PoseSequence],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    """Measure deterministic time-shuffle and all-zero-pose degradation."""

    baseline_harmonic: list[float] = []
    shuffled_harmonic: list[float] = []
    zero_confidences: list[float] = []
    zero_available: list[bool] = []
    zero_curve_stds: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            embeddings = model.encoder(batch.poses, batch.valid_mask)
            baseline = estimate_embedding_action_curves(
                embeddings,
                minimum_period=config.period.minimum,
                maximum_period=config.period.maximum,
                valid_mask=batch.valid_mask,
                timeline_lengths=batch.lengths,
            )
            shuffled_embeddings = _deterministically_shuffle_valid_embeddings(
                embeddings,
                batch.valid_mask,
                batch.video_ids,
            )
            shuffled = build_frequency_matched_action_curves(
                shuffled_embeddings,
                baseline.periods,
                batch.valid_mask,
                timeline_lengths=batch.lengths,
            )
            zero_embeddings = model.encoder(
                torch.zeros_like(batch.poses),
                batch.valid_mask,
            )
            zero_readout = estimate_embedding_action_curves(
                zero_embeddings,
                minimum_period=config.period.minimum,
                maximum_period=config.period.maximum,
                valid_mask=batch.valid_mask,
                timeline_lengths=batch.lengths,
            )
            baseline_harmonic.extend(
                _finite_float(value) for value in baseline.harmonic_energy_fractions
            )
            shuffled_harmonic.extend(
                _finite_float(value) for value in shuffled.harmonic_energy_fractions
            )
            zero_confidences.extend(
                _finite_float(value) for value in zero_readout.period_confidences
            )
            zero_available.extend(bool(value) for value in zero_readout.available)
            zero_curve_stds.extend(
                _finite_float(value)
                for value in zero_readout.curve_standard_deviations
            )

    baseline_summary = _summary(baseline_harmonic)
    shuffled_summary = _summary(shuffled_harmonic)
    baseline_median = _median(baseline_summary)
    shuffled_median = _median(shuffled_summary)
    median_ratio = (
        None
        if baseline_median is None or baseline_median <= 1e-12 or shuffled_median is None
        else shuffled_median / baseline_median
    )
    return {
        "time_shuffle": {
            "algorithm": (
                "SHA256-seeded permutation of valid embedding rows within each video; "
                "invalid dense slots and the baseline ACF period remain fixed"
            ),
            "baseline_harmonic_energy_fraction": baseline_summary,
            "shuffled_harmonic_energy_fraction": shuffled_summary,
            "shuffled_to_baseline_median_ratio": median_ratio,
        },
        "zero_pose": {
            "algorithm": (
                "replace every pose coordinate by zero while retaining the exact "
                "dense valid mask, then rerun encoder, ACF, and projection"
            ),
            "period_confidence": _summary(zero_confidences),
            "positive_period_confidence_share": float(
                np.mean(np.asarray(zero_confidences) > 0.0)
            ),
            "curve_available_share": float(np.mean(zero_available)),
            "action_curve_std": _summary(zero_curve_stds),
        },
    }


def _median(summary: Mapping[str, Any]) -> float | None:
    value = summary.get("median")
    return float(value) if isinstance(value, int | float) else None


def _gate_decision(
    *,
    period_distribution: Mapping[str, Any],
    readout_distribution: Mapping[str, Any],
    count_distribution: Mapping[str, Any],
    time_scale: Mapping[str, Any],
    corruption_controls: Mapping[str, Any],
) -> dict[str, Any]:
    period_scale = time_scale["period"]
    count_scale = time_scale["count"]
    criteria = {
        "upstream_period_boundary_share": _v14._criterion(
            float(period_distribution["boundary_share"]),
            operator="<",
            threshold=_THRESHOLDS[
                "upstream_period_boundary_share_maximum_exclusive"
            ],
        ),
        "upstream_period_mode_share": _v14._criterion(
            float(period_distribution["mode_share"]),
            operator="<",
            threshold=_THRESHOLDS[
                "upstream_period_mode_share_maximum_exclusive"
            ],
        ),
        "upstream_period_time_scale_eligible_fraction": _v14._criterion(
            float(period_scale["eligible_comparison_fraction"]),
            operator=">=",
            threshold=_THRESHOLDS[
                "upstream_period_time_scale_eligible_fraction_minimum"
            ],
        ),
        "upstream_period_time_scale_median_relative_error": _v14._criterion(
            _median(period_scale["relative_error"]),
            operator="<=",
            threshold=_THRESHOLDS[
                "upstream_period_time_scale_median_relative_error_maximum"
            ],
        ),
        "action_curve_std_median": _v14._criterion(
            _median(readout_distribution["action_curve_std"]),
            operator=">=",
            threshold=_THRESHOLDS["action_curve_std_median_minimum"],
        ),
        "algorithm1_majority_share": _v14._criterion(
            float(readout_distribution["algorithm1_majority_share"]),
            operator=">=",
            threshold=_THRESHOLDS["algorithm1_majority_share_minimum"],
        ),
        "count_time_scale_exact_fraction": _v14._criterion(
            float(count_scale["exact_fraction"]),
            operator=">=",
            threshold=_THRESHOLDS["count_time_scale_exact_fraction_minimum"],
        ),
        "count_time_scale_off_by_one_fraction": _v14._criterion(
            float(count_scale["within_one_fraction"]),
            operator=">=",
            threshold=_THRESHOLDS[
                "count_time_scale_off_by_one_fraction_minimum"
            ],
        ),
        "selected_count_fft_gap_median": _v14._criterion(
            _median(
                readout_distribution[
                    "selected_count_fft_reference_absolute_gap"
                ]
            ),
            operator="<=",
            threshold=_THRESHOLDS["selected_count_fft_gap_median_maximum"],
        ),
        "count_zero_share": _v14._criterion(
            float(count_distribution["zero_share"]),
            operator="<",
            threshold=_THRESHOLDS["count_zero_share_maximum_exclusive"],
        ),
        "count_mode_share": _v14._criterion(
            float(count_distribution["mode_share"]),
            operator="<",
            threshold=_THRESHOLDS["count_mode_share_maximum_exclusive"],
        ),
        "time_shuffle_harmonic_median_ratio": _v14._criterion(
            corruption_controls["time_shuffle"][
                "shuffled_to_baseline_median_ratio"
            ],
            operator="<=",
            threshold=_THRESHOLDS[
                "time_shuffle_harmonic_median_ratio_maximum"
            ],
        ),
        "zero_pose_positive_period_confidence_share": _v14._criterion(
            float(
                corruption_controls["zero_pose"][
                    "positive_period_confidence_share"
                ]
            ),
            operator="<=",
            threshold=_THRESHOLDS[
                "zero_pose_positive_period_confidence_share_maximum"
            ],
        ),
        "zero_pose_curve_available_share": _v14._criterion(
            float(corruption_controls["zero_pose"]["curve_available_share"]),
            operator="<=",
            threshold=_THRESHOLDS["zero_pose_curve_available_share_maximum"],
        ),
    }
    passed = all(bool(item["pass"]) for item in criteria.values())
    return {
        "thresholds_frozen_before_first_candidate_run": True,
        "criteria": criteria,
        "all_fourteen_criteria_pass": passed,
        "overall_pass": passed,
        "isolated_dev84_prediction_authorized": passed,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }


def _validate_candidate_config(
    upstream_config: PAMSConfig,
    candidate_config: PAMSConfig,
    *,
    candidate_config_sha256: str,
) -> None:
    if _EXPECTED_CANDIDATE_CONFIG_SHA256.startswith("__"):
        raise RuntimeError("candidate config identity has not been frozen")
    if candidate_config_sha256 != _EXPECTED_CANDIDATE_CONFIG_SHA256:
        raise ValueError("train337 gate requires the exact candidate config bytes")
    validate_embedding_action_curve_compatibility(
        upstream_config,
        candidate_config,
    )
    if (
        candidate_config.model.position_encoding_mode != "none"
        or candidate_config.consensus.expert_mode != "multi"
        or candidate_config.data.frames != 256
    ):
        raise ValueError("candidate config violates the frozen v16 readout scope")


def _input_identities(paths: Mapping[str, Path]) -> dict[str, tuple[str, int]]:
    return {name: _stable_file_sha256(path) for name, path in paths.items()}


def run_train337_gate(
    encoder_checkpoint_path: str | Path,
    encoder_progress_path: str | Path,
    upstream_config_path: str | Path,
    candidate_config_path: str | Path,
    pose_cache_dir: str | Path,
    source_receipt_path: str | Path,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate the exact frozen v16 encoder and train337 poses only."""

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
        "upstream_config": Path(upstream_config_path),
        "candidate_config": Path(candidate_config_path),
        "source_export_receipt": Path(source_receipt_path),
        "train337_gate_runner": Path(__file__),
        "embedding_curve_module": Path(
            estimate_embedding_action_curves.__code__.co_filename
        ),
        "v16_terminal_gate_dependency": Path(_v16.__file__),
        "v14_counterfactual_dependency": Path(_v14.__file__),
    }
    identities = _input_identities(paths)
    # Fail closed on any different encoder before deserializing it.
    _v16._validate_exact_terminal_artifact_identities(identities)
    _v14._validate_source_receipt_argument(
        paths["source_export_receipt"],
        receipt_sha256=identities["source_export_receipt"][0],
    )
    gate_code_source_git_sha = _v14._gate_code_source_revision()
    algorithm_source_git_sha = clean_git_revision(Path.cwd())
    if gate_code_source_git_sha != algorithm_source_git_sha:
        raise RuntimeError("gate and embedding readout source revisions differ")

    upstream_config = load_config(paths["upstream_config"])
    candidate_config = load_config(paths["candidate_config"])
    _v16._validate_exact_v16_config(
        upstream_config,
        config_sha256=identities["upstream_config"][0],
    )
    _validate_candidate_config(
        upstream_config,
        candidate_config,
        candidate_config_sha256=identities["candidate_config"][0],
    )
    stage, provenance = _peek_checkpoint(
        paths["encoder_checkpoint"], upstream_config
    )
    if stage != "encoder":
        raise ValueError("embedding action-curve gate requires an encoder checkpoint")
    if len(provenance.training_video_ids) != _EXPECTED_TRAINING_VIDEOS:
        raise ValueError("embedding action-curve gate requires exactly train337")
    validate_terminal_checkpoint(
        paths["encoder_checkpoint"],
        upstream_config,
        expected_stage="encoder",
        expected_provenance=provenance,
        progress_path=paths["encoder_progress"],
    )
    schedule = _v16._terminal_schedule_metadata(
        paths["encoder_checkpoint"],
        config=upstream_config,
    )
    resolved_device = _device(device)
    sequences, selected_receipts, full_receipts, selected_ids = _load_training_poses(
        pose_cache_dir=Path(pose_cache_dir),
        provenance=provenance,
        config=upstream_config,
        sample_size=0,
        seed=upstream_config.seed,
    )
    if len(sequences) != _EXPECTED_TRAINING_VIDEOS:
        raise RuntimeError("checkpoint-bound loader did not return train337")
    if any(sequence.num_frames != upstream_config.data.frames for sequence in sequences):
        raise ValueError("every v16 training pose must contain exactly 256 frames")
    selected_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=upstream_config.pose_fingerprint,
        entries=selected_receipts,
    )
    full_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=upstream_config.pose_fingerprint,
        entries=full_receipts,
    )
    if selected_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("full train337 selection and pose snapshots differ")
    if full_snapshot.fingerprint != provenance.pose_cache_set_sha256:
        raise ValueError("train337 pose-cache set differs from checkpoint provenance")

    model = load_model_checkpoint(
        paths["encoder_checkpoint"],
        upstream_config,
        device=resolved_device,
        expected_stage="encoder",
        expected_provenance=provenance,
    ).eval()
    model_state_before = _v16._epoch11._model_state_sha256(model)
    baseline = _encode_candidates(
        model,
        sequences,
        config=candidate_config,
        device=resolved_device,
        batch_size=batch_size,
    )
    period_distribution = _period_distribution(
        baseline,
        config=candidate_config,
    )
    readout_distribution = _readout_distribution(baseline)
    count_distribution = _count_distribution(baseline)
    time_scale = _time_scale_consistency(
        model,
        sequences,
        baseline,
        config=candidate_config,
        device=resolved_device,
        batch_size=batch_size,
    )
    corruption_controls = _corruption_controls(
        model,
        sequences,
        config=candidate_config,
        device=resolved_device,
        batch_size=batch_size,
    )
    decision = _gate_decision(
        period_distribution=period_distribution,
        readout_distribution=readout_distribution,
        count_distribution=count_distribution,
        time_scale=time_scale,
        corruption_controls=corruption_controls,
    )
    model_state_after = _v16._epoch11._model_state_sha256(model)
    if model_state_after != model_state_before:
        raise RuntimeError("model state changed during read-only gate")

    _, _, final_receipts, final_ids = _load_training_poses(
        pose_cache_dir=Path(pose_cache_dir),
        provenance=provenance,
        config=upstream_config,
        sample_size=2,
        seed=upstream_config.seed,
    )
    final_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=upstream_config.pose_fingerprint,
        entries=final_receipts,
    )
    if final_ids != selected_ids[:2] or final_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("train337 pose cache changed during gate")
    _v14._require_unchanged(paths, identities)
    if clean_git_revision(Path.cwd()) != algorithm_source_git_sha:
        raise RuntimeError("embedding readout source changed during gate")

    hardware = hardware_fingerprint()
    runtime = _v14._runtime_provenance(
        runner_sha256=identities["train337_gate_runner"][0],
        device=resolved_device,
    )
    if runtime["container"] is not None:
        runtime_revision = runtime["container"]["PAMS_CONTAINER_SOURCE_REVISION"]
        if runtime_revision != algorithm_source_git_sha:
            raise RuntimeError("candidate container source differs from clean source")
    runtime["algorithm_source_git_sha"] = algorithm_source_git_sha
    runtime["gate_code_source_git_sha"] = gate_code_source_git_sha
    code_names = (
        "train337_gate_runner",
        "embedding_curve_module",
        "v16_terminal_gate_dependency",
        "v14_counterfactual_dependency",
    )
    code_files_sha256 = {name: identities[name][0] for name in code_names}
    payload = {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "status": (
            "isolated_dev84_prediction_authorized"
            if decision["overall_pass"]
            else "scientific_rejection"
        ),
        "classification": _CLASSIFICATION,
        "disclosed_by_pams_authors": False,
        "eligible_for_paper_table": False,
        "protocol": candidate_config.protocol,
        "seed": candidate_config.seed,
        "label_firewall": {
            "accepted_scientific_inputs": [
                "exact_terminal_v16_encoder_checkpoint_and_progress",
                "exact_original_v16_config",
                "single_field_embedding_readout_candidate_config",
                "checkpoint_bound_train337_pose_cache",
                "exact_checkpoint_source_export_receipt",
            ],
            "dataset_manifest_argument_supported": False,
            "development_identity_media_pose_or_target_argument_supported": False,
            "sealed_test_identity_media_pose_or_target_argument_supported": False,
            "action_class_argument_supported": False,
            "repetition_count_argument_supported": False,
            "external_label_fields_accessed": [],
            "training_interface_supported": False,
        },
        "inputs": {
            **{f"{name}_sha256": value[0] for name, value in identities.items()},
            **{f"{name}_bytes": value[1] for name, value in identities.items()},
            "upstream_config_fingerprint": upstream_config.fingerprint,
            "candidate_config_fingerprint": candidate_config.fingerprint,
            "pose_fingerprint": candidate_config.pose_fingerprint,
            "encoder_provenance": provenance.to_dict(),
            "checkpoint_schedule": schedule,
            "train337_video_total": len(selected_ids),
            "train337_video_ids_sha256": _identifier_commitment(selected_ids),
            "train337_pose_cache_set_sha256": full_snapshot.fingerprint,
            "checkpoint_algorithm_source_git_sha": provenance.source_git_sha,
            "candidate_algorithm_source_git_sha": algorithm_source_git_sha,
            "gate_code_source_git_sha": gate_code_source_git_sha,
            "code_files_sha256": code_files_sha256,
            "code_files_sha256_commitment": sha256_json(code_files_sha256),
            "read_only_post_run_identity_verified": True,
        },
        "algorithm": {
            "period_estimator": "embedding_velocity_full_vector_acf",
            "action_curve": (
                "dense-time centered sine/cosine coefficients, deterministic "
                "rank-two principal embedding direction, full trajectory projection"
            ),
            "invalid_frames": "excluded from statistics and exact zero; holes not compacted",
            "counter": "unchanged MultiExpertCounter majority-first then FFT-nearest",
            "final_count_is_expert_count": True,
            "reference_count_timebase": "original dense timeline length",
        },
        "thresholds": dict(_THRESHOLDS),
        "period_distribution": period_distribution,
        "readout_distribution": readout_distribution,
        "count_distribution": count_distribution,
        "time_scale_consistency": time_scale,
        "corruption_controls": corruption_controls,
        "samples": [
            {
                "video_id": sample.video_id,
                "period_frames": sample.period_frames,
                "period_confidence": sample.period_confidence,
                "action_curve_std": sample.action_curve_std,
                "raw_action_curve_std": sample.raw_action_curve_std,
                "harmonic_energy_fraction": sample.harmonic_energy_fraction,
                "curve_available": sample.curve_available,
                "count": sample.count,
                "reference_count": sample.reference_count,
                "expert_counts": list(sample.expert_counts),
                "selected_expert": sample.selected_expert,
                "selection_rule": sample.selection_rule,
            }
            for sample in baseline
        ],
        "gate": decision,
        "scientific_caveats": [
            "The embedding action curve is independently inferred, not author-disclosed.",
            "Passing self-consistency does not establish repetition-count accuracy.",
            "No result from this gate can authorize dev scoring or test105 access.",
        ],
        "hardware": hardware,
        "hardware_sha256": sha256_json(hardware),
        "runtime": runtime,
        "runtime_sha256": sha256_json(runtime),
        "read_only_verification": {
            "all_file_inputs_unchanged": True,
            "train337_pose_cache_set_sha256_unchanged": True,
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after": model_state_after,
            "model_or_optimizer_state_updated": False,
            "training_steps_executed": 0,
            "pose_cache_write_operations": 0,
        },
    }
    del model
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
    if output.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite gate artifact or receipt")
    artifact = _v14._encoded_json(payload)
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    _v14._write_new_regular_file(output, artifact)
    receipt = {
        "schema_version": 1,
        "artifact_type": _RECEIPT_TYPE,
        "artifact_locator": output.name,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": len(artifact),
        "artifact_status": payload["status"],
        "overall_pass": payload["gate"]["overall_pass"],
        "isolated_dev84_prediction_authorized": payload["gate"][
            "isolated_dev84_prediction_authorized"
        ],
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
        "encoder_checkpoint_sha256": payload["inputs"][
            "encoder_checkpoint_sha256"
        ],
        "encoder_progress_sha256": payload["inputs"]["encoder_progress_sha256"],
        "upstream_config_sha256": payload["inputs"]["upstream_config_sha256"],
        "candidate_config_sha256": payload["inputs"]["candidate_config_sha256"],
        "source_export_receipt_sha256": payload["inputs"][
            "source_export_receipt_sha256"
        ],
        "train337_pose_cache_set_sha256": payload["inputs"][
            "train337_pose_cache_set_sha256"
        ],
        "candidate_algorithm_source_git_sha": payload["inputs"][
            "candidate_algorithm_source_git_sha"
        ],
        "code_files_sha256_commitment": payload["inputs"][
            "code_files_sha256_commitment"
        ],
        "hardware_sha256": payload["hardware_sha256"],
        "runtime_sha256": payload["runtime_sha256"],
    }
    _v14._write_new_regular_file(receipt_path, _v14._encoded_json(receipt))
    return receipt_path, artifact_sha256


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-progress", type=Path, required=True)
    parser.add_argument("--upstream-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--pose-cache-dir", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    if arguments.output.exists() or _receipt_path(arguments.output).exists():
        raise FileExistsError("gate output and receipt destinations must both be new")
    payload = run_train337_gate(
        arguments.encoder_checkpoint,
        arguments.encoder_progress,
        arguments.upstream_config,
        arguments.candidate_config,
        arguments.pose_cache_dir,
        arguments.source_receipt,
        device=arguments.device,
        batch_size=arguments.batch_size,
    )
    receipt_path, artifact_sha256 = _write_artifact_and_receipt(
        arguments.output,
        payload,
    )
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "output": str(arguments.output),
                "receipt": str(receipt_path),
                "overall_pass": payload["gate"]["overall_pass"],
                "isolated_dev84_prediction_authorized": payload["gate"][
                    "isolated_dev84_prediction_authorized"
                ],
                "dev84_scoring_authorized": False,
                "test105_evaluation_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0 if payload["gate"]["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
