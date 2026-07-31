"""Run the frozen train337-only v14 dual-path counterfactual diagnostic.

This read-only diagnostic asks one narrow causal question about the terminal
PAMS-v14 encoder: does the vector-velocity ACF collapse appear only after
absolute positional encoding and Transformer processing, or is it already
present in the pose-content projection immediately before positional
encoding?

Only the exact v14 config, its terminal encoder checkpoint and progress log,
the checkpoint-bound train337 pose cache, and the exact source-export receipt
are accepted.  There is no interface for development/test data, action
classes, repetition counts, targets, training, or model mutation.

The checkpoint algorithm source and this later diagnostic source are bound
separately.  ``PAMS_CONTAINER_SOURCE_REVISION`` identifies the exact source
export that produced the checkpoint, while
``PAMS_GATE_CODE_SOURCE_REVISION`` identifies this externally mounted runner.

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
from collections import Counter
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
from pams.period import (
    estimate_period_from_embedding_velocity_vectors,
    estimate_period_from_projected_pose,
)
from pams.reproducibility import clean_git_revision, hardware_fingerprint, sha256_json
from pams.training import (
    collate_pose_sequences,
    load_model_checkpoint,
    validate_terminal_checkpoint,
)
from pams.types import PoseSequence

try:
    from scripts.server import run_pams_v14_final_train_gate as _final_gate
    from scripts.server import run_pams_v14_predev_gate as _epoch11_gate
except ModuleNotFoundError:
    # Formal runs may mount the three audited gate files in /gate-code while
    # /workspace remains the exact checkpoint-producing source export.
    import run_pams_v14_final_train_gate as _final_gate  # type: ignore[no-redef]
    import run_pams_v14_predev_gate as _epoch11_gate  # type: ignore[no-redef]

_criterion = _epoch11_gate._criterion
_encoded_json = _epoch11_gate._encoded_json
_resample_for_time_scale = _epoch11_gate._resample_for_time_scale
_runtime_provenance = _epoch11_gate._runtime_provenance
_validate_v14_config = _epoch11_gate._validate_v14_config
_write_new_regular_file = _epoch11_gate._write_new_regular_file
_gate_code_source_revision = _final_gate._gate_code_source_revision
_require_unchanged = _final_gate._require_unchanged
_validate_runtime_binding = _final_gate._validate_runtime_binding

_ARTIFACT_TYPE = "pams_v14_terminal_encoder_dual_path_counterfactual"
_RECEIPT_TYPE = "pams_v14_terminal_encoder_dual_path_counterfactual_receipt"
_CLASSIFICATION = "inferred target-free read-only counterfactual diagnostic"
_EXPECTED_TRAINING_VIDEOS = 337
_EXPECTED_CONFIG_SHA256 = "3b11c331fd7f9adb1fcb21981424ef2366e7ae4df226e2d100acb2d7b295e39b"
_EXPECTED_CONFIG_FINGERPRINT = "42baeb7b4ff53ff519776ad4b8a2bf44183675810e7ac89f7e3d9918d08c4d45"
_EXPECTED_POSE_FINGERPRINT = "3dd0388320095796f42aa82d904073b6478b073ed351394ef8d03c30798a0116"
_TIME_SCALE_FACTORS = (0.50, 0.75)
_THRESHOLDS: dict[str, float] = {
    "projected_boundary_share_maximum_exclusive": 0.25,
    "projected_mode_share_maximum_exclusive": 0.25,
    "projected_time_scale_eligible_fraction_minimum": 0.50,
    "projected_time_scale_median_relative_error_maximum": 0.15,
    "post_pe_boundary_share_maximum_exclusive": 0.25,
    "post_pe_mode_share_maximum_exclusive": 0.25,
    "post_pe_time_scale_eligible_fraction_minimum": 0.50,
    "post_pe_time_scale_median_relative_error_maximum": 0.15,
    "cross_path_eligible_fraction_minimum": 0.50,
    "cross_path_median_relative_error_maximum": 0.25,
}


@dataclass(frozen=True, slots=True)
class _PeriodSample:
    """One global, label-free period estimate from one feature path."""

    video_id: str
    period_frames: float
    confidence: float


def _finite_float(value: Tensor | float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError("dual-path diagnostic produced a non-finite value")
    return result


def _estimate_dual_features(
    embeddings: Tensor,
    projected: Tensor,
    valid_mask: Tensor,
    *,
    config: PAMSConfig,
) -> dict[str, tuple[Tensor, Tensor]]:
    """Route each feature path through its exact, independently named estimator."""

    return {
        "post_pe": estimate_period_from_embedding_velocity_vectors(
            embeddings,
            minimum=config.period.minimum,
            maximum=config.period.maximum,
            valid_mask=valid_mask,
        ),
        "projected_pre_pe": estimate_period_from_projected_pose(
            projected,
            minimum=config.period.minimum,
            maximum=config.period.maximum,
            valid_mask=valid_mask,
        ),
    }


def _encode_dual_paths(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, tuple[_PeriodSample, ...]]:
    """Evaluate post-PE embeddings and the exact pre-PE projection together."""

    output: dict[str, list[_PeriodSample]] = {
        "post_pe": [],
        "projected_pre_pe": [],
    }
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            embeddings, projected = model.encoder.forward_with_pre_pe(
                batch.poses,
                batch.valid_mask,
            )
            estimates = _estimate_dual_features(
                embeddings,
                projected,
                batch.valid_mask,
                config=config,
            )
            for path_name, (periods, confidences) in estimates.items():
                output[path_name].extend(
                    _PeriodSample(
                        video_id=video_id,
                        period_frames=_finite_float(period),
                        confidence=_finite_float(confidence),
                    )
                    for video_id, period, confidence in zip(
                        batch.video_ids,
                        periods,
                        confidences,
                        strict=True,
                    )
                )
    return {name: tuple(rows) for name, rows in output.items()}


def _stable_period_key(period: float) -> str:
    return format(float(period), ".9g")


def _distribution(
    samples: Sequence[_PeriodSample],
    *,
    minimum: int,
    maximum: int,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("period distribution requires at least one sample")
    periods = np.asarray([row.period_frames for row in samples], dtype=np.float64)
    confidences = np.asarray([row.confidence for row in samples], dtype=np.float64)
    if not np.isfinite(periods).all() or not np.isfinite(confidences).all():
        raise RuntimeError("period distribution contains non-finite values")
    boundary = (periods <= float(minimum) + 1e-6) | (periods >= float(maximum) - 1e-6)
    frequencies = Counter(_stable_period_key(value) for value in periods)
    mode_key, mode_total = min(
        frequencies.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return {
        "record_total": len(samples),
        "boundary_share": float(np.mean(boundary)),
        "minimum_period_share": float(np.mean(periods <= float(minimum) + 1e-6)),
        "maximum_period_share": float(np.mean(periods >= float(maximum) - 1e-6)),
        "mode_period_frames": float(mode_key),
        "mode_frequency": mode_total,
        "mode_share": mode_total / len(samples),
        "unique_period_total": len(frequencies),
        "zero_confidence_share": float(np.mean(confidences <= 0.0)),
        "period_frames": _summary(periods),
        "confidence": _summary(confidences),
        "period_histogram": dict(sorted(frequencies.items())),
        "period_histogram_key_format": "finite_float_9_significant_digits",
    }


def _time_scale_consistency(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    baseline: Mapping[str, Sequence[_PeriodSample]],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    expected_ids = [sequence.video_id for sequence in sequences]
    for path_name, rows in baseline.items():
        if [row.video_id for row in rows] != expected_ids:
            raise ValueError(f"{path_name} baseline order differs from train337")
    accumulated: dict[str, dict[str, list[Any]]] = {
        path_name: {"errors": [], "rows": []} for path_name in baseline
    }
    for factor in _TIME_SCALE_FACTORS:
        scaled_sequences = tuple(
            _resample_for_time_scale(sequence, factor) for sequence in sequences
        )
        scaled = _encode_dual_paths(
            model,
            scaled_sequences,
            config=config,
            device=device,
            batch_size=batch_size,
        )
        for path_name, source_rows in baseline.items():
            scaled_rows = scaled[path_name]
            for source, candidate in zip(source_rows, scaled_rows, strict=True):
                expected = source.period_frames * factor
                eligible = (
                    source.confidence > 0.0
                    and candidate.confidence > 0.0
                    and config.period.minimum <= expected <= config.period.maximum
                )
                relative_error = (
                    abs(candidate.period_frames - expected) / expected if eligible else None
                )
                if relative_error is not None:
                    if not math.isfinite(relative_error):
                        raise RuntimeError("time-scale error is non-finite")
                    accumulated[path_name]["errors"].append(relative_error)
                accumulated[path_name]["rows"].append(
                    {
                        "video_id": source.video_id,
                        "time_scale_factor": factor,
                        "baseline_period_frames": source.period_frames,
                        "baseline_confidence": source.confidence,
                        "expected_scaled_period_frames": expected,
                        "scaled_period_frames": candidate.period_frames,
                        "scaled_confidence": candidate.confidence,
                        "eligible": eligible,
                        "relative_error": relative_error,
                    }
                )
    result: dict[str, dict[str, Any]] = {}
    for path_name, values in accumulated.items():
        rows = values["rows"]
        errors = values["errors"]
        result[path_name] = {
            "algorithm": (
                "Uniformly resample every checkpoint-bound train337 pose to "
                "round(T*factor) frames for factors 0.5 and 0.75, rerun the "
                "same path and vector-velocity ACF estimator, and compare "
                "P_scaled with factor*P. Both estimates require positive "
                "confidence and the expected period must remain in 4--128."
            ),
            "factors": list(_TIME_SCALE_FACTORS),
            "candidate_comparison_total": len(rows),
            "eligible_comparison_total": len(errors),
            "eligible_comparison_fraction": (len(errors) / len(rows) if rows else 0.0),
            "relative_error": _summary(errors),
            "rows": rows,
        }
    return result


def _cross_path_comparison(
    projected: Sequence[_PeriodSample],
    post_pe: Sequence[_PeriodSample],
) -> dict[str, Any]:
    """Compare common positive-confidence estimates using projected as reference."""

    if [row.video_id for row in projected] != [row.video_id for row in post_pe]:
        raise ValueError("cross-path rows do not share one train337 order")
    errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for teacher, candidate in zip(projected, post_pe, strict=True):
        eligible = teacher.confidence > 0.0 and candidate.confidence > 0.0
        error = (
            abs(candidate.period_frames - teacher.period_frames) / teacher.period_frames
            if eligible
            else None
        )
        if error is not None:
            if not math.isfinite(error):
                raise RuntimeError("cross-path relative error is non-finite")
            errors.append(error)
        rows.append(
            {
                "video_id": teacher.video_id,
                "projected_period_frames": teacher.period_frames,
                "projected_confidence": teacher.confidence,
                "post_pe_period_frames": candidate.period_frames,
                "post_pe_confidence": candidate.confidence,
                "eligible": eligible,
                "relative_error": error,
            }
        )
    return {
        "algorithm": (
            "A train337 video is common-valid when both paths have positive "
            "vector-ACF confidence. Relative error is "
            "abs(P_post_pe-P_projected)/P_projected; the projected pre-PE "
            "pose-content path is the counterfactual reference."
        ),
        "candidate_comparison_total": len(rows),
        "eligible_comparison_total": len(errors),
        "eligible_comparison_fraction": len(errors) / len(rows) if rows else 0.0,
        "relative_error": _summary(errors),
        "rows": rows,
    }


def _eligible_fraction(payload: Mapping[str, Any]) -> float:
    candidates = int(payload["candidate_comparison_total"])
    eligible = int(payload["eligible_comparison_total"])
    return eligible / candidates if candidates else 0.0


def _median(payload: Mapping[str, Any]) -> float | None:
    value = payload.get("median")
    return float(value) if isinstance(value, int | float) else None


def _gate_decision(
    *,
    projected_distribution: Mapping[str, Any],
    projected_time_scale: Mapping[str, Any],
    post_pe_distribution: Mapping[str, Any],
    post_pe_time_scale: Mapping[str, Any],
    cross_path: Mapping[str, Any],
) -> dict[str, Any]:
    criteria = {
        "projected_boundary_share": _criterion(
            float(projected_distribution["boundary_share"]),
            operator="<",
            threshold=_THRESHOLDS["projected_boundary_share_maximum_exclusive"],
        ),
        "projected_mode_share": _criterion(
            float(projected_distribution["mode_share"]),
            operator="<",
            threshold=_THRESHOLDS["projected_mode_share_maximum_exclusive"],
        ),
        "projected_time_scale_eligible_fraction": _criterion(
            _eligible_fraction(projected_time_scale),
            operator=">=",
            threshold=_THRESHOLDS["projected_time_scale_eligible_fraction_minimum"],
        ),
        "projected_time_scale_median_relative_error": _criterion(
            _median(projected_time_scale["relative_error"]),
            operator="<=",
            threshold=_THRESHOLDS["projected_time_scale_median_relative_error_maximum"],
        ),
        "post_pe_boundary_share": _criterion(
            float(post_pe_distribution["boundary_share"]),
            operator="<",
            threshold=_THRESHOLDS["post_pe_boundary_share_maximum_exclusive"],
        ),
        "post_pe_mode_share": _criterion(
            float(post_pe_distribution["mode_share"]),
            operator="<",
            threshold=_THRESHOLDS["post_pe_mode_share_maximum_exclusive"],
        ),
        "post_pe_time_scale_eligible_fraction": _criterion(
            _eligible_fraction(post_pe_time_scale),
            operator=">=",
            threshold=_THRESHOLDS["post_pe_time_scale_eligible_fraction_minimum"],
        ),
        "post_pe_time_scale_median_relative_error": _criterion(
            _median(post_pe_time_scale["relative_error"]),
            operator="<=",
            threshold=_THRESHOLDS["post_pe_time_scale_median_relative_error_maximum"],
        ),
        "cross_path_eligible_fraction": _criterion(
            _eligible_fraction(cross_path),
            operator=">=",
            threshold=_THRESHOLDS["cross_path_eligible_fraction_minimum"],
        ),
        "cross_path_median_relative_error": _criterion(
            _median(cross_path["relative_error"]),
            operator="<=",
            threshold=_THRESHOLDS["cross_path_median_relative_error_maximum"],
        ),
    }
    projected_names = {
        "projected_boundary_share",
        "projected_mode_share",
        "projected_time_scale_eligible_fraction",
        "projected_time_scale_median_relative_error",
    }
    v15_authorized = all(bool(criteria[name]["pass"]) for name in projected_names)
    return {
        "thresholds_frozen_before_terminal_checkpoint_evaluation": True,
        "criteria": criteria,
        "all_ten_diagnostic_criteria_pass": all(bool(item["pass"]) for item in criteria.values()),
        "v15_encoder_training_authorization_basis": sorted(projected_names),
        "v15_encoder_training_authorized": v15_authorized,
        "post_pe_and_cross_path_criteria_are_causal_diagnostics_only": True,
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }


def _validate_exact_v14_config(
    config: PAMSConfig,
    *,
    config_sha256: str,
) -> None:
    _validate_v14_config(config)
    actual = {
        "config_sha256": config_sha256,
        "config_fingerprint": config.fingerprint,
        "pose_fingerprint": config.pose_fingerprint,
    }
    expected = {
        "config_sha256": _EXPECTED_CONFIG_SHA256,
        "config_fingerprint": _EXPECTED_CONFIG_FINGERPRINT,
        "pose_fingerprint": _EXPECTED_POSE_FINGERPRINT,
    }
    if actual != expected:
        raise ValueError(
            "dual-path diagnostic requires the exact frozen v14 config: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )


def _validate_source_receipt_argument(
    source_receipt: Path,
    *,
    receipt_sha256: str,
) -> None:
    environment_path = os.environ.get("PAMS_SOURCE_EXPORT_RECEIPT", "").strip()
    environment_sha256 = os.environ.get(
        "PAMS_SOURCE_EXPORT_RECEIPT_SHA256",
        "",
    ).strip()
    if not environment_path or not environment_sha256:
        raise RuntimeError("source-export receipt environment is incomplete")
    if Path(environment_path).resolve(strict=True) != source_receipt.resolve(strict=True):
        raise RuntimeError("source-receipt argument differs from runtime binding")
    if receipt_sha256 != environment_sha256:
        raise RuntimeError("source-receipt bytes differ from runtime binding")


def _input_identities(paths: Mapping[str, Path]) -> dict[str, tuple[str, int]]:
    return {name: _stable_file_sha256(path) for name, path in paths.items()}


def run_counterfactual(
    encoder_checkpoint_path: str | Path,
    encoder_progress_path: str | Path,
    config_path: str | Path,
    pose_cache_dir: str | Path,
    source_receipt_path: str | Path,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate the two frozen v14 encoder paths without labels or mutation."""

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
        "config": Path(config_path),
        "source_export_receipt": Path(source_receipt_path),
        "counterfactual_runner": Path(__file__),
        "final_gate_dependency": Path(_final_gate.__file__),
        "epoch11_gate_dependency": Path(_epoch11_gate.__file__),
    }
    identities = _input_identities(paths)
    _validate_source_receipt_argument(
        paths["source_export_receipt"],
        receipt_sha256=identities["source_export_receipt"][0],
    )
    gate_code_source_git_sha = _gate_code_source_revision()
    config = load_config(paths["config"])
    _validate_exact_v14_config(
        config,
        config_sha256=identities["config"][0],
    )
    stage, provenance = _peek_checkpoint(paths["encoder_checkpoint"], config)
    if stage != "encoder":
        raise ValueError("dual-path diagnostic requires an encoder checkpoint")
    if len(provenance.training_video_ids) != _EXPECTED_TRAINING_VIDEOS:
        raise ValueError("dual-path diagnostic requires exactly train337")
    if provenance.upstream_encoder_checkpoint_sha256 is not None:
        raise ValueError("encoder provenance unexpectedly names an upstream checkpoint")
    if provenance.container_image_id is None or provenance.container_environment_sha256 is None:
        raise ValueError("dual-path diagnostic requires formal container provenance")
    validate_terminal_checkpoint(
        paths["encoder_checkpoint"],
        config,
        expected_stage="encoder",
        expected_provenance=provenance,
        progress_path=paths["encoder_progress"],
    )
    algorithm_source_git_sha = clean_git_revision(Path.cwd())
    runtime_container = _validate_runtime_binding(
        provenance,
        source_git_sha=algorithm_source_git_sha,
    )
    resolved_device = _device(device)

    sequences, selected_receipts, full_receipts, selected_ids = _load_training_poses(
        pose_cache_dir=Path(pose_cache_dir),
        provenance=provenance,
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
    if full_snapshot.fingerprint != provenance.pose_cache_set_sha256:
        raise ValueError("train337 pose-cache set differs from checkpoint provenance")

    model = load_model_checkpoint(
        paths["encoder_checkpoint"],
        config,
        device=resolved_device,
        expected_stage="encoder",
        expected_provenance=provenance,
    ).eval()
    baseline = _encode_dual_paths(
        model,
        sequences,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    projected_distribution = _distribution(
        baseline["projected_pre_pe"],
        minimum=config.period.minimum,
        maximum=config.period.maximum,
    )
    post_pe_distribution = _distribution(
        baseline["post_pe"],
        minimum=config.period.minimum,
        maximum=config.period.maximum,
    )
    time_scale = _time_scale_consistency(
        model,
        sequences,
        baseline,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    cross_path = _cross_path_comparison(
        baseline["projected_pre_pe"],
        baseline["post_pe"],
    )
    decision = _gate_decision(
        projected_distribution=projected_distribution,
        projected_time_scale=time_scale["projected_pre_pe"],
        post_pe_distribution=post_pe_distribution,
        post_pe_time_scale=time_scale["post_pe"],
        cross_path=cross_path,
    )

    _, _, final_receipts, final_ids = _load_training_poses(
        pose_cache_dir=Path(pose_cache_dir),
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
        raise RuntimeError("train337 pose selection changed during diagnostic")
    if final_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("train337 pose cache changed during diagnostic")
    _require_unchanged(paths, identities)
    if clean_git_revision(Path.cwd()) != algorithm_source_git_sha:
        raise RuntimeError("algorithm source changed during diagnostic")

    hardware = hardware_fingerprint()
    runtime = _runtime_provenance(
        runner_sha256=identities["counterfactual_runner"][0],
        device=resolved_device,
    )
    expected_container = {
        "PAMS_CONTAINER_IMAGE_ID": runtime_container["image_id"],
        "PAMS_CONTAINER_ENVIRONMENT_SHA256": runtime_container["environment_sha256"],
        "PAMS_CONTAINER_SOURCE_REVISION": runtime_container["source_revision"],
    }
    if runtime["container"] != expected_container:
        raise RuntimeError("runtime provenance changed while it was recorded")
    runtime["algorithm_source_git_sha"] = algorithm_source_git_sha
    runtime["gate_code_source_git_sha"] = gate_code_source_git_sha
    runtime["source_identity_model"] = (
        "checkpoint algorithm source and external diagnostic source are independently bound"
    )
    code_names = (
        "counterfactual_runner",
        "final_gate_dependency",
        "epoch11_gate_dependency",
    )
    code_files_sha256 = {name: identities[name][0] for name in code_names}
    payload = {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "status": (
            "projected_teacher_passed"
            if decision["v15_encoder_training_authorized"]
            else "projected_teacher_failed"
        ),
        "classification": _CLASSIFICATION,
        "disclosed_by_pams_authors": False,
        "eligible_for_paper_table": False,
        "protocol": config.protocol,
        "seed": config.seed,
        "label_firewall": {
            "accepted_scientific_inputs": [
                "exact_v14_experiment_config",
                "terminal_encoder_checkpoint_and_progress",
                "checkpoint_bound_train337_pose_cache",
                "exact_checkpoint_algorithm_source_export_receipt",
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
            **{f"{name}_sha256": identity[0] for name, identity in identities.items()},
            **{f"{name}_bytes": identity[1] for name, identity in identities.items()},
            "config_fingerprint": config.fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
            "encoder_provenance": provenance.to_dict(),
            "train337_video_total": len(selected_ids),
            "train337_video_ids_sha256": _identifier_commitment(selected_ids),
            "train337_pose_cache_set_sha256": full_snapshot.fingerprint,
            "checkpoint_algorithm_source_git_sha": algorithm_source_git_sha,
            "gate_code_source_git_sha": gate_code_source_git_sha,
            "code_files_sha256": code_files_sha256,
            "code_files_sha256_commitment": sha256_json(code_files_sha256),
            "read_only_post_run_identity_verified": True,
        },
        "paths": {
            "projected_pose_velocity_vector_acf": {
                "feature_source": ("encoder.input_projection output before positional encoding"),
                "distribution": projected_distribution,
                "time_scale_consistency": time_scale["projected_pre_pe"],
            },
            "embedding_velocity_vector_acf": {
                "feature_source": (
                    "L2-normalized Transformer output after absolute positional encoding"
                ),
                "distribution": post_pe_distribution,
                "time_scale_consistency": time_scale["post_pe"],
            },
        },
        "cross_path_comparison": cross_path,
        "gate": decision,
        "scientific_caveats": [
            (
                "All ten criteria are independently inferred anti-collapse or "
                "self-consistency diagnostics, not author-disclosed validation."
            ),
            (
                "Only the four projected pre-PE teacher criteria can authorize "
                "a new v15 encoder-training experiment. The post-PE and "
                "cross-path criteria are causal baselines expected to expose "
                "the known v14 failure and cannot veto that authorization."
            ),
            (
                "Development and sealed-test access remain unauthorized "
                "regardless of every diagnostic result."
            ),
            (
                "Time-scale consistency and cross-path agreement do not "
                "establish ground-truth period or repetition-count accuracy."
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
    if output.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic artifact: {output}")
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic artifact receipt: {receipt_path}")
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
        "v15_encoder_training_authorized": payload["gate"]["v15_encoder_training_authorized"],
        "encoder_checkpoint_sha256": payload["inputs"]["encoder_checkpoint_sha256"],
        "encoder_progress_sha256": payload["inputs"]["encoder_progress_sha256"],
        "config_sha256": payload["inputs"]["config_sha256"],
        "source_export_receipt_sha256": payload["inputs"]["source_export_receipt_sha256"],
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pose-cache-dir", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    receipt_path = _receipt_path(arguments.output)
    if arguments.output.exists() or receipt_path.exists():
        raise FileExistsError("diagnostic output and receipt destinations must both be new")
    payload = run_counterfactual(
        arguments.encoder_checkpoint,
        arguments.encoder_progress,
        arguments.config,
        arguments.pose_cache_dir,
        arguments.source_receipt,
        device=arguments.device,
        batch_size=arguments.batch_size,
    )
    receipt_path, artifact_sha256 = _write_artifact_and_receipt(
        arguments.output,
        payload,
    )
    authorized = payload["gate"]["v15_encoder_training_authorized"]
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "output": str(arguments.output),
                "receipt": str(receipt_path),
                "v15_encoder_training_authorized": authorized,
            },
            sort_keys=True,
        )
    )
    return 0 if authorized else 3


if __name__ == "__main__":
    raise SystemExit(main())
