"""Run a read-only in-memory PE-off probe on the exact terminal PAMS-v15 encoder.

The probe asks whether the terminal encoder's post-Transformer period collapse
is caused by applying absolute sinusoidal positional encoding at inference.
It first evaluates the checkpoint normally, then changes only the in-memory
``position_encoding_mode`` attribute to ``"none"`` inside ``try/finally`` and
repeats the complete train337 dual-path and 0.5/0.75 time-scale diagnostics.

The exact v15 config, terminal checkpoint and progress log, checkpoint-bound
train337 pose cache, and checkpoint source-export receipt are the only
scientific inputs.  There is no label, development, test, optimizer, or
training interface.  Passing this probe cannot independently authorize a
NoAbsPE run: a separate current alias-identifiability probe must also pass.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from pams.config import PAMSConfig, load_config
from pams.data import PoseCacheSetSnapshot
from pams.diagnostics import (
    _device,
    _identifier_commitment,
    _load_training_poses,
    _peek_checkpoint,
    _stable_file_sha256,
)
from pams.model import PAMSModel
from pams.reproducibility import clean_git_revision, hardware_fingerprint, sha256_json
from pams.training import (
    collate_pose_sequences,
    load_model_checkpoint,
    validate_terminal_checkpoint,
)
from pams.types import PoseSequence

try:
    from scripts.server import run_pams_v15_terminal_dual_path_gate as _v15
except ModuleNotFoundError:
    # Formal runs may mount this runner beside its audited dependencies while
    # /workspace remains the exact checkpoint-producing source export.
    import run_pams_v15_terminal_dual_path_gate as _v15  # type: ignore[no-redef]

_v14 = _v15._v14

_ARTIFACT_TYPE = "pams_v15_terminal_encoder_in_memory_peoff_dual_path_probe"
_RECEIPT_TYPE = "pams_v15_terminal_encoder_in_memory_peoff_dual_path_probe_receipt"
_CLASSIFICATION = "inferred target-free read-only in-memory causal probe"
_EXPECTED_TRAINING_VIDEOS = 337
_EXPECTED_ENCODER_CHECKPOINT_SHA256 = (
    "aa1750154e0ba36d41188cc023bb382db83db9f5c69b1867f310cdc636a48e19"
)
_EXPECTED_ENCODER_PROGRESS_SHA256 = (
    "70226a5f01a3d9e8b688fde53736c397a135c242d8e15ce429a36615552029c4"
)
_MAX_PROJECTED_TENSOR_DRIFT = 1e-7
_TIME_SCALE_FACTORS = _v14._TIME_SCALE_FACTORS


@dataclass(frozen=True, slots=True)
class _ProjectedBatch:
    video_ids: tuple[str, ...]
    values: Tensor


@dataclass(frozen=True, slots=True)
class _VariantEncoding:
    paths: Mapping[str, tuple[Any, ...]]
    projected_batches: tuple[_ProjectedBatch, ...]
    projected_tensor_sha256: str
    projected_tensor_elements: int
    projected_tensor_max_abs_drift: float | None


@dataclass(frozen=True, slots=True)
class _ModeEncoding:
    original: _VariantEncoding
    scaled: Mapping[float, _VariantEncoding]


def _hash_part(hasher: Any, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, byteorder="big", signed=False))
    hasher.update(value)


def _tensor_bytes(tensor: Tensor) -> bytes:
    contiguous = tensor.detach().cpu().contiguous().reshape(-1)
    return contiguous.view(torch.uint8).numpy().tobytes()


def _module_state_sha256(module: nn.Module) -> str:
    """Hash tensor state without serializing mutable Python attributes."""

    hasher = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        _hash_part(hasher, name.encode("utf-8"))
        _hash_part(hasher, str(tensor.dtype).encode("ascii"))
        _hash_part(
            hasher,
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"),
        )
        _hash_part(hasher, _tensor_bytes(tensor))
    return hasher.hexdigest()


def _update_projected_hash(
    hasher: Any,
    *,
    video_ids: Sequence[str],
    projected: Tensor,
) -> None:
    _hash_part(
        hasher,
        json.dumps(list(video_ids), ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
    )
    _hash_part(hasher, str(projected.dtype).encode("ascii"))
    _hash_part(
        hasher,
        json.dumps(list(projected.shape), separators=(",", ":")).encode("ascii"),
    )
    _hash_part(hasher, _tensor_bytes(projected))


def _encode_variant(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
    projected_reference: Sequence[_ProjectedBatch] | None = None,
    retain_projected: bool,
) -> _VariantEncoding:
    """Encode one sequence variant and audit its complete projected tensor stream."""

    output: dict[str, list[Any]] = {"post_pe": [], "projected_pre_pe": []}
    captured: list[_ProjectedBatch] = []
    projected_hasher = hashlib.sha256()
    projected_elements = 0
    maximum_drift = 0.0 if projected_reference is not None else None
    model.eval()
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(sequences), batch_size)):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            embeddings, projected = model.encoder.forward_with_pre_pe(
                batch.poses,
                batch.valid_mask,
            )
            projected_cpu = projected.detach().cpu().contiguous()
            video_ids = tuple(batch.video_ids)
            _update_projected_hash(
                projected_hasher,
                video_ids=video_ids,
                projected=projected_cpu,
            )
            projected_elements += projected_cpu.numel()
            if projected_reference is not None:
                if batch_index >= len(projected_reference):
                    raise RuntimeError("PE-off projected stream has additional batches")
                reference = projected_reference[batch_index]
                if reference.video_ids != video_ids:
                    raise RuntimeError("PE-off projected stream changed video order")
                if (
                    reference.values.dtype != projected_cpu.dtype
                    or reference.values.shape != projected_cpu.shape
                ):
                    raise RuntimeError("PE-off projected tensor schema changed")
                drift = (
                    0.0
                    if torch.equal(reference.values, projected_cpu)
                    else float((reference.values - projected_cpu).abs().max())
                )
                if not math.isfinite(drift):
                    raise RuntimeError("PE-off projected tensor drift is non-finite")
                maximum_drift = max(float(maximum_drift), drift)
            if retain_projected:
                captured.append(_ProjectedBatch(video_ids=video_ids, values=projected_cpu))
            estimates = _v14._estimate_dual_features(
                embeddings,
                projected,
                batch.valid_mask,
                config=config,
            )
            for path_name, (periods, confidences) in estimates.items():
                output[path_name].extend(
                    _v14._PeriodSample(
                        video_id=video_id,
                        period_frames=_v14._finite_float(period),
                        confidence=_v14._finite_float(confidence),
                    )
                    for video_id, period, confidence in zip(
                        batch.video_ids,
                        periods,
                        confidences,
                        strict=True,
                    )
                )
    expected_batch_total = math.ceil(len(sequences) / batch_size)
    if projected_reference is not None and len(projected_reference) != expected_batch_total:
        raise RuntimeError("baseline projected stream has additional batches")
    return _VariantEncoding(
        paths={name: tuple(rows) for name, rows in output.items()},
        projected_batches=tuple(captured),
        projected_tensor_sha256=projected_hasher.hexdigest(),
        projected_tensor_elements=projected_elements,
        projected_tensor_max_abs_drift=maximum_drift,
    )


def _encode_mode(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
    reference: _ModeEncoding | None = None,
    retain_projected: bool,
) -> _ModeEncoding:
    original = _encode_variant(
        model,
        sequences,
        config=config,
        device=device,
        batch_size=batch_size,
        projected_reference=(
            reference.original.projected_batches if reference is not None else None
        ),
        retain_projected=retain_projected,
    )
    scaled: dict[float, _VariantEncoding] = {}
    for factor in _TIME_SCALE_FACTORS:
        scaled_sequences = tuple(
            _v14._resample_for_time_scale(sequence, factor) for sequence in sequences
        )
        scaled[factor] = _encode_variant(
            model,
            scaled_sequences,
            config=config,
            device=device,
            batch_size=batch_size,
            projected_reference=(
                reference.scaled[factor].projected_batches if reference is not None else None
            ),
            retain_projected=retain_projected,
        )
    return _ModeEncoding(original=original, scaled=scaled)


def _run_in_memory_modes(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> tuple[_ModeEncoding, _ModeEncoding, dict[str, Any]]:
    """Run baseline then PE-off while guaranteeing mode and tensor-state restoration."""

    original_mode = model.encoder.position_encoding_mode
    if original_mode != "sinusoidal":
        raise ValueError("exact v15 checkpoint must load in sinusoidal position mode")
    state_before = _module_state_sha256(model)
    baseline = _encode_mode(
        model,
        sequences,
        config=config,
        device=device,
        batch_size=batch_size,
        retain_projected=True,
    )
    state_after_baseline = _module_state_sha256(model)
    if state_after_baseline != state_before:
        raise RuntimeError("normal inference changed model tensor state")
    try:
        model.encoder.position_encoding_mode = "none"
        peoff = _encode_mode(
            model,
            sequences,
            config=config,
            device=device,
            batch_size=batch_size,
            reference=baseline,
            retain_projected=False,
        )
    finally:
        model.encoder.position_encoding_mode = original_mode
    restored_mode = model.encoder.position_encoding_mode
    state_after = _module_state_sha256(model)
    if restored_mode != original_mode:
        raise RuntimeError("position encoding mode was not restored")
    if state_after != state_before:
        raise RuntimeError("PE-off inference changed model tensor state")
    return (
        baseline,
        peoff,
        {
            "original_position_encoding_mode": original_mode,
            "temporary_position_encoding_mode": "none",
            "restored_position_encoding_mode": restored_mode,
            "mode_restored": True,
            "model_state_sha256_before": state_before,
            "model_state_sha256_after_baseline": state_after_baseline,
            "model_state_sha256_after_restore": state_after,
            "model_tensor_state_unchanged": True,
        },
    )


def _time_scale_summary(
    mode: _ModeEncoding,
    *,
    config: PAMSConfig,
    path_name: str,
) -> dict[str, Any]:
    baseline = mode.original.paths[path_name]
    errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for factor in _TIME_SCALE_FACTORS:
        scaled = mode.scaled[factor].paths[path_name]
        for source, candidate in zip(baseline, scaled, strict=True):
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
                errors.append(relative_error)
            rows.append(
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
    return {
        "algorithm": (
            "Uniformly resample every checkpoint-bound train337 pose to "
            "round(T*factor) frames for factors 0.5 and 0.75, rerun the same "
            "path and vector-velocity ACF estimator, and compare P_scaled with "
            "factor*P. Both estimates require positive confidence and the "
            "expected period must remain in 4--128."
        ),
        "factors": list(_TIME_SCALE_FACTORS),
        "candidate_comparison_total": len(rows),
        "eligible_comparison_total": len(errors),
        "eligible_comparison_fraction": len(errors) / len(rows) if rows else 0.0,
        "relative_error": _v14._summary(errors),
        "rows": rows,
    }


def _projected_identity_audit(
    baseline: _ModeEncoding,
    peoff: _ModeEncoding,
) -> dict[str, Any]:
    variants: list[tuple[str, _VariantEncoding, _VariantEncoding]] = [
        ("original", baseline.original, peoff.original),
        *[
            (f"time_scale_{format(factor, 'g')}", baseline.scaled[factor], peoff.scaled[factor])
            for factor in _TIME_SCALE_FACTORS
        ],
    ]
    reports: dict[str, Any] = {}
    overall_drift = 0.0
    for name, normal, candidate in variants:
        normal_rows = normal.paths["projected_pre_pe"]
        candidate_rows = candidate.paths["projected_pre_pe"]
        period_confidence_exact = all(
            (
                left.video_id == right.video_id
                and left.period_frames == right.period_frames
                and left.confidence == right.confidence
            )
            for left, right in zip(normal_rows, candidate_rows, strict=True)
        )
        drift = candidate.projected_tensor_max_abs_drift
        if drift is None:
            raise RuntimeError("PE-off projected tensor drift was not measured")
        tensor_digest_exact = normal.projected_tensor_sha256 == candidate.projected_tensor_sha256
        reports[name] = {
            "record_total": len(normal_rows),
            "projected_period_confidence_exact": period_confidence_exact,
            "normal_projected_tensor_sha256": normal.projected_tensor_sha256,
            "peoff_projected_tensor_sha256": candidate.projected_tensor_sha256,
            "projected_tensor_sha256_exact": tensor_digest_exact,
            "projected_tensor_elements": normal.projected_tensor_elements,
            "projected_tensor_max_abs_drift": drift,
            "projected_tensor_max_abs_drift_threshold": _MAX_PROJECTED_TENSOR_DRIFT,
            "projected_tensor_drift_within_threshold": (drift <= _MAX_PROJECTED_TENSOR_DRIFT),
        }
        if not period_confidence_exact or not tensor_digest_exact:
            raise RuntimeError("PE-off changed the projected period path")
        if drift > _MAX_PROJECTED_TENSOR_DRIFT:
            raise RuntimeError("PE-off projected tensor drift exceeded tolerance")
        overall_drift = max(overall_drift, drift)
    return {
        "all_variants_projected_period_confidence_exact": True,
        "all_variants_projected_tensor_sha256_exact": True,
        "overall_projected_tensor_max_abs_drift": overall_drift,
        "overall_projected_tensor_max_abs_drift_threshold": (_MAX_PROJECTED_TENSOR_DRIFT),
        "variants": reports,
    }


def _gate_decision(
    *,
    projected_distribution: Mapping[str, Any],
    projected_time_scale: Mapping[str, Any],
    peoff_post_distribution: Mapping[str, Any],
    peoff_post_time_scale: Mapping[str, Any],
    cross_path: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the existing ten thresholds without granting standalone authority."""

    frozen = _v15._gate_decision(
        projected_distribution=projected_distribution,
        projected_time_scale=projected_time_scale,
        post_pe_distribution=peoff_post_distribution,
        post_pe_time_scale=peoff_post_time_scale,
        cross_path=cross_path,
    )
    renamed = {
        name.replace("post_pe_", "peoff_post_pe_"): value
        for name, value in frozen["criteria"].items()
    }
    if len(renamed) != 10:
        raise RuntimeError("terminal v15 dependency no longer exposes ten criteria")
    passed = all(bool(value["pass"]) for value in renamed.values())
    return {
        "thresholds_reused_unchanged_from_v15_terminal_gate": True,
        "criteria": renamed,
        "peoff_gate_pass": passed,
        "all_ten_diagnostic_criteria_pass": passed,
        "standalone_noabs_training_authorized": False,
        "fresh_train337_epoch11_staged_noabs_candidate_requires": [
            "peoff_gate_pass",
            "separate_current_alias_probe_reports_alias_false",
            "separate_current_alias_probe_reports_teacher_identifiable",
        ],
        "separate_alias_probe_result_available_to_this_runner": False,
        "sshead_training_authorized": False,
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }


def _input_identities(paths: Mapping[str, Path]) -> dict[str, tuple[str, int]]:
    return {name: _stable_file_sha256(path) for name, path in paths.items()}


def _validate_exact_v15_artifact_identities(
    identities: Mapping[str, tuple[str, int]],
) -> None:
    actual = {
        "encoder_checkpoint_sha256": identities["encoder_checkpoint"][0],
        "encoder_progress_sha256": identities["encoder_progress"][0],
    }
    expected = {
        "encoder_checkpoint_sha256": _EXPECTED_ENCODER_CHECKPOINT_SHA256,
        "encoder_progress_sha256": _EXPECTED_ENCODER_PROGRESS_SHA256,
    }
    if actual != expected:
        raise ValueError(
            "PE-off probe requires the exact frozen terminal v15 artifacts: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )


def run_peoff_probe(
    encoder_checkpoint_path: str | Path,
    encoder_progress_path: str | Path,
    config_path: str | Path,
    pose_cache_dir: str | Path,
    source_receipt_path: str | Path,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate normal and in-memory PE-off paths on checkpoint-bound train337."""

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
        "v15_peoff_probe_runner": Path(__file__),
        "v15_terminal_gate_dependency": Path(_v15.__file__),
        "v14_counterfactual_dependency": Path(_v14.__file__),
        "final_gate_dependency": Path(_v14._final_gate.__file__),
        "epoch11_gate_dependency": Path(_v14._epoch11_gate.__file__),
    }
    identities = _input_identities(paths)
    _validate_exact_v15_artifact_identities(identities)
    _v14._validate_source_receipt_argument(
        paths["source_export_receipt"],
        receipt_sha256=identities["source_export_receipt"][0],
    )
    gate_code_source_git_sha = _v14._gate_code_source_revision()
    config = load_config(paths["config"])
    _v15._validate_exact_v15_config(
        config,
        config_sha256=identities["config"][0],
    )
    stage, provenance = _peek_checkpoint(paths["encoder_checkpoint"], config)
    if stage != "encoder":
        raise ValueError("v15 PE-off probe requires an encoder checkpoint")
    if len(provenance.training_video_ids) != _EXPECTED_TRAINING_VIDEOS:
        raise ValueError("v15 PE-off probe requires exactly train337")
    if provenance.upstream_encoder_checkpoint_sha256 is not None:
        raise ValueError("encoder provenance unexpectedly names an upstream checkpoint")
    if provenance.container_image_id is None or provenance.container_environment_sha256 is None:
        raise ValueError("v15 PE-off probe requires formal container provenance")
    validate_terminal_checkpoint(
        paths["encoder_checkpoint"],
        config,
        expected_stage="encoder",
        expected_provenance=provenance,
        progress_path=paths["encoder_progress"],
    )
    algorithm_source_git_sha = clean_git_revision(Path.cwd())
    runtime_container = _v14._validate_runtime_binding(
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
        raise ValueError("every v15 training pose must contain exactly 256 frames")
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
    baseline, peoff, mutation_audit = _run_in_memory_modes(
        model,
        sequences,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    projected_identity = _projected_identity_audit(baseline, peoff)

    baseline_distributions = {
        path: _v14._distribution(
            rows,
            minimum=config.period.minimum,
            maximum=config.period.maximum,
        )
        for path, rows in baseline.original.paths.items()
    }
    baseline_time_scale = {
        path: _time_scale_summary(baseline, config=config, path_name=path)
        for path in baseline.original.paths
    }
    baseline_cross_path = _v14._cross_path_comparison(
        baseline.original.paths["projected_pre_pe"],
        baseline.original.paths["post_pe"],
    )
    peoff_distributions = {
        path: _v14._distribution(
            rows,
            minimum=config.period.minimum,
            maximum=config.period.maximum,
        )
        for path, rows in peoff.original.paths.items()
    }
    peoff_time_scale = {
        path: _time_scale_summary(peoff, config=config, path_name=path)
        for path in peoff.original.paths
    }
    peoff_cross_path = _v14._cross_path_comparison(
        peoff.original.paths["projected_pre_pe"],
        peoff.original.paths["post_pe"],
    )
    decision = _gate_decision(
        projected_distribution=peoff_distributions["projected_pre_pe"],
        projected_time_scale=peoff_time_scale["projected_pre_pe"],
        peoff_post_distribution=peoff_distributions["post_pe"],
        peoff_post_time_scale=peoff_time_scale["post_pe"],
        cross_path=peoff_cross_path,
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
        raise RuntimeError("train337 pose selection changed during PE-off probe")
    if final_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("train337 pose cache changed during PE-off probe")
    _v14._require_unchanged(paths, identities)
    if clean_git_revision(Path.cwd()) != algorithm_source_git_sha:
        raise RuntimeError("algorithm source changed during PE-off probe")

    hardware = hardware_fingerprint()
    runtime = _v14._runtime_provenance(
        runner_sha256=identities["v15_peoff_probe_runner"][0],
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
        "checkpoint algorithm source and external PE-off probe source are independently bound"
    )
    code_names = (
        "v15_peoff_probe_runner",
        "v15_terminal_gate_dependency",
        "v14_counterfactual_dependency",
        "final_gate_dependency",
        "epoch11_gate_dependency",
    )
    code_files_sha256 = {name: identities[name][0] for name in code_names}
    payload = {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "status": ("peoff_gate_pass" if decision["peoff_gate_pass"] else "peoff_gate_rejected"),
        "classification": _CLASSIFICATION,
        "disclosed_by_pams_authors": False,
        "eligible_for_paper_table": False,
        "protocol": config.protocol,
        "seed": config.seed,
        "label_firewall": {
            "accepted_scientific_inputs": [
                "exact_v15_experiment_config",
                "terminal_v15_encoder_checkpoint_and_progress",
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
        "normal_sinusoidal_baseline": {
            "position_encoding_mode": "sinusoidal",
            "paths": {
                "projected_pose_velocity_vector_acf": {
                    "distribution": baseline_distributions["projected_pre_pe"],
                    "time_scale_consistency": baseline_time_scale["projected_pre_pe"],
                },
                "embedding_velocity_vector_acf": {
                    "distribution": baseline_distributions["post_pe"],
                    "time_scale_consistency": baseline_time_scale["post_pe"],
                },
            },
            "cross_path_comparison": baseline_cross_path,
        },
        "in_memory_peoff": {
            "position_encoding_mode": "none",
            "checkpoint_weights_reloaded_or_modified": False,
            "paths": {
                "projected_pose_velocity_vector_acf": {
                    "distribution": peoff_distributions["projected_pre_pe"],
                    "time_scale_consistency": peoff_time_scale["projected_pre_pe"],
                },
                "embedding_velocity_vector_acf": {
                    "distribution": peoff_distributions["post_pe"],
                    "time_scale_consistency": peoff_time_scale["post_pe"],
                },
            },
            "cross_path_comparison": peoff_cross_path,
        },
        "projected_path_identity_audit": projected_identity,
        "in_memory_mutation_audit": mutation_audit,
        "gate": decision,
        "scientific_caveats": [
            (
                "This is an inference-time causal ablation of a sinusoidal-PE "
                "checkpoint, not a checkpoint trained without absolute PE."
            ),
            (
                "All ten thresholds are independently inferred diagnostics "
                "reused unchanged from the terminal v15 gate."
            ),
            (
                "Even a pass cannot independently authorize fresh NoAbsPE "
                "training; a separate current alias/identifiability probe is required."
            ),
            (
                "SSHead, development, and sealed-test access remain unauthorized "
                "regardless of this probe result."
            ),
        ],
        "hardware": hardware,
        "hardware_sha256": sha256_json(hardware),
        "runtime": runtime,
        "runtime_sha256": sha256_json(runtime),
        "read_only_verification": {
            "all_file_inputs_unchanged": True,
            "checkpoint_file_sha256_before_after_unchanged": True,
            "train337_pose_cache_set_sha256_unchanged": True,
            "checkpoint_algorithm_source_git_sha_unchanged": True,
            "model_tensor_state_sha256_before_after_unchanged": True,
            "position_encoding_mode_restored": True,
            "model_or_optimizer_tensor_state_updated": False,
            "training_steps_executed": 0,
            "optimizer_created": False,
            "pose_cache_write_operations": 0,
        },
    }
    del baseline, peoff, model
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
        raise FileExistsError(f"refusing to overwrite probe artifact: {output}")
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite probe receipt: {receipt_path}")
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
        "peoff_gate_pass": payload["gate"]["peoff_gate_pass"],
        "standalone_noabs_training_authorized": payload["gate"][
            "standalone_noabs_training_authorized"
        ],
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
    _v14._write_new_regular_file(receipt_path, _v14._encoded_json(receipt))
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
        raise FileExistsError("probe output and receipt destinations must both be new")
    payload = run_peoff_probe(
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
    passed = payload["gate"]["peoff_gate_pass"]
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "output": str(arguments.output),
                "receipt": str(receipt_path),
                "peoff_gate_pass": passed,
                "standalone_noabs_training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
