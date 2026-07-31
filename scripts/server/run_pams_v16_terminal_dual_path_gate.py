"""Run the frozen train337-only terminal dual-path gate for PAMS-v16.

This read-only, independently inferred gate accepts only the exact frozen v16
configuration, its exact terminal encoder checkpoint/progress pair,
checkpoint-bound train337 pose cache, and exact checkpoint-source receipt.
The checkpoint must contain 150 completed epochs: ten pose-proxy epochs
followed by 140 projected-pose teacher epochs, with absolute positional
encoding disabled.

The unchanged v14 dual-path measurements and all ten frozen criteria are
applied to the projected pre-position path and the post-Transformer NoAbsPE
path.  Passing authorizes only label-free SSHead training on the same train337
set.  Development prediction/scoring and sealed-test evaluation are never
authorized by this program.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from pams.config import PAMSConfig, load_config
from pams.data import PoseCacheSetSnapshot
from pams.diagnostics import (
    _device,
    _identifier_commitment,
    _load_training_poses,
    _peek_checkpoint,
    _stable_file_sha256,
)
from pams.reproducibility import clean_git_revision, hardware_fingerprint, sha256_json
from pams.training import load_model_checkpoint, validate_terminal_checkpoint

try:
    from scripts.server import run_pams_v16_epoch11_dual_path_gate as _epoch11
except ModuleNotFoundError:
    import run_pams_v16_epoch11_dual_path_gate as _epoch11  # type: ignore[no-redef]

_v14 = _epoch11._v14

_ARTIFACT_TYPE = "pams_v16_terminal_encoder_dual_path_gate"
_RECEIPT_TYPE = "pams_v16_terminal_encoder_dual_path_gate_receipt"
_CLASSIFICATION = "inferred target-free read-only terminal v16 gate"
_EXPECTED_TRAINING_VIDEOS = 337
_EXPECTED_COMPLETED_EPOCHS = 150
_EXPECTED_POSE_PROXY_EPOCHS = 10
_EXPECTED_PROJECTED_TEACHER_EPOCHS = 140
_EXPECTED_CONFIG_SHA256 = "49cbfd6526c64b18d2ffc266658e5c4e27d1d59454c5cdc6f1f42420caca3a04"
_EXPECTED_CONFIG_FINGERPRINT = "d00cfee1875ae597bbeec8a3fd8ae9f7bd01a7ad490d1d7048a3f7af22185c8e"
_EXPECTED_NONSEED_FINGERPRINT = "b1d303597ecd19a9ae3d8f7475909087e60ca56465b00f20981f758363f9f416"
_EXPECTED_POSE_FINGERPRINT = "3dd0388320095796f42aa82d904073b6478b073ed351394ef8d03c30798a0116"
# Frozen only after the deterministic terminal train337 artifacts completed.
# The validator still rejects any future placeholder/sentinel before this
# runner deserializes checkpoint bytes.
_EXPECTED_ENCODER_CHECKPOINT_SHA256 = (
    "491fd5df67fb4d1fc7669019e4f78fac7b5c681dfa94a4cc7a469da2f73f5196"
)
_EXPECTED_ENCODER_PROGRESS_SHA256 = (
    "047c6af919843cd5d1133033ae07314b8ddd9426d878e4b6519181b94eb13ec7"
)


def _validate_exact_v16_config(
    config: PAMSConfig,
    *,
    config_sha256: str,
) -> None:
    actual = {
        "config_sha256": config_sha256,
        "config_fingerprint": config.fingerprint,
        "nonseed_fingerprint": config.nonseed_fingerprint,
        "pose_fingerprint": config.pose_fingerprint,
        "protocol": config.protocol,
        "seed": config.seed,
        "frames": config.data.frames,
        "training_epochs": config.training.epochs,
        "pose_energy_epochs": config.period.pose_energy_epochs,
        "position_encoding_mode": config.model.position_encoding_mode,
        "post_warmup_source": config.period.post_warmup_source,
        "loss_scales": config.loss.scales,
    }
    expected = {
        "config_sha256": _EXPECTED_CONFIG_SHA256,
        "config_fingerprint": _EXPECTED_CONFIG_FINGERPRINT,
        "nonseed_fingerprint": _EXPECTED_NONSEED_FINGERPRINT,
        "pose_fingerprint": _EXPECTED_POSE_FINGERPRINT,
        "protocol": "ucfrep_526",
        "seed": 2026,
        "frames": 256,
        "training_epochs": _EXPECTED_COMPLETED_EPOCHS,
        "pose_energy_epochs": _EXPECTED_POSE_PROXY_EPOCHS,
        "position_encoding_mode": "none",
        "post_warmup_source": "projected_pose_velocity_vector_acf",
        "loss_scales": (0.5, 1.0, 1.5),
    }
    if actual != expected:
        raise ValueError(
            "terminal gate requires the exact frozen v16 config: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )


def _validate_exact_terminal_artifact_identities(
    identities: Mapping[str, tuple[str, int]],
) -> None:
    expected = {
        "encoder_checkpoint_sha256": _EXPECTED_ENCODER_CHECKPOINT_SHA256,
        "encoder_progress_sha256": _EXPECTED_ENCODER_PROGRESS_SHA256,
    }
    if any(
        len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
        for value in expected.values()
    ):
        raise RuntimeError(
            "v16 terminal checkpoint/progress identities have not been frozen in this gate source"
        )
    actual = {
        "encoder_checkpoint_sha256": identities["encoder_checkpoint"][0],
        "encoder_progress_sha256": identities["encoder_progress"][0],
    }
    if actual != expected:
        raise ValueError(
            "terminal gate requires the exact frozen v16 artifacts: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )


def _terminal_schedule_metadata(
    checkpoint: Path,
    *,
    config: PAMSConfig,
) -> dict[str, Any]:
    """Expose and independently recheck the terminal teacher schedule."""

    payload = torch.load(
        checkpoint,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("terminal checkpoint root must be a mapping")
    if payload.get("stage") != "encoder":
        raise ValueError("v16 terminal gate requires an encoder checkpoint")
    if payload.get("config_fingerprint") != config.fingerprint:
        raise ValueError("terminal checkpoint config fingerprint does not match")
    completed = payload.get("completed_epochs")
    if completed != _EXPECTED_COMPLETED_EPOCHS:
        raise ValueError("v16 terminal gate requires completed_epochs=150")
    history = payload.get("history")
    if not isinstance(history, list) or len(history) != _EXPECTED_COMPLETED_EPOCHS:
        raise ValueError("terminal checkpoint history must contain exactly 150 rows")
    sources: list[str] = []
    for expected_epoch, row in enumerate(history, start=1):
        if not isinstance(row, Mapping) or row.get("epoch") != expected_epoch:
            raise ValueError("terminal checkpoint history must be consecutive from one")
        source = row.get("period_source")
        if not isinstance(source, str):
            raise ValueError("terminal checkpoint history period_source is missing")
        sources.append(source)
    expected_sources = ["pose"] * _EXPECTED_POSE_PROXY_EPOCHS + [
        "projected_pose_velocity_vector_acf"
    ] * _EXPECTED_PROJECTED_TEACHER_EPOCHS
    if sources != expected_sources:
        raise ValueError(
            "terminal checkpoint must contain ten pose epochs followed by "
            "140 projected-pose-velocity-vector-ACF epochs"
        )
    return {
        "completed_epochs": completed,
        "pose_proxy_epoch_total": sources.count("pose"),
        "projected_teacher_epoch_total": sources.count("projected_pose_velocity_vector_acf"),
        "first_projected_teacher_epoch": _EXPECTED_POSE_PROXY_EPOCHS + 1,
        "last_projected_teacher_epoch": _EXPECTED_COMPLETED_EPOCHS,
        "period_source_schedule_sha256": sha256_json(sources),
        "terminal_checkpoint_validation_called": True,
    }


def _gate_decision(
    *,
    projected_distribution: Mapping[str, Any],
    projected_time_scale: Mapping[str, Any],
    post_transformer_distribution: Mapping[str, Any],
    post_transformer_time_scale: Mapping[str, Any],
    cross_path: Mapping[str, Any],
) -> dict[str, Any]:
    frozen = _v14._gate_decision(
        projected_distribution=projected_distribution,
        projected_time_scale=projected_time_scale,
        post_pe_distribution=post_transformer_distribution,
        post_pe_time_scale=post_transformer_time_scale,
        cross_path=cross_path,
    )
    criteria = frozen["criteria"]
    if len(criteria) != 10:
        raise RuntimeError("v14 counterfactual dependency no longer exposes ten criteria")
    authorized = all(bool(item["pass"]) for item in criteria.values())
    return {
        "thresholds_reused_unchanged_from_v14_counterfactual": True,
        "criteria": criteria,
        "all_ten_diagnostic_criteria_pass": authorized,
        "sshead_training_authorization_basis": sorted(criteria),
        "sshead_training_authorized": authorized,
        "authorized_scope": (
            {
                "protocol": "ucfrep_526",
                "seed": 2026,
                "training_video_total": 337,
                "stage": "sshead",
                "encoder_state": "frozen_exact_terminal_v16",
                "maximum_training_epochs": 30,
                "label_or_target_input": False,
            }
            if authorized
            else None
        ),
        "encoder_training_authorized": False,
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }


def _input_identities(paths: Mapping[str, Path]) -> dict[str, tuple[str, int]]:
    return {name: _stable_file_sha256(path) for name, path in paths.items()}


def run_terminal_gate(
    encoder_checkpoint_path: str | Path,
    encoder_progress_path: str | Path,
    config_path: str | Path,
    pose_cache_dir: str | Path,
    source_receipt_path: str | Path,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate the exact terminal v16 encoder on bound train337 only."""

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
        "v16_terminal_gate_runner": Path(__file__),
        "v16_epoch11_gate_dependency": Path(_epoch11.__file__),
        "v14_counterfactual_dependency": Path(_v14.__file__),
        "final_gate_dependency": Path(_v14._final_gate.__file__),
        "epoch11_gate_dependency": Path(_v14._epoch11_gate.__file__),
    }
    identities = _input_identities(paths)
    # Fail closed on an unfrozen or different checkpoint before torch.load.
    _validate_exact_terminal_artifact_identities(identities)
    _v14._validate_source_receipt_argument(
        paths["source_export_receipt"],
        receipt_sha256=identities["source_export_receipt"][0],
    )
    gate_code_source_git_sha = _v14._gate_code_source_revision()
    config = load_config(paths["config"])
    _validate_exact_v16_config(
        config,
        config_sha256=identities["config"][0],
    )
    stage, provenance = _peek_checkpoint(paths["encoder_checkpoint"], config)
    if stage != "encoder":
        raise ValueError("v16 terminal gate requires an encoder checkpoint")
    if len(provenance.training_video_ids) != _EXPECTED_TRAINING_VIDEOS:
        raise ValueError("v16 terminal gate requires exactly train337")
    if provenance.upstream_encoder_checkpoint_sha256 is not None:
        raise ValueError("encoder provenance unexpectedly names an upstream checkpoint")
    if provenance.container_image_id is None or provenance.container_environment_sha256 is None:
        raise ValueError("v16 terminal gate requires formal container provenance")
    validate_terminal_checkpoint(
        paths["encoder_checkpoint"],
        config,
        expected_stage="encoder",
        expected_provenance=provenance,
        progress_path=paths["encoder_progress"],
    )
    schedule_metadata = _terminal_schedule_metadata(
        paths["encoder_checkpoint"],
        config=config,
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
        raise ValueError("every v16 training pose must contain exactly 256 frames")
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
    model_state_before = _epoch11._model_state_sha256(model)
    baseline = _v14._encode_dual_paths(
        model,
        sequences,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    projected_distribution = _v14._distribution(
        baseline["projected_pre_pe"],
        minimum=config.period.minimum,
        maximum=config.period.maximum,
    )
    post_transformer_distribution = _v14._distribution(
        baseline["post_pe"],
        minimum=config.period.minimum,
        maximum=config.period.maximum,
    )
    time_scale = _v14._time_scale_consistency(
        model,
        sequences,
        baseline,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    cross_path = _v14._cross_path_comparison(
        baseline["projected_pre_pe"],
        baseline["post_pe"],
    )
    decision = _gate_decision(
        projected_distribution=projected_distribution,
        projected_time_scale=time_scale["projected_pre_pe"],
        post_transformer_distribution=post_transformer_distribution,
        post_transformer_time_scale=time_scale["post_pe"],
        cross_path=cross_path,
    )
    model_state_after = _epoch11._model_state_sha256(model)
    if model_state_after != model_state_before:
        raise RuntimeError("model state changed during read-only terminal gate")

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
        raise RuntimeError("train337 pose selection changed during terminal gate")
    if final_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("train337 pose cache changed during terminal gate")
    _v14._require_unchanged(paths, identities)
    if clean_git_revision(Path.cwd()) != algorithm_source_git_sha:
        raise RuntimeError("algorithm source changed during terminal gate")

    hardware = hardware_fingerprint()
    runtime = _v14._runtime_provenance(
        runner_sha256=identities["v16_terminal_gate_runner"][0],
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
        "checkpoint algorithm source and external terminal gate source are independently bound"
    )
    code_names = (
        "v16_terminal_gate_runner",
        "v16_epoch11_gate_dependency",
        "v14_counterfactual_dependency",
        "final_gate_dependency",
        "epoch11_gate_dependency",
    )
    code_files_sha256 = {name: identities[name][0] for name in code_names}
    payload = {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "status": (
            "train337_sshead_training_authorized"
            if decision["sshead_training_authorized"]
            else "train337_sshead_training_rejected"
        ),
        "classification": _CLASSIFICATION,
        "disclosed_by_pams_authors": False,
        "eligible_for_paper_table": False,
        "protocol": config.protocol,
        "seed": config.seed,
        "label_firewall": {
            "accepted_scientific_inputs": [
                "exact_v16_experiment_config",
                "exact_terminal_v16_encoder_checkpoint_and_progress",
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
            **{f"{name}_sha256": value[0] for name, value in identities.items()},
            **{f"{name}_bytes": value[1] for name, value in identities.items()},
            "config_fingerprint": config.fingerprint,
            "nonseed_fingerprint": config.nonseed_fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
            "encoder_provenance": provenance.to_dict(),
            "checkpoint_schedule": schedule_metadata,
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
            "post_transformer_no_absolute_pe_velocity_vector_acf": {
                "feature_source": (
                    "L2-normalized Transformer output with absolute positional encoding disabled"
                ),
                "distribution": post_transformer_distribution,
                "time_scale_consistency": time_scale["post_pe"],
            },
        },
        "time_scale_factors": list(_v14._TIME_SCALE_FACTORS),
        "cross_path_comparison": cross_path,
        "gate": decision,
        "scientific_caveats": [
            (
                "All ten criteria and thresholds are unchanged independently "
                "inferred v14 diagnostics, not author-disclosed validation."
            ),
            (
                "Passing authorizes only label-free SSHead training on the "
                "same train337 set with this exact frozen terminal encoder."
            ),
            (
                "Development prediction/scoring and sealed-test evaluation "
                "remain unauthorized regardless of every diagnostic result."
            ),
            "Passing does not establish repetition-count accuracy.",
        ],
        "hardware": hardware,
        "hardware_sha256": sha256_json(hardware),
        "runtime": runtime,
        "runtime_sha256": sha256_json(runtime),
        "read_only_verification": {
            "all_file_inputs_unchanged": True,
            "train337_pose_cache_set_sha256_unchanged": True,
            "checkpoint_algorithm_source_git_sha_unchanged": True,
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
    if output.exists():
        raise FileExistsError(f"refusing to overwrite gate artifact: {output}")
    if receipt_path.exists():
        raise FileExistsError(f"refusing to overwrite gate receipt: {receipt_path}")
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
        "sshead_training_authorized": payload["gate"]["sshead_training_authorized"],
        "encoder_training_authorized": False,
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
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
        raise FileExistsError("gate output and receipt destinations must both be new")
    payload = run_terminal_gate(
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
    authorized = payload["gate"]["sshead_training_authorized"]
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "output": str(arguments.output),
                "receipt": str(receipt_path),
                "train337_sshead_training_authorized": authorized,
                "dev84_prediction_authorized": False,
                "dev84_scoring_authorized": False,
                "test105_evaluation_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0 if authorized else 3


if __name__ == "__main__":
    raise SystemExit(main())
