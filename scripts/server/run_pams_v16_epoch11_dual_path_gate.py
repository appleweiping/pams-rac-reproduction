"""Run the frozen train337-only epoch-11 dual-path gate for PAMS-v16.

This read-only, independently inferred gate accepts only the exact frozen v16
configuration, its exact epoch-11 encoder checkpoint/progress pair,
checkpoint-bound train337 pose cache, and exact checkpoint-source receipt.
It validates ten pose-proxy epochs followed by one projected-pose teacher
epoch.  It deliberately does not call terminal-checkpoint validation.

The unchanged v14 dual-path measurements and all ten frozen criteria are
applied to the projected pre-position path and the post-Transformer path with
absolute positional encoding disabled.  Passing authorizes only encoder
epochs 12--150.  SSHead training and development/test access are never
authorized by this program.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields
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
from pams.reproducibility import clean_git_revision, hardware_fingerprint, sha256_json
from pams.training import (
    EncoderEpochStats,
    _progress_row,
    _read_progress_rows,
    load_model_checkpoint,
)

try:
    from scripts.server import run_pams_v14_dual_path_counterfactual as _v14
except ModuleNotFoundError:
    import run_pams_v14_dual_path_counterfactual as _v14  # type: ignore[no-redef]

_ARTIFACT_TYPE = "pams_v16_epoch11_encoder_dual_path_gate"
_RECEIPT_TYPE = "pams_v16_epoch11_encoder_dual_path_gate_receipt"
_CLASSIFICATION = "inferred target-free read-only epoch-11 v16 gate"
_EXPECTED_TRAINING_VIDEOS = 337
_EXPECTED_COMPLETED_EPOCHS = 11
_EXPECTED_CHECKPOINT_SCHEMA_VERSION = 5
_EXPECTED_CONFIG_SHA256 = "49cbfd6526c64b18d2ffc266658e5c4e27d1d59454c5cdc6f1f42420caca3a04"
_EXPECTED_CONFIG_FINGERPRINT = "d00cfee1875ae597bbeec8a3fd8ae9f7bd01a7ad490d1d7048a3f7af22185c8e"
_EXPECTED_NONSEED_FINGERPRINT = "b1d303597ecd19a9ae3d8f7475909087e60ca56465b00f20981f758363f9f416"
_EXPECTED_POSE_FINGERPRINT = "3dd0388320095796f42aa82d904073b6478b073ed351394ef8d03c30798a0116"
# These two identities are frozen in a follow-up gate-code-only commit after
# the deterministic epoch-11 artifacts exist.  A sentinel can never authorize
# or even execute the scientific gate.
_EXPECTED_ENCODER_CHECKPOINT_SHA256 = (
    "7ee1617fcac65261222a8c37c90977e92580b73c017a3d87465e794b205d7b31"
)
_EXPECTED_ENCODER_PROGRESS_SHA256 = (
    "c3782c40484916aaec97cd4b853ab6638398b00aa55ae7a3d69c2f3b862fa9ab"
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
        "training_epochs": 150,
        "position_encoding_mode": "none",
        "post_warmup_source": "projected_pose_velocity_vector_acf",
        "loss_scales": (0.5, 1.0, 1.5),
    }
    if actual != expected:
        raise ValueError(
            "epoch-11 dual-path gate requires the exact frozen v16 config: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )


def _validate_exact_epoch11_artifact_identities(
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
            "v16 epoch-11 checkpoint/progress identities have not been frozen in this gate source"
        )
    actual = {
        "encoder_checkpoint_sha256": identities["encoder_checkpoint"][0],
        "encoder_progress_sha256": identities["encoder_progress"][0],
    }
    if actual != expected:
        raise ValueError(
            "epoch-11 gate requires the exact frozen v16 artifacts: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )


def _checkpoint_epoch11_metadata(
    checkpoint: Path,
    progress: Path,
    *,
    config: PAMSConfig,
) -> dict[str, Any]:
    """Strictly validate a partial epoch-11 checkpoint and its exact log."""

    payload = torch.load(checkpoint, map_location=torch.device("cpu"), weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    expected_keys = {
        "schema_version",
        "stage",
        "config_fingerprint",
        "provenance",
        "completed_epochs",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "history",
        "cluster_assignments",
        "prototype_bank",
        "rng_state",
    }
    if set(payload) != expected_keys:
        raise ValueError("epoch-11 checkpoint top-level schema mismatch")
    if payload["schema_version"] != _EXPECTED_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported epoch-11 checkpoint schema")
    if payload["stage"] != "encoder":
        raise ValueError("v16 epoch-11 gate requires an encoder checkpoint")
    if payload["config_fingerprint"] != config.fingerprint:
        raise ValueError("checkpoint config fingerprint does not match v16")
    if payload["completed_epochs"] != _EXPECTED_COMPLETED_EPOCHS:
        raise ValueError("v16 epoch-11 gate requires completed_epochs=11")
    if not isinstance(payload["model_state"], Mapping):
        raise ValueError("epoch-11 checkpoint model_state must be a mapping")
    for field in ("optimizer_state", "scheduler_state", "rng_state"):
        if not isinstance(payload[field], Mapping):
            raise ValueError(f"epoch-11 checkpoint {field} must be a mapping")
    if not isinstance(payload["cluster_assignments"], Mapping):
        raise ValueError("epoch-11 checkpoint cluster assignments must be a mapping")
    if not isinstance(payload["prototype_bank"], Mapping):
        raise ValueError("epoch-11 checkpoint prototype bank must be a mapping")

    raw_history = payload["history"]
    if not isinstance(raw_history, list) or len(raw_history) != _EXPECTED_COMPLETED_EPOCHS:
        raise ValueError("checkpoint epoch history must contain exactly 11 rows")
    history: list[EncoderEpochStats] = []
    expected_fields = {field.name for field in fields(EncoderEpochStats)}
    integer_fields = {
        "epoch",
        "optimizer_steps",
        "cross_cluster_requested",
        "cross_cluster_actual",
        "cross_cluster_shortfall",
    }
    for expected_epoch, item in enumerate(raw_history, start=1):
        if not isinstance(item, dict):
            raise ValueError("checkpoint history rows must be mappings")
        row = dict(item)
        row.setdefault("position_permutation_consistency", 0.0)
        if set(row) != expected_fields:
            raise ValueError("checkpoint epoch history schema mismatch")
        for name, value in row.items():
            if name in integer_fields:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"checkpoint history {name} must be an integer")
            elif name == "clusters_refreshed":
                if not isinstance(value, bool):
                    raise ValueError("checkpoint history clusters_refreshed must be boolean")
            elif name == "period_source":
                if not isinstance(value, str):
                    raise ValueError("checkpoint history period_source must be text")
            elif not isinstance(value, float) or not math.isfinite(value):
                raise ValueError(f"checkpoint history {name} must be a finite float")
        statistics = EncoderEpochStats(**row)
        if statistics.epoch != expected_epoch:
            raise ValueError("checkpoint history epochs must be consecutive from one")
        if (
            statistics.cross_cluster_requested
            != statistics.cross_cluster_actual + statistics.cross_cluster_shortfall
        ):
            raise ValueError("checkpoint history cross-cluster accounting mismatch")
        if statistics.cross_cluster_shortfall != 0:
            raise ValueError("formal epoch-11 checkpoint has cross-cluster shortfall")
        history.append(statistics)

    expected_sources = ["pose"] * config.period.pose_energy_epochs + [
        "projected_pose_velocity_vector_acf"
    ]
    sources = [statistics.period_source for statistics in history]
    if sources != expected_sources:
        raise ValueError(
            "checkpoint must contain ten pose epochs followed by one "
            "projected-pose-velocity-vector-ACF epoch"
        )
    progress_rows = _read_progress_rows(progress)
    if len(progress_rows) != _EXPECTED_COMPLETED_EPOCHS:
        raise ValueError("epoch-11 progress log must contain exactly 11 rows")
    expected_progress = tuple(
        _progress_row(stage="encoder", stats=statistics, config=config) for statistics in history
    )
    if progress_rows != expected_progress:
        raise ValueError("epoch-11 progress log does not exactly match checkpoint history")
    return {
        "completed_epochs": _EXPECTED_COMPLETED_EPOCHS,
        "period_sources": sources,
        "pose_proxy_epoch_total": config.period.pose_energy_epochs,
        "projected_teacher_epoch_total": 1,
        "first_post_warmup_epoch": _EXPECTED_COMPLETED_EPOCHS,
        "terminal_checkpoint_validation_called": False,
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
        "encoder_epochs12_150_continuation_authorization_basis": sorted(criteria),
        "encoder_epochs12_150_continuation_authorized": authorized,
        "sshead_training_authorized": False,
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }


def _hash_part(hasher: Any, value: bytes) -> None:
    hasher.update(len(value).to_bytes(8, byteorder="big", signed=False))
    hasher.update(value)


def _model_state_sha256(model: nn.Module) -> str:
    hasher = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        if not isinstance(tensor, Tensor):
            raise TypeError("model state contains a non-tensor value")
        value = tensor.detach().cpu().contiguous()
        _hash_part(hasher, name.encode("utf-8"))
        _hash_part(hasher, str(value.dtype).encode("ascii"))
        _hash_part(
            hasher,
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"),
        )
        _hash_part(hasher, value.view(torch.uint8).numpy().tobytes())
    return hasher.hexdigest()


def _input_identities(paths: Mapping[str, Path]) -> dict[str, tuple[str, int]]:
    return {name: _stable_file_sha256(path) for name, path in paths.items()}


def run_epoch11_gate(
    encoder_checkpoint_path: str | Path,
    encoder_progress_path: str | Path,
    config_path: str | Path,
    pose_cache_dir: str | Path,
    source_receipt_path: str | Path,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate the exact v16 epoch-11 encoder on bound train337 only."""

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
        "v16_epoch11_gate_runner": Path(__file__),
        "v14_counterfactual_dependency": Path(_v14.__file__),
        "final_gate_dependency": Path(_v14._final_gate.__file__),
        "epoch11_gate_dependency": Path(_v14._epoch11_gate.__file__),
    }
    identities = _input_identities(paths)
    _validate_exact_epoch11_artifact_identities(identities)
    _v14._validate_source_receipt_argument(
        paths["source_export_receipt"],
        receipt_sha256=identities["source_export_receipt"][0],
    )
    gate_code_source_git_sha = _v14._gate_code_source_revision()
    config = load_config(paths["config"])
    _validate_exact_v16_config(config, config_sha256=identities["config"][0])
    stage, provenance = _peek_checkpoint(paths["encoder_checkpoint"], config)
    if stage != "encoder":
        raise ValueError("v16 epoch-11 gate requires an encoder checkpoint")
    if len(provenance.training_video_ids) != _EXPECTED_TRAINING_VIDEOS:
        raise ValueError("v16 epoch-11 gate requires exactly train337")
    if provenance.upstream_encoder_checkpoint_sha256 is not None:
        raise ValueError("encoder provenance unexpectedly names an upstream checkpoint")
    if provenance.container_image_id is None or provenance.container_environment_sha256 is None:
        raise ValueError("v16 epoch-11 gate requires formal container provenance")
    checkpoint_metadata = _checkpoint_epoch11_metadata(
        paths["encoder_checkpoint"],
        paths["encoder_progress"],
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
    model_state_before = _model_state_sha256(model)
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
    model_state_after = _model_state_sha256(model)
    if model_state_after != model_state_before:
        raise RuntimeError("model state changed during read-only epoch-11 gate")

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
        raise RuntimeError("train337 pose selection changed during epoch-11 gate")
    if final_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("train337 pose cache changed during epoch-11 gate")
    _v14._require_unchanged(paths, identities)
    if clean_git_revision(Path.cwd()) != algorithm_source_git_sha:
        raise RuntimeError("algorithm source changed during epoch-11 gate")

    hardware = hardware_fingerprint()
    runtime = _v14._runtime_provenance(
        runner_sha256=identities["v16_epoch11_gate_runner"][0],
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
        "checkpoint algorithm source and external epoch-11 gate source are independently bound"
    )
    code_names = (
        "v16_epoch11_gate_runner",
        "v14_counterfactual_dependency",
        "final_gate_dependency",
        "epoch11_gate_dependency",
    )
    code_files_sha256 = {name: identities[name][0] for name in code_names}
    payload = {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "status": (
            "encoder_epochs12_150_continuation_authorized"
            if decision["encoder_epochs12_150_continuation_authorized"]
            else "encoder_epochs12_150_continuation_rejected"
        ),
        "classification": _CLASSIFICATION,
        "disclosed_by_pams_authors": False,
        "eligible_for_paper_table": False,
        "protocol": config.protocol,
        "seed": config.seed,
        "label_firewall": {
            "accepted_scientific_inputs": [
                "exact_v16_experiment_config",
                "exact_v16_epoch11_encoder_checkpoint_and_progress",
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
            "checkpoint_metadata": checkpoint_metadata,
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
                "feature_source": "encoder.input_projection output before positional encoding",
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
                "Passing authorizes only encoder epochs 12--150 from this exact "
                "epoch-11 checkpoint; it cannot authorize SSHead or any dev/test access."
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
        "encoder_epochs12_150_continuation_authorized": payload["gate"][
            "encoder_epochs12_150_continuation_authorized"
        ],
        "sshead_training_authorized": False,
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
    payload = run_epoch11_gate(
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
    authorized = payload["gate"]["encoder_epochs12_150_continuation_authorized"]
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "output": str(arguments.output),
                "receipt": str(receipt_path),
                "encoder_epochs12_150_continuation_authorized": authorized,
                "sshead_training_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0 if authorized else 3


if __name__ == "__main__":
    raise SystemExit(main())
