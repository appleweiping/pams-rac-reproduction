"""Run the exact train337-only final SSHead gate for PAMS-v16.

This independently inferred, read-only gate accepts only the exact v16
NoAbsPE configuration, terminal encoder150/SSHead30 checkpoints and progress
logs, and their checkpoint-bound train337 pose cache.  It reuses the thirteen
frozen v14 anti-collapse and self-consistency criteria without changing any
threshold.

Only a 13/13 pass can authorize one isolated dev84 prediction process.
Development scoring and sealed-test evaluation are always unauthorized.
Failure is a scientific rejection and is emitted without reinterpretation.
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
from pams.training import (
    load_model_checkpoint,
    validate_sshead_encoder_binding,
    validate_terminal_checkpoint,
)

try:
    from scripts.server import run_pams_v14_final_train_gate as _v14
    from scripts.server import run_pams_v16_terminal_dual_path_gate as _v16
except ModuleNotFoundError:
    import run_pams_v14_final_train_gate as _v14  # type: ignore[no-redef]
    import run_pams_v16_terminal_dual_path_gate as _v16  # type: ignore[no-redef]

_ARTIFACT_TYPE = "pams_v16_final_train337_sshead_gate"
_RECEIPT_TYPE = "pams_v16_final_train337_sshead_gate_receipt"
_CLASSIFICATION = "inferred target-free terminal v16 SSHead diagnostic"
_EXPECTED_TRAINING_VIDEOS = 337
_EXPECTED_ENCODER_CHECKPOINT_SHA256 = (
    "491fd5df67fb4d1fc7669019e4f78fac7b5c681dfa94a4cc7a469da2f73f5196"
)
_EXPECTED_ENCODER_PROGRESS_SHA256 = (
    "047c6af919843cd5d1133033ae07314b8ddd9426d878e4b6519181b94eb13ec7"
)
_EXPECTED_SSHEAD_CHECKPOINT_SHA256 = (
    "bad56ddf7b53561733cfa3d4786df6ba8d286e9b02cd656335bdc212b9fc7b74"
)
_EXPECTED_SSHEAD_PROGRESS_SHA256 = (
    "0e820d03b6b4d86b2341074bbcb0d9793839a2bf2bbb4388f44343e397cd5401"
)
_EXPECTED_CONFIG_SHA256 = _v16._EXPECTED_CONFIG_SHA256
_EXPECTED_CONFIG_FINGERPRINT = _v16._EXPECTED_CONFIG_FINGERPRINT
_EXPECTED_NONSEED_FINGERPRINT = _v16._EXPECTED_NONSEED_FINGERPRINT
_EXPECTED_POSE_FINGERPRINT = _v16._EXPECTED_POSE_FINGERPRINT
_THRESHOLDS = _v14._THRESHOLDS


def _input_identities(paths: Mapping[str, Path]) -> dict[str, tuple[str, int]]:
    return {name: _stable_file_sha256(path) for name, path in paths.items()}


def _validate_exact_inputs(
    identities: Mapping[str, tuple[str, int]],
) -> None:
    expected = {
        "encoder_checkpoint": _EXPECTED_ENCODER_CHECKPOINT_SHA256,
        "encoder_progress": _EXPECTED_ENCODER_PROGRESS_SHA256,
        "sshead_checkpoint": _EXPECTED_SSHEAD_CHECKPOINT_SHA256,
        "sshead_progress": _EXPECTED_SSHEAD_PROGRESS_SHA256,
        "config": _EXPECTED_CONFIG_SHA256,
    }
    actual = {name: identities[name][0] for name in expected}
    if actual != expected:
        raise ValueError(
            "final v16 gate requires the exact frozen train-only artifacts: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )


def _validate_exact_v16_config(
    config: PAMSConfig,
    *,
    config_sha256: str,
) -> None:
    _v16._validate_exact_v16_config(config, config_sha256=config_sha256)
    if config.sshead.epochs != 30:
        raise ValueError("final v16 gate requires SSHead30")


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
    frozen = _v14._gate_decision(
        encoder_distribution=encoder_distribution,
        encoder_time_scale=encoder_time_scale,
        sshead_distribution=sshead_distribution,
        sshead_correlation=sshead_correlation,
        sshead_time_scale=sshead_time_scale,
        sshead_history=sshead_history,
        head_parameter_change=head_parameter_change,
    )
    criteria = frozen["criteria"]
    if len(criteria) != 13 or _THRESHOLDS != _v14._THRESHOLDS:
        raise RuntimeError("v14 final-gate criteria or thresholds changed")
    passed = all(bool(item["pass"]) for item in criteria.values())
    return {
        "thresholds_reused_unchanged_from_v14_final_gate": True,
        "criteria": criteria,
        "all_thirteen_diagnostic_criteria_pass": passed,
        "overall_pass": passed,
        "isolated_dev84_prediction_authorized": passed,
        "dev84_prediction_authorized": passed,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }


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
    """Evaluate exact encoder150/SSHead30 checkpoints using train337 only."""

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
        "v16_final_gate_runner": Path(__file__),
        "v14_final_gate_dependency": Path(_v14.__file__),
        "v14_epoch11_gate_dependency": Path(_v14._epoch11_gate.__file__),
        "v16_terminal_gate_dependency": Path(_v16.__file__),
    }
    identities = _input_identities(paths)
    _validate_exact_inputs(identities)
    gate_code_source_git_sha = _v14._gate_code_source_revision()
    config = load_config(paths["config"])
    _validate_exact_v16_config(config, config_sha256=identities["config"][0])
    encoder_stage, encoder_provenance = _peek_checkpoint(
        paths["encoder_checkpoint"],
        config,
    )
    sshead_stage, sshead_provenance = _peek_checkpoint(
        paths["sshead_checkpoint"],
        config,
    )
    if encoder_stage != "encoder" or sshead_stage != "sshead":
        raise ValueError("final gate requires encoder then SSHead checkpoints")
    _v14._validate_stage_bindings(
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
    runtime_container = _v14._validate_runtime_binding(
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
    if (
        full_snapshot.fingerprint != encoder_provenance.pose_cache_set_sha256
        or full_snapshot.fingerprint != sshead_provenance.pose_cache_set_sha256
    ):
        raise ValueError("train337 pose-cache set differs from checkpoint provenance")

    encoder_model = load_model_checkpoint(
        paths["encoder_checkpoint"],
        config,
        device=resolved_device,
        expected_stage="encoder",
        expected_provenance=encoder_provenance,
    ).eval()
    encoder_state_before = _v16._epoch11._model_state_sha256(encoder_model)
    encoder_samples = _v14._encode_training_sequences(
        encoder_model,
        sequences,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    encoder_distribution = _v14._training_distribution(
        encoder_samples,
        minimum=config.period.minimum,
        maximum=config.period.maximum,
    )
    encoder_time_scale = _v14._time_scale_consistency(
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
    sshead_state_before = _v16._epoch11._model_state_sha256(sshead_model)
    head_parameter_change = _v14._head_parameter_change(
        encoder_model,
        sshead_model,
    )
    sshead_samples = _v14._encode_head_sequences(
        sshead_model,
        sequences,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    sshead_distribution = _v14._training_distribution(
        sshead_samples,
        minimum=config.period.minimum,
        maximum=config.period.maximum,
    )
    sshead_correlation = _v14._centered_cross_video_correlation(sshead_samples)
    sshead_time_scale = _v14._head_time_scale_consistency(
        sshead_model,
        sequences,
        sshead_samples,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    sshead_history = _v14._sshead_history_audit(paths["sshead_checkpoint"])
    decision = _gate_decision(
        encoder_distribution=encoder_distribution,
        encoder_time_scale=encoder_time_scale,
        sshead_distribution=sshead_distribution,
        sshead_correlation=sshead_correlation,
        sshead_time_scale=sshead_time_scale,
        sshead_history=sshead_history,
        head_parameter_change=head_parameter_change,
    )
    encoder_state_after = _v16._epoch11._model_state_sha256(encoder_model)
    sshead_state_after = _v16._epoch11._model_state_sha256(sshead_model)
    if encoder_state_before != encoder_state_after or sshead_state_before != sshead_state_after:
        raise RuntimeError("model state changed during final read-only gate")

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
    if final_ids != selected_ids[:2] or final_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("train337 pose cache changed during final gate")
    _v14._require_unchanged(paths, identities)
    if clean_git_revision(Path.cwd()) != algorithm_source_git_sha:
        raise RuntimeError("algorithm source changed during final gate")

    hardware = hardware_fingerprint()
    runtime = _v14._runtime_provenance(
        runner_sha256=identities["v16_final_gate_runner"][0],
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
    code_names = (
        "v16_final_gate_runner",
        "v14_final_gate_dependency",
        "v14_epoch11_gate_dependency",
        "v16_terminal_gate_dependency",
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
        "protocol": config.protocol,
        "seed": config.seed,
        "label_firewall": {
            "accepted_scientific_inputs": [
                "exact_v16_experiment_config",
                "exact_terminal_encoder_checkpoint_and_progress",
                "exact_terminal_sshead_checkpoint_and_progress",
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
            **{f"{name}_sha256": value[0] for name, value in identities.items()},
            **{f"{name}_bytes": value[1] for name, value in identities.items()},
            "config_fingerprint": config.fingerprint,
            "nonseed_fingerprint": config.nonseed_fingerprint,
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
            "estimator": "post_transformer_noabs_velocity_vector_acf",
            "distribution": encoder_distribution,
            "time_scale_consistency": encoder_time_scale,
        },
        "sshead30": {
            "classification": "independently inferred, not author-disclosed",
            "distribution": sshead_distribution,
            "centered_cross_video_correlation": sshead_correlation,
            "time_scale_consistency": sshead_time_scale,
            "history": sshead_history,
            "head_parameter_change": head_parameter_change,
        },
        "gate": decision,
        "scientific_caveats": [
            "The thirteen criteria are inferred diagnostics, not author-disclosed validation.",
            "Passing self-consistency does not establish repetition-count accuracy.",
            (
                "Failure is retained as a scientific rejection; it cannot "
                "authorize development or sealed-test access."
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
            "encoder_model_state_sha256_before": encoder_state_before,
            "encoder_model_state_sha256_after": encoder_state_after,
            "sshead_model_state_sha256_before": sshead_state_before,
            "sshead_model_state_sha256_after": sshead_state_after,
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
    if output.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite final gate artifact or receipt")
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
    _v14._write_new_regular_file(receipt_path, _v14._encoded_json(receipt))
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
    receipt_path = _receipt_path(arguments.output)
    if arguments.output.exists() or receipt_path.exists():
        raise FileExistsError("final gate output and receipt destinations must both be new")
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
