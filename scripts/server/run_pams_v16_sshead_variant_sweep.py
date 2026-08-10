"""Train the frozen 2x2 PAMS-v16 SSHead variant sweep on train337 only.

The runner accepts the exact terminal v16 encoder/config, four preregistered
variant configs, the checkpoint-bound train337 pose cache, and a new output
root.  Variant configs may differ from the base config only in
``sshead.input_source`` and ``sshead.variance_weight``.  The complete allowlist
is the Cartesian product:

* input_source = {encoder_embedding, projected_pose_pre_pe}
* variance_weight = {1.0, 10.0}

Every SSHead30 starts from a fresh deterministic head and an exact tensor copy
of the same frozen upstream encoder.  This program has no development, test,
manifest, action, count, label, target, prediction, or scoring interface, and
its output never authorizes any such access.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from copy import deepcopy
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
    CheckpointProvenance,
    build_pams_model,
    load_model_checkpoint,
    train_sshead,
    validate_sshead_encoder_binding,
    validate_terminal_checkpoint,
)

try:
    from scripts.server import run_pams_v14_final_train_gate as _v14_final
    from scripts.server import run_pams_v16_terminal_dual_path_gate as _v16
except ModuleNotFoundError:
    import run_pams_v14_final_train_gate as _v14_final  # type: ignore[no-redef]
    import run_pams_v16_terminal_dual_path_gate as _v16  # type: ignore[no-redef]

_ARTIFACT_TYPE = "pams_v16_train337_sshead_variant_sweep"
_RECEIPT_TYPE = "pams_v16_train337_sshead_variant_sweep_receipt"
_CLASSIFICATION = "inferred label-free train337-only SSHead variant sweep"
_EXPECTED_TRAINING_VIDEOS = 337
_EXPECTED_VARIANT_TOTAL = 4
_EXPECTED_ENCODER_CHECKPOINT_SHA256 = (
    "491fd5df67fb4d1fc7669019e4f78fac7b5c681dfa94a4cc7a469da2f73f5196"
)
_EXPECTED_ENCODER_PROGRESS_SHA256 = (
    "047c6af919843cd5d1133033ae07314b8ddd9426d878e4b6519181b94eb13ec7"
)
_EXPECTED_CONFIG_SHA256 = _v16._EXPECTED_CONFIG_SHA256
_EXPECTED_CONFIG_FINGERPRINT = _v16._EXPECTED_CONFIG_FINGERPRINT
_EXPECTED_NONSEED_FINGERPRINT = _v16._EXPECTED_NONSEED_FINGERPRINT
_EXPECTED_POSE_FINGERPRINT = _v16._EXPECTED_POSE_FINGERPRINT
_ALLOWED_INPUT_SOURCES = frozenset({"encoder_embedding", "projected_pose_pre_pe"})
_ALLOWED_VARIANCE_WEIGHTS = frozenset({1.0, 10.0})
_ALLOWED_COMBINATIONS = frozenset(
    (source, weight) for source in _ALLOWED_INPUT_SOURCES for weight in _ALLOWED_VARIANCE_WEIGHTS
)
_EXPECTED_VARIANT_IDENTITIES: dict[str, dict[str, str]] = {
    "encoder_embedding__variance_1": {
        "config_sha256": ("8073a466bd2b6aaca10b482b01b9bec1ed9938cd6c80e8fce964186fcadb6262"),
        "config_fingerprint": ("26b38facffdc510d57f6a849ab6bd5fad80361143962860ccc726edb6ec1969d"),
    },
    "encoder_embedding__variance_10": {
        "config_sha256": ("0424fc7e2beb0839c9c4b194c3df8ac1e57516e8c6bcf74504e8916a97e9f236"),
        "config_fingerprint": ("d4867cb8f238daa597d346a2f6527d064551b03239ceb579db9e35bd416608f8"),
    },
    "projected_pose_pre_pe__variance_1": {
        "config_sha256": ("7ad353188b675409f6366ae63c4f7600cfe4183e9a6b3fe8f5fe4425201e6e94"),
        "config_fingerprint": ("61a32f5dafc3a560405d8782d9718653b5df62d7b8217f02fc0e86424038ff04"),
    },
    "projected_pose_pre_pe__variance_10": {
        "config_sha256": ("a12304bb110fd40541e2eaebd19df9bd8b80b617f09c9437e5c86d194a4986bb"),
        "config_fingerprint": ("68a31ecee2f0eb0a46b979782fec7fa508ac738e33817e8e2def0882b43daa23"),
    },
}


def _variant_id(input_source: str, variance_weight: float) -> str:
    weight = "1" if variance_weight == 1.0 else "10"
    return f"{input_source}__variance_{weight}"


def _input_identities(paths: Mapping[str, Path]) -> dict[str, tuple[str, int]]:
    return {name: _stable_file_sha256(path) for name, path in paths.items()}


def _validate_exact_base_inputs(
    identities: Mapping[str, tuple[str, int]],
) -> None:
    expected = {
        "encoder_checkpoint": _EXPECTED_ENCODER_CHECKPOINT_SHA256,
        "encoder_progress": _EXPECTED_ENCODER_PROGRESS_SHA256,
        "base_config": _EXPECTED_CONFIG_SHA256,
    }
    actual = {name: identities[name][0] for name in expected}
    if actual != expected:
        raise ValueError(
            "SSHead sweep requires the exact frozen v16 encoder/config: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )


def _normalized_config_payload(config: PAMSConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json")
    if not isinstance(payload, dict):
        raise TypeError("validated config did not serialize to an object")
    return payload


def _validate_variant_configs(
    base: PAMSConfig,
    variants: Sequence[tuple[Path, PAMSConfig, str, int]],
) -> tuple[dict[str, Any], ...]:
    """Require exactly one config for every preregistered 2x2 combination."""

    if len(variants) != _EXPECTED_VARIANT_TOTAL:
        raise ValueError("SSHead sweep requires exactly four variant configs")
    base_payload = _normalized_config_payload(base)
    reports: list[dict[str, Any]] = []
    observed: set[tuple[str, float]] = set()
    for path, config, digest, byte_count in variants:
        source = config.sshead.input_source
        weight = float(config.sshead.variance_weight)
        combination = (source, weight)
        if combination not in _ALLOWED_COMBINATIONS:
            raise ValueError(
                "variant is outside the frozen input_source x variance_weight allowlist"
            )
        if combination in observed:
            raise ValueError("SSHead sweep contains a duplicate variant combination")
        candidate_payload = _normalized_config_payload(config)
        normalized_candidate = deepcopy(candidate_payload)
        normalized_candidate["sshead"]["input_source"] = base_payload["sshead"]["input_source"]
        normalized_candidate["sshead"]["variance_weight"] = base_payload["sshead"][
            "variance_weight"
        ]
        if normalized_candidate != base_payload:
            raise ValueError(
                "variant config drifted outside sshead.input_source and sshead.variance_weight"
            )
        variant_id = _variant_id(source, weight)
        expected_identity = _EXPECTED_VARIANT_IDENTITIES[variant_id]
        actual_identity = {
            "config_sha256": digest,
            "config_fingerprint": config.fingerprint,
        }
        if actual_identity != expected_identity:
            raise ValueError(
                "variant config raw SHA/fingerprint differs from the exact "
                f"preregistered {variant_id}: "
                + json.dumps(
                    {
                        "expected": expected_identity,
                        "actual": actual_identity,
                    },
                    sort_keys=True,
                )
            )
        observed.add(combination)
        reports.append(
            {
                "variant_id": variant_id,
                "config_path": str(path.resolve()),
                "config_sha256": digest,
                "config_bytes": byte_count,
                "config_fingerprint": config.fingerprint,
                "nonseed_fingerprint": config.nonseed_fingerprint,
                "input_source": source,
                "variance_weight": weight,
            }
        )
    if observed != _ALLOWED_COMBINATIONS:
        raise ValueError("SSHead sweep does not cover the exact frozen 2x2 matrix")
    return tuple(sorted(reports, key=lambda row: str(row["variant_id"])))


def _variant_provenance(
    encoder: CheckpointProvenance,
    *,
    encoder_checkpoint_sha256: str,
) -> CheckpointProvenance:
    return CheckpointProvenance(
        protocol=encoder.protocol,
        dataset_fingerprint=encoder.dataset_fingerprint,
        training_video_ids=encoder.training_video_ids,
        pose_fingerprint=encoder.pose_fingerprint,
        pose_cache_set_sha256=encoder.pose_cache_set_sha256,
        source_git_sha=encoder.source_git_sha,
        container_image_id=encoder.container_image_id,
        container_environment_sha256=encoder.container_environment_sha256,
        upstream_encoder_checkpoint_sha256=encoder_checkpoint_sha256,
    )


def _fresh_variant_model(
    config: PAMSConfig,
    *,
    upstream_encoder: torch.nn.Module,
) -> torch.nn.Module:
    """Build a deterministic fresh head and copy only upstream encoder tensors."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = build_pams_model(config)
    model.encoder.load_state_dict(upstream_encoder.state_dict(), strict=True)
    return model


def _write_new_regular_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _encoded_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def run_variant_sweep(
    encoder_checkpoint_path: str | Path,
    encoder_progress_path: str | Path,
    base_config_path: str | Path,
    variant_config_paths: Sequence[str | Path],
    pose_cache_dir: str | Path,
    output_root: str | Path,
    *,
    device: str | torch.device | None = None,
    microbatch_size: int | None = None,
) -> dict[str, Any]:
    """Train four fresh deterministic SSHead30 variants on bound train337."""

    if isinstance(variant_config_paths, str | bytes):
        raise TypeError("variant_config_paths must be a sequence of four paths")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"refusing to reuse SSHead sweep output root: {output}")
    variants_paths = tuple(Path(path) for path in variant_config_paths)
    paths: dict[str, Path] = {
        "encoder_checkpoint": Path(encoder_checkpoint_path),
        "encoder_progress": Path(encoder_progress_path),
        "base_config": Path(base_config_path),
        "sweep_runner": Path(__file__),
    }
    paths.update(
        {f"variant_config_{index}": path for index, path in enumerate(variants_paths, start=1)}
    )
    identities = _input_identities(paths)
    _validate_exact_base_inputs(identities)
    base_config = load_config(paths["base_config"])
    _v16._validate_exact_v16_config(
        base_config,
        config_sha256=identities["base_config"][0],
    )
    loaded_variants = tuple(
        (
            path,
            load_config(path),
            identities[f"variant_config_{index}"][0],
            identities[f"variant_config_{index}"][1],
        )
        for index, path in enumerate(variants_paths, start=1)
    )
    variant_specs = _validate_variant_configs(base_config, loaded_variants)
    variant_by_id = {
        _variant_id(
            config.sshead.input_source,
            float(config.sshead.variance_weight),
        ): (path, config)
        for path, config, _, _ in loaded_variants
    }

    encoder_stage, encoder_provenance = _peek_checkpoint(
        paths["encoder_checkpoint"],
        base_config,
    )
    if encoder_stage != "encoder":
        raise ValueError("SSHead sweep requires an encoder checkpoint")
    if len(encoder_provenance.training_video_ids) != _EXPECTED_TRAINING_VIDEOS:
        raise ValueError("SSHead sweep requires exactly train337")
    if encoder_provenance.upstream_encoder_checkpoint_sha256 is not None:
        raise ValueError("base encoder unexpectedly names an upstream checkpoint")
    validate_terminal_checkpoint(
        paths["encoder_checkpoint"],
        base_config,
        expected_stage="encoder",
        expected_provenance=encoder_provenance,
        progress_path=paths["encoder_progress"],
    )
    algorithm_source_git_sha = clean_git_revision(Path.cwd())
    runtime_container = _v14_final._validate_runtime_binding(
        encoder_provenance,
        source_git_sha=algorithm_source_git_sha,
    )
    resolved_device = _device(device)
    sequences, selected_receipts, full_receipts, selected_ids = _load_training_poses(
        pose_cache_dir=Path(pose_cache_dir),
        provenance=encoder_provenance,
        config=base_config,
        sample_size=0,
        seed=base_config.seed,
    )
    if len(sequences) != _EXPECTED_TRAINING_VIDEOS:
        raise RuntimeError("checkpoint-bound pose loader did not return train337")
    selected_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=base_config.pose_fingerprint,
        entries=selected_receipts,
    )
    full_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=base_config.pose_fingerprint,
        entries=full_receipts,
    )
    if selected_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("full train337 selection and pose snapshots differ")
    if full_snapshot.fingerprint != encoder_provenance.pose_cache_set_sha256:
        raise ValueError("train337 pose cache differs from encoder provenance")

    base_model = load_model_checkpoint(
        paths["encoder_checkpoint"],
        base_config,
        device="cpu",
        expected_stage="encoder",
        expected_provenance=encoder_provenance,
    ).eval()
    base_encoder_sha256 = _v16._epoch11._model_state_sha256(base_model.encoder)
    provenance = _variant_provenance(
        encoder_provenance,
        encoder_checkpoint_sha256=identities["encoder_checkpoint"][0],
    )
    output.mkdir(parents=True, exist_ok=False)
    result_rows: list[dict[str, Any]] = []
    for spec in variant_specs:
        variant_id = str(spec["variant_id"])
        _, variant_config = variant_by_id[variant_id]
        variant_root = output / "variants" / variant_id
        checkpoint = variant_root / "sshead.pt"
        progress = variant_root / "logs" / "sshead.jsonl"
        model = _fresh_variant_model(
            variant_config,
            upstream_encoder=base_model.encoder,
        )
        initial_head_sha256 = _v16._epoch11._model_state_sha256(model.period_head)
        if _v16._epoch11._model_state_sha256(model.encoder) != base_encoder_sha256:
            raise RuntimeError("fresh variant encoder differs before SSHead training")
        trained = train_sshead(
            sequences,
            variant_config,
            model=model,
            device=resolved_device,
            microbatch_size=microbatch_size,
            checkpoint_path=checkpoint,
            progress_path=progress,
            resume=False,
            provenance=provenance,
        )
        if trained.completed_epochs != variant_config.sshead.epochs:
            raise RuntimeError("SSHead variant did not complete all 30 epochs")
        validate_terminal_checkpoint(
            checkpoint,
            variant_config,
            expected_stage="sshead",
            expected_provenance=provenance,
            progress_path=progress,
        )
        validate_sshead_encoder_binding(checkpoint, paths["encoder_checkpoint"])
        if _v16._epoch11._model_state_sha256(trained.model.encoder) != base_encoder_sha256:
            raise RuntimeError("SSHead training mutated the frozen encoder")
        checkpoint_identity = _stable_file_sha256(checkpoint)
        progress_identity = _stable_file_sha256(progress)
        result_rows.append(
            {
                **spec,
                "completed_epochs": trained.completed_epochs,
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": checkpoint_identity[0],
                "checkpoint_bytes": checkpoint_identity[1],
                "progress_path": str(progress.resolve()),
                "progress_sha256": progress_identity[0],
                "progress_bytes": progress_identity[1],
                "initial_head_sha256": initial_head_sha256,
                "final_head_sha256": _v16._epoch11._model_state_sha256(trained.model.period_head),
                "embedded_encoder_sha256": base_encoder_sha256,
                "tensor_exact_upstream_encoder_binding_verified": True,
                "provenance": provenance.to_dict(),
            }
        )
        del trained, model
        gc.collect()
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()

    if _v16._epoch11._model_state_sha256(base_model.encoder) != base_encoder_sha256:
        raise RuntimeError("in-memory upstream encoder changed during sweep")
    _, _, final_receipts, final_ids = _load_training_poses(
        pose_cache_dir=Path(pose_cache_dir),
        provenance=encoder_provenance,
        config=base_config,
        sample_size=2,
        seed=base_config.seed,
    )
    final_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=base_config.pose_fingerprint,
        entries=final_receipts,
    )
    if final_ids != selected_ids[:2] or final_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("train337 pose cache changed during SSHead sweep")
    final_identities = _input_identities(paths)
    if final_identities != identities:
        raise RuntimeError("SSHead sweep scientific inputs changed during training")
    if clean_git_revision(Path.cwd()) != algorithm_source_git_sha:
        raise RuntimeError("algorithm source changed during SSHead sweep")

    hardware = hardware_fingerprint()
    runtime = _v14_final._runtime_provenance(
        runner_sha256=identities["sweep_runner"][0],
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
    variant_outputs = {
        row["variant_id"]: {
            "config_sha256": row["config_sha256"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "progress_sha256": row["progress_sha256"],
        }
        for row in result_rows
    }
    payload = {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "status": "train337_sshead_variant_sweep_completed",
        "classification": _CLASSIFICATION,
        "disclosed_by_pams_authors": False,
        "eligible_for_paper_table": False,
        "protocol": base_config.protocol,
        "seed": base_config.seed,
        "label_firewall": {
            "accepted_scientific_inputs": [
                "exact_v16_base_config",
                "exact_terminal_v16_encoder_checkpoint_and_progress",
                "four_preregistered_variant_configs",
                "checkpoint_bound_train337_pose_cache",
            ],
            "dataset_manifest_argument_supported": False,
            "development_identity_media_pose_or_target_argument_supported": False,
            "sealed_test_identity_media_pose_or_target_argument_supported": False,
            "action_class_argument_supported": False,
            "repetition_count_or_label_argument_supported": False,
            "prediction_or_scoring_interface_supported": False,
            "external_label_fields_accessed": [],
        },
        "inputs": {
            **{f"{name}_sha256": value[0] for name, value in identities.items()},
            **{f"{name}_bytes": value[1] for name, value in identities.items()},
            "base_config_fingerprint": base_config.fingerprint,
            "base_nonseed_fingerprint": base_config.nonseed_fingerprint,
            "pose_fingerprint": base_config.pose_fingerprint,
            "encoder_provenance": encoder_provenance.to_dict(),
            "train337_video_total": len(selected_ids),
            "train337_video_ids_sha256": _identifier_commitment(selected_ids),
            "train337_pose_cache_set_sha256": full_snapshot.fingerprint,
            "algorithm_source_git_sha": algorithm_source_git_sha,
            "read_only_post_run_identity_verified": True,
        },
        "frozen_matrix": {
            "input_sources": sorted(_ALLOWED_INPUT_SOURCES),
            "variance_weights": sorted(_ALLOWED_VARIANCE_WEIGHTS),
            "variant_total": _EXPECTED_VARIANT_TOTAL,
            "only_allowed_config_differences": [
                "sshead.input_source",
                "sshead.variance_weight",
            ],
        },
        "variants": result_rows,
        "variant_outputs_sha256": sha256_json(variant_outputs),
        "authorization": {
            "dev84_prediction_authorized": False,
            "dev84_scoring_authorized": False,
            "test105_evaluation_authorized": False,
            "separate_train_only_gate_required": True,
        },
        "hardware": hardware,
        "hardware_sha256": sha256_json(hardware),
        "runtime": runtime,
        "runtime_sha256": sha256_json(runtime),
        "read_only_verification": {
            "all_scientific_file_inputs_unchanged": True,
            "train337_pose_cache_set_sha256_unchanged": True,
            "upstream_encoder_state_sha256": base_encoder_sha256,
            "all_four_embedded_encoders_tensor_exact": True,
            "input_pose_cache_write_operations": 0,
        },
    }
    del base_model
    gc.collect()
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()
    return payload


def _receipt_path(output_root: Path) -> Path:
    return output_root / "sweep.json.receipt.json"


def _artifact_path(output_root: Path) -> Path:
    return output_root / "sweep.json"


def _write_artifact_and_receipt(
    output_root: Path,
    payload: Mapping[str, Any],
) -> tuple[Path, Path, str]:
    artifact_path = _artifact_path(output_root)
    receipt_path = _receipt_path(output_root)
    if artifact_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite sweep artifact or receipt")
    artifact = _encoded_json(payload)
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    _write_new_regular_file(artifact_path, artifact)
    receipt = {
        "schema_version": 1,
        "artifact_type": _RECEIPT_TYPE,
        "artifact_locator": artifact_path.name,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": len(artifact),
        "artifact_status": payload["status"],
        "encoder_checkpoint_sha256": payload["inputs"]["encoder_checkpoint_sha256"],
        "encoder_progress_sha256": payload["inputs"]["encoder_progress_sha256"],
        "base_config_sha256": payload["inputs"]["base_config_sha256"],
        "train337_pose_cache_set_sha256": payload["inputs"]["train337_pose_cache_set_sha256"],
        "variant_outputs_sha256": payload["variant_outputs_sha256"],
        "variant_total": len(payload["variants"]),
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
        "hardware_sha256": payload["hardware_sha256"],
        "runtime_sha256": payload["runtime_sha256"],
    }
    _write_new_regular_file(receipt_path, _encoded_json(receipt))
    return artifact_path, receipt_path, artifact_sha256


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-progress", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument(
        "--variant-config",
        type=Path,
        action="append",
        required=True,
        dest="variant_configs",
    )
    parser.add_argument("--pose-cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--microbatch-size", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    payload = run_variant_sweep(
        arguments.encoder_checkpoint,
        arguments.encoder_progress,
        arguments.base_config,
        arguments.variant_configs,
        arguments.pose_cache_dir,
        arguments.output_root,
        device=arguments.device,
        microbatch_size=arguments.microbatch_size,
    )
    artifact_path, receipt_path, artifact_sha256 = _write_artifact_and_receipt(
        arguments.output_root,
        payload,
    )
    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "artifact_sha256": artifact_sha256,
                "receipt": str(receipt_path),
                "variant_total": len(payload["variants"]),
                "dev84_prediction_authorized": False,
                "dev84_scoring_authorized": False,
                "test105_evaluation_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
