"""Isolated dev84 prediction and scoring for the frozen NoAbsPE readout.

``predict`` accepts only an encoder checkpoint, its exact configuration, the
canonical label-free dev identity/ordering sidecar and commitment, the dev84
pose cache, and an output directory.  It has no target argument.

``score`` first validates and hashes already-frozen prediction bytes and their
receipt.  Only after that validation completes may the separate process open
the dev target manifest.  Neither command has any sealed-test input surface.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

import pams.consensus as consensus_module
import pams.data as data_module
import pams.period as period_module
from pams.config import PAMSConfig, load_config
from pams.consensus import MultiExpertCounter
from pams.data import (
    PoseInputManifest,
    UnlabeledVideoRecord,
    load_dev_target_manifest,
    load_pose_cache_set,
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
    validate_pose_input_binding,
)
from pams.diagnostics import (
    _device,
    _peek_checkpoint,
    _stable_file_sha256,
)
from pams.metrics import compute_count_metrics
from pams.reproducibility import hardware_fingerprint
from pams.training import collate_pose_sequences, load_model_checkpoint
from pams.types import CountResult, PoseSequence


def _load_sibling_predev_module() -> Any:
    source = Path(__file__).resolve().with_name(
        "run_noabspe_pose_reference_harmonic_predev.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_pams_noabspe_pose_reference_harmonic_predev",
        source,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load the sibling predev readout")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


predev = _load_sibling_predev_module()

METHOD_ID = "pams-noabspe-pose-reference-harmonic-multiexpert-inferred-v1"
CLASSIFICATION = "inferred target-free dev diagnostic"
PREDICTION_BATCH_SIZE = 8
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2026
_PREDICTION_NAME = "predictions.json"
_PREDICTION_RECEIPT_NAME = "prediction.receipt.json"
_EVALUATION_NAME = "evaluation.json"
_EVALUATION_RECEIPT_NAME = "evaluation.receipt.json"
_FROZEN_COUNTER_PARAMETERS: dict[str, Any] = {
    "sigma_multipliers": [0.05, 0.12, 0.15],
    "distance_multipliers": [0.5, 0.8, 1.2],
    "short_window_multiplier": 0.5,
    "long_window_multiplier": 2.0,
    "height_factor": 0.6,
    "prominence_factor": 0.25,
    "long_window_weight": 0.6,
    "expert_mode": "multi",
}
_EXPECTED_PREDEV_READOUT = {
    "embedding_evidence": "legacy_embedding_velocity_fft",
    "pose_reference_evidence": "raw_pose_energy_fft",
    "candidate_periods": "h * embedding_period for h=1..8 within [4,128]",
    "selection": (
        "minimum absolute log-ratio distance to pose-energy period; "
        "relative-distance and smaller-factor tie breaks"
    ),
    "confidence": "embedding_confidence * pose_energy_confidence",
    "legacy_estimator_modified": False,
}
_EXPECTED_PREDEV_CRITERIA = {
    "frame_index_r2": ("<=", 0.10),
    "zero_pose_period_confidence": ("<=", 0.10),
    "random_pose_period_confidence": ("<=", 0.10),
    "training_period_top_bin_share": ("<=", 0.25),
    "canonical_permuted_embedding_median_cosine": (">=", 0.95),
    "synthetic_period_median_relative_error": ("<=", 0.10),
}
_PREDEV_CRITERIA = set(_EXPECTED_PREDEV_CRITERIA)


def _sha256_file(path: str | Path) -> str:
    return _stable_file_sha256(Path(path))[0]


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json_object(path: Path, *, document: str) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant in {document}: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{document} root must be an object")
    return payload


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    with path.open("xb") as handle:
        handle.write(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _module_path(module: Any) -> Path:
    source = getattr(module, "__file__", None)
    if not source:
        raise RuntimeError(f"source path unavailable for module {module!r}")
    return Path(source).resolve()


def _source_hashes() -> dict[str, str]:
    paths = {
        "dev_runner": Path(__file__).resolve(),
        "predev_readout": _module_path(predev),
        "pams_consensus": _module_path(consensus_module),
        "pams_data": _module_path(data_module),
        "pams_period": _module_path(period_module),
    }
    return {name: _sha256_file(path) for name, path in paths.items()}


def _algorithm_parameters() -> dict[str, Any]:
    return {
        "method_id": METHOD_ID,
        "prediction_batch_size": PREDICTION_BATCH_SIZE,
        "period_readout": {
            "embedding_evidence": "legacy_embedding_velocity_fft",
            "pose_reference_evidence": "raw_pose_energy_fft",
            "harmonic_factors": list(range(1, 9)),
            "period_bounds_frames": [4, 128],
            "selection": (
                "minimum absolute log-ratio distance; 12-decimal numerical "
                "tie, relative-distance tie, then smaller harmonic factor"
            ),
            "confidence": "embedding_confidence_times_pose_energy_confidence",
        },
        "counting_stream": "signed_highest_variance_raw_pose_energy",
        "counter": _FROZEN_COUNTER_PARAMETERS,
        "rounding": "counter integer peak counts; no target-conditioned rounding",
    }


def _load_bound_dev_input(
    sidecar_path: Path,
    commitment_path: Path,
) -> tuple[PoseInputManifest, Any, str, str]:
    sidecar_sha256 = _sha256_file(sidecar_path)
    commitment_sha256 = _sha256_file(commitment_path)
    sidecar = load_pose_input_manifest(sidecar_path, validate_exact=True)
    commitment = load_pose_input_commitment(commitment_path)
    validate_pose_input_binding(
        sidecar,
        commitment,
        sidecar_sha256=sidecar_sha256,
    )
    if sidecar.protocol != "ucfrep_526" or sidecar.split != "dev":
        raise ValueError("prediction requires the canonical ucfrep_526 dev sidecar")
    if len(sidecar.records) != 84:
        raise ValueError("prediction requires exactly 84 dev identities")
    if _sha256_file(sidecar_path) != sidecar_sha256:
        raise RuntimeError("dev sidecar changed while it was loaded")
    if _sha256_file(commitment_path) != commitment_sha256:
        raise RuntimeError("dev commitment changed while it was loaded")
    return sidecar, commitment, sidecar_sha256, commitment_sha256


def _validate_passed_predev_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    config_sha256: str,
    config: PAMSConfig,
) -> None:
    """Require a passed gate for these exact bytes and frozen readout."""

    if payload.get("artifact_type") != (
        "pams_noabspe_pose_reference_harmonic_predev_gate"
    ):
        raise ValueError("passed-predev artifact_type is not the pose-reference gate")
    if payload.get("classification") != "inferred target-free predev diagnostic":
        raise ValueError("passed-predev classification mismatch")
    if payload.get("disclosed_by_pams_authors") is not False:
        raise ValueError("passed-predev must remain independently inferred")
    if payload.get("eligible_for_paper_table") is not False:
        raise ValueError("passed-predev cannot be paper-table eligible")
    if payload.get("readout") != _EXPECTED_PREDEV_READOUT:
        raise ValueError("passed-predev readout semantics differ from current resolver")

    inputs = payload.get("inputs")
    if not isinstance(inputs, Mapping):
        raise ValueError("passed-predev inputs must be an object")
    expected_inputs = {
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "config_fingerprint": config.fingerprint,
        "pose_fingerprint": config.pose_fingerprint,
        "position_encoding_mode": "none",
        "checkpoint_stage": "encoder",
    }
    mismatched = [
        field
        for field, expected in expected_inputs.items()
        if inputs.get(field) != expected
    ]
    if mismatched:
        raise ValueError(f"passed-predev input binding mismatch: {mismatched}")

    gate = payload.get("gate")
    if not isinstance(gate, Mapping):
        raise ValueError("passed-predev gate must be an object")
    expected_flags = {
        "thresholds_frozen_before_checkpoint_evaluation": True,
        "overall_pass": True,
        "dev84_prediction_authorized": True,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }
    mismatched_flags = [
        field
        for field, expected in expected_flags.items()
        if gate.get(field) is not expected
    ]
    if mismatched_flags:
        raise ValueError(f"passed-predev authorization mismatch: {mismatched_flags}")
    criteria = gate.get("criteria")
    if not isinstance(criteria, Mapping) or set(criteria) != _PREDEV_CRITERIA:
        raise ValueError("passed-predev must contain exactly the six frozen criteria")
    for name, item in criteria.items():
        if not isinstance(item, Mapping) or item.get("pass") is not True:
            raise ValueError(f"passed-predev criterion did not pass: {name}")
        value = item.get("value")
        threshold = item.get("threshold")
        expected_operator, expected_threshold = _EXPECTED_PREDEV_CRITERIA[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or item.get("operator") != expected_operator
            or float(threshold) != expected_threshold
        ):
            raise ValueError(f"passed-predev criterion is malformed: {name}")
        numeric_value = float(value)
        satisfied = (
            numeric_value <= expected_threshold
            if expected_operator == "<="
            else numeric_value >= expected_threshold
        )
        if not satisfied:
            raise ValueError(f"passed-predev criterion value failed: {name}")

    synthetic = payload.get("synthetic_period_recovery")
    if not isinstance(synthetic, Mapping):
        raise ValueError("passed-predev synthetic recovery must be an object")
    if (
        synthetic.get("embedding_estimator") != "legacy_embedding_velocity_fft"
        or synthetic.get("pose_reference_estimator") != "raw_pose_energy_fft"
        or synthetic.get("maximum_harmonic") != 8
        or synthetic.get("periods") != [4, 8, 16, 32, 64, 128]
    ):
        raise ValueError("passed-predev synthetic readout semantics mismatch")
    rows = synthetic.get("rows")
    if not isinstance(rows, list) or len(rows) != 6:
        raise ValueError("passed-predev requires six frozen synthetic probes")
    synthetic_errors: list[float] = []
    for expected, row in zip((4, 8, 16, 32, 64, 128), rows, strict=True):
        if (
            not isinstance(row, Mapping)
            or row.get("expected_period_frames") != expected
            or row.get("confidence_rule")
            != "embedding_confidence_times_pose_energy_confidence"
            or row.get("selected_harmonic_factor") not in range(1, 9)
        ):
            raise ValueError("passed-predev synthetic evidence is malformed")
        selected = row.get("selected_period_frames")
        reported_error = row.get("relative_error")
        if (
            isinstance(selected, bool)
            or not isinstance(selected, (int, float))
            or not math.isfinite(float(selected))
            or isinstance(reported_error, bool)
            or not isinstance(reported_error, (int, float))
            or not math.isfinite(float(reported_error))
        ):
            raise ValueError("passed-predev synthetic evidence is non-finite")
        computed_error = abs(float(selected) - expected) / expected
        if not math.isclose(
            float(reported_error),
            computed_error,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError("passed-predev synthetic relative error mismatch")
        synthetic_errors.append(computed_error)
    synthetic_median = float(np.median(np.asarray(synthetic_errors)))
    criterion_median = float(
        criteria["synthetic_period_median_relative_error"]["value"]
    )
    if not math.isclose(
        synthetic_median,
        criterion_median,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError("passed-predev synthetic median mismatch")

    read_only = payload.get("read_only_verification")
    if not isinstance(read_only, Mapping) or read_only != {
        "checkpoint_sha256_unchanged": True,
        "config_sha256_unchanged": True,
        "model_or_training_state_updated": False,
        "pose_cache_write_operations": 0,
    }:
        raise ValueError("passed-predev read-only verification mismatch")


def _load_passed_predev(
    path: Path,
    *,
    checkpoint_sha256: str,
    config_sha256: str,
    config: PAMSConfig,
) -> tuple[dict[str, Any], str]:
    artifact_sha256 = _sha256_file(path)
    payload = _load_json_object(path, document="passed predev artifact")
    _validate_passed_predev_payload(
        payload,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
        config=config,
    )
    if _sha256_file(path) != artifact_sha256:
        raise RuntimeError("passed-predev artifact changed while it was validated")
    return payload, artifact_sha256


def _validated_counter(config: PAMSConfig) -> MultiExpertCounter:
    actual = {
        "sigma_multipliers": list(config.consensus.sigma_multipliers),
        "distance_multipliers": list(config.consensus.distance_multipliers),
        "short_window_multiplier": config.consensus.short_window_multiplier,
        "long_window_multiplier": config.consensus.long_window_multiplier,
        "height_factor": config.consensus.height_factor,
        "prominence_factor": config.consensus.prominence_factor,
        "long_window_weight": config.consensus.long_window_weight,
        "expert_mode": config.consensus.expert_mode,
    }
    if actual != _FROZEN_COUNTER_PARAMETERS:
        raise ValueError("configuration differs from frozen multi-expert defaults")
    return MultiExpertCounter(
        sigma_multipliers=config.consensus.sigma_multipliers,
        distance_multipliers=config.consensus.distance_multipliers,
        short_window_multiplier=config.consensus.short_window_multiplier,
        long_window_multiplier=config.consensus.long_window_multiplier,
        height_factor=config.consensus.height_factor,
        prominence_factor=config.consensus.prominence_factor,
        long_window_weight=config.consensus.long_window_weight,
        expert_mode=config.consensus.expert_mode,
    )


def _prediction_record(
    sequence: PoseSequence,
    result: CountResult,
    evidence: Mapping[str, Any],
    *,
    video_sha256: str,
    pose_cache_sha256: str,
) -> dict[str, Any]:
    return {
        "video_id": sequence.video_id,
        "video_sha256": video_sha256,
        "pose_cache_sha256": pose_cache_sha256,
        **result.to_dict(include_stream=True),
        "valid_frames": int(np.sum(sequence.valid_mask)),
        "period_evidence": dict(evidence),
    }


def predict_pose_reference_harmonic_batches(
    model: Any,
    sequences: Sequence[PoseSequence],
    config: PAMSConfig,
    *,
    video_sha256: Mapping[str, str],
    pose_cache_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], ...]:
    """Produce frozen multi-expert counts from label-free pose sequences."""

    items = tuple(sequences)
    if not items:
        raise ValueError("prediction requires at least one pose sequence")
    counter = _validated_counter(config)
    try:
        model_device = next(model.parameters()).device
    except StopIteration:
        model_device = torch.device("cpu")
    model.eval()
    records: list[dict[str, Any]] = []
    for start in range(0, len(items), PREDICTION_BATCH_SIZE):
        batch_sequences = items[start : start + PREDICTION_BATCH_SIZE]
        batch = collate_pose_sequences(batch_sequences).to(model_device)
        _, evidence = predev._readout_batch(
            model,
            batch.poses,
            batch.valid_mask,
            config=config,
        )
        with torch.inference_mode():
            streams = period_module.pose_energy(batch.poses, batch.valid_mask)
        for index, (sequence, row) in enumerate(
            zip(batch_sequences, evidence, strict=True)
        ):
            length = sequence.num_frames
            result = counter.count(
                streams[index, :length],
                period_frames=float(row["selected_period_frames"]),
                valid_mask=batch.valid_mask[index, :length],
                period_confidence=float(row["selected_period_confidence"]),
            ).to_count_result()
            records.append(
                _prediction_record(
                    sequence,
                    result,
                    row,
                    video_sha256=video_sha256[sequence.video_id],
                    pose_cache_sha256=pose_cache_sha256[sequence.video_id],
                )
            )
    return tuple(records)


def run_predict(arguments: argparse.Namespace) -> dict[str, Any]:
    """Freeze dev84 predictions without opening or accepting a target file."""

    checkpoint = arguments.checkpoint.resolve()
    config_path = arguments.config.resolve()
    passed_predev_path = arguments.passed_predev.resolve()
    dev_inputs = arguments.dev_inputs.resolve()
    dev_commitment = arguments.dev_commitment.resolve()
    pose_cache_dir = arguments.pose_cache_dir.resolve()
    output_dir = arguments.output_dir.resolve()
    predictions_path = output_dir / _PREDICTION_NAME
    receipt_path = output_dir / _PREDICTION_RECEIPT_NAME
    collisions = [str(path) for path in (predictions_path, receipt_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite prediction artifacts: {collisions}")

    source_hashes = _source_hashes()
    source_code_sha256 = _sha256_json(source_hashes)
    parameters = _algorithm_parameters()
    parameter_sha256 = _sha256_json(parameters)
    checkpoint_sha256, checkpoint_bytes = _stable_file_sha256(checkpoint)
    config_sha256, config_bytes = _stable_file_sha256(config_path)
    config = load_config(config_path)
    predev._validate_noabspe_config(config)
    _validated_counter(config)
    passed_predev, passed_predev_sha256 = _load_passed_predev(
        passed_predev_path,
        checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256,
        config=config,
    )
    stage, provenance = _peek_checkpoint(checkpoint, config)
    if stage != "encoder":
        raise ValueError("prediction requires a frozen encoder checkpoint")
    device = _device(arguments.device)
    model = load_model_checkpoint(
        checkpoint,
        config,
        device=device,
        expected_stage="encoder",
        expected_provenance=provenance,
    )
    model.eval()

    sidecar, commitment, sidecar_sha256, commitment_sha256 = _load_bound_dev_input(
        dev_inputs,
        dev_commitment,
    )
    sequences, pose_snapshot = load_pose_cache_set(
        sidecar.records,
        cache_dir=pose_cache_dir,
        pose_fingerprint=config.pose_fingerprint,
    )
    expected_ids = tuple(record.video_id for record in sidecar.records)
    if tuple(sequence.video_id for sequence in sequences) != expected_ids:
        raise RuntimeError("dev pose-cache order differs from committed sidecar")
    video_hashes = {
        record.video_id: str(record.video_sha256) for record in sidecar.records
    }
    cache_hashes = {
        entry.video_id: entry.cache_sha256 for entry in pose_snapshot.entries
    }
    records = predict_pose_reference_harmonic_batches(
        model,
        sequences,
        config,
        video_sha256=video_hashes,
        pose_cache_sha256=cache_hashes,
    )
    if tuple(str(record["video_id"]) for record in records) != expected_ids:
        raise RuntimeError("prediction order differs from committed dev sidecar")

    if _stable_file_sha256(checkpoint) != (checkpoint_sha256, checkpoint_bytes):
        raise RuntimeError("checkpoint changed during prediction")
    if _stable_file_sha256(config_path) != (config_sha256, config_bytes):
        raise RuntimeError("configuration changed during prediction")
    if _sha256_file(dev_inputs) != sidecar_sha256:
        raise RuntimeError("dev sidecar changed during prediction")
    if _sha256_file(dev_commitment) != commitment_sha256:
        raise RuntimeError("dev commitment changed during prediction")
    if _sha256_file(passed_predev_path) != passed_predev_sha256:
        raise RuntimeError("passed-predev artifact changed during prediction")
    if _source_hashes() != source_hashes:
        raise RuntimeError("prediction source changed during prediction")

    payload = {
        "schema_version": 1,
        "artifact_type": "pams_noabspe_pose_reference_harmonic_dev_predictions",
        "method_id": METHOD_ID,
        "classification": CLASSIFICATION,
        "diagnostic_only": True,
        "eligible_for_paper_table": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "record_total": 84,
        "source_files_sha256": source_hashes,
        "source_code_sha256": source_code_sha256,
        "predev_source_sha256": source_hashes["predev_readout"],
        "algorithm_parameters": parameters,
        "algorithm_parameters_sha256": parameter_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_source_git_sha": provenance.source_git_sha,
        "config_sha256": config_sha256,
        "config_bytes": config_bytes,
        "config_fingerprint": config.fingerprint,
        "pose_fingerprint": config.pose_fingerprint,
        "passed_predev_sha256": passed_predev_sha256,
        "passed_predev_authorization": {
            "overall_pass": passed_predev["gate"]["overall_pass"],
            "dev84_prediction_authorized": (
                passed_predev["gate"]["dev84_prediction_authorized"]
            ),
            "dev84_scoring_authorized": (
                passed_predev["gate"]["dev84_scoring_authorized"]
            ),
            "test105_evaluation_authorized": (
                passed_predev["gate"]["test105_evaluation_authorized"]
            ),
        },
        "dev_inputs_sha256": sidecar_sha256,
        "dev_commitment_sha256": commitment_sha256,
        "dev_identity_sha256": commitment.identity_sha256,
        "dev_pose_cache_set_sha256": pose_snapshot.fingerprint,
        "runtime": hardware_fingerprint(),
        "mount_audit": {
            "network": "none",
            "checkpoint_mounted": True,
            "config_mounted": True,
            "passed_predev_mounted": True,
            "dev_identity_mounted": True,
            "dev_pose_mounted": True,
            "dev_targets_mounted": False,
            "test_identity_mounted": False,
            "test_pose_mounted": False,
            "test_targets_mounted": False,
        },
        "records": list(records),
    }
    prediction_sha256 = _write_json_exclusive(predictions_path, payload)
    receipt = {
        "schema_version": 1,
        "artifact_type": (
            "pams_noabspe_pose_reference_harmonic_dev_prediction_receipt"
        ),
        "method_id": METHOD_ID,
        "protocol": "ucfrep_526",
        "split": "dev",
        "prediction_file": _PREDICTION_NAME,
        "prediction_sha256": prediction_sha256,
        "prediction_bytes": predictions_path.stat().st_size,
        "record_total": 84,
        "source_code_sha256": source_code_sha256,
        "predev_source_sha256": source_hashes["predev_readout"],
        "algorithm_parameters_sha256": parameter_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "config_sha256": config_sha256,
        "passed_predev_sha256": passed_predev_sha256,
        "dev_inputs_sha256": sidecar_sha256,
        "dev_commitment_sha256": commitment_sha256,
        "dev_identity_sha256": commitment.identity_sha256,
        "dev_pose_cache_set_sha256": pose_snapshot.fingerprint,
        "label_firewall": {
            "prediction_loaded_dev_targets": False,
            "prediction_loaded_test_identity": False,
            "prediction_loaded_test_targets": False,
            "gt_count_or_action_oracle_used": False,
        },
    }
    receipt_sha256 = _write_json_exclusive(receipt_path, receipt)
    return {
        "method_id": METHOD_ID,
        "record_total": 84,
        "predictions_path": str(predictions_path),
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_path": str(receipt_path),
        "prediction_receipt_sha256": receipt_sha256,
        "dev_scoring_authorized": True,
        "test105_evaluation_authorized": False,
    }


def _validated_count_result(row: Mapping[str, Any]) -> CountResult:
    experts = row.get("expert_counts")
    if not isinstance(experts, list) or len(experts) != 3:
        raise ValueError("prediction expert_counts must contain three values")
    stream = row.get("period_stream")
    if not isinstance(stream, list) or len(stream) != 256:
        raise ValueError("prediction period_stream must contain 256 values")
    return CountResult(
        count=row.get("count"),  # type: ignore[arg-type]
        period_frames=float(row.get("period_frames", math.nan)),
        expert_counts=tuple(experts),  # type: ignore[arg-type]
        confidence=float(row.get("confidence", math.nan)),
        period_stream=np.asarray(stream, dtype=np.float32),
    )


def _validate_canonical_identity(payload: Mapping[str, Any]) -> None:
    rows = payload["records"]
    records = tuple(
        UnlabeledVideoRecord(
            video_id=str(row["video_id"]),
            video_path=f"frozen-dev/{row['video_id']}.avi",
            video_sha256=str(row["video_sha256"]),
        )
        for row in rows
    )
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split="dev",
        records=records,
    )
    manifest.validate_exact_membership()
    if pose_input_identity_sha256(records) != payload["dev_identity_sha256"]:
        raise ValueError("prediction rows do not bind the declared dev identity")


def _validate_frozen_predictions(
    predictions_path: Path,
    receipt_path: Path,
) -> tuple[dict[str, Any], str, str]:
    prediction_sha256 = _sha256_file(predictions_path)
    receipt_sha256 = _sha256_file(receipt_path)
    predictions = _load_json_object(predictions_path, document="predictions")
    receipt = _load_json_object(receipt_path, document="prediction receipt")
    if predictions.get("artifact_type") != (
        "pams_noabspe_pose_reference_harmonic_dev_predictions"
    ):
        raise ValueError("unexpected prediction artifact_type")
    if receipt.get("artifact_type") != (
        "pams_noabspe_pose_reference_harmonic_dev_prediction_receipt"
    ):
        raise ValueError("unexpected prediction receipt artifact_type")
    if predictions.get("method_id") != METHOD_ID or receipt.get("method_id") != METHOD_ID:
        raise ValueError("prediction method identity mismatch")
    if predictions.get("protocol") != "ucfrep_526" or predictions.get("split") != "dev":
        raise ValueError("prediction protocol/split mismatch")
    if receipt.get("prediction_sha256") != prediction_sha256:
        raise ValueError("prediction receipt SHA-256 mismatch")
    if receipt.get("prediction_bytes") != predictions_path.stat().st_size:
        raise ValueError("prediction receipt byte count mismatch")
    if predictions.get("source_files_sha256") != _source_hashes():
        raise ValueError("scoring source differs from prediction source")
    if predictions.get("source_code_sha256") != _sha256_json(_source_hashes()):
        raise ValueError("prediction aggregate source hash mismatch")
    if predictions.get("algorithm_parameters") != _algorithm_parameters():
        raise ValueError("prediction algorithm parameters drifted")
    if predictions.get("algorithm_parameters_sha256") != _sha256_json(
        _algorithm_parameters()
    ):
        raise ValueError("prediction algorithm parameter hash mismatch")
    for field in (
        "source_code_sha256",
        "predev_source_sha256",
        "algorithm_parameters_sha256",
        "checkpoint_sha256",
        "config_sha256",
        "dev_inputs_sha256",
        "dev_commitment_sha256",
        "dev_identity_sha256",
        "dev_pose_cache_set_sha256",
        "passed_predev_sha256",
    ):
        if receipt.get(field) != predictions.get(field):
            raise ValueError(f"prediction receipt metadata mismatch: {field}")
    rows = predictions.get("records")
    if not isinstance(rows, list) or len(rows) != 84:
        raise ValueError("prediction artifact must contain exactly 84 records")
    identifiers: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("prediction rows must be objects")
        if "target" in row or "action" in row:
            raise ValueError("prediction rows must remain target-free")
        identifier = str(row.get("video_id", ""))
        if not identifier:
            raise ValueError("prediction video_id must be non-empty")
        identifiers.append(identifier)
        _validated_count_result(row)
        evidence = row.get("period_evidence")
        if not isinstance(evidence, dict):
            raise ValueError("prediction period_evidence must be an object")
        for field in (
            "embedding_period_frames",
            "embedding_period_confidence",
            "pose_energy_period_frames",
            "pose_energy_period_confidence",
            "selected_harmonic_factor",
            "selected_period_frames",
            "selected_period_confidence",
        ):
            if field not in evidence:
                raise ValueError(f"prediction period evidence is missing {field}")
    if len(set(identifiers)) != 84:
        raise ValueError("prediction video_id values must be unique")
    _validate_canonical_identity(predictions)
    if _sha256_file(predictions_path) != prediction_sha256:
        raise RuntimeError("predictions changed while being validated")
    if _sha256_file(receipt_path) != receipt_sha256:
        raise RuntimeError("prediction receipt changed while being validated")
    return predictions, prediction_sha256, receipt_sha256


def run_score(arguments: argparse.Namespace) -> dict[str, Any]:
    """Score frozen predictions at the only label-bearing boundary."""

    predictions_path = arguments.predictions.resolve()
    receipt_path = arguments.prediction_receipt.resolve()
    output_dir = arguments.output_dir.resolve()
    evaluation_path = output_dir / _EVALUATION_NAME
    evaluation_receipt_path = output_dir / _EVALUATION_RECEIPT_NAME
    collisions = [
        str(path)
        for path in (evaluation_path, evaluation_receipt_path)
        if path.exists()
    ]
    if collisions:
        raise FileExistsError(f"refusing to overwrite evaluation artifacts: {collisions}")

    # This complete validator does not stat, hash, or deserialize labels.
    predictions, prediction_sha256, receipt_sha256 = _validate_frozen_predictions(
        predictions_path,
        receipt_path,
    )
    frozen_source_hashes = _source_hashes()

    # First permitted target access: predictions and receipt are already frozen.
    targets_path = arguments.dev_targets.resolve()
    dev_targets_sha256 = _sha256_file(targets_path)
    targets = load_dev_target_manifest(targets_path)
    rows = predictions["records"]
    prediction_ids = tuple(str(row["video_id"]) for row in rows)
    target_ids = tuple(record.video_id for record in targets.records)
    if prediction_ids != target_ids:
        raise ValueError("dev target order/identity differs from frozen predictions")
    if _sha256_file(targets_path) != dev_targets_sha256:
        raise RuntimeError("dev targets changed while being loaded")

    report = compute_count_metrics(
        [int(row["count"]) for row in rows],
        [record.count for record in targets.records],
        video_ids=prediction_ids,
        actions=[record.action for record in targets.records],
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        confidence_level=0.95,
    )
    if _sha256_file(predictions_path) != prediction_sha256:
        raise RuntimeError("predictions changed during scoring")
    if _sha256_file(receipt_path) != receipt_sha256:
        raise RuntimeError("prediction receipt changed during scoring")
    if _sha256_file(targets_path) != dev_targets_sha256:
        raise RuntimeError("dev targets changed during scoring")
    if _source_hashes() != frozen_source_hashes:
        raise RuntimeError("scoring source changed during scoring")

    evaluation = {
        "schema_version": 1,
        "artifact_type": (
            "pams_noabspe_pose_reference_harmonic_dev_evaluation"
        ),
        "method_id": METHOD_ID,
        "classification": CLASSIFICATION,
        "diagnostic_only": True,
        "eligible_for_paper_table": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "prediction_sha256": prediction_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "dev_targets_sha256": dev_targets_sha256,
        "bootstrap_pairing": "paired_prediction_target_rows",
        "metrics": report.to_dict(),
        "acceptance_gate": {
            "required_nmae_at_most": 0.228,
            "required_obo_at_least": 0.666,
            "nmae_pass": report.nmae <= 0.228,
            "obo_pass": report.obo >= 0.666,
            "joint_pass": report.nmae <= 0.228 and report.obo >= 0.666,
            "verified_reproduction": False,
        },
        "mount_audit": {
            "network": "none",
            "prediction_mounted": True,
            "dev_targets_mounted": True,
            "dev_pose_mounted": False,
            "checkpoint_mounted": False,
            "test_identity_mounted": False,
            "test_pose_mounted": False,
            "test_targets_mounted": False,
        },
    }
    evaluation_sha256 = _write_json_exclusive(evaluation_path, evaluation)
    evaluation_receipt_sha256 = _write_json_exclusive(
        evaluation_receipt_path,
        {
            "schema_version": 1,
            "artifact_type": (
                "pams_noabspe_pose_reference_harmonic_dev_evaluation_receipt"
            ),
            "method_id": METHOD_ID,
            "evaluation_sha256": evaluation_sha256,
            "evaluation_bytes": evaluation_path.stat().st_size,
            "prediction_sha256": prediction_sha256,
            "prediction_receipt_sha256": receipt_sha256,
            "dev_targets_sha256": dev_targets_sha256,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "test105_evaluation_authorized": False,
        },
    )
    metrics = report.to_dict()
    metrics.pop("per_video")
    return {
        "method_id": METHOD_ID,
        "evaluation_path": str(evaluation_path),
        "evaluation_sha256": evaluation_sha256,
        "evaluation_receipt_path": str(evaluation_receipt_path),
        "evaluation_receipt_sha256": evaluation_receipt_sha256,
        "prediction_sha256": prediction_sha256,
        "dev_targets_sha256": dev_targets_sha256,
        "metrics": metrics,
        "test105_evaluation_authorized": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--config", type=Path, required=True)
    predict.add_argument("--passed-predev", type=Path, required=True)
    predict.add_argument("--dev-inputs", type=Path, required=True)
    predict.add_argument("--dev-commitment", type=Path, required=True)
    predict.add_argument("--pose-cache-dir", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--device", default="auto")
    predict.set_defaults(handler=run_predict)

    score = subparsers.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--prediction-receipt", type=Path, required=True)
    score.add_argument("--dev-targets", type=Path, required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.set_defaults(handler=run_score)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    payload = arguments.handler(arguments)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
