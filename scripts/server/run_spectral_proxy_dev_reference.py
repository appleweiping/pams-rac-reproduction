"""Strict UCFRep-dev diagnostic for the deterministic spectral proxy.

This runner intentionally keeps protocol identity validation, label-free
prediction, and dev-only scoring in separate processes.  It must never be used
to present ``spectral-proxy`` as a paper baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pams.baselines.proxy import SpectralProxyAdapter
from pams.data import (
    LabelFreeProtocolInputs,
    load_dev_target_manifest,
    load_pose_cache_set,
    load_pose_input_commitment,
    load_pose_input_manifest,
    validate_pose_input_binding,
)
from pams.metrics import compute_count_metrics

METHOD_ID = "spectral-proxy"
EXPECTED_SOURCE_SHA = "d2248a17c0ec25c4b0a3d3db8104a713dff9aad0"
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 2026


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_exclusive(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        handle.write("\n")


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _require_source_sha(value: str) -> str:
    normalized = value.strip().lower()
    if normalized != EXPECTED_SOURCE_SHA:
        raise ValueError(
            f"this runner is frozen for source {EXPECTED_SOURCE_SHA}, received {value!r}"
        )
    return normalized


def _source_code_hashes(repository: Path, runner: Path) -> dict[str, str]:
    relative_files = (
        "src/pams/baselines/base.py",
        "src/pams/baselines/proxy.py",
        "src/pams/data.py",
        "src/pams/metrics.py",
        "src/pams/types.py",
    )
    hashes = {
        relative: _sha256_file(repository / relative) for relative in relative_files
    }
    hashes["scripts/server/run_spectral_proxy_dev_reference.py"] = _sha256_file(runner)
    return hashes


def _load_bound_input(
    sidecar_path: Path,
    commitment_path: Path,
):
    sidecar_sha256 = _sha256_file(sidecar_path)
    manifest = load_pose_input_manifest(sidecar_path)
    commitment = load_pose_input_commitment(commitment_path)
    validate_pose_input_binding(
        manifest,
        commitment,
        sidecar_sha256=sidecar_sha256,
    )
    if _sha256_file(sidecar_path) != sidecar_sha256:
        raise RuntimeError("pose-input sidecar changed while it was loaded")
    return manifest, commitment, sidecar_sha256


def _runtime_payload(*, container_image_id: str) -> dict[str, Any]:
    affinity = getattr(os, "sched_getaffinity", None)
    visible_cpu_count = (
        len(affinity(0)) if callable(affinity) else (os.cpu_count() or 0)
    )
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "visible_cpu_count": visible_cpu_count,
        "container_image_id": container_image_id,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_visible_devices": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "network_required": False,
        "inference_device": "cpu",
    }


def run_identity(arguments: argparse.Namespace) -> None:
    source_sha = _require_source_sha(arguments.source_git_sha)
    train, train_commitment, train_sha = _load_bound_input(
        arguments.train_inputs,
        arguments.train_commitment,
    )
    dev, dev_commitment, dev_sha = _load_bound_input(
        arguments.dev_inputs,
        arguments.dev_commitment,
    )
    test, test_commitment, test_sha = _load_bound_input(
        arguments.test_inputs,
        arguments.test_commitment,
    )
    protocol = LabelFreeProtocolInputs(
        protocol="ucfrep_526",
        train=train,
        dev=dev,
        test=test,
    )
    payload = {
        "schema_version": 1,
        "artifact_type": "label_free_protocol_identity_receipt",
        "created_at_utc": _utc_now(),
        "protocol": protocol.protocol,
        "source_git_sha": source_sha,
        "split_counts": protocol.split_counts,
        "protocol_identity_sha256": protocol.fingerprint,
        "training_identity_sha256": protocol.training_fingerprint(),
        "dev_identity_sha256": dev_commitment.identity_sha256,
        "test_identity_sha256": test_commitment.identity_sha256,
        "sidecars": {
            "train": {
                "sha256": train_sha,
                "commitment_sha256": _sha256_file(arguments.train_commitment),
                "fingerprint": train.fingerprint,
            },
            "dev": {
                "sha256": dev_sha,
                "commitment_sha256": _sha256_file(arguments.dev_commitment),
                "fingerprint": dev.fingerprint,
            },
            "test": {
                "sha256": test_sha,
                "commitment_sha256": _sha256_file(arguments.test_commitment),
                "fingerprint": test.fingerprint,
            },
        },
        "checks": {
            "canonical_337_84_105": protocol.split_counts
            == {"train": 337, "dev": 84, "test": 105},
            "identities_paths_and_video_hashes_disjoint": True,
            "count_or_action_fields_loaded": False,
            "test_targets_loaded": False,
        },
    }
    _write_json_exclusive(arguments.output, payload)


def _validate_identity_receipt(
    identity_receipt_path: Path,
    *,
    source_sha: str,
    dev_inputs_sha256: str,
    dev_commitment_sha256: str,
) -> tuple[dict[str, Any], str]:
    receipt_sha256 = _sha256_file(identity_receipt_path)
    payload = json.loads(identity_receipt_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("identity receipt must be a JSON object")
    if payload.get("artifact_type") != "label_free_protocol_identity_receipt":
        raise ValueError("unexpected identity receipt artifact_type")
    if payload.get("source_git_sha") != source_sha:
        raise ValueError("identity receipt source revision mismatch")
    if payload.get("split_counts") != {"dev": 84, "test": 105, "train": 337}:
        raise ValueError("identity receipt does not bind canonical 337/84/105 splits")
    sidecars = payload.get("sidecars")
    if not isinstance(sidecars, dict) or not isinstance(sidecars.get("dev"), dict):
        raise ValueError("identity receipt is missing dev sidecar bindings")
    if sidecars["dev"].get("sha256") != dev_inputs_sha256:
        raise ValueError("identity receipt dev sidecar SHA-256 mismatch")
    if sidecars["dev"].get("commitment_sha256") != dev_commitment_sha256:
        raise ValueError("identity receipt dev commitment SHA-256 mismatch")
    if _sha256_file(identity_receipt_path) != receipt_sha256:
        raise RuntimeError("identity receipt changed while it was loaded")
    return payload, receipt_sha256


def run_predict(arguments: argparse.Namespace) -> None:
    source_sha = _require_source_sha(arguments.source_git_sha)
    dev, dev_commitment, dev_inputs_sha256 = _load_bound_input(
        arguments.dev_inputs,
        arguments.dev_commitment,
    )
    if dev.split != "dev" or len(dev.records) != 84:
        raise ValueError("prediction requires the canonical 84-video dev sidecar")
    dev_commitment_sha256 = _sha256_file(arguments.dev_commitment)
    identity, identity_receipt_sha256 = _validate_identity_receipt(
        arguments.identity_receipt,
        source_sha=source_sha,
        dev_inputs_sha256=dev_inputs_sha256,
        dev_commitment_sha256=dev_commitment_sha256,
    )
    code_files_sha256 = _source_code_hashes(arguments.repository, arguments.runner)
    code_sha256 = _sha256_json(code_files_sha256)

    sequences, cache_snapshot = load_pose_cache_set(
        dev.records,
        cache_dir=arguments.pose_cache_dir,
        pose_fingerprint=arguments.pose_fingerprint,
    )
    if len(sequences) != 84 or len(cache_snapshot.entries) != 84:
        raise RuntimeError("dev pose-cache snapshot is incomplete")
    receipt_by_id = {entry.video_id: entry for entry in cache_snapshot.entries}
    adapter = SpectralProxyAdapter(minimum_period=4, maximum_period=128)
    records: list[dict[str, Any]] = []
    for sequence in sequences:
        result = adapter.predict(sequence)
        receipt = receipt_by_id[sequence.video_id]
        records.append(
            {
                "video_id": sequence.video_id,
                "prediction": result.count,
                "period_frames": result.period_frames,
                "expert_counts": list(result.expert_counts),
                "confidence": result.confidence,
                "valid_frames": int(np.count_nonzero(sequence.valid_mask)),
                "total_frames": sequence.num_frames,
                "pose_cache_sha256": receipt.cache_sha256,
                "pose_cache_bytes": receipt.bytes,
            }
        )
    if tuple(row["video_id"] for row in records) != tuple(
        record.video_id for record in dev.records
    ):
        raise RuntimeError("prediction order changed from the frozen dev sidecar")

    predictions_path = arguments.output_dir / "predictions.json"
    prediction_receipt_path = arguments.output_dir / "predictions.receipt.json"
    manifest_path = arguments.output_dir / "run.manifest.json"
    prediction_payload = {
        "schema_version": 1,
        "artifact_type": "spectral_proxy_label_free_dev_predictions",
        "created_at_utc": _utc_now(),
        "method_id": METHOD_ID,
        "classification": "deterministic_reference_only_not_paper_baseline",
        "diagnostic_only": True,
        "eligible_for_paper_table": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "sample_count": len(records),
        "source_git_sha": source_sha,
        "prediction_code_files_sha256": code_files_sha256,
        "prediction_code_sha256": code_sha256,
        "identity_receipt_sha256": identity_receipt_sha256,
        "protocol_identity_sha256": identity["protocol_identity_sha256"],
        "training_identity_sha256": identity["training_identity_sha256"],
        "dev_identity_sha256": identity["dev_identity_sha256"],
        "test_identity_sha256": identity["test_identity_sha256"],
        "dev_inputs_sha256": dev_inputs_sha256,
        "dev_commitment_sha256": dev_commitment_sha256,
        "pose_fingerprint": arguments.pose_fingerprint,
        "pose_cache_snapshot": cache_snapshot.to_dict(),
        "period_bounds_frames": [4, 128],
        "records": records,
    }
    _write_json_exclusive(predictions_path, prediction_payload)
    predictions_sha256 = _sha256_file(predictions_path)
    receipt_payload = {
        "schema_version": 1,
        "artifact_type": "spectral_proxy_prediction_receipt",
        "created_at_utc": _utc_now(),
        "source_git_sha": source_sha,
        "predictions_sha256": predictions_sha256,
        "sample_count": 84,
        "dev_inputs_sha256": dev_inputs_sha256,
        "dev_commitment_sha256": dev_commitment_sha256,
        "identity_receipt_sha256": identity_receipt_sha256,
        "pose_cache_snapshot_sha256": cache_snapshot.fingerprint,
        "prediction_code_sha256": code_sha256,
        "label_firewall": {
            "prediction_process_loaded_dev_targets": False,
            "prediction_process_loaded_test_targets": False,
            "prediction_process_loaded_train_targets": False,
            "prediction_inputs_contain_count_or_action": False,
            "gt_count_or_action_oracle_used": False,
        },
    }
    _write_json_exclusive(prediction_receipt_path, receipt_payload)
    manifest_payload = {
        "schema_version": 1,
        "artifact_type": "spectral_proxy_dev_run_manifest",
        "created_at_utc": _utc_now(),
        "source_git_sha": source_sha,
        "method_id": METHOD_ID,
        "classification": "reference_only",
        "seed": None,
        "deterministic": True,
        "runtime": _runtime_payload(container_image_id=arguments.container_image_id),
        "inputs": {
            "identity_receipt_sha256": identity_receipt_sha256,
            "dev_inputs_sha256": dev_inputs_sha256,
            "dev_commitment_sha256": dev_commitment_sha256,
            "pose_cache_snapshot_sha256": cache_snapshot.fingerprint,
        },
        "outputs": {
            "predictions_sha256": predictions_sha256,
            "prediction_receipt_sha256": _sha256_file(prediction_receipt_path),
        },
        "source_files": code_files_sha256,
        "network": "disabled",
        "device": "cpu",
        "test105_status": "identity_only_in_preflight; no predictions and no targets",
    }
    _write_json_exclusive(manifest_path, manifest_payload)


def _validate_prediction_artifacts(
    predictions_path: Path,
    receipt_path: Path,
    identity_receipt_path: Path,
    *,
    repository: Path,
    runner: Path,
    source_sha: str,
) -> tuple[dict[str, Any], str, str, str]:
    predictions_sha256 = _sha256_file(predictions_path)
    receipt_sha256 = _sha256_file(receipt_path)
    identity_sha256 = _sha256_file(identity_receipt_path)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    identity = json.loads(identity_receipt_path.read_text(encoding="utf-8"))
    if predictions.get("artifact_type") != "spectral_proxy_label_free_dev_predictions":
        raise ValueError("unexpected prediction artifact_type")
    if predictions.get("method_id") != METHOD_ID:
        raise ValueError("unexpected prediction method_id")
    if predictions.get("source_git_sha") != source_sha:
        raise ValueError("prediction source revision mismatch")
    if predictions.get("sample_count") != 84 or len(predictions.get("records", [])) != 84:
        raise ValueError("prediction artifact must contain exactly 84 records")
    if receipt.get("predictions_sha256") != predictions_sha256:
        raise ValueError("prediction receipt SHA-256 binding mismatch")
    if receipt.get("identity_receipt_sha256") != identity_sha256:
        raise ValueError("prediction receipt identity binding mismatch")
    if predictions.get("identity_receipt_sha256") != identity_sha256:
        raise ValueError("prediction artifact identity binding mismatch")
    expected_code_files = _source_code_hashes(repository, runner)
    if predictions.get("prediction_code_files_sha256") != expected_code_files:
        raise ValueError("prediction source files differ from scoring source files")
    if predictions.get("prediction_code_sha256") != _sha256_json(expected_code_files):
        raise ValueError("prediction aggregate code SHA-256 mismatch")
    if identity.get("source_git_sha") != source_sha:
        raise ValueError("identity receipt source revision mismatch")
    if _sha256_file(predictions_path) != predictions_sha256:
        raise RuntimeError("predictions changed while they were validated")
    if _sha256_file(receipt_path) != receipt_sha256:
        raise RuntimeError("prediction receipt changed while it was validated")
    if _sha256_file(identity_receipt_path) != identity_sha256:
        raise RuntimeError("identity receipt changed while it was validated")
    return predictions, predictions_sha256, receipt_sha256, identity_sha256


def run_score(arguments: argparse.Namespace) -> None:
    source_sha = _require_source_sha(arguments.source_git_sha)
    predictions, predictions_sha256, receipt_sha256, identity_sha256 = (
        _validate_prediction_artifacts(
            arguments.predictions,
            arguments.prediction_receipt,
            arguments.identity_receipt,
            repository=arguments.repository,
            runner=arguments.runner,
            source_sha=source_sha,
        )
    )

    # This is the first operation in the scorer permitted to stat or open labels.
    dev_targets_sha256 = _sha256_file(arguments.dev_targets)
    targets = load_dev_target_manifest(arguments.dev_targets)
    prediction_rows = predictions["records"]
    prediction_ids = tuple(str(row["video_id"]) for row in prediction_rows)
    target_ids = tuple(record.video_id for record in targets.records)
    if prediction_ids != target_ids:
        raise ValueError("dev target order does not exactly match frozen predictions")
    if _sha256_file(arguments.dev_targets) != dev_targets_sha256:
        raise RuntimeError("dev targets changed while they were loaded")

    metric_report = compute_count_metrics(
        [float(row["prediction"]) for row in prediction_rows],
        [record.count for record in targets.records],
        video_ids=prediction_ids,
        actions=[record.action for record in targets.records],
        bootstrap_samples=BOOTSTRAP_SAMPLES,
        bootstrap_seed=BOOTSTRAP_SEED,
        confidence_level=0.95,
    )
    if _sha256_file(arguments.predictions) != predictions_sha256:
        raise RuntimeError("predictions changed during scoring")
    if _sha256_file(arguments.prediction_receipt) != receipt_sha256:
        raise RuntimeError("prediction receipt changed during scoring")
    if _sha256_file(arguments.identity_receipt) != identity_sha256:
        raise RuntimeError("identity receipt changed during scoring")
    if _sha256_file(arguments.dev_targets) != dev_targets_sha256:
        raise RuntimeError("dev targets changed during scoring")

    metric_payload = metric_report.to_dict()
    payload = {
        "schema_version": 1,
        "artifact_type": "spectral_proxy_dev_evaluation",
        "created_at_utc": _utc_now(),
        "method_id": METHOD_ID,
        "classification": "deterministic_reference_only_not_paper_baseline",
        "diagnostic_only": True,
        "eligible_for_paper_table": False,
        "protocol": "ucfrep_526",
        "split": "dev",
        "source_git_sha": source_sha,
        "predictions_sha256": predictions_sha256,
        "prediction_receipt_sha256": receipt_sha256,
        "identity_receipt_sha256": identity_sha256,
        "dev_targets_sha256": dev_targets_sha256,
        "bootstrap_pairing": "paired_prediction_target_rows",
        "metrics": metric_payload,
        "acceptance_gate": {
            "required_nmae_at_most": 0.228,
            "required_obo_at_least": 0.666,
            "nmae_pass": metric_report.nmae <= 0.228,
            "obo_pass": metric_report.obo >= 0.666,
            "joint_pass": metric_report.nmae <= 0.228 and metric_report.obo >= 0.666,
            "verified_reproduction": False,
        },
        "label_firewall": {
            "prediction_process_loaded_dev_targets": False,
            "scoring_process_loaded_dev_targets": True,
            "test_targets_loaded": False,
            "test_predictions_created": False,
            "gt_count_or_action_oracle_used_for_prediction": False,
        },
    }
    _write_json_exclusive(arguments.output, payload)
    _write_json_exclusive(
        arguments.output.with_name("evaluation.receipt.json"),
        {
            "schema_version": 1,
            "artifact_type": "spectral_proxy_evaluation_receipt",
            "created_at_utc": _utc_now(),
            "evaluation_sha256": _sha256_file(arguments.output),
            "predictions_sha256": predictions_sha256,
            "prediction_receipt_sha256": receipt_sha256,
            "identity_receipt_sha256": identity_sha256,
            "dev_targets_sha256": dev_targets_sha256,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    )


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    identity = subparsers.add_parser("identity")
    identity.add_argument("--source-git-sha", required=True)
    for split in ("train", "dev", "test"):
        identity.add_argument(f"--{split}-inputs", type=Path, required=True)
        identity.add_argument(f"--{split}-commitment", type=Path, required=True)
    identity.add_argument("--output", type=Path, required=True)
    identity.set_defaults(handler=run_identity)

    predict = subparsers.add_parser("predict")
    _add_source_arguments(predict)
    predict.add_argument("--dev-inputs", type=Path, required=True)
    predict.add_argument("--dev-commitment", type=Path, required=True)
    predict.add_argument("--identity-receipt", type=Path, required=True)
    predict.add_argument("--pose-cache-dir", type=Path, required=True)
    predict.add_argument("--pose-fingerprint", required=True)
    predict.add_argument("--container-image-id", required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.set_defaults(handler=run_predict)

    score = subparsers.add_parser("score")
    _add_source_arguments(score)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--prediction-receipt", type=Path, required=True)
    score.add_argument("--identity-receipt", type=Path, required=True)
    score.add_argument("--dev-targets", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(handler=run_score)
    return parser


def main() -> int:
    arguments = build_parser().parse_args()
    arguments.handler(arguments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
