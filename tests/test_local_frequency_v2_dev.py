from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pams.data import (
    PoseInputManifest,
    UnlabeledVideoRecord,
    pose_input_identity_sha256,
)
from pams.metrics import compute_count_metrics
from pams.types import PoseSequence

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "server" / "run_local_frequency_v2_dev.py"
SELECTOR = ROOT / "scripts" / "server" / "run_local_frequency_v2_target_free_selector.py"


def _load_runner() -> ModuleType:
    name = "local_frequency_v2_dev_test_module"
    specification = importlib.util.spec_from_file_location(name, RUNNER)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _snapshot(
    runner: ModuleType,
    video_ids: list[str],
) -> dict[str, Any]:
    entries = [
        {
            "video_id": video_id,
            "cache_sha256": _digest(f"cache:{video_id}"),
            "bytes": 1000 + index,
        }
        for index, video_id in enumerate(sorted(video_ids))
    ]
    fingerprint = runner._sha256_json(
        {
            "schema_version": 1,
            "pose_fingerprint": runner._FROZEN_POSE_FINGERPRINT,
            "entries": entries,
        }
    )
    return {
        "schema_version": 1,
        "pose_fingerprint": runner._FROZEN_POSE_FINGERPRINT,
        "fingerprint": fingerprint,
        "entry_count": len(entries),
        "entries": entries,
    }


def _selector_payload(runner: ModuleType) -> dict[str, Any]:
    methods = runner._candidate_methods()
    selected = methods[0]
    audits: list[dict[str, Any]] = []
    for index, method in enumerate(methods):
        rank_key = [0, float(index), 0.0, 0.0, 0.0, 0.0, -3, method]
        audits.append(
            {
                "rank": index + 1,
                "candidate_key": method,
                "parameters": runner._candidate_parameters(method),
                "rank_key": rank_key,
                "degenerate_candidate_flag": False,
                "synthetic_count_sweep": {"nmae": float(index)},
                "synthetic_stress": {"nmae": 0.0},
                "unlabeled_train_consistency": {
                    "transform_relative_disagreement": 0.0,
                    "duplicate_time_relative_scale_error": 0.0,
                    "prediction_boundary_and_mode_penalty": 0.0,
                    "distinct_prediction_total": 3,
                },
            }
        )
    train_ids = [f"v_TrainAction_g01_c{index:02d}" for index in range(1, 65)]
    pose_snapshot = _snapshot(runner, train_ids)
    transformed = {
        video_id: {
            "original_pose_sha256": _digest(f"original:{video_id}"),
            "transforms": {name: _digest(f"{name}:{video_id}") for name in runner._TRANSFORM_NAMES},
            "duplicate_time_pose_sha256": _digest(f"duplicate:{video_id}"),
        }
        for video_id in train_ids
    }
    return {
        "schema_version": 1,
        "artifact_type": runner._SELECTOR_ARTIFACT_TYPE,
        "classification": runner._SELECTOR_CLASSIFICATION,
        "eligible_for_paper_table": False,
        "selection_status": "exploratory-derived_ineligible",
        "candidate_total": 512,
        "selected_candidate": selected,
        "selected_parameters": runner._candidate_parameters(selected),
        "selected_rank_key": audits[0]["rank_key"],
        "objective": {
            "type": "strict_lexicographic_minimum",
            "lexicographic_order": list(runner._LEXICOGRAPHIC_OBJECTIVE),
            "candidate_key_final_tie_break": True,
        },
        "frozen_protocol": {
            "grid": {
                "trims": list(runner._TRIMS),
                "features": list(runner._FEATURE_NAMES),
                "normalized_dimensions": list(runner._NORMALIZED_DIMENSIONS),
                "minimum_cycles": list(runner._MINIMUM_CYCLES),
                "windows": list(runner._WINDOWS),
                "statistics": list(runner._STATISTICS),
                "scale_reducers": list(runner._SCALE_REDUCERS),
            },
            "train_hash_sample_total": 64,
            "transforms": list(runner._TRANSFORM_NAMES),
            "synthetic_count_range_inclusive": [2, 40],
        },
        "label_firewall": {
            "development_inputs_loaded": False,
            "development_targets_loaded": False,
            "test_inputs_loaded": False,
            "test_targets_loaded": False,
            "dataset_count_labels_loaded": False,
            "dataset_action_labels_loaded": False,
        },
        "source": {
            "relative_path": runner._SELECTOR_SOURCE_RELATIVE_PATH,
            "sha256": runner._APPROVED_SELECTOR_SOURCE_SHA256,
            **runner._FROZEN_SELECTOR_SOURCE_ENVIRONMENT,
        },
        "inputs": {
            "train337_sidecar_sha256": (runner._FROZEN_TRAIN337_SIDECAR_SHA256),
            "frozen_train337_sidecar_sha256": (runner._FROZEN_TRAIN337_SIDECAR_SHA256),
            "train337_sidecar_fingerprint": (runner._FROZEN_TRAIN337_MANIFEST_FINGERPRINT),
            "frozen_train337_sidecar_fingerprint": (runner._FROZEN_TRAIN337_MANIFEST_FINGERPRINT),
            "train337_identity_sha256": (runner._FROZEN_TRAIN337_IDENTITY_SHA256),
            "frozen_train337_identity_sha256": (runner._FROZEN_TRAIN337_IDENTITY_SHA256),
            "pose_fingerprint": runner._FROZEN_POSE_FINGERPRINT,
            "selected_train_video_ids": train_ids,
            "selected_train_identity_sha256": _digest("selected-train"),
            "selected_pose_cache_set": pose_snapshot,
            "selected_pose_cache_set_sha256": pose_snapshot["fingerprint"],
            "transformed_pose_sha256": transformed,
        },
        "selected_predictions": {
            "synthetic_count_sweep": [],
            "synthetic_stress": [],
            "unlabeled_train": [],
        },
        "candidate_audits": audits,
    }


def _prediction_fixture(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], list[str], list[int]]:
    video_ids = [f"video-{index:03d}" for index in range(84)]
    video_hashes = [_digest(f"video:{video_id}") for video_id in video_ids]
    identity_records = tuple(
        UnlabeledVideoRecord(
            video_id=video_id,
            video_path=f"videos/{index:03d}.mp4",
            video_sha256=video_hash,
        )
        for index, (video_id, video_hash) in enumerate(zip(video_ids, video_hashes, strict=True))
    )
    identity_sha256 = pose_input_identity_sha256(identity_records)
    monkeypatch.setattr(
        runner,
        "_FROZEN_DEV_IDENTITY_SHA256",
        identity_sha256,
    )
    monkeypatch.setattr(
        runner,
        "_FROZEN_DEV_INPUTS_SHA256",
        _digest("synthetic-dev-inputs"),
    )
    pose_snapshot = _snapshot(runner, video_ids)
    cache_sha = {entry["video_id"]: entry["cache_sha256"] for entry in pose_snapshot["entries"]}
    selected_candidate = runner._FROZEN_SELECTED_CANDIDATE
    selected_parameters = runner._candidate_parameters(selected_candidate)
    predicted = [2 + index % 7 for index in range(84)]
    source_sha256 = hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    label_firewall = {
        "prediction_loaded_action_labels": False,
        "prediction_loaded_count_labels": False,
        "prediction_loaded_dev_targets": False,
        "prediction_loaded_test_identity": False,
        "prediction_loaded_test_pose": False,
        "prediction_loaded_test_targets": False,
        "gt_count_or_action_oracle_used": False,
    }
    payload = {
        "schema_version": 1,
        "artifact_type": runner._PREDICTION_ARTIFACT_TYPE,
        "method_id": runner._METHOD_ID,
        "classification": runner._CLASSIFICATION,
        "eligible_for_paper_table": False,
        "diagnostic_only": True,
        "protocol": "ucfrep_526",
        "split": "dev",
        "record_total": 84,
        "selected_candidate": selected_candidate,
        "selected_parameters": selected_parameters,
        "selected_parameters_sha256": runner._sha256_json(selected_parameters),
        "selector": {
            "artifact_sha256": runner._FROZEN_SELECTOR_ARTIFACT_SHA256,
            "artifact_bytes": runner._FROZEN_SELECTOR_ARTIFACT_BYTES,
            "source_sha256": runner._APPROVED_SELECTOR_SOURCE_SHA256,
            "source_git_sha": runner._APPROVED_SELECTOR_SOURCE_GIT_SHA,
            "candidate_total": 512,
            "canonical_train337_sidecar_sha256": (runner._FROZEN_TRAIN337_SIDECAR_SHA256),
            "canonical_train337_manifest_fingerprint": (
                runner._FROZEN_TRAIN337_MANIFEST_FINGERPRINT
            ),
            "canonical_train337_identity_sha256": (runner._FROZEN_TRAIN337_IDENTITY_SHA256),
        },
        "dev_input": {
            "sidecar_sha256": runner._FROZEN_DEV_INPUTS_SHA256,
            "manifest_fingerprint": _digest("manifest"),
            "identity_sha256": identity_sha256,
            "record_total": 84,
        },
        "dev_pose_cache_set": pose_snapshot,
        "dev_pose_cache_set_sha256": pose_snapshot["fingerprint"],
        "source": {
            "relative_path": runner._RUNNER_SOURCE_RELATIVE_PATH,
            "sha256": source_sha256,
        },
        "records": [
            {
                "video_id": video_id,
                "video_sha256": video_hash,
                "pose_cache_sha256": cache_sha[video_id],
                "frames": 256,
                "valid_frames": 240,
                "prediction": prediction,
            }
            for video_id, video_hash, prediction in zip(
                video_ids,
                video_hashes,
                predicted,
                strict=True,
            )
        ],
        "label_firewall": label_firewall,
        "mount_audit": {
            "network": "none",
            "selector_artifact_mounted": True,
            "selector_source_mounted": True,
            "dev_inputs_mounted": True,
            "dev_pose_mounted": True,
            "dev_targets_mounted": False,
            "test_identity_mounted": False,
            "test_pose_mounted": False,
            "test_targets_mounted": False,
        },
        "test105_evaluation_authorized": False,
    }
    return payload, video_ids, predicted


def _write_prediction_pair(
    runner: ModuleType,
    tmp_path: Path,
    payload: dict[str, Any],
    *,
    stem: str,
) -> tuple[Path, Path]:
    predictions = tmp_path / f"{stem}.predictions.json"
    receipt = tmp_path / f"{stem}.prediction.receipt.json"
    prediction_sha256, prediction_bytes = runner._write_json_exclusive(
        predictions,
        payload,
    )
    receipt_payload = {
        "schema_version": 1,
        "artifact_type": runner._PREDICTION_RECEIPT_TYPE,
        "method_id": runner._METHOD_ID,
        "protocol": "ucfrep_526",
        "split": "dev",
        "record_total": 84,
        "prediction_file": predictions.name,
        "prediction_bytes": prediction_bytes,
        "prediction_sha256": prediction_sha256,
        "runner_source_sha256": payload["source"]["sha256"],
        "selector_artifact_sha256": payload["selector"]["artifact_sha256"],
        "selector_source_sha256": payload["selector"]["source_sha256"],
        "selected_candidate": payload["selected_candidate"],
        "selected_parameters_sha256": payload["selected_parameters_sha256"],
        "dev_inputs_sha256": payload["dev_input"]["sidecar_sha256"],
        "dev_identity_sha256": payload["dev_input"]["identity_sha256"],
        "dev_pose_cache_set_sha256": payload["dev_pose_cache_set_sha256"],
        "label_firewall": payload["label_firewall"],
        "test105_evaluation_authorized": False,
    }
    runner._write_json_exclusive(receipt, receipt_payload)
    return predictions, receipt


def _bind_selector_fixture(
    runner: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    payload: dict[str, Any],
) -> None:
    raw = path.read_bytes()
    monkeypatch.setattr(
        runner,
        "_FROZEN_SELECTOR_ARTIFACT_SHA256",
        hashlib.sha256(raw).hexdigest(),
    )
    monkeypatch.setattr(
        runner,
        "_FROZEN_SELECTOR_ARTIFACT_BYTES",
        len(raw),
    )
    monkeypatch.setattr(
        runner,
        "_FROZEN_SELECTED_CANDIDATE",
        payload["selected_candidate"],
    )
    monkeypatch.setattr(
        runner,
        "_FROZEN_SELECTED_RANK_KEY",
        tuple(payload["selected_rank_key"]),
    )
    monkeypatch.setattr(
        runner,
        "_FROZEN_SELECTED_PARAMETERS",
        payload["selected_parameters"],
    )


def test_cli_boundaries_expose_only_the_authorized_inputs() -> None:
    runner = _load_runner()
    parser = runner.build_parser()
    predict = parser.parse_args(
        [
            "predict",
            "--selector-artifact",
            "selector.json",
            "--selector-source",
            "selector.py",
            "--dev-inputs",
            "dev.inputs.json",
            "--dev-pose-cache-dir",
            "pose-cache",
            "--predictions-output",
            "predictions.json",
            "--prediction-receipt-output",
            "prediction.receipt.json",
        ]
    )
    score = parser.parse_args(
        [
            "score",
            "--predictions",
            "predictions.json",
            "--prediction-receipt",
            "prediction.receipt.json",
            "--dev-targets",
            "dev.targets.json",
            "--evaluation-output",
            "evaluation.json",
            "--evaluation-receipt-output",
            "evaluation.receipt.json",
        ]
    )

    assert set(vars(predict)) == {
        "command",
        "selector_artifact",
        "selector_source",
        "dev_inputs",
        "dev_pose_cache_dir",
        "predictions_output",
        "prediction_receipt_output",
        "handler",
    }
    assert set(vars(score)) == {
        "command",
        "predictions",
        "prediction_receipt",
        "dev_targets",
        "evaluation_output",
        "evaluation_receipt_output",
        "handler",
    }
    assert set(inspect.signature(runner.run_predict).parameters) == {
        "selector_artifact",
        "selector_source",
        "dev_inputs",
        "dev_pose_cache_dir",
        "predictions_output",
        "prediction_receipt_output",
    }
    assert set(inspect.signature(runner.run_score).parameters) == {
        "predictions",
        "prediction_receipt",
        "dev_targets",
        "evaluation_output",
        "evaluation_receipt_output",
    }
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "predict",
                "--dev-targets",
                "forbidden.json",
            ]
        )
    source = RUNNER.read_text(encoding="utf-8")
    assert 'add_argument("--test' not in source


def test_selector_source_schema_grid_and_train_binding_are_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    selector_artifact = tmp_path / "selector.json"
    selector_payload = _selector_payload(runner)
    selector_artifact.write_text(
        json.dumps(selector_payload, allow_nan=False),
        encoding="utf-8",
    )
    _bind_selector_fixture(
        runner,
        monkeypatch,
        selector_artifact,
        selector_payload,
    )

    binding = runner._validate_selector_artifact(selector_artifact, SELECTOR)

    assert binding.source_sha256 == runner._APPROVED_SELECTOR_SOURCE_SHA256
    assert binding.selected_candidate == selector_payload["selected_candidate"]
    assert binding.selected_parameters == runner._candidate_parameters(binding.selected_candidate)
    tampered_payload = _selector_payload(runner)
    tampered_payload["selected_candidate"] = runner._candidate_methods()[1]
    tampered = tmp_path / "tampered-selector.json"
    tampered.write_text(
        json.dumps(tampered_payload, allow_nan=False),
        encoding="utf-8",
    )
    strict_parser = runner._strict_json_bytes
    monkeypatch.setattr(
        runner,
        "_strict_json_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rewritten selector must be rejected before JSON parsing")
        ),
    )
    with pytest.raises(ValueError, match="frozen target-free"):
        runner._validate_selector_artifact(tampered, SELECTOR)
    monkeypatch.setattr(runner, "_strict_json_bytes", strict_parser)

    degenerate_payload = _selector_payload(runner)
    degenerate_payload["selected_rank_key"][0] = 1
    for audit in degenerate_payload["candidate_audits"]:
        audit["rank_key"][0] = 1
        audit["degenerate_candidate_flag"] = True
    degenerate = tmp_path / "degenerate-selector.json"
    degenerate.write_text(
        json.dumps(degenerate_payload, allow_nan=False),
        encoding="utf-8",
    )
    _bind_selector_fixture(
        runner,
        monkeypatch,
        degenerate,
        degenerate_payload,
    )
    with pytest.raises(ValueError, match="degenerate selected candidate"):
        runner._validate_selector_artifact(degenerate, SELECTOR)

    _bind_selector_fixture(
        runner,
        monkeypatch,
        selector_artifact,
        selector_payload,
    )
    changed_source = tmp_path / "changed-selector.py"
    changed_source.write_bytes(SELECTOR.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="approved frozen source"):
        runner._validate_selector_artifact(selector_artifact, changed_source)


def test_dev_manifest_and_selected_candidate_prediction_are_identity_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    records = tuple(
        UnlabeledVideoRecord(
            video_id=f"video-{index:03d}",
            video_path=f"videos/{index:03d}.mp4",
            video_sha256=_digest(f"video:{index}"),
        )
        for index in range(84)
    )
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split="dev",
        records=records,
    )
    identity = pose_input_identity_sha256(records)
    sidecar = _digest("dev-sidecar")
    monkeypatch.setattr(runner, "_FROZEN_DEV_INPUTS_SHA256", sidecar)
    monkeypatch.setattr(runner, "_FROZEN_DEV_IDENTITY_SHA256", identity)

    assert (
        runner._validate_canonical_dev_manifest(
            manifest,
            sidecar_sha256=sidecar,
        )
        == identity
    )
    monkeypatch.setattr(runner, "_FROZEN_DEV_IDENTITY_SHA256", _digest("wrong"))
    with pytest.raises(ValueError, match="identity commitment"):
        runner._validate_canonical_dev_manifest(
            manifest,
            sidecar_sha256=sidecar,
        )

    selected = runner._candidate_methods()[7]
    sequences = tuple(
        PoseSequence(
            video_id=record.video_id,
            fps=30.0,
            xyz=np.zeros((2, 33, 3), dtype=np.float32),
            valid_mask=np.ones(2, dtype=np.bool_),
        )
        for record in records
    )

    class FakeSelector:
        _WORKER_TOTAL = 8

        @staticmethod
        def _predict_many(
            items: tuple[PoseSequence, ...],
            *,
            worker_total: int,
        ) -> dict[str, dict[str, int]]:
            assert worker_total == 8
            return {
                sequence.video_id: {
                    method: (17 if method == selected else 3)
                    for method in runner._candidate_methods()
                }
                for sequence in items
            }

    assert (
        runner._predict_selected_candidate(
            FakeSelector,
            sequences,
            selected,
        )
        == (17,) * 84
    )


def test_prediction_tampering_and_row_identity_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    payload, _, _ = _prediction_fixture(runner, monkeypatch)
    predictions, receipt = _write_prediction_pair(
        runner,
        tmp_path,
        payload,
        stem="valid",
    )
    validated = runner._validate_frozen_predictions(predictions, receipt)
    assert validated.prediction_sha256 == hashlib.sha256(predictions.read_bytes()).hexdigest()

    predictions.write_bytes(predictions.read_bytes() + b" ")
    with pytest.raises(ValueError, match="receipt does not bind"):
        runner._validate_frozen_predictions(predictions, receipt)

    identity_tamper = json.loads(json.dumps(payload))
    identity_tamper["records"][0]["video_sha256"] = _digest("replacement-video")
    tampered_predictions, tampered_receipt = _write_prediction_pair(
        runner,
        tmp_path,
        identity_tamper,
        stem="identity-tamper",
    )
    with pytest.raises(ValueError, match="row identity"):
        runner._validate_frozen_predictions(
            tampered_predictions,
            tampered_receipt,
        )


def test_isolated_scoring_recomputes_metrics_and_binds_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_runner()
    payload, video_ids, predicted = _prediction_fixture(runner, monkeypatch)
    predictions, prediction_receipt = _write_prediction_pair(
        runner,
        tmp_path,
        payload,
        stem="score",
    )
    targets_path = tmp_path / "dev.targets.json"
    targets_path.write_text('{"synthetic-test-fixture":true}\n', encoding="utf-8")
    target_sha256 = hashlib.sha256(targets_path.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "_FROZEN_DEV_TARGETS_SHA256", target_sha256)
    target_counts = [2 + index % 5 for index in range(84)]
    fake_targets = SimpleNamespace(
        records=tuple(
            SimpleNamespace(
                video_id=video_id,
                action="SyntheticAction",
                count=count,
            )
            for video_id, count in zip(video_ids, target_counts, strict=True)
        )
    )
    monkeypatch.setattr(
        runner,
        "load_dev_target_manifest",
        lambda path: fake_targets,
    )
    evaluation = tmp_path / "evaluation.json"
    evaluation_receipt = tmp_path / "evaluation.receipt.json"

    result = runner.run_score(
        predictions=predictions,
        prediction_receipt=prediction_receipt,
        dev_targets=targets_path,
        evaluation_output=evaluation,
        evaluation_receipt_output=evaluation_receipt,
    )

    report = json.loads(evaluation.read_text(encoding="utf-8"))
    receipt = json.loads(evaluation_receipt.read_text(encoding="utf-8"))
    expected = compute_count_metrics(
        predicted,
        target_counts,
        bootstrap_samples=0,
    )
    for name in ("nmae", "mae", "rmse", "obo", "exact"):
        assert report["metrics"][name] == getattr(expected, name)
        assert result["metrics"][name] == getattr(expected, name)
    assert report["metrics"]["bootstrap_samples"] == 10_000
    assert report["metrics"]["bootstrap_seed"] == 2026
    assert report["test105_evaluation_authorized"] is False
    assert report["mount_audit"]["selector_artifact_mounted"] is False
    assert report["mount_audit"]["dev_pose_mounted"] is False
    assert receipt["evaluation_sha256"] == hashlib.sha256(evaluation.read_bytes()).hexdigest()
    assert receipt["prediction_sha256"] == hashlib.sha256(predictions.read_bytes()).hexdigest()
    assert receipt["test105_evaluation_authorized"] is False
    with pytest.raises(FileExistsError):
        runner.run_score(
            predictions=predictions,
            prediction_receipt=prediction_receipt,
            dev_targets=targets_path,
            evaluation_output=evaluation,
            evaluation_receipt_output=evaluation_receipt,
        )
