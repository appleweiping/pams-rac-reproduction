from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from pams.run_manifest import ArtifactReceipt, CompletedRunReceipt, RunManifest


def _load_probe() -> Any:
    path = Path(__file__).parents[1] / "scripts" / "pams_encoder_supervised_probe.py"
    spec = importlib.util.spec_from_file_location("pams_encoder_supervised_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_probe()


def _write_encoder_receipt(
    tmp_path: Path,
    *,
    source_git_sha: str,
    config_fingerprint: str,
    config_file_sha256: str,
    checkpoint_sha256: str,
    progress_sha256: str,
) -> Path:
    timestamp = datetime.now(timezone.utc).isoformat()
    started = RunManifest(
        schema_version=2,
        receipt_type="started",
        run_id="probe-encoder-fixture",
        created_at_utc=timestamp,
        command=["pams", "train", "encoder"],
        git_sha=source_git_sha,
        config_sha256=config_fingerprint,
        dataset_sha256="d" * 64,
        seed=2026,
        protocol="ucfrep_526",
        status="started",
    )
    receipt = CompletedRunReceipt(
        schema_version=3,
        receipt_type="completed",
        run_id=started.run_id,
        status="completed",
        finished_at=timestamp,
        start_manifest_sha256="e" * 64,
        started=started,
        artifacts=(
            ArtifactReceipt(
                role="input_config",
                locator="config.yaml",
                sha256=config_file_sha256,
                bytes=10,
            ),
            ArtifactReceipt(
                role="output_encoder_checkpoint",
                locator="encoder.pt",
                sha256=checkpoint_sha256,
                bytes=10,
            ),
            ArtifactReceipt(
                role="progress_log",
                locator="encoder.jsonl",
                sha256=progress_sha256,
                bytes=10,
            ),
        ),
    )
    path = tmp_path / "encoder.completed.json"
    path.write_text(receipt.model_dump_json(indent=2), encoding="utf-8")
    return path


def test_train_target_loader_rejects_action_labels(tmp_path: Path) -> None:
    path = tmp_path / "targets.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_type": "train_count_targets",
                "protocol": "ucfrep_526",
                "split": "train",
                "source_annotation_sha256": "a" * 64,
                "records": [
                    {"video_id": "video-a", "count": 3, "action": "forbidden"}
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="action labels are forbidden"):
        probe._load_train_targets(path, ("video-a",))


def test_encoder_completion_receipt_binds_source_config_and_artifacts(
    tmp_path: Path,
) -> None:
    source_git_sha = "a" * 40
    config = tmp_path / "config.yaml"
    checkpoint = tmp_path / "encoder.pt"
    progress = tmp_path / "encoder.jsonl"
    config.write_bytes(b"frozen: config\n")
    checkpoint.write_bytes(b"checkpoint")
    progress.write_bytes(b"progress\n")
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    config_fingerprint = "b" * 64
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    progress_sha256 = hashlib.sha256(progress.read_bytes()).hexdigest()
    receipt = _write_encoder_receipt(
        tmp_path,
        source_git_sha=source_git_sha,
        config_fingerprint=config_fingerprint,
        config_file_sha256=config_sha256,
        checkpoint_sha256=checkpoint_sha256,
        progress_sha256=progress_sha256,
    )

    parsed, receipt_sha256 = probe._validate_encoder_completion_receipt(
        receipt,
        expected_source_git_sha=source_git_sha,
        config_fingerprint=config_fingerprint,
        config_file_sha256=config_sha256,
        checkpoint_sha256=checkpoint_sha256,
        progress_sha256=progress_sha256,
    )

    assert parsed.schema_version == 3
    assert parsed.status == "completed"
    assert receipt_sha256 == hashlib.sha256(receipt.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="started.git_sha"):
        probe._validate_encoder_completion_receipt(
            receipt,
            expected_source_git_sha="b" * 40,
            config_fingerprint=config_fingerprint,
            config_file_sha256=config_sha256,
            checkpoint_sha256=checkpoint_sha256,
            progress_sha256=progress_sha256,
        )
    with pytest.raises(ValueError, match="started.config_sha256"):
        probe._validate_encoder_completion_receipt(
            receipt,
            expected_source_git_sha=source_git_sha,
            config_fingerprint="c" * 64,
            config_file_sha256=config_sha256,
            checkpoint_sha256=checkpoint_sha256,
            progress_sha256=progress_sha256,
        )
    with pytest.raises(ValueError, match="input_config SHA-256"):
        probe._validate_encoder_completion_receipt(
            receipt,
            expected_source_git_sha=source_git_sha,
            config_fingerprint=config_fingerprint,
            config_file_sha256="c" * 64,
            checkpoint_sha256=checkpoint_sha256,
            progress_sha256=progress_sha256,
        )
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        probe._validate_encoder_completion_receipt(
            receipt,
            expected_source_git_sha=source_git_sha,
            config_fingerprint=config_fingerprint,
            config_file_sha256=config_sha256,
            checkpoint_sha256="f" * 64,
            progress_sha256=progress_sha256,
        )


def test_split_guard_rejects_test_manifest() -> None:
    train = SimpleNamespace(
        protocol="ucfrep_526",
        split="train",
        records=tuple(
            SimpleNamespace(video_id=f"train-{index}") for index in range(337)
        ),
    )
    test = SimpleNamespace(
        protocol="ucfrep_526",
        split="test",
        records=tuple(
            SimpleNamespace(video_id=f"test-{index}") for index in range(84)
        ),
    )
    with pytest.raises(ValueError, match="train337 and dev84"):
        probe._validate_split_manifests(train, test)


def test_masked_frequency_features_are_finite_and_mask_aware() -> None:
    generator = torch.Generator().manual_seed(7)
    embeddings = torch.randn((2, 256, 512), generator=generator)
    mask = torch.ones((2, 256), dtype=torch.bool)
    mask[0, 12:20] = False
    embeddings[0, 12:20] = 1e6

    first = probe._batch_features(embeddings, mask)
    embeddings[0, 12:20] = -1e9
    second = probe._batch_features(embeddings, mask)

    assert first.shape == (2, len(probe._feature_names()))
    assert torch.isfinite(first).all()
    torch.testing.assert_close(first[0], second[0])


def test_ridge_probe_is_fixed_and_deterministic() -> None:
    rng = np.random.default_rng(2026)
    train = rng.normal(size=(48, 12))
    targets = np.maximum(1.0, 5.0 + 1.5 * train[:, 0] - 0.7 * train[:, 2])

    first = probe._fit_ridge(train, targets)
    second = probe._fit_ridge(train, targets)

    assert first[1].alpha == 1.0
    np.testing.assert_array_equal(first[2], second[2])
    assert np.all(first[2] >= 0)


def test_train_parser_has_no_dev_or_test_argument() -> None:
    parser = probe._parser()
    train_parser = next(
        action.choices["train"]
        for action in parser._actions
        if getattr(action, "choices", None) and "train" in action.choices
    )
    destinations = {action.dest for action in train_parser._actions}
    assert "dev_inputs" not in destinations
    assert "dev_targets" not in destinations
    assert "test_inputs" not in destinations
    assert "test_targets" not in destinations
    assert "encoder_completion_receipt" in destinations


def test_predict_parser_requires_encoder_completion_receipt() -> None:
    parser = probe._parser()
    predict_parser = next(
        action.choices["predict"]
        for action in parser._actions
        if getattr(action, "choices", None) and "predict" in action.choices
    )
    receipt_actions = [
        action
        for action in predict_parser._actions
        if action.dest == "encoder_completion_receipt"
    ]
    assert len(receipt_actions) == 1
    assert receipt_actions[0].required


def test_code_hash_binding_is_fail_closed() -> None:
    expected = {"probe": "a" * 64, "pams_model": "b" * 64}
    probe._require_code_hash_binding(expected, expected, stage="fixture")
    with pytest.raises(ValueError, match="code hashes"):
        probe._require_code_hash_binding(
            {**expected, "probe": "c" * 64},
            expected,
            stage="fixture",
        )


def test_probe_source_revision_env_is_required_and_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PAMS_CONTAINER_SOURCE_REVISION", raising=False)
    with pytest.raises(RuntimeError, match="PAMS_CONTAINER_SOURCE_REVISION"):
        probe._probe_source_revision()
    monkeypatch.setenv("PAMS_CONTAINER_SOURCE_REVISION", "A" * 40)
    with pytest.raises(RuntimeError, match="lowercase"):
        probe._probe_source_revision()
    revision = "a" * 40
    monkeypatch.setenv("PAMS_CONTAINER_SOURCE_REVISION", revision)
    assert probe._probe_source_revision() == revision
    with pytest.raises(ValueError, match="source revision"):
        probe._require_source_revision_binding("b" * 40, revision, stage="fixture")


def test_frozen_target_hashes_reject_any_other_sidecar() -> None:
    probe._require_frozen_target_sha256(probe._TRAIN_TARGET_SHA256, split="train")
    probe._require_frozen_target_sha256(probe._DEV_TARGET_SHA256, split="dev")
    with pytest.raises(ValueError, match="frozen probe protocol"):
        probe._require_frozen_target_sha256("0" * 64, split="train")
    with pytest.raises(ValueError, match="frozen probe protocol"):
        probe._require_frozen_target_sha256("0" * 64, split="dev")


def test_prediction_run_rejects_wrong_prediction_hash(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_bytes(b'{"fixture":true}\n')
    prediction_sha256 = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    eligibility = {
        "pams_main_result": False,
        "self_supervised_result": False,
        "baseline_result": False,
        "classification": "diagnostic_supervised_encoder_upper_bound",
    }
    label_access = {
        "train337_counts": True,
        "train337_actions": False,
        "dev84_counts_during_run": False,
        "dev84_actions_during_run": False,
        "sealed_test105_identity_or_labels": False,
    }
    code_hashes = {"probe": "a" * 64}
    predictions = {
        "eligibility": eligibility,
        "label_access": label_access,
        "probe_source_revision": "b" * 40,
        "code_files_sha256": code_hashes,
        "config_file_sha256": "1" * 64,
        "encoder_checkpoint_sha256": "2" * 64,
        "encoder_progress_sha256": "3" * 64,
        "encoder_completion_receipt_sha256": "4" * 64,
        "model_sha256": "5" * 64,
        "dev_inputs_sha256": "6" * 64,
        "dev_pose_cache_snapshot": {"fingerprint": "7" * 64},
    }
    run = {
        "schema_version": 1,
        "artifact_type": "pams_encoder_supervised_upper_bound_predict_run",
        "method_key": probe._METHOD_KEY,
        "status": probe._STATUS,
        "seed": 2026,
        "probe_source_revision": predictions["probe_source_revision"],
        "eligibility": eligibility,
        "label_access": label_access,
        "inputs": {
            "config": predictions["config_file_sha256"],
            "encoder_checkpoint": predictions["encoder_checkpoint_sha256"],
            "encoder_progress": predictions["encoder_progress_sha256"],
            "encoder_completion_receipt": predictions[
                "encoder_completion_receipt_sha256"
            ],
            "model": predictions["model_sha256"],
            "dev_inputs": predictions["dev_inputs_sha256"],
        },
        "pose_cache_snapshot": predictions["dev_pose_cache_snapshot"],
        "code_files_sha256": code_hashes,
        "probe_checkout": {"revision": None, "clean": None},
        "runtime": {"device": "cuda"},
        "artifacts": {
            "predictions": {
                "path": "predictions.json",
                "sha256": "f" * 64,
            }
        },
        "test105_access": "none",
        "dev_target_reachable_by_predict_process": False,
    }
    run_path = tmp_path / "predict-run.json"
    run_path.write_text(json.dumps(run), encoding="utf-8")

    with pytest.raises(ValueError, match="predictions SHA-256"):
        probe._validate_prediction_run(
            run_path,
            predictions=predictions,
            prediction_sha256=prediction_sha256,
        )


def test_score_parser_requires_prediction_run() -> None:
    parser = probe._parser()
    score_parser = next(
        action.choices["score"]
        for action in parser._actions
        if getattr(action, "choices", None) and "score" in action.choices
    )
    run_actions = [
        action for action in score_parser._actions if action.dest == "prediction_run"
    ]
    assert len(run_actions) == 1
    assert run_actions[0].required
