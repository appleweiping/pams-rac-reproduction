from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

import pams.data as data_module
import pams.pams_dev as dev_module
from pams.cli import app
from pams.data import (
    DevTargetManifest,
    DevTargetRecord,
    LabelFreeProtocolInputs,
    PoseCacheEntryReceipt,
    PoseCacheSetSnapshot,
    PoseInputCommitment,
    PoseInputManifest,
    UnlabeledVideoRecord,
    pose_input_identity_sha256,
)
from pams.evaluation import PredictionRecord
from pams.pams_dev import run_pams_dev_prediction, score_pams_dev_predictions
from pams.reproducibility import sha256_file
from pams.run_manifest import (
    RunManifest,
    create_completed_receipt,
    write_manifest_exclusive,
)
from pams.training import CheckpointProvenance
from pams.types import CountResult, PoseSequence

REPOSITORY = Path(__file__).parents[1]
CONFIG = REPOSITORY / "configs/pams.yaml"
RUNNER = CliRunner()
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _ids(split: str) -> tuple[str, ...]:
    return tuple(
        (
            REPOSITORY
            / f"data/splits/ucfrep_526_{split}_{ {'train': 337, 'dev': 84, 'test': 105}[split] }.txt"
        )
        .read_text(encoding="utf-8")
        .splitlines()
    )


def _write_bound_inputs(
    tmp_path: Path,
    split: str,
) -> tuple[Path, Path, PoseInputManifest]:
    records = tuple(
        UnlabeledVideoRecord(
            video_id=video_id,
            video_path=f"videos/{video_id}.avi",
            video_sha256=hashlib.sha256(f"{split}:{video_id}".encode()).hexdigest(),
        )
        for video_id in _ids(split)
    )
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split=split,
        records=records,
    )
    sidecar = tmp_path / f"{split}.inputs.json"
    sidecar.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    commitment = PoseInputCommitment(
        protocol="ucfrep_526",
        split=split,
        record_total=len(records),
        identity_sha256=pose_input_identity_sha256(records),
        sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        sidecar_fingerprint=manifest.fingerprint,
    )
    commitment_path = tmp_path / f"{split}.inputs.commitment.json"
    commitment_path.write_text(
        json.dumps(commitment.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    return sidecar, commitment_path, manifest


def _snapshot(
    manifest: PoseInputManifest,
    *,
    pose_fingerprint: str,
) -> PoseCacheSetSnapshot:
    return PoseCacheSetSnapshot(
        pose_fingerprint=pose_fingerprint,
        entries=tuple(
            PoseCacheEntryReceipt(
                video_id=record.video_id,
                cache_sha256=hashlib.sha256(
                    f"pose:{record.video_id}".encode()
                ).hexdigest(),
                bytes=1,
            )
            for record in manifest.records
        ),
    )


def _write_completed_training_fixture(
    tmp_path: Path,
    *,
    stage: str = "encoder",
) -> dict[str, object]:
    package = tmp_path / f"{stage}-receipt-package"
    artifacts_dir = package / "artifacts"
    manifests_dir = package / "manifests"
    artifacts_dir.mkdir(parents=True)
    manifests_dir.mkdir()
    train_inputs, train_commitment, train = _write_bound_inputs(
        artifacts_dir,
        "train",
    )
    dev_inputs, dev_commitment, dev = _write_bound_inputs(artifacts_dir, "dev")
    test_inputs, test_commitment, test = _write_bound_inputs(artifacts_dir, "test")
    inputs = LabelFreeProtocolInputs(
        protocol="ucfrep_526",
        train=train,
        dev=dev,
        test=test,
    )
    config_path = artifacts_dir / "pams.yaml"
    config_path.write_bytes(CONFIG.read_bytes())
    config = dev_module.load_config(config_path)
    training_snapshot = _snapshot(
        train,
        pose_fingerprint=config.pose_fingerprint,
    )
    snapshot_path = artifacts_dir / "train-pose-cache-snapshot.json"
    snapshot_path.write_text(
        json.dumps(training_snapshot.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    checkpoint = artifacts_dir / f"{stage}.pt"
    progress = artifacts_dir / f"{stage}.jsonl"
    checkpoint.write_bytes(f"{stage}-checkpoint".encode())
    progress.write_text(f"{stage}-progress\n", encoding="utf-8")
    upstream_checkpoint: Path | None = None
    upstream_progress: Path | None = None
    completion_artifacts: dict[str, Path] = {
        "input_config": config_path,
        "input_dataset_manifest": train_inputs,
        "input_pose_cache_snapshot": snapshot_path,
        "progress_log": progress,
        "input_train_pose_inputs": train_inputs,
        "input_train_pose_input_commitment": train_commitment,
        "input_dev_pose_inputs": dev_inputs,
        "input_dev_pose_input_commitment": dev_commitment,
        "input_test_identity_pose_inputs": test_inputs,
        "input_test_identity_pose_input_commitment": test_commitment,
    }
    if stage == "encoder":
        completion_artifacts["output_encoder_checkpoint"] = checkpoint
        completed_epochs = config.training.epochs
    elif stage == "sshead":
        completion_artifacts["output_sshead_checkpoint"] = checkpoint
        completed_epochs = config.sshead.epochs
        upstream_checkpoint = artifacts_dir / "upstream-encoder.pt"
        upstream_progress = artifacts_dir / "upstream-encoder.jsonl"
        upstream_checkpoint.write_bytes(b"upstream-encoder-checkpoint")
        upstream_progress.write_text(
            "upstream-encoder-progress\n",
            encoding="utf-8",
        )
        completion_artifacts["input_encoder_checkpoint"] = upstream_checkpoint
        completion_artifacts["input_encoder_progress"] = upstream_progress
    else:
        raise ValueError(f"unsupported test stage: {stage}")
    run_id = f"{stage}-fixture"
    started = RunManifest(
        run_id=run_id,
        created_at_utc="2026-07-29T00:00:00+00:00",
        command=["pams", "train", stage],
        git_sha="b" * 40,
        config_sha256=config.fingerprint,
        dataset_sha256=inputs.training_fingerprint(include_dev=False),
        seed=config.seed,
        protocol=inputs.protocol,
        hardware={
            "container": {
                "image_id": f"sha256:{'2' * 64}",
                "environment_sha256": "3" * 64,
                "source_revision": "b" * 40,
            }
        },
    )
    started_path = manifests_dir / f"{run_id}.started.json"
    write_manifest_exclusive(started, started_path)
    completed = create_completed_receipt(
        started_path,
        artifacts=completion_artifacts,
        metrics={"completed_epochs": completed_epochs},
        finished_at="2026-07-29T00:01:00+00:00",
    )
    completed_path = manifests_dir / f"{run_id}.completed.json"
    write_manifest_exclusive(completed, completed_path)
    input_paths = {
        "train_inputs_sha256": train_inputs,
        "train_commitment_sha256": train_commitment,
        "dev_inputs_sha256": dev_inputs,
        "dev_commitment_sha256": dev_commitment,
        "test_identity_inputs_sha256": test_inputs,
        "test_identity_commitment_sha256": test_commitment,
    }
    return {
        "receipt_path": completed_path,
        "expected_stage": stage,
        "checkpoint_path": checkpoint,
        "progress_path": progress,
        "config_path": config_path,
        "config": config,
        "inputs": inputs,
        "input_paths": input_paths,
        "input_hashes": {
            role: sha256_file(path) for role, path in input_paths.items()
        },
        "training_pose_snapshot": training_snapshot,
        "upstream_encoder_checkpoint_path": upstream_checkpoint,
        "upstream_encoder_progress_path": upstream_progress,
    }


def _validate_training_fixture(fixture: dict[str, object]) -> object:
    return dev_module._validate_completed_training_receipt(
        fixture["receipt_path"],
        expected_stage=fixture["expected_stage"],
        checkpoint_path=fixture["checkpoint_path"],
        progress_path=fixture["progress_path"],
        config_path=fixture["config_path"],
        config=fixture["config"],
        inputs=fixture["inputs"],
        input_paths=fixture["input_paths"],
        input_hashes=fixture["input_hashes"],
        training_pose_snapshot=fixture["training_pose_snapshot"],
        upstream_encoder_checkpoint_path=fixture[
            "upstream_encoder_checkpoint_path"
        ],
        upstream_encoder_progress_path=fixture[
            "upstream_encoder_progress_path"
        ],
    )


def _patch_prediction_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    train: PoseInputManifest,
    dev: PoseInputManifest,
    checkpoint: Path,
    checkpoint_progress: Path,
    checkpoint_completion_receipt: Path,
    upstream_checkpoint: Path,
    upstream_progress: Path,
    upstream_completion_receipt: Path,
) -> None:
    config = dev_module.load_config(CONFIG)
    training_snapshot = _snapshot(train, pose_fingerprint=config.pose_fingerprint)
    dev_snapshot = _snapshot(dev, pose_fingerprint=config.pose_fingerprint)
    sequences = tuple(
        PoseSequence(
            video_id=record.video_id,
            fps=30.0,
            xyz=np.zeros((16, 33, 3), dtype=np.float32),
            valid_mask=np.ones(16, dtype=np.bool_),
        )
        for record in dev.records
    )

    def fake_load_pose_cache_set(
        records: tuple[UnlabeledVideoRecord, ...],
        **_kwargs: object,
    ) -> tuple[tuple[PoseSequence, ...], PoseCacheSetSnapshot]:
        if len(records) == 337:
            return (), training_snapshot
        assert len(records) == 84
        return sequences, dev_snapshot

    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    checkpoint_progress_sha256 = hashlib.sha256(
        checkpoint_progress.read_bytes()
    ).hexdigest()
    upstream_sha256 = hashlib.sha256(upstream_checkpoint.read_bytes()).hexdigest()
    upstream_progress_sha256 = hashlib.sha256(
        upstream_progress.read_bytes()
    ).hexdigest()
    checkpoint_completion_receipt_sha256 = hashlib.sha256(
        checkpoint_completion_receipt.read_bytes()
    ).hexdigest()
    upstream_completion_receipt_sha256 = hashlib.sha256(
        upstream_completion_receipt.read_bytes()
    ).hexdigest()
    provenance = CheckpointProvenance(
        protocol="ucfrep_526",
        dataset_fingerprint="1" * 64,
        training_video_ids=tuple(record.video_id for record in train.records),
        pose_fingerprint=config.pose_fingerprint,
        pose_cache_set_sha256=training_snapshot.fingerprint,
        source_git_sha="b" * 40,
        container_image_id=f"sha256:{'2' * 64}",
        container_environment_sha256="3" * 64,
        upstream_encoder_checkpoint_sha256=upstream_sha256,
    )
    monkeypatch.setattr(dev_module, "clean_git_revision", lambda _root: "a" * 40)
    monkeypatch.setattr(
        dev_module,
        "_runtime_container_identity",
        lambda _revision: (f"sha256:{'4' * 64}", "5" * 64),
    )
    monkeypatch.setattr(
        dev_module,
        "hardware_fingerprint",
        lambda: {"test": True},
    )
    monkeypatch.setattr(
        dev_module,
        "load_pose_cache_set",
        fake_load_pose_cache_set,
    )
    monkeypatch.setattr(
        dev_module,
        "_validate_completed_training_receipt",
        lambda path, **_kwargs: dev_module._CompletedTrainingBinding(
            receipt_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            source_git_sha="b" * 40,
            container_image_id=f"sha256:{'2' * 64}",
            container_environment_sha256="3" * 64,
        ),
    )
    monkeypatch.setattr(
        dev_module,
        "_validated_checkpoint_model",
        lambda **_kwargs: (
            object(),
            provenance,
            {
                "checkpoint_sha256": checkpoint_sha256,
                "checkpoint_progress_sha256": checkpoint_progress_sha256,
                "checkpoint_completion_receipt_sha256": (
                    checkpoint_completion_receipt_sha256
                ),
                "upstream_encoder_checkpoint_sha256": upstream_sha256,
                "upstream_encoder_progress_sha256": upstream_progress_sha256,
                "upstream_encoder_completion_receipt_sha256": (
                    upstream_completion_receipt_sha256
                ),
            },
        ),
    )
    monkeypatch.setattr(
        dev_module,
        "predict_sequences",
        lambda _model, items, _config, *, device: tuple(
            PredictionRecord(
                video_id=item.video_id,
                result=CountResult(
                    count=8,
                    period_frames=32.0,
                    expert_counts=(8, 8, 8),
                    confidence=1.0,
                    period_stream=np.zeros(item.num_frames, dtype=np.float32),
                ),
            )
            for item in items
        ),
    )


def _run_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    expert_mode: str | None = None,
) -> tuple[dict[str, object], Path, Path]:
    train_inputs, train_commitment, train = _write_bound_inputs(tmp_path, "train")
    dev_inputs, dev_commitment, dev = _write_bound_inputs(tmp_path, "dev")
    test_inputs, test_commitment, _ = _write_bound_inputs(tmp_path, "test")
    checkpoint = tmp_path / "sshead.pt"
    checkpoint_progress = tmp_path / "sshead.jsonl"
    checkpoint_completion_receipt = tmp_path / "sshead.completed.json"
    upstream_checkpoint = tmp_path / "encoder.pt"
    upstream_progress = tmp_path / "encoder.jsonl"
    upstream_completion_receipt = tmp_path / "encoder.completed.json"
    checkpoint.write_bytes(b"sshead")
    checkpoint_progress.write_text("sshead-progress\n", encoding="utf-8")
    checkpoint_completion_receipt.write_text(
        '{"receipt":"sshead-completed"}\n',
        encoding="utf-8",
    )
    upstream_checkpoint.write_bytes(b"encoder")
    upstream_progress.write_text("encoder-progress\n", encoding="utf-8")
    upstream_completion_receipt.write_text(
        '{"receipt":"encoder-completed"}\n',
        encoding="utf-8",
    )
    _patch_prediction_dependencies(
        monkeypatch,
        train=train,
        dev=dev,
        checkpoint=checkpoint,
        checkpoint_progress=checkpoint_progress,
        checkpoint_completion_receipt=checkpoint_completion_receipt,
        upstream_checkpoint=upstream_checkpoint,
        upstream_progress=upstream_progress,
        upstream_completion_receipt=upstream_completion_receipt,
    )
    cache = tmp_path / "pose-cache"
    cache.mkdir()
    output = tmp_path / "predict"
    result = run_pams_dev_prediction(
        checkpoint_path=checkpoint,
        checkpoint_progress_path=checkpoint_progress,
        checkpoint_completion_receipt_path=checkpoint_completion_receipt,
        train_inputs_path=train_inputs,
        train_commitment_path=train_commitment,
        dev_inputs_path=dev_inputs,
        dev_commitment_path=dev_commitment,
        test_identity_inputs_path=test_inputs,
        test_identity_commitment_path=test_commitment,
        pose_cache_dir=cache,
        output_dir=output,
        config_path=CONFIG,
        variant="sshead",
        expert_mode=expert_mode,  # type: ignore[arg-type]
        upstream_encoder_checkpoint_path=upstream_checkpoint,
        upstream_encoder_progress_path=upstream_progress,
        upstream_encoder_completion_receipt_path=upstream_completion_receipt,
        device="cpu",
        repository_root=REPOSITORY,
        command=("pams", "evaluate", "dev-predict"),
    )
    return result, output / "predictions.json", output / "prediction.receipt.json"


def _write_dev_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    records = tuple(
        DevTargetRecord(
            video_id=video_id,
            action=video_id.removeprefix("v_").split("_g", 1)[0],
            count=8,
        )
        for video_id in _ids("dev")
    )
    monkeypatch.setattr(
        data_module,
        "_UCFREP_526_DEV_ANNOTATION_SHA256",
        data_module._dev_target_annotation_sha256(records),
    )
    manifest = DevTargetManifest(
        protocol="ucfrep_526",
        records=records,
        source_annotation_sha256=(
            "d371f9f4609730d6484efc337413b444ed73752ad5e994366d02fb79a9960452"
        ),
    )
    path = tmp_path / "dev.targets.json"
    path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    return path


def test_prediction_boundary_has_no_target_parameter() -> None:
    parameters = set(inspect.signature(run_pams_dev_prediction).parameters)
    assert all("target" not in parameter for parameter in parameters)


def test_cli_prediction_help_exposes_no_target_option() -> None:
    group = RUNNER.invoke(app, ["evaluate", "--help"], env={"COLUMNS": "240"})
    predict = RUNNER.invoke(
        app,
        ["evaluate", "dev-predict", "--help"],
        env={"COLUMNS": "240"},
    )
    assert group.exit_code == predict.exit_code == 0
    group_output = ANSI_ESCAPE.sub("", group.output)
    predict_output = ANSI_ESCAPE.sub("", predict.output)
    assert {"dev-predict", "dev-score"} <= set(group_output.split())
    assert "--checkpoint-completion-receipt" in predict_output
    assert "--upstream-encoder-completion-receipt" in predict_output
    assert "--expert-mode" in predict_output
    assert "--dev-targets" not in predict_output
    assert "--targets" not in predict_output


def test_completed_training_receipt_binds_terminal_encoder(
    tmp_path: Path,
) -> None:
    fixture = _write_completed_training_fixture(tmp_path)
    binding = _validate_training_fixture(fixture)

    assert isinstance(binding, dev_module._CompletedTrainingBinding)
    assert binding.receipt_sha256 == sha256_file(fixture["receipt_path"])
    assert binding.source_git_sha == "b" * 40
    assert binding.container_image_id == f"sha256:{'2' * 64}"


def test_completed_training_receipt_rejects_tampered_artifact(
    tmp_path: Path,
) -> None:
    fixture = _write_completed_training_fixture(tmp_path)
    Path(fixture["checkpoint_path"]).write_bytes(b"tampered-checkpoint")

    with pytest.raises(ValueError, match="artifact (size|SHA-256)"):
        _validate_training_fixture(fixture)


def test_completed_training_receipt_rejects_nonterminal_mismatch(
    tmp_path: Path,
) -> None:
    fixture = _write_completed_training_fixture(tmp_path)
    receipt_path = Path(fixture["receipt_path"])
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    payload["metrics"]["completed_epochs"] = 149
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="not terminal"):
        _validate_training_fixture(fixture)


def test_completed_training_receipt_is_mandatory(tmp_path: Path) -> None:
    fixture = _write_completed_training_fixture(tmp_path)
    fixture["receipt_path"] = tmp_path / "missing.completed.json"

    with pytest.raises(FileNotFoundError):
        _validate_training_fixture(fixture)


def test_prediction_freezes_target_free_sshead_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, predictions_path, receipt_path = _run_prediction(tmp_path, monkeypatch)
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    assert payload["variant"] == "sshead"
    assert payload["schema_version"] == receipt["schema_version"] == 2
    assert payload["method_key"] == "pams-sshead-inferred"
    assert payload["table2_eligible"] is False
    assert payload["record_total"] == len(payload["records"]) == 84
    assert all(
        "target" not in row and "action" not in row for row in payload["records"]
    )
    assert len(payload["train_inputs_sha256"]) == 64
    assert len(payload["dev_inputs_sha256"]) == 64
    assert len(payload["test_identity_inputs_sha256"]) == 64
    assert len(payload["checkpoint_sha256"]) == 64
    assert len(payload["checkpoint_completion_receipt_sha256"]) == 64
    assert len(payload["upstream_encoder_checkpoint_sha256"]) == 64
    assert len(payload["upstream_encoder_completion_receipt_sha256"]) == 64
    assert (
        receipt["dev_pose_cache_snapshot_sha256"]
        == payload["dev_pose_cache_snapshot_sha256"]
    )
    assert receipt["prediction_sha256"] == hashlib.sha256(
        predictions_path.read_bytes()
    ).hexdigest()
    assert result["prediction_receipt_sha256"] == hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()

    legacy_payload = {**payload, "schema_version": 1}
    with pytest.raises(ValidationError, match="schema_version"):
        dev_module.PAMSDevPredictionArtifact.model_validate(legacy_payload)


def test_prediction_records_inferred_medium_only_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, predictions_path, receipt_path = _run_prediction(
        tmp_path,
        monkeypatch,
        expert_mode="medium_only",
    )
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    training_config = dev_module.load_config(CONFIG)
    inference_config = dev_module._inference_config(
        training_config,
        "medium_only",
    )

    assert payload["consensus_expert_mode"] == "medium_only"
    assert payload["ablation_status"] == "inferred single-expert ablation"
    assert payload["training_config_fingerprint"] == training_config.fingerprint
    assert payload["config_fingerprint"] == inference_config.fingerprint
    assert payload["config_fingerprint"] != payload["training_config_fingerprint"]
    assert all(
        row["selection_mode"] == "medium_only"
        and row["selected_expert"] == "medium"
        and row["count"] == row["expert_counts"][1]
        for row in payload["records"]
    )
    assert receipt["consensus_expert_mode"] == "medium_only"
    assert receipt["config_fingerprint"] == payload["config_fingerprint"]
    assert result["consensus_expert_mode"] == "medium_only"


def test_json_bundle_rolls_back_first_file_on_late_collision(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    collision = tmp_path / "collision.json"
    collision.write_bytes(b"racer-owned")

    with pytest.raises(FileExistsError):
        dev_module._write_json_bundle(
            ((first, {"first": True}), (collision, {"second": True}))
        )

    assert not first.exists()
    assert collision.read_bytes() == b"racer-owned"


def test_json_write_removes_owned_file_after_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "interrupted.json"

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(dev_module.os, "fsync", fail_fsync)
    with pytest.raises(OSError, match="injected fsync failure"):
        dev_module._write_json_new(output, {"partial": True})

    assert not output.exists()


def test_json_bundle_does_not_unlink_concurrently_changed_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    real_write = dev_module._write_json_new
    calls = 0

    def racing_write(path: Path, payload: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            first.write_bytes(b"concurrent-owner")
            raise FileExistsError("late collision")
        return real_write(path, payload)

    monkeypatch.setattr(dev_module, "_write_json_new", racing_write)
    with pytest.raises(RuntimeError, match="rollback was unsafe"):
        dev_module._write_json_bundle(
            ((first, {"first": True}), (second, {"second": True}))
        )

    assert first.read_bytes() == b"concurrent-owner"
    assert not second.exists()


def test_score_rejects_tampering_before_opening_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, predictions_path, receipt_path = _run_prediction(tmp_path, monkeypatch)
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    payload["records"][0]["count"] += 1
    predictions_path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        dev_module,
        "load_dev_target_manifest",
        lambda _path: pytest.fail("target loader crossed prediction firewall"),
    )

    with pytest.raises(ValueError, match="SHA-256"):
        score_pams_dev_predictions(
            predictions_path=predictions_path,
            prediction_receipt_path=receipt_path,
            dev_targets_path=tmp_path / "must-not-open.targets.json",
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )


def test_score_rejects_pose_snapshot_receipt_mismatch_before_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, predictions_path, receipt_path = _run_prediction(tmp_path, monkeypatch)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["dev_pose_cache_snapshot_sha256"] = "f" * 64
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(
        dev_module,
        "load_dev_target_manifest",
        lambda _path: pytest.fail("target loader crossed prediction firewall"),
    )

    with pytest.raises(ValueError, match="metadata mismatch"):
        score_pams_dev_predictions(
            predictions_path=predictions_path,
            prediction_receipt_path=receipt_path,
            dev_targets_path=tmp_path / "must-not-open.targets.json",
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )


def test_score_emits_independent_10k_paired_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, predictions_path, receipt_path = _run_prediction(tmp_path, monkeypatch)
    targets_path = _write_dev_targets(tmp_path, monkeypatch)
    output = tmp_path / "score"
    result = score_pams_dev_predictions(
        predictions_path=predictions_path,
        prediction_receipt_path=receipt_path,
        dev_targets_path=targets_path,
        output_dir=output,
        repository_root=REPOSITORY,
    )
    evaluation_path = output / "evaluation.json"
    evaluation_receipt_path = output / "evaluation.receipt.json"
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    report = payload["report"]

    assert payload["bootstrap_pairing"] == "paired_prediction_target_rows"
    assert report["bootstrap_samples"] == 10_000
    assert report["bootstrap_seed"] == 2026
    assert report["nmae"] == report["mae"] == report["rmse"] == 0.0
    assert report["obo"] == 1.0
    assert len(report["per_video"]) == len(payload["predictions"]) == 84
    assert result["evaluation_sha256"] == hashlib.sha256(
        evaluation_path.read_bytes()
    ).hexdigest()
    assert result["evaluation_receipt_sha256"] == hashlib.sha256(
        evaluation_receipt_path.read_bytes()
    ).hexdigest()
