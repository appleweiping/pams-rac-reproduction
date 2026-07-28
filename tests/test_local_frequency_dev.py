from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import pams.data as data_module
import pams.local_frequency_dev as dev_module
from pams.cli import app
from pams.data import (
    DevTargetManifest,
    DevTargetRecord,
    PoseCacheEntryReceipt,
    PoseCacheSetSnapshot,
    PoseInputCommitment,
    PoseInputManifest,
    UnlabeledVideoRecord,
    pose_input_identity_sha256,
)
from pams.local_frequency import SyntheticCandidateScore, SyntheticFreezeReport
from pams.local_frequency_dev import (
    run_dev_prediction,
    run_synthetic_freeze_replay,
    score_dev_predictions,
)
from pams.types import CountResult, PoseSequence

REPOSITORY = Path(__file__).parents[1]
FREQUENCY_CONFIG = REPOSITORY / "configs/readouts/local_frequency_synthetic_v1.yaml"
POSE_CONFIG = REPOSITORY / "configs/pams.yaml"
DEV_IDS = tuple(
    (REPOSITORY / "data/splits/ucfrep_526_dev_84.txt")
    .read_text(encoding="utf-8")
    .splitlines()
)
RUNNER = CliRunner()
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _write_dev_inputs(tmp_path: Path) -> tuple[Path, Path, PoseInputManifest]:
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split="dev",
        records=tuple(
            UnlabeledVideoRecord(
                video_id=video_id,
                video_path=f"videos/{video_id}.avi",
                video_sha256=hashlib.sha256(video_id.encode("utf-8")).hexdigest(),
            )
            for video_id in DEV_IDS
        ),
    )
    sidecar = tmp_path / "dev.inputs.json"
    sidecar.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    commitment = PoseInputCommitment(
        protocol="ucfrep_526",
        split="dev",
        record_total=84,
        identity_sha256=pose_input_identity_sha256(manifest.records),
        sidecar_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        sidecar_fingerprint=manifest.fingerprint,
    )
    commitment_path = tmp_path / "dev.commitment.json"
    commitment_path.write_text(
        json.dumps(commitment.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    return sidecar, commitment_path, manifest


def _patch_prediction_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    manifest: PoseInputManifest,
) -> None:
    pose_fingerprint = dev_module.load_config(POSE_CONFIG).pose_fingerprint
    sequences = tuple(
        PoseSequence(
            video_id=record.video_id,
            fps=30.0,
            xyz=np.zeros((16, 33, 3), dtype=np.float32),
            valid_mask=np.ones(16, dtype=np.bool_),
        )
        for record in manifest.records
    )
    snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=pose_fingerprint,
        entries=tuple(
            PoseCacheEntryReceipt(
                video_id=record.video_id,
                cache_sha256=hashlib.sha256(
                    f"cache:{record.video_id}".encode()
                ).hexdigest(),
                bytes=1,
            )
            for record in manifest.records
        ),
    )
    monkeypatch.setattr(dev_module, "clean_git_revision", lambda _root: "a" * 40)
    monkeypatch.setattr(
        dev_module,
        "load_pose_cache_set",
        lambda *_args, **_kwargs: (sequences, snapshot),
    )
    monkeypatch.setattr(
        dev_module.LocalFrequencyReadout,
        "predict",
        lambda _self, sample: CountResult(
            count=8,
            period_frames=32.0,
            expert_counts=(8, 8, 8),
            confidence=1.0,
            period_stream=np.zeros(sample.num_frames, dtype=np.float32),
        ),
    )


def _run_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], Path, Path]:
    sidecar, commitment, manifest = _write_dev_inputs(tmp_path)
    _patch_prediction_dependencies(monkeypatch, manifest)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    output = tmp_path / "prediction-run"
    receipt = run_dev_prediction(
        dev_inputs_path=sidecar,
        dev_commitment_path=commitment,
        pose_cache_dir=cache_dir,
        output_dir=output,
        config_path=FREQUENCY_CONFIG,
        pose_config_path=POSE_CONFIG,
        repository_root=REPOSITORY,
    )
    return receipt, output / "predictions.json", output / "prediction.receipt.json"


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
        for video_id in DEV_IDS
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


def test_prediction_boundary_has_no_target_or_test_input() -> None:
    parameters = set(inspect.signature(run_dev_prediction).parameters)
    assert parameters == {
        "dev_inputs_path",
        "dev_commitment_path",
        "pose_cache_dir",
        "output_dir",
        "config_path",
        "pose_config_path",
        "repository_root",
    }
    assert all("target" not in parameter and "test" not in parameter for parameter in parameters)


def test_cli_exposes_three_separated_boundaries() -> None:
    group = RUNNER.invoke(app, ["local-frequency", "--help"])
    predict = RUNNER.invoke(app, ["local-frequency", "dev-predict", "--help"])
    replay = RUNNER.invoke(app, ["local-frequency", "synthetic-replay", "--help"])
    assert group.exit_code == predict.exit_code == replay.exit_code == 0
    group_output = ANSI_ESCAPE.sub("", group.output)
    predict_output = ANSI_ESCAPE.sub("", predict.output)
    replay_output = ANSI_ESCAPE.sub("", replay.output)
    assert {"synthetic-replay", "dev-predict", "dev-score"} <= set(group_output.split())
    assert "--dev-targets" not in predict_output
    assert "--test-identity" not in predict_output
    assert "--targets" not in replay_output
    assert "--sequence" not in replay_output


def _matching_synthetic_report() -> SyntheticFreezeReport:
    candidate = SyntheticCandidateScore(
        candidate="frame_centered.min064.median",
        mean_absolute_error=0.02,
        normalized_mean_absolute_error=0.0025,
        exact_rate=0.98,
        obo=1.0,
    )
    return SyntheticFreezeReport(
        selected_candidate=candidate.candidate,
        sample_ids=tuple(f"synthetic-{index:02d}" for index in range(50)),
        scores=(candidate,),
    )


def test_synthetic_replay_has_no_external_samples_and_writes_hashed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = set(inspect.signature(run_synthetic_freeze_replay).parameters)
    assert parameters == {"config_path", "output_dir", "repository_root"}
    monkeypatch.setattr(dev_module, "clean_git_revision", lambda _root: "b" * 40)
    monkeypatch.setattr(
        dev_module,
        "replay_synthetic_freeze",
        lambda _config, *, repository_root: _matching_synthetic_report(),
    )

    receipt = run_synthetic_freeze_replay(
        config_path=FREQUENCY_CONFIG,
        output_dir=tmp_path / "synthetic-replay",
        repository_root=REPOSITORY,
    )
    replay_path = Path(str(receipt["replay_path"]))
    replay_receipt_path = Path(str(receipt["replay_receipt_path"]))
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    assert payload["expected_replay_match"] is True
    assert payload["expected_replay"] == payload["observed_replay"]
    assert receipt["replay_sha256"] == hashlib.sha256(replay_path.read_bytes()).hexdigest()
    assert receipt["replay_receipt_sha256"] == hashlib.sha256(
        replay_receipt_path.read_bytes()
    ).hexdigest()
    assert len(receipt["config_file_sha256"]) == 64
    assert len(receipt["selection_scope_sha256"]) == 64
    assert len(receipt["code_sha256"]) == 64
    assert len(receipt["source_git_sha"]) == 40


def test_synthetic_replay_mismatch_fails_without_success_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mismatched = SyntheticFreezeReport(
        selected_candidate="xyz.min096.mean",
        sample_ids=tuple(f"synthetic-{index:02d}" for index in range(50)),
        scores=(
            SyntheticCandidateScore(
                candidate="xyz.min096.mean",
                mean_absolute_error=1.0,
                normalized_mean_absolute_error=1.0,
                exact_rate=0.0,
                obo=0.0,
            ),
        ),
    )
    monkeypatch.setattr(dev_module, "clean_git_revision", lambda _root: "b" * 40)
    monkeypatch.setattr(
        dev_module,
        "replay_synthetic_freeze",
        lambda _config, *, repository_root: mismatched,
    )
    output = tmp_path / "synthetic-replay"
    with pytest.raises(ValueError, match="does not exactly match"):
        run_synthetic_freeze_replay(
            config_path=FREQUENCY_CONFIG,
            output_dir=output,
            repository_root=REPOSITORY,
        )
    assert not (output / "synthetic-freeze-replay.json").exists()
    assert not (output / "synthetic-freeze-replay.receipt.json").exists()


def test_prediction_freezes_target_free_rows_and_all_required_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, predictions_path, receipt_path = _run_prediction(tmp_path, monkeypatch)
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))

    assert payload["split"] == "dev"
    assert payload["record_total"] == 84
    assert len(payload["records"]) == 84
    assert all("target" not in row and "action" not in row for row in payload["records"])
    assert payload["eligible_for_paper_table"] is False
    for field in (
        "config_file_sha256",
        "config_fingerprint",
        "pose_config_file_sha256",
        "dev_inputs_sha256",
        "dev_commitment_sha256",
        "dev_identity_sha256",
        "pose_cache_set_sha256",
        "pose_cache_snapshot_sha256",
        "code_sha256",
    ):
        assert len(payload[field]) == 64
    assert receipt["predictions_sha256"] == hashlib.sha256(
        predictions_path.read_bytes()
    ).hexdigest()
    assert receipt["prediction_receipt_sha256"] == hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()


def test_score_rejects_prediction_tampering_before_loading_targets(
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
        lambda _path: pytest.fail("target loader crossed the frozen-prediction boundary"),
    )

    with pytest.raises(ValueError, match="SHA-256"):
        score_dev_predictions(
            predictions_path=predictions_path,
            prediction_receipt_path=receipt_path,
            dev_targets_path=tmp_path / "must-not-be-opened.targets.json",
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )


def test_score_emits_10k_paired_ci_per_video_and_terminal_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, predictions_path, receipt_path = _run_prediction(tmp_path, monkeypatch)
    targets_path = _write_dev_targets(tmp_path, monkeypatch)
    output = tmp_path / "score"
    receipt = score_dev_predictions(
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
    assert set(report["confidence_intervals"]) == {"nmae", "mae", "rmse", "obo", "exact"}
    assert report["nmae"] == report["mae"] == report["rmse"] == 0.0
    assert report["obo"] == 1.0
    assert len(report["per_video"]) == len(payload["predictions"]) == 84
    assert receipt["evaluation_sha256"] == hashlib.sha256(
        evaluation_path.read_bytes()
    ).hexdigest()
    assert receipt["evaluation_receipt_sha256"] == hashlib.sha256(
        evaluation_receipt_path.read_bytes()
    ).hexdigest()
    assert len(receipt["prediction_sha256"]) == 64
    assert len(receipt["dev_targets_sha256"]) == 64
    assert len(receipt["prediction_source_git_sha"]) == 40
    assert len(receipt["prediction_code_sha256"]) == 64
    assert len(receipt["scoring_source_git_sha"]) == 40
    assert len(receipt["scoring_code_sha256"]) == 64
