from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

import pams.data as data_module
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
from pams.types import PoseSequence

REPOSITORY = Path(__file__).parents[1]
RUNNER = REPOSITORY / "scripts/server/run_pose_spectral_consensus_dev_exploratory.py"
DEV_IDS = tuple(
    (REPOSITORY / "data/splits/ucfrep_526_dev_84.txt").read_text(encoding="utf-8").splitlines()
)
POSE_FINGERPRINT = "3dd0388320095796f42aa82d904073b6478b073ed351394ef8d03c30798a0116"


def _load_runner() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "pose_spectral_consensus_exploratory_test_module",
        RUNNER,
    )
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load exploratory runner")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


runner = _load_runner()


def _periodic_sequence(
    *,
    video_id: str,
    period: int = 32,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    corrupt_joints: int = 0,
) -> PoseSequence:
    time = np.arange(256, dtype=np.float64)
    xyz = np.zeros((256, 33, 3), dtype=np.float32)
    for joint in range(33):
        phase = (joint % 7) * 0.11
        # The deliberately stronger second harmonic tests T/2,T,2T
        # disambiguation rather than an easy single-sinusoid peak.
        xyz[:, joint, 0] = 0.4 * np.sin(2.0 * np.pi * time / period + phase) + 0.75 * np.sin(
            4.0 * np.pi * time / period + phase * 0.3
        )
        xyz[:, joint, 1] = 0.3 * np.cos(2.0 * np.pi * time / period - phase) + 0.4 * np.cos(
            4.0 * np.pi * time / period
        )
        xyz[:, joint, 2] = 0.2 * np.sin(2.0 * np.pi * time / period + phase * 2.0)
    if corrupt_joints:
        noise = np.random.default_rng(3407).normal(
            0.0,
            2.0,
            size=xyz[:, :corrupt_joints].shape,
        )
        xyz[:, :corrupt_joints] += noise.astype(np.float32)
    xyz += np.asarray(translation, dtype=np.float32)[None, None, :]
    return PoseSequence(
        video_id=video_id,
        fps=30.0,
        xyz=xyz,
        valid_mask=np.ones(256, dtype=np.bool_),
    )


@pytest.mark.parametrize("period", [8, 16, 24, 32, 48, 64, 96])
def test_multi_joint_harmonic_consensus_recovers_synthetic_period(period: int) -> None:
    result = runner.estimate_pose_spectral_consensus(
        _periodic_sequence(video_id=f"period-{period}", period=period)
    )
    assert result.selection_source == "multi-joint-multi-window-harmonic-consensus"
    assert result.period_frames == pytest.approx(period, abs=1.0)
    assert result.rounded_count == runner.round_count(255.0 / result.period_frames)
    assert result.window_count > result.scale_count >= 3
    assert result.active_joint_count >= 30
    assert result.harmonic_family_periods[1] == result.period_frames
    assert all(np.isfinite(result.harmonic_family_scores))


def test_consensus_is_translation_invariant_and_robust_to_bad_joints() -> None:
    original = _periodic_sequence(
        video_id="original",
        period=32,
        corrupt_joints=5,
    )
    translated = _periodic_sequence(
        video_id="translated",
        period=32,
        corrupt_joints=5,
        translation=(3.0, -2.0, 1.0),
    )
    first = runner.estimate_pose_spectral_consensus(original)
    shifted = runner.estimate_pose_spectral_consensus(translated)
    assert first.period_frames == shifted.period_frames == 32.0
    assert first.rounded_count == shifted.rounded_count == 8
    assert first.expert_counts == shifted.expert_counts == (8, 8, 8)
    assert first.confidence == pytest.approx(shifted.confidence, abs=1e-6)


def test_all_invalid_pose_abstains_without_spectral_evidence() -> None:
    sequence = PoseSequence(
        video_id="all-invalid",
        fps=30.0,
        xyz=np.ones((256, 33, 3), dtype=np.float32),
        valid_mask=np.zeros(256, dtype=np.bool_),
    )
    result = runner.estimate_pose_spectral_consensus(sequence)
    assert result.selection_source == "no-evidence"
    assert result.raw_count == result.confidence == 0.0
    assert result.rounded_count == 0
    assert result.expert_counts == (0, 0, 0)
    assert result.valid_frames == 0


def test_frozen_parameters_explicitly_include_joint_window_and_harmonic_family() -> None:
    parameters = runner._algorithm_parameters()
    assert parameters["window_lengths"] == [256, 192, 128, 96, 64]
    assert parameters["minimum_active_joints"] == 3
    assert parameters["harmonic_family"] == ["T/2", "T", "2T"]
    assert parameters["duration_convention"] == "sampled_frame_intervals_T_minus_1"
    assert runner.CLASSIFICATION.startswith("inferred exploratory target-free")


def test_cli_prediction_and_score_mount_boundaries_are_disjoint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = runner.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["predict", "--help"])
    prediction_help = capsys.readouterr().out
    with pytest.raises(SystemExit):
        parser.parse_args(["score", "--help"])
    score_help = capsys.readouterr().out

    assert "--dev-targets" not in prediction_help
    assert "--test" not in prediction_help
    assert "--pose-cache-dir" in prediction_help
    assert "--dev-inputs" in prediction_help
    assert "--dev-targets" in score_help
    assert "--pose-cache-dir" not in score_help
    assert "--dev-inputs" not in score_help
    assert "--test" not in score_help
    assert "target" not in inspect.getsource(runner.estimate_pose_spectral_consensus)


def _write_dev_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, PoseInputManifest]:
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split="dev",
        records=tuple(
            UnlabeledVideoRecord(
                video_id=video_id,
                video_path=f"videos/{video_id}.avi",
                video_sha256=hashlib.sha256(video_id.encode()).hexdigest(),
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
    commitment_path = tmp_path / "dev.inputs.commitment.json"
    commitment_path.write_text(
        json.dumps(commitment.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    return sidecar, commitment_path, manifest


def _patch_pose_prediction(
    monkeypatch: pytest.MonkeyPatch,
    manifest: PoseInputManifest,
) -> None:
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
        pose_fingerprint=POSE_FINGERPRINT,
        entries=tuple(
            PoseCacheEntryReceipt(
                video_id=record.video_id,
                cache_sha256=hashlib.sha256(f"pose:{record.video_id}".encode()).hexdigest(),
                bytes=1,
            )
            for record in manifest.records
        ),
    )
    monkeypatch.setattr(
        runner,
        "load_pose_cache_set",
        lambda *_args, **_kwargs: (sequences, snapshot),
    )
    monkeypatch.setattr(
        runner,
        "estimate_pose_spectral_consensus",
        lambda _sequence: runner.SpectralConsensusEstimate(
            period_frames=255.0 / 8.0,
            raw_count=8.0,
            rounded_count=8,
            confidence=1.0,
            selection_source="multi-joint-multi-window-harmonic-consensus",
            valid_frames=16,
            total_frames=16,
            active_joint_count=33,
            window_count=5,
            scale_count=3,
            expert_counts=(8, 8, 8),
            harmonic_family_periods=(255.0 / 16.0, 255.0 / 8.0, 255.0 / 4.0),
            harmonic_family_scores=(0.5, 1.0, 0.25),
        ),
    )


def _run_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    sidecar, commitment, manifest = _write_dev_inputs(tmp_path)
    _patch_pose_prediction(monkeypatch, manifest)
    cache = tmp_path / "pose"
    cache.mkdir()
    output = tmp_path / "prediction"
    runner.run_predict(
        Namespace(
            source_git_sha="a" * 40,
            repository=REPOSITORY,
            runner=RUNNER,
            dev_inputs=sidecar,
            dev_commitment=commitment,
            pose_cache_dir=cache,
            pose_fingerprint=POSE_FINGERPRINT,
            container_image_id="sha256:" + "b" * 64,
            output_dir=output,
        )
    )
    return output / "predictions.json", output / "prediction.receipt.json"


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


def test_prediction_artifact_is_target_free_and_hash_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions_path, receipt_path = _run_prediction(tmp_path, monkeypatch)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert predictions["record_total"] == len(predictions["records"]) == 84
    assert predictions["mount_audit"]["dev_targets_mounted"] is False
    assert predictions["mount_audit"]["test_targets_mounted"] is False
    assert all("target" not in row and "action" not in row for row in predictions["records"])
    assert receipt["prediction_sha256"] == hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    assert receipt["label_firewall"]["gt_count_or_action_oracle_used"] is False


def test_score_rejects_tampering_before_target_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions_path, receipt_path = _run_prediction(tmp_path, monkeypatch)
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    predictions["records"][0]["raw_count"] = 9.0
    predictions["records"][0]["rounded_count"] = 9
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "load_dev_target_manifest",
        lambda _path: pytest.fail("target loader crossed prediction receipt validation"),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        runner.run_score(
            Namespace(
                source_git_sha="a" * 40,
                repository=REPOSITORY,
                runner=RUNNER,
                predictions=predictions_path,
                prediction_receipt=receipt_path,
                dev_targets=tmp_path / "must-not-open.targets.json",
                output_dir=tmp_path / "score",
            )
        )


def test_separate_scorer_emits_paired_bootstrap_and_no_test_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions_path, receipt_path = _run_prediction(tmp_path, monkeypatch)
    targets_path = _write_dev_targets(tmp_path, monkeypatch)
    output = tmp_path / "score"
    runner.run_score(
        Namespace(
            source_git_sha="a" * 40,
            repository=REPOSITORY,
            runner=RUNNER,
            predictions=predictions_path,
            prediction_receipt=receipt_path,
            dev_targets=targets_path,
            output_dir=output,
        )
    )
    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    receipt = json.loads((output / "evaluation.receipt.json").read_text(encoding="utf-8"))
    assert evaluation["bootstrap_pairing"] == "paired_prediction_target_rows"
    assert evaluation["metrics"]["bootstrap_samples"] == 10_000
    assert evaluation["metrics"]["nmae"] == 0.0
    assert evaluation["metrics"]["obo"] == 1.0
    assert evaluation["mount_audit"]["dev_pose_mounted"] is False
    assert evaluation["mount_audit"]["test_targets_mounted"] is False
    assert (
        receipt["evaluation_sha256"]
        == hashlib.sha256((output / "evaluation.json").read_bytes()).hexdigest()
    )
