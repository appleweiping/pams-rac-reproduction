from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from typer.main import get_command
from typer.testing import CliRunner

import pams.teacher_period_dev as teacher
from pams.cli import app
from pams.config import load_config
from pams.data import UnlabeledVideoRecord, pose_input_identity_sha256
from pams.types import PoseSequence

REPOSITORY = Path(__file__).parents[1]
V8_CONFIG = REPOSITORY / "configs/experiments/pams_longest_contiguous_track_v8.yaml"
DEV_IDS = tuple(
    (REPOSITORY / "data/splits/ucfrep_526_dev_84.txt").read_text(encoding="utf-8").splitlines()
)
RUNNER = CliRunner()
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class _FakeEncoder(torch.nn.Module):
    def forward_with_pre_pe(
        self,
        inputs: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del valid
        return inputs, inputs


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.encoder = _FakeEncoder()


def _sequence(
    valid_count: int,
    *,
    frames: int = 256,
    video_id: str | None = None,
) -> PoseSequence:
    mask = np.zeros(frames, dtype=np.bool_)
    mask[:valid_count] = True
    return PoseSequence(
        video_id=video_id or f"valid-{valid_count}",
        fps=30.0,
        xyz=np.zeros((frames, 33, 3), dtype=np.float32),
        valid_mask=mask,
    )


def test_prediction_boundary_has_no_target_argument_and_cli_is_separated() -> None:
    parameters = set(inspect.signature(teacher.run_dev_prediction).parameters)
    assert all("target" not in name for name in parameters)

    group = RUNNER.invoke(app, ["teacher-period", "--help"], terminal_width=240)
    predict = RUNNER.invoke(
        app,
        ["teacher-period", "dev-predict", "--help"],
        terminal_width=240,
    )
    score = RUNNER.invoke(
        app,
        ["teacher-period", "dev-score", "--help"],
        terminal_width=240,
    )
    assert group.exit_code == predict.exit_code == score.exit_code == 0
    assert {"dev-predict", "dev-score"} <= set(ANSI_ESCAPE.sub("", group.output).split())
    prediction_help = ANSI_ESCAPE.sub("", predict.output)
    assert "--dev-targets" not in prediction_help
    assert "--targets" not in prediction_help
    root_command = get_command(app)
    teacher_command = root_command.commands["teacher-period"]
    predict_command = teacher_command.commands["dev-predict"]
    predict_options = {
        option for parameter in predict_command.params for option in getattr(parameter, "opts", ())
    }
    assert "--checkpoint-progress" in predict_options
    assert "--checkpoint-completion-receipt" in predict_options


def test_direct_teacher_formula_uses_valid_minus_one_and_half_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        teacher,
        "estimate_period_from_projected_pose",
        lambda *_args, **_kwargs: (
            torch.tensor([4.0]),
            torch.tensor([0.75]),
        ),
    )
    row = teacher.predict_teacher_period_direct(
        _FakeModel(),
        _sequence(11),
        load_config(V8_CONFIG),
    )
    assert row.raw_count == 2.5
    assert row.rounded_count == 3
    assert row.period_frames == 4.0
    assert row.confidence == 0.75
    assert row.expert_counts == (3, 3, 3)
    assert row.valid_frames == 11
    assert len(row.period_stream) == 256
    assert set(row.period_stream) == {4.0}


@pytest.mark.parametrize(
    ("valid_count", "confidence"),
    ((16, 0.0), (1, 1.0), (0, 1.0)),
)
def test_direct_teacher_zeroes_count_without_period_evidence(
    valid_count: int,
    confidence: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        teacher,
        "estimate_period_from_projected_pose",
        lambda *_args, **_kwargs: (
            torch.tensor([8.0]),
            torch.tensor([confidence]),
        ),
    )
    row = teacher.predict_teacher_period_direct(
        _FakeModel(),
        _sequence(valid_count),
        load_config(V8_CONFIG),
    )
    assert row.raw_count == 0.0
    assert row.rounded_count == 0
    assert row.expert_counts == (0, 0, 0)


def test_direct_teacher_batches_are_frozen_to_32(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_sizes: list[int] = []

    def fake_estimator(
        projected: torch.Tensor,
        *_args: object,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_sizes.append(projected.shape[0])
        return (
            torch.full((projected.shape[0],), 4.0, device=projected.device),
            torch.full((projected.shape[0],), 0.75, device=projected.device),
        )

    monkeypatch.setattr(teacher, "estimate_period_from_projected_pose", fake_estimator)
    sequences = tuple(_sequence(11, video_id=f"batch-{index:03d}") for index in range(65))
    rows = teacher.predict_teacher_period_direct_batches(
        _FakeModel(),
        sequences,
        load_config(V8_CONFIG),
    )
    assert batch_sizes == [32, 32, 1]
    assert tuple(row.video_id for row in rows) == tuple(sequence.video_id for sequence in sequences)
    assert all(row.raw_count == 2.5 and row.rounded_count == 3 for row in rows)


def test_prediction_row_rejects_formula_or_stream_substitution() -> None:
    common = {
        "video_id": DEV_IDS[0],
        "video_sha256": "a" * 64,
        "rounded_count": 8,
        "period_frames": 32.0,
        "valid_frames": 256,
        "expert_counts": (8, 8, 8),
        "confidence": 1.0,
    }
    with pytest.raises(ValueError, match="direct-period formula"):
        teacher.TeacherPeriodPredictionRow(
            **common,
            raw_count=8.0,
            period_stream=(32.0,) * 256,
        )
    with pytest.raises(ValueError, match="exactly 256"):
        teacher.TeacherPeriodPredictionRow(
            **common,
            raw_count=255.0 / 32.0,
            period_stream=(32.0,),
        )


def test_exact_v8_gate_rejects_any_checkpoint_substitution(tmp_path: Path) -> None:
    checkpoint = tmp_path / "encoder.pt"
    progress = tmp_path / "encoder.jsonl"
    receipt = tmp_path / "encoder.completed.json"
    checkpoint.write_bytes(b"not-the-v8-encoder")
    progress.write_bytes(b"not-the-v8-progress")
    receipt.write_bytes(b"not-the-v8-receipt")
    with pytest.raises(ValueError, match="exact frozen v8 encoder"):
        teacher._require_exact_v8_bindings(
            config_path=V8_CONFIG,
            config=load_config(V8_CONFIG),
            checkpoint_path=checkpoint,
            progress_path=progress,
            completion_receipt_path=receipt,
            input_hashes=dict(teacher._V8_INPUT_SHA256),
        )


def _prediction_bundle(tmp_path: Path) -> tuple[Path, Path]:
    records = tuple(
        UnlabeledVideoRecord(
            video_id=video_id,
            video_path=f"videos/{video_id}.avi",
            video_sha256=hashlib.sha256(video_id.encode()).hexdigest(),
        )
        for video_id in DEV_IDS
    )
    rows = tuple(
        teacher.TeacherPeriodPredictionRow(
            video_id=record.video_id,
            video_sha256=record.video_sha256 or "",
            raw_count=255.0 / 32.0,
            rounded_count=8,
            period_frames=32.0,
            valid_frames=256,
            expert_counts=(8, 8, 8),
            confidence=1.0,
            period_stream=(32.0,) * 256,
        )
        for record in records
    )
    code_files = teacher._code_file_hashes()
    artifact = teacher.TeacherPeriodPredictionArtifact(
        artifact_type="pams_teacher_period_direct_dev_predictions",
        classification=teacher._CLASSIFICATION,
        protocol="ucfrep_526",
        split="dev",
        method_key=teacher._METHOD_KEY,
        rounding=teacher._ROUNDING,
        record_total=84,
        config_file_sha256="a" * 64,
        config_fingerprint="b" * 64,
        pose_fingerprint="c" * 64,
        protocol_identity_sha256="d" * 64,
        training_identity_sha256="e" * 64,
        train_inputs_sha256="1" * 64,
        train_commitment_sha256="2" * 64,
        dev_inputs_sha256="3" * 64,
        dev_commitment_sha256="4" * 64,
        test_identity_inputs_sha256="5" * 64,
        test_identity_commitment_sha256="6" * 64,
        dev_identity_sha256=pose_input_identity_sha256(records),
        training_pose_cache_set_sha256="7" * 64,
        dev_pose_cache_set_sha256="8" * 64,
        dev_pose_cache_snapshot_sha256="9" * 64,
        checkpoint_sha256=teacher._V8_ENCODER_SHA256,
        checkpoint_progress_sha256=teacher._V8_ENCODER_PROGRESS_SHA256,
        checkpoint_completion_receipt_sha256=(teacher._V8_ENCODER_COMPLETION_RECEIPT_SHA256),
        checkpoint_source_git_sha="a" * 40,
        checkpoint_container_image_id=f"sha256:{'b' * 64}",
        checkpoint_container_environment_sha256="c" * 64,
        prediction_source_git_sha="d" * 40,
        prediction_container_image_id=f"sha256:{'e' * 64}",
        prediction_container_environment_sha256="f" * 64,
        prediction_code_files_sha256=code_files,
        prediction_code_sha256=teacher.sha256_json(code_files),
        records=rows,
    )
    predictions = tmp_path / "predictions.json"
    predictions.write_bytes(teacher.strict_dev._encoded_json(artifact.model_dump(mode="json")))
    receipt = teacher.TeacherPeriodPredictionReceipt(
        artifact_type="pams_teacher_period_direct_dev_prediction_receipt",
        protocol="ucfrep_526",
        split="dev",
        method_key=teacher._METHOD_KEY,
        prediction_file="predictions.json",
        prediction_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
        prediction_bytes=predictions.stat().st_size,
        config_fingerprint=artifact.config_fingerprint,
        dev_identity_sha256=artifact.dev_identity_sha256,
        checkpoint_sha256=artifact.checkpoint_sha256,
        checkpoint_progress_sha256=artifact.checkpoint_progress_sha256,
        checkpoint_completion_receipt_sha256=(artifact.checkpoint_completion_receipt_sha256),
        training_pose_cache_set_sha256=artifact.training_pose_cache_set_sha256,
        dev_pose_cache_set_sha256=artifact.dev_pose_cache_set_sha256,
        dev_pose_cache_snapshot_sha256=artifact.dev_pose_cache_snapshot_sha256,
        prediction_source_git_sha=artifact.prediction_source_git_sha,
        prediction_container_image_id=artifact.prediction_container_image_id,
        prediction_container_environment_sha256=(artifact.prediction_container_environment_sha256),
        prediction_code_sha256=artifact.prediction_code_sha256,
        command=("pams", "teacher-period", "dev-predict"),
        hardware={},
    )
    receipt_path = tmp_path / "prediction.receipt.json"
    receipt_path.write_bytes(teacher.strict_dev._encoded_json(receipt.model_dump(mode="json")))
    return predictions, receipt_path


def test_score_rejects_tampered_predictions_before_opening_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions, receipt = _prediction_bundle(tmp_path)
    predictions.write_bytes(predictions.read_bytes() + b"\n")
    monkeypatch.setattr(
        teacher,
        "load_dev_target_manifest",
        lambda _path: pytest.fail("target loader crossed prediction firewall"),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        teacher.score_dev_predictions(
            predictions_path=predictions,
            prediction_receipt_path=receipt,
            dev_targets_path=tmp_path / "must-not-open.targets.json",
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )


def test_score_rejects_self_consistent_non_v8_artifact_before_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions, receipt = _prediction_bundle(tmp_path)
    monkeypatch.setattr(
        teacher,
        "load_dev_target_manifest",
        lambda _path: pytest.fail("target loader crossed exact-v8 gate"),
    )
    with pytest.raises(ValueError, match="exact frozen v8 run"):
        teacher.score_dev_predictions(
            predictions_path=predictions,
            prediction_receipt_path=receipt,
            dev_targets_path=tmp_path / "must-not-open.targets.json",
            output_dir=tmp_path / "score",
            repository_root=REPOSITORY,
        )


def test_prediction_rows_never_contain_target_or_action(tmp_path: Path) -> None:
    predictions, _ = _prediction_bundle(tmp_path)
    payload = json.loads(predictions.read_text(encoding="utf-8"))
    assert payload["eligible_for_paper_table"] is False
    assert payload["test_evaluation_authorized"] is False
    assert len(payload["records"]) == 84
    assert all("target" not in row and "action" not in row for row in payload["records"])
    assert all({"raw_count", "rounded_count"} <= set(row) for row in payload["records"])
