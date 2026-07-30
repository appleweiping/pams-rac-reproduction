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

import pams.position_acf_dev as position
import pams.teacher_period_dev as teacher
from pams.cli import app
from pams.config import load_config
from pams.data import UnlabeledVideoRecord, pose_input_identity_sha256
from pams.period import ProjectedPositionSpectrumDiagnostic
from pams.types import PoseSequence

REPOSITORY = Path(__file__).parents[1]
V8_CONFIG = REPOSITORY / "configs/experiments/pams_longest_contiguous_track_v8.yaml"
READOUT_CONFIG = (
    REPOSITORY / "configs/readouts/projected_position_acf_direct_v1.yaml"
)
DEV_IDS = tuple(
    (REPOSITORY / "data/splits/ucfrep_526_dev_84.txt")
    .read_text(encoding="utf-8")
    .splitlines()
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
        projected = inputs.flatten(start_dim=2)
        return projected, projected


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.encoder = _FakeEncoder()


def _sequence(
    valid_count: int,
    *,
    video_id: str = "sample",
) -> PoseSequence:
    valid = np.zeros(256, dtype=np.bool_)
    valid[:valid_count] = True
    return PoseSequence(
        video_id=video_id,
        fps=30.0,
        xyz=np.zeros((256, 33, 3), dtype=np.float32),
        valid_mask=valid,
    )


def _fake_diagnostics(
    valid_mask: torch.Tensor,
    *,
    selected_bin: int | None,
    confidence: float,
    selected_period: float,
) -> tuple[ProjectedPositionSpectrumDiagnostic, ...]:
    bins = tuple(range(2, 65))
    return tuple(
        ProjectedPositionSpectrumDiagnostic(
            valid_length=int(mask.sum()),
            allowed_bins=bins,
            frequencies=tuple(bin_index / 256.0 for bin_index in bins),
            periods=tuple(256.0 / bin_index for bin_index in bins),
            power_shares=tuple(
                (
                    confidence
                    if bin_index == selected_bin
                    else (
                        1.0 - confidence
                        if selected_bin is not None
                        and confidence > 0.0
                        and bin_index == (2 if selected_bin != 2 else 3)
                        else 0.0
                    )
                )
                for bin_index in bins
            ),
            selected_bin=selected_bin,
            selected_period=selected_period,
            confidence=confidence,
        )
        for mask in valid_mask
    )


def _patch_estimators(
    monkeypatch: pytest.MonkeyPatch,
    *,
    period: float = 4.0,
    confidence: float = 0.75,
    selected_bin: int | None = 64,
    batch_sizes: list[int] | None = None,
) -> None:
    def position_estimator(
        projected: torch.Tensor,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_sizes is not None:
            batch_sizes.append(projected.shape[0])
        return (
            torch.full((projected.shape[0],), period, device=projected.device),
            torch.full(
                (projected.shape[0],),
                confidence,
                device=projected.device,
            ),
        )

    def diagnostics(
        _projected: torch.Tensor,
        **kwargs: object,
    ) -> tuple[ProjectedPositionSpectrumDiagnostic, ...]:
        valid_mask = kwargs["valid_mask"]
        assert isinstance(valid_mask, torch.Tensor)
        return _fake_diagnostics(
            valid_mask,
            selected_bin=selected_bin,
            confidence=confidence,
            selected_period=period,
        )

    def teacher_estimator(
        projected: torch.Tensor,
        **_kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.full((projected.shape[0],), period, device=projected.device),
            torch.full((projected.shape[0],), 0.5, device=projected.device),
        )

    monkeypatch.setattr(
        position,
        "estimate_period_from_projected_position",
        position_estimator,
    )
    monkeypatch.setattr(
        position,
        "projected_position_vector_acf_diagnostics",
        diagnostics,
    )
    monkeypatch.setattr(
        position,
        "estimate_period_from_projected_pose",
        teacher_estimator,
    )


def test_predictor_has_no_target_argument_and_cli_is_separated() -> None:
    parameters = set(inspect.signature(position.run_dev_prediction).parameters)
    assert all("target" not in name for name in parameters)

    group = RUNNER.invoke(app, ["position-acf", "--help"], terminal_width=240)
    predict = RUNNER.invoke(
        app,
        ["position-acf", "dev-predict", "--help"],
        terminal_width=240,
    )
    assert group.exit_code == predict.exit_code == 0
    assert {"dev-predict", "dev-score"} <= set(
        ANSI_ESCAPE.sub("", group.output).split()
    )
    prediction_help = ANSI_ESCAPE.sub("", predict.output)
    assert "--target" not in prediction_help
    root_command = get_command(app)
    command = root_command.commands["position-acf"].commands["dev-predict"]
    options = {
        option
        for parameter in command.params
        for option in getattr(parameter, "opts", ())
    }
    assert "--readout-config" in options
    assert "--checkpoint-completion-receipt" in options


def test_readout_config_is_exact_and_tamper_evident(tmp_path: Path) -> None:
    loaded = position.load_position_acf_readout_config(READOUT_CONFIG)
    assert loaded.fingerprint == position._READOUT_CONFIG_FINGERPRINT

    tampered = tmp_path / READOUT_CONFIG.name
    tampered.write_bytes(READOUT_CONFIG.read_bytes() + b"\n# tampered\n")
    with pytest.raises(ValueError, match="bytes differ"):
        position.load_position_acf_readout_config(tampered)


def test_position_acf_formula_spectrum_and_half_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_estimators(monkeypatch)
    row = position.predict_position_acf_direct_batches(
        _FakeModel(),
        (_sequence(11),),
        load_config(V8_CONFIG),
    )[0]
    assert row.raw_count == 2.5
    assert row.rounded_count == 3
    assert row.expert_counts == (3, 3, 3)
    assert row.period_frames == 4.0
    assert row.valid_frames == 11
    assert len(row.allowed_spectrum) == 63
    assert tuple(item.bin_index for item in row.allowed_spectrum) == tuple(
        range(2, 65)
    )
    assert row.position_selected_bin == 64
    assert len(row.period_stream) == 256


def test_real_position_acf_api_integrates_with_prediction_row() -> None:
    time = np.arange(256, dtype=np.float32)
    xyz = np.zeros((256, 33, 3), dtype=np.float32)
    xyz[:, 0, 0] = np.sin(2.0 * np.pi * time / 32.0)
    sequence = PoseSequence(
        video_id="synthetic-period-32",
        fps=30.0,
        xyz=xyz,
        valid_mask=np.ones(256, dtype=np.bool_),
    )
    row = position.predict_position_acf_direct_batches(
        _FakeModel(),
        (sequence,),
        load_config(V8_CONFIG),
    )[0]
    assert row.confidence > 0.0
    assert row.position_selected_bin is not None
    assert len(row.allowed_spectrum) == 63


@pytest.mark.parametrize(
    ("total", "expected_batches"),
    ((1, [1]), (33, [32, 1])),
)
def test_position_acf_batching_is_fixed_to_32(
    total: int,
    expected_batches: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[int] = []
    _patch_estimators(monkeypatch, batch_sizes=observed)
    sequences = tuple(
        _sequence(11, video_id=f"batch-{index:03d}") for index in range(total)
    )
    rows = position.predict_position_acf_direct_batches(
        _FakeModel(),
        sequences,
        load_config(V8_CONFIG),
    )
    assert observed == expected_batches
    assert tuple(row.video_id for row in rows) == tuple(
        sequence.video_id for sequence in sequences
    )


def test_zero_confidence_has_fixed_zero_spectrum_and_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_estimators(
        monkeypatch,
        period=128.0,
        confidence=0.0,
        selected_bin=None,
    )
    row = position.predict_position_acf_direct_batches(
        _FakeModel(),
        (_sequence(256),),
        load_config(V8_CONFIG),
    )[0]
    assert row.raw_count == row.rounded_count == 0
    assert row.position_selected_bin is None
    assert len(row.allowed_spectrum) == 63
    assert sum(item.power_share for item in row.allowed_spectrum) == 0.0


def test_exact_v8_gate_rejects_checkpoint_substitution(tmp_path: Path) -> None:
    checkpoint = tmp_path / "encoder.pt"
    progress = tmp_path / "encoder.jsonl"
    receipt = tmp_path / "encoder.completed.json"
    checkpoint.write_bytes(b"wrong")
    progress.write_bytes(b"wrong")
    receipt.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="exact frozen v8 encoder"):
        position._require_exact_v8_bindings(
            config_path=V8_CONFIG,
            config=load_config(V8_CONFIG),
            checkpoint_path=checkpoint,
            progress_path=progress,
            completion_receipt_path=receipt,
            input_hashes=dict(position._V8_INPUT_SHA256),
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
    spectrum = tuple(
        position.PositionACFSpectrumBin(
            bin_index=bin_index,
            period_frames=256.0 / bin_index,
            power_share=1.0 if bin_index == 8 else 0.0,
        )
        for bin_index in range(2, 65)
    )
    rows = tuple(
        position.PositionACFPredictionRow(
            video_id=record.video_id,
            video_sha256=record.video_sha256 or "",
            raw_count=255.0 / 32.0,
            rounded_count=8,
            period_frames=32.0,
            valid_frames=256,
            expert_counts=(8, 8, 8),
            confidence=1.0,
            position_selected_bin=8,
            allowed_spectrum=spectrum,
            teacher_bin_relation=position._teacher_relation(
                teacher_period=32.0,
                teacher_confidence=1.0,
                position_selected_bin=8,
            ),
            period_stream=(32.0,) * 256,
        )
        for record in records
    )
    code_files = position._code_file_hashes()
    artifact = position.PositionACFPredictionArtifact(
        artifact_type="pams_position_acf_direct_dev_predictions",
        classification=position._CLASSIFICATION,
        protocol="ucfrep_526",
        split="dev",
        method_key=position._METHOD_KEY,
        rounding=position._ROUNDING,
        record_total=84,
        config_file_sha256="a" * 64,
        config_fingerprint="b" * 64,
        readout_config_file_sha256=position._READOUT_CONFIG_FILE_SHA256,
        readout_config_fingerprint=position._READOUT_CONFIG_FINGERPRINT,
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
        checkpoint_sha256=position._V8_ENCODER_SHA256,
        checkpoint_progress_sha256=position._V8_ENCODER_PROGRESS_SHA256,
        checkpoint_completion_receipt_sha256=(
            position._V8_ENCODER_COMPLETION_RECEIPT_SHA256
        ),
        checkpoint_source_git_sha="a" * 40,
        checkpoint_container_image_id=f"sha256:{'b' * 64}",
        checkpoint_container_environment_sha256="c" * 64,
        prediction_source_git_sha="d" * 40,
        prediction_container_image_id=f"sha256:{'e' * 64}",
        prediction_container_environment_sha256="f" * 64,
        prediction_code_files_sha256=code_files,
        prediction_code_sha256=position.sha256_json(code_files),
        records=rows,
    )
    predictions = tmp_path / "predictions.json"
    predictions.write_bytes(
        position.strict_dev._encoded_json(artifact.model_dump(mode="json"))
    )
    receipt = position.PositionACFPredictionReceipt(
        artifact_type="pams_position_acf_direct_dev_prediction_receipt",
        protocol="ucfrep_526",
        split="dev",
        method_key=position._METHOD_KEY,
        prediction_file="predictions.json",
        prediction_sha256=hashlib.sha256(predictions.read_bytes()).hexdigest(),
        prediction_bytes=predictions.stat().st_size,
        config_fingerprint=artifact.config_fingerprint,
        readout_config_file_sha256=artifact.readout_config_file_sha256,
        readout_config_fingerprint=artifact.readout_config_fingerprint,
        dev_identity_sha256=artifact.dev_identity_sha256,
        checkpoint_sha256=artifact.checkpoint_sha256,
        checkpoint_progress_sha256=artifact.checkpoint_progress_sha256,
        checkpoint_completion_receipt_sha256=(
            artifact.checkpoint_completion_receipt_sha256
        ),
        training_pose_cache_set_sha256=artifact.training_pose_cache_set_sha256,
        dev_pose_cache_set_sha256=artifact.dev_pose_cache_set_sha256,
        dev_pose_cache_snapshot_sha256=artifact.dev_pose_cache_snapshot_sha256,
        prediction_source_git_sha=artifact.prediction_source_git_sha,
        prediction_container_image_id=artifact.prediction_container_image_id,
        prediction_container_environment_sha256=(
            artifact.prediction_container_environment_sha256
        ),
        prediction_code_sha256=artifact.prediction_code_sha256,
        command=("pams", "position-acf", "dev-predict"),
        hardware={},
    )
    receipt_path = tmp_path / "prediction.receipt.json"
    receipt_path.write_bytes(
        position.strict_dev._encoded_json(receipt.model_dump(mode="json"))
    )
    return predictions, receipt_path


def test_scorer_rejects_non_v8_artifact_before_target_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions, receipt = _prediction_bundle(tmp_path)
    monkeypatch.setattr(
        position,
        "load_dev_target_manifest",
        lambda _path: pytest.fail("target loader crossed exact-v8 firewall"),
    )
    with pytest.raises(ValueError, match="exact frozen v8 run"):
        position.score_dev_predictions(
            predictions_path=predictions,
            prediction_receipt_path=receipt,
            dev_targets_path=tmp_path / "must-not-open.targets.json",
            output_dir=tmp_path / "score",
            readout_config_path=READOUT_CONFIG,
            repository_root=REPOSITORY,
        )


def test_scorer_permanently_rejects_test105_before_target_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions, receipt = _prediction_bundle(tmp_path)
    artifact = position._load_prediction_artifact(predictions)
    monkeypatch.setattr(position, "_require_exact_v8_artifact_bindings", lambda _a: None)
    monkeypatch.setattr(position, "_validate_canonical_dev_identity", lambda _a: None)
    monkeypatch.setattr(
        position,
        "clean_git_revision",
        lambda _root: artifact.prediction_source_git_sha,
    )
    monkeypatch.setattr(
        position,
        "load_dev_target_manifest",
        lambda _path: pytest.fail("test105 target loader must never run"),
    )
    with pytest.raises(ValueError, match="rejects test targets"):
        position.score_dev_predictions(
            predictions_path=predictions,
            prediction_receipt_path=receipt,
            dev_targets_path=tmp_path / "ucfrep_test_105_targets.json",
            output_dir=tmp_path / "score",
            readout_config_path=READOUT_CONFIG,
            repository_root=REPOSITORY,
        )


def test_old_teacher_reconstruction_is_unchanged(
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


def test_serialized_prediction_rows_are_target_free(tmp_path: Path) -> None:
    predictions, _ = _prediction_bundle(tmp_path)
    payload = json.loads(predictions.read_text(encoding="utf-8"))
    assert payload["split"] == "dev"
    assert payload["test_evaluation_authorized"] is False
    assert len(payload["records"]) == 84
    assert all("target" not in row and "action" not in row for row in payload["records"])
