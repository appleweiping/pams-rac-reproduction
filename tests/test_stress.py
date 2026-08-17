from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import pams.stress_dev as stress_dev_module
from pams.stress import (
    apply_stress_condition,
    load_stress_protocol,
    pose_sequence_sha256,
)
from pams.stress_dev import (
    StressPredictionRow,
    _load_bound_replay_model,
    run_pams_dev_stress_prediction,
    score_pams_dev_stress_predictions,
)
from pams.training import CheckpointProvenance
from pams.types import PoseSequence


def _sequence(video_id: str = "video-a") -> PoseSequence:
    phase = np.linspace(0.0, 8.0 * np.pi, 256, dtype=np.float32)
    xyz = np.zeros((256, 33, 3), dtype=np.float32)
    joint_offsets = np.linspace(-0.2, 0.2, 33, dtype=np.float32)
    xyz[:, :, 0] = np.sin(phase)[:, None] + joint_offsets[None, :]
    xyz[:, :, 1] = np.cos(phase)[:, None] - joint_offsets[None, :]
    xyz[:, :, 2] = np.linspace(-1.0, 1.0, 256, dtype=np.float32)[:, None]
    return PoseSequence(video_id, 30.0, xyz, np.ones(256, dtype=np.bool_))


def _protocol():
    return load_stress_protocol(Path(__file__).parents[1] / "configs/stress.yaml")


def _condition(condition_id: str):
    return next(
        condition
        for condition in _protocol().conditions
        if condition.condition_id == condition_id
    )


def test_frozen_stress_config_expands_all_conditions() -> None:
    protocol = _protocol()
    assert protocol.seed == 2026
    assert [condition.condition_id for condition in protocol.conditions] == [
        "clean",
        "temporal_speed_pos0p5",
        "temporal_speed_pos2",
        "temporal_pause_middle",
        "pose_occlusion",
        "rotation_neg15deg",
        "rotation_pos15deg",
        "isotropic_scale_pos0p85",
        "isotropic_scale_pos1p15",
        "translation_neg0p1",
        "translation_pos0p1",
    ]


def test_every_perturbation_is_deterministic_and_keeps_256_frames() -> None:
    sequence = _sequence()
    for condition in _protocol().conditions:
        first, first_audit = apply_stress_condition(sequence, condition, seed=2026)
        second, second_audit = apply_stress_condition(sequence, condition, seed=2026)
        assert first.num_frames == 256
        assert first.video_id == sequence.video_id
        assert np.array_equal(first.xyz, second.xyz)
        assert np.array_equal(first.valid_mask, second.valid_mask)
        assert first_audit == second_audit
        assert first_audit.transformed_pose_sha256 == pose_sequence_sha256(first)


def test_temporal_speed_preserves_endpoints_and_uses_compensating_rate() -> None:
    sequence = _sequence()
    slow, slow_audit = apply_stress_condition(
        sequence,
        _condition("temporal_speed_pos0p5"),
        seed=2026,
    )
    fast, fast_audit = apply_stress_condition(
        sequence,
        _condition("temporal_speed_pos2"),
        seed=2026,
    )
    assert np.allclose(slow.xyz[[0, -1]], sequence.xyz[[0, -1]])
    assert np.allclose(fast.xyz[[0, -1]], sequence.xyz[[0, -1]])
    assert slow_audit.parameters["compensating_multiplier_second_half"] == 1.5
    assert fast_audit.parameters["compensating_multiplier_second_half"] == 0.0
    assert np.allclose(fast.xyz[128:], sequence.xyz[-1], atol=1e-5)


def test_middle_pause_freezes_configured_span_without_dropping_endpoints() -> None:
    sequence = _sequence()
    paused, audit = apply_stress_condition(
        sequence,
        _condition("temporal_pause_middle"),
        seed=2026,
    )
    assert np.allclose(paused.xyz[[0, -1]], sequence.xyz[[0, -1]])
    assert audit.parameters["frozen_output_frames"] in {51, 52}
    center = paused.xyz[103:153]
    assert np.max(np.abs(center - center[0])) < 1e-6


def test_occlusion_selects_exact_frames_and_joints_and_preserves_frame_mask() -> None:
    sequence = _sequence()
    transformed, audit = apply_stress_condition(
        sequence,
        _condition("pose_occlusion"),
        seed=2026,
    )
    assert audit.parameters["time_count"] == 51
    assert audit.parameters["joint_count"] == 10
    assert len(audit.parameters["joint_indices"]) == 10
    start = audit.parameters["time_start"]
    joints = audit.parameters["joint_indices"]
    assert np.all(transformed.xyz[start : start + 51, joints, :] == 0.0)
    assert np.array_equal(transformed.valid_mask, sequence.valid_mask)
    other, other_audit = apply_stress_condition(
        _sequence("video-b"),
        _condition("pose_occlusion"),
        seed=2026,
    )
    assert pose_sequence_sha256(other) != pose_sequence_sha256(transformed)
    assert other_audit.parameters != audit.parameters


def test_rotation_scale_and_translation_have_frozen_geometric_reference() -> None:
    sequence = _sequence()
    rotated, _ = apply_stress_condition(
        sequence,
        _condition("rotation_pos15deg"),
        seed=2026,
    )
    scaled, _ = apply_stress_condition(
        sequence,
        _condition("isotropic_scale_pos1p15"),
        seed=2026,
    )
    translated, audit = apply_stress_condition(
        sequence,
        _condition("translation_pos0p1"),
        seed=2026,
    )
    center = np.mean(sequence.xyz, axis=1, keepdims=True)
    original_radius = np.linalg.norm(sequence.xyz - center, axis=2)
    rotated_radius = np.linalg.norm(rotated.xyz - center, axis=2)
    scaled_radius = np.linalg.norm(scaled.xyz - center, axis=2)
    assert np.allclose(rotated_radius, original_radius, atol=1e-5)
    assert np.allclose(scaled_radius, original_radius * 1.15, atol=1e-5)
    delta = np.asarray(audit.parameters["translation_delta"], dtype=np.float32)
    assert np.allclose(translated.xyz - sequence.xyz, delta[None, None, :])


def test_prediction_runner_signature_cannot_accept_targets_or_test_inputs() -> None:
    parameters = set(inspect.signature(run_pams_dev_stress_prediction).parameters)
    assert not any("target" in name or "test" in name for name in parameters)
    assert {
        "source_clean_predictions_path",
        "source_clean_prediction_receipt_path",
        "checkpoint_path",
        "dev_inputs_path",
        "dev_commitment_path",
        "pose_cache_dir",
        "stress_config_path",
    } <= parameters


def test_stress_prediction_row_reconstructs_count_result() -> None:
    row = StressPredictionRow(
        video_id="v",
        video_sha256="0" * 64,
        count=4,
        period_frames=8.0,
        expert_counts=(3, 4, 5),
        selection_mode="multi",
        selected_expert=None,
        confidence=0.8,
        period_stream=(0.0, 1.0),
        perturbation_algorithm="identity",
        perturbation_parameters={},
        transformed_pose_sha256="1" * 64,
    )
    result = row.to_count_result()
    assert result.count == 4
    assert result.expert_counts == (3, 4, 5)


def test_formal_checkpoint_replay_supplies_verified_bound_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provenance = CheckpointProvenance(
        protocol="ucfrep_526",
        dataset_fingerprint="1" * 64,
        training_video_ids=tuple(f"train-{index:03d}" for index in range(337)),
        pose_fingerprint="2" * 64,
        pose_cache_set_sha256="3" * 64,
        source_git_sha="4" * 40,
        container_image_id="sha256:" + "5" * 64,
        container_environment_sha256="6" * 64,
        upstream_encoder_checkpoint_sha256="7" * 64,
    )
    checkpoint = tmp_path / "sshead.pt"
    torch.save(
        {
            "stage": "sshead",
            "provenance": provenance.to_dict(),
        },
        checkpoint,
    )
    clean_artifact = SimpleNamespace(
        protocol=provenance.protocol,
        pose_fingerprint=provenance.pose_fingerprint,
        training_pose_cache_set_sha256=provenance.pose_cache_set_sha256,
        checkpoint_source_git_sha=provenance.source_git_sha,
        checkpoint_container_image_id=provenance.container_image_id,
        checkpoint_container_environment_sha256=(
            provenance.container_environment_sha256
        ),
        upstream_encoder_checkpoint_sha256=(
            provenance.upstream_encoder_checkpoint_sha256
        ),
    )
    sentinel_model = object()
    observed: dict[str, object] = {}

    def fake_loader(
        path: Path,
        config: object,
        *,
        device: str | None,
        expected_stage: str,
        expected_provenance: CheckpointProvenance,
    ) -> object:
        observed.update(
            {
                "path": path,
                "config": config,
                "device": device,
                "stage": expected_stage,
                "provenance": expected_provenance,
            }
        )
        return sentinel_model

    monkeypatch.setattr(stress_dev_module, "load_model_checkpoint", fake_loader)
    config = object()
    model, provenance_sha256 = _load_bound_replay_model(
        checkpoint,
        config,
        clean_artifact,
        expected_stage="sshead",
        device="cuda:0",
    )

    assert model is sentinel_model
    assert observed == {
        "path": checkpoint,
        "config": config,
        "device": "cuda:0",
        "stage": "sshead",
        "provenance": provenance,
    }
    assert len(provenance_sha256) == 64


def test_scorer_does_not_touch_targets_before_prediction_validation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    predictions = tmp_path / "stress-predictions.json"
    receipt = tmp_path / "stress-prediction.receipt.json"
    stress_config = tmp_path / "stress.yaml"
    targets = tmp_path / "dev.targets.json"
    for path in (predictions, receipt, stress_config, targets):
        path.write_text("{}\n", encoding="utf-8")
    observed: list[Path] = []
    real_sha256 = stress_dev_module.sha256_file

    def observe_sha256(path: str | Path) -> str:
        source = Path(path)
        observed.append(source)
        return real_sha256(source)

    def reject_artifact(path: Path) -> stress_dev_module.StressPredictionArtifact:
        raise ValueError(f"invalid artifact: {path}")

    monkeypatch.setattr(stress_dev_module, "sha256_file", observe_sha256)
    monkeypatch.setattr(stress_dev_module, "_load_stress_artifact", reject_artifact)
    with pytest.raises(ValueError, match="invalid artifact"):
        score_pams_dev_stress_predictions(
            predictions_path=predictions,
            prediction_receipt_path=receipt,
            stress_config_path=stress_config,
            dev_targets_path=targets,
            output_dir=tmp_path / "score",
            repository_root=tmp_path,
        )
    assert targets not in observed
    assert stress_config not in observed
