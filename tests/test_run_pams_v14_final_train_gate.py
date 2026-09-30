from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from pams.training import CheckpointProvenance
from scripts.server import run_pams_v14_final_train_gate as runner


def _provenance(
    *,
    upstream: str | None,
    source: str = "1" * 40,
) -> CheckpointProvenance:
    return CheckpointProvenance(
        protocol="ucfrep_526",
        dataset_fingerprint="2" * 64,
        training_video_ids=tuple(f"v{index:03d}" for index in range(337)),
        pose_fingerprint="3" * 64,
        pose_cache_set_sha256="4" * 64,
        source_git_sha=source,
        container_image_id=f"sha256:{'5' * 64}",
        container_environment_sha256="6" * 64,
        upstream_encoder_checkpoint_sha256=upstream,
    )


def _sample(
    video_id: str,
    values: np.ndarray,
    *,
    valid: np.ndarray | None = None,
) -> runner._HeadSample:
    mask = np.ones(values.shape, dtype=np.bool_) if valid is None else valid.astype(np.bool_)
    selected = values[mask]
    return runner._HeadSample(
        video_id=video_id,
        valid_frames=int(mask.sum()),
        period_frames=16.0,
        confidence=1.0,
        period_stream=tuple(float(value) for value in values),
        valid_mask=tuple(bool(value) for value in mask),
        period_stream_std=float(np.std(selected)),
    )


def _passing_inputs() -> dict[str, object]:
    return {
        "encoder_distribution": {
            "boundary_share": 0.24,
            "mode_share": 0.24,
        },
        "encoder_time_scale": {
            "candidate_comparison_total": 10,
            "eligible_comparison_total": 5,
            "relative_error": {"median": 0.15},
        },
        "sshead_distribution": {
            "boundary_share": 0.24,
            "mode_share": 0.24,
            "period_stream_std": {"median": 0.05},
        },
        "sshead_correlation": {
            "eligible_pair_fraction": 0.90,
            "absolute_correlation": {"median": 0.89},
        },
        "sshead_time_scale": {
            "candidate_comparison_total": 10,
            "eligible_comparison_total": 5,
            "relative_error": {"median": 0.15},
        },
        "sshead_history": {"zero_grad_steps_total": 0},
        "head_parameter_change": {"delta_l2": 0.1},
    }


def _artifact_payload() -> dict[str, object]:
    digest = "a" * 64
    return {
        "status": "passed",
        "inputs": {
            "encoder_checkpoint_sha256": digest,
            "encoder_progress_sha256": digest,
            "sshead_checkpoint_sha256": digest,
            "sshead_progress_sha256": digest,
            "config_sha256": digest,
            "train337_pose_cache_set_sha256": digest,
            "checkpoint_algorithm_source_git_sha": "b" * 40,
            "gate_code_source_git_sha": "c" * 40,
            "code_files_sha256_commitment": digest,
        },
        "hardware_sha256": digest,
        "runtime_sha256": digest,
    }


def test_cli_surface_is_train337_only_and_reuses_frozen_epoch11_helpers() -> None:
    parsed = runner._parse_arguments(
        [
            "--encoder-checkpoint",
            "encoder.pt",
            "--encoder-progress",
            "encoder.jsonl",
            "--sshead-checkpoint",
            "sshead.pt",
            "--sshead-progress",
            "sshead.jsonl",
            "--config",
            "v14.yaml",
            "--pose-cache-dir",
            "train337-pose",
            "--output",
            "final-gate.json",
        ]
    )

    assert set(vars(parsed)) == {
        "encoder_checkpoint",
        "encoder_progress",
        "sshead_checkpoint",
        "sshead_progress",
        "config",
        "pose_cache_dir",
        "output",
        "device",
        "batch_size",
    }
    assert set(inspect.signature(runner.run_final_gate).parameters) == {
        "encoder_checkpoint_path",
        "encoder_progress_path",
        "sshead_checkpoint_path",
        "sshead_progress_path",
        "config_path",
        "pose_cache_dir",
        "device",
        "batch_size",
    }
    parser_source = inspect.getsource(runner._parse_arguments)
    for forbidden in (
        "--manifest",
        "--targets",
        "--action",
        "--count",
        "--dev",
        "--test",
    ):
        assert f'"{forbidden}"' not in parser_source
        assert f"'{forbidden}'" not in parser_source
    assert runner._encode_training_sequences.__module__ == "scripts.server.run_pams_v14_predev_gate"
    assert runner._training_distribution.__module__ == "scripts.server.run_pams_v14_predev_gate"
    assert runner._time_scale_consistency.__module__ == "scripts.server.run_pams_v14_predev_gate"


def test_stage_binding_requires_same_formal_context_and_live_encoder_hash() -> None:
    encoder_sha256 = "7" * 64
    encoder = _provenance(upstream=None)
    sshead = _provenance(upstream=encoder_sha256)

    runner._validate_stage_bindings(
        encoder,
        sshead,
        encoder_checkpoint_sha256=encoder_sha256,
    )

    with pytest.raises(ValueError, match="supplied encoder bytes"):
        runner._validate_stage_bindings(
            encoder,
            sshead,
            encoder_checkpoint_sha256="8" * 64,
        )
    with pytest.raises(ValueError, match="provenance differ"):
        runner._validate_stage_bindings(
            encoder,
            _provenance(upstream=encoder_sha256, source="9" * 40),
            encoder_checkpoint_sha256=encoder_sha256,
        )


def test_runtime_binding_requires_exact_checkpoint_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provenance = _provenance(upstream=None)
    monkeypatch.setenv(
        "PAMS_CONTAINER_IMAGE_ID",
        provenance.container_image_id or "",
    )
    monkeypatch.setenv(
        "PAMS_CONTAINER_ENVIRONMENT_SHA256",
        provenance.container_environment_sha256 or "",
    )
    monkeypatch.setenv(
        "PAMS_CONTAINER_SOURCE_REVISION",
        provenance.source_git_sha,
    )

    assert runner._validate_runtime_binding(
        provenance,
        source_git_sha=provenance.source_git_sha,
    ) == {
        "image_id": provenance.container_image_id,
        "environment_sha256": provenance.container_environment_sha256,
        "source_revision": provenance.source_git_sha,
    }

    monkeypatch.setenv("PAMS_CONTAINER_SOURCE_REVISION", "a" * 40)
    with pytest.raises(RuntimeError, match="runtime differs"):
        runner._validate_runtime_binding(
            provenance,
            source_git_sha=provenance.source_git_sha,
        )


def test_external_gate_code_requires_its_own_public_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PAMS_GATE_CODE_SOURCE_REVISION", raising=False)
    with pytest.raises(RuntimeError, match="PAMS_GATE_CODE_SOURCE_REVISION"):
        runner._gate_code_source_revision()

    later_gate_revision = "f" * 40
    monkeypatch.setenv("PAMS_GATE_CODE_SOURCE_REVISION", later_gate_revision)
    assert runner._gate_code_source_revision() == later_gate_revision
    assert later_gate_revision != _provenance(upstream=None).source_git_sha

    monkeypatch.setenv("PAMS_GATE_CODE_SOURCE_REVISION", "F" * 40)
    with pytest.raises(RuntimeError, match="lowercase Git SHA"):
        runner._gate_code_source_revision()


def test_absolute_centered_correlation_detects_sign_flipped_shared_template() -> None:
    time = np.arange(64, dtype=np.float64)
    template = np.sin(2.0 * np.pi * time / 16.0)
    shared = (
        _sample("a", template),
        _sample("b", 2.0 * template + 3.0),
        _sample("c", -template + 9.0),
    )
    result = runner._centered_cross_video_correlation(shared)

    assert result["eligible_pair_fraction"] == 1.0
    assert result["absolute_correlation"]["median"] == pytest.approx(1.0)
    assert result["signed_correlation"]["minimum"] == pytest.approx(-1.0)

    diverse = tuple(
        _sample(
            str(order),
            np.sin(2.0 * np.pi * (order + 1) * time / time.size),
        )
        for order in range(1, 6)
    )
    diverse_result = runner._centered_cross_video_correlation(diverse)
    assert diverse_result["eligible_pair_fraction"] == 1.0
    assert diverse_result["absolute_correlation"]["median"] < 0.01


def test_correlation_coverage_rejects_flat_or_insufficient_streams() -> None:
    flat = np.ones(16, dtype=np.float64)
    short_mask = np.asarray([True] * 7 + [False] * 9)
    result = runner._centered_cross_video_correlation(
        (
            _sample("flat-a", flat),
            _sample("flat-b", flat),
            _sample("short", np.arange(16, dtype=np.float64), valid=short_mask),
        )
    )

    assert result["candidate_pair_total"] == 3
    assert result["eligible_pair_total"] == 0
    assert result["eligible_pair_fraction"] == 0.0
    assert result["absolute_correlation"]["median"] is None


def test_all_thirteen_frozen_target_free_criteria_are_required() -> None:
    passing = _passing_inputs()
    decision = runner._gate_decision(**passing)

    assert decision["overall_pass"] is True
    assert len(decision["criteria"]) == 13
    assert decision["dev84_prediction_authorized"] is True
    assert decision["dev84_scoring_authorized"] is False
    assert decision["test105_evaluation_authorized"] is False

    failures: tuple[tuple[str, tuple[str, ...], object], ...] = (
        ("encoder_boundary_share", ("encoder_distribution", "boundary_share"), 0.25),
        ("encoder_mode_share", ("encoder_distribution", "mode_share"), 0.25),
        (
            "encoder_time_scale_eligible_fraction",
            ("encoder_time_scale", "eligible_comparison_total"),
            4,
        ),
        (
            "encoder_time_scale_median_relative_error",
            ("encoder_time_scale", "relative_error", "median"),
            0.151,
        ),
        (
            "sshead_stream_std_median",
            ("sshead_distribution", "period_stream_std", "median"),
            0.049,
        ),
        ("sshead_boundary_share", ("sshead_distribution", "boundary_share"), 0.25),
        ("sshead_mode_share", ("sshead_distribution", "mode_share"), 0.25),
        (
            "sshead_centered_cross_video_pair_coverage",
            ("sshead_correlation", "eligible_pair_fraction"),
            0.899,
        ),
        (
            "sshead_centered_cross_video_median_absolute_correlation",
            ("sshead_correlation", "absolute_correlation", "median"),
            0.90,
        ),
        (
            "sshead_time_scale_eligible_fraction",
            ("sshead_time_scale", "eligible_comparison_total"),
            4,
        ),
        (
            "sshead_time_scale_median_relative_error",
            ("sshead_time_scale", "relative_error", "median"),
            0.151,
        ),
        (
            "sshead_zero_grad_steps_total",
            ("sshead_history", "zero_grad_steps_total"),
            1,
        ),
        (
            "sshead_head_parameter_delta_l2",
            ("head_parameter_change", "delta_l2"),
            0.0,
        ),
    )
    for criterion, path, value in failures:
        failed = deepcopy(passing)
        destination = failed
        for field in path[:-1]:
            destination = destination[field]  # type: ignore[assignment,index]
        destination[path[-1]] = value  # type: ignore[index]
        failed_decision = runner._gate_decision(**failed)
        assert failed_decision["overall_pass"] is False
        assert failed_decision["criteria"][criterion]["pass"] is False


def test_head_parameter_delta_distinguishes_untouched_and_trained_heads() -> None:
    torch.manual_seed(7)
    initial_head = nn.Linear(4, 1)
    identical_head = nn.Linear(4, 1)
    identical_head.load_state_dict(initial_head.state_dict())
    changed_head = nn.Linear(4, 1)
    changed_head.load_state_dict(initial_head.state_dict())
    with torch.no_grad():
        changed_head.weight.add_(0.25)

    initial = SimpleNamespace(period_head=initial_head)
    identical = SimpleNamespace(period_head=identical_head)
    changed = SimpleNamespace(period_head=changed_head)
    assert runner._head_parameter_change(initial, identical)["delta_l2"] == 0.0
    change = runner._head_parameter_change(initial, changed)
    assert change["delta_l2"] > 0.0
    assert change["changed_tensor_total"] == 1


def test_artifact_and_receipt_are_exclusive_and_hash_bound(
    tmp_path: Path,
) -> None:
    output = tmp_path / "final-gate.json"
    payload = _artifact_payload()

    receipt_path, digest = runner._write_artifact_and_receipt(output, payload)

    artifact_bytes = output.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert digest == hashlib.sha256(artifact_bytes).hexdigest()
    assert receipt["artifact_sha256"] == digest
    assert receipt["artifact_bytes"] == len(artifact_bytes)
    assert receipt["encoder_progress_sha256"] == "a" * 64
    assert receipt["checkpoint_algorithm_source_git_sha"] == "b" * 40
    assert receipt["gate_code_source_git_sha"] == "c" * 40
    with pytest.raises(FileExistsError, match="overwrite"):
        runner._write_artifact_and_receipt(output, payload)
