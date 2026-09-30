from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.server import run_pams_v16_noabs_joint_authorization as runner


def _inputs(runner_key: str) -> dict[str, object]:
    code_files = {runner_key: "9" * 64, "dependency": "8" * 64}
    video_ids = [f"train-{index:03d}" for index in range(337)]
    return {
        "encoder_checkpoint_sha256": runner._EXPECTED_ENCODER_CHECKPOINT_SHA256,
        "encoder_progress_sha256": runner._EXPECTED_ENCODER_PROGRESS_SHA256,
        "config_sha256": runner._EXPECTED_CONFIG_SHA256,
        "config_fingerprint": runner._EXPECTED_CONFIG_FINGERPRINT,
        "pose_fingerprint": runner._EXPECTED_POSE_FINGERPRINT,
        "source_export_receipt_sha256": (runner._EXPECTED_SOURCE_EXPORT_RECEIPT_SHA256),
        "train337_pose_cache_set_sha256": (runner._EXPECTED_TRAIN337_POSE_CACHE_SET_SHA256),
        "checkpoint_algorithm_source_git_sha": (
            runner._EXPECTED_CHECKPOINT_ALGORITHM_SOURCE_GIT_SHA
        ),
        "gate_code_source_git_sha": (runner._EXPECTED_EVIDENCE_GATE_CODE_SOURCE_GIT_SHA),
        "train337_video_total": 337,
        "train337_video_ids_sha256": "7" * 64,
        "encoder_provenance": {
            "protocol": "ucfrep_526",
            "source_git_sha": runner._EXPECTED_CHECKPOINT_ALGORITHM_SOURCE_GIT_SHA,
            "pose_cache_set_sha256": (runner._EXPECTED_TRAIN337_POSE_CACHE_SET_SHA256),
            "pose_fingerprint": runner._EXPECTED_POSE_FINGERPRINT,
            "training_video_ids": video_ids,
        },
        f"{runner_key}_sha256": code_files[runner_key],
        "code_files_sha256": code_files,
        "code_files_sha256_commitment": runner._sha256_json(code_files),
        "read_only_post_run_identity_verified": True,
    }


def _false_authorizations() -> dict[str, bool]:
    return {
        "sshead_training_authorized": False,
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }


def _teacher_artifact() -> dict[str, object]:
    convergence_criteria = {
        "w8_absolute_difference": True,
        "s8_absolute_difference": False,
        "r8_relative_nll_advantage_difference": True,
        "aperm_relative_nll_advantage_difference": True,
        "gperm_current_gradient_cosine_difference": True,
        "gperm_center_gradient_cosine_difference": True,
        "classification_unchanged": True,
    }
    classification = {
        "alias_confirmed": False,
        "current_teacher_state": "identifiable",
        "mandatory_center_signal_viable": True,
    }
    return {
        "schema_version": 1,
        "artifact_type": "pams_v15_teacher_identifiability_probe",
        "status": "v16_short_continuation_rejected",
        "protocol": "ucfrep_526",
        "seed": 2026,
        "inputs": _inputs("teacher_identifiability_probe_runner"),
        "anchor_budget_measurements": {
            "32": {"classification": deepcopy(classification)},
            "64": {"classification": deepcopy(classification)},
        },
        "convergence": {
            "criteria": convergence_criteria,
            "classification_unchanged": True,
            "pass": False,
        },
        "operational_gate": {
            "criteria": {"train337": True, "read_only": True},
            "pass": True,
        },
        "decision": {
            "v16_short_continuation_authorized": False,
            **_false_authorizations(),
        },
        "read_only_verification": {
            "all_file_inputs_unchanged": True,
            "train337_pose_cache_set_sha256_unchanged": True,
            "checkpoint_algorithm_source_git_sha_unchanged": True,
            "model_state_sha256_before": "6" * 64,
            "model_state_sha256_after": "6" * 64,
            "model_or_optimizer_state_updated": False,
            "training_steps_executed": 0,
            "pose_cache_write_operations": 0,
        },
    }


def _peoff_artifact() -> dict[str, object]:
    criteria = {
        name: {"pass": True, "value": 0.0, "threshold": 1.0} for name in runner._PEOFF_CRITERIA
    }
    variants = {
        name: {
            "projected_period_confidence_exact": True,
            "projected_tensor_sha256_exact": True,
            "projected_tensor_drift_within_threshold": True,
            "projected_tensor_max_abs_drift": 0.0,
            "projected_tensor_max_abs_drift_threshold": 1e-7,
        }
        for name in ("original", "time_scale_0.5", "time_scale_0.75")
    }
    return {
        "schema_version": 1,
        "artifact_type": ("pams_v15_terminal_encoder_in_memory_peoff_dual_path_probe"),
        "status": "peoff_gate_pass",
        "protocol": "ucfrep_526",
        "seed": 2026,
        "inputs": _inputs("v15_peoff_probe_runner"),
        "projected_path_identity_audit": {
            "all_variants_projected_period_confidence_exact": True,
            "all_variants_projected_tensor_sha256_exact": True,
            "overall_projected_tensor_max_abs_drift": 0.0,
            "overall_projected_tensor_max_abs_drift_threshold": 1e-7,
            "variants": variants,
        },
        "in_memory_mutation_audit": {
            "mode_restored": True,
            "model_tensor_state_unchanged": True,
            "model_state_sha256_before": "5" * 64,
            "model_state_sha256_after_baseline": "5" * 64,
            "model_state_sha256_after_restore": "5" * 64,
        },
        "gate": {
            "criteria": criteria,
            "peoff_gate_pass": True,
            "all_ten_diagnostic_criteria_pass": True,
            "standalone_noabs_training_authorized": False,
            **_false_authorizations(),
        },
        "read_only_verification": {
            "all_file_inputs_unchanged": True,
            "checkpoint_file_sha256_before_after_unchanged": True,
            "train337_pose_cache_set_sha256_unchanged": True,
            "model_tensor_state_sha256_before_after_unchanged": True,
            "position_encoding_mode_restored": True,
            "model_or_optimizer_tensor_state_updated": False,
            "training_steps_executed": 0,
            "optimizer_created": False,
            "pose_cache_write_operations": 0,
        },
    }


def _receipt(
    artifact_path: Path,
    artifact: dict[str, object],
    *,
    receipt_type: str,
    extra: dict[str, object],
) -> dict[str, object]:
    encoded = artifact_path.read_bytes()
    inputs = artifact["inputs"]
    assert isinstance(inputs, dict)
    return {
        "schema_version": 1,
        "artifact_type": receipt_type,
        "artifact_locator": artifact_path.name,
        "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
        "artifact_bytes": len(encoded),
        "artifact_status": artifact["status"],
        "encoder_checkpoint_sha256": inputs["encoder_checkpoint_sha256"],
        "encoder_progress_sha256": inputs["encoder_progress_sha256"],
        "config_sha256": inputs["config_sha256"],
        "source_export_receipt_sha256": inputs["source_export_receipt_sha256"],
        "train337_pose_cache_set_sha256": inputs["train337_pose_cache_set_sha256"],
        "checkpoint_algorithm_source_git_sha": inputs["checkpoint_algorithm_source_git_sha"],
        "gate_code_source_git_sha": inputs["gate_code_source_git_sha"],
        "code_files_sha256_commitment": inputs["code_files_sha256_commitment"],
        **extra,
    }


def _write_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, Path]:
    teacher = _teacher_artifact()
    teacher_path = tmp_path / "teacher.json"
    teacher_path.write_bytes(runner._encoded_json(teacher))
    teacher_receipt = _receipt(
        teacher_path,
        teacher,
        receipt_type="pams_v15_teacher_identifiability_probe_receipt",
        extra={
            "v16_short_continuation_authorized": False,
            **_false_authorizations(),
        },
    )
    teacher_receipt_path = tmp_path / "teacher.json.receipt.json"
    teacher_receipt_path.write_bytes(runner._encoded_json(teacher_receipt))

    peoff = _peoff_artifact()
    peoff_path = tmp_path / "peoff.json"
    peoff_path.write_bytes(runner._encoded_json(peoff))
    peoff_receipt = _receipt(
        peoff_path,
        peoff,
        receipt_type=("pams_v15_terminal_encoder_in_memory_peoff_dual_path_probe_receipt"),
        extra={
            "peoff_gate_pass": True,
            "standalone_noabs_training_authorized": False,
        },
    )
    peoff_receipt_path = tmp_path / "peoff.json.receipt.json"
    peoff_receipt_path.write_bytes(runner._encoded_json(peoff_receipt))

    monkeypatch.setattr(
        runner,
        "_EXPECTED_TEACHER_IDENT_ARTIFACT_SHA256",
        hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runner,
        "_EXPECTED_TEACHER_IDENT_RECEIPT_SHA256",
        hashlib.sha256(teacher_receipt_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runner,
        "_EXPECTED_PEOFF_ARTIFACT_SHA256",
        hashlib.sha256(peoff_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runner,
        "_EXPECTED_PEOFF_RECEIPT_SHA256",
        hashlib.sha256(peoff_receipt_path.read_bytes()).hexdigest(),
    )
    return teacher_path, teacher_receipt_path, peoff_path, peoff_receipt_path


def test_cli_accepts_only_two_evidence_pairs_and_output() -> None:
    parsed = runner._parse_arguments(
        [
            "--teacher-ident-artifact",
            "teacher.json",
            "--teacher-ident-receipt",
            "teacher.receipt.json",
            "--peoff-artifact",
            "peoff.json",
            "--peoff-receipt",
            "peoff.receipt.json",
            "--output",
            "joint.json",
        ]
    )
    assert set(vars(parsed)) == {
        "teacher_ident_artifact",
        "teacher_ident_receipt",
        "peoff_artifact",
        "peoff_receipt",
        "output",
    }
    assert set(inspect.signature(runner.run_joint_authorization).parameters) == {
        "teacher_ident_artifact_path",
        "teacher_ident_receipt_path",
        "peoff_artifact_path",
        "peoff_receipt_path",
    }
    source = inspect.getsource(runner._parse_arguments)
    for forbidden in (
        "--data",
        "--video",
        "--pose",
        "--config",
        "--checkpoint",
        "--model",
        "--label",
        "--target",
        "--count",
        "--dev",
        "--test",
        "--train",
    ):
        assert f'"{forbidden}"' not in source


def test_joint_truth_table_ignores_only_s8_for_noabs_and_keeps_scope_closed() -> None:
    teacher = _teacher_artifact()
    peoff = _peoff_artifact()
    decision = runner._joint_decision(
        teacher,
        teacher["decision"],  # type: ignore[arg-type]
        peoff,
        peoff["gate"],  # type: ignore[arg-type]
    )

    assert all(decision["criteria"].values())
    assert decision["ident_s8_convergence_failure_is_noabs_blocker"] is False
    assert decision["fresh_seed2026_train337_noabspe_encoder_through_epoch11_authorized"] is True
    assert decision["authorized_scope"]["maximum_completed_epoch"] == 11
    assert decision["authorized_scope"]["initialization"] == "fresh"
    assert decision["mandatory_center_training_authorized"] is False
    assert decision["epoch12_or_later_training_authorized"] is False
    assert decision["sshead_training_authorized"] is False
    assert decision["dev84_prediction_authorized"] is False
    assert decision["test105_evaluation_authorized"] is False

    changed = deepcopy(teacher)
    changed["anchor_budget_measurements"]["64"]["classification"][  # type: ignore[index]
        "current_teacher_state"
    ] = "teacher_blind"
    rejected = runner._joint_decision(
        changed,
        changed["decision"],  # type: ignore[arg-type]
        peoff,
        peoff["gate"],  # type: ignore[arg-type]
    )
    assert rejected["fresh_seed2026_train337_noabspe_encoder_through_epoch11_authorized"] is False


def test_joint_runner_validates_evidence_and_writes_hash_bound_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _write_evidence(tmp_path, monkeypatch)
    monkeypatch.setenv("PAMS_GATE_CODE_SOURCE_REVISION", "d" * 40)
    before = [path.read_bytes() for path in paths]

    payload = runner.run_joint_authorization(*paths)

    assert payload["status"] == "noabs_epoch11_authorized"
    assert payload["inputs"]["read_only_post_run_identity_verified"] is True
    assert payload["inputs"]["joint_authorization_runner_sha256"]
    assert [path.read_bytes() for path in paths] == before
    output = tmp_path / "joint.json"
    receipt_path, digest = runner._write_artifact_and_receipt(output, payload)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert digest == hashlib.sha256(output.read_bytes()).hexdigest()
    assert receipt["artifact_sha256"] == digest
    assert receipt["fresh_seed2026_train337_noabspe_encoder_through_epoch11_authorized"] is True
    assert receipt["maximum_authorized_completed_epoch"] == 11
    assert receipt["epoch12_or_later_training_authorized"] is False
    assert receipt["sshead_training_authorized"] is False
    assert receipt["dev84_prediction_authorized"] is False
    assert receipt["test105_evaluation_authorized"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        runner._write_artifact_and_receipt(output, payload)


def test_exact_artifact_hash_and_receipt_binding_are_mandatory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher_path, receipt_path, _, _ = _write_evidence(tmp_path, monkeypatch)
    teacher_spec = runner._evidence_specs()[0]
    runner._load_bound_evidence(teacher_path, receipt_path, teacher_spec)

    monkeypatch.setattr(
        runner,
        "_EXPECTED_TEACHER_IDENT_ARTIFACT_SHA256",
        "0" * 64,
    )
    with pytest.raises(ValueError, match="evidence identity mismatch"):
        runner._load_bound_evidence(
            teacher_path,
            receipt_path,
            runner._evidence_specs()[0],
        )

    monkeypatch.setattr(
        runner,
        "_EXPECTED_TEACHER_IDENT_ARTIFACT_SHA256",
        hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["artifact_bytes"] += 1
    receipt_path.write_bytes(runner._encoded_json(receipt))
    with pytest.raises(ValueError, match="evidence identity mismatch"):
        runner._load_bound_evidence(
            teacher_path,
            receipt_path,
            runner._evidence_specs()[0],
        )
    monkeypatch.setattr(
        runner,
        "_EXPECTED_TEACHER_IDENT_RECEIPT_SHA256",
        hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    )
    with pytest.raises(ValueError, match="receipt binding mismatch"):
        runner._load_bound_evidence(
            teacher_path,
            receipt_path,
            runner._evidence_specs()[0],
        )


def test_semantically_equivalent_receipt_with_extra_field_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    teacher_path, receipt_path, _, _ = _write_evidence(tmp_path, monkeypatch)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["untrusted_extra_field"] = "does not alter the semantic binding"
    receipt_path.write_bytes(runner._encoded_json(receipt))

    with pytest.raises(ValueError, match="evidence identity mismatch"):
        runner._load_bound_evidence(
            teacher_path,
            receipt_path,
            runner._evidence_specs()[0],
        )


def test_prior_true_authorization_flag_is_a_conflict() -> None:
    teacher = _teacher_artifact()
    peoff = _peoff_artifact()
    teacher["decision"]["sshead_training_authorized"] = True  # type: ignore[index]

    decision = runner._joint_decision(
        teacher,
        teacher["decision"],  # type: ignore[arg-type]
        peoff,
        peoff["gate"],  # type: ignore[arg-type]
    )

    assert decision["criteria"]["prior_evidence_has_no_true_authorization_flags"] is False
    assert "decision.sshead_training_authorized" in decision["prior_true_authorization_flag_paths"]
    assert decision["fresh_seed2026_train337_noabspe_encoder_through_epoch11_authorized"] is False
