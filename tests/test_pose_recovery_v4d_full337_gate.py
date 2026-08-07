from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "scripts" / "server"
sys.path.insert(0, str(SERVER))

from audit_pose_recovery_v4d_full337 import (  # noqa: E402
    _anchor_supported,
    _base_fill_boundary_jumps,
    _decision_receipt,
    _longest_run,
    _longest_same_origin_run,
    _mask_transition_rate,
    _period_evidence,
)
from pose_recovery_v4d_full337_contract import (  # noqa: E402
    SAME39_AUDIT_SHA256,
    SAME39_CACHE_SET_SHA256,
    SAME39_FAILURE_RECEIPT_SHA256,
    SAME39_LEDGER_SHA256,
    SAME39_SELECTION_SHA256,
    validate_full337_gate,
)

GATE_SHA256 = "304af8669703a73bf3d953e19848633986b682d478ee3a5f8be78114c26f11b8"


def test_full337_gate_is_precommitted_and_retains_original_coverage_thresholds() -> None:
    path = ROOT / "configs/gates/pams_pose_recovery_v4d_full337.yaml"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == GATE_SHA256
    gate, digest = validate_full337_gate(path, expected_sha256=GATE_SHA256)
    assert digest == GATE_SHA256
    assert gate["preregistration"] == {
        "frozen_before_full337_outputs": True,
        "output_independent_thresholds": True,
        "same39_authorizes_extraction_only": True,
        "full337_pass_required_for_baseline_training": True,
    }
    assert gate["measurement_protocol"] == {
        "periodicity_coordinates": "xy_only",
        "periodicity_signal": "precomputed_pose_velocity_vectors",
        "eligible_velocity_pair_policy": (
            "adjacent_both_valid_and_same_extractor_origin_only"
        ),
        "excluded_velocity_pairs": ["invalid_gap", "base_to_fill", "fill_to_base"],
        "minimum_anchor_frames": 2,
        "minimum_anchor_fraction_of_final_valid_frames": 0.02,
    }
    assert gate["coverage_thresholds"] == {
        "v4_zero_video_maximum": 8,
        "recovered_reference_zero_video_minimum": 8,
        "reference_usable_to_v4_zero_video_maximum": 0,
        "source_coverage_mean_minimum": 0.80,
        "source_coverage_p10_minimum": 0.25,
        "source_coverage_p25_minimum": 0.60,
        "source_coverage_median_minimum": 0.97,
        "observed_at_most_8_video_maximum": 4,
        "longest_run_fraction_p10_minimum": 0.15,
        "longest_run_fraction_median_minimum": 0.80,
    }
    bindings = gate["bindings"]
    assert bindings["same39_failure_receipt_sha256"] == SAME39_FAILURE_RECEIPT_SHA256
    assert bindings["same39_audit_sha256"] == SAME39_AUDIT_SHA256
    assert bindings["same39_ledger_sha256"] == SAME39_LEDGER_SHA256
    assert bindings["same39_selection_sha256"] == SAME39_SELECTION_SHA256
    assert bindings["same39_candidate_cache_set_sha256"] == SAME39_CACHE_SET_SHA256


def test_v2_authorization_is_extraction_only_and_ignores_projection_for_decision() -> None:
    signer = (
        ROOT
        / "scripts/server/authorize_pose_recovery_v4d_full337_extraction_v2.py"
    ).read_text(encoding="utf-8")
    contract = (
        ROOT / "scripts/server/pose_recovery_v4d_full337_contract.py"
    ).read_text(encoding="utf-8")
    assert '"authorization_scope": "full337_pose_extraction_only"' in signer
    assert '"baseline_training_authorized": False' in signer
    assert '"training_runner_must_reject": True' in signer
    assert '"projection_consulted_for_decision": False' in signer
    failed_pilot_validator = contract.split(
        "def validate_same39_failed_pilot", maxsplit=1
    )[1].split("def validate_same39_authorization_v2", maxsplit=1)[0]
    assert 'audit.get("passed") is False' in failed_pilot_validator
    assert 'audit.get("full337_pose_extraction_authorized") is False' in failed_pilot_validator
    assert 'audit.get("projected_full337_gate_passed")' not in failed_pilot_validator
    assert "projected_criteria" not in signer
    assert "longest_run_fraction_p10" not in signer


def test_runner_enforces_exact_39_reuse_plus_298_fresh_and_no_training_authority() -> None:
    runner = (ROOT / "scripts/server/run_pose_recovery_v4d_full337.py").read_text(
        encoding="utf-8"
    )
    assert 'require(len(same39_ids) == 39' in runner
    assert 'require(len(remaining) == 298' in runner
    assert '"selected": 337' in runner
    assert '"baseline_training_authorized": False' in runner
    assert '"bytewise_reused_frozen_same39"' in runner
    assert "stable_file_bytes(source_path)" in runner
    assert "stable_file_bytes(base_path)" in runner


def test_shells_keep_authorizer_cpu_only_and_full_extraction_on_canonical_gpu1() -> None:
    authorization_shell = (
        ROOT
        / "scripts/server/run_pams_pose_recovery_v4d_full337_extraction_authorization_v2.sh"
    ).read_text(encoding="utf-8")
    full_shell = (ROOT / "scripts/server/run_pams_pose_recovery_v4d_full337.sh").read_text(
        encoding="utf-8"
    )
    assert "--gpus" not in authorization_shell
    assert "authorizer must not have GPU" in authorization_shell
    assert "full337_pose_extraction_only" in authorization_shell
    assert "--gpus device=1" in full_shell
    assert "gpu1.lock" in full_shell
    assert "--network none" in authorization_shell
    assert "--network none" in full_shell
    assert '--workdir /workspace "$IMAGE_ID"' in authorization_shell
    assert full_shell.count('--workdir /workspace "$IMAGE_ID"') == 2
    assert "actual container image-ID mismatch" in authorization_shell
    assert "post-run image-ID mismatch" in authorization_shell
    assert "extract actual image-ID mismatch" in full_shell
    assert "extract post-run image-ID mismatch" in full_shell
    assert "gate actual image-ID mismatch" in full_shell
    assert "gate post-run image-ID mismatch" in full_shell
    assert (
        authorization_shell.index('mkdir -- "$RUN_ROOT"')
        < authorization_shell.index("trap finalize EXIT")
        < authorization_shell.index('mkdir -- "$SOURCE_EXPORT"')
    )
    assert (
        full_shell.index('mkdir -- "$RUN_ROOT"')
        < full_shell.index("trap finalize EXIT")
        < full_shell.index('mkdir -- "$SOURCE_EXPORT"')
    )
    assert '! -e "$AUDIT_DIR/run.receipt.json"' in full_shell
    assert '&& ! -e "$TRAIN_DENIAL"' not in full_shell
    assert (
        "pams_pose_recovery_v4d_full337_extraction_authorization_v2_run_receipt"
        in authorization_shell
    )
    assert "--mount \"type=bind,src=${VIDEOS}" in full_shell
    gate_create = full_shell.split("docker create --name \"$GATE_NAME\"", maxsplit=1)[1]
    assert "src=${VIDEOS}" not in gate_create
    assert "src=${MODEL_ASSET}" not in gate_create


def test_track_helpers_are_deterministic_and_boundary_specific() -> None:
    mask = np.asarray([False, True, True, False, True, True, True], dtype=np.bool_)
    assert _longest_run(mask) == 3
    assert _mask_transition_rate(mask) == 3 / 6
    final_xyz = np.zeros((4, 33, 3), dtype=np.float32)
    final_xyz[1, :, 0] = 0.2
    final_xyz[2, :, 0] = 0.5
    final_xyz[3, :, 0] = 0.7
    final_mask = np.ones(4, dtype=np.bool_)
    base_mask = np.asarray([True, True, False, False], dtype=np.bool_)
    jumps = _base_fill_boundary_jumps(final_xyz, final_mask, base_mask)
    assert len(jumps) == 1
    assert np.isclose(jumps[0], np.sqrt((0.3**2) / 2.0))
    assert _longest_same_origin_run(final_mask, base_mask) == 2


def test_period_evidence_is_xy_only_and_excludes_extractor_boundaries() -> None:
    frames = 40
    pattern = np.asarray([0.0, 1.0, 0.0, -1.0], dtype=np.float32)
    xyz = np.zeros((frames, 33, 3), dtype=np.float32)
    xyz[:, :, 0] = np.resize(pattern, frames)[:, None]
    final_mask = np.ones(frames, dtype=np.bool_)
    base_mask = np.arange(frames) < 20
    baseline = _period_evidence(xyz, final_mask, base_mask)

    changed = xyz.copy()
    changed[20:, :, :2] += 256.0
    changed[:, :, 2] = np.arange(frames, dtype=np.float32)[:, None] * 1000.0
    observed = _period_evidence(changed, final_mask, base_mask)

    assert np.allclose(observed[:2], baseline[:2])
    assert observed[2] == baseline[2] == 38
    assert observed[3] == baseline[3] == 38 / 39


def test_anchor_support_requires_two_frames_and_two_percent() -> None:
    assert not _anchor_supported(0, 100)
    assert not _anchor_supported(1, 20)
    assert not _anchor_supported(2, 101)
    assert _anchor_supported(2, 100)
    assert _anchor_supported(3, 101)
    assert not _anchor_supported(5, 0)


def _audit_fixture(*, passed: bool) -> dict[str, Any]:
    criterion = {"value": 1, "relation": "at_least", "threshold": 1, "passed": passed}
    return {
        "source_revision": "a" * 40,
        "container_image_id": "sha256:" + "b" * 64,
        "passed": passed,
        "bindings": {"full337_cache_set_sha256": "c" * 64},
        "coverage_criteria": {"coverage": criterion},
        "track_quality_criteria": {"track": criterion},
        "periodicity_criteria": {"period": criterion},
        "anchor_risk_criteria": {"anchor": criterion},
        "integrity_criteria": {"integrity": criterion},
    }


def test_full337_gate_preserves_both_authorization_and_denial_receipts() -> None:
    passed = _decision_receipt(_audit_fixture(passed=True), audit_sha256="d" * 64)
    denied = _decision_receipt(_audit_fixture(passed=False), audit_sha256="e" * 64)
    assert passed["artifact_type"].endswith("training_authorization")
    assert passed["baseline_training_authorized"] is True
    assert passed["failed_criteria"] == []
    assert denied["artifact_type"].endswith("training_denial")
    assert denied["baseline_training_authorized"] is False
    assert len(denied["failed_criteria"]) == 5
    assert denied["full337_pose_extraction_only_receipt_is_insufficient"] is True
