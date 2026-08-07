from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from pams.config import load_config

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/server/run_pams_pose_recovery_v4c_pilot.sh"
GATE = ROOT / "configs/gates/pams_pose_recovery_v4c_pilot.yaml"
CONFIG = ROOT / "configs/experiments/pams_pose_recovery_v4c.yaml"


def test_v4c_pilot_runner_is_server_only_and_mounts_no_privileged_split() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    mount_lines = [line.casefold() for line in source.splitlines() if "--mount " in line]

    assert source.count("--network none") >= 3
    assert source.count("--read-only") >= 3
    assert source.count("--cap-drop ALL") >= 2
    assert "--security-opt no-new-privileges:true" in source
    assert "curl " not in source
    assert "wget " not in source
    assert "--env HOME=" not in source
    assert "video-views/train337" in source
    assert "pose-recovery-v4a/83c877007392-20260807T113023Z" in source
    assert "long-tail-e644c76/v4a-long-tail-mechanism.json" in source
    assert "pose_landmarker_heavy.task" in source
    assert "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b" in source
    assert "python scripts/server/run_pose_recovery_v4b_pilot.py" in source
    assert "python scripts/server/audit_pose_recovery_v4b_pilot.py" in source
    for line in mount_lines:
        assert "/dev84" not in line
        assert "/test105" not in line
        assert "/targets" not in line
        assert "/annotations" not in line


def test_v4c_pilot_audit_container_has_no_source_videos() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    audit_create = source.split("docker create", maxsplit=2)[2]

    assert "src=${TRAIN_VIDEO_VIEW}" not in audit_create
    assert "--v4a-paired-gate /pams/v4a-paired-gate.json" in audit_create
    assert "--v4b-ledger /pams/v4c-ledger.json" in audit_create
    assert "--output /pams/audit/projected-gate.json" in audit_create


def test_v4c_pilot_gate_binds_config_and_unchanged_thresholds() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    config = load_config(CONFIG)
    bindings = gate["bindings"]
    thresholds = gate["thresholds"]
    escalation = gate["full_extraction_escalation_thresholds"]

    assert gate["expected_pilot_records"] == 39
    assert gate["artifact_type"] == ("pams_pose_recovery_v4c_train337_long_tail_pilot_gate")
    assert bindings["v4c_config_file_sha256"] == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert bindings["v4c_config_fingerprint"] == config.fingerprint
    assert bindings["v4c_pose_fingerprint"] == config.pose_fingerprint
    assert bindings["heavy_model_asset_sha256"] == config.pose.recovery.heavy_model_asset_sha256
    assert thresholds["source_coverage_mean_minimum"] == 0.80
    assert thresholds["source_coverage_p10_minimum"] == 0.25
    assert thresholds["longest_run_fraction_p10_minimum"] == 0.15
    assert thresholds["observed_at_most_8_video_maximum"] == 4
    assert escalation["pilot_invariant_failure_maximum"] == 0
    assert escalation["pass0_mask_mismatch_maximum"] == 0
    assert escalation["recovered_valid_frames_total_minimum"] == 390
    assert escalation["pass0_zero_video_reduction_minimum"] == 3
    assert escalation["pass0_at_most_8_video_reduction_minimum"] == 6
    assert escalation["source_coverage_mean_gain_over_pass0_minimum"] == 0.15
    assert escalation["source_coverage_mean_gain_over_v4a_minimum"] == 0.10
    assert escalation["longest_run_fraction_mean_gain_over_v4a_minimum"] == 0.05
    assert escalation["candidate_longest_run_fraction_p25_minimum"] == 0.10


def test_v4c_runner_distinguishes_strict_projection_from_cost_gate() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert '"strict_projected_gate_passed"' in source
    assert '"worth_full_extraction"' in source
    assert '"full_extraction_cost_authorized"' in source
    assert "neither the strict projection nor the frozen full-extraction cost gate passed" in source
