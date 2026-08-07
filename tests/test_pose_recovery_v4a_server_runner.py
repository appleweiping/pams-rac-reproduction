from __future__ import annotations

from pathlib import Path

import yaml

from pams.config import load_config


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/server/run_pams_pose_recovery_v4a_train337.sh"
GATE = ROOT / "configs/gates/pams_pose_recovery_v4a_train337.yaml"
CONFIG = ROOT / "configs/experiments/pams_pose_recovery_v4a.yaml"


def test_runner_is_network_none_read_only_and_uses_preseeded_heavy_asset() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    assert source.count("--network none") >= 3
    assert source.count("--read-only") >= 3
    assert "--cap-drop ALL" in source
    assert "--security-opt no-new-privileges:true" in source
    assert "curl " not in source
    assert "wget " not in source
    assert (
        "/media/lenovo/data2/pams-rac/assets/mediapipe/"
        "pose_landmark_heavy.tflite"
    ) in source
    assert (
        "/opt/conda/lib/python3.11/site-packages/mediapipe/modules/"
        "pose_landmark/pose_landmark_heavy.tflite"
    ) in source
    assert "59e42d71bcd44cbdbabc419f0ff76686595fd265419566bd4009ef703ea8e1fe" in source
    assert '--mount "type=bind,src=${HEAVY_ASSET},dst=${HEAVY_CONTAINER_PATH},readonly"' in source
    assert '--heavy-model-asset "$HEAVY_CONTAINER_PATH"' in source


def test_runner_mounts_only_count_free_train337_inputs_for_extraction() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    mount_lines = [line.strip() for line in source.splitlines() if "--mount " in line]

    assert "${OFFICIAL_ROOT}/video-views/train337" in source
    assert "${OFFICIAL_ROOT}/protocol/train.inputs.json" in source
    assert "${OFFICIAL_ROOT}/protocol/train.inputs.commitment.json" in source
    assert source.count('src=${TRAIN_VIDEO_VIEW},dst=/pams/videos,readonly') == 1
    assert "--label-free-manifest" in source
    assert "--video-root /pams/videos" in source
    assert "--input-commitment /pams/protocol/inputs.commitment.json" in source
    for line in mount_lines:
        lowered = line.casefold()
        assert "/dev84" not in lowered
        assert "/test105" not in lowered
        assert "/targets" not in lowered
        assert "/annotations" not in lowered


def test_runner_audits_exact_reference_and_native_v4_cache_without_videos() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    audit_create = source.split("docker create", maxsplit=2)[2]

    assert "${OFFICIAL_ROOT}/stages/input-views/pose-train337" in source
    assert "${OFFICIAL_ROOT}/ledgers/train337.json" in source
    assert "--reference-cache-dir /pams/reference-cache" in audit_create
    assert "--reference-ledger /pams/reference-ledger.json" in audit_create
    assert "--v4-cache-dir /pams/v4-cache" in audit_create
    assert "--v4-ledger /pams/v4-ledger.json" in audit_create
    assert "src=${TRAIN_VIDEO_VIEW}" not in audit_create
    assert '"temporal_resampling": "none_native_timeline"' in source


def test_gate_binds_official_train337_config_pose_asset_and_reference() -> None:
    gate = yaml.safe_load(GATE.read_text(encoding="utf-8"))
    config = load_config(CONFIG)
    bindings = gate["bindings"]

    assert gate["artifact_type"] == "pams_pose_recovery_v4a_train337_gate"
    assert gate["split"] == "train"
    assert gate["expected_records"] == 337
    assert bindings["train_sidecar_sha256"] == (
        "f95df0050df21f05bbc9b42d0154470714dde279e04cb8b00d0212bf49057d16"
    )
    assert bindings["train_commitment_sha256"] == (
        "85d41d2f59872e0900e9058481efbdf6bce91407d4c26bcde0a6547776454e53"
    )
    assert bindings["train_identity_sha256"] == (
        "88367535e296213377c366ed69abbd1e808c38049d79bb948091688e4c5b7b3f"
    )
    assert bindings["reference_pose_cache_set_sha256"] == (
        "03259ec8d914af5613785ba2eee3003d276b638bda5f0cc993e425188a4259bd"
    )
    assert bindings["v4_config_fingerprint"] == config.fingerprint
    assert bindings["v4_pose_fingerprint"] == config.pose_fingerprint
    assert config.pose.recovery is not None
    assert config.pose.recovery.temporal_resampling == "none_native_timeline"
