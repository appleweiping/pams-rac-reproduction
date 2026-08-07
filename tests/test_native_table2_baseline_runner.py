from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pams.config import load_config
from scripts.server import validate_pams_native_baseline_inputs as validator
from scripts.server.validate_pams_native_baseline_inputs import (
    NativeBaselineInputError,
    _load_json,
    _parse_arguments,
    _validate_pose_recovery_authorization,
    _validate_proxy_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/pams_native_table2_baseline_proxy_v1.yaml"
CONFIG_B = ROOT / "configs/experiments/pams_native_table2_baseline_proxy_b_w16_s2.yaml"
CONFIG_C = ROOT / "configs/experiments/pams_native_table2_baseline_proxy_c_w24_s4.yaml"
V4A_CONFIG = ROOT / "configs/experiments/pams_pose_recovery_v4a.yaml"
RUNNER = ROOT / "scripts/server/run_pams_native_table2_baseline_train337_v1.sh"
VALIDATOR = ROOT / "scripts/server/validate_pams_native_baseline_inputs.py"


def test_proxy_config_is_the_preregistered_inferred_native_baseline() -> None:
    placeholder = load_config(CONFIG)
    v4a = load_config(V4A_CONFIG)
    authorized_pose = placeholder.pose.model_copy(
        update={
            "preprocessing_revision": (
                "official-segment-heavy-missing-retry-full-timeline-v4b"
            )
        }
    )
    candidate = placeholder.model_copy(update={"pose": authorized_pose})
    pose_recovery = v4a.model_copy(update={"pose": authorized_pose})

    _validate_proxy_config(candidate, pose_recovery)

    assert candidate.data == pose_recovery.data
    assert candidate.pose == pose_recovery.pose
    assert candidate.model.position_encoding_mode == "sinusoidal"
    assert candidate.period.training_mode == "fixed_period_inferred"
    assert candidate.period.fixed_period_frames == 16
    assert (candidate.period.minimum, candidate.period.maximum) == (4, 4096)
    assert candidate.period.maximum_mode == "half_timeline"
    assert candidate.period.direct_fft_timebase == "dense_resampled"
    assert candidate.readout.action_curve_source == "embedding_recurrence_carrier"
    assert candidate.loss.scales == (1.0,)
    assert candidate.loss.temperature == pytest.approx(0.1)
    assert candidate.loss.anchor_stride == 4
    assert candidate.loss.use_cross_cluster_negatives is False
    assert candidate.loss.exclude_other_scale_positives_from_denominator is False
    assert candidate.training.epochs == 150
    assert candidate.training.effective_batch_size == 32
    assert candidate.training.skeleton_augmentation.enabled is True


def test_proxy_config_rejects_the_failed_v4a_placeholder() -> None:
    candidate = load_config(CONFIG)
    v4a = load_config(V4A_CONFIG)

    with pytest.raises(NativeBaselineInputError, match="formally rejected"):
        _validate_proxy_config(candidate, v4a)


def test_frozen_candidate_matrix_changes_only_window_or_stride() -> None:
    configs = [load_config(path) for path in (CONFIG, CONFIG_B, CONFIG_C)]
    assert [
        (item.period.fixed_period_frames, item.loss.anchor_stride) for item in configs
    ] == [(16, 4), (16, 2), (24, 4)]
    payloads = [item.model_dump(mode="json") for item in configs]
    for payload in payloads:
        payload["period"]["fixed_period_frames"] = 0
        payload["loss"]["anchor_stride"] = 0
    assert payloads[0] == payloads[1] == payloads[2]


def test_proxy_config_discloses_every_inference_and_followup() -> None:
    source = CONFIG.read_text(encoding="utf-8")
    normalized = " ".join(source.replace("#", " ").split())

    for disclosure in (
        "independently inferred executable proxy",
        "not the authors' baseline",
        "not author-disclosed",
        "period.maximum=4096 is only a non-binding safety ceiling",
        "maximum_mode=half_timeline applies the per-sample native-duration bound",
        "never eligible to claim their Table-2 number",
        "v4a failed its paired gate",
    ):
        assert disclosure in normalized
    assert "performance" not in source.lower()
    assert "nmae" not in source.lower()
    assert "obo" not in source.lower()


def _passing_pose_authorization_payloads() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    authorization = {
        "schema_version": 1,
        "artifact_type": "pams_native_pose_recovery_train337_authorization",
        "version": "v4c",
        "authorized": True,
        "authorized_consumer": "pams_native_table2_baseline_proxy_train337_encoder",
        "classification": "trusted_native_pose_recovery_train337",
        "protocol": "ucfrep_526",
        "split": "train",
        "record_total": 337,
        "bindings": {
            "source_revision": "f" * 40,
            "container_image_id": "sha256:" + "1" * 64,
            "config_file_sha256": "d" * 64,
            "config_fingerprint": "9" * 64,
            "pose_fingerprint": "a" * 64,
            "paired_gate_sha256": "e" * 64,
            "run_receipt_sha256": "8" * 64,
            "ledger_sha256": "c" * 64,
            "pose_cache_set_sha256": "b" * 64,
        },
        "scope": {
            "pose_timeline": "native",
            "pose_cache": "train337_only",
            "source_videos_mounted": False,
            "dev84_mounted": False,
            "test105_mounted": False,
            "targets_mounted": False,
            "network_mode": "none",
            "source_export_read_only": True,
        },
    }
    paired_rows = [
        {
            "video_id_sha256": hashlib.sha256(f"video-{index}".encode()).hexdigest(),
            "source_frames": 100,
            "reference_cached_valid_frames": 90,
            "pass0_valid_frames": 80,
            "recovered_valid_frames": 10,
            "v4_cached_valid_frames": 90,
            "final_valid_frames": 90,
            "observed_span_frames": 95,
            "source_coverage": 0.9,
            "longest_run_fraction": 0.8,
            "pass0_preserved": True,
        }
        for index in range(337)
    ]
    metrics = {
        key: 337 if key == "record_total" else 0.0
        for key in validator._POSE_GATE_METRIC_KEYS
    }
    criteria = {
        key: {"value": 0.0, "relation": "at_most", "threshold": 0.0, "passed": True}
        for key in validator._POSE_GATE_CRITERION_KEYS
    }
    gate = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4c_train337_paired_audit",
        "protocol": "ucfrep_526",
        "split": "train",
        "passed": True,
        "bindings": {
            "gate_sha256": "1" * 64,
            "sidecar_sha256": validator._EXPECTED_SIDECAR_SHA256["train"],
            "commitment_sha256": validator._EXPECTED_COMMITMENT_SHA256["train"],
            "identity_sha256": "2" * 64,
            "reference_ledger_sha256": "3" * 64,
            "reference_pose_fingerprint": "4" * 64,
            "reference_pose_cache_set_sha256": "5" * 64,
            "v4_ledger_sha256": "c" * 64,
            "v4_pose_fingerprint": "a" * 64,
            "v4_pose_cache_set_sha256": "b" * 64,
            "heavy_model_asset_sha256": "6" * 64,
        },
        "metrics": metrics,
        "criteria": criteria,
        "paired_rows": paired_rows,
        "paired_rows_sha256": validator._canonical_sha256(paired_rows),
        "mount_audit": {
            "train_identity_mounted": True,
            "train_pose_caches_mounted": True,
            "source_videos_mounted": False,
            "dev84_mounted": False,
            "test105_mounted": False,
            "targets_mounted": False,
        },
    }
    receipt = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4c_train337_run_receipt",
        "source_revision": "f" * 40,
        "container_image_id": "sha256:" + "1" * 64,
        "config_file_sha256": "d" * 64,
        "config_fingerprint": "9" * 64,
        "pose_fingerprint": "a" * 64,
        "heavy_model_asset_sha256": "6" * 64,
        "temporal_resampling": "none_native_timeline",
        "scope": "count-free-train337-pose-input-recovery-only",
        "paired_gate_exit_status": 0,
        "paired_gate_sha256": "e" * 64,
    }
    return authorization, gate, receipt


def _validate_synthetic_authorization(
    authorization: dict[str, object],
    gate: dict[str, object],
    receipt: dict[str, object],
    *,
    expected_version: str = "v4c",
) -> None:
    _validate_pose_recovery_authorization(
        authorization,
        gate,
        receipt,
        expected_version=expected_version,
        gate_sha256="e" * 64,
        run_receipt_sha256="8" * 64,
        config_sha256="d" * 64,
        config_fingerprint="9" * 64,
        pose_fingerprint="a" * 64,
        ledger_sha256="c" * 64,
        pose_cache_set_sha256="b" * 64,
    )


def test_pose_authorization_accepts_a_hash_bound_train337_native_artifact() -> None:
    authorization, gate, receipt = _passing_pose_authorization_payloads()
    _validate_synthetic_authorization(authorization, gate, receipt)


def test_upstream_json_rejects_nested_sensitive_keys_before_use(tmp_path: Path) -> None:
    path = tmp_path / "upstream.json"
    path.write_text(
        json.dumps({"safe": {"labels": ["must-never-materialize"]}}),
        encoding="utf-8",
    )
    with pytest.raises(NativeBaselineInputError, match="forbidden privileged field"):
        _load_json(path, role="fixture upstream artifact")


def _passing_v4c_ledger() -> tuple[dict[str, object], object, str]:
    pose_config = load_config(V4A_CONFIG)
    pose_fingerprint = pose_config.pose_fingerprint
    cache_set_sha256 = "b" * 64
    recovery = pose_config.pose.recovery
    assert recovery is not None
    entries = [
        {"video_id": f"video-{index}", "cache_sha256": "a" * 64, "bytes": 10}
        for index in range(337)
    ]
    audit = {key: 0 for key in validator._POSE_LEDGER_RECOVERY_AUDIT_KEYS}
    audit.update(
        {
            "schema_version": 1,
            "recovery_mode": "tasks-video-multipose4-fill-missing-v4c",
            "heavy_model_id": recovery.heavy_model_id,
            "heavy_model_asset_sha256": recovery.heavy_model_asset_sha256,
            "temporal_resampling": recovery.temporal_resampling,
            "pose_coordinate_interpolation": recovery.pose_coordinate_interpolation,
            "pass0_observations_preserved": True,
            "pass0_valid_mask_sha256": "1" * 64,
            "final_valid_mask_sha256": "2" * 64,
            "heavy_video_valid_mask_sha256": "3" * 64,
            "heavy_video_timestamp_sha256": "4" * 64,
        }
    )
    caches = []
    for index in range(337):
        row = {key: None for key in validator._POSE_LEDGER_CACHE_KEYS}
        row.update(
            {
                "video_id": f"video-{index}",
                "pose_fingerprint": pose_fingerprint,
                "skipped": False,
                "recovery_audit": dict(audit),
            }
        )
        caches.append(row)
    ledger = {
        "schema_version": 2,
        "artifact_type": "pams_pose_recovery_v4c_train337_ledger",
        "input_kind": "label_free_train337_hashed_full_set",
        "protocol": "ucfrep_526",
        "split": "train",
        "sidecar_sha256": validator._EXPECTED_SIDECAR_SHA256["train"],
        "commitment_file_sha256": validator._EXPECTED_COMMITMENT_SHA256["train"],
        "full_train_identity_sha256": "5" * 64,
        "identity_sha256": "5" * 64,
        "selection_sha256": "6" * 64,
        "pose_fingerprint": pose_fingerprint,
        "pose_recovery": {
            "schema_version": 1,
            "preprocessing_revision": pose_config.pose.preprocessing_revision,
            "recovery_mode": "tasks-video-multipose4-fill-missing-v4c",
            "heavy_model_id": recovery.heavy_model_id,
            "heavy_model_asset_sha256": recovery.heavy_model_asset_sha256,
            "temporal_resampling": recovery.temporal_resampling,
            "pose_coordinate_interpolation": recovery.pose_coordinate_interpolation,
        },
        "successful_cache_snapshot": {
            "schema_version": 1,
            "pose_fingerprint": pose_fingerprint,
            "fingerprint": cache_set_sha256,
            "entry_count": 337,
            "entries": entries,
        },
        "selected": 337,
        "completed": 337,
        "extracted": 337,
        "skipped": 0,
        "failed": 0,
        "recovery_version": "v4c",
        "caches": caches,
        "failures": [],
    }
    return ledger, pose_config, cache_set_sha256


def test_v4c_full_ledger_requires_the_exact_canonical_schema() -> None:
    ledger, pose_config, cache_set_sha256 = _passing_v4c_ledger()
    validator._validate_pose_recovery_ledger(
        ledger,
        expected_version="v4c",
        pose_recovery=pose_config,
        pose_cache_set_sha256=cache_set_sha256,
    )
    ledger["caches"][0]["recovery_audit"]["unexpected_metric_result"] = 1
    with pytest.raises(NativeBaselineInputError, match="recovery audit 0 schema mismatch"):
        validator._validate_pose_recovery_ledger(
            ledger,
            expected_version="v4c",
            pose_recovery=pose_config,
            pose_cache_set_sha256=cache_set_sha256,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda authorization, gate, receipt: gate.update(passed=False),
            "did not pass",
        ),
        (
            lambda authorization, gate, receipt: gate["mount_audit"].update(
                dev84_mounted=True
            ),
            "dev84_mounted",
        ),
        (
            lambda authorization, gate, receipt: authorization["bindings"].update(
                pose_cache_set_sha256="f" * 64
            ),
            "binding mismatch: pose_cache_set_sha256",
        ),
        (
            lambda authorization, gate, receipt: receipt.update(
                paired_gate_sha256="f" * 64
            ),
            "does not bind",
        ),
        (
            lambda authorization, gate, receipt: authorization["scope"].update(
                source_export_read_only=False
            ),
            "source_export_read_only",
        ),
        (
            lambda authorization, gate, receipt: authorization.update(
                authorization_sha256="0" * 64
            ),
            "self-reported SHA",
        ),
        (
            lambda authorization, gate, receipt: authorization["scope"].update(
                unexpected_sensitive_summary="forbidden"
            ),
            "scope schema mismatch",
        ),
    ],
)
def test_pose_authorization_rejects_untrusted_upstream(
    mutator: object,
    message: str,
) -> None:
    authorization, gate, receipt = _passing_pose_authorization_payloads()
    mutator(authorization, gate, receipt)  # type: ignore[operator]

    with pytest.raises(NativeBaselineInputError, match=message):
        _validate_synthetic_authorization(authorization, gate, receipt)


@pytest.mark.parametrize(
    "version",
    ("v4a", "official-segment-heavy-missing-retry-full-timeline-v4a"),
)
def test_pose_authorization_formally_rejects_v4a(version: str) -> None:
    authorization, gate, receipt = _passing_pose_authorization_payloads()
    authorization["version"] = version

    with pytest.raises(NativeBaselineInputError, match="formally rejected"):
        _validate_synthetic_authorization(
            authorization,
            gate,
            receipt,
            expected_version=version,
        )


def test_validator_cli_exposes_no_target_or_dev_test_pose_surface() -> None:
    parser = _parse_arguments
    namespace = parser(
        [
            "--candidate-config",
            "candidate.yaml",
            "--pose-recovery-version",
            "v4b",
            "--pose-recovery-config",
            "pose-recovery.yaml",
            "--train-sidecar",
            "train.json",
            "--train-commitment",
            "train.commit.json",
            "--dev-identity-sidecar",
            "dev.json",
            "--dev-identity-commitment",
            "dev.commit.json",
            "--test-identity-sidecar",
            "test.json",
            "--test-identity-commitment",
            "test.commit.json",
            "--pose-recovery-authorization",
            "authorization.json",
            "--expected-pose-recovery-authorization-sha256",
            "c" * 64,
            "--pose-recovery-paired-gate",
            "gate.json",
            "--expected-pose-recovery-paired-gate-sha256",
            "a" * 64,
            "--pose-recovery-run-receipt",
            "receipt.json",
            "--expected-pose-recovery-run-receipt-sha256",
            "b" * 64,
            "--pose-recovery-ledger",
            "ledger.json",
            "--train-pose-cache",
            "train-pose",
            "--output",
            "preflight.json",
        ]
    )
    destinations = set(vars(namespace))

    assert "train_pose_cache" in destinations
    assert "dev_identity_sidecar" in destinations
    assert "test_identity_sidecar" in destinations
    assert not any("target" in value for value in destinations)
    assert "dev_pose_cache" not in destinations
    assert "test_pose_cache" not in destinations


def test_runner_has_epoch11_gate_then_immutable_resume_boundary() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    launch = source.index("CURRENT_STAGE='launch-authorization-create'")
    epoch11 = source.index("CURRENT_STAGE='encoder-epoch11-create'")
    gate = source.index("CURRENT_STAGE='epoch11-gate-create'")
    rejection = source.index('if [[ "$GATE_CONTAINER_EXIT" -eq 3 ]]')
    final_creation = source.index('mkdir -- "$FINAL_ENCODER_STAGE"')
    final = source.index("CURRENT_STAGE='encoder-final-create'")

    assert source.count("python -m pams train encoder") == 2
    assert source.count(
        'python "$LAUNCH_AUTHORIZATION_RELATIVE"'
    ) == 1
    assert source.count('--gpus "device=${GPU_DEVICE}"') == 3
    assert launch < epoch11 < gate < rejection < final_creation < final
    assert "--epochs 11" in source
    assert "--epochs 150" in source
    assert "--resume-checkpoint /pams/resume/encoder.pt" in source
    assert "--resume-progress /pams/resume/encoder.jsonl" in source
    assert "--encoder-completion-receipt /pams/epoch11/completion.receipt.json" in source
    assert "--expected-encoder-completion-receipt-sha256" in source
    assert "--candidate-launch-authorization /pams/launch/authorization.json" in source
    assert (
        "--candidate-launch-receipt "
        "/pams/launch/authorization.json.receipt.json"
    ) in source
    assert "\n    --include-dev" not in source
    assert "dst=/pams/dev-pose" not in source
    assert "dst=/pams/test-pose" not in source
    assert "dst=/pams/targets" not in source


def test_runner_mounts_only_train_pose_and_identity_only_protocol_sidecars() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    protocol_block = source.split("protocol_args=(", maxsplit=1)[1].split(
        ")\nlaunch_prior_mount_args=(", maxsplit=1
    )[0]

    assert protocol_block.count("dst=/pams/protocol/") == 6
    assert "train.inputs.json" in protocol_block
    assert "dev.inputs.json" in protocol_block
    assert "test-identity.inputs.json" in protocol_block
    assert ".targets" not in protocol_block
    assert "pose-cache" not in protocol_block
    assert source.count("dst=/pams/pose-cache,readonly") == 4
    assert "src=${POSE_RECOVERY_CACHE},dst=/pams/pose-cache,readonly" in source
    assert "dst=/pams/dev-pose" not in source
    assert "dst=/pams/test-pose" not in source


def test_runner_enforces_clean_source_image_gate_and_immutable_receipts() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    for required in (
        "PAMS_POSE_RECOVERY_VERSION",
        "PAMS_POSE_RECOVERY_RUN_ROOT",
        "PAMS_POSE_RECOVERY_CONFIG_PATH",
        "PAMS_POSE_RECOVERY_AUTHORIZATION_SHA256",
        "pams_native_pose_recovery_train337_authorization",
        "source_export_read_only",
        "pose_recovery_authorization_sha256",
        "status --porcelain=v1 --untracked-files=all",
        "org.opencontainers.image.revision",
        "org.opencontainers.image.pams.environment-sha256",
        "validate_pams_native_baseline_inputs.py",
        "run_pams_native_epoch11_train_gate.py",
        "pams_native_epoch11_train_gate_v1.yaml",
        "prepare_pams_native_candidate_launch_authorization.py",
        "pams_native_terminal_readout_gate_v1.yaml",
        "PAMS_PRIOR_A_REJECTION_ARTIFACT",
        "PAMS_PRIOR_B_REJECTION_ARTIFACT",
        "PAMS_CANDIDATE_ID",
        "PYTHONOPTIMIZE=",
        "input-preflight.json",
        "with Path(os.environ[\"RUN_RECEIPT_PATH\"]).open(",
        '"x", encoding="utf-8", newline="\\n"',
        '"paper_table2_value_claim_eligible": False',
        '"dev_authorized": False',
        '"test_authorized": False',
        "chmod -R a-w -- \"$RUN_ROOT\"",
    ):
        assert required in source
    assert "PAMS_V4A" not in source
    assert "/pams/v4a" not in source
    assert source.index("run_created_container \"$PREFLIGHT_NAME\"") < source.index(
        "CURRENT_STAGE='launch-authorization-create'"
    )
    assert source.index("CURRENT_STAGE='launch-authorization-create'") < source.index(
        "exec 9>\"$LOCK_PATH\""
    )
    assert source.index("exec 9>\"$LOCK_PATH\"") < source.index(
        "CURRENT_STAGE='encoder-epoch11-create'"
    )
    assert "assert " not in source


def test_validator_declares_identity_only_firewall_in_both_code_and_artifact() -> None:
    source = VALIDATOR.read_text(encoding="utf-8")

    for required in (
        '"dev84_pose_cache": False',
        '"test105_pose_cache": False',
        '"targets": False',
        '"undisclosed_period_head_training": False',
        '"sshead_training": False',
        '"dev_prediction_authorized": False',
        '"test_prediction_authorized": False',
        "official {split} identity sidecar SHA mismatch",
        "pose cache directory must contain 337 files",
        "pams_native_pose_recovery_train337_authorization",
        "source_export_read_only",
        "formally rejected",
    ):
        assert required in source
