from __future__ import annotations

from pathlib import Path

import pytest

from pams.config import load_config
from scripts.server.validate_pams_native_baseline_inputs import (
    NativeBaselineInputError,
    _parse_arguments,
    _validate_proxy_config,
    _validate_v4a_gate_payload,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/pams_native_table2_baseline_proxy_v1.yaml"
V4A_CONFIG = ROOT / "configs/experiments/pams_pose_recovery_v4a.yaml"
RUNNER = ROOT / "scripts/server/run_pams_native_table2_baseline_train337_v1.sh"
VALIDATOR = ROOT / "scripts/server/validate_pams_native_baseline_inputs.py"


def test_proxy_config_is_the_preregistered_inferred_native_baseline() -> None:
    candidate = load_config(CONFIG)
    v4a = load_config(V4A_CONFIG)

    _validate_proxy_config(candidate, v4a)

    assert candidate.data == v4a.data
    assert candidate.pose == v4a.pose
    assert candidate.model.position_encoding_mode == "sinusoidal"
    assert candidate.period.training_mode == "fixed_period_inferred"
    assert candidate.period.fixed_period_frames == 16
    assert (candidate.period.minimum, candidate.period.maximum) == (4, 512)
    assert candidate.loss.scales == (1.0,)
    assert candidate.loss.temperature == pytest.approx(0.1)
    assert candidate.loss.anchor_stride == 4
    assert candidate.loss.use_cross_cluster_negatives is False
    assert candidate.loss.exclude_other_scale_positives_from_denominator is False
    assert candidate.training.epochs == 150
    assert candidate.training.effective_batch_size == 32
    assert candidate.training.skeleton_augmentation.enabled is True


def test_proxy_config_discloses_every_inference_and_followup() -> None:
    source = CONFIG.read_text(encoding="utf-8")
    normalized = " ".join(source.replace("#", " ").split())

    for disclosure in (
        "independently inferred executable proxy",
        "not the authors' baseline",
        "not author-disclosed",
        "period.maximum=512 is a conservative temporary ceiling",
        "replace it with the preregistered half-timeline bound",
        "never eligible to claim their Table-2 number",
    ):
        assert disclosure in normalized
    assert "performance" not in source.lower()
    assert "nmae" not in source.lower()
    assert "obo" not in source.lower()


def _passing_gate_payloads() -> tuple[dict[str, object], dict[str, object]]:
    gate = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4a_train337_paired_audit",
        "protocol": "ucfrep_526",
        "split": "train",
        "passed": True,
        "metrics": {"record_total": 337},
        "mount_audit": {
            "train_identity_mounted": True,
            "train_pose_caches_mounted": True,
            "source_videos_mounted": False,
            "dev84_mounted": False,
            "test105_mounted": False,
            "targets_mounted": False,
        },
        "bindings": {
            "v4_pose_fingerprint": "a" * 64,
            "v4_pose_cache_set_sha256": "b" * 64,
            "v4_ledger_sha256": "c" * 64,
        },
    }
    receipt = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4a_train337_run_receipt",
        "config_file_sha256": "d" * 64,
        "pose_fingerprint": "a" * 64,
        "temporal_resampling": "none_native_timeline",
        "scope": "count-free-train337-pose-input-recovery-only",
        "paired_gate_exit_status": 0,
        "paired_gate_sha256": "e" * 64,
    }
    return gate, receipt


def _validate_synthetic_gate(
    gate: dict[str, object],
    receipt: dict[str, object],
) -> None:
    _validate_v4a_gate_payload(
        gate,
        receipt,
        gate_sha256="e" * 64,
        v4a_config_sha256="d" * 64,
        v4a_pose_fingerprint="a" * 64,
        v4a_ledger_sha256="c" * 64,
        pose_cache_set_sha256="b" * 64,
    )


def test_v4a_gate_contract_accepts_only_a_passed_train337_native_artifact() -> None:
    gate, receipt = _passing_gate_payloads()
    _validate_synthetic_gate(gate, receipt)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda gate, receipt: gate.update(passed=False), "did not pass"),
        (
            lambda gate, receipt: gate["mount_audit"].update(dev84_mounted=True),
            "dev84_mounted",
        ),
        (
            lambda gate, receipt: gate["bindings"].update(
                v4_pose_cache_set_sha256="f" * 64
            ),
            "pose-cache set mismatch",
        ),
        (
            lambda gate, receipt: receipt.update(paired_gate_sha256="f" * 64),
            "does not bind",
        ),
    ],
)
def test_v4a_gate_contract_rejects_untrusted_upstream(
    mutator: object,
    message: str,
) -> None:
    gate, receipt = _passing_gate_payloads()
    mutator(gate, receipt)  # type: ignore[operator]

    with pytest.raises(NativeBaselineInputError, match=message):
        _validate_synthetic_gate(gate, receipt)


def test_validator_cli_exposes_no_target_or_dev_test_pose_surface() -> None:
    parser = _parse_arguments
    namespace = parser(
        [
            "--candidate-config",
            "candidate.yaml",
            "--v4a-config",
            "v4a.yaml",
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
            "--v4a-paired-gate",
            "gate.json",
            "--expected-v4a-paired-gate-sha256",
            "a" * 64,
            "--v4a-run-receipt",
            "receipt.json",
            "--expected-v4a-run-receipt-sha256",
            "b" * 64,
            "--v4a-ledger",
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


def test_runner_has_one_encoder_only_gpu_boundary() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    encoder = source.split("CURRENT_STAGE='encoder-create'", maxsplit=1)[1].split(
        'verify_container "$ENCODER_NAME"', maxsplit=1
    )[0]

    assert source.count("python -m pams train encoder") == 1
    assert source.count('--gpus "device=${GPU_DEVICE}"') == 1
    assert "python -m pams train encoder" in encoder
    assert "--epochs 150" in encoder
    assert "--microbatch-size 32" in encoder
    assert "--label-free-inputs" in encoder
    assert "--include-dev" not in encoder
    assert "sshead" not in encoder.lower()
    assert "evaluate" not in encoder.lower()
    assert "predict" not in encoder.lower()
    assert "score" not in encoder.lower()
    assert ".targets" not in encoder
    assert "/dev-pose" not in encoder
    assert "/test-pose" not in encoder


def test_runner_mounts_only_train_pose_and_identity_only_protocol_sidecars() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    protocol_block = source.split("protocol_args=(", maxsplit=1)[1].split(
        ")\n\nCURRENT_STAGE='input-preflight-create'", maxsplit=1
    )[0]

    assert protocol_block.count("dst=/pams/protocol/") == 6
    assert "train.inputs.json" in protocol_block
    assert "dev.inputs.json" in protocol_block
    assert "test-identity.inputs.json" in protocol_block
    assert ".targets" not in protocol_block
    assert "pose-cache" not in protocol_block
    assert source.count("dst=/pams/pose-cache,readonly") == 2
    assert "src=${V4A_CACHE},dst=/pams/pose-cache,readonly" in source
    assert "dst=/pams/dev-pose" not in source
    assert "dst=/pams/test-pose" not in source


def test_runner_enforces_clean_source_image_gate_and_immutable_receipts() -> None:
    source = RUNNER.read_text(encoding="utf-8")

    for required in (
        "PAMS_V4A_PAIRED_GATE_SHA256",
        "PAMS_V4A_RUN_RECEIPT_SHA256",
        "PAMS_V4A_SOURCE_REVISION",
        "status --porcelain=v1 --untracked-files=all",
        "org.opencontainers.image.revision",
        "org.opencontainers.image.pams.environment-sha256",
        "validate_pams_native_baseline_inputs.py",
        "input-preflight.json",
        "with Path(os.environ[\"RUN_RECEIPT_PATH\"]).open(",
        '"x", encoding="utf-8", newline="\\n"',
        '"paper_table2_value_claim_eligible": False',
        '"dev_authorized": False',
        '"test_authorized": False',
        "chmod -R a-w -- \"$RUN_ROOT\"",
    ):
        assert required in source
    assert source.index("run_created_container \"$PREFLIGHT_NAME\"") < source.index(
        "exec 9>\"$LOCK_PATH\""
    )
    assert source.index("exec 9>\"$LOCK_PATH\"") < source.index(
        "CURRENT_STAGE='encoder-create'"
    )


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
    ):
        assert required in source
