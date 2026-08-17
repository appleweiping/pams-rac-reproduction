from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from pams.reproducibility import sha256_json

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/server/run_pams_position_acf_seed2026_strict.sh"
READOUT = ROOT / "configs/readouts/projected_position_acf_direct_v1.yaml"

READOUT_FILE_SHA256 = (
    "9c4db6210ecd82d0aa5e2a2437b4e8132eb993bff4afe808ad0db8772f9d7923"
)
READOUT_SEMANTIC_SHA256 = (
    "861e43963d45e4277d8daf93de7d556ebf63f1ab531bf4ded7077d6d20fd3f91"
)


def _runner() -> str:
    return RUNNER.read_text(encoding="utf-8")


def _block(source: str, start: str, stop: str) -> str:
    begin = source.index(start)
    end = source.index(stop, begin)
    return source[begin:end]


def _bash() -> str | None:
    if os.name != "nt":
        return shutil.which("bash")
    candidates = (
        Path(r"C:\Program Files\Git\bin\bash.exe"),
        Path(r"C:\Program Files\Git\usr\bin\bash.exe"),
    )
    return next((str(path) for path in candidates if path.is_file()), None)


def test_readout_config_bytes_semantics_and_harmonic_gate_are_frozen() -> None:
    payload_bytes = READOUT.read_bytes()
    payload = yaml.safe_load(payload_bytes)

    assert hashlib.sha256(payload_bytes).hexdigest() == READOUT_FILE_SHA256
    assert sha256_json(payload) == READOUT_SEMANTIC_SHA256
    assert payload["key"] == "pams-projected-position-acf-direct-v1"
    assert payload["eligible_for_paper_table"] is False
    assert payload["test105_evaluation_authorized"] is False
    assert payload["readout"]["temporal_transform"] == "none"
    assert payload["readout"]["minimum_period_frames"] == 4
    assert payload["readout"]["maximum_period_frames"] == 128

    synthetic = payload["predev_gates"]["synthetic"]
    assert synthetic == {
        "harmonic_orders": [2, 3, 4, 5, 6, 7],
        "frames": 256,
        "fundamental_period_frames": 64,
        "fundamental_amplitude": 1.0,
        "harmonic_amplitude": 0.75,
        "required_records": 6,
        "minimum_fundamental_selection_fraction": 1.0,
        "require_finite_outputs": True,
        "require_positive_confidence": True,
    }


def test_readout_config_freezes_dev_stop_reference_and_mount_policy() -> None:
    payload = yaml.safe_load(READOUT.read_text(encoding="utf-8"))
    gate = payload["dev_stop_gate"]
    assert gate["reference_evaluation_sha256"] == (
        "b6add8af9c229cd467be6bba282f267d0d70651a1a39b46ba1443016689add97"
    )
    assert gate["reference_nmae"] == 0.5290583435903633
    assert gate["reference_obo"] == 0.3333333333333333
    assert gate["require_strict_nmae_improvement"] is True
    assert gate["require_strict_obo_improvement"] is True
    assert gate["test105_evaluation_authorized"] is False

    policy = payload["mount_policy"]
    assert policy["network"] == "none"
    assert "dev84_pose_cache" in policy["dev_predict_inputs"]
    assert (
        "train337_pose_cache_for_checkpoint_provenance_validation"
        in policy["dev_predict_inputs"]
    )
    assert "readout_config" in policy["dev_score_inputs"]
    assert "dev84_targets" in policy["dev_score_inputs"]
    assert policy["prohibited_everywhere"] == [
        "test105_media",
        "test105_pose",
        "test105_labels",
    ]


def test_runner_has_valid_bash_syntax_when_a_real_bash_is_available() -> None:
    bash = _bash()
    if bash is None:
        pytest.skip("requires POSIX or Git Bash")
    result = subprocess.run(
        [bash, "-n", str(RUNNER)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_runner_expands_default_v8_root_at_runtime() -> None:
    bash = _bash()
    if bash is None:
        pytest.skip("requires POSIX or Git Bash")
    environment = os.environ.copy()
    environment.update(
        {
            "PAMS_ROOT": "/definitely-missing-pams-root",
            "PAMS_SOURCE_CHECKOUT": "/definitely-missing-pams-source",
            "PAMS_SOURCE_REVISION": "0" * 40,
            "PAMS_ATTEMPT_ID": "runtime-expansion-probe",
        }
    )
    result = subprocess.run(
        [bash, str(RUNNER)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=environment,
    )
    assert result.returncode != 0
    assert "bad substitution" not in result.stderr.lower()
    assert "错误的替换" not in result.stderr


def test_runner_requires_clean_committed_source_and_exact_v8_assets() -> None:
    launcher = _runner()
    assert 'git -C "$SOURCE_CHECKOUT" rev-parse HEAD' in launcher
    assert "status --porcelain=v1 --untracked-files=all" in launcher
    assert 'git -C "$SOURCE_CHECKOUT" archive' in launcher
    assert "git clone" not in launcher
    assert "docker cp" not in launcher
    assert "/pams/encoder/logs/encoder.jsonl" in launcher
    assert "/pams/encoder/encoder.jsonl" not in launcher
    assert (
        "/pams/encoder/manifests/"
        "20260730T025521Z-e6e6a2f657aa.completed.json"
        in launcher
    )
    assert "/pams/encoder/encoder.completed.json" not in launcher
    assert READOUT_FILE_SHA256 in launcher
    assert READOUT_SEMANTIC_SHA256 in launcher
    for digest in (
        "6817fe468d76c3fa2182dd831695d6fe3d6da208a2b7f8dfa90cffa6872e8053",
        "aadc8f9bf067b3489efcd80db43c42cf09dba3477c2976bbc16571ff31684622",
        "f4815e1961ba883c23ef479ddcdef44aea5cad8a50d54e17b8e6a4cccc613f3d",
        "eb4072a195608757cd2542855b33fb000c987f96c94b83a1e448ab78f97e9374",
        "f32d718ae922120778f535a6a4f467edba79cafeba90373b3a6a55bf11be6ee2",
        "681b0390ff252df9cf5b1e7c41bfcb6ecb8a0c5b4ef341087cacddc1bf5a3de1",
    ):
        assert digest in launcher


def test_predev_gates_use_position_not_velocity_and_run_in_order() -> None:
    launcher = _runner()
    assert "estimate_period_from_projected_position(" in launcher
    assert "projected_position_vector_acf_diagnostics(" in launcher
    assert "estimate_period_from_projected_pose(" not in launcher
    assert "generate_count_sweep" not in launcher
    assert 'gate["harmonic_orders"]' in launcher
    assert "tuple(range(2, 8))" in launcher
    assert "fundamental_period != 64.0" in launcher
    assert "fundamental_amplitude != 1.0" in launcher
    assert "harmonic_amplitude != 0.75" in launcher
    assert "diagnostic.selected_bin is None" in launcher
    assert "if period != float(diagnostic.selected_period)" not in launcher
    assert "if confidence != float(diagnostic.confidence)" not in launcher
    assert "rel_tol=1e-6" in launcher
    assert "abs_tol=1e-7" in launcher
    assert "expected_sizes = (337, 84, 105)" in launcher
    assert "assert len(sets) == len(expected_sizes)" in launcher
    assert "zip(sets, expected_sizes)" in launcher
    assert '"period_min": min(positive_periods) if positive_periods else None' in launcher
    assert '"period_max": max(positive_periods) if positive_periods else None' in launcher

    synthetic = launcher.index('CURRENT_STAGE="synthetic-k2-7-create"')
    train = launcher.index('CURRENT_STAGE="train337-distribution-create"')
    predict = launcher.index('CURRENT_STAGE="dev-predict-create"')
    target = launcher.index('CURRENT_STAGE="dev-score-target-boundary"')
    score = launcher.index('CURRENT_STAGE="dev-score-create"')
    decision = launcher.index('CURRENT_STAGE="dev-stop-decision"')
    assert synthetic < train < predict < target < score < decision


def test_predev_and_prediction_container_mounts_are_label_isolated() -> None:
    launcher = _runner()
    synthetic = _block(
        launcher,
        'CURRENT_STAGE="synthetic-k2-7-create"',
        'CURRENT_STAGE="train337-distribution-create"',
    )
    train = _block(
        launcher,
        'CURRENT_STAGE="train337-distribution-create"',
        'CURRENT_STAGE="dev-predict-create"',
    )
    predict = _block(
        launcher,
        'CURRENT_STAGE="dev-predict-create"',
        'CURRENT_STAGE="dev-score-target-boundary"',
    )

    for predev in (synthetic, train):
        assert "DEV_INPUT" not in predev
        assert "DEV_COMMIT" not in predev
        assert "DEV_TARGET" not in predev
        assert "TEST_ID_INPUT" not in predev
        assert "TEST_ID_COMMIT" not in predev
    assert "TRAIN_POSE_VIEW" in train
    assert "DEV_POSE_VIEW" not in train

    assert (
        "type=bind,src=${POSE_POOL},dst=/pams/pose-cache,readonly"
        in predict
    )
    assert "TRAIN_POSE_VIEW" not in predict
    assert "DEV_POSE_VIEW" not in predict
    assert '"${encoder_args[@]}"' in predict
    assert (
        "type=bind,src=${ENCODER_ROOT},dst=/pams/encoder,readonly"
        in launcher
    )
    assert "--checkpoint-progress /pams/encoder/logs/encoder.jsonl" in predict
    assert (
        "/pams/encoder/manifests/"
        "20260730T025521Z-e6e6a2f657aa.completed.json"
        in predict
    )
    assert "DEV_TARGET" not in predict
    assert "test-identity.inputs.json" in predict
    assert "test-identity.inputs.commitment.json" in predict
    assert "test.targets" not in predict
    assert "test.pose" not in predict
    assert "test.media" not in predict


def test_frozen_prediction_precedes_target_and_scorer_has_no_pose_or_checkpoint() -> None:
    launcher = _runner()
    prediction_frozen = launcher.index('chmod -R a-w "$PREDICT_STAGE"')
    target_boundary = launcher.index(
        'CURRENT_STAGE="dev-score-target-boundary"'
    )
    first_target_read = launcher.index(
        'assert_sha256 "$DEV_TARGET" "$DEV_TARGET_SHA256"'
    )
    assert prediction_frozen < target_boundary < first_target_read

    score = _block(
        launcher,
        'CURRENT_STAGE="dev-score-create"',
        'CURRENT_STAGE="dev-stop-decision"',
    )
    assert "PREDICT_STAGE}/predictions.json" in score
    assert "PREDICT_STAGE}/prediction.receipt.json" in score
    assert "DEV_TARGET" in score
    assert "READOUT_CONFIG_RELATIVE" in score
    assert "ENCODER_CHECKPOINT" not in score
    assert "ENCODER_PROGRESS" not in score
    assert "ENCODER_COMPLETION" not in score
    assert "POSE_VIEW" not in score
    assert "--readout-config /pams/input/readout-config.yaml" in score


def test_all_containers_are_networkless_and_test105_stays_identity_only() -> None:
    launcher = _runner()
    assert launcher.count('"${common_args[@]}"') == 4
    assert "--network none" in launcher
    assert 'host["NetworkMode"] == "none"' in launcher
    assert "python -m pams position-acf dev-predict" in launcher
    assert "python -m pams position-acf dev-score" in launcher
    assert "evaluate test" not in launcher
    assert "test105_evaluation_authorized\": true" not in launcher
    assert '"test105_media_mounted": false' in launcher
    assert '"test105_pose_mounted": false' in launcher
    assert '"test105_labels_mounted": false' in launcher


def test_dev_stop_gate_only_compares_public_local_frequency_metrics() -> None:
    launcher = _runner()
    decision = launcher[
        launcher.index('CURRENT_STAGE="dev-stop-decision"') :
    ]
    assert 'nmae < reference_nmae' in decision
    assert 'obo > reference_obo' in decision
    assert '"continue-dev-only-analysis" if passed' in decision
    assert '"stop-position-acf-line"' in decision
    assert '"authorizes_additional_seeds": False' in decision
    assert '"test105_evaluation_authorized": False' in decision
    assert "0.5290583435903633" in launcher
    assert "0.3333333333333333" in launcher


def test_attempts_are_non_overwriting_failure_preserving_and_receipted() -> None:
    launcher = _runner()
    assert 'mkdir -- "$RUN_ROOT" || fail "immutable run root exists' in launcher
    assert "trap on_exit EXIT" in launcher
    assert 'write_status "failed" "$CURRENT_STAGE" "$exit_code"' in launcher
    assert "freeze_failure_manifest" in launcher
    assert "failed-artifact-sha256.txt" in launcher
    assert "artifact-sha256.txt" in launcher
    assert 'test ! -e "$receipt_path"' in launcher
    assert 'with path.open("x"' in launcher
    assert "source-export.receipt.json" in launcher
    assert "gate.receipt.json" in launcher
    assert 'GATE_SCRIPT_PATH="$GATE_SCRIPT"' in launcher
    assert '\n  GATE_SCRIPT="$GATE_SCRIPT" \\\n' not in launcher
    assert "prediction.receipt.json" in launcher
    assert "evaluation.receipt.json" in launcher
    assert 'chmod -R a-w "$RUN_ROOT"' in launcher
