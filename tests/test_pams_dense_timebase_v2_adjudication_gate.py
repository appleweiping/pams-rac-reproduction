from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.server import run_pams_dense_timebase_v2_adjudication_gate as runner
from scripts.server import run_pams_dense_timebase_v1_train337_gate as train_runner

POLICY = (
    Path(__file__).resolve().parents[1]
    / "configs/readouts/pams_dense_resampled_timebase_v2.yaml"
)


def _decode(
    *,
    counts: list[int],
    count: int,
    reference: int,
    selected: str,
    period: float,
) -> dict[str, object]:
    return {
        "count": count,
        "expert_counts": counts,
        "fft_bin": reference,
        "has_majority": max(counts.count(value) for value in set(counts)) >= 2,
        "period_confidence": 0.5,
        "period_frames": period,
        "reference_count": reference,
        "selected_expert": selected,
        "selection_mode": "multi",
    }


def _predecessor_rows() -> dict[str, object]:
    checks = {name: True for name in runner._EXPECTED_V1_CHECKS}
    for name in runner._EXPECTED_FAILED_CHECKS:
        checks[name] = False
    return {
        "checks": checks,
        "audit_rows": {
            "clean_count": [
                {
                    "target_count": 2,
                    "dense": _decode(
                        counts=[2, 1, 0],
                        count=2,
                        reference=2,
                        selected="fast",
                        period=128.0,
                    ),
                }
            ],
            "harmonic": [
                {
                    "fundamental_period_frames": 32.0,
                    "dense": _decode(
                        counts=[8, 7, 3],
                        count=8,
                        reference=8,
                        selected="fast",
                        period=32.0,
                    ),
                }
            ],
        },
    }


def test_cli_has_only_policy_predecessor_evidence_output_and_source() -> None:
    parsed = runner._parse_arguments(
        [
            "--policy",
            "policy.yaml",
            "--predecessor-artifact",
            "v1.json",
            "--predecessor-receipt",
            "v1.receipt.json",
            "--output",
            "v2.json",
            "--source-git-sha",
            "a" * 40,
        ]
    )

    assert set(vars(parsed)) == {
        "policy",
        "predecessor_artifact",
        "predecessor_receipt",
        "output",
        "source_git_sha",
    }
    source = inspect.getsource(runner._parse_arguments)
    for forbidden in (
        "--checkpoint",
        "--pose-cache",
        "--manifest",
        "--target",
        "--label",
        "--count",
        "--dev",
        "--test",
        "--train",
    ):
        assert f'"{forbidden}' not in source
        assert f"'{forbidden}" not in source


def test_policy_is_exact_post_synthetic_and_keeps_candidate_unchanged(
    tmp_path: Path,
) -> None:
    policy, digest, semantic = runner._load_policy(POLICY)

    assert digest == runner._EXPECTED_POLICY_SHA256
    assert digest == hashlib.sha256(POLICY.read_bytes()).hexdigest()
    assert semantic == runner.sha256_json(policy)
    assert policy["candidate_id"] == runner._EXPECTED_CANDIDATE
    assert policy["model_candidate_id"] == runner._EXPECTED_MODEL_CANDIDATE
    assert policy["table2_eligible"] is False
    assert policy["authorization"]["test105_attempt_budget"] == 0
    assert policy["selection_boundary"]["revision_does_not_change"]

    drifted = deepcopy(policy)
    drifted["candidate_source_lock"]["period_py_sha256"] = "0" * 64
    path = tmp_path / "drifted.yaml"
    path.write_text(runner.yaml.safe_dump(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="exact frozen"):
        runner._load_policy(path)


def test_no_majority_adjudication_checks_algorithm1_fallback_and_truth() -> None:
    predecessor = _predecessor_rows()
    evaluation, checks = runner._adjudicate_rows(
        predecessor,
        expected_counts={
            "clean_count": 1,
            "harmonic": 1,
            "clean_no_majority": 1,
            "harmonic_no_majority": 1,
        },
    )

    assert all(checks.values())
    assert evaluation["metrics"]["clean_no_majority_fft_nearest_fraction"] == 1.0
    assert evaluation["metrics"]["harmonic_no_majority_generated_truth_fraction"] == 1.0
    assert evaluation["audit"]["clean"]["no_majority_rows"][0][
        "selected_expert"
    ] == "fast"


def test_no_majority_adjudication_fails_when_selected_candidate_is_not_nearest() -> None:
    predecessor = _predecessor_rows()
    predecessor["audit_rows"]["clean_count"][0]["dense"] = _decode(
        counts=[2, 1, 0],
        count=1,
        reference=2,
        selected="medium",
        period=128.0,
    )

    _, checks = runner._adjudicate_rows(
        predecessor,
        expected_counts={
            "clean_count": 1,
            "harmonic": 1,
            "clean_no_majority": 1,
            "harmonic_no_majority": 1,
        },
    )

    assert checks["clean_no_majority_fft_nearest_fraction"] is False
    assert checks["clean_no_majority_generated_truth_fraction"] is False


def test_artifact_receipt_is_exclusive_and_hash_bound(tmp_path: Path) -> None:
    output = tmp_path / "v2.json"
    digest = "a" * 64
    payload = {
        "status": "passed",
        "candidate_id": runner._EXPECTED_CANDIDATE,
        "model_candidate_id": runner._EXPECTED_MODEL_CANDIDATE,
        "source": {
            "policy_sha256": digest,
            "policy_semantic_sha256": digest,
            "runner_source_git_sha": "b" * 40,
            "runner_sha256": digest,
        },
        "inputs": {
            "predecessor_artifact_sha256": runner._EXPECTED_PREDECESSOR[
                "artifact_sha256"
            ],
            "predecessor_receipt_sha256": runner._EXPECTED_PREDECESSOR[
                "receipt_sha256"
            ],
        },
        "authorization": {"train337_gate_authorized": True},
    }

    receipt_path, artifact_sha256 = runner._write_artifact_and_receipt(output, payload)

    artifact = output.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert artifact_sha256 == hashlib.sha256(artifact).hexdigest()
    assert receipt["artifact_sha256"] == artifact_sha256
    assert receipt["predecessor_artifact_sha256"] == runner._EXPECTED_PREDECESSOR[
        "artifact_sha256"
    ]
    assert receipt["train337_gate_authorized"] is True
    assert receipt["dev84_prediction_authorized"] is False
    assert receipt["test105_evaluation_authorized"] is False
    with pytest.raises(FileExistsError, match="destinations must be new"):
        runner._write_artifact_and_receipt(output, payload)


def test_train337_loader_and_prerequisite_accept_exact_v2_adjudication(
    tmp_path: Path,
) -> None:
    policy, policy_sha256, policy_semantic = train_runner._load_policy(POLICY)
    source_git_sha = "c" * 40
    adjudication_runner_sha256 = "d" * 64
    artifact_path = tmp_path / "v2.json"
    receipt_path = tmp_path / "v2.json.receipt.json"
    metrics = {
        "predecessor_only_expected_failures": True,
        "predecessor_output_quality_checks": True,
        "clean_no_majority_case_count": 16,
        "harmonic_no_majority_case_count": 19,
        "clean_no_majority_fft_nearest_fraction": 1.0,
        "harmonic_no_majority_fft_nearest_fraction": 1.0,
        "clean_no_majority_selected_expert_fraction": 1.0,
        "harmonic_no_majority_selected_expert_fraction": 1.0,
        "clean_no_majority_generated_truth_fraction": 1.0,
        "harmonic_no_majority_generated_truth_fraction": 1.0,
        "candidate_source_files_unchanged": True,
    }
    source_files = {
        "src/pams/period.py": policy["candidate_source_lock"]["period_py_sha256"],
        "src/pams/consensus.py": policy["candidate_source_lock"][
            "consensus_py_sha256"
        ],
        "src/pams/config.py": policy["candidate_source_lock"]["config_py_sha256"],
    }
    artifact = {
        "schema_version": 1,
        "artifact_type": "pams_dense_timebase_v2_synthetic_adjudication_gate",
        "candidate_id": train_runner._EXPECTED_V2_CANDIDATE,
        "model_candidate_id": train_runner._EXPECTED_CANDIDATE,
        "classification": policy["classification"],
        "status": "passed",
        "passed": True,
        "table2_eligible": False,
        "labels_accessed": False,
        "dataset_inputs_accessed": False,
        "checkpoint_inputs_accessed": False,
        "dev84_inputs_accessed": False,
        "test105_inputs_accessed": False,
        "source": {
            "runner_source_git_sha": source_git_sha,
            "runner_sha256": adjudication_runner_sha256,
            "policy_sha256": policy_sha256,
            "policy_semantic_sha256": policy_semantic,
            "candidate_source_sha256": source_files,
        },
        "inputs": {
            "predecessor_artifact_sha256": train_runner._EXPECTED_V2_PREDECESSOR[
                "artifact_sha256"
            ],
            "predecessor_artifact_bytes": 1,
            "predecessor_receipt_sha256": train_runner._EXPECTED_V2_PREDECESSOR[
                "receipt_sha256"
            ],
            "predecessor_receipt_bytes": 1,
        },
        "protocol_correction": {
            "removed_checks": sorted(runner._EXPECTED_FAILED_CHECKS),
            "reason": "Algorithm 1",
            "candidate_algorithm_changed": False,
            "expert_parameters_changed": False,
            "synthetic_cases_changed": False,
            "generated_truth_changed": False,
        },
        "thresholds": policy["synthetic_adjudication_gate"]["thresholds"],
        "metrics": metrics,
        "checks": {
            name: True for name in train_runner._EXPECTED_V2_SYNTHETIC_CHECKS
        },
        "authorization": {
            "predecessor_failure_verified": True,
            "train337_gate_authorized": True,
            "dev84_prediction_authorized": False,
            "dev84_scoring_authorized": False,
            "test105_evaluation_authorized": False,
        },
    }
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    artifact_bytes = artifact_path.read_bytes()
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    receipt = {
        "schema_version": 1,
        "artifact_type": (
            "pams_dense_timebase_v2_synthetic_adjudication_gate_receipt"
        ),
        "artifact_locator": artifact_path.name,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": len(artifact_bytes),
        "artifact_status": "passed",
        "candidate_id": train_runner._EXPECTED_V2_CANDIDATE,
        "model_candidate_id": train_runner._EXPECTED_CANDIDATE,
        "policy_sha256": policy_sha256,
        "policy_semantic_sha256": policy_semantic,
        "runner_source_git_sha": source_git_sha,
        "runner_sha256": adjudication_runner_sha256,
        "predecessor_artifact_sha256": train_runner._EXPECTED_V2_PREDECESSOR[
            "artifact_sha256"
        ],
        "predecessor_receipt_sha256": train_runner._EXPECTED_V2_PREDECESSOR[
            "receipt_sha256"
        ],
        "labels_accessed": False,
        "dataset_inputs_accessed": False,
        "checkpoint_inputs_accessed": False,
        "train337_gate_authorized": True,
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_bytes = receipt_path.read_bytes()
    identities = {
        "synthetic_gate_artifact": (artifact_sha256, len(artifact_bytes)),
        "synthetic_gate_receipt": (
            hashlib.sha256(receipt_bytes).hexdigest(),
            len(receipt_bytes),
        ),
        "synthetic_gate_runner": (adjudication_runner_sha256, 1),
    }

    binding = train_runner._validate_v2_adjudication_gate_pair(
        artifact_path,
        receipt_path,
        identities=identities,
        policy=policy,
        policy_sha256=policy_sha256,
        policy_semantic_sha256=policy_semantic,
        runtime_source_git_sha=source_git_sha,
    )

    assert binding["predecessor_failure_verified"] is True
    assert binding["train337_gate_authorized"] is True
