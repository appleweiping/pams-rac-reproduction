from __future__ import annotations

import hashlib
import inspect
import json
import math
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from scripts.server import run_pams_dense_timebase_v1_train337_gate as runner

POLICY = (
    Path(__file__).resolve().parents[1]
    / "configs/readouts/pams_dense_resampled_timebase_v1.yaml"
)


def _decode_stub(
    *,
    period: float = 16.0,
    confidence: float = 0.75,
    count: int = 8,
) -> dict[str, object]:
    return {
        "timebase": "dense_resampled",
        "timeline_frames": 128,
        "valid_frames": 100,
        "clock_frames": 128,
        "period_frames": period,
        "period_confidence": confidence,
        "period_evidence": confidence > 0.0,
        "selected_bin": round(128 / period),
        "count": count,
        "expert_counts": [count, count, count + 1],
        "reference_count": count,
        "selected_expert": "medium",
        "selection_mode": "multi",
        "consensus_confidence": 0.5,
    }


def _passing_metrics() -> dict[str, object]:
    return {
        "completed_records": 337,
        "finite_records": 337,
        "period_evidence_records": 337,
        "collapsed_fraction": 0.0,
        "invalid_payload_exact_invariance_fraction": 1.0,
        "period_bin_closure_error_maximum": 0.0,
        "fixed_internal_gap_period_bin_retention": 1.0,
        "fixed_internal_gap_count_within_one": 1.0,
        "two_x_resample_period_scaling_median_error": 0.0,
        "two_x_resample_period_scaling_p90_error": 0.0,
        "two_x_resample_count_equal_fraction": 1.0,
        "compact_decoder_non_regression": True,
        "mask_robustness_relative_improvement": 1.0,
        "boundary_rate_increase": 0.0,
        "coverage_strata_reported": True,
        "all_selected_modes_multi": True,
    }


def _artifact_payload() -> dict[str, object]:
    digest = "a" * 64
    return {
        "status": "passed",
        "candidate_id": runner._EXPECTED_CANDIDATE,
        "source": {
            "policy_sha256": digest,
            "source_git_sha": "b" * 40,
            "runner_sha256": digest,
        },
        "inputs": {
            "train337_pose_cache_set_sha256": digest,
            "train337_pose_snapshot_sha256": digest,
            "synthetic_gate_artifact_sha256": digest,
            "synthetic_gate_receipt_sha256": digest,
        },
        "authorization": {
            "train337_gate_passed": True,
            "dev84_prediction_authorized": True,
        },
    }


def test_cli_accepts_only_frozen_train337_scientific_inputs() -> None:
    parsed = runner._parse_arguments(
        [
            "--policy",
            "policy.yaml",
            "--training-config",
            "old.yaml",
            "--sshead-checkpoint",
            "head.pt",
            "--sshead-progress",
            "head.jsonl",
            "--sshead-completion-receipt",
            "head.completed.json",
            "--encoder-checkpoint",
            "encoder.pt",
            "--encoder-progress",
            "encoder.jsonl",
            "--encoder-completion-receipt",
            "encoder.completed.json",
            "--train-sidecar",
            "train.json",
            "--train-commitment",
            "train.commitment.json",
            "--pose-cache",
            "poses",
            "--train-pose-snapshot",
            "train-pose-snapshot.json",
            "--synthetic-gate-artifact",
            "synthetic.json",
            "--synthetic-gate-receipt",
            "synthetic.receipt.json",
            "--output",
            "gate.json",
            "--device",
            "cuda:0",
            "--source-git-sha",
            "b" * 40,
            "--container-image-id",
            "sha256:" + "c" * 64,
            "--container-environment-sha256",
            "d" * 64,
        ]
    )

    assert set(vars(parsed)) == {
        "policy",
        "training_config",
        "sshead_checkpoint",
        "sshead_progress",
        "sshead_completion_receipt",
        "encoder_checkpoint",
        "encoder_progress",
        "encoder_completion_receipt",
        "train_sidecar",
        "train_commitment",
        "pose_cache",
        "train_pose_snapshot",
        "synthetic_gate_artifact",
        "synthetic_gate_receipt",
        "output",
        "device",
        "source_git_sha",
        "container_image_id",
        "container_environment_sha256",
    }
    source = inspect.getsource(runner._parse_arguments)
    for forbidden in ("--dev", "--test", "--target", "--count", "--action", "--label"):
        assert f'"{forbidden}"' not in source
        assert f"'{forbidden}'" not in source


def test_policy_loader_requires_exact_file_and_frozen_identities(tmp_path: Path) -> None:
    policy, digest, semantic = runner._load_policy(POLICY)

    assert digest == runner._EXPECTED_POLICY_SHA256
    assert digest == hashlib.sha256(POLICY.read_bytes()).hexdigest()
    assert semantic == runner.sha256_json(policy)
    assert policy["frozen_base"] == runner._EXPECTED_FROZEN_BASE
    assert (
        policy["train337_gate"]["metric_definitions"]
        == runner._EXPECTED_POLICY_METRIC_DEFINITIONS
    )
    assert set(policy["train337_gate"]["thresholds"]) == runner._REQUIRED_THRESHOLDS
    assert policy["authorization"]["test105_attempt_budget"] == 0

    drifted = deepcopy(policy)
    drifted["train337_gate"]["thresholds"]["collapsed_fraction_maximum"] = 1.0
    path = tmp_path / "drifted.yaml"
    path.write_text(runner.yaml.safe_dump(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="exact frozen"):
        runner._load_policy(path)


def test_internal_gap_uses_exact_center_eighth_and_affected_definition() -> None:
    mask = torch.zeros(257, dtype=torch.bool)
    expected_start = math.floor(7 * 257 / 16)
    expected_stop = math.ceil(9 * 257 / 16)
    mask[expected_start] = True
    mask[0] = True

    gapped, start, stop, affected = runner._internal_gap(mask)

    assert (start, stop) == (expected_start, expected_stop)
    assert affected is True
    assert not bool(gapped[start:stop].any())
    assert bool(gapped[0])
    outside_only = torch.zeros(257, dtype=torch.bool)
    outside_only[0] = True
    _, _, _, unaffected = runner._internal_gap(outside_only)
    assert unaffected is False


def test_invalid_pollution_and_repeat_are_deterministic() -> None:
    stream = torch.arange(8, dtype=torch.float64)
    mask = torch.tensor([True, False, False, True, False, True, False, True])

    polluted = runner._pollute_invalid(stream, mask)
    assert torch.equal(polluted[mask], stream[mask])
    assert polluted.tolist() == [0.0, -1e6, 1e6, 3.0, 1e6, 5.0, 1e6, 7.0]
    repeated_stream, repeated_mask = runner._repeat_two_x(stream, mask)
    assert torch.equal(repeated_stream[0::2], stream)
    assert torch.equal(repeated_stream[1::2], stream)
    assert torch.equal(repeated_mask[0::2], mask)
    assert torch.equal(repeated_mask[1::2], mask)


def test_collapse_uses_population_std_over_original_valid_samples() -> None:
    stream = torch.tensor([1.0, 100.0, 3.0, 999.0], dtype=torch.float64)
    mask = torch.tensor([True, False, True, False])

    assert runner._population_std(stream, mask) == 1.0
    assert runner._population_std(stream, torch.tensor([True, False, False, False])) == 0.0
    assert runner._population_std(stream, torch.zeros(4, dtype=torch.bool)) == 0.0


def test_period_closure_and_gap_robustness_follow_frozen_formulas() -> None:
    decode = _decode_stub(period=128.0 / 7.0, count=7)
    closure = runner._period_bin_closure(decode)
    assert closure == pytest.approx(0.0, abs=1e-15)

    base = _decode_stub(period=16.0, count=8)
    gap = _decode_stub(period=20.0, count=10)
    expected = abs(math.log2(20.0 / 16.0)) + 2.0 / 8.0
    assert runner._gap_robustness(base, gap) == pytest.approx(expected)


def test_invalid_payload_equality_is_byte_exact_for_float_fields() -> None:
    left = _decode_stub(period=16.0)
    right = deepcopy(left)
    assert runner._invalid_payload_exact(left, right)

    right["period_confidence"] = -0.0
    left["period_confidence"] = 0.0
    assert not runner._invalid_payload_exact(left, right)


def test_boundary_bins_use_decoder_rfftfreq_band() -> None:
    allowed = runner._allowed_bins(256, minimum=4, maximum=128)
    frequencies = torch.fft.rfftfreq(256)
    expected = tuple(
        int(index)
        for index in torch.nonzero(
            (frequencies >= 1 / 128) & (frequencies <= 1 / 4),
            as_tuple=False,
        ).flatten()
    )
    assert allowed == expected

    low_boundary = _decode_stub(period=128.0, count=2)
    low_boundary["timeline_frames"] = 256
    low_boundary["clock_frames"] = 256
    low_boundary["selected_bin"] = 2
    assert runner._is_boundary_decode(low_boundary, minimum=4, maximum=128)
    interior = deepcopy(low_boundary)
    interior["period_frames"] = 16.0
    interior["selected_bin"] = 16
    assert not runner._is_boundary_decode(interior, minimum=4, maximum=128)


@pytest.mark.parametrize(
    ("coverage", "expected"),
    [
        (0.0, "empty"),
        (0.001, "low"),
        (0.249, "low"),
        (0.25, "medium"),
        (0.749, "medium"),
        (0.75, "high"),
        (0.999, "high"),
        (1.0, "full"),
    ],
)
def test_coverage_strata_are_exact(coverage: float, expected: str) -> None:
    assert runner._coverage_stratum(coverage) == expected


def test_policy_threshold_decision_is_exact_and_fails_closed() -> None:
    policy, _, _ = runner._load_policy(POLICY)
    thresholds = policy["train337_gate"]["thresholds"]
    passing = _passing_metrics()

    assert all(runner._evaluate_checks(passing, thresholds).values())
    failures = {
        "completed_records": 336,
        "finite_records": 336,
        "period_evidence_records": 299,
        "collapsed_fraction": 0.11,
        "invalid_payload_exact_invariance_fraction": 0.99,
        "period_bin_closure_error_maximum": 0.1,
        "fixed_internal_gap_period_bin_retention": 0.0,
        "fixed_internal_gap_count_within_one": 0.0,
        "two_x_resample_period_scaling_median_error": 0.1,
        "two_x_resample_period_scaling_p90_error": 0.1,
        "two_x_resample_count_equal_fraction": 0.0,
        "compact_decoder_non_regression": False,
        "mask_robustness_relative_improvement": 0.0,
        "boundary_rate_increase": 0.1,
        "coverage_strata_reported": False,
        "all_selected_modes_multi": False,
    }
    for field, value in failures.items():
        candidate = dict(passing)
        candidate[field] = value
        assert not all(runner._evaluate_checks(candidate, thresholds).values()), field

    for nullable in (
        "period_bin_closure_error_maximum",
        "fixed_internal_gap_period_bin_retention",
        "fixed_internal_gap_count_within_one",
        "two_x_resample_period_scaling_median_error",
        "two_x_resample_period_scaling_p90_error",
        "two_x_resample_count_equal_fraction",
        "mask_robustness_relative_improvement",
        "boundary_rate_increase",
    ):
        candidate = dict(passing)
        candidate[nullable] = None
        assert not all(runner._evaluate_checks(candidate, thresholds).values()), nullable


def test_runner_generates_one_stream_set_then_decodes_same_stream_both_ways() -> None:
    source = inspect.getsource(runner.run_train337_gate)
    evaluate_source = inspect.getsource(runner._evaluate_records)

    assert source.count("_generate_period_head_streams_once(") == 1
    assert source.index("observed_inputs != exact_inputs") < source.index("_peek_checkpoint(")
    assert "validate_sshead_encoder_binding(" in source
    assert source.count("validate_terminal_checkpoint(") == 2
    assert source.count("_validate_completion_receipt(") == 2
    assert "_validate_synthetic_gate_pair(" in source
    assert "load_pose_cache_set(" in source
    assert 'timebase="dense_resampled"' in evaluate_source
    assert 'timebase="compact_valid"' in evaluate_source
    assert "_pollute_invalid(stream, mask)" in evaluate_source
    assert "_repeat_two_x(stream, mask)" in evaluate_source
    assert "shared_gap_rows" in evaluate_source
    assert "dense_base[\"period_evidence\"] and compact_base[\"period_evidence\"]" in (
        evaluate_source
    )
    assert "paired boundary observation accounting mismatch" in evaluate_source
    assert "predict_sequence" not in source


def test_synthetic_prerequisite_is_source_runner_policy_and_receipt_bound(
    tmp_path: Path,
) -> None:
    policy, policy_sha256, policy_semantic_sha256 = runner._load_policy(POLICY)
    source_git_sha = "e" * 40
    synthetic_runner_sha256 = "f" * 64
    artifact_path = tmp_path / "synthetic.json"
    receipt_path = tmp_path / "synthetic.json.receipt.json"
    artifact = {
        "schema_version": 1,
        "artifact_type": "pams_dense_timebase_v1_synthetic_gate",
        "candidate_id": runner._EXPECTED_CANDIDATE,
        "status": "passed",
        "passed": True,
        "table2_eligible": False,
        "labels_accessed": False,
        "dataset_inputs_accessed": False,
        "checkpoint_inputs_accessed": False,
        "dev84_inputs_accessed": False,
        "test105_inputs_accessed": False,
        "source": {
            "policy_sha256": policy_sha256,
            "policy_semantic_sha256": policy_semantic_sha256,
            "runner_source_git_sha": source_git_sha,
            "runner_sha256": synthetic_runner_sha256,
        },
        "thresholds": policy["synthetic_gate"]["thresholds"],
        "configuration": {
            "direct_fft_timebase": "dense_resampled",
            "comparison_timebase": "compact_valid",
            "consensus_expert_mode": "multi",
            "candidate_count": 1,
            "parameter_sweep": False,
        },
        "checks": {name: True for name in runner._EXPECTED_SYNTHETIC_CHECKS},
        "authorization": {
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
        "artifact_type": "pams_dense_timebase_v1_synthetic_gate_receipt",
        "artifact_locator": artifact_path.name,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": len(artifact_bytes),
        "artifact_status": "passed",
        "candidate_id": runner._EXPECTED_CANDIDATE,
        "policy_sha256": policy_sha256,
        "policy_semantic_sha256": policy_semantic_sha256,
        "runner_source_git_sha": source_git_sha,
        "runner_sha256": synthetic_runner_sha256,
        "labels_accessed": False,
        "dataset_inputs_accessed": False,
        "checkpoint_inputs_accessed": False,
        "train337_gate_authorized": True,
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    identities = {
        "synthetic_gate_artifact": (artifact_sha256, len(artifact_bytes)),
        "synthetic_gate_receipt": (
            hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            len(receipt_path.read_bytes()),
        ),
        "synthetic_gate_runner": (synthetic_runner_sha256, 123),
    }

    binding = runner._validate_synthetic_gate_pair(
        artifact_path,
        receipt_path,
        identities=identities,
        policy_sha256=policy_sha256,
        policy_semantic_sha256=policy_semantic_sha256,
        synthetic_thresholds=policy["synthetic_gate"]["thresholds"],
        runtime_source_git_sha=source_git_sha,
    )
    assert binding["train337_gate_authorized"] is True

    artifact["checks"].pop(next(iter(runner._EXPECTED_SYNTHETIC_CHECKS)))
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="all-pass check set"):
        runner._validate_synthetic_gate_pair(
            artifact_path,
            receipt_path,
            identities=identities,
            policy_sha256=policy_sha256,
            policy_semantic_sha256=policy_semantic_sha256,
            synthetic_thresholds=policy["synthetic_gate"]["thresholds"],
            runtime_source_git_sha=source_git_sha,
        )


def test_metric_definitions_disclose_every_frozen_formula() -> None:
    definitions = runner._definitions()
    assert set(definitions) == {
        "stream_collapse",
        "period_evidence",
        "period_bin_closure",
        "fixed_internal_gap",
        "invalid_payload_invariance",
        "two_x_repeat_resample",
        "mask_robustness",
        "boundary_rate",
        "coverage_strata",
        "finite_records",
    }
    assert "unbiased=False" in definitions["stream_collapse"]
    assert "[floor(7L/16),ceil(9L/16))" in definitions["fixed_internal_gap"]
    assert "repeat_interleave(2)" in definitions["two_x_repeat_resample"]
    assert "abs(log2(T_gap/T_base))" in definitions["mask_robustness"]
    assert "Core paired eligibility" in definitions["mask_robustness"]
    assert "boundary denominators are paired" in definitions["boundary_rate"]


def test_artifact_and_receipt_are_exclusive_hash_bound_and_scope_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "train337.json"
    payload = _artifact_payload()

    receipt_path, digest = runner._write_artifact_and_receipt(output, payload)

    artifact = output.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert digest == hashlib.sha256(artifact).hexdigest()
    assert receipt["artifact_sha256"] == digest
    assert receipt["artifact_bytes"] == len(artifact)
    assert receipt["train337_pose_cache_set_sha256"] == "a" * 64
    assert receipt["train337_pose_snapshot_file_sha256"] == "a" * 64
    assert receipt["labels_accessed"] is False
    assert receipt["counts_accessed"] is False
    assert receipt["actions_accessed"] is False
    assert receipt["dev84_inputs_accessed"] is False
    assert receipt["test105_inputs_accessed"] is False
    assert receipt["train337_gate_passed"] is True
    assert receipt["synthetic_gate_artifact_sha256"] == "a" * 64
    assert receipt["synthetic_gate_receipt_sha256"] == "a" * 64
    assert receipt["dev84_prediction_authorized"] is True
    assert receipt["dev84_scoring_authorized"] is False
    assert receipt["test105_evaluation_authorized"] is False
    with pytest.raises(FileExistsError, match="both be new"):
        runner._write_artifact_and_receipt(output, payload)
