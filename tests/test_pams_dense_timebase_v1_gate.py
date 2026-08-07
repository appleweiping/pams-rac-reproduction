from __future__ import annotations

import hashlib
import inspect
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from scripts.server import run_pams_dense_timebase_v1_gate as runner

POLICY = (
    Path(__file__).resolve().parents[1]
    / "configs/readouts/pams_dense_resampled_timebase_v1.yaml"
)


def _passing_metrics() -> dict[str, object]:
    return {
        "legacy_all_valid_exact_parity": True,
        "median_period_relative_error": 0.0,
        "p95_period_relative_error": 0.0,
        "harmonic_alias_fraction": 0.0,
        "chain_nmae": 0.0,
        "chain_obo": 1.0,
        "clean_majority_fraction": 1.0,
        "harmonic_majority_fraction": 1.0,
        "resample_period_relative_error_maximum": 0.0,
        "resample_count_equal_fraction": 1.0,
        "invalid_payload_exact_invariance": True,
        "all_selected_modes_multi": True,
        "existing_576_counter_gate_passed": True,
    }


def _artifact_payload() -> dict[str, object]:
    digest = "a" * 64
    return {
        "status": "passed",
        "candidate_id": runner._EXPECTED_CANDIDATE,
        "source": {
            "policy_sha256": digest,
            "policy_semantic_sha256": digest,
            "runner_source_git_sha": "b" * 40,
            "runner_sha256": digest,
        },
        "authorization": {"train337_gate_authorized": True},
    }


def test_cli_is_synthetic_only_and_has_no_external_scientific_inputs() -> None:
    parsed = runner._parse_arguments(
        [
            "synthetic",
            "--policy",
            "policy.yaml",
            "--output",
            "synthetic.json",
            "--repository-root",
            ".",
        ]
    )

    assert set(vars(parsed)) == {"command", "policy", "output", "repository_root"}
    assert parsed.command == "synthetic"
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


def test_policy_loader_freezes_single_dense_multi_candidate(tmp_path: Path) -> None:
    policy, digest, semantic = runner._load_policy(POLICY)

    assert digest == hashlib.sha256(POLICY.read_bytes()).hexdigest()
    assert semantic == runner.sha256_json(policy)
    assert policy["candidate_id"] == runner._EXPECTED_CANDIDATE
    assert policy["readout"]["direct_fft_timebase"] == "dense_resampled"
    assert policy["readout"]["consensus_expert_mode"] == "multi"
    assert policy["selection_boundary"]["candidate_count"] == 1
    assert policy["authorization"]["test105_attempt_budget"] == 0

    drifted = deepcopy(policy)
    drifted["readout"]["consensus_expert_mode"] = "reference_nearest"
    path = tmp_path / "drifted.yaml"
    path.write_text(runner.yaml.safe_dump(drifted), encoding="utf-8")
    with pytest.raises(ValueError, match="consensus_expert_mode"):
        runner._load_policy(path)


def test_mask_and_repeat_generators_are_deterministic_and_gap_safe() -> None:
    stream = torch.arange(256, dtype=torch.float64)
    masks = runner._mask_variants(256)

    assert set(masks) == {
        "long_gap",
        "fragmented",
        "leading_and_trailing_padding",
    }
    assert not bool(masks["long_gap"][80:176].any())
    assert bool(masks["long_gap"][:80].all())
    assert bool(masks["long_gap"][176:].all())
    polluted = runner._pollute_invalid(stream, masks["long_gap"])
    assert torch.equal(polluted[masks["long_gap"]], stream[masks["long_gap"]])
    assert not torch.equal(
        polluted[~masks["long_gap"]], stream[~masks["long_gap"]]
    )

    repeated_stream, repeated_mask = runner._repeat_two_x(
        stream,
        masks["fragmented"],
    )
    assert repeated_stream.shape == (512,)
    assert repeated_mask.shape == (512,)
    assert torch.equal(repeated_stream[0::2], stream)
    assert torch.equal(repeated_stream[1::2], stream)
    assert torch.equal(repeated_mask[0::2], masks["fragmented"])
    assert torch.equal(repeated_mask[1::2], masks["fragmented"])


def test_threshold_decision_is_policy_driven_and_fail_closed() -> None:
    policy, _, _ = runner._load_policy(POLICY)
    thresholds = policy["synthetic_gate"]["thresholds"]
    passing = _passing_metrics()

    assert all(runner._evaluate_checks(passing, thresholds).values())

    failures = {
        "legacy_all_valid_exact_parity": False,
        "median_period_relative_error": 1.0,
        "p95_period_relative_error": 1.0,
        "harmonic_alias_fraction": 1.0,
        "chain_nmae": 1.0,
        "chain_obo": 0.0,
        "clean_majority_fraction": 0.0,
        "harmonic_majority_fraction": 0.0,
        "resample_period_relative_error_maximum": 1.0,
        "resample_count_equal_fraction": 0.0,
        "invalid_payload_exact_invariance": False,
        "all_selected_modes_multi": False,
        "existing_576_counter_gate_passed": False,
    }
    for field, value in failures.items():
        candidate = dict(passing)
        candidate[field] = value
        checks = runner._evaluate_checks(candidate, thresholds)
        assert not all(checks.values()), field


def test_runner_uses_dense_direct_fft_unchanged_multi_counter_and_counter_gate() -> None:
    source = inspect.getsource(runner.run_synthetic_gate)

    assert "run_counter_sign_phase_gate(counter)" in source
    assert 'MultiExpertCounter(expert_mode="multi")' in source
    assert 'timebase="dense_resampled"' in source
    assert 'timebase="compact_valid"' in source
    assert "_repeat_two_x" in source
    assert "_pollute_invalid" in source
    assert "labels_accessed" in source
    assert "dev84_inputs_accessed" in source
    assert "test105_inputs_accessed" in source


def test_artifact_and_receipt_are_exclusive_hash_bound_and_scope_closed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "synthetic.json"
    payload = _artifact_payload()

    receipt_path, digest = runner._write_artifact_and_receipt(output, payload)

    artifact = output.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert digest == hashlib.sha256(artifact).hexdigest()
    assert receipt["artifact_sha256"] == digest
    assert receipt["artifact_bytes"] == len(artifact)
    assert receipt["labels_accessed"] is False
    assert receipt["dataset_inputs_accessed"] is False
    assert receipt["checkpoint_inputs_accessed"] is False
    assert receipt["train337_gate_authorized"] is True
    assert receipt["dev84_prediction_authorized"] is False
    assert receipt["dev84_scoring_authorized"] is False
    assert receipt["test105_evaluation_authorized"] is False
    with pytest.raises(FileExistsError, match="both be new"):
        runner._write_artifact_and_receipt(output, payload)
