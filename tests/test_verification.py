from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import yaml

from pams.verification import (
    FROZEN_GATE,
    MetricPair,
    SeedResult,
    VerificationEvidence,
    VerificationStatus,
    evaluate_verification,
)


def _passing_ablations() -> dict[str, MetricPair]:
    return {
        "baseline": MetricPair(0.34, 0.42),
        "pams_no_multi_scale": MetricPair(0.30, 0.50),
        "pams_multi_scale": MetricPair(0.26, 0.58),
        "multi_expert_no_multi_expert": MetricPair(0.29, 0.62),
        "multi_expert": MetricPair(0.24, 0.65),
        "full": MetricPair(0.20, 0.70),
    }


def _seed_results() -> tuple[SeedResult, ...]:
    return (
        SeedResult(42, MetricPair(0.21, 0.68)),
        SeedResult(2026, MetricPair(0.22, 0.67)),
        SeedResult(3407, MetricPair(0.23, 0.65)),
    )


def test_no_evidence_is_explicit_no_results() -> None:
    decision = evaluate_verification(VerificationEvidence())
    assert decision.status is VerificationStatus.NO_RESULTS
    assert decision.mean_metrics is None
    assert all(check.passed in {False, None} for check in decision.checks)


def test_incomplete_evidence_is_unverified_not_partial() -> None:
    evidence = VerificationEvidence(seed_results=(_seed_results()[0],))
    decision = evaluate_verification(evidence)
    assert decision.status is VerificationStatus.UNVERIFIED
    assert decision.missing_seeds == (2026, 3407)
    assert decision.mean_metrics is None


def test_all_frozen_gates_produce_verified_status() -> None:
    evidence = VerificationEvidence(
        seed_results=_seed_results(),
        ablations=_passing_ablations(),
    )
    decision = evaluate_verification(evidence)
    assert decision.status is VerificationStatus.VERIFIED
    assert decision.verified
    assert decision.mean_metrics is not None
    assert decision.mean_metrics.nmae == pytest.approx(0.22)
    assert decision.mean_metrics.obo == pytest.approx(2.0 / 3.0)
    assert decision.individual_passes == 2
    assert all(check.passed is True for check in decision.checks)


def test_complete_but_failed_ablation_is_partial() -> None:
    ablations = _passing_ablations()
    ablations["pams_multi_scale"] = MetricPair(0.31, 0.49)
    decision = evaluate_verification(
        VerificationEvidence(seed_results=_seed_results(), ablations=ablations)
    )
    assert decision.status is VerificationStatus.PARTIAL
    multiscale = next(check for check in decision.checks if check.key == "multiscale_improves")
    assert multiscale.passed is False


def test_unknown_seed_or_ablation_cannot_move_the_gate() -> None:
    with pytest.raises(ValueError, match="non-preregistered seeds"):
        evaluate_verification(
            VerificationEvidence(seed_results=(SeedResult(7, MetricPair(0.1, 0.9)),))
        )
    with pytest.raises(ValueError, match="non-preregistered ablations"):
        evaluate_verification(
            VerificationEvidence(ablations={"secret_variant": MetricPair(0.1, 0.9)})
        )


def test_gate_and_evidence_are_immutable() -> None:
    evidence = VerificationEvidence(ablations=_passing_ablations())
    with pytest.raises(TypeError):
        evidence.ablations["full"] = MetricPair(0.0, 1.0)  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        FROZEN_GATE.maximum_mean_nmae = 1.0  # type: ignore[misc]


def test_frozen_protocol_and_stress_configs_match_gate() -> None:
    root = Path(__file__).parents[1]
    with (root / "configs/protocols/ucfrep_526.yaml").open("r", encoding="utf-8") as handle:
        standard = yaml.safe_load(handle)
    with (root / "configs/protocols/ucfrep_pose_110.yaml").open("r", encoding="utf-8") as handle:
        pose = yaml.safe_load(handle)
    with (root / "configs/ablations/table2.yaml").open("r", encoding="utf-8") as handle:
        ablations = yaml.safe_load(handle)
    with (root / "configs/stress.yaml").open("r", encoding="utf-8") as handle:
        stress = yaml.safe_load(handle)

    assert standard["splits"]["train_pool"]["count"] == 421
    assert standard["splits"]["sealed_test"]["count"] == 105
    assert standard["splits"]["development"]["train_count"] == 337
    assert standard["splits"]["development"]["dev_count"] == 84
    assert (
        standard["dataset"]["annotation_archive"]["sha256"]
        == "08b2b8a88c2728e9aa6de6c62dc6b02f1941dfa2c3adb28dab5852c12695ae3c"
    )
    assert standard["splits"]["train_pool"]["source_mat_directory"] == "annotations/train"
    assert standard["splits"]["sealed_test"]["source_mat_directory"] == "annotations/val"
    assert (
        standard["splits"]["development"]["algorithm"] == "pams.data.deterministic_stratified_split"
    )
    assert standard["splits"]["development"]["count_bins"][-2:] == [
        {"minimum": 21, "maximum": 40},
        {"minimum": 41, "maximum": None},
    ]
    assert (
        standard["split_integrity"]["annotation_only_manifest_sha256"]
        == "e0fde9919b5fa3b918d3e97636066e9008a7a5f0572d83fe684f0bb4f4132e30"
    )
    assert pose["splits"]["train_pool"]["count"] == 89
    assert pose["splits"]["sealed_test"]["count"] == 21
    assert pose["splits"]["development"]["train_count"] == 71
    assert pose["splits"]["development"]["dev_count"] == 18
    assert tuple(standard["seeds"]) == FROZEN_GATE.required_seeds
    assert tuple(pose["seeds"]) == FROZEN_GATE.required_seeds
    assert standard["label_firewall"]["sealed_test_labels"] == "evaluator_only"
    assert pose["label_firewall"]["ground_truth_count_oracle"]["allowed"] is False
    assert tuple(row["key"] for row in ablations["rows"]) == FROZEN_GATE.required_ablations
    assert stress["synthetic"]["count_values"][-1] == 40
    occlusion = next(
        row for row in stress["ucfrep_perturbations"] if row["key"] == "pose_occlusion"
    )
    assert occlusion["parameters"] == {
        "time_fraction": 0.20,
        "joint_fraction": 0.30,
        "fill_value": 0.0,
        "mark_invalid": True,
    }
