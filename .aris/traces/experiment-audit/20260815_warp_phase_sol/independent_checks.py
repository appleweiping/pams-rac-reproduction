"""Read-only clean-room checks for the 2026-08-15 WARP-PHASE audit.

The script reads only the enumerated audit inputs, source code under ``src/pams``
and ``scripts/experiments``, and the receipt-bound synthetic Gate-2 fixture pack.
It never opens real, test, sealed, held-out, server, feature, vault, or result data.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from pams.warp_phase.evaluator import (  # noqa: E402
    EvaluationError,
    classify_harmonic,
    evaluate_counts,
    evaluate_harmonic_gate,
    evaluate_k4,
    evaluate_supplied_track_period_ap,
    paired_component_bootstrap,
    period_median,
)


ENUMERATED_INPUTS = (
    "src/pams/warp_phase/evaluator.py",
    "src/pams/warp_phase/selector.py",
    "src/pams/warp_phase/gates.py",
    "src/pams/warp_phase/cli.py",
    "scripts/experiments/generate_warp_phase_fixture_pack.py",
    "scripts/experiments/generate_warp_phase_selector_unit_fixtures.py",
    "data/warp_phase_pilot_v1/audit/gate0/gate0-local-inputs.receipt.json",
    "data/warp_phase_pilot_v1/audit/gate1/gate1-v2-unit-v1.receipt.json",
    "data/warp_phase_pilot_v1/audit/gate1/selector-unit-fixtures-v1/selector-unit-fixtures-v1.receipt.json",
    "data/warp_phase_pilot_v1/audit/gate1/selector_unit_fixture_pack_audit.md",
    "data/warp_phase_pilot_v1/audit/gate1/selector_unit_fixture_pack_audit.json",
    "data/warp_phase_pilot_v1/audit/gate1/candidate-20260815-sol-v2/canonical_fixture_pack.receipt.json",
    "data/warp_phase_pilot_v1/audit/gate1/fixture_candidate_v2_audit.md",
    "data/warp_phase_pilot_v1/audit/gate1/fixture_candidate_v2_audit.json",
    "data/warp_phase_pilot_v1/audit/gate1/fixture_candidate_v2_repair_adjudication.md",
    "data/warp_phase_pilot_v1/audit/gate1/fixture_candidate_v2_repair_adjudication.json",
    "data/warp_phase_pilot_v1/audit/gate2/gate2-stochastic-v2.receipt.json",
    "data/warp_phase_pilot_v1/audit/gate3/gate3-cpu-sanity.receipt.json",
    "refine-logs/EXPERIMENT_TRACKER.md",
    "refine-logs/PIPELINE_SUMMARY.md",
    "idea-stage/docs/research_contract.md",
    "refine-logs/FINAL_PROPOSAL.md",
    "configs/experiments/warp_phase_pilot_v1.yaml",
    "pyproject.toml",
    ".aris/compute/provider-env.json",
    ".aris/compute/output-budget.json",
    "PILOT_DATA_SCHEMA_AUDIT.md",
    "PILOT_DATA_SCHEMA_AUDIT.json",
    "PILOT_PERIOD_SCHEMA_ADDENDUM.md",
    "PILOT_PERIOD_SCHEMA_ADDENDUM.json",
    "refine-logs/SERVER_PREFLIGHT.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gate2_fixture_check() -> dict[str, object]:
    candidate = ROOT / "data/warp_phase_pilot_v1/audit/gate1/candidate-20260815-sol-v2"
    pack = candidate / "pack"
    receipt_path = candidate / "canonical_fixture_pack.receipt.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    canonical = json.dumps(
        receipt,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if canonical != receipt_bytes:
        raise AssertionError("canonical fixture receipt is not canonical JSON")

    aggregate = hashlib.sha256()
    member_hash_mismatches: list[str] = []
    for name in sorted(receipt["members"]):
        payload = (pack / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != receipt["members"][name]:
            member_hash_mismatches.append(name)
        aggregate.update(name.encode("utf-8"))
        aggregate.update(b"\x00")
        aggregate.update(payload)
        aggregate.update(b"\n")

    selected_raw = np.load(pack / "expected_selected_period.npy", allow_pickle=False)
    weak_raw = np.load(pack / "expected_weak_selected_period.npy", allow_pickle=False)
    semantic = np.asarray(np.load(pack / "semantic_period.npy", allow_pickle=False), dtype=np.float64)
    selected = np.asarray(selected_raw, dtype=np.int64)
    weak = np.asarray(weak_raw, dtype=np.int64)
    valid_pair = (selected > 0) & (weak > 0)
    agreement = valid_pair & (10 * np.abs(weak - selected) <= selected)
    valid_selected = selected > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        half = valid_selected & (np.abs(selected / (0.5 * semantic) - 1.0) <= 0.10)
        double = valid_selected & (np.abs(selected / (2.0 * semantic) - 1.0) <= 0.10)
    symmetric = np.zeros(1000, dtype=np.bool_)
    symmetric[250:500] = True
    counts = {
        "canonical_weak_agreement": int(np.sum(agreement, dtype=np.int64)),
        "half_overall": int(np.sum(half, dtype=np.int64)),
        "double_overall": int(np.sum(double, dtype=np.int64)),
        "half_symmetric": int(np.sum(half & symmetric, dtype=np.int64)),
        "double_symmetric": int(np.sum(double & symmetric, dtype=np.int64)),
    }
    expected = {
        "canonical_weak_agreement": 246,
        "half_overall": 55,
        "double_overall": 144,
        "half_symmetric": 50,
        "double_symmetric": 66,
    }
    if counts != expected:
        raise AssertionError(f"Gate-2 count mismatch: {counts}")
    return {
        "canonical_receipt_sha256": sha256(receipt_path),
        "member_count": len(receipt["members"]),
        "member_hash_mismatches": member_hash_mismatches,
        "pack_sha256_recomputed": aggregate.hexdigest(),
        "pack_sha256_receipt": receipt["pack_sha256"],
        "counts": counts,
        "thresholds_all_pass": bool(
            counts["canonical_weak_agreement"] >= 950
            and counts["half_overall"] <= 20
            and counts["double_overall"] <= 20
            and counts["half_symmetric"] <= 12
            and counts["double_symmetric"] <= 12
        ),
        "canonical_failure_count": int(np.sum(selected <= 0, dtype=np.int64)),
    }


def evaluator_smoke_check() -> dict[str, object]:
    ground_truth = {"A": {"p0": 2.0, "p1": 4.0}, "B": {"p0": 5.0}}
    control = {"A": {"p0": 3.0, "p1": 2.0}, "B": {"p0": 7.0}}
    treatment = {"A": {"p0": 2.0, "p1": 3.0}, "B": {"p0": 6.0}}
    control_score = evaluate_counts(ground_truth, control)
    treatment_score = evaluate_counts(ground_truth, treatment)
    if not math.isclose(control_score.normalized_video_first_mae, 0.45):
        raise AssertionError("control video-first AvgMAE mismatch")
    if not math.isclose(treatment_score.normalized_video_first_mae, 0.1625):
        raise AssertionError("treatment video-first AvgMAE mismatch")

    nine_gt = {f"v{i}": {"p": 10.0} for i in range(9)}
    nine_control = {f"v{i}": {"p": 12.0} for i in range(9)}
    nine_treatment = {f"v{i}": {"p": 11.0} for i in range(9)}
    seed_map_control = {seed: nine_control for seed in (20270815, 20270816, 20270817)}
    seed_map_treatment = {seed: nine_treatment for seed in (20270815, 20270816, 20270817)}
    component = {f"v{i}": f"c{i}" for i in range(9)}
    endpoint = paired_component_bootstrap(
        nine_gt,
        seed_map_control,
        seed_map_treatment,
        component,
    )

    period = evaluate_supplied_track_period_ap(
        {"track": ((0, 10), (20, 30))},
        {"track": ((0.0, 10.0, 0.9), (20.0, 30.0, 0.8))},
    )
    harmonic = evaluate_harmonic_gate(
        {"video": {"person0": (10.0, 10.0), "person1": (20.0, 20.0)}}
    )
    k4 = evaluate_k4(
        control=0.4,
        treatment=0.2,
        clock=0.38,
        mask=0.38,
        control_shuffle=0.5,
        treatment_shuffle=0.48,
        control_unseen=0.5,
        treatment_unseen=0.32,
    )
    try:
        classify_harmonic(float("nan"), 10.0)
    except EvaluationError:
        nonfinite_harmonic_behavior = "raises_EvaluationError"
    else:
        nonfinite_harmonic_behavior = "returns_class"

    return {
        "count_fixture": {
            "control_normalized_video_first_mae": control_score.normalized_video_first_mae,
            "treatment_normalized_video_first_mae": treatment_score.normalized_video_first_mae,
            "control_avg_obo": control_score.avg_obo,
            "treatment_avg_obo": treatment_score.avg_obo,
        },
        "bootstrap": {
            "control_mean": endpoint.control_mean,
            "treatment_mean": endpoint.treatment_mean,
            "absolute_effect": endpoint.absolute_effect,
            "relative_improvement": endpoint.relative_improvement,
            "ci_lower": endpoint.ci_lower,
            "ci_upper": endpoint.ci_upper,
            "draws": endpoint.draws,
            "passed": endpoint.passed,
        },
        "period_ap": {
            "period_map": period.period_map,
            "ap50": period.ap50,
            "ap75": period.ap75,
            "scale": period.scale,
        },
        "period_median": period_median(((0, 10), (20, 32)), 32),
        "harmonic": {"fractions": harmonic.fractions, "passed": harmonic.passed},
        "k4": {
            "q_clock": k4.q_clock,
            "q_mask": k4.q_mask,
            "p_shuffle": k4.p_shuffle,
            "p_unseen": k4.p_unseen,
            "passed": k4.passed,
        },
        "nonfinite_harmonic_behavior": nonfinite_harmonic_behavior,
    }


def evaluator_reference_scan() -> dict[str, list[str]]:
    roots = (ROOT / "src/pams", ROOT / "scripts/experiments")
    references: dict[str, list[str]] = {
        "static_imports_of_pams_warp_phase_evaluator": [],
        "literal_module_references": [],
    }
    evaluator_path = (ROOT / "src/pams/warp_phase/evaluator.py").resolve()
    for scan_root in roots:
        for path in sorted(scan_root.rglob("*.py")):
            if path.resolve() == evaluator_path or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=str(path))
            for node in ast.walk(tree):
                imported_module: str | None = None
                if isinstance(node, ast.ImportFrom):
                    imported_module = node.module
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "pams.warp_phase.evaluator":
                            references["static_imports_of_pams_warp_phase_evaluator"].append(
                                f"{path.relative_to(ROOT).as_posix()}:{node.lineno}"
                            )
                if imported_module == "pams.warp_phase.evaluator":
                    references["static_imports_of_pams_warp_phase_evaluator"].append(
                        f"{path.relative_to(ROOT).as_posix()}:{node.lineno}"
                    )
            for line_number, line in enumerate(text.splitlines(), start=1):
                if "pams.warp_phase.evaluator" in line:
                    references["literal_module_references"].append(
                        f"{path.relative_to(ROOT).as_posix()}:{line_number}"
                    )
    return references


def main() -> None:
    input_sha256 = {relative: sha256(ROOT / relative) for relative in ENUMERATED_INPUTS}
    receipts = {}
    for gate, relative in (
        ("gate0", "data/warp_phase_pilot_v1/audit/gate0/gate0-local-inputs.receipt.json"),
        ("gate1", "data/warp_phase_pilot_v1/audit/gate1/gate1-v2-unit-v1.receipt.json"),
        ("gate2", "data/warp_phase_pilot_v1/audit/gate2/gate2-stochastic-v2.receipt.json"),
        ("gate3", "data/warp_phase_pilot_v1/audit/gate3/gate3-cpu-sanity.receipt.json"),
    ):
        payload = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        receipts[gate] = {
            "status": payload["status"],
            "authorizes": payload["authorizes"],
            "blockers": payload["blockers"],
        }
    output = {
        "schema": "warp_phase_experiment_audit_trace.v1",
        "reviewer_model": "gpt-5.6-sol",
        "reviewer_reasoning": "ultra",
        "network_or_server_access": False,
        "real_test_sealed_or_heldout_data_access": False,
        "input_sha256": input_sha256,
        "gate_receipts": receipts,
        "gate2_synthetic_recomputation": gate2_fixture_check(),
        "evaluator_in_memory_smoke": evaluator_smoke_check(),
        "evaluator_references_outside_module": evaluator_reference_scan(),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
