#!/usr/bin/env python3
"""Validate implementation provenance without upgrading proposed modules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evidence_common import COMMIT_RE, add_check, emit_report, finish_report, hash_matches, json_load, resolve_inside

REQUIRED_COMPONENTS = {
    "shared_pose_encoder",
    "per_identity_state",
    "local_masked_period_estimator",
    "soft_tempo_router",
    "shared_tempo_experts",
    "response_fusion",
    "normalized_overlap_add",
    "single_decode_per_track",
}
REQUIRED_SCOPES = {
    "data_loaders",
    "pseudo_labels_and_router_targets",
    "auxiliary_losses",
    "hyperparameter_tuning",
    "checkpoint_selection",
    "early_stopping",
    "calibration",
    "inference",
    "pose_estimation",
    "detection",
    "tracking",
    "evaluation",
}


def _anchor_ok(root: Path, anchor: Any) -> bool:
    if not isinstance(anchor, dict) or set(anchor) - {"path", "line", "sha256"}:
        return False
    if not isinstance(anchor.get("line"), int) or anchor["line"] < 1:
        return False
    try:
        path = resolve_inside(root, anchor.get("path"))
    except (TypeError, ValueError):
        return False
    if not path.is_file():
        return False
    expected = anchor.get("sha256")
    return expected is None or hash_matches(path, expected)


def _tests_ok(root: Path, tests: Any) -> bool:
    if not isinstance(tests, list) or not tests:
        return False
    for item in tests:
        if not isinstance(item, dict) or item.get("status") != "PASS":
            return False
        try:
            path = resolve_inside(root, item.get("path"))
        except (TypeError, ValueError):
            return False
        if not path.is_file() or not hash_matches(path, item.get("sha256")):
            return False
    return True


def validate(paper_dir: Path, mode: str) -> tuple[dict[str, Any], int]:
    root = paper_dir.parent.resolve()
    path = paper_dir / "evidence/method_manifest.yaml"
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        data = json_load(path)
    except Exception as exc:  # fail closed on parser/file errors
        add_check(checks, "manifest_parse", False, type(exc).__name__)
        return finish_report(gate="method_manifest", mode=mode, checks=checks, structural_failure=True)

    add_check(checks, "schema_version", data.get("schema_version") == "1.0", "expected schema 1.0")
    add_check(checks, "document_type", data.get("document_type") == "icassp_method_manifest", "expected ICASSP method manifest")
    truth = data.get("tracked_implementation_truth", {})
    truth_ok = (
        truth.get("classification") == "single_person_dominant_pose_counter"
        and truth.get("input_contract") == "PoseSequence[T,33,3]"
        and truth.get("output_contract") == "one scalar CountResult"
        and truth.get("multi_person_identity_state") is False
        and truth.get("learned_soft_tempo_router") is False
        and truth.get("normalized_overlap_add") is False
    )
    add_check(checks, "tracked_truth_not_inflated", truth_ok, "tracked checkout remains classified as single-person/scalar")

    proposed = data.get("proposed_submission_method", {})
    components = proposed.get("components", [])
    component_ids = {item.get("id") for item in components if isinstance(item, dict)}
    add_check(checks, "component_inventory", component_ids == REQUIRED_COMPONENTS, "required proposed component inventory is exact")
    firewall = data.get("supervision_firewall", {})
    scopes = firewall.get("scopes", {})
    add_check(checks, "supervision_scope_inventory", set(scopes) == REQUIRED_SCOPES, "all supervision scopes are enumerated")

    sync_tokens = json.dumps(data, ensure_ascii=False).count("SYNC-REQUIRED")
    synchronized = data.get("status") == "SYNCHRONIZED" and proposed.get("status") == "SYNCHRONIZED" and sync_tokens == 0
    if not synchronized:
        blockers.append("method manifest contains unsynchronized proposed components")

    commit = proposed.get("implementation_commit")
    if synchronized and (not isinstance(commit, str) or not COMMIT_RE.fullmatch(commit)):
        add_check(checks, "implementation_commit", False, "synchronized manifest requires a 40-hex commit")
    else:
        add_check(checks, "implementation_commit", synchronized and bool(COMMIT_RE.fullmatch(commit or "")), "collaborator commit is bound" if synchronized else "not yet bound", blocking=False)

    if synchronized:
        task = proposed.get("task_contract", {})
        add_check(checks, "task_contract_anchor", task.get("status") == "IMPLEMENTED" and _anchor_ok(root, task.get("source_anchor")) and _tests_ok(root, task.get("tests")), "task contract has byte-bound source/test evidence")
        all_components = all(
            item.get("status") == "IMPLEMENTED"
            and _anchor_ok(root, item.get("source_anchor"))
            and _tests_ok(root, item.get("tests"))
            for item in components
        )
        add_check(checks, "component_source_tests", all_components, "every component has a byte-bound source anchor and passing tests")
        objective = proposed.get("training_objective", {})
        add_check(
            checks,
            "training_objective_bound",
            objective.get("status") == "IMPLEMENTED"
            and bool(objective.get("declared_terms"))
            and isinstance(objective.get("weights"), dict)
            and objective.get("gradient_paths") is not None
            and _anchor_ok(root, objective.get("source_anchor"))
            and _tests_ok(root, objective.get("tests")),
            "training objective, weights, gradients, source, and tests are bound",
        )
        fallback = proposed.get("fallback_behavior", {})
        add_check(
            checks,
            "fallback_bound",
            fallback.get("status") in {"IMPLEMENTED", "NOT_APPLICABLE"}
            and bool(fallback.get("definition"))
            and _anchor_ok(root, fallback.get("source_anchor"))
            and _tests_ok(root, fallback.get("tests")),
            "fallback behavior is implemented or explicitly not applicable with evidence",
        )
        add_check(checks, "supervision_scopes_complete", firewall.get("status") == "COMPLETE" and all(value == "COMPLETE" for value in scopes.values()), "all supervision scopes are complete")
        sources = firewall.get("external_supervision_sources", [])
        source_components = {item.get("component") for item in sources if isinstance(item, dict)}
        complete_sources = bool(sources) and {"pose_estimation", "tracking"}.issubset(source_components) and all(
            all(item.get(field) for field in ("component", "model", "training_data", "checkpoint", "license"))
            for item in sources if isinstance(item, dict)
        )
        add_check(checks, "external_supervision_disclosed", complete_sources, "pose/tracker supervision and checkpoint provenance are disclosed")
        required = data.get("required_submission_evidence", {})
        add_check(checks, "submission_evidence_flags", all(required.get(key) for key in ("collaborator_commit", "source_anchors_complete", "tests_complete", "training_configuration", "supervision_disclosure_complete", "author_verified_sync_report")), "all method evidence flags are bound")

    structural = any(item["status"] == "FAIL" for item in checks)
    return finish_report(gate="method_manifest", mode=mode, checks=checks, blockers=blockers, structural_failure=structural)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=("draft", "submission"), default="draft")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report, code = validate(args.paper_dir.resolve(), args.mode)
    emit_report(report, args.output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
