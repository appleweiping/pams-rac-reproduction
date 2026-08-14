#!/usr/bin/env python3
"""Aggregate all deterministic ICASSP draft/submission gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from claim_scan import validate as validate_prose
from evidence_common import emit_report
from generate_evidence_tex import run as validate_generated
from validate_author_metadata import validate as validate_authors
from validate_method_manifest import validate as validate_method
from validate_pdf_4plus1 import validate as validate_pdf
from validate_results_manifest import validate_manifest as validate_results
from validate_venue_kit import validate as validate_venue


def _status(data: dict[str, Any]) -> str:
    return str(data.get("status") or data.get("verdict") or data.get("overall_verdict") or "").upper()


def _assurance_gate(paper_dir: Path, mode: str) -> tuple[dict[str, Any], int]:
    blockers: list[str] = []
    assurance_path = paper_dir / ".aris/assurance.txt"
    assurance = assurance_path.read_text(encoding="utf-8").strip() if assurance_path.is_file() else ""
    if assurance != "submission":
        blockers.append("ARIS assurance is not submission")
    health_path = paper_dir / ".aris/reviewer-health.json"
    health: dict[str, Any] = {}
    if health_path.is_file():
        try:
            health = json.loads(health_path.read_text(encoding="utf-8"))
        except Exception:
            health = {}
    cross_family = health.get("cross_family_review_obtained") is True and _status(health) in {"PASS", "HEALTHY"}
    if not cross_family:
        blockers.append("healthy substantive cross-family review is absent")

    audit_specs = {
        "experiment_audit": (paper_dir / "EXPERIMENT_AUDIT.json", {"PASS"}),
        "paper_claim_audit": (paper_dir / "PAPER_CLAIM_AUDIT.json", {"PASS"}),
        "citation_audit": (paper_dir / "CITATION_AUDIT.json", {"PASS"}),
        "kill_argument": (paper_dir / "KILL_ARGUMENT.json", {"PASS"}),
        "proof_audit": (paper_dir / "PROOF_AUDIT.json", {"PASS", "NOT_APPLICABLE"}),
    }
    audit_states: dict[str, str] = {}
    for name, (path, permitted) in audit_specs.items():
        state = "MISSING"
        reason = ""
        if path.is_file():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                state = _status(payload)
                reason = str(payload.get("reason") or payload.get("summary") or "")
            except Exception:
                state = "INVALID"
        if state not in permitted or (state == "NOT_APPLICABLE" and not reason.strip()):
            blockers.append(f"{name} is missing, invalid, or blocking")
        audit_states[name] = state

    verifier_candidates = (paper_dir / ".aris/final-verifier.json", paper_dir / "SUBMISSION_GATE_REPORT.json")
    verifier_ok = False
    for path in verifier_candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        verifier_ok = _status(payload) == "PASS" and payload.get("exit_code", 0) == 0
        if verifier_ok:
            break
    if not verifier_ok:
        blockers.append("final ARIS verifier PASS/exit 0 is absent")
    ready = not blockers
    report = {
        "schema_version": "1.0",
        "gate": "aris_assurance",
        "mode": mode,
        "status": "PASS" if ready else ("PROVISIONAL" if mode == "draft" else "FAIL"),
        "submission_ready": ready and mode == "submission",
        "checks": {
            "assurance_submission": assurance == "submission",
            "cross_family_review": cross_family,
            "audits": audit_states,
            "final_verifier": verifier_ok,
        },
        "blockers": sorted(blockers),
    }
    return report, 0 if ready or mode == "draft" else 1


def validate(paper_dir: Path, mode: str, pdf: Path | None) -> tuple[dict[str, Any], int]:
    root = paper_dir.parent.resolve()
    manifest = paper_dir / "evidence/results_manifest.json"
    generated = paper_dir / "generated/evidence_values.tex"
    gates: dict[str, dict[str, Any]] = {}
    codes: dict[str, int] = {}

    gates["method"], codes["method"] = validate_method(paper_dir, mode)
    gates["results"], codes["results"] = validate_results(manifest, root, mode)
    gates["generated"], codes["generated"] = validate_generated(manifest, generated, root, mode, verify_only=True)
    gates["prose"], codes["prose"] = validate_prose(paper_dir, mode, None, None)
    gates["authors"], codes["authors"] = validate_authors(paper_dir, mode)
    gates["venue"], codes["venue"] = validate_venue(paper_dir, mode)
    gates["assurance"], codes["assurance"] = _assurance_gate(paper_dir, mode)
    if pdf is not None:
        gates["pdf"], codes["pdf"] = validate_pdf(pdf, paper_dir, mode)
    elif mode == "submission":
        gates["pdf"] = {"gate": "pdf_4plus1", "mode": mode, "status": "FAIL", "blockers": ["submission PDF argument is required"]}
        codes["pdf"] = 1
    else:
        gates["pdf"] = {"gate": "pdf_4plus1", "mode": mode, "status": "SKIPPED", "blockers": ["draft PDF not supplied to aggregator"]}
        codes["pdf"] = 0

    failed = sorted(name for name, code in codes.items() if code != 0)
    blockers = sorted(
        {
            str(blocker)
            for report in gates.values()
            for blocker in report.get("blockers", [])
            if str(blocker).strip()
        }
    )
    if mode == "draft":
        structural_failures = sorted(name for name in failed if gates[name].get("status") == "FAIL")
        status = "FAIL" if structural_failures else ("PROVISIONAL" if blockers else "PASS")
        code = 1 if structural_failures else 0
    else:
        status = "PASS" if not failed and all(report.get("status") == "PASS" for report in gates.values()) else "FAIL"
        code = 0 if status == "PASS" else 1
    return {
        "schema_version": "1.0",
        "gate": "icassp_submission_readiness",
        "mode": mode,
        "status": status,
        "submission_ready": mode == "submission" and status == "PASS",
        "failed_gates": failed,
        "blockers": blockers,
        "gates": gates,
    }, code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=("draft", "submission"), default="draft")
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--result-manifest", type=Path, help="deprecated; the fixed ICASSP path is used")
    parser.add_argument("--external-receipt", type=Path, help="deprecated; hashes are embedded in the fixed manifest")
    args = parser.parse_args()
    report, code = validate(args.paper_dir.resolve(), args.mode, args.pdf.resolve() if args.pdf else None)
    emit_report(report, args.output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
