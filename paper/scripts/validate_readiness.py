#!/usr/bin/env python3
"""Aggregate every deterministic and semantic paper gate, failing closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from validate_delivery_freeze import source_freeze


ACCEPTABLE = {"PASS"}
AUDITS = {
    "experiment_audit": ("EXPERIMENT_AUDIT.json", False),
    "result_to_claim": ("RESULT_TO_CLAIM.json", False),
    "proof_audit": ("paper/PROOF_AUDIT.json", True),
    "paper_claim_audit": ("paper/PAPER_CLAIM_AUDIT.json", False),
    "citation_audit": ("paper/CITATION_AUDIT.json", False),
    "kill_argument": ("paper/KILL_ARGUMENT.json", False),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_outside(root: Path, path: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return path.is_absolute()
    return False


def paper_source_files(paper: Path) -> list[Path]:
    paths = [
        paper / "main.tex",
        paper / "preamble.tex",
        paper / "math_commands.tex",
        paper / "references.bib",
    ]
    for directory in ("sections", "appendix", "tables", "generated"):
        paths.extend(sorted((paper / directory).glob("*.tex")))
    paths.extend([paper / "figures/task_comparison.pdf", paper / "figures/framework.pdf"])
    return [path for path in paths if path.is_file()]


def verdict_of(data: dict) -> str | None:
    for key in ("verdict", "overall_verdict", "integrity_status", "status", "overall_assurance"):
        value = data.get(key)
        if isinstance(value, str):
            normalized = value.upper().replace("-", "_")
            return {"ACCEPTED": "PASS", "PROVISIONAL": "BLOCKED"}.get(normalized, normalized)
    return None


def parse_time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def run_gate(name: str, command: list[str], report_path: Path | None = None) -> dict:
    try:
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=300,
        )
    except Exception as exc:
        return {"name": name, "status": "ERROR", "detail": type(exc).__name__}
    record: dict[str, object] = {
        "name": name,
        "status": "PASS" if completed.returncode == 0 else ("BLOCKED" if completed.returncode == 2 else "FAIL"),
        "exit_code": completed.returncode,
        "stdout_sha256": hashlib.sha256(completed.stdout.encode("utf-8")).hexdigest(),
        "stderr_sha256": hashlib.sha256(completed.stderr.encode("utf-8")).hexdigest(),
    }
    if report_path and report_path.is_file():
        record["report"] = str(report_path)
        record["report_sha256"] = sha256(report_path)
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            reported = verdict_of(payload)
            record["reported_verdict"] = reported
            if reported in {"FAIL", "ERROR"}:
                record["status"] = "FAIL"
            elif reported in {"BLOCKED", "PROVISIONAL"}:
                record["status"] = "BLOCKED"
            elif reported != "PASS":
                record["status"] = "FAIL"
            evidence_readiness = payload.get("evidence_readiness")
            if isinstance(evidence_readiness, str) and evidence_readiness.upper() != "PASS":
                record["evidence_readiness"] = evidence_readiness
                if evidence_readiness.upper() != "BLOCKED" or record["status"] != "FAIL":
                    record["status"] = "BLOCKED" if evidence_readiness.upper() == "BLOCKED" else "FAIL"
        except Exception:
            record["status"] = "FAIL"
            record["report_parse"] = "FAIL"
    return record


def run_aris_verifier(root: Path, paper: Path, gate_dir: Path, explicit: Path | None) -> dict:
    script = (explicit or (Path.home() / ".aris/repos/auto-research-in-sleep-git/tools/verify_paper_audits.sh")).resolve()
    if not script.is_file():
        return {"name": "aris_audit_verifier_runtime", "status": "BLOCKED", "detail": "pinned verifier script missing"}
    repository = script.parents[1]
    try:
        head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    except Exception as exc:
        return {"name": "aris_audit_verifier_runtime", "status": "FAIL", "detail": f"verifier_git:{type(exc).__name__}"}
    lock_text = (root / "ARIS_LOCK.md").read_text(encoding="utf-8", errors="replace")
    if head not in lock_text:
        return {"name": "aris_audit_verifier_runtime", "status": "FAIL", "detail": "verifier HEAD differs from ARIS_LOCK", "verifier_commit": head}
    git_bash = Path("C:/Program Files/Git/bin/bash.exe")
    bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")
    if not bash:
        return {"name": "aris_audit_verifier_runtime", "status": "BLOCKED", "detail": "bash unavailable", "verifier_commit": head}
    report = gate_dir / "aris-audit-verifier-runtime.json"
    record = run_gate(
        "aris_audit_verifier_runtime",
        [bash, str(script), str(paper), "--assurance", "submission", "--json-out", str(report)],
        report,
    )
    record["verifier_commit"] = head
    record["verifier_sha256"] = sha256(script)
    return record


def audit_check(root: Path, name: str, rel: str, allow_na: bool, freeze_digest: str, receipt_time: datetime | None) -> dict:
    path = root / rel
    if not path.is_file():
        return {"name": name, "path": rel, "status": "BLOCKED", "detail": "missing"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"name": name, "path": rel, "status": "FAIL", "detail": f"parse:{type(exc).__name__}"}
    verdict = verdict_of(data)
    reason = str(data.get("reason_code", data.get("reason", ""))).casefold()
    na_ok = allow_na and verdict == "NOT_APPLICABLE" and reason in {"no_theorems", "no_theorem", "no theorems"}
    status = "PASS" if verdict in ACCEPTABLE or na_ok else "FAIL"
    bound_freeze = data.get("paper_freeze", data.get("source_freeze_sha256", data.get("evidence_freeze")))
    if bound_freeze != freeze_digest:
        status = "FAIL"
    generated_at = parse_time(data.get("generated_at"))
    fresh = generated_at is not None and (receipt_time is None or generated_at <= receipt_time)
    if not fresh:
        status = "FAIL"
    return {
        "name": name,
        "path": rel,
        "sha256": sha256(path),
        "status": status,
        "verdict": verdict,
        "paper_freeze": bound_freeze,
        "same_source_freeze": bound_freeze == freeze_digest,
        "generated_at": data.get("generated_at"),
        "timestamp_valid_and_not_after_receipt": fresh,
        "allowed_not_applicable": na_ok,
    }


def audit_verifier_artifact_check(root: Path, audit_records: list[dict]) -> dict:
    rel = "paper/.aris/audit-verifier-report.json"
    path = root / rel
    if not path.is_file():
        return {"name": "audit_verifier", "path": rel, "status": "BLOCKED", "detail": "missing"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"name": "audit_verifier", "path": rel, "status": "FAIL", "detail": f"parse:{type(exc).__name__}"}
    rows = data.get("audits", [])
    semantic_same_freeze = all(record.get("status") == "PASS" for record in audit_records if record.get("name") in {"proof_audit", "paper_claim_audit", "citation_audit", "kill_argument"})
    checks = {
        "submission_assurance": data.get("assurance") == "submission",
        "accepted": verdict_of(data) == "PASS",
        "not_submission_blocking": data.get("submission_blocking") is False,
        "four_rows": len(rows) == 4,
        "rows_nonblocking": len(rows) == 4 and all(row.get("status") == "OK" and row.get("verdict") in {"PASS", "NOT_APPLICABLE"} for row in rows),
        "semantic_audits_same_freeze": semantic_same_freeze,
        "timestamp_valid": parse_time(data.get("generated_at")) is not None,
    }
    return {"name": "audit_verifier", "path": rel, "sha256": sha256(path), "status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def build_check(root: Path, build_report: Path, pdf: Path, freeze_digest: str) -> dict:
    if not build_report.is_file() or not pdf.is_file():
        return {"name": "build_page_font", "status": "BLOCKED", "detail": "build report or PDF missing"}
    try:
        data = json.loads(build_report.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"name": "build_page_font", "status": "FAIL", "detail": f"parse:{type(exc).__name__}"}
    expected_paths = paper_source_files(root / "paper")
    expected_records = [
        {"path": path.relative_to(root / "paper").as_posix(), "sha256": sha256(path)}
        for path in expected_paths
    ]
    reported_records = data.get("source_files")
    exact_records = isinstance(reported_records, list) and reported_records == expected_records
    bundle_payload = "\n".join(
        f"{item['path']}\t{item['sha256']}" for item in expected_records
    ).encode("utf-8")
    expected_bundle = hashlib.sha256(bundle_payload).hexdigest()
    checks = data.get("checks", {})
    blocking = {
        "reported_pass": verdict_of(data) == "PASS",
        "pdf_digest": data.get("pdf", {}).get("sha256") == sha256(pdf),
        "source_exact_coverage_and_digests": exact_records and bool(expected_records),
        "source_bundle_digest": data.get("source_bundle_sha256") == expected_bundle,
        "source_freeze_binding": data.get("source_freeze_sha256") == freeze_digest,
        "page_limit": checks.get("conclusion_within_eight_pages") is True,
        "letter_page_size": checks.get("letter_page_size") is True,
        "zero_log_warnings": checks.get("final_log_warning_count") == 0,
        "zero_unembedded_fonts": checks.get("unembedded_font_count") == 0,
        "zero_duplicate_labels": checks.get("duplicate_labels") == [],
        "zero_undefined_citations": checks.get("undefined_citation_keys") == [],
        "no_unresolved_question_marks": checks.get("unresolved_question_marks") is False,
        "anonymous_author_visible": checks.get("anonymous_author_visible") is True,
    }
    return {
        "name": "build_page_font",
        "status": "PASS" if all(blocking.values()) else "FAIL",
        "path": str(build_report),
        "sha256": sha256(build_report),
        "checks": blocking,
        "expected_source_file_count": len(expected_records),
        "reported_source_file_count": len(reported_records) if isinstance(reported_records, list) else None,
        "source_freeze_sha256": data.get("source_freeze_sha256"),
    }


def template_check(root: Path) -> dict:
    path = root / "paper/TEMPLATE_PROVENANCE.md"
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    official = bool(re.search(r"official\s+CVPR\s*2027", text, re.I))
    final = not re.search(r"pending|provisional|temporary|not yet available", text, re.I)
    tag_match = re.search(r"^-\s*Tag:\s*`([^`]+)`\s*$", text, re.I | re.M)
    commit_match = re.search(r"^-\s*Commit:\s*`([0-9a-f]{40})`\s*$", text, re.I | re.M)
    rows = re.findall(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|\s*$", text, re.I | re.M)
    row_paths = [item[0] for item in rows]
    digest_errors: list[str] = []
    for rel, digest in rows:
        candidate = root / "paper" / rel
        if not candidate.is_file() or sha256(candidate) != digest.lower():
            digest_errors.append(rel)
    official_root = root / "paper/vendor/cvpr2027"
    vendored = sorted(
        item.relative_to(root / "paper").as_posix()
        for item in official_root.rglob("*")
        if item.is_file()
    ) if official_root.is_dir() else []
    exact_vendor_coverage = bool(vendored) and sorted(row_paths) == vendored
    main_text = (root / "paper/main.tex").read_text(encoding="utf-8", errors="replace")
    bibliography_uses_2027 = bool(re.search(r"\\bibliographystyle\{vendor/cvpr2027/", main_text))
    style_uses_review = bool(re.search(r"\\usepackage\[review\]\{cvpr\}", main_text))
    checks = {
        "official_cvpr2027": official,
        "not_pending_or_provisional": final,
        "immutable_tag": bool(tag_match),
        "immutable_commit": bool(commit_match),
        "digest_rows_present": bool(rows),
        "vendored_digests_exact": not digest_errors and bool(rows),
        "official_vendor_tree_exact_coverage": exact_vendor_coverage,
        "main_uses_review_style": style_uses_review,
        "bibliography_uses_cvpr2027_vendor": bibliography_uses_2027,
    }
    return {
        "name": "official_cvpr2027_template",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "path": "paper/TEMPLATE_PROVENANCE.md",
        "sha256": sha256(path) if path.is_file() else None,
        "tag": tag_match.group(1) if tag_match else None,
        "commit": commit_match.group(1) if commit_match else None,
        "digest_errors": digest_errors,
        "checks": checks,
    }


def reviewer_health_check(root: Path, freeze_digest: str, receipt_time: datetime | None) -> dict:
    root = root.resolve()
    path = root / "paper/.aris/reviewer-health.json"
    if not path.is_file():
        return {"name": "cross_family_reviewer_health", "status": "BLOCKED", "detail": "missing reviewer-health.json"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"name": "cross_family_reviewer_health", "status": "FAIL", "detail": f"parse:{type(exc).__name__}"}
    checked_at = parse_time(data.get("checked_at"))
    provider = " ".join(str(data.get(key, "")) for key in ("provider", "model_family", "reviewer_family")).casefold()

    def bound_artifact(name: str) -> tuple[bool, dict[str, object]]:
        nested = data.get(name)
        if isinstance(nested, dict):
            rel, digest = nested.get("path"), nested.get("sha256")
        else:
            rel, digest = data.get(f"{name}_path"), data.get(f"{name}_sha256")
        detail: dict[str, object] = {"path": rel, "declared_sha256": digest}
        if not isinstance(rel, str) or not isinstance(digest, str):
            return False, detail
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return False, detail
        ok = candidate.is_file() and candidate.stat().st_size > 0 and digest.removeprefix("sha256:") == sha256(candidate)
        if candidate.is_file():
            detail["actual_sha256"] = sha256(candidate)
            detail["bytes"] = candidate.stat().st_size
        return ok, detail

    trace_ok, trace_detail = bound_artifact("trace")
    probe_ok, probe_detail = bound_artifact("probe")
    checks = {
        "verdict_pass": verdict_of(data) == "PASS",
        "claude_family": "claude" in provider or "anthropic" in provider,
        "same_source_freeze": data.get("paper_freeze", data.get("source_freeze_sha256")) == freeze_digest,
        "timestamp_valid": checked_at is not None and (receipt_time is None or checked_at <= receipt_time),
        "trace_digest_bound": trace_ok,
        "probe_digest_bound": probe_ok,
        "probe_success_explicit": data.get("probe_status") == "PASS" or (
            isinstance(data.get("probe"), dict) and data["probe"].get("status") == "PASS"
        ),
    }
    return {
        "name": "cross_family_reviewer_health",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "sha256": sha256(path),
        "checks": checks,
        "trace": trace_detail,
        "probe": probe_detail,
    }


def review_trace_check(root: Path, name: str, candidates: list[Path], freeze_digest: str, receipt_time: datetime | None) -> dict:
    trace = next((path for path in candidates if path.is_file() and path.stat().st_size > 0), None)
    if trace is None:
        return {"name": name, "status": "BLOCKED", "detail": "reviewer trace missing"}
    sidecar = next((path for path in (trace.with_name("review.json"), trace.with_name("routing.json")) if path.is_file()), None)
    if sidecar is None:
        return {
            "name": name,
            "status": "BLOCKED",
            "path": trace.relative_to(root).as_posix(),
            "sha256": sha256(trace),
            "detail": "same-freeze review sidecar missing",
        }
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"name": name, "status": "FAIL", "detail": f"sidecar_parse:{type(exc).__name__}"}
    generated_at = parse_time(data.get("generated_at", data.get("checked_at")))
    checks = {
        "trace_digest_bound": data.get("review_sha256", data.get("artifact_sha256")) == sha256(trace),
        "same_source_freeze": data.get("paper_freeze", data.get("source_freeze_sha256")) == freeze_digest,
        "timestamp_valid": generated_at is not None and (receipt_time is None or generated_at <= receipt_time),
        "resolution_pass": verdict_of(data) == "PASS",
    }
    return {
        "name": name,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "path": trace.relative_to(root).as_posix(),
        "sha256": sha256(trace),
        "sidecar": sidecar.relative_to(root).as_posix(),
        "sidecar_sha256": sha256(sidecar),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("draft", "submission"), default="submission")
    parser.add_argument("--secret-file", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--pdf-text", type=Path)
    parser.add_argument("--result-manifest", type=Path)
    parser.add_argument("--result-receipt", type=Path, help="truly external result-bundle receipt used by evidence_precheck")
    parser.add_argument("--build-report", type=Path)
    parser.add_argument("--aris-verifier", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    paper = root / "paper"
    scripts = paper / "scripts"
    freeze = json.loads((paper / "evidence/freeze_inventory.json").read_text(encoding="utf-8"))
    if args.output:
        output = args.output.resolve()
    elif args.mode == "submission":
        output = Path(freeze["external_delivery_receipt"]).with_name("rac-cvpr27-readiness-report.json")
    else:
        output = paper / ".aris/readiness-report.json"
    try:
        output.relative_to(root)
        output_inside_repository = True
    except ValueError:
        output_inside_repository = False
    if args.mode == "submission" and output_inside_repository:
        print("READINESS=BLOCKED; submission-mode output must be outside the repository so final validation preserves the clean delivery tree")
        return 2
    gate_dir = (output.parent / "readiness-gates") if args.mode == "submission" else paper / ".aris/readiness-gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    pdf = (args.pdf or paper / "main.pdf").resolve()
    pdf_text = (args.pdf_text or paper / "rendered/main.txt").resolve()
    result_manifest = (args.result_manifest or paper / "evidence/result_manifest.json").resolve()
    result_receipt = (
        args.result_receipt.resolve()
        if args.result_receipt
        else Path(freeze["external_delivery_receipt"]).with_name("rac-cvpr27-results-delivery-receipt.json").resolve()
    )
    build_report = (args.build_report or paper / ".aris/build-qa.json").resolve()

    freeze_digest, _ = source_freeze(root, freeze)
    receipt_path = Path(freeze["external_delivery_receipt"])
    receipt: dict = {}
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except Exception:
            receipt = {}
    receipt_time = parse_time(receipt.get("created_at"))

    deterministic: list[dict] = []
    delivery_output = gate_dir / "delivery-freeze.json"
    deterministic.append(run_gate("delivery_freeze", [sys.executable, str(scripts / "validate_delivery_freeze.py"), "--root", str(root), "--mode", args.mode, "--output", str(delivery_output)], delivery_output))
    story_output = gate_dir / "story.json"
    deterministic.append(run_gate("story", [sys.executable, str(scripts / "validate_story.py"), "--paper-dir", str(paper), "--pdf", str(pdf), "--pdf-text", str(pdf_text), "--output", str(story_output)], story_output))
    claim_output = gate_dir / "claims.json"
    claim_command = [sys.executable, str(scripts / "claim_scan.py"), "--paper-dir", str(paper), "--mode", "submission", "--pdf", str(pdf), "--pdf-text", str(pdf_text), "--output", str(claim_output)]
    if result_manifest.is_file():
        claim_command.extend(["--result-manifest", str(result_manifest)])
    deterministic.append(run_gate("claim_scan", claim_command, claim_output))
    figure_output = gate_dir / "figures.json"
    deterministic.append(run_gate("figures", [sys.executable, str(scripts / "validate_figures.py"), "--paper-dir", str(paper), "--mode", "submission", "--output", str(figure_output)], figure_output))
    method_output = gate_dir / "method.json"
    deterministic.append(run_gate("method_contract", [sys.executable, str(scripts / "validate_method_contract.py"), "--paper-dir", str(paper), "--output", str(method_output)], method_output))
    result_command = [sys.executable, str(scripts / "validate_result_manifest.py"), str(result_manifest), "--schema", str(paper / "evidence/result_manifest.schema.json"), "--bundle-root", str(root)]
    deterministic.append(run_gate("result_manifest", result_command))
    evidence_output = gate_dir / "evidence-precheck.json"
    if result_manifest.is_file() and path_outside(root, result_receipt) and result_receipt.is_file():
        evidence_command = [
            sys.executable,
            str(scripts / "evidence_precheck.py"),
            str(result_manifest),
            "--bundle-root",
            str(root),
            "--external-receipt",
            str(result_receipt),
            "--output",
            str(evidence_output),
        ]
        deterministic.append(run_gate("evidence_precheck", evidence_command, evidence_output))
        generated_evidence = paper / "generated/evidence_values.tex"
        generation_record = run_gate(
            "generate_evidence_tex_exact",
            [
                sys.executable,
                str(scripts / "generate_evidence_tex.py"),
                str(result_manifest),
                str(generated_evidence),
                "--bundle-root",
                str(root),
                "--external-receipt",
                str(result_receipt),
            ],
        )
        generation_record["generator_sha256"] = sha256(scripts / "generate_evidence_tex.py")
        generation_record["generated_path"] = "paper/generated/evidence_values.tex"
        generation_record["generated_sha256"] = sha256(generated_evidence) if generated_evidence.is_file() else None
        generation_record["result_manifest_sha256"] = sha256(result_manifest)
        generation_record["external_receipt_sha256"] = sha256(result_receipt)
        deterministic.append(generation_record)
    else:
        missing_detail = {
            "manifest_present": result_manifest.is_file(),
            "external_receipt_present": result_receipt.is_file(),
            "external_receipt_outside_repository": path_outside(root, result_receipt),
            "external_receipt_path": str(result_receipt),
        }
        deterministic.append({"name": "evidence_precheck", "status": "BLOCKED", "detail": missing_detail})
        deterministic.append({"name": "generate_evidence_tex_exact", "status": "BLOCKED", "detail": missing_detail})
    anonymity_output = gate_dir / "anonymity.json"
    deterministic.append(run_gate("anonymity", [sys.executable, str(scripts / "anonymity_scan.py"), "--root", str(root), "--secret-file", str(args.secret_file.resolve()), "--pdf", str(pdf), "--pdf-text", str(pdf_text), "--output", str(anonymity_output)], anonymity_output))
    deterministic.append(run_gate("submission_source", [sys.executable, str(scripts / "validate_submission.py"), "--paper-dir", str(paper)]))
    deterministic.append(run_aris_verifier(root, paper, gate_dir, args.aris_verifier.resolve() if args.aris_verifier else None))

    semantic = [audit_check(root, name, rel, allow_na, freeze_digest, receipt_time) for name, (rel, allow_na) in AUDITS.items()]
    semantic.append(audit_verifier_artifact_check(root, semantic))
    for round_name in ("round1", "round2"):
        semantic.append(review_trace_check(
            root,
            f"paper_review_{round_name}",
            [
                paper / f".aris/traces/paper-improvement/{round_name}/reviewer.md",
                root / f".aris/traces/paper-improvement/{round_name}/reviewer.md",
            ],
            freeze_digest,
            receipt_time,
        ))

    build = build_check(root, build_report, pdf, freeze_digest)
    template = template_check(root)
    reviewer = reviewer_health_check(root, freeze_digest, receipt_time)
    contract_text = (root / "PAPER_ACCEPTANCE_CONTRACT.md").read_text(encoding="utf-8", errors="replace")
    contract_accepted = bool(re.search(r"^status:\s*accepted\s*$", contract_text, re.M | re.I))
    contract_review = review_trace_check(
        root,
        "acceptance_contract_review",
        [
            paper / ".aris/traces/acceptance-contract/round3/reviewer.md",
            root / ".aris/traces/acceptance-contract/round3/reviewer.md",
            paper / ".aris/traces/acceptance-contract/round2/reviewer.md",
            root / ".aris/traces/acceptance-contract/round2/reviewer.md",
        ],
        freeze_digest,
        receipt_time,
    )
    contract = {
        "name": "accepted_contract",
        "status": "PASS" if contract_accepted and contract_review["status"] == "PASS" else "BLOCKED",
        "sha256": sha256(root / "PAPER_ACCEPTANCE_CONTRACT.md"),
        "review_trace": contract_review,
    }

    all_records = deterministic + semantic + [build, template, reviewer, contract]
    statuses = {record["status"] for record in all_records}
    verdict = "FAIL" if ("FAIL" in statuses or "ERROR" in statuses) else ("BLOCKED" if "BLOCKED" in statuses else "PASS")
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "mode": args.mode,
        "verdict": verdict,
        "submission_ready": verdict == "PASS",
        "source_freeze_sha256": freeze_digest,
        "delivery_receipt": {
            "path": freeze["external_delivery_receipt"],
            "present": receipt_path.is_file(),
            "sha256": sha256(receipt_path) if receipt_path.is_file() else None,
        },
        "result_delivery_receipt": {
            "path": str(result_receipt),
            "outside_repository": path_outside(root, result_receipt),
            "present": result_receipt.is_file(),
            "sha256": sha256(result_receipt) if result_receipt.is_file() else None,
        },
        "deterministic_gates": deterministic,
        "semantic_audits": semantic,
        "package_checks": [build, template, reviewer, contract],
        "failure_policy": "Any missing, stale, cross-freeze, BLOCKED, FAIL, ERROR, provisional template, unhealthy reviewer, or unacceptable verdict prevents submission readiness.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"READINESS={verdict}; ready={report['submission_ready']}; gates={len(all_records)}")
    return 0 if verdict == "PASS" else (2 if verdict == "BLOCKED" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
