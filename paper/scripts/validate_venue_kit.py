#!/usr/bin/env python3
"""Validate the temporary scaffold and fail closed on a missing 2027 kit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evidence_common import add_check, emit_report, finish_report, hash_matches, resolve_inside

OFFICIAL_URLS = (
    "https://2027.ieeeicassp.org/call-for-papers/",
    "https://2027.ieeeicassp.org/publishing-and-paper-presentation-options/",
    "https://2027.ieeeicassp.org/about/editorial-policies/",
)
LIVE_PAGE_RULE = "optional fifth page containing **only references**"
SCAFFOLD_HASHES = {
    "paper/vendor/icassp2026/spconf.sty": "sha256:dc5d632639040cb183f2ab62780f314845aa021be73048c8a9ec9c2072d64a86",
    "paper/vendor/icassp2026/IEEEbib.bst": "sha256:7e6ca0c8b72158d504a12bb091f817c07032a021ba41ca783cca4c2dd80d570b",
    "paper/vendor/icassp2026/upstream-Template.tex": "sha256:7a17173e83714540733cc53dad0c25f10bc6a498d7b66a7f17e37eede0a79314",
}


def verified_kit(paper_dir: Path) -> tuple[bool, str, list[str]]:
    root = paper_dir.parent.resolve()
    path = paper_dir / "vendor/icassp2027/kit_manifest.json"
    if not path.is_file():
        return False, "references-only", ["official 2027 kit manifest is missing"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, "references-only", [type(exc).__name__]
    errors: list[str] = []
    if data.get("schema_version") != "1.0" or data.get("venue_state") != "ICASSP2027-kit-verified":
        errors.append("kit manifest state is not verified")
    if data.get("official_source_url") not in OFFICIAL_URLS:
        errors.append("kit source is not an approved official 2027 URL")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        errors.append("kit file inventory is empty")
    else:
        for item in files:
            try:
                file_path = resolve_inside(root, item.get("path"))
            except (TypeError, ValueError):
                errors.append("kit path is invalid")
                continue
            if not file_path.is_file() or not hash_matches(file_path, item.get("sha256")):
                errors.append("kit file hash mismatch")
    policy = data.get("page5_policy", "references-only")
    if policy not in {"references-only", "references-funding-ethics"}:
        errors.append("unknown page 5 policy")
        policy = "references-only"
    if policy != "references-only" and data.get("page5_policy_source_url") not in OFFICIAL_URLS:
        errors.append("relaxed page 5 policy lacks a verified official source")
        policy = "references-only"
    return not errors, policy, errors


def validate(paper_dir: Path, mode: str) -> tuple[dict[str, Any], int]:
    root = paper_dir.parent.resolve()
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    provenance_path = paper_dir / "TEMPLATE_PROVENANCE.md"
    compliance_path = paper_dir / "VENUE_COMPLIANCE.md"
    try:
        provenance = provenance_path.read_text(encoding="utf-8")
        compliance = compliance_path.read_text(encoding="utf-8")
    except Exception as exc:
        add_check(checks, "venue_provenance_parse", False, type(exc).__name__)
        return finish_report(gate="venue_kit", mode=mode, checks=checks, structural_failure=True)
    official_urls_present = all(url in provenance or url in compliance for url in OFFICIAL_URLS)
    add_check(checks, "official_urls", official_urls_present, "official ICASSP 2027 sources recorded")
    live_rule_present = LIVE_PAGE_RULE in provenance and "references-only" in compliance
    add_check(checks, "live_page_rule", live_rule_present, "live 2027 page 5 references-only rule recorded")
    scaffold_ok = True
    for declared, expected in SCAFFOLD_HASHES.items():
        path = resolve_inside(root, declared)
        scaffold_ok = scaffold_ok and path.is_file() and hash_matches(path, expected)
    add_check(checks, "draft_scaffold_hashes", scaffold_ok, "official 2026 scaffold files match recorded hashes")

    kit_ok, policy, kit_errors = verified_kit(paper_dir)
    provenance_verified = "Venue state: `ICASSP2027-kit-verified`" in provenance
    if not (kit_ok and provenance_verified):
        blockers.append("official ICASSP 2027 kit is not installed and byte-verified")
    add_check(checks, "official_2027_kit", kit_ok and provenance_verified, "; ".join(kit_errors) if kit_errors else "verified kit and provenance agree", blocking=False)
    report, code = finish_report(
        gate="venue_kit",
        mode=mode,
        checks=checks,
        blockers=blockers,
        structural_failure=any(item["status"] == "FAIL" for item in checks),
    )
    report["official_sources"] = list(OFFICIAL_URLS)
    report["effective_page5_policy"] = policy if kit_ok else "references-only"
    return report, code


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
