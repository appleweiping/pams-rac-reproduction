#!/usr/bin/env python3
"""Check private author completion without emitting private values."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from evidence_common import add_check, emit_report, finish_report

REQUIRED_DEFINITIONS = (
    "RACAuthorMetadataComplete",
    "RACAuthorNames",
    "RACAuthorAffiliations",
    "RACAuthorEmails",
)
REQUIRED_ATTESTATIONS = (
    "roster_matches_submission_system",
    "every_author_approved_roster_order",
    "every_author_approved_final_manuscript",
    "originality_attested",
    "no_simultaneous_submission",
    "nine_paper_limit_attested",
    "funding_coi_complete",
    "ethics_complete",
    "ai_policy_reviewed",
    "edics_confirmed",
)


def _definitions(text: str) -> dict[str, str]:
    definitions: dict[str, str] = {}
    for match in re.finditer(r"\\def\\([A-Za-z@]+)\{", text):
        depth = 1
        cursor = match.end()
        start = cursor
        while cursor < len(text) and depth:
            if text[cursor] == "{" and (cursor == 0 or text[cursor - 1] != "\\"):
                depth += 1
            elif text[cursor] == "}" and (cursor == 0 or text[cursor - 1] != "\\"):
                depth -= 1
            cursor += 1
        if depth == 0:
            definitions[match.group(1)] = text[start : cursor - 1].strip()
    return definitions


def _tracked_files(root: Path) -> list[Path]:
    completed = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        raise RuntimeError("git ls-files working-tree inventory failed")
    return [root / item.decode("utf-8", errors="strict") for item in completed.stdout.split(b"\0") if item]


def _privacy_leaks(root: Path, private_values: list[str]) -> list[str]:
    needles = [value.casefold() for value in private_values if len(value.strip()) >= 6]
    leaks: list[str] = []
    for path in _tracked_files(root):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8").casefold()
        except (UnicodeDecodeError, OSError):
            continue
        if any(needle in text for needle in needles):
            leaks.append(path.relative_to(root).as_posix())
    return leaks


def _draft_pdf_leaks(paper_dir: Path, private_values: list[str]) -> list[str]:
    """Check tracked/internal draft PDFs, never the single-blind submission PDF."""
    from pypdf import PdfReader

    needles = [value.casefold() for value in private_values if len(value.strip()) >= 6]
    candidates = [paper_dir / "main.pdf", *sorted(paper_dir.glob("main_round*.pdf"))]
    leaks: list[str] = []
    for path in candidates:
        if not path.is_file():
            continue
        reader = PdfReader(str(path), strict=True)
        payload = "\n".join(page.extract_text() or "" for page in reader.pages)
        payload += "\n" + "\n".join(str(value) for value in (reader.metadata or {}).values())
        folded = payload.casefold()
        if any(needle in folded for needle in needles):
            leaks.append(path.relative_to(paper_dir.parent).as_posix())
    return leaks


def validate(paper_dir: Path, mode: str) -> tuple[dict[str, Any], int]:
    root = paper_dir.parent.resolve()
    path = paper_dir / "private/author_metadata.tex"
    attestation_path = paper_dir / "private/author_attestations.json"
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    definitions: dict[str, str] = {}
    if path.is_file():
        try:
            definitions = _definitions(path.read_text(encoding="utf-8", errors="strict"))
        except Exception as exc:
            add_check(checks, "private_metadata_parse", False, type(exc).__name__)
            return finish_report(gate="author_metadata", mode=mode, checks=checks, structural_failure=True)
    else:
        blockers.append("private author metadata file is missing")
    definitions_complete = all(definitions.get(name) for name in REQUIRED_DEFINITIONS)
    add_check(checks, "required_private_definitions", definitions_complete, "all required definitions are present" if definitions_complete else "private definitions incomplete", blocking=False)
    complete_flag = definitions.get("RACAuthorMetadataComplete", "").lower() == "true"
    values = [definitions.get(name, "") for name in REQUIRED_DEFINITIONS[1:]]
    no_placeholders = all(value and not re.search(r"PENDING|TBD|UNKNOWN|VERIFY|\[|\]", value, re.IGNORECASE) for value in values)
    email_shape = bool(re.fullmatch(r"[^\s@{}]+@[^\s@{}]+\.[^\s@{}]+(?:\s*[,;]\s*[^\s@{}]+@[^\s@{}]+\.[^\s@{}]+)*", definitions.get("RACAuthorEmails", "")))
    if not (complete_flag and definitions_complete and no_placeholders and email_shape):
        blockers.append("complete author roster, affiliations, and emails are not confirmed")
    add_check(checks, "metadata_complete", complete_flag and definitions_complete and no_placeholders and email_shape, "private metadata complete" if complete_flag else "private metadata remains incomplete", blocking=False)

    attestations: dict[str, Any] = {}
    if attestation_path.is_file():
        try:
            attestations = json.loads(attestation_path.read_text(encoding="utf-8"))
        except Exception:
            attestations = {}
    attestation_ok = (
        attestations.get("schema_version") == "1.0"
        and isinstance(attestations.get("author_count"), int)
        and attestations.get("author_count", 0) >= 1
        and all(attestations.get(key) is True for key in REQUIRED_ATTESTATIONS)
    )
    if not attestation_ok:
        blockers.append("private author and venue attestations are incomplete")
    add_check(checks, "author_attestations", attestation_ok, "all authors assented and declarations are complete" if attestation_ok else "attestations missing or incomplete", blocking=False)

    leaks: list[str] = []
    privacy_error = None
    pdf_leaks: list[str] = []
    if definitions:
        try:
            # Check exact private fields and individual e-mail addresses; never
            # serialize the values themselves into this report.
            fragments = [
                item.strip()
                for value in values
                for item in re.split(r"[,;]|\\and|\band\b", value, flags=re.IGNORECASE)
                if item.strip()
            ]
            private_values = values + fragments + re.findall(r"[^\s,;{}]+@[^\s,;{}]+", definitions.get("RACAuthorEmails", ""))
            leaks = _privacy_leaks(root, private_values)
            pdf_leaks = _draft_pdf_leaks(paper_dir, private_values)
        except Exception as exc:
            privacy_error = type(exc).__name__
    privacy_ok = not leaks and not pdf_leaks and privacy_error is None
    add_check(checks, "tracked_tree_privacy", not leaks and privacy_error is None, f"leak_files={len(leaks)}" if privacy_error is None else privacy_error)
    add_check(checks, "draft_pdf_privacy", not pdf_leaks and privacy_error is None, f"leak_pdfs={len(pdf_leaks)}" if privacy_error is None else privacy_error)

    report, code = finish_report(
        gate="author_metadata",
        mode=mode,
        checks=checks,
        blockers=blockers,
        structural_failure=not privacy_ok,
    )
    report["privacy"] = {
        "private_values_emitted": False,
        "tracked_leak_count": len(leaks),
        "tracked_leak_paths": leaks,
        "draft_pdf_leak_count": len(pdf_leaks),
        "draft_pdf_leak_paths": pdf_leaks,
    }
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
