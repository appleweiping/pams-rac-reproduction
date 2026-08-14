#!/usr/bin/env python3
"""Scan active submission prose for placeholders and unlicensed claims."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any

from evidence_common import add_check, emit_report, finish_report

FORBIDDEN = {
    "first": re.compile(r"\bfirst\b", re.IGNORECASE),
    "annotation-free": re.compile(r"\bannotation[- ]free\b", re.IGNORECASE),
    "label-free": re.compile(r"\blabel[- ]free\b", re.IGNORECASE),
    "end-to-end": re.compile(r"\bend[- ]to[- ]end\b", re.IGNORECASE),
    "constant-time": re.compile(r"\bconstant[- ]time\b", re.IGNORECASE),
    "sota": re.compile(r"\bSOTA\b|\bstate[- ]of[- ]the[- ]art\b", re.IGNORECASE),
}
PLACEHOLDERS = {
    "PENDING": re.compile(r"\bPENDING\b", re.IGNORECASE),
    "SYNC-REQUIRED": re.compile(r"\bSYNC-REQUIRED\b", re.IGNORECASE),
    "[VERIFY]": re.compile(r"\[\s*VERIFY\s*\]", re.IGNORECASE),
    "TBD": re.compile(r"\bTBD\b", re.IGNORECASE),
    "TODO": re.compile(r"\bTODO\b", re.IGNORECASE),
}
INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}|\\InputIfFileExists\s*\{([^}]+)\}")
PROSE_EXCLUDE = {"preamble.tex", "math_commands.tex", "submission_wrapper.tex"}
CORE_LEDGER_IDS = {*(f"AC{number:02d}" for number in range(1, 11)), "AC13"}


def strip_comments(text: str) -> str:
    cleaned: list[str] = []
    for line in text.splitlines():
        index = 0
        while True:
            marker = line.find("%", index)
            if marker < 0:
                cleaned.append(line)
                break
            backslashes = 0
            cursor = marker - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                cleaned.append(line[:marker])
                break
            index = marker + 1
    return "\n".join(cleaned)


def active_tex_files(paper_dir: Path, entry: Path) -> list[Path]:
    paper_dir = paper_dir.resolve()
    pending = [entry.resolve()]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen or not path.is_file():
            continue
        try:
            path.relative_to(paper_dir)
        except ValueError:
            continue
        seen.add(path)
        text = strip_comments(path.read_text(encoding="utf-8", errors="strict"))
        for match in INPUT_RE.finditer(text):
            declared = match.group(1) or match.group(2)
            candidate = Path(declared)
            if not candidate.suffix:
                candidate = candidate.with_suffix(".tex")
            candidates = ((path.parent / candidate).resolve(), (paper_dir / candidate).resolve())
            for resolved in candidates:
                if resolved.is_file():
                    pending.append(resolved)
                    break
    return sorted(seen)


def _line_context(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()[:240]


def _allowed_forbidden_context(name: str, line: str) -> bool:
    lower = re.sub(r"\\[A-Za-z@]+\*?", " ", line).lower()
    if name == "first" and re.search(r"\b(first|second|third)\b\s*[,;:]", lower):
        return True
    return bool(
        re.search(
            r"\b(?:do not|does not|did not|not|never)\s+(?:claim\s+)?(?:to\s+be\s+)?(?:the\s+)?(?:first|annotation[- ]free|label[- ]free|end[- ]to[- ]end|constant[- ]time|sota|state[- ]of[- ]the[- ]art)\b",
            lower,
        )
        or re.search(r"\b(?:prior|previous|earlier|existing)\s+(?:work|method|study|approach)s?\b", lower)
        or re.search(r"\b(?:introduced|proposed|reported|described|presented)\s+by\b", lower)
    )


def scan_text(text: str, label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    forbidden: list[dict[str, Any]] = []
    placeholders: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for name, pattern in FORBIDDEN.items():
            if pattern.search(line) and not _allowed_forbidden_context(name, line):
                forbidden.append({"source": label, "line": line_number, "token": name, "context": _line_context(line)})
        for name, pattern in PLACEHOLDERS.items():
            if pattern.search(line):
                placeholders.append({"source": label, "line": line_number, "token": name, "context": _line_context(line)})
    return forbidden, placeholders


def _load_ledger(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        return {}, [type(exc).__name__]
    required_columns = {"claim_id", "claim_class", "claim", "evidence_required", "evidence_paths", "status", "submission_licensed"}
    if not rows or set(rows[0]) != required_columns:
        errors.append("ledger columns do not match the fixed schema")
    ledger: dict[str, dict[str, str]] = {}
    for row in rows:
        claim_id = row.get("claim_id", "")
        if claim_id in ledger or not re.fullmatch(r"AC(?:0[1-9]|1[0-6])", claim_id):
            errors.append("ledger has a duplicate or malformed claim id")
        ledger[claim_id] = row
    expected = {f"AC{number:02d}" for number in range(1, 17)}
    if set(ledger) != expected:
        errors.append("ledger must contain exactly AC01 through AC16")
    return ledger, errors


def _binding_checks(texts: list[str], generated: str | None) -> list[str]:
    combined = "\n".join(texts)
    result_uses = set(re.findall(r"\\R\{([^}]*)\}", combined))
    protocol_uses = set(re.findall(r"\\Protocol\{([^}]*)\}", combined))
    claim_uses = set(re.findall(r"\\C\{([^}]*)\}", combined))
    issues: list[str] = []
    if "" in result_uses | protocol_uses | claim_uses:
        issues.append("empty evidence macro key")
    if generated is None:
        if result_uses or protocol_uses or claim_uses or "\\ControlledAbstractResult" in combined:
            issues.append("evidence macros are used but generated/evidence_values.tex is absent")
        return issues
    declarations = {
        "result": set(re.findall(r"\\DeclareResult\{([^}]+)\}", generated)),
        "protocol": set(re.findall(r"\\DeclareProtocolField\{([^}]+)\}", generated)),
        "claim": set(re.findall(r"\\DeclareResultClaim\{([^}]+)\}", generated)),
    }
    if result_uses - declarations["result"]:
        issues.append("unbound result macro keys")
    if protocol_uses - declarations["protocol"]:
        issues.append("unbound protocol macro keys")
    if claim_uses - declarations["claim"]:
        issues.append("unbound result-claim macro keys")
    if "\\ControlledAbstractResult" in combined and "\\DeclareAbstractResult{" not in generated:
        issues.append("controlled abstract result is unbound")
    return issues


def _empty_table_issues(text: str) -> int:
    issues = 0
    for match in re.finditer(r"\\begin\{tabular\*?\}.*?\\end\{tabular\*?\}", text, re.DOTALL):
        body = match.group(0)
        if "\\\\" not in body or re.search(r"&\s*(?:&|\\\\)", body):
            issues += 1
    return issues


def validate(paper_dir: Path, mode: str, entry: Path | None, pdf_text: Path | None) -> tuple[dict[str, Any], int]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    entry = entry or paper_dir / "main.tex"
    try:
        files = active_tex_files(paper_dir, entry)
    except Exception as exc:
        add_check(checks, "active_source_graph", False, type(exc).__name__)
        return finish_report(gate="submission_prose", mode=mode, checks=checks, structural_failure=True)
    add_check(checks, "active_source_graph", bool(files), f"{len(files)} active TeX files")

    forbidden: list[dict[str, Any]] = []
    placeholders: list[dict[str, Any]] = []
    prose_texts: list[str] = []
    for path in files:
        if path.name in PROSE_EXCLUDE:
            continue
        text = strip_comments(path.read_text(encoding="utf-8", errors="strict"))
        prose_texts.append(text)
        bad, pending = scan_text(text, path.relative_to(paper_dir).as_posix())
        forbidden.extend(bad)
        placeholders.extend(pending)
    if pdf_text is not None:
        text = pdf_text.read_text(encoding="utf-8", errors="replace")
        body = re.split(r"(?im)^\s*references\s*$", text, maxsplit=1)[0]
        bad, pending = scan_text(body, "PDF_BODY")
        forbidden.extend(bad)
        placeholders.extend(pending)

    if forbidden:
        blockers.append("forbidden claim language remains in submission prose")
    if placeholders:
        blockers.append("PENDING, SYNC-REQUIRED, [VERIFY], TBD, or TODO remains in submission prose")
    add_check(checks, "forbidden_claims", not forbidden, f"{len(forbidden)} unallowlisted hits", blocking=False)
    add_check(checks, "placeholders", not placeholders, f"{len(placeholders)} hits", blocking=False)

    table_issues = sum(_empty_table_issues(text) for text in prose_texts)
    if table_issues:
        blockers.append("empty table cells or bodies remain")
    add_check(checks, "empty_tables", table_issues == 0, f"{table_issues} suspect tables", blocking=False)

    generated_path = paper_dir / "generated/evidence_values.tex"
    generated = generated_path.read_text(encoding="utf-8") if generated_path.is_file() else None
    binding_issues = _binding_checks(prose_texts, generated)
    if binding_issues:
        blockers.extend(binding_issues)
    add_check(checks, "evidence_macro_bindings", not binding_issues, "; ".join(binding_issues) if binding_issues else "all used macros are bound", blocking=False)

    ledger, ledger_errors = _load_ledger(paper_dir / "evidence/claim_evidence.csv")
    add_check(checks, "claim_ledger_schema", not ledger_errors, "; ".join(ledger_errors) if ledger_errors else "exactly 16 claims")
    mapped = set(re.findall(r"\\ClaimMap\{(AC(?:0[1-9]|1[0-6]))\}", "\n".join(prose_texts)))
    unknown_maps = mapped - set(ledger)
    unlicensed_maps = {claim_id for claim_id in mapped if ledger.get(claim_id, {}).get("submission_licensed", "").lower() != "true"}
    core_unlicensed = {claim_id for claim_id in CORE_LEDGER_IDS if ledger.get(claim_id, {}).get("submission_licensed", "").lower() != "true"}
    if unknown_maps:
        blockers.append("manuscript contains claim mappings absent from the ledger")
    if unlicensed_maps or core_unlicensed:
        blockers.append("claim ledger has unlicensed submission-facing claims")
    add_check(checks, "claim_mapping", not unknown_maps and not unlicensed_maps and not core_unlicensed, f"mapped={len(mapped)}, unlicensed_core={len(core_unlicensed)}", blocking=False)

    report, code = finish_report(
        gate="submission_prose",
        mode=mode,
        checks=checks,
        blockers=blockers,
        structural_failure=any(item["status"] == "FAIL" for item in checks),
    )
    report["sources"] = [path.relative_to(paper_dir).as_posix() for path in files]
    report["violations"] = {"forbidden": forbidden, "placeholders": placeholders, "empty_tables": table_issues}
    report["claim_mapping"] = {"mapped_ids": sorted(mapped), "unknown_ids": sorted(unknown_maps), "unlicensed_ids": sorted(unlicensed_maps | core_unlicensed)}
    return report, code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=("draft", "submission"), default="draft")
    parser.add_argument("--entry", type=Path)
    parser.add_argument("--pdf-text", type=Path)
    parser.add_argument("--pdf", type=Path, help="accepted for compatibility; use --pdf-text for deterministic scanning")
    parser.add_argument("--result-manifest", type=Path, help="deprecated compatibility option")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paper_dir = args.paper_dir.resolve()
    entry = args.entry.resolve() if args.entry else None
    report, code = validate(paper_dir, args.mode, entry, args.pdf_text.resolve() if args.pdf_text else None)
    emit_report(report, args.output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
