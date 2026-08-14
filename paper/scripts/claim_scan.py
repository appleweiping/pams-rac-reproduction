#!/usr/bin/env python3
"""Scan submission sources and a digest-bound PDF for unsupported claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


TEXT_SUFFIXES = {".tex", ".drawio", ".svg", ".txt", ".md", ".csv", ".json", ".jsonl", ".yaml", ".yml"}
SUBMISSION_ROOTS = (
    "sections", "appendix", "tables", "figures", "generated", "supplement", "supplementary"
)
TOP_LEVEL_TEXT = {"main.tex", "preamble.tex", "math_commands.tex", "submission_wrapper.tex"}
PATTERNS = {
    # Catch grammatical primacy variants, not only the literal "we are the
    # first".  In particular, headline claims such as "the first multi-person
    # framework" and "we introduce the first pose-driven method" must not slip
    # through merely because a qualifier occurs between ``first`` and the
    # artefact noun.
    "primacy": re.compile(
        r"\b(?:"
        r"(?:we\s+(?:are|present|introduce|propose|develop|build)\s+)?(?:the\s+)?"
        r"first(?:[-\s]+ever)?\s+"
        r"(?:(?:multi[-\s]*person|person[-\s]*wise|asynchronous|variable[-\s]*tempo|"
        r"tempo[-\s]*adaptive|pose[-\s]*driven|self[-\s]*supervised|repetition[-\s]*counting)\s+){0,4}"
        r"(?:framework|method|system|approach|model|counter|work|architecture|pipeline)"
        r"|(?:we\s+(?:are\s+)?)?(?:the\s+)?first\s+to\b"
        r")",
        re.I,
    ),
    "annotation_free": re.compile(r"\bannotation[- ]free\b", re.I),
    "label_free": re.compile(r"\blabel[- ]free\b", re.I),
    "end_to_end": re.compile(r"\bend[- ]to[- ]end\b", re.I),
    "constant_time": re.compile(r"\bconstant[- ]time\b", re.I),
    "sota": re.compile(r"\b(?:SOTA|state[- ]of[- ]the[- ]art)\b", re.I),
    "positive_comparison": re.compile(
        r"\b(?:our|ours|proposed)\b[^.]{0,180}?\b(?:outperform\w*|superior(?:ity)?|better than|improv(?:e|es|ed|ement)|competitive)\b",
        re.I,
    ),
}

# Exact phrase-level exceptions. A generic nearby negation is intentionally not
# sufficient: changing one of these sentences forces review of the exception.
ALLOWLIST = {
    "end_to_end": {
        "prior_art_boundary": re.compile(r"this modular design is not described as.{0,240}end-to-\s*end", re.I),
    },
    "positive_comparison": {
        "draft_denial": re.compile(r"does not assert that.{0,180}the proposed method outperforms any baseline", re.I),
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_tex_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def normalize(text: str) -> str:
    return " ".join(text.split())


def submission_text_files(paper_dir: Path) -> list[Path]:
    paths = [paper_dir / name for name in TOP_LEVEL_TEXT]
    for root_name in SUBMISSION_ROOTS:
        directory = paper_dir / root_name
        if directory.is_dir():
            paths.extend(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES)
    return sorted({path.resolve() for path in paths if path.is_file()})


def local_context(text: str, start: int, end: int, radius: int = 260) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return normalize(text[left:right])


def allowlisted(rule: str, context: str, matched_text: str) -> str | None:
    normalized_match = normalize(matched_text)
    occurrence = context.casefold().find(normalized_match.casefold())
    if occurrence < 0:
        return None
    for allowlist_id, pattern in ALLOWLIST.get(rule, {}).items():
        allowed = pattern.search(context)
        if allowed and allowed.start() <= occurrence and occurrence + len(normalized_match) <= allowed.end():
            return allowlist_id
    return None


def load_bindings(path: Path | None) -> tuple[dict[str, dict], set[str], str | None]:
    if not path or not path.is_file():
        return {}, set(), None
    data = json.loads(path.read_text(encoding="utf-8"))
    claims = {item["key"]: item for item in data.get("paper_bindings", {}).get("claims", [])}
    comparisons = {item["comparison_id"] for item in data.get("comparisons", [])}
    return claims, comparisons, sha256(path)


def scan_text(label: str, text: str, file_sha256: str, claims: dict[str, dict], comparisons: set[str]) -> tuple[list[dict], list[dict], dict]:
    violations: list[dict] = []
    exceptions: list[dict] = []
    for rule, pattern in PATTERNS.items():
        for match in pattern.finditer(text):
            context = local_context(text, match.start(), match.end())
            allowlist_id = allowlisted(rule, context, match.group())
            if allowlist_id:
                exceptions.append({
                    "rule": rule,
                    "file": label,
                    "allowlist_id": allowlist_id,
                    "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
                })
                continue
            detail: dict[str, object] = {}
            if rule == "positive_comparison":
                claim_keys = re.findall(r"\\C\{([^}]+)\}", context)
                comparison_ids = re.findall(r"\\Comparison\{([^}]+)\}", context)
                resolved_claims = [key for key in claim_keys if key in claims]
                resolved_comparisons = [key for key in comparison_ids if key in comparisons]
                if resolved_claims and resolved_comparisons:
                    continue
                detail = {
                    "required": "nearby \\C{manifest-claim-key} and \\Comparison{manifest-comparison-id}",
                    "claim_keys": claim_keys,
                    "comparison_ids": comparison_ids,
                    "resolved_claim_keys": resolved_claims,
                    "resolved_comparison_ids": resolved_comparisons,
                }
            line = text.count("\n", 0, match.start()) + 1
            violations.append({
                "rule": rule,
                "file": label,
                "line": line,
                "context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
                **detail,
            })
    return violations, exceptions, {"file": label, "sha256": file_sha256}


def extract_pdf_text(pdf: Path) -> bytes:
    return subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("draft", "submission"), default="draft")
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--pdf-text", type=Path)
    parser.add_argument("--result-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paper_dir = args.paper_dir.resolve()
    claims, comparisons, result_manifest_sha256 = load_bindings(args.result_manifest.resolve() if args.result_manifest else None)

    violations: list[dict] = []
    exceptions: list[dict] = []
    scanned: list[dict] = []
    for path in submission_text_files(paper_dir):
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix.lower() == ".tex":
            text = strip_tex_comments(text)
        found, allowed, record = scan_text(path.relative_to(paper_dir).as_posix(), text, sha256(path), claims, comparisons)
        violations.extend(found)
        exceptions.extend(allowed)
        scanned.append(record)

    pdf_binding: dict[str, object] = {"required": args.mode == "submission", "status": "NOT_PROVIDED"}
    pdf = args.pdf.resolve() if args.pdf else None
    pdf_text = args.pdf_text.resolve() if args.pdf_text else None
    if bool(pdf) != bool(pdf_text):
        violations.append({"rule": "pdf_binding", "file": "main.pdf", "detail": "--pdf and --pdf-text must be supplied together"})
        pdf_binding["status"] = "FAIL"
    elif pdf and pdf_text:
        if not pdf.is_file() or not pdf_text.is_file():
            violations.append({"rule": "pdf_binding", "file": "main.pdf", "detail": "PDF or extracted text is missing"})
            pdf_binding["status"] = "FAIL"
        else:
            supplied_bytes = pdf_text.read_bytes()
            try:
                extracted_bytes = extract_pdf_text(pdf)
            except Exception as exc:
                extracted_bytes = b""
                violations.append({"rule": "pdf_binding", "file": "main.pdf", "detail": f"extractor error:{type(exc).__name__}"})
            exact = supplied_bytes.replace(b"\r\n", b"\n") == extracted_bytes.replace(b"\r\n", b"\n")
            if not exact:
                violations.append({"rule": "pdf_binding", "file": "main.pdf", "detail": "extracted text is not byte-bound to this PDF"})
            pdf_binding = {
                "status": "PASS" if exact else "FAIL",
                "pdf_sha256": sha256(pdf),
                "extracted_text_sha256": sha256(pdf_text),
                "extractor": "pdftotext -layout",
                "exact_extraction_match": exact,
            }
            text = supplied_bytes.decode("utf-8", errors="replace")
            found, allowed, record = scan_text("main.pdf:extracted-text", text, sha256(pdf_text), claims, comparisons)
            violations.extend(found)
            exceptions.extend(allowed)
            scanned.append(record)
    elif args.mode == "submission":
        violations.append({"rule": "pdf_binding", "file": "main.pdf", "detail": "submission mode requires --pdf and --pdf-text"})

    if violations:
        verdict = "FAIL"
    elif args.mode == "draft" and pdf_binding["status"] == "NOT_PROVIDED":
        verdict = "PROVISIONAL"
    else:
        verdict = "PASS"
    report = {
        "schema_version": 2,
        "mode": args.mode,
        "verdict": verdict,
        "allowlist_policy": "Only the exact phrase-level contextual exceptions named in this report are allowed; generic nearby negation is not an exception.",
        "positive_comparison_policy": "Own-method positive comparisons require both a manifest-resolved claim binding and a manifest-resolved comparison binding in local context.",
        "result_manifest_sha256": result_manifest_sha256,
        "pdf_binding": pdf_binding,
        "scanned": scanned,
        "allowlisted_occurrences": exceptions,
        "violations": violations,
    }
    output = args.output or paper_dir / ".aris/claim-scan.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"CLAIM_SCAN={verdict}; files={len(scanned)}; violations={len(violations)}")
    return 0 if verdict in {"PASS", "PROVISIONAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
