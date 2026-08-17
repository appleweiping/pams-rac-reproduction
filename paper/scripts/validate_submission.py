#!/usr/bin/env python3
"""Fail closed when the paper is not ready for a submission build."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PLACEHOLDER_PATTERNS = {
    "TBD": re.compile(r"\bTBD\b", re.IGNORECASE),
    "SYNC-REQUIRED": re.compile(r"SYNC-REQUIRED", re.IGNORECASE),
    "[VERIFY]": re.compile(r"\[VERIFY\]", re.IGNORECASE),
    "draft banner": re.compile(r"DRAFT\s*-{2,3}\s*RESULTS\s+PENDING", re.IGNORECASE),
    "withheld result": re.compile(r"result-dependent statement withheld", re.IGNORECASE),
    "pending evidence prose": re.compile(r"evidence is pending synchronization", re.IGNORECASE),
}

POSITIVE_HEADLINE_PATTERNS = {
    "first claim": re.compile(r"(?<!not\s)\bfirst(?:-ever)?\b", re.IGNORECASE),
    "annotation-free claim": re.compile(r"(?<!not\s)\bannotation[- ]free\b", re.IGNORECASE),
    "label-free claim": re.compile(r"(?<!not\s)\blabel[- ]free\b", re.IGNORECASE),
    "end-to-end claim": re.compile(r"(?<!not\s)\bend[- ]to[- ]end\b", re.IGNORECASE),
    "constant-time claim": re.compile(r"(?<!not\s)\bconstant[- ]time\b", re.IGNORECASE),
    "SOTA claim": re.compile(r"\b(?:SOTA|state[- ]of[- ]the[- ]art)\b", re.IGNORECASE),
}

EMAIL_PATTERN = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def find_tex_files(paper_dir: Path) -> list[Path]:
    roots = [
        paper_dir / "main.tex",
        paper_dir / "preamble.tex",
        paper_dir / "math_commands.tex",
        paper_dir / "author_public.tex",
        paper_dir / "submission_wrapper.tex",
    ]
    roots.extend(sorted((paper_dir / "sections").glob("*.tex")))
    roots.extend(sorted((paper_dir / "appendix").glob("*.tex")))
    roots.extend(sorted((paper_dir / "tables").glob("*.tex")))
    roots.extend(sorted((paper_dir / "generated").glob("*.tex")))
    return [path for path in roots if path.is_file()]


def declared_keys(evidence: str, command: str) -> set[str]:
    return set(re.findall(r"\\" + re.escape(command) + r"\{([^}]+)\}", evidence))


def used_keys(text: str, command: str) -> set[str]:
    return set(re.findall(r"\\" + re.escape(command) + r"\{([^}]+)\}", text))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, required=True)
    args = parser.parse_args()
    paper_dir = args.paper_dir.resolve()

    files = find_tex_files(paper_dir)
    if not files or not (paper_dir / "main.tex").is_file():
        print("ERROR: paper source is incomplete", file=sys.stderr)
        return 2

    contents = {path: strip_comments(path.read_text(encoding="utf-8")) for path in files}
    combined = "\n".join(contents.values())
    failures: list[str] = []

    for label, pattern in PLACEHOLDER_PATTERNS.items():
        for path, text in contents.items():
            if pattern.search(text):
                failures.append(f"{label}: {path.relative_to(paper_dir)}")

    headline_paths = [
        paper_dir / "main.tex",
        paper_dir / "sections" / "0_abstract.tex",
        paper_dir / "sections" / "1_introduction.tex",
        paper_dir / "sections" / "5_conclusion.tex",
    ]
    for path in headline_paths:
        text = contents.get(path, "")
        for label, pattern in POSITIVE_HEADLINE_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"{label}: {path.relative_to(paper_dir)}")

    main_text = contents.get(paper_dir / "main.tex", "")
    if "\\input{author_public}" not in main_text:
        failures.append("missing approved single-anonymous author include")
    if re.search(r"AUTHOR\s+ROSTER\s+PENDING|AFFILIATIONS\s+PENDING", combined, re.I):
        failures.append("obsolete author placeholder found")

    evidence_path = paper_dir / "generated" / "evidence_values.tex"
    if not evidence_path.is_file():
        failures.append("missing generated/evidence_values.tex")
        evidence = ""
    else:
        evidence = strip_comments(evidence_path.read_text(encoding="utf-8"))

    abstract_declarations = len(re.findall(r"\\DeclareAbstractResult\{", evidence))
    if abstract_declarations != 1:
        failures.append(
            f"expected exactly one audited abstract result declaration, found {abstract_declarations}"
        )

    for command in ("DeclareResult", "DeclareProtocolField", "DeclareResultClaim"):
        for key, value in re.findall(
            r"\\" + command + r"\{([^}]*)\}\{([^}]*)\}", evidence
        ):
            if not key.strip() or not value.strip():
                failures.append(f"empty {command} binding: {key or '<empty-key>'}")

    for use_command, declaration_command in (
        ("R", "DeclareResult"),
        ("Protocol", "DeclareProtocolField"),
        ("C", "DeclareResultClaim"),
    ):
        missing = sorted(
            used_keys(combined, use_command) - declared_keys(evidence, declaration_command)
        )
        if missing:
            failures.append(f"unbound {use_command} keys: {', '.join(missing)}")

    if failures:
        print("SUBMISSION GATE: FAIL", file=sys.stderr)
        for failure in sorted(set(failures)):
            print(f"- {failure}", file=sys.stderr)
        return 1

    print("SUBMISSION GATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
