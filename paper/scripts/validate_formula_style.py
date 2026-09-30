#!/usr/bin/env python3
"""Enforce the requested punctuation-free display-equation style."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paper_dir = args.paper_dir.resolve()
    source = paper_dir / "sections/3_method.tex"
    text = source.read_text(encoding="utf-8")
    matches = list(re.finditer(r"\\begin\{equation\}(.*?)\\end\{equation\}", text, re.DOTALL))
    checks: list[dict] = []
    checks.append({
        "id": "exactly_four_display_equation_groups",
        "status": "PASS" if len(matches) == 4 else "FAIL",
        "detail": len(matches),
    })
    for index, match in enumerate(matches, start=1):
        body = re.sub(r"\\label\{[^}]+\}", "", match.group(1))
        literal_punctuation = sorted(set(re.findall(r"[,.;:]", body)))
        punctuation_commands = sorted(set(re.findall(
            r"\\(?:ldots|cdots|dots|dotsc|dotsb|dotsi|dotsm|colon)\b",
            body,
        )))
        punctuation = literal_punctuation + punctuation_commands
        checks.append({
            "id": f"equation_{index}_contains_no_punctuation",
            "status": "PASS" if not punctuation else "FAIL",
            "detail": punctuation,
        })
        tail = text[match.end():]
        next_nonblank = next((line.strip() for line in tail.splitlines() if line.strip()), "")
        checks.append({
            "id": f"equation_{index}_explanation_flush_left",
            "status": "PASS" if next_nonblank.startswith(r"\noindent") else "FAIL",
            "detail": next_nonblank[:120],
        })
        checks.append({
            "id": f"equation_{index}_no_trailing_punctuation",
            "status": "PASS" if not re.match(r"\s*[,.;:]", tail) else "FAIL",
        })
    failures = [item for item in checks if item["status"] == "FAIL"]
    report = {
        "schema_version": "1.0",
        "verdict": "PASS" if not failures else "FAIL",
        "source": "sections/3_method.tex",
        "checks": checks,
    }
    output = args.output or paper_dir / ".aris/formula-style-user-revision.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"FORMULA_STYLE={report['verdict']}; equations={len(matches)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
