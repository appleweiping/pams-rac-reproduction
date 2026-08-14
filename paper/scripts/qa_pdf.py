#!/usr/bin/env python3
"""Deterministic post-compile QA for the ICASSP pre-results draft PDF."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone

from pypdf import PdfReader
WARNING = re.compile(
    r"Overfull|undefined|Citation.*undefined|Reference.*undefined|"
    r"multiply defined|duplicate",
    re.I,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_files(paper_dir: Path) -> list[Path]:
    paths = [
        paper_dir / "main.tex", paper_dir / "preamble.tex",
        paper_dir / "math_commands.tex", paper_dir / "author_public.tex",
        paper_dir / "references.bib",
    ]
    for directory in ("sections", "generated"):
        paths.extend(sorted((paper_dir / directory).glob("*.tex")))
    paths.extend([
        paper_dir / "figures/introduction_source_full.pdf",
        paper_dir / "figures/framework_source_full.pdf",
    ])
    return [path for path in paths if path.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paper_dir = args.paper_dir.resolve()
    pdf = (args.pdf or paper_dir / "main.pdf").resolve()
    log = paper_dir / "main.log"
    report_path = args.output or paper_dir / ".aris/build-qa.json"

    warnings = [line.strip() for line in log.read_text(encoding="utf-8", errors="replace").splitlines() if WARNING.search(line)]
    reader = PdfReader(str(pdf))
    page_text = [page.extract_text() or "" for page in reader.pages]
    conclusion_pages = [
        index + 1 for index, text in enumerate(page_text)
        if re.search(r"\bConclusion\b", text, re.I)
    ]
    letter_pages = all(
        abs(float(page.mediabox.width) - 612) < 1 and abs(float(page.mediabox.height) - 792) < 1
        for page in reader.pages
    )

    pdffonts = shutil.which("pdffonts")
    if not pdffonts:
        raise SystemExit("pdffonts not found")
    font_output = subprocess.check_output([pdffonts, str(pdf)], text=True, errors="replace")
    font_lines = [line for line in font_output.splitlines()[2:] if line.strip()]
    unembedded = [line for line in font_lines if len(line.split()) < 5 or line.split()[-5].lower() != "yes"]
    type3_count = sum("Type 3" in line for line in font_lines)

    source_paths = source_files(paper_dir)
    source_records = [
        {"path": path.relative_to(paper_dir).as_posix(), "sha256": sha(path)}
        for path in source_paths
    ]
    bundle_payload = "\n".join(f"{item['path']}\t{item['sha256']}" for item in source_records).encode("utf-8")

    labels: list[str] = []
    citations: set[str] = set()
    for path in source_paths:
        if path.suffix != ".tex":
            continue
        text = path.read_text(encoding="utf-8")
        labels.extend(re.findall(r"\\label\{([^}]+)\}", text))
        for group in re.findall(r"\\cite\w*\{([^}]+)\}", text):
            citations.update(item.strip() for item in group.split(","))
    bib_keys = set(re.findall(r"^@\w+\{([^,]+),", (paper_dir / "references.bib").read_text(encoding="utf-8"), re.M))

    checks = {
        "final_log_warning_count": len(warnings),
        "page_count": len(reader.pages),
        "letter_page_size": letter_pages,
        "conclusion_page": min(conclusion_pages) if conclusion_pages else None,
        "conclusion_within_four_pages": bool(conclusion_pages and min(conclusion_pages) <= 4),
        "font_count": len(font_lines),
        "unembedded_font_count": len(unembedded),
        "type3_font_count_advisory": type3_count,
        "duplicate_labels": sorted({label for label in labels if labels.count(label) > 1}),
        "undefined_citation_keys": sorted(citations - bib_keys),
        "uncited_bib_keys": sorted(bib_keys - citations),
        "approved_author_fields_visible": all(
            value in "\n".join(page_text)
            for value in (
                "Weiping Yan",
                "University of Minnesota Twin Cities",
                "yan00944@umn.edu",
            )
        ),
        "draft_banner_visible": False,
        "unresolved_question_marks": any("??" in text for text in page_text),
    }
    # Normalize the three common TeX/PDF dash renderings without depending on
    # the PDF extractor's treatment of a literal Unicode dash.
    checks["draft_banner_visible"] = any(
        re.search(r"DRAFT\s+[-\N{EN DASH}\N{EM DASH}]+\s+RESULTS PENDING", text)
        for text in page_text
    )
    blocking = (
        warnings
        or not letter_pages
        or len(reader.pages) != 5
        or not checks["conclusion_within_four_pages"]
        or unembedded
        or checks["duplicate_labels"]
        or checks["undefined_citation_keys"]
        or checks["uncited_bib_keys"]
        or not checks["approved_author_fields_visible"]
        or not checks["draft_banner_visible"]
        or checks["unresolved_question_marks"]
    )
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "verdict": "PASS" if not blocking else "FAIL",
        "pdf": {"path": pdf.name, "sha256": sha(pdf), "bytes": pdf.stat().st_size},
        "source_bundle_sha256": hashlib.sha256(bundle_payload).hexdigest(),
        "source_files": source_records,
        "checks": checks,
        "final_log_warnings": warnings,
        "unembedded_fonts": unembedded,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"BUILD_QA={report['verdict']}; pages={len(reader.pages)}; warnings={len(warnings)}; fonts_unembedded={len(unembedded)}")
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
