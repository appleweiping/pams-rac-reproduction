#!/usr/bin/env python3
"""Page-aware ICASSP 4+1 PDF gate.

The live 2027 rule is four technical pages plus an optional references-only
fifth page. A later relaxation is honored only through a byte-verified official
2027 kit manifest.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from evidence_common import add_check, emit_report, finish_report
from validate_venue_kit import OFFICIAL_URLS, verified_kit

REFERENCE_HEADING = re.compile(r"(?im)^\s*(?:\d+\.?\s*)?references\s*$")
REFERENCE_ENTRY = re.compile(r"(?m)^\s*\[(\d+)\]\s+")
TECHNICAL_HEADING = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)*\.?\s*)?(?:abstract|introduction|related work|"
    r"method(?:ology)?|experiments?|results?|discussion|conclusion|limitations?|"
    r"appendix|supplementary material|algorithm)\s*$"
)
CAPTION = re.compile(r"(?im)^\s*(?:fig(?:ure)?\.?|table|algorithm)\s*[A-Z0-9IVX]+\s*[:.]", re.IGNORECASE)
DECLARATION_HEADING = re.compile(
    r"(?im)^\s*(?:acknowledg(?:e)?ments?|funding|conflicts? of interest|"
    r"compliance with ethical standards|ethical compliance|ethics statement)\s*$"
)


def _page_graphics(page: Any, reader: Any) -> tuple[int, int]:
    """Return (image/form XObjects, painted vector paths) for one page."""
    from pypdf.generic import ContentStream

    xobjects = 0
    resources = page.get("/Resources")
    if resources and resources.get("/XObject"):
        xobjects = len(resources["/XObject"])
    vector_operators = {b"S", b"s", b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*"}
    painted_paths = 0
    contents = page.get_contents()
    if contents is not None:
        stream = ContentStream(contents, reader)
        painted_paths = sum(operator in vector_operators for _operands, operator in stream.operations)
    return xobjects, painted_paths


def _reference_continuation(page4: str, page5: str, heading_pages: list[int]) -> tuple[bool, list[int]]:
    """Recognize a numbered bibliography that begins before and flows onto page 5.

    ICASSP templates do not repeat the References heading at a column or page
    break. We therefore require an earlier heading, consecutive numbered labels
    across the page boundary, and only a bounded carry-over prefix before the
    first new label. The other page-five checks still reject technical headings,
    captions, declarations, images, and vector drawings.
    """
    page4_entries = [int(value) for value in REFERENCE_ENTRY.findall(page4)]
    page5_entries = [int(value) for value in REFERENCE_ENTRY.findall(page5)]
    first_entry = REFERENCE_ENTRY.search(page5)
    carry_over = page5[: first_entry.start()] if first_entry else page5
    consecutive = bool(page5_entries) and all(
        right == left + 1 for left, right in zip(page5_entries, page5_entries[1:])
    )
    continuation = (
        bool(heading_pages)
        and max(heading_pages) < 4
        and bool(page4_entries)
        and bool(page5_entries)
        and page5_entries[0] == page4_entries[-1] + 1
        and consecutive
        and len(carry_over.strip()) <= 600
    )
    return continuation, page5_entries


def validate(pdf: Path, paper_dir: Path, mode: str) -> tuple[dict[str, Any], int]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    if not pdf.is_file():
        blockers.append("compiled PDF is missing")
        add_check(checks, "pdf_present", False, "missing PDF", blocking=False)
        return finish_report(gate="pdf_4plus1", mode=mode, checks=checks, blockers=blockers)
    try:
        from pypdf import PdfReader

        document = PdfReader(str(pdf), strict=True)
    except Exception as exc:
        add_check(checks, "pdf_parse", False, type(exc).__name__)
        return finish_report(gate="pdf_4plus1", mode=mode, checks=checks, structural_failure=True)
    pages = len(document.pages)
    add_check(checks, "page_count", 1 <= pages <= 5, f"pages={pages}")
    page_texts = [document.pages[index].extract_text() or "" for index in range(pages)]
    full_text = "\n".join(page_texts)
    references_exist = bool(REFERENCE_HEADING.search(full_text))
    if mode == "submission" and not references_exist:
        blockers.append("PDF has no References heading")
    add_check(
        checks,
        "references_heading",
        references_exist,
        "References heading found" if references_exist else "References heading absent",
        blocking=False,
    )

    kit_ok, kit_policy, _ = verified_kit(paper_dir)
    policy = kit_policy if kit_ok else "references-only"
    page5_report: dict[str, Any] = {"present": pages == 5, "policy": policy}
    if pages == 5:
        page = document.pages[4]
        text = page_texts[4]
        heading = REFERENCE_HEADING.search(text)
        heading_pages = [index for index, page_text in enumerate(page_texts) if REFERENCE_HEADING.search(page_text)]
        before = text[: heading.start()] if heading else ""
        before_residue = re.sub(r"[\d\s\-\u2013\u2014]+", "", before)
        continuation, page5_entries = _reference_continuation(page_texts[3], text, heading_pages)
        reference_region_valid = (bool(heading) and not before_residue) or continuation
        images, drawings = _page_graphics(page, document)
        technical_headings = len(TECHNICAL_HEADING.findall(text))
        captions = len(CAPTION.findall(text))
        declarations = len(DECLARATION_HEADING.findall(text))
        references_only = (
            reference_region_valid
            and technical_headings == 0
            and captions == 0
            and images == 0
            and drawings == 0
        )
        if policy == "references-only":
            references_only = references_only and declarations == 0
        if not references_only:
            blockers.append("page 5 contains material outside the verified page policy")
        add_check(
            checks,
            "page5_restriction",
            references_only,
            "page 5 contains references only"
            if policy == "references-only" and references_only
            else "page 5 object/text restriction failed",
            blocking=False,
        )
        page5_report.update(
            {
                "reference_heading": bool(heading),
                "reference_continuation": continuation,
                "reference_entry_labels": page5_entries,
                "non_page_number_text_before_references": bool(before_residue),
                "technical_heading_count": technical_headings,
                "caption_count": captions,
                "declaration_heading_count": declarations,
                "image_count": images,
                "significant_drawing_count": drawings,
            }
        )
    report, code = finish_report(
        gate="pdf_4plus1",
        mode=mode,
        checks=checks,
        blockers=blockers,
        structural_failure=not (1 <= pages <= 5),
    )
    report["page_count"] = pages
    report["page5"] = page5_report
    report["live_rule_source"] = OFFICIAL_URLS[1]
    return report, code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--paper-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mode", choices=("draft", "submission"), default="draft")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report, code = validate(args.pdf.resolve(), args.paper_dir.resolve(), args.mode)
    emit_report(report, args.output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
