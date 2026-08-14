#!/usr/bin/env python3
"""Validate the two exact PPT-sourced manuscript figures and their captions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

from pypdf import PdfReader


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add(checks: list[dict], check_id: str, ok: bool, detail: object = None) -> None:
    record: dict[str, object] = {"id": check_id, "status": "PASS" if ok else "FAIL"}
    if detail is not None:
        record["detail"] = detail
    checks.append(record)


def slide_count(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(
            1 for name in archive.namelist()
            if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)
        )


def embedded_fonts(path: Path) -> tuple[bool, list[str]]:
    output = subprocess.check_output(["pdffonts", str(path)], text=True, errors="replace")
    rows = [line for line in output.splitlines()[2:] if line.strip()]
    bad = []
    for row in rows:
        fields = row.split()
        if len(fields) < 6 or fields[-5].lower() != "yes":
            bad.append(row)
    return bool(rows) and not bad, bad


def caption(text: str, label: str) -> str:
    pattern = r"\\caption\{(.*?)\}\s*\\label\{" + re.escape(label) + r"\}"
    match = re.search(pattern, text, re.DOTALL)
    return " ".join(match.group(1).split()) if match else ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("draft", "submission"), default="draft")
    parser.add_argument("--visual-review", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paper_dir = args.paper_dir.resolve()
    root = paper_dir.parent
    manifest_path = paper_dir / "figures/figure_manifest.json"
    checks: list[dict] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        manifest = {}
        add(checks, "manifest_parse", False, type(exc).__name__)
    else:
        add(checks, "manifest_parse", manifest.get("schema_version") == "2.0")

    main_tex = (paper_dir / "main.tex").read_text(encoding="utf-8", errors="replace")
    includes = (paper_dir / "figures/latex_includes.tex").read_text(encoding="utf-8", errors="replace")
    section_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted((paper_dir / "sections").glob("*.tex"))
    )
    add(checks, "active_include_loaded", "\\input{figures/latex_includes}" in main_tex)

    records = manifest.get("active_figures", []) if isinstance(manifest, dict) else []
    add(checks, "exactly_two_active_figures", len(records) == 2, len(records))
    bound_hashes: dict[str, dict[str, str]] = {}
    for record in records:
        figure_id = record.get("id", "unknown")
        source_spec = record.get("source", {})
        output_spec = record.get("output", {})
        source = root / source_spec.get("file", "")
        output = root / output_spec.get("file", "")
        present = source.is_file() and output.is_file()
        add(checks, f"files_present:{figure_id}", present)
        if not present:
            continue
        source_hash = sha256(source)
        output_hash = sha256(output)
        bound_hashes[figure_id] = {"source": source_hash, "output": output_hash}
        add(checks, f"source_hash:{figure_id}", source_hash == source_spec.get("sha256"), source_hash)
        add(checks, f"output_hash:{figure_id}", output_hash == output_spec.get("sha256"), output_hash)
        add(checks, f"single_source_slide:{figure_id}", slide_count(source) == 1)

        reader = PdfReader(str(output), strict=True)
        one_page = len(reader.pages) == 1
        add(checks, f"single_output_page:{figure_id}", one_page)
        if one_page:
            page = reader.pages[0]
            size = [round(float(page.mediabox.width)), round(float(page.mediabox.height))]
            add(checks, f"full_16x9_canvas:{figure_id}", size == [720, 405], size)
        fonts_ok, bad_fonts = embedded_fonts(output)
        add(checks, f"fonts_embedded:{figure_id}", fonts_ok, bad_fonts)
        add(checks, f"editable_source:{figure_id}", source_spec.get("editable") is True)
        add(checks, f"visual_reference_only:{figure_id}", source_spec.get("authority") == "visual_reference_only")

        expected_file = output_spec.get("file", "").split("paper/", 1)[-1]
        add(checks, f"included_graphic:{figure_id}", expected_file in includes)
        cap = caption(section_text, figure_id)
        missing = [term for term in record.get("caption_contract", []) if term.casefold() not in cap.casefold()]
        add(checks, f"caption_contract:{figure_id}", bool(cap) and not missing, missing)
        media = str(record.get("media_provenance", ""))
        media_ok = "SYNC-REQUIRED" in media if args.mode == "draft" else "SYNC-REQUIRED" not in media
        add(checks, f"media_provenance:{figure_id}", media_ok, media)

    visual_path = args.visual_review.resolve() if args.visual_review else paper_dir / ".aris/figure-visual-review-user-revision.json"
    if visual_path.is_file():
        try:
            visual = json.loads(visual_path.read_text(encoding="utf-8"))
        except Exception as exc:
            add(checks, "visual_review_parse", False, type(exc).__name__)
        else:
            add(checks, "visual_review_verdict", visual.get("verdict") == "PASS")
            add(checks, "visual_review_figure_binding", visual.get("figure_hashes") == bound_hashes)
    else:
        add(checks, "visual_review_present", False)

    failures = [item for item in checks if item["status"] == "FAIL"]
    report = {
        "schema_version": "2.0",
        "mode": args.mode,
        "verdict": "PASS" if not failures else "FAIL",
        "figure_hashes": bound_hashes,
        "checks": checks,
    }
    output_path = args.output or paper_dir / ".aris/figure-validation-user-revision.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"FIGURE_VALIDATION={report['verdict']}; checks={len(checks)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
