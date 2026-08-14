#!/usr/bin/env python3
"""Validate editable/vector figure triplets, captions, and visual-review binding."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


FIGURES = {
    "task_comparison": {
        "caption_source": "sections/1_introduction.tex",
        "caption_require": ("Target task and output", "problem illustration", "not result evidence"),
    },
    "framework": {
        "caption_source": "sections/3_method.tex",
        "caption_require": ("Provisional method contract", "synchronization", "frozen training code"),
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add(checks: list[dict], check_id: str, status: str, detail: object = None) -> None:
    record: dict[str, object] = {"id": check_id, "status": status}
    if detail is not None:
        record["detail"] = detail
    checks.append(record)


def drawio_diagnostics(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return [f"xml:{exc}"], []
    if root.tag != "mxfile":
        errors.append("root is not mxfile")
    diagrams = root.findall("diagram")
    if len(diagrams) != 1:
        errors.append(f"expected one diagram; found {len(diagrams)}")
    models = root.findall(".//mxGraphModel")
    if len(models) != 1:
        errors.append(f"expected one mxGraphModel; found {len(models)}")
    cells = root.findall(".//mxCell")
    ids = [cell.get("id") for cell in cells]
    if any(item is None or item == "" for item in ids):
        errors.append("mxCell without id")
    if len(ids) != len(set(ids)):
        errors.append("duplicate mxCell id")
    id_set = {item for item in ids if item}
    for cell in cells:
        parent = cell.get("parent")
        if parent and parent not in id_set:
            errors.append(f"missing parent for cell {cell.get('id')}")
        if cell.get("edge") == "1":
            source, target = cell.get("source"), cell.get("target")
            if source and source not in id_set:
                errors.append(f"missing edge source for {cell.get('id')}")
            if target and target not in id_set:
                errors.append(f"missing edge target for {cell.get('id')}")
        if cell.get("vertex") == "1":
            geometry = cell.find("mxGeometry")
            if geometry is None:
                errors.append(f"vertex {cell.get('id')} has no geometry")
            else:
                try:
                    width = float(geometry.get("width", "0"))
                    height = float(geometry.get("height", "0"))
                    if width <= 0 or height <= 0:
                        errors.append(f"vertex {cell.get('id')} has non-positive geometry")
                except ValueError:
                    errors.append(f"vertex {cell.get('id')} has invalid geometry")
    return sorted(set(errors)), sorted(set(warnings))


def svg_diagnostics(path: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return [f"xml:{exc}"], []
    if not root.tag.endswith("svg"):
        errors.append("root is not svg")
    if not root.get("viewBox"):
        errors.append("missing viewBox")
    if not root.get("width") or not root.get("height"):
        errors.append("missing width/height")
    return errors, warnings


def pdf_is_vector(path: Path) -> tuple[bool, str]:
    try:
        output = subprocess.check_output(["pdfimages", "-list", str(path)], text=True, errors="replace")
    except Exception as exc:
        return False, f"pdfimages:{type(exc).__name__}"
    image_rows = [line for line in output.splitlines()[2:] if line.strip()]
    return not image_rows, f"embedded_raster_count={len(image_rows)}"


def caption_block(text: str, label: str) -> str:
    match = re.search(r"\\caption\{(.*?)\}\s*\\label\{" + re.escape(label) + r"\}", text, re.DOTALL)
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
    visual_path = args.visual_review.resolve() if args.visual_review else paper_dir / ".aris/figure-visual-review.json"
    checks: list[dict] = []
    figure_hashes: dict[str, dict[str, str]] = {}

    for name, spec in FIGURES.items():
        paths = {suffix: paper_dir / "figures" / f"{name}.{suffix}" for suffix in ("drawio", "svg", "pdf")}
        complete = all(path.is_file() for path in paths.values())
        add(checks, f"triplet_present:{name}", "PASS" if complete else "FAIL")
        if not complete:
            continue
        figure_hashes[name] = {suffix: sha256(path) for suffix, path in paths.items()}
        drawio_errors, drawio_warnings = drawio_diagnostics(paths["drawio"])
        svg_errors, svg_warnings = svg_diagnostics(paths["svg"])
        vector, vector_detail = pdf_is_vector(paths["pdf"])
        add(checks, f"drawio_xml_errors:{name}", "PASS" if not drawio_errors else "FAIL", drawio_errors)
        add(checks, f"drawio_xml_warnings:{name}", "PASS" if not drawio_warnings else "FAIL", drawio_warnings)
        add(checks, f"svg_xml_errors:{name}", "PASS" if not svg_errors else "FAIL", svg_errors)
        add(checks, f"svg_xml_warnings:{name}", "PASS" if not svg_warnings else "FAIL", svg_warnings)
        add(checks, f"vector_pdf:{name}", "PASS" if vector else "FAIL", vector_detail)
        source = (paper_dir / spec["caption_source"]).read_text(encoding="utf-8", errors="replace")
        caption = caption_block(source, f"fig:{name}")
        missing_caption_terms = [term for term in spec["caption_require"] if term.casefold() not in caption.casefold()]
        add(checks, f"conceptual_caption:{name}", "PASS" if caption and not missing_caption_terms else "FAIL", missing_caption_terms)

    narrative = (root / "NARRATIVE_REPORT.md").read_text(encoding="utf-8", errors="replace")
    method_contract = (paper_dir / "evidence/method_contract.yaml").read_text(encoding="utf-8", errors="replace")
    ppt_names = (
        "Multi-Person-Tempo-Adaptive-Counting-Introduction-Rounded-TNR.pptx",
        "Multi-Person-Tempo-Adaptive-Counting-Framework-Rounded-TNR.pptx",
    )
    classification_ok = (
        all(name in narrative and name in method_contract for name in ppt_names)
        and "visual_reference_only" in narrative
        and bool(re.search(r"source_class:\s*committed_ppt.*?authority:\s*visual_reference_only", method_contract, re.S))
        and "conceptual_only" not in narrative
    )
    add(checks, "ppt_visual_reference_only", "PASS" if classification_ok else "FAIL")

    visual_review: dict = {}
    if visual_path.is_file():
        try:
            visual_review = json.loads(visual_path.read_text(encoding="utf-8"))
        except Exception as exc:
            add(checks, "visual_review_parse", "FAIL", type(exc).__name__)
        else:
            add(checks, "visual_review_parse", "PASS")
            add(checks, "standalone_visual_review", "PASS" if visual_review.get("standalone_review", {}).get("verdict") == "PASS" else "FAIL")
            bound = visual_review.get("figure_hashes", {})
            add(checks, "visual_review_figure_digest_binding", "PASS" if bound == figure_hashes else "FAIL")
            page_review = visual_review.get("pdf_page_review", {})
            pdf = paper_dir / "main.pdf"
            render_records = page_review.get("renders", [])
            render_ok = bool(render_records)
            for record in render_records:
                render = root / record.get("path", "")
                render_ok = render_ok and render.is_file() and sha256(render) == record.get("sha256")
            pdf_ok = pdf.is_file() and page_review.get("pdf_sha256") == sha256(pdf)
            page_pass = page_review.get("verdict") == "PASS" and render_ok and pdf_ok
            add(checks, "pdf_page_visual_review_digest_binding", "PASS" if page_pass else "BLOCKED", {
                "review_verdict": page_review.get("verdict"), "render_count": len(render_records), "pdf_bound": pdf_ok,
            })
    else:
        add(checks, "visual_review_present", "BLOCKED")

    statuses = {item["status"] for item in checks}
    if "FAIL" in statuses:
        verdict = "FAIL"
    elif "BLOCKED" in statuses:
        verdict = "PROVISIONAL" if args.mode == "draft" else "BLOCKED"
    else:
        verdict = "PASS"
    report = {
        "schema_version": 1,
        "mode": args.mode,
        "verdict": verdict,
        "figure_hashes": figure_hashes,
        "visual_review_artifact": visual_path.relative_to(root).as_posix() if root in visual_path.parents else str(visual_path),
        "checks": checks,
    }
    output = args.output or paper_dir / ".aris/figure-validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"FIGURE_VALIDATION={verdict}; checks={len(checks)}")
    return 0 if verdict in {"PASS", "PROVISIONAL"} else (2 if verdict == "BLOCKED" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
