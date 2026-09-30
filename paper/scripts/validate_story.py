#!/usr/bin/env python3
"""Validate the canonical story, title, abstract, maps, notation, and order."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader


CANONICAL_STORY = {
    "what": "pose-driven per-person repetition counting from identity-indexed masked tracks",
    "why": "people can act asynchronously while tempo changes within each identity trajectory",
    "so_what": "track-indexed window-local routing separates people and adapts temporal scale without person-wise count or period supervision in the intended counting objective",
}
CORE_CLAIM = re.compile(r"^C\d{2}$")
NOTATION_FALLBACK = {
    "lag": re.compile(r"lag grid\s+\\\(\\delta", re.I),
    "routing_weights": re.compile(r"\\bg_\{i\\ell\}\s*(?:&\s*)?=", re.S),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_ids(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def normalize_text(text: str) -> str:
    text = text.replace("\\", " ")
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    return " ".join(text.split())


def strip_tex_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def add(checks: list[dict], check_id: str, passed: bool, detail: object = None) -> None:
    record: dict[str, object] = {"id": check_id, "status": "PASS" if passed else "FAIL"}
    if detail is not None:
        record["detail"] = detail
    checks.append(record)


def subsection_blocks(text: str) -> list[tuple[str, str]]:
    matches = list(re.finditer(r"\\subsection\{([^}]+)\}", text))
    return [
        (match.group(1), text[match.end(): matches[index + 1].start() if index + 1 < len(matches) else len(text)])
        for index, match in enumerate(matches)
    ]


def equation_has_terminal_punctuation(body: str) -> bool:
    body = re.sub(r"\\label\{[^}]+\}", "", body)
    body = re.sub(r"\\end\{(?:aligned|gathered|split)\}", "", body)
    body = re.sub(r"\\(?:quad|qquad|,|;|!)\s*$", "", body.strip())
    return bool(re.search(r"[.,;:]\s*$", body))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--pdf", type=Path)
    parser.add_argument("--pdf-text", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paper_dir = args.paper_dir.resolve()
    root = paper_dir.parent

    story_path = paper_dir / "evidence/story_contract.json"
    notation_path = paper_dir / "evidence/notation_inventory.json"
    story = json.loads(story_path.read_text(encoding="utf-8"))
    notation = json.loads(notation_path.read_text(encoding="utf-8"))
    intro = strip_tex_comments((paper_dir / "sections/1_introduction.tex").read_text(encoding="utf-8"))
    abstract = strip_tex_comments((paper_dir / "sections/0_abstract.tex").read_text(encoding="utf-8"))
    method = strip_tex_comments((paper_dir / "sections/3_method.tex").read_text(encoding="utf-8"))
    main_tex = strip_tex_comments((paper_dir / "main.tex").read_text(encoding="utf-8"))
    checks: list[dict] = []

    add(checks, "story_contract_schema", story.get("schema_version") == "2.0")
    add(checks, "canonical_what_why_so_what_keys", set(story.get("story", {})) == {"what", "why", "so_what"})
    for key, expected in CANONICAL_STORY.items():
        add(checks, f"canonical_story:{key}", story.get("story", {}).get(key) == expected)
        anchors = story.get("canonical_source_anchors", {}).get(key, [])
        add(checks, f"canonical_story_anchors_declared:{key}", bool(anchors))
        for index, anchor in enumerate(anchors, start=1):
            path = paper_dir / anchor.get("path", "")
            found = path.is_file() and normalize_text(anchor.get("text", "")).casefold() in normalize_text(strip_tex_comments(path.read_text(encoding="utf-8", errors="replace"))).casefold()
            add(checks, f"canonical_story_anchor:{key}:{index}", bool(found), anchor.get("path"))

    title_match = re.search(r"\\title\{(.*?)\}", main_tex, re.DOTALL)
    source_title = normalize_text(title_match.group(1) if title_match else "")
    expected_title = normalize_text(story["working_title"])
    add(checks, "exact_source_title", source_title == expected_title, {"observed": source_title})

    pdf = (args.pdf or paper_dir / "main.pdf").resolve()
    pdf_text_path = args.pdf_text.resolve() if args.pdf_text else None
    pdf_text = ""
    pdf_record: dict[str, object] = {"path": pdf.name, "present": pdf.is_file()}
    if pdf.is_file():
        pdf_record["sha256"] = sha256(pdf)
        if pdf_text_path and pdf_text_path.is_file():
            pdf_text = pdf_text_path.read_text(encoding="utf-8", errors="replace")
            pdf_record["extracted_text_path"] = str(pdf_text_path)
            pdf_record["extracted_text_sha256"] = sha256(pdf_text_path)
        else:
            pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(str(pdf)).pages)
            pdf_record["extractor"] = "pypdf"
    normalized_pdf = normalize_text(pdf_text)
    add(checks, "exact_extracted_pdf_title", bool(pdf_text) and expected_title in normalized_pdf,
        {"pdf_present": pdf.is_file(), "text_present": bool(pdf_text)})

    controlled_count = len(re.findall(r"\\ControlledAbstractResult\b", abstract))
    declared_count = len(re.findall(r"\\DeclareAbstractResult\b", abstract))
    unresolved_result_macros = re.findall(r"\\(?:R|C|Protocol)\{[^}]+\}", abstract)
    draft_numbers = re.findall(r"(?<![A-Za-z\\])\d+(?:\.\d+)?(?:\s*%|\s*(?:\u00b1|\\pm)\s*\d+(?:\.\d+)?)?", abstract)
    add(checks, "abstract_exactly_one_controlled_placeholder", controlled_count == 1, controlled_count)
    add(checks, "abstract_no_declared_result", declared_count == 0, declared_count)
    add(checks, "abstract_no_result_binding_macros", not unresolved_result_macros, unresolved_result_macros)
    add(checks, "abstract_no_draft_result_numbers", not draft_numbers, draft_numbers)

    lower_intro = normalize_text(intro).casefold()
    for phrase in story["required_mechanism_qualifiers"]:
        add(checks, f"intro_qualifier:{phrase}", phrase.casefold() in lower_intro)

    contribution_section = re.search(
        r"The paper makes three falsifiable.*?\\begin\{itemize\}(.*?)\\end\{itemize\}",
        intro,
        re.DOTALL,
    )
    contribution_text = contribution_section.group(1) if contribution_section else ""
    contribution_items = re.findall(r"\\item\b(.*?)(?=\\item\b|\Z)", contribution_text, re.DOTALL)
    contribution_markers = [re.findall(r"\\ClaimMap\{([^}]+)\}", item) for item in contribution_items]
    expected_contributions = [
        ids for _, ids in sorted(story["authoritative_contribution_map"].items(), key=lambda item: int(item[0]))
    ]
    observed_contributions = [normalize_ids(markers[0]) if len(markers) == 1 else [] for markers in contribution_markers]
    add(checks, "exactly_three_contribution_bullets", len(contribution_items) == 3, len(contribution_items))
    add(checks, "one_claim_map_per_contribution", len(contribution_markers) == 3 and all(len(item) == 1 for item in contribution_markers))
    add(checks, "contribution_claim_map_exact", observed_contributions == expected_contributions,
        {"observed": observed_contributions, "expected": expected_contributions})
    flattened = [claim for claims in expected_contributions for claim in claims]
    add(checks, "contribution_claim_ids_well_formed_unique", len(flattened) == len(set(flattened)) and all(CORE_CLAIM.fullmatch(item) for item in flattened))
    ledger_ids = set(re.findall(r"\bC\d{2}\b", (root / "CLAIMS_EVIDENCE_MATRIX.md").read_text(encoding="utf-8", errors="replace")))
    add(checks, "contribution_claim_ids_resolve_to_ledger", set(flattened).issubset(ledger_ids), sorted(set(flattened) - ledger_ids))

    mapped_locators = set(story["experiment_claim_map"])
    observed_locators: set[str] = set()
    for rel in ("sections/4_experiments.tex", "appendix/a_experiment_contract.tex"):
        text = strip_tex_comments((paper_dir / rel).read_text(encoding="utf-8"))
        for heading, block in subsection_blocks(text):
            locator = f"{rel}#{heading}"
            observed_locators.add(locator)
            markers = re.findall(r"\\ClaimMap\{([^}]+)\}", block)
            expected = story["experiment_claim_map"].get(locator, [])
            observed = normalize_ids(markers[0]) if len(markers) == 1 else []
            add(checks, f"experiment_subsection_single_map:{locator}", len(markers) == 1, len(markers))
            add(checks, f"experiment_claim_map:{locator}", observed == expected,
                {"observed": observed, "expected": expected})
    add(checks, "experiment_subsection_map_complete_one_to_one", observed_locators == mapped_locators,
        {"unmapped": sorted(observed_locators - mapped_locators), "orphan_contract_entries": sorted(mapped_locators - observed_locators)})
    experiment_ids = {claim for claims in story["experiment_claim_map"].values() for claim in claims}
    add(checks, "experiment_claim_ids_resolve_to_ledger", experiment_ids.issubset(ledger_ids), sorted(experiment_ids - ledger_ids))

    method_headings = re.findall(r"\\subsection\{([^}]+)\}", method)
    add(checks, "method_subsection_order", method_headings == story["method_subsection_order"],
        {"observed": method_headings, "expected": story["method_subsection_order"]})
    equation_labels = re.findall(r"\\label\{(eq:[^}]+)\}", method)
    core_observed = [item for item in equation_labels if item in story["core_equation_order"]]
    add(checks, "core_equation_order", core_observed == story["core_equation_order"],
        {"observed": core_observed, "expected": story["core_equation_order"]})
    equation_bodies = re.findall(r"\\begin\{equation\}(.*?)\\end\{equation\}", method, re.DOTALL)
    punctuated = [index + 1 for index, body in enumerate(equation_bodies) if equation_has_terminal_punctuation(body)]
    add(checks, "display_equations_no_terminal_punctuation", not punctuated, punctuated)
    forbidden_window_sum = re.search(
        r"\\sum[^\n]{0,160}(?:round(?:ed|ing)?|integer)[^\n]{0,80}window|"
        r"(?:round(?:ed|ing)?|integer)[^\n]{0,80}window[^\n]{0,160}\\sum",
        method,
        re.I,
    )
    add(checks, "no_final_sum_of_independently_rounded_windows", forbidden_window_sum is None)

    for symbol in notation["symbols"]:
        exact = normalize_text(symbol["definition_text"]) in normalize_text(method)
        fallback = NOTATION_FALLBACK.get(symbol["id"])
        add(checks, f"notation:{symbol['id']}", exact or bool(fallback and fallback.search(method)))

    freeze = json.loads((paper_dir / "evidence/freeze_inventory.json").read_text(encoding="utf-8"))
    for rel in freeze["required_consistent_freeze_documents"]:
        path = root / rel
        add(checks, f"freeze_field:{rel}", path.is_file() and freeze["base_commit"] in path.read_text(encoding="utf-8", errors="replace"))

    verdict = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    source_paths = [story_path, notation_path, paper_dir / "main.tex", paper_dir / "sections/0_abstract.tex", paper_dir / "sections/1_introduction.tex", paper_dir / "sections/3_method.tex", paper_dir / "sections/4_experiments.tex", paper_dir / "appendix/a_experiment_contract.tex"]
    report = {
        "schema_version": 2,
        "verdict": verdict,
        "canonical_story": story["story"],
        "source_hashes": {path.relative_to(root).as_posix(): sha256(path) for path in source_paths},
        "pdf_binding": pdf_record,
        "checks": checks,
    }
    output = args.output or paper_dir / ".aris/story-check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"STORY_CHECK={verdict}; checks={len(checks)}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
