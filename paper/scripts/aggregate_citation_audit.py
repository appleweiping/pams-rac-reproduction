#!/usr/bin/env python3
"""Aggregate independently reviewed citation entries into ARIS audit artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from validate_delivery_freeze import source_freeze


CITE_RE = re.compile(r"\\cite\w*\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]+)\}")


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def exact_context(text: str, start: int, end: int) -> str:
    """Return the exact logical paragraph containing one citation command."""
    left = text.rfind("\n\n", 0, start)
    right = text.find("\n\n", end)
    left = 0 if left < 0 else left + 2
    right = len(text) if right < 0 else right
    return text[left:right]


def reviewed_contexts_match(
    expected_rows: list[dict[str, object]], reviewed: object
) -> bool:
    expected = {
        (item["file"], item["line"], item["context_sha256"])
        for item in expected_rows
    }
    if not isinstance(reviewed, list) or not reviewed:
        return False
    observed: set[tuple[object, object, object]] = set()
    for row in reviewed:
        if not isinstance(row, dict) or set(row) != {"file", "line", "context_sha256", "verdict"}:
            return False
        if row.get("verdict") != "SUPPORTS":
            return False
        observed.add((row.get("file"), row.get("line"), row.get("context_sha256")))
    return len(observed) == len(reviewed) and observed == expected


def citation_sites(paper_dir: Path) -> tuple[dict[str, list[dict[str, object]]], list[Path]]:
    sites: dict[str, list[dict[str, object]]] = {}
    bearing: list[Path] = []
    for path in sorted(paper_dir.rglob("*.tex")):
        if any(part in {"vendor", ".aris"} for part in path.relative_to(paper_dir).parts):
            continue
        text = path.read_text(encoding="utf-8")
        found = False
        for match in CITE_RE.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            context = exact_context(text, match.start(), match.end())
            context_digest = "sha256:" + hashlib.sha256(context.encode("utf-8")).hexdigest()
            for key in (item.strip() for item in match.group(1).split(",")):
                if key:
                    sites.setdefault(key, []).append(
                        {
                            "file": path.relative_to(paper_dir).as_posix(),
                            "line": line,
                            "context_sha256": context_digest,
                        }
                    )
                    found = True
        if found:
            bearing.append(path)
    return sites, bearing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--context-template", type=Path)
    args = parser.parse_args()
    paper_dir = args.paper_dir.resolve()
    entry_dir = paper_dir / ".aris" / "citation-audit" / "entries"
    sites, bearing = citation_sites(paper_dir)
    cited_keys = sorted(sites)

    freeze_path = paper_dir / "evidence" / "freeze_inventory.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    paper_freeze, freeze_records = source_freeze(paper_dir.parent, freeze)

    entries: list[dict[str, object]] = []
    missing: list[str] = []
    invalid_contexts: list[str] = []
    invalid_traces: list[str] = []
    for key in cited_keys:
        path = entry_dir / f"{key}.json"
        if not path.is_file():
            missing.append(key)
            continue
        source = json.loads(path.read_text(encoding="utf-8"))
        trace = paper_dir / ".aris" / "traces" / "citation-audit" / key / "reviewer.md"
        trace_ok = trace.is_file() and trace.stat().st_size > 0
        if not trace_ok:
            invalid_traces.append(key)
        reviewed = source.get("reviewed_contexts")
        context_ok = reviewed_contexts_match(sites[key], reviewed)
        if not context_ok:
            invalid_contexts.append(key)
        post = source.get("post_fix_status")
        current = "KEEP" if post == "PASS" and context_ok and trace_ok else "FIX"
        entries.append(
            {
                "key": key,
                "verdict": current,
                "initial_verdict": source.get("verdict"),
                "post_fix_status": post,
                "axis_failures": [] if current == "KEEP" else [
                    *([] if post == "PASS" else ["UNRESOLVED"]),
                    *([] if context_ok else ["EXACT_CONTEXT_BINDING"]),
                    *([] if trace_ok else ["REVIEW_TRACE"]),
                ],
                "contexts": [
                    {**item, "verdict": "SUPPORTS"}
                    for item in sites[key]
                ] if context_ok else sites[key],
                "entry_path": path.relative_to(paper_dir).as_posix(),
                "entry_sha256": sha256(path),
                "review_trace": trace.relative_to(paper_dir).as_posix() if trace_ok else None,
                "review_trace_sha256": sha256(trace) if trace_ok else None,
                "exact_context_binding": context_ok,
                "note": source.get("fix", {}).get("description", "Verified without change."),
                "confidence": source.get("confidence"),
                "primary_urls": source.get("primary_urls", []),
            }
        )

    unresolved = [item["key"] for item in entries if item["verdict"] != "KEEP"]
    substantive_unresolved = [
        item["key"] for item in entries if item["post_fix_status"] != "PASS"
    ]
    if missing:
        verdict, reason = "BLOCKED", "entry_reviews_missing"
        summary = f"Missing fresh per-entry reviews for {len(missing)} cited keys."
    elif substantive_unresolved:
        verdict, reason = "FAIL", "unresolved_entry_findings"
        summary = (
            f"Substantive citation findings remain unresolved for {len(substantive_unresolved)} cited keys; "
            f"exact-context failures={len(invalid_contexts)}, trace failures={len(invalid_traces)}."
        )
    elif invalid_contexts or invalid_traces:
        verdict, reason = "BLOCKED", "exact_context_or_trace_bindings_missing"
        summary = (
            "All per-entry substantive reviews are PASS, but submission remains blocked until "
            f"the reviewed entries bind the exact current contexts ({len(invalid_contexts)}) "
            f"and traces ({len(invalid_traces)})."
        )
    else:
        verdict, reason = "PASS", "all_entries_keep"
        summary = f"All {len(entries)} cited entries pass existence, final-metadata, and context checks after recorded fixes."

    inputs = [paper_dir / "references.bib", paper_dir / "main.tex", *bearing]
    hashes = {
        path.relative_to(paper_dir).as_posix(): sha256(path)
        for path in dict.fromkeys(inputs)
    }
    artifact = {
        "audit_skill": "citation-audit",
        "verdict": verdict,
        "reason_code": reason,
        "summary": summary,
        "paper_freeze": paper_freeze,
        "source_freeze_sha256": paper_freeze,
        "source_freeze_algorithm": "sha256-path-digest-list-v1",
        "source_freeze_file_count": len(freeze_records),
        "audited_input_hashes": hashes,
        "trace_path": ".aris/traces/citation-audit/",
        "agent_id": "multi-agent:one-fresh-reviewer-per-entry",
        "executor_model": "codex-gpt-5.6-sol",
        "executor_family": "openai",
        "reviewer_model": "gpt-5.6-sol",
        "reviewer_family": "openai",
        "review_independence": "same-family",
        "acceptance_status": "provisional",
        "reviewer_reasoning": "xhigh",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "details": {
            "total_entries": len(entries),
            "cited_keys": cited_keys,
            "missing_entries": missing,
            "unresolved_entries": unresolved,
            "invalid_exact_context_entries": invalid_contexts,
            "invalid_trace_entries": invalid_traces,
            "per_entry": entries,
        },
    }
    (paper_dir / "CITATION_AUDIT.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    template_path = args.context_template or (
        paper_dir / ".aris" / "citation-audit" / "context-binding-template.json"
    )
    context_template = {
        "schema_version": "1.0",
        "paper_freeze": paper_freeze,
        "instruction": (
            "After confirming that the recorded fresh reviewer assessed these exact current "
            "contexts, copy each reviewed_contexts array into the corresponding entry JSON. "
            "Do not auto-approve after a context change."
        ),
        "entries": {
            item["key"]: {
                "entry_path": item["entry_path"],
                "entry_sha256_before_binding": item["entry_sha256"],
                "review_trace": item["review_trace"],
                "review_trace_sha256": item["review_trace_sha256"],
                "reviewed_contexts": [
                    {**context, "verdict": "SUPPORTS"}
                    for context in sites[item["key"]]
                ],
            }
            for item in entries
        },
    }
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(
        json.dumps(context_template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    lines = [
        "# Citation Audit",
        "",
        f"**Verdict:** `{verdict}` (`{reason}`; same-family/provisional)",
        "",
        summary,
        "",
        "Each cited key was assigned to a separate fresh zero-context reviewer. "
        "The ledger records any initial repair while reporting the post-fix state.",
        "",
        "| Key | Initial | Current | Context sites | Confidence |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in entries:
        lines.append(
            f"| `{item['key']}` | {item['initial_verdict']} | {item['verdict']} | "
            f"{len(item['contexts'])} | {item['confidence']} |"
        )
    if missing:
        lines.extend(["", "Missing reviews: " + ", ".join(f"`{key}`" for key in missing)])
    lines.extend(
        [
            "",
            "## Scope and provenance",
            "",
            "The audit checks citation existence, version-of-record metadata, and every "
            "citing context. Google Scholar was not used as the metadata authority; "
            "publisher, CVF, Crossref, DBLP, and official project records were preferred. "
            "Per-entry reviewer traces are under `.aris/traces/citation-audit/`.",
            "",
            "The Codex-native route is same-family and therefore provisional even when "
            "the substantive verdict is `PASS`.",
        ]
    )
    (paper_dir / "CITATION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"CITATION_AUDIT={verdict}; cited={len(cited_keys)}; missing={len(missing)}; unresolved={len(unresolved)}")
    return 0 if verdict == "PASS" else (2 if verdict == "BLOCKED" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
