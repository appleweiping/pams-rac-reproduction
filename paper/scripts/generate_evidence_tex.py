#!/usr/bin/env python3
"""Generate ``evidence_values.tex`` only from an eligible frozen bundle.

Draft mode is deliberately a no-op while evidence is pending. Submission mode
fails closed unless the manifest, audits, and every JSON/CSV input hash pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from evidence_common import (
    emit_report,
    json_load,
    resolve_inside,
    safe_latex_scalar,
    sha256,
    source_value,
    stable_rel,
)
from validate_results_manifest import validate_manifest


def braced(command: str, key: str | None, value: str) -> str:
    return f"\\{command}{{{value}}}" if key is None else f"\\{command}{{{key}}}{{{value}}}"


def _resolved_sources(data: dict[str, Any], root: Path) -> dict[str, tuple[dict[str, Any], Path]]:
    sources: dict[str, tuple[dict[str, Any], Path]] = {}
    for item in data.get("source_artifacts", []):
        sources[item["id"]] = (item, resolve_inside(root, item["path"]))
    return sources


def _binding_value(binding: dict[str, Any], sources: dict[str, tuple[dict[str, Any], Path]]) -> str:
    source, path = sources[binding["source_id"]]
    return safe_latex_scalar(source_value(path, source["format"], binding["locator"]))


def render_payload(data: dict[str, Any], root: Path, manifest: Path) -> str:
    sources = _resolved_sources(data, root)
    bindings = data["paper_bindings"]
    lines = [
        "% GENERATED FILE -- DO NOT EDIT BY HAND",
        "% Values below were resolved from frozen JSON/CSV inputs.",
        f"% artifact_id: {data['artifact_id']}",
        f"% results_manifest: {stable_rel(manifest, root)} {sha256(manifest)}",
    ]
    for source_id in sorted(sources):
        source, path = sources[source_id]
        lines.append(f"% input: {source_id} {stable_rel(path, root)} {source['sha256']}")
    command_by_group = {
        "protocol_fields": "DeclareProtocolField",
        "results": "DeclareResult",
        "claims": "DeclareResultClaim",
    }
    for group in ("protocol_fields", "results", "claims"):
        for binding in sorted(bindings.get(group, []), key=lambda item: item["key"]):
            lines.append(braced(command_by_group[group], binding["key"], _binding_value(binding, sources)))
    abstract = bindings["abstract_result"]
    lines.append(braced("DeclareAbstractResult", None, _binding_value(abstract, sources)))
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def run(
    manifest: Path,
    output: Path,
    root: Path,
    mode: str,
    *,
    verify_only: bool = False,
    receipt: Path | None = None,
) -> tuple[dict[str, Any], int]:
    try:
        data = json_load(manifest)
    except Exception as exc:
        return {"gate": "evidence_generation", "mode": mode, "status": "FAIL", "reason": type(exc).__name__}, 1

    if data.get("status") != "frozen-eligible":
        stale = output.is_file()
        if stale:
            return {
                "gate": "evidence_generation",
                "mode": mode,
                "status": "FAIL",
                "reason": "generated evidence file exists while manifest is not frozen-eligible",
            }, 1
        if mode == "draft":
            return {
                "gate": "evidence_generation",
                "mode": mode,
                "status": "PROVISIONAL",
                "reason": "data-pending; no evidence_values.tex generated",
                "output_present": False,
            }, 0
        return {
            "gate": "evidence_generation",
            "mode": mode,
            "status": "FAIL",
            "reason": "submission requires a frozen-eligible results manifest",
        }, 1

    validation, validation_code = validate_manifest(manifest, root, "submission")
    if validation_code:
        return {
            "gate": "evidence_generation",
            "mode": mode,
            "status": "FAIL",
            "reason": "results eligibility validation failed",
            "validation": validation,
        }, 1
    try:
        payload = render_payload(data, root, manifest)
    except Exception as exc:
        return {"gate": "evidence_generation", "mode": mode, "status": "FAIL", "reason": type(exc).__name__}, 1

    if verify_only:
        if not output.is_file() or output.read_text(encoding="utf-8") != payload:
            return {
                "gate": "evidence_generation",
                "mode": mode,
                "status": "FAIL",
                "reason": "existing evidence_values.tex does not exactly match frozen inputs",
            }, 1
    else:
        _atomic_write(output, payload)

    provenance = {
        "schema_version": "1.0",
        "artifact_id": data["artifact_id"],
        "manifest": stable_rel(manifest, root),
        "manifest_sha256": sha256(manifest),
        "inputs": [
            {"id": item["id"], "path": item["path"], "sha256": item["sha256"]}
            for item in sorted(data["source_artifacts"], key=lambda value: value["id"])
        ],
        "output": stable_rel(output, root),
        "output_sha256": "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "generator_sha256": sha256(Path(__file__).resolve()),
    }
    if receipt is not None and not verify_only:
        _atomic_write(receipt, json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return {
        "gate": "evidence_generation",
        "mode": mode,
        "status": "PASS",
        "verified_only": verify_only,
        "provenance": provenance,
    }, 0


def main() -> int:
    parser = argparse.ArgumentParser()
    paper_dir = Path(__file__).resolve().parents[1]
    parser.add_argument("manifest", type=Path, nargs="?", default=paper_dir / "evidence/results_manifest.json")
    parser.add_argument("output", type=Path, nargs="?", default=paper_dir / "generated/evidence_values.tex")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--mode", choices=("draft", "submission"), default="draft")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    root = (args.root or manifest.parents[2]).resolve()
    report, code = run(
        manifest,
        output,
        root,
        args.mode,
        verify_only=args.verify_only,
        receipt=args.receipt.resolve() if args.receipt else None,
    )
    emit_report(report, args.report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
