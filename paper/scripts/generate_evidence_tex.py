#!/usr/bin/env python3
"""Verify and reproduce LaTeX bindings from an audited result manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def braced(command: str, key: str | None, value: str) -> str:
    if key is None:
        return f"\\{command}{{{value}}}"
    return f"\\{command}{{{key}}}{{{value}}}"


def render_payload(data: dict) -> str:
    bindings = data.get("paper_bindings", {})
    abstract_result = bindings.get("abstract_result", {}).get("sentence")
    if not isinstance(abstract_result, str) or not abstract_result.strip():
        raise ValueError("Manifest must contain exactly one non-empty abstract_result")
    lines = [
        "% GENERATED FILE -- DO NOT EDIT BY HAND",
        "% exact bytes are bound by the package manifest and external delivery receipt",
        f"% result_artifact_id: {data['artifact_id']}",
    ]
    for binding in sorted(bindings.get("protocol_fields", []), key=lambda item: item["key"]):
        lines.append(braced("DeclareProtocolField", binding["key"], str(binding["value"])))
    for binding in sorted(bindings.get("results", []), key=lambda item: item["key"]):
        lines.append(braced("DeclareResult", binding["key"], str(binding["formatted"])))
    for binding in sorted(bindings.get("claims", []), key=lambda item: item["key"]):
        lines.append(braced("DeclareResultClaim", binding["key"], str(binding["text"])))
    lines.append(braced("DeclareAbstractResult", None, abstract_result.strip()))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--external-receipt", type=Path, required=True)
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    bundle_root = (args.bundle_root or args.manifest.parent).resolve()
    checks = (
        [sys.executable, str(scripts_dir / "validate_result_manifest.py"), str(args.manifest), "--bundle-root", str(bundle_root)],
        [
            sys.executable,
            str(scripts_dir / "evidence_precheck.py"),
            str(args.manifest),
            "--bundle-root",
            str(bundle_root),
            "--external-receipt",
            str(args.external_receipt.resolve()),
        ],
    )
    for command in checks:
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise SystemExit(f"Refusing generation; prerequisite failed with exit {completed.returncode}: {command[1]}")

    data = json.loads(args.manifest.read_text(encoding="utf-8"))
    audit = data.get("audits", {})
    required = ("experiment_audit", "result_to_claim", "paper_claim_audit")
    failed = [name for name in required if audit.get(name, {}).get("verdict") != "PASS"]
    if failed:
        raise SystemExit("Refusing generation; audits not PASS: " + ", ".join(failed))

    try:
        payload = render_payload(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    # A passing precheck means every package byte is already frozen.  Rewriting a
    # missing or different output here would invalidate that freeze, so this
    # command is deliberately a deterministic no-op verifier at submission time.
    if not args.output.is_file():
        raise SystemExit("Refusing generation into an audited bundle; output must already exist and be digest-bound")
    if args.output.read_text(encoding="utf-8") != payload:
        raise SystemExit("Refusing to mutate an audited bundle; deterministic output differs from the frozen file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
