#!/usr/bin/env python3
"""Fail-closed validator for the source freeze, package manifest, and receipt."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path, PurePosixPath


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_FREEZE_FIELDS = {
    "schema_version",
    "base_commit",
    "external_delivery_receipt",
    "delivery_receipt_required_fields",
    "delivery_receipt_schema_version",
    "package_manifest",
    "package_manifest_excludes",
    "manifest_scope_roots",
    "source_freeze_roots",
    "source_freeze_excludes",
    "required_consistent_freeze_documents",
    "anonymity_release_globs",
    "clean_tree_required_after_delivery_commit",
    "current_status",
}
EXPECTED_RECEIPT_FIELDS = {
    "schema_version",
    "delivery_commit",
    "clean_status",
    "package_manifest",
    "package_manifest_sha256",
    "source_freeze_sha256",
    "created_at",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), *args], stderr=subprocess.STDOUT
    ).decode("utf-8", errors="replace")


def path_in_roots(path: str, roots: list[str]) -> bool:
    return any(path == root.rstrip("/") or path.startswith(root.rstrip("/") + "/") for root in roots)


def is_excluded(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def repository_files(root: Path, include_untracked: bool) -> list[str]:
    args = ["ls-files", "--cached"]
    if include_untracked:
        args.extend(["--others", "--exclude-standard"])
    raw = subprocess.check_output(["git", "-C", str(root), *args, "-z"])
    return sorted({item.decode("utf-8").replace("\\", "/") for item in raw.split(b"\0") if item})


def source_freeze(root: Path, freeze: dict) -> tuple[str, list[dict[str, str]]]:
    candidates = repository_files(root, include_untracked=True)
    selected = [
        item for item in candidates
        if path_in_roots(item, freeze["source_freeze_roots"])
        and not is_excluded(item, freeze["source_freeze_excludes"])
        and (root / item).is_file()
    ]
    records = [{"path": item, "sha256": sha256(root / item)} for item in selected]
    payload = "".join(f"{item['path']}\0{item['sha256']}\n" for item in records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), records


def parse_manifest(path: Path) -> tuple[dict[str, str], list[str]]:
    entries: dict[str, str] = {}
    errors: list[str] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([0-9a-f]{64})\s+[ *]?(.+)", line)
        if not match:
            errors.append(f"line {line_no}: malformed manifest entry")
            continue
        digest, rel = match.groups()
        rel = rel.replace("\\", "/")
        posix = PurePosixPath(rel)
        if posix.is_absolute() or ".." in posix.parts or rel.startswith("./"):
            errors.append(f"line {line_no}: unsafe/non-canonical path")
            continue
        if rel in entries:
            errors.append(f"line {line_no}: duplicate path {rel}")
            continue
        entries[rel] = digest
    return entries, errors


def add(checks: list[dict], check_id: str, passed: bool, *, blocked: bool = False, detail: object = None) -> None:
    status = "PASS" if passed else ("BLOCKED" if blocked else "FAIL")
    record: dict[str, object] = {"id": check_id, "status": status}
    if detail is not None:
        record["detail"] = detail
    checks.append(record)


def validate(root: Path, mode: str) -> dict:
    paper_dir = root / "paper"
    freeze_path = paper_dir / "evidence/freeze_inventory.json"
    checks: list[dict] = []
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"schema_version": 1, "verdict": "FAIL", "checks": [{"id": "freeze_inventory_parse", "status": "FAIL", "detail": type(exc).__name__}]}

    add(checks, "freeze_inventory_exact_fields", set(freeze) == EXPECTED_FREEZE_FIELDS,
        detail={"missing": sorted(EXPECTED_FREEZE_FIELDS - set(freeze)), "extra": sorted(set(freeze) - EXPECTED_FREEZE_FIELDS)})
    add(checks, "freeze_schema_version", freeze.get("schema_version") == "2.0")
    add(checks, "base_commit_shape", bool(COMMIT_RE.fullmatch(str(freeze.get("base_commit", "")))))
    add(checks, "manifest_self_exclusion", freeze.get("package_manifest_excludes") == [freeze.get("package_manifest")] == ["MANIFEST.sha256"])
    add(checks, "receipt_required_fields_exact", set(freeze.get("delivery_receipt_required_fields", [])) == EXPECTED_RECEIPT_FIELDS)
    add(checks, "clean_tree_policy", freeze.get("clean_tree_required_after_delivery_commit") is True)

    source_digest, source_records = source_freeze(root, freeze)
    add(checks, "source_freeze_nonempty", bool(source_records), detail={"files": len(source_records), "sha256": source_digest})

    for rel in freeze.get("required_consistent_freeze_documents", []):
        path = root / rel
        present = path.is_file()
        contains = present and freeze.get("base_commit", "") in path.read_text(encoding="utf-8", errors="replace")
        add(checks, f"base_commit_binding:{rel}", bool(contains), blocked=not present)

    manifest_path = root / str(freeze.get("package_manifest", "MANIFEST.sha256"))
    manifest_entries: dict[str, str] = {}
    if not manifest_path.is_file():
        add(checks, "package_manifest_present", False, blocked=True)
    else:
        add(checks, "package_manifest_present", True)
        manifest_entries, parse_errors = parse_manifest(manifest_path)
        add(checks, "package_manifest_syntax", not parse_errors, detail=parse_errors)
        tracked = repository_files(root, include_untracked=False)
        expected = {
            item for item in tracked
            if path_in_roots(item, freeze.get("manifest_scope_roots", []))
            and not is_excluded(item, freeze.get("package_manifest_excludes", []))
        }
        observed = set(manifest_entries)
        add(checks, "package_manifest_exact_coverage", observed == expected,
            detail={"missing": sorted(expected - observed), "extra": sorted(observed - expected)})
        digest_errors: list[str] = []
        for rel, expected_digest in manifest_entries.items():
            path = root / rel
            if not path.is_file():
                digest_errors.append(f"missing:{rel}")
            elif sha256(path) != expected_digest:
                digest_errors.append(f"digest:{rel}")
        add(checks, "package_manifest_entries_verify", not digest_errors, detail=digest_errors)

    receipt_path = Path(str(freeze.get("external_delivery_receipt", "")))
    try:
        receipt_path.resolve().relative_to(root.resolve())
        receipt_outside = False
    except ValueError:
        receipt_outside = receipt_path.is_absolute()
    add(checks, "receipt_path_outside_repository", receipt_outside)

    try:
        head = git(root, "rev-parse", "HEAD").strip()
        status = git(root, "status", "--porcelain")
    except Exception as exc:
        head, status = "", ""
        add(checks, "git_repository", False, detail=type(exc).__name__)
    else:
        add(checks, "git_repository", True)
        try:
            subprocess.check_call(
                ["git", "-C", str(root), "merge-base", "--is-ancestor", freeze["base_commit"], head],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            base_is_ancestor = True
        except (subprocess.CalledProcessError, OSError):
            base_is_ancestor = False
        add(checks, "delivery_descends_from_evidence_base", base_is_ancestor)
    add(checks, "worktree_clean", status == "", blocked=status != "", detail={"porcelain_line_count": len(status.splitlines())})

    if not receipt_path.is_file():
        add(checks, "external_receipt_present", False, blocked=True)
    else:
        add(checks, "external_receipt_present", True)
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except Exception as exc:
            add(checks, "external_receipt_parse", False, detail=type(exc).__name__)
        else:
            add(checks, "external_receipt_exact_fields", set(receipt) == EXPECTED_RECEIPT_FIELDS,
                detail={"missing": sorted(EXPECTED_RECEIPT_FIELDS - set(receipt)), "extra": sorted(set(receipt) - EXPECTED_RECEIPT_FIELDS)})
            add(checks, "external_receipt_schema", receipt.get("schema_version") == freeze.get("delivery_receipt_schema_version"))
            add(checks, "receipt_head_binding", receipt.get("delivery_commit") == head and bool(COMMIT_RE.fullmatch(head)))
            add(checks, "receipt_clean_binding", receipt.get("clean_status") == "" and status == "")
            add(checks, "receipt_manifest_path_binding", receipt.get("package_manifest") == freeze.get("package_manifest"))
            manifest_digest = sha256(manifest_path) if manifest_path.is_file() else None
            add(checks, "receipt_manifest_digest_binding", receipt.get("package_manifest_sha256") == manifest_digest and bool(manifest_digest))
            add(checks, "receipt_source_freeze_binding", receipt.get("source_freeze_sha256") == source_digest)
            try:
                datetime.fromisoformat(str(receipt.get("created_at", "")).replace("Z", "+00:00"))
            except ValueError:
                add(checks, "receipt_timestamp", False)
            else:
                add(checks, "receipt_timestamp", True)

    statuses = {item["status"] for item in checks}
    verdict = "FAIL" if "FAIL" in statuses else ("BLOCKED" if "BLOCKED" in statuses else "PASS")
    return {
        "schema_version": 1,
        "mode": mode,
        "verdict": verdict,
        "source_freeze": {"algorithm": "sha256-path-digest-list-v1", "sha256": source_digest, "file_count": len(source_records)},
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("draft", "submission"), default="draft")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    freeze = json.loads((root / "paper/evidence/freeze_inventory.json").read_text(encoding="utf-8"))
    if args.output:
        output = args.output.resolve()
    elif args.mode == "submission":
        output = Path(freeze["external_delivery_receipt"]).with_name("rac-cvpr27-delivery-freeze-check.json")
    else:
        output = root / "paper/.aris/delivery-freeze-check.json"
    try:
        output.relative_to(root)
        output_inside_repository = True
    except ValueError:
        output_inside_repository = False
    if args.mode == "submission" and output_inside_repository:
        print("DELIVERY_FREEZE=BLOCKED; submission-mode report must be outside the repository so validation preserves the clean tree")
        return 2
    report = validate(root, args.mode)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"DELIVERY_FREEZE={report['verdict']}; checks={len(report.get('checks', []))}")
    return 0 if report["verdict"] == "PASS" else (2 if report["verdict"] == "BLOCKED" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
