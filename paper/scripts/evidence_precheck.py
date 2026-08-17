#!/usr/bin/env python3
"""Verify exact bytes, safe paths, media, package manifest, and delivery receipt.

The package manifest hashes the result manifest and every referenced artifact,
excluding only itself.  A receipt selected by an explicit external CLI path
hashes the exact result manifest and package manifest.  The manifest stores only
the receipt's logical ID, so this directed binding has no digest cycle.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


PACKAGE_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def digest_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_resolve(root: Path, value: str) -> Path:
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    if "\\" in value or pure.as_posix() != value or (pure.parts and ":" in pure.parts[0]):
        raise ValueError(f"unsafe relative path: {value!r}")
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root.joinpath(*pure.parts).resolve(strict=True)
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"path escapes bundle root: {value!r}")
    if not candidate.is_file():
        raise ValueError(f"not a regular file: {value!r}")
    return candidate


def artifacts(value: Any, location: str = "$") -> Iterator[tuple[str, dict[str, str]]]:
    if isinstance(value, dict):
        artifact_keys = {"path", "sha256", "media_type", "role"}
        if artifact_keys.issubset(value):
            yield location, value
        for key, child in value.items():
            yield from artifacts(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from artifacts(child, f"{location}[{index}]")


def verify_media(media_type: str, payload: bytes) -> str | None:
    try:
        if media_type == "application/json":
            json.loads(payload.decode("utf-8"))
        elif media_type == "application/jsonl":
            lines = [line for line in payload.decode("utf-8").splitlines() if line.strip()]
            if not lines:
                return "JSONL has no records"
            for line in lines:
                json.loads(line)
        elif media_type == "text/csv":
            rows = list(csv.reader(io.StringIO(payload.decode("utf-8"))))
            if not rows or not rows[0]:
                return "CSV has no header"
        elif media_type == "application/vnd.apache.parquet":
            if len(payload) < 8 or payload[:4] != b"PAR1" or payload[-4:] != b"PAR1":
                return "Parquet magic bytes are absent"
        elif media_type in {"text/plain", "text/x-log"}:
            payload.decode("utf-8")
        elif media_type == "application/pdf":
            if not payload.startswith(b"%PDF-"):
                return "PDF signature is absent"
        elif media_type == "image/svg+xml":
            root = ET.fromstring(payload.decode("utf-8"))
            if root.tag.split("}")[-1].lower() != "svg":
                return "XML root is not svg"
        else:
            return f"unsupported media type {media_type!r}"
    except (UnicodeDecodeError, json.JSONDecodeError, csv.Error, ET.ParseError) as exc:
        return f"media parse failed: {exc}"
    return None


def read_package_manifest(path: Path, root: Path, errors: list[str]) -> dict[str, str]:
    entries: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        errors.append(f"cannot read package manifest: {exc}")
        return entries
    if not lines:
        errors.append("package manifest is empty")
    for line_number, line in enumerate(lines, start=1):
        match = PACKAGE_LINE.fullmatch(line)
        if not match:
            errors.append(f"package manifest line {line_number}: expected '<sha256>  <relative/path>'")
            continue
        digest, relative = match.groups()
        if relative in entries:
            errors.append(f"package manifest duplicate path: {relative}")
            continue
        try:
            candidate = safe_resolve(root, relative)
        except (OSError, ValueError) as exc:
            errors.append(f"package manifest path {relative!r}: {exc}")
            continue
        actual = digest_bytes(candidate.read_bytes())
        if actual != digest:
            errors.append(f"package manifest digest mismatch: {relative}")
        entries[relative] = digest
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--external-receipt",
        type=Path,
        help="absolute path to the out-of-bundle receipt identified by manifest_integrity.external_delivery_receipt_id",
    )
    args = parser.parse_args()
    if not args.manifest.is_file():
        print(f"EVIDENCE_PRECHECK=BLOCKED; synchronized manifest is absent: {args.manifest}")
        return 2
    root = (args.bundle_root or args.manifest.parent).resolve()
    errors: list[str] = []
    try:
        manifest_bytes = args.manifest.read_bytes()
        data = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"EVIDENCE_PRECHECK=FAIL; manifest_parse_error={exc}")
        return 1

    integrity = data.get("manifest_integrity", {})
    required_integrity = {
        "algorithm", "result_manifest_path", "package_manifest_path",
        "external_delivery_receipt_id", "package_manifest_excludes",
    }
    if set(integrity) != required_integrity or integrity.get("algorithm") != "sha256":
        errors.append("manifest_integrity has invalid keys or algorithm")
        package_path = receipt_path = None
    else:
        try:
            declared_manifest_path = safe_resolve(root, integrity["result_manifest_path"])
            if declared_manifest_path != args.manifest.resolve(strict=True):
                errors.append("result_manifest_path does not identify the validated manifest")
            package_path = safe_resolve(root, integrity["package_manifest_path"])
        except (OSError, ValueError) as exc:
            errors.append(f"manifest_integrity path failure: {exc}")
            package_path = None
        receipt_path = args.external_receipt.resolve() if args.external_receipt else None
        if receipt_path is None:
            errors.append("a truly external receipt must be supplied with --external-receipt")
        else:
            try:
                receipt_path.relative_to(root)
            except ValueError:
                pass
            else:
                errors.append("--external-receipt must resolve outside the bundle root")
            if not receipt_path.is_absolute() or not receipt_path.is_file():
                errors.append("--external-receipt is not an existing absolute regular file")

    artifact_paths: dict[str, tuple[str, str]] = {}
    artifact_count = 0
    for location, artifact in artifacts(data):
        artifact_count += 1
        relative = artifact["path"]
        expected = artifact["sha256"]
        if not isinstance(expected, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected):
            errors.append(f"{location}: malformed sha256")
            continue
        signature = (expected, artifact["media_type"])
        previous = artifact_paths.setdefault(relative, signature)
        if previous != signature:
            errors.append(f"{location}: path reused with conflicting digest/media: {relative}")
            continue
        try:
            candidate = safe_resolve(root, relative)
            payload = candidate.read_bytes()
        except (OSError, ValueError) as exc:
            errors.append(f"{location}: {exc}")
            continue
        actual = digest_bytes(payload)
        if actual != expected.removeprefix("sha256:"):
            errors.append(f"{location}: artifact digest mismatch: {relative}")
        media_error = verify_media(artifact["media_type"], payload)
        if media_error:
            errors.append(f"{location}: {relative}: {media_error}")

    package_entries: dict[str, str] = {}
    if package_path is not None:
        package_entries = read_package_manifest(package_path, root, errors)
        excluded = set(integrity["package_manifest_excludes"])
        required_excluded = {integrity["package_manifest_path"]}
        if excluded != required_excluded:
            errors.append("package_manifest_excludes must be exactly the package manifest; the external receipt is not a package path")
        if excluded.intersection(package_entries):
            errors.append("package manifest contains a path declared as excluded")
        expected_entries = set(artifact_paths) | {integrity["result_manifest_path"]}
        if set(package_entries) != expected_entries:
            missing = sorted(expected_entries - set(package_entries))
            extra = sorted(set(package_entries) - expected_entries)
            errors.append(f"package manifest coverage mismatch; missing={missing}; extra={extra}")

    receipt: dict[str, Any] = {}
    if receipt_path is not None and package_path is not None:
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read external delivery receipt: {exc}")
        required_receipt = {
            "schema_version", "algorithm", "receipt_id", "delivery_commit",
            "result_manifest", "package_manifest",
        }
        if receipt and set(receipt) != required_receipt:
            errors.append(f"external delivery receipt keys must be exactly {sorted(required_receipt)}")
        elif receipt:
            if receipt["schema_version"] != "2.0" or receipt["algorithm"] != "sha256":
                errors.append("external delivery receipt schema/algorithm mismatch")
            if receipt.get("receipt_id") != integrity.get("external_delivery_receipt_id"):
                errors.append("external delivery receipt logical identifier mismatch")
            if not isinstance(receipt["delivery_commit"], str) or not COMMIT.fullmatch(receipt["delivery_commit"]):
                errors.append("external delivery receipt has malformed delivery_commit")
            for key, declared_path, actual_path in (
                ("result_manifest", integrity["result_manifest_path"], args.manifest.resolve(strict=True)),
                ("package_manifest", integrity["package_manifest_path"], package_path),
            ):
                binding = receipt.get(key)
                if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
                    errors.append(f"external delivery receipt {key} binding is malformed")
                    continue
                if binding["path"] != declared_path:
                    errors.append(f"external delivery receipt {key} path mismatch")
                expected_digest = f"sha256:{digest_bytes(actual_path.read_bytes())}"
                if binding["sha256"] != expected_digest:
                    errors.append(f"external delivery receipt {key} digest mismatch")

    report = {
        "schema_version": "1.0",
        "evaluated_manifest_created_at": data.get("created_at"),
        "verdict": "FAIL" if errors else "PASS",
        "manifest": integrity.get("result_manifest_path"),
        "manifest_sha256": f"sha256:{digest_bytes(manifest_bytes)}",
        "artifact_references_checked": artifact_count,
        "unique_artifacts_checked": len(artifact_paths),
        "package_entries_checked": len(package_entries),
        "external_receipt": {
            "id": integrity.get("external_delivery_receipt_id"),
            "path": str(receipt_path) if receipt_path else None,
            "sha256": f"sha256:{digest_bytes(receipt_path.read_bytes())}" if receipt_path and receipt_path.is_file() else None,
            "outside_bundle": bool(receipt_path and not receipt_path.is_relative_to(root)),
        },
        "errors": errors,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if errors:
        print("EVIDENCE_PRECHECK=FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"EVIDENCE_PRECHECK=PASS; artifacts={len(artifact_paths)}; package_entries={len(package_entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
