#!/usr/bin/env python3
"""Shared deterministic helpers for the ICASSP evidence gates.

The helpers intentionally use only the Python standard library.  Semantic
support remains the responsibility of the recorded ARIS reviews; these gates
only establish structure, existence, and byte-level provenance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import os
import re
from pathlib import Path
from typing import Any, Iterable

SHA256_RE = re.compile(r"^sha256:([0-9a-f]{64})$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_load(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_inside(root: Path, declared: str) -> Path:
    if not isinstance(declared, str) or not declared.strip():
        raise ValueError("path must be a non-empty string")
    raw = Path(declared)
    if raw.is_absolute() or re.match(r"^[A-Za-z]:", declared):
        raise ValueError("absolute paths are not permitted in evidence manifests")
    root = root.resolve()
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("evidence path escapes the declared bundle root") from exc
    return candidate


def hash_matches(path: Path, expected: Any) -> bool:
    return isinstance(expected, str) and bool(SHA256_RE.fullmatch(expected)) and sha256(path) == expected


def add_check(checks: list[dict[str, Any]], check_id: str, ok: bool, detail: str, *, blocking: bool = True) -> None:
    checks.append(
        {
            "id": check_id,
            "status": "PASS" if ok else ("FAIL" if blocking else "BLOCKED"),
            "detail": detail,
        }
    )


def finish_report(
    *,
    gate: str,
    mode: str,
    checks: list[dict[str, Any]],
    blockers: Iterable[str] = (),
    structural_failure: bool = False,
) -> tuple[dict[str, Any], int]:
    blockers = sorted(set(str(item) for item in blockers if str(item).strip()))
    failed = [item for item in checks if item["status"] == "FAIL"]
    if structural_failure or failed:
        status = "FAIL"
        code = 1
    elif blockers:
        status = "PROVISIONAL" if mode == "draft" else "FAIL"
        code = 0 if mode == "draft" else 1
    else:
        status = "PASS"
        code = 0
    return {
        "schema_version": "1.0",
        "gate": gate,
        "mode": mode,
        "status": status,
        "submission_ready": status == "PASS" and mode == "submission",
        "checks": checks,
        "blockers": blockers,
    }, code


def emit_report(report: dict[str, Any], output: Path | None) -> None:
    payload = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload, encoding="utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(payload, end="")


def json_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("JSON pointer must begin with '/'")
    value = document
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not token.isdigit():
                raise KeyError(token)
            value = value[int(token)]
        elif isinstance(value, dict):
            value = value[token]
        else:
            raise KeyError(token)
    return value


def source_value(source_path: Path, source_format: str, locator: dict[str, Any]) -> Any:
    if source_format == "json":
        if set(locator) != {"json_pointer"}:
            raise ValueError("JSON bindings require exactly one json_pointer")
        return json_pointer(json_load(source_path), locator["json_pointer"])
    if source_format == "csv":
        if set(locator) != {"row", "column"}:
            raise ValueError("CSV bindings require exactly row and column")
        rows = csv_load(source_path)
        row = locator["row"]
        column = locator["column"]
        if not isinstance(row, int) or row < 0 or row >= len(rows):
            raise IndexError("CSV row is out of range")
        if column not in rows[row]:
            raise KeyError(column)
        return rows[row][column]
    raise ValueError("source format must be json or csv")


def stable_rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def safe_latex_scalar(value: Any) -> str:
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, (int, float, str)) and not isinstance(value, complex):
        rendered = str(value)
    else:
        raise ValueError("paper binding must resolve to a scalar")
    if not rendered.strip() or "\n" in rendered or "\r" in rendered:
        raise ValueError("paper binding must be one non-empty line")
    if len(rendered) > 2000:
        raise ValueError("paper binding is unexpectedly long")
    dangerous = re.compile(
        r"\\(?:input|include|openout|write|immediate|read|usepackage|documentclass|"
        r"newcommand|renewcommand|def|gdef|xdef|csname|catcode|loop)\b",
        re.IGNORECASE,
    )
    if dangerous.search(rendered):
        raise ValueError("paper binding contains a prohibited TeX command")
    if rendered.count("{") != rendered.count("}"):
        raise ValueError("paper binding contains unbalanced braces")
    return rendered


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
