"""Strict read-only structural precheck for an A005 Round-9 successor pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REPLACEMENTS = (
    ("2026-08-17T13:48:53+08:00", "2026-08-19T19:54:24+08:00"),
    ("20260817_134853", "20260819_195424"),
    (
        "FD32/original result failure emits no bytes exit124; sole FD33 restoration "
        "failure retains prior stage or selects OUTPUT and exits123 on valid FD1.",
        "FD32 failure,32->1 restoration failure,or unusable FD1 emits no bytes and "
        "exits124; if and only if FD32 and 32->1 remain usable and the sole restoration "
        "failure is FD33 or 33->2,the controller retains the first failure,emits the "
        "exact sequence-zero type0x7f failure frame on valid FD1,then EOF,and exits123.",
    ),
    (
        "Parent resumes only by PTRACE_SYSCALL and retains32,33,34,48,51,52 for that "
        "child.",
        "Resume authority is in exactly one state: ATTACH_BOOTSTRAP drives only the "
        "reviewed attach/read/exec suffix under PTRACE_SYSCALL and consumes its special "
        "bootstrap EXIT without an authority-transition gate; AUTHORITY_TRANSITION "
        "requires the ordinary or FUTEX closed gate before PTRACE_SYSCALL; "
        "FAILURE_TERMINATION follows failure_cleanup/SIGKILL and is never forced "
        "through an authority-transition gate. The parent retains32,33,34,48,51,52 "
        "for that child.",
    ),
    (
        "resume every stopped live task only with PTRACE_SYSCALL(tid,0,SIGKILL);",
        "enter FAILURE_TERMINATION and resume every stopped live task only with "
        "PTRACE_SYSCALL(tid,0,SIGKILL),without selecting an AUTHORITY_TRANSITION gate;",
    ),
    (
        "Every PTRACE_SYSCALL resume selects exactly one closed gate.",
        "Every AUTHORITY_TRANSITION PTRACE_SYSCALL resume selects exactly one closed "
        "ordinary or FUTEX gate. ATTACH_BOOTSTRAP and FAILURE_TERMINATION are mutually "
        "exclusive states with their own closed rules and never select this gate.",
    ),
    (
        "Before every PTRACE_SYSCALL resume exactly one ordinary gate or one four-row "
        "whole-span FUTEX gate validates the available prefix.",
        "In AUTHORITY_TRANSITION,before every PTRACE_SYSCALL resume exactly one ordinary "
        "gate or one four-row whole-span FUTEX gate validates the available prefix; "
        "ATTACH_BOOTSTRAP and FAILURE_TERMINATION use only their separate closed resume "
        "rules.",
    ),
    (
        "selects exactly one expanded outcome by (K,current stage,action,mapped "
        "token),also ignoring emitter.",
        "selects exactly one expanded row by observable (K,current stage),also ignoring "
        "emitter; action and mapped token are outputs of that selected row and are never "
        "selector inputs.",
    ),
    (
        "unique expanded outcomes by K,stage,action,token.",
        "exactly one expanded row by observable K,stage; action/token are row outputs.",
    ),
    (
        "all expanded stage/action/token outcomes",
        "all expanded stage rows including action/token output bytes",
    ),
    (
        "The expanded (definition digest,stage,action,token) keys are unique across all "
        "forms. Expanded uniqueness is keyed by observable K,stage,action,token across "
        "all emitter metadata.",
        "The expanded (definition digest,stage) keys are unique across all forms. "
        "Expanded uniqueness is keyed by observable K,stage across all emitter metadata; "
        "action and mapped token are outputs of the uniquely selected row.",
    ),
)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load(path: Path) -> tuple[bytes, Any]:
    raw = path.read_bytes()
    return raw, json.loads(raw, object_pairs_hook=_strict_pairs)


def _transform(value: str) -> str:
    for old, new in REPLACEMENTS:
        value = value.replace(old, new)
    return value


def _compare(old: Any, new: Any, path: str, changed: list[str]) -> None:
    if type(old) is not type(new):
        raise ValueError(f"type changed at {path}")
    if isinstance(old, dict):
        if tuple(old) != tuple(new):
            raise ValueError(f"object keys/order changed at {path}")
        for key in old:
            _compare(old[key], new[key], f"{path}/{key}", changed)
        return
    if isinstance(old, list):
        if len(old) != len(new):
            raise ValueError(f"array length changed at {path}")
        for index, (left, right) in enumerate(zip(old, new, strict=True)):
            _compare(left, right, f"{path}/{index}", changed)
        return
    if isinstance(old, str):
        if old != new:
            if _transform(old) != new:
                raise ValueError(f"unapproved string change at {path}")
            changed.append(path)
        return
    if old != new:
        raise ValueError(f"non-string value changed at {path}: {old!r} -> {new!r}")


def _appendix(markdown: bytes) -> bytes:
    start_marker = b"~~~json\n"
    end_marker = b"\n~~~\n"
    start = markdown.index(start_marker) + len(start_marker)
    end = markdown.index(end_marker, start)
    return markdown[start:end] + b"\n"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-json", required=True, type=Path)
    parser.add_argument("--new-json", required=True, type=Path)
    parser.add_argument("--new-md", required=True, type=Path)
    args = parser.parse_args()
    old_raw, old = _load(args.old_json)
    new_raw, new = _load(args.new_json)
    markdown = args.new_md.read_bytes()
    if _appendix(markdown) != new_raw:
        raise ValueError("Markdown normative appendix differs from standalone JSON bytes")
    changed: list[str] = []
    _compare(old, new, "", changed)
    if not changed:
        raise ValueError("candidate has no approved JSON changes")
    text = new_raw.decode("utf-8")
    forbidden = (
        "FD32/original result failure",
        "Parent resumes only by PTRACE_SYSCALL",
        "Every PTRACE_SYSCALL resume selects exactly one closed gate",
        "selects exactly one expanded outcome by (K,current stage,action,mapped token)",
        "unique expanded outcomes by K,stage,action,token",
    )
    residual = [value for value in forbidden if value in text]
    if residual:
        raise ValueError(f"obsolete semantic strings remain: {residual}")
    print(
        json.dumps(
            {
                "changed_json_string_paths": len(changed),
                "markdown_sha256": _sha(markdown),
                "new_json_sha256": _sha(new_raw),
                "old_json_sha256": _sha(old_raw),
                "result": "PASS",
                "schema": "temporac.a005-round9-author-precheck.v1",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
