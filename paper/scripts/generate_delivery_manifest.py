#!/usr/bin/env python3
"""Generate the acyclic delivery manifest from the staged Git index."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from validate_delivery_freeze import is_excluded, path_in_roots


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def staged_files(root: Path) -> list[str]:
    raw = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "--cached", "-z"]
    )
    return sorted(
        item.decode("utf-8").replace("\\", "/")
        for item in raw.split(b"\0")
        if item
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    freeze = json.loads(
        (root / "paper/evidence/freeze_inventory.json").read_text(encoding="utf-8")
    )
    output = (args.output or root / freeze["package_manifest"]).resolve()
    selected = [
        path
        for path in staged_files(root)
        if path_in_roots(path, freeze["manifest_scope_roots"])
        and not is_excluded(path, freeze["package_manifest_excludes"])
    ]
    if not selected:
        raise SystemExit("delivery scope is empty; stage the package before generation")
    missing = [path for path in selected if not (root / path).is_file()]
    if missing:
        raise SystemExit(f"staged delivery paths are missing: {missing}")
    output.write_text(
        "".join(f"{sha256(root / path)}  {path}\n" for path in selected),
        encoding="utf-8",
        newline="\n",
    )
    print(f"DELIVERY_MANIFEST=GENERATED; entries={len(selected)}; path={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
