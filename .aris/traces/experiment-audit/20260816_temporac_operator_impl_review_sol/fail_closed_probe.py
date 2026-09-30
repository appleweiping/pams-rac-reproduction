"""Trace-only Linux probe for exclusive-create behavior."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


def main() -> int:
    target = Path("/tmp/exclusive-existing")
    target.write_bytes(b"original")
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(
        "generator",
        "/tmp/repo/scripts/experiments/generate_temporac_operator_manifest_v4.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module._write_exclusive(target, b"replacement")
    except FileExistsError:
        pass
    else:
        raise AssertionError("exclusive create overwrote an existing file")
    after = hashlib.sha256(target.read_bytes()).hexdigest()
    assert before == after and target.read_bytes() == b"original"
    print(f"write_exclusive_existing_file=PASS unchanged_sha256={after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
