#!/usr/bin/env python3
"""Reserve the canonical conventional-cycleback launch registry."""

# ruff: noqa: I001

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pams.conventional_cycleback.authority import launch_registry_main


if __name__ == "__main__":
    raise SystemExit(launch_registry_main())
