"""Server entry point for the train337-only cycle-back geometry/PE/null gate."""

# ruff: noqa: I001

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pams.conventional_cycleback.audit import main


if __name__ == "__main__":
    raise SystemExit(main())
