"""Server entry point for the cycle-back 256-step mechanism probe."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pams.conventional_cycleback.probe import main


if __name__ == "__main__":
    raise SystemExit(main())
