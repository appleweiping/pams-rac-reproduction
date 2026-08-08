"""One-shot adapter from the preregistered unified-2D PASS to cycle-back."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pams.conventional_cycleback.authority import main


if __name__ == "__main__":
    raise SystemExit(main())
