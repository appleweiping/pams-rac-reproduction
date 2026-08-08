#!/usr/bin/env python3
"""Fail-closed data launcher for segment-local spectral single-expert v2.

The pure synthetic selector is callable from the science module.  This data
launcher remains fixed at exit 3 until reviewed representation adapters and a
separate train337 authorization mechanism exist.  A synthetic pass is never
accepted here as training, dev84, or test105 authority.
"""

from __future__ import annotations

import json
import sys

FAIL_CLOSED_REASON = (
    "v2 representation adapter is unwired; synthetic permission is not "
    "train337/dev84/test105 authority"
)


def main() -> int:
    payload = {
        "schema_version": 2,
        "method_key": "segment_local_spectral_single_v2",
        "status": "rejected",
        "reason": FAIL_CLOSED_REASON,
        "synthetic_selector_authorized": True,
        "train337_authorized": False,
        "dev84_authorized": False,
        "test105_authorized": False,
        "targets_mounted": False,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
