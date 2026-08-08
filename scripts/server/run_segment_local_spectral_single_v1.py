#!/usr/bin/env python3
"""Fail-closed placeholder for the segment-local single-expert server run.

The science core is intentionally not wired to pose/encoder artifacts until
the v4e representation authority and cycleback full-segment encoding schemas
are merged at exact reviewed revisions.  This entry point must remain a hard
denial until that integration supplies independently verified receipts.
"""

from __future__ import annotations

import json
import sys

FAIL_CLOSED_REASON = (
    "authoritative v4e representation and cycleback full-segment encoding "
    "receipt adapters are not merged; no synthetic, train337, dev84, or test105 "
    "run is authorized"
)


def main() -> int:
    payload = {
        "schema_version": 1,
        "status": "rejected",
        "classification": "fail_closed_unwired_science_core",
        "reason": FAIL_CLOSED_REASON,
        "targets_mounted": False,
        "dev84_authorized": False,
        "test105_authorized": False,
    }
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), file=sys.stderr)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
