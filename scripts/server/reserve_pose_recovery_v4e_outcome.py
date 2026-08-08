#!/usr/bin/env python3
"""Exclusively reserve one canonical v4e unified-2d outcome before observation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from pose_recovery_v4d_full337_contract import (
    Full337ContractError,
    require,
    write_json_exclusive,
)

CANONICAL_PARENT = Path(
    "/media/lenovo/data2/pams-rac/runs/pams-cycleback-unified-2d-representation-v1"
)


def reserve_outcome(
    *, reservation_id: str, source_revision: str, container_image_id: str,
) -> dict[str, object]:
    require(
        len(reservation_id) == 64
        and all(character in "0123456789abcdef" for character in reservation_id),
        "reservation ID must be lowercase 64-hex",
    )
    require(
        len(source_revision) == 40
        and all(character in "0123456789abcdef" for character in source_revision),
        "source revision must be lowercase 40-hex",
    )
    require(
        container_image_id.startswith("sha256:")
        and len(container_image_id) == 71
        and all(character in "0123456789abcdef" for character in container_image_id[7:]),
        "container image ID must be sha256-prefixed 64-hex",
    )
    require(os.environ.get("PAMS_CONTAINER_SOURCE_REVISION") == source_revision, "source mismatch")
    require(os.environ.get("PAMS_CONTAINER_IMAGE_ID") == container_image_id, "image mismatch")
    canonical_parent = CANONICAL_PARENT.resolve(strict=True)
    run_root = canonical_parent / reservation_id
    require(not run_root.exists(), "canonical outcome reservation already exists")
    run_root.mkdir(mode=0o755)
    for relative in (
        "preregistration", "evidence", "gate-output", "authorization",
        "pilot/same39/output", "pilot/same39/pose-cache",
        "output", "output/raw-evidence", "output/joint-mask", "pose-cache", "audit",
    ):
        (run_root / relative).mkdir(parents=True, mode=0o755, exist_ok=False)
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_canonical_outcome_reservation_v1",
        "status": "reserved",
        "reservation_scope": "single-use-v4e-unified2d-representation-outcome",
        "outcome_family": "pams-cycleback-unified-2d-representation-v1",
        "reservation_id": reservation_id,
        "outcome_locator": str(run_root),
        "source_revision": source_revision,
        "container_image_id": container_image_id,
        "label_free": True,
        "same39_observed": False,
        "full337_observed": False,
        "cycleback_representation_input_authorized": False,
        "baseline_training_authorized": False,
    }
    reservation_path = run_root / "attempt.reservation.json"
    write_json_exclusive(reservation_path, receipt)
    os.chmod(reservation_path, 0o444)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reservation-id", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--container-image-id", required=True)
    args = parser.parse_args(argv)
    try:
        result = reserve_outcome(
            reservation_id=args.reservation_id,
            source_revision=args.source_revision,
            container_image_id=args.container_image_id,
        )
    except (OSError, TypeError, ValueError, Full337ContractError) as exc:
        print(f"v4e canonical outcome reservation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
