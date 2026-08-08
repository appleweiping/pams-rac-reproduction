#!/usr/bin/env python3
"""Write the frozen synthetic threshold grid and disjoint seed manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from pose_recovery_v4d_full337_contract import (
    Full337ContractError,
    write_json_exclusive,
)

from pams.v4e_synthetic_contract import (
    FROZEN_THRESHOLD_FIELD_ORDER,
    SYNTHETIC_SAMPLES_PER_FAMILY,
    canonical_threshold_grid_rows,
    frozen_synthetic_seeds,
    frozen_threshold_axes,
)


def preregister(*, grid_path: Path, calibration_path: Path, heldout_path: Path) -> dict[str, object]:
    axes = frozen_threshold_axes()
    rows = canonical_threshold_grid_rows(axes)
    calibration = frozen_synthetic_seeds("calibration")
    heldout = frozen_synthetic_seeds("heldout")
    if (
        len(set(calibration)) != SYNTHETIC_SAMPLES_PER_FAMILY
        or len(set(heldout)) != SYNTHETIC_SAMPLES_PER_FAMILY
    ):
        raise ValueError("frozen seed manifest contains a collision")
    if set(calibration) & set(heldout):
        raise ValueError("frozen calibration and heldout seeds overlap")
    grid = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_frozen_threshold_grid_v1",
        "field_order": list(FROZEN_THRESHOLD_FIELD_ORDER),
        "axes": axes,
        "rows": rows,
        "real_data_observed": False,
        "label_free": True,
    }
    write_json_exclusive(grid_path, grid)
    for split, path, seeds in (
        ("calibration", calibration_path, calibration),
        ("heldout", heldout_path, heldout),
    ):
        write_json_exclusive(
            path,
            {
                "schema_version": 1,
                "artifact_type": "pams_pose_recovery_v4e_synthetic_seed_manifest_v1",
                "split": split,
                "sample_total": SYNTHETIC_SAMPLES_PER_FAMILY,
                "seeds": seeds,
                "real_data_observed": False,
                "label_free": True,
            },
        )
    return {
        "threshold_tuple_total": len(rows),
        "samples_per_family_per_split": SYNTHETIC_SAMPLES_PER_FAMILY,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-output", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--heldout-output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = preregister(
            grid_path=args.grid_output,
            calibration_path=args.calibration_output,
            heldout_path=args.heldout_output,
        )
    except (OSError, TypeError, ValueError, Full337ContractError) as exc:
        print(f"v4e synthetic design preregistration failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
