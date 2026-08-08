#!/usr/bin/env python3
"""Write the frozen synthetic threshold grid and disjoint seed manifests."""

from __future__ import annotations

import argparse
import hashlib
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
    canonical_threshold_grid_rows,
)

FROZEN_AXES = {
    "maximum_candidate_window_frames": [24],
    "maximum_frame_center_step": [0.5, 0.3],
    "maximum_frame_joint_mask_flicker_fraction": [0.5, 0.3],
    "maximum_frame_log_scale_step": [0.5, 0.3],
    "minimum_dual_path_agreement": [0.85, 0.95],
    "minimum_frame_local_ambiguity_gap": [0.0, 0.02],
    "minimum_longest_trainable_segment_fraction": [0.8],
    "minimum_longest_trainable_segment_frames": [48],
    "minimum_source_coverage": [0.8],
    "minimum_window_joint_support_fraction": [0.75],
    "minimum_window_stable_action_joints": [4],
}


def _seeds(split: str) -> list[int]:
    return [
        int.from_bytes(
            hashlib.sha256(
                f"pams-v4e-synthetic-v1:{split}:{index}".encode()
            ).digest()[:8],
            "big",
        )
        & ((1 << 63) - 1)
        for index in range(512)
    ]


def preregister(*, grid_path: Path, calibration_path: Path, heldout_path: Path) -> dict[str, object]:
    rows = canonical_threshold_grid_rows(FROZEN_AXES)
    calibration = _seeds("calibration")
    heldout = _seeds("heldout")
    if len(set(calibration)) != 512 or len(set(heldout)) != 512:
        raise ValueError("frozen seed manifest contains a collision")
    if set(calibration) & set(heldout):
        raise ValueError("frozen calibration and heldout seeds overlap")
    grid = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_frozen_threshold_grid_v1",
        "field_order": list(FROZEN_THRESHOLD_FIELD_ORDER),
        "axes": FROZEN_AXES,
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
                "sample_total": 512,
                "seeds": seeds,
                "real_data_observed": False,
                "label_free": True,
            },
        )
    return {"threshold_tuple_total": len(rows), "samples_per_family_per_split": 512}


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
