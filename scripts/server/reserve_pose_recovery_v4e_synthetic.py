#!/usr/bin/env python3
"""Reserve the synthetic-only v4e threshold experiment before any real cohort."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from pose_recovery_v4d_full337_contract import (
    MODEL_ASSET_SHA256,
    Full337ContractError,
    require,
    sha256_file,
    write_json_exclusive,
)

from pams.config import load_config
from pams.keypoint_single_source import V4E_PREPROCESSING_REVISION
from pams.v4e_synthetic_contract import (
    FROZEN_THRESHOLD_FIELD_ORDER,
    canonical_threshold_grid_rows,
)

CANONICAL_OUTCOME_PARENT = Path(
    "/media/lenovo/data2/pams-rac/runs/pams-cycleback-unified-2d-representation-v1"
)


def reserve_synthetic(
    *,
    source_revision: str,
    container_image_id: str,
    config_path: Path,
    model_asset_path: Path,
    generator_source_path: Path,
    threshold_grid_path: Path,
    calibration_seed_manifest_path: Path,
    heldout_seed_manifest_path: Path,
    registry_reservation_path: Path,
    expected_registry_reservation_sha256: str,
    future_evidence_path: Path,
    output_path: Path,
) -> dict[str, object]:
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
    require(not future_evidence_path.exists(), "synthetic evidence already exists before reservation")
    require(
        sha256_file(registry_reservation_path) == expected_registry_reservation_sha256,
        "canonical registry reservation SHA mismatch",
    )
    registry = json.loads(registry_reservation_path.read_text(encoding="utf-8"))
    require(isinstance(registry, Mapping), "registry reservation must be an object")
    resolved_registry = registry_reservation_path.resolve(strict=True)
    run_root = resolved_registry.parent
    require(
        registry.get("artifact_type")
        == "pams_pose_recovery_v4e_canonical_outcome_reservation_v1"
        and registry.get("status") == "reserved"
        and registry.get("reservation_scope")
        == "single-use-v4e-unified2d-representation-outcome"
        and registry.get("outcome_family")
        == "pams-cycleback-unified-2d-representation-v1"
        and registry.get("label_free") is True
        and registry.get("same39_observed") is False
        and registry.get("full337_observed") is False,
        "canonical outcome reservation contract mismatch",
    )
    require(
        resolved_registry.name == "attempt.reservation.json"
        and run_root.parent == CANONICAL_OUTCOME_PARENT
        and registry.get("outcome_locator") == str(run_root),
        "canonical outcome reservation locator mismatch",
    )
    require(
        stat.S_IMODE(resolved_registry.stat().st_mode) & 0o222 == 0,
        "canonical outcome reservation must be sealed read-only",
    )
    expected_reservation_output = run_root / "preregistration/v4e-synthetic.reservation.json"
    expected_evidence_output = run_root / "evidence/v4e-synthetic-grid.evidence.json"
    require(
        output_path.parent.resolve(strict=True) / output_path.name == expected_reservation_output,
        "synthetic reservation output locator mismatch",
    )
    require(
        future_evidence_path.parent.resolve(strict=True) / future_evidence_path.name
        == expected_evidence_output,
        "synthetic evidence locator mismatch",
    )
    config_file_sha256 = sha256_file(config_path)
    config = load_config(config_path.resolve(strict=True))
    require(config.pose.preprocessing_revision == V4E_PREPROCESSING_REVISION, "config is not v4e")
    require(sha256_file(model_asset_path) == MODEL_ASSET_SHA256, "model asset mismatch")
    grid = json.loads(threshold_grid_path.read_text(encoding="utf-8"))
    require(isinstance(grid, Mapping), "threshold grid must be an object")
    require(grid.get("artifact_type") == "pams_pose_recovery_v4e_frozen_threshold_grid_v1", "grid type mismatch")
    require(grid.get("field_order") == list(FROZEN_THRESHOLD_FIELD_ORDER), "grid field order mismatch")
    axes = grid.get("axes")
    require(isinstance(axes, Mapping), "threshold grid axes missing")
    canonical_grid_rows = canonical_threshold_grid_rows(axes)
    grid_rows = grid.get("rows")
    require(grid_rows == canonical_grid_rows, "threshold grid is not the frozen Cartesian product")

    bindings: dict[str, object] = {
        "source_revision": source_revision,
        "container_image_id": container_image_id,
        "config_file_sha256": config_file_sha256,
        "config_file_bytes": config_path.stat().st_size,
        "config_fingerprint": config.fingerprint,
        "pose_fingerprint": config.pose_fingerprint,
        "model_asset_sha256": MODEL_ASSET_SHA256,
        "model_asset_bytes": model_asset_path.stat().st_size,
        "generator_source_sha256": sha256_file(generator_source_path),
        "generator_source_bytes": generator_source_path.stat().st_size,
        "threshold_grid_sha256": sha256_file(threshold_grid_path),
        "threshold_grid_bytes": threshold_grid_path.stat().st_size,
        "calibration_seed_manifest_sha256": sha256_file(calibration_seed_manifest_path),
        "calibration_seed_manifest_bytes": calibration_seed_manifest_path.stat().st_size,
        "heldout_seed_manifest_sha256": sha256_file(heldout_seed_manifest_path),
        "heldout_seed_manifest_bytes": heldout_seed_manifest_path.stat().st_size,
        "canonical_registry_reservation_sha256": expected_registry_reservation_sha256,
        "canonical_registry_reservation_bytes": registry_reservation_path.stat().st_size,
        "canonical_outcome_locator": str(run_root),
        "canonical_registry_artifact_type": registry["artifact_type"],
    }
    require(
        bindings["calibration_seed_manifest_sha256"]
        != bindings["heldout_seed_manifest_sha256"],
        "calibration and heldout seed manifests must differ",
    )
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_synthetic_reservation_v1",
        "status": "reserved",
        "label_free": True,
        "full337_observed": False,
        "same39_observed": False,
        "threshold_grid": grid_rows,
        "threshold_axes": dict(axes),
        "threshold_grid_construction": "cartesian-ordered-axes-severity-index-v1",
        "selection_policy": {
            "one_sided_clopper_pearson_alpha": 0.05,
            "minimum_known_identity_retention_lower_bound": 0.90,
            "maximum_false_eligible_upper_bound": 0.01,
            "confidence_scope": (
                "one-sided-bounds-apply-only-to-the-preregistered-synthetic-distribution"
            ),
            "samples_per_family_per_split": 512,
            "tuple_selection": "minimum-severity-sum-then-frozen-field-lexicographic-v1",
            "frozen_threshold_field_order": list(FROZEN_THRESHOLD_FIELD_ORDER),
            "minimum_same39_representation_eligible_fraction": 0.80,
            "minimum_same39_prefix_suffix_path_agreement": 0.90,
            "same39_selects_thresholds": False,
            "heldout_selects_thresholds": False,
        },
        "bindings": bindings,
        "future_evidence_locator": str(future_evidence_path),
        "same39_raw_extraction_authorized": False,
        "full337_raw_extraction_authorized": False,
        "baseline_training_authorized": False,
    }
    write_json_exclusive(output_path, receipt)
    os.chmod(output_path, 0o444)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--registry-reservation-sha256", required=True)
    for flag in (
        "config", "model-asset", "generator-source", "threshold-grid",
        "calibration-seed-manifest", "heldout-seed-manifest",
        "registry-reservation", "future-evidence", "output",
    ):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = reserve_synthetic(
            source_revision=args.source_revision,
            container_image_id=args.container_image_id,
            config_path=args.config,
            model_asset_path=args.model_asset,
            generator_source_path=args.generator_source,
            threshold_grid_path=args.threshold_grid,
            calibration_seed_manifest_path=args.calibration_seed_manifest,
            heldout_seed_manifest_path=args.heldout_seed_manifest,
            registry_reservation_path=args.registry_reservation,
            expected_registry_reservation_sha256=args.registry_reservation_sha256,
            future_evidence_path=args.future_evidence,
            output_path=args.output,
        )
    except (OSError, TypeError, ValueError, Full337ContractError, KeyError) as exc:
        print(f"v4e synthetic reservation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
