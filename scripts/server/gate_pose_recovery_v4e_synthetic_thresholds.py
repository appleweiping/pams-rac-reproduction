#!/usr/bin/env python3
"""Select the unique widest v4e threshold tuple from frozen synthetic evidence."""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pose_recovery_v4d_full337_contract import (
    Full337ContractError,
    require,
    sha256_file,
    write_json_exclusive,
)

from pams.v4e_synthetic_contract import (
    SYNTHETIC_DIAGNOSTIC_FAMILIES,
    SYNTHETIC_IDENTITY_NULL_FAMILIES,
    SYNTHETIC_JOINT_NULL_FAMILIES,
    SYNTHETIC_POSITIVE_FAMILIES,
    canonical_threshold_grid_rows,
)

THRESHOLD_KEYS = {
    "maximum_candidate_window_frames",
    "maximum_frame_center_step",
    "maximum_frame_log_scale_step",
    "maximum_frame_morphology_step",
    "maximum_frame_joint_mask_flicker_fraction",
    "minimum_dual_path_agreement",
    "minimum_frame_local_ambiguity_gap",
    "minimum_longest_trainable_segment_fraction",
    "minimum_longest_trainable_segment_frames",
    "minimum_source_coverage",
    "minimum_window_joint_support_fraction",
    "minimum_window_action_motion",
    "minimum_window_stable_action_joints",
}
REQUIRED_POSITIVE_FAMILIES = set(SYNTHETIC_POSITIVE_FAMILIES)
REQUIRED_DIAGNOSTIC_FAMILIES = set(SYNTHETIC_DIAGNOSTIC_FAMILIES)
REQUIRED_IDENTITY_NULL_FAMILIES = set(SYNTHETIC_IDENTITY_NULL_FAMILIES)
REQUIRED_JOINT_NULL_FAMILIES = set(SYNTHETIC_JOINT_NULL_FAMILIES)


def _object(path: Path, role: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, Mapping), f"{role} must be a JSON object")
    return value


def _decision_bitset_count(value: Any) -> int:
    require(isinstance(value, str) and len(value) == 128, "decision bitset must encode 512 bits")
    try:
        raw = bytes.fromhex(value)
    except ValueError as exc:
        raise Full337ContractError("decision bitset is not lowercase hex") from exc
    require(value == value.lower() and len(raw) == 64, "decision bitset encoding mismatch")
    return sum(byte.bit_count() for byte in raw)


def _binomial_cdf(k: int, n: int, probability: float) -> float:
    return float(
        sum(
            math.comb(n, value)
            * probability**value
            * (1.0 - probability) ** (n - value)
            for value in range(k + 1)
        )
    )


def _clopper_pearson_one_sided(
    successes: int,
    total: int,
    *,
    alpha: float,
) -> tuple[float, float]:
    require(0 <= successes <= total and total > 0, "invalid binomial counts")
    require(0.0 < alpha < 1.0, "invalid Clopper-Pearson alpha")
    lower = 0.0
    if successes > 0:
        left, right = 0.0, 1.0
        for _ in range(80):
            midpoint = 0.5 * (left + right)
            tail = 1.0 - _binomial_cdf(successes - 1, total, midpoint)
            if tail < alpha:
                left = midpoint
            else:
                right = midpoint
        lower = 0.5 * (left + right)
    upper = 1.0
    if successes < total:
        left, right = 0.0, 1.0
        for _ in range(80):
            midpoint = 0.5 * (left + right)
            cdf = _binomial_cdf(successes, total, midpoint)
            if cdf > alpha:
                left = midpoint
            else:
                right = midpoint
        upper = 0.5 * (left + right)
    return lower, upper


def gate_synthetic_thresholds(
    *,
    reservation_path: Path,
    expected_reservation_sha256: str,
    evidence_path: Path,
    expected_evidence_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    require(sha256_file(reservation_path) == expected_reservation_sha256, "reservation SHA mismatch")
    require(sha256_file(evidence_path) == expected_evidence_sha256, "synthetic evidence SHA mismatch")
    reservation = _object(reservation_path, "synthetic reservation")
    evidence = _object(evidence_path, "synthetic evidence")
    require(
        reservation.get("artifact_type") == "pams_pose_recovery_v4e_synthetic_reservation_v1"
        and reservation.get("status") == "reserved"
        and reservation.get("full337_observed") is False
        and reservation.get("same39_observed") is False,
        "synthetic reservation is not pre-data",
    )
    require(
        evidence.get("artifact_type") == "pams_pose_recovery_v4e_synthetic_grid_evidence_v1"
        and evidence.get("label_free") is True
        and evidence.get("reservation_sha256") == expected_reservation_sha256,
        "synthetic evidence contract mismatch",
    )
    require(
        evidence.get("mechanics_chain")
        == "synthetic-raw-kprcnn-channels-to-shared-production-filter-canonical-sort-to-dual-viterbi-to-canonical-frame-evidence-to-body-centered-cache-to-track-stability-and-usable-action-motion-v2"
        and evidence.get("truth_role")
        == "synthetic-actor-retention-track-stability-and-false-eligible-scoring-only"
        and evidence.get("period_invariance_checked") is True
        and evidence.get("candidate_order_permutation_checked") is True,
        "synthetic mechanics replay contract mismatch",
    )
    require(
        evidence_path.resolve(strict=True) == Path(str(reservation["future_evidence_locator"]))
        and stat.S_IMODE(evidence_path.stat().st_mode) & 0o222 == 0,
        "synthetic evidence canonical locator/sealing mismatch",
    )
    reservation_bindings = reservation.get("bindings")
    evidence_bindings = evidence.get("bindings")
    require(
        isinstance(reservation_bindings, Mapping)
        and isinstance(evidence_bindings, Mapping),
        "synthetic producer bindings missing",
    )
    expected_output = Path(str(reservation_bindings["canonical_outcome_locator"])) / (
        "gate-output/v4e-synthetic-threshold.receipt.json"
    )
    require(
        reservation_path.resolve(strict=True)
        == Path(str(reservation_bindings["canonical_outcome_locator"]))
        / "preregistration/v4e-synthetic.reservation.json"
        and stat.S_IMODE(reservation_path.stat().st_mode) & 0o222 == 0,
        "synthetic reservation canonical locator/sealing mismatch",
    )
    require(
        output_path.parent.resolve(strict=True) / output_path.name == expected_output,
        "synthetic gate output locator mismatch",
    )
    for binding in (
        "source_revision",
        "container_image_id",
        "config_file_sha256",
        "config_file_bytes",
        "config_fingerprint",
        "pose_fingerprint",
        "model_asset_sha256",
        "model_asset_bytes",
        "generator_source_sha256",
        "generator_source_bytes",
        "threshold_grid_sha256",
        "threshold_grid_bytes",
        "calibration_seed_manifest_sha256",
        "calibration_seed_manifest_bytes",
        "heldout_seed_manifest_sha256",
        "heldout_seed_manifest_bytes",
        "canonical_registry_reservation_sha256",
        "canonical_registry_reservation_bytes",
        "canonical_outcome_locator",
        "canonical_registry_artifact_type",
    ):
        require(
            evidence_bindings.get(binding) == reservation_bindings.get(binding),
            f"synthetic producer binding mismatch: {binding}",
        )
    for field, expected in (
        ("positive_families", REQUIRED_POSITIVE_FAMILIES),
        ("identity_null_families", REQUIRED_IDENTITY_NULL_FAMILIES),
        ("joint_null_families", REQUIRED_JOINT_NULL_FAMILIES),
        ("diagnostic_families", REQUIRED_DIAGNOSTIC_FAMILIES),
    ):
        families = evidence.get(field)
        require(
            isinstance(families, list)
            and len(families) == len(expected)
            and set(families) == expected,
            f"synthetic {field} coverage mismatch",
        )
    policy = reservation.get("selection_policy")
    require(isinstance(policy, Mapping), "selection policy missing")
    alpha = float(policy["one_sided_clopper_pearson_alpha"])
    minimum_positive_lower = float(policy["minimum_known_identity_retention_lower_bound"])
    maximum_negative_upper = float(policy["maximum_false_eligible_upper_bound"])
    require(alpha == 0.05, "synthetic gate requires one-sided 95% Clopper-Pearson")
    require(minimum_positive_lower == 0.90, "positive LCB must be frozen at 0.90")
    require(maximum_negative_upper == 0.01, "null UCB must be frozen at 0.01")
    require(0.0 <= minimum_positive_lower <= 1.0, "positive policy invalid")
    require(0.0 <= maximum_negative_upper <= 1.0, "negative policy invalid")
    require(policy.get("tuple_selection") == "minimum-severity-sum-then-frozen-field-lexicographic-v1", "selection rule mismatch")
    field_order = policy.get("frozen_threshold_field_order")
    require(isinstance(field_order, list) and set(field_order) == THRESHOLD_KEYS, "threshold order mismatch")

    grid = reservation.get("threshold_grid")
    axes = reservation.get("threshold_axes")
    rows = evidence.get("rows")
    require(isinstance(grid, list) and grid, "threshold grid missing")
    require(
        isinstance(axes, Mapping)
        and reservation.get("threshold_grid_construction")
        == "cartesian-ordered-axes-severity-index-v1"
        and grid == canonical_threshold_grid_rows(axes),
        "threshold grid Cartesian construction mismatch",
    )
    require(isinstance(rows, list) and len(rows) == len(grid), "synthetic rows mismatch")
    grid_by_rank: dict[int, Mapping[str, Any]] = {}
    for value in grid:
        require(isinstance(value, Mapping), "threshold grid row must be an object")
        require(
            set(value) == {"permissiveness_rank", "thresholds", "severity_components"},
            "threshold grid row schema mismatch",
        )
        rank = int(value["permissiveness_rank"])
        thresholds = value.get("thresholds")
        require(isinstance(thresholds, Mapping) and set(thresholds) == THRESHOLD_KEYS, "grid threshold mismatch")
        severity = value.get("severity_components")
        require(
            isinstance(severity, list)
            and len(severity) == len(field_order)
            and all(math.isfinite(float(component)) for component in severity),
            "grid severity components mismatch",
        )
        require(
            int(thresholds["minimum_longest_trainable_segment_frames"]) == 48
            and int(thresholds["maximum_candidate_window_frames"]) == 24,
            "window/segment constants must be 24/48",
        )
        require(rank not in grid_by_rank, "duplicate permissiveness rank")
        grid_by_rank[rank] = value
    require(set(grid_by_rank) == set(range(len(grid))), "grid ranks are not an exact permutation")
    severity_order = sorted(
        grid_by_rank,
        key=lambda rank: (
            sum(float(value) for value in grid_by_rank[rank]["severity_components"]),
            *(float(value) for value in grid_by_rank[rank]["severity_components"]),
        ),
    )
    require(severity_order == list(range(len(grid))), "grid ranks do not match frozen severity order")

    passing: list[tuple[int, Mapping[str, Any], dict[str, Any]]] = []
    audited_rows: list[dict[str, Any]] = []
    seen_row_ranks: set[int] = set()
    calibration_seed_identities: set[str] = set()
    heldout_seed_identities: set[str] = set()
    for value in rows:
        require(isinstance(value, Mapping), "synthetic evidence row must be an object")
        require(
            set(value) == {"permissiveness_rank", "thresholds", "calibration", "heldout"},
            "synthetic evidence row schema mismatch",
        )
        rank = int(value["permissiveness_rank"])
        require(rank in grid_by_rank, "evidence rank absent from grid")
        require(rank not in seen_row_ranks, "duplicate synthetic evidence rank")
        seen_row_ranks.add(rank)
        require(value.get("thresholds") == grid_by_rank[rank].get("thresholds"), "evidence thresholds changed")
        split_results: dict[str, Any] = {}
        split_passes: dict[str, bool] = {}
        for split in ("calibration", "heldout"):
            counts = value.get(split)
            require(isinstance(counts, Mapping), f"{split} counts missing")
            require(
                set(counts)
                == {
                    "positive_families",
                    "identity_null_families",
                    "joint_null_families",
                    "diagnostic_families",
                    "determinism_failures",
                    "determinism_failure_bitset_hex",
                    "candidate_order_permutation_failures",
                    "candidate_order_permutation_failure_bitset_hex",
                    "seed_identity_sha256",
                },
                f"{split} result schema mismatch",
            )
            positives = counts.get("positive_families")
            identity_nulls = counts.get("identity_null_families")
            joint_nulls = counts.get("joint_null_families")
            diagnostics = counts.get("diagnostic_families")
            require(
                isinstance(positives, Mapping)
                and set(positives) == REQUIRED_POSITIVE_FAMILIES,
                "positive families mismatch",
            )
            require(
                isinstance(identity_nulls, Mapping)
                and set(identity_nulls) == REQUIRED_IDENTITY_NULL_FAMILIES,
                "identity-null families mismatch",
            )
            require(
                isinstance(joint_nulls, Mapping)
                and set(joint_nulls) == REQUIRED_JOINT_NULL_FAMILIES,
                "joint-null families mismatch",
            )
            require(
                isinstance(diagnostics, Mapping)
                and set(diagnostics) == REQUIRED_DIAGNOSTIC_FAMILIES,
                "diagnostic families mismatch",
            )
            positive_bounds: dict[str, float] = {}
            for family, family_counts in positives.items():
                require(isinstance(family_counts, Mapping), "positive counts invalid")
                require(
                    set(family_counts) == {"total", "retained", "decision_bitset_hex"},
                    "positive family count schema mismatch",
                )
                total = int(family_counts["total"])
                require(total == 512, "each positive family requires 512 samples")
                require(
                    int(family_counts["retained"])
                    == _decision_bitset_count(family_counts["decision_bitset_hex"]),
                    "positive count/bitset mismatch",
                )
                lower, _ = _clopper_pearson_one_sided(
                    int(family_counts["retained"]), total, alpha=alpha
                )
                positive_bounds[str(family)] = lower
            null_bounds: dict[str, float] = {}
            for family, family_counts in {
                **dict(identity_nulls),
                **dict(joint_nulls),
            }.items():
                require(isinstance(family_counts, Mapping), "null counts invalid")
                require(
                    set(family_counts) == {"total", "false_eligible", "decision_bitset_hex"},
                    "null family count schema mismatch",
                )
                total = int(family_counts["total"])
                require(total == 512, "each null family requires 512 samples")
                require(
                    int(family_counts["false_eligible"])
                    == _decision_bitset_count(family_counts["decision_bitset_hex"]),
                    "null count/bitset mismatch",
                )
                _, upper = _clopper_pearson_one_sided(
                    int(family_counts["false_eligible"]), total, alpha=alpha
                )
                null_bounds[str(family)] = upper
            diagnostic_failures = 0
            for family_counts in diagnostics.values():
                require(isinstance(family_counts, Mapping), "diagnostic counts invalid")
                require(
                    set(family_counts) == {"total", "period_supported", "decision_bitset_hex"},
                    "diagnostic family count schema mismatch",
                )
                require(int(family_counts["total"]) == 512, "each diagnostic family requires 512 samples")
                require(
                    int(family_counts["period_supported"])
                    == _decision_bitset_count(family_counts["decision_bitset_hex"]),
                    "diagnostic count/bitset mismatch",
                )
                diagnostic_failures += int(family_counts["period_supported"])
            deterministic_failures = int(counts["determinism_failures"])
            permutation_failures = int(counts["candidate_order_permutation_failures"])
            require(
                deterministic_failures
                == _decision_bitset_count(counts["determinism_failure_bitset_hex"])
                and permutation_failures
                == _decision_bitset_count(
                    counts["candidate_order_permutation_failure_bitset_hex"]
                ),
                "determinism/permutation count bitsets mismatch",
            )
            split_passed = (
                all(bound >= minimum_positive_lower for bound in positive_bounds.values())
                and all(bound <= maximum_negative_upper for bound in null_bounds.values())
                and deterministic_failures == 0
                and permutation_failures == 0
                and diagnostic_failures == 0
            )
            split_results[split] = {
                "positive_family_clopper_pearson_lower": positive_bounds,
                "identity_null_family_clopper_pearson_upper": null_bounds,
                "determinism_failures": deterministic_failures,
                "candidate_order_permutation_failures": permutation_failures,
                "diagnostic_period_supported_failures": diagnostic_failures,
                "passed": split_passed,
            }
            split_passes[split] = split_passed
        audited = {
            "permissiveness_rank": rank,
            "splits": split_results,
            "calibration_feasible": split_passes["calibration"],
            "heldout_passed": split_passes["heldout"],
        }
        audited_rows.append(audited)
        require(
            value["calibration"]["seed_identity_sha256"]
            != value["heldout"]["seed_identity_sha256"],
            "calibration and heldout seed identities overlap",
        )
        calibration_seed_identities.add(str(value["calibration"]["seed_identity_sha256"]))
        heldout_seed_identities.add(str(value["heldout"]["seed_identity_sha256"]))
        if split_passes["calibration"]:
            passing.append((rank, grid_by_rank[rank]["thresholds"], audited))
    require(seen_row_ranks == set(grid_by_rank), "synthetic evidence rank permutation mismatch")
    require(
        len(calibration_seed_identities) == 1
        and len(heldout_seed_identities) == 1
        and calibration_seed_identities.isdisjoint(heldout_seed_identities),
        "synthetic split seed identities are inconsistent or overlapping",
    )
    require(passing, "no synthetic threshold tuple satisfies preregistered bounds")
    def selection_key(item: tuple[int, Mapping[str, Any], dict[str, Any]]) -> tuple[Any, ...]:
        rank, thresholds, _ = item
        grid_row = grid_by_rank[rank]
        components = tuple(float(value) for value in grid_row["severity_components"])
        return (sum(components), *components)

    ordered = sorted(passing, key=selection_key)
    require(
        len(ordered) == 1 or selection_key(ordered[0]) != selection_key(ordered[1]),
        "synthetic selected threshold tuple is not unique",
    )
    selected_rank, frozen_thresholds, _ = ordered[0]
    selected_audit = next(
        row for row in audited_rows if row["permissiveness_rank"] == selected_rank
    )
    require(
        selected_audit["heldout_passed"] is True,
        "calibration-selected threshold tuple failed heldout; no fallback is allowed",
    )

    receipt = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_synthetic_threshold_receipt_v1",
        "status": "passed",
        "overall_pass": True,
        "label_free": True,
        "threshold_origin": "synthetic-frozen-generator-seed-split-preset-grid-v1",
        "calibration_passed": True,
        "heldout_passed": True,
        "heldout_role": "selected-tuple-validation-only-never-used-for-selection",
        "frozen_thresholds": dict(frozen_thresholds),
        "selected_permissiveness_rank": selected_rank,
        "selection_policy": dict(policy),
        "audited_grid_rows": audited_rows,
        "positive_families": sorted(REQUIRED_POSITIVE_FAMILIES),
        "identity_null_families": sorted(REQUIRED_IDENTITY_NULL_FAMILIES),
        "joint_null_families": sorted(REQUIRED_JOINT_NULL_FAMILIES),
        "diagnostic_families": sorted(REQUIRED_DIAGNOSTIC_FAMILIES),
        "bindings": {
            **dict(reservation_bindings),
            "synthetic_reservation_sha256": expected_reservation_sha256,
            "synthetic_grid_evidence_sha256": expected_evidence_sha256,
        },
        "same39_raw_extraction_authorized": True,
        "full337_raw_extraction_authorized": False,
        "cycleback_representation_input_authorized": False,
        "baseline_training_authorized": False,
    }
    write_json_exclusive(output_path, receipt)
    os.chmod(output_path, 0o444)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--reservation-sha256", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--evidence-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = gate_synthetic_thresholds(
            reservation_path=args.reservation,
            expected_reservation_sha256=args.reservation_sha256,
            evidence_path=args.evidence,
            expected_evidence_sha256=args.evidence_sha256,
            output_path=args.output,
        )
    except (OSError, TypeError, ValueError, Full337ContractError, KeyError) as exc:
        print(f"v4e synthetic threshold gate failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
