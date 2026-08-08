#!/usr/bin/env python3
"""Validate frozen-threshold same39 raw evidence; never select thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
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


def _object(path: Path, role: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, Mapping), f"{role} must be a JSON object")
    return value


def gate_same39(
    *,
    authorization_path: Path,
    expected_authorization_sha256: str,
    synthetic_receipt_path: Path,
    expected_synthetic_receipt_sha256: str,
    raw_ledger_path: Path,
    pose_snapshot_path: Path,
    segment_index_path: Path,
    segment_policy_path: Path,
    joint_mask_snapshot_path: Path,
    pair_eligibility_path: Path,
    identity_map_path: Path,
    replay_receipt_path: Path,
    expected_replay_receipt_sha256: str,
    overlap_receipt_path: Path,
    expected_overlap_receipt_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    require(sha256_file(authorization_path) == expected_authorization_sha256, "same39 auth SHA mismatch")
    require(sha256_file(synthetic_receipt_path) == expected_synthetic_receipt_sha256, "synthetic receipt SHA mismatch")
    authorization = _object(authorization_path, "same39 authorization")
    synthetic = _object(synthetic_receipt_path, "synthetic receipt")
    require(
        authorization.get("artifact_type") == "pams_pose_recovery_v4e_raw_extraction_authorization_v1"
        and authorization.get("authorization_scope")
        == "canonical_train337_same39_source_video_kprcnn_raw_evidence_only"
        and authorization.get("status") == "authorized",
        "same39 authorization mismatch",
    )
    auth_bindings = authorization.get("bindings")
    require(isinstance(auth_bindings, Mapping), "same39 authorization bindings missing")
    outcome_locator = Path(str(auth_bindings["canonical_outcome_locator"]))
    require(
        authorization_path.resolve(strict=True)
        == outcome_locator / "authorization/v4e-same39-raw.authorization.json"
        and stat.S_IMODE(authorization_path.stat().st_mode) & 0o222 == 0,
        "same39 authorization canonical locator/sealing mismatch",
    )
    require(
        synthetic_receipt_path.resolve(strict=True)
        == outcome_locator / "gate-output/v4e-synthetic-threshold.receipt.json"
        and stat.S_IMODE(synthetic_receipt_path.stat().st_mode) & 0o222 == 0,
        "synthetic receipt canonical locator/sealing mismatch",
    )
    require(
        output_path.parent.resolve(strict=True) / output_path.name
        == outcome_locator / "gate-output/v4e-same39.gate.json",
        "same39 gate output locator mismatch",
    )
    pilot_root = outcome_locator / "pilot/same39"
    expected_inputs = {
        raw_ledger_path: pilot_root / "output/raw-ledger.json",
        pose_snapshot_path: pilot_root / "output/pose-cache-set.snapshot.json",
        segment_index_path: pilot_root / "output/segment-index.json",
        segment_policy_path: pilot_root / "output/segment-reset.policy.json",
        joint_mask_snapshot_path: pilot_root / "output/joint-mask-set.snapshot.json",
        pair_eligibility_path: pilot_root / "output/cycleback-pair-eligibility.json",
        identity_map_path: pilot_root / "output/identity-map.json",
        replay_receipt_path: pilot_root / "output/pose-cache-validation.receipt.json",
        overlap_receipt_path: pilot_root / "output/prefix-suffix-overlap.receipt.json",
    }
    require(
        all(path.resolve(strict=True) == expected for path, expected in expected_inputs.items()),
        "same39 gate input locator contract mismatch",
    )
    require(
        auth_bindings.get("synthetic_threshold_receipt_sha256")
        == expected_synthetic_receipt_sha256,
        "same39 authorization did not bind synthetic receipt",
    )
    authorized_members = authorization.get("selected_video_id_sha256")
    authorized_zero11 = authorization.get("selected_zero11_video_id_sha256")
    require(
        isinstance(authorized_members, list)
        and len(authorized_members) == 39
        and len(set(str(value) for value in authorized_members)) == 39,
        "same39 authorization membership invalid",
    )
    require(
        isinstance(authorized_zero11, list)
        and len(authorized_zero11) == 11
        and set(str(value) for value in authorized_zero11)
        <= set(str(value) for value in authorized_members),
        "zero11 authorization membership invalid",
    )
    require(
        auth_bindings.get("zero11_record_total") == 11
        and isinstance(auth_bindings.get("zero11_identity_sha256"), str),
        "zero11 authorization binding missing",
    )
    require(
        synthetic.get("artifact_type") == "pams_pose_recovery_v4e_synthetic_threshold_receipt_v1"
        and synthetic.get("overall_pass") is True,
        "synthetic threshold receipt did not pass",
    )
    require(
        authorization.get("frozen_thresholds") == synthetic.get("frozen_thresholds"),
        "same39 thresholds differ from synthetic receipt",
    )
    policy = synthetic.get("selection_policy")
    require(isinstance(policy, Mapping), "same39 utility policy missing")
    minimum_eligible_fraction = float(
        policy["minimum_same39_representation_eligible_fraction"]
    )
    minimum_overlap_agreement = float(
        policy["minimum_same39_prefix_suffix_path_agreement"]
    )
    require(0.0 <= minimum_eligible_fraction <= 1.0, "same39 utility threshold invalid")

    ledger_sha256 = sha256_file(raw_ledger_path)
    ledger = _object(raw_ledger_path, "same39 raw ledger")
    require(
        ledger.get("artifact_type") == "pams_pose_recovery_v4e_same39_raw_ledger_v1"
        and ledger.get("status") == "complete"
        and ledger.get("selected") == 39
        and ledger.get("completed") == 39
        and ledger.get("failed") == 0,
        "same39 raw ledger incomplete",
    )
    require(
        ledger.get("raw_extraction_authorization_sha256") == expected_authorization_sha256,
        "same39 ledger authorization mismatch",
    )
    rows = ledger.get("caches")
    require(isinstance(rows, list) and len(rows) == 39, "same39 cache rows mismatch")
    eligible = 0
    for value in rows:
        require(isinstance(value, Mapping), "same39 row must be an object")
        for prefix in (
            "candidate_evidence",
            "raw_detector_evidence",
            "path_segment_evidence",
            "joint_valid_mask",
        ):
            artifact_path = Path(str(value[f"{prefix}_path"])).resolve(strict=True)
            require(sha256_file(artifact_path) == value[f"{prefix}_sha256"], f"{prefix} SHA mismatch")
            require(artifact_path.stat().st_size == value[f"{prefix}_bytes"], f"{prefix} bytes mismatch")
    replay_receipt_sha256 = sha256_file(replay_receipt_path)
    require(replay_receipt_sha256 == expected_replay_receipt_sha256, "replay receipt SHA mismatch")
    replay_receipt = _object(replay_receipt_path, "same39 replay receipt")
    require(
        replay_receipt.get("artifact_type")
        == "pams_cycleback_unified_2d_pose_cache_validation_receipt_v1"
        and replay_receipt.get("status") == "passed"
        and replay_receipt.get("overall_pass") is True
        and replay_receipt.get("entry_count") == 39,
        "same39 independent replay did not pass",
    )
    replay_bindings = replay_receipt.get("bindings")
    require(isinstance(replay_bindings, Mapping), "replay bindings missing")
    require(replay_bindings.get("raw_ledger_sha256") == ledger_sha256, "replay ledger binding mismatch")
    replay_rows = replay_receipt.get("replay_rows")
    require(isinstance(replay_rows, list) and len(replay_rows) == 39, "replay rows mismatch")
    eligible = sum(row.get("representation_eligible") is True for row in replay_rows)
    replay_ids = {str(row["video_id_sha256"]) for row in replay_rows}
    require(len(replay_ids) == 39, "same39 replay identities invalid")
    require(
        replay_ids == set(str(value) for value in authorized_members),
        "same39 replay membership differs from authorization",
    )
    require(
        all(int(row.get("eligible_pair_start_count", 0)) > 0 for row in replay_rows if row.get("representation_eligible") is True),
        "eligible same39 replay row lacks exact 2W pair",
    )
    eligible_fraction = eligible / 39.0
    utility_passed = eligible_fraction >= minimum_eligible_fraction

    overlap_receipt_sha256 = sha256_file(overlap_receipt_path)
    require(overlap_receipt_sha256 == expected_overlap_receipt_sha256, "overlap receipt SHA mismatch")
    overlap = _object(overlap_receipt_path, "same39 overlap receipt")
    require(
        overlap.get("artifact_type") == "pams_pose_recovery_v4e_same39_prefix_suffix_overlap_replay_v1"
        and overlap.get("entry_count") == 39,
        "same39 prefix/suffix overlap replay schema mismatch",
    )
    overlap_bindings = overlap.get("bindings")
    require(isinstance(overlap_bindings, Mapping), "overlap bindings missing")
    require(
        overlap_bindings.get("raw_ledger_sha256") == ledger_sha256
        and overlap_bindings.get("independent_replay_receipt_sha256")
        == replay_receipt_sha256,
        "overlap replay binding mismatch",
    )
    overlap_rows = overlap.get("rows")
    require(
        isinstance(overlap_rows, list) and len(overlap_rows) == 39,
        "same39 overlap replay rows missing",
    )
    overlap_ids: set[str] = set()
    zero11_total = 0
    observed_zero11: set[str] = set()
    overlap_passed = True
    for row in overlap_rows:
        require(isinstance(row, Mapping), "same39 overlap row must be an object")
        opaque_id = str(row["video_id_sha256"])
        require(len(opaque_id) == 64 and opaque_id not in overlap_ids, "overlap identity invalid")
        overlap_ids.add(opaque_id)
        require(int(row["overlap_observed_frames"]) > 0, "overlap has no observed frames")
        primary_agreement = float(row["prefix_suffix_primary_path_agreement"])
        secondary_agreement = float(row["prefix_suffix_secondary_path_agreement"])
        overlap_passed = overlap_passed and (
            primary_agreement >= minimum_overlap_agreement
            and secondary_agreement >= minimum_overlap_agreement
        )
        if row.get("v4a_zero_anchor") is True:
            zero11_total += 1
            observed_zero11.add(opaque_id)
    require(zero11_total == 11, "same39 overlap zero11 stratum mismatch")
    require(overlap_ids == replay_ids, "overlap/replay identity sets differ")
    require(
        observed_zero11 == set(str(value) for value in authorized_zero11),
        "zero11 overlap membership differs from authorization",
    )
    require(
        overlap_bindings.get("zero11_identity_sha256")
        == auth_bindings.get("zero11_identity_sha256")
        and overlap_bindings.get("zero11_record_total") == 11,
        "zero11 overlap binding mismatch",
    )

    def recompute_stratum(group_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
        member_ids = sorted(str(row["video_id_sha256"]) for row in group_rows)
        return {
            "record_total": len(group_rows),
            "opaque_member_set_sha256": hashlib.sha256(
                json.dumps(
                    member_ids,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest(),
            "representation_eligible_total": sum(
                row["representation_eligible"] is True for row in group_rows
            ),
            "with_eligible_pair_total": sum(
                int(row["eligible_pair_start_count"]) > 0 for row in group_rows
            ),
            "minimum_primary_overlap_agreement": min(
                float(row["prefix_suffix_primary_path_agreement"]) for row in group_rows
            ),
            "minimum_secondary_overlap_agreement": min(
                float(row["prefix_suffix_secondary_path_agreement"]) for row in group_rows
            ),
            "mean_valid_frame_fraction": sum(
                float(row["valid_frame_fraction"]) for row in group_rows
            ) / len(group_rows),
            "mean_active_joint_count": sum(
                float(row["mean_active_joint_count"]) for row in group_rows
            ) / len(group_rows),
            "mean_active_action_joint_count": sum(
                float(row["mean_active_action_joint_count"]) for row in group_rows
            ) / len(group_rows),
        }

    overlap_strata = overlap.get("strata")
    require(isinstance(overlap_strata, Mapping), "same39 overlap strata missing")
    zero_rows = [row for row in overlap_rows if row.get("v4a_zero_anchor") is True]
    other_rows = [row for row in overlap_rows if row.get("v4a_zero_anchor") is False]
    require(
        overlap_strata
        == {
            "v4a_zero11": recompute_stratum(zero_rows),
            "v4a_nonzero28": recompute_stratum(other_rows),
        },
        "same39 overlap stratum statistics do not replay",
    )

    pose_snapshot_sha256 = sha256_file(pose_snapshot_path)
    segment_index_sha256 = sha256_file(segment_index_path)
    segment_policy_sha256 = sha256_file(segment_policy_path)
    joint_mask_snapshot_sha256 = sha256_file(joint_mask_snapshot_path)
    pair_eligibility_sha256 = sha256_file(pair_eligibility_path)
    identity_map_sha256 = sha256_file(identity_map_path)
    pose_snapshot = _object(pose_snapshot_path, "same39 pose snapshot")
    segment_index = _object(segment_index_path, "same39 segment index")
    joint_snapshot = _object(joint_mask_snapshot_path, "same39 joint-mask snapshot")
    pair_eligibility = _object(pair_eligibility_path, "same39 pair eligibility")
    identity_map = _object(identity_map_path, "same39 identity map")
    require(pose_snapshot.get("entry_count") == 39, "same39 pose snapshot count mismatch")
    require(segment_index.get("entry_count") == 39, "same39 segment index count mismatch")
    require(joint_snapshot.get("entry_count") == 39, "same39 joint-mask snapshot count mismatch")
    require(pair_eligibility.get("entry_count") == 39, "same39 pair eligibility count mismatch")
    identity_entries = identity_map.get("entries")
    require(
        identity_map.get("artifact_type") == "pams_pose_recovery_v4e_identity_map_v1"
        and identity_map.get("entry_count") == 39
        and isinstance(identity_entries, list)
        and len(identity_entries) == 39,
        "same39 identity-map schema mismatch",
    )
    identity_opaque = {
        str(entry["video_id_sha256"])
        for entry in identity_entries
        if isinstance(entry, Mapping)
    }
    require(
        identity_opaque == replay_ids == set(str(value) for value in authorized_members),
        "same39 identity-map/replay/authorization membership mismatch",
    )
    require(ledger.get("segment_index_sha256") == segment_index_sha256, "segment index binding mismatch")
    require(ledger.get("segment_reset_policy_sha256") == segment_policy_sha256, "segment policy binding mismatch")
    require(ledger.get("joint_mask_snapshot_sha256") == joint_mask_snapshot_sha256, "joint snapshot binding mismatch")
    require(ledger.get("cycleback_pair_eligibility_sha256") == pair_eligibility_sha256, "pair eligibility binding mismatch")
    require(ledger.get("identity_map_sha256") == identity_map_sha256, "identity map binding mismatch")

    overall_pass = bool(utility_passed and overlap_passed)
    result = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_same39_gate_v1",
        "status": "passed" if overall_pass else "denied",
        "overall_pass": overall_pass,
        "label_free": True,
        "threshold_origin": "synthetic-frozen-generator-seed-split-preset-grid-v1",
        "synthetic_threshold_receipt_verified": True,
        "same39_threshold_selection_used": False,
        "same39_validation_passed": overall_pass,
        "same39_representation_eligible": eligible,
        "same39_representation_eligible_fraction": eligible_fraction,
        "minimum_same39_representation_eligible_fraction": minimum_eligible_fraction,
        "minimum_same39_prefix_suffix_path_agreement": minimum_overlap_agreement,
        "same39_prefix_suffix_overlap_passed": overlap_passed,
        "same39_zero11_total": zero11_total,
        "frozen_thresholds": synthetic["frozen_thresholds"],
        "bindings": {
            "source_revision": authorization["source_revision"],
            "container_image_id": authorization["container_image_id"],
            "config_file_sha256": auth_bindings["config_file_sha256"],
            "model_asset_sha256": auth_bindings["model_asset_sha256"],
            "same39_identity_sha256": auth_bindings["same39_identity_sha256"],
            "zero11_identity_sha256": auth_bindings["zero11_identity_sha256"],
            "zero11_record_total": 11,
            "canonical_outcome_locator": str(outcome_locator),
            "canonical_registry_reservation_sha256": auth_bindings[
                "canonical_registry_reservation_sha256"
            ],
            "canonical_registry_artifact_type": auth_bindings[
                "canonical_registry_artifact_type"
            ],
            "synthetic_threshold_receipt_sha256": expected_synthetic_receipt_sha256,
            "same39_authorization_sha256": expected_authorization_sha256,
            "same39_raw_ledger_sha256": ledger_sha256,
            "same39_pose_snapshot_sha256": pose_snapshot_sha256,
            "same39_pose_cache_set_sha256": pose_snapshot["fingerprint"],
            "same39_segment_index_sha256": segment_index_sha256,
            "same39_segment_reset_policy_sha256": segment_policy_sha256,
            "same39_joint_mask_snapshot_sha256": joint_mask_snapshot_sha256,
            "same39_joint_mask_set_sha256": joint_snapshot["fingerprint"],
            "same39_cycleback_pair_eligibility_sha256": pair_eligibility_sha256,
            "same39_identity_map_sha256": identity_map_sha256,
            "same39_independent_replay_receipt_sha256": replay_receipt_sha256,
            "same39_prefix_suffix_overlap_receipt_sha256": overlap_receipt_sha256,
        },
        "full337_raw_extraction_authorized": overall_pass,
        "cycleback_representation_input_authorized": False,
        "baseline_training_authorized": False,
    }
    write_json_exclusive(output_path, result)
    os.chmod(output_path, 0o444)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "authorization-sha256", "synthetic-receipt-sha256",
        "replay-receipt-sha256", "overlap-receipt-sha256",
    ):
        parser.add_argument(f"--{flag}", required=True)
    for flag in (
        "authorization", "synthetic-receipt", "raw-ledger", "pose-snapshot",
        "segment-index", "segment-policy", "joint-mask-snapshot",
        "pair-eligibility", "identity-map", "replay-receipt",
        "overlap-receipt", "output",
    ):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = gate_same39(
            authorization_path=args.authorization,
            expected_authorization_sha256=args.authorization_sha256,
            synthetic_receipt_path=args.synthetic_receipt,
            expected_synthetic_receipt_sha256=args.synthetic_receipt_sha256,
            raw_ledger_path=args.raw_ledger,
            pose_snapshot_path=args.pose_snapshot,
            segment_index_path=args.segment_index,
            segment_policy_path=args.segment_policy,
            joint_mask_snapshot_path=args.joint_mask_snapshot,
            pair_eligibility_path=args.pair_eligibility,
            identity_map_path=args.identity_map,
            replay_receipt_path=args.replay_receipt,
            expected_replay_receipt_sha256=args.replay_receipt_sha256,
            overlap_receipt_path=args.overlap_receipt,
            expected_overlap_receipt_sha256=args.overlap_receipt_sha256,
            output_path=args.output,
        )
    except (OSError, TypeError, ValueError, Full337ContractError, KeyError) as exc:
        print(f"v4e same39 gate failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
