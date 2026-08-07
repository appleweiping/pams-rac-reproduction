#!/usr/bin/env python3
"""Issue a v2 extraction-only authorization from frozen same39 measured evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from pose_recovery_v4d_full337_contract import (
    CONTAINER_IMAGE_ID,
    MODEL_ASSET_SHA256,
    PARENT_V4D_SOURCE_REVISION,
    SAME39_IDENTITY_SHA256,
    TRAIN337_COMMITMENT_SHA256,
    TRAIN337_IDENTITY_SHA256,
    TRAIN337_SIDECAR_SHA256,
    V4A_CACHE_SET_SHA256,
    V4A_LEDGER_SHA256,
    V4A_PAIRED_GATE_SHA256,
    V4A_POSE_FINGERPRINT,
    V4D_CONFIG_FILE_SHA256,
    V4D_CONFIG_FINGERPRINT,
    V4D_POSE_FINGERPRINT,
    Full337ContractError,
    require,
    sha256_file,
    validate_full337_gate,
    validate_same39_failed_pilot,
    validate_v4a_inputs,
    write_json_exclusive,
)
from pams.data import (
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
    validate_pose_input_binding,
)


def authorize_extraction_v2(
    *,
    source_revision: str,
    container_image_id: str,
    expected_gate_sha256: str,
    expected_failure_receipt_sha256: str,
    gate_path: Path,
    train_input_path: Path,
    train_commitment_path: Path,
    v4a_cache_dir: Path,
    v4a_ledger_path: Path,
    v4a_paired_gate_path: Path,
    same39_failure_receipt_path: Path,
    same39_audit_path: Path,
    same39_ledger_path: Path,
    same39_selection_path: Path,
    same39_cache_dir: Path,
    output_path: Path,
) -> dict[str, object]:
    """Authorize full extraction, while remaining unusable as training authority."""

    require(
        len(source_revision) == 40
        and all(character in "0123456789abcdef" for character in source_revision),
        "source revision must be lowercase 40-hex",
    )
    require(container_image_id == CONTAINER_IMAGE_ID, "container image ID mismatch")
    require(
        os.environ.get("PAMS_CONTAINER_SOURCE_REVISION") == source_revision,
        "container source-revision environment mismatch",
    )
    require(
        os.environ.get("PAMS_CONTAINER_IMAGE_ID") == container_image_id,
        "container image-ID environment mismatch",
    )
    _, gate_sha256 = validate_full337_gate(
        gate_path,
        expected_sha256=expected_gate_sha256,
    )
    manifest = load_pose_input_manifest(train_input_path, validate_exact=True)
    commitment = load_pose_input_commitment(train_commitment_path)
    sidecar_sha256 = sha256_file(train_input_path)
    commitment_sha256 = sha256_file(train_commitment_path)
    validate_pose_input_binding(manifest, commitment, sidecar_sha256=sidecar_sha256)
    require(
        manifest.split == "train" and len(manifest.records) == 337,
        "v2 authorizer accepts only train337",
    )
    require(sidecar_sha256 == TRAIN337_SIDECAR_SHA256, "train337 sidecar SHA mismatch")
    require(commitment_sha256 == TRAIN337_COMMITMENT_SHA256, "train337 commitment SHA mismatch")
    require(
        pose_input_identity_sha256(manifest.records) == TRAIN337_IDENTITY_SHA256,
        "train337 identity mismatch",
    )
    v4a = validate_v4a_inputs(
        manifest.records,
        cache_dir=v4a_cache_dir,
        ledger_path=v4a_ledger_path,
        paired_gate_path=v4a_paired_gate_path,
    )
    evidence = validate_same39_failed_pilot(
        manifest.records,
        expected_failure_receipt_sha256=expected_failure_receipt_sha256,
        failure_receipt_path=same39_failure_receipt_path,
        audit_path=same39_audit_path,
        ledger_path=same39_ledger_path,
        selection_path=same39_selection_path,
        cache_dir=same39_cache_dir,
    )
    require(v4a.snapshot.fingerprint == V4A_CACHE_SET_SHA256, "v4a bytewise check failed")
    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4d_full337_extraction_authorization_v2",
        "authorization_protocol_version": 2,
        "authorization_scope": "full337_pose_extraction_only",
        "source_revision": source_revision,
        "container_image_id": container_image_id,
        "protocol": "ucfrep_526",
        "split": "train",
        "label_free": True,
        "full337_pose_extraction_authorized": True,
        "baseline_training_authorized": False,
        "training_runner_must_reject": True,
        "training_authorization_requires": (
            "unchanged-full337-label-free-track-quality-periodicity-gate-pass"
        ),
        "authorization_basis": {
            "actual_criteria_names": sorted(
                {
                    "base_v4a_mask_mismatch_total",
                    "candidate_at_most_8_video_total",
                    "candidate_longest_run_fraction_p25",
                    "candidate_zero_video_total",
                    "invariant_failure_total",
                    "keypointrcnn_fill_frames_total",
                    "longest_run_fraction_mean_gain_over_v4a",
                    "recovered_base_zero_video_total",
                    "source_coverage_mean_gain_over_v4a",
                }
            ),
            "old_pilot_passed": False,
            "old_full337_pose_extraction_authorized": False,
            "zero11_passed": True,
            "same39_cost_gate_passed": True,
            "projection_consulted_for_decision": False,
        },
        "bindings": {
            "full337_gate_sha256": gate_sha256,
            "same39_source_revision": PARENT_V4D_SOURCE_REVISION,
            "same39_failure_receipt_sha256": evidence.failure_receipt_sha256,
            "same39_audit_sha256": evidence.audit_sha256,
            "same39_ledger_sha256": evidence.ledger_sha256,
            "same39_selection_sha256": evidence.selection_sha256,
            "same39_identity_sha256": SAME39_IDENTITY_SHA256,
            "same39_candidate_cache_set_sha256": evidence.snapshot.fingerprint,
            "container_image_id": CONTAINER_IMAGE_ID,
            "v4d_config_file_sha256": V4D_CONFIG_FILE_SHA256,
            "v4d_config_fingerprint": V4D_CONFIG_FINGERPRINT,
            "v4d_pose_fingerprint": V4D_POSE_FINGERPRINT,
            "keypointrcnn_model_asset_sha256": MODEL_ASSET_SHA256,
            "train337_sidecar_sha256": TRAIN337_SIDECAR_SHA256,
            "train337_commitment_sha256": TRAIN337_COMMITMENT_SHA256,
            "train337_identity_sha256": TRAIN337_IDENTITY_SHA256,
            "v4a_ledger_sha256": V4A_LEDGER_SHA256,
            "v4a_paired_gate_sha256": V4A_PAIRED_GATE_SHA256,
            "v4a_pose_fingerprint": V4A_POSE_FINGERPRINT,
            "v4a_pose_cache_set_sha256": V4A_CACHE_SET_SHA256,
        },
    }
    write_json_exclusive(output_path, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--same39-failure-receipt-sha256", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-commitment", type=Path, required=True)
    parser.add_argument("--v4a-cache-dir", type=Path, required=True)
    parser.add_argument("--v4a-ledger", type=Path, required=True)
    parser.add_argument("--v4a-paired-gate", type=Path, required=True)
    parser.add_argument("--same39-failure-receipt", type=Path, required=True)
    parser.add_argument("--same39-audit", type=Path, required=True)
    parser.add_argument("--same39-ledger", type=Path, required=True)
    parser.add_argument("--same39-selection", type=Path, required=True)
    parser.add_argument("--same39-cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = authorize_extraction_v2(
            source_revision=args.source_revision,
            container_image_id=args.container_image_id,
            expected_gate_sha256=args.gate_sha256,
            expected_failure_receipt_sha256=args.same39_failure_receipt_sha256,
            gate_path=args.gate,
            train_input_path=args.train_input,
            train_commitment_path=args.train_commitment,
            v4a_cache_dir=args.v4a_cache_dir,
            v4a_ledger_path=args.v4a_ledger,
            v4a_paired_gate_path=args.v4a_paired_gate,
            same39_failure_receipt_path=args.same39_failure_receipt,
            same39_audit_path=args.same39_audit,
            same39_ledger_path=args.same39_ledger,
            same39_selection_path=args.same39_selection,
            same39_cache_dir=args.same39_cache_dir,
            output_path=args.output,
        )
    except (OSError, TypeError, ValueError, Full337ContractError) as exc:
        print(f"v4d extraction authorization v2 failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
