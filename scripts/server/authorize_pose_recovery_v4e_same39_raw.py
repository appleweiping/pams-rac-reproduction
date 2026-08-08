#!/usr/bin/env python3
"""Authorize only frozen same39 raw extraction from a synthetic threshold receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from pose_recovery_v4d_full337_contract import (
    MODEL_ASSET_SHA256,
    TRAIN337_COMMITMENT_SHA256,
    TRAIN337_IDENTITY_SHA256,
    TRAIN337_SIDECAR_SHA256,
    V4A_CACHE_SET_SHA256,
    V4A_LEDGER_SHA256,
    V4A_PAIRED_GATE_SHA256,
    V4A_POSE_FINGERPRINT,
    Full337ContractError,
    require,
    sha256_file,
    validate_v4a_inputs,
    write_json_exclusive,
)

from pams.config import load_config
from pams.data import (
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
    validate_pose_input_binding,
)
from pams.keypoint_single_source import V4E_PREPROCESSING_REVISION

THRESHOLD_KEYS = {
    "maximum_frame_center_step",
    "maximum_frame_log_scale_step",
    "maximum_frame_joint_mask_flicker_fraction",
    "minimum_dual_path_agreement",
    "minimum_frame_local_ambiguity_gap",
    "minimum_longest_trainable_segment_fraction",
    "minimum_longest_trainable_segment_frames",
    "minimum_source_coverage",
    "maximum_candidate_window_frames",
    "minimum_window_joint_support_fraction",
    "minimum_window_stable_action_joints",
}


def authorize_same39(
    *,
    source_revision: str,
    container_image_id: str,
    config_path: Path,
    train_input_path: Path,
    train_commitment_path: Path,
    frozen_selection_path: Path,
    synthetic_threshold_receipt_path: Path,
    expected_synthetic_threshold_receipt_sha256: str,
    v4a_cache_dir: Path,
    v4a_ledger_path: Path,
    v4a_paired_gate_path: Path,
    model_asset_path: Path,
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
    config_file_sha256 = sha256_file(config_path)
    config = load_config(config_path.resolve(strict=True))
    require(config.pose.preprocessing_revision == V4E_PREPROCESSING_REVISION, "config is not v4e")
    settings = config.pose.keypoint_single_source
    require(settings is not None and not settings.raw_extraction_authorizes_training, "unsafe config")
    require(settings.base_pose_fingerprint == V4A_POSE_FINGERPRINT, "v4a pose mismatch")
    require(settings.base_pose_cache_set_sha256 == V4A_CACHE_SET_SHA256, "v4a set mismatch")
    require(settings.base_pose_ledger_sha256 == V4A_LEDGER_SHA256, "v4a ledger mismatch")
    require(settings.base_paired_gate_sha256 == V4A_PAIRED_GATE_SHA256, "v4a gate mismatch")
    require(sha256_file(model_asset_path) == MODEL_ASSET_SHA256, "model asset mismatch")

    synthetic_sha256 = sha256_file(synthetic_threshold_receipt_path)
    require(synthetic_sha256 == expected_synthetic_threshold_receipt_sha256, "synthetic receipt SHA mismatch")
    synthetic = json.loads(synthetic_threshold_receipt_path.read_text(encoding="utf-8"))
    require(isinstance(synthetic, Mapping), "synthetic receipt must be an object")
    require(
        synthetic.get("artifact_type") == "pams_pose_recovery_v4e_synthetic_threshold_receipt_v1"
        and synthetic.get("status") == "passed"
        and synthetic.get("overall_pass") is True,
        "synthetic threshold gate did not pass",
    )
    require(
        synthetic.get("threshold_origin")
        == "synthetic-frozen-generator-seed-split-preset-grid-v1",
        "synthetic threshold origin mismatch",
    )
    require(
        synthetic.get("calibration_passed") is True
        and synthetic.get("heldout_passed") is True,
        "synthetic calibration or heldout gate failed",
    )
    synthetic_bindings = synthetic.get("bindings")
    require(isinstance(synthetic_bindings, Mapping), "synthetic bindings missing")
    require(synthetic_bindings.get("source_revision") == source_revision, "synthetic source mismatch")
    require(synthetic_bindings.get("container_image_id") == container_image_id, "synthetic image mismatch")
    require(synthetic_bindings.get("config_file_sha256") == config_file_sha256, "synthetic config mismatch")
    require(synthetic_bindings.get("model_asset_sha256") == MODEL_ASSET_SHA256, "synthetic model mismatch")
    outcome_locator = Path(str(synthetic_bindings["canonical_outcome_locator"]))
    require(
        synthetic_threshold_receipt_path.resolve(strict=True)
        == outcome_locator / "gate-output/v4e-synthetic-threshold.receipt.json"
        and stat.S_IMODE(synthetic_threshold_receipt_path.stat().st_mode) & 0o222 == 0,
        "synthetic receipt canonical locator/sealing mismatch",
    )
    require(
        output_path.parent.resolve(strict=True) / output_path.name
        == outcome_locator / "authorization/v4e-same39-raw.authorization.json",
        "same39 authorization output locator mismatch",
    )
    thresholds = synthetic.get("frozen_thresholds")
    require(isinstance(thresholds, Mapping) and set(thresholds) == THRESHOLD_KEYS, "threshold schema mismatch")

    manifest = load_pose_input_manifest(train_input_path, validate_exact=True)
    commitment = load_pose_input_commitment(train_commitment_path)
    sidecar_sha256 = sha256_file(train_input_path)
    commitment_sha256 = sha256_file(train_commitment_path)
    validate_pose_input_binding(manifest, commitment, sidecar_sha256=sidecar_sha256)
    require(manifest.split == "train" and len(manifest.records) == 337, "expected train337")
    require(sidecar_sha256 == TRAIN337_SIDECAR_SHA256, "train337 sidecar mismatch")
    require(commitment_sha256 == TRAIN337_COMMITMENT_SHA256, "train337 commitment mismatch")
    require(pose_input_identity_sha256(manifest.records) == TRAIN337_IDENTITY_SHA256, "train337 identity mismatch")
    v4a = validate_v4a_inputs(
        manifest.records,
        cache_dir=v4a_cache_dir,
        ledger_path=v4a_ledger_path,
        paired_gate_path=v4a_paired_gate_path,
    )
    require(v4a.snapshot.fingerprint == V4A_CACHE_SET_SHA256, "v4a set mismatch")

    selection_sha256 = sha256_file(frozen_selection_path)
    require(selection_sha256 == settings.frozen_same39_selection_sha256, "same39 selection SHA mismatch")
    selection = json.loads(frozen_selection_path.read_text(encoding="utf-8"))
    require(isinstance(selection, Mapping), "same39 selection must be an object")
    require(
        selection.get("artifact_type")
        == "pams_pose_recovery_v4c_train337_long_tail_pilot_selection"
        and selection.get("label_free") is True
        and selection.get("record_total") == 39,
        "same39 selection schema mismatch",
    )
    raw_rows = selection.get("rows")
    require(isinstance(raw_rows, list) and len(raw_rows) == 39, "same39 rows mismatch")
    selected_hashes = sorted(
        str(row["video_id_sha256"])
        for row in raw_rows
        if isinstance(row, Mapping) and isinstance(row.get("video_id_sha256"), str)
    )
    require(len(selected_hashes) == 39 and len(set(selected_hashes)) == 39, "same39 hashes mismatch")
    records = tuple(
        record
        for record in manifest.records
        if hashlib.sha256(record.video_id.encode("utf-8")).hexdigest() in set(selected_hashes)
    )
    require(len(records) == 39, "same39 records missing")
    require(pose_input_identity_sha256(records) == settings.frozen_same39_identity_sha256, "same39 identity mismatch")
    v4a_ledger = json.loads(v4a_ledger_path.read_text(encoding="utf-8"))
    require(isinstance(v4a_ledger, Mapping), "v4a ledger must be an object")
    v4a_rows = v4a_ledger.get("caches")
    require(isinstance(v4a_rows, list) and len(v4a_rows) == 337, "v4a ledger rows mismatch")
    selected_ids = {record.video_id for record in records}
    zero_ids = {
        str(row["video_id"])
        for row in v4a_rows
        if isinstance(row, Mapping)
        and str(row.get("video_id")) in selected_ids
        and int((row.get("recovery_audit") or {}).get("final_valid_frames", -1)) == 0
    }
    zero_records = tuple(record for record in manifest.records if record.video_id in zero_ids)
    require(len(zero_records) == 11, "same39 zero-anchor stratum must contain 11 records")
    require(
        pose_input_identity_sha256(zero_records) == settings.frozen_zero11_identity_sha256,
        "same39 zero11 identity mismatch",
    )
    zero_hashes = sorted(
        hashlib.sha256(record.video_id.encode("utf-8")).hexdigest()
        for record in zero_records
    )

    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_raw_extraction_authorization_v1",
        "status": "authorized",
        "authorization_scope": "canonical_train337_same39_source_video_kprcnn_raw_evidence_only",
        "label_free": True,
        "source_revision": source_revision,
        "container_image_id": container_image_id,
        "raw_extraction_authorized": True,
        "cycleback_representation_input_authorized": False,
        "baseline_training_authorized": False,
        "scientific_training_authorized": False,
        "dev_evaluation_authorized": False,
        "sealed_test_authorized": False,
        "v4d_denied_cache_consumed": False,
        "selected_video_id_sha256": selected_hashes,
        "selected_zero11_video_id_sha256": zero_hashes,
        "frozen_thresholds": dict(thresholds),
        "bindings": {
            "source_revision": source_revision,
            "container_image_id": container_image_id,
            "config_file_sha256": config_file_sha256,
            "config_fingerprint": config.fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
            "model_asset_sha256": MODEL_ASSET_SHA256,
            "train337_sidecar_sha256": sidecar_sha256,
            "train337_commitment_sha256": commitment_sha256,
            "train337_identity_sha256": TRAIN337_IDENTITY_SHA256,
            "v4a_cache_set_sha256": V4A_CACHE_SET_SHA256,
            "frozen_same39_selection_sha256": selection_sha256,
            "same39_identity_sha256": settings.frozen_same39_identity_sha256,
            "zero11_identity_sha256": settings.frozen_zero11_identity_sha256,
            "zero11_record_total": 11,
            "synthetic_threshold_receipt_sha256": synthetic_sha256,
            "canonical_outcome_locator": str(outcome_locator),
            "canonical_registry_reservation_sha256": synthetic_bindings[
                "canonical_registry_reservation_sha256"
            ],
            "canonical_registry_artifact_type": synthetic_bindings[
                "canonical_registry_artifact_type"
            ],
        },
        "raw_outcome_must_not_be_used_as_training_authority": True,
    }
    write_json_exclusive(output_path, receipt)
    os.chmod(output_path, 0o444)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("source-revision", "container-image-id", "synthetic-threshold-receipt-sha256"):
        parser.add_argument(f"--{flag}", required=True)
    for flag in (
        "config", "train-input", "train-commitment", "frozen-selection",
        "synthetic-threshold-receipt", "v4a-cache-dir", "v4a-ledger",
        "v4a-paired-gate", "model-asset", "output",
    ):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = authorize_same39(
            source_revision=args.source_revision,
            container_image_id=args.container_image_id,
            config_path=args.config,
            train_input_path=args.train_input,
            train_commitment_path=args.train_commitment,
            frozen_selection_path=args.frozen_selection,
            synthetic_threshold_receipt_path=args.synthetic_threshold_receipt,
            expected_synthetic_threshold_receipt_sha256=args.synthetic_threshold_receipt_sha256,
            v4a_cache_dir=args.v4a_cache_dir,
            v4a_ledger_path=args.v4a_ledger,
            v4a_paired_gate_path=args.v4a_paired_gate,
            model_asset_path=args.model_asset,
            output_path=args.output,
        )
    except (OSError, TypeError, ValueError, Full337ContractError) as exc:
        print(f"v4e same39 authorization failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
