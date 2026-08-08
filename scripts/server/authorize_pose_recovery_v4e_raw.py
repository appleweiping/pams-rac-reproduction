#!/usr/bin/env python3
"""Authorize full337 v4e raw extraction only after the sealed same39 gate."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path

from pose_recovery_v4d_full337_contract import (
    MODEL_ASSET_SHA256,
    SAME39_IDENTITY_SHA256,
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


def authorize_raw_extraction(
    *,
    source_revision: str,
    container_image_id: str,
    config_path: Path,
    train_input_path: Path,
    train_commitment_path: Path,
    v4a_cache_dir: Path,
    v4a_ledger_path: Path,
    v4a_paired_gate_path: Path,
    model_asset_path: Path,
    same39_gate_path: Path,
    expected_same39_gate_sha256: str,
    output_path: Path,
) -> dict[str, object]:
    """Bind immutable inputs without granting any representation authority."""

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
    require(
        os.environ.get("PAMS_CONTAINER_SOURCE_REVISION") == source_revision,
        "container source revision mismatch",
    )
    require(
        os.environ.get("PAMS_CONTAINER_IMAGE_ID") == container_image_id,
        "container image ID environment mismatch",
    )
    config_file_sha256 = sha256_file(config_path)
    config = load_config(config_path.resolve(strict=True))
    require(
        config.pose.preprocessing_revision == V4E_PREPROCESSING_REVISION,
        "raw authorization config is not v4e",
    )
    settings = config.pose.keypoint_single_source
    require(settings is not None, "v4e single-source settings missing")
    require(not settings.raw_extraction_authorizes_training, "raw config grants training")
    require(settings.base_pose_fingerprint == V4A_POSE_FINGERPRINT, "config v4a pose mismatch")
    require(
        settings.base_pose_cache_set_sha256 == V4A_CACHE_SET_SHA256,
        "config v4a cache set mismatch",
    )
    require(settings.base_pose_ledger_sha256 == V4A_LEDGER_SHA256, "config v4a ledger mismatch")
    require(
        settings.base_paired_gate_sha256 == V4A_PAIRED_GATE_SHA256,
        "config v4a paired gate mismatch",
    )
    require(
        settings.frozen_same39_identity_sha256 == SAME39_IDENTITY_SHA256,
        "config same39 identity mismatch",
    )
    require(sha256_file(model_asset_path) == MODEL_ASSET_SHA256, "model asset SHA mismatch")

    require(
        len(expected_same39_gate_sha256) == 64
        and all(
            character in "0123456789abcdef"
            for character in expected_same39_gate_sha256
        ),
        "same39 gate SHA must be lowercase 64-hex",
    )
    require(
        sha256_file(same39_gate_path) == expected_same39_gate_sha256,
        "same39 gate SHA mismatch",
    )
    preregistered_gate = json.loads(same39_gate_path.read_text(encoding="utf-8"))
    require(isinstance(preregistered_gate, dict), "same39 gate must be an object")
    require(
        preregistered_gate.get("artifact_type")
        == "pams_pose_recovery_v4e_same39_gate_v1",
        "same39 gate type mismatch",
    )
    require(
        preregistered_gate.get("status") == "passed"
        and preregistered_gate.get("overall_pass") is True
        and preregistered_gate.get("label_free") is True,
        "same39 gate did not pass",
    )
    require(
        preregistered_gate.get("threshold_origin")
        == "synthetic-frozen-generator-seed-split-preset-grid-v1",
        "thresholds were not selected exclusively from the frozen synthetic grid",
    )
    require(
        preregistered_gate.get("synthetic_threshold_receipt_verified") is True
        and preregistered_gate.get("same39_validation_passed") is True,
        "synthetic or same39 prereg stage failed",
    )
    require(
        preregistered_gate.get("same39_threshold_selection_used") is False,
        "same39 must validate frozen thresholds, not select them",
    )
    require(
        preregistered_gate.get("full337_raw_extraction_authorized") is True,
        "preregistered gate does not authorize full337 raw extraction",
    )
    prereg_bindings = preregistered_gate.get("bindings")
    require(isinstance(prereg_bindings, dict), "same39 gate bindings missing")
    outcome_locator = Path(str(prereg_bindings["canonical_outcome_locator"]))
    require(
        same39_gate_path.resolve(strict=True)
        == outcome_locator / "gate-output/v4e-same39.gate.json"
        and stat.S_IMODE(same39_gate_path.stat().st_mode) & 0o222 == 0,
        "same39 gate canonical locator/sealing mismatch",
    )
    require(
        output_path.parent.resolve(strict=True) / output_path.name
        == outcome_locator / "authorization/v4e-full337-raw.authorization.json",
        "full337 authorization output locator mismatch",
    )
    require(prereg_bindings.get("source_revision") == source_revision, "prereg source mismatch")
    require(
        prereg_bindings.get("container_image_id") == container_image_id,
        "prereg image mismatch",
    )
    require(
        prereg_bindings.get("config_file_sha256") == config_file_sha256,
        "prereg config mismatch",
    )
    require(
        prereg_bindings.get("model_asset_sha256") == MODEL_ASSET_SHA256,
        "prereg model mismatch",
    )
    require(
        prereg_bindings.get("same39_identity_sha256") == SAME39_IDENTITY_SHA256,
        "prereg same39 identity mismatch",
    )
    require(
        prereg_bindings.get("zero11_identity_sha256")
        == settings.frozen_zero11_identity_sha256
        and prereg_bindings.get("zero11_record_total") == 11,
        "prereg zero11 identity mismatch",
    )
    for required_binding in (
        "same39_authorization_sha256",
        "same39_raw_ledger_sha256",
        "same39_pose_snapshot_sha256",
        "same39_pose_cache_set_sha256",
        "same39_segment_index_sha256",
        "same39_segment_reset_policy_sha256",
        "same39_joint_mask_snapshot_sha256",
        "same39_joint_mask_set_sha256",
        "same39_cycleback_pair_eligibility_sha256",
        "same39_identity_map_sha256",
        "same39_independent_replay_receipt_sha256",
        "same39_prefix_suffix_overlap_receipt_sha256",
        "synthetic_threshold_receipt_sha256",
    ):
        value = prereg_bindings.get(required_binding)
        require(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value),
            f"same39 gate binding invalid: {required_binding}",
        )
    thresholds = preregistered_gate.get("frozen_thresholds")
    require(isinstance(thresholds, dict), "frozen threshold tuple missing")
    require(
        set(thresholds)
        == {
            "maximum_frame_center_step",
            "maximum_frame_log_scale_step",
            "maximum_frame_morphology_step",
            "maximum_frame_joint_mask_flicker_fraction",
            "minimum_dual_path_agreement",
            "minimum_frame_local_ambiguity_gap",
            "minimum_longest_trainable_segment_fraction",
            "minimum_longest_trainable_segment_frames",
            "minimum_source_coverage",
            "maximum_candidate_window_frames",
            "minimum_window_joint_support_fraction",
            "minimum_window_action_motion",
            "minimum_window_stable_action_joints",
        },
        "frozen threshold schema mismatch",
    )

    manifest = load_pose_input_manifest(train_input_path, validate_exact=True)
    commitment = load_pose_input_commitment(train_commitment_path)
    sidecar_sha256 = sha256_file(train_input_path)
    commitment_sha256 = sha256_file(train_commitment_path)
    validate_pose_input_binding(manifest, commitment, sidecar_sha256=sidecar_sha256)
    require(manifest.split == "train" and len(manifest.records) == 337, "expected train337")
    require(sidecar_sha256 == TRAIN337_SIDECAR_SHA256, "train337 sidecar SHA mismatch")
    require(
        commitment_sha256 == TRAIN337_COMMITMENT_SHA256,
        "train337 commitment SHA mismatch",
    )
    identity_sha256 = pose_input_identity_sha256(manifest.records)
    require(identity_sha256 == TRAIN337_IDENTITY_SHA256, "train337 identity mismatch")
    v4a = validate_v4a_inputs(
        manifest.records,
        cache_dir=v4a_cache_dir,
        ledger_path=v4a_ledger_path,
        paired_gate_path=v4a_paired_gate_path,
    )
    require(v4a.snapshot.fingerprint == V4A_CACHE_SET_SHA256, "v4a cache set mismatch")

    receipt: dict[str, object] = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_raw_extraction_authorization_v1",
        "status": "authorized",
        "authorization_scope": "canonical_train337_source_video_kprcnn_raw_evidence_only",
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
        "bindings": {
            "config_file_sha256": config_file_sha256,
            "config_fingerprint": config.fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
            "model_asset_sha256": MODEL_ASSET_SHA256,
            "train337_sidecar_sha256": sidecar_sha256,
            "train337_commitment_sha256": commitment_sha256,
            "train337_identity_sha256": identity_sha256,
            "v4a_pose_fingerprint": V4A_POSE_FINGERPRINT,
            "v4a_ledger_sha256": V4A_LEDGER_SHA256,
            "v4a_paired_gate_sha256": V4A_PAIRED_GATE_SHA256,
            "v4a_cache_set_sha256": v4a.snapshot.fingerprint,
            "same39_gate_sha256": expected_same39_gate_sha256,
            "synthetic_threshold_receipt_sha256": prereg_bindings[
                "synthetic_threshold_receipt_sha256"
            ],
            "zero11_identity_sha256": prereg_bindings["zero11_identity_sha256"],
            "zero11_record_total": 11,
            "canonical_outcome_locator": str(outcome_locator),
            "canonical_registry_reservation_sha256": prereg_bindings[
                "canonical_registry_reservation_sha256"
            ],
            "canonical_registry_artifact_type": prereg_bindings[
                "canonical_registry_artifact_type"
            ],
            **{
                key: prereg_bindings[key]
                for key in (
                    "same39_authorization_sha256",
                    "same39_raw_ledger_sha256",
                    "same39_pose_snapshot_sha256",
                    "same39_pose_cache_set_sha256",
                    "same39_segment_index_sha256",
                    "same39_segment_reset_policy_sha256",
                    "same39_joint_mask_snapshot_sha256",
                    "same39_joint_mask_set_sha256",
                    "same39_cycleback_pair_eligibility_sha256",
                    "same39_identity_map_sha256",
                    "same39_independent_replay_receipt_sha256",
                    "same39_prefix_suffix_overlap_receipt_sha256",
                )
            },
        },
        "frozen_thresholds": thresholds,
        "full_run_requires_separate_preregistered_pilot_and_synthetic_gate": True,
        "synthetic_then_same39_gate_verified": True,
        "same39_bytewise_reuse_required": True,
        "fresh_extraction_record_total": 298,
        "raw_outcome_must_not_be_used_as_training_authority": True,
    }
    write_json_exclusive(output_path, receipt)
    os.chmod(output_path, 0o444)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--container-image-id", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--train-commitment", type=Path, required=True)
    parser.add_argument("--v4a-cache-dir", type=Path, required=True)
    parser.add_argument("--v4a-ledger", type=Path, required=True)
    parser.add_argument("--v4a-paired-gate", type=Path, required=True)
    parser.add_argument("--model-asset", type=Path, required=True)
    parser.add_argument("--same39-gate", type=Path, required=True)
    parser.add_argument("--same39-gate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = authorize_raw_extraction(
            source_revision=args.source_revision,
            container_image_id=args.container_image_id,
            config_path=args.config,
            train_input_path=args.train_input,
            train_commitment_path=args.train_commitment,
            v4a_cache_dir=args.v4a_cache_dir,
            v4a_ledger_path=args.v4a_ledger,
            v4a_paired_gate_path=args.v4a_paired_gate,
            model_asset_path=args.model_asset,
            same39_gate_path=args.same39_gate,
            expected_same39_gate_sha256=args.same39_gate_sha256,
            output_path=args.output,
        )
    except (OSError, TypeError, ValueError, Full337ContractError) as exc:
        print(f"v4e raw extraction authorization failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
