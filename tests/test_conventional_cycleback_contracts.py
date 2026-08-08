from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from pams.conventional_cycleback.config import (
    ConventionalCycleBackConfig,
    load_conventional_cycleback_config,
    parse_conventional_cycleback_config,
)
from pams.conventional_cycleback.runtime import (
    build_encoder,
    load_identity_map,
    load_pair_eligibility,
    load_segment_index,
    load_stable_json_object,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = (
    ROOT / "configs/conventional_cycleback/w16_hop4_v1.yaml",
    ROOT / "configs/conventional_cycleback/w16_hop2_v1.yaml",
    ROOT / "configs/conventional_cycleback/w24_hop4_v1.yaml",
)


def _configs() -> tuple[ConventionalCycleBackConfig, ...]:
    return tuple(load_conventional_cycleback_config(path) for path in CONFIG_PATHS)


def _write_json(path: Path, payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _representation_contracts(
    tmp_path: Path,
) -> tuple[dict[str, str], object, Path, str, str]:
    video_ids = ("video-a", "video-b")
    opaque = {
        identifier: hashlib.sha256(identifier.encode()).hexdigest()
        for identifier in video_ids
    }
    identity_payload = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_identity_map_v1",
        "entry_count": 2,
        "entries": [
            {"video_id": identifier, "video_id_sha256": opaque[identifier]}
            for identifier in video_ids
        ],
        "mapping_rule": "utf8-video-id-sha256",
    }
    identity_path = tmp_path / "identity-map.json"
    identity_sha = _write_json(identity_path, identity_payload)
    identity_map, _ = load_identity_map(
        identity_path,
        expected_video_ids=video_ids,
        expected_sha256=identity_sha,
    )
    variants_a = {
        "W16_H2": [0, 2, 4],
        "W16_H4": [0, 4],
        "W16_H4_PE0": [0, 4],
        "W24_H4": [0],
    }
    variants_b = {key: [] for key in variants_a}
    pair_payload = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_cycleback_pair_eligibility_v1",
        "policy": "representation_authorized_exact_2w_starts_only",
        "start_grid_policy": "native_zero_based_start_mod_hop_equals_zero",
        "variant_geometry": {
            "W16_H2": {"window_frames": 16, "hop_frames": 2},
            "W16_H4": {"window_frames": 16, "hop_frames": 4},
            "W16_H4_PE0": {"window_frames": 16, "hop_frames": 4},
            "W24_H4": {"window_frames": 24, "hop_frames": 4},
        },
        "frozen_variant_aliases": {"W16_H4_PE0": "W16_H4"},
        "alias_semantics": (
            "listed-starts-must-be-bytewise-equal-no-consumer-inference"
        ),
        "entry_count": 2,
        "frozen_thresholds": {
            "maximum_candidate_window_frames": 24,
            "maximum_frame_center_step": 0.20,
            "maximum_frame_log_scale_step": 0.20,
            "maximum_frame_joint_mask_flicker_fraction": 0.30,
            "minimum_dual_path_agreement": 0.80,
            "minimum_frame_local_ambiguity_gap": 0.10,
            "minimum_longest_trainable_segment_fraction": 0.20,
            "minimum_longest_trainable_segment_frames": 48,
            "minimum_source_coverage": 0.50,
            "minimum_window_stable_action_joints": 6,
            "minimum_window_joint_support_fraction": 0.75,
        },
        "entries": [
            {
                "video_id": opaque["video-a"],
                "native_length": 64,
                "representation_eligible": True,
                "base_valid_starts_by_variant": variants_a,
                "starts_by_variant": variants_a,
            },
            {
                "video_id": opaque["video-b"],
                "native_length": 64,
                "representation_eligible": False,
                "base_valid_starts_by_variant": variants_b,
                "starts_by_variant": variants_b,
            },
        ],
        "consumer_may_expand_starts": False,
        "label_free": True,
    }
    pair_path = tmp_path / "cycleback-pair-eligibility.json"
    pair_sha = _write_json(pair_path, pair_payload)
    pair_eligibility = load_pair_eligibility(
        pair_path,
        expected_sha256=pair_sha,
        identity_map=identity_map,
    )
    return identity_map, pair_eligibility, pair_path, pair_sha, "d" * 64


def test_candidate_matrix_changes_only_window_geometry() -> None:
    configs = _configs()

    assert [
        (item.candidate_id, item.window_pair.length_frames, item.window_pair.hop_frames)
        for item in configs
    ] == [("W16_H4", 16, 4), ("W16_H2", 16, 2), ("W24_H4", 24, 4)]
    assert len({item.fingerprint for item in configs}) == 3
    assert len({item.candidate_invariant_fingerprint for item in configs}) == 1
    assert all(item.paper_table_claim_eligible is False for item in configs)
    assert all(
        item.generation_failure_policy
        == "all_three_sinusoidal_candidates_rejected_terminates_generation_pe_off_diagnostic_only"
        for item in configs
    )


def test_fingerprint_binds_thresholds_and_rejects_nonfinite_values() -> None:
    config = _configs()[0]
    payload = config.model_dump(mode="python")
    changed = copy.deepcopy(payload)
    changed["geometry_audit"]["thresholds"]["minimum_eligible_pair_fraction"] = 0.51
    changed_config = ConventionalCycleBackConfig.model_validate(changed)

    assert changed_config.fingerprint != config.fingerprint
    nonfinite = copy.deepcopy(payload)
    nonfinite["objective"]["variance_floor"] = float("nan")
    with pytest.raises(ValueError):
        ConventionalCycleBackConfig.model_validate(nonfinite)


def test_duplicate_yaml_and_json_fields_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="duplicate YAML field"):
        parse_conventional_cycleback_config("schema_version: 1\nschema_version: 1\n")

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON field"):
        load_stable_json_object(duplicate, role="test duplicate")


def test_v4e_pair_contract_binds_explicit_pe0_alias_and_no_expansion(
    tmp_path: Path,
) -> None:
    identity_map, pair_eligibility, _, _, _ = _representation_contracts(tmp_path)

    assert set(identity_map) == {"video-a", "video-b"}
    assert pair_eligibility.start_grid_policy == (
        "native_zero_based_start_mod_hop_equals_zero"
    )
    assert pair_eligibility.starts_by_variant["W16_H4_PE0"]["video-a"] == (
        pair_eligibility.starts_by_variant["W16_H4"]["video-a"]
    )
    assert pair_eligibility.starts_by_variant["W16_H4"]["video-b"] == ()


def test_v4e_pair_contract_rejects_alias_drift(tmp_path: Path) -> None:
    identity_map, _, pair_path, _, _ = _representation_contracts(tmp_path)
    payload = json.loads(pair_path.read_text(encoding="utf-8"))
    payload["entries"][0]["starts_by_variant"]["W16_H4_PE0"] = [0]
    pair_sha = _write_json(pair_path, payload)

    with pytest.raises(ValueError, match="PE0 alias"):
        load_pair_eligibility(
            pair_path,
            expected_sha256=pair_sha,
            identity_map=identity_map,
        )


def test_v4e_segment_contract_maps_opaque_ids_and_quarantines_ineligible(
    tmp_path: Path,
) -> None:
    identity_map, pair_eligibility, _, _, policy_sha = _representation_contracts(
        tmp_path
    )
    segment_payload = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_segment_index_v1",
        "segment_reset_policy_sha256": policy_sha,
        "entry_count": 2,
        "entries": [
            {
                "video_id": identity_map["video-a"],
                "native_length": 64,
                "association_resets": [],
                "eligible_frame_ranges": [
                    {"start": 0, "stop": 64, "frames": 64}
                ],
                "eligible_pair_starts_by_candidate": {
                    variant: list(values["video-a"])
                    for variant, values in pair_eligibility.starts_by_variant.items()
                },
                "representation_eligible": True,
                "ineligibility_reason": None,
                "no_unreported_internal_reset": True,
            },
            {
                "video_id": identity_map["video-b"],
                "native_length": 64,
                "association_resets": [],
                "eligible_frame_ranges": [],
                "eligible_pair_starts_by_candidate": {
                    variant: [] for variant in pair_eligibility.starts_by_variant
                },
                "representation_eligible": False,
                "ineligibility_reason": "quarantined",
                "no_unreported_internal_reset": True,
            },
        ],
        "all_resets_explicit_and_invalid_or_segment_sidecar_complete": True,
    }
    segment_path = tmp_path / "segment-index.json"
    segment_sha = _write_json(segment_path, segment_payload)

    segments, _ = load_segment_index(
        segment_path,
        expected_video_ids=("video-a", "video-b"),
        expected_sha256=segment_sha,
        expected_segment_reset_policy_sha256=policy_sha,
        identity_map=identity_map,
        pair_eligibility=pair_eligibility,
    )

    assert segments == {"video-a": ((0, 64),), "video-b": ()}


def test_pe_off_encoder_is_position_permutation_invariant() -> None:
    base = _configs()[0]
    payload = base.model_dump(mode="python")
    payload["encoder"].update(
        {
            "model_dim": 8,
            "embedding_dim": 8,
            "layers": 1,
            "heads": 2,
            "feedforward_dim": 16,
            "dropout": 0.0,
            "maximum_native_length": 64,
        }
    )
    config = ConventionalCycleBackConfig.model_validate(payload)
    torch.manual_seed(19)
    canonical = build_encoder(config).eval()
    pe_off = build_encoder(config, position_encoding_mode="none").eval()
    pe_off.load_state_dict(canonical.state_dict(), strict=True)
    poses = torch.zeros((1, 8, 33, 3))
    valid = torch.ones((1, 8), dtype=torch.bool)
    positions = torch.arange(8).unsqueeze(0)
    permuted = positions.flip(1)

    with torch.inference_mode():
        canonical_rows = canonical(poses, valid, position_indices=positions)
        permuted_rows = canonical(poses, valid, position_indices=permuted)
        pe_off_rows = pe_off(poses, valid, position_indices=positions)
        pe_off_permuted_rows = pe_off(poses, valid, position_indices=permuted)

    assert not torch.equal(canonical_rows, permuted_rows)
    assert torch.equal(pe_off_rows, pe_off_permuted_rows)


def test_unified_2d_runtime_never_calls_legacy_minmax_preprocessing() -> None:
    runtime_source = (
        ROOT / "src/pams/conventional_cycleback/runtime.py"
    ).read_text(encoding="utf-8")

    assert "preprocess_native_pose_sequence" not in runtime_source
    assert "preprocess_pose_sequence" not in runtime_source
