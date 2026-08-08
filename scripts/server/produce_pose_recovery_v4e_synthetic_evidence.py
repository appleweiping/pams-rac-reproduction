#!/usr/bin/env python3
"""Produce deterministic pre-data synthetic v4e threshold evidence.

This producer consumes only frozen generator/grid/seed artifacts.  It never
opens UCFRep train/dev/test videos, pose caches, annotations, or count labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pose_recovery_v4d_full337_contract import (
    Full337ContractError,
    require,
    sha256_file,
    write_json_exclusive,
)

from pams.config import KeypointRCNNSingleSourceConfig, load_config
from pams.keypoint_single_source import (
    AssociationWeights,
    RELIABLE_ACTION_JOINTS,
    TrackStabilityThresholds,
    ViterbiPath,
    _local_ambiguity_gap_rows,
    assess_track_stability,
    build_single_source_sequence,
    effective_maximum_bridge_gap_frames,
    reconstruct_canonical_frame_evidence,
    select_segmented_top2_viterbi_paths,
)

POSITIVE_FAMILIES = (
    "clean_known_identity",
    "short_occlusion_random30pct_joints_for20pct_time",
    "inplane_affine_rotation15_scale15_translation10pct",
)
IDENTITY_NULL_FAMILIES = (
    "crossing_with_identity_swap",
    "explicit_candidate_identity_swap",
    "long_gap_with_identity_change",
)
JOINT_NULL_FAMILIES = (
    "periodic_joint_mask_flicker",
    "torso_only_without_action_joints",
    "alternating_limb_dropout_without_stable_window_support",
)
DIAGNOSTIC_FAMILIES = (
    "fast_motion_with_large_static_bystander_period_must_be_unsupported",
)
SAMPLES_PER_FAMILY = 512


def _object(path: Path, role: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, Mapping), f"{role} must be a JSON object")
    return value


def _seed_manifest(path: Path, *, split: str) -> tuple[int, ...]:
    value = _object(path, f"{split} seed manifest")
    require(
        value.get("artifact_type")
        == "pams_pose_recovery_v4e_synthetic_seed_manifest_v1"
        and value.get("split") == split,
        f"{split} seed manifest schema mismatch",
    )
    seeds = value.get("seeds")
    require(
        isinstance(seeds, list)
        and len(seeds) == SAMPLES_PER_FAMILY
        and len(set(seeds)) == SAMPLES_PER_FAMILY
        and all(isinstance(seed, int) and 0 <= seed < 2**63 for seed in seeds),
        f"{split} seed manifest must contain 512 unique uint63 values",
    )
    return tuple(int(seed) for seed in seeds)


def _bitset_hex(values: Sequence[bool]) -> str:
    require(len(values) == SAMPLES_PER_FAMILY, "synthetic decision vector length mismatch")
    packed = np.packbits(np.asarray(values, dtype=np.uint8), bitorder="little")
    return packed.tobytes().hex()


_BASE_SHAPE = np.asarray(
    [
        [0.00, -0.36], [-0.04, -0.39], [0.04, -0.39], [-0.08, -0.37], [0.08, -0.37],
        [-0.15, -0.22], [0.15, -0.22], [-0.23, -0.05], [0.23, -0.05],
        [-0.28, 0.14], [0.28, 0.14], [-0.10, 0.00], [0.10, 0.00],
        [-0.11, 0.22], [0.11, 0.22], [-0.12, 0.42], [0.12, 0.42],
    ],
    dtype=np.float32,
)


def _candidate(
    *, center: tuple[float, float], scale: float, phase: float, mask: np.ndarray,
    box_score: float, affine: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    shape = np.array(_BASE_SHAPE, copy=True)
    motion = np.float32(0.06 * np.sin(phase))
    shape[[9, 10], 1] += np.asarray([motion, -motion], dtype=np.float32)
    shape[[15, 16], 0] += np.asarray([-motion, motion], dtype=np.float32)
    xy = np.asarray(center, dtype=np.float32) + np.float32(scale) * shape
    if affine is not None:
        angle_degrees, affine_scale, shift_x, shift_y = affine
        angle = np.deg2rad(angle_degrees)
        rotation = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            dtype=np.float32,
        )
        xy = (xy - 0.5) @ rotation * np.float32(affine_scale) + 0.5
        xy += np.asarray([shift_x, shift_y], dtype=np.float32)
    xy = np.clip(xy, 0.0, 1.0).astype(np.float32)
    logits = np.where(mask, np.float32(3.0), np.float32(0.0))
    visibility = np.zeros(17, dtype=np.float32)
    visibility[mask] = np.float32(1.0 / (1.0 + np.exp(-3.0)) * box_score)
    value = np.zeros((17, 6), dtype=np.float32)
    value[:, :2] = xy
    value[:, 3] = visibility
    value[:, 4] = logits
    value[:, 5] = np.float32(box_score)
    return value


def _synthetic_candidates(
    family: str, seed: int,
) -> tuple[list[np.ndarray | None], list[tuple[str, ...]], str]:
    rng = np.random.default_rng(seed)
    frames = 256 if family == "short_occlusion_random30pct_joints_for20pct_time" else 96
    phase_offset = float(rng.uniform(-np.pi, np.pi))
    target_center = (
        float(0.50 + rng.uniform(-0.025, 0.025)),
        float(0.50 + rng.uniform(-0.015, 0.015)),
    )
    target_scale = float(rng.uniform(0.68, 0.75))
    distractor_scale = float(rng.uniform(0.31, 0.37))
    full_mask = np.ones(17, dtype=np.bool_)
    candidates: list[np.ndarray | None] = []
    truth_rows: list[tuple[str, ...]] = []
    affine = None
    if family == "inplane_affine_rotation15_scale15_translation10pct":
        affine = (
            float(rng.uniform(-15.0, 15.0)),
            float(rng.uniform(0.85, 1.15)),
            float(rng.uniform(-0.10, 0.10)),
            float(rng.uniform(-0.10, 0.10)),
        )
    occluded = np.asarray([], dtype=np.int64)
    occlusion_start = 0
    if family == "short_occlusion_random30pct_joints_for20pct_time":
        selectable = np.asarray([i for i in range(17) if i not in (5, 6, 11, 12)])
        while True:
            occluded = np.asarray(rng.choice(selectable, size=5, replace=False), dtype=np.int64)
            if len(set(int(value) for value in occluded) & set(RELIABLE_ACTION_JOINTS)) <= 4:
                break
        occlusion_start = frames - int(round(0.20 * frames))
    gap_start = 36 + int(seed % 9)
    gap_stop = gap_start + 16
    crossing_edge = float(rng.uniform(0.25, 0.30))
    explicit_separation = float(rng.uniform(0.14, 0.18))

    for frame in range(frames):
        phase = 2.0 * np.pi * frame / 16.0 + phase_offset
        mask_a = np.array(full_mask, copy=True)
        mask_b = np.array(full_mask, copy=True)
        if family == "short_occlusion_random30pct_joints_for20pct_time" and frame >= occlusion_start:
            mask_a[occluded] = False
        if family == "periodic_joint_mask_flicker":
            mask_a[:] = False
            mask_a[[5, 6, 11, 12]] = True
            mask_a[
                [0, 1, 7, 8, 9, 10]
                if frame % 2 == 0
                else [2, 3, 4, 13, 14, 15, 16]
            ] = True
        if family == "torso_only_without_action_joints":
            mask_a[:] = False
            mask_a[[5, 6, 11, 12]] = True
        if family == "alternating_limb_dropout_without_stable_window_support":
            mask_a[list(RELIABLE_ACTION_JOINTS)] = False
            mask_a[[7, 8, 9, 10] if frame % 2 == 0 else [13, 14, 15, 16]] = True

        if family == "long_gap_with_identity_change" and gap_start <= frame < gap_stop:
            candidates.append(None)
            truth_rows.append(())
            continue
        if family in {"crossing_with_identity_swap", "explicit_candidate_identity_swap"}:
            if family == "crossing_with_identity_swap":
                progress = frame / (frames - 1)
                centers = (
                    (
                        crossing_edge + (1.0 - 2.0 * crossing_edge) * progress,
                        target_center[1],
                    ),
                    (
                        1.0 - crossing_edge - (1.0 - 2.0 * crossing_edge) * progress,
                        target_center[1],
                    ),
                )
            else:
                centers = (
                    (target_center[0] - explicit_separation, target_center[1]),
                    (target_center[0] + explicit_separation, target_center[1]),
                )
            frame_candidates = np.stack(
                (
                    _candidate(center=centers[0], scale=0.65, phase=0.0, mask=mask_a, box_score=0.9),
                    _candidate(center=centers[1], scale=0.65, phase=0.0, mask=mask_b, box_score=0.9),
                ),
                axis=0,
            )
            swap_frame = (
                2 * frames // 3 + int(seed % 5) - 2
                if family == "crossing_with_identity_swap"
                else frames // 2 + int(seed % 7) - 3
            )
            swapped = frame >= swap_frame
            truth = ("B", "A") if swapped else ("A", "B")
        elif family == "long_gap_with_identity_change":
            frame_candidates = np.stack(
                (_candidate(center=target_center, scale=target_scale, phase=phase, mask=mask_a, box_score=0.9),),
                axis=0,
            )
            truth = ("A" if frame < gap_start else "B",)
        elif family == "fast_motion_with_large_static_bystander_period_must_be_unsupported":
            frame_candidates = np.stack(
                (
                    _candidate(center=(0.30 + 0.003 * frame, 0.50), scale=0.48, phase=phase, mask=mask_a, box_score=0.9),
                    _candidate(center=(0.68, 0.50), scale=0.78, phase=0.0, mask=mask_b, box_score=0.9),
                ),
                axis=0,
            )
            truth = ("A", "B")
        else:
            frame_candidates = np.stack(
                (
                    _candidate(center=target_center, scale=target_scale, phase=phase, mask=mask_a, box_score=0.9, affine=affine),
                    _candidate(center=(0.78, target_center[1]), scale=distractor_scale, phase=0.0, mask=mask_b, box_score=0.95, affine=affine),
                ),
                axis=0,
            )
            truth = ("A", "B")
        candidates.append(np.ascontiguousarray(frame_candidates, dtype=np.float32))
        truth_rows.append(truth)
    return candidates, truth_rows, "A"


def _period_supported(xyz: np.ndarray, valid_mask: np.ndarray) -> bool:
    valid = np.asarray(xyz[valid_mask, :17, :2], dtype=np.float64)
    if len(valid) < 24:
        return False
    signal = valid.reshape(len(valid), -1)
    signal -= np.mean(signal, axis=0, keepdims=True)
    energy = float(np.sum(np.square(signal)))
    if not np.isfinite(energy) or energy <= 1.0e-8:
        return False
    for lag in range(4, min(32, len(signal) // 3) + 1):
        numerator = float(np.sum(signal[:-lag] * signal[lag:]))
        denominator = float(
            np.sqrt(np.sum(np.square(signal[:-lag])) * np.sum(np.square(signal[lag:])))
        )
        if denominator > 1.0e-8 and numerator / denominator >= 0.60:
            return True
    return False


def _run_mechanics(
    *, family: str, seed: int, candidates: Sequence[np.ndarray | None],
    truth_rows: Sequence[tuple[str, ...]], settings: KeypointRCNNSingleSourceConfig,
) -> dict[str, Any]:
    no_anchors = tuple(None for _ in candidates)
    primary_weights = AssociationWeights(
        settings.primary_center_weight, settings.primary_log_scale_weight,
        settings.primary_shape_weight, settings.primary_anchor_weight,
    )
    secondary_weights = AssociationWeights(
        settings.secondary_center_weight, settings.secondary_log_scale_weight,
        settings.secondary_shape_weight, settings.secondary_anchor_weight,
    )
    maximum_gap = effective_maximum_bridge_gap_frames(
        fps=25.0,
        maximum_seconds=settings.maximum_bridge_gap_seconds,
        frame_cap=settings.maximum_bridge_gap_frame_cap,
    )
    primary, segments = select_segmented_top2_viterbi_paths(
        candidates, anchor_residuals=no_anchors, weights=primary_weights,
        maximum_bridge_gap_frames=maximum_gap,
    )
    secondary, secondary_segments = select_segmented_top2_viterbi_paths(
        candidates, anchor_residuals=no_anchors, weights=secondary_weights,
        maximum_bridge_gap_frames=maximum_gap,
    )
    require(segments == secondary_segments, "synthetic dual path segment mismatch")
    ambiguity: dict[int, float] = {}
    for segment in segments:
        start, stop = segment[0], segment[-1] + 1
        subpath = ViterbiPath(
            primary.selected_indices[start:stop], None, None, len(segment), "synthetic"
        )
        ambiguity.update(
            {
                start + index: gap
                for index, gap in _local_ambiguity_gap_rows(
                    candidates[start:stop], anchor_residuals=no_anchors[start:stop],
                    weights=primary_weights, selected_path=subpath,
                )
            }
        )
    video_id = f"synthetic:{family}:{seed}"
    sequence = build_single_source_sequence(
        video_id=video_id, fps=25.0, source_frames=len(candidates),
        decoded_frames=len(candidates), candidates=candidates, primary_path=primary,
    )
    diagnostics = [
        None
        if frame is None
        else {
            "detector_output_total": len(frame), "rejected_non_person": 0,
            "rejected_low_box_score": 0, "rejected_low_total_joint_support": 0,
            "rejected_low_action_joint_support": 0, "rejected_unreliable_torso": 0,
            "eligible_before_top4": len(frame), "dropped_by_top4": 0,
        }
        for frame in candidates
    ]
    frame_evidence = reconstruct_canonical_frame_evidence(
        candidates=candidates, primary=primary, secondary=secondary,
        association_segments=segments, sequence=sequence,
        ambiguity_by_frame=ambiguity, maximum_bridge_gap_frames=maximum_gap,
        detector_diagnostics=diagnostics,
        keypoint_logit_threshold=settings.keypoint_logit_threshold,
    )
    comparable = [
        index for index, frame in enumerate(candidates)
        if frame is not None and len(frame) > 1
        and primary.selected_indices[index] is not None
        and secondary.selected_indices[index] is not None
    ]
    agreement = (
        sum(primary.selected_indices[index] == secondary.selected_indices[index] for index in comparable)
        / len(comparable)
        if comparable else (1.0 if primary.observed_frames else 0.0)
    )
    digest = hashlib.sha256(video_id.encode("utf-8")).hexdigest()
    selected_truth = tuple(
        None if selected is None else truth_rows[index][int(selected)]
        for index, selected in enumerate(primary.selected_indices)
    )
    audit = {
        "video_id_sha256": digest,
        "video_evidence_sha256": hashlib.sha256(
            json.dumps(frame_evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "source_frames": len(candidates),
        "dual_path_agreement": float(agreement),
        "frame_evidence": frame_evidence,
    }
    return {
        "audit": audit,
        "sequence_xyz": sequence.xyz,
        "sequence_valid_mask": sequence.valid_mask,
        "selected_truth": selected_truth,
        "period_supported": _period_supported(sequence.xyz, sequence.valid_mask),
    }


def _fixture(
    family: str, seed: int, settings: KeypointRCNNSingleSourceConfig,
) -> tuple[dict[str, Any], bool, bool]:
    candidates, truths, target = _synthetic_candidates(family, seed)
    first = _run_mechanics(
        family=family, seed=seed, candidates=candidates, truth_rows=truths, settings=settings
    )
    repeat = _run_mechanics(
        family=family, seed=seed, candidates=candidates, truth_rows=truths, settings=settings
    )
    determinism_failure = not (
        first["audit"] == repeat["audit"]
        and np.array_equal(first["sequence_xyz"], repeat["sequence_xyz"])
        and np.array_equal(first["sequence_valid_mask"], repeat["sequence_valid_mask"])
        and first["selected_truth"] == repeat["selected_truth"]
    )
    rng = np.random.default_rng(seed ^ 0x5A17)
    permuted_candidates: list[np.ndarray | None] = []
    permuted_truths: list[tuple[str, ...]] = []
    for frame, truth in zip(candidates, truths, strict=True):
        if frame is None:
            permuted_candidates.append(None)
            permuted_truths.append(())
            continue
        order = rng.permutation(len(frame))
        permuted_candidates.append(np.ascontiguousarray(frame[order], dtype=np.float32))
        permuted_truths.append(tuple(truth[int(index)] for index in order))
    permuted = _run_mechanics(
        family=family, seed=seed, candidates=permuted_candidates,
        truth_rows=permuted_truths, settings=settings,
    )
    permutation_failure = not (
        np.array_equal(first["sequence_xyz"], permuted["sequence_xyz"])
        and np.array_equal(first["sequence_valid_mask"], permuted["sequence_valid_mask"])
        and first["selected_truth"] == permuted["selected_truth"]
    )
    first["permuted_audit"] = permuted["audit"]
    selected = {value for value in first["selected_truth"] if value is not None}
    if family in POSITIVE_FAMILIES:
        first["truth_condition"] = selected == {target}
    elif family in IDENTITY_NULL_FAMILIES:
        first["truth_condition"] = len(selected) > 1
    else:
        first["truth_condition"] = True
    return first, determinism_failure, permutation_failure


def produce_synthetic_evidence(
    *,
    reservation_path: Path,
    expected_reservation_sha256: str,
    config_path: Path,
    model_asset_path: Path,
    generator_source_path: Path,
    threshold_grid_path: Path,
    calibration_seed_manifest_path: Path,
    heldout_seed_manifest_path: Path,
    registry_reservation_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    require(sha256_file(reservation_path) == expected_reservation_sha256, "reservation SHA mismatch")
    reservation = _object(reservation_path, "synthetic reservation")
    require(
        reservation.get("artifact_type") == "pams_pose_recovery_v4e_synthetic_reservation_v1"
        and reservation.get("status") == "reserved"
        and reservation.get("full337_observed") is False
        and reservation.get("same39_observed") is False,
        "synthetic reservation is not pre-data",
    )
    require(
        output_path.parent.resolve(strict=True) / output_path.name
        == Path(str(reservation["future_evidence_locator"])),
        "synthetic evidence output locator mismatch",
    )
    bindings = reservation.get("bindings")
    require(isinstance(bindings, Mapping), "reservation bindings missing")
    require(os.environ.get("PAMS_CONTAINER_SOURCE_REVISION") == bindings["source_revision"], "source mismatch")
    require(os.environ.get("PAMS_CONTAINER_IMAGE_ID") == bindings["container_image_id"], "image mismatch")
    bound_paths = {
        "config_file": config_path,
        "model_asset": model_asset_path,
        "generator_source": generator_source_path,
        "threshold_grid": threshold_grid_path,
        "calibration_seed_manifest": calibration_seed_manifest_path,
        "heldout_seed_manifest": heldout_seed_manifest_path,
        "canonical_registry_reservation": registry_reservation_path,
    }
    for name, path in bound_paths.items():
        require(sha256_file(path) == bindings[f"{name}_sha256"], f"{name} SHA mismatch")
        require(path.stat().st_size == bindings[f"{name}_bytes"], f"{name} bytes mismatch")
    require(generator_source_path.resolve(strict=True) == Path(__file__).resolve(strict=True), "wrong synthetic producer source")
    grid = _object(threshold_grid_path, "threshold grid")
    require(grid.get("rows") == reservation.get("threshold_grid"), "threshold grid/reservation mismatch")
    split_seeds = {
        "calibration": _seed_manifest(calibration_seed_manifest_path, split="calibration"),
        "heldout": _seed_manifest(heldout_seed_manifest_path, split="heldout"),
    }
    require(set(split_seeds["calibration"]).isdisjoint(split_seeds["heldout"]), "seed splits overlap")
    config = load_config(config_path.resolve(strict=True))
    settings = config.pose.keypoint_single_source
    require(settings is not None, "v4e synthetic mechanics config missing")
    grid_rows = list(reservation["threshold_grid"])
    threshold_by_rank = {
        int(row["permissiveness_rank"]): TrackStabilityThresholds(**dict(row["thresholds"]))
        for row in grid_rows
        if isinstance(row, Mapping)
    }
    require(len(threshold_by_rank) == len(grid_rows), "synthetic threshold ranks duplicate")
    all_families = (
        *POSITIVE_FAMILIES, *IDENTITY_NULL_FAMILIES,
        *JOINT_NULL_FAMILIES, *DIAGNOSTIC_FAMILIES,
    )
    decision_vectors = {
        rank: {
            split: {family: [] for family in all_families}
            for split in split_seeds
        }
        for rank in threshold_by_rank
    }
    diagnostic_period_vectors = {
        split: {family: [] for family in DIAGNOSTIC_FAMILIES}
        for split in split_seeds
    }
    determinism_vectors = {
        split: [False] * SAMPLES_PER_FAMILY for split in split_seeds
    }
    permutation_vectors = {
        split: [False] * SAMPLES_PER_FAMILY for split in split_seeds
    }
    for split, seeds in split_seeds.items():
        for sample_index, seed in enumerate(seeds):
            for family in all_families:
                fixture, determinism_failure, permutation_failure = _fixture(
                    family, seed, settings
                )
                if family in IDENTITY_NULL_FAMILIES:
                    require(
                        fixture["truth_condition"] is True,
                        f"identity-null generator did not produce a truth switch: {family}",
                    )
                determinism_vectors[split][sample_index] |= determinism_failure
                permutation_vectors[split][sample_index] |= permutation_failure
                for rank, thresholds in threshold_by_rank.items():
                    decision_period_false = assess_track_stability(
                        fixture["audit"], thresholds=thresholds,
                        same_source_period_supported=False,
                    )
                    decision_period_true = assess_track_stability(
                        fixture["audit"], thresholds=thresholds,
                        same_source_period_supported=True,
                    )
                    if decision_period_false != decision_period_true:
                        determinism_vectors[split][sample_index] = True
                    permuted_decision = assess_track_stability(
                        fixture["permuted_audit"], thresholds=thresholds,
                        same_source_period_supported=False,
                    )
                    decision_fields = (
                        "eligible", "frame_identity_eligible_mask",
                        "track_stability_frame_ranges", "eligible_pair_spans",
                        "base_valid_pair_starts_by_variant",
                        "eligible_pair_starts_by_variant", "criteria", "quarantine_reasons",
                    )
                    if any(
                        decision_period_false[field] != permuted_decision[field]
                        for field in decision_fields
                    ):
                        permutation_vectors[split][sample_index] = True
                    eligible = bool(decision_period_false["eligible"])
                    if family in POSITIVE_FAMILIES:
                        outcome = eligible and bool(fixture["truth_condition"])
                    elif family in IDENTITY_NULL_FAMILIES:
                        outcome = eligible and bool(fixture["truth_condition"])
                    elif family in JOINT_NULL_FAMILIES:
                        outcome = eligible
                    else:
                        outcome = bool(fixture["period_supported"])
                    decision_vectors[rank][split][family].append(outcome)
                if family in DIAGNOSTIC_FAMILIES:
                    diagnostic_period_vectors[split][family].append(
                        bool(fixture["period_supported"])
                    )

    rows: list[dict[str, Any]] = []
    for grid_row in grid_rows:
        rank = int(grid_row["permissiveness_rank"])
        result_row: dict[str, Any] = {
            "permissiveness_rank": rank,
            "thresholds": dict(grid_row["thresholds"]),
        }
        for split, seeds in split_seeds.items():
            family_vectors = decision_vectors[rank][split]
            determinism = determinism_vectors[split]
            permutation = permutation_vectors[split]
            result_row[split] = {
                "positive_families": {
                    family: {
                        "total": SAMPLES_PER_FAMILY,
                        "retained": sum(family_vectors[family]),
                        "decision_bitset_hex": _bitset_hex(family_vectors[family]),
                    }
                    for family in POSITIVE_FAMILIES
                },
                "identity_null_families": {
                    family: {
                        "total": SAMPLES_PER_FAMILY,
                        "false_eligible": sum(family_vectors[family]),
                        "decision_bitset_hex": _bitset_hex(family_vectors[family]),
                    }
                    for family in IDENTITY_NULL_FAMILIES
                },
                "joint_null_families": {
                    family: {
                        "total": SAMPLES_PER_FAMILY,
                        "false_eligible": sum(family_vectors[family]),
                        "decision_bitset_hex": _bitset_hex(family_vectors[family]),
                    }
                    for family in JOINT_NULL_FAMILIES
                },
                "diagnostic_families": {
                    family: {
                        "total": SAMPLES_PER_FAMILY,
                        "period_supported": sum(diagnostic_period_vectors[split][family]),
                        "decision_bitset_hex": _bitset_hex(
                            diagnostic_period_vectors[split][family]
                        ),
                    }
                    for family in DIAGNOSTIC_FAMILIES
                },
                "determinism_failures": sum(determinism),
                "determinism_failure_bitset_hex": _bitset_hex(determinism),
                "candidate_order_permutation_failures": sum(permutation),
                "candidate_order_permutation_failure_bitset_hex": _bitset_hex(permutation),
                "seed_identity_sha256": sha256_file(
                    calibration_seed_manifest_path if split == "calibration" else heldout_seed_manifest_path
                ),
            }
        rows.append(result_row)

    result = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_synthetic_grid_evidence_v1",
        "label_free": True,
        "reservation_sha256": expected_reservation_sha256,
        "positive_families": list(POSITIVE_FAMILIES),
        "identity_null_families": list(IDENTITY_NULL_FAMILIES),
        "joint_null_families": list(JOINT_NULL_FAMILIES),
        "diagnostic_families": list(DIAGNOSTIC_FAMILIES),
        "rows": rows,
        "mechanics_chain": (
            "synthetic-coco17-candidates-to-dual-viterbi-to-canonical-frame-evidence-"
            "to-body-centered-cache-to-track-stability-v1"
        ),
        "truth_role": "synthetic-identity-retention-and-false-eligible-scoring-only",
        "period_invariance_checked": True,
        "candidate_order_permutation_checked": True,
        "bindings": dict(bindings),
        "real_dataset_opened": False,
        "annotations_or_count_labels_opened": False,
        "same39_observed": False,
        "full337_observed": False,
        "cycleback_representation_input_authorized": False,
        "baseline_training_authorized": False,
    }
    write_json_exclusive(output_path, result)
    os.chmod(output_path, 0o444)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reservation-sha256", required=True)
    for flag in (
        "reservation", "config", "model-asset", "generator-source", "threshold-grid",
        "calibration-seed-manifest", "heldout-seed-manifest", "registry-reservation", "output",
    ):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = produce_synthetic_evidence(
            reservation_path=args.reservation,
            expected_reservation_sha256=args.reservation_sha256,
            config_path=args.config,
            model_asset_path=args.model_asset,
            generator_source_path=args.generator_source,
            threshold_grid_path=args.threshold_grid,
            calibration_seed_manifest_path=args.calibration_seed_manifest,
            heldout_seed_manifest_path=args.heldout_seed_manifest,
            registry_reservation_path=args.registry_reservation,
            output_path=args.output,
        )
    except (OSError, TypeError, ValueError, Full337ContractError, KeyError) as exc:
        print(f"v4e synthetic evidence production failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
