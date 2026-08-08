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
    RELIABLE_ACTION_JOINTS,
    TrackStabilityThresholds,
    assess_track_stability,
    association_weights_from_settings,
    build_single_source_sequence,
    candidate_evidence_bundle,
    canonicalize_raw_detector_frame,
    effective_maximum_bridge_gap_frames,
    reconstruct_canonical_frame_evidence,
    select_segmented_stable_actor_paths,
)
from pams.v4e_synthetic_contract import (
    SYNTHETIC_DIAGNOSTIC_FAMILIES,
    SYNTHETIC_IDENTITY_NULL_FAMILIES,
    SYNTHETIC_JOINT_NULL_FAMILIES,
    SYNTHETIC_POSITIVE_FAMILIES,
    SYNTHETIC_SAMPLES_PER_FAMILY,
    frozen_synthetic_seeds,
)

POSITIVE_FAMILIES = SYNTHETIC_POSITIVE_FAMILIES
IDENTITY_NULL_FAMILIES = SYNTHETIC_IDENTITY_NULL_FAMILIES
JOINT_NULL_FAMILIES = SYNTHETIC_JOINT_NULL_FAMILIES
DIAGNOSTIC_FAMILIES = SYNTHETIC_DIAGNOSTIC_FAMILIES
SAMPLES_PER_FAMILY = SYNTHETIC_SAMPLES_PER_FAMILY
# Deliberately empty in the science-core commit. A later secure-launch commit
# must replace this with an independently inspected, source-bound authority.
APPROVED_SYNTHETIC_SECURE_LAUNCH_AUTHORITY_SHA256 = ""


def _object(path: Path, role: str) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, Mapping), f"{role} must be a JSON object")
    return value


def _seed_manifest(path: Path, *, split: str) -> tuple[int, ...]:
    value = _object(path, f"{split} seed manifest")
    require(
        set(value)
        == {
            "schema_version",
            "artifact_type",
            "split",
            "sample_total",
            "seeds",
            "real_data_observed",
            "label_free",
        }
        and value.get("schema_version") == 1
        and value.get("artifact_type")
        == "pams_pose_recovery_v4e_synthetic_seed_manifest_v1"
        and value.get("split") == split
        and value.get("sample_total") == SAMPLES_PER_FAMILY
        and value.get("real_data_observed") is False
        and value.get("label_free") is True,
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
    require(
        seeds == frozen_synthetic_seeds(split),
        f"{split} seed manifest differs from frozen derivation",
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


def _person_raw_channels(
    *,
    center: tuple[float, float],
    scale: float,
    phase: float,
    mask: np.ndarray,
    box_score: float,
    morphology: str,
    affine: tuple[float, float, float, float] | None = None,
    detector_jitter_amplitude: float = 0.0,
    articulated_motion_amplitude: float = 0.06,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate raw detector channels before any production filter."""

    shape = np.array(_BASE_SHAPE, copy=True)
    if morphology == "B":
        shape[[5, 11], 0] -= np.asarray([0.045, -0.018], dtype=np.float32)
        shape[[6, 12], 0] += np.asarray([0.045, -0.018], dtype=np.float32)
        shape[[0, 1, 2, 3, 4], 1] -= np.float32(0.025)
    elif morphology != "A":
        raise ValueError("unknown synthetic morphology")
    motion = np.float32(articulated_motion_amplitude * np.sin(phase))
    shape[[9, 10], 1] += np.asarray([motion, -motion], dtype=np.float32)
    shape[[15, 16], 0] += np.asarray([-motion, motion], dtype=np.float32)
    if detector_jitter_amplitude:
        jitter = np.float32(detector_jitter_amplitude * np.sin(phase))
        shape[list(RELIABLE_ACTION_JOINTS), 0] += jitter
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
    logits = np.where(mask, np.float32(3.0), np.float32(0.0))
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, :2] = xy * np.float32(1000.0)
    keypoints[:, 2] = np.float32(1.0)
    minimum = np.min(keypoints[:, :2], axis=0)
    maximum = np.max(keypoints[:, :2], axis=0)
    box = np.asarray([minimum[0], minimum[1], maximum[0], maximum[1]], dtype=np.float32)
    return keypoints, logits, box, float(box_score)


def _raw_frame(
    people: Sequence[tuple[str, tuple[np.ndarray, np.ndarray, np.ndarray, float]]],
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    if not people:
        return (
            {
                "labels": np.empty(0, dtype=np.int64),
                "scores": np.empty(0, dtype=np.float32),
                "boxes": np.empty((0, 4), dtype=np.float32),
                "keypoints": np.empty((0, 17, 3), dtype=np.float32),
                "keypoint_logits": np.empty((0, 17), dtype=np.float32),
            },
            (),
        )
    identities = tuple(identity for identity, _ in people)
    channels = tuple(value for _, value in people)
    return (
        {
            "labels": np.ones(len(channels), dtype=np.int64),
            "scores": np.asarray([value[3] for value in channels], dtype=np.float32),
            "boxes": np.stack([value[2] for value in channels], axis=0).astype(np.float32),
            "keypoints": np.stack([value[0] for value in channels], axis=0).astype(
                np.float32
            ),
            "keypoint_logits": np.stack([value[1] for value in channels], axis=0).astype(
                np.float32
            ),
        },
        identities,
    )


def _synthetic_raw_outputs(
    family: str, seed: int,
) -> tuple[list[dict[str, np.ndarray]], list[tuple[str, ...]], str]:
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
    raw_outputs: list[dict[str, np.ndarray]] = []
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
        if family == "low_amplitude_periodic_mask_flicker":
            action = list(RELIABLE_ACTION_JOINTS)
            for offset in range(3):
                mask_a[action[(frame + offset) % len(action)]] = False
        if family == "torso_only_without_action_joints":
            mask_a[:] = False
            mask_a[[5, 6, 11, 12]] = True
        if family == "alternating_limb_dropout_without_stable_window_support":
            mask_a[list(RELIABLE_ACTION_JOINTS)] = False
            mask_a[[7, 8, 9, 10] if frame % 2 == 0 else [13, 14, 15, 16]] = True
        if family == "eight_reliable_but_fewer_than_four_action_joints":
            mask_a[:] = False
            mask_a[[0, 1, 2, 3, 5, 6, 11, 12]] = True

        bridge_gap = max(1, min(6, int(seed % 6) + 1))
        if family == "long_gap_with_identity_change" and gap_start <= frame < gap_stop:
            raw, truth = _raw_frame(())
            raw_outputs.append(raw)
            truth_rows.append(truth)
            continue
        if family == "continuous_near_size_crossing_safe_abstention":
            progress = frame / (frames - 1)
            centers = (
                (
                    crossing_edge + (1.0 - 2.0 * crossing_edge) * progress,
                    target_center[1],
                ),
                (
                    1.0 - crossing_edge - (1.0 - 2.0 * crossing_edge) * progress,
                    target_center[1] + 0.005,
                ),
            )
            raw, truth = _raw_frame(
                (
                    (
                        "A",
                        _person_raw_channels(
                            center=centers[0], scale=0.66, phase=phase,
                            mask=mask_a, box_score=0.90, morphology="A",
                        ),
                    ),
                    (
                        "B",
                        _person_raw_channels(
                            center=centers[1], scale=0.65, phase=phase + 1.2,
                            mask=mask_b, box_score=0.90, morphology="A",
                        ),
                    ),
                )
            )
        elif family == "near_identical_alternating_score_complementary_phase":
            score_delta = np.float32(0.04 if frame % 2 == 0 else -0.04)
            raw, truth = _raw_frame(
                (
                    (
                        "A",
                        _person_raw_channels(
                            center=target_center,
                            scale=0.68,
                            phase=phase,
                            mask=mask_a,
                            box_score=float(np.float32(0.90) + score_delta),
                            morphology="A",
                        ),
                    ),
                    (
                        "B",
                        _person_raw_channels(
                            center=target_center,
                            scale=0.68,
                            phase=phase + np.pi,
                            mask=mask_b,
                            box_score=float(np.float32(0.90) - score_delta),
                            morphology="A",
                        ),
                    ),
                )
            )
        elif family == "forced_handoff_a_terminates_b_continues":
            swap_frame = frames // 2 + int(seed % 7) - 3
            identity = "A" if frame < swap_frame else "B"
            handoff_center = (
                target_center
                if identity == "A"
                else (target_center[0] + 0.05, target_center[1])
            )
            raw, truth = _raw_frame(
                (
                    (
                        identity,
                        _person_raw_channels(
                            center=handoff_center,
                            scale=0.68,
                            phase=phase if identity == "A" else phase + 1.2,
                            mask=mask_a,
                            box_score=0.90,
                            morphology=identity,
                        ),
                    ),
                )
            )
        elif family == "long_gap_with_identity_change":
            identity = "A" if frame < gap_start else "B"
            raw, truth = _raw_frame(
                ((
                    identity,
                    _person_raw_channels(
                        center=target_center, scale=target_scale,
                        phase=phase if identity == "A" else phase + 1.2,
                        mask=mask_a, box_score=0.90, morphology=identity,
                    ),
                ),)
            )
        elif family == "short_bridge_range_identity_change":
            bridge_start = frames // 2
            if bridge_start <= frame < bridge_start + bridge_gap:
                raw, truth = _raw_frame(())
            else:
                identity = "A" if frame < bridge_start else "B"
                bridge_center = (
                    target_center
                    if identity == "A"
                    else (target_center[0] + 0.30, target_center[1])
                )
                raw, truth = _raw_frame(
                    ((
                        identity,
                        _person_raw_channels(
                            center=bridge_center, scale=target_scale,
                            phase=phase if identity == "A" else phase + 1.2,
                            mask=mask_a, box_score=0.90, morphology=identity,
                        ),
                    ),)
                )
        elif family == "fast_motion_actor_with_large_static_bystander":
            raw, truth = _raw_frame(
                (
                    (
                        "A",
                        _person_raw_channels(
                            center=(0.30 + 0.0025 * frame, 0.50), scale=0.62,
                            phase=phase, mask=mask_a, box_score=0.94, morphology="A",
                        ),
                    ),
                    (
                        "B",
                        _person_raw_channels(
                            center=(0.70, 0.50), scale=0.80, phase=0.0,
                            mask=mask_b, box_score=0.90, morphology="B",
                        ),
                    ),
                )
            )
        else:
            jitter = (
                0.001
                if family == "periodic_detector_jitter_below_usable_motion"
                else 0.0
            )
            raw, truth = _raw_frame(
                (
                    (
                        "A",
                        _person_raw_channels(
                            center=target_center, scale=target_scale,
                            phase=phase, mask=mask_a, box_score=0.94,
                            morphology="A", affine=affine,
                            detector_jitter_amplitude=jitter,
                            articulated_motion_amplitude=(0.0 if jitter else 0.06),
                        ),
                    ),
                    (
                        "B",
                        _person_raw_channels(
                            center=(0.78, target_center[1]), scale=distractor_scale,
                            phase=0.0, mask=mask_b, box_score=0.95,
                            morphology="B", affine=affine,
                        ),
                    ),
                )
            )
        raw_outputs.append(raw)
        truth_rows.append(truth)
    return raw_outputs, truth_rows, "A"


def _canonicalize_synthetic_frames(
    raw_outputs: Sequence[Mapping[str, np.ndarray]],
    truth_rows: Sequence[tuple[str, ...]],
    *,
    settings: KeypointRCNNSingleSourceConfig,
) -> tuple[
    list[np.ndarray | None],
    list[tuple[str, ...]],
    list[dict[str, int]],
    tuple[str, ...],
]:
    candidates: list[np.ndarray | None] = []
    canonical_truth: list[tuple[str, ...]] = []
    diagnostics: list[dict[str, int]] = []
    canonical_digests: list[str] = []
    for raw, truth in zip(raw_outputs, truth_rows, strict=True):
        canonical = canonicalize_raw_detector_frame(
            labels=raw["labels"],
            scores=raw["scores"],
            boxes=raw["boxes"],
            keypoints=raw["keypoints"],
            keypoint_logits=raw["keypoint_logits"],
            image_width=1000,
            image_height=1000,
            settings=settings,
        )
        candidates.append(canonical.top_candidates)
        canonical_truth.append(
            tuple(truth[index] for index in canonical.canonical_source_indices[:4])
        )
        diagnostics.append(dict(canonical.diagnostics))
        canonical_digests.append(canonical.canonical_eligible_sha256)
    return candidates, canonical_truth, diagnostics, tuple(canonical_digests)


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
    *,
    family: str,
    seed: int,
    candidates: Sequence[np.ndarray | None],
    truth_rows: Sequence[tuple[str, ...]],
    detector_diagnostics: Sequence[Mapping[str, int]],
    canonical_filter_digests: Sequence[str],
    settings: KeypointRCNNSingleSourceConfig,
) -> dict[str, Any]:
    no_anchors = tuple(None for _ in candidates)
    primary_weights, secondary_weights = association_weights_from_settings(settings)
    maximum_gap = effective_maximum_bridge_gap_frames(
        fps=25.0,
        maximum_seconds=settings.maximum_bridge_gap_seconds,
        frame_cap=settings.maximum_bridge_gap_frame_cap,
    )
    primary_bank = select_segmented_stable_actor_paths(
        candidates, anchor_residuals=no_anchors, weights=primary_weights,
        maximum_bridge_gap_frames=maximum_gap,
    )
    secondary_bank = select_segmented_stable_actor_paths(
        candidates, anchor_residuals=no_anchors, weights=secondary_weights,
        maximum_bridge_gap_frames=maximum_gap,
    )
    primary = primary_bank.selected_path
    secondary = secondary_bank.selected_path
    segments = primary_bank.association_segments
    require(
        segments == secondary_bank.association_segments,
        "synthetic dual path segment mismatch",
    )
    video_id = f"synthetic:{family}:{seed}"
    sequence = build_single_source_sequence(
        video_id=video_id, fps=25.0, source_frames=len(candidates),
        decoded_frames=len(candidates), candidates=candidates, primary_path=primary,
    )
    frame_evidence = reconstruct_canonical_frame_evidence(
        candidates=candidates, primary=primary, secondary=secondary,
        association_segments=segments, sequence=sequence,
        ambiguity_by_frame=primary_bank.local_identity_gap_by_frame,
        maximum_bridge_gap_frames=maximum_gap,
        detector_diagnostics=detector_diagnostics,
        keypoint_logit_threshold=settings.keypoint_logit_threshold,
        identity_alternative_reachable_by_frame=(
            primary_bank.identity_alternative_reachable_by_frame
        ),
        identity_unresolvable_frames=primary_bank.identity_unresolvable_frames,
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
    candidate_bundle = candidate_evidence_bundle(candidates)
    audit = {
        "video_id_sha256": digest,
        "video_evidence_sha256": hashlib.sha256(
            json.dumps(frame_evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "source_frames": len(candidates),
        "dual_path_agreement": float(agreement),
        "identity_association_motion_free": True,
        "primary_identity_hypothesis_table_sha256": (
            primary_bank.identity_hypothesis_table_sha256
        ),
        "secondary_identity_hypothesis_table_sha256": (
            secondary_bank.identity_hypothesis_table_sha256
        ),
        "primary_stable_track_hypotheses": list(primary_bank.hypothesis_rows),
        "secondary_stable_track_hypotheses": list(secondary_bank.hypothesis_rows),
        "selected_actor_pair_utility_margin_by_variant": {
            name: {str(start): float(value) for start, value in rows.items()}
            for name, rows in (
                primary_bank.selected_pair_utility_margin_by_variant.items()
            )
        },
        "frame_evidence": frame_evidence,
        "canonical_filter_digest_sha256": hashlib.sha256(
            "".join(canonical_filter_digests).encode()
        ).hexdigest(),
        "canonical_candidate_bundle_sha256": hashlib.sha256(
            candidate_bundle.frame_offsets.tobytes(order="C")
            + candidate_bundle.candidates.tobytes(order="C")
        ).hexdigest(),
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
    raw_outputs, raw_truths, target = _synthetic_raw_outputs(family, seed)
    candidates, truths, diagnostics, canonical_digests = _canonicalize_synthetic_frames(
        raw_outputs, raw_truths, settings=settings
    )
    first = _run_mechanics(
        family=family,
        seed=seed,
        candidates=candidates,
        truth_rows=truths,
        detector_diagnostics=diagnostics,
        canonical_filter_digests=canonical_digests,
        settings=settings,
    )
    repeat_candidates, repeat_truths, repeat_diagnostics, repeat_digests = (
        _canonicalize_synthetic_frames(raw_outputs, raw_truths, settings=settings)
    )
    repeat = _run_mechanics(
        family=family,
        seed=seed,
        candidates=repeat_candidates,
        truth_rows=repeat_truths,
        detector_diagnostics=repeat_diagnostics,
        canonical_filter_digests=repeat_digests,
        settings=settings,
    )
    determinism_failure = not (
        first["audit"] == repeat["audit"]
        and np.array_equal(first["sequence_xyz"], repeat["sequence_xyz"])
        and np.array_equal(first["sequence_valid_mask"], repeat["sequence_valid_mask"])
        and first["selected_truth"] == repeat["selected_truth"]
    )
    rng = np.random.default_rng(seed ^ 0x5A17)
    permuted_raw_outputs: list[dict[str, np.ndarray]] = []
    permuted_raw_truths: list[tuple[str, ...]] = []
    for raw, truth in zip(raw_outputs, raw_truths, strict=True):
        order = rng.permutation(len(raw["labels"]))
        permuted_raw_outputs.append(
            {
                key: np.ascontiguousarray(value[order])
                for key, value in raw.items()
            }
        )
        permuted_raw_truths.append(tuple(truth[int(index)] for index in order))
    (
        permuted_candidates,
        permuted_truths,
        permuted_diagnostics,
        permuted_digests,
    ) = _canonicalize_synthetic_frames(
        permuted_raw_outputs, permuted_raw_truths, settings=settings
    )
    permuted = _run_mechanics(
        family=family,
        seed=seed,
        candidates=permuted_candidates,
        truth_rows=permuted_truths,
        detector_diagnostics=permuted_diagnostics,
        canonical_filter_digests=permuted_digests,
        settings=settings,
    )
    permutation_failure = not (
        np.array_equal(first["sequence_xyz"], permuted["sequence_xyz"])
        and np.array_equal(first["sequence_valid_mask"], permuted["sequence_valid_mask"])
        and first["selected_truth"] == permuted["selected_truth"]
        and first["audit"] == permuted["audit"]
    )
    first["permuted_audit"] = permuted["audit"]
    selected = {value for value in first["selected_truth"] if value is not None}
    if family in POSITIVE_FAMILIES:
        first["truth_condition"] = selected == {target}
    elif family in IDENTITY_NULL_FAMILIES:
        # Identity-null validity is fixed by generator construction, never by
        # whether the algorithm happened to switch.  Any eligible decision is
        # therefore a false-eligible outcome for these families.
        first["truth_condition"] = True
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
    require(
        len(APPROVED_SYNTHETIC_SECURE_LAUNCH_AUTHORITY_SHA256) == 64,
        "v4e synthetic production is disabled pending monotonic secure authority",
    )
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
                        outcome = eligible
                    elif family in JOINT_NULL_FAMILIES:
                        outcome = eligible
                    else:
                        # The crossing diagnostic preregisters safe abstention;
                        # it makes no target-ID retention claim. Eligibility is
                        # therefore the diagnostic failure bit.
                        outcome = eligible
                    decision_vectors[rank][split][family].append(outcome)

    rows: list[dict[str, Any]] = []
    for grid_row in grid_rows:
        rank = int(grid_row["permissiveness_rank"])
        result_row: dict[str, Any] = {
            "permissiveness_rank": rank,
            "thresholds": dict(grid_row["thresholds"]),
        }
        for split, _seeds in split_seeds.items():
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
                        "false_eligible": sum(family_vectors[family]),
                        "decision_bitset_hex": _bitset_hex(family_vectors[family]),
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
            "synthetic-raw-kprcnn-channels-to-shared-production-filter-canonical-sort-"
            "to-dual-motion-free-stable-track-banks-to-whole-track-actor-utility-"
            "to-canonical-frame-evidence-to-body-centered-cache-to-track-stability-v3"
        ),
        "truth_role": (
            "synthetic-actor-retention-track-stability-and-false-eligible-scoring-only"
        ),
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
