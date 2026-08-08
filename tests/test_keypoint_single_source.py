from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pams.config import load_config
from pams.keypoint_single_source import (
    AssociationWeights,
    TrackStabilityThresholds,
    _anchor_residuals,
    assess_track_stability,
    body_centered_uniform_scale_xy,
    build_single_source_sequence,
    candidate_evidence_bundle,
    canonicalize_raw_detector_frame,
    coco17_xy_to_padded_pose,
    effective_maximum_bridge_gap_frames,
    local_ambiguity_gaps,
    load_candidate_evidence_npz,
    select_top2_viterbi_paths,
    similarity_procrustes_residual,
    write_candidate_evidence_npz,
)
from pams.types import PoseSequence
from pams.v4e_synthetic_contract import (
    canonical_threshold_grid_rows,
    frozen_synthetic_seeds,
    frozen_threshold_axes,
)


def _shape() -> np.ndarray:
    index = np.arange(17, dtype=np.float32)
    return np.stack(
        (
            0.15 + 0.035 * index + 0.02 * np.sin(index),
            0.20 + 0.025 * index + 0.03 * np.cos(index * 0.7),
        ),
        axis=1,
    ).astype(np.float32)


def _candidate(xy: np.ndarray, confidence: float = 0.9) -> np.ndarray:
    value = np.zeros((17, 4), dtype=np.float32)
    value[:, :2] = xy
    value[:, 3] = confidence
    return value


def _raw_candidate(xy: np.ndarray, *, box_score: float = 0.9) -> np.ndarray:
    value = np.zeros((17, 6), dtype=np.float32)
    value[:, :2] = xy
    value[:, 4] = 3.0
    value[:, 5] = box_score
    value[:, 3] = np.float32(1.0 / (1.0 + np.exp(-3.0)) * box_score)
    return value


def _mapped_base(xy: np.ndarray, frames: int) -> PoseSequence:
    xyz = np.zeros((frames, 33, 3), dtype=np.float32)
    # The exact shared MediaPipe representatives in COCO order.
    indices = np.asarray([0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])
    xyz[:, indices, :2] = xy
    return PoseSequence(
        video_id="fixture",
        fps=25.0,
        xyz=xyz,
        valid_mask=np.ones(frames, dtype=np.bool_),
    )


def test_v4e_config_is_single_source_and_raw_non_authoritative() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs/experiments/pams_pose_recovery_v4e_single_source_raw.yaml"
    )
    settings = config.pose.keypoint_single_source

    assert settings is not None
    assert settings.single_source_full_track is True
    assert settings.v4d_candidate_cache_consumed is False
    assert settings.raw_extraction_authorizes_training is False
    assert settings.minimum_confident_keypoints == 8
    assert settings.minimum_confident_action_keypoints == 4
    assert settings.primary_action_motion_weight == 12.0
    assert settings.secondary_action_motion_weight == 9.0
    assert config.data.normalization == "body_centered_uniform_scale"


def test_generic_encoder_hard_rejects_v4e_before_materializing_inputs() -> None:
    from pams.training import train_encoder

    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs/experiments/pams_pose_recovery_v4e_single_source_raw.yaml"
    )
    with pytest.raises(RuntimeError, match="integration-authorized custom runner"):
        train_encoder((), config)


def test_generic_sshead_hard_rejects_v4e_before_touching_model() -> None:
    from pams.training import train_sshead

    root = Path(__file__).resolve().parents[1]
    config = load_config(
        root / "configs/experiments/pams_pose_recovery_v4e_single_source_raw.yaml"
    )
    with pytest.raises(RuntimeError, match="integration-authorized custom runner"):
        train_sshead((), config, model=None)  # type: ignore[arg-type]


def test_synthetic_axes_and_seed_splits_are_exact_and_not_caller_selectable() -> None:
    axes = frozen_threshold_axes()
    rows = canonical_threshold_grid_rows(axes)
    calibration = frozen_synthetic_seeds("calibration")
    heldout = frozen_synthetic_seeds("heldout")

    assert rows
    assert len(calibration) == len(set(calibration)) == 512
    assert len(heldout) == len(set(heldout)) == 512
    assert not set(calibration) & set(heldout)
    changed = frozen_threshold_axes()
    changed["minimum_window_action_motion"] = [0.0]
    with pytest.raises(ValueError, match="differ from the frozen synthetic contract"):
        canonical_threshold_grid_rows(changed)


def test_body_centered_uniform_scale_uses_coco17_and_zero_padding() -> None:
    normalized = body_centered_uniform_scale_xy(_shape())
    center = 0.5 * (normalized[11] + normalized[12])
    radius = np.sqrt(np.mean(np.sum(np.square(normalized), axis=1)))
    padded = coco17_xy_to_padded_pose(_shape())

    assert np.allclose(center, 0.0, atol=1e-6)
    assert np.isclose(radius, 1.0, atol=1e-6)
    assert padded.shape == (33, 3)
    assert np.array_equal(padded[:17, :2], normalized)
    assert np.all(padded[:, 2] == 0.0)
    assert np.all(padded[17:] == 0.0)


def test_procrustes_allows_rotation_but_forbids_reflection() -> None:
    source = _shape()
    rotation = np.asarray([[0.0, -1.0], [1.0, 0.0]], dtype=np.float32)
    rotated = source @ rotation + np.asarray([0.2, 0.3], dtype=np.float32)
    reflected = source * np.asarray([-1.0, 1.0], dtype=np.float32)

    assert similarity_procrustes_residual(source, rotated) < 1e-5
    assert similarity_procrustes_residual(source, reflected) > 1e-3


def test_top2_viterbi_emits_margin_and_dual_weight_path_evidence() -> None:
    shape = _shape()
    frames = 5
    base = _mapped_base(shape, frames)
    candidates = []
    for frame in range(frames):
        target = _candidate(shape + np.asarray([0.01 * frame, 0.0], dtype=np.float32), 0.92)
        distractor_shape = shape.copy()
        distractor_shape[0] += np.asarray([0.3, -0.2], dtype=np.float32)
        distractor = _candidate(
            distractor_shape + np.asarray([0.7 - 0.02 * frame, 0.1], dtype=np.float32),
            0.88,
        )
        candidates.append(np.stack((target, distractor), axis=0))
    anchors = _anchor_residuals(candidates, base)
    path = select_top2_viterbi_paths(
        candidates,
        anchor_residuals=anchors,
        weights=AssociationWeights(center=1.0, log_scale=0.25, shape=0.0, anchor=2.0),
    )
    gaps = local_ambiguity_gaps(
        candidates,
        anchor_residuals=anchors,
        weights=AssociationWeights(center=1.0, log_scale=0.25, shape=0.0, anchor=2.0),
        selected_path=path,
    )
    sequence = build_single_source_sequence(
        video_id="fixture",
        fps=25.0,
        source_frames=frames,
        decoded_frames=frames,
        candidates=candidates,
        primary_path=path,
    )

    assert path.selected_indices == (0, 0, 0, 0, 0)
    assert path.runner_up_score is not None
    assert path.margin_per_observed_frame is not None
    assert path.margin_per_observed_frame > 0.0
    assert len(gaps) == frames
    assert min(gaps) > 0.0
    assert sequence.valid_mask.tolist() == [True] * frames
    assert np.all(sequence.xyz[:, 17:] == 0.0)


def test_dominant_subject_unary_rejects_small_high_confidence_bystander() -> None:
    shape = _shape()
    small = _candidate(0.20 + 0.20 * shape, confidence=0.99)
    large = _candidate(0.05 + 0.85 * shape, confidence=0.80)
    path = select_top2_viterbi_paths(
        [np.stack((small, large), axis=0)],
        anchor_residuals=[None],
        weights=AssociationWeights(center=1.0, log_scale=0.25, shape=0.0, anchor=0.0),
    )

    assert path.selected_indices == (1,)


def test_nonrigid_action_motion_retains_actor_over_larger_static_bystander() -> None:
    base = _shape()
    candidates = []
    for frame in range(96):
        actor_xy = 0.12 + 0.62 * base
        motion = np.float32(0.06 * np.sin(2.0 * np.pi * frame / 16.0))
        actor_xy = np.array(actor_xy, copy=True)
        actor_xy[[9, 10], 1] += np.asarray([motion, -motion], dtype=np.float32)
        actor_xy[[15, 16], 0] += np.asarray([-motion, motion], dtype=np.float32)
        actor = _raw_candidate(actor_xy, box_score=0.94)
        static = _raw_candidate(0.08 + 0.80 * base, box_score=0.90)
        candidates.append(np.stack((actor, static), axis=0))
    path = select_top2_viterbi_paths(
        candidates,
        anchor_residuals=[None] * len(candidates),
        weights=AssociationWeights(
            center=1.0,
            log_scale=0.25,
            shape=0.5,
            anchor=0.0,
            action_motion=12.0,
        ),
    )

    assert path.selected_indices == (0,) * len(candidates)


def test_pretop4_candidate_artifact_retains_fifth_candidate_for_crowd_audit() -> None:
    candidates = np.stack(
        tuple(_raw_candidate(_shape() + 0.01 * index) for index in range(5)),
        axis=0,
    )
    bundle = candidate_evidence_bundle([candidates])

    assert bundle.frame_offsets.tolist() == [0, 5]
    assert bundle.candidates.shape == (5, 17, 6)


def test_raw_detector_filter_is_permutation_invariant_and_npz_roundtrips(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    settings = load_config(
        root / "configs/experiments/pams_pose_recovery_v4e_single_source_raw.yaml"
    ).pose.keypoint_single_source
    assert settings is not None
    xy = np.stack((_shape(), 0.1 + 0.75 * _shape()), axis=0) * np.float32(1000.0)
    keypoints = np.zeros((2, 17, 3), dtype=np.float32)
    keypoints[:, :, :2] = xy
    keypoints[:, :, 2] = 1.0
    boxes = np.stack(
        tuple(
            np.asarray(
                [
                    np.min(row[:, 0]),
                    np.min(row[:, 1]),
                    np.max(row[:, 0]),
                    np.max(row[:, 1]),
                ],
                dtype=np.float32,
            )
            for row in xy
        ),
        axis=0,
    )
    inputs = {
        "labels": np.ones(2, dtype=np.int64),
        "scores": np.asarray([0.83, 0.91], dtype=np.float32),
        "boxes": boxes,
        "keypoints": keypoints,
        "keypoint_logits": np.full((2, 17), 3.0, dtype=np.float32),
    }
    first = canonicalize_raw_detector_frame(
        **inputs, image_width=1000, image_height=1000, settings=settings
    )
    order = np.asarray([1, 0], dtype=np.int64)
    permuted = canonicalize_raw_detector_frame(
        **{key: value[order] for key, value in inputs.items()},
        image_width=1000,
        image_height=1000,
        settings=settings,
    )

    assert np.array_equal(first.top_candidates, permuted.top_candidates)
    assert np.array_equal(first.all_eligible_candidates, permuted.all_eligible_candidates)
    assert first.canonical_eligible_sha256 == permuted.canonical_eligible_sha256
    assert first.raw_output_sha256 != permuted.raw_output_sha256
    assert first.diagnostics == permuted.diagnostics
    bundle = candidate_evidence_bundle([first.all_eligible_candidates])
    artifact = tmp_path / "candidates.npz"
    write_candidate_evidence_npz(artifact, bundle)
    loaded = load_candidate_evidence_npz(artifact)
    assert np.array_equal(loaded.frame_offsets, bundle.frame_offsets)
    assert np.array_equal(loaded.candidates, bundle.candidates)


def test_torso_only_raw_detection_is_rejected_by_shared_production_filter() -> None:
    root = Path(__file__).resolve().parents[1]
    settings = load_config(
        root / "configs/experiments/pams_pose_recovery_v4e_single_source_raw.yaml"
    ).pose.keypoint_single_source
    assert settings is not None
    keypoints = np.zeros((1, 17, 3), dtype=np.float32)
    keypoints[0, :, :2] = _shape() * np.float32(1000.0)
    logits = np.zeros((1, 17), dtype=np.float32)
    logits[0, [5, 6, 11, 12]] = 3.0
    result = canonicalize_raw_detector_frame(
        labels=np.ones(1, dtype=np.int64),
        scores=np.asarray([0.9], dtype=np.float32),
        boxes=np.asarray([[0.0, 0.0, 900.0, 900.0]], dtype=np.float32),
        keypoints=keypoints,
        keypoint_logits=logits,
        image_width=1000,
        image_height=1000,
        settings=settings,
    )

    assert result.top_candidates is None
    assert result.all_eligible_candidates is None
    assert result.diagnostics["rejected_low_total_joint_support"] == 1


def test_kprcnn_stability_gate_never_claims_target_identity() -> None:
    thresholds = TrackStabilityThresholds(
        minimum_frame_local_ambiguity_gap=0.02,
        minimum_dual_path_agreement=0.90,
        maximum_frame_center_step=0.30,
        maximum_frame_log_scale_step=0.30,
        maximum_frame_morphology_step=0.10,
        maximum_frame_joint_mask_flicker_fraction=0.25,
        minimum_source_coverage=0.80,
        minimum_longest_trainable_segment_frames=48,
        minimum_longest_trainable_segment_fraction=0.80,
        maximum_candidate_window_frames=24,
        minimum_window_joint_support_fraction=0.75,
        minimum_window_stable_action_joints=4,
        minimum_window_action_motion=0.0,
    )
    evidence = {
        "video_id_sha256": "1" * 64,
        "video_evidence_sha256": "2" * 64,
        "anchor_observed_frames": 0,
        "anchor_residual_p90": None,
        "local_ambiguous_frame_total": 10,
        "local_ambiguity_gap_p10": 0.06,
        "dual_path_agreement": 1.0,
        "continuity_step_p95": 0.1,
        "source_coverage": 1.0,
        "source_frames": 48,
        "frame_evidence": [
            {
                "frame_index": index,
                "candidate_count": 1,
                "primary_index": 0,
                "candidates": [
                    {
                        "raw_keypoint_logits": [3.0] * 17,
                        "body_centered_uniform_scale_xy": (
                            body_centered_uniform_scale_xy(_shape()).tolist()
                        ),
                    }
                ],
                "raw_normalization_eligible": True,
                "continuity_from_previous": (
                    None
                    if index == 0
                    else {
                        "previous_frame_index": index - 1,
                        "center_step_normalized_per_frame": 0.1,
                        "absolute_log_scale_step_per_frame": 0.0,
                        "torso_morphology_step": 0.0,
                    }
                ),
            }
            for index in range(48)
        ],
    }
    decision = assess_track_stability(
        evidence,
        thresholds=thresholds,
        same_source_period_supported=True,
    )
    period_false_decision = assess_track_stability(
        evidence,
        thresholds=thresholds,
        same_source_period_supported=False,
    )

    assert decision == period_false_decision
    assert decision["eligible"] is True
    assert decision["mediapipe_shape_evidence_scope"] == "diagnostic-only-not-used-for-eligibility"
    assert decision["identity_claim"] == "kprcnn-stability-only-no-target-identity-ground-truth"
    assert decision["baseline_training_authorized"] is False
    assert decision["eligible_pair_starts_by_variant"] == {
        "W16_H2": list(range(0, 17, 2)),
        "W16_H4": list(range(0, 17, 4)),
        "W16_H4_PE0": list(range(0, 17, 4)),
        "W24_H4": [0],
    }
    assert decision["pair_variant_aliases"] == {"W16_H4_PE0": "W16_H4"}


def test_physical_gap_rule_has_no_hidden_one_frame_floor() -> None:
    assert effective_maximum_bridge_gap_frames(fps=2.0, maximum_seconds=0.25, frame_cap=8) == 0
    assert effective_maximum_bridge_gap_frames(fps=25.0, maximum_seconds=0.25, frame_cap=8) == 6
    assert effective_maximum_bridge_gap_frames(fps=120.0, maximum_seconds=0.25, frame_cap=8) == 8


def test_low_confidence_joints_are_exact_zero_after_normalization() -> None:
    visibility = np.ones(17, dtype=np.float32)
    visibility[3] = 0.0
    normalized = body_centered_uniform_scale_xy(_shape(), visibility=visibility)

    assert np.array_equal(normalized[3], np.zeros(2, dtype=np.float32))


def test_cache_builder_respects_zero_partial_and_full_decoded_boundaries() -> None:
    shape = _shape()
    candidates = [np.stack((_candidate(shape),), axis=0) for _ in range(4)]
    path = select_top2_viterbi_paths(
        candidates,
        anchor_residuals=[None] * 4,
        weights=AssociationWeights(center=1.0, log_scale=0.25, shape=0.0, anchor=0.0),
    )

    masks = [
        build_single_source_sequence(
            video_id="fixture",
            fps=25.0,
            source_frames=4,
            decoded_frames=decoded,
            candidates=candidates,
            primary_path=path,
        ).valid_mask.tolist()
        for decoded in (0, 2, 4)
    ]

    assert masks == [
        [False, False, False, False],
        [True, True, False, False],
        [True, True, True, True],
    ]


def test_periodic_mask_flicker_and_torso_only_are_not_trainable() -> None:
    thresholds = TrackStabilityThresholds(
        minimum_frame_local_ambiguity_gap=0.0,
        minimum_dual_path_agreement=0.85,
        maximum_frame_center_step=0.5,
        maximum_frame_log_scale_step=0.5,
        maximum_frame_morphology_step=0.12,
        maximum_frame_joint_mask_flicker_fraction=0.3,
        minimum_source_coverage=0.8,
        minimum_longest_trainable_segment_frames=48,
        minimum_longest_trainable_segment_fraction=0.8,
        maximum_candidate_window_frames=24,
        minimum_window_joint_support_fraction=0.75,
        minimum_window_stable_action_joints=4,
        minimum_window_action_motion=0.0,
    )

    def decision(*, torso_only: bool) -> dict[str, object]:
        rows = []
        for index in range(96):
            reliable = np.ones(17, dtype=np.bool_)
            if torso_only:
                reliable[[7, 8, 9, 10, 13, 14, 15, 16]] = False
            else:
                reliable[[7, 8, 9, 10] if index % 2 else [13, 14, 15, 16]] = False
            rows.append(
                {
                    "frame_index": index,
                    "candidate_count": 1,
                    "primary_index": 0,
                    "candidates": [
                        {
                            "raw_keypoint_logits": [
                                3.0 if value else 0.0 for value in reliable
                            ],
                            "body_centered_uniform_scale_xy": (
                                body_centered_uniform_scale_xy(_shape()).tolist()
                            ),
                        }
                    ],
                    "raw_normalization_eligible": not torso_only,
                    "continuity_from_previous": (
                        None
                        if index == 0
                        else {
                            "previous_frame_index": index - 1,
                            "center_step_normalized_per_frame": 0.01,
                            "absolute_log_scale_step_per_frame": 0.01,
                            "torso_morphology_step": 0.0,
                        }
                    ),
                }
            )
        return assess_track_stability(
            {
                "video_id_sha256": "3" * 64,
                "video_evidence_sha256": "4" * 64,
                "source_frames": 96,
                "dual_path_agreement": 1.0,
                "frame_evidence": rows,
            },
            thresholds=thresholds,
            same_source_period_supported=False,
        )

    assert decision(torso_only=False)["eligible"] is False
    assert decision(torso_only=True)["eligible"] is False


def test_candidate_order_permutation_preserves_selected_subject_geometry() -> None:
    large = _candidate(_shape(), confidence=0.85)
    small = _candidate(0.35 + 0.2 * _shape(), confidence=0.99)
    original = [np.stack((large, small), axis=0) for _ in range(8)]
    permuted = [frame[[1, 0]] if index % 2 else frame for index, frame in enumerate(original)]
    weights = AssociationWeights(center=1.0, log_scale=0.25, shape=0.0, anchor=0.0)
    original_path = select_top2_viterbi_paths(
        original, anchor_residuals=[None] * 8, weights=weights
    )
    permuted_path = select_top2_viterbi_paths(
        permuted, anchor_residuals=[None] * 8, weights=weights
    )

    original_selected = [
        original[index][int(selected)]
        for index, selected in enumerate(original_path.selected_indices)
    ]
    permuted_selected = [
        permuted[index][int(selected)]
        for index, selected in enumerate(permuted_path.selected_indices)
    ]
    assert all(
        np.array_equal(left, right)
        for left, right in zip(original_selected, permuted_selected, strict=True)
    )


def test_v4e_full337_cli_source_is_fail_closed_on_canonical_authority_chain() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "scripts/server/run_pose_recovery_v4e_single_source_raw.py").read_text(
        encoding="utf-8"
    )
    full_authorizer = (root / "scripts/server/authorize_pose_recovery_v4e_raw.py").read_text(
        encoding="utf-8"
    )
    synthetic_producer = (
        root / "scripts/server/produce_pose_recovery_v4e_synthetic_evidence.py"
    ).read_text(encoding="utf-8")

    assert "v4e-full337-raw.authorization.json" in runner
    assert "canonical_registry_reservation_sha256" in runner
    assert "same39_bytewise_reuse_required" in runner
    assert "v4e-same39.gate.json" in full_authorizer
    assert 'preregistered_gate.get("status") == "passed"' in full_authorizer
    assert 'preregistered_gate.get("overall_pass") is True' in full_authorizer.replace(
        "\n", " "
    ).replace("        ", " ")
    assert '"cycleback_representation_input_authorized": False' in full_authorizer
    assert '"baseline_training_authorized": False' in full_authorizer
    assert 'APPROVED_RAW_EXTRACTION_SECURE_LAUNCH_AUTHORITY_SHA256 = ""' in runner
    assert 'APPROVED_SYNTHETIC_SECURE_LAUNCH_AUTHORITY_SHA256 = ""' in synthetic_producer
