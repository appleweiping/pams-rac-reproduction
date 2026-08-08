"""Pure preregistration contract for the v4e synthetic threshold grid."""

from __future__ import annotations

import hashlib
import itertools
import math
from collections.abc import Mapping
from typing import Any

FROZEN_THRESHOLD_FIELD_ORDER = (
    "maximum_candidate_window_frames",
    "maximum_frame_center_step",
    "maximum_frame_joint_mask_flicker_fraction",
    "maximum_frame_log_scale_step",
    "maximum_frame_morphology_step",
    "minimum_dual_path_agreement",
    "minimum_frame_local_ambiguity_gap",
    "minimum_longest_trainable_segment_fraction",
    "minimum_longest_trainable_segment_frames",
    "minimum_source_coverage",
    "minimum_window_joint_support_fraction",
    "minimum_window_action_motion",
    "minimum_window_stable_action_joints",
)
_DESCENDING_AXES = {
    "maximum_frame_center_step",
    "maximum_frame_joint_mask_flicker_fraction",
    "maximum_frame_log_scale_step",
    "maximum_frame_morphology_step",
}
_ASCENDING_AXES = {
    "minimum_dual_path_agreement",
    "minimum_frame_local_ambiguity_gap",
    "minimum_longest_trainable_segment_fraction",
    "minimum_source_coverage",
    "minimum_window_joint_support_fraction",
    "minimum_window_action_motion",
    "minimum_window_stable_action_joints",
}
SYNTHETIC_SAMPLES_PER_FAMILY = 512
SYNTHETIC_POSITIVE_FAMILIES = (
    "clean_known_identity",
    "short_occlusion_random30pct_joints_for20pct_time",
    "inplane_affine_rotation15_scale15_translation10pct",
    "fast_motion_actor_with_large_static_bystander",
)
SYNTHETIC_IDENTITY_NULL_FAMILIES = (
    "near_identical_alternating_score_complementary_phase",
    "forced_handoff_a_terminates_b_continues",
    "long_gap_with_identity_change",
    "short_bridge_range_identity_change",
)
SYNTHETIC_JOINT_NULL_FAMILIES = (
    "periodic_joint_mask_flicker",
    "low_amplitude_periodic_mask_flicker",
    "torso_only_without_action_joints",
    "alternating_limb_dropout_without_stable_window_support",
    "eight_reliable_but_fewer_than_four_action_joints",
    "periodic_detector_jitter_below_usable_motion",
)
SYNTHETIC_DIAGNOSTIC_FAMILIES = (
    # Pose-only evidence can be information-theoretically ambiguous at a
    # near-identical crossing.  v1 preregisters safe abstention as a reported
    # diagnostic and makes no target-identity retention claim for this family.
    "continuous_near_size_crossing_safe_abstention",
)
_FROZEN_THRESHOLD_AXES: dict[str, tuple[int | float, ...]] = {
    "maximum_candidate_window_frames": (24,),
    "maximum_frame_center_step": (0.5, 0.3),
    "maximum_frame_joint_mask_flicker_fraction": (0.5, 0.3),
    "maximum_frame_log_scale_step": (0.5, 0.3),
    "maximum_frame_morphology_step": (0.12, 0.08),
    "minimum_dual_path_agreement": (0.85, 0.95),
    "minimum_frame_local_ambiguity_gap": (0.0, 0.02),
    "minimum_longest_trainable_segment_fraction": (0.8,),
    "minimum_longest_trainable_segment_frames": (48,),
    "minimum_source_coverage": (0.8,),
    "minimum_window_joint_support_fraction": (0.75,),
    "minimum_window_action_motion": (0.005, 0.01),
    "minimum_window_stable_action_joints": (4,),
}
_INTEGER_THRESHOLD_FIELDS = {
    "maximum_candidate_window_frames",
    "minimum_longest_trainable_segment_frames",
    "minimum_window_stable_action_joints",
}


def frozen_threshold_axes() -> dict[str, list[int | float]]:
    """Return a fresh JSON-compatible copy of the only allowed grid axes."""

    return {
        field: list(_FROZEN_THRESHOLD_AXES[field])
        for field in FROZEN_THRESHOLD_FIELD_ORDER
    }


def frozen_synthetic_seeds(split: str) -> list[int]:
    """Derive the exact disjoint uint63 seed identities before real data."""

    if split not in {"calibration", "heldout"}:
        raise ValueError("synthetic seed split must be calibration or heldout")
    return [
        int.from_bytes(
            hashlib.sha256(
                f"pams-v4e-synthetic-v1:{split}:{index}".encode()
            ).digest()[:8],
            "big",
        )
        & ((1 << 63) - 1)
        for index in range(SYNTHETIC_SAMPLES_PER_FAMILY)
    ]


def canonical_threshold_grid_rows(
    axes_value: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate ordered axes and return the unique Cartesian severity order."""

    if tuple(axes_value) != FROZEN_THRESHOLD_FIELD_ORDER:
        raise ValueError("threshold axes are not in frozen field order")
    if {
        field: list(axes_value[field])
        for field in FROZEN_THRESHOLD_FIELD_ORDER
    } != frozen_threshold_axes():
        raise ValueError("threshold axes differ from the frozen synthetic contract")
    axes: dict[str, list[int | float]] = {}
    for field in FROZEN_THRESHOLD_FIELD_ORDER:
        raw = axes_value.get(field)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"threshold axis missing: {field}")
        if not all(
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in raw
        ):
            raise ValueError(f"threshold axis is non-numeric: {field}")
        if field in _INTEGER_THRESHOLD_FIELDS and not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in raw
        ):
            raise ValueError(f"integer threshold axis has wrong scalar type: {field}")
        if field not in _INTEGER_THRESHOLD_FIELDS and not all(
            isinstance(value, float) for value in raw
        ):
            raise ValueError(f"float threshold axis has wrong scalar type: {field}")
        if len({float(value) for value in raw}) != len(raw):
            raise ValueError(f"threshold axis contains duplicates: {field}")
        if field in _DESCENDING_AXES and any(
            float(left) <= float(right)
            for left, right in zip(raw, raw[1:], strict=False)
        ):
            raise ValueError(f"maximum threshold axis is not permissive-to-strict: {field}")
        if field in _ASCENDING_AXES and any(
            float(left) >= float(right)
            for left, right in zip(raw, raw[1:], strict=False)
        ):
            raise ValueError(f"minimum threshold axis is not permissive-to-strict: {field}")
        axes[field] = list(raw)
    if axes["maximum_candidate_window_frames"] != [24]:
        raise ValueError("maximum candidate window axis must be exactly [24]")
    if axes["minimum_longest_trainable_segment_frames"] != [48]:
        raise ValueError("minimum trainable segment axis must be exactly [48]")
    if any(
        not 0.0 <= float(value) <= 1.0
        for field in (
            "maximum_frame_joint_mask_flicker_fraction",
            "minimum_dual_path_agreement",
            "minimum_longest_trainable_segment_fraction",
            "minimum_source_coverage",
            "minimum_window_joint_support_fraction",
        )
        for value in axes[field]
    ):
        raise ValueError("fraction threshold axis is outside [0,1]")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 8
        for value in axes["minimum_window_stable_action_joints"]
    ):
        raise ValueError("stable-action-joint axis must contain integers in [1,8]")

    unsorted: list[dict[str, Any]] = []
    index_ranges = [range(len(axes[field])) for field in FROZEN_THRESHOLD_FIELD_ORDER]
    for indices in itertools.product(*index_ranges):
        severity = list(indices)
        thresholds = {
            field: axes[field][index]
            for field, index in zip(FROZEN_THRESHOLD_FIELD_ORDER, indices, strict=True)
        }
        unsorted.append(
            {
                "thresholds": thresholds,
                "severity_components": severity,
            }
        )
    ordered = sorted(
        unsorted,
        key=lambda row: (
            sum(row["severity_components"]),
            *row["severity_components"],
        ),
    )
    return [
        {"permissiveness_rank": rank, **row}
        for rank, row in enumerate(ordered)
    ]
