"""Pure preregistration contract for the v4e synthetic threshold grid."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

FROZEN_THRESHOLD_FIELD_ORDER = (
    "maximum_candidate_window_frames",
    "maximum_frame_center_step",
    "maximum_frame_joint_mask_flicker_fraction",
    "maximum_frame_log_scale_step",
    "minimum_dual_path_agreement",
    "minimum_frame_local_ambiguity_gap",
    "minimum_longest_trainable_segment_fraction",
    "minimum_longest_trainable_segment_frames",
    "minimum_source_coverage",
    "minimum_window_joint_support_fraction",
    "minimum_window_stable_action_joints",
)
_DESCENDING_AXES = {
    "maximum_frame_center_step",
    "maximum_frame_joint_mask_flicker_fraction",
    "maximum_frame_log_scale_step",
}
_ASCENDING_AXES = {
    "minimum_dual_path_agreement",
    "minimum_frame_local_ambiguity_gap",
    "minimum_longest_trainable_segment_fraction",
    "minimum_source_coverage",
    "minimum_window_joint_support_fraction",
    "minimum_window_stable_action_joints",
}


def canonical_threshold_grid_rows(
    axes_value: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate ordered axes and return the unique Cartesian severity order."""

    if tuple(axes_value) != FROZEN_THRESHOLD_FIELD_ORDER:
        raise ValueError("threshold axes are not in frozen field order")
    axes: dict[str, list[int | float]] = {}
    for field in FROZEN_THRESHOLD_FIELD_ORDER:
        raw = axes_value.get(field)
        if not isinstance(raw, list) or not raw:
            raise ValueError(f"threshold axis missing: {field}")
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in raw
        ):
            raise ValueError(f"threshold axis is non-numeric: {field}")
        if len({float(value) for value in raw}) != len(raw):
            raise ValueError(f"threshold axis contains duplicates: {field}")
        if field in _DESCENDING_AXES and any(
            float(left) <= float(right) for left, right in zip(raw, raw[1:])
        ):
            raise ValueError(f"maximum threshold axis is not permissive-to-strict: {field}")
        if field in _ASCENDING_AXES and any(
            float(left) >= float(right) for left, right in zip(raw, raw[1:])
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
