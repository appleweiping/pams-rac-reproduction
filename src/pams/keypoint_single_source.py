"""Single-source Keypoint R-CNN pose representation for v4e.

The v4d recovery cache spliced MediaPipe coordinates and Keypoint R-CNN
coordinates on one timeline.  This module intentionally does not consume that
cache.  It decodes the canonical source video again, selects one whole-video
Keypoint R-CNN path, and emits only that detector's COCO-17 observations.
MediaPipe v4a is read solely for an isolated post-selection shape diagnostic;
it has zero weight in both KPRCNN Viterbi paths and cannot block extraction.

Raw extraction is evidence production, never training authorization.  A
separate, sealed pilot/synthetic/full gate must classify each result as
eligible or quarantined before a bounded consumer may use it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pams.config import KeypointRCNNSingleSourceConfig
from pams.keypoint_recovery import (
    COCO17_TO_MEDIAPIPE33,
    KeypointRCNNRuntime,
)
from pams.pose import sha256_file
from pams.types import PoseSequence

V4E_PREPROCESSING_REVISION = (
    "official-segment-keypointrcnn-single-source-coco17-full-timeline-v4e"
)
V4E_RECOVERY_MODE = "keypointrcnn-single-source-kprcnn-only-path-v4e"
V4E_REPRESENTATION = "unified_2d"
V4E_REPRESENTATION_DETAIL = {
    "joints": "coco17",
    "coordinates": "body-centered-uniform-rms-scale-xy-z0-v1",
    "storage": "coco17-first17-zero-pad33-v1",
    "temporal": "native",
}
RELIABLE_VISIBILITY_MINIMUM = 0.1
RELIABLE_TORSO_JOINTS = (5, 6, 11, 12)
RELIABLE_ACTION_JOINTS = (7, 8, 9, 10, 13, 14, 15, 16)
MINIMUM_RELIABLE_JOINTS = 4
CYCLEBACK_PAIR_VARIANTS: dict[str, tuple[int, int]] = {
    "W16_H2": (16, 2),
    "W16_H4": (16, 4),
    "W16_H4_PE0": (16, 4),
    "W24_H4": (24, 4),
}
CYCLEBACK_PAIR_VARIANT_ALIASES = {"W16_H4_PE0": "W16_H4"}


class KeypointSingleSourceError(RuntimeError):
    """Raised when a v4e extraction or evidence invariant fails."""


@dataclass(frozen=True, slots=True)
class AssociationWeights:
    """One frozen Viterbi objective used for path stability comparison."""

    center: float
    log_scale: float
    shape: float
    anchor: float

    def __post_init__(self) -> None:
        values = (self.center, self.log_scale, self.shape, self.anchor)
        if any(not math.isfinite(float(value)) or value < 0.0 for value in values):
            raise ValueError("association weights must be finite and non-negative")
        if not any(value > 0.0 for value in values):
            raise ValueError("association weights must be non-degenerate")

    def to_dict(self) -> dict[str, float]:
        return {
            "anchor": float(self.anchor),
            "center": float(self.center),
            "log_scale": float(self.log_scale),
            "shape": float(self.shape),
        }


@dataclass(frozen=True, slots=True)
class ViterbiPath:
    """Best path, global runner-up score, and immutable evidence."""

    selected_indices: tuple[int | None, ...]
    score: float | None
    runner_up_score: float | None
    observed_frames: int
    selected_path_sha256: str

    @property
    def margin_per_observed_frame(self) -> float | None:
        if self.score is None or self.runner_up_score is None or self.observed_frames < 1:
            return None
        return float((self.score - self.runner_up_score) / self.observed_frames)


@dataclass(frozen=True, slots=True)
class CandidateEvidenceBundle:
    """Canonical ragged fresh-detector evidence persisted independently."""

    frame_offsets: NDArray[np.int64]
    candidates: NDArray[np.float32]


def candidate_evidence_bundle(
    frame_candidates: Sequence[NDArray[np.float32] | None],
) -> CandidateEvidenceBundle:
    offsets = np.zeros(len(frame_candidates) + 1, dtype=np.int64)
    rows: list[NDArray[np.float32]] = []
    for frame_index, value in enumerate(frame_candidates):
        if value is not None:
            array = np.ascontiguousarray(value, dtype=np.float32)
            if array.ndim != 3 or array.shape[1:] != (17, 6):
                raise ValueError("fresh candidate evidence must have shape [N,17,6]")
            if not np.isfinite(array).all():
                raise ValueError("fresh candidate evidence contains non-finite values")
            rows.append(array)
            offsets[frame_index + 1] = offsets[frame_index] + len(array)
        else:
            offsets[frame_index + 1] = offsets[frame_index]
    packed = (
        np.concatenate(rows, axis=0).astype(np.float32, copy=False)
        if rows
        else np.empty((0, 17, 6), dtype=np.float32)
    )
    return CandidateEvidenceBundle(np.ascontiguousarray(offsets), np.ascontiguousarray(packed))


def write_candidate_evidence_npz(
    path: str | Path,
    bundle: CandidateEvidenceBundle,
) -> None:
    """Write one immutable, independently reloadable candidate artifact."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(1, dtype=np.int64),
            frame_offsets=np.ascontiguousarray(bundle.frame_offsets, dtype=np.int64),
            candidates=np.ascontiguousarray(bundle.candidates, dtype=np.float32),
        )


def load_candidate_evidence_npz(path: str | Path) -> CandidateEvidenceBundle:
    """Fail closed on dtype, shape, offsets, finite values, and key schema."""

    with np.load(Path(path), allow_pickle=False) as payload:
        if set(payload.files) != {"schema_version", "frame_offsets", "candidates"}:
            raise ValueError("candidate evidence NPZ schema mismatch")
        schema = np.asarray(payload["schema_version"])
        offsets = np.asarray(payload["frame_offsets"])
        candidates = np.asarray(payload["candidates"])
    if schema.shape != () or schema.dtype != np.int64 or int(schema) != 1:
        raise ValueError("candidate evidence schema version mismatch")
    if offsets.dtype != np.int64 or offsets.ndim != 1 or len(offsets) < 1:
        raise ValueError("candidate frame offsets are invalid")
    if candidates.dtype != np.float32 or candidates.ndim != 3 or candidates.shape[1:] != (17, 6):
        raise ValueError("candidate packed array is invalid")
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(candidates):
        raise ValueError("candidate offsets do not cover packed candidates")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("candidate offsets are not monotonic")
    if np.any(offsets[1:] - offsets[:-1] > 100):
        raise ValueError("pre-truncation candidate count exceeds frozen detector top100")
    if not np.isfinite(candidates).all():
        raise ValueError("candidate evidence contains non-finite values")
    if len(candidates):
        if np.any(candidates[:, :, :2] < 0.0) or np.any(candidates[:, :, :2] > 1.0):
            raise ValueError("candidate XY is outside normalized image bounds")
        if not np.all(candidates[:, :, 2] == 0.0):
            raise ValueError("candidate reserved channel 2 is not exact zero")
        if np.any(candidates[:, :, 3] < 0.0) or np.any(candidates[:, :, 3] > 1.0):
            raise ValueError("candidate visibility is outside [0,1]")
        for candidate in candidates:
            box = float(candidate[0, 5])
            if not 0.2 <= box <= 1.0 or not np.all(candidate[:, 5] == candidate[0, 5]):
                raise ValueError("candidate box-score channel is invalid")
            confident = candidate[:, 4] > 2.0
            if int(np.count_nonzero(confident)) < 8:
                raise ValueError("candidate has fewer than eight reliable joints")
            if not all(bool(confident[index]) for index in RELIABLE_TORSO_JOINTS):
                raise ValueError("candidate lacks reliable torso")
            if int(np.count_nonzero(confident[list(RELIABLE_ACTION_JOINTS)])) < 4:
                raise ValueError("candidate lacks reliable action joints")
            expected_visibility = np.zeros(17, dtype=np.float32)
            expected_visibility[confident] = (
                1.0
                / (1.0 + np.exp(-candidate[confident, 4].astype(np.float64)))
                * box
            ).astype(np.float32)
            if not np.array_equal(candidate[:, 3], expected_visibility):
                raise ValueError("candidate visibility/logit/box channels disagree")
    return CandidateEvidenceBundle(
        np.ascontiguousarray(offsets, dtype=np.int64),
        np.ascontiguousarray(candidates, dtype=np.float32),
    )


@dataclass(frozen=True, slots=True)
class TrackStabilityThresholds:
    """Caller-supplied thresholds frozen outside the raw extractor.

    No defaults are provided deliberately.  The full337 runner must bind an
    authorization receipt created before the full run; it may not infer these
    values from full337 diagnostics.
    """

    minimum_frame_local_ambiguity_gap: float
    minimum_dual_path_agreement: float
    maximum_frame_center_step: float
    maximum_frame_log_scale_step: float
    maximum_frame_joint_mask_flicker_fraction: float
    minimum_source_coverage: float
    minimum_longest_trainable_segment_frames: int
    minimum_longest_trainable_segment_fraction: float
    maximum_candidate_window_frames: int
    minimum_window_joint_support_fraction: float
    minimum_window_stable_action_joints: int

    def __post_init__(self) -> None:
        bounded = (
            self.minimum_dual_path_agreement,
            self.maximum_frame_joint_mask_flicker_fraction,
            self.minimum_source_coverage,
            self.minimum_longest_trainable_segment_fraction,
            self.minimum_window_joint_support_fraction,
        )
        if any(not math.isfinite(float(value)) or not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("track-stability fractions must be in [0,1]")
        if any(
            not math.isfinite(float(value)) or value < 0.0
            for value in (self.maximum_frame_center_step, self.maximum_frame_log_scale_step)
        ):
            raise ValueError("track continuity thresholds must be finite and non-negative")
        if (
            not math.isfinite(float(self.minimum_frame_local_ambiguity_gap))
            or self.minimum_frame_local_ambiguity_gap < 0.0
        ):
            raise ValueError("minimum local ambiguity gap must be finite and non-negative")
        if self.minimum_longest_trainable_segment_frames < 1:
            raise ValueError("minimum trainable segment length must be positive")
        if self.maximum_candidate_window_frames != 24:
            raise ValueError("maximum candidate window must be frozen at 24 frames")
        if self.minimum_longest_trainable_segment_frames != 48:
            raise ValueError("minimum trainable segment must be frozen at 48 frames")
        if not 1 <= self.minimum_window_stable_action_joints <= len(RELIABLE_ACTION_JOINTS):
            raise ValueError("stable action joint threshold is invalid")


def effective_maximum_bridge_gap_frames(
    *,
    fps: float,
    maximum_seconds: float,
    frame_cap: int,
) -> int:
    """Apply the pre-registered physical gap rule without a one-frame floor."""

    if not math.isfinite(float(fps)) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    if not math.isfinite(float(maximum_seconds)) or maximum_seconds < 0.0:
        raise ValueError("maximum gap seconds must be finite and non-negative")
    if frame_cap < 0:
        raise ValueError("maximum gap frame cap must be non-negative")
    return min(int(frame_cap), int(math.floor(float(maximum_seconds) * float(fps))))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KeypointSingleSourceError(message)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def _infer_coco17_batch(
    runtime: KeypointRCNNRuntime,
    frames: Sequence[NDArray[np.uint8]],
    *,
    settings: KeypointRCNNSingleSourceConfig,
) -> tuple[
    tuple[NDArray[np.float32] | None, NDArray[np.float32] | None, dict[str, int]], ...
]:
    """Infer raw COCO-17 candidates while retaining logits and box score."""

    tensors = []
    dimensions: list[tuple[int, int]] = []
    for frame in frames:
        _require(frame.ndim == 3 and frame.shape[2] == 3, "decoded frame must be BGR")
        rgb = np.ascontiguousarray(frame[:, :, ::-1])
        tensor = runtime.torch.from_numpy(rgb).permute(2, 0, 1)
        tensors.append(tensor.to(runtime.device, dtype=runtime.torch.float32).div_(255.0))
        dimensions.append((int(frame.shape[1]), int(frame.shape[0])))
    with runtime.torch.inference_mode():
        outputs = runtime.model(tensors)
    _require(len(outputs) == len(frames), "detector batch output count mismatch")
    results: list[
        tuple[NDArray[np.float32] | None, NDArray[np.float32] | None, dict[str, int]]
    ] = []
    for output, (width, height) in zip(outputs, dimensions, strict=True):
        _require(width > 0 and height > 0, "decoded frame dimensions must be positive")
        _require(
            isinstance(output, Mapping)
            and {"labels", "scores", "keypoints", "keypoints_scores"}.issubset(output),
            "detector output schema mismatch",
        )
        labels = np.asarray(output["labels"].detach().cpu().numpy(), dtype=np.int64)
        scores = np.asarray(output["scores"].detach().cpu().numpy(), dtype=np.float32)
        keypoints = np.asarray(output["keypoints"].detach().cpu().numpy(), dtype=np.float32)
        logits = np.asarray(
            output["keypoints_scores"].detach().cpu().numpy(), dtype=np.float32
        )
        _require(labels.ndim == 1 and scores.shape == labels.shape, "detector score shape mismatch")
        _require(
            keypoints.shape == (len(labels), 17, 3)
            and logits.shape == (len(labels), 17),
            "detector keypoint shape mismatch",
        )
        _require(
            np.isfinite(scores).all()
            and np.isfinite(keypoints).all()
            and np.isfinite(logits).all(),
            "detector output contains non-finite values",
        )
        retained: list[tuple[float, bytes, NDArray[np.float32]]] = []
        counts = {
            "detector_output_total": len(labels),
            "rejected_non_person": 0,
            "rejected_low_box_score": 0,
            "rejected_low_total_joint_support": 0,
            "rejected_low_action_joint_support": 0,
            "rejected_unreliable_torso": 0,
            "eligible_before_top4": 0,
            "dropped_by_top4": 0,
        }
        for index, box_score_value in enumerate(scores):
            box_score = float(box_score_value)
            confident = logits[index] > settings.keypoint_logit_threshold
            if int(labels[index]) != settings.person_label:
                counts["rejected_non_person"] += 1
                continue
            if box_score < settings.box_score_threshold:
                counts["rejected_low_box_score"] += 1
                continue
            if int(np.count_nonzero(confident)) < settings.minimum_confident_keypoints:
                counts["rejected_low_total_joint_support"] += 1
                continue
            if (
                int(np.count_nonzero(confident[list(RELIABLE_ACTION_JOINTS)]))
                < settings.minimum_confident_action_keypoints
            ):
                counts["rejected_low_action_joint_support"] += 1
                continue
            if not all(bool(confident[joint]) for joint in RELIABLE_TORSO_JOINTS):
                counts["rejected_unreliable_torso"] += 1
                continue
            xy = np.array(keypoints[index, :, :2], dtype=np.float32, copy=True)
            xy[:, 0] = np.clip(xy[:, 0] / float(width), 0.0, 1.0)
            xy[:, 1] = np.clip(xy[:, 1] / float(height), 0.0, 1.0)
            visibility = (
                1.0 / (1.0 + np.exp(-logits[index].astype(np.float64)))
            ).astype(np.float32) * np.float32(box_score)
            visibility[~confident] = np.float32(0.0)
            candidate = np.zeros((17, 6), dtype=np.float32)
            candidate[:, :2] = xy
            candidate[:, 3] = visibility
            candidate[:, 4] = logits[index]
            candidate[:, 5] = np.float32(box_score)
            digest = hashlib.sha256(candidate.tobytes(order="C")).digest()
            retained.append((box_score, digest, candidate))
        retained.sort(key=lambda item: (-item[0], item[1]))
        counts["eligible_before_top4"] = len(retained)
        counts["dropped_by_top4"] = max(0, len(retained) - settings.maximum_candidates_per_frame)
        values = [item[2] for item in retained[: settings.maximum_candidates_per_frame]]
        all_values = [item[2] for item in retained]
        results.append(
            (
                None if not values else np.stack(values, axis=0).astype(np.float32),
                None
                if not all_values
                else np.stack(all_values, axis=0).astype(np.float32),
                counts,
            )
        )
    return tuple(results)


def mediapipe33_to_coco17_xy(xyz: NDArray[np.float32]) -> NDArray[np.float32]:
    """Project a MediaPipe frame into the exact shared 17-joint anchor space."""

    array = np.asarray(xyz, dtype=np.float32)
    if array.shape != (33, 3):
        raise ValueError("MediaPipe pose must have shape [33,3]")
    result = np.ascontiguousarray(array[COCO17_TO_MEDIAPIPE33, :2], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("MediaPipe anchor contains non-finite values")
    return result


def body_centered_uniform_scale_xy(
    xy: NDArray[np.float32],
    *,
    visibility: NDArray[np.float32] | None = None,
    epsilon: float = 1e-6,
) -> NDArray[np.float32]:
    """Normalize COCO-17 XY using one translation and one isotropic scale.

    The center is the hip midpoint (COCO joints 11 and 12).  The scalar is the
    unweighted root-mean-square radius of reliable joints about that center.
    X and Y share the same scalar; unreliable joints are stored as exact zero.
    """

    points = np.asarray(xy, dtype=np.float64)
    if points.shape != (17, 2) or not np.isfinite(points).all():
        raise ValueError("COCO-17 XY must be a finite [17,2] array")
    if not math.isfinite(float(epsilon)) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    if visibility is None:
        reliable = np.ones(17, dtype=np.bool_)
        weights = np.ones(17, dtype=np.float64)
    else:
        raw_visibility = np.asarray(visibility, dtype=np.float64)
        if raw_visibility.shape != (17,) or not np.isfinite(raw_visibility).all():
            raise ValueError("COCO-17 visibility must be a finite [17] array")
        reliable = raw_visibility >= RELIABLE_VISIBILITY_MINIMUM
        if int(np.count_nonzero(reliable)) < MINIMUM_RELIABLE_JOINTS or not all(
            bool(reliable[index]) for index in RELIABLE_TORSO_JOINTS
        ):
            raise ValueError("COCO-17 frame lacks reliable shoulders or hips")
        weights = reliable.astype(np.float64)
    center = 0.5 * (points[11] + points[12])
    centered = points - center
    squared_radius = np.sum(np.square(centered), axis=1)
    scale = float(np.sqrt(np.sum(weights * squared_radius) / np.sum(weights)))
    if not math.isfinite(scale) or scale <= epsilon:
        raise ValueError("COCO-17 frame has degenerate uniform body scale")
    normalized = np.asarray(centered / scale, dtype=np.float32)
    normalized[~reliable] = np.float32(0.0)
    return normalized


def coco17_xy_to_padded_pose(
    xy: NDArray[np.float32],
    *,
    visibility: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Store canonical COCO-17 XY in the encoder's fixed 33x3 cache envelope."""

    normalized = body_centered_uniform_scale_xy(xy, visibility=visibility)
    padded = np.zeros((33, 3), dtype=np.float32)
    padded[:17, :2] = normalized
    return padded


def similarity_procrustes_residual(
    source_xy: NDArray[np.float32],
    target_xy: NDArray[np.float32],
    *,
    source_visibility: NDArray[np.float32] | None = None,
) -> float:
    """Return reflection-free similarity-aligned RMS residual in shared space."""

    source = np.asarray(
        body_centered_uniform_scale_xy(source_xy, visibility=source_visibility),
        dtype=np.float64,
    )
    target = np.asarray(
        body_centered_uniform_scale_xy(target_xy, visibility=source_visibility),
        dtype=np.float64,
    )
    weights = (
        np.ones(17, dtype=np.float64)
        if source_visibility is None
        else np.where(
            np.asarray(source_visibility, dtype=np.float64) >= RELIABLE_VISIBILITY_MINIMUM,
            np.clip(np.asarray(source_visibility, dtype=np.float64), 0.0, 1.0),
            0.0,
        )
    )
    covariance = (source * weights[:, None]).T @ target
    left, _, right_t = np.linalg.svd(covariance, full_matrices=False)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    aligned = source @ rotation
    squared = np.sum(np.square(aligned - target), axis=1)
    residual = float(np.sqrt(np.sum(weights * squared) / np.sum(weights)))
    if not math.isfinite(residual) or residual < 0.0:
        raise KeypointSingleSourceError("Procrustes residual is not finite")
    return residual


def _candidate_center_scale(candidate: NDArray[np.float32]) -> tuple[NDArray[np.float64], float]:
    array = np.asarray(candidate, dtype=np.float32)
    reliable = (
        np.asarray(array[:, 4], dtype=np.float64) > 2.0
        if array.shape[1] >= 5
        else np.asarray(array[:, 3], dtype=np.float64) >= RELIABLE_VISIBILITY_MINIMUM
    )
    if int(np.count_nonzero(reliable)) < MINIMUM_RELIABLE_JOINTS or not all(
        bool(reliable[index]) for index in RELIABLE_TORSO_JOINTS
    ):
        raise KeypointSingleSourceError("candidate lacks reliable torso geometry")
    xy = np.asarray(array[reliable, :2], dtype=np.float64)
    minimum = np.min(xy, axis=0)
    maximum = np.max(xy, axis=0)
    center = 0.5 * (minimum + maximum)
    scale = max(float(np.linalg.norm(maximum - minimum)), 1e-6)
    return center, scale


def _candidate_dominance_score(candidate: NDArray[np.float32]) -> float:
    """Freeze the label-free dominant-subject extent x confidence prior."""

    array = np.asarray(candidate, dtype=np.float32)
    visibility = np.clip(np.asarray(array[:, 3], dtype=np.float64), 0.0, 1.0)
    reliable = (
        np.asarray(array[:, 4], dtype=np.float64) > 2.0
        if array.shape[1] >= 5
        else visibility >= RELIABLE_VISIBILITY_MINIMUM
    )
    if int(np.count_nonzero(reliable)) < MINIMUM_RELIABLE_JOINTS:
        raise KeypointSingleSourceError("candidate lacks enough reliable joints")
    xy = np.asarray(array[reliable, :2], dtype=np.float64)
    extent = np.max(xy, axis=0) - np.min(xy, axis=0)
    spatial_extent = max(
        float(extent[0] * extent[1]),
        float(np.linalg.norm(extent)),
        1e-9,
    )
    score = spatial_extent * float(np.mean(visibility[reliable]))
    return max(score, 1e-12)


def _transition_cost(
    previous: NDArray[np.float32],
    current: NDArray[np.float32],
    *,
    frame_gap: int,
    weights: AssociationWeights,
) -> float:
    previous_center, previous_scale = _candidate_center_scale(previous)
    current_center, current_scale = _candidate_center_scale(current)
    elapsed = max(int(frame_gap), 1)
    common_scale = max(0.5 * (previous_scale + current_scale), 1e-6)
    center = float(np.linalg.norm(current_center - previous_center)) / (common_scale * elapsed)
    log_scale = abs(math.log(current_scale / previous_scale)) / elapsed
    shape = 0.0
    if weights.shape > 0.0:
        previous_shape = body_centered_uniform_scale_xy(previous[:, :2])
        current_shape = body_centered_uniform_scale_xy(current[:, :2])
        shape = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(
                            np.asarray(current_shape, dtype=np.float64)
                            - np.asarray(previous_shape, dtype=np.float64)
                        ),
                        axis=1,
                    )
                )
            )
        ) / elapsed
    return weights.center * center + weights.log_scale * log_scale + weights.shape * shape


def _anchor_residuals(
    frame_candidates: Sequence[NDArray[np.float32] | None],
    base_sequence: PoseSequence,
) -> tuple[tuple[float, ...] | None, ...]:
    if len(frame_candidates) != base_sequence.num_frames:
        raise ValueError("candidate and MediaPipe timelines must match")
    rows: list[tuple[float, ...] | None] = []
    for index, candidates in enumerate(frame_candidates):
        if candidates is None or not bool(base_sequence.valid_mask[index]):
            rows.append(None)
            continue
        anchor = mediapipe33_to_coco17_xy(base_sequence.xyz[index])
        rows.append(
            tuple(
                similarity_procrustes_residual(
                    candidate[:, :2],
                    anchor,
                    source_visibility=candidate[:, 3],
                )
                for candidate in np.asarray(candidates, dtype=np.float32)
            )
        )
    return tuple(rows)


def _path_digest(indices: Sequence[int | None]) -> str:
    return _canonical_sha256([None if value is None else int(value) for value in indices])


def _unary_score(
    candidate: NDArray[np.float32],
    *,
    anchor_residual: float | None,
    weights: AssociationWeights,
) -> float:
    score = math.log(_candidate_dominance_score(candidate))
    if anchor_residual is not None:
        score -= weights.anchor * anchor_residual
    return score


def select_top2_viterbi_paths(
    frame_candidates: Sequence[NDArray[np.float32] | None],
    *,
    anchor_residuals: Sequence[tuple[float, ...] | None],
    weights: AssociationWeights,
) -> ViterbiPath:
    """Select the best path and exact global runner-up with deterministic ties."""

    if not frame_candidates:
        raise ValueError("candidate timeline must be non-empty")
    if len(anchor_residuals) != len(frame_candidates):
        raise ValueError("anchor residual timeline must match candidates")
    candidates = tuple(
        None if item is None else np.asarray(item, dtype=np.float32)
        for item in frame_candidates
    )
    observed = tuple(
        index for index, item in enumerate(candidates) if item is not None and len(item) > 0
    )
    empty = tuple(None for _ in candidates)
    if not observed:
        return ViterbiPath(empty, None, None, 0, _path_digest(empty))

    # Each node stores up to two (score, previous-candidate, previous-rank)
    # hypotheses.  Retaining two at every node is sufficient for the global
    # best and runner-up paths in this first-order DAG.
    histories: list[list[list[tuple[float, int, int]]]] = []
    first_index = observed[0]
    first = candidates[first_index]
    _require(first is not None, "observed candidate frame unexpectedly empty")
    first_anchors = anchor_residuals[first_index]
    first_nodes: list[list[tuple[float, int, int]]] = []
    for candidate_index, candidate in enumerate(first):
        unary = _unary_score(
            candidate,
            anchor_residual=(
                None if first_anchors is None else first_anchors[candidate_index]
            ),
            weights=weights,
        )
        first_nodes.append([(unary, -1, -1)])
    histories.append(first_nodes)

    for offset in range(1, len(observed)):
        previous_frame = observed[offset - 1]
        current_frame = observed[offset]
        previous = candidates[previous_frame]
        current = candidates[current_frame]
        _require(
            previous is not None and current is not None,
            "observed candidate frame unexpectedly empty",
        )
        current_anchors = anchor_residuals[current_frame]
        current_nodes: list[list[tuple[float, int, int]]] = []
        for current_index, current_candidate in enumerate(current):
            unary = _unary_score(
                current_candidate,
                anchor_residual=(
                    None if current_anchors is None else current_anchors[current_index]
                ),
                weights=weights,
            )
            hypotheses: list[tuple[float, int, int]] = []
            for previous_index, previous_candidate in enumerate(previous):
                transition = _transition_cost(
                    previous_candidate,
                    current_candidate,
                    frame_gap=current_frame - previous_frame,
                    weights=weights,
                )
                for previous_rank, (score, _, _) in enumerate(
                    histories[-1][previous_index]
                ):
                    hypotheses.append(
                        (score - transition + unary, previous_index, previous_rank)
                    )
            hypotheses.sort(key=lambda value: (-value[0], value[1], value[2]))
            current_nodes.append(hypotheses[:2])
        histories.append(current_nodes)

    terminals: list[tuple[float, int, int]] = []
    for candidate_index, hypotheses in enumerate(histories[-1]):
        for rank, (score, _, _) in enumerate(hypotheses):
            terminals.append((score, candidate_index, rank))
    terminals.sort(key=lambda value: (-value[0], value[1], value[2]))

    def reconstruct(terminal: tuple[float, int, int]) -> tuple[int | None, ...]:
        _, candidate_index, rank = terminal
        selected: list[int | None] = [None] * len(candidates)
        for offset in range(len(observed) - 1, -1, -1):
            selected[observed[offset]] = candidate_index
            _, parent_index, parent_rank = histories[offset][candidate_index][rank]
            candidate_index, rank = parent_index, parent_rank
        return tuple(selected)

    best = terminals[0]
    best_indices = reconstruct(best)
    runner_score = terminals[1][0] if len(terminals) > 1 else None
    return ViterbiPath(
        selected_indices=best_indices,
        score=float(best[0]),
        runner_up_score=None if runner_score is None else float(runner_score),
        observed_frames=len(observed),
        selected_path_sha256=_path_digest(best_indices),
    )


def _observed_segments(
    frame_candidates: Sequence[NDArray[np.float32] | None],
    *,
    maximum_bridge_gap_frames: int,
) -> tuple[tuple[int, ...], ...]:
    if maximum_bridge_gap_frames < 0:
        raise ValueError("maximum bridge gap must be non-negative")
    observed = [
        index
        for index, item in enumerate(frame_candidates)
        if item is not None and len(item) > 0
    ]
    if not observed:
        return ()
    segments: list[list[int]] = [[observed[0]]]
    for index in observed[1:]:
        missing_between = index - segments[-1][-1] - 1
        if missing_between > maximum_bridge_gap_frames:
            segments.append([index])
        else:
            segments[-1].append(index)
    return tuple(tuple(segment) for segment in segments)


def select_segmented_top2_viterbi_paths(
    frame_candidates: Sequence[NDArray[np.float32] | None],
    *,
    anchor_residuals: Sequence[tuple[float, ...] | None],
    weights: AssociationWeights,
    maximum_bridge_gap_frames: int,
) -> tuple[ViterbiPath, tuple[tuple[int, ...], ...]]:
    """Run independent Viterbi paths across pre-registered gap resets."""

    segments = _observed_segments(
        frame_candidates,
        maximum_bridge_gap_frames=maximum_bridge_gap_frames,
    )
    selected: list[int | None] = [None] * len(frame_candidates)
    total_score = 0.0
    observed_total = 0
    runner_deltas: list[float] = []
    for segment in segments:
        start, stop = segment[0], segment[-1] + 1
        result = select_top2_viterbi_paths(
            frame_candidates[start:stop],
            anchor_residuals=anchor_residuals[start:stop],
            weights=weights,
        )
        for relative, candidate_index in enumerate(result.selected_indices):
            if candidate_index is not None:
                selected[start + relative] = candidate_index
        if result.score is not None:
            total_score += result.score
        if result.score is not None and result.runner_up_score is not None:
            runner_deltas.append(result.score - result.runner_up_score)
        observed_total += result.observed_frames
    score = total_score if observed_total else None
    runner = (
        None
        if score is None or not runner_deltas
        else score - min(runner_deltas)
    )
    path = tuple(selected)
    return (
        ViterbiPath(path, score, runner, observed_total, _path_digest(path)),
        segments,
    )


def _local_ambiguity_gap_rows(
    frame_candidates: Sequence[NDArray[np.float32] | None],
    *,
    anchor_residuals: Sequence[tuple[float, ...] | None],
    weights: AssociationWeights,
    selected_path: ViterbiPath,
) -> tuple[tuple[int, float], ...]:
    """Return length-stable max-marginal gaps on ambiguous frames.

    For each frame with at least two candidates, this computes the score of
    the best complete path constrained to every candidate.  The reported gap
    is the selected path's max-marginal minus the best alternative candidate's
    max-marginal at that same frame.  Unlike a whole-path margin divided by
    video length, these fixed-frame gaps do not vanish merely because a video
    contains more unambiguous frames.
    """

    if len(frame_candidates) != len(anchor_residuals):
        raise ValueError("anchor residual timeline must match candidates")
    candidates = tuple(
        None if item is None else np.asarray(item, dtype=np.float32)
        for item in frame_candidates
    )
    observed = tuple(
        index for index, item in enumerate(candidates) if item is not None and len(item) > 0
    )
    if not observed:
        return ()

    forward: list[NDArray[np.float64]] = []
    for offset, frame_index in enumerate(observed):
        current = candidates[frame_index]
        _require(current is not None, "observed candidate frame unexpectedly empty")
        current_anchors = anchor_residuals[frame_index]
        unary = np.asarray(
            [
                _unary_score(
                    candidate,
                    anchor_residual=(
                        None if current_anchors is None else current_anchors[index]
                    ),
                    weights=weights,
                )
                for index, candidate in enumerate(current)
            ],
            dtype=np.float64,
        )
        if offset == 0:
            forward.append(unary)
            continue
        previous_frame = observed[offset - 1]
        previous = candidates[previous_frame]
        _require(previous is not None, "observed candidate frame unexpectedly empty")
        scores = np.empty(len(current), dtype=np.float64)
        for current_index, current_candidate in enumerate(current):
            scores[current_index] = unary[current_index] + max(
                forward[-1][previous_index]
                - _transition_cost(
                    previous_candidate,
                    current_candidate,
                    frame_gap=frame_index - previous_frame,
                    weights=weights,
                )
                for previous_index, previous_candidate in enumerate(previous)
            )
        forward.append(scores)

    backward: list[NDArray[np.float64]] = [
        np.empty(0, dtype=np.float64) for _ in observed
    ]
    last = candidates[observed[-1]]
    _require(last is not None, "observed candidate frame unexpectedly empty")
    backward[-1] = np.zeros(len(last), dtype=np.float64)
    for offset in range(len(observed) - 2, -1, -1):
        frame_index = observed[offset]
        next_frame = observed[offset + 1]
        current = candidates[frame_index]
        following = candidates[next_frame]
        _require(
            current is not None and following is not None,
            "observed candidate frame unexpectedly empty",
        )
        following_anchors = anchor_residuals[next_frame]
        following_unary = [
            _unary_score(
                candidate,
                anchor_residual=(
                    None if following_anchors is None else following_anchors[index]
                ),
                weights=weights,
            )
            for index, candidate in enumerate(following)
        ]
        scores = np.empty(len(current), dtype=np.float64)
        for current_index, current_candidate in enumerate(current):
            scores[current_index] = max(
                following_unary[next_index]
                + backward[offset + 1][next_index]
                - _transition_cost(
                    current_candidate,
                    next_candidate,
                    frame_gap=next_frame - frame_index,
                    weights=weights,
                )
                for next_index, next_candidate in enumerate(following)
            )
        backward[offset] = scores

    gaps: list[tuple[int, float]] = []
    for offset, frame_index in enumerate(observed):
        current = candidates[frame_index]
        _require(current is not None, "observed candidate frame unexpectedly empty")
        if len(current) < 2:
            continue
        selected_index = selected_path.selected_indices[frame_index]
        _require(selected_index is not None, "selected path misses an observed frame")
        max_marginals = forward[offset] + backward[offset]
        alternatives = [
            float(score)
            for index, score in enumerate(max_marginals)
            if index != selected_index
        ]
        gap = float(max_marginals[selected_index] - max(alternatives))
        _require(math.isfinite(gap), "local ambiguity gap is not finite")
        _require(gap >= -1e-9, "selected path is not the local max-marginal optimum")
        gaps.append((frame_index, max(gap, 0.0)))
    return tuple(gaps)


def local_ambiguity_gaps(
    frame_candidates: Sequence[NDArray[np.float32] | None],
    *,
    anchor_residuals: Sequence[tuple[float, ...] | None],
    weights: AssociationWeights,
    selected_path: ViterbiPath,
) -> tuple[float, ...]:
    """Return only the gap values; raw extraction retains indexed rows too."""

    return tuple(
        value
        for _, value in _local_ambiguity_gap_rows(
            frame_candidates,
            anchor_residuals=anchor_residuals,
            weights=weights,
            selected_path=selected_path,
        )
    )


def _candidate_evidence_sha256(
    frame_candidates: Sequence[NDArray[np.float32] | None],
) -> str:
    digest = hashlib.sha256()
    for frame_index, candidates in enumerate(frame_candidates):
        digest.update(int(frame_index).to_bytes(8, "little", signed=False))
        if candidates is None:
            digest.update((0).to_bytes(4, "little", signed=False))
            continue
        array = np.ascontiguousarray(candidates, dtype=np.float32)
        digest.update(int(len(array)).to_bytes(4, "little", signed=False))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _strict_contiguous_valid_segments(
    valid_mask: NDArray[np.bool_],
) -> tuple[tuple[int, int], ...]:
    """Return half-open runs; no missing frame or path reset may be crossed."""

    mask = np.asarray(valid_mask, dtype=np.bool_)
    if mask.ndim != 1:
        raise ValueError("valid mask must be one-dimensional")
    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if bool(value) and start is None:
            start = index
        elif not bool(value) and start is not None:
            ranges.append((start, index))
            start = None
    if start is not None:
        ranges.append((start, len(mask)))
    return tuple(ranges)


def _selected_continuity_rows(
    track: Sequence[NDArray[np.float32] | None],
    *,
    maximum_bridge_gap_frames: int,
) -> dict[int, dict[str, float | int]]:
    """Return reproducible selected-path center/scale transitions by right frame."""

    observed = [index for index, candidate in enumerate(track) if candidate is not None]
    rows: dict[int, dict[str, float | int]] = {}
    for left, right in zip(observed, observed[1:], strict=False):
        missing = right - left - 1
        if missing > maximum_bridge_gap_frames:
            continue
        previous = track[left]
        current = track[right]
        _require(previous is not None and current is not None, "selected track missing")
        previous_center, previous_scale = _candidate_center_scale(previous)
        current_center, current_scale = _candidate_center_scale(current)
        elapsed = max(right - left, 1)
        normalization = max(0.5 * (previous_scale + current_scale), 1e-6)
        rows[right] = {
            "previous_frame_index": left,
            "frame_gap": elapsed,
            "center_step_normalized_per_frame": float(
                np.linalg.norm(current_center - previous_center) / normalization / elapsed
            ),
            "absolute_log_scale_step_per_frame": float(
                abs(math.log(current_scale / previous_scale)) / elapsed
            ),
        }
    return rows


def _anchor_evidence_sha256(
    residuals: Sequence[tuple[float, ...] | None],
) -> str:
    canonical = [
        None if row is None else [float(value) for value in row]
        for row in residuals
    ]
    return _canonical_sha256(canonical)


def _selected_candidates(
    candidates: Sequence[NDArray[np.float32] | None],
    path: ViterbiPath,
) -> tuple[NDArray[np.float32] | None, ...]:
    selected: list[NDArray[np.float32] | None] = []
    for frame, candidate_index in zip(candidates, path.selected_indices, strict=True):
        if frame is None or candidate_index is None:
            selected.append(None)
        else:
            selected.append(np.asarray(frame[candidate_index], dtype=np.float32))
    return tuple(selected)


def _continuity_steps(
    track: Sequence[NDArray[np.float32] | None],
    *,
    maximum_bridge_gap_frames: int,
) -> tuple[float, ...]:
    observed = [index for index, candidate in enumerate(track) if candidate is not None]
    steps: list[float] = []
    for left, right in zip(observed, observed[1:], strict=False):
        if right - left - 1 > maximum_bridge_gap_frames:
            continue
        previous = track[left]
        current = track[right]
        _require(
            previous is not None and current is not None,
            "selected track unexpectedly misses an observed frame",
        )
        previous_center, previous_scale = _candidate_center_scale(previous)
        current_center, current_scale = _candidate_center_scale(current)
        elapsed = max(right - left, 1)
        normalization = max(0.5 * (previous_scale + current_scale), 1e-6)
        steps.append(float(np.linalg.norm(current_center - previous_center)) / normalization / elapsed)
    return tuple(steps)


def build_single_source_sequence(
    *,
    video_id: str,
    fps: float,
    source_frames: int,
    decoded_frames: int,
    candidates: Sequence[NDArray[np.float32] | None],
    primary_path: ViterbiPath,
) -> PoseSequence:
    """Create a fixed-envelope cache whose only nonzero joints are COCO-17."""

    if len(candidates) != source_frames or not 0 <= decoded_frames <= source_frames:
        raise ValueError("v4e candidate timeline does not match source frame accounting")
    selected = _selected_candidates(candidates, primary_path)
    xyz = np.zeros((source_frames, 33, 3), dtype=np.float32)
    mask = np.zeros(source_frames, dtype=np.bool_)
    for index, candidate in enumerate(selected):
        if index >= decoded_frames or candidate is None:
            continue
        try:
            xyz[index] = coco17_xy_to_padded_pose(
                candidate[:, :2],
                visibility=candidate[:, 3],
            )
        except ValueError:
            continue
        mask[index] = True
    return PoseSequence(video_id=video_id, fps=fps, xyz=xyz, valid_mask=mask)


def reconstruct_canonical_frame_evidence(
    *,
    candidates: Sequence[NDArray[np.float32] | None],
    primary: ViterbiPath,
    secondary: ViterbiPath,
    association_segments: Sequence[Sequence[int]],
    sequence: PoseSequence,
    ambiguity_by_frame: Mapping[int, float],
    maximum_bridge_gap_frames: int,
    detector_diagnostics: Sequence[Mapping[str, int] | None],
    keypoint_logit_threshold: float,
) -> list[dict[str, Any]]:
    """Rebuild every eligibility-relevant frame field from immutable inputs."""

    source_frames = sequence.num_frames
    if not (
        len(candidates)
        == len(primary.selected_indices)
        == len(secondary.selected_indices)
        == len(detector_diagnostics)
        == source_frames
    ):
        raise ValueError("canonical frame evidence timeline mismatch")
    selected = _selected_candidates(candidates, primary)
    selected_continuity = _selected_continuity_rows(
        selected,
        maximum_bridge_gap_frames=maximum_bridge_gap_frames,
    )
    association_segment_by_frame: dict[int, int] = {}
    for segment_id, segment in enumerate(association_segments):
        for frame_index in segment:
            association_segment_by_frame[int(frame_index)] = segment_id
    strict_segments = _strict_contiguous_valid_segments(sequence.valid_mask)
    strict_segment_by_frame: dict[int, int] = {}
    for segment_id, (start, stop) in enumerate(strict_segments):
        for frame_index in range(start, stop):
            strict_segment_by_frame[frame_index] = segment_id

    result: list[dict[str, Any]] = []
    for frame_index, frame_candidates in enumerate(candidates):
        candidate_rows: list[dict[str, Any]] = []
        if frame_candidates is not None:
            for candidate_index, candidate in enumerate(frame_candidates):
                center, scale = _candidate_center_scale(candidate)
                confident = candidate[:, 4] > keypoint_logit_threshold
                candidate_rows.append(
                    {
                        "candidate_index": candidate_index,
                        "box_score": float(candidate[0, 5]),
                        "raw_keypoint_logits": [float(value) for value in candidate[:, 4]],
                        "active_joint_count": int(np.count_nonzero(confident)),
                        "active_action_joint_count": int(
                            np.count_nonzero(confident[list(RELIABLE_ACTION_JOINTS)])
                        ),
                        "center_xy": [float(center[0]), float(center[1])],
                        "scale": float(scale),
                        "candidate_sha256": hashlib.sha256(
                            np.ascontiguousarray(candidate, dtype=np.float32).tobytes(order="C")
                        ).hexdigest(),
                    }
                )
        primary_index = primary.selected_indices[frame_index]
        secondary_index = secondary.selected_indices[frame_index]
        diagnostics = detector_diagnostics[frame_index]
        result.append(
            {
                "frame_index": frame_index,
                "candidate_count": len(candidate_rows),
                "candidates": candidate_rows,
                "primary_index": primary_index,
                "secondary_index": secondary_index,
                "dual_path_agrees": (
                    primary_index == secondary_index if primary_index is not None else None
                ),
                "ambiguous_max_marginal_gap": ambiguity_by_frame.get(frame_index),
                "continuity_from_previous": selected_continuity.get(frame_index),
                "association_segment_id": association_segment_by_frame.get(frame_index),
                "trainable_segment_id": strict_segment_by_frame.get(frame_index),
                "raw_normalization_eligible": bool(sequence.valid_mask[frame_index]),
                "detector_filter_counts": None if diagnostics is None else dict(diagnostics),
                "crowd_quarantined": bool(
                    diagnostics is not None and int(diagnostics["dropped_by_top4"]) > 0
                ),
                "final_identity_eligible": None,
            }
        )
    return result


def extract_single_source_video(
    video_path: str | Path,
    *,
    video_id: str,
    clip_start_frame: int,
    clip_end_frame: int,
    base_sequence: PoseSequence,
    base_decoded_frames: int,
    expected_video_sha256: str,
    runtime: KeypointRCNNRuntime,
    settings: KeypointRCNNSingleSourceConfig,
) -> tuple[PoseSequence, dict[str, Any], CandidateEvidenceBundle]:
    """Decode canonical video and emit a raw, non-authoritative v4e outcome."""

    source = Path(video_path).resolve(strict=True)
    before = source.stat()
    source_sha256_before = sha256_file(source)
    _require(source_sha256_before == expected_video_sha256, "source video SHA mismatch")
    source_frames = int(clip_end_frame) - int(clip_start_frame)
    _require(source_frames > 0, "v4e requires a non-empty official segment")
    _require(base_sequence.video_id == video_id, "v4a anchor identity mismatch")
    _require(base_sequence.num_frames == source_frames, "v4a anchor timeline mismatch")
    _require(0 <= base_decoded_frames <= source_frames, "sealed v4a decoded count invalid")
    _require(settings.single_source_full_track, "v4e single-source contract disabled")
    _require(not settings.v4d_candidate_cache_consumed, "v4d denied cache consumption forbidden")

    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise KeypointSingleSourceError(f"OpenCV could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    _require(math.isfinite(fps) and fps > 0.0, "video reports invalid FPS")
    _require(
        math.isclose(fps, float(base_sequence.fps), rel_tol=0.0, abs_tol=1e-6),
        "v4e decoder FPS differs from v4a anchor cache",
    )
    if clip_start_frame:
        seek_ok = bool(capture.set(cv2.CAP_PROP_POS_FRAMES, int(clip_start_frame)))
        positioned = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
        _require(
            seek_ok and math.isfinite(positioned) and abs(positioned - clip_start_frame) <= 0.5,
            "OpenCV could not seek to official clip start",
        )

    candidates: list[NDArray[np.float32] | None] = [None] * source_frames
    all_candidate_evidence: list[NDArray[np.float32] | None] = [None] * source_frames
    pending_frames: list[NDArray[np.uint8]] = []
    pending_indices: list[int] = []
    decoded_digest = hashlib.sha256()
    detector_rejected_unreliable_torso_total = 0
    detector_diagnostics: list[dict[str, int] | None] = [None] * source_frames

    def flush() -> None:
        nonlocal pending_frames, pending_indices, detector_rejected_unreliable_torso_total
        if not pending_frames:
            return
        outputs = _infer_coco17_batch(runtime, pending_frames, settings=settings)
        for index, (retained, all_retained, diagnostics) in zip(
            pending_indices, outputs, strict=True
        ):
            detector_rejected_unreliable_torso_total += diagnostics[
                "rejected_unreliable_torso"
            ]
            candidates[index] = retained
            all_candidate_evidence[index] = all_retained
            detector_diagnostics[index] = diagnostics
        pending_frames = []
        pending_indices = []

    decoded = 0
    try:
        while decoded < base_decoded_frames:
            positioned_before = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            expected_position = int(clip_start_frame) + decoded
            _require(
                math.isfinite(positioned_before)
                and abs(positioned_before - expected_position) <= 0.5,
                "decoder frame position drifted before read",
            )
            ok, frame = capture.read()
            _require(bool(ok) and frame is not None, "v4e decode ended before sealed v4a boundary")
            decoded_digest.update(np.asarray(frame.shape, dtype=np.int64).tobytes())
            decoded_digest.update(np.ascontiguousarray(frame).tobytes(order="C"))
            positioned_after = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            _require(
                math.isfinite(positioned_after)
                and abs(positioned_after - (expected_position + 1)) <= 0.5,
                "decoder frame position drifted after read",
            )
            pending_frames.append(frame)
            pending_indices.append(decoded)
            if len(pending_frames) == settings.inference_batch_size:
                flush()
            decoded += 1
        flush()
    finally:
        capture.release()
    _require(decoded == base_decoded_frames, "v4e decoded frame count mismatch")

    after = source.stat()
    source_sha256_after = sha256_file(source)
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    _require(source_sha256_after == source_sha256_before, "source video changed during decode")
    _require(
        all(getattr(before, field) == getattr(after, field) for field in identity_fields),
        "source video stat identity changed during decode",
    )

    # The processed v4a cache has lost image position.  Its shape-only
    # Procrustes residual is therefore not an identity anchor and must have no
    # control-flow influence over KPRCNN selection.  A diagnostic is attempted
    # only after both paths and the canonical cache have been constructed.
    no_anchors: tuple[tuple[float, ...] | None, ...] = tuple(
        None for _ in candidates
    )
    effective_bridge_gap = effective_maximum_bridge_gap_frames(
        fps=fps,
        maximum_seconds=settings.maximum_bridge_gap_seconds,
        frame_cap=settings.maximum_bridge_gap_frame_cap,
    )
    primary_weights = AssociationWeights(
        settings.primary_center_weight,
        settings.primary_log_scale_weight,
        settings.primary_shape_weight,
        settings.primary_anchor_weight,
    )
    secondary_weights = AssociationWeights(
        settings.secondary_center_weight,
        settings.secondary_log_scale_weight,
        settings.secondary_shape_weight,
        settings.secondary_anchor_weight,
    )
    primary, segments = select_segmented_top2_viterbi_paths(
        candidates,
        anchor_residuals=no_anchors,
        weights=primary_weights,
        maximum_bridge_gap_frames=effective_bridge_gap,
    )
    secondary, secondary_segments = select_segmented_top2_viterbi_paths(
        candidates,
        anchor_residuals=no_anchors,
        weights=secondary_weights,
        maximum_bridge_gap_frames=effective_bridge_gap,
    )
    _require(segments == secondary_segments, "dual Viterbi segment boundaries differ")
    ambiguity_gap_rows: list[tuple[int, float]] = []
    for segment in segments:
        start, stop = segment[0], segment[-1] + 1
        subpath = ViterbiPath(
            primary.selected_indices[start:stop],
            None,
            None,
            len(segment),
            _path_digest(primary.selected_indices[start:stop]),
        )
        ambiguity_gap_rows.extend(
            (start + relative_index, value)
            for relative_index, value in _local_ambiguity_gap_rows(
                candidates[start:stop],
                anchor_residuals=no_anchors[start:stop],
                weights=primary_weights,
                selected_path=subpath,
            )
        )
    ambiguity_gaps = tuple(value for _, value in ambiguity_gap_rows)
    ambiguity_by_frame = dict(ambiguity_gap_rows)
    sequence = build_single_source_sequence(
        video_id=video_id,
        fps=fps,
        source_frames=source_frames,
        decoded_frames=decoded,
        candidates=candidates,
        primary_path=primary,
    )
    selected = _selected_candidates(candidates, primary)
    selected_normalization_failure_frames = sum(
        index < decoded
        and candidate is not None
        and not bool(sequence.valid_mask[index])
        for index, candidate in enumerate(selected)
    )
    filter_eligible_normalization_failure_total = 0
    for frame_candidates in all_candidate_evidence[:decoded]:
        if frame_candidates is None:
            continue
        for candidate in frame_candidates:
            try:
                body_centered_uniform_scale_xy(
                    candidate[:, :2], visibility=candidate[:, 3]
                )
            except ValueError:
                filter_eligible_normalization_failure_total += 1
    continuity = _continuity_steps(
        selected,
        maximum_bridge_gap_frames=effective_bridge_gap,
    )
    comparable = [
        index
        for index, (left, right) in enumerate(
            zip(primary.selected_indices, secondary.selected_indices, strict=True)
        )
        if (
            left is not None
            and right is not None
            and candidates[index] is not None
            and len(candidates[index]) > 1
        )
    ]
    agreement = (
        sum(
            primary.selected_indices[index] == secondary.selected_indices[index]
            for index in comparable
        )
        / len(comparable)
        if comparable
        else (1.0 if primary.observed_frames > 0 else 0.0)
    )
    anchor_diagnostic_status = "unavailable"
    anchor_diagnostic_error: dict[str, str] | None = None
    anchors: tuple[tuple[float, ...] | None, ...] | None = None
    selected_anchor_residuals: list[float] = []
    try:
        anchors = _anchor_residuals(candidates, base_sequence)
        selected_anchor_residuals = [
            anchors[index][candidate_index]
            for index, candidate_index in enumerate(primary.selected_indices)
            if candidate_index is not None and anchors[index] is not None
        ]
        anchor_diagnostic_status = "available"
    except Exception as exc:  # diagnostic isolation: never blocks KPRCNN output
        anchor_diagnostic_error = {
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    candidate_counts = [0 if item is None else int(len(item)) for item in candidates]
    strict_segments = _strict_contiguous_valid_segments(sequence.valid_mask)
    frame_evidence = reconstruct_canonical_frame_evidence(
        candidates=candidates,
        primary=primary,
        secondary=secondary,
        association_segments=segments,
        sequence=sequence,
        ambiguity_by_frame=ambiguity_by_frame,
        maximum_bridge_gap_frames=effective_bridge_gap,
        detector_diagnostics=detector_diagnostics,
        keypoint_logit_threshold=settings.keypoint_logit_threshold,
    )
    audit: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_single_source_raw_video_evidence_v1",
        "preprocessing_revision": V4E_PREPROCESSING_REVISION,
        "recovery_mode": V4E_RECOVERY_MODE,
        "representation": V4E_REPRESENTATION,
        "representation_detail": dict(V4E_REPRESENTATION_DETAIL),
        "video_id_sha256": hashlib.sha256(video_id.encode("utf-8")).hexdigest(),
        "source_frames": source_frames,
        "decoded_frames": decoded,
        "padded_tail_frames": source_frames - decoded,
        "valid_frames": int(np.count_nonzero(sequence.valid_mask)),
        "source_coverage": float(np.count_nonzero(sequence.valid_mask) / source_frames),
        "candidate_observed_frames": sum(count > 0 for count in candidate_counts),
        "candidate_total": sum(candidate_counts),
        "candidate_maximum_per_frame": max(candidate_counts, default=0),
        "eligible_candidate_before_top4_total": sum(
            diagnostics["eligible_before_top4"]
            for diagnostics in detector_diagnostics
            if diagnostics is not None
        ),
        "candidate_dropped_by_top4_total": sum(
            diagnostics["dropped_by_top4"]
            for diagnostics in detector_diagnostics
            if diagnostics is not None
        ),
        "crowd_quarantined_frame_total": sum(
            diagnostics is not None and diagnostics["dropped_by_top4"] > 0
            for diagnostics in detector_diagnostics
        ),
        "detector_rejected_unreliable_torso_total": (
            detector_rejected_unreliable_torso_total
        ),
        "filter_eligible_normalization_failure_total": (
            filter_eligible_normalization_failure_total
        ),
        "selected_normalization_failure_frames": selected_normalization_failure_frames,
        "normalization_valid_mask_sha256": hashlib.sha256(
            bytes(int(value) for value in sequence.valid_mask)
        ).hexdigest(),
        "normalization_visibility_policy": (
            "raw-logit>2-at-least8-total-at-least4-of8-action-and-bilateral-shoulders-hips-v1"
        ),
        "raw_keypoint_logits_retained_in_candidate_evidence": True,
        "raw_box_scores_retained_in_candidate_evidence": True,
        "selected_top4_candidate_evidence_sha256": _candidate_evidence_sha256(candidates),
        "anchor_diagnostic_status": anchor_diagnostic_status,
        "anchor_diagnostic_error": anchor_diagnostic_error,
        "anchor_observed_frames": len(selected_anchor_residuals),
        "anchor_residual_median": _percentile(selected_anchor_residuals, 0.5),
        "anchor_residual_p90": _percentile(selected_anchor_residuals, 0.9),
        "anchor_evidence_sha256": (
            None if anchors is None else _anchor_evidence_sha256(anchors)
        ),
        "primary_weights": primary_weights.to_dict(),
        "secondary_weights": secondary_weights.to_dict(),
        "dominant_subject_unary_policy": (
            "log(max(area,diagonal)*mean-visible-confidence)-kprcnn-only-v1"
        ),
        "primary_path_score": primary.score,
        "primary_runner_up_score": primary.runner_up_score,
        "primary_global_margin_per_observed_frame": primary.margin_per_observed_frame,
        "primary_global_margin_scope": "diagnostic-only-length-biased-not-a-gate",
        "primary_runner_up_available": primary.runner_up_score is not None,
        "primary_path_sha256": primary.selected_path_sha256,
        "local_ambiguous_frame_total": len(ambiguity_gaps),
        "local_ambiguity_gap_p10": _percentile(ambiguity_gaps, 0.1),
        "local_ambiguity_gap_median": _percentile(ambiguity_gaps, 0.5),
        "secondary_path_sha256": secondary.selected_path_sha256,
        "dual_path_comparable_frames": len(comparable),
        "dual_path_agreement": float(agreement),
        "continuity_step_p95": _percentile(continuity, 0.95),
        "continuity_step_maximum": max(continuity, default=None),
        "maximum_bridge_gap_seconds": settings.maximum_bridge_gap_seconds,
        "maximum_bridge_gap_frame_cap": settings.maximum_bridge_gap_frame_cap,
        "effective_maximum_bridge_gap_frames": effective_bridge_gap,
        "track_segment_count": len(segments),
        "track_segment_scope": (
            "independent-pseudotracks-after-reset-no-cross-segment-identity-claim"
        ),
        "track_segment_ranges": [
            {"start": segment[0], "stop": segment[-1] + 1}
            for segment in segments
        ],
        "trainable_segments": [
            {"start": start, "stop": stop, "frames": stop - start}
            for start, stop in strict_segments
        ],
        "trainable_segment_policy": (
            "half-open-contiguous-valid-runs-no-missing-frame-or-reset-crossing-v1"
        ),
        "cycleback_pair_policy": "both-W-windows-contained-in-one-trainable-segment",
        "longest_trainable_segment_frames": max(
            (stop - start for start, stop in strict_segments), default=0
        ),
        "longest_track_segment_fraction": (
            max((len(segment) for segment in segments), default=0) / source_frames
        ),
        "longest_trainable_segment_fraction": (
            max((stop - start for start, stop in strict_segments), default=0)
            / source_frames
        ),
        "frame_evidence": frame_evidence,
        "frame_evidence_sha256": _canonical_sha256(frame_evidence),
        "decoder_frame_bytes_sha256": decoded_digest.hexdigest(),
        "source_video_sha256_before": source_sha256_before,
        "source_video_sha256_after": source_sha256_after,
        "source_video_stat_stable": True,
        "keypointrcnn_model_id": settings.model_id,
        "keypointrcnn_model_asset_sha256": settings.model_asset_sha256,
        "v4a_shape_anchor_pose_fingerprint": settings.base_pose_fingerprint,
        "v4a_anchor_evidence_scope": (
            "shape-only-processed-cache-not-image-position-not-target-identity-ground-truth"
        ),
        "v4d_denied_cache_consumed": False,
        "label_free": True,
        "raw_extraction_only": True,
        "cycleback_representation_input_authorized": False,
        "baseline_training_authorized": False,
    }
    audit["video_evidence_sha256"] = _canonical_sha256(audit)
    return sequence, audit, candidate_evidence_bundle(all_candidate_evidence)


def assess_track_stability(
    audit: Mapping[str, Any],
    *,
    thresholds: TrackStabilityThresholds,
    same_source_period_supported: bool,
) -> dict[str, Any]:
    """Classify one frozen raw result without consulting action/count labels.

    Period support is deliberately ignored and omitted from this artifact. It
    belongs to a later geometry gate and cannot alter even the serialized
    track-stability decision because detector jitter can create a strong
    short-lag peak. Every track remains explicitly marked as lacking a
    semantic target-identity claim.
    """

    del same_source_period_supported

    raw_frame_evidence = audit.get("frame_evidence")
    if not isinstance(raw_frame_evidence, Sequence):
        raise ValueError("raw frame evidence is required for framewise eligibility")
    identity_mask: list[bool] = []
    frame_reasons: list[list[str]] = []
    for expected_index, row in enumerate(raw_frame_evidence):
        if not isinstance(row, Mapping) or row.get("frame_index") != expected_index:
            raise ValueError("raw frame evidence must be ordered and contiguous")
        reasons: list[str] = []
        if row.get("raw_normalization_eligible") is not True:
            reasons.append("missing_or_normalization_rejected")
        if row.get("crowd_quarantined") is True:
            reasons.append("candidate_top4_truncation_crowd")
        candidate_count = int(row.get("candidate_count", 0))
        if candidate_count > 1:
            gap = row.get("ambiguous_max_marginal_gap")
            if gap is None or float(gap) < thresholds.minimum_frame_local_ambiguity_gap:
                reasons.append("local_ambiguity_gap")
            if row.get("dual_path_agrees") is not True:
                reasons.append("dual_path_disagreement")
        transition = row.get("continuity_from_previous")
        if transition is not None:
            if not isinstance(transition, Mapping):
                raise ValueError("continuity evidence must be an object")
            step = float(transition["center_step_normalized_per_frame"])
            if step > thresholds.maximum_frame_center_step:
                reasons.append("abnormal_center_boundary")
            log_scale_step = float(transition["absolute_log_scale_step_per_frame"])
            if log_scale_step > thresholds.maximum_frame_log_scale_step:
                reasons.append("abnormal_log_scale_boundary")
            previous_index = int(transition["previous_frame_index"])
            previous_row = raw_frame_evidence[previous_index]
            current_selected = row.get("primary_index")
            previous_selected = previous_row.get("primary_index")
            current_candidates = row.get("candidates")
            previous_candidates = previous_row.get("candidates")
            if (
                current_selected is None
                or previous_selected is None
                or not isinstance(current_candidates, Sequence)
                or not isinstance(previous_candidates, Sequence)
            ):
                reasons.append("joint_mask_flicker_unavailable")
            else:
                current_logits = current_candidates[int(current_selected)].get(
                    "raw_keypoint_logits"
                )
                previous_logits = previous_candidates[int(previous_selected)].get(
                    "raw_keypoint_logits"
                )
                if (
                    not isinstance(current_logits, Sequence)
                    or not isinstance(previous_logits, Sequence)
                    or len(current_logits) != 17
                    or len(previous_logits) != 17
                ):
                    reasons.append("joint_mask_flicker_unavailable")
                else:
                    current_mask = np.asarray(current_logits, dtype=np.float64) > 2.0
                    previous_mask = np.asarray(previous_logits, dtype=np.float64) > 2.0
                    flicker = float(np.mean(current_mask != previous_mask))
                    if flicker > thresholds.maximum_frame_joint_mask_flicker_fraction:
                        reasons.append("joint_mask_flicker")
        frame_reasons.append(reasons)
        identity_mask.append(not reasons)
    identity_segments = _strict_contiguous_valid_segments(
        np.asarray(identity_mask, dtype=np.bool_)
    )
    def base_pair_starts(window_frames: int, hop_frames: int) -> list[int]:
        span_frames = 2 * window_frames
        starts: list[int] = []
        for start, stop in identity_segments:
            if stop - start < span_frames:
                continue
            starts.extend(
                window_start
                for window_start in range(start, stop - span_frames + 1)
                if window_start % hop_frames == 0
            )
        return starts

    def supported_pair_starts(window_frames: int, hop_frames: int) -> list[int]:
        span_frames = 2 * window_frames
        starts: list[int] = []
        for window_start in base_pair_starts(window_frames, hop_frames):
            pair_supported = True
            joint_rows: list[list[bool]] = []
            for frame_index in range(window_start, window_start + span_frames):
                row = raw_frame_evidence[frame_index]
                selected_index = row.get("primary_index")
                candidates = row.get("candidates")
                if selected_index is None or not isinstance(candidates, Sequence):
                    pair_supported = False
                    break
                candidate = candidates[int(selected_index)]
                if not isinstance(candidate, Mapping):
                    pair_supported = False
                    break
                logits = candidate.get("raw_keypoint_logits")
                if not isinstance(logits, Sequence) or len(logits) != 17:
                    pair_supported = False
                    break
                joint_rows.append([float(value) > 2.0 for value in logits])
            if not pair_supported:
                continue
            support = np.mean(np.asarray(joint_rows, dtype=np.float64), axis=0)
            stable_action = int(
                np.count_nonzero(
                    support[list(RELIABLE_ACTION_JOINTS)]
                    >= thresholds.minimum_window_joint_support_fraction
                )
            )
            if stable_action < thresholds.minimum_window_stable_action_joints:
                continue
            starts.append(window_start)
        return starts

    base_variant_starts = {
        name: base_pair_starts(window_frames, hop_frames)
        for name, (window_frames, hop_frames) in CYCLEBACK_PAIR_VARIANTS.items()
    }
    variant_starts = {
        name: supported_pair_starts(window_frames, hop_frames)
        for name, (window_frames, hop_frames) in CYCLEBACK_PAIR_VARIANTS.items()
    }
    if base_variant_starts["W16_H4_PE0"] != base_variant_starts["W16_H4"]:
        raise KeypointSingleSourceError("PE0 base-start alias diverged from W16_H4")
    if variant_starts["W16_H4_PE0"] != variant_starts["W16_H4"]:
        raise KeypointSingleSourceError("PE0 eligible-start alias diverged from W16_H4")
    maximum_window_starts = variant_starts["W24_H4"]
    span_frames = 2 * thresholds.maximum_candidate_window_frames
    eligible_pair_spans = [
        (start, start + span_frames) for start in maximum_window_starts
    ]
    eligible_frames = sum(identity_mask)
    source_frames = int(audit["source_frames"])
    if source_frames != len(identity_mask):
        raise ValueError("frame evidence length does not match source frames")
    identity_coverage = eligible_frames / source_frames
    longest_segment_frames = max(
        (stop - start for start, stop in identity_segments), default=0
    )
    longest_segment_fraction = longest_segment_frames / source_frames
    agreement_passed = (
        float(audit["dual_path_agreement"]) >= thresholds.minimum_dual_path_agreement
    )
    coverage_passed = identity_coverage >= thresholds.minimum_source_coverage
    segment_frames_passed = (
        longest_segment_frames >= thresholds.minimum_longest_trainable_segment_frames
    )
    segment_fraction_passed = (
        longest_segment_fraction >= thresholds.minimum_longest_trainable_segment_fraction
    )
    window_support_passed = bool(eligible_pair_spans)
    eligible = bool(
        agreement_passed
        and coverage_passed
        and segment_frames_passed
        and segment_fraction_passed
        and window_support_passed
    )
    reasons: list[str] = []
    if not coverage_passed:
        reasons.append("identity_eligible_coverage")
    if not agreement_passed:
        reasons.append("dual_path_agreement")
    if not segment_frames_passed:
        reasons.append("minimum_contiguous_trainable_segment_frames")
    if not segment_fraction_passed:
        reasons.append("minimum_contiguous_trainable_segment_fraction")
    if not window_support_passed:
        reasons.append("window_joint_support")
    return {
        "schema_version": 1,
        "artifact_type": "pams_pose_recovery_v4e_track_stability_decision_v1",
        "video_id_sha256": audit["video_id_sha256"],
        "video_evidence_sha256": audit["video_evidence_sha256"],
        "label_free": True,
        "eligible": eligible,
        "quarantined": not eligible,
        "mediapipe_shape_evidence_scope": "diagnostic-only-not-used-for-eligibility",
        "identity_claim": "kprcnn-stability-only-no-target-identity-ground-truth",
        "period_scope": "diagnostic-only-separate-from-identity-and-trainability",
        "frame_identity_eligible_mask": identity_mask,
        "frame_identity_eligible_mask_sha256": hashlib.sha256(
            bytes(int(value) for value in identity_mask)
        ).hexdigest(),
        "frame_quarantine_reasons": frame_reasons,
        "eligible_frames": eligible_frames,
        "identity_eligible_coverage": identity_coverage,
        "track_stability_frame_ranges": [
            {"start": start, "stop": stop, "frames": stop - start}
            for start, stop in identity_segments
        ],
        "eligible_pair_spans": [
            {"start": start, "stop": stop, "frames": stop - start}
            for start, stop in eligible_pair_spans
        ],
        "eligible_pair_span_frames": span_frames,
        "eligible_pair_start_count": len(eligible_pair_spans),
        "pair_span_semantics": "exact-only-no-consumer-expansion",
        "pair_start_grid": "native-zero-based-start-mod-hop-equals-zero",
        "pair_variant_aliases": dict(CYCLEBACK_PAIR_VARIANT_ALIASES),
        "base_valid_pair_starts_by_variant": base_variant_starts,
        "eligible_pair_starts_by_variant": variant_starts,
        "longest_eligible_trainable_segment_frames": longest_segment_frames,
        "longest_eligible_trainable_segment_fraction": longest_segment_fraction,
        "criteria": {
            "coverage": coverage_passed,
            "dual_path_agreement": agreement_passed,
            "minimum_contiguous_trainable_segment_frames": segment_frames_passed,
            "minimum_contiguous_trainable_segment_fraction": segment_fraction_passed,
            "window_joint_support": window_support_passed,
            "framewise_identity_mask_computed": True,
        },
        "quarantine_reasons": reasons if not eligible else [],
        "thresholds_sha256": _canonical_sha256(
            {
                "maximum_frame_center_step": thresholds.maximum_frame_center_step,
                "maximum_frame_log_scale_step": thresholds.maximum_frame_log_scale_step,
                "maximum_frame_joint_mask_flicker_fraction": (
                    thresholds.maximum_frame_joint_mask_flicker_fraction
                ),
                "minimum_dual_path_agreement": thresholds.minimum_dual_path_agreement,
                "minimum_frame_local_ambiguity_gap": thresholds.minimum_frame_local_ambiguity_gap,
                "minimum_longest_trainable_segment_fraction": (
                    thresholds.minimum_longest_trainable_segment_fraction
                ),
                "minimum_longest_trainable_segment_frames": (
                    thresholds.minimum_longest_trainable_segment_frames
                ),
                "minimum_source_coverage": thresholds.minimum_source_coverage,
                "maximum_candidate_window_frames": thresholds.maximum_candidate_window_frames,
                "minimum_window_joint_support_fraction": (
                    thresholds.minimum_window_joint_support_fraction
                ),
                "minimum_window_stable_action_joints": (
                    thresholds.minimum_window_stable_action_joints
                ),
            }
        ),
        "raw_extraction_authority_accepted": False,
        "baseline_training_authorized": False,
    }
