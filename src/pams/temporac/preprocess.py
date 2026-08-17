"""COCO-17 preprocessing and v4 geometry/channel construction.

All certification arithmetic is binary64.  The only rounding performed here
is the contract's final duplicate-collapse rounding and the model-input
float32 materialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pams.temporac.quadrature import neumaier_sum

Float32Array = NDArray[np.float32]
Float64Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
Int32Array = NDArray[np.int32]
Int64Array = NDArray[np.int64]


JOINT_COUNT: Final = 17
CONFIDENCE_THRESHOLD: Final = np.float64(0.20)
MIN_COMMON_JOINTS: Final = 8
MIN_FRAME_JOINTS: Final = 8
MIN_SAMPLES: Final = 17
P_MIN: Final = 5
DIRECTION_NORM_FLOOR: Final = np.float64(1e-8)
HIPS: Final = (11, 12)
SHOULDERS: Final = (5, 6)
BONES: Final[tuple[tuple[int, int], ...]] = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
    (0, 1),
    (0, 2),
)


class PreprocessError(ValueError):
    """Fail-closed preprocessing error carrying the v4 reason name."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(f"{reason}: {message}")
        self.reason = reason


def _readonly(*arrays: NDArray[np.generic]) -> None:
    for array in arrays:
        array.setflags(write=False)


@dataclass(frozen=True, slots=True)
class CollapsedTrack:
    """Duplicate-collapsed, root-centred identity track before trimming."""

    motion: Float32Array
    clocks: Int64Array
    joint_mask: BoolArray
    frame_valid: BoolArray
    scale: float
    source_groups: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        _readonly(self.motion, self.clocks, self.joint_mask, self.frame_valid)


@dataclass(frozen=True, slots=True)
class GeometryFeatures:
    """Exact v4 geometry plus the 215- and 269-channel model inputs."""

    pose: Float64Array
    bones: Float64Array
    geometry: Float64Array
    element_mask: BoolArray
    coordinate_mask: BoolArray
    edge_coordinate_mask: BoolArray
    edge_lengths: Float64Array
    direction: Float64Array
    direction_element_mask: BoolArray
    teacher_input: Float32Array
    response_input: Float32Array
    run_bounds: Int32Array

    def __post_init__(self) -> None:
        _readonly(
            self.pose,
            self.bones,
            self.geometry,
            self.element_mask,
            self.coordinate_mask,
            self.edge_coordinate_mask,
            self.edge_lengths,
            self.direction,
            self.direction_element_mask,
            self.teacher_input,
            self.response_input,
            self.run_bounds,
        )


@dataclass(frozen=True, slots=True)
class PreprocessedIdentity:
    """A feature-eligible natural identity and its retained clock span."""

    motion: Float32Array
    clocks: Int64Array
    joint_mask: BoolArray
    frame_valid: BoolArray
    scale: float
    trim_bounds: tuple[int, int]
    geometry_features: GeometryFeatures

    def __post_init__(self) -> None:
        _readonly(self.motion, self.clocks, self.joint_mask, self.frame_valid)

    @property
    def run_bounds(self) -> Int32Array:
        return self.geometry_features.run_bounds

    @property
    def teacher_input(self) -> Float32Array:
        return self.geometry_features.teacher_input

    @property
    def response_input(self) -> Float32Array:
        return self.geometry_features.response_input

    @property
    def edge_lengths(self) -> Float64Array:
        return self.geometry_features.edge_lengths


def _validate_source_arrays(
    motion: ArrayLike,
    frame_mask: ArrayLike,
    clocks: ArrayLike,
) -> tuple[NDArray[np.generic], BoolArray, Int64Array]:
    raw_motion = np.asarray(motion)
    raw_frame_mask = np.asarray(frame_mask)
    raw_clocks = np.asarray(clocks)
    if raw_motion.ndim != 3 or raw_motion.shape[1:] != (JOINT_COUNT, 3):
        raise PreprocessError("SOURCE_SCHEMA", "motion must have shape [T,17,3]")
    sample_count = raw_motion.shape[0]
    if raw_frame_mask.shape != (sample_count,):
        raise PreprocessError("SOURCE_SCHEMA", "frame_mask must have shape [T]")
    if raw_frame_mask.dtype.kind not in "bu" or not np.all(
        (raw_frame_mask == 0) | (raw_frame_mask == 1)
    ):
        raise PreprocessError("SOURCE_SCHEMA", "frame_mask must be binary")
    if raw_clocks.shape != (sample_count,) or raw_clocks.dtype.kind not in "iu":
        raise PreprocessError("SOURCE_SCHEMA", "sampled_frame_indices must be integer shape [T]")
    if sample_count == 0:
        raise PreprocessError("SOURCE_SCHEMA", "empty source track")
    return (
        raw_motion,
        np.asarray(raw_frame_mask, dtype=np.bool_),
        np.asarray(raw_clocks, dtype=np.int64),
    )


def _raw_joint_state(
    motion: NDArray[np.generic], frame_mask: BoolArray
) -> tuple[Float64Array, Float64Array, BoolArray]:
    values = np.asarray(motion, dtype=np.float64)
    finite = np.all(np.isfinite(values), axis=2)
    clipped_confidence = np.clip(values[:, :, 2], 0.0, 1.0)
    joint_mask = frame_mask[:, None] & finite & (clipped_confidence > CONFIDENCE_THRESHOLD)
    coordinates = np.zeros((values.shape[0], JOINT_COUNT, 2), dtype=np.float64)
    confidence = np.zeros((values.shape[0], JOINT_COUNT), dtype=np.float64)
    coordinates[joint_mask] = values[:, :, :2][joint_mask]
    confidence[joint_mask] = clipped_confidence[joint_mask]
    return coordinates, confidence, joint_mask


def _roots(coordinates: Float64Array, joint_mask: BoolArray) -> tuple[Float64Array, BoolArray]:
    roots = np.zeros((coordinates.shape[0], 2), dtype=np.float64)
    root_valid = joint_mask[:, HIPS[0]] | joint_mask[:, HIPS[1]]
    for index in np.flatnonzero(root_valid):
        valid_hips = [hip for hip in HIPS if joint_mask[index, hip]]
        roots[index] = np.mean(coordinates[index, valid_hips], axis=0, dtype=np.float64)
    return roots, root_valid


def _clock_groups(clocks: Int64Array) -> tuple[tuple[int, int], ...]:
    if clocks.size == 0:
        raise PreprocessError("CLOCK_ORDER", "clock sequence is empty")
    differences = np.diff(clocks)
    if np.any(differences < 0):
        raise PreprocessError("CLOCK_ORDER", "clocks decrease or a clock reappears noncontiguously")
    starts = np.concatenate((np.array([0], dtype=np.int64), np.flatnonzero(differences != 0) + 1))
    stops = np.concatenate((starts[1:], np.array([clocks.size], dtype=np.int64)))
    groups = tuple((int(start), int(stop)) for start, stop in zip(starts, stops, strict=True))
    if not groups or any(start >= stop for start, stop in groups):
        raise PreprocessError("CLOCK_ORDER", "empty duplicate clock group")
    return groups


def _track_scale(
    coordinates: Float64Array,
    joint_mask: BoolArray,
    roots: Float64Array,
    root_valid: BoolArray,
    groups: tuple[tuple[int, int], ...],
) -> float:
    distances: list[float] = []
    for start, _ in groups:
        valid_shoulders = [joint for joint in SHOULDERS if joint_mask[start, joint]]
        if not root_valid[start] or not valid_shoulders:
            continue
        shoulder_mean = np.mean(coordinates[start, valid_shoulders], axis=0, dtype=np.float64)
        distance = float(np.linalg.norm(shoulder_mean - roots[start]))
        if np.isfinite(distance) and distance > 0.0:
            distances.append(distance)
    if not distances:
        raise PreprocessError("NORMALIZATION", "no positive root-to-shoulder scale")
    scale = max(1e-3, float(np.median(np.asarray(distances, dtype=np.float64))))
    if not np.isfinite(scale) or scale <= 0.0:
        raise PreprocessError("NORMALIZATION", "track scale is nonfinite or nonpositive")
    return scale


def _pair_agrees(
    a: int,
    b: int,
    coordinates: Float64Array,
    confidence: Float64Array,
    joint_mask: BoolArray,
    roots: Float64Array,
    root_valid: BoolArray,
    scale: float,
) -> bool:
    if not np.array_equal(joint_mask[a], joint_mask[b]):
        return False
    if bool(root_valid[a]) != bool(root_valid[b]) or not root_valid[a]:
        return False
    common = joint_mask[a] & joint_mask[b]
    if int(np.count_nonzero(common)) < MIN_COMMON_JOINTS:
        return False
    if float(np.max(np.abs(confidence[a] - confidence[b]))) > 1e-6:
        return False
    normalized_a = (coordinates[a] - roots[a]) / np.float64(scale)
    normalized_b = (coordinates[b] - roots[b]) / np.float64(scale)
    weights = np.minimum(confidence[a], confidence[b])
    indices = np.flatnonzero(common)
    denominator = neumaier_sum(float(weights[joint]) for joint in indices)
    numerator = neumaier_sum(
        float(
            weights[joint]
            * np.dot(
                normalized_a[joint] - normalized_b[joint], normalized_a[joint] - normalized_b[joint]
            )
        )
        for joint in indices
    )
    if denominator <= 0.0 or not np.isfinite(denominator) or not np.isfinite(numerator):
        return False
    distance = float(np.sqrt(np.float64(numerator / denominator)))
    return np.isfinite(distance) and distance <= 0.02


def collapse_duplicate_clocks(
    motion: ArrayLike,
    frame_mask: ArrayLike,
    sampled_frame_indices: ArrayLike,
) -> CollapsedTrack:
    """Validate and collapse contiguous equal-clock COCO-17 samples."""

    raw_motion, valid_frames, clocks = _validate_source_arrays(
        motion, frame_mask, sampled_frame_indices
    )
    coordinates, confidence, joint_mask = _raw_joint_state(raw_motion, valid_frames)
    roots, root_valid = _roots(coordinates, joint_mask)
    groups = _clock_groups(clocks)
    scale = _track_scale(coordinates, joint_mask, roots, root_valid, groups)

    collapsed_raw = np.zeros((len(groups), JOINT_COUNT, 3), dtype=np.float64)
    collapsed_joint_mask = np.zeros((len(groups), JOINT_COUNT), dtype=np.bool_)
    collapsed_clocks = np.empty(len(groups), dtype=np.int64)
    for output_index, (start, stop) in enumerate(groups):
        collapsed_clocks[output_index] = clocks[start]
        if stop - start > 1:
            if not np.all(root_valid[start:stop]):
                raise PreprocessError(
                    "NORMALIZATION",
                    f"duplicate clock {int(clocks[start])} has no valid root",
                )
            for a in range(start, stop):
                for b in range(a + 1, stop):
                    if not _pair_agrees(
                        a,
                        b,
                        coordinates,
                        confidence,
                        joint_mask,
                        roots,
                        root_valid,
                        scale,
                    ):
                        raise PreprocessError(
                            "DUPLICATE_CONFLICT",
                            f"duplicate clock {int(clocks[start])} does not agree",
                        )
        group_mask = joint_mask[start].copy()
        if not all(np.array_equal(group_mask, joint_mask[index]) for index in range(start, stop)):
            raise PreprocessError("DUPLICATE_CONFLICT", "duplicate joint-valid bits differ")
        collapsed_joint_mask[output_index] = group_mask
        for joint in np.flatnonzero(group_mask):
            weights = confidence[start:stop, joint]
            denominator = neumaier_sum(float(weight) for weight in weights)
            if denominator <= 0.0 or not np.isfinite(denominator):
                raise PreprocessError("DUPLICATE_CONFLICT", "duplicate confidence sum is invalid")
            for coordinate in range(2):
                numerator = neumaier_sum(
                    float(confidence[index, joint] * coordinates[index, joint, coordinate])
                    for index in range(start, stop)
                )
                collapsed_raw[output_index, joint, coordinate] = numerator / denominator
            collapsed_raw[output_index, joint, 2] = float(np.max(confidence[start:stop, joint]))

    # F1 aggregation has one explicit little-endian f4 materialization before
    # the post-collapse root normalization.
    collapsed_rounded = collapsed_raw.astype(np.dtype("<f4")).astype(np.float64)
    collapsed_roots, collapsed_root_valid = _roots(
        collapsed_rounded[:, :, :2], collapsed_joint_mask
    )
    normalized = np.zeros_like(collapsed_raw)
    for index in range(len(groups)):
        if not collapsed_root_valid[index]:
            # No normalized coordinate exists without a root.  It remains an
            # invalid frame eligible only for leading/trailing trimming.
            collapsed_joint_mask[index] = False
            continue
        normalized[index, :, :2][collapsed_joint_mask[index]] = (
            collapsed_rounded[index, :, :2][collapsed_joint_mask[index]] - collapsed_roots[index]
        ) / np.float64(scale)
        normalized[index, :, 2][collapsed_joint_mask[index]] = collapsed_rounded[index, :, 2][
            collapsed_joint_mask[index]
        ]

    frame_valid = (np.count_nonzero(collapsed_joint_mask, axis=1) >= MIN_FRAME_JOINTS) & (
        collapsed_joint_mask[:, HIPS[0]] | collapsed_joint_mask[:, HIPS[1]]
    )
    normalized_f32 = np.ascontiguousarray(normalized.astype(np.dtype("<f4")))
    # Re-canonicalize every invalid payload after the sole f4 rounding.
    normalized_f32[~collapsed_joint_mask] = np.float32(0.0)
    return CollapsedTrack(
        motion=normalized_f32,
        clocks=np.ascontiguousarray(collapsed_clocks, dtype=np.int64),
        joint_mask=np.ascontiguousarray(collapsed_joint_mask),
        frame_valid=np.ascontiguousarray(frame_valid),
        scale=scale,
        source_groups=groups,
    )


def validate_run_bounds(run_bounds: ArrayLike, edge_count: int) -> Int32Array:
    """Validate ordered, nonadjacent half-open edge runs."""

    raw = np.asarray(run_bounds)
    if raw.ndim != 2 or raw.shape[1:] != (2,) or raw.dtype.kind not in "iu":
        raise PreprocessError("EDGE_SUPPORT", "run_bounds must be integer shape [R,2]")
    bounds = np.asarray(raw, dtype=np.int64)
    previous_stop = -1
    for start, stop in bounds.tolist():
        if not (0 <= start < stop <= edge_count):
            raise PreprocessError("EDGE_SUPPORT", "run bound is empty or out of range")
        if start <= previous_stop:
            raise PreprocessError("EDGE_SUPPORT", "different runs must be ordered and nonadjacent")
        previous_stop = stop
    return np.ascontiguousarray(bounds, dtype=np.int32)


def mask_to_half_open_runs(mask: ArrayLike) -> Int32Array:
    """Convert a binary edge mask to maximal half-open true runs."""

    raw = np.asarray(mask)
    if raw.ndim != 1 or raw.dtype.kind not in "bu" or not np.all((raw == 0) | (raw == 1)):
        raise PreprocessError("EDGE_SUPPORT", "run mask must be a one-dimensional binary array")
    value = np.asarray(raw, dtype=np.int8)
    padded = np.pad(value, (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return np.ascontiguousarray(np.stack((starts, stops), axis=1), dtype=np.int32)


def _geometry_masks(joint_mask: BoolArray) -> tuple[BoolArray, BoolArray]:
    bone_mask = np.empty((joint_mask.shape[0], len(BONES)), dtype=np.bool_)
    for index, (source, target) in enumerate(BONES):
        bone_mask[:, index] = joint_mask[:, source] & joint_mask[:, target]
    element_mask = np.concatenate((joint_mask, bone_mask), axis=1)
    coordinate_mask = np.repeat(element_mask, 2, axis=1)
    return np.ascontiguousarray(element_mask), np.ascontiguousarray(coordinate_mask)


def _masked_norm(vector: Float64Array, mask: BoolArray) -> float:
    value = neumaier_sum(float(item * item) for item in vector[mask])
    if not np.isfinite(value) or value < 0.0:
        return float("nan")
    return float(np.sqrt(np.float64(value)))


def build_geometry_features(
    normalized_motion: ArrayLike,
    joint_mask: ArrayLike,
    run_bounds: ArrayLike,
) -> GeometryFeatures:
    """Build F4 geometry, 215 teacher channels, and 269 response channels."""

    motion = np.asarray(normalized_motion, dtype=np.float64)
    raw_mask = np.asarray(joint_mask)
    if motion.ndim != 3 or motion.shape[1:] != (JOINT_COUNT, 3):
        raise PreprocessError("SOURCE_SCHEMA", "normalized_motion must be [T,17,3]")
    sample_count = motion.shape[0]
    edge_count = sample_count - 1
    if sample_count < 2:
        raise PreprocessError("TOO_SHORT", "geometry requires at least two samples")
    if raw_mask.shape != (sample_count, JOINT_COUNT) or raw_mask.dtype.kind not in "bu":
        raise PreprocessError("SOURCE_SCHEMA", "joint_mask must be binary [T,17]")
    if not np.all((raw_mask == 0) | (raw_mask == 1)):
        raise PreprocessError("SOURCE_SCHEMA", "joint_mask must be binary")
    mask = np.asarray(raw_mask, dtype=np.bool_)
    if not np.all(np.isfinite(motion[mask])):
        raise PreprocessError("NORMALIZATION", "valid normalized joints must be finite")
    bounds = validate_run_bounds(run_bounds, edge_count)

    coordinates = np.zeros((sample_count, JOINT_COUNT, 2), dtype=np.float64)
    coordinates[mask] = motion[:, :, :2][mask]
    confidence = np.zeros((sample_count, JOINT_COUNT), dtype=np.float64)
    confidence[mask] = motion[:, :, 2][mask]
    pose = np.ascontiguousarray(coordinates.reshape(sample_count, 34))
    bones = np.zeros((sample_count, 32), dtype=np.float64)
    for index, (source, target) in enumerate(BONES):
        supported = mask[:, source] & mask[:, target]
        bones[supported, 2 * index : 2 * index + 2] = (
            coordinates[supported, target] - coordinates[supported, source]
        )
    geometry = np.ascontiguousarray(np.concatenate((pose, bones), axis=1))
    element_mask, coordinate_mask = _geometry_masks(mask)

    edge_coordinate_mask = np.zeros((edge_count, 66), dtype=np.bool_)
    edge_lengths = np.zeros(edge_count, dtype=np.float64)
    for start, stop in bounds.tolist():
        for edge in range(start, stop):
            edge_mask = coordinate_mask[edge] & coordinate_mask[edge + 1]
            edge_coordinate_mask[edge] = edge_mask
            dimension_count = int(np.count_nonzero(edge_mask))
            if dimension_count < 32:
                raise PreprocessError(
                    "GEOMETRY_LENGTH", f"edge {edge} has fewer than 32 geometry coordinates"
                )
            delta = geometry[edge + 1] - geometry[edge]
            squared_sum = neumaier_sum(float(value * value) for value in delta[edge_mask])
            scaled = np.float64(66.0 / dimension_count) * np.float64(squared_sum)
            length = float(np.sqrt(scaled)) if scaled >= 0.0 else float("nan")
            if not np.isfinite(squared_sum) or not np.isfinite(length) or length <= 0.0:
                raise PreprocessError("GEOMETRY_LENGTH", f"edge {edge} has invalid masked length")
            edge_lengths[edge] = length

    direction = np.zeros((sample_count, 66), dtype=np.float64)
    direction_element_mask = np.zeros((sample_count, 33), dtype=np.bool_)
    for start, stop in bounds.tolist():
        # Edge run [start, stop) owns sample run [start, stop].
        for sample in range(start + 1, stop):
            common_elements = (
                element_mask[sample - 1] & element_mask[sample] & element_mask[sample + 1]
            )
            common_coordinates = np.repeat(common_elements, 2)
            if not np.any(common_coordinates):
                continue
            incoming = geometry[sample] - geometry[sample - 1]
            outgoing = geometry[sample + 1] - geometry[sample]
            incoming_norm = _masked_norm(incoming, common_coordinates)
            outgoing_norm = _masked_norm(outgoing, common_coordinates)
            if (
                not np.isfinite(incoming_norm)
                or not np.isfinite(outgoing_norm)
                or incoming_norm <= DIRECTION_NORM_FLOOR
                or outgoing_norm <= DIRECTION_NORM_FLOOR
            ):
                continue
            chord_sum = np.zeros(66, dtype=np.float64)
            chord_sum[common_coordinates] = incoming[common_coordinates] / np.float64(
                incoming_norm
            ) + outgoing[common_coordinates] / np.float64(outgoing_norm)
            chord_sum_norm = _masked_norm(chord_sum, common_coordinates)
            if not np.isfinite(chord_sum_norm) or chord_sum_norm <= DIRECTION_NORM_FLOOR:
                continue
            direction[sample, common_coordinates] = chord_sum[common_coordinates] / np.float64(
                chord_sum_norm
            )
            direction_element_mask[sample] = common_elements

    teacher_input_f64 = np.concatenate(
        (
            geometry,
            direction,
            confidence,
            element_mask.astype(np.float64),
            direction_element_mask.astype(np.float64),
        ),
        axis=1,
    )
    if teacher_input_f64.shape != (sample_count, 215):
        raise AssertionError("teacher channel construction did not produce 215 channels")

    response_input_f64 = np.zeros((sample_count, 269), dtype=np.float64)
    response_input_f64[:, 0:66] = geometry
    response_input_f64[:, 66:83] = confidence
    response_input_f64[:, 185:202] = mask.astype(np.float64)
    response_input_f64[:, 202:218] = element_mask[:, 17:33].astype(np.float64)
    for lag, value_slice, mask_slice in (
        (4, slice(83, 117), slice(218, 235)),
        (2, slice(117, 151), slice(235, 252)),
        (1, slice(151, 185), slice(252, 269)),
    ):
        for start, stop in bounds.tolist():
            for sample in range(start + lag, stop + 1):
                response_input_f64[sample, value_slice] = pose[sample - lag]
                response_input_f64[sample, mask_slice] = mask[sample - lag].astype(np.float64)
    if not np.all(np.isfinite(teacher_input_f64)) or not np.all(np.isfinite(response_input_f64)):
        raise PreprocessError("NORMALIZATION", "model input contains a nonfinite value")

    return GeometryFeatures(
        pose=pose,
        bones=np.ascontiguousarray(bones),
        geometry=geometry,
        element_mask=element_mask,
        coordinate_mask=coordinate_mask,
        edge_coordinate_mask=np.ascontiguousarray(edge_coordinate_mask),
        edge_lengths=np.ascontiguousarray(edge_lengths),
        direction=np.ascontiguousarray(direction),
        direction_element_mask=np.ascontiguousarray(direction_element_mask),
        teacher_input=np.ascontiguousarray(teacher_input_f64.astype(np.dtype("<f4"))),
        response_input=np.ascontiguousarray(response_input_f64.astype(np.dtype("<f4"))),
        run_bounds=bounds,
    )


def preprocess_identity(
    motion: ArrayLike,
    frame_mask: ArrayLike,
    sampled_frame_indices: ArrayLike,
) -> PreprocessedIdentity:
    """Execute duplicate collapse, trimming, support, F4, and channel checks."""

    collapsed = collapse_duplicate_clocks(motion, frame_mask, sampled_frame_indices)
    valid_indices = np.flatnonzero(collapsed.frame_valid)
    if valid_indices.size == 0:
        raise PreprocessError("FRAME_SUPPORT", "no valid frame remains")
    first = int(valid_indices[0])
    stop = int(valid_indices[-1]) + 1
    retained_motion = np.ascontiguousarray(collapsed.motion[first:stop])
    retained_clocks = np.ascontiguousarray(collapsed.clocks[first:stop])
    retained_joint_mask = np.ascontiguousarray(collapsed.joint_mask[first:stop])
    retained_frame_valid = np.ascontiguousarray(collapsed.frame_valid[first:stop])
    if not np.all(retained_frame_valid):
        raise PreprocessError("FRAME_SUPPORT", "invalid frame occurs inside trimmed span")
    if retained_motion.shape[0] < MIN_SAMPLES:
        raise PreprocessError("TOO_SHORT", "surviving span needs at least 17 samples")
    clock_delta = np.diff(retained_clocks)
    if not np.all((clock_delta >= 1) & (clock_delta < P_MIN)):
        raise PreprocessError("EDGE_SUPPORT", "trimmed span contains an unsupported physical edge")
    edge_count = retained_motion.shape[0] - 1
    bounds = np.asarray([[0, edge_count]], dtype=np.int32)
    geometry = build_geometry_features(retained_motion, retained_joint_mask, bounds)
    return PreprocessedIdentity(
        motion=retained_motion,
        clocks=retained_clocks,
        joint_mask=retained_joint_mask,
        frame_valid=retained_frame_valid,
        scale=collapsed.scale,
        trim_bounds=(first, stop),
        geometry_features=geometry,
    )


__all__ = [
    "BONES",
    "CollapsedTrack",
    "GeometryFeatures",
    "P_MIN",
    "PreprocessError",
    "PreprocessedIdentity",
    "build_geometry_features",
    "collapse_duplicate_clocks",
    "mask_to_half_open_runs",
    "preprocess_identity",
    "validate_run_bounds",
]
