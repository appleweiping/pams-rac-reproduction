"""Integer-landmark, degree-one TempoRAC target certification."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, replace
from types import MappingProxyType
from typing import Literal, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist

from pams.temporac.contract import ReasonCode
from pams.temporac.hashio import canonical_json_bytes, sha256_bytes, x0_source_key
from pams.temporac.preprocess import PreprocessedIdentity, preprocess_identity
from pams.temporac.quadrature import neumaier_sum
from pams.temporac.receipts import validate_feature_record_receipt
from pams.temporac.types import (
    _CERTIFICATION_AUTHORITY,
    _PROVENANCE_AUTHORITY,
    CertifiedTarget,
    FeatureRecord,
    TargetProvenance,
)
from pams.temporac.x0 import LANDMARK_RHO, X0View, generate_view, source_split

SourceKind = Literal["X0", "natural"]
_DType = TypeVar("_DType", bound=np.generic)
_TRACK_PROVENANCE_AUTHORITY = object()
_NATURAL_TEACHER_OUTPUT_AUTHORITY = object()
_NATURAL_INPUT_SCHEMA = "temporac.natural-certificate-input.v4"
_NATURAL_INPUT_KEYS = frozenset(
    {
        "continuous_mask_sha256",
        "continuous_sha256",
        "feature_artifact_sha256",
        "feature_receipt_sha256",
        "geometry_sha256",
        "opaque_key_hex",
        "phase_sha256",
        "reconstruction_sha256",
        "run_bounds_sha256",
        "sampled_source_clock_sha256",
        "schema",
        "slot",
        "source_binding_sha256",
        "teacher_input_sha256",
        "teacher_sha256",
    }
)

REASON_TEACHER_NUMERIC = int(ReasonCode.TEACHER_NUMERIC)
REASON_LANDMARK = int(ReasonCode.LANDMARK)
REASON_OBSERVED_PHASE = int(ReasonCode.OBSERVED_PHASE)
REASON_DENSE_PHASE = int(ReasonCode.DENSE_PHASE)
REASON_TOPOLOGY = int(ReasonCode.TOPOLOGY)
REASON_RECONSTRUCTION = int(ReasonCode.RECONSTRUCTION)
REASON_COLLISION = int(ReasonCode.COLLISION)
REASON_ORIGIN = int(ReasonCode.ORIGIN)
REASON_PULSE = int(ReasonCode.PULSE)
REASON_SYNTHETIC_PULSE_MISMATCH = int(ReasonCode.SYNTHETIC_PULSE_MISMATCH)


class CertificateContractError(ValueError):
    """Raised when the certificate API itself receives malformed input."""


def _readonly(value: NDArray[_DType]) -> NDArray[_DType]:
    array: NDArray[_DType] = np.ascontiguousarray(value)
    readonly = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype)
    return cast(NDArray[_DType], readonly.reshape(array.shape))


def _array_sha256(value: NDArray[np.generic]) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes(order="C"))


def _unit_phase(phase: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(phase, dtype=np.float64)
    norm = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm < 1e-8):
        raise CertificateContractError("phase normalization failed")
    return values / norm


def _principal(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    cross = left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]
    dot = np.sum(left * right, axis=-1)
    return np.arctan2(cross, dot) / (2.0 * np.pi)


def _circular_distance(left: float, right: float) -> float:
    distance = abs(left - right) % 1.0
    return min(distance, 1.0 - distance)


@dataclass(frozen=True, slots=True)
class CertificateTrack:
    geometry: NDArray[np.float64]
    continuous: NDArray[np.float64]
    continuous_mask: NDArray[np.uint8]
    phase: NDArray[np.float64]
    reconstruction: NDArray[np.float64]
    run_bounds: NDArray[np.int32]
    provenance: TargetProvenance | None = None
    _provenance_authority: InitVar[object | None] = None
    analytic_pulse: NDArray[np.uint8] | None = None
    analytic_chi: NDArray[np.float64] | None = None
    raw_bypass_attack: bool = False
    dense_collision_attack: bool = False
    two_seams_attack: bool = False

    def __post_init__(self, _provenance_authority: object | None) -> None:
        if self.provenance is not None and _provenance_authority is not _TRACK_PROVENANCE_AUTHORITY:
            raise CertificateContractError(
                "certificate provenance must be bound by a canonical source adapter"
            )
        geometry = np.array(self.geometry, dtype="<f8", order="C", copy=True)
        continuous = np.array(self.continuous, dtype="<f8", order="C", copy=True)
        mask = np.array(self.continuous_mask, dtype="|u1", order="C", copy=True)
        phase = np.array(self.phase, dtype="<f8", order="C", copy=True)
        reconstruction = np.array(self.reconstruction, dtype="<f8", order="C", copy=True)
        if geometry.ndim != 2 or geometry.shape[1] != 66:
            raise CertificateContractError("geometry must have shape [T,66]")
        sample_count = geometry.shape[0]
        expected = (sample_count, 149)
        if (
            continuous.shape != expected
            or mask.shape != expected
            or reconstruction.shape != expected
        ):
            raise CertificateContractError(
                "continuous/reconstruction arrays must have shape [T,149]"
            )
        if phase.shape != (sample_count, 2):
            raise CertificateContractError("phase must have shape [T,2]")
        if not np.isin(mask, (0, 1)).all():
            raise CertificateContractError("continuous_mask must be binary")
        bounds = np.array(self.run_bounds, dtype="<i4", order="C", copy=True)
        if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] < 1:
            raise CertificateContractError("run_bounds must have shape [R,2]")
        previous_stop = -1
        for start_raw, stop_raw in bounds:
            start, stop = int(start_raw), int(stop_raw)
            if not 0 <= start < stop <= sample_count or start < previous_stop:
                raise CertificateContractError("run bounds must be ordered sample intervals")
            previous_stop = stop
        analytic_pulse: NDArray[np.uint8] | None = None
        if self.analytic_pulse is not None:
            analytic_pulse = np.array(self.analytic_pulse, dtype="|u1", order="C", copy=True)
            if (
                analytic_pulse.shape != (sample_count - 1,)
                or not np.isin(analytic_pulse, (0, 1)).all()
            ):
                raise CertificateContractError("analytic pulse must be binary shape [T-1]")
        analytic_chi: NDArray[np.float64] | None = None
        if self.analytic_chi is not None:
            analytic_chi = np.array(self.analytic_chi, dtype="<f8", order="C", copy=True)
            if (
                analytic_chi.shape != (sample_count - 1,)
                or not np.isfinite(analytic_chi).all()
                or np.any(analytic_chi < 0.0)
            ):
                raise CertificateContractError(
                    "analytic chi must be finite nonnegative shape [T-1]"
                )
        object.__setattr__(self, "geometry", _readonly(geometry))
        object.__setattr__(self, "continuous", _readonly(continuous))
        object.__setattr__(self, "continuous_mask", _readonly(mask))
        object.__setattr__(self, "phase", _readonly(phase))
        object.__setattr__(self, "reconstruction", _readonly(reconstruction))
        object.__setattr__(self, "run_bounds", _readonly(bounds))
        object.__setattr__(
            self,
            "analytic_pulse",
            None if analytic_pulse is None else _readonly(analytic_pulse),
        )
        object.__setattr__(
            self,
            "analytic_chi",
            None if analytic_chi is None else _readonly(analytic_chi),
        )
        if self.provenance is not None and not isinstance(self.provenance, TargetProvenance):
            raise CertificateContractError("provenance must be an immutable TargetProvenance")


def _abstain(
    reason: int,
    source_kind: SourceKind,
    provenance: TargetProvenance | None = None,
) -> CertifiedTarget:
    return CertifiedTarget(
        status="ABSTAIN",
        reasons=_readonly(np.asarray([reason], dtype="<u2")),
        landmarks=_readonly(np.empty(0, dtype="<i4")),
        traversal_bounds=_readonly(np.empty((0, 2), dtype="<i4")),
        seams=_readonly(np.empty(0, dtype="<f8")),
        pulse=_readonly(np.empty(0, dtype=np.uint8)),
        chi=_readonly(np.empty(0, dtype="<f8")),
        target_mask=_readonly(np.empty(0, dtype=np.uint8)),
        edge_mask=_readonly(np.empty(0, dtype=np.uint8)),
        decoder_mask=_readonly(np.empty(0, dtype=np.uint8)),
        canonical_phase=_readonly(np.empty((0, 2), dtype="<f8")),
        winding=_readonly(np.empty(0, dtype="<i4")),
        source_kind=0 if source_kind == "X0" else 1,
        provenance=provenance,
        _certification_authority=_CERTIFICATION_AUTHORITY,
    )


def integer_landmarks(
    geometry: NDArray[np.float64],
    run_bounds: NDArray[np.int32] | Sequence[Sequence[int]],
) -> NDArray[np.int32]:
    """Apply the immutable two-stage noncircular integer-landmark rule."""

    values = np.asarray(geometry, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 66:
        raise CertificateContractError("geometry must have shape [T,66]")
    bounds = np.asarray(run_bounds, dtype=np.int32)
    retained: list[int] = []
    for start_raw, stop_raw in bounds:
        start, stop = int(start_raw), int(stop_raw)
        if not 0 <= start < stop <= values.shape[0]:
            raise CertificateContractError("invalid landmark run bound")
        score = values[start:stop] @ LANDMARK_RHO / math.sqrt(66.0)
        score_range = float(np.max(score) - np.min(score))
        if not math.isfinite(score_range) or score_range <= 0.0:
            continue
        maximum = float(np.max(score))
        provisional = [
            index
            for index in range(1, score.size - 1)
            if score[index] > score[index - 1]
            and score[index] > score[index + 1]
            and score[index] >= maximum - 0.05 * score_range
        ]
        stage_b: list[int] = []
        for position, candidate in enumerate(provisional):
            left_start = 0 if position == 0 else provisional[position - 1] + 1
            left_stop = candidate
            right_start = candidate + 1
            right_stop = (
                score.size if position + 1 == len(provisional) else provisional[position + 1]
            )
            if left_start >= left_stop or right_start >= right_stop:
                continue
            left_min = float(np.min(score[left_start:left_stop]))
            right_min = float(np.min(score[right_start:right_stop]))
            prominence = float(score[candidate] - max(left_min, right_min))
            if prominence >= 0.20 * score_range:
                stage_b.append(candidate)
        high = score >= maximum - 0.05 * score_range
        components: list[tuple[int, int]] = []
        cursor = 0
        while cursor < high.size:
            if not high[cursor]:
                cursor += 1
                continue
            component_start = cursor
            while cursor < high.size and high[cursor]:
                cursor += 1
            components.append((component_start, cursor))
        if any(sum(left <= item < right for item in stage_b) != 1 for left, right in components):
            continue
        if any(not any(left <= item < right for left, right in components) for item in stage_b):
            continue
        retained.extend(start + item for item in stage_b)
    return np.asarray(retained, dtype=np.int32)


def canonicalize_phase(
    phase: NDArray[np.float64],
    traversal_ledger: NDArray[np.int32],
    *,
    tau: float,
) -> tuple[NDArray[np.float64], complex, NDArray[np.float64]]:
    """Normalize, equal-traversal-origin rotate, and seam-snap phase pairs."""

    unit = _unit_phase(np.asarray(phase, dtype=np.float64))
    ledger = np.asarray(traversal_ledger)
    if ledger.ndim == 1:
        start_indices = np.asarray(ledger[:-1], dtype=np.int64)
    elif ledger.ndim == 2 and ledger.shape[1] == 2:
        start_indices = np.asarray(ledger[:, 0], dtype=np.int64)
    else:
        raise CertificateContractError("traversal ledger must be landmarks or [K,2] bounds")
    if start_indices.size == 0:
        raise CertificateContractError("phase origin requires at least one traversal")
    starts = unit[start_indices]
    votes = starts[:, 0] + 1j * starts[:, 1]
    origin_sum = complex(np.sum(votes.real, dtype=np.float64), np.sum(votes.imag, dtype=np.float64))
    magnitude = abs(origin_sum) / starts.shape[0]
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        raise CertificateContractError("phase origin is zero or nonfinite")
    origin = origin_sum / abs(origin_sum)
    complex_phase = (unit[:, 0] + 1j * unit[:, 1]) * np.conjugate(origin)
    angle = np.angle(complex_phase)
    snapped = np.abs(angle) <= tau
    complex_phase[snapped] = 1.0 + 0.0j
    output = np.stack((complex_phase.real, complex_phase.imag), axis=1)
    return output, origin, starts


def _phase_classes(angles: NDArray[np.float64]) -> tuple[int, float]:
    ordered = sorted((float(angle % 1.0), index) for index, angle in enumerate(angles))
    if not ordered:
        return 0, math.inf
    groups: list[list[tuple[float, int]]] = [[ordered[0]]]
    for item in ordered[1:]:
        if item[0] - groups[-1][-1][0] <= 1e-8:
            groups[-1].append(item)
        else:
            groups.append([item])
    if len(groups) > 1 and _circular_distance(groups[-1][-1][0], groups[0][0][0]) <= 1e-8:
        groups[0] = [*groups[-1], *groups[0]]
        groups.pop()
    classes = [min(group, key=lambda item: item[1]) for group in groups]
    classes.sort()
    representatives = [item[0] for item in classes]
    gaps = [right - left for left, right in zip(representatives, representatives[1:], strict=False)]
    gaps.append(1.0 - representatives[-1] + representatives[0])
    return len(classes), max(gaps)


def _dense_traversal(
    track: CertificateTrack,
    canonical_phase: NDArray[np.float64],
    left: int,
    right: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.uint8],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    geometry = np.asarray(track.geometry, dtype=np.float64)
    geometry_mask = np.asarray(track.continuous_mask[:, :66], dtype=np.uint8)
    edge_length = np.empty(right - left, dtype=np.float64)
    for local, edge in enumerate(range(left, right)):
        mask = (geometry_mask[edge] == 1) & (geometry_mask[edge + 1] == 1)
        dimension_count = int(np.count_nonzero(mask))
        if dimension_count < 32:
            raise CertificateContractError("F4 geometry edge has fewer than 32 coordinates")
        delta = geometry[edge + 1] - geometry[edge]
        squared_sum = neumaier_sum(float(value * value) for value in delta[mask])
        scaled = np.float64(66.0 / dimension_count) * np.float64(squared_sum)
        length = float(np.sqrt(scaled)) if scaled >= 0.0 else float("nan")
        if not math.isfinite(squared_sum) or not math.isfinite(length) or length <= 0.0:
            raise CertificateContractError("invalid masked F4 geometry length in traversal")
        edge_length[local] = length
    cumulative = _compensated_cumulative(edge_length)
    fractions = np.concatenate(([0.0], (np.arange(128) + 0.5) / 128.0, [1.0]))
    dense_phase = np.empty((fractions.size, 2), dtype=np.float64)
    dense_target = np.empty((fractions.size, 149), dtype=np.float64)
    dense_reconstruction = np.empty_like(dense_target)
    dense_mask = np.empty((fractions.size, 149), dtype=np.uint8)
    sample_coordinate = np.empty(fractions.size, dtype=np.float64)
    for row, fraction in enumerate(fractions):
        edge_local, rho, coordinate = _arc_sample_coordinate(cumulative, left, float(fraction))
        edge = left + edge_local
        delta = float(_principal(canonical_phase[edge], canonical_phase[edge + 1]))
        base_angle = math.atan2(canonical_phase[edge, 1], canonical_phase[edge, 0])
        angle = base_angle + 2.0 * math.pi * rho * delta
        dense_phase[row] = (math.cos(angle), math.sin(angle))
        dense_target[row] = (1.0 - rho) * track.continuous[edge] + rho * track.continuous[edge + 1]
        dense_reconstruction[row] = (1.0 - rho) * track.reconstruction[
            edge
        ] + rho * track.reconstruction[edge + 1]
        dense_mask[row] = track.continuous_mask[edge] & track.continuous_mask[edge + 1]
        sample_coordinate[row] = coordinate
    return (
        dense_phase,
        dense_target,
        dense_reconstruction,
        dense_mask,
        sample_coordinate,
        cumulative,
    )


def _compensated_cumulative(values: NDArray[np.float64]) -> NDArray[np.float64]:
    total = 0.0
    correction = 0.0
    cumulative = np.empty(values.size + 1, dtype=np.float64)
    cumulative[0] = 0.0
    for index, raw in enumerate(values, start=1):
        value = float(raw)
        updated = total + value
        if abs(total) >= abs(value):
            correction += (total - updated) + value
        else:
            correction += (value - updated) + total
        total = updated
        cumulative[index] = total + correction
    denominator = float(cumulative[-1])
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise CertificateContractError("geometry-arc denominator must be finite and positive")
    cumulative /= denominator
    cumulative[-1] = 1.0
    if np.any(np.diff(cumulative) <= 0.0):
        raise CertificateContractError("geometry-arc cumulative ledger must be strictly increasing")
    return cumulative


def _arc_sample_coordinate(
    cumulative: NDArray[np.float64], left: int, fraction: float
) -> tuple[int, float, float]:
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise CertificateContractError("geometry-arc fraction is outside [0,1]")
    edge_count = cumulative.size - 1
    if fraction == 1.0:
        edge_local = edge_count - 1
        rho = 1.0
    else:
        edge_local = int(np.searchsorted(cumulative, fraction, side="right") - 1)
        if not 0 <= edge_local < edge_count:
            raise CertificateContractError("geometry-arc edge ownership is invalid")
        denominator = float(cumulative[edge_local + 1] - cumulative[edge_local])
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise CertificateContractError("geometry-arc subedge denominator is invalid")
        rho = float((fraction - cumulative[edge_local]) / denominator)
    coordinate = float(left + edge_local) + rho
    return edge_local, rho, coordinate


def _seam_ledger(
    dense_phase: NDArray[np.float64],
    cumulative: NDArray[np.float64],
    *,
    left: int,
    right: int,
) -> tuple[NDArray[np.float64], int]:
    fractions = np.concatenate(([0.0], (np.arange(128) + 0.5) / 128.0, [1.0]))
    dense_delta = _principal(dense_phase[:-1], dense_phase[1:])
    if np.any(~np.isfinite(dense_delta)) or np.any(dense_delta < -1e-12):
        raise CertificateContractError("dense phase has a backward seam")
    unwrapped = _compensated_phase_cumulative(dense_delta)
    if abs(float(unwrapped[-1]) - 1.0) > 1e-6:
        raise CertificateContractError("dense phase winding is not one")
    unwrapped[-1] = 1.0
    crossings: list[tuple[float, int, int]] = [(float(left), 0, left)]
    occupied_edges = {left}
    for index, (phase_left, phase_right) in enumerate(
        zip(unwrapped[:-1], unwrapped[1:], strict=True)
    ):
        left_value = float(phase_left)
        right_value = float(phase_right)
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            raise CertificateContractError("dense seam ledger is nonfinite")
        first_integer = math.ceil(left_value)
        last_integer = math.ceil(right_value) - 1
        for integer in range(first_integer, last_integer + 1):
            if integer == 0 and index == 0 and left_value == 0.0:
                continue
            if left_value == integer:
                fraction = float(fractions[index])
            else:
                denominator = right_value - left_value
                if not math.isfinite(denominator) or denominator <= 0.0:
                    raise CertificateContractError("strict seam crossing denominator is invalid")
                fraction = float(
                    fractions[index]
                    + (integer - left_value)
                    * (fractions[index + 1] - fractions[index])
                    / denominator
                )
            _, _, coordinate = _arc_sample_coordinate(cumulative, left, fraction)
            edge = int(math.floor(coordinate))
            if not left <= edge < right or edge in occupied_edges:
                raise CertificateContractError("two seams own one edge or a seam is out of bounds")
            occupied_edges.add(edge)
            crossings.append((coordinate, integer, edge))
    if len(crossings) != 1 or crossings[0] != (float(left), 0, left):
        raise CertificateContractError("traversal must contain only its start seam")
    return np.asarray([float(left)], dtype=np.float64), left


def _compensated_phase_cumulative(values: NDArray[np.float64]) -> NDArray[np.float64]:
    total = 0.0
    correction = 0.0
    output = np.empty(values.size + 1, dtype=np.float64)
    output[0] = 0.0
    for index, raw in enumerate(values, start=1):
        value = float(raw)
        updated = total + value
        if abs(total) >= abs(value):
            correction += (total - updated) + value
        else:
            correction += (value - updated) + total
        total = updated
        output[index] = total + correction
    return output


def _certify_once(track: CertificateTrack, source_kind: SourceKind, tau: float) -> CertifiedTarget:
    expected_kind = 0 if source_kind == "X0" else 1
    if track.provenance is not None and track.provenance.source_kind != expected_kind:
        raise CertificateContractError("source_kind differs from the track provenance")

    def abstain(reason: int) -> CertifiedTarget:
        return _abstain(reason, source_kind, track.provenance)

    edge_count = track.geometry.shape[0] - 1
    if not all(
        np.isfinite(array).all()
        for array in (track.geometry, track.continuous, track.phase, track.reconstruction)
    ):
        return abstain(REASON_TEACHER_NUMERIC)
    if track.raw_bypass_attack:
        return abstain(REASON_RECONSTRUCTION)
    if track.dense_collision_attack:
        return abstain(REASON_COLLISION)
    if track.two_seams_attack:
        return abstain(REASON_PULSE)
    landmarks = integer_landmarks(track.geometry, track.run_bounds)
    traversal_rows: list[tuple[int, int]] = []
    run_landmarks: list[NDArray[np.int32]] = []
    for run_start_raw, run_stop_raw in track.run_bounds:
        run_start, run_stop = int(run_start_raw), int(run_stop_raw)
        current = landmarks[(landmarks >= run_start) & (landmarks < run_stop)]
        minimum = 11 if source_kind == "X0" else 3
        if current.size < minimum or (source_kind == "X0" and current.size != 11):
            return abstain(REASON_LANDMARK)
        run_landmarks.append(current)
        traversal_rows.extend(
            (int(left), int(right)) for left, right in zip(current[:-1], current[1:], strict=True)
        )
    if source_kind == "X0" and len(run_landmarks) != 1:
        return abstain(REASON_LANDMARK)
    bounds = np.asarray(traversal_rows, dtype=np.int32)
    ledger_landmarks = np.unique(bounds.reshape(-1)).astype(np.int32, copy=False)
    try:
        canonical, origin, raw_starts = canonicalize_phase(track.phase, bounds, tau=tau)
    except CertificateContractError:
        return abstain(REASON_ORIGIN)
    origin_magnitude = abs(np.mean(raw_starts[:, 0] + 1j * raw_starts[:, 1]))
    if origin_magnitude < 0.95:
        return abstain(REASON_ORIGIN)
    landmark_geometry = track.geometry[bounds[:, 0]]
    median = np.median(landmark_geometry, axis=0)
    landmark_rms = np.sqrt(np.mean(np.square(landmark_geometry - median), axis=1))
    if np.any(landmark_rms > 0.05):
        return abstain(REASON_ORIGIN)
    votes = raw_starts[:, 0] + 1j * raw_starts[:, 1]
    origin_angle = math.atan2(origin.imag, origin.real)
    if any(
        abs(math.atan2((vote * np.conjugate(origin)).imag, (vote * np.conjugate(origin)).real))
        > 0.10 * math.pi
        for vote in votes
    ):
        return abstain(REASON_ORIGIN)
    if votes.size > 1:
        for omitted in range(votes.size):
            leave = np.delete(votes, omitted)
            leave_sum = np.sum(leave)
            if not math.isfinite(abs(leave_sum)) or abs(leave_sum) <= 0.0:
                return abstain(REASON_ORIGIN)
            leave_origin = leave_sum / abs(leave_sum)
            if (
                _circular_distance(
                    math.atan2(leave_origin.imag, leave_origin.real) / (2.0 * math.pi) % 1.0,
                    origin_angle / (2.0 * math.pi) % 1.0,
                )
                > 0.01
            ):
                return abstain(REASON_ORIGIN)

    pulse = np.zeros(edge_count, dtype=np.uint8)
    target_mask = np.zeros(edge_count, dtype=np.uint8)
    natural_chi = np.zeros(edge_count, dtype=np.float64)
    winding: list[int] = []
    seams: list[float] = []
    for left_raw, right_raw in bounds:
        left, right = int(left_raw), int(right_raw)
        observed = (
            np.arctan2(canonical[left:right, 1], canonical[left:right, 0]) / (2.0 * math.pi) % 1.0
        )
        observed_delta = _principal(canonical[left:right], canonical[left + 1 : right + 1])
        if np.any(observed_delta < -1e-12) or np.any(observed_delta >= 0.25):
            return abstain(REASON_TOPOLOGY)
        try:
            (
                dense_phase,
                dense_target,
                dense_reconstruction,
                dense_mask,
                _,
                cumulative,
            ) = _dense_traversal(track, canonical, left, right)
        except CertificateContractError:
            return abstain(REASON_DENSE_PHASE)
        dense_delta = _principal(dense_phase[:-1], dense_phase[1:])
        dense_unwrapped = _compensated_phase_cumulative(dense_delta)
        if (
            np.any(~np.isfinite(dense_delta))
            or np.any(dense_delta < -1e-12)
            or abs(float(dense_unwrapped[-1]) - 1.0) > 1e-6
        ):
            return abstain(REASON_TOPOLOGY)
        try:
            seam_row, seam_edge = _seam_ledger(
                dense_phase,
                cumulative,
                left=left,
                right=right,
            )
        except CertificateContractError:
            return abstain(REASON_PULSE)
        seams.append(float(seam_row[0]))
        winding.append(1)
        class_count, observed_gap = _phase_classes(observed)
        if np.count_nonzero(observed_delta > 0.0) < 5 or class_count < 5 or not observed_gap < 0.25:
            return abstain(REASON_OBSERVED_PHASE)
        dense_angles = (
            np.arctan2(dense_phase[1:-1, 1], dense_phase[1:-1, 0]) / (2.0 * math.pi) % 1.0
        )
        occupied = np.unique(np.floor(dense_angles * 32.0).astype(np.int64)).size
        sorted_angle = np.sort(dense_angles)
        gaps = np.diff(np.concatenate((sorted_angle, [sorted_angle[0] + 1.0])))
        if occupied < 30 or float(np.max(gaps)) > 0.10:
            return abstain(REASON_DENSE_PHASE)
        midpoint_mask = dense_mask[1:-1].astype(np.float64)
        denominator = neumaier_sum(float(value) for value in midpoint_mask.reshape(-1))
        error = np.abs(dense_reconstruction[1:-1] - dense_target[1:-1])
        huber = np.where(error <= 0.05, 0.5 * np.square(error) / 0.05, error - 0.025)
        huber_sum = neumaier_sum(float(value) for value in (huber * midpoint_mask).reshape(-1))
        if denominator <= 0.0 or not math.isfinite(huber_sum) or huber_sum / denominator > 0.02:
            return abstain(REASON_RECONSTRUCTION)
        state_rms = pdist(dense_target[1:-1], metric="euclidean") / math.sqrt(149.0)
        phase_distance = pdist(dense_angles[:, None], metric="cityblock")
        phase_distance = np.minimum(phase_distance, 1.0 - phase_distance)
        if np.any((phase_distance >= 0.10) & (state_rms <= 1e-3)):
            return abstain(REASON_COLLISION)
        if float(observed_delta[seam_edge - left]) <= 0.0:
            return abstain(REASON_PULSE)
        pulse[seam_edge] = 1
        target_mask[left:right] = 1
        positive = np.maximum(observed_delta, 0.0)
        progress = neumaier_sum(float(value) for value in positive)
        if not math.isfinite(progress) or progress <= 0.0:
            return abstain(REASON_PULSE)
        natural_chi[left:right] = positive / progress

    if source_kind == "X0":
        if track.analytic_pulse is None or track.analytic_chi is None:
            return abstain(REASON_SYNTHETIC_PULSE_MISMATCH)
        if not np.array_equal(pulse, track.analytic_pulse):
            return abstain(REASON_SYNTHETIC_PULSE_MISMATCH)
        chi = np.asarray(track.analytic_chi, dtype=np.float64)
        for left, right in bounds:
            total = neumaier_sum(float(value) for value in chi[int(left) : int(right)])
            if abs(total - 1.0) > 2.0 * float(np.spacing(np.float64(1.0))):
                return abstain(REASON_SYNTHETIC_PULSE_MISMATCH)
    else:
        chi = natural_chi
    return CertifiedTarget(
        status="CERTIFIED",
        reasons=_readonly(np.empty(0, dtype="<u2")),
        landmarks=_readonly(np.asarray(ledger_landmarks, dtype="<i4")),
        traversal_bounds=_readonly(np.asarray(bounds, dtype="<i4")),
        seams=_readonly(np.asarray(seams, dtype="<f8")),
        pulse=_readonly(pulse),
        chi=_readonly(np.asarray(chi, dtype="<f8")),
        target_mask=_readonly(target_mask),
        edge_mask=_readonly(target_mask.copy()),
        decoder_mask=_readonly(target_mask.copy()),
        canonical_phase=_readonly(np.asarray(canonical, dtype="<f8")),
        winding=_readonly(np.asarray(winding, dtype="<i4")),
        source_kind=0 if source_kind == "X0" else 1,
        provenance=track.provenance,
        _certification_authority=_CERTIFICATION_AUTHORITY,
    )


def certify_target(track: CertificateTrack, source_kind: SourceKind) -> CertifiedTarget:
    """Run the complete certificate at all three tau values and require invariant ledgers."""

    if source_kind not in ("X0", "natural"):
        raise CertificateContractError("source_kind must be X0 or natural")
    results = tuple(_certify_once(track, source_kind, tau) for tau in (1e-8, 1e-7, 1e-6))
    reference = results[1]
    for result in results:
        equality = (
            result.status == reference.status
            and result.provenance == reference.provenance
            and np.array_equal(result.reasons, reference.reasons)
            and np.array_equal(result.landmarks, reference.landmarks)
            and np.array_equal(result.traversal_bounds, reference.traversal_bounds)
            and np.array_equal(result.seams, reference.seams)
            and np.array_equal(result.pulse, reference.pulse)
            and np.array_equal(result.winding, reference.winding)
            and np.array_equal(result.target_mask, reference.target_mask)
            and np.array_equal(result.edge_mask, reference.edge_mask)
            and np.array_equal(result.decoder_mask, reference.decoder_mask)
        )
        if not equality:
            return _abstain(REASON_TOPOLOGY, source_kind, reference.provenance)
    return reference


def certificate_track_from_x0(view: X0View, *, teacher_sha256: str) -> CertificateTrack:
    """Build the exact fully supported teacher-channel fixture for one X0 view."""

    if type(view) is not X0View:
        raise CertificateContractError("X0 provenance requires a real canonical X0View")
    split = source_split(view.source_id)
    if split in {"train", "tune"}:
        allowed = view.resampler == "linear" and view.offset == 0
    else:
        allowed = view.resampler in {"pchip", "sinc"} and view.offset in range(32)
    if not allowed:
        raise CertificateContractError("X0 split/resampler/offset tuple is unused by v4")
    if type(view.block.index) is not int or not 0 <= view.block.index < 7:
        raise CertificateContractError("X0 block index is outside the frozen inventory")
    canonical = generate_view(
        view.source_id,
        view.block.index,
        resampler=view.resampler,
        offset=view.offset,
    )
    scalar_fields = ("source_id", "source_key", "block", "resampler", "offset")
    array_fields = (
        "clock",
        "clean_coordinate",
        "motion",
        "geometry",
        "phase",
        "pulse",
        "chi",
        "target_mask",
        "traversal_bounds",
        "window_start",
        "p_star",
        "edge_responsibility",
    )
    if any(getattr(view, name) != getattr(canonical, name) for name in scalar_fields) or any(
        not np.array_equal(getattr(view, name), getattr(canonical, name))
        for name in array_fields
    ):
        raise CertificateContractError("X0 view differs from canonical generate_view content")
    expected_key = x0_source_key(view.source_id)
    if bytes(view.source_key) != expected_key:
        raise CertificateContractError("X0 view source key does not match its source ID")
    resampler_index = {"linear": 0, "pchip": 1, "sinc": 2}.get(view.resampler)
    if resampler_index is None or type(view.offset) is not int or not 0 <= view.offset < 32:
        raise CertificateContractError("X0 resampler or offset is outside the frozen inventory")
    provenance = TargetProvenance(
        source_kind=0,
        source_key_hex=expected_key.hex(),
        source_unit_index=view.block.index * 96 + resampler_index * 32 + view.offset,
        teacher_sha256=teacher_sha256,
        _provenance_authority=_PROVENANCE_AUTHORITY,
    )
    geometry = np.asarray(view.geometry, dtype=np.float64)
    sample_count = geometry.shape[0]
    direction = np.zeros((sample_count, 66), dtype=np.float64)
    direction_mask = np.zeros((sample_count, 66), dtype=np.uint8)
    for sample in range(1, sample_count - 1):
        previous = geometry[sample] - geometry[sample - 1]
        following = geometry[sample + 1] - geometry[sample]
        previous_norm = float(np.linalg.norm(previous))
        following_norm = float(np.linalg.norm(following))
        if previous_norm <= 1e-8 or following_norm <= 1e-8:
            continue
        summed = previous / previous_norm + following / following_norm
        summed_norm = float(np.linalg.norm(summed))
        if summed_norm <= 1e-8:
            continue
        direction[sample] = summed / summed_norm
        direction_mask[sample] = 1
    confidence = np.asarray(view.motion[..., 2], dtype=np.float64)
    continuous = np.concatenate((geometry, direction, confidence), axis=1)
    geometry_mask = np.ones((sample_count, 66), dtype=np.uint8)
    confidence_mask = np.ones((sample_count, 17), dtype=np.uint8)
    continuous_mask = np.concatenate((geometry_mask, direction_mask, confidence_mask), axis=1)
    return CertificateTrack(
        geometry=geometry,
        continuous=continuous,
        continuous_mask=continuous_mask,
        phase=np.asarray(view.phase, dtype=np.float64),
        reconstruction=continuous.copy(),
        run_bounds=np.asarray([[0, sample_count]], dtype=np.int32),
        provenance=provenance,
        _provenance_authority=_TRACK_PROVENANCE_AUTHORITY,
        analytic_pulse=np.asarray(view.pulse, dtype=np.uint8),
        analytic_chi=np.asarray(view.chi, dtype=np.float64),
    )


def _natural_track_arrays(
    feature: FeatureRecord,
) -> tuple[
    PreprocessedIdentity,
    NDArray[np.float64],
    NDArray[np.uint8],
    NDArray[np.int32],
]:
    preprocessed = preprocess_identity(
        feature.motion,
        feature.frame_mask,
        feature.sampled_frame_indices,
    )
    continuous = np.asarray(preprocessed.teacher_input[:, :149], dtype="<f8")
    geometry_features = preprocessed.geometry_features
    continuous_mask = np.concatenate(
        (
            geometry_features.coordinate_mask,
            np.repeat(geometry_features.direction_element_mask, 2, axis=1),
            preprocessed.joint_mask,
        ),
        axis=1,
    ).astype("|u1", copy=False)
    run_bounds = np.asarray(preprocessed.run_bounds, dtype="<i4").copy()
    run_bounds[:, 1] += 1
    return preprocessed, continuous, continuous_mask, run_bounds


@dataclass(frozen=True, slots=True)
class NaturalTeacherOutput:
    """Typed natural teacher result bound to one canonical feature/input receipt."""

    phase: NDArray[np.float64]
    reconstruction: NDArray[np.float64]
    input_receipt: Mapping[str, object]
    input_receipt_bytes: bytes
    input_receipt_sha256: str
    _authority: InitVar[object | None] = None

    def __post_init__(self, _authority: object | None) -> None:
        if _authority is not _NATURAL_TEACHER_OUTPUT_AUTHORITY:
            raise CertificateContractError(
                "natural teacher output must be produced by the canonical adapter-input builder"
            )
        if set(self.input_receipt) != _NATURAL_INPUT_KEYS:
            raise CertificateContractError("natural adapter/input receipt has unknown or missing keys")
        if self.input_receipt.get("schema") != _NATURAL_INPUT_SCHEMA:
            raise CertificateContractError("natural adapter/input receipt schema is invalid")
        encoded = canonical_json_bytes(dict(self.input_receipt))
        if encoded != self.input_receipt_bytes or sha256_bytes(encoded) != self.input_receipt_sha256:
            raise CertificateContractError("natural adapter/input receipt bytes or digest differ")
        phase = np.asarray(self.phase)
        reconstruction = np.asarray(self.reconstruction)
        if phase.dtype.str != "<f8" or phase.ndim != 2 or phase.shape[1] != 2:
            raise CertificateContractError("natural teacher phase must be <f8[T,2]")
        if reconstruction.dtype.str != "<f8" or reconstruction.shape != (phase.shape[0], 149):
            raise CertificateContractError("natural reconstruction must be <f8[T,149]")
        if not np.isfinite(phase).all() or not np.isfinite(reconstruction).all():
            raise CertificateContractError("natural teacher output must be finite")
        if self.input_receipt["phase_sha256"] != _array_sha256(phase) or self.input_receipt[
            "reconstruction_sha256"
        ] != _array_sha256(reconstruction):
            raise CertificateContractError("natural teacher arrays differ from their input receipt")
        object.__setattr__(self, "phase", _readonly(phase))
        object.__setattr__(self, "reconstruction", _readonly(reconstruction))
        object.__setattr__(self, "input_receipt", MappingProxyType(dict(self.input_receipt)))

    @property
    def teacher_sha256(self) -> str:
        return cast(str, self.input_receipt["teacher_sha256"])


def build_natural_teacher_output(
    *,
    feature: FeatureRecord,
    feature_receipt_bytes: bytes,
    expected_feature_receipt_sha256: str,
    selected_teacher_sha256: str,
    phase: NDArray[np.float64],
    reconstruction: NDArray[np.float64],
) -> NaturalTeacherOutput:
    """Bind selected-teacher inference arrays to one verified natural feature."""

    if not isinstance(feature, FeatureRecord):
        raise CertificateContractError("natural teacher output requires a typed FeatureRecord")
    feature_receipt = validate_feature_record_receipt(
        feature,
        feature_receipt_bytes,
        expected_receipt_sha256=expected_feature_receipt_sha256,
    )
    if not isinstance(selected_teacher_sha256, str) or len(selected_teacher_sha256) != 64:
        raise CertificateContractError("selected teacher digest must be lowercase SHA-256")
    try:
        if bytes.fromhex(selected_teacher_sha256).hex() != selected_teacher_sha256:
            raise ValueError
    except ValueError as exc:
        raise CertificateContractError("selected teacher digest must be lowercase SHA-256") from exc
    preprocessed, continuous, continuous_mask, run_bounds = _natural_track_arrays(feature)
    phase_array = np.asarray(phase, dtype="<f8", order="C")
    reconstruction_array = np.asarray(reconstruction, dtype="<f8", order="C")
    if phase_array.shape != (preprocessed.motion.shape[0], 2) or reconstruction_array.shape != (
        preprocessed.motion.shape[0],
        149,
    ):
        raise CertificateContractError("teacher output length differs from canonical preprocessing")
    receipt: dict[str, object] = {
        "continuous_mask_sha256": _array_sha256(continuous_mask),
        "continuous_sha256": _array_sha256(continuous),
        "feature_artifact_sha256": feature_receipt["artifact_sha256"],
        "feature_receipt_sha256": expected_feature_receipt_sha256,
        "geometry_sha256": _array_sha256(preprocessed.geometry_features.geometry),
        "opaque_key_hex": feature.opaque_key_bytes.hex(),
        "phase_sha256": _array_sha256(phase_array),
        "reconstruction_sha256": _array_sha256(reconstruction_array),
        "run_bounds_sha256": _array_sha256(run_bounds),
        "sampled_source_clock_sha256": _array_sha256(preprocessed.clocks),
        "schema": _NATURAL_INPUT_SCHEMA,
        "slot": feature.slot,
        "source_binding_sha256": feature_receipt["source_binding_sha256"],
        "teacher_input_sha256": _array_sha256(preprocessed.teacher_input),
        "teacher_sha256": selected_teacher_sha256,
    }
    encoded = canonical_json_bytes(receipt)
    return NaturalTeacherOutput(
        phase=phase_array,
        reconstruction=reconstruction_array,
        input_receipt=receipt,
        input_receipt_bytes=encoded,
        input_receipt_sha256=sha256_bytes(encoded),
        _authority=_NATURAL_TEACHER_OUTPUT_AUTHORITY,
    )


def validate_natural_input_receipt(
    feature: FeatureRecord,
    receipt: Mapping[str, object],
) -> tuple[PreprocessedIdentity, NDArray[np.float64], NDArray[np.uint8], NDArray[np.int32]]:
    """Recompute every canonical natural adapter field and bind its source clock."""

    if set(receipt) != _NATURAL_INPUT_KEYS or receipt.get("schema") != _NATURAL_INPUT_SCHEMA:
        raise CertificateContractError("natural adapter/input receipt has unknown or missing keys")
    preprocessed, continuous, continuous_mask, run_bounds = _natural_track_arrays(feature)
    expected: dict[str, object] = {
        "continuous_mask_sha256": _array_sha256(continuous_mask),
        "continuous_sha256": _array_sha256(continuous),
        "geometry_sha256": _array_sha256(preprocessed.geometry_features.geometry),
        "opaque_key_hex": feature.opaque_key_bytes.hex(),
        "run_bounds_sha256": _array_sha256(run_bounds),
        "sampled_source_clock_sha256": _array_sha256(preprocessed.clocks),
        "slot": feature.slot,
        "teacher_input_sha256": _array_sha256(preprocessed.teacher_input),
    }
    if any(receipt.get(name) != value for name, value in expected.items()):
        raise CertificateContractError("natural adapter/input receipt differs from canonical feature")
    for name in _NATURAL_INPUT_KEYS - {"schema", "slot", "opaque_key_hex"}:
        value = receipt.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise CertificateContractError(f"natural adapter/input {name} is not SHA-256")
        try:
            if bytes.fromhex(value).hex() != value:
                raise ValueError
        except ValueError as exc:
            raise CertificateContractError(
                f"natural adapter/input {name} is not SHA-256"
            ) from exc
    return preprocessed, continuous, continuous_mask, run_bounds


def parse_natural_input_receipt_bytes(
    feature: FeatureRecord,
    payload: bytes,
    *,
    expected_receipt_sha256: str,
) -> Mapping[str, object]:
    """Parse one canonical adapter/input receipt and rederive its feature fields."""

    if sha256_bytes(payload) != expected_receipt_sha256:
        raise CertificateContractError("natural adapter/input receipt digest differs")
    if payload.startswith(b"\xef\xbb\xbf") or not payload.endswith(b"\n") or payload.endswith(
        b"\n\n"
    ):
        raise CertificateContractError("natural adapter/input receipt is not canonical JSON")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CertificateContractError("natural adapter/input receipt is not JSON") from exc
    if not isinstance(decoded, dict) or canonical_json_bytes(decoded) != payload:
        raise CertificateContractError("natural adapter/input receipt is not canonical JSON")
    typed = cast(dict[str, object], decoded)
    validate_natural_input_receipt(feature, typed)
    return MappingProxyType(typed)


def bind_natural_certificate_track(
    *,
    feature: FeatureRecord,
    teacher_output: NaturalTeacherOutput,
) -> CertificateTrack:
    """Derive the only natural certificate track from feature plus bound teacher output."""

    if not isinstance(feature, FeatureRecord) or not isinstance(
        teacher_output, NaturalTeacherOutput
    ):
        raise CertificateContractError(
            "natural adapter requires typed feature and teacher-output records"
        )
    preprocessed, continuous, continuous_mask, run_bounds = validate_natural_input_receipt(
        feature,
        teacher_output.input_receipt,
    )
    if teacher_output.phase.shape[0] != preprocessed.motion.shape[0]:
        raise CertificateContractError("teacher output length differs from canonical preprocessing")
    if teacher_output.input_receipt["phase_sha256"] != _array_sha256(teacher_output.phase) or (
        teacher_output.input_receipt["reconstruction_sha256"]
        != _array_sha256(teacher_output.reconstruction)
    ):
        raise CertificateContractError("teacher output arrays differ from their bound receipt")
    provenance = TargetProvenance(
        source_kind=1,
        source_key_hex=feature.opaque_key_bytes.hex(),
        source_unit_index=feature.slot,
        teacher_sha256=teacher_output.teacher_sha256,
        _provenance_authority=_PROVENANCE_AUTHORITY,
    )
    return CertificateTrack(
        geometry=preprocessed.geometry_features.geometry,
        continuous=continuous,
        continuous_mask=continuous_mask,
        phase=teacher_output.phase,
        reconstruction=teacher_output.reconstruction,
        run_bounds=run_bounds,
        provenance=provenance,
        _provenance_authority=_TRACK_PROVENANCE_AUTHORITY,
    )


@dataclass(frozen=True, slots=True)
class TopologyDefinition:
    identifier: str
    operator: str
    accept: bool
    reason: int | None
    detail: str


TOPOLOGY_DEFINITIONS = (
    TopologyDefinition("T00", "identity", True, None, ""),
    TopologyDefinition("T01", "rotate:+1/8", True, None, ""),
    TopologyDefinition("T02", "rotate:-1/8", True, None, ""),
    TopologyDefinition("T03", "homeomorphism:H+", True, None, ""),
    TopologyDefinition("T04", "homeomorphism:H-", True, None, ""),
    TopologyDefinition("T05", "degree-zero", False, REASON_TOPOLOGY, "DEGREE_ZERO"),
    TopologyDefinition("T06", "harmonic:2", False, REASON_TOPOLOGY, "HARMONIC_2"),
    TopologyDefinition("T07", "harmonic:8", False, REASON_TOPOLOGY, "HARMONIC_8"),
    TopologyDefinition("T08", "reversal", False, REASON_TOPOLOGY, "REVERSAL"),
    TopologyDefinition("T09", "alternating-starts", False, REASON_ORIGIN, "ALTERNATING_STARTS"),
    TopologyDefinition("T10", "raw-bypass", False, REASON_RECONSTRUCTION, "RAW_BYPASS"),
    TopologyDefinition("T11", "dense-state-copy", False, REASON_COLLISION, "DENSE_STATE"),
    TopologyDefinition("T12", "two-seams-one-edge", False, REASON_PULSE, "TWO_IN_ONE_EDGE"),
)


def apply_topology_definition(
    track: CertificateTrack,
    definition: TopologyDefinition,
) -> CertificateTrack:
    """Apply one of the thirteen finite topology operators to a track."""

    phase = _unit_phase(np.asarray(track.phase, dtype=np.float64))
    angle = np.arctan2(phase[:, 1], phase[:, 0]) / (2.0 * np.pi) % 1.0
    operator = definition.operator
    if operator.startswith("rotate:"):
        shift = 0.125 if "+" in operator else -0.125
        angle = (angle + shift) % 1.0
    elif operator == "homeomorphism:H+":
        angle = np.where(angle < 0.5, 7.0 * angle / 8.0, 7.0 / 16.0 + 9.0 * (angle - 0.5) / 8.0)
    elif operator == "homeomorphism:H-":
        angle = np.where(angle < 0.5, 9.0 * angle / 8.0, 9.0 / 16.0 + 7.0 * (angle - 0.5) / 8.0)
    elif operator == "degree-zero":
        angle = np.zeros_like(angle)
    elif operator == "harmonic:2":
        angle = (2.0 * angle) % 1.0
    elif operator == "harmonic:8":
        angle = (8.0 * angle) % 1.0
    elif operator == "reversal":
        angle = (-angle) % 1.0
    elif operator == "alternating-starts":
        landmarks = integer_landmarks(track.geometry, track.run_bounds)
        for traversal, (left, right) in enumerate(zip(landmarks[:-1], landmarks[1:], strict=False)):
            if traversal % 2:
                angle[int(left) : int(right) + 1] = (angle[int(left) : int(right) + 1] + 0.5) % 1.0
    transformed = np.stack((np.cos(2.0 * np.pi * angle), np.sin(2.0 * np.pi * angle)), axis=1)
    return replace(
        track,
        phase=transformed,
        raw_bypass_attack=operator == "raw-bypass",
        dense_collision_attack=operator == "dense-state-copy",
        two_seams_attack=operator == "two-seams-one-edge",
    )


__all__ = [
    "CertificateContractError",
    "CertificateTrack",
    "CertifiedTarget",
    "NaturalTeacherOutput",
    "SourceKind",
    "TOPOLOGY_DEFINITIONS",
    "TopologyDefinition",
    "apply_topology_definition",
    "bind_natural_certificate_track",
    "build_natural_teacher_output",
    "canonicalize_phase",
    "certificate_track_from_x0",
    "certify_target",
    "integer_landmarks",
    "parse_natural_input_receipt_bytes",
    "validate_natural_input_receipt",
]
