"""Deterministic TempoRAC X0 primitive-orbit population.

This module is deliberately self-contained.  It generates only the synthetic
``temporac.execution.v4`` population and has no natural-data or evaluator
entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, TypeVar, cast

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree
from scipy.special import i0

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
UInt8Array = NDArray[np.uint8]
Resampler = Literal["linear", "pchip", "sinc"]
X0Split = Literal["train", "tune", "heldout"]
_DType = TypeVar("_DType", bound=np.generic)

SOURCE_COUNT = 40
TRAVERSAL_COUNT = 10
BLOCK_COUNT = 7
OFFSETS = tuple(range(32))
STAT_PERIODS = (5, 8, 16, 32, 64, 127, 128)
MOVING_JOINTS = (7, 8, 9, 10, 13, 14, 15, 16)
X0_SCALE = np.float64(25.0 / 6.0)
BONES = (
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

BASE_POSE = np.asarray(
    [
        (0.00, 0.90),
        (-0.05, 0.95),
        (0.05, 0.95),
        (-0.10, 0.93),
        (0.10, 0.93),
        (-0.18, 0.65),
        (0.18, 0.65),
        (-0.30, 0.40),
        (0.30, 0.40),
        (-0.38, 0.15),
        (0.38, 0.15),
        (-0.12, 0.00),
        (0.12, 0.00),
        (-0.14, -0.40),
        (0.14, -0.40),
        (-0.15, -0.85),
        (0.15, -0.85),
    ],
    dtype=np.float64,
)


class X0ContractError(ValueError):
    """Raised when a deterministic X0 contract check fails."""


def _readonly(value: NDArray[_DType]) -> NDArray[_DType]:
    array: NDArray[_DType] = np.ascontiguousarray(value)
    immutable = np.frombuffer(array.tobytes(order="C"), dtype=array.dtype)
    return cast(NDArray[_DType], immutable.reshape(array.shape))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _seed(domain: bytes, source_id: int, attempt: int = 0, view: int = 0) -> int:
    preimage = domain + b"\0" + struct.pack(">III", source_id, attempt, view)
    return int.from_bytes(hashlib.sha256(preimage).digest()[:8], "little", signed=False)


def source_key(source_id: int) -> bytes:
    """Return the exact 32-byte X0 source-unit key."""

    if not 0 <= source_id < SOURCE_COUNT:
        raise X0ContractError("source_id must be in [0, 40)")
    return hashlib.sha256(b"temporac.x0-unit.v4\0" + struct.pack(">I", source_id)).digest()


def source_split(source_id: int) -> X0Split:
    if not 0 <= source_id < SOURCE_COUNT:
        raise X0ContractError("source_id must be in [0, 40)")
    if source_id < 24:
        return "train"
    if source_id < 32:
        return "tune"
    return "heldout"


def pose_geometry(pose: FloatArray) -> FloatArray:
    """Map ``[...,17,2]`` COCO poses to root-relative ``[...,66]`` geometry."""

    values = np.asarray(pose, dtype=np.float64)
    if values.shape[-2:] != (17, 2):
        raise X0ContractError("pose must end in [17,2]")
    root = (values[..., 11, :] + values[..., 12, :]) / 2.0
    centered = values - root[..., None, :]
    bones = np.stack([centered[..., b, :] - centered[..., a, :] for a, b in BONES], axis=-2)
    return np.concatenate(
        (centered.reshape(*centered.shape[:-2], 34), bones.reshape(*bones.shape[:-2], 32)), axis=-1
    )


def _landmark_signs() -> FloatArray:
    bits: list[float] = []
    counter = 0
    while len(bits) < 66:
        digest = hashlib.sha256(b"temporac.landmark.v4\0" + struct.pack(">I", counter)).digest()
        for byte in digest:
            for shift in range(7, -1, -1):
                bits.append(1.0 if (byte >> shift) & 1 else -1.0)
                if len(bits) == 66:
                    break
            if len(bits) == 66:
                break
        counter += 1
    return np.asarray(bits, dtype=np.float64)


LANDMARK_RHO = _landmark_signs()


def _moving_map() -> FloatArray:
    base = pose_geometry(BASE_POSE)
    columns: list[FloatArray] = []
    for coordinate in range(16):
        displaced = BASE_POSE.copy()
        joint = MOVING_JOINTS[coordinate // 2]
        axis = coordinate % 2
        displaced[joint, axis] += 1.0
        columns.append(pose_geometry(displaced) - base)
    return np.stack(columns, axis=1)


MOVING_TO_GEOMETRY = _moving_map()
_raw_w = MOVING_TO_GEOMETRY.T @ LANDMARK_RHO
_w_norm = float(np.linalg.norm(_raw_w))
if not math.isfinite(_w_norm) or _w_norm <= 0.0:  # pragma: no cover - frozen constant guard
    raise RuntimeError("frozen X0 landmark direction has zero norm")
LANDMARK_DIRECTION = _raw_w / _w_norm


def _neumaier_sum(values: NDArray[np.float64]) -> float:
    total = 0.0
    correction = 0.0
    for item in np.ravel(values, order="C"):
        value = float(item)
        updated = total + value
        if abs(total) >= abs(value):
            correction += (total - updated) + value
        else:
            correction += (value - updated) + total
        total = updated
    return total + correction


def _dot(left: FloatArray, right: FloatArray) -> float:
    return _neumaier_sum(np.asarray(left * right, dtype=np.float64))


def _norm(value: FloatArray) -> float:
    return math.sqrt(max(0.0, _dot(value, value)))


def _orthonormal_coefficients(source_id: int, attempt: int) -> tuple[int, FloatArray]:
    seed = _seed(b"temporac.x0.coefficient.v4", source_id, attempt, 0)
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = rng.uniform(-1.0, 1.0, size=(6, 16)).astype(np.float64, copy=False)
    accepted: list[FloatArray] = []
    for draw in draws:
        vector = draw.copy()
        for _ in range(2):
            for basis in (LANDMARK_DIRECTION, *accepted):
                vector -= _dot(vector, basis) * basis
        norm = _norm(vector)
        if not math.isfinite(norm) or norm <= 1e-10:
            raise X0ContractError("coefficient Gram-Schmidt rejection")
        accepted.append(vector / norm)
    return seed, np.stack(accepted, axis=0)


def _moving_displacement(phi: FloatArray, coefficients: FloatArray) -> FloatArray:
    phase = np.asarray(phi, dtype=np.float64)
    flat = phase.reshape(-1)
    displacement = 0.12 * np.sin(2.0 * np.pi * flat)[:, None] * LANDMARK_DIRECTION
    for index, harmonic in enumerate((2, 3, 4)):
        u_h = coefficients[2 * index]
        v_h = coefficients[2 * index + 1]
        displacement += (0.03 / harmonic) * (
            np.cos(2.0 * np.pi * harmonic * flat)[:, None] * u_h
            + np.sin(2.0 * np.pi * harmonic * flat)[:, None] * v_h
        )
    displacement *= X0_SCALE
    return displacement.reshape(*phase.shape, 16)


def _moving_tangent(phi: FloatArray, coefficients: FloatArray) -> FloatArray:
    """Return the analytic derivative of F8 with respect to phase cycles."""

    phase = np.asarray(phi, dtype=np.float64)
    flat = phase.reshape(-1)
    tangent = 0.24 * np.pi * np.cos(2.0 * np.pi * flat)[:, None] * LANDMARK_DIRECTION
    for index, harmonic in enumerate((2, 3, 4)):
        u_h = coefficients[2 * index]
        v_h = coefficients[2 * index + 1]
        tangent += (
            0.06
            * np.pi
            * (
                -np.sin(2.0 * np.pi * harmonic * flat)[:, None] * u_h
                + np.cos(2.0 * np.pi * harmonic * flat)[:, None] * v_h
            )
        )
    tangent *= X0_SCALE
    return tangent.reshape(*phase.shape, 16)


def evaluate_orbit(coefficients: FloatArray, phi: FloatArray | float) -> FloatArray:
    """Evaluate the frozen clean COCO-17 orbit at phase values in cycles."""

    phases = np.asarray(phi, dtype=np.float64)
    displacement = _moving_displacement(phases, coefficients)
    result = np.broadcast_to(BASE_POSE, (*phases.shape, 17, 2)).copy()
    for local_index, joint in enumerate(MOVING_JOINTS):
        result[..., joint, :] += displacement[..., 2 * local_index : 2 * local_index + 2]
    return result


def evaluate_orbit_tangent(coefficients: FloatArray, phi: FloatArray | float) -> FloatArray:
    """Evaluate the exact analytic F8 pose tangent with respect to phase cycles."""

    phases = np.asarray(phi, dtype=np.float64)
    moving_tangent = _moving_tangent(phases, coefficients)
    result = np.zeros((*phases.shape, 17, 2), dtype=np.float64)
    for local_index, joint in enumerate(MOVING_JOINTS):
        result[..., joint, :] = moving_tangent[..., 2 * local_index : 2 * local_index + 2]
    return result


def evaluate_confidence(phi: FloatArray | float) -> FloatArray:
    phases = np.asarray(phi, dtype=np.float64)
    joints = np.arange(17, dtype=np.float64)
    raw = 0.90 + 0.05 * np.sin(2.0 * np.pi * phases[..., None] + 2.0 * np.pi * joints / 17.0)
    return np.clip(raw, 0.0, 1.0)


def _candidate_passes(coefficients: FloatArray) -> bool:
    phase = np.arange(4096, dtype=np.float64) / 4096.0
    pose = np.ascontiguousarray(evaluate_orbit(coefficients, phase), dtype=np.float64)
    if not np.isfinite(pose).all() or np.any(np.abs(pose) > 2.0):
        return False
    geometry = pose_geometry(pose)
    shoulder = (pose[:, 5] + pose[:, 6]) / 2.0
    hip = (pose[:, 11] + pose[:, 12]) / 2.0
    if float(np.min(np.linalg.norm(shoulder - hip, axis=1))) < 0.1:
        return False
    closed = np.concatenate((geometry, geometry[:1]), axis=0)
    if float(np.sum(np.linalg.norm(np.diff(closed, axis=0), axis=1))) < 1.0:
        return False

    tangent = evaluate_orbit_tangent(coefficients, phase)
    if float(np.min(np.linalg.norm(tangent.reshape(4096, -1), axis=1))) < 1e-6:
        return False

    for divisor in range(2, 9):
        shifted = np.ascontiguousarray(
            evaluate_orbit(coefficients, (phase + 1.0 / divisor) % 1.0),
            dtype=np.float64,
        )
        difference = pose - shifted
        squared_difference = np.square(difference)
        mean_square = float(np.mean(squared_difference, dtype=np.float64))
        rms = math.sqrt(mean_square)
        if rms <= 0.05:
            return False

    flattened = pose.reshape(4096, -1)
    pairs = cKDTree(flattened).query_pairs(r=1e-3 * math.sqrt(flattened.shape[1]))
    for left, right in pairs:
        separation = abs(left - right) / 4096.0
        if min(separation, 1.0 - separation) >= 0.10:
            return False

    reversed_pose = pose[(-np.arange(4096)) % 4096]
    fft_pose = np.fft.rfft(flattened, axis=0)
    fft_reverse = np.fft.rfft(reversed_pose.reshape(4096, -1), axis=0)
    correlation = np.fft.irfft(np.sum(np.conjugate(fft_pose) * fft_reverse, axis=1), n=4096)
    energy = float(np.sum(np.square(flattened)))
    rms_by_shift = np.sqrt(np.maximum(0.0, (2.0 * energy - 2.0 * correlation) / flattened.size))
    if float(np.min(rms_by_shift)) <= 0.01:
        return False

    score = geometry @ LANDMARK_RHO / math.sqrt(66.0)
    if (
        np.count_nonzero(score == np.max(score)) != 1
        or np.count_nonzero(score == np.min(score)) != 1
    ):
        return False
    high = score >= float(np.max(score) - 0.05 * (np.max(score) - np.min(score)))
    starts = np.count_nonzero(high & ~np.roll(high, 1))
    return int(starts) == 1


@dataclass(frozen=True, slots=True)
class X0Orbit:
    source_id: int
    split: X0Split
    source_key: bytes
    accepted_attempt: int
    coefficient_seed_u64: int
    coefficients: FloatArray
    coefficient_sha256: str
    orbit_grid_sha256: str

    @property
    def name(self) -> str:
        return f"U{self.source_id:04d}"


@lru_cache(maxsize=SOURCE_COUNT)
def generate_orbit(source_id: int) -> X0Orbit:
    """Generate and validate one source orbit, retaining the first passing attempt."""

    if not 0 <= source_id < SOURCE_COUNT:
        raise X0ContractError("source_id must be in [0, 40)")
    for attempt in range(64):
        try:
            seed, coefficients = _orthonormal_coefficients(source_id, attempt)
        except X0ContractError:
            continue
        if not _candidate_passes(coefficients):
            continue
        grid = evaluate_orbit(coefficients, np.arange(4096, dtype=np.float64) / 4096.0)
        coefficients_le = np.asarray(coefficients, dtype="<f8", order="C")
        grid_le = np.asarray(grid, dtype="<f8", order="C")
        return X0Orbit(
            source_id=source_id,
            split=source_split(source_id),
            source_key=source_key(source_id),
            accepted_attempt=attempt,
            coefficient_seed_u64=seed,
            coefficients=_readonly(coefficients_le),
            coefficient_sha256=_sha256(coefficients_le.tobytes(order="C")),
            orbit_grid_sha256=_sha256(grid_le.tobytes(order="C")),
        )
    raise X0ContractError(f"all 64 coefficient attempts failed for U{source_id:04d}")


@dataclass(frozen=True, slots=True)
class X0Block:
    index: int
    name: str
    durations: tuple[int, ...]


_DRIFT_PATTERNS = ("SMF", "SFM", "MSF", "MFS", "FSM", "FMS")
_TEMPO_DURATION = {"S": 64, "M": 32, "F": 16}


def blocks_for_source(source_id: int) -> tuple[X0Block, ...]:
    if not 0 <= source_id < SOURCE_COUNT:
        raise X0ContractError("source_id must be in [0, 40)")
    stationary = STAT_PERIODS[source_id % len(STAT_PERIODS)]
    blocks = [X0Block(0, f"stationary-{stationary}", (stationary,) * 10)]
    for index, pattern in enumerate(_DRIFT_PATTERNS, start=1):
        symbols = (pattern * 3) + pattern[0]
        blocks.append(X0Block(index, pattern, tuple(_TEMPO_DURATION[symbol] for symbol in symbols)))
    return tuple(blocks)


def _direct_map(durations: tuple[int, ...], offset: int) -> tuple[IntArray, FloatArray, IntArray]:
    if len(durations) != TRAVERSAL_COUNT or any(duration <= 0 for duration in durations):
        raise X0ContractError("durations must contain ten positive integers")
    if offset not in OFFSETS:
        raise X0ContractError("offset must be in [0, 32)")
    boundaries = np.empty(11, dtype=np.int64)
    boundaries[0] = 32 + offset
    boundaries[1:] = boundaries[0] + np.cumsum(np.asarray(durations, dtype=np.int64))
    terminal = int(boundaries[-1] + 32)
    clock = np.arange(terminal + 1, dtype=np.int64)
    u = np.empty(clock.size, dtype=np.float64)
    before = clock < boundaries[0]
    u[before] = -8.0 + 16.0 * clock[before] / (32.0 + offset)
    for traversal, duration in enumerate(durations):
        owned = (clock >= boundaries[traversal]) & (clock < boundaries[traversal + 1])
        u[owned] = 8.0 + 32.0 * traversal + 32.0 * (clock[owned] - boundaries[traversal]) / duration
    after = clock >= boundaries[-1]
    u[after] = 328.0 + (clock[after] - boundaries[-1]) / 2.0
    return clock, u, boundaries


def _sample_knots(
    values: FloatArray, knots: IntArray, query: FloatArray, resampler: Resampler
) -> FloatArray:
    flat = values.reshape(values.shape[0], -1)
    if resampler == "linear":
        result = np.stack(
            [np.interp(query, knots, flat[:, column]) for column in range(flat.shape[1])], axis=1
        )
    elif resampler == "pchip":
        result = np.asarray(PchipInterpolator(knots, flat, axis=0, extrapolate=False)(query))
    elif resampler == "sinc":
        result = np.empty((query.size, flat.shape[1]), dtype=np.float64)
        lower = int(knots[0])
        for row, point in enumerate(query):
            taps = np.arange(math.floor(float(point)) - 31, math.floor(float(point)) + 32)
            if int(taps[0]) < lower or int(taps[-1]) > int(knots[-1]):
                raise X0ContractError("sinc tap escaped guarded knot bank")
            distance = point - taps
            weights = (
                np.sinc(distance) * i0(8.6 * np.sqrt(1.0 - np.square(distance / 32.0))) / i0(8.6)
            )
            denominator = _neumaier_sum(np.asarray(weights, dtype=np.float64))
            if not math.isfinite(denominator) or denominator == 0.0:
                raise X0ContractError("nonfinite or zero sinc normalizer")
            result[row] = (weights[:, None] * flat[taps - lower]).sum(axis=0) / denominator
    else:  # pragma: no cover - Literal plus runtime guard
        raise X0ContractError(f"unknown resampler {resampler!r}")
    if not np.isfinite(result).all():
        raise X0ContractError("resampler emitted nonfinite values")
    return result.reshape(query.size, *values.shape[1:])


def window_starts(sample_count: int) -> tuple[int, ...]:
    if sample_count < 2:
        raise X0ContractError("a view needs at least two samples")
    if sample_count <= 128:
        return (0,)
    starts = list(range(0, sample_count - 128 + 1, 32))
    terminal = sample_count - 128
    if starts[-1] != terminal:
        starts.append(terminal)
    return tuple(starts)


def _analytic_responsibility(
    u: FloatArray,
    geometry: FloatArray,
) -> tuple[IntArray, FloatArray, FloatArray]:
    starts = window_starts(int(u.size))
    rates: list[float] = []
    betas: list[float] = []
    cover = np.zeros(u.size - 1, dtype=np.int64)
    for start in starts:
        stop = min(u.size, start + 128)
        cover[start : stop - 1] += 1
    lengths = np.linalg.norm(np.diff(geometry, axis=0), axis=1)
    for start in starts:
        stop = min(u.size, start + 128)
        edges = np.arange(start, stop - 1)
        midpoint = edges.astype(np.float64) + 0.5
        if midpoint.size < 3:
            raise X0ContractError("analytic responsibility window has fewer than three edges")
        trapezoid = np.empty(midpoint.size, dtype=np.float64)
        trapezoid[0] = (midpoint[1] - midpoint[0]) / 2.0
        trapezoid[-1] = (midpoint[-1] - midpoint[-2]) / 2.0
        trapezoid[1:-1] = (midpoint[2:] - midpoint[:-2]) / 2.0
        hann = np.sin(np.pi * (midpoint - midpoint[0]) / (midpoint[-1] - midpoint[0])) ** 2
        quadrature = trapezoid * hann
        total = _neumaier_sum(quadrature)
        if total <= 0.0:
            raise X0ContractError("analytic quadrature has zero weight")
        log_rate = np.log(np.diff(u)[edges])
        rates.append(_neumaier_sum(quadrature * log_rate) / total)
        betas.append(_neumaier_sum(quadrature * lengths[edges] / cover[edges].astype(np.float64)))
    order = sorted(range(len(starts)), key=lambda index: (rates[index], starts[index]))
    half = sum(betas) / 2.0
    cumulative = 0.0
    reference = rates[order[-1]]
    for index in order:
        cumulative += betas[index]
        if cumulative >= half:
            reference = rates[index]
            break
    centers = np.asarray((-math.log(1.5), 0.0, math.log(1.5)), dtype=np.float64)
    sigma = math.log(1.5) / 2.0
    p_star = np.empty((len(starts), 3), dtype=np.float64)
    for row, rate in enumerate(rates):
        logits = -np.square(rate - reference - centers) / (2.0 * sigma * sigma)
        logits -= float(np.max(logits))
        probability = np.exp(logits)
        p_star[row] = probability / float(np.sum(probability))
    taper_sum = np.zeros(u.size - 1, dtype=np.float64)
    responsibility_sum = np.zeros((u.size - 1, 3), dtype=np.float64)
    for row, start in enumerate(starts):
        stop = min(u.size, start + 128)
        local = np.arange(stop - start - 1, dtype=np.float64)
        taper = 1e-3 + (1.0 - 1e-3) * np.sin(np.pi * (local + 0.5) / 127.0) ** 2
        taper_sum[start : stop - 1] += taper
        responsibility_sum[start : stop - 1] += taper[:, None] * p_star[row]
    edge_responsibility = responsibility_sum / taper_sum[:, None]
    return np.asarray(starts, dtype=np.int64), p_star, edge_responsibility


@dataclass(frozen=True, slots=True)
class X0View:
    source_id: int
    source_key: bytes
    block: X0Block
    resampler: Resampler
    offset: int
    clock: IntArray
    clean_coordinate: FloatArray
    motion: NDArray[np.float32]
    geometry: FloatArray
    phase: FloatArray
    pulse: UInt8Array
    chi: FloatArray
    target_mask: UInt8Array
    traversal_bounds: NDArray[np.int32]
    window_start: IntArray
    p_star: FloatArray
    edge_responsibility: FloatArray

    @property
    def event_count(self) -> int:
        return int(np.sum(self.pulse, dtype=np.int64))


def generate_view(
    source_id: int,
    block_index: int,
    *,
    resampler: Resampler = "linear",
    offset: int = 0,
) -> X0View:
    """Generate one F9/F10 X0 view with guards, masks, pulses, and responsibilities."""

    orbit = generate_orbit(source_id)
    blocks = blocks_for_source(source_id)
    if not 0 <= block_index < len(blocks):
        raise X0ContractError("block_index must be in [0, 7)")
    block = blocks[block_index]
    clock, clean_coordinate, boundaries = _direct_map(block.durations, offset)
    knot = np.arange(-39, 376, dtype=np.int64)
    knot_pose = evaluate_orbit(orbit.coefficients, knot.astype(np.float64) / 32.0)
    knot_confidence = evaluate_confidence(knot.astype(np.float64) / 32.0)
    pose = _sample_knots(knot_pose, knot, clean_coordinate, resampler)
    confidence = np.clip(
        _sample_knots(knot_confidence, knot, clean_coordinate, resampler), 0.0, 1.0
    )
    motion = np.concatenate((pose, confidence[..., None]), axis=-1).astype("<f4")
    geometry = pose_geometry(pose)
    theta = clean_coordinate / 32.0 - 0.25
    phase = np.stack((np.cos(2.0 * np.pi * theta), np.sin(2.0 * np.pi * theta)), axis=1)
    edge_count = clock.size - 1
    target_mask = np.zeros(edge_count, dtype=np.uint8)
    target_mask[int(boundaries[0]) : int(boundaries[-1])] = 1
    pulse = np.zeros(edge_count, dtype=np.uint8)
    pulse[boundaries[:-1]] = 1
    chi = np.zeros(edge_count, dtype=np.float64)
    traversal_bounds = np.empty((10, 2), dtype=np.int32)
    for traversal, duration in enumerate(block.durations):
        left = int(boundaries[traversal])
        right = int(boundaries[traversal + 1])
        traversal_bounds[traversal] = (left, right)
        chi[left:right] = 1.0 / duration
    starts, p_star, edge_responsibility = _analytic_responsibility(clean_coordinate, geometry)
    return X0View(
        source_id=source_id,
        source_key=orbit.source_key,
        block=block,
        resampler=resampler,
        offset=offset,
        clock=_readonly(np.asarray(clock, dtype="<i8")),
        clean_coordinate=_readonly(np.asarray(clean_coordinate, dtype="<f8")),
        motion=_readonly(np.asarray(motion, dtype="<f4")),
        geometry=_readonly(np.asarray(geometry, dtype="<f8")),
        phase=_readonly(np.asarray(phase, dtype="<f8")),
        pulse=_readonly(pulse),
        chi=_readonly(np.asarray(chi, dtype="<f8")),
        target_mask=_readonly(target_mask),
        traversal_bounds=_readonly(traversal_bounds),
        window_start=_readonly(np.asarray(starts, dtype="<i8")),
        p_star=_readonly(np.asarray(p_star, dtype="<f8")),
        edge_responsibility=_readonly(np.asarray(edge_responsibility, dtype="<f8")),
    )


def x0_manifest(generator_contract_sha256: str) -> bytes:
    """Return canonical LF-terminated JSON for the exact forty-orbit manifest."""

    if type(generator_contract_sha256) is not str or len(generator_contract_sha256) != 64:
        raise X0ContractError("generator contract SHA-256 must be 64 lowercase hex characters")
    try:
        generator_contract_digest = bytes.fromhex(generator_contract_sha256)
    except ValueError as error:
        raise X0ContractError(
            "generator contract SHA-256 must be 64 lowercase hex characters"
        ) from error
    if (
        len(generator_contract_digest) != 32
        or generator_contract_sha256 != generator_contract_sha256.lower()
    ):
        raise X0ContractError("generator contract SHA-256 must be 64 lowercase hex characters")
    rows = []
    for source_id in range(SOURCE_COUNT):
        orbit = generate_orbit(source_id)
        rows.append(
            {
                "accepted_attempt": orbit.accepted_attempt,
                "coefficient_bytes_sha256": orbit.coefficient_sha256,
                "coefficient_seed_u64": orbit.coefficient_seed_u64,
                "generator_contract_sha256": generator_contract_sha256,
                "id": orbit.name,
                "orbit_grid_sha256": orbit.orbit_grid_sha256,
                "source_key_hex": orbit.source_key.hex(),
                "split": orbit.split,
            }
        )
    return (json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "utf-8"
    )


__all__ = [
    "BASE_POSE",
    "BLOCK_COUNT",
    "BONES",
    "LANDMARK_DIRECTION",
    "LANDMARK_RHO",
    "MOVING_JOINTS",
    "OFFSETS",
    "SOURCE_COUNT",
    "STAT_PERIODS",
    "TRAVERSAL_COUNT",
    "X0_SCALE",
    "X0Block",
    "X0ContractError",
    "X0Orbit",
    "X0View",
    "blocks_for_source",
    "evaluate_confidence",
    "evaluate_orbit",
    "evaluate_orbit_tangent",
    "generate_orbit",
    "generate_view",
    "pose_geometry",
    "source_key",
    "source_split",
    "window_starts",
    "x0_manifest",
]
