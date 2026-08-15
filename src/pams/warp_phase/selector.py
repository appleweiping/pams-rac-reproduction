"""Frozen pseudo-cycle selector for the WARP-PHASE supplied-track pilot.

The selector is deliberately independent of evaluator labels.  It consumes only
pose values, frozen joint masks, and source-frame clocks.  All arithmetic that
defines the selector is performed in float64; the model-facing normalized pose
is exposed separately as float32.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class SelectorError(ValueError):
    """Raised when a fail-closed selector precondition is not satisfied."""


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class NormalizedTrack:
    """One duplicate-collapsed track after the frozen COCO17 normalization."""

    clocks: NDArray[np.int64]
    coordinates: FloatArray
    joint_valid: BoolArray
    feature_frame_valid: BoolArray
    model_pose: NDArray[np.float32]
    scale: float


@dataclass(frozen=True)
class SupportDiagnostics:
    """Deterministic support arrays computed before the final period decision."""

    candidates: NDArray[np.int64]
    acf: FloatArray
    score_by_period: FloatArray
    score_valid: BoolArray
    local_maximum: BoolArray
    fft_power: FloatArray
    harmonic_power: FloatArray
    fft_grid: FloatArray
    fft_mask: BoolArray


@dataclass(frozen=True)
class SupportSelection:
    """A nullable selector decision plus its always-preserved diagnostics."""

    period: int | None
    score: float | None
    diagnostics: SupportDiagnostics

    @property
    def candidates(self) -> NDArray[np.int64]:
        return self.diagnostics.candidates

    @property
    def acf(self) -> FloatArray:
        return self.diagnostics.acf

    @property
    def score_by_period(self) -> FloatArray:
        return self.diagnostics.score_by_period

    @property
    def score_valid(self) -> BoolArray:
        return self.diagnostics.score_valid

    @property
    def local_maximum(self) -> BoolArray:
        return self.diagnostics.local_maximum

    @property
    def fft_power(self) -> FloatArray:
        return self.diagnostics.fft_power

    @property
    def harmonic_power(self) -> FloatArray:
        return self.diagnostics.harmonic_power

    @property
    def fft_grid(self) -> FloatArray:
        return self.diagnostics.fft_grid

    @property
    def fft_mask(self) -> BoolArray:
        return self.diagnostics.fft_mask


@dataclass(frozen=True)
class ParentPerturbation:
    """One fully materialized rejection view, before any selector evaluation."""

    start: int
    length: int
    jitter_raw: FloatArray
    dropout_raw: FloatArray
    weak_coordinates: FloatArray
    weak_joint_valid: BoolArray
    jitter_sha256: str
    dropout_sha256: str
    weak_pose_sha256: str
    weak_mask_sha256: str


@dataclass(frozen=True)
class PreDrawnTrackPerturbations:
    """All ordered draws and hashes for a track, not yet receipt-bound."""

    parents: tuple[ParentPerturbation, ...]


@dataclass(frozen=True)
class FrozenTrackPerturbations:
    """Pre-drawn perturbations bound to the caller's frozen complete-order receipt."""

    parents: tuple[ParentPerturbation, ...]
    order_receipt_sha256: str


@dataclass(frozen=True)
class ParentSelection:
    """Authoritative and rejection-only decisions for one canonical parent."""

    start: int
    length: int
    canonical: int | None
    weak: int | None
    first64: int | None
    last64: int | None
    reversal: int | None
    harmonic_margin: float | None
    accepted: bool
    jitter_sha256: str
    dropout_sha256: str
    weak_pose_sha256: str
    weak_mask_sha256: str


@dataclass(frozen=True)
class TrackSelection:
    """Frozen track pseudo-period and all parent-level audit decisions."""

    period: float
    parents: tuple[ParentSelection, ...]
    order_receipt_sha256: str


def _require_pose_shape(pose: NDArray[np.generic]) -> None:
    if pose.ndim != 3 or pose.shape[1:] != (17, 3):
        raise SelectorError("pose must have shape [K,17,3]")


def normalize_coco17(
    pose: NDArray[np.generic],
    joint_mask: NDArray[np.generic],
    clocks: NDArray[np.generic],
) -> NormalizedTrack:
    """Apply the single frozen root/scale normalization to a complete track."""

    raw = np.asarray(pose, dtype=np.float64)
    mask = np.asarray(joint_mask, dtype=np.bool_)
    q = np.asarray(clocks, dtype=np.int64)
    _require_pose_shape(raw)
    if mask.shape != raw.shape[:2]:
        raise SelectorError("joint_mask must have shape [K,17]")
    if q.ndim != 1 or q.shape[0] != raw.shape[0]:
        raise SelectorError("clocks must have shape [K]")
    if q.size < 2 or np.any(np.diff(q) <= 0):
        raise SelectorError("clocks must be strictly increasing")

    valid = mask & np.isfinite(raw).all(axis=2) & (raw[:, :, 2] >= 0.20)
    roots = np.zeros((raw.shape[0], 2), dtype=np.float64)
    root_valid = np.zeros(raw.shape[0], dtype=np.bool_)
    scales: list[float] = []

    for index in range(raw.shape[0]):
        hip_indices = [joint for joint in (11, 12) if valid[index, joint]]
        if not hip_indices:
            continue
        root = np.mean(raw[index, hip_indices, :2], axis=0, dtype=np.float64)
        roots[index] = root
        root_valid[index] = True
        shoulder_indices = [joint for joint in (5, 6) if valid[index, joint]]
        if not shoulder_indices:
            continue
        shoulder = np.mean(raw[index, shoulder_indices, :2], axis=0, dtype=np.float64)
        distance = float(np.linalg.norm(shoulder - root))
        if math.isfinite(distance) and distance > 0.0:
            scales.append(distance)

    if not scales:
        raise SelectorError("track has no positive finite COCO17 shoulder-to-hip scale")
    scale = max(float(np.median(np.asarray(scales, dtype=np.float64))), 1e-3)
    if not math.isfinite(scale) or scale <= 0.0:
        raise SelectorError("track scale is not positive and finite")

    valid &= root_valid[:, None]
    coordinates = np.zeros((raw.shape[0], 17, 2), dtype=np.float64)
    centered = (raw[:, :, :2] - roots[:, None, :]) / scale
    coordinates[valid] = centered[valid]
    feature_frame_valid = np.sum(valid, axis=1, dtype=np.int64) >= 8

    model_pose = np.zeros_like(raw, dtype=np.float32)
    model_pose[:, :, :2] = coordinates.astype(np.float32)
    confidence = raw[:, :, 2].astype(np.float32)
    model_pose[:, :, 2][valid] = confidence[valid]
    model_pose[~valid] = np.float32(0.0)
    return NormalizedTrack(
        clocks=q,
        coordinates=coordinates,
        joint_valid=valid,
        feature_frame_valid=feature_frame_valid,
        model_pose=model_pose,
        scale=scale,
    )


def canonical_parent_starts(length: int) -> tuple[int, ...]:
    """Return the frozen 128-clock, stride-64 parent starts."""

    if length < 64:
        raise SelectorError("eligible tracks require at least 64 retained clocks")
    if length < 128:
        return (0,)
    starts = list(range(0, length - 127, 64))
    final_start = length - 128
    if final_start not in starts:
        starts.append(final_start)
    return tuple(starts)


@dataclass(frozen=True)
class _VelocityField:
    clocks: FloatArray
    velocity: FloatArray
    valid: BoolArray


def _velocity_field(clocks: FloatArray, coordinates: FloatArray, joint_valid: BoolArray) -> _VelocityField:
    if clocks.ndim != 1 or coordinates.shape != (clocks.size, 17, 2):
        raise SelectorError("normalized support shapes do not agree")
    if joint_valid.shape != (clocks.size, 17):
        raise SelectorError("normalized joint mask shape does not agree")
    delta = np.diff(clocks)
    if clocks.size < 2 or np.any(~np.isfinite(clocks)) or np.any(delta <= 0.0):
        raise SelectorError("support clocks must be finite and strictly increasing")
    flat = coordinates.reshape(clocks.size, 34)
    coord_valid = np.repeat(joint_valid, 2, axis=1)
    cell_valid = coord_valid[:-1] & coord_valid[1:]
    velocity = np.zeros((clocks.size - 1, 34), dtype=np.float64)
    raw_velocity = np.diff(flat, axis=0) / delta[:, None]
    cell_valid &= np.isfinite(raw_velocity)
    velocity[cell_valid] = raw_velocity[cell_valid]
    return _VelocityField(clocks=clocks, velocity=velocity, valid=cell_valid)


def _query_velocity(field: _VelocityField, query: FloatArray) -> tuple[FloatArray, BoolArray]:
    values = np.zeros((query.size, 34), dtype=np.float64)
    valid = np.zeros((query.size, 34), dtype=np.bool_)
    inside = np.isfinite(query) & (query >= field.clocks[0]) & (query <= field.clocks[-1])
    if not np.any(inside):
        return values, valid
    interior_query = query[inside]
    cells = np.searchsorted(field.clocks, interior_query, side="right") - 1
    cells = np.minimum(cells, field.velocity.shape[0] - 1)
    cells = np.maximum(cells, 0)
    values[inside] = field.velocity[cells]
    valid[inside] = field.valid[cells]
    values[~valid] = 0.0
    return values, valid


def _interpolate_power(power: FloatArray, continuous_bin: float) -> float:
    if not math.isfinite(continuous_bin) or continuous_bin <= 0.0 or continuous_bin >= 128.0:
        return 0.0
    lower = int(math.floor(continuous_bin))
    upper = int(math.ceil(continuous_bin))
    if lower == upper:
        return float(power[lower])
    return float((upper - continuous_bin) * power[lower] + (continuous_bin - lower) * power[upper])


def compute_support_diagnostics(
    clocks: NDArray[np.generic],
    coordinates: NDArray[np.generic],
    joint_valid: NDArray[np.generic],
) -> SupportDiagnostics:
    """Compute the frozen ACF/FFT arrays without requiring a final decision."""

    q = np.asarray(clocks, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    mask = np.asarray(joint_valid, dtype=np.bool_)
    field = _velocity_field(q, coords, mask)
    span = float(q[-1] - q[0])
    if not math.isfinite(span) or span <= 0.0:
        raise SelectorError("selector support has non-positive span")
    p_max = min(128, int(math.floor(span / 2.0)))
    if p_max < 4:
        raise SelectorError("selector support has no candidate period")

    fft_grid = q[0] + np.arange(256, dtype=np.float64) * (span / 255.0)
    fft_velocity, fft_mask = _query_velocity(field, fft_grid)
    coordinate_counts = np.sum(fft_mask, axis=0, dtype=np.int64)
    means = np.zeros(34, dtype=np.float64)
    nonempty_coordinates = coordinate_counts > 0
    means[nonempty_coordinates] = (
        np.sum(fft_velocity[:, nonempty_coordinates], axis=0, dtype=np.float64)
        / coordinate_counts[nonempty_coordinates]
    )
    hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(256, dtype=np.float64) / 256.0)
    weighted = fft_mask * hann[:, None] * (fft_velocity - means[None, :])
    weighted[~fft_mask] = 0.0
    fft_denominator = float(np.sum(fft_mask * hann[:, None] ** 2, dtype=np.float64))
    if not math.isfinite(fft_denominator) or fft_denominator <= 0.0:
        raise SelectorError("FFT normalization support is empty")
    spectrum = np.fft.rfft(weighted, n=256, axis=0, norm="forward")
    power = np.sum(np.abs(spectrum) ** 2, axis=1, dtype=np.float64) / fft_denominator
    if power.shape != (129,) or np.any(~np.isfinite(power)):
        raise SelectorError("FFT power is not finite")
    spectral_denominator = float(np.sum(power[1:128], dtype=np.float64))
    if not math.isfinite(spectral_denominator) or spectral_denominator <= 0.0:
        raise SelectorError("spectral normalization is zero")

    candidates = np.arange(4, p_max + 1, dtype=np.int64)
    acf = np.full(candidates.shape, np.nan, dtype=np.float64)
    scores = np.full(candidates.shape, np.nan, dtype=np.float64)
    score_valid = np.zeros(candidates.shape, dtype=np.bool_)
    harmonic_power = np.zeros((candidates.size, 3), dtype=np.float64)
    sample_spacing = span / 255.0

    for index, period_value in enumerate(candidates):
        period = float(period_value)
        pair_mask = fft_grid + period <= q[-1]
        if not np.any(pair_mask):
            continue
        left_query = fft_grid[pair_mask]
        right_query = left_query + period
        left_value, left_valid = _query_velocity(field, left_query)
        right_value, right_valid = _query_velocity(field, right_query)
        pair_valid = left_valid & right_valid
        if int(np.sum(np.any(pair_valid, axis=1), dtype=np.int64)) < 16:
            continue
        numerator = float(np.sum(pair_valid * left_value * right_value, dtype=np.float64))
        left_energy = float(np.sum(pair_valid * left_value**2, dtype=np.float64))
        right_energy = float(np.sum(pair_valid * right_value**2, dtype=np.float64))
        correlation = numerator / (math.sqrt(left_energy * right_energy) + 1e-8)
        if not math.isfinite(correlation):
            continue
        acf[index] = correlation
        continuous_bin = 256.0 * sample_spacing / period
        harmonics = np.asarray(
            [
                _interpolate_power(power, continuous_bin),
                _interpolate_power(power, 2.0 * continuous_bin),
                _interpolate_power(power, 3.0 * continuous_bin),
            ],
            dtype=np.float64,
        )
        harmonic_power[index] = harmonics
        spectral = float((harmonics[0] + 0.5 * harmonics[1] + 0.25 * harmonics[2]) / (1.75 * spectral_denominator + 1e-8))
        score = 0.5 * correlation + 0.5 * spectral
        if math.isfinite(score):
            scores[index] = score
            score_valid[index] = correlation >= 0.25

    local_maximum = np.zeros(candidates.shape, dtype=np.bool_)
    for index in range(candidates.size):
        if not score_valid[index] or not math.isfinite(float(scores[index])):
            continue
        if candidates.size == 1:
            local_maximum[index] = True
        elif index == 0:
            local_maximum[index] = float(scores[index]) >= float(scores[index + 1])
        elif index == candidates.size - 1:
            local_maximum[index] = float(scores[index]) >= float(scores[index - 1])
        else:
            local_maximum[index] = (
                float(scores[index]) >= float(scores[index - 1])
                and float(scores[index]) >= float(scores[index + 1])
            )
    return SupportDiagnostics(
        candidates=candidates,
        acf=acf,
        score_by_period=scores,
        score_valid=score_valid,
        local_maximum=local_maximum,
        fft_power=power[1:128].copy(),
        harmonic_power=harmonic_power,
        fft_grid=fft_grid,
        fft_mask=fft_mask,
    )


def select_support_period(
    clocks: NDArray[np.generic],
    coordinates: NDArray[np.generic],
    joint_valid: NDArray[np.generic],
) -> SupportSelection:
    """Return the nullable final decision while preserving computed diagnostics."""

    diagnostics = compute_support_diagnostics(clocks, coordinates, joint_valid)
    eligible_indices = np.flatnonzero(diagnostics.local_maximum)
    if eligible_indices.size == 0:
        return SupportSelection(period=None, score=None, diagnostics=diagnostics)
    selected_index = min(
        (int(index) for index in eligible_indices),
        key=lambda index: (
            -float(diagnostics.score_by_period[index]),
            int(diagnostics.candidates[index]),
        ),
    )
    return SupportSelection(
        period=int(diagnostics.candidates[selected_index]),
        score=float(diagnostics.score_by_period[selected_index]),
        diagnostics=diagnostics,
    )


def harmonic_margin(selection: SupportSelection) -> float:
    """Return the minimum selected-score margin over integer half/double candidates."""

    if selection.period is None or selection.score is None:
        raise SelectorError("harmonic margin requires a selected local maximum")
    alternatives: list[int] = []
    if selection.period % 2 == 0 and selection.period // 2 >= 4:
        alternatives.append(selection.period // 2)
    if selection.period * 2 <= int(selection.candidates[-1]):
        alternatives.append(selection.period * 2)
    if not alternatives:
        return math.inf
    margins: list[float] = []
    for alternative in alternatives:
        index = alternative - int(selection.candidates[0])
        other_score = float(selection.score_by_period[index])
        if not math.isfinite(other_score):
            raise SelectorError("harmonic comparison candidate lacks a finite score")
        margins.append(selection.score - other_score)
    return min(margins)


def _array_sha256(array: NDArray[np.generic], dtype: str) -> str:
    canonical = np.ascontiguousarray(array, dtype=np.dtype(dtype))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def make_weak_view(
    coordinates: FloatArray,
    joint_valid: BoolArray,
    jitter_raw: FloatArray,
    dropout_raw: FloatArray,
) -> tuple[FloatArray, BoolArray]:
    """Apply the frozen rejection-only jitter/dropout transform."""

    if jitter_raw.shape != (*joint_valid.shape, 2):
        raise SelectorError("weak jitter shape does not match parent")
    if dropout_raw.shape != joint_valid.shape:
        raise SelectorError("weak dropout shape does not match parent")
    weak_coordinates = np.asarray(coordinates, dtype=np.float64).copy()
    weak_mask = np.asarray(joint_valid, dtype=np.bool_).copy()
    clipped = np.clip(np.asarray(jitter_raw, dtype=np.float64), -0.03, 0.03)
    weak_coordinates[weak_mask] += clipped[weak_mask]
    for frame in range(weak_mask.shape[0]):
        canonical_indices = np.flatnonzero(joint_valid[frame])
        dropped = canonical_indices[dropout_raw[frame, canonical_indices] < 0.10]
        weak_mask[frame, dropped] = False
        target = min(8, int(canonical_indices.size))
        if int(np.sum(weak_mask[frame], dtype=np.int64)) < target:
            for joint in dropped:
                weak_mask[frame, joint] = True
                if int(np.sum(weak_mask[frame], dtype=np.int64)) >= target:
                    break
    weak_coordinates[~weak_mask] = 0.0
    return weak_coordinates, weak_mask


def draw_track_perturbations(
    track: NormalizedTrack,
    rng: np.random.Generator,
) -> PreDrawnTrackPerturbations:
    """Consume and hash every parent draw without running the selector.

    The caller must invoke this for all tracks in the contract's complete global
    order, freeze the resulting order receipt, and only then bind each track's
    draws with :func:`freeze_track_perturbations`.
    """

    parents: list[ParentPerturbation] = []
    for start in canonical_parent_starts(int(track.clocks.size)):
        stop = min(start + 128, int(track.clocks.size))
        length = stop - start
        jitter_raw = np.asarray(
            rng.normal(0.0, 0.01, size=(length, 17, 2)),
            dtype=np.float64,
        )
        dropout_raw = np.asarray(
            rng.random(size=(length, 17)),
            dtype=np.float64,
        )
        weak_coordinates, weak_mask = make_weak_view(
            track.coordinates[start:stop],
            track.joint_valid[start:stop],
            jitter_raw,
            dropout_raw,
        )
        weak_pose = np.zeros((length, 17, 3), dtype=np.float32)
        weak_pose[:, :, :2] = weak_coordinates.astype(np.float32)
        weak_pose[:, :, 2][weak_mask] = track.model_pose[start:stop, :, 2][weak_mask]
        hashes = (
            _array_sha256(jitter_raw, "<f8"),
            _array_sha256(dropout_raw, "<f8"),
            _array_sha256(weak_pose, "<f4"),
            _array_sha256(weak_mask, "|u1"),
        )
        for array in (jitter_raw, dropout_raw, weak_coordinates, weak_mask):
            array.flags.writeable = False
        parents.append(
            ParentPerturbation(
                start=start,
                length=length,
                jitter_raw=jitter_raw,
                dropout_raw=dropout_raw,
                weak_coordinates=weak_coordinates,
                weak_joint_valid=weak_mask,
                jitter_sha256=hashes[0],
                dropout_sha256=hashes[1],
                weak_pose_sha256=hashes[2],
                weak_mask_sha256=hashes[3],
            )
        )
    return PreDrawnTrackPerturbations(parents=tuple(parents))


def freeze_track_perturbations(
    pre_drawn: PreDrawnTrackPerturbations,
    *,
    order_receipt_sha256: str,
) -> FrozenTrackPerturbations:
    """Bind pre-drawn bytes to a caller-frozen complete global order receipt."""

    if len(order_receipt_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in order_receipt_sha256
    ):
        raise SelectorError("order receipt must be one lowercase SHA-256")
    return FrozenTrackPerturbations(
        parents=pre_drawn.parents,
        order_receipt_sha256=order_receipt_sha256,
    )


def _safe_period(
    clocks: FloatArray,
    coordinates: FloatArray,
    joint_valid: BoolArray,
) -> SupportSelection | None:
    try:
        return select_support_period(clocks, coordinates, joint_valid)
    except SelectorError:
        return None


def _within_ten_percent(value: int | None, reference: int) -> bool:
    return value is not None and 10 * abs(value - reference) <= reference


def select_track_period(
    track: NormalizedTrack,
    perturbations: FrozenTrackPerturbations,
) -> TrackSelection:
    """Evaluate one track only from receipt-bound, already materialized draws."""

    if not isinstance(perturbations, FrozenTrackPerturbations):
        raise SelectorError("track evaluation requires frozen pre-drawn perturbations")
    expected_starts = canonical_parent_starts(int(track.clocks.size))
    if tuple(parent.start for parent in perturbations.parents) != expected_starts:
        raise SelectorError("pre-drawn parent order differs from canonical parent starts")
    parent_results: list[ParentSelection] = []
    accepted_periods: list[float] = []
    for parent in perturbations.parents:
        start = parent.start
        stop = min(start + 128, int(track.clocks.size))
        q = track.clocks[start:stop].astype(np.float64)
        coordinates = track.coordinates[start:stop]
        joint_valid = track.joint_valid[start:stop]
        length = int(q.size)
        if parent.length != length:
            raise SelectorError("pre-drawn parent length differs from canonical support")
        weak_coordinates, weak_mask = make_weak_view(
            coordinates,
            joint_valid,
            parent.jitter_raw,
            parent.dropout_raw,
        )
        if not np.array_equal(weak_coordinates, parent.weak_coordinates) or not np.array_equal(
            weak_mask,
            parent.weak_joint_valid,
        ):
            raise SelectorError("pre-drawn weak view differs from its frozen materialization")
        weak_pose = np.zeros((length, 17, 3), dtype=np.float32)
        weak_pose[:, :, :2] = weak_coordinates.astype(np.float32)
        weak_pose[:, :, 2][weak_mask] = track.model_pose[start:stop, :, 2][weak_mask]
        observed_hashes = (
            _array_sha256(parent.jitter_raw, "<f8"),
            _array_sha256(parent.dropout_raw, "<f8"),
            _array_sha256(weak_pose, "<f4"),
            _array_sha256(weak_mask, "|u1"),
        )
        expected_hashes = (
            parent.jitter_sha256,
            parent.dropout_sha256,
            parent.weak_pose_sha256,
            parent.weak_mask_sha256,
        )
        if observed_hashes != expected_hashes:
            raise SelectorError("pre-drawn perturbation hash differs from frozen materialization")

        canonical_result = _safe_period(q, coordinates, joint_valid)
        weak_result = _safe_period(q, weak_coordinates, weak_mask)
        first_result = _safe_period(q[:64], coordinates[:64], joint_valid[:64])
        last_result = _safe_period(q[-64:], coordinates[-64:], joint_valid[-64:])
        reverse_q = q[0] + (q[-1] - q[::-1])
        reverse_result = _safe_period(
            reverse_q,
            coordinates[::-1],
            joint_valid[::-1],
        )

        canonical_period = None if canonical_result is None else canonical_result.period
        weak_period = None if weak_result is None else weak_result.period
        first_period = None if first_result is None else first_result.period
        last_period = None if last_result is None else last_result.period
        reversal_period = None if reverse_result is None else reverse_result.period
        margin: float | None = None
        if canonical_result is not None and canonical_result.period is not None:
            try:
                margin = harmonic_margin(canonical_result)
            except SelectorError:
                margin = None
        accepted = bool(
            canonical_period is not None
            and _within_ten_percent(weak_period, canonical_period)
            and _within_ten_percent(first_period, canonical_period)
            and _within_ten_percent(last_period, canonical_period)
            and _within_ten_percent(reversal_period, canonical_period)
            and margin is not None
            and margin >= 0.10
        )
        if accepted and canonical_period is not None:
            accepted_periods.append(float(canonical_period))
        parent_results.append(
            ParentSelection(
                start=start,
                length=length,
                canonical=canonical_period,
                weak=weak_period,
                first64=first_period,
                last64=last_period,
                reversal=reversal_period,
                harmonic_margin=margin,
                accepted=accepted,
                jitter_sha256=parent.jitter_sha256,
                dropout_sha256=parent.dropout_sha256,
                weak_pose_sha256=parent.weak_pose_sha256,
                weak_mask_sha256=parent.weak_mask_sha256,
            )
        )
    if not accepted_periods:
        raise SelectorError("no canonical parent survives all rejection checks")
    period = float(np.median(np.asarray(accepted_periods, dtype=np.float64)))
    return TrackSelection(
        period=period,
        parents=tuple(parent_results),
        order_receipt_sha256=perturbations.order_receipt_sha256,
    )
