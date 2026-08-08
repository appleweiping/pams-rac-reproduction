"""Period-adaptive, segment-local spectral single-expert readout v2.

This is an independently inferred successor to the frozen v1 science core.
It is not an author-disclosed PAMS period head.  The v1 synthetic outcome was
``0/32 DENIED``; the hashes below bind that negative result as lineage only.

The v2 ordering is deliberately non-circular:

1. analyze the complete authorized segment once and propose a frequency;
2. derive one local scale from that proposal and the frozen candidate;
3. classify every fixed-grid window as periodic, static-zero-rate, or ambiguous;
4. robustly fuse overlapping window rates on source-frame intervals; and
5. integrate exactly ``L - 1`` interval rates and round once at video level.

No result-dependent scale change, activity trimming, expert vote, target label,
or multi-segment count sum exists in this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import model_validator
from scipy.stats import beta as beta_distribution

from pams.config import StrictModel
from pams.single_expert_readout import AuthorizedSegment, RepresentationVideo

PREDECESSOR_DENIAL_RESULT_SHA256 = (
    "be2fe34f74b0ab2d9187edd3fcf955db1d977d9a5c32743b10a9250858685821"
)
PREDECESSOR_DENIAL_RECEIPT_SHA256 = (
    "562543f75d0533687ebf947439b6ef5ab4da103cc2ad6e112a99e91d5f585480"
)
PREDECESSOR_DENIAL_RESULT_BYTES = 102245
PREDECESSOR_DENIAL_RECEIPT_BYTES = 1273

WindowState = Literal["periodic", "static_zero_rate", "ambiguous"]
ProposalState = Literal["periodic", "static_zero_rate", "ambiguous"]
EstimateStatus = Literal["eligible", "abstain", "undefined_multi_segment"]
MapRole = Literal["map1", "map2"]

_ACTION_JOINTS: tuple[int, ...] = (7, 8, 9, 10, 13, 14, 15, 16)
_EPSILON = np.finfo(np.float64).eps
_POSITIVE_FAMILY_COUNTS: tuple[tuple[str, int], ...] = (
    ("active_support", 8),
    ("constant_tempo_count", 26),
    ("corruption", 24),
    ("duration_count", 12),
    ("harmonic_stress", 24),
    ("variable_tempo", 24),
)
_NULL_FAMILIES: tuple[str, ...] = (
    "constant",
    "drift_only",
    "white_noise",
    "random_walk",
    "ar1_rho_0.9",
    "time_shuffled_periodic",
    "independent_incoherent_frequency_phase",
    "constant_coordinates_oscillating_masks",
)
_INVARIANCE_NAMES: tuple[str, ...] = (
    "reverse",
    "sign_flip",
    "feature_permutation",
    "orthogonal_mixing",
    "scale",
)
_DIAGNOSTIC_STAGE_ORDER: tuple[str, ...] = (
    "informative_dimensions",
    "global_proposal",
    "window_state",
    "candidate_peaks",
    "signed_vector_acf",
    "interval_coverage",
    "integrated_count",
)


class PredecessorDenial(StrictModel):
    """Immutable v1 denial lineage; never an authority token."""

    method_key: Literal["segment_local_spectral_single_v1"]
    outcome: Literal["denied_0_of_32"]
    result_sha256: str
    result_bytes: Literal[102245]
    receipt_sha256: str
    receipt_bytes: Literal[1273]
    grants_training_authority: Literal[False]

    @model_validator(mode="after")
    def validate_exact_denial(self) -> PredecessorDenial:
        if self.result_sha256 != PREDECESSOR_DENIAL_RESULT_SHA256:
            raise ValueError("v1 denial result SHA-256 changed")
        if self.receipt_sha256 != PREDECESSOR_DENIAL_RECEIPT_SHA256:
            raise ValueError("v1 denial receipt SHA-256 changed")
        return self


class CandidateAxesV2(StrictModel):
    """Finite selector axes preregistered before any v2 synthetic run."""

    cycles_per_window: tuple[float, float]
    direct_to_dominant_minimum: tuple[float, float]

    @model_validator(mode="after")
    def validate_exact_axes(self) -> CandidateAxesV2:
        if self.cycles_per_window != (3.0, 4.0):
            raise ValueError("cycles_per_window must remain the frozen two-point axis")
        if self.direct_to_dominant_minimum != (0.12, 0.2):
            raise ValueError("direct-support axis changed")
        return self


class FixedProtocolV2(StrictModel):
    """Non-selectable structural identity shared by every v2 candidate."""

    feature_order: Literal["level_only"]
    embedding_context: Literal["full_authorized_segment_once"]
    embedding_execution_mode: Literal["eval_deterministic_no_grad_no_optimizer_update"]
    position_indices: Literal["absolute_native_unchanged"]
    proposal_scope: Literal["full_authorized_segment"]
    proposal_fft_oversampling: Literal[8]
    proposal_top_k_peaks: Literal[8]
    fundamental_divisors: tuple[Literal[1], Literal[2], Literal[3]]
    minimum_observed_cycles: float
    maximum_frequency_cycles_per_frame: float
    harmonic_band_share_minimum: float
    coherent_dimension_fraction_minimum: float
    signed_vector_acf_minimum: float
    signed_vector_acf_half_lag_contrast_minimum: float
    local_minimum_window_frames: Literal[24]
    local_hop_divisor: Literal[4]
    static_probe_window_frames: Literal[24]
    static_probe_hop_divisor: Literal[4]
    terminal_window: Literal["unique_terminal_after_zero_grid"]
    local_frequency_ratio_to_proposal: tuple[float, float]
    window_states: tuple[
        Literal["periodic"],
        Literal["static_zero_rate"],
        Literal["ambiguous"],
    ]
    static_maximum_relative_velocity: float
    static_maximum_relative_residual_energy: float
    interval_fusion: Literal["confidence_weighted_median_over_overlapping_windows"]
    static_interval_rate: Literal[0.0]
    minimum_known_interval_union_fraction: float
    maximum_ambiguous_gap_period_fraction: float
    ambiguous_internal_fill: Literal["linear_between_bracketing_known_rates"]
    ambiguous_boundary_fill: Literal["global_proposal_rate_unless_static_boundary"]
    duration: Literal["segment_length_minus_1"]
    video_rounding: Literal["half_up_once_exactly_one_segment"]
    multi_segment_video_total: Literal["undefined_never_sum"]
    coordinate_schema: Literal["coco17_xy"]
    minimum_joints_per_frame: Literal[8]
    minimum_stable_joints: Literal[6]
    minimum_stable_action_joints: Literal[3]
    prohibited_operations: tuple[
        Literal[
            "result_driven_scale_switch",
            "nearest_periodic_extrapolation_across_static",
            "activity_trim",
            "expert_vote",
            "count_clip",
        ],
        ...,
    ]

    @model_validator(mode="after")
    def validate_exact_fixed_protocol(self) -> FixedProtocolV2:
        expected_numeric = (
            2.0,
            0.25,
            0.25,
            0.5,
            0.2,
            0.2,
            0.000001,
            0.000001,
            0.8,
            1.0,
        )
        actual_numeric = (
            self.minimum_observed_cycles,
            self.maximum_frequency_cycles_per_frame,
            self.harmonic_band_share_minimum,
            self.coherent_dimension_fraction_minimum,
            self.signed_vector_acf_minimum,
            self.signed_vector_acf_half_lag_contrast_minimum,
            self.static_maximum_relative_velocity,
            self.static_maximum_relative_residual_energy,
            self.minimum_known_interval_union_fraction,
            self.maximum_ambiguous_gap_period_fraction,
        )
        if actual_numeric != expected_numeric:
            raise ValueError("fixed v2 numeric protocol changed")
        if self.fundamental_divisors != (1, 2, 3):
            raise ValueError("fundamental divisors must be exactly (1, 2, 3)")
        if self.local_frequency_ratio_to_proposal != (0.45, 2.25):
            raise ValueError("local proposal ratio changed")
        if self.window_states != ("periodic", "static_zero_rate", "ambiguous"):
            raise ValueError("window-state order changed")
        expected_operations = (
            "result_driven_scale_switch",
            "nearest_periodic_extrapolation_across_static",
            "activity_trim",
            "expert_vote",
            "count_clip",
        )
        if self.prohibited_operations != expected_operations:
            raise ValueError("prohibited-operation identity changed")
        return self


class SyntheticProtocolV2(StrictModel):
    """Frozen selector and no-fallback held-out replay."""

    selector_seed: Literal[62027]
    heldout_seed: Literal[91411]
    dimensions: tuple[Literal[34], Literal[512]]
    null_trials_per_family: Literal[64]
    reset_trials: Literal[64]
    positive_eligible_minimum: float
    overall_nmae_maximum: float
    overall_obo_minimum: float
    variable_tempo_nmae_maximum: float
    invariance_rounded_agreement_minimum: float
    null_false_eligible_cp_confidence: float
    null_false_eligible_cp_ucb_maximum: float
    reset_video_total_errors_maximum: Literal[0]
    positive_family_gate_policy: Literal[
        "all_frozen_generation_truth_families_hard_gated"
    ]
    positive_family_eligible_minimum: float
    positive_family_nmae_maximum: float
    positive_family_obo_minimum: float
    ranking: tuple[
        Literal[
            "null_cp_ucb",
            "variable_tempo_nmae",
            "overall_nmae",
            "negative_obo",
            "abstention_rate",
            "canonical_candidate_id",
        ],
        ...,
    ]
    heldout_policy: Literal["selected_candidate_only_same_gates_no_fallback"]
    null_replication_policy: Literal["every_replicate_seed_varies_source_arrays"]
    static_null_scoring: Literal[
        "eligible_zero_static_state_is_not_false_positive"
    ]

    @model_validator(mode="after")
    def validate_exact_synthetic_protocol(self) -> SyntheticProtocolV2:
        thresholds = (
            self.positive_eligible_minimum,
            self.overall_nmae_maximum,
            self.overall_obo_minimum,
            self.variable_tempo_nmae_maximum,
            self.invariance_rounded_agreement_minimum,
            self.null_false_eligible_cp_confidence,
            self.null_false_eligible_cp_ucb_maximum,
            self.positive_family_eligible_minimum,
            self.positive_family_nmae_maximum,
            self.positive_family_obo_minimum,
        )
        if thresholds != (0.95, 0.08, 0.95, 0.1, 0.95, 0.95, 0.05, 0.95, 0.1, 0.95):
            raise ValueError("synthetic v2 gates changed")
        expected_ranking = (
            "null_cp_ucb",
            "variable_tempo_nmae",
            "overall_nmae",
            "negative_obo",
            "abstention_rate",
            "canonical_candidate_id",
        )
        if self.ranking != expected_ranking:
            raise ValueError("synthetic ranking changed")
        return self


class SeedDerivationProtocol(StrictModel):
    method: Literal["sha256_domain_separated_uint64_le"]
    domain: Literal["segment_local_spectral_single_v2_map_seed"]
    roles: tuple[Literal["map1"], Literal["map2"]]
    fields: tuple[
        Literal["global_seed"],
        Literal["video_id"],
        Literal["segment_id"],
        Literal["role"],
    ]

    @model_validator(mode="after")
    def validate_exact_seed_protocol(self) -> SeedDerivationProtocol:
        if self.roles != ("map1", "map2"):
            raise ValueError("map roles must preserve exact order")
        if self.fields != ("global_seed", "video_id", "segment_id", "role"):
            raise ValueError("map seed fields must preserve exact order")
        return self


class AuthorityProtocolV2(StrictModel):
    synthetic_selector_authorized: Literal[True]
    train337_authorized: Literal[False]
    dev84_authorized: Literal[False]
    test105_authorized: Literal[False]
    representation_adapter_status: Literal["unwired_fail_closed"]
    launcher_status: Literal["fixed_exit_3"]


class SegmentLocalSpectralConfigV2(StrictModel):
    """Strict identity for ``segment_local_spectral_single_v2``."""

    schema_version: Literal[2]
    key: Literal["segment_local_spectral_single_v2"]
    classification: Literal["inferred_label_free_single_expert_baseline_successor"]
    eligible_as_exact_author_baseline: Literal[False]
    predecessor_denial: PredecessorDenial
    axes: CandidateAxesV2
    fixed: FixedProtocolV2
    synthetic: SyntheticProtocolV2
    seed_derivation: SeedDerivationProtocol
    authority: AuthorityProtocolV2

    @property
    def candidates(self) -> tuple[SpectralCandidateV2, ...]:
        candidates = tuple(
            SpectralCandidateV2(cycles, direct)
            for cycles in self.axes.cycles_per_window
            for direct in self.axes.direct_to_dominant_minimum
        )
        if len(candidates) != 4 or len({item.canonical_id for item in candidates}) != 4:
            raise RuntimeError("v2 selector grid must contain four unique candidates")
        return candidates

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True, order=True)
class SpectralCandidateV2:
    cycles_per_window: float
    direct_to_dominant_minimum: float

    def __post_init__(self) -> None:
        if self.cycles_per_window not in (3.0, 4.0):
            raise ValueError("cycles_per_window is outside the frozen v2 axis")
        if self.direct_to_dominant_minimum not in (0.12, 0.2):
            raise ValueError("direct support is outside the frozen v2 axis")

    @property
    def canonical_id(self) -> str:
        return (
            f"cpw{self.cycles_per_window:g}."
            f"direct{self.direct_to_dominant_minimum:g}"
        )


def load_segment_local_spectral_config_v2(
    path: str | Path,
) -> SegmentLocalSpectralConfigV2:
    """Load v2 without accepting unknown, relaxed, or v1 values."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    config = SegmentLocalSpectralConfigV2.model_validate(payload)
    _ = config.candidates
    return config


def derive_domain_separated_map_seed(
    global_seed: int,
    video_id: str,
    segment_id: str,
    role: MapRole,
) -> int:
    """Derive the only permitted map1/map2 seed from all identity fields."""

    if isinstance(global_seed, bool) or int(global_seed) != global_seed:
        raise TypeError("global_seed must be an integer")
    identifier = str(video_id).strip()
    segment = str(segment_id).strip()
    if not identifier or identifier != video_id:
        raise ValueError("video_id must be canonical and non-empty")
    if not segment or segment != segment_id:
        raise ValueError("segment_id must be canonical and non-empty")
    if role not in ("map1", "map2"):
        raise ValueError("role must be map1 or map2")
    digest = hashlib.sha256()
    digest.update(b"segment_local_spectral_single_v2_map_seed\0")
    digest.update(int(global_seed).to_bytes(8, byteorder="little", signed=True))
    for value in (identifier, segment, role):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)


def _round_half_up(value: float) -> int:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("count must be finite and non-negative")
    return int(math.floor(value + 0.5))


def _linear_detrend(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("features must have shape [time>=2, dimensions]")
    time = np.arange(values.shape[0], dtype=np.float64)
    time -= float(np.mean(time))
    centered = values - np.mean(values, axis=0, keepdims=True)
    denominator = float(np.dot(time, time))
    slopes = np.sum(centered * time[:, None], axis=0) / denominator
    return centered - time[:, None] * slopes[None, :]


def _signed_vector_acf(values: NDArray[np.float64], lag: int) -> float:
    if lag < 1 or lag >= values.shape[0]:
        return -1.0
    left = values[:-lag]
    right = values[lag:]
    denominator = math.sqrt(float(np.sum(left * left)) * float(np.sum(right * right)))
    if denominator <= _EPSILON:
        return -1.0
    return float(np.clip(np.sum(left * right) / denominator, -1.0, 1.0))


def _static_diagnostics(values: NDArray[np.float64]) -> tuple[bool, float, float]:
    """Return scale-relative velocity/residual diagnostics for exact plateaus."""

    centered = values - np.mean(values, axis=0, keepdims=True)
    scale = max(float(np.sqrt(np.mean(values * values))), _EPSILON)
    velocity = np.diff(values, axis=0)
    relative_velocity = float(np.sqrt(np.mean(velocity * velocity)) / scale)
    residual = _linear_detrend(values)
    relative_residual = float(np.sqrt(np.mean(residual * residual)) / scale)
    centered_size = float(np.sqrt(np.mean(centered * centered)))
    level_static = centered_size / scale <= 0.000001
    is_static = bool(
        level_static
        and relative_velocity <= 0.000001
        and relative_residual <= 0.000001
    )
    return is_static, relative_velocity, relative_residual


def _next_power_of_two(value: int) -> int:
    if value < 1:
        raise ValueError("FFT length input must be positive")
    return 1 << (value - 1).bit_length()


@dataclass(frozen=True, slots=True)
class SpectralPeak:
    frequency: float
    power: float
    rank: int

    def __post_init__(self) -> None:
        if not np.isfinite(self.frequency) or self.frequency <= 0.0:
            raise ValueError("spectral peak frequency must be positive")
        if not np.isfinite(self.power) or self.power <= 0.0:
            raise ValueError("spectral peak power must be positive")
        if self.rank < 1:
            raise ValueError("spectral peak rank must be positive")


@dataclass(frozen=True, slots=True)
class FundamentalCandidateDiagnostic:
    frequency: float
    source_peak_frequency: float
    divisor: Literal[1, 2, 3]
    direct_to_dominant: float
    harmonic_band_share: float
    coherent_dimension_fraction: float
    signed_acf: float
    half_lag_acf: float
    acf_contrast: float
    trusted: bool
    failed_checks: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.failed_checks, tuple):
            raise TypeError("failed_checks must be an exact tuple")
        bounded = (
            self.direct_to_dominant,
            self.harmonic_band_share,
            self.coherent_dimension_fraction,
        )
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in bounded):
            raise ValueError("spectral support diagnostics must be in [0, 1]")
        correlations = (self.signed_acf, self.half_lag_acf, self.acf_contrast)
        if any(not np.isfinite(value) or not -2.0 <= value <= 2.0 for value in correlations):
            raise ValueError("ACF diagnostics are invalid")
        if self.trusted != (not self.failed_checks):
            raise ValueError("trusted flag must replay from failed_checks")


@dataclass(frozen=True, slots=True)
class FrequencyAnalysis:
    state: ProposalState
    frequency: float | None
    confidence: float
    reason: str | None
    informative_dimensions: int
    relative_velocity: float
    relative_residual_energy: float
    peaks: tuple[SpectralPeak, ...]
    fundamental_candidates: tuple[FundamentalCandidateDiagnostic, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.peaks, tuple) or not isinstance(
            self.fundamental_candidates, tuple
        ):
            raise TypeError("frequency-analysis nested containers must be tuples")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("frequency confidence must be in [0, 1]")
        if self.state == "periodic":
            if self.frequency is None or self.frequency <= 0.0 or self.reason is not None:
                raise ValueError("periodic analysis requires a positive frequency")
        else:
            if self.frequency is not None or not self.reason:
                raise ValueError("non-periodic analysis requires a reason and no frequency")


def _parabolic_frequency(
    aggregate: NDArray[np.float64],
    index: int,
    fft_size: int,
) -> float:
    if index <= 0 or index >= aggregate.size - 1:
        return index / fft_size
    left = math.log(max(float(aggregate[index - 1]), _EPSILON))
    center = math.log(max(float(aggregate[index]), _EPSILON))
    right = math.log(max(float(aggregate[index + 1]), _EPSILON))
    denominator = left - 2.0 * center + right
    offset = 0.0 if abs(denominator) <= _EPSILON else 0.5 * (left - right) / denominator
    return float((index + np.clip(offset, -0.5, 0.5)) / fft_size)


def _top_spectral_peaks(
    aggregate: NDArray[np.float64],
    allowed_indices: NDArray[np.int64],
    *,
    fft_size: int,
    frame_count: int,
    top_k: int,
) -> tuple[SpectralPeak, ...]:
    local: list[int] = []
    allowed_set = set(int(value) for value in allowed_indices)
    for index in allowed_indices:
        current = int(index)
        left = aggregate[current - 1] if current - 1 in allowed_set else -math.inf
        right = aggregate[current + 1] if current + 1 in allowed_set else -math.inf
        if aggregate[current] >= left and aggregate[current] >= right:
            local.append(current)
    if not local:
        return ()
    # The Hann main lobe spans roughly two bins of the unpadded transform on
    # either side.  Suppress the whole lobe so top-K represents distinct
    # physical peaks instead of eight samples from one harmonic.
    separation = max(1, int(math.ceil(2.0 * fft_size / (frame_count - 1))))
    selected: list[int] = []
    for index in sorted(local, key=lambda item: (-float(aggregate[item]), item)):
        if all(abs(index - previous) >= separation for previous in selected):
            selected.append(index)
        if len(selected) == top_k:
            break
    minimum_allowed = float(allowed_indices[0] / fft_size)
    maximum_allowed = float(allowed_indices[-1] / fft_size)
    return tuple(
        SpectralPeak(
            # A boundary peak's unconstrained parabola can fall outside the
            # preregistered band.  Clamp to the exact allowed FFT-bin domain;
            # this preserves the L-1 count=2 endpoint instead of silently
            # deleting it from fundamental candidates.
            frequency=float(
                np.clip(
                    _parabolic_frequency(aggregate, index, fft_size),
                    minimum_allowed,
                    maximum_allowed,
                )
            ),
            power=float(aggregate[index]),
            rank=rank,
        )
        for rank, index in enumerate(selected, start=1)
    )


def _cluster_indices(
    frequencies: NDArray[np.float64],
    center: float,
    half_width: float,
) -> NDArray[np.int64]:
    return np.flatnonzero(np.abs(frequencies - center) <= half_width)


def _cluster_power(
    aggregate: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    center: float,
    half_width: float,
) -> float:
    indices = _cluster_indices(frequencies, center, half_width)
    if indices.size == 0:
        return 0.0
    return float(np.sum(aggregate[indices]))


def _harmonic_union_power(
    aggregate: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    frequency: float,
    half_width: float,
    maximum_frequency: float,
) -> float:
    mask = np.zeros(frequencies.shape, dtype=np.bool_)
    for multiplier in (1, 2, 3):
        center = frequency * multiplier
        if center <= maximum_frequency:
            mask |= np.abs(frequencies - center) <= half_width
    return float(np.sum(aggregate[mask]))


def _harmonic_union_mask(
    frequencies: NDArray[np.float64],
    frequency: float,
    half_width: float,
    maximum_frequency: float,
) -> NDArray[np.bool_]:
    mask = np.zeros(frequencies.shape, dtype=np.bool_)
    for multiplier in (1, 2, 3):
        center = frequency * multiplier
        if center <= maximum_frequency:
            mask |= np.abs(frequencies - center) <= half_width
    return mask


def _frequency_analysis(
    features: NDArray[np.float64],
    candidate: SpectralCandidateV2,
    protocol: FixedProtocolV2,
    *,
    frequency_bounds: tuple[float, float] | None = None,
    proposal_mode: bool = False,
) -> FrequencyAnalysis:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("frequency analysis requires [time>=2, dimensions>=1]")
    is_static, relative_velocity, relative_residual = _static_diagnostics(values)
    if is_static:
        return FrequencyAnalysis(
            state="static_zero_rate",
            frequency=None,
            confidence=1.0,
            reason="frozen_low_velocity_and_low_residual_rule",
            informative_dimensions=0,
            relative_velocity=relative_velocity,
            relative_residual_energy=relative_residual,
            peaks=(),
            fundamental_candidates=(),
        )

    detrended = _linear_detrend(values)
    variance = np.mean(detrended * detrended, axis=0)
    maximum_variance = float(np.max(variance))
    informative = variance > max(_EPSILON, maximum_variance * 1e-12)
    informative_count = int(np.count_nonzero(informative))
    if informative_count == 0:
        return FrequencyAnalysis(
            state="ambiguous",
            frequency=None,
            confidence=0.0,
            reason="no_informative_dimensions_after_detrend",
            informative_dimensions=0,
            relative_velocity=relative_velocity,
            relative_residual_energy=relative_residual,
            peaks=(),
            fundamental_candidates=(),
        )

    signal = detrended[:, informative]
    frame_count = int(signal.shape[0])
    duration = frame_count - 1
    # Using an exact multiple of L-1 makes an endpoint-defined integer count
    # (including count=2) land exactly on an FFT bin.  The returned frequency
    # is cycles per source-frame interval, never cycles per array length.
    fft_size = protocol.proposal_fft_oversampling * duration
    tapered = signal * np.hanning(frame_count)[:, None]
    power = np.abs(np.fft.rfft(tapered, n=fft_size, axis=0)) ** 2
    aggregate = np.sum(power, axis=1)
    frequencies = np.fft.rfftfreq(fft_size)
    lower = protocol.minimum_observed_cycles / duration
    upper = protocol.maximum_frequency_cycles_per_frame
    if frequency_bounds is not None:
        bound_low, bound_high = frequency_bounds
        if not 0.0 < bound_low < bound_high:
            raise ValueError("frequency bounds must be positive and ordered")
        lower = max(lower, bound_low)
        upper = min(upper, bound_high)
    allowed = (frequencies >= lower) & (frequencies <= upper)
    allowed_indices = np.flatnonzero(allowed)
    allowed_total = float(np.sum(aggregate[allowed]))
    if allowed_indices.size < 3 or allowed_total <= _EPSILON:
        return FrequencyAnalysis(
            state="ambiguous",
            frequency=None,
            confidence=0.0,
            reason="no_nonzero_frequency_band_support",
            informative_dimensions=informative_count,
            relative_velocity=relative_velocity,
            relative_residual_energy=relative_residual,
            peaks=(),
            fundamental_candidates=(),
        )
    peaks = _top_spectral_peaks(
        aggregate,
        allowed_indices,
        fft_size=fft_size,
        frame_count=frame_count,
        top_k=protocol.proposal_top_k_peaks,
    )
    if not peaks:
        return FrequencyAnalysis(
            state="ambiguous",
            frequency=None,
            confidence=0.0,
            reason="no_local_spectral_peak",
            informative_dimensions=informative_count,
            relative_velocity=relative_velocity,
            relative_residual_energy=relative_residual,
            peaks=(),
            fundamental_candidates=(),
        )

    half_width = 1.0 / duration
    dominant_cluster = max(
        _cluster_power(aggregate, frequencies, peak.frequency, half_width)
        for peak in peaks
    )
    raw_candidates: list[tuple[float, float, Literal[1, 2, 3]]] = []
    for peak in peaks:
        for divisor in protocol.fundamental_divisors:
            frequency = peak.frequency / divisor
            if lower <= frequency <= upper:
                raw_candidates.append((frequency, peak.frequency, divisor))
    raw_candidates.sort(key=lambda item: (item[0], item[2], item[1]))
    deduplicated: list[tuple[float, float, Literal[1, 2, 3]]] = []
    tolerance = 0.5 / fft_size
    for item in raw_candidates:
        if not deduplicated or abs(item[0] - deduplicated[-1][0]) > tolerance:
            deduplicated.append(item)

    diagnostics: list[FundamentalCandidateDiagnostic] = []
    for frequency, source_peak, divisor in deduplicated:
        direct = _cluster_power(aggregate, frequencies, frequency, half_width)
        direct_ratio = float(np.clip(direct / max(dominant_cluster, _EPSILON), 0.0, 1.0))
        harmonic = _harmonic_union_power(
            aggregate,
            frequencies,
            frequency,
            half_width,
            upper,
        )
        harmonic_share = float(np.clip(harmonic / allowed_total, 0.0, 1.0))
        harmonic_mask = _harmonic_union_mask(
            frequencies,
            frequency,
            half_width,
            upper,
        )
        per_dimension_allowed = np.sum(power[allowed], axis=0)
        per_dimension_harmonic = np.sum(power[harmonic_mask], axis=0)
        per_dimension_share = np.divide(
            per_dimension_harmonic,
            per_dimension_allowed,
            out=np.zeros_like(per_dimension_harmonic),
            where=per_dimension_allowed > _EPSILON,
        )
        coherent_dimension_fraction = float(
            np.mean(per_dimension_share >= protocol.harmonic_band_share_minimum)
        )
        lag = int(math.floor(1.0 / frequency + 0.5))
        half_lag = max(1, int(math.floor(0.5 / frequency + 0.5)))
        signed_acf = _signed_vector_acf(signal, lag)
        half_acf = _signed_vector_acf(signal, half_lag)
        contrast = signed_acf - half_acf
        failed: list[str] = []
        observed_cycles = frequency * duration
        if observed_cycles + 1e-12 < protocol.minimum_observed_cycles:
            failed.append("fewer_than_two_observed_cycles")
        if direct_ratio < candidate.direct_to_dominant_minimum:
            failed.append("insufficient_direct_fundamental_support")
        if harmonic_share < protocol.harmonic_band_share_minimum:
            failed.append("insufficient_harmonic_family_support")
        if coherent_dimension_fraction < protocol.coherent_dimension_fraction_minimum:
            failed.append("insufficient_coherent_dimension_fraction")
        # A full-segment proposal must remain usable under smoothly varying
        # tempo, for which one exact lag is not expected to correlate.  Its
        # signed evidence is the frozen positive half-lag contrast.  Local
        # windows retain the stricter absolute ACF check.
        if not proposal_mode and signed_acf < protocol.signed_vector_acf_minimum:
            failed.append("signed_vector_acf_below_minimum")
        if contrast < protocol.signed_vector_acf_half_lag_contrast_minimum:
            failed.append("half_lag_contrast_below_minimum")
        diagnostics.append(
            FundamentalCandidateDiagnostic(
                frequency=frequency,
                source_peak_frequency=source_peak,
                divisor=divisor,
                direct_to_dominant=direct_ratio,
                harmonic_band_share=harmonic_share,
                coherent_dimension_fraction=coherent_dimension_fraction,
                signed_acf=signed_acf,
                half_lag_acf=half_acf,
                acf_contrast=contrast,
                trusted=not failed,
                failed_checks=tuple(failed),
            )
        )
    trusted = tuple(item for item in diagnostics if item.trusted)
    if not trusted:
        return FrequencyAnalysis(
            state="ambiguous",
            frequency=None,
            confidence=0.0,
            reason="no_trusted_fundamental_candidate",
            informative_dimensions=informative_count,
            relative_velocity=relative_velocity,
            relative_residual_energy=relative_residual,
            peaks=peaks,
            fundamental_candidates=tuple(diagnostics),
        )
    selected = max(
        trusted,
        key=lambda item: (
            item.direct_to_dominant
            * item.harmonic_band_share
            * item.coherent_dimension_fraction
            * float(np.clip((item.signed_acf + 1.0) / 2.0, 0.0, 1.0))
            * float(np.clip(item.acf_contrast / 2.0, 0.0, 1.0)),
            -item.frequency,
            -item.divisor,
        ),
    )
    acf_strength = float(np.clip((selected.signed_acf + 1.0) / 2.0, 0.0, 1.0))
    contrast_strength = float(np.clip(selected.acf_contrast / 2.0, 0.0, 1.0))
    confidence = float(
        np.clip(
            (
                selected.direct_to_dominant
                * selected.harmonic_band_share
                * acf_strength
                * contrast_strength
            )
            ** 0.25,
            0.0,
            1.0,
        )
    )
    return FrequencyAnalysis(
        state="periodic",
        frequency=selected.frequency,
        confidence=confidence,
        reason=None,
        informative_dimensions=informative_count,
        relative_velocity=relative_velocity,
        relative_residual_energy=relative_residual,
        peaks=peaks,
        fundamental_candidates=tuple(diagnostics),
    )


@dataclass(frozen=True, slots=True)
class AdaptiveWindow:
    segment_id: str
    start: int
    stop: int
    center: float
    stable_joints: tuple[int, ...]
    geometry_authorized: bool

    def __post_init__(self) -> None:
        if not isinstance(self.stable_joints, tuple):
            raise TypeError("stable_joints must be an exact tuple")
        if not self.segment_id or self.start < 0 or self.stop <= self.start:
            raise ValueError("adaptive window identity is invalid")
        if not self.start <= self.center < self.stop:
            raise ValueError("adaptive window center must lie inside the window")

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.segment_id, self.start, self.stop)


def _adaptive_window_length(
    segment_length: int,
    proposal: FrequencyAnalysis,
    candidate: SpectralCandidateV2,
    protocol: FixedProtocolV2,
) -> int:
    if segment_length < 2:
        raise ValueError("adaptive scale requires segment length >=2")
    if proposal.state == "static_zero_rate":
        return min(segment_length, max(protocol.local_minimum_window_frames, 64))
    if proposal.state != "periodic" or proposal.frequency is None:
        raise ValueError("periodic proposal is required to derive a periodic scale")
    # A W-frame window spans W-1 intervals.  Add one after deriving the
    # requested interval duration so scale and integration share one clock.
    low_ratio = protocol.local_frequency_ratio_to_proposal[0]
    requested_intervals = max(
        candidate.cycles_per_window / proposal.frequency,
        protocol.minimum_observed_cycles / (proposal.frequency * low_ratio),
    )
    proposed = int(math.ceil(requested_intervals)) + 1
    return min(segment_length, max(protocol.local_minimum_window_frames, proposed))


def _build_adaptive_windows(
    video: RepresentationVideo,
    segment: AuthorizedSegment,
    window_length: int,
    protocol: FixedProtocolV2,
    *,
    hop_divisor: int | None = None,
) -> tuple[AdaptiveWindow, ...]:
    if window_length < 2 or window_length > segment.length:
        raise ValueError("adaptive window length must lie within the segment")
    divisor = protocol.local_hop_divisor if hop_divisor is None else int(hop_divisor)
    if divisor < 1:
        raise ValueError("window hop divisor must be positive")
    hop = max(1, window_length // divisor)
    relative_terminal = segment.length - window_length
    relative_starts = list(range(0, relative_terminal + 1, hop))
    if not relative_starts or relative_starts[-1] != relative_terminal:
        relative_starts.append(relative_terminal)
    if len(relative_starts) != len(set(relative_starts)):
        raise RuntimeError("adaptive zero-grid plus terminal produced a duplicate")
    windows: list[AdaptiveWindow] = []
    for relative_start in relative_starts:
        start = segment.start + relative_start
        stop = start + window_length
        valid = bool(np.all(video.valid_mask[start:stop]))
        masks = np.asarray(video.joint_mask[start:stop], dtype=np.bool_)
        per_frame = np.sum(masks, axis=1)
        stable = np.flatnonzero(np.all(masks, axis=0)) if valid else np.empty(0, dtype=np.int64)
        stable_action = np.intersect1d(
            stable,
            np.asarray(_ACTION_JOINTS, dtype=np.int64),
            assume_unique=True,
        )
        geometry_authorized = bool(
            valid
            and np.all(per_frame >= protocol.minimum_joints_per_frame)
            and stable.size >= protocol.minimum_stable_joints
            and stable_action.size >= protocol.minimum_stable_action_joints
        )
        windows.append(
            AdaptiveWindow(
                segment_id=segment.segment_id,
                start=start,
                stop=stop,
                center=start + (window_length - 1) / 2.0,
                stable_joints=tuple(int(value) for value in stable),
                geometry_authorized=geometry_authorized,
            )
        )
    return tuple(windows)


@dataclass(frozen=True, slots=True)
class AdaptiveWindowEstimate:
    window: AdaptiveWindow
    state: WindowState
    interval_rate: float | None
    confidence: float
    reason: str | None
    analysis: FrequencyAnalysis | None

    def __post_init__(self) -> None:
        if self.state == "periodic":
            if self.interval_rate is None or self.interval_rate <= 0.0:
                raise ValueError("periodic window requires a positive interval rate")
            if self.analysis is None or self.analysis.state != "periodic":
                raise ValueError("periodic window requires periodic analysis")
            if self.reason is not None:
                raise ValueError("periodic window cannot carry a reason")
        elif self.state == "static_zero_rate":
            if self.interval_rate != 0.0:
                raise ValueError("static window rate must be exact zero")
            if self.analysis is None or self.analysis.state != "static_zero_rate":
                raise ValueError("static window requires static analysis")
            if not self.reason:
                raise ValueError("static window must retain its classification reason")
        else:
            if self.interval_rate is not None or not self.reason:
                raise ValueError("ambiguous window requires a reason and no rate")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("window confidence must be in [0, 1]")


def _estimate_adaptive_window(
    video: RepresentationVideo,
    window: AdaptiveWindow,
    proposal: FrequencyAnalysis,
    candidate: SpectralCandidateV2,
    protocol: FixedProtocolV2,
) -> AdaptiveWindowEstimate:
    if not window.geometry_authorized:
        return AdaptiveWindowEstimate(
            window=window,
            state="ambiguous",
            interval_rate=None,
            confidence=0.0,
            reason="geometry_not_authorized",
            analysis=None,
        )
    features = np.asarray(video.features[window.start : window.stop], dtype=np.float64)
    if proposal.state == "periodic" and proposal.frequency is not None:
        low_ratio, high_ratio = protocol.local_frequency_ratio_to_proposal
        bounds = (proposal.frequency * low_ratio, proposal.frequency * high_ratio)
    else:
        bounds = None
    analysis = _frequency_analysis(
        features,
        candidate,
        protocol,
        frequency_bounds=bounds,
    )
    if analysis.state == "periodic" and analysis.frequency is not None:
        return AdaptiveWindowEstimate(
            window=window,
            state="periodic",
            interval_rate=analysis.frequency,
            confidence=analysis.confidence,
            reason=None,
            analysis=analysis,
        )
    if analysis.state == "static_zero_rate":
        return AdaptiveWindowEstimate(
            window=window,
            state="static_zero_rate",
            interval_rate=0.0,
            confidence=1.0,
            reason=analysis.reason,
            analysis=analysis,
        )
    return AdaptiveWindowEstimate(
        window=window,
        state="ambiguous",
        interval_rate=None,
        confidence=0.0,
        reason=analysis.reason,
        analysis=analysis,
    )


def _estimate_fixed_static_probe(
    video: RepresentationVideo,
    window: AdaptiveWindow,
    candidate: SpectralCandidateV2,
    protocol: FixedProtocolV2,
) -> AdaptiveWindowEstimate | None:
    """Return only proven static support from the frozen 24-frame probe grid."""

    if not window.geometry_authorized:
        return None
    features = np.asarray(video.features[window.start : window.stop], dtype=np.float64)
    analysis = _frequency_analysis(features, candidate, protocol)
    if analysis.state != "static_zero_rate":
        return None
    return AdaptiveWindowEstimate(
        window=window,
        state="static_zero_rate",
        interval_rate=0.0,
        confidence=1.0,
        reason=analysis.reason,
        analysis=analysis,
    )


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    value_array = np.asarray(tuple(values), dtype=np.float64)
    weight_array = np.asarray(tuple(weights), dtype=np.float64)
    if value_array.ndim != 1 or value_array.size == 0 or weight_array.shape != value_array.shape:
        raise ValueError("weighted median requires matching non-empty vectors")
    order = np.argsort(value_array, kind="stable")
    ordered_values = value_array[order]
    ordered_weights = weight_array[order]
    total = float(np.sum(ordered_weights))
    if total <= _EPSILON:
        raise ValueError("weighted median requires positive total weight")
    index = int(np.searchsorted(np.cumsum(ordered_weights), total / 2.0, side="left"))
    return float(ordered_values[min(index, ordered_values.size - 1)])


def _known_interval_rates(
    segment: AuthorizedSegment,
    windows: Sequence[AdaptiveWindowEstimate],
) -> tuple[list[float | None], list[bool]]:
    duration = segment.length - 1
    contributions: list[list[tuple[float, float]]] = [[] for _ in range(duration)]
    static_support = [False] * duration
    for item in windows:
        if item.interval_rate is None:
            continue
        local_start = item.window.start - segment.start
        local_stop = item.window.stop - segment.start - 1
        for interval in range(local_start, local_stop):
            contributions[interval].append((item.interval_rate, max(item.confidence, _EPSILON)))
            if item.state == "static_zero_rate":
                static_support[interval] = True
    rates: list[float | None] = []
    for interval, values in enumerate(contributions):
        if not values:
            rates.append(None)
            continue
        # Static evidence has priority on its own source interval.  A periodic
        # overlap can never propagate a positive rate through a known plateau.
        if static_support[interval]:
            rates.append(0.0)
            continue
        rates.append(
            _weighted_median(
                tuple(item[0] for item in values),
                tuple(item[1] for item in values),
            )
        )
    return rates, static_support


def _missing_runs(values: Sequence[float | None]) -> tuple[tuple[int, int], ...]:
    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(values):
        if values[index] is not None:
            index += 1
            continue
        start = index
        while index < len(values) and values[index] is None:
            index += 1
        runs.append((start, index))
    return tuple(runs)


def _fill_ambiguous_intervals(
    rates: Sequence[float | None],
    static_support: Sequence[bool],
    *,
    proposal_rate: float,
    proposal_period: float,
    protocol: FixedProtocolV2,
) -> tuple[tuple[float, ...] | None, tuple[bool, ...], int]:
    if len(rates) != len(static_support):
        raise ValueError("interval rates/static support lengths differ")
    output = list(rates)
    filled = [False] * len(output)
    maximum_gap = max(
        1,
        int(math.floor(protocol.maximum_ambiguous_gap_period_fraction * proposal_period + 0.5)),
    )
    runs = _missing_runs(output)
    largest = max((stop - start for start, stop in runs), default=0)
    for start, stop in runs:
        if stop - start > maximum_gap:
            return None, tuple(filled), largest
        left = output[start - 1] if start > 0 else None
        right = output[stop] if stop < len(output) else None
        left_static = start > 0 and static_support[start - 1]
        right_static = stop < len(output) and static_support[stop]
        if left is not None and right is not None:
            for offset, index in enumerate(range(start, stop), start=1):
                fraction = offset / (stop - start + 1)
                output[index] = (1.0 - fraction) * left + fraction * right
                filled[index] = True
        elif left is not None:
            boundary = 0.0 if left_static else proposal_rate
            for index in range(start, stop):
                output[index] = boundary
                filled[index] = True
        elif right is not None:
            boundary = 0.0 if right_static else proposal_rate
            for index in range(start, stop):
                output[index] = boundary
                filled[index] = True
        else:
            return None, tuple(filled), largest
    if any(value is None for value in output):
        raise RuntimeError("ambiguous fill left an interval undefined")
    return tuple(float(value) for value in output if value is not None), tuple(filled), largest


@dataclass(frozen=True, slots=True)
class SegmentEstimateV2:
    segment: AuthorizedSegment
    proposal: FrequencyAnalysis
    adaptive_window_frames: int | None
    windows: tuple[AdaptiveWindowEstimate, ...]
    status: Literal["eligible", "abstain"]
    abstention_reason: str | None
    known_interval_union_fraction: float
    largest_ambiguous_gap_intervals: int
    interval_rates: tuple[float | None, ...]
    interval_filled: tuple[bool, ...]
    float_count: float | None
    static_zero_only: bool
    diagnostic_stages: tuple[str, ...]

    def __post_init__(self) -> None:
        tuple_fields = (
            self.windows,
            self.interval_rates,
            self.interval_filled,
            self.diagnostic_stages,
        )
        if any(not isinstance(value, tuple) for value in tuple_fields):
            raise TypeError("segment nested containers must be exact tuples")
        if self.diagnostic_stages != _DIAGNOSTIC_STAGE_ORDER:
            raise ValueError("diagnostic stage order changed")
        duration = self.segment.length - 1
        if len(self.interval_rates) != duration or len(self.interval_filled) != duration:
            raise ValueError("interval fields must cover exactly L-1 intervals")
        if not 0.0 <= self.known_interval_union_fraction <= 1.0:
            raise ValueError("known interval coverage must be in [0, 1]")
        if self.status == "eligible":
            if self.float_count is None or self.abstention_reason is not None:
                raise ValueError("eligible segment requires a count and no abstention")
            if any(value is None for value in self.interval_rates):
                raise ValueError("eligible segment cannot expose ambiguous interval rates")
            if self.static_zero_only != all(value == 0.0 for value in self.interval_rates):
                raise ValueError("static_zero_only does not replay from interval rates")
        else:
            if self.float_count is not None or not self.abstention_reason:
                raise ValueError("abstaining segment requires a reason and no count")
            if self.static_zero_only:
                raise ValueError("abstaining segment cannot claim static_zero_only")


@dataclass(frozen=True, slots=True)
class ViewEstimateV2:
    video_id: str
    candidate_id: str
    status: EstimateStatus
    abstention_reason: str | None
    float_count: float | None
    rounded_count: int | None
    static_zero_only: bool
    periodic_confidence: float
    segments: tuple[SegmentEstimateV2, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.segments, tuple):
            raise TypeError("view segments must be an exact tuple")
        if not self.video_id or not self.candidate_id:
            raise ValueError("view identity must be non-empty")
        if not 0.0 <= self.periodic_confidence <= 1.0:
            raise ValueError("periodic confidence must be in [0, 1]")
        if self.status == "eligible":
            if len(self.segments) != 1 or self.segments[0].status != "eligible":
                raise ValueError("eligible video requires one eligible segment")
            if self.float_count is None or self.rounded_count is None:
                raise ValueError("eligible video requires float and rounded counts")
            if self.rounded_count != _round_half_up(self.float_count):
                raise ValueError("video count must be rounded half-up once")
            if self.static_zero_only != self.segments[0].static_zero_only:
                raise ValueError("video static state must replay from its segment")
        elif self.status == "undefined_multi_segment":
            if len(self.segments) <= 1:
                raise ValueError("undefined multi-segment requires multiple segments")
            if self.float_count is not None or self.rounded_count is not None:
                raise ValueError("multi-segment total must remain undefined")
            if self.abstention_reason != "multiple_authorized_segments_never_sum":
                raise ValueError("multi-segment denial reason changed")
        else:
            if self.float_count is not None or self.rounded_count is not None:
                raise ValueError("abstaining video cannot expose a count")
            if not self.abstention_reason:
                raise ValueError("abstaining video requires a reason")


def _abstaining_segment(
    segment: AuthorizedSegment,
    proposal: FrequencyAnalysis,
    reason: str,
    *,
    adaptive_window_frames: int | None = None,
    windows: tuple[AdaptiveWindowEstimate, ...] = (),
    known_fraction: float = 0.0,
    largest_gap: int | None = None,
    interval_rates: tuple[float | None, ...] | None = None,
) -> SegmentEstimateV2:
    duration = segment.length - 1
    rates = interval_rates if interval_rates is not None else (None,) * duration
    return SegmentEstimateV2(
        segment=segment,
        proposal=proposal,
        adaptive_window_frames=adaptive_window_frames,
        windows=windows,
        status="abstain",
        abstention_reason=reason,
        known_interval_union_fraction=known_fraction,
        largest_ambiguous_gap_intervals=(duration if largest_gap is None else largest_gap),
        interval_rates=rates,
        interval_filled=(False,) * duration,
        float_count=None,
        static_zero_only=False,
        diagnostic_stages=_DIAGNOSTIC_STAGE_ORDER,
    )


def _estimate_segment(
    video: RepresentationVideo,
    segment: AuthorizedSegment,
    candidate: SpectralCandidateV2,
    protocol: FixedProtocolV2,
) -> SegmentEstimateV2:
    features = np.asarray(video.features[segment.start : segment.stop], dtype=np.float64)
    proposal = _frequency_analysis(features, candidate, protocol, proposal_mode=True)
    if proposal.state == "ambiguous":
        return _abstaining_segment(segment, proposal, "ambiguous_full_segment_proposal")
    window_length = _adaptive_window_length(segment.length, proposal, candidate, protocol)
    authorities = _build_adaptive_windows(video, segment, window_length, protocol)
    primary_windows = tuple(
        _estimate_adaptive_window(video, item, proposal, candidate, protocol)
        for item in authorities
    )
    # Static support uses its own frozen 24-frame scale.  It is not selected
    # from outcomes and cannot contribute a periodic rate.  This is what lets
    # genuine pauses write zero on their intervals even when the period-derived
    # window must be longer than the pause itself.
    static_length = min(segment.length, protocol.static_probe_window_frames)
    static_authorities = _build_adaptive_windows(
        video,
        segment,
        static_length,
        protocol,
        hop_divisor=protocol.static_probe_hop_divisor,
    )
    primary_keys = {item.window.key for item in primary_windows}
    static_windows = tuple(
        estimate
        for item in static_authorities
        if item.key not in primary_keys
        for estimate in (_estimate_fixed_static_probe(video, item, candidate, protocol),)
        if estimate is not None
    )
    windows = tuple(
        sorted(
            primary_windows + static_windows,
            key=lambda item: (item.window.start, item.window.stop, item.state),
        )
    )
    rates, static_support = _known_interval_rates(segment, windows)
    duration = segment.length - 1
    known = sum(value is not None for value in rates)
    known_fraction = known / duration
    if known_fraction < protocol.minimum_known_interval_union_fraction:
        return _abstaining_segment(
            segment,
            proposal,
            "insufficient_periodic_plus_static_interval_union",
            adaptive_window_frames=window_length,
            windows=windows,
            known_fraction=known_fraction,
            largest_gap=max((stop - start for start, stop in _missing_runs(rates)), default=0),
            interval_rates=tuple(rates),
        )
    if proposal.state == "static_zero_rate":
        proposal_rate = 0.0
        proposal_period = float(segment.length)
    elif proposal.frequency is not None:
        proposal_rate = proposal.frequency
        proposal_period = 1.0 / proposal.frequency
    else:
        raise RuntimeError("non-ambiguous proposal lost its state")
    filled_rates, filled_flags, largest_gap = _fill_ambiguous_intervals(
        rates,
        static_support,
        proposal_rate=proposal_rate,
        proposal_period=proposal_period,
        protocol=protocol,
    )
    if filled_rates is None:
        return _abstaining_segment(
            segment,
            proposal,
            "ambiguous_interval_gap_exceeds_frozen_period_fraction",
            adaptive_window_frames=window_length,
            windows=windows,
            known_fraction=known_fraction,
            largest_gap=largest_gap,
            interval_rates=tuple(rates),
        )
    float_count = float(np.sum(np.asarray(filled_rates, dtype=np.float64)))
    static_zero_only = all(value == 0.0 for value in filled_rates)
    return SegmentEstimateV2(
        segment=segment,
        proposal=proposal,
        adaptive_window_frames=window_length,
        windows=windows,
        status="eligible",
        abstention_reason=None,
        known_interval_union_fraction=known_fraction,
        largest_ambiguous_gap_intervals=largest_gap,
        interval_rates=tuple(filled_rates),
        interval_filled=filled_flags,
        float_count=float_count,
        static_zero_only=static_zero_only,
        diagnostic_stages=_DIAGNOSTIC_STAGE_ORDER,
    )


def estimate_representation_v2(
    video: RepresentationVideo,
    candidate: SpectralCandidateV2,
    config: SegmentLocalSpectralConfigV2,
) -> ViewEstimateV2:
    """Estimate one representation without any label or fallback candidate."""

    if candidate not in config.candidates:
        raise ValueError("candidate is outside the frozen v2 selector")
    segment_estimates = tuple(
        _estimate_segment(video, segment, candidate, config.fixed)
        for segment in video.segments
    )
    periodic_confidences = tuple(
        window.confidence
        for segment in segment_estimates
        for window in segment.windows
        if window.state == "periodic"
    )
    confidence = (
        float(np.mean(np.asarray(periodic_confidences, dtype=np.float64)))
        if periodic_confidences
        else 0.0
    )
    if len(segment_estimates) > 1:
        return ViewEstimateV2(
            video_id=video.video_id,
            candidate_id=candidate.canonical_id,
            status="undefined_multi_segment",
            abstention_reason="multiple_authorized_segments_never_sum",
            float_count=None,
            rounded_count=None,
            static_zero_only=False,
            periodic_confidence=confidence,
            segments=segment_estimates,
        )
    if not segment_estimates:
        return ViewEstimateV2(
            video_id=video.video_id,
            candidate_id=candidate.canonical_id,
            status="abstain",
            abstention_reason="no_authorized_segment",
            float_count=None,
            rounded_count=None,
            static_zero_only=False,
            periodic_confidence=confidence,
            segments=(),
        )
    only = segment_estimates[0]
    if only.status == "abstain" or only.float_count is None:
        return ViewEstimateV2(
            video_id=video.video_id,
            candidate_id=candidate.canonical_id,
            status="abstain",
            abstention_reason=only.abstention_reason,
            float_count=None,
            rounded_count=None,
            static_zero_only=False,
            periodic_confidence=confidence,
            segments=(only,),
        )
    return ViewEstimateV2(
        video_id=video.video_id,
        candidate_id=candidate.canonical_id,
        status="eligible",
        abstention_reason=None,
        float_count=only.float_count,
        rounded_count=_round_half_up(only.float_count),
        static_zero_only=only.static_zero_only,
        periodic_confidence=confidence,
        segments=(only,),
    )


SyntheticKindV2 = Literal["count", "variable_tempo", "corruption", "null", "reset"]


@dataclass(frozen=True, slots=True)
class SyntheticCaseSpecV2:
    case_id: str
    family: str
    kind: SyntheticKindV2
    frames: int
    dimension: Literal[34, 512]
    target_count: float | None
    replicate: int = 0
    profile: str = "constant"


@dataclass(frozen=True, slots=True)
class SyntheticCaseV2:
    spec: SyntheticCaseSpecV2
    video: RepresentationVideo
    source_sha256: str


def synthetic_case_plan_v2() -> tuple[SyntheticCaseSpecV2, ...]:
    """Return the frozen v2 family plan without allocating feature arrays."""

    specs: list[SyntheticCaseSpecV2] = []
    for dimension in (34, 512):
        for count in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40):
            specs.append(
                SyntheticCaseSpecV2(
                    f"v2.count.t256.c{count}.d{dimension}",
                    "constant_tempo_count",
                    "count",
                    256,
                    dimension,
                    float(count),
                )
            )
        for frames in (192, 384):
            for count in (4, 12, 24):
                specs.append(
                    SyntheticCaseSpecV2(
                        f"v2.count.t{frames}.c{count}.d{dimension}",
                        "duration_count",
                        "count",
                        frames,
                        dimension,
                        float(count),
                    )
                )
        for count in (4, 8, 16, 24):
            for profile in ("ramp_up", "up_then_down", "sinusoidal_tempo"):
                specs.append(
                    SyntheticCaseSpecV2(
                        f"v2.tempo.{profile}.c{count}.d{dimension}",
                        "variable_tempo",
                        "variable_tempo",
                        256,
                        dimension,
                        float(count),
                        profile=profile,
                    )
                )
        for count in (8, 16):
            for support in ("active_support_60", "active_support_80"):
                specs.append(
                    SyntheticCaseSpecV2(
                        f"v2.support.{support}.c{count}.d{dimension}",
                        "active_support",
                        "count",
                        256,
                        dimension,
                        float(count),
                        profile=support,
                    )
                )
            for harmonic in (
                "second_harmonic_amp_0.75",
                "second_harmonic_amp_1.25",
                "subharmonic_amp_0.5",
                "second_harmonic_amp_1.25_phase_pi_over_2",
                "asymmetric_duty_30_percent",
                "alternating_cycle_amplitude",
            ):
                specs.append(
                    SyntheticCaseSpecV2(
                        f"v2.harmonic.{harmonic}.c{count}.d{dimension}",
                        "harmonic_stress",
                        "corruption",
                        256,
                        dimension,
                        float(count),
                        profile=harmonic,
                    )
                )
            for corruption in (
                "affine_drift",
                "amplitude_ramp_0.5_to_1.5",
                "noise_sigma_0.02",
                "feature_dropout_20_percent",
                "deterministic_mask_flicker",
                "contiguous_joint_feature_occlusion",
            ):
                specs.append(
                    SyntheticCaseSpecV2(
                        f"v2.corrupt.{corruption}.c{count}.d{dimension}",
                        "corruption",
                        "corruption",
                        256,
                        dimension,
                        float(count),
                        profile=corruption,
                    )
                )
    for family in _NULL_FAMILIES:
        for replicate in range(64):
            dimension: Literal[34, 512] = 34 if replicate % 2 == 0 else 512
            specs.append(
                SyntheticCaseSpecV2(
                    f"v2.null.{family}.r{replicate:02d}.d{dimension}",
                    family,
                    "null",
                    256,
                    dimension,
                    None,
                    replicate=replicate,
                    profile=family,
                )
            )
    for replicate in range(64):
        dimension = 34 if replicate % 2 == 0 else 512
        specs.append(
            SyntheticCaseSpecV2(
                f"v2.reset.two_segment.r{replicate:02d}.d{dimension}",
                "two_segment_reset",
                "reset",
                256,
                dimension,
                None,
                replicate=replicate,
                profile="two_segment_reset",
            )
        )
    identifiers = tuple(item.case_id for item in specs)
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("v2 synthetic case IDs are not unique")
    return tuple(specs)


def _case_seed(seed: int, case_id: str) -> int:
    digest = hashlib.sha256(f"v2\0{int(seed)}\0{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


@lru_cache(maxsize=8)
def _fixed_mixing(dimension: int, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(_case_seed(seed, f"mixing.d{dimension}"))
    matrix = rng.normal(size=(dimension, 8))
    orthogonal, _ = np.linalg.qr(matrix, mode="reduced")
    result = np.asarray(orthogonal, dtype=np.float64)
    result.setflags(write=False)
    return result


def _tempo_profile(frames: int, profile: str) -> NDArray[np.float64]:
    intervals = frames - 1
    time = np.linspace(0.0, 1.0, intervals, dtype=np.float64)
    if profile == "ramp_up":
        speed = np.linspace(0.5, 1.5, intervals, dtype=np.float64)
    elif profile == "up_then_down":
        speed = np.interp(time, (0.0, 0.5, 1.0), (0.5, 1.5, 0.5))
    elif profile == "sinusoidal_tempo":
        speed = 1.0 + 0.4 * np.sin(2.0 * np.pi * time)
    elif profile in ("active_support_60", "active_support_80"):
        support = 0.6 if profile == "active_support_60" else 0.8
        edge = (1.0 - support) / 2.0
        speed = ((time >= edge) & (time <= 1.0 - edge)).astype(np.float64)
    else:
        speed = np.ones(intervals, dtype=np.float64)
    return speed / float(np.sum(speed))


def _periodic_latent(
    frames: int,
    count: float,
    profile: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    increments = _tempo_profile(frames, profile)
    phase = np.zeros(frames, dtype=np.float64)
    phase[1:] = 2.0 * np.pi * count * np.cumsum(increments)
    phase[-1] = 2.0 * np.pi * count
    amplitude = np.ones(frames, dtype=np.float64)
    if profile == "amplitude_ramp_0.5_to_1.5":
        amplitude = np.linspace(0.5, 1.5, frames, dtype=np.float64)
    base_sin = np.sin(phase)
    base_cos = np.cos(phase)
    second_sin = np.sin(2.0 * phase)
    second_cos = np.cos(2.0 * phase)
    if profile == "second_harmonic_amp_0.75":
        base_sin += 0.75 * second_sin
        base_cos += 0.75 * second_cos
    elif profile == "second_harmonic_amp_1.25":
        base_sin += 1.25 * second_sin
        base_cos += 1.25 * second_cos
    elif profile == "subharmonic_amp_0.5":
        base_sin += 0.5 * np.sin(0.5 * phase)
    elif profile == "second_harmonic_amp_1.25_phase_pi_over_2":
        base_sin += 1.25 * np.sin(2.0 * phase + np.pi / 2.0)
        base_cos += 1.25 * np.cos(2.0 * phase + np.pi / 2.0)
    elif profile == "asymmetric_duty_30_percent":
        fractional = np.mod(phase / (2.0 * np.pi), 1.0)
        base_sin = np.where(fractional < 0.3, fractional / 0.3, (1.0 - fractional) / 0.7)
        base_sin = 2.0 * base_sin - 1.0
    elif profile == "alternating_cycle_amplitude":
        cycle_index = np.floor(phase / (2.0 * np.pi)).astype(np.int64)
        amplitude *= np.where(cycle_index % 2 == 0, 0.55, 1.45)
    latent = np.stack(
        (
            amplitude * base_sin,
            amplitude * base_cos,
            amplitude * second_sin,
            amplitude * second_cos,
            amplitude * np.sin(phase + np.pi / 4.0),
            amplitude * np.cos(phase + np.pi / 4.0),
            amplitude * np.sin(3.0 * phase),
            amplitude * np.cos(3.0 * phase),
        ),
        axis=1,
    )
    return latent, phase


def _raw_pose_from_phase(phase: NDArray[np.float64]) -> NDArray[np.float64]:
    joints = np.arange(17, dtype=np.float64)
    base_x = (joints % 5.0 - 2.0) / 5.0
    base_y = (joints // 5.0 - 1.5) / 5.0
    raw = np.empty((phase.size, 17, 2), dtype=np.float64)
    joint_phase = joints * (np.pi / 17.0)
    raw[:, :, 0] = base_x[None, :] + 0.12 * np.sin(phase[:, None] + joint_phase[None, :])
    raw[:, :, 1] = base_y[None, :] + 0.10 * np.cos(phase[:, None] - joint_phase[None, :])
    return raw


def _source_digest(
    features: NDArray[np.float64],
    raw: NDArray[np.float64],
    mask: NDArray[np.bool_],
    valid: NDArray[np.bool_],
) -> str:
    digest = hashlib.sha256()
    digest.update(b"segment_local_spectral_single_v2_synthetic_source\0")
    digest.update(np.ascontiguousarray(features, dtype="<f4").tobytes())
    digest.update(np.ascontiguousarray(raw, dtype="<f4").tobytes())
    digest.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    digest.update(np.ascontiguousarray(valid, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def materialize_synthetic_case_v2(
    spec: SyntheticCaseSpecV2,
    *,
    seed: int,
) -> SyntheticCaseV2:
    """Materialize a case; every null/reset replicate changes source bytes."""

    rng = np.random.default_rng(_case_seed(seed, spec.case_id))
    frames = spec.frames
    mask = np.ones((frames, 17), dtype=np.bool_)
    valid = np.ones(frames, dtype=np.bool_)
    segments: tuple[AuthorizedSegment, ...] = (AuthorizedSegment("segment-0", 0, frames),)
    if spec.kind in ("count", "variable_tempo", "corruption"):
        if spec.target_count is None:
            raise RuntimeError("positive v2 case requires generation truth")
        latent, phase = _periodic_latent(frames, spec.target_count, spec.profile)
        features = latent @ _fixed_mixing(spec.dimension, seed).T
        raw = _raw_pose_from_phase(phase)
        if spec.profile == "affine_drift":
            drift = np.linspace(-0.25, 0.25, frames, dtype=np.float64)
            features += drift[:, None] * np.linspace(0.5, 1.0, spec.dimension)[None, :]
        elif spec.profile == "noise_sigma_0.02":
            features += rng.normal(0.0, 0.02, size=features.shape)
        elif spec.profile == "feature_dropout_20_percent":
            dropped = rng.choice(spec.dimension, size=max(1, spec.dimension // 5), replace=False)
            features[:, dropped] = 0.0
        elif spec.profile == "deterministic_mask_flicker":
            flicker = np.asarray((0, 1, 2, 3), dtype=np.int64)
            frames_with_flicker = np.flatnonzero(np.arange(frames) % 4 < 2)
            mask[np.ix_(frames_with_flicker, flicker)] = False
        elif spec.profile == "contiguous_joint_feature_occlusion":
            start = frames * 2 // 5
            stop = frames * 3 // 5
            occluded = np.asarray((0, 1, 2, 3, 5, 6), dtype=np.int64)
            mask[start:stop, occluded] = False
            features[start:stop, : max(1, spec.dimension // 5)] = 0.0
        elif spec.profile == "amplitude_ramp_0.5_to_1.5":
            raw *= np.linspace(0.5, 1.5, frames)[:, None, None]
    elif spec.kind == "null":
        time = np.linspace(-1.0, 1.0, frames, dtype=np.float64)
        raw_constant = rng.normal(0.0, 0.1, size=(17, 2))
        raw = np.broadcast_to(raw_constant, (frames, 17, 2)).copy()
        if spec.profile == "constant":
            level = rng.normal(0.0, 0.5, size=spec.dimension)
            features = np.broadcast_to(level, (frames, spec.dimension)).copy()
        elif spec.profile == "drift_only":
            direction = rng.normal(size=spec.dimension)
            direction /= max(float(np.linalg.norm(direction)), _EPSILON)
            intercept = rng.normal(0.0, 0.2, size=spec.dimension)
            slope = rng.uniform(0.5, 1.5)
            features = intercept[None, :] + slope * time[:, None] * direction[None, :]
        elif spec.profile == "white_noise":
            features = rng.normal(0.0, 1.0, size=(frames, spec.dimension))
        elif spec.profile == "random_walk":
            step_scale = rng.uniform(0.07, 0.13)
            features = np.cumsum(
                rng.normal(0.0, step_scale, size=(frames, spec.dimension)),
                axis=0,
            )
        elif spec.profile == "ar1_rho_0.9":
            rho = rng.uniform(0.87, 0.93)
            innovations = rng.normal(
                0.0,
                math.sqrt(1.0 - rho**2),
                size=(frames, spec.dimension),
            )
            features = np.zeros_like(innovations)
            features[0] = rng.normal(size=spec.dimension)
            for index in range(1, frames):
                features[index] = rho * features[index - 1] + innovations[index]
        elif spec.profile == "time_shuffled_periodic":
            latent, phase = _periodic_latent(frames, rng.uniform(6.0, 10.0), "constant")
            permutation = rng.permutation(frames)
            features = (latent @ _fixed_mixing(spec.dimension, seed).T)[permutation]
            raw = _raw_pose_from_phase(phase)[permutation]
        elif spec.profile == "independent_incoherent_frequency_phase":
            frequencies = rng.uniform(1.0 / 128.0, 1.0 / 4.0, size=spec.dimension)
            phases = rng.uniform(-np.pi, np.pi, size=spec.dimension)
            grid = np.arange(frames, dtype=np.float64)
            features = np.sin(2.0 * np.pi * grid[:, None] * frequencies + phases[None, :])
        elif spec.profile == "constant_coordinates_oscillating_masks":
            level = rng.normal(0.0, 0.5, size=spec.dimension)
            features = np.broadcast_to(level, (frames, spec.dimension)).copy()
            phase_offset = int(rng.integers(0, 8))
            for frame in range(frames):
                mask[frame, (phase_offset + frame // 4) % 8] = False
                mask[frame, 8 + (phase_offset + frame // 4) % 8] = False
        else:
            raise ValueError(f"unknown v2 null profile: {spec.profile}")
    elif spec.kind == "reset":
        count = rng.uniform(10.0, 14.0)
        latent, phase = _periodic_latent(frames, count, "constant")
        scales = rng.uniform(0.8, 1.2, size=spec.dimension)
        features = (latent @ _fixed_mixing(spec.dimension, seed).T) * scales[None, :]
        raw = _raw_pose_from_phase(phase) + rng.normal(0.0, 0.005, size=(1, 17, 2))
        valid[112:144] = False
        mask[112:144] = False
        features[112:144] = 0.0
        raw[112:144] = 0.0
        segments = (
            AuthorizedSegment("segment-0", 0, 112),
            AuthorizedSegment("segment-1", 144, 256),
        )
    else:
        raise ValueError(f"unknown v2 synthetic kind: {spec.kind}")
    raw[~mask] = 0.0
    source_sha256 = _source_digest(features, raw, mask, valid)
    return SyntheticCaseV2(
        spec=spec,
        video=RepresentationVideo(
            video_id=spec.case_id,
            features=np.asarray(features, dtype=np.float32),
            raw_xy=np.asarray(raw, dtype=np.float32),
            joint_mask=mask,
            valid_mask=valid,
            segments=segments,
        ),
        source_sha256=source_sha256,
    )


def _replace_features(
    video: RepresentationVideo,
    features: NDArray[np.float64],
    *,
    reverse_time: bool = False,
) -> RepresentationVideo:
    raw = np.asarray(video.raw_xy)
    mask = np.asarray(video.joint_mask)
    valid = np.asarray(video.valid_mask)
    if reverse_time:
        raw = raw[::-1]
        mask = mask[::-1]
        valid = valid[::-1]
    return RepresentationVideo(
        video_id=video.video_id,
        features=np.asarray(features, dtype=np.float32),
        raw_xy=raw,
        joint_mask=mask,
        valid_mask=valid,
        segments=video.segments,
    )


def synthetic_invariance_variants_v2(
    case: SyntheticCaseV2,
    *,
    seed: int,
) -> Mapping[str, RepresentationVideo]:
    features = np.asarray(case.video.features, dtype=np.float64)
    dimension = features.shape[1]
    rng = np.random.default_rng(_case_seed(seed, f"invariance\0{case.spec.case_id}"))
    signs = np.where(rng.random(dimension) < 0.5, -1.0, 1.0)
    permutation = rng.permutation(dimension)
    direction = rng.normal(size=dimension)
    direction /= max(float(np.linalg.norm(direction)), _EPSILON)
    projected = features @ direction
    householder = features - 2.0 * projected[:, None] * direction[None, :]
    return {
        "reverse": _replace_features(case.video, features[::-1], reverse_time=True),
        "sign_flip": _replace_features(case.video, features * signs[None, :]),
        "feature_permutation": _replace_features(case.video, features[:, permutation]),
        "orthogonal_mixing": _replace_features(case.video, householder),
        "scale": _replace_features(case.video, features * 1.7),
    }


def _one_sided_cp_upper(failures: int, trials: int, confidence: float) -> float:
    if trials < 1 or failures < 0 or failures > trials:
        raise ValueError("CP counts must satisfy 0 <= failures <= trials")
    if failures == trials:
        return 1.0
    return float(beta_distribution.ppf(confidence, failures + 1, trials - failures))


@dataclass(frozen=True, slots=True)
class PositiveFamilyScoreV2:
    family: str
    trials: int
    eligible: int
    eligible_rate: float
    nmae: float
    obo: float


@dataclass(frozen=True, slots=True)
class NullFamilyScoreV2:
    family: str
    trials: int
    false_positive_or_wrong_state: int
    one_sided_cp_ucb: float
    hard_pass: bool


@dataclass(frozen=True, slots=True)
class SyntheticCandidateScoreV2:
    candidate: SpectralCandidateV2
    seed: int
    positive_total: int
    positive_eligible: int
    positive_eligible_rate: float
    overall_nmae: float
    overall_obo: float
    variable_tempo_nmae: float
    positive_family_scores: tuple[PositiveFamilyScoreV2, ...]
    invariance_agreement: tuple[tuple[str, float], ...]
    null_total: int
    null_false_positive_or_wrong_state: int
    null_cp_ucb: float
    null_family_scores: tuple[NullFamilyScoreV2, ...]
    reset_total: int
    reset_video_total_errors: int
    abstention_rate: float
    hard_pass: bool
    failed_gates: tuple[str, ...]

    def __post_init__(self) -> None:
        tuple_fields = (
            self.positive_family_scores,
            self.invariance_agreement,
            self.null_family_scores,
            self.failed_gates,
        )
        if any(not isinstance(value, tuple) for value in tuple_fields):
            raise TypeError("synthetic score nested containers must be tuples")
        if self.hard_pass != (not self.failed_gates):
            raise ValueError("hard pass must replay from failed gates")

    @property
    def rank(self) -> tuple[float, float, float, float, float, str]:
        return (
            self.null_cp_ucb,
            self.variable_tempo_nmae,
            self.overall_nmae,
            -self.overall_obo,
            self.abstention_rate,
            self.candidate.canonical_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate.canonical_id,
            "seed": self.seed,
            "positive_total": self.positive_total,
            "positive_eligible": self.positive_eligible,
            "positive_eligible_rate": self.positive_eligible_rate,
            "overall_nmae": self.overall_nmae,
            "overall_obo": self.overall_obo,
            "variable_tempo_nmae": self.variable_tempo_nmae,
            "positive_family_scores": {
                item.family: {
                    "trials": item.trials,
                    "eligible": item.eligible,
                    "eligible_rate": item.eligible_rate,
                    "nmae": item.nmae,
                    "obo": item.obo,
                }
                for item in self.positive_family_scores
            },
            "invariance_rounded_agreement": dict(self.invariance_agreement),
            "null_total": self.null_total,
            "null_false_positive_or_wrong_state": self.null_false_positive_or_wrong_state,
            "null_worst_family_one_sided_95pct_cp_ucb": self.null_cp_ucb,
            "null_family_scores": {
                item.family: {
                    "trials": item.trials,
                    "false_positive_or_wrong_state": item.false_positive_or_wrong_state,
                    "one_sided_95pct_cp_ucb": item.one_sided_cp_ucb,
                    "hard_pass": item.hard_pass,
                }
                for item in self.null_family_scores
            },
            "reset_total": self.reset_total,
            "reset_video_total_errors": self.reset_video_total_errors,
            "abstention_rate": self.abstention_rate,
            "hard_pass": self.hard_pass,
            "failed_gates": list(self.failed_gates),
        }


def _validated_plan(config: SegmentLocalSpectralConfigV2) -> tuple[SyntheticCaseSpecV2, ...]:
    plan = synthetic_case_plan_v2()
    positives = tuple(item for item in plan if item.kind in ("count", "variable_tempo", "corruption"))
    nulls = tuple(item for item in plan if item.kind == "null")
    resets = tuple(item for item in plan if item.kind == "reset")
    observed_counts = tuple(
        (family, sum(item.family == family for item in positives))
        for family, _ in _POSITIVE_FAMILY_COUNTS
    )
    if observed_counts != _POSITIVE_FAMILY_COUNTS:
        raise RuntimeError("v2 positive family plan changed")
    for family in _NULL_FAMILIES:
        members = tuple(item for item in nulls if item.family == family)
        if len(members) != config.synthetic.null_trials_per_family:
            raise RuntimeError(f"v2 null family {family} count changed")
        if tuple(item.replicate for item in members) != tuple(range(64)):
            raise RuntimeError(f"v2 null family {family} replicate order changed")
    if len(resets) != config.synthetic.reset_trials:
        raise RuntimeError("v2 reset count changed")
    return plan


def score_synthetic_candidate_v2(
    config: SegmentLocalSpectralConfigV2,
    candidate: SpectralCandidateV2,
    *,
    seed: int,
) -> SyntheticCandidateScoreV2:
    """Run the real v2 algorithm over every preregistered synthetic family."""

    if seed not in (config.synthetic.selector_seed, config.synthetic.heldout_seed):
        raise ValueError("synthetic seed is outside selector/held-out identities")
    if candidate not in config.candidates:
        raise ValueError("candidate is outside the frozen v2 grid")
    normalized_errors: list[float] = []
    absolute_errors: list[float] = []
    variable_errors: list[float] = []
    family_errors: dict[str, list[float]] = {}
    family_absolute: dict[str, list[float]] = {}
    family_eligible = {family: 0 for family, _ in _POSITIVE_FAMILY_COUNTS}
    invariance_success = {name: 0 for name in _INVARIANCE_NAMES}
    invariance_trials = {name: 0 for name in _INVARIANCE_NAMES}
    null_trials = {family: 0 for family in _NULL_FAMILIES}
    null_failures = {family: 0 for family in _NULL_FAMILIES}
    null_digests = {family: set() for family in _NULL_FAMILIES}
    positive_eligible = 0
    reset_total = 0
    reset_errors = 0
    for spec in _validated_plan(config):
        case = materialize_synthetic_case_v2(spec, seed=seed)
        estimate = estimate_representation_v2(case.video, candidate, config)
        if spec.kind in ("count", "variable_tempo", "corruption"):
            if spec.target_count is None:
                raise RuntimeError("positive v2 case lost generation truth")
            prediction = estimate.rounded_count if estimate.rounded_count is not None else 0
            absolute = abs(float(prediction) - spec.target_count)
            normalized = absolute / spec.target_count
            absolute_errors.append(absolute)
            normalized_errors.append(normalized)
            family_errors.setdefault(spec.family, []).append(normalized)
            family_absolute.setdefault(spec.family, []).append(absolute)
            if estimate.status == "eligible":
                positive_eligible += 1
                family_eligible[spec.family] += 1
            if spec.kind == "variable_tempo":
                variable_errors.append(normalized)
            for name, variant in synthetic_invariance_variants_v2(case, seed=seed).items():
                transformed = estimate_representation_v2(variant, candidate, config)
                invariance_trials[name] += 1
                if (
                    estimate.status == "eligible"
                    and transformed.status == "eligible"
                    and estimate.rounded_count == transformed.rounded_count
                ):
                    invariance_success[name] += 1
        elif spec.kind == "null":
            null_trials[spec.family] += 1
            null_digests[spec.family].add(case.source_sha256)
            wrong_state = estimate.status == "eligible" and not estimate.static_zero_only
            positive_count = estimate.rounded_count is not None and estimate.rounded_count > 0
            if wrong_state or positive_count:
                null_failures[spec.family] += 1
        else:
            reset_total += 1
            if (
                estimate.status != "undefined_multi_segment"
                or estimate.float_count is not None
                or estimate.rounded_count is not None
            ):
                reset_errors += 1
    for family in _NULL_FAMILIES:
        if len(null_digests[family]) != config.synthetic.null_trials_per_family:
            raise RuntimeError(f"v2 null family {family} is not truly seed-varying")
    positive_total = len(normalized_errors)
    positive_family_scores = tuple(
        PositiveFamilyScoreV2(
            family=family,
            trials=len(family_errors[family]),
            eligible=family_eligible[family],
            eligible_rate=family_eligible[family] / len(family_errors[family]),
            nmae=float(np.mean(family_errors[family])),
            obo=float(np.mean(np.asarray(family_absolute[family]) <= 1.0)),
        )
        for family, _ in _POSITIVE_FAMILY_COUNTS
    )
    invariance = tuple(
        (name, invariance_success[name] / invariance_trials[name])
        for name in _INVARIANCE_NAMES
    )
    null_family_scores_list: list[NullFamilyScoreV2] = []
    for family in _NULL_FAMILIES:
        upper = _one_sided_cp_upper(
            null_failures[family],
            null_trials[family],
            config.synthetic.null_false_eligible_cp_confidence,
        )
        null_family_scores_list.append(
            NullFamilyScoreV2(
                family=family,
                trials=null_trials[family],
                false_positive_or_wrong_state=null_failures[family],
                one_sided_cp_ucb=upper,
                hard_pass=upper <= config.synthetic.null_false_eligible_cp_ucb_maximum,
            )
        )
    null_family_scores = tuple(null_family_scores_list)
    positive_rate = positive_eligible / positive_total
    overall_nmae = float(np.mean(normalized_errors))
    overall_obo = float(np.mean(np.asarray(absolute_errors) <= 1.0))
    variable_nmae = float(np.mean(variable_errors))
    failures: list[str] = []
    if positive_rate < config.synthetic.positive_eligible_minimum:
        failures.append("positive_eligible_rate")
    if overall_nmae > config.synthetic.overall_nmae_maximum:
        failures.append("overall_nmae")
    if overall_obo < config.synthetic.overall_obo_minimum:
        failures.append("overall_obo")
    for item in positive_family_scores:
        if item.eligible_rate < config.synthetic.positive_family_eligible_minimum:
            failures.append(f"positive_family_eligible_rate:{item.family}")
        if item.nmae > config.synthetic.positive_family_nmae_maximum:
            failures.append(f"positive_family_nmae:{item.family}")
        if item.obo < config.synthetic.positive_family_obo_minimum:
            failures.append(f"positive_family_obo:{item.family}")
    if variable_nmae > config.synthetic.variable_tempo_nmae_maximum:
        failures.append("variable_tempo_nmae")
    if any(
        value < config.synthetic.invariance_rounded_agreement_minimum
        for _, value in invariance
    ):
        failures.append("invariance_rounded_agreement")
    for item in null_family_scores:
        if not item.hard_pass:
            failures.append(f"null_false_positive_or_wrong_state:{item.family}")
    if reset_errors > config.synthetic.reset_video_total_errors_maximum:
        failures.append("reset_video_total_errors")
    return SyntheticCandidateScoreV2(
        candidate=candidate,
        seed=seed,
        positive_total=positive_total,
        positive_eligible=positive_eligible,
        positive_eligible_rate=positive_rate,
        overall_nmae=overall_nmae,
        overall_obo=overall_obo,
        variable_tempo_nmae=variable_nmae,
        positive_family_scores=positive_family_scores,
        invariance_agreement=invariance,
        null_total=sum(null_trials.values()),
        null_false_positive_or_wrong_state=sum(null_failures.values()),
        null_cp_ucb=max(item.one_sided_cp_ucb for item in null_family_scores),
        null_family_scores=null_family_scores,
        reset_total=reset_total,
        reset_video_total_errors=reset_errors,
        abstention_rate=1.0 - positive_rate,
        hard_pass=not failures,
        failed_gates=tuple(failures),
    )


@dataclass(frozen=True, slots=True)
class SyntheticGridReportV2:
    schema_version: Literal[2]
    config_sha256: str
    seed: int
    predecessor_denial_result_sha256: str
    predecessor_denial_result_bytes: int
    predecessor_denial_receipt_sha256: str
    predecessor_denial_receipt_bytes: int
    scores: tuple[SyntheticCandidateScoreV2, ...]
    selected_candidate_id: str | None
    hard_pass_count: int
    grants_train337_authority: Literal[False] = False
    grants_dev84_authority: Literal[False] = False
    grants_test105_authority: Literal[False] = False

    def __post_init__(self) -> None:
        if not isinstance(self.scores, tuple):
            raise TypeError("grid scores must be an exact tuple")
        if (
            self.predecessor_denial_result_sha256 != PREDECESSOR_DENIAL_RESULT_SHA256
            or self.predecessor_denial_result_bytes != PREDECESSOR_DENIAL_RESULT_BYTES
            or self.predecessor_denial_receipt_sha256 != PREDECESSOR_DENIAL_RECEIPT_SHA256
            or self.predecessor_denial_receipt_bytes != PREDECESSOR_DENIAL_RECEIPT_BYTES
        ):
            raise ValueError("predecessor denial hash/byte lineage changed")
        passes = tuple(item for item in self.scores if item.hard_pass)
        if self.hard_pass_count != len(passes):
            raise ValueError("grid hard-pass count does not replay")
        expected = min(passes, key=lambda item: item.rank).candidate.canonical_id if passes else None
        if self.selected_candidate_id != expected:
            raise ValueError("selected candidate does not replay from hard passes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "config_sha256": self.config_sha256,
            "seed": self.seed,
            "predecessor_denial_result_sha256": self.predecessor_denial_result_sha256,
            "predecessor_denial_result_bytes": self.predecessor_denial_result_bytes,
            "predecessor_denial_receipt_sha256": self.predecessor_denial_receipt_sha256,
            "predecessor_denial_receipt_bytes": self.predecessor_denial_receipt_bytes,
            "scores": [item.to_dict() for item in self.scores],
            "selected_candidate_id": self.selected_candidate_id,
            "hard_pass_count": self.hard_pass_count,
            "grants_train337_authority": self.grants_train337_authority,
            "grants_dev84_authority": self.grants_dev84_authority,
            "grants_test105_authority": self.grants_test105_authority,
        }


def evaluate_synthetic_grid_v2(
    config: SegmentLocalSpectralConfigV2,
    *,
    seed: int,
) -> SyntheticGridReportV2:
    """Evaluate the entire finite grid; a pass still grants no data authority."""

    scores = tuple(
        score_synthetic_candidate_v2(config, candidate, seed=seed)
        for candidate in config.candidates
    )
    passes = tuple(item for item in scores if item.hard_pass)
    selected = min(passes, key=lambda item: item.rank) if passes else None
    return SyntheticGridReportV2(
        schema_version=2,
        config_sha256=config.fingerprint,
        seed=seed,
        predecessor_denial_result_sha256=PREDECESSOR_DENIAL_RESULT_SHA256,
        predecessor_denial_result_bytes=PREDECESSOR_DENIAL_RESULT_BYTES,
        predecessor_denial_receipt_sha256=PREDECESSOR_DENIAL_RECEIPT_SHA256,
        predecessor_denial_receipt_bytes=PREDECESSOR_DENIAL_RECEIPT_BYTES,
        scores=scores,
        selected_candidate_id=(selected.candidate.canonical_id if selected is not None else None),
        hard_pass_count=len(passes),
    )
