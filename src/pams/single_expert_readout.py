"""Segment-local spectral single-expert readout and label-free gates.

This module is an independently inferred baseline completion.  It is not an
author-disclosed PAMS period head and must not be described as an exact
reproduction of the paper's undisclosed baseline readout.

The implementation deliberately separates three boundaries:

* :class:`RepresentationVideo` is the representation-agnostic science core.
* :class:`MechanismVideo` binds the learned ``L`` representation to the raw
  COCO-17 ``R`` control and the ``E0``/``Epi`` mechanism controls.
* synthetic selection and the train337 gate have no target-label argument.

No window may cross an authorized identity/reset segment.  A multi-segment
video exposes per-segment floating estimates for diagnostics, but its video
total is always undefined; summing those estimates is not a prediction API.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from pydantic import model_validator
from scipy.stats import beta as beta_distribution

from pams.config import StrictModel

WindowSize = Literal[64, 96]
PowerReduction = Literal["sum", "per_feature_band_normalized_mean"]
MinimumCycles = float
RidgeAggregation = Literal[
    "ridge_integral",
    "confidence_weighted_median_rate",
]
PeakShare = float
ViewName = Literal["L", "R", "E0", "Epi"]
EstimateStatus = Literal["eligible", "abstain", "undefined_multi_segment"]

_WINDOWS: tuple[WindowSize, WindowSize] = (64, 96)
_POWER_REDUCTIONS: tuple[PowerReduction, PowerReduction] = (
    "sum",
    "per_feature_band_normalized_mean",
)
_MINIMUM_CYCLES: tuple[MinimumCycles, MinimumCycles] = (1.0, 1.5)
_RIDGE_AGGREGATIONS: tuple[RidgeAggregation, RidgeAggregation] = (
    "ridge_integral",
    "confidence_weighted_median_rate",
)
_PEAK_SHARES: tuple[PeakShare, PeakShare] = (0.1, 0.2)
_ACTION_JOINTS: tuple[int, ...] = (7, 8, 9, 10, 13, 14, 15, 16)
_EPSILON = np.finfo(np.float64).eps


class CandidateAxes(StrictModel):
    """The exact five-axis, 32-member selector space."""

    window_frames: tuple[WindowSize, WindowSize]
    power_reduction: tuple[PowerReduction, PowerReduction]
    minimum_cycles: tuple[float, float]
    ridge_aggregation: tuple[RidgeAggregation, RidgeAggregation]
    minimum_peak_share: tuple[float, float]

    @model_validator(mode="after")
    def validate_exact_axes(self) -> CandidateAxes:
        expected = (
            _WINDOWS,
            _POWER_REDUCTIONS,
            _MINIMUM_CYCLES,
            _RIDGE_AGGREGATIONS,
            _PEAK_SHARES,
        )
        actual = (
            self.window_frames,
            self.power_reduction,
            self.minimum_cycles,
            self.ridge_aggregation,
            self.minimum_peak_share,
        )
        if actual != expected:
            raise ValueError(f"candidate axes must equal the frozen 32-grid: {expected!r}")
        return self


class FixedSpectralProtocol(StrictModel):
    """Non-selectable method identity shared by every candidate."""

    feature_order: Literal["level_only"]
    embedding_context: Literal["full_authorized_segment_once"]
    embedding_execution_mode: Literal["eval_deterministic_no_grad_no_optimizer_update"]
    position_indices: Literal["absolute_native_unchanged"]
    temporal_detrend: Literal["per_dimension_linear"]
    taper: Literal["hann_periodic_false"]
    fft_length: Literal["4_times_window"]
    hop: Literal["window_div_4"]
    upper_frequency_cycles_per_frame: float
    absolute_minimum_frequency_cycles_per_frame: float
    signed_vector_acf_minimum: float
    confidence_weight: Literal["sqrt_peak_share_times_positive_signed_acf"]
    ridge_boundary: Literal["nearest_center_extension"]
    ridge_interpolation: Literal["linear_between_centers"]
    ridge_integral: Literal["source_frame_interval_trapezoid"]
    duration: Literal["segment_length_minus_1"]
    video_rounding: Literal["half_up_once_exactly_one_segment"]
    multi_segment_video_total: Literal["undefined_never_sum"]
    coordinate_schema: Literal["coco17_xy"]
    raw_feature_policy: Literal["window_stable_joints_only"]
    minimum_joints_per_frame: Literal[8]
    minimum_stable_joints: Literal[6]
    minimum_stable_action_joints: Literal[3]
    prohibited_operations: tuple[
        Literal[
            "activity_trim",
            "harmonic_bonus",
            "expert_vote",
            "feature_minmax",
            "count_clip",
        ],
        ...,
    ]

    @model_validator(mode="after")
    def validate_prohibited_operations(self) -> FixedSpectralProtocol:
        expected = (
            "activity_trim",
            "harmonic_bonus",
            "expert_vote",
            "feature_minmax",
            "count_clip",
        )
        if self.prohibited_operations != expected:
            raise ValueError(f"prohibited_operations must equal {expected!r}")
        numerics = (
            self.upper_frequency_cycles_per_frame,
            self.absolute_minimum_frequency_cycles_per_frame,
            self.signed_vector_acf_minimum,
        )
        if numerics != (0.25, 1.0 / 128.0, 0.1):
            raise ValueError("fixed spectral numeric constants were changed")
        return self


class SyntheticGateProtocol(StrictModel):
    """Frozen synthetic calibration and no-fallback held-out replay."""

    selector_seed: Literal[2026]
    heldout_seed: Literal[3407]
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
        "pooled_all_generation_truth_positives_only_no_family_thresholds"
    ]
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

    @model_validator(mode="after")
    def validate_synthetic_identity(self) -> SyntheticGateProtocol:
        if self.dimensions != (34, 512):
            raise ValueError("synthetic dimensions must be exactly (34, 512)")
        expected = (
            "null_cp_ucb",
            "variable_tempo_nmae",
            "overall_nmae",
            "negative_obo",
            "abstention_rate",
            "canonical_candidate_id",
        )
        if self.ranking != expected:
            raise ValueError(f"synthetic ranking must equal {expected!r}")
        thresholds = (
            self.positive_eligible_minimum,
            self.overall_nmae_maximum,
            self.overall_obo_minimum,
            self.variable_tempo_nmae_maximum,
            self.invariance_rounded_agreement_minimum,
            self.null_false_eligible_cp_confidence,
            self.null_false_eligible_cp_ucb_maximum,
        )
        if thresholds != (0.95, 0.08, 0.95, 0.1, 0.95, 0.95, 0.05):
            raise ValueError("synthetic gate thresholds were changed")
        return self


class Train337TransformRecipe(StrictModel):
    """Exact label-free transformations bound into every train337 row."""

    recipe_id: Literal["segment_local_spectral_train337_transforms_v1"]
    reverse: Literal["source_frames_reverse_then_full_segment_reencode"]
    warp_075: Literal["endpoint_preserving_time_warp_0.75_then_full_segment_reencode"]
    warp_125: Literal["endpoint_preserving_time_warp_1.25_then_full_segment_reencode"]
    duplicate_time: Literal["duplicate_source_time_then_full_segment_reencode"]
    legal_split: Literal["authorized_single_segment_two_part_float_additivity_only"]
    raw_rotation: Literal["coco17_xy_rotation_plus_minus_15_degrees"]
    raw_scale: Literal["coco17_xy_scale_0.85_and_1.15"]
    raw_joint_dropout: Literal["deterministic_supported_joint_dropout"]
    learned_augmentations: Literal["two_frozen_cycleback_augmentation_views_reencode"]
    static_null: Literal["constant_coordinates_same_authority_and_masks"]
    second_derangement: Literal[
        "independent_preencoder_segment_derangement_distinct_seed_and_map"
    ]

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()


class Train337GateProtocol(StrictModel):
    """Frozen, label-free expansion gate for the 337 training videos."""

    expected_video_count: Literal[337]
    hash_subset_seed: Literal[2026]
    hash_subset_size: Literal[64]
    one_segment_sufficient_share_minimum: float
    one_segment_eligible_minimum: float
    low_band_boundary_share_maximum_exclusive: float
    high_band_boundary_share_maximum_exclusive: float
    rounded_mode_share_maximum_exclusive: float
    rounded_bin_minimum: Literal[8]
    reverse_relative_error_median_maximum: float
    reverse_relative_error_p90_maximum: float
    time_warp_disagreement_median_maximum: float
    time_warp_disagreement_p90_maximum: float
    split_additivity_median_maximum: float
    split_additivity_p90_maximum: float
    raw_invariance_rounded_agreement_minimum: float
    learned_augmentation_disagreement_median_maximum: float
    learned_augmentation_disagreement_p90_maximum: float
    static_null_positive_share_maximum: float
    shuffle_confidence_ratio_median_maximum: float
    real_minus_shuffle_peak_margin_median_minimum: float
    mechanism_rule: Literal["L_must_pass_Epi_must_fail_E0_report_only"]
    label_policy: Literal["no_action_or_count_targets"]
    authority_adapter_status: Literal["unwired_fail_closed"]
    transform_recipe: Train337TransformRecipe

    @model_validator(mode="after")
    def validate_exact_thresholds(self) -> Train337GateProtocol:
        thresholds = (
            self.one_segment_sufficient_share_minimum,
            self.one_segment_eligible_minimum,
            self.low_band_boundary_share_maximum_exclusive,
            self.high_band_boundary_share_maximum_exclusive,
            self.rounded_mode_share_maximum_exclusive,
            self.reverse_relative_error_median_maximum,
            self.reverse_relative_error_p90_maximum,
            self.time_warp_disagreement_median_maximum,
            self.time_warp_disagreement_p90_maximum,
            self.split_additivity_median_maximum,
            self.split_additivity_p90_maximum,
            self.raw_invariance_rounded_agreement_minimum,
            self.learned_augmentation_disagreement_median_maximum,
            self.learned_augmentation_disagreement_p90_maximum,
            self.static_null_positive_share_maximum,
            self.shuffle_confidence_ratio_median_maximum,
            self.real_minus_shuffle_peak_margin_median_minimum,
        )
        expected = (
            0.9,
            0.9,
            0.1,
            0.1,
            0.5,
            0.05,
            0.2,
            0.1,
            0.25,
            0.1,
            0.25,
            0.9,
            0.1,
            0.25,
            0.05,
            0.75,
            0.05,
        )
        if thresholds != expected:
            raise ValueError("train337 gate thresholds were changed")
        return self


class SegmentLocalSpectralConfig(StrictModel):
    """Strict semantic configuration for ``segment_local_spectral_single_v1``."""

    schema_version: Literal[1]
    key: Literal["segment_local_spectral_single_v1"]
    classification: Literal["inferred_label_free_single_expert_baseline"]
    eligible_as_exact_author_baseline: Literal[False]
    axes: CandidateAxes
    fixed: FixedSpectralProtocol
    synthetic: SyntheticGateProtocol
    train337: Train337GateProtocol

    @property
    def candidates(self) -> tuple[SpectralCandidate, ...]:
        candidates = tuple(
            SpectralCandidate(window, power, cycles, aggregation, peak_share)
            for window in self.axes.window_frames
            for power in self.axes.power_reduction
            for cycles in self.axes.minimum_cycles
            for aggregation in self.axes.ridge_aggregation
            for peak_share in self.axes.minimum_peak_share
        )
        if len(candidates) != 32 or len({item.canonical_id for item in candidates}) != 32:
            raise RuntimeError("frozen candidate grid did not produce 32 unique members")
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
class SpectralCandidate:
    """One member of the frozen 32-item selector grid."""

    window_frames: WindowSize
    power_reduction: PowerReduction
    minimum_cycles: MinimumCycles
    ridge_aggregation: RidgeAggregation
    minimum_peak_share: PeakShare

    def __post_init__(self) -> None:
        if self.window_frames not in _WINDOWS:
            raise ValueError("window_frames is outside frozen grid")
        if self.power_reduction not in _POWER_REDUCTIONS:
            raise ValueError("power_reduction is outside frozen grid")
        if self.minimum_cycles not in _MINIMUM_CYCLES:
            raise ValueError("minimum_cycles is outside frozen grid")
        if self.ridge_aggregation not in _RIDGE_AGGREGATIONS:
            raise ValueError("ridge_aggregation is outside frozen grid")
        if self.minimum_peak_share not in _PEAK_SHARES:
            raise ValueError("minimum_peak_share is outside frozen grid")

    @property
    def canonical_id(self) -> str:
        power = "sum" if self.power_reduction == "sum" else "bnmean"
        aggregation = "integral" if self.ridge_aggregation == "ridge_integral" else "wmedian"
        return (
            f"w{self.window_frames}.{power}.c{self.minimum_cycles:g}."
            f"{aggregation}.ps{self.minimum_peak_share:g}"
        )


def load_segment_local_spectral_config(path: str | Path) -> SegmentLocalSpectralConfig:
    """Load a strict config; unknown or relaxed values are rejected."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {config_path}")
    config = SegmentLocalSpectralConfig.model_validate(payload)
    _ = config.candidates
    return config


def _immutable_array(value: ArrayLike, dtype: np.dtype[Any] | type[Any]) -> NDArray[Any]:
    contiguous = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    immutable = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype)
    return immutable.reshape(contiguous.shape)


@dataclass(frozen=True, slots=True)
class AuthorizedSegment:
    """Exact half-open stable eligible-frame range authorized upstream.

    This is narrower than a merely associated identity span: every frame in
    the range must be valid and carry the minimum joint support.  It is the
    full context encoded once with absolute native positional encoding.
    """

    segment_id: str
    start: int
    stop: int

    def __post_init__(self) -> None:
        identifier = str(self.segment_id).strip()
        if not identifier:
            raise ValueError("segment_id must be non-empty")
        if isinstance(self.start, bool) or isinstance(self.stop, bool):
            raise TypeError("segment bounds must be integers")
        start = int(self.start)
        stop = int(self.stop)
        if start != self.start or stop != self.stop or start < 0 or stop <= start:
            raise ValueError("segment must satisfy 0 <= start < stop")
        object.__setattr__(self, "segment_id", identifier)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "stop", stop)

    @property
    def length(self) -> int:
        return self.stop - self.start


def _validate_segments(
    segments: Sequence[AuthorizedSegment],
    frame_count: int,
) -> tuple[AuthorizedSegment, ...]:
    frozen = tuple(segments)
    if len({segment.segment_id for segment in frozen}) != len(frozen):
        raise ValueError("authorized segment_id values must be unique")
    previous_stop = 0
    for index, segment in enumerate(frozen):
        if segment.stop > frame_count:
            raise ValueError("authorized segment exceeds representation length")
        if index and segment.start < previous_stop:
            raise ValueError("authorized segments must be ordered and non-overlapping")
        previous_stop = segment.stop
    return frozen


@dataclass(frozen=True, slots=True, eq=False)
class RepresentationVideo:
    """One feature view plus the common COCO-17 geometry authority."""

    video_id: str
    features: NDArray[np.float32]
    raw_xy: NDArray[np.float32]
    joint_mask: NDArray[np.bool_]
    valid_mask: NDArray[np.bool_]
    segments: tuple[AuthorizedSegment, ...]

    def __post_init__(self) -> None:
        video_id = str(self.video_id).strip()
        if not video_id:
            raise ValueError("video_id must be non-empty")
        features = np.asarray(self.features, dtype=np.float32)
        raw_xy = np.asarray(self.raw_xy, dtype=np.float32)
        joint_mask = np.asarray(self.joint_mask, dtype=np.bool_)
        valid_mask = np.asarray(self.valid_mask, dtype=np.bool_)
        if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] < 1:
            raise ValueError("features must have shape [frames, dimensions]")
        frame_count = int(features.shape[0])
        if raw_xy.shape != (frame_count, 17, 2):
            raise ValueError("raw_xy must have shape [frames, 17, 2]")
        if joint_mask.shape != (frame_count, 17):
            raise ValueError("joint_mask must have shape [frames, 17]")
        if valid_mask.shape != (frame_count,):
            raise ValueError("valid_mask must have shape [frames]")
        if bool(np.any(joint_mask & ~valid_mask[:, None])):
            raise ValueError("invalid frames must not claim supported joints")
        if not np.isfinite(features[valid_mask]).all():
            raise ValueError("features must be finite on valid frames")
        if not np.isfinite(raw_xy[joint_mask & valid_mask[:, None]]).all():
            raise ValueError("raw_xy must be finite on valid, supported joints")
        if bool(np.any(features[~valid_mask] != 0.0)):
            raise ValueError("features on invalid/padded frames must be exact zero")
        if bool(np.any(raw_xy[~joint_mask] != 0.0)):
            raise ValueError("raw_xy on weak/invalid joints must be exact zero")
        canonical_features = np.array(features, copy=True, order="C")
        canonical_raw = np.array(raw_xy, copy=True, order="C")
        segments = _validate_segments(self.segments, frame_count)
        for segment in segments:
            segment_valid = valid_mask[segment.start : segment.stop]
            if not bool(np.all(segment_valid)):
                raise ValueError("authorized segments must contain only valid frames")
            supported = np.sum(joint_mask[segment.start : segment.stop], axis=1)
            if bool(np.any(supported < 8)):
                raise ValueError("authorized segment frames must support at least eight joints")
        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "features", _immutable_array(canonical_features, np.float32))
        object.__setattr__(self, "raw_xy", _immutable_array(canonical_raw, np.float32))
        object.__setattr__(self, "joint_mask", _immutable_array(joint_mask, np.bool_))
        object.__setattr__(self, "valid_mask", _immutable_array(valid_mask, np.bool_))
        object.__setattr__(self, "segments", segments)

    @property
    def frame_count(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True, slots=True)
class SegmentPermutation:
    """One global-index permutation confined to an authorized segment."""

    segment_id: str
    source_indices: tuple[int, ...]
    source_pose_sha256: str
    deranged_pose_sha256: str


@dataclass(frozen=True, slots=True)
class TemporalDerangementReceipt:
    """Identity of the pre-encoder, segment-local Epi derangement."""

    method: Literal["pose_pre_encoder_segment_derangement_v1"]
    video_id: str
    seed: int
    permutations: tuple[SegmentPermutation, ...]
    permutation_map_sha256: str

    def __post_init__(self) -> None:
        if self.method != "pose_pre_encoder_segment_derangement_v1":
            raise ValueError("unsupported temporal derangement method")
        if isinstance(self.seed, bool) or int(self.seed) != self.seed:
            raise TypeError("derangement seed must be an integer")
        segment_ids = tuple(item.segment_id for item in self.permutations)
        if not segment_ids or len(set(segment_ids)) != len(segment_ids):
            raise ValueError("derangement permutations require unique segment IDs")
        if not isinstance(self.permutation_map_sha256, str):
            raise ValueError("permutation_map_sha256 must be a lowercase SHA-256")
        digest = self.permutation_map_sha256
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("permutation_map_sha256 must be a lowercase SHA-256")
        for item in self.permutations:
            for name, value in (
                ("source_pose_sha256", item.source_pose_sha256),
                ("deranged_pose_sha256", item.deranged_pose_sha256),
            ):
                if len(value) != 64 or any(
                    character not in "0123456789abcdef" for character in value
                ):
                    raise ValueError(f"{name} must be a lowercase SHA-256")
        video_id = str(self.video_id).strip()
        if not video_id or video_id != self.video_id:
            raise ValueError("derangement receipt video_id must be canonical and non-empty")
        actual = _permutation_digest(
            self.seed,
            self.permutations,
            video_id=video_id,
        )
        if digest != actual:
            raise ValueError(f"permutation map SHA-256 mismatch: expected={digest}, actual={actual}")
        object.__setattr__(self, "video_id", video_id)
        object.__setattr__(self, "permutation_map_sha256", digest)


def _permutation_digest(
    seed: int,
    permutations: Sequence[SegmentPermutation],
    *,
    video_id: str = "",
) -> str:
    payload = {
        "method": "pose_pre_encoder_segment_derangement_v1",
        "video_id": video_id,
        "seed": int(seed),
        "permutations": [
            {
                "segment_id": item.segment_id,
                "source_indices": list(item.source_indices),
                "source_pose_sha256": item.source_pose_sha256,
                "deranged_pose_sha256": item.deranged_pose_sha256,
            }
            for item in permutations
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _segment_pose_digest(
    video_id: str,
    raw_xy: NDArray[np.float32],
    joint_mask: NDArray[np.bool_],
    valid_mask: NDArray[np.bool_],
    segment: AuthorizedSegment,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"coco17_xy_segment_bytes_v1\0")
    digest.update(video_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(segment.segment_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(np.asarray((segment.start, segment.stop), dtype="<i8").tobytes())
    digest.update(
        np.ascontiguousarray(raw_xy[segment.start : segment.stop], dtype="<f4").tobytes()
    )
    digest.update(
        np.ascontiguousarray(joint_mask[segment.start : segment.stop], dtype=np.uint8).tobytes()
    )
    digest.update(
        np.ascontiguousarray(valid_mask[segment.start : segment.stop], dtype=np.uint8).tobytes()
    )
    return digest.hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class DerangedPoseInput:
    """Pose/mask arrays to encode for Epi, with their immutable receipt."""

    raw_xy: NDArray[np.float32]
    joint_mask: NDArray[np.bool_]
    valid_mask: NDArray[np.bool_]
    receipt: TemporalDerangementReceipt


def build_preencoder_segment_derangement(
    video_id: str,
    raw_xy: ArrayLike,
    joint_mask: ArrayLike,
    valid_mask: ArrayLike,
    segments: Sequence[AuthorizedSegment],
    *,
    seed: int,
) -> DerangedPoseInput:
    """Derange source poses within each segment before calling the L encoder.

    The returned arrays are encoder inputs, not shuffled embeddings.  Each
    authorized segment receives an independent permutation with no fixed
    point.  A segment shorter than two frames is rejected rather than silently
    falling back to an identity mapping.
    """

    identifier = str(video_id).strip()
    if not identifier:
        raise ValueError("derangement video_id must be non-empty")
    xy = np.asarray(raw_xy, dtype=np.float32)
    mask = np.asarray(joint_mask, dtype=np.bool_)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if xy.ndim != 3 or xy.shape[1:] != (17, 2):
        raise ValueError("raw_xy must have shape [frames, 17, 2]")
    if mask.shape != xy.shape[:2] or valid.shape != (xy.shape[0],):
        raise ValueError("joint_mask/valid_mask shapes do not match raw_xy")
    authority = RepresentationVideo(
        video_id=identifier,
        features=np.zeros((xy.shape[0], 1), dtype=np.float32),
        raw_xy=xy,
        joint_mask=mask,
        valid_mask=valid,
        segments=tuple(segments),
    )
    xy = np.asarray(authority.raw_xy)
    mask = np.asarray(authority.joint_mask)
    valid = np.asarray(authority.valid_mask)
    frozen_segments = authority.segments
    output_xy = np.array(xy, copy=True, order="C")
    output_mask = np.array(mask, copy=True, order="C")
    output_valid = np.array(valid, copy=True, order="C")
    rng = np.random.default_rng(int(seed))
    permutations: list[SegmentPermutation] = []
    for segment in frozen_segments:
        if segment.length < 2:
            raise ValueError("derangement requires every authorized segment to have >=2 frames")
        local = np.arange(segment.length, dtype=np.int64)
        permutation = rng.permutation(segment.length)
        attempts = 1
        while np.any(permutation == local):
            permutation = rng.permutation(segment.length)
            attempts += 1
            if attempts > 10_000:
                raise RuntimeError("failed to construct deterministic segment derangement")
        source = permutation + segment.start
        output_xy[segment.start : segment.stop] = xy[source]
        output_mask[segment.start : segment.stop] = mask[source]
        output_valid[segment.start : segment.stop] = valid[source]
        permutations.append(
            SegmentPermutation(
                segment_id=segment.segment_id,
                source_indices=tuple(source.tolist()),
                source_pose_sha256=_segment_pose_digest(identifier, xy, mask, valid, segment),
                deranged_pose_sha256=_segment_pose_digest(
                    identifier,
                    output_xy,
                    output_mask,
                    output_valid,
                    segment,
                ),
            )
        )
    receipt = TemporalDerangementReceipt(
        method="pose_pre_encoder_segment_derangement_v1",
        video_id=identifier,
        seed=int(seed),
        permutations=tuple(permutations),
        permutation_map_sha256=_permutation_digest(
            seed,
            permutations,
            video_id=identifier,
        ),
    )
    return DerangedPoseInput(
        raw_xy=_immutable_array(output_xy, np.float32),
        joint_mask=_immutable_array(output_mask, np.bool_),
        valid_mask=_immutable_array(output_valid, np.bool_),
        receipt=receipt,
    )


@dataclass(frozen=True, slots=True)
class WindowAuthority:
    """One window that passed segment, frame, and joint-mask eligibility."""

    segment_id: str
    start: int
    stop: int
    center: float
    stable_joints: tuple[int, ...]

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.segment_id, self.start, self.stop)


@dataclass(frozen=True, slots=True)
class SegmentWindowPlan:
    """Candidate and authorized windows for one reset segment."""

    segment: AuthorizedSegment
    candidate_window_count: int
    windows: tuple[WindowAuthority, ...]


def build_window_plan(
    video: RepresentationVideo,
    candidate: SpectralCandidate,
) -> tuple[SegmentWindowPlan, ...]:
    """Build the sole window authority shared by L/R/E0/Epi.

    Starts are segment-relative fixed-hop positions.  No terminal window is
    appended off-grid.  Geometry-ineligible windows remain visible through
    ``candidate_window_count`` but cannot enter any view's spectral estimate.
    """

    window = candidate.window_frames
    hop = window // 4
    plans: list[SegmentWindowPlan] = []
    for segment in video.segments:
        starts = tuple(range(segment.start, segment.stop - window + 1, hop))
        authorized: list[WindowAuthority] = []
        for start in starts:
            stop = start + window
            if not bool(np.all(video.valid_mask[start:stop])):
                continue
            mask = np.asarray(video.joint_mask[start:stop], dtype=np.bool_)
            if bool(np.any(np.sum(mask, axis=1) < 8)):
                continue
            stable = np.flatnonzero(np.all(mask, axis=0))
            if stable.size < 6:
                continue
            stable_action = np.intersect1d(
                stable,
                np.asarray(_ACTION_JOINTS, dtype=np.int64),
                assume_unique=True,
            )
            if stable_action.size < 3:
                continue
            authorized.append(
                WindowAuthority(
                    segment_id=segment.segment_id,
                    start=start,
                    stop=stop,
                    center=start + (window - 1) / 2.0,
                    stable_joints=tuple(int(value) for value in stable),
                )
            )
        plans.append(
            SegmentWindowPlan(
                segment=segment,
                candidate_window_count=len(starts),
                windows=tuple(authorized),
            )
        )
    return tuple(plans)


def _linear_detrend(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("spectral values must have shape [time>=2, dimensions]")
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


@dataclass(frozen=True, slots=True)
class WindowEstimate:
    """Spectral outcome for one geometry-authorized window."""

    key: tuple[str, int, int]
    center: float
    accepted: bool
    abstention_reason: str | None
    frequency: float | None
    peak_share: float
    signed_vector_acf: float
    confidence: float
    low_band_boundary: bool
    high_band_boundary: bool
    informative_dimensions: int

    def __post_init__(self) -> None:
        segment_id, start, stop = self.key
        if not segment_id or start < 0 or stop <= start:
            raise ValueError("window estimate key is invalid")
        if not np.isfinite(self.center) or not start <= self.center < stop:
            raise ValueError("window center must lie inside its source window")
        if not 0.0 <= self.peak_share <= 1.0:
            raise ValueError("window peak_share must be in [0, 1]")
        if not -1.0 <= self.signed_vector_acf <= 1.0:
            raise ValueError("window signed_vector_acf must be in [-1, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("window confidence must be in [0, 1]")
        if self.informative_dimensions < 0:
            raise ValueError("informative_dimensions must be non-negative")
        if self.accepted:
            if self.frequency is None or not np.isfinite(self.frequency) or self.frequency <= 0.0:
                raise ValueError("accepted window requires a positive finite frequency")
            if self.abstention_reason is not None or self.confidence <= 0.0:
                raise ValueError("accepted window cannot carry abstention or zero confidence")
        elif self.confidence != 0.0:
            raise ValueError("rejected window confidence must be exact zero")


def _spectral_window(
    features: NDArray[np.float64],
    authority: WindowAuthority,
    candidate: SpectralCandidate,
) -> WindowEstimate:
    values = _linear_detrend(features)
    tapered = values * np.hanning(values.shape[0])[:, None]
    fft_size = 4 * candidate.window_frames
    power = np.abs(np.fft.rfft(tapered, n=fft_size, axis=0)) ** 2
    frequencies = np.fft.rfftfreq(fft_size)
    low = max(1.0 / 128.0, candidate.minimum_cycles / candidate.window_frames)
    allowed = (frequencies >= low) & (frequencies <= 1.0 / 4.0)
    indices = np.flatnonzero(allowed)
    band = power[indices]
    band_totals = np.sum(band, axis=0)
    informative = np.isfinite(band_totals) & (band_totals > _EPSILON)
    informative_count = int(np.count_nonzero(informative))
    if informative_count == 0:
        return WindowEstimate(
            key=authority.key,
            center=authority.center,
            accepted=False,
            abstention_reason="no_informative_dimensions",
            frequency=None,
            peak_share=0.0,
            signed_vector_acf=-1.0,
            confidence=0.0,
            low_band_boundary=False,
            high_band_boundary=False,
            informative_dimensions=0,
        )
    informative_band = band[:, informative]
    if candidate.power_reduction == "sum":
        aggregate = np.sum(informative_band, axis=1)
    else:
        normalized = informative_band / band_totals[informative][None, :]
        aggregate = np.mean(normalized, axis=1)
    aggregate_total = float(np.sum(aggregate))
    if not np.isfinite(aggregate_total) or aggregate_total <= _EPSILON:
        return WindowEstimate(
            key=authority.key,
            center=authority.center,
            accepted=False,
            abstention_reason="zero_band_power",
            frequency=None,
            peak_share=0.0,
            signed_vector_acf=-1.0,
            confidence=0.0,
            low_band_boundary=False,
            high_band_boundary=False,
            informative_dimensions=informative_count,
        )
    local_peak = int(np.argmax(aggregate))
    frequency = float(frequencies[indices[local_peak]])
    peak_share = float(np.clip(aggregate[local_peak] / aggregate_total, 0.0, 1.0))
    lag = int(np.clip(math.floor(1.0 / frequency + 0.5), 1, values.shape[0] - 1))
    signed_acf = _signed_vector_acf(values[:, informative], lag)
    accepted = bool(
        peak_share >= candidate.minimum_peak_share and signed_acf >= 0.1
    )
    reason: str | None = None
    if peak_share < candidate.minimum_peak_share:
        reason = "peak_share_below_threshold"
    elif signed_acf < 0.1:
        reason = "signed_vector_acf_below_threshold"
    confidence = math.sqrt(peak_share * max(signed_acf, 0.0)) if accepted else 0.0
    return WindowEstimate(
        key=authority.key,
        center=authority.center,
        accepted=accepted,
        abstention_reason=reason,
        frequency=frequency,
        peak_share=peak_share,
        signed_vector_acf=signed_acf,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        low_band_boundary=local_peak == 0,
        high_band_boundary=local_peak == len(indices) - 1,
        informative_dimensions=informative_count,
    )


def _weighted_median(values: NDArray[np.float64], weights: NDArray[np.float64]) -> float:
    if values.ndim != 1 or weights.shape != values.shape or values.size == 0:
        raise ValueError("weighted median requires non-empty matching vectors")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    total = float(np.sum(ordered_weights))
    if total <= _EPSILON:
        raise ValueError("weighted median requires positive total weight")
    index = int(np.searchsorted(np.cumsum(ordered_weights), total / 2.0, side="left"))
    return float(ordered_values[min(index, values.size - 1)])


def _ridge_integral(
    segment: AuthorizedSegment,
    accepted: Sequence[WindowEstimate],
) -> float:
    centers = np.asarray([item.center for item in accepted], dtype=np.float64)
    rates = np.asarray([item.frequency for item in accepted], dtype=np.float64)
    frame_grid = np.arange(segment.start, segment.stop, dtype=np.float64)
    ridge = np.interp(frame_grid, centers, rates, left=rates[0], right=rates[-1])
    interval_widths = np.diff(frame_grid)
    return float(np.sum(0.5 * (ridge[:-1] + ridge[1:]) * interval_widths))


def _round_half_up_once(value: float) -> int:
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("count must be finite and non-negative before final rounding")
    return int(math.floor(value + 0.5))


@dataclass(frozen=True, slots=True)
class SegmentEstimate:
    """Per-segment float estimate; never implicitly summed across resets."""

    segment: AuthorizedSegment
    candidate_window_count: int
    available_window_count: int
    accepted_window_count: int
    single_window_estimate: bool
    status: Literal["eligible", "abstain"]
    abstention_reason: str | None
    float_count: float | None
    windows: tuple[WindowEstimate, ...]

    def __post_init__(self) -> None:
        counts = (
            self.candidate_window_count,
            self.available_window_count,
            self.accepted_window_count,
        )
        if any(isinstance(value, bool) or value < 0 for value in counts):
            raise ValueError("segment window counts must be non-negative integers")
        if not self.accepted_window_count <= self.available_window_count <= self.candidate_window_count:
            raise ValueError("segment window counts are inconsistent")
        if len(self.windows) != self.available_window_count:
            raise ValueError("segment windows do not match available_window_count")
        if sum(item.accepted for item in self.windows) != self.accepted_window_count:
            raise ValueError("segment windows do not match accepted_window_count")
        if any(item.key[0] != self.segment.segment_id for item in self.windows):
            raise ValueError("segment estimate contains a foreign window")
        if self.status == "eligible":
            if self.accepted_window_count < 1 or self.float_count is None:
                raise ValueError("eligible segment requires an accepted window and float count")
            if not np.isfinite(self.float_count) or self.float_count < 0.0:
                raise ValueError("eligible segment float count must be finite and non-negative")
            if self.abstention_reason is not None:
                raise ValueError("eligible segment cannot carry an abstention reason")
            if self.single_window_estimate != (self.accepted_window_count == 1):
                raise ValueError("single_window_estimate flag is inconsistent")
        else:
            if self.float_count is not None or self.accepted_window_count != 0:
                raise ValueError("abstaining segment cannot expose a count")
            if not self.abstention_reason or self.single_window_estimate:
                raise ValueError("abstaining segment requires a reason and no single-window flag")


@dataclass(frozen=True, slots=True)
class ViewEstimate:
    """One view's video result under one exact selected candidate."""

    video_id: str
    view: ViewName
    candidate_id: str
    status: EstimateStatus
    abstention_reason: str | None
    float_count: float | None
    rounded_count: int | None
    periodic_confidence: float
    mean_peak_share: float
    segments: tuple[SegmentEstimate, ...]

    @property
    def window_keys(self) -> tuple[tuple[str, int, int], ...]:
        return tuple(window.key for segment in self.segments for window in segment.windows)

    def __post_init__(self) -> None:
        if not self.video_id or self.view not in ("L", "R", "E0", "Epi"):
            raise ValueError("view estimate identity is invalid")
        if not self.candidate_id:
            raise ValueError("view estimate candidate_id must be non-empty")
        if not 0.0 <= self.periodic_confidence <= 1.0:
            raise ValueError("periodic_confidence must be in [0, 1]")
        if not 0.0 <= self.mean_peak_share <= 1.0:
            raise ValueError("mean_peak_share must be in [0, 1]")
        if self.status == "eligible":
            if len(self.segments) != 1 or self.segments[0].status != "eligible":
                raise ValueError("eligible video requires exactly one eligible segment")
            if self.float_count is None or self.rounded_count is None:
                raise ValueError("eligible video requires float and rounded count")
            if self.rounded_count != _round_half_up_once(self.float_count):
                raise ValueError("video count must be rounded half-up exactly once")
            if self.abstention_reason is not None:
                raise ValueError("eligible video cannot carry an abstention reason")
        elif self.status == "undefined_multi_segment":
            if len(self.segments) <= 1:
                raise ValueError("undefined_multi_segment requires multiple authority segments")
            if self.float_count is not None or self.rounded_count is not None:
                raise ValueError("multi-segment video total must remain undefined")
            if self.abstention_reason != "multiple_authorized_segments_never_sum":
                raise ValueError("multi-segment status requires the frozen reason")
        else:
            if len(self.segments) > 1:
                raise ValueError("multi-segment video cannot be downgraded to abstain")
            if self.float_count is not None or self.rounded_count is not None:
                raise ValueError("abstaining video cannot expose a count")
            if not self.abstention_reason:
                raise ValueError("abstaining video requires a reason")


def _estimate_from_plan(
    video: RepresentationVideo,
    candidate: SpectralCandidate,
    plan: Sequence[SegmentWindowPlan],
    *,
    view: ViewName,
    raw_control: bool,
) -> ViewEstimate:
    segment_estimates: list[SegmentEstimate] = []
    all_windows: list[WindowEstimate] = []
    for segment_plan in plan:
        outcomes: list[WindowEstimate] = []
        for authority in segment_plan.windows:
            if raw_control:
                stable = np.asarray(authority.stable_joints, dtype=np.int64)
                raw = np.asarray(video.raw_xy[authority.start : authority.stop], dtype=np.float64)
                features = raw[:, stable, :].reshape(candidate.window_frames, -1)
            else:
                features = np.asarray(
                    video.features[authority.start : authority.stop],
                    dtype=np.float64,
                )
            outcomes.append(_spectral_window(features, authority, candidate))
        accepted = tuple(item for item in outcomes if item.accepted)
        if not accepted:
            if segment_plan.candidate_window_count == 0:
                reason = "segment_shorter_than_window"
            elif not segment_plan.windows:
                reason = "no_joint_mask_authorized_window"
            else:
                reason = "no_spectral_window_accepted"
            segment_estimates.append(
                SegmentEstimate(
                    segment=segment_plan.segment,
                    candidate_window_count=segment_plan.candidate_window_count,
                    available_window_count=len(segment_plan.windows),
                    accepted_window_count=0,
                    single_window_estimate=False,
                    status="abstain",
                    abstention_reason=reason,
                    float_count=None,
                    windows=tuple(outcomes),
                )
            )
            all_windows.extend(outcomes)
            continue
        frequencies = np.asarray([item.frequency for item in accepted], dtype=np.float64)
        weights = np.asarray([item.confidence for item in accepted], dtype=np.float64)
        if candidate.ridge_aggregation == "ridge_integral":
            float_count = _ridge_integral(segment_plan.segment, accepted)
        else:
            rate = _weighted_median(frequencies, weights)
            float_count = rate * float(segment_plan.segment.length - 1)
        segment_estimates.append(
            SegmentEstimate(
                segment=segment_plan.segment,
                candidate_window_count=segment_plan.candidate_window_count,
                available_window_count=len(segment_plan.windows),
                accepted_window_count=len(accepted),
                single_window_estimate=len(accepted) == 1,
                status="eligible",
                abstention_reason=None,
                float_count=float_count,
                windows=tuple(outcomes),
            )
        )
        all_windows.extend(outcomes)

    if all_windows:
        periodic_confidence = float(np.mean([item.confidence for item in all_windows]))
        mean_peak_share = float(np.mean([item.peak_share for item in all_windows]))
    else:
        periodic_confidence = 0.0
        mean_peak_share = 0.0
    if len(segment_estimates) > 1:
        return ViewEstimate(
            video_id=video.video_id,
            view=view,
            candidate_id=candidate.canonical_id,
            status="undefined_multi_segment",
            abstention_reason="multiple_authorized_segments_never_sum",
            float_count=None,
            rounded_count=None,
            periodic_confidence=periodic_confidence,
            mean_peak_share=mean_peak_share,
            segments=tuple(segment_estimates),
        )
    if not segment_estimates:
        return ViewEstimate(
            video_id=video.video_id,
            view=view,
            candidate_id=candidate.canonical_id,
            status="abstain",
            abstention_reason="no_authorized_segment",
            float_count=None,
            rounded_count=None,
            periodic_confidence=periodic_confidence,
            mean_peak_share=mean_peak_share,
            segments=(),
        )
    only = segment_estimates[0]
    if only.status == "abstain" or only.float_count is None:
        return ViewEstimate(
            video_id=video.video_id,
            view=view,
            candidate_id=candidate.canonical_id,
            status="abstain",
            abstention_reason=only.abstention_reason,
            float_count=None,
            rounded_count=None,
            periodic_confidence=periodic_confidence,
            mean_peak_share=mean_peak_share,
            segments=(only,),
        )
    return ViewEstimate(
        video_id=video.video_id,
        view=view,
        candidate_id=candidate.canonical_id,
        status="eligible",
        abstention_reason=None,
        float_count=only.float_count,
        rounded_count=_round_half_up_once(only.float_count),
        periodic_confidence=periodic_confidence,
        mean_peak_share=mean_peak_share,
        segments=(only,),
    )


def estimate_representation(
    video: RepresentationVideo,
    candidate: SpectralCandidate,
    *,
    view: ViewName = "L",
    raw_control: bool = False,
) -> ViewEstimate:
    """Estimate one view using only segment-authorized fixed-hop windows."""

    return _estimate_from_plan(
        video,
        candidate,
        build_window_plan(video, candidate),
        view=view,
        raw_control=raw_control,
    )


@dataclass(frozen=True, slots=True)
class SegmentEncodingContext:
    """Input/output bytes proving full-segment, not window-local, encoding."""

    segment_id: str
    start: int
    stop: int
    input_pose_sha256: str
    output_features_sha256: str
    repeat_output_features_sha256: str
    context_fingerprint: str


@dataclass(frozen=True, slots=True)
class FullSegmentEncodingReceipt:
    """Lineage contract for one learned/control representation view."""

    view: Literal["L", "E0", "Epi"]
    video_id: str
    context_policy: Literal["full_authorized_segment_absolute_native_pe_v1"]
    execution_mode: Literal["eval_deterministic_no_grad_no_optimizer_update"]
    position_indices: Literal["absolute_native_unchanged"]
    encoder_state_sha256: str
    encoder_config_sha256: str
    encoder_implementation_sha256: str
    initialization_state_sha256: str
    checkpoint_role: Literal["trained_cycleback", "frozen_untrained_initialization"]
    derangement_map_sha256: str | None
    contexts: tuple[SegmentEncodingContext, ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        identifier = str(self.video_id).strip()
        if not identifier or identifier != self.video_id:
            raise ValueError("encoding receipt video_id must be canonical and non-empty")
        if self.view not in ("L", "E0", "Epi"):
            raise ValueError("encoding receipt view is invalid")
        if self.context_policy != "full_authorized_segment_absolute_native_pe_v1":
            raise ValueError("window-local/overlap-add encoding receipts are forbidden")
        if self.execution_mode != "eval_deterministic_no_grad_no_optimizer_update":
            raise ValueError("readout embeddings require deterministic eval-mode inference")
        if self.position_indices != "absolute_native_unchanged":
            raise ValueError("readout embeddings require unchanged absolute native positions")
        if self.checkpoint_role not in (
            "trained_cycleback",
            "frozen_untrained_initialization",
        ):
            raise ValueError("encoding checkpoint_role is invalid")
        for name, value in (
            ("encoder_state_sha256", self.encoder_state_sha256),
            ("encoder_config_sha256", self.encoder_config_sha256),
            ("encoder_implementation_sha256", self.encoder_implementation_sha256),
            ("initialization_state_sha256", self.initialization_state_sha256),
            ("receipt_sha256", self.receipt_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        if self.view == "Epi":
            if self.derangement_map_sha256 is None:
                raise ValueError("Epi encoding receipt requires derangement_map_sha256")
        elif self.derangement_map_sha256 is not None:
            raise ValueError("only Epi may carry derangement_map_sha256")
        if self.checkpoint_role == "frozen_untrained_initialization":
            if self.encoder_state_sha256 != self.initialization_state_sha256:
                raise ValueError("untrained encoder state must equal initialization state")
        elif self.encoder_state_sha256 == self.initialization_state_sha256:
            raise ValueError("trained encoder state must differ from initialization state")
        for context in self.contexts:
            if context.stop <= context.start or context.start < 0:
                raise ValueError("encoding context bounds are invalid")
            for name, value in (
                ("input_pose_sha256", context.input_pose_sha256),
                ("output_features_sha256", context.output_features_sha256),
                ("repeat_output_features_sha256", context.repeat_output_features_sha256),
                ("context_fingerprint", context.context_fingerprint),
            ):
                if len(value) != 64 or any(
                    character not in "0123456789abcdef" for character in value
                ):
                    raise ValueError(f"{name} must be a lowercase SHA-256")
            if context.repeat_output_features_sha256 != context.output_features_sha256:
                raise ValueError("repeated eval-mode encoding bytes must match exactly")
            expected = _encoding_context_fingerprint(
                view=self.view,
                video_id=self.video_id,
                encoder_state_sha256=self.encoder_state_sha256,
                encoder_config_sha256=self.encoder_config_sha256,
                encoder_implementation_sha256=self.encoder_implementation_sha256,
                initialization_state_sha256=self.initialization_state_sha256,
                checkpoint_role=self.checkpoint_role,
                derangement_map_sha256=self.derangement_map_sha256,
                segment_id=context.segment_id,
                start=context.start,
                stop=context.stop,
                input_pose_sha256=context.input_pose_sha256,
                output_features_sha256=context.output_features_sha256,
                repeat_output_features_sha256=context.repeat_output_features_sha256,
            )
            if context.context_fingerprint != expected:
                raise ValueError("segment encoding context fingerprint mismatch")
        actual = _encoding_receipt_digest(
            self.view,
            self.video_id,
            self.encoder_state_sha256,
            self.encoder_config_sha256,
            self.encoder_implementation_sha256,
            self.initialization_state_sha256,
            self.checkpoint_role,
            self.derangement_map_sha256,
            self.contexts,
        )
        if self.receipt_sha256 != actual:
            raise ValueError("full-segment encoding receipt SHA-256 mismatch")
        object.__setattr__(self, "video_id", identifier)


def _feature_segment_digest(
    features: NDArray[np.float32],
    segment: AuthorizedSegment,
) -> str:
    digest = hashlib.sha256()
    digest.update(b"full_segment_embedding_bytes_v1\0")
    digest.update(np.asarray((segment.start, segment.stop), dtype="<i8").tobytes())
    digest.update(
        np.ascontiguousarray(features[segment.start : segment.stop], dtype="<f4").tobytes()
    )
    return digest.hexdigest()


def _encoding_context_fingerprint(
    *,
    view: Literal["L", "E0", "Epi"],
    video_id: str,
    encoder_state_sha256: str,
    encoder_config_sha256: str,
    encoder_implementation_sha256: str,
    initialization_state_sha256: str,
    checkpoint_role: Literal["trained_cycleback", "frozen_untrained_initialization"],
    derangement_map_sha256: str | None,
    segment_id: str,
    start: int,
    stop: int,
    input_pose_sha256: str,
    output_features_sha256: str,
    repeat_output_features_sha256: str,
) -> str:
    payload = {
        "context_policy": "full_authorized_segment_absolute_native_pe_v1",
        "execution_mode": "eval_deterministic_no_grad_no_optimizer_update",
        "position_indices": "absolute_native_unchanged",
        "view": view,
        "video_id": video_id,
        "encoder_state_sha256": encoder_state_sha256,
        "encoder_config_sha256": encoder_config_sha256,
        "encoder_implementation_sha256": encoder_implementation_sha256,
        "initialization_state_sha256": initialization_state_sha256,
        "checkpoint_role": checkpoint_role,
        "derangement_map_sha256": derangement_map_sha256,
        "segment_id": segment_id,
        "start": int(start),
        "stop": int(stop),
        "input_pose_sha256": input_pose_sha256,
        "output_features_sha256": output_features_sha256,
        "repeat_output_features_sha256": repeat_output_features_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _encoding_receipt_digest(
    view: Literal["L", "E0", "Epi"],
    video_id: str,
    encoder_state_sha256: str,
    encoder_config_sha256: str,
    encoder_implementation_sha256: str,
    initialization_state_sha256: str,
    checkpoint_role: Literal["trained_cycleback", "frozen_untrained_initialization"],
    derangement_map_sha256: str | None,
    contexts: Sequence[SegmentEncodingContext],
) -> str:
    payload = {
        "context_policy": "full_authorized_segment_absolute_native_pe_v1",
        "execution_mode": "eval_deterministic_no_grad_no_optimizer_update",
        "position_indices": "absolute_native_unchanged",
        "view": view,
        "video_id": video_id,
        "encoder_state_sha256": encoder_state_sha256,
        "encoder_config_sha256": encoder_config_sha256,
        "encoder_implementation_sha256": encoder_implementation_sha256,
        "initialization_state_sha256": initialization_state_sha256,
        "checkpoint_role": checkpoint_role,
        "derangement_map_sha256": derangement_map_sha256,
        "contexts": [
            {
                "segment_id": item.segment_id,
                "start": item.start,
                "stop": item.stop,
                "input_pose_sha256": item.input_pose_sha256,
                "output_features_sha256": item.output_features_sha256,
                "repeat_output_features_sha256": item.repeat_output_features_sha256,
                "context_fingerprint": item.context_fingerprint,
            }
            for item in contexts
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_full_segment_encoding_receipt(
    *,
    view: Literal["L", "E0", "Epi"],
    video_id: str,
    encoder_state_sha256: str,
    encoder_config_sha256: str,
    encoder_implementation_sha256: str,
    initialization_state_sha256: str,
    checkpoint_role: Literal["trained_cycleback", "frozen_untrained_initialization"],
    features: ArrayLike,
    repeat_features: ArrayLike,
    segments: Sequence[AuthorizedSegment],
    input_pose_sha256_by_segment: Mapping[str, str],
    derangement_map_sha256: str | None = None,
) -> FullSegmentEncodingReceipt:
    """Construct a receipt after full-context per-segment encoding."""

    feature_array = np.asarray(features, dtype=np.float32)
    repeat_array = np.asarray(repeat_features, dtype=np.float32)
    if feature_array.ndim != 2 or repeat_array.shape != feature_array.shape:
        raise ValueError("encoded features must have shape [frames, dimensions]")
    frozen_segments = _validate_segments(segments, int(feature_array.shape[0]))
    identifier = str(video_id).strip()
    if not identifier:
        raise ValueError("encoding receipt video_id must be non-empty")
    if checkpoint_role == "frozen_untrained_initialization":
        if encoder_state_sha256 != initialization_state_sha256:
            raise ValueError("untrained encoder state must equal initialization state")
    elif encoder_state_sha256 == initialization_state_sha256:
        raise ValueError("trained encoder state must differ from initialization state")
    if set(input_pose_sha256_by_segment) != {
        segment.segment_id for segment in frozen_segments
    }:
        raise ValueError("input pose digest keys must equal the authorized segment IDs")
    contexts: list[SegmentEncodingContext] = []
    for segment in frozen_segments:
        try:
            input_digest = input_pose_sha256_by_segment[segment.segment_id]
        except KeyError as error:
            raise ValueError("input pose digests must cover every segment") from error
        output_digest = _feature_segment_digest(feature_array, segment)
        repeat_output_digest = _feature_segment_digest(repeat_array, segment)
        if repeat_output_digest != output_digest:
            raise ValueError("repeated eval-mode segment encoding bytes are not deterministic")
        context_digest = _encoding_context_fingerprint(
            view=view,
            video_id=identifier,
            encoder_state_sha256=encoder_state_sha256,
            encoder_config_sha256=encoder_config_sha256,
            encoder_implementation_sha256=encoder_implementation_sha256,
            initialization_state_sha256=initialization_state_sha256,
            checkpoint_role=checkpoint_role,
            derangement_map_sha256=derangement_map_sha256,
            segment_id=segment.segment_id,
            start=segment.start,
            stop=segment.stop,
            input_pose_sha256=input_digest,
            output_features_sha256=output_digest,
            repeat_output_features_sha256=repeat_output_digest,
        )
        contexts.append(
            SegmentEncodingContext(
                segment_id=segment.segment_id,
                start=segment.start,
                stop=segment.stop,
                input_pose_sha256=input_digest,
                output_features_sha256=output_digest,
                repeat_output_features_sha256=repeat_output_digest,
                context_fingerprint=context_digest,
            )
        )
    return FullSegmentEncodingReceipt(
        view=view,
        video_id=identifier,
        context_policy="full_authorized_segment_absolute_native_pe_v1",
        execution_mode="eval_deterministic_no_grad_no_optimizer_update",
        position_indices="absolute_native_unchanged",
        encoder_state_sha256=encoder_state_sha256,
        encoder_config_sha256=encoder_config_sha256,
        encoder_implementation_sha256=encoder_implementation_sha256,
        initialization_state_sha256=initialization_state_sha256,
        checkpoint_role=checkpoint_role,
        derangement_map_sha256=derangement_map_sha256,
        contexts=tuple(contexts),
        receipt_sha256=_encoding_receipt_digest(
            view,
            identifier,
            encoder_state_sha256,
            encoder_config_sha256,
            encoder_implementation_sha256,
            initialization_state_sha256,
            checkpoint_role,
            derangement_map_sha256,
            contexts,
        ),
    )


@dataclass(frozen=True, slots=True, eq=False)
class MechanismVideo:
    """L/R/E0/Epi views sharing exact geometry and window authority."""

    video_id: str
    learned: NDArray[np.float32]
    untrained: NDArray[np.float32]
    epi: NDArray[np.float32]
    raw_xy: NDArray[np.float32]
    joint_mask: NDArray[np.bool_]
    valid_mask: NDArray[np.bool_]
    segments: tuple[AuthorizedSegment, ...]
    epi_receipt: TemporalDerangementReceipt
    learned_encoding_receipt: FullSegmentEncodingReceipt
    untrained_encoding_receipt: FullSegmentEncodingReceipt
    epi_encoding_receipt: FullSegmentEncodingReceipt

    def __post_init__(self) -> None:
        learned = np.asarray(self.learned, dtype=np.float32)
        untrained = np.asarray(self.untrained, dtype=np.float32)
        epi = np.asarray(self.epi, dtype=np.float32)
        if learned.ndim != 2 or untrained.shape != learned.shape or epi.shape != learned.shape:
            raise ValueError("learned, untrained, and epi must share [frames, dimensions]")
        reference = RepresentationVideo(
            video_id=self.video_id,
            features=learned,
            raw_xy=self.raw_xy,
            joint_mask=self.joint_mask,
            valid_mask=self.valid_mask,
            segments=self.segments,
        )
        receipt_ids = tuple(item.segment_id for item in self.epi_receipt.permutations)
        segment_ids = tuple(item.segment_id for item in reference.segments)
        if self.epi_receipt.video_id != reference.video_id:
            raise ValueError("Epi derangement receipt video_id mismatch")
        if receipt_ids != segment_ids:
            raise ValueError("Epi receipt must cover the exact authorized segments in order")
        replay_xy = np.array(reference.raw_xy, copy=True, order="C")
        replay_joint_mask = np.array(reference.joint_mask, copy=True, order="C")
        replay_valid_mask = np.array(reference.valid_mask, copy=True, order="C")
        for segment, item in zip(reference.segments, self.epi_receipt.permutations, strict=True):
            expected = set(range(segment.start, segment.stop))
            actual = tuple(item.source_indices)
            if len(actual) != segment.length or set(actual) != expected:
                raise ValueError("Epi permutation must be bijective and segment-local")
            if any(source == segment.start + offset for offset, source in enumerate(actual)):
                raise ValueError("Epi permutation must have no fixed points")
            source = np.asarray(actual, dtype=np.int64)
            replay_xy[segment.start : segment.stop] = reference.raw_xy[source]
            replay_joint_mask[segment.start : segment.stop] = reference.joint_mask[source]
            replay_valid_mask[segment.start : segment.stop] = reference.valid_mask[source]
            replayed_digest = _segment_pose_digest(
                reference.video_id,
                replay_xy,
                replay_joint_mask,
                replay_valid_mask,
                segment,
            )
            if replayed_digest != item.deranged_pose_sha256:
                raise ValueError("Epi deranged pose digest does not replay from authority bytes")
        if not np.isfinite(untrained[reference.valid_mask]).all():
            raise ValueError("untrained features must be finite on valid frames")
        if not np.isfinite(epi[reference.valid_mask]).all():
            raise ValueError("Epi features must be finite on valid frames")
        if bool(np.any(untrained[~reference.valid_mask] != 0.0)):
            raise ValueError("untrained features on invalid frames must be exact zero")
        if bool(np.any(epi[~reference.valid_mask] != 0.0)):
            raise ValueError("Epi features on invalid frames must be exact zero")
        original_pose_digests = {
            segment.segment_id: _segment_pose_digest(
                reference.video_id,
                reference.raw_xy,
                reference.joint_mask,
                reference.valid_mask,
                segment,
            )
            for segment in reference.segments
        }
        deranged_pose_digests = {
            item.segment_id: item.deranged_pose_sha256 for item in self.epi_receipt.permutations
        }
        for item in self.epi_receipt.permutations:
            if item.source_pose_sha256 != original_pose_digests[item.segment_id]:
                raise ValueError("Epi source pose bytes differ from mechanism source")
        for receipt, expected_view, feature_array, expected_inputs in (
            (self.learned_encoding_receipt, "L", learned, original_pose_digests),
            (self.untrained_encoding_receipt, "E0", untrained, original_pose_digests),
            (self.epi_encoding_receipt, "Epi", epi, deranged_pose_digests),
        ):
            if receipt.view != expected_view:
                raise ValueError(f"encoding receipt view must be {expected_view}")
            if receipt.video_id != reference.video_id:
                raise ValueError("encoding receipt video_id mismatch")
            if tuple(item.segment_id for item in receipt.contexts) != segment_ids:
                raise ValueError("encoding receipt must cover exact segments in order")
            if expected_view == "Epi" and (
                receipt.derangement_map_sha256 != self.epi_receipt.permutation_map_sha256
            ):
                raise ValueError("Epi encoding receipt/map mismatch")
            for segment, context in zip(reference.segments, receipt.contexts, strict=True):
                if (context.start, context.stop) != (segment.start, segment.stop):
                    raise ValueError("encoding receipt bounds differ from authority")
                if context.input_pose_sha256 != expected_inputs[segment.segment_id]:
                    raise ValueError("encoding receipt input bytes mismatch")
                if context.output_features_sha256 != _feature_segment_digest(
                    feature_array,
                    segment,
                ):
                    raise ValueError("encoding receipt output feature bytes mismatch")
        learned_receipt = self.learned_encoding_receipt
        untrained_receipt = self.untrained_encoding_receipt
        epi_receipt = self.epi_encoding_receipt
        if learned_receipt.checkpoint_role != "trained_cycleback":
            raise ValueError("L must be a trained cycleback checkpoint")
        if epi_receipt.checkpoint_role != "trained_cycleback":
            raise ValueError("Epi must use the trained cycleback checkpoint")
        if untrained_receipt.checkpoint_role != "frozen_untrained_initialization":
            raise ValueError("E0 must be the frozen untrained initialization")
        shared_identity = {
            (
                receipt.encoder_config_sha256,
                receipt.encoder_implementation_sha256,
                receipt.initialization_state_sha256,
            )
            for receipt in (learned_receipt, untrained_receipt, epi_receipt)
        }
        if len(shared_identity) != 1:
            raise ValueError("L/E0/Epi must share encoder config, source, and initialization")
        if learned_receipt.encoder_state_sha256 != epi_receipt.encoder_state_sha256:
            raise ValueError("L and Epi must use the exact same learned encoder state")
        if untrained_receipt.encoder_state_sha256 == learned_receipt.encoder_state_sha256:
            raise ValueError("E0 state must differ from the learned L/Epi state")
        object.__setattr__(self, "video_id", reference.video_id)
        object.__setattr__(self, "learned", reference.features)
        object.__setattr__(self, "untrained", _immutable_array(untrained, np.float32))
        object.__setattr__(self, "epi", _immutable_array(epi, np.float32))
        object.__setattr__(self, "raw_xy", reference.raw_xy)
        object.__setattr__(self, "joint_mask", reference.joint_mask)
        object.__setattr__(self, "valid_mask", reference.valid_mask)
        object.__setattr__(self, "segments", reference.segments)


@dataclass(frozen=True, slots=True)
class MechanismEstimate:
    """Four exact-window estimates; L is the sole primary prediction."""

    candidate_id: str
    primary: ViewEstimate
    raw_control: ViewEstimate
    untrained_control: ViewEstimate
    temporal_derangement_control: ViewEstimate
    epi_permutation_map_sha256: str
    learned_encoding_receipt_sha256: str
    untrained_encoding_receipt_sha256: str
    epi_encoding_receipt_sha256: str

    def __post_init__(self) -> None:
        views = (
            self.primary,
            self.raw_control,
            self.untrained_control,
            self.temporal_derangement_control,
        )
        if tuple(item.view for item in views) != ("L", "R", "E0", "Epi"):
            raise ValueError("mechanism estimate views must be L/R/E0/Epi in fixed roles")
        if any(item.candidate_id != self.candidate_id for item in views):
            raise ValueError("mechanism estimate candidates differ")
        if len({item.video_id for item in views}) != 1:
            raise ValueError("mechanism estimate views differ in video_id")
        if len({item.window_keys for item in views}) != 1:
            raise ValueError("mechanism estimate views differ in window authority")
        for name, value in (
            ("epi_permutation_map_sha256", self.epi_permutation_map_sha256),
            ("learned_encoding_receipt_sha256", self.learned_encoding_receipt_sha256),
            ("untrained_encoding_receipt_sha256", self.untrained_encoding_receipt_sha256),
            ("epi_encoding_receipt_sha256", self.epi_encoding_receipt_sha256),
        ):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError(f"{name} must be a lowercase SHA-256")


def estimate_mechanism_views(
    video: MechanismVideo,
    candidate: SpectralCandidate,
) -> MechanismEstimate:
    """Evaluate L/R/E0/Epi without allowing a control to alter L."""

    learned = RepresentationVideo(
        video.video_id,
        video.learned,
        video.raw_xy,
        video.joint_mask,
        video.valid_mask,
        video.segments,
    )
    plan = build_window_plan(learned, candidate)
    primary = _estimate_from_plan(learned, candidate, plan, view="L", raw_control=False)
    raw = _estimate_from_plan(learned, candidate, plan, view="R", raw_control=True)
    untrained_video = RepresentationVideo(
        video.video_id,
        video.untrained,
        video.raw_xy,
        video.joint_mask,
        video.valid_mask,
        video.segments,
    )
    epi_video = RepresentationVideo(
        video.video_id,
        video.epi,
        video.raw_xy,
        video.joint_mask,
        video.valid_mask,
        video.segments,
    )
    untrained = _estimate_from_plan(
        untrained_video,
        candidate,
        plan,
        view="E0",
        raw_control=False,
    )
    epi = _estimate_from_plan(epi_video, candidate, plan, view="Epi", raw_control=False)
    signatures = {item.window_keys for item in (primary, raw, untrained, epi)}
    if len(signatures) != 1:
        raise RuntimeError("L/R/E0/Epi did not preserve the exact shared window authority")
    return MechanismEstimate(
        candidate_id=candidate.canonical_id,
        primary=primary,
        raw_control=raw,
        untrained_control=untrained,
        temporal_derangement_control=epi,
        epi_permutation_map_sha256=video.epi_receipt.permutation_map_sha256,
        learned_encoding_receipt_sha256=video.learned_encoding_receipt.receipt_sha256,
        untrained_encoding_receipt_sha256=video.untrained_encoding_receipt.receipt_sha256,
        epi_encoding_receipt_sha256=video.epi_encoding_receipt.receipt_sha256,
    )


SyntheticKind = Literal["count", "variable_tempo", "corruption", "null", "reset"]


@dataclass(frozen=True, slots=True)
class SyntheticCaseSpec:
    """Lightweight recipe; arrays are materialized one case at a time."""

    case_id: str
    family: str
    kind: SyntheticKind
    frames: int
    dimension: Literal[34, 512]
    target_count: float | None
    replicate: int = 0
    profile: str = "constant"


@dataclass(frozen=True, slots=True)
class SyntheticCase:
    """Materialized generated case and generation truth, never dataset truth."""

    spec: SyntheticCaseSpec
    video: RepresentationVideo


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


def synthetic_case_plan() -> tuple[SyntheticCaseSpec, ...]:
    """Return the exact selector/held-out family plan without allocating arrays."""

    specs: list[SyntheticCaseSpec] = []
    for dimension in (34, 512):
        for count in (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40):
            specs.append(
                SyntheticCaseSpec(
                    case_id=f"count.t256.c{count}.d{dimension}",
                    family="constant_tempo_count",
                    kind="count",
                    frames=256,
                    dimension=dimension,
                    target_count=float(count),
                )
            )
        for frames in (192, 384):
            for count in (4, 12, 24):
                specs.append(
                    SyntheticCaseSpec(
                        case_id=f"count.t{frames}.c{count}.d{dimension}",
                        family="duration_count",
                        kind="count",
                        frames=frames,
                        dimension=dimension,
                        target_count=float(count),
                    )
                )
        for count in (4, 8, 16, 24):
            for profile in ("ramp_up", "up_then_down", "sinusoidal_tempo"):
                specs.append(
                    SyntheticCaseSpec(
                        case_id=f"tempo.{profile}.c{count}.d{dimension}",
                        family="variable_tempo",
                        kind="variable_tempo",
                        frames=256,
                        dimension=dimension,
                        target_count=float(count),
                        profile=profile,
                    )
                )
        for count in (8, 16):
            for support in ("idle_60_percent", "idle_80_percent"):
                specs.append(
                    SyntheticCaseSpec(
                        case_id=f"idle.{support}.c{count}.d{dimension}",
                        family="idle_support",
                        kind="count",
                        frames=256,
                        dimension=dimension,
                        target_count=float(count),
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
                    SyntheticCaseSpec(
                        case_id=f"harmonic.{harmonic}.c{count}.d{dimension}",
                        family="harmonic_stress",
                        kind="corruption",
                        frames=256,
                        dimension=dimension,
                        target_count=float(count),
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
                    SyntheticCaseSpec(
                        case_id=f"corrupt.{corruption}.c{count}.d{dimension}",
                        family="corruption",
                        kind="corruption",
                        frames=256,
                        dimension=dimension,
                        target_count=float(count),
                        profile=corruption,
                    )
                )
    for family in _NULL_FAMILIES:
        for replicate in range(64):
            dimension: Literal[34, 512] = 34 if replicate % 2 == 0 else 512
            specs.append(
                SyntheticCaseSpec(
                    case_id=f"null.{family}.r{replicate:02d}.d{dimension}",
                    family=family,
                    kind="null",
                    frames=256,
                    dimension=dimension,
                    target_count=None,
                    replicate=replicate,
                    profile=family,
                )
            )
    for replicate in range(64):
        dimension = 34 if replicate % 2 == 0 else 512
        specs.append(
            SyntheticCaseSpec(
                case_id=f"reset.two_segment.r{replicate:02d}.d{dimension}",
                family="two_segment_reset",
                kind="reset",
                frames=256,
                dimension=dimension,
                target_count=None,
                replicate=replicate,
                profile="two_segment_reset",
            )
        )
    identifiers = tuple(item.case_id for item in specs)
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("synthetic case identifiers are not unique")
    return tuple(specs)


def _case_seed(seed: int, case_id: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{case_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


@lru_cache(maxsize=8)
def _fixed_mixing(dimension: int, seed: int) -> NDArray[np.float64]:
    """Return a seed/dimension-fixed normalized orthogonal-column mixing."""

    if dimension not in (34, 512):
        raise ValueError("synthetic mixing dimension must be 34 or 512")
    rng = np.random.default_rng(_case_seed(seed, f"mixing.d{dimension}"))
    matrix = rng.normal(size=(dimension, 8))
    orthogonal, _ = np.linalg.qr(matrix, mode="reduced")
    result = np.asarray(orthogonal, dtype=np.float64)
    result.setflags(write=False)
    return result


def synthetic_mixing_sha256(seed: int, dimension: Literal[34, 512]) -> str:
    matrix = np.ascontiguousarray(_fixed_mixing(dimension, seed), dtype="<f8")
    return hashlib.sha256(matrix.tobytes(order="C")).hexdigest()


def _tempo_profile(frames: int, profile: str) -> NDArray[np.float64]:
    if frames < 2:
        raise ValueError("synthetic phase requires at least two source frames")
    intervals = frames - 1
    time = np.linspace(0.0, 1.0, intervals, dtype=np.float64)
    if profile == "ramp_up":
        speed = np.linspace(0.5, 1.5, intervals, dtype=np.float64)
    elif profile == "up_then_down":
        speed = np.interp(time, (0.0, 0.5, 1.0), (0.5, 1.5, 0.5))
    elif profile == "sinusoidal_tempo":
        speed = 1.0 + 0.4 * np.sin(2.0 * np.pi * time)
    elif profile in ("idle_60_percent", "idle_80_percent"):
        support = 0.6 if profile == "idle_60_percent" else 0.8
        edge = (1.0 - support) / 2.0
        speed = ((time >= edge) & (time <= 1.0 - edge)).astype(np.float64)
    else:
        speed = np.ones(intervals, dtype=np.float64)
    total = float(np.sum(speed))
    if total <= _EPSILON:
        raise RuntimeError("synthetic tempo profile has zero support")
    return speed / total


def _periodic_latent(
    frames: int,
    count: float,
    profile: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    increments = _tempo_profile(frames, profile)
    phase = np.zeros(frames, dtype=np.float64)
    phase[1:] = 2.0 * np.pi * count * np.cumsum(increments)
    amplitude = np.ones(frames, dtype=np.float64)
    if profile == "amplitude_ramp_0.5_to_1.5":
        amplitude = np.linspace(0.5, 1.5, frames, dtype=np.float64)
    base_sin = np.sin(phase)
    base_cos = np.cos(phase)
    second_sin = np.sin(2.0 * phase)
    second_cos = np.cos(2.0 * phase)
    subharmonic = np.sin(0.5 * phase)
    if profile == "second_harmonic_amp_0.75":
        base_sin += 0.75 * second_sin
        base_cos += 0.75 * second_cos
    elif profile == "second_harmonic_amp_1.25":
        base_sin += 1.25 * second_sin
        base_cos += 1.25 * second_cos
    elif profile == "subharmonic_amp_0.5":
        base_sin += 0.5 * subharmonic
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
    frames = int(phase.size)
    joints = np.arange(17, dtype=np.float64)
    base_x = (joints % 5.0 - 2.0) / 5.0
    base_y = (joints // 5.0 - 1.5) / 5.0
    raw = np.empty((frames, 17, 2), dtype=np.float64)
    joint_phase = joints * (np.pi / 17.0)
    raw[:, :, 0] = base_x[None, :] + 0.12 * np.sin(phase[:, None] + joint_phase[None, :])
    raw[:, :, 1] = base_y[None, :] + 0.10 * np.cos(phase[:, None] - joint_phase[None, :])
    return raw


def materialize_synthetic_case(spec: SyntheticCaseSpec, *, seed: int) -> SyntheticCase:
    """Materialize one deterministic case for selector or held-out replay."""

    rng = np.random.default_rng(_case_seed(seed, spec.case_id))
    frames = spec.frames
    mask = np.ones((frames, 17), dtype=np.bool_)
    valid = np.ones(frames, dtype=np.bool_)
    segments: tuple[AuthorizedSegment, ...] = (AuthorizedSegment("segment-0", 0, frames),)

    if spec.kind in ("count", "variable_tempo", "corruption"):
        if spec.target_count is None:
            raise RuntimeError("positive synthetic case requires target_count")
        latent, phase = _periodic_latent(frames, spec.target_count, spec.profile)
        features = latent @ _fixed_mixing(spec.dimension, seed).T
        raw = _raw_pose_from_phase(phase)
        if spec.profile == "affine_drift":
            drift = np.linspace(-0.25, 0.25, frames, dtype=np.float64)
            features += drift[:, None] * np.linspace(0.5, 1.0, spec.dimension)[None, :]
            raw += drift[:, None, None] * np.asarray((1.0, -0.5))[None, None, :]
        elif spec.profile == "noise_sigma_0.02":
            features += rng.normal(0.0, 0.02, size=features.shape)
            raw += rng.normal(0.0, 0.02, size=raw.shape)
        elif spec.profile == "feature_dropout_20_percent":
            dropped = rng.choice(spec.dimension, size=max(1, spec.dimension // 5), replace=False)
            features[:, dropped] = 0.0
        elif spec.profile == "deterministic_mask_flicker":
            flicker_joints = np.asarray((0, 1, 2, 3), dtype=np.int64)
            flicker_frames = np.flatnonzero(np.arange(frames) % 4 < 2)
            mask[np.ix_(flicker_frames, flicker_joints)] = False
        elif spec.profile == "contiguous_joint_feature_occlusion":
            start = frames * 2 // 5
            stop = frames * 3 // 5
            occluded = np.asarray((0, 1, 2, 3, 5, 6), dtype=np.int64)
            mask[start:stop, occluded] = False
            features[start:stop, : max(1, spec.dimension // 5)] = 0.0
        elif spec.profile == "amplitude_ramp_0.5_to_1.5":
            scale = np.linspace(0.5, 1.5, frames, dtype=np.float64)
            raw *= scale[:, None, None]
    elif spec.kind == "null":
        time = np.linspace(-1.0, 1.0, frames, dtype=np.float64)
        raw = np.zeros((frames, 17, 2), dtype=np.float64)
        if spec.profile == "constant":
            features = np.zeros((frames, spec.dimension), dtype=np.float64)
        elif spec.profile == "drift_only":
            direction = rng.normal(size=spec.dimension)
            direction /= max(float(np.linalg.norm(direction)), _EPSILON)
            features = time[:, None] * direction[None, :]
        elif spec.profile == "white_noise":
            features = rng.normal(0.0, 1.0, size=(frames, spec.dimension))
        elif spec.profile == "random_walk":
            features = np.cumsum(
                rng.normal(0.0, 0.1, size=(frames, spec.dimension)),
                axis=0,
            )
        elif spec.profile == "ar1_rho_0.9":
            innovations = rng.normal(0.0, math.sqrt(1.0 - 0.9**2), size=(frames, spec.dimension))
            features = np.zeros_like(innovations)
            for index in range(1, frames):
                features[index] = 0.9 * features[index - 1] + innovations[index]
        elif spec.profile == "time_shuffled_periodic":
            latent, phase = _periodic_latent(frames, 8.0, "constant")
            permutation = rng.permutation(frames)
            features = (latent @ _fixed_mixing(spec.dimension, seed).T)[permutation]
            raw = _raw_pose_from_phase(phase)[permutation]
        elif spec.profile == "independent_incoherent_frequency_phase":
            frequencies = rng.uniform(1.0 / 128.0, 1.0 / 4.0, size=spec.dimension)
            phases = rng.uniform(-np.pi, np.pi, size=spec.dimension)
            grid = np.arange(frames, dtype=np.float64)
            features = np.sin(2.0 * np.pi * grid[:, None] * frequencies + phases[None, :])
        elif spec.profile == "constant_coordinates_oscillating_masks":
            features = np.zeros((frames, spec.dimension), dtype=np.float64)
            for frame in range(frames):
                mask[frame, (frame // 4) % 8] = False
                mask[frame, 8 + (frame // 4) % 8] = False
        else:
            raise ValueError(f"unknown null family: {spec.profile}")
    elif spec.kind == "reset":
        latent, phase = _periodic_latent(frames, 12.0, "constant")
        features = latent @ _fixed_mixing(spec.dimension, seed).T
        raw = _raw_pose_from_phase(phase)
        valid[112:144] = False
        mask[112:144] = False
        features[112:144] = 0.0
        segments = (
            AuthorizedSegment("segment-0", 0, 112),
            AuthorizedSegment("segment-1", 144, 256),
        )
    else:
        raise ValueError(f"unknown synthetic kind: {spec.kind}")

    raw[~mask] = 0.0
    return SyntheticCase(
        spec=spec,
        video=RepresentationVideo(
            video_id=spec.case_id,
            features=np.asarray(features, dtype=np.float32),
            raw_xy=np.asarray(raw, dtype=np.float32),
            joint_mask=mask,
            valid_mask=valid,
            segments=segments,
        ),
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


def _invariance_variants(
    case: SyntheticCase,
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
        raise ValueError("Clopper-Pearson counts must satisfy 0 <= failures <= trials")
    if not 0.0 < confidence < 1.0:
        raise ValueError("Clopper-Pearson confidence must be in (0, 1)")
    if failures == trials:
        return 1.0
    return float(beta_distribution.ppf(confidence, failures + 1, trials - failures))


@dataclass(frozen=True, slots=True)
class NullFamilyScore:
    """Family-wise false-eligibility control; pooling is forbidden."""

    family: str
    trials: int
    false_eligible: int
    one_sided_cp_ucb: float
    hard_pass: bool


@dataclass(frozen=True, slots=True)
class PositiveFamilyScore:
    """Generation-truth metrics retained for every positive stress family."""

    family: str
    trials: int
    eligible: int
    eligible_rate: float
    nmae: float
    obo: float


@dataclass(frozen=True, slots=True)
class SyntheticCandidateScore:
    """Full hard-gate and ranking record for one candidate on one seed."""

    candidate: SpectralCandidate
    seed: int
    positive_total: int
    positive_eligible: int
    positive_eligible_rate: float
    overall_nmae: float
    overall_obo: float
    variable_tempo_nmae: float
    positive_family_scores: tuple[PositiveFamilyScore, ...]
    invariance_agreement: tuple[tuple[str, float], ...]
    null_total: int
    null_false_eligible: int
    null_cp_ucb: float
    null_family_scores: tuple[NullFamilyScore, ...]
    reset_total: int
    reset_video_total_errors: int
    abstention_rate: float
    hard_pass: bool
    failed_gates: tuple[str, ...]

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
            "positive_gate_policy": (
                "pooled_all_generation_truth_positives_only_no_family_thresholds"
            ),
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
            "null_false_eligible": self.null_false_eligible,
            "null_false_eligible_one_sided_95pct_cp_ucb": self.null_cp_ucb,
            "null_family_scores": {
                item.family: {
                    "trials": item.trials,
                    "false_eligible": item.false_eligible,
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


def _validated_synthetic_plan(
    config: SegmentLocalSpectralConfig,
) -> tuple[SyntheticCaseSpec, ...]:
    specs = synthetic_case_plan()
    null_specs = tuple(item for item in specs if item.kind == "null")
    reset_specs = tuple(item for item in specs if item.kind == "reset")
    observed_families = {item.family for item in null_specs}
    if observed_families != set(_NULL_FAMILIES):
        raise RuntimeError("synthetic null family membership changed")
    for family in _NULL_FAMILIES:
        members = tuple(item for item in null_specs if item.family == family)
        if len(members) != config.synthetic.null_trials_per_family:
            raise RuntimeError(f"synthetic null family {family} must contain exactly 64 cases")
        if {item.replicate for item in members} != set(
            range(config.synthetic.null_trials_per_family)
        ):
            raise RuntimeError(f"synthetic null family {family} replicate identities changed")
    if len(reset_specs) != config.synthetic.reset_trials:
        raise RuntimeError("synthetic reset family must contain exactly 64 cases")
    if {item.replicate for item in reset_specs} != set(range(config.synthetic.reset_trials)):
        raise RuntimeError("synthetic reset replicate identities changed")
    return specs


def score_synthetic_candidate(
    config: SegmentLocalSpectralConfig,
    candidate: SpectralCandidate,
    *,
    seed: int,
) -> SyntheticCandidateScore:
    """Score one candidate using generated truth and no dataset interface."""

    allowed_seeds = (config.synthetic.selector_seed, config.synthetic.heldout_seed)
    if seed not in allowed_seeds:
        raise ValueError(f"synthetic score seed must be one of {allowed_seeds!r}")
    if candidate not in config.candidates:
        raise ValueError("candidate is outside the frozen 32-member grid")

    absolute_errors: list[float] = []
    normalized_errors: list[float] = []
    variable_normalized_errors: list[float] = []
    family_absolute_errors: dict[str, list[float]] = {}
    family_normalized_errors: dict[str, list[float]] = {}
    family_eligible: dict[str, int] = {}
    eligible = 0
    obo = 0
    agreement_successes = {name: 0 for name in _INVARIANCE_NAMES}
    agreement_trials = {name: 0 for name in _INVARIANCE_NAMES}
    null_trials = {family: 0 for family in _NULL_FAMILIES}
    null_failures = {family: 0 for family in _NULL_FAMILIES}
    reset_total = 0
    reset_errors = 0

    for spec in _validated_synthetic_plan(config):
        case = materialize_synthetic_case(spec, seed=seed)
        estimate = estimate_representation(case.video, candidate)
        if spec.kind in ("count", "variable_tempo", "corruption"):
            if spec.target_count is None:
                raise RuntimeError("positive case lost generation truth")
            prediction = estimate.rounded_count if estimate.rounded_count is not None else 0
            error = abs(float(prediction) - spec.target_count)
            absolute_errors.append(error)
            normalized_errors.append(error / spec.target_count)
            family_absolute_errors.setdefault(spec.family, []).append(error)
            family_normalized_errors.setdefault(spec.family, []).append(error / spec.target_count)
            family_eligible.setdefault(spec.family, 0)
            if spec.kind == "variable_tempo":
                variable_normalized_errors.append(error / spec.target_count)
            if estimate.status == "eligible":
                eligible += 1
                family_eligible[spec.family] += 1
            if error <= 1.0:
                obo += 1
            variants = _invariance_variants(case, seed=seed)
            for name, variant in variants.items():
                transformed = estimate_representation(variant, candidate)
                agreement_trials[name] += 1
                if (
                    estimate.status == "eligible"
                    and transformed.status == "eligible"
                    and estimate.rounded_count == transformed.rounded_count
                ):
                    agreement_successes[name] += 1
        elif spec.kind == "null":
            null_trials[spec.family] += 1
            if estimate.status == "eligible":
                null_failures[spec.family] += 1
        elif spec.kind == "reset":
            reset_total += 1
            if (
                estimate.status != "undefined_multi_segment"
                or estimate.float_count is not None
                or estimate.rounded_count is not None
            ):
                reset_errors += 1

    positive_total = len(absolute_errors)
    if positive_total < 1 or not variable_normalized_errors:
        raise RuntimeError("synthetic plan did not produce required positive families")
    agreements = tuple(
        (name, agreement_successes[name] / agreement_trials[name])
        for name in _INVARIANCE_NAMES
    )
    positive_eligible_rate = eligible / positive_total
    positive_family_scores = tuple(
        PositiveFamilyScore(
            family=family,
            trials=len(normalized_errors_for_family),
            eligible=family_eligible[family],
            eligible_rate=family_eligible[family] / len(normalized_errors_for_family),
            nmae=float(np.mean(normalized_errors_for_family)),
            obo=float(np.mean(np.asarray(family_absolute_errors[family]) <= 1.0)),
        )
        for family, normalized_errors_for_family in sorted(family_normalized_errors.items())
    )
    overall_nmae = float(np.mean(normalized_errors))
    overall_obo = obo / positive_total
    variable_nmae = float(np.mean(variable_normalized_errors))
    null_family_scores = tuple(
        NullFamilyScore(
            family=family,
            trials=null_trials[family],
            false_eligible=null_failures[family],
            one_sided_cp_ucb=_one_sided_cp_upper(
                null_failures[family],
                null_trials[family],
                config.synthetic.null_false_eligible_cp_confidence,
            ),
            hard_pass=_one_sided_cp_upper(
                null_failures[family],
                null_trials[family],
                config.synthetic.null_false_eligible_cp_confidence,
            )
            <= config.synthetic.null_false_eligible_cp_ucb_maximum,
        )
        for family in _NULL_FAMILIES
    )
    null_total = sum(null_trials.values())
    null_false_eligible = sum(null_failures.values())
    null_ucb = max(item.one_sided_cp_ucb for item in null_family_scores)
    abstention = 1.0 - positive_eligible_rate
    failed: list[str] = []
    if positive_eligible_rate < config.synthetic.positive_eligible_minimum:
        failed.append("positive_eligible_rate")
    if overall_nmae > config.synthetic.overall_nmae_maximum:
        failed.append("overall_nmae")
    if overall_obo < config.synthetic.overall_obo_minimum:
        failed.append("overall_obo")
    if variable_nmae > config.synthetic.variable_tempo_nmae_maximum:
        failed.append("variable_tempo_nmae")
    if any(
        value < config.synthetic.invariance_rounded_agreement_minimum
        for _, value in agreements
    ):
        failed.append("invariance_rounded_agreement")
    failed_null_families = tuple(item.family for item in null_family_scores if not item.hard_pass)
    if failed_null_families:
        failed.extend(f"null_false_eligible_cp_ucb:{family}" for family in failed_null_families)
    if reset_errors > config.synthetic.reset_video_total_errors_maximum:
        failed.append("reset_video_total_errors")
    return SyntheticCandidateScore(
        candidate=candidate,
        seed=seed,
        positive_total=positive_total,
        positive_eligible=eligible,
        positive_eligible_rate=positive_eligible_rate,
        overall_nmae=overall_nmae,
        overall_obo=overall_obo,
        variable_tempo_nmae=variable_nmae,
        positive_family_scores=positive_family_scores,
        invariance_agreement=agreements,
        null_total=null_total,
        null_false_eligible=null_false_eligible,
        null_cp_ucb=null_ucb,
        null_family_scores=null_family_scores,
        reset_total=reset_total,
        reset_video_total_errors=reset_errors,
        abstention_rate=abstention,
        hard_pass=not failed,
        failed_gates=tuple(failed),
    )


_INVARIANCE_NAMES: tuple[str, ...] = (
    "reverse",
    "sign_flip",
    "feature_permutation",
    "orthogonal_mixing",
    "scale",
)


class SyntheticSelectionFailure(RuntimeError):
    """Raised when calibration or held-out replay cannot authorize a method."""


@dataclass(frozen=True, slots=True)
class SyntheticSelectionReport:
    """Calibration report binding the sole selected candidate."""

    config_fingerprint: str
    selector_seed: int
    mixing_sha256: tuple[tuple[int, str], ...]
    selected_candidate_id: str
    selected_rank: tuple[float, float, float, float, float, str]
    scores: tuple[SyntheticCandidateScore, ...]

    @property
    def canonical_sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "classification": "synthetic_generation_truth_only_calibration",
            "config_fingerprint": self.config_fingerprint,
            "selector_seed": self.selector_seed,
            "mixing_sha256": {str(key): value for key, value in self.mixing_sha256},
            "selected_candidate_id": self.selected_candidate_id,
            "selected_rank": list(self.selected_rank),
            "scores": [score.to_dict() for score in self.scores],
            "dataset_targets_consumed": False,
        }


def _selection_report_from_scores(
    config: SegmentLocalSpectralConfig,
    scores: Sequence[SyntheticCandidateScore],
) -> SyntheticSelectionReport:
    frozen_scores = tuple(scores)
    expected_ids = tuple(candidate.canonical_id for candidate in config.candidates)
    actual_ids = tuple(score.candidate.canonical_id for score in frozen_scores)
    if actual_ids != expected_ids or len(set(actual_ids)) != 32:
        raise SyntheticSelectionFailure("selection must score the complete ordered 32-candidate grid")
    if any(score.seed != config.synthetic.selector_seed for score in frozen_scores):
        raise SyntheticSelectionFailure("selection scores must use only the frozen selector seed")
    passing = tuple(score for score in frozen_scores if score.hard_pass)
    if not passing:
        raise SyntheticSelectionFailure("no candidate passed every synthetic calibration gate")
    ordered = tuple(sorted(passing, key=lambda item: item.rank))
    selected = ordered[0]
    if len(ordered) > 1 and ordered[0].rank == ordered[1].rank:
        raise SyntheticSelectionFailure("synthetic calibration did not select a unique candidate")
    return SyntheticSelectionReport(
        config_fingerprint=config.fingerprint,
        selector_seed=config.synthetic.selector_seed,
        mixing_sha256=tuple(
            (dimension, synthetic_mixing_sha256(config.synthetic.selector_seed, dimension))
            for dimension in config.synthetic.dimensions
        ),
        selected_candidate_id=selected.candidate.canonical_id,
        selected_rank=selected.rank,
        scores=frozen_scores,
    )


def _recompute_synthetic_selection(
    config: SegmentLocalSpectralConfig,
) -> SyntheticSelectionReport:
    scores = tuple(
        score_synthetic_candidate(config, candidate, seed=config.synthetic.selector_seed)
        for candidate in config.candidates
    )
    return _selection_report_from_scores(config, scores)


def select_synthetic_candidate(
    config: SegmentLocalSpectralConfig,
) -> SyntheticSelectionReport:
    """Select exactly once on seed 2026 after filtering by all hard gates."""

    return _recompute_synthetic_selection(config)


def replay_and_validate_synthetic_selection(
    config: SegmentLocalSpectralConfig,
    report: SyntheticSelectionReport,
) -> SpectralCandidate:
    """Recompute all 32 scores and reject any altered selection artifact."""

    expected = _recompute_synthetic_selection(config)
    if report.to_dict() != expected.to_dict():
        raise SyntheticSelectionFailure(
            "synthetic selection artifact differs from the complete deterministic replay"
        )
    if report.canonical_sha256 != expected.canonical_sha256:
        raise SyntheticSelectionFailure("synthetic selection canonical SHA-256 mismatch")
    return next(
        candidate
        for candidate in config.candidates
        if candidate.canonical_id == report.selected_candidate_id
    )


@dataclass(frozen=True, slots=True)
class SyntheticHeldoutReport:
    """Seed-3407 replay of the calibration winner, with no reranking."""

    config_fingerprint: str
    selection_report_sha256: str
    selected_candidate_id: str
    heldout_seed: int
    mixing_sha256: tuple[tuple[int, str], ...]
    score: SyntheticCandidateScore
    authorized_for_train337: bool

    @property
    def canonical_sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "classification": "synthetic_selected_candidate_heldout_replay",
            "config_fingerprint": self.config_fingerprint,
            "selection_report_sha256": self.selection_report_sha256,
            "selected_candidate_id": self.selected_candidate_id,
            "heldout_seed": self.heldout_seed,
            "mixing_sha256": {str(key): value for key, value in self.mixing_sha256},
            "score": self.score.to_dict(),
            "authorized_for_train337": self.authorized_for_train337,
            "reranked": False,
            "fallback_used": False,
            "dataset_targets_consumed": False,
        }


def replay_selected_candidate_heldout(
    config: SegmentLocalSpectralConfig,
    selection: SyntheticSelectionReport,
) -> SyntheticHeldoutReport:
    """Replay only the selected tuple on seed 3407; failure has no fallback."""

    candidate = replay_and_validate_synthetic_selection(config, selection)
    score = score_synthetic_candidate(
        config,
        candidate,
        seed=config.synthetic.heldout_seed,
    )
    report = SyntheticHeldoutReport(
        config_fingerprint=config.fingerprint,
        selection_report_sha256=selection.canonical_sha256,
        selected_candidate_id=candidate.canonical_id,
        heldout_seed=config.synthetic.heldout_seed,
        mixing_sha256=tuple(
            (dimension, synthetic_mixing_sha256(config.synthetic.heldout_seed, dimension))
            for dimension in config.synthetic.dimensions
        ),
        score=score,
        authorized_for_train337=score.hard_pass,
    )
    if not score.hard_pass:
        raise SyntheticSelectionFailure(
            "selected candidate failed seed-3407 held-out replay; no rerank or fallback is allowed"
        )
    return report


def replay_and_validate_synthetic_heldout(
    config: SegmentLocalSpectralConfig,
    selection: SyntheticSelectionReport,
    heldout: SyntheticHeldoutReport,
) -> SpectralCandidate:
    """Recompute the bound winner on seed 3407 without any second selection."""

    candidate = replay_and_validate_synthetic_selection(config, selection)
    score = score_synthetic_candidate(
        config,
        candidate,
        seed=config.synthetic.heldout_seed,
    )
    expected = SyntheticHeldoutReport(
        config_fingerprint=config.fingerprint,
        selection_report_sha256=selection.canonical_sha256,
        selected_candidate_id=candidate.canonical_id,
        heldout_seed=config.synthetic.heldout_seed,
        mixing_sha256=tuple(
            (dimension, synthetic_mixing_sha256(config.synthetic.heldout_seed, dimension))
            for dimension in config.synthetic.dimensions
        ),
        score=score,
        authorized_for_train337=score.hard_pass,
    )
    if heldout.to_dict() != expected.to_dict():
        raise SyntheticSelectionFailure(
            "held-out artifact differs from the selected-candidate-only deterministic replay"
        )
    if heldout.canonical_sha256 != expected.canonical_sha256:
        raise SyntheticSelectionFailure("synthetic held-out canonical SHA-256 mismatch")
    if not heldout.authorized_for_train337 or not heldout.score.hard_pass:
        raise SyntheticSelectionFailure("seed-3407 held-out replay did not pass all frozen gates")
    return candidate


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    digest = value
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return digest


def canonical_video_ids_sha256(video_ids: Sequence[str]) -> str:
    """Hash an exact, ordered canonical video-ID registry."""

    identifiers = tuple(str(item).strip() for item in video_ids)
    if any(not item for item in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("canonical video IDs must be non-empty and unique")
    encoded = json.dumps(
        {"schema_version": 1, "ordered_video_ids": list(identifiers)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class Train337Authority:
    """Fail-closed placeholder contract for future authoritative adapters."""

    adapter_status: Literal["unwired_fail_closed"]
    canonical_video_ids: tuple[str, ...]
    canonical_video_ids_sha256: str
    v4e_representation_authority_sha256: str
    cycleback_checkpoint_authority_sha256: str
    transform_recipe_sha256: str
    selection_report_sha256: str
    heldout_report_sha256: str
    authority_receipt_sha256: str

    @property
    def source_authority_sha256(self) -> str:
        encoded = json.dumps(
            {
                "v4e_representation_authority_sha256": (
                    self.v4e_representation_authority_sha256
                ),
                "cycleback_checkpoint_authority_sha256": (
                    self.cycleback_checkpoint_authority_sha256
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def __post_init__(self) -> None:
        if self.adapter_status != "unwired_fail_closed":
            raise ValueError("authoritative train337 adapter is not implemented")
        if len(self.canonical_video_ids) != 337:
            raise ValueError("train337 authority requires exactly 337 canonical video IDs")
        if tuple(str(item).strip() for item in self.canonical_video_ids) != self.canonical_video_ids:
            raise ValueError("canonical train337 video IDs must not require normalization")
        expected_ids_digest = canonical_video_ids_sha256(self.canonical_video_ids)
        if self.canonical_video_ids_sha256 != expected_ids_digest:
            raise ValueError("canonical train337 video-ID digest mismatch")
        for name, value in (
            ("v4e_representation_authority_sha256", self.v4e_representation_authority_sha256),
            ("cycleback_checkpoint_authority_sha256", self.cycleback_checkpoint_authority_sha256),
            ("transform_recipe_sha256", self.transform_recipe_sha256),
            ("selection_report_sha256", self.selection_report_sha256),
            ("heldout_report_sha256", self.heldout_report_sha256),
            ("authority_receipt_sha256", self.authority_receipt_sha256),
        ):
            _require_sha256(value, name)
        expected_receipt = _train337_authority_digest(
            canonical_video_ids=self.canonical_video_ids,
            canonical_video_ids_sha256=self.canonical_video_ids_sha256,
            v4e_representation_authority_sha256=self.v4e_representation_authority_sha256,
            cycleback_checkpoint_authority_sha256=self.cycleback_checkpoint_authority_sha256,
            transform_recipe_sha256=self.transform_recipe_sha256,
            selection_report_sha256=self.selection_report_sha256,
            heldout_report_sha256=self.heldout_report_sha256,
        )
        if self.authority_receipt_sha256 != expected_receipt:
            raise ValueError("train337 authority receipt SHA-256 mismatch")


def _train337_authority_digest(
    *,
    canonical_video_ids: Sequence[str],
    canonical_video_ids_sha256: str,
    v4e_representation_authority_sha256: str,
    cycleback_checkpoint_authority_sha256: str,
    transform_recipe_sha256: str,
    selection_report_sha256: str,
    heldout_report_sha256: str,
) -> str:
    payload = {
        "schema_version": 1,
        "adapter_status": "unwired_fail_closed",
        "canonical_video_ids": list(canonical_video_ids),
        "canonical_video_ids_sha256": canonical_video_ids_sha256,
        "v4e_representation_authority_sha256": v4e_representation_authority_sha256,
        "cycleback_checkpoint_authority_sha256": cycleback_checkpoint_authority_sha256,
        "transform_recipe_sha256": transform_recipe_sha256,
        "selection_report_sha256": selection_report_sha256,
        "heldout_report_sha256": heldout_report_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def build_fail_closed_train337_authority(
    config: SegmentLocalSpectralConfig,
    *,
    canonical_video_ids: Sequence[str],
    v4e_representation_authority_sha256: str,
    cycleback_checkpoint_authority_sha256: str,
    selection: SyntheticSelectionReport,
    heldout: SyntheticHeldoutReport,
) -> Train337Authority:
    """Build the only currently allowed, explicitly non-authorizing authority."""

    identifiers = tuple(canonical_video_ids)
    ids_digest = canonical_video_ids_sha256(identifiers)
    receipt = _train337_authority_digest(
        canonical_video_ids=identifiers,
        canonical_video_ids_sha256=ids_digest,
        v4e_representation_authority_sha256=v4e_representation_authority_sha256,
        cycleback_checkpoint_authority_sha256=cycleback_checkpoint_authority_sha256,
        transform_recipe_sha256=config.train337.transform_recipe.fingerprint,
        selection_report_sha256=selection.canonical_sha256,
        heldout_report_sha256=heldout.canonical_sha256,
    )
    return Train337Authority(
        adapter_status="unwired_fail_closed",
        canonical_video_ids=identifiers,
        canonical_video_ids_sha256=ids_digest,
        v4e_representation_authority_sha256=v4e_representation_authority_sha256,
        cycleback_checkpoint_authority_sha256=cycleback_checkpoint_authority_sha256,
        transform_recipe_sha256=config.train337.transform_recipe.fingerprint,
        selection_report_sha256=selection.canonical_sha256,
        heldout_report_sha256=heldout.canonical_sha256,
        authority_receipt_sha256=receipt,
    )


def _window_estimate_payload(item: WindowEstimate) -> dict[str, Any]:
    return {
        "key": list(item.key),
        "center": item.center,
        "accepted": item.accepted,
        "abstention_reason": item.abstention_reason,
        "frequency": item.frequency,
        "peak_share": item.peak_share,
        "signed_vector_acf": item.signed_vector_acf,
        "confidence": item.confidence,
        "low_band_boundary": item.low_band_boundary,
        "high_band_boundary": item.high_band_boundary,
        "informative_dimensions": item.informative_dimensions,
    }


def _view_estimate_payload(item: ViewEstimate) -> dict[str, Any]:
    return {
        "video_id": item.video_id,
        "view": item.view,
        "candidate_id": item.candidate_id,
        "status": item.status,
        "abstention_reason": item.abstention_reason,
        "float_count": item.float_count,
        "rounded_count": item.rounded_count,
        "periodic_confidence": item.periodic_confidence,
        "mean_peak_share": item.mean_peak_share,
        "segments": [
            {
                "segment_id": segment.segment.segment_id,
                "start": segment.segment.start,
                "stop": segment.segment.stop,
                "candidate_window_count": segment.candidate_window_count,
                "available_window_count": segment.available_window_count,
                "accepted_window_count": segment.accepted_window_count,
                "single_window_estimate": segment.single_window_estimate,
                "status": segment.status,
                "abstention_reason": segment.abstention_reason,
                "float_count": segment.float_count,
                "windows": [_window_estimate_payload(window) for window in segment.windows],
            }
            for segment in item.segments
        ],
    }


def _mechanism_estimate_payload(item: MechanismEstimate) -> dict[str, Any]:
    return {
        "candidate_id": item.candidate_id,
        "primary": _view_estimate_payload(item.primary),
        "raw_control": _view_estimate_payload(item.raw_control),
        "untrained_control": _view_estimate_payload(item.untrained_control),
        "temporal_derangement_control": _view_estimate_payload(
            item.temporal_derangement_control
        ),
        "epi_permutation_map_sha256": item.epi_permutation_map_sha256,
        "learned_encoding_receipt_sha256": item.learned_encoding_receipt_sha256,
        "untrained_encoding_receipt_sha256": item.untrained_encoding_receipt_sha256,
        "epi_encoding_receipt_sha256": item.epi_encoding_receipt_sha256,
    }


_TRAIN337_ARTIFACT_NAMES: tuple[str, ...] = (
    "baseline",
    "reverse_primary",
    "warp_075_primary",
    "warp_125_primary",
    "duplicate_time_primary",
    "legal_split_part_float_counts",
    "raw_rotation_minus15",
    "raw_rotation_plus15",
    "raw_scale_085",
    "raw_scale_115",
    "raw_joint_dropout",
    "learned_augmentation_a",
    "learned_augmentation_b",
    "static_null_primary",
    "second_derangement_shuffle",
)


def _artifact_sha256(name: str, payload: Any) -> str:
    encoded = json.dumps(
        {"artifact_name": name, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class Train337TransformLineage:
    """Hashes every baseline/transform outcome under one frozen recipe."""

    video_id: str
    source_authority_sha256: str
    transform_recipe_sha256: str
    artifacts: tuple[tuple[str, str], ...]
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not self.video_id or self.video_id != self.video_id.strip():
            raise ValueError("transform lineage video_id must be canonical")
        _require_sha256(self.source_authority_sha256, "source_authority_sha256")
        _require_sha256(self.transform_recipe_sha256, "transform_recipe_sha256")
        if tuple(name for name, _ in self.artifacts) != _TRAIN337_ARTIFACT_NAMES:
            raise ValueError("transform lineage artifact set/order changed")
        for name, digest in self.artifacts:
            _require_sha256(digest, f"transform artifact {name}")
        expected = _transform_lineage_digest(
            video_id=self.video_id,
            source_authority_sha256=self.source_authority_sha256,
            transform_recipe_sha256=self.transform_recipe_sha256,
            artifacts=self.artifacts,
        )
        if self.receipt_sha256 != expected:
            raise ValueError("transform lineage receipt SHA-256 mismatch")


def _transform_lineage_digest(
    *,
    video_id: str,
    source_authority_sha256: str,
    transform_recipe_sha256: str,
    artifacts: Sequence[tuple[str, str]],
) -> str:
    payload = {
        "schema_version": 1,
        "video_id": video_id,
        "source_authority_sha256": source_authority_sha256,
        "transform_recipe_sha256": transform_recipe_sha256,
        "artifacts": [{"name": name, "sha256": digest} for name, digest in artifacts],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class Train337VideoAudit:
    """Label-free estimates needed to recompute one train337 audit row.

    All transformed estimates must be produced from the same source video and
    selected candidate by the eventual authoritative server runner.  No action
    or count target is representable here.
    """

    video_id: str
    baseline: MechanismEstimate
    reverse_primary: ViewEstimate
    warp_075_primary: ViewEstimate
    warp_125_primary: ViewEstimate
    duplicate_time_primary: ViewEstimate
    legal_split_part_float_counts: tuple[float | None, float | None]
    raw_rotation_minus15: ViewEstimate
    raw_rotation_plus15: ViewEstimate
    raw_scale_085: ViewEstimate
    raw_scale_115: ViewEstimate
    raw_joint_dropout: ViewEstimate
    learned_augmentation_a: ViewEstimate
    learned_augmentation_b: ViewEstimate
    static_null_primary: ViewEstimate
    second_derangement_shuffle: MechanismEstimate
    transform_lineage: Train337TransformLineage

    def __post_init__(self) -> None:
        identifier = str(self.video_id).strip()
        if not identifier:
            raise ValueError("train337 audit video_id must be non-empty")
        estimates = (
            self.baseline.primary,
            self.baseline.raw_control,
            self.baseline.untrained_control,
            self.baseline.temporal_derangement_control,
            self.reverse_primary,
            self.warp_075_primary,
            self.warp_125_primary,
            self.duplicate_time_primary,
            self.raw_rotation_minus15,
            self.raw_rotation_plus15,
            self.raw_scale_085,
            self.raw_scale_115,
            self.raw_joint_dropout,
            self.learned_augmentation_a,
            self.learned_augmentation_b,
            self.static_null_primary,
            self.second_derangement_shuffle.primary,
            self.second_derangement_shuffle.untrained_control,
            self.second_derangement_shuffle.temporal_derangement_control,
        )
        if any(item.video_id != identifier for item in estimates):
            raise ValueError("every train337 transformed estimate must retain source video_id")
        candidate_ids = {
            self.baseline.candidate_id,
            self.second_derangement_shuffle.candidate_id,
        }
        candidate_ids.update(item.candidate_id for item in estimates)
        if len(candidate_ids) != 1:
            raise ValueError("every train337 audit arm must use the same selected candidate")
        if (
            self.baseline.epi_permutation_map_sha256
            == self.second_derangement_shuffle.epi_permutation_map_sha256
        ):
            raise ValueError("Epi real and second-derangement null maps must be independent")
        if self.transform_lineage.video_id != identifier:
            raise ValueError("transform lineage video_id differs from audit row")
        expected_artifacts = _train337_artifact_hashes_from_values(
            baseline=self.baseline,
            reverse_primary=self.reverse_primary,
            warp_075_primary=self.warp_075_primary,
            warp_125_primary=self.warp_125_primary,
            duplicate_time_primary=self.duplicate_time_primary,
            legal_split_part_float_counts=self.legal_split_part_float_counts,
            raw_rotation_minus15=self.raw_rotation_minus15,
            raw_rotation_plus15=self.raw_rotation_plus15,
            raw_scale_085=self.raw_scale_085,
            raw_scale_115=self.raw_scale_115,
            raw_joint_dropout=self.raw_joint_dropout,
            learned_augmentation_a=self.learned_augmentation_a,
            learned_augmentation_b=self.learned_augmentation_b,
            static_null_primary=self.static_null_primary,
            second_derangement_shuffle=self.second_derangement_shuffle,
        )
        if self.transform_lineage.artifacts != expected_artifacts:
            raise ValueError("transform lineage artifacts do not bind the audit values")
        for value in self.legal_split_part_float_counts:
            if value is not None and (not np.isfinite(value) or value < 0.0):
                raise ValueError("legal split float counts must be finite and non-negative")
        object.__setattr__(self, "video_id", identifier)


def _train337_artifact_hashes_from_values(
    *,
    baseline: MechanismEstimate,
    reverse_primary: ViewEstimate,
    warp_075_primary: ViewEstimate,
    warp_125_primary: ViewEstimate,
    duplicate_time_primary: ViewEstimate,
    legal_split_part_float_counts: tuple[float | None, float | None],
    raw_rotation_minus15: ViewEstimate,
    raw_rotation_plus15: ViewEstimate,
    raw_scale_085: ViewEstimate,
    raw_scale_115: ViewEstimate,
    raw_joint_dropout: ViewEstimate,
    learned_augmentation_a: ViewEstimate,
    learned_augmentation_b: ViewEstimate,
    static_null_primary: ViewEstimate,
    second_derangement_shuffle: MechanismEstimate,
) -> tuple[tuple[str, str], ...]:
    payloads: tuple[tuple[str, Any], ...] = (
        ("baseline", _mechanism_estimate_payload(baseline)),
        ("reverse_primary", _view_estimate_payload(reverse_primary)),
        ("warp_075_primary", _view_estimate_payload(warp_075_primary)),
        ("warp_125_primary", _view_estimate_payload(warp_125_primary)),
        ("duplicate_time_primary", _view_estimate_payload(duplicate_time_primary)),
        (
            "legal_split_part_float_counts",
            list(legal_split_part_float_counts),
        ),
        ("raw_rotation_minus15", _view_estimate_payload(raw_rotation_minus15)),
        ("raw_rotation_plus15", _view_estimate_payload(raw_rotation_plus15)),
        ("raw_scale_085", _view_estimate_payload(raw_scale_085)),
        ("raw_scale_115", _view_estimate_payload(raw_scale_115)),
        ("raw_joint_dropout", _view_estimate_payload(raw_joint_dropout)),
        ("learned_augmentation_a", _view_estimate_payload(learned_augmentation_a)),
        ("learned_augmentation_b", _view_estimate_payload(learned_augmentation_b)),
        ("static_null_primary", _view_estimate_payload(static_null_primary)),
        (
            "second_derangement_shuffle",
            _mechanism_estimate_payload(second_derangement_shuffle),
        ),
    )
    return tuple((name, _artifact_sha256(name, payload)) for name, payload in payloads)


def _build_transform_lineage_for_values(
    *,
    video_id: str,
    source_authority_sha256: str,
    transform_recipe_sha256: str,
    artifacts: Sequence[tuple[str, str]],
) -> Train337TransformLineage:
    frozen_artifacts = tuple(artifacts)
    return Train337TransformLineage(
        video_id=video_id,
        source_authority_sha256=source_authority_sha256,
        transform_recipe_sha256=transform_recipe_sha256,
        artifacts=frozen_artifacts,
        receipt_sha256=_transform_lineage_digest(
            video_id=video_id,
            source_authority_sha256=source_authority_sha256,
            transform_recipe_sha256=transform_recipe_sha256,
            artifacts=frozen_artifacts,
        ),
    )


def build_train337_video_audit(
    *,
    video_id: str,
    source_authority_sha256: str,
    transform_recipe_sha256: str,
    baseline: MechanismEstimate,
    reverse_primary: ViewEstimate,
    warp_075_primary: ViewEstimate,
    warp_125_primary: ViewEstimate,
    duplicate_time_primary: ViewEstimate,
    legal_split_part_float_counts: tuple[float | None, float | None],
    raw_rotation_minus15: ViewEstimate,
    raw_rotation_plus15: ViewEstimate,
    raw_scale_085: ViewEstimate,
    raw_scale_115: ViewEstimate,
    raw_joint_dropout: ViewEstimate,
    learned_augmentation_a: ViewEstimate,
    learned_augmentation_b: ViewEstimate,
    static_null_primary: ViewEstimate,
    second_derangement_shuffle: MechanismEstimate,
) -> Train337VideoAudit:
    """Build one audit row while hashing every frozen transform outcome."""

    artifacts = _train337_artifact_hashes_from_values(
        baseline=baseline,
        reverse_primary=reverse_primary,
        warp_075_primary=warp_075_primary,
        warp_125_primary=warp_125_primary,
        duplicate_time_primary=duplicate_time_primary,
        legal_split_part_float_counts=legal_split_part_float_counts,
        raw_rotation_minus15=raw_rotation_minus15,
        raw_rotation_plus15=raw_rotation_plus15,
        raw_scale_085=raw_scale_085,
        raw_scale_115=raw_scale_115,
        raw_joint_dropout=raw_joint_dropout,
        learned_augmentation_a=learned_augmentation_a,
        learned_augmentation_b=learned_augmentation_b,
        static_null_primary=static_null_primary,
        second_derangement_shuffle=second_derangement_shuffle,
    )
    lineage = _build_transform_lineage_for_values(
        video_id=video_id,
        source_authority_sha256=source_authority_sha256,
        transform_recipe_sha256=transform_recipe_sha256,
        artifacts=artifacts,
    )
    return Train337VideoAudit(
        video_id=video_id,
        baseline=baseline,
        reverse_primary=reverse_primary,
        warp_075_primary=warp_075_primary,
        warp_125_primary=warp_125_primary,
        duplicate_time_primary=duplicate_time_primary,
        legal_split_part_float_counts=legal_split_part_float_counts,
        raw_rotation_minus15=raw_rotation_minus15,
        raw_rotation_plus15=raw_rotation_plus15,
        raw_scale_085=raw_scale_085,
        raw_scale_115=raw_scale_115,
        raw_joint_dropout=raw_joint_dropout,
        learned_augmentation_a=learned_augmentation_a,
        learned_augmentation_b=learned_augmentation_b,
        static_null_primary=static_null_primary,
        second_derangement_shuffle=second_derangement_shuffle,
        transform_lineage=lineage,
    )


def hash_fixed_subset(
    video_ids: Iterable[str],
    *,
    seed: int,
    size: int,
) -> tuple[tuple[str, ...], str]:
    """Select a deterministic, order-independent SHA-256-ranked subset."""

    identifiers = tuple(str(item).strip() for item in video_ids)
    if any(not item for item in identifiers):
        raise ValueError("hash subset video_ids must be non-empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("hash subset video_ids must be unique")
    if size < 1 or size > len(identifiers):
        raise ValueError("hash subset size must be in [1, number of videos]")
    ranked = sorted(
        identifiers,
        key=lambda item: (hashlib.sha256(f"{seed}\0{item}".encode()).hexdigest(), item),
    )
    selected = tuple(ranked[:size])
    payload = json.dumps(
        {"seed": int(seed), "video_ids": list(selected)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return selected, hashlib.sha256(payload).hexdigest()


def _relative_error(reference: float | None, value: float | None) -> float:
    if reference is None or value is None or not np.isfinite(reference) or not np.isfinite(value):
        return math.inf
    if reference <= _EPSILON:
        return 0.0 if abs(value - reference) <= _EPSILON else math.inf
    return abs(value - reference) / reference


def _median_and_p90(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("audit statistic requires a non-empty vector")
    if not np.isfinite(array).all():
        return math.inf, math.inf
    return float(np.median(array)), float(np.quantile(array, 0.9, method="linear"))


@dataclass(frozen=True, slots=True)
class ArmMechanismSummary:
    """Recomputed real-vs-independent-shuffle statistic for one arm."""

    arm: Literal["L", "E0", "Epi"]
    shuffle_confidence_ratio_median: float
    real_minus_shuffle_peak_margin_median: float
    mechanism_pass: bool
    per_video: tuple[tuple[str, float, float], ...]


def _arm_view(estimate: MechanismEstimate, arm: Literal["L", "E0", "Epi"]) -> ViewEstimate:
    if arm == "L":
        return estimate.primary
    if arm == "E0":
        return estimate.untrained_control
    return estimate.temporal_derangement_control


def _mechanism_summary(
    records: Sequence[Train337VideoAudit],
    arm: Literal["L", "E0", "Epi"],
    protocol: Train337GateProtocol,
) -> ArmMechanismSummary:
    rows: list[tuple[str, float, float]] = []
    for record in records:
        real = _arm_view(record.baseline, arm)
        shuffled = _arm_view(record.second_derangement_shuffle, arm)
        ratio = shuffled.periodic_confidence / max(real.periodic_confidence, _EPSILON)
        margin = real.mean_peak_share - shuffled.mean_peak_share
        rows.append((record.video_id, float(ratio), float(margin)))
    ratio_median = float(np.median([row[1] for row in rows]))
    margin_median = float(np.median([row[2] for row in rows]))
    passed = bool(
        ratio_median <= protocol.shuffle_confidence_ratio_median_maximum
        and margin_median >= protocol.real_minus_shuffle_peak_margin_median_minimum
    )
    return ArmMechanismSummary(
        arm=arm,
        shuffle_confidence_ratio_median=ratio_median,
        real_minus_shuffle_peak_margin_median=margin_median,
        mechanism_pass=passed,
        per_video=tuple(rows),
    )


@dataclass(frozen=True, slots=True)
class Train337GateReport:
    """Complete target-free authorization decision for dev84 prediction."""

    config_fingerprint: str
    candidate_id: str
    selection_report_sha256: str
    heldout_report_sha256: str
    authority_receipt_sha256: str
    authority_adapter_status: Literal["unwired_fail_closed"]
    video_ids_sha256: str
    hash_subset_sha256: str
    hash_subset_video_ids: tuple[str, ...]
    metrics: tuple[tuple[str, float | int], ...]
    mechanism: tuple[ArmMechanismSummary, ...]
    failed_gates: tuple[str, ...]
    authorization_blockers: tuple[str, ...]
    numerical_gates_passed: bool
    authorized_for_dev84: bool

    def __post_init__(self) -> None:
        if self.authority_adapter_status != "unwired_fail_closed":
            raise ValueError("unknown train337 authority adapter status")
        if self.authorized_for_dev84:
            raise ValueError("unwired train337 authority can never authorize dev84")
        if self.numerical_gates_passed != (not self.failed_gates):
            raise ValueError("train337 numerical decision is inconsistent")
        if not self.authorization_blockers:
            raise ValueError("fail-closed train337 report requires an authorization blocker")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "classification": "train337_label_free_single_expert_gate",
            "config_fingerprint": self.config_fingerprint,
            "candidate_id": self.candidate_id,
            "selection_report_sha256": self.selection_report_sha256,
            "heldout_report_sha256": self.heldout_report_sha256,
            "authority_receipt_sha256": self.authority_receipt_sha256,
            "authority_adapter_status": self.authority_adapter_status,
            "video_ids_sha256": self.video_ids_sha256,
            "hash_subset_sha256": self.hash_subset_sha256,
            "hash_subset_video_ids": list(self.hash_subset_video_ids),
            "metrics": dict(self.metrics),
            "mechanism": {
                item.arm: {
                    "shuffle_confidence_ratio_median": (
                        item.shuffle_confidence_ratio_median
                    ),
                    "real_minus_shuffle_peak_margin_median": (
                        item.real_minus_shuffle_peak_margin_median
                    ),
                    "mechanism_pass": item.mechanism_pass,
                    "per_video": [
                        {
                            "video_id": video_id,
                            "shuffle_confidence_ratio": ratio,
                            "real_minus_shuffle_peak_margin": margin,
                        }
                        for video_id, ratio, margin in item.per_video
                    ],
                }
                for item in self.mechanism
            },
            "failed_gates": list(self.failed_gates),
            "authorization_blockers": list(self.authorization_blockers),
            "numerical_gates_passed": self.numerical_gates_passed,
            "authorized_for_dev84": self.authorized_for_dev84,
            "action_or_count_targets_consumed": False,
        }


def evaluate_train337_gate(
    config: SegmentLocalSpectralConfig,
    selection: SyntheticSelectionReport,
    heldout: SyntheticHeldoutReport,
    authority: Train337Authority,
    records: Sequence[Train337VideoAudit],
) -> Train337GateReport:
    """Recompute every frozen train337 gate without accepting labels."""

    protocol = config.train337
    selected_candidate = replay_and_validate_synthetic_heldout(config, selection, heldout)
    if authority.adapter_status != protocol.authority_adapter_status:
        raise ValueError("train337 authority adapter status differs from frozen config")
    if authority.transform_recipe_sha256 != protocol.transform_recipe.fingerprint:
        raise ValueError("train337 transform recipe authority mismatch")
    if authority.selection_report_sha256 != selection.canonical_sha256:
        raise ValueError("train337 authority does not bind exact synthetic selection")
    if authority.heldout_report_sha256 != heldout.canonical_sha256:
        raise ValueError("train337 authority does not bind exact held-out PASS")
    frozen = tuple(records)
    if len(frozen) != protocol.expected_video_count:
        raise ValueError(f"train337 gate requires exactly {protocol.expected_video_count} rows")
    identifiers = tuple(record.video_id for record in frozen)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("train337 audit video_id values must be unique")
    if identifiers != authority.canonical_video_ids:
        raise ValueError("train337 rows must equal the canonical 337 ID registry in exact order")
    if canonical_video_ids_sha256(identifiers) != authority.canonical_video_ids_sha256:
        raise ValueError("train337 row video-ID digest differs from canonical authority")
    if any(record.baseline.candidate_id != selected_candidate.canonical_id for record in frozen):
        raise ValueError("train337 rows do not match selected candidate")
    for record in frozen:
        if record.transform_lineage.source_authority_sha256 != authority.source_authority_sha256:
            raise ValueError("train337 transform lineage source authority mismatch")
        if record.transform_lineage.transform_recipe_sha256 != protocol.transform_recipe.fingerprint:
            raise ValueError("train337 row transform recipe mismatch")
    ids_sha256 = authority.canonical_video_ids_sha256
    subset_ids, subset_sha256 = hash_fixed_subset(
        identifiers,
        seed=protocol.hash_subset_seed,
        size=protocol.hash_subset_size,
    )
    by_id = {record.video_id: record for record in frozen}
    subset = tuple(by_id[video_id] for video_id in subset_ids)

    sufficient = tuple(
        record
        for record in frozen
        if len(record.baseline.primary.segments) == 1
        and record.baseline.primary.segments[0].candidate_window_count > 0
        and record.baseline.primary.segments[0].available_window_count > 0
    )
    sufficient_share = len(sufficient) / protocol.expected_video_count
    eligible_records = tuple(
        record for record in sufficient if record.baseline.primary.status == "eligible"
    )
    canonical_eligible_share = len(eligible_records) / protocol.expected_video_count
    conditional_eligible_share = len(eligible_records) / len(sufficient) if sufficient else 0.0
    accepted_windows = tuple(
        window
        for record in eligible_records
        for segment in record.baseline.primary.segments
        for window in segment.windows
        if window.accepted
    )
    if accepted_windows:
        low_boundary_share = float(np.mean([item.low_band_boundary for item in accepted_windows]))
        high_boundary_share = float(np.mean([item.high_band_boundary for item in accepted_windows]))
    else:
        low_boundary_share = 1.0
        high_boundary_share = 1.0
    rounded = tuple(
        int(record.baseline.primary.rounded_count)
        for record in eligible_records
        if record.baseline.primary.rounded_count is not None
    )
    if rounded:
        _, mode_counts = np.unique(np.asarray(rounded, dtype=np.int64), return_counts=True)
        rounded_mode_share = int(np.max(mode_counts)) / len(rounded)
        rounded_bin_count = len(set(rounded))
    else:
        rounded_mode_share = 1.0
        rounded_bin_count = 0

    reverse_errors: list[float] = []
    warp_errors: list[float] = []
    split_errors: list[float] = []
    raw_agreements = 0
    learned_augmentation_errors: list[float] = []
    static_null_positives = 0
    for record in subset:
        primary = record.baseline.primary.float_count
        reverse_errors.append(_relative_error(primary, record.reverse_primary.float_count))
        warp_errors.append(
            max(
                _relative_error(primary, record.warp_075_primary.float_count),
                _relative_error(primary, record.warp_125_primary.float_count),
                _relative_error(primary, record.duplicate_time_primary.float_count),
            )
        )
        split_left, split_right = record.legal_split_part_float_counts
        split_total = (
            split_left + split_right
            if split_left is not None and split_right is not None
            else None
        )
        split_errors.append(_relative_error(primary, split_total))
        raw_count = record.baseline.raw_control.rounded_count
        raw_transforms = (
            record.raw_rotation_minus15,
            record.raw_rotation_plus15,
            record.raw_scale_085,
            record.raw_scale_115,
            record.raw_joint_dropout,
        )
        if raw_count is not None and all(
            item.status == "eligible" and item.rounded_count == raw_count
            for item in raw_transforms
        ):
            raw_agreements += 1
        learned_augmentation_errors.append(
            _relative_error(
                record.learned_augmentation_a.float_count,
                record.learned_augmentation_b.float_count,
            )
        )
        if (
            record.static_null_primary.status == "eligible"
            and record.static_null_primary.rounded_count is not None
            and record.static_null_primary.rounded_count > 0
        ):
            static_null_positives += 1

    reverse_median, reverse_p90 = _median_and_p90(reverse_errors)
    warp_median, warp_p90 = _median_and_p90(warp_errors)
    split_median, split_p90 = _median_and_p90(split_errors)
    learned_aug_median, learned_aug_p90 = _median_and_p90(learned_augmentation_errors)
    raw_agreement = raw_agreements / len(subset)
    static_positive_share = static_null_positives / len(subset)
    mechanisms = tuple(
        _mechanism_summary(subset, arm, protocol) for arm in ("L", "E0", "Epi")
    )
    mechanism_by_arm = {item.arm: item for item in mechanisms}

    metrics: tuple[tuple[str, float | int], ...] = (
        ("canonical_video_count", protocol.expected_video_count),
        ("one_segment_sufficient_video_count", len(sufficient)),
        ("one_segment_sufficient_share", sufficient_share),
        ("canonical_one_segment_eligible_share", canonical_eligible_share),
        ("conditional_eligible_given_sufficient_share_report_only", conditional_eligible_share),
        ("accepted_window_count", len(accepted_windows)),
        ("low_band_boundary_share", low_boundary_share),
        ("high_band_boundary_share", high_boundary_share),
        ("rounded_mode_share", rounded_mode_share),
        ("rounded_bin_count", rounded_bin_count),
        ("reverse_relative_error_median", reverse_median),
        ("reverse_relative_error_p90", reverse_p90),
        ("time_warp_disagreement_median", warp_median),
        ("time_warp_disagreement_p90", warp_p90),
        ("split_additivity_relative_error_median", split_median),
        ("split_additivity_relative_error_p90", split_p90),
        ("raw_invariance_rounded_agreement", raw_agreement),
        ("learned_augmentation_disagreement_median", learned_aug_median),
        ("learned_augmentation_disagreement_p90", learned_aug_p90),
        ("static_null_positive_share", static_positive_share),
    )
    failed: list[str] = []
    if sufficient_share < protocol.one_segment_sufficient_share_minimum:
        failed.append("one_segment_sufficient_share")
    if canonical_eligible_share < protocol.one_segment_eligible_minimum:
        failed.append("canonical_one_segment_eligible_share")
    if not low_boundary_share < protocol.low_band_boundary_share_maximum_exclusive:
        failed.append("low_band_boundary_share")
    if not high_boundary_share < protocol.high_band_boundary_share_maximum_exclusive:
        failed.append("high_band_boundary_share")
    if not rounded_mode_share < protocol.rounded_mode_share_maximum_exclusive:
        failed.append("rounded_mode_share")
    if rounded_bin_count < protocol.rounded_bin_minimum:
        failed.append("rounded_bin_count")
    if reverse_median > protocol.reverse_relative_error_median_maximum:
        failed.append("reverse_relative_error_median")
    if reverse_p90 > protocol.reverse_relative_error_p90_maximum:
        failed.append("reverse_relative_error_p90")
    if warp_median > protocol.time_warp_disagreement_median_maximum:
        failed.append("time_warp_disagreement_median")
    if warp_p90 > protocol.time_warp_disagreement_p90_maximum:
        failed.append("time_warp_disagreement_p90")
    if split_median > protocol.split_additivity_median_maximum:
        failed.append("split_additivity_median")
    if split_p90 > protocol.split_additivity_p90_maximum:
        failed.append("split_additivity_p90")
    if raw_agreement < protocol.raw_invariance_rounded_agreement_minimum:
        failed.append("raw_invariance_rounded_agreement")
    if learned_aug_median > protocol.learned_augmentation_disagreement_median_maximum:
        failed.append("learned_augmentation_disagreement_median")
    if learned_aug_p90 > protocol.learned_augmentation_disagreement_p90_maximum:
        failed.append("learned_augmentation_disagreement_p90")
    if static_positive_share > protocol.static_null_positive_share_maximum:
        failed.append("static_null_positive_share")
    if not mechanism_by_arm["L"].mechanism_pass:
        failed.append("L_time_mechanism")
    if mechanism_by_arm["Epi"].mechanism_pass:
        failed.append("Epi_control_must_fail_time_mechanism")

    authorization_blockers = ("authoritative_train337_adapter_unwired_fail_closed",)
    return Train337GateReport(
        config_fingerprint=config.fingerprint,
        candidate_id=selected_candidate.canonical_id,
        selection_report_sha256=selection.canonical_sha256,
        heldout_report_sha256=heldout.canonical_sha256,
        authority_receipt_sha256=authority.authority_receipt_sha256,
        authority_adapter_status=authority.adapter_status,
        video_ids_sha256=ids_sha256,
        hash_subset_sha256=subset_sha256,
        hash_subset_video_ids=subset_ids,
        metrics=metrics,
        mechanism=mechanisms,
        failed_gates=tuple(failed),
        authorization_blockers=authorization_blockers,
        numerical_gates_passed=not failed,
        authorized_for_dev84=False,
    )
