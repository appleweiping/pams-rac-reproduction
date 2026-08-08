from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

import pams.single_expert_readout as single_readout
from pams.single_expert_readout import (
    AuthorizedSegment,
    FullSegmentEncodingReceipt,
    MechanismEstimate,
    MechanismVideo,
    NullFamilyScore,
    PositiveFamilyScore,
    RepresentationVideo,
    SecondDerangementControls,
    SecondDerangementEstimate,
    SegmentEstimate,
    SegmentPermutation,
    SpectralCandidate,
    SyntheticCandidateScore,
    SyntheticSelectionFailure,
    TemporalDerangementReceipt,
    Train337VideoAudit,
    Train337TransformApplicability,
    ViewEstimate,
    WindowEstimate,
    assess_train337_transform_applicability,
    build_fail_closed_train337_authority,
    build_full_segment_encoding_receipt,
    build_preencoder_segment_derangement,
    build_train337_video_audit,
    build_window_plan,
    estimate_legal_split_views,
    estimate_mechanism_views,
    estimate_representation,
    estimate_second_derangement_controls,
    evaluate_train337_gate,
    hash_fixed_subset,
    load_segment_local_spectral_config,
    ordered_train337_records_sha256,
    replay_and_validate_synthetic_heldout,
    replay_and_validate_synthetic_selection,
    replay_selected_candidate_heldout,
    select_synthetic_candidate,
    synthetic_case_plan,
    train337_source_authority_sha256,
    transform_applicability_registry_sha256,
)

REPOSITORY = Path(__file__).parents[1]
CONFIG_PATH = REPOSITORY / "configs/readouts/segment_local_spectral_single_v1.yaml"


def _config():
    return load_segment_local_spectral_config(CONFIG_PATH)


def _candidate() -> SpectralCandidate:
    return SpectralCandidate(64, "sum", 1.0, "ridge_integral", 0.1)


def _periodic_video(
    *,
    video_id: str = "periodic",
    frames: int = 256,
    count: float = 8.0,
    segments: tuple[AuthorizedSegment, ...] | None = None,
) -> RepresentationVideo:
    phase = np.linspace(0.0, 2.0 * np.pi * count, frames, endpoint=False)
    features = np.stack(
        [np.sin(phase + index * 0.07) for index in range(34)],
        axis=1,
    ).astype(np.float32)
    raw = np.zeros((frames, 17, 2), dtype=np.float32)
    for joint in range(17):
        raw[:, joint, 0] = 0.1 * np.sin(phase + joint * 0.11)
        raw[:, joint, 1] = 0.1 * np.cos(phase - joint * 0.09)
    mask = np.ones((frames, 17), dtype=np.bool_)
    valid = np.ones(frames, dtype=np.bool_)
    return RepresentationVideo(
        video_id=video_id,
        features=features,
        raw_xy=raw,
        joint_mask=mask,
        valid_mask=valid,
        segments=segments or (AuthorizedSegment("segment-0", 0, frames),),
    )


def test_config_is_exact_32_grid_and_strictly_inferred() -> None:
    config = _config()
    assert config.eligible_as_exact_author_baseline is False
    assert len(config.candidates) == 32
    assert len({item.canonical_id for item in config.candidates}) == 32
    assert config.synthetic.selector_seed == 2026
    assert config.synthetic.heldout_seed == 3407
    assert config.fixed.terminal_window == "none_zero_grid_only"
    assert config.fixed.minimum_accepted_interval_union_fraction == 0.8
    assert config.synthetic.positive_family_gate_policy == (
        "all_frozen_generation_truth_families_hard_gated"
    )
    assert config.train337.one_segment_sufficient_share_minimum == 0.9
    assert config.train337.minimum_transform_applicable_video_count == 64
    assert "target" not in inspect.signature(evaluate_train337_gate).parameters


def test_config_rejects_unknown_and_relaxed_gate_values(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["dev84_targets"] = "forbidden.json"
    invalid = tmp_path / "unknown.yaml"
    invalid.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_segment_local_spectral_config(invalid)

    payload.pop("dev84_targets")
    payload["synthetic"]["overall_nmae_maximum"] = 0.081
    relaxed = tmp_path / "relaxed.yaml"
    relaxed.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="thresholds were changed"):
        load_segment_local_spectral_config(relaxed)


def test_periodic_ridge_counts_without_trim_clip_or_expert_vote() -> None:
    estimate = estimate_representation(_periodic_video(), _candidate())
    assert estimate.status == "eligible"
    assert estimate.rounded_count == 8
    assert estimate.float_count == pytest.approx(8.0, abs=0.1)
    assert estimate.segments[0].candidate_window_count == 13
    assert estimate.segments[0].available_window_count == 13
    assert estimate.segments[0].accepted_window_count >= 1
    assert estimate.abstention_reason is None


def test_window_authority_never_crosses_reset_or_uses_weak_joint() -> None:
    frames = 256
    base = _periodic_video(
        frames=frames,
        segments=(
            AuthorizedSegment("left", 0, 112),
            AuthorizedSegment("right", 144, 256),
        ),
    )
    features = np.asarray(base.features).copy()
    raw = np.asarray(base.raw_xy).copy()
    mask = np.asarray(base.joint_mask).copy()
    valid = np.asarray(base.valid_mask).copy()
    features[112:144] = 0.0
    raw[112:144] = 0.0
    mask[112:144] = False
    valid[112:144] = False
    mask[0:64, 0] = False
    raw[0:64, 0] = 0.0
    video = RepresentationVideo("reset", features, raw, mask, valid, base.segments)
    plans = build_window_plan(video, _candidate())
    keys = [window for plan in plans for window in plan.windows]
    assert keys
    assert all(
        any(
            window.start >= segment.start and window.stop <= segment.stop
            for segment in video.segments
        )
        for window in keys
    )
    first = next(window for window in keys if window.start == 0)
    assert 0 not in first.stable_joints


def test_constant_abstains_and_multi_segment_total_is_always_undefined() -> None:
    periodic = _periodic_video()
    constant = RepresentationVideo(
        "constant",
        np.zeros_like(periodic.features),
        np.zeros_like(periodic.raw_xy),
        periodic.joint_mask,
        periodic.valid_mask,
        periodic.segments,
    )
    assert estimate_representation(constant, _candidate()).status == "abstain"

    multi = _periodic_video(
        frames=256,
        segments=(
            AuthorizedSegment("left", 0, 112),
            AuthorizedSegment("right", 144, 256),
        ),
    )
    features = np.asarray(multi.features).copy()
    raw = np.asarray(multi.raw_xy).copy()
    mask = np.asarray(multi.joint_mask).copy()
    valid = np.asarray(multi.valid_mask).copy()
    features[112:144] = 0.0
    raw[112:144] = 0.0
    mask[112:144] = False
    valid[112:144] = False
    multi = RepresentationVideo("multi", features, raw, mask, valid, multi.segments)
    result = estimate_representation(multi, _candidate())
    assert result.status == "undefined_multi_segment"
    assert result.float_count is None
    assert result.rounded_count is None
    assert result.abstention_reason == "multiple_authorized_segments_never_sum"


def test_representation_boundary_rejects_nonzero_padding_and_weak_coordinates() -> None:
    video = _periodic_video(frames=128, count=4)
    features = np.asarray(video.features).copy()
    raw = np.asarray(video.raw_xy).copy()
    mask = np.asarray(video.joint_mask).copy()
    valid = np.asarray(video.valid_mask).copy()
    valid[-1] = False
    mask[-1] = False
    raw[-1] = 0.0
    with pytest.raises(ValueError, match="features on invalid"):
        RepresentationVideo("bad-padding", features, raw, mask, valid, video.segments)

    features[-1] = 0.0
    with pytest.raises(ValueError, match="authorized segments must contain only valid"):
        RepresentationVideo("bad-context", features, raw, mask, valid, video.segments)

    valid[-1] = True
    mask[-1] = True
    mask[0, 0] = False
    with pytest.raises(ValueError, match="weak/invalid joints"):
        RepresentationVideo("bad-weak", features, raw, mask, valid, video.segments)


def _encoding_receipt(
    view: str,
    video_id: str,
    features: np.ndarray,
    segments: tuple[AuthorizedSegment, ...],
    input_hashes: dict[str, str],
    *,
    state: str,
    derangement_hash: str | None = None,
) -> FullSegmentEncodingReceipt:
    initialization = "2" * 64
    role = (
        "frozen_untrained_initialization"
        if view in ("E0", "E0pi2")
        else "trained_cycleback"
    )
    return build_full_segment_encoding_receipt(
        view=view,  # type: ignore[arg-type]
        video_id=video_id,
        encoder_state_sha256=state,
        encoder_config_sha256="5" * 64,
        encoder_implementation_sha256="6" * 64,
        initialization_state_sha256=initialization,
        checkpoint_role=role,  # type: ignore[arg-type]
        features=features,
        repeat_features=np.array(features, copy=True),
        segments=segments,
        input_pose_sha256_by_segment=input_hashes,
        derangement_map_sha256=derangement_hash,
    )


def test_mechanism_views_require_preencoder_derangement_and_full_segment_receipts() -> None:
    video = _periodic_video(frames=128, count=4)
    deranged = build_preencoder_segment_derangement(
        video.video_id,
        video.raw_xy,
        video.joint_mask,
        video.valid_mask,
        video.segments,
        seed=2026,
    )
    permutation = np.asarray(deranged.receipt.permutations[0].source_indices)
    learned = np.asarray(video.features)
    untrained = np.zeros_like(learned)
    epi = learned[permutation]
    source_hashes = {
        item.segment_id: item.source_pose_sha256 for item in deranged.receipt.permutations
    }
    epi_hashes = {
        item.segment_id: item.deranged_pose_sha256 for item in deranged.receipt.permutations
    }
    mechanism = MechanismVideo(
        video_id=video.video_id,
        learned=learned,
        untrained=untrained,
        epi=epi,
        raw_xy=video.raw_xy,
        joint_mask=video.joint_mask,
        valid_mask=video.valid_mask,
        segments=video.segments,
        epi_receipt=deranged.receipt,
        learned_encoding_receipt=_encoding_receipt(
            "L", video.video_id, learned, video.segments, source_hashes, state="1" * 64
        ),
        untrained_encoding_receipt=_encoding_receipt(
            "E0", video.video_id, untrained, video.segments, source_hashes, state="2" * 64
        ),
        epi_encoding_receipt=_encoding_receipt(
            "Epi",
            video.video_id,
            epi,
            video.segments,
            epi_hashes,
            state="1" * 64,
            derangement_hash=deranged.receipt.permutation_map_sha256,
        ),
    )
    result = estimate_mechanism_views(mechanism, _candidate())
    assert result.primary.view == "L"
    assert result.raw_control.view == "R"
    assert result.untrained_control.view == "E0"
    assert result.temporal_derangement_control.view == "Epi"
    assert result.primary.window_keys == result.raw_control.window_keys
    assert result.primary.window_keys == result.untrained_control.window_keys
    assert result.primary.window_keys == result.temporal_derangement_control.window_keys
    assert mechanism.learned_encoding_receipt.context_policy == (
        "full_authorized_segment_absolute_native_pe_v1"
    )
    assert mechanism.learned_encoding_receipt.execution_mode == (
        "eval_deterministic_no_grad_no_optimizer_update"
    )

    map2 = build_preencoder_segment_derangement(
        video.video_id,
        video.raw_xy,
        video.joint_mask,
        video.valid_mask,
        video.segments,
        seed=3407,
    )
    map2_indices = np.asarray(map2.receipt.permutations[0].source_indices)
    assert tuple(map2_indices) != tuple(permutation)
    trained_map2 = learned[map2_indices]
    untrained_map2 = untrained[map2_indices]
    map2_hashes = {
        item.segment_id: item.deranged_pose_sha256
        for item in map2.receipt.permutations
    }
    controls = SecondDerangementControls(
        video_id=video.video_id,
        trained=trained_map2,
        untrained=untrained_map2,
        raw_xy=video.raw_xy,
        joint_mask=video.joint_mask,
        valid_mask=video.valid_mask,
        segments=video.segments,
        derangement_receipt=map2.receipt,
        trained_encoding_receipt=_encoding_receipt(
            "Lpi2",
            video.video_id,
            trained_map2,
            video.segments,
            map2_hashes,
            state="1" * 64,
            derangement_hash=map2.receipt.permutation_map_sha256,
        ),
        untrained_encoding_receipt=_encoding_receipt(
            "E0pi2",
            video.video_id,
            untrained_map2,
            video.segments,
            map2_hashes,
            state="2" * 64,
            derangement_hash=map2.receipt.permutation_map_sha256,
        ),
    )
    second = estimate_second_derangement_controls(mechanism, controls, _candidate())
    assert second.trained_control.view == "Lpi2"
    assert second.untrained_control.view == "E0pi2"
    assert second.derangement_receipt == map2.receipt

    pivot, split_left, split_right = estimate_legal_split_views(
        mechanism,
        _candidate(),
    )
    assert pivot == 63
    assert split_left.segments[0].segment == AuthorizedSegment("segment-0", 0, 64)
    assert split_right.segments[0].segment == AuthorizedSegment("segment-0", 63, 128)
    assert (
        split_left.segments[0].segment.length - 1
        + split_right.segments[0].segment.length
        - 1
        == video.segments[0].length - 1
    )

    wrong_epi_receipt = _encoding_receipt(
        "Epi",
        video.video_id,
        epi,
        video.segments,
        epi_hashes,
        state="7" * 64,
        derangement_hash=deranged.receipt.permutation_map_sha256,
    )
    with pytest.raises(ValueError, match="exact same learned encoder state"):
        MechanismVideo(
            video_id=video.video_id,
            learned=learned,
            untrained=untrained,
            epi=epi,
            raw_xy=video.raw_xy,
            joint_mask=video.joint_mask,
            valid_mask=video.valid_mask,
            segments=video.segments,
            epi_receipt=deranged.receipt,
            learned_encoding_receipt=mechanism.learned_encoding_receipt,
            untrained_encoding_receipt=mechanism.untrained_encoding_receipt,
            epi_encoding_receipt=wrong_epi_receipt,
        )

    original_permutation = deranged.receipt.permutations[0]
    forged_permutation = SegmentPermutation(
        segment_id=original_permutation.segment_id,
        source_indices=original_permutation.source_indices,
        source_pose_sha256=original_permutation.source_pose_sha256,
        deranged_pose_sha256="f" * 64,
    )
    forged_derangement = TemporalDerangementReceipt(
        method="pose_pre_encoder_segment_derangement_v1",
        video_id=video.video_id,
        seed=2026,
        permutations=(forged_permutation,),
        permutation_map_sha256=single_readout._permutation_digest(
            2026,
            (forged_permutation,),
            video_id=video.video_id,
        ),
    )
    forged_epi_receipt = _encoding_receipt(
        "Epi",
        video.video_id,
        epi,
        video.segments,
        {"segment-0": "f" * 64},
        state="1" * 64,
        derangement_hash=forged_derangement.permutation_map_sha256,
    )
    with pytest.raises(ValueError, match="does not replay from authority bytes"):
        MechanismVideo(
            video_id=video.video_id,
            learned=learned,
            untrained=untrained,
            epi=epi,
            raw_xy=video.raw_xy,
            joint_mask=video.joint_mask,
            valid_mask=video.valid_mask,
            segments=video.segments,
            epi_receipt=forged_derangement,
            learned_encoding_receipt=mechanism.learned_encoding_receipt,
            untrained_encoding_receipt=mechanism.untrained_encoding_receipt,
            epi_encoding_receipt=forged_epi_receipt,
        )


def test_synthetic_plan_freezes_all_families_without_allocating_arrays() -> None:
    specs = synthetic_case_plan()
    positives = [item for item in specs if item.kind in ("count", "variable_tempo", "corruption")]
    nulls = [item for item in specs if item.kind == "null"]
    resets = [item for item in specs if item.kind == "reset"]
    assert len(positives) == 118
    assert len(nulls) == 8 * 64
    assert len(resets) == 64
    assert {item.dimension for item in positives} == {34, 512}
    assert len({item.case_id for item in specs}) == len(specs)
    assert sum(item.family == "variable_tempo" for item in specs) == 24
    assert {
        item.profile for item in specs if item.family == "active_support"
    } == {"active_support_60", "active_support_80"}
    assert _config().synthetic.positive_family_gate_policy == (
        "all_frozen_generation_truth_families_hard_gated"
    )


def test_every_positive_synthetic_profile_realizes_exact_interval_count() -> None:
    cases = {
        (item.frames, float(item.target_count), item.profile)
        for item in synthetic_case_plan()
        if item.target_count is not None
    }
    for frames, count, profile in cases:
        _, phase = single_readout._periodic_latent(frames, count, profile)
        assert phase.shape == (frames,)
        assert phase[0] == 0.0
        assert phase[-1] == np.float64(2.0 * np.pi * count)
        assert np.isfinite(phase).all()
        assert np.all(np.diff(phase) >= 0.0)


def _window(
    video_id: str,
    segment: AuthorizedSegment | None = None,
    *,
    start: int | None = None,
    window_frames: int = 64,
) -> WindowEstimate:
    del video_id
    authority = segment or AuthorizedSegment("segment-0", 0, 128)
    window_start = authority.start if start is None else start
    stop = min(authority.stop, window_start + window_frames)
    return WindowEstimate(
        key=(authority.segment_id, window_start, stop),
        center=window_start + (stop - window_start - 1) / 2.0,
        accepted=True,
        abstention_reason=None,
        frequency=0.05,
        peak_share=0.2,
        signed_vector_acf=0.5,
        confidence=0.5,
        low_band_boundary=False,
        high_band_boundary=False,
        informative_dimensions=34,
    )


def _view(
    video_id: str,
    view: str,
    count: float | None,
    *,
    confidence: float = 0.5,
    peak: float = 0.2,
    candidate_window_count: int = 5,
    segment: AuthorizedSegment | None = None,
) -> ViewEstimate:
    segment = segment or AuthorizedSegment("segment-0", 0, 128)
    if count is None:
        segment_estimate = SegmentEstimate(
            segment,
            candidate_window_count,
            0,
            0,
            False,
            "abstain",
            "no_spectral_window_accepted",
            None,
            0.0,
            0,
            max(segment.length - 1, 0),
            0,
            (),
        )
        return ViewEstimate(
            video_id,
            view,  # type: ignore[arg-type]
            _candidate().canonical_id,
            "abstain",
            "no_spectral_window_accepted",
            None,
            None,
            confidence,
            peak,
            (segment_estimate,),
        )
    starts = tuple(range(segment.start, segment.stop - 64 + 1, 16))
    windows = tuple(_window(video_id, segment, start=start) for start in starts)
    coverage = single_readout._accepted_interval_coverage(segment, windows)
    segment_estimate = SegmentEstimate(
        segment=segment,
        candidate_window_count=len(starts),
        available_window_count=len(windows),
        accepted_window_count=len(windows),
        single_window_estimate=len(windows) == 1,
        status="eligible",
        abstention_reason=None,
        float_count=float(count),
        accepted_interval_union_fraction=coverage[0],
        initial_uncovered_intervals=coverage[1],
        terminal_uncovered_intervals=coverage[2],
        max_internal_uncovered_gap_intervals=coverage[3],
        windows=windows,
    )
    return ViewEstimate(
        video_id,
        view,  # type: ignore[arg-type]
        _candidate().canonical_id,
        "eligible",
        None,
        float(count),
        int(np.floor(float(count) + 0.5)),
        confidence,
        peak,
        (segment_estimate,),
    )


def test_interval_union_uses_l_minus_one_and_zero_grid_boundaries() -> None:
    for window_frames, hop, segment_length in ((64, 16, 79), (96, 24, 119)):
        segment = AuthorizedSegment("segment-0", 0, segment_length)
        starts = tuple(range(0, segment_length - window_frames + 1, hop))
        assert starts == (0,)
        accepted = tuple(
            _window(
                "coverage",
                segment,
                start=start,
                window_frames=window_frames,
            )
            for start in starts
        )
        fraction, initial, terminal, internal = (
            single_readout._accepted_interval_coverage(segment, accepted)
        )
        assert fraction == (window_frames - 1) / (segment_length - 1)
        assert fraction >= 0.8
        assert (initial, terminal, internal) == (0, hop - 1, 0)

    sparse_segment = AuthorizedSegment("segment-0", 0, 256)
    sparse = (_window("sparse", sparse_segment),)
    fraction, initial, terminal, internal = (
        single_readout._accepted_interval_coverage(sparse_segment, sparse)
    )
    assert fraction == 63 / 255
    assert (initial, terminal, internal) == (0, 192, 0)

    with pytest.raises(ValueError, match="coverage diagnostics do not replay"):
        SegmentEstimate(
            segment=AuthorizedSegment("segment-0", 0, 128),
            candidate_window_count=5,
            available_window_count=1,
            accepted_window_count=1,
            single_window_estimate=True,
            status="eligible",
            abstention_reason=None,
            float_count=4.0,
            accepted_interval_union_fraction=1.0,
            initial_uncovered_intervals=0,
            terminal_uncovered_intervals=0,
            max_internal_uncovered_gap_intervals=0,
            windows=(_window("forged"),),
        )


def test_selected_window_w_and_w_minus_one_have_exact_zero_grid_behavior() -> None:
    for window_frames in (64, 96):
        hop = window_frames // 4
        candidate = SpectralCandidate(
            window_frames,
            "sum",
            1.0,
            "ridge_integral",
            0.1,
        )
        short_plan = build_window_plan(
            _periodic_video(frames=window_frames - 1, count=2.0),
            candidate,
        )[0]
        exact_plan = build_window_plan(
            _periodic_video(frames=window_frames, count=2.0),
            candidate,
        )[0]
        no_terminal_plan = build_window_plan(
            _periodic_video(frames=window_frames + hop - 1, count=2.0),
            candidate,
        )[0]
        assert short_plan.candidate_window_count == 0
        assert short_plan.windows == ()
        assert exact_plan.candidate_window_count == 1
        assert tuple(item.start for item in exact_plan.windows) == (0,)
        assert no_terminal_plan.candidate_window_count == 1
        assert tuple(item.start for item in no_terminal_plan.windows) == (0,)


def test_transform_applicability_is_source_only_and_stays_fail_closed() -> None:
    config = _config()
    source = _periodic_video()
    altered = RepresentationVideo(
        video_id=source.video_id,
        features=-np.asarray(source.features),
        raw_xy=source.raw_xy,
        joint_mask=source.joint_mask,
        valid_mask=source.valid_mask,
        segments=source.segments,
    )
    first = assess_train337_transform_applicability(
        source,
        _candidate(),
        transform_recipe_sha256=config.train337.transform_recipe.fingerprint,
    )
    second = assess_train337_transform_applicability(
        altered,
        _candidate(),
        transform_recipe_sha256=config.train337.transform_recipe.fingerprint,
    )
    assert first == second
    assert dict(first.flags)["legal_split"] is True
    assert dict(first.flags)["raw_joint_dropout"] is False
    assert first.intersection_applicable is False


def _derangement_receipt(
    video_id: str,
    *,
    seed: int,
    shift: int,
) -> TemporalDerangementReceipt:
    indices = tuple((index + shift) % 128 for index in range(128))
    permutation = SegmentPermutation(
        segment_id="segment-0",
        source_indices=indices,
        source_pose_sha256="a" * 64,
        deranged_pose_sha256=("b" if shift == 1 else "c") * 64,
    )
    return TemporalDerangementReceipt(
        method="pose_pre_encoder_segment_derangement_v1",
        video_id=video_id,
        seed=seed,
        permutations=(permutation,),
        permutation_map_sha256=single_readout._permutation_digest(
            seed,
            (permutation,),
            video_id=video_id,
        ),
    )


def _mechanism(
    video_id: str,
    count: int,
    *,
    insufficient: bool = False,
    zero_available: bool = False,
) -> MechanismEstimate:
    l_confidence, l_peak = 0.5, 0.2
    epi_confidence, epi_peak = 0.1, 0.2
    unavailable = insufficient or zero_available
    represented_count = None if unavailable else count
    candidate_windows = 0 if insufficient else 5
    receipt = _derangement_receipt(video_id, seed=2026, shift=1)
    return MechanismEstimate(
        candidate_id=_candidate().canonical_id,
        primary=_view(
            video_id,
            "L",
            represented_count,
            confidence=0.0 if unavailable else l_confidence,
            peak=0.0 if unavailable else l_peak,
            candidate_window_count=candidate_windows,
        ),
        raw_control=_view(
            video_id,
            "R",
            represented_count,
            confidence=0.0 if unavailable else 0.5,
            peak=0.0 if unavailable else 0.2,
            candidate_window_count=candidate_windows,
        ),
        untrained_control=_view(
            video_id,
            "E0",
            represented_count,
            confidence=0.0 if unavailable else 0.1,
            peak=0.0 if unavailable else 0.2,
            candidate_window_count=candidate_windows,
        ),
        temporal_derangement_control=_view(
            video_id,
            "Epi",
            represented_count,
            confidence=0.0 if unavailable else epi_confidence,
            peak=0.0 if unavailable else epi_peak,
            candidate_window_count=candidate_windows,
        ),
        epi_derangement_receipt=receipt,
        epi_permutation_map_sha256=receipt.permutation_map_sha256,
        learned_encoding_receipt_sha256="1" * 64,
        untrained_encoding_receipt_sha256="2" * 64,
        epi_encoding_receipt_sha256="3" * 64,
    )


def _second_derangement_estimate(
    video_id: str,
    count: int,
    *,
    insufficient: bool = False,
    zero_available: bool = False,
) -> SecondDerangementEstimate:
    unavailable = insufficient or zero_available
    represented_count = None if unavailable else count
    candidate_windows = 0 if insufficient else 5
    receipt = _derangement_receipt(video_id, seed=3407, shift=2)
    return SecondDerangementEstimate(
        candidate_id=_candidate().canonical_id,
        trained_control=_view(
            video_id,
            "Lpi2",
            represented_count,
            confidence=0.0 if unavailable else 0.1,
            peak=0.0 if unavailable else 0.1,
            candidate_window_count=candidate_windows,
        ),
        untrained_control=_view(
            video_id,
            "E0pi2",
            represented_count,
            confidence=0.0 if unavailable else 0.1,
            peak=0.0 if unavailable else 0.2,
            candidate_window_count=candidate_windows,
        ),
        derangement_receipt=receipt,
        trained_encoding_receipt_sha256="4" * 64,
        untrained_encoding_receipt_sha256="5" * 64,
    )


def _applicability(
    video_id: str,
    config,
    *,
    applicable: bool,
) -> Train337TransformApplicability:
    flags = tuple(
        (name, applicable) for name in single_readout._TRAIN337_APPLICABILITY_NAMES
    )
    source_digest = ("6" if applicable else "7") * 64
    recipe_digest = config.train337.transform_recipe.fingerprint
    receipt = single_readout._transform_applicability_digest(
        video_id=video_id,
        candidate_id=_candidate().canonical_id,
        source_geometry_sha256=source_digest,
        transform_recipe_sha256=recipe_digest,
        flags=flags,
    )
    return Train337TransformApplicability(
        video_id=video_id,
        candidate_id=_candidate().canonical_id,
        source_geometry_sha256=source_digest,
        transform_recipe_sha256=recipe_digest,
        flags=flags,
        intersection_applicable=applicable,
        receipt_sha256=receipt,
    )


def _train_record(
    index: int,
    *,
    config,
    source_authority_sha256: str,
    insufficient: bool = False,
    zero_available: bool = False,
) -> Train337VideoAudit:
    video_id = f"train-{index:03d}"
    count = 4 + index % 12
    baseline = _mechanism(
        video_id,
        count,
        insufficient=insufficient,
        zero_available=zero_available,
    )
    unavailable = insufficient or zero_available
    represented_count = None if unavailable else count
    candidate_windows = 0 if insufficient else 5
    same_l = _view(
        video_id,
        "L",
        represented_count,
        confidence=0.0 if unavailable else 0.5,
        peak=0.0 if unavailable else 0.2,
        candidate_window_count=candidate_windows,
    )
    same_r = _view(
        video_id,
        "R",
        represented_count,
        confidence=0.0 if unavailable else 0.5,
        peak=0.0 if unavailable else 0.2,
        candidate_window_count=candidate_windows,
    )
    split_pivot = None if unavailable else 63
    split_left = _view(
        video_id,
        "L",
        None if unavailable else float(count) / 2.0,
        confidence=0.0 if unavailable else 0.5,
        peak=0.0 if unavailable else 0.2,
        candidate_window_count=candidate_windows,
        segment=(
            AuthorizedSegment("segment-0", 0, 128)
            if unavailable
            else AuthorizedSegment("segment-0", 0, 64)
        ),
    )
    split_right = _view(
        video_id,
        "L",
        None if unavailable else float(count) / 2.0,
        confidence=0.0 if unavailable else 0.5,
        peak=0.0 if unavailable else 0.2,
        candidate_window_count=candidate_windows,
        segment=(
            AuthorizedSegment("segment-0", 0, 128)
            if unavailable
            else AuthorizedSegment("segment-0", 63, 128)
        ),
    )
    return build_train337_video_audit(
        video_id=video_id,
        source_authority_sha256=source_authority_sha256,
        transform_recipe_sha256=config.train337.transform_recipe.fingerprint,
        transform_applicability=_applicability(
            video_id,
            config,
            applicable=not unavailable,
        ),
        baseline=baseline,
        reverse_primary=same_l,
        warp_075_primary=same_l,
        warp_125_primary=same_l,
        duplicate_time_primary=same_l,
        legal_split_pivot=split_pivot,
        legal_split_left=split_left,
        legal_split_right=split_right,
        raw_rotation_minus15=same_r,
        raw_rotation_plus15=same_r,
        raw_scale_085=same_r,
        raw_scale_115=same_r,
        raw_joint_dropout=same_r,
        learned_augmentation_a=same_l,
        learned_augmentation_b=same_l,
        static_null_primary=_view(video_id, "L", None, confidence=0.0, peak=0.0),
        second_derangement_controls=_second_derangement_estimate(
            video_id,
            count,
            insufficient=insufficient,
            zero_available=zero_available,
        ),
    )


def _rebuild_train_record(
    record: Train337VideoAudit,
    **overrides,
) -> Train337VideoAudit:
    values = {
        "video_id": record.video_id,
        "source_authority_sha256": (
            record.transform_lineage.source_authority_sha256
        ),
        "transform_recipe_sha256": (
            record.transform_lineage.transform_recipe_sha256
        ),
        "transform_applicability": record.transform_applicability,
        "baseline": record.baseline,
        "reverse_primary": record.reverse_primary,
        "warp_075_primary": record.warp_075_primary,
        "warp_125_primary": record.warp_125_primary,
        "duplicate_time_primary": record.duplicate_time_primary,
        "legal_split_pivot": record.legal_split_pivot,
        "legal_split_left": record.legal_split_left,
        "legal_split_right": record.legal_split_right,
        "raw_rotation_minus15": record.raw_rotation_minus15,
        "raw_rotation_plus15": record.raw_rotation_plus15,
        "raw_scale_085": record.raw_scale_085,
        "raw_scale_115": record.raw_scale_115,
        "raw_joint_dropout": record.raw_joint_dropout,
        "learned_augmentation_a": record.learned_augmentation_a,
        "learned_augmentation_b": record.learned_augmentation_b,
        "static_null_primary": record.static_null_primary,
        "second_derangement_controls": record.second_derangement_controls,
    }
    values.update(overrides)
    return build_train337_video_audit(**values)


def _fake_synthetic_score(
    config,
    candidate: SpectralCandidate,
    *,
    seed: int,
) -> SyntheticCandidateScore:
    winner = candidate == config.candidates[0]
    offset = config.candidates.index(candidate) / 1000.0
    null_scores = tuple(
        NullFamilyScore(family, 64, 0, 0.0457, True)
        for family in single_readout._NULL_FAMILIES
    )
    positive_scores = tuple(
        PositiveFamilyScore(
            family,
            trials,
            trials if winner else 0,
            1.0 if winner else 0.0,
            0.01 + offset,
            1.0 if winner else 0.0,
        )
        for family, trials in single_readout._POSITIVE_FAMILY_COUNTS
    )
    return SyntheticCandidateScore(
        candidate=candidate,
        seed=seed,
        positive_total=118,
        positive_eligible=118 if winner else 0,
        positive_eligible_rate=1.0 if winner else 0.0,
        overall_nmae=0.01 + offset,
        overall_obo=1.0 if winner else 0.0,
        variable_tempo_nmae=0.01 + offset,
        positive_family_scores=positive_scores,
        invariance_agreement=tuple((name, 1.0) for name in single_readout._INVARIANCE_NAMES),
        null_total=512,
        null_false_eligible=0,
        null_cp_ucb=0.0457,
        null_family_scores=null_scores,
        reset_total=64,
        reset_video_total_errors=0,
        abstention_rate=0.0 if winner else 1.0,
        hard_pass=winner,
        failed_gates=() if winner else ("positive_eligible_rate",),
    )


def test_each_positive_family_is_hard_gated_even_when_pool_would_pass() -> None:
    config = _config()
    score = _fake_synthetic_score(
        config,
        config.candidates[0],
        seed=config.synthetic.selector_seed,
    )
    active = score.positive_family_scores[0]
    family_scores = (
        replace(active, eligible=7, eligible_rate=7 / 8, nmae=0.11, obo=7 / 8),
        *score.positive_family_scores[1:],
    )
    assert single_readout._positive_family_gate_failures(
        family_scores,
        config.synthetic,
    ) == (
        "positive_family_eligible_rate:active_support",
        "positive_family_nmae:active_support",
        "positive_family_obo:active_support",
    )


def _synthetic_evidence(monkeypatch: pytest.MonkeyPatch):
    config = _config()
    monkeypatch.setattr(single_readout, "score_synthetic_candidate", _fake_synthetic_score)
    selection = select_synthetic_candidate(config)
    heldout = replay_selected_candidate_heldout(config, selection)
    return config, selection, heldout


def test_selection_and_heldout_are_complete_strict_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, selection, heldout = _synthetic_evidence(monkeypatch)
    assert replay_and_validate_synthetic_selection(config, selection) == config.candidates[0]
    assert replay_and_validate_synthetic_heldout(config, selection, heldout) == (
        config.candidates[0]
    )
    assert len(selection.scores) == 32

    with pytest.raises(ValueError, match="selector seed"):
        replace(selection, selector_seed=42)
    with pytest.raises(ValueError, match="exact 32"):
        replace(selection, scores=selection.scores[:-1])
    with pytest.raises(ValueError, match="dimension order"):
        replace(
            selection,
            mixing_sha256=(
                (34, "f" * 64),
                (34, selection.mixing_sha256[0][1]),
                selection.mixing_sha256[1],
            ),
        )
    altered = replace(selection, config_fingerprint="f" * 64)
    with pytest.raises(SyntheticSelectionFailure, match="deterministic replay"):
        replay_and_validate_synthetic_selection(config, altered)
    altered_nonwinner = replace(
        selection,
        scores=(
            selection.scores[0],
            replace(selection.scores[1], overall_nmae=0.2),
            *selection.scores[2:],
        ),
    )
    with pytest.raises(SyntheticSelectionFailure, match="deterministic replay"):
        replay_and_validate_synthetic_selection(config, altered_nonwinner)

    with pytest.raises(SyntheticSelectionFailure, match="held-out artifact"):
        replay_and_validate_synthetic_heldout(
            config,
            selection,
            replace(heldout, config_fingerprint="f" * 64),
        )
    altered_heldout_score = replace(
        heldout,
        score=replace(heldout.score, overall_nmae=0.02),
    )
    with pytest.raises(SyntheticSelectionFailure, match="held-out artifact"):
        replay_and_validate_synthetic_heldout(
            config,
            selection,
            altered_heldout_score,
        )


def test_heldout_failure_never_reranks_or_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()

    def heldout_trap_score(config, candidate, *, seed):
        score = _fake_synthetic_score(config, candidate, seed=seed)
        if seed != config.synthetic.heldout_seed:
            return score
        if candidate == config.candidates[1]:
            return replace(
                score,
                positive_eligible=score.positive_total,
                positive_eligible_rate=1.0,
                positive_family_scores=tuple(
                    replace(
                        item,
                        eligible=item.trials,
                        eligible_rate=1.0,
                        nmae=0.01,
                        obo=1.0,
                    )
                    for item in score.positive_family_scores
                ),
                overall_obo=1.0,
                abstention_rate=0.0,
                hard_pass=True,
                failed_gates=(),
            )
        return replace(score, hard_pass=False, failed_gates=("heldout_trap",))

    monkeypatch.setattr(single_readout, "score_synthetic_candidate", heldout_trap_score)
    selection = select_synthetic_candidate(config)
    assert selection.selected_candidate_id == config.candidates[0].canonical_id
    assert heldout_trap_score(
        config,
        config.candidates[1],
        seed=config.synthetic.heldout_seed,
    ).hard_pass
    with pytest.raises(SyntheticSelectionFailure, match="no rerank or fallback"):
        replay_selected_candidate_heldout(config, selection)


def _train_authority(config, selection, heldout, video_ids, records):
    return build_fail_closed_train337_authority(
        config,
        canonical_video_ids=video_ids,
        v4e_representation_authority_sha256="a" * 64,
        cycleback_checkpoint_authority_sha256="b" * 64,
        selection=selection,
        heldout=heldout,
        transform_applicability_registry_sha256=(
            transform_applicability_registry_sha256(records)
        ),
        ordered_records_sha256=ordered_train337_records_sha256(records),
    )


def test_caller_summaries_are_hash_fixed_but_can_never_authorize_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, selection, heldout = _synthetic_evidence(monkeypatch)
    video_ids = tuple(f"train-{index:03d}" for index in range(337))
    source_authority_sha256 = train337_source_authority_sha256("a" * 64, "b" * 64)
    records = tuple(
        _train_record(
            index,
            config=config,
            source_authority_sha256=source_authority_sha256,
        )
        for index in range(337)
    )
    authority = _train_authority(config, selection, heldout, video_ids, records)
    report = evaluate_train337_gate(config, selection, heldout, authority, records)
    assert report.numerical_gates_passed is True
    assert report.authorized_for_dev84 is False
    assert report.authorization_blockers == (
        "authoritative_train337_adapter_unwired_fail_closed",
    )
    assert len(report.hash_subset_video_ids) == 64
    mechanism = {item.arm: item for item in report.mechanism}
    assert mechanism["L"].mechanism_pass is True
    assert mechanism["Epi"].mechanism_pass is False
    assert report.to_dict()["action_or_count_targets_consumed"] is False
    metrics = dict(report.metrics)
    assert metrics["transform_applicable_count:baseline"] == 337
    assert metrics["transform_applicable_share:baseline"] == 1.0
    assert report.transform_applicability_registry_sha256 == (
        authority.transform_applicability_registry_sha256
    )
    assert report.ordered_records_sha256 == authority.ordered_records_sha256
    with pytest.raises(ValueError, match="can never authorize"):
        replace(report, authorized_for_dev84=True)

    broken_index = int(report.hash_subset_video_ids[0].split("-")[1])
    broken_record = _rebuild_train_record(
        records[broken_index],
        reverse_primary=_view(
            records[broken_index].video_id,
            "L",
            None,
            confidence=0.0,
            peak=0.0,
        ),
    )
    broken_records = (
        *records[:broken_index],
        broken_record,
        *records[broken_index + 1 :],
    )
    broken_authority = _train_authority(
        config,
        selection,
        heldout,
        video_ids,
        broken_records,
    )
    broken_report = evaluate_train337_gate(
        config,
        selection,
        heldout,
        broken_authority,
        broken_records,
    )
    assert dict(broken_report.metrics)["transform_output_contract_failure_count"] == 1
    assert "transform_output_contract" in broken_report.failed_gates

    selected, digest = hash_fixed_subset(
        reversed(tuple(record.video_id for record in records)),
        seed=2026,
        size=64,
    )
    assert selected == report.hash_subset_video_ids
    assert digest == report.hash_subset_sha256


def test_train337_rejects_registry_order_and_transform_authority_fabrication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, selection, heldout = _synthetic_evidence(monkeypatch)
    video_ids = tuple(f"train-{index:03d}" for index in range(337))
    source_authority_sha256 = train337_source_authority_sha256("a" * 64, "b" * 64)
    records = tuple(
        _train_record(
            index,
            config=config,
            source_authority_sha256=source_authority_sha256,
        )
        for index in range(337)
    )
    authority = _train_authority(config, selection, heldout, video_ids, records)
    reversed_authority = _train_authority(
        config,
        selection,
        heldout,
        tuple(reversed(video_ids)),
        records,
    )
    with pytest.raises(ValueError, match="exact order"):
        evaluate_train337_gate(config, selection, heldout, reversed_authority, records)

    forged_records = (
        _train_record(
            0,
            config=config,
            source_authority_sha256="9" * 64,
        ),
        *records[1:],
    )
    with pytest.raises(ValueError, match="source authority mismatch"):
        evaluate_train337_gate(config, selection, heldout, authority, forged_records)

    changed_first = _rebuild_train_record(
        records[0],
        reverse_primary=_view("train-000", "L", 99),
    )
    with pytest.raises(ValueError, match="ordered record authority mismatch"):
        evaluate_train337_gate(
            config,
            selection,
            heldout,
            authority,
            (changed_first, *records[1:]),
        )

    same_indices_map2 = _derangement_receipt("train-000", seed=3407, shift=1)
    forged_map2 = replace(
        records[0].second_derangement_controls,
        derangement_receipt=same_indices_map2,
    )
    with pytest.raises(ValueError, match="permutations must differ per segment"):
        _rebuild_train_record(
            records[0],
            second_derangement_controls=forged_map2,
        )

    with pytest.raises(ValueError, match="do not bind the audit values"):
        replace(records[0], reverse_primary=_view("train-000", "L", 99))
    with pytest.raises(ValueError, match="video-ID digest mismatch"):
        replace(authority, canonical_video_ids_sha256="f" * 64)
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(authority, v4e_representation_authority_sha256="A" * 64)


def test_train337_coverage_uses_all_337_as_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, selection, heldout = _synthetic_evidence(monkeypatch)
    video_ids = tuple(f"train-{index:03d}" for index in range(337))
    source_authority_sha256 = train337_source_authority_sha256("a" * 64, "b" * 64)
    records = tuple(
        _train_record(
            index,
            config=config,
            source_authority_sha256=source_authority_sha256,
            insufficient=index < 20,
            zero_available=20 <= index < 40,
        )
        for index in range(337)
    )
    authority = _train_authority(config, selection, heldout, video_ids, records)
    assert records[0].baseline.primary.segments[0].candidate_window_count == 0
    assert records[20].baseline.primary.segments[0].candidate_window_count == 5
    assert records[20].baseline.primary.segments[0].available_window_count == 0
    report = evaluate_train337_gate(config, selection, heldout, authority, records)
    metrics = dict(report.metrics)
    assert metrics["canonical_video_count"] == 337
    assert metrics["one_segment_sufficient_video_count"] == 297
    assert metrics["one_segment_sufficient_share"] == pytest.approx(297 / 337)
    assert "one_segment_sufficient_share" in report.failed_gates
    assert "canonical_one_segment_eligible_share" in report.failed_gates


def test_train337_transform_intersection_below_64_never_forms_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, selection, heldout = _synthetic_evidence(monkeypatch)
    video_ids = tuple(f"train-{index:03d}" for index in range(337))
    source_authority_sha256 = train337_source_authority_sha256("a" * 64, "b" * 64)
    records = tuple(
        _train_record(
            index,
            config=config,
            source_authority_sha256=source_authority_sha256,
            insufficient=index >= 63,
        )
        for index in range(337)
    )
    authority = _train_authority(config, selection, heldout, video_ids, records)
    report = evaluate_train337_gate(config, selection, heldout, authority, records)
    metrics = dict(report.metrics)
    assert metrics["transform_intersection_applicable_video_count"] == 63
    assert report.hash_subset_video_ids == ()
    assert "transform_intersection_applicable_video_count" in report.failed_gates
    assert np.isinf(metrics["reverse_relative_error_median"])


def test_server_launcher_is_fail_closed_until_authoritative_adapters_merge() -> None:
    launcher = REPOSITORY / "scripts/server/run_segment_local_spectral_single_v1.py"
    text = launcher.read_text(encoding="utf-8")
    assert "return 3" in text
    assert "targets_mounted" in text
    assert "test105_authorized" in text
