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
    SegmentEstimate,
    SegmentPermutation,
    SpectralCandidate,
    SyntheticCandidateScore,
    SyntheticSelectionFailure,
    TemporalDerangementReceipt,
    Train337VideoAudit,
    ViewEstimate,
    WindowEstimate,
    build_fail_closed_train337_authority,
    build_full_segment_encoding_receipt,
    build_preencoder_segment_derangement,
    build_train337_video_audit,
    build_window_plan,
    estimate_mechanism_views,
    estimate_representation,
    evaluate_train337_gate,
    hash_fixed_subset,
    load_segment_local_spectral_config,
    replay_and_validate_synthetic_heldout,
    replay_and_validate_synthetic_selection,
    replay_selected_candidate_heldout,
    select_synthetic_candidate,
    synthetic_case_plan,
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
        any(window.start >= segment.start and window.stop <= segment.stop for segment in video.segments)
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
    role = "frozen_untrained_initialization" if view == "E0" else "trained_cycleback"
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
    assert _config().synthetic.positive_family_gate_policy == (
        "pooled_all_generation_truth_positives_only_no_family_thresholds"
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
        assert phase[-1] / (2.0 * np.pi) == pytest.approx(count, abs=1e-12)
        assert np.all(np.diff(phase) >= 0.0)


def _window(video_id: str) -> WindowEstimate:
    return WindowEstimate(
        key=("segment-0", 0, 64),
        center=31.5,
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
    count: int | None,
    *,
    confidence: float = 0.5,
    peak: float = 0.2,
    candidate_window_count: int = 5,
) -> ViewEstimate:
    segment = AuthorizedSegment("segment-0", 0, 128)
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
    window = _window(video_id)
    segment_estimate = SegmentEstimate(
        segment,
        candidate_window_count,
        1,
        1,
        True,
        "eligible",
        None,
        float(count),
        (window,),
    )
    return ViewEstimate(
        video_id,
        view,  # type: ignore[arg-type]
        _candidate().canonical_id,
        "eligible",
        None,
        float(count),
        count,
        confidence,
        peak,
        (segment_estimate,),
    )


def _mechanism(
    video_id: str,
    count: int,
    *,
    shuffled: bool,
    insufficient: bool = False,
    zero_available: bool = False,
) -> MechanismEstimate:
    if shuffled:
        l_confidence, l_peak = 0.1, 0.1
        epi_confidence, epi_peak = 0.1, 0.2
        suffix = "b"
    else:
        l_confidence, l_peak = 0.5, 0.2
        epi_confidence, epi_peak = 0.1, 0.2
        suffix = "a"
    unavailable = insufficient or zero_available
    represented_count = None if unavailable else count
    candidate_windows = 0 if insufficient else 5
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
        epi_permutation_map_sha256=suffix * 64,
        learned_encoding_receipt_sha256="1" * 64,
        untrained_encoding_receipt_sha256="2" * 64,
        epi_encoding_receipt_sha256=("3" if not shuffled else "4") * 64,
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
        shuffled=False,
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
    split_counts = (
        (None, None)
        if insufficient or zero_available
        else (float(count) / 2.0, float(count) / 2.0)
    )
    return build_train337_video_audit(
        video_id=video_id,
        source_authority_sha256=source_authority_sha256,
        transform_recipe_sha256=config.train337.transform_recipe.fingerprint,
        baseline=baseline,
        reverse_primary=same_l,
        warp_075_primary=same_l,
        warp_125_primary=same_l,
        duplicate_time_primary=same_l,
        legal_split_part_float_counts=split_counts,
        raw_rotation_minus15=same_r,
        raw_rotation_plus15=same_r,
        raw_scale_085=same_r,
        raw_scale_115=same_r,
        raw_joint_dropout=same_r,
        learned_augmentation_a=same_l,
        learned_augmentation_b=same_l,
        static_null_primary=_view(video_id, "L", None, confidence=0.0, peak=0.0),
        second_derangement_shuffle=_mechanism(
            video_id,
            count,
            shuffled=True,
            insufficient=insufficient,
            zero_available=zero_available,
        ),
    )


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
    return SyntheticCandidateScore(
        candidate=candidate,
        seed=seed,
        positive_total=118,
        positive_eligible=118 if winner else 0,
        positive_eligible_rate=1.0 if winner else 0.0,
        overall_nmae=0.01 + offset,
        overall_obo=1.0 if winner else 0.0,
        variable_tempo_nmae=0.01 + offset,
        positive_family_scores=(
            PositiveFamilyScore("constant_tempo_count", 26, 26, 1.0, 0.01, 1.0),
        ),
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

    altered_reports = (
        replace(selection, selector_seed=42),
        replace(selection, scores=selection.scores[:-1]),
        replace(selection, mixing_sha256=((34, "f" * 64), (512, "e" * 64))),
        replace(selection, selected_candidate_id=config.candidates[1].canonical_id),
        replace(
            selection,
            scores=(replace(selection.scores[0], overall_nmae=0.2), *selection.scores[1:]),
        ),
    )
    for altered in altered_reports:
        with pytest.raises(SyntheticSelectionFailure, match="deterministic replay"):
            replay_and_validate_synthetic_selection(config, altered)

    with pytest.raises(SyntheticSelectionFailure, match="held-out artifact"):
        replay_and_validate_synthetic_heldout(
            config,
            selection,
            replace(heldout, selected_candidate_id=config.candidates[1].canonical_id),
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


def _train_authority(config, selection, heldout, video_ids):
    return build_fail_closed_train337_authority(
        config,
        canonical_video_ids=video_ids,
        v4e_representation_authority_sha256="a" * 64,
        cycleback_checkpoint_authority_sha256="b" * 64,
        selection=selection,
        heldout=heldout,
    )


def test_train337_gate_is_hash_fixed_label_free_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, selection, heldout = _synthetic_evidence(monkeypatch)
    video_ids = tuple(f"train-{index:03d}" for index in range(337))
    authority = _train_authority(config, selection, heldout, video_ids)
    records = tuple(
        _train_record(
            index,
            config=config,
            source_authority_sha256=authority.source_authority_sha256,
        )
        for index in range(337)
    )
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
    authority = _train_authority(config, selection, heldout, video_ids)
    records = tuple(
        _train_record(
            index,
            config=config,
            source_authority_sha256=authority.source_authority_sha256,
        )
        for index in range(337)
    )
    reversed_authority = _train_authority(
        config,
        selection,
        heldout,
        tuple(reversed(video_ids)),
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
    authority = _train_authority(config, selection, heldout, video_ids)
    records = tuple(
        _train_record(
            index,
            config=config,
            source_authority_sha256=authority.source_authority_sha256,
            insufficient=index < 20,
            zero_available=20 <= index < 40,
        )
        for index in range(337)
    )
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


def test_server_launcher_is_fail_closed_until_authoritative_adapters_merge() -> None:
    launcher = REPOSITORY / "scripts/server/run_segment_local_spectral_single_v1.py"
    text = launcher.read_text(encoding="utf-8")
    assert "return 3" in text
    assert "targets_mounted" in text
    assert "test105_authorized" in text
