from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pams.warp_phase.selector import (
    NormalizedTrack,
    SelectorError,
    canonical_parent_starts,
    draw_track_perturbations,
    freeze_track_perturbations,
    make_weak_view,
    normalize_coco17,
    select_support_period,
    select_track_period,
)


def _periodic_coordinates(period: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clocks = np.arange(128, dtype=np.float64)
    coordinates = np.zeros((128, 17, 2), dtype=np.float64)
    for joint in range(17):
        phase = 2.0 * np.pi * clocks / period + joint * 0.1
        coordinates[:, joint, 0] = np.sin(phase)
        coordinates[:, joint, 1] = np.cos(phase)
    return clocks, coordinates, np.ones((128, 17), dtype=np.bool_)


def _selector_unit_pose(clock_count: int, source_period: int) -> tuple[np.ndarray, ...]:
    clocks = np.arange(clock_count, dtype="<i8")
    pose = np.zeros((clock_count, 17, 3), dtype="<f8")
    joint_mask = np.ones((clock_count, 17), dtype="|u1")
    pose[:, :, 2] = np.float64(0.9)
    pose[:, 11, :2] = (-0.5, 0.0)
    pose[:, 12, :2] = (0.5, 0.0)
    pose[:, 5, :2] = (-0.5, 2.0)
    pose[:, 6, :2] = (0.5, 2.0)
    for clock_index in range(clock_count):
        for joint in range(17):
            if joint in {5, 6, 11, 12}:
                continue
            base_x = np.float64(joint % 5 - 2) / np.float64(4.0)
            base_y = np.float64(joint // 5) / np.float64(4.0)
            theta = (
                np.float64(2.0)
                * np.pi
                * np.float64(clocks[clock_index])
                / np.float64(source_period)
                + np.float64(joint) * np.float64(0.1)
            )
            pose[clock_index, joint, 0] = base_x + np.float64(0.2) * np.sin(theta)
            pose[clock_index, joint, 1] = base_y + np.float64(0.2) * np.cos(theta)
    return clocks, pose, joint_mask


@pytest.mark.parametrize("period", [4.0, 20.0, 63.0])
def test_selector_recovers_off_bin_and_endpoint_periods(period: float) -> None:
    clocks, coordinates, mask = _periodic_coordinates(period)
    selection = select_support_period(clocks, coordinates, mask)
    assert selection.period == int(period)
    assert selection.fft_power.shape == (127,)
    assert selection.fft_grid[0] == 0.0
    assert selection.fft_grid[-1] == 127.0


@pytest.mark.parametrize(
    ("fixture_id", "clock_count", "source_period", "p_max", "selected_period"),
    (
        ("offbin_delta1_p20", 256, 20, 127, 20),
        ("lower_endpoint_p4", 64, 4, 31, 4),
        ("upper_endpoint_span126_p63", 127, 63, 63, 63),
    ),
)
def test_amendment_001_exact_label_free_selector_unit_fixtures(
    fixture_id: str,
    clock_count: int,
    source_period: int,
    p_max: int,
    selected_period: int,
) -> None:
    clocks, pose, joint_mask = _selector_unit_pose(clock_count, source_period)
    normalized = normalize_coco17(pose, joint_mask, clocks)

    assert normalized.scale == 2.0
    np.testing.assert_array_equal(normalized.coordinates, pose[:, :, :2] / np.float64(2.0))
    assert np.all(normalized.joint_valid)
    assert np.all(normalized.feature_frame_valid)
    velocity = np.diff(normalized.coordinates.reshape(clock_count, 34), axis=0)
    velocity /= np.diff(clocks.astype(np.float64))[:, None]
    assert velocity.shape == (clock_count - 1, 34)
    assert np.isfinite(velocity).all()
    np.testing.assert_array_equal(velocity[:, 10:14], 0.0)
    np.testing.assert_array_equal(velocity[:, 22:26], 0.0)

    selection = select_support_period(
        normalized.clocks,
        normalized.coordinates,
        normalized.joint_valid,
    )
    span = float(clocks[-1] - clocks[0])
    delta = span / 255.0
    assert selection.candidates[0] == 4
    assert selection.candidates[-1] == p_max
    assert selection.period == selected_period
    assert selection.fft_grid[0] == 0.0
    assert selection.fft_grid[-1] == float(clock_count - 1)
    assert np.all(selection.fft_mask)

    if fixture_id == "offbin_delta1_p20":
        exact_grid = np.arange(256, dtype=np.float64)
        assert delta == 1.0
        assert selection.fft_grid.tobytes(order="C") == exact_grid.tobytes(order="C")
        continuous_bin = np.float64(256.0) * np.float64(delta) / np.float64(20.0)
        assert float(continuous_bin) == 12.8
        interpolated = (
            (np.float64(13.0) - continuous_bin) * selection.fft_power[11]
            + (continuous_bin - np.float64(12.0)) * selection.fft_power[12]
        )
        assert selection.harmonic_power[16, 0] == interpolated
    elif fixture_id == "lower_endpoint_p4":
        assert p_max > 4
        assert selection.score_valid[0]
        assert selection.score_by_period[0] >= selection.score_by_period[1]
        assert selection.local_maximum[0]
    else:
        assert span == 126.0
        assert p_max == 63
        assert selection.score_valid[-1]
        assert selection.score_by_period[-1] >= selection.score_by_period[-2]
        assert selection.local_maximum[-1]


def test_selector_unit_fixture_namespace_is_absent_from_stochastic_generator() -> None:
    stochastic_generator = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "experiments"
        / "generate_warp_phase_fixture_pack.py"
    ).read_text(encoding="utf-8")
    assert "selector-unit-fixtures-v1" not in stochastic_generator
    for fixture_id in (
        "offbin_delta1_p20",
        "lower_endpoint_p4",
        "upper_endpoint_span126_p63",
    ):
        assert fixture_id not in stochastic_generator


def test_fixture_id10_failure_preserves_computed_diagnostics() -> None:
    fixture_root = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "warp_phase_pilot_v1"
        / "audit"
        / "gate1"
        / "candidate-20260815-sol"
        / "pack"
    )
    pose = np.load(fixture_root / "pose.npy", mmap_mode="r", allow_pickle=False)
    joint_mask = np.load(
        fixture_root / "joint_mask.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    q = np.load(fixture_root / "q.npy", allow_pickle=False)
    expected_selected = np.load(
        fixture_root / "expected_selected_period.npy",
        allow_pickle=False,
    )
    normalized = normalize_coco17(pose[10], joint_mask[10], q.astype(np.int64))

    selection = select_support_period(
        normalized.clocks,
        normalized.coordinates,
        normalized.joint_valid,
    )

    assert int(expected_selected[10]) == -1
    assert selection.period is None
    assert selection.score is None
    assert not np.any(selection.local_maximum)
    assert np.any(selection.score_valid)
    assert np.any(selection.fft_power != 0.0)
    assert np.any(selection.harmonic_power != 0.0)
    assert np.any(np.isfinite(selection.acf))
    assert np.any(np.isfinite(selection.score_by_period))


def test_normalization_uses_coco_hips_and_shoulders() -> None:
    pose = np.zeros((64, 17, 3), dtype=np.float64)
    pose[:, :, 2] = 0.9
    pose[:, 11, :2] = (-1.0, 0.0)
    pose[:, 12, :2] = (1.0, 0.0)
    pose[:, 5, :2] = (-1.0, 2.0)
    pose[:, 6, :2] = (1.0, 2.0)
    for joint in range(17):
        if joint not in {5, 6, 11, 12}:
            pose[:, joint, :2] = (float(joint) / 17.0, 1.0)
    normalized = normalize_coco17(
        pose,
        np.ones((64, 17), dtype=np.bool_),
        np.arange(64, dtype=np.int64),
    )
    assert normalized.scale == 2.0
    np.testing.assert_allclose(
        normalized.coordinates[:, 11, :],
        np.tile(np.asarray((-0.5, 0.0)), (64, 1)),
    )
    assert np.all(normalized.feature_frame_valid)
    assert normalized.model_pose.dtype == np.float32


def test_normalization_fails_without_scale() -> None:
    pose = np.zeros((64, 17, 3), dtype=np.float64)
    pose[:, :, 2] = 0.9
    mask = np.ones((64, 17), dtype=np.bool_)
    mask[:, 5:7] = False
    with pytest.raises(SelectorError, match="scale"):
        normalize_coco17(pose, mask, np.arange(64, dtype=np.int64))


def test_weak_view_consumes_masks_and_restores_lowest_indices() -> None:
    coordinates = np.zeros((1, 17, 2), dtype=np.float64)
    mask = np.ones((1, 17), dtype=np.bool_)
    jitter = np.full((1, 17, 2), 0.1, dtype=np.float64)
    dropout = np.zeros((1, 17), dtype=np.float64)
    weak_coordinates, weak_mask = make_weak_view(coordinates, mask, jitter, dropout)
    assert np.flatnonzero(weak_mask[0]).tolist() == list(range(8))
    np.testing.assert_allclose(weak_coordinates[0, :8], 0.03)
    np.testing.assert_array_equal(weak_coordinates[0, 8:], 0.0)


def test_parent_starts_append_final_support_without_duplicate() -> None:
    assert canonical_parent_starts(64) == (0,)
    assert canonical_parent_starts(128) == (0,)
    assert canonical_parent_starts(200) == (0, 64, 72)
    with pytest.raises(SelectorError):
        canonical_parent_starts(63)


def test_track_evaluation_requires_receipt_bound_pre_draws() -> None:
    clocks, coordinates, joint_valid = _periodic_coordinates(20.0)
    integer_clocks = clocks.astype(np.int64)
    model_pose = np.zeros((128, 17, 3), dtype=np.float32)
    model_pose[:, :, :2] = coordinates.astype(np.float32)
    model_pose[:, :, 2] = np.float32(0.9)
    track = NormalizedTrack(
        clocks=integer_clocks,
        coordinates=coordinates,
        joint_valid=joint_valid,
        feature_frame_valid=np.ones(128, dtype=np.bool_),
        model_pose=model_pose,
        scale=1.0,
    )
    rng = np.random.Generator(np.random.PCG64(20270815))
    pre_drawn = draw_track_perturbations(track, rng)

    assert len(pre_drawn.parents) == 1
    assert not pre_drawn.parents[0].jitter_raw.flags.writeable
    assert not pre_drawn.parents[0].dropout_raw.flags.writeable
    with pytest.raises(SelectorError, match="frozen pre-drawn"):
        select_track_period(track, pre_drawn)  # type: ignore[arg-type]
    with pytest.raises(SelectorError, match="lowercase SHA-256"):
        freeze_track_perturbations(pre_drawn, order_receipt_sha256="not-frozen")

    frozen = freeze_track_perturbations(
        pre_drawn,
        order_receipt_sha256="a" * 64,
    )
    with pytest.raises(SelectorError, match="no canonical parent"):
        select_track_period(track, frozen)
