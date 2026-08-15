from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pams.warp_phase import data, packing
from pams.warp_phase.types import EligibilityInput, FeatureShard


def _motion(length: int) -> np.ndarray:
    motion = np.zeros((length, 17, 3), dtype="<f4")
    motion[:, :, 0] = np.arange(17, dtype=np.float32)[None, :] / np.float32(20.0)
    motion[:, :, 1] = np.float32(0.5)
    motion[:, :, 2] = np.float32(0.9)
    motion[:, 11, :2] = (0.0, 0.0)
    motion[:, 12, :2] = (2.0, 0.0)
    motion[:, 5, :2] = (0.0, 1.0)
    motion[:, 6, :2] = (2.0, 1.0)
    return motion


def _feature(clocks: np.ndarray, *, key: str = "c" * 64, slot: int = 0) -> FeatureShard:
    return FeatureShard(
        motion=_motion(len(clocks)),
        person_mask=True,
        frame_mask=np.ones(len(clocks), dtype="|u1"),
        sampled_frame_indices=np.asarray(clocks, dtype="<i8"),
        source_length=int(clocks[-1]) + 1,
        opaque_sample_key=key,
        local_person_slot=slot,
    )


def test_duplicate_clock_agreement_retains_first_without_averaging() -> None:
    clocks = np.insert(np.arange(65, dtype="<i8"), 11, 10)
    feature = _feature(clocks)
    feature.motion[11] = feature.motion[10]
    collapsed = data.collapse_duplicate_clocks(feature)
    assert collapsed.sampled_frame_indices.tolist() == list(range(65))
    assert collapsed.conflict_clocks == ()
    assert collapsed.motion.shape == (65, 17, 3)
    assert collapsed.normalization_scale == pytest.approx(1.0)


def test_conflicting_duplicate_is_invalidated_and_breaks_the_cell() -> None:
    clocks = np.insert(np.arange(65, dtype="<i8"), 11, 10)
    feature = _feature(clocks)
    feature.motion[11, 0, 0] += np.float32(0.1)
    collapsed = data.collapse_duplicate_clocks(feature)
    assert 10 not in collapsed.sampled_frame_indices
    assert collapsed.conflict_clocks == (10,)
    gap = int(np.flatnonzero(collapsed.sampled_frame_indices == 9)[0])
    assert collapsed.sampled_frame_indices[gap + 1] == 11
    assert collapsed.cell_mask[gap] == 0


def test_eligibility_is_count_blind_and_ambiguous_slots_are_always_excluded() -> None:
    feature = _feature(np.arange(65, dtype="<i8"))
    assessment = data.assess_count_blind_eligibility(
        EligibilityInput(
            feature=feature,
            association_ambiguous=True,
            association_one_to_one=True,
            pose_coverage=1.0,
        )
    )
    assert not assessment.decision.eligible
    assert assessment.decision.reasons == ("association_ambiguous",)
    assert assessment.decision.retained_distinct_clocks == 65
    assert assessment.decision.valid_adjacent_cells == 64
    assert assessment.decision.feature_frame_coverage == 1.0


def test_eligibility_rejects_low_pose_coverage_without_any_label_input() -> None:
    assessment = data.assess_count_blind_eligibility(
        EligibilityInput(
            feature=_feature(np.arange(65, dtype="<i8")),
            association_ambiguous=False,
            association_one_to_one=True,
            pose_coverage=0.79,
        )
    )
    assert assessment.decision.reasons == ("pose_coverage_below_minimum",)


def test_person_invalid_slot_returns_an_exclusion_receipt() -> None:
    feature = _feature(np.arange(65, dtype="<i8"))
    invalid = FeatureShard(
        motion=feature.motion,
        person_mask=False,
        frame_mask=feature.frame_mask,
        sampled_frame_indices=feature.sampled_frame_indices,
        source_length=feature.source_length,
        opaque_sample_key=feature.opaque_sample_key,
        local_person_slot=feature.local_person_slot,
    )
    assessment = data.assess_count_blind_eligibility(
        EligibilityInput(
            feature=invalid,
            association_ambiguous=False,
            association_one_to_one=True,
            pose_coverage=1.0,
        )
    )
    assert assessment.track is None
    assert assessment.decision.reasons == (
        "person_invalid",
        "insufficient_valid_adjacent_cells",
        "insufficient_feature_frame_coverage",
    )


def test_collate_keeps_opaque_routing_values_outside_arrays() -> None:
    first = data.collapse_duplicate_clocks(
        _feature(np.arange(65, dtype="<i8"), key="d" * 64, slot=1)
    )
    second = data.collapse_duplicate_clocks(
        _feature(np.arange(70, dtype="<i8"), key="e" * 64, slot=3)
    )
    batch = data.collate_tracks((first, second))
    assert batch.motion.shape == (2, 70, 17, 3)
    assert batch.padding_mask[0].sum() == 65
    assert batch.padding_mask[1].sum() == 70
    assert batch.opaque_sample_keys == ("d" * 64, "e" * 64)
    assert batch.local_person_slots == (1, 3)
    assert all(array.dtype.kind != "S" for array in (batch.motion, batch.sampled_frame_indices))


def test_malicious_zip_member_is_rejected_before_numpy_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / f"{'f' * 64}.0.npz"
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("../payload.pkl", b"not a pickle")
    path.write_bytes(payload.getvalue())
    called = False

    def forbidden_numpy_load(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("np.load must not run before ZIP inspection passes")

    monkeypatch.setattr(data.np, "load", forbidden_numpy_load)
    with pytest.raises(data.FeatureArchiveError):
        data.load_feature_shard(path)
    assert not called


def test_feature_load_calls_numpy_with_allow_pickle_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = packing.create_pilot_layout(tmp_path / "run")
    source = _feature(np.rint(np.linspace(0, 399, 320)).astype("<i8"))
    source = FeatureShard(
        motion=source.motion,
        person_mask=source.person_mask,
        frame_mask=source.frame_mask,
        sampled_frame_indices=source.sampled_frame_indices,
        source_length=400,
        opaque_sample_key=source.opaque_sample_key,
        local_person_slot=source.local_person_slot,
    )
    receipt = packing.write_feature_shard(layout, "train", source)
    observed: list[bool] = []
    original_load = np.load

    def guarded_load(*args: object, **kwargs: object) -> object:
        observed.append(kwargs.get("allow_pickle") is False)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(data.np, "load", guarded_load)
    loaded = data.load_feature_shard(receipt.path)
    assert loaded.motion.shape == (320, 17, 3)
    assert observed == [True]


def test_feature_split_reader_refuses_vault_root(tmp_path: Path) -> None:
    vault = tmp_path / "vault"
    (vault / "train").mkdir(parents=True)
    with pytest.raises(data.FeatureDataError, match="features"):
        data.load_feature_split(vault, "train")


def test_schema_fixture_feature_view_hides_annotation_and_vault() -> None:
    feature_view = data.load_feature_schema_view(packing.schema_fixture_bytes())
    assert set(feature_view) == set(packing.FEATURE_FIELD_NAMES)
    assert "annotation" not in feature_view
    assert "evaluator_vault" not in feature_view
