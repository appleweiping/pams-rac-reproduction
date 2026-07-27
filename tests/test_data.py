from pathlib import Path

import numpy as np
import pytest

from pams.data import (
    TrainingPoseDataset,
    UCFRepManifest,
    UCFRepRecord,
    deterministic_stratified_split,
    load_pose_cache,
    load_ucfrep_manifest,
    longest_valid_span,
    per_frame_minmax,
    pose_cache_path,
    preprocess_pose_sequence,
    save_ucfrep_manifest,
    split_ucfrep_pose_train_dev,
    split_ucfrep_train_dev,
    uniform_resample,
    write_pose_cache,
)
from pams.types import PoseSequence


def _record(index: int, *, split: str = "train") -> UCFRepRecord:
    return UCFRepRecord(
        video_id=f"video-{index:04d}",
        video_path=f"videos/video-{index:04d}.mp4",
        split=split,
        action=f"action-{index % 7}",
        count=index % 45 + 1,
        video_sha256=f"{index:064x}",
    )


def test_per_frame_minmax_and_missing_frames() -> None:
    xyz = np.arange(3 * 33 * 3, dtype=np.float32).reshape(3, 33, 3)
    xyz[1] = 5
    xyz[2] = np.nan
    normalized = per_frame_minmax(xyz, [True, True, False])
    assert normalized[0].min() == 0
    assert normalized[0].max() == 1
    assert np.all(normalized[1] == 0)
    assert np.all(normalized[2] == 0)


def test_uniform_resample_preserves_gap_mask_and_zeros() -> None:
    xyz = np.ones((3, 33, 3), dtype=np.float32)
    xyz[1] = 9
    output, mask = uniform_resample(xyz, np.asarray([True, False, True]), target_frames=5)
    assert output.shape == (5, 33, 3)
    np.testing.assert_array_equal(mask, [True, False, False, False, True])
    assert np.all(output[~mask] == 0)


def test_longest_valid_span_prefers_earliest_tie() -> None:
    span = longest_valid_span([False, True, True, False, True, True])
    assert span == slice(1, 3)
    with pytest.raises(ValueError, match="no valid"):
        longest_valid_span([False, False])


def test_preprocess_pose_produces_256_normalized_frames() -> None:
    rng = np.random.default_rng(1)
    source = PoseSequence(
        "pose",
        25,
        rng.normal(size=(40, 33, 3)),
        np.ones(40, dtype=bool),
    )
    output = preprocess_pose_sequence(source)
    assert output.xyz.shape == (256, 33, 3)
    assert output.xyz.min() >= 0
    assert output.xyz.max() <= 1
    assert output.fps == pytest.approx(25 * 255 / 39)


@pytest.mark.parametrize("suffix", [".json", ".csv"])
def test_manifest_round_trip_strict_schema(tmp_path: Path, suffix: str) -> None:
    manifest = UCFRepManifest(
        protocol="ucfrep_526",
        records=(_record(1), _record(2, split="test")),
    )
    path = tmp_path / f"manifest{suffix}"
    save_ucfrep_manifest(manifest, path)
    loaded = load_ucfrep_manifest(path, protocol="ucfrep_526", validate_exact=False)
    assert loaded == manifest
    assert len(loaded.fingerprint) == 64


def test_manifest_fingerprint_ignores_local_mount_path() -> None:
    original = _record(1)
    moved = UCFRepRecord(
        video_id=original.video_id,
        video_path="D:/another/mount/video.mp4",
        split=original.split,
        action=original.action,
        count=original.count,
        video_sha256=original.video_sha256,
    )
    first = UCFRepManifest(protocol="ucfrep_526", records=(original,))
    second = UCFRepManifest(protocol="ucfrep_526", records=(moved,))
    assert first.fingerprint == second.fingerprint


def test_exact_standard_protocol_split_validation() -> None:
    records = tuple(_record(index) for index in range(421)) + tuple(
        _record(421 + index, split="test") for index in range(105)
    )
    manifest = UCFRepManifest("ucfrep_526", records)
    manifest.validate_exact_official_splits()
    bad = UCFRepManifest("ucfrep_526", records[:-1])
    with pytest.raises(ValueError, match="exact split"):
        bad.validate_exact_official_splits()


def test_manifest_rejects_duplicate_identity() -> None:
    duplicate = _record(1, split="test")
    with pytest.raises(ValueError, match="duplicate video_id"):
        UCFRepManifest("ucfrep_526", (_record(1), duplicate))


def test_stratified_split_is_exact_disjoint_and_deterministic() -> None:
    records = tuple(_record(index) for index in range(421))
    first_train, first_dev = split_ucfrep_train_dev(records)
    next_train, next_dev = split_ucfrep_train_dev(records)
    assert (len(first_train), len(first_dev)) == (337, 84)
    assert [row.video_id for row in first_dev] == [row.video_id for row in next_dev]
    assert {row.video_id for row in first_train}.isdisjoint(row.video_id for row in first_dev)
    assert all(row.split == "dev" for row in first_dev)

    pose_train, pose_dev = split_ucfrep_pose_train_dev(records[:89])
    assert (len(pose_train), len(pose_dev)) == (71, 18)


def test_generic_stratifier_handles_singletons_and_exact_size() -> None:
    rows = tuple(
        UCFRepRecord(
            video_id=f"id-{index}",
            video_path=f"{index}.mp4",
            split="train",
            action=f"unique-{index}",
            count=index + 1,
        )
        for index in range(10)
    )
    train, dev = deterministic_stratified_split(rows, dev_size=3, seed=5)
    assert (len(train), len(dev)) == (7, 3)


def test_pose_cache_hash_guards_and_training_dataset_no_label_leak(
    tmp_path: Path,
) -> None:
    config_hash = "a" * 64
    video_hash = "b" * 64
    sequence = PoseSequence(
        "cached",
        30,
        np.ones((8, 33, 3), dtype=np.float32),
        np.ones(8, dtype=bool),
    )
    cache_path = pose_cache_path(tmp_path, sequence.video_id)
    write_pose_cache(
        cache_path,
        sequence,
        video_sha256=video_hash,
        config_sha256=config_hash,
    )
    loaded, metadata = load_pose_cache(
        cache_path,
        expected_video_sha256=video_hash,
        expected_config_sha256=config_hash,
    )
    assert loaded.video_id == "cached"
    assert metadata.frames == 8
    with pytest.raises(ValueError, match="config"):
        load_pose_cache(cache_path, expected_config_sha256="c" * 64)

    record = UCFRepRecord(
        video_id="cached",
        video_path="cached.mp4",
        split="train",
        action="jump",
        count=999,
        video_sha256=video_hash,
    )
    dataset = TrainingPoseDataset([record], cache_dir=tmp_path, config_sha256=config_hash)
    assert dataset[0].video_id == "cached"
    assert not hasattr(dataset._items[0], "count")
    assert not hasattr(dataset._items[0], "action")


def test_training_dataset_rejects_test_and_dev_by_default(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sealed"):
        TrainingPoseDataset(
            [_record(1, split="test")],
            cache_dir=tmp_path,
            config_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="disallowed"):
        TrainingPoseDataset(
            [UCFRepRecord("dev", "dev.mp4", "dev", "jump", 5)],
            cache_dir=tmp_path,
            config_sha256="a" * 64,
        )
