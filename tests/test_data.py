import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import pams.data as data_module
from pams.data import (
    DevTargetManifest,
    DevTargetRecord,
    LabelFreeProtocolInputs,
    PoseInputCommitment,
    PoseInputManifest,
    TrainingPoseDataset,
    UCFRepManifest,
    UCFRepRecord,
    UnlabeledVideoRecord,
    assert_split_disjoint,
    deterministic_stratified_split,
    load_dev_target_manifest,
    load_pose_cache,
    load_pose_input_manifest,
    load_ucfrep_manifest,
    longest_valid_span,
    per_frame_minmax,
    pose_cache_path,
    pose_input_identity_sha256,
    preprocess_pose_sequence,
    save_ucfrep_manifest,
    split_ucfrep_pose_train_dev,
    split_ucfrep_train_dev,
    uniform_resample,
    write_pose_cache,
)
from pams.types import PoseSequence

REPOSITORY = Path(__file__).parents[1]


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


def test_preprocess_marks_zero_span_frames_invalid() -> None:
    xyz = np.zeros((3, 33, 3), dtype=np.float32)
    xyz[0, :, 0] = np.linspace(0.0, 1.0, 33, dtype=np.float32)
    xyz[1] = 5.0
    xyz[2, :, 1] = np.linspace(0.0, 1.0, 33, dtype=np.float32)
    sequence = PoseSequence(
        video_id="zero-span",
        fps=30.0,
        xyz=xyz,
        valid_mask=np.ones(3, dtype=np.bool_),
    )

    processed = preprocess_pose_sequence(sequence, target_frames=3)

    assert processed.valid_mask.tolist() == [True, False, True]
    assert np.count_nonzero(processed.xyz[1]) == 0


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
        rng.normal(size=(40, 33, 3)).astype(np.float32),
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


@pytest.mark.parametrize("suffix", [".json", ".csv"])
def test_official_segment_manifest_round_trip_and_fingerprint(
    tmp_path: Path,
    suffix: str,
) -> None:
    annotation = "c" * 64
    records = tuple(
        replace(
            _record(index, split="train" if index == 1 else "test"),
            annotation_sha256=annotation,
            clip_start_frame=6,
            clip_end_frame=19,
        )
        for index in (1, 2)
    )
    manifest = UCFRepManifest("ucfrep_526", records)
    path = tmp_path / f"official-segment{suffix}"

    save_ucfrep_manifest(manifest, path)
    loaded = load_ucfrep_manifest(path, protocol="ucfrep_526", validate_exact=False)

    assert loaded == manifest
    changed_range = UCFRepManifest(
        "ucfrep_526",
        tuple(replace(record, clip_end_frame=20) for record in records),
    )
    changed_annotation = UCFRepManifest(
        "ucfrep_526",
        tuple(replace(record, annotation_sha256="d" * 64) for record in records),
    )
    assert changed_range.fingerprint != manifest.fingerprint
    assert changed_annotation.training_fingerprint() != manifest.training_fingerprint()


def test_clip_provenance_is_an_all_or_none_half_open_bundle() -> None:
    with pytest.raises(ValueError, match="must be supplied together"):
        replace(_record(1), clip_start_frame=0, clip_end_frame=10)
    with pytest.raises(ValueError, match="greater than"):
        replace(
            _record(1),
            annotation_sha256="a" * 64,
            clip_start_frame=10,
            clip_end_frame=10,
        )


def test_pose_input_manifest_is_exact_count_free_and_portable(
    tmp_path: Path,
) -> None:
    test_ids = (
        REPOSITORY / "data" / "splits" / "ucfrep_526_test_105.txt"
    ).read_text(encoding="utf-8").splitlines()
    records = tuple(
        UnlabeledVideoRecord(
            video_id=video_id,
            video_path=f"videos/{video_id}.avi",
            video_sha256=hashlib.sha256(video_id.encode("utf-8")).hexdigest(),
        )
        for video_id in test_ids
    )
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split="test",
        records=records,
    )
    manifest.validate_exact_membership()
    payload = manifest.to_dict()
    encoded = json.dumps(payload, sort_keys=True)
    for forbidden in (
        "source_manifest_file_sha256",
        "source_manifest_fingerprint",
        "sealed_dataset_fingerprint",
        "count",
        "action",
    ):
        assert forbidden not in encoded

    path = tmp_path / "pose-inputs.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_pose_input_manifest(path, validate_exact=True)
    assert loaded == manifest
    privileged_payload = dict(payload)
    privileged_payload["source_manifest_file_sha256"] = "a" * 64
    path.write_text(json.dumps(privileged_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fields mismatch"):
        load_pose_input_manifest(path, validate_exact=False)
    moved = PoseInputManifest(
        protocol=manifest.protocol,
        split=manifest.split,
        records=tuple(
            UnlabeledVideoRecord(
                video_id=record.video_id,
                video_path=f"portable-videos/{record.video_id}.avi",
                video_sha256=record.video_sha256,
            )
            for record in records
        ),
    )
    assert moved.fingerprint != manifest.fingerprint
    with pytest.raises(ValueError, match="must be relative"):
        PoseInputManifest(
            protocol=manifest.protocol,
            split=manifest.split,
            records=tuple(
                UnlabeledVideoRecord(
                    video_id=record.video_id,
                    video_path=f"D:/machine-root/{record.video_id}.avi",
                    video_sha256=record.video_sha256,
                )
                for record in records
            ),
        )


def test_pose_input_sidecar_exposes_clip_identity_but_no_cycle_boundaries() -> None:
    record = UnlabeledVideoRecord(
        video_id="v_BenchPress_g01_c01",
        video_path="videos/v_BenchPress_g01_c01.avi",
        video_sha256="a" * 64,
        annotation_sha256="b" * 64,
        clip_start_frame=6,
        clip_end_frame=19,
    )
    manifest = PoseInputManifest("ucfrep_526", "train", (record,))
    encoded = json.dumps(manifest.to_dict(), sort_keys=True)

    assert '"annotation_sha256"' in encoded
    assert '"clip_start_frame": 6' in encoded
    assert '"clip_end_frame": 19' in encoded
    assert '"count"' not in encoded
    assert '"action"' not in encoded
    assert "temporal_bound" not in encoded
    changed = PoseInputManifest(
        "ucfrep_526",
        "train",
        (replace(record, clip_end_frame=20),),
    )
    assert changed.fingerprint != manifest.fingerprint


def test_pose_input_manifest_rejects_noncanonical_test_membership() -> None:
    record = UnlabeledVideoRecord(
        video_id="v_BenchPress_g01_c01",
        video_path="video.avi",
        video_sha256="a" * 64,
    )
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split="test",
        records=(record,),
    )
    with pytest.raises(ValueError, match="exactly"):
        manifest.validate_exact_membership()


@pytest.mark.parametrize(
    ("encoded_key", "sentinel"),
    [
        ('"count"', ""),
        ('"action"', ""),
        ('"co\\u0075nt"', "NaN"),
        ('"act\\u0069on"', "{malformed"),
    ],
)
def test_pose_input_loader_rejects_privileged_key_before_value_deserialization(
    tmp_path: Path,
    encoded_key: str,
    sentinel: str,
) -> None:
    path = tmp_path / "privileged.json"
    path.write_text(f"{{{encoded_key}: {sentinel}}}", encoding="utf-8")

    with pytest.raises(ValueError, match="forbidden privileged field"):
        load_pose_input_manifest(path, validate_exact=False)


def test_pose_input_identity_commitment_is_sorted_and_path_free() -> None:
    records = (
        UnlabeledVideoRecord("video-b", "root-b/video.avi", "b" * 64),
        UnlabeledVideoRecord("video-a", "root-a/video.avi", "a" * 64),
    )
    moved_and_reordered = (
        UnlabeledVideoRecord("video-a", "moved/a.avi", "a" * 64),
        UnlabeledVideoRecord("video-b", "moved/b.avi", "b" * 64),
    )
    identity = pose_input_identity_sha256(records)
    assert identity == pose_input_identity_sha256(moved_and_reordered)
    commitment = PoseInputCommitment(
        protocol="ucfrep_526",
        split="test",
        record_total=2,
        identity_sha256=identity,
        sidecar_sha256="c" * 64,
        sidecar_fingerprint="d" * 64,
    )
    encoded = json.dumps(commitment.to_dict(), sort_keys=True)
    assert "root-a" not in encoded
    assert "moved" not in encoded


def test_exact_standard_protocol_split_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_ids = (
        (REPOSITORY / "data" / "splits" / "ucfrep_526_train_421.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    test_ids = (
        (REPOSITORY / "data" / "splits" / "ucfrep_526_test_105.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )

    def canonical_record(video_id: str, split: str, index: int) -> UCFRepRecord:
        action = video_id.removeprefix("v_").split("_g", 1)[0]
        return UCFRepRecord(
            video_id=video_id,
            video_path=f"videos/{video_id}.avi",
            split=split,
            action=action,
            count=1,
            video_sha256=f"{index + 1:064x}",
        )

    records = tuple(
        canonical_record(video_id, "train", index) for index, video_id in enumerate(train_ids)
    ) + tuple(
        canonical_record(video_id, "test", len(train_ids) + index)
        for index, video_id in enumerate(test_ids)
    )
    manifest = UCFRepManifest("ucfrep_526", records)
    monkeypatch.setattr(
        "pams.data._UCFREP_526_CANONICAL_ANNOTATION_SHA256",
        manifest.sealed_dataset_fingerprint,
    )
    manifest.validate_exact_official_splits()
    reordered = UCFRepManifest("ucfrep_526", tuple(reversed(records)))
    assert reordered.sealed_dataset_fingerprint == manifest.sealed_dataset_fingerprint
    development_assignment = UCFRepManifest(
        "ucfrep_526",
        (replace(records[0], split="dev"), *records[1:]),
    )
    assert development_assignment.sealed_dataset_fingerprint == manifest.sealed_dataset_fingerprint

    tampered_records = list(records)
    tampered_records[-1] = replace(
        tampered_records[-1],
        count=tampered_records[-1].count + 1,
    )
    tampered = UCFRepManifest("ucfrep_526", tuple(tampered_records))
    with pytest.raises(ValueError, match="frozen official"):
        tampered.validate_exact_official_splits()

    swapped = UCFRepManifest(
        "ucfrep_526",
        (
            *(
                canonical_record(
                    record.video_id,
                    "test" if record.video_id == train_ids[0] else record.split,
                    index,
                )
                for index, record in enumerate(records)
                if record.video_id != test_ids[0]
            ),
            canonical_record(test_ids[0], "train", len(records)),
        ),
    )
    with pytest.raises(ValueError, match="preregistered list|source group"):
        swapped.validate_exact_official_splits()

    bad = UCFRepManifest("ucfrep_526", records[:-1])
    with pytest.raises(ValueError, match="exact split"):
        bad.validate_exact_official_splits()


def test_manifest_rejects_duplicate_identity() -> None:
    duplicate = _record(1, split="test")
    with pytest.raises(ValueError, match="duplicate video_id"):
        UCFRepManifest("ucfrep_526", (_record(1), duplicate))


def test_split_disjoint_rejects_same_known_content_under_different_ids() -> None:
    digest = "a" * 64
    with pytest.raises(ValueError, match="video_sha256.*both"):
        assert_split_disjoint(
            (
                UCFRepRecord("train-a", "train.avi", "train", "jump", 3, digest),
                UCFRepRecord("test-b", "test.avi", "test", "jump", 3, digest),
            )
        )


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
    pose_hash = "a" * 64
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
        pose_fingerprint=pose_hash,
    )
    loaded, metadata = load_pose_cache(
        cache_path,
        expected_video_sha256=video_hash,
        expected_pose_fingerprint=pose_hash,
    )
    assert loaded.video_id == "cached"
    assert metadata.frames == 8
    assert metadata.schema_version == 2
    assert metadata.pose_fingerprint == pose_hash
    with pytest.raises(ValueError, match="fingerprint"):
        load_pose_cache(cache_path, expected_pose_fingerprint="c" * 64)

    record = UnlabeledVideoRecord(
        video_id="cached",
        video_path="cached.mp4",
        video_sha256=video_hash,
    )
    dataset = TrainingPoseDataset(
        [record],
        cache_dir=tmp_path,
        pose_fingerprint=pose_hash,
    )
    assert dataset[0].video_id == "cached"
    assert not hasattr(dataset._items[0], "count")
    assert not hasattr(dataset._items[0], "action")
    materialized, snapshot = dataset.materialize_snapshot()
    assert materialized[0].video_id == "cached"
    assert snapshot.entries[0].cache_sha256 == hashlib.sha256(cache_path.read_bytes()).hexdigest()
    assert snapshot.to_dict()["fingerprint"] == snapshot.fingerprint

    changed = PoseSequence(
        "cached",
        30,
        np.zeros((8, 33, 3), dtype=np.float32),
        np.ones(8, dtype=bool),
    )
    write_pose_cache(
        cache_path,
        changed,
        video_sha256=video_hash,
        pose_fingerprint=pose_hash,
        overwrite=True,
    )
    _, changed_snapshot = dataset.materialize_snapshot()
    assert changed_snapshot.fingerprint != snapshot.fingerprint


def test_official_segment_cache_binds_range_annotation_and_decode_coverage(
    tmp_path: Path,
) -> None:
    sequence = PoseSequence(
        "official",
        30,
        np.ones((256, 33, 3), dtype=np.float32),
        np.ones(256, dtype=bool),
    )
    path = pose_cache_path(tmp_path, sequence.video_id)
    write_pose_cache(
        path,
        sequence,
        video_sha256="a" * 64,
        pose_fingerprint="b" * 64,
        annotation_sha256="c" * 64,
        clip_start_frame=6,
        clip_end_frame=19,
        decoded_clip_frames=10,
        expected_clip_frames=13,
        padded_tail_frames=3,
        incomplete_clip_policy="pad_invalid_tail",
    )
    _, metadata = load_pose_cache(
        path,
        expected_video_sha256="a" * 64,
        expected_pose_fingerprint="b" * 64,
        expected_annotation_sha256="c" * 64,
        expected_clip_start_frame=6,
        expected_clip_end_frame=19,
    )

    assert metadata.schema_version == 3
    assert metadata.decoded_clip_frames == 10
    assert metadata.expected_clip_frames == 13
    assert metadata.padded_tail_frames == 3
    with pytest.raises(ValueError, match="clip provenance mismatch"):
        load_pose_cache(
            path,
            expected_annotation_sha256="c" * 64,
            expected_clip_start_frame=6,
            expected_clip_end_frame=20,
        )
    legacy_record = UnlabeledVideoRecord("official", "official.avi", "a" * 64)
    with pytest.raises(ValueError, match="clip provenance mismatch"):
        TrainingPoseDataset(
            (legacy_record,),
            cache_dir=tmp_path,
            pose_fingerprint="b" * 64,
        )[0]


@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_training_dataset_rejects_every_label_bearing_record(
    tmp_path: Path,
    split: str,
) -> None:
    with pytest.raises(TypeError, match="only UnlabeledVideoRecord"):
        TrainingPoseDataset(
            [_record(1, split=split)],
            cache_dir=tmp_path,
            pose_fingerprint="a" * 64,
        )


def test_training_dataset_requires_source_content_hash(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="content-hashed.*video_sha256"):
        TrainingPoseDataset(
            [UnlabeledVideoRecord("train", "train.avi", None)],
            cache_dir=tmp_path,
            pose_fingerprint="a" * 64,
        )


def test_label_free_protocol_identity_preserves_checkpoint_training_fingerprint() -> None:
    split_files = {
        "train": "ucfrep_526_train_337.txt",
        "dev": "ucfrep_526_dev_84.txt",
        "test": "ucfrep_526_test_105.txt",
    }
    sidecars: dict[str, PoseInputManifest] = {}
    labeled_records: list[UCFRepRecord] = []
    for split, filename in split_files.items():
        identifiers = (
            REPOSITORY / "data" / "splits" / filename
        ).read_text(encoding="utf-8").splitlines()
        unlabeled: list[UnlabeledVideoRecord] = []
        for video_id in identifiers:
            digest = hashlib.sha256(video_id.encode("utf-8")).hexdigest()
            locator = f"videos/{video_id}.avi"
            unlabeled.append(UnlabeledVideoRecord(video_id, locator, digest))
            action = video_id.removeprefix("v_").split("_g", 1)[0]
            labeled_records.append(
                UCFRepRecord(
                    video_id,
                    locator,
                    split,
                    action,
                    1,
                    digest,
                )
            )
        sidecars[split] = PoseInputManifest(
            protocol="ucfrep_526",
            split=split,
            records=tuple(unlabeled),
        )
    label_free = LabelFreeProtocolInputs(
        protocol="ucfrep_526",
        train=sidecars["train"],
        dev=sidecars["dev"],
        test=sidecars["test"],
    )
    legacy = UCFRepManifest("ucfrep_526", tuple(labeled_records))

    assert label_free.split_counts == {"train": 337, "dev": 84, "test": 105}
    assert len(label_free.training_records(include_dev=True)) == 421
    assert label_free.training_fingerprint() == legacy.training_fingerprint()
    assert label_free.training_fingerprint(
        include_dev=True
    ) == legacy.training_fingerprint(include_dev=True)


def test_dev_target_manifest_is_exactly_dev_only_and_strictly_loaded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identifiers = (
        REPOSITORY / "data" / "splits" / "ucfrep_526_dev_84.txt"
    ).read_text(encoding="utf-8").splitlines()
    records = tuple(
        DevTargetRecord(
            video_id=video_id,
            action=video_id.removeprefix("v_").split("_g", 1)[0],
            count=index + 1,
        )
        for index, video_id in enumerate(identifiers)
    )
    monkeypatch.setattr(
        data_module,
        "_UCFREP_526_DEV_ANNOTATION_SHA256",
        data_module._dev_target_annotation_sha256(records),
    )
    targets = DevTargetManifest(
        protocol="ucfrep_526",
        records=records,
        source_annotation_sha256=(
            "d371f9f4609730d6484efc337413b444ed73752ad5e994366d02fb79a9960452"
        ),
    )
    path = tmp_path / "dev-targets.json"
    path.write_text(json.dumps(targets.to_dict()), encoding="utf-8")

    assert load_dev_target_manifest(path) == targets
    with pytest.raises(ValueError, match="training-family"):
        DevTargetRecord("v_BenchPress_g21_c01", "BenchPress", 1)

    payload = targets.to_dict()
    payload["records"][0]["video_path"] = "forbidden.avi"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="record 0 fields mismatch"):
        load_dev_target_manifest(path)

    payload = targets.to_dict()
    payload["records"][0]["count"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="counts/actions"):
        load_dev_target_manifest(path)


def test_training_fingerprint_cannot_read_labels_or_sealed_test_rows() -> None:
    train = UCFRepRecord(
        "train",
        "machine-a/train.avi",
        "train",
        "action-a",
        4,
        "a" * 64,
    )
    test = UCFRepRecord(
        "test",
        "machine-a/test.avi",
        "test",
        "action-b",
        9,
        "b" * 64,
    )
    base = UCFRepManifest("ucfrep_526", (train, test))
    relabelled = UCFRepManifest(
        "ucfrep_526",
        (
            UCFRepRecord(
                "train",
                "machine-b/train.avi",
                "train",
                "different-train-action",
                99,
                "a" * 64,
            ),
            UCFRepRecord(
                "test",
                "machine-b/test.avi",
                "test",
                "different-test-action",
                123,
                "c" * 64,
            ),
        ),
    )
    assert base.training_fingerprint() == relabelled.training_fingerprint()
    assert base.fingerprint != relabelled.fingerprint

    changed_train_content = UCFRepManifest(
        "ucfrep_526",
        (
            UCFRepRecord(
                "train",
                "train.avi",
                "train",
                "action-a",
                4,
                "d" * 64,
            ),
            test,
        ),
    )
    assert base.training_fingerprint() != changed_train_content.training_fingerprint()


def test_legacy_full_config_pose_cache_is_rejected_readably(tmp_path: Path) -> None:
    path = tmp_path / "legacy.npz"
    metadata = {
        "schema_version": 1,
        "video_id": "legacy",
        "video_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "pose_model": "mediapipe-pose-0.10.14",
        "fps": 30.0,
        "frames": 4,
        "keypoints": 33,
        "coordinates": 3,
    }
    np.savez_compressed(
        path,
        xyz=np.zeros((4, 33, 3), dtype=np.float32),
        valid_mask=np.ones(4, dtype=np.bool_),
        metadata=np.asarray(json.dumps(metadata)),
    )
    with pytest.raises(ValueError, match="legacy pose cache.*regenerate"):
        load_pose_cache(path)
