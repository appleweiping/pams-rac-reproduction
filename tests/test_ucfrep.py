import hashlib
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from pams import data as data_module
from pams import ucfrep
from pams.data import UCFRepManifest, UCFRepRecord
from pams.ucfrep import (
    _annotation_count,
    build_official_manifest,
    materialize_standard_split_ids,
    resolve_ucfrep_video_path,
)


class _Label:
    def __init__(self, boundaries: np.ndarray) -> None:
        self.temporal_bound = boundaries


def _official_archive_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, tuple[tuple[str, str], ...]]:
    annotation = io.BytesIO()
    savemat(
        annotation,
        {
            "label": {
                "temporal_bound": np.asarray([[1], [10], [20]]),
            }
        },
    )
    payload = annotation.getvalue()
    rows: list[tuple[str, str]] = []
    archive_path = tmp_path / "ucf526_annotations.zip"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for index in range(421):
            action = f"Action{index % 7}"
            group = index % 20 + 1
            video_id = f"v_{action}_g{group:02d}_c{index + 1:04d}"
            archive.writestr(f"annotations/train/{video_id}.mat", payload)
            rows.append((video_id, action))
        for index in range(105):
            action = f"Action{index % 7}"
            group = index % 5 + 21
            video_id = f"v_{action}_g{group:02d}_c{index + 1:04d}"
            archive.writestr(f"annotations/val/{video_id}.mat", payload)
            rows.append((video_id, action))
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    monkeypatch.setattr(ucfrep, "OFFICIAL_ANNOTATION_SHA256", digest)
    train_ids = sorted(video_id for video_id, _ in rows[:421])
    test_ids = sorted(video_id for video_id, _ in rows[421:])
    canonical_hashes = dict(data_module._UCFREP_526_CANONICAL_ID_SHA256)
    canonical_hashes["train_pool"] = hashlib.sha256(
        ("\n".join(train_ids) + "\n").encode("utf-8")
    ).hexdigest()
    canonical_hashes["test"] = hashlib.sha256(
        ("\n".join(test_ids) + "\n").encode("utf-8")
    ).hexdigest()
    monkeypatch.setattr(
        data_module,
        "_UCFREP_526_CANONICAL_ID_SHA256",
        canonical_hashes,
    )
    semantic_records = tuple(
        UCFRepRecord(
            video_id=video_id,
            video_path=f"{video_id}.avi",
            split="train" if index < 421 else "test",
            action=action,
            count=2,
        )
        for index, (video_id, action) in enumerate(rows)
    )
    monkeypatch.setattr(
        data_module,
        "_UCFREP_526_CANONICAL_ANNOTATION_SHA256",
        data_module._ucfrep_526_annotation_fingerprint(semantic_records),
    )
    return archive_path, tuple(rows)


def test_annotation_count_matches_official_boundary_definition(tmp_path: Path) -> None:
    path = tmp_path / "annotation.mat"
    label = {
        "duration": np.asarray([[100]]),
        "start_frame": np.asarray([[1]]),
        "end_frame": np.asarray([[100]]),
        "temporal_bound_num": np.asarray([[5]]),
        "temporal_bound": np.asarray([[1], [25], [50], [75], [100]]),
        "offset_next_estimate": np.zeros((1, 100)),
        "offset_pre_estimate": np.zeros((1, 100)),
    }
    savemat(path, {"label": label})
    assert _annotation_count(path.read_bytes()) == 4


def test_annotation_count_rejects_missing_boundaries() -> None:
    buffer = io.BytesIO()
    savemat(buffer, {"other": np.asarray([1])})
    with pytest.raises(ValueError, match="label"):
        _annotation_count(buffer.getvalue())


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("UCF-101") / "JumpRope" / "v_JumpRope_g01_c01.avi",
        Path("JumpRope") / "v_JumpRope_g01_c01.avi",
        Path("flat") / "v_JumpRope_g01_c01.avi",
    ],
)
def test_video_resolver_supports_all_three_declared_layouts(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    expected = tmp_path / relative_path
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"video")
    resolved = resolve_ucfrep_video_path(
        tmp_path,
        video_id="v_JumpRope_g01_c01",
        action="JumpRope",
    )
    assert resolved == expected


@pytest.mark.parametrize(
    "relative_path",
    [
        Path("UCF-101") / "HandstandPushups" / "v_HandStandPushups_g02_c04.avi",
        Path("HandstandPushups") / "v_HandStandPushups_g02_c04.avi",
    ],
)
def test_video_resolver_supports_audited_official_handstand_directory_alias(
    tmp_path: Path,
    relative_path: Path,
) -> None:
    expected = tmp_path / relative_path
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"video")

    resolved = resolve_ucfrep_video_path(
        tmp_path,
        video_id="v_HandStandPushups_g02_c04",
        action="HandStandPushups",
    )

    assert resolved.samefile(expected)


def test_video_resolver_rejects_canonical_and_alias_directory_ambiguity(
    tmp_path: Path,
) -> None:
    if os.path.normcase("HandStandPushups") == os.path.normcase("HandstandPushups"):
        pytest.skip("case-insensitive filesystems cannot materialize both directory spellings")
    filename = "v_HandStandPushups_g02_c04.avi"
    canonical = tmp_path / "UCF-101" / "HandStandPushups" / filename
    alias = tmp_path / "UCF-101" / "HandstandPushups" / filename
    canonical.parent.mkdir(parents=True)
    alias.parent.mkdir(parents=True)
    canonical.write_bytes(b"canonical")
    alias.write_bytes(b"alias")

    with pytest.raises(ValueError, match="ambiguous UCFRep video path"):
        resolve_ucfrep_video_path(
            tmp_path,
            video_id="v_HandStandPushups_g02_c04",
            action="HandStandPushups",
        )


def test_video_resolver_rejects_ambiguous_layouts_even_annotation_only(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "UCF-101" / "JumpRope" / "v_JumpRope_g01_c01.avi"
    flat = tmp_path / "flat" / "v_JumpRope_g01_c01.avi"
    nested.parent.mkdir(parents=True)
    flat.parent.mkdir(parents=True)
    nested.write_bytes(b"nested")
    flat.write_bytes(b"flat")
    with pytest.raises(ValueError, match="ambiguous UCFRep video path"):
        resolve_ucfrep_video_path(
            tmp_path,
            video_id="v_JumpRope_g01_c01",
            action="JumpRope",
            allow_missing=True,
        )


def test_video_resolver_requires_explicit_annotation_only_for_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="missing UCFRep video"):
        resolve_ucfrep_video_path(
            tmp_path,
            video_id="v_JumpRope_g01_c01",
            action="JumpRope",
        )
    placeholder = resolve_ucfrep_video_path(
        tmp_path,
        video_id="v_JumpRope_g01_c01",
        action="JumpRope",
        allow_missing=True,
    )
    assert placeholder == tmp_path / "JumpRope" / "v_JumpRope_g01_c01.avi"


def test_video_resolver_materializes_relative_root_as_absolute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / "videos" / "JumpRope" / "v_JumpRope_g01_c01.avi"
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"video")

    resolved = resolve_ucfrep_video_path(
        Path("videos"),
        video_id="v_JumpRope_g01_c01",
        action="JumpRope",
    )

    assert resolved == expected.resolve()
    assert resolved.is_absolute()


def test_strict_official_manifest_resolves_and_hashes_all_526_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, rows = _official_archive_fixture(tmp_path, monkeypatch)
    video_root = tmp_path / "videos"
    for index, (video_id, action) in enumerate(rows):
        if index % 3 == 0:
            path = video_root / "UCF-101" / action / f"{video_id}.avi"
        elif index % 3 == 1:
            path = video_root / action / f"{video_id}.avi"
        else:
            path = video_root / "flat" / f"{video_id}.avi"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(video_id.encode("utf-8"))

    manifest = build_official_manifest(archive, video_root=video_root)
    assert manifest.split_counts == {"test": 105, "train": 421}
    assert len(manifest.records) == 526
    assert all(Path(record.video_path).is_file() for record in manifest.records)
    assert all(
        record.video_sha256 is not None and len(record.video_sha256) == 64
        for record in manifest.records
    )


def test_annotation_only_manifest_explicitly_allows_missing_videos(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive, _ = _official_archive_fixture(tmp_path, monkeypatch)
    video_root = tmp_path / "missing-videos"
    manifest = build_official_manifest(
        archive,
        video_root=video_root,
        hash_existing_videos=False,
        allow_missing_videos=True,
    )
    assert len(manifest.records) == 526
    assert all(record.video_sha256 is None for record in manifest.records)
    assert all(Path(record.video_path).parent.name == record.action for record in manifest.records)


def test_materialize_standard_split_ids_is_label_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_ids = (
        (Path(__file__).parents[1] / "data" / "splits" / "ucfrep_526_train_421.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    test_ids = (
        (Path(__file__).parents[1] / "data" / "splits" / "ucfrep_526_test_105.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    records = tuple(
        UCFRepRecord(
            video_id=video_id,
            video_path=f"videos/train-{index}.avi",
            split="train",
            action=video_id.removeprefix("v_").split("_g", 1)[0],
            count=index % 45 + 1,
        )
        for index, video_id in enumerate(train_ids)
    ) + tuple(
        UCFRepRecord(
            video_id=video_id,
            video_path=f"videos/test-{index}.avi",
            split="test",
            action=video_id.removeprefix("v_").split("_g", 1)[0],
            count=index % 45 + 1,
        )
        for index, video_id in enumerate(test_ids)
    )
    manifest = UCFRepManifest("ucfrep_526", records)
    monkeypatch.setattr(
        data_module,
        "_UCFREP_526_CANONICAL_ANNOTATION_SHA256",
        manifest.sealed_dataset_fingerprint,
    )
    outputs = materialize_standard_split_ids(manifest, tmp_path)
    assert len((tmp_path / "ucfrep_526_train_421.txt").read_text().splitlines()) == 421
    assert len((tmp_path / "ucfrep_526_dev_84.txt").read_text().splitlines()) == 84
    assert set(outputs) == {
        "ucfrep_526_train_421.txt",
        "ucfrep_526_test_105.txt",
        "ucfrep_526_train_337.txt",
        "ucfrep_526_dev_84.txt",
    }
    assert "action-" not in (tmp_path / "ucfrep_526_train_337.txt").read_text()
    for output in outputs.values():
        assert b"\r\n" not in output.read_bytes()
