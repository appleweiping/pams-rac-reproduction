import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pams.data import (
    PoseCacheSetSnapshot,
    PoseInputManifest,
    UnlabeledVideoRecord,
    load_pose_cache_with_receipt,
    pose_cache_path,
    pose_input_identity_sha256,
    write_pose_cache,
)
from pams.types import PoseSequence
from scripts.server import audit_pose_cache_v2_v3 as audit_module


def _record(video_id: str) -> UnlabeledVideoRecord:
    return UnlabeledVideoRecord(
        video_id=video_id,
        video_path=f"videos/{video_id}.avi",
        video_sha256=hashlib.sha256(video_id.encode("utf-8")).hexdigest(),
    )


def _write_sidecar(
    path: Path,
    *,
    split: str,
    records: tuple[UnlabeledVideoRecord, ...],
) -> PoseInputManifest:
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split=split,
        records=records,
    )
    path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _sequence(
    video_id: str,
    *,
    valid_frames: int,
) -> PoseSequence:
    xyz = np.zeros((256, 33, 3), dtype=np.float32)
    valid_mask = np.zeros(256, dtype=np.bool_)
    if valid_frames:
        frame = np.linspace(0.0, 1.0, 99, dtype=np.float32).reshape(33, 3)
        xyz[:valid_frames] = frame
        valid_mask[:valid_frames] = True
    return PoseSequence(
        video_id=video_id,
        fps=30.0,
        xyz=xyz,
        valid_mask=valid_mask,
    )


def _write_cache(
    root: Path,
    record: UnlabeledVideoRecord,
    *,
    pose_fingerprint: str,
    valid_frames: int,
) -> None:
    write_pose_cache(
        pose_cache_path(root, record.video_id),
        _sequence(record.video_id, valid_frames=valid_frames),
        video_sha256=record.video_sha256 or "",
        pose_fingerprint=pose_fingerprint,
    )


def _snapshot(
    root: Path,
    records: tuple[UnlabeledVideoRecord, ...],
    *,
    pose_fingerprint: str,
) -> PoseCacheSetSnapshot:
    receipts = []
    for record in records:
        _, _, receipt = load_pose_cache_with_receipt(
            pose_cache_path(root, record.video_id),
            expected_video_sha256=record.video_sha256,
            expected_pose_fingerprint=pose_fingerprint,
        )
        receipts.append(receipt)
    return PoseCacheSetSnapshot(
        pose_fingerprint=pose_fingerprint,
        entries=tuple(receipts),
    )


def _write_ledger(
    path: Path,
    *,
    sidecar_path: Path,
    sidecar: PoseInputManifest,
    cache_root: Path,
    source_stats: dict[str, tuple[int, int, int]],
    pose_fingerprint: str,
) -> None:
    summaries = []
    for record in sidecar.records:
        sequence, metadata, _ = load_pose_cache_with_receipt(
            pose_cache_path(cache_root, record.video_id),
            expected_video_sha256=record.video_sha256,
            expected_pose_fingerprint=pose_fingerprint,
        )
        source_frames, source_valid_frames, selected_source_frames = source_stats[
            record.video_id
        ]
        summaries.append(
            {
                "video_id": record.video_id,
                "video_path": record.video_path,
                "cache_path": str(pose_cache_path(cache_root, record.video_id)),
                "video_sha256": record.video_sha256,
                "pose_fingerprint": pose_fingerprint,
                "source_frames": source_frames,
                "source_valid_frames": source_valid_frames,
                "selected_source_frames": selected_source_frames,
                "cached_frames": sequence.num_frames,
                "cached_valid_frames": int(np.count_nonzero(sequence.valid_mask)),
                "fps": sequence.fps,
                "pose_model": metadata.pose_model,
                "skipped": False,
            }
        )
    sidecar_sha256 = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    payload = {
        "schema_version": 2,
        "input_kind": "label_free_sidecar",
        "protocol": "ucfrep_526",
        "split": sidecar.split,
        "input_file_sha256": sidecar_sha256,
        "input_fingerprint": sidecar.fingerprint,
        "sidecar_sha256": sidecar_sha256,
        "sidecar_fingerprint": sidecar.fingerprint,
        "commitment_file_sha256": "a" * 64,
        "commitment_fingerprint": "b" * 64,
        "identity_sha256": pose_input_identity_sha256(sidecar.records),
        "pose_fingerprint": pose_fingerprint,
        "successful_cache_snapshot": _snapshot(
            cache_root,
            sidecar.records,
            pose_fingerprint=pose_fingerprint,
        ).to_dict(),
        "selected": len(sidecar.records),
        "completed": len(sidecar.records),
        "extracted": len(sidecar.records),
        "skipped": 0,
        "failed": 0,
        "caches": summaries,
        "failures": [],
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


@pytest.fixture
def small_exact_membership(monkeypatch: pytest.MonkeyPatch) -> None:
    real_loader = audit_module.load_pose_input_manifest

    def load_without_official_membership(
        path: Path,
        *,
        validate_exact: bool,
    ) -> PoseInputManifest:
        assert validate_exact is True
        return real_loader(path, validate_exact=False)

    monkeypatch.setattr(
        audit_module,
        "load_pose_input_manifest",
        load_without_official_membership,
    )
    monkeypatch.setattr(
        audit_module,
        "_EXPECTED_SPLIT_SIZES",
        {"train": 2, "dev": 1},
    )
    monkeypatch.setattr(audit_module, "_EXPECTED_TOTAL", 3)


def test_label_free_v2_v3_audit_reports_split_and_total_metrics(
    tmp_path: Path,
    small_exact_membership: None,
) -> None:
    del small_exact_membership
    train_records = (_record("train-a"), _record("train-b"))
    dev_records = (_record("dev-c"),)
    train_path = tmp_path / "train.json"
    dev_path = tmp_path / "dev.json"
    train = _write_sidecar(train_path, split="train", records=train_records)
    dev = _write_sidecar(dev_path, split="dev", records=dev_records)
    v2_root = tmp_path / "v2"
    v3_root = tmp_path / "v3"

    for record, v2_valid, v3_valid in (
        (train_records[0], 256, 256),
        (train_records[1], 0, 0),
        (dev_records[0], 128, 0),
    ):
        _write_cache(
            v2_root,
            record,
            pose_fingerprint=audit_module._FROZEN_V2_POSE_FINGERPRINT,
            valid_frames=v2_valid,
        )
        _write_cache(
            v3_root,
            record,
            pose_fingerprint=audit_module._FROZEN_V3_POSE_FINGERPRINT,
            valid_frames=v3_valid,
        )

    train_ledger = tmp_path / "train-ledger.json"
    dev_ledger = tmp_path / "dev-ledger.json"
    _write_ledger(
        train_ledger,
        sidecar_path=train_path,
        sidecar=train,
        cache_root=v3_root,
        source_stats={"train-a": (20, 20, 20), "train-b": (30, 0, 0)},
        pose_fingerprint=audit_module._FROZEN_V3_POSE_FINGERPRINT,
    )
    _write_ledger(
        dev_ledger,
        sidecar_path=dev_path,
        sidecar=dev,
        cache_root=v3_root,
        source_stats={"dev-c": (10, 1, 1)},
        pose_fingerprint=audit_module._FROZEN_V3_POSE_FINGERPRINT,
    )

    result = audit_module.audit_pose_cache_v2_v3(
        train_sidecar=train_path,
        dev_sidecar=dev_path,
        v2_cache_root=v2_root,
        v3_train_ledger=train_ledger,
        v3_dev_ledger=dev_ledger,
        v3_cache_root=v3_root,
    )

    assert result["status"] == "passed"
    assert result["scope"] == {
        "protocol": "ucfrep_526",
        "splits": ["train337", "dev84"],
        "sample_count": 3,
        "test105_accessed": False,
        "source_videos_accessed": False,
        "labels_or_targets_accessed": False,
    }
    assert result["splits"]["train"]["sample_count"] == 2
    assert result["splits"]["dev"]["sample_count"] == 1
    total = result["total"]
    assert total["sample_count"] == 3
    assert total["v2"]["all_invalid_count"] == 1
    assert total["v3"]["all_invalid_count"] == 2
    assert total["v2"]["valid_frame_rate_distribution"]["mean"] == pytest.approx(0.5)
    assert total["v3"]["valid_frame_rate_distribution"]["mean"] == pytest.approx(1 / 3)
    assert total["v3"]["source_frames_distribution"]["count"] == 3
    assert total["v3"]["source_valid_frames_distribution"]["count"] == 3
    assert total["v3"]["selected_source_frames_distribution"]["count"] == 3
    assert total["v3"]["single_frame_converted_to_all_invalid_count"] == 1
    assert total["video_ids"]["v2_all_invalid"] == ["train-b"]
    assert total["video_ids"]["v3_all_invalid"] == ["dev-c", "train-b"]
    assert total["video_ids"]["v3_extremely_short"] == ["dev-c"]
    assert total["video_ids"]["v3_single_frame_converted_to_all_invalid"] == [
        "dev-c"
    ]
    assert len(total["v2"]["cache_snapshot_sha256"]) == 64
    assert len(total["v3"]["cache_set_sha256"]) == 64


@pytest.mark.parametrize("forbidden", ["count", "action", "target", "targets"])
def test_json_inputs_fail_closed_on_label_or_target_fields(
    tmp_path: Path,
    forbidden: str,
) -> None:
    path = tmp_path / "forbidden.json"
    path.write_text(
        json.dumps({"schema_version": 2, "nested": {forbidden: 7}}),
        encoding="utf-8",
    )

    with pytest.raises(
        audit_module.ComparisonAuditError,
        match="forbidden field",
    ):
        audit_module._load_label_free_json(path, document="test fixture")


def test_sidecar_pair_rejects_overlap() -> None:
    shared = _record("shared")
    train = PoseInputManifest(
        protocol="ucfrep_526",
        split="train",
        records=(shared,),
    )
    dev = PoseInputManifest(
        protocol="ucfrep_526",
        split="dev",
        records=(shared,),
    )

    with pytest.raises(audit_module.ComparisonAuditError, match="overlap"):
        audit_module._validate_sidecar_pair(train, dev)


def test_ledger_rejects_cache_snapshot_mismatch(
    tmp_path: Path,
    small_exact_membership: None,
) -> None:
    del small_exact_membership
    train_records = (_record("train-a"), _record("train-b"))
    train_path = tmp_path / "train.json"
    train = _write_sidecar(train_path, split="train", records=train_records)
    v3_root = tmp_path / "v3"
    for record in train_records:
        _write_cache(
            v3_root,
            record,
            pose_fingerprint=audit_module._FROZEN_V3_POSE_FINGERPRINT,
            valid_frames=256,
        )
    ledger_path = tmp_path / "train-ledger.json"
    _write_ledger(
        ledger_path,
        sidecar_path=train_path,
        sidecar=train,
        cache_root=v3_root,
        source_stats={"train-a": (10, 10, 10), "train-b": (10, 10, 10)},
        pose_fingerprint=audit_module._FROZEN_V3_POSE_FINGERPRINT,
    )
    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    payload["successful_cache_snapshot"]["fingerprint"] = "f" * 64
    ledger_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    observations = audit_module._load_cache_observations(
        train.records,
        cache_root=v3_root,
        expected_pose_fingerprint=audit_module._FROZEN_V3_POSE_FINGERPRINT,
        cache_revision="v3",
    )

    with pytest.raises(audit_module.ComparisonAuditError, match="snapshot mismatch"):
        audit_module._validate_v3_ledger(
            ledger_path,
            split="train",
            sidecar=train,
            sidecar_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),
            cache_observations=observations,
            expected_pose_fingerprint=audit_module._FROZEN_V3_POSE_FINGERPRINT,
        )
