import hashlib
from pathlib import Path

import numpy as np
import pytest

from pams.config import PAMSConfig
from pams.data import UCFRepRecord, pose_cache_path, write_pose_cache
from pams.types import PoseSequence
from scripts.server.audit_pose_cache import (
    AuditError,
    _audit_cache_entry,
    _validate_cache_scope,
    _validate_identity_ledger,
)


def _clean_identity_ledger() -> dict[str, object]:
    return {
        "schema_version": 1,
        "selected": 421,
        "completed": 421,
        "extracted": 0,
        "skipped": 421,
        "failed": 0,
        "failures": [],
    }


def test_identity_ledger_accepts_clean_full_pool_resume() -> None:
    counts = _validate_identity_ledger(_clean_identity_ledger())
    assert counts == {
        "schema_version": 1,
        "selected": 421,
        "completed": 421,
        "extracted": 0,
        "skipped": 421,
        "failed": 0,
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("completed", 420, "completed \\+ failed == selected"),
        ("skipped", 420, "extracted \\+ skipped == completed"),
    ],
)
def test_identity_ledger_rejects_broken_invariants(
    field: str,
    value: int,
    message: str,
) -> None:
    ledger = _clean_identity_ledger()
    ledger[field] = value
    with pytest.raises(AuditError, match=message):
        _validate_identity_ledger(ledger)


def test_identity_ledger_rejects_failure_list_count_mismatch() -> None:
    ledger = _clean_identity_ledger()
    ledger["completed"] = 420
    ledger["skipped"] = 420
    ledger["failed"] = 1
    with pytest.raises(AuditError, match="failed count"):
        _validate_identity_ledger(ledger)


def _source_record(tmp_path: Path) -> tuple[Path, UCFRepRecord]:
    source = tmp_path / "video.avi"
    source.write_bytes(b"stable-video")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return source, UCFRepRecord(
        video_id="v_Audit_g01_c01",
        video_path=str(source),
        split="train",
        action="Audit",
        count=1,
        video_sha256=digest,
    )


def _normalized_sequence(video_id: str, frames: int) -> PoseSequence:
    frame = np.linspace(0.0, 1.0, 99, dtype=np.float32).reshape(33, 3)
    xyz = np.repeat(frame[None, ...], frames, axis=0)
    return PoseSequence(
        video_id=video_id,
        fps=30.0,
        xyz=xyz,
        valid_mask=np.ones(frames, dtype=np.bool_),
    )


def test_cache_entry_accepts_exact_npz_identity(tmp_path: Path) -> None:
    _, record = _source_record(tmp_path)
    config = PAMSConfig()
    cache_dir = tmp_path / "cache"
    write_pose_cache(
        pose_cache_path(cache_dir, record.video_id),
        _normalized_sequence(record.video_id, config.data.frames),
        video_sha256=record.video_sha256 or "",
        pose_fingerprint=config.pose_fingerprint,
    )

    audited = _audit_cache_entry(
        record,
        manifest_dir=tmp_path,
        cache_dir=cache_dir,
        config=config,
    )

    assert audited.receipt.video_id == record.video_id
    assert audited.cached_frames == 256
    assert audited.cached_valid_frames == 256


def test_cache_entry_rejects_wrong_pose_fingerprint(tmp_path: Path) -> None:
    _, record = _source_record(tmp_path)
    config = PAMSConfig()
    cache_dir = tmp_path / "cache"
    write_pose_cache(
        pose_cache_path(cache_dir, record.video_id),
        _normalized_sequence(record.video_id, config.data.frames),
        video_sha256=record.video_sha256 or "",
        pose_fingerprint="f" * 64,
    )

    with pytest.raises(AuditError, match="metadata or content identity"):
        _audit_cache_entry(
            record,
            manifest_dir=tmp_path,
            cache_dir=cache_dir,
            config=config,
        )


def test_cache_scope_rejects_extra_npz(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    expected = cache_dir / "expected.npz"
    expected.write_bytes(b"expected")
    (cache_dir / "extra.npz").write_bytes(b"extra")

    with pytest.raises(AuditError, match="1 extra"):
        _validate_cache_scope(cache_dir, {expected})
