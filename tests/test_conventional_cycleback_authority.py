from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import pams.conventional_cycleback.authority as authority_module
from pams.cli import _reject_v4e_from_generic_training
from pams.conventional_cycleback.authority import (
    validate_cycleback_launch_registry,
    validate_unified_representation_authority,
    write_json_exclusive,
)
from pams.conventional_cycleback.runtime import (
    JointMaskEntryReceipt,
    JointMaskSetSnapshot,
)


def _write_json(path: Path, payload: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_secure_scaffold_cannot_activate_its_own_checkout_or_representation(
    tmp_path: Path,
) -> None:
    forged = tmp_path / ("a" * 64)
    forged.mkdir()

    assert authority_module._APPROVED_REPRESENTATION_LOCATOR == ""
    assert authority_module._APPROVED_CYCLEBACK_SOURCE_REVISION == ""
    assert authority_module._APPROVED_CYCLEBACK_SOURCE_TREE_SHA == ""
    assert authority_module._APPROVED_CYCLEBACK_CONFIGS == {}
    assert authority_module._APPROVED_SYNTHETIC_PREREG_SHA256 == ""

    with pytest.raises(ValueError, match="independently preregistered"):
        validate_unified_representation_authority(
            forged,
            declared_host_root=forged,
        )
    with pytest.raises(ValueError, match="disabled pending an independent activation"):
        validate_cycleback_launch_registry(
            forged,
            declared_host_root=authority_module._CANONICAL_LAUNCH_REGISTRY_ROOT,
        )


def test_v4e_snapshot_is_parsed_then_normalized_without_schema_aliasing(
    tmp_path: Path,
) -> None:
    entries = (
        JointMaskEntryReceipt(video_id="person-a", sidecar_sha256="1" * 64, bytes=11),
        JointMaskEntryReceipt(video_id="person-b", sidecar_sha256="2" * 64, bytes=12),
    )
    v4e_identity = {
        "schema_version": 1,
        "entries": [entry.to_dict() for entry in entries],
    }
    v4e_fingerprint = hashlib.sha256(
        json.dumps(
            v4e_identity,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    source = tmp_path / "joint-mask-set.snapshot.json"
    source_sha = _write_json(
        source,
        {
            **v4e_identity,
            "artifact_type": "pams_pose_recovery_v4e_joint_mask_set_snapshot_v1",
            "entry_count": 2,
            "fingerprint": v4e_fingerprint,
        },
    )

    loaded, observed_sha, _ = authority_module._load_v4e_joint_mask_snapshot(source)
    normalized = JointMaskSetSnapshot(entries=entries)

    assert observed_sha == source_sha
    assert loaded.entries == entries
    assert loaded.fingerprint == v4e_fingerprint
    assert normalized.fingerprint != loaded.fingerprint
    assert (
        normalized.to_dict()["artifact_type"]
        == "pams_cycleback_unified_2d_joint_mask_set_snapshot_v1"
    )


def test_exclusive_json_writer_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    write_json_exclusive(target, {"status": "first"})

    with pytest.raises(FileExistsError):
        write_json_exclusive(target, {"status": "replacement"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "first"}


@pytest.mark.parametrize("operation", ["generic encoder training", "generic SSHead training"])
def test_generic_training_rejects_v4e_unified_2d(operation: str) -> None:
    config = SimpleNamespace(
        pose=SimpleNamespace(
            preprocessing_revision=(
                "official-segment-keypointrcnn-single-source-coco17-full-timeline-v4e"
            )
        )
    )

    with pytest.raises(ValueError, match="full-stable-range encode-once"):
        _reject_v4e_from_generic_training(config, operation=operation)


def test_authority_has_no_direct_full337_training_entrypoint() -> None:
    source = Path(authority_module.__file__).read_text(encoding="utf-8")

    assert not hasattr(authority_module, "Full337Authority")
    assert not hasattr(authority_module, "validate_full337_training_authority")
    assert "--full337-root" not in source
    assert "--expected-full337" not in source
    assert "raw_full337_used_as_extraction_evidence_only" in source
    assert "raw_full337_training_authority_accepted" in source
