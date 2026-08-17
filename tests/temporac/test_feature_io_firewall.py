from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pams.temporac.feature_io import (
    FeatureIOError,
    load_feature_archive,
    reject_forbidden_path,
    reject_privileged_keys,
    validate_feature_mapping_keys,
)
from pams.temporac.hashio import sha256_bytes
from pams.temporac.receipts import parse_receipt_bytes
from pams.temporac.trusted_packer import (
    PackerError,
    PopulationRow,
    PrivilegedAnnotation,
    TrustedSourceObject,
    pack_source_object,
    population_manifest_bytes,
    publish_packed_identity,
    verify_source_file,
)


def _source() -> TrustedSourceObject:
    motion = np.zeros((3, 320, 17, 3), dtype="<f4")
    motion[..., 2] = np.float32(0.9)
    motion[0, 4, 2] = np.asarray([np.nan, 5.0, 0.9], dtype="<f4")
    motion[0, 5, 3] = np.asarray([9.0, 8.0, 0.1], dtype="<f4")
    return TrustedSourceObject(
        motion=motion,
        person_mask=np.asarray([True, False, True], dtype=np.bool_),
        frame_mask=np.ones((3, 320), dtype=np.bool_),
        sampled_frame_indices=np.arange(320, dtype="<i8"),
        source_length=320,
        person_object_ids=("person-a", "ignored", "person-c"),
    )


def test_trusted_packer_preserves_slots_separates_vault_and_canonicalizes_invalid_joints(
    tmp_path: Path,
) -> None:
    source_hash = "1" * 64
    packed = pack_source_object(
        _source(),
        split="train",
        source_pickle_sha256=source_hash,
        object_ordinal=7,
        annotations={
            "person-a": PrivilegedAnnotation(3, ((0, 10), (10, 20), (20, 30))),
            "person-c": PrivilegedAnnotation(2, ((2, 12), (20, 31))),
        },
    )
    assert tuple(item.feature.slot for item in packed) == (0, 2)
    assert packed[0].feature.motion[4, 2].tobytes() == np.zeros(3, dtype="<f4").tobytes()
    assert packed[0].feature.motion[5, 3].tobytes() == np.zeros(3, dtype="<f4").tobytes()
    features = tmp_path / "features" / "train"
    receipts = tmp_path / "receipts" / "train"
    vault = tmp_path / "private-vault" / "train"
    features.mkdir(parents=True)
    receipts.mkdir(parents=True)
    vault.mkdir(parents=True)
    key = packed[0].feature.opaque_key_bytes.hex()
    feature_path = features / f"{key}.0.npz"
    receipt_path = receipts / f"{key}.0.receipt.json"
    vault_path = vault / f"{key}.0.json"
    artifact_hash, receipt_hash, vault_hash = publish_packed_identity(
        packed[0],
        feature_path=feature_path,
        receipt_path=receipt_path,
        vault_path=vault_path,
    )
    routed = load_feature_archive(feature_path)
    assert routed.identity == (bytes.fromhex(key), 0)
    assert set(routed.preprocessing_arrays()) == {
        "frame_mask",
        "motion",
        "person_mask",
        "sampled_frame_indices",
        "source_length",
    }
    assert "opaque_sample_key" not in routed.preprocessing_arrays()
    receipt = parse_receipt_bytes(receipt_path.read_bytes())
    assert receipt["artifact_sha256"] == artifact_hash
    assert sha256_bytes(receipt_path.read_bytes()) == receipt_hash
    assert sha256_bytes(vault_path.read_bytes()) == vault_hash
    assert json.loads(vault_path.read_text(encoding="utf-8"))["count_gt"] == 3


def test_learner_firewall_rejects_paths_and_keys_before_loader_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden_loader(*args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        raise AssertionError("archive loader must not be reached")

    monkeypatch.setattr("pams.temporac.feature_io.read_deterministic_npz", forbidden_loader)
    with pytest.raises(FeatureIOError, match="forbidden"):
        load_feature_archive(Path("sealed") / ("0" * 64 + ".0.npz"))
    assert not called
    with pytest.raises(FeatureIOError, match="forbidden"):
        reject_forbidden_path(Path("results") / "artifact.npz")
    with pytest.raises(FeatureIOError, match="privileged"):
        reject_privileged_keys({"motion": [], "count_gt": 4})
    with pytest.raises(FeatureIOError, match="seven-field"):
        validate_feature_mapping_keys({"motion": object()})


def test_trusted_source_path_is_denied_before_read(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    def forbidden_read(*args: object, **kwargs: object) -> bytes:
        nonlocal called
        called = True
        return b""

    monkeypatch.setattr("pams.temporac.trusted_packer.read_regular_file", forbidden_read)
    path = Path("heldout") / "source.bin"
    with pytest.raises(PackerError, match="forbidden"):
        verify_source_file(
            path,
            split="val",
            allowlisted_path=path,
            expected_sha256="0" * 64,
            expected_bytes=100,
        )
    assert not called


def test_join_and_population_rows_fail_closed() -> None:
    with pytest.raises(PackerError, match="missing, extra, or non-total"):
        pack_source_object(
            _source(),
            split="train",
            source_pickle_sha256="1" * 64,
            object_ordinal=0,
            annotations={"person-a": PrivilegedAnnotation(1, ((0, 10),))},
        )
    row = PopulationRow(
        split="train",
        opaque_key_hex="1" * 64,
        slot=0,
        component_key_hex="2" * 64,
        source_binding_sha256="3" * 64,
        feature_receipt_sha256_or_null="4" * 64,
        eligible=True,
        reason_codes=(),
    )
    payload = population_manifest_bytes([row], enforce_frozen_totals=False)
    assert list(json.loads(payload)) == [row.as_dict()]
    with pytest.raises(PackerError, match="268 train and 134 val"):
        population_manifest_bytes([row])
