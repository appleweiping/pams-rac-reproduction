from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest

from pams.temporac.contract import (
    ALLOCATED_A6000_HOURS,
    CONTRACT_NAME,
    CONTRACT_SHA256,
    EFFECTIVE_CONTRACT_BYTES,
    F23_STAGES,
    FEATURE_MEMBER_SCHEMA,
    JOB_NAMES,
    PROPOSAL_SHA256,
    SEEDS,
    ReasonCode,
)
from pams.temporac.hashio import (
    ArchiveError,
    H,
    HashIOError,
    canonical_json_bytes,
    component_key,
    deterministic_npz_bytes,
    effective_contract_index_bytes,
    opaque_sample_key,
    parse_effective_contract_index_bytes,
    parse_strict_json_bytes,
    read_deterministic_npz,
    receipt_owner_key,
    write_bytes_exclusive,
)
from pams.temporac.types import (
    CheckpointReceiptOwner,
    CheckpointRecord,
    FeatureRecord,
    IdentityReceiptOwner,
    NamedReceiptOwner,
    PredictionReceiptOwner,
    ResourceRecord,
    RunReceiptOwner,
    SchemaReceiptOwner,
    SeedReceiptOwner,
    TargetReceiptOwner,
)


def _feature() -> FeatureRecord:
    return FeatureRecord(
        frame_mask=np.ones(320, dtype="|u1"),
        local_person_slot=np.asarray([2], dtype="<i8"),
        motion=np.zeros((320, 17, 3), dtype="<f4"),
        opaque_sample_key=np.arange(32, dtype="|u1"),
        person_mask=np.asarray([1], dtype="|u1"),
        sampled_frame_indices=np.arange(320, dtype="<i8"),
        source_length=np.asarray([320], dtype="<i8"),
    )


def test_contract_inventory_and_reasons_are_exact() -> None:
    assert CONTRACT_NAME == "temporac.execution.v4"
    assert PROPOSAL_SHA256 == "3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391"
    assert CONTRACT_SHA256 == "c203f4aab00879acf041b90829f0124d0b1de231294f191b03f3166f728e3d5c"
    assert SEEDS == (20260815, 20260816, 20260817)
    assert len(JOB_NAMES) == len(set(JOB_NAMES)) == 27
    assert ALLOCATED_A6000_HOURS == 93.0
    assert tuple(int(reason) for reason in ReasonCode) == tuple(range(1, 25))
    assert F23_STAGES[0] == "S0"
    assert F23_STAGES[-1] == "K7"
    assert "capability grant" in F23_STAGES


def test_effective_contract_index_is_the_exact_accepted_five_row_preimage() -> None:
    encoded = effective_contract_index_bytes()
    assert len(encoded) == EFFECTIVE_CONTRACT_BYTES == 665
    assert hashlib.sha256(encoded).hexdigest() == CONTRACT_SHA256
    rows = parse_effective_contract_index_bytes(encoded)
    assert tuple(row.role for row in rows) == (
        "canonical_proposal",
        "amendment_markdown",
        "amendment_json",
        "accepted_review_markdown",
        "accepted_review_json",
    )
    assert rows[0].sha256 == PROPOSAL_SHA256
    decoded = json.loads(encoded)
    decoded["rows"][0]["bytes"] += 1
    with pytest.raises(HashIOError, match="differs"):
        parse_effective_contract_index_bytes(canonical_json_bytes(decoded))


@pytest.mark.parametrize(
    "payload",
    (
        b'{"a":1,"a":1}\n',
        b'{"a":null,"extra":0}\r\n',
        b'{"a":NaN}\n',
        b'\xef\xbb\xbf{"a":1}\n',
        b'{ "a":1}\n',
    ),
)
def test_strict_canonical_json_parser_rejects_duplicate_nonfinite_and_transport_attacks(
    payload: bytes,
) -> None:
    with pytest.raises(HashIOError):
        parse_strict_json_bytes(payload, canonical=True)


def test_h_is_length_prefixed_and_canonical_json_has_one_lf() -> None:
    expected = hashlib.sha256(
        b"example.tag\0" + struct.pack(">Q", 1) + b"a" + struct.pack(">Q", 2) + b"bc"
    ).digest()
    assert H("example.tag", b"a", b"bc") == expected
    assert H("example.tag", b"a", b"bc") != H("example.tag", b"ab", b"c")
    payload = canonical_json_bytes({"z": "雪", "a": 1})
    assert payload == b'{"a":1,"z":"\xe9\x9b\xaa"}\n'
    assert json.loads(payload) == {"a": 1, "z": "雪"}
    with pytest.raises(HashIOError):
        canonical_json_bytes({"x": float("nan")})


def test_canonical_json_rejects_negative_zero_recursively_but_accepts_positive_zero() -> None:
    assert canonical_json_bytes(0.0) == b"0.0\n"
    assert parse_strict_json_bytes(b'{"a":[0.0,{"b":0.0}]}\n', canonical=True) == {
        "a": [0.0, {"b": 0.0}]
    }
    for value in (-0.0, {"a": -0.0}, {"a": [0.0, {"b": -0.0}]}):
        with pytest.raises(HashIOError, match="negative-zero"):
            canonical_json_bytes(value)
    for encoded in (b"-0.0\n", b'{"a":-0.0}\n', b'{"a":[0.0,{"b":-0.0}]}\n'):
        with pytest.raises(HashIOError, match="negative-zero"):
            parse_strict_json_bytes(encoded, canonical=True)


def test_resource_and_checkpoint_records_reject_negative_zero() -> None:
    resource = ResourceRecord(
        completed_steps=500,
        gpu_seconds=0.0,
        max_cuda_bytes=0,
        wall_seconds=0.0,
    )
    CheckpointRecord(
        completed_step=500,
        artifact_sha256="0" * 64,
        tune_objective=0.0,
        resource=resource,
    )
    with pytest.raises(ValueError, match="finite nonnegative"):
        ResourceRecord(
            completed_steps=500,
            gpu_seconds=-0.0,
            max_cuda_bytes=0,
            wall_seconds=0.0,
        )
    with pytest.raises(ValueError, match="finite nonnegative"):
        ResourceRecord(
            completed_steps=500,
            gpu_seconds=0.0,
            max_cuda_bytes=0,
            wall_seconds=-0.0,
        )
    with pytest.raises(ValueError, match="finite float64"):
        CheckpointRecord(
            completed_step=500,
            artifact_sha256="0" * 64,
            tune_objective=-0.0,
            resource=resource,
        )


def test_receipt_owner_keys_use_exact_class_specific_typed_preimages() -> None:
    opaque = bytes(range(32)).hex()
    vectors = (
        (
            "feature",
            IdentityReceiptOwner(split="train", opaque_key_hex=opaque, slot=7),
            "c8a4dc8a65edfa24950638ee30b1709575c04cf01044aa3afe4eafd2860972d3",
        ),
        (
            "natural-certificate-input",
            IdentityReceiptOwner(split="val", opaque_key_hex=opaque, slot=7),
            "9c66114c122c9794119f672257cb708d34c1d193557187068b39702e0a8b2b91",
        ),
        (
            "teacher-checkpoint",
            CheckpointReceiptOwner(seed=20260815, step=500),
            "4a8d41a08044d2093bec9132c380dbbb06273f40866c964a732b7566493ac9d3",
        ),
        (
            "teacher-tune-evaluation",
            CheckpointReceiptOwner(seed=20260815, step=500),
            "b4a57f5678d91ee56e2e06bb3545e309dab9c681cf715960b6c3469cfa6a18e8",
        ),
        (
            "run",
            RunReceiptOwner(job_name="temporac.execution.v4/teacher/seed=20260815", seed=20260815),
            "5e2a69ecb82c77843d6aaa73928e62ddcc16f679b54511004757c5fcfed06b77",
        ),
        (
            "prediction",
            PredictionReceiptOwner(
                split="val",
                opaque_key_hex=opaque,
                slot=7,
                arm="local",
                seed=20260815,
                condition="natural-clean",
            ),
            "e5e0006f81e87e00ad4f406bd4ae21cc16be1920dece7cd7ecca492cc4d8d32f",
        ),
        (
            "target",
            TargetReceiptOwner(source_kind=0, source_key_hex=opaque, source_unit_index=7),
            "63b62f423eba8f56c813177c4cd6cdfab32b396131b008b182e2b0797010ec96",
        ),
        (
            "x0-inference",
            SeedReceiptOwner(seed=20260815),
            "8e1f019d741495aaf94799f6e29efceafaf4d64689b615952373823849d86623",
        ),
        (
            "natural-prediction-completion",
            SeedReceiptOwner(seed=20260815),
            "eb7a3856abcfc5b7f00bc2ff1817978adfe1bf4097ca98fd670f6bc5c4378f64",
        ),
        (
            "pre-g5a-stage",
            NamedReceiptOwner(owner="P00-IMPLEMENT"),
            "295e6272fabc305a38c5a645c6aa7c7ab76bafdc6f9325fcf2ef1eceb90621c7",
        ),
        (
            "g1-teacher-selection",
            NamedReceiptOwner(owner="G1-TEACHER-AGG"),
            "95e8454b83bd9211ab9355546d670486b71a63b54c42d9eafe8f15e2fec3f5fc",
        ),
        (
            "k1-certificate-outcome",
            NamedReceiptOwner(owner="K1-COVERAGE"),
            "3dfcc2dabaee09c8aff668d5e8d8204cdfb73566bd535692a82cf2b4976642c1",
        ),
        (
            "k3",
            NamedReceiptOwner(owner="K3-ROUTE"),
            "5fcb9f226f5e97fc9a5e4928bcc4d7efeabbbcce5d9b9e7cda0b83068c3f3f7f",
        ),
        (
            "k4",
            NamedReceiptOwner(owner="K4-DIAG"),
            "4d32ccf8daeb500cc62137adaae7582b2dacba26decc66c021b435f95538aba4",
        ),
        (
            "teacher-tune-input",
            SchemaReceiptOwner(schema="temporac.teacher-tune-input-receipt.v4"),
            "1eaccb83bc9c94bcc652520c23319426667579e9eb278bb673b238e44d839e04",
        ),
        (
            "teacher-selection",
            SchemaReceiptOwner(schema="temporac.teacher-selection-receipt.v4"),
            "51f19c7487b052e5e247892647aef3d519c269513b48c8b78795eedf7f0a85c5",
        ),
    )
    assert tuple(receipt_owner_key(node_class, owner) for node_class, owner, _ in vectors) == tuple(
        expected for _node_class, _owner, expected in vectors
    )
    with pytest.raises(HashIOError, match="fields do not match"):
        receipt_owner_key("feature", NamedReceiptOwner(owner="P00-IMPLEMENT"))
    with pytest.raises(HashIOError, match="frozen job/seed"):
        receipt_owner_key(
            "run", RunReceiptOwner(job_name="arbitrary/slash/seed=20260815", seed=20260815)
        )
    with pytest.raises(HashIOError, match="exact class schema"):
        receipt_owner_key(
            "teacher-selection",
            SchemaReceiptOwner(schema="temporac.teacher-tune-input-receipt.v4"),
        )


def test_opaque_key_uses_the_non_h_frozen_preimage() -> None:
    source = bytes(range(32))
    expected = hashlib.sha256(
        b"temporac.opaque-key.v4\0train\0" + source + struct.pack(">I", 7)
    ).digest()
    assert opaque_sample_key("train", source, 7) == expected
    component_expected = hashlib.sha256(
        b"temporac.component.v4\0train\0" + bytes(32) + bytes([1]) * 32
    ).digest()
    assert component_key("train", [bytes([1]) * 32, bytes(32), bytes(32)]) == component_expected


def test_feature_record_is_shape_checked_copied_and_readonly() -> None:
    source = np.zeros((320, 17, 3), dtype="<f4")
    record = _feature()
    assert not record.motion.flags.writeable
    source[0, 0, 0] = 9.0
    assert record.motion[0, 0, 0] == 0.0
    with pytest.raises(ValueError):
        record.motion[0, 0, 0] = 1.0
    with pytest.raises(ValueError):
        record.motion.flags.writeable = True
    with pytest.raises(ValueError, match="shape"):
        FeatureRecord(
            frame_mask=np.ones(319, dtype="|u1"),
            local_person_slot=np.asarray([0], dtype="<i8"),
            motion=source,
            opaque_sample_key=np.zeros(32, dtype="|u1"),
            person_mask=np.asarray([1], dtype="|u1"),
            sampled_frame_indices=np.arange(320, dtype="<i8"),
            source_length=np.asarray([320], dtype="<i8"),
        )


def test_npz_is_npy_v2_stored_sorted_fixed_and_round_trips(tmp_path: Path) -> None:
    record = _feature()
    first, members = deterministic_npz_bytes(record.as_arrays(), schema=FEATURE_MEMBER_SCHEMA)
    second, _ = deterministic_npz_bytes(record.as_arrays(), schema=FEATURE_MEMBER_SCHEMA)
    assert first == second
    assert [member.name for member in members] == sorted(member.name for member in members)
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert archive.comment == b""
        assert archive.namelist() == [f"{name}.npy" for name in sorted(FEATURE_MEMBER_SCHEMA)]
        for info in archive.infolist():
            assert info.compress_type == zipfile.ZIP_STORED
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.external_attr == 0
            assert info.create_system == 0
            assert archive.read(info).startswith(b"\x93NUMPY\x02\x00")
    path = tmp_path / "identity.npz"
    write_bytes_exclusive(path, first)
    arrays, observed_members = read_deterministic_npz(path, schema=FEATURE_MEMBER_SCHEMA)
    assert tuple(arrays) == tuple(sorted(FEATURE_MEMBER_SCHEMA))
    assert np.array_equal(arrays["motion"], record.motion)
    for array in arrays.values():
        with pytest.raises(ValueError):
            array.flags.writeable = True
    assert observed_members == members
    with pytest.raises(HashIOError, match="overwrite"):
        write_bytes_exclusive(path, first)


def test_npz_reader_rejects_noncanonical_python_zip_metadata(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, (dtype, shape) in sorted(FEATURE_MEMBER_SCHEMA.items()):
            actual_shape = tuple(1 if axis is None else axis for axis in shape)
            array = np.zeros(actual_shape, dtype=dtype)
            handle = io.BytesIO()
            np.lib.format.write_array(handle, array, version=(2, 0), allow_pickle=False)
            archive.writestr(f"{name}.npy", handle.getvalue())
    with pytest.raises(ArchiveError, match="metadata"):
        read_deterministic_npz(path, schema=FEATURE_MEMBER_SCHEMA)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="platform has no symlink support")
def test_regular_reader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"x")
    link = tmp_path / "link"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is not permitted")
    from pams.temporac.hashio import read_regular_file

    with pytest.raises(HashIOError, match="non-symlink"):
        read_regular_file(link)
