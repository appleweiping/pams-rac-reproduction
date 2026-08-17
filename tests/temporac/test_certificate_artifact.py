from __future__ import annotations

import io
import json
import zipfile
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import pams.temporac.certify as certify_module
import pams.temporac.types as types_module
from pams.temporac.certify import (
    CertificateContractError,
    CertificateTrack,
    NaturalTeacherOutput,
    bind_natural_certificate_track,
    build_natural_teacher_output,
    certificate_track_from_x0,
    certify_target,
)
from pams.temporac.contract import CONTRACT_SHA256, TARGET_MEMBER_SCHEMA, TARGET_RECEIPT_SCHEMA
from pams.temporac.hashio import (
    ArchiveError,
    canonical_json_bytes,
    deterministic_npz_bytes,
    feature_npz_bytes,
    read_target_npz_bytes,
    sha256_bytes,
    target_npz_bytes,
    x0_source_key,
)
from pams.temporac.preprocess import preprocess_identity
from pams.temporac.receipts import (
    ReceiptError,
    build_target_artifact,
    load_target_artifact,
    member_payload,
    parse_receipt_bytes,
    receipt_bytes,
)
from pams.temporac.types import ContractError, FeatureRecord, TargetProvenance
from pams.temporac.x0 import X0View, generate_view


def _ideal_track() -> CertificateTrack:
    sample_count = 345
    clock = np.arange(sample_count, dtype=np.float64)
    theta = (clock - 8.0) / 32.0
    rho = certify_module.LANDMARK_RHO / np.linalg.norm(certify_module.LANDMARK_RHO)
    orthogonal = np.arange(1.0, 67.0)
    orthogonal -= rho * float(np.dot(orthogonal, rho))
    orthogonal /= np.linalg.norm(orthogonal)
    geometry = 0.3 * (
        np.cos(2.0 * np.pi * theta)[:, None] * rho
        + np.sin(2.0 * np.pi * theta)[:, None] * orthogonal
    )
    continuous = np.zeros((sample_count, 149), dtype=np.float64)
    continuous[:, :66] = geometry
    continuous[:, 66:] = 0.3 * np.sin(2.0 * np.pi * theta)[:, None]
    phase = np.stack((np.cos(2.0 * np.pi * theta), np.sin(2.0 * np.pi * theta)), axis=1)
    pulse = np.zeros(sample_count - 1, dtype=np.uint8)
    chi = np.zeros(sample_count - 1, dtype=np.float64)
    for left in range(8, 328, 32):
        pulse[left] = 1
        chi[left : left + 32] = 1.0 / 32.0
    return CertificateTrack(
        geometry=geometry,
        continuous=continuous,
        continuous_mask=np.ones((sample_count, 149), dtype=np.uint8),
        phase=phase,
        reconstruction=continuous.copy(),
        run_bounds=np.asarray([[0, sample_count]], dtype=np.int32),
        analytic_pulse=pulse,
        analytic_chi=chi,
    )


def _feature() -> FeatureRecord:
    view = generate_view(3, 0, resampler="linear", offset=0)
    motion = np.asarray(view.motion[:320], dtype="<f4").copy()
    motion[312:] = 0.0
    frame_mask = np.ones(320, dtype="|u1")
    frame_mask[312:] = 0
    return FeatureRecord(
        frame_mask=frame_mask,
        local_person_slot=np.asarray([7], dtype="<i8"),
        motion=motion,
        opaque_sample_key=np.arange(32, dtype="|u1"),
        person_mask=np.asarray([1], dtype="|u1"),
        sampled_frame_indices=np.asarray(view.clock[:320], dtype="<i8"),
        source_length=np.asarray([int(view.clock[-1]) + 1], dtype="<i8"),
    )


def _feature_receipt(feature: FeatureRecord) -> tuple[bytes, str]:
    artifact, members = feature_npz_bytes(feature)
    payload: dict[str, object] = {
        "artifact_bytes": len(artifact),
        "artifact_sha256": sha256_bytes(artifact),
        "contract_sha256": CONTRACT_SHA256,
        "members": member_payload(members),
        "opaque_key_hex": feature.opaque_key_bytes.hex(),
        "schema": "temporac.feature-receipt.v4",
        "slot": feature.slot,
        "source_binding_sha256": "a" * 64,
    }
    encoded = receipt_bytes(payload, expected_schema="temporac.feature-receipt.v4")
    return encoded, sha256_bytes(encoded)


def _teacher_output(
    feature: FeatureRecord, *, teacher_sha256: str = "1" * 64
) -> NaturalTeacherOutput:
    feature_receipt, feature_receipt_sha256 = _feature_receipt(feature)
    preprocessed = preprocess_identity(
        feature.motion,
        feature.frame_mask,
        feature.sampled_frame_indices,
    )
    view = generate_view(3, 0, resampler="linear", offset=0)
    phase = np.asarray(view.phase[: preprocessed.motion.shape[0]], dtype="<f8")
    reconstruction = np.asarray(preprocessed.teacher_input[:, :149], dtype="<f8")
    return build_natural_teacher_output(
        feature=feature,
        feature_receipt_bytes=feature_receipt,
        expected_feature_receipt_sha256=feature_receipt_sha256,
        selected_teacher_sha256=teacher_sha256,
        phase=phase,
        reconstruction=reconstruction,
    )


def _natural_track() -> tuple[FeatureRecord, NaturalTeacherOutput, CertificateTrack]:
    feature = _feature()
    teacher_output = _teacher_output(feature)
    track = bind_natural_certificate_track(
        feature=feature,
        teacher_output=teacher_output,
    )
    return feature, teacher_output, track


def _natural_target() -> tuple[FeatureRecord, Any]:
    feature, _, track = _natural_track()
    target = certify_target(track, "natural")
    assert target.certified
    return feature, target


def _write_pair(
    directory: Path,
    artifact: bytes,
    encoded_receipt: bytes,
) -> tuple[Path, Path]:
    artifact_path = directory / "target.npz"
    receipt_path = directory / "target.receipt.json"
    artifact_path.write_bytes(artifact)
    receipt_path.write_bytes(encoded_receipt)
    return artifact_path, receipt_path


def test_real_natural_certificate_builds_exact_npz_receipt_and_loads(tmp_path: Path) -> None:
    feature, target = _natural_target()
    first = build_target_artifact(target)
    second = build_target_artifact(target)
    assert first.artifact_bytes == second.artifact_bytes
    assert first.receipt_bytes == second.receipt_bytes
    assert not target.chi.flags.writeable
    for array in (
        target.pulse,
        target.chi,
        target.target_mask,
        target.edge_mask,
        target.decoder_mask,
    ):
        with pytest.raises(ValueError):
            array.flags.writeable = True

    expected_names = [f"{name}.npy" for name in sorted(TARGET_MEMBER_SCHEMA)]
    with zipfile.ZipFile(io.BytesIO(first.artifact_bytes)) as archive:
        assert archive.namelist() == expected_names
    arrays, members = read_target_npz_bytes(first.artifact_bytes)
    assert tuple(arrays) == tuple(sorted(TARGET_MEMBER_SCHEMA))
    assert arrays["chi"].dtype == np.dtype("<f8")
    assert arrays["chi"].shape == (311,)
    assert arrays["contract_sha256"].tobytes() == bytes.fromhex(CONTRACT_SHA256)
    assert arrays["edge_mask"].dtype == np.dtype("|u1")
    assert arrays["pulse"].dtype == np.dtype("|u1")
    assert arrays["source_kind"].tolist() == [1]
    assert arrays["target_mask"].dtype == np.dtype("|u1")
    assert arrays["teacher_sha256"].tobytes() == bytes.fromhex("1" * 64)
    assert members == first.members

    receipt = parse_receipt_bytes(first.receipt_bytes, expected_schema=TARGET_RECEIPT_SCHEMA)
    assert set(receipt) == {
        "artifact_bytes",
        "artifact_sha256",
        "certificate_status",
        "contract_sha256",
        "members",
        "schema",
        "source_key_hex",
        "source_kind",
        "source_unit_index",
        "teacher_sha256",
    }
    assert receipt["artifact_bytes"] == len(first.artifact_bytes)
    assert receipt["artifact_sha256"] == first.artifact_sha256
    assert receipt["certificate_status"] == "CERTIFIED"
    assert receipt["source_key_hex"] == feature.opaque_key_bytes.hex()
    assert receipt["source_kind"] == 1
    assert receipt["source_unit_index"] == feature.slot
    assert receipt["teacher_sha256"] == "1" * 64

    artifact_path, receipt_path = _write_pair(tmp_path, first.artifact_bytes, first.receipt_bytes)
    loaded = load_target_artifact(
        artifact_path,
        receipt_path,
        expected_receipt_sha256=first.receipt_sha256,
    )
    assert loaded.artifact_sha256 == first.artifact_sha256
    assert loaded.receipt_sha256 == first.receipt_sha256
    assert np.array_equal(loaded.arrays["pulse"], target.pulse)
    for array in loaded.arrays.values():
        with pytest.raises(ValueError):
            array.flags.writeable = True


@pytest.mark.parametrize("field", ("pulse", "chi", "target_mask", "edge_mask"))
def test_builder_revalidates_any_post_certificate_ledger_tamper(field: str) -> None:
    _, target = _natural_target()
    changed = np.array(getattr(target, field), copy=True)
    if field == "pulse":
        first = int(np.flatnonzero(changed)[0])
        changed[first] = 0
        changed[first + 1] = 1
    elif field == "chi":
        first = int(np.flatnonzero(target.pulse)[0])
        changed[first] += 0.01
    else:
        first = int(np.flatnonzero(changed)[0])
        changed[first] = 0
    object.__setattr__(target, field, changed)
    with pytest.raises(ContractError):
        build_target_artifact(target)


def test_joint_loader_rejects_shifted_pulse_bytes_and_loaded_arrays_are_immutable(
    tmp_path: Path,
) -> None:
    _, target = _natural_target()
    valid = build_target_artifact(target)
    arrays = {name: np.array(array, copy=True) for name, array in target.artifact_arrays().items()}
    first = int(np.flatnonzero(arrays["pulse"])[0])
    arrays["pulse"][first] = 0
    arrays["pulse"][first + 1] = 1
    tampered, _ = deterministic_npz_bytes(arrays, schema=TARGET_MEMBER_SCHEMA)
    artifact_path, receipt_path = _write_pair(tmp_path, tampered, valid.receipt_bytes)
    with pytest.raises(ReceiptError, match="do not agree"):
        load_target_artifact(
            artifact_path,
            receipt_path,
            expected_receipt_sha256=valid.receipt_sha256,
        )


def test_provenance_cannot_be_rebound_for_natural_x0_or_teacher() -> None:
    feature, teacher_output, natural_track = _natural_track()
    natural = certify_target(natural_track, "natural")
    assert natural.certified
    assert natural.provenance is not None
    with pytest.raises(TypeError):
        bind_natural_certificate_track(
            track=natural_track,  # type: ignore[call-arg]
            feature=feature,
            teacher_output=teacher_output,
        )
    with pytest.raises(ContractError, match="canonical source adapter"):
        TargetProvenance(
            source_kind=1,
            source_key_hex="3" * 64,
            source_unit_index=8,
            teacher_sha256="2" * 64,
        )
    other_feature = replace(feature, local_person_slot=np.asarray([8], dtype="<i8"))
    with pytest.raises(CertificateContractError, match="canonical feature"):
        bind_natural_certificate_track(feature=other_feature, teacher_output=teacher_output)
    with pytest.raises(CertificateContractError, match="canonical adapter-input builder"):
        NaturalTeacherOutput(
            phase=teacher_output.phase,
            reconstruction=teacher_output.reconstruction,
            input_receipt=teacher_output.input_receipt,
            input_receipt_bytes=teacher_output.input_receipt_bytes,
            input_receipt_sha256=teacher_output.input_receipt_sha256,
        )
    other_teacher_output = _teacher_output(feature, teacher_sha256="2" * 64)
    other_teacher = bind_natural_certificate_track(
        feature=feature,
        teacher_output=other_teacher_output,
    )
    assert other_teacher.provenance is not None
    with pytest.raises(ContractError, match="produced by certify_target"):
        replace(natural, provenance=other_teacher.provenance)
    with pytest.raises(FrozenInstanceError):
        natural.provenance.teacher_sha256 = "2" * 64  # type: ignore[misc]
    with pytest.raises(TypeError):
        build_target_artifact(natural, teacher_sha256="2" * 64)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        build_target_artifact(natural, source_key_hex="3" * 64)  # type: ignore[call-arg]

    view = generate_view(3, 0, resampler="linear", offset=0)
    x0_track = certificate_track_from_x0(view, teacher_sha256="4" * 64)
    assert x0_track.provenance is not None
    assert x0_track.provenance.source_key_hex == x0_source_key(3).hex()
    assert x0_track.provenance.source_unit_index == 0
    assert x0_track.provenance.teacher_sha256 == "4" * 64
    other_view = generate_view(4, 0, resampler="linear", offset=0)
    other_x0 = certificate_track_from_x0(other_view, teacher_sha256="5" * 64)
    assert other_x0.provenance is not None
    with pytest.raises(CertificateContractError, match="canonical source adapter"):
        replace(x0_track, provenance=other_x0.provenance)
    x0_target = certify_target(x0_track, "X0")
    assert x0_target.certified
    with pytest.raises(ContractError, match="produced by certify_target"):
        replace(x0_target, provenance=other_x0.provenance)
    x0_receipt = parse_receipt_bytes(build_target_artifact(x0_target).receipt_bytes)
    assert x0_receipt["source_key_hex"] == x0_source_key(3).hex()
    assert x0_receipt["source_unit_index"] == 0
    assert x0_receipt["teacher_sha256"] == "4" * 64

    with pytest.raises(CertificateContractError, match="real canonical X0View"):
        certificate_track_from_x0(cast(Any, object()), teacher_sha256="4" * 64)
    unused = generate_view(3, 0, resampler="sinc", offset=31)
    with pytest.raises(CertificateContractError, match="unused"):
        certificate_track_from_x0(unused, teacher_sha256="4" * 64)
    changed_phase = np.array(view.phase, copy=True)
    changed_phase[0] *= -1.0
    noncanonical = replace(view, phase=changed_phase)
    assert type(noncanonical) is X0View
    with pytest.raises(CertificateContractError, match="canonical generate_view"):
        certificate_track_from_x0(noncanonical, teacher_sha256="4" * 64)


def test_abstention_and_unbound_certificate_have_no_artifact_or_receipt() -> None:
    feature = _feature()
    teacher_output = _teacher_output(feature)
    broken_phase = np.zeros_like(teacher_output.phase)
    feature_receipt, feature_receipt_sha256 = _feature_receipt(feature)
    broken_output = build_natural_teacher_output(
        feature=feature,
        feature_receipt_bytes=feature_receipt,
        expected_feature_receipt_sha256=feature_receipt_sha256,
        selected_teacher_sha256="1" * 64,
        phase=broken_phase,
        reconstruction=teacher_output.reconstruction,
    )
    bound = bind_natural_certificate_track(
        feature=feature,
        teacher_output=broken_output,
    )
    abstained = certify_target(bound, "natural")
    assert abstained.status == "ABSTAIN"
    assert abstained.provenance == bound.provenance
    with pytest.raises(ContractError, match="no target artifact"):
        target_npz_bytes(abstained)
    with pytest.raises(ReceiptError, match="no target artifact or target receipt"):
        build_target_artifact(abstained)

    unbound = certify_target(_ideal_track(), "X0")
    assert unbound.certified and unbound.provenance is None
    with pytest.raises(ReceiptError, match="unbound certificate"):
        build_target_artifact(unbound)


def test_loader_rejects_empty_superset_and_hash_tamper(tmp_path: Path) -> None:
    _, target = _natural_target()
    valid = build_target_artifact(target)
    valid_receipt = dict(parse_receipt_bytes(valid.receipt_bytes))

    empty_arrays = dict(target.artifact_arrays())
    for name, dtype in (
        ("chi", "<f8"),
        ("edge_mask", "|u1"),
        ("pulse", "|u1"),
        ("target_mask", "|u1"),
    ):
        empty_arrays[name] = np.empty(0, dtype=dtype)
    empty_artifact, empty_members = deterministic_npz_bytes(
        empty_arrays,
        schema=TARGET_MEMBER_SCHEMA,
    )
    empty_receipt = dict(valid_receipt)
    empty_receipt.update(
        artifact_bytes=len(empty_artifact),
        artifact_sha256=sha256_bytes(empty_artifact),
        members=member_payload(empty_members),
    )
    empty_receipt_bytes = receipt_bytes(empty_receipt, expected_schema=TARGET_RECEIPT_SCHEMA)
    artifact_path, receipt_path = _write_pair(tmp_path, empty_artifact, empty_receipt_bytes)
    with pytest.raises(ArchiveError, match="at least one edge"):
        load_target_artifact(
            artifact_path,
            receipt_path,
            expected_receipt_sha256=sha256_bytes(empty_receipt_bytes),
        )

    superset_arrays = dict(target.artifact_arrays())
    superset_arrays["unexpected"] = np.asarray([1], dtype="|u1")
    superset, _ = deterministic_npz_bytes(superset_arrays)
    artifact_path.write_bytes(superset)
    receipt_path.write_bytes(valid.receipt_bytes)
    with pytest.raises(ArchiveError, match="exact bytewise-sorted schema"):
        load_target_artifact(
            artifact_path,
            receipt_path,
            expected_receipt_sha256=valid.receipt_sha256,
        )

    tampered_receipt = json.loads(valid.receipt_bytes)
    tampered_receipt["artifact_sha256"] = "0" * 64
    tampered_bytes = receipt_bytes(tampered_receipt, expected_schema=TARGET_RECEIPT_SCHEMA)
    artifact_path.write_bytes(valid.artifact_bytes)
    receipt_path.write_bytes(tampered_bytes)
    with pytest.raises(ReceiptError, match="do not agree"):
        load_target_artifact(
            artifact_path,
            receipt_path,
            expected_receipt_sha256=sha256_bytes(tampered_bytes),
        )
    with pytest.raises(ReceiptError, match="expected receipt hash"):
        load_target_artifact(
            artifact_path,
            receipt_path,
            expected_receipt_sha256=valid.receipt_sha256,
        )

    member_tamper = json.loads(valid.receipt_bytes)
    member_tamper["members"][0]["sha256"] = "f" * 64
    member_tamper_bytes = receipt_bytes(member_tamper, expected_schema=TARGET_RECEIPT_SCHEMA)
    receipt_path.write_bytes(member_tamper_bytes)
    with pytest.raises(ReceiptError, match="do not agree"):
        load_target_artifact(
            artifact_path,
            receipt_path,
            expected_receipt_sha256=sha256_bytes(member_tamper_bytes),
        )

    receipt_superset = json.loads(valid.receipt_bytes)
    receipt_superset["unexpected"] = True
    receipt_superset_bytes = canonical_json_bytes(receipt_superset)
    receipt_path.write_bytes(receipt_superset_bytes)
    with pytest.raises(ReceiptError, match="unknown/missing keys"):
        load_target_artifact(
            artifact_path,
            receipt_path,
            expected_receipt_sha256=sha256_bytes(receipt_superset_bytes),
        )


@pytest.mark.parametrize(
    "attack",
    [
        "nonfinite_chi",
        "negative_chi",
        "mask_disagreement",
        "nonbinary_mask",
        "pulse_outside_mask",
        "chi_off_mask",
        "zero_chi_on_pulse",
    ],
)
def test_target_reader_rejects_every_semantic_artifact_attack(attack: str) -> None:
    _, target = _natural_target()
    arrays = {name: array.copy() for name, array in target.artifact_arrays().items()}
    pulse_edge = int(np.flatnonzero(arrays["pulse"])[0])
    if attack == "nonfinite_chi":
        arrays["chi"][pulse_edge] = np.nan
    elif attack == "negative_chi":
        arrays["chi"][pulse_edge] = -1.0
    elif attack == "mask_disagreement":
        arrays["edge_mask"][0] = 1
    elif attack == "nonbinary_mask":
        arrays["edge_mask"][pulse_edge] = 2
    elif attack == "pulse_outside_mask":
        arrays["pulse"][0] = 1
    elif attack == "chi_off_mask":
        arrays["chi"][0] = 0.1
    elif attack == "zero_chi_on_pulse":
        arrays["chi"][pulse_edge] = 0.0
    artifact, _ = deterministic_npz_bytes(arrays, schema=TARGET_MEMBER_SCHEMA)
    with pytest.raises(ArchiveError, match="target"):
        read_target_npz_bytes(artifact)


def test_certificate_model_has_no_duplicate_result_or_target_record() -> None:
    assert not hasattr(certify_module, "CertificateResult")
    assert not hasattr(types_module, "TargetRecord")
    assert certify_module.CertifiedTarget is types_module.CertifiedTarget
