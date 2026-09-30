"""Immutable per-identity prediction/stub construction with one decode."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from pams.temporac.contract import (
    CANONICAL_JOB_NAMES,
    CAPACITY_CONTROL_JOB_NAMES,
    CONTRACT_SHA256,
    PREDICTION_RECEIPT_SCHEMA,
    SEEDS,
)
from pams.temporac.decode import decode_identity
from pams.temporac.hashio import prediction_npz_bytes, sha256_bytes
from pams.temporac.receipts import member_payload, receipt_bytes
from pams.temporac.types import ArrayMemberRecord, ContractError, PredictionRecord

PredictionCondition = Literal["natural-clean", "natural-drift"]
PredictionArm = Literal["local", "global", "uniform", "capacity-control"]
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class PredictionIdentity:
    """The opaque identity tuple permitted in a prediction artifact."""

    opaque_sample_key: bytes
    local_person_slot: int

    def __post_init__(self) -> None:
        if not isinstance(self.opaque_sample_key, bytes) or len(self.opaque_sample_key) != 32:
            raise ContractError("prediction identity key must contain exactly 32 raw bytes")
        if type(self.local_person_slot) is not int or self.local_person_slot < 0:
            raise ContractError("prediction local_person_slot must be nonnegative")


def build_identity_prediction(
    identity: PredictionIdentity,
    *,
    condition: PredictionCondition,
    response: ArrayLike | None = None,
    edge_mask: ArrayLike | None = None,
    decoder_mask: ArrayLike | None = None,
    run_bounds: ArrayLike | None = None,
    abstain_reasons: ArrayLike = (),
) -> PredictionRecord:
    """Decode exactly once and construct the fourteen immutable members."""

    if condition not in {"natural-clean", "natural-drift"}:
        raise ContractError("prediction condition is outside the two frozen conditions")
    raw_reasons = np.asarray(abstain_reasons)
    abstaining = bool(raw_reasons.size)
    supplied = (response, edge_mask, decoder_mask, run_bounds)
    if abstaining and any(value is not None for value in supplied):
        raise ContractError("an abstention cannot carry a partial response target")
    if not abstaining and any(value is None for value in supplied):
        raise ContractError("a non-abstention requires response, both masks, and runs")

    decoded = decode_identity(
        response,
        decoder_mask,
        run_bounds,
        abstain_reasons=raw_reasons,
    )
    if decoded.decoder_invocations != 1:
        raise ContractError("identity prediction did not invoke the decoder exactly once")
    if abstaining:
        canonical_edge_mask = np.empty(0, dtype=np.uint8)
    else:
        edge_raw = np.asarray(edge_mask)
        if (
            edge_raw.shape != decoded.response.shape
            or edge_raw.dtype.kind not in "bu"
            or not np.all((edge_raw == 0) | (edge_raw == 1))
        ):
            raise ContractError("edge_mask must be binary and match the decoded response")
        canonical_edge_mask = np.ascontiguousarray(edge_raw, dtype=np.uint8)
        if np.any(decoded.decoder_mask > canonical_edge_mask):
            raise ContractError("decoder_mask cannot exceed edge_mask support")
        if np.any(decoded.response[canonical_edge_mask == 0] != np.float32(0.0)):
            raise ContractError("serialized response must be canonical zero outside edge_mask")

    return PredictionRecord(
        abstain=np.asarray([int(decoded.abstain)], dtype="|u1"),
        abstain_reasons=np.asarray(decoded.abstain_reasons, dtype="<u2"),
        component_bounds=np.asarray(decoded.component_bounds, dtype="<i4"),
        component_location=np.asarray(decoded.component_location, dtype="<i4"),
        component_score=np.asarray(decoded.component_score, dtype="<f4"),
        condition=np.asarray([0 if condition == "natural-clean" else 1], dtype="|u1"),
        contract_sha256=np.frombuffer(bytes.fromhex(CONTRACT_SHA256), dtype="|u1"),
        count=np.asarray([decoded.count], dtype="<i8"),
        decoder_mask=np.asarray(decoded.decoder_mask, dtype="|u1"),
        edge_mask=canonical_edge_mask,
        local_person_slot=np.asarray([identity.local_person_slot], dtype="<i8"),
        opaque_sample_key=np.frombuffer(identity.opaque_sample_key, dtype="|u1"),
        response=np.asarray(decoded.response, dtype="<f4"),
        run_bounds=np.asarray(decoded.run_bounds, dtype="<i4"),
    )


@dataclass(frozen=True, slots=True)
class PredictionArtifact:
    """In-memory immutable artifact and canonical detached receipt bytes."""

    record: PredictionRecord
    artifact_bytes: bytes
    artifact_sha256: str
    members: tuple[ArrayMemberRecord, ...]
    receipt_bytes: bytes
    receipt_sha256: str

    def __post_init__(self) -> None:
        if sha256_bytes(self.artifact_bytes) != self.artifact_sha256:
            raise ContractError("prediction artifact digest does not match its bytes")
        if sha256_bytes(self.receipt_bytes) != self.receipt_sha256:
            raise ContractError("prediction receipt digest does not match its bytes")


def build_prediction_artifact(
    record: PredictionRecord,
    *,
    arm: PredictionArm,
    condition: PredictionCondition,
    checkpoint_sha256: str,
    feature_receipt_sha256: str,
    job_name: str,
    seed: int,
) -> PredictionArtifact:
    """Serialize one deterministic NPZ and its exact closed receipt in memory."""

    if arm not in {"local", "global", "uniform", "capacity-control"}:
        raise ContractError("prediction arm is outside the four-arm inventory")
    if condition not in {"natural-clean", "natural-drift"}:
        raise ContractError("prediction condition is outside the two-condition inventory")
    expected_condition = 0 if condition == "natural-clean" else 1
    if int(record.condition[0]) != expected_condition:
        raise ContractError("prediction record condition differs from its receipt condition")
    if seed not in SEEDS or not job_name.endswith(f"seed={seed}"):
        raise ContractError("prediction seed does not match the job name")
    canonical_job = job_name in CANONICAL_JOB_NAMES
    capacity_job = job_name in CAPACITY_CONTROL_JOB_NAMES
    if (arm == "capacity-control") != capacity_job or (
        arm != "capacity-control" and not canonical_job
    ):
        raise ContractError("prediction arm and checkpoint job family do not match")
    checkpoint = _digest(checkpoint_sha256, name="checkpoint_sha256")
    feature_receipt = _digest(feature_receipt_sha256, name="feature_receipt_sha256")
    artifact, members = prediction_npz_bytes(record)
    artifact_digest = sha256_bytes(artifact)
    receipt_payload: dict[str, object] = {
        "arm": arm,
        "artifact_bytes": len(artifact),
        "artifact_sha256": artifact_digest,
        "checkpoint_sha256": checkpoint,
        "contract_sha256": CONTRACT_SHA256,
        "condition": condition,
        "feature_receipt_sha256": feature_receipt,
        "job_name_hex": job_name.encode("ascii").hex(),
        "members": member_payload(members),
        "schema": PREDICTION_RECEIPT_SCHEMA,
        "seed": seed,
    }
    encoded_receipt = receipt_bytes(receipt_payload, expected_schema=PREDICTION_RECEIPT_SCHEMA)
    return PredictionArtifact(
        record=record,
        artifact_bytes=artifact,
        artifact_sha256=artifact_digest,
        members=members,
        receipt_bytes=encoded_receipt,
        receipt_sha256=sha256_bytes(encoded_receipt),
    )


__all__ = [
    "PredictionArm",
    "PredictionArtifact",
    "PredictionCondition",
    "PredictionIdentity",
    "build_identity_prediction",
    "build_prediction_artifact",
]
