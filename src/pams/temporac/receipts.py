"""Closed canonical receipt schemas for ``temporac.execution.v4``."""

from __future__ import annotations

import math
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from types import MappingProxyType
from typing import cast

import numpy as np

from pams.temporac.contract import (
    CAPABILITY_GRANT_SCHEMA,
    CAPABILITY_REQUEST_SCHEMA,
    CONTRACT_SHA256,
    FEATURE_MEMBER_SCHEMA,
    FEATURE_RECEIPT_SCHEMA,
    G1_TEACHER_SELECTION_RECEIPT_SCHEMA,
    G5A_RECEIPT_SCHEMA,
    G5B_RECEIPT_SCHEMA,
    JOB_NAMES,
    K1_CERTIFICATE_OUTCOME_INDEX_KEYS,
    K1_CERTIFICATE_OUTCOME_INDEX_SCHEMA,
    K1_CERTIFICATE_OUTCOME_RECEIPT_SCHEMA,
    K1_CERTIFICATE_OUTCOME_ROW_KEYS,
    K3_RECEIPT_SCHEMA,
    K4_RECEIPT_SCHEMA,
    K7_RECEIPT_SCHEMA,
    MEMBER_RECEIPT_KEYS,
    NATURAL_PREDICTION_COMPLETION_RECEIPT_SCHEMA,
    OPERATOR_CANDIDATE_AUTHORITY_KEYS,
    OPERATOR_CANDIDATE_BINDING_KEYS,
    OPERATOR_CANDIDATE_BLOCKERS,
    OPERATOR_CANDIDATE_DIRECTORY_NAME,
    OPERATOR_CANDIDATE_MEMBERS,
    OPERATOR_CANDIDATE_RECEIPT_KEYS,
    OPERATOR_CANDIDATE_RECEIPT_SCHEMA,
    OPERATOR_CANDIDATE_SOURCE_SNAPSHOT_SHA256,
    OPERATOR_MANIFEST_ROW_KEYS,
    OPERATOR_REVIEW_AUTHORITY_KEYS,
    OPERATOR_REVIEW_JSON_BYTES,
    OPERATOR_REVIEW_JSON_SHA256,
    OPERATOR_REVIEW_KEYS,
    OPERATOR_REVIEW_MARKDOWN_BYTES,
    OPERATOR_REVIEW_MARKDOWN_SHA256,
    OPERATOR_REVIEW_SCHEMA,
    PRE_G5A_INVENTORY_KEYS,
    PRE_G5A_INVENTORY_ROW_KEYS,
    PRE_G5A_OWNER_ROSTER,
    PRE_G5A_RECEIPT_INVENTORY_SCHEMA,
    PRE_G5A_STAGE_RECEIPT_SCHEMA,
    PREDICTION_MEMBER_SCHEMA,
    PREDICTION_RECEIPT_SCHEMA,
    PROPOSAL_SHA256,
    RECEIPT_DAG_EDGE_KEYS,
    RECEIPT_DAG_KEYS,
    RECEIPT_DAG_NODE_CLASSES,
    RECEIPT_DAG_NODE_KEYS,
    RECEIPT_DAG_ROLES,
    RECEIPT_DAG_SCHEMA,
    RECEIPT_KEYS,
    RESOURCE_RECEIPT_KEYS,
    RUN_RECEIPT_SCHEMA,
    SEEDS,
    TARGET_MEMBER_SCHEMA,
    TARGET_RECEIPT_SCHEMA,
    X0_INFERENCE_RECEIPT_SCHEMA,
)
from pams.temporac.hashio import (
    canonical_json_bytes,
    feature_npz_bytes,
    parse_strict_json_bytes,
    read_regular_file,
    read_target_npz_bytes,
    receipt_owner_key,
    sha256_bytes,
    standalone_json_sha256,
    target_npz_bytes,
    write_bytes_exclusive,
)
from pams.temporac.types import (
    ArrayMemberRecord,
    CertifiedTarget,
    ContractError,
    FeatureRecord,
    FrozenOperatorCandidate,
    GenericArray,
    K1OutcomeIndex,
    K1OutcomeRow,
    NamedReceiptOwner,
    NaturalPredictionCompletionReceipt,
    OperatorCandidateMember,
    PreG5AInventoryRow,
    PreG5AReceiptInventory,
    ReceiptDag,
    ReceiptDagEdge,
    ReceiptDagNode,
    ResourceRecord,
    RunReceiptOwner,
    SeedReceiptOwner,
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_LOWER_HEX_RE = re.compile(r"(?:[0-9a-f]{2})*\Z")
_PREDICTION_ARMS = frozenset({"local", "global", "uniform", "capacity-control"})
_PREDICTION_CONDITIONS = frozenset({"natural-clean", "natural-drift"})
_PRE_G5A_STAGE_RECEIPT_KEYS = frozenset(
    {
        "code_index_sha256",
        "contract_sha256",
        "environment_sha256_or_null",
        "owner",
        "payload_index_bytes",
        "payload_index_sha256",
        "schema",
        "status",
        "upstream_receipt_sha256",
    }
)
_G1_TEACHER_SELECTION_RECEIPT_KEYS = frozenset(
    {
        "code_index_sha256",
        "contract_sha256",
        "environment_sha256",
        "schema",
        "score_index_sha256",
        "selection_receipt_sha256",
        "status",
        "teacher_checkpoint_evidence_index_sha256",
        "teacher_run_receipt_sha256",
        "teacher_tune_input_receipt_sha256",
    }
)
_K1_CERTIFICATE_OUTCOME_RECEIPT_KEYS = frozenset(
    {
        "contract_sha256",
        "development_certified_component_count",
        "development_certified_identity_count",
        "outcome_index_bytes",
        "outcome_index_sha256",
        "schema",
        "status",
        "train_certified_component_count",
        "train_certified_identity_count",
    }
)
_K3_K4_RECEIPT_KEYS = frozenset(
    {
        "code_index_sha256",
        "contract_sha256",
        "environment_sha256",
        "owner",
        "payload_index_bytes",
        "payload_index_sha256",
        "schema",
        "status",
        "upstream_receipt_sha256",
    }
)
_X0_INFERENCE_RECEIPT_KEYS = frozenset(
    {
        "code_index_sha256",
        "contract_sha256",
        "environment_sha256",
        "owner",
        "prediction_index_sha256",
        "schema",
        "seed",
        "status",
        "upstream_receipt_sha256",
    }
)
_PRE_ENVIRONMENT_STAGE_OWNERS = frozenset(row[1] for row in PRE_G5A_OWNER_ROSTER[:7])
_INVENTORY_RECEIPT_KEYS = MappingProxyType(
    {
        G1_TEACHER_SELECTION_RECEIPT_SCHEMA: _G1_TEACHER_SELECTION_RECEIPT_KEYS,
        K1_CERTIFICATE_OUTCOME_RECEIPT_SCHEMA: _K1_CERTIFICATE_OUTCOME_RECEIPT_KEYS,
        K3_RECEIPT_SCHEMA: _K3_K4_RECEIPT_KEYS,
        K4_RECEIPT_SCHEMA: _K3_K4_RECEIPT_KEYS,
        NATURAL_PREDICTION_COMPLETION_RECEIPT_SCHEMA: RECEIPT_KEYS[
            NATURAL_PREDICTION_COMPLETION_RECEIPT_SCHEMA
        ],
        PRE_G5A_STAGE_RECEIPT_SCHEMA: _PRE_G5A_STAGE_RECEIPT_KEYS,
        RUN_RECEIPT_SCHEMA: RECEIPT_KEYS[RUN_RECEIPT_SCHEMA],
        X0_INFERENCE_RECEIPT_SCHEMA: _X0_INFERENCE_RECEIPT_KEYS,
    }
)


class ReceiptError(ContractError):
    """Unknown, missing, mistyped, or inconsistent canonical receipt data."""


@dataclass(frozen=True, slots=True)
class TargetArtifact:
    """One immutable certified-target NPZ and its detached canonical receipt."""

    target: CertifiedTarget
    artifact_bytes: bytes
    artifact_sha256: str
    members: tuple[ArrayMemberRecord, ...]
    receipt_bytes: bytes
    receipt_sha256: str

    def __post_init__(self) -> None:
        if not self.target.certified:
            raise ReceiptError("ABSTAIN has no target artifact or target receipt")
        if self.target.provenance is None:
            raise ReceiptError("unbound certificate has no target artifact or target receipt")
        if sha256_bytes(self.artifact_bytes) != self.artifact_sha256:
            raise ReceiptError("target artifact digest does not match its bytes")
        if sha256_bytes(self.receipt_bytes) != self.receipt_sha256:
            raise ReceiptError("target receipt digest does not match its bytes")
        receipt = parse_receipt_bytes(
            self.receipt_bytes,
            expected_schema=TARGET_RECEIPT_SCHEMA,
        )
        if (
            receipt["artifact_bytes"] != len(self.artifact_bytes)
            or receipt["artifact_sha256"] != self.artifact_sha256
            or receipt["members"] != [record.as_dict() for record in self.members]
            or receipt["source_kind"] != self.target.source_kind
            or receipt["source_key_hex"] != self.target.provenance.source_key_hex
            or receipt["source_unit_index"] != self.target.provenance.source_unit_index
            or receipt["teacher_sha256"] != self.target.provenance.teacher_sha256
        ):
            raise ReceiptError("target artifact and receipt ledgers are inconsistent")
        arrays, observed_members = read_target_npz_bytes(self.artifact_bytes)
        if observed_members != self.members or not (
            np.array_equal(arrays["chi"], self.target.chi)
            and arrays["contract_sha256"].tobytes(order="C").hex() == receipt["contract_sha256"]
            and np.array_equal(arrays["edge_mask"], self.target.edge_mask)
            and np.array_equal(arrays["pulse"], self.target.pulse)
            and np.array_equal(arrays["target_mask"], self.target.target_mask)
            and int(arrays["source_kind"][0]) == self.target.source_kind
            and arrays["teacher_sha256"].tobytes(order="C").hex() == receipt["teacher_sha256"]
        ):
            raise ReceiptError("target artifact bytes do not match the certified target receipt")


@dataclass(frozen=True, slots=True)
class LoadedTargetArtifact:
    """Validated on-disk target bytes, arrays, members, and detached receipt."""

    arrays: Mapping[str, GenericArray]
    artifact_bytes: bytes
    artifact_sha256: str
    members: tuple[ArrayMemberRecord, ...]
    receipt: Mapping[str, object]
    receipt_bytes: bytes
    receipt_sha256: str

    def __post_init__(self) -> None:
        arrays, members = read_target_npz_bytes(self.artifact_bytes)
        artifact_digest = sha256_bytes(self.artifact_bytes)
        receipt_digest = sha256_bytes(self.receipt_bytes)
        receipt = parse_receipt_bytes(
            self.receipt_bytes,
            expected_schema=TARGET_RECEIPT_SCHEMA,
        )
        if (
            artifact_digest != self.artifact_sha256
            or receipt_digest != self.receipt_sha256
            or members != self.members
            or receipt != self.receipt
            or receipt["artifact_bytes"] != len(self.artifact_bytes)
            or receipt["artifact_sha256"] != artifact_digest
            or receipt["members"] != [record.as_dict() for record in members]
            or arrays["contract_sha256"].tobytes(order="C").hex() != receipt["contract_sha256"]
            or int(arrays["source_kind"][0]) != receipt["source_kind"]
            or arrays["teacher_sha256"].tobytes(order="C").hex() != receipt["teacher_sha256"]
        ):
            raise ReceiptError("loaded target artifact fields are not jointly consistent")
        object.__setattr__(self, "arrays", arrays)
        object.__setattr__(self, "members", members)
        object.__setattr__(self, "receipt", receipt)


def _exact_keys(payload: Mapping[str, object], expected: frozenset[str], *, name: str) -> None:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        missing = (
            sorted(expected - set(payload)) if isinstance(payload, Mapping) else sorted(expected)
        )
        extra = (
            sorted(repr(value) for value in set(payload) - expected)
            if isinstance(payload, Mapping)
            else []
        )
        raise ReceiptError(f"{name} has unknown/missing keys; missing={missing}, extra={extra}")


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or cast(int, value) < minimum:
        raise ReceiptError(f"{name} must be an integer >= {minimum}")
    return cast(int, value)


def _finite_float(value: object, *, name: str) -> float:
    if type(value) is not float:
        raise ReceiptError(f"{name} must be a finite nonnegative float64 JSON number")
    typed = cast(float, value)
    if (
        not math.isfinite(typed)
        or typed < 0.0
        or (typed == 0.0 and math.copysign(1.0, typed) < 0.0)
    ):
        raise ReceiptError(f"{name} must be a finite nonnegative float64 JSON number")
    return typed


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ReceiptError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _hex_bytes(value: object, *, name: str, byte_count: int | None = None) -> bytes:
    if not isinstance(value, str) or _LOWER_HEX_RE.fullmatch(value) is None:
        raise ReceiptError(f"{name} must be lowercase even-length hexadecimal")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as exc:  # defensive; the regular expression already excludes this
        raise ReceiptError(f"{name} is not hexadecimal") from exc
    if byte_count is not None and len(decoded) != byte_count:
        raise ReceiptError(f"{name} must contain exactly {byte_count} bytes")
    return decoded


def _member(payload: object) -> ArrayMemberRecord:
    if not isinstance(payload, Mapping):
        raise ReceiptError("member receipt must be an object")
    typed = cast(Mapping[str, object], payload)
    _exact_keys(typed, MEMBER_RECEIPT_KEYS, name="member receipt")
    name = typed["name"]
    dtype = typed["dtype"]
    sha256 = typed["sha256"]
    shape = typed["shape"]
    if not isinstance(name, str) or not isinstance(dtype, str):
        raise ReceiptError("member name and dtype must be strings")
    if not isinstance(shape, list) or any(type(axis) is not int or axis < 0 for axis in shape):
        raise ReceiptError("member shape must be a JSON array of nonnegative integers")
    return ArrayMemberRecord(
        bytes=_integer(typed["bytes"], name="member bytes"),
        dtype=dtype,
        name=name,
        sha256=_digest(sha256, name="member sha256"),
        shape=tuple(cast(list[int], shape)),
    )


def _members(payload: object, *, artifact_bytes: int) -> tuple[ArrayMemberRecord, ...]:
    if not isinstance(payload, list) or not payload:
        raise ReceiptError("members must be a nonempty JSON array")
    records = tuple(_member(member) for member in payload)
    names = [record.name for record in records]
    if names != sorted(names, key=lambda value: value.encode("ascii")) or len(names) != len(
        set(names)
    ):
        raise ReceiptError("member receipts must be unique and bytewise ASCII sorted")
    if sum(record.bytes for record in records) >= artifact_bytes:
        raise ReceiptError("artifact bytes must include positive ZIP container overhead")
    return records


def _member_inventory(
    records: tuple[ArrayMemberRecord, ...],
    schema: Mapping[str, tuple[str, tuple[int | None, ...]]],
    *,
    name: str,
) -> dict[str, ArrayMemberRecord]:
    observed = {record.name.removesuffix(".npy"): record for record in records}
    if set(observed) != set(schema):
        raise ReceiptError(f"{name} members do not match the exact artifact schema")
    for field, (dtype, shape) in schema.items():
        record = observed[field]
        if (
            record.dtype != dtype
            or len(record.shape) != len(shape)
            or any(
                expected is not None and actual != expected
                for actual, expected in zip(record.shape, shape, strict=True)
            )
        ):
            raise ReceiptError(f"{name} member {field!r} has a noncanonical dtype or shape")
    return observed


def _contract_digest(payload: Mapping[str, object]) -> None:
    if _digest(payload["contract_sha256"], name="contract_sha256") != CONTRACT_SHA256:
        raise ReceiptError("receipt does not bind the canonical v4 contract bytes")


def _job_name(value: object) -> str:
    decoded = _hex_bytes(value, name="job_name_hex")
    try:
        result = decoded.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ReceiptError("job_name_hex does not encode ASCII") from exc
    if result not in JOB_NAMES:
        raise ReceiptError("job_name_hex is not in the frozen 27-job inventory")
    return result


def _digest_array(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ReceiptError(f"{name} must be an ordered JSON array")
    return tuple(_digest(item, name=f"{name} item") for item in value)


def _ascii_array(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ReceiptError(f"{name} must be an ordered JSON array")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or not item.isascii() or "\x00" in item:
            raise ReceiptError(f"{name} items must be nonempty NUL-free ASCII strings")
        result.append(item)
    return tuple(result)


def _object(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ReceiptError(f"{name} must be a JSON object with string keys")
    return cast(Mapping[str, object], value)


def _nullable_digest(value: object, *, name: str) -> str | None:
    return None if value is None else _digest(value, name=name)


def _resource(payload: object) -> ResourceRecord:
    if not isinstance(payload, Mapping):
        raise ReceiptError("run resource must be an object")
    typed = cast(Mapping[str, object], payload)
    _exact_keys(typed, RESOURCE_RECEIPT_KEYS, name="run resource")
    return ResourceRecord(
        completed_steps=_integer(typed["completed_steps"], name="completed_steps"),
        gpu_seconds=_finite_float(typed["gpu_seconds"], name="gpu_seconds"),
        max_cuda_bytes=_integer(typed["max_cuda_bytes"], name="max_cuda_bytes"),
        wall_seconds=_finite_float(typed["wall_seconds"], name="wall_seconds"),
    )


def _validate_feature(payload: Mapping[str, object]) -> None:
    _contract_digest(payload)
    artifact_bytes = _integer(payload["artifact_bytes"], name="artifact_bytes", minimum=1)
    _digest(payload["artifact_sha256"], name="artifact_sha256")
    records = _members(payload["members"], artifact_bytes=artifact_bytes)
    _member_inventory(records, FEATURE_MEMBER_SCHEMA, name="feature receipt")
    _hex_bytes(payload["opaque_key_hex"], name="opaque_key_hex", byte_count=32)
    _integer(payload["slot"], name="slot")
    _digest(payload["source_binding_sha256"], name="source_binding_sha256")


def _validate_target(payload: Mapping[str, object]) -> None:
    _contract_digest(payload)
    artifact_bytes = _integer(payload["artifact_bytes"], name="artifact_bytes", minimum=1)
    _digest(payload["artifact_sha256"], name="artifact_sha256")
    records = _members(payload["members"], artifact_bytes=artifact_bytes)
    observed = _member_inventory(records, TARGET_MEMBER_SCHEMA, name="target receipt")
    edge_count = observed["chi"].shape[0]
    if any(
        observed[field].shape != (edge_count,) for field in ("edge_mask", "pulse", "target_mask")
    ):
        raise ReceiptError("target variable member axes do not share the same edge count")
    if payload["certificate_status"] != "CERTIFIED":
        raise ReceiptError("target certificate_status must be exact ASCII CERTIFIED")
    _hex_bytes(payload["source_key_hex"], name="source_key_hex", byte_count=32)
    if type(payload["source_kind"]) is not int or payload["source_kind"] not in {0, 1}:
        raise ReceiptError("target source_kind must be integer zero or one")
    unit_index = _integer(payload["source_unit_index"], name="source_unit_index")
    if payload["source_kind"] == 0 and unit_index >= 7 * 96:
        raise ReceiptError("X0 source_unit_index is outside block/resampler/offset inventory")
    _digest(payload["teacher_sha256"], name="teacher_sha256")


def _validate_prediction(payload: Mapping[str, object]) -> None:
    _contract_digest(payload)
    artifact_bytes = _integer(payload["artifact_bytes"], name="artifact_bytes", minimum=1)
    _digest(payload["artifact_sha256"], name="artifact_sha256")
    records = _members(payload["members"], artifact_bytes=artifact_bytes)
    observed = _member_inventory(records, PREDICTION_MEMBER_SCHEMA, name="prediction receipt")
    edge_count = observed["response"].shape[0]
    if any(observed[field].shape != (edge_count,) for field in ("decoder_mask", "edge_mask")):
        raise ReceiptError("prediction response and edge-mask axes differ")
    component_count = observed["component_bounds"].shape[0]
    if any(
        observed[field].shape != (component_count,)
        for field in ("component_location", "component_score")
    ):
        raise ReceiptError("prediction component member axes differ")
    if payload["arm"] not in _PREDICTION_ARMS:
        raise ReceiptError("prediction arm is outside the four frozen natural arms")
    if payload["condition"] not in _PREDICTION_CONDITIONS:
        raise ReceiptError("prediction condition must be natural-clean or natural-drift")
    _digest(payload["checkpoint_sha256"], name="checkpoint_sha256")
    _digest(payload["feature_receipt_sha256"], name="feature_receipt_sha256")
    job_name = _job_name(payload["job_name_hex"])
    seed = _integer(payload["seed"], name="seed")
    if seed not in SEEDS or not job_name.endswith(f"seed={seed}"):
        raise ReceiptError("prediction seed does not match the job name")


def _validate_run(payload: Mapping[str, object]) -> None:
    _contract_digest(payload)
    for key in (
        "code_sha256",
        "config_sha256",
        "environment_sha256",
        "optimizer_sha256",
        "rng_sha256",
        "source_cycle_sha256",
    ):
        _digest(payload[key], name=key)
    _digest_array(payload["ordered_checkpoint_sha256"], name="ordered_checkpoint_sha256")
    _digest_array(payload["upstream_receipt_sha256"], name="upstream_receipt_sha256")
    resource = _resource(payload["resource"])
    job_name = _job_name(payload["job_name_hex"])
    seed = _integer(payload["seed"], name="seed")
    if seed not in SEEDS or not job_name.endswith(f"seed={seed}"):
        raise ReceiptError("run seed does not match the job name")
    expected_step = 20_000 if "/teacher/" in job_name else 10_000
    if _integer(payload["final_step"], name="final_step") != expected_step:
        raise ReceiptError("run final_step does not equal the fixed job horizon")
    if resource.completed_steps != expected_step:
        raise ReceiptError("run resource does not report the fixed completed step")
    if payload["status"] != "SUCCESS":
        raise ReceiptError("a canonical completed run receipt status must be SUCCESS")
    checkpoints = cast(list[object], payload["ordered_checkpoint_sha256"])
    expected_checkpoints = 40 if expected_step == 20_000 else 20
    if len(checkpoints) != expected_checkpoints:
        raise ReceiptError("run receipt has the wrong checkpoint candidate count")


def _validate_g5a(payload: Mapping[str, object]) -> None:
    _contract_digest(payload)
    for key in RECEIPT_KEYS[G5A_RECEIPT_SCHEMA] - {
        "contract_sha256",
        "schema",
        "stub_inclusive_artifact_count",
    }:
        _digest(payload[key], name=key)
    if (
        _integer(payload["stub_inclusive_artifact_count"], name="stub_inclusive_artifact_count")
        != 9_648
    ):
        raise ReceiptError("G5a stub-inclusive artifact count must be exactly 9,648")


def _validate_capability_request(payload: Mapping[str, object]) -> None:
    _digest(payload["g5a_receipt_sha256"], name="g5a_receipt_sha256")
    _digest(payload["previous_record_sha256"], name="previous_record_sha256")
    sequence = _integer(payload["sequence"], name="sequence")
    if sequence == 0 and payload["previous_record_sha256"] != "0" * 64:
        raise ReceiptError("first capability request must bind 64 ASCII zeros")


def _validate_capability_grant(payload: Mapping[str, object]) -> None:
    for key in RECEIPT_KEYS[CAPABILITY_GRANT_SCHEMA] - {"schema", "invocation_count"}:
        _digest(payload[key], name=key)
    if _integer(payload["invocation_count"], name="invocation_count") != 1:
        raise ReceiptError("capability grant invocation_count must be one")


def _validate_g5b(payload: Mapping[str, object]) -> None:
    for key in (
        "g5a_receipt_sha256",
        "join_commitment_sha256",
        "vault_root_sha256",
    ):
        _digest(payload[key], name=key)
    if payload["status"] != "PASS":
        raise ReceiptError("G5b status must be exact ASCII PASS")
    if _integer(payload["join_cardinality"], name="join_cardinality") != 402:
        raise ReceiptError("G5b join_cardinality must be exactly 402")


def _validate_k7(payload: Mapping[str, object]) -> None:
    for key in ("g5a_receipt_sha256", "g5b_receipt_sha256", "metric_payload_sha256"):
        _digest(payload[key], name=key)
    if payload["status"] not in {"PASS", "FAIL"}:
        raise ReceiptError("K7 status must be exact ASCII PASS or FAIL")


def natural_prediction_completion_receipt(
    payload: Mapping[str, object],
) -> NaturalPredictionCompletionReceipt:
    """Validate and type the exact 14-key NATP completion receipt."""

    _exact_keys(
        payload,
        RECEIPT_KEYS[NATURAL_PREDICTION_COMPLETION_RECEIPT_SCHEMA],
        name=NATURAL_PREDICTION_COMPLETION_RECEIPT_SCHEMA,
    )
    if payload["schema"] != NATURAL_PREDICTION_COMPLETION_RECEIPT_SCHEMA:
        raise ReceiptError("NATP receipt schema is invalid")
    _contract_digest(payload)
    seed = _integer(payload["seed"], name="NATP seed")
    if seed not in SEEDS or payload["owner"] != f"NATP-{seed}":
        raise ReceiptError("NATP owner and seed do not identify one frozen completion")
    upstream = _digest_array(
        payload["upstream_receipt_sha256"], name="NATP upstream_receipt_sha256"
    )
    if len(upstream) != 1:
        raise ReceiptError("NATP upstream_receipt_sha256 must contain exactly K6")
    try:
        return NaturalPredictionCompletionReceipt(
            clean_seed_index_sha256=_digest(
                payload["clean_seed_index_sha256"], name="clean_seed_index_sha256"
            ),
            clean_seed_root_sha256=_digest(
                payload["clean_seed_root_sha256"], name="clean_seed_root_sha256"
            ),
            code_index_sha256=_digest(payload["code_index_sha256"], name="code_index_sha256"),
            contract_sha256=cast(str, payload["contract_sha256"]),
            drift_seed_index_sha256=_digest(
                payload["drift_seed_index_sha256"], name="drift_seed_index_sha256"
            ),
            drift_seed_root_sha256=_digest(
                payload["drift_seed_root_sha256"], name="drift_seed_root_sha256"
            ),
            environment_sha256=_digest(payload["environment_sha256"], name="environment_sha256"),
            k6_receipt_sha256=_digest(payload["k6_receipt_sha256"], name="k6_receipt_sha256"),
            owner=cast(str, payload["owner"]),
            pilot_scope_manifest_sha256=_digest(
                payload["pilot_scope_manifest_sha256"], name="pilot_scope_manifest_sha256"
            ),
            seed=seed,
            status=cast(str, payload["status"]),
            upstream_receipt_sha256=upstream,
        )
    except ContractError as exc:
        raise ReceiptError("NATP receipt values violate the exact 14-key contract") from exc


def _validate_natp(payload: Mapping[str, object]) -> None:
    natural_prediction_completion_receipt(payload)


_VALIDATORS = {
    FEATURE_RECEIPT_SCHEMA: _validate_feature,
    TARGET_RECEIPT_SCHEMA: _validate_target,
    PREDICTION_RECEIPT_SCHEMA: _validate_prediction,
    RUN_RECEIPT_SCHEMA: _validate_run,
    G5A_RECEIPT_SCHEMA: _validate_g5a,
    CAPABILITY_REQUEST_SCHEMA: _validate_capability_request,
    CAPABILITY_GRANT_SCHEMA: _validate_capability_grant,
    G5B_RECEIPT_SCHEMA: _validate_g5b,
    K7_RECEIPT_SCHEMA: _validate_k7,
    NATURAL_PREDICTION_COMPLETION_RECEIPT_SCHEMA: _validate_natp,
}


def validate_receipt(
    payload: Mapping[str, object], *, expected_schema: str | None = None
) -> Mapping[str, object]:
    """Validate a receipt's exact closed key set and value semantics."""

    if not isinstance(payload, Mapping):
        raise ReceiptError("receipt root must be an object")
    schema = payload.get("schema")
    if not isinstance(schema, str) or schema not in RECEIPT_KEYS:
        raise ReceiptError("receipt schema is missing or unknown")
    if expected_schema is not None and schema != expected_schema:
        raise ReceiptError("receipt schema differs from the expected schema")
    _exact_keys(payload, RECEIPT_KEYS[schema], name=schema)
    _VALIDATORS[schema](payload)
    return MappingProxyType(dict(payload))


def receipt_bytes(payload: Mapping[str, object], *, expected_schema: str | None = None) -> bytes:
    validate_receipt(payload, expected_schema=expected_schema)
    return canonical_json_bytes(payload)


def parse_receipt_bytes(
    payload: bytes, *, expected_schema: str | None = None
) -> Mapping[str, object]:
    """Parse only an already canonical receipt byte string."""

    try:
        decoded = parse_strict_json_bytes(payload, canonical=True)
    except ContractError as exc:
        raise ReceiptError("receipt is not duplicate-free canonical UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ReceiptError("receipt root must be a canonical JSON object")
    return validate_receipt(cast(dict[str, object], decoded), expected_schema=expected_schema)


def load_receipt(path: Path, *, expected_schema: str | None = None) -> Mapping[str, object]:
    return parse_receipt_bytes(read_regular_file(path), expected_schema=expected_schema)


def _inventory_edge_role(row: PreG5AInventoryRow) -> str:
    if row.node_class == "run":
        return "run.upstream"
    if row.node_class == "g1-teacher-selection":
        return "g1.teacher-run"
    if row.node_class == "k1-certificate-outcome":
        return "k1.upstream"
    if row.node_class == "natural-prediction-completion":
        return "natp.k6"
    return "stage.upstream"


def _edge_sort_key(edge: ReceiptDagEdge) -> tuple[bytes, bytes, int, bytes]:
    return (
        bytes.fromhex(edge.from_receipt_sha256),
        edge.role.encode("ascii"),
        edge.ordinal,
        bytes.fromhex(edge.to_receipt_sha256),
    )


def _topological_receipts(
    receipt_order: Sequence[str], edges: Sequence[ReceiptDagEdge]
) -> tuple[str, ...]:
    if len(set(receipt_order)) != len(receipt_order):
        raise ReceiptError("receipt DAG node digests must be unique")
    order = {digest: index for index, digest in enumerate(receipt_order)}
    dependencies = {digest: set[str]() for digest in receipt_order}
    dependents = {digest: set[str]() for digest in receipt_order}
    for edge in edges:
        if edge.from_receipt_sha256 not in order or edge.to_receipt_sha256 not in order:
            raise ReceiptError("receipt DAG edge has a dangling endpoint")
        dependencies[edge.from_receipt_sha256].add(edge.to_receipt_sha256)
        dependents[edge.to_receipt_sha256].add(edge.from_receipt_sha256)
    ready: list[tuple[int, str]] = []
    for digest in receipt_order:
        if not dependencies[digest]:
            heappush(ready, (order[digest], digest))
    result: list[str] = []
    while ready:
        _, digest = heappop(ready)
        result.append(digest)
        for dependent in sorted(dependents[digest], key=order.__getitem__):
            dependencies[dependent].discard(digest)
            if not dependencies[dependent]:
                heappush(ready, (order[dependent], dependent))
    if len(result) != len(receipt_order):
        raise ReceiptError("receipt DAG contains a dependency cycle")
    return tuple(result)


def validate_pre_g5a_receipt_inventory(
    payload: Mapping[str, object],
) -> PreG5AReceiptInventory:
    """Validate the exact 54-owner roster and derive its 106 typed edges."""

    _exact_keys(payload, PRE_G5A_INVENTORY_KEYS, name=PRE_G5A_RECEIPT_INVENTORY_SCHEMA)
    if payload["schema"] != PRE_G5A_RECEIPT_INVENTORY_SCHEMA:
        raise ReceiptError("pre-G5a inventory schema is invalid")
    _contract_digest(payload)
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, list) or len(raw_rows) != len(PRE_G5A_OWNER_ROSTER):
        raise ReceiptError("pre-G5a inventory must contain exactly 54 rows")
    rows: list[PreG5AInventoryRow] = []
    owner_to_digest: dict[str, str] = {}
    for ordinal, (raw_row, expected) in enumerate(zip(raw_rows, PRE_G5A_OWNER_ROSTER, strict=True)):
        row = _object(raw_row, name=f"pre-G5a inventory row {ordinal}")
        _exact_keys(row, PRE_G5A_INVENTORY_ROW_KEYS, name=f"pre-G5a inventory row {ordinal}")
        upstream_owners = _ascii_array(
            row["upstream_owner_tokens"], name=f"inventory row {ordinal} upstream owners"
        )
        upstream_digests = _digest_array(
            row["upstream_receipt_sha256"], name=f"inventory row {ordinal} upstream digests"
        )
        node_class, owner, receipt_schema, expected_upstream = expected
        if (
            row["node_class"] != node_class
            or row["owner"] != owner
            or row["receipt_schema"] != receipt_schema
            or upstream_owners != expected_upstream
        ):
            raise ReceiptError("pre-G5a inventory row differs from the frozen owner roster")
        if len(upstream_digests) != len(expected_upstream):
            raise ReceiptError("pre-G5a inventory upstream arrays differ in length")
        for upstream_owner, upstream_digest in zip(
            expected_upstream, upstream_digests, strict=True
        ):
            if owner_to_digest.get(upstream_owner) != upstream_digest:
                raise ReceiptError("pre-G5a inventory edge does not bind the prior owner receipt")
        receipt_digest = _digest(row["receipt_sha256"], name=f"inventory row {ordinal} receipt")
        if receipt_digest in owner_to_digest.values():
            raise ReceiptError("pre-G5a inventory receipt digests must be unique")
        try:
            typed = PreG5AInventoryRow(
                node_class=node_class,
                owner=owner,
                receipt_schema=receipt_schema,
                receipt_sha256=receipt_digest,
                upstream_owner_tokens=upstream_owners,
                upstream_receipt_sha256=upstream_digests,
            )
        except ContractError as exc:
            raise ReceiptError("pre-G5a inventory row has an invalid type or value") from exc
        rows.append(typed)
        owner_to_digest[owner] = receipt_digest
    edges = tuple(
        sorted(
            (
                ReceiptDagEdge(
                    from_receipt_sha256=row.receipt_sha256,
                    ordinal=ordinal,
                    role=_inventory_edge_role(row),
                    to_receipt_sha256=owner_to_digest[upstream_owner],
                )
                for row in rows
                for ordinal, upstream_owner in enumerate(row.upstream_owner_tokens)
            ),
            key=_edge_sort_key,
        )
    )
    if len(edges) != 106:
        raise ReceiptError("pre-G5a inventory must derive exactly 106 direct edges")
    k1_row = rows[15]
    k1_edges = tuple(
        sorted(
            (edge for edge in edges if edge.from_receipt_sha256 == k1_row.receipt_sha256),
            key=lambda edge: edge.ordinal,
        )
    )
    if tuple((edge.ordinal, edge.role) for edge in k1_edges) != (
        (0, "k1.upstream"),
        (1, "k1.upstream"),
    ) or tuple(edge.to_receipt_sha256 for edge in k1_edges) != (
        owner_to_digest["G0-ACQUIRE"],
        owner_to_digest["G1-TEACHER-AGG"],
    ):
        raise ReceiptError("K1 upstream ordinals must bind G0 then G1 exactly")
    topology = _topological_receipts([row.receipt_sha256 for row in rows], edges)
    return PreG5AReceiptInventory(
        contract_sha256=CONTRACT_SHA256,
        rows=tuple(rows),
        direct_edges=edges,
        topological_receipt_sha256=topology,
    )


def parse_pre_g5a_receipt_inventory_bytes(payload: bytes) -> PreG5AReceiptInventory:
    try:
        decoded = parse_strict_json_bytes(payload, canonical=True)
    except ContractError as exc:
        raise ReceiptError("pre-G5a inventory bytes are not duplicate-free canonical JSON") from exc
    return validate_pre_g5a_receipt_inventory(_object(decoded, name="pre-G5a receipt inventory"))


def _inventory_receipt_owner_key(
    row: PreG5AInventoryRow,
    payload: Mapping[str, object],
) -> str:
    """Validate one exact canonical receipt payload and derive its typed owner key."""

    if payload.get("schema") != row.receipt_schema:
        raise ReceiptError("inventory receipt schema does not equal its roster row")
    expected_keys = _INVENTORY_RECEIPT_KEYS.get(row.receipt_schema)
    if expected_keys is None:
        raise ReceiptError("inventory row uses an unsupported receipt schema")
    _exact_keys(payload, expected_keys, name=row.receipt_schema)
    _contract_digest(payload)
    upstream = row.upstream_receipt_sha256
    if row.receipt_schema == PRE_G5A_STAGE_RECEIPT_SCHEMA:
        if payload["owner"] != row.owner:
            raise ReceiptError("pre-G5a stage receipt owner does not equal its roster row")
        if (
            _digest_array(payload["upstream_receipt_sha256"], name="stage upstream_receipt_sha256")
            != upstream
        ):
            raise ReceiptError("pre-G5a stage receipt upstream array differs from its roster row")
        _digest(payload["code_index_sha256"], name="stage code_index_sha256")
        _digest(payload["payload_index_sha256"], name="stage payload_index_sha256")
        _integer(payload["payload_index_bytes"], name="stage payload_index_bytes", minimum=1)
        environment = payload["environment_sha256_or_null"]
        if row.owner in _PRE_ENVIRONMENT_STAGE_OWNERS:
            if environment is not None:
                raise ReceiptError("pre-environment stage receipt must have null environment")
        else:
            _digest(environment, name="stage environment_sha256_or_null")
        if payload["status"] != "PASS":
            raise ReceiptError("pre-G5a stage receipt status must be exact ASCII PASS")
        return receipt_owner_key(row.node_class, NamedReceiptOwner(owner=row.owner))
    if row.receipt_schema == RUN_RECEIPT_SCHEMA:
        validate_receipt(payload, expected_schema=RUN_RECEIPT_SCHEMA)
        job_name = _job_name(payload["job_name_hex"])
        seed = _integer(payload["seed"], name="run seed")
        if job_name != row.owner:
            raise ReceiptError("run receipt decoded job does not equal its roster owner")
        if (
            _digest_array(payload["upstream_receipt_sha256"], name="run upstream_receipt_sha256")
            != upstream
        ):
            raise ReceiptError("run receipt upstream array differs from its roster row")
        return receipt_owner_key(row.node_class, RunReceiptOwner(job_name=job_name, seed=seed))
    if row.receipt_schema == G1_TEACHER_SELECTION_RECEIPT_SCHEMA:
        teacher_runs = _digest_array(
            payload["teacher_run_receipt_sha256"], name="G1 teacher_run_receipt_sha256"
        )
        if teacher_runs != upstream:
            raise ReceiptError("G1 teacher-run array differs from its roster row")
        for key in _G1_TEACHER_SELECTION_RECEIPT_KEYS - {
            "contract_sha256",
            "schema",
            "status",
            "teacher_run_receipt_sha256",
        }:
            _digest(payload[key], name=f"G1 {key}")
        if payload["status"] != "PASS":
            raise ReceiptError("G1 receipt status must be exact ASCII PASS")
        return receipt_owner_key(row.node_class, NamedReceiptOwner(owner=row.owner))
    if row.receipt_schema == K1_CERTIFICATE_OUTCOME_RECEIPT_SCHEMA:
        _digest(payload["outcome_index_sha256"], name="K1 outcome_index_sha256")
        _integer(payload["outcome_index_bytes"], name="K1 outcome_index_bytes", minimum=1)
        for key in (
            "development_certified_component_count",
            "development_certified_identity_count",
            "train_certified_component_count",
            "train_certified_identity_count",
        ):
            _integer(payload[key], name=f"K1 {key}")
        if payload["status"] != "PASS":
            raise ReceiptError("K1 receipt status must be exact ASCII PASS")
        return receipt_owner_key(row.node_class, NamedReceiptOwner(owner=row.owner))
    if row.receipt_schema == X0_INFERENCE_RECEIPT_SCHEMA:
        seed = _integer(payload["seed"], name="X0I seed")
        if seed not in SEEDS or payload["owner"] != row.owner or row.owner != f"X0I-{seed}":
            raise ReceiptError("X0I receipt owner and seed do not equal its roster row")
        if (
            _digest_array(payload["upstream_receipt_sha256"], name="X0I upstream_receipt_sha256")
            != upstream
        ):
            raise ReceiptError("X0I receipt upstream array differs from its roster row")
        for key in ("code_index_sha256", "environment_sha256", "prediction_index_sha256"):
            _digest(payload[key], name=f"X0I {key}")
        if payload["status"] != "PASS":
            raise ReceiptError("X0I receipt status must be exact ASCII PASS")
        return receipt_owner_key(row.node_class, SeedReceiptOwner(seed=seed))
    if row.receipt_schema in {K3_RECEIPT_SCHEMA, K4_RECEIPT_SCHEMA}:
        if payload["owner"] != row.owner:
            raise ReceiptError("K3/K4 receipt owner does not equal its roster row")
        if (
            _digest_array(payload["upstream_receipt_sha256"], name="K3/K4 upstream_receipt_sha256")
            != upstream
        ):
            raise ReceiptError("K3/K4 receipt upstream array differs from its roster row")
        for key in ("code_index_sha256", "environment_sha256", "payload_index_sha256"):
            _digest(payload[key], name=f"K3/K4 {key}")
        _integer(payload["payload_index_bytes"], name="K3/K4 payload_index_bytes", minimum=1)
        if payload["status"] != "PASS":
            raise ReceiptError("K3/K4 receipt status must be exact ASCII PASS")
        return receipt_owner_key(row.node_class, NamedReceiptOwner(owner=row.owner))
    if row.receipt_schema == NATURAL_PREDICTION_COMPLETION_RECEIPT_SCHEMA:
        typed = natural_prediction_completion_receipt(payload)
        if typed.owner != row.owner or typed.upstream_receipt_sha256 != upstream:
            raise ReceiptError("NATP receipt owner/upstream differs from its roster row")
        return receipt_owner_key(row.node_class, SeedReceiptOwner(seed=typed.seed))
    raise ReceiptError("inventory row uses an unsupported receipt schema")


def validate_pre_g5a_receipt_evidence(
    inventory: PreG5AReceiptInventory,
    receipt_bytes_by_sha256: Mapping[str, bytes],
) -> Mapping[str, str]:
    """Bind every roster row to exact canonical receipt bytes and a typed owner key.

    This validates receipt-byte identity and the closed receipt fields needed by
    the pre-G5a roster.  It does not claim validation of separately referenced
    payload-index artifacts, runtime discovery, or a future full receipt DAG.
    """

    if not isinstance(inventory, PreG5AReceiptInventory):
        raise ReceiptError("pre-G5a receipt evidence requires a typed inventory")
    revalidated_inventory = validate_pre_g5a_receipt_inventory(
        {
            "contract_sha256": inventory.contract_sha256,
            "rows": [row.as_dict() for row in inventory.rows],
            "schema": PRE_G5A_RECEIPT_INVENTORY_SCHEMA,
        }
    )
    if revalidated_inventory != inventory:
        raise ReceiptError("typed pre-G5a inventory differs from the exact derived roster")
    if not isinstance(receipt_bytes_by_sha256, Mapping):
        raise ReceiptError("pre-G5a receipt evidence must be a digest-to-bytes mapping")
    expected_digests = {row.receipt_sha256 for row in inventory.rows}
    if set(receipt_bytes_by_sha256) != expected_digests:
        raise ReceiptError("pre-G5a receipt evidence must contain exactly all 54 receipt bytes")
    owner_keys: dict[str, str] = {}
    for row in inventory.rows:
        encoded = receipt_bytes_by_sha256[row.receipt_sha256]
        if type(encoded) is not bytes:
            raise ReceiptError("pre-G5a receipt evidence values must be immutable bytes")
        if sha256_bytes(encoded) != row.receipt_sha256:
            raise ReceiptError("pre-G5a receipt evidence digest does not match its exact bytes")
        try:
            decoded = parse_strict_json_bytes(encoded, canonical=True)
            receipt = _object(decoded, name="pre-G5a receipt evidence payload")
            owner_keys[row.receipt_sha256] = _inventory_receipt_owner_key(row, receipt)
        except ContractError as exc:
            raise ReceiptError(
                "pre-G5a receipt evidence violates its closed typed payload"
            ) from exc
    return MappingProxyType(owner_keys)


def validate_receipt_dag(
    payload: Mapping[str, object],
    *,
    inventory: PreG5AReceiptInventory | None = None,
    receipt_bytes_by_sha256: Mapping[str, bytes] | None = None,
) -> ReceiptDag:
    """Validate the exact evidence-backed 54-node/106-edge pre-G5a DAG.

    No generic/full-DAG mode exists yet: absent its future authoritative
    artifact indexes, omitting either evidence input fails closed.
    """

    if inventory is None or receipt_bytes_by_sha256 is None:
        raise ReceiptError(
            "receipt DAG validation requires exact pre-G5a inventory and receipt-byte evidence"
        )
    owner_keys = validate_pre_g5a_receipt_evidence(inventory, receipt_bytes_by_sha256)

    _exact_keys(payload, RECEIPT_DAG_KEYS, name=RECEIPT_DAG_SCHEMA)
    if payload["schema"] != RECEIPT_DAG_SCHEMA:
        raise ReceiptError("receipt DAG schema is invalid")
    _contract_digest(payload)
    raw_nodes = payload["nodes"]
    raw_edges = payload["edges"]
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ReceiptError("receipt DAG nodes must be a nonempty array")
    if not isinstance(raw_edges, list):
        raise ReceiptError("receipt DAG edges must be an array")
    nodes: list[ReceiptDagNode] = []
    for ordinal, raw_node in enumerate(raw_nodes):
        node = _object(raw_node, name=f"receipt DAG node {ordinal}")
        _exact_keys(node, RECEIPT_DAG_NODE_KEYS, name=f"receipt DAG node {ordinal}")
        node_class = node["class"]
        if not isinstance(node_class, str) or node_class not in RECEIPT_DAG_NODE_CLASSES:
            raise ReceiptError("receipt DAG node class is outside the closed token set")
        try:
            nodes.append(
                ReceiptDagNode(
                    node_class=node_class,
                    owner_key=_digest(node["owner_key"], name="receipt DAG owner_key"),
                    receipt_sha256=_digest(
                        node["receipt_sha256"], name="receipt DAG receipt_sha256"
                    ),
                )
            )
        except ContractError as exc:
            raise ReceiptError("receipt DAG node has an invalid type or value") from exc
    canonical_nodes = sorted(
        nodes,
        key=lambda node: (
            node.node_class.encode("ascii"),
            bytes.fromhex(node.owner_key),
            bytes.fromhex(node.receipt_sha256),
        ),
    )
    if nodes != canonical_nodes:
        raise ReceiptError("receipt DAG nodes are not in canonical node order")
    if len({node.receipt_sha256 for node in nodes}) != len(nodes) or len(
        {(node.node_class, node.owner_key) for node in nodes}
    ) != len(nodes):
        raise ReceiptError("receipt DAG nodes contain a duplicate receipt or owner")
    edges: list[ReceiptDagEdge] = []
    for ordinal, raw_edge in enumerate(raw_edges):
        edge_payload = _object(raw_edge, name=f"receipt DAG edge {ordinal}")
        _exact_keys(edge_payload, RECEIPT_DAG_EDGE_KEYS, name=f"receipt DAG edge {ordinal}")
        role = edge_payload["role"]
        if not isinstance(role, str) or role not in RECEIPT_DAG_ROLES:
            raise ReceiptError("receipt DAG edge role is outside the closed token set")
        try:
            edges.append(
                ReceiptDagEdge(
                    from_receipt_sha256=_digest(
                        edge_payload["from_receipt_sha256"], name="receipt DAG from digest"
                    ),
                    ordinal=_integer(edge_payload["ordinal"], name="receipt DAG ordinal"),
                    role=role,
                    to_receipt_sha256=_digest(
                        edge_payload["to_receipt_sha256"], name="receipt DAG to digest"
                    ),
                )
            )
        except ContractError as exc:
            raise ReceiptError("receipt DAG edge has an invalid type or value") from exc
    if edges != sorted(edges, key=_edge_sort_key) or len(set(edges)) != len(edges):
        raise ReceiptError("receipt DAG edges are duplicate or not in canonical order")
    grouped: dict[tuple[str, str], list[int]] = {}
    for edge in edges:
        grouped.setdefault((edge.from_receipt_sha256, edge.role), []).append(edge.ordinal)
    if any(ordinals != list(range(len(ordinals))) for ordinals in grouped.values()):
        raise ReceiptError("receipt DAG role ordinals must be consecutive from zero")
    receipt_order = [node.receipt_sha256 for node in nodes]
    topology = _topological_receipts(receipt_order, edges)
    expected_nodes = tuple(
        sorted(
            (
                ReceiptDagNode(
                    node_class=row.node_class,
                    owner_key=owner_keys[row.receipt_sha256],
                    receipt_sha256=row.receipt_sha256,
                )
                for row in inventory.rows
            ),
            key=lambda node: (
                node.node_class.encode("ascii"),
                bytes.fromhex(node.owner_key),
                bytes.fromhex(node.receipt_sha256),
            ),
        )
    )
    if tuple(nodes) != expected_nodes:
        raise ReceiptError("receipt DAG nodes do not equal the exact typed 54-node roster")
    if tuple(edges) != inventory.direct_edges:
        raise ReceiptError("receipt DAG edges do not equal the exact 106-edge roster")
    return ReceiptDag(
        contract_sha256=CONTRACT_SHA256,
        nodes=tuple(nodes),
        edges=tuple(edges),
        topological_receipt_sha256=topology,
    )


def parse_receipt_dag_bytes(
    payload: bytes,
    *,
    inventory: PreG5AReceiptInventory | None = None,
    receipt_bytes_by_sha256: Mapping[str, bytes] | None = None,
) -> ReceiptDag:
    try:
        decoded = parse_strict_json_bytes(payload, canonical=True)
    except ContractError as exc:
        raise ReceiptError("receipt DAG bytes are not duplicate-free canonical JSON") from exc
    return validate_receipt_dag(
        _object(decoded, name="receipt DAG"),
        inventory=inventory,
        receipt_bytes_by_sha256=receipt_bytes_by_sha256,
    )


def validate_k1_outcome_row(payload: Mapping[str, object]) -> K1OutcomeRow:
    """Validate one exact-13-key K1 row including all null/status rules."""

    _exact_keys(payload, K1_CERTIFICATE_OUTCOME_ROW_KEYS, name="K1 outcome row")
    reasons = payload["certificate_reasons"]
    if not isinstance(reasons, list):
        raise ReceiptError("K1 certificate_reasons must be an ordered JSON array")
    try:
        return K1OutcomeRow(
            certificate_reasons=tuple(cast(list[int], reasons)),
            certificate_status=cast(str, payload["certificate_status"]),
            component_key_hex=_digest(payload["component_key_hex"], name="component_key_hex"),
            eligible=cast(bool, payload["eligible"]),
            feature_artifact_sha256_or_null=_nullable_digest(
                payload["feature_artifact_sha256_or_null"],
                name="feature_artifact_sha256_or_null",
            ),
            feature_receipt_sha256_or_null=_nullable_digest(
                payload["feature_receipt_sha256_or_null"],
                name="feature_receipt_sha256_or_null",
            ),
            natural_input_receipt_sha256_or_null=_nullable_digest(
                payload["natural_input_receipt_sha256_or_null"],
                name="natural_input_receipt_sha256_or_null",
            ),
            opaque_key_hex=_digest(payload["opaque_key_hex"], name="opaque_key_hex"),
            population_row_sha256=_digest(
                payload["population_row_sha256"], name="population_row_sha256"
            ),
            slot=cast(int, payload["slot"]),
            split=cast(str, payload["split"]),
            target_artifact_sha256_or_null=_nullable_digest(
                payload["target_artifact_sha256_or_null"],
                name="target_artifact_sha256_or_null",
            ),
            target_receipt_sha256_or_null=_nullable_digest(
                payload["target_receipt_sha256_or_null"],
                name="target_receipt_sha256_or_null",
            ),
        )
    except ContractError as exc:
        raise ReceiptError("K1 outcome row has an invalid type, null, or status relation") from exc


def k1_outcome_row_bytes(payload: Mapping[str, object] | K1OutcomeRow) -> bytes:
    row = payload if isinstance(payload, K1OutcomeRow) else validate_k1_outcome_row(payload)
    return canonical_json_bytes(row.as_dict())


def k1_outcome_row_sha256(payload: Mapping[str, object] | K1OutcomeRow) -> str:
    row = payload if isinstance(payload, K1OutcomeRow) else validate_k1_outcome_row(payload)
    return standalone_json_sha256(row.as_dict())


def parse_k1_outcome_row_bytes(payload: bytes) -> K1OutcomeRow:
    try:
        decoded = parse_strict_json_bytes(payload, canonical=True)
    except ContractError as exc:
        raise ReceiptError("K1 outcome row bytes are not duplicate-free canonical JSON") from exc
    row = validate_k1_outcome_row(_object(decoded, name="K1 outcome row"))
    if k1_outcome_row_bytes(row) != payload:
        raise ReceiptError("K1 outcome row does not reproduce its standalone preimage")
    return row


def validate_k1_outcome_index(payload: Mapping[str, object]) -> K1OutcomeIndex:
    """Validate the closed 402-row K1 index, row order, hashes, and coverage floors."""

    _exact_keys(
        payload, K1_CERTIFICATE_OUTCOME_INDEX_KEYS, name=K1_CERTIFICATE_OUTCOME_INDEX_SCHEMA
    )
    if payload["schema"] != K1_CERTIFICATE_OUTCOME_INDEX_SCHEMA:
        raise ReceiptError("K1 outcome index schema is invalid")
    _contract_digest(payload)
    population_manifest_sha256 = _digest(
        payload["population_manifest_sha256"], name="K1 population_manifest_sha256"
    )
    raw_rows = payload["rows"]
    if not isinstance(raw_rows, list) or len(raw_rows) != 402:
        raise ReceiptError("K1 outcome index must contain exactly 402 rows")
    rows = tuple(
        validate_k1_outcome_row(_object(raw, name=f"K1 outcome row {ordinal}"))
        for ordinal, raw in enumerate(raw_rows)
    )
    identities = [
        (0 if row.split == "train" else 1, bytes.fromhex(row.opaque_key_hex), row.slot)
        for row in rows
    ]
    if identities != sorted(identities) or len(set(identities)) != len(identities):
        raise ReceiptError("K1 rows must be unique in train/val, opaque-key, slot order")
    if len({row.population_row_sha256 for row in rows}) != len(rows):
        raise ReceiptError("K1 population_row_sha256 values must be unique")
    for split in ("train", "val"):
        eligible = [row for row in rows if row.split == split and row.eligible]
        certified = [row for row in eligible if row.certificate_status == "CERTIFIED"]
        eligible_components = {row.component_key_hex for row in eligible}
        certified_components = {row.component_key_hex for row in certified}
        if not eligible or not eligible_components:
            raise ReceiptError("K1 eligible identity/component denominators must be positive")
        if 5 * len(certified) < 4 * len(eligible) or 5 * len(certified_components) < 4 * len(
            eligible_components
        ):
            raise ReceiptError("K1 certified identity/component coverage is below 80 percent")
        if split == "val" and (len(certified) < 108 or len(certified_components) < 8):
            raise ReceiptError("K1 development coverage is below the retained 108/8 floors")
    return K1OutcomeIndex(
        contract_sha256=CONTRACT_SHA256,
        population_manifest_sha256=population_manifest_sha256,
        rows=rows,
        row_sha256=tuple(k1_outcome_row_sha256(row) for row in rows),
    )


def parse_k1_outcome_index_bytes(payload: bytes) -> K1OutcomeIndex:
    try:
        decoded = parse_strict_json_bytes(payload, canonical=True)
    except ContractError as exc:
        raise ReceiptError("K1 outcome index bytes are not duplicate-free canonical JSON") from exc
    return validate_k1_outcome_index(_object(decoded, name="K1 outcome index"))


def _validate_frozen_project_records(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list) or not value:
        raise ReceiptError(f"{name} must be a nonempty ordered array")
    paths: list[str] = []
    for ordinal, raw_record in enumerate(value):
        record = _object(raw_record, name=f"{name} row {ordinal}")
        _exact_keys(
            record,
            frozenset({"bytes", "path", "roles", "sha256"}),
            name=f"{name} row {ordinal}",
        )
        path = record["path"]
        roles = record["roles"]
        if (
            not isinstance(path, str)
            or not path
            or not path.isascii()
            or path.startswith(("/", "\\"))
            or "\\" in path
            or ".." in path.split("/")
        ):
            raise ReceiptError(f"{name} contains a noncanonical relative path")
        _integer(record["bytes"], name=f"{name} bytes", minimum=1)
        _digest(record["sha256"], name=f"{name} sha256")
        role_tokens = _ascii_array(roles, name=f"{name} roles")
        if len(role_tokens) != 1:
            raise ReceiptError(f"{name} rows must have exactly one role")
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ReceiptError(f"{name} paths must be unique in their frozen ledger order")
    return value


def _validate_operator_candidate_receipt(
    payload: Mapping[str, object],
) -> tuple[Mapping[str, object], tuple[tuple[str, bool], ...]]:
    _exact_keys(
        payload,
        OPERATOR_CANDIDATE_RECEIPT_KEYS,
        name=OPERATOR_CANDIDATE_RECEIPT_SCHEMA,
    )
    if (
        payload["schema"] != OPERATOR_CANDIDATE_RECEIPT_SCHEMA
        or payload["candidate_status"] != "FROZEN_CANDIDATE_PENDING_FRESH_IMPLEMENTATION_REVIEW"
        or payload["p2_status"] != "NOT_CLAIMED"
        or payload["s0_status"] != "NOT_CLAIMED"
        or payload["validated_row_count"] != 10_368
    ):
        raise ReceiptError("operator candidate receipt status or row count is invalid")
    blockers = payload["blockers"]
    if not isinstance(blockers, list) or tuple(blockers) != OPERATOR_CANDIDATE_BLOCKERS:
        raise ReceiptError("operator candidate receipt blockers differ from the frozen order")
    authority = _object(payload["authority"], name="operator candidate authority")
    _exact_keys(authority, OPERATOR_CANDIDATE_AUTHORITY_KEYS, name="operator candidate authority")
    if any(type(value) is not bool or value for value in authority.values()):
        raise ReceiptError("operator candidate receipt must carry zero authority")
    expected_specs = {
        name: (byte_count, digest) for name, byte_count, digest in OPERATOR_CANDIDATE_MEMBERS
    }
    members = _object(payload["members"], name="operator candidate members")
    expected_nonreceipt = {
        name: digest
        for name, (_byte_count, digest) in expected_specs.items()
        if name != "candidate_receipt.json"
    }
    if dict(members) != expected_nonreceipt:
        raise ReceiptError("operator candidate receipt does not bind the exact four members")
    bindings = _object(payload["bindings"], name="operator candidate bindings")
    _exact_keys(bindings, OPERATOR_CANDIDATE_BINDING_KEYS, name="operator candidate bindings")
    for key, value in bindings.items():
        if key.endswith("_sha256"):
            _digest(value, name=f"operator candidate binding {key}")
    if (
        bindings["proposal_sha256"] != PROPOSAL_SHA256
        or bindings["runtime_lock_sha256"] != expected_specs["environment_lock.json"][1]
        or bindings["source_snapshot_sha256"] != OPERATOR_CANDIDATE_SOURCE_SNAPSHOT_SHA256
    ):
        raise ReceiptError("operator candidate receipt has an invalid frozen binding")
    _validate_frozen_project_records(
        bindings["generation_project_files"], name="generation_project_files"
    )
    _validate_frozen_project_records(bindings["review_project_files"], name="review_project_files")
    return bindings, tuple(sorted(cast(Mapping[str, bool], authority).items()))


def _validate_operator_manifest(value: object) -> None:
    if not isinstance(value, list) or len(value) != 10_368:
        raise ReceiptError("operator manifest must contain exactly 10,368 rows")
    previous_key: str | None = None
    digest_keys = tuple(key for key in OPERATOR_MANIFEST_ROW_KEYS if key.endswith("_sha256"))
    for ordinal, raw_row in enumerate(value):
        row = _object(raw_row, name=f"operator manifest row {ordinal}")
        _exact_keys(row, frozenset(OPERATOR_MANIFEST_ROW_KEYS), name="operator manifest row")
        key_hex = row["key_hex"]
        if not isinstance(key_hex, str):
            raise ReceiptError("operator manifest key_hex must be a string")
        _hex_bytes(key_hex, name="operator manifest key_hex", byte_count=10)
        if previous_key is not None and key_hex <= previous_key:
            raise ReceiptError("operator manifest keys must be unique and strictly ordered")
        previous_key = key_hex
        for key in digest_keys:
            _digest(row[key], name=f"operator manifest {key}")
        if type(row["amplitude"]) is not float or row["amplitude"] not in {0.60, 0.75, 1.00}:
            raise ReceiptError("operator manifest amplitude is outside the frozen domain")
        for key in ("edge_count", "gap", "offset", "truth_negative_components", "width"):
            _integer(row[key], name=f"operator manifest {key}")
        reset_edge = row["reset_edge"]
        if reset_edge is not None:
            _integer(reset_edge, name="operator manifest reset_edge")
        if row["layout"] not in {"interior", "left-boundary", "right-boundary", "run-reset"}:
            raise ReceiptError("operator manifest layout is outside the frozen domain")


def _validate_operator_review(
    review: Mapping[str, object],
    *,
    candidate_receipt: Mapping[str, object],
    bindings: Mapping[str, object],
) -> None:
    _exact_keys(review, OPERATOR_REVIEW_KEYS, name=OPERATOR_REVIEW_SCHEMA)
    if (
        review["schema"] != OPERATOR_REVIEW_SCHEMA
        or review["candidate_disposition"] != "ACCEPTED_AS_IMMUTABLE_P2_INPUT_ONLY"
        or review["deterministic_evidence_acceptance"] != "accepted"
        or review["verdict"] != "PASS"
        or review["overall_verdict"] != "pass"
        or review["integrity_status"] != "pass"
        or review["acceptance_status"] != "provisional"
        or review["review_independence"] != "same-family"
        or review["blockers"] != []
        or review["findings"] != []
    ):
        raise ReceiptError("operator review does not have the frozen candidate-only disposition")
    review_md = _object(review["review_md"], name="operator review markdown binding")
    _exact_keys(
        review_md,
        frozenset({"bytes", "path", "sha256"}),
        name="operator review markdown binding",
    )
    if dict(review_md) != {
        "bytes": OPERATOR_REVIEW_MARKDOWN_BYTES,
        "path": "refine-logs/temporac/TEMPORAC_OPERATOR_IMPLEMENTATION_REVIEW_V2_20260816.md",
        "sha256": OPERATOR_REVIEW_MARKDOWN_SHA256,
    }:
        raise ReceiptError("operator review markdown foreign key is invalid")
    authority = _object(review["authority"], name="operator review authority")
    _exact_keys(authority, OPERATOR_REVIEW_AUTHORITY_KEYS, name="operator review authority")
    for key, value in authority.items():
        expected = key in {
            "candidate_only",
            "eligible_input_to_future_separately_authorized_P2_fixture_gate",
        }
        if type(value) is not bool or value is not expected:
            raise ReceiptError("operator review authority exceeds candidate-only eligibility")
    audited = _object(review["audited_inputs"], name="operator review audited_inputs")
    _exact_keys(
        audited,
        frozenset(
            {
                "bindings",
                "candidate_directory",
                "candidate_members",
                "canonical_proposals",
                "project_files",
                "source_snapshot",
            }
        ),
        name="operator review audited_inputs",
    )
    if audited["candidate_directory"] != (
        f"data/temporac_p00_v4/{OPERATOR_CANDIDATE_DIRECTORY_NAME}"
    ):
        raise ReceiptError("operator review names a different candidate directory")
    expected_review_members = {
        name: {"bytes": byte_count, "sha256": digest}
        for name, byte_count, digest in OPERATOR_CANDIDATE_MEMBERS
    }
    if audited["candidate_members"] != expected_review_members:
        raise ReceiptError("operator review does not bind all five candidate members")
    source_snapshot = _object(audited["source_snapshot"], name="operator review source snapshot")
    if dict(source_snapshot) != {
        "file_count": 28,
        "sha256": OPERATOR_CANDIDATE_SOURCE_SNAPSHOT_SHA256,
    }:
        raise ReceiptError("operator review source snapshot foreign key is invalid")
    review_bindings = _object(audited["bindings"], name="operator review contract bindings")
    expected_review_bindings = {
        key: value
        for key, value in bindings.items()
        if key
        not in {
            "generation_project_files",
            "review_project_files",
            "runtime_lock_sha256",
            "source_snapshot_sha256",
        }
    }
    if dict(review_bindings) != expected_review_bindings:
        raise ReceiptError("operator review and candidate contract bindings disagree")
    frozen_review_records = cast(list[object], bindings["review_project_files"])
    projected_review_records = [
        {
            "bytes": _object(record, name="candidate review project record")["bytes"],
            "path": _object(record, name="candidate review project record")["path"],
            "sha256": _object(record, name="candidate review project record")["sha256"],
        }
        for record in frozen_review_records
    ]
    if audited["project_files"] != projected_review_records:
        raise ReceiptError("operator review project ledger differs from frozen candidate history")
    runtime = _object(review["runtime_replay"], name="operator review runtime replay")
    generation = _object(runtime.get("generation"), name="operator review generation")
    expected_receipt_sha256 = next(
        digest
        for name, _byte_count, digest in OPERATOR_CANDIDATE_MEMBERS
        if name == "candidate_receipt.json"
    )
    if (
        generation.get("receipt_sha256") != expected_receipt_sha256
        or generation.get("member_count") != 5
        or generation.get("byte_identical_member_count") != 5
        or generation.get("byte_differences") != 0
    ):
        raise ReceiptError("operator review generation does not bind the exact five-member result")
    manifest_audit = _object(review["manifest_audit"], name="operator review manifest audit")
    if (
        manifest_audit.get("expected_row_count") != 10_368
        or manifest_audit.get("parsed_row_count") != 10_368
        or any(
            manifest_audit.get(key) != 0
            for key in (
                "duplicate_keys",
                "missing_keys",
                "extra_keys",
                "row_mismatches",
                "array_hash_mismatches",
                "canonical_byte_mismatches",
            )
        )
    ):
        raise ReceiptError("operator review manifest audit is incomplete or nonzero")
    if candidate_receipt["validated_row_count"] != manifest_audit["parsed_row_count"]:
        raise ReceiptError("operator review row count differs from the candidate receipt")


def load_frozen_operator_candidate(
    candidate_dir: Path,
    review_markdown_path: Path,
    review_json_path: Path,
) -> FrozenOperatorCandidate:
    """Read and validate the immutable operator-v2 candidate and detached review.

    The returned handle is candidate-only eligibility, never P2/Gate PASS.  The
    historical 28-file generation snapshot is verified as a detached foreign
    key; current live source files are deliberately not read or compared.
    """

    directory = Path(candidate_dir)
    try:
        before = directory.lstat()
    except FileNotFoundError as exc:
        raise ReceiptError("operator candidate directory does not exist") from exc
    if (
        directory.name != OPERATOR_CANDIDATE_DIRECTORY_NAME
        or stat.S_ISLNK(before.st_mode)
        or not stat.S_ISDIR(before.st_mode)
    ):
        raise ReceiptError("operator candidate must be the exact regular frozen directory")
    expected_names = {name for name, _byte_count, _digest_value in OPERATOR_CANDIDATE_MEMBERS}
    names_before = {path.name for path in directory.iterdir()}
    if names_before != expected_names:
        raise ReceiptError("operator candidate directory must contain exactly five members")
    encoded: dict[str, bytes] = {}
    typed_members: list[OperatorCandidateMember] = []
    for name, byte_count, expected_digest in OPERATOR_CANDIDATE_MEMBERS:
        member_bytes = read_regular_file(directory / name, max_bytes=byte_count)
        if len(member_bytes) != byte_count or sha256_bytes(member_bytes) != expected_digest:
            raise ReceiptError(f"operator candidate member {name!r} differs from frozen bytes")
        encoded[name] = member_bytes
        typed_members.append(
            OperatorCandidateMember(name=name, bytes=byte_count, sha256=expected_digest)
        )
    after = directory.lstat()
    names_after = {path.name for path in directory.iterdir()}
    if names_after != names_before or any(
        getattr(before, field) != getattr(after, field)
        for field in ("st_dev", "st_ino", "st_mtime_ns")
    ):
        raise ReceiptError("operator candidate directory changed while it was read")
    parsed = {
        name: parse_strict_json_bytes(member_bytes, canonical=True, max_bytes=len(member_bytes))
        for name, member_bytes in encoded.items()
    }
    candidate_receipt = _object(parsed["candidate_receipt.json"], name="candidate receipt")
    bindings, candidate_authority = _validate_operator_candidate_receipt(candidate_receipt)
    manifest = parsed["temporac.operator-manifest.v4.json"]
    _validate_operator_manifest(manifest)
    schema = _object(parsed["operator_manifest_schema.json"], name="operator manifest schema")
    if (
        schema.get("schema") != "temporac.operator-manifest-schema.v2"
        or schema.get("manifest_filename") != "temporac.operator-manifest.v4.json"
        or schema.get("manifest_row_count") != 10_368
        or schema.get("manifest_row_keys") != list(OPERATOR_MANIFEST_ROW_KEYS)
    ):
        raise ReceiptError("operator manifest schema metadata is invalid")
    runtime_lock = _object(parsed["environment_lock.json"], name="operator runtime lock")
    if runtime_lock.get("schema") != "temporac.operator-runtime-lock.v3":
        raise ReceiptError("operator runtime lock schema is invalid")
    witness = _object(parsed["replay_witness.json"], name="operator replay witness")
    _exact_keys(
        witness,
        frozenset(
            {
                "actual_inactive_b255",
                "aggregate_roots",
                "candidate_manifest_bytes",
                "candidate_manifest_sha256",
                "candidate_status",
                "normative_roots",
                "row_witnesses",
                "schema",
                "validated_row_count",
            }
        ),
        name="operator replay witness",
    )
    manifest_spec = next(
        (byte_count, digest)
        for name, byte_count, digest in OPERATOR_CANDIDATE_MEMBERS
        if name == "temporac.operator-manifest.v4.json"
    )
    schema_roots = _object(schema.get("normative_roots"), name="operator schema roots")
    witness_roots = _object(witness["normative_roots"], name="operator witness roots")
    root_crosswalk = {
        "edge0_digest_inventory": "edge0_digest_inventory_sha256",
        "inactive_lookup": "inactive_lookup_sha256",
        "key_inventory": "key_inventory_sha256",
        "semantic_inventory": "semantic_rows_sha256",
    }
    roots_match = all(
        _object(witness_roots.get(witness_name), name=f"operator {witness_name} root").get("sha256")
        == schema_roots.get(schema_name)
        for witness_name, schema_name in root_crosswalk.items()
    )
    if (
        witness["schema"] != "temporac.operator-replay-witness.v2"
        or witness["candidate_manifest_bytes"] != manifest_spec[0]
        or witness["candidate_manifest_sha256"] != manifest_spec[1]
        or witness["validated_row_count"] != 10_368
        or witness["candidate_status"] != candidate_receipt["candidate_status"]
        or not roots_match
    ):
        raise ReceiptError("operator replay witness foreign keys are invalid")
    review_markdown = read_regular_file(
        Path(review_markdown_path), max_bytes=OPERATOR_REVIEW_MARKDOWN_BYTES
    )
    review_json = read_regular_file(Path(review_json_path), max_bytes=OPERATOR_REVIEW_JSON_BYTES)
    if (
        len(review_markdown) != OPERATOR_REVIEW_MARKDOWN_BYTES
        or sha256_bytes(review_markdown) != OPERATOR_REVIEW_MARKDOWN_SHA256
        or len(review_json) != OPERATOR_REVIEW_JSON_BYTES
        or sha256_bytes(review_json) != OPERATOR_REVIEW_JSON_SHA256
    ):
        raise ReceiptError("detached operator review bytes differ from the trusted V2 review")
    try:
        decoded_review = parse_strict_json_bytes(
            review_json, canonical=False, max_bytes=OPERATOR_REVIEW_JSON_BYTES
        )
    except ContractError as exc:
        raise ReceiptError("operator review is not duplicate-free UTF-8 JSON") from exc
    _validate_operator_review(
        _object(decoded_review, name="operator review"),
        candidate_receipt=candidate_receipt,
        bindings=bindings,
    )
    expected_receipt_sha256 = next(
        digest
        for name, _byte_count, digest in OPERATOR_CANDIDATE_MEMBERS
        if name == "candidate_receipt.json"
    )
    return FrozenOperatorCandidate(
        members=tuple(typed_members),
        candidate_receipt_sha256=expected_receipt_sha256,
        generation_snapshot_sha256=OPERATOR_CANDIDATE_SOURCE_SNAPSHOT_SHA256,
        review_markdown_sha256=OPERATOR_REVIEW_MARKDOWN_SHA256,
        review_json_sha256=OPERATOR_REVIEW_JSON_SHA256,
        validated_row_count=10_368,
        candidate_only=True,
        eligible_for_separate_p2=True,
        authority=candidate_authority,
    )


def load_target_artifact(
    artifact_path: Path,
    receipt_path: Path,
    *,
    expected_receipt_sha256: str,
) -> LoadedTargetArtifact:
    """Jointly validate an on-disk target NPZ and its hash-bound closed receipt."""

    expected_receipt = _digest(
        expected_receipt_sha256,
        name="expected_receipt_sha256",
    )
    artifact = read_regular_file(artifact_path)
    encoded_receipt = read_regular_file(receipt_path)
    receipt_digest = sha256_bytes(encoded_receipt)
    if receipt_digest != expected_receipt:
        raise ReceiptError("target receipt bytes do not match the expected receipt hash")
    receipt = parse_receipt_bytes(
        encoded_receipt,
        expected_schema=TARGET_RECEIPT_SCHEMA,
    )
    arrays, members = read_target_npz_bytes(artifact)
    artifact_digest = sha256_bytes(artifact)
    if (
        receipt["artifact_bytes"] != len(artifact)
        or receipt["artifact_sha256"] != artifact_digest
        or receipt["members"] != [record.as_dict() for record in members]
        or arrays["contract_sha256"].tobytes(order="C").hex() != receipt["contract_sha256"]
        or int(arrays["source_kind"][0]) != receipt["source_kind"]
        or arrays["teacher_sha256"].tobytes(order="C").hex() != receipt["teacher_sha256"]
    ):
        raise ReceiptError("target NPZ and receipt hashes or identity fields do not agree")
    return LoadedTargetArtifact(
        arrays=arrays,
        artifact_bytes=artifact,
        artifact_sha256=artifact_digest,
        members=members,
        receipt=receipt,
        receipt_bytes=encoded_receipt,
        receipt_sha256=receipt_digest,
    )


def validate_feature_record_receipt(
    feature: FeatureRecord,
    encoded_receipt: bytes,
    *,
    expected_receipt_sha256: str,
) -> Mapping[str, object]:
    """Jointly bind a typed feature record to its detached canonical receipt."""

    if not isinstance(feature, FeatureRecord):
        raise ReceiptError("feature receipt binding requires a typed FeatureRecord")
    expected = _digest(expected_receipt_sha256, name="expected feature receipt sha256")
    if sha256_bytes(encoded_receipt) != expected:
        raise ReceiptError("feature receipt bytes do not match the expected receipt hash")
    receipt = parse_receipt_bytes(encoded_receipt, expected_schema=FEATURE_RECEIPT_SCHEMA)
    artifact, members = feature_npz_bytes(feature)
    if (
        receipt["artifact_bytes"] != len(artifact)
        or receipt["artifact_sha256"] != sha256_bytes(artifact)
        or receipt["members"] != [record.as_dict() for record in members]
        or receipt["opaque_key_hex"] != feature.opaque_key_bytes.hex()
        or receipt["slot"] != feature.slot
    ):
        raise ReceiptError("feature record and detached receipt are inconsistent")
    return receipt


def write_receipt_exclusive(
    path: Path,
    payload: Mapping[str, object],
    *,
    expected_schema: str | None = None,
    mode: int = 0o444,
) -> str:
    """Validate and exclusively publish one canonical receipt."""

    return write_bytes_exclusive(
        path,
        receipt_bytes(payload, expected_schema=expected_schema),
        mode=mode,
    )


def member_payload(records: Sequence[ArrayMemberRecord]) -> list[dict[str, object]]:
    """Convert typed members to their exact five-key JSON objects."""

    names = [record.name for record in records]
    if names != sorted(names, key=lambda value: value.encode("ascii")) or len(names) != len(
        set(names)
    ):
        raise ReceiptError("typed member records must be unique and ASCII sorted")
    return [record.as_dict() for record in records]


def build_target_artifact(
    target: CertifiedTarget,
) -> TargetArtifact:
    """Build the sole v4 CERTIFIED target NPZ/receipt pair in memory."""

    if not isinstance(target, CertifiedTarget) or not target.certified:
        raise ReceiptError("ABSTAIN has no target artifact or target receipt")
    provenance = target.provenance
    if provenance is None:
        raise ReceiptError("unbound certificate has no target artifact or target receipt")
    artifact, members = target_npz_bytes(target)
    artifact_digest = sha256_bytes(artifact)
    receipt_payload: dict[str, object] = {
        "artifact_bytes": len(artifact),
        "artifact_sha256": artifact_digest,
        "certificate_status": "CERTIFIED",
        "contract_sha256": CONTRACT_SHA256,
        "members": member_payload(members),
        "schema": TARGET_RECEIPT_SCHEMA,
        "source_key_hex": provenance.source_key_hex,
        "source_kind": provenance.source_kind,
        "source_unit_index": provenance.source_unit_index,
        "teacher_sha256": provenance.teacher_sha256,
    }
    encoded_receipt = receipt_bytes(receipt_payload, expected_schema=TARGET_RECEIPT_SCHEMA)
    return TargetArtifact(
        target=target,
        artifact_bytes=artifact,
        artifact_sha256=artifact_digest,
        members=members,
        receipt_bytes=encoded_receipt,
        receipt_sha256=sha256_bytes(encoded_receipt),
    )


def validate_target_count_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Validate and order the closed target-count-root row schema."""

    expected = frozenset({"opaque_key_hex", "slot", "teacher_target_count"})
    normalized: list[dict[str, object]] = []
    previous: tuple[str, int] | None = None
    for row in rows:
        _exact_keys(row, expected, name="target_count_root row")
        key = row["opaque_key_hex"]
        _hex_bytes(key, name="opaque_key_hex", byte_count=32)
        slot = _integer(row["slot"], name="slot")
        count = _integer(row["teacher_target_count"], name="teacher_target_count")
        identity = (cast(str, key), slot)
        if previous is not None and identity <= previous:
            raise ReceiptError("target count rows must be strictly ordered by opaque key and slot")
        previous = identity
        normalized.append({"opaque_key_hex": key, "slot": slot, "teacher_target_count": count})
    return tuple(normalized)


def target_count_root_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    return sha256_bytes(canonical_json_bytes(validate_target_count_rows(rows)))
