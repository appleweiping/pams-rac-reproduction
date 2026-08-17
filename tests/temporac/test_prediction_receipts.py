from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pams.temporac.contract import (
    CONTRACT_SHA256,
    FEATURE_MEMBER_SCHEMA,
    FEATURE_RECEIPT_SCHEMA,
    G1_TEACHER_SELECTION_RECEIPT_SCHEMA,
    K1_CERTIFICATE_OUTCOME_INDEX_SCHEMA,
    K1_CERTIFICATE_OUTCOME_RECEIPT_SCHEMA,
    K3_RECEIPT_SCHEMA,
    K4_RECEIPT_SCHEMA,
    NATURAL_PREDICTION_COMPLETION_RECEIPT_SCHEMA,
    PRE_G5A_OWNER_ROSTER,
    PRE_G5A_RECEIPT_INVENTORY_SCHEMA,
    PRE_G5A_STAGE_RECEIPT_SCHEMA,
    PREDICTION_MEMBER_SCHEMA,
    RECEIPT_DAG_SCHEMA,
    RUN_RECEIPT_SCHEMA,
    X0_INFERENCE_RECEIPT_SCHEMA,
    ReasonCode,
)
from pams.temporac.hashio import canonical_json_bytes, deterministic_npz_bytes
from pams.temporac.receipts import (
    ReceiptError,
    k1_outcome_row_bytes,
    k1_outcome_row_sha256,
    member_payload,
    natural_prediction_completion_receipt,
    parse_k1_outcome_index_bytes,
    parse_k1_outcome_row_bytes,
    parse_pre_g5a_receipt_inventory_bytes,
    parse_receipt_bytes,
    parse_receipt_dag_bytes,
    receipt_bytes,
    target_count_root_sha256,
    validate_k1_outcome_index,
    validate_k1_outcome_row,
    validate_pre_g5a_receipt_evidence,
    validate_pre_g5a_receipt_inventory,
    validate_receipt,
    validate_receipt_dag,
    write_receipt_exclusive,
)
from pams.temporac.types import PredictionRecord


def _u8_hash() -> np.ndarray:
    return np.frombuffer(bytes.fromhex(CONTRACT_SHA256), dtype="|u1").copy()


def _feature_arrays() -> dict[str, np.ndarray]:
    return {
        "frame_mask": np.ones(320, dtype="|u1"),
        "local_person_slot": np.asarray([0], dtype="<i8"),
        "motion": np.zeros((320, 17, 3), dtype="<f4"),
        "opaque_sample_key": np.zeros(32, dtype="|u1"),
        "person_mask": np.asarray([1], dtype="|u1"),
        "sampled_frame_indices": np.arange(320, dtype="<i8"),
        "source_length": np.asarray([320], dtype="<i8"),
    }


def _stub() -> PredictionRecord:
    return PredictionRecord(
        abstain=np.asarray([1], dtype="|u1"),
        abstain_reasons=np.asarray([ReasonCode.TOO_SHORT], dtype="<u2"),
        component_bounds=np.empty((0, 2), dtype="<i4"),
        component_location=np.empty(0, dtype="<i4"),
        component_score=np.empty(0, dtype="<f4"),
        condition=np.asarray([0], dtype="|u1"),
        contract_sha256=_u8_hash(),
        count=np.asarray([-1], dtype="<i8"),
        decoder_mask=np.empty(0, dtype="|u1"),
        edge_mask=np.empty(0, dtype="|u1"),
        local_person_slot=np.asarray([0], dtype="<i8"),
        opaque_sample_key=np.zeros(32, dtype="|u1"),
        response=np.empty(0, dtype="<f4"),
        run_bounds=np.empty((0, 2), dtype="<i4"),
    )


def test_prediction_stub_and_nonabstention_validate_exact_decoder_semantics() -> None:
    stub = _stub()
    assert not stub.response.flags.writeable
    deterministic_npz_bytes(stub.as_arrays(), schema=PREDICTION_MEMBER_SCHEMA)
    response = np.asarray([0.1, 0.5, 0.8, 0.2, 0.9], dtype="<f4")
    prediction = PredictionRecord(
        abstain=np.asarray([0], dtype="|u1"),
        abstain_reasons=np.empty(0, dtype="<u2"),
        component_bounds=np.asarray([[1, 3], [4, 5]], dtype="<i4"),
        component_location=np.asarray([2, 4], dtype="<i4"),
        component_score=np.asarray([0.8, 0.9], dtype="<f4"),
        condition=np.asarray([1], dtype="|u1"),
        contract_sha256=_u8_hash(),
        count=np.asarray([2], dtype="<i8"),
        decoder_mask=np.ones(5, dtype="|u1"),
        edge_mask=np.ones(5, dtype="|u1"),
        local_person_slot=np.asarray([4], dtype="<i8"),
        opaque_sample_key=np.arange(32, dtype="|u1"),
        response=response,
        run_bounds=np.asarray([[0, 5]], dtype="<i4"),
    )
    assert int(prediction.count[0]) == 2
    with pytest.raises(ValueError, match="count"):
        PredictionRecord(
            **{
                **{
                    name: np.array(value, copy=True)
                    for name, value in prediction.as_arrays().items()
                },
                "count": np.asarray([1], dtype="<i8"),
            }
        )


def test_feature_receipt_is_canonical_closed_and_exclusive(tmp_path: Path) -> None:
    artifact, members = deterministic_npz_bytes(_feature_arrays(), schema=FEATURE_MEMBER_SCHEMA)
    payload: dict[str, object] = {
        "artifact_bytes": len(artifact),
        "artifact_sha256": "1" * 64,
        "contract_sha256": CONTRACT_SHA256,
        "members": member_payload(members),
        "opaque_key_hex": "2" * 64,
        "schema": FEATURE_RECEIPT_SCHEMA,
        "slot": 0,
        "source_binding_sha256": "3" * 64,
    }
    validate_receipt(payload, expected_schema=FEATURE_RECEIPT_SCHEMA)
    encoded = receipt_bytes(payload)
    assert encoded.endswith(b"\n") and not encoded.endswith(b"\n\n")
    assert parse_receipt_bytes(encoded) == payload
    path = tmp_path / "receipt.json"
    digest = write_receipt_exclusive(path, payload)
    assert len(digest) == 64 and path.read_bytes() == encoded
    with pytest.raises(ValueError, match="overwrite"):
        write_receipt_exclusive(path, payload)
    with pytest.raises(ReceiptError, match="unknown/missing"):
        validate_receipt({**payload, "unknown": 1})
    missing = dict(payload)
    del missing["slot"]
    with pytest.raises(ReceiptError, match="unknown/missing"):
        validate_receipt(missing)


def test_target_count_root_rows_are_closed_and_strictly_ordered() -> None:
    rows = [
        {"opaque_key_hex": "0" * 64, "slot": 0, "teacher_target_count": 3},
        {"opaque_key_hex": "1" * 64, "slot": 0, "teacher_target_count": 4},
    ]
    assert len(target_count_root_sha256(rows)) == 64
    with pytest.raises(ReceiptError, match="strictly ordered"):
        target_count_root_sha256(list(reversed(rows)))
    with pytest.raises(ReceiptError, match="unknown/missing"):
        target_count_root_sha256([{**rows[0], "count_gt": 3}])


def _inventory_receipt_payload(
    *,
    index: int,
    owner: str,
    schema: str,
    upstream: list[str],
) -> dict[str, object]:
    digest = f"{index + 1000:064x}"
    if schema == PRE_G5A_STAGE_RECEIPT_SCHEMA:
        return {
            "code_index_sha256": digest,
            "contract_sha256": CONTRACT_SHA256,
            "environment_sha256_or_null": None if index < 7 else "e" * 64,
            "owner": owner,
            "payload_index_bytes": 1,
            "payload_index_sha256": "d" * 64,
            "schema": schema,
            "status": "PASS",
            "upstream_receipt_sha256": upstream,
        }
    if schema == RUN_RECEIPT_SCHEMA:
        seed = int(owner.rsplit("=", 1)[1])
        final_step = 20_000 if "/teacher/" in owner else 10_000
        return {
            "code_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "contract_sha256": CONTRACT_SHA256,
            "environment_sha256": "3" * 64,
            "final_step": final_step,
            "job_name_hex": owner.encode("ascii").hex(),
            "optimizer_sha256": "4" * 64,
            "ordered_checkpoint_sha256": [
                "5" * 64 for _ in range(40 if final_step == 20_000 else 20)
            ],
            "resource": {
                "completed_steps": final_step,
                "gpu_seconds": 0.0,
                "max_cuda_bytes": 0,
                "wall_seconds": 0.0,
            },
            "rng_sha256": "6" * 64,
            "schema": schema,
            "seed": seed,
            "source_cycle_sha256": "7" * 64,
            "status": "SUCCESS",
            "upstream_receipt_sha256": upstream,
        }
    if schema == G1_TEACHER_SELECTION_RECEIPT_SCHEMA:
        return {
            "code_index_sha256": "1" * 64,
            "contract_sha256": CONTRACT_SHA256,
            "environment_sha256": "2" * 64,
            "schema": schema,
            "score_index_sha256": "3" * 64,
            "selection_receipt_sha256": "4" * 64,
            "status": "PASS",
            "teacher_checkpoint_evidence_index_sha256": "5" * 64,
            "teacher_run_receipt_sha256": upstream,
            "teacher_tune_input_receipt_sha256": "6" * 64,
        }
    if schema == K1_CERTIFICATE_OUTCOME_RECEIPT_SCHEMA:
        return {
            "contract_sha256": CONTRACT_SHA256,
            "development_certified_component_count": 8,
            "development_certified_identity_count": 108,
            "outcome_index_bytes": 1,
            "outcome_index_sha256": "1" * 64,
            "schema": schema,
            "status": "PASS",
            "train_certified_component_count": 16,
            "train_certified_identity_count": 216,
        }
    if schema == X0_INFERENCE_RECEIPT_SCHEMA:
        seed = int(owner.removeprefix("X0I-"))
        return {
            "code_index_sha256": "1" * 64,
            "contract_sha256": CONTRACT_SHA256,
            "environment_sha256": "2" * 64,
            "owner": owner,
            "prediction_index_sha256": "3" * 64,
            "schema": schema,
            "seed": seed,
            "status": "PASS",
            "upstream_receipt_sha256": upstream,
        }
    if schema in {K3_RECEIPT_SCHEMA, K4_RECEIPT_SCHEMA}:
        return {
            "code_index_sha256": "1" * 64,
            "contract_sha256": CONTRACT_SHA256,
            "environment_sha256": "2" * 64,
            "owner": owner,
            "payload_index_bytes": 1,
            "payload_index_sha256": "3" * 64,
            "schema": schema,
            "status": "PASS",
            "upstream_receipt_sha256": upstream,
        }
    if schema == NATURAL_PREDICTION_COMPLETION_RECEIPT_SCHEMA:
        seed = int(owner.removeprefix("NATP-"))
        payload = _natp_payload(seed)
        payload["k6_receipt_sha256"] = upstream[0]
        payload["upstream_receipt_sha256"] = upstream
        return payload
    raise AssertionError(f"unsupported inventory receipt schema {schema}")


def _inventory_bundle() -> tuple[dict[str, object], dict[str, bytes]]:
    receipt_by_owner: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    evidence: dict[str, bytes] = {}
    for index, (node_class, owner, schema, upstream_owners) in enumerate(PRE_G5A_OWNER_ROSTER):
        upstream = [receipt_by_owner[upstream_owner] for upstream_owner in upstream_owners]
        receipt_payload = _inventory_receipt_payload(
            index=index,
            owner=owner,
            schema=schema,
            upstream=upstream,
        )
        encoded_receipt = canonical_json_bytes(receipt_payload)
        receipt_sha256 = hashlib.sha256(encoded_receipt).hexdigest()
        rows.append(
            {
                "node_class": node_class,
                "owner": owner,
                "receipt_schema": schema,
                "receipt_sha256": receipt_sha256,
                "upstream_owner_tokens": list(upstream_owners),
                "upstream_receipt_sha256": upstream,
            }
        )
        receipt_by_owner[owner] = receipt_sha256
        evidence[receipt_sha256] = encoded_receipt
    return (
        {
            "contract_sha256": CONTRACT_SHA256,
            "rows": rows,
            "schema": PRE_G5A_RECEIPT_INVENTORY_SCHEMA,
        },
        evidence,
    )


def _inventory_payload() -> dict[str, object]:
    return _inventory_bundle()[0]


def _natp_payload(seed: int = 20260815) -> dict[str, object]:
    return {
        "clean_seed_index_sha256": "1" * 64,
        "clean_seed_root_sha256": "2" * 64,
        "code_index_sha256": "3" * 64,
        "contract_sha256": CONTRACT_SHA256,
        "drift_seed_index_sha256": "4" * 64,
        "drift_seed_root_sha256": "5" * 64,
        "environment_sha256": "6" * 64,
        "k6_receipt_sha256": "7" * 64,
        "owner": f"NATP-{seed}",
        "pilot_scope_manifest_sha256": "8" * 64,
        "schema": NATURAL_PREDICTION_COMPLETION_RECEIPT_SCHEMA,
        "seed": seed,
        "status": "PASS",
        "upstream_receipt_sha256": ["7" * 64],
    }


def test_run_receipt_resource_rejects_negative_zero_but_accepts_positive_zero() -> None:
    payload = _inventory_receipt_payload(
        index=11,
        owner="temporac.execution.v4/teacher/seed=20260815",
        schema=RUN_RECEIPT_SCHEMA,
        upstream=["a" * 64, "b" * 64],
    )
    validate_receipt(payload, expected_schema=RUN_RECEIPT_SCHEMA)
    for field in ("gpu_seconds", "wall_seconds"):
        attacked = dict(payload)
        raw_resource = attacked["resource"]
        assert isinstance(raw_resource, dict)
        resource = dict(raw_resource)
        resource[field] = -0.0
        attacked["resource"] = resource
        with pytest.raises(ReceiptError, match="finite nonnegative"):
            validate_receipt(attacked, expected_schema=RUN_RECEIPT_SCHEMA)


def _k1_row(index: int, *, split: str) -> dict[str, object]:
    identity = f"{index + 1:064x}"
    return {
        "certificate_reasons": [],
        "certificate_status": "CERTIFIED",
        "component_key_hex": f"{10_000 + index % 10:064x}",
        "eligible": True,
        "feature_artifact_sha256_or_null": "1" * 64,
        "feature_receipt_sha256_or_null": "2" * 64,
        "natural_input_receipt_sha256_or_null": "3" * 64,
        "opaque_key_hex": identity,
        "population_row_sha256": hashlib.sha256(f"{split}:{index}".encode()).hexdigest(),
        "slot": 0,
        "split": split,
        "target_artifact_sha256_or_null": "4" * 64,
        "target_receipt_sha256_or_null": "5" * 64,
    }


def test_natp_exact14_binds_only_k6_and_rejects_type_null_hex_and_extra_attacks() -> None:
    payload = _natp_payload()
    encoded = receipt_bytes(payload, expected_schema=NATURAL_PREDICTION_COMPLETION_RECEIPT_SCHEMA)
    typed = natural_prediction_completion_receipt(parse_receipt_bytes(encoded))
    assert typed.upstream_receipt_sha256 == (typed.k6_receipt_sha256,)
    for attack in (
        {**payload, "upstream_receipt_sha256": ["9" * 64]},
        {**payload, "owner": "NATP-20260816"},
        {**payload, "seed": None},
        {**payload, "k6_receipt_sha256": "A" * 64},
        {**payload, "extra": False},
    ):
        with pytest.raises(ReceiptError):
            validate_receipt(attack)


def test_pre_g5a_inventory_freezes_54_owners_106_edges_and_k1_g0_g1_ordinals() -> None:
    payload = _inventory_payload()
    inventory = validate_pre_g5a_receipt_inventory(payload)
    assert len(inventory.rows) == 54
    assert len(inventory.direct_edges) == 106
    assert inventory.topological_receipt_sha256 == tuple(
        row.receipt_sha256 for row in inventory.rows
    )
    k1 = inventory.rows[15]
    k1_edges = sorted(
        (edge for edge in inventory.direct_edges if edge.from_receipt_sha256 == k1.receipt_sha256),
        key=lambda edge: edge.ordinal,
    )
    assert [(edge.ordinal, edge.role) for edge in k1_edges] == [
        (0, "k1.upstream"),
        (1, "k1.upstream"),
    ]
    assert [edge.to_receipt_sha256 for edge in k1_edges] == [
        inventory.rows[9].receipt_sha256,
        inventory.rows[14].receipt_sha256,
    ]
    encoded = canonical_json_bytes(payload)
    assert parse_pre_g5a_receipt_inventory_bytes(encoded) == inventory

    reordered = _inventory_payload()
    rows = reordered["rows"]
    assert isinstance(rows, list)
    rows[14], rows[15] = rows[15], rows[14]
    with pytest.raises(ReceiptError, match="frozen owner roster"):
        validate_pre_g5a_receipt_inventory(reordered)
    wrong_edge = _inventory_payload()
    wrong_rows = wrong_edge["rows"]
    assert isinstance(wrong_rows, list) and isinstance(wrong_rows[15], dict)
    wrong_rows[15]["upstream_receipt_sha256"][0] = "f" * 64
    with pytest.raises(ReceiptError, match="prior owner"):
        validate_pre_g5a_receipt_inventory(wrong_edge)
    duplicate = encoded.replace(
        b'{"contract_sha256":',
        b'{"contract_sha256":"' + CONTRACT_SHA256.encode() + b'","contract_sha256":',
        1,
    )
    with pytest.raises(ReceiptError, match="duplicate-free"):
        parse_pre_g5a_receipt_inventory_bytes(duplicate)


def test_receipt_dag_is_typed_canonical_acyclic_and_contains_exact_roster_edges() -> None:
    inventory_payload, evidence = _inventory_bundle()
    inventory = validate_pre_g5a_receipt_inventory(inventory_payload)
    owner_keys = validate_pre_g5a_receipt_evidence(inventory, evidence)
    nodes: list[dict[str, object]] = sorted(
        (
            {
                "class": row.node_class,
                "owner_key": owner_keys[row.receipt_sha256],
                "receipt_sha256": row.receipt_sha256,
            }
            for row in inventory.rows
        ),
        key=lambda row: (
            str(row["class"]).encode("ascii"),
            bytes.fromhex(str(row["owner_key"])),
            bytes.fromhex(str(row["receipt_sha256"])),
        ),
    )
    edge_payloads = [edge.as_dict() for edge in inventory.direct_edges]
    payload: dict[str, object] = {
        "contract_sha256": CONTRACT_SHA256,
        "edges": edge_payloads,
        "nodes": nodes,
        "schema": RECEIPT_DAG_SCHEMA,
    }
    dag = validate_receipt_dag(
        payload,
        inventory=inventory,
        receipt_bytes_by_sha256=evidence,
    )
    assert len(dag.nodes) == 54 and len(dag.edges) == 106
    assert (
        parse_receipt_dag_bytes(
            canonical_json_bytes(payload),
            inventory=inventory,
            receipt_bytes_by_sha256=evidence,
        )
        == dag
    )

    with pytest.raises(ReceiptError, match="requires exact pre-G5a"):
        validate_receipt_dag(payload)
    with pytest.raises(ReceiptError, match="requires exact pre-G5a"):
        validate_receipt_dag(payload, inventory=inventory)

    missing = {**payload, "edges": edge_payloads[:-1]}
    with pytest.raises(ReceiptError, match="exact 106-edge roster"):
        validate_receipt_dag(
            missing,
            inventory=inventory,
            receipt_bytes_by_sha256=evidence,
        )
    cyclic_edges = list(edge_payloads)
    cyclic_edges.append(
        {
            "from_receipt_sha256": inventory.rows[0].receipt_sha256,
            "ordinal": 0,
            "role": "stage.upstream",
            "to_receipt_sha256": inventory.rows[-1].receipt_sha256,
        }
    )
    cyclic_edges.sort(
        key=lambda edge: (
            bytes.fromhex(str(edge["from_receipt_sha256"])),
            str(edge["role"]).encode("ascii"),
            int(str(edge["ordinal"])),
            bytes.fromhex(str(edge["to_receipt_sha256"])),
        )
    )
    with pytest.raises(ReceiptError, match="cycle"):
        validate_receipt_dag(
            {**payload, "edges": cyclic_edges},
            inventory=inventory,
            receipt_bytes_by_sha256=evidence,
        )


def test_receipt_dag_rejects_node_edge_and_evidence_closure_attacks() -> None:
    inventory_payload, evidence = _inventory_bundle()
    inventory = validate_pre_g5a_receipt_inventory(inventory_payload)
    owner_keys = validate_pre_g5a_receipt_evidence(inventory, evidence)
    nodes: list[dict[str, object]] = sorted(
        (
            {
                "class": row.node_class,
                "owner_key": owner_keys[row.receipt_sha256],
                "receipt_sha256": row.receipt_sha256,
            }
            for row in inventory.rows
        ),
        key=lambda row: (
            str(row["class"]).encode("ascii"),
            bytes.fromhex(str(row["owner_key"])),
            bytes.fromhex(str(row["receipt_sha256"])),
        ),
    )
    edges = [edge.as_dict() for edge in inventory.direct_edges]

    def sort_nodes(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return sorted(
            rows,
            key=lambda row: (
                str(row["class"]).encode("ascii"),
                bytes.fromhex(str(row["owner_key"])),
                bytes.fromhex(str(row["receipt_sha256"])),
            ),
        )

    def sort_edges(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        return sorted(
            rows,
            key=lambda edge: (
                bytes.fromhex(str(edge["from_receipt_sha256"])),
                str(edge["role"]).encode("ascii"),
                int(str(edge["ordinal"])),
                bytes.fromhex(str(edge["to_receipt_sha256"])),
            ),
        )

    def reject(
        *,
        attacked_nodes: list[dict[str, object]] | None = None,
        attacked_edges: list[dict[str, object]] | None = None,
    ) -> None:
        with pytest.raises(ReceiptError):
            validate_receipt_dag(
                {
                    "contract_sha256": CONTRACT_SHA256,
                    "edges": edges if attacked_edges is None else attacked_edges,
                    "nodes": nodes if attacked_nodes is None else attacked_nodes,
                    "schema": RECEIPT_DAG_SCHEMA,
                },
                inventory=inventory,
                receipt_bytes_by_sha256=evidence,
            )

    reject(attacked_nodes=nodes[:-1])
    reject(attacked_nodes=sort_nodes([*nodes, dict(nodes[0])]))
    reject(
        attacked_nodes=sort_nodes(
            [*nodes, {"class": "feature", "owner_key": "f" * 64, "receipt_sha256": "e" * 64}]
        )
    )
    wrong_class = [dict(row) for row in nodes]
    wrong_class[0]["class"] = "feature"
    reject(attacked_nodes=sort_nodes(wrong_class))
    wrong_owner = [dict(row) for row in nodes]
    wrong_owner[0]["owner_key"] = "f" * 64
    reject(attacked_nodes=sort_nodes(wrong_owner))

    reject(attacked_edges=sort_edges([*edges, dict(edges[0])]))
    extra_role = [
        *edges,
        {
            "from_receipt_sha256": inventory.rows[-1].receipt_sha256,
            "ordinal": 0,
            "role": "g1.selection",
            "to_receipt_sha256": inventory.rows[0].receipt_sha256,
        },
    ]
    reject(attacked_edges=sort_edges(extra_role))
    wrong_ordinal = [dict(edge) for edge in edges]
    ordinal = wrong_ordinal[0]["ordinal"]
    assert type(ordinal) is int
    wrong_ordinal[0]["ordinal"] = ordinal + 1
    reject(attacked_edges=sort_edges(wrong_ordinal))
    dangling = [dict(edge) for edge in edges]
    dangling[0]["to_receipt_sha256"] = "f" * 64
    reject(attacked_edges=sort_edges(dangling))

    missing_evidence = dict(evidence)
    missing_evidence.pop(next(iter(missing_evidence)))
    with pytest.raises(ReceiptError, match="exactly all 54"):
        validate_pre_g5a_receipt_evidence(inventory, missing_evidence)
    extra_evidence = {**evidence, "f" * 64: b"{}\n"}
    with pytest.raises(ReceiptError, match="exactly all 54"):
        validate_pre_g5a_receipt_evidence(inventory, extra_evidence)
    mutable_evidence: dict[str, bytes | bytearray] = dict(evidence)
    first_digest = next(iter(mutable_evidence))
    mutable_evidence[first_digest] = bytearray(mutable_evidence[first_digest])
    with pytest.raises(ReceiptError, match="immutable bytes"):
        validate_pre_g5a_receipt_evidence(inventory, mutable_evidence)  # type: ignore[arg-type]
    changed_evidence = dict(evidence)
    changed_evidence[first_digest] += b" "
    with pytest.raises(ReceiptError, match="digest"):
        validate_pre_g5a_receipt_evidence(inventory, changed_evidence)


@pytest.mark.parametrize(
    "field,value",
    (
        ("owner", "NATP-20260816"),
        ("schema", X0_INFERENCE_RECEIPT_SCHEMA),
        ("extra", False),
    ),
)
def test_inventory_receipt_bytes_bind_schema_typed_owner_and_closed_payload(
    field: str,
    value: object,
) -> None:
    inventory_payload, evidence = _inventory_bundle()
    rows = inventory_payload["rows"]
    assert isinstance(rows, list) and isinstance(rows[-1], dict)
    original_digest = str(rows[-1]["receipt_sha256"])
    terminal = json.loads(evidence[original_digest])
    terminal[field] = value
    encoded = canonical_json_bytes(terminal)
    replacement_digest = hashlib.sha256(encoded).hexdigest()
    rows[-1]["receipt_sha256"] = replacement_digest
    evidence.pop(original_digest)
    evidence[replacement_digest] = encoded
    inventory = validate_pre_g5a_receipt_inventory(inventory_payload)
    with pytest.raises(ReceiptError, match="closed typed payload"):
        validate_pre_g5a_receipt_evidence(inventory, evidence)


def test_k1_standalone_row_preimage_and_402_row_index_are_closed_and_ordered() -> None:
    row_payload = _k1_row(0, split="train")
    row = validate_k1_outcome_row(row_payload)
    encoded = k1_outcome_row_bytes(row)
    assert encoded.endswith(b"\n") and not encoded.endswith(b"\n\n")
    assert k1_outcome_row_sha256(row) == hashlib.sha256(encoded).hexdigest()
    assert parse_k1_outcome_row_bytes(encoded) == row
    with pytest.raises(ReceiptError):
        validate_k1_outcome_row({**row_payload, "target_receipt_sha256_or_null": None})
    with pytest.raises(ReceiptError):
        validate_k1_outcome_row({**row_payload, "certificate_reasons": None})
    with pytest.raises(ReceiptError, match="duplicate-free"):
        parse_k1_outcome_row_bytes(
            encoded.replace(b'{"certificate_reasons":', b'{"slot":0,"certificate_reasons":', 1)
        )

    rows = [*(_k1_row(index, split="train") for index in range(268))]
    rows.extend(_k1_row(index, split="val") for index in range(134))
    payload: dict[str, object] = {
        "contract_sha256": CONTRACT_SHA256,
        "population_manifest_sha256": "a" * 64,
        "rows": rows,
        "schema": K1_CERTIFICATE_OUTCOME_INDEX_SCHEMA,
    }
    index = validate_k1_outcome_index(payload)
    assert len(index.rows) == len(index.row_sha256) == 402
    assert parse_k1_outcome_index_bytes(canonical_json_bytes(payload)) == index
    with pytest.raises(ReceiptError, match="unique in train/val"):
        validate_k1_outcome_index({**payload, "rows": list(reversed(rows))})
