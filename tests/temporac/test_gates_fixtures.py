from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pams.temporac.fixtures import (
    OPERATOR_ROW_COUNT,
    TOPOLOGY_ROW_COUNT,
    TOPOLOGY_SEMANTIC_ROWS,
    CountMetricFixtureHashes,
    FixtureStatus,
    NamedHash,
    OperatorFixtureCommitment,
    OperatorFixtureKey,
    TopologyFixtureCommitment,
    check_operator_semantic_hash,
    check_topology_semantic_hash,
    operator_fixture_keys,
    operator_layout,
    operator_row_key_bytes,
    operator_row_key_preimage,
    operator_semantic_bytes,
    topology_semantic_bytes,
    validate_operator_inventory_witnesses,
    x0_generation_status,
)
from pams.temporac.gates import GateBlockedError, GateMachine, GateStatus
from pams.temporac.types import ContractError

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_RECEIPT = (
    ROOT / "data/temporac_p00_v4/operator_candidate_v2_20260816/candidate_receipt.json"
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_local_gate_machine_stops_at_pending_p3_and_never_launches() -> None:
    machine = GateMachine()
    machine.pass_p0("1" * 64)
    machine.pass_p1("2" * 64)
    machine.pass_p2("3" * 64, "4" * 64)
    assert machine.next_gate == "P3"
    assert machine.records["P2-FIXTURE"].status is GateStatus.PASS
    assert machine.records["P2-METRIC"].status is GateStatus.PASS
    assert machine.records["P3"].status is GateStatus.PENDING
    assert machine.records["S0"].status is GateStatus.PENDING
    assert not machine.launch_authorized
    assert not machine.capability_request_allowed
    with pytest.raises(GateBlockedError, match="P3 is PENDING"):
        machine.request_s0_pass("5" * 64)
    with pytest.raises(GateBlockedError, match="S0 is not PASS"):
        machine.record_campaign_gate("G0", "6" * 64)


def test_gate_failure_blocks_every_later_receipt() -> None:
    machine = GateMachine()
    machine.pass_p0("1" * 64)
    machine.fail_current("P1", "2" * 64, detail="static import graph mismatch")
    assert machine.killed
    assert machine.records["P1"].status is GateStatus.FAIL
    assert machine.records["P2-FIXTURE"].status is GateStatus.BLOCKED
    assert machine.records["K7"].status is GateStatus.BLOCKED
    with pytest.raises(GateBlockedError, match="killed"):
        machine.pass_p1("3" * 64)


def test_fixture_inventories_and_typed_normative_blockers() -> None:
    candidate_receipt = json.loads(CANDIDATE_RECEIPT.read_bytes())
    assert candidate_receipt["schema"] == "temporac.operator-manifest-candidate-receipt.v3"
    assert candidate_receipt["p2_status"] == candidate_receipt["s0_status"] == "NOT_CLAIMED"
    assert set(candidate_receipt["authority"]) == {
        "P2",
        "S0",
        "data",
        "gate",
        "git",
        "launch",
        "paper_claim",
        "server",
        "training",
    }
    assert not any(candidate_receipt["authority"].values())

    x0_status = x0_generation_status()
    assert x0_status.status is FixtureStatus.BLOCKED_PENDING_F8_AMPLITUDE_AMENDMENT
    assert x0_status.row_count == 0

    topology_payload = topology_semantic_bytes()
    topology = check_topology_semantic_hash(_sha(topology_payload))
    assert topology.status is FixtureStatus.VALIDATED
    assert topology.row_count == TOPOLOGY_ROW_COUNT == 13

    keys = operator_fixture_keys()
    assert len(keys) == OPERATOR_ROW_COUNT == 10_368
    assert keys[0] == OperatorFixtureKey(4, 1, 0.60, 0, "interior")
    assert keys[-1] == OperatorFixtureKey(129, 4, 1.00, 31, "run-reset")
    operator_payload = operator_semantic_bytes()
    operator = check_operator_semantic_hash(_sha(operator_payload))
    assert operator.status is FixtureStatus.VALIDATED
    assert operator_row_key_bytes(keys[0]).hex() == "00040001000000000000"
    assert operator_row_key_preimage(keys[0]).hex() == (
        "74656d706f7261632e6f70657261746f722e763400000000000000000a0004000100000000000000000000"
    )
    witnesses = validate_operator_inventory_witnesses()
    assert witnesses.row_count == witnesses.unique_key_count == 10_368
    assert witnesses.semantic_bytes == 746_714
    with pytest.raises(ContractError, match="committed"):
        check_operator_semantic_hash("0" * 64)

    topology_commitment = TopologyFixtureCommitment(
        semantic=TOPOLOGY_SEMANTIC_ROWS[0],
        operator_parameter_sha256="1" * 64,
        array_hashes=(NamedHash("pulse", "2" * 64),),
        expected_field_hashes=(NamedHash("status", "3" * 64),),
    )
    assert topology_commitment.semantic.identifier == "T00"
    operator_commitment = OperatorFixtureCommitment(
        key=keys[0],
        truth_plateaus_sha256="1" * 64,
        masks_sha256="2" * 64,
        expected_components_sha256="3" * 64,
        generated_response_sha256="4" * 64,
    )
    assert operator_commitment.key == keys[0]


def test_operator_layout_boundaries_and_reset_are_exact() -> None:
    interior = operator_layout(OperatorFixtureKey(4, 1, 0.60, 0, "interior"))
    assert interior.run_bounds == ((0, 128),)
    assert interior.truth_plateaus == ((32, 33), (37, 38), (42, 43))
    assert interior.truth_negative_components == 4

    right = operator_layout(OperatorFixtureKey(4, 1, 0.60, 0, "right-boundary"))
    assert right.truth_plateaus == ((117, 118), (122, 123), (127, 128))
    assert right.truth_plateaus[-1][1] == right.edge_count

    reset = operator_layout(OperatorFixtureKey(4, 1, 0.60, 0, "run-reset"))
    assert reset.run_bounds == ((0, 32), (33, 128))
    assert reset.reset_edge == 32
    assert reset.truth_plateaus == ((0, 1), (5, 6), (33, 34))
    assert reset.truth_negative_components == 3


def test_desensitized_count_metric_hash_index_is_fail_closed() -> None:
    payloads = {
        "evaluator_code": b"synthetic evaluator source",
        "fixture_input": b"opaque-video-0,positive-count-2",
        "expected_output": b"avgmae=0.25,avgobo=1.0",
        "attack_global_fail": b"duplicate-person=>GLOBAL_FAIL",
    }
    hashes = CountMetricFixtureHashes(
        evaluator_code_sha256=_sha(payloads["evaluator_code"]),
        fixture_input_sha256=_sha(payloads["fixture_input"]),
        expected_output_sha256=_sha(payloads["expected_output"]),
        attack_global_fail_sha256=_sha(payloads["attack_global_fail"]),
    )
    hashes.verify(**payloads)
    assert hashes.index_bytes().endswith(b"\n")
    with pytest.raises(ContractError, match="committed"):
        hashes.verify(
            evaluator_code=b"changed",
            fixture_input=payloads["fixture_input"],
            expected_output=payloads["expected_output"],
            attack_global_fail=payloads["attack_global_fail"],
        )
