"""Closed-schema checks for the non-authorizing X0 manifest candidate."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import scipy

from pams.temporac import contract, x0

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "data/temporac_p00_v4/x0_candidate_20260816"
EXPECTED_CONTRACT_SHA256 = "3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391"
MANIFEST_KEYS = frozenset(
    {
        "accepted_attempt",
        "coefficient_bytes_sha256",
        "coefficient_seed_u64",
        "generator_contract_sha256",
        "id",
        "orbit_grid_sha256",
        "source_key_hex",
        "split",
    }
)
ATTEMPT_KEYS = frozenset(
    {
        "attempt",
        "coefficient_bytes_sha256",
        "coefficient_seed_u64",
        "guard_pass",
        "id",
        "orbit_grid_sha256",
    }
)
RECEIPT_KEYS = frozenset(
    {
        "artifact_type",
        "authoritative",
        "authorizes",
        "bindings",
        "blockers",
        "candidate_status",
        "members",
        "p2_status",
        "s0_status",
    }
)


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_file(path: Path) -> Any:
    payload = path.read_bytes()
    assert payload.endswith(b"\n")
    assert not payload.endswith(b"\n\n")
    parsed = json.loads(payload)
    assert payload == _canonical_json_bytes(parsed)
    return parsed


def test_x0_manifest_is_exact_closed_canonical_inventory() -> None:
    payload = x0.x0_manifest(EXPECTED_CONTRACT_SHA256)
    rows = json.loads(payload)
    assert payload == _canonical_json_bytes(rows)
    assert len(rows) == 40
    assert [row["id"] for row in rows] == [f"U{source_id:04d}" for source_id in range(40)]
    assert all(frozenset(row) == MANIFEST_KEYS for row in rows)
    assert [row["split"] for row in rows] == ["train"] * 24 + ["tune"] * 8 + ["heldout"] * 8
    assert {row["accepted_attempt"] for row in rows} == {0}
    assert {row["generator_contract_sha256"] for row in rows} == {EXPECTED_CONTRACT_SHA256}

    for invalid in ("0" * 63, "g" * 64, "A" * 64):
        with pytest.raises(x0.X0ContractError):
            x0.x0_manifest(invalid)


def test_locked_runtime_candidate_is_closed_non_authorizing_and_replay_complete() -> None:
    assert CANDIDATE.is_dir()
    expected_members = {
        "attempt_replay.json",
        "candidate_receipt.json",
        "environment_lock.json",
        "replay_witness.json",
        "temporac.x0-manifest.v4.json",
    }
    assert {path.name for path in CANDIDATE.iterdir()} == expected_members

    receipt = _canonical_json_file(CANDIDATE / "candidate_receipt.json")
    assert frozenset(receipt) == RECEIPT_KEYS
    assert receipt["artifact_type"] == "temporac_x0_manifest_candidate_v4"
    assert receipt["authoritative"] is False
    assert receipt["authorizes"] == []
    assert receipt["candidate_status"] == "CANDIDATE_PENDING_FRESH_REVIEW"
    assert receipt["p2_status"] == "NOT_CLAIMED"
    assert receipt["s0_status"] == "NOT_CLAIMED"
    assert "P2_not_claimed" in receipt["blockers"]
    assert "S0_not_claimed" in receipt["blockers"]

    member_names = expected_members - {"candidate_receipt.json"}
    assert set(receipt["members"]) == member_names
    for name in member_names:
        assert receipt["members"][name] == _sha256_file(CANDIDATE / name)

    environment = _canonical_json_file(CANDIDATE / "environment_lock.json")
    assert environment["architecture"] == "x86_64"
    assert environment["byteorder"] == "little"
    assert environment["docker_image"] == "python:3.12.4"
    assert environment["numpy_version"] == "2.1.0"
    assert environment["platform_system"] == "Linux"
    assert environment["python_implementation"] == "CPython"
    assert environment["python_version"] == "3.12.4"
    assert environment["scipy_version"] == "1.14.1"

    bindings = receipt["bindings"]
    assert bindings["proposal_sha256"] == EXPECTED_CONTRACT_SHA256
    assert bindings["proposal_temporac_copy_sha256"] == EXPECTED_CONTRACT_SHA256
    assert bindings["implementation_source_sha256"] == _sha256_file(
        ROOT / "src/pams/temporac/x0.py"
    )
    assert bindings["contract_module_sha256"] == (
        "5ab8fb62dc558ba78e95fc07e52c1f1a1afbff5b1c5003834079eea6143d745f"
    )
    assert bindings["generator_source_sha256"] == _sha256_file(
        ROOT / "scripts/experiments/generate_temporac_x0_manifest_v4.py"
    )
    assert bindings["test_source_sha256"] == _sha256_file(ROOT / "tests/temporac/test_x0.py")
    assert bindings["runtime_lock_sha256"] == _sha256_file(CANDIDATE / "environment_lock.json")
    assert contract.PROPOSAL_SHA256 == EXPECTED_CONTRACT_SHA256

    attempts = _canonical_json_file(CANDIDATE / "attempt_replay.json")
    assert len(attempts) == 40 * 64
    assert all(frozenset(row) == ATTEMPT_KEYS for row in attempts)
    assert [row["id"] for row in attempts] == [
        f"U{source_id:04d}" for source_id in range(40) for _ in range(64)
    ]
    assert [row["attempt"] for row in attempts] == list(range(64)) * 40
    assert all(row["guard_pass"] is True for row in attempts)

    witness = _canonical_json_file(CANDIDATE / "replay_witness.json")
    assert witness["candidate_status"] == "CANDIDATE_PENDING_FRESH_REVIEW"
    assert witness["manifest_row_count"] == 40
    assert witness["retained_source_count"] == 40
    assert witness["retained_attempt"] == 0
    assert witness["replay"] == {
        "attempt_count": 2560,
        "attempt_replay_sha256": _sha256_file(CANDIDATE / "attempt_replay.json"),
        "guard_failures": 0,
        "guard_passes": 2560,
        "mgs_rejections": 0,
    }

    manifest_path = CANDIDATE / "temporac.x0-manifest.v4.json"
    manifest = _canonical_json_file(manifest_path)
    assert len(manifest) == 40
    assert all(frozenset(row) == MANIFEST_KEYS for row in manifest)
    for source_id, row in enumerate(manifest):
        attempt_zero = attempts[source_id * 64]
        assert row["id"] == attempt_zero["id"]
        assert row["accepted_attempt"] == attempt_zero["attempt"] == 0
        assert row["coefficient_seed_u64"] == attempt_zero["coefficient_seed_u64"]
        assert row["coefficient_bytes_sha256"] == attempt_zero["coefficient_bytes_sha256"]
        assert row["orbit_grid_sha256"] == attempt_zero["orbit_grid_sha256"]

    locked_runtime = (
        platform.system() == "Linux"
        and platform.machine().lower() == "x86_64"
        and platform.python_version() == "3.12.4"
        and np.__version__ == "2.1.0"
        and scipy.__version__ == "1.14.1"
        and sys.byteorder == "little"
    )
    if locked_runtime:
        assert manifest_path.read_bytes() == x0.x0_manifest(EXPECTED_CONTRACT_SHA256)
