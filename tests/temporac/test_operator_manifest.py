"""Closed review checks for the Amendment 004 + 006 v3 operator candidate."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pams.temporac import contract
from pams.temporac.decode import decode_identity, quantize_response
from pams.temporac.fixtures import (
    OPERATOR_EDGE0_DIGEST_INVENTORY_SHA256,
    OPERATOR_INACTIVE_LOOKUP_SHA256,
    OPERATOR_KEY_INVENTORY_SHA256,
    OPERATOR_ROW_COUNT,
    OPERATOR_SEMANTIC_SHA256,
    OperatorFixtureKey,
    operator_edge_digest,
    operator_fixture_keys,
    operator_inactive_lookup_bytes,
    operator_inactive_value,
    operator_inactive_value_from_byte,
    operator_row_key_bytes,
    operator_row_key_preimage,
    validate_operator_fixture,
    validate_operator_inventory_witnesses,
)
from pams.temporac.nola import nola_reconstruct
from pams.temporac.receipts import ReceiptError, load_frozen_operator_candidate
from pams.temporac.types import ContractError

ROOT = Path(__file__).resolve().parents[2]
CANDIDATE = ROOT / "data/temporac_p00_v4/operator_candidate_v2_20260816"
HISTORICAL_CANDIDATE = ROOT / "data/temporac_p00_v4/operator_candidate_20260816"
AMENDMENT_004_PATH = (
    ROOT / "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA.json"
)
AMENDMENT_006_PATH = (
    ROOT / "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH.json"
)
MANIFEST_NAME = "temporac.operator-manifest.v4.json"
EXPECTED_PROPOSAL_SHA256 = "3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391"
EXPECTED_EFFECTIVE_CONTRACT_SHA256 = (
    "c203f4aab00879acf041b90829f0124d0b1de231294f191b03f3166f728e3d5c"
)
REVIEW_MD = ROOT / "refine-logs/temporac/TEMPORAC_OPERATOR_IMPLEMENTATION_REVIEW_V2_20260816.md"
REVIEW_JSON = ROOT / "refine-logs/temporac/TEMPORAC_OPERATOR_IMPLEMENTATION_REVIEW_V2_20260816.json"
EXPECTED_MEMBERS: Mapping[str, tuple[int, str]] = {
    MANIFEST_NAME: (
        15_039_512,
        "1f39378d6f7a970b1f183558ed145bd97107f2aeda41902f4736e6778930e1bb",
    ),
    "operator_manifest_schema.json": (
        3_607,
        "3b29937c641274e73d10afd19bea943f28288b3d2851e8a5bf2313d7ea3e0fc3",
    ),
    "environment_lock.json": (
        3_976,
        "47c2227b74b627a03bf9b5e4930dd02484caaae5a695ae5c0129f04781115cd8",
    ),
    "replay_witness.json": (
        2_896,
        "348254cfe64cea35d6e29861755489721ebbbd4031f3ec1c044d7fa24fe38010",
    ),
}
HISTORICAL_MEMBERS: Mapping[str, tuple[int, str]] = {
    "candidate_receipt.json": (
        1_729,
        "2e457e01166c6151e4edc688e8fd6f981d1278231fcd8c2bc000179f7a825102",
    ),
    "candidate_schema.json": (
        1_301,
        "9f4b2ced821e667f950c1f56bbf9adbdc14b2c387c1932953048caf3f747a8cd",
    ),
    "environment_lock.json": (
        458,
        "709db0943752965524b507dd6340554a8d7d218be125317af0fc8b7748a5a947",
    ),
    "replay_witness.json": (
        3_908,
        "36f60f41ac111a7b05e5f4846464cc774e09bb9faa44feed10be5a4f35418653",
    ),
    "temporac.operator-manifest.candidate-v1.json": (
        15_039_512,
        "1f39378d6f7a970b1f183558ed145bd97107f2aeda41902f4736e6778930e1bb",
    ),
}

GENERATION_PROJECT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scripts/experiments/generate_temporac_operator_manifest_v4.py", ("generation-entrypoint",)),
    ("src/pams/__init__.py", ("generation-bootstrap",)),
    ("src/pams/types.py", ("generation-bootstrap",)),
    ("src/pams/temporac/__init__.py", ("generation-bootstrap",)),
    ("src/pams/temporac/contract.py", ("generation-module",)),
    ("src/pams/temporac/decode.py", ("generation-module",)),
    ("src/pams/temporac/fixtures.py", ("generation-module",)),
    ("src/pams/temporac/hashio.py", ("generation-module",)),
    ("src/pams/temporac/types.py", ("generation-module",)),
)
REVIEW_PROJECT_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scripts/experiments/generate_temporac_operator_manifest_v4.py", ("review-read-input",)),
    ("src/pams/__init__.py", ("review-module",)),
    ("src/pams/types.py", ("review-module",)),
    ("src/pams/temporac/__init__.py", ("review-module",)),
    ("src/pams/temporac/contract.py", ("review-module",)),
    ("src/pams/temporac/decode.py", ("review-module",)),
    ("src/pams/temporac/fixtures.py", ("review-module",)),
    ("src/pams/temporac/hashio.py", ("review-module",)),
    ("src/pams/temporac/types.py", ("review-module",)),
    ("src/pams/temporac/gates.py", ("review-module",)),
    ("src/pams/temporac/nola.py", ("review-module",)),
    ("tests/__init__.py", ("review-module",)),
    ("tests/temporac/__init__.py", ("review-module",)),
    ("tests/temporac/test_gates_fixtures.py", ("review-entrypoint",)),
    ("tests/temporac/test_operator_manifest.py", ("review-entrypoint",)),
)
CONTRACT_BINDINGS: tuple[tuple[str, str, str], ...] = (
    (
        "amendment_002_json_sha256",
        "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY.json",
        "d524556f292ec17a155cdcd962cf8a88bea26f32b810ffb7bd8e45744f1c54fc",
    ),
    (
        "amendment_002_md_sha256",
        "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY.md",
        "899d6f071c89f898bf3825adfb30a326904f9bc7fe44d44351da8a5d0d9ccb14",
    ),
    (
        "amendment_002_review_json_sha256",
        "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY_REVIEW_ROUND2.json",
        "3730d96eff1d3815499602eaf183774af032c514a588d4e92fe8a1cccf053e64",
    ),
    (
        "amendment_002_review_md_sha256",
        "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_002_OPERATOR_ROW_KEY_REVIEW_ROUND2.md",
        "a86da733f3ae048b867ec8b2366a9f0262ca2b390ef08687ca7d710654e8301f",
    ),
    (
        "amendment_004_json_sha256",
        "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA.json",
        "61ce255e3f618041f3b54e19ab0c5d95f3817bafe68ac4b1e6db36d26e3083f4",
    ),
    (
        "amendment_004_md_sha256",
        "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA.md",
        "ece0025f4844dc04f838b5ec7f629a7fb3b10bd92e6f0bb3d5fe941cbcc83c82",
    ),
    (
        "amendment_004_review_json_sha256",
        "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA_REVIEW_ROUND2.json",
        "f62135f6ac502b4ca9ccfda8534201c12a60a2853ae4a298bf088c7a24bbd50f",
    ),
    (
        "amendment_004_review_md_sha256",
        "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA_REVIEW_ROUND2.md",
        "6c45d2bc333c8b89915c3e3b881b893bf26203085bab010a28ad55158f9468a7",
    ),
    (
        "amendment_006_json_sha256",
        "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH.json",
        "e443453b93c96dfda4c6daf68332c0f667429cbf2a036e16e0e6c43c89d9fa61",
    ),
    (
        "amendment_006_md_sha256",
        "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH.md",
        "9b1e3a9c36c360e8d9011145d120f7cff8e6922f4620c60445d8194809070517",
    ),
    (
        "amendment_006_review_json_sha256",
        "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH_REVIEW_ACCEPTED.json",
        "b59d381b5e515b7657861f636d2a50647faae7b18b2c73341550547109b57353",
    ),
    (
        "amendment_006_review_md_sha256",
        "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH_REVIEW_ACCEPTED.md",
        "ef84a0a3fde4a0269b34ba37fb2a4d41f7164a5382f8f3efaf4dc5948716388f",
    ),
)
PROPOSAL_PATH = "refine-logs/temporac/FINAL_PROPOSAL.md"
SOURCE_SNAPSHOT_PATHS = tuple(
    sorted(
        {
            *(path for path, _roles in REVIEW_PROJECT_SPECS),
            PROPOSAL_PATH,
            *(path for _key, path, _sha256 in CONTRACT_BINDINGS),
        },
        key=str.encode,
    )
)
MANIFEST_ROW_KEYS = (
    "amplitude",
    "association_sha256",
    "component_location_sha256",
    "component_score_sha256",
    "decoded_components_sha256",
    "decoder_mask_sha256",
    "edge_count",
    "edge_mask_sha256",
    "expected_components_sha256",
    "gap",
    "generated_response_sha256",
    "key_hex",
    "layout",
    "nola_quantized_response_sha256",
    "offset",
    "reset_edge",
    "run_bounds_sha256",
    "target_mask_sha256",
    "truth_negative_components",
    "truth_plateau_mask_sha256",
    "truth_plateaus_sha256",
    "width",
    "windows_sha256",
)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")


def _json_bytes(payload: bytes) -> Any:
    assert not payload.startswith(b"\xef\xbb\xbf")
    return json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)


def _canonical_json_file(path: Path) -> Any:
    payload = path.read_bytes()
    parsed = _json_bytes(payload)
    assert payload == _canonical_json_bytes(parsed)
    return parsed


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _f4_hex(value: np.float32) -> str:
    return np.asarray([value], dtype=np.dtype("<f4")).tobytes(order="C").hex()


def _source_snapshot_sha256() -> str:
    digest = hashlib.sha256(b"temporac.operator-generation-snapshot.v1\x00")
    for relative in SOURCE_SNAPSHOT_PATHS:
        path_bytes = relative.encode("ascii")
        payload = (ROOT / relative).read_bytes()
        digest.update(len(path_bytes).to_bytes(8, "big"))
        digest.update(path_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _project_records(specs: Sequence[tuple[str, tuple[str, ...]]]) -> list[dict[str, Any]]:
    return [
        {
            "bytes": (ROOT / path).stat().st_size,
            "path": path,
            "roles": list(roles),
            "sha256": _sha256_file(ROOT / path),
        }
        for path, roles in specs
    ]


def _array_bytes(array: np.ndarray[Any, Any], dtype: str, shape: tuple[int, ...]) -> bytes:
    assert isinstance(array, np.ndarray)
    assert (array.dtype.str, array.shape) == (dtype, shape)
    assert array.flags.c_contiguous and np.isfinite(array).all()
    return array.tobytes(order="C")


def _replay_row(key: OperatorFixtureKey) -> tuple[dict[str, Any], dict[str, bytes]]:
    validation = validate_operator_fixture(key)
    fixture = validation.fixture
    edge_count = fixture.layout.edge_count
    run_count = 2 if key.layout == "run-reset" else 1
    windows = np.asarray(validation.windows, dtype=np.dtype("<i4")).reshape((-1, 2))
    arrays: dict[str, tuple[np.ndarray[Any, Any], str, tuple[int, ...]]] = {
        "association_sha256": (validation.association, "|u1", (3, 3)),
        "component_location_sha256": (validation.component_location, "<i4", (3,)),
        "component_score_sha256": (validation.component_score, "<f4", (3,)),
        "decoded_components_sha256": (validation.component_bounds, "<i4", (3, 2)),
        "decoder_mask_sha256": (fixture.decoder_mask, "|u1", (edge_count,)),
        "edge_mask_sha256": (fixture.edge_mask, "|u1", (edge_count,)),
        "expected_components_sha256": (fixture.expected_components, "<i4", (3, 2)),
        "generated_response_sha256": (fixture.response, "<f4", (edge_count,)),
        "nola_quantized_response_sha256": (
            validation.nola_quantized_response,
            "<f4",
            (edge_count,),
        ),
        "run_bounds_sha256": (fixture.run_bounds, "<i4", (run_count, 2)),
        "target_mask_sha256": (fixture.target_mask, "|u1", (edge_count,)),
        "truth_plateau_mask_sha256": (fixture.truth_plateau_mask, "|u1", (edge_count,)),
        "truth_plateaus_sha256": (fixture.truth_plateaus, "<i4", (3, 2)),
        "windows_sha256": (windows, "<i4", (len(validation.windows), 2)),
    }
    payloads = {
        field: _array_bytes(array, dtype, shape) for field, (array, dtype, shape) in arrays.items()
    }
    row = {
        "amplitude": key.amplitude,
        **{field: _sha256_bytes(payload) for field, payload in payloads.items()},
        "edge_count": edge_count,
        "gap": key.gap,
        "key_hex": operator_row_key_bytes(key).hex(),
        "layout": key.layout,
        "offset": key.offset,
        "reset_edge": fixture.layout.reset_edge,
        "truth_negative_components": fixture.layout.truth_negative_components,
        "width": key.width,
    }
    aggregate_payloads = {
        "association": payloads["association_sha256"],
        "decoded_components": payloads["decoded_components_sha256"],
        "decoder_masks": payloads["decoder_mask_sha256"],
        "edge_masks": payloads["edge_mask_sha256"],
        "nola_responses": payloads["nola_quantized_response_sha256"],
        "responses": payloads["generated_response_sha256"],
        "target_masks": payloads["target_mask_sha256"],
        "truth_masks": payloads["truth_plateau_mask_sha256"],
        "truth_plateaus": payloads["truth_plateaus_sha256"],
    }
    return row, aggregate_payloads


def test_normative_inventory_and_all_five_amendment_witnesses() -> None:
    witnesses = validate_operator_inventory_witnesses()
    assert witnesses.row_count == witnesses.unique_key_count == OPERATOR_ROW_COUNT == 10_368
    assert (witnesses.semantic_bytes, witnesses.semantic_sha256) == (
        746_714,
        OPERATOR_SEMANTIC_SHA256,
    )
    assert (witnesses.inactive_lookup_bytes, witnesses.inactive_lookup_sha256) == (
        1_024,
        OPERATOR_INACTIVE_LOOKUP_SHA256,
    )
    assert (witnesses.key_inventory_bytes, witnesses.key_inventory_sha256) == (
        186_624,
        OPERATOR_KEY_INVENTORY_SHA256,
    )
    assert (witnesses.edge0_digest_inventory_bytes, witnesses.edge0_digest_inventory_sha256) == (
        331_776,
        OPERATOR_EDGE0_DIGEST_INVENTORY_SHA256,
    )
    rows = (
        (
            OperatorFixtureKey(4, 1, 0.60, 0, "interior"),
            0,
            "54ac35705b42e8296cf324b893e5450c123b8b48c45e912cba928a6c432f60c4",
            "65984b3e",
            "65984b3e",
        ),
        (
            OperatorFixtureKey(4, 1, 0.60, 0, "interior"),
            32,
            "dee0ad937ac053341db0c0b1dc6a179c6684e9a6108b76336070f4cb910e1170",
            "1fecb83e",
            "6666263f",
        ),
        (
            OperatorFixtureKey(5, 2, 0.75, 7, "right-boundary"),
            17,
            "9406bcc666d0a8487362b65700a5daebd6d3f4496c54c5b2da1f849ba52d3ee1",
            "26598c3e",
            "26598c3e",
        ),
        (
            OperatorFixtureKey(129, 4, 1.00, 31, "run-reset"),
            319,
            "fb06baef760aaeb7ab2a4aeaad5d13dc457afc989fb75db931e1535c4213c215",
            "fe63ca3e",
            "fe63ca3e",
        ),
        (
            OperatorFixtureKey(4, 1, 0.60, 0, "left-boundary"),
            42,
            "ff78168a29882190e18b1fd5957cf1eb549c721c227344c8fbfc20ec8d9f4fb3",
            "cccccc3e",
            "cccccc3e",
        ),
    )
    for key, edge, digest_hex, inactive_hex, actual_hex in rows:
        validation = validate_operator_fixture(key)
        assert operator_edge_digest(key, edge).hex() == digest_hex
        assert _f4_hex(operator_inactive_value(key, edge)) == inactive_hex
        assert validation.fixture.response[edge].tobytes().hex() == actual_hex
    assert operator_row_key_bytes(rows[0][0]).hex() == "00040001000000000000"
    assert (
        operator_row_key_preimage(rows[0][0], 0).hex()
        == "74656d706f7261632e6f70657261746f722e763400000000000000000a0004000100000000000000000000"
    )
    with pytest.raises(ContractError, match="uint32"):
        operator_row_key_preimage(rows[0][0], 2**32)


def test_inactive_operation_order_cap_and_flat_plateaus_are_exact() -> None:
    lookup = operator_inactive_lookup_bytes()
    assert len(lookup) == 1_024
    assert hashlib.sha256(lookup).hexdigest() == OPERATOR_INACTIVE_LOOKUP_SHA256
    assert _f4_hex(operator_inactive_value_from_byte(254)) != "cccccc3e"
    assert _f4_hex(operator_inactive_value_from_byte(255)) == "cccccc3e"
    assert float(np.float64(operator_inactive_value_from_byte(255))) == 0.3999999761581421
    assert 0.5 - float(np.float64(operator_inactive_value_from_byte(255))) == 0.10000002384185791
    for width in (2, 4):
        validation = validate_operator_fixture(OperatorFixtureKey(8, width, 1.00, 11, "run-reset"))
        for index, (start, stop) in enumerate(validation.fixture.layout.truth_plateaus):
            assert (
                validation.fixture.response[start:stop].tobytes()
                == bytes.fromhex("0000803f") * width
            )
            assert validation.component_location[index] == start + (width - 1) // 2


@pytest.mark.parametrize("layout", ("interior", "left-boundary", "right-boundary", "run-reset"))
def test_fixture_desired_windows_reconstruct_through_production_nola_and_decoder(
    layout: str,
) -> None:
    validation = validate_operator_fixture(OperatorFixtureKey(128, 4, 0.75, 31, layout))
    fixture = validation.fixture
    probabilities = np.repeat(fixture.response.astype(np.float64)[:, None], 3, axis=1)
    gates = np.zeros((len(validation.windows), 3), dtype=np.float64)
    gates[:, 0] = 1.0
    reconstructed = nola_reconstruct(probabilities, gates, validation.windows, fixture.edge_mask)
    assert quantize_response(reconstructed.response).tobytes() == fixture.response.tobytes()
    decoded = decode_identity(reconstructed.response, fixture.decoder_mask, fixture.run_bounds)
    assert decoded.decoder_invocations == 1
    assert np.array_equal(decoded.component_bounds, fixture.expected_components)
    assert np.array_equal(decoded.component_location, validation.component_location)
    assert np.array_equal(decoded.component_score, validation.component_score)


def test_candidate_is_exact_closed_non_authoritative_and_historical_is_regression_only() -> None:
    expected_names = {*EXPECTED_MEMBERS, "candidate_receipt.json"}
    assert {path.name for path in CANDIDATE.iterdir()} == expected_names
    for name, expected in EXPECTED_MEMBERS.items():
        payload = (CANDIDATE / name).read_bytes()
        assert (len(payload), _sha256_bytes(payload)) == expected
        assert payload == _canonical_json_bytes(_json_bytes(payload))

    amendment_004 = _json_bytes(AMENDMENT_004_PATH.read_bytes())
    amendment_006 = _json_bytes(AMENDMENT_006_PATH.read_bytes())
    schema = _canonical_json_file(CANDIDATE / "operator_manifest_schema.json")
    runtime = _canonical_json_file(CANDIDATE / "environment_lock.json")
    witness = _canonical_json_file(CANDIDATE / "replay_witness.json")
    manifest = _canonical_json_file(CANDIDATE / MANIFEST_NAME)
    assert schema == amendment_004["manifest_schema_artifact"]["expected_object"]
    assert runtime == amendment_006["corrected_runtime_lock"]
    assert len(manifest) == OPERATOR_ROW_COUNT
    assert all(tuple(sorted(row, key=str.encode)) == MANIFEST_ROW_KEYS for row in manifest)
    assert [row["key_hex"] for row in manifest] == sorted(
        (row["key_hex"] for row in manifest), key=str.encode
    )
    assert len({row["key_hex"] for row in manifest}) == OPERATOR_ROW_COUNT
    assert witness == {
        "actual_inactive_b255": amendment_004["replay_witness"]["expected_actual_inactive_b255"],
        "aggregate_roots": amendment_004["replay_witness"]["expected_aggregate_roots"],
        "candidate_manifest_bytes": EXPECTED_MEMBERS[MANIFEST_NAME][0],
        "candidate_manifest_sha256": EXPECTED_MEMBERS[MANIFEST_NAME][1],
        "candidate_status": "FROZEN_CANDIDATE_PENDING_FRESH_IMPLEMENTATION_REVIEW",
        "normative_roots": amendment_004["replay_witness"]["expected_normative_roots"],
        "row_witnesses": amendment_004["replay_witness"]["expected_row_witnesses"],
        "schema": "temporac.operator-replay-witness.v2",
        "validated_row_count": OPERATOR_ROW_COUNT,
    }

    receipt = _canonical_json_file(CANDIDATE / "candidate_receipt.json")
    assert set(receipt) == {
        "authority",
        "bindings",
        "blockers",
        "candidate_status",
        "members",
        "p2_status",
        "s0_status",
        "schema",
        "validated_row_count",
    }
    assert receipt["schema"] == "temporac.operator-manifest-candidate-receipt.v3"
    assert receipt["candidate_status"] == "FROZEN_CANDIDATE_PENDING_FRESH_IMPLEMENTATION_REVIEW"
    assert receipt["p2_status"] == receipt["s0_status"] == "NOT_CLAIMED"
    assert receipt["validated_row_count"] == OPERATOR_ROW_COUNT
    assert receipt["authority"] == {
        "P2": False,
        "S0": False,
        "data": False,
        "gate": False,
        "git": False,
        "launch": False,
        "paper_claim": False,
        "server": False,
        "training": False,
    }
    assert receipt["blockers"] == [
        "fresh_implementation_review_required",
        "P2_not_claimed",
        "S0_not_claimed",
        "launch_authorization_zero",
    ]
    assert receipt["members"] == {
        name: digest for name, (_size, digest) in EXPECTED_MEMBERS.items()
    }

    bindings = receipt["bindings"]
    assert set(bindings) == {
        *(key for key, _path, _sha256 in CONTRACT_BINDINGS),
        "generation_project_files",
        "proposal_sha256",
        "review_project_files",
        "runtime_lock_sha256",
        "source_snapshot_sha256",
    }
    assert len(bindings) == 17
    for key, _path, expected_sha256 in CONTRACT_BINDINGS:
        assert bindings[key] == expected_sha256
    assert bindings["proposal_sha256"] == EXPECTED_PROPOSAL_SHA256
    assert contract.PROPOSAL_SHA256 == EXPECTED_PROPOSAL_SHA256
    assert contract.CONTRACT_SHA256 == EXPECTED_EFFECTIVE_CONTRACT_SHA256
    assert bindings["runtime_lock_sha256"] == EXPECTED_MEMBERS["environment_lock.json"][1]
    assert bindings["source_snapshot_sha256"] == (
        "ded975e31ced2cf1c7d69fb217610125f65d419f641a8e0832d115ec7a4870c0"
    )

    handle = load_frozen_operator_candidate(CANDIDATE, REVIEW_MD, REVIEW_JSON)
    assert handle.candidate_receipt_sha256 == (
        "d53fcc1138754eee90a255952606cd054aa04c72cb3f97d66bf4464d5de672cb"
    )
    assert handle.generation_snapshot_sha256 == bindings["source_snapshot_sha256"]
    assert handle.validated_row_count == OPERATOR_ROW_COUNT
    assert handle.candidate_only and handle.eligible_for_separate_p2
    assert not any(value for _name, value in handle.authority)

    assert {path.name for path in HISTORICAL_CANDIDATE.iterdir()} == set(HISTORICAL_MEMBERS)
    for name, expected in HISTORICAL_MEMBERS.items():
        payload = (HISTORICAL_CANDIDATE / name).read_bytes()
        assert (len(payload), _sha256_bytes(payload)) == expected
    old_receipt = _canonical_json_file(HISTORICAL_CANDIDATE / "candidate_receipt.json")
    assert old_receipt["candidate_status"] == "CANDIDATE_PENDING_FRESH_REVIEW"
    assert old_receipt["authoritative"] is False
    assert old_receipt["p2_status"] == old_receipt["s0_status"] == "NOT_CLAIMED"
    assert (HISTORICAL_CANDIDATE / "temporac.operator-manifest.candidate-v1.json").read_bytes() == (
        CANDIDATE / MANIFEST_NAME
    ).read_bytes()


def test_frozen_operator_consumer_rejects_detached_review_and_directory_attacks(
    tmp_path: Path,
) -> None:
    with pytest.raises(ReceiptError, match="exact regular frozen directory"):
        load_frozen_operator_candidate(HISTORICAL_CANDIDATE, REVIEW_MD, REVIEW_JSON)

    attacked_review = tmp_path / REVIEW_JSON.name
    review_bytes = REVIEW_JSON.read_bytes()
    attacked_review.write_bytes(review_bytes[:-2] + b" \n")
    with pytest.raises(ReceiptError, match="trusted V2 review"):
        load_frozen_operator_candidate(CANDIDATE, REVIEW_MD, attacked_review)

    copied_parent = tmp_path / "copy"
    copied_parent.mkdir()
    copied_candidate = copied_parent / CANDIDATE.name
    shutil.copytree(CANDIDATE, copied_candidate)
    (copied_candidate / "extra.json").write_bytes(b"{}\n")
    with pytest.raises(ReceiptError, match="exactly five members"):
        load_frozen_operator_candidate(copied_candidate, REVIEW_MD, REVIEW_JSON)


def test_all_candidate_rows_replay_exactly_to_closed_aggregate_witnesses() -> None:
    manifest = _canonical_json_file(CANDIDATE / MANIFEST_NAME)
    witness = _canonical_json_file(CANDIDATE / "replay_witness.json")
    aggregate_names = (
        "association",
        "decoded_components",
        "decoder_masks",
        "edge_masks",
        "nola_responses",
        "responses",
        "target_masks",
        "truth_masks",
        "truth_plateaus",
    )
    roots = {name: hashlib.sha256() for name in aggregate_names}
    byte_counts = {name: 0 for name in aggregate_names}
    for key, recorded in zip(operator_fixture_keys(), manifest, strict=True):
        replayed, payloads = _replay_row(key)
        assert replayed == recorded
        for name, payload in payloads.items():
            roots[name].update(payload)
            byte_counts[name] += len(payload)
    assert witness["aggregate_roots"] == {
        name: {"bytes": byte_counts[name], "sha256": roots[name].hexdigest()}
        for name in aggregate_names
    }
