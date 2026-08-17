"""Generate the closed Amendment 004 + 006 v3 TempoRAC operator candidate.

This entrypoint is intentionally limited to deterministic candidate generation.
It creates no P2/S0 receipt and grants no launch, data, training, Git, or paper
authority. The five candidate members are written only in a new absent
directory, with the no-self-hash receipt written last.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import sys
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

from pams.temporac import contract
from pams.temporac.fixtures import (
    OPERATOR_EDGE0_DIGEST_INVENTORY_SHA256,
    OPERATOR_INACTIVE_LOOKUP_SHA256,
    OPERATOR_KEY_INVENTORY_SHA256,
    OPERATOR_ROW_COUNT,
    OPERATOR_SEMANTIC_SHA256,
    OperatorFixtureKey,
    operator_edge_digest,
    operator_fixture_keys,
    operator_inactive_value,
    operator_row_key_bytes,
    validate_operator_fixture,
    validate_operator_inventory_witnesses,
)

CANDIDATE_STATUS = "FROZEN_CANDIDATE_PENDING_FRESH_IMPLEMENTATION_REVIEW"
OUTPUT_DIRECTORY_NAME = "operator_candidate_v2_20260816"
MANIFEST_NAME = "temporac.operator-manifest.v4.json"
SCHEMA_NAME = "operator_manifest_schema.json"
RUNTIME_NAME = "environment_lock.json"
WITNESS_NAME = "replay_witness.json"
RECEIPT_NAME = "candidate_receipt.json"

EXPECTED_CONTRACT_SHA256 = "3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391"
EXPECTED_AMENDMENT_004_MD_SHA256 = (
    "ece0025f4844dc04f838b5ec7f629a7fb3b10bd92e6f0bb3d5fe941cbcc83c82"
)
EXPECTED_AMENDMENT_004_JSON_SHA256 = (
    "61ce255e3f618041f3b54e19ab0c5d95f3817bafe68ac4b1e6db36d26e3083f4"
)
EXPECTED_AMENDMENT_006_MD_SHA256 = (
    "9b1e3a9c36c360e8d9011145d120f7cff8e6922f4620c60445d8194809070517"
)
EXPECTED_AMENDMENT_006_JSON_SHA256 = (
    "e443453b93c96dfda4c6daf68332c0f667429cbf2a036e16e0e6c43c89d9fa61"
)
EXPECTED_AMENDMENT_006_REVIEW_MD_SHA256 = (
    "ef84a0a3fde4a0269b34ba37fb2a4d41f7164a5382f8f3efaf4dc5948716388f"
)
EXPECTED_AMENDMENT_006_REVIEW_JSON_SHA256 = (
    "b59d381b5e515b7657861f636d2a50647faae7b18b2c73341550547109b57353"
)
EXPECTED_RUNTIME_SHA256 = "47c2227b74b627a03bf9b5e4930dd02484caaae5a695ae5c0129f04781115cd8"
EXPECTED_RECORD_BYTES = 100_089
EXPECTED_RECORD_SHA256 = "76ae0d49a0d236c7d045e8394d649e1252a6ef2219ef2dac33f72ac01ea76cca"
EXPECTED_DOCKER_IMAGE = (
    "docker.io/library/python@sha256:"
    "e8be0ea148390d08bc077840cf87ac6a538d80b0ea1e8752b3e3982987cd0a53"
)

DERIVED_MEMBER_COMMITMENTS: Mapping[str, tuple[int, str]] = {
    MANIFEST_NAME: (
        15_039_512,
        "1f39378d6f7a970b1f183558ed145bd97107f2aeda41902f4736e6778930e1bb",
    ),
    SCHEMA_NAME: (
        3_607,
        "3b29937c641274e73d10afd19bea943f28288b3d2851e8a5bf2313d7ea3e0fc3",
    ),
    RUNTIME_NAME: (3_976, EXPECTED_RUNTIME_SHA256),
    WITNESS_NAME: (
        2_896,
        "348254cfe64cea35d6e29861755489721ebbbd4031f3ec1c044d7fa24fe38010",
    ),
}

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

PROPOSAL_PATH = "refine-logs/temporac/FINAL_PROPOSAL.md"
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
        EXPECTED_AMENDMENT_004_JSON_SHA256,
    ),
    (
        "amendment_004_md_sha256",
        "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_004_OPERATOR_MANIFEST_SCHEMA.md",
        EXPECTED_AMENDMENT_004_MD_SHA256,
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
        EXPECTED_AMENDMENT_006_JSON_SHA256,
    ),
    (
        "amendment_006_md_sha256",
        "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH.md",
        EXPECTED_AMENDMENT_006_MD_SHA256,
    ),
    (
        "amendment_006_review_json_sha256",
        "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH_REVIEW_ACCEPTED.json",
        EXPECTED_AMENDMENT_006_REVIEW_JSON_SHA256,
    ),
    (
        "amendment_006_review_md_sha256",
        "refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_006_OPERATOR_RUNTIME_WHEEL_PATH_REVIEW_ACCEPTED.md",
        EXPECTED_AMENDMENT_006_REVIEW_MD_SHA256,
    ),
)

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
if len(SOURCE_SNAPSHOT_PATHS) != 28:  # pragma: no cover - import invariant
    raise RuntimeError("post-Amendment-006 generation snapshot must contain exactly 28 paths")


class CandidateGenerationError(RuntimeError):
    """Raised when a locked-runtime or candidate invariant fails."""


def _blocked(message: str) -> NoReturn:
    raise CandidateGenerationError(message)


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _uint64_be(value: int) -> bytes:
    if type(value) is not int or not 0 <= value < 2**64:
        _blocked("snapshot framing value is outside uint64")
    return value.to_bytes(8, byteorder="big", signed=False)


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _blocked(f"duplicate JSON key in bound input: {key}")
        result[key] = value
    return result


def _load_json(payload: bytes, *, name: str) -> Any:
    if payload.startswith(b"\xef\xbb\xbf"):
        _blocked(f"{name} contains a forbidden BOM")
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateGenerationError(f"{name} is not duplicate-free UTF-8 JSON") from exc


def _repository_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "scripts/experiments/generate_temporac_operator_manifest_v4.py").is_file():
        _blocked("repository snapshot does not contain the generator entrypoint")
    return root


def _forbidden_snapshot_path(relative: str) -> bool:
    parts = relative.split("/")
    name = parts[-1]
    return (
        any(part == "__pycache__" for part in parts)
        or name.endswith((".pyc", ".pth"))
        or name in {"sitecustomize.py", "usercustomize.py"}
    )


def _read_source_snapshot(root: Path) -> dict[str, bytes]:
    """Read the exact clean 28-file repository snapshot once."""

    observed: dict[str, Path] = {}
    normalized: dict[str, str] = {}
    inode_owner: dict[tuple[int, int], str] = {}
    for candidate in root.rglob("*"):
        metadata = candidate.lstat()
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            _blocked(f"source snapshot contains a symlink: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            _blocked(f"source snapshot contains a non-regular path: {relative}")
        if not relative.isascii() or _forbidden_snapshot_path(relative):
            _blocked(f"source snapshot contains a forbidden path: {relative}")
        normalized_key = unicodedata.normalize("NFC", relative).casefold()
        if normalized_key in normalized:
            _blocked(f"source snapshot has a normalized-path collision: {relative}")
        normalized[normalized_key] = relative
        inode = (int(metadata.st_dev), int(metadata.st_ino))
        if inode in inode_owner:
            _blocked(f"source snapshot has a hard-link alias: {relative}")
        inode_owner[inode] = relative
        observed[relative] = candidate

    expected = set(SOURCE_SNAPSHOT_PATHS)
    if set(observed) != expected:
        missing = sorted(expected - set(observed), key=str.encode)
        extra = sorted(set(observed) - expected, key=str.encode)
        _blocked(f"source snapshot path mismatch: missing={missing!r}, extra={extra!r}")

    payloads: dict[str, bytes] = {}
    for relative in SOURCE_SNAPSHOT_PATHS:
        path = observed[relative]
        before = path.stat()
        payload = path.read_bytes()
        after = path.stat()
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or len(payload) != after.st_size:
            _blocked(f"source snapshot file changed while read: {relative}")
        payloads[relative] = payload
    return payloads


def _source_snapshot_sha256(snapshot: Mapping[str, bytes]) -> str:
    digest = hashlib.sha256()
    digest.update(b"temporac.operator-generation-snapshot.v1\x00")
    for relative in SOURCE_SNAPSHOT_PATHS:
        path_bytes = relative.encode("ascii")
        payload = snapshot[relative]
        digest.update(_uint64_be(len(path_bytes)))
        digest.update(path_bytes)
        digest.update(_uint64_be(len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _project_records(
    snapshot: Mapping[str, bytes],
    specs: Sequence[tuple[str, tuple[str, ...]]],
) -> list[dict[str, Any]]:
    return [
        {
            "bytes": len(snapshot[path]),
            "path": path,
            "roles": list(roles),
            "sha256": _sha256_bytes(snapshot[path]),
        }
        for path, roles in specs
    ]


def _load_and_validate_amendments(
    snapshot: Mapping[str, bytes],
) -> tuple[dict[str, Any], dict[str, Any]]:
    for _binding_key, path, expected_sha256 in CONTRACT_BINDINGS:
        if _sha256_bytes(snapshot[path]) != expected_sha256:
            _blocked(f"bound contract/review digest mismatch: {path}")
    if _sha256_bytes(snapshot[PROPOSAL_PATH]) != EXPECTED_CONTRACT_SHA256:
        _blocked("canonical TempoRAC proposal digest mismatch")
    if contract.CONTRACT_SHA256 != EXPECTED_CONTRACT_SHA256:
        _blocked("contract module does not name the canonical proposal digest")

    amendment_path = CONTRACT_BINDINGS[4][1]
    amendment = _load_json(snapshot[amendment_path], name=amendment_path)
    if not isinstance(amendment, dict):
        _blocked("Amendment 004 JSON must be an object")
    if amendment.get("schema") != "temporac.contract-amendment.operator-manifest-schema.v2":
        _blocked("Amendment 004 schema mismatch")
    review_path = CONTRACT_BINDINGS[6][1]
    review = _load_json(snapshot[review_path], name=review_path)
    if not isinstance(review, dict) or review.get("verdict") != "ACCEPT":
        _blocked("bound Amendment 004 Round-2 review is not ACCEPT")
    if review.get("acceptance_status") != "provisional" or review.get("authoritative") is not False:
        _blocked("Amendment 004 review assurance boundary drift")
    amendment_006_path = CONTRACT_BINDINGS[8][1]
    amendment_006 = _load_json(snapshot[amendment_006_path], name=amendment_006_path)
    if not isinstance(amendment_006, dict):
        _blocked("Amendment 006 JSON must be an object")
    if amendment_006.get("schema") != "temporac.contract-amendment.operator-runtime-wheel-path.v1":
        _blocked("Amendment 006 schema mismatch")
    if amendment_006.get("status") != "PROPOSED_PENDING_FRESH_REVIEW":
        _blocked("Amendment 006 author status drift")
    amendment_006_review_path = CONTRACT_BINDINGS[10][1]
    amendment_006_review = _load_json(
        snapshot[amendment_006_review_path], name=amendment_006_review_path
    )
    if (
        not isinstance(amendment_006_review, dict)
        or amendment_006_review.get("verdict") != "ACCEPT"
    ):
        _blocked("bound Amendment 006 review is not ACCEPT")
    if (
        amendment_006_review.get("acceptance_status") != "provisional"
        or amendment_006_review.get("authoritative") is not False
    ):
        _blocked("Amendment 006 review assurance boundary drift")
    return amendment, amendment_006


def _distribution_record_bytes(distribution_name: str) -> bytes:
    record = importlib.metadata.distribution(distribution_name).read_text("RECORD")
    if record is None:
        _blocked(f"{distribution_name} RECORD is missing")
    return record.encode("utf-8")


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _runtime_lock_bytes(amendment_006: Mapping[str, Any]) -> bytes:
    runtime = amendment_006.get("corrected_runtime_lock")
    if not isinstance(runtime, dict):
        _blocked("Amendment 006 corrected_runtime_lock is missing")
    payload = _canonical_json_bytes(runtime)
    _verify_member_commitment(RUNTIME_NAME, payload)

    record_contract = amendment_006.get("installed_record_contract")
    if not isinstance(record_contract, dict):
        _blocked("Amendment 006 installed RECORD contract is missing")
    if record_contract.get("authoritative_preimage") != (
        'importlib.metadata.distribution("numpy").read_text("RECORD").encode("utf-8")'
    ):
        _blocked("installed RECORD preimage expression drift")
    authoritative_record = record_contract.get("authoritative_value")
    if authoritative_record != {
        "bytes": EXPECTED_RECORD_BYTES,
        "fresh_container_repetitions": 2,
        "sha256": EXPECTED_RECORD_SHA256,
    }:
        _blocked("installed RECORD authoritative commitment drift")

    numpy_lock = runtime["numpy"]
    python_lock = runtime["python"]
    environment = runtime["container_policy"]["environment"]
    record_bytes = _distribution_record_bytes("numpy")
    if (len(record_bytes), _sha256_bytes(record_bytes)) != (
        EXPECTED_RECORD_BYTES,
        EXPECTED_RECORD_SHA256,
    ):
        _blocked("installed NumPy RECORD normalized-text bytes mismatch")
    observed = {
        "architecture": platform.machine().lower(),
        "byteorder": sys.byteorder,
        "numpy_init_sha256": _sha256_file(Path(np.__file__).resolve()),
        "numpy_record_sha256": _sha256_bytes(record_bytes),
        "numpy_version": np.__version__,
        "platform_system": platform.system(),
        "python_executable_sha256": _sha256_file(Path(sys.executable).resolve()),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }
    expected = {
        "architecture": runtime["architecture"],
        "byteorder": runtime["byteorder"],
        "numpy_init_sha256": numpy_lock["init_sha256"],
        "numpy_record_sha256": numpy_lock["record_sha256"],
        "numpy_version": numpy_lock["version"],
        "platform_system": runtime["platform_system"],
        "python_executable_sha256": python_lock["executable_sha256"],
        "python_implementation": python_lock["implementation"],
        "python_version": python_lock["version"],
    }
    if observed != expected:
        _blocked(f"locked runtime mismatch: expected={expected!r}, observed={observed!r}")
    for name, value in environment.items():
        if os.environ.get(name) != value:
            _blocked(f"locked generation environment mismatch: {name}")
    if environment.get("TEMPORAC_OPERATOR_DOCKER_IMAGE") != EXPECTED_DOCKER_IMAGE:
        _blocked("locked Docker image environment binding mismatch")
    return payload


def _verify_member_commitment(name: str, payload: bytes) -> None:
    expected_bytes, expected_sha256 = DERIVED_MEMBER_COMMITMENTS[name]
    observed = (len(payload), _sha256_bytes(payload))
    if observed != (expected_bytes, expected_sha256):
        _blocked(
            f"{name} byte commitment mismatch: "
            f"expected={(expected_bytes, expected_sha256)!r}, observed={observed!r}"
        )


def _schema_bytes(amendment: Mapping[str, Any]) -> bytes:
    schema = amendment["manifest_schema_artifact"]["expected_object"]
    payload = _canonical_json_bytes(schema)
    _verify_member_commitment(SCHEMA_NAME, payload)
    return payload


def _validated_array_bytes(
    array: np.ndarray[Any, Any],
    *,
    name: str,
    dtype: str,
    shape: tuple[int, ...],
) -> bytes:
    if not isinstance(array, np.ndarray):
        _blocked(f"{name} is not a NumPy array")
    if array.dtype.str != dtype or array.shape != shape:
        _blocked(
            f"{name} dtype/shape mismatch: "
            f"expected={(dtype, shape)!r}, observed={(array.dtype.str, array.shape)!r}"
        )
    if not array.flags.c_contiguous or not np.isfinite(array).all():
        _blocked(f"{name} must be finite and C-contiguous before hashing")
    return array.tobytes(order="C")


def _manifest_and_witness_bytes(amendment: Mapping[str, Any]) -> tuple[bytes, bytes]:
    normative = validate_operator_inventory_witnesses()
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
    aggregate_roots = {name: hashlib.sha256() for name in aggregate_names}
    aggregate_bytes = {name: 0 for name in aggregate_names}
    rows: list[dict[str, Any]] = []
    actual_b255_edges = 0
    actual_b255_rows = 0
    first_actual_b255: dict[str, Any] | None = None

    for key in operator_fixture_keys():
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
            "truth_plateau_mask_sha256": (
                fixture.truth_plateau_mask,
                "|u1",
                (edge_count,),
            ),
            "truth_plateaus_sha256": (fixture.truth_plateaus, "<i4", (3, 2)),
            "windows_sha256": (windows, "<i4", (len(validation.windows), 2)),
        }
        payloads = {
            field: _validated_array_bytes(array, name=field, dtype=dtype, shape=shape)
            for field, (array, dtype, shape) in arrays.items()
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
        if tuple(sorted(row, key=str.encode)) != MANIFEST_ROW_KEYS:
            _blocked("operator manifest row schema drift")
        rows.append(row)

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
        for name, payload in aggregate_payloads.items():
            aggregate_roots[name].update(payload)
            aggregate_bytes[name] += len(payload)

        row_b255 = 0
        for run_start, run_stop in fixture.layout.run_bounds:
            for edge_index in range(run_start, run_stop):
                if (
                    fixture.truth_plateau_mask[edge_index] == 0
                    and operator_edge_digest(key, edge_index)[0] == 255
                ):
                    row_b255 += 1
                    if first_actual_b255 is None:
                        first_actual_b255 = {
                            "edge": edge_index,
                            "key_hex": operator_row_key_bytes(key).hex(),
                            "response_f4_le_hex": fixture.response[edge_index].tobytes().hex(),
                        }
        actual_b255_edges += row_b255
        actual_b255_rows += int(row_b255 > 0)

    manifest_bytes = _canonical_json_bytes(rows)
    _verify_member_commitment(MANIFEST_NAME, manifest_bytes)

    witness_keys = operator_fixture_keys()
    row_witnesses = [
        {
            "actual_f4_le_hex": validate_operator_fixture(key)
            .fixture.response[edge]
            .tobytes()
            .hex(),
            "digest_hex": operator_edge_digest(key, edge).hex(),
            "edge": edge,
            "inactive_function_f4_le_hex": operator_inactive_value(key, edge).tobytes().hex(),
            "key_hex": operator_row_key_bytes(key).hex(),
        }
        for key, edge in (
            (witness_keys[0], 0),
            (witness_keys[0], 32),
            (OperatorFixtureKey(5, 2, 0.75, 7, "right-boundary"), 17),
            (witness_keys[-1], 319),
            (witness_keys[1], 42),
        )
    ]
    witness = {
        "actual_inactive_b255": {
            "edge_count": actual_b255_edges,
            "first": first_actual_b255,
            "row_count": actual_b255_rows,
        },
        "aggregate_roots": {
            name: {"bytes": aggregate_bytes[name], "sha256": aggregate_roots[name].hexdigest()}
            for name in aggregate_names
        },
        "candidate_manifest_bytes": len(manifest_bytes),
        "candidate_manifest_sha256": _sha256_bytes(manifest_bytes),
        "candidate_status": CANDIDATE_STATUS,
        "normative_roots": {
            "edge0_digest_inventory": {
                "bytes": normative.edge0_digest_inventory_bytes,
                "sha256": OPERATOR_EDGE0_DIGEST_INVENTORY_SHA256,
            },
            "inactive_lookup": {
                "bytes": normative.inactive_lookup_bytes,
                "sha256": OPERATOR_INACTIVE_LOOKUP_SHA256,
            },
            "key_inventory": {
                "bytes": normative.key_inventory_bytes,
                "sha256": OPERATOR_KEY_INVENTORY_SHA256,
            },
            "semantic_inventory": {
                "bytes": normative.semantic_bytes,
                "sha256": OPERATOR_SEMANTIC_SHA256,
            },
        },
        "row_witnesses": row_witnesses,
        "schema": "temporac.operator-replay-witness.v2",
        "validated_row_count": len(rows),
    }
    witness_bytes = _canonical_json_bytes(witness)
    _verify_member_commitment(WITNESS_NAME, witness_bytes)

    expected = amendment["replay_witness"]
    if witness["actual_inactive_b255"] != expected["expected_actual_inactive_b255"]:
        _blocked("actual inactive b=255 witness drift")
    if witness["aggregate_roots"] != expected["expected_aggregate_roots"]:
        _blocked("aggregate root witness drift")
    if witness["row_witnesses"] != expected["expected_row_witnesses"]:
        _blocked("five row witnesses drift")
    return manifest_bytes, witness_bytes


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _validate_new_output(output: Path) -> None:
    if not output.is_absolute() or output.name != OUTPUT_DIRECTORY_NAME:
        _blocked(f"output must be an absolute path ending in {OUTPUT_DIRECTORY_NAME!r}")
    if _path_exists(output):
        _blocked(f"refusing to overwrite existing candidate: {output}")
    try:
        parent = output.parent.lstat()
    except FileNotFoundError as exc:
        raise CandidateGenerationError(
            f"candidate output parent is absent: {output.parent}"
        ) from exc
    if stat.S_ISLNK(parent.st_mode) or not stat.S_ISDIR(parent.st_mode):
        _blocked("candidate output parent must be a regular non-symlink directory")


def _write_exclusive(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _receipt_bytes(
    *,
    snapshot: Mapping[str, bytes],
    source_snapshot_sha256: str,
    member_payloads: Mapping[str, bytes],
) -> bytes:
    bindings: dict[str, Any] = {
        binding_key: expected_sha256 for binding_key, _path, expected_sha256 in CONTRACT_BINDINGS
    }
    bindings.update(
        {
            "generation_project_files": _project_records(snapshot, GENERATION_PROJECT_SPECS),
            "proposal_sha256": EXPECTED_CONTRACT_SHA256,
            "review_project_files": _project_records(snapshot, REVIEW_PROJECT_SPECS),
            "runtime_lock_sha256": EXPECTED_RUNTIME_SHA256,
            "source_snapshot_sha256": source_snapshot_sha256,
        }
    )
    receipt = {
        "authority": {
            "P2": False,
            "S0": False,
            "data": False,
            "gate": False,
            "git": False,
            "launch": False,
            "paper_claim": False,
            "server": False,
            "training": False,
        },
        "bindings": bindings,
        "blockers": [
            "fresh_implementation_review_required",
            "P2_not_claimed",
            "S0_not_claimed",
            "launch_authorization_zero",
        ],
        "candidate_status": CANDIDATE_STATUS,
        "members": {
            name: _sha256_bytes(member_payloads[name])
            for name in (RUNTIME_NAME, SCHEMA_NAME, WITNESS_NAME, MANIFEST_NAME)
        },
        "p2_status": "NOT_CLAIMED",
        "s0_status": "NOT_CLAIMED",
        "schema": "temporac.operator-manifest-candidate-receipt.v3",
        "validated_row_count": OPERATOR_ROW_COUNT,
    }
    expected_keys = {
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
    if set(receipt) != expected_keys or len(bindings) != 17:
        _blocked("candidate receipt closed schema drift")
    return _canonical_json_bytes(receipt)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = args.output
    # The retry/overwrite guard is deliberately first: an existing candidate
    # must be left byte- and mtime-identical without any regeneration work.
    _validate_new_output(output)

    root = _repository_root()
    snapshot = _read_source_snapshot(root)
    amendment_004, amendment_006 = _load_and_validate_amendments(snapshot)
    source_snapshot_sha256 = _source_snapshot_sha256(snapshot)
    runtime_bytes = _runtime_lock_bytes(amendment_006)
    schema_bytes = _schema_bytes(amendment_004)
    manifest_bytes, witness_bytes = _manifest_and_witness_bytes(amendment_004)
    member_payloads = {
        MANIFEST_NAME: manifest_bytes,
        SCHEMA_NAME: schema_bytes,
        RUNTIME_NAME: runtime_bytes,
        WITNESS_NAME: witness_bytes,
    }
    receipt_bytes = _receipt_bytes(
        snapshot=snapshot,
        source_snapshot_sha256=source_snapshot_sha256,
        member_payloads=member_payloads,
    )

    output.mkdir(mode=0o755)
    _write_exclusive(output / MANIFEST_NAME, manifest_bytes)
    _write_exclusive(output / SCHEMA_NAME, schema_bytes)
    _write_exclusive(output / RUNTIME_NAME, runtime_bytes)
    _write_exclusive(output / WITNESS_NAME, witness_bytes)
    # Receipt is intentionally last: without it the tree is incomplete.
    _write_exclusive(output / RECEIPT_NAME, receipt_bytes)
    sys.stdout.buffer.write(
        _canonical_json_bytes(
            {
                "candidate_receipt_sha256": _sha256_bytes(receipt_bytes),
                "candidate_status": CANDIDATE_STATUS,
                "output": output.as_posix(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CandidateGenerationError as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        raise SystemExit(2) from error
