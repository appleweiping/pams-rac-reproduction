"""Synthetic fixture inventories and hash-only verification primitives.

Natural inputs are intentionally absent.  Operator response construction is a
pure implementation of canonical proposal Section 9; manifest serialization
is deliberately left to explicitly non-authoritative candidate tooling because
the proposal does not freeze a manifest row schema or aggregate preimage.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from typing import cast

import numpy as np
from numpy.typing import NDArray

from pams.temporac.contract import CONTRACT_SHA256
from pams.temporac.hashio import canonical_json_bytes, sha256_bytes
from pams.temporac.types import ContractError

X0_ROW_COUNT = 40
TOPOLOGY_ROW_COUNT = 13
OPERATOR_ROW_COUNT = 10_368
COUNT_METRIC_FIXTURE_SCHEMA = "temporac.count-metric-fixture-receipt.v1"
OPERATOR_GAPS = (4, 5, 8, 16, 32, 64, 127, 128, 129)
OPERATOR_WIDTHS = (1, 2, 4)
OPERATOR_AMPLITUDES = (0.60, 0.75, 1.00)
OPERATOR_OFFSETS = tuple(range(32))
OPERATOR_LAYOUTS = ("interior", "left-boundary", "right-boundary", "run-reset")
OPERATOR_TAG = b"temporac.operator.v4"
OPERATOR_SEMANTIC_SHA256 = "0fe8b2619a3f33cc4a83485053888c04c567b015eea65f81bf6cd9ec715ee4f2"
OPERATOR_KEY_INVENTORY_SHA256 = "18dfdafe237b77565de36bab6d4642efb314ec1cbbd76e852938ed5c4d424a55"
OPERATOR_EDGE0_DIGEST_INVENTORY_SHA256 = (
    "ba797f6ae59cbbb40f19a61199afaaa1ac5b2c001a3c9fc34f1dba8f79cf78e4"
)
OPERATOR_INACTIVE_LOOKUP_SHA256 = "afa6eff0d744acfdf9cf78adc43aa8daf01633f3345d7567d642a7782885040d"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class FixtureStatus(str, Enum):
    VALIDATED = "VALIDATED"
    BLOCKED_PENDING_F8_AMPLITUDE_AMENDMENT = "BLOCKED_PENDING_F8_AMPLITUDE_AMENDMENT"
    BLOCKED_UNDEFINED_ROW_KEY_PREIMAGE = "BLOCKED_UNDEFINED_ROW_KEY_PREIMAGE"


class UndefinedRowKeyPreimageError(ContractError):
    """Legacy error retained for callers of the pre-canonicalized proposal."""


def _digest(value: str, *, name: str) -> str:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def verify_sha256(payload: bytes, expected_sha256: str, *, name: str = "payload") -> None:
    """Fail closed unless exact bytes match one lowercase committed digest."""

    expected = _digest(expected_sha256, name=f"{name} sha256")
    if sha256_bytes(payload) != expected:
        raise ContractError(f"{name} bytes do not match the committed SHA-256")


@dataclass(frozen=True, slots=True)
class NamedHash:
    """A named exact-byte commitment without choosing a manifest serialization."""

    name: str
    sha256: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.isascii():
            raise ContractError("fixture hash name must be nonempty ASCII")
        _digest(self.sha256, name=f"{self.name} sha256")

    def verify(self, payload: bytes) -> None:
        verify_sha256(payload, self.sha256, name=self.name)


@dataclass(frozen=True, slots=True)
class InventoryCheck:
    status: FixtureStatus
    row_count: int
    semantic_sha256: str
    detail: str

    def __post_init__(self) -> None:
        if type(self.row_count) is not int or self.row_count < 0:
            raise ContractError("fixture inventory row_count must be nonnegative")
        _digest(self.semantic_sha256, name="semantic_sha256")
        if not self.detail:
            raise ContractError("fixture inventory check requires a detail")


@dataclass(frozen=True, slots=True)
class X0ManifestRow:
    """The exact eight-field row schema from proposal 6.1."""

    id: str
    split: str
    source_key_hex: str
    accepted_attempt: int
    coefficient_seed_u64: int
    coefficient_bytes_sha256: str
    orbit_grid_sha256: str
    generator_contract_sha256: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"U\d{4}", self.id):
            raise ContractError("X0 row id must be U followed by four digits")
        source_id = int(self.id[1:])
        expected_split = "train" if source_id < 24 else "tune" if source_id < 32 else "heldout"
        if not 0 <= source_id < 40 or self.split != expected_split:
            raise ContractError("X0 row id/split is outside the frozen population")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_key_hex):
            raise ContractError("X0 source key must be 32 lowercase hexadecimal bytes")
        if type(self.accepted_attempt) is not int or not 0 <= self.accepted_attempt < 64:
            raise ContractError("X0 accepted attempt must be in [0,64)")
        if type(self.coefficient_seed_u64) is not int or not 0 <= self.coefficient_seed_u64 < 2**64:
            raise ContractError("X0 coefficient seed must be uint64")
        _digest(self.coefficient_bytes_sha256, name="coefficient_bytes_sha256")
        _digest(self.orbit_grid_sha256, name="orbit_grid_sha256")
        if (
            _digest(self.generator_contract_sha256, name="generator_contract_sha256")
            != CONTRACT_SHA256
        ):
            raise ContractError("X0 row does not bind the v4 generator contract")

    def as_dict(self) -> dict[str, str | int]:
        return {
            "accepted_attempt": self.accepted_attempt,
            "coefficient_bytes_sha256": self.coefficient_bytes_sha256,
            "coefficient_seed_u64": self.coefficient_seed_u64,
            "generator_contract_sha256": self.generator_contract_sha256,
            "id": self.id,
            "orbit_grid_sha256": self.orbit_grid_sha256,
            "source_key_hex": self.source_key_hex,
            "split": self.split,
        }


def x0_manifest_candidate_bytes(rows: tuple[X0ManifestRow, ...]) -> bytes:
    """Validate exact row order and return candidate bytes, never S0 acceptance."""

    if len(rows) != X0_ROW_COUNT:
        raise ContractError("X0 candidate manifest must contain exactly 40 rows")
    if tuple(row.id for row in rows) != tuple(f"U{index:04d}" for index in range(40)):
        raise ContractError("X0 candidate rows must be ordered U0000 through U0039")
    if len({row.source_key_hex for row in rows}) != 40:
        raise ContractError("X0 candidate source keys must be unique")
    return canonical_json_bytes([row.as_dict() for row in rows])


def check_x0_inventory(rows: tuple[X0ManifestRow, ...]) -> InventoryCheck:
    candidate = x0_manifest_candidate_bytes(rows)
    return InventoryCheck(
        FixtureStatus.BLOCKED_PENDING_F8_AMPLITUDE_AMENDMENT,
        len(rows),
        sha256_bytes(candidate),
        "FINAL_PROPOSAL:373-400 is infeasible until the F8 amplitude amendment is reviewed",
    )


def x0_generation_status() -> InventoryCheck:
    """Expose the current blocker without constructing any manifest row."""

    return InventoryCheck(
        FixtureStatus.BLOCKED_PENDING_F8_AMPLITUDE_AMENDMENT,
        0,
        sha256_bytes(b""),
        "FINAL_PROPOSAL:373-400 prevents the required 40-row X0 population",
    )


@dataclass(frozen=True, slots=True)
class TopologySemanticRow:
    identifier: str
    operator: str
    required_result: str
    reason: int | None
    detail: str

    def as_dict(self) -> dict[str, str | int | None]:
        return {
            "detail": self.detail,
            "id": self.identifier,
            "operator": self.operator,
            "reason": self.reason,
            "required_result": self.required_result,
        }


@dataclass(frozen=True, slots=True)
class TopologyFixtureCommitment:
    """Minimal hashes required by proposal 7.4, without invented JSON keys."""

    semantic: TopologySemanticRow
    operator_parameter_sha256: str
    array_hashes: tuple[NamedHash, ...]
    expected_field_hashes: tuple[NamedHash, ...]

    def __post_init__(self) -> None:
        _digest(self.operator_parameter_sha256, name="operator_parameter_sha256")
        for label, values in (
            ("topology array hashes", self.array_hashes),
            ("topology expected-field hashes", self.expected_field_hashes),
        ):
            names = tuple(value.name for value in values)
            if (
                not values
                or names != tuple(sorted(names, key=str.encode))
                or len(set(names)) != len(names)
            ):
                raise ContractError(f"{label} must be nonempty, unique, and ASCII sorted")


TOPOLOGY_SEMANTIC_ROWS = (
    TopologySemanticRow("T00", "identity", "accept", None, ""),
    TopologySemanticRow("T01", "rotate:+1/8", "accept", None, ""),
    TopologySemanticRow("T02", "rotate:-1/8", "accept", None, ""),
    TopologySemanticRow("T03", "homeomorphism:H+", "accept", None, ""),
    TopologySemanticRow("T04", "homeomorphism:H-", "accept", None, ""),
    TopologySemanticRow("T05", "degree-zero", "reject", 14, "DEGREE_ZERO"),
    TopologySemanticRow("T06", "harmonic:2", "reject", 14, "HARMONIC_2"),
    TopologySemanticRow("T07", "harmonic:8", "reject", 14, "HARMONIC_8"),
    TopologySemanticRow("T08", "reversal", "reject", 14, "REVERSAL"),
    TopologySemanticRow("T09", "alternating-starts", "reject", 17, "ALTERNATING_STARTS"),
    TopologySemanticRow("T10", "raw-bypass", "reject", 15, "RAW_BYPASS"),
    TopologySemanticRow("T11", "dense-state-copy", "reject", 16, "DENSE_STATE"),
    TopologySemanticRow("T12", "two-seams-one-edge", "reject", 18, "TWO_IN_ONE_EDGE"),
)


def topology_semantic_bytes() -> bytes:
    if len(TOPOLOGY_SEMANTIC_ROWS) != TOPOLOGY_ROW_COUNT:
        raise RuntimeError("topology semantic inventory drift")
    return canonical_json_bytes([row.as_dict() for row in TOPOLOGY_SEMANTIC_ROWS])


def check_topology_semantic_hash(expected_sha256: str) -> InventoryCheck:
    payload = topology_semantic_bytes()
    verify_sha256(payload, expected_sha256, name="topology semantic inventory")
    return InventoryCheck(
        FixtureStatus.VALIDATED,
        TOPOLOGY_ROW_COUNT,
        sha256_bytes(payload),
        "thirteen ordered topology semantics and first reasons match",
    )


@dataclass(frozen=True, slots=True, order=True)
class OperatorFixtureKey:
    gap: int
    width: int
    amplitude: float
    offset: int
    layout: str

    def __post_init__(self) -> None:
        if (
            self.gap not in OPERATOR_GAPS
            or self.width not in OPERATOR_WIDTHS
            or self.amplitude not in OPERATOR_AMPLITUDES
            or self.offset not in OPERATOR_OFFSETS
            or self.layout not in OPERATOR_LAYOUTS
        ):
            raise ContractError("operator fixture key is outside the frozen Cartesian inventory")

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "amplitude": self.amplitude,
            "gap": self.gap,
            "layout": self.layout,
            "offset": self.offset,
            "width": self.width,
        }


@dataclass(frozen=True, slots=True)
class OperatorLayout:
    run_bounds: tuple[tuple[int, int], ...]
    truth_plateaus: tuple[tuple[int, int], ...]
    expected_components: tuple[tuple[int, int], ...]
    reset_edge: int | None
    truth_negative_components: int

    @property
    def edge_count(self) -> int:
        return self.run_bounds[-1][1]


@dataclass(frozen=True, slots=True)
class OperatorFixture:
    """One exact Section 9 fixture before any manifest-schema choice."""

    key: OperatorFixtureKey
    layout: OperatorLayout
    run_bounds: NDArray[np.int32]
    truth_plateaus: NDArray[np.int32]
    expected_components: NDArray[np.int32]
    truth_plateau_mask: NDArray[np.uint8]
    target_mask: NDArray[np.uint8]
    edge_mask: NDArray[np.uint8]
    decoder_mask: NDArray[np.uint8]
    response: NDArray[np.float32]

    def __post_init__(self) -> None:
        for array in (
            self.run_bounds,
            self.truth_plateaus,
            self.expected_components,
            self.truth_plateau_mask,
            self.target_mask,
            self.edge_mask,
            self.decoder_mask,
            self.response,
        ):
            array.setflags(write=False)


@dataclass(frozen=True, slots=True)
class OperatorFixtureValidation:
    """Closed NOLA/decoder/association validation for one generated fixture."""

    fixture: OperatorFixture
    windows: tuple[tuple[int, int], ...]
    nola_response: NDArray[np.float64]
    nola_quantized_response: NDArray[np.float32]
    component_bounds: NDArray[np.int32]
    component_location: NDArray[np.int32]
    component_score: NDArray[np.float32]
    association: NDArray[np.uint8]
    positive_margin: float
    negative_margin: float

    def __post_init__(self) -> None:
        for array in (
            self.nola_response,
            self.nola_quantized_response,
            self.component_bounds,
            self.component_location,
            self.component_score,
            self.association,
        ):
            array.setflags(write=False)


@dataclass(frozen=True, slots=True)
class OperatorInventoryWitnesses:
    """Normative witnesses whose aggregate preimages are fixed by Section 9."""

    row_count: int
    unique_key_count: int
    semantic_bytes: int
    semantic_sha256: str
    inactive_lookup_bytes: int
    inactive_lookup_sha256: str
    key_inventory_bytes: int
    key_inventory_sha256: str
    edge0_digest_inventory_bytes: int
    edge0_digest_inventory_sha256: str


@dataclass(frozen=True, slots=True)
class OperatorFixtureCommitment:
    """The four row-level hashes named by proposal 9, independent of preimage encoding."""

    key: OperatorFixtureKey
    truth_plateaus_sha256: str
    masks_sha256: str
    expected_components_sha256: str
    generated_response_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "truth_plateaus_sha256",
            "masks_sha256",
            "expected_components_sha256",
            "generated_response_sha256",
        ):
            _digest(getattr(self, name), name=name)


def _multiple_at_least(value: int) -> int:
    return max(32, ((value + 31) // 32) * 32)


def _multiple_strictly_greater(value: int) -> int:
    return (value // 32 + 1) * 32


def _negative_component_count(
    runs: tuple[tuple[int, int], ...],
    plateaus: tuple[tuple[int, int], ...],
) -> int:
    count = 0
    for run_start, run_stop in runs:
        cursor = run_start
        for start, stop in plateaus:
            if not run_start <= start < stop <= run_stop:
                continue
            if cursor < start:
                count += 1
            cursor = stop
        if cursor < run_stop:
            count += 1
    return count


def operator_layout(key: OperatorFixtureKey) -> OperatorLayout:
    """Construct exact run and truth geometry without undefined response bytes."""

    step = key.width + key.gap
    reset_edge: int | None = None
    starts: tuple[int, ...]
    runs: tuple[tuple[int, int], ...]
    if key.layout == "left-boundary":
        starts = (key.offset, key.offset + step, key.offset + 2 * step)
        end = _multiple_at_least(starts[-1] + key.width + 64)
        runs = ((0, end),)
    elif key.layout in {"interior", "right-boundary"}:
        provisional = (
            32 + key.offset,
            32 + key.offset + step,
            32 + key.offset + 2 * step,
        )
        end = _multiple_at_least(provisional[-1] + key.width + 64)
        if key.layout == "right-boundary":
            translation = end - (provisional[-1] + key.width) - key.offset
            starts = tuple(start + translation for start in provisional)
        else:
            starts = provisional
        runs = ((0, end),)
    else:
        first_starts = (key.offset, key.offset + step)
        first_end = _multiple_at_least(key.offset + step + key.width + 1)
        reset_edge = first_end
        third_start = first_end + 1 + key.offset
        second_end = _multiple_strictly_greater(third_start + key.width + 64)
        starts = (*first_starts, third_start)
        runs = ((0, first_end), (first_end + 1, second_end))
    plateaus = tuple((start, start + key.width) for start in starts)
    for index, (start, stop) in enumerate(plateaus):
        if index and start < plateaus[index - 1][1]:
            raise ContractError("operator truth plateaus overlap")
        if not any(run_start <= start < stop <= run_stop for run_start, run_stop in runs):
            raise ContractError("operator truth plateau is outside its run")
    return OperatorLayout(
        run_bounds=runs,
        truth_plateaus=plateaus,
        expected_components=plateaus,
        reset_edge=reset_edge,
        truth_negative_components=_negative_component_count(runs, plateaus),
    )


@lru_cache(maxsize=1)
def operator_fixture_keys() -> tuple[OperatorFixtureKey, ...]:
    rows = tuple(
        OperatorFixtureKey(gap, width, amplitude, offset, layout)
        for gap in OPERATOR_GAPS
        for width in OPERATOR_WIDTHS
        for amplitude in OPERATOR_AMPLITUDES
        for offset in OPERATOR_OFFSETS
        for layout in OPERATOR_LAYOUTS
    )
    if len(rows) != OPERATOR_ROW_COUNT or len(set(rows)) != OPERATOR_ROW_COUNT:
        raise RuntimeError("operator Cartesian inventory drift")
    return rows


def operator_semantic_bytes() -> bytes:
    return canonical_json_bytes([row.as_dict() for row in operator_fixture_keys()])


def check_operator_semantic_hash(expected_sha256: str) -> InventoryCheck:
    payload = operator_semantic_bytes()
    verify_sha256(payload, OPERATOR_SEMANTIC_SHA256, name="canonical operator semantic inventory")
    verify_sha256(payload, expected_sha256, name="operator semantic inventory")
    return InventoryCheck(
        FixtureStatus.VALIDATED,
        OPERATOR_ROW_COUNT,
        sha256_bytes(payload),
        "canonical Section 9 operator semantics and Amendment 002 row-key bytes match",
    )


def _unsigned_bytes(value: int, width: int, *, name: str) -> bytes:
    if type(value) is not int or not 0 <= value < 1 << (8 * width):
        raise ContractError(f"{name} must be uint{8 * width}")
    return value.to_bytes(width, byteorder="big", signed=False)


def operator_row_key_bytes(key: OperatorFixtureKey) -> bytes:
    """Return canonical Amendment 002 key ``K`` (exactly ten bytes)."""

    amplitude_index = OPERATOR_AMPLITUDES.index(key.amplitude)
    layout_index = OPERATOR_LAYOUTS.index(key.layout)
    encoded = b"".join(
        (
            _unsigned_bytes(key.gap, 2, name="operator gap"),
            _unsigned_bytes(key.width, 2, name="operator width"),
            _unsigned_bytes(amplitude_index, 2, name="operator amplitude index"),
            _unsigned_bytes(key.offset, 2, name="operator offset"),
            _unsigned_bytes(layout_index, 2, name="operator layout index"),
        )
    )
    if len(encoded) != 10:  # pragma: no cover - construction invariant
        raise AssertionError("operator key must contain exactly ten bytes")
    return encoded


def operator_row_key_preimage(key: OperatorFixtureKey, edge_index: int = 0) -> bytes:
    """Return exact ``P(e)`` with one key-length prefix and raw uint32 edge."""

    key_bytes = operator_row_key_bytes(key)
    return b"".join(
        (
            OPERATOR_TAG,
            b"\x00",
            _unsigned_bytes(len(key_bytes), 8, name="operator key length"),
            key_bytes,
            _unsigned_bytes(edge_index, 4, name="operator edge index"),
        )
    )


def operator_edge_digest(key: OperatorFixtureKey, edge_index: int) -> bytes:
    """Return raw ``D(e)``; response selection always uses byte zero."""

    return hashlib.sha256(operator_row_key_preimage(key, edge_index)).digest()


def _float32_from_bits(bits: int) -> np.float32:
    encoded = _unsigned_bytes(bits, 4, name="binary32 bits")
    return np.frombuffer(encoded[::-1], dtype=np.dtype("<f4"), count=1)[0]


def operator_inactive_value_from_byte(digest_byte: int) -> np.float32:
    """Apply the frozen integer/divide/cast/cap operation exactly once."""

    if type(digest_byte) is not int or not 0 <= digest_byte <= 255:
        raise ContractError("operator digest byte must be uint8")
    numerator = 255 + 3 * digest_byte
    binary64 = np.float64(numerator) / np.float64(2550)
    rounded = cast(np.float32, np.asarray(binary64, dtype=np.dtype("<f4"))[()])
    cap = _float32_from_bits(0x3ECCCCCC)
    return cap if rounded > cap else rounded


@lru_cache(maxsize=1)
def operator_inactive_lookup_bytes() -> bytes:
    """Return the 256-entry lookup as raw increasing-``b`` little-endian bits."""

    values = np.asarray(
        [operator_inactive_value_from_byte(value) for value in range(256)],
        dtype=np.dtype("<f4"),
    )
    payload = values.view(np.dtype("<u4")).tobytes(order="C")
    if len(payload) != 1024 or sha256_bytes(payload) != OPERATOR_INACTIVE_LOOKUP_SHA256:
        raise RuntimeError("operator inactive lookup witness drift")
    return payload


def operator_inactive_value(key: OperatorFixtureKey, edge_index: int) -> np.float32:
    """Evaluate the inactive function, including for a hypothetical active edge."""

    digest_byte = int(operator_edge_digest(key, edge_index)[0])
    lookup = np.frombuffer(operator_inactive_lookup_bytes(), dtype=np.dtype("<f4"))
    return lookup[digest_byte]


_PLATEAU_BITS = (
    (0x3F19999A, 0x3F266666),
    (0x3F400000, 0x3F4CCCCD),
    (0x3F800000, 0x3F800000),
)


def operator_fixture(key: OperatorFixtureKey) -> OperatorFixture:
    """Generate exact masks, truth/component tables, and little-endian response."""

    layout = operator_layout(key)
    run_bounds = np.asarray(layout.run_bounds, dtype=np.dtype("<i4"))
    truth_plateaus = np.asarray(layout.truth_plateaus, dtype=np.dtype("<i4"))
    expected_components = np.asarray(layout.expected_components, dtype=np.dtype("<i4"))
    valid_mask = np.zeros(layout.edge_count, dtype=np.dtype("|u1"))
    for start, stop in layout.run_bounds:
        valid_mask[start:stop] = 1
    truth_plateau_mask = np.zeros(layout.edge_count, dtype=np.dtype("|u1"))
    response = np.zeros(layout.edge_count, dtype=np.dtype("<f4"))
    lookup = np.frombuffer(operator_inactive_lookup_bytes(), dtype=np.dtype("<f4"))
    for run_start, run_stop in layout.run_bounds:
        for edge_index in range(run_start, run_stop):
            response[edge_index] = lookup[int(operator_edge_digest(key, edge_index)[0])]

    amplitude_index = OPERATOR_AMPLITUDES.index(key.amplitude)
    ordinary = _float32_from_bits(_PLATEAU_BITS[amplitude_index][0])
    boosted = _float32_from_bits(_PLATEAU_BITS[amplitude_index][1])
    for start, stop in layout.truth_plateaus:
        truth_plateau_mask[start:stop] = 1
        response[start:stop] = ordinary
        lower_middle = start + (key.width - 1) // 2
        response[lower_middle] = boosted

    return OperatorFixture(
        key=key,
        layout=layout,
        run_bounds=np.ascontiguousarray(run_bounds),
        truth_plateaus=np.ascontiguousarray(truth_plateaus),
        expected_components=np.ascontiguousarray(expected_components),
        truth_plateau_mask=np.ascontiguousarray(truth_plateau_mask),
        target_mask=np.ascontiguousarray(valid_mask.copy()),
        edge_mask=np.ascontiguousarray(valid_mask.copy()),
        decoder_mask=np.ascontiguousarray(valid_mask.copy()),
        response=np.ascontiguousarray(response, dtype=np.dtype("<f4")),
    )


def operator_windows(run_bounds: NDArray[np.int32]) -> tuple[tuple[int, int], ...]:
    """Apply the frozen 127-edge/stride-32/terminal-right-aligned schedule."""

    windows: list[tuple[int, int]] = []
    for run_start, run_stop in run_bounds.tolist():
        if run_stop - run_start <= 127:
            starts = [run_start]
        else:
            final_start = run_stop - 127
            starts = list(range(run_start, final_start + 1, 32))
            if starts[-1] != final_start:
                starts.append(final_start)
        windows.extend((start, min(start + 127, run_stop)) for start in starts)
    return tuple(windows)


def _neumaier_add_array(
    total: NDArray[np.float64],
    correction: NDArray[np.float64],
    value: NDArray[np.float64],
) -> None:
    candidate = total + value
    use_total = np.abs(total) >= np.abs(value)
    correction += np.where(
        use_total,
        (total - candidate) + value,
        (value - candidate) + total,
    )
    total[:] = candidate


def _operator_nola_response(
    fixture: OperatorFixture,
) -> tuple[tuple[tuple[int, int], ...], NDArray[np.float64]]:
    """Reconstruct fixture windows with the exact F13 binary64 operation order."""

    edge_count = fixture.layout.edge_count
    windows = operator_windows(fixture.run_bounds)
    numerator = np.zeros(edge_count, dtype=np.float64)
    numerator_correction = np.zeros(edge_count, dtype=np.float64)
    denominator = np.zeros(edge_count, dtype=np.float64)
    denominator_correction = np.zeros(edge_count, dtype=np.float64)
    desired = fixture.response.astype(np.float64)
    valid = fixture.edge_mask.astype(np.bool_)
    for start, stop in windows:
        slot = np.arange(stop - start, dtype=np.float64)
        taper = np.float64(1e-3) + np.float64(1.0 - 1e-3) * np.sin(
            np.pi * (slot + np.float64(0.5)) / np.float64(127.0)
        ) ** np.float64(2.0)
        local_valid = valid[start:stop].astype(np.float64)
        numerator_value = taper * local_valid * desired[start:stop]
        denominator_value = taper * local_valid
        _neumaier_add_array(
            numerator[start:stop],
            numerator_correction[start:stop],
            numerator_value,
        )
        _neumaier_add_array(
            denominator[start:stop],
            denominator_correction[start:stop],
            denominator_value,
        )
    numerator += numerator_correction
    denominator += denominator_correction
    if (
        not np.all(np.isfinite(numerator))
        or not np.all(np.isfinite(denominator))
        or np.any(denominator[valid] < np.float64(1e-3))
        or np.any(denominator[~valid] != 0.0)
        or np.any(numerator[~valid] != 0.0)
    ):
        raise ContractError("operator fixture violates exact positive NOLA")
    reconstructed = np.zeros(edge_count, dtype=np.float64)
    reconstructed[valid] = numerator[valid] / denominator[valid]
    if not np.all(np.isfinite(reconstructed)):
        raise ContractError("operator NOLA response is nonfinite")
    return windows, np.ascontiguousarray(reconstructed)


def validate_operator_fixture(key: OperatorFixtureKey) -> OperatorFixtureValidation:
    """Validate every response byte, mask, NOLA result, decode, and association."""

    from pams.temporac.decode import decode_identity

    fixture = operator_fixture(key)
    if not (
        np.array_equal(fixture.target_mask, fixture.edge_mask)
        and np.array_equal(fixture.edge_mask, fixture.decoder_mask)
    ):
        raise ContractError("operator target/NOLA/decoder masks differ")
    if np.any(fixture.response[fixture.edge_mask == 0] != np.float32(0.0)):
        raise ContractError("operator response outside edge_mask is not canonical zero")
    if not np.array_equal(
        fixture.truth_plateau_mask,
        (fixture.response >= np.float32(0.5)).astype(np.uint8),
    ):
        raise ContractError("operator truth mask does not equal thresholded response")

    windows, nola_response = _operator_nola_response(fixture)
    decoded = decode_identity(nola_response, fixture.decoder_mask, fixture.run_bounds)
    if decoded.decoder_invocations != 1 or decoded.abstain or decoded.count != 3:
        raise ContractError("operator decoder was not invoked exactly once for three truths")
    if decoded.response.tobytes(order="C") != fixture.response.tobytes(order="C"):
        raise ContractError("operator NOLA quantization changed generated response bytes")
    if not np.array_equal(decoded.component_bounds, fixture.expected_components):
        raise ContractError("operator decoded components differ from expected components")

    expected_locations = np.asarray(
        [start + (key.width - 1) // 2 for start, _ in fixture.layout.truth_plateaus],
        dtype=np.dtype("<i4"),
    )
    expected_scores = np.asarray(
        [np.max(fixture.response[start:stop]) for start, stop in fixture.layout.truth_plateaus],
        dtype=np.dtype("<f4"),
    )
    if not np.array_equal(decoded.component_location, expected_locations):
        raise ContractError("operator decoder locations violate the lower-middle rule")
    if not np.array_equal(decoded.component_score, expected_scores):
        raise ContractError("operator decoder scores differ from exact plateau maxima")

    association = np.zeros((3, decoded.count), dtype=np.dtype("|u1"))
    for truth_index, (truth_start, truth_stop) in enumerate(fixture.truth_plateaus.tolist()):
        for component_index, (component_start, component_stop) in enumerate(
            decoded.component_bounds.tolist()
        ):
            association[truth_index, component_index] = int(
                max(truth_start, component_start) < min(truth_stop, component_stop)
            )
    if not (np.all(np.sum(association, axis=0) == 1) and np.all(np.sum(association, axis=1) == 1)):
        raise ContractError("operator truth/component association is not one-to-one")

    positive_values = np.asarray(
        [np.max(fixture.response[start:stop]) for start, stop in fixture.layout.truth_plateaus],
        dtype=np.float64,
    )
    negative_support = (fixture.edge_mask == 1) & (fixture.truth_plateau_mask == 0)
    if positive_values.size == 0 or not np.any(negative_support):
        raise ContractError("operator positive or negative margin support is empty")
    positive_margin = float(np.min(positive_values) - np.float64(0.5))
    negative_margin = float(
        np.float64(0.5) - np.max(fixture.response[negative_support].astype(np.float64))
    )
    if positive_margin <= 0.0 or negative_margin <= 0.0:
        raise ContractError("operator response does not have strict decoder margins")

    if key.amplitude == 1.00 and key.width in {2, 4}:
        one = _float32_from_bits(0x3F800000)
        for start, stop in fixture.layout.truth_plateaus:
            if not np.all(fixture.response[start:stop] == one):
                raise ContractError("amplitude-one width-2/4 plateau is not exactly flat")

    return OperatorFixtureValidation(
        fixture=fixture,
        windows=windows,
        nola_response=nola_response,
        nola_quantized_response=np.ascontiguousarray(decoded.response, dtype=np.dtype("<f4")),
        component_bounds=np.ascontiguousarray(decoded.component_bounds, dtype=np.dtype("<i4")),
        component_location=np.ascontiguousarray(decoded.component_location, dtype=np.dtype("<i4")),
        component_score=np.ascontiguousarray(decoded.component_score, dtype=np.dtype("<f4")),
        association=np.ascontiguousarray(association),
        positive_margin=positive_margin,
        negative_margin=negative_margin,
    )


@lru_cache(maxsize=1)
def validate_operator_inventory_witnesses() -> OperatorInventoryWitnesses:
    """Recompute all four frozen full-inventory/lookup witnesses fail-closed."""

    keys = operator_fixture_keys()
    encoded_keys = tuple(operator_row_key_bytes(key) for key in keys)
    if (
        len(encoded_keys) != OPERATOR_ROW_COUNT
        or any(len(value) != 10 for value in encoded_keys)
        or len(set(encoded_keys)) != OPERATOR_ROW_COUNT
        or any(left >= right for left, right in zip(encoded_keys, encoded_keys[1:], strict=False))
    ):
        raise ContractError("operator keys are not 10,368 unique strictly ordered 10-byte values")

    semantic = operator_semantic_bytes()
    lookup = operator_inactive_lookup_bytes()
    key_inventory = b"".join(
        _unsigned_bytes(10, 8, name="operator key length") + key for key in encoded_keys
    )
    edge0_inventory = b"".join(operator_edge_digest(key, 0) for key in keys)
    observed = OperatorInventoryWitnesses(
        row_count=len(keys),
        unique_key_count=len(set(encoded_keys)),
        semantic_bytes=len(semantic),
        semantic_sha256=sha256_bytes(semantic),
        inactive_lookup_bytes=len(lookup),
        inactive_lookup_sha256=sha256_bytes(lookup),
        key_inventory_bytes=len(key_inventory),
        key_inventory_sha256=sha256_bytes(key_inventory),
        edge0_digest_inventory_bytes=len(edge0_inventory),
        edge0_digest_inventory_sha256=sha256_bytes(edge0_inventory),
    )
    expected = OperatorInventoryWitnesses(
        row_count=10_368,
        unique_key_count=10_368,
        semantic_bytes=746_714,
        semantic_sha256=OPERATOR_SEMANTIC_SHA256,
        inactive_lookup_bytes=1_024,
        inactive_lookup_sha256=OPERATOR_INACTIVE_LOOKUP_SHA256,
        key_inventory_bytes=186_624,
        key_inventory_sha256=OPERATOR_KEY_INVENTORY_SHA256,
        edge0_digest_inventory_bytes=331_776,
        edge0_digest_inventory_sha256=OPERATOR_EDGE0_DIGEST_INVENTORY_SHA256,
    )
    if observed != expected:
        raise ContractError(f"operator inventory witness mismatch: {observed!r}")
    return observed


@dataclass(frozen=True, slots=True)
class CountMetricFixtureHashes:
    """The four P2-METRIC byte commitments explicitly required by the plan."""

    evaluator_code_sha256: str
    fixture_input_sha256: str
    expected_output_sha256: str
    attack_global_fail_sha256: str
    schema: str = COUNT_METRIC_FIXTURE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != COUNT_METRIC_FIXTURE_SCHEMA:
            raise ContractError("count-metric fixture schema differs from the plan")
        for name in (
            "evaluator_code_sha256",
            "fixture_input_sha256",
            "expected_output_sha256",
            "attack_global_fail_sha256",
        ):
            _digest(getattr(self, name), name=name)

    def index_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "attack_global_fail_sha256": self.attack_global_fail_sha256,
                "evaluator_code_sha256": self.evaluator_code_sha256,
                "expected_output_sha256": self.expected_output_sha256,
                "fixture_input_sha256": self.fixture_input_sha256,
                "schema": self.schema,
            }
        )

    def verify(
        self,
        *,
        evaluator_code: bytes,
        fixture_input: bytes,
        expected_output: bytes,
        attack_global_fail: bytes,
    ) -> None:
        verify_sha256(evaluator_code, self.evaluator_code_sha256, name="evaluator code")
        verify_sha256(fixture_input, self.fixture_input_sha256, name="fixture input")
        verify_sha256(expected_output, self.expected_output_sha256, name="expected output")
        verify_sha256(
            attack_global_fail,
            self.attack_global_fail_sha256,
            name="attack expected-global-fail",
        )


__all__ = [
    "COUNT_METRIC_FIXTURE_SCHEMA",
    "OPERATOR_AMPLITUDES",
    "OPERATOR_EDGE0_DIGEST_INVENTORY_SHA256",
    "OPERATOR_GAPS",
    "OPERATOR_INACTIVE_LOOKUP_SHA256",
    "OPERATOR_KEY_INVENTORY_SHA256",
    "OPERATOR_LAYOUTS",
    "OPERATOR_OFFSETS",
    "OPERATOR_ROW_COUNT",
    "OPERATOR_SEMANTIC_SHA256",
    "OPERATOR_TAG",
    "OPERATOR_WIDTHS",
    "TOPOLOGY_ROW_COUNT",
    "TOPOLOGY_SEMANTIC_ROWS",
    "X0_ROW_COUNT",
    "CountMetricFixtureHashes",
    "FixtureStatus",
    "InventoryCheck",
    "NamedHash",
    "OperatorFixture",
    "OperatorFixtureKey",
    "OperatorFixtureCommitment",
    "OperatorFixtureValidation",
    "OperatorInventoryWitnesses",
    "OperatorLayout",
    "TopologyFixtureCommitment",
    "TopologySemanticRow",
    "UndefinedRowKeyPreimageError",
    "X0ManifestRow",
    "check_operator_semantic_hash",
    "check_topology_semantic_hash",
    "check_x0_inventory",
    "operator_edge_digest",
    "operator_fixture",
    "operator_fixture_keys",
    "operator_inactive_lookup_bytes",
    "operator_inactive_value",
    "operator_inactive_value_from_byte",
    "operator_layout",
    "operator_row_key_bytes",
    "operator_row_key_preimage",
    "operator_semantic_bytes",
    "operator_windows",
    "topology_semantic_bytes",
    "validate_operator_fixture",
    "validate_operator_inventory_witnesses",
    "verify_sha256",
    "x0_generation_status",
    "x0_manifest_candidate_bytes",
]
