"""Independent, trace-only replay of TempoRAC Amendment 002 operator semantics.

This audit harness deliberately does not import ``pams.temporac``.  It rebuilds
the frozen Cartesian population, byte preimages, float32 responses, masks,
window schedule, NOLA reconstruction, one-pass decoder, and associations from
the normative text, then checks every candidate row and detached receipt.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


GAPS = (4, 5, 8, 16, 32, 64, 127, 128, 129)
WIDTHS = (1, 2, 4)
AMPLITUDES = (0.60, 0.75, 1.00)
OFFSETS = tuple(range(32))
LAYOUTS = ("interior", "left-boundary", "right-boundary", "run-reset")
TAG = b"temporac.operator.v4"
ROW_COUNT = 10_368
EXPECTED_ROOTS = {
    "semantic": (746_714, "0fe8b2619a3f33cc4a83485053888c04c567b015eea65f81bf6cd9ec715ee4f2"),
    "inactive_lookup": (1_024, "afa6eff0d744acfdf9cf78adc43aa8daf01633f3345d7567d642a7782885040d"),
    "key_inventory": (186_624, "18dfdafe237b77565de36bab6d4642efb314ec1cbbd76e852938ed5c4d424a55"),
    "edge0_digest_inventory": (
        331_776,
        "ba797f6ae59cbbb40f19a61199afaaa1ac5b2c001a3c9fc34f1dba8f79cf78e4",
    ),
}
PLATEAU_BITS = (
    (0x3F19999A, 0x3F266666),
    (0x3F400000, 0x3F4CCCCD),
    (0x3F800000, 0x3F800000),
)
MANIFEST_ROW_KEYS = frozenset(
    {
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
    }
)


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(payload: Any) -> bytes:
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


def no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssertionError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_canonical(path: Path) -> tuple[Any, bytes]:
    payload = path.read_bytes()
    parsed = json.loads(payload, object_pairs_hook=no_duplicate_object)
    assert payload == canonical(parsed), f"noncanonical JSON: {path}"
    return parsed, payload


def f4_from_bits(bits: int) -> np.float32:
    return np.frombuffer(struct.pack("<I", bits), dtype=np.dtype("<f4"), count=1)[0]


CAP = f4_from_bits(0x3ECCCCCC)


def inactive_from_byte(value: int) -> np.float32:
    assert type(value) is int and 0 <= value <= 255
    numerator = 255 + 3 * value
    rounded = np.asarray(
        np.float64(numerator) / np.float64(2550), dtype=np.dtype("<f4")
    )[()]
    return CAP if rounded > CAP else rounded


INACTIVE = np.asarray(
    [inactive_from_byte(value) for value in range(256)], dtype=np.dtype("<f4")
)


def key_bytes(gap: int, width: int, aidx: int, offset: int, lidx: int) -> bytes:
    for value in (gap, width, aidx, offset, lidx):
        assert type(value) is int and 0 <= value < 2**16
    encoded = struct.pack(">HHHHH", gap, width, aidx, offset, lidx)
    assert len(encoded) == 10
    return encoded


def edge_digest(key: bytes, edge: int) -> bytes:
    assert len(key) == 10 and type(edge) is int and 0 <= edge < 2**32
    preimage = TAG + b"\x00" + struct.pack(">Q", 10) + key + struct.pack(">I", edge)
    return hashlib.sha256(preimage).digest()


def multiple_at_least(value: int) -> int:
    return max(32, ((value + 31) // 32) * 32)


def multiple_strictly_greater(value: int) -> int:
    return (value // 32 + 1) * 32


def layout_geometry(
    gap: int, width: int, offset: int, layout: str
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...], int | None]:
    step = width + gap
    reset: int | None = None
    if layout == "left-boundary":
        starts = (offset, offset + step, offset + 2 * step)
        stop = multiple_at_least(starts[-1] + width + 64)
        runs = ((0, stop),)
    elif layout in {"interior", "right-boundary"}:
        provisional = (32 + offset, 32 + offset + step, 32 + offset + 2 * step)
        stop = multiple_at_least(provisional[-1] + width + 64)
        if layout == "right-boundary":
            translation = stop - (provisional[-1] + width) - offset
            starts = tuple(start + translation for start in provisional)
        else:
            starts = provisional
        runs = ((0, stop),)
    else:
        assert layout == "run-reset"
        first = (offset, offset + step)
        first_stop = multiple_at_least(offset + step + width + 1)
        reset = first_stop
        third = first_stop + 1 + offset
        second_stop = multiple_strictly_greater(third + width + 64)
        starts = (*first, third)
        runs = ((0, first_stop), (first_stop + 1, second_stop))
    plateaus = tuple((start, start + width) for start in starts)
    assert len(plateaus) == 3
    for index, (start, stop) in enumerate(plateaus):
        assert start < stop
        assert any(left <= start < stop <= right for left, right in runs)
        if index:
            assert plateaus[index - 1][1] <= start
    return runs, plateaus, reset


def negative_components(
    runs: tuple[tuple[int, int], ...], plateaus: tuple[tuple[int, int], ...]
) -> int:
    count = 0
    for run_start, run_stop in runs:
        cursor = run_start
        for start, stop in plateaus:
            if run_start <= start < stop <= run_stop:
                count += int(cursor < start)
                cursor = stop
        count += int(cursor < run_stop)
    return count


def window_schedule(runs: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for run_start, run_stop in runs:
        if run_stop - run_start <= 127:
            starts = [run_start]
        else:
            final = run_stop - 127
            starts = list(range(run_start, final + 1, 32))
            if starts[-1] != final:
                starts.append(final)
        result.extend((start, min(start + 127, run_stop)) for start in starts)
    return tuple(result)


def neumaier_add(total: np.ndarray[Any, Any], correction: np.ndarray[Any, Any], value: np.ndarray[Any, Any]) -> None:
    candidate = total + value
    correction += np.where(
        np.abs(total) >= np.abs(value),
        (total - candidate) + value,
        (value - candidate) + total,
    )
    total[:] = candidate


def nola(
    response: np.ndarray[Any, Any],
    valid: np.ndarray[Any, Any],
    windows: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    numerator = np.zeros(response.size, dtype=np.float64)
    numerator_c = np.zeros(response.size, dtype=np.float64)
    denominator = np.zeros(response.size, dtype=np.float64)
    denominator_c = np.zeros(response.size, dtype=np.float64)
    widened = response.astype(np.float64)
    valid_bool = valid.astype(np.bool_)
    for start, stop in windows:
        slot = np.arange(stop - start, dtype=np.float64)
        taper = np.float64(1e-3) + np.float64(1.0 - 1e-3) * np.sin(
            np.pi * (slot + np.float64(0.5)) / np.float64(127.0)
        ) ** np.float64(2.0)
        local = valid_bool[start:stop].astype(np.float64)
        neumaier_add(
            numerator[start:stop], numerator_c[start:stop], taper * local * widened[start:stop]
        )
        neumaier_add(denominator[start:stop], denominator_c[start:stop], taper * local)
    numerator += numerator_c
    denominator += denominator_c
    assert np.all(np.isfinite(numerator)) and np.all(np.isfinite(denominator))
    assert np.all(denominator[valid_bool] >= np.float64(1e-3))
    assert np.all(denominator[~valid_bool] == 0.0)
    assert np.all(numerator[~valid_bool] == 0.0)
    result = np.zeros(response.size, dtype=np.float64)
    result[valid_bool] = numerator[valid_bool] / denominator[valid_bool]
    assert np.all(np.isfinite(result))
    return result, denominator


def decode_once(
    response: np.ndarray[Any, Any],
    decoder_mask: np.ndarray[Any, Any],
    runs: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    active = (decoder_mask == 1) & (response >= np.float32(0.5))
    bounds: list[tuple[int, int]] = []
    locations: list[int] = []
    scores: list[np.float32] = []
    for run_start, run_stop in runs:
        edge = run_start
        while edge < run_stop:
            if not active[edge]:
                edge += 1
                continue
            start = edge
            while edge < run_stop and active[edge]:
                edge += 1
            stop = edge
            values = response[start:stop]
            score = np.max(values)
            maxima = np.flatnonzero(values == score)
            location = (start + int(maxima[0]) + start + int(maxima[-1])) // 2
            bounds.append((start, stop))
            locations.append(location)
            scores.append(score)
    return (
        np.asarray(bounds, dtype=np.dtype("<i4")).reshape((-1, 2)),
        np.asarray(locations, dtype=np.dtype("<i4")),
        np.asarray(scores, dtype=np.dtype("<f4")),
    )


def array_sha(array: np.ndarray[Any, Any]) -> str:
    return sha(np.ascontiguousarray(array).tobytes(order="C"))


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    candidate = root / "data/temporac_p00_v4/operator_candidate_20260816"
    manifest, manifest_bytes = load_canonical(
        candidate / "temporac.operator-manifest.candidate-v1.json"
    )
    schema, schema_bytes = load_canonical(candidate / "candidate_schema.json")
    environment, environment_bytes = load_canonical(candidate / "environment_lock.json")
    witness, witness_bytes = load_canonical(candidate / "replay_witness.json")
    receipt, receipt_bytes = load_canonical(candidate / "candidate_receipt.json")

    assert len(manifest) == ROW_COUNT
    assert schema["authoritative"] is False
    assert schema["candidate_only"] is True
    assert schema["normative_schema_claimed"] is False
    assert schema["manifest_row_keys"] == sorted(MANIFEST_ROW_KEYS, key=str.encode)
    assert receipt["authoritative"] is False and receipt["authorizes"] == []
    assert receipt["manifest_schema_authoritative"] is False
    assert receipt["candidate_status"] == "CANDIDATE_PENDING_FRESH_REVIEW"
    assert receipt["p2_status"] == receipt["s0_status"] == "NOT_CLAIMED"
    assert set(receipt["blockers"]) == {
        "canonical_manifest_schema_not_frozen",
        "fresh_candidate_review_required",
        "P2_not_claimed",
        "S0_not_claimed",
        "launch_authorization_zero",
    }

    expected_names = {
        "temporac.operator-manifest.candidate-v1.json",
        "candidate_schema.json",
        "environment_lock.json",
        "replay_witness.json",
        "candidate_receipt.json",
    }
    actual_names = {path.name for path in candidate.iterdir()}
    assert actual_names == expected_names
    for path in candidate.iterdir():
        assert path.is_file() and not path.is_symlink()
    member_payloads = {
        "temporac.operator-manifest.candidate-v1.json": manifest_bytes,
        "candidate_schema.json": schema_bytes,
        "environment_lock.json": environment_bytes,
        "replay_witness.json": witness_bytes,
    }
    assert receipt["members"] == {name: sha(payload) for name, payload in member_payloads.items()}

    runtime_expected = {
        "architecture": platform.machine().lower(),
        "byteorder": sys.byteorder,
        "docker_image": os.environ.get("TEMPORAC_OPERATOR_DOCKER_IMAGE"),
        "numpy_version": np.__version__,
        "platform_system": platform.system(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }
    for name, value in runtime_expected.items():
        assert environment[name] == value
    record = importlib.metadata.distribution("numpy").read_text("RECORD")
    assert record is not None
    assert environment["numpy_init_sha256"] == file_sha(Path(np.__file__).resolve())
    assert environment["numpy_record_sha256"] == sha(record.encode("utf-8"))
    assert environment["python_executable_sha256"] == file_sha(Path(sys.executable).resolve())

    binding_paths = {
        "contract_module_sha256": root / "src/pams/temporac/contract.py",
        "cue_source_sha256": root / "src/pams/temporac/cue.py",
        "decoder_source_sha256": root / "src/pams/temporac/decode.py",
        "fixture_source_sha256": root / "src/pams/temporac/fixtures.py",
        "generator_source_sha256": root
        / "scripts/experiments/generate_temporac_operator_manifest_v4.py",
        "nola_source_sha256": root / "src/pams/temporac/nola.py",
        "proposal_sha256": root / "refine-logs/FINAL_PROPOSAL.md",
        "proposal_temporac_copy_sha256": root / "refine-logs/temporac/FINAL_PROPOSAL.md",
        "test_source_sha256": root / "tests/temporac/test_operator_manifest.py",
    }
    expected_bindings = {name: file_sha(path) for name, path in binding_paths.items()}
    expected_bindings["runtime_lock_sha256"] = sha(environment_bytes)
    assert receipt["bindings"] == expected_bindings

    semantic_rows: list[dict[str, Any]] = []
    encoded_keys: list[bytes] = []
    edge0_digests: list[bytes] = []
    aggregate_names = (
        "association",
        "components",
        "decoder_masks",
        "edge_masks",
        "nola_responses",
        "responses",
        "target_masks",
        "truth_masks",
        "truth_plateaus",
    )
    aggregate = {name: hashlib.sha256() for name in aggregate_names}
    aggregate_bytes = {name: 0 for name in aggregate_names}
    seen_keys: set[bytes] = set()
    b255_edges = 0
    b255_rows = 0
    first_b255: dict[str, Any] | None = None
    flat_rows = 0
    flat_plateaus = 0
    minimum_denominator = float("inf")
    worst_positive_margin = float("inf")
    worst_negative_margin = float("inf")
    maximum_nola_float32_error = 0.0
    row_index = 0

    for gap in GAPS:
        for width in WIDTHS:
            for aidx, amplitude in enumerate(AMPLITUDES):
                for offset in OFFSETS:
                    for lidx, layout in enumerate(LAYOUTS):
                        row = manifest[row_index]
                        assert frozenset(row) == MANIFEST_ROW_KEYS
                        assert (row["gap"], row["width"], row["amplitude"]) == (
                            gap,
                            width,
                            amplitude,
                        )
                        assert (row["offset"], row["layout"]) == (offset, layout)
                        semantic_rows.append(
                            {
                                "amplitude": amplitude,
                                "gap": gap,
                                "layout": layout,
                                "offset": offset,
                                "width": width,
                            }
                        )
                        key = key_bytes(gap, width, aidx, offset, lidx)
                        assert key not in seen_keys
                        if encoded_keys:
                            assert encoded_keys[-1] < key
                        seen_keys.add(key)
                        encoded_keys.append(key)
                        edge0_digests.append(edge_digest(key, 0))
                        assert row["key_hex"] == key.hex()

                        runs, plateaus, reset = layout_geometry(gap, width, offset, layout)
                        edge_count = runs[-1][1]
                        assert row["edge_count"] == edge_count
                        assert row["reset_edge"] == reset
                        assert row["truth_negative_components"] == negative_components(
                            runs, plateaus
                        )
                        run_bounds = np.asarray(runs, dtype=np.dtype("<i4"))
                        truth_plateaus = np.asarray(plateaus, dtype=np.dtype("<i4"))
                        expected_components = np.asarray(plateaus, dtype=np.dtype("<i4"))
                        valid = np.zeros(edge_count, dtype=np.dtype("|u1"))
                        truth_mask = np.zeros(edge_count, dtype=np.dtype("|u1"))
                        response = np.zeros(edge_count, dtype=np.dtype("<f4"))
                        row_b255 = 0
                        for run_start, run_stop in runs:
                            valid[run_start:run_stop] = 1
                            for edge in range(run_start, run_stop):
                                digest_byte = edge_digest(key, edge)[0]
                                response[edge] = INACTIVE[digest_byte]
                        ordinary = f4_from_bits(PLATEAU_BITS[aidx][0])
                        boosted = f4_from_bits(PLATEAU_BITS[aidx][1])
                        for start, stop in plateaus:
                            truth_mask[start:stop] = 1
                            response[start:stop] = ordinary
                            response[start + (width - 1) // 2] = boosted
                        for run_start, run_stop in runs:
                            for edge in range(run_start, run_stop):
                                if truth_mask[edge] == 0 and edge_digest(key, edge)[0] == 255:
                                    row_b255 += 1
                                    if first_b255 is None:
                                        first_b255 = {
                                            "edge": edge,
                                            "key_hex": key.hex(),
                                            "response_f4_le_hex": response[edge].tobytes().hex(),
                                        }
                        b255_edges += row_b255
                        b255_rows += int(row_b255 > 0)
                        if amplitude == 1.00 and width in {2, 4}:
                            flat_rows += 1
                            flat_plateaus += 3
                            for start, stop in plateaus:
                                assert response[start:stop].tobytes() == bytes.fromhex(
                                    "0000803f"
                                ) * width
                        assert np.all(response[valid == 0] == np.float32(0.0))
                        assert np.array_equal(
                            truth_mask, (response >= np.float32(0.5)).astype(np.uint8)
                        )

                        windows_tuple = window_schedule(runs)
                        windows = np.asarray(windows_tuple, dtype=np.dtype("<i4")).reshape((-1, 2))
                        reconstructed, denominator = nola(response, valid, windows_tuple)
                        quantized = np.asarray(reconstructed, dtype=np.dtype("<f4"))
                        assert quantized.tobytes(order="C") == response.tobytes(order="C")
                        minimum_denominator = min(
                            minimum_denominator, float(np.min(denominator[valid == 1]))
                        )
                        maximum_nola_float32_error = max(
                            maximum_nola_float32_error,
                            float(
                                np.max(
                                    np.abs(
                                        reconstructed[valid == 1]
                                        - response[valid == 1].astype(np.float64)
                                    )
                                )
                            ),
                        )
                        decoded, locations, scores = decode_once(quantized, valid, runs)
                        assert decoded.shape == (3, 2)
                        assert np.array_equal(decoded, expected_components)
                        expected_locations = np.asarray(
                            [start + (width - 1) // 2 for start, _ in plateaus],
                            dtype=np.dtype("<i4"),
                        )
                        assert np.array_equal(locations, expected_locations)
                        expected_scores = np.asarray(
                            [np.max(response[start:stop]) for start, stop in plateaus],
                            dtype=np.dtype("<f4"),
                        )
                        assert np.array_equal(scores, expected_scores)
                        association = np.zeros((3, 3), dtype=np.dtype("|u1"))
                        for truth_index, (truth_start, truth_stop) in enumerate(plateaus):
                            for component_index, (component_start, component_stop) in enumerate(
                                decoded.tolist()
                            ):
                                association[truth_index, component_index] = int(
                                    max(truth_start, component_start)
                                    < min(truth_stop, component_stop)
                                )
                        assert np.all(np.sum(association, axis=0) == 1)
                        assert np.all(np.sum(association, axis=1) == 1)
                        positive = min(
                            float(np.max(response[start:stop]).astype(np.float64))
                            for start, stop in plateaus
                        ) - 0.5
                        negative_support = (valid == 1) & (truth_mask == 0)
                        assert np.any(negative_support)
                        negative = 0.5 - float(
                            np.max(response[negative_support]).astype(np.float64)
                        )
                        assert positive >= 0.10 and negative >= 0.10
                        worst_positive_margin = min(worst_positive_margin, positive)
                        worst_negative_margin = min(worst_negative_margin, negative)

                        arrays = {
                            "association_sha256": association,
                            "component_location_sha256": locations,
                            "component_score_sha256": scores,
                            "decoded_components_sha256": decoded,
                            "decoder_mask_sha256": valid,
                            "edge_mask_sha256": valid,
                            "expected_components_sha256": expected_components,
                            "generated_response_sha256": response,
                            "nola_quantized_response_sha256": quantized,
                            "run_bounds_sha256": run_bounds,
                            "target_mask_sha256": valid,
                            "truth_plateau_mask_sha256": truth_mask,
                            "truth_plateaus_sha256": truth_plateaus,
                            "windows_sha256": windows,
                        }
                        for field, array in arrays.items():
                            assert row[field] == array_sha(array), (row_index, field)

                        payloads = {
                            "association": association.tobytes(order="C"),
                            "components": expected_components.tobytes(order="C"),
                            "decoder_masks": valid.tobytes(order="C"),
                            "edge_masks": valid.tobytes(order="C"),
                            "nola_responses": quantized.tobytes(order="C"),
                            "responses": response.tobytes(order="C"),
                            "target_masks": valid.tobytes(order="C"),
                            "truth_masks": truth_mask.tobytes(order="C"),
                            "truth_plateaus": truth_plateaus.tobytes(order="C"),
                        }
                        for name, payload in payloads.items():
                            aggregate[name].update(payload)
                            aggregate_bytes[name] += len(payload)
                        row_index += 1

    assert row_index == ROW_COUNT and len(seen_keys) == ROW_COUNT
    semantic_bytes = canonical(semantic_rows)
    lookup_bytes = INACTIVE.view(np.dtype("<u4")).tobytes(order="C")
    key_inventory = b"".join(struct.pack(">Q", 10) + key for key in encoded_keys)
    edge0_inventory = b"".join(edge0_digests)
    observed_roots = {
        "semantic": (len(semantic_bytes), sha(semantic_bytes)),
        "inactive_lookup": (len(lookup_bytes), sha(lookup_bytes)),
        "key_inventory": (len(key_inventory), sha(key_inventory)),
        "edge0_digest_inventory": (len(edge0_inventory), sha(edge0_inventory)),
    }
    assert observed_roots == EXPECTED_ROOTS

    uncapped = np.asarray(
        [np.asarray(np.float64(255 + 3 * value) / np.float64(2550), dtype="<f4")[()] for value in range(256)],
        dtype=np.dtype("<f4"),
    )
    changed = np.flatnonzero(uncapped.view("<u4") != INACTIVE.view("<u4")).tolist()
    assert changed == [255]
    assert INACTIVE[255].tobytes().hex() == "cccccc3e"
    assert float(np.float64(INACTIVE[255])) == 0.3999999761581421
    assert 0.5 - float(np.float64(INACTIVE[255])) == 0.10000002384185791
    assert (b255_edges, b255_rows, first_b255) == (
        8_945,
        5_665,
        {"edge": 42, "key_hex": "00040001000000000001", "response_f4_le_hex": "cccccc3e"},
    )
    assert (flat_rows, flat_plateaus) == (2_304, 6_912)

    expected_aggregates = {
        name: {
            "bytes": aggregate_bytes[name],
            "order": "direct raw C-order row-array concatenation in frozen Cartesian order",
            "sha256": digest.hexdigest(),
        }
        for name, digest in aggregate.items()
    }
    assert witness["candidate_aggregate_preimages"] == expected_aggregates
    assert witness["candidate_manifest_bytes"] == len(manifest_bytes)
    assert witness["candidate_manifest_sha256"] == sha(manifest_bytes)
    assert witness["candidate_status"] == "CANDIDATE_PENDING_FRESH_REVIEW"
    assert witness["actual_inactive_b255"] == {
        "edge_count": 8_945,
        "first": first_b255,
        "row_count": 5_665,
    }
    assert witness["all_rows_validated"] == {
        "association_rule": "truth/component adjacency iff half-open edge sets intersect",
        "decoder_invocations_per_row": 1,
        "flat_amplitude_one_plateaus": 6_912,
        "flat_amplitude_one_rows": 2_304,
        "nola_rule": "every covering window receives the same desired float32 response",
        "response_mask_component_nola_decoder_association": True,
        "row_count": ROW_COUNT,
    }
    assert receipt["validated_row_count"] == ROW_COUNT

    result = {
        "authority": {
            "authoritative": False,
            "authorizes": [],
            "manifest_schema_authoritative": False,
            "p2_status": "NOT_CLAIMED",
            "s0_status": "NOT_CLAIMED",
        },
        "candidate_files_canonical_and_bound": True,
        "candidate_manifest_bytes": len(manifest_bytes),
        "candidate_manifest_sha256": sha(manifest_bytes),
        "candidate_receipt_sha256": sha(receipt_bytes),
        "changed_uncapped_lookup_indices": changed,
        "flat_amplitude_one_plateaus": flat_plateaus,
        "flat_amplitude_one_rows": flat_rows,
        "inactive_b255_edges": b255_edges,
        "inactive_b255_rows": b255_rows,
        "maximum_nola_float64_deviation_before_f4": maximum_nola_float32_error,
        "minimum_positive_nola_denominator": minimum_denominator,
        "normative_roots": {
            name: {"bytes": value[0], "sha256": value[1]}
            for name, value in observed_roots.items()
        },
        "rows_independently_replayed": row_index,
        "runtime": runtime_expected,
        "unique_10_byte_keys": len(seen_keys),
        "worst_negative_margin": worst_negative_margin,
        "worst_positive_margin": worst_positive_margin,
    }
    sys.stdout.buffer.write(canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
