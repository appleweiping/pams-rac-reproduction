"""Generate the non-authorizing WARP-PHASE selector unit-fixture candidate.

This executable implements accepted Amendment 001.  It uses no RNG and may
run only in the dedicated CPython 3.12.13 / NumPy 2.4.6 fixture environment.
It creates the exact separate Gate-1 unit-fixture root once and never freezes,
promotes, or authorizes the candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as npy_format
from numpy.typing import NDArray

if __name__ != "__main__":
    raise RuntimeError("the selector unit-fixture generator is an executable")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pams.warp_phase.selector import (  # noqa: E402
    SelectorError,
    SupportSelection,
    normalize_coco17,
    select_support_period,
)


class UnitFixtureGenerationError(RuntimeError):
    """Raised when the candidate cannot be generated exactly and exclusively."""


Array = NDArray[np.generic]
FloatArray = NDArray[np.float64]

PYTHON_IMPLEMENTATION = "CPython"
PYTHON_VERSION = "3.12.13"
NUMPY_VERSION = "2.4.6"
ARTIFACT_RELATIVE_ROOT = Path(
    "data/warp_phase_pilot_v1/audit/gate1/selector-unit-fixtures-v1"
)
RECEIPT_FILENAME = "selector-unit-fixtures-v1.receipt.json"
PACK_HASH_ALGORITHM = "sha256(sorted(filename_utf8 || 0x00 || file_bytes || 0x0a))"

AMENDMENT_SHA256 = {
    "refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001.json": (
        "d5227f04af4452c2224d3a6b0a8342a7a02aa18edf6b5452e9e06290636bc1c9"
    ),
    "refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001.md": (
        "25fe82bbcad386b42e4c5a95a762d1f84e4c926796655f02ba5d2ab8ae7dc484"
    ),
}
ACCEPTANCE_REVIEW_SHA256 = {
    "refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001_REVIEW.json": (
        "8937446bbb9708e205f850e367c48fce70d42de9054c1066522865104352c8f6"
    ),
    "refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001_REVIEW.md": (
        "a1fcd938fcb849cc820275ffb54d4b233314accbd7d1a21aa2ce3742e2dafa9c"
    ),
}
CANONICAL_DOCUMENT_SHA256 = {
    "refine-logs/EXPERIMENT_EXECUTION_CONTRACT.md": (
        "70f80912e4b9693fae9e859eb12f90cc754b7d523b59af766879518770d69b2e"
    ),
    "refine-logs/EXPERIMENT_PLAN.md": (
        "020626027a73939f3fd4dfa479e6592699d19f652da592682b90d9235724ce5c"
    ),
    "refine-logs/FINAL_PROPOSAL.md": (
        "e8ff129b3a67ac07be0e5ff1af09555bb26f000e7cde9315b13493733cda6418"
    ),
    "refine-logs/round-4-refinement.md": (
        "2031d5c84b8c6c329dd0e43e5b8a09210ed9a3e8b17b47578f2f878a4c2b4e0f"
    ),
}
SOURCE_PATHS = (
    "scripts/experiments/generate_warp_phase_fixture_pack.py",
    "scripts/experiments/generate_warp_phase_selector_unit_fixtures.py",
    "src/pams/warp_phase/gates.py",
    "src/pams/warp_phase/selector.py",
    "tests/test_warp_phase_selector.py",
)
UNCHANGED_SOURCE_SHA256 = {
    "scripts/experiments/generate_warp_phase_fixture_pack.py": (
        "7af1dac0ce246979d608b3db8e9fcaef9fe18dbfe2a5ecc803616e30919bb4ce"
    ),
    "src/pams/warp_phase/selector.py": (
        "73b31b7d6c58a67efa8542df754971a58926d783258c9776269ec06765dafa0e"
    ),
}


@dataclass(frozen=True)
class FixtureSpec:
    fixture_id: str
    clock_count: int
    source_period: int
    span: int
    p_max: int
    selected_period: int


FIXTURE_SPECS = (
    FixtureSpec("offbin_delta1_p20", 256, 20, 255, 127, 20),
    FixtureSpec("lower_endpoint_p4", 64, 4, 63, 31, 4),
    FixtureSpec("upper_endpoint_span126_p63", 127, 63, 126, 63, 63),
)

ARRAY_SUFFIX_SCHEMAS: dict[str, tuple[str, tuple[int | str, ...]]] = {
    "input_clocks.npy": ("<i8", ("N",)),
    "input_pose.npy": ("<f8", ("N", 17, 3)),
    "input_joint_mask.npy": ("|u1", ("N", 17)),
    "expected_scale.npy": ("<f8", (1,)),
    "expected_normalized_coordinates.npy": ("<f8", ("N", 17, 2)),
    "expected_normalized_joint_mask.npy": ("|u1", ("N", 17)),
    "expected_feature_frame_valid.npy": ("|u1", ("N",)),
    "expected_velocity.npy": ("<f8", ("N-1", 34)),
    "expected_velocity_mask.npy": ("|u1", ("N-1", 34)),
    "expected_fft_x.npy": ("<f8", (256,)),
    "expected_fft_mask.npy": ("|u1", (256, 34)),
    "expected_fft_power.npy": ("<f8", (127,)),
    "expected_candidate_present.npy": ("|u1", (125,)),
    "expected_acf.npy": ("<f8", (125,)),
    "expected_acf_computed.npy": ("|u1", (125,)),
    "expected_harmonic_power.npy": ("<f8", (125, 3)),
    "expected_score.npy": ("<f8", (125,)),
    "expected_score_computed.npy": ("|u1", (125,)),
    "expected_score_valid.npy": ("|u1", (125,)),
    "expected_local_max.npy": ("|u1", (125,)),
    "expected_selected_period.npy": ("<i2", (1,)),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    try:
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise UnitFixtureGenerationError("payload is not canonical-JSON serializable") from exc


def _validate_runtime() -> None:
    implementation = platform.python_implementation()
    version = platform.python_version()
    if implementation != PYTHON_IMPLEMENTATION or version != PYTHON_VERSION:
        raise UnitFixtureGenerationError(
            f"requires {PYTHON_IMPLEMENTATION} {PYTHON_VERSION}, got {implementation} {version}"
        )
    if np.__version__ != NUMPY_VERSION:
        raise UnitFixtureGenerationError(
            f"requires NumPy {NUMPY_VERSION}, got {np.__version__}"
        )


def _verify_frozen_inputs() -> None:
    expected = {
        **AMENDMENT_SHA256,
        **ACCEPTANCE_REVIEW_SHA256,
        **CANONICAL_DOCUMENT_SHA256,
        **UNCHANGED_SOURCE_SHA256,
    }
    for relative, frozen_hash in expected.items():
        observed = _sha256_file(REPOSITORY_ROOT / relative)
        if observed != frozen_hash:
            raise UnitFixtureGenerationError(
                f"frozen input SHA-256 differs for {relative}: {observed}"
            )


def _source_hashes() -> dict[str, str]:
    return {
        relative: _sha256_file(REPOSITORY_ROOT / relative)
        for relative in sorted(SOURCE_PATHS)
    }


def _constructor(spec: FixtureSpec) -> tuple[NDArray[np.int64], FloatArray, NDArray[np.uint8]]:
    clocks = np.arange(spec.clock_count, dtype="<i8")
    pose = np.zeros((spec.clock_count, 17, 3), dtype="<f8", order="C")
    joint_mask = np.ones((spec.clock_count, 17), dtype="|u1", order="C")
    pose[:, :, 2] = np.float64(0.9)
    pose[:, 11, :2] = (np.float64(-0.5), np.float64(0.0))
    pose[:, 12, :2] = (np.float64(0.5), np.float64(0.0))
    pose[:, 5, :2] = (np.float64(-0.5), np.float64(2.0))
    pose[:, 6, :2] = (np.float64(0.5), np.float64(2.0))
    anchors = {5, 6, 11, 12}
    for clock_index in range(spec.clock_count):
        for joint in range(17):
            if joint in anchors:
                continue
            base_x = np.float64(joint % 5 - 2) / np.float64(4.0)
            base_y = np.float64(joint // 5) / np.float64(4.0)
            theta = (
                np.float64(2.0)
                * np.pi
                * np.float64(clocks[clock_index])
                / np.float64(spec.source_period)
                + np.float64(joint) * np.float64(0.1)
            )
            pose[clock_index, joint, 0] = base_x + np.float64(0.2) * np.sin(theta)
            pose[clock_index, joint, 1] = base_y + np.float64(0.2) * np.cos(theta)
    return clocks, pose, joint_mask


def _serialize_diagnostics(
    selection: SupportSelection,
) -> tuple[dict[str, Array], str, str | None]:
    diagnostics = selection.diagnostics
    candidate_indices = np.asarray(diagnostics.candidates - 4, dtype=np.int64)
    if np.any(candidate_indices < 0) or np.any(candidate_indices >= 125):
        raise UnitFixtureGenerationError("selector candidates are outside periods 4 through 128")
    candidate_present = np.zeros(125, dtype="|u1")
    acf = np.zeros(125, dtype="<f8")
    acf_computed = np.zeros(125, dtype="|u1")
    harmonic_power = np.zeros((125, 3), dtype="<f8")
    score = np.zeros(125, dtype="<f8")
    score_computed = np.zeros(125, dtype="|u1")
    score_valid = np.zeros(125, dtype="|u1")
    local_maximum = np.zeros(125, dtype="|u1")

    candidate_present[candidate_indices] = np.uint8(1)
    finite_acf = np.isfinite(diagnostics.acf)
    finite_score = np.isfinite(diagnostics.score_by_period)
    acf[candidate_indices] = np.where(finite_acf, diagnostics.acf, 0.0)
    acf_computed[candidate_indices] = finite_acf.astype(np.uint8)
    score[candidate_indices] = np.where(finite_score, diagnostics.score_by_period, 0.0)
    score_computed[candidate_indices] = finite_score.astype(np.uint8)
    score_valid[candidate_indices] = diagnostics.score_valid.astype(np.uint8)
    local_maximum[candidate_indices] = diagnostics.local_maximum.astype(np.uint8)
    harmonic = np.asarray(diagnostics.harmonic_power, dtype=np.float64).copy()
    harmonic[~finite_score] = np.float64(0.0)
    harmonic_power[candidate_indices] = harmonic
    selected = -1 if selection.period is None else selection.period
    stage = "decision_failed" if selection.period is None else "complete"
    reason = "no_local_maximum" if selection.period is None else None
    return (
        {
            "expected_fft_x.npy": np.ascontiguousarray(diagnostics.fft_grid, dtype="<f8"),
            "expected_fft_mask.npy": np.ascontiguousarray(
                diagnostics.fft_mask,
                dtype="|u1",
            ),
            "expected_fft_power.npy": np.ascontiguousarray(
                diagnostics.fft_power,
                dtype="<f8",
            ),
            "expected_candidate_present.npy": candidate_present,
            "expected_acf.npy": acf,
            "expected_acf_computed.npy": acf_computed,
            "expected_harmonic_power.npy": harmonic_power,
            "expected_score.npy": score,
            "expected_score_computed.npy": score_computed,
            "expected_score_valid.npy": score_valid,
            "expected_local_max.npy": local_maximum,
            "expected_selected_period.npy": np.asarray((selected,), dtype="<i2"),
        },
        stage,
        reason,
    )


def _generate_fixture(spec: FixtureSpec) -> tuple[dict[str, Array], dict[str, Any]]:
    clocks, pose, joint_mask = _constructor(spec)
    try:
        normalized = normalize_coco17(pose, joint_mask, clocks)
        selection = select_support_period(
            normalized.clocks,
            normalized.coordinates,
            normalized.joint_valid,
        )
    except SelectorError as exc:
        raise UnitFixtureGenerationError(
            f"{spec.fixture_id} failed before diagnostics: {exc}"
        ) from exc
    delta = np.diff(clocks.astype(np.float64))
    velocity = np.diff(normalized.coordinates.reshape(spec.clock_count, 34), axis=0)
    velocity = np.ascontiguousarray(velocity / delta[:, None], dtype="<f8")
    diagnostic_arrays, stage, reason = _serialize_diagnostics(selection)
    arrays: dict[str, Array] = {
        "input_clocks.npy": clocks,
        "input_pose.npy": pose,
        "input_joint_mask.npy": joint_mask,
        "expected_scale.npy": np.asarray((normalized.scale,), dtype="<f8"),
        "expected_normalized_coordinates.npy": np.ascontiguousarray(
            normalized.coordinates,
            dtype="<f8",
        ),
        "expected_normalized_joint_mask.npy": np.ascontiguousarray(
            normalized.joint_valid,
            dtype="|u1",
        ),
        "expected_feature_frame_valid.npy": np.ascontiguousarray(
            normalized.feature_frame_valid,
            dtype="|u1",
        ),
        "expected_velocity.npy": velocity,
        "expected_velocity_mask.npy": np.ones((spec.clock_count - 1, 34), dtype="|u1"),
        **diagnostic_arrays,
    }
    return arrays, {"failure_reason": reason, "stage": stage}


def _resolved_shape(shape: tuple[int | str, ...], clock_count: int) -> tuple[int, ...]:
    substitutions = {"N": clock_count, "N-1": clock_count - 1}
    return tuple(substitutions[value] if isinstance(value, str) else value for value in shape)


def _expanded_schemas() -> dict[str, tuple[str, tuple[int, ...]]]:
    schemas: dict[str, tuple[str, tuple[int, ...]]] = {
        "candidate_period.npy": ("<u2", (125,))
    }
    for spec in FIXTURE_SPECS:
        for suffix, (dtype, shape) in ARRAY_SUFFIX_SCHEMAS.items():
            schemas[f"{spec.fixture_id}.{suffix}"] = (
                dtype,
                _resolved_shape(shape, spec.clock_count),
            )
    return schemas


def _validate_fixture(spec: FixtureSpec, arrays: Mapping[str, Array]) -> None:
    expected_suffixes = set(ARRAY_SUFFIX_SCHEMAS)
    if set(arrays) != expected_suffixes:
        raise UnitFixtureGenerationError(f"{spec.fixture_id} member set differs")
    for suffix, (dtype, shape) in ARRAY_SUFFIX_SCHEMAS.items():
        array = np.asarray(arrays[suffix])
        expected_shape = _resolved_shape(shape, spec.clock_count)
        if array.dtype.str != dtype or array.shape != expected_shape:
            raise UnitFixtureGenerationError(
                f"{spec.fixture_id}.{suffix} has {array.dtype.str}/{array.shape}, "
                f"expected {dtype}/{expected_shape}"
            )
        if array.dtype.hasobject or not array.flags.c_contiguous:
            raise UnitFixtureGenerationError(
                f"{spec.fixture_id}.{suffix} is object-typed or non-C-order"
            )
        if array.dtype.kind == "f" and not np.isfinite(array).all():
            raise UnitFixtureGenerationError(f"{spec.fixture_id}.{suffix} is non-finite")

    clocks = arrays["input_clocks.npy"]
    pose = np.asarray(arrays["input_pose.npy"], dtype=np.float64)
    if not np.array_equal(clocks, np.arange(spec.clock_count, dtype="<i8")):
        raise UnitFixtureGenerationError(f"{spec.fixture_id} clocks differ")
    if not np.all(arrays["input_joint_mask.npy"] == 1):
        raise UnitFixtureGenerationError(f"{spec.fixture_id} input mask is not all-valid")
    if not np.all(pose[:, :, 2] == np.float64(0.9)):
        raise UnitFixtureGenerationError(f"{spec.fixture_id} confidence differs")
    if float(arrays["expected_scale.npy"][0]) != 2.0:
        raise UnitFixtureGenerationError(f"{spec.fixture_id} scale is not exact 2.0")
    if not np.array_equal(
        arrays["expected_normalized_coordinates.npy"],
        pose[:, :, :2] / np.float64(2.0),
    ):
        raise UnitFixtureGenerationError(f"{spec.fixture_id} normalization is not exact")
    for suffix in (
        "expected_normalized_joint_mask.npy",
        "expected_feature_frame_valid.npy",
        "expected_velocity_mask.npy",
        "expected_fft_mask.npy",
    ):
        if not np.all(arrays[suffix] == 1):
            raise UnitFixtureGenerationError(f"{spec.fixture_id}.{suffix} is not all-valid")
    expected_velocity = np.diff(
        arrays["expected_normalized_coordinates.npy"].reshape(spec.clock_count, 34),
        axis=0,
    )
    if not np.array_equal(arrays["expected_velocity.npy"], expected_velocity):
        raise UnitFixtureGenerationError(f"{spec.fixture_id} velocity differs")
    present = arrays["expected_candidate_present.npy"].astype(np.bool_)
    expected_present = np.zeros(125, dtype=np.bool_)
    expected_present[: spec.p_max - 3] = True
    if not np.array_equal(present, expected_present):
        raise UnitFixtureGenerationError(f"{spec.fixture_id} candidate support differs")
    if int(arrays["expected_selected_period.npy"][0]) != spec.selected_period:
        raise UnitFixtureGenerationError(f"{spec.fixture_id} selected period differs")
    for suffix in (
        "input_clocks.npy",
        "expected_fft_power.npy",
        "expected_candidate_present.npy",
        "expected_acf.npy",
        "expected_acf_computed.npy",
        "expected_harmonic_power.npy",
        "expected_score.npy",
        "expected_score_computed.npy",
        "expected_score_valid.npy",
        "expected_local_max.npy",
    ):
        if np.any(arrays[suffix] == -1):
            raise UnitFixtureGenerationError(
                f"{spec.fixture_id}.{suffix} contains the decision-only -1 sentinel"
            )
    if np.any(arrays["expected_acf.npy"][~arrays["expected_acf_computed.npy"].astype(bool)] != 0.0):
        raise UnitFixtureGenerationError(f"{spec.fixture_id} ACF fill is not exact +0.0")
    score_computed = arrays["expected_score_computed.npy"].astype(bool)
    if np.any(arrays["expected_score.npy"][~score_computed] != 0.0):
        raise UnitFixtureGenerationError(f"{spec.fixture_id} score fill is not exact +0.0")
    if np.any(arrays["expected_harmonic_power.npy"][~score_computed] != 0.0):
        raise UnitFixtureGenerationError(f"{spec.fixture_id} harmonic fill is not exact +0.0")

    fft_x = arrays["expected_fft_x.npy"]
    delta = float((fft_x[-1] - fft_x[0]) / np.float64(255.0))
    if spec.fixture_id == "offbin_delta1_p20":
        exact_grid = np.arange(256, dtype="<f8")
        if fft_x.tobytes(order="C") != exact_grid.tobytes(order="C") or delta != 1.0:
            raise UnitFixtureGenerationError("off-bin FFT grid is not byte-exact Delta=1")
        continuous_bin = np.float64(256.0) * np.float64(delta) / np.float64(20.0)
        if float(continuous_bin) != 12.8:
            raise UnitFixtureGenerationError("off-bin k(20) is not exact float64 12.8")
        power = arrays["expected_fft_power.npy"]
        interpolated = (
            (np.float64(13.0) - continuous_bin) * power[11]
            + (continuous_bin - np.float64(12.0)) * power[12]
        )
        if arrays["expected_harmonic_power.npy"][16, 0] != interpolated:
            raise UnitFixtureGenerationError("off-bin interpolation does not use bins 12 and 13")
    score = arrays["expected_score.npy"]
    local = arrays["expected_local_max.npy"].astype(bool)
    if spec.fixture_id == "lower_endpoint_p4" and not (
        score[0] >= score[1] and local[0]
    ):
        raise UnitFixtureGenerationError("lower endpoint is not a one-sided maximum")
    if spec.fixture_id == "upper_endpoint_span126_p63" and not (
        score[59] >= score[58] and local[59]
    ):
        raise UnitFixtureGenerationError("upper endpoint is not a one-sided maximum")


def _metadata(
    schemas: Mapping[str, tuple[str, tuple[int, ...]]],
    diagnostics: Mapping[str, Mapping[str, Any]],
    source_sha256: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "acceptance_review_sha256": ACCEPTANCE_REVIEW_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "array_schemas": {
            name: {"dtype": dtype, "shape": list(shape)}
            for name, (dtype, shape) in sorted(schemas.items())
        },
        "artifact_root": ARTIFACT_RELATIVE_ROOT.as_posix(),
        "artifact_type": "warp_phase_selector_unit_fixture_pack_candidate",
        "authoritative": False,
        "authorizes": [],
        "candidate_only": True,
        "canonical_document_sha256": CANONICAL_DOCUMENT_SHA256,
        "constructor": {
            "all_confidence": 0.9,
            "all_joint_masks_valid": True,
            "anchors_overwritten": False,
            "base_x": "float64(j%5-2)/float64(4.0)",
            "base_y": "float64(floor(j/5))/float64(4.0)",
            "dynamic_joint_amplitude": 0.2,
            "dynamic_joint_phase_offset": "float64(j)*float64(0.1)",
            "hips": {"11": [-0.5, 0.0], "12": [0.5, 0.0]},
            "iteration_order": "increasing_clock_then_joint",
            "label_or_evaluator_data_used": False,
            "normalized_root": [0.0, 0.0],
            "normalized_scale": 2.0,
            "shoulders": {"5": [-0.5, 2.0], "6": [0.5, 2.0]},
            "theta": (
                "float64(2.0)*pi*float64(q_i)/float64(P_src)"
                "+float64(j)*float64(0.1)"
            ),
        },
        "decision_failure_sentinel": -1,
        "diagnostic_fill": 0.0,
        "diagnostic_status": {name: diagnostics[name] for name in sorted(diagnostics)},
        "environment": {
            "architecture": platform.machine(),
            "endianness": sys.byteorder,
            "numpy_version": np.__version__,
            "os": platform.system(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "fixture_table": {
            spec.fixture_id: {
                "clock_count": spec.clock_count,
                "clocks": f"int64 0..{spec.clock_count - 1}",
                "fft_delta_exact": {
                    "offbin_delta1_p20": "1",
                    "lower_endpoint_p4": "21/85",
                    "upper_endpoint_span126_p63": "42/85",
                }[spec.fixture_id],
                "k_at_source_period_exact": {
                    "offbin_delta1_p20": "64/5",
                    "lower_endpoint_p4": "1344/85",
                    "upper_endpoint_span126_p63": "512/255",
                }[spec.fixture_id],
                "p_max": spec.p_max,
                "required_selected_period": spec.selected_period,
                "required_interface_assertion": {
                    "offbin_delta1_p20": "adjacent FFT bins 12 and 13",
                    "lower_endpoint_p4": "one-sided S(4)>=S(5)",
                    "upper_endpoint_span126_p63": "one-sided S(63)>=S(62)",
                }[spec.fixture_id],
                "source_period": spec.source_period,
                "span": spec.span,
            }
            for spec in FIXTURE_SPECS
        },
        "npy_format_version": "2.0",
        "pack_member_count_excluding_receipt": 65,
        "rng_calls": 0,
        "rng_used": False,
        "role": "deterministic_non_efficacy_gate1_selector_unit_tests",
        "root_member_count_including_receipt": 66,
        "source_sha256": dict(source_sha256),
        "status": "CANDIDATE_PENDING_FRESH_REVIEW",
    }


def _exclusive_write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise UnitFixtureGenerationError(f"refusing to overwrite output: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_npy_v2(path: Path, array: Array) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise UnitFixtureGenerationError(f"refusing to overwrite output: {path}") from exc
    with os.fdopen(descriptor, "wb") as handle:
        npy_format.write_array(handle, array, version=(2, 0), allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())


def _pack_hash(root: Path, names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        path = root / name
        if path.name != name or not path.is_file():
            raise UnitFixtureGenerationError(f"pack member is missing: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_output(output: Path) -> Path:
    expected = (REPOSITORY_ROOT / ARTIFACT_RELATIVE_ROOT).resolve(strict=False)
    resolved = output.resolve(strict=False)
    if resolved != expected:
        raise UnitFixtureGenerationError(f"output must be exactly {expected}")
    if resolved.exists():
        raise UnitFixtureGenerationError(f"exclusive candidate root already exists: {resolved}")
    return resolved


def _write_candidate(output: Path) -> tuple[str, str]:
    schemas = _expanded_schemas()
    all_arrays: dict[str, Array] = {
        "candidate_period.npy": np.arange(4, 129, dtype="<u2")
    }
    diagnostics: dict[str, dict[str, Any]] = {}
    for spec in FIXTURE_SPECS:
        fixture_arrays, diagnostic = _generate_fixture(spec)
        _validate_fixture(spec, fixture_arrays)
        diagnostics[spec.fixture_id] = diagnostic
        for suffix, array in fixture_arrays.items():
            all_arrays[f"{spec.fixture_id}.{suffix}"] = array
    if set(all_arrays) != set(schemas):
        raise UnitFixtureGenerationError("complete NPY member set differs from the schema")
    if np.any(all_arrays["candidate_period.npy"] == -1):
        raise UnitFixtureGenerationError("candidate periods contain the decision-only -1 sentinel")

    source_sha256 = _source_hashes()
    metadata_bytes = _canonical_json_bytes(_metadata(schemas, diagnostics, source_sha256))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    _exclusive_write(output / "metadata.json", metadata_bytes)
    for name in sorted(all_arrays):
        _write_npy_v2(output / name, all_arrays[name])

    member_order = sorted(("metadata.json", *all_arrays))
    member_sha256 = {name: _sha256_file(output / name) for name in member_order}
    pack_sha256 = _pack_hash(output, member_order)
    receipt = {
        "acceptance_review_sha256": ACCEPTANCE_REVIEW_SHA256,
        "amendment_sha256": AMENDMENT_SHA256,
        "artifact_type": "warp_phase_selector_unit_fixture_pack_candidate_receipt",
        "authoritative": False,
        "authorizes": [],
        "candidate_only": True,
        "canonical_document_sha256": CANONICAL_DOCUMENT_SHA256,
        "freeze_action_performed": False,
        "fresh_review_required": True,
        "member_sha256": member_sha256,
        "pack_hash_algorithm": PACK_HASH_ALGORITHM,
        "pack_member_order": member_order,
        "pack_sha256": pack_sha256,
        "schema_version": 1,
        "source_sha256": source_sha256,
        "status": "CANDIDATE_NON_AUTHORIZING",
    }
    receipt_bytes = _canonical_json_bytes(receipt)
    _exclusive_write(output / RECEIPT_FILENAME, receipt_bytes)
    return pack_sha256, _sha256_bytes(receipt_bytes)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / ARTIFACT_RELATIVE_ROOT,
        help="Exact Amendment-001 selector unit-fixture root",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _validate_runtime()
    _verify_frozen_inputs()
    output = _validate_output(Path(args.output))
    pack_sha256, receipt_sha256 = _write_candidate(output)
    summary = {
        "artifact_root": str(output),
        "authoritative": False,
        "authorizes": [],
        "candidate_receipt_sha256": receipt_sha256,
        "pack_sha256": pack_sha256,
        "status": "CANDIDATE_NON_AUTHORIZING",
    }
    sys.stdout.buffer.write(_canonical_json_bytes(summary) + b"\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, UnitFixtureGenerationError) as exc:
        print(f"selector unit-fixture generation failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
