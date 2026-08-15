"""Generate the non-authorizing WARP-PHASE canonical fixture-pack candidate.

This executable is intentionally outside ``src``.  It may run only in the
dedicated CPython 3.12.13 / NumPy 2.4.6 fixture environment, consumes no real
data, and creates a new candidate directory exactly once.  A later fresh
review must freeze the candidate; this program never declares Gate 1 passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib import format as npy_format
from numpy.typing import NDArray

if __name__ != "__main__":
    raise RuntimeError("the fixture generator is an executable and must never be imported")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from pams.warp_phase.selector import (  # noqa: E402
    SelectorError,
    SupportSelection,
    harmonic_margin,
    make_weak_view,
    normalize_coco17,
    select_support_period,
)


class FixtureGenerationError(RuntimeError):
    """Raised when any canonical-generation precondition is not exact."""


Array = NDArray[np.generic]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

PYTHON_IMPLEMENTATION = "CPython"
PYTHON_VERSION = "3.12.13"
NUMPY_VERSION = "2.4.6"
SEED = 20270815
FIXTURE_COUNT = 1000
FRAME_COUNT = 128
JOINT_COUNT = 17
PCG64_WITNESS_SHA256 = "22cf6053f9a31cee979636aa1f1f629b890a4a88956d42c99d5bfb4333e3bcc0"
PCG64_WITNESS = (
    9521446988715047786,
    14249219314912930443,
    6690348893508556226,
    11556683178623268117,
    13450199692548364832,
    4748815881888473085,
    12717261861082187763,
    6192873454472717068,
    5919130353202294623,
    13700469945621656742,
    8320809674766470240,
    14314247923956437868,
    13599109478297638902,
    16225517010854040016,
    12798711397771314810,
    1237992528928537918,
)
ENVIRONMENT_LOCK_FIELDS = {
    "python_implementation",
    "python_version",
    "interpreter_sha256",
    "os",
    "architecture",
    "endianness",
    "numpy_version",
    "numpy_wheel",
    "numpy_wheel_sha256",
    "installer_version",
    "source_sha256",
    "detached_receipt_sha256",
}
CATEGORY_NAMES = (
    "fundamental",
    "three_segment_warp",
    "symmetric_half_harmonic_challenge",
    "symmetric_double_harmonic_challenge",
    "off_grid_off_bin",
    "pause",
    "reversal",
    "alias_boundary",
    "endpoint_maxima",
)
CATEGORY_COUNTS = (150, 100, 125, 125, 100, 100, 100, 100, 100)
BILATERAL_PAIRS = ((5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16))
ARRAY_SCHEMAS: dict[str, tuple[str, tuple[int, ...]]] = {
    "candidate_period.npy": ("<u2", (125,)),
    "category.npy": ("<u2", (1000,)),
    "expected_acf.npy": ("<f8", (1000, 125)),
    "expected_fft_mask.npy": ("|u1", (1000, 256, 34)),
    "expected_fft_power.npy": ("<f8", (1000, 127)),
    "expected_fft_x.npy": ("<f8", (1000, 256)),
    "expected_first64_selected_period.npy": ("<i2", (1000,)),
    "expected_harmonic_power.npy": ("<f8", (1000, 125, 3)),
    "expected_last64_selected_period.npy": ("<i2", (1000,)),
    "expected_local_max.npy": ("|u1", (1000, 125)),
    "expected_parent_accept.npy": ("|u1", (1000,)),
    "expected_reversal_selected_period.npy": ("<i2", (1000,)),
    "expected_score.npy": ("<f8", (1000, 125)),
    "expected_score_valid.npy": ("|u1", (1000, 125)),
    "expected_selected_period.npy": ("<i2", (1000,)),
    "expected_weak_selected_period.npy": ("<i2", (1000,)),
    "joint_mask.npy": ("|u1", (1000, 128, 17)),
    "pose.npy": ("<f8", (1000, 128, 17, 3)),
    "q.npy": ("<f8", (128,)),
    "semantic_period.npy": ("<f8", (1000,)),
    "weak_joint_mask.npy": ("|u1", (1000, 128, 17)),
    "weak_pose.npy": ("<f8", (1000, 128, 17, 3)),
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


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
        raise FixtureGenerationError("payload is not canonical-JSON serializable") from exc


def _validate_runtime() -> None:
    if platform.python_implementation() != PYTHON_IMPLEMENTATION:
        raise FixtureGenerationError(
            f"requires {PYTHON_IMPLEMENTATION}, got {platform.python_implementation()}"
        )
    if platform.python_version() != PYTHON_VERSION:
        raise FixtureGenerationError(
            f"requires CPython {PYTHON_VERSION}, got {platform.python_version()}"
        )
    if np.__version__ != NUMPY_VERSION:
        raise FixtureGenerationError(f"requires NumPy {NUMPY_VERSION}, got {np.__version__}")


def _pcg64_witness() -> str:
    witness_rng = np.random.Generator(np.random.PCG64(SEED))
    values = np.ascontiguousarray(witness_rng.bit_generator.random_raw(16), dtype="<u8")
    if tuple(int(value) for value in values) != PCG64_WITNESS:
        raise FixtureGenerationError("PCG64 random_raw values differ from the frozen witness")
    payload = values.tobytes(order="C")
    if len(payload) != 128:
        raise FixtureGenerationError("PCG64 witness must contain exactly 128 bytes")
    digest = _sha256_bytes(payload)
    if digest != PCG64_WITNESS_SHA256:
        raise FixtureGenerationError("PCG64 witness SHA-256 differs from the frozen contract")
    return digest


def _read_environment_lock(path: Path, source_sha256: str) -> tuple[dict[str, str], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FixtureGenerationError(f"cannot read fixture environment lock: {path}") from exc
    if raw.startswith(b"\xef\xbb\xbf") or raw.endswith(b"\n") or raw.endswith(b"\r"):
        raise FixtureGenerationError(
            "fixture environment lock must have no BOM or terminal newline"
        )
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixtureGenerationError(
            "fixture environment lock is not canonical UTF-8 JSON"
        ) from exc
    if not isinstance(parsed, dict) or set(parsed) != ENVIRONMENT_LOCK_FIELDS:
        raise FixtureGenerationError(
            "fixture environment lock fields differ from the frozen contract"
        )
    if raw != _canonical_json_bytes(parsed):
        raise FixtureGenerationError("fixture environment lock bytes are not canonical JSON")
    lock = {str(key): str(value) for key, value in parsed.items()}
    if any(not value.strip() for value in lock.values()):
        raise FixtureGenerationError("fixture environment lock contains an empty field")
    actual_values = {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "interpreter_sha256": _sha256_file(Path(sys.executable).resolve()),
        "os": platform.system(),
        "architecture": platform.machine(),
        "endianness": sys.byteorder,
        "numpy_version": np.__version__,
        "source_sha256": source_sha256,
    }
    for key, actual in actual_values.items():
        if lock[key] != actual:
            raise FixtureGenerationError(
                f"fixture environment lock field {key!r} is stale or false"
            )
    wheel_name = lock["numpy_wheel"]
    if (
        Path(wheel_name).name != wheel_name
        or not wheel_name.endswith(".whl")
        or not wheel_name.startswith("numpy-2.4.6-")
        or "-cp312-" not in wheel_name
    ):
        raise FixtureGenerationError("numpy_wheel must be one exact wheel filename")
    for key in (
        "interpreter_sha256",
        "numpy_wheel_sha256",
        "source_sha256",
        "detached_receipt_sha256",
    ):
        if not _is_sha256(lock[key]):
            raise FixtureGenerationError(
                f"fixture environment lock field {key!r} is not lowercase SHA-256"
            )
    return lock, _sha256_bytes(raw)


def _category_code(fixture_id: int) -> int:
    bounds = (150, 250, 375, 500, 600, 700, 800, 900, 1000)
    for code, stop in enumerate(bounds):
        if fixture_id < stop:
            return code
    raise FixtureGenerationError(f"fixture ID outside the frozen population: {fixture_id}")


def _three_segment_clock(rng: np.random.Generator, q: FloatArray) -> FloatArray:
    for _attempt in range(128):
        b1 = float(rng.uniform(0.20, 0.40))
        b2 = float(rng.uniform(0.60, 0.80))
        log_r = np.asarray(
            rng.uniform(math.log(0.5), math.log(1.5), size=(3,)),
            dtype=np.float64,
        )
        pause_u = float(rng.random())
        rates = np.exp(log_r)
        paused = pause_u < 0.20
        if paused:
            rates[1] = 0.0
        lengths = np.asarray((b1, b2 - b1, 1.0 - b2), dtype=np.float64)
        normalization = float(np.sum(lengths * rates, dtype=np.float64))
        if not math.isfinite(normalization) or normalization == 0.0:
            continue
        slopes = rates / normalization
        if np.any(~np.isfinite(slopes)):
            continue
        if paused:
            accepted = bool(
                slopes[1] == 0.0 and 0.4 <= slopes[0] <= 2.0 and 0.4 <= slopes[2] <= 2.0
            )
        else:
            accepted = bool(np.all((slopes >= 0.4) & (slopes <= 2.0)))
        if not accepted:
            continue
        unit_q = q / np.float64(127.0)
        mapped = np.empty_like(unit_q, dtype=np.float64)
        first = unit_q < b1
        second = (unit_q >= b1) & (unit_q < b2)
        third = unit_q >= b2
        mapped[first] = slopes[0] * unit_q[first]
        mapped[second] = slopes[0] * b1 + slopes[1] * (unit_q[second] - b1)
        mapped[third] = slopes[0] * b1 + slopes[1] * (b2 - b1) + slopes[2] * (unit_q[third] - b2)
        return np.ascontiguousarray(np.float64(127.0) * mapped, dtype="<f8")
    raise FixtureGenerationError("three-segment schedule exhausted 128 complete attempts")


def _category_parameters(
    fixture_id: int,
    code: int,
    rng: np.random.Generator,
    q: FloatArray,
) -> tuple[float, FloatArray]:
    if code == 0:
        period = float(rng.integers(12, 49))
        return period, q
    if code == 1:
        period = float(rng.integers(12, 49))
        return period, _three_segment_clock(rng, q)
    if code == 2:
        period = float(2 * int(rng.integers(8, 25)))
        return period, q
    if code == 3:
        period = float(rng.integers(8, 25))
        return period, q
    if code == 4:
        for _attempt in range(128):
            period = float(rng.uniform(12.25, 48.75))
            if abs(period - float(np.rint(period))) >= 0.20:
                return period, q
        raise FixtureGenerationError("off-grid period exhausted 128 scalar draws")
    if code == 5:
        period = float(rng.integers(12, 49))
        pause_start = int(rng.integers(32, 65))
        pause_length = int(rng.integers(8, 25))
        phase_clock = np.where(
            q < pause_start,
            q,
            np.where(q <= pause_start + pause_length, float(pause_start), q - pause_length),
        )
        return period, np.ascontiguousarray(phase_clock, dtype="<f8")
    if code == 6:
        period = float(rng.integers(12, 49))
        return period, np.ascontiguousarray(np.float64(127.0) - q, dtype="<f8")
    if code == 7:
        limits = (4.05, 4.45) if fixture_id < 850 else (62.55, 62.95)
        period = float(rng.uniform(*limits))
        return period, q
    if code == 8:
        period = 4.0 if fixture_id < 950 else 63.0
        return period, q
    raise FixtureGenerationError(f"unknown category code: {code}")


def _pose_template(
    code: int,
    q: FloatArray,
    phase_clock: FloatArray,
    period: float,
    phi: float,
    amplitude: float,
    sx: NDArray[np.int64],
    sy: NDArray[np.int64],
    h2: float,
    h3: float,
) -> FloatArray:
    joint_index = np.arange(JOINT_COUNT, dtype=np.int64)
    base_x = ((joint_index % 5 - 2) / np.float64(4.0)).astype(np.float64)
    base_y = ((joint_index // 5) / np.float64(4.0)).astype(np.float64)
    if code in {2, 3}:
        sx_used = sx.copy()
        sy_used = sy.copy()
        for left, right in BILATERAL_PAIRS:
            sx_used[right] = -sx_used[left]
            sy_used[right] = sy_used[left]
        if code == 2:
            theta = np.float64(2.0) * np.pi * q / period
            x_wave = np.float64(0.35) * np.sin(theta + phi) + np.sin(np.float64(2.0) * theta + phi)
            y_wave = np.float64(0.35) * np.cos(theta + phi) + np.cos(np.float64(2.0) * theta + phi)
        else:
            theta0 = np.pi * q / period
            x_wave = np.sin(theta0 + phi) + np.float64(0.35) * np.sin(
                np.float64(2.0) * theta0 + phi
            )
            y_wave = np.cos(theta0 + phi) + np.float64(0.35) * np.cos(
                np.float64(2.0) * theta0 + phi
            )
    else:
        sx_used = sx
        sy_used = sy
        theta = np.float64(2.0) * np.pi * phase_clock / period
        x_wave = (
            np.sin(theta + phi)
            + h2 * np.sin(np.float64(2.0) * theta + phi)
            + h3 * np.sin(np.float64(3.0) * theta + phi)
        )
        y_wave = (
            np.cos(theta + phi)
            + h2 * np.cos(np.float64(2.0) * theta + phi)
            + h3 * np.cos(np.float64(3.0) * theta + phi)
        )
    pose = np.empty((FRAME_COUNT, JOINT_COUNT, 3), dtype="<f8")
    pose[:, :, 0] = base_x[None, :] + amplitude * x_wave[:, None] * sx_used[None, :]
    pose[:, :, 1] = base_y[None, :] + amplitude * y_wave[:, None] * sy_used[None, :]
    pose[:, :, 2] = np.float64(0.9)
    return pose


def _canonical_mask(
    fixture_id: int,
    code: int,
    frame_u: FloatArray,
    joint_u: FloatArray,
) -> BoolArray:
    heavily_missing = fixture_id % 10 in {0, 1, 2, 3, 4}
    frame_threshold = 0.40 if heavily_missing else 0.10
    joint_threshold = 0.30 if heavily_missing else 0.10
    joint_valid = joint_u >= joint_threshold
    if code in {2, 3}:
        for left, right in BILATERAL_PAIRS:
            pair_valid = joint_u[:, left] >= joint_threshold
            joint_valid[:, left] = pair_valid
            joint_valid[:, right] = pair_valid
    joint_valid[frame_u < frame_threshold] = False
    return np.ascontiguousarray(joint_valid, dtype=np.bool_)


def _safe_selection(
    q: FloatArray,
    coordinates: FloatArray,
    joint_valid: BoolArray,
) -> SupportSelection | None:
    try:
        return select_support_period(q, coordinates, joint_valid)
    except SelectorError:
        return None


def _selected_period(selection: SupportSelection | None) -> int:
    return -1 if selection is None or selection.period is None else selection.period


def _within_ten_percent(value: int, reference: int) -> bool:
    return value != -1 and 10 * abs(value - reference) <= reference


def _fill_expected_selector_arrays(
    fixture_id: int,
    selection: SupportSelection,
    arrays: dict[str, Array],
) -> None:
    candidate_indices = selection.candidates - 4
    if np.any(candidate_indices < 0) or np.any(candidate_indices >= 125):
        raise FixtureGenerationError("selector returned candidates outside frozen 4..128 range")
    arrays["expected_fft_x.npy"][fixture_id] = selection.fft_grid
    arrays["expected_fft_mask.npy"][fixture_id] = selection.fft_mask.astype(np.uint8)
    arrays["expected_fft_power.npy"][fixture_id] = selection.fft_power
    finite_acf = np.isfinite(selection.acf)
    finite_score = np.isfinite(selection.score_by_period)
    valid_candidates = selection.score_valid & finite_acf & finite_score
    acf_values = np.where(valid_candidates, selection.acf, 0.0)
    score_values = np.where(valid_candidates, selection.score_by_period, 0.0)
    arrays["expected_acf.npy"][fixture_id, candidate_indices] = acf_values
    arrays["expected_score.npy"][fixture_id, candidate_indices] = score_values
    arrays["expected_score_valid.npy"][fixture_id, candidate_indices] = (
        selection.score_valid.astype(np.uint8)
    )
    arrays["expected_local_max.npy"][fixture_id, candidate_indices] = (
        selection.local_maximum.astype(np.uint8)
    )
    harmonic = selection.harmonic_power.copy()
    harmonic[~valid_candidates] = 0.0
    arrays["expected_harmonic_power.npy"][fixture_id, candidate_indices] = harmonic


def _empty_arrays() -> dict[str, Array]:
    arrays: dict[str, Array] = {}
    for name, (dtype, shape) in ARRAY_SCHEMAS.items():
        arrays[name] = np.zeros(shape, dtype=np.dtype(dtype), order="C")
    arrays["expected_selected_period.npy"].fill(-1)
    arrays["expected_weak_selected_period.npy"].fill(-1)
    arrays["expected_first64_selected_period.npy"].fill(-1)
    arrays["expected_last64_selected_period.npy"].fill(-1)
    arrays["expected_reversal_selected_period.npy"].fill(-1)
    arrays["q.npy"] = np.arange(FRAME_COUNT, dtype="<f8")
    arrays["candidate_period.npy"] = np.arange(4, 129, dtype="<u2")
    return arrays


def _generate_arrays() -> dict[str, Array]:
    arrays = _empty_arrays()
    q = np.asarray(arrays["q.npy"], dtype=np.float64)
    integer_q = np.arange(FRAME_COUNT, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(SEED))
    for fixture_id in range(FIXTURE_COUNT):
        code = _category_code(fixture_id)
        phi = float(rng.uniform(0.0, np.float64(2.0) * np.pi))
        amplitude = float(rng.uniform(0.08, 0.20))
        sx = 2 * rng.integers(0, 2, size=(JOINT_COUNT,), dtype=np.int64) - 1
        sy = 2 * rng.integers(0, 2, size=(JOINT_COUNT,), dtype=np.int64) - 1
        period, phase_clock = _category_parameters(fixture_id, code, rng, q)
        h2 = float(rng.uniform(0.0, 0.20))
        h3 = float(rng.uniform(0.0, 0.10))
        frame_u = np.asarray(rng.random(size=(FRAME_COUNT,)), dtype=np.float64)
        joint_u = np.asarray(rng.random(size=(FRAME_COUNT, JOINT_COUNT)), dtype=np.float64)
        weak_jitter = np.asarray(
            rng.normal(0.0, 0.01, size=(FRAME_COUNT, JOINT_COUNT, 2)),
            dtype=np.float64,
        )
        weak_dropout = np.asarray(
            rng.random(size=(FRAME_COUNT, JOINT_COUNT)),
            dtype=np.float64,
        )

        pose = _pose_template(code, q, phase_clock, period, phi, amplitude, sx, sy, h2, h3)
        canonical_mask = _canonical_mask(fixture_id, code, frame_u, joint_u)
        pose[~canonical_mask] = np.float64(0.0)
        weak_coordinates, weak_mask = make_weak_view(
            pose[:, :, :2],
            canonical_mask,
            weak_jitter,
            weak_dropout,
        )
        weak_pose = np.zeros_like(pose, dtype="<f8")
        weak_pose[:, :, :2] = weak_coordinates
        weak_pose[:, :, 2][weak_mask] = np.float64(0.9)

        arrays["pose.npy"][fixture_id] = pose
        arrays["joint_mask.npy"][fixture_id] = canonical_mask.astype(np.uint8)
        arrays["weak_pose.npy"][fixture_id] = weak_pose
        arrays["weak_joint_mask.npy"][fixture_id] = weak_mask.astype(np.uint8)
        arrays["category.npy"][fixture_id] = code
        arrays["semantic_period.npy"][fixture_id] = period

        try:
            canonical_track = normalize_coco17(pose, canonical_mask, integer_q)
        except SelectorError as exc:
            raise FixtureGenerationError(f"fixture {fixture_id} cannot be normalized") from exc
        canonical = _safe_selection(q, canonical_track.coordinates, canonical_track.joint_valid)
        if canonical is not None:
            _fill_expected_selector_arrays(fixture_id, canonical, arrays)

        try:
            weak_track = normalize_coco17(weak_pose, weak_mask, integer_q)
        except SelectorError:
            weak = None
        else:
            weak = _safe_selection(q, weak_track.coordinates, weak_track.joint_valid)
        first = _safe_selection(
            q[:64],
            canonical_track.coordinates[:64],
            canonical_track.joint_valid[:64],
        )
        last = _safe_selection(
            q[-64:],
            canonical_track.coordinates[-64:],
            canonical_track.joint_valid[-64:],
        )
        reverse_q = q[0] + (q[-1] - q[::-1])
        reversal = _safe_selection(
            reverse_q,
            canonical_track.coordinates[::-1],
            canonical_track.joint_valid[::-1],
        )
        canonical_period = _selected_period(canonical)
        weak_period = _selected_period(weak)
        first_period = _selected_period(first)
        last_period = _selected_period(last)
        reversal_period = _selected_period(reversal)
        margin = -math.inf
        if canonical is not None and canonical.period is not None:
            try:
                margin = harmonic_margin(canonical)
            except SelectorError:
                margin = -math.inf
        accepted = bool(
            canonical_period != -1
            and _within_ten_percent(weak_period, canonical_period)
            and _within_ten_percent(first_period, canonical_period)
            and _within_ten_percent(last_period, canonical_period)
            and _within_ten_percent(reversal_period, canonical_period)
            and margin >= 0.10
        )
        arrays["expected_selected_period.npy"][fixture_id] = canonical_period
        arrays["expected_weak_selected_period.npy"][fixture_id] = weak_period
        arrays["expected_first64_selected_period.npy"][fixture_id] = first_period
        arrays["expected_last64_selected_period.npy"][fixture_id] = last_period
        arrays["expected_reversal_selected_period.npy"][fixture_id] = reversal_period
        arrays["expected_parent_accept.npy"][fixture_id] = int(accepted)
    return arrays


def _validate_arrays(arrays: Mapping[str, Array]) -> None:
    if set(arrays) != set(ARRAY_SCHEMAS):
        raise FixtureGenerationError("generated NPY member set differs from the frozen contract")
    for name, (dtype, shape) in ARRAY_SCHEMAS.items():
        array = arrays[name]
        if array.shape != shape:
            raise FixtureGenerationError(f"{name} shape differs from {shape}")
        if array.dtype.str != np.dtype(dtype).str:
            raise FixtureGenerationError(f"{name} dtype differs from explicit {dtype}")
        if not array.flags.c_contiguous:
            raise FixtureGenerationError(f"{name} is not C-contiguous")
        if array.dtype.hasobject:
            raise FixtureGenerationError(f"{name} contains a forbidden object dtype")
    category_counts = np.bincount(
        np.asarray(arrays["category.npy"], dtype=np.int64),
        minlength=len(CATEGORY_COUNTS),
    )
    if tuple(int(value) for value in category_counts) != CATEGORY_COUNTS:
        raise FixtureGenerationError("category membership or counts changed")
    candidates = np.asarray(arrays["candidate_period.npy"], dtype=np.uint16)
    if not np.array_equal(candidates, np.arange(4, 129, dtype=np.uint16)):
        raise FixtureGenerationError("candidate_period is not exactly 4..128")
    for name in (
        "pose.npy",
        "weak_pose.npy",
        "expected_fft_x.npy",
        "expected_fft_power.npy",
        "expected_acf.npy",
        "expected_score.npy",
        "expected_harmonic_power.npy",
    ):
        if np.any(~np.isfinite(arrays[name])):
            raise FixtureGenerationError(f"{name} contains a non-finite expected value")
    score_valid = np.asarray(arrays["expected_score_valid.npy"], dtype=np.bool_)
    for name in ("expected_acf.npy", "expected_score.npy"):
        if np.any(arrays[name][~score_valid] != 0.0):
            raise FixtureGenerationError(f"{name} is not exact zero where invalid")
    harmonic = np.asarray(arrays["expected_harmonic_power.npy"], dtype=np.float64)
    if np.any(harmonic[~score_valid] != 0.0):
        raise FixtureGenerationError(
            "expected_harmonic_power.npy is not exact zero outside a valid candidate"
        )


def _exclusive_write(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise FixtureGenerationError(f"refusing to overwrite output: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise


def _write_npy_v2(path: Path, array: Array) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise FixtureGenerationError(f"refusing to overwrite output: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            npy_format.write_array(handle, array, version=(2, 0), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise


def _pack_hash(pack_directory: Path, names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        path = pack_directory / name
        if path.name != name or not path.is_file():
            raise FixtureGenerationError(f"pack member is missing or not a simple file: {name}")
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_output_path(output: Path) -> Path:
    resolved = output.expanduser().resolve()
    forbidden = {"features", "vault", "results"}
    if any(part.lower() in forbidden for part in resolved.parts):
        raise FixtureGenerationError("candidate output may not enter features, vault, or results")
    if resolved.exists():
        raise FixtureGenerationError(f"exclusive candidate output already exists: {resolved}")
    return resolved


def _write_candidate(
    output: Path,
    arrays: Mapping[str, Array],
    environment_lock: Mapping[str, str],
    environment_lock_sha256: str,
    generator_source_sha256: str,
    selector_source_sha256: str,
    witness_sha256: str,
) -> tuple[str, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    pack_directory = output / "pack"
    pack_directory.mkdir(exist_ok=False)
    metadata = {
        "array_schemas": {
            name: {"dtype": dtype, "shape": list(shape)}
            for name, (dtype, shape) in sorted(ARRAY_SCHEMAS.items())
        },
        "artifact_type": "warp_phase_fixture_pack_candidate",
        "authoritative": False,
        "category_counts": {
            str(code): CATEGORY_COUNTS[code] for code in range(len(CATEGORY_COUNTS))
        },
        "category_names": {str(code): CATEGORY_NAMES[code] for code in range(len(CATEGORY_NAMES))},
        "environment": {
            "architecture": environment_lock["architecture"],
            "endianness": environment_lock["endianness"],
            "numpy_version": environment_lock["numpy_version"],
            "os": environment_lock["os"],
            "python_implementation": environment_lock["python_implementation"],
            "python_version": environment_lock["python_version"],
        },
        "fixture_count": FIXTURE_COUNT,
        "fixture_environment_lock_sha256": environment_lock_sha256,
        "generator_source_sha256": generator_source_sha256,
        "npy_format_version": "2.0",
        "pcg64_witness_sha256": witness_sha256,
        "rng": "numpy.random.Generator(numpy.random.PCG64(20270815))",
        "selector_source_sha256": selector_source_sha256,
        "status": "CANDIDATE_PENDING_FRESH_REVIEW",
    }
    metadata_bytes = _canonical_json_bytes(metadata)
    _exclusive_write(pack_directory / "metadata.json", metadata_bytes)
    for name in sorted(arrays):
        _write_npy_v2(pack_directory / name, arrays[name])

    pack_names = ["metadata.json", *arrays.keys()]
    pack_sha256 = _pack_hash(pack_directory, pack_names)
    file_sha256 = {name: _sha256_file(pack_directory / name) for name in sorted(pack_names)}
    receipt = {
        "artifact_type": "warp_phase_fixture_pack_candidate_receipt",
        "authoritative": False,
        "authorizes": [],
        "category_counts": {
            str(code): CATEGORY_COUNTS[code] for code in range(len(CATEGORY_COUNTS))
        },
        "fixture_environment_lock_sha256": environment_lock_sha256,
        "freeze_action_performed": False,
        "fresh_review_required": True,
        "generator_source_sha256": generator_source_sha256,
        "member_sha256": file_sha256,
        "pack_hash_algorithm": "sha256(sorted(filename_utf8 || 0x00 || file_bytes || 0x0a))",
        "pack_member_order": sorted(pack_names),
        "pack_sha256": pack_sha256,
        "pcg64_witness_sha256": witness_sha256,
        "selector_source_sha256": selector_source_sha256,
        "status": "CANDIDATE_NON_AUTHORIZING",
    }
    receipt_bytes = _canonical_json_bytes(receipt)
    _exclusive_write(output / "candidate_receipt.json", receipt_bytes)
    return pack_sha256, _sha256_bytes(receipt_bytes)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--environment-lock",
        required=True,
        type=Path,
        help="Frozen canonical fixture_environment.lock.json",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New candidate directory; it must not already exist",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    _validate_runtime()
    witness_sha256 = _pcg64_witness()
    source_path = Path(__file__).resolve()
    generator_source_sha256 = _sha256_file(source_path)
    selector_source_path = REPOSITORY_ROOT / "src" / "pams" / "warp_phase" / "selector.py"
    selector_source_sha256 = _sha256_file(selector_source_path)
    environment_lock, environment_lock_sha256 = _read_environment_lock(
        Path(args.environment_lock),
        generator_source_sha256,
    )
    output = _validate_output_path(Path(args.output))
    arrays = _generate_arrays()
    _validate_arrays(arrays)
    if _sha256_file(source_path) != generator_source_sha256:
        raise FixtureGenerationError("generator source changed during candidate computation")
    if _sha256_file(selector_source_path) != selector_source_sha256:
        raise FixtureGenerationError("selector source changed during candidate computation")
    pack_sha256, receipt_sha256 = _write_candidate(
        output,
        arrays,
        environment_lock,
        environment_lock_sha256,
        generator_source_sha256,
        selector_source_sha256,
        witness_sha256,
    )
    summary = {
        "candidate_directory": str(output),
        "candidate_receipt_sha256": receipt_sha256,
        "freeze_action_performed": False,
        "pack_sha256": pack_sha256,
        "status": "CANDIDATE_NON_AUTHORIZING",
    }
    sys.stdout.buffer.write(_canonical_json_bytes(summary) + b"\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FixtureGenerationError, OSError) as exc:
        print(f"fixture generation failed closed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
