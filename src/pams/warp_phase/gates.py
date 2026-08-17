"""Fail-closed deterministic gates for the WARP-PHASE pilot.

Gate functions return auditable receipts; they never silently authorize the
next stage.  The caller must persist the receipt, bind its hash, and verify the
declared dependencies before launching a subsequent process.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import stat
import sys
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.lib import format as npy_format
from numpy.typing import NDArray


class GateError(RuntimeError):
    """Raised whenever a frozen gate precondition is not exactly satisfied."""


SCHEMA_FIXTURE = (
    '{"annotation":{"height":"int","length":"int>1","object_schema":'
    '{"bbox":"<f8[L,4] finite","count":"int K","period":"int[K,2] with '
    '0<=s<e<=L","periodicity":"empty list"},"width":"int"},"evaluator_vault":'
    '{"P_eval":"numpy.median(asarray(e-s,dtype=float64))","integrity":"K>0; '
    'Python int endpoints; 0<=s<e<=L","interval_semantics":"[s,e) source-frame '
    'boundary units"},"feature_shard":{"frame_mask":"bool[P,320]",'
    '"local_person_slot":"int routing only","motion":"<f4[P,320,17,3]",'
    '"opaque_sample_key":"non-semantic digest routing only","person_mask":'
    '"bool[P]","sampled_frame_indices":"<i8[320]","source_length":"Python int"}}'
)
SCHEMA_FIXTURE_SHA256 = "31f81549f9c5dbedc6ddb278e868793246826a234de90c06c169c71387428275"
PCG64_WITNESS_SHA256 = "22cf6053f9a31cee979636aa1f1f629b890a4a88956d42c99d5bfb4333e3bcc0"
K4_VECTOR_SHA256 = "5e2f35a11bd8a376e44d5ad7d3e5067a0bb5a03e2d61edd84f5f5bfd6cf351ec"
TRAINING_PYTHON_VERSION = "3.12.13"
TRAINING_NUMPY_VERSION = "1.26.4"
TRAINING_PYPROJECT_SHA256 = (
    "14b3b35bbfa57fc982e24ef2f999df288893c0f0d12a53a9f82120babefead38"
)
TRAINING_UV_LOCK_SHA256 = (
    "5a964f1cd841948eae09ef253098b0b16eb5d2e6bd660e2d3496b389c4dc72a7"
)
SELECTOR_UNIT_FIXTURE_RELATIVE_ROOT = Path(
    "data/warp_phase_pilot_v1/audit/gate1/selector-unit-fixtures-v1"
)
SELECTOR_UNIT_FIXTURE_RECEIPT = "selector-unit-fixtures-v1.receipt.json"
SELECTOR_UNIT_PACK_HASH_ALGORITHM = (
    "sha256(sorted(filename_utf8 || 0x00 || file_bytes || 0x0a))"
)
SELECTOR_UNIT_AMENDMENT_SHA256 = {
    "refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001.json": (
        "d5227f04af4452c2224d3a6b0a8342a7a02aa18edf6b5452e9e06290636bc1c9"
    ),
    "refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001.md": (
        "25fe82bbcad386b42e4c5a95a762d1f84e4c926796655f02ba5d2ab8ae7dc484"
    ),
}
SELECTOR_UNIT_ACCEPTANCE_REVIEW_SHA256 = {
    "refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001_REVIEW.json": (
        "8937446bbb9708e205f850e367c48fce70d42de9054c1066522865104352c8f6"
    ),
    "refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001_REVIEW.md": (
        "a1fcd938fcb849cc820275ffb54d4b233314accbd7d1a21aa2ce3742e2dafa9c"
    ),
}
SELECTOR_UNIT_CANONICAL_DOCUMENT_SHA256 = {
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
SELECTOR_UNIT_SOURCE_PATHS = frozenset(
    {
        "scripts/experiments/generate_warp_phase_fixture_pack.py",
        "scripts/experiments/generate_warp_phase_selector_unit_fixtures.py",
        "src/pams/warp_phase/gates.py",
        "src/pams/warp_phase/selector.py",
        "tests/test_warp_phase_selector.py",
    }
)

_TRAINING_LOCK_FIELDS = {
    "architecture",
    "endianness",
    "interpreter_sha256",
    "numpy_version",
    "os",
    "pcg64_witness_sha256",
    "pyproject_sha256",
    "python_implementation",
    "python_version",
    "role",
    "schema_version",
    "uv_lock_sha256",
}
_OUTPUT_SPACE_FIELDS = {
    "created_empty",
    "directory_device",
    "directory_inode",
    "output_root",
    "role",
    "schema_version",
}

_PCG64_WITNESS = (
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


@dataclass(frozen=True)
class GateReceipt:
    """Canonical, hashable gate result."""

    gate: str
    status: str
    checks: dict[str, bool]
    bindings: dict[str, str]
    blockers: tuple[str, ...]
    authorizes: tuple[str, ...]


@dataclass(frozen=True)
class _SelectorUnitFixtureSpec:
    fixture_id: str
    clock_count: int
    source_period: int
    span: int
    p_max: int
    selected_period: int


_SELECTOR_UNIT_FIXTURE_SPECS = (
    _SelectorUnitFixtureSpec("offbin_delta1_p20", 256, 20, 255, 127, 20),
    _SelectorUnitFixtureSpec("lower_endpoint_p4", 64, 4, 63, 31, 4),
    _SelectorUnitFixtureSpec("upper_endpoint_span126_p63", 127, 63, 126, 63, 63),
)


@dataclass(frozen=True)
class _Gate2FixtureStatistics:
    agreement: int
    half_overall: int
    double_overall: int
    half_symmetric: int
    double_symmetric: int

    def checks(self) -> dict[str, bool]:
        return {
            "fixture_agreement_at_least_950": self.agreement >= 950,
            "fixture_half_overall_at_most_20": self.half_overall <= 20,
            "fixture_double_overall_at_most_20": self.double_overall <= 20,
            "fixture_half_symmetric_at_most_12": self.half_symmetric <= 12,
            "fixture_double_symmetric_at_most_12": self.double_symmetric <= 12,
        }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize one receipt using the contract's canonical JSON rules."""

    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise GateError("receipt is not canonical-JSON serializable") from exc
    return encoded


def _read_stable_regular_file(path: Path) -> bytes:
    """Read immutable-looking bytes once and reject links or concurrent edits."""

    candidate = Path(path)
    try:
        before = candidate.lstat()
    except FileNotFoundError as exc:
        raise GateError(f"required file does not exist: {candidate}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise GateError(f"required path is not a regular non-symlink file: {candidate}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(candidate, flags)
    try:
        opened = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            payload = handle.read()
            closed = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = candidate.lstat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(closed, field)
        or getattr(closed, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise GateError(f"required file changed while it was read: {candidate}")
    if len(payload) != after.st_size:
        raise GateError(f"required file size changed while it was read: {candidate}")
    return payload


def _write_bytes_exclusive(path: Path, payload: bytes, *, mode: int = 0o444) -> str:
    """Durably create one immutable byte artifact and never overwrite it."""

    candidate = Path(path)
    if not candidate.parent.is_dir():
        raise GateError(f"receipt parent must already exist: {candidate.parent}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(candidate, flags, mode)
    except FileExistsError as exc:
        raise GateError(f"refusing to overwrite frozen artifact: {candidate}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        with suppress(FileNotFoundError):
            candidate.unlink()
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    with suppress(OSError):
        candidate.chmod(mode)
    return sha256_bytes(payload)


def write_canonical_json_exclusive(path: Path, payload: Any) -> str:
    """Create one canonical-JSON artifact with ``O_EXCL`` semantics."""

    return _write_bytes_exclusive(Path(path), canonical_json_bytes(payload))


def _load_canonical_json(path: Path) -> dict[str, Any]:
    payload = _read_stable_regular_file(Path(path))
    if payload.startswith(b"\xef\xbb\xbf") or payload.endswith(b"\n"):
        raise GateError(f"canonical JSON has a BOM or terminal newline: {path}")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"artifact is not UTF-8 canonical JSON: {path}") from exc
    if not isinstance(decoded, dict):
        raise GateError(f"canonical JSON root must be an object: {path}")
    if canonical_json_bytes(decoded) != payload:
        raise GateError(f"JSON bytes are not in the canonical form: {path}")
    return cast(dict[str, Any], decoded)


def receipt_bytes(receipt: GateReceipt) -> bytes:
    return canonical_json_bytes(asdict(receipt))


def write_receipt_exclusive(path: Path, receipt: GateReceipt) -> str:
    """Create a receipt once, refusing overwrite or non-canonical bytes."""

    payload = receipt_bytes(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(path, flags, 0o444)
    except FileExistsError as exc:
        raise GateError(f"refusing to overwrite frozen gate receipt: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        with suppress(FileNotFoundError):
            path.unlink()
        raise
    return sha256_bytes(payload)


def _training_environment_payload(repository_root: Path) -> dict[str, Any]:
    """Build the exact consumption-only training lock for the active process."""

    root = Path(repository_root).resolve(strict=True)
    if not root.is_dir():
        raise GateError("repository_root must be an existing directory")
    pyproject = root / "pyproject.toml"
    uv_lock = root / "uv.lock"
    pyproject_hash = sha256_bytes(_read_stable_regular_file(pyproject))
    uv_lock_hash = sha256_bytes(_read_stable_regular_file(uv_lock))
    if pyproject_hash != TRAINING_PYPROJECT_SHA256:
        raise GateError("pyproject.toml differs from the frozen training contract")
    if uv_lock_hash != TRAINING_UV_LOCK_SHA256:
        raise GateError("uv.lock differs from the frozen training contract")
    implementation = platform.python_implementation()
    python_version = platform.python_version()
    try:
        numpy_version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError as exc:
        raise GateError("NumPy is not installed in the training environment") from exc
    if implementation != "CPython" or python_version != TRAINING_PYTHON_VERSION:
        raise GateError(
            "training lock requires CPython "
            f"{TRAINING_PYTHON_VERSION}, got {implementation} {python_version}"
        )
    if numpy_version != TRAINING_NUMPY_VERSION:
        raise GateError(
            "training lock requires NumPy "
            f"{TRAINING_NUMPY_VERSION}, got {numpy_version}"
        )
    interpreter = Path(sys.executable).resolve(strict=True)
    return {
        "architecture": platform.machine(),
        "endianness": sys.byteorder,
        "interpreter_sha256": sha256_bytes(_read_stable_regular_file(interpreter)),
        "numpy_version": numpy_version,
        "os": platform.system(),
        "pcg64_witness_sha256": pcg64_witness(),
        "pyproject_sha256": pyproject_hash,
        "python_implementation": implementation,
        "python_version": python_version,
        "role": "training_consumption_only",
        "schema_version": 1,
        "uv_lock_sha256": uv_lock_hash,
    }


def create_training_environment_lock(repository_root: Path, lock_path: Path) -> str:
    """Freeze the active CPython/NumPy training environment exactly once."""

    payload = _training_environment_payload(repository_root)
    return write_canonical_json_exclusive(lock_path, payload)


def verify_training_environment_lock(repository_root: Path, lock_path: Path) -> str:
    """Verify canonical bytes and every live/frozen training-environment binding."""

    observed = _load_canonical_json(lock_path)
    if set(observed) != _TRAINING_LOCK_FIELDS:
        missing = sorted(_TRAINING_LOCK_FIELDS - set(observed))
        extra = sorted(set(observed) - _TRAINING_LOCK_FIELDS)
        raise GateError(
            f"training environment lock fields differ; missing={missing}, extra={extra}"
        )
    expected = _training_environment_payload(repository_root)
    if observed != expected:
        changed = sorted(
            key for key in _TRAINING_LOCK_FIELDS if observed.get(key) != expected.get(key)
        )
        raise GateError(f"training environment lock bindings differ: {changed}")
    return sha256_bytes(_read_stable_regular_file(lock_path))


def _require_output_root_name(output_root: Path) -> None:
    if output_root.name.casefold() in {"features", "vault", "audit"}:
        raise GateError("training output root cannot be features, vault, or audit")


def create_output_space_receipt(output_root: Path, receipt_path: Path) -> str:
    """Reserve a new empty output directory and bind it with one exclusive receipt.

    The receipt is intentionally outside the output directory so the directory
    remains observably empty until a later, separately authorized training job.
    """

    output = Path(output_root).resolve(strict=False)
    receipt = Path(receipt_path).resolve(strict=False)
    _require_output_root_name(output)
    if output == receipt.parent or output in receipt.parents:
        raise GateError("output-space receipt must be outside the reserved output root")
    if output.exists():
        raise GateError(f"training output root already exists: {output}")
    if not output.parent.is_dir():
        raise GateError(f"training output parent must already exist: {output.parent}")
    if not receipt.parent.is_dir():
        raise GateError(f"output-space receipt parent must already exist: {receipt.parent}")
    try:
        output.mkdir(mode=0o750)
    except FileExistsError as exc:
        raise GateError(f"training output root already exists: {output}") from exc
    try:
        metadata = output.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise GateError("reserved output root is not a regular directory")
        if any(output.iterdir()):
            raise GateError("newly reserved output root is not empty")
        payload = {
            "created_empty": True,
            "directory_device": int(metadata.st_dev),
            "directory_inode": int(metadata.st_ino),
            "output_root": str(output),
            "role": "training_output_space",
            "schema_version": 1,
        }
        return write_canonical_json_exclusive(receipt, payload)
    except Exception:
        with suppress(OSError):
            if output.is_dir() and not any(output.iterdir()):
                output.rmdir()
        raise


def verify_output_space_receipt(output_root: Path, receipt_path: Path) -> str:
    """Require the same reserved directory to remain regular and completely empty."""

    output = Path(output_root).resolve(strict=True)
    _require_output_root_name(output)
    observed = _load_canonical_json(receipt_path)
    if set(observed) != _OUTPUT_SPACE_FIELDS:
        missing = sorted(_OUTPUT_SPACE_FIELDS - set(observed))
        extra = sorted(set(observed) - _OUTPUT_SPACE_FIELDS)
        raise GateError(f"output-space receipt fields differ; missing={missing}, extra={extra}")
    metadata = output.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise GateError("reserved output root is not a regular non-symlink directory")
    if any(output.iterdir()):
        raise GateError("reserved training output root is no longer empty")
    expected = {
        "created_empty": True,
        "directory_device": int(metadata.st_dev),
        "directory_inode": int(metadata.st_ino),
        "output_root": str(output),
        "role": "training_output_space",
        "schema_version": 1,
    }
    if observed != expected:
        raise GateError("output-space receipt does not bind the supplied empty directory")
    return sha256_bytes(_read_stable_regular_file(receipt_path))


def verify_schema_fixture(payload: bytes = SCHEMA_FIXTURE.encode("utf-8")) -> str:
    """Verify the exact 617-byte desensitized schema precondition."""

    if len(payload) != 617:
        raise GateError(f"schema fixture must be exactly 617 bytes, got {len(payload)}")
    if payload.endswith(b"\n") or payload.startswith(b"\xef\xbb\xbf"):
        raise GateError("schema fixture must have no BOM or terminal newline")
    digest = sha256_bytes(payload)
    if digest != SCHEMA_FIXTURE_SHA256:
        raise GateError("schema fixture SHA-256 does not match the frozen contract")
    return digest


def pcg64_witness() -> str:
    """Reproduce the one-call PCG64 raw witness in the current environment."""

    rng = np.random.Generator(np.random.PCG64(20270815))
    values = np.asarray(rng.bit_generator.random_raw(16), dtype="<u8")
    if tuple(int(value) for value in values) != _PCG64_WITNESS:
        raise GateError("PCG64 raw values differ from the frozen witness")
    payload = np.ascontiguousarray(values, dtype="<u8").tobytes(order="C")
    if len(payload) != 128:
        raise GateError("PCG64 witness must be exactly 128 bytes")
    digest = sha256_bytes(payload)
    if digest != PCG64_WITNESS_SHA256:
        raise GateError("PCG64 witness byte hash differs from the frozen contract")
    return digest


def generate_k4_vectors() -> NDArray[np.float64]:
    """Generate the four frozen orthonormal Stage-B control vectors."""

    dimension = 128
    indices = np.arange(dimension, dtype=np.int64)
    scale = math.sqrt(float(dimension))
    u_q = np.where(indices % 2 == 0, 1.0, -1.0).astype(np.float64) / scale
    u_l = np.where(indices < 64, 1.0, -1.0).astype(np.float64) / scale
    rng = np.random.Generator(np.random.PCG64(20270815))

    def orthogonal_rademacher(basis: Sequence[NDArray[np.float64]]) -> NDArray[np.float64]:
        raw = 2 * rng.integers(0, 2, size=dimension, dtype=np.int64) - 1
        vector = raw.astype(np.float64)
        for member in basis:
            vector = vector - float(np.dot(vector, member)) * member
        norm = float(np.linalg.norm(vector))
        if not math.isfinite(norm) or norm == 0.0:
            raise GateError("K4 modified Gram-Schmidt produced a zero vector")
        vector = vector / norm
        nonzero = np.flatnonzero(vector != 0.0)
        if nonzero.size == 0:
            raise GateError("K4 vector has no nonzero entry")
        if vector[int(nonzero[0])] < 0.0:
            vector = -vector
        return vector

    u_m1 = orthogonal_rademacher((u_q, u_l))
    u_m2 = orthogonal_rademacher((u_q, u_l, u_m1))
    vectors = np.ascontiguousarray(np.stack((u_q, u_l, u_m1, u_m2)), dtype="<f8")
    payload = vectors.tobytes(order="C")
    if len(payload) != 4096:
        raise GateError("K4 vector payload must be exactly 4,096 bytes")
    digest = sha256_bytes(payload)
    if digest != K4_VECTOR_SHA256:
        raise GateError("K4 vector hash differs from the frozen NumPy 2.4.6 bytes")
    return vectors


def verify_environment_lock(
    lock: Mapping[str, Any],
    *,
    expected_python: str,
    expected_numpy: str,
) -> None:
    """Reject incomplete or differently versioned environment-lock records."""

    required = {
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
    if set(lock) != required:
        missing = sorted(required - set(lock))
        extra = sorted(set(lock) - required)
        raise GateError(f"environment lock fields differ; missing={missing}, extra={extra}")
    if any(value is None or str(value).strip() == "" for value in lock.values()):
        raise GateError("environment lock contains an empty field")
    if str(lock["python_version"]) != expected_python:
        raise GateError("environment Python version differs from the frozen contract")
    if str(lock["numpy_version"]) != expected_numpy:
        raise GateError("environment NumPy version differs from the frozen contract")
    for key in ("interpreter_sha256", "numpy_wheel_sha256", "source_sha256", "detached_receipt_sha256"):
        value = str(lock[key])
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise GateError(f"environment lock field {key!r} is not lowercase SHA-256")


def verify_pack_hash(files: Mapping[str, bytes], expected_sha256: str) -> str:
    """Verify the frozen filename-NUL-bytes-LF fixture-pack digest."""

    if not files:
        raise GateError("fixture pack must be nonempty")
    digest = hashlib.sha256()
    for name in sorted(files):
        if not name or "\x00" in name or Path(name).name != name:
            raise GateError("fixture member names must be simple nonempty filenames")
        digest.update(name.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(files[name])
        digest.update(b"\n")
    actual = digest.hexdigest()
    if actual != expected_sha256:
        raise GateError("fixture-pack hash differs from its frozen receipt")
    return actual


def validate_gate2_counts(
    canonical: NDArray[np.generic],
    weak: NDArray[np.generic],
    half_count: int,
    double_count: int,
    symmetric_half_count: int,
    symmetric_double_count: int,
) -> None:
    """Apply the inclusive frozen Gate-2 aggregate thresholds."""

    canonical_values = np.asarray(canonical)
    weak_values = np.asarray(weak)
    canonical_period = np.asarray(canonical_values, dtype=np.int64)
    weak_period = np.asarray(weak_values, dtype=np.int64)
    if canonical_period.shape != (1000,) or weak_period.shape != (1000,):
        raise GateError("Gate 2 requires exactly 1,000 canonical and weak fixture decisions")
    if (
        not np.issubdtype(canonical_values.dtype, np.number)
        or not np.issubdtype(weak_values.dtype, np.number)
        or not np.isfinite(canonical_values).all()
        or not np.isfinite(weak_values).all()
        or not np.array_equal(canonical_values, canonical_period)
        or not np.array_equal(weak_values, weak_period)
    ):
        raise GateError("Gate-2 fixture decisions must be finite exact integers")
    valid = (canonical_period > 0) & (weak_period > 0)
    agreement = valid & (10 * np.abs(weak_period - canonical_period) <= canonical_period)
    counts = (half_count, double_count, symmetric_half_count, symmetric_double_count)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
        raise GateError("Gate-2 harmonic counts must be nonnegative integers")
    statistics = _Gate2FixtureStatistics(
        agreement=int(np.sum(agreement, dtype=np.int64)),
        half_overall=half_count,
        double_overall=double_count,
        half_symmetric=symmetric_half_count,
        double_symmetric=symmetric_double_count,
    )
    threshold_checks = statistics.checks()
    if not all(threshold_checks.values()):
        reasons: list[str] = []
        if not threshold_checks["fixture_agreement_at_least_950"]:
            reasons.append("canonical/rejection agreement is below 95 percent")
        if not (
            threshold_checks["fixture_half_overall_at_most_20"]
            and threshold_checks["fixture_double_overall_at_most_20"]
        ):
            reasons.append("overall half/double classification exceeds 2 percent")
        if not (
            threshold_checks["fixture_half_symmetric_at_most_12"]
            and threshold_checks["fixture_double_symmetric_at_most_12"]
        ):
            reasons.append(
                "symmetric half/double classification exceeds the frozen count ceiling"
            )
        values = (
            f"agreement={statistics.agreement}/1000,"
            f"half_overall={statistics.half_overall},"
            f"double_overall={statistics.double_overall},"
            f"half_symmetric={statistics.half_symmetric}/250,"
            f"double_symmetric={statistics.double_symmetric}/250"
        )
        raise GateError(f"Gate-2 thresholds fail ({values}): {'; '.join(reasons)}")


def _blocker(code: str, detail: str | None = None) -> str:
    return code if detail is None else f"{code}:{detail}"


def gate0_pack_receipt(
    *,
    repository_root: Path,
    train_source: Path,
    val_source: Path,
    run_root: Path,
) -> GateReceipt:
    """Verify the Gate-0 boundary and fail closed before an unspecified unpack.

    The current package has exact trusted source hashing and per-identity
    feature/vault writers, but it deliberately has no normative adapter from
    the historical v44 pickle object graph to those typed records.  Inventing
    that adapter here would open labels under an unreviewed interpretation.
    This orchestration therefore verifies everything it can before unpickling
    and emits an explicit blocker instead of producing a pretend pack.
    """

    from pams.warp_phase.packing import (
        schema_fixture_bytes,
        verify_source_pickle,
    )

    checks = {
        "approved_train_source": False,
        "approved_val_source": False,
        "schema_fixture": False,
        "pack_adapter_frozen": False,
        "pack_vault_audit_complete": False,
    }
    bindings: dict[str, str] = {"run_root": str(Path(run_root).resolve(strict=False))}
    blockers: list[str] = []
    for split, source in (("train", train_source), ("val", val_source)):
        try:
            verified = verify_source_pickle(
                Path(source),
                repository_root=Path(repository_root),
                split=split,
            )
        except Exception as exc:
            blockers.append(_blocker(f"{split}_source_verification_failed", type(exc).__name__))
        else:
            checks[f"approved_{split}_source"] = True
            bindings[f"{split}_source_sha256"] = verified.sha256
            bindings[f"{split}_source_relative_path"] = verified.relative_path
    try:
        fixture = schema_fixture_bytes()
    except Exception as exc:
        blockers.append(_blocker("schema_fixture_failed", type(exc).__name__))
    else:
        checks["schema_fixture"] = True
        bindings["schema_fixture_sha256"] = sha256_bytes(fixture)
    blockers.extend(
        (
            "v44_typed_identity_adapter_not_frozen",
            "pack_vault_audit_outputs_not_created",
            "nine_component_eligibility_manifest_not_created",
            "training_import_isolation_receipt_not_created",
        )
    )
    return GateReceipt(
        gate="GATE0",
        status="BLOCKED",
        checks=checks,
        bindings=bindings,
        blockers=tuple(blockers),
        authorizes=(),
    )


_FIXTURE_MEMBER_SCHEMA: dict[str, tuple[str, tuple[int, ...]]] = {
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
_FIXTURE_FILENAMES = frozenset((*_FIXTURE_MEMBER_SCHEMA, "metadata.json"))

_SELECTOR_UNIT_SUFFIX_SCHEMA: dict[str, tuple[str, tuple[int | str, ...]]] = {
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


def _selector_unit_shape(
    shape: tuple[int | str, ...],
    clock_count: int,
) -> tuple[int, ...]:
    substitutions = {"N": clock_count, "N-1": clock_count - 1}
    return tuple(substitutions[value] if isinstance(value, str) else value for value in shape)


def _selector_unit_member_schema() -> dict[str, tuple[str, tuple[int, ...]]]:
    schemas: dict[str, tuple[str, tuple[int, ...]]] = {
        "candidate_period.npy": ("<u2", (125,))
    }
    for spec in _SELECTOR_UNIT_FIXTURE_SPECS:
        for suffix, (dtype, shape) in _SELECTOR_UNIT_SUFFIX_SCHEMA.items():
            schemas[f"{spec.fixture_id}.{suffix}"] = (
                dtype,
                _selector_unit_shape(shape, spec.clock_count),
            )
    return schemas


_SELECTOR_UNIT_MEMBER_SCHEMA = _selector_unit_member_schema()
_SELECTOR_UNIT_PACK_FILENAMES = frozenset(
    (*_SELECTOR_UNIT_MEMBER_SCHEMA, "metadata.json")
)
_SELECTOR_UNIT_ROOT_FILENAMES = frozenset(
    (*_SELECTOR_UNIT_PACK_FILENAMES, SELECTOR_UNIT_FIXTURE_RECEIPT)
)


def _load_npy_bytes(payload: bytes, *, name: str) -> NDArray[np.generic]:
    try:
        loaded = np.load(io.BytesIO(payload), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise GateError(f"fixture member is not a non-pickle NPY array: {name}") from exc
    if not isinstance(loaded, np.ndarray):
        raise GateError(f"fixture member is not one NPY array: {name}")
    array = np.asarray(loaded)
    expected_dtype, expected_shape = _FIXTURE_MEMBER_SCHEMA[name]
    if array.dtype.str != expected_dtype or array.shape != expected_shape:
        raise GateError(
            f"fixture member {name} has dtype/shape {array.dtype.str}/{array.shape}, "
            f"expected {expected_dtype}/{expected_shape}"
        )
    if array.dtype.hasobject or not array.flags.c_contiguous:
        raise GateError(f"fixture member is object-typed or non-C-order: {name}")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise GateError(f"fixture member contains a non-finite value: {name}")
    return array


def _load_selector_unit_npy_bytes(payload: bytes, *, name: str) -> NDArray[np.generic]:
    handle = io.BytesIO(payload)
    try:
        version = npy_format.read_magic(handle)
    except (EOFError, ValueError) as exc:
        raise GateError(f"selector unit member has no valid NPY header: {name}") from exc
    if version != (2, 0):
        raise GateError(f"selector unit member is NPY {version}, expected (2, 0): {name}")
    handle.seek(0)
    try:
        loaded = np.load(handle, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise GateError(f"selector unit member is not a non-pickle NPY array: {name}") from exc
    if not isinstance(loaded, np.ndarray):
        raise GateError(f"selector unit member is not one NPY array: {name}")
    array = np.asarray(loaded)
    expected_dtype, expected_shape = _SELECTOR_UNIT_MEMBER_SCHEMA[name]
    if array.dtype.str != expected_dtype or array.shape != expected_shape:
        raise GateError(
            f"selector unit member {name} has dtype/shape {array.dtype.str}/{array.shape}, "
            f"expected {expected_dtype}/{expected_shape}"
        )
    if array.dtype.hasobject or not array.flags.c_contiguous:
        raise GateError(f"selector unit member is object-typed or non-C-order: {name}")
    if array.dtype.kind == "f" and not np.isfinite(array).all():
        raise GateError(f"selector unit member contains a non-finite value: {name}")
    return array


def _selector_unit_constructor(
    spec: _SelectorUnitFixtureSpec,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.uint8]]:
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


def _selector_unit_expected_arrays(
    clocks: NDArray[np.generic],
    pose: NDArray[np.generic],
    joint_mask: NDArray[np.generic],
) -> dict[str, NDArray[np.generic]]:
    from pams.warp_phase.selector import normalize_coco17, select_support_period

    normalized = normalize_coco17(pose, joint_mask, clocks)
    selection = select_support_period(
        normalized.clocks,
        normalized.coordinates,
        normalized.joint_valid,
    )
    diagnostics = selection.diagnostics
    candidate_indices = np.asarray(diagnostics.candidates - 4, dtype=np.int64)
    if np.any(candidate_indices < 0) or np.any(candidate_indices >= 125):
        raise GateError("selector unit candidates are outside periods 4 through 128")
    candidate_present = np.zeros(125, dtype="|u1")
    acf = np.zeros(125, dtype="<f8")
    acf_computed = np.zeros(125, dtype="|u1")
    harmonic = np.zeros((125, 3), dtype="<f8")
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
    computed_harmonic = np.asarray(diagnostics.harmonic_power, dtype=np.float64).copy()
    computed_harmonic[~finite_score] = np.float64(0.0)
    harmonic[candidate_indices] = computed_harmonic
    clock_float = np.asarray(clocks, dtype=np.float64)
    velocity = np.diff(normalized.coordinates.reshape(clock_float.size, 34), axis=0)
    velocity /= np.diff(clock_float)[:, None]
    selected = -1 if selection.period is None else selection.period
    return {
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
        "expected_velocity.npy": np.ascontiguousarray(velocity, dtype="<f8"),
        "expected_velocity_mask.npy": np.ascontiguousarray(
            np.repeat(normalized.joint_valid, 2, axis=1)[:-1]
            & np.repeat(normalized.joint_valid, 2, axis=1)[1:],
            dtype="|u1",
        ),
        "expected_fft_x.npy": np.ascontiguousarray(diagnostics.fft_grid, dtype="<f8"),
        "expected_fft_mask.npy": np.ascontiguousarray(diagnostics.fft_mask, dtype="|u1"),
        "expected_fft_power.npy": np.ascontiguousarray(diagnostics.fft_power, dtype="<f8"),
        "expected_candidate_present.npy": candidate_present,
        "expected_acf.npy": acf,
        "expected_acf_computed.npy": acf_computed,
        "expected_harmonic_power.npy": harmonic,
        "expected_score.npy": score,
        "expected_score_computed.npy": score_computed,
        "expected_score_valid.npy": score_valid,
        "expected_local_max.npy": local_maximum,
        "expected_selected_period.npy": np.asarray((selected,), dtype="<i2"),
    }


def _require_hash_mapping(
    value: object,
    *,
    expected: Mapping[str, str],
    label: str,
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise GateError(f"selector unit {label} paths differ")
    observed = {str(key): str(item) for key, item in value.items()}
    if observed != dict(expected):
        raise GateError(f"selector unit {label} SHA-256 bindings differ")
    return observed


def _require_live_hash_mapping(
    repository_root: Path,
    value: object,
    *,
    paths: frozenset[str],
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(paths):
        raise GateError("selector unit source snapshot paths differ")
    observed = {str(key): str(item) for key, item in value.items()}
    for relative in sorted(paths):
        actual = sha256_bytes(
            _read_stable_regular_file(Path(repository_root) / relative)
        )
        if observed[relative] != actual:
            raise GateError(f"selector unit source snapshot differs: {relative}")
    return observed


def _verify_selector_unit_metadata(
    metadata: Mapping[str, Any],
    *,
    source_sha256: Mapping[str, str],
) -> None:
    fields = {
        "acceptance_review_sha256",
        "amendment_sha256",
        "array_schemas",
        "artifact_root",
        "artifact_type",
        "authoritative",
        "authorizes",
        "candidate_only",
        "canonical_document_sha256",
        "constructor",
        "decision_failure_sentinel",
        "diagnostic_fill",
        "diagnostic_status",
        "environment",
        "fixture_table",
        "npy_format_version",
        "pack_member_count_excluding_receipt",
        "rng_calls",
        "rng_used",
        "role",
        "root_member_count_including_receipt",
        "source_sha256",
        "status",
    }
    if set(metadata) != fields:
        raise GateError("selector unit metadata fields differ")
    expected_schemas = {
        name: {"dtype": dtype, "shape": list(shape)}
        for name, (dtype, shape) in sorted(_SELECTOR_UNIT_MEMBER_SCHEMA.items())
    }
    fixture_table = {
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
        for spec in _SELECTOR_UNIT_FIXTURE_SPECS
    }
    constructor = {
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
    }
    diagnostics = {
        spec.fixture_id: {"failure_reason": None, "stage": "complete"}
        for spec in _SELECTOR_UNIT_FIXTURE_SPECS
    }
    scalar_expected = {
        "array_schemas": expected_schemas,
        "artifact_root": SELECTOR_UNIT_FIXTURE_RELATIVE_ROOT.as_posix(),
        "artifact_type": "warp_phase_selector_unit_fixture_pack_candidate",
        "authoritative": False,
        "authorizes": [],
        "candidate_only": True,
        "constructor": constructor,
        "decision_failure_sentinel": -1,
        "diagnostic_fill": 0.0,
        "diagnostic_status": diagnostics,
        "fixture_table": fixture_table,
        "npy_format_version": "2.0",
        "pack_member_count_excluding_receipt": 65,
        "rng_calls": 0,
        "rng_used": False,
        "role": "deterministic_non_efficacy_gate1_selector_unit_tests",
        "root_member_count_including_receipt": 66,
        "source_sha256": dict(source_sha256),
        "status": "CANDIDATE_PENDING_FRESH_REVIEW",
    }
    changed = [key for key, expected in scalar_expected.items() if metadata[key] != expected]
    if changed:
        raise GateError(f"selector unit metadata values differ: {sorted(changed)}")
    environment = metadata["environment"]
    environment_fields = {
        "architecture",
        "endianness",
        "numpy_version",
        "os",
        "python_implementation",
        "python_version",
    }
    if not isinstance(environment, dict) or set(environment) != environment_fields:
        raise GateError("selector unit metadata environment fields differ")
    if (
        environment["python_implementation"] != "CPython"
        or environment["python_version"] != "3.12.13"
        or environment["numpy_version"] != "2.4.6"
        or environment["endianness"] != "little"
        or any(not str(environment[key]).strip() for key in environment_fields)
    ):
        raise GateError("selector unit metadata environment differs from the contract")


def _compare_selector_unit_arrays(
    spec: _SelectorUnitFixtureSpec,
    arrays: Mapping[str, NDArray[np.generic]],
) -> None:
    prefix = f"{spec.fixture_id}."
    clocks = arrays[f"{prefix}input_clocks.npy"]
    pose = np.asarray(arrays[f"{prefix}input_pose.npy"], dtype=np.float64)
    joint_mask = arrays[f"{prefix}input_joint_mask.npy"]
    expected_clocks, expected_pose, expected_mask = _selector_unit_constructor(spec)
    try:
        np.testing.assert_array_equal(clocks, expected_clocks)
        np.testing.assert_array_equal(joint_mask, expected_mask)
        np.testing.assert_allclose(
            pose,
            expected_pose,
            rtol=1e-12,
            atol=1e-12,
            equal_nan=False,
        )
    except AssertionError as exc:
        raise GateError(f"selector unit {spec.fixture_id} constructor differs") from exc
    recomputed = _selector_unit_expected_arrays(clocks, pose, joint_mask)
    for suffix, expected in recomputed.items():
        observed = arrays[f"{prefix}{suffix}"]
        try:
            if observed.dtype.kind == "f":
                np.testing.assert_allclose(
                    np.asarray(observed, dtype=np.float64),
                    np.asarray(expected, dtype=np.float64),
                    rtol=1e-12,
                    atol=1e-12,
                    equal_nan=False,
                )
            else:
                np.testing.assert_array_equal(observed, expected)
        except AssertionError as exc:
            raise GateError(
                f"selector unit {spec.fixture_id} recomputation differs: {suffix}"
            ) from exc
    selected = int(arrays[f"{prefix}expected_selected_period.npy"][0])
    if selected != spec.selected_period:
        raise GateError(f"selector unit {spec.fixture_id} selected period differs")
    if selected < 0 and selected != -1:
        raise GateError("selector unit decision uses a forbidden failure sentinel")
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
        if np.any(arrays[f"{prefix}{suffix}"] == -1):
            raise GateError(
                f"selector unit {spec.fixture_id}.{suffix} contains decision-only -1"
            )
    for suffix in (
        "expected_normalized_joint_mask.npy",
        "expected_feature_frame_valid.npy",
        "expected_velocity_mask.npy",
        "expected_fft_mask.npy",
    ):
        if not np.all(arrays[f"{prefix}{suffix}"] == 1):
            raise GateError(f"selector unit {spec.fixture_id} is not all-valid: {suffix}")
    if float(arrays[f"{prefix}expected_scale.npy"][0]) != 2.0:
        raise GateError(f"selector unit {spec.fixture_id} scale is not exact 2.0")
    present = arrays[f"{prefix}expected_candidate_present.npy"].astype(np.bool_)
    expected_present = np.zeros(125, dtype=np.bool_)
    expected_present[: spec.p_max - 3] = True
    if not np.array_equal(present, expected_present):
        raise GateError(f"selector unit {spec.fixture_id} candidate support differs")
    acf_computed = arrays[f"{prefix}expected_acf_computed.npy"].astype(np.bool_)
    score_computed = arrays[f"{prefix}expected_score_computed.npy"].astype(np.bool_)
    if int(np.sum(acf_computed, dtype=np.int64)) != spec.p_max - 3:
        raise GateError(f"selector unit {spec.fixture_id} ACF computed mask differs")
    if int(np.sum(score_computed, dtype=np.int64)) != spec.p_max - 3:
        raise GateError(f"selector unit {spec.fixture_id} score computed mask differs")
    fft_x = arrays[f"{prefix}expected_fft_x.npy"]
    delta = float((fft_x[-1] - fft_x[0]) / np.float64(255.0))
    if spec.fixture_id == "offbin_delta1_p20":
        exact_grid = np.arange(256, dtype="<f8")
        if fft_x.tobytes(order="C") != exact_grid.tobytes(order="C") or delta != 1.0:
            raise GateError("off-bin selector unit grid is not byte-exact Delta=1")
        continuous_bin = np.float64(256.0) * np.float64(delta) / np.float64(20.0)
        if float(continuous_bin) != 12.8:
            raise GateError("off-bin selector unit k(20) is not float64 12.8")
        power = arrays[f"{prefix}expected_fft_power.npy"]
        interpolated = (
            (np.float64(13.0) - continuous_bin) * power[11]
            + (continuous_bin - np.float64(12.0)) * power[12]
        )
        harmonic = arrays[f"{prefix}expected_harmonic_power.npy"]
        if harmonic[16, 0] != interpolated:
            raise GateError("off-bin selector unit does not interpolate bins 12 and 13")
    score = arrays[f"{prefix}expected_score.npy"]
    local = arrays[f"{prefix}expected_local_max.npy"].astype(np.bool_)
    if spec.fixture_id == "lower_endpoint_p4" and not (score[0] >= score[1] and local[0]):
        raise GateError("lower endpoint selector unit does not use the one-sided maximum")
    if spec.fixture_id == "upper_endpoint_span126_p63" and not (
        score[59] >= score[58] and local[59]
    ):
        raise GateError("upper endpoint selector unit does not use the one-sided maximum")


def verify_selector_unit_fixture_pack(
    repository_root: Path,
    fixture_root: Path,
    pack_receipt_path: Path,
) -> tuple[dict[str, NDArray[np.generic]], str, str]:
    """Verify Amendment-001 unit members, hashes, and NumPy-1.26 recomputation."""

    repository = Path(repository_root).resolve(strict=True)
    root = Path(fixture_root).resolve(strict=True)
    root_metadata = root.lstat()
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise GateError("selector unit fixture root must be a regular non-symlink directory")
    entries = {entry.name: entry for entry in root.iterdir()}
    if set(entries) != _SELECTOR_UNIT_ROOT_FILENAMES:
        missing = sorted(_SELECTOR_UNIT_ROOT_FILENAMES - set(entries))
        extra = sorted(set(entries) - _SELECTOR_UNIT_ROOT_FILENAMES)
        raise GateError(f"selector unit filenames differ; missing={missing}, extra={extra}")
    receipt_path = Path(pack_receipt_path).resolve(strict=True)
    if receipt_path != entries[SELECTOR_UNIT_FIXTURE_RECEIPT].resolve(strict=True):
        raise GateError("selector unit receipt is not the exact receipt beside the members")
    receipt = _load_canonical_json(receipt_path)
    receipt_fields = {
        "acceptance_review_sha256",
        "amendment_sha256",
        "artifact_type",
        "authoritative",
        "authorizes",
        "candidate_only",
        "canonical_document_sha256",
        "freeze_action_performed",
        "fresh_review_required",
        "member_sha256",
        "pack_hash_algorithm",
        "pack_member_order",
        "pack_sha256",
        "schema_version",
        "source_sha256",
        "status",
    }
    if set(receipt) != receipt_fields:
        raise GateError("selector unit candidate receipt fields differ")
    static_receipt = {
        "artifact_type": "warp_phase_selector_unit_fixture_pack_candidate_receipt",
        "authoritative": False,
        "authorizes": [],
        "candidate_only": True,
        "freeze_action_performed": False,
        "fresh_review_required": True,
        "pack_hash_algorithm": SELECTOR_UNIT_PACK_HASH_ALGORITHM,
        "schema_version": 1,
        "status": "CANDIDATE_NON_AUTHORIZING",
    }
    changed = [key for key, expected in static_receipt.items() if receipt[key] != expected]
    if changed:
        raise GateError(f"selector unit candidate receipt values differ: {sorted(changed)}")
    _require_hash_mapping(
        receipt["amendment_sha256"],
        expected=SELECTOR_UNIT_AMENDMENT_SHA256,
        label="amendment",
    )
    _require_hash_mapping(
        receipt["acceptance_review_sha256"],
        expected=SELECTOR_UNIT_ACCEPTANCE_REVIEW_SHA256,
        label="acceptance review",
    )
    _require_hash_mapping(
        receipt["canonical_document_sha256"],
        expected=SELECTOR_UNIT_CANONICAL_DOCUMENT_SHA256,
        label="canonical document",
    )
    for bindings in (
        SELECTOR_UNIT_AMENDMENT_SHA256,
        SELECTOR_UNIT_ACCEPTANCE_REVIEW_SHA256,
        SELECTOR_UNIT_CANONICAL_DOCUMENT_SHA256,
    ):
        for relative, expected_hash in bindings.items():
            observed_hash = sha256_bytes(_read_stable_regular_file(repository / relative))
            if observed_hash != expected_hash:
                raise GateError(f"selector unit frozen input differs: {relative}")
    source_sha256 = _require_live_hash_mapping(
        repository,
        receipt["source_sha256"],
        paths=SELECTOR_UNIT_SOURCE_PATHS,
    )
    member_order = sorted(_SELECTOR_UNIT_PACK_FILENAMES)
    if receipt["pack_member_order"] != member_order:
        raise GateError("selector unit candidate receipt member order differs")
    member_sha256 = receipt["member_sha256"]
    if not isinstance(member_sha256, dict) or set(member_sha256) != set(member_order):
        raise GateError("selector unit candidate receipt member hashes differ")
    pack_sha256 = str(receipt["pack_sha256"])
    if len(pack_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in pack_sha256
    ):
        raise GateError("selector unit candidate receipt pack hash is not lowercase SHA-256")

    files: dict[str, bytes] = {}
    arrays: dict[str, NDArray[np.generic]] = {}
    metadata_payload: dict[str, Any] | None = None
    for name in member_order:
        payload = _read_stable_regular_file(entries[name])
        observed_hash = sha256_bytes(payload)
        if member_sha256.get(name) != observed_hash:
            raise GateError(f"selector unit member hash differs: {name}")
        files[name] = payload
        if name.endswith(".npy"):
            arrays[name] = _load_selector_unit_npy_bytes(payload, name=name)
        else:
            try:
                parsed = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GateError("selector unit metadata is not UTF-8 JSON") from exc
            if not isinstance(parsed, dict) or canonical_json_bytes(parsed) != payload:
                raise GateError("selector unit metadata is not canonical JSON")
            metadata_payload = cast(dict[str, Any], parsed)
    verify_pack_hash(files, pack_sha256)
    if metadata_payload is None:
        raise GateError("selector unit metadata is missing")
    _verify_selector_unit_metadata(metadata_payload, source_sha256=source_sha256)
    _require_hash_mapping(
        metadata_payload["amendment_sha256"],
        expected=SELECTOR_UNIT_AMENDMENT_SHA256,
        label="metadata amendment",
    )
    _require_hash_mapping(
        metadata_payload["acceptance_review_sha256"],
        expected=SELECTOR_UNIT_ACCEPTANCE_REVIEW_SHA256,
        label="metadata acceptance review",
    )
    _require_hash_mapping(
        metadata_payload["canonical_document_sha256"],
        expected=SELECTOR_UNIT_CANONICAL_DOCUMENT_SHA256,
        label="metadata canonical document",
    )
    candidates = arrays["candidate_period.npy"]
    if not np.array_equal(candidates, np.arange(4, 129, dtype="<u2")):
        raise GateError("selector unit candidates are not exactly integer periods 4 through 128")
    for spec in _SELECTOR_UNIT_FIXTURE_SPECS:
        _compare_selector_unit_arrays(spec, arrays)
    receipt_sha256 = sha256_bytes(_read_stable_regular_file(receipt_path))
    return arrays, pack_sha256, receipt_sha256


def verify_fixture_pack(
    fixture_root: Path,
    pack_receipt_path: Path,
) -> tuple[dict[str, NDArray[np.generic]], str, str]:
    """Verify a separately generated immutable 1,000-fixture pack exactly."""

    root = Path(fixture_root).resolve(strict=True)
    metadata = root.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise GateError("fixture root must be a regular non-symlink directory")
    entries = {entry.name: entry for entry in root.iterdir()}
    if set(entries) != _FIXTURE_FILENAMES:
        missing = sorted(_FIXTURE_FILENAMES - set(entries))
        extra = sorted(set(entries) - _FIXTURE_FILENAMES)
        raise GateError(f"fixture filenames differ; missing={missing}, extra={extra}")
    receipt = _load_canonical_json(pack_receipt_path)
    expected_receipt_fields = {
        "generator_source_sha256",
        "members",
        "pack_sha256",
        "role",
        "schema_version",
    }
    if set(receipt) != expected_receipt_fields:
        raise GateError("fixture-pack receipt fields differ from the frozen verifier schema")
    if receipt["role"] != "canonical_fixture_pack" or receipt["schema_version"] != 1:
        raise GateError("fixture-pack receipt role or schema version differs")
    generator_hash = str(receipt["generator_source_sha256"])
    expected_pack_hash = str(receipt["pack_sha256"])
    for name, value in (
        ("generator_source_sha256", generator_hash),
        ("pack_sha256", expected_pack_hash),
    ):
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise GateError(f"fixture receipt {name} is not lowercase SHA-256")
    member_hashes = receipt["members"]
    if not isinstance(member_hashes, dict) or set(member_hashes) != _FIXTURE_FILENAMES:
        raise GateError("fixture receipt must hash every and only canonical member")

    files: dict[str, bytes] = {}
    arrays: dict[str, NDArray[np.generic]] = {}
    for name in sorted(_FIXTURE_FILENAMES):
        payload = _read_stable_regular_file(entries[name])
        observed_hash = sha256_bytes(payload)
        if member_hashes.get(name) != observed_hash:
            raise GateError(f"fixture member hash differs: {name}")
        files[name] = payload
        if name.endswith(".npy"):
            arrays[name] = _load_npy_bytes(payload, name=name)
        else:
            metadata_payload = json.loads(payload.decode("utf-8"))
            if canonical_json_bytes(metadata_payload) != payload:
                raise GateError("fixture metadata is not canonical JSON")
    verify_pack_hash(files, expected_pack_hash)
    categories = arrays["category.npy"]
    expected_categories = np.repeat(np.arange(9, dtype=np.uint16), (150, 100, 125, 125, 100, 100, 100, 100, 100))
    if not np.array_equal(categories, expected_categories):
        raise GateError("fixture category IDs/counts differ from the frozen 1,000 schedule")
    candidates = arrays["candidate_period.npy"]
    if not np.array_equal(candidates, np.arange(4, 129, dtype=np.uint16)):
        raise GateError("fixture candidates are not exactly integer periods 4 through 128")
    return arrays, expected_pack_hash, generator_hash


def _verify_ten_case_recomputation(arrays: Mapping[str, NDArray[np.generic]]) -> None:
    from pams.warp_phase.selector import (
        SelectorError,
        SupportSelection,
        normalize_coco17,
        select_support_period,
    )

    def safe_selection(
        clocks: NDArray[np.generic],
        coordinates: NDArray[np.generic],
        joint_valid: NDArray[np.generic],
    ) -> SupportSelection | None:
        try:
            return select_support_period(clocks, coordinates, joint_valid)
        except SelectorError:
            return None

    def selected_or_failure(selection: SupportSelection | None) -> int:
        if selection is None or selection.period is None:
            return -1
        return selection.period

    fixture_ids = (0, 150, 250, 375, 500, 600, 700, 800, 900, 999)
    q_float = np.asarray(arrays["q.npy"], dtype=np.float64)
    q_integer = q_float.astype(np.int64)
    if not np.array_equal(q_float, q_integer.astype(np.float64)):
        raise GateError("fixture q clocks are not exact integers")
    for fixture_id in fixture_ids:
        normalized = normalize_coco17(
            arrays["pose.npy"][fixture_id],
            arrays["joint_mask.npy"][fixture_id],
            q_integer,
        )
        selection = safe_selection(
            normalized.clocks,
            normalized.coordinates,
            normalized.joint_valid,
        )
        if selection is None:
            raise GateError(f"fixture {fixture_id} canonical diagnostics are unavailable")
        expected_selected = int(arrays["expected_selected_period.npy"][fixture_id])
        observed_selected = selected_or_failure(selection)
        if observed_selected != expected_selected:
            raise GateError(f"fixture {fixture_id} selected period differs")
        try:
            weak_normalized = normalize_coco17(
                arrays["weak_pose.npy"][fixture_id],
                arrays["weak_joint_mask.npy"][fixture_id],
                q_integer,
            )
        except SelectorError:
            weak_selection = None
        else:
            weak_selection = safe_selection(
                weak_normalized.clocks,
                weak_normalized.coordinates,
                weak_normalized.joint_valid,
            )
        first_selection = safe_selection(
            normalized.clocks[:64],
            normalized.coordinates[:64],
            normalized.joint_valid[:64],
        )
        last_selection = safe_selection(
            normalized.clocks[-64:],
            normalized.coordinates[-64:],
            normalized.joint_valid[-64:],
        )
        reverse_q = q_float[0] + (q_float[-1] - q_float[::-1])
        reversal_selection = safe_selection(
            reverse_q,
            normalized.coordinates[::-1],
            normalized.joint_valid[::-1],
        )
        selected_results = (
            ("weak", weak_selection, "expected_weak_selected_period.npy"),
            ("first64", first_selection, "expected_first64_selected_period.npy"),
            ("last64", last_selection, "expected_last64_selected_period.npy"),
            ("reversal", reversal_selection, "expected_reversal_selected_period.npy"),
        )
        for label, observed, member in selected_results:
            if selected_or_failure(observed) != int(arrays[member][fixture_id]):
                raise GateError(f"fixture {fixture_id} {label} selected period differs")
        candidate_indices = selection.candidates - 4
        if np.any(candidate_indices < 0) or np.any(candidate_indices >= 125):
            raise GateError(f"fixture {fixture_id} candidates are outside 4..128")
        observed_acf = np.zeros(125, dtype=np.float64)
        observed_score = np.zeros(125, dtype=np.float64)
        observed_score_valid = np.zeros(125, dtype=np.bool_)
        observed_local_max = np.zeros(125, dtype=np.bool_)
        observed_harmonic = np.zeros((125, 3), dtype=np.float64)
        finite_acf = np.isfinite(selection.acf)
        finite_score = np.isfinite(selection.score_by_period)
        valid_candidates = selection.score_valid & finite_acf & finite_score
        observed_acf[candidate_indices] = np.where(valid_candidates, selection.acf, 0.0)
        observed_score[candidate_indices] = np.where(
            valid_candidates,
            selection.score_by_period,
            0.0,
        )
        observed_score_valid[candidate_indices] = selection.score_valid
        observed_local_max[candidate_indices] = selection.local_maximum
        harmonic = selection.harmonic_power.copy()
        harmonic[~valid_candidates] = 0.0
        observed_harmonic[candidate_indices] = harmonic
        try:
            np.testing.assert_allclose(
                selection.fft_grid,
                arrays["expected_fft_x.npy"][fixture_id],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=False,
            )
            np.testing.assert_array_equal(
                selection.fft_mask,
                arrays["expected_fft_mask.npy"][fixture_id].astype(np.bool_),
            )
            np.testing.assert_allclose(
                selection.fft_power,
                arrays["expected_fft_power.npy"][fixture_id],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=False,
            )
            np.testing.assert_allclose(
                observed_acf,
                arrays["expected_acf.npy"][fixture_id],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=False,
            )
            np.testing.assert_allclose(
                observed_score,
                arrays["expected_score.npy"][fixture_id],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=False,
            )
            np.testing.assert_array_equal(
                observed_score_valid,
                arrays["expected_score_valid.npy"][fixture_id].astype(np.bool_),
            )
            np.testing.assert_array_equal(
                observed_local_max,
                arrays["expected_local_max.npy"][fixture_id].astype(np.bool_),
            )
            np.testing.assert_allclose(
                observed_harmonic,
                arrays["expected_harmonic_power.npy"][fixture_id],
                rtol=1e-12,
                atol=1e-12,
                equal_nan=False,
            )
        except AssertionError as exc:
            raise GateError(f"fixture {fixture_id} selector diagnostics differ") from exc


def gate1_fixture_receipt(
    *,
    repository_root: Path,
    training_lock_path: Path | None,
    fixture_lock_path: Path | None,
    fixture_root: Path | None,
    pack_receipt_path: Path | None,
    selector_unit_fixture_root: Path | None = None,
    selector_unit_pack_receipt_path: Path | None = None,
) -> GateReceipt:
    """Validate all currently representable Gate-1 inputs without regeneration."""

    checks = {
        "training_environment_lock": False,
        "fixture_environment_lock": False,
        "pcg64_witness": False,
        "fixture_pack": False,
        "ten_case_recomputation": False,
        "selector_unit_fixture_pack": False,
        "complete_numerical_witness_set": False,
    }
    bindings: dict[str, str] = {}
    blockers: list[str] = []
    try:
        checks["pcg64_witness"] = pcg64_witness() == PCG64_WITNESS_SHA256
        bindings["pcg64_witness_sha256"] = PCG64_WITNESS_SHA256
    except Exception as exc:
        blockers.append(_blocker("pcg64_witness_failed", type(exc).__name__))
    if training_lock_path is None:
        blockers.append("training_environment_lock_missing")
    else:
        try:
            bindings["training_environment_lock_sha256"] = verify_training_environment_lock(
                repository_root,
                training_lock_path,
            )
            checks["training_environment_lock"] = True
        except Exception as exc:
            blockers.append(_blocker("training_environment_lock_failed", type(exc).__name__))
    if fixture_lock_path is None:
        blockers.append("fixture_environment_lock_missing")
    else:
        try:
            fixture_lock = _load_canonical_json(fixture_lock_path)
            verify_environment_lock(
                fixture_lock,
                expected_python="3.12.13",
                expected_numpy="2.4.6",
            )
            bindings["fixture_environment_lock_sha256"] = sha256_bytes(
                _read_stable_regular_file(fixture_lock_path)
            )
            checks["fixture_environment_lock"] = True
        except Exception as exc:
            blockers.append(_blocker("fixture_environment_lock_failed", type(exc).__name__))
    arrays: dict[str, NDArray[np.generic]] | None = None
    if fixture_root is None or pack_receipt_path is None:
        blockers.append("canonical_fixture_pack_or_receipt_missing")
    else:
        try:
            arrays, pack_hash, generator_hash = verify_fixture_pack(
                fixture_root,
                pack_receipt_path,
            )
            bindings["fixture_pack_sha256"] = pack_hash
            bindings["fixture_generator_source_sha256"] = generator_hash
            checks["fixture_pack"] = True
        except Exception as exc:
            blockers.append(_blocker("fixture_pack_failed", type(exc).__name__))
    if arrays is not None:
        try:
            _verify_ten_case_recomputation(arrays)
            checks["ten_case_recomputation"] = True
        except Exception as exc:
            blockers.append(_blocker("ten_case_recomputation_failed", type(exc).__name__))
    if selector_unit_fixture_root is None or selector_unit_pack_receipt_path is None:
        blockers.append("selector_unit_fixture_pack_or_receipt_missing")
    else:
        try:
            expected_root = (
                Path(repository_root).resolve(strict=True)
                / SELECTOR_UNIT_FIXTURE_RELATIVE_ROOT
            ).resolve(strict=False)
            supplied_root = Path(selector_unit_fixture_root).resolve(strict=True)
            if supplied_root != expected_root:
                raise GateError("selector unit fixture root is not the canonical Gate-1 root")
            _, unit_pack_hash, unit_receipt_hash = verify_selector_unit_fixture_pack(
                repository_root,
                supplied_root,
                selector_unit_pack_receipt_path,
            )
            checks["selector_unit_fixture_pack"] = True
            bindings["selector_unit_fixture_pack_sha256"] = unit_pack_hash
            bindings["selector_unit_fixture_receipt_sha256"] = unit_receipt_hash
        except Exception as exc:
            blockers.append(_blocker("selector_unit_fixture_pack_failed", type(exc).__name__))
    blockers.append(
        "complete_gate1_duplicate_warp_recurrence_optimizer_decode_evaluator_witness_schema_missing"
    )
    return GateReceipt(
        gate="GATE1",
        status="BLOCKED",
        checks=checks,
        bindings=bindings,
        blockers=tuple(blockers),
        authorizes=(),
    )


def _gate2_fixture_statistics(
    arrays: Mapping[str, NDArray[np.generic]],
) -> _Gate2FixtureStatistics:
    canonical_values = np.asarray(arrays["expected_selected_period.npy"])
    weak_values = np.asarray(arrays["expected_weak_selected_period.npy"])
    semantic = np.asarray(arrays["semantic_period.npy"], dtype=np.float64)
    canonical = np.asarray(canonical_values, dtype=np.int64)
    weak = np.asarray(weak_values, dtype=np.int64)
    if canonical.shape != (1000,) or weak.shape != (1000,) or semantic.shape != (1000,):
        raise GateError("Gate 2 statistics require the exact 1,000-row fixture arrays")
    if (
        not np.isfinite(canonical_values).all()
        or not np.isfinite(weak_values).all()
        or not np.isfinite(semantic).all()
        or not np.array_equal(canonical_values, canonical)
        or not np.array_equal(weak_values, weak)
    ):
        raise GateError("Gate 2 statistics require finite exact fixture decisions")
    valid_pair = (canonical > 0) & (weak > 0)
    agreement = valid_pair & (10 * np.abs(weak - canonical) <= canonical)
    valid_canonical = canonical > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        half = valid_canonical & (
            np.abs(canonical / (np.float64(0.5) * semantic) - np.float64(1.0))
            <= np.float64(0.10)
        )
        double = valid_canonical & (
            np.abs(canonical / (np.float64(2.0) * semantic) - np.float64(1.0))
            <= np.float64(0.10)
        )
    symmetric = np.zeros(1000, dtype=np.bool_)
    symmetric[250:500] = True
    return _Gate2FixtureStatistics(
        agreement=int(np.sum(agreement, dtype=np.int64)),
        half_overall=int(np.sum(half, dtype=np.int64)),
        double_overall=int(np.sum(double, dtype=np.int64)),
        half_symmetric=int(np.sum(half & symmetric, dtype=np.int64)),
        double_symmetric=int(np.sum(double & symmetric, dtype=np.int64)),
    )


def verify_gate2_fixture_thresholds(
    fixture_root: Path,
    pack_receipt_path: Path,
) -> tuple[str, str]:
    """Verify immutable fixture bytes and apply the unchanged Gate-2 thresholds."""

    arrays, pack_hash, generator_hash = verify_fixture_pack(fixture_root, pack_receipt_path)
    statistics = _gate2_fixture_statistics(arrays)
    validate_gate2_counts(
        arrays["expected_selected_period.npy"],
        arrays["expected_weak_selected_period.npy"],
        statistics.half_overall,
        statistics.double_overall,
        statistics.half_symmetric,
        statistics.double_symmetric,
    )
    return pack_hash, generator_hash


def gate2_selector_receipt(
    *,
    fixture_root: Path | None,
    pack_receipt_path: Path | None,
    features_root: Path | None,
    order_receipt_path: Path | None,
) -> GateReceipt:
    """Validate immutable fixtures and expose the honest real-track blocker.

    The selector now separates perturbation materialization from receipt-bound
    evaluation.  This gate still lacks the frozen eligible-track manifest and
    complete pre-draw/order artifacts, so it never opens real feature shards,
    never evaluates a real track, and never reports Gate 2 as passed.
    """

    checks = {
        "immutable_1000_fixture_pack": False,
        "fixture_thresholds": False,
        "fixture_agreement_at_least_950": False,
        "fixture_half_overall_at_most_20": False,
        "fixture_double_overall_at_most_20": False,
        "fixture_half_symmetric_at_most_12": False,
        "fixture_double_symmetric_at_most_12": False,
        "eligible_real_track_manifest": False,
        "pre_evaluation_order_receipt": False,
        "all_real_tracks_have_selector_answer": False,
    }
    bindings: dict[str, str] = {}
    blockers: list[str] = []
    terminal_failure = False
    if fixture_root is None or pack_receipt_path is None:
        blockers.append("canonical_fixture_pack_or_receipt_missing")
    else:
        try:
            arrays, pack_hash, generator_hash = verify_fixture_pack(
                fixture_root,
                pack_receipt_path,
            )
            checks["immutable_1000_fixture_pack"] = True
            bindings["fixture_pack_sha256"] = pack_hash
            bindings["fixture_generator_source_sha256"] = generator_hash
        except Exception as exc:
            terminal_failure = True
            blockers.append(_blocker("immutable_fixture_pack_failed", str(exc)))
        else:
            try:
                statistics = _gate2_fixture_statistics(arrays)
            except Exception as exc:
                terminal_failure = True
                blockers.append(_blocker("fixture_aggregate_computation_failed", str(exc)))
            else:
                statistic_checks = statistics.checks()
                checks.update(statistic_checks)
                checks["fixture_thresholds"] = all(statistic_checks.values())
                bindings.update(
                    {
                        "fixture_agreement_count": str(statistics.agreement),
                        "fixture_agreement_denominator": "1000",
                        "fixture_agreement_minimum": "950",
                        "fixture_half_overall_count": str(statistics.half_overall),
                        "fixture_half_overall_maximum": "20",
                        "fixture_double_overall_count": str(statistics.double_overall),
                        "fixture_double_overall_maximum": "20",
                        "fixture_half_symmetric_count": str(statistics.half_symmetric),
                        "fixture_half_symmetric_denominator": "250",
                        "fixture_half_symmetric_maximum": "12",
                        "fixture_double_symmetric_count": str(statistics.double_symmetric),
                        "fixture_double_symmetric_denominator": "250",
                        "fixture_double_symmetric_maximum": "12",
                    }
                )
                observed_and_limits = (
                    (
                        "fixture_agreement_at_least_950",
                        statistics.agreement,
                        ">=950/1000",
                    ),
                    (
                        "fixture_half_overall_at_most_20",
                        statistics.half_overall,
                        "<=20/1000",
                    ),
                    (
                        "fixture_double_overall_at_most_20",
                        statistics.double_overall,
                        "<=20/1000",
                    ),
                    (
                        "fixture_half_symmetric_at_most_12",
                        statistics.half_symmetric,
                        "<=12/250",
                    ),
                    (
                        "fixture_double_symmetric_at_most_12",
                        statistics.double_symmetric,
                        "<=12/250",
                    ),
                )
                for check_name, observed, required in observed_and_limits:
                    if not statistic_checks[check_name]:
                        blockers.append(
                            _blocker(
                                "fixture_threshold_failed",
                                f"{check_name}:observed={observed}:required={required}",
                            )
                        )
                        terminal_failure = True
    if features_root is None:
        blockers.append("eligible_training_feature_root_missing")
    else:
        bindings["features_root"] = str(Path(features_root).resolve(strict=False))
    if order_receipt_path is None:
        blockers.append("real_track_order_receipt_path_missing")
    else:
        bindings["order_receipt_path"] = str(Path(order_receipt_path).resolve(strict=False))
    blockers.extend(
        (
            "eligible_real_track_aggregate_manifest_schema_missing",
            "frozen_real_track_predraw_order_not_supplied",
            "real_track_selector_not_run",
        )
    )
    return GateReceipt(
        gate="GATE2",
        status="FAIL" if terminal_failure else "BLOCKED",
        checks=checks,
        bindings=bindings,
        blockers=tuple(blockers),
        authorizes=(),
    )


def gate3_cpu_sanity_receipt() -> GateReceipt:
    """Run a deterministic CPU-only execution smoke; never launch a GPU job."""

    import torch

    from pams.warp_phase.model import WarpPhaseModel
    from pams.warp_phase.training import (
        FRESH_STEPS_PER_UPDATE,
        IDENTITIES_PER_FRESH_STEP,
        FreshStepLoss,
        build_optimizer,
        run_optimizer_update,
    )

    checks = {
        "finite_gradients": False,
        "loss_decrease": False,
        "identity_isolation": False,
        "invalid_state_hold": False,
        "single_source_clock_commit": False,
        "four_fresh_backwards": False,
        "prediction_schema": False,
        "tiny_gpu_jobs": False,
    }
    bindings: dict[str, str] = {}
    blockers: list[str] = []
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20270815)
            model = WarpPhaseModel()
            optimizer = build_optimizer(model)
            generator = torch.Generator(device="cpu").manual_seed(20270815)
            pose = torch.randn(
                IDENTITIES_PER_FRESH_STEP,
                4,
                17,
                3,
                dtype=torch.float32,
                generator=generator,
            )
            joint = torch.ones(IDENTITIES_PER_FRESH_STEP, 4, 17, dtype=torch.bool)
            frame = torch.ones(IDENTITIES_PER_FRESH_STEP, 4, dtype=torch.bool)
            clock = torch.ones(IDENTITIES_PER_FRESH_STEP, 4, dtype=torch.bool)
            frame[0, 1] = False

            def objective() -> FreshStepLoss:
                output = model(pose, joint, frame, clock)
                return FreshStepLoss(
                    loss=output.hidden.square().mean(),
                    identity_count=IDENTITIES_PER_FRESH_STEP,
                )

            before_output = model(pose, joint, frame, clock)
            before_loss = float(before_output.hidden.square().mean().detach().item())
            checks["invalid_state_hold"] = bool(
                torch.equal(before_output.hidden[0, 1], before_output.hidden[0, 0])
            )
            checks["single_source_clock_commit"] = bool(
                torch.equal(
                    before_output.commit_count,
                    torch.full_like(before_output.commit_count, 4),
                )
            )
            factories = [lambda _model: objective() for _ in range(FRESH_STEPS_PER_UPDATE)]
            update = run_optimizer_update(model, optimizer, factories)
            after_loss = float(model(pose, joint, frame, clock).hidden.square().mean().item())
            checks["finite_gradients"] = math.isfinite(update.gradient_norm_before_clip)
            checks["loss_decrease"] = after_loss < before_loss
            checks["four_fresh_backwards"] = bool(
                update.backward_calls == 4
                and update.optimizer_steps == 1
                and update.identity_draws == 32
            )
            isolation_pose = pose[:2].clone()
            isolation_joint = joint[:2]
            isolation_frame = frame[:2]
            isolation_clock = clock[:2]
            model.eval()
            with torch.no_grad():
                first = model(
                    isolation_pose,
                    isolation_joint,
                    isolation_frame,
                    isolation_clock,
                ).phase
                isolation_pose[0] += 1000.0
                second = model(
                    isolation_pose,
                    isolation_joint,
                    isolation_frame,
                    isolation_clock,
                ).phase
            checks["identity_isolation"] = bool(torch.equal(first[1], second[1]))
            diagnostic = np.asarray(
                [before_loss, after_loss, update.gradient_norm_before_clip],
                dtype="<f8",
            )
            bindings["cpu_diagnostic_sha256"] = sha256_bytes(diagnostic.tobytes(order="C"))
    except Exception as exc:
        blockers.append(_blocker("cpu_sanity_execution_failed", type(exc).__name__))
    failed_cpu = sorted(
        name
        for name in (
            "finite_gradients",
            "loss_decrease",
            "identity_isolation",
            "invalid_state_hold",
            "single_source_clock_commit",
            "four_fresh_backwards",
        )
        if not checks[name]
    )
    blockers.extend(_blocker("cpu_check_failed", name) for name in failed_cpu)
    blockers.extend(
        (
            "prediction_output_schema_not_frozen_in_package_types",
            "three_tiny_gpu_receipts_missing",
            "gates_0_1_2_pass_receipts_missing",
        )
    )
    return GateReceipt(
        gate="GATE3-CPU",
        status="BLOCKED",
        checks=checks,
        bindings=bindings,
        blockers=tuple(blockers),
        authorizes=(),
    )


def static_gate_receipt() -> GateReceipt:
    """Run only environment-independent immutable byte checks.

    This receipt does not pass Gate 0 or Gate 1 and authorizes no training.
    """

    checks = {
        "schema_fixture": verify_schema_fixture() == SCHEMA_FIXTURE_SHA256,
        "pcg64_witness": pcg64_witness() == PCG64_WITNESS_SHA256,
        "k4_vectors": sha256_bytes(generate_k4_vectors().tobytes(order="C")) == K4_VECTOR_SHA256,
    }
    return GateReceipt(
        gate="STATIC-PREFLIGHT",
        status="PASS_NON_AUTHORIZING",
        checks=checks,
        bindings={
            "schema_fixture_sha256": SCHEMA_FIXTURE_SHA256,
            "pcg64_witness_sha256": PCG64_WITNESS_SHA256,
            "k4_vector_sha256": K4_VECTOR_SHA256,
        },
        blockers=(
            "real Gate 0 pack/vault/audit isolation has not run",
            "dedicated NumPy 2.4.6 fixture environment has not been frozen",
            "1,000-fixture pack and all eligible real-track selector receipts do not exist",
        ),
        authorizes=(),
    )
