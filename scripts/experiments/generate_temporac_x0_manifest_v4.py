"""Generate the non-authorizing TempoRAC X0 v4 locked-runtime candidate.

This generator is intentionally synthetic-only and fail-closed.  It must run
inside the declared Linux/amd64 CPython image with the exact NumPy and SciPy
versions.  The final candidate directory is created exclusively and is never
updated in place.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from pams.temporac import contract, x0

EXPECTED_CONTRACT_SHA256 = "3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391"
EXPECTED_DOCKER_IMAGE = "python:3.12.4"
EXPECTED_NUMPY_VERSION = "2.1.0"
EXPECTED_PYTHON_VERSION = "3.12.4"
EXPECTED_SCIPY_VERSION = "1.14.1"
OUTPUT_RELATIVE = Path("data/temporac_p00_v4/x0_candidate_20260816")
MANIFEST_ROW_KEYS = frozenset(
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
ATTEMPT_ROW_KEYS = frozenset(
    {
        "attempt",
        "coefficient_bytes_sha256",
        "coefficient_seed_u64",
        "guard_pass",
        "id",
        "orbit_grid_sha256",
    }
)


class CandidateGenerationError(RuntimeError):
    """Raised when a locked-runtime or candidate invariant fails."""


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution_record_sha256(distribution_name: str) -> str:
    record = importlib.metadata.distribution(distribution_name).read_text("RECORD")
    if record is None:
        raise CandidateGenerationError(f"{distribution_name} RECORD is missing")
    return _sha256_bytes(record.encode("utf-8"))


def _runtime_lock() -> dict[str, Any]:
    marker = os.environ.get("TEMPORAC_X0_DOCKER_IMAGE")
    machine = platform.machine().lower()
    observed = {
        "architecture": machine,
        "byteorder": sys.byteorder,
        "docker_image": marker,
        "numpy_version": np.__version__,
        "platform_system": platform.system(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "scipy_version": scipy.__version__,
    }
    expected = {
        "architecture": "x86_64",
        "byteorder": "little",
        "docker_image": EXPECTED_DOCKER_IMAGE,
        "numpy_version": EXPECTED_NUMPY_VERSION,
        "platform_system": "Linux",
        "python_implementation": "CPython",
        "python_version": EXPECTED_PYTHON_VERSION,
        "scipy_version": EXPECTED_SCIPY_VERSION,
    }
    if observed != expected:
        raise CandidateGenerationError(
            f"locked runtime mismatch: expected={expected!r}, observed={observed!r}"
        )
    numpy_path = Path(np.__file__).resolve()
    scipy_path = Path(scipy.__file__).resolve()
    python_path = Path(sys.executable).resolve()
    return {
        **observed,
        "numpy_init_sha256": _sha256_file(numpy_path),
        "numpy_record_sha256": _distribution_record_sha256("numpy"),
        "python_executable_sha256": _sha256_file(python_path),
        "scipy_init_sha256": _sha256_file(scipy_path),
        "scipy_record_sha256": _distribution_record_sha256("scipy"),
    }


def _repository_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "src/pams/temporac/x0.py").is_file():
        raise CandidateGenerationError("repository root does not contain the X0 implementation")
    return root


def _bound_hashes(root: Path) -> dict[str, str]:
    paths = {
        "contract_module_sha256": root / "src/pams/temporac/contract.py",
        "generator_source_sha256": Path(__file__).resolve(),
        "implementation_source_sha256": root / "src/pams/temporac/x0.py",
        "proposal_sha256": root / "refine-logs/FINAL_PROPOSAL.md",
        "proposal_temporac_copy_sha256": root / "refine-logs/temporac/FINAL_PROPOSAL.md",
        "test_source_sha256": root / "tests/temporac/test_x0.py",
    }
    hashes = {name: _sha256_file(path) for name, path in paths.items()}
    if contract.CONTRACT_SHA256 != EXPECTED_CONTRACT_SHA256:
        raise CandidateGenerationError("contract module does not name the canonical proposal hash")
    if hashes["proposal_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise CandidateGenerationError("canonical proposal hash mismatch")
    if hashes["proposal_temporac_copy_sha256"] != EXPECTED_CONTRACT_SHA256:
        raise CandidateGenerationError("TempoRAC proposal copy hash mismatch")
    return hashes


def _replay_all_attempts() -> tuple[bytes, dict[str, Any]]:
    phase = np.arange(4096, dtype=np.float64) / 4096.0
    rows: list[dict[str, Any]] = []
    passing = 0
    for source_id in range(x0.SOURCE_COUNT):
        for attempt in range(64):
            try:
                seed, coefficients = x0._orthonormal_coefficients(source_id, attempt)
            except x0.X0ContractError as error:
                raise CandidateGenerationError(
                    f"MGS rejected U{source_id:04d} attempt {attempt}"
                ) from error
            guard_pass = x0._candidate_passes(coefficients)
            grid = x0.evaluate_orbit(coefficients, phase)
            coefficients_le = np.asarray(coefficients, dtype="<f8", order="C")
            grid_le = np.asarray(grid, dtype="<f8", order="C")
            row = {
                "attempt": attempt,
                "coefficient_bytes_sha256": _sha256_bytes(coefficients_le.tobytes(order="C")),
                "coefficient_seed_u64": seed,
                "guard_pass": guard_pass,
                "id": f"U{source_id:04d}",
                "orbit_grid_sha256": _sha256_bytes(grid_le.tobytes(order="C")),
            }
            if frozenset(row) != ATTEMPT_ROW_KEYS:
                raise CandidateGenerationError("attempt replay row schema drift")
            rows.append(row)
            passing += int(guard_pass)
    expected_count = x0.SOURCE_COUNT * 64
    if len(rows) != expected_count or passing != expected_count:
        raise CandidateGenerationError(
            f"full candidate replay failed: passing={passing}, expected={expected_count}"
        )
    replay_bytes = _canonical_json_bytes(rows)
    return replay_bytes, {
        "attempt_count": expected_count,
        "attempt_replay_sha256": _sha256_bytes(replay_bytes),
        "guard_failures": 0,
        "guard_passes": passing,
        "mgs_rejections": 0,
    }


def _manifest_bytes() -> bytes:
    manifest = x0.x0_manifest(EXPECTED_CONTRACT_SHA256)
    parsed = json.loads(manifest)
    if type(parsed) is not list or len(parsed) != x0.SOURCE_COUNT:
        raise CandidateGenerationError("X0 manifest must be a forty-row JSON array")
    expected_ids = [f"U{source_id:04d}" for source_id in range(x0.SOURCE_COUNT)]
    if [row.get("id") for row in parsed] != expected_ids:
        raise CandidateGenerationError("X0 manifest row order mismatch")
    if any(type(row) is not dict or frozenset(row) != MANIFEST_ROW_KEYS for row in parsed):
        raise CandidateGenerationError("X0 manifest row schema drift")
    if any(row["accepted_attempt"] != 0 for row in parsed):
        raise CandidateGenerationError("every X0 source must retain attempt zero")
    if manifest != _canonical_json_bytes(parsed):
        raise CandidateGenerationError("X0 manifest is not canonical LF-terminated JSON")
    return manifest


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def main() -> int:
    root = _repository_root()
    output = root / OUTPUT_RELATIVE
    if output.exists():
        raise CandidateGenerationError(f"refusing to overwrite existing candidate: {output}")

    runtime_bytes = _canonical_json_bytes(_runtime_lock())
    bindings = _bound_hashes(root)
    replay_bytes, replay_summary = _replay_all_attempts()
    manifest_bytes = _manifest_bytes()

    live_bindings = _bound_hashes(root)
    if live_bindings != bindings:
        raise CandidateGenerationError("bound source bytes changed during generation")
    runtime_lock_sha256 = _sha256_bytes(runtime_bytes)
    witness = {
        "artifact_type": "temporac_x0_full_replay_witness_v4",
        "candidate_status": "CANDIDATE_PENDING_FRESH_REVIEW",
        "commands": {
            "docker": (
                "docker run --rm --platform linux/amd64 "
                "-e TEMPORAC_X0_DOCKER_IMAGE=python:3.12.4 "
                "-e PYTHONPATH=/workspace/src -v <repository>:/workspace "
                "-w /workspace python:3.12.4 bash -lc "
                '"python -m pip install --no-cache-dir numpy==2.1.0 scipy==1.14.1 '
                '&& python scripts/experiments/generate_temporac_x0_manifest_v4.py"'
            ),
            "install": ("python -m pip install --no-cache-dir numpy==2.1.0 scipy==1.14.1"),
            "replay": "python scripts/experiments/generate_temporac_x0_manifest_v4.py",
        },
        "contract_sha256": EXPECTED_CONTRACT_SHA256,
        "manifest_row_count": x0.SOURCE_COUNT,
        "replay": replay_summary,
        "retained_attempt": 0,
        "retained_source_count": x0.SOURCE_COUNT,
        "runtime_lock_sha256": runtime_lock_sha256,
    }
    witness_bytes = _canonical_json_bytes(witness)
    members = {
        "attempt_replay.json": _sha256_bytes(replay_bytes),
        "environment_lock.json": runtime_lock_sha256,
        "replay_witness.json": _sha256_bytes(witness_bytes),
        "temporac.x0-manifest.v4.json": _sha256_bytes(manifest_bytes),
    }
    receipt = {
        "artifact_type": "temporac_x0_manifest_candidate_v4",
        "authoritative": False,
        "authorizes": [],
        "bindings": {**bindings, "runtime_lock_sha256": runtime_lock_sha256},
        "blockers": [
            "fresh_candidate_review_required",
            "P2_not_claimed",
            "S0_not_claimed",
            "launch_authorization_zero",
        ],
        "candidate_status": "CANDIDATE_PENDING_FRESH_REVIEW",
        "members": members,
        "p2_status": "NOT_CLAIMED",
        "s0_status": "NOT_CLAIMED",
    }
    receipt_bytes = _canonical_json_bytes(receipt)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()
    _write_exclusive(output / "attempt_replay.json", replay_bytes)
    _write_exclusive(output / "environment_lock.json", runtime_bytes)
    _write_exclusive(output / "replay_witness.json", witness_bytes)
    _write_exclusive(output / "temporac.x0-manifest.v4.json", manifest_bytes)
    _write_exclusive(output / "candidate_receipt.json", receipt_bytes)
    sys.stdout.buffer.write(
        _canonical_json_bytes(
            {
                "candidate_receipt_sha256": _sha256_bytes(receipt_bytes),
                "candidate_status": receipt["candidate_status"],
                "output": OUTPUT_RELATIVE.as_posix(),
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
