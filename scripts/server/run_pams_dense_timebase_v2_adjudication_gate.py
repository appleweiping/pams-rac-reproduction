"""Adjudicate the failed dense-timebase v1 synthetic gate under Algorithm 1.

V1 evaluated the already frozen decoder correctly, but two gate checks required
an arbitrary prevalence of majority votes.  PAMS Algorithm 1 does not impose
that condition: when the three experts disagree, it selects the expert count
nearest the FFT reference.  This gate accepts only the exact immutable v1
artifact/receipt pair, verifies the candidate source files are unchanged, and
checks every observed no-majority row directly.  It accepts no dataset,
checkpoint, development, test, target, count-label, or action-label argument.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

import yaml

from pams.reproducibility import clean_git_revision, sha256_json

_ARTIFACT_TYPE = "pams_dense_timebase_v2_synthetic_adjudication_gate"
_RECEIPT_TYPE = "pams_dense_timebase_v2_synthetic_adjudication_gate_receipt"
_EXPECTED_CANDIDATE = "pams-reference-relative-dense-resampled-masked-dft-multi-v2"
_EXPECTED_MODEL_CANDIDATE = (
    "pams-reference-relative-dense-resampled-masked-dft-multi-v1"
)
_EXPECTED_POLICY_SHA256 = (
    "197da26bc0cb8c42ccded4c4678f5537b78876a13a082920937dfad2b5b240d6"
)
_EXPECTED_PREDECESSOR = {
    "source_git_sha": "61adcdec6f89aa22faba45a59a32b902dc70880a",
    "policy_sha256": "828a6e64565274bfed1ff5c1822d98d1e3720bd0bea625d4c065ccc7aee63da4",
    "policy_semantic_sha256": (
        "e5b057305d17fb71c91d7fd746aa0cfa56871a9495fa52414abe3dd80858c126"
    ),
    "runner_sha256": "a2b04d5321f5ceea1c09d039f34cf200fc99897d71bd7401bc9c601eefca9684",
    "artifact_sha256": "944648276d146bd92c938425c0be72242ddb5e36066e1430cc8c84ed93f0fa02",
    "receipt_sha256": "5587f370347aa20aba80002b7904236955051b08775ac78ff2e066b03de798c5",
}
_EXPECTED_SOURCE_FILES = {
    "src/pams/period.py": "2409fe1ce341f72b3304236031f639c17a62e35133f680165d90550bcc28a85c",
    "src/pams/consensus.py": (
        "20f3e33651b44288327374b5810adadaad81ce2544d5ef47921b19988ac1cb51"
    ),
    "src/pams/config.py": "699b90929a5f94cdba922894e1ffe527e702b2809d749d816722b70903a250eb",
}
_EXPECTED_FAILED_CHECKS = {
    "clean_majority_fraction",
    "harmonic_majority_fraction",
}
_EXPECTED_V1_CHECKS = {
    "legacy_all_valid_exact_parity",
    "median_period_relative_error",
    "p95_period_relative_error",
    "harmonic_alias_fraction",
    "chain_nmae",
    "chain_obo",
    "clean_majority_fraction",
    "harmonic_majority_fraction",
    "resample_period_relative_error",
    "resample_count_equal_fraction",
    "invalid_payload_exact_invariance",
    "all_selected_modes_multi",
    "existing_576_counter_gate",
}
_EXPECTED_THRESHOLDS = {
    "predecessor_only_expected_failures",
    "predecessor_output_quality_checks",
    "clean_no_majority_fft_nearest_fraction_minimum",
    "harmonic_no_majority_fft_nearest_fraction_minimum",
    "clean_no_majority_selected_expert_fraction_minimum",
    "harmonic_no_majority_selected_expert_fraction_minimum",
    "clean_no_majority_generated_truth_fraction_minimum",
    "harmonic_no_majority_generated_truth_fraction_minimum",
    "no_majority_case_counts_exact",
    "candidate_source_files_unchanged",
}
_EXPERT_INDEX = {"fast": 0, "medium": 1, "slow": 2}


def _encoded_json(payload: object) -> bytes:
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


def _stable_file_identity(path: str | Path) -> tuple[str, int]:
    """Hash one regular non-symlink file and detect concurrent mutation."""

    source = Path(path)
    before = source.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"gate input must be a regular non-symlink file: {source}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        opened = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
            closed = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = source.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(closed, field)
        or getattr(closed, field) != getattr(after, field)
        for field in fields
    ):
        raise RuntimeError(f"gate input changed while it was read: {source}")
    if byte_count != after.st_size:
        raise RuntimeError(f"gate input byte count changed while reading: {source}")
    return digest.hexdigest(), byte_count


def _load_strict_json_object(path: Path, *, document_name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field in {document_name}: {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {document_name}: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {document_name} JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{document_name} root must be an object")
    return payload


def _load_policy(path: str | Path) -> tuple[dict[str, Any], str, str]:
    source = Path(path)
    digest, _ = _stable_file_identity(source)
    if digest != _EXPECTED_POLICY_SHA256:
        raise ValueError("adjudication requires the exact frozen dense-timebase v2 policy")
    parsed = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError("dense-timebase v2 policy root must be a mapping")
    policy = dict(parsed)
    expected_header = {
        "schema_version": 1,
        "candidate_id": _EXPECTED_CANDIDATE,
        "model_candidate_id": _EXPECTED_MODEL_CANDIDATE,
        "protocol_revision": "post-synthetic-v2",
        "table2_eligible": False,
    }
    for key, expected in expected_header.items():
        if policy.get(key) != expected:
            raise ValueError(f"dense-timebase v2 policy field {key!r} drifted")
    source_lock = policy.get("candidate_source_lock")
    if not isinstance(source_lock, Mapping):
        raise ValueError("dense-timebase v2 policy is missing candidate_source_lock")
    expected_source_lock = {
        "predecessor_source_git_sha": _EXPECTED_PREDECESSOR["source_git_sha"],
        "period_py_sha256": _EXPECTED_SOURCE_FILES["src/pams/period.py"],
        "consensus_py_sha256": _EXPECTED_SOURCE_FILES["src/pams/consensus.py"],
        "config_py_sha256": _EXPECTED_SOURCE_FILES["src/pams/config.py"],
    }
    if dict(source_lock) != expected_source_lock:
        raise ValueError("candidate source lock drifted")
    predecessor = policy.get("predecessor_gate")
    if not isinstance(predecessor, Mapping):
        raise ValueError("dense-timebase v2 policy is missing predecessor_gate")
    for key in (
        "policy_sha256",
        "policy_semantic_sha256",
        "runner_sha256",
        "artifact_sha256",
        "receipt_sha256",
    ):
        if predecessor.get(key) != _EXPECTED_PREDECESSOR[key]:
            raise ValueError(f"predecessor gate field {key!r} drifted")
    if predecessor.get("required_status") != "failed" or set(
        predecessor.get("required_failed_checks", ())
    ) != _EXPECTED_FAILED_CHECKS:
        raise ValueError("predecessor failure contract drifted")
    gate = policy.get("synthetic_adjudication_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("dense-timebase v2 policy is missing synthetic gate")
    thresholds = gate.get("thresholds")
    if not isinstance(thresholds, Mapping) or set(thresholds) != _EXPECTED_THRESHOLDS:
        raise ValueError("adjudication threshold names drifted")
    if any(value is not True for key, value in thresholds.items() if key.endswith("_exact")):
        raise ValueError("exact adjudication thresholds must remain true")
    for key, value in thresholds.items():
        if key.endswith("_fraction_minimum") and float(value) != 1.0:
            raise ValueError("no-majority adjudication fractions must remain 1.0")
    expected_counts = gate.get("expected_case_counts")
    if expected_counts != {
        "clean_count": 156,
        "harmonic": 72,
        "clean_no_majority": 16,
        "harmonic_no_majority": 19,
    }:
        raise ValueError("adjudication case-count contract drifted")
    authorization = policy.get("authorization")
    if not isinstance(authorization, Mapping) or authorization.get(
        "test105_attempt_budget"
    ) != 0:
        raise ValueError("dense-timebase v2 policy must keep test105 sealed")
    if _stable_file_identity(source)[0] != digest:
        raise RuntimeError("dense-timebase v2 policy changed during validation")
    return policy, digest, sha256_json(policy)


def _validate_candidate_source_files(repo_root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in _EXPECTED_SOURCE_FILES.items():
        digest, _ = _stable_file_identity(repo_root / relative)
        observed[relative] = digest
        if digest != expected:
            raise ValueError(f"frozen candidate source changed: {relative}")
    return observed


def _validate_predecessor_pair(
    artifact_path: Path,
    receipt_path: Path,
    *,
    identities: Mapping[str, tuple[str, int]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if identities["predecessor_artifact"][0] != _EXPECTED_PREDECESSOR[
        "artifact_sha256"
    ] or identities["predecessor_receipt"][0] != _EXPECTED_PREDECESSOR[
        "receipt_sha256"
    ]:
        raise ValueError("predecessor artifact or receipt SHA-256 differs from v2 policy")
    artifact = _load_strict_json_object(
        artifact_path,
        document_name="dense-timebase v1 synthetic gate artifact",
    )
    receipt = _load_strict_json_object(
        receipt_path,
        document_name="dense-timebase v1 synthetic gate receipt",
    )
    expected_artifact = {
        "schema_version": 1,
        "artifact_type": "pams_dense_timebase_v1_synthetic_gate",
        "candidate_id": _EXPECTED_MODEL_CANDIDATE,
        "status": "failed",
        "passed": False,
        "table2_eligible": False,
        "labels_accessed": False,
        "dataset_inputs_accessed": False,
        "checkpoint_inputs_accessed": False,
        "dev84_inputs_accessed": False,
        "test105_inputs_accessed": False,
    }
    for key, expected in expected_artifact.items():
        if artifact.get(key) != expected:
            raise ValueError(f"predecessor artifact field {key!r} drifted")
    source = artifact.get("source")
    if not isinstance(source, Mapping) or dict(source) != {
        "runner_source_git_sha": _EXPECTED_PREDECESSOR["source_git_sha"],
        "runner_sha256": _EXPECTED_PREDECESSOR["runner_sha256"],
        "policy_sha256": _EXPECTED_PREDECESSOR["policy_sha256"],
        "policy_semantic_sha256": _EXPECTED_PREDECESSOR["policy_semantic_sha256"],
    }:
        raise ValueError("predecessor artifact source binding drifted")
    checks = artifact.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != _EXPECTED_V1_CHECKS:
        raise ValueError("predecessor v1 check schema drifted")
    failed = {name for name, value in checks.items() if value is not True}
    if failed != _EXPECTED_FAILED_CHECKS:
        raise ValueError("predecessor failed checks differ from the disclosed pair")
    expected_receipt = {
        "schema_version": 1,
        "artifact_type": "pams_dense_timebase_v1_synthetic_gate_receipt",
        "artifact_locator": artifact_path.name,
        "artifact_sha256": identities["predecessor_artifact"][0],
        "artifact_bytes": identities["predecessor_artifact"][1],
        "artifact_status": "failed",
        "candidate_id": _EXPECTED_MODEL_CANDIDATE,
        "policy_sha256": _EXPECTED_PREDECESSOR["policy_sha256"],
        "policy_semantic_sha256": _EXPECTED_PREDECESSOR["policy_semantic_sha256"],
        "runner_source_git_sha": _EXPECTED_PREDECESSOR["source_git_sha"],
        "runner_sha256": _EXPECTED_PREDECESSOR["runner_sha256"],
        "labels_accessed": False,
        "dataset_inputs_accessed": False,
        "checkpoint_inputs_accessed": False,
        "train337_gate_authorized": False,
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }
    if receipt != expected_receipt:
        raise ValueError("predecessor receipt schema or binding drifted")
    return artifact, receipt


def _fraction(values: Sequence[bool]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty adjudication set")
    return float(sum(values) / len(values))


def _validate_dense_decode(raw: Any, *, row_name: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{row_name} dense decode must be a mapping")
    dense = dict(raw)
    counts = dense.get("expert_counts")
    if (
        not isinstance(counts, list)
        or len(counts) != 3
        or any(isinstance(value, bool) or not isinstance(value, int) for value in counts)
    ):
        raise ValueError(f"{row_name} expert counts drifted")
    count = dense.get("count")
    reference = dense.get("reference_count")
    selected = dense.get("selected_expert")
    period = dense.get("period_frames")
    if isinstance(count, bool) or not isinstance(count, int):
        raise ValueError(f"{row_name} final count drifted")
    if isinstance(reference, bool) or not isinstance(reference, int):
        raise ValueError(f"{row_name} reference count drifted")
    if selected not in _EXPERT_INDEX:
        raise ValueError(f"{row_name} selected expert drifted")
    if not isinstance(period, (int, float)) or not math.isfinite(float(period)):
        raise ValueError(f"{row_name} period drifted")
    computed_majority = max(counts.count(value) for value in set(counts)) >= 2
    if dense.get("has_majority") is not computed_majority:
        raise ValueError(f"{row_name} majority flag disagrees with expert counts")
    if dense.get("selection_mode") != "multi":
        raise ValueError(f"{row_name} did not use the frozen multi-expert mode")
    if counts[_EXPERT_INDEX[str(selected)]] != count:
        raise ValueError(f"{row_name} selected expert does not produce final count")
    expected_reference = math.floor(256.0 / float(period))
    if reference != expected_reference:
        raise ValueError(f"{row_name} FFT reference count drifted")
    return dense


def _adjudicate_rows(
    predecessor: Mapping[str, Any],
    *,
    expected_counts: Mapping[str, int],
) -> tuple[dict[str, Any], dict[str, bool]]:
    audit = predecessor.get("audit_rows")
    if not isinstance(audit, Mapping):
        raise ValueError("predecessor artifact is missing audit rows")
    clean_rows = audit.get("clean_count")
    harmonic_rows = audit.get("harmonic")
    if not isinstance(clean_rows, list) or len(clean_rows) != expected_counts["clean_count"]:
        raise ValueError("predecessor clean-count rows drifted")
    if not isinstance(harmonic_rows, list) or len(harmonic_rows) != expected_counts["harmonic"]:
        raise ValueError("predecessor harmonic rows drifted")

    results: dict[str, dict[str, Any]] = {}
    for group_name, rows in (("clean", clean_rows), ("harmonic", harmonic_rows)):
        nearest: list[bool] = []
        selected_consistent: list[bool] = []
        truth_exact: list[bool] = []
        no_majority_rows: list[dict[str, Any]] = []
        for index, raw_row in enumerate(rows):
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"{group_name} row {index} must be a mapping")
            dense = _validate_dense_decode(
                raw_row.get("dense"),
                row_name=f"{group_name} row {index}",
            )
            if dense["has_majority"]:
                continue
            counts = dense["expert_counts"]
            reference = int(dense["reference_count"])
            final = int(dense["count"])
            selected = str(dense["selected_expert"])
            expected_truth = (
                int(raw_row["target_count"])
                if group_name == "clean"
                else int(round(256.0 / float(raw_row["fundamental_period_frames"])))
            )
            minimum_distance = min(abs(value - reference) for value in counts)
            nearest.append(final in counts and abs(final - reference) == minimum_distance)
            selected_consistent.append(counts[_EXPERT_INDEX[selected]] == final)
            truth_exact.append(final == expected_truth)
            no_majority_rows.append(
                {
                    "row_index": index,
                    "generated_truth": expected_truth,
                    "reference_count": reference,
                    "expert_counts": counts,
                    "selected_expert": selected,
                    "final_count": final,
                }
            )
        expected_no_majority = expected_counts[f"{group_name}_no_majority"]
        if len(no_majority_rows) != expected_no_majority:
            raise ValueError(f"{group_name} no-majority case count drifted")
        results[group_name] = {
            "total_case_count": len(rows),
            "no_majority_case_count": len(no_majority_rows),
            "fft_nearest_fraction": _fraction(nearest),
            "selected_expert_consistency_fraction": _fraction(selected_consistent),
            "generated_truth_exact_fraction": _fraction(truth_exact),
            "no_majority_rows": no_majority_rows,
        }

    predecessor_checks = predecessor["checks"]
    only_expected_failures = {
        name for name, value in predecessor_checks.items() if value is not True
    } == _EXPECTED_FAILED_CHECKS
    output_quality_checks = all(
        value is True
        for name, value in predecessor_checks.items()
        if name not in _EXPECTED_FAILED_CHECKS
    )
    metrics = {
        "predecessor_only_expected_failures": only_expected_failures,
        "predecessor_output_quality_checks": output_quality_checks,
        "clean_no_majority_case_count": results["clean"]["no_majority_case_count"],
        "harmonic_no_majority_case_count": results["harmonic"]["no_majority_case_count"],
        "clean_no_majority_fft_nearest_fraction": results["clean"][
            "fft_nearest_fraction"
        ],
        "harmonic_no_majority_fft_nearest_fraction": results["harmonic"][
            "fft_nearest_fraction"
        ],
        "clean_no_majority_selected_expert_fraction": results["clean"][
            "selected_expert_consistency_fraction"
        ],
        "harmonic_no_majority_selected_expert_fraction": results["harmonic"][
            "selected_expert_consistency_fraction"
        ],
        "clean_no_majority_generated_truth_fraction": results["clean"][
            "generated_truth_exact_fraction"
        ],
        "harmonic_no_majority_generated_truth_fraction": results["harmonic"][
            "generated_truth_exact_fraction"
        ],
        "candidate_source_files_unchanged": True,
    }
    checks = {
        "predecessor_only_expected_failures": only_expected_failures,
        "predecessor_output_quality_checks": output_quality_checks,
        "clean_no_majority_fft_nearest_fraction": metrics[
            "clean_no_majority_fft_nearest_fraction"
        ]
        >= 1.0,
        "harmonic_no_majority_fft_nearest_fraction": metrics[
            "harmonic_no_majority_fft_nearest_fraction"
        ]
        >= 1.0,
        "clean_no_majority_selected_expert_fraction": metrics[
            "clean_no_majority_selected_expert_fraction"
        ]
        >= 1.0,
        "harmonic_no_majority_selected_expert_fraction": metrics[
            "harmonic_no_majority_selected_expert_fraction"
        ]
        >= 1.0,
        "clean_no_majority_generated_truth_fraction": metrics[
            "clean_no_majority_generated_truth_fraction"
        ]
        >= 1.0,
        "harmonic_no_majority_generated_truth_fraction": metrics[
            "harmonic_no_majority_generated_truth_fraction"
        ]
        >= 1.0,
        "no_majority_case_counts_exact": (
            metrics["clean_no_majority_case_count"]
            == expected_counts["clean_no_majority"]
            and metrics["harmonic_no_majority_case_count"]
            == expected_counts["harmonic_no_majority"]
        ),
        "candidate_source_files_unchanged": True,
    }
    return {"metrics": metrics, "audit": results}, checks


def run_adjudication_gate(
    policy_path: str | Path,
    predecessor_artifact_path: str | Path,
    predecessor_receipt_path: str | Path,
    *,
    source_git_sha: str,
    runner_path: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate the immutable v1 evidence under the corrected v2 protocol."""

    root = Path.cwd() if repo_root is None else Path(repo_root)
    runner = Path(__file__).resolve() if runner_path is None else Path(runner_path)
    paths = {
        "policy": Path(policy_path),
        "predecessor_artifact": Path(predecessor_artifact_path),
        "predecessor_receipt": Path(predecessor_receipt_path),
        "runner": runner,
    }
    identities = {name: _stable_file_identity(path) for name, path in paths.items()}
    policy, policy_sha256, policy_semantic_sha256 = _load_policy(paths["policy"])
    if clean_git_revision(root) != source_git_sha:
        raise ValueError("runtime source Git SHA does not match the clean repository")
    source_files = _validate_candidate_source_files(root)
    predecessor, _ = _validate_predecessor_pair(
        paths["predecessor_artifact"],
        paths["predecessor_receipt"],
        identities=identities,
    )
    gate = policy["synthetic_adjudication_gate"]
    evaluation, checks = _adjudicate_rows(
        predecessor,
        expected_counts=gate["expected_case_counts"],
    )
    passed = all(checks.values())
    for name, path in paths.items():
        if _stable_file_identity(path) != identities[name]:
            raise RuntimeError(f"gate input changed during adjudication: {name}")
    if _validate_candidate_source_files(root) != source_files:
        raise RuntimeError("candidate source changed during adjudication")
    if clean_git_revision(root) != source_git_sha:
        raise RuntimeError("runtime source changed during adjudication")
    return {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "candidate_id": policy["candidate_id"],
        "model_candidate_id": policy["model_candidate_id"],
        "classification": policy["classification"],
        "status": "passed" if passed else "failed",
        "passed": passed,
        "table2_eligible": False,
        "labels_accessed": False,
        "dataset_inputs_accessed": False,
        "checkpoint_inputs_accessed": False,
        "dev84_inputs_accessed": False,
        "test105_inputs_accessed": False,
        "source": {
            "runner_source_git_sha": source_git_sha,
            "runner_sha256": identities["runner"][0],
            "policy_sha256": policy_sha256,
            "policy_semantic_sha256": policy_semantic_sha256,
            "candidate_source_sha256": source_files,
        },
        "inputs": {
            "predecessor_artifact_sha256": identities["predecessor_artifact"][0],
            "predecessor_artifact_bytes": identities["predecessor_artifact"][1],
            "predecessor_receipt_sha256": identities["predecessor_receipt"][0],
            "predecessor_receipt_bytes": identities["predecessor_receipt"][1],
        },
        "protocol_correction": {
            "removed_checks": sorted(_EXPECTED_FAILED_CHECKS),
            "reason": policy["predecessor_gate"]["adjudication_basis"],
            "candidate_algorithm_changed": False,
            "expert_parameters_changed": False,
            "synthetic_cases_changed": False,
            "generated_truth_changed": False,
        },
        "thresholds": gate["thresholds"],
        "metrics": evaluation["metrics"],
        "checks": checks,
        "audit": evaluation["audit"],
        "authorization": {
            "predecessor_failure_verified": True,
            "train337_gate_authorized": passed,
            "dev84_prediction_authorized": False,
            "dev84_scoring_authorized": False,
            "test105_evaluation_authorized": False,
        },
    }


def _receipt_path(output: Path) -> Path:
    return Path(str(output) + ".receipt.json")


def _write_new_regular_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with suppress(OSError):
            path.unlink()
        raise


def _write_artifact_and_receipt(
    output: str | Path,
    payload: Mapping[str, Any],
) -> tuple[Path, str]:
    destination = Path(output)
    receipt_path = _receipt_path(destination)
    if destination.exists() or receipt_path.exists():
        raise FileExistsError("adjudication artifact and receipt destinations must be new")
    artifact = _encoded_json(payload)
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    receipt = {
        "schema_version": 1,
        "artifact_type": _RECEIPT_TYPE,
        "artifact_locator": destination.name,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": len(artifact),
        "artifact_status": payload["status"],
        "candidate_id": payload["candidate_id"],
        "model_candidate_id": payload["model_candidate_id"],
        "policy_sha256": payload["source"]["policy_sha256"],
        "policy_semantic_sha256": payload["source"]["policy_semantic_sha256"],
        "runner_source_git_sha": payload["source"]["runner_source_git_sha"],
        "runner_sha256": payload["source"]["runner_sha256"],
        "predecessor_artifact_sha256": payload["inputs"][
            "predecessor_artifact_sha256"
        ],
        "predecessor_receipt_sha256": payload["inputs"][
            "predecessor_receipt_sha256"
        ],
        "labels_accessed": False,
        "dataset_inputs_accessed": False,
        "checkpoint_inputs_accessed": False,
        "train337_gate_authorized": payload["authorization"][
            "train337_gate_authorized"
        ],
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }
    _write_new_regular_file(destination, artifact)
    try:
        _write_new_regular_file(receipt_path, _encoded_json(receipt))
    except BaseException:
        if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == (
            artifact_sha256
        ):
            destination.unlink()
        raise
    return receipt_path, artifact_sha256


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--predecessor-artifact", type=Path, required=True)
    parser.add_argument("--predecessor-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-git-sha", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    payload = run_adjudication_gate(
        arguments.policy,
        arguments.predecessor_artifact,
        arguments.predecessor_receipt,
        source_git_sha=arguments.source_git_sha,
    )
    receipt_path, artifact_sha256 = _write_artifact_and_receipt(
        arguments.output,
        payload,
    )
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "output": str(arguments.output),
                "receipt": str(receipt_path),
                "status": payload["status"],
                "train337_gate_authorized": payload["authorization"][
                    "train337_gate_authorized"
                ],
                "dev84_prediction_authorized": False,
                "dev84_scoring_authorized": False,
                "test105_evaluation_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
