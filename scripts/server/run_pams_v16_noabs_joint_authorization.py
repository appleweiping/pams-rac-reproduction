"""Join the two frozen v15 probes into a narrow PAMS-v16 NoAbsPE authorization.

This evidence-only gate accepts exactly two immutable artifact/receipt pairs:
the terminal-v15 teacher-identifiability probe and the in-memory PE-off probe.
It reads no dataset, pose, video, label, target, config, checkpoint, or model.

The frozen branch rule is deliberately narrow.  A passing PE-off gate together
with ``alias_confirmed=false`` and a teacher-identifiable current selector at
both anchor budgets authorizes only a fresh seed-2026, train337, NoAbsPE
encoder run through epoch 11.  The identifiability probe's S8 convergence
failure belongs to the rejected mandatory-center branch and does not block
this NoAbsPE branch.  Epoch 12+, SSHead, dev84, and test105 remain unauthorized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_ARTIFACT_TYPE = "pams_v16_noabs_joint_authorization"
_RECEIPT_TYPE = "pams_v16_noabs_joint_authorization_receipt"
_CLASSIFICATION = "inferred target-free evidence-only joint authorization"

_EXPECTED_TEACHER_IDENT_ARTIFACT_SHA256 = (
    "723f678a83492bc43f0239ca64fbe773037c5f4197aceb672fb2a31f387380ac"
)
_EXPECTED_PEOFF_ARTIFACT_SHA256 = "171de419a3ef58c8d86b6fa5b9c2c56b3f3d29db293369fa522f7e16dcdd6bf5"
_EXPECTED_ENCODER_CHECKPOINT_SHA256 = (
    "aa1750154e0ba36d41188cc023bb382db83db9f5c69b1867f310cdc636a48e19"
)
_EXPECTED_ENCODER_PROGRESS_SHA256 = (
    "70226a5f01a3d9e8b688fde53736c397a135c242d8e15ce429a36615552029c4"
)
_EXPECTED_CONFIG_SHA256 = "3e9d641c0e8b4710c3f0542b823b4a51d319765f7e6c72f7f6195f8971fec688"
_EXPECTED_CONFIG_FINGERPRINT = "666f01ece7d179d4ac8c61f9d1dc63aee5e8575c0f79a01d5d3d7e20e9d6242b"
_EXPECTED_POSE_FINGERPRINT = "3dd0388320095796f42aa82d904073b6478b073ed351394ef8d03c30798a0116"
_EXPECTED_SOURCE_EXPORT_RECEIPT_SHA256 = (
    "0235635128894aa24533cde02aa1d5986faaf8f1d1cc39d713eaadca03348588"
)
_EXPECTED_TRAIN337_POSE_CACHE_SET_SHA256 = (
    "f32d718ae922120778f535a6a4f467edba79cafeba90373b3a6a55bf11be6ee2"
)
_EXPECTED_CHECKPOINT_ALGORITHM_SOURCE_GIT_SHA = "df2e7cdfcc91bb436eb99ee10852ddd66ee971e4"
_EXPECTED_EVIDENCE_GATE_CODE_SOURCE_GIT_SHA = "4e60f67cc996a14fd7c1ad60710ec4f65dbcd60e"
_GIT_SHA_PATTERN = re.compile(r"[0-9a-f]{40}")

_PEOFF_CRITERIA = {
    "projected_boundary_share",
    "projected_mode_share",
    "projected_time_scale_eligible_fraction",
    "projected_time_scale_median_relative_error",
    "peoff_post_pe_boundary_share",
    "peoff_post_pe_mode_share",
    "peoff_post_pe_time_scale_eligible_fraction",
    "peoff_post_pe_time_scale_median_relative_error",
    "cross_path_eligible_fraction",
    "cross_path_median_relative_error",
}


@dataclass(frozen=True, slots=True)
class _EvidenceSpec:
    name: str
    artifact_sha256: str
    artifact_type: str
    receipt_type: str
    status: str


@dataclass(frozen=True, slots=True)
class _LoadedEvidence:
    artifact: Mapping[str, Any]
    receipt: Mapping[str, Any]
    artifact_sha256: str
    artifact_bytes: int
    receipt_sha256: str
    receipt_bytes: int


def _evidence_specs() -> tuple[_EvidenceSpec, _EvidenceSpec]:
    return (
        _EvidenceSpec(
            name="teacher_identifiability",
            artifact_sha256=_EXPECTED_TEACHER_IDENT_ARTIFACT_SHA256,
            artifact_type="pams_v15_teacher_identifiability_probe",
            receipt_type="pams_v15_teacher_identifiability_probe_receipt",
            status="v16_short_continuation_rejected",
        ),
        _EvidenceSpec(
            name="peoff",
            artifact_sha256=_EXPECTED_PEOFF_ARTIFACT_SHA256,
            artifact_type=("pams_v15_terminal_encoder_in_memory_peoff_dual_path_probe"),
            receipt_type=("pams_v15_terminal_encoder_in_memory_peoff_dual_path_probe_receipt"),
            status="peoff_gate_pass",
        ),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _encoded_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _path(value: Mapping[str, Any], *fields: str) -> Any:
    current: Any = value
    for field in fields:
        if not isinstance(current, Mapping) or field not in current:
            raise ValueError(f"missing evidence field: {'.'.join(fields)}")
        current = current[field]
    return current


def _read_json(path: Path, *, name: str) -> tuple[Mapping[str, Any], str, int]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an existing regular non-symlink file")
    encoded = path.read_bytes()
    try:
        value = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from exc
    return _mapping(value, name), _sha256_bytes(encoded), len(encoded)


def _load_bound_evidence(
    artifact_path: Path,
    receipt_path: Path,
    spec: _EvidenceSpec,
) -> _LoadedEvidence:
    artifact, artifact_sha256, artifact_bytes = _read_json(
        artifact_path,
        name=f"{spec.name} artifact",
    )
    receipt, receipt_sha256, receipt_bytes = _read_json(
        receipt_path,
        name=f"{spec.name} receipt",
    )
    expected = {
        "artifact_sha256": spec.artifact_sha256,
        "artifact_type": spec.artifact_type,
        "artifact_status": spec.status,
        "receipt_type": spec.receipt_type,
    }
    actual = {
        "artifact_sha256": artifact_sha256,
        "artifact_type": artifact.get("artifact_type"),
        "artifact_status": artifact.get("status"),
        "receipt_type": receipt.get("artifact_type"),
    }
    if actual != expected:
        raise ValueError(
            f"{spec.name} evidence identity mismatch: "
            + json.dumps({"expected": expected, "actual": actual}, sort_keys=True)
        )
    if artifact.get("schema_version") != 1 or receipt.get("schema_version") != 1:
        raise ValueError(f"{spec.name} evidence schema_version must be 1")
    receipt_binding = {
        "artifact_sha256": receipt.get("artifact_sha256"),
        "artifact_bytes": receipt.get("artifact_bytes"),
        "artifact_status": receipt.get("artifact_status"),
        "artifact_locator": receipt.get("artifact_locator"),
    }
    expected_binding = {
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": artifact_bytes,
        "artifact_status": spec.status,
        "artifact_locator": artifact_path.name,
    }
    if receipt_binding != expected_binding:
        raise ValueError(
            f"{spec.name} receipt binding mismatch: "
            + json.dumps(
                {"expected": expected_binding, "actual": receipt_binding},
                sort_keys=True,
            )
        )
    return _LoadedEvidence(
        artifact=artifact,
        receipt=receipt,
        artifact_sha256=artifact_sha256,
        artifact_bytes=artifact_bytes,
        receipt_sha256=receipt_sha256,
        receipt_bytes=receipt_bytes,
    )


def _true_authorization_paths(value: Any, prefix: str = "") -> tuple[str, ...]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).endswith("_authorized") and child is True:
                found.append(path)
            found.extend(_true_authorization_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_true_authorization_paths(child, f"{prefix}[{index}]"))
    return tuple(found)


def _validate_code_binding(evidence: _LoadedEvidence, *, name: str) -> None:
    code_files = _mapping(
        _path(evidence.artifact, "inputs", "code_files_sha256"),
        f"{name} code_files_sha256",
    )
    commitment = _path(
        evidence.artifact,
        "inputs",
        "code_files_sha256_commitment",
    )
    if commitment != _sha256_json(code_files):
        raise ValueError(f"{name} code-files commitment is invalid")
    if evidence.receipt.get("code_files_sha256_commitment") != commitment:
        raise ValueError(f"{name} receipt does not bind the code-files commitment")
    runner_key = (
        "teacher_identifiability_probe_runner"
        if name == "teacher_identifiability"
        else "v15_peoff_probe_runner"
    )
    if _path(evidence.artifact, "inputs", f"{runner_key}_sha256") != code_files.get(runner_key):
        raise ValueError(f"{name} runner SHA is not code-map bound")


def _validate_shared_identities(
    teacher: _LoadedEvidence,
    peoff: _LoadedEvidence,
) -> dict[str, Any]:
    expected_artifact_fields = {
        "encoder_checkpoint_sha256": _EXPECTED_ENCODER_CHECKPOINT_SHA256,
        "encoder_progress_sha256": _EXPECTED_ENCODER_PROGRESS_SHA256,
        "config_sha256": _EXPECTED_CONFIG_SHA256,
        "config_fingerprint": _EXPECTED_CONFIG_FINGERPRINT,
        "pose_fingerprint": _EXPECTED_POSE_FINGERPRINT,
        "source_export_receipt_sha256": _EXPECTED_SOURCE_EXPORT_RECEIPT_SHA256,
        "train337_pose_cache_set_sha256": (_EXPECTED_TRAIN337_POSE_CACHE_SET_SHA256),
        "checkpoint_algorithm_source_git_sha": (_EXPECTED_CHECKPOINT_ALGORITHM_SOURCE_GIT_SHA),
        "gate_code_source_git_sha": _EXPECTED_EVIDENCE_GATE_CODE_SOURCE_GIT_SHA,
    }
    receipt_fields = {
        "encoder_checkpoint_sha256",
        "encoder_progress_sha256",
        "config_sha256",
        "source_export_receipt_sha256",
        "train337_pose_cache_set_sha256",
        "checkpoint_algorithm_source_git_sha",
        "gate_code_source_git_sha",
    }
    for name, evidence in (
        ("teacher_identifiability", teacher),
        ("peoff", peoff),
    ):
        inputs = _mapping(evidence.artifact.get("inputs"), f"{name} inputs")
        actual = {key: inputs.get(key) for key in expected_artifact_fields}
        if actual != expected_artifact_fields:
            raise ValueError(
                f"{name} frozen input identities mismatch: "
                + json.dumps(
                    {"expected": expected_artifact_fields, "actual": actual},
                    sort_keys=True,
                )
            )
        for key in receipt_fields:
            if evidence.receipt.get(key) != expected_artifact_fields[key]:
                raise ValueError(f"{name} receipt identity mismatch for {key}")
        if (
            evidence.artifact.get("protocol") != "ucfrep_526"
            or evidence.artifact.get("seed") != 2026
            or inputs.get("train337_video_total") != 337
            or inputs.get("read_only_post_run_identity_verified") is not True
        ):
            raise ValueError(f"{name} protocol/seed/train337 binding is invalid")
        _validate_code_binding(evidence, name=name)

    teacher_inputs = _mapping(teacher.artifact["inputs"], "teacher inputs")
    peoff_inputs = _mapping(peoff.artifact["inputs"], "peoff inputs")
    for key in ("train337_video_ids_sha256", "encoder_provenance"):
        if teacher_inputs.get(key) != peoff_inputs.get(key):
            raise ValueError(f"evidence artifacts do not share {key}")
    provenance = _mapping(
        teacher_inputs.get("encoder_provenance"),
        "shared encoder provenance",
    )
    training_ids = provenance.get("training_video_ids")
    if (
        provenance.get("protocol") != "ucfrep_526"
        or provenance.get("source_git_sha") != _EXPECTED_CHECKPOINT_ALGORITHM_SOURCE_GIT_SHA
        or provenance.get("pose_cache_set_sha256") != _EXPECTED_TRAIN337_POSE_CACHE_SET_SHA256
        or provenance.get("pose_fingerprint") != _EXPECTED_POSE_FINGERPRINT
        or not isinstance(training_ids, list)
        or len(training_ids) != 337
        or len(set(training_ids)) != 337
    ):
        raise ValueError("shared encoder provenance is invalid")
    return {
        **expected_artifact_fields,
        "protocol": "ucfrep_526",
        "seed": 2026,
        "train337_video_total": 337,
        "train337_video_ids_sha256": teacher_inputs["train337_video_ids_sha256"],
        "encoder_provenance_sha256": _sha256_json(provenance),
    }


def _ident_read_only(artifact: Mapping[str, Any]) -> bool:
    audit = _mapping(
        _path(artifact, "read_only_verification"),
        "ident read_only_verification",
    )
    return (
        audit.get("all_file_inputs_unchanged") is True
        and audit.get("train337_pose_cache_set_sha256_unchanged") is True
        and audit.get("checkpoint_algorithm_source_git_sha_unchanged") is True
        and audit.get("model_state_sha256_before") == audit.get("model_state_sha256_after")
        and audit.get("model_or_optimizer_state_updated") is False
        and audit.get("training_steps_executed") == 0
        and audit.get("pose_cache_write_operations") == 0
    )


def _peoff_read_only(artifact: Mapping[str, Any]) -> bool:
    audit = _mapping(
        _path(artifact, "read_only_verification"),
        "PE-off read_only_verification",
    )
    mutation = _mapping(
        _path(artifact, "in_memory_mutation_audit"),
        "PE-off mutation audit",
    )
    return (
        audit.get("all_file_inputs_unchanged") is True
        and audit.get("checkpoint_file_sha256_before_after_unchanged") is True
        and audit.get("train337_pose_cache_set_sha256_unchanged") is True
        and audit.get("model_tensor_state_sha256_before_after_unchanged") is True
        and audit.get("position_encoding_mode_restored") is True
        and audit.get("model_or_optimizer_tensor_state_updated") is False
        and audit.get("training_steps_executed") == 0
        and audit.get("optimizer_created") is False
        and audit.get("pose_cache_write_operations") == 0
        and mutation.get("mode_restored") is True
        and mutation.get("model_tensor_state_unchanged") is True
        and mutation.get("model_state_sha256_before")
        == mutation.get("model_state_sha256_after_baseline")
        == mutation.get("model_state_sha256_after_restore")
    )


def _projected_identity_exact(artifact: Mapping[str, Any]) -> bool:
    audit = _mapping(
        _path(artifact, "projected_path_identity_audit"),
        "projected path identity audit",
    )
    variants = _mapping(audit.get("variants"), "projected identity variants")
    if set(variants) != {"original", "time_scale_0.5", "time_scale_0.75"}:
        return False
    variant_pass = all(
        isinstance(row, Mapping)
        and row.get("projected_period_confidence_exact") is True
        and row.get("projected_tensor_sha256_exact") is True
        and row.get("projected_tensor_drift_within_threshold") is True
        and isinstance(row.get("projected_tensor_max_abs_drift"), int | float)
        and isinstance(row.get("projected_tensor_max_abs_drift_threshold"), int | float)
        and float(row["projected_tensor_max_abs_drift"])
        <= float(row["projected_tensor_max_abs_drift_threshold"])
        for row in variants.values()
    )
    return (
        audit.get("all_variants_projected_period_confidence_exact") is True
        and audit.get("all_variants_projected_tensor_sha256_exact") is True
        and isinstance(audit.get("overall_projected_tensor_max_abs_drift"), int | float)
        and isinstance(
            audit.get("overall_projected_tensor_max_abs_drift_threshold"),
            int | float,
        )
        and float(audit["overall_projected_tensor_max_abs_drift"])
        <= float(audit["overall_projected_tensor_max_abs_drift_threshold"])
        and variant_pass
    )


def _joint_decision(
    teacher: Mapping[str, Any],
    teacher_receipt: Mapping[str, Any],
    peoff: Mapping[str, Any],
    peoff_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    operational = _mapping(
        _path(teacher, "operational_gate"),
        "ident operational gate",
    )
    operational_criteria = _mapping(
        operational.get("criteria"),
        "ident operational criteria",
    )
    classifications = {
        budget: _mapping(
            _path(
                teacher,
                "anchor_budget_measurements",
                budget,
                "classification",
            ),
            f"ident classification {budget}",
        )
        for budget in ("32", "64")
    }
    convergence = _mapping(_path(teacher, "convergence"), "ident convergence")
    convergence_criteria = _mapping(
        convergence.get("criteria"),
        "ident convergence criteria",
    )
    failed_convergence = {
        name for name, passed in convergence_criteria.items() if passed is not True
    }
    peoff_gate = _mapping(_path(peoff, "gate"), "PE-off gate")
    peoff_criteria = _mapping(peoff_gate.get("criteria"), "PE-off criteria")
    authorization_conflicts = (
        _true_authorization_paths(teacher)
        + _true_authorization_paths(teacher_receipt)
        + _true_authorization_paths(peoff)
        + _true_authorization_paths(peoff_receipt)
    )
    criteria = {
        "ident_operational_pass": (
            operational.get("pass") is True
            and bool(operational_criteria)
            and all(value is True for value in operational_criteria.values())
        ),
        "ident_alias_not_confirmed_at_32_and_64": all(
            row.get("alias_confirmed") is False for row in classifications.values()
        ),
        "ident_current_teacher_identifiable_at_32_and_64": all(
            row.get("current_teacher_state") == "identifiable" for row in classifications.values()
        ),
        "ident_classification_unchanged_across_budgets": (
            convergence.get("classification_unchanged") is True
            and convergence_criteria.get("classification_unchanged") is True
        ),
        "ident_convergence_expected_only_s8_failure": (
            convergence.get("pass") is False and failed_convergence == {"s8_absolute_difference"}
        ),
        "peoff_all_ten_gate_pass": (
            peoff_gate.get("peoff_gate_pass") is True
            and peoff_gate.get("all_ten_diagnostic_criteria_pass") is True
            and set(peoff_criteria) == _PEOFF_CRITERIA
            and all(
                isinstance(value, Mapping) and value.get("pass") is True
                for value in peoff_criteria.values()
            )
        ),
        "peoff_projected_path_identity_exact": _projected_identity_exact(peoff),
        "ident_inputs_and_model_read_only": _ident_read_only(teacher),
        "peoff_inputs_and_model_read_only": _peoff_read_only(peoff),
        "prior_evidence_has_no_true_authorization_flags": not authorization_conflicts,
    }
    authorized = all(criteria.values())
    return {
        "truth_table": (
            "Authorize only a fresh seed2026 train337 NoAbsPE encoder through "
            "epoch 11 when the ident probe is operational, alias is false and "
            "the current teacher is identifiable at budgets 32 and 64, and "
            "the PE-off probe passes all ten criteria with exact projected-path "
            "identity and read-only state."
        ),
        "criteria": criteria,
        "prior_true_authorization_flag_paths": list(authorization_conflicts),
        "ident_s8_convergence_failure_is_noabs_blocker": False,
        "ident_convergence_scope": "mandatory-center branch only",
        "fresh_seed2026_train337_noabspe_encoder_through_epoch11_authorized": (authorized),
        "authorized_scope": (
            {
                "protocol": "ucfrep_526",
                "seed": 2026,
                "training_video_total": 337,
                "stage": "encoder",
                "position_encoding_mode": "none",
                "initialization": "fresh",
                "maximum_completed_epoch": 11,
            }
            if authorized
            else None
        ),
        "mandatory_center_training_authorized": False,
        "epoch12_or_later_training_authorized": False,
        "sshead_training_authorized": False,
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
    }


def _gate_code_source_revision() -> str:
    revision = os.environ.get("PAMS_GATE_CODE_SOURCE_REVISION", "").strip()
    if not _GIT_SHA_PATTERN.fullmatch(revision):
        raise RuntimeError(
            "PAMS_GATE_CODE_SOURCE_REVISION must be a 40-character lowercase Git SHA"
        )
    return revision


def _input_identities(paths: Mapping[str, Path]) -> dict[str, tuple[str, int]]:
    identities: dict[str, tuple[str, int]] = {}
    for name, path in paths.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"{name} must be an existing regular non-symlink file")
        encoded = path.read_bytes()
        identities[name] = (_sha256_bytes(encoded), len(encoded))
    return identities


def run_joint_authorization(
    teacher_ident_artifact_path: str | Path,
    teacher_ident_receipt_path: str | Path,
    peoff_artifact_path: str | Path,
    peoff_receipt_path: str | Path,
) -> dict[str, Any]:
    """Validate the two exact evidence pairs and apply the frozen joint rule."""

    paths = {
        "teacher_ident_artifact": Path(teacher_ident_artifact_path),
        "teacher_ident_receipt": Path(teacher_ident_receipt_path),
        "peoff_artifact": Path(peoff_artifact_path),
        "peoff_receipt": Path(peoff_receipt_path),
        "joint_authorization_runner": Path(__file__),
    }
    identities_before = _input_identities(paths)
    teacher_spec, peoff_spec = _evidence_specs()
    teacher = _load_bound_evidence(
        paths["teacher_ident_artifact"],
        paths["teacher_ident_receipt"],
        teacher_spec,
    )
    peoff = _load_bound_evidence(
        paths["peoff_artifact"],
        paths["peoff_receipt"],
        peoff_spec,
    )
    shared = _validate_shared_identities(teacher, peoff)
    decision = _joint_decision(
        teacher.artifact,
        teacher.receipt,
        peoff.artifact,
        peoff.receipt,
    )
    identities_after = _input_identities(paths)
    if identities_after != identities_before:
        raise RuntimeError("joint-authorization inputs or runner changed during validation")
    gate_revision = _gate_code_source_revision()
    evidence = {
        "teacher_identifiability": {
            "artifact_sha256": teacher.artifact_sha256,
            "artifact_bytes": teacher.artifact_bytes,
            "receipt_sha256": teacher.receipt_sha256,
            "receipt_bytes": teacher.receipt_bytes,
            "artifact_type": teacher.artifact["artifact_type"],
            "receipt_type": teacher.receipt["artifact_type"],
            "artifact_status": teacher.artifact["status"],
        },
        "peoff": {
            "artifact_sha256": peoff.artifact_sha256,
            "artifact_bytes": peoff.artifact_bytes,
            "receipt_sha256": peoff.receipt_sha256,
            "receipt_bytes": peoff.receipt_bytes,
            "artifact_type": peoff.artifact["artifact_type"],
            "receipt_type": peoff.receipt["artifact_type"],
            "artifact_status": peoff.artifact["status"],
        },
    }
    payload = {
        "schema_version": 1,
        "artifact_type": _ARTIFACT_TYPE,
        "status": (
            "noabs_epoch11_authorized"
            if decision["fresh_seed2026_train337_noabspe_encoder_through_epoch11_authorized"]
            else "noabs_epoch11_rejected"
        ),
        "classification": _CLASSIFICATION,
        "disclosed_by_pams_authors": False,
        "eligible_for_paper_table": False,
        "label_firewall": {
            "accepted_inputs": [
                "exact_teacher_identifiability_artifact_and_receipt",
                "exact_peoff_artifact_and_receipt",
            ],
            "dataset_video_pose_config_checkpoint_or_model_argument_supported": False,
            "development_or_sealed_test_argument_supported": False,
            "action_class_or_repetition_count_argument_supported": False,
            "training_interface_supported": False,
        },
        "inputs": {
            **{f"{name}_sha256": identity[0] for name, identity in identities_before.items()},
            **{f"{name}_bytes": identity[1] for name, identity in identities_before.items()},
            "joint_gate_code_source_git_sha": gate_revision,
            "evidence_gate_code_source_git_sha": (_EXPECTED_EVIDENCE_GATE_CODE_SOURCE_GIT_SHA),
            "evidence": evidence,
            "shared_checkpoint_and_data_identities": shared,
            "shared_checkpoint_and_data_identities_sha256": _sha256_json(shared),
            "read_only_post_run_identity_verified": True,
        },
        "decision": decision,
        "scientific_disposition": {
            "selected_branch": "fresh_noabspe_encoder",
            "rejected_branch": "mandatory_teacher_center",
            "ident_convergence_observation": (
                "Only S8 convergence failed; alias=false and current teacher "
                "identifiable were unchanged at budgets 32 and 64."
            ),
            "why_s8_does_not_block": (
                "Convergence was preregistered for the mandatory-center branch. "
                "The original PE-off truth table requires PE-off pass plus "
                "alias=false and a current identifiable teacher for NoAbsPE."
            ),
        },
        "read_only_verification": {
            "all_four_evidence_files_unchanged": True,
            "joint_runner_sha256_unchanged": True,
            "data_model_or_training_state_accessed": False,
            "files_written_before_return": 0,
        },
    }
    return payload


def _receipt_path(output: Path) -> Path:
    return Path(str(output) + ".receipt.json")


def _write_new_regular_file(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())


def _write_artifact_and_receipt(
    output: Path,
    payload: Mapping[str, Any],
) -> tuple[Path, str]:
    receipt_path = _receipt_path(output)
    if output.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite joint artifact or receipt")
    artifact = _encoded_json(payload)
    artifact_sha256 = _sha256_bytes(artifact)
    _write_new_regular_file(output, artifact)
    receipt = {
        "schema_version": 1,
        "artifact_type": _RECEIPT_TYPE,
        "artifact_locator": output.name,
        "artifact_sha256": artifact_sha256,
        "artifact_bytes": len(artifact),
        "artifact_status": payload["status"],
        "fresh_seed2026_train337_noabspe_encoder_through_epoch11_authorized": (
            payload["decision"][
                "fresh_seed2026_train337_noabspe_encoder_through_epoch11_authorized"
            ]
        ),
        "maximum_authorized_completed_epoch": 11,
        "epoch12_or_later_training_authorized": False,
        "sshead_training_authorized": False,
        "dev84_prediction_authorized": False,
        "dev84_scoring_authorized": False,
        "test105_evaluation_authorized": False,
        "teacher_ident_artifact_sha256": payload["inputs"]["teacher_ident_artifact_sha256"],
        "peoff_artifact_sha256": payload["inputs"]["peoff_artifact_sha256"],
        "joint_authorization_runner_sha256": payload["inputs"]["joint_authorization_runner_sha256"],
        "joint_gate_code_source_git_sha": payload["inputs"]["joint_gate_code_source_git_sha"],
        "evidence_gate_code_source_git_sha": payload["inputs"]["evidence_gate_code_source_git_sha"],
        "shared_checkpoint_and_data_identities_sha256": payload["inputs"][
            "shared_checkpoint_and_data_identities_sha256"
        ],
    }
    _write_new_regular_file(receipt_path, _encoded_json(receipt))
    return receipt_path, artifact_sha256


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-ident-artifact", type=Path, required=True)
    parser.add_argument("--teacher-ident-receipt", type=Path, required=True)
    parser.add_argument("--peoff-artifact", type=Path, required=True)
    parser.add_argument("--peoff-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    receipt_path = _receipt_path(arguments.output)
    if arguments.output.exists() or receipt_path.exists():
        raise FileExistsError("joint output and receipt destinations must both be new")
    payload = run_joint_authorization(
        arguments.teacher_ident_artifact,
        arguments.teacher_ident_receipt,
        arguments.peoff_artifact,
        arguments.peoff_receipt,
    )
    receipt_path, artifact_sha256 = _write_artifact_and_receipt(
        arguments.output,
        payload,
    )
    authorized = payload["decision"][
        "fresh_seed2026_train337_noabspe_encoder_through_epoch11_authorized"
    ]
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "output": str(arguments.output),
                "receipt": str(receipt_path),
                "noabs_epoch11_authorized": authorized,
                "epoch12_or_later_training_authorized": False,
                "sshead_training_authorized": False,
                "dev84_prediction_authorized": False,
                "test105_evaluation_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0 if authorized else 3


if __name__ == "__main__":
    raise SystemExit(main())
