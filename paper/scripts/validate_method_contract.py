#!/usr/bin/env python3
"""Validate method transitions, source digests, and the supervision firewall."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml


REQUIRED_COMPONENT_FIELDS = {
    "id", "status", "sync_required", "evidence_class", "sources",
    "implementation_anchors", "test_artifacts", "closure_rule",
}
REQUIRED_ANCHOR_FIELDS = {"path", "symbol", "commit", "sha256", "verdict"}
REQUIRED_TEST_FIELDS = {"id", "path", "commit", "sha256", "verdict"}
REQUIRED_FIREWALL = {
    "dataset_loaders", "pseudo_label_generation", "losses",
    "router_inputs_and_targets", "hyperparameter_tuning",
    "checkpoint_selection", "early_stopping", "calibration",
    "prediction_and_inference",
}
COMPONENT_STATUSES = {"BLOCKED", "CONFIRMED"}
VERDICTS = {"PASS", "BLOCKED", "FAIL"}
SHA256_RE = re.compile(r"(?:sha256:)?([0-9a-f]{64})$", re.I)
COMMIT_RE = re.compile(r"[0-9a-f]{40}$", re.I)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_digest(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    match = SHA256_RE.fullmatch(value.strip())
    return match.group(1).lower() if match else None


def safe_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts:
        return None
    candidate = (root / rel).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate


def check_file_binding(
    root: Path, record: dict[str, Any], label: str, errors: list[str]
) -> Path | None:
    path = safe_path(root, record.get("path"))
    digest = normalized_digest(record.get("sha256"))
    if path is None:
        errors.append(f"{label}: unsafe or missing path")
        return None
    if not path.is_file():
        errors.append(f"{label}: file does not exist: {record.get('path')}")
        return None
    if digest is None:
        errors.append(f"{label}: invalid SHA-256")
    elif sha256(path) != digest:
        errors.append(f"{label}: SHA-256 mismatch")
    return path


def git_text(repository: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repository), *args], stderr=subprocess.STDOUT
    ).decode("utf-8", errors="replace").strip()


def collaborator_manifest_index(
    package_root: Path, freeze: dict[str, Any], errors: list[str]
) -> tuple[Path | None, Path | None, dict[str, str]]:
    """Validate the collaborator repository/manifest and return its artifact index."""
    raw_root = freeze.get("repository_root")
    if not isinstance(raw_root, str) or not raw_root.strip():
        errors.append("collaborator_freeze: repository_root missing")
        return None, None, {}
    collaborator_root = Path(raw_root).expanduser()
    if not collaborator_root.is_absolute() or not collaborator_root.is_dir():
        errors.append("collaborator_freeze: repository_root must be an existing absolute directory")
        return None, None, {}
    collaborator_root = collaborator_root.resolve()
    commit = freeze.get("commit")
    try:
        top = Path(git_text(collaborator_root, "rev-parse", "--show-toplevel")).resolve()
        head = git_text(collaborator_root, "rev-parse", "HEAD")
        dirty = git_text(collaborator_root, "status", "--porcelain")
    except (OSError, subprocess.CalledProcessError) as exc:
        errors.append(f"collaborator_freeze: repository_root is not verifiable: {type(exc).__name__}")
        return collaborator_root, None, {}
    if top != collaborator_root:
        errors.append("collaborator_freeze: repository_root is not the exact Git top level")
    if head != commit:
        errors.append("collaborator_freeze: repository_root HEAD differs from frozen commit")
    if dirty:
        errors.append("collaborator_freeze: repository_root is dirty at validation time")

    raw_manifest = freeze.get("manifest_path")
    if not isinstance(raw_manifest, str) or not raw_manifest.strip():
        errors.append("collaborator_freeze: manifest_path missing")
        return collaborator_root, None, {}
    candidate = Path(raw_manifest).expanduser()
    manifest_path = candidate.resolve() if candidate.is_absolute() else (package_root / candidate).resolve()
    try:
        manifest_path.relative_to(collaborator_root)
    except ValueError:
        pass
    else:
        errors.append("collaborator_freeze: manifest_path must be an external/package sidecar, not a self-referential file inside repository_root")
        return collaborator_root, None, {}
    if not manifest_path.is_file():
        errors.append("collaborator_freeze: manifest_path does not exist")
        return collaborator_root, manifest_path, {}
    declared_manifest_digest = normalized_digest(freeze.get("manifest_sha256"))
    if declared_manifest_digest is None or sha256(manifest_path) != declared_manifest_digest:
        errors.append("collaborator_freeze: manifest SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"collaborator_freeze: manifest unreadable: {type(exc).__name__}")
        return collaborator_root, manifest_path, {}
    if manifest.get("commit") != commit:
        errors.append("collaborator_freeze: manifest commit differs from frozen commit")
    manifest_root = manifest.get("repository_root")
    if not isinstance(manifest_root, str) or Path(manifest_root).expanduser().resolve() != collaborator_root:
        errors.append("collaborator_freeze: manifest repository_root mismatch")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or not rows:
        errors.append("collaborator_freeze: manifest artifacts must be a non-empty list")
        return collaborator_root, manifest_path, {}
    index: dict[str, str] = {}
    for row_number, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            errors.append(f"collaborator_freeze: manifest artifact[{row_number}] fields invalid")
            continue
        rel = row.get("path")
        digest = normalized_digest(row.get("sha256"))
        path = safe_path(collaborator_root, rel)
        if path is None or not isinstance(rel, str) or Path(rel).as_posix() != rel:
            errors.append(f"collaborator_freeze: manifest artifact[{row_number}] path invalid")
            continue
        if rel in index:
            errors.append(f"collaborator_freeze: duplicate manifest artifact {rel}")
            continue
        if digest is None or not path.is_file() or sha256(path) != digest:
            errors.append(f"collaborator_freeze: manifest artifact digest mismatch: {rel}")
            continue
        index[rel] = digest
    return collaborator_root, manifest_path, index


def check_collaborator_binding(
    collaborator_root: Path | None,
    manifest_index: dict[str, str],
    record: dict[str, Any],
    frozen_commit: Any,
    label: str,
    errors: list[str],
) -> None:
    if collaborator_root is None:
        errors.append(f"{label}: collaborator repository is unavailable")
        return
    path = check_file_binding(collaborator_root, record, label, errors)
    rel = record.get("path")
    digest = normalized_digest(record.get("sha256"))
    if record.get("commit") != frozen_commit:
        errors.append(f"{label}: commit differs from collaborator freeze")
    if not isinstance(rel, str) or manifest_index.get(rel) != digest:
        errors.append(f"{label}: path/digest is absent from collaborator manifest")
    if path is not None and isinstance(rel, str):
        try:
            committed = subprocess.check_output(
                ["git", "-C", str(collaborator_root), "show", f"{frozen_commit}:{rel}"],
                stderr=subprocess.STDOUT,
            )
        except (OSError, subprocess.CalledProcessError):
            # Test/log artifacts may deliberately be untracked but must still be
            # frozen in the collaborator manifest. Source anchors, identified by
            # a non-empty symbol, must exist at the commit itself.
            if record.get("symbol"):
                errors.append(f"{label}: source anchor is absent at frozen Git commit")
        else:
            if hashlib.sha256(committed).hexdigest() != digest:
                errors.append(f"{label}: Git object digest differs from collaborator manifest")


def strip_tex_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paper_dir = args.paper_dir.resolve()
    repo_root = paper_dir.parent
    contract_path = paper_dir / "evidence/method_contract.yaml"
    data = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    transition = data.get("transition_schema", {})
    if set(transition.get("allowed_component_statuses", [])) != COMPONENT_STATUSES:
        errors.append("transition_schema: component-status enum mismatch")
    if set(transition.get("allowed_verdicts", [])) != VERDICTS:
        errors.append("transition_schema: verdict enum mismatch")
    if set(transition.get("anchor_fields", [])) != REQUIRED_ANCHOR_FIELDS:
        errors.append("transition_schema: anchor fields mismatch")
    if set(transition.get("test_fields", [])) != REQUIRED_TEST_FIELDS:
        errors.append("transition_schema: test fields mismatch")

    freeze = data.get("collaborator_freeze", {})
    freeze_status = freeze.get("status")
    freeze_commit = freeze.get("commit")
    collaborator_root: Path | None = None
    collaborator_manifest_path: Path | None = None
    collaborator_artifacts: dict[str, str] = {}
    if freeze_status not in COMPONENT_STATUSES:
        errors.append("collaborator_freeze: invalid status")
    if freeze_status == "CONFIRMED":
        if not isinstance(freeze_commit, str) or not COMMIT_RE.fullmatch(freeze_commit):
            errors.append("collaborator_freeze: invalid confirmed commit")
        collaborator_root, collaborator_manifest_path, collaborator_artifacts = collaborator_manifest_index(
            repo_root, freeze, errors
        )
    elif any(freeze.get(key) is not None for key in (
        "commit", "repository_root", "manifest_path", "manifest_sha256"
    )):
        errors.append("collaborator_freeze: BLOCKED freeze must keep bindings null")

    known_sources = data.get("known_sources", {})
    if not isinstance(known_sources, dict) or not known_sources:
        errors.append("known_sources: empty or invalid")
    else:
        for source_id, record in known_sources.items():
            if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
                errors.append(f"known_sources/{source_id}: invalid fields")
                continue
            check_file_binding(repo_root, record, f"known_sources/{source_id}", errors)

    modules = list(data.get("modules", []))
    objective_components = list(data.get("training_objective", {}).get("components", []))
    components = modules + objective_components
    module_orders = [module.get("order") for module in modules]
    if module_orders != list(range(1, len(modules) + 1)):
        errors.append("modules: order must be unique, contiguous, and declared order")
    ids = [component.get("id") for component in components]
    if None in ids or len(ids) != len(set(ids)):
        errors.append("components: missing or duplicate id")

    for component in components:
        cid = component.get("id", "<missing>")
        missing = REQUIRED_COMPONENT_FIELDS - set(component)
        if missing:
            errors.append(f"{cid}: missing component fields {sorted(missing)}")
            continue
        if component.get("status") not in COMPONENT_STATUSES:
            errors.append(f"{cid}: invalid status")
        if not isinstance(component.get("sync_required"), bool):
            errors.append(f"{cid}: sync_required must be boolean")
        sources = component.get("sources")
        if not isinstance(sources, list) or not sources:
            errors.append(f"{cid}: sources must be non-empty")
        else:
            unresolved = sorted(set(sources) - set(known_sources))
            if unresolved:
                errors.append(f"{cid}: unresolved sources {unresolved}")

        anchors = component.get("implementation_anchors", [])
        tests = component.get("test_artifacts", [])
        pass_anchors: list[dict[str, Any]] = []
        for index, anchor in enumerate(anchors):
            label = f"{cid}/anchor/{index}"
            if set(anchor) != REQUIRED_ANCHOR_FIELDS:
                errors.append(f"{label}: invalid fields")
                continue
            if anchor.get("verdict") not in VERDICTS:
                errors.append(f"{label}: invalid verdict")
            populated = any(anchor.get(key) is not None for key in ("path", "symbol", "commit", "sha256"))
            if populated:
                if component.get("status") == "CONFIRMED":
                    check_collaborator_binding(
                        collaborator_root, collaborator_artifacts, anchor, freeze_commit, label, errors
                    )
                else:
                    check_file_binding(repo_root, anchor, label, errors)
                if not anchor.get("symbol"):
                    errors.append(f"{label}: symbol missing")
                if not isinstance(anchor.get("commit"), str) or not COMMIT_RE.fullmatch(anchor["commit"]):
                    errors.append(f"{label}: invalid commit")
            if anchor.get("verdict") == "PASS":
                pass_anchors.append(anchor)

        all_tests_pass = bool(tests)
        for index, test in enumerate(tests):
            label = f"{cid}/test/{index}"
            if set(test) != REQUIRED_TEST_FIELDS:
                errors.append(f"{label}: invalid fields")
                all_tests_pass = False
                continue
            if test.get("verdict") not in VERDICTS:
                errors.append(f"{label}: invalid verdict")
            populated = any(test.get(key) is not None for key in ("path", "commit", "sha256"))
            if populated:
                if component.get("status") == "CONFIRMED":
                    check_collaborator_binding(
                        collaborator_root, collaborator_artifacts, test, freeze_commit, label, errors
                    )
                else:
                    check_file_binding(repo_root, test, label, errors)
                if not isinstance(test.get("commit"), str) or not COMMIT_RE.fullmatch(test["commit"]):
                    errors.append(f"{label}: invalid commit")
            all_tests_pass = all_tests_pass and test.get("verdict") == "PASS"

        if component.get("status") == "CONFIRMED":
            pass_anchor_at_freeze = any(anchor.get("commit") == freeze_commit for anchor in pass_anchors)
            tests_at_freeze = all(test.get("commit") == freeze_commit for test in tests)
            if freeze_status != "CONFIRMED":
                errors.append(f"{cid}: CONFIRMED before collaborator freeze")
            if not pass_anchor_at_freeze or not all_tests_pass or not tests_at_freeze:
                errors.append(f"{cid}: CONFIRMED without one frozen PASS anchor and all frozen PASS tests")

    supervision = data.get("task", {}).get("supervision_claim", {})
    canonical = "without person-wise count or period supervision in the intended counting objective"
    if supervision.get("allowed_text") != canonical:
        errors.append("supervision claim must equal the canonical allowed wording")
    if set(supervision.get("firewall_scope", [])) != REQUIRED_FIREWALL:
        errors.append("supervision firewall scope must exactly match the nine required scopes")
    firewall_path = safe_path(repo_root, supervision.get("required_artifact"))
    firewall: dict[str, Any] = {}
    if firewall_path is None or not firewall_path.is_file():
        errors.append("supervision firewall artifact is missing or unsafe")
    else:
        try:
            firewall = json.loads(firewall_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"supervision firewall unreadable: {exc}")
    if firewall:
        if firewall.get("allowed_text") != canonical:
            errors.append("supervision firewall allowed_text mismatch")
        if firewall.get("status") not in {"BLOCKED", "PASS", "FAIL"}:
            errors.append("supervision firewall invalid status")
        scope_rows = firewall.get("scope_audits", [])
        scopes = [row.get("scope") for row in scope_rows if isinstance(row, dict)]
        if len(scopes) != len(set(scopes)) or set(scopes) != REQUIRED_FIREWALL:
            errors.append("supervision firewall scopes are missing or duplicated")
        for row in scope_rows:
            if not isinstance(row, dict) or set(row) != {"scope", "source_anchors", "verdict"}:
                errors.append("supervision firewall scope row has invalid fields")
                continue
            if row.get("verdict") not in {"PASS", "BLOCKED", "FAIL"}:
                errors.append(f"supervision firewall/{row.get('scope')}: invalid verdict")
            anchors = row.get("source_anchors", [])
            if not isinstance(anchors, list):
                errors.append(f"supervision firewall/{row.get('scope')}: anchors must be a list")
                continue
            for index, anchor in enumerate(anchors):
                label = f"supervision firewall/{row.get('scope')}/{index}"
                if set(anchor) != REQUIRED_ANCHOR_FIELDS:
                    errors.append(f"{label}: invalid anchor fields")
                    continue
                if firewall.get("status") == "PASS":
                    check_collaborator_binding(
                        collaborator_root, collaborator_artifacts, anchor, freeze_commit, label, errors
                    )
                else:
                    check_file_binding(repo_root, anchor, label, errors)
                if anchor.get("commit") != freeze_commit or anchor.get("verdict") != "PASS":
                    errors.append(f"{label}: not PASS at collaborator freeze")
        if firewall.get("status") == "PASS":
            if freeze_status != "CONFIRMED" or firewall.get("collaborator_commit") != freeze_commit:
                errors.append("supervision firewall PASS is not bound to confirmed collaborator freeze")
            if any(row.get("verdict") != "PASS" or not row.get("source_anchors") for row in scope_rows):
                errors.append("supervision firewall PASS without PASS evidence for every scope")

    tex_paths = [paper_dir / "main.tex"] + sorted((paper_dir / "sections").glob("*.tex"))
    manuscript = " ".join(
        strip_tex_comments(path.read_text(encoding="utf-8")) for path in tex_paths if path.is_file()
    )
    normalized_manuscript = re.sub(r"\s+", " ", manuscript)
    if canonical not in normalized_manuscript:
        errors.append("canonical supervision wording is absent from manuscript source")
    for phrase in ("fully self-supervised pipeline", "annotation-free", "label-free"):
        if re.search(rf"\b{re.escape(phrase)}\b", normalized_manuscript, re.I):
            errors.append(f"prohibited supervision shortcut in manuscript: {phrase}")

    evidence_readiness = "PASS" if (
        freeze_status == "CONFIRMED"
        and all(component.get("status") == "CONFIRMED" for component in components)
        and firewall.get("status") == "PASS"
    ) else "BLOCKED"
    report = {
        "schema_version": 2,
        "verdict": "PASS" if not errors else "FAIL",
        "evidence_readiness": evidence_readiness,
        "collaborator_freeze_status": freeze_status,
        "component_count": len(components),
        "blocked_components": sum(component.get("status") == "BLOCKED" for component in components),
        "audited_input_hashes": {
            "paper/evidence/method_contract.yaml": sha256(contract_path),
            "paper/evidence/supervision_firewall.json": sha256(firewall_path) if firewall_path and firewall_path.is_file() else None,
            "collaborator_manifest": sha256(collaborator_manifest_path) if collaborator_manifest_path and collaborator_manifest_path.is_file() else None,
        },
        "collaborator_repository_root": str(collaborator_root) if collaborator_root else None,
        "collaborator_manifest_artifact_count": len(collaborator_artifacts),
        "errors": errors,
    }
    output = args.output or (paper_dir / ".aris/method-contract-check.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"METHOD_CONTRACT={report['verdict']}; readiness={evidence_readiness}; "
        f"components={len(components)}; blocked={report['blocked_components']}"
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
