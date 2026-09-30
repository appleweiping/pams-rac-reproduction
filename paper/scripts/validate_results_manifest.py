#!/usr/bin/env python3
"""Fail-closed structural and byte-provenance gate for paper results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evidence_common import (
    COMMIT_RE,
    add_check,
    emit_report,
    finish_report,
    hash_matches,
    json_load,
    resolve_inside,
    safe_latex_scalar,
    source_value,
)

REQUIRED_MAIN_METRICS = {"Period-mAP", "Period-AP50", "Period-AP75", "AvgMAE", "AvgOBO"}
PREDICTED_TRACK_METRICS = {"HOTA", "IDF1", "IDSW"}
AUDITS = ("experiment_audit", "result_to_claim", "paper_claim_audit")


def _schema_errors(data: dict[str, Any], schema_path: Path) -> list[str]:
    try:
        import jsonschema
    except ImportError:
        return []
    schema = json_load(schema_path)
    validator = jsonschema.Draft202012Validator(schema)
    return ["/".join(str(part) for part in error.absolute_path) or "<root>" for error in sorted(validator.iter_errors(data), key=lambda item: list(item.absolute_path))]


def _path_hash_list_ok(root: Path, items: Any) -> bool:
    if not isinstance(items, list) or not items:
        return False
    for item in items:
        if not isinstance(item, dict):
            return False
        try:
            path = resolve_inside(root, item.get("path"))
        except (TypeError, ValueError):
            return False
        if not path.is_file() or not hash_matches(path, item.get("sha256")):
            return False
    return True


def validate_manifest(manifest: Path, root: Path, mode: str, schema: Path | None = None) -> tuple[dict[str, Any], int]:
    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    try:
        data = json_load(manifest)
    except Exception as exc:
        add_check(checks, "manifest_parse", False, type(exc).__name__)
        return finish_report(gate="results_manifest", mode=mode, checks=checks, structural_failure=True)
    schema = schema or manifest.with_name("results_manifest.schema.json")
    try:
        errors = _schema_errors(data, schema)
    except Exception as exc:
        add_check(checks, "schema_load", False, type(exc).__name__)
        return finish_report(gate="results_manifest", mode=mode, checks=checks, structural_failure=True)
    add_check(checks, "json_schema", not errors, "schema valid" if not errors else "invalid fields: " + ", ".join(errors[:8]))

    status = data.get("status")
    eligible = status == "frozen-eligible"
    if not eligible:
        blockers.append("results manifest is not frozen-eligible")
        add_check(checks, "result_state", False, f"state={status}", blocking=False)
    else:
        add_check(checks, "result_state", True, "frozen-eligible")

    if eligible:
        add_check(checks, "frozen_table_eligible", data.get("frozen") is True and data.get("table_eligible") is True, "artifact is frozen and table-eligible")
        add_check(checks, "artifact_identity", all(isinstance(data.get(key), str) and data[key].strip() for key in ("artifact_id", "artifact_family")), "artifact id and family are present")
        add_check(checks, "method_commit", isinstance(data.get("method_commit"), str) and bool(COMMIT_RE.fullmatch(data["method_commit"])), "method commit is exact")
        dataset = data.get("dataset", {})
        add_check(checks, "dataset_binding", dataset.get("name") == "MultiRep" and all(dataset.get(key) for key in ("release", "split", "checksum")), "MultiRep release, split, and checksum are bound")
        protocol = data.get("protocol", {})
        add_check(checks, "protocol_binding", protocol.get("native_contextual_separated") is True and bool(protocol.get("track_modes")) and bool(protocol.get("evaluation_script")) and protocol.get("evaluation_type") == "real_gt", "track mode and real-GT evaluation protocol are bound")
        seeds = data.get("seeds", [])
        add_check(checks, "seed_scope", isinstance(seeds, list) and len(seeds) >= 3 and len(seeds) == len(set(seeds)), "at least three unique seeds")
        stats = data.get("statistics", {})
        add_check(checks, "uncertainty", all(stats.get(key) is True for key in ("mean", "sample_standard_deviation", "paired_video_cluster_bootstrap_95ci")) and stats.get("bootstrap_unit") in {"video", "video_cluster", "paired_video_cluster"}, "mean, sample SD, and paired video-cluster bootstrap 95% CI")
        add_check(checks, "config_hashes", _path_hash_list_ok(root, data.get("configuration_files")), "configuration files exist and match hashes")
        add_check(checks, "run_log_hashes", _path_hash_list_ok(root, data.get("run_logs")), "run logs exist and match hashes")
        add_check(checks, "prediction_hashes", _path_hash_list_ok(root, data.get("prediction_artifacts")), "per-instance predictions exist and match hashes")

        sources = data.get("source_artifacts", [])
        source_map: dict[str, dict[str, Any]] = {}
        source_ok = bool(sources)
        for item in sources if isinstance(sources, list) else []:
            source_id = item.get("id") if isinstance(item, dict) else None
            if not isinstance(source_id, str) or source_id in source_map:
                source_ok = False
                continue
            source_map[source_id] = item
            try:
                source_path = resolve_inside(root, item.get("path"))
            except (TypeError, ValueError):
                source_ok = False
                continue
            if item.get("format") not in {"json", "csv"} or item.get("frozen") is not True or not source_path.is_file() or not hash_matches(source_path, item.get("sha256")):
                source_ok = False
        add_check(checks, "source_artifact_hashes", source_ok, "all JSON/CSV source artifacts are frozen and byte-bound")

        metrics = set(data.get("metrics", {}))
        missing = REQUIRED_MAIN_METRICS - metrics
        if "predicted" in protocol.get("track_modes", []):
            missing |= PREDICTED_TRACK_METRICS - metrics
        add_check(checks, "required_metrics", not missing, "all required metrics are declared" if not missing else "missing metric declarations: " + ", ".join(sorted(missing)))

        audits = data.get("audits", {})
        audit_ok = True
        for name in AUDITS:
            item = audits.get(name, {})
            if item.get("status") != "PASS":
                audit_ok = False
                continue
            try:
                audit_path = resolve_inside(root, item.get("path"))
            except (TypeError, ValueError):
                audit_ok = False
                continue
            if not audit_path.is_file() or not hash_matches(audit_path, item.get("sha256")):
                audit_ok = False
        add_check(checks, "audit_hashes", audit_ok, "experiment, result-to-claim, and paper-claim audits PASS and match hashes")

        bindings = data.get("paper_bindings", {})
        binding_ok = source_ok
        keys: set[str] = set()
        binding_count = 0
        for group in ("protocol_fields", "results", "claims"):
            for binding in bindings.get(group, []) if isinstance(bindings, dict) else []:
                binding_count += 1
                key = binding.get("key") if isinstance(binding, dict) else None
                if not isinstance(key, str) or key in keys or binding.get("source_id") not in source_map:
                    binding_ok = False
                    continue
                keys.add(key)
                source = source_map[binding["source_id"]]
                try:
                    value = source_value(resolve_inside(root, source["path"]), source["format"], binding.get("locator", {}))
                    safe_latex_scalar(value)
                except Exception:
                    binding_ok = False
        abstract = bindings.get("abstract_result") if isinstance(bindings, dict) else None
        if not isinstance(abstract, dict) or abstract.get("source_id") not in source_map:
            binding_ok = False
        else:
            source = source_map[abstract["source_id"]]
            try:
                sentence = safe_latex_scalar(source_value(resolve_inside(root, source["path"]), source["format"], abstract.get("locator", {})))
                if len(sentence.split()) < 5:
                    binding_ok = False
            except Exception:
                binding_ok = False
        add_check(checks, "paper_bindings", binding_ok and binding_count > 0, "every paper value resolves to a hashed JSON/CSV scalar and the abstract sentence is bound")
        eligibility = data.get("eligibility", {})
        add_check(checks, "eligibility_flags", bool(eligibility) and all(value is True for value in eligibility.values()), "all eligibility conditions are explicitly true")
        add_check(checks, "no_blocking_reasons", data.get("blocking_reasons") == [], "eligible manifest has no blocking reason")

    structural = any(item["status"] == "FAIL" for item in checks)
    report, code = finish_report(gate="results_manifest", mode=mode, checks=checks, blockers=blockers, structural_failure=structural)
    report["manifest_sha256"] = None
    try:
        from evidence_common import sha256

        report["manifest_sha256"] = sha256(manifest)
    except OSError:
        pass
    return report, code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, nargs="?", default=Path(__file__).resolve().parents[1] / "evidence/results_manifest.json")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--mode", choices=("draft", "submission"), default="draft")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    root = (args.root or manifest.parents[2]).resolve()
    report, code = validate_manifest(manifest, root, args.mode, args.schema.resolve() if args.schema else None)
    emit_report(report, args.output)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
