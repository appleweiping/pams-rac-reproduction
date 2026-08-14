#!/usr/bin/env python3
"""Validate a synchronized result bundle and recompute every numeric claim.

This validator intentionally fails when no synchronized manifest exists.  It
never manufactures a pre-results manifest or treats a fixture as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from pathlib import Path, PurePosixPath
from typing import Any

import jsonschema


COUNT_METRICS = {"Period-mAP", "Period-AP50", "Period-AP75", "AvgMAE", "AvgOBO"}
TRACK_METRICS = {"HOTA", "IDF1", "IDSW"}
REQUIRED_COMMON = {"RepNet", "TransRAC", "PoseRAC", "PAMS", "Ours"}
REQUIRED_NATIVE = {"MultiCounter", "MultiCounter+"}
REQUIRED_AUDITS = {"experiment_audit", "result_to_claim", "paper_claim_audit"}
TOLERANCE = 1e-12


def close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=TOLERANCE, abs_tol=TOLERANCE)


def safe_resolve(root: Path, value: str) -> Path:
    """Resolve a forward-slash relative path without permitting root escape."""
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    if "\\" in value or pure.as_posix() != value or (pure.parts and ":" in pure.parts[0]):
        raise ValueError(f"unsafe relative path: {value!r}")
    resolved_root = root.resolve(strict=True)
    candidate = resolved_root.joinpath(*pure.parts).resolve(strict=True)
    if not candidate.is_relative_to(resolved_root):
        raise ValueError(f"path escapes bundle root: {value!r}")
    if not candidate.is_file():
        raise ValueError(f"not a regular file: {value!r}")
    return candidate


def resolve_json_pointer(value: Any, pointer: str) -> Any:
    """Resolve an RFC 6901 pointer and fail on absent/ambiguous targets."""
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("target_path must be a non-root RFC 6901 JSON Pointer")
    current = value
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if token not in current:
                raise KeyError(token)
            current = current[token]
        elif isinstance(current, list):
            if not token.isdigit() or (len(token) > 1 and token.startswith("0")):
                raise KeyError(token)
            index = int(token)
            if index >= len(current):
                raise KeyError(token)
            current = current[index]
        else:
            raise KeyError(token)
    return current


def binding_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def verify_protocol_source(
    field: dict[str, Any], bundle_root: Path, expected_value: str, errors: list[str]
) -> None:
    key = field.get("key", "<missing>")
    source = field.get("source", {})
    try:
        path = safe_resolve(bundle_root, source["path"])
        payload = path.read_bytes()
    except (KeyError, OSError, ValueError) as exc:
        errors.append(f"protocol binding {key}: source cannot be opened: {exc}")
        return
    actual_digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    if source.get("sha256") != actual_digest:
        errors.append(f"protocol binding {key}: source digest mismatch")
        return
    text = payload.decode("utf-8", errors="replace")
    represented = expected_value in text
    if not represented:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            stack = [parsed]
            while stack and not represented:
                item = stack.pop()
                if isinstance(item, dict):
                    stack.extend(item.values())
                elif isinstance(item, list):
                    stack.extend(item)
                else:
                    represented = binding_text(item) == expected_value
    if not represented:
        errors.append(f"protocol binding {key}: exact target value is absent from digest-bound source")


def metric_in_range(metric_id: str, value: Any, contract: dict[str, Any], context: str, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        errors.append(f"{context}/{metric_id}: non-finite or non-numeric value")
        return
    definition = contract[metric_id]
    maximum = definition["maximum"]
    if value < definition["minimum"] or (maximum is not None and value > maximum):
        errors.append(f"{context}/{metric_id}: outside declared range")


def validate_metric_contract(contract: dict[str, Any], errors: list[str]) -> None:
    for metric_id, definition in contract.items():
        minimum = definition["minimum"]
        maximum = definition["maximum"]
        if not math.isfinite(minimum) or (maximum is not None and not math.isfinite(maximum)):
            errors.append(f"metric_contract/{metric_id}: non-finite bound")
        if maximum is not None and minimum > maximum:
            errors.append(f"metric_contract/{metric_id}: minimum exceeds maximum")
        scale = definition["scale"]
        if scale == "fraction_0_1" and (minimum != 0 or maximum != 1):
            errors.append(f"metric_contract/{metric_id}: fraction_0_1 must use [0,1]")
        if scale == "percent_0_100" and (minimum != 0 or maximum != 100):
            errors.append(f"metric_contract/{metric_id}: percent_0_100 must use [0,100]")
        if scale in {"nonnegative_ratio", "count"} and minimum < 0:
            errors.append(f"metric_contract/{metric_id}: nonnegative scale has negative minimum")


def percentile_type7(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_interval(differences: list[float], samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(differences)
    replicates = [
        statistics.fmean(differences[rng.randrange(n)] for _ in range(n))
        for _ in range(samples)
    ]
    return percentile_type7(replicates, 0.025), percentile_type7(replicates, 0.975)


def validate_cluster_bootstrap(
    comparison: dict[str, Any],
    lhs_runs: dict[str, dict[str, Any]],
    rhs_runs: dict[str, dict[str, Any]],
    contract: dict[str, Any],
    bundle_root: Path,
    errors: list[str],
) -> None:
    cid = comparison["comparison_id"]
    try:
        source_path = safe_resolve(bundle_root, comparison["paired_ids"]["path"])
        source = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"{cid}: cannot read paired video-cluster artifact: {exc}")
        return

    required_keys = {"schema_version", "comparison_id", "metric_id", "pairing_unit", "clusters"}
    if set(source) != required_keys:
        errors.append(f"{cid}: paired artifact keys must be exactly {sorted(required_keys)}")
        return
    if source["schema_version"] != "1.0" or source["comparison_id"] != cid:
        errors.append(f"{cid}: paired artifact identity mismatch")
    if source["metric_id"] != comparison["metric_id"] or source["pairing_unit"] != "video_cluster":
        errors.append(f"{cid}: paired artifact metric/pairing mismatch")

    declared_pairs = {
        (pair["seed"], pair["lhs_run_id"], pair["rhs_run_id"])
        for pair in comparison["run_pairs"]
    }
    clusters = source.get("clusters")
    if not isinstance(clusters, list) or len(clusters) < 2:
        errors.append(f"{cid}: video-cluster bootstrap requires at least two clusters")
        return
    cluster_ids: set[str] = set()
    differences: list[float] = []
    for index, cluster in enumerate(clusters):
        if not isinstance(cluster, dict) or set(cluster) != {"cluster_id", "seed_pairs", "difference"}:
            errors.append(f"{cid}: cluster[{index}] has invalid keys")
            continue
        cluster_id = cluster["cluster_id"]
        if not isinstance(cluster_id, str) or not cluster_id:
            errors.append(f"{cid}: cluster[{index}] has invalid cluster_id")
            continue
        if cluster_id in cluster_ids:
            errors.append(f"{cid}: duplicate cluster_id {cluster_id}")
        cluster_ids.add(cluster_id)
        seed_pairs = cluster["seed_pairs"]
        if not isinstance(seed_pairs, list):
            errors.append(f"{cid}/{cluster_id}: seed_pairs is not a list")
            continue
        observed_pairs: set[tuple[int, str, str]] = set()
        pair_differences: list[float] = []
        for pair in seed_pairs:
            if not isinstance(pair, dict) or set(pair) != {"seed", "lhs_run_id", "rhs_run_id", "lhs_value", "rhs_value"}:
                errors.append(f"{cid}/{cluster_id}: invalid seed-pair record")
                continue
            identity = (pair["seed"], pair["lhs_run_id"], pair["rhs_run_id"])
            observed_pairs.add(identity)
            lhs = lhs_runs.get(pair["lhs_run_id"])
            rhs = rhs_runs.get(pair["rhs_run_id"])
            if lhs is None or rhs is None or lhs.get("seed") != pair["seed"] or rhs.get("seed") != pair["seed"]:
                errors.append(f"{cid}/{cluster_id}: unresolved or seed-mismatched run pair {identity}")
            lhs_value = pair["lhs_value"]
            rhs_value = pair["rhs_value"]
            metric_in_range(comparison["metric_id"], lhs_value, contract, f"{cid}/{cluster_id}/lhs", errors)
            metric_in_range(comparison["metric_id"], rhs_value, contract, f"{cid}/{cluster_id}/rhs", errors)
            if isinstance(lhs_value, (int, float)) and not isinstance(lhs_value, bool) and math.isfinite(lhs_value) and isinstance(rhs_value, (int, float)) and not isinstance(rhs_value, bool) and math.isfinite(rhs_value):
                pair_differences.append(float(lhs_value) - float(rhs_value))
        if observed_pairs != declared_pairs:
            errors.append(f"{cid}/{cluster_id}: seed/run pairs do not exactly match comparison.run_pairs")
        if len(pair_differences) != len(declared_pairs) or not pair_differences:
            continue
        expected_difference = statistics.fmean(pair_differences)
        reported_difference = cluster["difference"]
        if not isinstance(reported_difference, (int, float)) or isinstance(reported_difference, bool) or not math.isfinite(reported_difference):
            errors.append(f"{cid}/{cluster_id}: non-finite difference")
            continue
        if not close(float(reported_difference), expected_difference):
            errors.append(f"{cid}/{cluster_id}: difference does not recompute")
        differences.append(expected_difference)

    if len(differences) != len(clusters):
        return
    estimate = statistics.fmean(differences)
    if not close(comparison["estimate"], estimate):
        errors.append(f"{cid}: estimate does not equal the mean video-cluster paired difference")
    interval = comparison["confidence_interval"]
    lower = interval["lower"]
    upper = interval["upper"]
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (lower, upper)):
        errors.append(f"{cid}: non-finite confidence bound")
        return
    if lower > upper:
        errors.append(f"{cid}: confidence bounds are reversed")
    expected_lower, expected_upper = bootstrap_interval(
        differences, interval["bootstrap_samples"], interval["bootstrap_seed"]
    )
    if not close(lower, expected_lower) or not close(upper, expected_upper):
        errors.append(f"{cid}: percentile-bootstrap bounds do not deterministically recompute")
    definition = contract[comparison["metric_id"]]
    if definition["maximum"] is not None:
        minimum_difference = definition["minimum"] - definition["maximum"]
        maximum_difference = definition["maximum"] - definition["minimum"]
        if any(value < minimum_difference or value > maximum_difference for value in (comparison["estimate"], lower, upper)):
            errors.append(f"{cid}: estimate or confidence bound exceeds the metric's possible difference range")


def validate(data: dict[str, Any], manifest_path: Path, bundle_root: Path) -> list[str]:
    errors: list[str] = []
    validate_metric_contract(data["metric_contract"], errors)
    families = data["result_families"]
    family_by_id: dict[str, dict[str, Any]] = {}
    method_identity: dict[str, str] = {}
    run_by_id: dict[str, tuple[str, dict[str, Any]]] = {}
    aggregate_by_id: dict[str, tuple[str, dict[str, Any]]] = {}

    try:
        declared_manifest = safe_resolve(bundle_root, data["manifest_integrity"]["result_manifest_path"])
        if declared_manifest != manifest_path.resolve(strict=True):
            errors.append("manifest_integrity.result_manifest_path does not identify the validated manifest")
    except (OSError, ValueError) as exc:
        errors.append(f"manifest_integrity.result_manifest_path: {exc}")
    excludes = set(data["manifest_integrity"]["package_manifest_excludes"])
    required_excludes = {data["manifest_integrity"]["package_manifest_path"]}
    if excludes != required_excludes:
        errors.append("manifest_integrity.package_manifest_excludes must contain exactly the package manifest; the receipt is external and is not a package path")

    for family in families:
        fid = family["family_id"]
        if fid in family_by_id:
            errors.append(f"duplicate family_id: {fid}")
        family_by_id[fid] = family

        method = family["method"]
        identity = json.dumps(method, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        previous = method_identity.setdefault(method["id"], identity)
        if previous != identity:
            errors.append(f"{fid}: method.id {method['id']} changes semantic identity across protocol families")

        eligibility = family["eligibility"]
        is_eligible = eligibility["status"] == "eligible"
        if family["table_eligible"] != is_eligible:
            errors.append(f"{fid}: table_eligible disagrees with eligibility.status")
        if is_eligible:
            if family["provenance"]["evidence_class"] != "real_gt":
                errors.append(f"{fid}: eligible family provenance is not real_gt")
            if set(eligibility["required_audits"]) != REQUIRED_AUDITS:
                errors.append(f"{fid}: eligible family must require exactly {sorted(REQUIRED_AUDITS)}")
            for audit_name in REQUIRED_AUDITS:
                if data["audits"][audit_name]["verdict"] != "PASS":
                    errors.append(f"{fid}: required audit is not PASS: {audit_name}")
        elif eligibility["required_audits"]:
            errors.append(f"{fid}: ineligible/context family must not imply completed eligibility audits")

        runs = family["runs"]
        seeds = [run["seed"] for run in runs]
        if len(set(seeds)) != len(seeds):
            errors.append(f"{fid}: duplicate seed")
        if family["run_policy"] == "multi_seed_reproduction" and len(runs) < 3:
            errors.append(f"{fid}: fewer than three distinct completed seeds")
        if family["run_policy"] == "official_single_checkpoint" and len(runs) != 1:
            errors.append(f"{fid}: official checkpoint family must contain exactly one run")

        expected_metrics = COUNT_METRICS | (TRACK_METRICS if family["protocol"]["mode"] == "common_predicted_tracks" else set())
        family_run_ids: set[str] = set()
        run_metric_values: dict[str, dict[str, float]] = {}
        for run in runs:
            run_id = run["run_id"]
            if run_id in run_by_id:
                errors.append(f"{fid}: globally reused run_id {run_id}")
            run_by_id[run_id] = (fid, run)
            family_run_ids.add(run_id)
            metrics = run["metrics"]
            metric_ids = [metric["metric_id"] for metric in metrics]
            if len(metric_ids) != len(set(metric_ids)) or set(metric_ids) != expected_metrics:
                errors.append(f"{fid}/{run_id}: metric set must be exactly {sorted(expected_metrics)}")
            run_metric_values[run_id] = {}
            for metric in metrics:
                metric_in_range(metric["metric_id"], metric["value"], data["metric_contract"], run_id, errors)
                run_metric_values[run_id][metric["metric_id"]] = metric["value"]
            if family["protocol"]["mode"] == "common_predicted_tracks":
                tracking = family["protocol"]["tracking_metrics"]
                for metric_id in TRACK_METRICS:
                    if metric_id in run_metric_values[run_id] and not close(run_metric_values[run_id][metric_id], tracking[metric_id]):
                        errors.append(f"{fid}/{run_id}/{metric_id}: run value differs from frozen common tracking metric")

        aggregates = family["aggregate_metrics"]
        aggregate_metric_ids = [aggregate["metric_id"] for aggregate in aggregates]
        if len(aggregate_metric_ids) != len(set(aggregate_metric_ids)) or set(aggregate_metric_ids) != expected_metrics:
            errors.append(f"{fid}: aggregate metric set must be exactly {sorted(expected_metrics)}")
        for aggregate in aggregates:
            aggregate_id = aggregate["aggregate_id"]
            if aggregate_id in aggregate_by_id:
                errors.append(f"{fid}: globally reused aggregate_id {aggregate_id}")
            aggregate_by_id[aggregate_id] = (fid, aggregate)
            metric_id = aggregate["metric_id"]
            if set(aggregate["run_ids"]) != family_run_ids or aggregate["n_runs"] != len(runs):
                errors.append(f"{fid}/{metric_id}: aggregate run linkage mismatch")
                continue
            values = [run_metric_values[run_id][metric_id] for run_id in aggregate["run_ids"] if metric_id in run_metric_values.get(run_id, {})]
            if len(values) != len(runs):
                errors.append(f"{fid}/{metric_id}: aggregate lacks complete run values")
                continue
            expected_mean = statistics.fmean(values)
            if not math.isfinite(aggregate["mean"]) or not close(aggregate["mean"], expected_mean):
                errors.append(f"{fid}/{metric_id}: aggregate mean does not recompute")
            metric_in_range(metric_id, aggregate["mean"], data["metric_contract"], f"{fid}/aggregate", errors)
            reported_sd = aggregate["sample_standard_deviation"]
            if family["run_policy"] == "multi_seed_reproduction":
                expected_sd = statistics.stdev(values)
                if reported_sd is None or not math.isfinite(reported_sd) or not close(reported_sd, expected_sd):
                    errors.append(f"{fid}/{metric_id}: sample SD does not recompute")
            elif reported_sd is not None:
                errors.append(f"{fid}/{metric_id}: official checkpoint must have null SD")

    eligible_by_mode: dict[str, list[dict[str, Any]]] = {
        "oracle_tracks": [], "common_predicted_tracks": [], "native_pipeline": []
    }
    for family in families:
        if family["table_eligible"]:
            eligible_by_mode[family["protocol"]["mode"]].append(family)
    for mode, required in (("oracle_tracks", REQUIRED_COMMON), ("common_predicted_tracks", REQUIRED_COMMON), ("native_pipeline", REQUIRED_NATIVE)):
        names = [family["method"]["name"] for family in eligible_by_mode[mode]]
        if len(names) != len(set(names)) or set(names) != required:
            errors.append(f"{mode}: eligible family set must be exactly {sorted(required)} once each; got {sorted(names)}")
    for mode in ("oracle_tracks", "common_predicted_tracks"):
        for family in eligible_by_mode[mode]:
            expected_role = "proposed_method" if family["method"]["name"] == "Ours" else "common_frontend_baseline"
            if family["evidence_role"] != expected_role:
                errors.append(f"{family['family_id']}: wrong evidence_role for {mode}")
    for family in eligible_by_mode["native_pipeline"]:
        if family["evidence_role"] != "native_context":
            errors.append(f"{family['family_id']}: native row must have native_context evidence_role")

    predicted = eligible_by_mode["common_predicted_tracks"]
    if predicted:
        signatures = {
            json.dumps({
                "pipeline_kind": family["protocol"]["pipeline"]["kind"],
                "pipeline_name": family["protocol"]["pipeline"]["name"],
                "pipeline_revision": family["protocol"]["pipeline"]["revision"],
                "pipeline_config_sha256": family["protocol"]["pipeline"]["config"]["sha256"],
                "pipeline_source_sha256": family["protocol"]["pipeline"]["source_artifact"]["sha256"],
                "track_sha256": family["protocol"]["track_artifact"]["sha256"],
                "tracking_metrics": family["protocol"]["tracking_metrics"],
            }, sort_keys=True, separators=(",", ":"))
            for family in predicted
        }
        if len(signatures) != 1:
            errors.append("common_predicted_tracks: frontend/config/source/track hashes or tracking metrics differ")
    oracle_track_hashes = {family["protocol"]["track_artifact"]["sha256"] for family in eligible_by_mode["oracle_tracks"]}
    if oracle_track_hashes and len(oracle_track_hashes) != 1:
        errors.append("oracle_tracks: eligible families do not share one oracle-track hash")

    comparison_by_id: dict[str, dict[str, Any]] = {}
    primary: list[dict[str, Any]] = []
    for comparison in data["comparisons"]:
        cid = comparison["comparison_id"]
        if cid in comparison_by_id:
            errors.append(f"duplicate comparison_id: {cid}")
        comparison_by_id[cid] = comparison
        lhs = family_by_id.get(comparison["lhs_family_id"])
        rhs = family_by_id.get(comparison["rhs_family_id"])
        if lhs is None or rhs is None:
            errors.append(f"{cid}: unresolved family reference")
            continue
        if comparison["paper_eligible"] and (not lhs["table_eligible"] or not rhs["table_eligible"]):
            errors.append(f"{cid}: paper-eligible comparison targets an ineligible family")
        if not isinstance(comparison["estimate"], (int, float)) or isinstance(comparison["estimate"], bool) or not math.isfinite(comparison["estimate"]):
            errors.append(f"{cid}: non-finite estimate")
        else:
            definition = data["metric_contract"][comparison["metric_id"]]
            if definition["maximum"] is not None:
                lower_difference = definition["minimum"] - definition["maximum"]
                upper_difference = definition["maximum"] - definition["minimum"]
                if not lower_difference <= comparison["estimate"] <= upper_difference:
                    errors.append(f"{cid}: estimate exceeds the metric's possible difference range")
        if comparison["paired"]:
            if comparison["paired_ids"]["media_type"] != "application/json":
                errors.append(f"{cid}: paired video-cluster artifact must declare application/json")
            lhs_runs = {run["run_id"]: run for run in lhs["runs"]}
            rhs_runs = {run["run_id"]: run for run in rhs["runs"]}
            pairs = comparison["run_pairs"]
            identities = [(pair["seed"], pair["lhs_run_id"], pair["rhs_run_id"]) for pair in pairs]
            if len(identities) != len(set(identities)):
                errors.append(f"{cid}: duplicate seed/run pair")
            if {pair["lhs_run_id"] for pair in pairs} != set(lhs_runs) or {pair["rhs_run_id"] for pair in pairs} != set(rhs_runs):
                errors.append(f"{cid}: run_pairs must cover both families exactly once")
            if len({pair["seed"] for pair in pairs}) != len(pairs):
                errors.append(f"{cid}: paired seeds are not unique")
            for pair in pairs:
                left = lhs_runs.get(pair["lhs_run_id"])
                right = rhs_runs.get(pair["rhs_run_id"])
                if left is None or right is None or left["seed"] != pair["seed"] or right["seed"] != pair["seed"]:
                    errors.append(f"{cid}: run pair does not resolve to the declared common seed")
            validate_cluster_bootstrap(comparison, lhs_runs, rhs_runs, data["metric_contract"], bundle_root, errors)
        else:
            if comparison.get("confidence_interval") is not None or comparison.get("paired_ids") is not None or comparison.get("pairing_unit") is not None:
                errors.append(f"{cid}: unpaired comparison carries paired-only fields")
        if comparison["kind"] == "primary_paired":
            primary.append(comparison)
    if len(primary) != 1:
        errors.append(f"exactly one primary_paired comparison is required; got {len(primary)}")
    else:
        comparison = primary[0]
        lhs = family_by_id.get(comparison["lhs_family_id"])
        rhs = family_by_id.get(comparison["rhs_family_id"])
        if lhs and rhs and not (
            comparison["paper_eligible"]
            and comparison["metric_id"] == "Period-mAP"
            and lhs["method"]["name"] == "Ours"
            and rhs["method"]["name"] == "PAMS"
            and lhs["protocol"]["mode"] == rhs["protocol"]["mode"] == "common_predicted_tracks"
        ):
            errors.append(f"{comparison['comparison_id']}: primary must be eligible common-predicted Ours-vs-PAMS Period-mAP")

    output_ids: set[str] = set()
    for output in data["generated_outputs"]:
        output_id = output["output_id"]
        if output_id in output_ids:
            errors.append(f"duplicate output_id: {output_id}")
        output_ids.add(output_id)
        for fid in output["family_ids"]:
            if fid not in family_by_id:
                errors.append(f"{output_id}: unresolved family {fid}")
            elif not family_by_id[fid]["table_eligible"]:
                errors.append(f"{output_id}: targets ineligible family {fid}")

    bindings = data["paper_bindings"]
    result_bindings: dict[str, dict[str, Any]] = {}
    for binding in bindings["results"]:
        key = binding["key"]
        if key in result_bindings:
            errors.append(f"duplicate paper result binding: {key}")
        result_bindings[key] = binding
        family = family_by_id.get(binding["family_id"])
        if family is None or not family["table_eligible"]:
            errors.append(f"{key}: binding family is unresolved or ineligible")
            continue
        target_type = binding["target_type"]
        target_id = binding["target_id"]
        expected_value: float | None = None
        expected_source_sha256: str | None = None
        if target_type == "run_metric":
            resolved = run_by_id.get(target_id)
            if resolved is None or resolved[0] != binding["family_id"]:
                errors.append(f"{key}: run target does not resolve inside family")
            else:
                metric = next((item for item in resolved[1]["metrics"] if item["metric_id"] == binding["metric_id"]), None)
                if metric is None or binding["statistic"] != "single_checkpoint":
                    errors.append(f"{key}: run target metric/statistic mismatch")
                else:
                    expected_value = metric["value"]
                    expected_source_sha256 = metric["source"]["sha256"]
        elif target_type == "aggregate_metric":
            resolved = aggregate_by_id.get(target_id)
            if resolved is None or resolved[0] != binding["family_id"] or resolved[1]["metric_id"] != binding["metric_id"]:
                errors.append(f"{key}: aggregate target does not resolve inside family")
            elif binding["statistic"] == "mean":
                expected_value = resolved[1]["mean"]
                expected_source_sha256 = resolved[1]["source"]["sha256"]
            elif binding["statistic"] == "sample_standard_deviation" and resolved[1]["sample_standard_deviation"] is not None:
                expected_value = resolved[1]["sample_standard_deviation"]
                expected_source_sha256 = resolved[1]["source"]["sha256"]
            else:
                errors.append(f"{key}: aggregate statistic mismatch")
        else:
            comparison = comparison_by_id.get(target_id)
            if comparison is None or not comparison["paper_eligible"] or comparison["lhs_family_id"] != binding["family_id"] or comparison["metric_id"] != binding["metric_id"]:
                errors.append(f"{key}: comparison target does not resolve to an eligible lhs family")
            elif binding["statistic"] == "paired_difference":
                expected_value = comparison["estimate"]
                expected_source_sha256 = comparison["source"]["sha256"]
            elif binding["statistic"] == "ci_lower":
                expected_value = comparison["confidence_interval"]["lower"]
                expected_source_sha256 = comparison["source"]["sha256"]
            elif binding["statistic"] == "ci_upper":
                expected_value = comparison["confidence_interval"]["upper"]
                expected_source_sha256 = comparison["source"]["sha256"]
            else:
                errors.append(f"{key}: comparison statistic mismatch")
        if expected_value is not None and not close(binding["raw_value"], expected_value):
            errors.append(f"{key}: raw_value differs from exact target")
        if expected_source_sha256 is not None and binding["source"]["sha256"] != expected_source_sha256:
            errors.append(f"{key}: source digest differs from exact target source")

    abstract_keys = set(bindings["abstract_result"]["result_keys"])
    if not abstract_keys.issubset(result_bindings):
        errors.append(f"abstract_result: unresolved result keys {sorted(abstract_keys - set(result_bindings))}")
    else:
        expected_abstract_sources = {result_bindings[key]["source"]["sha256"] for key in abstract_keys}
        actual_abstract_sources = {source["sha256"] for source in bindings["abstract_result"]["sources"]}
        if actual_abstract_sources != expected_abstract_sources:
            errors.append("abstract_result: sources do not exactly match bound result targets")
    for field in bindings["protocol_fields"]:
        unresolved = [fid for fid in field["family_ids"] if fid not in family_by_id or not family_by_id[fid]["table_eligible"]]
        if unresolved:
            errors.append(f"protocol binding {field['key']}: unresolved/ineligible families {sorted(unresolved)}")
            continue
        scope = field["target_scope"]
        targets = [data["dataset"]] if scope == "dataset" else [family_by_id[fid] for fid in field["family_ids"]]
        resolved_values: list[str] = []
        for target in targets:
            try:
                resolved_values.append(binding_text(resolve_json_pointer(target, field["target_path"])))
            except (KeyError, ValueError) as exc:
                errors.append(f"protocol binding {field['key']}: target does not resolve: {exc}")
        if resolved_values and (len(set(resolved_values)) != 1 or resolved_values[0] != field["value"]):
            errors.append(
                f"protocol binding {field['key']}: declared value differs from exact target_path value(s)"
            )
        if resolved_values and len(resolved_values) == len(targets):
            verify_protocol_source(field, bundle_root, field["value"], errors)
    for field_name in ("protocol_fields", "claims"):
        keys = [item["key"] for item in bindings[field_name]]
        if len(keys) != len(set(keys)):
            errors.append(f"duplicate paper binding key in {field_name}")
    for claim in bindings["claims"]:
        missing_results = set(claim["result_keys"]) - set(result_bindings)
        missing_comparisons = {
            cid for cid in claim["comparison_ids"]
            if cid not in comparison_by_id or not comparison_by_id[cid]["paper_eligible"]
        }
        if missing_results or missing_comparisons:
            errors.append(f"claim binding {claim['key']}: unresolved result/comparison targets")
        else:
            expected_sources = {result_bindings[key]["source"]["sha256"] for key in claim["result_keys"]}
            expected_sources.update(comparison_by_id[cid]["source"]["sha256"] for cid in claim["comparison_ids"])
            actual_sources = {source["sha256"] for source in claim["sources"]}
            if actual_sources != expected_sources:
                errors.append(f"claim binding {claim['key']}: sources do not exactly match target evidence")

    if any(family["table_eligible"] for family in families) and data["provenance"]["evidence_class"] != "real_gt":
        errors.append("table-eligible bundle-level provenance is not real_gt")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--schema", type=Path)
    parser.add_argument("--bundle-root", type=Path)
    args = parser.parse_args()
    if not args.manifest.is_file():
        print(f"RESULT_MANIFEST=BLOCKED; synchronized manifest is absent: {args.manifest}")
        return 2
    schema_path = args.schema or Path(__file__).parents[1] / "evidence/result_manifest.schema.json"
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        data = json.loads(args.manifest.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker()).validate(data)
    except (OSError, UnicodeError, json.JSONDecodeError, jsonschema.SchemaError, jsonschema.ValidationError) as exc:
        print(f"RESULT_MANIFEST=FAIL; schema_or_parse_error={exc}")
        return 1
    bundle_root = (args.bundle_root or args.manifest.parent).resolve()
    errors = validate(data, args.manifest, bundle_root)
    if errors:
        print("RESULT_MANIFEST=FAIL")
        for error in errors:
            print(f"- {error}")
        return 1
    run_count = sum(len(family["runs"]) for family in data["result_families"])
    print(f"RESULT_MANIFEST=PASS; families={len(data['result_families'])}; runs={run_count}; comparisons={len(data['comparisons'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
