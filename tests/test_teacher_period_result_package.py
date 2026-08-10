from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

REPOSITORY = Path(__file__).parents[1]
RESULT = (
    REPOSITORY
    / "results"
    / "dev-negative"
    / "pams_teacher_period_direct_inferred_v9_seed2026_07192de"
)

EXPECTED_METRICS: dict[str, float | int] = {
    "sample_count": 84,
    "nmae": 2.8601196735895007,
    "mae": 13.30952380952381,
    "rmse": 19.871012627874375,
    "obo": 0.16666666666666666,
    "exact": 0.08333333333333333,
    "bootstrap_samples": 10_000,
    "bootstrap_seed": 2026,
}
EXPECTED_CONFIDENCE_INTERVALS = {
    "nmae": {
        "level": 0.95,
        "low": 2.107281280555218,
        "high": 3.7010357293027396,
    },
    "mae": {
        "level": 0.95,
        "low": 10.25,
        "high": 16.512202380952377,
    },
    "rmse": {
        "level": 0.95,
        "low": 15.936611786213934,
        "high": 23.426271112095552,
    },
    "obo": {
        "level": 0.95,
        "low": 0.09523809523809523,
        "high": 0.25,
    },
    "exact": {
        "level": 0.95,
        "low": 0.03571428571428571,
        "high": 0.14285714285714285,
    },
}
FORBIDDEN_PREDICTION_KEY_FRAGMENTS = (
    "target",
    "ground_truth",
    "groundtruth",
    "action",
    "label",
)
SENSITIVE_PATTERNS = {
    "server IPv4 address": re.compile(rb"8\.133\.245\.52", re.IGNORECASE),
    "server SSH port": re.compile(rb"(?<![0-9])33123(?![0-9])"),
    "private-key filename": re.compile(rb"id_ed25519", re.IGNORECASE),
    "private-key PEM": re.compile(
        rb"BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY",
        re.IGNORECASE,
    ),
    "SSH login": re.compile(rb"lenovo@", re.IGNORECASE),
    "server home path": re.compile(rb"/home/lenovo", re.IGNORECASE),
    "server media path": re.compile(rb"/media/(?:lenovo/)?", re.IGNORECASE),
    "Windows user path": re.compile(rb"[A-Za-z]:\\Users\\", re.IGNORECASE),
    "WeChat migration path": re.compile(
        "电脑管家迁移文件|xwechat_files|wxid_".encode(),
        re.IGNORECASE,
    ),
}


def _json(relative_path: str) -> dict[str, Any]:
    payload = json.loads((RESULT / relative_path).read_text(encoding="utf-8"))
    assert isinstance(payload, dict), relative_path
    return payload


def _sha256(relative_path: str) -> str:
    return hashlib.sha256((RESULT / relative_path).read_bytes()).hexdigest()


def _contains_forbidden_prediction_key(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            if any(fragment in normalized for fragment in FORBIDDEN_PREDICTION_KEY_FRAGMENTS):
                return True
            if _contains_forbidden_prediction_key(item):
                return True
        return False
    if isinstance(value, list):
        return any(_contains_forbidden_prediction_key(item) for item in value)
    return False


def _assert_binding(binding: dict[str, Any]) -> None:
    locator = binding["locator"]
    assert isinstance(locator, str) and locator
    assert "\\" not in locator
    candidate = (RESULT / locator).resolve()
    assert candidate.is_file(), locator
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == binding["sha256"]


def _assert_core_report(report: dict[str, Any]) -> None:
    for name, expected in EXPECTED_METRICS.items():
        assert report[name] == expected, name
    assert report["confidence_intervals"] == EXPECTED_CONFIDENCE_INTERVALS
    assert len(report["per_video"]) == 84


def test_one_off_and_formal_predictions_are_unique_target_free_dev84() -> None:
    for relative_path in (
        "one-off/predict/predictions.json",
        "formal-cli/predict/predictions.json",
    ):
        artifact = _json(relative_path)
        rows = artifact["records"]
        identifiers = [row["video_id"] for row in rows]

        assert artifact["record_total"] == 84
        assert len(rows) == 84
        assert len(set(identifiers)) == 84
        assert not _contains_forbidden_prediction_key(artifact)

    one_off_receipt = _json("one-off/predict/prediction.receipt.json")
    formal_receipt = _json("formal-cli/predict/prediction.receipt.json")
    for relative_path, receipt in (
        ("one-off/predict/predictions.json", one_off_receipt),
        ("formal-cli/predict/predictions.json", formal_receipt),
    ):
        source = RESULT / relative_path
        assert receipt["prediction_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert receipt["prediction_bytes"] == source.stat().st_size


def test_formal_prediction_rows_obey_frozen_formula_and_countresult_invariants() -> None:
    artifact = _json("formal-cli/predict/predictions.json")
    assert artifact["rounding"] == "nearest_integer_half_up"
    assert artifact["prediction_batch_size"] == 32
    assert artifact["eligible_for_paper_table"] is False
    assert artifact["test_evaluation_authorized"] is False

    for row in artifact["records"]:
        confidence = row["confidence"]
        valid_frames = row["valid_frames"]
        period = row["period_frames"]
        raw = row["raw_count"]
        rounded = row["rounded_count"]
        stream = row["period_stream"]

        assert isinstance(valid_frames, int) and 0 <= valid_frames <= 256
        assert math.isfinite(period) and period > 0
        assert math.isfinite(raw) and raw >= 0
        assert math.isfinite(confidence) and 0 <= confidence <= 1
        if confidence > 0:
            assert valid_frames >= 2
            assert raw == (valid_frames - 1) / period
        else:
            assert raw == 0
            assert rounded == 0
        assert rounded == math.floor(raw + 0.5)
        assert row["expert_counts"] == [rounded, rounded, rounded]
        assert len(stream) == 256
        assert all(value == period for value in stream)


def test_formal_and_one_off_metrics_and_intervals_are_exact() -> None:
    formal = _json("formal-cli/score/evaluation.json")
    one_off = _json("one-off/score/evaluation.json")
    summary = _json("summary.json")

    _assert_core_report(formal["report"])
    _assert_core_report(one_off["report"])
    for name in (*EXPECTED_METRICS, "confidence_intervals"):
        assert formal["report"][name] == one_off["report"][name]
    assert [row["video_id"] for row in formal["report"]["per_video"]] == [
        row["video_id"] for row in one_off["report"]["per_video"]
    ]

    summary_metrics = summary["formal_metrics"]
    for name, expected in EXPECTED_METRICS.items():
        if name != "sample_count":
            assert summary_metrics[name] == expected
    assert summary_metrics["confidence_level"] == 0.95
    assert summary_metrics["confidence_intervals"] == {
        metric: {"low": bounds["low"], "high": bounds["high"]}
        for metric, bounds in EXPECTED_CONFIDENCE_INTERVALS.items()
    }

    formal_receipt = _json("formal-cli/score/evaluation.receipt.json")
    one_off_receipt = _json("one-off/score/evaluation.receipt.json")
    for relative_path, receipt in (
        ("formal-cli/score/evaluation.json", formal_receipt),
        ("one-off/score/evaluation.json", one_off_receipt),
    ):
        source = RESULT / relative_path
        assert receipt["evaluation_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
        assert receipt["evaluation_bytes"] == source.stat().st_size


def test_formal_equivalence_has_zero_mismatch_and_binds_both_runs() -> None:
    equivalence = _json("formal-cli/recovery/formal-cli-equivalence.json")

    assert equivalence["record_total"] == 84
    assert equivalence["all_prediction_fields_exact"] is True
    assert equivalence["all_metrics_exact"] is True
    assert set(equivalence["field_mismatch_counts"].values()) == {0}
    assert set(equivalence["max_absolute_difference"].values()) == {0.0}
    assert equivalence["first_mismatches"] == []
    assert set(equivalence["metrics_exact_equal"].values()) == {True}
    assert equivalence["one_off_prediction_sha256"] == _sha256(
        "one-off/predict/predictions.json"
    )
    assert equivalence["formal_prediction_sha256"] == _sha256(
        "formal-cli/predict/predictions.json"
    )
    assert equivalence["one_off_evaluation_sha256"] == _sha256(
        "one-off/score/evaluation.json"
    )
    assert equivalence["formal_evaluation_sha256"] == _sha256(
        "formal-cli/score/evaluation.json"
    )


def test_mount_audit_proves_prediction_and_scoring_isolation() -> None:
    mount = _json("formal-cli/recovery/mount-boundary-audit.json")
    provenance = _json("provenance.json")

    assert mount["network_mode"] == {"prediction": "none", "score": "none"}
    assert mount["prediction_has_dev_targets"] is False
    assert mount["prediction_has_test_identity_sidecars"] is True
    assert mount["prediction_has_test_media_pose_or_labels"] is False
    assert mount["score_has_dev_targets"] is True
    assert mount["score_has_pose_or_encoder"] is False

    assert provenance["prediction_isolation"] == {
        "development_targets_mounted": False,
        "test_identity_sidecars_only": True,
        "test_media_mounted": False,
        "test_pose_mounted": False,
        "test_labels_mounted": False,
        "network_mode": "none",
    }
    assert provenance["score_isolation"] == {
        "frozen_prediction_validated_before_target_load": True,
        "pose_mounted": False,
        "encoder_mounted": False,
        "development_targets_mounted": True,
        "network_mode": "none",
    }


def test_predev_gate_passed_without_dev_test_or_targets() -> None:
    gate = _json("predev-gate/passed-recovery/gate.json")
    status = _json("predev-gate/passed-recovery/status.json")

    assert gate["status"] == "passed"
    assert status["status"] == "completed"
    assert status["exit_code"] == 0
    assert gate["dev_inputs_mounted"] is False
    assert gate["test_inputs_mounted"] is False
    assert gate["targets_mounted"] is False
    assert set(gate["gates"].values()) == {True}
    assert gate["train337"]["records"] == 337
    assert gate["train337"]["positive_confidence_records"] == 321
    assert gate["train337"]["zero_confidence_records"] == 16
    assert gate["synthetic_count_2_40"]["records"] == 39
    assert gate["synthetic_count_2_40"]["exact_fraction"] == 1.0
    assert gate["synthetic_count_2_40"]["obo_fraction"] == 1.0
    assert gate["synthetic_stress_count8"]["records"] == 12
    assert gate["synthetic_stress_count8"]["obo_fraction"] == 1.0
    assert all(
        abs(row["rounded_count"] - row["target_count"]) <= 1
        for row in gate["synthetic_stress_count8"]["rows"]
    )


def test_r1_and_python39_post_score_r2_failures_are_disclosed_without_rerun() -> None:
    recovery = _json("formal-cli/recovery/recovery.json")
    before = _json("formal-cli/orchestration-status-before-recovery.json")
    after = _json("formal-cli/recovery/status.json")

    assert recovery["eligible_for_paper_table"] is False
    assert recovery["test_evaluation_authorized"] is False
    assert recovery["r1"]["failure_stage"] == "preflight"
    assert "incorrect expected dev-target SHA" in recovery["r1"]["failure_cause"]
    assert "no source export, container, prediction, or evaluation was produced" in (
        recovery["r1"]["boundary_disposition"]
    )
    assert recovery["r2"]["failure_stage"] == "equivalence"
    assert recovery["r2"]["failure_cause"] == (
        "host Python 3.9 does not support zip(strict=True)"
    )
    assert "prediction and scoring completed" in recovery["r2"]["boundary_disposition"]
    assert before == {
        "schema_version": 1,
        "status": "failed",
        "stage": "equivalence",
        "exit_code": 1,
        "updated_utc": "2026-07-30T16:24:52Z",
    }
    assert recovery["recovery"]["inference_rerun"] is False
    assert recovery["recovery"]["scoring_rerun"] is False
    assert after["status"] == "completed"
    assert after["stage"] == "post-score-recovery"
    assert after["inference_rerun"] is False
    assert after["scoring_rerun"] is False


def test_failure_formula_and_paired_analyses_are_internally_consistent() -> None:
    failure = _json("failure-analysis.json")
    formula = _json("formula-audit.json")
    paired = _json("paired-comparisons.json")
    formal_report = _json("formal-cli/score/evaluation.json")["report"]

    for artifact in (failure, formula, paired):
        assert artifact["sample_count"] == 84
        assert artifact["paper_table_eligible"] is False
        assert artifact["test_evaluation_authorized"] is False
        for binding in artifact["input_bindings"].values():
            if isinstance(binding, dict) and {"locator", "sha256"} <= set(binding):
                _assert_binding(binding)

    assert failure["formal_cli_equivalence"] == {
        "record_total": 84,
        "compared_fields": [
            "video_id",
            "video_sha256",
            "raw_count",
            "rounded_count",
            "period_frames",
            "confidence",
            "valid_frames",
        ],
        "all_prediction_fields_exact": True,
        "all_metrics_exact": True,
        "field_mismatch_total": 0,
        "max_absolute_numeric_difference": 0.0,
    }
    for metric in ("nmae", "mae", "rmse", "obo", "exact"):
        assert failure["reported_metrics_recomputed"][metric] == formal_report[metric]
    directional = failure["directional_error"]
    assert (
        directional["over_count"]["samples"]
        + directional["under_count"]["samples"]
        + directional["exact_count"]["samples"]
        == 84
    )
    assert (
        directional["over_count"]["absolute_error_sum"]
        + directional["under_count"]["absolute_error_sum"]
        == failure["reported_metrics_recomputed"]["absolute_error_sum"]
        == 1118
    )

    checks = formula["aggregate_checks"]
    assert checks == {
        "canonical_raw_formula_exact_match_count": 84,
        "canonical_raw_formula_mismatch_count": 0,
        "alternative_span_or_length_rounded_mismatch_count": 0,
        "canonical_half_up_vs_bankers_mismatch_count": 0,
        "formal_cli_vs_one_off_core_field_mismatch_count": 0,
        "formal_cli_vs_one_off_metric_mismatch_count": 0,
    }
    for variant in formula["formula_variants"]:
        assert variant["metrics_after_half_up"] == {
            metric: formal_report[metric]
            for metric in ("nmae", "mae", "rmse", "obo", "exact")
        }

    assert paired["alignment_audit"] == {
        "same_video_id_set": True,
        "same_video_id_order_after_explicit_id_join": True,
        "unique_video_ids": 84,
        "missing_rows": 0,
        "duplicate_rows": 0,
        "teacher_formal_and_one_off_core_fields_exact": True,
    }
    assert paired["bootstrap"] == {
        "pairing": "same 84 video rows resampled by shared indices within each method pair",
        "samples": 10_000,
        "seed": 2026,
        "confidence_level": 0.95,
        "interval": "numpy linear percentile at 0.025 and 0.975",
        "difference_direction": "A minus B",
        "rounding": "nearest integer, half up",
    }
    for comparison in paired["comparisons"]:
        absolute = comparison["absolute_error_comparison"]
        assert absolute["a_better"] + absolute["a_worse"] + absolute["tied"] == 84
        assert absolute["a_absolute_error_sum"] == 1118
        for metric in comparison["metrics"].values():
            assert metric["samples"] == 10_000
            assert metric["seed"] == 2026
            assert metric["confidence_interval_95"]["level"] == 0.95


def test_package_status_keeps_result_partial_and_test105_sealed() -> None:
    summary = _json("summary.json")
    status = _json("status.json")
    provenance = _json("provenance.json")

    assert summary["status"] == status["status"] == "partial_reproduction"
    assert summary["paper_table_eligible"] is False
    assert status["paper_table_eligible"] is False
    assert status["verified"] is False
    assert status["test105_touched"] is False
    assert status["test105_evaluation_authorized"] is False
    assert status["seeds_completed"] == [2026]
    assert status["seeds_42_3407_authorized"] is False
    assert summary["authorization"] == {
        "test105": False,
        "seeds_42_3407": False,
        "paper_table": False,
    }
    assert provenance["authorization"] == {
        "paper_table": False,
        "test105": False,
        "seeds_42_3407": False,
    }


def test_package_recursively_contains_no_server_or_key_material() -> None:
    files = sorted(path for path in RESULT.rglob("*") if path.is_file())
    assert files
    for path in files:
        payload = path.read_bytes()
        for description, pattern in SENSITIVE_PATTERNS.items():
            assert pattern.search(payload) is None, (
                f"{description} leaked in {path.relative_to(RESULT).as_posix()}"
            )


def test_recursive_hash_manifest_is_canonical_complete_and_exact() -> None:
    manifest_path = RESULT / "artifact-sha256.txt"
    assert manifest_path.is_file(), (
        "artifact-sha256.txt must be generated only after the package is final"
    )

    entries: list[tuple[str, str]] = []
    line_pattern = re.compile(r"([0-9a-f]{64})  (.+)")
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        match = line_pattern.fullmatch(line)
        assert match is not None, f"invalid manifest line {line_number}: {line!r}"
        digest, relative_path = match.groups()
        pure = PurePosixPath(relative_path)
        assert "\\" not in relative_path
        assert not pure.is_absolute()
        assert pure.as_posix() == relative_path
        assert pure.parts
        assert all(part not in {"", ".", ".."} for part in pure.parts)
        assert relative_path != manifest_path.name
        entries.append((relative_path, digest))

    paths = [relative_path for relative_path, _ in entries]
    assert paths == sorted(paths, key=str.casefold)
    assert len(paths) == len(set(paths))
    assert len(paths) == len({path.casefold() for path in paths})

    actual_paths = {
        path.relative_to(RESULT).as_posix()
        for path in RESULT.rglob("*")
        if path.is_file() and path != manifest_path
    }
    assert set(paths) == actual_paths
    for relative_path, expected_digest in entries:
        source = RESULT / PurePosixPath(relative_path)
        assert source.is_file()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected_digest
