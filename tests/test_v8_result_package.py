import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

REPOSITORY = Path(__file__).parents[1]
RESULT = (
    REPOSITORY
    / "results"
    / "dev-negative"
    / "pams_sshead_longest_track_v8_seed2026_07192de"
)


def _json(name: str) -> dict[str, Any]:
    payload = json.loads((RESULT / name).read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _contains_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return bool(forbidden.intersection(value)) or any(
            _contains_key(item, forbidden) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def test_v8_summary_matches_frozen_evaluation_and_gate() -> None:
    summary = _json("summary.json")
    evaluation = _json("evaluation.json")
    decision = _json("extension-decision.json")

    assert summary["status"] == "partial_reproduction"
    assert summary["protocol"]["test105_evaluated"] is False
    for metric in ("nmae", "mae", "rmse", "obo", "exact", "sample_count"):
        assert summary["development_metrics"][metric] == pytest.approx(
            evaluation["report"][metric]
        )
    assert summary["development_metrics"]["confidence_intervals"] == evaluation[
        "report"
    ]["confidence_intervals"]
    assert len(evaluation["report"]["per_video"]) == 84
    assert decision["decision"] == "do-not-extend"
    assert decision["authorized"] is False
    assert decision["authorized_seeds"] == []
    assert summary["frozen_extension_gate"]["decision"] == decision["decision"]
    assert summary["access_control"]["test_evaluation_authorized"] is False


def test_v8_prediction_artifact_is_target_free_and_source_bound() -> None:
    summary = _json("summary.json")
    predictions = _json("predictions.json")
    encoder_started = _json("encoder.started.json")
    sshead_started = _json("sshead.started.json")

    assert predictions["record_total"] == 84
    assert len(predictions["records"]) == 84
    assert len({row["video_id"] for row in predictions["records"]}) == 84
    assert not _contains_key(
        predictions,
        {"target", "targets", "ground_truth", "ground_truth_count"},
    )
    revision = summary["source"]["git_sha"]
    image_id = summary["source"]["container_image_id"]
    for started in (encoder_started, sshead_started):
        assert started["git_sha"] == revision
        assert started["hardware"]["container"]["source_revision"] == revision
        assert started["hardware"]["container"]["image_id"] == image_id
    assert predictions["prediction_source_git_sha"] == revision
    assert predictions["prediction_container_image_id"] == image_id


def test_v8_completion_receipts_and_pose_audit_are_complete() -> None:
    encoder = _json("encoder.completed.json")
    sshead = _json("sshead.completed.json")
    pose = _json("pose-cache-comparison.json")
    pose_status = _json("pose-audit.status.json")
    build_status = _json("build.status.json")

    assert encoder["status"] == sshead["status"] == "completed"
    assert encoder["metrics"]["completed_epochs"] == 150
    assert sshead["metrics"]["completed_epochs"] == 30
    assert (
        encoder["metrics"]["final_epoch"]["cross_cluster_shortfall"]
        == 0
    )
    assert sshead["metrics"]["final_epoch"]["zero_grad_steps"] == 0
    assert pose["status"] == "passed"
    assert pose["scope"]["sample_count"] == 421
    assert pose["scope"]["test105_accessed"] is False
    assert pose["scope"]["labels_or_targets_accessed"] is False
    assert pose_status["status"] == "completed"
    assert build_status["status"] == "completed"


def test_v8_public_package_hash_manifest_and_paths() -> None:
    manifest_path = RESULT / "artifact-sha256.txt"
    expected: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", maxsplit=1)
        assert name == Path(name).name
        assert name not in expected
        expected[name] = digest

    actual_names = {
        path.name for path in RESULT.iterdir() if path != manifest_path
    }
    assert set(expected) == actual_names
    assert len({name.casefold() for name in expected}) == len(expected)
    assert list(expected) == sorted(expected, key=str.casefold)
    for name, digest in expected.items():
        assert hashlib.sha256((RESULT / name).read_bytes()).hexdigest() == digest

    for path in RESULT.iterdir():
        if path.suffix not in {".json", ".jsonl", ".md", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "/media/" not in text
        assert "8.133.245.52" not in text
        assert "id_ed25519" not in text
