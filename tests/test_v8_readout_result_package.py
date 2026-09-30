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
    / "pams_v8_frozen_readout_diagnostics_seed2026_07192de"
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


@pytest.mark.parametrize(
    ("prefix", "nmae", "obo"),
    [
        ("literal", 0.6646265395341749, 0.2619047619047619),
        ("local-frequency", 0.5290583435903633, 0.3333333333333333),
    ],
)
def test_readout_predictions_are_target_free_and_metrics_match(
    prefix: str,
    nmae: float,
    obo: float,
) -> None:
    predictions = _json(f"{prefix}.predictions.json")
    evaluation = _json(f"{prefix}.evaluation.json")

    assert predictions["record_total"] == 84
    assert len(predictions["records"]) == 84
    assert len({row["video_id"] for row in predictions["records"]}) == 84
    assert not _contains_key(
        predictions,
        {"target", "targets", "ground_truth", "ground_truth_count"},
    )
    report = evaluation["report"]
    assert report["sample_count"] == 84
    assert len(report["per_video"]) == 84
    assert report["bootstrap_samples"] == 10_000
    assert report["bootstrap_seed"] == 2026
    assert report["nmae"] == pytest.approx(nmae)
    assert report["obo"] == pytest.approx(obo)


def test_synthetic_replay_and_paired_comparisons_are_frozen() -> None:
    replay = _json("local-frequency.synthetic-freeze-replay.json")
    paired = _json("paired-comparisons.json")
    summary = _json("summary.json")

    assert replay["status"] == "verified"
    assert replay["expected_replay_match"] is True
    assert replay["observed_replay"] == replay["expected_replay"]
    assert replay["observed_replay"] == {
        "exact_rate": 0.98,
        "mean_absolute_error": 0.02,
        "normalized_mean_absolute_error": 0.0025,
        "obo": 1.0,
        "sample_count": 50,
        "selected_candidate": "frame_centered.min064.median",
    }

    comparisons = {
        (item["a"], item["b"]): item for item in paired["comparisons"]
    }
    versus_sshead = comparisons[("local_frequency_v3", "sshead_v3")]
    assert versus_sshead["metrics"]["nmae"]["difference_a_minus_b"] == pytest.approx(
        -0.11406491872932767
    )
    nmae_ci = versus_sshead["metrics"]["nmae"]["confidence_interval_95"]
    assert nmae_ci["high"] < 0
    assert versus_sshead["absolute_error_comparison"] == {
        "a_better": 46,
        "a_worse": 15,
        "tied": 23,
    }
    versus_v2 = comparisons[("local_frequency_v3", "local_frequency_v2")]
    assert versus_v2["metrics"]["nmae"]["difference_a_minus_b"] > 0
    assert versus_v2["metrics"]["obo"]["difference_a_minus_b"] == 0

    decision = summary["decision"]
    assert decision["authorizes_seeds_42_3407"] is False
    assert decision["authorizes_test105"] is False
    assert decision["readout_strictly_improves_v8_sshead_nmae_and_obo"] is True
    assert decision["strictly_improves_same_readout_v2_nmae_and_obo"] is False


def test_readout_provenance_is_bound_and_test_remains_sealed() -> None:
    provenance = _json("provenance.json")
    literal = _json("literal.prediction.receipt.json")
    local_frequency = _json("local-frequency.prediction.receipt.json")

    assert provenance["source_revision"] == (
        "07192debb7f5748c53a1c6d4f0c0d3f16228e229"
    )
    assert provenance["encoder_checkpoint_sha256"] == (
        "6817fe468d76c3fa2182dd831695d6fe3d6da208a2b7f8dfa90cffa6872e8053"
    )
    isolation = provenance["prediction_isolation"]
    assert isolation["development_targets_mounted"] is False
    assert isolation["test_videos_mounted"] is False
    assert isolation["test_pose_mounted"] is False
    assert isolation["test_targets_mounted"] is False
    assert provenance["authorization"] == {
        "paper_table": False,
        "seeds_42_3407": False,
        "test105": False,
    }
    assert "dev.targets" not in json.dumps(literal)
    assert "dev.targets" not in json.dumps(local_frequency)


def test_readout_public_package_hash_manifest_and_paths() -> None:
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
