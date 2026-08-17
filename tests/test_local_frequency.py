from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from pams.evaluation import evaluate_predictions
from pams.local_frequency import (
    LocalFrequencyReadout,
    load_local_frequency_config,
    normalized_file_sha256,
    predict_local_frequency_sequences,
    replay_synthetic_freeze,
)
from pams.synthetic import SyntheticSpec, generate_synthetic_sample
from pams.types import PoseSequence

REPOSITORY = Path(__file__).parents[1]
CONFIG_PATH = REPOSITORY / "configs/readouts/local_frequency_synthetic_v1.yaml"


def _config():
    return load_local_frequency_config(CONFIG_PATH)


def test_config_is_explicitly_inferred_ineligible_and_scope_bound() -> None:
    config = _config()
    assert config.classification == "inferred synthetic-frozen label-free readout"
    assert config.eligible_for_paper_table is False
    assert config.selection.label_source == "synthetic_generation_truth_only"
    assert all("UCFRep" in source for source in config.selection.prohibited_label_sources)
    assert config.selected_candidate == "frame_centered.min064.median"
    assert normalized_file_sha256(REPOSITORY / config.selection.scope_config) == (
        config.selection.scope_config_sha256
    )


def test_unknown_config_keys_are_rejected(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["selection"]["ucfrep_dev_targets"] = "forbidden.json"
    destination = tmp_path / "invalid.yaml"
    destination.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_local_frequency_config(destination)


def test_loader_rejects_tampered_selection_scope(tmp_path: Path) -> None:
    scope = tmp_path / "configs/stress.yaml"
    scope.parent.mkdir(parents=True)
    scope.write_text("tampered: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_local_frequency_config(CONFIG_PATH, repository_root=tmp_path)


def test_synthetic_freeze_replays_selected_candidate_and_metrics() -> None:
    config = _config()
    report = replay_synthetic_freeze(config, repository_root=REPOSITORY)
    expected = config.selection.expected_replay
    selected = report.selected_score

    assert report.selected_candidate == expected.selected_candidate
    assert len(report.sample_ids) == expected.sample_count
    assert len(set(report.sample_ids)) == expected.sample_count
    assert len(report.scores) == 12
    assert selected.mean_absolute_error == pytest.approx(expected.mean_absolute_error)
    assert selected.normalized_mean_absolute_error == pytest.approx(
        expected.normalized_mean_absolute_error
    )
    assert selected.exact_rate == pytest.approx(expected.exact_rate)
    assert selected.obo == pytest.approx(expected.obo)


def test_readout_is_deterministic_and_translation_invariant() -> None:
    config = _config()
    readout = LocalFrequencyReadout(config)
    original = generate_synthetic_sample(
        SyntheticSpec(video_id="original", count=11, seed=91)
    ).sequence
    translated = generate_synthetic_sample(
        SyntheticSpec(
            video_id="translated",
            count=11,
            seed=91,
            translation=(0.1, -0.1, 0.05),
        )
    ).sequence

    first = readout.predict(original)
    second = readout.predict(original)
    shifted = readout.predict(translated)
    assert first.to_dict() == second.to_dict()
    assert first.count == shifted.count == 11
    assert first.period_frames == pytest.approx(shifted.period_frames, rel=1e-5)
    assert first.period_stream.shape == (original.num_frames,)
    assert 0.0 <= first.confidence <= 1.0


def test_readout_handles_constant_and_fully_invalid_sequences() -> None:
    readout = LocalFrequencyReadout(_config())
    constant = PoseSequence(
        video_id="constant",
        fps=30.0,
        xyz=np.ones((32, 33, 3), dtype=np.float32),
        valid_mask=np.ones(32, dtype=np.bool_),
    )
    invalid = PoseSequence(
        video_id="invalid",
        fps=30.0,
        xyz=np.ones((32, 33, 3), dtype=np.float32),
        valid_mask=np.zeros(32, dtype=np.bool_),
    )
    for sample in (constant, invalid):
        result = readout.predict(sample)
        assert result.count == 0
        assert result.expert_counts == (0, 0, 0)
        assert result.confidence == 0.0
        assert result.period_frames == 128.0
        np.testing.assert_array_equal(result.period_stream, np.zeros(32))


def test_standard_prediction_records_feed_sealed_evaluator() -> None:
    readout = LocalFrequencyReadout(_config())
    samples = tuple(
        generate_synthetic_sample(
            SyntheticSpec(video_id=f"sample-{count}", count=count, seed=100 + count)
        ).sequence
        for count in (5, 13)
    )
    records = predict_local_frequency_sequences(readout, samples)
    result = evaluate_predictions(
        records,
        {"sample-5": 5, "sample-13": 13},
        bootstrap_samples=0,
    )
    assert [record.video_id for record in records] == ["sample-5", "sample-13"]
    assert result.report.nmae == 0.0
    assert result.report.obo == 1.0


def test_prediction_records_reject_duplicate_ids() -> None:
    sample = generate_synthetic_sample(SyntheticSpec(video_id="duplicate")).sequence
    with pytest.raises(ValueError, match="unique"):
        predict_local_frequency_sequences(LocalFrequencyReadout(_config()), (sample, sample))
