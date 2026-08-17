import numpy as np
import pytest

from pams.config import (
    DataConfig,
    ModelConfig,
    PAMSConfig,
    PeriodConfig,
)
from pams.evaluation import (
    PredictionRecord,
    evaluate_model,
    evaluate_predictions,
)
from pams.training import build_pams_model
from pams.types import CountResult, PoseSequence


def _record(identifier: str, count: int) -> PredictionRecord:
    return PredictionRecord(
        video_id=identifier,
        result=CountResult(
            count=count,
            period_frames=8.0,
            expert_counts=(count, count, count),
            confidence=1.0,
            period_stream=np.zeros(16, dtype=np.float32),
        ),
    )


def _sequence(identifier: str) -> PoseSequence:
    return PoseSequence(
        video_id=identifier,
        fps=16.0,
        xyz=np.zeros((16, 33, 3), dtype=np.float32),
        valid_mask=np.ones(16, dtype=np.bool_),
    )


def test_evaluator_introduces_targets_after_prediction_records() -> None:
    result = evaluate_predictions(
        (_record("a", 4), _record("b", 7)),
        {"a": 5, "b": 7},
        actions={"a": "jump", "b": "pushup"},
        bootstrap_samples=0,
    )
    assert result.report.sample_count == 2
    assert result.report.nmae == pytest.approx(0.1)
    assert result.report.obo == 1.0
    assert result.report.per_video[0].action == "jump"
    assert result.predictions[1].to_dict(include_stream=False)["count"] == 7


def test_evaluator_rejects_missing_or_unexpected_target_ids() -> None:
    with pytest.raises(ValueError, match="exactly match"):
        evaluate_predictions(
            (_record("a", 4),),
            {"other": 4},
            bootstrap_samples=0,
        )


def test_evaluate_model_cpu_smoke() -> None:
    config = PAMSConfig(
        data=DataConfig(frames=16),
        model=ModelConfig(
            input_dim=99,
            model_dim=8,
            embedding_dim=8,
            layers=1,
            heads=2,
            feedforward_dim=16,
            dropout=0.0,
            period_head_hidden_dim=4,
        ),
        period=PeriodConfig(minimum=4, maximum=8, pose_energy_epochs=1),
    )
    model = build_pams_model(config)
    sequences = (_sequence("a"), _sequence("b"))
    output = evaluate_model(
        model,
        sequences,
        config,
        {"a": 2, "b": 3},
        device="cpu",
        bootstrap_samples=0,
    )
    assert output.report.sample_count == 2
    assert [record.video_id for record in output.predictions] == ["a", "b"]
    assert all(record.result.period_stream.shape == (16,) for record in output.predictions)
