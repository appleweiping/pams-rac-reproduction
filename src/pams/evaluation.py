"""Sealed-boundary prediction and evaluation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pams.config import PAMSConfig
from pams.metrics import MetricReport, compute_count_metrics
from pams.model import PAMSModel
from pams.training import predict_sequence
from pams.types import CountResult, PoseSequence


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One label-free, auditable model prediction."""

    video_id: str
    result: CountResult

    def __post_init__(self) -> None:
        if not str(self.video_id).strip():
            raise ValueError("video_id must be non-empty")
        if not isinstance(self.result, CountResult):
            raise TypeError("result must be a CountResult")

    @property
    def prediction(self) -> int:
        return self.result.count

    def to_dict(self, *, include_stream: bool = True) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            **self.result.to_dict(include_stream=include_stream),
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Predictions together with the evaluator-only metric report."""

    report: MetricReport
    predictions: tuple[PredictionRecord, ...]

    def __post_init__(self) -> None:
        if self.report.sample_count != len(self.predictions):
            raise ValueError("report and prediction lengths must match")

    def to_dict(self, *, include_streams: bool = True) -> dict[str, Any]:
        return {
            "report": self.report.to_dict(),
            "predictions": [
                record.to_dict(include_stream=include_streams) for record in self.predictions
            ],
        }


def predict_sequences(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    config: PAMSConfig,
    *,
    device: str | None = None,
) -> tuple[PredictionRecord, ...]:
    """Run label-free prediction and preserve input order."""

    items = tuple(sequences)
    if not items:
        raise ValueError("prediction requires at least one PoseSequence")
    identifiers = [item.video_id for item in items]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("prediction video_id values must be unique")
    return tuple(
        PredictionRecord(
            video_id=sequence.video_id,
            result=predict_sequence(
                model,
                sequence,
                config,
                device=device,
            ),
        )
        for sequence in items
    )


def evaluate_predictions(
    predictions: Sequence[PredictionRecord],
    ground_truths: Mapping[str, int],
    *,
    actions: Mapping[str, str] | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 2026,
    confidence_level: float = 0.95,
) -> EvaluationResult:
    """Introduce counts only at the sealed evaluator boundary."""

    records = tuple(predictions)
    if not records:
        raise ValueError("evaluation requires at least one prediction")
    identifiers = tuple(record.video_id for record in records)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("prediction video_id values must be unique")
    target_ids = set(ground_truths)
    prediction_ids = set(identifiers)
    if target_ids != prediction_ids:
        missing = sorted(prediction_ids - target_ids)
        unexpected = sorted(target_ids - prediction_ids)
        raise ValueError(
            "ground-truth identifiers must exactly match predictions; "
            f"missing={missing}, unexpected={unexpected}"
        )
    action_values = (
        None if actions is None else tuple(actions.get(identifier) for identifier in identifiers)
    )
    report = compute_count_metrics(
        [record.prediction for record in records],
        [ground_truths[identifier] for identifier in identifiers],
        video_ids=identifiers,
        actions=action_values,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
    return EvaluationResult(report=report, predictions=records)


def evaluate_model(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    config: PAMSConfig,
    ground_truths: Mapping[str, int],
    *,
    actions: Mapping[str, str] | None = None,
    device: str | None = None,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 2026,
    confidence_level: float = 0.95,
) -> EvaluationResult:
    """Predict without labels, then pass records into the sealed evaluator."""

    predictions = predict_sequences(model, sequences, config, device=device)
    return evaluate_predictions(
        predictions,
        ground_truths,
        actions=actions,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        confidence_level=confidence_level,
    )
