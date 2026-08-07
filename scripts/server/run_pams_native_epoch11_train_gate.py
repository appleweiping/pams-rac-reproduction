"""Read-only train337 mechanism gate for a native fixed-period encoder.

The command accepts no dataset manifest, media, annotation, class, repetition,
development, or sealed-evaluation interface.  It binds an epoch-11 encoder and
progress log to their checkpoint provenance, the exact train337 pose-cache
snapshot, one experiment config, and one preregistered gate specification.

The scientific gate asks only whether optimization produced non-collapsed,
pose-conditioned recurrence at the fixed training period.  It deliberately
does not use the adaptive-period time-scale checks from earlier experiments:
those checks conflict with a fixed-window baseline and are not a label-free
measure of its epoch-11 training mechanism.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.nn import functional as F

from pams.config import PAMSConfig, load_config
from pams.data import (
    PoseCacheEntryReceipt,
    PoseCacheSetSnapshot,
    _reject_json_object_keys_before_deserialization,
)
from pams.diagnostics import (
    _device,
    _identifier_commitment,
    _load_training_poses,
    _peek_checkpoint,
    _stable_file_sha256,
    _summary,
)
from pams.recurrence_carrier import build_recurrence_carrier_curves
from pams.reproducibility import clean_git_revision
from pams.training import (
    EncoderEpochStats,
    _progress_row,
    _read_progress_rows,
    collate_pose_sequences,
    load_model_checkpoint,
)
from pams.types import PoseSequence

_CHECKPOINT_SCHEMA_VERSION = 5
_PROGRESS_SCHEMA_VERSION = 2
_GATE_SCHEMA_VERSION = 1
_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "action",
        "actions",
        "count",
        "counts",
        "dev",
        "development",
        "split",
        "target",
        "targets",
        "test",
    }
)
_FORBIDDEN_PATH_TOKEN = re.compile(
    r"(?:^|[^a-z0-9])(?:dev(?:elopment)?[0-9]*|test[0-9]*|targets?|counts?)(?:$|[^a-z0-9])",
    flags=re.IGNORECASE,
)
_GATE_KEYS = {
    "schema_version",
    "artifact_type",
    "classification",
    "expected_protocol",
    "expected_seed",
    "expected_training_video_total",
    "expected_completed_epochs",
    "minimum_lag_pair_total",
    "near_collapse_rms",
    "thresholds",
}
_THRESHOLD_KEYS = {
    "loss_relative_drop_minimum",
    "fixed_period_evidence_fraction_minimum",
    "near_collapsed_fraction_maximum",
    "lag_eligible_fraction_minimum",
    "real_cycle_margin_median_minimum",
    "stronger_null_separation_median_minimum",
    "real_beats_both_nulls_fraction_minimum",
}


@dataclass(frozen=True, slots=True)
class GateSpecification:
    artifact_type: str
    classification: str
    expected_protocol: str
    expected_seed: int
    expected_training_video_total: int
    expected_completed_epochs: int
    minimum_lag_pair_total: int
    near_collapse_rms: float
    thresholds: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class MechanismSample:
    video_id: str
    temporal_rms: float
    lag_eligible: bool
    real_cycle_margin: float | None
    shuffled_cycle_margin: float | None
    zero_cycle_margin: float | None
    stronger_null_separation: float | None
    real_beats_both_nulls: bool | None
    fixed_carrier_available: bool
    fixed_carrier_support: float
    fixed_carrier_gate_energy: float
    shuffled_carrier_available: bool
    shuffled_carrier_support: float
    shuffled_carrier_gate_energy: float
    zero_carrier_available: bool
    zero_carrier_support: float
    zero_carrier_gate_energy: float


def _strict_json(encoded: str, *, document: str) -> dict[str, Any]:
    _reject_json_object_keys_before_deserialization(
        encoded,
        forbidden=_FORBIDDEN_FIELD_NAMES,
        document_name=document,
    )

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{document} contains duplicate JSON field {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{document} contains non-finite JSON constant {value}")

    payload = json.loads(
        encoded,
        object_pairs_hook=reject_duplicate,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"{document} root must be a JSON object")
    return payload


def _reject_privileged_path(path: Path, *, role: str) -> None:
    normalized = str(path.resolve()).replace("\\", "/")
    if _FORBIDDEN_PATH_TOKEN.search(normalized):
        raise ValueError(f"{role} path contains a forbidden privileged token")


def _require_source_tree_membership(
    source_root: Path,
    paths: Mapping[str, Path],
) -> dict[str, str]:
    root = source_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("source root must be a directory")
    relative_paths: dict[str, str] = {}
    for role, path in paths.items():
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"source-bound {role} must be a regular file")
        try:
            relative = resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"source-bound {role} is outside the source root") from exc
        relative_paths[role] = relative.as_posix()
    return relative_paths


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def load_gate_specification(path: str | Path) -> GateSpecification:
    source = Path(path)
    _reject_privileged_path(source, role="gate specification")
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != _GATE_KEYS:
        raise ValueError("gate specification schema mismatch")
    if raw["schema_version"] != _GATE_SCHEMA_VERSION:
        raise ValueError("unsupported gate specification schema")
    thresholds = raw["thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != _THRESHOLD_KEYS:
        raise ValueError("gate threshold schema mismatch")
    normalized_thresholds = {
        key: _finite_number(value, name=f"thresholds.{key}") for key, value in thresholds.items()
    }
    unit_interval = {
        "loss_relative_drop_minimum",
        "fixed_period_evidence_fraction_minimum",
        "near_collapsed_fraction_maximum",
        "lag_eligible_fraction_minimum",
        "real_beats_both_nulls_fraction_minimum",
    }
    if any(not 0.0 <= normalized_thresholds[key] <= 1.0 for key in unit_interval):
        raise ValueError("fraction thresholds must lie in [0, 1]")
    near_collapse = _finite_number(raw["near_collapse_rms"], name="near_collapse_rms")
    if near_collapse <= 0.0:
        raise ValueError("near_collapse_rms must be positive")
    return GateSpecification(
        artifact_type=str(raw["artifact_type"]).strip(),
        classification=str(raw["classification"]).strip(),
        expected_protocol=str(raw["expected_protocol"]).strip(),
        expected_seed=_positive_integer(raw["expected_seed"], name="expected_seed"),
        expected_training_video_total=_positive_integer(
            raw["expected_training_video_total"],
            name="expected_training_video_total",
        ),
        expected_completed_epochs=_positive_integer(
            raw["expected_completed_epochs"],
            name="expected_completed_epochs",
        ),
        minimum_lag_pair_total=_positive_integer(
            raw["minimum_lag_pair_total"],
            name="minimum_lag_pair_total",
        ),
        near_collapse_rms=near_collapse,
        thresholds=normalized_thresholds,
    )


def _validate_candidate_config(
    config: PAMSConfig,
    specification: GateSpecification,
) -> None:
    actual = {
        "protocol": config.protocol,
        "seed": config.seed,
        "training_epochs": config.training.epochs,
        "period_training_mode": config.period.training_mode,
        "fixed_period_frames": config.period.fixed_period_frames,
        "loss_scales": config.loss.scales,
        "use_cross_cluster_negatives": config.loss.use_cross_cluster_negatives,
    }
    if actual["protocol"] != specification.expected_protocol:
        raise ValueError("candidate protocol differs from gate specification")
    if actual["seed"] != specification.expected_seed:
        raise ValueError("candidate seed differs from gate specification")
    if actual["training_epochs"] < specification.expected_completed_epochs:
        raise ValueError("candidate training schedule ends before the gate epoch")
    if actual["period_training_mode"] != "fixed_period_inferred":
        raise ValueError("native epoch-11 gate requires fixed_period_inferred")
    if actual["fixed_period_frames"] < 4:
        raise ValueError("fixed period must be at least four frames")
    if actual["loss_scales"] != (1.0,):
        raise ValueError("native baseline gate requires one conventional-TCC scale")
    if actual["use_cross_cluster_negatives"] is not False:
        raise ValueError("native baseline gate requires prototype-bank negatives off")


def _checkpoint_epoch_metadata(
    checkpoint: Path,
    progress: Path,
    *,
    config: PAMSConfig,
    specification: GateSpecification,
) -> tuple[dict[str, Any], tuple[EncoderEpochStats, ...]]:
    progress_encoded = progress.read_text(encoding="utf-8")
    _reject_json_object_keys_before_deserialization(
        progress_encoded,
        forbidden=_FORBIDDEN_FIELD_NAMES,
        document_name="encoder progress",
    )
    payload = torch.load(
        checkpoint,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    expected_keys = {
        "schema_version",
        "stage",
        "config_fingerprint",
        "provenance",
        "completed_epochs",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "history",
        "cluster_assignments",
        "prototype_bank",
        "rng_state",
    }
    if set(payload) != expected_keys:
        raise ValueError("checkpoint top-level schema mismatch")
    if payload["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    if payload["stage"] != "encoder":
        raise ValueError("gate requires an encoder checkpoint")
    if payload["config_fingerprint"] != config.fingerprint:
        raise ValueError("checkpoint config fingerprint mismatch")
    expected_epochs = specification.expected_completed_epochs
    if payload["completed_epochs"] != expected_epochs:
        raise ValueError(f"gate requires completed_epochs={expected_epochs}")
    for name in (
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "cluster_assignments",
        "prototype_bank",
        "rng_state",
    ):
        if not isinstance(payload[name], Mapping):
            raise ValueError(f"checkpoint {name} must be a mapping")

    raw_history = payload["history"]
    if not isinstance(raw_history, list) or len(raw_history) != expected_epochs:
        raise ValueError("checkpoint history length mismatch")
    field_names = {field.name for field in fields(EncoderEpochStats)}
    integer_fields = {
        "epoch",
        "optimizer_steps",
        "cross_cluster_requested",
        "cross_cluster_actual",
        "cross_cluster_shortfall",
    }
    float_fields = {
        "loss",
        "learning_rate",
        "period_confidence_mean",
        "period_valid_fraction",
        "position_permutation_consistency",
    }
    history: list[EncoderEpochStats] = []
    for expected_epoch, raw_row in enumerate(raw_history, start=1):
        if not isinstance(raw_row, dict):
            raise ValueError("checkpoint history rows must be mappings")
        row = dict(raw_row)
        row.setdefault("position_permutation_consistency", 0.0)
        if set(row) != field_names:
            raise ValueError("checkpoint history row schema mismatch")
        for name in integer_fields:
            if isinstance(row[name], bool) or not isinstance(row[name], int):
                raise ValueError(f"checkpoint history {name} must be an integer")
        for name in float_fields:
            if isinstance(row[name], bool) or not isinstance(row[name], float):
                raise ValueError(f"checkpoint history {name} must be a float")
        if not isinstance(row["period_source"], str):
            raise ValueError("checkpoint history period_source must be text")
        if not isinstance(row["clusters_refreshed"], bool):
            raise ValueError("checkpoint history clusters_refreshed must be boolean")
        statistics = EncoderEpochStats(**row)
        if statistics.epoch != expected_epoch:
            raise ValueError("checkpoint history epochs must be consecutive")
        numeric = asdict(statistics)
        for name, value in numeric.items():
            if name in {"period_source", "clusters_refreshed"}:
                continue
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"checkpoint history {name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"checkpoint history {name} must be finite")
        if statistics.period_source != "fixed_period_inferred":
            raise ValueError("every gate epoch must use the fixed-period source")
        if abs(statistics.period_confidence_mean - statistics.period_valid_fraction) > 1e-12:
            raise ValueError("fixed-period confidence and valid fraction differ")
        if statistics.optimizer_steps < 1:
            raise ValueError("every gate epoch must contain an optimizer step")
        if (
            statistics.cross_cluster_requested
            or statistics.cross_cluster_actual
            or statistics.cross_cluster_shortfall
        ):
            raise ValueError("prototype-bank negative accounting must remain zero")
        history.append(statistics)

    rows = _read_progress_rows(progress)
    if len(rows) != expected_epochs:
        raise ValueError("progress row total differs from gate epoch")
    if any(row.get("schema_version") != _PROGRESS_SCHEMA_VERSION for row in rows):
        raise ValueError("progress schema mismatch")
    expected_rows = tuple(
        _progress_row(stage="encoder", stats=item, config=config) for item in history
    )
    if rows != expected_rows:
        raise ValueError("progress rows do not exactly match checkpoint history")
    first_losses = np.asarray([item.loss for item in history[:3]], dtype=np.float64)
    final_losses = np.asarray([item.loss for item in history[-3:]], dtype=np.float64)
    first_median = float(np.median(first_losses))
    final_median = float(np.median(final_losses))
    if first_median <= 0.0:
        raise ValueError("initial loss median must be positive")
    return (
        {
            "completed_epochs": expected_epochs,
            "period_source": "fixed_period_inferred",
            "loss_first_three_median": first_median,
            "loss_final_three_median": final_median,
            "loss_relative_drop": 1.0 - final_median / first_median,
            "fixed_period_evidence_fraction_mean": float(
                np.mean([item.period_valid_fraction for item in history])
            ),
            "optimizer_steps_total": sum(item.optimizer_steps for item in history),
            "prototype_bank_negative_tallies_all_zero": True,
            "progress_exactly_matches_checkpoint_history": True,
        },
        tuple(history),
    )


def _load_snapshot(path: Path) -> PoseCacheSetSnapshot:
    payload = _strict_json(path.read_text(encoding="utf-8"), document="pose snapshot")
    expected = {
        "schema_version",
        "pose_fingerprint",
        "fingerprint",
        "entry_count",
        "entries",
    }
    if set(payload) != expected or payload["schema_version"] != 1:
        raise ValueError("pose snapshot schema mismatch")
    raw_entries = payload["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError("pose snapshot entries must be a list")
    entries: list[PoseCacheEntryReceipt] = []
    for item in raw_entries:
        if not isinstance(item, dict) or set(item) != {"video_id", "cache_sha256", "bytes"}:
            raise ValueError("pose snapshot entry schema mismatch")
        entries.append(
            PoseCacheEntryReceipt(
                video_id=item["video_id"],
                cache_sha256=item["cache_sha256"],
                bytes=item["bytes"],
            )
        )
    snapshot = PoseCacheSetSnapshot(
        schema_version=payload["schema_version"],
        pose_fingerprint=payload["pose_fingerprint"],
        entries=tuple(entries),
    )
    if payload["entry_count"] != len(snapshot.entries):
        raise ValueError("pose snapshot entry total mismatch")
    if payload["fingerprint"] != snapshot.fingerprint:
        raise ValueError("pose snapshot fingerprint mismatch")
    return snapshot


def _shuffle_valid_pose_rows(
    poses: Tensor,
    valid_mask: Tensor,
    video_ids: Sequence[str],
) -> Tensor:
    if poses.ndim != 4 or valid_mask.shape != poses.shape[:2]:
        raise ValueError("shuffle expects pose batch and matching valid mask")
    if len(video_ids) != poses.shape[0]:
        raise ValueError("shuffle video IDs must match pose batch")
    shuffled = poses.detach().clone()
    for sample_index, video_id in enumerate(video_ids):
        valid_indices = torch.nonzero(
            valid_mask[sample_index].detach().cpu(),
            as_tuple=False,
        ).flatten()
        if valid_indices.numel() < 2:
            continue
        digest = hashlib.sha256(
            f"pams-native-epoch11-pose-shuffle-v1\0{video_id}".encode()
        ).digest()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))
        permutation = torch.randperm(valid_indices.numel(), generator=generator)
        destination = valid_indices.to(device=poses.device)
        source = valid_indices[permutation].to(device=poses.device)
        shuffled[sample_index, destination] = poses[sample_index, source]
    return shuffled


def _temporal_rms(embeddings: Tensor, valid_mask: Tensor, timeline_length: int) -> float:
    selected = embeddings[:timeline_length][valid_mask[:timeline_length]]
    if selected.shape[0] < 2:
        return 0.0
    centered = selected - selected.mean(dim=0, keepdim=True)
    return float(centered.square().sum(dim=1).mean().sqrt())


def _lag_similarity(
    embeddings: Tensor,
    valid_mask: Tensor,
    *,
    timeline_length: int,
    lag: int,
    minimum_pairs: int,
) -> float | None:
    if lag >= timeline_length:
        return None
    values = F.normalize(
        embeddings[:timeline_length].detach().to(dtype=torch.float32),
        p=2,
        dim=1,
        eps=1e-12,
    )
    valid = valid_mask[:timeline_length].to(dtype=torch.bool)
    pair_valid = valid[:-lag] & valid[lag:]
    if int(pair_valid.sum()) < minimum_pairs:
        return None
    similarities = (values[:-lag] * values[lag:]).sum(dim=1)
    return float(similarities[pair_valid].mean())


def _recurrence_lags(period: int) -> tuple[int, int, int]:
    """Return P/2, P, and 3P/2 using explicit round-half-up semantics."""

    if isinstance(period, bool) or not isinstance(period, int) or period < 2:
        raise ValueError("period must be an integer of at least two frames")
    half = max(1, math.floor(0.5 * period + 0.5))
    three_half = max(period + 1, math.floor(1.5 * period + 0.5))
    return half, period, three_half


def _cycle_margin(
    embeddings: Tensor,
    valid_mask: Tensor,
    *,
    timeline_length: int,
    period: int,
    minimum_pairs: int,
) -> float | None:
    lags = _recurrence_lags(period)
    values = tuple(
        _lag_similarity(
            embeddings,
            valid_mask,
            timeline_length=timeline_length,
            lag=lag,
            minimum_pairs=minimum_pairs,
        )
        for lag in lags
    )
    if any(value is None for value in values):
        return None
    half, full, three_half = values
    assert half is not None and full is not None and three_half is not None
    return full - 0.5 * (half + three_half)


def mechanism_samples_from_embeddings(
    *,
    video_ids: Sequence[str],
    real_embeddings: Tensor,
    shuffled_embeddings: Tensor,
    zero_embeddings: Tensor,
    valid_mask: Tensor,
    timeline_lengths: Tensor,
    period: int,
    minimum_pairs: int,
) -> tuple[MechanismSample, ...]:
    expected = real_embeddings.shape
    if (
        real_embeddings.ndim != 3
        or shuffled_embeddings.shape != expected
        or zero_embeddings.shape != expected
    ):
        raise ValueError("all embedding batches must share [batch, time, dimension]")
    if valid_mask.shape != expected[:2] or timeline_lengths.shape != expected[:1]:
        raise ValueError("embedding masks or timeline lengths are incompatible")
    if len(video_ids) != expected[0]:
        raise ValueError("embedding video IDs must match batch")
    fixed_periods = torch.full(
        (expected[0],),
        float(period),
        dtype=real_embeddings.dtype,
        device=real_embeddings.device,
    )
    confidences = torch.ones_like(fixed_periods)
    carrier_arguments = {
        "periods": fixed_periods,
        "valid_mask": valid_mask,
        "timeline_lengths": timeline_lengths,
        "period_confidences": confidences,
    }
    real_carrier = build_recurrence_carrier_curves(real_embeddings, **carrier_arguments)
    shuffled_carrier = build_recurrence_carrier_curves(
        shuffled_embeddings,
        **carrier_arguments,
    )
    zero_carrier = build_recurrence_carrier_curves(zero_embeddings, **carrier_arguments)
    samples: list[MechanismSample] = []
    for index, video_id in enumerate(video_ids):
        timeline_length = int(timeline_lengths[index])
        margins = tuple(
            _cycle_margin(
                embeddings[index],
                valid_mask[index],
                timeline_length=timeline_length,
                period=period,
                minimum_pairs=minimum_pairs,
            )
            for embeddings in (real_embeddings, shuffled_embeddings, zero_embeddings)
        )
        real_margin, shuffled_margin, zero_margin = margins
        eligible = all(value is not None for value in margins)
        separation: float | None = None
        beats_nulls: bool | None = None
        if eligible:
            assert real_margin is not None
            assert shuffled_margin is not None
            assert zero_margin is not None
            separation = real_margin - max(shuffled_margin, zero_margin)
            beats_nulls = separation > 0.0
        samples.append(
            MechanismSample(
                video_id=str(video_id),
                temporal_rms=_temporal_rms(
                    real_embeddings[index],
                    valid_mask[index],
                    timeline_length,
                ),
                lag_eligible=eligible,
                real_cycle_margin=real_margin,
                shuffled_cycle_margin=shuffled_margin,
                zero_cycle_margin=zero_margin,
                stronger_null_separation=separation,
                real_beats_both_nulls=beats_nulls,
                fixed_carrier_available=bool(real_carrier.available[index]),
                fixed_carrier_support=float(real_carrier.active_support_fractions[index]),
                fixed_carrier_gate_energy=float(real_carrier.recurrence_gate_energies[index]),
                shuffled_carrier_available=bool(shuffled_carrier.available[index]),
                shuffled_carrier_support=float(shuffled_carrier.active_support_fractions[index]),
                shuffled_carrier_gate_energy=float(
                    shuffled_carrier.recurrence_gate_energies[index]
                ),
                zero_carrier_available=bool(zero_carrier.available[index]),
                zero_carrier_support=float(zero_carrier.active_support_fractions[index]),
                zero_carrier_gate_energy=float(zero_carrier.recurrence_gate_energies[index]),
            )
        )
    return tuple(samples)


def _encode_mechanism_samples(
    model: nn.Module,
    sequences: Sequence[PoseSequence],
    *,
    period: int,
    minimum_pairs: int,
    device: torch.device,
    batch_size: int,
) -> tuple[MechanismSample, ...]:
    samples: list[MechanismSample] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            shuffled_poses = _shuffle_valid_pose_rows(
                batch.poses,
                batch.valid_mask,
                batch.video_ids,
            )
            real = model.encoder(batch.poses, batch.valid_mask)
            shuffled = model.encoder(shuffled_poses, batch.valid_mask)
            zero = model.encoder(torch.zeros_like(batch.poses), batch.valid_mask)
            samples.extend(
                mechanism_samples_from_embeddings(
                    video_ids=batch.video_ids,
                    real_embeddings=real,
                    shuffled_embeddings=shuffled,
                    zero_embeddings=zero,
                    valid_mask=batch.valid_mask,
                    timeline_lengths=batch.lengths,
                    period=period,
                    minimum_pairs=minimum_pairs,
                )
            )
    return tuple(samples)


def _median(values: Sequence[float]) -> float | None:
    return None if not values else float(np.median(np.asarray(values, dtype=np.float64)))


def _mechanism_distribution(
    samples: Sequence[MechanismSample],
    *,
    near_collapse_rms: float,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("mechanism distribution requires samples")
    eligible = tuple(sample for sample in samples if sample.lag_eligible)
    real = [float(sample.real_cycle_margin) for sample in eligible]
    shuffled = [float(sample.shuffled_cycle_margin) for sample in eligible]
    zero = [float(sample.zero_cycle_margin) for sample in eligible]
    separation = [float(sample.stronger_null_separation) for sample in eligible]
    temporal_rms = [sample.temporal_rms for sample in samples]
    carrier_real_energy = [sample.fixed_carrier_gate_energy for sample in samples]
    carrier_shuffled_energy = [sample.shuffled_carrier_gate_energy for sample in samples]
    real_energy_median = _median(carrier_real_energy)
    shuffled_energy_median = _median(carrier_shuffled_energy)
    return {
        "record_total": len(samples),
        "temporal_rms": _summary(temporal_rms),
        "near_collapse_rms": near_collapse_rms,
        "near_collapsed_fraction": float(
            np.mean(np.asarray(temporal_rms, dtype=np.float64) <= near_collapse_rms)
        ),
        "lag_eligible_total": len(eligible),
        "lag_eligible_fraction": len(eligible) / len(samples),
        "real_cycle_margin": _summary(real),
        "shuffled_cycle_margin": _summary(shuffled),
        "zero_cycle_margin": _summary(zero),
        "stronger_null_separation": _summary(separation),
        "real_beats_both_nulls_fraction": (
            float(np.mean([bool(sample.real_beats_both_nulls) for sample in eligible]))
            if eligible
            else 0.0
        ),
        "fixed_period_carrier_advisory": {
            "authorization_role": "diagnostic_only",
            "real_available_fraction": float(
                np.mean([sample.fixed_carrier_available for sample in samples])
            ),
            "real_support": _summary([sample.fixed_carrier_support for sample in samples]),
            "real_gate_energy": _summary(carrier_real_energy),
            "shuffled_available_fraction": float(
                np.mean([sample.shuffled_carrier_available for sample in samples])
            ),
            "shuffled_support": _summary([sample.shuffled_carrier_support for sample in samples]),
            "shuffled_gate_energy": _summary(carrier_shuffled_energy),
            "zero_available_fraction": float(
                np.mean([sample.zero_carrier_available for sample in samples])
            ),
            "zero_support": _summary([sample.zero_carrier_support for sample in samples]),
            "zero_gate_energy": _summary([sample.zero_carrier_gate_energy for sample in samples]),
            "shuffled_to_real_gate_energy_median_ratio": (
                None
                if real_energy_median is None
                or real_energy_median <= 1e-12
                or shuffled_energy_median is None
                else shuffled_energy_median / real_energy_median
            ),
        },
    }


def _criterion(value: float | None, *, operator: str, threshold: float) -> dict[str, Any]:
    passed = False
    if value is not None and math.isfinite(value):
        if operator == ">=":
            passed = value >= threshold
        elif operator == "<=":
            passed = value <= threshold
        else:
            raise ValueError("unsupported gate operator")
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "pass": passed,
    }


def gate_decision(
    *,
    schedule: Mapping[str, Any],
    distribution: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    criteria = {
        "loss_relative_drop": _criterion(
            float(schedule["loss_relative_drop"]),
            operator=">=",
            threshold=thresholds["loss_relative_drop_minimum"],
        ),
        "fixed_period_evidence_fraction": _criterion(
            float(schedule["fixed_period_evidence_fraction_mean"]),
            operator=">=",
            threshold=thresholds["fixed_period_evidence_fraction_minimum"],
        ),
        "near_collapsed_fraction": _criterion(
            float(distribution["near_collapsed_fraction"]),
            operator="<=",
            threshold=thresholds["near_collapsed_fraction_maximum"],
        ),
        "lag_eligible_fraction": _criterion(
            float(distribution["lag_eligible_fraction"]),
            operator=">=",
            threshold=thresholds["lag_eligible_fraction_minimum"],
        ),
        "real_cycle_margin_median": _criterion(
            distribution["real_cycle_margin"]["median"],
            operator=">=",
            threshold=thresholds["real_cycle_margin_median_minimum"],
        ),
        "stronger_null_separation_median": _criterion(
            distribution["stronger_null_separation"]["median"],
            operator=">=",
            threshold=thresholds["stronger_null_separation_median_minimum"],
        ),
        "real_beats_both_nulls_fraction": _criterion(
            float(distribution["real_beats_both_nulls_fraction"]),
            operator=">=",
            threshold=thresholds["real_beats_both_nulls_fraction_minimum"],
        ),
    }
    passed = all(bool(item["pass"]) for item in criteria.values())
    return {
        "thresholds_frozen_before_native_candidate_training": True,
        "criteria": criteria,
        "all_core_criteria_pass": passed,
        "encoder_continuation_authorized": passed,
        "prediction_or_scoring_authorized": False,
    }


def _model_state_sha256(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def run_gate(
    encoder_checkpoint_path: str | Path,
    encoder_progress_path: str | Path,
    config_path: str | Path,
    gate_specification_path: str | Path,
    pose_cache_dir: str | Path,
    pose_snapshot_path: str | Path,
    *,
    device: str | torch.device | None = None,
    batch_size: int = 16,
) -> dict[str, Any]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 32:
        raise ValueError("batch_size must be an integer in [1, 32]")
    paths = {
        "encoder_checkpoint": Path(encoder_checkpoint_path),
        "encoder_progress": Path(encoder_progress_path),
        "experiment_config": Path(config_path),
        "gate_specification": Path(gate_specification_path),
        "pose_cache_directory": Path(pose_cache_dir),
        "pose_snapshot": Path(pose_snapshot_path),
    }
    for role, path in paths.items():
        _reject_privileged_path(path, role=role)
    source_root = Path.cwd().resolve(strict=True)
    source_covered_paths = _require_source_tree_membership(
        source_root,
        {
            "gate_runner": Path(__file__),
            "experiment_config": paths["experiment_config"],
            "gate_specification": paths["gate_specification"],
        },
    )
    file_roles = tuple(role for role in paths if role != "pose_cache_directory")
    identities = {role: _stable_file_sha256(paths[role]) for role in file_roles}
    specification = load_gate_specification(paths["gate_specification"])
    config = load_config(paths["experiment_config"])
    _validate_candidate_config(config, specification)
    stage, provenance = _peek_checkpoint(paths["encoder_checkpoint"], config)
    if stage != "encoder":
        raise ValueError("native mechanism gate requires an encoder checkpoint")
    if len(provenance.training_video_ids) != specification.expected_training_video_total:
        raise ValueError("checkpoint provenance does not bind the expected training set")
    if provenance.upstream_encoder_checkpoint_sha256 is not None:
        raise ValueError("encoder checkpoint unexpectedly names an upstream encoder")
    if provenance.source_git_sha != clean_git_revision(source_root):
        raise ValueError("checkpoint source revision differs from gate source")
    runtime_image = os.environ.get("PAMS_CONTAINER_IMAGE_ID", "").strip()
    runtime_environment = os.environ.get(
        "PAMS_CONTAINER_ENVIRONMENT_SHA256",
        "",
    ).strip()
    runtime_source = os.environ.get("PAMS_CONTAINER_SOURCE_REVISION", "").strip()
    if (
        runtime_image != provenance.container_image_id
        or runtime_environment != provenance.container_environment_sha256
        or runtime_source != provenance.source_git_sha
    ):
        raise ValueError("runtime container identity differs from checkpoint provenance")
    schedule, _ = _checkpoint_epoch_metadata(
        paths["encoder_checkpoint"],
        paths["encoder_progress"],
        config=config,
        specification=specification,
    )
    declared_snapshot = _load_snapshot(paths["pose_snapshot"])
    sequences, selected_receipts, full_receipts, selected_ids = _load_training_poses(
        pose_cache_dir=paths["pose_cache_directory"],
        provenance=provenance,
        config=config,
        sample_size=0,
        seed=config.seed,
    )
    if len(sequences) != specification.expected_training_video_total:
        raise ValueError("pose loader did not return the complete training set")
    selected_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=selected_receipts,
    )
    full_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=full_receipts,
    )
    if selected_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("selected and complete pose snapshots differ")
    if full_snapshot.fingerprint != declared_snapshot.fingerprint:
        raise ValueError("declared pose snapshot differs from cache bytes")
    if full_snapshot.fingerprint != provenance.pose_cache_set_sha256:
        raise ValueError("pose cache bytes differ from checkpoint provenance")

    resolved_device = _device(device)
    model = load_model_checkpoint(
        paths["encoder_checkpoint"],
        config,
        device=resolved_device,
        expected_stage="encoder",
        expected_provenance=provenance,
    ).eval()
    model_state_before = _model_state_sha256(model)
    samples = _encode_mechanism_samples(
        model,
        sequences,
        period=config.period.fixed_period_frames,
        minimum_pairs=specification.minimum_lag_pair_total,
        device=resolved_device,
        batch_size=batch_size,
    )
    model_state_after = _model_state_sha256(model)
    if model_state_after != model_state_before:
        raise RuntimeError("model state changed during read-only gate")
    distribution = _mechanism_distribution(
        samples,
        near_collapse_rms=specification.near_collapse_rms,
    )
    decision = gate_decision(
        schedule=schedule,
        distribution=distribution,
        thresholds=specification.thresholds,
    )

    _, _, final_receipts, final_ids = _load_training_poses(
        pose_cache_dir=paths["pose_cache_directory"],
        provenance=provenance,
        config=config,
        sample_size=2,
        seed=config.seed,
    )
    if final_ids != selected_ids[:2]:
        raise RuntimeError("training pose selection changed during gate")
    final_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=final_receipts,
    )
    if final_snapshot.fingerprint != full_snapshot.fingerprint:
        raise RuntimeError("training pose bytes changed during gate")
    final_identities = {role: _stable_file_sha256(paths[role]) for role in file_roles}
    if final_identities != identities:
        raise RuntimeError("gate file input changed during evaluation")

    payload = {
        "schema_version": 1,
        "artifact_type": specification.artifact_type,
        "status": (
            "encoder_continuation_authorized"
            if decision["encoder_continuation_authorized"]
            else "encoder_continuation_rejected"
        ),
        "classification": specification.classification,
        "protocol": config.protocol,
        "seed": config.seed,
        "label_firewall": {
            "accepted_scientific_inputs": [
                "epoch11_encoder_checkpoint",
                "epoch11_encoder_progress",
                "checkpoint_bound_train337_pose_cache",
                "train337_pose_snapshot",
            ],
            "manifest_interface_supported": False,
            "media_interface_supported": False,
            "external_label_fields_accessed": [],
            "privileged_interface_fields_and_paths_rejected": True,
            "training_interface_supported": False,
        },
        "inputs": {
            **{f"{role}_sha256": value[0] for role, value in identities.items()},
            **{f"{role}_bytes": value[1] for role, value in identities.items()},
            "config_fingerprint": config.fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
            "pose_cache_set_sha256": full_snapshot.fingerprint,
            "training_video_total": len(selected_ids),
            "training_video_ids_sha256": _identifier_commitment(selected_ids),
            "source_git_sha": provenance.source_git_sha,
            "source_receipt_covered_paths": source_covered_paths,
            "container_image_id": provenance.container_image_id,
            "container_environment_sha256": provenance.container_environment_sha256,
        },
        "schedule": schedule,
        "representation": distribution,
        "gate": decision,
        "scientific_scope": {
            "fixed_training_period_frames": config.period.fixed_period_frames,
            "dense_frame_indices_preserved": True,
            "invalid_slots_never_compacted": True,
            "real_shuffled_and_zero_views_share_one_valid_mask": True,
            "fixed_period_carrier_is_advisory_at_epoch11": True,
            "adaptive_time_scale_criterion_used": False,
        },
        "read_only_verification": {
            "all_file_inputs_unchanged": True,
            "pose_cache_set_unchanged": True,
            "model_state_sha256_before": model_state_before,
            "model_state_sha256_after": model_state_after,
            "model_or_optimizer_state_updated": False,
            "training_steps_executed": 0,
            "pose_cache_write_operations": 0,
        },
    }
    if any(key.lower() in _FORBIDDEN_FIELD_NAMES for key in _walk_mapping_keys(payload)):
        raise RuntimeError("gate output unexpectedly contains a privileged field")
    return payload


def _walk_mapping_keys(value: Any) -> tuple[str, ...]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.append(str(key))
            keys.extend(_walk_mapping_keys(item))
    elif isinstance(value, list | tuple):
        for item in value:
            keys.extend(_walk_mapping_keys(item))
    return tuple(keys)


def _encoded_json(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def write_gate_artifact(output: Path, payload: Mapping[str, Any]) -> tuple[Path, str]:
    receipt_path = Path(str(output) + ".receipt.json")
    if output.exists() or receipt_path.exists():
        raise FileExistsError("gate output and receipt must both be new")
    encoded = _encoded_json(payload)
    digest = hashlib.sha256(encoded).hexdigest()
    _write_new(output, encoded)
    receipt = {
        "schema_version": 1,
        "artifact_type": "pams_native_epoch11_train_mechanism_gate_receipt_v1",
        "artifact_sha256": digest,
        "artifact_bytes": len(encoded),
        "artifact_status": payload["status"],
        "encoder_continuation_authorized": payload["gate"]["encoder_continuation_authorized"],
        "encoder_checkpoint_sha256": payload["inputs"]["encoder_checkpoint_sha256"],
        "encoder_progress_sha256": payload["inputs"]["encoder_progress_sha256"],
        "pose_cache_set_sha256": payload["inputs"]["pose_cache_set_sha256"],
        "gate_specification_sha256": payload["inputs"]["gate_specification_sha256"],
        "source_git_sha": payload["inputs"]["source_git_sha"],
    }
    _write_new(receipt_path, _encoded_json(receipt))
    return receipt_path, digest


def _parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-progress", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gate-specification", type=Path, required=True)
    parser.add_argument("--pose-cache-dir", type=Path, required=True)
    parser.add_argument("--pose-snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_arguments(argv)
    _reject_privileged_path(arguments.output, role="gate output")
    payload = run_gate(
        arguments.encoder_checkpoint,
        arguments.encoder_progress,
        arguments.config,
        arguments.gate_specification,
        arguments.pose_cache_dir,
        arguments.pose_snapshot,
        device=arguments.device,
        batch_size=arguments.batch_size,
    )
    receipt, digest = write_gate_artifact(arguments.output, payload)
    print(
        json.dumps(
            {
                "artifact_sha256": digest,
                "output": str(arguments.output),
                "receipt": str(receipt),
                "encoder_continuation_authorized": payload["gate"][
                    "encoder_continuation_authorized"
                ],
            },
            sort_keys=True,
        )
    )
    return 0 if payload["gate"]["encoder_continuation_authorized"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
