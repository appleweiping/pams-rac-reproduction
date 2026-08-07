"""Label-free train337 probe for a pose-phase-anchored inferred SSHead loss.

This is an independently inferred diagnostic, not an implementation of a
loss disclosed by the PAMS authors.  It accepts only the canonical UCFRep-526
train337 pose sidecar and synthetic skeletons.  Counts, actions, development
inputs/targets, and test105 are outside the command-line interface.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW

from pams.config import PAMSConfig, load_config
from pams.consensus import MultiExpertCounter
from pams.data import (
    load_pose_cache_set,
    load_pose_input_manifest,
    preprocess_pose_sequence,
)
from pams.losses import SSHeadLoss
from pams.period import estimate_period_batch
from pams.synthetic import generate_count_sweep, synthetic_stress_suite
from pams.training import (
    _estimate_post_warmup_periods,
    build_pams_model,
    collate_pose_sequences,
)


@dataclass(frozen=True, slots=True)
class Candidate:
    name: str
    phase_weight: float


@dataclass(frozen=True, slots=True)
class PhaseAnchorOutput:
    loss: Tensor
    contributing_samples: int
    confidence_sum: float


_CANDIDATES = (
    Candidate("continuous_confidence_only", 0.0),
    Candidate("continuous_confidence_phase_0p25", 0.25),
    Candidate("continuous_confidence_phase_1p0", 1.0),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_sha256(module: nn.Module) -> str:
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_encoder_checkpoint(
    checkpoint: Path,
    config: PAMSConfig,
    device: torch.device,
) -> nn.Module:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("stage") != "encoder":
        raise ValueError("phase-anchor diagnosis requires an encoder checkpoint")
    if payload.get("config_fingerprint") != config.fingerprint:
        raise ValueError("checkpoint/config fingerprint mismatch")
    state = payload.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("checkpoint model_state must be a mapping")
    model = build_pams_model(config)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    return model


def pose_band_phase_teacher(
    projected_pose: Tensor,
    valid_mask: Tensor,
    periods: Tensor,
    period_confidence: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build a signed pose-dependent waveform in the target frequency band.

    For each sample, projected pose is centered, transformed to the frequency
    domain, and reconstructed from the target bin plus its two neighbors.  A
    small QR/eigendecomposition finds the dominant temporal component in that
    at-most-six-dimensional real Fourier subspace.  Its sign is fixed by the
    largest loading in the encoder's fixed projection basis.
    """

    if projected_pose.ndim != 3:
        raise ValueError("projected_pose must have shape [batch, time, feature]")
    batch, time, _ = projected_pose.shape
    if valid_mask.shape != (batch, time):
        raise ValueError("valid_mask shape mismatch")
    if periods.shape != (batch,) or period_confidence.shape != (batch,):
        raise ValueError("period and confidence shapes must match batch")

    teachers = projected_pose.new_zeros((batch, time))
    available = torch.zeros(batch, dtype=torch.bool, device=projected_pose.device)
    frame_index = torch.arange(
        time,
        dtype=projected_pose.dtype,
        device=projected_pose.device,
    )
    for sample_index in range(batch):
        sample_valid = valid_mask[sample_index]
        valid_count = int(sample_valid.sum())
        if valid_count < 8 or float(period_confidence[sample_index]) <= 0.0:
            continue
        values = projected_pose[sample_index]
        count = sample_valid.sum().to(values.dtype)
        mean = (values * sample_valid.unsqueeze(-1)).sum(dim=0) / count
        centered = (values - mean) * sample_valid.unsqueeze(-1)
        spectrum = torch.fft.rfft(centered.float(), dim=0)
        target_bin = int(round(time / float(periods[sample_index])))
        target_bin = min(max(target_bin, 1), spectrum.shape[0] - 1)
        low = max(1, target_bin - 1)
        high = min(spectrum.shape[0], target_bin + 2)
        bins = torch.arange(low, high, device=projected_pose.device)
        if bins.numel() == 0:
            continue

        angles = (
            2.0
            * math.pi
            * frame_index[:, None].float()
            * bins[None, :].float()
            / time
        )
        basis = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)
        coefficients = torch.cat(
            (
                spectrum[low:high].real,
                -spectrum[low:high].imag,
            ),
            dim=0,
        )
        valid_basis = basis[sample_valid]
        q_basis, r_basis = torch.linalg.qr(valid_basis, mode="reduced")
        small_mapping = r_basis @ coefficients
        gram = small_mapping @ small_mapping.T
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
        if not bool(torch.isfinite(eigenvalues).all()) or float(eigenvalues[-1]) <= 1e-12:
            continue
        temporal_valid = q_basis @ eigenvectors[:, -1]
        loading = small_mapping.T @ eigenvectors[:, -1]
        orientation_index = int(loading.abs().argmax())
        if float(loading[orientation_index]) < 0.0:
            temporal_valid = -temporal_valid
        temporal_valid = temporal_valid - temporal_valid.mean()
        deviation = temporal_valid.std(unbiased=False)
        if not bool(torch.isfinite(deviation)) or float(deviation) <= 1e-6:
            continue
        teachers[sample_index, sample_valid] = (
            temporal_valid / deviation
        ).to(projected_pose.dtype)
        available[sample_index] = True
    return teachers, available


def phase_anchor_loss(
    stream: Tensor,
    teacher: Tensor,
    valid_mask: Tensor,
    teacher_available: Tensor,
    period_confidence: Tensor,
) -> PhaseAnchorOutput:
    """Confidence-weighted centered MSE against the signed pose phase."""

    if stream.ndim != 2 or teacher.shape != stream.shape:
        raise ValueError("stream and teacher must share shape [batch, time]")
    batch, _ = stream.shape
    if valid_mask.shape != stream.shape:
        raise ValueError("valid_mask shape mismatch")
    if teacher_available.shape != (batch,) or period_confidence.shape != (batch,):
        raise ValueError("availability and confidence must match batch")
    losses: list[Tensor] = []
    weights: list[Tensor] = []
    zero = stream.sum() * 0.0
    for values, target, valid, available, confidence in zip(
        stream,
        teacher,
        valid_mask,
        teacher_available,
        period_confidence,
        strict=True,
    ):
        if not bool(available) or float(confidence) <= 0.0 or int(valid.sum()) < 2:
            continue
        selected = values[valid]
        centered = selected - selected.mean()
        losses.append((centered - target[valid]).square().mean())
        weights.append(confidence.detach().to(stream.dtype))
    if not losses:
        return PhaseAnchorOutput(zero, 0, 0.0)
    stacked_weights = torch.stack(weights)
    loss = (
        torch.stack(losses) * stacked_weights
    ).sum() / stacked_weights.sum().clamp_min(1e-12)
    return PhaseAnchorOutput(
        loss,
        len(losses),
        float(stacked_weights.detach().sum()),
    )


def _constant_gradient_proof(device: torch.device) -> dict[str, float | int]:
    time = 256
    phase = torch.arange(time, device=device) * (2.0 * math.pi / 16.0)
    teacher = torch.sin(phase).unsqueeze(0)
    teacher = teacher / teacher.std(unbiased=False)
    stream = torch.zeros((1, time), device=device, requires_grad=True)
    output = phase_anchor_loss(
        stream,
        teacher,
        torch.ones((1, time), dtype=torch.bool, device=device),
        torch.ones(1, dtype=torch.bool, device=device),
        torch.ones(1, device=device),
    )
    gradient = torch.autograd.grad(output.loss, stream)[0]
    return {
        "loss": float(output.loss.detach()),
        "stream_gradient_l2": float(gradient.norm()),
        "nonzero_gradient_elements": int(torch.count_nonzero(gradient)),
    }


def _summary(values: Tensor) -> dict[str, float]:
    array = values.detach().double().cpu().numpy()
    return {
        "minimum": float(np.min(array)),
        "p10": float(np.percentile(array, 10.0)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.percentile(array, 90.0)),
        "maximum": float(np.max(array)),
    }


def _stream_statistics(
    stream: Tensor,
    valid_mask: Tensor,
    periods: Tensor,
    confidences: Tensor,
    teacher: Tensor,
    teacher_available: Tensor,
) -> dict[str, Any]:
    deviations: list[float] = []
    phase_correlations: list[float] = []
    correlation_streams: list[Tensor] = []
    for values, valid, target, available in zip(
        stream,
        valid_mask,
        teacher,
        teacher_available,
        strict=True,
    ):
        selected = values[valid]
        if selected.numel() < 2:
            continue
        deviations.append(float(selected.std(unbiased=False)))
        if bool(available):
            centered = selected - selected.mean()
            target_selected = target[valid]
            denominator = centered.norm() * target_selected.norm()
            if float(denominator) > 0.0:
                phase_correlations.append(
                    float((centered * target_selected).sum() / denominator)
                )
        if bool(valid.all()):
            centered = values - values.mean()
            norm = centered.norm()
            if float(norm) > 0.0:
                correlation_streams.append((centered / norm).cpu())

    estimated_periods, _ = estimate_period_batch(
        stream,
        minimum=4,
        maximum=128,
        valid_mask=valid_mask,
    )
    confident = confidences > 0
    relative_errors = (
        (estimated_periods[confident] - periods[confident]).abs()
        / periods[confident].clamp_min(1e-12)
    )
    weights = confidences[confident]
    weighted_relative_error = float(
        (relative_errors * weights).sum() / weights.sum().clamp_min(1e-12)
    )
    rounded_periods = estimated_periods.round().to(torch.int64)
    unique, counts = torch.unique(rounded_periods, return_counts=True)
    mode_index = int(counts.argmax())
    cross_video: dict[str, float | int] | None = None
    if len(correlation_streams) >= 2:
        matrix = torch.stack(correlation_streams)
        upper = (matrix @ matrix.T)[
            torch.triu(
                torch.ones(
                    (len(matrix), len(matrix)),
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
        ]
        cross_video = {
            "video_count": len(matrix),
            "mean": float(upper.mean()),
            "median": float(upper.median()),
        }
    return {
        "usable_stream_standard_deviation": _summary(
            torch.tensor(deviations)
        ),
        "phase_correlation": (
            _summary(torch.tensor(phase_correlations))
            if phase_correlations
            else None
        ),
        "fully_valid_centered_cross_video_correlation": cross_video,
        "head_period": {
            **_summary(estimated_periods),
            "mode_rounded_period": int(unique[mode_index]),
            "mode_fraction": float(counts[mode_index] / len(estimated_periods)),
            "minimum_period_fraction": float((rounded_periods == 4).float().mean()),
            "maximum_period_fraction": float((rounded_periods == 128).float().mean()),
        },
        "training_target_period_relative_error": {
            **_summary(relative_errors),
            "within_10_percent_fraction": float(
                (relative_errors <= 0.1).float().mean()
            ),
            "confidence_weighted_mean": weighted_relative_error,
        },
    }


def _counter(config: PAMSConfig) -> MultiExpertCounter:
    consensus = config.consensus
    return MultiExpertCounter(
        sigma_multipliers=consensus.sigma_multipliers,
        distance_multipliers=consensus.distance_multipliers,
        short_window_multiplier=consensus.short_window_multiplier,
        long_window_multiplier=consensus.long_window_multiplier,
        height_factor=consensus.height_factor,
        prominence_factor=consensus.prominence_factor,
        long_window_weight=consensus.long_window_weight,
        expert_mode=consensus.expert_mode,
    )


def _synthetic_evaluation(
    model: nn.Module,
    head: nn.Module,
    config: PAMSConfig,
    device: torch.device,
) -> dict[str, Any]:
    samples = list(generate_count_sweep())
    stress = synthetic_stress_suite(count=12.0)
    samples.extend(stress.values())
    counter = _counter(config)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for sample in samples:
            sequence = preprocess_pose_sequence(
                sample.sequence,
                target_frames=config.data.frames,
            )
            batch = collate_pose_sequences((sequence,)).to(device)
            _, projected = model.encoder.forward_with_pre_pe(
                batch.poses,
                batch.valid_mask,
            )
            stream = head(projected).masked_fill(~batch.valid_mask, 0.0)
            periods, confidence = estimate_period_batch(
                stream,
                minimum=config.period.minimum,
                maximum=config.period.maximum,
                valid_mask=batch.valid_mask,
            )
            result = counter.count(
                stream[0],
                float(periods[0]),
                batch.valid_mask[0],
                period_confidence=float(confidence[0]),
            )
            expected_period = config.data.frames / sample.target_count
            rows.append(
                {
                    "video_id": sample.sequence.video_id,
                    "target_count": sample.target_count,
                    "predicted_count": result.count,
                    "period_frames": float(periods[0]),
                    "expected_period_frames": expected_period,
                    "period_relative_error": abs(
                        float(periods[0]) - expected_period
                    )
                    / expected_period,
                }
            )
    sweep_rows = rows[:39]
    stress_rows = rows[39:]
    targets = np.asarray([row["target_count"] for row in sweep_rows])
    predictions = np.asarray([row["predicted_count"] for row in sweep_rows])
    return {
        "count_sweep_2_to_40": {
            "cases": len(sweep_rows),
            "period_within_10_percent_fraction": float(
                np.mean(
                    [
                        row["period_relative_error"] <= 0.1
                        for row in sweep_rows
                    ]
                )
            ),
            "count_exact_fraction": float(np.mean(predictions == targets)),
            "count_obo_fraction": float(np.mean(np.abs(predictions - targets) <= 1)),
            "count_nmae": float(np.mean(np.abs(predictions - targets) / targets)),
        },
        "stress_count12": {
            row["video_id"]: {
                "predicted_count": row["predicted_count"],
                "period_frames": row["period_frames"],
                "period_relative_error": row["period_relative_error"],
            }
            for row in stress_rows
        },
    }


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    config = load_config(args.config)
    if config.sshead.input_source != "projected_pose_pre_pe":
        raise ValueError("phase-anchor probe requires pre-PE SSHead input")
    manifest = load_pose_input_manifest(args.train_inputs, validate_exact=True)
    if (
        manifest.protocol != "ucfrep_526"
        or manifest.split != "train"
        or len(manifest.records) != 337
    ):
        raise ValueError("phase-anchor probe requires canonical UCFRep train337")
    sequences, snapshot = load_pose_cache_set(
        manifest.records,
        cache_dir=args.pose_cache_dir,
        pose_fingerprint=config.pose_fingerprint,
    )
    _seed_everything(config.seed)
    model = _load_encoder_checkpoint(args.encoder_checkpoint, config, device)
    initial_head_state = copy.deepcopy(model.period_head.state_dict())

    projected_batches: list[Tensor] = []
    valid_batches: list[Tensor] = []
    period_batches: list[Tensor] = []
    confidence_batches: list[Tensor] = []
    teacher_batches: list[Tensor] = []
    availability_batches: list[Tensor] = []
    with torch.inference_mode():
        for start in range(0, len(sequences), args.batch_size):
            batch = collate_pose_sequences(
                sequences[start : start + args.batch_size]
            ).to(device)
            embeddings, projected = model.encoder.forward_with_pre_pe(
                batch.poses,
                batch.valid_mask,
            )
            periods, confidences, _ = _estimate_post_warmup_periods(
                config=config,
                embeddings=embeddings,
                projected_pose=projected,
                valid_mask=batch.valid_mask,
                timeline_lengths=batch.lengths,
            )
            teacher, available = pose_band_phase_teacher(
                projected,
                batch.valid_mask,
                periods,
                confidences,
            )
            projected_batches.append(projected.cpu())
            valid_batches.append(batch.valid_mask.cpu())
            period_batches.append(periods.cpu())
            confidence_batches.append(confidences.cpu())
            teacher_batches.append(teacher.cpu())
            availability_batches.append(available.cpu())
    projected_all = torch.cat(projected_batches)
    valid_all = torch.cat(valid_batches)
    periods_all = torch.cat(period_batches)
    confidences_all = torch.cat(confidence_batches)
    teacher_all = torch.cat(teacher_batches)
    availability_all = torch.cat(availability_batches)

    heads: dict[str, nn.Module] = {}
    optimizers: dict[str, AdamW] = {}
    for candidate in _CANDIDATES:
        head = copy.deepcopy(model.period_head)
        head.load_state_dict(initial_head_state)
        head.to(device).train()
        heads[candidate.name] = head
        optimizers[candidate.name] = AdamW(
            head.parameters(),
            lr=config.sshead.learning_rate,
            weight_decay=config.sshead.weight_decay,
        )
    objective = SSHeadLoss(
        cycle_weight=config.sshead.cycle_weight,
        spectral_weight=config.sshead.spectral_weight,
        variance_weight=config.sshead.variance_weight,
        smoothness_weight=config.sshead.smoothness_weight,
        confidence_weighted_period_losses=True,
    )
    history: dict[str, list[dict[str, float]]] = {
        candidate.name: [] for candidate in _CANDIDATES
    }
    for epoch in range(args.epochs):
        generator = torch.Generator().manual_seed(config.seed + 10_000 + epoch)
        ordering = torch.randperm(len(sequences), generator=generator)
        epoch_sums = {
            candidate.name: {
                "total": 0.0,
                "base": 0.0,
                "phase": 0.0,
            }
            for candidate in _CANDIDATES
        }
        for start in range(0, len(sequences), args.batch_size):
            indices = ordering[start : start + args.batch_size]
            features = projected_all[indices].to(device)
            valid = valid_all[indices].to(device)
            periods = periods_all[indices].to(device)
            confidences = confidences_all[indices].to(device)
            teachers = teacher_all[indices].to(device)
            available = availability_all[indices].to(device)
            usable = valid.sum(dim=1) >= 2
            batch_fraction = len(indices) / len(sequences)
            for candidate in _CANDIDATES:
                head = heads[candidate.name]
                optimizer = optimizers[candidate.name]
                optimizer.zero_grad(set_to_none=True)
                stream = head(features).masked_fill(~valid, 0.0)
                base = objective.compute(
                    stream[usable],
                    periods[usable],
                    valid[usable],
                    period_confidence=confidences[usable],
                )
                phase = phase_anchor_loss(
                    stream,
                    teachers,
                    valid,
                    available,
                    confidences,
                )
                total = base.total + candidate.phase_weight * phase.loss
                if not bool(torch.isfinite(total)):
                    raise RuntimeError("candidate produced non-finite loss")
                total.backward()
                optimizer.step()
                epoch_sums[candidate.name]["total"] += (
                    float(total.detach()) * batch_fraction
                )
                epoch_sums[candidate.name]["base"] += (
                    float(base.total.detach()) * batch_fraction
                )
                epoch_sums[candidate.name]["phase"] += (
                    float(phase.loss.detach()) * batch_fraction
                )
        for candidate in _CANDIDATES:
            history[candidate.name].append(
                {
                    "epoch": float(epoch + 1),
                    **epoch_sums[candidate.name],
                }
            )

    candidate_results: dict[str, Any] = {}
    for candidate in _CANDIDATES:
        head = heads[candidate.name].eval()
        stream_parts: list[Tensor] = []
        with torch.inference_mode():
            for start in range(0, len(sequences), args.batch_size):
                features = projected_all[start : start + args.batch_size].to(device)
                valid = valid_all[start : start + args.batch_size].to(device)
                stream_parts.append(
                    head(features).masked_fill(~valid, 0.0).cpu()
                )
        stream_all = torch.cat(stream_parts).to(device)
        candidate_results[candidate.name] = {
            "phase_weight": candidate.phase_weight,
            "head_state_sha256": _state_sha256(head.cpu()),
            "epoch_1": history[candidate.name][0],
            "epoch_final": history[candidate.name][-1],
            "train337": _stream_statistics(
                stream_all,
                valid_all.to(device),
                periods_all.to(device),
                confidences_all.to(device),
                teacher_all.to(device),
                availability_all.to(device),
            ),
            "synthetic": _synthetic_evaluation(
                model,
                head.to(device),
                config,
                device,
            ),
        }

    confidence_positive = confidences_all[confidences_all > 0]
    payload = {
        "schema_version": 1,
        "classification": "inferred label-free train337/synthetic phase-anchor probe",
        "source_git_sha": args.source_git_sha,
        "config_fingerprint": config.fingerprint,
        "encoder_checkpoint_sha256": _sha256_file(args.encoder_checkpoint),
        "train_inputs_sha256": _sha256_file(args.train_inputs),
        "train_pose_cache_set_sha256": snapshot.fingerprint,
        "train_rows": len(sequences),
        "development_inputs_loaded": False,
        "development_targets_loaded": False,
        "test105_accessed": False,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "period_confidence": {
            "all": _summary(confidences_all),
            "positive": _summary(confidence_positive),
            "positive_fraction": float((confidences_all > 0).float().mean()),
            "normalized_weight_effective_sample_size": float(
                confidences_all.sum().square()
                / confidences_all.square().sum().clamp_min(1e-12)
            ),
        },
        "training_target_period": _summary(periods_all),
        "pose_phase_teacher": {
            "available_rows": int(availability_all.sum()),
            "available_fraction": float(availability_all.float().mean()),
        },
        "constant_stream_phase_anchor": _constant_gradient_proof(device),
        "candidates": candidate_results,
        "claim_boundary": {
            "paper_disclosed": False,
            "verified_reproduction": False,
            "development_selection": False,
            "sealed_test_result": False,
        },
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as destination:
            json.dump(payload, destination, sort_keys=True, separators=(",", ":"))
            destination.write("\n")
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, required=True)
    parser.add_argument("--train-inputs", type=Path, required=True)
    parser.add_argument("--pose-cache-dir", type=Path, required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.batch_size < 1 or arguments.epochs < 1:
        parser.error("--batch-size and --epochs must be positive")
    return arguments


if __name__ == "__main__":
    print(
        json.dumps(
            diagnose(_parse_args()),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
