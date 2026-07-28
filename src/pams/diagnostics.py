"""Label-free diagnostics for encoder temporal-position shortcuts.

Everything in this module is an independently inferred diagnostic.  It is not
part of the PAMS method disclosed by the authors and is never eligible for a
paper-table result.  The only accepted data are checkpoint-bound training
identifiers and their pose-cache files; dataset manifests, actions, and
repetition-count labels are deliberately outside the API.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from torch import Tensor

from pams.config import PAMSConfig, load_config
from pams.data import (
    PoseCacheEntryReceipt,
    PoseCacheSetSnapshot,
    load_pose_cache_with_receipt,
    pose_cache_path,
)
from pams.losses import periodic_correspondence_indices
from pams.model import PAMSModel
from pams.period import (
    estimate_period_from_embeddings,
    estimate_period_from_pose,
)
from pams.training import (
    CheckpointProvenance,
    collate_pose_sequences,
    load_model_checkpoint,
)
from pams.types import PoseSequence

_DIAGNOSTIC_ID = "pams-encoder-shortcut-inferred-v1"
_CHECKPOINT_SCHEMA_VERSION = 5


@dataclass(frozen=True, slots=True)
class _EncodedSample:
    """One in-memory, label-free encoder result."""

    video_id: str
    embeddings: Tensor
    valid_mask: Tensor
    period: float
    confidence: float


def _stable_file_sha256(path: Path) -> tuple[str, int]:
    """Hash a regular file while rejecting replacement or mutation."""

    before = path.lstat()
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"diagnostic input must be a regular non-symlink file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        opened = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
                byte_count += len(chunk)
            closed = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = path.lstat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(closed, field)
        or getattr(closed, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise RuntimeError(f"diagnostic input changed while it was read: {path}")
    if byte_count != after.st_size:
        raise RuntimeError(f"diagnostic input byte count changed while reading: {path}")
    return digest.hexdigest(), byte_count


def _device(value: str | torch.device | None) -> torch.device:
    if value is None or str(value).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(value)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return resolved


def _summary(values: Sequence[float] | np.ndarray) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "observations": 0,
            "minimum": None,
            "p10": None,
            "median": None,
            "mean": None,
            "p90": None,
            "maximum": None,
            "standard_deviation": None,
        }
    if not np.isfinite(array).all():
        raise RuntimeError("diagnostic statistic contains a non-finite value")
    return {
        "observations": int(array.size),
        "minimum": float(np.min(array)),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
        "standard_deviation": float(np.std(array)),
    }


def _identifier_commitment(identifiers: Sequence[str]) -> str:
    encoded = json.dumps(
        sorted(str(identifier) for identifier in identifiers),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ranked_identifiers(
    identifiers: Sequence[str],
    *,
    seed: int,
    purpose: str,
) -> tuple[str, ...]:
    """Return a deterministic hash ordering without exposing identifiers."""

    prefix = f"{_DIAGNOSTIC_ID}\0{seed}\0{purpose}\0".encode()
    return tuple(
        sorted(
            identifiers,
            key=lambda identifier: (
                hashlib.sha256(prefix + identifier.encode("utf-8")).digest(),
                identifier,
            ),
        )
    )


def _peek_checkpoint(
    checkpoint_path: Path,
    config: PAMSConfig,
) -> tuple[Literal["encoder", "sshead"], CheckpointProvenance]:
    """Validate checkpoint identity before the normal model loader is called."""

    payload = torch.load(
        checkpoint_path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    if payload.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    raw_stage = payload.get("stage")
    if raw_stage not in {"encoder", "sshead"}:
        raise ValueError(f"unknown checkpoint stage: {raw_stage!r}")
    if payload.get("config_fingerprint") != config.fingerprint:
        raise ValueError("checkpoint config fingerprint does not match")
    raw_provenance = payload.get("provenance")
    if not isinstance(raw_provenance, Mapping):
        raise ValueError("shortcut diagnostic requires bound checkpoint provenance")
    provenance = CheckpointProvenance.from_mapping(raw_provenance)
    if provenance.protocol != config.protocol:
        raise ValueError("checkpoint provenance protocol does not match config")
    if provenance.pose_fingerprint != config.pose_fingerprint:
        raise ValueError("checkpoint provenance pose fingerprint does not match config")
    return cast(Literal["encoder", "sshead"], raw_stage), provenance


def _load_training_poses(
    *,
    pose_cache_dir: Path,
    provenance: CheckpointProvenance,
    config: PAMSConfig,
    sample_size: int,
    seed: int,
) -> tuple[
    tuple[PoseSequence, ...],
    tuple[PoseCacheEntryReceipt, ...],
    tuple[PoseCacheEntryReceipt, ...],
    tuple[str, ...],
]:
    identifiers = provenance.training_video_ids
    if len(identifiers) < 2:
        raise ValueError("shortcut diagnostic requires at least two training videos")
    if sample_size < 0:
        raise ValueError("sample_size must be non-negative")
    selected_total = len(identifiers) if sample_size == 0 else min(sample_size, len(identifiers))
    if selected_total < 2:
        raise ValueError("shortcut diagnostic sample_size must select at least two videos")
    ranked = _ranked_identifiers(identifiers, seed=seed, purpose="sample")
    selected = ranked[:selected_total]

    selected_set = set(selected)
    selected_sequences: dict[str, PoseSequence] = {}
    selected_receipts: dict[str, PoseCacheEntryReceipt] = {}
    full_receipts: list[PoseCacheEntryReceipt] = []
    # The complete receipt set is always checked against checkpoint
    # provenance. sample_size limits encoder work, not input-integrity checks.
    for identifier in identifiers:
        sequence, _, receipt = load_pose_cache_with_receipt(
            pose_cache_path(pose_cache_dir, identifier),
            expected_pose_fingerprint=config.pose_fingerprint,
        )
        if sequence.video_id != identifier or receipt.video_id != identifier:
            raise ValueError("pose-cache video_id does not match checkpoint provenance")
        full_receipts.append(receipt)
        if identifier in selected_set:
            selected_sequences[identifier] = sequence
            selected_receipts[identifier] = receipt
    return (
        tuple(selected_sequences[identifier] for identifier in selected),
        tuple(selected_receipts[identifier] for identifier in selected),
        tuple(full_receipts),
        selected,
    )


def _projection_and_encoding_norms(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    projection_norms: list[float] = []
    effective_projection_norms: list[float] = []
    maximum_length = max(sequence.num_frames for sequence in sequences)
    position_rows = model.encoder.position_encoding.encoding[:maximum_length].detach().float()
    position_norms = torch.linalg.vector_norm(position_rows, dim=-1).cpu().numpy()
    scale_mode = model.encoder.input_projection_scale
    scale_factor = math.sqrt(model.encoder.model_dim) if scale_mode == "sqrt_model_dim" else 1.0

    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            flattened = batch.poses.flatten(start_dim=2)
            projected = model.encoder.input_projection(flattened)
            raw_norm = torch.linalg.vector_norm(projected, dim=-1)
            selected = raw_norm[batch.valid_mask].detach().float().cpu().numpy()
            projection_norms.extend(selected.tolist())
            effective_projection_norms.extend((selected * scale_factor).tolist())

    position_summary = _summary(position_norms)
    raw_summary = _summary(projection_norms)
    effective_summary = _summary(effective_projection_norms)
    position_mean = position_summary["mean"]
    effective_mean = effective_summary["mean"]
    return {
        "classification": "inferred diagnostic",
        "algorithm": (
            "L2 norms are measured per positional-encoding frame index and per valid "
            "checkpoint-bound pose frame before addition; the effective projection "
            "includes the configured input-projection scale."
        ),
        "input_projection_scale": scale_mode,
        "input_projection_scale_factor": scale_factor,
        "positional_encoding_frame_norm": position_summary,
        "pose_projection_frame_norm_unscaled": raw_summary,
        "pose_projection_frame_norm_effective": effective_summary,
        "effective_pose_projection_to_position_mean_ratio": (
            effective_mean / position_mean
            if isinstance(effective_mean, float)
            and isinstance(position_mean, float)
            and position_mean > 0
            else None
        ),
    }


def _encode_training_samples(
    model: PAMSModel,
    sequences: Sequence[PoseSequence],
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> tuple[_EncodedSample, ...]:
    encoded: list[_EncodedSample] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            batch = collate_pose_sequences(sequences[start : start + batch_size]).to(device)
            embeddings = model.encoder(batch.poses, batch.valid_mask)
            periods, confidences = estimate_period_from_embeddings(
                embeddings,
                minimum=config.period.minimum,
                maximum=config.period.maximum,
                valid_mask=batch.valid_mask,
            )
            for index, identifier in enumerate(batch.video_ids):
                length = int(batch.lengths[index])
                encoded.append(
                    _EncodedSample(
                        video_id=identifier,
                        embeddings=embeddings[index, :length].detach().float().cpu(),
                        valid_mask=batch.valid_mask[index, :length].detach().cpu(),
                        period=float(periods[index]),
                        confidence=float(confidences[index]),
                    )
                )
    return tuple(encoded)


def _period_probe(
    model: PAMSModel,
    poses: Tensor,
    *,
    config: PAMSConfig,
    valid_mask: Tensor,
) -> dict[str, float]:
    with torch.inference_mode():
        embeddings = model.encoder(poses, valid_mask)
        embedding_period, embedding_confidence = estimate_period_from_embeddings(
            embeddings,
            minimum=config.period.minimum,
            maximum=config.period.maximum,
            valid_mask=valid_mask,
        )
        pose_period, pose_confidence = estimate_period_from_pose(
            poses,
            minimum=config.period.minimum,
            maximum=config.period.maximum,
            valid_mask=valid_mask,
        )
    return {
        "embedding_period_frames": float(embedding_period[0]),
        "embedding_period_confidence": float(embedding_confidence[0]),
        "pose_signal_period_frames": float(pose_period[0]),
        "pose_signal_period_confidence": float(pose_confidence[0]),
    }


def _zero_random_period_probes(
    model: PAMSModel,
    *,
    config: PAMSConfig,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    frames = config.data.frames
    shape = (1, frames, config.data.keypoints, config.data.coordinates)
    valid = torch.ones((1, frames), dtype=torch.bool, device=device)
    zero = torch.zeros(shape, dtype=torch.float32, device=device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1_041_729)
    random_pose = torch.rand(shape, generator=generator, dtype=torch.float32).to(device)
    return {
        "classification": "inferred diagnostic",
        "algorithm": (
            "The encoder is evaluated on an all-zero pose and seeded IID U[0,1] pose "
            "noise with every frame valid; periods use the same bounded "
            "embedding-velocity FFT estimator as training."
        ),
        "frames": frames,
        "zero_pose": _period_probe(
            model,
            zero,
            config=config,
            valid_mask=valid,
        ),
        "seeded_uniform_random_pose": _period_probe(
            model,
            random_pose,
            config=config,
            valid_mask=valid,
        ),
    }


def _period_histogram(
    samples: Sequence[_EncodedSample],
    *,
    minimum: int,
    maximum: int,
) -> dict[str, Any]:
    frequencies = {period: 0 for period in range(minimum, maximum + 1)}
    for sample in samples:
        rounded = math.floor(sample.period + 0.5)
        frequencies[min(max(rounded, minimum), maximum)] += 1
    populated = {str(period): value for period, value in frequencies.items() if value}
    probabilities = np.asarray(
        [value / len(samples) for value in frequencies.values() if value],
        dtype=np.float64,
    )
    entropy = float(-np.sum(probabilities * np.log(probabilities))) if probabilities.size else 0.0
    maximum_support = min(len(samples), len(frequencies))
    normalized_entropy = (
        entropy / math.log(maximum_support) if maximum_support > 1 else 0.0
    )
    top_period, top_frequency = min(
        frequencies.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return {
        "classification": "inferred diagnostic",
        "algorithm": (
            "Fractional embedding periods are rounded by floor(period+0.5), clipped "
            "to the configured period range, and summarized as a categorical "
            "histogram. Entropy uses natural logarithms."
        ),
        "rounding": "floor(period_frames + 0.5)",
        "nonzero_bin_frequencies": populated,
        "entropy_nats": entropy,
        "entropy_normalized_by_log_min_sample_and_bin_total": normalized_entropy,
        "top_bin_period_frames": top_period,
        "top_bin_frequency": top_frequency,
        "top_bin_share": top_frequency / len(samples),
    }


def _frame_index_probe(
    samples: Sequence[_EncodedSample],
    *,
    seed: int,
) -> dict[str, Any]:
    usable = tuple(sample for sample in samples if int(sample.valid_mask.sum()) >= 2)
    if len(usable) < 2:
        return {
            "classification": "inferred diagnostic",
            "available": False,
            "reason": "fewer than two sampled videos contain at least two valid frames",
            "r2": None,
        }
    ranked_ids = _ranked_identifiers(
        [sample.video_id for sample in usable],
        seed=seed,
        purpose="frame-index-probe-split",
    )
    training_total = min(max(1, math.floor(0.8 * len(ranked_ids))), len(ranked_ids) - 1)
    training_ids = set(ranked_ids[:training_total])

    train_features: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    test_features: list[np.ndarray] = []
    test_targets: list[np.ndarray] = []
    maximum_frames_per_video = 64
    for sample in usable:
        mask = sample.valid_mask.numpy().astype(bool, copy=False)
        valid_indices = np.flatnonzero(mask)
        if valid_indices.size > maximum_frames_per_video:
            selection = np.rint(
                np.linspace(
                    0,
                    valid_indices.size - 1,
                    num=maximum_frames_per_video,
                )
            ).astype(np.int64)
            valid_indices = valid_indices[selection]
        features = sample.embeddings.numpy()[valid_indices]
        frame_indices = valid_indices.astype(np.float64, copy=False)
        targets = frame_indices / max(sample.embeddings.shape[0] - 1, 1)
        if sample.video_id in training_ids:
            train_features.append(features)
            train_targets.append(targets)
        else:
            test_features.append(features)
            test_targets.append(targets)

    train_x = np.concatenate(train_features, axis=0).astype(np.float64, copy=False)
    train_y = np.concatenate(train_targets, axis=0)
    test_x = np.concatenate(test_features, axis=0).astype(np.float64, copy=False)
    test_y = np.concatenate(test_targets, axis=0)
    probe = Ridge(alpha=1.0, fit_intercept=True, solver="lsqr", tol=1e-8)
    probe.fit(train_x, train_y)
    predictions = probe.predict(test_x)
    residual = float(np.square(test_y - predictions).sum())
    total = float(np.square(test_y - test_y.mean()).sum())
    r_squared = None if total <= 1e-15 else 1.0 - residual / total
    return {
        "classification": "inferred diagnostic",
        "available": r_squared is not None,
        "algorithm": (
            "A Ridge(alpha=1, solver='lsqr', tol=1e-8) linear probe predicts "
            "absolute frame_index/(sequence_length-1). Entire videos are assigned "
            "to an 80/20 split by a seeded SHA-256 ordering, and at most 64 valid "
            "frames per video are sampled uniformly to balance videos and bound memory."
        ),
        "split_unit": "video",
        "maximum_uniform_valid_frames_per_video": maximum_frames_per_video,
        "training_video_total": training_total,
        "test_video_total": len(ranked_ids) - training_total,
        "training_frame_total": int(train_x.shape[0]),
        "test_frame_total": int(test_x.shape[0]),
        "target": "absolute_frame_index_normalized_to_[0,1]",
        "r2": r_squared,
    }


def _positive_masks_and_boundaries(
    sample: _EncodedSample,
    *,
    scales: Sequence[float],
    device: torch.device,
) -> tuple[list[Tensor], list[tuple[int, int, int]]]:
    embeddings = sample.embeddings.unsqueeze(0).to(device)
    valid = sample.valid_mask.unsqueeze(0).to(device)
    time = embeddings.shape[1]
    period = torch.tensor([sample.period], dtype=embeddings.dtype, device=device)
    indices = torch.arange(time, device=device)
    offsets = indices.view(1, time) - indices.view(time, 1)
    rounded_period = max(1, int(round(sample.period)))
    within_candidate = valid.unsqueeze(1) & valid.unsqueeze(2)
    within_candidate &= ~torch.eye(time, dtype=torch.bool, device=device).unsqueeze(0)
    positive_masks: list[Tensor] = []
    boundary_rows: list[tuple[int, int, int]] = []

    for scale in scales:
        correspondences = periodic_correspondence_indices(
            embeddings,
            period,
            scale=scale,
            valid_mask=valid,
        )
        positives = torch.zeros(
            (1, time, time),
            dtype=torch.bool,
            device=device,
        )
        if time > 1:
            adjacent = torch.arange(time - 1, device=device)
            positives[:, adjacent, adjacent + 1] = True
            positives[:, adjacent + 1, adjacent] = True
        radius = max(1, int(round(rounded_period * scale / 2.0)))
        selected_total = 0
        effective_boundary_hits = 0
        nominal_boundary_hits = 0
        for direction_slot, direction in enumerate((-1, 1)):
            selected = correspondences[0, :, direction_slot]
            present = selected >= 0
            anchor_indices = torch.nonzero(present, as_tuple=False).flatten()
            positives[0, anchor_indices, selected[anchor_indices]] = True
            center = direction * rounded_period
            allowed = ((offsets - center).abs() <= radius) & valid[0].unsqueeze(0)
            candidate_grid = indices.unsqueeze(0).expand(time, time)
            effective_minimum = candidate_grid.masked_fill(~allowed, time).min(dim=1).values
            effective_maximum = candidate_grid.masked_fill(~allowed, -1).max(dim=1).values
            selected_total += int(present.sum())
            effective_boundary_hits += int(
                (
                    present
                    & (
                        (selected == effective_minimum)
                        | (selected == effective_maximum)
                    )
                ).sum()
            )
            selected_offsets = selected - indices
            nominal_boundary_hits += int(
                (
                    present
                    & ((selected_offsets - center).abs() == radius)
                ).sum()
            )
        positives &= within_candidate
        positive_masks.append(positives[0].detach().cpu())
        boundary_rows.append(
            (selected_total, effective_boundary_hits, nominal_boundary_hits)
        )
    return positive_masks, boundary_rows


def _correspondence_diagnostics(
    samples: Sequence[_EncodedSample],
    *,
    scales: Sequence[float],
    device: torch.device,
) -> dict[str, Any]:
    scale_values = tuple(float(scale) for scale in scales)
    selected_totals = [0 for _ in scale_values]
    effective_hits = [0 for _ in scale_values]
    nominal_hits = [0 for _ in scale_values]
    pair_totals = {
        (source, target): [0, 0]
        for source in range(len(scale_values))
        for target in range(len(scale_values))
        if source != target
    }

    with torch.inference_mode():
        for sample in samples:
            positives, boundaries = _positive_masks_and_boundaries(
                sample,
                scales=scale_values,
                device=device,
            )
            valid = sample.valid_mask
            time = valid.numel()
            candidate = valid.unsqueeze(0) & valid.unsqueeze(1)
            candidate &= ~torch.eye(time, dtype=torch.bool)
            for index, (selected, effective, nominal) in enumerate(boundaries):
                selected_totals[index] += selected
                effective_hits[index] += effective
                nominal_hits[index] += nominal
            for (source, target), totals in pair_totals.items():
                eligible = positives[source] & candidate
                conflict = eligible & ~positives[target]
                totals[0] += int(conflict.sum())
                totals[1] += int(eligible.sum())

    per_scale: dict[str, Any] = {}
    for index, scale in enumerate(scale_values):
        total = selected_totals[index]
        per_scale[format(scale, "g")] = {
            "selected_correspondence_total": total,
            "effective_window_boundary_hits": effective_hits[index],
            "effective_window_boundary_hit_rate": (
                effective_hits[index] / total if total else None
            ),
            "nominal_window_boundary_hits": nominal_hits[index],
            "nominal_window_boundary_hit_rate": nominal_hits[index] / total if total else None,
        }

    conflict_total = sum(values[0] for values in pair_totals.values())
    eligible_total = sum(values[1] for values in pair_totals.values())
    pairwise: dict[str, Any] = {}
    for (source, target), (conflicts, eligible) in pair_totals.items():
        key = f"{format(scale_values[source], 'g')}->{format(scale_values[target], 'g')}"
        pairwise[key] = {
            "conflicting_directed_positive_pairs": conflicts,
            "eligible_directed_positive_pairs": eligible,
            "rate": conflicts / eligible if eligible else None,
        }
    return {
        "classification": "inferred diagnostic",
        "algorithm": (
            "For each embedding-period correspondence, an effective boundary hit "
            "means the selected index is the first or last valid candidate in that "
            "scale's inferred symmetric window; a nominal hit means center±radius. "
            "For every ordered scale pair A->B, an A positive that is a valid "
            "within-video denominator candidate but is not a B positive is counted "
            "as a positive-as-negative conflict."
        ),
        "scales": list(scale_values),
        "boundary_by_scale": per_scale,
        "positive_as_negative_conflict": {
            "conflicting_directed_positive_pairs": conflict_total,
            "eligible_directed_positive_pairs": eligible_total,
            "rate": conflict_total / eligible_total if eligible_total else None,
            "ordered_scale_pairs": pairwise,
        },
    }


def _random_orthogonal_matrix(dimension: int, *, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 8_675_309)
    gaussian = torch.randn(
        (dimension, dimension),
        generator=generator,
        dtype=torch.float64,
    )
    orthogonal, triangular = torch.linalg.qr(gaussian)
    signs = torch.sign(torch.diagonal(triangular))
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return (orthogonal * signs.unsqueeze(0)).float()


def _orthogonal_basis_sensitivity(
    samples: Sequence[_EncodedSample],
    *,
    config: PAMSConfig,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    dimension = samples[0].embeddings.shape[-1]
    orthogonal = _random_orthogonal_matrix(dimension, seed=seed).to(device)
    transformed_periods: list[float] = []
    transformed_confidences: list[float] = []
    with torch.inference_mode():
        for sample in samples:
            embeddings = sample.embeddings.to(device) @ orthogonal
            valid = sample.valid_mask.to(device)
            period, confidence = estimate_period_from_embeddings(
                embeddings,
                minimum=config.period.minimum,
                maximum=config.period.maximum,
                valid_mask=valid,
            )
            transformed_periods.append(float(period[0]))
            transformed_confidences.append(float(confidence[0]))

    original_periods = np.asarray([sample.period for sample in samples], dtype=np.float64)
    original_confidences = np.asarray(
        [sample.confidence for sample in samples],
        dtype=np.float64,
    )
    transformed = np.asarray(transformed_periods, dtype=np.float64)
    transformed_confidence = np.asarray(transformed_confidences, dtype=np.float64)
    absolute_difference = np.abs(original_periods - transformed)
    rounded_original = np.floor(original_periods + 0.5)
    rounded_transformed = np.floor(transformed + 0.5)
    identity = torch.eye(dimension, dtype=orthogonal.dtype, device=device)
    orthogonality_error = float(
        (orthogonal.transpose(0, 1) @ orthogonal - identity).abs().max()
    )
    return {
        "classification": "inferred diagnostic",
        "algorithm": (
            "A seeded dense Gaussian matrix is reduced by CPU float64 QR with "
            "canonicalized diagonal signs. Right-multiplying every embedding by "
            "the resulting orthogonal matrix preserves all pairwise similarities; "
            "the embedding-velocity period estimator is then rerun."
        ),
        "pairwise_similarity_preserved_in_exact_arithmetic": True,
        "orthogonality_max_abs_error_float32": orthogonality_error,
        "period_before_transform_frames": _summary(original_periods),
        "period_after_transform_frames": _summary(transformed),
        "period_signed_difference_frames": _summary(
            transformed - original_periods
        ),
        "period_absolute_difference_frames": _summary(absolute_difference),
        "period_unchanged_within_1e-6_share": float(
            np.mean(absolute_difference <= 1e-6)
        ),
        "rounded_period_unchanged_share": float(
            np.mean(rounded_original == rounded_transformed)
        ),
        "confidence_change": _summary(transformed_confidence - original_confidences),
    }


def run_encoder_shortcut_diagnostic(
    checkpoint_path: str | Path,
    config: PAMSConfig,
    *,
    config_path: str | Path,
    pose_cache_dir: str | Path,
    sample_size: int = 64,
    seed: int = 2026,
    device: str | torch.device | None = None,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Run the read-only, label-free inferred encoder-shortcut diagnostic.

    ``sample_size=0`` selects every checkpoint-bound training video.  No
    argument can name a dataset manifest or label file.
    """

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    if isinstance(sample_size, bool) or not isinstance(sample_size, int):
        raise TypeError("sample_size must be an integer")
    source = Path(checkpoint_path)
    configuration_source = Path(config_path)
    checkpoint_sha256, checkpoint_bytes = _stable_file_sha256(source)
    config_sha256, config_bytes = _stable_file_sha256(configuration_source)
    if load_config(configuration_source).fingerprint != config.fingerprint:
        raise ValueError("config object does not match the supplied config file")
    stage, provenance = _peek_checkpoint(source, config)
    resolved_device = _device(device)
    with torch.random.fork_rng(devices=[]):
        model = load_model_checkpoint(
            source,
            config,
            device=resolved_device,
            expected_stage=stage,
            expected_provenance=provenance,
        )
    model.eval()
    sequences, sample_receipts, full_receipts, selected_ids = _load_training_poses(
        pose_cache_dir=Path(pose_cache_dir),
        provenance=provenance,
        config=config,
        sample_size=sample_size,
        seed=seed,
    )
    sample_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=sample_receipts,
    )
    full_snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=full_receipts,
    )
    full_set_selected = len(selected_ids) == len(provenance.training_video_ids)
    full_set_verified = full_snapshot.fingerprint == provenance.pose_cache_set_sha256
    if not full_set_verified:
        raise ValueError(
            "full training pose-cache set does not match checkpoint provenance"
        )

    frame_norms = _projection_and_encoding_norms(
        model,
        sequences,
        device=resolved_device,
        batch_size=batch_size,
    )
    synthetic_probes = _zero_random_period_probes(
        model,
        config=config,
        device=resolved_device,
        seed=seed,
    )
    encoded = _encode_training_samples(
        model,
        sequences,
        config=config,
        device=resolved_device,
        batch_size=batch_size,
    )
    periods = [sample.period for sample in encoded]
    confidences = [sample.confidence for sample in encoded]
    period_distribution = _period_histogram(
        encoded,
        minimum=config.period.minimum,
        maximum=config.period.maximum,
    )
    frame_probe = _frame_index_probe(encoded, seed=seed)
    correspondence = _correspondence_diagnostics(
        encoded,
        scales=config.loss.scales,
        device=resolved_device,
    )
    orthogonal = _orthogonal_basis_sensitivity(
        encoded,
        config=config,
        seed=seed,
        device=resolved_device,
    )

    checkpoint_sha256_after, checkpoint_bytes_after = _stable_file_sha256(source)
    config_sha256_after, config_bytes_after = _stable_file_sha256(configuration_source)
    if (checkpoint_sha256_after, checkpoint_bytes_after) != (
        checkpoint_sha256,
        checkpoint_bytes,
    ):
        raise RuntimeError("checkpoint changed during shortcut diagnostic")
    if (config_sha256_after, config_bytes_after) != (config_sha256, config_bytes):
        raise RuntimeError("configuration changed during shortcut diagnostic")

    return {
        "schema_version": 1,
        "diagnostic_id": _DIAGNOSTIC_ID,
        "classification": "inferred diagnostic",
        "disclosed_by_pams_authors": False,
        "eligible_for_paper_table": False,
        "label_firewall": {
            "input_surface": ["checkpoint", "config", "pose_cache_directory"],
            "dataset_manifest_argument_supported": False,
            "dataset_manifest_loaded": False,
            "label_fields_accessed": [],
            "checkpoint_training_ids_are_only_used_for_cache_lookup_and_hash_splits": True,
        },
        "inputs": {
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_bytes": checkpoint_bytes,
            "checkpoint_stage": stage,
            "config_sha256": config_sha256,
            "config_bytes": config_bytes,
            "config_fingerprint": config.fingerprint,
            "pose_fingerprint": config.pose_fingerprint,
            "checkpoint_training_video_total": len(provenance.training_video_ids),
            "sampled_training_video_total": len(selected_ids),
            "sampled_training_video_ids_sha256": _identifier_commitment(selected_ids),
            "sample_selection": (
                "lowest seeded SHA-256 ranks; sample_size=0 means the full bound set; "
                "all bound cache bytes are still hashed for provenance verification"
            ),
            "sample_pose_cache_set_sha256": sample_snapshot.fingerprint,
            "full_pose_cache_set_sha256": full_snapshot.fingerprint,
            "all_checkpoint_pose_caches_hashed": True,
            "full_checkpoint_pose_cache_set_selected": full_set_selected,
            "full_checkpoint_pose_cache_set_verified": full_set_verified,
            "diagnostic_seed": seed,
            "device_type": resolved_device.type,
        },
        "frame_norms": frame_norms,
        "zero_and_random_pose_period_probes": synthetic_probes,
        "training_embedding_periods": {
            "classification": "inferred diagnostic",
            "sample_period_frames": _summary(periods),
            "period_confidence": _summary(confidences),
            "zero_confidence_share": float(
                np.mean(np.asarray(confidences, dtype=np.float64) == 0.0)
            ),
            "histogram": period_distribution,
        },
        "frame_index_linear_probe": frame_probe,
        "multiscale_correspondence": correspondence,
        "orthogonal_basis_period_sensitivity": orthogonal,
        "read_only_verification": {
            "checkpoint_sha256_unchanged": True,
            "config_sha256_unchanged": True,
            "model_or_training_state_updated": False,
            "pose_cache_write_operations": 0,
        },
    }
