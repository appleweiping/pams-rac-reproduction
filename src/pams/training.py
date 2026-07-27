"""Label-free encoder and self-supervised period-head training.

This module deliberately accepts only :class:`~pams.types.PoseSequence`
objects.  Counts, actions, and split metadata do not appear in either training
API, which makes the sealed-evaluation boundary straightforward to audit.
"""

from __future__ import annotations

import os
import random
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
import torch
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from pams.config import PAMSConfig
from pams.consensus import MultiExpertCounter
from pams.losses import PAMSTCCLoss, SSHeadLoss
from pams.model import PAMSEncoder, PAMSModel, PeriodHead
from pams.period import (
    estimate_period_batch,
    estimate_period_from_embeddings,
    estimate_period_from_pose,
)
from pams.reproducibility import seed_everything
from pams.types import CountResult, PoseSequence

_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PoseBatch:
    """A padded, label-free minibatch."""

    video_ids: tuple[str, ...]
    poses: Tensor
    valid_mask: Tensor
    fps: Tensor
    lengths: Tensor

    def __post_init__(self) -> None:
        if self.poses.ndim != 4 or self.poses.shape[2:] != (33, 3):
            raise ValueError("poses must have shape [batch, time, 33, 3]")
        batch, time = self.poses.shape[:2]
        if len(self.video_ids) != batch or len(set(self.video_ids)) != batch:
            raise ValueError("video_ids must be unique and match the batch")
        if self.valid_mask.shape != (batch, time):
            raise ValueError("valid_mask must have shape [batch, time]")
        if self.fps.shape != (batch,) or self.lengths.shape != (batch,):
            raise ValueError("fps and lengths must have shape [batch]")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean")

    @property
    def batch_size(self) -> int:
        return len(self.video_ids)

    def to(self, device: str | torch.device) -> PoseBatch:
        """Copy tensor members to ``device`` while preserving identifiers."""

        return PoseBatch(
            video_ids=self.video_ids,
            poses=self.poses.to(device=device),
            valid_mask=self.valid_mask.to(device=device),
            fps=self.fps.to(device=device),
            lengths=self.lengths.to(device=device),
        )


def collate_pose_sequences(samples: Sequence[PoseSequence]) -> PoseBatch:
    """Pad variable-length pose sequences with invalid, exact-zero frames."""

    items = tuple(samples)
    if not items:
        raise ValueError("cannot collate an empty batch")
    if not all(isinstance(item, PoseSequence) for item in items):
        raise TypeError("all batch items must be PoseSequence instances")
    video_ids = tuple(item.video_id for item in items)
    if len(set(video_ids)) != len(video_ids):
        raise ValueError("a batch cannot contain duplicate video_id values")

    maximum_length = max(item.num_frames for item in items)
    poses = torch.zeros(
        (len(items), maximum_length, 33, 3),
        dtype=torch.float32,
    )
    valid_mask = torch.zeros(
        (len(items), maximum_length),
        dtype=torch.bool,
    )
    for index, item in enumerate(items):
        length = item.num_frames
        poses[index, :length] = torch.from_numpy(np.asarray(item.xyz).copy())
        valid_mask[index, :length] = torch.from_numpy(np.asarray(item.valid_mask).copy())
    return PoseBatch(
        video_ids=video_ids,
        poses=poses,
        valid_mask=valid_mask,
        fps=torch.tensor([item.fps for item in items], dtype=torch.float32),
        lengths=torch.tensor(
            [item.num_frames for item in items],
            dtype=torch.long,
        ),
    )


def build_pams_model(config: PAMSConfig) -> PAMSModel:
    """Build either the disclosed architecture or a config-defined smoke model."""

    model = config.model
    encoder = PAMSEncoder(
        input_dim=model.input_dim,
        model_dim=model.model_dim,
        embedding_dim=model.embedding_dim,
        num_layers=model.layers,
        num_heads=model.heads,
        feedforward_dim=model.feedforward_dim,
        dropout=model.dropout,
        max_length=max(4096, config.data.frames),
        norm_first=model.norm_first,
    )
    head = PeriodHead(
        embedding_dim=model.embedding_dim,
        hidden_dim=model.period_head_hidden_dim,
    )
    return PAMSModel(encoder=encoder, period_head=head)


@dataclass(frozen=True, slots=True)
class EncoderEpochStats:
    """One completed encoder epoch."""

    epoch: int
    loss: float
    learning_rate: float
    period_source: Literal["pose", "embedding"]
    optimizer_steps: int
    clusters_refreshed: bool


@dataclass(frozen=True, slots=True)
class SSHeadEpochStats:
    """One completed independent SSHead epoch."""

    epoch: int
    total: float
    cycle: float
    spectral: float
    variance: float
    smoothness: float
    learning_rate: float
    optimizer_steps: int


@dataclass(frozen=True, slots=True)
class EncoderTrainingResult:
    """Trained model, audit history, and latest clustering state."""

    model: PAMSModel
    history: tuple[EncoderEpochStats, ...]
    cluster_assignments: Mapping[str, int]
    completed_epochs: int
    checkpoint_path: Path | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "cluster_assignments",
            MappingProxyType(dict(self.cluster_assignments)),
        )


@dataclass(frozen=True, slots=True)
class SSHeadTrainingResult:
    """Frozen-encoder SSHead training result."""

    model: PAMSModel
    history: tuple[SSHeadEpochStats, ...]
    completed_epochs: int
    checkpoint_path: Path | None


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def _materialize_sequences(
    sequences: Sequence[PoseSequence],
) -> tuple[PoseSequence, ...]:
    items = tuple(sequences)
    if not items:
        raise ValueError("training requires at least one PoseSequence")
    if not all(isinstance(item, PoseSequence) for item in items):
        raise TypeError("training accepts only PoseSequence instances")
    identifiers = [item.video_id for item in items]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("training video_id values must be unique")
    return items


def _default_microbatch_size(dataset_size: int, effective_batch_size: int) -> int:
    upper = min(dataset_size, effective_batch_size)
    for candidate in range(upper, 0, -1):
        if effective_batch_size % candidate == 0:
            return candidate
    return 1


def _validate_microbatch_size(
    dataset_size: int,
    effective_batch_size: int,
    microbatch_size: int | None,
) -> int:
    selected = (
        _default_microbatch_size(dataset_size, effective_batch_size)
        if microbatch_size is None
        else int(microbatch_size)
    )
    if selected < 1:
        raise ValueError("microbatch_size must be positive")
    if selected > effective_batch_size:
        raise ValueError("microbatch_size cannot exceed effective_batch_size")
    if effective_batch_size % selected:
        raise ValueError("effective_batch_size must be divisible by microbatch_size")
    return selected


def _data_loader(
    sequences: Sequence[PoseSequence],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[PoseSequence]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        cast(Dataset[PoseSequence], sequences),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=collate_pose_sequences,
        generator=generator,
        drop_last=False,
    )


def _pooled_embeddings(
    encoder: PAMSEncoder,
    sequences: Sequence[PoseSequence],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[tuple[str, ...], np.ndarray]:
    previous_training = encoder.training
    encoder.eval()
    identifiers: list[str] = []
    pooled: list[np.ndarray] = []
    with torch.inference_mode():
        loader = _data_loader(
            sequences,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
        )
        for raw_batch in loader:
            if not isinstance(raw_batch, PoseBatch):
                raise TypeError("pose collator returned an unexpected batch type")
            batch = raw_batch.to(device)
            embeddings = encoder(batch.poses, batch.valid_mask)
            weights = batch.valid_mask.to(embeddings.dtype).unsqueeze(-1)
            counts = weights.sum(dim=1).clamp_min(1.0)
            video_embedding = (embeddings * weights).sum(dim=1) / counts
            pooled.extend(video_embedding.detach().cpu().numpy())
            identifiers.extend(batch.video_ids)
    encoder.train(previous_training)
    return tuple(identifiers), np.asarray(pooled, dtype=np.float32)


def refresh_video_clusters(
    encoder: PAMSEncoder,
    sequences: Sequence[PoseSequence],
    config: PAMSConfig,
    *,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    epoch: int = 0,
) -> dict[str, int]:
    """Pool each video and deterministically refit the video-level KMeans."""

    items = _materialize_sequences(sequences)
    resolved_device = _resolve_device(device)
    encoder.to(resolved_device)
    selected_batch_size = batch_size or min(
        config.training.effective_batch_size,
        len(items),
    )
    if selected_batch_size < 1:
        raise ValueError("batch_size must be positive")
    identifiers, features = _pooled_embeddings(
        encoder,
        items,
        batch_size=selected_batch_size,
        device=resolved_device,
    )
    cluster_count = min(config.loss.kmeans_clusters, len(items))
    if cluster_count == 1:
        assignments = np.zeros(len(items), dtype=np.int64)
    else:
        estimator = KMeans(
            n_clusters=cluster_count,
            random_state=config.seed + epoch,
            n_init=10,
            algorithm="lloyd",
        )
        assignments = estimator.fit_predict(features)
    return {
        identifier: int(assignment)
        for identifier, assignment in zip(identifiers, assignments, strict=True)
    }


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _checkpoint_payload(
    *,
    stage: Literal["encoder", "sshead"],
    config: PAMSConfig,
    model: PAMSModel,
    optimizer: Optimizer,
    completed_epochs: int,
    history: Sequence[EncoderEpochStats] | Sequence[SSHeadEpochStats],
    scheduler: ReduceLROnPlateau | None = None,
    cluster_assignments: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "stage": stage,
        "config_fingerprint": config.fingerprint,
        "completed_epochs": completed_epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "history": [asdict(item) for item in history],
        "cluster_assignments": dict(cluster_assignments or {}),
        "rng_state": _capture_rng_state(),
    }


def _load_checkpoint(
    path: Path,
    *,
    stage: Literal["encoder", "sshead"],
    config: PAMSConfig,
    model: PAMSModel,
    optimizer: Optimizer,
    device: torch.device,
    scheduler: ReduceLROnPlateau | None = None,
) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    if payload.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    if payload.get("stage") != stage:
        raise ValueError(f"expected a {stage} checkpoint, received {payload.get('stage')!r}")
    if payload.get("config_fingerprint") != config.fingerprint:
        raise ValueError("checkpoint config fingerprint does not match")
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler_state = payload.get("scheduler_state")
    if scheduler is not None:
        if scheduler_state is None:
            raise ValueError("encoder checkpoint is missing scheduler state")
        scheduler.load_state_dict(scheduler_state)
    _restore_rng_state(payload["rng_state"])
    return payload


def load_model_checkpoint(
    path: str | Path,
    config: PAMSConfig,
    *,
    device: str | torch.device | None = None,
    expected_stage: Literal["encoder", "sshead"] | None = None,
) -> PAMSModel:
    """Load model weights after validating checkpoint provenance.

    Unlike training resume, this inference-oriented loader intentionally does
    not restore optimizer, scheduler, or process RNG state.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    resolved_device = _resolve_device(device)
    payload = torch.load(source, map_location=resolved_device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    if payload.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    stage = payload.get("stage")
    if stage not in {"encoder", "sshead"}:
        raise ValueError(f"unknown checkpoint stage: {stage!r}")
    if expected_stage is not None and stage != expected_stage:
        raise ValueError(f"expected a {expected_stage} checkpoint, received {stage!r}")
    if payload.get("config_fingerprint") != config.fingerprint:
        raise ValueError("checkpoint config fingerprint does not match")
    model = build_pams_model(config).to(resolved_device)
    model.load_state_dict(payload["model_state"])
    return model


def _target_end_epoch(total_epochs: int, stop_after_epoch: int | None) -> int:
    if stop_after_epoch is None:
        return total_epochs
    if stop_after_epoch < 0:
        raise ValueError("stop_after_epoch must be non-negative")
    return min(total_epochs, stop_after_epoch)


def train_encoder(
    sequences: Sequence[PoseSequence],
    config: PAMSConfig,
    *,
    model: PAMSModel | None = None,
    device: str | torch.device | None = None,
    microbatch_size: int | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    stop_after_epoch: int | None = None,
) -> EncoderTrainingResult:
    """Train the encoder with PAMS-TCC without accepting any ground truth."""

    items = _materialize_sequences(sequences)
    seed_everything(config.seed)
    resolved_device = _resolve_device(device)
    trained_model = build_pams_model(config) if model is None else model
    trained_model.to(resolved_device)
    selected_microbatch = _validate_microbatch_size(
        len(items),
        config.training.effective_batch_size,
        microbatch_size,
    )
    accumulation_steps = config.training.effective_batch_size // selected_microbatch

    for parameter in trained_model.period_head.parameters():
        parameter.requires_grad_(False)
    for parameter in trained_model.encoder.parameters():
        parameter.requires_grad_(True)
    optimizer = AdamW(
        trained_model.encoder.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.training.scheduler_factor,
        patience=config.training.scheduler_patience,
        min_lr=config.training.minimum_learning_rate,
    )
    objective = PAMSTCCLoss(
        scales=config.loss.scales,
        temperature=config.loss.temperature,
        correspondence_tolerance=config.loss.correspondence_tolerance,
    )
    destination = None if checkpoint_path is None else Path(checkpoint_path)
    history: list[EncoderEpochStats] = []
    cluster_assignments: dict[str, int] = {}
    completed_epochs = 0

    if resume:
        if destination is None:
            raise ValueError("resume=True requires checkpoint_path")
        payload = _load_checkpoint(
            destination,
            stage="encoder",
            config=config,
            model=trained_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=resolved_device,
        )
        completed_epochs = int(payload["completed_epochs"])
        history = [EncoderEpochStats(**row) for row in payload["history"]]
        cluster_assignments = {
            str(identifier): int(cluster)
            for identifier, cluster in payload["cluster_assignments"].items()
        }

    end_epoch = _target_end_epoch(
        config.training.epochs,
        stop_after_epoch,
    )
    if completed_epochs > end_epoch:
        raise ValueError("checkpoint has already passed the requested stop_after_epoch")

    for epoch_index in range(completed_epochs, end_epoch):
        refresh = not cluster_assignments or epoch_index % config.loss.kmeans_refresh_epochs == 0
        if refresh:
            cluster_assignments = refresh_video_clusters(
                trained_model.encoder,
                items,
                config,
                device=resolved_device,
                batch_size=selected_microbatch,
                epoch=epoch_index,
            )

        loader = _data_loader(
            items,
            batch_size=selected_microbatch,
            shuffle=True,
            seed=config.seed + epoch_index,
        )
        trained_model.encoder.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        optimizer_steps = 0
        batch_count = len(loader)
        for batch_index, raw_batch in enumerate(loader):
            if not isinstance(raw_batch, PoseBatch):
                raise TypeError("pose collator returned an unexpected batch type")
            batch = raw_batch.to(resolved_device)
            embeddings = trained_model.encoder(batch.poses, batch.valid_mask)
            if epoch_index < config.period.pose_energy_epochs:
                periods, _ = estimate_period_from_pose(
                    batch.poses,
                    minimum=config.period.minimum,
                    maximum=config.period.maximum,
                    valid_mask=batch.valid_mask,
                )
                period_source: Literal["pose", "embedding"] = "pose"
            else:
                periods, _ = estimate_period_from_embeddings(
                    embeddings.detach(),
                    minimum=config.period.minimum,
                    maximum=config.period.maximum,
                    valid_mask=batch.valid_mask,
                )
                period_source = "embedding"
            batch_clusters = torch.tensor(
                [cluster_assignments[identifier] for identifier in batch.video_ids],
                dtype=torch.long,
                device=resolved_device,
            )
            loss = objective(
                embeddings,
                periods,
                batch.valid_mask,
                cluster_labels=batch_clusters,
            )

            group_start = (batch_index // accumulation_steps) * accumulation_steps
            group_stop = min(group_start + accumulation_steps, batch_count)
            samples_before_group = group_start * selected_microbatch
            group_sample_count = min(
                config.training.effective_batch_size,
                len(items) - samples_before_group,
            )
            scaled_loss = loss * (batch.batch_size / group_sample_count)
            scaled_loss.backward()
            epoch_loss += float(loss.detach()) * batch.batch_size

            end_of_group = (
                batch_index + 1
            ) % accumulation_steps == 0 or batch_index + 1 == batch_count
            if end_of_group:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            if group_stop <= group_start:
                raise RuntimeError("invalid gradient-accumulation group")

        mean_loss = epoch_loss / len(items)
        scheduler.step(mean_loss)
        completed_epochs = epoch_index + 1
        history.append(
            EncoderEpochStats(
                epoch=completed_epochs,
                loss=mean_loss,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                period_source=period_source,
                optimizer_steps=optimizer_steps,
                clusters_refreshed=refresh,
            )
        )
        if destination is not None:
            _atomic_torch_save(
                _checkpoint_payload(
                    stage="encoder",
                    config=config,
                    model=trained_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    completed_epochs=completed_epochs,
                    history=history,
                    cluster_assignments=cluster_assignments,
                ),
                destination,
            )

    return EncoderTrainingResult(
        model=trained_model,
        history=tuple(history),
        cluster_assignments=cluster_assignments,
        completed_epochs=completed_epochs,
        checkpoint_path=destination,
    )


def train_sshead(
    sequences: Sequence[PoseSequence],
    config: PAMSConfig,
    *,
    model: PAMSModel,
    device: str | torch.device | None = None,
    microbatch_size: int | None = None,
    checkpoint_path: str | Path | None = None,
    resume: bool = False,
    stop_after_epoch: int | None = None,
) -> SSHeadTrainingResult:
    """Train only the inferred SSHead objective with the encoder frozen."""

    items = _materialize_sequences(sequences)
    seed_everything(config.seed)
    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    selected_microbatch = _validate_microbatch_size(
        len(items),
        config.training.effective_batch_size,
        microbatch_size,
    )
    accumulation_steps = config.training.effective_batch_size // selected_microbatch
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.period_head.parameters():
        parameter.requires_grad_(True)
    model.encoder.eval()
    optimizer = AdamW(
        model.period_head.parameters(),
        lr=config.sshead.learning_rate,
        weight_decay=config.sshead.weight_decay,
    )
    objective = SSHeadLoss(
        cycle_weight=config.sshead.cycle_weight,
        spectral_weight=config.sshead.spectral_weight,
        variance_weight=config.sshead.variance_weight,
        smoothness_weight=config.sshead.smoothness_weight,
    )
    destination = None if checkpoint_path is None else Path(checkpoint_path)
    history: list[SSHeadEpochStats] = []
    completed_epochs = 0

    if resume:
        if destination is None:
            raise ValueError("resume=True requires checkpoint_path")
        payload = _load_checkpoint(
            destination,
            stage="sshead",
            config=config,
            model=model,
            optimizer=optimizer,
            device=resolved_device,
        )
        completed_epochs = int(payload["completed_epochs"])
        history = [SSHeadEpochStats(**row) for row in payload["history"]]

    end_epoch = _target_end_epoch(config.sshead.epochs, stop_after_epoch)
    if completed_epochs > end_epoch:
        raise ValueError("checkpoint has already passed the requested stop_after_epoch")

    for epoch_index in range(completed_epochs, end_epoch):
        loader = _data_loader(
            items,
            batch_size=selected_microbatch,
            shuffle=True,
            seed=config.seed + 10_000 + epoch_index,
        )
        model.encoder.eval()
        model.period_head.train()
        optimizer.zero_grad(set_to_none=True)
        sums = {
            "total": 0.0,
            "cycle": 0.0,
            "spectral": 0.0,
            "variance": 0.0,
            "smoothness": 0.0,
        }
        optimizer_steps = 0
        batch_count = len(loader)
        for batch_index, raw_batch in enumerate(loader):
            if not isinstance(raw_batch, PoseBatch):
                raise TypeError("pose collator returned an unexpected batch type")
            batch = raw_batch.to(resolved_device)
            with torch.no_grad():
                embeddings = model.encoder(batch.poses, batch.valid_mask)
                periods, _ = estimate_period_from_embeddings(
                    embeddings,
                    minimum=config.period.minimum,
                    maximum=config.period.maximum,
                    valid_mask=batch.valid_mask,
                )
            stream = model.period_head(embeddings.detach())
            stream = stream.masked_fill(~batch.valid_mask, 0.0)
            details = objective.compute(stream, periods, batch.valid_mask)

            group_start = (batch_index // accumulation_steps) * accumulation_steps
            samples_before_group = group_start * selected_microbatch
            group_sample_count = min(
                config.training.effective_batch_size,
                len(items) - samples_before_group,
            )
            scaled_loss = details.total * (batch.batch_size / group_sample_count)
            scaled_loss.backward()
            for name in sums:
                sums[name] += float(getattr(details, name).detach()) * batch.batch_size

            end_of_group = (
                batch_index + 1
            ) % accumulation_steps == 0 or batch_index + 1 == batch_count
            if end_of_group:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1

        completed_epochs = epoch_index + 1
        history.append(
            SSHeadEpochStats(
                epoch=completed_epochs,
                total=sums["total"] / len(items),
                cycle=sums["cycle"] / len(items),
                spectral=sums["spectral"] / len(items),
                variance=sums["variance"] / len(items),
                smoothness=sums["smoothness"] / len(items),
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                optimizer_steps=optimizer_steps,
            )
        )
        if destination is not None:
            _atomic_torch_save(
                _checkpoint_payload(
                    stage="sshead",
                    config=config,
                    model=model,
                    optimizer=optimizer,
                    completed_epochs=completed_epochs,
                    history=history,
                ),
                destination,
            )

    return SSHeadTrainingResult(
        model=model,
        history=tuple(history),
        completed_epochs=completed_epochs,
        checkpoint_path=destination,
    )


def predict_sequence(
    model: PAMSModel,
    sequence: PoseSequence,
    config: PAMSConfig,
    *,
    device: str | torch.device | None = None,
    counter: MultiExpertCounter | None = None,
) -> CountResult:
    """Predict one count from the head stream, its FFT period, and consensus."""

    if not isinstance(sequence, PoseSequence):
        raise TypeError("sequence must be a PoseSequence")
    resolved_device = _resolve_device(device)
    model.to(resolved_device)
    previous_training = model.training
    model.eval()
    batch = collate_pose_sequences((sequence,)).to(resolved_device)
    with torch.inference_mode():
        _, stream_batch = model(batch.poses, batch.valid_mask)
        periods, _ = estimate_period_batch(
            stream_batch,
            minimum=config.period.minimum,
            maximum=config.period.maximum,
            valid_mask=batch.valid_mask,
        )
    stream = stream_batch[0, : sequence.num_frames]
    mask = batch.valid_mask[0, : sequence.num_frames]
    if counter is None:
        consensus = config.consensus
        counter = MultiExpertCounter(
            sigma_multipliers=consensus.sigma_multipliers,
            distance_multipliers=consensus.distance_multipliers,
            short_window_multiplier=consensus.short_window_multiplier,
            long_window_multiplier=consensus.long_window_multiplier,
            height_factor=consensus.height_factor,
            prominence_factor=consensus.prominence_factor,
            long_window_weight=consensus.long_window_weight,
        )
    result = counter.count(
        stream,
        period_frames=float(periods[0]),
        valid_mask=mask,
    ).to_count_result()
    model.train(previous_training)
    return result
