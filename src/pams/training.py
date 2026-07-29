"""Label-free encoder and self-supervised period-head training.

This module deliberately accepts only :class:`~pams.types.PoseSequence`
objects.  Counts, actions, and split metadata do not appear in either training
API, which makes the sealed-evaluation boundary straightforward to audit.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
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
    estimate_period_from_projected_pose,
)
from pams.reproducibility import durable_mkdir, fsync_directory, seed_everything
from pams.types import CountResult, PoseSequence

_CHECKPOINT_SCHEMA_VERSION = 5
_PROGRESS_SCHEMA_VERSION = 2
_SHA256_HEX_LENGTH = 64
_GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_IMAGE_ID_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")

PeriodHistorySource = Literal[
    "pose",
    "embedding",
    "projected_pose_velocity_vector_acf",
    "fixed_period_inferred",
]


def _canonical_sha256(value: str, name: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")
    return digest


@dataclass(frozen=True, slots=True)
class CheckpointProvenance:
    """Caller-owned identity of the data and upstream artifact used for training."""

    protocol: str
    dataset_fingerprint: str
    training_video_ids: tuple[str, ...]
    pose_fingerprint: str
    pose_cache_set_sha256: str
    source_git_sha: str
    container_image_id: str | None = None
    container_environment_sha256: str | None = None
    upstream_encoder_checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        protocol = str(self.protocol).strip()
        if not protocol:
            raise ValueError("protocol must be non-empty")
        if isinstance(self.training_video_ids, str | bytes):
            raise TypeError("training_video_ids must be a sequence of identifiers")
        identifiers = tuple(str(identifier).strip() for identifier in self.training_video_ids)
        if not identifiers or any(not identifier for identifier in identifiers):
            raise ValueError("training_video_ids must contain non-empty identifiers")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("training_video_ids must be unique")

        upstream = self.upstream_encoder_checkpoint_sha256
        source_revision = str(self.source_git_sha).strip()
        if not _GIT_SHA_PATTERN.fullmatch(source_revision):
            raise ValueError("source_git_sha must be a 40-character lowercase Git SHA")
        image_id = self.container_image_id
        environment_sha256 = self.container_environment_sha256
        if (image_id is None) != (environment_sha256 is None):
            raise ValueError(
                "container_image_id and container_environment_sha256 must be supplied together"
            )
        if image_id is not None and not _IMAGE_ID_PATTERN.fullmatch(str(image_id)):
            raise ValueError("container_image_id must be an immutable sha256:<digest> ID")
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(
            self,
            "dataset_fingerprint",
            _canonical_sha256(self.dataset_fingerprint, "dataset_fingerprint"),
        )
        # Sorting makes the ordered representation independent of manifest or
        # filesystem iteration order while retaining the exact training set.
        object.__setattr__(self, "training_video_ids", tuple(sorted(identifiers)))
        object.__setattr__(
            self,
            "pose_fingerprint",
            _canonical_sha256(self.pose_fingerprint, "pose_fingerprint"),
        )
        object.__setattr__(
            self,
            "pose_cache_set_sha256",
            _canonical_sha256(
                self.pose_cache_set_sha256,
                "pose_cache_set_sha256",
            ),
        )
        object.__setattr__(self, "source_git_sha", source_revision)
        object.__setattr__(
            self,
            "container_image_id",
            None if image_id is None else str(image_id),
        )
        object.__setattr__(
            self,
            "container_environment_sha256",
            (
                None
                if environment_sha256 is None
                else _canonical_sha256(
                    environment_sha256,
                    "container_environment_sha256",
                )
            ),
        )
        object.__setattr__(
            self,
            "upstream_encoder_checkpoint_sha256",
            (
                None
                if upstream is None
                else _canonical_sha256(
                    upstream,
                    "upstream_encoder_checkpoint_sha256",
                )
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable checkpoint representation."""

        return {
            "protocol": self.protocol,
            "dataset_fingerprint": self.dataset_fingerprint,
            "training_video_ids": list(self.training_video_ids),
            "pose_fingerprint": self.pose_fingerprint,
            "pose_cache_set_sha256": self.pose_cache_set_sha256,
            "source_git_sha": self.source_git_sha,
            "container_image_id": self.container_image_id,
            "container_environment_sha256": self.container_environment_sha256,
            "upstream_encoder_checkpoint_sha256": (self.upstream_encoder_checkpoint_sha256),
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CheckpointProvenance:
        """Parse and validate provenance from an untrusted checkpoint mapping."""

        required = {
            "protocol",
            "dataset_fingerprint",
            "training_video_ids",
            "pose_fingerprint",
            "pose_cache_set_sha256",
            "source_git_sha",
            "container_image_id",
            "container_environment_sha256",
            "upstream_encoder_checkpoint_sha256",
        }
        missing = sorted(required - set(payload))
        unknown = sorted(set(payload) - required)
        if missing or unknown:
            raise ValueError(
                f"checkpoint provenance fields mismatch: missing={missing}, unknown={unknown}"
            )
        identifiers = payload["training_video_ids"]
        if not isinstance(identifiers, list | tuple):
            raise ValueError("checkpoint training_video_ids must be a list")
        if not all(isinstance(identifier, str) for identifier in identifiers):
            raise ValueError("checkpoint training_video_ids must contain strings")
        canonical_identifiers = tuple(sorted(identifiers))
        if tuple(identifiers) != canonical_identifiers:
            raise ValueError("checkpoint training_video_ids are not in canonical order")
        upstream = payload["upstream_encoder_checkpoint_sha256"]
        if upstream is not None and not isinstance(upstream, str):
            raise ValueError("checkpoint upstream encoder SHA-256 must be a string or null")
        image_id = payload["container_image_id"]
        environment_sha256 = payload["container_environment_sha256"]
        if image_id is not None and not isinstance(image_id, str):
            raise ValueError("checkpoint container image ID must be a string or null")
        if environment_sha256 is not None and not isinstance(environment_sha256, str):
            raise ValueError("checkpoint container environment SHA-256 must be a string or null")
        return cls(
            protocol=str(payload["protocol"]),
            dataset_fingerprint=str(payload["dataset_fingerprint"]),
            training_video_ids=canonical_identifiers,
            pose_fingerprint=str(payload["pose_fingerprint"]),
            pose_cache_set_sha256=str(payload["pose_cache_set_sha256"]),
            source_git_sha=str(payload["source_git_sha"]),
            container_image_id=image_id,
            container_environment_sha256=environment_sha256,
            upstream_encoder_checkpoint_sha256=upstream,
        )


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
        input_projection_scale=model.input_projection_scale,
    )
    head = PeriodHead(
        embedding_dim=model.embedding_dim,
        hidden_dim=model.period_head_hidden_dim,
    )
    return PAMSModel(encoder=encoder, period_head=head)


def _post_warmup_period_history_source(config: PAMSConfig) -> PeriodHistorySource:
    if config.period.training_mode == "fixed_period_inferred":
        return "fixed_period_inferred"
    source = config.period.post_warmup_source
    if source == "embedding_velocity_coordinate":
        # Preserve the exact historical receipt label for the default route.
        return "embedding"
    if source == "projected_pose_velocity_vector_acf":
        return "projected_pose_velocity_vector_acf"
    raise AssertionError(f"unreachable validated period source: {source!r}")


def _encoder_period_history_source(
    config: PAMSConfig,
    *,
    epoch: int,
) -> PeriodHistorySource:
    if config.period.training_mode == "fixed_period_inferred":
        return "fixed_period_inferred"
    if epoch <= config.period.pose_energy_epochs:
        return "pose"
    return _post_warmup_period_history_source(config)


def _estimate_fixed_training_periods(
    *,
    config: PAMSConfig,
    valid_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return the preregistered fixed-period proxy and mask-aware evidence.

    The paper describes its Table 2 baseline only as conventional TCC with a
    fixed window.  It does not publish the chosen window or enough geometry to
    recover that implementation.  Our opt-in proxy substitutes the frozen
    ``fixed_period_frames`` for every per-video period used by the current TCC
    and SSHead objectives.  Confidence is one only when two complete periods
    are available, matching the minimum evidence required by the adaptive
    estimator; otherwise periodic positives are disabled for that sample.
    """

    if config.period.training_mode != "fixed_period_inferred":
        raise ValueError("fixed-period estimates require fixed_period_inferred mode")
    if valid_mask.ndim != 2:
        raise ValueError("valid_mask must have shape [batch, time]")
    fixed = config.period.fixed_period_frames
    periods = torch.full(
        (valid_mask.shape[0],),
        float(fixed),
        dtype=torch.float32,
        device=valid_mask.device,
    )
    valid_counts = valid_mask.to(dtype=torch.bool).sum(dim=1)
    confidences = (valid_counts >= 2 * fixed).to(dtype=torch.float32)
    return periods, confidences


def _estimate_post_warmup_periods(
    *,
    config: PAMSConfig,
    embeddings: Tensor,
    projected_pose: Tensor | None,
    valid_mask: Tensor,
) -> tuple[Tensor, Tensor, PeriodHistorySource]:
    source = _post_warmup_period_history_source(config)
    if source == "fixed_period_inferred":
        periods, confidences = _estimate_fixed_training_periods(
            config=config,
            valid_mask=valid_mask,
        )
        return periods, confidences, source
    if source == "embedding":
        periods, confidences = estimate_period_from_embeddings(
            embeddings.detach(),
            minimum=config.period.minimum,
            maximum=config.period.maximum,
            valid_mask=valid_mask,
        )
        return periods, confidences, source
    if projected_pose is None:
        raise RuntimeError(
            "projected-pose period source requires pre-PE encoder features"
        )
    periods, confidences = estimate_period_from_projected_pose(
        projected_pose.detach(),
        minimum=config.period.minimum,
        maximum=config.period.maximum,
        valid_mask=valid_mask,
    )
    return periods, confidences, source


@dataclass(frozen=True, slots=True)
class VideoPrototypeBank:
    """Frozen full-training-set video prototypes refreshed with KMeans."""

    video_ids: tuple[str, ...]
    features: Tensor
    cluster_labels: Tensor

    def __post_init__(self) -> None:
        identifiers = tuple(str(identifier).strip() for identifier in self.video_ids)
        if not identifiers or any(not identifier for identifier in identifiers):
            raise ValueError("prototype bank video_ids must be non-empty")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("prototype bank video_ids must be unique")
        if self.features.ndim != 2 or self.features.shape[0] != len(identifiers):
            raise ValueError("prototype bank features must have shape [videos, dimension]")
        if self.features.shape[1] < 1:
            raise ValueError("prototype bank feature dimension must be positive")
        if self.cluster_labels.shape != (len(identifiers),):
            raise ValueError("prototype bank cluster_labels must have shape [videos]")

        order = sorted(range(len(identifiers)), key=identifiers.__getitem__)
        canonical_ids = tuple(identifiers[index] for index in order)
        order_tensor = torch.tensor(order, dtype=torch.long, device=self.features.device)
        features = (
            self.features.detach()
            .index_select(0, order_tensor)
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
            .clone()
        )
        cluster_labels = (
            self.cluster_labels.detach()
            .index_select(
                0,
                order_tensor.to(device=self.cluster_labels.device),
            )
            .to(device="cpu", dtype=torch.long)
            .contiguous()
            .clone()
        )
        if not torch.isfinite(features).all():
            raise ValueError("prototype bank features must be finite")
        if (cluster_labels < 0).any():
            raise ValueError("prototype bank cluster labels must be non-negative")
        object.__setattr__(self, "video_ids", canonical_ids)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "cluster_labels", cluster_labels)

    @property
    def assignments(self) -> dict[str, int]:
        return {
            identifier: int(cluster)
            for identifier, cluster in zip(
                self.video_ids,
                self.cluster_labels.tolist(),
                strict=True,
            )
        }

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "video_ids": list(self.video_ids),
            "features": self.features.clone(),
            "cluster_labels": self.cluster_labels.clone(),
        }

    @classmethod
    def from_checkpoint(cls, payload: Mapping[str, Any]) -> VideoPrototypeBank:
        expected_keys = {"video_ids", "features", "cluster_labels"}
        if set(payload) != expected_keys:
            raise ValueError("checkpoint prototype bank schema mismatch")
        video_ids = payload["video_ids"]
        features = payload["features"]
        cluster_labels = payload["cluster_labels"]
        if not isinstance(video_ids, list | tuple) or not all(
            isinstance(identifier, str) for identifier in video_ids
        ):
            raise ValueError("checkpoint prototype bank video_ids must be strings")
        if not isinstance(features, Tensor) or not isinstance(cluster_labels, Tensor):
            raise ValueError("checkpoint prototype bank tensors are invalid")
        return cls(
            video_ids=tuple(video_ids),
            features=features,
            cluster_labels=cluster_labels,
        )


@dataclass(frozen=True, slots=True)
class EncoderEpochStats:
    """One completed encoder epoch."""

    epoch: int
    loss: float
    learning_rate: float
    period_source: PeriodHistorySource
    period_confidence_mean: float
    period_valid_fraction: float
    optimizer_steps: int
    clusters_refreshed: bool
    cross_cluster_requested: int
    cross_cluster_actual: int
    cross_cluster_shortfall: int


@dataclass(frozen=True, slots=True)
class SSHeadEpochStats:
    """One completed independent SSHead epoch."""

    epoch: int
    total: float
    cycle: float
    spectral: float
    variance: float
    smoothness: float
    period_confidence_mean: float
    period_valid_fraction: float
    stream_std_min: float
    stream_std_p10: float
    stream_std_median: float
    stream_std_mean: float
    collapsed_fraction_1e6: float
    near_collapsed_fraction_1e3: float
    head_grad_rms_max: float
    head_grad_to_param_ratio_max: float
    zero_grad_steps: int
    learning_rate: float
    optimizer_steps: int


def _validate_encoder_period_history(
    history: Sequence[EncoderEpochStats],
    config: PAMSConfig,
) -> None:
    for statistics in history:
        expected_period_source = _encoder_period_history_source(
            config,
            epoch=statistics.epoch,
        )
        if statistics.period_source != expected_period_source:
            raise ValueError("encoder history period-source schedule mismatch")


@dataclass(frozen=True, slots=True)
class EncoderTrainingResult:
    """Trained model, audit history, and latest clustering state."""

    model: PAMSModel
    history: tuple[EncoderEpochStats, ...]
    cluster_assignments: Mapping[str, int]
    prototype_bank: VideoPrototypeBank | None
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


@dataclass(frozen=True, slots=True)
class _HeadGradientDiagnostics:
    """Norms used by the label-free SSHead optimizer-step guards."""

    grad_l2: float
    grad_rms: float
    param_l2: float
    grad_to_param_ratio: float


def _require_finite_sshead_tensor(value: Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"SSHead produced non-finite {name}")


def _valid_stream_standard_deviations(
    stream: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """Return one population standard deviation per video's valid frames."""

    if stream.ndim != 2 or valid_mask.shape != stream.shape:
        raise ValueError("SSHead stream and valid_mask must share shape [batch, time]")
    deviations: list[Tensor] = []
    for values, sample_valid in zip(stream, valid_mask, strict=True):
        selected = values[sample_valid]
        deviations.append(
            selected.std(unbiased=False) if selected.numel() >= 2 else values.new_tensor(0.0)
        )
    return torch.stack(deviations)


def _head_gradient_diagnostics(
    parameters: Sequence[Tensor],
) -> _HeadGradientDiagnostics:
    """Measure the complete period-head gradient before an optimizer step."""

    parameter_count = sum(parameter.numel() for parameter in parameters)
    if parameter_count == 0:
        raise RuntimeError("SSHead period head has no parameters")

    gradient_square_sum = 0.0
    parameter_square_sum = 0.0
    for parameter in parameters:
        if not bool(torch.isfinite(parameter.detach()).all()):
            raise RuntimeError("SSHead produced a non-finite head parameter")
        parameter_norm = float(torch.linalg.vector_norm(parameter.detach().float()))
        parameter_square_sum += parameter_norm * parameter_norm
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all()):
            raise RuntimeError("SSHead produced a non-finite head gradient")
        gradient_norm = float(torch.linalg.vector_norm(gradient.detach().float()))
        gradient_square_sum += gradient_norm * gradient_norm

    grad_l2 = math.sqrt(gradient_square_sum)
    param_l2 = math.sqrt(parameter_square_sum)
    grad_rms = grad_l2 / math.sqrt(parameter_count)
    grad_to_param_ratio = (
        grad_l2 / param_l2 if param_l2 > 0.0 else (0.0 if grad_l2 == 0.0 else math.inf)
    )
    return _HeadGradientDiagnostics(
        grad_l2=grad_l2,
        grad_rms=grad_rms,
        param_l2=param_l2,
        grad_to_param_ratio=grad_to_param_ratio,
    )


def _guard_sshead_optimizer_step(
    *,
    loss_value: float,
    minimum_stream_std: float,
    diagnostics: _HeadGradientDiagnostics,
) -> None:
    """Reject exact collapse and unstable near-collapse before mutation."""

    if not math.isfinite(loss_value) or not math.isfinite(minimum_stream_std):
        raise RuntimeError("SSHead produced non-finite loss or stream statistics")
    if diagnostics.grad_rms > 10.0 or diagnostics.grad_to_param_ratio > 100.0:
        raise RuntimeError(
            "SSHead near-collapse-gradient-spike guard triggered before optimizer.step"
        )
    if loss_value > 0.05 and diagnostics.grad_l2 <= 1e-12 and minimum_stream_std <= 1e-6:
        raise RuntimeError(
            "SSHead exact-collapse guard triggered before optimizer.step: "
            "positive loss, zero gradient, and collapsed valid-frame stream"
        )


def _require_finite_encoder_tensors(
    named_tensors: Sequence[tuple[str, Tensor]],
) -> None:
    """Reject non-finite encoder state without altering any tensor."""

    values = tuple(named_tensors)
    if not values:
        raise RuntimeError("encoder finite-value guard received no tensors")
    finite = torch.stack(
        [torch.isfinite(tensor.detach()).all() for _, tensor in values]
    ).detach()
    statuses = finite.to(device="cpu").tolist()
    for (name, _), is_finite in zip(values, statuses, strict=True):
        if not bool(is_finite):
            raise RuntimeError(
                f"encoder produced non-finite {name} before optimizer.step"
            )


def _guard_encoder_gradients_before_step(
    named_parameters: Sequence[tuple[str, Tensor]],
) -> None:
    """Require one finite gradient for every trainable encoder parameter."""

    gradients: list[tuple[str, Tensor]] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        if gradient is None:
            raise RuntimeError(
                "encoder trainable parameter "
                f"{name!r} has no gradient before optimizer.step"
            )
        gradients.append((f"gradient for parameter {name!r}", gradient))
    if not gradients:
        raise RuntimeError("encoder has no trainable gradients before optimizer.step")
    _require_finite_encoder_tensors(gradients)


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
    # Provenance binds the training set, not caller iteration order. Sorting
    # makes seeded shuffling, prototype refresh, and resume bitwise-stable
    # even if an equivalent manifest is materialized in a different order.
    return tuple(sorted(items, key=lambda item: item.video_id))


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


def _validate_encoder_microbatch_size(
    dataset_size: int,
    effective_batch_size: int,
    microbatch_size: int | None,
) -> int:
    """Require one complete physical contrastive batch per optimizer step.

    Cross-video and cross-cluster negatives are constructed only from the
    current forward pass. Gradient accumulation cannot recreate that negative
    pool, so encoder training must never present a smaller physical batch as
    the configured effective batch.
    """

    if dataset_size < effective_batch_size:
        raise ValueError(
            "encoder contrastive training requires dataset_size >= "
            f"effective_batch_size ({dataset_size} < {effective_batch_size})"
        )
    selected = effective_batch_size if microbatch_size is None else int(microbatch_size)
    if selected != effective_batch_size:
        raise ValueError(
            "encoder contrastive training requires physical microbatch_size == "
            f"effective_batch_size ({selected} != {effective_batch_size}); "
            "gradient accumulation cannot preserve batch-local cross-video and "
            "cross-cluster negatives"
        )
    return selected


def _data_loader(
    sequences: Sequence[PoseSequence],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_last: bool = False,
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
        drop_last=drop_last,
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


def refresh_video_prototype_bank(
    encoder: PAMSEncoder,
    sequences: Sequence[PoseSequence],
    config: PAMSConfig,
    *,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    epoch: int = 0,
) -> VideoPrototypeBank:
    """Refit KMeans and freeze one full-training-set prototype per video."""

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
    return VideoPrototypeBank(
        video_ids=identifiers,
        features=torch.from_numpy(features.copy()),
        cluster_labels=torch.from_numpy(np.asarray(assignments, dtype=np.int64)),
    )


def refresh_video_clusters(
    encoder: PAMSEncoder,
    sequences: Sequence[PoseSequence],
    config: PAMSConfig,
    *,
    device: str | torch.device | None = None,
    batch_size: int | None = None,
    epoch: int = 0,
) -> dict[str, int]:
    """Compatibility wrapper returning assignments from the refreshed bank."""

    return refresh_video_prototype_bank(
        encoder,
        sequences,
        config,
        device=device,
        batch_size=batch_size,
        epoch=epoch,
    ).assignments


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
    torch_state = state["torch"]
    if not isinstance(torch_state, Tensor):
        raise ValueError("checkpoint torch RNG state must be a tensor")
    # torch.set_rng_state is CPU-only. Checkpoints are loaded onto CPU before
    # restoration, and the explicit conversion protects older checkpoints
    # that may have been serialized from a device-mapped payload.
    torch.set_rng_state(torch_state.detach().cpu())
    if torch.cuda.is_available() and "cuda" in state:
        cuda_states = state["cuda"]
        if not isinstance(cuda_states, Sequence) or not all(
            isinstance(item, Tensor) for item in cuda_states
        ):
            raise ValueError("checkpoint CUDA RNG state must be a tensor sequence")
        torch.cuda.set_rng_state_all([item.detach().cpu() for item in cuda_states])


def _atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    durable_mkdir(path.parent)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            torch.save(dict(payload), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    except Exception:
        if temporary is not None and temporary.exists():
            temporary.unlink()
            fsync_directory(path.parent)
        raise


def _validate_stage_provenance(
    stage: Literal["encoder", "sshead"],
    provenance: CheckpointProvenance,
) -> None:
    upstream = provenance.upstream_encoder_checkpoint_sha256
    if stage == "encoder" and upstream is not None:
        raise ValueError("encoder provenance cannot name an upstream encoder checkpoint")
    if stage == "sshead" and upstream is None:
        raise ValueError("SSHead provenance requires an upstream encoder checkpoint SHA-256")


def _validate_provenance_context(
    provenance: CheckpointProvenance,
    *,
    stage: Literal["encoder", "sshead"],
    config: PAMSConfig,
    training_video_ids: Sequence[str] | None = None,
) -> None:
    _validate_stage_provenance(stage, provenance)
    if provenance.protocol != config.protocol:
        raise ValueError("checkpoint provenance protocol does not match config")
    if provenance.pose_fingerprint != config.pose_fingerprint:
        raise ValueError("checkpoint provenance pose fingerprint does not match config")
    if training_video_ids is not None:
        canonical_ids = tuple(sorted(str(identifier) for identifier in training_video_ids))
        if provenance.training_video_ids != canonical_ids:
            raise ValueError("checkpoint provenance training_video_ids do not match inputs")


def _checkpoint_provenance(
    payload: Mapping[str, Any],
    *,
    stage: Literal["encoder", "sshead"],
    expected: CheckpointProvenance | None,
) -> CheckpointProvenance | None:
    raw = payload.get("provenance")
    if raw is None:
        if expected is not None:
            raise ValueError("checkpoint provenance is missing")
        return None
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint provenance must be a mapping")
    actual = CheckpointProvenance.from_mapping(raw)
    _validate_stage_provenance(stage, actual)
    if expected is None:
        raise ValueError(
            "checkpoint contains bound provenance; caller must provide expected_provenance"
        )
    fields = (
        "protocol",
        "dataset_fingerprint",
        "training_video_ids",
        "pose_fingerprint",
        "pose_cache_set_sha256",
        "source_git_sha",
        "container_image_id",
        "container_environment_sha256",
        "upstream_encoder_checkpoint_sha256",
    )
    for field in fields:
        if getattr(actual, field) != getattr(expected, field):
            raise ValueError(f"checkpoint provenance mismatch for {field}")
    return actual


def _validate_checkpoint_destination(path: Path | None, *, resume: bool) -> None:
    if resume:
        if path is None:
            raise ValueError("resume=True requires checkpoint_path")
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        return
    if path is not None and path.exists():
        raise FileExistsError(f"refusing to overwrite existing checkpoint: {path}")


def _validate_progress_destination(
    path: Path | None,
    *,
    checkpoint_path: Path | None,
    resume: bool,
) -> None:
    if path is None:
        return
    if checkpoint_path is None:
        raise ValueError("progress_path requires checkpoint_path")
    if path.resolve() == checkpoint_path.resolve():
        raise ValueError("progress_path and checkpoint_path must be different files")
    if resume:
        if path.exists() and not path.is_file():
            raise ValueError(f"progress log is not a regular file: {path}")
        return
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing progress log: {path}")


def _progress_row(
    *,
    stage: Literal["encoder", "sshead"],
    stats: EncoderEpochStats | SSHeadEpochStats,
    config: PAMSConfig,
) -> dict[str, Any]:
    statistics = asdict(stats)
    epoch = int(statistics.pop("epoch"))
    if config.period.training_mode == "fixed_period_inferred":
        # Checkpoint identity already binds these values through the config
        # fingerprint.  Repeat them in every terminal progress row so a human
        # audit never has to reverse a hash to discover the effective switch.
        statistics["period_training_mode"] = config.period.training_mode
        statistics["fixed_period_frames"] = config.period.fixed_period_frames
        if stage == "sshead":
            statistics["period_source"] = "fixed_period_inferred"
    elif (
        stage == "sshead"
        and config.period.post_warmup_source
        == "projected_pose_velocity_vector_acf"
    ):
        # SSHead history predates configurable period evidence. Keep the
        # historical default artifact schema intact, while making every
        # inferred vector-ACF progress row explicit and terminally auditable.
        statistics["period_source"] = _post_warmup_period_history_source(config)
    return {
        "schema_version": _PROGRESS_SCHEMA_VERSION,
        "stage": stage,
        "epoch": epoch,
        "stats": statistics,
        "config_fingerprint": config.fingerprint,
        "checkpoint_role": f"{stage}_checkpoint",
    }


def _append_progress_row(
    path: Path,
    row: Mapping[str, Any],
    *,
    create: bool,
) -> None:
    durable_mkdir(path.parent)
    if not create and not path.is_file():
        raise FileNotFoundError(f"progress log disappeared during training: {path}")
    serialized = json.dumps(
        dict(row),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    mode = "x" if create else "a"
    with path.open(mode, encoding="utf-8", newline="\n") as handle:
        handle.write(serialized + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if create:
        fsync_directory(path.parent)


def _read_progress_rows(path: Path) -> tuple[dict[str, Any], ...]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"progress log contains duplicate JSON field: {key}")
            result[key] = value
        return result

    def reject_non_finite(value: str) -> None:
        raise ValueError(f"progress log contains non-finite JSON constant: {value}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"progress log contains a blank line at {line_number}")
            try:
                row = json.loads(
                    line,
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=reject_non_finite,
                )
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"progress log contains invalid JSON at line {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"progress log line {line_number} must be a JSON object")
            expected_keys = {
                "schema_version",
                "stage",
                "epoch",
                "stats",
                "config_fingerprint",
                "checkpoint_role",
            }
            if set(row) != expected_keys:
                raise ValueError(f"progress log schema mismatch at line {line_number}")
            epoch = row["epoch"]
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 1:
                raise ValueError(f"progress log epoch is invalid at line {line_number}")
            if not isinstance(row["stats"], dict):
                raise ValueError(f"progress log stats must be an object at line {line_number}")
            rows.append(row)
    return tuple(rows)


def _validate_and_reconcile_progress(
    path: Path,
    *,
    stage: Literal["encoder", "sshead"],
    history: Sequence[EncoderEpochStats] | Sequence[SSHeadEpochStats],
    completed_epochs: int,
    config: PAMSConfig,
) -> None:
    if completed_epochs != len(history):
        raise ValueError("checkpoint completed_epochs does not match checkpoint history")
    expected_rows = tuple(
        _progress_row(
            stage=stage,
            stats=stats,
            config=config,
        )
        for stats in history
    )
    for expected_epoch, row in enumerate(expected_rows, start=1):
        if row["epoch"] != expected_epoch:
            raise ValueError("checkpoint history epochs are not consecutive from 1")
    rows = _read_progress_rows(path) if path.exists() else ()
    if len(rows) > len(expected_rows):
        raise ValueError("progress log has more epochs than checkpoint history")
    missing_rows = len(expected_rows) - len(rows)
    if missing_rows > 1:
        raise ValueError("progress log is missing more than the final checkpoint epoch")

    fields = (
        "schema_version",
        "stage",
        "epoch",
        "stats",
        "config_fingerprint",
        "checkpoint_role",
    )
    for line_number, (actual, expected) in enumerate(
        zip(rows, expected_rows, strict=False),
        start=1,
    ):
        for field in fields:
            if actual[field] != expected[field]:
                raise ValueError(f"progress log {field} mismatch at line {line_number}")

    # A process can be interrupted after the atomic checkpoint replacement but
    # before the corresponding append. Only that single tail row is safely
    # reconstructable from checkpoint history.
    if missing_rows == 1:
        _append_progress_row(
            path,
            expected_rows[-1],
            create=not path.exists(),
        )


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
    prototype_bank: VideoPrototypeBank | None = None,
    provenance: CheckpointProvenance | None = None,
) -> dict[str, Any]:
    if provenance is not None:
        _validate_stage_provenance(stage, provenance)
    if (
        isinstance(completed_epochs, bool)
        or not isinstance(completed_epochs, int)
        or completed_epochs != len(history)
    ):
        raise ValueError("checkpoint completed_epochs must exactly match history length")
    typed_history = cast(
        Sequence[EncoderEpochStats | SSHeadEpochStats],
        history,
    )
    if any(
        isinstance(item.epoch, bool) or item.epoch != expected_epoch
        for expected_epoch, item in enumerate(typed_history, start=1)
    ):
        raise ValueError("checkpoint history epochs must be consecutive from 1")
    if stage == "encoder":
        _validate_encoder_period_history(
            cast(Sequence[EncoderEpochStats], history),
            config,
        )
    return {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "stage": stage,
        "config_fingerprint": config.fingerprint,
        # Direct API smoke tests may deliberately be unbound. Formal CLI
        # commands always supply this record and loaders require the caller to
        # provide the same expected value before accepting a bound checkpoint.
        "provenance": None if provenance is None else provenance.to_dict(),
        "completed_epochs": completed_epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
        "history": [asdict(item) for item in history],
        "cluster_assignments": dict(cluster_assignments or {}),
        "prototype_bank": (None if prototype_bank is None else prototype_bank.to_checkpoint()),
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
    expected_provenance: CheckpointProvenance | None = None,
) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    # RNG states must remain CPU tensors: torch.set_rng_state rejects CUDA
    # tensors. Model tensors are copied into the already device-resident model,
    # and Optimizer.load_state_dict casts parameter state to each parameter's
    # device according to PyTorch's optimizer state policy.
    payload = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    if payload.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    if payload.get("stage") != stage:
        raise ValueError(f"expected a {stage} checkpoint, received {payload.get('stage')!r}")
    if payload.get("config_fingerprint") != config.fingerprint:
        raise ValueError("checkpoint config fingerprint does not match")
    if expected_provenance is not None:
        _validate_provenance_context(
            expected_provenance,
            stage=stage,
            config=config,
        )
    _checkpoint_provenance(
        payload,
        stage=stage,
        expected=expected_provenance,
    )
    _restore_rng_state(payload["rng_state"])
    model.load_state_dict(payload["model_state"])
    model.to(device)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler_state = payload.get("scheduler_state")
    if scheduler is not None:
        if scheduler_state is None:
            raise ValueError("encoder checkpoint is missing scheduler state")
        scheduler.load_state_dict(scheduler_state)
    return payload


def load_model_checkpoint(
    path: str | Path,
    config: PAMSConfig,
    *,
    device: str | torch.device | None = None,
    expected_stage: Literal["encoder", "sshead"] | None = None,
    expected_provenance: CheckpointProvenance | None = None,
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
    typed_stage = cast(Literal["encoder", "sshead"], stage)
    if expected_provenance is not None:
        _validate_provenance_context(
            expected_provenance,
            stage=typed_stage,
            config=config,
        )
    _checkpoint_provenance(
        payload,
        stage=typed_stage,
        expected=expected_provenance,
    )
    model = build_pams_model(config).to(resolved_device)
    model.load_state_dict(payload["model_state"])
    return model


def validate_terminal_checkpoint(
    path: str | Path,
    config: PAMSConfig,
    *,
    expected_stage: Literal["encoder", "sshead"],
    expected_provenance: CheckpointProvenance,
    progress_path: str | Path,
) -> CheckpointProvenance:
    """Validate that a formal checkpoint and progress log reached all epochs.

    This is deliberately read-only.  Resume-time progress reconciliation is
    useful after interruption, but publication must require an already exact
    terminal log rather than repairing evidence during evaluation.
    """

    source = Path(path)
    progress = Path(progress_path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    if not progress.is_file():
        raise FileNotFoundError(f"terminal checkpoint progress log is missing: {progress}")
    payload = torch.load(source, map_location=torch.device("cpu"), weights_only=False)
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
        raise ValueError("terminal checkpoint top-level schema mismatch")
    if payload["schema_version"] != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema")
    if payload["stage"] != expected_stage:
        raise ValueError(
            f"expected a {expected_stage} checkpoint, received {payload['stage']!r}"
        )
    if payload["config_fingerprint"] != config.fingerprint:
        raise ValueError("checkpoint config fingerprint does not match")
    _validate_provenance_context(
        expected_provenance,
        stage=expected_stage,
        config=config,
    )
    actual_provenance = _checkpoint_provenance(
        payload,
        stage=expected_stage,
        expected=expected_provenance,
    )
    if actual_provenance is None:
        raise ValueError("terminal checkpoint must contain bound provenance")
    model_state = payload["model_state"]
    if not isinstance(model_state, Mapping):
        raise ValueError("terminal checkpoint model_state must be a mapping")

    completed_epochs = payload["completed_epochs"]
    if (
        isinstance(completed_epochs, bool)
        or not isinstance(completed_epochs, int)
        or completed_epochs < 1
    ):
        raise ValueError("checkpoint completed_epochs must be a positive integer")
    required_epochs = (
        config.training.epochs if expected_stage == "encoder" else config.sshead.epochs
    )
    if completed_epochs != required_epochs:
        raise ValueError(
            f"{expected_stage} checkpoint is partial: completed_epochs="
            f"{completed_epochs}, required={required_epochs}"
        )

    raw_history = payload["history"]
    if not isinstance(raw_history, list) or len(raw_history) != completed_epochs:
        raise ValueError("terminal checkpoint history length does not match completed_epochs")
    statistics_type = EncoderEpochStats if expected_stage == "encoder" else SSHeadEpochStats
    integer_fields = (
        {
            "epoch",
            "optimizer_steps",
            "cross_cluster_requested",
            "cross_cluster_actual",
            "cross_cluster_shortfall",
        }
        if expected_stage == "encoder"
        else {"epoch", "zero_grad_steps", "optimizer_steps"}
    )
    boolean_fields = {"clusters_refreshed"} if expected_stage == "encoder" else set()
    text_fields = {"period_source"} if expected_stage == "encoder" else set()
    expected_history_fields = set(statistics_type.__dataclass_fields__)
    history: list[EncoderEpochStats | SSHeadEpochStats] = []
    for expected_epoch, raw_statistics in enumerate(raw_history, start=1):
        if not isinstance(raw_statistics, dict):
            raise ValueError(f"checkpoint history epoch {expected_epoch} must be a mapping")
        if set(raw_statistics) != expected_history_fields:
            raise ValueError(
                f"checkpoint history schema mismatch at epoch {expected_epoch}"
            )
        for name, value in raw_statistics.items():
            if name in integer_fields:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(
                        f"checkpoint history field {name!r} must be an integer"
                    )
            elif name in boolean_fields:
                if not isinstance(value, bool):
                    raise ValueError(
                        f"checkpoint history field {name!r} must be boolean"
                    )
            elif name in text_fields:
                if not isinstance(value, str):
                    raise ValueError(
                        f"checkpoint history field {name!r} must be text"
                    )
            elif not isinstance(value, float):
                raise ValueError(
                    f"checkpoint history field {name!r} must be a float"
                )
        try:
            statistics = statistics_type(**raw_statistics)
        except TypeError as exc:
            raise ValueError(
                f"checkpoint history schema mismatch at epoch {expected_epoch}"
            ) from exc
        if (
            isinstance(statistics.epoch, bool)
            or not isinstance(statistics.epoch, int)
            or statistics.epoch != expected_epoch
        ):
            raise ValueError("checkpoint history epochs are not consecutive from 1")
        for name, value in asdict(statistics).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"checkpoint history contains non-finite {name!r} "
                    f"at epoch {expected_epoch}"
                )
        history.append(statistics)

    if expected_stage == "encoder":
        encoder_history = cast(list[EncoderEpochStats], history)
        _validate_encoder_period_history(encoder_history, config)
        for statistics in encoder_history:
            if (
                statistics.cross_cluster_requested
                != statistics.cross_cluster_actual + statistics.cross_cluster_shortfall
            ):
                raise ValueError("encoder history cross-cluster accounting mismatch")
            if statistics.cross_cluster_shortfall != 0:
                raise ValueError("formal terminal encoder contains cross-cluster shortfall")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            initialized_model = build_pams_model(config)
        initialized_head = {
            f"period_head.{key}": value
            for key, value in initialized_model.period_head.state_dict().items()
        }
        checkpoint_head = {
            str(key): value
            for key, value in model_state.items()
            if str(key).startswith("period_head.")
        }
        if set(checkpoint_head) != set(initialized_head):
            raise ValueError("literal encoder period-head tensor keys differ from initialization")
        for key in sorted(initialized_head):
            value = checkpoint_head[key]
            if not isinstance(value, Tensor) or not torch.equal(
                value.detach().cpu(),
                initialized_head[key].detach().cpu(),
            ):
                raise ValueError(
                    "literal encoder period head differs from deterministic initialization"
                )
        raw_assignments = payload["cluster_assignments"]
        raw_bank = payload["prototype_bank"]
        if not isinstance(raw_assignments, Mapping) or not isinstance(raw_bank, Mapping):
            raise ValueError("terminal encoder is missing cluster/prototype state")
        if not all(
            isinstance(identifier, str)
            and isinstance(cluster, int)
            and not isinstance(cluster, bool)
            for identifier, cluster in raw_assignments.items()
        ):
            raise ValueError("terminal encoder cluster assignments have invalid types")
        assignments = dict(raw_assignments)
        bank = VideoPrototypeBank.from_checkpoint(raw_bank)
        if bank.video_ids != actual_provenance.training_video_ids:
            raise ValueError("terminal encoder prototype IDs do not match provenance")
        if bank.features.shape[1] != config.model.embedding_dim:
            raise ValueError("terminal encoder prototype feature dimension is invalid")
        if bank.assignments != assignments:
            raise ValueError("terminal encoder cluster and prototype assignments differ")
        if not isinstance(payload["scheduler_state"], Mapping):
            raise ValueError("terminal encoder is missing scheduler state")
    else:
        if payload["cluster_assignments"] != {} or payload["prototype_bank"] is not None:
            raise ValueError("terminal SSHead checkpoint contains encoder clustering state")
        if payload["scheduler_state"] is not None:
            raise ValueError("terminal SSHead checkpoint contains unexpected scheduler state")

    for field in ("optimizer_state", "rng_state"):
        if not isinstance(payload[field], Mapping):
            raise ValueError(f"terminal checkpoint {field} must be a mapping")

    rows = _read_progress_rows(progress)
    if len(rows) != completed_epochs:
        raise ValueError("terminal progress log length does not match completed_epochs")
    expected_rows = tuple(
        _progress_row(
            stage=expected_stage,
            stats=statistics,
            config=config,
        )
        for statistics in history
    )
    if rows != expected_rows:
        raise ValueError("terminal progress log does not exactly match checkpoint history")
    return actual_provenance


def validate_sshead_encoder_binding(
    sshead_checkpoint: str | Path,
    encoder_checkpoint: str | Path,
) -> None:
    """Require the SSHead's frozen encoder tensors to equal its named upstream."""

    head_payload = torch.load(
        Path(sshead_checkpoint),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    encoder_payload = torch.load(
        Path(encoder_checkpoint),
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    if not isinstance(head_payload, dict) or not isinstance(encoder_payload, dict):
        raise ValueError("checkpoint root must be a mapping")
    if head_payload.get("stage") != "sshead" or encoder_payload.get("stage") != "encoder":
        raise ValueError("SSHead encoder binding requires sshead and encoder stages")
    head_state = head_payload.get("model_state")
    encoder_state = encoder_payload.get("model_state")
    if not isinstance(head_state, Mapping) or not isinstance(encoder_state, Mapping):
        raise ValueError("checkpoint model_state must be a mapping")
    head_encoder = {
        key: value for key, value in head_state.items() if str(key).startswith("encoder.")
    }
    upstream_encoder = {
        key: value for key, value in encoder_state.items() if str(key).startswith("encoder.")
    }
    if not head_encoder or set(head_encoder) != set(upstream_encoder):
        raise ValueError("SSHead and upstream encoder tensor keys differ")
    for key in sorted(head_encoder):
        head_tensor = head_encoder[key]
        upstream_tensor = upstream_encoder[key]
        if not isinstance(head_tensor, Tensor) or not isinstance(upstream_tensor, Tensor):
            raise ValueError(f"encoder state value is not a tensor: {key}")
        if not torch.equal(head_tensor, upstream_tensor):
            raise ValueError(
                f"SSHead embedded encoder differs from upstream checkpoint: {key}"
            )


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
    progress_path: str | Path | None = None,
    resume: bool = False,
    stop_after_epoch: int | None = None,
    provenance: CheckpointProvenance | None = None,
    allow_negative_shortfall: bool = True,
) -> EncoderTrainingResult:
    """Train the encoder with PAMS-TCC without accepting any ground truth."""

    items = _materialize_sequences(sequences)
    destination = None if checkpoint_path is None else Path(checkpoint_path)
    progress_destination = None if progress_path is None else Path(progress_path)
    _validate_checkpoint_destination(destination, resume=resume)
    _validate_progress_destination(
        progress_destination,
        checkpoint_path=destination,
        resume=resume,
    )
    if not isinstance(allow_negative_shortfall, bool):
        raise TypeError("allow_negative_shortfall must be boolean")
    if provenance is not None:
        _validate_provenance_context(
            provenance,
            stage="encoder",
            config=config,
            training_video_ids=[item.video_id for item in items],
        )
    seed_everything(config.seed)
    resolved_device = _resolve_device(device)
    trained_model = build_pams_model(config) if model is None else model
    trained_model.to(resolved_device)
    selected_microbatch = _validate_encoder_microbatch_size(
        len(items),
        config.training.effective_batch_size,
        microbatch_size,
    )

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
        exclude_other_scale_positives_from_denominator=(
            config.loss.exclude_other_scale_positives_from_denominator
        ),
    )
    history: list[EncoderEpochStats] = []
    cluster_assignments: dict[str, int] = {}
    prototype_bank: VideoPrototypeBank | None = None
    completed_epochs = 0

    if resume:
        assert destination is not None
        payload = _load_checkpoint(
            destination,
            stage="encoder",
            config=config,
            model=trained_model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=resolved_device,
            expected_provenance=provenance,
        )
        completed_epochs = int(payload["completed_epochs"])
        history = [EncoderEpochStats(**row) for row in payload["history"]]
        _validate_encoder_period_history(history, config)
        historical_shortfall = sum(stats.cross_cluster_shortfall for stats in history)
        if historical_shortfall and not allow_negative_shortfall:
            raise RuntimeError(
                "checkpoint history contains cross-cluster prototype bank "
                f"shortfall={historical_shortfall}"
            )
        cluster_assignments = {
            str(identifier): int(cluster)
            for identifier, cluster in payload["cluster_assignments"].items()
        }
        raw_bank = payload.get("prototype_bank")
        if not isinstance(raw_bank, Mapping):
            raise ValueError("encoder checkpoint is missing its prototype bank")
        prototype_bank = VideoPrototypeBank.from_checkpoint(raw_bank)
        expected_video_ids = tuple(sorted(item.video_id for item in items))
        if prototype_bank.video_ids != expected_video_ids:
            raise ValueError("checkpoint prototype bank video_ids do not match training inputs")
        if prototype_bank.features.shape[1] != config.model.embedding_dim:
            raise ValueError("checkpoint prototype bank feature dimension does not match config")
        if prototype_bank.assignments != cluster_assignments:
            raise ValueError("checkpoint prototype bank cluster labels do not match assignments")
        if progress_destination is not None:
            _validate_and_reconcile_progress(
                progress_destination,
                stage="encoder",
                history=history,
                completed_epochs=completed_epochs,
                config=config,
            )

    end_epoch = _target_end_epoch(
        config.training.epochs,
        stop_after_epoch,
    )
    if completed_epochs > end_epoch:
        raise ValueError("checkpoint has already passed the requested stop_after_epoch")
    progress_needs_creation = progress_destination is not None and not progress_destination.exists()

    for epoch_index in range(completed_epochs, end_epoch):
        refresh = prototype_bank is None or epoch_index % config.loss.kmeans_refresh_epochs == 0
        if refresh:
            prototype_bank = refresh_video_prototype_bank(
                trained_model.encoder,
                items,
                config,
                device=resolved_device,
                batch_size=selected_microbatch,
                epoch=epoch_index,
            )
            cluster_assignments = prototype_bank.assignments
        assert prototype_bank is not None
        bank_features = prototype_bank.features.to(device=resolved_device)
        bank_cluster_labels = prototype_bank.cluster_labels.to(device=resolved_device)

        loader = _data_loader(
            items,
            batch_size=selected_microbatch,
            shuffle=True,
            seed=config.seed + epoch_index,
            drop_last=True,
        )
        trained_model.encoder.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        processed_samples = 0
        optimizer_steps = 0
        confidence_sum = 0.0
        period_valid_samples = 0
        cross_cluster_requested = 0
        cross_cluster_actual = 0
        cross_cluster_shortfall = 0
        for raw_batch in loader:
            if not isinstance(raw_batch, PoseBatch):
                raise TypeError("pose collator returned an unexpected batch type")
            batch = raw_batch.to(resolved_device)
            if batch.batch_size != config.training.effective_batch_size:
                raise RuntimeError(
                    "encoder loader produced an incomplete physical contrastive batch"
                )
            projected_pose: Tensor | None = None
            if (
                config.period.training_mode == "adaptive"
                and epoch_index >= config.period.pose_energy_epochs
                and config.period.post_warmup_source
                == "projected_pose_velocity_vector_acf"
            ):
                embeddings, projected_pose = trained_model.encoder.forward_with_pre_pe(
                    batch.poses,
                    batch.valid_mask,
                )
            else:
                embeddings = trained_model.encoder(batch.poses, batch.valid_mask)
            period_source: PeriodHistorySource
            if config.period.training_mode == "fixed_period_inferred":
                periods, period_confidences = _estimate_fixed_training_periods(
                    config=config,
                    valid_mask=batch.valid_mask,
                )
                period_source = "fixed_period_inferred"
            elif epoch_index < config.period.pose_energy_epochs:
                periods, period_confidences = estimate_period_from_pose(
                    batch.poses,
                    minimum=config.period.minimum,
                    maximum=config.period.maximum,
                    valid_mask=batch.valid_mask,
                )
                period_source = "pose"
            else:
                (
                    periods,
                    period_confidences,
                    period_source,
                ) = _estimate_post_warmup_periods(
                    config=config,
                    embeddings=embeddings,
                    projected_pose=projected_pose,
                    valid_mask=batch.valid_mask,
                )
            batch_clusters = torch.tensor(
                [cluster_assignments[identifier] for identifier in batch.video_ids],
                dtype=torch.long,
                device=resolved_device,
            )
            details = objective.compute(
                embeddings,
                periods,
                batch.valid_mask,
                period_confidence=period_confidences,
                cluster_labels=batch_clusters,
                video_ids=batch.video_ids,
                bank_features=bank_features,
                bank_cluster_labels=bank_cluster_labels,
                bank_video_ids=prototype_bank.video_ids,
            )
            loss = details.total
            batch_cross_cluster_requested = sum(details.cross_cluster_requested_counts)
            batch_cross_cluster_actual = sum(details.cross_cluster_actual_counts)
            batch_cross_cluster_shortfall = sum(details.cross_cluster_shortfall_counts)
            if batch_cross_cluster_shortfall and not allow_negative_shortfall:
                raise RuntimeError(
                    "cross-cluster prototype bank shortfall before optimizer.step "
                    f"in epoch {epoch_index + 1}: "
                    f"requested={batch_cross_cluster_requested}, "
                    f"actual={batch_cross_cluster_actual}, "
                    f"shortfall={batch_cross_cluster_shortfall}"
                )

            _require_finite_encoder_tensors(
                (
                    ("embeddings", embeddings),
                    ("period estimates", periods),
                    ("period confidences", period_confidences),
                    ("loss", loss),
                )
            )
            loss.backward()
            _guard_encoder_gradients_before_step(
                tuple(trained_model.encoder.named_parameters())
            )
            epoch_loss += float(loss.detach()) * batch.batch_size
            processed_samples += batch.batch_size
            confidence_sum += float(period_confidences.detach().sum())
            period_valid_samples += int((period_confidences != 0).sum())
            cross_cluster_requested += batch_cross_cluster_requested
            cross_cluster_actual += batch_cross_cluster_actual
            cross_cluster_shortfall += batch_cross_cluster_shortfall
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        if processed_samples == 0:
            raise RuntimeError("encoder loader produced no complete contrastive batch")
        mean_loss = epoch_loss / processed_samples
        scheduler.step(mean_loss)
        completed_epochs = epoch_index + 1
        history.append(
            EncoderEpochStats(
                epoch=completed_epochs,
                loss=mean_loss,
                learning_rate=float(optimizer.param_groups[0]["lr"]),
                period_source=period_source,
                period_confidence_mean=confidence_sum / processed_samples,
                period_valid_fraction=period_valid_samples / processed_samples,
                optimizer_steps=optimizer_steps,
                clusters_refreshed=refresh,
                cross_cluster_requested=cross_cluster_requested,
                cross_cluster_actual=cross_cluster_actual,
                cross_cluster_shortfall=cross_cluster_shortfall,
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
                    prototype_bank=prototype_bank,
                    provenance=provenance,
                ),
                destination,
            )
            if progress_destination is not None:
                _append_progress_row(
                    progress_destination,
                    _progress_row(
                        stage="encoder",
                        stats=history[-1],
                        config=config,
                    ),
                    create=progress_needs_creation,
                )
                progress_needs_creation = False

    return EncoderTrainingResult(
        model=trained_model,
        history=tuple(history),
        cluster_assignments=cluster_assignments,
        prototype_bank=prototype_bank,
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
    progress_path: str | Path | None = None,
    resume: bool = False,
    stop_after_epoch: int | None = None,
    provenance: CheckpointProvenance | None = None,
) -> SSHeadTrainingResult:
    """Train only the inferred SSHead objective with the encoder frozen."""

    items = _materialize_sequences(sequences)
    destination = None if checkpoint_path is None else Path(checkpoint_path)
    progress_destination = None if progress_path is None else Path(progress_path)
    _validate_checkpoint_destination(destination, resume=resume)
    _validate_progress_destination(
        progress_destination,
        checkpoint_path=destination,
        resume=resume,
    )
    if provenance is not None:
        _validate_provenance_context(
            provenance,
            stage="sshead",
            config=config,
            training_video_ids=[item.video_id for item in items],
        )
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
    head_parameters = tuple(model.period_head.parameters())
    optimizer = AdamW(
        head_parameters,
        lr=config.sshead.learning_rate,
        weight_decay=config.sshead.weight_decay,
    )
    objective = SSHeadLoss(
        cycle_weight=config.sshead.cycle_weight,
        spectral_weight=config.sshead.spectral_weight,
        variance_weight=config.sshead.variance_weight,
        smoothness_weight=config.sshead.smoothness_weight,
    )
    history: list[SSHeadEpochStats] = []
    completed_epochs = 0

    if resume:
        assert destination is not None
        payload = _load_checkpoint(
            destination,
            stage="sshead",
            config=config,
            model=model,
            optimizer=optimizer,
            device=resolved_device,
            expected_provenance=provenance,
        )
        completed_epochs = int(payload["completed_epochs"])
        history = [SSHeadEpochStats(**row) for row in payload["history"]]
        if progress_destination is not None:
            _validate_and_reconcile_progress(
                progress_destination,
                stage="sshead",
                history=history,
                completed_epochs=completed_epochs,
                config=config,
            )

    end_epoch = _target_end_epoch(config.sshead.epochs, stop_after_epoch)
    if completed_epochs > end_epoch:
        raise ValueError("checkpoint has already passed the requested stop_after_epoch")
    progress_needs_creation = progress_destination is not None and not progress_destination.exists()

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
        confidence_sum = 0.0
        period_valid_samples = 0
        stream_standard_deviations: list[float] = []
        head_grad_rms_max = 0.0
        head_grad_to_param_ratio_max = 0.0
        zero_grad_steps = 0
        group_loss_value = 0.0
        group_minimum_stream_std = math.inf
        for batch_index, raw_batch in enumerate(loader):
            if not isinstance(raw_batch, PoseBatch):
                raise TypeError("pose collator returned an unexpected batch type")
            batch = raw_batch.to(resolved_device)
            with torch.no_grad():
                needs_projected_pose = (
                    config.sshead.input_source == "projected_pose_pre_pe"
                    or (
                        config.period.training_mode == "adaptive"
                        and config.period.post_warmup_source
                        == "projected_pose_velocity_vector_acf"
                    )
                )
                if needs_projected_pose:
                    embeddings, projected_pose = model.encoder.forward_with_pre_pe(
                        batch.poses,
                        batch.valid_mask,
                    )
                else:
                    embeddings = model.encoder(batch.poses, batch.valid_mask)
                    projected_pose = None
                (
                    periods,
                    period_confidences,
                    _,
                ) = _estimate_post_warmup_periods(
                    config=config,
                    embeddings=embeddings,
                    projected_pose=projected_pose,
                    valid_mask=batch.valid_mask,
                )
            if config.sshead.input_source == "encoder_embedding":
                head_inputs = embeddings
            elif config.sshead.input_source == "projected_pose_pre_pe":
                if projected_pose is None:
                    raise RuntimeError(
                        "projected-pose SSHead input was not materialized"
                    )
                head_inputs = projected_pose
            else:
                raise AssertionError(
                    "unreachable validated SSHead input source: "
                    f"{config.sshead.input_source!r}"
                )
            stream = model.period_head(head_inputs.detach())
            stream = stream.masked_fill(~batch.valid_mask, 0.0)
            _require_finite_sshead_tensor(stream, "period stream")
            with torch.no_grad():
                batch_stream_stds = _valid_stream_standard_deviations(
                    stream,
                    batch.valid_mask,
                )
            _require_finite_sshead_tensor(
                batch_stream_stds,
                "valid-frame stream standard deviation",
            )
            batch_std_values = [float(value) for value in batch_stream_stds.detach().cpu()]
            stream_standard_deviations.extend(batch_std_values)
            details = objective.compute(
                stream,
                periods,
                batch.valid_mask,
                period_confidence=period_confidences,
            )
            for name in sums:
                _require_finite_sshead_tensor(
                    getattr(details, name),
                    f"{name} loss",
                )

            group_start = (batch_index // accumulation_steps) * accumulation_steps
            samples_before_group = group_start * selected_microbatch
            group_sample_count = min(
                config.training.effective_batch_size,
                len(items) - samples_before_group,
            )
            scaled_loss = details.total * (batch.batch_size / group_sample_count)
            scaled_loss.backward()
            # Validate every microbatch's accumulated gradients, even when the
            # actual optimizer step occurs only after gradient accumulation.
            _head_gradient_diagnostics(head_parameters)
            group_loss_value += float(scaled_loss.detach())
            group_minimum_stream_std = min(
                group_minimum_stream_std,
                min(batch_std_values),
            )
            for name in sums:
                sums[name] += float(getattr(details, name).detach()) * batch.batch_size
            confidence_sum += float(period_confidences.detach().sum())
            period_valid_samples += int((period_confidences != 0).sum())

            end_of_group = (
                batch_index + 1
            ) % accumulation_steps == 0 or batch_index + 1 == batch_count
            if end_of_group:
                diagnostics = _head_gradient_diagnostics(head_parameters)
                _guard_sshead_optimizer_step(
                    loss_value=group_loss_value,
                    minimum_stream_std=group_minimum_stream_std,
                    diagnostics=diagnostics,
                )
                head_grad_rms_max = max(
                    head_grad_rms_max,
                    diagnostics.grad_rms,
                )
                head_grad_to_param_ratio_max = max(
                    head_grad_to_param_ratio_max,
                    diagnostics.grad_to_param_ratio,
                )
                zero_grad_steps += int(diagnostics.grad_l2 <= 1e-12)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                group_loss_value = 0.0
                group_minimum_stream_std = math.inf

        completed_epochs = epoch_index + 1
        stream_std_array = np.asarray(stream_standard_deviations, dtype=np.float64)
        history.append(
            SSHeadEpochStats(
                epoch=completed_epochs,
                total=sums["total"] / len(items),
                cycle=sums["cycle"] / len(items),
                spectral=sums["spectral"] / len(items),
                variance=sums["variance"] / len(items),
                smoothness=sums["smoothness"] / len(items),
                period_confidence_mean=confidence_sum / len(items),
                period_valid_fraction=period_valid_samples / len(items),
                stream_std_min=float(np.min(stream_std_array)),
                stream_std_p10=float(np.percentile(stream_std_array, 10.0)),
                stream_std_median=float(np.median(stream_std_array)),
                stream_std_mean=float(np.mean(stream_std_array)),
                collapsed_fraction_1e6=float(np.mean(stream_std_array <= 1e-6)),
                near_collapsed_fraction_1e3=float(np.mean(stream_std_array <= 1e-3)),
                head_grad_rms_max=head_grad_rms_max,
                head_grad_to_param_ratio_max=head_grad_to_param_ratio_max,
                zero_grad_steps=zero_grad_steps,
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
                    provenance=provenance,
                ),
                destination,
            )
            if progress_destination is not None:
                _append_progress_row(
                    progress_destination,
                    _progress_row(
                        stage="sshead",
                        stats=history[-1],
                        config=config,
                    ),
                    create=progress_needs_creation,
                )
                progress_needs_creation = False

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
        _, stream_batch = model.forward_with_head_source(
            batch.poses,
            batch.valid_mask,
            head_input_source=config.sshead.input_source,
        )
        periods, period_confidences = estimate_period_batch(
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
            expert_mode=consensus.expert_mode,
        )
    result = counter.count(
        stream,
        period_frames=float(periods[0]),
        valid_mask=mask,
        period_confidence=float(period_confidences[0]),
    ).to_count_result()
    model.train(previous_training)
    return result
