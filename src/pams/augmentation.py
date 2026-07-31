"""Deterministic, target-free skeleton augmentation for encoder training."""

from __future__ import annotations

import hashlib
import math

import torch
from torch import Tensor

from pams.config import SkeletonAugmentationConfig


def skeleton_augmentation_seed(
    *,
    base_seed: int,
    epoch_index: int,
    batch_index: int,
) -> int:
    """Derive a stable torch seed from an experiment seed and batch identity."""

    if epoch_index < 0:
        raise ValueError("epoch_index must be non-negative")
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative")
    payload = f"pams-skeleton-augmentation-v1:{base_seed}:{epoch_index}:{batch_index}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") & ((1 << 63) - 1)


def make_skeleton_augmentation_generator(
    *,
    base_seed: int,
    epoch_index: int,
    batch_index: int,
    device: torch.device,
) -> torch.Generator:
    """Create the isolated generator for one deterministic training batch."""

    generator = torch.Generator(device=device)
    generator.manual_seed(
        skeleton_augmentation_seed(
            base_seed=base_seed,
            epoch_index=epoch_index,
            batch_index=batch_index,
        )
    )
    return generator


def _rotation_matrices(angles: Tensor) -> Tensor:
    """Return ``Rz @ Ry @ Rx`` matrices for ``angles`` shaped ``[batch, 3]``."""

    sine = torch.sin(angles)
    cosine = torch.cos(angles)
    sx, sy, sz = sine.unbind(dim=1)
    cx, cy, cz = cosine.unbind(dim=1)
    zeros = torch.zeros_like(sx)
    ones = torch.ones_like(sx)

    rotation_x = torch.stack(
        (
            ones,
            zeros,
            zeros,
            zeros,
            cx,
            -sx,
            zeros,
            sx,
            cx,
        ),
        dim=1,
    ).reshape(-1, 3, 3)
    rotation_y = torch.stack(
        (
            cy,
            zeros,
            sy,
            zeros,
            ones,
            zeros,
            -sy,
            zeros,
            cy,
        ),
        dim=1,
    ).reshape(-1, 3, 3)
    rotation_z = torch.stack(
        (
            cz,
            -sz,
            zeros,
            sz,
            cz,
            zeros,
            zeros,
            zeros,
            ones,
        ),
        dim=1,
    ).reshape(-1, 3, 3)
    return rotation_z @ rotation_y @ rotation_x


def augment_pose_batch(
    poses: Tensor,
    valid_mask: Tensor,
    config: SkeletonAugmentationConfig,
    *,
    generator: torch.Generator,
) -> Tensor:
    """Apply one rigid transform per sequence plus valid-joint Gaussian jitter.

    Rotation and isotropic scaling share one draw across every valid frame in
    a sequence. Each frame is transformed around its 33-joint centroid so the
    augmentation does not create artificial global translation. Invalid and
    padded frames are reset to exact zero after every transform.
    """

    if not config.enabled:
        raise ValueError("augment_pose_batch requires enabled augmentation")
    if poses.ndim != 4 or poses.shape[-1] != 3:
        raise ValueError("poses must have shape [batch, time, joints, 3]")
    if valid_mask.shape != poses.shape[:2]:
        raise ValueError("valid_mask must match poses batch and time dimensions")
    if valid_mask.dtype is not torch.bool:
        raise TypeError("valid_mask must have boolean dtype")
    if not poses.is_floating_point():
        raise TypeError("poses must use a floating-point dtype")
    if generator.device != poses.device:
        raise ValueError("generator and poses must use the same device")

    batch_size = poses.shape[0]
    radians = torch.tensor(
        config.rotation_degrees,
        dtype=poses.dtype,
        device=poses.device,
    ) * (math.pi / 180.0)
    angle_uniform = torch.rand(
        (batch_size, 3),
        dtype=poses.dtype,
        device=poses.device,
        generator=generator,
    )
    angles = (2.0 * angle_uniform - 1.0) * radians.unsqueeze(0)
    rotation = _rotation_matrices(angles)

    scale_minimum, scale_maximum = config.scale_range
    scale_uniform = torch.rand(
        (batch_size, 1, 1, 1),
        dtype=poses.dtype,
        device=poses.device,
        generator=generator,
    )
    scales = scale_minimum + (scale_maximum - scale_minimum) * scale_uniform

    valid = valid_mask.unsqueeze(-1).unsqueeze(-1)
    clean = poses.masked_fill(~valid, 0.0)
    centers = clean.mean(dim=2, keepdim=True)
    centered = clean - centers
    rotated = torch.matmul(
        centered,
        rotation.transpose(1, 2).unsqueeze(1),
    )
    augmented = rotated * scales + centers
    if config.jitter_std > 0.0:
        jitter = torch.randn(
            poses.shape,
            dtype=poses.dtype,
            device=poses.device,
            generator=generator,
        )
        augmented = augmented + config.jitter_std * jitter
    return augmented.masked_fill(~valid, 0.0)
