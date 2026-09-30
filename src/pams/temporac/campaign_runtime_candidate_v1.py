"""Additive end-to-end candidate runtime for TempoRAC teacher and targets.

The frozen model and contract modules are imported unchanged.  This module
supplies the missing executable glue: X0 teacher loss, resampled natural
feature binding, teacher inference, certification, pseudo-target creation,
and differentiable response losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from pams.temporac.certify import (
    bind_natural_certificate_track,
    build_natural_teacher_output,
    certify_target,
)
from pams.temporac.contract import FEATURE_RECEIPT_SCHEMA
from pams.temporac.cue import compute_routing
from pams.temporac.hashio import feature_npz_bytes, sha256_bytes
from pams.temporac.nola import nola_reconstruct_torch, window_taper
from pams.temporac.objective import traversal_unit_loss
from pams.temporac.preprocess import preprocess_identity
from pams.temporac.preprocess_resampled_v2 import (
    ResamplingReceipt,
    resample_feature_record,
)
from pams.temporac.receipts import (
    TargetArtifact,
    build_target_artifact,
    member_payload,
    receipt_bytes,
    validate_feature_record_receipt,
)
from pams.temporac.response import TempoRACResponse
from pams.temporac.teacher import TempoRACTeacher, arc_static_features, phase_increments
from pams.temporac.types import CertifiedTarget, FeatureRecord
from pams.temporac.x0 import X0View, generate_view


class CampaignRuntimeError(RuntimeError):
    """Raised when the candidate runtime cannot produce an auditable unit."""


@dataclass(frozen=True, slots=True)
class TeacherView:
    view: X0View
    teacher_input: Tensor
    reconstruction_target: Tensor
    reconstruction_mask: Tensor
    static_features: Tensor
    clean_coordinate: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class NaturalPseudoTarget:
    adapted_feature: FeatureRecord
    adapted_feature_receipt_bytes: bytes
    adapted_feature_receipt_sha256: str
    resampling_receipt: ResamplingReceipt
    target: CertifiedTarget
    target_artifact: TargetArtifact | None


@lru_cache(maxsize=224)
def _teacher_view(source_id: int, block_index: int, device: torch.device) -> TeacherView:
    view = generate_view(source_id, block_index, resampler="linear", offset=0)
    prepared = preprocess_identity(
        view.motion,
        np.ones(view.motion.shape[0], dtype=np.uint8),
        view.clock,
    )
    continuous = np.asarray(prepared.teacher_input[:, :149], dtype=np.float64)
    static = arc_static_features(continuous, prepared.edge_lengths, prepared.run_bounds)
    # The first 66 mask channels correspond to the 149 reconstruction channels
    # as 66 geometry, 66 direction, and 17 confidence values.
    geometry_mask = np.asarray(prepared.teacher_input[:, 149:182] != 0, dtype=np.float32)
    direction_mask = np.asarray(prepared.teacher_input[:, 182:215] != 0, dtype=np.float32)
    mask149 = np.concatenate(
        (
            np.repeat(geometry_mask, 2, axis=1),
            np.repeat(direction_mask, 2, axis=1),
            np.ones((continuous.shape[0], 17), dtype=np.float32),
        ),
        axis=1,
    )
    if mask149.shape != continuous.shape:
        raise CampaignRuntimeError("teacher reconstruction mask width differs from 149")
    return TeacherView(
        view=view,
        teacher_input=torch.from_numpy(np.array(prepared.teacher_input, copy=True)).to(device),
        reconstruction_target=torch.from_numpy(continuous.astype(np.float32)).to(device),
        reconstruction_mask=torch.from_numpy(mask149).to(device),
        static_features=torch.from_numpy(static.astype(np.float32)).to(device),
        clean_coordinate=np.asarray(view.clean_coordinate, dtype=np.float64),
    )


def _masked_huber(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    error = torch.abs(prediction - target)
    delta = 0.05
    loss = torch.where(error <= delta, 0.5 * error.square() / delta, error - 0.5 * delta)
    denominator = torch.sum(mask)
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 0:
        raise CampaignRuntimeError("teacher reconstruction mask is empty")
    return torch.sum(loss * mask) / denominator


def _phase_at_coordinate(
    phase: Tensor,
    coordinate: NDArray[np.float64],
    query: NDArray[np.float64],
) -> Tensor:
    right = np.searchsorted(coordinate, query, side="right")
    right = np.clip(right, 1, len(coordinate) - 1)
    left = right - 1
    denominator = coordinate[right] - coordinate[left]
    fraction = np.divide(
        query - coordinate[left], denominator, out=np.zeros_like(query), where=denominator > 0
    )
    left_phase = phase[torch.as_tensor(left, device=phase.device)]
    right_phase = phase[torch.as_tensor(right, device=phase.device)]
    cross = left_phase[:, 0] * right_phase[:, 1] - left_phase[:, 1] * right_phase[:, 0]
    dot = torch.sum(left_phase * right_phase, dim=1)
    angle = torch.atan2(cross, dot)
    base = torch.atan2(left_phase[:, 1], left_phase[:, 0])
    fraction_tensor = torch.as_tensor(fraction, dtype=phase.dtype, device=phase.device)
    output = base + fraction_tensor * angle
    return torch.stack((torch.cos(output), torch.sin(output)), dim=1)


def teacher_source_loss(
    model: TempoRACTeacher,
    source_id: int,
    *,
    device: torch.device,
) -> Tensor:
    """Compute one complete seven-block X0 teacher source loss."""

    if not 0 <= source_id < 32:
        raise CampaignRuntimeError("teacher source must be train/tune X0")
    views = [_teacher_view(source_id, block, device) for block in range(7)]
    lengths = [int(row.teacher_input.shape[0]) for row in views]
    maximum = max(lengths)
    teacher_batch = torch.zeros((7, maximum, 215), dtype=torch.float32, device=device)
    target_batch = torch.zeros((7, maximum, 149), dtype=torch.float32, device=device)
    mask_batch = torch.zeros((7, maximum, 149), dtype=torch.float32, device=device)
    static_batch = torch.stack([row.static_features for row in views])
    for index, row in enumerate(views):
        length = lengths[index]
        teacher_batch[index, :length] = row.teacher_input
        target_batch[index, :length] = row.reconstruction_target
        mask_batch[index, :length] = row.reconstruction_mask
    _static, phase_batch, reconstruction_batch = model(teacher_batch, static_batch)
    phases: list[Tensor] = []
    rec_losses: list[Tensor] = []
    orientation_losses: list[Tensor] = []
    alias_losses: list[Tensor] = []
    for index, _row in enumerate(views):
        length = lengths[index]
        phase = phase_batch[index, :length]
        reconstruction = reconstruction_batch[index, :length]
        phases.append(phase)
        rec_losses.append(
            _masked_huber(
                reconstruction,
                target_batch[index, :length],
                mask_batch[index, :length],
            )
        )
        increments = phase_increments(phase)
        orientation_losses.append(torch.mean(torch.relu(-increments)))
        alias_losses.append(torch.mean(torch.relu(torch.abs(increments) - 0.25)))
    queries = np.concatenate(
        [traversal + (np.arange(128, dtype=np.float64) + 0.5) / 128.0 for traversal in range(10)]
    )
    reference = _phase_at_coordinate(phases[0], views[0].clean_coordinate, queries)
    correspondence = torch.mean(
        torch.stack(
            [
                torch.mean(
                    1.0
                    - torch.sum(
                        reference
                        * _phase_at_coordinate(phases[index], views[index].clean_coordinate, queries),
                        dim=1,
                    )
                )
                for index in range(1, 7)
            ]
        )
    )
    total = (
        torch.mean(torch.stack(rec_losses))
        + correspondence
        + 0.25 * torch.mean(torch.stack(orientation_losses))
        + 0.25 * torch.mean(torch.stack(alias_losses))
    )
    if not bool(torch.isfinite(total)):
        raise CampaignRuntimeError("teacher source loss is nonfinite")
    return total


def adapted_feature_receipt(
    source_feature: FeatureRecord,
    source_receipt_bytes: bytes,
    *,
    expected_source_receipt_sha256: str,
) -> tuple[FeatureRecord, bytes, str, ResamplingReceipt]:
    """Bind a resampled feature to the verified source receipt and standard schema."""

    source_receipt = validate_feature_record_receipt(
        source_feature,
        source_receipt_bytes,
        expected_receipt_sha256=expected_source_receipt_sha256,
    )
    adapted, resampling = resample_feature_record(source_feature)
    artifact, members = feature_npz_bytes(adapted)
    payload: dict[str, object] = {
        "artifact_bytes": len(artifact),
        "artifact_sha256": sha256_bytes(artifact),
        "contract_sha256": source_receipt["contract_sha256"],
        "members": member_payload(members),
        "opaque_key_hex": adapted.opaque_key_bytes.hex(),
        "schema": FEATURE_RECEIPT_SCHEMA,
        "slot": adapted.slot,
        "source_binding_sha256": source_receipt["source_binding_sha256"],
    }
    encoded = receipt_bytes(payload, expected_schema=FEATURE_RECEIPT_SCHEMA)
    return adapted, encoded, sha256_bytes(encoded), resampling


def infer_natural_pseudo_target(
    model: TempoRACTeacher,
    source_feature: FeatureRecord,
    source_receipt_bytes: bytes,
    *,
    expected_source_receipt_sha256: str,
    selected_teacher_sha256: str,
    device: torch.device,
) -> NaturalPseudoTarget:
    """Resample, infer, certify, and package one natural pseudo-target."""

    adapted, feature_receipt, feature_receipt_sha, resampling = adapted_feature_receipt(
        source_feature,
        source_receipt_bytes,
        expected_source_receipt_sha256=expected_source_receipt_sha256,
    )
    prepared = preprocess_identity(
        adapted.motion, adapted.frame_mask, adapted.sampled_frame_indices
    )
    continuous = np.asarray(prepared.teacher_input[:, :149], dtype=np.float64)
    static = arc_static_features(continuous, prepared.edge_lengths, prepared.run_bounds)
    with torch.inference_mode():
        _code, phase, reconstruction = model(
            torch.from_numpy(np.array(prepared.teacher_input, copy=True)).to(device),
            torch.from_numpy(static.astype(np.float32)).to(device),
        )
    output = build_natural_teacher_output(
        feature=adapted,
        feature_receipt_bytes=feature_receipt,
        expected_feature_receipt_sha256=feature_receipt_sha,
        selected_teacher_sha256=selected_teacher_sha256,
        phase=phase.detach().cpu().numpy().astype("<f8"),
        reconstruction=reconstruction.detach().cpu().numpy().astype("<f8"),
    )
    target = certify_target(
        bind_natural_certificate_track(feature=adapted, teacher_output=output),
        "natural",
    )
    artifact = build_target_artifact(target) if target.certified else None
    return NaturalPseudoTarget(
        adapted_feature=adapted,
        adapted_feature_receipt_bytes=feature_receipt,
        adapted_feature_receipt_sha256=feature_receipt_sha,
        resampling_receipt=resampling,
        target=target,
        target_artifact=artifact,
    )


def _edge_responsibility(
    gates: NDArray[np.float64],
    windows: list[tuple[int, int]],
    edge_mask: NDArray[np.uint8],
) -> NDArray[np.float64]:
    numerator = np.zeros((edge_mask.size, 3), dtype=np.float64)
    denominator = np.zeros(edge_mask.size, dtype=np.float64)
    for gate, (start, stop) in zip(gates, windows, strict=True):
        taper = window_taper(stop - start)
        valid = edge_mask[start:stop].astype(np.float64)
        numerator[start:stop] += taper[:, None] * valid[:, None] * gate[None, :]
        denominator[start:stop] += taper * valid
    result = np.zeros_like(numerator)
    supported = denominator > 0
    result[supported] = numerator[supported] / denominator[supported, None]
    return result


def natural_response_loss(
    model: TempoRACResponse,
    feature: FeatureRecord,
    target: CertifiedTarget,
    *,
    device: torch.device,
) -> Tensor:
    """Compute differentiable local-routing response loss for one natural identity."""

    if not target.certified:
        raise CampaignRuntimeError("natural response loss requires a certified target")
    prepared = preprocess_identity(feature.motion, feature.frame_mask, feature.sampled_frame_indices)
    geometry = prepared.geometry_features
    edge_mask = np.asarray(target.edge_mask, dtype=np.uint8)
    routing = compute_routing(
        geometry.geometry,
        geometry.edge_coordinate_mask,
        prepared.clocks,
        edge_mask,
        geometry.edge_lengths,
        prepared.run_bounds,
    )
    inputs = torch.from_numpy(np.array(prepared.response_input, copy=True)).to(device)
    branch = model.cached_responses(
        inputs, np.array(prepared.run_bounds, copy=True)
    ).edge_probabilities
    windows = [(window.edge_start, window.edge_stop) for window in routing.windows]
    fused = nola_reconstruct_torch(branch, routing.local_gates, windows, edge_mask).response
    responsibility = _edge_responsibility(routing.local_gates, windows, edge_mask)
    return traversal_unit_loss(
        branch,
        fused.to(dtype=branch.dtype),
        torch.from_numpy(np.asarray(target.pulse, dtype=np.float32)).to(device),
        torch.from_numpy(edge_mask).to(device),
        torch.from_numpy(np.asarray(target.chi, dtype=np.float32)).to(device),
        torch.from_numpy(responsibility.astype(np.float32)).to(device),
        target.traversal_bounds,
    ).total


def x0_response_loss(
    model: TempoRACResponse,
    source_id: int,
    *,
    device: torch.device,
) -> Tensor:
    """Compute the equal-seven-block analytic X0 response loss for one source."""

    if not 0 <= source_id < 32:
        raise CampaignRuntimeError("response source must be train/tune X0")
    block_losses: list[Tensor] = []
    for block_index in range(7):
        view = generate_view(source_id, block_index, resampler="linear", offset=0)
        prepared = preprocess_identity(
            view.motion,
            np.ones(view.motion.shape[0], dtype=np.uint8),
            view.clock,
        )
        first, stop = prepared.trim_bounds
        branch = model.cached_responses(
            torch.from_numpy(np.array(prepared.response_input, copy=True)).to(device),
            np.array(prepared.run_bounds, copy=True),
        ).edge_probabilities
        responsibility = np.asarray(view.edge_responsibility[first : stop - 1], dtype=np.float32)
        pulse = np.array(view.pulse[first : stop - 1], dtype=np.float32, copy=True)
        chi = np.array(view.chi[first : stop - 1], dtype=np.float32, copy=True)
        target_mask = np.array(
            view.target_mask[first : stop - 1], dtype=np.uint8, copy=True
        )
        traversal_bounds = np.array(view.traversal_bounds, dtype=np.int32, copy=True) - first
        if responsibility.shape != tuple(branch.shape):
            raise CampaignRuntimeError("X0 responsibility differs from branch response shape")
        fused = torch.sum(
            branch * torch.from_numpy(responsibility).to(device), dim=1
        )
        block_losses.append(
            traversal_unit_loss(
                branch,
                fused,
                torch.from_numpy(pulse).to(device),
                torch.from_numpy(target_mask).to(device),
                torch.from_numpy(chi).to(device),
                torch.from_numpy(responsibility).to(device),
                traversal_bounds,
            ).total
        )
    result = torch.mean(torch.stack(block_losses))
    if not bool(torch.isfinite(result)):
        raise CampaignRuntimeError("X0 response loss is nonfinite")
    return result


__all__ = [
    "CampaignRuntimeError",
    "NaturalPseudoTarget",
    "adapted_feature_receipt",
    "infer_natural_pseudo_target",
    "natural_response_loss",
    "teacher_source_loss",
    "x0_response_loss",
]
