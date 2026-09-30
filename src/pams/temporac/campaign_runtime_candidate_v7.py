"""Natural pseudo-target runtime with predeclared count-blind phase closure."""

from __future__ import annotations

import numpy as np
import torch

from pams.temporac.campaign_runtime_candidate_v1 import (
    NaturalPseudoTarget,
    adapted_feature_receipt,
)
from pams.temporac.certify import (
    bind_natural_certificate_track,
    build_natural_teacher_output,
)
from pams.temporac.certify_adaptive_v2 import hybrid_integer_landmarks
from pams.temporac.certify_target_adaptive_v2 import certify_target_adaptive_v2
from pams.temporac.phase_closure_candidate_v1 import close_phase_on_landmarks
from pams.temporac.preprocess import preprocess_identity
from pams.temporac.receipts import build_target_artifact
from pams.temporac.teacher import TempoRACTeacher, arc_static_features
from pams.temporac.types import FeatureRecord


def infer_natural_pseudo_target_v7(
    model: TempoRACTeacher,
    source_feature: FeatureRecord,
    source_receipt_bytes: bytes,
    *,
    expected_source_receipt_sha256: str,
    selected_teacher_sha256: str,
    device: torch.device,
) -> NaturalPseudoTarget:
    """Infer then close natural phase on count-blind geometry returns."""

    adapted, feature_receipt, feature_receipt_sha, resampling = adapted_feature_receipt(
        source_feature,
        source_receipt_bytes,
        expected_source_receipt_sha256=expected_source_receipt_sha256,
    )
    prepared = preprocess_identity(
        adapted.motion,
        adapted.frame_mask,
        adapted.sampled_frame_indices,
    )
    continuous = np.asarray(prepared.teacher_input[:, :149], dtype=np.float64)
    static = arc_static_features(continuous, prepared.edge_lengths, prepared.run_bounds)
    with torch.inference_mode():
        code, phase, _raw_reconstruction = model(
            torch.from_numpy(np.array(prepared.teacher_input, copy=True)).to(device),
            torch.from_numpy(static.astype(np.float32)).to(device),
        )
    sample_bounds = np.array(prepared.run_bounds, dtype=np.int32, copy=True)
    sample_bounds[:, 1] += 1
    landmark_result = hybrid_integer_landmarks(
        prepared.geometry_features.geometry,
        sample_bounds,
    )
    closed_phase = close_phase_on_landmarks(
        phase.detach().cpu().numpy().astype("<f8"),
        landmark_result.landmarks,
        geometry=prepared.geometry_features.geometry,
    )
    closed_phase_tensor = torch.from_numpy(np.array(closed_phase, dtype=np.float32, copy=True)).to(
        device
    )
    decoder = model.base.reconstruction if hasattr(model, "base") else model.reconstruction
    with torch.inference_mode():
        reconstruction = decoder(closed_phase_tensor, code)
    output = build_natural_teacher_output(
        feature=adapted,
        feature_receipt_bytes=feature_receipt,
        expected_feature_receipt_sha256=feature_receipt_sha,
        selected_teacher_sha256=selected_teacher_sha256,
        phase=closed_phase,
        reconstruction=reconstruction.detach().cpu().numpy().astype("<f8"),
    )
    target = certify_target_adaptive_v2(
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


__all__ = ["infer_natural_pseudo_target_v7"]
