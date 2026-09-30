"""Natural diagnostic runtime with stable-pair periodic reconstruction."""

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
from pams.temporac.certify_adaptive_v4 import stable_pair_integer_landmarks
from pams.temporac.certify_target_adaptive_v3 import certify_target_adaptive_v3
from pams.temporac.periodic_template_candidate_v1 import (
    periodic_template_reconstruction,
)
from pams.temporac.phase_closure_candidate_v1 import close_phase_on_landmarks
from pams.temporac.preprocess import preprocess_identity
from pams.temporac.receipts import build_target_artifact
from pams.temporac.teacher import TempoRACTeacher, arc_static_features
from pams.temporac.types import FeatureRecord


def infer_natural_pseudo_target_v8(
    model: TempoRACTeacher,
    source_feature: FeatureRecord,
    source_receipt_bytes: bytes,
    *,
    expected_source_receipt_sha256: str,
    selected_teacher_sha256: str,
    device: torch.device,
) -> NaturalPseudoTarget:
    """Infer a bound phase, then use a count-blind stable periodic template."""

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
        _code, phase, _raw_reconstruction = model(
            torch.from_numpy(np.array(prepared.teacher_input, copy=True)).to(device),
            torch.from_numpy(static.astype(np.float32)).to(device),
        )
    sample_bounds = np.array(prepared.run_bounds, dtype=np.int32, copy=True)
    sample_bounds[:, 1] += 1
    landmark_result = stable_pair_integer_landmarks(
        prepared.geometry_features.geometry,
        sample_bounds,
    )
    if landmark_result.landmarks.size < 3:
        closed_phase = phase.detach().cpu().numpy().astype("<f8")
        reconstruction = np.zeros_like(continuous)
    else:
        closed_phase = close_phase_on_landmarks(
            phase.detach().cpu().numpy().astype("<f8"),
            landmark_result.landmarks,
            geometry=prepared.geometry_features.geometry,
        )
        geometry_features = prepared.geometry_features
        continuous_mask = np.concatenate(
            (
                geometry_features.coordinate_mask,
                np.repeat(geometry_features.direction_element_mask, 2, axis=1),
                prepared.joint_mask,
            ),
            axis=1,
        ).astype(np.float64)
        reconstruction = periodic_template_reconstruction(
            continuous,
            continuous_mask,
            geometry_features.geometry,
            landmark_result.landmarks,
        )
    output = build_natural_teacher_output(
        feature=adapted,
        feature_receipt_bytes=feature_receipt,
        expected_feature_receipt_sha256=feature_receipt_sha,
        selected_teacher_sha256=selected_teacher_sha256,
        phase=closed_phase,
        reconstruction=reconstruction,
    )
    target = certify_target_adaptive_v3(
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


__all__ = ["infer_natural_pseudo_target_v8"]
