"""Fixed v13 runtime with an all-return-valid landmark fallback."""

from __future__ import annotations

import torch

from pams.temporac.campaign_runtime_candidate_v1 import NaturalPseudoTarget
from pams.temporac.campaign_runtime_candidate_v13 import infer_natural_pseudo_target_v13
from pams.temporac.campaign_runtime_candidate_v15_landmarks import (
    infer_natural_pseudo_target_v15_landmarks,
)
from pams.temporac.teacher import TempoRACTeacher
from pams.temporac.types import FeatureRecord


def infer_natural_pseudo_target_v15(
    model: TempoRACTeacher,
    source_feature: FeatureRecord,
    source_receipt_bytes: bytes,
    *,
    expected_source_receipt_sha256: str,
    selected_teacher_sha256: str,
    device: torch.device,
) -> NaturalPseudoTarget:
    """Use v13 when certified, then apply the deterministic v15 fallback."""

    established = infer_natural_pseudo_target_v13(
        model,
        source_feature,
        source_receipt_bytes,
        expected_source_receipt_sha256=expected_source_receipt_sha256,
        selected_teacher_sha256=selected_teacher_sha256,
        device=device,
    )
    if established.target.certified:
        return established
    return infer_natural_pseudo_target_v15_landmarks(
        model,
        source_feature,
        source_receipt_bytes,
        expected_source_receipt_sha256=expected_source_receipt_sha256,
        selected_teacher_sha256=selected_teacher_sha256,
        device=device,
    )


__all__ = ["infer_natural_pseudo_target_v15"]

