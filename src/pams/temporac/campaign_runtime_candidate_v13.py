"""Deterministic complete-chain runtime with a conservative stable-pair fallback."""

from __future__ import annotations

import torch

from pams.temporac.campaign_runtime_candidate_v1 import NaturalPseudoTarget
from pams.temporac.campaign_runtime_candidate_v9 import infer_natural_pseudo_target_v9
from pams.temporac.campaign_runtime_candidate_v12 import infer_natural_pseudo_target_v12
from pams.temporac.teacher import TempoRACTeacher
from pams.temporac.types import FeatureRecord


def infer_natural_pseudo_target_v13(
    model: TempoRACTeacher,
    source_feature: FeatureRecord,
    source_receipt_bytes: bytes,
    *,
    expected_source_receipt_sha256: str,
    selected_teacher_sha256: str,
    device: torch.device,
) -> NaturalPseudoTarget:
    """Use v12 when certified and otherwise apply the fixed v9 fallback."""

    complete_chain = infer_natural_pseudo_target_v12(
        model,
        source_feature,
        source_receipt_bytes,
        expected_source_receipt_sha256=expected_source_receipt_sha256,
        selected_teacher_sha256=selected_teacher_sha256,
        device=device,
    )
    if complete_chain.target.certified:
        return complete_chain
    return infer_natural_pseudo_target_v9(
        model,
        source_feature,
        source_receipt_bytes,
        expected_source_receipt_sha256=expected_source_receipt_sha256,
        selected_teacher_sha256=selected_teacher_sha256,
        device=device,
    )


__all__ = ["infer_natural_pseudo_target_v13"]
