from __future__ import annotations

import numpy as np
import torch

from pams.temporac.campaign_runtime_candidate_v1 import (
    adapted_feature_receipt,
    infer_natural_pseudo_target,
    teacher_source_loss,
    x0_response_loss,
)
from pams.temporac.contract import CONTRACT_SHA256, FEATURE_RECEIPT_SCHEMA
from pams.temporac.hashio import feature_npz_bytes, sha256_bytes
from pams.temporac.receipts import member_payload, receipt_bytes
from pams.temporac.response import TempoRACResponse
from pams.temporac.teacher import TempoRACTeacher
from pams.temporac.types import FeatureRecord


def _feature_and_receipt() -> tuple[FeatureRecord, bytes, str]:
    clock = np.arange(320, dtype="<i8")
    motion = np.zeros((320, 17, 3), dtype="<f4")
    for joint in range(17):
        motion[:, joint, 0] = joint + np.sin(clock / (8.0 + joint / 20.0))
        motion[:, joint, 1] = np.cos(clock / (11.0 + joint / 20.0))
        motion[:, joint, 2] = 0.95
    mask = np.ones(320, dtype="|u1")
    mask[100:120] = 0
    motion[100:120] = 0
    feature = FeatureRecord(
        frame_mask=mask,
        local_person_slot=np.asarray([0], dtype="<i8"),
        motion=motion,
        opaque_sample_key=np.arange(32, dtype="|u1"),
        person_mask=np.ones(1, dtype="|u1"),
        sampled_frame_indices=clock,
        source_length=np.asarray([320], dtype="<i8"),
    )
    artifact, members = feature_npz_bytes(feature)
    payload = {
        "artifact_bytes": len(artifact),
        "artifact_sha256": sha256_bytes(artifact),
        "contract_sha256": CONTRACT_SHA256,
        "members": member_payload(members),
        "opaque_key_hex": feature.opaque_key_bytes.hex(),
        "schema": FEATURE_RECEIPT_SCHEMA,
        "slot": feature.slot,
        "source_binding_sha256": "1" * 64,
    }
    encoded = receipt_bytes(payload, expected_schema=FEATURE_RECEIPT_SCHEMA)
    return feature, encoded, sha256_bytes(encoded)


def test_teacher_source_loss_is_finite_and_reaches_all_modules() -> None:
    model = TempoRACTeacher()
    loss = teacher_source_loss(model, 0, device=torch.device("cpu"))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_resampled_receipt_and_pseudo_target_runtime_are_bound() -> None:
    feature, encoded, digest = _feature_and_receipt()
    adapted, adapted_receipt, adapted_digest, resampling = adapted_feature_receipt(
        feature, encoded, expected_source_receipt_sha256=digest
    )
    assert adapted.opaque_key_bytes == feature.opaque_key_bytes
    assert adapted_digest == sha256_bytes(adapted_receipt)
    assert resampling.interpolated_rows == 20
    model = TempoRACTeacher().eval()
    result = infer_natural_pseudo_target(
        model,
        feature,
        encoded,
        expected_source_receipt_sha256=digest,
        selected_teacher_sha256="2" * 64,
        device=torch.device("cpu"),
    )
    assert result.adapted_feature_receipt_sha256 == adapted_digest
    assert result.target.status in {"CERTIFIED", "ABSTAIN"}
    assert (result.target_artifact is not None) == result.target.certified


def test_x0_response_loss_is_differentiable() -> None:
    model = TempoRACResponse()
    loss = x0_response_loss(model, 0, device=torch.device("cpu"))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
