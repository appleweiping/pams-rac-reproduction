from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import pams.temporac.prediction as prediction_module
from pams.temporac.contract import CANONICAL_JOB_NAMES, ReasonCode
from pams.temporac.prediction import (
    PredictionIdentity,
    build_identity_prediction,
    build_prediction_artifact,
)
from pams.temporac.receipts import parse_receipt_bytes
from pams.temporac.types import ContractError


def test_identity_prediction_decodes_once_and_is_immutable(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    real_decode = prediction_module.decode_identity

    def counted_decode(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(prediction_module, "decode_identity", counted_decode)
    identity = PredictionIdentity(bytes(range(32)), 7)
    record = build_identity_prediction(
        identity,
        condition="natural-drift",
        response=np.asarray([0.1, 0.5, 0.8, 0.2, 0.9], dtype=np.float64),
        edge_mask=np.ones(5, dtype=np.uint8),
        decoder_mask=np.ones(5, dtype=np.uint8),
        run_bounds=np.asarray([[0, 5]], dtype=np.int32),
    )
    assert calls == 1
    assert int(record.count[0]) == 2
    assert record.component_bounds.tolist() == [[1, 3], [4, 5]]
    assert not record.response.flags.writeable
    assert record.response.dtype == np.dtype("<f4")

    job = CANONICAL_JOB_NAMES[0]
    artifact = build_prediction_artifact(
        record,
        arm="local",
        condition="natural-drift",
        checkpoint_sha256="1" * 64,
        feature_receipt_sha256="2" * 64,
        job_name=job,
        seed=20260815,
    )
    duplicate = build_prediction_artifact(
        record,
        arm="local",
        condition="natural-drift",
        checkpoint_sha256="1" * 64,
        feature_receipt_sha256="2" * 64,
        job_name=job,
        seed=20260815,
    )
    assert artifact.artifact_bytes == duplicate.artifact_bytes
    assert artifact.receipt_bytes == duplicate.receipt_bytes
    receipt = parse_receipt_bytes(artifact.receipt_bytes)
    assert receipt["artifact_sha256"] == artifact.artifact_sha256


def test_stub_invokes_empty_decoder_once_and_has_no_partial_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    real_decode = prediction_module.decode_identity

    def counted_decode(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        return real_decode(*args, **kwargs)

    monkeypatch.setattr(prediction_module, "decode_identity", counted_decode)
    identity = PredictionIdentity(b"k" * 32, 0)
    stub = build_identity_prediction(
        identity,
        condition="natural-clean",
        abstain_reasons=(ReasonCode.TOO_SHORT, ReasonCode.RESPONSE_NONFINITE),
    )
    assert calls == 1
    assert int(stub.abstain[0]) == 1
    assert int(stub.count[0]) == -1
    assert stub.abstain_reasons.tolist() == [9, 21]
    assert stub.response.shape == (0,)
    assert stub.run_bounds.shape == (0, 2)
    assert stub.component_bounds.shape == (0, 2)

    with pytest.raises(ContractError, match="partial"):
        build_identity_prediction(
            identity,
            condition="natural-clean",
            response=np.empty(0),
            abstain_reasons=(ReasonCode.TOO_SHORT,),
        )


def test_two_identity_artifacts_do_not_couple() -> None:
    def artifact_for(key: bytes, response: np.ndarray) -> bytes:
        record = build_identity_prediction(
            PredictionIdentity(key, 0),
            condition="natural-clean",
            response=response,
            edge_mask=np.ones(4, dtype=np.uint8),
            decoder_mask=np.ones(4, dtype=np.uint8),
            run_bounds=np.asarray([[0, 4]], dtype=np.int32),
        )
        return build_prediction_artifact(
            record,
            arm="uniform",
            condition="natural-clean",
            checkpoint_sha256="3" * 64,
            feature_receipt_sha256="4" * 64,
            job_name=CANONICAL_JOB_NAMES[0],
            seed=20260815,
        ).artifact_bytes

    identity_b = artifact_for(b"b" * 32, np.asarray([0.1, 0.7, 0.2, 0.8]))
    _ = artifact_for(b"a" * 32, np.asarray([0.9, 0.9, 0.9, 0.9]))
    identity_b_after = artifact_for(b"b" * 32, np.asarray([0.1, 0.7, 0.2, 0.8]))
    assert identity_b == identity_b_after
