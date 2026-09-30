from __future__ import annotations

import numpy as np

from pams.temporac.preprocess_resampled_v2 import (
    PROTOCOL,
    preprocess_resampled_identity,
    resample_feature_record,
)
from pams.temporac.types import FeatureRecord


def _feature() -> FeatureRecord:
    clock = np.arange(320, dtype="<i8")
    motion = np.zeros((320, 17, 3), dtype="<f4")
    for joint in range(17):
        motion[:, joint, 0] = np.sin(clock / (7.0 + joint / 10.0)) + joint
        motion[:, joint, 1] = np.cos(clock / (9.0 + joint / 10.0))
        motion[:, joint, 2] = 0.9
    mask = np.ones(320, dtype="|u1")
    mask[80:110] = 0
    motion[80:110] = 0.0
    return FeatureRecord(
        frame_mask=mask,
        local_person_slot=np.asarray([2], dtype="<i8"),
        motion=motion,
        opaque_sample_key=np.arange(32, dtype="|u1"),
        person_mask=np.ones(1, dtype="|u1"),
        sampled_frame_indices=clock,
        source_length=np.asarray([640], dtype="<i8"),
    )


def test_resampling_is_additive_deterministic_and_preprocessable() -> None:
    source = _feature()
    source_motion = source.motion.copy()
    first, receipt_a = resample_feature_record(source)
    second, receipt_b = resample_feature_record(source)
    assert receipt_a == receipt_b
    assert receipt_a.protocol == PROTOCOL
    assert receipt_a.interpolated_rows == 30
    assert np.array_equal(source.motion, source_motion)
    assert np.array_equal(first.motion, second.motion)
    assert np.all(first.frame_mask == 1)
    assert np.array_equal(first.sampled_frame_indices, np.arange(320))
    prepared, receipt_c = preprocess_resampled_identity(
        source.motion, source.frame_mask, source.sampled_frame_indices
    )
    assert receipt_c == receipt_a
    assert prepared.response_input.shape == (320, 269)
    assert prepared.teacher_input.shape == (320, 215)
    assert prepared.run_bounds.tolist() == [[0, 319]]
