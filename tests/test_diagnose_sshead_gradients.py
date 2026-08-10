from __future__ import annotations

import pytest
import torch

from scripts.diagnose_sshead_gradients import _select_head_inputs


def test_sshead_diagnostic_routes_configured_input_source() -> None:
    embeddings = torch.full((2, 3, 4), 1.0)
    projected_pose = torch.full((2, 3, 4), 2.0)

    assert (
        _select_head_inputs(
            "encoder_embedding",
            embeddings,
            projected_pose,
        )
        is embeddings
    )
    assert (
        _select_head_inputs(
            "projected_pose_pre_pe",
            embeddings,
            projected_pose,
        )
        is projected_pose
    )


def test_sshead_diagnostic_rejects_missing_pre_pe_input() -> None:
    with pytest.raises(
        RuntimeError,
        match="projected-pose SSHead input was not materialized",
    ):
        _select_head_inputs(
            "projected_pose_pre_pe",
            torch.zeros((1, 2, 3)),
            None,
        )
