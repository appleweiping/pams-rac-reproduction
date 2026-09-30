from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

import pams.temporac.decode as decode_module
from pams.temporac.cue import window_schedule
from pams.temporac.decode import DecodeError, decode_identity
from pams.temporac.nola import (
    NOLAError,
    nola_reconstruct,
    nola_reconstruct_torch,
    probability_fusion,
)


def test_probability_level_fusion_and_exact_positive_nola() -> None:
    edge_count = 200
    probabilities = np.linspace(0.1, 0.9, edge_count)[:, None] * np.ones((1, 3))
    windows = window_schedule(np.array([[0, edge_count]], dtype=np.int32))
    gates = np.tile(np.array([0.2, 0.3, 0.5]), (len(windows), 1))
    edge_mask = np.ones(edge_count, dtype=np.uint8)
    edge_mask[80:85] = 0
    nola_windows = tuple((window.edge_start, window.edge_stop) for window in windows)
    result = nola_reconstruct(probabilities, gates, nola_windows, edge_mask)

    assert np.all(result.denominator[edge_mask == 1] >= 1e-3)
    assert np.all(result.denominator[edge_mask == 0] == 0.0)
    assert np.all(result.response[edge_mask == 0] == 0.0)
    assert np.allclose(
        result.response[edge_mask == 1], probabilities[edge_mask == 1, 0], rtol=0.0, atol=1e-15
    )
    fused = probability_fusion(
        np.array([[0.1, 0.4, 0.9]], dtype=np.float64),
        np.array([0.2, 0.3, 0.5], dtype=np.float64),
    )
    assert fused[0] == pytest.approx(0.59)


def test_nola_has_no_epsilon_fallback_for_uncovered_valid_edge() -> None:
    probabilities = np.full((10, 3), 0.5)
    with pytest.raises(NOLAError, match="denominator"):
        nola_reconstruct(
            probabilities,
            np.array([[1.0 / 3.0] * 3]),
            [(0, 9)],
            np.ones(10, dtype=np.uint8),
        )


def test_torch_nola_keeps_fused_probability_gradient_and_detaches_gate() -> None:
    probabilities = torch.full((16, 3), 0.6, dtype=torch.float32, requires_grad=True)
    gates = torch.full((1, 3), 1.0 / 3.0, dtype=torch.float64, requires_grad=True)
    result = nola_reconstruct_torch(
        probabilities,
        gates,
        [(0, 16)],
        torch.ones(16, dtype=torch.uint8),
    )
    result.response.sum().backward()
    assert probabilities.grad is not None
    assert torch.all(torch.isfinite(probabilities.grad))
    assert torch.linalg.vector_norm(probabilities.grad) > 1e-12
    assert gates.grad is None
    assert not result.denominator.requires_grad


def test_float32_quantization_precedes_the_single_half_threshold() -> None:
    response = np.array([0.5 - 1e-10, 0.49, 0.7, 0.7, 0.1], dtype=np.float64)
    decoded = decode_identity(
        response,
        np.ones(5, dtype=np.uint8),
        np.array([[0, 5]], dtype=np.int32),
    )
    assert decoded.response[0] == np.float32(0.5)
    assert decoded.component_bounds.tolist() == [[0, 1], [2, 4]]
    assert decoded.component_location.tolist() == [0, 2]
    assert decoded.component_score.tolist() == [0.5, np.float32(0.7)]
    assert decoded.count == 2


def test_components_and_adjacency_reset_at_run_boundaries() -> None:
    decoded = decode_identity(
        np.array([0.1, 0.6, 0.8, 0.9, 0.7, 0.6, 0.1], dtype=np.float64),
        np.array([1, 1, 1, 0, 1, 1, 1], dtype=np.uint8),
        np.array([[0, 3], [4, 7]], dtype=np.int32),
    )
    assert decoded.component_bounds.tolist() == [[1, 3], [4, 6]]
    assert decoded.component_location.tolist() == [2, 4]
    assert decoded.count == 2


def test_abstaining_stub_still_invokes_decoder_once(monkeypatch: pytest.MonkeyPatch) -> None:
    original = decode_module.connected_components
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> decode_module.ComponentTable:
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(decode_module, "connected_components", counted)
    decoded = decode_module.decode_identity(None, None, None, abstain_reasons=[8, 20])
    assert calls == 1
    assert decoded.decoder_invocations == 1
    assert decoded.abstain
    assert decoded.count == -1
    assert decoded.response.shape == (0,)
    assert decoded.run_bounds.shape == (0, 2)
    assert decoded.component_bounds.shape == (0, 2)


def test_decoder_rejects_unsorted_reasons_and_adjacent_run_rows() -> None:
    with pytest.raises(DecodeError, match="ascending"):
        decode_identity(None, None, None, abstain_reasons=[20, 8])
    with pytest.raises(DecodeError, match="nonadjacent"):
        decode_identity(
            np.zeros(4, dtype=np.float64),
            np.zeros(4, dtype=np.uint8),
            np.array([[0, 2], [2, 4]], dtype=np.int32),
        )
