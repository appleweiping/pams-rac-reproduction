from __future__ import annotations

import math

import pytest
import torch

from pams.warp_phase.decode import (
    decode_track,
    positive_interval_weights,
    round_half_even,
    window_starts,
)


def _phase(length: int, *, origin: int = 0, increment: float = 0.2) -> torch.Tensor:
    angle = (torch.arange(length, dtype=torch.float64) + origin) * increment
    return torch.stack((torch.cos(angle), torch.sin(angle)), dim=-1)


def test_positive_nola_weights_and_final_window_rule() -> None:
    weights = positive_interval_weights()

    assert weights.dtype == torch.float64
    assert weights.shape == (63,)
    assert torch.isfinite(weights).all()
    assert float(weights.min()) >= 1e-3
    assert window_starts(63) == (0,)
    assert window_starts(64) == (0,)
    assert window_starts(120) == (0, 32, 56)
    assert window_starts(128) == (0, 32, 64)


@pytest.mark.parametrize("origin", [0, 8, 16, 24])
def test_one_track_decode_conserves_mass_across_grid_origins(origin: int) -> None:
    phase = _phase(120, origin=origin)
    present = torch.ones(119, dtype=torch.bool)
    gate = torch.ones(119, dtype=torch.bool)

    decoded = decode_track(phase, present, gate)
    expected = 119 * 0.2 / (2.0 * math.pi)

    assert decoded.continuous == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert decoded.interval_mass.sum().item() == pytest.approx(expected, rel=1e-12)
    assert torch.all(decoded.denominator >= 1e-3)
    assert decoded.starts == (0, 32, 56)


def test_gate_is_exact_and_padding_support_never_creates_mass() -> None:
    phase = _phase(70)
    present = torch.ones(69, dtype=torch.bool)
    present[-2:] = False
    gate = present.clone()
    gate[10] = False

    decoded = decode_track(phase, present, gate)
    expected_intervals = int(gate.sum().item())
    expected = expected_intervals * 0.2 / (2.0 * math.pi)

    assert decoded.continuous == pytest.approx(expected, rel=1e-12)
    assert decoded.interval_mass[10].item() == 0.0
    assert decoded.interval_mass[-2:].tolist() == [0.0, 0.0]
    assert decoded.denominator[-2:].tolist() == [0.0, 0.0]


def test_half_even_rounding_is_secondary_after_the_continuous_sum() -> None:
    assert round_half_even(2.5) == 2
    assert round_half_even(3.5) == 4

    increment = math.pi / 2.0
    decoded = decode_track(
        _phase(11, increment=increment),
        torch.ones(10, dtype=torch.bool),
        torch.ones(10, dtype=torch.bool),
    )
    assert decoded.continuous == pytest.approx(2.5)
    assert decoded.rounded_half_even == 2


def test_decode_shape_dtype_and_gate_contract_fails_closed() -> None:
    phase = _phase(5)
    present = torch.ones(4, dtype=torch.bool)
    gate = present.clone()
    gate[0] = False
    bad_present = present.clone()
    bad_present[0] = False

    with pytest.raises(ValueError, match="cannot enable"):
        decode_track(phase, bad_present, present)
    with pytest.raises(TypeError, match="torch.bool"):
        decode_track(phase, present.to(torch.uint8), gate)
    with pytest.raises(ValueError, match="shape"):
        decode_track(phase, present[:-1], gate)
