"""Probability fusion and exact strictly-positive TempoRAC NOLA."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

Float64Array = NDArray[np.float64]


class NOLAError(ValueError):
    """Raised when exact positive reconstruction cannot be established."""


@runtime_checkable
class EdgeWindow(Protocol):
    edge_start: int
    edge_stop: int


@dataclass(frozen=True, slots=True)
class NOLAResult:
    numerator: Float64Array
    denominator: Float64Array
    response: Float64Array
    window_response: tuple[Float64Array, ...]

    def __post_init__(self) -> None:
        self.numerator.setflags(write=False)
        self.denominator.setflags(write=False)
        self.response.setflags(write=False)
        for value in self.window_response:
            value.setflags(write=False)


@dataclass(frozen=True, slots=True)
class TorchNOLAResult:
    numerator: Tensor
    denominator: Tensor
    response: Tensor
    window_response: tuple[Tensor, ...]


def _window_interval(window: EdgeWindow | Sequence[int]) -> tuple[int, int]:
    if isinstance(window, EdgeWindow):
        start, stop = int(window.edge_start), int(window.edge_stop)
    else:
        if len(window) != 2:
            raise NOLAError("window tuples must contain (edge_start, edge_stop)")
        start, stop = int(window[0]), int(window[1])
    if not (0 <= start < stop):
        raise NOLAError("windows must be nonempty half-open edge intervals")
    return start, stop


def _validated_windows(
    windows: Sequence[EdgeWindow | Sequence[int]], edge_count: int
) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    previous_start = -1
    for window in windows:
        start, stop = _window_interval(window)
        if stop > edge_count:
            raise NOLAError("window exceeds the response edge sequence")
        if start < previous_start:
            raise NOLAError("NOLA windows must follow window-start order")
        previous_start = start
        result.append((start, stop))
    return tuple(result)


def _binary_edge_mask(mask: ArrayLike, edge_count: int) -> NDArray[np.bool_]:
    raw = np.asarray(mask)
    if raw.shape != (edge_count,) or raw.dtype.kind not in "bu":
        raise NOLAError("edge_mask must be binary shape [E]")
    if not np.all((raw == 0) | (raw == 1)):
        raise NOLAError("edge_mask must be binary")
    return np.asarray(raw, dtype=np.bool_)


def window_taper(edge_slots: int) -> Float64Array:
    """Return the exact strictly-positive F13 taper for local edge slots."""

    if not (1 <= edge_slots <= 127):
        raise NOLAError("a local window must contain between 1 and 127 edge slots")
    slot = np.arange(edge_slots, dtype=np.float64)
    taper = np.float64(1e-3) + np.float64(1.0 - 1e-3) * np.sin(
        np.pi * (slot + np.float64(0.5)) / np.float64(127.0)
    ) ** np.float64(2.0)
    if not np.all(np.isfinite(taper)) or not np.all(taper >= np.float64(1e-3)):
        raise NOLAError("NOLA taper is not strictly positive")
    return np.ascontiguousarray(taper)


def _validate_numpy_probabilities(
    branch_probabilities: ArrayLike, gates: ArrayLike
) -> tuple[Float64Array, Float64Array]:
    probabilities = np.asarray(branch_probabilities, dtype=np.float64)
    gate_array = np.asarray(gates, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise NOLAError("branch_probabilities must have shape [E,3]")
    if gate_array.ndim != 2 or gate_array.shape[1] != 3:
        raise NOLAError("gates must have shape [W,3]")
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise NOLAError("branch probabilities must be finite values in [0,1]")
    if not np.all(np.isfinite(gate_array)) or np.any(gate_array < 0.0):
        raise NOLAError("gates must be finite and nonnegative")
    if not np.allclose(np.sum(gate_array, axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise NOLAError("each routing gate must sum to one")
    return probabilities, gate_array


def probability_fusion(branch_probabilities: ArrayLike, gate: ArrayLike) -> Float64Array:
    """Fuse sigmoid probabilities (never logits) with one three-way gate."""

    probabilities = np.asarray(branch_probabilities, dtype=np.float64)
    gate_array = np.asarray(gate, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise NOLAError("branch_probabilities must have shape [N,3]")
    if gate_array.shape != (3,) or not np.all(np.isfinite(gate_array)):
        raise NOLAError("gate must be finite shape [3]")
    if (
        np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or not np.all(np.isfinite(probabilities))
    ):
        raise NOLAError("branch probabilities must be finite values in [0,1]")
    if np.any(gate_array < 0.0) or not np.isclose(np.sum(gate_array), 1.0, rtol=0.0, atol=1e-12):
        raise NOLAError("gate must be nonnegative and sum to one")
    return np.ascontiguousarray(probabilities @ gate_array, dtype=np.float64)


def fuse_window_probabilities(
    branch_probabilities: ArrayLike,
    gates: ArrayLike,
    windows: Sequence[EdgeWindow | Sequence[int]],
) -> tuple[Float64Array, ...]:
    """Slice cached run-level probabilities and fuse every window once."""

    probabilities, gate_array = _validate_numpy_probabilities(branch_probabilities, gates)
    intervals = _validated_windows(windows, probabilities.shape[0])
    if gate_array.shape[0] != len(intervals):
        raise NOLAError("gate row count must equal window count")
    return tuple(
        probability_fusion(probabilities[start:stop], gate_array[index])
        for index, (start, stop) in enumerate(intervals)
    )


def _neumaier_add_array(total: Float64Array, correction: Float64Array, value: Float64Array) -> None:
    candidate = total + value
    use_total = np.abs(total) >= np.abs(value)
    correction += np.where(
        use_total,
        (total - candidate) + value,
        (value - candidate) + total,
    )
    total[:] = candidate


def nola_reconstruct(
    branch_probabilities: ArrayLike,
    gates: ArrayLike,
    windows: Sequence[EdgeWindow | Sequence[int]],
    edge_mask: ArrayLike,
) -> NOLAResult:
    """Fuse F12 and reconstruct F13 with no epsilon, clamp, or fallback."""

    probabilities, gate_array = _validate_numpy_probabilities(branch_probabilities, gates)
    edge_count = probabilities.shape[0]
    intervals = _validated_windows(windows, edge_count)
    if gate_array.shape[0] != len(intervals):
        raise NOLAError("gate row count must equal window count")
    valid = _binary_edge_mask(edge_mask, edge_count)
    numerator = np.zeros(edge_count, dtype=np.float64)
    numerator_correction = np.zeros(edge_count, dtype=np.float64)
    denominator = np.zeros(edge_count, dtype=np.float64)
    denominator_correction = np.zeros(edge_count, dtype=np.float64)
    fused_windows: list[Float64Array] = []

    for window_index, (start, stop) in enumerate(intervals):
        fused = probability_fusion(probabilities[start:stop], gate_array[window_index])
        fused_windows.append(fused)
        taper = window_taper(stop - start)
        window_valid = valid[start:stop].astype(np.float64)
        numerator_value = taper * window_valid * fused
        denominator_value = taper * window_valid
        _neumaier_add_array(
            numerator[start:stop], numerator_correction[start:stop], numerator_value
        )
        _neumaier_add_array(
            denominator[start:stop], denominator_correction[start:stop], denominator_value
        )

    numerator += numerator_correction
    denominator += denominator_correction
    if not np.all(np.isfinite(numerator)) or not np.all(np.isfinite(denominator)):
        raise NOLAError("NOLA numerator or denominator is nonfinite")
    if np.any(denominator[valid] < np.float64(1e-3)):
        raise NOLAError("every valid edge requires exact NOLA denominator >= 1e-3")
    if np.any(denominator[~valid] != 0.0) or np.any(numerator[~valid] != 0.0):
        raise NOLAError("masked NOLA edges must remain exact zero")
    response = np.zeros(edge_count, dtype=np.float64)
    # This division is executed only after the strictly-positive proof.  There
    # is intentionally no epsilon in the denominator.
    response[valid] = numerator[valid] / denominator[valid]
    if (
        not np.all(np.isfinite(response))
        or np.any(response[valid] < 0.0)
        or np.any(response[valid] > 1.0)
    ):
        raise NOLAError("reconstructed response is not a finite probability")
    return NOLAResult(
        numerator=np.ascontiguousarray(numerator),
        denominator=np.ascontiguousarray(denominator),
        response=np.ascontiguousarray(response),
        window_response=tuple(fused_windows),
    )


def _torch_neumaier_add(total: Tensor, correction: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
    candidate = total + value
    updated_correction = correction + torch.where(
        torch.abs(total) >= torch.abs(value),
        (total - candidate) + value,
        (value - candidate) + total,
    )
    return candidate, updated_correction


def nola_reconstruct_torch(
    branch_probabilities: Tensor,
    gates: Tensor | ArrayLike,
    windows: Sequence[EdgeWindow | Sequence[int]],
    edge_mask: Tensor | ArrayLike,
) -> TorchNOLAResult:
    """Differentiable F12/F13; only cue/gate/taper/mask/denominator detach."""

    if branch_probabilities.ndim != 2 or branch_probabilities.shape[1] != 3:
        raise NOLAError("branch_probabilities must have shape [E,3]")
    edge_count = int(branch_probabilities.shape[0])
    intervals = _validated_windows(windows, edge_count)
    device = branch_probabilities.device
    probabilities = branch_probabilities.to(dtype=torch.float64)
    if not bool(torch.all(torch.isfinite(probabilities)).item()) or not bool(
        torch.all((probabilities >= 0.0) & (probabilities <= 1.0)).item()
    ):
        raise NOLAError("branch probabilities must be finite values in [0,1]")
    gate_tensor = torch.as_tensor(gates, dtype=torch.float64, device=device).detach()
    if gate_tensor.shape != (len(intervals), 3):
        raise NOLAError("gates must have shape [W,3]")
    if not bool(torch.all(torch.isfinite(gate_tensor)).item()) or not bool(
        torch.all(gate_tensor >= 0.0).item()
    ):
        raise NOLAError("gates must be finite and nonnegative")
    if not bool(torch.all(torch.abs(torch.sum(gate_tensor, dim=1) - 1.0) <= 1e-12).item()):
        raise NOLAError("each routing gate must sum to one")
    mask_tensor = torch.as_tensor(edge_mask, device=device)
    if mask_tensor.shape != (edge_count,):
        raise NOLAError("edge_mask must have shape [E]")
    if not bool(torch.all((mask_tensor == 0) | (mask_tensor == 1)).item()):
        raise NOLAError("edge_mask must be binary")
    valid = mask_tensor.to(dtype=torch.bool).detach()

    numerator_terms: list[list[Tensor]] = [[] for _ in range(edge_count)]
    denominator_terms: list[list[Tensor]] = [[] for _ in range(edge_count)]
    fused_windows: list[Tensor] = []
    for window_index, (start, stop) in enumerate(intervals):
        fused = probabilities[start:stop] @ gate_tensor[window_index]
        fused_windows.append(fused)
        taper = torch.as_tensor(window_taper(stop - start), device=device).detach()
        window_valid = valid[start:stop].to(dtype=torch.float64)
        for local_index, edge in enumerate(range(start, stop)):
            numerator_terms[edge].append(
                taper[local_index] * window_valid[local_index] * fused[local_index]
            )
            denominator_terms[edge].append(taper[local_index] * window_valid[local_index])

    numerator_values: list[Tensor] = []
    denominator_values: list[Tensor] = []
    zero = torch.zeros((), dtype=torch.float64, device=device)
    for edge in range(edge_count):
        numerator_total = zero
        numerator_correction = zero
        for value in numerator_terms[edge]:
            numerator_total, numerator_correction = _torch_neumaier_add(
                numerator_total, numerator_correction, value
            )
        denominator_total = zero
        denominator_correction = zero
        for value in denominator_terms[edge]:
            denominator_total, denominator_correction = _torch_neumaier_add(
                denominator_total, denominator_correction, value
            )
        numerator_values.append(numerator_total + numerator_correction)
        denominator_values.append((denominator_total + denominator_correction).detach())
    numerator = torch.stack(numerator_values)
    denominator = torch.stack(denominator_values)
    if not bool(torch.all(torch.isfinite(numerator)).item()) or not bool(
        torch.all(torch.isfinite(denominator)).item()
    ):
        raise NOLAError("NOLA numerator or denominator is nonfinite")
    if bool(torch.any(denominator[valid] < 1e-3).item()):
        raise NOLAError("every valid edge requires exact NOLA denominator >= 1e-3")
    if bool(torch.any(denominator[~valid] != 0.0).item()) or bool(
        torch.any(numerator[~valid] != 0.0).item()
    ):
        raise NOLAError("masked NOLA edges must remain exact zero")
    response = torch.zeros_like(numerator)
    # Boolean indexed assignment would hide the exact division in an in-place
    # op.  where keeps the autograd path while substituting one only on masked
    # edges that are immediately reset to zero.
    safe_denominator = torch.where(valid, denominator, torch.ones_like(denominator))
    response = torch.where(valid, numerator / safe_denominator, torch.zeros_like(numerator))
    if not bool(torch.all(torch.isfinite(response)).item()):
        raise NOLAError("reconstructed response is nonfinite")
    return TorchNOLAResult(
        numerator=numerator,
        denominator=denominator,
        response=response,
        window_response=tuple(fused_windows),
    )


__all__ = [
    "NOLAError",
    "NOLAResult",
    "TorchNOLAResult",
    "fuse_window_probabilities",
    "nola_reconstruct",
    "nola_reconstruct_torch",
    "probability_fusion",
    "window_taper",
]
