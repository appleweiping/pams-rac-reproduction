"""Cached causal TempoRAC response graph and fixed shortcut adapters."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as torch_functional

ShortcutName = Literal[
    "no-pose-timestamp",
    "nuisance-only",
    "pose-shuffle",
    "static-code-only",
    "warp-metadata",
    "track-length",
]
SHORTCUT_DIMENSIONS: dict[str, int] = {
    "no-pose-timestamp": 20,
    "nuisance-only": 51,
    "pose-shuffle": 269,
    "static-code-only": 32,
    "warp-metadata": 2,
    "track-length": 1,
}


class ResponseContractError(ValueError):
    """Raised when a response or shortcut input violates v4."""


class CausalResidualBlock(nn.Module):
    """LayerNorm-depthwise-GELU-pointwise-GELU causal residual block."""

    def __init__(self, width: int, dilation: int, *, kernel_size: int = 5) -> None:
        super().__init__()
        if width <= 0 or dilation <= 0 or kernel_size <= 0:
            raise ResponseContractError("causal block dimensions must be positive")
        self.width = width
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.normalization = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(
            width,
            width,
            kernel_size,
            dilation=dilation,
            groups=width,
            bias=True,
        )
        self.pointwise = nn.Conv1d(width, width, 1, bias=True)

    @property
    def left_context(self) -> int:
        return self.dilation * (self.kernel_size - 1)

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 3 or value.shape[-1] != self.width:
            raise ResponseContractError("causal block expects [batch,time,width]")
        residual = value
        hidden = self.normalization(value).transpose(1, 2)
        hidden = torch_functional.pad(hidden, (self.left_context, 0))
        hidden = torch_functional.gelu(self.depthwise(hidden))
        hidden = torch_functional.gelu(self.pointwise(hidden)).transpose(1, 2)
        return residual + hidden


class ResponseEncoder(nn.Module):
    """The shared 269->64 encoder with dilations 1 and 2."""

    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(269, 64)
        self.blocks = nn.ModuleList((CausalResidualBlock(64, 1), CausalResidualBlock(64, 2)))

    def forward(self, value: Tensor) -> Tensor:
        if value.ndim != 3 or value.shape[-1] != 269:
            raise ResponseContractError("response encoder expects [batch,time,269]")
        hidden = self.input(value)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden


class ExpertBranch(nn.Module):
    """The single branch class instantiated at fixed dilations 4, 2, and 1."""

    def __init__(self, dilation: int) -> None:
        super().__init__()
        if dilation not in (1, 2, 4):
            raise ResponseContractError("expert dilation must be one of 1,2,4")
        self.dilation = dilation
        self.blocks = nn.ModuleList(
            (CausalResidualBlock(64, dilation), CausalResidualBlock(64, dilation))
        )
        self.normalization = nn.LayerNorm(64)
        self.output = nn.Linear(64, 1)

    @property
    def receptive_field(self) -> int:
        return 1 + sum(cast(CausalResidualBlock, block).left_context for block in self.blocks)

    def forward_logits(self, encoded: Tensor) -> Tensor:
        hidden = encoded
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.normalization(hidden)).squeeze(-1)

    def forward(self, encoded: Tensor) -> Tensor:
        return torch.sigmoid(self.forward_logits(encoded))


@dataclass(frozen=True, slots=True)
class CachedResponses:
    sample_logits: Tensor
    edge_logits: Tensor
    edge_probabilities: Tensor


class TempoRACResponse(nn.Module):
    """Shared encoder and three equal-shape cached causal branches."""

    branch_order = ("slow", "medium", "fast")

    def __init__(self) -> None:
        super().__init__()
        self.encoder = ResponseEncoder()
        self.experts = nn.ModuleDict(
            {
                "slow": ExpertBranch(4),
                "medium": ExpertBranch(2),
                "fast": ExpertBranch(1),
            }
        )

    def _run_logits(self, value: Tensor) -> Tensor:
        encoded = self.encoder(value)
        return torch.stack(
            [
                cast(ExpertBranch, self.experts[name]).forward_logits(encoded)
                for name in self.branch_order
            ],
            dim=-1,
        )

    def cached_responses(
        self,
        value: Tensor,
        run_bounds: Sequence[Sequence[int]] | Tensor | None = None,
    ) -> CachedResponses:
        """Execute each edge run once and expose its sample-``t+1`` logits."""

        squeeze = value.ndim == 2
        batch_value = value.unsqueeze(0) if squeeze else value
        if batch_value.ndim != 3 or batch_value.shape[-1] != 269:
            raise ResponseContractError("response input must be [T,269] or [B,T,269]")
        if batch_value.dtype != torch.float32:
            raise ResponseContractError("response input must have dtype float32")
        if not bool(torch.all(torch.isfinite(batch_value)).item()):
            raise ResponseContractError("response input must be finite")
        batch, sample_count, _ = batch_value.shape
        if sample_count < 2:
            raise ResponseContractError("response input needs at least two samples")
        edge_count = sample_count - 1
        bounds: tuple[tuple[int, int], ...]
        if run_bounds is None:
            bounds = ((0, edge_count),)
        else:
            try:
                bound_array = torch.as_tensor(run_bounds)
            except (TypeError, ValueError) as exc:
                raise ResponseContractError("run bounds must be integer shape [R,2]") from exc
            if (
                bound_array.ndim != 2
                or bound_array.shape[1] != 2
                or bound_array.dtype
                not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
            ):
                raise ResponseContractError("run bounds must be integer shape [R,2]")
            bounds = tuple((int(row[0]), int(row[1])) for row in bound_array.tolist())
        sample_logits = batch_value.new_zeros((batch, sample_count, 3))
        previous_stop = -1
        for start, stop in bounds:
            if not 0 <= start < stop <= edge_count:
                raise ResponseContractError("edge run is empty or out of range")
            if start <= previous_stop:
                raise ResponseContractError("edge runs must be ordered and nonadjacent")
            sample_logits[:, start : stop + 1] = self._run_logits(batch_value[:, start : stop + 1])
            previous_stop = stop
        edge_logits = sample_logits[:, 1:, :]
        probabilities = torch.sigmoid(edge_logits)
        if squeeze:
            sample_logits = sample_logits.squeeze(0)
            edge_logits = edge_logits.squeeze(0)
            probabilities = probabilities.squeeze(0)
        return CachedResponses(sample_logits, edge_logits, probabilities)

    def forward(self, value: Tensor) -> Tensor:
        return self.cached_responses(value).edge_probabilities

    def branch_parameter_shapes(self) -> tuple[tuple[tuple[int, ...], ...], ...]:
        return tuple(
            tuple(tuple(parameter.shape) for parameter in self.experts[name].parameters())
            for name in self.branch_order
        )


class CapacityControl(TempoRACResponse):
    """Separately trained equal-capacity graph with gate-free probability mean."""

    def fused_response(
        self,
        value: Tensor,
        run_bounds: Sequence[Sequence[int]] | Tensor | None = None,
    ) -> Tensor:
        return torch.mean(self.cached_responses(value, run_bounds).edge_probabilities, dim=-1)


def _projection_digest(name: str, row: int, column_block: int) -> bytes:
    encoded = name.encode("ascii")
    return hashlib.sha256(
        b"temporac.shortcut-projection.v4\0"
        + struct.pack(">H", len(encoded))
        + encoded
        + struct.pack(">II", row, column_block)
    ).digest()


@lru_cache(maxsize=6)
def shortcut_projection(name: ShortcutName) -> NDArray[np.float32]:
    """Generate the exact fixed 269xd shortcut projection."""

    if name not in SHORTCUT_DIMENSIONS:
        raise ResponseContractError(f"unknown shortcut {name!r}")
    dimension = SHORTCUT_DIMENSIONS[name]
    result = np.empty((269, dimension), dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(dimension))
    for row in range(269):
        digests: dict[int, bytes] = {}
        for column in range(dimension):
            block = column // 256
            digest = digests.setdefault(block, _projection_digest(name, row, block))
            bit_index = column % 256
            byte = digest[bit_index // 8]
            bit = (byte >> (7 - bit_index % 8)) & 1
            result[row, column] = np.float32((1.0 if bit else -1.0) * scale)
    result.setflags(write=False)
    return result


class ShortcutAdapter(nn.Module):
    """Parameter-free projection from one frozen shortcut vector to width 269."""

    projection: Tensor

    def __init__(self, name: ShortcutName) -> None:
        super().__init__()
        if name not in SHORTCUT_DIMENSIONS:
            raise ResponseContractError(f"unknown shortcut {name!r}")
        self.name = name
        self.raw_dimension = SHORTCUT_DIMENSIONS[name]
        self.register_buffer(
            "projection",
            torch.from_numpy(np.array(shortcut_projection(name), copy=True)),
            persistent=True,
        )

    def forward(self, raw: Tensor) -> Tensor:
        if raw.shape[-1] != self.raw_dimension:
            raise ResponseContractError(
                f"{self.name} raw input must have width {self.raw_dimension}"
            )
        rounded = raw.to(dtype=torch.float32)
        return torch_functional.linear(rounded, self.projection)


class ShortcutResponse(nn.Module):
    """Same-capacity response graph fed only through a fixed shortcut adapter."""

    def __init__(self, name: ShortcutName) -> None:
        super().__init__()
        self.adapter = ShortcutAdapter(name)
        self.response = TempoRACResponse()

    def forward(
        self,
        raw: Tensor,
        run_bounds: Sequence[Sequence[int]] | Tensor | None = None,
    ) -> CachedResponses:
        return self.response.cached_responses(self.adapter(raw), run_bounds)


def _ordered_hash_indices(domain: bytes, source_key: bytes, run: int, count: int) -> list[int]:
    if len(source_key) != 32:
        raise ResponseContractError("source key must contain exactly 32 bytes")
    return sorted(
        range(count),
        key=lambda index: (
            hashlib.sha256(domain + b"\0" + source_key + struct.pack(">II", run, index)).digest(),
            index,
        ),
    )


def pose_shuffle_raw(
    response_input: NDArray[np.float32],
    source_key: bytes,
    run_bounds: Sequence[Sequence[int]],
) -> NDArray[np.float32]:
    """Apply the exact within-run sample/channel/sign pose-shuffle transform."""

    values = np.asarray(response_input, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 269:
        raise ResponseContractError("pose shuffle expects [T,269]")
    output = np.zeros_like(values)
    previous_stop = 0
    for run, pair in enumerate(run_bounds):
        if len(pair) != 2:
            raise ResponseContractError("run bound must contain two integers")
        start, stop = int(pair[0]), int(pair[1])
        if not 0 <= start < stop <= values.shape[0] or start < previous_stop:
            raise ResponseContractError("invalid pose-shuffle run bounds")
        sample_order_local = _ordered_hash_indices(
            b"temporac.pose-shuffle.v4", source_key, run, stop - start
        )
        sample_order = np.asarray([start + index for index in sample_order_local])
        channel_order = _ordered_hash_indices(
            b"temporac.pose-channel-order.v4", source_key, run, 269
        )
        signs = np.empty(269, dtype=np.float32)
        for channel in range(269):
            digest = hashlib.sha256(
                b"temporac.pose-channel-sign.v4\0" + source_key + struct.pack(">II", run, channel)
            ).digest()
            signs[channel] = np.float32(1.0 if digest[0] & 0x80 else -1.0)
        output[start:stop] = values[sample_order][:, channel_order] * signs[None, :]
        previous_stop = stop
    return np.asarray(output, dtype="<f4", order="C")


def no_pose_timestamp_vector(
    joint_mask: NDArray[np.bool_],
    frame_valid: NDArray[np.bool_],
    run_relative_clock: NDArray[np.float32],
    delta_q: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Build the exact 20-channel no-pose timestamp vector."""

    mask = np.asarray(joint_mask, dtype=np.float32)
    frame = np.asarray(frame_valid, dtype=np.float32)
    clock = np.asarray(run_relative_clock, dtype=np.float32)
    delta = np.asarray(delta_q, dtype=np.float32)
    if mask.ndim != 2 or mask.shape[1] != 17:
        raise ResponseContractError("joint_mask must have shape [T,17]")
    count = mask.shape[0]
    if frame.shape != (count,) or clock.shape != (count,) or delta.shape != (count,):
        raise ResponseContractError("no-pose timestamp scalar shapes mismatch")
    return np.concatenate(
        (mask, frame[:, None], clock[:, None], np.clip(delta[:, None] / 4.0, 0.0, 1.0)),
        axis=1,
    ).astype(np.float32, copy=False)


__all__ = [
    "CachedResponses",
    "CapacityControl",
    "CausalResidualBlock",
    "ExpertBranch",
    "ResponseContractError",
    "ResponseEncoder",
    "SHORTCUT_DIMENSIONS",
    "ShortcutAdapter",
    "ShortcutName",
    "ShortcutResponse",
    "TempoRACResponse",
    "no_pose_timestamp_vector",
    "pose_shuffle_raw",
    "shortcut_projection",
]
