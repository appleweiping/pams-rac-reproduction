"""Neural modules used by the independent PAMS reproduction.

The paper discloses a four-layer, post-layer-normalized Transformer encoder
and a two-layer scalar period head.  This module keeps those defaults while
allowing small configurations for CPU smoke tests.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Add deterministic sinusoidal positions to batch-first sequences."""

    encoding: Tensor

    def __init__(self, dimension: int, max_length: int = 4096) -> None:
        super().__init__()
        if dimension < 1:
            raise ValueError("dimension must be positive")
        if max_length < 1:
            raise ValueError("max_length must be positive")

        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        even_dimensions = torch.arange(0, dimension, 2, dtype=torch.float32)
        frequencies = torch.exp(-math.log(10_000.0) * even_dimensions / dimension)
        encoding = torch.zeros(max_length, dimension, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * frequencies)
        if dimension > 1:
            encoding[:, 1::2] = torch.cos(position * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=True)

    def forward(
        self,
        inputs: Tensor,
        *,
        position_indices: Tensor | None = None,
    ) -> Tensor:
        """Return ``inputs`` with positions added.

        Args:
            inputs: A ``[batch, time, dimension]`` tensor.
            position_indices: Optional integer ``[batch, time]`` rows of the
                sinusoidal table.  ``None`` retains the historical canonical
                ``0..time-1`` path exactly.
        """

        if inputs.ndim != 3:
            raise ValueError("positional encoding expects [batch, time, dimension]")
        batch, time, _ = inputs.shape
        if time > self.encoding.shape[0]:
            raise ValueError(
                f"sequence length {time} exceeds positional capacity {self.encoding.shape[0]}"
            )
        if position_indices is None:
            positions = self.encoding[:time].to(
                device=inputs.device,
                dtype=inputs.dtype,
            )
            return inputs + positions.unsqueeze(0)
        if not isinstance(position_indices, Tensor):
            raise TypeError("position_indices must be a tensor")
        if position_indices.shape != (batch, time):
            raise ValueError(
                "position_indices must have shape "
                f"{(batch, time)}, got {tuple(position_indices.shape)}"
            )
        if position_indices.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        }:
            raise TypeError("position_indices must use an integer dtype")
        indices = position_indices.to(
            device=self.encoding.device,
            dtype=torch.long,
        )
        if indices.numel() and (
            bool((indices < 0).any())
            or bool((indices >= self.encoding.shape[0]).any())
        ):
            raise ValueError(
                "position_indices must lie inside the positional encoding capacity"
            )
        positions = self.encoding[indices].to(
            device=inputs.device,
            dtype=inputs.dtype,
        )
        return inputs + positions


class PAMSEncoder(nn.Module):
    """Framewise pose encoder with the architecture disclosed by PAMS.

    ``valid_mask`` follows the package-wide convention that ``True`` means a
    valid frame.  Invalid output rows are exactly zero.  Fully masked examples
    are handled without passing an all-masked row to PyTorch attention, which
    otherwise produces NaNs on some versions of PyTorch.
    """

    def __init__(
        self,
        input_dim: int = 99,
        model_dim: int = 512,
        embedding_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 16,
        feedforward_dim: int = 2048,
        dropout: float = 0.1,
        *,
        max_length: int = 4096,
        norm_first: bool = False,
        input_projection_scale: Literal["none", "sqrt_model_dim"] = "none",
        position_encoding_mode: Literal["sinusoidal", "none"] = "sinusoidal",
    ) -> None:
        super().__init__()
        if input_dim < 1 or model_dim < 1 or embedding_dim < 1:
            raise ValueError("model dimensions must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if input_projection_scale not in {"none", "sqrt_model_dim"}:
            raise ValueError(
                "input_projection_scale must be 'none' or 'sqrt_model_dim'"
            )
        if position_encoding_mode not in {"sinusoidal", "none"}:
            raise ValueError(
                "position_encoding_mode must be 'sinusoidal' or 'none'"
            )

        self.input_dim = input_dim
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim
        self.norm_first = norm_first
        self.input_projection_scale = input_projection_scale
        self.position_encoding_mode = position_encoding_mode

        self.input_projection = nn.Linear(input_dim, model_dim)
        # Keep the historical persistent buffer in both modes so that the
        # checkpoint tensor schema remains structurally compatible.
        self.position_encoding = SinusoidalPositionalEncoding(model_dim, max_length)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_projection: nn.Module
        if embedding_dim == model_dim:
            self.output_projection = nn.Identity()
        else:
            self.output_projection = nn.Linear(model_dim, embedding_dim)

    def _flatten_pose(self, inputs: Tensor) -> Tensor:
        if inputs.ndim == 4:
            inputs = inputs.flatten(start_dim=2)
        if inputs.ndim != 3:
            raise ValueError("encoder expects [batch, time, features] or [batch, time, K, C]")
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"expected input dimension {self.input_dim}, got {inputs.shape[-1]}")
        if not inputs.is_floating_point():
            inputs = inputs.float()
        return inputs

    def _validated_inputs_and_mask(
        self,
        inputs: Tensor,
        valid_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        inputs = self._flatten_pose(inputs)
        batch, time, _ = inputs.shape
        if valid_mask is None:
            valid = torch.ones((batch, time), dtype=torch.bool, device=inputs.device)
        else:
            if valid_mask.shape != (batch, time):
                raise ValueError(
                    f"valid_mask must have shape {(batch, time)}, got {tuple(valid_mask.shape)}"
                )
            valid = valid_mask.to(device=inputs.device, dtype=torch.bool)
        return inputs, valid

    def _project_pre_pe(self, inputs: Tensor) -> Tensor:
        projected = self.input_projection(inputs)
        if self.input_projection_scale == "sqrt_model_dim":
            projected = projected * math.sqrt(self.model_dim)
        return projected

    def _forward_with_pre_pe(
        self,
        inputs: Tensor,
        valid_mask: Tensor | None,
        *,
        position_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        inputs, valid = self._validated_inputs_and_mask(inputs, valid_mask)
        _, time, _ = inputs.shape

        # Avoid NaNs for samples in which pose extraction failed on every frame.
        attention_valid = valid.clone()
        fully_invalid = ~attention_valid.any(dim=1)
        if fully_invalid.any() and time:
            attention_valid[fully_invalid, 0] = True

        projected = self._project_pre_pe(inputs)
        if self.position_encoding_mode == "sinusoidal":
            hidden = self.position_encoding(
                projected,
                position_indices=position_indices,
            )
        else:
            # ``position_indices`` intentionally has no effect when absolute
            # positional encoding is disabled.  This gives the target-free
            # canonical/permuted diagnostic an exact nuisance-invariance
            # control while retaining the historical state-dict schema.
            hidden = projected
        hidden = self.transformer(
            hidden,
            src_key_padding_mask=~attention_valid,
        )
        embeddings = self.output_projection(hidden)
        embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-12)
        invalid = ~valid.unsqueeze(-1)
        return (
            embeddings.masked_fill(invalid, 0.0),
            projected.masked_fill(invalid, 0.0),
        )

    def forward(
        self,
        inputs: Tensor,
        valid_mask: Tensor | None = None,
        *,
        position_indices: Tensor | None = None,
    ) -> Tensor:
        """Encode pose frames and L2-normalize every valid embedding."""

        embeddings, _ = self._forward_with_pre_pe(
            inputs,
            valid_mask,
            position_indices=position_indices,
        )
        return embeddings

    def forward_with_pre_pe(
        self,
        inputs: Tensor,
        valid_mask: Tensor | None = None,
        *,
        position_indices: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return embeddings and the projected features immediately before PE.

        This opt-in interface reuses the exact forward computation.  Calling
        :meth:`forward` remains numerically unchanged, while inferred
        diagnostics can inspect pose-content projection features without
        positional encoding or Transformer context.  Invalid projected rows
        are exact zero and must still be accompanied by ``valid_mask`` when
        used for temporal differences.
        """

        return self._forward_with_pre_pe(
            inputs,
            valid_mask,
            position_indices=position_indices,
        )


class PeriodHead(nn.Module):
    """The disclosed ``embedding -> 128 -> scalar`` GELU period head."""

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 128) -> None:
        super().__init__()
        if embedding_dim < 1 or hidden_dim < 1:
            raise ValueError("period-head dimensions must be positive")
        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        embeddings: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        del valid_mask
        if embeddings.ndim < 2:
            raise ValueError("period head expects [..., time, embedding] inputs")
        return self.network(embeddings).squeeze(-1)


class TemporalPeriodHead(PeriodHead):
    """Inferred local-context repair of the disclosed scalar period head.

    A same-padded depthwise five-frame convolution adds local context before
    the unchanged disclosed ``embedding -> 128 -> scalar`` framewise core.
    Its bias-free residual branch is zero-initialized, making the initial
    valid-frame output exactly equal to the pointwise head.  Invalid inputs
    are zeroed before the convolution and invalid outputs afterwards, so
    missing or padded frames cannot inject values into neighboring valid
    predictions.
    """

    temporal_kernel_size = 5

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 128) -> None:
        super().__init__(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
        self.embedding_dim = embedding_dim
        self.temporal_conv = nn.Conv1d(
            embedding_dim,
            embedding_dim,
            kernel_size=self.temporal_kernel_size,
            padding=self.temporal_kernel_size // 2,
            groups=embedding_dim,
            bias=False,
        )
        nn.init.zeros_(self.temporal_conv.weight)

    def forward(
        self,
        embeddings: Tensor,
        *,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        if embeddings.ndim != 3:
            raise ValueError(
                "temporal period head expects [batch, time, embedding] inputs"
            )
        batch, time, dimension = embeddings.shape
        if dimension != self.embedding_dim:
            raise ValueError(
                f"expected embedding dimension {self.embedding_dim}, got {dimension}"
            )
        if valid_mask is None:
            valid = torch.ones(
                (batch, time),
                dtype=torch.bool,
                device=embeddings.device,
            )
        else:
            if valid_mask.shape != (batch, time):
                raise ValueError(
                    "valid_mask must have shape "
                    f"{(batch, time)}, got {tuple(valid_mask.shape)}"
                )
            valid = valid_mask.to(device=embeddings.device, dtype=torch.bool)

        valid_rows = valid.unsqueeze(-1)
        masked_embeddings = embeddings.masked_fill(~valid_rows, 0.0)
        temporal = self.temporal_conv(
            masked_embeddings.transpose(1, 2)
        ).transpose(1, 2)
        contextualized = (masked_embeddings + temporal).masked_fill(
            ~valid_rows,
            0.0,
        )
        stream = super().forward(contextualized)
        return stream.masked_fill(~valid, 0.0)


class PAMSModel(nn.Module):
    """Convenience composition of the encoder and scalar period head."""

    def __init__(
        self,
        encoder: PAMSEncoder | None = None,
        period_head: PeriodHead | TemporalPeriodHead | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else PAMSEncoder()
        self.period_head = (
            period_head
            if period_head is not None
            else PeriodHead(self.encoder.embedding_dim, hidden_dim=128)
        )

    def forward(
        self,
        inputs: Tensor,
        valid_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self.forward_with_head_source(
            inputs,
            valid_mask,
            head_input_source="encoder_embedding",
        )

    def forward_with_head_source(
        self,
        inputs: Tensor,
        valid_mask: Tensor | None = None,
        *,
        head_input_source: Literal[
            "encoder_embedding",
            "projected_pose_pre_pe",
        ],
    ) -> tuple[Tensor, Tensor]:
        """Run the head from an explicitly selected, provenance-bound source.

        ``encoder_embedding`` is the historical behavior.  The pre-PE option
        is an inferred repair that removes the deterministic positional
        template diagnosed in the original SSHead readout; callers must bind
        the choice through :class:`pams.config.SSHeadConfig`.
        """

        if head_input_source == "encoder_embedding":
            embeddings = self.encoder(inputs, valid_mask)
            head_inputs = embeddings
        elif head_input_source == "projected_pose_pre_pe":
            embeddings, head_inputs = self.encoder.forward_with_pre_pe(
                inputs,
                valid_mask,
            )
        else:
            raise ValueError(
                f"unsupported period-head input source: {head_input_source!r}"
            )
        stream = self.period_head(
            head_inputs,
            valid_mask=valid_mask,
        )
        if valid_mask is not None:
            stream = stream.masked_fill(~valid_mask.to(device=stream.device, dtype=torch.bool), 0.0)
        return embeddings, stream
