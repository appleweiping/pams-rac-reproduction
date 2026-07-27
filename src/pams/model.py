"""Neural modules used by the independent PAMS reproduction.

The paper discloses a four-layer, post-layer-normalized Transformer encoder
and a two-layer scalar period head.  This module keeps those defaults while
allowing small configurations for CPU smoke tests.
"""

from __future__ import annotations

import math

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

    def forward(self, inputs: Tensor) -> Tensor:
        """Return ``inputs`` with positions added.

        Args:
            inputs: A ``[batch, time, dimension]`` tensor.
        """

        if inputs.ndim != 3:
            raise ValueError("positional encoding expects [batch, time, dimension]")
        time = inputs.shape[1]
        if time > self.encoding.shape[0]:
            raise ValueError(
                f"sequence length {time} exceeds positional capacity {self.encoding.shape[0]}"
            )
        positions = self.encoding[:time].to(device=inputs.device, dtype=inputs.dtype)
        return inputs + positions.unsqueeze(0)


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

        self.input_dim = input_dim
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim
        self.norm_first = norm_first

        self.input_projection = nn.Linear(input_dim, model_dim)
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

    def forward(self, inputs: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        """Encode pose frames and L2-normalize every valid embedding."""

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

        # Avoid NaNs for samples in which pose extraction failed on every frame.
        attention_valid = valid.clone()
        fully_invalid = ~attention_valid.any(dim=1)
        if fully_invalid.any() and time:
            attention_valid[fully_invalid, 0] = True

        hidden = self.input_projection(inputs)
        hidden = self.position_encoding(hidden)
        hidden = self.transformer(
            hidden,
            src_key_padding_mask=~attention_valid,
        )
        embeddings = self.output_projection(hidden)
        embeddings = F.normalize(embeddings, p=2, dim=-1, eps=1e-12)
        return embeddings.masked_fill(~valid.unsqueeze(-1), 0.0)


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

    def forward(self, embeddings: Tensor) -> Tensor:
        if embeddings.ndim < 2:
            raise ValueError("period head expects [..., time, embedding] inputs")
        return self.network(embeddings).squeeze(-1)


class PAMSModel(nn.Module):
    """Convenience composition of the encoder and scalar period head."""

    def __init__(
        self,
        encoder: PAMSEncoder | None = None,
        period_head: PeriodHead | None = None,
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
        embeddings = self.encoder(inputs, valid_mask)
        stream = self.period_head(embeddings)
        if valid_mask is not None:
            stream = stream.masked_fill(~valid_mask.to(device=stream.device, dtype=torch.bool), 0.0)
        return embeddings, stream
