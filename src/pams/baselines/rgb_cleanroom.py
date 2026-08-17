"""Clean-room neural primitives for RGB repetition-counting baselines.

These modules are architecture-level scaffolds operating on *precomputed frame
features*.  They are not source-parity implementations of RepNet, TransRAC,
Every Shot Counts, or IVAC-P2L, and they are deliberately not wired into the
baseline registry.  A baseline may become registry-ready only after its
source-specific video encoder, preprocessing, supervision, checkpoint
conversion, and numerical parity have been independently validated.

Nothing in this module is a proxy substitution for a named baseline.  The
small configurable dimensions exist solely to make the architectural pieces
trainable and testable on CPU while the parity work remains outstanding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from pams.model import SinusoidalPositionalEncoding


@dataclass(frozen=True, slots=True)
class RepNetStyleOutput:
    """Structured output of :class:`RepNetStyleCounter`."""

    temporal_similarity: Tensor
    period_logits: Tensor
    within_period_logits: Tensor
    within_period_probability: Tensor
    period_length_distribution: Tensor
    density: Tensor
    count: Tensor

    @property
    def period_length_logits(self) -> Tensor:
        """Alias clarifying that ``period_logits`` encode period length."""

        return self.period_logits

    @property
    def periodicity_logits(self) -> Tensor:
        """Alias for the binary within-period classifier logits."""

        return self.within_period_logits

    @property
    def periodicity(self) -> Tensor:
        """Alias for the probability that each frame is within repetition."""

        return self.within_period_probability


@dataclass(frozen=True, slots=True)
class DensityRegressionOutput:
    """A nonnegative temporal density and its integral count."""

    density: Tensor
    count: Tensor
    hidden: Tensor


@dataclass(frozen=True, slots=True)
class ExemplarCountingOutput:
    """Density/count output plus query-to-exemplar attention weights."""

    density: Tensor
    count: Tensor
    cross_attention: Tensor


@dataclass(frozen=True, slots=True)
class PeriodLengthCountingOutput:
    """Period-length distribution and the count obtained from reciprocal rate."""

    period_length_logits: Tensor
    period_length_distribution: Tensor
    periodicity: Tensor
    expected_period: Tensor
    density: Tensor
    count: Tensor


def _validate_features(
    features: Tensor,
    *,
    expected_dim: int | None,
    name: str,
) -> Tensor:
    if not isinstance(features, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if features.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, time, features]")
    batch, time, dimension = features.shape
    if batch < 1 or time < 1 or dimension < 1:
        raise ValueError(f"{name} dimensions must all be non-empty")
    if expected_dim is not None and dimension != expected_dim:
        raise ValueError(f"{name} feature dimension must be {expected_dim}, got {dimension}")
    if not features.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")
    if not bool(torch.isfinite(features).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return features


def _validate_mask(features: Tensor, valid_mask: Tensor | None, *, name: str) -> Tensor:
    expected = features.shape[:2]
    if valid_mask is None:
        return torch.ones(expected, dtype=torch.bool, device=features.device)
    if not isinstance(valid_mask, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if valid_mask.shape != expected:
        raise ValueError(f"{name} must have shape {tuple(expected)}")
    if valid_mask.dtype is not torch.bool:
        raise TypeError(f"{name} must have boolean dtype")
    return valid_mask.to(device=features.device)


def _validate_transformer_dimensions(
    *,
    input_dim: int,
    model_dim: int,
    num_layers: int,
    num_heads: int,
    feedforward_dim: int,
    dropout: float,
) -> None:
    if min(input_dim, model_dim, num_layers, num_heads, feedforward_dim) < 1:
        raise ValueError("all model dimensions and layer counts must be positive")
    if model_dim % num_heads:
        raise ValueError("model_dim must be divisible by num_heads")
    if not 0.0 <= dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")


def _period_tensor(values: Sequence[int]) -> Tensor:
    periods = tuple(values)
    if not periods:
        raise ValueError("period_bins must not be empty")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in periods):
        raise TypeError("period_bins must contain integers")
    if any(value < 1 for value in periods):
        raise ValueError("period_bins must be positive")
    if tuple(sorted(set(periods))) != periods:
        raise ValueError("period_bins must be strictly increasing and unique")
    return torch.tensor(periods, dtype=torch.float32)


def temporal_self_similarity(
    features: Tensor,
    valid_mask: Tensor | None = None,
    *,
    eps: float = 1e-12,
) -> Tensor:
    """Return a cosine temporal self-similarity matrix.

    Args:
        features: Precomputed frame features shaped ``[batch, time, dimension]``.
        valid_mask: Optional boolean ``[batch, time]`` mask; ``True`` is valid.
        eps: Numerical floor used for feature normalization.

    Invalid frame pairs are exactly zero.  This utility does not extract RGB
    features and therefore cannot establish parity with any source baseline.
    """

    _validate_features(features, expected_dim=None, name="features")
    mask = _validate_mask(features, valid_mask, name="valid_mask")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    normalized = F.normalize(features, p=2, dim=-1, eps=eps)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2))
    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    return similarity.masked_fill(~pair_mask, 0.0)


class _TemporalTransformer(nn.Module):
    """Mask-safe batch-first Transformer used by the public scaffolds."""

    def __init__(
        self,
        *,
        input_dim: int,
        model_dim: int,
        num_layers: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        max_length: int,
    ) -> None:
        super().__init__()
        _validate_transformer_dimensions(
            input_dim=input_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        if max_length < 1:
            raise ValueError("max_length must be positive")

        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, model_dim)
        self.position_encoding = SinusoidalPositionalEncoding(model_dim, max_length)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def forward(self, features: Tensor, valid_mask: Tensor) -> Tensor:
        _validate_features(
            features,
            expected_dim=self.input_dim,
            name="transformer features",
        )
        mask = _validate_mask(features, valid_mask, name="valid_mask")
        clean = features.masked_fill(~mask.unsqueeze(-1), 0.0)

        attention_mask = mask.clone()
        fully_invalid = ~attention_mask.any(dim=1)
        attention_mask[fully_invalid, 0] = True

        hidden = self.input_projection(clean)
        hidden = self.position_encoding(hidden)
        hidden = self.transformer(hidden, src_key_padding_mask=~attention_mask)
        return hidden.masked_fill(~mask.unsqueeze(-1), 0.0)


class RepNetStyleCounter(nn.Module):
    """RepNet-style heads over precomputed RGB frame features.

    The scaffold constructs a temporal self-similarity context, predicts a
    periodicity score and candidate period-length ("within-period") classes,
    then integrates ``periodicity / period_length`` into a count.  It omits the
    source model's video encoder and exact training/checkpoint semantics, so it
    must remain unavailable in the baseline registry until parity is proven.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        model_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        period_bins: Sequence[int] = tuple(range(2, 65)),
        similarity_temperature: float = 0.1,
        max_length: int = 4096,
    ) -> None:
        super().__init__()
        _validate_transformer_dimensions(
            input_dim=input_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        if similarity_temperature <= 0.0:
            raise ValueError("similarity_temperature must be positive")

        self.input_dim = input_dim
        self.similarity_temperature = similarity_temperature
        self.period_bins: Tensor
        self.register_buffer("period_bins", _period_tensor(period_bins), persistent=True)
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.GELU(),
        )
        self.similarity_fusion = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.GELU(),
        )
        self.temporal_encoder = _TemporalTransformer(
            input_dim=model_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            max_length=max_length,
        )
        self.period_head = nn.Linear(model_dim, len(self.period_bins))
        self.within_period_head = nn.Linear(model_dim, 1)

    def forward(
        self,
        frame_features: Tensor,
        valid_mask: Tensor | None = None,
    ) -> RepNetStyleOutput:
        _validate_features(
            frame_features,
            expected_dim=self.input_dim,
            name="frame_features",
        )
        mask = _validate_mask(frame_features, valid_mask, name="valid_mask")
        clean = frame_features.masked_fill(~mask.unsqueeze(-1), 0.0)
        similarity = temporal_self_similarity(clean, mask)

        safe_keys = mask.clone()
        safe_keys[~safe_keys.any(dim=1), 0] = True
        attention_logits = similarity / self.similarity_temperature
        attention_logits = attention_logits.masked_fill(
            ~safe_keys.unsqueeze(1),
            torch.finfo(attention_logits.dtype).min,
        )
        similarity_attention = torch.softmax(attention_logits, dim=-1)
        similarity_attention = similarity_attention * mask.unsqueeze(-1)

        projected = self.feature_projection(clean)
        similarity_context = torch.bmm(similarity_attention, projected)
        fused = self.similarity_fusion(torch.cat((projected, similarity_context), dim=-1))
        hidden = self.temporal_encoder(fused, mask)

        period_logits = self.period_head(hidden)
        within_period_logits = self.within_period_head(hidden).squeeze(-1)
        period_distribution = torch.softmax(period_logits, dim=-1)
        within_period_probability = torch.sigmoid(within_period_logits)

        validity = mask.to(dtype=hidden.dtype)
        period_logits = period_logits.masked_fill(~mask.unsqueeze(-1), 0.0)
        within_period_logits = within_period_logits.masked_fill(~mask, 0.0)
        within_period_probability = within_period_probability * validity
        period_distribution = period_distribution * validity.unsqueeze(-1)
        inverse_period = (period_distribution / self.period_bins.to(dtype=hidden.dtype)).sum(dim=-1)
        density = within_period_probability * inverse_period
        count = density.sum(dim=1)
        return RepNetStyleOutput(
            temporal_similarity=similarity,
            period_logits=period_logits,
            within_period_logits=within_period_logits,
            within_period_probability=within_period_probability,
            period_length_distribution=period_distribution,
            density=density,
            count=count,
        )


class TransRACStyleRegressor(nn.Module):
    """Transformer density regressor over externally supplied frame features.

    This is a clean-room architectural primitive, not a TransRAC reproduction:
    the source video backbone, feature sampling, density targets, checkpoint,
    and numerical parity are intentionally outside this module.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        model_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 8,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        max_length: int = 4096,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.temporal_encoder = _TemporalTransformer(
            input_dim=input_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            max_length=max_length,
        )
        self.density_head = nn.Linear(model_dim, 1)

    def forward(
        self,
        frame_features: Tensor,
        valid_mask: Tensor | None = None,
    ) -> DensityRegressionOutput:
        _validate_features(
            frame_features,
            expected_dim=self.input_dim,
            name="frame_features",
        )
        mask = _validate_mask(frame_features, valid_mask, name="valid_mask")
        hidden = self.temporal_encoder(frame_features, mask)
        density = F.softplus(self.density_head(hidden).squeeze(-1))
        density = density * mask.to(dtype=density.dtype)
        return DensityRegressionOutput(
            density=density,
            count=density.sum(dim=1),
            hidden=hidden,
        )


class ESCountsStyleRegressor(nn.Module):
    """Exemplar/query cross-attention density scaffold.

    ``query_features`` and ``exemplar_features`` must already be tokens from
    source-appropriate video encoders.  This class only provides the
    cross-attention and density-regression primitive; it is not an Every Shot
    Counts replacement and requires encoder/checkpoint/protocol parity before
    any registry status may change.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        model_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        max_length: int = 4096,
    ) -> None:
        super().__init__()
        _validate_transformer_dimensions(
            input_dim=input_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )
        self.input_dim = input_dim
        self.query_projection = nn.Linear(input_dim, model_dim)
        self.exemplar_projection = nn.Linear(input_dim, model_dim)
        self.query_positions = SinusoidalPositionalEncoding(model_dim, max_length)
        self.exemplar_positions = SinusoidalPositionalEncoding(model_dim, max_length)
        self.cross_attention = nn.MultiheadAttention(
            model_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * model_dim, model_dim),
            nn.GELU(),
        )
        self.temporal_encoder = _TemporalTransformer(
            input_dim=model_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            max_length=max_length,
        )
        self.density_head = nn.Linear(model_dim, 1)

    def forward(
        self,
        query_features: Tensor,
        exemplar_features: Tensor,
        query_mask: Tensor | None = None,
        exemplar_mask: Tensor | None = None,
    ) -> ExemplarCountingOutput:
        _validate_features(
            query_features,
            expected_dim=self.input_dim,
            name="query_features",
        )
        _validate_features(
            exemplar_features,
            expected_dim=self.input_dim,
            name="exemplar_features",
        )
        if query_features.shape[0] != exemplar_features.shape[0]:
            raise ValueError("query_features and exemplar_features must share batch size")

        query_valid = _validate_mask(query_features, query_mask, name="query_mask")
        exemplar_valid = _validate_mask(
            exemplar_features,
            exemplar_mask,
            name="exemplar_mask",
        )
        clean_query = query_features.masked_fill(~query_valid.unsqueeze(-1), 0.0)
        clean_exemplar = exemplar_features.masked_fill(~exemplar_valid.unsqueeze(-1), 0.0)

        query_hidden = self.query_positions(self.query_projection(clean_query))
        exemplar_hidden = self.exemplar_positions(self.exemplar_projection(clean_exemplar))
        query_hidden = query_hidden.masked_fill(~query_valid.unsqueeze(-1), 0.0)
        exemplar_hidden = exemplar_hidden.masked_fill(~exemplar_valid.unsqueeze(-1), 0.0)

        safe_exemplar = exemplar_valid.clone()
        no_exemplar = ~safe_exemplar.any(dim=1)
        safe_exemplar[no_exemplar, 0] = True
        attended, attention = self.cross_attention(
            query_hidden,
            exemplar_hidden,
            exemplar_hidden,
            key_padding_mask=~safe_exemplar,
            need_weights=True,
            average_attn_weights=True,
        )
        attention = attention.masked_fill(~query_valid.unsqueeze(-1), 0.0)
        attention = attention.masked_fill(~exemplar_valid.unsqueeze(1), 0.0)

        active_mask = query_valid & (~no_exemplar).unsqueeze(1)
        attended = attended.masked_fill(~active_mask.unsqueeze(-1), 0.0)
        fused = self.fusion(torch.cat((query_hidden, attended), dim=-1))
        hidden = self.temporal_encoder(fused, active_mask)
        density = F.softplus(self.density_head(hidden).squeeze(-1))
        density = density * active_mask.to(dtype=density.dtype)
        return ExemplarCountingOutput(
            density=density,
            count=density.sum(dim=1),
            cross_attention=attention,
        )


class IVACP2LStyleCounter(nn.Module):
    """IVAC-P2L-style period-length distribution/count scaffold.

    The head predicts a per-frame period-length distribution and periodicity
    gate, then sums the expected reciprocal period.  Source-specific RGB
    encoding, P2L supervision, loss calibration, checkpoints, and parity are
    not implemented, so this primitive must not be registered as runnable.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        model_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 8,
        feedforward_dim: int = 512,
        dropout: float = 0.1,
        period_bins: Sequence[int] = tuple(range(2, 65)),
        max_length: int = 4096,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.period_bins: Tensor
        self.register_buffer("period_bins", _period_tensor(period_bins), persistent=True)
        self.temporal_encoder = _TemporalTransformer(
            input_dim=input_dim,
            model_dim=model_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            max_length=max_length,
        )
        self.period_length_head = nn.Linear(model_dim, len(self.period_bins))
        self.periodicity_head = nn.Linear(model_dim, 1)

    def forward(
        self,
        frame_features: Tensor,
        valid_mask: Tensor | None = None,
    ) -> PeriodLengthCountingOutput:
        _validate_features(
            frame_features,
            expected_dim=self.input_dim,
            name="frame_features",
        )
        mask = _validate_mask(frame_features, valid_mask, name="valid_mask")
        hidden = self.temporal_encoder(frame_features, mask)

        period_logits = self.period_length_head(hidden)
        period_distribution = torch.softmax(period_logits, dim=-1)
        periodicity = torch.sigmoid(self.periodicity_head(hidden).squeeze(-1))
        validity = mask.to(dtype=hidden.dtype)

        period_logits = period_logits.masked_fill(~mask.unsqueeze(-1), 0.0)
        period_distribution = period_distribution * validity.unsqueeze(-1)
        periodicity = periodicity * validity
        periods = self.period_bins.to(dtype=hidden.dtype)
        expected_period = (period_distribution * periods).sum(dim=-1)
        density = periodicity * (period_distribution / periods).sum(dim=-1)
        return PeriodLengthCountingOutput(
            period_length_logits=period_logits,
            period_length_distribution=period_distribution,
            periodicity=periodicity,
            expected_period=expected_period,
            density=density,
            count=density.sum(dim=1),
        )


__all__ = [
    "DensityRegressionOutput",
    "ESCountsStyleRegressor",
    "ExemplarCountingOutput",
    "IVACP2LStyleCounter",
    "PeriodLengthCountingOutput",
    "RepNetStyleCounter",
    "RepNetStyleOutput",
    "TransRACStyleRegressor",
    "temporal_self_similarity",
]
