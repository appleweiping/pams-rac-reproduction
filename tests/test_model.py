import math

import pytest
import torch

from pams.model import (
    PAMSEncoder,
    PAMSModel,
    PeriodHead,
    SinusoidalPositionalEncoding,
    TemporalPeriodHead,
)


def test_disclosed_default_architecture() -> None:
    encoder = PAMSEncoder()
    assert encoder.input_projection.in_features == 99
    assert encoder.model_dim == 512
    assert encoder.embedding_dim == 512
    assert len(encoder.transformer.layers) == 4
    assert encoder.transformer.layers[0].norm_first is False
    assert encoder.transformer.layers[0].linear1.out_features == 2048
    assert encoder.transformer.layers[0].self_attn.num_heads == 16
    assert encoder.input_projection_scale == "none"
    assert encoder.position_encoding_mode == "sinusoidal"


def test_default_projection_scale_is_bitwise_compatible_with_explicit_none() -> None:
    torch.manual_seed(17)
    default = PAMSEncoder(
        input_dim=6,
        model_dim=8,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
    ).eval()
    explicit_none = PAMSEncoder(
        input_dim=6,
        model_dim=8,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
        input_projection_scale="none",
    ).eval()
    explicit_none.load_state_dict(default.state_dict())
    inputs = torch.randn(2, 7, 6)
    valid = torch.ones(2, 7, dtype=torch.bool)

    with torch.no_grad():
        default_output = default(inputs, valid)
        explicit_output = explicit_none(inputs, valid)

    assert torch.equal(default_output, explicit_output)


def test_default_position_mode_is_bitwise_compatible_with_explicit_sinusoidal() -> None:
    torch.manual_seed(17)
    default = PAMSEncoder(
        input_dim=6,
        model_dim=8,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
    ).eval()
    explicit = PAMSEncoder(
        input_dim=6,
        model_dim=8,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
        position_encoding_mode="sinusoidal",
    ).eval()
    explicit.load_state_dict(default.state_dict(), strict=True)
    inputs = torch.randn(2, 7, 6)
    valid = torch.ones(2, 7, dtype=torch.bool)

    with torch.no_grad():
        default_output = default(inputs, valid)
        explicit_output = explicit(inputs, valid)

    assert torch.equal(default_output, explicit_output)


def test_optional_position_indices_gather_rows_and_none_is_bitwise_unchanged() -> None:
    encoding = SinusoidalPositionalEncoding(dimension=6, max_length=8)
    inputs = torch.zeros(2, 3, 6)
    indices = torch.tensor(
        [
            [2, 1, 0],
            [3, 5, 7],
        ],
        dtype=torch.long,
    )

    omitted = encoding(inputs)
    explicit_none = encoding(inputs, position_indices=None)
    selected = encoding(inputs, position_indices=indices)

    assert torch.equal(omitted, explicit_none)
    assert torch.equal(
        omitted,
        encoding.encoding[:3].unsqueeze(0).expand(2, -1, -1),
    )
    assert torch.equal(selected, encoding.encoding[indices])


@pytest.mark.parametrize(
    ("indices", "error", "message"),
    [
        (torch.tensor([0, 1, 2]), ValueError, "shape"),
        (torch.zeros(2, 3), TypeError, "integer dtype"),
        (torch.zeros(2, 3, dtype=torch.bool), TypeError, "integer dtype"),
        (
            torch.tensor([[0, 1, -1], [0, 1, 2]]),
            ValueError,
            "capacity",
        ),
        (
            torch.tensor([[0, 1, 8], [0, 1, 2]]),
            ValueError,
            "capacity",
        ),
        ([[0, 1, 2], [0, 1, 2]], TypeError, "tensor"),
    ],
)
def test_position_indices_validate_shape_dtype_and_range(
    indices: object,
    error: type[Exception],
    message: str,
) -> None:
    encoding = SinusoidalPositionalEncoding(dimension=6, max_length=8)
    inputs = torch.zeros(2, 3, 6)

    with pytest.raises(error, match=message):
        encoding(
            inputs,
            position_indices=indices,  # type: ignore[arg-type]
        )


def test_encoder_threads_explicit_position_indices_without_changing_none_path() -> None:
    torch.manual_seed(17)
    encoder = PAMSEncoder(
        input_dim=6,
        model_dim=8,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
    ).eval()
    inputs = torch.randn(2, 5, 6)
    valid = torch.ones(2, 5, dtype=torch.bool)
    reversed_indices = torch.arange(4, -1, -1).expand(2, -1)

    with torch.no_grad():
        omitted = encoder(inputs, valid)
        explicit_none = encoder(inputs, valid, position_indices=None)
        reversed_output = encoder(
            inputs,
            valid,
            position_indices=reversed_indices,
        )

    assert torch.equal(omitted, explicit_none)
    assert not torch.equal(omitted, reversed_output)


def test_no_absolute_pe_ignores_position_indices_and_keeps_checkpoint_schema() -> None:
    torch.manual_seed(17)
    sinusoidal = PAMSEncoder(
        input_dim=6,
        model_dim=8,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
        position_encoding_mode="sinusoidal",
    ).eval()
    no_absolute_pe = PAMSEncoder(
        input_dim=6,
        model_dim=8,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
        position_encoding_mode="none",
    ).eval()

    assert set(no_absolute_pe.state_dict()) == set(sinusoidal.state_dict())
    no_absolute_pe.load_state_dict(sinusoidal.state_dict(), strict=True)
    inputs = torch.randn(2, 5, 6)
    valid = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, False, True, True],
        ]
    )
    ignored_indices = torch.full((2, 5), 100_000, dtype=torch.long)

    with torch.no_grad():
        canonical = no_absolute_pe(inputs, valid)
        indexed = no_absolute_pe(
            inputs,
            valid,
            position_indices=ignored_indices,
        )
        canonical_with_pre_pe = no_absolute_pe.forward_with_pre_pe(
            inputs,
            valid,
        )
        indexed_with_pre_pe = no_absolute_pe.forward_with_pre_pe(
            inputs,
            valid,
            position_indices=ignored_indices,
        )

    assert torch.equal(canonical, indexed)
    assert torch.equal(canonical_with_pre_pe[0], indexed_with_pre_pe[0])
    assert torch.equal(canonical_with_pre_pe[1], indexed_with_pre_pe[1])


def test_encoder_rejects_unknown_position_encoding_mode() -> None:
    with pytest.raises(ValueError, match="position_encoding_mode"):
        PAMSEncoder(
            input_dim=6,
            model_dim=8,
            embedding_dim=8,
            num_layers=1,
            num_heads=2,
            feedforward_dim=16,
            position_encoding_mode="learned",  # type: ignore[arg-type]
        )


def test_sqrt_model_dim_scales_linear_projection_before_positions() -> None:
    plain = PAMSEncoder(
        input_dim=6,
        model_dim=8,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
        input_projection_scale="none",
    ).eval()
    scaled = PAMSEncoder(
        input_dim=6,
        model_dim=8,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
        input_projection_scale="sqrt_model_dim",
    ).eval()
    scaled.load_state_dict(plain.state_dict())
    inputs = torch.randn(1, 5, 6)
    captured_plain: list[torch.Tensor] = []
    captured_scaled: list[torch.Tensor] = []
    plain_hook = plain.position_encoding.register_forward_pre_hook(
        lambda _module, args: captured_plain.append(args[0].detach().clone())
    )
    scaled_hook = scaled.position_encoding.register_forward_pre_hook(
        lambda _module, args: captured_scaled.append(args[0].detach().clone())
    )

    try:
        with torch.no_grad():
            plain(inputs)
            scaled(inputs)
    finally:
        plain_hook.remove()
        scaled_hook.remove()

    assert len(captured_plain) == len(captured_scaled) == 1
    assert torch.allclose(
        captured_scaled[0],
        captured_plain[0] * math.sqrt(plain.model_dim),
        rtol=1e-6,
        atol=1e-7,
    )


def test_label_free_projection_probe_input_delta_exceeds_position_energy() -> None:
    """Magnitude probe only: this is not a count-label or effect-size metric."""

    model_dim = 64
    plain = PAMSEncoder(
        input_dim=6,
        model_dim=model_dim,
        embedding_dim=model_dim,
        num_layers=1,
        num_heads=8,
        feedforward_dim=128,
        dropout=0.0,
        input_projection_scale="none",
    ).eval()
    scaled = PAMSEncoder(
        input_dim=6,
        model_dim=model_dim,
        embedding_dim=model_dim,
        num_layers=1,
        num_heads=8,
        feedforward_dim=128,
        dropout=0.0,
        input_projection_scale="sqrt_model_dim",
    ).eval()
    with torch.no_grad():
        plain.input_projection.weight.fill_(0.25 / plain.input_dim)
        plain.input_projection.bias.zero_()
    scaled.load_state_dict(plain.state_dict())

    pose_a = torch.zeros(1, 1, plain.input_dim)
    pose_b = torch.ones(1, 1, plain.input_dim)
    with torch.no_grad():
        plain_delta = (
            plain.input_projection(pose_b) - plain.input_projection(pose_a)
        ).norm()
        scaled_delta = plain_delta * math.sqrt(model_dim)
        position_energy = scaled.position_encoding.encoding[0].norm()

    assert plain_delta < position_energy
    assert scaled_delta > position_energy
    assert scaled_delta.item() == pytest.approx(
        plain_delta.item() * math.sqrt(model_dim)
    )


def test_encoder_rejects_unknown_input_projection_scale() -> None:
    with pytest.raises(ValueError, match="input_projection_scale"):
        PAMSEncoder(input_projection_scale="sqrt_input_dim")  # type: ignore[arg-type]


def test_encoder_normalizes_valid_rows_and_zeros_invalid_rows() -> None:
    torch.manual_seed(4)
    encoder = PAMSEncoder(
        input_dim=6,
        model_dim=16,
        embedding_dim=8,
        num_layers=2,
        num_heads=4,
        feedforward_dim=32,
        dropout=0.0,
    )
    encoder.eval()
    inputs = torch.randn(2, 7, 2, 3)
    valid = torch.tensor(
        [
            [True, True, True, True, False, False, False],
            [True, True, True, True, True, True, True],
        ]
    )
    output = encoder(inputs, valid)
    assert output.shape == (2, 7, 8)
    assert torch.allclose(output[0, 4:], torch.zeros(3, 8))
    norms = output[valid].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)


def test_encoder_exposes_exact_pre_pe_projection_without_changing_forward() -> None:
    torch.manual_seed(2026)
    encoder = PAMSEncoder(
        input_dim=6,
        model_dim=8,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
    ).eval()
    inputs = torch.randn(2, 7, 2, 3)
    valid = torch.tensor(
        [
            [True, True, True, False, False, False, False],
            [True, True, True, True, True, True, True],
        ]
    )
    with torch.no_grad():
        ordinary = encoder(inputs, valid)
        exposed, pre_pe = encoder.forward_with_pre_pe(inputs, valid)
        expected_projection = encoder.input_projection(inputs.flatten(start_dim=2))
        expected_projection = expected_projection.masked_fill(
            ~valid.unsqueeze(-1),
            0.0,
        )

    assert torch.equal(ordinary, exposed)
    assert torch.equal(pre_pe, expected_projection)
    assert torch.count_nonzero(pre_pe[~valid]) == 0


def test_encoder_handles_fully_missing_pose_without_nan() -> None:
    encoder = PAMSEncoder(
        input_dim=6,
        model_dim=8,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
    )
    output = encoder(torch.zeros(1, 5, 6), torch.zeros(1, 5, dtype=torch.bool))
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output) == 0


def test_period_head_and_composed_model_are_differentiable() -> None:
    head = PeriodHead(embedding_dim=8, hidden_dim=4)
    embeddings = torch.randn(2, 5, 8, requires_grad=True)
    stream = head(embeddings)
    assert stream.shape == (2, 5)
    stream.sum().backward()
    assert embeddings.grad is not None

    encoder = PAMSEncoder(
        input_dim=6,
        model_dim=8,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
    )
    model = PAMSModel(encoder=encoder, period_head=PeriodHead(8, 4))
    valid = torch.tensor([[True, True, False]])
    encoded, predicted = model(torch.randn(1, 3, 6), valid)
    assert encoded.shape == (1, 3, 8)
    assert predicted.shape == (1, 3)
    assert predicted[0, 2] == 0


def test_temporal_period_head_is_mask_aware_and_differentiable() -> None:
    torch.manual_seed(2026)
    head = TemporalPeriodHead(embedding_dim=8, hidden_dim=4)
    valid = torch.tensor(
        [
            [True, True, False, True, True, False, False],
            [False, False, False, False, False, False, False],
        ]
    )
    clean = torch.randn(2, 7, 8, requires_grad=True)
    altered = clean.detach().clone()
    altered[~valid] = 10_000.0

    clean_stream = head(clean, valid_mask=valid)
    altered_stream = head(altered, valid_mask=valid)
    expected_initial = head.network(
        clean.masked_fill(~valid.unsqueeze(-1), 0.0)
    ).squeeze(-1)

    assert head.temporal_conv.kernel_size == (5,)
    assert head.temporal_conv.padding == (2,)
    assert head.temporal_conv.groups == 8
    assert head.temporal_conv.bias is None
    assert head.temporal_conv.weight.numel() == 8 * 5
    assert torch.count_nonzero(head.temporal_conv.weight) == 0
    assert clean_stream.shape == (2, 7)
    assert torch.equal(clean_stream[valid], expected_initial[valid])
    assert torch.equal(clean_stream[valid], altered_stream[valid])
    assert torch.count_nonzero(clean_stream[~valid]) == 0
    assert torch.isfinite(clean_stream).all()
    clean_stream[valid].sum().backward()
    assert clean.grad is not None
    assert torch.count_nonzero(clean.grad[~valid]) == 0


def test_composed_model_can_bind_period_head_to_pre_pe_projection() -> None:
    torch.manual_seed(3407)
    encoder = PAMSEncoder(
        input_dim=6,
        model_dim=8,
        embedding_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        dropout=0.0,
    ).eval()
    model = PAMSModel(encoder=encoder, period_head=PeriodHead(8, 4)).eval()
    inputs = torch.randn(1, 7, 2, 3)
    valid = torch.tensor([[True, True, True, True, True, False, False]])

    with torch.no_grad():
        expected_embeddings, projected = encoder.forward_with_pre_pe(inputs, valid)
        expected_stream = model.period_head(projected).masked_fill(~valid, 0.0)
        embeddings, stream = model.forward_with_head_source(
            inputs,
            valid,
            head_input_source="projected_pose_pre_pe",
        )

    assert torch.equal(embeddings, expected_embeddings)
    assert torch.equal(stream, expected_stream)
    assert torch.count_nonzero(stream[~valid]) == 0

    with pytest.raises(ValueError, match="input source"):
        model.forward_with_head_source(
            inputs,
            valid,
            head_input_source="unknown",  # type: ignore[arg-type]
        )
