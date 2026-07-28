import math

import pytest
import torch

from pams.model import PAMSEncoder, PAMSModel, PeriodHead


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
