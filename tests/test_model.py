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
