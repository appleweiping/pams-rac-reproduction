import inspect
import math

import numpy as np
import torch

from pams.config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    PAMSConfig,
    PeriodConfig,
    SSHeadConfig,
    TrainingConfig,
)
from pams.training import (
    build_pams_model,
    collate_pose_sequences,
    predict_sequence,
    train_encoder,
    train_sshead,
)
from pams.types import PoseSequence


def _tiny_config(*, encoder_epochs: int = 2, head_epochs: int = 1) -> PAMSConfig:
    return PAMSConfig(
        seed=17,
        data=DataConfig(frames=16),
        model=ModelConfig(
            input_dim=99,
            model_dim=8,
            embedding_dim=8,
            layers=1,
            heads=2,
            feedforward_dim=16,
            dropout=0.0,
            period_head_hidden_dim=4,
        ),
        period=PeriodConfig(minimum=4, maximum=8, pose_energy_epochs=1),
        loss=LossConfig(
            scales=(1.0,),
            temperature=0.1,
            correspondence_tolerance=0.1,
            kmeans_clusters=2,
            kmeans_refresh_epochs=1,
        ),
        training=TrainingConfig(
            epochs=encoder_epochs,
            effective_batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            scheduler_factor=0.5,
            scheduler_patience=2,
            minimum_learning_rate=1e-6,
        ),
        sshead=SSHeadConfig(
            epochs=head_epochs,
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
    )


def _sequence(identifier: str, *, phase: float = 0.0, frames: int = 16) -> PoseSequence:
    time = np.arange(frames, dtype=np.float32)
    wave = np.sin(2.0 * math.pi * time / 4.0 + phase)
    xyz = np.zeros((frames, 33, 3), dtype=np.float32)
    xyz[:, :, 0] = wave[:, None]
    xyz[:, :, 1] = np.cos(2.0 * math.pi * time / 4.0 + phase)[:, None]
    xyz[:, :, 2] = np.linspace(0.0, 1.0, frames, dtype=np.float32)[:, None]
    return PoseSequence(
        video_id=identifier,
        fps=16.0,
        xyz=xyz,
        valid_mask=np.ones(frames, dtype=np.bool_),
    )


def test_collate_pads_without_creating_valid_frames() -> None:
    batch = collate_pose_sequences((_sequence("long"), _sequence("short", frames=12)))
    assert batch.poses.shape == (2, 16, 33, 3)
    assert batch.valid_mask.shape == (2, 16)
    assert batch.valid_mask[1, 12:].sum() == 0
    assert torch.count_nonzero(batch.poses[1, 12:]) == 0
    assert batch.lengths.tolist() == [16, 12]


def test_train_api_is_label_free_and_resume_is_bitwise_deterministic(tmp_path) -> None:
    signature = inspect.signature(train_encoder)
    assert "targets" not in signature.parameters
    assert "ground_truths" not in signature.parameters
    items = (
        _sequence("a", phase=0.0),
        _sequence("b", phase=0.4),
    )
    config = _tiny_config(encoder_epochs=2)

    uninterrupted = train_encoder(
        items,
        config,
        device="cpu",
        microbatch_size=1,
    )
    checkpoint = tmp_path / "encoder.pt"
    partial = train_encoder(
        items,
        config,
        device="cpu",
        microbatch_size=1,
        checkpoint_path=checkpoint,
        stop_after_epoch=1,
    )
    assert partial.completed_epochs == 1
    resumed = train_encoder(
        items,
        config,
        device="cpu",
        microbatch_size=1,
        checkpoint_path=checkpoint,
        resume=True,
    )
    assert resumed.completed_epochs == 2
    assert [item.period_source for item in resumed.history] == ["pose", "embedding"]
    assert set(resumed.cluster_assignments) == {"a", "b"}
    for name, expected in uninterrupted.model.state_dict().items():
        assert torch.equal(expected, resumed.model.state_dict()[name]), name


def test_sshead_keeps_encoder_frozen_and_prediction_is_well_formed() -> None:
    config = _tiny_config(encoder_epochs=1, head_epochs=1)
    model = build_pams_model(config)
    encoder_before = {
        name: value.detach().clone() for name, value in model.encoder.state_dict().items()
    }
    head_before = {
        name: value.detach().clone() for name, value in model.period_head.state_dict().items()
    }
    items = (_sequence("a"), _sequence("b", phase=0.7))
    trained = train_sshead(
        items,
        config,
        model=model,
        device="cpu",
        microbatch_size=1,
    )
    assert trained.completed_epochs == 1
    for name, expected in encoder_before.items():
        assert torch.equal(expected, trained.model.encoder.state_dict()[name])
    assert any(
        not torch.equal(value, trained.model.period_head.state_dict()[name])
        for name, value in head_before.items()
    )

    result = predict_sequence(trained.model, items[0], config, device="cpu")
    assert result.count >= 0
    assert 4 <= result.period_frames <= 8
    assert len(result.expert_counts) == 3
    assert result.period_stream.shape == (16,)
