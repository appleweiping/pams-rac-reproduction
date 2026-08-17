import math
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from pams.augmentation import (
    augment_pose_batch,
    make_skeleton_augmentation_generator,
    skeleton_augmentation_seed,
)
from pams.config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    PAMSConfig,
    PeriodConfig,
    SkeletonAugmentationConfig,
    SSHeadConfig,
    TrainingConfig,
    load_config,
)
from pams.training import build_pams_model, train_encoder
from pams.types import PoseSequence


def _augmentation(
    *,
    rotation_degrees: tuple[float, float, float] = (15.0, 10.0, 5.0),
    scale_range: tuple[float, float] = (0.85, 1.15),
    jitter_std: float = 0.01,
) -> SkeletonAugmentationConfig:
    return SkeletonAugmentationConfig(
        enabled=True,
        rotation_degrees=rotation_degrees,
        scale_range=scale_range,
        jitter_std=jitter_std,
    )


def _poses() -> tuple[torch.Tensor, torch.Tensor]:
    poses = torch.arange(2 * 5 * 33 * 3, dtype=torch.float32).reshape(2, 5, 33, 3)
    poses = poses / poses.max()
    valid_mask = torch.tensor(
        [
            [True, True, False, True, False],
            [False, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    return poses, valid_mask


def _generator(*, epoch_index: int = 3, batch_index: int = 2) -> torch.Generator:
    return make_skeleton_augmentation_generator(
        base_seed=2026,
        epoch_index=epoch_index,
        batch_index=batch_index,
        device=torch.device("cpu"),
    )


def _sequence(identifier: str, phase: float) -> PoseSequence:
    time = np.arange(16, dtype=np.float32)
    xyz = np.zeros((16, 33, 3), dtype=np.float32)
    xyz[:, :, 0] = np.sin(2.0 * math.pi * time / 4.0 + phase)[:, None]
    xyz[:, :, 1] = np.cos(2.0 * math.pi * time / 4.0 + phase)[:, None]
    xyz[:, :, 2] = np.linspace(0.0, 1.0, 16, dtype=np.float32)[:, None]
    valid_mask = np.ones(16, dtype=np.bool_)
    valid_mask[5] = False
    xyz[~valid_mask] = 0.0
    return PoseSequence(
        video_id=identifier,
        fps=16.0,
        xyz=xyz,
        valid_mask=valid_mask,
    )


def _tiny_training_config(
    *,
    epochs: int,
    pose_energy_epochs: int = 1,
) -> PAMSConfig:
    return PAMSConfig(
        seed=23,
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
        period=PeriodConfig(
            minimum=4,
            maximum=8,
            pose_energy_epochs=pose_energy_epochs,
        ),
        loss=LossConfig(
            scales=(1.0,),
            temperature=0.1,
            kmeans_clusters=2,
            kmeans_refresh_epochs=1,
        ),
        training=TrainingConfig(
            epochs=epochs,
            effective_batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            scheduler_factor=0.5,
            scheduler_patience=2,
            minimum_learning_rate=1e-6,
            skeleton_augmentation=_augmentation(),
        ),
        sshead=SSHeadConfig(
            epochs=1,
            learning_rate=1e-3,
            weight_decay=0.0,
        ),
    )


def test_augmentation_is_batch_addressed_deterministic_and_mask_safe() -> None:
    poses, valid_mask = _poses()
    original_mask = valid_mask.clone()
    config = _augmentation()

    first = augment_pose_batch(
        poses,
        valid_mask,
        config,
        generator=_generator(),
    )
    repeated = augment_pose_batch(
        poses,
        valid_mask,
        config,
        generator=_generator(),
    )
    next_batch = augment_pose_batch(
        poses,
        valid_mask,
        config,
        generator=_generator(batch_index=3),
    )

    assert torch.equal(first, repeated)
    assert not torch.equal(first[valid_mask], next_batch[valid_mask])
    assert torch.equal(valid_mask, original_mask)
    assert torch.count_nonzero(first[~valid_mask]) == 0
    assert torch.isfinite(first).all()


def test_rigid_scale_is_centered_on_each_valid_frame() -> None:
    poses, valid_mask = _poses()
    config = _augmentation(
        rotation_degrees=(0.0, 0.0, 0.0),
        scale_range=(1.25, 1.25),
        jitter_std=0.0,
    )
    augmented = augment_pose_batch(
        poses,
        valid_mask,
        config,
        generator=_generator(),
    )

    original_centers = poses[valid_mask].mean(dim=1)
    augmented_centers = augmented[valid_mask].mean(dim=1)
    original_bones = poses[valid_mask][:, 1] - poses[valid_mask][:, 0]
    augmented_bones = augmented[valid_mask][:, 1] - augmented[valid_mask][:, 0]
    assert torch.allclose(augmented_centers, original_centers, atol=1e-6)
    assert torch.allclose(
        torch.linalg.vector_norm(augmented_bones, dim=-1),
        1.25 * torch.linalg.vector_norm(original_bones, dim=-1),
        atol=1e-6,
    )


def test_seed_derivation_rejects_invalid_batch_identity() -> None:
    assert skeleton_augmentation_seed(
        base_seed=2026,
        epoch_index=1,
        batch_index=4,
    ) == skeleton_augmentation_seed(
        base_seed=2026,
        epoch_index=1,
        batch_index=4,
    )
    with pytest.raises(ValueError, match="epoch_index"):
        skeleton_augmentation_seed(base_seed=1, epoch_index=-1, batch_index=0)
    with pytest.raises(ValueError, match="batch_index"):
        skeleton_augmentation_seed(base_seed=1, epoch_index=0, batch_index=-1)


@pytest.mark.parametrize(
    "payload",
    [
        {"enabled": "true"},
        {"enabled": True},
        {"rotation_degrees": [1.0, 0.0, 0.0]},
        {"enabled": True, "rotation_degrees": [181.0, 0.0, 0.0]},
        {"enabled": True, "scale_range": [1.2, 0.8]},
        {"enabled": True, "jitter_std": 1.1},
        {"enabled": True, "unknown": 1},
    ],
)
def test_augmentation_config_strictly_rejects_invalid_states(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        SkeletonAugmentationConfig.model_validate(payload)


def test_disabled_default_preserves_historical_config_identity() -> None:
    explicit = PAMSConfig()
    implicit_payload = explicit.model_dump()
    implicit_payload["training"].pop("skeleton_augmentation")
    implicit = PAMSConfig.model_validate(implicit_payload)

    assert explicit.fingerprint == implicit.fingerprint
    assert explicit.nonseed_fingerprint == implicit.nonseed_fingerprint
    assert explicit.pose_fingerprint == implicit.pose_fingerprint


def test_v13_experiment_changes_only_training_augmentation() -> None:
    root = Path(__file__).parents[1]
    previous = load_config(
        root / "configs" / "experiments" / "pams_longest_contiguous_track_v8.yaml"
    )
    augmented = load_config(
        root / "configs" / "experiments" / "pams_skeleton_augmentation_v13.yaml"
    )

    assert augmented.training.skeleton_augmentation.enabled
    restored = augmented.model_dump()
    restored["training"]["skeleton_augmentation"] = (
        SkeletonAugmentationConfig().model_dump()
    )
    assert restored == previous.model_dump()
    assert augmented.fingerprint != previous.fingerprint
    assert augmented.nonseed_fingerprint != previous.nonseed_fingerprint
    assert augmented.pose_fingerprint == previous.pose_fingerprint


def test_post_warmup_period_evidence_uses_clean_not_augmented_view(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import training as training_module

    items = (_sequence("a", 0.0), _sequence("b", 0.4))
    config = _tiny_training_config(epochs=1, pose_energy_epochs=0)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = build_pams_model(config)

    encoder_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def record_encoder_call(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        encoder_calls.append((inputs[0].detach().clone(), output))

    handle = model.encoder.register_forward_hook(record_encoder_call)
    original_estimator = training_module._estimate_post_warmup_periods
    observed = False

    def assert_clean_period_view(
        *,
        config: PAMSConfig,
        embeddings: torch.Tensor,
        projected_pose: torch.Tensor | None,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, str]:
        nonlocal observed
        observed = True
        # Calls are: clean prototype view, clean period-evidence view, then
        # the augmented optimization view. The estimator must receive the
        # second output, never the third.
        assert embeddings.data_ptr() == encoder_calls[-2][1].data_ptr()
        assert embeddings.data_ptr() != encoder_calls[-1][1].data_ptr()
        assert not torch.equal(encoder_calls[-2][0], encoder_calls[-1][0])
        return original_estimator(
            config=config,
            embeddings=embeddings,
            projected_pose=projected_pose,
            valid_mask=valid_mask,
        )

    monkeypatch.setattr(
        training_module,
        "_estimate_post_warmup_periods",
        assert_clean_period_view,
    )
    try:
        train_encoder(
            items,
            config,
            model=model,
            device="cpu",
            microbatch_size=2,
        )
    finally:
        handle.remove()
    assert observed


def test_augmented_encoder_resume_matches_uninterrupted_training(tmp_path: Path) -> None:
    items = (_sequence("a", 0.0), _sequence("b", 0.4))
    config = _tiny_training_config(epochs=2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        uninterrupted_model = build_pams_model(config)
    uninterrupted = train_encoder(
        items,
        config,
        model=uninterrupted_model,
        device="cpu",
        microbatch_size=2,
    )

    checkpoint = tmp_path / "augmented-encoder.pt"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        resumed_model = build_pams_model(config)
    partial = train_encoder(
        items,
        config,
        model=resumed_model,
        device="cpu",
        microbatch_size=2,
        checkpoint_path=checkpoint,
        stop_after_epoch=1,
    )
    assert partial.completed_epochs == 1
    resumed = train_encoder(
        items,
        config,
        model=resumed_model,
        device="cpu",
        microbatch_size=2,
        checkpoint_path=checkpoint,
        resume=True,
    )

    assert resumed.completed_epochs == 2
    for name, expected in uninterrupted.model.state_dict().items():
        assert torch.equal(expected, resumed.model.state_dict()[name]), name
