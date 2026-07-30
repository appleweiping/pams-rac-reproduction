import hashlib
import inspect
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
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
from pams.model import TemporalPeriodHead
from pams.training import (
    CheckpointProvenance,
    build_pams_model,
    collate_pose_sequences,
    load_model_checkpoint,
    predict_sequence,
    train_encoder,
    train_sshead,
    validate_sshead_encoder_binding,
    validate_terminal_checkpoint,
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


def _with_projected_vector_period(config: PAMSConfig) -> PAMSConfig:
    payload = config.model_dump()
    payload["period"][
        "post_warmup_source"
    ] = "projected_pose_velocity_vector_acf"
    return PAMSConfig.model_validate(payload)


def _with_pre_pe_head(config: PAMSConfig) -> PAMSConfig:
    payload = config.model_dump()
    payload["sshead"]["input_source"] = "projected_pose_pre_pe"
    return PAMSConfig.model_validate(payload)


def _with_temporal_head(config: PAMSConfig) -> PAMSConfig:
    payload = config.model_dump()
    payload["sshead"]["architecture"] = "temporal_conv"
    return PAMSConfig.model_validate(payload)


def _with_fixed_period(config: PAMSConfig, *, frames: int = 4) -> PAMSConfig:
    payload = config.model_dump()
    payload["period"]["training_mode"] = "fixed_period_inferred"
    payload["period"]["fixed_period_frames"] = frames
    return PAMSConfig.model_validate(payload)


def _with_position_permutation_consistency(
    config: PAMSConfig,
    *,
    weight: float = 1.0,
) -> PAMSConfig:
    payload = config.model_dump()
    payload["training"]["position_permutation_consistency_weight"] = weight
    return PAMSConfig.model_validate(payload)


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


def _provenance(
    config: PAMSConfig,
    identifiers: tuple[str, ...] = ("b", "a"),
    *,
    dataset_fingerprint: str = "d" * 64,
    upstream_encoder_checkpoint_sha256: str | None = None,
) -> CheckpointProvenance:
    return CheckpointProvenance(
        protocol=config.protocol,
        dataset_fingerprint=dataset_fingerprint,
        training_video_ids=identifiers,
        pose_fingerprint=config.pose_fingerprint,
        pose_cache_set_sha256="c" * 64,
        source_git_sha="a" * 40,
        upstream_encoder_checkpoint_sha256=upstream_encoder_checkpoint_sha256,
    )


@pytest.mark.parametrize("stage", ["encoder", "sshead"])
def test_public_training_boundary_rejects_objects_containing_test_labels(
    stage: str,
) -> None:
    label_bearing_test_object = {
        "video_id": "v_BenchPress_g21_c01",
        "split": "test",
        "action": "BenchPress",
        "count": 7,
    }
    config = _tiny_config()

    with pytest.raises(TypeError, match="only PoseSequence"):
        if stage == "encoder":
            train_encoder([label_bearing_test_object], config)  # type: ignore[list-item]
        else:
            train_sshead(  # type: ignore[list-item]
                [label_bearing_test_object],
                config,
                model=object(),  # type: ignore[arg-type]
            )


def _write_terminal_encoder_fixture(
    tmp_path: Path,
    config: PAMSConfig,
) -> tuple[Path, Path, CheckpointProvenance, Any]:
    from pams import training as training_module

    checkpoint = tmp_path / "encoder.pt"
    progress = tmp_path / "encoder.jsonl"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = build_pams_model(config)
    optimizer = training_module.AdamW(model.encoder.parameters(), lr=1e-3)
    scheduler = training_module.ReduceLROnPlateau(optimizer)
    provenance = _provenance(config)
    bank = training_module.VideoPrototypeBank(
        video_ids=provenance.training_video_ids,
        features=torch.zeros(
            len(provenance.training_video_ids),
            config.model.embedding_dim,
        ),
        cluster_labels=torch.arange(len(provenance.training_video_ids), dtype=torch.long),
    )
    history = [
        training_module.EncoderEpochStats(
            epoch=epoch,
            loss=1.0 / epoch,
            learning_rate=1e-3,
            period_source=(
                "fixed_period_inferred"
                if config.period.training_mode == "fixed_period_inferred"
                else (
                    "pose"
                    if epoch <= config.period.pose_energy_epochs
                    else (
                        "embedding"
                        if config.period.post_warmup_source
                        == "embedding_velocity_coordinate"
                        else "projected_pose_velocity_vector_acf"
                    )
                )
            ),
            period_confidence_mean=0.8,
            period_valid_fraction=1.0,
            optimizer_steps=1,
            clusters_refreshed=True,
            cross_cluster_requested=2,
            cross_cluster_actual=2,
            cross_cluster_shortfall=0,
        )
        for epoch in range(1, config.training.epochs + 1)
    ]
    payload = training_module._checkpoint_payload(
        stage="encoder",
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_epochs=config.training.epochs,
        history=history,
        cluster_assignments=bank.assignments,
        prototype_bank=bank,
        provenance=provenance,
    )
    torch.save(payload, checkpoint)
    rows = [
        training_module._progress_row(
            stage="encoder",
            stats=statistics,
            config=config,
        )
        for statistics in history
    ]
    progress.write_text(
        "".join(
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return checkpoint, progress, provenance, model


def test_fixed_period_training_proxy_is_constant_mask_aware_and_audited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import training as training_module

    config = _with_fixed_period(_tiny_config(encoder_epochs=1), frames=4)
    valid_mask = torch.tensor(
        [
            [True] * 8 + [False] * 8,
            [True] * 7 + [False] * 9,
        ]
    )
    periods, confidences = training_module._estimate_fixed_training_periods(
        config=config,
        valid_mask=valid_mask,
    )
    assert periods.tolist() == [4.0, 4.0]
    assert confidences.tolist() == [1.0, 0.0]

    def forbidden_adaptive_estimator(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("fixed-period training called an adaptive estimator")

    monkeypatch.setattr(
        training_module,
        "estimate_period_from_pose",
        forbidden_adaptive_estimator,
    )
    monkeypatch.setattr(
        training_module,
        "estimate_period_from_embeddings",
        forbidden_adaptive_estimator,
    )
    monkeypatch.setattr(
        training_module,
        "estimate_period_from_projected_pose",
        forbidden_adaptive_estimator,
    )
    checkpoint = tmp_path / "encoder.pt"
    progress = tmp_path / "encoder.jsonl"
    result = train_encoder(
        (_sequence("a"), _sequence("b", phase=0.4)),
        config,
        device="cpu",
        microbatch_size=2,
        checkpoint_path=checkpoint,
        progress_path=progress,
    )

    assert [row.period_source for row in result.history] == ["fixed_period_inferred"]
    progress_row = json.loads(progress.read_text(encoding="utf-8"))
    assert progress_row["stats"]["period_source"] == "fixed_period_inferred"
    assert progress_row["stats"]["period_training_mode"] == "fixed_period_inferred"
    assert progress_row["stats"]["fixed_period_frames"] == 4
    assert progress_row["config_fingerprint"] == config.fingerprint
    head_checkpoint = tmp_path / "sshead.pt"
    head_progress = tmp_path / "sshead.jsonl"
    head_result = train_sshead(
        (_sequence("a"), _sequence("b", phase=0.4)),
        config,
        model=result.model,
        device="cpu",
        microbatch_size=1,
        checkpoint_path=head_checkpoint,
        progress_path=head_progress,
    )
    assert head_result.completed_epochs == 1
    head_progress_row = json.loads(head_progress.read_text(encoding="utf-8"))
    assert head_progress_row["stats"]["period_source"] == "fixed_period_inferred"
    assert head_progress_row["stats"]["period_training_mode"] == "fixed_period_inferred"
    assert head_progress_row["stats"]["fixed_period_frames"] == 4


def test_collate_pads_without_creating_valid_frames() -> None:
    batch = collate_pose_sequences((_sequence("long"), _sequence("short", frames=12)))
    assert batch.poses.shape == (2, 16, 33, 3)
    assert batch.valid_mask.shape == (2, 16)
    assert batch.valid_mask[1, 12:].sum() == 0
    assert torch.count_nonzero(batch.poses[1, 12:]) == 0
    assert batch.lengths.tolist() == [16, 12]


def test_train_api_is_label_free_and_resume_is_bitwise_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
        microbatch_size=2,
    )
    checkpoint = tmp_path / "encoder.pt"
    progress = tmp_path / "encoder.jsonl"
    partial = train_encoder(
        items,
        config,
        device="cpu",
        microbatch_size=2,
        checkpoint_path=checkpoint,
        progress_path=progress,
        stop_after_epoch=1,
    )
    assert partial.completed_epochs == 1
    partial_rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
    assert len(partial_rows) == 1
    assert partial_rows[0]["stage"] == "encoder"
    assert partial_rows[0]["epoch"] == 1
    assert partial_rows[0]["config_fingerprint"] == config.fingerprint
    assert partial_rows[0]["checkpoint_role"] == "encoder_checkpoint"
    assert "epoch" not in partial_rows[0]["stats"]
    assert "period_confidence_mean" in partial_rows[0]["stats"]
    assert "period_valid_fraction" in partial_rows[0]["stats"]
    assert "cross_cluster_shortfall" in partial_rows[0]["stats"]
    assert "position_permutation_consistency" not in partial_rows[0]["stats"]
    assert set(partial_rows[0]) == {
        "schema_version",
        "stage",
        "epoch",
        "stats",
        "config_fingerprint",
        "checkpoint_role",
    }
    real_torch_load = torch.load
    checkpoint_map_locations: list[object] = []

    def recording_torch_load(*args: Any, **kwargs: Any) -> Any:
        checkpoint_map_locations.append(kwargs.get("map_location"))
        return real_torch_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_torch_load)
    resumed = train_encoder(
        tuple(reversed(items)),
        config,
        device="cpu",
        microbatch_size=2,
        checkpoint_path=checkpoint,
        progress_path=progress,
        resume=True,
    )
    assert resumed.completed_epochs == 2
    assert checkpoint_map_locations == [torch.device("cpu")]
    assert [item.period_source for item in resumed.history] == ["pose", "embedding"]
    resumed_rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
    assert [row["epoch"] for row in resumed_rows] == [1, 2]
    assert set(resumed.cluster_assignments) == {"a", "b"}
    assert resumed.prototype_bank is not None
    assert uninterrupted.prototype_bank is not None
    assert resumed.prototype_bank.video_ids == uninterrupted.prototype_bank.video_ids
    assert torch.equal(
        resumed.prototype_bank.features,
        uninterrupted.prototype_bank.features,
    )
    assert torch.equal(
        resumed.prototype_bank.cluster_labels,
        uninterrupted.prototype_bank.cluster_labels,
    )
    for stats in resumed.history:
        assert stats.position_permutation_consistency == 0.0
        assert 0.0 <= stats.period_confidence_mean <= 1.0
        assert 0.0 <= stats.period_valid_fraction <= 1.0
        assert (
            stats.cross_cluster_requested
            == stats.cross_cluster_actual + stats.cross_cluster_shortfall
        )
        assert isinstance(stats.cross_cluster_requested, int)
        assert isinstance(stats.cross_cluster_actual, int)
        assert isinstance(stats.cross_cluster_shortfall, int)
    for name, expected in uninterrupted.model.state_dict().items():
        assert torch.equal(expected, resumed.model.state_dict()[name]), name
    resumed_payload = real_torch_load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert all(
        "position_permutation_consistency" not in row
        for row in resumed_payload["history"]
    )


def test_valid_position_permutations_are_seeded_and_mask_local() -> None:
    from pams import training as training_module

    valid = torch.tensor(
        [
            [True, True, False, True, False, True],
            [False, True, True, True, False, False],
            [False, False, False, False, False, False],
        ]
    )
    canonical = torch.arange(valid.shape[1]).expand(valid.shape[0], -1)

    torch.manual_seed(3407)
    first = training_module._permuted_valid_position_indices(valid)
    torch.manual_seed(3407)
    second = training_module._permuted_valid_position_indices(valid)

    assert torch.equal(first, second)
    assert torch.equal(first[~valid], canonical[~valid])
    for sample_index in range(valid.shape[0]):
        expected = canonical[sample_index, valid[sample_index]].sort().values
        actual = first[sample_index, valid[sample_index]].sort().values
        assert torch.equal(actual, expected)


def test_pe_permutation_training_is_resume_deterministic_and_audited(
    tmp_path: Path,
) -> None:
    config = _with_position_permutation_consistency(
        _tiny_config(encoder_epochs=2)
    )
    items = (
        _sequence("a", phase=0.0),
        _sequence("b", phase=0.4),
    )
    uninterrupted = train_encoder(
        items,
        config,
        device="cpu",
        microbatch_size=2,
    )
    checkpoint = tmp_path / "pe-permutation.pt"
    progress = tmp_path / "pe-permutation.jsonl"
    train_encoder(
        items,
        config,
        device="cpu",
        microbatch_size=2,
        checkpoint_path=checkpoint,
        progress_path=progress,
        stop_after_epoch=1,
    )
    resumed = train_encoder(
        tuple(reversed(items)),
        config,
        device="cpu",
        microbatch_size=2,
        checkpoint_path=checkpoint,
        progress_path=progress,
        resume=True,
    )

    assert resumed.history == uninterrupted.history
    assert all(
        math.isfinite(row.position_permutation_consistency)
        and row.position_permutation_consistency > 0.0
        for row in resumed.history
    )
    for name, expected in uninterrupted.model.state_dict().items():
        assert torch.equal(expected, resumed.model.state_dict()[name]), name
    rows = [
        json.loads(line)
        for line in progress.read_text(encoding="utf-8").splitlines()
    ]
    assert all(
        row["stats"]["position_permutation_consistency"] > 0.0
        for row in rows
    )
    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    assert all(
        row["position_permutation_consistency"] > 0.0
        for row in payload["history"]
    )


def test_projected_vector_period_routes_encoder_and_sshead_to_same_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import training as training_module

    config = _with_projected_vector_period(
        _tiny_config(encoder_epochs=2, head_epochs=1)
    )
    calls: list[tuple[int, int]] = []

    def projected_vector_period(
        projected_pose: torch.Tensor,
        minimum: int,
        maximum: int,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del maximum
        assert projected_pose.shape[:2] == valid_mask.shape
        assert projected_pose.shape[-1] == config.model.model_dim
        assert not projected_pose.requires_grad
        calls.append((projected_pose.shape[0], projected_pose.shape[1]))
        return (
            torch.full(
                (projected_pose.shape[0],),
                float(minimum),
                dtype=projected_pose.dtype,
                device=projected_pose.device,
            ),
            torch.full(
                (projected_pose.shape[0],),
                0.75,
                dtype=projected_pose.dtype,
                device=projected_pose.device,
            ),
        )

    def forbidden_embedding_period(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("vector-period config routed through embedding coordinate")

    monkeypatch.setattr(
        training_module,
        "estimate_period_from_projected_pose",
        projected_vector_period,
    )
    monkeypatch.setattr(
        training_module,
        "estimate_period_from_embeddings",
        forbidden_embedding_period,
    )
    items = (_sequence("a"), _sequence("b", phase=0.4))
    encoder_checkpoint = tmp_path / "encoder.pt"
    encoder_progress = tmp_path / "encoder.jsonl"
    partial = train_encoder(
        items,
        config,
        device="cpu",
        microbatch_size=2,
        checkpoint_path=encoder_checkpoint,
        progress_path=encoder_progress,
        stop_after_epoch=1,
    )
    assert [row.period_source for row in partial.history] == ["pose"]
    tampered_resume = tmp_path / "encoder-wrong-source.pt"
    tampered_payload = torch.load(
        encoder_checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    tampered_payload["history"][0][
        "period_source"
    ] = "projected_pose_velocity_vector_acf"
    torch.save(tampered_payload, tampered_resume)
    with pytest.raises(ValueError, match="period-source schedule mismatch"):
        train_encoder(
            items,
            config,
            device="cpu",
            microbatch_size=2,
            checkpoint_path=tampered_resume,
            resume=True,
        )
    encoder_result = train_encoder(
        tuple(reversed(items)),
        config,
        model=partial.model,
        device="cpu",
        microbatch_size=2,
        checkpoint_path=encoder_checkpoint,
        progress_path=encoder_progress,
        resume=True,
    )

    head_checkpoint = tmp_path / "head.pt"
    head_progress = tmp_path / "head.jsonl"
    head_result = train_sshead(
        items,
        config,
        model=encoder_result.model,
        device="cpu",
        microbatch_size=1,
        checkpoint_path=head_checkpoint,
        progress_path=head_progress,
    )

    assert [row.period_source for row in encoder_result.history] == [
        "pose",
        "projected_pose_velocity_vector_acf",
    ]
    assert head_result.completed_epochs == 1
    assert len(calls) == 3
    progress_row = json.loads(head_progress.read_text(encoding="utf-8"))
    assert (
        progress_row["stats"]["period_source"]
        == "projected_pose_velocity_vector_acf"
    )


def test_sshead_pre_pe_source_routes_training_and_prediction_consistently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import training as training_module

    config = _with_pre_pe_head(_tiny_config(encoder_epochs=1, head_epochs=1))
    model = build_pams_model(config)
    observed_inputs: list[torch.Tensor] = []

    def fake_forward_with_pre_pe(
        inputs: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, time = inputs.shape[:2]
        dimension = config.model.embedding_dim
        embeddings = torch.zeros(
            (batch, time, dimension),
            dtype=inputs.dtype,
            device=inputs.device,
        )
        phase = torch.arange(time, dtype=inputs.dtype, device=inputs.device)
        projected = torch.zeros_like(embeddings)
        projected[..., 0] = torch.sin(2.0 * math.pi * phase / 4.0)
        projected[..., 1] = torch.cos(2.0 * math.pi * phase / 4.0)
        projected = projected.masked_fill(~valid_mask.unsqueeze(-1), 0.0)
        return embeddings, projected

    def fixed_period(
        embeddings: torch.Tensor,
        minimum: int,
        maximum: int,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del minimum, maximum, valid_mask
        return (
            torch.full(
                (embeddings.shape[0],),
                4,
                dtype=torch.long,
                device=embeddings.device,
            ),
            torch.ones(
                embeddings.shape[0],
                dtype=embeddings.dtype,
                device=embeddings.device,
            ),
        )

    def capture_head_input(
        _module: torch.nn.Module,
        args: tuple[torch.Tensor, ...],
    ) -> None:
        observed_inputs.append(args[0].detach().cpu().clone())

    monkeypatch.setattr(
        model.encoder,
        "forward_with_pre_pe",
        fake_forward_with_pre_pe,
    )
    monkeypatch.setattr(
        training_module,
        "estimate_period_from_embeddings",
        fixed_period,
    )
    hook = model.period_head.register_forward_pre_hook(capture_head_input)
    items = (_sequence("a"), _sequence("b", phase=0.7))
    trained = train_sshead(
        items,
        config,
        model=model,
        device="cpu",
        microbatch_size=2,
        checkpoint_path=tmp_path / "head.pt",
        progress_path=tmp_path / "head.jsonl",
    )
    training_input_count = len(observed_inputs)
    assert training_input_count == 1
    expected_first_channel = torch.sin(
        2.0 * math.pi * torch.arange(16, dtype=torch.float32) / 4.0
    )
    assert torch.allclose(
        observed_inputs[0][0, :, 0],
        expected_first_channel,
    )

    result = predict_sequence(trained.model, items[0], config, device="cpu")
    hook.remove()
    assert len(observed_inputs) == training_input_count + 1
    assert torch.allclose(
        observed_inputs[-1][0, :, 0],
        expected_first_channel,
    )
    assert result.period_frames == pytest.approx(4.0)


def test_temporal_sshead_routes_the_same_valid_mask_through_training_and_prediction(
    tmp_path: Path,
) -> None:
    config = _with_temporal_head(
        _with_pre_pe_head(
            _with_fixed_period(
                _tiny_config(encoder_epochs=1, head_epochs=1),
            )
        )
    )
    model = build_pams_model(config)
    assert isinstance(model.period_head, TemporalPeriodHead)
    observed_masks: list[torch.Tensor] = []

    def capture_valid_mask(
        _module: torch.nn.Module,
        _args: tuple[torch.Tensor, ...],
        kwargs: dict[str, object],
    ) -> None:
        valid_mask = kwargs.get("valid_mask")
        assert isinstance(valid_mask, torch.Tensor)
        observed_masks.append(valid_mask.detach().cpu().clone())

    hook = model.period_head.register_forward_pre_hook(
        capture_valid_mask,
        with_kwargs=True,
    )
    mask = np.ones(16, dtype=np.bool_)
    mask[[2, 9, 15]] = False
    first = replace(_sequence("a"), valid_mask=mask)
    second = _sequence("b", phase=0.7)
    trained = train_sshead(
        (first, second),
        config,
        model=model,
        device="cpu",
        microbatch_size=2,
        checkpoint_path=tmp_path / "temporal-head.pt",
        progress_path=tmp_path / "temporal-head.jsonl",
    )
    assert isinstance(trained.model.period_head, TemporalPeriodHead)
    assert torch.count_nonzero(
        trained.model.period_head.temporal_conv.weight
    ) > 0
    assert len(observed_masks) == 1
    assert any(torch.equal(row, torch.from_numpy(mask)) for row in observed_masks[0])

    result = predict_sequence(trained.model, first, config, device="cpu")
    hook.remove()
    assert len(observed_masks) == 2
    assert torch.equal(observed_masks[-1][0], torch.from_numpy(mask))
    assert np.count_nonzero(result.period_stream[~mask]) == 0


def test_encoder_routes_cross_scale_denominator_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import training as training_module

    payload = _tiny_config(encoder_epochs=1).model_dump()
    payload["loss"]["exclude_other_scale_positives_from_denominator"] = True
    config = PAMSConfig.model_validate(payload)
    observed: list[bool] = []
    real_loss = training_module.PAMSTCCLoss

    class CapturingLoss(real_loss):
        def __init__(
            self,
            *args: Any,
            exclude_other_scale_positives_from_denominator: bool = False,
            **kwargs: Any,
        ) -> None:
            observed.append(exclude_other_scale_positives_from_denominator)
            super().__init__(
                *args,
                exclude_other_scale_positives_from_denominator=(
                    exclude_other_scale_positives_from_denominator
                ),
                **kwargs,
            )

    monkeypatch.setattr(training_module, "PAMSTCCLoss", CapturingLoss)
    train_encoder(
        (_sequence("a"), _sequence("b", phase=0.4)),
        config,
        device="cpu",
        microbatch_size=2,
    )

    assert observed == [True]


def test_encoder_rejects_gradient_accumulation_as_contrastive_batch_substitute() -> None:
    items = (_sequence("a"), _sequence("b", phase=0.4))
    config = _tiny_config(encoder_epochs=1)

    with pytest.raises(
        ValueError,
        match=r"physical microbatch_size == effective_batch_size.*"
        r"gradient accumulation cannot preserve.*cross-video.*cross-cluster",
    ):
        train_encoder(
            items,
            config,
            device="cpu",
            microbatch_size=1,
        )


def test_encoder_drops_incomplete_tail_and_records_zero_period_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import training as training_module

    def zero_confidence(
        poses: torch.Tensor,
        minimum: int,
        maximum: int,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del maximum, valid_mask
        batch = poses.shape[0]
        return (
            torch.full((batch,), minimum, dtype=torch.long, device=poses.device),
            torch.zeros(batch, dtype=poses.dtype, device=poses.device),
        )

    monkeypatch.setattr(training_module, "estimate_period_from_pose", zero_confidence)
    items = (
        _sequence("a"),
        _sequence("b", phase=0.4),
        _sequence("c", phase=0.8),
    )
    result = train_encoder(
        items,
        _tiny_config(encoder_epochs=1),
        device="cpu",
        microbatch_size=2,
    )
    assert result.history[0].optimizer_steps == 1
    assert result.history[0].period_confidence_mean == 0.0
    assert result.history[0].period_valid_fraction == 0.0


def test_encoder_strict_mode_fails_on_prototype_bank_shortfall(
    tmp_path: Path,
) -> None:
    items = (_sequence("a"), _sequence("b", phase=0.4))
    config = _tiny_config(encoder_epochs=1)
    model = build_pams_model(config)
    encoder_before = {
        name: value.detach().clone() for name, value in model.encoder.state_dict().items()
    }
    checkpoint = tmp_path / "strict.pt"
    progress = tmp_path / "strict.jsonl"
    with pytest.raises(
        RuntimeError,
        match=r"prototype bank shortfall before optimizer\.step",
    ):
        train_encoder(
            items,
            config,
            model=model,
            device="cpu",
            microbatch_size=2,
            checkpoint_path=checkpoint,
            progress_path=progress,
            allow_negative_shortfall=False,
        )
    assert not checkpoint.exists()
    assert not progress.exists()
    for name, expected in encoder_before.items():
        assert torch.equal(expected, model.encoder.state_dict()[name])


@pytest.mark.parametrize(
    ("failure_case", "message"),
    (
        ("embedding", r"non-finite embeddings before optimizer\.step"),
        ("loss", r"non-finite loss before optimizer\.step"),
        (
            "gradient",
            r"non-finite gradient for parameter 'input_projection\.weight' "
            r"before optimizer\.step",
        ),
    ),
)
def test_encoder_non_finite_values_are_blocked_before_step_or_artifact_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_case: str,
    message: str,
) -> None:
    from pams import training as training_module

    config = _tiny_config(encoder_epochs=1)
    model = build_pams_model(config)
    encoder_before = {
        name: value.detach().clone() for name, value in model.encoder.state_dict().items()
    }

    if failure_case == "embedding":
        original_forward = model.encoder.forward
        forward_calls = 0

        def poisoned_forward(
            inputs: torch.Tensor,
            valid_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            nonlocal forward_calls
            forward_calls += 1
            embeddings = original_forward(inputs, valid_mask)
            if forward_calls > 1:
                embeddings = embeddings.clone()
                embeddings[0, 0, 0] = float("nan")
            return embeddings

        monkeypatch.setattr(model.encoder, "forward", poisoned_forward)
    elif failure_case == "loss":
        original_compute = training_module.PAMSTCCLoss.compute

        def poisoned_compute(*args: Any, **kwargs: Any) -> Any:
            details = original_compute(*args, **kwargs)
            embeddings = args[1]
            return replace(
                details,
                total=embeddings.sum() * embeddings.new_tensor(float("nan")),
            )

        monkeypatch.setattr(training_module.PAMSTCCLoss, "compute", poisoned_compute)
    else:
        parameter = dict(model.encoder.named_parameters())["input_projection.weight"]
        parameter.register_hook(
            lambda gradient: torch.full_like(gradient, float("nan"))
        )

    optimizer_step_called = False
    original_step = training_module.AdamW.step

    def tracking_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal optimizer_step_called
        optimizer_step_called = True
        return original_step(*args, **kwargs)

    monkeypatch.setattr(training_module.AdamW, "step", tracking_step)
    checkpoint = tmp_path / f"{failure_case}.pt"
    progress = tmp_path / f"{failure_case}.jsonl"
    with pytest.raises(RuntimeError, match=message):
        train_encoder(
            (_sequence("a"), _sequence("b", phase=0.4)),
            config,
            model=model,
            device="cpu",
            microbatch_size=2,
            checkpoint_path=checkpoint,
            progress_path=progress,
        )

    assert not optimizer_step_called
    assert not checkpoint.exists()
    assert not progress.exists()
    for name, expected in encoder_before.items():
        assert torch.equal(expected, model.encoder.state_dict()[name])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_encoder_resume_loads_rng_on_cpu_before_device_migration(
    tmp_path: Path,
) -> None:
    items = (_sequence("a"), _sequence("b", phase=0.4))
    config = _tiny_config(encoder_epochs=2)
    checkpoint = tmp_path / "encoder-cuda.pt"
    partial = train_encoder(
        items,
        config,
        device="cuda",
        microbatch_size=2,
        checkpoint_path=checkpoint,
        stop_after_epoch=1,
    )
    assert partial.completed_epochs == 1

    resumed = train_encoder(
        items,
        config,
        device="cuda",
        microbatch_size=2,
        checkpoint_path=checkpoint,
        resume=True,
    )
    assert resumed.completed_epochs == 2
    assert next(resumed.model.parameters()).device.type == "cuda"


def test_sshead_keeps_encoder_frozen_and_prediction_is_well_formed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import training as training_module

    def zero_confidence(
        embeddings: torch.Tensor,
        minimum: int,
        maximum: int,
        valid_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del maximum, valid_mask
        batch = embeddings.shape[0]
        return (
            torch.full(
                (batch,),
                minimum,
                dtype=torch.long,
                device=embeddings.device,
            ),
            torch.zeros(
                batch,
                dtype=embeddings.dtype,
                device=embeddings.device,
            ),
        )

    monkeypatch.setattr(
        training_module,
        "estimate_period_from_embeddings",
        zero_confidence,
    )
    config = _tiny_config(encoder_epochs=1, head_epochs=2)
    model = build_pams_model(config)
    encoder_before = {
        name: value.detach().clone() for name, value in model.encoder.state_dict().items()
    }
    head_before = {
        name: value.detach().clone() for name, value in model.period_head.state_dict().items()
    }
    items = (_sequence("a"), _sequence("b", phase=0.7))
    checkpoint = tmp_path / "sshead.pt"
    progress = tmp_path / "sshead.jsonl"
    partial = train_sshead(
        items,
        config,
        model=model,
        device="cpu",
        microbatch_size=1,
        checkpoint_path=checkpoint,
        progress_path=progress,
        stop_after_epoch=1,
    )
    trained = train_sshead(
        items,
        config,
        model=partial.model,
        device="cpu",
        microbatch_size=1,
        checkpoint_path=checkpoint,
        progress_path=progress,
        resume=True,
    )
    assert trained.completed_epochs == 2
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
    rows = [json.loads(line) for line in progress.read_text(encoding="utf-8").splitlines()]
    assert [row["stage"] for row in rows] == ["sshead", "sshead"]
    assert [row["epoch"] for row in rows] == [1, 2]
    assert all(row["stats"]["period_confidence_mean"] == 0.0 for row in rows)
    assert all(row["stats"]["period_valid_fraction"] == 0.0 for row in rows)
    assert trained.history[0] == partial.history[0]
    monitor_fields = {
        "stream_std_min",
        "stream_std_p10",
        "stream_std_median",
        "stream_std_mean",
        "collapsed_fraction_1e6",
        "near_collapsed_fraction_1e3",
        "head_grad_rms_max",
        "head_grad_to_param_ratio_max",
        "zero_grad_steps",
    }
    assert all(monitor_fields <= set(row["stats"]) for row in rows)
    for stats in trained.history:
        monitored_values = (
            stats.stream_std_min,
            stats.stream_std_p10,
            stats.stream_std_median,
            stats.stream_std_mean,
            stats.collapsed_fraction_1e6,
            stats.near_collapsed_fraction_1e3,
            stats.head_grad_rms_max,
            stats.head_grad_to_param_ratio_max,
        )
        assert all(math.isfinite(value) for value in monitored_values)
        assert 0.0 <= stats.stream_std_min <= stats.stream_std_p10
        assert stats.stream_std_min <= stats.stream_std_median
        assert stats.stream_std_min <= stats.stream_std_mean
        assert 0.0 <= stats.collapsed_fraction_1e6 <= 1.0
        assert stats.collapsed_fraction_1e6 <= stats.near_collapsed_fraction_1e3 <= 1.0
        assert stats.head_grad_rms_max >= 0.0
        assert stats.head_grad_to_param_ratio_max >= 0.0
        assert 0 <= stats.zero_grad_steps <= stats.optimizer_steps


def test_sshead_exact_collapse_is_blocked_before_step_or_artifact_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import training as training_module

    config = _tiny_config(encoder_epochs=1, head_epochs=1)
    model = build_pams_model(config)
    with torch.no_grad():
        for parameter in model.period_head.parameters():
            parameter.zero_()

    optimizer_step_called = False
    original_step = training_module.AdamW.step

    def tracking_step(*args: Any, **kwargs: Any) -> Any:
        nonlocal optimizer_step_called
        optimizer_step_called = True
        return original_step(*args, **kwargs)

    monkeypatch.setattr(training_module.AdamW, "step", tracking_step)
    checkpoint = tmp_path / "collapsed.pt"
    progress = tmp_path / "collapsed.jsonl"
    with pytest.raises(RuntimeError, match="exact-collapse guard.*before optimizer.step"):
        train_sshead(
            (_sequence("a"), _sequence("b", phase=0.7)),
            config,
            model=model,
            device="cpu",
            microbatch_size=2,
            checkpoint_path=checkpoint,
            progress_path=progress,
        )

    assert not optimizer_step_called
    assert not checkpoint.exists()
    assert not progress.exists()


def test_sshead_near_collapse_gradient_spike_guard_helper() -> None:
    from pams import training as training_module

    parameter = torch.nn.Parameter(torch.full((4,), 1e-6))
    parameter.grad = torch.ones_like(parameter)
    diagnostics = training_module._head_gradient_diagnostics((parameter,))
    assert diagnostics.grad_rms == pytest.approx(1.0)
    assert diagnostics.grad_to_param_ratio > 100.0

    with pytest.raises(RuntimeError, match="near-collapse-gradient-spike guard"):
        training_module._guard_sshead_optimizer_step(
            loss_value=0.01,
            minimum_stream_std=1.0,
            diagnostics=diagnostics,
        )


def test_prediction_confidence_is_zero_without_spectral_period_evidence() -> None:
    config = _tiny_config(encoder_epochs=1, head_epochs=1)
    model = build_pams_model(config)
    with torch.no_grad():
        for parameter in model.period_head.parameters():
            parameter.zero_()

    result = predict_sequence(model, _sequence("constant-head"), config, device="cpu")

    assert result.confidence == 0.0


def test_bound_checkpoint_records_and_validates_canonical_provenance(
    tmp_path: Path,
) -> None:
    config = _tiny_config(encoder_epochs=1)
    items = (_sequence("a"), _sequence("b", phase=0.4))
    provenance = _provenance(config)
    checkpoint = tmp_path / "encoder.pt"

    train_encoder(
        items,
        config,
        device="cpu",
        microbatch_size=2,
        checkpoint_path=checkpoint,
        provenance=provenance,
    )

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["schema_version"] == 5
    assert payload["stage"] == "encoder"
    assert payload["provenance"] == provenance.to_dict()
    assert payload["provenance"]["training_video_ids"] == ["a", "b"]
    assert payload["prototype_bank"]["video_ids"] == ["a", "b"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        expected_initial_model = build_pams_model(config)
    for key, expected in expected_initial_model.period_head.state_dict().items():
        assert torch.equal(payload["model_state"][f"period_head.{key}"], expected)
    loaded = load_model_checkpoint(
        checkpoint,
        config,
        device="cpu",
        expected_stage="encoder",
        expected_provenance=provenance,
    )
    assert isinstance(loaded, type(build_pams_model(config)))

    with pytest.raises(ValueError, match="dataset_fingerprint"):
        load_model_checkpoint(
            checkpoint,
            config,
            device="cpu",
            expected_stage="encoder",
            expected_provenance=replace(
                provenance,
                dataset_fingerprint="e" * 64,
            ),
        )
    with pytest.raises(ValueError, match="training_video_ids"):
        load_model_checkpoint(
            checkpoint,
            config,
            device="cpu",
            expected_stage="encoder",
            expected_provenance=replace(
                provenance,
                training_video_ids=("a", "different"),
            ),
        )
    with pytest.raises(ValueError, match="pose_cache_set_sha256"):
        load_model_checkpoint(
            checkpoint,
            config,
            device="cpu",
            expected_stage="encoder",
            expected_provenance=replace(
                provenance,
                pose_cache_set_sha256="f" * 64,
            ),
        )
    with pytest.raises(ValueError, match="source_git_sha"):
        load_model_checkpoint(
            checkpoint,
            config,
            device="cpu",
            expected_stage="encoder",
            expected_provenance=replace(
                provenance,
                source_git_sha="b" * 40,
            ),
        )
    with pytest.raises(ValueError, match="caller must provide expected_provenance"):
        load_model_checkpoint(
            checkpoint,
            config,
            device="cpu",
            expected_stage="encoder",
        )


def test_terminal_checkpoint_requires_full_finite_history_and_exact_progress(
    tmp_path: Path,
) -> None:
    config = _tiny_config(encoder_epochs=2)
    checkpoint, progress, provenance, _model = _write_terminal_encoder_fixture(
        tmp_path,
        config,
    )
    validate_terminal_checkpoint(
        checkpoint,
        config,
        expected_stage="encoder",
        expected_provenance=provenance,
        progress_path=progress,
    )

    complete_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    complete_progress = progress.read_bytes()
    altered_head_payload = dict(complete_payload)
    altered_head_payload["model_state"] = dict(complete_payload["model_state"])
    head_key = next(
        key
        for key in altered_head_payload["model_state"]
        if key.startswith("period_head.")
    )
    altered_head_payload["model_state"][head_key] = (
        altered_head_payload["model_state"][head_key].clone() + 1.0
    )
    torch.save(altered_head_payload, checkpoint)
    with pytest.raises(ValueError, match="differs from deterministic initialization"):
        validate_terminal_checkpoint(
            checkpoint,
            config,
            expected_stage="encoder",
            expected_provenance=provenance,
            progress_path=progress,
        )

    partial_payload = dict(complete_payload)
    partial_payload["completed_epochs"] = 1
    partial_payload["history"] = partial_payload["history"][:1]
    torch.save(partial_payload, checkpoint)
    with pytest.raises(ValueError, match="checkpoint is partial"):
        validate_terminal_checkpoint(
            checkpoint,
            config,
            expected_stage="encoder",
            expected_provenance=provenance,
            progress_path=progress,
        )
    assert progress.read_bytes() == complete_progress

    non_finite_payload = dict(complete_payload)
    non_finite_payload["history"] = [dict(row) for row in complete_payload["history"]]
    non_finite_payload["history"][0]["loss"] = float("nan")
    torch.save(non_finite_payload, checkpoint)
    with pytest.raises(ValueError, match="non-finite"):
        validate_terminal_checkpoint(
            checkpoint,
            config,
            expected_stage="encoder",
            expected_provenance=provenance,
            progress_path=progress,
        )

    torch.save(complete_payload, checkpoint)
    first_line, *remaining = complete_progress.decode("utf-8").splitlines()
    duplicate_epoch = first_line.replace('"epoch":1', '"epoch":1,"epoch":1', 1)
    progress.write_text(
        "\n".join([duplicate_epoch, *remaining]) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        validate_terminal_checkpoint(
            checkpoint,
            config,
            expected_stage="encoder",
            expected_provenance=provenance,
            progress_path=progress,
        )


def test_terminal_sshead_binds_exact_upstream_encoder_tensors(
    tmp_path: Path,
) -> None:
    from pams import training as training_module

    config = _tiny_config(encoder_epochs=1, head_epochs=1)
    encoder, encoder_progress, encoder_provenance, model = (
        _write_terminal_encoder_fixture(tmp_path, config)
    )
    validate_terminal_checkpoint(
        encoder,
        config,
        expected_stage="encoder",
        expected_provenance=encoder_provenance,
        progress_path=encoder_progress,
    )
    encoder_sha256 = hashlib.sha256(encoder.read_bytes()).hexdigest()
    head_provenance = _provenance(
        config,
        upstream_encoder_checkpoint_sha256=encoder_sha256,
    )
    optimizer = training_module.AdamW(model.period_head.parameters(), lr=1e-3)
    statistics = training_module.SSHeadEpochStats(
        epoch=1,
        total=1.0,
        cycle=0.4,
        spectral=0.4,
        variance=0.1,
        smoothness=0.1,
        period_confidence_mean=0.8,
        period_valid_fraction=1.0,
        stream_std_min=0.2,
        stream_std_p10=0.25,
        stream_std_median=0.3,
        stream_std_mean=0.35,
        collapsed_fraction_1e6=0.0,
        near_collapsed_fraction_1e3=0.0,
        head_grad_rms_max=0.1,
        head_grad_to_param_ratio_max=0.1,
        zero_grad_steps=0,
        learning_rate=1e-3,
        optimizer_steps=1,
    )
    head = tmp_path / "sshead.pt"
    head_progress = tmp_path / "sshead.jsonl"
    head_payload = training_module._checkpoint_payload(
        stage="sshead",
        config=config,
        model=model,
        optimizer=optimizer,
        completed_epochs=1,
        history=[statistics],
        provenance=head_provenance,
    )
    torch.save(head_payload, head)
    progress_row = training_module._progress_row(
        stage="sshead",
        stats=statistics,
        config=config,
    )
    head_progress.write_text(
        json.dumps(
            progress_row,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    validate_terminal_checkpoint(
        head,
        config,
        expected_stage="sshead",
        expected_provenance=head_provenance,
        progress_path=head_progress,
    )
    validate_sshead_encoder_binding(head, encoder)

    altered = tmp_path / "altered-sshead.pt"
    altered_payload = torch.load(head, map_location="cpu", weights_only=False)
    encoder_key = next(
        key for key in altered_payload["model_state"] if key.startswith("encoder.")
    )
    altered_payload["model_state"][encoder_key] = (
        altered_payload["model_state"][encoder_key].clone() + 1.0
    )
    torch.save(altered_payload, altered)
    with pytest.raises(ValueError, match="differs from upstream"):
        validate_sshead_encoder_binding(altered, encoder)


def test_terminal_encoder_strictly_validates_projected_vector_period_source(
    tmp_path: Path,
) -> None:
    payload = _with_projected_vector_period(
        _tiny_config(encoder_epochs=11, head_epochs=1)
    ).model_dump()
    payload["period"]["pose_energy_epochs"] = 10
    config = PAMSConfig.model_validate(payload)
    checkpoint, progress, provenance, _ = _write_terminal_encoder_fixture(
        tmp_path,
        config,
    )

    validate_terminal_checkpoint(
        checkpoint,
        config,
        expected_stage="encoder",
        expected_provenance=provenance,
        progress_path=progress,
    )
    rows = [
        json.loads(line)
        for line in progress.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["stats"]["period_source"] for row in rows[:10]] == ["pose"] * 10
    assert (
        rows[10]["stats"]["period_source"]
        == "projected_pose_velocity_vector_acf"
    )

    tampered = tmp_path / "tampered-period-source.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["history"][-1]["period_source"] = "embedding"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="period-source schedule mismatch"):
        validate_terminal_checkpoint(
            tampered,
            config,
            expected_stage="encoder",
            expected_provenance=provenance,
            progress_path=progress,
        )


def test_terminal_encoder_strictly_validates_fixed_period_proxy(
    tmp_path: Path,
) -> None:
    config = _with_fixed_period(_tiny_config(encoder_epochs=1), frames=4)
    checkpoint, progress, provenance, _ = _write_terminal_encoder_fixture(
        tmp_path,
        config,
    )

    validate_terminal_checkpoint(
        checkpoint,
        config,
        expected_stage="encoder",
        expected_provenance=provenance,
        progress_path=progress,
    )
    progress_row = json.loads(progress.read_text(encoding="utf-8"))
    assert progress_row["stats"]["period_source"] == "fixed_period_inferred"
    assert progress_row["stats"]["period_training_mode"] == "fixed_period_inferred"
    assert progress_row["stats"]["fixed_period_frames"] == 4

    tampered = tmp_path / "tampered-fixed-period-source.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["history"][0]["period_source"] = "pose"
    torch.save(payload, tampered)
    with pytest.raises(ValueError, match="period-source schedule mismatch"):
        validate_terminal_checkpoint(
            tampered,
            config,
            expected_stage="encoder",
            expected_provenance=provenance,
            progress_path=progress,
        )


def test_resume_rejects_provenance_mismatch(tmp_path: Path) -> None:
    config = _tiny_config(encoder_epochs=2)
    items = (_sequence("a"), _sequence("b", phase=0.4))
    checkpoint = tmp_path / "encoder.pt"
    provenance = _provenance(config)
    train_encoder(
        items,
        config,
        device="cpu",
        microbatch_size=2,
        checkpoint_path=checkpoint,
        stop_after_epoch=1,
        provenance=provenance,
    )

    with pytest.raises(ValueError, match="dataset_fingerprint"):
        train_encoder(
            items,
            config,
            device="cpu",
            microbatch_size=2,
            checkpoint_path=checkpoint,
            resume=True,
            provenance=replace(provenance, dataset_fingerprint="e" * 64),
        )


def test_progress_resume_rejects_tampering_and_repairs_one_missing_tail(
    tmp_path: Path,
) -> None:
    config = _tiny_config(encoder_epochs=2)
    items = (_sequence("a"), _sequence("b", phase=0.4))
    checkpoint = tmp_path / "encoder.pt"
    progress = tmp_path / "encoder.jsonl"
    train_encoder(
        items,
        config,
        device="cpu",
        microbatch_size=2,
        checkpoint_path=checkpoint,
        progress_path=progress,
        stop_after_epoch=1,
    )
    original_line = progress.read_text(encoding="utf-8").strip()
    tampered = json.loads(original_line)
    tampered["stats"]["loss"] = float(tampered["stats"]["loss"]) + 1.0
    progress.write_text(json.dumps(tampered) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="progress log stats mismatch"):
        train_encoder(
            items,
            config,
            device="cpu",
            microbatch_size=2,
            checkpoint_path=checkpoint,
            progress_path=progress,
            resume=True,
        )

    # Simulate interruption after the atomic checkpoint write but before its
    # one corresponding JSONL append. Resume may repair only this final row.
    progress.write_text("", encoding="utf-8")
    resumed = train_encoder(
        items,
        config,
        device="cpu",
        microbatch_size=2,
        checkpoint_path=checkpoint,
        progress_path=progress,
        resume=True,
    )
    assert resumed.completed_epochs == 2
    lines = progress.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["epoch"] for line in lines] == [1, 2]

    with progress.open("a", encoding="utf-8") as handle:
        handle.write(lines[-1] + "\n")
    with pytest.raises(ValueError, match="more epochs than checkpoint history"):
        train_encoder(
            items,
            config,
            device="cpu",
            microbatch_size=2,
            checkpoint_path=checkpoint,
            progress_path=progress,
            resume=True,
        )


def test_checkpoint_destination_mode_is_strict_for_both_stages(tmp_path: Path) -> None:
    config = _tiny_config(encoder_epochs=1, head_epochs=1)
    items = (_sequence("a"), _sequence("b", phase=0.4))
    existing = tmp_path / "existing.pt"
    existing.write_bytes(b"do-not-overwrite")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        train_encoder(
            items,
            config,
            device="cpu",
            microbatch_size=2,
            checkpoint_path=existing,
        )
    assert existing.read_bytes() == b"do-not-overwrite"

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        train_sshead(
            items,
            config,
            model=build_pams_model(config),
            device="cpu",
            microbatch_size=1,
            checkpoint_path=existing,
        )
    assert existing.read_bytes() == b"do-not-overwrite"

    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        train_encoder(
            items,
            config,
            device="cpu",
            microbatch_size=2,
            checkpoint_path=missing,
            resume=True,
        )
    with pytest.raises(FileNotFoundError, match="checkpoint does not exist"):
        train_sshead(
            items,
            config,
            model=build_pams_model(config),
            device="cpu",
            microbatch_size=1,
            checkpoint_path=missing,
            resume=True,
        )

    progress = tmp_path / "existing.jsonl"
    progress.write_text("do-not-overwrite\n", encoding="utf-8")
    new_checkpoint = tmp_path / "new.pt"
    with pytest.raises(FileExistsError, match="refusing to overwrite existing progress log"):
        train_encoder(
            items,
            config,
            device="cpu",
            microbatch_size=2,
            checkpoint_path=new_checkpoint,
            progress_path=progress,
        )
    assert not new_checkpoint.exists()
    assert progress.read_text(encoding="utf-8") == "do-not-overwrite\n"


def test_sshead_checkpoint_binds_upstream_encoder_hash(tmp_path: Path) -> None:
    config = _tiny_config(encoder_epochs=1, head_epochs=1)
    items = (_sequence("a"), _sequence("b", phase=0.4))
    provenance = _provenance(
        config,
        upstream_encoder_checkpoint_sha256="a" * 64,
    )
    checkpoint = tmp_path / "sshead.pt"
    train_sshead(
        items,
        config,
        model=build_pams_model(config),
        device="cpu",
        microbatch_size=1,
        checkpoint_path=checkpoint,
        provenance=provenance,
    )

    load_model_checkpoint(
        checkpoint,
        config,
        device="cpu",
        expected_stage="sshead",
        expected_provenance=provenance,
    )
    with pytest.raises(ValueError, match="upstream_encoder_checkpoint_sha256"):
        load_model_checkpoint(
            checkpoint,
            config,
            device="cpu",
            expected_stage="sshead",
            expected_provenance=replace(
                provenance,
                upstream_encoder_checkpoint_sha256="b" * 64,
            ),
        )
