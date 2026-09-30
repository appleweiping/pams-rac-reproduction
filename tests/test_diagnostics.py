from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from typer.testing import CliRunner

from pams import cli as cli_module
from pams.cli import app
from pams.config import (
    DataConfig,
    LossConfig,
    ModelConfig,
    PAMSConfig,
    PeriodConfig,
    TrainingConfig,
)
from pams.data import (
    PoseCacheSetSnapshot,
    load_pose_cache_with_receipt,
    pose_cache_path,
    write_pose_cache,
)
from pams.diagnostics import run_encoder_shortcut_diagnostic
from pams.training import CheckpointProvenance, build_pams_model
from pams.types import PoseSequence


def _tiny_config() -> PAMSConfig:
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
            scales=(0.5, 1.0, 1.5),
            temperature=0.1,
            kmeans_clusters=2,
            kmeans_refresh_epochs=1,
        ),
        training=TrainingConfig(
            epochs=1,
            effective_batch_size=2,
            learning_rate=1e-3,
            weight_decay=0.0,
            scheduler_factor=0.5,
            scheduler_patience=1,
            minimum_learning_rate=1e-6,
        ),
    )


def _sequence(identifier: str, phase_offset: float) -> PoseSequence:
    frame = np.arange(16, dtype=np.float32)
    phase = 2.0 * np.pi * frame / 4.0 + phase_offset
    xyz = np.zeros((16, 33, 3), dtype=np.float32)
    xyz[:, :, 0] = (0.5 + 0.5 * np.sin(phase))[:, None]
    xyz[:, :, 1] = (0.5 + 0.5 * np.cos(phase))[:, None]
    xyz[:, :, 2] = np.linspace(0.0, 1.0, 16, dtype=np.float32)[:, None]
    return PoseSequence(
        video_id=identifier,
        fps=16.0,
        xyz=xyz,
        valid_mask=np.ones(16, dtype=np.bool_),
    )


def _diagnostic_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, PAMSConfig, tuple[str, ...]]:
    from pams import training as training_module

    config = _tiny_config()
    config_path = tmp_path / "diagnostic.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    cache_dir = tmp_path / "pose-cache"
    cache_dir.mkdir()
    identifiers = tuple(f"private-training-video-{index}" for index in range(4))
    receipts = []
    for index, identifier in enumerate(identifiers):
        write_pose_cache(
            pose_cache_path(cache_dir, identifier),
            _sequence(identifier, phase_offset=index * 0.2),
            video_sha256=hashlib.sha256(identifier.encode()).hexdigest(),
            pose_fingerprint=config.pose_fingerprint,
        )
        _, _, receipt = load_pose_cache_with_receipt(
            pose_cache_path(cache_dir, identifier),
            expected_pose_fingerprint=config.pose_fingerprint,
        )
        receipts.append(receipt)
    snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=config.pose_fingerprint,
        entries=tuple(receipts),
    )
    provenance = CheckpointProvenance(
        protocol=config.protocol,
        dataset_fingerprint="d" * 64,
        training_video_ids=identifiers,
        pose_fingerprint=config.pose_fingerprint,
        pose_cache_set_sha256=snapshot.fingerprint,
        source_git_sha="a" * 40,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = build_pams_model(config)
    optimizer = training_module.AdamW(model.encoder.parameters(), lr=1e-3)
    scheduler = training_module.ReduceLROnPlateau(optimizer)
    payload = training_module._checkpoint_payload(
        stage="encoder",
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        completed_epochs=0,
        history=(),
        provenance=provenance,
    )
    checkpoint = tmp_path / "encoder.pt"
    torch.save(payload, checkpoint)
    return checkpoint, config_path, cache_dir, config, identifiers


def _all_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def test_encoder_shortcut_diagnostic_is_label_free_read_only_and_complete(
    tmp_path: Path,
) -> None:
    checkpoint, config_path, cache_dir, config, identifiers = _diagnostic_fixture(tmp_path)
    checkpoint_before = checkpoint.read_bytes()
    config_before = config_path.read_bytes()

    payload = run_encoder_shortcut_diagnostic(
        checkpoint,
        config,
        config_path=config_path,
        pose_cache_dir=cache_dir,
        sample_size=2,
        seed=91,
        device="cpu",
        batch_size=2,
    )

    assert payload["classification"] == "inferred diagnostic"
    assert payload["disclosed_by_pams_authors"] is False
    assert payload["eligible_for_paper_table"] is False
    assert payload["label_firewall"] == {
        "input_surface": ["checkpoint", "config", "pose_cache_directory"],
        "dataset_manifest_argument_supported": False,
        "dataset_manifest_loaded": False,
        "label_fields_accessed": [],
        "checkpoint_training_ids_are_only_used_for_cache_lookup_and_hash_splits": True,
    }
    inputs = payload["inputs"]
    assert inputs["sampled_training_video_total"] == 2
    assert inputs["all_checkpoint_pose_caches_hashed"] is True
    assert inputs["full_checkpoint_pose_cache_set_selected"] is False
    assert inputs["full_checkpoint_pose_cache_set_verified"] is True
    assert len(inputs["sampled_training_video_ids_sha256"]) == 64

    assert payload["frame_norms"]["positional_encoding_frame_norm"]["observations"] == 16
    assert (
        payload["frame_norms"]["pose_projection_frame_norm_effective"]["observations"]
        == 32
    )
    probes = payload["zero_and_random_pose_period_probes"]
    assert set(probes["zero_pose"]) == {
        "embedding_period_frames",
        "embedding_period_confidence",
        "pose_signal_period_frames",
        "pose_signal_period_confidence",
    }
    histogram = payload["training_embedding_periods"]["histogram"]
    assert sum(histogram["nonzero_bin_frequencies"].values()) == 2
    assert 0.0 <= histogram["top_bin_share"] <= 1.0
    assert payload["frame_index_linear_probe"]["available"] is True
    assert payload["frame_index_linear_probe"]["r2"] is not None
    correspondence = payload["multiscale_correspondence"]
    assert set(correspondence["boundary_by_scale"]) == {"0.5", "1", "1.5"}
    conflict = correspondence["positive_as_negative_conflict"]
    assert 0.0 <= conflict["rate"] <= 1.0
    orthogonal = payload["orthogonal_basis_period_sensitivity"]
    assert 0.0 <= orthogonal["rounded_period_unchanged_share"] <= 1.0
    assert payload["read_only_verification"]["model_or_training_state_updated"] is False
    assert checkpoint.read_bytes() == checkpoint_before
    assert config_path.read_bytes() == config_before
    assert _all_finite(payload)

    serialized = json.dumps(payload, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert all(identifier not in serialized for identifier in identifiers)


def test_encoder_shortcut_cli_is_explicitly_inferred_and_has_no_manifest_input(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    checkpoint, config_path, cache_dir, _, _ = _diagnostic_fixture(tmp_path)
    output = tmp_path / "diagnostic.json"
    observed: dict[str, Any] = {}

    def fake_run(*args: Any, **kwargs: Any) -> dict[str, Any]:
        observed["args"] = args
        observed["kwargs"] = kwargs
        return {
            "diagnostic_id": "fixture-inferred",
            "classification": "inferred diagnostic",
            "dataset_manifest_loaded": False,
        }

    def forbidden_manifest_loader(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("diagnostic must not load a dataset manifest")

    monkeypatch.setattr(
        "pams.diagnostics.run_encoder_shortcut_diagnostic",
        fake_run,
    )
    monkeypatch.setattr(cli_module, "load_ucfrep_manifest", forbidden_manifest_loader)
    result = CliRunner().invoke(
        app,
        [
            "diagnostic",
            "encoder-shortcut-inferred",
            str(checkpoint),
            "--config",
            str(config_path),
            "--pose-cache-dir",
            str(cache_dir),
            "--sample-size",
            "2",
            "--device",
            "cpu",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["classification"] == "inferred diagnostic"
    assert json.loads(output.read_text(encoding="utf-8"))["diagnostic_id"] == (
        "fixture-inferred"
    )
    assert observed["kwargs"]["sample_size"] == 2
    assert "manifest" not in observed["kwargs"]
