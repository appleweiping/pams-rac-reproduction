from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from pams.config import PAMSConfig, load_config


def test_default_config_matches_disclosed_dimensions() -> None:
    config = PAMSConfig()
    assert config.model.input_dim == 33 * 3
    assert config.model.model_dim == 512
    assert len(config.fingerprint) == 64
    assert len(config.pose_fingerprint) == 64
    assert config.pose.model_id == "mediapipe-pose-0.10.14"
    assert config.pose.preprocessing_revision == "detected-span-minmax-zero-span-invalid-v2"


def test_config_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValidationError, match="input_dim"):
        PAMSConfig.model_validate({"model": {"input_dim": 98}})


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump({"unknown": True}), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_repository_config_loads() -> None:
    path = Path(__file__).parents[1] / "configs" / "pams.yaml"
    config = load_config(path)
    assert config.loss.scales == (0.5, 1.0, 1.5)
    assert config.pose.crop_to_detected_span is True
    assert config.model.input_projection_scale == "none"


def test_inferred_pe_scale_experiment_changes_only_projection_scale() -> None:
    root = Path(__file__).parents[1]
    formal = load_config(root / "configs" / "pams.yaml")
    experiment = load_config(
        root / "configs" / "experiments" / "pams_pe_scale_v2.yaml"
    )

    assert experiment.seed == 2026
    assert experiment.model.input_projection_scale == "sqrt_model_dim"
    assert experiment.fingerprint != formal.fingerprint
    assert experiment.nonseed_fingerprint != formal.nonseed_fingerprint
    experiment_payload = experiment.model_dump()
    experiment_payload["model"]["input_projection_scale"] = "none"
    assert experiment_payload == formal.model_dump()


def test_config_rejects_unknown_input_projection_scale() -> None:
    with pytest.raises(ValidationError, match="input_projection_scale"):
        PAMSConfig.model_validate(
            {"model": {"input_projection_scale": "sqrt_input_dim"}}
        )


def test_encoder_smoke_config_changes_only_training_epoch_count() -> None:
    root = Path(__file__).parents[1]
    formal = load_config(root / "configs" / "pams.yaml")
    smoke = load_config(root / "configs" / "smoke" / "pams_encoder_1epoch.yaml")

    assert smoke.training.epochs == 1
    smoke_payload = smoke.model_dump()
    smoke_payload["training"]["epochs"] = formal.training.epochs
    assert smoke_payload == formal.model_dump()


def test_two_stage_smoke_config_changes_only_both_epoch_counts() -> None:
    root = Path(__file__).parents[1]
    formal = load_config(root / "configs" / "pams.yaml")
    smoke = load_config(root / "configs" / "smoke" / "pams_two_stage_1epoch.yaml")

    assert smoke.training.epochs == 1
    assert smoke.sshead.epochs == 1
    smoke_payload = smoke.model_dump()
    smoke_payload["training"]["epochs"] = formal.training.epochs
    smoke_payload["sshead"]["epochs"] = formal.sshead.epochs
    assert smoke_payload == formal.model_dump()


def test_pose_fingerprint_ignores_seed_and_training_hyperparameters() -> None:
    fingerprints = {PAMSConfig(seed=seed).pose_fingerprint for seed in (42, 2026, 3407)}
    assert len(fingerprints) == 1
    nonseed_fingerprints = {
        PAMSConfig(seed=seed).nonseed_fingerprint for seed in (42, 2026, 3407)
    }
    assert nonseed_fingerprints == {
        "2c995b374bc8cf97df3568d745dabd94f111aa54224c86ece51468cbe419b47b"
    }

    base = PAMSConfig()
    changed_training = PAMSConfig.model_validate(
        {
            "seed": 42,
            "training": {
                **base.training.model_dump(),
                "learning_rate": 2e-4,
            },
        }
    )
    assert changed_training.pose_fingerprint == base.pose_fingerprint
    assert changed_training.fingerprint != base.fingerprint
    assert changed_training.nonseed_fingerprint != base.nonseed_fingerprint


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_id", "mediapipe-pose-future"),
        ("model_complexity", 2),
        ("smooth_landmarks", False),
        ("min_detection_confidence", 0.6),
        ("min_tracking_confidence", 0.6),
        ("crop_to_detected_span", False),
    ],
)
def test_every_pose_extractor_setting_changes_pose_fingerprint(
    field: str,
    value: object,
) -> None:
    base = PAMSConfig()
    pose = base.pose.model_dump()
    pose[field] = value
    changed = PAMSConfig.model_validate({"pose": pose})
    assert changed.pose_fingerprint != base.pose_fingerprint


def test_pose_preprocessing_revision_is_frozen() -> None:
    with pytest.raises(ValidationError, match="preprocessing_revision"):
        PAMSConfig.model_validate(
            {"pose": {"preprocessing_revision": "unsupported-revision"}}
        )


def test_data_preprocessing_changes_pose_fingerprint() -> None:
    base = PAMSConfig()
    data = base.data.model_dump()
    data["frames"] = 128
    changed = PAMSConfig.model_validate({"data": data})
    assert changed.pose_fingerprint != base.pose_fingerprint


@pytest.mark.parametrize(
    "data",
    [
        {"normalization": "zscore"},
        {"missing_value": -1.0},
    ],
)
def test_unimplemented_data_semantics_are_rejected(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        PAMSConfig.model_validate({"data": data})
