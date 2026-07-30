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
    assert config.period.training_mode == "adaptive"
    assert config.period.fixed_period_frames == 16
    assert config.period.post_warmup_source == "embedding_velocity_coordinate"
    assert config.loss.exclude_other_scale_positives_from_denominator is False
    assert config.sshead.architecture == "pointwise_mlp"
    assert config.sshead.input_source == "encoder_embedding"
    assert config.sshead.period_confidence_mode == "nonzero_gate"
    assert config.consensus.expert_mode == "multi"


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


def test_inferred_projected_vector_period_changes_only_post_warmup_source() -> None:
    root = Path(__file__).parents[1]
    formal = load_config(root / "configs" / "pams.yaml")
    experiment = load_config(
        root
        / "configs"
        / "experiments"
        / "pams_projected_vector_period_v3.yaml"
    )

    assert experiment.seed == 2026
    assert experiment.model.input_projection_scale == "none"
    assert (
        experiment.period.post_warmup_source
        == "projected_pose_velocity_vector_acf"
    )
    assert experiment.fingerprint != formal.fingerprint
    assert experiment.nonseed_fingerprint != formal.nonseed_fingerprint
    experiment_payload = experiment.model_dump()
    experiment_payload["period"][
        "post_warmup_source"
    ] = "embedding_velocity_coordinate"
    assert experiment_payload == formal.model_dump()


def test_explicit_default_period_source_preserves_historical_fingerprint() -> None:
    implicit = PAMSConfig()
    payload = implicit.model_dump()
    payload["period"].pop("post_warmup_source")
    explicit = PAMSConfig.model_validate(implicit.model_dump())
    reconstructed_implicit = PAMSConfig.model_validate(payload)

    assert explicit.fingerprint == reconstructed_implicit.fingerprint
    assert explicit.nonseed_fingerprint == reconstructed_implicit.nonseed_fingerprint


def test_inferred_pre_pe_head_is_explicit_identity_and_preserves_pose_cache() -> None:
    root = Path(__file__).parents[1]
    period_only = load_config(
        root
        / "configs"
        / "experiments"
        / "pams_projected_vector_period_v3.yaml"
    )
    repair = load_config(
        root / "configs" / "experiments" / "pams_sshead_pre_pe_v5.yaml"
    )
    implicit_payload = period_only.model_dump()
    implicit_payload["sshead"].pop("input_source")
    reconstructed_implicit = PAMSConfig.model_validate(implicit_payload)

    assert repair.sshead.input_source == "projected_pose_pre_pe"
    assert repair.period == period_only.period
    assert repair.fingerprint != period_only.fingerprint
    assert repair.nonseed_fingerprint != period_only.nonseed_fingerprint
    assert repair.pose_fingerprint == period_only.pose_fingerprint
    assert reconstructed_implicit.fingerprint == period_only.fingerprint


def test_inferred_confidence_weighting_changes_only_sshead_confidence_mode() -> None:
    root = Path(__file__).parents[1]
    pre_pe = load_config(
        root / "configs" / "experiments" / "pams_sshead_pre_pe_v5.yaml"
    )
    weighted = load_config(
        root
        / "configs"
        / "experiments"
        / "pams_sshead_confidence_weighted_v6.yaml"
    )
    implicit_payload = pre_pe.model_dump()
    implicit_payload["sshead"].pop("period_confidence_mode")
    reconstructed_implicit = PAMSConfig.model_validate(implicit_payload)

    assert weighted.sshead.period_confidence_mode == "normalized_weight"
    weighted_payload = weighted.model_dump()
    weighted_payload["sshead"]["period_confidence_mode"] = "nonzero_gate"
    assert weighted_payload == pre_pe.model_dump()
    assert weighted.fingerprint != pre_pe.fingerprint
    assert weighted.nonseed_fingerprint != pre_pe.nonseed_fingerprint
    assert weighted.pose_fingerprint == pre_pe.pose_fingerprint
    assert reconstructed_implicit.fingerprint == pre_pe.fingerprint


def test_inferred_temporal_head_changes_only_sshead_architecture() -> None:
    root = Path(__file__).parents[1]
    weighted = load_config(
        root
        / "configs"
        / "experiments"
        / "pams_sshead_confidence_weighted_v6.yaml"
    )
    temporal = load_config(
        root / "configs" / "experiments" / "pams_sshead_temporal_conv_v7.yaml"
    )
    implicit_payload = weighted.model_dump()
    implicit_payload["sshead"].pop("architecture")
    reconstructed_implicit = PAMSConfig.model_validate(implicit_payload)

    assert temporal.sshead.architecture == "temporal_conv"
    temporal_payload = temporal.model_dump()
    temporal_payload["sshead"]["architecture"] = "pointwise_mlp"
    assert temporal_payload == weighted.model_dump()
    assert temporal.fingerprint != weighted.fingerprint
    assert temporal.nonseed_fingerprint != weighted.nonseed_fingerprint
    assert temporal.pose_fingerprint == weighted.pose_fingerprint
    assert reconstructed_implicit.fingerprint == weighted.fingerprint


def test_longest_track_v8_changes_only_pose_preprocessing_revision() -> None:
    root = Path(__file__).parents[1]
    temporal = load_config(
        root / "configs" / "experiments" / "pams_sshead_temporal_conv_v7.yaml"
    )
    corrected = load_config(
        root / "configs" / "experiments" / "pams_longest_contiguous_track_v8.yaml"
    )

    assert (
        corrected.pose.preprocessing_revision
        == "longest-contiguous-track-minmax-zero-span-invalid-v3"
    )
    corrected_payload = corrected.model_dump()
    corrected_payload["pose"][
        "preprocessing_revision"
    ] = "detected-span-minmax-zero-span-invalid-v2"
    assert corrected_payload == temporal.model_dump()
    assert corrected.pose_fingerprint != temporal.pose_fingerprint
    assert corrected.fingerprint != temporal.fingerprint
    assert corrected.nonseed_fingerprint != temporal.nonseed_fingerprint


def test_fixed_period_proxy_is_explicit_identity_and_default_preserves_history() -> None:
    root = Path(__file__).parents[1]
    formal = load_config(root / "configs" / "pams.yaml")
    fixed = load_config(
        root / "configs" / "ablations" / "pams_fixed_period16_inferred.yaml"
    )
    implicit_payload = formal.model_dump()
    implicit_payload["period"].pop("training_mode")
    implicit_payload["period"].pop("fixed_period_frames")
    reconstructed_implicit = PAMSConfig.model_validate(implicit_payload)

    assert formal.fingerprint == reconstructed_implicit.fingerprint
    assert formal.nonseed_fingerprint == reconstructed_implicit.nonseed_fingerprint
    assert fixed.period.training_mode == "fixed_period_inferred"
    assert fixed.period.fixed_period_frames == 16
    assert fixed.loss.scales == (1.0,)
    assert fixed.fingerprint != formal.fingerprint
    assert fixed.nonseed_fingerprint != formal.nonseed_fingerprint
    assert fixed.pose_fingerprint == formal.pose_fingerprint


def test_fixed_period_proxy_must_stay_inside_period_bounds() -> None:
    with pytest.raises(ValidationError, match="fixed_period_frames"):
        PAMSConfig.model_validate(
            {
                "period": {
                    "minimum": 4,
                    "maximum": 8,
                    "training_mode": "fixed_period_inferred",
                    "fixed_period_frames": 16,
                }
            }
        )


def test_inferred_union_repair_changes_only_cross_scale_denominator() -> None:
    root = Path(__file__).parents[1]
    vector_period = load_config(
        root
        / "configs"
        / "experiments"
        / "pams_projected_vector_period_v3.yaml"
    )
    union = load_config(
        root
        / "configs"
        / "experiments"
        / "pams_projected_vector_period_union_v4.yaml"
    )

    assert vector_period.loss.exclude_other_scale_positives_from_denominator is False
    assert union.loss.exclude_other_scale_positives_from_denominator is True
    assert union.fingerprint != vector_period.fingerprint
    assert union.nonseed_fingerprint != vector_period.nonseed_fingerprint
    union_payload = union.model_dump()
    union_payload["loss"]["exclude_other_scale_positives_from_denominator"] = False
    assert union_payload == vector_period.model_dump()


def test_explicit_default_union_repair_preserves_historical_fingerprint() -> None:
    implicit = PAMSConfig()
    payload = implicit.model_dump()
    payload["loss"].pop("exclude_other_scale_positives_from_denominator")
    explicit = PAMSConfig.model_validate(implicit.model_dump())
    reconstructed_implicit = PAMSConfig.model_validate(payload)

    assert explicit.fingerprint == reconstructed_implicit.fingerprint
    assert explicit.nonseed_fingerprint == reconstructed_implicit.nonseed_fingerprint


def test_medium_only_is_inference_identity_and_default_preserves_history() -> None:
    implicit = PAMSConfig()
    payload = implicit.model_dump()
    payload["consensus"].pop("expert_mode")
    explicit_default = PAMSConfig.model_validate(implicit.model_dump())
    reconstructed_implicit = PAMSConfig.model_validate(payload)
    medium_only = PAMSConfig.model_validate(
        {
            **implicit.model_dump(),
            "consensus": {
                **implicit.consensus.model_dump(),
                "expert_mode": "medium_only",
            },
        }
    )

    assert explicit_default.fingerprint == reconstructed_implicit.fingerprint
    assert explicit_default.nonseed_fingerprint == (
        reconstructed_implicit.nonseed_fingerprint
    )
    assert medium_only.fingerprint != implicit.fingerprint
    assert medium_only.nonseed_fingerprint != implicit.nonseed_fingerprint
    assert medium_only.pose_fingerprint == implicit.pose_fingerprint


def test_config_rejects_unknown_input_projection_scale() -> None:
    with pytest.raises(ValidationError, match="input_projection_scale"):
        PAMSConfig.model_validate(
            {"model": {"input_projection_scale": "sqrt_input_dim"}}
        )


def test_config_rejects_unknown_post_warmup_period_source() -> None:
    with pytest.raises(ValidationError, match="post_warmup_source"):
        PAMSConfig.model_validate(
            {"period": {"post_warmup_source": "transformer_coordinate"}}
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


def test_longest_contiguous_track_revision_is_explicit_pose_identity() -> None:
    legacy = PAMSConfig()
    payload = legacy.model_dump()
    payload["pose"]["preprocessing_revision"] = (
        "longest-contiguous-track-minmax-zero-span-invalid-v3"
    )
    corrected = PAMSConfig.model_validate(payload)

    assert corrected.pose_fingerprint != legacy.pose_fingerprint
    assert corrected.fingerprint != legacy.fingerprint
    with pytest.raises(ValidationError, match="requires.*true"):
        PAMSConfig.model_validate(
            {
                "pose": {
                    "preprocessing_revision": (
                        "longest-contiguous-track-minmax-zero-span-invalid-v3"
                    ),
                    "crop_to_detected_span": False,
                }
            }
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
