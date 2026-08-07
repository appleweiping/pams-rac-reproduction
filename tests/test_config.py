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
    assert config.period.direct_fft_timebase == "compact_valid"
    assert config.loss.anchor_stride == 1
    assert config.loss.use_cross_cluster_negatives is True
    assert config.loss.exclude_other_scale_positives_from_denominator is False
    assert config.model.position_encoding_mode == "sinusoidal"
    assert config.sshead.architecture == "pointwise_mlp"
    assert config.sshead.input_source == "encoder_embedding"
    assert config.sshead.period_confidence_mode == "nonzero_gate"
    assert config.sshead.shape_normalization == "raw"
    assert config.consensus.expert_mode == "multi"
    assert config.readout.action_curve_source == "learned_period_head"


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


def test_v14_paper_alignment_bundle_has_frozen_method_identity() -> None:
    root = Path(__file__).parents[1]
    config = load_config(
        root
        / "configs"
        / "experiments"
        / "pams_paper_aligned_corrections_v14.yaml"
    )

    assert config.seed == 2026
    assert (
        config.pose.preprocessing_revision
        == "longest-contiguous-track-minmax-zero-span-invalid-v3"
    )
    assert config.period.pose_energy_epochs == 10
    assert config.period.post_warmup_source == "embedding_velocity_vector_acf"
    assert config.training.skeleton_augmentation.enabled is True
    assert config.training.skeleton_augmentation.rotation_degrees == (
        15.0,
        15.0,
        15.0,
    )
    assert config.training.skeleton_augmentation.scale_range == (0.85, 1.15)
    assert config.training.skeleton_augmentation.jitter_std == 0.01
    assert config.model.position_encoding_mode == "sinusoidal"
    assert config.sshead.architecture == "pointwise_mlp"
    assert config.sshead.input_source == "encoder_embedding"
    assert config.sshead.period_confidence_mode == "nonzero_gate"
    assert config.loss.scales == (0.5, 1.0, 1.5)
    assert config.consensus.expert_mode == "multi"


def test_v15_projected_teacher_changes_only_post_warmup_source() -> None:
    root = Path(__file__).parents[1]
    v14 = load_config(
        root
        / "configs"
        / "experiments"
        / "pams_paper_aligned_corrections_v14.yaml"
    )
    v15 = load_config(
        root / "configs" / "experiments" / "pams_projected_teacher_v15.yaml"
    )

    assert (
        v15.period.post_warmup_source
        == "projected_pose_velocity_vector_acf"
    )
    assert v15.fingerprint != v14.fingerprint
    assert v15.nonseed_fingerprint != v14.nonseed_fingerprint
    assert v15.pose_fingerprint == v14.pose_fingerprint
    restored = v15.model_dump()
    restored["period"]["post_warmup_source"] = (
        "embedding_velocity_vector_acf"
    )
    assert restored == v14.model_dump()


def test_masked_rms_sshead_candidate_changes_only_one_config_field() -> None:
    root = Path(__file__).parents[1]
    upstream = load_config(
        root / "configs" / "experiments" / "pams_noabs_projected_teacher_v16.yaml"
    )
    candidate = load_config(
        root
        / "configs"
        / "experiments"
        / "pams_noabs_projected_teacher_v16_sshead_masked_rms_v1.yaml"
    )

    assert upstream.sshead.shape_normalization == "raw"
    assert (
        upstream.fingerprint
        == "d00cfee1875ae597bbeec8a3fd8ae9f7bd01a7ad490d1d7048a3f7af22185c8e"
    )
    assert candidate.sshead.shape_normalization == "masked_rms"
    assert candidate.fingerprint != upstream.fingerprint
    assert candidate.nonseed_fingerprint != upstream.nonseed_fingerprint
    assert candidate.pose_fingerprint == upstream.pose_fingerprint
    restored = candidate.model_dump()
    restored["sshead"]["shape_normalization"] = "raw"
    assert restored == upstream.model_dump()


def test_v16_embedding_curve_candidate_changes_only_inference_readout() -> None:
    root = Path(__file__).parents[1]
    upstream = load_config(
        root / "configs" / "experiments" / "pams_noabs_projected_teacher_v16.yaml"
    )
    candidate = load_config(
        root
        / "configs"
        / "experiments"
        / "pams_noabs_projected_teacher_v16_embedding_curve_v1.yaml"
    )

    assert upstream.readout.action_curve_source == "learned_period_head"
    assert candidate.readout.action_curve_source == (
        "embedding_frequency_projection"
    )
    assert candidate.fingerprint != upstream.fingerprint
    assert candidate.nonseed_fingerprint != upstream.nonseed_fingerprint
    assert candidate.pose_fingerprint == upstream.pose_fingerprint
    restored = candidate.model_dump()
    restored["readout"]["action_curve_source"] = "learned_period_head"
    assert restored == upstream.model_dump()


def test_explicit_default_action_curve_preserves_historical_fingerprint() -> None:
    explicit = PAMSConfig()
    implicit_payload = explicit.model_dump()
    implicit_payload.pop("readout")
    implicit = PAMSConfig.model_validate(implicit_payload)

    assert explicit.fingerprint == implicit.fingerprint
    assert explicit.nonseed_fingerprint == implicit.nonseed_fingerprint


def test_explicit_raw_sshead_shape_preserves_historical_fingerprint() -> None:
    implicit = PAMSConfig()
    payload = implicit.model_dump()
    payload["sshead"].pop("shape_normalization")
    reconstructed = PAMSConfig.model_validate(payload)
    explicit = PAMSConfig.model_validate(implicit.model_dump())

    assert explicit.fingerprint == reconstructed.fingerprint
    assert explicit.nonseed_fingerprint == reconstructed.nonseed_fingerprint


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


def test_reference_relative_head_is_opt_in_and_preserves_legacy_identity() -> None:
    legacy = PAMSConfig()
    implicit_payload = legacy.model_dump()
    implicit_payload["sshead"].pop("input_source")
    reconstructed_legacy = PAMSConfig.model_validate(implicit_payload)
    inferred_payload = legacy.model_dump()
    inferred_payload["sshead"][
        "input_source"
    ] = "projected_pose_reference_relative"
    inferred = PAMSConfig.model_validate(inferred_payload)

    assert legacy.fingerprint == reconstructed_legacy.fingerprint
    assert legacy.nonseed_fingerprint == reconstructed_legacy.nonseed_fingerprint
    assert inferred.sshead.input_source == "projected_pose_reference_relative"
    assert inferred.fingerprint != legacy.fingerprint
    assert inferred.nonseed_fingerprint != legacy.nonseed_fingerprint
    assert inferred.pose_fingerprint == legacy.pose_fingerprint


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


def test_pe_permutation_v9_changes_only_inferred_training_weight() -> None:
    root = Path(__file__).parents[1]
    v8 = load_config(
        root / "configs" / "experiments" / "pams_longest_contiguous_track_v8.yaml"
    )
    v9 = load_config(
        root
        / "configs"
        / "experiments"
        / "pams_pe_permutation_consistency_v9.yaml"
    )

    assert v8.training.position_permutation_consistency_weight == 0.0
    assert v9.training.position_permutation_consistency_weight == 1.0
    restored = v9.model_dump()
    restored["training"]["position_permutation_consistency_weight"] = 0.0
    assert restored == v8.model_dump()
    assert v9.fingerprint != v8.fingerprint
    assert v9.nonseed_fingerprint != v8.nonseed_fingerprint
    assert v9.pose_fingerprint == v8.pose_fingerprint


def test_no_absolute_pe_v11_changes_only_position_encoding_mode() -> None:
    root = Path(__file__).parents[1]
    v8 = load_config(
        root / "configs" / "experiments" / "pams_longest_contiguous_track_v8.yaml"
    )
    v11 = load_config(
        root / "configs" / "experiments" / "pams_no_absolute_pe_v11.yaml"
    )

    assert v8.model.position_encoding_mode == "sinusoidal"
    assert v11.model.position_encoding_mode == "none"
    assert v11.training.position_permutation_consistency_weight == 0.0
    restored = v11.model_dump()
    restored["model"]["position_encoding_mode"] = "sinusoidal"
    assert restored == v8.model_dump()
    assert v11.fingerprint != v8.fingerprint
    assert v11.nonseed_fingerprint != v8.nonseed_fingerprint
    assert v11.pose_fingerprint == v8.pose_fingerprint


def test_no_absolute_pe_seed42_is_exact_nonseed_replica() -> None:
    root = Path(__file__).parents[1]
    seed2026 = load_config(
        root / "configs" / "experiments" / "pams_no_absolute_pe_v11.yaml"
    )
    seed42 = load_config(
        root
        / "configs"
        / "experiments"
        / "pams_no_absolute_pe_seed42_v11.yaml"
    )

    assert seed2026.seed == 2026
    assert seed42.seed == 42
    restored = seed42.model_dump()
    restored["seed"] = seed2026.seed
    assert restored == seed2026.model_dump()
    assert seed42.fingerprint != seed2026.fingerprint
    assert seed42.nonseed_fingerprint == seed2026.nonseed_fingerprint
    assert seed42.pose_fingerprint == seed2026.pose_fingerprint


def test_default_position_encoding_preserves_exact_v8_historical_identity() -> None:
    root = Path(__file__).parents[1]
    explicit = load_config(
        root / "configs" / "experiments" / "pams_longest_contiguous_track_v8.yaml"
    )
    implicit_payload = explicit.model_dump()
    implicit_payload["model"].pop("position_encoding_mode")
    implicit = PAMSConfig.model_validate(implicit_payload)

    assert explicit.fingerprint == (
        "eaf9e2ce6047c4a13c13daaef10af1ae288542ad94ff9fb513d19e225912adf2"
    )
    assert explicit.fingerprint == implicit.fingerprint
    assert explicit.nonseed_fingerprint == implicit.nonseed_fingerprint


def test_position_encoding_mode_validation_rejects_invalid_or_nonsensical_modes() -> None:
    with pytest.raises(ValidationError, match="position_encoding_mode"):
        PAMSConfig.model_validate(
            {"model": {"position_encoding_mode": "learned"}}
        )
    with pytest.raises(
        ValidationError,
        match="position-permutation consistency requires",
    ):
        PAMSConfig.model_validate(
            {
                "model": {"position_encoding_mode": "none"},
                "training": {
                    "position_permutation_consistency_weight": 1.0,
                },
            }
        )


def test_disabled_pe_permutation_weight_preserves_historical_identity() -> None:
    explicit = PAMSConfig()
    implicit_payload = explicit.model_dump()
    implicit_payload["training"].pop(
        "position_permutation_consistency_weight"
    )
    implicit = PAMSConfig.model_validate(implicit_payload)

    assert explicit.fingerprint == implicit.fingerprint
    assert explicit.nonseed_fingerprint == implicit.nonseed_fingerprint
    with pytest.raises(
        ValidationError,
        match="position_permutation_consistency_weight",
    ):
        PAMSConfig.model_validate(
            {
                "training": {
                    "position_permutation_consistency_weight": -0.1,
                }
            }
        )


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


def test_tcc_anchor_stride_is_identity_changing_and_default_preserves_history() -> None:
    implicit = PAMSConfig()
    payload = implicit.model_dump()
    payload["loss"].pop("anchor_stride")
    reconstructed_implicit = PAMSConfig.model_validate(payload)
    explicit_default = PAMSConfig.model_validate(implicit.model_dump())
    sparse = PAMSConfig.model_validate(
        {
            **implicit.model_dump(),
            "loss": {
                **implicit.loss.model_dump(),
                "anchor_stride": 4,
            },
        }
    )

    assert explicit_default.fingerprint == reconstructed_implicit.fingerprint
    assert explicit_default.nonseed_fingerprint == (
        reconstructed_implicit.nonseed_fingerprint
    )
    assert sparse.fingerprint != implicit.fingerprint
    assert sparse.nonseed_fingerprint != implicit.nonseed_fingerprint
    assert sparse.pose_fingerprint == implicit.pose_fingerprint


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_tcc_anchor_stride_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValidationError, match="anchor_stride"):
        PAMSConfig.model_validate({"loss": {"anchor_stride": value}})


def test_cross_cluster_negative_switch_is_identity_changing_and_default_preserves_history() -> None:
    implicit = PAMSConfig()
    payload = implicit.model_dump()
    payload["loss"].pop("use_cross_cluster_negatives")
    reconstructed_implicit = PAMSConfig.model_validate(payload)
    explicit_default = PAMSConfig.model_validate(implicit.model_dump())
    disabled = PAMSConfig.model_validate(
        {
            **implicit.model_dump(),
            "loss": {
                **implicit.loss.model_dump(),
                "use_cross_cluster_negatives": False,
            },
        }
    )

    assert explicit_default.fingerprint == reconstructed_implicit.fingerprint
    assert explicit_default.nonseed_fingerprint == (
        reconstructed_implicit.nonseed_fingerprint
    )
    assert disabled.fingerprint != implicit.fingerprint
    assert disabled.nonseed_fingerprint != implicit.nonseed_fingerprint
    assert disabled.pose_fingerprint == implicit.pose_fingerprint


@pytest.mark.parametrize("value", [0, 1, "true"])
def test_cross_cluster_negative_switch_requires_strict_boolean(value: object) -> None:
    with pytest.raises(ValidationError, match="use_cross_cluster_negatives"):
        PAMSConfig.model_validate(
            {"loss": {"use_cross_cluster_negatives": value}}
        )


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


def test_dense_resampled_direct_fft_is_opt_in_inference_identity() -> None:
    explicit_default = PAMSConfig()
    implicit_payload = explicit_default.model_dump()
    implicit_payload["period"].pop("direct_fft_timebase")
    reconstructed_implicit = PAMSConfig.model_validate(implicit_payload)
    dense_payload = explicit_default.model_dump()
    dense_payload["period"]["direct_fft_timebase"] = "dense_resampled"
    dense = PAMSConfig.model_validate(dense_payload)

    assert explicit_default.fingerprint == reconstructed_implicit.fingerprint
    assert explicit_default.nonseed_fingerprint == (
        reconstructed_implicit.nonseed_fingerprint
    )
    assert dense.period.direct_fft_timebase == "dense_resampled"
    assert dense.fingerprint != explicit_default.fingerprint
    assert dense.nonseed_fingerprint != explicit_default.nonseed_fingerprint
    assert dense.pose_fingerprint == explicit_default.pose_fingerprint


def test_dense_timebase_field_preserves_frozen_reference_checkpoint_identity() -> None:
    root = Path(__file__).parents[1]
    config = load_config(
        root
        / "configs"
        / "experiments"
        / "pams_official_segment_reference_relative_v1_pad_invalid_tail.yaml"
    )

    assert config.period.direct_fft_timebase == "compact_valid"
    assert config.fingerprint == (
        "32d086e1c2f76ee1c2beb76c984e0a6e6c60a05e1330c510fc9d0dde6b32054e"
    )
    assert config.pose_fingerprint == (
        "f89cfc3e520cd3f45282ec06f42d5702f5e7c30c28ee6a3655ebfb5e58766047"
    )


def test_config_rejects_unknown_direct_fft_timebase() -> None:
    with pytest.raises(ValidationError, match="direct_fft_timebase"):
        PAMSConfig.model_validate(
            {"period": {"direct_fft_timebase": "original_video_fps"}}
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


def test_official_segment_timeline_and_padding_are_explicit_pose_identities() -> None:
    repository = Path(__file__).parents[1]
    legacy = load_config(
        repository / "configs" / "experiments" / "pams_longest_contiguous_track_v8.yaml"
    )
    strict = load_config(
        repository / "configs" / "experiments" / "pams_official_segment_v1.yaml"
    )
    padded = load_config(
        repository
        / "configs"
        / "experiments"
        / "pams_official_segment_v1_pad_invalid_tail.yaml"
    )

    assert strict.pose.preprocessing_revision == "official-segment-full-timeline-v1"
    assert strict.pose.crop_to_detected_span is False
    assert strict.pose.incomplete_clip_policy == "error"
    assert padded.pose.incomplete_clip_policy == "pad_invalid_tail"
    assert len({legacy.pose_fingerprint, strict.pose_fingerprint, padded.pose_fingerprint}) == 3
    assert len({legacy.fingerprint, strict.fingerprint, padded.fingerprint}) == 3
    with pytest.raises(ValidationError, match="requires.*false"):
        PAMSConfig.model_validate(
            {
                "pose": {
                    "preprocessing_revision": "official-segment-full-timeline-v1",
                    "crop_to_detected_span": True,
                }
            }
        )
    with pytest.raises(ValidationError, match="only valid"):
        PAMSConfig.model_validate(
            {"pose": {"incomplete_clip_policy": "pad_invalid_tail"}}
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
