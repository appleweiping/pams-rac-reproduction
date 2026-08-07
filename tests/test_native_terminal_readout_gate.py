from __future__ import annotations

import hashlib
import inspect
import json
import math
import tempfile
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from pams.config import PAMSConfig, load_config
from pams.run_manifest import ArtifactReceipt, CompletedRunReceipt, RunManifest
from scripts.server import run_pams_native_terminal_readout_gate as runner

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "server" / "run_pams_native_terminal_readout_gate.py"
SPECIFICATION = ROOT / "configs" / "gates" / "pams_native_terminal_readout_gate_v1.yaml"
BASE_CONFIG = ROOT / "configs" / "experiments" / "pams_native_table2_baseline_proxy_v1.yaml"


def _config(*, candidate_id: str = "A") -> PAMSConfig:
    payload = load_config(BASE_CONFIG).model_dump(mode="json")
    if candidate_id == "B":
        payload["loss"]["anchor_stride"] = 2
    elif candidate_id == "C":
        payload["period"]["fixed_period_frames"] = 24
    elif candidate_id != "A":
        raise ValueError(candidate_id)
    return PAMSConfig.model_validate(payload)


def _periodic_embeddings(
    period: float,
    *,
    frames: int | None = None,
    phase_offset: float = 0.0,
) -> torch.Tensor:
    frames = max(192, int(math.ceil(period * 8))) if frames is None else frames
    time = torch.arange(frames, dtype=torch.float32)
    phase = 2.0 * math.pi * time / period + phase_offset
    return torch.stack(
        (
            torch.cos(phase),
            torch.sin(phase),
            0.25 * torch.cos(2.0 * phase),
            0.25 * torch.sin(2.0 * phase),
        ),
        dim=1,
    ).unsqueeze(0)


def _random_embeddings(*, batch: int, frames: int, dimension: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(810_733)
    return torch.randn((batch, frames, dimension), generator=generator)


def _summary(median: float) -> dict[str, float | int]:
    return {
        "observations": 4,
        "minimum": median,
        "p10": median,
        "median": median,
        "mean": median,
        "p90": median,
        "maximum": median,
        "standard_deviation": 0.0,
    }


def _passing_aggregates() -> dict[str, object]:
    return {
        "representation": {
            "near_collapsed_share": 0.0,
            "lag_eligible_share": 0.8,
            "all_real_eligible_null_margins_complete": True,
            "real_cycle_margin": _summary(0.2),
            "real_minus_strongest_null": _summary(0.1),
            "real_beats_all_nulls_share": 0.8,
        },
        "period": {"boundary_share": 0.1, "mode_share": 0.1},
        "readout": {
            "carrier_eligible_available_share": 0.8,
            "expert_majority_share": 0.8,
            "selected_vs_active_reference_absolute_gap": _summary(1.0),
        },
        "peak_total": {"zero_share": 0.1, "mode_share": 0.1},
        "time_scale": {
            "period": {
                "eligible_comparison_share": 0.8,
                "relative_error": _summary(0.1),
            },
            "peak_total": {
                "eligible_comparison_share": 0.8,
                "exact_share": 0.8,
                "within_one_share": 0.95,
            },
        },
        "embedding_shuffle_fixed_real_period": {
            "shuffled_to_real_harmonic_median_ratio": 0.5,
            "shuffled_to_real_gate_energy_median_ratio": 0.5,
        },
        "zero_pose": {
            "positive_period_confidence_share": 0.0,
            "carrier_available_share": 0.0,
            "positive_support_share": 0.0,
        },
        "static_pose": {
            "positive_period_confidence_share": 0.0,
            "carrier_available_share": 0.0,
            "positive_support_share": 0.0,
        },
    }


def _readout_sample(
    *,
    video_id: str,
    period: float,
    peak_total: int,
    carrier_eligible: bool = True,
    period_eligible: bool = True,
    upper: int = 128,
) -> runner.ReadoutSample:
    return runner.ReadoutSample(
        video_id=video_id,
        timeline_length=256,
        valid_total=256,
        period_upper_bound=upper,
        period_frames=period,
        period_confidence=0.8 if period_eligible else 0.0,
        period_lag_eligible=period_eligible,
        carrier_eligible=carrier_eligible,
        raw_carrier_available=carrier_eligible,
        harmonic_energy_fraction=0.4,
        recurrence_gate_energy=0.2,
        active_support_fraction=0.8,
        normalized_curve_rms=1.0,
        raw_curve_rms=0.5,
        selected_peak_total=peak_total,
        active_reference_peak_total=peak_total,
        full_timeline_reference_peak_total=peak_total,
        expert_peak_totals=(peak_total, peak_total, max(0, peak_total - 1)),
        selected_expert="medium",
        selection_rule="majority_first" if carrier_eligible else "ineligible_no_peak_readout",
    )


def _minimal_terminal_payload(
    *,
    candidate_id: str,
    gate_specification_sha256: str,
    pose_cache_set_sha256: str,
    source_git_sha: str,
    prior: list[dict[str, str]],
    passed: bool,
) -> dict[str, object]:
    specification = runner.load_gate_specification(SPECIFICATION)
    profile = specification.candidate_profiles[candidate_id]
    digest = "d" * 64
    summary = _summary(0.2)
    aggregates: dict[str, object] = {
        "representation": {
            "record_total": 337,
            "temporal_rms": summary,
            "near_collapsed_share": 0.0 if passed else 1.0,
            "lag_eligible_total": 337,
            "lag_eligible_share": 1.0,
            "null_complete_total": 337,
            "null_complete_share_of_real_eligible": 1.0,
            "all_real_eligible_null_margins_complete": True,
            "real_cycle_margin": summary,
            "pose_shuffle_cycle_margin": _summary(0.0),
            "embedding_shuffle_cycle_margin": _summary(0.0),
            "zero_pose_cycle_margin": _summary(0.0),
            "static_pose_cycle_margin": _summary(0.0),
            "real_minus_strongest_null": _summary(0.2),
            "real_beats_all_nulls_share": 1.0,
        },
        "period": {
            "record_total": 337,
            "per_sample_upper_bound_used": True,
            "boundary_share": 0.0,
            "minimum_boundary_share": 0.0,
            "upper_boundary_share": 0.0,
            "mode_period_frames": 16.0,
            "mode_frequency": 10,
            "mode_share": 0.1,
            "unique_period_total": 20,
            "positive_confidence_share": 1.0,
            "period_frames": _summary(16.0),
            "period_confidence_non_authorizing": _summary(0.8),
            "period_histogram_aggregate_only": {"16": 10},
        },
        "readout": {
            "record_total": 337,
            "carrier_eligible_available_total": 337,
            "carrier_eligible_available_share": 1.0,
            "raw_carrier_available_share_non_authorizing": 1.0,
            "expert_majority_share": 1.0,
            "expert_fft_nearest_fallback_share": 0.0,
            "selected_vs_active_reference_absolute_gap": _summary(0.0),
            "normalized_curve_rms_non_authorizing": _summary(1.0),
            "raw_curve_rms_non_authorizing": _summary(0.5),
            "harmonic_energy_fraction_non_authorizing": _summary(0.4),
            "active_support_fraction_non_authorizing": _summary(0.8),
            "recurrence_gate_energy_non_authorizing": _summary(0.2),
        },
        "peak_total": {
            "record_total": 337,
            "eligible_readout_total": 337,
            "eligible_readout_share": 1.0,
            "zero_share": 0.0,
            "mode_peak_total": 8,
            "mode_frequency": 10,
            "mode_share": 0.1,
            "unique_peak_total": 20,
            "absolute_distribution_non_authorizing": _summary(8.0),
            "histogram_aggregate_only": {"8": 10},
        },
        "time_scale": {
            "factors": [0.75, 1.25],
            "period": {
                "candidate_comparison_total": 674,
                "eligible_comparison_total": 674,
                "eligible_comparison_share": 1.0,
                "relative_error": _summary(0.0),
            },
            "peak_total": {
                "candidate_comparison_total": 674,
                "eligible_comparison_total": 674,
                "eligible_comparison_share": 1.0,
                "absolute_difference": _summary(0.0),
                "exact_share": 1.0,
                "within_one_share": 1.0,
            },
            "per_video_rows_persisted": False,
        },
        "embedding_shuffle_fixed_real_period": {
            "real_eligibility_sets_denominator": True,
            "real_eligible_total": 337,
            "real_harmonic_energy_fraction": _summary(1.0),
            "shuffled_harmonic_energy_fraction": _summary(0.0),
            "shuffled_to_real_harmonic_median_ratio": 0.0,
            "real_recurrence_gate_energy": _summary(1.0),
            "shuffled_recurrence_gate_energy": _summary(0.0),
            "shuffled_to_real_gate_energy_median_ratio": 0.0,
        },
        "zero_pose": {
            "record_total": 337,
            "positive_period_confidence_share": 0.0,
            "carrier_available_share": 0.0,
            "confidence_and_structure_eligible_share_non_authorizing": 0.0,
            "positive_support_share": 0.0,
        },
        "static_pose": {
            "record_total": 337,
            "positive_period_confidence_share": 0.0,
            "carrier_available_share": 0.0,
            "confidence_and_structure_eligible_share_non_authorizing": 0.0,
            "positive_support_share": 0.0,
        },
        "per_video_rows_persisted": False,
    }
    decision = runner.gate_decision(
        candidate_id=candidate_id,
        aggregates=aggregates,
        thresholds=specification.thresholds,
    )
    if decision["overall_pass"] is not passed:
        raise RuntimeError("terminal fixture did not produce the requested decision")
    code_hashes = {"gate_runner": digest}
    inputs = {
        "encoder_checkpoint_sha256": digest,
        "encoder_progress_sha256": digest,
        "encoder_completion_receipt_sha256": digest,
        "encoder_started_receipt_sha256": digest,
        "source_export_receipt_sha256": digest,
        "experiment_config_sha256": digest,
        "gate_specification_sha256": gate_specification_sha256,
        "pose_snapshot_sha256": digest,
        "pose_cache_set_sha256": pose_cache_set_sha256,
        "epoch11_gate_artifact_sha256": digest,
        "epoch11_gate_artifact_bytes": 101,
        "epoch11_gate_receipt_sha256": digest,
        "epoch11_gate_receipt_bytes": 102,
        "candidate_launch_authorization_sha256": digest,
        "candidate_launch_authorization_bytes": 103,
        "candidate_launch_authorization_receipt_sha256": digest,
        "candidate_launch_authorization_receipt_bytes": 104,
        "candidate_registry_id": digest,
        "candidate_registry_reservation_sha256": digest,
        "candidate_registry_reservation_bytes": 105,
        "train_run_receipt_sha256": digest,
        "train_run_receipt_bytes": 106,
        "container_audits_sha256_commitment": digest,
        "source_git_sha": source_git_sha,
        "training_video_total": 337,
        "read_only_post_run_identity_verified": True,
        "code_files_sha256": code_hashes,
        "code_files_sha256_commitment": runner.sha256_json(code_hashes),
    }
    hardware = {"fixture": "hardware"}
    runtime = {"source_git_sha": source_git_sha}
    return {
        "schema_version": 1,
        "artifact_type": runner._EXPECTED_ARTIFACT_TYPE,
        "status": "terminal_readout_eligible" if passed else "scientific_rejection",
        "classification": specification.classification,
        "protocol": specification.expected_protocol,
        "seed": specification.expected_seed,
        "candidate": {
            "id": candidate_id,
            "fixed_training_period_frames_provenance_only": (
                profile.fixed_training_period_frames
            ),
            "anchor_stride": profile.anchor_stride,
            "first_pass_only": True,
            "prior_scientific_rejections": prior,
            "cross_candidate_metric_ranking_forbidden": True,
        },
        "label_firewall": {
            "accepted_scientific_inputs": ["fixture"],
            "manifest_interface_supported": False,
            "media_interface_supported": False,
            "development_identity_media_pose_or_target_interface_supported": False,
            "sealed_evaluation_identity_media_pose_or_target_interface_supported": False,
            "action_class_interface_supported": False,
            "repetition_annotation_interface_supported": False,
            "external_label_fields_accessed": [],
            "training_interface_supported": False,
        },
        "inputs": inputs,
        "algorithm": {
            "period_estimator": "full_vector_embedding_velocity_acf",
            "per_sample_period_upper_bound": runner._period_upper_bound_metadata(
                _config(candidate_id=candidate_id)
            ),
            "period_confidence_part_of_carrier_eligibility": True,
            "carrier_eligibility": "fixture",
            "pose_time_shuffle": "fixture",
            "embedding_time_shuffle": "fixture",
            "zero_pose": "fixture",
            "static_pose": "fixture",
            "cycle_margin": "fixture",
            "time_scale_factors": [0.75, 1.25],
            "peak_readout": "fixture",
            "full_timeline_reference_authorizing": False,
            "fixed_training_period_used_at_inference": False,
        },
        "thresholds": dict(specification.thresholds),
        "aggregates": aggregates,
        "hard_invariants": {
            **{key: True for key in runner._HARD_INVARIANT_KEYS},
            "per_video_predictions_persisted": False,
        },
        "gate": decision,
        "scientific_caveats": list(runner._SCIENTIFIC_CAVEATS),
        "hardware": hardware,
        "hardware_sha256": runner.sha256_json(hardware),
        "runtime": runtime,
        "runtime_sha256": runner.sha256_json(runtime),
        "read_only_verification": {
            "all_file_inputs_unchanged": True,
            "train337_pose_cache_set_unchanged": True,
            "model_state_sha256_before": digest,
            "model_state_sha256_after": digest,
            "model_or_optimizer_state_updated": False,
            "training_steps_executed": 0,
            "pose_cache_write_operations": 0,
            "per_video_prediction_write_operations": 0,
        },
    }


def test_gate_specification_freezes_profiles_thresholds_and_advisories() -> None:
    specification = runner.load_gate_specification(SPECIFICATION)
    assert tuple(specification.candidate_profiles) == ("A", "B", "C")
    assert {
        key: (
            value.fixed_training_period_frames,
            value.anchor_stride,
            value.required_prior_scientific_rejections,
        )
        for key, value in specification.candidate_profiles.items()
    } == {
        "A": (16, 4, ()),
        "B": (16, 2, ("A",)),
        "C": (24, 4, ("A", "B")),
    }
    assert specification.time_scale_factors == (0.75, 1.25)
    assert set(specification.thresholds) == runner._THRESHOLD_KEYS
    assert len(specification.thresholds) == 25
    assert set(specification.non_authorizing_metrics) == runner._NON_AUTHORIZING_METRICS


def test_gate_yaml_rejects_duplicate_and_privileged_fields() -> None:
    encoded = SPECIFICATION.read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="pams-gate-") as directory:
        duplicate = Path(directory) / "duplicate.yaml"
        duplicate.write_text(encoded + "\nschema_version: 1\n", encoding="utf-8")
        with pytest.raises(ValueError, match="duplicate YAML field"):
            runner.load_gate_specification(duplicate)
        privileged = Path(directory) / "privileged.yaml"
        payload = yaml.safe_load(encoded)
        payload["target"] = "forbidden"
        privileged.write_text(yaml.safe_dump(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="forbidden privileged field"):
            runner.load_gate_specification(privileged)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("norm_first", 0),
        ("skeleton_augmentation_enabled", 1),
        ("model_input_dim", 99.0),
        ("loss_scales", [1]),
    ],
)
def test_gate_yaml_rejects_candidate_invariant_scalar_type_drift(
    key: str,
    value: object,
) -> None:
    payload = yaml.safe_load(SPECIFICATION.read_text(encoding="utf-8"))
    payload["candidate_invariants"][key] = value
    with tempfile.TemporaryDirectory(prefix="pams-gate-") as directory:
        path = Path(directory) / "type-drift.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        with pytest.raises((TypeError, ValueError)):
            runner.load_gate_specification(path)


def test_progress_firewall_runs_before_checkpoint_deserialization() -> None:
    with tempfile.TemporaryDirectory(prefix="pams-gate-") as directory:
        path = Path(directory) / "encoder.jsonl"
        path.write_text('{"schema_version":2,"epoch":1}\n', encoding="utf-8")
        runner._preflight_progress_jsonl(path)
        path.write_text(
            '{"schema_version":2,"schema_version":2,"epoch":1}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="duplicate JSON field"):
            runner._preflight_progress_jsonl(path)
        path.write_text('{"target":"forbidden"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="forbidden privileged field"):
            runner._preflight_progress_jsonl(path)


@pytest.mark.parametrize("candidate_id", ["A", "B", "C"])
def test_candidate_profiles_accept_only_the_preregistered_training_variant(
    candidate_id: str,
) -> None:
    specification = runner.load_gate_specification(SPECIFICATION)
    profile = runner._validate_candidate_config(
        _config(candidate_id=candidate_id),
        specification,
        candidate_id=candidate_id,
    )
    assert profile == specification.candidate_profiles[candidate_id]


def test_candidate_profile_rejects_inference_or_architecture_drift() -> None:
    specification = runner.load_gate_specification(SPECIFICATION)
    payload = _config().model_dump(mode="json")
    payload["period"]["maximum_mode"] = "fixed"
    with pytest.raises(ValueError, match="candidate invariant mismatch"):
        runner._validate_candidate_config(
            PAMSConfig.model_validate(payload), specification, candidate_id="A"
        )
    payload = _config().model_dump(mode="json")
    payload["model"]["position_encoding_mode"] = "none"
    with pytest.raises(ValueError, match="candidate invariant mismatch"):
        runner._validate_candidate_config(
            PAMSConfig.model_validate(payload), specification, candidate_id="A"
        )


def test_interface_has_no_privileged_scientific_surface() -> None:
    assert set(inspect.signature(runner.run_gate).parameters) == {
        "encoder_checkpoint_path",
        "encoder_progress_path",
        "encoder_completion_receipt_path",
        "source_receipt_path",
        "config_path",
        "gate_specification_path",
        "candidate_launch_authorization_path",
        "candidate_launch_receipt_path",
        "epoch11_gate_artifact_path",
        "epoch11_gate_receipt_path",
        "pose_cache_dir",
        "pose_snapshot_path",
        "candidate_registry_root",
        "train_run_receipt_path",
        "candidate_id",
        "device",
        "batch_size",
    }
    source = RUNNER.read_text(encoding="utf-8")
    for option in (
        "--manifest",
        "--dev",
        "--test",
        "--target",
        "--count",
        "--video",
        "--annotation",
        "--train",
    ):
        assert f'add_argument("{option}"' not in source


def test_forged_predecessor_json_has_no_authorization_surface() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    launch_source = (
        ROOT
        / "scripts"
        / "server"
        / "prepare_pams_native_candidate_launch_authorization.py"
    ).read_text(encoding="utf-8")
    outer_source = (
        ROOT
        / "scripts"
        / "server"
        / "run_pams_native_table2_baseline_train337_v1.sh"
    ).read_text(encoding="utf-8")
    assert 'add_argument("--prior-rejection-artifact"' not in source
    assert 'add_argument("--prior-rejection-receipt"' not in source
    assert "--prior-rejection-artifact" not in launch_source
    assert "--prior-rejection-receipt" not in launch_source
    assert "PAMS_PRIOR_" not in outer_source

    with tempfile.TemporaryDirectory(prefix="pams-gate-") as directory:
        root = Path(directory)
        registry = root / "registry"
        registry.mkdir()
        (root / "forged-predecessor.json").write_text(
            '{"status":"scientific_rejection"}\n', encoding="utf-8"
        )
        with pytest.raises(FileNotFoundError):
            runner._load_predecessor_chain_from_registry(
                candidate_id="B",
                profile=runner.CandidateProfile(16, 2, ("A",)),
                registry_root=registry,
                specification=runner.load_gate_specification(SPECIFICATION),
                gate_specification_sha256="a" * 64,
                pose_cache_set_sha256="b" * 64,
                source_git_sha="c" * 40,
            )


@pytest.mark.parametrize(
    "path",
    [
        "runs/dev84/gate.json",
        "runs/test105/gate.json",
        "runs/targets/gate.json",
        "runs/counts/gate.json",
    ],
)
def test_privileged_path_tokens_are_rejected(path: str) -> None:
    with pytest.raises(ValueError, match="forbidden privileged token"):
        runner._reject_privileged_path(Path(path), role="fixture")


@pytest.mark.parametrize("period", [6, 8, 12, 16, 24, 32, 64])
def test_full_vector_readout_recovers_periods_not_locked_to_training_profile(
    period: int,
) -> None:
    embeddings = _periodic_embeddings(period)
    frames = embeddings.shape[1]
    samples = runner.readout_samples_from_embeddings(
        video_ids=(f"period-{period}",),
        embeddings=embeddings,
        valid_mask=torch.ones((1, frames), dtype=torch.bool),
        timeline_lengths=torch.tensor([frames]),
        config=_config(),
    )
    sample = samples[0]
    assert sample.period_frames == pytest.approx(period, abs=0.2)
    assert sample.period_upper_bound == frames // 2
    assert sample.period_confidence > 0.0
    assert sample.carrier_eligible is True
    assert sample.selected_peak_total in sample.expert_peak_totals


def test_fractional_cycle_margin_is_phase_and_orthogonal_basis_invariant() -> None:
    base = _periodic_embeddings(15.5, frames=192)
    shifted = _periodic_embeddings(15.5, frames=192, phase_offset=1.3)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(71)
    q, _ = torch.linalg.qr(torch.randn((4, 4), generator=generator))
    rotated = base @ q
    valid = torch.ones(192, dtype=torch.bool)
    margins = [
        runner._fractional_cycle_margin(
            values[0],
            valid,
            timeline_length=192,
            period=15.5,
            minimum_pairs=8,
        )
        for values in (base, shifted, rotated)
    ]
    assert all(value is not None and value > 1.0 for value in margins)
    assert margins[0] == pytest.approx(margins[1], abs=1e-4)
    assert margins[0] == pytest.approx(margins[2], abs=1e-4)


def test_representation_nulls_reject_position_only_adversary() -> None:
    real = _periodic_embeddings(16).repeat(4, 1, 1)
    frames = real.shape[1]
    valid = torch.ones((4, frames), dtype=torch.bool)
    lengths = torch.full((4,), frames, dtype=torch.long)
    adversarial = runner.representation_samples_from_embeddings(
        real_embeddings=real,
        pose_shuffle_embeddings=real,
        embedding_shuffle_embeddings=real,
        zero_pose_embeddings=real,
        static_pose_embeddings=real,
        valid_mask=valid,
        timeline_lengths=lengths,
        real_periods=torch.full((4,), 16.0),
        pose_shuffle_periods=torch.full((4,), 16.0),
        zero_pose_periods=torch.full((4,), 16.0),
        static_pose_periods=torch.full((4,), 16.0),
        minimum_pairs=8,
    )
    distribution = runner._representation_distribution(
        adversarial, near_collapse_rms=1e-3
    )
    assert distribution["real_cycle_margin"]["median"] > 1.0
    assert distribution["real_minus_strongest_null"]["median"] == pytest.approx(0.0)
    aggregates = _passing_aggregates()
    aggregates["representation"] = distribution
    decision = runner.gate_decision(
        candidate_id="A",
        aggregates=aggregates,
        thresholds=runner.load_gate_specification(SPECIFICATION).thresholds,
    )
    assert decision["overall_pass"] is False
    assert decision["criteria"]["real_minus_strongest_null_median"]["pass"] is False


def test_null_views_use_their_own_period_and_missing_null_fails_closed() -> None:
    frames = 192
    generator = torch.Generator(device="cpu")
    generator.manual_seed(144)
    real = _periodic_embeddings(16, frames=frames)
    real = real + 0.35 * torch.randn(real.shape, generator=generator)
    pose_null = _periodic_embeddings(24, frames=frames)
    valid = torch.ones((1, frames), dtype=torch.bool)
    complete = runner.representation_samples_from_embeddings(
        real_embeddings=real,
        pose_shuffle_embeddings=pose_null,
        embedding_shuffle_embeddings=torch.zeros_like(real),
        zero_pose_embeddings=torch.zeros_like(real),
        static_pose_embeddings=torch.zeros_like(real),
        valid_mask=valid,
        timeline_lengths=torch.tensor([frames]),
        real_periods=torch.tensor([16.0]),
        pose_shuffle_periods=torch.tensor([24.0]),
        zero_pose_periods=torch.tensor([16.0]),
        static_pose_periods=torch.tensor([16.0]),
        minimum_pairs=8,
    )[0]
    assert complete.pose_shuffle_cycle_margin is not None
    assert complete.real_cycle_margin is not None
    assert complete.pose_shuffle_cycle_margin > complete.real_cycle_margin
    assert complete.real_beats_all_nulls is False

    missing = runner.representation_samples_from_embeddings(
        real_embeddings=_periodic_embeddings(8, frames=64),
        pose_shuffle_embeddings=_periodic_embeddings(8, frames=64),
        embedding_shuffle_embeddings=_periodic_embeddings(8, frames=64),
        zero_pose_embeddings=_periodic_embeddings(8, frames=64),
        static_pose_embeddings=_periodic_embeddings(8, frames=64),
        valid_mask=torch.ones((1, 64), dtype=torch.bool),
        timeline_lengths=torch.tensor([64]),
        real_periods=torch.tensor([8.0]),
        pose_shuffle_periods=torch.tensor([48.0]),
        zero_pose_periods=torch.tensor([8.0]),
        static_pose_periods=torch.tensor([8.0]),
        minimum_pairs=8,
    )[0]
    assert missing.lag_eligible is True
    assert missing.null_margins_complete is False
    assert missing.real_beats_all_nulls is False
    distribution = runner._representation_distribution(
        (missing,), near_collapse_rms=1e-3
    )
    assert distribution["all_real_eligible_null_margins_complete"] is False
    aggregates = _passing_aggregates()
    aggregates["representation"] = distribution
    decision = runner.gate_decision(
        candidate_id="A",
        aggregates=aggregates,
        thresholds=runner.load_gate_specification(SPECIFICATION).thresholds,
    )
    assert decision["criteria"]["real_minus_strongest_null_median"]["pass"] is False


def test_pose_control_raw_availability_is_not_confidence_gated() -> None:
    distribution = runner._pose_control_distribution(
        (
            runner.PoseControlSample(
                positive_period_confidence=False,
                carrier_eligible=False,
                raw_carrier_available=True,
                positive_support=False,
            ),
        )
    )
    assert distribution["carrier_available_share"] == 1.0
    assert distribution["confidence_and_structure_eligible_share_non_authorizing"] == 0.0


def test_peak_distribution_excludes_ineligible_placeholder_zeros() -> None:
    samples = (
        _readout_sample(video_id="eligible", period=16.0, peak_total=5),
        *(
            _readout_sample(
                video_id=f"ineligible-{index}",
                period=16.0,
                peak_total=0,
                carrier_eligible=False,
            )
            for index in range(3)
        ),
    )
    distribution = runner._peak_total_distribution(samples)
    assert distribution["record_total"] == 4
    assert distribution["eligible_readout_total"] == 1
    assert distribution["eligible_readout_share"] == 0.25
    assert distribution["zero_share"] == 0.0
    assert distribution["mode_peak_total"] == 5


def test_every_carrier_batch_rejects_nonfinite_scores() -> None:
    embeddings = _periodic_embeddings(16, frames=96)
    valid = torch.ones((1, 96), dtype=torch.bool)
    lengths = torch.tensor([96])
    readout = runner.estimate_recurrence_carrier_curves(
        embeddings,
        minimum_period=4,
        maximum_period=4096,
        valid_mask=valid,
        timeline_lengths=lengths,
        maximum_mode="half_timeline",
    )
    corrupted_scores = readout.recurrence_scores.clone()
    corrupted_scores[0, 0] = float("nan")
    corrupted = replace(readout, recurrence_scores=corrupted_scores)
    with pytest.raises(RuntimeError, match="non-finite recurrence score"):
        runner._validate_recurrence_carrier_batch(
            corrupted,
            valid_mask=valid,
            timeline_lengths=lengths,
            role="fixture",
        )


def test_period_bound_metadata_matches_short_timeline_helper() -> None:
    config = _config()
    metadata = runner._period_upper_bound_metadata(config)
    assert metadata == {
        "mode": "half_timeline",
        "formula": "min(4096, max(4, floor(timeline_length/2)))",
        "short_valid_fallback": (
            "if timeline_length < 8 or valid_embedding_velocity_total < 8: "
            "period=4, confidence=0"
        ),
    }
    assert runner._period_upper_bound(
        6,
        minimum=config.period.minimum,
        maximum=config.period.maximum,
        maximum_mode=config.period.maximum_mode,
    ) == 4
    periods, confidence = runner.estimate_period_from_embedding_velocity_vectors(
        torch.randn((1, 8, 4)),
        minimum=4,
        maximum=4096,
        valid_mask=torch.ones((1, 8), dtype=torch.bool),
        timeline_lengths=torch.tensor([8]),
        maximum_mode="half_timeline",
    )
    assert periods.tolist() == [4.0]
    assert confidence.tolist() == [0.0]


def test_invalid_padding_payload_cannot_change_readout() -> None:
    base = _periodic_embeddings(16, frames=96)
    base_sample = runner.readout_samples_from_embeddings(
        video_ids=("masked",),
        embeddings=base,
        valid_mask=torch.ones((1, 96), dtype=torch.bool),
        timeline_lengths=torch.tensor([96]),
        config=_config(),
    )[0]
    polluted = torch.cat((base, _random_embeddings(batch=1, frames=32, dimension=4)), dim=1)
    valid = torch.cat(
        (torch.ones((1, 96), dtype=torch.bool), torch.zeros((1, 32), dtype=torch.bool)),
        dim=1,
    )
    polluted_sample = runner.readout_samples_from_embeddings(
        video_ids=("masked",),
        embeddings=polluted,
        valid_mask=valid,
        timeline_lengths=torch.tensor([96]),
        config=_config(),
    )[0]
    assert polluted_sample == base_sample


def test_localized_periodic_span_is_gated_without_using_full_timeline_reference() -> None:
    frames = 256
    embeddings = torch.zeros((1, frames, 4), dtype=torch.float32)
    periodic = _periodic_embeddings(16, frames=128)[0]
    embeddings[0, 64:192] = periodic
    sample = runner.readout_samples_from_embeddings(
        video_ids=("localized",),
        embeddings=embeddings,
        valid_mask=torch.ones((1, frames), dtype=torch.bool),
        timeline_lengths=torch.tensor([frames]),
        config=_config(),
    )[0]
    assert sample.period_frames == pytest.approx(16.0, abs=0.2)
    assert sample.carrier_eligible is True
    assert 0.0 < sample.active_support_fraction < 0.8
    assert sample.selected_peak_total == sample.active_reference_peak_total
    assert sample.full_timeline_reference_peak_total > sample.selected_peak_total


def test_invalid_interior_gap_payload_cannot_create_motion_or_change_readout() -> None:
    embeddings = _periodic_embeddings(16, frames=192)
    valid = torch.ones((1, 192), dtype=torch.bool)
    valid[:, 72:88] = False
    canonical = embeddings.clone()
    canonical[:, 72:88] = 0.0
    polluted = canonical.clone()
    polluted[:, 72:88] = _random_embeddings(batch=1, frames=16, dimension=4)
    arguments = {
        "video_ids": ("gap",),
        "valid_mask": valid,
        "timeline_lengths": torch.tensor([192]),
        "config": _config(),
    }
    first = runner.readout_samples_from_embeddings(
        embeddings=canonical, **arguments
    )[0]
    second = runner.readout_samples_from_embeddings(
        embeddings=polluted, **arguments
    )[0]
    assert second == first


def test_insufficient_mask_is_explicitly_ineligible() -> None:
    embeddings = _periodic_embeddings(16, frames=96)
    valid = torch.zeros((1, 96), dtype=torch.bool)
    valid[:, :12] = True
    sample = runner.readout_samples_from_embeddings(
        video_ids=("short",),
        embeddings=embeddings,
        valid_mask=valid,
        timeline_lengths=torch.tensor([96]),
        config=_config(),
    )[0]
    assert sample.period_lag_eligible is False
    assert sample.carrier_eligible is False
    assert sample.selected_peak_total == 0
    assert sample.selection_rule == "ineligible_no_peak_readout"


def test_embedding_shuffle_is_deterministic_and_preserves_invalid_slots() -> None:
    embeddings = torch.arange(2 * 16 * 4, dtype=torch.float32).reshape(2, 16, 4)
    valid = torch.tensor(
        [
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    first = runner._deterministically_shuffle_valid_embeddings(
        embeddings, valid, ("alpha", "beta")
    )
    second = runner._deterministically_shuffle_valid_embeddings(
        embeddings, valid, ("alpha", "beta")
    )
    assert torch.equal(first, second)
    assert torch.equal(first[~valid], embeddings[~valid])
    for index in range(2):
        assert torch.equal(
            first[index, valid[index], 0].sort().values,
            embeddings[index, valid[index], 0].sort().values,
        )


def test_time_scale_aggregation_uses_dynamic_bounds_and_eligible_intersection() -> None:
    baseline = (
        _readout_sample(video_id="a", period=16.0, peak_total=8),
        _readout_sample(video_id="b", period=24.0, peak_total=5),
    )
    scaled = {
        0.75: (
            _readout_sample(video_id="a-s", period=12.0, peak_total=8, upper=96),
            _readout_sample(video_id="b-s", period=18.0, peak_total=5, upper=96),
        ),
        1.25: (
            _readout_sample(video_id="a-l", period=20.0, peak_total=8, upper=160),
            _readout_sample(
                video_id="b-l",
                period=30.0,
                peak_total=0,
                carrier_eligible=False,
                upper=160,
            ),
        ),
    }
    result = runner.time_scale_aggregates_from_samples(
        baseline, scaled, factors=(0.75, 1.25), minimum_period=4
    )
    assert result["period"]["eligible_comparison_share"] == 1.0
    assert result["period"]["relative_error"]["median"] == pytest.approx(0.0)
    assert result["peak_total"]["eligible_comparison_total"] == 3
    assert result["peak_total"]["exact_share"] == 1.0
    assert result["per_video_rows_persisted"] is False


def test_native_period_boundary_uses_each_sample_half_timeline_upper_bound() -> None:
    samples = (
        _readout_sample(video_id="a", period=48.0, peak_total=3, upper=48),
        _readout_sample(video_id="b", period=48.0, peak_total=4, upper=96),
    )
    distribution = runner._period_distribution(samples, minimum_period=4)
    assert distribution["boundary_share"] == 0.5
    assert distribution["upper_boundary_share"] == 0.5


def test_every_frozen_threshold_is_authorizing() -> None:
    specification = runner.load_gate_specification(SPECIFICATION)
    passing = runner.gate_decision(
        candidate_id="A",
        aggregates=_passing_aggregates(),
        thresholds=specification.thresholds,
    )
    assert passing["overall_pass"] is True
    assert passing["criterion_total"] == len(specification.thresholds) == 25
    for name in specification.thresholds:
        thresholds = dict(specification.thresholds)
        if name.endswith("maximum") or name.endswith("maximum_exclusive"):
            thresholds[name] = -1.0
        else:
            thresholds[name] = 2.1
        decision = runner.gate_decision(
            candidate_id="A",
            aggregates=_passing_aggregates(),
            thresholds=thresholds,
        )
        assert decision["overall_pass"] is False, name


def test_candidate_registry_reservation_is_exclusive_across_attempt_ids() -> None:
    gate_sha = "a" * 64
    pose_sha = "b" * 64
    source_sha = "c" * 40
    with tempfile.TemporaryDirectory(prefix="pams-gate-") as directory:
        root = Path(directory)
        registry = root / "registry"
        registry.mkdir()
        attempt = root / "attempt.reservation.json"
        registry_id = runner.candidate_registry_id(
            source_git_sha=source_sha,
            gate_specification_sha256=gate_sha,
            pose_cache_set_sha256=pose_sha,
            candidate_id="A",
        )
        attempt.write_text(
            json.dumps(
                {
                    "attempt_id": "first-attempt",
                    "source_revision": source_sha,
                    "candidate_id": "A",
                    "candidate_registry": {
                        "registry_id": registry_id,
                        "run_locator": "runs/native/first-attempt",
                        "exclusive_first_pass_required": True,
                    },
                    "candidate_launch_policy": {
                        "terminal_gate_specification_sha256": gate_sha,
                        "required_prior_scientific_rejection_candidate_ids": [],
                        "authorization_must_exist_before_encoder_container_creation": True,
                        "same_authorization_pair_required_for_epoch11_and_final": True,
                    },
                    "upstream_pose_recovery": {
                        "pose_cache_set_sha256": pose_sha,
                    },
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        def reserve() -> tuple[Path, dict[str, object], tuple[str, int]]:
            return runner.reserve_candidate_registry_slot(
                registry,
                source_git_sha=source_sha,
                gate_specification_sha256=gate_sha,
                pose_cache_set_sha256=pose_sha,
                candidate_id="A",
                run_reservation_path=attempt,
                run_locator="runs/native/first-attempt",
                source_export_receipt_sha256="d" * 64,
                experiment_config_sha256="e" * 64,
                pose_snapshot_sha256="f" * 64,
                prior_outcome_registry_ids=(),
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(reserve) for _ in range(2)]
        successes = []
        failures = []
        for future in futures:
            try:
                successes.append(future.result())
            except FileExistsError as error:
                failures.append(error)
        assert len(successes) == 1
        assert len(failures) == 1
        assert successes[0][1]["registry_id"] == registry_id

        second_attempt = root / "second-attempt.reservation.json"
        second_payload = json.loads(attempt.read_text(encoding="utf-8"))
        second_payload["attempt_id"] = "second-attempt"
        second_payload["candidate_registry"]["run_locator"] = (
            "runs/native/second-attempt"
        )
        second_attempt.write_text(
            json.dumps(second_payload, sort_keys=True) + "\n", encoding="utf-8"
        )
        with pytest.raises(FileExistsError):
            runner.reserve_candidate_registry_slot(
                registry,
                source_git_sha=source_sha,
                gate_specification_sha256=gate_sha,
                pose_cache_set_sha256=pose_sha,
                candidate_id="A",
                run_reservation_path=second_attempt,
                run_locator="runs/native/second-attempt",
                source_export_receipt_sha256="d" * 64,
                experiment_config_sha256="e" * 64,
                pose_snapshot_sha256="f" * 64,
                prior_outcome_registry_ids=(),
            )


def test_registry_predecessor_binds_train_terminal_and_launch_lineage() -> None:
    source_sha = "c" * 40
    gate_sha = "a" * 64
    pose_sha = "b" * 64
    digest = "d" * 64
    with tempfile.TemporaryDirectory(prefix="pams-gate-") as directory:
        root = Path(directory)
        registry = root / "registry"
        registry.mkdir()
        key = runner._candidate_registry_key(
            source_git_sha=source_sha,
            gate_specification_sha256=gate_sha,
            pose_cache_set_sha256=pose_sha,
            candidate_id="A",
        )
        registry_id = runner.sha256_json(key)
        paths = runner._candidate_registry_paths(registry, registry_id)
        reservation = {
            "schema_version": 1,
            "artifact_type": runner._CANDIDATE_REGISTRY_RESERVATION_TYPE,
            "status": "reserved",
            "registry_id": registry_id,
            "key": key,
            "run_binding": {
                "attempt_id": "fixture",
                "run_locator": "runs/native/fixture",
                "attempt_reservation_sha256": "1" * 64,
                "attempt_reservation_bytes": 1,
                "source_export_receipt_sha256": digest,
                "experiment_config_sha256": digest,
                "pose_snapshot_sha256": digest,
            },
            "prior_outcome_registry_ids": [],
        }
        runner._write_new(paths["reservation"], runner._encoded_json(reservation))
        reservation_identity = runner._stable_file_sha256(paths["reservation"])
        audit_roles = (
            "create_id",
            "configuration_inspect",
            "configuration_verification",
            "post_run_inspect",
            "exit_code",
        )
        audits: dict[str, object] = {
            stage: {
                role: {
                    "locator": f"audit/{stage}.{role}.json",
                    "sha256": "9" * 64,
                    "bytes": 1,
                }
                for role in audit_roles
            }
            for stage in (
                "preflight",
                "launch_authorization",
                "encoder_epoch11",
                "epoch11_gate",
                "encoder_final",
            )
        }
        audit_commitment = runner.sha256_json(audits)
        lineage = {
            "container_audits_sha256_commitment": audit_commitment,
            "candidate_launch_authorization_sha256": digest,
            "candidate_launch_authorization_bytes": 103,
            "candidate_launch_authorization_receipt_sha256": digest,
            "candidate_launch_authorization_receipt_bytes": 104,
            "epoch11_completion_receipt_sha256": "3" * 64,
            "epoch11_completion_receipt_bytes": 201,
            "epoch11_gate_artifact_sha256": digest,
            "epoch11_gate_artifact_bytes": 101,
            "epoch11_gate_receipt_sha256": digest,
            "epoch11_gate_receipt_bytes": 102,
            "final_started_receipt_sha256": "4" * 64,
            "final_started_receipt_bytes": 301,
            "final_completion_receipt_sha256": "5" * 64,
            "final_completion_receipt_bytes": 302,
            "final_checkpoint_sha256": "6" * 64,
            "final_checkpoint_bytes": 303,
            "final_progress_sha256": "7" * 64,
            "final_progress_bytes": 304,
            "final_pose_snapshot_sha256": "8" * 64,
            "final_pose_snapshot_bytes": 305,
        }
        train_receipt = {
            "status": "completed",
            "candidate_id": "A",
            "source_revision": source_sha,
            "source_export_receipt_sha256": digest,
            "config_file_sha256": digest,
            "pose_cache_set_sha256": pose_sha,
            "candidate_registry": {
                "registry_id": registry_id,
                "reservation_sha256": reservation_identity[0],
                "reservation_bytes": reservation_identity[1],
                "exclusive_first_pass_reservation": True,
            },
            "candidate_launch_authorization": {
                "authorization_sha256": digest,
                "authorization_bytes": 103,
                "authorization_receipt_sha256": digest,
                "authorization_receipt_bytes": 104,
            },
            "epoch11_gate": {
                "encoder_completion_receipt_sha256": "3" * 64,
                "encoder_completion_receipt_bytes": 201,
                "gate_artifact_sha256": digest,
                "gate_artifact_bytes": 101,
                "gate_receipt_sha256": digest,
                "gate_receipt_bytes": 102,
            },
            "final_encoder": {
                "started_receipt_sha256": "4" * 64,
                "started_receipt_bytes": 301,
                "completion_receipt_sha256": "5" * 64,
                "completion_receipt_bytes": 302,
                "checkpoint_sha256": "6" * 64,
                "checkpoint_bytes": 303,
                "progress_sha256": "7" * 64,
                "progress_bytes": 304,
                "pose_snapshot_sha256": "8" * 64,
                "pose_snapshot_bytes": 305,
            },
            "container_audits": audits,
            "container_audits_sha256_commitment": audit_commitment,
        }
        train_path = root / "run.receipt.json"
        runner._write_new(train_path, runner._encoded_json(train_receipt))
        train_identity = runner._stable_file_sha256(train_path)
        payload = _minimal_terminal_payload(
            candidate_id="A",
            gate_specification_sha256=gate_sha,
            pose_cache_set_sha256=pose_sha,
            source_git_sha=source_sha,
            prior=[],
            passed=False,
        )
        payload["inputs"].update(
            {
                "candidate_registry_id": registry_id,
                "candidate_registry_reservation_sha256": reservation_identity[0],
                "candidate_registry_reservation_bytes": reservation_identity[1],
                "train_run_receipt_sha256": train_identity[0],
                "train_run_receipt_bytes": train_identity[1],
                "container_audits_sha256_commitment": audit_commitment,
                "train_lineage": lineage,
            }
        )
        terminal_path = root / "terminal.json"
        terminal_receipt, _ = runner.write_gate_artifact(terminal_path, payload)
        runner.write_candidate_registry_outcome(
            registry,
            train_run_receipt_path=train_path,
            terminal_artifact_path=terminal_path,
            terminal_receipt_path=terminal_receipt,
            payload=payload,
        )
        chain = runner._load_predecessor_chain_from_registry(
            candidate_id="B",
            profile=runner.CandidateProfile(16, 2, ("A",)),
            registry_root=registry,
            specification=runner.load_gate_specification(SPECIFICATION),
            gate_specification_sha256=gate_sha,
            pose_cache_set_sha256=pose_sha,
            source_git_sha=source_sha,
        )
        assert chain[0]["registry_id"] == registry_id
        assert chain[0]["train_run_receipt_sha256"] == train_identity[0]
        assert chain[0]["container_audits_sha256_commitment"] == audit_commitment
        archived_train_path = paths["train_run_receipt"]
        archived_train = runner._strict_json(
            archived_train_path, document="archived train tamper fixture"
        )
        archived_train["container_audits"]["preflight"]["exit_code"][
            "sha256"
        ] = "0" * 64
        archived_train_path.chmod(0o600)
        archived_train_path.write_bytes(runner._encoded_json(archived_train))
        with pytest.raises(ValueError, match="outcome binding mismatch"):
            runner._load_predecessor_chain_from_registry(
                candidate_id="B",
                profile=runner.CandidateProfile(16, 2, ("A",)),
                registry_root=registry,
                specification=runner.load_gate_specification(SPECIFICATION),
                gate_specification_sha256=gate_sha,
                pose_cache_set_sha256=pose_sha,
                source_git_sha=source_sha,
            )


def test_launch_authorization_pair_binds_candidate_and_training_inputs() -> None:
    specification = runner.load_gate_specification(SPECIFICATION)
    config = _config()
    profile = specification.candidate_profiles["A"]
    digest = "a" * 64
    source_sha = "b" * 40
    registry_reservation = {
        "registry_id": "3" * 64,
        "run_binding": {"run_locator": "runs/native/fixture"},
    }
    registry_reservation_identity = ("4" * 64, 16)
    identities = {
        "experiment_config": (digest, 10),
        "gate_specification": ("c" * 64, 11),
        "pose_snapshot": ("d" * 64, 12),
        "source_export_receipt": ("e" * 64, 13),
        "launch_authorization_runner": ("f" * 64, 14),
        "gate_runner": ("1" * 64, 15),
    }
    covered = {
        "launch_authorization_runner": (
            "scripts/server/prepare_pams_native_candidate_launch_authorization.py"
        ),
        "terminal_gate_runner": (
            "scripts/server/run_pams_native_terminal_readout_gate.py"
        ),
        "experiment_config": (
            "configs/experiments/pams_native_table2_baseline_proxy_v1.yaml"
        ),
        "gate_specification": (
            "configs/gates/pams_native_terminal_readout_gate_v1.yaml"
        ),
    }
    inputs = {
        "experiment_config_sha256": identities["experiment_config"][0],
        "experiment_config_bytes": identities["experiment_config"][1],
        "gate_specification_sha256": identities["gate_specification"][0],
        "gate_specification_bytes": identities["gate_specification"][1],
        "pose_snapshot_sha256": identities["pose_snapshot"][0],
        "pose_snapshot_bytes": identities["pose_snapshot"][1],
        "pose_cache_set_sha256": "2" * 64,
        "source_export_receipt_sha256": identities["source_export_receipt"][0],
        "source_export_receipt_bytes": identities["source_export_receipt"][1],
        "launch_authorization_runner_sha256": identities[
            "launch_authorization_runner"
        ][0],
        "launch_authorization_runner_bytes": identities[
            "launch_authorization_runner"
        ][1],
        "terminal_gate_runner_sha256": identities["gate_runner"][0],
        "terminal_gate_runner_bytes": identities["gate_runner"][1],
        "config_fingerprint": config.fingerprint,
        "pose_fingerprint": config.pose_fingerprint,
        "training_video_total": 337,
        "source_git_sha": source_sha,
        "source_receipt_covered_paths": covered,
        "candidate_registry_id": registry_reservation["registry_id"],
        "candidate_registry_reservation_sha256": (
            registry_reservation_identity[0]
        ),
        "candidate_registry_reservation_bytes": (
            registry_reservation_identity[1]
        ),
        "candidate_registry_run_locator": "runs/native/fixture",
    }
    payload = {
        "schema_version": 1,
        "artifact_type": runner._CANDIDATE_LAUNCH_ARTIFACT_TYPE,
        "status": "candidate_training_authorized",
        "classification": specification.classification,
        "protocol": specification.expected_protocol,
        "seed": specification.expected_seed,
        "candidate": runner._candidate_mapping("A", profile, ()),
        "inputs": inputs,
        "authorization": {
            "gate_frozen_before_candidate_a": True,
            "candidate_training_authorized": True,
            "encoder_training_scope": "train337_only",
            "terminal_checkpoint_or_prediction_authorized": False,
            "dev84_identity_media_pose_or_scoring_authorized": False,
            "test105_evaluation_authorized": False,
            "aggregate_only_prior_receipts": True,
            "candidate_outcome_registry_reserved_exclusively": True,
        },
    }
    with tempfile.TemporaryDirectory(prefix="pams-gate-") as directory:
        output = Path(directory) / "launch.json"
        receipt, _ = runner.write_candidate_launch_authorization(output, payload)
        runner._validate_candidate_launch_authorization(
            output,
            receipt,
            candidate_id="A",
            profile=profile,
            prior_chain=(),
            specification=specification,
            config=config,
            identities=identities,
            pose_cache_set_sha256="2" * 64,
            source_git_sha=source_sha,
            source_covered_paths=covered,
            registry_reservation=registry_reservation,
            registry_reservation_identity=registry_reservation_identity,
        )
        broken = runner._strict_json(receipt, document="launch receipt fixture")
        broken["candidate_id"] = "B"
        receipt.unlink()
        runner._write_new(receipt, runner._encoded_json(broken))
        with pytest.raises(ValueError, match="receipt binding mismatch"):
            runner._validate_candidate_launch_authorization(
                output,
                receipt,
                candidate_id="A",
                profile=profile,
                prior_chain=(),
                specification=specification,
                config=config,
                identities=identities,
                pose_cache_set_sha256="2" * 64,
                source_git_sha=source_sha,
                source_covered_paths=covered,
                registry_reservation=registry_reservation,
                registry_reservation_identity=registry_reservation_identity,
            )


def test_artifact_is_aggregate_only_canonical_and_write_once() -> None:
    payload = _minimal_terminal_payload(
        candidate_id="A",
        gate_specification_sha256="a" * 64,
        pose_cache_set_sha256="b" * 64,
        source_git_sha="c" * 40,
        prior=[],
        passed=True,
    )
    with tempfile.TemporaryDirectory(prefix="pams-gate-") as directory:
        output = Path(directory) / "artifact.json"
        receipt, digest = runner.write_gate_artifact(output, payload)
        assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
        decoded = runner._strict_json(output, document="terminal artifact fixture")
        keys = runner._walk_mapping_keys(decoded)
        assert "samples" not in keys
        assert "rows" not in keys
        assert "video_id" not in keys
        assert decoded["aggregates"]["per_video_rows_persisted"] is False
        receipt_payload = runner._strict_json(
            receipt, document="terminal artifact receipt fixture"
        )
        assert set(receipt_payload) == runner._RECEIPT_KEYS
        assert receipt_payload["aggregate_only"] is True
        with pytest.raises(FileExistsError):
            runner.write_gate_artifact(output, payload)


def test_epoch11_artifact_and_receipt_are_hash_and_lineage_bound() -> None:
    config = _config()
    source_sha = "c" * 40
    pose_sha = "b" * 64
    config_sha = "a" * 64
    snapshot_sha = "e" * 64
    launch_identity = ("9" * 64, 17)
    launch_receipt_identity = ("0" * 64, 18)
    with tempfile.TemporaryDirectory(prefix="pams-gate-") as directory:
        root = Path(directory)
        artifact_path = root / "epoch11.json"
        receipt_path = root / "epoch11.receipt.json"
        artifact = {
            "schema_version": 1,
            "artifact_type": runner._EXPECTED_EPOCH11_ARTIFACT_TYPE,
            "status": "encoder_continuation_authorized",
            "classification": "fixture",
            "protocol": "ucfrep_526",
            "seed": 2026,
            "label_firewall": {
                "accepted_scientific_inputs": ["fixture"],
                "manifest_interface_supported": False,
                "media_interface_supported": False,
                "external_label_fields_accessed": [],
                "privileged_interface_fields_and_paths_rejected": True,
                "training_interface_supported": False,
            },
            "inputs": {
                "experiment_config_sha256": config_sha,
                "pose_snapshot_sha256": snapshot_sha,
                "config_fingerprint": config.fingerprint,
                "pose_fingerprint": config.pose_fingerprint,
                "pose_cache_set_sha256": pose_sha,
                "training_video_total": 337,
                "source_git_sha": source_sha,
                "encoder_checkpoint_sha256": "1" * 64,
                "encoder_checkpoint_bytes": 11,
                "encoder_progress_sha256": "2" * 64,
                "encoder_progress_bytes": 12,
                "encoder_completion_receipt_sha256": "4" * 64,
                "encoder_completion_receipt_bytes": 13,
                "experiment_config_bytes": 14,
                "gate_specification_sha256": "3" * 64,
                "gate_specification_bytes": 15,
                "pose_snapshot_bytes": 16,
                "candidate_launch_authorization_sha256": launch_identity[0],
                "candidate_launch_authorization_bytes": launch_identity[1],
                "candidate_launch_authorization_receipt_sha256": (
                    launch_receipt_identity[0]
                ),
                "candidate_launch_authorization_receipt_bytes": (
                    launch_receipt_identity[1]
                ),
                "training_video_ids_sha256": "5" * 64,
                "source_receipt_covered_paths": {},
                "container_image_id": "sha256:" + "6" * 64,
                "container_environment_sha256": "7" * 64,
                "encoder_completion_receipt": {
                    "schema_version": 3,
                    "run_id": "epoch11-fixture",
                    "sha256": "4" * 64,
                    "bytes": 13,
                    "completed_epochs": 11,
                    "artifact_roles": sorted(
                        {
                            "input_config",
                            "input_dataset_manifest",
                            "input_pose_cache_snapshot",
                            "input_train_pose_inputs",
                            "input_train_pose_input_commitment",
                            "input_dev_pose_inputs",
                            "input_dev_pose_input_commitment",
                            "input_test_identity_pose_inputs",
                            "input_test_identity_pose_input_commitment",
                            "input_candidate_launch_authorization",
                            "input_candidate_launch_authorization_receipt",
                            "output_encoder_checkpoint",
                            "progress_log",
                        }
                    ),
                    "candidate_launch_authorization_sha256": launch_identity[0],
                    "candidate_launch_authorization_bytes": launch_identity[1],
                    "candidate_launch_authorization_receipt_sha256": (
                        launch_receipt_identity[0]
                    ),
                    "candidate_launch_authorization_receipt_bytes": (
                        launch_receipt_identity[1]
                    ),
                },
            },
            "schedule": {
                "completed_epochs": 11,
                "period_source": "fixed_period_inferred",
                "loss_first_three_median": 1.0,
                "loss_final_three_median": 0.5,
                "loss_relative_drop": 0.5,
                "fixed_period_evidence_fraction_mean": 1.0,
                "optimizer_steps_total": 11,
                "prototype_bank_negative_tallies_all_zero": True,
                "progress_exactly_matches_checkpoint_history": True,
            },
            "representation": {
                "record_total": 337,
                "temporal_rms": _summary(1.0),
                "near_collapse_rms": 0.001,
                "near_collapsed_fraction": 0.0,
                "lag_eligible_total": 337,
                "lag_eligible_fraction": 1.0,
                "real_cycle_margin": _summary(1.0),
                "shuffled_cycle_margin": _summary(0.0),
                "zero_cycle_margin": _summary(0.0),
                "stronger_null_separation": _summary(1.0),
                "real_beats_both_nulls_fraction": 1.0,
                "fixed_period_carrier_advisory": {
                    "authorization_role": "diagnostic_only",
                    "real_available_fraction": 1.0,
                    "real_support": _summary(1.0),
                    "real_gate_energy": _summary(1.0),
                    "shuffled_available_fraction": 0.0,
                    "shuffled_support": _summary(0.0),
                    "shuffled_gate_energy": _summary(0.0),
                    "zero_available_fraction": 0.0,
                    "zero_support": _summary(0.0),
                    "zero_gate_energy": _summary(0.0),
                    "shuffled_to_real_gate_energy_median_ratio": 0.0,
                },
            },
            "gate": {
                "thresholds_frozen_before_native_candidate_training": True,
                "criteria": {
                    name: {
                        "value": 1.0,
                        "operator": ">=",
                        "threshold": 0.0,
                        "pass": True,
                    }
                    for name in {
                        "loss_relative_drop",
                        "fixed_period_evidence_fraction",
                        "near_collapsed_fraction",
                        "lag_eligible_fraction",
                        "real_cycle_margin_median",
                        "stronger_null_separation_median",
                        "real_beats_both_nulls_fraction",
                    }
                },
                "encoder_continuation_authorized": True,
                "prediction_or_scoring_authorized": False,
                "all_core_criteria_pass": True,
            },
            "scientific_scope": {
                "fixed_training_period_frames": 16,
                "dense_frame_indices_preserved": True,
                "invalid_slots_never_compacted": True,
                "real_shuffled_and_zero_views_share_one_valid_mask": True,
                "fixed_period_carrier_is_advisory_at_epoch11": True,
                "adaptive_time_scale_criterion_used": False,
            },
            "read_only_verification": {
                "all_file_inputs_unchanged": True,
                "pose_cache_set_unchanged": True,
                "model_state_sha256_before": "8" * 64,
                "model_state_sha256_after": "8" * 64,
                "model_or_optimizer_state_updated": False,
                "training_steps_executed": 0,
                "pose_cache_write_operations": 0,
            },
        }
        encoded = runner._encoded_json(artifact)
        runner._write_new(artifact_path, encoded)
        receipt = {
            "schema_version": 1,
            "artifact_type": runner._EXPECTED_EPOCH11_RECEIPT_TYPE,
            "artifact_sha256": hashlib.sha256(encoded).hexdigest(),
            "artifact_bytes": len(encoded),
            "artifact_status": "encoder_continuation_authorized",
            "encoder_continuation_authorized": True,
            "encoder_checkpoint_sha256": "1" * 64,
            "encoder_progress_sha256": "2" * 64,
            "encoder_completion_receipt_sha256": "4" * 64,
            "candidate_launch_authorization_sha256": launch_identity[0],
            "candidate_launch_authorization_bytes": launch_identity[1],
            "candidate_launch_authorization_receipt_sha256": (
                launch_receipt_identity[0]
            ),
            "candidate_launch_authorization_receipt_bytes": (
                launch_receipt_identity[1]
            ),
            "pose_cache_set_sha256": pose_sha,
            "gate_specification_sha256": "3" * 64,
            "source_git_sha": source_sha,
        }
        runner._write_new(receipt_path, runner._encoded_json(receipt))
        _, validated_receipt = runner._validate_epoch11_gate_pair(
            artifact_path,
            receipt_path,
            config=config,
            specification=runner.load_gate_specification(SPECIFICATION),
            config_sha256=config_sha,
            pose_snapshot_sha256=snapshot_sha,
            pose_cache_set_sha256=pose_sha,
            source_git_sha=source_sha,
            candidate_launch_authorization_identity=launch_identity,
            candidate_launch_receipt_identity=launch_receipt_identity,
        )
        assert validated_receipt["artifact_sha256"] == hashlib.sha256(encoded).hexdigest()
        broken = deepcopy(receipt)
        broken["candidate_launch_authorization_bytes"] += 1
        receipt_path.unlink()
        runner._write_new(receipt_path, runner._encoded_json(broken))
        with pytest.raises(ValueError, match="does not bind"):
            runner._validate_epoch11_gate_pair(
                artifact_path,
                receipt_path,
                config=config,
                specification=runner.load_gate_specification(SPECIFICATION),
                config_sha256=config_sha,
                pose_snapshot_sha256=snapshot_sha,
                pose_cache_set_sha256=pose_sha,
                source_git_sha=source_sha,
                candidate_launch_authorization_identity=launch_identity,
                candidate_launch_receipt_identity=launch_receipt_identity,
            )


def test_completion_receipt_requires_epoch11_resume_lineage_and_started_bytes() -> None:
    config = _config()
    source_sha = "c" * 40
    container = {
        "image_id": "sha256:" + "1" * 64,
        "environment_sha256": "2" * 64,
        "source_revision": source_sha,
    }
    provenance = SimpleNamespace(
        dataset_fingerprint="4" * 64,
        container_image_id=container["image_id"],
        container_environment_sha256=container["environment_sha256"],
        source_git_sha=source_sha,
    )
    with tempfile.TemporaryDirectory(prefix="pams-gate-") as directory:
        root = Path(directory)
        role_to_identity_key = {
            "input_config": "experiment_config",
            "input_pose_cache_snapshot": "pose_snapshot",
            "output_encoder_checkpoint": "encoder_checkpoint",
            "progress_log": "encoder_progress",
            "input_candidate_launch_authorization": (
                "candidate_launch_authorization"
            ),
            "input_candidate_launch_authorization_receipt": (
                "candidate_launch_authorization_receipt"
            ),
        }
        role_paths: dict[str, Path] = {}
        identities: dict[str, tuple[str, int]] = {}
        for role, identity_key in role_to_identity_key.items():
            target = root / f"{role}.bin"
            runner._write_new(target, f"fixture:{role}".encode())
            role_paths[role] = target
            identities[identity_key] = runner._stable_file_sha256(target)
        for role in ("input_resume_checkpoint", "input_resume_progress"):
            target = root / f"{role}.bin"
            runner._write_new(target, f"fixture:{role}".encode())
            role_paths[role] = target
        resume_checkpoint_identity = runner._stable_file_sha256(
            role_paths["input_resume_checkpoint"]
        )
        resume_progress_identity = runner._stable_file_sha256(
            role_paths["input_resume_progress"]
        )
        epoch11_receipt = {
            "encoder_checkpoint_sha256": resume_checkpoint_identity[0],
            "encoder_progress_sha256": resume_progress_identity[0],
        }
        epoch11_artifact = {
            "inputs": {
                "encoder_checkpoint_bytes": resume_checkpoint_identity[1],
                "encoder_progress_bytes": resume_progress_identity[1],
            }
        }
        started = RunManifest(
            schema_version=2,
            receipt_type="started",
            run_id="terminal-fixture",
            created_at_utc="2026-08-07T00:00:00+00:00",
            command=[
                "pams",
                "train",
                "encoder",
                "--candidate-launch-authorization",
                "authorization.json",
                "--candidate-launch-receipt",
                "authorization.receipt.json",
            ],
            git_sha=source_sha,
            config_sha256=config.fingerprint,
            dataset_sha256=provenance.dataset_fingerprint,
            seed=2026,
            protocol="ucfrep_526",
            status="started",
            hardware={"container": container},
        )
        started_encoded = runner._encoded_json(started.model_dump(mode="json"))
        artifacts = tuple(
            ArtifactReceipt(
                role=role,
                locator=path.name,
                sha256=runner._stable_file_sha256(path)[0],
                bytes=runner._stable_file_sha256(path)[1],
            )
            for role, path in role_paths.items()
        )
        completion = CompletedRunReceipt(
            schema_version=3,
            receipt_type="completed",
            run_id=started.run_id,
            status="completed",
            finished_at="2026-08-07T00:01:00+00:00",
            start_manifest_sha256=hashlib.sha256(started_encoded).hexdigest(),
            started=started,
            artifacts=artifacts,
            metrics={"completed_epochs": 150},
        )
        started_path = root / f"{started.run_id}.started.json"
        completion_path = root / "terminal.completed.json"
        runner._write_new(started_path, started_encoded)
        runner._write_new(
            completion_path,
            runner._encoded_json(completion.model_dump(mode="json")),
        )
        parsed, validated_started_path, _, resolved = (
            runner._validate_encoder_completion_receipt(
                completion_path,
                expected_source_git_sha=source_sha,
                specification=runner.load_gate_specification(SPECIFICATION),
                config=config,
                identities=identities,
                epoch11_artifact=epoch11_artifact,
                epoch11_receipt=epoch11_receipt,
            )
        )
        assert parsed.run_id == started.run_id
        assert validated_started_path == started_path
        assert set(resolved) == set(role_paths)
        runner._validate_completion_provenance(parsed, provenance)
        with pytest.raises(ValueError, match="dataset fingerprint mismatch"):
            runner._validate_completion_provenance(
                parsed,
                SimpleNamespace(
                    dataset_fingerprint="0" * 64,
                    container_image_id=provenance.container_image_id,
                    container_environment_sha256=(
                        provenance.container_environment_sha256
                    ),
                    source_git_sha=provenance.source_git_sha,
                ),
            )

        tampered_launch = completion.model_copy(
            update={
                "artifacts": tuple(
                    item.model_copy(update={"bytes": item.bytes + 1})
                    if item.role == "input_candidate_launch_authorization"
                    else item
                    for item in artifacts
                )
            }
        )
        completion_path.unlink()
        runner._write_new(
            completion_path,
            runner._encoded_json(tampered_launch.model_dump(mode="json")),
        )
        with pytest.raises(ValueError):
            runner._validate_encoder_completion_receipt(
                completion_path,
                expected_source_git_sha=source_sha,
                specification=runner.load_gate_specification(SPECIFICATION),
                config=config,
                identities=identities,
                epoch11_artifact=epoch11_artifact,
                epoch11_receipt=epoch11_receipt,
            )
        completion_path.unlink()
        runner._write_new(
            completion_path,
            runner._encoded_json(completion.model_dump(mode="json")),
        )

        missing_resume = completion.model_copy(
            update={
                "artifacts": tuple(
                    item for item in artifacts if item.role != "input_resume_progress"
                )
            }
        )
        completion_path.unlink()
        runner._write_new(
            completion_path,
            runner._encoded_json(missing_resume.model_dump(mode="json")),
        )
        with pytest.raises(ValueError, match="missing required artifact roles"):
            runner._validate_encoder_completion_receipt(
                completion_path,
                expected_source_git_sha=source_sha,
                specification=runner.load_gate_specification(SPECIFICATION),
                config=config,
                identities=identities,
                epoch11_artifact=epoch11_artifact,
                epoch11_receipt=epoch11_receipt,
            )
