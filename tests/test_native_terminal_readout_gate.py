from __future__ import annotations

import hashlib
import inspect
import math
import tempfile
from copy import deepcopy
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
    digest = "d" * 64
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
        "epoch11_gate_receipt_sha256": digest,
        "source_git_sha": source_git_sha,
        "code_files_sha256_commitment": digest,
    }
    next_candidate = None
    if not passed and candidate_id != "C":
        next_candidate = runner._CANDIDATE_ORDER[
            runner._CANDIDATE_ORDER.index(candidate_id) + 1
        ]
    return {
        "schema_version": 1,
        "artifact_type": runner._EXPECTED_ARTIFACT_TYPE,
        "status": "terminal_readout_eligible" if passed else "scientific_rejection",
        "classification": "fixture",
        "protocol": "ucfrep_526",
        "seed": 2026,
        "candidate": {
            "id": candidate_id,
            "prior_scientific_rejections": prior,
        },
        "label_firewall": {},
        "inputs": inputs,
        "algorithm": {},
        "thresholds": {},
        "aggregates": {"per_video_rows_persisted": False},
        "hard_invariants": {"per_video_predictions_persisted": False},
        "gate": {
            "overall_pass": passed,
            "eligible_for_single_frozen_dev84_protocol_build": passed,
            "next_candidate_training_authorized": next_candidate,
        },
        "scientific_caveats": [],
        "hardware": {},
        "hardware_sha256": digest,
        "runtime": {},
        "runtime_sha256": digest,
        "read_only_verification": {},
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
        "epoch11_gate_artifact_path",
        "epoch11_gate_receipt_path",
        "pose_cache_dir",
        "pose_snapshot_path",
        "candidate_id",
        "prior_rejection_artifact_paths",
        "prior_rejection_receipt_paths",
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
        periods=torch.full((4,), 16.0),
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


def test_candidate_order_requires_scientific_rejection_receipts() -> None:
    specification_sha256 = "a" * 64
    pose_sha256 = "b" * 64
    source_sha = "c" * 40
    with tempfile.TemporaryDirectory(prefix="pams-gate-") as directory:
        root = Path(directory)
        output = root / "candidate-a.json"
        payload = _minimal_terminal_payload(
            candidate_id="A",
            gate_specification_sha256=specification_sha256,
            pose_cache_set_sha256=pose_sha256,
            source_git_sha=source_sha,
            prior=[],
            passed=False,
        )
        receipt, artifact_sha256 = runner.write_gate_artifact(output, payload)
        chain = runner._validate_predecessor_chain(
            candidate_id="B",
            profile=runner.CandidateProfile(16, 2, ("A",)),
            artifact_paths=(output,),
            receipt_paths=(receipt,),
            gate_specification_sha256=specification_sha256,
            pose_cache_set_sha256=pose_sha256,
            source_git_sha=source_sha,
        )
        assert chain == (
            {
                "candidate_id": "A",
                "artifact_sha256": artifact_sha256,
                "receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
            },
        )
        with pytest.raises(ValueError, match="requires exactly 1"):
            runner._validate_predecessor_chain(
                candidate_id="B",
                profile=runner.CandidateProfile(16, 2, ("A",)),
                artifact_paths=(),
                receipt_paths=(),
                gate_specification_sha256=specification_sha256,
                pose_cache_set_sha256=pose_sha256,
                source_git_sha=source_sha,
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
            "label_firewall": {},
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
            },
            "schedule": {},
            "representation": {},
            "gate": {
                "encoder_continuation_authorized": True,
                "prediction_or_scoring_authorized": False,
                "all_core_criteria_pass": True,
            },
            "scientific_scope": {},
            "read_only_verification": {},
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
        )
        assert validated_receipt["artifact_sha256"] == hashlib.sha256(encoded).hexdigest()
        broken = deepcopy(receipt)
        broken["artifact_sha256"] = "0" * 64
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
    identities = {
        "experiment_config": ("a" * 64, 1),
        "pose_snapshot": ("b" * 64, 1),
        "encoder_checkpoint": ("d" * 64, 1),
        "encoder_progress": ("e" * 64, 1),
    }
    epoch11_receipt = {
        "encoder_checkpoint_sha256": "f" * 64,
        "encoder_progress_sha256": "9" * 64,
    }
    epoch11_artifact = {
        "inputs": {
            "encoder_checkpoint_bytes": 1,
            "encoder_progress_bytes": 1,
        }
    }
    started = RunManifest(
        schema_version=2,
        receipt_type="started",
        run_id="terminal-fixture",
        created_at_utc="2026-08-07T00:00:00+00:00",
        command=["pams", "train", "encoder"],
        git_sha=source_sha,
        config_sha256=config.fingerprint,
        dataset_sha256=provenance.dataset_fingerprint,
        seed=2026,
        protocol="ucfrep_526",
        status="started",
        hardware={"container": container},
    )
    started_encoded = runner._encoded_json(started.model_dump(mode="json"))
    artifact_hashes = {
        "input_config": identities["experiment_config"][0],
        "input_pose_cache_snapshot": identities["pose_snapshot"][0],
        "input_resume_checkpoint": epoch11_receipt["encoder_checkpoint_sha256"],
        "input_resume_progress": epoch11_receipt["encoder_progress_sha256"],
        "output_encoder_checkpoint": identities["encoder_checkpoint"][0],
        "progress_log": identities["encoder_progress"][0],
    }
    artifacts = tuple(
        ArtifactReceipt(
            role=role,
            locator=f"{role}.bin",
            sha256=sha256,
            bytes=1,
        )
        for role, sha256 in artifact_hashes.items()
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
    with tempfile.TemporaryDirectory(prefix="pams-gate-") as directory:
        root = Path(directory)
        started_path = root / f"{started.run_id}.started.json"
        completion_path = root / "terminal.completed.json"
        runner._write_new(started_path, started_encoded)
        runner._write_new(
            completion_path,
            runner._encoded_json(completion.model_dump(mode="json")),
        )
        parsed, validated_started_path, _ = runner._validate_encoder_completion_receipt(
            completion_path,
            expected_source_git_sha=source_sha,
            specification=runner.load_gate_specification(SPECIFICATION),
            config=config,
            identities=identities,
            epoch11_artifact=epoch11_artifact,
            epoch11_receipt=epoch11_receipt,
        )
        assert parsed.run_id == started.run_id
        assert validated_started_path == started_path
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
