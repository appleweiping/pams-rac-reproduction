from __future__ import annotations

import hashlib
import inspect
import json
import math
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from torch.nn import functional as F

from pams.config import PAMSConfig
from pams.run_manifest import ArtifactReceipt, CompletedRunReceipt, RunManifest
from pams.training import EncoderEpochStats, _progress_row
from scripts.server import run_pams_native_epoch11_train_gate as runner

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "server" / "run_pams_native_epoch11_train_gate.py"
SPECIFICATION = ROOT / "configs" / "gates" / "pams_native_epoch11_train_gate_v1.yaml"


def _periodic_embeddings(
    *,
    batch: int = 1,
    frames: int = 96,
    period: int = 16,
) -> torch.Tensor:
    time = torch.arange(frames, dtype=torch.float32)
    phase = 2.0 * math.pi * time / period
    base = torch.stack(
        (
            torch.cos(phase),
            torch.sin(phase),
            0.25 * torch.cos(2.0 * phase),
            0.25 * torch.sin(2.0 * phase),
        ),
        dim=1,
    )
    return F.normalize(base, dim=1).unsqueeze(0).repeat(batch, 1, 1)


def _random_embeddings(*, batch: int, frames: int, dimension: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(913_771)
    return F.normalize(
        torch.randn((batch, frames, dimension), generator=generator),
        dim=2,
    )


def _mechanism_samples(
    real: torch.Tensor,
    shuffled: torch.Tensor,
    zero: torch.Tensor,
    *,
    valid: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
    period: int = 16,
):
    batch, frames, _ = real.shape
    if valid is None:
        valid = torch.ones((batch, frames), dtype=torch.bool)
    if lengths is None:
        lengths = torch.full((batch,), frames, dtype=torch.long)
    return runner.mechanism_samples_from_embeddings(
        video_ids=tuple(f"video-{index}" for index in range(batch)),
        real_embeddings=real,
        shuffled_embeddings=shuffled,
        zero_embeddings=zero,
        valid_mask=valid,
        timeline_lengths=lengths,
        period=period,
        minimum_pairs=8,
    )


def _passing_schedule() -> dict[str, float]:
    return {
        "loss_relative_drop": 0.20,
        "fixed_period_evidence_fraction_mean": 0.90,
    }


def test_gate_specification_is_preregistered_and_complete() -> None:
    specification = runner.load_gate_specification(SPECIFICATION)
    assert specification.expected_protocol == "ucfrep_526"
    assert specification.expected_seed == 2026
    assert specification.expected_training_video_total == 337
    assert specification.expected_completed_epochs == 11
    assert specification.expected_completion_receipt_schema_version == 3
    assert specification.minimum_lag_pair_total == 8
    assert specification.near_collapse_rms == pytest.approx(1e-3)
    assert set(specification.thresholds) == runner._THRESHOLD_KEYS


@pytest.mark.parametrize(
    ("period", "expected"),
    [
        (5, (3, 5, 8)),
        (7, (4, 7, 11)),
        (16, (8, 16, 24)),
        (24, (12, 24, 36)),
    ],
)
def test_recurrence_lags_use_explicit_round_half_up(
    period: int,
    expected: tuple[int, int, int],
) -> None:
    assert runner._recurrence_lags(period) == expected


def test_pose_conditioned_periodic_embeddings_pass_core_gate() -> None:
    real = _periodic_embeddings(batch=4)
    shuffled = _random_embeddings(batch=4, frames=96, dimension=4)
    zero = F.normalize(torch.ones_like(real), dim=2)
    samples = _mechanism_samples(real, shuffled, zero)
    distribution = runner._mechanism_distribution(
        samples,
        near_collapse_rms=1e-3,
    )
    specification = runner.load_gate_specification(SPECIFICATION)
    decision = runner.gate_decision(
        schedule=_passing_schedule(),
        distribution=distribution,
        thresholds=specification.thresholds,
    )
    assert distribution["lag_eligible_fraction"] == 1.0
    assert distribution["real_cycle_margin"]["median"] > 1.0
    assert distribution["stronger_null_separation"]["median"] > 1.0
    assert decision["encoder_continuation_authorized"] is True


def test_constant_embedding_collapse_is_rejected() -> None:
    constant = F.normalize(torch.ones((4, 96, 4)), dim=2)
    samples = _mechanism_samples(constant, constant, constant)
    distribution = runner._mechanism_distribution(
        samples,
        near_collapse_rms=1e-3,
    )
    specification = runner.load_gate_specification(SPECIFICATION)
    decision = runner.gate_decision(
        schedule=_passing_schedule(),
        distribution=distribution,
        thresholds=specification.thresholds,
    )
    assert distribution["near_collapsed_fraction"] == 1.0
    assert decision["criteria"]["near_collapsed_fraction"]["pass"] is False
    assert decision["encoder_continuation_authorized"] is False


def test_position_only_periodicity_fails_pose_conditioned_null_separation() -> None:
    position_only = _periodic_embeddings(batch=4)
    samples = _mechanism_samples(position_only, position_only, position_only)
    distribution = runner._mechanism_distribution(
        samples,
        near_collapse_rms=1e-3,
    )
    specification = runner.load_gate_specification(SPECIFICATION)
    decision = runner.gate_decision(
        schedule=_passing_schedule(),
        distribution=distribution,
        thresholds=specification.thresholds,
    )
    assert distribution["real_cycle_margin"]["median"] > 1.0
    assert distribution["stronger_null_separation"]["median"] == pytest.approx(0.0)
    assert decision["criteria"]["stronger_null_separation_median"]["pass"] is False
    assert decision["encoder_continuation_authorized"] is False


def test_invalid_padding_payload_cannot_change_mechanism_sample() -> None:
    real = _periodic_embeddings(frames=96)
    shuffled = _random_embeddings(batch=1, frames=96, dimension=4)
    zero = F.normalize(torch.ones_like(real), dim=2)
    base = _mechanism_samples(real, shuffled, zero)[0]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(73)

    def padded(values: torch.Tensor) -> torch.Tensor:
        tail = torch.randn((1, 32, 4), generator=generator)
        return torch.cat((values, tail), dim=1)

    valid = torch.cat(
        (
            torch.ones((1, 96), dtype=torch.bool),
            torch.zeros((1, 32), dtype=torch.bool),
        ),
        dim=1,
    )
    polluted = _mechanism_samples(
        padded(real),
        padded(shuffled),
        padded(zero),
        valid=valid,
        lengths=torch.tensor([96]),
    )[0]
    assert asdict(polluted) == asdict(base)


def test_mechanism_metrics_are_batch_order_invariant() -> None:
    first = _periodic_embeddings(frames=96)
    second = torch.roll(first, shifts=3, dims=1)
    real = torch.cat((first, second), dim=0)
    shuffled = _random_embeddings(batch=2, frames=96, dimension=4)
    zero = F.normalize(torch.ones_like(real), dim=2)
    forward = _mechanism_samples(real, shuffled, zero)
    reverse = runner.mechanism_samples_from_embeddings(
        video_ids=("video-1", "video-0"),
        real_embeddings=real.flip(0),
        shuffled_embeddings=shuffled.flip(0),
        zero_embeddings=zero.flip(0),
        valid_mask=torch.ones((2, 96), dtype=torch.bool),
        timeline_lengths=torch.tensor([96, 96]),
        period=16,
        minimum_pairs=8,
    )
    assert {sample.video_id: asdict(sample) for sample in forward} == {
        sample.video_id: asdict(sample) for sample in reverse
    }


def test_pose_shuffle_is_deterministic_and_preserves_invalid_rows() -> None:
    poses = torch.arange(2 * 12 * 33 * 3, dtype=torch.float32).reshape(2, 12, 33, 3)
    valid = torch.tensor(
        [
            [1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    first = runner._shuffle_valid_pose_rows(poses, valid, ("alpha", "beta"))
    second = runner._shuffle_valid_pose_rows(poses, valid, ("alpha", "beta"))
    assert torch.equal(first, second)
    assert torch.equal(first[~valid], poses[~valid])
    for index in range(2):
        expected = poses[index, valid[index], 0, 0].sort().values
        actual = first[index, valid[index], 0, 0].sort().values
        assert torch.equal(actual, expected)


def test_lag_eligibility_is_explicit_and_low_coverage_fails() -> None:
    real = _periodic_embeddings(batch=4, frames=40)
    shuffled = _random_embeddings(batch=4, frames=40, dimension=4)
    zero = F.normalize(torch.ones_like(real), dim=2)
    valid = torch.ones((4, 40), dtype=torch.bool)
    valid[1:, 8:] = False
    samples = _mechanism_samples(real, shuffled, zero, valid=valid)
    distribution = runner._mechanism_distribution(
        samples,
        near_collapse_rms=1e-3,
    )
    specification = runner.load_gate_specification(SPECIFICATION)
    decision = runner.gate_decision(
        schedule=_passing_schedule(),
        distribution=distribution,
        thresholds=specification.thresholds,
    )
    assert distribution["lag_eligible_total"] == 1
    assert distribution["lag_eligible_fraction"] == 0.25
    assert decision["criteria"]["lag_eligible_fraction"]["pass"] is False


def _fixed_config() -> PAMSConfig:
    payload = yaml.safe_load(
        (ROOT / "configs" / "ablations" / "pams_fixed_period16_inferred.yaml").read_text(
            encoding="utf-8"
        )
    )
    payload["loss"]["anchor_stride"] = 4
    payload["loss"]["use_cross_cluster_negatives"] = False
    return PAMSConfig.model_validate(payload)


def _history() -> tuple[EncoderEpochStats, ...]:
    return tuple(
        EncoderEpochStats(
            epoch=epoch,
            loss=3.0 - 0.15 * epoch,
            learning_rate=1e-4,
            period_source="fixed_period_inferred",
            period_confidence_mean=0.9,
            period_valid_fraction=0.9,
            optimizer_steps=10,
            clusters_refreshed=epoch in {1, 6, 11},
            cross_cluster_requested=0,
            cross_cluster_actual=0,
            cross_cluster_shortfall=0,
        )
        for epoch in range(1, 12)
    )


def _write_completion_receipt(
    path: Path,
    *,
    config: PAMSConfig,
    extra_role: str | None = None,
) -> tuple[str, dict[str, tuple[str, int]], SimpleNamespace]:
    source = "a" * 40
    image = "sha256:" + "b" * 64
    environment = "c" * 64
    dataset = "d" * 64
    identities = {
        "encoder_checkpoint": ("1" * 64, 101),
        "encoder_progress": ("2" * 64, 102),
        "experiment_config": ("3" * 64, 103),
        "pose_snapshot": ("4" * 64, 104),
    }
    started = RunManifest(
        run_id="20260807T000000Z-fixture",
        created_at_utc="2026-08-07T00:00:00+00:00",
        command=[
            "python",
            "-m",
            "pams",
            "train",
            "encoder",
            "--epochs",
            "11",
            "--candidate-launch-authorization",
            "/pams/launch/authorization.json",
            "--candidate-launch-receipt",
            "/pams/launch/authorization.json.receipt.json",
        ],
        git_sha=source,
        config_sha256=config.fingerprint,
        dataset_sha256=dataset,
        seed=2026,
        protocol="ucfrep_526",
        hardware={
            "container": {
                "image_id": image,
                "environment_sha256": environment,
                "source_revision": source,
            }
        },
    )
    role_hashes = {
        "input_config": identities["experiment_config"],
        "input_dataset_manifest": ("5" * 64, 105),
        "input_pose_cache_snapshot": identities["pose_snapshot"],
        "input_train_pose_inputs": ("5" * 64, 105),
        "input_train_pose_input_commitment": ("6" * 64, 106),
        "input_dev_pose_inputs": ("7" * 64, 107),
        "input_dev_pose_input_commitment": ("8" * 64, 108),
        "input_test_identity_pose_inputs": ("9" * 64, 109),
        "input_test_identity_pose_input_commitment": ("0" * 64, 110),
        "input_candidate_launch_authorization": ("a" * 64, 111),
        "input_candidate_launch_authorization_receipt": ("b" * 64, 112),
        "output_encoder_checkpoint": identities["encoder_checkpoint"],
        "progress_log": identities["encoder_progress"],
    }
    if extra_role is not None:
        role_hashes[extra_role] = ("e" * 64, 111)
    artifacts = tuple(
        ArtifactReceipt(
            role=role,
            locator=f"inputs/{index:02d}.artifact",
            sha256=digest,
            bytes=byte_count,
        )
        for index, (role, (digest, byte_count)) in enumerate(sorted(role_hashes.items()))
    )
    receipt = CompletedRunReceipt(
        run_id=started.run_id,
        finished_at="2026-08-07T00:01:00+00:00",
        start_manifest_sha256="f" * 64,
        started=started,
        artifacts=artifacts,
        metrics={"completed_epochs": 11, "final_epoch": {"epoch": 11}},
    )
    encoded = (
        json.dumps(receipt.model_dump(mode="json"), sort_keys=True, indent=2) + "\n"
    ).encode()
    path.write_bytes(encoded)
    provenance = SimpleNamespace(
        source_git_sha=source,
        dataset_fingerprint=dataset,
    )
    return hashlib.sha256(encoded).hexdigest(), identities, provenance


def test_completion_receipt_is_caller_pinned_and_exact(tmp_path: Path) -> None:
    config = _fixed_config()
    specification = runner.load_gate_specification(SPECIFICATION)
    path = tmp_path / "completion.receipt.json"
    digest, identities, provenance = _write_completion_receipt(path, config=config)
    summary = runner._validate_encoder_completion_receipt(
        path,
        expected_sha256=digest,
        identities=identities,
        config=config,
        specification=specification,
        provenance=provenance,
        runtime_image="sha256:" + "b" * 64,
        runtime_environment="c" * 64,
        runtime_source="a" * 40,
    )
    assert summary["sha256"] == digest
    assert summary["completed_epochs"] == 11


def test_completion_receipt_rejects_extra_artifact_role(tmp_path: Path) -> None:
    config = _fixed_config()
    path = tmp_path / "completion.receipt.json"
    digest, identities, provenance = _write_completion_receipt(
        path,
        config=config,
        extra_role="unexpected_artifact",
    )
    with pytest.raises(ValueError, match="artifact roles"):
        runner._validate_encoder_completion_receipt(
            path,
            expected_sha256=digest,
            identities=identities,
            config=config,
            specification=runner.load_gate_specification(SPECIFICATION),
            provenance=provenance,
            runtime_image="sha256:" + "b" * 64,
            runtime_environment="c" * 64,
            runtime_source="a" * 40,
        )


def _write_partial_checkpoint(
    path: Path,
    *,
    config: PAMSConfig,
    history: tuple[EncoderEpochStats, ...],
) -> None:
    torch.save(
        {
            "schema_version": 5,
            "stage": "encoder",
            "config_fingerprint": config.fingerprint,
            "provenance": {},
            "completed_epochs": len(history),
            "model_state": {},
            "optimizer_state": {},
            "scheduler_state": {},
            "history": [
                {
                    key: value
                    for key, value in asdict(item).items()
                    if key != "position_permutation_consistency"
                }
                for item in history
            ],
            "cluster_assignments": {},
            "prototype_bank": {},
            "rng_state": {},
        },
        path,
    )


def _write_progress(
    path: Path,
    *,
    config: PAMSConfig,
    history: tuple[EncoderEpochStats, ...],
) -> None:
    path.write_text(
        "".join(
            json.dumps(
                _progress_row(stage="encoder", stats=item, config=config),
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for item in history
        ),
        encoding="utf-8",
    )


def test_partial_checkpoint_and_progress_are_exactly_reconciled(tmp_path: Path) -> None:
    config = _fixed_config()
    history = _history()
    checkpoint = tmp_path / "encoder.pt"
    progress = tmp_path / "encoder.jsonl"
    _write_partial_checkpoint(checkpoint, config=config, history=history)
    _write_progress(progress, config=config, history=history)
    specification = runner.load_gate_specification(SPECIFICATION)
    metadata, decoded = runner._checkpoint_epoch_metadata(
        checkpoint,
        progress,
        config=config,
        specification=specification,
    )
    assert len(decoded) == 11
    assert metadata["completed_epochs"] == 11
    assert metadata["progress_exactly_matches_checkpoint_history"] is True
    assert metadata["loss_relative_drop"] > 0.10


def test_partial_checkpoint_rejects_progress_drift(tmp_path: Path) -> None:
    config = _fixed_config()
    history = _history()
    checkpoint = tmp_path / "encoder.pt"
    progress = tmp_path / "encoder.jsonl"
    _write_partial_checkpoint(checkpoint, config=config, history=history)
    _write_progress(progress, config=config, history=history)
    rows = progress.read_text(encoding="utf-8").splitlines()
    payload = json.loads(rows[-1])
    payload["stats"]["loss"] += 1.0
    rows[-1] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    progress.write_text("\n".join(rows) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="progress rows"):
        runner._checkpoint_epoch_metadata(
            checkpoint,
            progress,
            config=config,
            specification=runner.load_gate_specification(SPECIFICATION),
        )


def test_partial_checkpoint_rejects_noncanonical_history_types(tmp_path: Path) -> None:
    config = _fixed_config()
    history = _history()
    checkpoint = tmp_path / "encoder.pt"
    progress = tmp_path / "encoder.jsonl"
    _write_partial_checkpoint(checkpoint, config=config, history=history)
    _write_progress(progress, config=config, history=history)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["history"][0]["optimizer_steps"] = 10.0
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="optimizer_steps must be an integer"):
        runner._checkpoint_epoch_metadata(
            checkpoint,
            progress,
            config=config,
            specification=runner.load_gate_specification(SPECIFICATION),
        )


def test_fixed_candidate_contract_rejects_prototype_bank_negatives() -> None:
    config = _fixed_config()
    payload = config.model_dump(mode="json")
    payload["loss"]["use_cross_cluster_negatives"] = True
    with pytest.raises(ValueError, match="prototype-bank negatives off"):
        runner._validate_candidate_config(
            PAMSConfig.model_validate(payload),
            runner.load_gate_specification(SPECIFICATION),
        )


def test_interface_has_no_privileged_scientific_arguments() -> None:
    assert set(inspect.signature(runner.run_gate).parameters) == {
        "encoder_checkpoint_path",
        "encoder_progress_path",
        "encoder_completion_receipt_path",
        "expected_encoder_completion_receipt_sha256",
        "config_path",
        "gate_specification_path",
        "pose_cache_dir",
        "pose_snapshot_path",
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
    ):
        assert f'add_argument("{option}"' not in source


def test_runner_and_gate_controls_are_source_receipt_covered(tmp_path: Path) -> None:
    covered = runner._require_source_tree_membership(
        ROOT,
        {
            "gate_runner": RUNNER,
            "gate_specification": SPECIFICATION,
        },
    )
    assert covered == {
        "gate_runner": "scripts/server/run_pams_native_epoch11_train_gate.py",
        "gate_specification": "configs/gates/pams_native_epoch11_train_gate_v1.yaml",
    }
    outside = tmp_path / "gate.yaml"
    outside.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the source root"):
        runner._require_source_tree_membership(ROOT, {"gate_specification": outside})


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


@pytest.mark.parametrize("field", sorted(runner._FORBIDDEN_FIELD_NAMES))
def test_privileged_json_fields_are_rejected_before_use(field: str) -> None:
    with pytest.raises(ValueError, match="forbidden privileged field"):
        runner._strict_json(json.dumps({field: "never materialized"}), document="fixture")


def test_every_gate_criterion_is_required() -> None:
    specification = runner.load_gate_specification(SPECIFICATION)
    real = _periodic_embeddings(batch=4)
    samples = _mechanism_samples(
        real,
        _random_embeddings(batch=4, frames=96, dimension=4),
        F.normalize(torch.ones_like(real), dim=2),
    )
    distribution = runner._mechanism_distribution(samples, near_collapse_rms=1e-3)
    passing = runner.gate_decision(
        schedule=_passing_schedule(),
        distribution=distribution,
        thresholds=specification.thresholds,
    )
    assert passing["encoder_continuation_authorized"] is True
    for threshold_name in specification.thresholds:
        thresholds = dict(specification.thresholds)
        if threshold_name.endswith("maximum"):
            thresholds[threshold_name] = -1.0
        else:
            thresholds[threshold_name] = 2.1
        failing = runner.gate_decision(
            schedule=_passing_schedule(),
            distribution=distribution,
            thresholds=thresholds,
        )
        assert failing["encoder_continuation_authorized"] is False
