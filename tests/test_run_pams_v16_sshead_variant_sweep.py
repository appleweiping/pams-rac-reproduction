from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from pams.config import PAMSConfig, load_config
from pams.training import CheckpointProvenance
from scripts.server import run_pams_v16_sshead_variant_sweep as runner


def _base_config() -> PAMSConfig:
    root = Path(__file__).resolve().parents[1]
    return load_config(root / "configs/experiments/pams_noabs_projected_teacher_v16.yaml")


def _variant_entries(
    base: PAMSConfig,
) -> tuple[tuple[Path, PAMSConfig, str, int], ...]:
    entries: list[tuple[Path, PAMSConfig, str, int]] = []
    for index, (source, weight) in enumerate(
        sorted(runner._ALLOWED_COMBINATIONS),
        start=1,
    ):
        config = base.model_copy(
            update={
                "sshead": base.sshead.model_copy(
                    update={
                        "input_source": source,
                        "variance_weight": weight,
                    }
                )
            }
        )
        variant_id = runner._variant_id(source, weight)
        entries.append(
            (
                Path(f"variant-{index}.yaml"),
                config,
                runner._EXPECTED_VARIANT_IDENTITIES[variant_id]["config_sha256"],
                index,
            )
        )
    return tuple(entries)


def _artifact_payload() -> dict[str, object]:
    digest = "a" * 64
    return {
        "status": "train337_sshead_variant_sweep_completed",
        "inputs": {
            "encoder_checkpoint_sha256": digest,
            "encoder_progress_sha256": digest,
            "base_config_sha256": digest,
            "train337_pose_cache_set_sha256": digest,
        },
        "variant_outputs_sha256": digest,
        "variants": [{}, {}, {}, {}],
        "hardware_sha256": digest,
        "runtime_sha256": digest,
    }


def test_cli_is_train337_only_and_requires_four_repeatable_variant_configs() -> None:
    parsed = runner._parse_arguments(
        [
            "--encoder-checkpoint",
            "encoder.pt",
            "--encoder-progress",
            "encoder.jsonl",
            "--base-config",
            "base.yaml",
            "--variant-config",
            "a.yaml",
            "--variant-config",
            "b.yaml",
            "--variant-config",
            "c.yaml",
            "--variant-config",
            "d.yaml",
            "--pose-cache-dir",
            "train337-pose",
            "--output-root",
            "sweep",
        ]
    )
    assert parsed.variant_configs == [
        Path("a.yaml"),
        Path("b.yaml"),
        Path("c.yaml"),
        Path("d.yaml"),
    ]
    assert set(inspect.signature(runner.run_variant_sweep).parameters) == {
        "encoder_checkpoint_path",
        "encoder_progress_path",
        "base_config_path",
        "variant_config_paths",
        "pose_cache_dir",
        "output_root",
        "device",
        "microbatch_size",
    }
    source = inspect.getsource(runner._parse_arguments)
    for forbidden in (
        "--manifest",
        "--source-receipt",
        "--action",
        "--count",
        "--label",
        "--target",
        "--dev",
        "--test",
        "--predict",
        "--score",
    ):
        assert f'"{forbidden}"' not in source
        assert f"'{forbidden}'" not in source


def test_exact_base_encoder_progress_and_config_hashes_are_required() -> None:
    identities = {
        "encoder_checkpoint": (
            "491fd5df67fb4d1fc7669019e4f78fac7b5c681dfa94a4cc7a469da2f73f5196",
            1,
        ),
        "encoder_progress": (
            "047c6af919843cd5d1133033ae07314b8ddd9426d878e4b6519181b94eb13ec7",
            2,
        ),
        "base_config": (runner._EXPECTED_CONFIG_SHA256, 3),
    }
    runner._validate_exact_base_inputs(identities)
    identities["encoder_progress"] = ("0" * 64, 2)
    with pytest.raises(ValueError, match="exact frozen v16"):
        runner._validate_exact_base_inputs(identities)


def test_variant_matrix_allows_only_two_fields_and_exact_2x2_coverage() -> None:
    base = _base_config()
    reports = runner._validate_variant_configs(base, _variant_entries(base))

    assert len(reports) == 4
    assert {
        (report["input_source"], report["variance_weight"]) for report in reports
    } == runner._ALLOWED_COMBINATIONS
    assert [report["variant_id"] for report in reports] == sorted(
        report["variant_id"] for report in reports
    )

    byte_drift = list(_variant_entries(base))
    path, config, _, byte_count = byte_drift[0]
    byte_drift[0] = (path, config, "0" * 64, byte_count)
    with pytest.raises(ValueError, match="raw SHA/fingerprint"):
        runner._validate_variant_configs(base, byte_drift)

    duplicate = list(_variant_entries(base))
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="duplicate"):
        runner._validate_variant_configs(base, duplicate)

    drift = list(_variant_entries(base))
    path, config, digest, byte_count = drift[0]
    changed = config.model_copy(
        update={"training": config.training.model_copy(update={"learning_rate": 0.0002})}
    )
    drift[0] = (path, changed, digest, byte_count)
    with pytest.raises(ValueError, match="drifted outside"):
        runner._validate_variant_configs(base, drift)

    outside = list(_variant_entries(base))
    path, config, digest, byte_count = outside[0]
    changed_head = config.model_copy(
        update={"sshead": config.sshead.model_copy(update={"variance_weight": 2.0})}
    )
    outside[0] = (path, changed_head, digest, byte_count)
    with pytest.raises(ValueError, match="outside"):
        runner._validate_variant_configs(base, outside)


def test_variant_provenance_binds_same_train337_source_container_and_encoder() -> None:
    encoder = CheckpointProvenance(
        protocol="ucfrep_526",
        dataset_fingerprint="1" * 64,
        training_video_ids=tuple(f"video-{index:03d}" for index in range(337)),
        pose_fingerprint="2" * 64,
        pose_cache_set_sha256="3" * 64,
        source_git_sha="4" * 40,
        container_image_id=f"sha256:{'5' * 64}",
        container_environment_sha256="6" * 64,
        upstream_encoder_checkpoint_sha256=None,
    )
    upstream_sha256 = "7" * 64

    variant = runner._variant_provenance(
        encoder,
        encoder_checkpoint_sha256=upstream_sha256,
    )

    assert variant.protocol == encoder.protocol
    assert variant.dataset_fingerprint == encoder.dataset_fingerprint
    assert variant.training_video_ids == encoder.training_video_ids
    assert variant.pose_fingerprint == encoder.pose_fingerprint
    assert variant.pose_cache_set_sha256 == encoder.pose_cache_set_sha256
    assert variant.source_git_sha == encoder.source_git_sha
    assert variant.container_image_id == encoder.container_image_id
    assert variant.container_environment_sha256 == encoder.container_environment_sha256
    assert variant.upstream_encoder_checkpoint_sha256 == upstream_sha256


def test_training_source_explicitly_reinitializes_head_and_checks_tensor_binding() -> None:
    fresh_source = inspect.getsource(runner._fresh_variant_model)
    run_source = inspect.getsource(runner.run_variant_sweep)

    assert "torch.manual_seed(config.seed)" in fresh_source
    assert "build_pams_model(config)" in fresh_source
    assert "model.encoder.load_state_dict" in fresh_source
    assert "validate_terminal_checkpoint(" in run_source
    assert "validate_sshead_encoder_binding(" in run_source
    assert "resume=False" in run_source
    assert '"dev84_prediction_authorized": False' in run_source
    assert '"dev84_scoring_authorized": False' in run_source
    assert '"test105_evaluation_authorized": False' in run_source


def test_sweep_artifact_receipt_is_exclusive_hash_bound_and_never_authorizes(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "sweep"
    output_root.mkdir()
    payload = _artifact_payload()

    artifact_path, receipt_path, digest = runner._write_artifact_and_receipt(
        output_root,
        payload,
    )

    artifact = artifact_path.read_bytes()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert digest == hashlib.sha256(artifact).hexdigest()
    assert receipt["artifact_sha256"] == digest
    assert receipt["artifact_bytes"] == len(artifact)
    assert receipt["variant_total"] == 4
    assert receipt["dev84_prediction_authorized"] is False
    assert receipt["dev84_scoring_authorized"] is False
    assert receipt["test105_evaluation_authorized"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        runner._write_artifact_and_receipt(output_root, payload)
