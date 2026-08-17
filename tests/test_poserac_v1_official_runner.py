from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pams.baselines import poserac_v1_official_runner as runner
from pams.baselines.repnet_official_runner import _load_label_free_inputs
from pams.data import (
    PoseInputCommitment,
    PoseInputManifest,
    PoseSequence,
    UnlabeledVideoRecord,
    pose_cache_path,
    pose_input_identity_sha256,
    write_pose_cache,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _asset(tmp_path: Path, name: str, content: bytes) -> runner.VerifiedAsset:
    path = tmp_path / name
    path.write_bytes(content)
    spec = runner.AssetSpec(
        name=name,
        byte_count=len(content),
        sha256=_sha256(content),
        role=f"fixture_{name}",
        url="https://example.invalid/fixture",
    )
    return runner._verify_asset_with_spec(path, spec)


def _input_pair(
    tmp_path: Path,
    record: UnlabeledVideoRecord,
) -> tuple[Path, Path, Path]:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split="dev",
        records=(record,),
    )
    sidecar = tmp_path / "inputs.json"
    sidecar.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    commitment = PoseInputCommitment(
        protocol=manifest.protocol,
        split=manifest.split,
        record_total=1,
        identity_sha256=pose_input_identity_sha256(manifest.records),
        sidecar_sha256=_sha256(sidecar.read_bytes()),
        sidecar_fingerprint=manifest.fingerprint,
    )
    commitment_path = tmp_path / "inputs.commitment.json"
    commitment_path.write_text(
        json.dumps(commitment.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    return sidecar, commitment_path, video_root


class _FakeBackend:
    source_commit = runner.OFFICIAL_SOURCE_COMMIT
    source_tree_sha256 = runner.OFFICIAL_SOURCE_TREE_SHA256
    runtime_versions = {
        "python": "3.11.10",
        "torch": "2.5.1",
        "device": "fixture",
    }
    restore_audit = {
        "checkpoint_key_count": 74,
        "model_key_count": 74,
        "loaded_key_count": 74,
        "missing_keys": [],
        "unexpected_keys": [],
        "strict_key_coverage": True,
    }

    def predict(
        self,
        sequence: PoseSequence,
        config: runner.PoseRACV1OfficialConfig,
    ) -> runner.BackendPrediction:
        assert sequence.num_frames == config.expected_frames
        return runner.BackendPrediction(
            count=3,
            selected_channel=2,
            selected_action="squat",
            channel_counts=(0, 1, 3, 0, 2, 0, 0, 1),
            channel_dynamic_ranges=(0.1, 0.2, 0.9, 0.0, 0.3, 0.1, 0.1, 0.2),
            probability_min=0.01,
            probability_max=0.99,
            frames=sequence.num_frames,
            valid_frames=int(np.sum(sequence.valid_mask)),
        )


def test_frozen_assets_architecture_and_protocol_classification() -> None:
    config = runner.FROZEN_POSERAC_V1_CONFIG

    assert runner.OFFICIAL_SOURCE_COMMIT == (
        "469590b611bde3595eaf163b517263da634e2096"
    )
    assert runner.SOURCE_ARCHIVE_SPEC.byte_count == 62_833_091
    assert runner.CHECKPOINT_SPEC.byte_count == 10_772_679
    assert runner.CHECKPOINT_SPEC.sha256 == (
        "21afc8334333e5f5751f0e5d765376a778415285c0ac3078d8358ccb83b2b34a"
    )
    assert (config.dim, config.heads, config.encoder_layers, config.classes) == (
        99,
        9,
        6,
        8,
    )
    assert config.expected_frames == 256
    assert config.channel_selection == "smoothed_dynamic_range_max_lowest_index"
    assert len(config.fingerprint) == 64
    assert runner.ACTION_NAMES == (
        "front_raise",
        "pull_up",
        "squat",
        "bench_pressing",
        "jump_jack",
        "situp",
        "push_up",
        "pommelhorse",
    )
    assert "v1" in runner.METHOD_ID
    assert "ICONIP24" not in runner.METHOD_ID


def test_official_axis_normalization_is_per_coordinate_and_masks_invalid() -> None:
    xyz = np.zeros((2, 33, 3), dtype=np.float32)
    xyz[0, :, 0] = np.linspace(-5, 5, 33)
    xyz[0, :, 1] = np.linspace(10, 30, 33)
    xyz[0, :, 2] = np.linspace(-2, -1, 33)
    xyz[1] = 99
    output = runner.official_axis_normalize(
        xyz,
        np.asarray([True, False], dtype=np.bool_),
    )

    np.testing.assert_allclose(output[0].min(axis=0), np.zeros(3), atol=1e-7)
    np.testing.assert_allclose(output[0].max(axis=0), np.ones(3), atol=1e-7)
    assert np.count_nonzero(output[1]) == 0


def test_released_smoothing_trigger_and_inferred_dynamic_range_tie_break() -> None:
    probabilities = np.full((5, 8), 0.5, dtype=np.float32)
    probabilities[:, 3] = np.asarray([0.1, 0.9, 0.1, 0.9, 0.1])
    probabilities[:, 6] = probabilities[:, 3]
    smoothed = runner.smooth_scores(probabilities, momentum=0.0)
    selected, ranges = runner.select_dynamic_range_channel(smoothed)

    assert selected == 3
    assert ranges[3] == pytest.approx(ranges[6])
    assert runner.count_trigger_cycles(
        smoothed[:, selected],
        enter_threshold=0.78,
        exit_threshold=0.4,
    ) == 2


def test_exact_checkpoint_restore_has_complete_74_key_coverage(tmp_path: Path) -> None:
    model = runner.ExactPoseRACV1(runner.FROZEN_POSERAC_V1_CONFIG)
    checkpoint_path = tmp_path / "fixture.pth"
    torch.save(model.state_dict(), checkpoint_path)
    checkpoint = runner._verify_asset_with_spec(
        checkpoint_path,
        runner.AssetSpec(
            name=checkpoint_path.name,
            byte_count=checkpoint_path.stat().st_size,
            sha256=_sha256(checkpoint_path.read_bytes()),
            role="fixture_checkpoint",
            url="https://example.invalid/checkpoint",
        ),
    )

    backend = runner.ExactCheckpointBackend.from_checkpoint(
        checkpoint,
        runner.FROZEN_POSERAC_V1_CONFIG,
        device="cpu",
    )
    assert backend.restore_audit == {
        "checkpoint_format": "plain_ordered_tensor_state_dict",
        "checkpoint_key_count": 74,
        "model_key_count": 74,
        "loaded_key_count": 74,
        "missing_keys": [],
        "unexpected_keys": [],
        "strict_key_coverage": True,
        "model_parameter_count": 2_686_682,
        "fc1_weight_shape": [8, 99],
    }


def test_verified_run_is_label_free_hashes_caches_and_marks_protocol_gap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_id = "fixture-dev-video"
    video_sha256 = "a" * 64
    pose_fingerprint = "b" * 64
    record = UnlabeledVideoRecord(
        video_id=video_id,
        video_path="fixture.avi",
        video_sha256=video_sha256,
    )
    sidecar, commitment, video_root = _input_pair(tmp_path, record)
    inputs = _load_label_free_inputs(
        sidecar,
        commitment,
        video_root,
        require_exact_membership=False,
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    phase = np.linspace(0, 8 * np.pi, 256, dtype=np.float32)
    xyz = np.zeros((256, 33, 3), dtype=np.float32)
    xyz[:, :, 0] = (np.sin(phase) + 1.0)[:, None]
    xyz[:, :, 1] = np.linspace(0.0, 1.0, 33)[None, :]
    sequence = PoseSequence(
        video_id=video_id,
        fps=30.0,
        xyz=xyz,
        valid_mask=np.ones(256, dtype=np.bool_),
    )
    write_pose_cache(
        pose_cache_path(cache_dir, video_id),
        sequence,
        video_sha256=video_sha256,
        pose_fingerprint=pose_fingerprint,
    )
    source_archive = _asset(tmp_path, "source.fixture", b"source")
    checkpoint = _asset(tmp_path, "checkpoint.fixture", b"checkpoint")
    source_root = tmp_path / "source-root"
    source_root.mkdir()
    monkeypatch.setattr(
        runner,
        "_source_tree_digest",
        lambda _root: (
            runner.OFFICIAL_SOURCE_TREE_FILES,
            runner.OFFICIAL_SOURCE_TREE_BYTES,
            runner.OFFICIAL_SOURCE_TREE_SHA256,
        ),
    )

    result = runner._run_verified_backend(
        inputs=inputs,
        source_archive=source_archive,
        checkpoint=checkpoint,
        source_root=source_root,
        cache_dir=cache_dir,
        pose_fingerprint=pose_fingerprint,
        backend=_FakeBackend(),
        runner_git_sha="c" * 40,
        runner_code_sha256="d" * 64,
    )
    payload = result.to_dict()

    assert payload["labels_loaded"] is False
    assert payload["scoring_performed"] is False
    assert payload["ground_truth_count_channel_oracle"] is False
    assert payload["eligible_for_original_poserac_v1_cell"] is False
    assert payload["method_version"] == "PoseRAC-v1-2023"
    assert payload["distinct_from"] == "PoseRAC-ICONIP24"
    assert payload["selection"]["rule_provenance"] == "inferred"
    assert payload["predictions"][0]["raw_count"] == 3.0
    assert payload["predictions"][0]["channel_selection_uses_count_label"] is False
    assert payload["input"]["pose_cache_snapshot"]["entry_count"] == 1
    assert len(payload["input"]["pose_cache_snapshot"]["fingerprint"]) == 64
    output = tmp_path / "predictions.json"
    written, prediction_sha256, receipt_path, receipt_sha256 = (
        runner.write_poserac_v1_result_exclusive(output, result)
    )
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert written == output
    assert prediction_sha256 == _sha256(output.read_bytes())
    assert receipt_sha256 == _sha256(receipt_path.read_bytes())
    assert (
        receipt_payload["compatibility_image_digest"]
        == runner.COMPATIBILITY_IMAGE_DIGEST
    )
    assert receipt_payload["ground_truth_count_channel_oracle"] is False
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.write_poserac_v1_result_exclusive(output, result)


def test_cache_temporal_protocol_and_pose_model_are_hard_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_id = "fixture-dev-video"
    record = UnlabeledVideoRecord(video_id, "fixture.avi", "a" * 64)
    sidecar, commitment, video_root = _input_pair(tmp_path, record)
    inputs = _load_label_free_inputs(
        sidecar,
        commitment,
        video_root,
        require_exact_membership=False,
    )
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    sequence = PoseSequence(
        video_id,
        30.0,
        np.zeros((255, 33, 3), dtype=np.float32),
        np.zeros(255, dtype=np.bool_),
    )
    write_pose_cache(
        pose_cache_path(cache_dir, video_id),
        sequence,
        video_sha256="a" * 64,
        pose_fingerprint="b" * 64,
        pose_model="wrong-model",
    )
    source_root = tmp_path / "source-root"
    source_root.mkdir()
    monkeypatch.setattr(
        runner,
        "_source_tree_digest",
        lambda _root: (
            runner.OFFICIAL_SOURCE_TREE_FILES,
            runner.OFFICIAL_SOURCE_TREE_BYTES,
            runner.OFFICIAL_SOURCE_TREE_SHA256,
        ),
    )

    with pytest.raises(runner.PoseRACV1PoseCacheError, match="pose model mismatch"):
        runner._run_verified_backend(
            inputs=inputs,
            source_archive=_asset(tmp_path, "source.fixture", b"source"),
            checkpoint=_asset(tmp_path, "checkpoint.fixture", b"checkpoint"),
            source_root=source_root,
            cache_dir=cache_dir,
            pose_fingerprint="b" * 64,
            backend=_FakeBackend(),
            runner_git_sha="c" * 40,
            runner_code_sha256="d" * 64,
        )


def test_asset_tamper_and_source_tree_mutation_are_detected(tmp_path: Path) -> None:
    asset = _asset(tmp_path, "fixture.bin", b"official")
    asset.assert_unchanged()
    asset.path.write_bytes(b"tampered")
    with pytest.raises(runner.PoseRACV1OfficialError, match="changed"):
        asset.assert_unchanged()

    source = tmp_path / "tree"
    source.mkdir()
    (source / "a.py").write_text("print('a')\n", encoding="utf-8")
    first = runner._source_tree_digest(source)
    (source / "a.py").write_text("print('b')\n", encoding="utf-8")
    second = runner._source_tree_digest(source)
    assert first != second


def test_public_runner_has_no_scoring_or_label_argument() -> None:
    parameters = inspect.signature(runner.run_poserac_v1_official).parameters
    forbidden = {
        "action",
        "count",
        "ground_truth",
        "gt",
        "label",
        "score",
        "target",
    }
    assert forbidden.isdisjoint(parameters)
    assert "metrics" not in {
        imported
        for imported in runner.__dict__
        if imported.startswith("pams.")
    }
