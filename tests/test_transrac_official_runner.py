from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from pams.baselines import transrac_official_runner as runner
from pams.data import (
    PoseInputCommitment,
    PoseInputManifest,
    UnlabeledVideoRecord,
    pose_input_identity_sha256,
)

FIXTURE_GIT_SHA = "a" * 40
FIXTURE_RUNNER_SHA256 = "b" * 64


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_inputs(
    tmp_path: Path,
    records: tuple[UnlabeledVideoRecord, ...],
) -> tuple[Path, Path]:
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split="dev",
        records=records,
    )
    sidecar = tmp_path / "inputs.json"
    sidecar.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    commitment = PoseInputCommitment(
        protocol=manifest.protocol,
        split=manifest.split,
        record_total=len(records),
        identity_sha256=pose_input_identity_sha256(records),
        sidecar_sha256=_sha256(sidecar.read_bytes()),
        sidecar_fingerprint=manifest.fingerprint,
    )
    commitment_path = tmp_path / "inputs.commitment.json"
    commitment_path.write_text(
        json.dumps(commitment.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    return sidecar, commitment_path


def _tiny_asset(tmp_path: Path, name: str, content: bytes) -> runner.VerifiedAsset:
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


class _FakeBackend:
    source_commit = runner.OFFICIAL_SOURCE_COMMIT
    source_tree_sha256 = runner.OFFICIAL_SOURCE_TREE_SHA256
    runtime_versions = {
        "python": "3.11.10",
        "torch": "2.5.1+cu124",
        "cuda_runtime": "12.4",
        "mmcv": "1.4.0",
        "timm": "0.4.12",
        "einops": "0.3.2",
        "kornia": "0.5.11",
        "gpu_name": "fixture",
    }
    restore_audit = {
        "checkpoint_epoch": 174,
        "checkpoint_filename_epoch_token": 171,
        "state_dict_keys": 230,
        "loaded_key_count": 230,
        "missing_keys": [],
        "unexpected_keys": [],
        "strict_key_coverage": True,
    }

    def predict(
        self,
        video_path: Path,
        config: runner.TransRACOfficialConfig,
    ) -> runner.BackendPrediction:
        assert config is runner.FROZEN_TRANSRAC_OFFICIAL_CONFIG
        if video_path.name == "decode.avi":
            raise runner.TransRACOfficialDecodeError("fixture decode failure")
        return runner.BackendPrediction(
            raw_count=3.5,
            decoded_frame_count=91,
            sampled_frame_count=64,
            density_min=0.0,
            density_max=0.5,
            density_mean=0.0546875,
            peak_allocated_mib=700.0,
            peak_reserved_mib=800.0,
        )


def test_frozen_identity_and_modern_compat_config() -> None:
    config = runner.FROZEN_TRANSRAC_OFFICIAL_CONFIG

    assert runner.OFFICIAL_SOURCE_COMMIT == (
        "68bdd4daa60ed7c3174a7f6bf86f6537b6fa0979"
    )
    assert config.num_frames == 64
    assert (config.resize_height, config.resize_width) == (224, 224)
    assert config.scales == (1, 4, 8)
    assert config.torch_seed == 1
    assert config.gpu_memory_limit_mib == 8192
    assert config.compatibility_image_digest == runner.COMPATIBILITY_IMAGE_DIGEST
    assert "modern compatibility" in runner.CLASSIFICATION.lower()
    assert len(config.fingerprint) == 64

    with pytest.raises(ValueError, match="schedule is frozen"):
        runner.TransRACOfficialConfig(scales=(1, 2, 4))
    with pytest.raises(ValueError, match="image digest is frozen"):
        runner.TransRACOfficialConfig(compatibility_image_digest="sha256:" + "0" * 64)


def test_asset_and_extracted_source_tree_tampering_are_rejected(
    tmp_path: Path,
) -> None:
    asset = _tiny_asset(tmp_path, "fixture.bin", b"official-bytes")
    asset.assert_unchanged()
    asset.path.write_bytes(b"counterfeit!!")
    with pytest.raises(RuntimeError, match="changed"):
        asset.assert_unchanged()

    source = tmp_path / "source"
    source.mkdir()
    (source / "a.py").write_bytes(b"print('a')\n")
    (source / "nested").mkdir()
    (source / "nested" / "b.txt").write_bytes(b"b")
    file_count, byte_count, digest = runner._source_tree_digest(source)
    spec = runner.SourceTreeSpec(file_count, byte_count, digest)
    assert runner._verify_source_tree_with_spec(source, spec) == source.resolve()
    (source / "nested" / "b.txt").write_bytes(b"x")
    with pytest.raises(runner.TransRACOfficialSourceError, match="mismatch"):
        runner._verify_source_tree_with_spec(source, spec)


def test_asset_full_rehash_detects_same_size_tamper_with_restored_mtime(
    tmp_path: Path,
) -> None:
    asset = _tiny_asset(tmp_path, "fixture.bin", b"official-bytes")
    metadata = asset.path.stat()
    asset.path.write_bytes(b"tampered-bytes")
    os.utime(
        asset.path,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
    )

    with pytest.raises(
        runner.TransRACOfficialAssetError,
        match="bytes/SHA-256 changed",
    ):
        asset.assert_unchanged()


def test_source_import_guard_rejects_preload_checks_origin_and_restores_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_path = list(sys.path)
    preloaded = types.ModuleType("models")
    monkeypatch.setitem(sys.modules, "models", preloaded)
    fixture_asset = _tiny_asset(tmp_path, "source.fixture", b"source")
    with pytest.raises(runner.TransRACOfficialSourceError, match="already loaded"):
        runner.ModernCompatBackend.from_assets(
            tmp_path,
            fixture_asset,
            fixture_asset,
            runner.FROZEN_TRANSRAC_OFFICIAL_CONFIG,
        )
    with (
        pytest.raises(runner.TransRACOfficialSourceError, match="already loaded"),
        runner._isolated_source_import_path(tmp_path),
    ):
        raise AssertionError("guard must reject before entering")
    assert sys.path == original_path
    monkeypatch.delitem(sys.modules, "models")

    source = tmp_path / "source"
    source.mkdir()
    with runner._isolated_source_import_path(source):
        assert sys.path[0] == str(source)
    assert sys.path == original_path

    dataset = source / "dataset"
    models = source / "models"
    dataset.mkdir()
    models.mkdir()
    loader_file = dataset / "UCFRep_loader.py"
    model_file = models / "TransRAC.py"
    loader_file.write_text("", encoding="utf-8")
    model_file.write_text("", encoding="utf-8")
    loader_module = types.ModuleType("dataset.UCFRep_loader")
    loader_module.__file__ = str(loader_file)
    model_module = types.ModuleType("models.TransRAC")
    model_module.__file__ = str(model_file)
    monkeypatch.setitem(sys.modules, "dataset.UCFRep_loader", loader_module)
    monkeypatch.setitem(sys.modules, "models.TransRAC", model_module)
    runner._assert_source_module_origins(source.resolve())

    model_module.__file__ = str(tmp_path / "outside.py")
    (tmp_path / "outside.py").write_text("", encoding="utf-8")
    with pytest.raises(runner.TransRACOfficialSourceError, match="outside"):
        runner._assert_source_module_origins(source.resolve())


def test_sidecar_requires_exact_commitment_before_assets_or_torch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    dev_ids = (
        (Path(__file__).parents[1] / "data/splits/ucfrep_526_dev_84.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    records = tuple(
        UnlabeledVideoRecord(video_id, f"{video_id}.avi", "a" * 64)
        for video_id in dev_ids
    )
    sidecar, commitment = _write_inputs(tmp_path, records)
    payload = json.loads(commitment.read_text(encoding="utf-8"))
    payload["sidecar_sha256"] = "0" * 64
    commitment.write_text(json.dumps(payload), encoding="utf-8")
    asset_calls = 0

    def forbidden_assets(*_args: object) -> None:
        nonlocal asset_calls
        asset_calls += 1
        raise AssertionError("asset gate must follow the input commitment gate")

    monkeypatch.setattr(runner, "verify_official_assets", forbidden_assets)
    with pytest.raises(ValueError, match="does not bind this exact"):
        runner.run_transrac_official(
            sidecar_path=sidecar,
            commitment_path=commitment,
            video_root=video_root,
            source_archive_path=tmp_path / "source.tar.gz",
            source_root=tmp_path / "source",
            backbone_path=tmp_path / runner.BACKBONE_SPEC.name,
            checkpoint_path=tmp_path / runner.CHECKPOINT_SPEC.name,
            repository_root=tmp_path,
        )
    assert asset_calls == 0


def test_label_free_failure_ledger_hashes_inputs_and_writes_receipt(
    tmp_path: Path,
) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    ok = b"ok-video"
    decode = b"decode-video"
    (video_root / "ok.avi").write_bytes(ok)
    (video_root / "decode.avi").write_bytes(decode)
    records = (
        UnlabeledVideoRecord("ok", "ok.avi", _sha256(ok)),
        UnlabeledVideoRecord("decode", "decode.avi", _sha256(decode)),
        UnlabeledVideoRecord("missing", "missing.avi", "f" * 64),
    )
    sidecar, commitment = _write_inputs(tmp_path, records)
    inputs = runner._load_label_free_inputs(
        sidecar,
        commitment,
        video_root,
        require_exact_membership=False,
    )
    source = _tiny_asset(tmp_path, "source.fixture", b"source")
    backbone = _tiny_asset(tmp_path, "backbone.fixture", b"backbone")
    checkpoint = _tiny_asset(tmp_path, "checkpoint.fixture", b"checkpoint")

    result = runner._run_verified_backend(
        inputs,
        source,
        backbone,
        checkpoint,
        _FakeBackend(),
        runner_git_sha=FIXTURE_GIT_SHA,
        runner_code_sha256=FIXTURE_RUNNER_SHA256,
    )
    payload = result.to_dict()

    assert [row["decode_status"] for row in payload["predictions"]] == [
        "ok",
        "decode_failed",
        "missing_video",
    ]
    assert payload["decode_ok_total"] == 1
    assert payload["failure_total"] == 2
    assert payload["predictions"][0]["rounded_count"] == 4
    assert payload["labels_loaded"] is False
    assert payload["scoring_performed"] is False
    assert payload["runner_provenance"] == {
        "source_git_sha": FIXTURE_GIT_SHA,
        "runner_code_sha256": FIXTURE_RUNNER_SHA256,
        "compatibility_image_digest": runner.COMPATIBILITY_IMAGE_DIGEST,
    }
    assert '"target"' not in json.dumps(payload, sort_keys=True)

    output = tmp_path / "results" / "prediction.json"
    _, prediction_sha, receipt_path, receipt_sha = (
        runner.write_transrac_result_exclusive(output, result)
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert prediction_sha == _sha256(output.read_bytes())
    assert receipt_sha == _sha256(receipt_path.read_bytes())
    assert receipt["prediction_sha256"] == prediction_sha
    assert receipt["checkpoint_sha256"] == checkpoint.spec.sha256
    assert receipt["runner_source_git_sha"] == FIXTURE_GIT_SHA
    assert receipt["runner_code_sha256"] == FIXTURE_RUNNER_SHA256
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.write_transrac_result_exclusive(output, result)


def test_video_full_rehash_detects_same_size_tamper_with_restored_mtime(
    tmp_path: Path,
) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    video = video_root / "ok.avi"
    original = b"ok-video"
    video.write_bytes(original)
    sidecar, commitment = _write_inputs(
        tmp_path,
        (UnlabeledVideoRecord("ok", "ok.avi", _sha256(original)),),
    )
    inputs = runner._load_label_free_inputs(
        sidecar,
        commitment,
        video_root,
        require_exact_membership=False,
    )
    source = _tiny_asset(tmp_path, "source.fixture", b"source")
    backbone = _tiny_asset(tmp_path, "backbone.fixture", b"backbone")
    checkpoint = _tiny_asset(tmp_path, "checkpoint.fixture", b"checkpoint")

    class TamperingBackend(_FakeBackend):
        def predict(
            self,
            video_path: Path,
            config: runner.TransRACOfficialConfig,
        ) -> runner.BackendPrediction:
            metadata = video_path.stat()
            prediction = super().predict(video_path, config)
            video_path.write_bytes(b"no-video")
            os.utime(
                video_path,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
            )
            return prediction

    result = runner._run_verified_backend(
        inputs,
        source,
        backbone,
        checkpoint,
        TamperingBackend(),
        runner_git_sha=FIXTURE_GIT_SHA,
        runner_code_sha256=FIXTURE_RUNNER_SHA256,
    )

    assert result.predictions[0].decode_status is runner.DecodeStatus.VIDEO_CHANGED
    assert "SHA-256 changed" in str(result.predictions[0].failure_reason)


def test_paired_prediction_write_preflight_and_failure_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_root = tmp_path / "videos"
    video_root.mkdir()
    video = video_root / "ok.avi"
    content = b"ok-video"
    video.write_bytes(content)
    sidecar, commitment = _write_inputs(
        tmp_path,
        (UnlabeledVideoRecord("ok", "ok.avi", _sha256(content)),),
    )
    inputs = runner._load_label_free_inputs(
        sidecar,
        commitment,
        video_root,
        require_exact_membership=False,
    )
    source = _tiny_asset(tmp_path, "source.fixture", b"source")
    backbone = _tiny_asset(tmp_path, "backbone.fixture", b"backbone")
    checkpoint = _tiny_asset(tmp_path, "checkpoint.fixture", b"checkpoint")
    result = runner._run_verified_backend(
        inputs,
        source,
        backbone,
        checkpoint,
        _FakeBackend(),
        runner_git_sha=FIXTURE_GIT_SHA,
        runner_code_sha256=FIXTURE_RUNNER_SHA256,
    )

    output = tmp_path / "collision" / "prediction.json"
    output.parent.mkdir()
    receipt = output.with_suffix(".receipt.json")
    receipt.write_text("occupied", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.write_transrac_result_exclusive(output, result)
    assert not output.exists()

    receipt.unlink()
    real_writer = runner._write_json_exclusive
    calls = 0

    def fail_second_write(path: Path, payload: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fixture receipt failure")
        assert isinstance(payload, dict)
        return real_writer(path, payload)

    monkeypatch.setattr(runner, "_write_json_exclusive", fail_second_write)
    with pytest.raises(OSError, match="fixture receipt failure"):
        runner.write_transrac_result_exclusive(output, result)
    assert not output.exists()
    assert not receipt.exists()


def test_runner_cli_has_no_target_or_label_argument() -> None:
    option_strings = {
        option
        for action in runner._build_parser()._actions
        for option in action.option_strings
    }
    assert {
        "--sidecar",
        "--commitment",
        "--video-root",
        "--source-archive",
        "--source-root",
        "--backbone",
        "--checkpoint",
        "--repository-root",
        "--video-id",
        "--output",
    } <= option_strings
    assert not {
        "--target",
        "--targets",
        "--dev-targets",
        "--count",
        "--label",
        "--action",
    } & option_strings


def test_runner_import_does_not_require_torch() -> None:
    repository = Path(__file__).parents[1]
    script = f"""
import importlib
import sys
sys.path.insert(0, {str(repository / "src")!r})
class BlockTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError("runner imported torch at module import time")
        return None
sys.meta_path.insert(0, BlockTorch())
importlib.import_module("pams.baselines.transrac_official_runner")
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
