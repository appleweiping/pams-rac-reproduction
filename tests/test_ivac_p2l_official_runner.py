from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from pams.baselines import ivac_p2l_official_runner as runner
from pams.data import (
    PoseInputCommitment,
    PoseInputManifest,
    UnlabeledVideoRecord,
    pose_input_identity_sha256,
)

FIXTURE_GIT_SHA = "a" * 40
FIXTURE_RUNNER_SHA256 = "b" * 64
REPOSITORY = Path(__file__).parents[1]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_inputs(
    root: Path,
    records: tuple[UnlabeledVideoRecord, ...],
) -> tuple[Path, Path]:
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split="dev",
        records=records,
    )
    sidecar = root / "inputs.json"
    sidecar.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
    commitment = PoseInputCommitment(
        protocol="ucfrep_526",
        split="dev",
        record_total=len(records),
        identity_sha256=pose_input_identity_sha256(records),
        sidecar_sha256=_sha256(sidecar.read_bytes()),
        sidecar_fingerprint=manifest.fingerprint,
    )
    commitment_path = root / "inputs.commitment.json"
    commitment_path.write_text(
        json.dumps(commitment.to_dict()),
        encoding="utf-8",
    )
    return sidecar, commitment_path


def _asset(root: Path, name: str, content: bytes) -> runner.VerifiedAsset:
    path = root / name
    path.write_bytes(content)
    spec = runner.AssetSpec(
        name=name,
        byte_count=len(content),
        sha256=_sha256(content),
        role=f"fixture_{name}",
        url="https://example.invalid/fixture",
    )
    return runner._verify_asset_with_spec(path, spec)


class _Backend:
    source_commit = runner.OFFICIAL_SOURCE_COMMIT
    source_tree_sha256 = runner.OFFICIAL_SOURCE_TREE_SHA256
    runtime_versions = {"python": "fixture"}
    restore_audit = {"checkpoint_epoch": 67}

    def __init__(self, mutate: bool = False) -> None:
        self.mutate = mutate

    def predict(
        self,
        video_path: Path,
        config: runner.IVACP2LOfficialConfig,
    ) -> runner.BackendPrediction:
        assert config is runner.FROZEN_IVAC_P2L_OFFICIAL_CONFIG
        if self.mutate:
            metadata = video_path.stat()
            video_path.write_bytes(b"tampered")
            os.utime(
                video_path,
                ns=(metadata.st_atime_ns, metadata.st_mtime_ns),
            )
        return runner.BackendPrediction(
            raw_count=3.5,
            decoded_frame_count=90,
            sampled_frame_count=64,
            density_min=0.0,
            density_max=0.5,
            density_mean=3.5 / 64,
            peak_allocated_mib=700.0,
            peak_reserved_mib=800.0,
        )


def test_frozen_identity_config_and_full_hash_guards(tmp_path: Path) -> None:
    config = runner.FROZEN_IVAC_P2L_OFFICIAL_CONFIG
    assert runner.OFFICIAL_SOURCE_COMMIT == (
        "0b1149e6958268ca5131d60ff66498c6e07f3b79"
    )
    assert config.scales == (1, 4, 8)
    assert config.rounding == "nearest_integer_half_up"
    assert config.compatibility_image_digest == runner.COMPATIBILITY_IMAGE_DIGEST
    with pytest.raises(ValueError, match="schedule is frozen"):
        runner.IVACP2LOfficialConfig(scales=(1, 2, 4))

    asset = _asset(tmp_path, "asset.bin", b"official")
    metadata = asset.path.stat()
    asset.path.write_bytes(b"tampered")
    os.utime(asset.path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns))
    with pytest.raises(runner.IVACP2LAssetError, match="changed"):
        asset.assert_unchanged()

    source = tmp_path / "source"
    source.mkdir()
    (source / "module.py").write_text("value = 1\n", encoding="utf-8")
    observed = runner._source_tree_digest(source)
    spec = runner.SourceTreeSpec(*observed)
    assert runner._verify_source_tree_with_spec(source, spec) == source.resolve()
    (source / "module.py").write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(runner.IVACP2LSourceError, match="mismatch"):
        runner._verify_source_tree_with_spec(source, spec)


def test_source_module_preload_and_origin_are_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preloaded = types.ModuleType("models")
    monkeypatch.setitem(sys.modules, "models", preloaded)
    fixture = _asset(tmp_path, "fixture.bin", b"fixture")
    with pytest.raises(runner.IVACP2LSourceError, match="before the frozen tree"):
        runner.ModernCompatBackend.from_assets(
            tmp_path,
            fixture,
            fixture,
            runner.FROZEN_IVAC_P2L_OFFICIAL_CONFIG,
        )
    monkeypatch.delitem(sys.modules, "models")

    source = tmp_path / "official"
    (source / "models").mkdir(parents=True)
    (source / "mmaction").mkdir()
    ivac_file = source / "models/IVAC.py"
    mmaction_file = source / "mmaction/__init__.py"
    ivac_file.write_text("", encoding="utf-8")
    mmaction_file.write_text("", encoding="utf-8")
    ivac = types.ModuleType("models.IVAC")
    ivac.__file__ = str(ivac_file)
    mmaction = types.ModuleType("mmaction")
    mmaction.__file__ = str(mmaction_file)
    monkeypatch.setitem(sys.modules, "models.IVAC", ivac)
    monkeypatch.setitem(sys.modules, "mmaction", mmaction)
    runner._assert_source_module_origins(source.resolve())
    ivac.__file__ = str(tmp_path / "outside.py")
    (tmp_path / "outside.py").write_text("", encoding="utf-8")
    with pytest.raises(runner.IVACP2LSourceError, match="outside"):
        runner._assert_source_module_origins(source.resolve())


def test_commitment_gate_precedes_assets_and_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dev_ids = (
        (Path(__file__).parents[1] / "data/splits/ucfrep_526_dev_84.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    sidecar, commitment = _write_inputs(
        tmp_path,
        tuple(
            UnlabeledVideoRecord(video_id, f"{video_id}.avi", "a" * 64)
            for video_id in dev_ids
        ),
    )
    payload = json.loads(commitment.read_text(encoding="utf-8"))
    payload["sidecar_sha256"] = "0" * 64
    commitment.write_text(json.dumps(payload), encoding="utf-8")
    calls = 0

    def forbidden(*_args: object) -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(runner, "verify_official_assets", forbidden)
    with pytest.raises(ValueError, match="does not bind"):
        runner.run_ivac_p2l_official(
            sidecar_path=sidecar,
            commitment_path=commitment,
            video_root=tmp_path,
            source_archive_path=tmp_path / "source.tar.gz",
            source_root=tmp_path,
            backbone_path=tmp_path / "backbone.pth",
            checkpoint_path=tmp_path / "checkpoint.pt",
            repository_root=tmp_path,
        )
    assert calls == 0


def test_dirty_checkout_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replies = iter(
        (
            types.SimpleNamespace(stdout=FIXTURE_GIT_SHA + "\n"),
            types.SimpleNamespace(stdout="?? untracked-result.json\n"),
        )
    )
    monkeypatch.setattr(runner.subprocess, "run", lambda *_args, **_kwargs: next(replies))
    with pytest.raises(runner.IVACP2LSourceError, match="clean full-SHA"):
        runner._clean_runner_git_revision(tmp_path)


def test_runner_byte_change_after_inference_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dev_ids = (
        (REPOSITORY / "data/splits/ucfrep_526_dev_84.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    sidecar, commitment = _write_inputs(
        tmp_path,
        tuple(
            UnlabeledVideoRecord(video_id, f"{video_id}.avi", "a" * 64)
            for video_id in dev_ids
        ),
    )
    runner_code = REPOSITORY / runner.RUNNER_CODE_RELATIVE_PATH
    initial_digest = runner._stable_file_digest(runner_code)
    digest_calls = 0

    def changed_runner_digest(path: Path) -> object:
        nonlocal digest_calls
        assert path == runner_code.resolve()
        digest_calls += 1
        return initial_digest if digest_calls == 1 else object()

    monkeypatch.setattr(
        runner,
        "_clean_runner_git_revision",
        lambda _root: FIXTURE_GIT_SHA,
    )
    monkeypatch.setattr(runner, "_stable_file_digest", changed_runner_digest)
    monkeypatch.setattr(
        runner,
        "verify_official_assets",
        lambda *_args: (object(), object(), object()),
    )
    monkeypatch.setattr(
        runner,
        "verify_official_source_tree",
        lambda _root: tmp_path,
    )
    monkeypatch.setattr(
        runner.ModernCompatBackend,
        "from_assets",
        lambda *_args: object(),
    )
    sentinel = object()
    monkeypatch.setattr(
        runner,
        "_run_verified_backend",
        lambda *_args, **_kwargs: sentinel,
    )
    monkeypatch.setattr(
        runner,
        "_source_tree_digest",
        lambda _root: (
            runner.OFFICIAL_SOURCE_TREE_FILES,
            runner.OFFICIAL_SOURCE_TREE_BYTES,
            runner.OFFICIAL_SOURCE_TREE_SHA256,
        ),
    )
    with pytest.raises(runner.IVACP2LSourceError, match="runner code changed"):
        runner.run_ivac_p2l_official(
            sidecar_path=sidecar,
            commitment_path=commitment,
            video_root=tmp_path,
            source_archive_path=tmp_path / "source.tar.gz",
            source_root=tmp_path,
            backbone_path=tmp_path / "backbone.pth",
            checkpoint_path=tmp_path / "checkpoint.pt",
            repository_root=REPOSITORY,
        )


def test_prediction_binds_video_and_pair_write_rolls_back(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "video.avi"
    video.write_bytes(b"official")
    sidecar, commitment = _write_inputs(
        tmp_path,
        (UnlabeledVideoRecord("v", video.name, _sha256(video.read_bytes())),),
    )
    inputs = runner._load_label_free_inputs(
        sidecar,
        commitment,
        tmp_path,
        require_exact_membership=False,
    )
    source = _asset(tmp_path, "source.bin", b"source")
    backbone = _asset(tmp_path, "backbone.bin", b"backbone")
    checkpoint = _asset(tmp_path, "checkpoint.bin", b"checkpoint")
    result = runner._run_verified_backend(
        inputs,
        source,
        backbone,
        checkpoint,
        _Backend(),
        runner_git_sha=FIXTURE_GIT_SHA,
        runner_code_sha256=FIXTURE_RUNNER_SHA256,
    )
    row = result.to_dict()["predictions"][0]
    assert row["video_locator"] == video.name
    assert row["expected_video_sha256"] == row["observed_video_sha256"]
    assert row["rounded_count"] == 4
    assert result.to_dict()["labels_loaded"] is False
    assert result.to_dict()["runner_provenance"] == {
        "source_git_sha": FIXTURE_GIT_SHA,
        "runner_code_sha256": FIXTURE_RUNNER_SHA256,
        "compatibility_image_digest": runner.COMPATIBILITY_IMAGE_DIGEST,
    }
    bound_output = tmp_path / "bound/prediction.json"
    _, _, bound_receipt, _ = runner.write_ivac_p2l_result_exclusive(
        bound_output,
        result,
    )
    receipt_payload = json.loads(bound_receipt.read_text(encoding="utf-8"))
    assert receipt_payload["runner_source_git_sha"] == FIXTURE_GIT_SHA
    assert receipt_payload["runner_code_sha256"] == FIXTURE_RUNNER_SHA256

    real_writer = runner._write_json_exclusive
    calls = 0

    def fail_receipt(path: Path, payload: object) -> str:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("fixture receipt failure")
        assert isinstance(payload, dict)
        return real_writer(path, payload)

    monkeypatch.setattr(runner, "_write_json_exclusive", fail_receipt)
    output = tmp_path / "result/prediction.json"
    with pytest.raises(OSError, match="fixture receipt failure"):
        runner.write_ivac_p2l_result_exclusive(output, result)
    assert not output.exists()
    assert not output.with_suffix(".receipt.json").exists()


def test_video_full_rehash_and_label_free_cli_boundary(tmp_path: Path) -> None:
    video = tmp_path / "video.avi"
    video.write_bytes(b"official")
    sidecar, commitment = _write_inputs(
        tmp_path,
        (UnlabeledVideoRecord("v", video.name, _sha256(video.read_bytes())),),
    )
    inputs = runner._load_label_free_inputs(
        sidecar,
        commitment,
        tmp_path,
        require_exact_membership=False,
    )
    assets = (
        _asset(tmp_path, "source.bin", b"source"),
        _asset(tmp_path, "backbone.bin", b"backbone"),
        _asset(tmp_path, "checkpoint.bin", b"checkpoint"),
    )
    result = runner._run_verified_backend(
        inputs,
        *assets,
        _Backend(mutate=True),
        runner_git_sha=FIXTURE_GIT_SHA,
        runner_code_sha256=FIXTURE_RUNNER_SHA256,
    )
    assert result.predictions[0].decode_status is runner.DecodeStatus.VIDEO_CHANGED

    options = {
        option
        for action in runner._build_parser()._actions
        for option in action.option_strings
    }
    assert "--sidecar" in options and "--commitment" in options
    assert "--repository-root" in options
    assert not {"--target", "--targets", "--label", "--annotation"} & options
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import pams.baselines.ivac_p2l_official_runner; "
            "assert 'torch' not in sys.modules",
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
