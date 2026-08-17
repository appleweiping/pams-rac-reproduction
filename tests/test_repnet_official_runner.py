from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pams.baselines import repnet_official_runner as runner
from pams.baselines.repnet_official_runner import (
    OFFICIAL_NOTEBOOK_SHA256,
    OFFICIAL_SOURCE_COMMIT,
    BackendPrediction,
    CheckpointFileSpec,
    DecodeStatus,
    RepNetOfficialBlockedError,
    RepNetOfficialCheckpointError,
    RepNetOfficialConfig,
    RepNetOfficialDecodeError,
    RepNetOfficialDependencyError,
    RepNetOfficialSourceError,
)
from pams.data import (
    PoseInputCommitment,
    PoseInputManifest,
    UnlabeledVideoRecord,
    pose_input_identity_sha256,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_label_free_pair(
    tmp_path: Path,
    records: tuple[UnlabeledVideoRecord, ...],
    *,
    split: str = "train",
) -> tuple[Path, Path]:
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split=split,
        records=records,
    )
    sidecar = tmp_path / "inputs.json"
    sidecar.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    sidecar_sha256 = _sha256_bytes(sidecar.read_bytes())
    commitment = PoseInputCommitment(
        protocol=manifest.protocol,
        split=manifest.split,
        record_total=len(manifest.records),
        identity_sha256=pose_input_identity_sha256(manifest.records),
        sidecar_sha256=sidecar_sha256,
        sidecar_fingerprint=manifest.fingerprint,
    )
    commitment_path = tmp_path / "inputs.commitment.json"
    commitment_path.write_text(
        json.dumps(commitment.to_dict(), sort_keys=True),
        encoding="utf-8",
    )
    return sidecar, commitment_path


def _write_tiny_checkpoint(
    tmp_path: Path,
) -> tuple[Path, tuple[CheckpointFileSpec, ...]]:
    directory = tmp_path / "checkpoint"
    directory.mkdir()
    objects = {
        "checkpoint": b'model_checkpoint_path: "ckpt-70"\n',
        "ckpt-70.index": b"index-fixture",
        "ckpt-70.data-00000-of-00001": b"data-fixture",
    }
    specs: list[CheckpointFileSpec] = []
    for name, value in objects.items():
        (directory / name).write_bytes(value)
        specs.append(
            CheckpointFileSpec(
                name=name,
                byte_count=len(value),
                md5=hashlib.md5(value, usedforsecurity=False).hexdigest(),
            )
        )
    return directory, tuple(specs)


def test_frozen_config_contains_the_audited_notebook_parameters() -> None:
    config = runner.FROZEN_REPNET_OFFICIAL_CONFIG

    assert config.source_commit == OFFICIAL_SOURCE_COMMIT
    assert config.checkpoint_prefix == "ckpt-70"
    assert config.window_frames == 64
    assert (config.decode_width, config.decode_height) == (224, 224)
    assert config.model_image_size == 112
    assert config.strides == (1, 2, 3, 4)
    assert config.inference_batch_size == 20
    assert config.global_periodicity_threshold == 0.2
    assert config.frame_periodicity_threshold == 0.5
    assert config.constant_speed is False
    assert config.median_filter is True
    assert config.fully_periodic is False
    assert len(config.fingerprint) == 64

    with pytest.raises(ValueError, match="schedule is frozen"):
        RepNetOfficialConfig(strides=(1, 2, 3))


@pytest.mark.parametrize("forbidden", ["count", "action"])
def test_label_firewall_rejects_privileged_key_before_commitment_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forbidden: str,
) -> None:
    root = tmp_path / "videos"
    root.mkdir()
    sidecar = tmp_path / "forbidden.json"
    sidecar.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "manifest_type": "pose_inputs",
                "protocol": "ucfrep_526",
                "split": "train",
                "records": [
                    {
                        "video_id": "fixture",
                        "video_path": "fixture.avi",
                        "video_sha256": "a" * 64,
                        "nested": {forbidden: 999},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    commitment = tmp_path / "commitment.json"
    commitment.write_text("{}\n", encoding="utf-8")
    commitment_loads = 0

    def forbidden_commitment_loader(_path: object) -> None:
        nonlocal commitment_loads
        commitment_loads += 1
        raise AssertionError("commitment loading must happen after the raw label firewall")

    monkeypatch.setattr(
        runner,
        "load_pose_input_commitment",
        forbidden_commitment_loader,
    )
    with pytest.raises(ValueError, match=f"forbidden privileged field '{forbidden}'"):
        runner._load_label_free_inputs(
            sidecar,
            commitment,
            root,
            require_exact_membership=False,
        )
    assert commitment_loads == 0


def test_public_runner_hits_label_firewall_before_checkpoint_or_tensorflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "videos"
    root.mkdir()
    sidecar = tmp_path / "labeled.json"
    sidecar.write_text('{"count": 123, "action": "forbidden"}\n', encoding="utf-8")
    commitment = tmp_path / "commitment.json"
    commitment.write_text("{}\n", encoding="utf-8")
    checkpoint_calls = 0

    def forbidden_checkpoint(*_args: object, **_kwargs: object) -> None:
        nonlocal checkpoint_calls
        checkpoint_calls += 1
        raise AssertionError("checkpoint gate must not run before label firewall")

    monkeypatch.setattr(runner, "verify_official_checkpoint", forbidden_checkpoint)
    with pytest.raises(ValueError, match="forbidden privileged field 'count'"):
        runner.run_repnet_official(
            sidecar,
            commitment,
            root,
            tmp_path / "missing-checkpoint",
            tmp_path / "missing-source",
        )
    assert checkpoint_calls == 0


def test_label_free_sidecar_requires_its_exact_commitment(tmp_path: Path) -> None:
    root = tmp_path / "videos"
    root.mkdir()
    payload = b"video"
    record = UnlabeledVideoRecord(
        video_id="fixture",
        video_path="fixture.avi",
        video_sha256=_sha256_bytes(payload),
    )
    sidecar, commitment = _write_label_free_pair(tmp_path, (record,))
    commitment_payload = json.loads(commitment.read_text(encoding="utf-8"))
    commitment_payload["sidecar_sha256"] = "0" * 64
    commitment.write_text(json.dumps(commitment_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="does not bind this exact"):
        runner._load_label_free_inputs(
            sidecar,
            commitment,
            root,
            require_exact_membership=False,
        )


def test_checkpoint_gate_verifies_every_object_and_aggregate_hash(
    tmp_path: Path,
) -> None:
    directory, specs = _write_tiny_checkpoint(tmp_path)

    verified = runner._verify_checkpoint_with_specs(directory, specs)

    assert [item.name for item in verified.files] == [item.name for item in specs]
    assert len(verified.aggregate_sha256) == 64
    assert all(len(item.sha256) == 64 for item in verified.files)
    verified.assert_unchanged()

    (directory / "ckpt-70.index").write_bytes(b"X" * len(b"index-fixture"))
    with pytest.raises(RepNetOfficialCheckpointError, match="MD5 mismatch"):
        runner._verify_checkpoint_with_specs(directory, specs)


class _FakeOfficialBackend:
    source_commit = OFFICIAL_SOURCE_COMMIT
    source_notebook_sha256 = OFFICIAL_NOTEBOOK_SHA256
    runtime_versions = {
        "opencv": "fixture",
        "numpy": "fixture",
        "scipy": "fixture",
        "tensorflow": "fixture",
    }

    def __init__(self) -> None:
        self.seen: list[str] = []

    def predict(
        self,
        video_path: Path,
        config: RepNetOfficialConfig,
    ) -> BackendPrediction:
        self.seen.append(video_path.name)
        assert config == runner.FROZEN_REPNET_OFFICIAL_CONFIG
        if video_path.name == "decode.avi":
            raise RepNetOfficialDecodeError("no valid RGB frames")
        return BackendPrediction(
            raw_count=3.5,
            chosen_stride=2,
            confidence=0.75,
            decoded_frames=64,
        )


def test_prediction_ledger_is_label_free_hashed_and_preserves_failures(
    tmp_path: Path,
) -> None:
    root = tmp_path / "videos"
    root.mkdir()
    ok_bytes = b"ok-video"
    decode_bytes = b"decode-video"
    (root / "ok.avi").write_bytes(ok_bytes)
    (root / "decode.avi").write_bytes(decode_bytes)
    records = (
        UnlabeledVideoRecord("ok", "ok.avi", _sha256_bytes(ok_bytes)),
        UnlabeledVideoRecord("decode", "decode.avi", _sha256_bytes(decode_bytes)),
        UnlabeledVideoRecord("missing", "missing.avi", "b" * 64),
    )
    sidecar, commitment = _write_label_free_pair(tmp_path, records)
    inputs = runner._load_label_free_inputs(
        sidecar,
        commitment,
        root,
        require_exact_membership=False,
    )
    checkpoint_dir, specs = _write_tiny_checkpoint(tmp_path)
    checkpoint = runner._verify_checkpoint_with_specs(checkpoint_dir, specs)
    backend = _FakeOfficialBackend()

    result = runner._run_verified_backend(inputs, checkpoint, backend)
    payload = result.to_dict()

    assert backend.seen == ["ok.avi", "decode.avi"]
    assert [row["decode_status"] for row in payload["predictions"]] == [
        "ok",
        "decode_failed",
        "missing_video",
    ]
    success = payload["predictions"][0]
    assert success["raw_count"] == 3.5
    assert success["rounded_count"] == 4
    assert success["chosen_stride"] == 2
    assert success["confidence"] == 0.75
    assert success["observed_video_sha256"] == _sha256_bytes(ok_bytes)
    assert payload["input"]["sidecar_sha256"] == _sha256_bytes(sidecar.read_bytes())
    assert payload["checkpoint"]["aggregate_sha256"] == checkpoint.aggregate_sha256
    assert payload["config"]["sha256"] == runner.FROZEN_REPNET_OFFICIAL_CONFIG.fingerprint
    assert payload["input"]["sample_count"] == 3
    assert payload["selection"]["selected_count"] == 3
    assert payload["selection"]["selected_video_ids"] == ["ok", "decode", "missing"]
    encoded = json.dumps(payload, sort_keys=True)
    assert '"target"' not in encoded
    assert '"action"' not in encoded


def test_subset_is_selected_after_full_binding_and_follows_sidecar_order(
    tmp_path: Path,
) -> None:
    root = tmp_path / "videos"
    root.mkdir()
    video_bytes = {
        "first": b"first",
        "second": b"second",
        "third": b"third",
    }
    for video_id, value in video_bytes.items():
        (root / f"{video_id}.avi").write_bytes(value)
    records = tuple(
        UnlabeledVideoRecord(
            video_id,
            f"{video_id}.avi",
            _sha256_bytes(value),
        )
        for video_id, value in video_bytes.items()
    )
    sidecar, commitment = _write_label_free_pair(tmp_path, records)
    inputs = runner._load_label_free_inputs(
        sidecar,
        commitment,
        root,
        require_exact_membership=False,
    )
    checkpoint_dir, specs = _write_tiny_checkpoint(tmp_path)
    checkpoint = runner._verify_checkpoint_with_specs(checkpoint_dir, specs)
    backend = _FakeOfficialBackend()

    result = runner._run_verified_backend(
        inputs,
        checkpoint,
        backend,
        video_ids=("third", "first"),
    )
    payload = result.to_dict()

    assert result.selected_video_ids == ("first", "third")
    assert [item.video_id for item in result.predictions] == ["first", "third"]
    assert backend.seen == ["first.avi", "third.avi"]
    assert payload["input"]["sample_count"] == 3
    assert payload["input"]["sidecar_sha256"] == _sha256_bytes(sidecar.read_bytes())
    assert payload["selection"]["selected_count"] == 2
    assert payload["selection"]["selected_video_ids"] == ["first", "third"]
    assert len(payload["selection"]["selected_video_ids_sha256"]) == 64

    with pytest.raises(ValueError, match="not present in the verified sidecar"):
        runner._run_verified_backend(
            inputs,
            checkpoint,
            _FakeOfficialBackend(),
            video_ids=("unknown",),
        )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        runner._run_verified_backend(
            inputs,
            checkpoint,
            _FakeOfficialBackend(),
            video_ids=("first", "first"),
        )


def test_video_hash_mismatch_never_reaches_backend(tmp_path: Path) -> None:
    root = tmp_path / "videos"
    root.mkdir()
    (root / "changed.avi").write_bytes(b"changed")
    records = (
        UnlabeledVideoRecord("changed", "changed.avi", _sha256_bytes(b"expected")),
    )
    sidecar, commitment = _write_label_free_pair(tmp_path, records)
    inputs = runner._load_label_free_inputs(
        sidecar,
        commitment,
        root,
        require_exact_membership=False,
    )
    checkpoint_dir, specs = _write_tiny_checkpoint(tmp_path)
    checkpoint = runner._verify_checkpoint_with_specs(checkpoint_dir, specs)
    backend = _FakeOfficialBackend()

    result = runner._run_verified_backend(inputs, checkpoint, backend)

    assert backend.seen == []
    assert result.predictions[0].decode_status is DecodeStatus.VIDEO_HASH_MISMATCH
    assert result.predictions[0].raw_count is None


def test_result_writer_is_exclusive_and_returns_written_sha256(tmp_path: Path) -> None:
    root = tmp_path / "videos"
    root.mkdir()
    video = root / "ok.avi"
    video.write_bytes(b"video")
    sidecar, commitment = _write_label_free_pair(
        tmp_path,
        (
            UnlabeledVideoRecord(
                "ok",
                "ok.avi",
                _sha256_bytes(video.read_bytes()),
            ),
        ),
    )
    inputs = runner._load_label_free_inputs(
        sidecar,
        commitment,
        root,
        require_exact_membership=False,
    )
    checkpoint_dir, specs = _write_tiny_checkpoint(tmp_path)
    checkpoint = runner._verify_checkpoint_with_specs(checkpoint_dir, specs)
    result = runner._run_verified_backend(
        inputs,
        checkpoint,
        _FakeOfficialBackend(),
    )
    output = tmp_path / "results" / "repnet.json"

    digest = runner.write_repnet_result_exclusive(output, result)

    assert digest == _sha256_bytes(output.read_bytes())
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.write_repnet_result_exclusive(output, result)


def _minimal_official_notebook(*, include_mapping: bool) -> bytes:
    definitions = [
        "class ResnetPeriodEstimator:\n  pass\n",
        "def get_sims():\n  pass\n",
        "def flatten_sequential_feats():\n  pass\n",
        "def scaled_dot_product_attention():\n  pass\n",
        "def point_wise_feed_forward_network():\n  pass\n",
        "class MultiHeadAttention:\n  pass\n",
        "class TransformerLayer:\n  pass\n",
        "def pairwise_l2_distance():\n  pass\n",
        "def get_repnet_model():\n  pass\n",
        "def read_video():\n  pass\n",
        "def get_score():\n  pass\n",
        "def get_counts():\n  pass\n",
        "def load_ckpt_with_custom_layer_mapping():\n  pass\n",
    ]
    if include_mapping:
        definitions.append("MAPPING_NEW_TO_OLD_LAYER_NAMES = [('new', 'old')]\n")
    return json.dumps(
        {
            "cells": [
                {
                    "cell_type": "code",
                    "source": definitions,
                }
            ]
        }
    ).encode("utf-8")


def test_notebook_extractor_requires_and_preserves_complete_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_import_official_dependencies",
        lambda: ({}, {"tensorflow": "fixture"}),
    )
    notebook = _minimal_official_notebook(include_mapping=True)

    module, digest = runner._official_definition_module(notebook)

    assert module.MAPPING_NEW_TO_OLD_LAYER_NAMES == [("new", "old")]
    assert callable(module.load_ckpt_with_custom_layer_mapping)
    assert digest == _sha256_bytes(notebook)

    with pytest.raises(RepNetOfficialSourceError, match="MAPPING_NEW_TO_OLD"):
        runner._official_definition_module(
            _minimal_official_notebook(include_mapping=False)
        )


def test_standalone_notebook_requires_exact_frozen_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    notebook = tmp_path / "official.ipynb"
    expected = b'{"cells": []}\n'
    notebook.write_bytes(expected)
    monkeypatch.setattr(runner, "OFFICIAL_NOTEBOOK_BYTES", len(expected))
    monkeypatch.setattr(
        runner,
        "OFFICIAL_NOTEBOOK_SHA256",
        _sha256_bytes(expected),
    )

    assert runner._read_frozen_official_notebook_file(notebook) == expected

    notebook.write_bytes(b"X" * len(expected))
    with pytest.raises(RepNetOfficialSourceError, match="byte count/SHA-256"):
        runner._read_frozen_official_notebook_file(notebook)


def test_dependency_gate_reports_missing_tensorflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_import(name: str) -> object:
        if name == "tensorflow":
            raise ImportError("fixture has no TensorFlow")
        return SimpleNamespace()

    monkeypatch.setattr(runner.importlib, "import_module", fake_import)

    with pytest.raises(RepNetOfficialDependencyError, match="'tensorflow'.*unavailable"):
        runner._import_official_dependencies()


def test_model_restore_blocks_incomplete_official_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint_dir, specs = _write_tiny_checkpoint(tmp_path)
    checkpoint = runner._verify_checkpoint_with_specs(checkpoint_dir, specs)
    api = SimpleNamespace(
        get_repnet_model=lambda _path: SimpleNamespace(weights=[object()]),
        MAPPING_NEW_TO_OLD_LAYER_NAMES=[],
        load_ckpt_with_custom_layer_mapping=lambda *_args: pytest.fail(
            "incomplete mapping must be rejected before assignment"
        ),
        __official_runtime_versions__={"tensorflow": "fixture"},
    )
    monkeypatch.setattr(
        runner,
        "_read_frozen_official_notebook",
        lambda _root: b"official-notebook",
    )
    monkeypatch.setattr(
        runner,
        "_official_definition_module",
        lambda _notebook: (api, "a" * 64),
    )

    with pytest.raises(
        RepNetOfficialBlockedError,
        match=r"mapping length.*\(0 != 1\)",
    ):
        runner.OfficialNotebookBackend.from_checkout(tmp_path, checkpoint)


def test_argparse_cli_forwards_standalone_notebook_and_subset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "result.json"
    captured: dict[str, object] = {}

    def fake_run(
        sidecar_path: Path,
        commitment_path: Path,
        video_root: Path,
        checkpoint_dir: Path,
        official_source_root: Path | None,
        *,
        official_notebook_path: Path | None,
        video_ids: list[str] | None,
        output_path: Path | None,
    ) -> SimpleNamespace:
        captured.update(
            {
                "sidecar": sidecar_path,
                "commitment": commitment_path,
                "video_root": video_root,
                "checkpoint_dir": checkpoint_dir,
                "source_root": official_source_root,
                "notebook": official_notebook_path,
                "video_ids": video_ids,
                "output": output_path,
            }
        )
        assert output_path is not None
        output_path.write_text('{"predictions": []}\n', encoding="utf-8")
        return SimpleNamespace(selected_video_ids=("first", "third"))

    monkeypatch.setattr(runner, "run_repnet_official", fake_run)
    exit_code = runner.main(
        [
            "--sidecar",
            str(tmp_path / "inputs.json"),
            "--commitment",
            str(tmp_path / "inputs.commitment.json"),
            "--video-root",
            str(tmp_path / "videos"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoint"),
            "--official-notebook",
            str(tmp_path / "repnet.ipynb"),
            "--video-id",
            "third",
            "--video-id",
            "first",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert captured["source_root"] is None
    assert captured["notebook"] == tmp_path / "repnet.ipynb"
    assert captured["video_ids"] == ["third", "first"]
    summary = json.loads(capsys.readouterr().out)
    assert summary["selected_count"] == 2
    assert summary["selected_video_ids"] == ["first", "third"]
    assert summary["output_sha256"] == _sha256_bytes(output.read_bytes())


def test_runner_import_does_not_require_torch() -> None:
    repository = Path(__file__).parents[1]
    source = repository / "src"
    script = f"""
import importlib
import sys

sys.path.insert(0, {str(source)!r})

class BlockTorch:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise RuntimeError("RepNet runner attempted to import torch")
        return None

sys.meta_path.insert(0, BlockTorch())
importlib.import_module("pams.baselines.repnet_official_runner")
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
