from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from typer.testing import CliRunner

from pams import cli as cli_module
from pams import ucfrep as ucfrep_module
from pams.cli import app
from pams.data import (
    PoseInputCommitment,
    PoseInputManifest,
    UnlabeledVideoRecord,
    pose_input_identity_sha256,
    write_pose_cache,
)
from pams.training import CheckpointProvenance
from pams.types import PoseSequence

runner = CliRunner()
REPOSITORY = Path(__file__).parents[1]
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _json_output(value: str) -> dict[str, object]:
    return json.loads(value)


def _plain_text(value: str) -> str:
    return ANSI_ESCAPE.sub("", value)


def _pose_cache(tmp_path: Path) -> Path:
    frames = 64
    phase = np.arange(frames, dtype=np.float32) * 2.0 * np.pi / 8.0
    xyz = np.zeros((frames, 33, 3), dtype=np.float32)
    xyz[:, :, 0] = np.sin(phase)[:, None]
    xyz[:, :, 1] = np.cos(phase)[:, None]
    sequence = PoseSequence(
        "periodic",
        30.0,
        xyz,
        np.ones(frames, dtype=bool),
    )
    path = tmp_path / "pose.npz"
    write_pose_cache(
        path,
        sequence,
        video_sha256="a" * 64,
        pose_fingerprint="b" * 64,
    )
    return path


class _FakeManifest:
    protocol = "ucfrep_526"
    fingerprint = "d" * 64
    sealed_dataset_fingerprint = "9" * 64

    def training_fingerprint(self, *, include_dev: bool = False) -> str:
        del include_dev
        return "e" * 64

    def records_for(self, split: str) -> tuple[SimpleNamespace, ...]:
        if split == "train":
            return (
                SimpleNamespace(video_id="train-b"),
                SimpleNamespace(video_id="train-a"),
            )
        if split == "dev":
            return (SimpleNamespace(video_id="dev-a"),)
        return (SimpleNamespace(video_id="test-a", count=4, action="action"),)

    def training_records(
        self,
        *,
        include_dev: bool = False,
    ) -> tuple[SimpleNamespace, ...]:
        records = self.records_for("train")
        if include_dev:
            records = (*records, *self.records_for("dev"))
        return records


class _FakePoseSnapshot:
    fingerprint = "c" * 64

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "pose_fingerprint": "b" * 64,
            "fingerprint": self.fingerprint,
            "entry_count": 1,
            "entries": [
                {
                    "video_id": "fixture",
                    "cache_sha256": "a" * 64,
                    "bytes": 1,
                }
            ],
        }


def _set_fake_container_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PAMS_CONTAINER_IMAGE_ID", f"sha256:{'1' * 64}")
    monkeypatch.setenv("PAMS_CONTAINER_ENVIRONMENT_SHA256", "2" * 64)
    monkeypatch.setenv("PAMS_CONTAINER_SOURCE_REVISION", "a" * 40)


def test_root_help_has_only_frozen_command_groups() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    expected = {
        "config",
        "data",
        "pose",
        "train",
        "evaluate",
        "baseline",
        "report",
        "synthetic",
    }
    for command in expected:
        assert command in result.stdout


def test_config_validate_outputs_fingerprint() -> None:
    result = runner.invoke(
        app,
        ["config", "validate", str(REPOSITORY / "configs" / "pams.yaml")],
    )
    assert result.exit_code == 0, result.output
    payload = _json_output(result.stdout)
    assert payload["valid"] is True
    assert payload["protocol"] == "ucfrep_526"
    assert len(str(payload["fingerprint"])) == 64
    assert len(str(payload["pose_fingerprint"])) == 64


@pytest.mark.parametrize("command", ["encoder", "sshead", "dev-evaluate"])
def test_formal_training_and_dev_evaluation_reject_legacy_manifest_before_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    manifest = tmp_path / "labeled.json"
    manifest.write_text('{"count": 999, "action": "sealed"}', encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    output_dir = tmp_path / f"output-{command}"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"placeholder")
    progress = tmp_path / "progress.jsonl"
    progress.write_text("{}\n", encoding="utf-8")
    loader_calls = 0

    def forbidden_loader(*_args: object, **_kwargs: object) -> None:
        nonlocal loader_calls
        loader_calls += 1
        raise AssertionError("legacy labeled loader must not be called")

    monkeypatch.setenv("PAMS_AUDIT_MODE", "sealed")
    monkeypatch.setattr(cli_module, "load_ucfrep_manifest", forbidden_loader)
    if command == "encoder":
        arguments = [
            "train",
            "encoder",
            str(manifest),
            str(cache_dir),
            str(output_dir),
        ]
    elif command == "sshead":
        arguments = [
            "train",
            "sshead",
            str(checkpoint),
            str(manifest),
            str(cache_dir),
            str(output_dir),
            "--encoder-progress",
            str(progress),
        ]
    else:
        arguments = [
            "evaluate",
            "checkpoint",
            str(checkpoint),
            str(manifest),
            str(cache_dir),
            str(output_dir),
            "--variant",
            "literal",
            "--split",
            "dev",
        ]

    result = runner.invoke(app, arguments)

    assert result.exit_code == 2, result.output
    assert "requires --label-free-inputs" in _plain_text(result.output)
    assert loader_calls == 0


def test_formal_label_free_protocol_loader_never_calls_labeled_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split_files = {
        "train": "ucfrep_526_train_337.txt",
        "dev": "ucfrep_526_dev_84.txt",
        "test": "ucfrep_526_test_105.txt",
    }
    sidecars: dict[str, Path] = {}
    commitments: dict[str, Path] = {}
    for split, filename in split_files.items():
        identifiers = (
            REPOSITORY / "data" / "splits" / filename
        ).read_text(encoding="utf-8").splitlines()
        pose_inputs = PoseInputManifest(
            protocol="ucfrep_526",
            split=split,
            records=tuple(
                UnlabeledVideoRecord(
                    video_id=video_id,
                    video_path=f"videos/{video_id}.avi",
                    video_sha256=hashlib.sha256(video_id.encode("utf-8")).hexdigest(),
                )
                for video_id in identifiers
            ),
        )
        sidecar_path = tmp_path / f"{split}.inputs.json"
        sidecar_path.write_text(
            json.dumps(pose_inputs.to_dict(), sort_keys=True),
            encoding="utf-8",
        )
        sidecar_sha256 = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
        commitment = PoseInputCommitment(
            protocol="ucfrep_526",
            split=split,
            record_total=len(pose_inputs.records),
            identity_sha256=pose_input_identity_sha256(pose_inputs.records),
            sidecar_sha256=sidecar_sha256,
            sidecar_fingerprint=pose_inputs.fingerprint,
        )
        commitment_path = tmp_path / f"{split}.commitment.json"
        commitment_path.write_text(
            json.dumps(commitment.to_dict(), sort_keys=True),
            encoding="utf-8",
        )
        sidecars[split] = sidecar_path
        commitments[split] = commitment_path

    monkeypatch.setattr(
        cli_module,
        "load_ucfrep_manifest",
        lambda *_args, **_kwargs: pytest.fail("labeled loader crossed the firewall"),
    )
    bundle = cli_module._load_label_free_protocol_inputs(
        sidecars["train"],
        train_commitment_path=commitments["train"],
        dev_inputs_path=sidecars["dev"],
        dev_commitment_path=commitments["dev"],
        test_identity_inputs_path=sidecars["test"],
        test_identity_commitment_path=commitments["test"],
    )

    assert bundle.inputs.split_counts == {"train": 337, "dev": 84, "test": 105}
    assert len(bundle.inputs.training_records(include_dev=True)) == 421
    assert len(bundle.artifacts) == len(bundle.artifact_sha256) == 6


def test_json_output_creation_is_concurrency_exclusive(tmp_path: Path) -> None:
    destination = tmp_path / "race.json"

    def write(candidate: int) -> str:
        try:
            cli_module._write_json_exclusive(destination, {"candidate": candidate})
        except FileExistsError:
            return "lost"
        return "won"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(write, (1, 2)))

    assert sorted(outcomes) == ["lost", "won"]
    assert json.loads(destination.read_text(encoding="utf-8"))["candidate"] in {1, 2}


def test_prediction_reader_supports_count_rows_and_rejects_nonfinite(
    tmp_path: Path,
) -> None:
    count_rows = tmp_path / "count-rows.json"
    count_rows.write_text(
        '[{"video_id": "a", "count": 4}, {"video_id": "b", "count": 7.5}]\n',
        encoding="utf-8",
    )
    assert cli_module._read_predictions(count_rows) == {"a": 4.0, "b": 7.5}

    ambiguous = tmp_path / "ambiguous.json"
    ambiguous.write_text(
        '[{"video_id": "a", "count": 4, "prediction": 4}]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one"):
        cli_module._read_predictions(ambiguous)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"a": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="finite non-negative"):
        cli_module._read_predictions(nonfinite)


def test_spectral_proxy_is_explicitly_diagnostic(tmp_path: Path) -> None:
    cache = _pose_cache(tmp_path)
    result = runner.invoke(app, ["baseline", "predict", "spectral-proxy", str(cache)])
    assert result.exit_code == 0, result.output
    payload = _json_output(result.stdout)
    assert payload["method"] == "spectral-proxy"
    assert payload["diagnostic_only"] is True
    assert payload["eligible_for_paper_table"] is False
    assert payload["pose_cache_pose_fingerprint"] == "b" * 64


def test_pose_extract_help_exposes_safe_resume_flag() -> None:
    result = runner.invoke(app, ["pose", "extract", "--help"])
    assert result.exit_code == 0, result.output
    assert "--skip-existing" in _plain_text(result.stdout)
    assert "--label-free-manifest" in _plain_text(result.stdout)
    assert "--keep-full-timeline" not in result.stdout


def test_pose_input_compiler_emits_no_count_or_action_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "video.json"
    video.write_bytes(b"video")
    source_manifest = tmp_path / "manifest.json"
    source_manifest.write_text('{"fixture": true}\n', encoding="utf-8")
    record = SimpleNamespace(
        video_id="v_BenchPress_g21_c01",
        video_path=str(video),
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
    )
    fake_manifest = SimpleNamespace(
        protocol="ucfrep_526",
        fingerprint="b" * 64,
        sealed_dataset_fingerprint="d" * 64,
        records_for=lambda split: (record,) if split == "test" else (),
    )
    monkeypatch.setattr(
        cli_module,
        "load_ucfrep_manifest",
        lambda *_args, **_kwargs: fake_manifest,
    )
    monkeypatch.setattr(cli_module, "_validate_experiment_split", lambda _manifest: None)
    monkeypatch.setattr(
        cli_module.PoseInputManifest,
        "validate_exact_membership",
        lambda _manifest: None,
    )
    output = tmp_path / "pose-inputs.json"

    result = runner.invoke(
        app,
        [
            "data",
            "pose-inputs",
            str(source_manifest),
            "--output",
            str(output),
            "--split",
            "test",
        ],
    )

    assert result.exit_code == 0, result.output
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["manifest_type"] == "pose_inputs"
    assert set(persisted["records"][0]) == {
        "video_id",
        "video_path",
        "video_sha256",
    }
    sidecar_text = output.read_text(encoding="utf-8")
    for forbidden in (
        "source_manifest_file_sha256",
        "source_manifest_fingerprint",
        "sealed_dataset_fingerprint",
        "count",
        "action",
    ):
        assert forbidden not in sidecar_text
    commitment = json.loads(
        (tmp_path / "pose-inputs.commitment.json").read_text(encoding="utf-8")
    )
    assert commitment["sidecar_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert commitment["record_total"] == 1
    commitment_text = json.dumps(commitment, sort_keys=True)
    for forbidden in (
        "source_manifest_file_sha256",
        "source_manifest_fingerprint",
        "sealed_dataset_fingerprint",
        "count",
        "action",
    ):
        assert forbidden not in commitment_text
    original_video = video.read_bytes()
    aliases_video = runner.invoke(
        app,
        [
            "data",
            "pose-inputs",
            str(source_manifest),
            "--output",
            str(video),
            "--split",
            "test",
            "--overwrite",
        ],
    )
    assert aliases_video.exit_code == 2
    assert "aliases selected video" in aliases_video.output
    assert video.read_bytes() == original_video


def test_pose_input_compiler_rejects_non_json_and_input_alias_without_writing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "manifest.json"
    original = b'{"fixture": true}\n'
    source.write_bytes(original)
    non_json = runner.invoke(
        app,
        ["data", "pose-inputs", str(source), "--output", str(tmp_path / "sidecar.txt")],
    )
    assert non_json.exit_code == 2
    assert "must end in .json" in non_json.output
    alias = runner.invoke(
        app,
        [
            "data",
            "pose-inputs",
            str(source),
            "--output",
            str(source),
            "--overwrite",
        ],
    )
    assert alias.exit_code == 2
    assert "must not overwrite the source manifest" in alias.output
    assert source.read_bytes() == original


def test_pose_extract_label_free_path_never_loads_labelled_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sidecar = tmp_path / "pose-inputs.json"
    sidecar.write_text("{}\n", encoding="utf-8")
    (tmp_path / "pose-inputs.commitment.json").write_text("{}\n", encoding="utf-8")
    video = tmp_path / "video.avi"
    video.write_bytes(b"video")
    record = SimpleNamespace(
        video_id="opaque-video",
        video_path="video.avi",
        video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
    )
    fake_inputs = SimpleNamespace(
        protocol="ucfrep_526",
        split="test",
        records=(record,),
        fingerprint="e" * 64,
    )
    monkeypatch.setattr(
        cli_module,
        "load_pose_input_manifest",
        lambda *_args, **_kwargs: fake_inputs,
    )
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    fake_commitment = SimpleNamespace(
        protocol="ucfrep_526",
        split="test",
        record_total=1,
        identity_sha256=cli_module.pose_input_identity_sha256((record,)),
        sidecar_sha256=sidecar_sha256,
        sidecar_fingerprint="e" * 64,
        fingerprint="f" * 64,
    )
    monkeypatch.setattr(
        cli_module,
        "load_pose_input_commitment",
        lambda *_args, **_kwargs: fake_commitment,
    )
    monkeypatch.setattr(
        cli_module,
        "load_pose_cache_set",
        lambda *_args, **_kwargs: (
            (),
            SimpleNamespace(
                to_dict=lambda: {
                    "schema_version": 1,
                    "pose_fingerprint": "a" * 64,
                    "fingerprint": "b" * 64,
                    "entry_count": 1,
                    "entries": [],
                }
            ),
        ),
    )

    def fail_labelled_loader(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("labelled manifest loader must not be called")

    monkeypatch.setattr(cli_module, "load_ucfrep_manifest", fail_labelled_loader)
    summary = SimpleNamespace(
        video_id="opaque-video",
        skipped=False,
        to_dict=lambda: {"video_id": "opaque-video"},
    )
    monkeypatch.setattr(
        "pams.pose.extract_many_with_failures",
        lambda *_args, **_kwargs: ((summary,), ()),
    )

    result = runner.invoke(
        app,
        [
            "pose",
            "extract",
            str(sidecar),
            str(tmp_path / "cache"),
            "--config",
            str(REPOSITORY / "configs" / "pams.yaml"),
            "--label-free-manifest",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = _json_output(result.stdout)
    assert payload["label_free_manifest"] is True
    assert payload["input_manifest_fingerprint"] == "e" * 64
    ledger = json.loads(
        (tmp_path / "cache" / "failures.json").read_text(encoding="utf-8")
    )

    def nested_keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {
                key
                for child in value.values()
                for key in nested_keys(child)
            }
        if isinstance(value, list):
            return {key for child in value for key in nested_keys(child)}
        return set()

    ledger_keys = nested_keys(ledger)
    for forbidden in (
        "source_manifest_file_sha256",
        "source_manifest_fingerprint",
        "sealed_dataset_fingerprint",
        "count",
        "action",
    ):
        assert forbidden not in ledger_keys


def test_blocked_baseline_never_falls_back_to_proxy(tmp_path: Path) -> None:
    cache = _pose_cache(tmp_path)
    result = runner.invoke(app, ["baseline", "predict", "repnet", str(cache)])
    assert result.exit_code != 0
    assert "blocked_unimplemented" in result.output
    assert '"method": "spectral-proxy"' not in result.output


def test_synthetic_evaluation_requires_no_pose_dependency() -> None:
    result = runner.invoke(
        app,
        [
            "synthetic",
            "evaluate",
            "--config",
            str(REPOSITORY / "configs" / "pams.yaml"),
            "--counts",
            "2,4",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _json_output(result.stdout)
    assert payload["method"] == "spectral-proxy"
    assert payload["sample_count"] == 2
    assert payload["diagnostic_only"] is True


def test_prepare_ucfrep_defaults_to_strict_hashing_and_exposes_annotation_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    fake_archive = tmp_path / "annotations.zip"
    fake_archive.write_bytes(b"fixture")
    fake_manifest = SimpleNamespace(
        protocol="ucfrep_526",
        split_counts={"train": 421, "test": 105},
        fingerprint="f" * 64,
    )

    def fake_download(path: Path) -> Path:
        observed["annotation_path"] = path
        return fake_archive

    def fake_build(
        archive: Path,
        *,
        video_root: Path,
        hash_existing_videos: bool,
        allow_missing_videos: bool,
    ) -> object:
        observed.update(
            {
                "archive": archive,
                "video_root": video_root,
                "hash_existing_videos": hash_existing_videos,
                "allow_missing_videos": allow_missing_videos,
            }
        )
        return fake_manifest

    def fake_materialize(
        manifest: object,
        output_dir: Path,
        *,
        seed: int,
        overwrite: bool,
    ) -> dict[str, Path]:
        assert manifest is fake_manifest
        assert seed == 2026
        return {"split.txt": output_dir / "split.txt"}

    monkeypatch.setattr(ucfrep_module, "download_official_annotations", fake_download)
    monkeypatch.setattr(ucfrep_module, "build_official_manifest", fake_build)
    monkeypatch.setattr(ucfrep_module, "materialize_standard_split_ids", fake_materialize)
    monkeypatch.setattr(cli_module, "save_ucfrep_manifest", lambda *_args: None)

    result = runner.invoke(
        app,
        [
            "data",
            "prepare-ucfrep",
            str(tmp_path / "videos"),
            "--output",
            str(tmp_path / "manifest.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _json_output(result.stdout)
    assert observed["hash_existing_videos"] is True
    assert observed["allow_missing_videos"] is False
    assert payload["strict_video_files"] is True
    assert payload["annotation_only"] is False

    result = runner.invoke(
        app,
        [
            "data",
            "prepare-ucfrep",
            str(tmp_path / "videos"),
            "--output",
            str(tmp_path / "annotation-only.json"),
            "--annotation-only",
            "--no-hash-videos",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = _json_output(result.stdout)
    assert observed["hash_existing_videos"] is False
    assert observed["allow_missing_videos"] is True
    assert payload["strict_video_files"] is False
    assert payload["annotation_only"] is True


@pytest.mark.parametrize(
    "arguments",
    [
        ("pose", "extract", "{manifest}", "{cache}"),
        ("train", "encoder", "{manifest}", "{cache}", "{output}"),
        (
            "evaluate",
            "checkpoint",
            "{checkpoint}",
            "{manifest}",
            "{cache}",
            "{output}",
            "--variant",
            "literal",
        ),
    ],
)
def test_pose_train_and_evaluate_reject_config_manifest_protocol_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: tuple[str, ...],
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    mismatched = SimpleNamespace(protocol="ucfrep_pose_110")
    monkeypatch.setattr(cli_module, "load_ucfrep_manifest", lambda *_args, **_kwargs: mismatched)
    replacements = {
        "{manifest}": str(manifest_path),
        "{cache}": str(cache_dir),
        "{output}": str(output_dir),
        "{checkpoint}": str(checkpoint),
    }
    command = [replacements.get(argument, argument) for argument in arguments]
    command.extend(["--config", str(REPOSITORY / "configs" / "pams.yaml")])

    result = runner.invoke(app, command)

    assert result.exit_code != 0
    assert "config/manifest protocol mismatch" in result.output


def test_evaluate_checkpoint_requires_variant() -> None:
    result = runner.invoke(app, ["evaluate", "checkpoint", "--help"])
    assert result.exit_code == 0, result.output
    assert "--variant" in _plain_text(result.stdout)
    assert "literal" in result.stdout
    assert "sshead" in result.stdout
    assert "required" in result.stdout.lower()


def test_checkpoint_variant_parser_is_frozen_and_case_normalized() -> None:
    assert cli_module._normalize_checkpoint_variant(" Literal ") == "literal"
    assert cli_module._normalize_checkpoint_variant("SSHEAD") == "sshead"
    with pytest.raises(ValueError, match="literal.*sshead"):
        cli_module._normalize_checkpoint_variant("trained")


def test_validate_run_requires_sibling_start_and_every_artifact(
    tmp_path: Path,
) -> None:
    from pams.run_manifest import (
        create_completed_receipt,
        create_run_manifest,
        write_manifest_exclusive,
    )

    started = create_run_manifest(
        command=["pams", "fixture"],
        config_sha256="a" * 64,
        dataset_sha256="b" * 64,
        seed=2026,
        protocol="fixture",
        cwd=tmp_path,
    )
    manifests = tmp_path / "manifests"
    started_path = write_manifest_exclusive(
        started,
        manifests / f"{started.run_id}.started.json",
    )
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"bound-artifact")
    completed = create_completed_receipt(
        started_path,
        artifacts={"fixture": artifact},
    )
    completed_path = write_manifest_exclusive(
        completed,
        manifests / f"{started.run_id}.completed.json",
    )

    valid = runner.invoke(app, ["data", "validate-run", str(completed_path)])
    assert valid.exit_code == 0, valid.output

    artifact.write_bytes(b"BOUND-artifact")
    wrong_hash = runner.invoke(
        app,
        ["data", "validate-run", str(completed_path)],
    )
    assert wrong_hash.exit_code != 0
    assert "SHA-256" in wrong_hash.output

    artifact.write_bytes(b"short")
    wrong_size = runner.invoke(
        app,
        ["data", "validate-run", str(completed_path)],
    )
    assert wrong_size.exit_code != 0
    assert "size does not match" in wrong_size.output

    artifact.write_bytes(b"bound-artifact")
    artifact.unlink()
    missing_artifact = runner.invoke(
        app,
        ["data", "validate-run", str(completed_path)],
    )
    assert missing_artifact.exit_code != 0
    assert "missing artifact" in missing_artifact.output

    artifact.write_bytes(b"bound-artifact")
    started_path.unlink()
    missing_start = runner.invoke(
        app,
        ["data", "validate-run", str(completed_path)],
    )
    assert missing_start.exit_code != 0
    assert "missing sibling started receipt" in missing_start.output


def test_validate_run_rejects_duplicate_json_fields(tmp_path: Path) -> None:
    from pams.run_manifest import (
        create_completed_receipt,
        create_run_manifest,
        write_manifest_exclusive,
    )

    manifests = tmp_path / "manifests"
    started = create_run_manifest(
        command=["pams", "fixture"],
        config_sha256="a" * 64,
        dataset_sha256="b" * 64,
        seed=2026,
        protocol="fixture",
        cwd=tmp_path,
    )
    started_path = write_manifest_exclusive(
        started,
        manifests / f"{started.run_id}.started.json",
    )
    completed = create_completed_receipt(started_path)
    completed_path = write_manifest_exclusive(
        completed,
        manifests / f"{started.run_id}.completed.json",
    )
    duplicated = completed_path.read_text(encoding="utf-8").replace(
        '"schema_version": 3',
        '"schema_version": 999,\n  "schema_version": 3',
        1,
    )
    completed_path.write_text(duplicated, encoding="utf-8")

    result = runner.invoke(app, ["data", "validate-run", str(completed_path)])

    assert result.exit_code != 0
    assert "duplicate JSON field in run receipt: schema_version" in result.output


def test_validate_run_rejects_locator_outside_receipt_artifact_root(
    tmp_path: Path,
) -> None:
    from pams.run_manifest import (
        ArtifactReceipt,
        CompletedRunReceipt,
        create_run_manifest,
        write_manifest_exclusive,
    )

    package = tmp_path / "package"
    manifests = package / "manifests"
    started = create_run_manifest(
        command=["pams", "fixture"],
        config_sha256="a" * 64,
        dataset_sha256="b" * 64,
        seed=2026,
        protocol="fixture",
        cwd=tmp_path,
    )
    started_path = write_manifest_exclusive(
        started,
        manifests / f"{started.run_id}.started.json",
    )
    secret = tmp_path / "secret.txt"
    secret_payload = b"must-not-be-hashed-through-an-untrusted-receipt"
    secret.write_bytes(secret_payload)
    secret_sha256 = hashlib.sha256(secret_payload).hexdigest()
    completed = CompletedRunReceipt(
        run_id=started.run_id,
        finished_at=started.created_at_utc,
        start_manifest_sha256=hashlib.sha256(started_path.read_bytes()).hexdigest(),
        started=started,
        artifacts=(
            ArtifactReceipt(
                role="escape",
                locator="../../secret.txt",
                sha256="0" * 64,
                bytes=len(secret_payload),
            ),
        ),
    )
    completed_path = write_manifest_exclusive(
        completed,
        manifests / f"{started.run_id}.completed.json",
    )

    result = runner.invoke(app, ["data", "validate-run", str(completed_path)])

    assert result.exit_code != 0
    assert "escapes the trusted receipt artifact root" in result.output
    assert secret_sha256 not in result.output
    assert str(secret) not in result.output


def test_validate_run_uses_receipt_relative_locator_after_tree_move_and_safe_remap(
    tmp_path: Path,
) -> None:
    from pams.run_manifest import (
        create_completed_receipt,
        create_run_manifest,
        write_manifest_exclusive,
    )

    original_root = tmp_path / "original"
    manifests = original_root / "manifests"
    manifests.mkdir(parents=True)
    started = create_run_manifest(
        command=["pams", "fixture"],
        config_sha256="a" * 64,
        dataset_sha256="b" * 64,
        seed=2026,
        protocol="fixture",
        cwd=tmp_path,
    )
    started_path = write_manifest_exclusive(
        started,
        manifests / f"{started.run_id}.started.json",
    )
    artifact = original_root / "artifacts" / "fixture.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"portable-artifact")
    completed = create_completed_receipt(
        started_path,
        artifacts={"fixture": artifact},
    )
    completed_path = write_manifest_exclusive(
        completed,
        manifests / f"{started.run_id}.completed.json",
    )

    moved_root = tmp_path / "moved"
    original_root.rename(moved_root)
    moved_receipt = moved_root / "manifests" / completed_path.name
    moved_valid = runner.invoke(
        app,
        ["data", "validate-run", str(moved_receipt)],
    )
    assert moved_valid.exit_code == 0, moved_valid.output

    relocated_artifact = tmp_path / "relocated" / "fixture.bin"
    relocated_artifact.parent.mkdir()
    (moved_root / "artifacts" / "fixture.bin").rename(relocated_artifact)
    missing_default = runner.invoke(
        app,
        ["data", "validate-run", str(moved_receipt)],
    )
    assert missing_default.exit_code != 0
    assert "missing artifact" in missing_default.output

    remapped = runner.invoke(
        app,
        [
            "data",
            "validate-run",
            str(moved_receipt),
            "--artifact-remap",
            f"fixture={relocated_artifact.resolve()}",
        ],
    )
    assert remapped.exit_code == 0, remapped.output

    relative_remap = runner.invoke(
        app,
        [
            "data",
            "validate-run",
            str(moved_receipt),
            "--artifact-remap",
            "fixture=relative.bin",
        ],
    )
    assert relative_remap.exit_code != 0
    assert "absolute path" in relative_remap.output

    unknown_role = runner.invoke(
        app,
        [
            "data",
            "validate-run",
            str(moved_receipt),
            "--artifact-remap",
            f"unknown={relocated_artifact.resolve()}",
        ],
    )
    assert unknown_role.exit_code != 0
    assert "not present in the receipt" in unknown_role.output


def test_cli_completion_snapshots_external_artifacts_into_movable_run_tree(
    tmp_path: Path,
) -> None:
    from pams.cli import _complete_cli_run_manifest
    from pams.run_manifest import (
        ArtifactReceipt,
        CompletedRunReceipt,
        create_run_manifest,
        resolve_artifact_path,
        validate_artifact_receipt,
        write_manifest_exclusive,
    )

    run_root = tmp_path / "run"
    started = create_run_manifest(
        command=["pams", "fixture"],
        config_sha256="1" * 64,
        dataset_sha256="2" * 64,
        seed=2026,
        protocol="ucfrep_526",
        cwd=REPOSITORY,
    )
    started_path = write_manifest_exclusive(
        started,
        run_root / "manifests" / f"{started.run_id}.started.json",
    )
    external = tmp_path / "sealed-attempt-registry" / "attempt.json"
    external.parent.mkdir()
    external.write_bytes(b"immutable external receipt")
    digest = hashlib.sha256(external.read_bytes()).hexdigest()

    completed_path, _ = _complete_cli_run_manifest(
        started_path,
        artifacts={"sealed_attempt_receipt": external},
        expected_artifact_sha256={"sealed_attempt_receipt": digest},
        metrics={},
    )
    completed = CompletedRunReceipt.model_validate_json(
        completed_path.read_text(encoding="utf-8")
    )
    assert len(completed.artifacts) == 1
    artifact = completed.artifacts[0]
    assert isinstance(artifact, ArtifactReceipt)
    assert artifact.locator.startswith("../inputs/receipt-artifacts/")
    portable = resolve_artifact_path(completed_path, artifact)
    assert portable.read_bytes() == b"immutable external receipt"

    moved_root = tmp_path / "published-run"
    run_root.rename(moved_root)
    external.unlink()
    moved_receipt = moved_root / "manifests" / completed_path.name
    moved_completed = CompletedRunReceipt.model_validate_json(
        moved_receipt.read_text(encoding="utf-8")
    )
    moved_artifact = moved_completed.artifacts[0]
    moved_portable = resolve_artifact_path(moved_receipt, moved_artifact)
    validate_artifact_receipt(moved_portable, moved_artifact)


def test_cli_completion_isolates_same_role_across_runs_and_tree_move(
    tmp_path: Path,
) -> None:
    from pams.cli import _complete_cli_run_manifest
    from pams.run_manifest import (
        CompletedRunReceipt,
        create_run_manifest,
        resolve_artifact_path,
        write_manifest_exclusive,
    )

    run_root = tmp_path / "run-package"
    completed_paths: list[Path] = []
    expected_payloads = (b"first-run-artifact", b"second-run-artifact")
    for index, payload in enumerate(expected_payloads):
        started = create_run_manifest(
            command=["pams", "fixture", str(index)],
            config_sha256=f"{index + 1:x}" * 64,
            dataset_sha256=f"{index + 3:x}" * 64,
            seed=2026 + index,
            protocol="ucfrep_526",
            cwd=REPOSITORY,
        )
        started_path = write_manifest_exclusive(
            started,
            run_root / "manifests" / f"{started.run_id}.started.json",
        )
        external = tmp_path / "external" / f"attempt-{index}.json"
        external.parent.mkdir(exist_ok=True)
        external.write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        completed_path, _ = _complete_cli_run_manifest(
            started_path,
            artifacts={"sealed_attempt_receipt": external},
            expected_artifact_sha256={"sealed_attempt_receipt": digest},
            metrics={},
        )
        completed_paths.append(completed_path)

    receipts = [
        CompletedRunReceipt.model_validate_json(path.read_text(encoding="utf-8"))
        for path in completed_paths
    ]
    locators = [receipt.artifacts[0].locator for receipt in receipts]
    assert len(set(locators)) == 2
    assert all(
        locator.startswith("../inputs/receipt-artifacts/") for locator in locators
    )
    assert [
        resolve_artifact_path(path, receipt.artifacts[0]).read_bytes()
        for path, receipt in zip(completed_paths, receipts, strict=True)
    ] == list(expected_payloads)

    moved_root = tmp_path / "published-package"
    run_root.rename(moved_root)
    for completed_path in completed_paths:
        moved_receipt = moved_root / "manifests" / completed_path.name
        validated = runner.invoke(
            app,
            ["data", "validate-run", str(moved_receipt)],
        )
        assert validated.exit_code == 0, validated.output


def test_portable_receipt_snapshot_retry_reuses_only_exact_bytes(
    tmp_path: Path,
) -> None:
    from pams.cli import _materialize_portable_receipt_artifacts
    from pams.run_manifest import create_run_manifest, write_manifest_exclusive

    run_root = tmp_path / "run-package"
    started = create_run_manifest(
        command=["pams", "fixture"],
        config_sha256="1" * 64,
        dataset_sha256="2" * 64,
        seed=2026,
        protocol="ucfrep_526",
        cwd=REPOSITORY,
    )
    started_path = write_manifest_exclusive(
        started,
        run_root / "manifests" / f"{started.run_id}.started.json",
    )
    external = tmp_path / "external.json"
    payload = b"retry-safe-artifact"
    external.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()

    first = _materialize_portable_receipt_artifacts(
        started_path,
        artifacts={"sealed_attempt_receipt": external},
        expected_artifact_sha256={"sealed_attempt_receipt": digest},
    )
    second = _materialize_portable_receipt_artifacts(
        started_path,
        artifacts={"sealed_attempt_receipt": external},
        expected_artifact_sha256={"sealed_attempt_receipt": digest},
    )

    assert first == second
    assert first["sealed_attempt_receipt"].read_bytes() == payload


def test_portable_receipt_snapshot_rejects_corrupted_existing_target(
    tmp_path: Path,
) -> None:
    from pams.cli import _materialize_portable_receipt_artifacts
    from pams.run_manifest import create_run_manifest, write_manifest_exclusive

    run_root = tmp_path / "run-package"
    started = create_run_manifest(
        command=["pams", "fixture"],
        config_sha256="1" * 64,
        dataset_sha256="2" * 64,
        seed=2026,
        protocol="ucfrep_526",
        cwd=REPOSITORY,
    )
    started_path = write_manifest_exclusive(
        started,
        run_root / "manifests" / f"{started.run_id}.started.json",
    )
    external = tmp_path / "external.json"
    payload = b"expected-artifact"
    external.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    started_digest = hashlib.sha256(started_path.read_bytes()).hexdigest()
    role_digest = hashlib.sha256(b"sealed_attempt_receipt").hexdigest()
    destination = (
        run_root
        / "inputs"
        / "receipt-artifacts"
        / started_digest
        / f"{role_digest}.artifact"
    )
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"corrupted-existing-artifact")

    with pytest.raises(FileExistsError, match="different bytes"):
        _materialize_portable_receipt_artifacts(
            started_path,
            artifacts={"sealed_attempt_receipt": external},
            expected_artifact_sha256={"sealed_attempt_receipt": digest},
        )

    assert destination.read_bytes() == b"corrupted-existing-artifact"


def test_portable_receipt_snapshot_rejects_symlink_target(tmp_path: Path) -> None:
    from pams.cli import _materialize_portable_receipt_artifacts
    from pams.run_manifest import create_run_manifest, write_manifest_exclusive

    run_root = tmp_path / "run-package"
    started = create_run_manifest(
        command=["pams", "fixture"],
        config_sha256="1" * 64,
        dataset_sha256="2" * 64,
        seed=2026,
        protocol="ucfrep_526",
        cwd=REPOSITORY,
    )
    started_path = write_manifest_exclusive(
        started,
        run_root / "manifests" / f"{started.run_id}.started.json",
    )
    external = tmp_path / "external.json"
    payload = b"expected-artifact"
    external.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    started_digest = hashlib.sha256(started_path.read_bytes()).hexdigest()
    role_digest = hashlib.sha256(b"sealed_attempt_receipt").hexdigest()
    destination = (
        run_root
        / "inputs"
        / "receipt-artifacts"
        / started_digest
        / f"{role_digest}.artifact"
    )
    destination.parent.mkdir(parents=True)
    attacker_target = tmp_path / "attacker-controlled.bin"
    attacker_target.write_bytes(payload)
    try:
        destination.symlink_to(attacker_target)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with pytest.raises(FileExistsError, match="non-symlink"):
        _materialize_portable_receipt_artifacts(
            started_path,
            artifacts={"sealed_attempt_receipt": external},
            expected_artifact_sha256={"sealed_attempt_receipt": digest},
        )

    assert destination.is_symlink()
    assert attacker_target.read_bytes() == payload


def test_validate_run_accepts_schema_v2_receipt_with_explicit_migration_remap(
    tmp_path: Path,
) -> None:
    from pams.run_manifest import (
        CompletedRunReceipt,
        LegacyArtifactReceipt,
        create_run_manifest,
        write_manifest_exclusive,
    )

    started = create_run_manifest(
        command=["pams", "legacy-fixture"],
        config_sha256="a" * 64,
        dataset_sha256="b" * 64,
        seed=2026,
        protocol="fixture",
        cwd=tmp_path,
    )
    manifests = tmp_path / "manifests"
    started_path = write_manifest_exclusive(
        started,
        manifests / f"{started.run_id}.started.json",
    )
    migrated_artifact = tmp_path / "migrated" / "fixture.bin"
    migrated_artifact.parent.mkdir()
    payload = b"legacy-portable"
    migrated_artifact.write_bytes(payload)
    completed = CompletedRunReceipt(
        schema_version=2,
        run_id=started.run_id,
        finished_at=started.created_at_utc,
        start_manifest_sha256=hashlib.sha256(started_path.read_bytes()).hexdigest(),
        started=started,
        artifacts=(
            LegacyArtifactReceipt(
                role="fixture",
                path=str((tmp_path / "stale" / "fixture.bin").resolve()),
                sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload),
            ),
        ),
    )
    completed_path = write_manifest_exclusive(
        completed,
        manifests / f"{started.run_id}.completed.json",
    )

    stale_default = runner.invoke(
        app,
        ["data", "validate-run", str(completed_path)],
    )
    assert stale_default.exit_code != 0
    assert "requires an explicit absolute role remap" in stale_default.output

    migrated = runner.invoke(
        app,
        [
            "data",
            "validate-run",
            str(completed_path),
            "--artifact-remap",
            f"fixture={migrated_artifact.resolve()}",
        ],
    )
    assert migrated.exit_code == 0, migrated.output


def test_evaluate_literal_binds_stage_and_writes_audit_fingerprints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import evaluation as evaluation_module
    from pams import training as training_module

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "encoder.pt"
    checkpoint_bytes = b"encoder-checkpoint"
    checkpoint.write_bytes(checkpoint_bytes)
    checkpoint_progress = tmp_path / "encoder.jsonl"
    checkpoint_progress.write_text("{}\n", encoding="utf-8")
    fake_manifest = _FakeManifest()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module,
        "load_ucfrep_manifest",
        lambda *_args, **_kwargs: fake_manifest,
    )
    monkeypatch.setattr(cli_module, "_validate_experiment_split", lambda _manifest: None)
    monkeypatch.setattr(
        cli_module,
        "_cached_sequences",
        lambda *_args, **_kwargs: ([object()], ["test-a"], _FakePoseSnapshot()),
    )
    monkeypatch.setattr(
        cli_module,
        "_training_pose_cache_snapshot",
        lambda *_args, **_kwargs: _FakePoseSnapshot(),
    )
    monkeypatch.setattr(
        cli_module,
        "_validate_final_training_pool",
        lambda *_args, **_kwargs: None,
    )

    def fake_create_manifest(**kwargs: object) -> Path:
        observed["started_dataset_sha256"] = kwargs["dataset_sha256"]
        return tmp_path / "run.json"

    monkeypatch.setattr(cli_module, "_create_cli_run_manifest", fake_create_manifest)
    monkeypatch.setattr(
        cli_module,
        "_complete_cli_run_manifest",
        lambda *_args, **_kwargs: (tmp_path / "completed.json", "f" * 64),
    )

    def fake_load(
        _path: Path,
        _config: object,
        *,
        expected_stage: str | None = None,
        expected_provenance: object = None,
        device: str = "auto",
    ) -> object:
        observed["stage"] = expected_stage
        observed["provenance"] = expected_provenance
        observed["device"] = device
        return object()

    class _Evaluation:
        report = SimpleNamespace(to_dict=lambda: {"nmae": 0.1, "per_video": []})

        def to_dict(self, *, include_streams: bool) -> dict[str, object]:
            assert include_streams is True
            return {"report": {"nmae": 0.1}, "predictions": []}

    monkeypatch.setattr(cli_module, "_load_checkpoint_model", fake_load)
    terminal_calls: list[tuple[Path, str, Path]] = []

    def fake_validate_terminal(
        path: Path,
        _config: object,
        *,
        expected_stage: str,
        expected_provenance: object,
        progress_path: Path,
    ) -> object:
        del expected_provenance
        terminal_calls.append((path, expected_stage, progress_path))
        return object()

    monkeypatch.setattr(
        training_module,
        "validate_terminal_checkpoint",
        fake_validate_terminal,
    )
    monkeypatch.setattr(
        evaluation_module,
        "predict_sequences",
        lambda *_args, **_kwargs: (SimpleNamespace(video_id="test-a"),),
    )
    monkeypatch.setattr(
        evaluation_module,
        "evaluate_predictions",
        lambda *_args, **_kwargs: _Evaluation(),
    )
    monkeypatch.setattr(
        "pams.reproducibility.clean_git_revision",
        lambda _cwd: "a" * 40,
    )
    _set_fake_container_identity(monkeypatch)
    run_root = tmp_path / "run-root"
    monkeypatch.setenv("PAMS_RUN_ROOT", str(run_root))
    attempt_registry = run_root / "sealed-test-attempts" / "ucfrep_526"

    altered_config = tmp_path / "altered-pams.yaml"
    altered_config.write_text(
        (REPOSITORY / "configs" / "pams.yaml")
        .read_text(encoding="utf-8")
        .replace("learning_rate: 0.0001", "learning_rate: 0.0002", 1),
        encoding="utf-8",
    )
    altered_result = runner.invoke(
        app,
        [
            "evaluate",
            "checkpoint",
            str(checkpoint),
            str(manifest_path),
            str(cache_dir),
            str(tmp_path / "altered-output"),
            "--variant",
            "literal",
            "--checkpoint-progress",
            str(checkpoint_progress),
            "--method-id",
            "pams-literal",
            "--sealed-attempt-registry",
            str(attempt_registry),
            "--config",
            str(altered_config),
        ],
    )
    assert altered_result.exit_code != 0
    assert "differs from the frozen non-seed specification" in altered_result.output
    assert not attempt_registry.exists()
    assert terminal_calls == []

    result = runner.invoke(
        app,
        [
            "evaluate",
            "checkpoint",
            str(checkpoint),
            str(manifest_path),
            str(cache_dir),
            str(output_dir),
            "--variant",
            "literal",
            "--checkpoint-progress",
            str(checkpoint_progress),
            "--method-id",
            "pams-literal",
            "--sealed-attempt-registry",
            str(attempt_registry),
            "--config",
            str(REPOSITORY / "configs" / "pams.yaml"),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads((output_dir / "evaluation.json").read_text(encoding="utf-8"))
    assert observed["stage"] == "encoder"
    assert terminal_calls == [(checkpoint, "encoder", checkpoint_progress)]
    provenance = observed["provenance"]
    assert isinstance(provenance, CheckpointProvenance)
    assert provenance.training_video_ids == ("train-a", "train-b")
    assert provenance.dataset_fingerprint == "e" * 64
    assert provenance.upstream_encoder_checkpoint_sha256 is None
    assert observed["started_dataset_sha256"] == "d" * 64
    assert payload["variant"] == "literal"
    assert payload["stage"] == "encoder"
    assert payload["checkpoint_sha256"] == hashlib.sha256(checkpoint_bytes).hexdigest()
    assert payload["checkpoint_progress_sha256"] == hashlib.sha256(
        checkpoint_progress.read_bytes()
    ).hexdigest()
    assert payload["dataset_fingerprint"] == "d" * 64
    assert len(payload["config_fingerprint"]) == 64
    assert len(payload["pose_fingerprint"]) == 64


def test_evaluate_sshead_requires_and_binds_upstream_encoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import evaluation as evaluation_module
    from pams import training as training_module

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    output_dir = tmp_path / "output"
    checkpoint = tmp_path / "sshead.pt"
    checkpoint.write_bytes(b"head")
    checkpoint_progress = tmp_path / "sshead.jsonl"
    checkpoint_progress.write_text("{}\n", encoding="utf-8")
    upstream = tmp_path / "encoder.pt"
    upstream_bytes = b"upstream"
    upstream.write_bytes(upstream_bytes)
    upstream_progress = tmp_path / "encoder.jsonl"
    upstream_progress.write_text("{}\n", encoding="utf-8")
    fake_manifest = _FakeManifest()
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        cli_module,
        "load_ucfrep_manifest",
        lambda *_args, **_kwargs: fake_manifest,
    )
    monkeypatch.setattr(cli_module, "_validate_experiment_split", lambda _manifest: None)
    monkeypatch.setattr(
        cli_module,
        "_cached_sequences",
        lambda *_args, **_kwargs: ([object()], ["test-a"], _FakePoseSnapshot()),
    )
    monkeypatch.setattr(
        cli_module,
        "_training_pose_cache_snapshot",
        lambda *_args, **_kwargs: _FakePoseSnapshot(),
    )
    monkeypatch.setattr(
        cli_module,
        "_validate_final_training_pool",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        cli_module,
        "_create_cli_run_manifest",
        lambda **_kwargs: tmp_path / "run.json",
    )
    monkeypatch.setattr(
        cli_module,
        "_complete_cli_run_manifest",
        lambda *_args, **_kwargs: (tmp_path / "completed.json", "f" * 64),
    )

    def fake_load(
        _path: Path,
        _config: object,
        *,
        expected_stage: str | None = None,
        expected_provenance: object = None,
        device: str = "auto",
    ) -> object:
        observed["stage"] = expected_stage
        observed["provenance"] = expected_provenance
        return object()

    class _Evaluation:
        report = SimpleNamespace(to_dict=lambda: {"nmae": 0.1, "per_video": []})

        def to_dict(self, *, include_streams: bool) -> dict[str, object]:
            return {"report": {}, "predictions": []}

    monkeypatch.setattr(cli_module, "_load_checkpoint_model", fake_load)
    terminal_stages: list[str] = []
    binding_calls: list[tuple[Path, Path]] = []

    def fake_validate_terminal(
        _path: Path,
        _config: object,
        *,
        expected_stage: str,
        expected_provenance: object,
        progress_path: Path,
    ) -> object:
        del expected_provenance, progress_path
        terminal_stages.append(expected_stage)
        return object()

    monkeypatch.setattr(
        training_module,
        "validate_terminal_checkpoint",
        fake_validate_terminal,
    )
    monkeypatch.setattr(
        training_module,
        "validate_sshead_encoder_binding",
        lambda head, encoder: binding_calls.append((head, encoder)),
    )
    monkeypatch.setattr(
        evaluation_module,
        "predict_sequences",
        lambda *_args, **_kwargs: (SimpleNamespace(video_id="test-a"),),
    )
    monkeypatch.setattr(
        evaluation_module,
        "evaluate_predictions",
        lambda *_args, **_kwargs: _Evaluation(),
    )
    monkeypatch.setattr(
        "pams.reproducibility.clean_git_revision",
        lambda _cwd: "a" * 40,
    )
    _set_fake_container_identity(monkeypatch)
    run_root = tmp_path / "run-root"
    monkeypatch.setenv("PAMS_RUN_ROOT", str(run_root))
    attempt_registry = run_root / "sealed-test-attempts" / "ucfrep_526"

    missing = runner.invoke(
        app,
        [
            "evaluate",
            "checkpoint",
            str(checkpoint),
            str(manifest_path),
            str(cache_dir),
            str(output_dir),
            "--variant",
            "sshead",
            "--config",
            str(REPOSITORY / "configs" / "pams.yaml"),
        ],
    )
    assert missing.exit_code != 0
    assert "--upstream-encoder-checkpoint is required" in missing.output

    result = runner.invoke(
        app,
        [
            "evaluate",
            "checkpoint",
            str(checkpoint),
            str(manifest_path),
            str(cache_dir),
            str(output_dir),
            "--variant",
            "sshead",
            "--checkpoint-progress",
            str(checkpoint_progress),
            "--upstream-encoder-checkpoint",
            str(upstream),
            "--upstream-encoder-progress",
            str(upstream_progress),
            "--method-id",
            "pams-sshead",
            "--sealed-attempt-registry",
            str(attempt_registry),
            "--config",
            str(REPOSITORY / "configs" / "pams.yaml"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert terminal_stages == ["sshead", "encoder"]
    assert binding_calls == [(checkpoint, upstream)]
    assert observed["stage"] == "sshead"
    provenance = observed["provenance"]
    assert isinstance(provenance, CheckpointProvenance)
    assert (
        provenance.upstream_encoder_checkpoint_sha256 == hashlib.sha256(upstream_bytes).hexdigest()
    )


def test_training_cli_always_supplies_bound_provenance_and_refuses_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import training as training_module

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    fake_manifest = _FakeManifest()
    observed: dict[str, object] = {}
    encoder_calls = 0

    monkeypatch.setattr(
        cli_module,
        "_training_inputs",
        lambda *_args, **_kwargs: (
            fake_manifest,
            (object(),),
            _FakePoseSnapshot(),
        ),
    )
    monkeypatch.setattr(
        "pams.reproducibility.clean_git_revision",
        lambda _cwd: "a" * 40,
    )
    _set_fake_container_identity(monkeypatch)
    started_dataset_sha256: list[object] = []

    def fake_create_manifest(**kwargs: object) -> Path:
        started_dataset_sha256.append(kwargs["dataset_sha256"])
        return tmp_path / "run.json"

    monkeypatch.setattr(cli_module, "_create_cli_run_manifest", fake_create_manifest)
    monkeypatch.setattr(
        cli_module,
        "_complete_cli_run_manifest",
        lambda *_args, **_kwargs: (tmp_path / "completed.json", "f" * 64),
    )

    def fake_train_encoder(*_args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal encoder_calls
        encoder_calls += 1
        observed["encoder_provenance"] = kwargs["provenance"]
        observed["encoder_progress_path"] = kwargs["progress_path"]
        observed["allow_negative_shortfall"] = kwargs["allow_negative_shortfall"]
        kwargs["checkpoint_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["checkpoint_path"].write_bytes(b"encoder-checkpoint")
        kwargs["progress_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["progress_path"].write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            completed_epochs=1,
            checkpoint_path=kwargs["checkpoint_path"],
            cluster_assignments={},
            history=(),
        )

    monkeypatch.setattr(training_module, "train_encoder", fake_train_encoder)
    encoder_output = tmp_path / "encoder-output"
    encoder_result = runner.invoke(
        app,
        [
            "train",
            "encoder",
            str(manifest_path),
            str(cache_dir),
            str(encoder_output),
            "--config",
            str(REPOSITORY / "configs" / "pams.yaml"),
        ],
    )
    assert encoder_result.exit_code == 0, encoder_result.output
    encoder_payload = _json_output(encoder_result.stdout)
    expected_encoder_progress = encoder_output / "logs" / "encoder.jsonl"
    assert observed["encoder_progress_path"] == expected_encoder_progress
    assert observed["allow_negative_shortfall"] is False
    assert encoder_payload["progress_path"] == str(expected_encoder_progress.resolve())
    encoder_provenance = observed["encoder_provenance"]
    assert isinstance(encoder_provenance, CheckpointProvenance)
    assert encoder_provenance.dataset_fingerprint == "e" * 64
    assert encoder_provenance.training_video_ids == ("train-a", "train-b")
    assert encoder_provenance.upstream_encoder_checkpoint_sha256 is None
    assert encoder_provenance.container_image_id == f"sha256:{'1' * 64}"
    assert encoder_provenance.container_environment_sha256 == "2" * 64
    assert started_dataset_sha256[0] == "e" * 64

    log_only_output = tmp_path / "log-only-output"
    existing_log = log_only_output / "logs" / "encoder.jsonl"
    existing_log.parent.mkdir(parents=True)
    existing_log.write_text("sentinel-log\n", encoding="utf-8")
    log_overwrite_result = runner.invoke(
        app,
        [
            "train",
            "encoder",
            str(manifest_path),
            str(cache_dir),
            str(log_only_output),
            "--config",
            str(REPOSITORY / "configs" / "pams.yaml"),
        ],
    )
    assert log_overwrite_result.exit_code != 0
    assert "refusing to overwrite existing progress log" in log_overwrite_result.output
    assert encoder_calls == 1
    assert existing_log.read_text(encoding="utf-8") == "sentinel-log\n"

    destination = encoder_output / "encoder.pt"
    destination.write_bytes(b"sentinel")
    overwrite_result = runner.invoke(
        app,
        [
            "train",
            "encoder",
            str(manifest_path),
            str(cache_dir),
            str(encoder_output),
            "--config",
            str(REPOSITORY / "configs" / "pams.yaml"),
        ],
    )
    assert overwrite_result.exit_code != 0
    assert "refusing to overwrite existing checkpoint" in overwrite_result.output
    assert encoder_calls == 1
    assert destination.read_bytes() == b"sentinel"

    upstream = tmp_path / "upstream-encoder.pt"
    upstream_bytes = b"upstream-encoder"
    upstream.write_bytes(upstream_bytes)
    upstream_progress = tmp_path / "upstream-encoder.jsonl"
    upstream_progress.write_text("{}\n", encoding="utf-8")

    def fake_load(
        _path: Path,
        _config: object,
        *,
        expected_stage: str | None = None,
        expected_provenance: object = None,
        device: str = "auto",
    ) -> object:
        observed["load_stage"] = expected_stage
        observed["load_provenance"] = expected_provenance
        return object()

    def fake_train_sshead(*_args: object, **kwargs: object) -> SimpleNamespace:
        observed["sshead_provenance"] = kwargs["provenance"]
        observed["sshead_progress_path"] = kwargs["progress_path"]
        kwargs["checkpoint_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["checkpoint_path"].write_bytes(b"sshead-checkpoint")
        kwargs["progress_path"].parent.mkdir(parents=True, exist_ok=True)
        kwargs["progress_path"].write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(
            completed_epochs=1,
            checkpoint_path=kwargs["checkpoint_path"],
            history=(),
        )

    monkeypatch.setattr(cli_module, "_load_checkpoint_model", fake_load)
    monkeypatch.setattr(training_module, "train_sshead", fake_train_sshead)
    terminal_calls: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        training_module,
        "validate_terminal_checkpoint",
        lambda path, _config, **kwargs: terminal_calls.append(
            (path, kwargs["progress_path"])
        ),
    )
    head_result = runner.invoke(
        app,
        [
            "train",
            "sshead",
            str(upstream),
            str(manifest_path),
            str(cache_dir),
            str(tmp_path / "head-output"),
            "--encoder-progress",
            str(upstream_progress),
            "--config",
            str(REPOSITORY / "configs" / "pams.yaml"),
        ],
    )
    assert head_result.exit_code == 0, head_result.output
    head_payload = _json_output(head_result.stdout)
    expected_head_progress = tmp_path / "head-output" / "logs" / "sshead.jsonl"
    assert observed["sshead_progress_path"] == expected_head_progress
    assert terminal_calls == [(upstream, upstream_progress)]
    assert head_payload["progress_path"] == str(expected_head_progress.resolve())
    assert observed["load_stage"] == "encoder"
    load_provenance = observed["load_provenance"]
    assert isinstance(load_provenance, CheckpointProvenance)
    assert load_provenance.upstream_encoder_checkpoint_sha256 is None
    head_provenance = observed["sshead_provenance"]
    assert isinstance(head_provenance, CheckpointProvenance)
    assert head_provenance.dataset_fingerprint == "e" * 64
    assert started_dataset_sha256[1] == "e" * 64
    assert (
        head_provenance.upstream_encoder_checkpoint_sha256
        == hashlib.sha256(upstream_bytes).hexdigest()
    )


def test_encoder_cli_resume_copies_immutable_inputs_into_a_new_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import training as training_module

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    prior_checkpoint = tmp_path / "prior" / "encoder.pt"
    prior_progress = tmp_path / "prior" / "encoder.jsonl"
    prior_checkpoint.parent.mkdir()
    prior_checkpoint.write_bytes(b"immutable-prior-checkpoint")
    prior_progress.write_bytes(b"immutable-prior-progress\n")
    prior_checkpoint_sha256 = hashlib.sha256(prior_checkpoint.read_bytes()).hexdigest()
    prior_progress_sha256 = hashlib.sha256(prior_progress.read_bytes()).hexdigest()
    output_dir = tmp_path / "resumed-output"
    completion: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module,
        "_training_inputs",
        lambda *_args, **_kwargs: (
            _FakeManifest(),
            (object(),),
            _FakePoseSnapshot(),
        ),
    )
    monkeypatch.setattr(
        "pams.reproducibility.clean_git_revision",
        lambda _cwd: "a" * 40,
    )
    _set_fake_container_identity(monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "_create_cli_run_manifest",
        lambda **_kwargs: tmp_path / "started.json",
    )

    def fake_complete(
        _started: Path,
        *,
        artifacts: dict[str, Path],
        expected_artifact_sha256: dict[str, str],
        metrics: dict[str, object],
    ) -> tuple[Path, str]:
        completion["artifacts"] = artifacts
        completion["expected"] = expected_artifact_sha256
        completion["metrics"] = metrics
        return tmp_path / "completed.json", "f" * 64

    monkeypatch.setattr(cli_module, "_complete_cli_run_manifest", fake_complete)

    def fake_train(*_args: object, **kwargs: object) -> SimpleNamespace:
        assert kwargs["resume"] is True
        checkpoint_path = kwargs["checkpoint_path"]
        progress_path = kwargs["progress_path"]
        assert checkpoint_path.read_bytes() == prior_checkpoint.read_bytes()
        assert progress_path.read_bytes() == prior_progress.read_bytes()
        checkpoint_path.write_bytes(b"resumed-checkpoint")
        progress_path.write_bytes(b"resumed-progress\n")
        return SimpleNamespace(
            completed_epochs=2,
            checkpoint_path=checkpoint_path,
            cluster_assignments={},
            history=(),
        )

    monkeypatch.setattr(training_module, "train_encoder", fake_train)
    result = runner.invoke(
        app,
        [
            "train",
            "encoder",
            str(manifest_path),
            str(cache_dir),
            str(output_dir),
            "--resume",
            "--resume-checkpoint",
            str(prior_checkpoint),
            "--resume-progress",
            str(prior_progress),
            "--config",
            str(REPOSITORY / "configs" / "pams.yaml"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert prior_checkpoint.read_bytes() == b"immutable-prior-checkpoint"
    assert prior_progress.read_bytes() == b"immutable-prior-progress\n"
    assert (output_dir / "encoder.pt").read_bytes() == b"resumed-checkpoint"
    assert (output_dir / "logs" / "encoder.jsonl").read_bytes() == b"resumed-progress\n"
    artifacts = completion["artifacts"]
    expected = completion["expected"]
    assert isinstance(artifacts, dict)
    assert isinstance(expected, dict)
    assert artifacts["input_resume_checkpoint"] == prior_checkpoint
    assert artifacts["input_resume_progress"] == prior_progress
    assert expected["input_resume_checkpoint"] == prior_checkpoint_sha256
    assert expected["input_resume_progress"] == prior_progress_sha256


def test_report_metrics_reserves_canonical_test_attempt_after_prediction_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    experiment_config = tmp_path / "repnet.yaml"
    experiment_config.write_text("checkpoint: frozen\n", encoding="utf-8")
    records = (
        SimpleNamespace(video_id="test-a", count=4, action="jump"),
        SimpleNamespace(video_id="test-b", count=7, action="squat"),
    )
    fake_manifest = SimpleNamespace(
        protocol="ucfrep_526",
        fingerprint="d" * 64,
        sealed_dataset_fingerprint="9" * 64,
        records_for=lambda split: records if split == "test" else (),
    )
    monkeypatch.setattr(
        cli_module,
        "load_ucfrep_manifest",
        lambda *_args, **_kwargs: fake_manifest,
    )
    monkeypatch.setattr(cli_module, "_validate_experiment_split", lambda _manifest: None)
    monkeypatch.setattr(
        cli_module,
        "_FROZEN_SEALED_REPORT_METHOD_IDS_BY_PROTOCOL",
        {"ucfrep_526": frozenset({"fixture-ready"})},
    )
    monkeypatch.setattr(
        "pams.reproducibility.clean_git_revision",
        lambda _cwd: "a" * 40,
    )
    run_root = tmp_path / "run-root"
    monkeypatch.setenv("PAMS_RUN_ROOT", str(run_root))
    registry = run_root / "sealed-test-attempts" / "ucfrep_526"

    bad_predictions = tmp_path / "bad.json"
    bad_predictions.write_text('{"wrong-id": 4}\n', encoding="utf-8")
    common = [
        str(manifest_path),
        "--method-id",
        "fixture-ready",
        "--experiment-seed",
        "2026",
        "--experiment-config",
        str(experiment_config),
        "--bootstrap-samples",
        "10000",
    ]
    invalid_bootstrap = runner.invoke(
        app,
        [
            "report",
            "metrics",
            str(bad_predictions),
            *common[:-1],
            "0",
            "--output",
            str(tmp_path / "invalid-bootstrap.json"),
        ],
    )
    assert invalid_bootstrap.exit_code != 0
    assert "exactly 10,000" in invalid_bootstrap.output
    assert not tuple(registry.glob("*.json"))

    bad = runner.invoke(
        app,
        [
            "report",
            "metrics",
            str(bad_predictions),
            *common,
            "--output",
            str(tmp_path / "bad-metrics.json"),
        ],
    )
    assert bad.exit_code != 0
    assert "prediction IDs do not match" in bad.output
    assert not tuple(registry.glob("*.json"))

    predictions = tmp_path / "predictions.json"
    predictions.write_text('{"test-a": 4, "test-b": 6}\n', encoding="utf-8")
    invalid_seed = runner.invoke(
        app,
        [
            "report",
            "metrics",
            str(predictions),
            *common,
            "--bootstrap-seed",
            "42",
            "--output",
            str(tmp_path / "invalid-bootstrap-seed.json"),
        ],
    )
    assert invalid_seed.exit_code != 0
    assert "bootstrap seed 2026" in invalid_seed.output
    assert not tuple(registry.glob("*.json"))

    pams_side_door = list(common)
    pams_side_door[pams_side_door.index("fixture-ready")] = "pams-sshead"
    pams_report = runner.invoke(
        app,
        [
            "report",
            "metrics",
            str(predictions),
            *pams_side_door,
            "--output",
            str(tmp_path / "pams-side-door.json"),
        ],
    )
    assert pams_report.exit_code != 0
    assert "source-audit registry" in pams_report.output
    assert not tuple(registry.glob("*.json"))

    output = tmp_path / "metrics.json"
    first = runner.invoke(
        app,
        ["report", "metrics", str(predictions), *common, "--output", str(output)],
    )
    assert first.exit_code == 0, first.output
    payload = _json_output(first.stdout)
    assert payload["sample_count"] == 2
    assert payload["sealed_attempt_path"] is not None
    assert len(tuple(registry.glob("*.json"))) == 1

    repeated = runner.invoke(
        app,
        [
            "report",
            "metrics",
            str(predictions),
            *common,
            "--output",
            str(tmp_path / "metrics-again.json"),
        ],
    )
    assert repeated.exit_code != 0
    assert "already reserved" in repeated.output

    alternate_common = list(common)
    alternate_common[alternate_common.index("2026")] = "3407"
    alternate_registry = runner.invoke(
        app,
        [
            "report",
            "metrics",
            str(predictions),
            *alternate_common,
            "--sealed-attempt-registry",
            str(tmp_path / "attempts-b"),
            "--output",
            str(tmp_path / "alternate.json"),
        ],
    )
    assert alternate_registry.exit_code != 0
    assert "canonical shared path" in alternate_registry.output


def test_sealed_method_registry_is_protocol_and_variant_scoped() -> None:
    assert (
        cli_module._normalize_frozen_method_id(
            "ucfrep_526",
            "PAMS-LITERAL",
            checkpoint_variant="literal",
        )
        == "pams-literal"
    )
    with pytest.raises(ValueError, match="requires checkpoint variant"):
        cli_module._normalize_frozen_method_id(
            "ucfrep_526",
            "pams-sshead",
            checkpoint_variant="literal",
        )
    with pytest.raises(ValueError, match="ucfrep_pose_110"):
        cli_module._normalize_frozen_method_id(
            "ucfrep_pose_110",
            "repnet-official-current-ckpt70",
        )
    with pytest.raises(ValueError, match="ucfrep_526"):
        cli_module._normalize_frozen_method_id(
            "ucfrep_526",
            "gmfl-cleanroom",
        )


def test_synthetic_safety_commands_publish_and_propagate_gate_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pams import safety

    counter_report = {
        "schema_version": 1,
        "gate": "counter-sign-phase",
        "passed": True,
        "metrics": {"case_count": 576},
    }
    collapse_report = {
        "schema_version": 1,
        "gate": "sshead-collapse",
        "passed": False,
        "diagnosis": {"exact_dead_point_count": 32},
    }
    monkeypatch.setattr(
        safety,
        "run_counter_sign_phase_gate",
        lambda: counter_report,
    )
    monkeypatch.setattr(
        safety,
        "run_sshead_collapse_diagnostic",
        lambda: collapse_report,
    )

    destination = tmp_path / "counter-gate.json"
    passed = runner.invoke(
        app,
        ["synthetic", "counter-gate", "--output", str(destination)],
    )
    assert passed.exit_code == 0, passed.output
    assert _json_output(passed.stdout) == counter_report
    assert json.loads(destination.read_text(encoding="utf-8")) == counter_report

    refused = runner.invoke(
        app,
        ["synthetic", "counter-gate", "--output", str(destination)],
    )
    assert refused.exit_code != 0
    assert "refusing to overwrite" in refused.output
    assert json.loads(destination.read_text(encoding="utf-8")) == counter_report

    failed_destination = tmp_path / "sshead-collapse.json"
    failed = runner.invoke(
        app,
        ["synthetic", "sshead-collapse", "--output", str(failed_destination)],
    )
    assert failed.exit_code == 1
    assert _json_output(failed.stdout) == collapse_report
    assert json.loads(failed_destination.read_text(encoding="utf-8")) == collapse_report
