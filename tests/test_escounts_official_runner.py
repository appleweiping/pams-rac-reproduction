from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pams.baselines import escounts_official_runner as runner
from pams.baselines import escounts_official_worker as worker
from pams.data import (
    PoseInputCommitment,
    PoseInputManifest,
    UnlabeledVideoRecord,
    pose_input_identity_sha256,
)

REPOSITORY = Path(__file__).parents[1]


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@pytest.fixture
def audited_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    repository = tmp_path / "audited-repository"
    baseline_dir = repository / "src/pams/baselines"
    baseline_dir.mkdir(parents=True)
    runner_path = baseline_dir / "escounts_official_runner.py"
    worker_path = baseline_dir / "escounts_official_worker.py"
    runner_path.write_bytes(Path(runner.__file__).resolve(strict=True).read_bytes())
    worker_path.write_bytes(runner._WORKER_PATH.read_bytes())
    _git(repository, "init")
    _git(repository, "config", "user.email", "tests@example.invalid")
    _git(repository, "config", "user.name", "ESCounts Tests")
    _git(repository, "add", "--", ".")
    _git(repository, "commit", "-m", "freeze runner and worker")
    monkeypatch.setattr(runner, "_RUNNER_PATH", runner_path.resolve(strict=True))
    monkeypatch.setattr(runner, "_WORKER_PATH", worker_path.resolve(strict=True))
    return repository.resolve(strict=True)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_inputs(
    tmp_path: Path,
    records: tuple[UnlabeledVideoRecord, ...],
) -> tuple[Path, Path, Path]:
    video_root = tmp_path / "videos"
    video_root.mkdir(exist_ok=True)
    manifest = PoseInputManifest(
        protocol="ucfrep_526",
        split="dev",
        records=records,
    )
    sidecar = tmp_path / "inputs.json"
    sidecar.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
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
        json.dumps(commitment.to_dict()),
        encoding="utf-8",
    )
    return sidecar, commitment_path, video_root


class _FrozenSource:
    commit = runner.OFFICIAL_SOURCE_COMMIT
    tree = runner.OFFICIAL_SOURCE_TREE

    def assert_unchanged(self) -> None:
        return

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository": runner.OFFICIAL_SOURCE_REPOSITORY,
            "commit": runner.OFFICIAL_SOURCE_COMMIT,
            "tree": runner.OFFICIAL_SOURCE_TREE,
            "license": runner.OFFICIAL_SOURCE_LICENSE,
            "files": [
                {
                    "relative_path": item.relative_path,
                    "bytes": item.byte_count,
                    "sha256": item.sha256,
                }
                for item in runner.OFFICIAL_SOURCE_FILES
            ],
            "pytorchvideo_repository": runner.PYTORCHVIDEO_REPOSITORY,
            "pytorchvideo_commit": runner.PYTORCHVIDEO_COMMIT,
            "pytorchvideo_tree": runner.PYTORCHVIDEO_TREE,
        }


class _FrozenAssets:
    def assert_unchanged(self) -> None:
        return

    def to_dict(self) -> dict[str, Any]:
        files = [
            runner.OFFICIAL_ENCODER.public_dict(),
            runner.OFFICIAL_DECODER.public_dict(),
        ]
        return {
            "aggregate_sha256": runner.sha256_json({"files": files}),
            "files": files,
        }


class _FakeBackend:
    source_commit = runner.OFFICIAL_SOURCE_COMMIT
    source_tree = runner.OFFICIAL_SOURCE_TREE
    _BASE_RUNTIME_VERSIONS = {
        "python": sys.version.split()[0],
        "torch": "fixture",
        "torchvision": "fixture",
        "cuda": "fixture",
        "numpy": "fixture",
        "opencv": "fixture",
        "av": "fixture",
        "container_image_id": f"sha256:{'9' * 64}",
        "pytorchvideo_commit": runner.PYTORCHVIDEO_COMMIT,
        "worker_command_sha256": "2" * 64,
        "module_origins_sha256": "3" * 64,
        "pytorchvideo_origin_sha256": "4" * 64,
        "encoder_just_encode_unused_parameters_json": (
            runner.ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_JSON
        ),
        "encoder_just_encode_unused_parameters_sha256": (
            runner.ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_SHA256
        ),
    }

    def __init__(self, *, oom_name: str | None = None) -> None:
        self.oom_name = oom_name
        self.runtime_versions = {
            **self._BASE_RUNTIME_VERSIONS,
            "worker_code_sha256": runner._stable_file_digest(runner._WORKER_PATH).sha256,
        }

    def predict(
        self,
        video_path: Path,
        config: runner.ESCountsOfficialConfig,
    ) -> runner.BackendPrediction:
        assert config is runner.FROZEN_ESCOUNTS_OFFICIAL_CONFIG
        if video_path.name == self.oom_name:
            raise runner.ESCountsResourceExhaustedError("fixture CUDA OOM")
        if video_path.name == "decode.mp4":
            raise runner.ESCountsDecodeError("fixture decode failure")
        return runner.BackendPrediction(raw_count=4.5, decoded_frames=32)


def test_module_import_does_not_import_torch() -> None:
    code = (
        "import sys; "
        "assert 'torch' not in sys.modules; "
        "import pams.baselines.escounts_official_runner; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )


def test_worker_source_is_python38_parseable_and_does_not_import_pams() -> None:
    worker = REPOSITORY / "src/pams/baselines/escounts_official_worker.py"
    source = worker.read_text(encoding="utf-8")
    ast.parse(source, filename=str(worker), feature_version=(3, 8))
    assert "import pams" not in source


def test_worker_accepts_and_records_exact_just_encode_unused_allowlist() -> None:
    observed = [
        {
            "name": item["name"],
            "shape": tuple(item["shape"]),
            "dtype": item["dtype"],
            "just_encode_unused": item["just_encode_unused"],
        }
        for item in reversed(worker.ENCODER_JUST_ENCODE_UNUSED_PARAMETERS)
    ]
    audit = worker.validate_encoder_just_encode_unmatched(observed)
    assert (
        audit["parameters_json"]
        == runner.ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_JSON
        == worker.ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_JSON
    )
    assert (
        audit["parameters_sha256"]
        == runner.ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_SHA256
    )
    assert json.loads(audit["parameters_json"]) == list(
        runner.ENCODER_JUST_ENCODE_UNUSED_PARAMETERS
    )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("name", "unexpected.weight"),
        ("shape", [2, 512]),
        ("dtype", "float16"),
    ),
)
def test_worker_rejects_just_encode_allowlist_descriptor_drift(
    field: str,
    replacement: object,
) -> None:
    observed = [
        dict(item) for item in worker.ENCODER_JUST_ENCODE_UNUSED_PARAMETERS
    ]
    observed[0][field] = replacement
    with pytest.raises(RuntimeError, match="exact just_encode-unused allowlist"):
        worker.validate_encoder_just_encode_unmatched(observed)


def test_worker_rejects_fifth_unmatched_encoder_tensor() -> None:
    observed = [
        dict(item) for item in worker.ENCODER_JUST_ENCODE_UNUSED_PARAMETERS
    ]
    observed.append(
        {
            "name": "fifth.weight",
            "shape": [1],
            "dtype": "float32",
            "just_encode_unused": True,
        }
    )
    with pytest.raises(RuntimeError, match="exact just_encode-unused allowlist"):
        worker.validate_encoder_just_encode_unmatched(observed)


@pytest.mark.parametrize(
    "arguments",
    (
        (
            "predict",
            "--sidecar",
            "sidecar.json",
            "--commitment",
            "commitment.json",
            "--video-root",
            "videos",
            "--encoder",
            "encoder.pyth",
            "--decoder",
            "decoder.pth",
            "--official-source-root",
            "official",
            "--pytorchvideo-source-root",
            "pytorchvideo",
            "--worker-command-json",
            '["worker"]',
            "--expected-container-image-id",
            f"sha256:{'1' * 64}",
            "--output",
            "primary.json",
        ),
        (
            "create-retry-request",
            "--primary-predictions",
            "primary.json",
            "--output",
            "request.json",
        ),
        (
            "merge",
            "--primary-predictions",
            "primary.json",
            "--retry-request",
            "request.json",
            "--retry-predictions",
            "retry.json",
            "--output",
            "merged.json",
            "--receipt",
            "receipt.json",
        ),
    ),
)
def test_runner_cli_commands_require_repository_root(
    arguments: tuple[str, ...],
) -> None:
    with pytest.raises(SystemExit):
        runner._build_parser().parse_args(arguments)


def test_jsonl_backend_reuses_fake_subprocess_and_binds_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audited_repository: Path,
) -> None:
    content = b"video"
    records = (UnlabeledVideoRecord("v", "v.mp4", _sha256(content)),)
    sidecar, commitment, video_root = _write_inputs(tmp_path, records)
    video = video_root / "v.mp4"
    video.write_bytes(content)
    inputs = runner._load_label_free_inputs(
        sidecar,
        commitment,
        video_root,
        require_exact_membership=False,
    )
    encoder = tmp_path / "encoder.pyth"
    decoder = tmp_path / "decoder.pyth"
    encoder.write_bytes(b"encoder")
    decoder.write_bytes(b"decoder")
    pytorchvideo_root = tmp_path / "pytorchvideo"
    pytorchvideo_root.mkdir()
    image_id = f"sha256:{'a' * 64}"
    exchanges: list[dict[str, Any]] = []
    repository = runner.verify_runner_repository(audited_repository)

    class FakeStdout:
        def __init__(self) -> None:
            self.lines: list[str] = []

        def readline(self) -> str:
            return self.lines.pop(0) if self.lines else ""

    class FakeStdin:
        def __init__(self, process: FakeProcess) -> None:
            self.process = process
            self.pending = ""

        def write(self, value: str) -> int:
            self.pending += value
            return len(value)

        def flush(self) -> None:
            request = json.loads(self.pending)
            self.pending = ""
            supplied = request.pop("message_sha256")
            assert supplied == runner.sha256_json(request)
            exchanges.append(request)
            if request["type"] == "init":
                response = {
                    "schema_version": 1,
                    "type": "ready",
                    "nonce": request["nonce"],
                    "worker_sha256": request["worker_sha256"],
                    "container_image_id": image_id,
                    "source_commit": runner.OFFICIAL_SOURCE_COMMIT,
                    "source_tree": runner.OFFICIAL_SOURCE_TREE,
                    "pytorchvideo_origin": {
                        "repository_commit": runner.PYTORCHVIDEO_COMMIT,
                        "repository_tree": runner.PYTORCHVIDEO_TREE,
                    },
                    "module_origins": {
                        "official_model": {"sha256": "1" * 64},
                        "slowfast_parser": {"sha256": "2" * 64},
                        "slowfast_loaded_modules": {
                            "module_count": 4,
                            "sha256": "3" * 64,
                        },
                    },
                    "runtime_versions": {
                        "python": "3.8.13",
                        "torch": "1.10.0",
                        "container_image_id": image_id,
                        "encoder_just_encode_unused_parameters_json": (
                            runner.ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_JSON
                        ),
                        "encoder_just_encode_unused_parameters_sha256": (
                            runner.ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_SHA256
                        ),
                    },
                }
            elif request["type"] == "predict":
                response = {
                    "schema_version": 1,
                    "type": "result",
                    "request_id": request["request_id"],
                    "video_id": request["video_id"],
                    "video_sha256": request["video_sha256"],
                    "status": "ok",
                    "raw_count": 4.25,
                    "decoded_frames": 32,
                    "error_type": None,
                    "error_message": None,
                }
            else:
                response = {
                    "schema_version": 1,
                    "type": "shutdown_ack",
                    "nonce": request["nonce"],
                    "worker_sha256": runner._stable_file_digest(
                        runner._WORKER_PATH.resolve(strict=True)
                    ).sha256,
                }
            response["message_sha256"] = runner.sha256_json(response)
            self.process.stdout.lines.append(json.dumps(response) + "\n")

        def close(self) -> None:
            return

    class FakeProcess:
        def __init__(self, command: list[str], **_kwargs: Any) -> None:
            self.command = command
            self.stdout = FakeStdout()
            self.stdin = FakeStdin(self)
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: int) -> int:
            assert timeout > 0
            self.returncode = 0
            return 0

        def terminate(self) -> None:
            self.returncode = -15

        def kill(self) -> None:
            self.returncode = -9

    processes: list[FakeProcess] = []

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        process = FakeProcess(command, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)

    def fake_git_source_state(root: Path) -> tuple[str, str, str]:
        if root == pytorchvideo_root:
            return (
                runner.PYTORCHVIDEO_COMMIT,
                runner.PYTORCHVIDEO_TREE,
                "",
            )
        if root == repository.root:
            return (repository.git_sha, repository.git_tree, "")
        raise AssertionError(f"unexpected Git checkout: {root}")

    monkeypatch.setattr(runner, "_git_source_state", fake_git_source_state)
    source = SimpleNamespace(
        root=tmp_path,
        commit=runner.OFFICIAL_SOURCE_COMMIT,
        tree=runner.OFFICIAL_SOURCE_TREE,
    )
    assets = SimpleNamespace(
        encoder=SimpleNamespace(path=encoder),
        decoder=SimpleNamespace(path=decoder),
    )
    backend = runner.JSONLWorkerBackend(
        source,  # type: ignore[arg-type]
        assets,  # type: ignore[arg-type]
        inputs,
        repository,
        worker_command=["fake-worker"],
        pytorchvideo_source_root=pytorchvideo_root,
        expected_container_image_id=image_id,
        device="cuda:0",
        memory_limit_bytes=runner.PRIMARY_MEMORY_LIMIT_BYTES,
    )
    first = backend.predict(video.resolve(), runner.FROZEN_ESCOUNTS_OFFICIAL_CONFIG)
    second = backend.predict(video.resolve(), runner.FROZEN_ESCOUNTS_OFFICIAL_CONFIG)
    backend.close()
    assert first.raw_count == second.raw_count == 4.25
    assert len(processes) == 1
    assert [item["type"] for item in exchanges] == [
        "init",
        "predict",
        "predict",
        "shutdown",
    ]
    assert (
        backend.runtime_versions[
            "encoder_just_encode_unused_parameters_sha256"
        ]
        == runner.ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_SHA256
    )


@pytest.mark.parametrize(
    "locator",
    ("../escape.mp4", "nested/../escape.mp4", r"C:\video.mp4", r"a\b.mp4"),
)
def test_portable_locator_rejects_traversal_and_windows_paths(locator: str) -> None:
    with pytest.raises(ValueError, match="relative|unsafe|portable"):
        runner._validate_portable_locator(locator, field="fixture")


def test_full_rehash_detects_in_place_change_with_restored_mtime(
    tmp_path: Path,
) -> None:
    path = tmp_path / "critical.bin"
    path.write_bytes(b"AAAA")
    digest = runner._stable_file_digest(path)
    before = path.stat()
    path.write_bytes(b"BBBB")
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(RuntimeError, match="changed"):
        runner._assert_file_unchanged(path, digest, role="critical fixture")


def test_runner_repository_rejects_dirty_worker_checkout(
    audited_repository: Path,
) -> None:
    verified = runner.verify_runner_repository(audited_repository)
    worker = runner._WORKER_PATH
    original = worker.read_bytes()
    worker.write_bytes(original + b"\n# dirty worker\n")
    with pytest.raises(runner.ESCountsSourceError, match="dirty|changed"):
        verified.assert_unchanged()
    with pytest.raises(runner.ESCountsSourceError, match="clean"):
        runner.verify_runner_repository(audited_repository)


def test_label_and_commitment_gates_precede_source_and_assets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"video"
    records = (UnlabeledVideoRecord("v", "v.mp4", _sha256(content)),)
    sidecar, commitment, video_root = _write_inputs(tmp_path, records)
    (video_root / "v.mp4").write_bytes(content)
    commitment_payload = json.loads(commitment.read_text(encoding="utf-8"))
    commitment_payload["sidecar_sha256"] = "0" * 64
    commitment.write_text(json.dumps(commitment_payload), encoding="utf-8")
    source_calls = 0

    def forbidden_source(_path: object) -> None:
        nonlocal source_calls
        source_calls += 1
        raise AssertionError("source verification must follow input binding")

    monkeypatch.setattr(runner, "verify_official_source", forbidden_source)
    with pytest.raises(ValueError, match="bind"):
        runner.run_escounts_official(
            sidecar,
            commitment,
            video_root,
            tmp_path / "encoder",
            tmp_path / "decoder",
            tmp_path / "source",
            repository_root=tmp_path / "repository-not-opened",
            require_exact_membership=False,
        )
    assert source_calls == 0

    sidecar.write_text(
        '{"schema_version":2,"manifest_type":"pose_inputs",'
        '"protocol":"ucfrep_526","split":"dev","records":[],'
        '"label":NaN}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden label field 'label'"):
        runner._load_label_free_inputs(
            sidecar,
            commitment,
            video_root,
            require_exact_membership=False,
        )


def test_failure_ledger_has_no_synthetic_counts_and_is_exclusive(
    tmp_path: Path,
    audited_repository: Path,
) -> None:
    ok = b"ok"
    decode = b"decode"
    records = (
        UnlabeledVideoRecord("ok", "ok.mp4", _sha256(ok)),
        UnlabeledVideoRecord("decode", "decode.mp4", _sha256(decode)),
        UnlabeledVideoRecord("missing", "missing.mp4", "f" * 64),
    )
    sidecar, commitment, video_root = _write_inputs(tmp_path, records)
    (video_root / "ok.mp4").write_bytes(ok)
    (video_root / "decode.mp4").write_bytes(decode)
    inputs = runner._load_label_free_inputs(
        sidecar,
        commitment,
        video_root,
        require_exact_membership=False,
    )
    repository = runner.verify_runner_repository(audited_repository)
    result = runner._run_verified_backend(
        inputs,
        _FrozenSource(),  # type: ignore[arg-type]
        _FrozenAssets(),  # type: ignore[arg-type]
        _FakeBackend(),
        repository,
    )
    payload = result.to_dict()
    assert payload["success_total"] == 1
    assert [row["status"] for row in payload["failures"]] == [
        "decode_failed",
        "missing_video",
    ]
    assert all(row["raw_count"] is None for row in payload["failures"])
    assert payload["target_access"] is False
    assert payload["runner_source_git_sha"] == repository.git_sha
    assert payload["runner_code_sha256"] == repository.runner_digest.sha256
    assert payload["worker_code_sha256"] == repository.worker_digest.sha256

    output = tmp_path / "predictions.json"
    runner.write_escounts_result_exclusive(output, result)
    with pytest.raises(FileExistsError, match="overwrite"):
        runner.write_escounts_result_exclusive(output, result)


def test_retry_is_bounded_and_merge_preserves_primary_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    audited_repository: Path,
) -> None:
    contents = {"a.mp4": b"a", "oom.mp4": b"oom", "b.mp4": b"b"}
    records = tuple(
        UnlabeledVideoRecord(name.removesuffix(".mp4"), name, _sha256(content))
        for name, content in contents.items()
    )
    sidecar, commitment, video_root = _write_inputs(tmp_path, records)
    for name, content in contents.items():
        (video_root / name).write_bytes(content)
    inputs = runner._load_label_free_inputs(
        sidecar,
        commitment,
        video_root,
        require_exact_membership=False,
    )
    source = _FrozenSource()
    assets = _FrozenAssets()
    repository = runner.verify_runner_repository(audited_repository)
    primary_result = runner._run_verified_backend(
        inputs,
        source,  # type: ignore[arg-type]
        assets,  # type: ignore[arg-type]
        _FakeBackend(oom_name="oom.mp4"),
        repository,
    )
    primary_path = tmp_path / "primary.json"
    runner.write_escounts_result_exclusive(primary_path, primary_result)
    request_path = tmp_path / "retry.request.json"
    request_receipt = runner.create_retry_request(
        primary_predictions_path=primary_path,
        output_path=request_path,
        repository_root=audited_repository,
    )
    assert request_receipt["video_ids"] == ["oom"]
    request = json.loads(request_path.read_text(encoding="utf-8"))
    retry_binding = {
        "retry_request_sha256": request_receipt["retry_request_sha256"],
        "source_primary_sha256": request_receipt["source_primary_sha256"],
        "primary_success_rows_sha256": request["primary_success_rows_sha256"],
        "records_sha256": request["records_sha256"],
    }
    retry_result = runner._run_verified_backend(
        inputs,
        source,  # type: ignore[arg-type]
        assets,  # type: ignore[arg-type]
        _FakeBackend(),
        repository,
        video_ids=("oom",),
        resource_tier=runner.RETRY_RESOURCE_TIER,
        retry_binding=retry_binding,
    )
    retry_path = tmp_path / "retry.json"
    runner.write_escounts_result_exclusive(retry_path, retry_result)
    merged_path = tmp_path / "merged.json"
    merge_receipt = tmp_path / "merge.receipt.json"
    result = runner.merge_retry_artifacts(
        primary_predictions_path=primary_path,
        retry_request_path=request_path,
        retry_predictions_path=retry_path,
        output_path=merged_path,
        receipt_path=merge_receipt,
        repository_root=audited_repository,
    )
    merged = json.loads(merged_path.read_text(encoding="utf-8"))
    primary = json.loads(primary_path.read_text(encoding="utf-8"))
    assert result["primary_rows_canonical_value_identical"] is True
    assert merged["primary_success_rows_sha256"] == runner.sha256_json(primary["predictions"])

    tampered = json.loads(retry_path.read_text(encoding="utf-8"))
    tampered["retry_binding"]["records_sha256"] = "0" * 64
    tampered_path = tmp_path / "retry.tampered.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(runner.ESCountsMergeError, match="bound"):
        runner.merge_retry_artifacts(
            primary_predictions_path=primary_path,
            retry_request_path=request_path,
            retry_predictions_path=tampered_path,
            output_path=tmp_path / "bad.merged.json",
            receipt_path=tmp_path / "bad.receipt.json",
            repository_root=audited_repository,
        )

    worker_tampered = json.loads(primary_path.read_text(encoding="utf-8"))
    worker_tampered["worker_code_sha256"] = "0" * 64
    worker_tampered["runtime_versions"]["worker_code_sha256"] = "0" * 64
    worker_tampered_path = tmp_path / "primary.worker-tampered.json"
    worker_tampered_path.write_text(json.dumps(worker_tampered), encoding="utf-8")
    with pytest.raises(runner.ESCountsSourceError, match="current clean"):
        runner.create_retry_request(
            primary_predictions_path=worker_tampered_path,
            output_path=tmp_path / "worker-tampered.request.json",
            repository_root=audited_repository,
        )

    original_writer = runner._write_json_exclusive
    writes = 0

    def fail_second_write(path: Path, payload: dict[str, Any]) -> Any:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise OSError("fixture receipt write failure")
        return original_writer(path, payload)

    monkeypatch.setattr(runner, "_write_json_exclusive", fail_second_write)
    orphan = tmp_path / "transaction.merged.json"
    with pytest.raises(OSError, match="receipt write"):
        runner.merge_retry_artifacts(
            primary_predictions_path=primary_path,
            retry_request_path=request_path,
            retry_predictions_path=retry_path,
            output_path=orphan,
            receipt_path=tmp_path / "transaction.receipt.json",
            repository_root=audited_repository,
        )
    assert not orphan.exists()

    marker = audited_repository / "commit-marker.txt"
    marker.write_text("second clean commit\n", encoding="utf-8")
    _git(audited_repository, "add", "--", marker.name)
    _git(audited_repository, "commit", "-m", "advance provenance only")
    with pytest.raises(runner.ESCountsSourceError, match="current clean"):
        runner.merge_retry_artifacts(
            primary_predictions_path=primary_path,
            retry_request_path=request_path,
            retry_predictions_path=retry_path,
            output_path=tmp_path / "cross-commit.merged.json",
            receipt_path=tmp_path / "cross-commit.receipt.json",
            repository_root=audited_repository,
        )


def test_prediction_artifact_label_key_is_rejected_before_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "escounts-label-firewall.json"
    path.write_text('{"label":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden label field"):
        runner.load_prediction_artifact(path)
