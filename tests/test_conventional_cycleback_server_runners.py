from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "scripts/server/pams_conventional_cycleback_secure_launcher.py"
CONTRACT_PATH = ROOT / "scripts/server/pams_conventional_cycleback_wrapper_contract.py"
SEED_VALIDATOR_PATH = (
    ROOT / "scripts/server/validate_pams_conventional_cycleback_mechanism_seed.py"
)
GEOMETRY = ROOT / "scripts/server/run_pams_conventional_cycleback_geometry_gate_v1.sh"
MECHANISM = ROOT / "scripts/server/run_pams_conventional_cycleback_mechanism_probe_v1.sh"
ADAPTER = ROOT / "scripts/server/run_pams_conventional_cycleback_pose_input_authorization_v1.sh"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load test module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _contract_fixture(tmp_path: Path) -> tuple[ModuleType, Path, Path, Path, dict]:
    module = _load_module("cycleback_wrapper_contract_test", CONTRACT_PATH)
    container_id = "a" * 64
    create_id = tmp_path / "create-id.txt"
    create_id.write_text(container_id + "\n", encoding="ascii")
    spec = {
        "schema_version": 2,
        "stage": "geometry",
        "container_name": "pams-cycleback-geometry-deadbeefcafe-probe",
        "image_id": "sha256:" + "b" * 64,
        "entrypoint": ["python"],
        "path": "python",
        "args": ["-I", "runner.py", "--output", "/pams/output/result.json"],
        "user": "1000:1000",
        "working_dir": "/workspace",
        "expected_environment": {
            "LANG": "C.UTF-8",
            "PAMS_CONTAINER_SOURCE_REVISION": "c" * 40,
        },
        "forbidden_environment": ["PYTHONOPTIMIZE", "PAMS_DEV_ROOT"],
        "gpu_device": "1",
        "allowed_exit_codes": [0, 3],
        "host_config_exact": {
            "NetworkMode": "none",
            "ReadonlyRootfs": True,
            "CapDrop": ["ALL"],
            "SecurityOpt": ["no-new-privileges:true"],
            "PidsLimit": 4096,
            "Memory": 1024,
            "NanoCpus": 2_000_000_000,
            "IpcMode": "private",
            "PidMode": "",
            "UsernsMode": "",
            "Tmpfs": {"/tmp": "rw,noexec"},
            "AutoRemove": False,
        },
        "mounts": [
            {
                "destination": "/workspace",
                "type": "bind",
                "source": "/sealed/source",
                "rw": False,
                "propagation": "rprivate",
            }
        ],
    }
    contract = tmp_path / "contract.json"
    _write_json(contract, spec)
    inspect = [
        {
            "Id": "sha256:" + container_id,
            "Name": "/" + spec["container_name"],
            "Image": spec["image_id"],
            "Path": spec["path"],
            "Args": spec["args"],
            "Config": {
                "Image": spec["image_id"],
                "Entrypoint": spec["entrypoint"],
                "Cmd": spec["args"],
                "User": spec["user"],
                "WorkingDir": spec["working_dir"],
                "ExposedPorts": None,
                "Volumes": None,
                "Env": [f"{key}={value}" for key, value in spec["expected_environment"].items()],
            },
            "HostConfig": {
                **spec["host_config_exact"],
                "CapAdd": None,
                "Devices": [],
                "Binds": None,
                "VolumesFrom": None,
                "PortBindings": {},
                "Dns": [],
                "DnsOptions": [],
                "DnsSearch": [],
                "ExtraHosts": None,
                "Links": None,
                "Privileged": False,
                "PublishAllPorts": False,
                "RestartPolicy": {"Name": "no", "MaximumRetryCount": 0},
                "DeviceRequests": [
                    {
                        "Driver": "",
                        "Count": 0,
                        "DeviceIDs": ["1"],
                        "Capabilities": [["gpu"]],
                        "Options": {},
                    }
                ],
            },
            "Mounts": [
                {
                    "Destination": "/workspace",
                    "Type": "bind",
                    "Source": "/sealed/source",
                    "RW": False,
                    "Propagation": "rprivate",
                }
            ],
            "NetworkSettings": {"Ports": {}},
            "State": {
                "Status": "created",
                "Running": False,
                "OOMKilled": False,
                "Error": "",
                "Pid": 0,
                "ExitCode": 0,
            },
        }
    ]
    inspect_path = tmp_path / "inspect.json"
    _write_json(inspect_path, inspect)
    return module, contract, create_id, inspect_path, inspect[0]


def test_exact_wrapper_contract_accepts_only_the_frozen_pre_spec(tmp_path: Path) -> None:
    module, contract, create_id, inspect_path, _ = _contract_fixture(tmp_path)

    module.validate_inspect(
        inspect_path=inspect_path,
        create_id_path=create_id,
        spec_path=contract,
        phase="pre",
        expected_exit_code=None,
    )


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("Config", "Entrypoint", ["/evil-entrypoint"]),
        ("top", "Path", "/evil-entrypoint"),
        ("top", "Args", ["--skip-contract"]),
        ("HostConfig", "Privileged", True),
        ("HostConfig", "Binds", ["/host:/escape"]),
        ("HostConfig", "NetworkMode", "host"),
        ("State", "OOMKilled", True),
        ("State", "Error", "injected"),
        ("State", "Pid", 999),
    ],
)
def test_exact_wrapper_contract_rejects_mutated_inspect(
    tmp_path: Path,
    section: str,
    key: str,
    value: object,
) -> None:
    module, contract, create_id, inspect_path, item = _contract_fixture(tmp_path)
    mutated = copy.deepcopy(item)
    if section == "top":
        mutated[key] = value
    else:
        mutated[section][key] = value
    _write_json(inspect_path, [mutated])

    with pytest.raises(ValueError):
        module.validate_inspect(
            inspect_path=inspect_path,
            create_id_path=create_id,
            spec_path=contract,
            phase="pre",
            expected_exit_code=None,
        )


def test_exact_wrapper_contract_rejects_mount_environment_and_post_exit_mutations(
    tmp_path: Path,
) -> None:
    module, contract, create_id, inspect_path, item = _contract_fixture(tmp_path)
    for mutate in (
        lambda value: value["Mounts"][0].update(Source="/forged/source"),
        lambda value: value["Config"]["Env"].append("PYTHONOPTIMIZE=1"),
        lambda value: value["HostConfig"].update(DeviceRequests=[]),
    ):
        mutated = copy.deepcopy(item)
        mutate(mutated)
        _write_json(inspect_path, [mutated])
        with pytest.raises(ValueError):
            module.validate_inspect(
                inspect_path=inspect_path,
                create_id_path=create_id,
                spec_path=contract,
                phase="pre",
                expected_exit_code=None,
            )

    post = copy.deepcopy(item)
    post["State"].update(Status="exited", ExitCode=3)
    _write_json(inspect_path, [post])
    module.validate_inspect(
        inspect_path=inspect_path,
        create_id_path=create_id,
        spec_path=contract,
        phase="post",
        expected_exit_code=3,
    )
    post["State"]["ExitCode"] = 0
    _write_json(inspect_path, [post])
    with pytest.raises(ValueError, match="state/exit"):
        module.validate_inspect(
            inspect_path=inspect_path,
            create_id_path=create_id,
            spec_path=contract,
            phase="post",
            expected_exit_code=3,
        )


def test_exact_id_cleanup_failure_is_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_module("cycleback_secure_launcher_cleanup_test", LAUNCHER_PATH)
    container_id = "sha256:" + "d" * 64
    calls: list[list[str]] = []

    def fake_run(arguments, **kwargs):
        del kwargs
        calls.append(list(arguments))
        if arguments[1] == "inspect":
            return subprocess.CompletedProcess(
                arguments,
                0,
                stdout=json.dumps([{"Id": container_id}]).encode("utf-8"),
                stderr=b"",
            )
        return subprocess.CompletedProcess(arguments, 1, stdout=b"", stderr=b"failed")

    monkeypatch.setattr(launcher, "_run", fake_run)
    with pytest.raises(launcher.LaunchFailure, match="removal failed"):
        launcher._cleanup_exact_container(container_id)

    assert calls[1] == [launcher.DOCKER_BINARY, "rm", "-f", container_id]
    assert len(calls) == 2


def test_docker_create_uses_a_dedicated_cidfile(tmp_path: Path) -> None:
    launcher = _load_module("cycleback_secure_launcher_cidfile_test", LAUNCHER_PATH)
    cidfile = tmp_path / "created.cid"
    arguments = launcher._docker_create_arguments(
        stage="adapter",
        container_name="pams-cycleback-pose-auth-deadbeefcafe-attempt",
        cidfile_path=cidfile,
        launch=SimpleNamespace(container_image_id="sha256:" + "a" * 64),
        command=("-I", "runner.py"),
        mounts=(),
        required_environment={},
        gpu_device=None,
    )

    position = arguments.index("--cidfile")
    assert arguments[position + 1] == str(cidfile)
    assert arguments[-3:] == ["sha256:" + "a" * 64, "-I", "runner.py"]


def test_exact_id_cleanup_requires_proven_absence(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = _load_module("cycleback_secure_launcher_absence_test", LAUNCHER_PATH)
    raw_container_id = "e" * 64
    container_id = "sha256:" + raw_container_id
    responses = iter(
        (
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps([{"Id": container_id}]).encode("utf-8"),
                stderr=b"",
            ),
            subprocess.CompletedProcess([], 0, stdout=raw_container_id.encode(), stderr=b""),
            subprocess.CompletedProcess(
                [],
                1,
                stdout=b"",
                stderr=b"error during connect: Docker daemon unavailable",
            ),
        )
    )
    monkeypatch.setattr(launcher, "_run", lambda *args, **kwargs: next(responses))

    with pytest.raises(launcher.LaunchFailure, match="absence is not proven"):
        launcher._cleanup_exact_container(container_id)


def test_exact_id_cleanup_accepts_only_exact_removed_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_module("cycleback_secure_launcher_cleanup_success_test", LAUNCHER_PATH)
    raw_container_id = "f" * 64
    container_id = "sha256:" + raw_container_id
    responses = iter(
        (
            subprocess.CompletedProcess(
                [],
                0,
                stdout=json.dumps([{"Id": container_id}]).encode("utf-8"),
                stderr=b"",
            ),
            subprocess.CompletedProcess(
                [],
                0,
                stdout=(raw_container_id + "\n").encode("ascii"),
                stderr=b"",
            ),
            subprocess.CompletedProcess(
                [],
                1,
                stdout=b"",
                stderr=b"Error: No such object: " + raw_container_id.encode("ascii"),
            ),
        )
    )
    monkeypatch.setattr(launcher, "_run", lambda *args, **kwargs: next(responses))

    cleanup = launcher._cleanup_exact_container(container_id)

    assert cleanup["removed_by_exact_id"] == container_id
    assert cleanup["absence_verified"] is True


def test_unverified_cleanup_cannot_publish_or_seal_staging(tmp_path: Path) -> None:
    launcher = _load_module("cycleback_secure_launcher_failure_test", LAUNCHER_PATH)
    staging = tmp_path / ".incomplete-run"
    final = tmp_path / "run"
    staging.mkdir()
    (staging / "partial.txt").write_text("partial", encoding="utf-8")

    launcher._write_failure_and_publish(
        stage="geometry",
        staging=staging,
        final_root=final,
        reservation_path=None,
        payload={"schema_version": 1, "status": "failed"},
        container_absence_verified=False,
    )

    assert staging.is_dir()
    assert not final.exists()
    fallback = tmp_path / "run.failure.receipt.json"
    assert fallback.is_file()
    payload = json.loads(fallback.read_text(encoding="utf-8"))
    assert payload["staging_publication_forbidden_while_container_may_exist"] is True


def test_primary_failure_receipt_error_keeps_staging_unpublished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_module("cycleback_secure_launcher_primary_failure_test", LAUNCHER_PATH)
    staging = tmp_path / ".incomplete-run"
    final = tmp_path / "run"
    audit = staging / "audit"
    audit.mkdir(parents=True)
    (staging / "partial.txt").write_text("partial", encoding="utf-8")
    original = launcher._write_json_exclusive

    def fail_primary(path, payload, **kwargs):
        if path == audit / "failure.receipt.json":
            raise OSError("injected primary failure")
        return original(path, payload, **kwargs)

    monkeypatch.setattr(launcher, "_write_json_exclusive", fail_primary)
    launcher._write_failure_and_publish(
        stage="geometry",
        staging=staging,
        final_root=final,
        reservation_path=None,
        payload={"schema_version": 1, "status": "failed"},
        container_absence_verified=True,
    )

    assert staging.is_dir()
    assert not final.exists()
    fallback = tmp_path / "run.failure.receipt.json"
    payload = json.loads(fallback.read_text(encoding="utf-8"))
    assert payload["staging_publication_forbidden_after_primary_write_failure"] is True


def test_lock_rejects_symlink_parent_and_symlink_file(tmp_path: Path) -> None:
    launcher = _load_module("cycleback_secure_launcher_lock_test", LAUNCHER_PATH)
    real = tmp_path / "real"
    real.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real, target_is_directory=True)

    with pytest.raises(launcher.LaunchFailure, match="safe directory"):
        launcher._open_lock(linked_parent / "gpu1.lock")

    real_lock = real / "real.lock"
    real_lock.write_text("", encoding="utf-8")
    linked_lock = real / "linked.lock"
    linked_lock.symlink_to(real_lock)
    with pytest.raises((OSError, launcher.LaunchFailure)):
        launcher._open_lock(linked_lock)


def test_stage_reservation_releases_lock_on_failure(tmp_path: Path) -> None:
    launcher = _load_module("cycleback_secure_launcher_reservation_lock_test", LAUNCHER_PATH)
    launcher.CANONICAL_ROOT = tmp_path
    launch = SimpleNamespace(
        source_revision="a" * 40,
        source_tree_sha="b" * 64,
        registry_id="c" * 64,
        authorization_sha256="d" * 64,
        receipt_sha256="e" * 64,
        container_image_id="sha256:" + "f" * 64,
    )
    run_parent = tmp_path / launcher.STAGE_METADATA["geometry"]["run_parent"]
    run_parent.mkdir(parents=True)
    run_key = f"{launch.source_revision[:12]}-attempt"
    (run_parent / run_key).mkdir()

    with pytest.raises(launcher.LaunchFailure, match="already exists"):
        launcher._reserve_stage(
            stage="geometry",
            attempt_id="attempt",
            candidate_id="W16_H4",
            launch=launch,
            config_identity=None,
            predecessors=(),
        )

    descriptor = launcher._open_lock(run_parent / f".{run_key}.launch.lock")
    os.close(descriptor)


def test_empty_bootstrap_rejects_before_registry_checkout_or_dynamic_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_module("cycleback_secure_launcher_bootstrap_test", LAUNCHER_PATH)
    forged_registry = tmp_path / "forged-registry"
    forged_registry.mkdir()
    _write_json(
        forged_registry / "launch.authorization.json",
        {"identity": {"source_revision": "a" * 40, "source_tree_sha": "b" * 40}},
    )
    alternate_checkout = tmp_path / "clean-alternate-checkout"
    alternate_checkout.mkdir()
    launcher.LAUNCH_REGISTRY_ROOT = forged_registry
    before_sys_path = tuple(sys.path)
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        del args, kwargs
        calls.append("forbidden")
        raise AssertionError("bootstrap touched untrusted state before activation")

    monkeypatch.setattr(launcher, "_stable_bytes", forbidden)
    monkeypatch.setattr(launcher, "_validate_directory", forbidden)
    monkeypatch.setattr(launcher, "_git_text", forbidden)
    monkeypatch.setattr(launcher, "_run", forbidden)
    monkeypatch.setattr(launcher, "_write_failure_and_publish", forbidden)
    monkeypatch.setattr(launcher.importlib.util, "spec_from_file_location", forbidden)

    with pytest.raises(launcher.LaunchFailure, match="disabled pending"):
        launcher._load_launch_bootstrap()
    with pytest.raises(launcher.LaunchFailure, match="disabled pending"):
        launcher.launch_stage(
            stage="adapter",
            attempt_id="attempt",
            candidate_id=None,
            gpu_device=None,
        )
    with pytest.raises(TypeError):
        launcher._load_launch_bootstrap(alternate_checkout)

    assert calls == []
    assert tuple(sys.path) == before_sys_path
    assert launcher._BOOTSTRAP_SOURCE_CHECKOUT_LOCATOR == ""
    assert launcher._BOOTSTRAP_SOURCE_REVISION == ""
    assert launcher._BOOTSTRAP_SOURCE_TREE_SHA == ""
    assert launcher._BOOTSTRAP_LAUNCH_AUTHORIZATION_SHA256 == ""
    assert launcher._BOOTSTRAP_INTEGRATION_OUTCOME_LOCATOR == ""
    assert launcher._BOOTSTRAP_INTEGRATION_AUTHORIZATION_SHA256 == ""


def test_mechanism_command_and_output_bind_pass_only_seed_checkpoint(
    tmp_path: Path,
) -> None:
    launcher = _load_module("cycleback_mechanism_seed_output_test", LAUNCHER_PATH)
    launch = SimpleNamespace(
        source_revision="a" * 40,
        container_image_id="sha256:" + "b" * 64,
    )
    pose = launcher.Predecessor(
        role="pose_input",
        root=tmp_path / "pose",
        output_path=tmp_path / "pose/authorization.json",
        output_identity=launcher.FileIdentity("c" * 64, 10),
        receipt_path=tmp_path / "pose/receipt.json",
        receipt_identity=launcher.FileIdentity("d" * 64, 11),
        tree_sha256="e" * 64,
        tree_bytes=12,
        tree_files=2,
    )
    geometry = launcher.Predecessor(
        role="geometry",
        root=tmp_path / "geometry",
        output_path=tmp_path / "geometry/output/geometry-gate.json",
        output_identity=launcher.FileIdentity("f" * 64, 13),
        receipt_path=tmp_path / "geometry/audit/run.receipt.json",
        receipt_identity=launcher.FileIdentity("1" * 64, 14),
        tree_sha256="2" * 64,
        tree_bytes=15,
        tree_files=3,
    )
    command, _, output_relative = launcher._stage_command(
        stage="mechanism",
        candidate_id="W16_H4",
        config_relative=launcher.CONFIGS["W16_H4"],
        source_export_manifest_sha256="3" * 64,
        launch=launch,
        predecessors=(pose, geometry),
    )
    assert output_relative == "output/mechanism-bundle/mechanism-probe.json"
    assert command[command.index("--seed-checkpoint-output") + 1] == (
        "/pams/output/mechanism-bundle/learned-encoder-L.pt"
    )
    assert command[command.index("--source-export-manifest-sha256") + 1] == (
        "3" * 64
    )
    launcher.OUTCOME_REGISTRY_ROOT = tmp_path / "outcomes"
    assert launcher._outcome_registry_path(
        "mechanism",
        "W16_H4",
    ) == tmp_path / "outcomes/mechanism/W16_H4.outcome.json"
    with pytest.raises(launcher.LaunchFailure, match="separate epoch11 authority"):
        launcher._load_predecessor(
            role="mechanism",
            candidate_id="W16_H4",
            launch=launch,
        )

    output_root = tmp_path / "stage-output/mechanism-bundle"
    output_root.mkdir()
    output = output_root / "mechanism-probe.json"
    checkpoint = output_root / "learned-encoder-L.pt"
    checkpoint.write_bytes(b"exact-learned-L")
    checkpoint_identity = launcher._identity(
        checkpoint,
        role="test mechanism seed",
    )
    seed = {
        "artifact_type": (
            "pams_conventional_cycleback_mechanism_seed_checkpoint_v1"
        ),
        "produced": True,
        "publication_status": "passed_checkpoint_published",
        "relative_path": "learned-encoder-L.pt",
        "sha256": checkpoint_identity.sha256,
        "bytes": checkpoint_identity.bytes,
        "rejected_checkpoint_written": False,
        "boundary_state_captured_before_diagnostics": True,
        "diagnostics_restored_exact_boundary": True,
        "checkpoint_payload_validated_before_publication": True,
        "retroactive_checkpoint_reconstruction_allowed": False,
        "epoch11_train337_continuation_authorized": False,
        "direct_epoch150_start_authorized": False,
        "captured_boundary_model_state_sha256": "4" * 64,
        "captured_boundary_optimizer_state_sha256": "5" * 64,
        "captured_boundary_rng_state_sha256": "6" * 64,
        "captured_boundary_backend_state_sha256": "7" * 64,
        "captured_sampler_state_sha256": "8" * 64,
        "captured_view_state_sha256": "c" * 64,
        "captured_consumed_video_batch_chain_sha256": "d" * 64,
        "captured_consumed_pair_row_chain_sha256": "e" * 64,
        "trainer_contract_fingerprint": "9" * 64,
        "representation_contract_sha256": "a" * 64,
        "seed_predecessor_lineage_fingerprint": "b" * 64,
    }
    _write_json(
        output,
        {
            "status": "passed",
            "gate": {"overall_pass": True},
            "training": {
                "final_model_state_sha256": "4" * 64,
                "optimizer_state_sha256_at_step256": "5" * 64,
                "mechanism_sampler_state_sha256": "8" * 64,
                "mechanism_view_state_sha256": "c" * 64,
                "mechanism_consumed_video_batch_chain_sha256": "d" * 64,
                "mechanism_consumed_pair_row_chain_sha256": "e" * 64,
                "trainer_contract_fingerprint": "9" * 64,
            },
            "mechanism_seed_checkpoint": seed,
        },
    )
    _, _, status = launcher._validate_stage_output(
        stage="mechanism",
        output_path=output,
        exit_code=0,
    )
    assert status == "passed"

    checkpoint.write_bytes(b"tampered-learned-L")
    with pytest.raises(launcher.LaunchFailure, match="bytes differ"):
        launcher._validate_stage_output(
            stage="mechanism",
            output_path=output,
            exit_code=0,
        )
    checkpoint.unlink()
    rejected = copy.deepcopy(seed)
    rejected.update(
        produced=False,
        publication_status="rejected_not_published",
        relative_path=None,
        sha256=None,
        bytes=None,
    )
    _write_json(
        output,
        {
            "status": "rejected",
            "gate": {"overall_pass": False},
            "training": {
                "final_model_state_sha256": "4" * 64,
                "optimizer_state_sha256_at_step256": "5" * 64,
                "mechanism_sampler_state_sha256": "8" * 64,
                "mechanism_view_state_sha256": "c" * 64,
                "mechanism_consumed_video_batch_chain_sha256": "d" * 64,
                "mechanism_consumed_pair_row_chain_sha256": "e" * 64,
                "trainer_contract_fingerprint": "9" * 64,
            },
            "mechanism_seed_checkpoint": rejected,
        },
    )
    _, _, status = launcher._validate_stage_output(
        stage="mechanism",
        output_path=output,
        exit_code=3,
    )
    assert status == "rejected"
    checkpoint.write_bytes(b"forbidden")
    with pytest.raises(launcher.LaunchFailure, match="rejected mechanism"):
        launcher._validate_stage_output(
            stage="mechanism",
            output_path=output,
            exit_code=3,
        )


def test_mechanism_outcome_registry_binds_checkpoint_and_receipt(
    tmp_path: Path,
) -> None:
    launcher = _load_module("cycleback_mechanism_outcome_test", LAUNCHER_PATH)
    launcher.OUTCOME_REGISTRY_ROOT = tmp_path / "outcomes"
    root = tmp_path / "run"
    output_root = root / "output/mechanism-bundle"
    audit_root = root / "audit"
    output_root.mkdir(parents=True)
    audit_root.mkdir()
    checkpoint = output_root / "learned-encoder-L.pt"
    checkpoint.write_bytes(b"sealed-L")
    identity = launcher._identity(checkpoint, role="test sealed L")
    seed = {
        "artifact_type": (
            "pams_conventional_cycleback_mechanism_seed_checkpoint_v1"
        ),
        "produced": True,
        "publication_status": "passed_checkpoint_published",
        "relative_path": "learned-encoder-L.pt",
        "sha256": identity.sha256,
        "bytes": identity.bytes,
        "captured_boundary_model_state_sha256": "1" * 64,
        "captured_boundary_optimizer_state_sha256": "2" * 64,
        "captured_boundary_rng_state_sha256": "3" * 64,
        "captured_boundary_backend_state_sha256": "4" * 64,
        "captured_sampler_state_sha256": "5" * 64,
        "captured_view_state_sha256": "9" * 64,
        "captured_consumed_video_batch_chain_sha256": "a" * 64,
        "captured_consumed_pair_row_chain_sha256": "b" * 64,
        "trainer_contract_fingerprint": "6" * 64,
        "representation_contract_sha256": "7" * 64,
        "seed_predecessor_lineage_fingerprint": "8" * 64,
        "boundary_state_captured_before_diagnostics": True,
        "diagnostics_restored_exact_boundary": True,
        "checkpoint_payload_validated_before_publication": True,
        "retroactive_checkpoint_reconstruction_allowed": False,
        "rejected_checkpoint_written": False,
        "epoch11_train337_continuation_authorized": False,
        "direct_epoch150_start_authorized": False,
    }
    output = output_root / "mechanism-probe.json"
    _write_json(
        output,
        {
            "training": {
                "final_model_state_sha256": "1" * 64,
                "optimizer_state_sha256_at_step256": "2" * 64,
                "mechanism_sampler_state_sha256": "5" * 64,
                "mechanism_view_state_sha256": "9" * 64,
                "mechanism_consumed_video_batch_chain_sha256": "a" * 64,
                "mechanism_consumed_pair_row_chain_sha256": "b" * 64,
                "trainer_contract_fingerprint": "6" * 64,
            },
            "mechanism_seed_checkpoint": seed,
        },
    )
    output_identity = launcher._identity(output, role="test mechanism output")
    semantic_log = audit_root / "mechanism-seed.semantic-validation.log"
    semantic_log.write_text("validated\n", encoding="utf-8")
    semantic_identity = launcher._identity(
        semantic_log,
        role="test semantic validation",
    )
    _write_json(
        audit_root / "run.receipt.json",
        {
            "mechanism_seed_checkpoint_sha256": identity.sha256,
            "mechanism_seed_checkpoint_bytes": identity.bytes,
            "mechanism_seed_model_state_sha256": "1" * 64,
            "mechanism_seed_optimizer_state_sha256": "2" * 64,
            "mechanism_seed_rng_state_sha256": "3" * 64,
            "mechanism_seed_backend_state_sha256": "4" * 64,
            "mechanism_seed_sampler_state_sha256": "5" * 64,
            "mechanism_seed_view_state_sha256": "9" * 64,
            "mechanism_seed_consumed_video_batch_chain_sha256": "a" * 64,
            "mechanism_seed_consumed_pair_row_chain_sha256": "b" * 64,
            "mechanism_seed_trainer_contract_fingerprint": "6" * 64,
            "mechanism_seed_predecessor_lineage_fingerprint": "8" * 64,
            "mechanism_seed_diagnostics_restored_exact_boundary": True,
            "mechanism_seed_checkpoint_produced": True,
            "mechanism_seed_checkpoint_relative": (
                "output/mechanism-bundle/learned-encoder-L.pt"
            ),
            "mechanism_seed_epoch11_continuation_authorized": False,
            "artifact_type": (
                "pams_conventional_cycleback_mechanism_run_receipt_v1"
            ),
            "status": "passed",
            "mechanism_probe_sha256": output_identity.sha256,
            "output_sha256": output_identity.sha256,
            "output_bytes": output_identity.bytes,
            "mechanism_seed_host_semantic_validation": (
                semantic_identity.to_dict()
            ),
        },
    )
    launch = SimpleNamespace(
        registry_id="9" * 64,
        source_revision="a" * 40,
        container_image_id="sha256:" + "b" * 64,
    )
    launcher._publish_outcome_registry(
        role="mechanism",
        candidate_id="W16_H4",
        final_root=root,
        output_relative="output/mechanism-bundle/mechanism-probe.json",
        launch=launch,
    )
    record = json.loads(
        (tmp_path / "outcomes/mechanism/W16_H4.outcome.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["mechanism_seed_checkpoint"] == identity.to_dict()
    assert record["mechanism_seed_model_state_sha256"] == "1" * 64
    assert record["epoch11_train337_continuation_authorized"] is False
    assert record["direct_epoch150_start_authorized"] is False


def test_failed_mechanism_tree_cannot_retain_seed_checkpoint(tmp_path: Path) -> None:
    launcher = _load_module("cycleback_failed_seed_cleanup_test", LAUNCHER_PATH)
    root = tmp_path / "failed-run"
    bundle = root / "output/mechanism-bundle"
    bundle.mkdir(parents=True)
    seed = bundle / "learned-encoder-L.pt"
    seed.write_bytes(b"must-not-survive")
    (bundle / "mechanism-probe.json").write_text("{}\n", encoding="utf-8")
    for path in (seed, bundle / "mechanism-probe.json"):
        path.chmod(0o444)
    bundle.chmod(0o555)
    (root / "output").chmod(0o555)
    root.chmod(0o555)

    launcher._remove_mechanism_seed_from_failed_tree(root)

    assert not seed.exists()
    assert (bundle / "mechanism-probe.json").exists()


def test_host_semantic_validator_failure_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launcher = _load_module("cycleback_host_semantic_failure_test", LAUNCHER_PATH)
    source = tmp_path / "source"
    output = tmp_path / "output"
    audit = tmp_path / "audit"
    source.mkdir()
    (output / "mechanism-bundle").mkdir(parents=True)
    audit.mkdir()
    mechanism_output = output / "mechanism-bundle/mechanism-probe.json"
    mechanism_output.write_text("{}\n", encoding="utf-8")
    (mechanism_output.with_name("learned-encoder-L.pt")).write_bytes(b"bad")
    observed: list[str] = []

    def failed_run(
        arguments: list[str],
        **_kwargs: object,
    ) -> subprocess.CompletedProcess[bytes]:
        observed.extend(arguments)
        return subprocess.CompletedProcess(arguments, 7, b"", b"")

    monkeypatch.setattr(launcher, "_run", failed_run)
    with pytest.raises(launcher.LaunchFailure, match="semantic validation failed"):
        launcher._validate_mechanism_seed_semantics_in_fresh_container(
            launch=SimpleNamespace(container_image_id="sha256:" + "a" * 64),
            source_export=source,
            output_root=output,
            output_path=mechanism_output,
            config_relative="configs/conventional_cycleback/w16_hop4_v1.yaml",
            audit_root=audit,
        )
    assert "--network" in observed
    assert "none" in observed
    assert "--read-only" in observed
    assert observed[observed.index("--user") + 1] == "1000:1000"


def test_fresh_seed_validator_rejects_arbitrary_checkpoint_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_module("cycleback_seed_semantic_negative_test", SEED_VALIDATOR_PATH)
    monkeypatch.setattr(
        validator,
        "validate_mechanism_probe_contract",
        lambda *_args: None,
    )
    checkpoint = tmp_path / "learned-encoder-L.pt"
    checkpoint.write_bytes(b"sealed-L-is-not-a-checkpoint")
    output = tmp_path / "mechanism-probe.json"
    _write_json(
        output,
        {
            "mechanism_seed_checkpoint": {
                "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "bytes": checkpoint.stat().st_size,
            }
        },
    )
    with pytest.raises((RuntimeError, ValueError, EOFError, pickle.UnpicklingError)):
        validator.validate_seed(
            output,
            checkpoint,
            ROOT / "configs/conventional_cycleback/w16_hop4_v1.yaml",
        )


def test_thin_wrappers_are_identity_free_and_python_isolated() -> None:
    for path in (ADAPTER, GEOMETRY, MECHANISM):
        source = path.read_text(encoding="utf-8")
        assert "/usr/bin/env -i" in source
        assert "/usr/bin/python3 -I" in source
        assert "PYTHONOPTIMIZE" not in source
        assert "PAMS_SOURCE_REVISION" in source
        assert "PAMS_IMAGE_ID" in source
        assert "PAMS_TRAIN337_POSE_SNAPSHOT" not in source
        assert "assert " not in source


def test_launcher_freezes_science_and_security_boundaries() -> None:
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    validator_source = SEED_VALIDATOR_PATH.read_text(encoding="utf-8")
    cli_source = (ROOT / "src/pams/cli.py").read_text(encoding="utf-8")

    for required in (
        'GIT_BINARY = "/usr/bin/git"',
        'DOCKER_BINARY = "/usr/bin/docker"',
        '"--cidfile"',
        '"docker_create_cidfile_sha256"',
        '"docker_create_stdout_sha256"',
        '"docker_create_stderr_sha256"',
        '"--entrypoint"',
        '"--network"',
        '"none"',
        '"--read-only"',
        '"--cap-drop"',
        '"ALL"',
        '"--encoder-batch-size"',
        '"8"',
        "_cleanup_exact_container(container_id)",
        "_write_source_manifest",
        "_artifact_manifest",
        "_require_bootstrap_activation()",
        "_BOOTSTRAP_SOURCE_CHECKOUT_LOCATOR = \"\"",
        "_BOOTSTRAP_LAUNCH_AUTHORIZATION_SHA256 = \"\"",
        "_BOOTSTRAP_INTEGRATION_AUTHORIZATION_SHA256 = \"\"",
        "_outcome_slot_lock_path",
        "not container_create_attempted",
        "validate_cycleback_pose_authority",
        '"--seed-checkpoint-output"',
        '"--source-export-manifest-sha256"',
        "mechanism_seed_checkpoint_sha256",
        "mechanism_seed_checkpoint_relative",
    ):
        assert required in source
    assert "assert " not in source
    assert "_CONTRACT_SPEC" not in source
    assert "PAMS_TRAIN337_POSE_SNAPSHOT" not in source
    assert 'model.load_state_dict(value["model_state"], strict=True)' in (
        validator_source
    )
    assert 'optimizer.load_state_dict(value["optimizer_state"])' in validator_source
    assert "_reject_v4e_from_generic_training(config" in cli_source
    assert cli_source.count("_reject_v4e_from_generic_training(config, operation=") == 2


def test_candidate_configs_disclose_proxy_status_without_result_claims() -> None:
    for path in (ROOT / "configs/conventional_cycleback").glob("*.yaml"):
        source = path.read_text(encoding="utf-8").lower()
        assert "independently inferred" in source
        assert "not the authors' baseline" in source
        assert "paper_table_claim_eligible: false" in source
        assert "nmae" not in source
        assert "obo" not in source
        assert "performance" not in source
