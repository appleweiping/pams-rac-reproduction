from __future__ import annotations

import copy
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = ROOT / "scripts/server/pams_conventional_cycleback_secure_launcher.py"
CONTRACT_PATH = ROOT / "scripts/server/pams_conventional_cycleback_wrapper_contract.py"
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
    ):
        assert required in source
    assert "assert " not in source
    assert "_CONTRACT_SPEC" not in source
    assert "PAMS_TRAIN337_POSE_SNAPSHOT" not in source
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
