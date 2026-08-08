#!/usr/bin/env python3
"""Exact, stable-FD Docker pre/post contract for cycle-back stages."""

from __future__ import annotations

import argparse
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _stable_bytes(path: Path, *, role: str) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{role} must be a regular non-symlink file")
    flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            encoded = handle.read()
            closed = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = path.lstat()
    for field in ("st_dev", "st_ino", "st_size", "st_mtime_ns"):
        if not (
            getattr(before, field)
            == getattr(opened, field)
            == getattr(closed, field)
            == getattr(after, field)
        ):
            raise RuntimeError(f"{role} changed while being read")
    return encoded


def _json_no_duplicates(encoded: bytes, *, role: str) -> Any:
    def reject(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{role} contains duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(encoded, object_pairs_hook=reject)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is not valid JSON") from exc


def _read_json(path: Path, *, role: str) -> Any:
    return _json_no_duplicates(_stable_bytes(path, role=role), role=role)


def _container_id(path: Path) -> str:
    try:
        value = _stable_bytes(path, role="docker create ID").decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("docker create ID is not ASCII") from exc
    value = value.removeprefix("sha256:")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("docker create ID is malformed")
    return f"sha256:{value}"


def _environment(values: object) -> dict[str, str]:
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError("container environment is malformed")
    result: dict[str, str] = {}
    for encoded in values:
        key, separator, value = encoded.partition("=")
        if not separator or not key or key in result:
            raise ValueError("container environment contains a malformed or duplicate key")
        result[key] = value
    return result


def _actual_mounts(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError("container mounts are malformed")
    result: dict[str, dict[str, Any]] = {}
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError("container mount row is malformed")
        destination = raw.get("Destination")
        if not isinstance(destination, str) or destination in result:
            raise ValueError("container mount destination is malformed or duplicated")
        result[destination] = {
            "type": raw.get("Type"),
            "source": raw.get("Source"),
            "rw": raw.get("RW"),
            "propagation": raw.get("Propagation"),
        }
    return result


def _expected_mounts(spec: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = spec.get("mounts")
    if not isinstance(rows, list):
        raise ValueError("wrapper contract mount schema mismatch")
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        if not isinstance(raw, Mapping) or set(raw) != {
            "destination",
            "type",
            "source",
            "rw",
            "propagation",
        }:
            raise ValueError("wrapper contract mount row mismatch")
        destination = raw["destination"]
        if not isinstance(destination, str) or destination in result:
            raise ValueError("wrapper contract mount destination mismatch")
        result[destination] = {
            "type": raw["type"],
            "source": raw["source"],
            "rw": raw["rw"],
            "propagation": raw["propagation"],
        }
    return result


def _validate_device_requests(value: object, expected_device: object) -> None:
    requests = [] if value is None else value
    if expected_device is None:
        if requests:
            raise ValueError("CPU-only container unexpectedly requests a GPU")
        return
    if not isinstance(expected_device, str) or not expected_device.isdigit():
        raise ValueError("wrapper contract GPU device is invalid")
    if not isinstance(requests, list) or len(requests) != 1:
        raise ValueError("container GPU request count mismatch")
    request = requests[0]
    if not isinstance(request, Mapping):
        raise ValueError("container GPU request is malformed")
    expected = {
        "Driver": "",
        "Count": 0,
        "DeviceIDs": [expected_device],
        "Capabilities": [["gpu"]],
        "Options": {},
    }
    if dict(request) != expected:
        raise ValueError("container GPU device mapping mismatch")


def _exact_host_config(host: Mapping[str, Any], spec: Mapping[str, Any], *, phase: str) -> None:
    expected = spec.get("host_config_exact")
    if not isinstance(expected, Mapping):
        raise ValueError("wrapper host-config contract is missing")
    for key, value in expected.items():
        if host.get(key) != value:
            raise ValueError(f"{phase} HostConfig.{key} mismatch")
    forbidden_nonempty = (
        "CapAdd",
        "Devices",
        "Binds",
        "VolumesFrom",
        "PortBindings",
        "Dns",
        "DnsOptions",
        "DnsSearch",
        "ExtraHosts",
        "Links",
    )
    for key in forbidden_nonempty:
        if host.get(key) not in (None, [], {}):
            raise ValueError(f"{phase} HostConfig.{key} is unexpectedly non-empty")
    if host.get("Privileged") is not False or host.get("PublishAllPorts") is not False:
        raise ValueError(f"{phase} privileged or published-port state mismatch")
    restart = host.get("RestartPolicy")
    if restart != {"Name": "no", "MaximumRetryCount": 0}:
        raise ValueError(f"{phase} restart policy mismatch")


def validate_inspect(
    *,
    inspect_path: Path,
    create_id_path: Path,
    spec_path: Path,
    phase: str,
    expected_exit_code: int | None,
) -> None:
    spec = _read_json(spec_path, role="wrapper contract")
    if not isinstance(spec, Mapping) or spec.get("schema_version") != 2:
        raise ValueError("wrapper contract schema mismatch")
    inspect = _read_json(inspect_path, role=f"{phase} inspect")
    if not isinstance(inspect, list) or len(inspect) != 1 or not isinstance(inspect[0], Mapping):
        raise ValueError(f"{phase} inspect root mismatch")
    item = inspect[0]
    expected_id = _container_id(create_id_path)
    if item.get("Id") != expected_id:
        raise ValueError(f"{phase} inspect/create ID mismatch")
    if item.get("Name") != f"/{spec.get('container_name')}":
        raise ValueError(f"{phase} container name mismatch")
    if item.get("Image") != spec.get("image_id"):
        raise ValueError(f"{phase} actual image ID mismatch")
    if item.get("Path") != spec.get("path") or item.get("Args") != spec.get("args"):
        raise ValueError(f"{phase} final Path/Args mismatch")
    config = item.get("Config")
    host = item.get("HostConfig")
    state = item.get("State")
    network = item.get("NetworkSettings")
    if not all(isinstance(value, Mapping) for value in (config, host, state, network)):
        raise ValueError(f"{phase} inspect sections are missing")
    if config.get("Image") != spec.get("image_id"):
        raise ValueError(f"{phase} configured image mismatch")
    if config.get("Entrypoint") != spec.get("entrypoint"):
        raise ValueError(f"{phase} Entrypoint mismatch")
    if config.get("Cmd") != spec.get("args"):
        raise ValueError(f"{phase} Cmd mismatch")
    if config.get("User") != spec.get("user") or config.get("WorkingDir") != spec.get(
        "working_dir"
    ):
        raise ValueError(f"{phase} user/workdir mismatch")
    if config.get("ExposedPorts") not in (None, {}):
        raise ValueError(f"{phase} exposed ports are forbidden")
    if config.get("Volumes") not in (None, {}):
        raise ValueError(f"{phase} image volumes are forbidden")
    actual_environment = _environment(config.get("Env"))
    expected_environment = spec.get("expected_environment")
    if not isinstance(expected_environment, Mapping) or actual_environment != dict(
        expected_environment
    ):
        raise ValueError(f"{phase} exact environment set mismatch")
    forbidden_environment = spec.get("forbidden_environment")
    if not isinstance(forbidden_environment, list) or any(
        not isinstance(key, str) for key in forbidden_environment
    ):
        raise ValueError("wrapper contract forbidden environment schema mismatch")
    if any(key in actual_environment for key in forbidden_environment):
        raise ValueError(f"{phase} forbidden environment was injected")
    _exact_host_config(host, spec, phase=phase)
    if _actual_mounts(item.get("Mounts")) != _expected_mounts(spec):
        raise ValueError(f"{phase} exact mount set mismatch")
    _validate_device_requests(host.get("DeviceRequests"), spec.get("gpu_device"))
    if network.get("Ports") not in (None, {}):
        raise ValueError(f"{phase} network ports are forbidden")
    if phase == "pre":
        if not (
            state.get("Running") is False
            and state.get("Status") == "created"
            and state.get("OOMKilled") is False
            and state.get("Error") == ""
            and state.get("Pid") == 0
        ):
            raise ValueError("pre-run container is not in exact created state")
    elif phase == "post":
        if expected_exit_code is None:
            raise ValueError("post-run validation requires the observed exit code")
        allowed = spec.get("allowed_exit_codes")
        if not isinstance(allowed, list) or expected_exit_code not in allowed:
            raise ValueError("post-run exit code is outside the contract")
        if not (
            state.get("Running") is False
            and state.get("Status") == "exited"
            and state.get("ExitCode") == expected_exit_code
            and state.get("OOMKilled") is False
            and state.get("Error") == ""
            and state.get("Pid") == 0
        ):
            raise ValueError("post-run container state/exit mismatch")
    else:
        raise ValueError("inspect phase must be pre or post")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", type=Path, required=True)
    parser.add_argument("--create-id", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--phase", choices=("pre", "post"), required=True)
    parser.add_argument("--expected-exit-code", type=int)
    arguments = parser.parse_args(argv)
    try:
        validate_inspect(
            inspect_path=arguments.inspect,
            create_id_path=arguments.create_id,
            spec_path=arguments.spec,
            phase=arguments.phase,
            expected_exit_code=arguments.expected_exit_code,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"cycleback wrapper inspect verification failed: {exc}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
