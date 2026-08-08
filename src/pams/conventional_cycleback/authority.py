"""Fail-closed unified-2D-to-cycleback pose-input authority chain.

Only one independently preregistered representation PASS may enter the launch
registry.  The denied v4d full337 outcome is exact diagnostic evidence only;
it is never a cycle-back training authority or a second adapter entrypoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pams.conventional_cycleback.runtime import (
    JointMaskEntryReceipt,
    JointMaskSetSnapshot,
    joint_mask_path,
    load_identity_map,
    load_joint_mask_sidecar,
    load_joint_mask_snapshot,
    load_pair_eligibility,
    load_segment_index,
    load_unified_2d_pose_cache_with_receipt,
    validate_unified_2d_sequence,
)
from pams.data import (
    PoseCacheEntryReceipt,
    PoseCacheSetSnapshot,
    pose_cache_path,
)

_SHA256_LENGTH = 64
_ADAPTER_AUTH_TYPE = "pams_conventional_cycleback_pose_input_authorization_v1"
_ADAPTER_RECEIPT_TYPE = (
    "pams_conventional_cycleback_pose_input_authorization_run_receipt_v1"
)
_LAUNCH_AUTH_TYPE = "pams_conventional_cycleback_launch_authorization_v1"
_LAUNCH_RECEIPT_TYPE = "pams_conventional_cycleback_launch_registry_receipt_v1"
_REPRESENTATION_GATE_TYPE = "pams_cycleback_unified_2d_representation_gate_v1"
_REPRESENTATION_AUTH_TYPE = "pams_cycleback_unified_2d_input_authorization_v1"
_REPRESENTATION_RECEIPT_TYPE = "pams_cycleback_unified_2d_run_receipt_v1"
_CANONICAL_LAUNCH_REGISTRY_ROOT = Path(
    "/media/lenovo/data2/pams-rac/authorizations/"
    "pams-conventional-cycleback-launch-v1"
)
_CANONICAL_ADAPTER_PARENT = Path(
    "/media/lenovo/data2/pams-rac/runs/"
    "pams-conventional-cycleback-pose-input-authorization-v1"
)
_CANONICAL_REPRESENTATION_PARENT = Path(
    "/media/lenovo/data2/pams-rac/runs/"
    "pams-cycleback-unified-2d-representation-v1"
)
_CANONICAL_SYNTHETIC_PREREG_ROOT = Path(
    "/media/lenovo/data2/pams-rac/authorizations/"
    "pams-conventional-cycleback-synthetic-prereg-v1"
)
_CONTAINER_REPRESENTATION_ROOT = Path("/pams/representation")
_CONTAINER_LAUNCH_REGISTRY_ROOT = Path("/pams/launch-registry")
_CONTAINER_POSE_AUTHORITY_ROOT = Path("/pams/pose-authority")
# Fail-closed preregistration slots.  A later reviewed commit must replace all
# six values with one actually sealed representation-adapter outcome.  Until
# then no launch registry can be reserved and raw full337 can never authorize
# this candidate by itself.
_APPROVED_REPRESENTATION_LOCATOR = ""
_APPROVED_REPRESENTATION_RESERVATION_SHA256 = ""
_APPROVED_REPRESENTATION_GATE_SHA256 = ""
_APPROVED_REPRESENTATION_AUTHORIZATION_SHA256 = ""
_APPROVED_REPRESENTATION_RUN_RECEIPT_SHA256 = ""
_APPROVED_REPRESENTATION_SNAPSHOT_SHA256 = ""
_APPROVED_SYNTHETIC_PREREG_SHA256 = ""
_RAW_FULL337_DENIED_LOCATOR = (
    "/media/lenovo/data2/pams-rac/runs/pose-recovery-v4d-full337/"
    "4984e8f328fc-20260808T171000Z"
)
_RAW_FULL337_GATE_SHA256 = "cf1342f5b77540b3206471fc2da2dc4967e1b40c98a91b7769703dec1a0e2946"
_RAW_FULL337_GATE_BYTES = 369_003
_RAW_FULL337_DENIAL_SHA256 = "dac38c47bbdc88978d618bcec3523d828cd6bfcc5d2250926b78a3b642bfc9a1"
_RAW_FULL337_DENIAL_BYTES = 2_431
_RAW_FULL337_RUN_SHA256 = "01c82e61c2a214448d231706b51c5963c6a846e7411e9fe4cd1edf10c2e20210"
_RAW_FULL337_RUN_BYTES = 1_900
_RAW_FULL337_LEDGER_SHA256 = "ebc9f85c657ca15d2b79e22dbe064183b799529a852d63cfc9b068e0bd34a601"
_RAW_FULL337_LEDGER_BYTES = 2_261_308
_RAW_FULL337_CACHE_SET_SHA256 = "a8cb1d5e5d448f68ebe143a898e2476df08befd7145d1e743d3c292fda537e76"
_CYCLEBACK_IMAGE_ID = (
    "sha256:0a4d42c2d9911f147a17860e4e15095746b21c4e618e4fc1b20b62dd443c5898"
)
_CYCLEBACK_CONFIG_PATHS = (
    "configs/conventional_cycleback/w16_hop4_v1.yaml",
    "configs/conventional_cycleback/w16_hop2_v1.yaml",
    "configs/conventional_cycleback/w24_hop4_v1.yaml",
)
# Deliberately empty activation slots.  The secure-scaffold commit cannot
# authorize its own checkout.  A later, independently reviewed activation
# commit must copy this exact tuple from the sealed representation outcome.
_APPROVED_CYCLEBACK_SOURCE_REVISION = ""
_APPROVED_CYCLEBACK_SOURCE_TREE_SHA = ""
_APPROVED_CYCLEBACK_CONFIGS: Mapping[str, Mapping[str, Any]] = {}


@dataclass(frozen=True, slots=True)
class UnifiedRepresentationAuthority:
    """Future independent unified-2D representation PASS authority."""

    root: Path
    source_revision: str
    container_image_id: str
    gate_sha256: str
    authorization_sha256: str
    run_receipt_sha256: str
    snapshot_sha256: str
    segment_index_path: Path
    segment_index_sha256: str
    segment_reset_policy_sha256: str
    joint_mask_dir: Path
    joint_mask_snapshot_path: Path
    joint_mask_snapshot_sha256: str
    joint_mask_snapshot: V4EJointMaskSnapshot
    identity_map_path: Path
    identity_map_sha256: str
    pair_eligibility_path: Path
    pair_eligibility_sha256: str
    pose_fingerprint: str
    cache_set_sha256: str
    snapshot: PoseCacheSetSnapshot
    bindings: Mapping[str, Any]
    authorized_cycleback_consumer: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CycleBackPoseAuthority:
    """Validated downstream-facing adapter authority."""

    root: Path
    snapshot_path: Path
    cache_dir: Path
    segment_index_path: Path
    joint_mask_dir: Path
    joint_mask_snapshot_path: Path
    identity_map_path: Path
    pair_eligibility_path: Path
    authorization_path: Path
    run_receipt_path: Path
    snapshot: PoseCacheSetSnapshot
    authorization_sha256: str
    run_receipt_sha256: str
    segment_index_sha256: str
    segment_reset_policy_sha256: str
    joint_mask_snapshot_sha256: str
    joint_mask_set_sha256: str
    identity_map_sha256: str
    pair_eligibility_sha256: str
    representation_bindings: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CycleBackLaunchAuthority:
    """Validated one-shot canonical launch reservation."""

    root: Path
    authorization_path: Path
    receipt_path: Path
    authorization_sha256: str
    receipt_sha256: str
    registry_id: str
    source_revision: str
    source_tree_sha: str
    container_image_id: str
    configs: Mapping[str, Mapping[str, Any]]
    representation_root: str
    representation_bindings: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class V4EJointMaskSnapshot:
    """Exact producer-side joint-mask snapshot before adapter normalization."""

    entries: tuple[JointMaskEntryReceipt, ...]
    fingerprint: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_regular_bytes(path: Path, *, role: str) -> tuple[bytes, str, int]:
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode), f"{role} must be a regular non-symlink file")
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
    stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, key) != getattr(opened, key)
        or getattr(opened, key) != getattr(closed, key)
        or getattr(closed, key) != getattr(after, key)
        for key in stable
    ):
        raise RuntimeError(f"{role} changed while being read")
    return encoded, hashlib.sha256(encoded).hexdigest(), len(encoded)


def _read_json_object(path: Path, *, role: str) -> tuple[dict[str, Any], str, int]:
    encoded, digest, byte_count = _read_regular_bytes(path, role=role)
    try:
        value = _json_loads_no_duplicates(encoded, role=role)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is not valid JSON") from exc
    _require(isinstance(value, dict), f"{role} must contain a JSON object")
    return value, digest, byte_count


def _json_loads_no_duplicates(encoded: bytes | str, *, role: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{role} contains duplicate JSON field {key!r}")
            result[key] = value
        return result

    return json.loads(encoded, object_pairs_hook=reject_duplicates)


def _require_sealed_tree(root: Path, *, role: str) -> Path:
    resolved = root.resolve(strict=True)
    root_stat = root.lstat()
    _require(not stat.S_ISLNK(root_stat.st_mode), f"{role} root must not be a symlink")
    _require(stat.S_ISDIR(root_stat.st_mode), f"{role} root must be a directory")
    for candidate in (resolved, *resolved.rglob("*")):
        details = candidate.lstat()
        _require(not stat.S_ISLNK(details.st_mode), f"{role} contains a symlink")
        _require(details.st_mode & 0o222 == 0, f"{role} contains a writable path")
        _require(
            stat.S_ISDIR(details.st_mode) or stat.S_ISREG(details.st_mode),
            f"{role} contains a non-file filesystem object",
        )
    return resolved


def _require_declared_or_fixed_mount(
    *,
    requested_root: Path,
    sealed_root: Path,
    declared_host_root: Path,
    fixed_container_mount: Path,
    role: str,
) -> None:
    _require(declared_host_root.is_absolute(), f"{role} declared host root must be absolute")
    if requested_root == fixed_container_mount:
        _require(
            sealed_root == fixed_container_mount.resolve(strict=True),
            f"{role} fixed container mount identity mismatch",
        )
        return
    _require(
        requested_root == declared_host_root
        and sealed_root == declared_host_root.resolve(strict=True),
        f"{role} bytes are not the declared host root or fixed container mount",
    )


def _snapshot_from_mapping(value: object, *, role: str) -> PoseCacheSetSnapshot:
    _require(isinstance(value, Mapping), f"{role} must be an object")
    payload = dict(value)
    _require(
        set(payload)
        == {"schema_version", "pose_fingerprint", "fingerprint", "entry_count", "entries"},
        f"{role} schema mismatch",
    )
    raw_entries = payload["entries"]
    _require(isinstance(raw_entries, list), f"{role} entries must be a list")
    entries: list[PoseCacheEntryReceipt] = []
    for raw in raw_entries:
        _require(isinstance(raw, Mapping), f"{role} entry must be an object")
        row = dict(raw)
        _require(
            set(row) == {"video_id", "cache_sha256", "bytes"},
            f"{role} entry schema mismatch",
        )
        entries.append(
            PoseCacheEntryReceipt(
                video_id=row["video_id"],
                cache_sha256=row["cache_sha256"],
                bytes=row["bytes"],
            )
        )
    snapshot = PoseCacheSetSnapshot(
        schema_version=payload["schema_version"],
        pose_fingerprint=payload["pose_fingerprint"],
        entries=tuple(entries),
    )
    _require(payload["entry_count"] == len(snapshot.entries), f"{role} count mismatch")
    _require(payload["fingerprint"] == snapshot.fingerprint, f"{role} digest mismatch")
    return snapshot


def _load_v4e_joint_mask_snapshot(
    path: Path,
) -> tuple[V4EJointMaskSnapshot, str, int]:
    """Load the exact v4e producer schema without weakening the adapter schema."""

    payload, digest, byte_count = _read_json_object(
        path,
        role="v4e joint-mask snapshot",
    )
    _require(
        set(payload)
        == {
            "schema_version",
            "artifact_type",
            "entry_count",
            "fingerprint",
            "entries",
        }
        and payload["schema_version"] == 1
        and payload["artifact_type"]
        == "pams_pose_recovery_v4e_joint_mask_set_snapshot_v1"
        and isinstance(payload["entries"], list),
        "v4e joint-mask snapshot schema mismatch",
    )
    entries: list[JointMaskEntryReceipt] = []
    for raw in payload["entries"]:
        _require(
            isinstance(raw, Mapping)
            and set(raw) == {"video_id", "sidecar_sha256", "bytes"},
            "v4e joint-mask snapshot entry schema mismatch",
        )
        entries.append(JointMaskEntryReceipt(**dict(raw)))
    ordered = tuple(sorted(entries, key=lambda entry: entry.video_id))
    _require(
        tuple(entries) == ordered
        and len({entry.video_id for entry in entries}) == len(entries)
        and payload["entry_count"] == len(entries),
        "v4e joint-mask snapshot membership/order mismatch",
    )
    identity = {
        "schema_version": 1,
        "entries": [entry.to_dict() for entry in entries],
    }
    fingerprint = _canonical_json_sha256(identity)
    _require(
        payload["fingerprint"] == fingerprint,
        "v4e joint-mask snapshot fingerprint mismatch",
    )
    return (
        V4EJointMaskSnapshot(entries=tuple(entries), fingerprint=fingerprint),
        digest,
        byte_count,
    )


def _binding_object(payload: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    value = payload.get("bindings")
    _require(isinstance(value, Mapping), f"{role} bindings are missing")
    return dict(value)


def _read_json_object_list(path: Path, *, role: str) -> tuple[dict[str, Any], str, int]:
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode), f"{role} must be a regular non-symlink file")
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
    _require(
        all(
            getattr(before, key) == getattr(opened, key)
            == getattr(closed, key) == getattr(after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        ),
        f"{role} changed while being read",
    )
    try:
        value = _json_loads_no_duplicates(encoded, role=role)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} is not valid JSON") from exc
    _require(
        isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict),
        f"{role} inspect schema mismatch",
    )
    return value[0], hashlib.sha256(encoded).hexdigest(), len(encoded)


def _environment_map(value: object, *, role: str) -> dict[str, str]:
    _require(
        isinstance(value, list) and all(isinstance(item, str) for item in value),
        f"{role} environment is malformed",
    )
    result: dict[str, str] = {}
    for encoded in value:
        key, separator, item = encoded.partition("=")
        _require(
            bool(separator) and bool(key) and key not in result,
            f"{role} environment contains a malformed or duplicate key",
        )
        result[key] = item
    return result


def _mount_map(value: object, *, role: str) -> dict[str, dict[str, Any]]:
    _require(isinstance(value, list), f"{role} mounts are malformed")
    result: dict[str, dict[str, Any]] = {}
    for raw in value:
        _require(isinstance(raw, Mapping), f"{role} mount row is malformed")
        destination = raw.get("Destination")
        _require(
            isinstance(destination, str) and destination not in result,
            f"{role} mount destination is malformed or duplicated",
        )
        result[destination] = {
            "type": raw.get("Type"),
            "source": raw.get("Source"),
            "rw": raw.get("RW"),
            "propagation": raw.get("Propagation"),
        }
    return result


def _expected_mount_map(value: object) -> dict[str, dict[str, Any]]:
    _require(isinstance(value, list), "adapter container contract mounts are missing")
    result: dict[str, dict[str, Any]] = {}
    expected_fields = {"destination", "type", "source", "rw", "propagation"}
    for raw in value:
        _require(
            isinstance(raw, Mapping) and set(raw) == expected_fields,
            "adapter container contract mount row mismatch",
        )
        destination = raw["destination"]
        _require(
            isinstance(destination, str) and destination not in result,
            "adapter container contract mount destination mismatch",
        )
        result[destination] = {
            "type": raw["type"],
            "source": raw["source"],
            "rw": raw["rw"],
            "propagation": raw["propagation"],
        }
    return result


def _manifest_bytes(
    root: Path,
    *,
    excluded: set[Path] | None = None,
    role: str,
) -> bytes:
    omitted = set() if excluded is None else {path.resolve() for path in excluded}
    rows: list[str] = []
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        if path.resolve() in omitted:
            continue
        encoded, digest, byte_count = _read_regular_bytes(path, role=role)
        _require(len(encoded) == byte_count, f"{role} byte count mismatch")
        rows.append(
            f"{digest} {byte_count} {path.relative_to(root).as_posix()}\n"
        )
    return "".join(rows).encode("utf-8")


def _validate_adapter_container_inspect(
    *,
    item: Mapping[str, Any],
    contract: Mapping[str, Any],
    expected_container_id: str,
    phase: str,
) -> None:
    _require(item.get("Id") == expected_container_id, "adapter inspect/create ID mismatch")
    _require(
        item.get("Name") == f"/{contract['container_name']}"
        and item.get("Image") == contract["image_id"],
        f"adapter {phase} inspect name/image mismatch",
    )
    _require(
        item.get("Path") == contract["path"]
        and item.get("Args") == contract["args"],
        f"adapter {phase} final Path/Args mismatch",
    )
    config = item.get("Config")
    host = item.get("HostConfig")
    state = item.get("State")
    network = item.get("NetworkSettings")
    _require(
        all(isinstance(value, Mapping) for value in (config, host, state, network)),
        f"adapter {phase} inspect sections are missing",
    )
    _require(
        config.get("Image") == contract["image_id"]
        and config.get("Entrypoint") == contract["entrypoint"]
        and config.get("Cmd") == contract["args"]
        and config.get("User") == contract["user"]
        and config.get("WorkingDir") == contract["working_dir"]
        and config.get("ExposedPorts") in (None, {})
        and config.get("Volumes") in (None, {}),
        f"adapter {phase} Docker Config mismatch",
    )
    actual_environment = _environment_map(
        config.get("Env"), role=f"adapter {phase} inspect"
    )
    _require(
        actual_environment == contract["expected_environment"],
        f"adapter {phase} exact environment mismatch",
    )
    _require(
        not any(key in actual_environment for key in contract["forbidden_environment"]),
        f"adapter {phase} contains a forbidden environment variable",
    )
    for key, expected in contract["host_config_exact"].items():
        _require(host.get(key) == expected, f"adapter {phase} HostConfig.{key} mismatch")
    for key in (
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
    ):
        _require(
            host.get(key) in (None, [], {}),
            f"adapter {phase} HostConfig.{key} is unexpectedly non-empty",
        )
    _require(
        host.get("Privileged") is False
        and host.get("PublishAllPorts") is False
        and host.get("RestartPolicy") == {"Name": "no", "MaximumRetryCount": 0}
        and host.get("DeviceRequests") in (None, []),
        f"adapter {phase} privilege/restart/device state mismatch",
    )
    _require(
        _mount_map(item.get("Mounts"), role=f"adapter {phase}")
        == _expected_mount_map(contract["mounts"]),
        f"adapter {phase} exact mount set mismatch",
    )
    _require(
        network.get("Ports") in (None, {}),
        f"adapter {phase} network ports are forbidden",
    )
    expected_state = {
        "Running": False,
        "Status": "created" if phase == "pre" else "exited",
        "OOMKilled": False,
        "Error": "",
        "Pid": 0,
    }
    if phase == "post":
        expected_state["ExitCode"] = 0
    _require(
        all(state.get(key) == expected for key, expected in expected_state.items()),
        f"adapter {phase} exact container state mismatch",
    )


def _copy_cache_exclusive(source: Path, destination: Path) -> None:
    source_before = source.lstat()
    _require(stat.S_ISREG(source_before.st_mode), "authorized cache source must be a regular file")
    _require(not source.is_symlink(), "authorized cache source must not be a symlink")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(destination, flags, 0o440)
    source_flags = os.O_RDONLY | int(getattr(os, "O_BINARY", 0))
    source_flags |= int(getattr(os, "O_NOFOLLOW", 0))
    source_flags |= int(getattr(os, "O_CLOEXEC", 0))
    source_descriptor = os.open(source, source_flags)
    try:
        source_opened = os.fstat(source_descriptor)
        with os.fdopen(source_descriptor, "rb") as read_handle, os.fdopen(
            descriptor, "wb"
        ) as write_handle:
            source_descriptor = -1
            descriptor = -1
            shutil.copyfileobj(read_handle, write_handle, length=1024 * 1024)
            write_handle.flush()
            os.fsync(write_handle.fileno())
            source_closed = os.fstat(read_handle.fileno())
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if descriptor >= 0:
            os.close(descriptor)
    source_after = source.lstat()
    _require(
        all(
            getattr(source_before, key) == getattr(source_opened, key)
            == getattr(source_closed, key) == getattr(source_after, key)
            for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
        ),
        "authorized cache source changed while being copied",
    )
    _fsync_directory(destination.parent)


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical audited JSON with O_EXCL and fsync."""

    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= int(getattr(os, "O_NOFOLLOW", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(path, flags, 0o440)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            descriptor = -1
            json.dump(dict(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | int(getattr(os, "O_DIRECTORY", 0))
    flags |= int(getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["/usr/bin/git", "-C", os.fspath(repository), *arguments],
        check=False,
        env={
            "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        },
        capture_output=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"git command failed: {message}")
    return completed.stdout


def _git_text(repository: Path, *arguments: str) -> str:
    return _git_bytes(repository, *arguments).decode("ascii").strip()


def _validate_synthetic_preregistration(
    *,
    configs: Mapping[str, Mapping[str, Any]],
) -> tuple[Path, str, int]:
    _require(
        _is_sha256(_APPROVED_SYNTHETIC_PREREG_SHA256),
        "no synthetic cycle-back threshold preregistration has been approved",
    )
    root = _require_sealed_tree(
        _CANONICAL_SYNTHETIC_PREREG_ROOT,
        role="cycleback synthetic preregistration",
    )
    path = root / "cycleback-synthetic-prereg.json"
    payload, digest, byte_count = _read_json_object(
        path,
        role="cycleback synthetic preregistration",
    )
    _require(
        digest == _APPROVED_SYNTHETIC_PREREG_SHA256,
        "synthetic preregistration differs from the reviewed bytes",
    )
    _require(
        payload.get("schema_version") == 1
        and payload.get("artifact_type")
        == "pams_conventional_cycleback_synthetic_preregistration_v1"
        and payload.get("status") == "passed"
        and payload.get("label_free") is True
        and payload.get("candidate_configs")
        == {key: dict(value) for key, value in configs.items()}
        and payload.get("thresholds_frozen_before_train337") is True
        and payload.get("thresholds_achievable_on_synthetic_cycles") is True
        and payload.get("joint_support_nulls_evaluable") is True,
        "synthetic cycle-back preregistration contract mismatch",
    )
    for forbidden in (
        "scientific_authority_granted",
        "baseline_training_authorized",
        "full_training_authorized",
        "development_evaluation_authorized",
        "sealed_evaluation_authorized",
    ):
        _require(payload.get(forbidden) is False, f"synthetic prereg illegally sets {forbidden}")
    return path, digest, byte_count


def validate_unified_representation_authority(
    root: str | Path,
    *,
    declared_host_root: str | Path,
) -> UnifiedRepresentationAuthority:
    """Validate a future independent unified-2D PASS, never raw full337 alone."""

    declared = Path(declared_host_root)
    approved = (
        _APPROVED_REPRESENTATION_LOCATOR,
        _APPROVED_REPRESENTATION_RESERVATION_SHA256,
        _APPROVED_REPRESENTATION_GATE_SHA256,
        _APPROVED_REPRESENTATION_AUTHORIZATION_SHA256,
        _APPROVED_REPRESENTATION_RUN_RECEIPT_SHA256,
        _APPROVED_REPRESENTATION_SNAPSHOT_SHA256,
    )
    _require(
        bool(approved[0]) and all(_is_sha256(value) for value in approved[1:]),
        "no unified representation outcome has been independently preregistered",
    )
    _require(
        declared == Path(_APPROVED_REPRESENTATION_LOCATOR)
        and declared.parent == _CANONICAL_REPRESENTATION_PARENT,
        "representation adapter root is not the unique approved outcome",
    )
    sealed = _require_sealed_tree(Path(root), role="unified representation authority")
    _require_declared_or_fixed_mount(
        requested_root=Path(root),
        sealed_root=sealed,
        declared_host_root=declared,
        fixed_container_mount=_CONTAINER_REPRESENTATION_ROOT,
        role="representation authority",
    )
    _require(
        not (sealed / "audit/failure.receipt.json").exists()
        and not sealed.with_name(f"{sealed.name}.failure.receipt.json").exists(),
        "representation adapter has a failure receipt",
    )
    gate, gate_sha, _ = _read_json_object(
        sealed / "gate-output/representation-gate.json",
        role="unified representation gate",
    )
    authorization, authorization_sha, _ = _read_json_object(
        sealed / "gate-output/cycleback-input.authorization.json",
        role="unified representation authorization",
    )
    receipt, receipt_sha, _ = _read_json_object(
        sealed / "audit/run.receipt.json",
        role="unified representation run receipt",
    )
    snapshot_payload, snapshot_sha, _ = _read_json_object(
        sealed / "output/pose-cache-set.snapshot.json",
        role="unified representation snapshot",
    )
    _, segment_index_sha, _ = _read_json_object(
        sealed / "output/segment-index.json",
        role="unified representation segment index",
    )
    segment_policy, segment_policy_sha, _ = _read_json_object(
        sealed / "output/segment-reset.policy.json",
        role="unified representation segment/reset policy",
    )
    cache_validation, cache_validation_sha, _ = _read_json_object(
        sealed / "output/pose-cache-validation.receipt.json",
        role="unified-2D cache validation receipt",
    )
    joint_snapshot_path = sealed / "output/joint-mask-set.snapshot.json"
    joint_snapshot, joint_snapshot_sha, _ = _load_v4e_joint_mask_snapshot(
        joint_snapshot_path
    )
    identity_map_path = sealed / "output/identity-map.json"
    _, identity_map_sha, _ = _read_json_object(
        identity_map_path,
        role="unified-2D identity map",
    )
    pair_eligibility_path = sealed / "output/cycleback-pair-eligibility.json"
    _, pair_eligibility_sha, _ = _read_json_object(
        pair_eligibility_path,
        role="cycleback pair eligibility",
    )
    reservation_path = sealed / "attempt.reservation.json"
    reservation, reservation_sha, _ = _read_json_object(
        reservation_path, role="unified representation reservation"
    )
    _require(
        reservation_sha == _APPROVED_REPRESENTATION_RESERVATION_SHA256,
        "representation reservation differs from the preregistered bytes",
    )
    _require(
        gate_sha == _APPROVED_REPRESENTATION_GATE_SHA256,
        "representation gate differs from the preregistered bytes",
    )
    _require(
        authorization_sha == _APPROVED_REPRESENTATION_AUTHORIZATION_SHA256,
        "representation authorization differs from the preregistered bytes",
    )
    _require(
        receipt_sha == _APPROVED_REPRESENTATION_RUN_RECEIPT_SHA256,
        "representation run receipt differs from the preregistered bytes",
    )
    _require(
        snapshot_sha == _APPROVED_REPRESENTATION_SNAPSHOT_SHA256,
        "representation snapshot differs from the preregistered bytes",
    )
    _require(
        set(reservation)
        == {
            "schema_version",
            "artifact_type",
            "reservation_id",
            "scope",
            "family",
            "final_outcome_root",
            "authority_granted",
        }
        and reservation["schema_version"] == 1
        and reservation["artifact_type"]
        == "pams_pose_recovery_v4e_canonical_outcome_reservation_v1"
        and reservation["scope"] == "single-use-v4e-unified2d-representation-outcome"
        and reservation["family"] == "pams-cycleback-unified-2d-representation-v1"
        and isinstance(reservation["reservation_id"], str)
        and _is_sha256(reservation["reservation_id"])
        and reservation["reservation_id"] == sealed.name
        and reservation["final_outcome_root"] == os.fspath(declared)
        and reservation["authority_granted"] is False,
        "representation reservation identity/scope mismatch",
    )
    _require(
        gate.get("artifact_type") == _REPRESENTATION_GATE_TYPE
        and gate.get("schema_version") == 1
        and gate.get("status") == "passed"
        and gate.get("overall_pass") is True,
        "unified representation gate did not pass",
    )
    _require(
        gate.get("label_free") is True
        and gate.get("representation") == "unified_2d"
        and gate.get("representation_detail")
        == {
            "joints": "coco17",
            "coordinates": "body-centered-uniform-rms-scale-xy-z0-v1",
            "temporal": "native",
        }
        and gate.get("cycleback_representation_input_authorized") is True,
        "representation gate scope mismatch",
    )
    _require(
        gate.get("raw_full337_training_gate_passed") is False
        and gate.get("raw_full337_training_authorized") is False
        and gate.get("raw_full337_used_as_extraction_evidence_only") is True,
        "representation gate illegally upgrades denied raw full337",
    )
    diagnostic = gate.get("diagnostic_evidence")
    expected_denied_diagnostic = {
        "locator": _RAW_FULL337_DENIED_LOCATOR,
        "gate_sha256": _RAW_FULL337_GATE_SHA256,
        "gate_bytes": _RAW_FULL337_GATE_BYTES,
        "training_denial_sha256": _RAW_FULL337_DENIAL_SHA256,
        "training_denial_bytes": _RAW_FULL337_DENIAL_BYTES,
        "run_receipt_sha256": _RAW_FULL337_RUN_SHA256,
        "run_receipt_bytes": _RAW_FULL337_RUN_BYTES,
        "ledger_sha256": _RAW_FULL337_LEDGER_SHA256,
        "ledger_bytes": _RAW_FULL337_LEDGER_BYTES,
        "cache_set_sha256": _RAW_FULL337_CACHE_SET_SHA256,
        "not_consumed": True,
        "training_authority": False,
    }
    _require(
        diagnostic == expected_denied_diagnostic,
        "representation diagnostic does not exactly quarantine denied v4d evidence",
    )
    criteria = gate.get("criteria")
    _require(isinstance(criteria, Mapping) and criteria, "representation criteria missing")
    _require(
        all(
            isinstance(item, Mapping) and item.get("passed") is True
            for item in criteria.values()
        ),
        "representation gate has a failed criterion",
    )
    _require(
        authorization.get("artifact_type") == _REPRESENTATION_AUTH_TYPE
        and authorization.get("schema_version") == 1
        and authorization.get("status") == "passed"
        and authorization.get("cycleback_representation_input_authorized") is True,
        "unified representation authorization did not pass",
    )
    _require(
        authorization.get("authorization_scope")
        == "cycleback_unified_2d_train337_input_only"
        and authorization.get("label_free") is True
        and authorization.get("representation") == "unified_2d"
        and authorization.get("representation_detail")
        == gate.get("representation_detail")
        and authorization.get("raw_full337_training_authority_accepted") is False,
        "unified representation authorization scope mismatch",
    )
    authorized_consumer = authorization.get("authorized_cycleback_consumer")
    _require(
        isinstance(authorized_consumer, Mapping)
        and set(authorized_consumer)
        == {"source_revision", "source_tree_sha", "container_image_id", "configs"},
        "representation integration outcome lacks an exact cycleback consumer tuple",
    )
    approved_consumer = {
        "source_revision": _APPROVED_CYCLEBACK_SOURCE_REVISION,
        "source_tree_sha": _APPROVED_CYCLEBACK_SOURCE_TREE_SHA,
        "container_image_id": _CYCLEBACK_IMAGE_ID,
        "configs": {
            key: dict(value) for key, value in _APPROVED_CYCLEBACK_CONFIGS.items()
        },
    }
    _require(
        _is_git_sha(_APPROVED_CYCLEBACK_SOURCE_REVISION)
        and _is_git_sha(_APPROVED_CYCLEBACK_SOURCE_TREE_SHA)
        and set(_APPROVED_CYCLEBACK_CONFIGS) == set(_CYCLEBACK_CONFIG_PATHS)
        and dict(authorized_consumer) == approved_consumer
        and gate.get("authorized_cycleback_consumer") == approved_consumer
        and receipt.get("authorized_cycleback_consumer") == approved_consumer,
        "cycleback consumer tuple is not independently activated",
    )
    _require(
        receipt.get("artifact_type") == _REPRESENTATION_RECEIPT_TYPE
        and receipt.get("schema_version") == 1
        and receipt.get("status") == "passed"
        and receipt.get("container_exit_code") == 0,
        "unified representation run receipt did not pass",
    )
    snapshot = _snapshot_from_mapping(
        snapshot_payload, role="unified representation snapshot"
    )
    _require(len(snapshot.entries) == 337, "representation snapshot must contain 337 caches")
    identity_map, _ = load_identity_map(
        identity_map_path,
        expected_video_ids=[entry.video_id for entry in snapshot.entries],
        expected_sha256=identity_map_sha,
    )
    pair_eligibility = load_pair_eligibility(
        pair_eligibility_path,
        expected_sha256=pair_eligibility_sha,
        identity_map=identity_map,
    )
    load_segment_index(
        sealed / "output/segment-index.json",
        expected_video_ids=[entry.video_id for entry in snapshot.entries],
        expected_sha256=segment_index_sha,
        expected_segment_reset_policy_sha256=segment_policy_sha,
        identity_map=identity_map,
        pair_eligibility=pair_eligibility,
    )
    _require(
        len(joint_snapshot.entries) == 337
        and tuple(entry.video_id for entry in joint_snapshot.entries)
        == tuple(entry.video_id for entry in snapshot.entries),
        "unified-2D joint-mask snapshot differs from pose membership",
    )
    _require(
        set(segment_policy)
        == {
            "schema_version",
            "artifact_type",
            "association_bridge_rule",
            "maximum_bridge_gap_seconds",
            "maximum_bridge_gap_frame_cap",
            "reset_identity_semantics",
            "cross_reset_count_aggregation",
            "trainable_segment_rule",
            "cycleback_pair_rule",
            "missing_frame_bridge_for_training",
            "label_free",
        }
        and segment_policy["schema_version"] == 1
        and segment_policy["artifact_type"]
        == "pams_pose_recovery_v4e_segment_reset_policy_v1"
        and segment_policy["association_bridge_rule"]
        == "min(frame_cap,floor(seconds*fps))-zero-allowed-v1"
        and isinstance(segment_policy["maximum_bridge_gap_seconds"], float)
        and segment_policy["maximum_bridge_gap_seconds"] >= 0.0
        and isinstance(segment_policy["maximum_bridge_gap_frame_cap"], int)
        and not isinstance(segment_policy["maximum_bridge_gap_frame_cap"], bool)
        and segment_policy["maximum_bridge_gap_frame_cap"] >= 0
        and segment_policy["reset_identity_semantics"]
        == "each-association-segment-is-an-independent-pseudotrack"
        and segment_policy["cross_reset_count_aggregation"]
        == "undefined-requires-separate-preregistered-policy"
        and segment_policy["trainable_segment_rule"]
        == "exact-prevalidated-2W-pair-spans-only-v1"
        and segment_policy["cycleback_pair_rule"]
        == "consume-only-listed-start-stop-without-expansion"
        and segment_policy["missing_frame_bridge_for_training"] is False
        and segment_policy["label_free"] is True,
        "representation segment/reset policy mismatch",
    )
    _require(
        cache_validation.get("schema_version") == 1
        and cache_validation.get("artifact_type")
        == "pams_cycleback_unified_2d_pose_cache_validation_receipt_v1"
        and cache_validation.get("status") == "passed"
        and cache_validation.get("label_free") is True
        and cache_validation.get("entry_count") == 337
        and cache_validation.get("pose_fingerprint") == snapshot.pose_fingerprint
        and cache_validation.get("pose_cache_set_sha256") == snapshot.fingerprint
        and cache_validation.get("joint_mask_snapshot_sha256") == joint_snapshot_sha
        and cache_validation.get("joint_mask_set_sha256") == joint_snapshot.fingerprint
        and cache_validation.get("identity_map_sha256") == identity_map_sha
        and cache_validation.get("cycleback_pair_eligibility_sha256")
        == pair_eligibility_sha
        and cache_validation.get("segment_index_sha256") == segment_index_sha
        and cache_validation.get("segment_reset_policy_sha256") == segment_policy_sha,
        "unified-2D cache validator receipt identity mismatch",
    )
    _require(
        cache_validation.get("normalization_contract")
        == {
            "joint_valid_source": "raw_keypoint_logit_strictly_greater_than_2",
            "body_center": "coco17_hips_11_12_midpoint",
            "scale": "unweighted_rms_of_selected_reliable_xy_about_body_center",
            "xy_scale_shared": True,
            "validation_absolute_tolerance": 1e-5,
            "masked_joint_value": "exact_zero",
            "z_value": "exact_zero",
            "padded_joints_17_to_32": "exact_zero",
        },
        "unified-2D cache normalization contract mismatch",
    )
    cache_criteria = cache_validation.get("criteria")
    _require(
        isinstance(cache_criteria, Mapping)
        and set(cache_criteria)
        == {
            "finite_valid_coordinates",
            "invalid_frames_exact_zero",
            "padded_joints_17_to_32_exact_zero",
            "z_coordinates_exact_zero",
            "active_coco17_body_centered",
            "active_coco17_uniform_rms_scale",
            "mask_and_native_timeline_closed",
            "cache_sha_and_bytes_closed",
        }
        and all(
            isinstance(item, Mapping) and item.get("passed") is True
            for item in cache_criteria.values()
        ),
        "unified-2D cache validator receipt has an incomplete/failed criterion",
    )
    bindings = _binding_object(authorization, role="unified representation authorization")
    required_binding_keys = {
        "predecessor_registry_reservation_sha256",
        "predecessor_registry_receipt_sha256",
        "train337_identity_sha256",
        "train337_sidecar_sha256",
        "train337_commitment_sha256",
        "source_tree_sha256",
        "representation_gate_sha256",
        "representation_config_sha256",
        "representation_model_sha256",
        "eligible_ledger_sha256",
        "quarantine_ledger_sha256",
        "representation_snapshot_sha256",
        "segment_index_sha256",
        "segment_reset_policy_sha256",
        "unified_2d_cache_validation_receipt_sha256",
        "joint_mask_snapshot_sha256",
        "joint_mask_set_sha256",
        "identity_map_sha256",
        "cycleback_pair_eligibility_sha256",
        "representation_cache_set_sha256",
        "representation_pose_fingerprint",
    }
    _require(
        required_binding_keys.issubset(bindings)
        and all(_is_sha256(value) for value in bindings.values()),
        "representation binding schema or digest is invalid",
    )
    _require(
        bindings["representation_gate_sha256"] == gate_sha
        and bindings["representation_snapshot_sha256"] == snapshot_sha
        and bindings["segment_index_sha256"] == segment_index_sha
        and bindings["segment_reset_policy_sha256"] == segment_policy_sha
        and bindings["unified_2d_cache_validation_receipt_sha256"]
        == cache_validation_sha
        and bindings["joint_mask_snapshot_sha256"] == joint_snapshot_sha
        and bindings["joint_mask_set_sha256"] == joint_snapshot.fingerprint
        and bindings["identity_map_sha256"] == identity_map_sha
        and bindings["cycleback_pair_eligibility_sha256"] == pair_eligibility_sha
        and bindings["representation_cache_set_sha256"] == snapshot.fingerprint
        and bindings["representation_pose_fingerprint"] == snapshot.pose_fingerprint,
        "representation output binding mismatch",
    )
    _require(
        receipt.get("representation_gate_sha256") == gate_sha
        and receipt.get("representation_authorization_sha256") == authorization_sha
        and receipt.get("representation_snapshot_sha256") == snapshot_sha
        and receipt.get("segment_index_sha256") == segment_index_sha
        and receipt.get("segment_reset_policy_sha256") == segment_policy_sha
        and receipt.get("unified_2d_cache_validation_receipt_sha256")
        == cache_validation_sha
        and receipt.get("joint_mask_snapshot_sha256") == joint_snapshot_sha
        and receipt.get("joint_mask_set_sha256") == joint_snapshot.fingerprint
        and receipt.get("identity_map_sha256") == identity_map_sha
        and receipt.get("cycleback_pair_eligibility_sha256") == pair_eligibility_sha
        and receipt.get("bindings") == bindings,
        "representation receipt byte lineage mismatch",
    )
    source_revision = receipt.get("source_git_sha")
    image_id = receipt.get("container_image_id")
    _require(
        isinstance(source_revision, str)
        and len(source_revision) == 40
        and all(character in "0123456789abcdef" for character in source_revision),
        "representation source revision is invalid",
    )
    _require(
        isinstance(image_id, str) and image_id.startswith("sha256:")
        and _is_sha256(image_id[7:]),
        "representation image ID is invalid",
    )
    _require(
        authorization.get("source_git_sha") == source_revision
        and authorization.get("container_image_id") == image_id
        and gate.get("source_git_sha") == source_revision
        and gate.get("container_image_id") == image_id,
        "representation source/image lineage mismatch",
    )
    for payload, role in ((gate, "gate"), (authorization, "authorization"), (receipt, "receipt")):
        for field in (
            "scientific_authority_granted",
            "baseline_training_authorized",
            "full_training_authorized",
            "epoch11_encoder_continuation_authorized",
            "development_evaluation_authorized",
            "sealed_evaluation_authorized",
        ):
            _require(payload.get(field) is False, f"representation {role} illegally sets {field}")
    cache_dir = sealed / "output/pose-cache"
    _require(cache_dir.is_dir() and not cache_dir.is_symlink(), "representation cache missing")
    joint_mask_dir = sealed / "output/joint-mask"
    _require(
        joint_mask_dir.is_dir() and not joint_mask_dir.is_symlink(),
        "representation joint-mask sidecars missing",
    )
    return UnifiedRepresentationAuthority(
        root=sealed,
        source_revision=source_revision,
        container_image_id=image_id,
        gate_sha256=gate_sha,
        authorization_sha256=authorization_sha,
        run_receipt_sha256=receipt_sha,
        snapshot_sha256=snapshot_sha,
        segment_index_path=sealed / "output/segment-index.json",
        segment_index_sha256=segment_index_sha,
        segment_reset_policy_sha256=segment_policy_sha,
        joint_mask_dir=joint_mask_dir,
        joint_mask_snapshot_path=joint_snapshot_path,
        joint_mask_snapshot_sha256=joint_snapshot_sha,
        joint_mask_snapshot=joint_snapshot,
        identity_map_path=identity_map_path,
        identity_map_sha256=identity_map_sha,
        pair_eligibility_path=pair_eligibility_path,
        pair_eligibility_sha256=pair_eligibility_sha,
        pose_fingerprint=snapshot.pose_fingerprint,
        cache_set_sha256=snapshot.fingerprint,
        snapshot=snapshot,
        bindings=dict(bindings),
        authorized_cycleback_consumer=dict(authorized_consumer),
    )


def reserve_cycleback_launch_registry(
    *,
    repository_root: str | Path,
    output_root: str | Path = _CANONICAL_LAUNCH_REGISTRY_ROOT,
) -> CycleBackLaunchAuthority:
    """Preregister the only source/image/config/representation launch tuple."""

    _require(
        _is_git_sha(_APPROVED_CYCLEBACK_SOURCE_REVISION)
        and _is_git_sha(_APPROVED_CYCLEBACK_SOURCE_TREE_SHA)
        and set(_APPROVED_CYCLEBACK_CONFIGS) == set(_CYCLEBACK_CONFIG_PATHS),
        "cycleback consumer source/config tuple has not been independently activated",
    )
    repository = Path(repository_root).resolve(strict=True)
    _require(
        repository.is_dir() and not repository.is_symlink(),
        "launch registry repository must be a non-symlink directory",
    )
    source_revision = _git_text(repository, "rev-parse", "HEAD^{commit}")
    source_tree_sha = _git_text(repository, "rev-parse", "HEAD^{tree}")
    _require(
        source_revision == _APPROVED_CYCLEBACK_SOURCE_REVISION,
        "launch source revision differs from the independently activated consumer",
    )
    _require(
        source_tree_sha == _APPROVED_CYCLEBACK_SOURCE_TREE_SHA,
        "launch source tree differs from the independently activated consumer",
    )
    _require(
        not _git_bytes(repository, "status", "--porcelain", "--untracked-files=all"),
        "launch registry requires a clean exact source checkout",
    )
    configs: dict[str, dict[str, Any]] = {}
    for relative in _CYCLEBACK_CONFIG_PATHS:
        blob = _git_bytes(repository, "show", f"{source_revision}:{relative}")
        working, working_sha, working_bytes = _read_regular_bytes(
            repository / relative,
            role=f"launch config {relative}",
        )
        _require(working == blob, f"working config differs from Git object: {relative}")
        configs[relative] = {
            "sha256": working_sha,
            "bytes": working_bytes,
        }
    _require(
        configs
        == {key: dict(value) for key, value in _APPROVED_CYCLEBACK_CONFIGS.items()},
        "launch configs differ from the independently activated consumer tuple",
    )
    synthetic_path, synthetic_sha, synthetic_bytes = (
        _validate_synthetic_preregistration(configs=configs)
    )

    _require(
        bool(_APPROVED_REPRESENTATION_LOCATOR),
        "launch registry is blocked until a representation outcome is preregistered",
    )
    declared_representation = Path(_APPROVED_REPRESENTATION_LOCATOR).resolve(
        strict=True
    )
    representation = validate_unified_representation_authority(
        declared_representation,
        declared_host_root=declared_representation,
    )
    _require(
        representation.authorized_cycleback_consumer
        == {
            "source_revision": source_revision,
            "source_tree_sha": source_tree_sha,
            "container_image_id": _CYCLEBACK_IMAGE_ID,
            "configs": configs,
        },
        "representation integration outcome authorizes a different cycleback consumer",
    )
    representation_bindings = {
        **dict(representation.bindings),
        "representation_source_revision": representation.source_revision,
        "representation_container_image_sha256": representation.container_image_id[7:],
        "representation_authorization_sha256": representation.authorization_sha256,
        "representation_run_receipt_sha256": representation.run_receipt_sha256,
    }
    identity = {
        "source_revision": source_revision,
        "source_tree_sha": source_tree_sha,
        "container_image_id": _CYCLEBACK_IMAGE_ID,
        "configs": configs,
        "representation_root": os.fspath(declared_representation),
        "representation_bindings": representation_bindings,
        "synthetic_preregistration": {
            "sha256": synthetic_sha,
            "bytes": synthetic_bytes,
        },
    }
    registry_id = _canonical_json_sha256(identity)
    authorization = {
        "schema_version": 1,
        "artifact_type": _LAUNCH_AUTH_TYPE,
        "registry_id": registry_id,
        "identity": identity,
        "label_free": True,
        "authorized_stages": [
            "pose_input_adapter",
            "geometry_gate",
            "mechanism_probe_256_steps",
        ],
        "caller_source_override_allowed": False,
        "caller_image_override_allowed": False,
        "caller_config_digest_override_allowed": False,
        "scientific_authority_granted": False,
        "baseline_training_authorized": False,
        "full_training_authorized": False,
        "epoch11_encoder_continuation_authorized": False,
        "development_evaluation_authorized": False,
        "sealed_evaluation_authorized": False,
    }

    root = Path(output_root)
    _require(root.is_absolute(), "launch registry output root must be absolute")
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    _require(not root.exists(), "launch registry already exists")
    reservation_path = root.with_name(f"{root.name}.reservation.json")
    reservation = {
        "schema_version": 1,
        "artifact_type": "pams_conventional_cycleback_launch_registry_reservation_v1",
        "registry_id": registry_id,
        "source_revision": source_revision,
        "source_tree_sha": source_tree_sha,
        "container_image_id": _CYCLEBACK_IMAGE_ID,
        "final_registry_root": os.fspath(root),
        "authority_granted": False,
    }
    write_json_exclusive(reservation_path, reservation)
    reservation_sha = _read_regular_bytes(
        reservation_path, role="launch registry reservation"
    )[1]
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.incomplete.", dir=os.fspath(parent))
    )
    try:
        authorization_path = staging / "launch.authorization.json"
        reservation_archive_path = staging / "registry.reservation.json"
        synthetic_archive_path = staging / "synthetic-preregistration.json"
        _copy_cache_exclusive(reservation_path, reservation_archive_path)
        _copy_cache_exclusive(synthetic_path, synthetic_archive_path)
        write_json_exclusive(authorization_path, authorization)
        authorization_sha = _read_regular_bytes(
            authorization_path, role="launch authorization"
        )[1]
        receipt = {
            "schema_version": 1,
            "artifact_type": _LAUNCH_RECEIPT_TYPE,
            "status": "reserved",
            "registry_id": registry_id,
            "source_revision": source_revision,
            "source_tree_sha": source_tree_sha,
            "container_image_id": _CYCLEBACK_IMAGE_ID,
            "launch_authorization_sha256": authorization_sha,
            "reservation_sha256": reservation_sha,
            "reservation_archive_sha256": reservation_sha,
            "representation_bindings": representation_bindings,
            "synthetic_preregistration_sha256": synthetic_sha,
            "scientific_authority_granted": False,
            "baseline_training_authorized": False,
            "full_training_authorized": False,
            "epoch11_encoder_continuation_authorized": False,
            "development_evaluation_authorized": False,
            "sealed_evaluation_authorized": False,
        }
        write_json_exclusive(staging / "registry.receipt.json", receipt)
        for candidate in staging.rglob("*"):
            candidate.chmod(0o444 if candidate.is_file() else 0o555)
        staging.chmod(0o555)
        os.rename(staging, root)
    except BaseException:
        if staging.exists():
            staging.chmod(0o700)
        raise
    reservation_path.chmod(0o444)
    return validate_cycleback_launch_registry(
        root,
        declared_host_root=root,
        source_root=repository,
    )


def validate_cycleback_launch_registry(
    root: str | Path,
    *,
    declared_host_root: str | Path,
    source_root: str | Path | None = None,
) -> CycleBackLaunchAuthority:
    """Validate the single canonical registry and every frozen launch byte."""

    _require(
        _is_git_sha(_APPROVED_CYCLEBACK_SOURCE_REVISION)
        and _is_git_sha(_APPROVED_CYCLEBACK_SOURCE_TREE_SHA)
        and set(_APPROVED_CYCLEBACK_CONFIGS) == set(_CYCLEBACK_CONFIG_PATHS),
        "cycleback launch registry is disabled pending an independent activation commit",
    )
    declared = Path(declared_host_root)
    _require(
        declared == _CANONICAL_LAUNCH_REGISTRY_ROOT,
        "launch registry is outside the canonical audited namespace",
    )
    sealed = _require_sealed_tree(Path(root), role="cycleback launch registry")
    _require_declared_or_fixed_mount(
        requested_root=Path(root),
        sealed_root=sealed,
        declared_host_root=declared,
        fixed_container_mount=_CONTAINER_LAUNCH_REGISTRY_ROOT,
        role="launch registry",
    )
    authorization_path = sealed / "launch.authorization.json"
    receipt_path = sealed / "registry.receipt.json"
    authorization, authorization_sha, _ = _read_json_object(
        authorization_path, role="cycleback launch authorization"
    )
    receipt, receipt_sha, _ = _read_json_object(
        receipt_path, role="cycleback launch registry receipt"
    )
    reservation_path = sealed / "registry.reservation.json"
    reservation, reservation_sha, _ = _read_json_object(
        reservation_path, role="cycleback launch registry reservation"
    )
    reservation_mode = reservation_path.lstat().st_mode
    _require(reservation_mode & 0o222 == 0, "launch registry reservation is writable")
    _require(
        authorization.get("artifact_type") == _LAUNCH_AUTH_TYPE
        and authorization.get("schema_version") == 1,
        "wrong cycleback launch authorization type",
    )
    _require(
        receipt.get("artifact_type") == _LAUNCH_RECEIPT_TYPE
        and receipt.get("schema_version") == 1
        and receipt.get("status") == "reserved",
        "wrong cycleback launch registry receipt",
    )
    identity = authorization.get("identity")
    _require(isinstance(identity, Mapping), "launch registry identity is missing")
    registry_id = _canonical_json_sha256(identity)
    _require(
        authorization.get("registry_id") == registry_id
        and receipt.get("registry_id") == registry_id
        and reservation.get("registry_id") == registry_id,
        "launch registry identity mismatch",
    )
    _require(
        receipt.get("launch_authorization_sha256") == authorization_sha
        and receipt.get("reservation_sha256") == reservation_sha
        and receipt.get("reservation_archive_sha256") == reservation_sha,
        "launch registry receipt byte binding mismatch",
    )
    source_revision = identity.get("source_revision")
    source_tree_sha = identity.get("source_tree_sha")
    image_id = identity.get("container_image_id")
    _require(
        source_revision == _APPROVED_CYCLEBACK_SOURCE_REVISION,
        "launch registry source revision is not independently activated",
    )
    _require(
        source_tree_sha == _APPROVED_CYCLEBACK_SOURCE_TREE_SHA,
        "launch registry source tree is not independently activated",
    )
    _require(image_id == _CYCLEBACK_IMAGE_ID, "launch registry image is not audited")
    configs = identity.get("configs")
    _require(
        isinstance(configs, Mapping) and set(configs) == set(_CYCLEBACK_CONFIG_PATHS),
        "launch registry config matrix mismatch",
    )
    _require(
        {key: dict(value) for key, value in configs.items()}
        == {key: dict(value) for key, value in _APPROVED_CYCLEBACK_CONFIGS.items()},
        "launch registry configs are not independently activated",
    )
    for relative in _CYCLEBACK_CONFIG_PATHS:
        row = configs[relative]
        _require(
            isinstance(row, Mapping)
            and set(row) == {"sha256", "bytes"}
            and _is_sha256(row.get("sha256"))
            and isinstance(row.get("bytes"), int)
            and row["bytes"] > 0,
            f"launch registry config identity is malformed: {relative}",
        )
        if source_root is not None:
            _, digest, byte_count = _read_regular_bytes(
                Path(source_root) / relative,
                role=f"frozen launch config {relative}",
            )
            _require(
                digest == row["sha256"] and byte_count == row["bytes"],
                f"frozen launch config differs from registry: {relative}",
            )
    synthetic = identity.get("synthetic_preregistration")
    synthetic_payload, synthetic_sha, synthetic_bytes = _read_json_object(
        sealed / "synthetic-preregistration.json",
        role="archived synthetic cycle-back preregistration",
    )
    _require(
        isinstance(synthetic, Mapping)
        and synthetic == {"sha256": synthetic_sha, "bytes": synthetic_bytes}
        and synthetic_sha == _APPROVED_SYNTHETIC_PREREG_SHA256
        and receipt.get("synthetic_preregistration_sha256") == synthetic_sha,
        "launch registry synthetic preregistration byte binding mismatch",
    )
    _require(
        synthetic_payload.get("artifact_type")
        == "pams_conventional_cycleback_synthetic_preregistration_v1"
        and synthetic_payload.get("status") == "passed"
        and synthetic_payload.get("candidate_configs")
        == {key: dict(value) for key, value in configs.items()}
        and synthetic_payload.get("thresholds_frozen_before_train337") is True
        and synthetic_payload.get("thresholds_achievable_on_synthetic_cycles") is True
        and synthetic_payload.get("joint_support_nulls_evaluable") is True,
        "archived synthetic preregistration contract mismatch",
    )
    representation_root = identity.get("representation_root")
    representation_bindings = identity.get("representation_bindings")
    _require(
        isinstance(representation_root, str)
        and Path(representation_root).parent == _CANONICAL_REPRESENTATION_PARENT,
        "launch registry representation root is not canonical",
    )
    _require(
        isinstance(representation_bindings, Mapping),
        "launch representation bindings missing",
    )
    expected_representation_keys = {
        "predecessor_registry_reservation_sha256",
        "predecessor_registry_receipt_sha256",
        "train337_identity_sha256",
        "train337_sidecar_sha256",
        "train337_commitment_sha256",
        "source_tree_sha256",
        "representation_gate_sha256",
        "representation_config_sha256",
        "representation_model_sha256",
        "eligible_ledger_sha256",
        "quarantine_ledger_sha256",
        "representation_snapshot_sha256",
        "segment_index_sha256",
        "segment_reset_policy_sha256",
        "unified_2d_cache_validation_receipt_sha256",
        "joint_mask_snapshot_sha256",
        "joint_mask_set_sha256",
        "identity_map_sha256",
        "cycleback_pair_eligibility_sha256",
        "representation_cache_set_sha256",
        "representation_pose_fingerprint",
        "representation_source_revision",
        "representation_container_image_sha256",
        "representation_authorization_sha256",
        "representation_run_receipt_sha256",
    }
    _require(
        expected_representation_keys.issubset(representation_bindings),
        "launch representation binding schema mismatch",
    )
    for key, value in representation_bindings.items():
        if key == "representation_source_revision":
            continue
        _require(
            _is_sha256(value),
            f"invalid launch representation binding: {key}",
        )
    representation_source = representation_bindings.get("representation_source_revision")
    _require(
        isinstance(representation_source, str)
        and len(representation_source) == 40
        and all(character in "0123456789abcdef" for character in representation_source),
        "invalid representation source revision binding",
    )
    _require(
        receipt.get("source_revision") == source_revision
        and receipt.get("source_tree_sha") == source_tree_sha
        and receipt.get("container_image_id") == image_id
        and receipt.get("representation_bindings") == representation_bindings,
        "launch registry receipt lineage mismatch",
    )
    _require(
        reservation.get("source_revision") == source_revision
        and reservation.get("source_tree_sha") == source_tree_sha
        and reservation.get("container_image_id") == image_id
        and reservation.get("final_registry_root") == os.fspath(declared),
        "launch registry reservation lineage mismatch",
    )
    for payload, role in (
        (authorization, "authorization"),
        (receipt, "receipt"),
    ):
        for field in (
            "scientific_authority_granted",
            "baseline_training_authorized",
            "full_training_authorized",
            "epoch11_encoder_continuation_authorized",
            "development_evaluation_authorized",
            "sealed_evaluation_authorized",
        ):
            _require(payload.get(field) is False, f"launch {role} illegally sets {field}")
    _require(
        authorization.get("caller_source_override_allowed") is False
        and authorization.get("caller_image_override_allowed") is False
        and authorization.get("caller_config_digest_override_allowed") is False,
        "launch registry permits a caller identity override",
    )
    return CycleBackLaunchAuthority(
        root=sealed,
        authorization_path=authorization_path,
        receipt_path=receipt_path,
        authorization_sha256=authorization_sha,
        receipt_sha256=receipt_sha,
        registry_id=registry_id,
        source_revision=source_revision,
        source_tree_sha=source_tree_sha,
        container_image_id=image_id,
        configs={key: dict(value) for key, value in configs.items()},
        representation_root=representation_root,
        representation_bindings=dict(representation_bindings),
    )


def materialize_cycleback_pose_authority(
    authority: UnifiedRepresentationAuthority,
    *,
    output_root: str | Path,
    launch_authority: CycleBackLaunchAuthority,
) -> dict[str, Any]:
    """Copy and independently verify all authorized cache bytes once."""

    expected_launch_bindings = {
        **dict(authority.bindings),
        "representation_source_revision": authority.source_revision,
        "representation_container_image_sha256": authority.container_image_id[7:],
        "representation_authorization_sha256": authority.authorization_sha256,
        "representation_run_receipt_sha256": authority.run_receipt_sha256,
    }
    _require(
        all(
            launch_authority.representation_bindings.get(key) == value
            for key, value in expected_launch_bindings.items()
        ),
        "launch registry and validated representation authority differ",
    )

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    details = root.lstat()
    _require(
        stat.S_ISDIR(details.st_mode) and not root.is_symlink(),
        "adapter output root must be a non-symlink directory",
    )
    _require(not any(root.iterdir()), "adapter output root must be empty")
    cache_output = root / "pose-cache"
    joint_mask_output = root / "joint-mask"
    output_dir = root / "output"
    authorization_dir = root / "authorization"
    cache_output.mkdir()
    joint_mask_output.mkdir()
    output_dir.mkdir()
    authorization_dir.mkdir()
    observed: list[PoseCacheEntryReceipt] = []
    source_cache = authority.root / "output/pose-cache"
    source_joint_masks = authority.joint_mask_dir
    joint_mask_by_id = {
        entry.video_id: entry for entry in authority.joint_mask_snapshot.entries
    }
    _, observed_segment_sha, _ = _read_json_object(
        authority.segment_index_path,
        role="authorized representation segment index",
    )
    _require(
        observed_segment_sha == authority.segment_index_sha256,
        "representation segment index changed before materialization",
    )
    identity_map, _ = load_identity_map(
        authority.identity_map_path,
        expected_video_ids=[entry.video_id for entry in authority.snapshot.entries],
        expected_sha256=authority.identity_map_sha256,
    )
    pair_eligibility = load_pair_eligibility(
        authority.pair_eligibility_path,
        expected_sha256=authority.pair_eligibility_sha256,
        identity_map=identity_map,
    )
    segment_rows, _ = load_segment_index(
        authority.segment_index_path,
        expected_video_ids=[entry.video_id for entry in authority.snapshot.entries],
        expected_sha256=authority.segment_index_sha256,
        expected_segment_reset_policy_sha256=(
            authority.segment_reset_policy_sha256
        ),
        identity_map=identity_map,
        pair_eligibility=pair_eligibility,
    )
    for expected in authority.snapshot.entries:
        source = pose_cache_path(source_cache, expected.video_id)
        destination = pose_cache_path(cache_output, expected.video_id)
        _copy_cache_exclusive(source, destination)
        joint_source = joint_mask_path(source_joint_masks, expected.video_id)
        joint_destination = joint_mask_path(joint_mask_output, expected.video_id)
        _copy_cache_exclusive(joint_source, joint_destination)
        pose, _, receipt = load_unified_2d_pose_cache_with_receipt(
            destination,
            expected_pose_fingerprint=authority.pose_fingerprint,
        )
        joint_mask = load_joint_mask_sidecar(
            joint_destination,
            expected=joint_mask_by_id[expected.video_id],
            native_length=pose.num_frames,
        )
        sequence = validate_unified_2d_sequence(pose, joint_mask)
        _require(
            receipt == expected and sequence.video_id == expected.video_id,
            "materialized cache differs from representation authority",
        )
        _require(
            sequence.num_frames == pair_eligibility.native_lengths[expected.video_id],
            "materialized cache length differs from pair eligibility",
        )
        _require(
            all(
                0 <= start < stop <= sequence.num_frames
                for start, stop in segment_rows[expected.video_id]
            ),
            "materialized segment range lies outside the pose timeline",
        )
        observed.append(receipt)
    snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=authority.pose_fingerprint,
        entries=tuple(observed),
    )
    _require(
        snapshot == authority.snapshot,
        "materialized cache snapshot differs from representation authority",
    )
    snapshot_path = output_dir / "pose-cache-set.snapshot.json"
    write_json_exclusive(snapshot_path, snapshot.to_dict())
    snapshot_sha = _read_regular_bytes(
        snapshot_path,
        role="materialized pose snapshot",
    )[1]
    segment_index_path = output_dir / "segment-index.json"
    _copy_cache_exclusive(authority.segment_index_path, segment_index_path)
    _require(
        _read_regular_bytes(
            segment_index_path,
            role="materialized segment index",
        )[1]
        == authority.segment_index_sha256,
        "materialized segment index changed",
    )
    joint_snapshot_path = output_dir / "joint-mask-set.snapshot.json"
    normalized_joint_snapshot = JointMaskSetSnapshot(
        entries=authority.joint_mask_snapshot.entries
    )
    write_json_exclusive(joint_snapshot_path, normalized_joint_snapshot.to_dict())
    normalized_joint_snapshot_sha = _read_regular_bytes(
        joint_snapshot_path,
        role="materialized joint-mask snapshot",
    )[1]
    identity_map_path = output_dir / "identity-map.json"
    _copy_cache_exclusive(authority.identity_map_path, identity_map_path)
    pair_eligibility_path = output_dir / "cycleback-pair-eligibility.json"
    _copy_cache_exclusive(authority.pair_eligibility_path, pair_eligibility_path)
    _require(
        _read_regular_bytes(identity_map_path, role="materialized identity map")[1]
        == authority.identity_map_sha256
        and _read_regular_bytes(
            pair_eligibility_path,
            role="materialized pair eligibility",
        )[1]
        == authority.pair_eligibility_sha256,
        "materialized identity/pair-eligibility bytes changed",
    )
    bindings = expected_launch_bindings
    authorization = {
        "schema_version": 1,
        "artifact_type": _ADAPTER_AUTH_TYPE,
        "label_free": True,
        "authorization_scope": "cycleback_train337_unified_2d_input_only",
        "cycleback_pose_input_authorized": True,
        "baseline_training_authorized": False,
        "scientific_authority_granted": False,
        "full_training_authorized": False,
        "epoch11_encoder_continuation_authorized": False,
        "development_evaluation_authorized": False,
        "sealed_evaluation_authorized": False,
        "adapter_source_git_sha": launch_authority.source_revision,
        "adapter_container_image_id": launch_authority.container_image_id,
        "launch_registry_id": launch_authority.registry_id,
        "launch_authorization_sha256": launch_authority.authorization_sha256,
        "launch_registry_receipt_sha256": launch_authority.receipt_sha256,
        "upstream_source_git_sha": authority.source_revision,
        "upstream_container_image_id": authority.container_image_id,
        "pose_snapshot_sha256": snapshot_sha,
        "pose_fingerprint": snapshot.pose_fingerprint,
        "pose_cache_set_sha256": snapshot.fingerprint,
        "pose_cache_entry_count": len(snapshot.entries),
        "segment_index_sha256": authority.segment_index_sha256,
        "segment_reset_policy_sha256": authority.segment_reset_policy_sha256,
        "joint_mask_snapshot_sha256": normalized_joint_snapshot_sha,
        "joint_mask_set_sha256": normalized_joint_snapshot.fingerprint,
        "identity_map_sha256": authority.identity_map_sha256,
        "cycleback_pair_eligibility_sha256": authority.pair_eligibility_sha256,
        "label_firewall": {
            "video_id_handling": "opaque_identifier_for_hash_order_and_same_video_equality_only",
            "video_id_tokens_parsed": False,
            "action_or_repetition_annotations_accessed": False,
        },
        "bindings": bindings,
    }
    authorization_path = authorization_dir / "cycleback-pose-input.authorization.json"
    write_json_exclusive(authorization_path, authorization)
    return authorization


def validate_cycleback_pose_authority(
    root: str | Path,
    *,
    declared_host_root: str | Path,
    launch_authority: CycleBackLaunchAuthority,
    expected_authorization_sha256: str,
    expected_run_receipt_sha256: str,
) -> CycleBackPoseAuthority:
    """Validate a sealed adapter output for geometry/mechanism consumption."""

    _require(_is_sha256(expected_authorization_sha256), "adapter authorization SHA is invalid")
    _require(_is_sha256(expected_run_receipt_sha256), "adapter run-receipt SHA is invalid")
    declared = Path(declared_host_root)
    _require(
        declared.parent == _CANONICAL_ADAPTER_PARENT
        and declared.name.startswith(f"{launch_authority.source_revision[:12]}-")
        and len(declared.name) > 13,
        "pose-input adapter root is outside the canonical launch namespace",
    )
    sealed = _require_sealed_tree(Path(root), role="cycleback pose-input authority")
    _require_declared_or_fixed_mount(
        requested_root=Path(root),
        sealed_root=sealed,
        declared_host_root=declared,
        fixed_container_mount=_CONTAINER_POSE_AUTHORITY_ROOT,
        role="pose-input authority",
    )
    _require(
        not (sealed / "audit/failure.receipt.json").exists(),
        "pose-input adapter failure receipt forbids downstream use",
    )
    _require(
        not sealed.with_name(f"{sealed.name}.failure.receipt.json").exists(),
        "pose-input adapter fallback failure receipt forbids downstream use",
    )
    artifact_root = sealed / "artifact"
    authorization_path = artifact_root / "authorization/cycleback-pose-input.authorization.json"
    snapshot_path = artifact_root / "output/pose-cache-set.snapshot.json"
    segment_index_path = artifact_root / "output/segment-index.json"
    joint_snapshot_path = artifact_root / "output/joint-mask-set.snapshot.json"
    identity_map_path = artifact_root / "output/identity-map.json"
    pair_eligibility_path = artifact_root / "output/cycleback-pair-eligibility.json"
    cache_dir = artifact_root / "pose-cache"
    joint_mask_dir = artifact_root / "joint-mask"
    run_receipt_path = sealed / "audit/run.receipt.json"
    authorization, authorization_sha, _ = _read_json_object(
        authorization_path, role="cycleback pose-input authorization"
    )
    receipt, receipt_sha, _ = _read_json_object(
        run_receipt_path, role="cycleback pose-input adapter run receipt"
    )
    snapshot_payload, snapshot_sha, _ = _read_json_object(
        snapshot_path, role="cycleback pose snapshot"
    )
    _, segment_index_sha, _ = _read_json_object(
        segment_index_path,
        role="cycleback segment index",
    )
    _, joint_snapshot_sha, _ = _read_json_object(
        joint_snapshot_path,
        role="cycleback joint-mask snapshot",
    )
    joint_snapshot = load_joint_mask_snapshot(joint_snapshot_path)
    _, identity_map_sha, _ = _read_json_object(
        identity_map_path,
        role="cycleback identity map",
    )
    _, pair_eligibility_sha, _ = _read_json_object(
        pair_eligibility_path,
        role="cycleback pair eligibility",
    )
    _require(
        authorization_sha == expected_authorization_sha256,
        "adapter authorization hash mismatch",
    )
    _require(
        receipt_sha == expected_run_receipt_sha256,
        "adapter run-receipt hash mismatch",
    )
    _require(
        authorization.get("artifact_type") == _ADAPTER_AUTH_TYPE,
        "wrong adapter authority type",
    )
    _require(authorization.get("schema_version") == 1, "wrong adapter authority schema")
    _require(authorization.get("label_free") is True, "adapter authority is not label-free")
    _require(
        authorization.get("authorization_scope")
        == "cycleback_train337_unified_2d_input_only",
        "adapter scope mismatch",
    )
    _require(
        authorization.get("adapter_source_git_sha") == launch_authority.source_revision
        and authorization.get("adapter_container_image_id")
        == launch_authority.container_image_id,
        "adapter source/image differs from the launch registry",
    )
    _require(
        authorization.get("upstream_source_git_sha")
        == launch_authority.representation_bindings["representation_source_revision"]
        and authorization.get("upstream_container_image_id", "").removeprefix("sha256:")
        == launch_authority.representation_bindings[
            "representation_container_image_sha256"
        ],
        "adapter upstream source/image differs from representation authority",
    )
    _require(
        authorization.get("cycleback_pose_input_authorized") is True,
        "adapter did not authorize the cycleback pose input",
    )
    _require(
        authorization.get("launch_registry_id") == launch_authority.registry_id
        and authorization.get("launch_authorization_sha256")
        == launch_authority.authorization_sha256
        and authorization.get("launch_registry_receipt_sha256")
        == launch_authority.receipt_sha256,
        "adapter is not bound to the canonical launch registry",
    )
    for forbidden in (
        "baseline_training_authorized",
        "epoch11_encoder_continuation_authorized",
        "development_evaluation_authorized",
        "sealed_evaluation_authorized",
    ):
        _require(authorization.get(forbidden) is False, f"adapter illegally sets {forbidden}")
    snapshot = _snapshot_from_mapping(snapshot_payload, role="cycleback pose snapshot")
    _require(len(snapshot.entries) == 337, "adapter snapshot must contain exactly 337 entries")
    identity_map, _ = load_identity_map(
        identity_map_path,
        expected_video_ids=[entry.video_id for entry in snapshot.entries],
        expected_sha256=identity_map_sha,
    )
    pair_eligibility = load_pair_eligibility(
        pair_eligibility_path,
        expected_sha256=pair_eligibility_sha,
        identity_map=identity_map,
    )
    load_segment_index(
        segment_index_path,
        expected_video_ids=[entry.video_id for entry in snapshot.entries],
        expected_sha256=segment_index_sha,
        expected_segment_reset_policy_sha256=(
            authorization["segment_reset_policy_sha256"]
        ),
        identity_map=identity_map,
        pair_eligibility=pair_eligibility,
    )
    _require(
        tuple(entry.video_id for entry in joint_snapshot.entries)
        == tuple(entry.video_id for entry in snapshot.entries),
        "adapter joint-mask membership differs from pose snapshot",
    )
    _require(authorization.get("pose_snapshot_sha256") == snapshot_sha, "snapshot hash mismatch")
    _require(
        authorization.get("pose_fingerprint") == snapshot.pose_fingerprint
        and authorization.get("pose_cache_set_sha256") == snapshot.fingerprint
        and authorization.get("pose_cache_entry_count") == 337,
        "adapter snapshot binding mismatch",
    )
    _require(
        authorization.get("segment_index_sha256") == segment_index_sha
        and authorization.get("segment_reset_policy_sha256")
        == launch_authority.representation_bindings["segment_reset_policy_sha256"]
        and authorization.get("joint_mask_snapshot_sha256") == joint_snapshot_sha
        and authorization.get("joint_mask_set_sha256") == joint_snapshot.fingerprint,
        "adapter segment/joint-mask binding mismatch",
    )
    _require(
        authorization.get("identity_map_sha256") == identity_map_sha
        and authorization.get("cycleback_pair_eligibility_sha256")
        == pair_eligibility_sha,
        "adapter identity/pair-eligibility binding mismatch",
    )
    bindings = _binding_object(authorization, role="cycleback pose-input authorization")
    _require(
        bindings == launch_authority.representation_bindings,
        "adapter representation binding schema or bytes mismatch",
    )
    _require(
        bindings["representation_pose_fingerprint"] == snapshot.pose_fingerprint
        and bindings["representation_cache_set_sha256"] == snapshot.fingerprint,
        "adapter/representation pose identity mismatch",
    )
    _require(receipt.get("artifact_type") == _ADAPTER_RECEIPT_TYPE, "wrong adapter run receipt")
    _require(receipt.get("schema_version") == 1, "wrong adapter run-receipt schema")
    _require(receipt.get("status") == "passed", "adapter run did not pass")
    _require(receipt.get("container_exit_code") == 0, "adapter container did not exit cleanly")
    _require(
        receipt.get("pose_input_authorization_sha256") == authorization_sha
        and receipt.get("pose_snapshot_sha256") == snapshot_sha
        and receipt.get("pose_cache_set_sha256") == snapshot.fingerprint,
        "adapter run receipt output binding mismatch",
    )
    _require(
        receipt.get("segment_index_sha256") == segment_index_sha
        and receipt.get("joint_mask_snapshot_sha256") == joint_snapshot_sha
        and receipt.get("joint_mask_set_sha256") == joint_snapshot.fingerprint,
        "adapter run receipt segment/joint-mask output mismatch",
    )
    _require(
        receipt.get("identity_map_sha256") == identity_map_sha
        and receipt.get("cycleback_pair_eligibility_sha256") == pair_eligibility_sha,
        "adapter run receipt identity/pair-eligibility output mismatch",
    )
    _require(
        receipt.get("representation_bindings") == bindings,
        "adapter run receipt representation lineage mismatch",
    )
    _require(
        receipt.get("launch_registry_id") == launch_authority.registry_id
        and receipt.get("launch_authorization_sha256")
        == launch_authority.authorization_sha256
        and receipt.get("launch_registry_receipt_sha256")
        == launch_authority.receipt_sha256,
        "adapter run receipt launch-registry binding mismatch",
    )
    _require(
        receipt.get("source_git_sha") == authorization.get("adapter_source_git_sha")
        and receipt.get("container_image_id")
        == authorization.get("adapter_container_image_id"),
        "adapter authorization/run-receipt source or image mismatch",
    )
    for payload, role in ((authorization, "authorization"), (receipt, "run receipt")):
        for forbidden in (
            "scientific_authority_granted",
            "baseline_training_authorized",
            "full_training_authorized",
            "epoch11_encoder_continuation_authorized",
            "development_evaluation_authorized",
            "sealed_evaluation_authorized",
        ):
            _require(payload.get(forbidden) is False, f"adapter {role} illegally sets {forbidden}")
    container_name = receipt.get("container_name")
    expected_container = re.compile(
        rf"^pams-cycleback-pose-auth-{launch_authority.source_revision[:12]}-"
        r"[a-z0-9][a-z0-9._-]{0,39}$"
    )
    _require(
        isinstance(container_name, str) and expected_container.fullmatch(container_name),
        "adapter container name is not the exact stage/attempt form",
    )
    evidence_paths = {
        "create_id_sha256": sealed / "audit" / f"{container_name}.create-id.txt",
        "docker_create_cidfile_sha256": (
            sealed / "audit" / f"{container_name}.docker-create.cid"
        ),
        "docker_create_stdout_sha256": (
            sealed / "audit" / f"{container_name}.docker-create.stdout"
        ),
        "docker_create_stderr_sha256": (
            sealed / "audit" / f"{container_name}.docker-create.stderr"
        ),
        "container_contract_sha256": sealed / "audit/container.contract.json",
        "pre_run_inspect_sha256": sealed / "audit" / f"{container_name}.pre-run.inspect.json",
        "post_run_inspect_sha256": sealed / "audit" / f"{container_name}.post-run.inspect.json",
        "source_export_pre_manifest_sha256": sealed / "audit/source-export.pre.sha256",
        "source_export_post_manifest_sha256": sealed / "audit/source-export.post.sha256",
        "pre_receipt_artifact_manifest_sha256": (
            sealed / "audit/pre-receipt-artifact.sha256"
        ),
        "stage_reservation_archive_sha256": sealed / "audit/stage.reservation.json",
    }
    for key, path in evidence_paths.items():
        _, digest, _ = _read_regular_bytes(path, role=f"adapter {key}")
        _require(receipt.get(key) == digest, f"adapter evidence hash mismatch: {key}")
    contract, _, _ = _read_json_object(
        evidence_paths["container_contract_sha256"],
        role="adapter container contract",
    )
    contract_fields = {
        "schema_version",
        "stage",
        "container_name",
        "image_id",
        "entrypoint",
        "path",
        "args",
        "user",
        "working_dir",
        "expected_environment",
        "forbidden_environment",
        "gpu_device",
        "allowed_exit_codes",
        "host_config_exact",
        "mounts",
    }
    staging_root = declared.parent / f".incomplete-{declared.name}"
    expected_command = [
        "-I",
        "scripts/server/run_pams_conventional_cycleback_pose_input_authorization.py",
        "--representation-root",
        "/pams/representation",
        "--declared-representation-host-root",
        launch_authority.representation_root,
        "--launch-registry-root",
        "/pams/launch-registry",
        "--declared-launch-registry-host-root",
        os.fspath(_CANONICAL_LAUNCH_REGISTRY_ROOT),
        "--source-root",
        "/workspace",
        "--output-root",
        "/pams/output",
    ]
    expected_mounts = [
        {
            "destination": "/workspace",
            "type": "bind",
            "source": os.fspath(staging_root / "source"),
            "rw": False,
            "propagation": "rprivate",
        },
        {
            "destination": "/pams/launch-registry",
            "type": "bind",
            "source": os.fspath(_CANONICAL_LAUNCH_REGISTRY_ROOT),
            "rw": False,
            "propagation": "rprivate",
        },
        {
            "destination": "/pams/output",
            "type": "bind",
            "source": os.fspath(staging_root / "artifact"),
            "rw": True,
            "propagation": "rprivate",
        },
        {
            "destination": "/pams/representation",
            "type": "bind",
            "source": launch_authority.representation_root,
            "rw": False,
            "propagation": "rprivate",
        },
    ]
    expected_host = {
        "NetworkMode": "none",
        "ReadonlyRootfs": True,
        "CapDrop": ["ALL"],
        "SecurityOpt": ["no-new-privileges:true"],
        "PidsLimit": 4096,
        "Memory": 32 * 1024**3,
        "NanoCpus": 8_000_000_000,
        "IpcMode": "private",
        "PidMode": "",
        "UsernsMode": "",
        "Tmpfs": {
            "/tmp": "rw,noexec,nosuid,nodev,uid=1000,gid=1000,mode=1777,size=4g"
        },
        "AutoRemove": False,
    }
    _require(
        set(contract) == contract_fields
        and contract["schema_version"] == 2
        and contract["stage"] == "adapter"
        and contract["container_name"] == container_name
        and contract["image_id"] == launch_authority.container_image_id
        and contract["entrypoint"] == ["python"]
        and contract["path"] == "python"
        and contract["args"] == expected_command
        and contract["user"] == "1000:1000"
        and contract["working_dir"] == "/workspace"
        and contract["gpu_device"] is None
        and contract["allowed_exit_codes"] == [0]
        and contract["host_config_exact"] == expected_host
        and contract["mounts"] == expected_mounts
        and contract["forbidden_environment"]
        == [
            "PAMS_DEV_ROOT",
            "PAMS_TEST_ROOT",
            "PAMS_TARGETS",
            "PYTHONOPTIMIZE",
            "PYTHONINSPECT",
            "PYTHONSTARTUP",
        ]
        and isinstance(contract["expected_environment"], Mapping),
        "adapter container contract is not the exact frozen adapter spec",
    )
    expected_environment = contract["expected_environment"]
    _require(
        expected_environment.get("PYTHONDONTWRITEBYTECODE") == "1"
        and expected_environment.get("PYTHONHASHSEED") == "0"
        and expected_environment.get("PAMS_CONTAINER_SOURCE_REVISION")
        == launch_authority.source_revision
        and expected_environment.get("PAMS_CONTAINER_IMAGE_ID")
        == launch_authority.container_image_id
        and not any(
            key in expected_environment for key in contract["forbidden_environment"]
        ),
        "adapter container contract environment lineage mismatch",
    )
    reservation, archived_reservation_sha, archived_reservation_bytes = _read_json_object(
        evidence_paths["stage_reservation_archive_sha256"],
        role="adapter archived stage reservation",
    )
    reservation_fields = {
        "schema_version",
        "artifact_type",
        "stage",
        "attempt_id",
        "candidate_id",
        "run_key",
        "final_root",
        "staging_root",
        "launch_registry_id",
        "launch_authorization_sha256",
        "launch_registry_receipt_sha256",
        "source_revision",
        "source_tree_sha",
        "container_image_id",
        "config",
        "predecessors",
        "process_id",
        "authority_granted",
    }
    attempt_id = declared.name.removeprefix(
        f"{launch_authority.source_revision[:12]}-"
    )
    external_reservation_path = declared.parent / f"{declared.name}.reservation.json"
    _require(
        receipt.get("stage_reservation_path")
        == os.fspath(external_reservation_path)
        and receipt.get("stage_reservation_sha256") == archived_reservation_sha
        and receipt.get("stage_reservation_bytes") == archived_reservation_bytes,
        "adapter archived stage reservation receipt mismatch",
    )
    if Path(root) != _CONTAINER_POSE_AUTHORITY_ROOT:
        external_reservation, external_sha, external_bytes = _read_json_object(
            external_reservation_path,
            role="adapter external stage reservation",
        )
        _require(
            external_reservation == reservation
            and external_sha == archived_reservation_sha
            and external_bytes == archived_reservation_bytes
            and external_reservation_path.lstat().st_mode & 0o222 == 0,
            "adapter external/archive stage reservation mismatch",
        )
    _require(
        set(reservation) == reservation_fields
        and reservation["schema_version"] == 1
        and reservation["artifact_type"]
        == "pams_conventional_cycleback_stage_reservation_v1"
        and reservation["stage"] == "adapter"
        and reservation["attempt_id"] == attempt_id
        and reservation["candidate_id"] is None
        and reservation["run_key"] == declared.name
        and reservation["final_root"] == os.fspath(declared)
        and reservation["staging_root"] == os.fspath(staging_root)
        and reservation["launch_registry_id"] == launch_authority.registry_id
        and reservation["launch_authorization_sha256"]
        == launch_authority.authorization_sha256
        and reservation["launch_registry_receipt_sha256"]
        == launch_authority.receipt_sha256
        and reservation["source_revision"] == launch_authority.source_revision
        and reservation["source_tree_sha"] == launch_authority.source_tree_sha
        and reservation["container_image_id"] == launch_authority.container_image_id
        and reservation["config"] is None
        and reservation["predecessors"] == receipt.get("predecessors")
        and isinstance(reservation["process_id"], int)
        and not isinstance(reservation["process_id"], bool)
        and reservation["process_id"] > 0
        and reservation["authority_granted"] is False,
        "adapter stage reservation lineage mismatch",
    )
    predecessor_rows = reservation["predecessors"]
    _require(
        isinstance(predecessor_rows, list) and len(predecessor_rows) == 1,
        "adapter must have exactly one representation predecessor",
    )
    predecessor = predecessor_rows[0]
    representation_root = Path(launch_authority.representation_root)
    _require(
        isinstance(predecessor, Mapping)
        and isinstance(predecessor.get("output"), Mapping)
        and isinstance(predecessor.get("receipt"), Mapping),
        "adapter representation predecessor row is malformed",
    )
    predecessor_output = predecessor["output"]
    predecessor_receipt = predecessor["receipt"]
    _require(
        set(predecessor)
        == {
            "role",
            "root",
            "output",
            "receipt",
            "tree_sha256",
            "tree_bytes",
            "tree_files",
        }
        and predecessor["role"] == "representation"
        and predecessor["root"] == launch_authority.representation_root
        and predecessor_output
        == {
            "path": os.fspath(
                representation_root
                / "gate-output/cycleback-input.authorization.json"
            ),
            "sha256": launch_authority.representation_bindings[
                "representation_authorization_sha256"
            ],
            "bytes": predecessor_output.get("bytes"),
        }
        and isinstance(predecessor_output.get("bytes"), int)
        and predecessor_output["bytes"] > 0
        and predecessor_receipt
        == {
            "path": os.fspath(representation_root / "audit/run.receipt.json"),
            "sha256": launch_authority.representation_bindings[
                "representation_run_receipt_sha256"
            ],
            "bytes": predecessor_receipt.get("bytes"),
        }
        and isinstance(predecessor_receipt.get("bytes"), int)
        and predecessor_receipt["bytes"] > 0
        and _is_sha256(predecessor["tree_sha256"])
        and isinstance(predecessor["tree_bytes"], int)
        and predecessor["tree_bytes"] > 0
        and isinstance(predecessor["tree_files"], int)
        and predecessor["tree_files"] > 0,
        "adapter representation predecessor identity mismatch",
    )
    create_bytes, _, _ = _read_regular_bytes(
        evidence_paths["create_id_sha256"], role="adapter create ID"
    )
    try:
        create_encoded = create_bytes.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ValueError("adapter create ID is not ASCII") from exc
    if create_encoded.startswith("sha256:"):
        create_encoded = create_encoded[7:]
    _require(
        len(create_encoded) == 64
        and all(character in "0123456789abcdef" for character in create_encoded),
        "adapter create ID is malformed",
    )
    expected_container_id = f"sha256:{create_encoded}"
    _require(
        receipt.get("container_id") == expected_container_id,
        "adapter receipt/create ID mismatch",
    )
    for key in ("docker_create_cidfile_sha256", "docker_create_stdout_sha256"):
        raw_create_id, _, _ = _read_regular_bytes(
            evidence_paths[key],
            role=f"adapter {key}",
        )
        try:
            observed_create_id = raw_create_id.decode("ascii").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"adapter {key} is not ASCII") from exc
        _require(
            observed_create_id.removeprefix("sha256:") == create_encoded,
            f"adapter {key} differs from the exact created container ID",
        )
    create_stderr, _, _ = _read_regular_bytes(
        evidence_paths["docker_create_stderr_sha256"],
        role="adapter Docker create stderr",
    )
    _require(not create_stderr, "adapter Docker create unexpectedly wrote stderr")
    for phase in ("pre", "post"):
        item, _, _ = _read_json_object_list(
            evidence_paths[f"{phase}_run_inspect_sha256"],
            role=f"adapter {phase}-run inspect",
        )
        _validate_adapter_container_inspect(
            item=item,
            contract=contract,
            expected_container_id=expected_container_id,
            phase=phase,
        )
    cleanup = receipt.get("container_cleanup")
    _require(
        isinstance(cleanup, Mapping)
        and cleanup.get("removed_by_exact_id") == expected_container_id
        and cleanup.get("id_verified_before_remove") is True
        and cleanup.get("remove_succeeded") is True
        and cleanup.get("absence_verified") is True,
        "adapter exact-ID cleanup is not proven",
    )
    _require(
        receipt.get("source_export_manifests_identical") is True
        and receipt.get("source_export_pre_manifest_sha256")
        == receipt.get("source_export_post_manifest_sha256"),
        "adapter source export pre/post manifests differ",
    )
    source_manifest = _manifest_bytes(
        sealed / "source",
        role="adapter frozen source export",
    )
    for key in (
        "source_export_pre_manifest_sha256",
        "source_export_post_manifest_sha256",
    ):
        manifest_bytes, manifest_sha, manifest_count = _read_regular_bytes(
            evidence_paths[key], role=f"adapter {key}"
        )
        _require(
            manifest_bytes == source_manifest
            and receipt.get(key) == manifest_sha
            and len(manifest_bytes) == manifest_count,
            f"adapter {key} does not describe the frozen source tree",
        )
    artifact_manifest_path = evidence_paths[
        "pre_receipt_artifact_manifest_sha256"
    ]
    expected_artifact_manifest = _manifest_bytes(
        sealed,
        excluded={artifact_manifest_path, run_receipt_path},
        role="adapter pre-receipt artifact",
    )
    manifest_bytes, manifest_sha, manifest_count = _read_regular_bytes(
        artifact_manifest_path,
        role="adapter pre-receipt artifact manifest",
    )
    _require(
        manifest_bytes == expected_artifact_manifest
        and receipt.get("pre_receipt_artifact_manifest_sha256") == manifest_sha
        and receipt.get("pre_receipt_artifact_manifest_bytes") == manifest_count,
        "adapter pre-receipt artifact manifest does not close the published tree",
    )
    _require(
        (sealed / "source").is_dir() and not (sealed / "source").is_symlink(),
        "adapter frozen source export is missing",
    )
    _require(cache_dir.is_dir() and not cache_dir.is_symlink(), "adapter pose cache is missing")
    _require(
        joint_mask_dir.is_dir() and not joint_mask_dir.is_symlink(),
        "adapter joint-mask sidecars are missing",
    )
    return CycleBackPoseAuthority(
        root=sealed,
        snapshot_path=snapshot_path,
        cache_dir=cache_dir,
        segment_index_path=segment_index_path,
        joint_mask_dir=joint_mask_dir,
        joint_mask_snapshot_path=joint_snapshot_path,
        identity_map_path=identity_map_path,
        pair_eligibility_path=pair_eligibility_path,
        authorization_path=authorization_path,
        run_receipt_path=run_receipt_path,
        snapshot=snapshot,
        authorization_sha256=authorization_sha,
        run_receipt_sha256=receipt_sha,
        segment_index_sha256=segment_index_sha,
        segment_reset_policy_sha256=authorization["segment_reset_policy_sha256"],
        joint_mask_snapshot_sha256=joint_snapshot_sha,
        joint_mask_set_sha256=joint_snapshot.fingerprint,
        identity_map_sha256=identity_map_sha,
        pair_eligibility_sha256=pair_eligibility_sha,
        representation_bindings=dict(bindings),
    )


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--representation-root", type=Path, required=True)
    parser.add_argument("--declared-representation-host-root", type=Path, required=True)
    parser.add_argument("--launch-registry-root", type=Path, required=True)
    parser.add_argument("--declared-launch-registry-host-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        launch = validate_cycleback_launch_registry(
            arguments.launch_registry_root,
            declared_host_root=arguments.declared_launch_registry_host_root,
            source_root=arguments.source_root,
        )
        authority = validate_unified_representation_authority(
            arguments.representation_root,
            declared_host_root=arguments.declared_representation_host_root,
        )
        if os.fspath(arguments.declared_representation_host_root) != launch.representation_root:
            raise ValueError("mounted representation root differs from launch registry")
        authorization = materialize_cycleback_pose_authority(
            authority,
            output_root=arguments.output_root,
            launch_authority=launch,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"cycleback pose-input authorization failed: {exc}", file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "cycleback_pose_input_authorized": authorization[
                    "cycleback_pose_input_authorized"
                ],
                "pose_cache_entry_count": authorization["pose_cache_entry_count"],
                "pose_cache_set_sha256": authorization["pose_cache_set_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


def launch_registry_main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Reserve the canonical cycle-back source/image/config/representation registry."
    )
    parser.parse_args(argv)
    repository_root = Path(__file__).resolve().parents[3]
    try:
        authority = reserve_cycleback_launch_registry(
            repository_root=repository_root,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"cycleback launch registry reservation failed: {exc}", file=os.sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "registry_id": authority.registry_id,
                "source_revision": authority.source_revision,
                "container_image_id": authority.container_image_id,
                "authorization_sha256": authority.authorization_sha256,
                "receipt_sha256": authority.receipt_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "CycleBackLaunchAuthority",
    "CycleBackPoseAuthority",
    "UnifiedRepresentationAuthority",
    "materialize_cycleback_pose_authority",
    "reserve_cycleback_launch_registry",
    "validate_cycleback_pose_authority",
    "validate_cycleback_launch_registry",
    "validate_unified_representation_authority",
    "write_json_exclusive",
]
