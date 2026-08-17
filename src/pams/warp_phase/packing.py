"""Trusted offline packer boundary for the WARP-PHASE pilot.

This is the only module in the package that imports :mod:`pickle`.  It hashes
immutable bytes and checks the canonical source path before executing the first
pickle opcode.  The training-side reader in :mod:`pams.warp_phase.data` does
not import this module and has no vault reader.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import re
import stat
import sys
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from pams.warp_phase.types import (
    FEATURE_FIELD_NAMES,
    ArrayMemberReceipt,
    ArtifactReceipt,
    EligibilityDecision,
    EvaluatorVaultRecord,
    FeatureShard,
    PackedIdentityReceipt,
    PilotSplit,
    SourceVerificationReceipt,
    WarpPhaseContractError,
)

CANONICAL_SOURCE_RELATIVE_PATHS: dict[PilotSplit, Path] = {
    "train": Path("data/counting_multirep_skeleton_pose_expanded_v44_len320/train.pkl"),
    "val": Path("data/counting_multirep_skeleton_pose_expanded_v44_len320/val.pkl"),
}
CANONICAL_SOURCE_SHA256: dict[PilotSplit, str] = {
    "train": "c96fc1dfa233ec4bf1af19e7480ee8c721f4cddee9f121185f4e494c7244e1eb",
    "val": "4064ef24dc2970e9b07de8adddf1ecd4932aecb90453a67b4f90d7bd35d10251",
}

SCHEMA_FIXTURE_SHA256 = "31f81549f9c5dbedc6ddb278e868793246826a234de90c06c169c71387428275"
SCHEMA_FIXTURE_BYTE_COUNT = 617
SCHEMA_FIXTURE = (
    '{"annotation":{"height":"int","length":"int>1","object_schema":'
    '{"bbox":"<f8[L,4] finite","count":"int K","period":"int[K,2] with '
    '0<=s<e<=L","periodicity":"empty list"},"width":"int"},'
    '"evaluator_vault":{"P_eval":"numpy.median(asarray(e-s,dtype=float64))",'
    '"integrity":"K>0; Python int endpoints; 0<=s<e<=L",'
    '"interval_semantics":"[s,e) source-frame boundary units"},'
    '"feature_shard":{"frame_mask":"bool[P,320]",'
    '"local_person_slot":"int routing only","motion":"<f4[P,320,17,3]",'
    '"opaque_sample_key":"non-semantic digest routing only",'
    '"person_mask":"bool[P]","sampled_frame_indices":"<i8[320]",'
    '"source_length":"Python int"}}'
)

_OPAQUE_KEY = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN_PATH_MARKERS = (
    "test",
    "sealed",
    "heldout",
    "held-out",
    "results",
    "output",
    "real_disjoint",
    "official_complete",
)
_MAX_SOURCE_BYTES = 512 * 1024 * 1024
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_ZIP_FILE_MODE = stat.S_IFREG | 0o600


class PackingContractError(WarpPhaseContractError):
    """Base exception for an offline packing contract failure."""


class UnapprovedSplitError(PackingContractError):
    """Raised before I/O when a split other than canonical train/val is requested."""


class ForbiddenSourcePathError(PackingContractError):
    """Raised before I/O for a test, sealed, held-out, result, or noncanonical path."""


class SourceHashMismatchError(PackingContractError):
    """Raised when canonical source bytes do not match the frozen SHA-256."""


class PhysicalSeparationError(PackingContractError):
    """Raised when features, vault, and audit roots are not physically distinct."""

    def __init__(self, message: str, *, receipt: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.receipt = dict(receipt) if receipt is not None else None


class FeatureSchemaError(PackingContractError):
    """Raised when a feature shard is not exactly the seven-field schema."""


class VaultSchemaError(PackingContractError):
    """Raised when a privileged evaluator record is malformed."""


class EligibilityManifestError(PackingContractError):
    """Raised when count-blind eligibility/component receipts are incomplete."""


@dataclass(frozen=True, slots=True)
class PilotLayout:
    """Three physically separate roots beneath one prospective run directory."""

    run_root: Path
    features: Path
    vault: Path
    audit: Path

    def split_features(self, split: PilotSplit) -> Path:
        return self.features / split


@dataclass(frozen=True, slots=True)
class _PreparedArtifact:
    receipt: ArtifactReceipt
    artifact_payload: bytes
    receipt_payload: bytes
    mode: int


def _normalize_split(split: str) -> PilotSplit:
    normalized = split.strip().lower()
    if normalized not in {"train", "val"}:
        raise UnapprovedSplitError("WARP-PHASE source and feature splits are only train or val")
    return cast(PilotSplit, normalized)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _ensure_regular_directory(path: Path, *, mode: int) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        path.mkdir(parents=True, mode=mode)
        metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise PhysicalSeparationError(f"packing root is not a regular directory: {path}")
    with suppress(OSError):
        path.chmod(mode)
    # Windows ACLs are environment-owned.  The non-overlap invariant still
    # holds and the emitted vault files remain write-restricted where POSIX
    # mode bits are supported.


def create_pilot_layout(run_root: Path) -> PilotLayout:
    """Create the non-overlapping ``features/``, ``vault/``, and ``audit/`` roots."""

    raw_root = Path(run_root)
    if raw_root.exists() and stat.S_ISLNK(raw_root.lstat().st_mode):
        raise PhysicalSeparationError("run root must not be a symbolic link")
    resolved = raw_root.resolve(strict=False)
    layout = PilotLayout(
        run_root=resolved,
        features=resolved / "features",
        vault=resolved / "vault",
        audit=resolved / "audit",
    )
    roots = (layout.features.resolve(strict=False), layout.vault.resolve(strict=False), layout.audit.resolve(strict=False))
    if len(set(roots)) != 3:
        raise PhysicalSeparationError("features, vault, and audit roots must be distinct")
    _ensure_regular_directory(layout.run_root, mode=0o750)
    _ensure_regular_directory(layout.features, mode=0o750)
    _ensure_regular_directory(layout.vault, mode=0o700)
    _ensure_regular_directory(layout.audit, mode=0o750)
    for split in ("train", "val"):
        _ensure_regular_directory(layout.features / split, mode=0o750)
        _ensure_regular_directory(layout.vault / split, mode=0o700)
    return layout


def _permission_root_evidence(path: Path) -> tuple[dict[str, Any], bool]:
    metadata = path.lstat()
    acl_names: tuple[str, ...] = ()
    acl_inspected = False
    listxattr = getattr(os, "listxattr", None)
    if os.name == "posix" and listxattr is not None:
        try:
            acl_names = tuple(
                sorted(
                    name
                    for name in listxattr(path, follow_symlinks=False)
                    if "acl" in str(name).casefold()
                )
            )
        except OSError:
            pass
        else:
            acl_inspected = True
    return (
        {
            "acl_xattrs": list(acl_names),
            "device": int(metadata.st_dev),
            "group_principal": f"gid:{int(metadata.st_gid)}",
            "mode": f"{stat.S_IMODE(metadata.st_mode):04o}",
            "owner_principal": f"uid:{int(metadata.st_uid)}",
            "path": str(path),
            "regular_directory": bool(
                stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode)
            ),
        },
        acl_inspected,
    )


def permission_separation_receipt(layout: PilotLayout) -> dict[str, Any]:
    """Return live, machine-readable OS-principal evidence for the three roots.

    Linux POSIX owner/mode separation is accepted only when the feature and vault
    roots have distinct owners, every vault directory denies group/other
    access, and extended ACLs were inspectable and absent.  Windows ACLs are
    deliberately reported as ``BLOCKED`` because Python's portable filesystem
    API cannot prove their effective-principal semantics.
    """

    named_roots = {
        "audit": layout.audit,
        "features": layout.features,
        "features_train": layout.features / "train",
        "features_val": layout.features / "val",
        "run_root": layout.run_root,
        "vault": layout.vault,
        "vault_train": layout.vault / "train",
        "vault_val": layout.vault / "val",
    }
    evidence: dict[str, dict[str, Any]] = {}
    acl_inspection: dict[str, bool] = {}
    for name, path in named_roots.items():
        evidence[name], acl_inspection[name] = _permission_root_evidence(path)

    feature_owner = str(evidence["features"]["owner_principal"])
    vault_owner = str(evidence["vault"]["owner_principal"])
    feature_names = ("features", "features_train", "features_val")
    vault_names = ("vault", "vault_train", "vault_val")
    posix_permissions = os.name == "posix" and sys.platform.startswith("linux")
    get_effective_uid = getattr(os, "geteuid", None)
    effective_uid = (
        int(get_effective_uid())
        if posix_permissions and get_effective_uid is not None
        else -1
    )
    checks = {
        "all_roots_regular_non_symlink": all(
            bool(root["regular_directory"]) for root in evidence.values()
        ),
        "all_roots_same_filesystem": len(
            {int(root["device"]) for root in evidence.values()}
        )
        == 1,
        "acl_semantics_inspected": posix_permissions
        and all(acl_inspection[name] for name in named_roots),
        "feature_owner_consistent": all(
            evidence[name]["owner_principal"] == feature_owner for name in feature_names
        ),
        "vault_owner_consistent": all(
            evidence[name]["owner_principal"] == vault_owner for name in vault_names
        ),
        "feature_and_vault_principals_distinct": feature_owner != vault_owner,
        "training_principal_is_unprivileged": feature_owner != "uid:0",
        "trusted_packer_can_assign_both_principals": posix_permissions
        and effective_uid == 0,
        "vault_denies_group_and_other": posix_permissions
        and all(int(str(evidence[name]["mode"]), 8) & 0o077 == 0 for name in vault_names),
        "vault_has_no_extended_acl": posix_permissions
        and all(not evidence[name]["acl_xattrs"] for name in vault_names),
    }
    required_checks = tuple(checks)
    blockers = tuple(name for name in required_checks if not checks[name])
    if not posix_permissions:
        blockers = (*blockers, "portable_windows_acl_effective_access_not_verifiable")
    passed = not blockers
    return {
        "authorizes": ["identity_publication"] if passed else [],
        "blockers": list(blockers),
        "checks": checks,
        "mechanism": (
            "posix_distinct_owner_mode_without_extended_acl"
            if posix_permissions
            else "portable_acl_verification_unavailable"
        ),
        "platform": os.name,
        "principals": {
            "evaluator": vault_owner if posix_permissions else "UNVERIFIED",
            "packer": f"uid:{effective_uid}" if posix_permissions else "UNVERIFIED",
            "training": feature_owner if posix_permissions else "UNVERIFIED",
        },
        "roots": evidence,
        "schema_version": 1,
        "status": "PASS" if passed else "BLOCKED",
        "type": "warp_phase_permission_principal_separation",
    }


def verify_permission_separation(
    layout: PilotLayout,
    receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute and require a PASS permission receipt before label access."""

    observed = permission_separation_receipt(layout)
    if receipt is not None and _canonical_json_bytes(dict(receipt)) != _canonical_json_bytes(
        observed
    ):
        raise PhysicalSeparationError(
            "permission/principal receipt does not match live filesystem evidence",
            receipt=observed,
        )
    if observed["status"] != "PASS":
        raise PhysicalSeparationError(
            "BLOCKED: OS-principal feature/vault permission separation is not proven",
            receipt=observed,
        )
    return observed


def _forbidden_path_marker(path: Path, repository_root: Path) -> str | None:
    try:
        inspected = path.resolve(strict=False).relative_to(repository_root.resolve(strict=False))
    except ValueError:
        inspected = path.resolve(strict=False)
    lowered_parts = tuple(part.casefold() for part in inspected.parts)
    for marker in _FORBIDDEN_PATH_MARKERS:
        if any(marker in part for part in lowered_parts):
            return marker
    return None


def _approved_source_path(
    path: Path,
    *,
    repository_root: Path,
    split: str,
) -> tuple[PilotSplit, Path, str]:
    approved_split = _normalize_split(split)
    root = repository_root.resolve(strict=False)
    candidate = Path(path).resolve(strict=False)
    marker = _forbidden_path_marker(candidate, root)
    if marker is not None:
        raise ForbiddenSourcePathError(
            f"forbidden source path marker {marker!r}: {candidate}"
        )
    expected_relative = CANONICAL_SOURCE_RELATIVE_PATHS[approved_split]
    expected = (root / expected_relative).resolve(strict=False)
    if candidate != expected:
        raise ForbiddenSourcePathError(
            "source pickle path is not the exact canonical v44 "
            f"{approved_split} path: {candidate}"
        )
    if candidate.suffix.casefold() != ".pkl":
        raise ForbiddenSourcePathError("approved source must be a .pkl file")
    return approved_split, candidate, expected_relative.as_posix()


def _read_stable_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"source file does not exist: {path}") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ForbiddenSourcePathError(f"source must be a regular non-symlink file: {path}")
    if before.st_size > maximum_bytes:
        raise PackingContractError(f"source exceeds the fixed safety limit: {path}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        byte_count = 0
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            while chunk := handle.read(1024 * 1024):
                byte_count += len(chunk)
                if byte_count > maximum_bytes:
                    raise PackingContractError(f"source exceeds the fixed safety limit: {path}")
                chunks.append(chunk)
            closed = os.fstat(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    after = path.lstat()
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    if any(
        getattr(before, field) != getattr(opened, field)
        or getattr(opened, field) != getattr(closed, field)
        or getattr(closed, field) != getattr(after, field)
        for field in stable_fields
    ):
        raise PackingContractError(f"source changed while it was being hashed: {path}")
    return b"".join(chunks)


def _verify_source_bytes(
    path: Path,
    *,
    repository_root: Path,
    split: str,
) -> tuple[bytes, SourceVerificationReceipt]:
    approved_split, approved_path, relative_path = _approved_source_path(
        path,
        repository_root=repository_root,
        split=split,
    )
    payload = _read_stable_regular_file(approved_path, maximum_bytes=_MAX_SOURCE_BYTES)
    observed = _sha256_bytes(payload)
    expected = CANONICAL_SOURCE_SHA256[approved_split]
    if observed != expected:
        raise SourceHashMismatchError(
            f"canonical {approved_split} pickle SHA-256 mismatch: {observed} != {expected}"
        )
    receipt = SourceVerificationReceipt(
        split=approved_split,
        path=approved_path,
        relative_path=relative_path,
        sha256=observed,
        byte_count=len(payload),
    )
    return payload, receipt


def verify_source_pickle(
    path: Path,
    *,
    repository_root: Path,
    split: str,
) -> SourceVerificationReceipt:
    """Verify the exact canonical v44 path and SHA without deserializing it."""

    _, receipt = _verify_source_bytes(
        path,
        repository_root=repository_root,
        split=split,
    )
    return receipt


def trusted_load_source_pickle(
    path: Path,
    *,
    repository_root: Path,
    split: str,
) -> tuple[Any, SourceVerificationReceipt]:
    """Hash immutable in-memory bytes before executing the first pickle opcode."""

    payload, receipt = _verify_source_bytes(
        path,
        repository_root=repository_root,
        split=split,
    )
    # ``payload`` is the exact byte string whose SHA-256 matched above.  Using
    # pickle.loads rather than reopening the path removes the hash/unpickle
    # time-of-check/time-of-use gap.
    decoded = pickle.loads(payload)
    return decoded, receipt


def schema_fixture_bytes() -> bytes:
    """Return the normative 617-byte desensitized fixture after self-verification."""

    encoded = SCHEMA_FIXTURE.encode("utf-8")
    observed = _sha256_bytes(encoded)
    if len(encoded) != SCHEMA_FIXTURE_BYTE_COUNT or observed != SCHEMA_FIXTURE_SHA256:
        raise PackingContractError(
            "embedded schema fixture does not match the normative byte/hash receipt"
        )
    return encoded


def load_evaluator_schema_view(payload: bytes) -> dict[str, str]:
    """Verify the normative fixture and expose only its evaluator-vault view."""

    if len(payload) != SCHEMA_FIXTURE_BYTE_COUNT or _sha256_bytes(payload) != SCHEMA_FIXTURE_SHA256:
        raise PackingContractError("schema fixture byte/hash receipt does not match")
    decoded = json.loads(payload.decode("utf-8"))
    if not isinstance(decoded, dict) or set(decoded) != {
        "annotation",
        "evaluator_vault",
        "feature_shard",
    }:
        raise PackingContractError("schema fixture root does not match the frozen contract")
    evaluator_view = decoded["evaluator_vault"]
    if not isinstance(evaluator_view, dict) or set(evaluator_view) != {
        "P_eval",
        "integrity",
        "interval_semantics",
    }:
        raise PackingContractError("schema fixture evaluator view is malformed")
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in evaluator_view.items()
    ):
        raise PackingContractError("schema fixture evaluator entries must be strings")
    return cast(dict[str, str], dict(evaluator_view))


def _write_exclusive(path: Path, payload: bytes, *, mode: int) -> None:
    _ensure_regular_directory(path.parent, mode=0o700 if mode == 0o600 else 0o750)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(path, flags, mode)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _member_payload(member: ArrayMemberReceipt) -> dict[str, Any]:
    return {
        "byte_count": member.byte_count,
        "dtype": member.dtype,
        "name": member.name,
        "sha256": member.sha256,
        "shape": list(member.shape),
    }


def _artifact_payload(
    *,
    role: str,
    relative_path: str,
    sha256: str,
    byte_count: int,
    members: Sequence[ArrayMemberReceipt],
) -> dict[str, Any]:
    return {
        "byte_count": byte_count,
        "members": [_member_payload(member) for member in members],
        "relative_path": relative_path,
        "role": role,
        "schema_version": 1,
        "sha256": sha256,
    }


def _prepare_artifact(
    layout: PilotLayout,
    *,
    role: str,
    relative_path: Path,
    audit_relative_path: Path,
    payload: bytes,
    mode: int,
    members: Sequence[ArrayMemberReceipt] = (),
) -> _PreparedArtifact:
    artifact_path = layout.run_root / relative_path
    artifact_sha256 = _sha256_bytes(payload)
    receipt_payload = _canonical_json_bytes(
        _artifact_payload(
            role=role,
            relative_path=relative_path.as_posix(),
            sha256=artifact_sha256,
            byte_count=len(payload),
            members=members,
        )
    )
    receipt_path = layout.audit / audit_relative_path
    return _PreparedArtifact(
        receipt=ArtifactReceipt(
            role=role,
            path=artifact_path,
            relative_path=relative_path.as_posix(),
            sha256=artifact_sha256,
            byte_count=len(payload),
            receipt_path=receipt_path,
            receipt_sha256=_sha256_bytes(receipt_payload),
            members=tuple(members),
        ),
        artifact_payload=payload,
        receipt_payload=receipt_payload,
        mode=mode,
    )


def _ensure_target_parent(layout: PilotLayout, target: Path) -> None:
    try:
        relative = target.relative_to(layout.run_root)
    except ValueError as exc:
        raise PhysicalSeparationError("publication target escapes the run root") from exc
    if not relative.parts or relative.name in {"", ".", ".."}:
        raise PhysicalSeparationError("publication target is not a regular run-root child")
    current = layout.run_root
    for part in relative.parent.parts:
        if part in {"", ".", ".."}:
            raise PhysicalSeparationError("publication target contains an unsafe path segment")
        current /= part
        mode = 0o700 if relative.parts[0] in {"vault", ".packing-staging"} else 0o750
        _ensure_regular_directory(current, mode=mode)


def _target_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _fsync_directory(path: Path) -> bool:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        if os.name == "posix":
            raise
        return False
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


def _assign_staged_target_principal(
    layout: PilotLayout,
    staged_path: Path,
    target: Path,
    *,
    mode: int,
) -> None:
    if os.name != "posix":
        return
    relative = target.relative_to(layout.run_root)
    owner_root = {
        "features": layout.features,
        "vault": layout.vault,
        "audit": layout.audit,
    }.get(relative.parts[0], layout.run_root)
    owner = owner_root.lstat()
    chown = getattr(os, "chown", None)
    if chown is None:
        raise PhysicalSeparationError("POSIX principal assignment is unavailable")
    chown(staged_path, int(owner.st_uid), int(owner.st_gid))
    staged_path.chmod(mode)
    descriptor = os.open(staged_path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _transaction_entries(
    prepared: Sequence[_PreparedArtifact],
    *,
    commit: _PreparedArtifact,
) -> tuple[tuple[Path, bytes, int], ...]:
    if commit not in prepared:
        raise PackingContractError("identity commit artifact is absent from its transaction")
    ordered = [artifact for artifact in prepared if artifact is not commit]
    ordered.append(commit)
    entries: list[tuple[Path, bytes, int]] = []
    for artifact in ordered:
        entries.append((artifact.receipt.receipt_path, artifact.receipt_payload, 0o600))
        entries.append((artifact.receipt.path, artifact.artifact_payload, artifact.mode))
    return tuple(entries)


def _publish_prepared_artifacts(
    layout: PilotLayout,
    prepared: Sequence[_PreparedArtifact],
    *,
    commit: _PreparedArtifact,
) -> None:
    """Stage, fsync, validate, and publish one no-overwrite artifact transaction.

    Every byte is first written to a private directory on the run filesystem.
    Hard-link publication provides atomic no-replace visibility for each file;
    the commit artifact is linked last.  A synchronous failure rolls back only
    inodes created by this transaction.  A crash before the commit leaves a
    detectable partial target set that all retries refuse.
    """

    entries = _transaction_entries(prepared, commit=commit)
    targets = tuple(target for target, _payload, _mode in entries)
    if len(set(targets)) != len(targets):
        raise PackingContractError("publication transaction contains duplicate targets")
    for target in targets:
        _ensure_target_parent(layout, target)
    existing = tuple(str(target) for target in targets if _target_exists(target))
    if existing:
        raise FileExistsError(
            "refusing partial or existing identity publication targets: " + ", ".join(existing)
        )

    staging_root = layout.run_root / ".packing-staging"
    _ensure_regular_directory(staging_root, mode=0o700)
    if staging_root.stat().st_dev != layout.run_root.stat().st_dev:
        raise PhysicalSeparationError("private staging is not on the run-root filesystem")

    published: list[tuple[Path, int, int]] = []
    with tempfile.TemporaryDirectory(prefix="identity-", dir=staging_root) as raw_stage:
        stage = Path(raw_stage)
        with suppress(OSError):
            stage.chmod(0o700)
        staged: list[tuple[Path, Path, bytes]] = []
        for index, (target, payload, mode) in enumerate(entries):
            staged_path = stage / f"{index:04d}.stage"
            _write_exclusive(staged_path, payload, mode=mode)
            _assign_staged_target_principal(
                layout,
                staged_path,
                target,
                mode=mode,
            )
            if staged_path.stat().st_dev != target.parent.stat().st_dev:
                raise PhysicalSeparationError(
                    "staged identity output is not on its publication filesystem"
                )
            observed = staged_path.read_bytes()
            if len(observed) != len(payload) or _sha256_bytes(observed) != _sha256_bytes(payload):
                raise PackingContractError("staged identity output failed byte validation")
            staged.append((staged_path, target, payload))
        _fsync_directory(stage)

        try:
            for staged_path, target, payload in staged:
                os.link(staged_path, target, follow_symlinks=False)
                metadata = target.lstat()
                published.append((target, int(metadata.st_dev), int(metadata.st_ino)))
                if not stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                    raise PackingContractError("published identity target is not a regular file")
                if metadata.st_size != len(payload):
                    raise PackingContractError("published identity target size changed")
                staged_path.unlink()
                _fsync_directory(target.parent)
        except Exception:
            for target, expected_device, expected_inode in reversed(published):
                try:
                    metadata = target.lstat()
                except FileNotFoundError:
                    continue
                if (
                    int(metadata.st_dev) == expected_device
                    and int(metadata.st_ino) == expected_inode
                    and stat.S_ISREG(metadata.st_mode)
                ):
                    target.unlink()
                    _fsync_directory(target.parent)
            raise


def _existing_prepared_artifact(prepared: _PreparedArtifact) -> bool:
    artifact_exists = _target_exists(prepared.receipt.path)
    receipt_exists = _target_exists(prepared.receipt.receipt_path)
    if artifact_exists != receipt_exists:
        raise FileExistsError("refusing a partial existing artifact/receipt pair")
    if not artifact_exists:
        return False
    for path, expected in (
        (prepared.receipt.path, prepared.artifact_payload),
        (prepared.receipt.receipt_path, prepared.receipt_payload),
    ):
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise FileExistsError("existing artifact/receipt target must be a regular non-symlink")
        if path.read_bytes() != expected:
            raise FileExistsError("existing artifact/receipt target has different bytes")
    return True


def _publish_with_receipt(
    layout: PilotLayout,
    *,
    role: str,
    relative_path: Path,
    audit_relative_path: Path,
    payload: bytes,
    mode: int,
    members: Sequence[ArrayMemberReceipt] = (),
) -> ArtifactReceipt:
    prepared = _prepare_artifact(
        layout,
        role=role,
        relative_path=relative_path,
        audit_relative_path=audit_relative_path,
        payload=payload,
        mode=mode,
        members=members,
    )
    _publish_prepared_artifacts(layout, (prepared,), commit=prepared)
    return prepared.receipt


def write_permission_separation_receipt(layout: PilotLayout) -> ArtifactReceipt:
    """Persist or exactly reuse the current OS-principal separation evidence."""

    payload = _canonical_json_bytes(permission_separation_receipt(layout))
    prepared = _prepare_artifact(
        layout,
        role="permission_principal_separation",
        relative_path=Path("audit/permissions/principal-separation.json"),
        audit_relative_path=Path("permissions/principal-separation.receipt.json"),
        payload=payload,
        mode=0o600,
    )
    if not _existing_prepared_artifact(prepared):
        _publish_prepared_artifacts(layout, (prepared,), commit=prepared)
    return prepared.receipt


def _require_persisted_permission_separation(
    layout: PilotLayout,
) -> tuple[dict[str, Any], ArtifactReceipt]:
    evidence = permission_separation_receipt(layout)
    artifact = write_permission_separation_receipt(layout)
    return verify_permission_separation(layout, evidence), artifact


def write_schema_fixture(layout: PilotLayout) -> ArtifactReceipt:
    """Persist the exact schema fixture and its detached canonical receipt."""

    return _publish_with_receipt(
        layout,
        role="desensitized_schema_fixture",
        relative_path=Path("audit/schema/desensitized_schema_fixture.json"),
        audit_relative_path=Path("schema/desensitized_schema_fixture.receipt.json"),
        payload=schema_fixture_bytes(),
        mode=0o600,
    )


def _scalar_value(value: object, *, field: str) -> object:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    if array.shape == (1,):
        return array[0].item()
    raise FeatureSchemaError(f"{field} must be a scalar or one-element array")


def feature_shard_from_mapping(payload: Mapping[str, object]) -> FeatureShard:
    """Construct a typed feature record only from the exact persisted whitelist."""

    supplied = set(payload)
    expected = set(FEATURE_FIELD_NAMES)
    if supplied != expected:
        raise FeatureSchemaError(
            f"feature fields must be exactly {sorted(expected)}; "
            f"missing={sorted(expected - supplied)}, extra={sorted(supplied - expected)}"
        )
    raw_key = _scalar_value(payload["opaque_sample_key"], field="opaque_sample_key")
    if isinstance(raw_key, bytes):
        try:
            opaque_key = raw_key.decode("ascii")
        except UnicodeDecodeError as exc:
            raise FeatureSchemaError("opaque_sample_key must be ASCII") from exc
    elif isinstance(raw_key, str):
        opaque_key = raw_key
    else:
        raise FeatureSchemaError("opaque_sample_key must be bytes or str")
    raw_person = _scalar_value(payload["person_mask"], field="person_mask")
    if not isinstance(raw_person, (bool, np.bool_, int, np.integer)):
        raise FeatureSchemaError("person_mask must be boolean")
    person_value = int(raw_person)
    if person_value not in {0, 1}:
        raise FeatureSchemaError("person_mask must contain only 0 or 1")
    raw_length = _scalar_value(payload["source_length"], field="source_length")
    raw_slot = _scalar_value(payload["local_person_slot"], field="local_person_slot")
    if isinstance(raw_length, bool) or not isinstance(raw_length, (int, np.integer)):
        raise FeatureSchemaError("source_length must be an integer")
    if isinstance(raw_slot, bool) or not isinstance(raw_slot, (int, np.integer)):
        raise FeatureSchemaError("local_person_slot must be an integer")
    return FeatureShard(
        motion=np.asarray(payload["motion"]),
        person_mask=bool(person_value),
        frame_mask=np.asarray(payload["frame_mask"]),
        sampled_frame_indices=np.asarray(payload["sampled_frame_indices"]),
        source_length=int(raw_length),
        opaque_sample_key=opaque_key,
        local_person_slot=int(raw_slot),
    )


def _require_feature_record(record: FeatureShard) -> None:
    if record.motion.dtype != np.dtype("<f4"):
        raise FeatureSchemaError("motion must have exact little-endian float32 dtype")
    if record.motion.shape != (320, 17, 3):
        raise FeatureSchemaError("motion must have shape [320,17,3]")
    if not record.motion.flags.c_contiguous or not np.isfinite(record.motion).all():
        raise FeatureSchemaError("motion must be finite and C-contiguous")
    if record.frame_mask.shape != (320,):
        raise FeatureSchemaError("frame_mask must have shape [320]")
    if record.frame_mask.dtype not in {np.dtype("|u1"), np.dtype("bool")}:
        raise FeatureSchemaError("frame_mask must be bool or uint8")
    if not np.isin(record.frame_mask, (0, 1)).all():
        raise FeatureSchemaError("frame_mask must contain only 0 or 1")
    if record.sampled_frame_indices.dtype != np.dtype("<i8"):
        raise FeatureSchemaError(
            "sampled_frame_indices must have exact little-endian int64 dtype"
        )
    if record.sampled_frame_indices.shape != (320,):
        raise FeatureSchemaError("sampled_frame_indices must have shape [320]")
    if np.any(np.diff(record.sampled_frame_indices) < 0):
        raise FeatureSchemaError("sampled_frame_indices must be nondecreasing")
    if type(record.source_length) is not int or record.source_length <= 1:
        raise FeatureSchemaError("source_length must be a Python int greater than one")
    if (
        int(record.sampled_frame_indices[0]) != 0
        or int(record.sampled_frame_indices[-1]) != record.source_length - 1
        or np.any(record.sampled_frame_indices < 0)
        or np.any(record.sampled_frame_indices >= record.source_length)
    ):
        raise FeatureSchemaError("sampled_frame_indices must bind source endpoints and domain")
    if not record.person_mask:
        raise FeatureSchemaError("only person-valid complete identities may be persisted")
    if _OPAQUE_KEY.fullmatch(record.opaque_sample_key) is None:
        raise FeatureSchemaError("opaque_sample_key must be 64 lowercase hexadecimal bytes")
    if type(record.local_person_slot) is not int or record.local_person_slot < 0:
        raise FeatureSchemaError("local_person_slot must be a non-negative Python int")


def _feature_arrays(record: FeatureShard) -> dict[str, NDArray[np.generic]]:
    _require_feature_record(record)
    return {
        "frame_mask": np.ascontiguousarray(record.frame_mask, dtype="|u1"),
        "local_person_slot": np.asarray([record.local_person_slot], dtype="<i8"),
        "motion": np.ascontiguousarray(record.motion, dtype="<f4"),
        "opaque_sample_key": np.asarray(
            [record.opaque_sample_key.encode("ascii")], dtype="|S64"
        ),
        "person_mask": np.asarray([1], dtype="|u1"),
        "sampled_frame_indices": np.ascontiguousarray(
            record.sampled_frame_indices, dtype="<i8"
        ),
        "source_length": np.asarray([record.source_length], dtype="<i8"),
    }


def _npy_bytes(array: NDArray[np.generic]) -> bytes:
    handle = io.BytesIO()
    np.lib.format.write_array(
        handle,
        np.ascontiguousarray(array),
        version=(2, 0),
        allow_pickle=False,
    )
    return handle.getvalue()


def deterministic_feature_npz_bytes(
    feature: FeatureShard | Mapping[str, object],
) -> tuple[bytes, tuple[ArrayMemberReceipt, ...]]:
    """Serialize byte-identical, fixed-metadata NPZ bytes for one identity."""

    record = feature if isinstance(feature, FeatureShard) else feature_shard_from_mapping(feature)
    arrays = _feature_arrays(record)
    output = io.BytesIO()
    receipts: list[ArrayMemberReceipt] = []
    with zipfile.ZipFile(output, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.comment = b""
        for field in sorted(arrays):
            member_name = f"{field}.npy"
            member_bytes = _npy_bytes(arrays[field])
            info = zipfile.ZipInfo(member_name, date_time=_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = _ZIP_FILE_MODE << 16
            info.extra = b""
            info.comment = b""
            archive.writestr(info, member_bytes)
            receipts.append(
                ArrayMemberReceipt(
                    name=member_name,
                    dtype=arrays[field].dtype.str,
                    shape=tuple(int(value) for value in arrays[field].shape),
                    sha256=_sha256_bytes(member_bytes),
                    byte_count=len(member_bytes),
                )
            )
    return output.getvalue(), tuple(receipts)


def write_feature_shard(
    layout: PilotLayout,
    split: str,
    feature: FeatureShard | Mapping[str, object],
) -> ArtifactReceipt:
    """Write one seven-field feature shard and its detached audit receipt."""

    approved_split = _normalize_split(split)
    record = feature if isinstance(feature, FeatureShard) else feature_shard_from_mapping(feature)
    payload, members = deterministic_feature_npz_bytes(record)
    filename = f"{record.opaque_sample_key}.{record.local_person_slot}.npz"
    return _publish_with_receipt(
        layout,
        role="feature_shard",
        relative_path=Path("features") / approved_split / filename,
        audit_relative_path=Path("feature-shards") / approved_split / f"{filename}.receipt.json",
        payload=payload,
        mode=0o600,
        members=members,
    )


def _validated_vault_payload(record: EvaluatorVaultRecord) -> dict[str, Any]:
    if _OPAQUE_KEY.fullmatch(record.opaque_sample_key) is None:
        raise VaultSchemaError("vault opaque key must be 64 lowercase hexadecimal characters")
    if type(record.local_person_slot) is not int or record.local_person_slot < 0:
        raise VaultSchemaError("vault local slot must be a non-negative Python int")
    if type(record.source_length) is not int or record.source_length <= 1:
        raise VaultSchemaError("vault source_length must be a Python int greater than one")
    if type(record.count) is not int or record.count <= 0:
        raise VaultSchemaError("vault count must be a positive Python int")
    if not record.periods or len(record.periods) != record.count:
        raise VaultSchemaError("vault periods must be nonempty and count-aligned")
    previous_end = -1
    rows: list[list[int]] = []
    for period in record.periods:
        if len(period) != 2:
            raise VaultSchemaError("each vault period must contain exactly two endpoints")
        start, end = period
        if type(start) is not int or type(end) is not int:
            raise VaultSchemaError("vault period endpoints must be Python integers")
        if not 0 <= start < end <= record.source_length:
            raise VaultSchemaError("vault periods must satisfy 0 <= start < end <= source_length")
        if start < previous_end:
            raise VaultSchemaError("vault periods must be sorted and non-overlapping")
        previous_end = end
        rows.append([start, end])
    return {
        "count": record.count,
        "local_person_slot": record.local_person_slot,
        "opaque_sample_key": record.opaque_sample_key,
        "periods": rows,
        "source_length": record.source_length,
    }


def write_evaluator_vault_record(
    layout: PilotLayout,
    split: str,
    record: EvaluatorVaultRecord,
) -> ArtifactReceipt:
    """Write one privileged record beneath ``vault/`` only."""

    approved_split = _normalize_split(split)
    payload = _canonical_json_bytes(_validated_vault_payload(record))
    _require_persisted_permission_separation(layout)
    filename = f"{record.opaque_sample_key}.{record.local_person_slot}.json"
    return _publish_with_receipt(
        layout,
        role="evaluator_vault_record",
        relative_path=Path("vault") / approved_split / filename,
        audit_relative_path=Path("vault-records") / approved_split / f"{filename}.receipt.json",
        payload=payload,
        mode=0o600,
    )


def _eligibility_payload(
    decision: EligibilityDecision,
    source: SourceVerificationReceipt,
) -> dict[str, Any]:
    return {
        "association_ambiguous": decision.association_ambiguous,
        "association_one_to_one": decision.association_one_to_one,
        "conflict_clocks": list(decision.conflict_clocks),
        "eligible": decision.eligible,
        "feature_frame_coverage": decision.feature_frame_coverage,
        "local_person_slot": decision.local_person_slot,
        "normalization_scale": decision.normalization_scale,
        "opaque_sample_key": decision.opaque_sample_key,
        "pose_coverage": decision.pose_coverage,
        "reasons": list(decision.reasons),
        "retained_distinct_clocks": decision.retained_distinct_clocks,
        "source": {
            "byte_count": source.byte_count,
            "relative_path": source.relative_path,
            "sha256": source.sha256,
            "split": source.split,
        },
        "valid_adjacent_cells": decision.valid_adjacent_cells,
    }


def _validated_eligibility_payload(
    split: str,
    decision: EligibilityDecision,
    source: SourceVerificationReceipt,
) -> tuple[PilotSplit, dict[str, Any]]:
    approved_split = _normalize_split(split)
    if source.split != approved_split:
        raise EligibilityManifestError("source and eligibility split do not agree")
    if (
        type(decision.association_ambiguous) is not bool
        or type(decision.association_one_to_one) is not bool
        or not np.isfinite(decision.pose_coverage)
        or not 0.0 <= decision.pose_coverage <= 1.0
    ):
        raise EligibilityManifestError("eligibility association/coverage fields are malformed")
    if decision.eligible and (
        decision.reasons
        or decision.association_ambiguous
        or not decision.association_one_to_one
        or not np.isfinite(decision.pose_coverage)
        or decision.pose_coverage < 0.80
        or decision.retained_distinct_clocks < 64
        or decision.valid_adjacent_cells < 63
        or decision.feature_frame_coverage < 0.60
    ):
        raise EligibilityManifestError(
            "an eligible decision violates the frozen count-blind predicates"
        )
    if not decision.eligible and not decision.reasons:
        raise EligibilityManifestError("an excluded decision requires a count-blind reason")
    required_reasons: set[str] = set()
    if decision.association_ambiguous:
        required_reasons.add("association_ambiguous")
    if not decision.association_one_to_one:
        required_reasons.add("association_not_one_to_one")
    if decision.pose_coverage < 0.80:
        required_reasons.add("pose_coverage_below_minimum")
    if decision.retained_distinct_clocks < 64:
        required_reasons.add("insufficient_distinct_clocks")
    if decision.valid_adjacent_cells < 63:
        required_reasons.add("insufficient_valid_adjacent_cells")
    if decision.feature_frame_coverage < 0.60:
        required_reasons.add("insufficient_feature_frame_coverage")
    if not required_reasons.issubset(decision.reasons):
        raise EligibilityManifestError("eligibility receipt omits a frozen exclusion reason")
    return approved_split, _eligibility_payload(decision, source)


def write_eligibility_audit(
    layout: PilotLayout,
    split: str,
    decision: EligibilityDecision,
    source: SourceVerificationReceipt,
) -> ArtifactReceipt:
    """Persist an audit-only, label-free eligibility decision."""

    approved_split, eligibility_payload = _validated_eligibility_payload(
        split,
        decision,
        source,
    )
    payload = _canonical_json_bytes(eligibility_payload)
    filename = f"{decision.opaque_sample_key}.{decision.local_person_slot}.eligibility.json"
    return _publish_with_receipt(
        layout,
        role="count_blind_eligibility",
        relative_path=Path("audit") / "eligibility" / approved_split / filename,
        audit_relative_path=Path("eligibility-receipts") / approved_split / f"{filename}.receipt.json",
        payload=payload,
        mode=0o600,
    )


def validate_development_component_coverage(
    decisions: Sequence[EligibilityDecision],
    *,
    component_by_identity: Mapping[tuple[str, int], str],
    expected_components: Sequence[str],
) -> dict[str, int]:
    """Require all nine original development components to remain nonempty."""

    frozen_components = tuple(expected_components)
    if len(frozen_components) != 9 or len(set(frozen_components)) != 9:
        raise EligibilityManifestError("exactly nine unique original components are required")
    counts = {component: 0 for component in frozen_components}
    for decision in decisions:
        identity = (decision.opaque_sample_key, decision.local_person_slot)
        if identity not in component_by_identity:
            raise EligibilityManifestError(f"missing source component for identity {identity}")
        component = component_by_identity[identity]
        if component not in counts:
            raise EligibilityManifestError(f"identity uses an unknown source component: {component}")
        if decision.eligible:
            counts[component] += 1
    empty = tuple(component for component in frozen_components if counts[component] == 0)
    if empty:
        raise EligibilityManifestError(
            f"original development components became empty after eligibility: {empty}"
        )
    return counts


def pack_identity(
    layout: PilotLayout,
    *,
    split: str,
    feature: FeatureShard,
    eligibility: EligibilityDecision,
    source: SourceVerificationReceipt,
    vault_factory: Callable[[], EvaluatorVaultRecord],
) -> PackedIdentityReceipt:
    """Pack one identity while opening evaluator labels only after eligibility.

    An excluded identity writes only its audit decision.  In particular,
    ``vault_factory`` is never called for an excluded slot.
    """

    if (
        feature.opaque_sample_key != eligibility.opaque_sample_key
        or feature.local_person_slot != eligibility.local_person_slot
    ):
        raise EligibilityManifestError("feature and eligibility routing identities differ")
    approved_split, eligibility_payload = _validated_eligibility_payload(
        split,
        eligibility,
        source,
    )
    if not eligibility.eligible:
        audit = write_eligibility_audit(layout, approved_split, eligibility, source)
        return PackedIdentityReceipt(
            eligibility=eligibility,
            audit=audit,
            feature=None,
            vault=None,
        )

    feature_payload, feature_members = deterministic_feature_npz_bytes(feature)
    filename = f"{feature.opaque_sample_key}.{feature.local_person_slot}"
    feature_prepared = _prepare_artifact(
        layout,
        role="feature_shard",
        relative_path=Path("features") / approved_split / f"{filename}.npz",
        audit_relative_path=(
            Path("feature-shards") / approved_split / f"{filename}.npz.receipt.json"
        ),
        payload=feature_payload,
        mode=0o600,
        members=feature_members,
    )

    verified_permission, permission_artifact = _require_persisted_permission_separation(layout)

    vault_relative = Path("vault") / approved_split / f"{filename}.json"
    vault_receipt_relative = (
        Path("vault-records") / approved_split / f"{filename}.json.receipt.json"
    )
    audit_relative = (
        Path("audit") / "eligibility" / approved_split / f"{filename}.eligibility.json"
    )
    audit_receipt_relative = (
        Path("eligibility-receipts")
        / approved_split
        / f"{filename}.eligibility.json.receipt.json"
    )
    identity_targets = (
        feature_prepared.receipt.path,
        feature_prepared.receipt.receipt_path,
        layout.run_root / vault_relative,
        layout.audit / vault_receipt_relative,
        layout.run_root / audit_relative,
        layout.audit / audit_receipt_relative,
    )
    partial = tuple(str(path) for path in identity_targets if _target_exists(path))
    if partial:
        raise FileExistsError(
            "refusing partial or existing identity publication targets: " + ", ".join(partial)
        )

    vault_record = vault_factory()
    if (
        vault_record.opaque_sample_key != feature.opaque_sample_key
        or vault_record.local_person_slot != feature.local_person_slot
        or vault_record.source_length != feature.source_length
    ):
        raise VaultSchemaError("feature/vault one-to-one join identity is inconsistent")
    vault_payload = _canonical_json_bytes(_validated_vault_payload(vault_record))
    vault_prepared = _prepare_artifact(
        layout,
        role="evaluator_vault_record",
        relative_path=vault_relative,
        audit_relative_path=vault_receipt_relative,
        payload=vault_payload,
        mode=0o600,
    )

    eligibility_payload["identity_publication"] = {
        "atomicity": {
            "commit_artifact_published_last": True,
            "crash_partial_targets_refused": True,
            "normal_failure_rollback": True,
            "private_same_filesystem_staging": True,
            "publication_primitive": "atomic_hard_link_no_replace",
        },
        "feature": {
            "relative_path": feature_prepared.receipt.relative_path,
            "receipt_sha256": feature_prepared.receipt.receipt_sha256,
            "sha256": feature_prepared.receipt.sha256,
        },
        "permission_separation": {
            "evidence": verified_permission,
            "relative_path": permission_artifact.relative_path,
            "receipt_sha256": permission_artifact.receipt_sha256,
            "sha256": permission_artifact.sha256,
        },
        "status": "PASS",
        "vault": {
            "relative_path": vault_prepared.receipt.relative_path,
            "receipt_sha256": vault_prepared.receipt.receipt_sha256,
            "sha256": vault_prepared.receipt.sha256,
        },
    }
    audit_prepared = _prepare_artifact(
        layout,
        role="count_blind_eligibility_identity_commit",
        relative_path=audit_relative,
        audit_relative_path=audit_receipt_relative,
        payload=_canonical_json_bytes(eligibility_payload),
        mode=0o600,
    )
    _publish_prepared_artifacts(
        layout,
        (feature_prepared, vault_prepared, audit_prepared),
        commit=audit_prepared,
    )
    return PackedIdentityReceipt(
        eligibility=eligibility,
        audit=audit_prepared.receipt,
        feature=feature_prepared.receipt,
        vault=vault_prepared.receipt,
    )
