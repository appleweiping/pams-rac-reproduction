"""Audited, label-free runner for the official ESCounts zero-shot demo.

The upstream project calls the method "Every Shot Counts" and publishes a
demo that combines a Kinetics-pretrained VideoMAE encoder with a
RepCount-trained decoder.  This module freezes that public zero-shot path at a
specific source revision, but deliberately does not redistribute either
checkpoint.

Prediction accepts only a strict :class:`~pams.data.PoseInputManifest` and its
independent commitment.  Counts, actions, dev targets, and test targets are
not imported or accepted anywhere in this module.  A primary 8 GiB allocator
tier and one finite 12 GiB OOM retry tier are represented by separate,
hash-bound artifacts so a retry cannot silently replace successful primary
predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Protocol, cast

from pams.data import (
    PoseInputCommitment,
    PoseInputManifest,
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
    validate_pose_input_binding,
)
from pams.metrics import round_count

METHOD_ID = "escounts-official-f18fcf1-zeroshot"
PREDICTION_ARTIFACT_TYPE = "escounts_official_predictions"
MERGED_ARTIFACT_TYPE = "escounts_official_merged_predictions"
RETRY_REQUEST_ARTIFACT_TYPE = "escounts_official_retry_request"
CLASSIFICATION = (
    "official-f18fcf1 RepCount-checkpoint zero-shot / independent label-free UCFRep evaluation"
)

OFFICIAL_SOURCE_REPOSITORY = "https://github.com/sinhasaptarshi/EveryShotCounts"
OFFICIAL_SOURCE_COMMIT = "f18fcf1933abb3d4f199fd12ee3cafc53581576a"
OFFICIAL_SOURCE_TREE = "3bedc8d117c4c86954fb58d891326e7c9c6a1a3d"
OFFICIAL_SOURCE_LICENSE = "MIT"
PYTORCHVIDEO_REPOSITORY = "https://github.com/facebookresearch/pytorchvideo"
PYTORCHVIDEO_COMMIT = "fae0d89a194a2c1ca99e59eab6eedd40bde38726"
PYTORCHVIDEO_TREE = "bd12d614191c96f3625f111284dec2b0a56a178a"

PRIMARY_RESOURCE_TIER = "allocator_8g_primary"
RETRY_RESOURCE_TIER = "allocator_12g_retry"
PRIMARY_MEMORY_LIMIT_BYTES = 8 * 1024 * 1024 * 1024
RETRY_MEMORY_LIMIT_BYTES = 12 * 1024 * 1024 * 1024
_RESOURCE_LIMITS = {
    PRIMARY_RESOURCE_TIER: PRIMARY_MEMORY_LIMIT_BYTES,
    RETRY_RESOURCE_TIER: RETRY_MEMORY_LIMIT_BYTES,
}
_RUNNER_PATH = Path(__file__).resolve(strict=True)
_WORKER_PATH = _RUNNER_PATH.with_name("escounts_official_worker.py").resolve(strict=True)

_FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "action",
        "count",
        "ground_truth",
        "ground_truth_count",
        "gt",
        "gt_count",
        "label",
        "labels",
        "target",
        "targets",
    }
)
_SHA256_HEX = frozenset("0123456789abcdef")
ENCODER_JUST_ENCODE_UNUSED_PARAMETERS = (
    {
        "name": "example_spatial_pos_embed",
        "shape": [196, 512],
        "dtype": "float32",
        "just_encode_unused": True,
    },
    {
        "name": "shot_token",
        "shape": [1568, 512],
        "dtype": "float32",
        "just_encode_unused": True,
    },
    {
        "name": "map.weight",
        "shape": [1, 512],
        "dtype": "float32",
        "just_encode_unused": True,
    },
    {
        "name": "map.bias",
        "shape": [1],
        "dtype": "float32",
        "just_encode_unused": True,
    },
)
ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_JSON = json.dumps(
    list(ENCODER_JUST_ENCODE_UNUSED_PARAMETERS),
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
)
ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_SHA256 = hashlib.sha256(
    ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_JSON.encode("utf-8")
).hexdigest()


def _canonical_sha256(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


def _canonical_git_sha(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in {40, 64}
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise ValueError(f"{field} must be a full lowercase Git object ID")
    return value


def _validate_portable_locator(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{field} must be a portable relative path")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ValueError(f"{field} must be relative")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"{field} contains an unsafe path component")
    return value


def sha256_json(payload: Any) -> str:
    """Return a canonical JSON SHA-256 without importing the model runtime."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_pairs(
    pairs: list[tuple[str, Any]],
    *,
    field: str,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"{field} contains duplicate field {key!r}")
        output[key] = value
    return output


def fsync_directory(path: str | Path) -> None:
    """Persist directory entries on POSIX and intentionally no-op on Windows."""

    if os.name != "posix":
        return
    descriptor = os.open(Path(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_mkdir(path: str | Path) -> Path:
    """Create a directory hierarchy and persist newly created POSIX entries."""

    target = Path(path)
    missing: list[Path] = []
    cursor = target
    while not cursor.exists():
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    if cursor.exists() and not cursor.is_dir():
        raise NotADirectoryError(f"directory ancestor is not a directory: {cursor}")
    for directory in reversed(missing):
        directory.mkdir()
        fsync_directory(directory.parent)
    if not target.is_dir():
        raise NotADirectoryError(f"path is not a directory: {target}")
    return target


class ESCountsOfficialError(RuntimeError):
    """Base class for an audited official-runner failure."""


class ESCountsSourceError(ESCountsOfficialError):
    """The upstream source checkout is not the frozen public revision."""


class ESCountsAssetError(ESCountsOfficialError):
    """A required external checkpoint is absent or has changed."""


class ESCountsDependencyError(ESCountsOfficialError):
    """The isolated runtime is missing an exact demo dependency."""


class _ESCountsAuditedVideoError(ESCountsOfficialError):
    def __init__(
        self,
        message: str,
        *,
        reported_frame_count: int = 0,
        decoded_frames: int = 0,
        tail_shortfall: int = 0,
    ) -> None:
        super().__init__(message)
        self.reported_frame_count = reported_frame_count
        self.decoded_frames = decoded_frames
        self.tail_shortfall = tail_shortfall


class ESCountsDecodeError(_ESCountsAuditedVideoError):
    """The official PyAV/OpenCV decode path failed."""


class ESCountsResourceExhaustedError(_ESCountsAuditedVideoError):
    """Inference exceeded the explicitly authorized allocator tier."""


class ESCountsMergeError(ESCountsOfficialError):
    """Primary and retry artifacts cannot be merged without mutation."""


@dataclass(frozen=True, slots=True)
class SourceFileSpec:
    """Frozen identity for one critical upstream source file."""

    relative_path: str
    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        _validate_portable_locator(self.relative_path, field="relative_path")
        if self.byte_count < 1:
            raise ValueError("source byte_count must be positive")
        _canonical_sha256(self.sha256, field="source sha256")


OFFICIAL_SOURCE_FILES: tuple[SourceFileSpec, ...] = (
    SourceFileSpec(
        "demo.py",
        7_368,
        "bce255b1274d6f3feedd42cf671125e844354f158c6865590e302fd704216f38",
    ),
    SourceFileSpec(
        "video_mae_cross_full_attention.py",
        21_511,
        "e1597a0ec2fcf5f76deb7b1396a9059de8b0edcca7ff8a64ba25762b5f498ace",
    ),
    SourceFileSpec(
        "configs/pretrain_config.yaml",
        1_876,
        "38960518be9cd640573c58a5aaf3e850f721dc92d445d5b8e787ddbbfed0bf67",
    ),
    SourceFileSpec(
        "slowfast/utils/parser.py",
        3_051,
        "76b08fd4392ccbe103c76480517cd2d1219033339756a7f47c71a78b4113d218",
    ),
    SourceFileSpec(
        "LICENSE",
        1_072,
        "8b2845569ad2b93013c3f71f0153713c6845a5ca79ff46151786491ebe740da7",
    ),
)


@dataclass(frozen=True, slots=True)
class AssetSpec:
    """Published identity and provenance for one external checkpoint."""

    role: str
    filename: str
    byte_count: int
    sha256: str
    source_url: str
    license_caveat: str

    def __post_init__(self) -> None:
        if not self.role or Path(self.filename).name != self.filename:
            raise ValueError("asset role and plain filename are required")
        if self.byte_count < 1:
            raise ValueError("asset byte_count must be positive")
        _canonical_sha256(self.sha256, field=f"{self.role} sha256")
        if not self.source_url.startswith("https://"):
            raise ValueError("asset source_url must use HTTPS")
        if not self.license_caveat:
            raise ValueError("asset license_caveat must be explicit")

    def public_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "filename": self.filename,
            "bytes": self.byte_count,
            "sha256": self.sha256,
            "source_url": self.source_url,
            "bundled": False,
            "license_caveat": self.license_caveat,
        }


_WEIGHT_LICENSE_CAVEAT = (
    "Checkpoint bytes are not redistributed. The upstream MIT source license "
    "does not by itself establish checkpoint or training-data redistribution "
    "rights; users must review the checkpoint host and dataset terms."
)

OFFICIAL_ENCODER = AssetSpec(
    role="kinetics_videomae_encoder",
    filename="VIT_B_16x4_MAE_PT.pyth",
    byte_count=1_207_498_009,
    sha256="b6d1d0b539dbdc992c3a3544a9bc5bbb0591179b615dc728f951285875d824e8",
    source_url=("https://dl.fbaipublicfiles.com/pyslowfast/masked_models/VIT_B_16x4_MAE_PT.pyth"),
    license_caveat=_WEIGHT_LICENSE_CAVEAT,
)
OFFICIAL_DECODER = AssetSpec(
    role="repcount_trained_decoder",
    filename="repcount_trained.pyth",
    byte_count=239_106_669,
    sha256="297fc53000417ac6ffa4e08d6876e033df89800c97632fa09d2f3945ba225559",
    source_url="https://drive.google.com/uc?id=1cwUtgUM0XotOx5fM4v4ZU29hlKUxze48",
    license_caveat=_WEIGHT_LICENSE_CAVEAT,
)


@dataclass(frozen=True, slots=True)
class ESCountsOfficialConfig:
    """Frozen values exercised by the upstream zero-shot ``demo.py`` path."""

    schema_version: int = 1
    source_commit: str = OFFICIAL_SOURCE_COMMIT
    encodings: str = "mae"
    num_frames_per_clip: int = 16
    clip_start_stride: int = 16
    clip_window_frames: int = 64
    encoded_clip_stride: int = 4
    transform_min_size: int = 224
    transform_crop_size: int = 224
    video_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    video_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    token_pool_ratio: float = 0.4
    decoder_window: tuple[int, int, int] = (4, 7, 7)
    shot_num: int = 0
    density_scale: float = 100.0
    rounding: str = "nearest_integer_half_up"

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.source_commit != OFFICIAL_SOURCE_COMMIT:
            raise ValueError("ESCounts source/config schema is frozen")
        if (
            self.encodings != "mae"
            or self.num_frames_per_clip != 16
            or self.clip_start_stride != 16
            or self.clip_window_frames != 64
            or self.encoded_clip_stride != 4
        ):
            raise ValueError("ESCounts temporal encoding schedule is frozen")
        if (
            self.transform_min_size != 224
            or self.transform_crop_size != 224
            or self.video_mean != (0.485, 0.456, 0.406)
            or self.video_std != (0.229, 0.224, 0.225)
        ):
            raise ValueError("ESCounts test transform is frozen")
        if (
            not math.isclose(self.token_pool_ratio, 0.4)
            or self.decoder_window != (4, 7, 7)
            or self.shot_num != 0
            or not math.isclose(self.density_scale, 100.0)
        ):
            raise ValueError("ESCounts zero-shot decoder settings are frozen")
        if self.rounding != "nearest_integer_half_up":
            raise ValueError("shared evaluator rounding is frozen")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
            "encodings": self.encodings,
            "num_frames_per_clip": self.num_frames_per_clip,
            "clip_start_stride": self.clip_start_stride,
            "clip_window_frames": self.clip_window_frames,
            "encoded_clip_stride": self.encoded_clip_stride,
            "transform_min_size": self.transform_min_size,
            "transform_crop_size": self.transform_crop_size,
            "video_mean": list(self.video_mean),
            "video_std": list(self.video_std),
            "token_pool_ratio": self.token_pool_ratio,
            "decoder_window": list(self.decoder_window),
            "shot_num": self.shot_num,
            "density_scale": self.density_scale,
            "rounding": self.rounding,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


FROZEN_ESCOUNTS_OFFICIAL_CONFIG = ESCountsOfficialConfig()


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    byte_count: int
    modified_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> _FileIdentity:
        return cls(
            device=value.st_dev,
            inode=value.st_ino,
            byte_count=value.st_size,
            modified_ns=value.st_mtime_ns,
        )


@dataclass(frozen=True, slots=True)
class _StableFileDigest:
    sha256: str
    byte_count: int
    identity: _FileIdentity


def _stable_file_digest(path: Path) -> _StableFileDigest:
    """Hash one regular non-symlink file and reject concurrent replacement."""

    try:
        before = path.lstat()
    except OSError:
        raise
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"input must be a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"input stopped being a regular file: {path}")
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
        closed = os.fstat(handle.fileno())
    after = path.lstat()
    identities = tuple(_FileIdentity.from_stat(value) for value in (before, opened, closed, after))
    if len(set(identities)) != 1 or byte_count != before.st_size:
        raise RuntimeError(f"input changed while it was hashed: {path}")
    return _StableFileDigest(
        sha256=digest.hexdigest(),
        byte_count=byte_count,
        identity=identities[0],
    )


def _assert_file_unchanged(
    path: Path,
    expected: _StableFileDigest,
    *,
    role: str,
) -> None:
    """Rehash a critical input; metadata-only checks are not sufficient."""

    try:
        observed = _stable_file_digest(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"{role} disappeared during prediction: {path}") from exc
    if observed != expected:
        raise RuntimeError(f"{role} changed during prediction: {path}")


@dataclass(frozen=True, slots=True)
class VerifiedSourceFile:
    spec: SourceFileSpec
    path: Path
    _digest: _StableFileDigest

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.spec.relative_path,
            "bytes": self.spec.byte_count,
            "sha256": self.spec.sha256,
        }


@dataclass(frozen=True, slots=True)
class VerifiedOfficialSource:
    """Clean frozen upstream checkout plus critical-file identities."""

    root: Path
    files: tuple[VerifiedSourceFile, ...]
    commit: str = OFFICIAL_SOURCE_COMMIT
    tree: str = OFFICIAL_SOURCE_TREE

    def assert_unchanged(self) -> None:
        for item in self.files:
            _assert_file_unchanged(
                item.path,
                item._digest,
                role=f"official source file {item.spec.relative_path!r}",
            )
        commit, tree, status = _git_source_state(self.root)
        if commit != self.commit or tree != self.tree or status:
            raise ESCountsSourceError("official source checkout changed during prediction")

    def to_dict(self) -> dict[str, Any]:
        return {
            "repository": OFFICIAL_SOURCE_REPOSITORY,
            "commit": self.commit,
            "tree": self.tree,
            "license": OFFICIAL_SOURCE_LICENSE,
            "files": [item.to_dict() for item in self.files],
            "pytorchvideo_repository": PYTORCHVIDEO_REPOSITORY,
            "pytorchvideo_commit": PYTORCHVIDEO_COMMIT,
            "pytorchvideo_tree": PYTORCHVIDEO_TREE,
        }


def _git_text(root: Path, arguments: Sequence[str]) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ESCountsSourceError(f"unable to audit Git checkout: {root}") from exc


def _git_source_state(root: Path) -> tuple[str, str, str]:
    commit = _git_text(root, ("rev-parse", "HEAD"))
    tree = _git_text(root, ("rev-parse", "HEAD^{tree}"))
    status = _git_text(
        root,
        ("status", "--porcelain=v1", "--untracked-files=all"),
    )
    return commit, tree, status


def _tracked_repository_file_digest(
    repository_root: Path,
    path: Path,
    *,
    role: str,
) -> _StableFileDigest:
    resolved = path.expanduser().resolve(strict=True)
    if resolved == repository_root or repository_root not in resolved.parents:
        raise ESCountsSourceError(f"{role} is outside the audited repository: {resolved}")
    relative_path = resolved.relative_to(repository_root).as_posix()
    tracked = _git_text(
        repository_root,
        ("ls-files", "--error-unmatch", "--", relative_path),
    ).splitlines()
    if tracked != [relative_path]:
        raise ESCountsSourceError(f"{role} is not uniquely tracked: {relative_path}")
    head_blob = _git_text(repository_root, ("rev-parse", f"HEAD:{relative_path}"))
    working_blob = _git_text(
        repository_root,
        ("hash-object", "--no-filters", "--", relative_path),
    )
    if head_blob != working_blob:
        raise ESCountsSourceError(f"{role} bytes differ from HEAD: {relative_path}")
    return _stable_file_digest(resolved)


@dataclass(frozen=True, slots=True)
class VerifiedRunnerRepository:
    """Clean project checkout containing the exact runner and worker bytes."""

    root: Path
    git_sha: str
    git_tree: str
    runner_path: Path
    runner_digest: _StableFileDigest
    worker_path: Path
    worker_digest: _StableFileDigest

    def assert_unchanged(self) -> None:
        commit, tree, status = _git_source_state(self.root)
        if commit != self.git_sha or tree != self.git_tree or status:
            raise ESCountsSourceError("runner repository changed or became dirty")
        _assert_file_unchanged(
            self.runner_path,
            self.runner_digest,
            role="ESCounts host runner",
        )
        _assert_file_unchanged(
            self.worker_path,
            self.worker_digest,
            role="ESCounts JSONL worker",
        )

    def provenance_dict(self) -> dict[str, str]:
        return {
            "runner_source_git_sha": self.git_sha,
            "runner_code_sha256": self.runner_digest.sha256,
            "worker_code_sha256": self.worker_digest.sha256,
        }

    def verify_tracked_file(self, path: str | Path, *, role: str) -> _StableFileDigest:
        self.assert_unchanged()
        return _tracked_repository_file_digest(
            self.root,
            Path(path),
            role=role,
        )


def verify_runner_repository(repository_root: str | Path) -> VerifiedRunnerRepository:
    """Verify the executing runner/worker belong to one clean full-SHA checkout."""

    root = Path(repository_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ESCountsSourceError(f"runner repository is not a directory: {root}")
    top_level = Path(_git_text(root, ("rev-parse", "--show-toplevel"))).resolve(strict=True)
    if top_level != root:
        raise ESCountsSourceError("repository_root must be the Git checkout top level")
    commit, tree, status = _git_source_state(root)
    _canonical_git_sha(commit, field="runner repository Git SHA")
    _canonical_git_sha(tree, field="runner repository Git tree")
    if status:
        raise ESCountsSourceError("runner repository must be clean")
    runner_digest = _tracked_repository_file_digest(
        root,
        _RUNNER_PATH,
        role="ESCounts host runner",
    )
    worker_digest = _tracked_repository_file_digest(
        root,
        _WORKER_PATH,
        role="ESCounts JSONL worker",
    )
    verified = VerifiedRunnerRepository(
        root=root,
        git_sha=commit,
        git_tree=tree,
        runner_path=_RUNNER_PATH,
        runner_digest=runner_digest,
        worker_path=_WORKER_PATH,
        worker_digest=worker_digest,
    )
    verified.assert_unchanged()
    return verified


def _verify_source_files_with_specs(
    root: str | Path,
    specs: Sequence[SourceFileSpec],
) -> tuple[VerifiedSourceFile, ...]:
    directory = Path(root).expanduser().resolve(strict=True)
    if not directory.is_dir():
        raise ESCountsSourceError(f"source root is not a directory: {directory}")
    observed: list[VerifiedSourceFile] = []
    for spec in specs:
        path = directory / PurePosixPath(spec.relative_path)
        try:
            digest = _stable_file_digest(path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise ESCountsSourceError(
                f"unable to verify official source file {spec.relative_path!r}: {exc}"
            ) from exc
        if digest.byte_count != spec.byte_count or digest.sha256 != spec.sha256:
            raise ESCountsSourceError(
                f"official source file identity mismatch: {spec.relative_path}"
            )
        observed.append(VerifiedSourceFile(spec=spec, path=path, _digest=digest))
    return tuple(observed)


def verify_official_source(source_root: str | Path) -> VerifiedOfficialSource:
    """Verify the clean exact upstream commit, tree, and critical source files."""

    root = Path(source_root).expanduser().resolve(strict=True)
    commit, tree, status = _git_source_state(root)
    if commit != OFFICIAL_SOURCE_COMMIT:
        raise ESCountsSourceError(
            f"official source commit mismatch: {commit} != {OFFICIAL_SOURCE_COMMIT}"
        )
    if tree != OFFICIAL_SOURCE_TREE:
        raise ESCountsSourceError("official source tree does not match frozen commit")
    if status:
        raise ESCountsSourceError("official source checkout must be clean")
    return VerifiedOfficialSource(
        root=root,
        files=_verify_source_files_with_specs(root, OFFICIAL_SOURCE_FILES),
    )


@dataclass(frozen=True, slots=True)
class VerifiedAsset:
    spec: AssetSpec
    path: Path
    _digest: _StableFileDigest

    def assert_unchanged(self) -> None:
        _assert_file_unchanged(
            self.path,
            self._digest,
            role=f"official asset {self.spec.role!r}",
        )

    def to_dict(self) -> dict[str, Any]:
        return self.spec.public_dict()


@dataclass(frozen=True, slots=True)
class VerifiedAssets:
    encoder: VerifiedAsset
    decoder: VerifiedAsset
    aggregate_sha256: str

    def assert_unchanged(self) -> None:
        self.encoder.assert_unchanged()
        self.decoder.assert_unchanged()

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate_sha256": self.aggregate_sha256,
            "files": [self.encoder.to_dict(), self.decoder.to_dict()],
        }


def _verify_asset_with_spec(path: str | Path, spec: AssetSpec) -> VerifiedAsset:
    resolved = Path(path).expanduser().resolve(strict=True)
    try:
        digest = _stable_file_digest(resolved)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ESCountsAssetError(f"unable to verify {spec.role} checkpoint: {exc}") from exc
    if digest.byte_count != spec.byte_count or digest.sha256 != spec.sha256:
        raise ESCountsAssetError(f"{spec.role} checkpoint byte count/SHA-256 mismatch")
    return VerifiedAsset(spec=spec, path=resolved, _digest=digest)


def verify_official_assets(
    encoder_path: str | Path,
    decoder_path: str | Path,
) -> VerifiedAssets:
    """Verify both external checkpoint files without loading PyTorch."""

    encoder = _verify_asset_with_spec(encoder_path, OFFICIAL_ENCODER)
    decoder = _verify_asset_with_spec(decoder_path, OFFICIAL_DECODER)
    canonical = {"files": [encoder.to_dict(), decoder.to_dict()]}
    return VerifiedAssets(
        encoder=encoder,
        decoder=decoder,
        aggregate_sha256=sha256_json(canonical),
    )


@dataclass(frozen=True, slots=True)
class VerifiedLabelFreeInputs:
    manifest: PoseInputManifest
    commitment: PoseInputCommitment
    video_root: Path
    sidecar_sha256: str
    commitment_sha256: str
    identity_sha256: str
    _sidecar_path: Path
    _commitment_path: Path
    _sidecar_digest: _StableFileDigest
    _commitment_digest: _StableFileDigest

    def assert_unchanged(self) -> None:
        _assert_file_unchanged(
            self._sidecar_path,
            self._sidecar_digest,
            role="label-free input sidecar",
        )
        _assert_file_unchanged(
            self._commitment_path,
            self._commitment_digest,
            role="label-free input commitment",
        )

    def binding_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.manifest.protocol,
            "split": self.manifest.split,
            "sample_count": len(self.manifest.records),
            "sidecar_sha256": self.sidecar_sha256,
            "commitment_sha256": self.commitment_sha256,
            "sidecar_fingerprint": self.manifest.fingerprint,
            "identity_sha256": self.identity_sha256,
        }


def _load_label_free_inputs(
    sidecar_path: str | Path,
    commitment_path: str | Path,
    video_root: str | Path,
    *,
    require_exact_membership: bool,
) -> VerifiedLabelFreeInputs:
    """Load and bind label-free inputs before source, assets, or Torch."""

    sidecar = Path(sidecar_path).expanduser().resolve(strict=True)
    commitment_file = Path(commitment_path).expanduser().resolve(strict=True)
    root = Path(video_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"video_root is not a directory: {root}")
    sidecar_digest = _stable_file_digest(sidecar)
    commitment_digest = _stable_file_digest(commitment_file)
    for label_free_path, document_name in (
        (sidecar, "label-free input sidecar"),
        (commitment_file, "label-free input commitment"),
    ):
        try:
            encoded = label_free_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"{document_name} must be UTF-8 JSON") from exc
        _reject_forbidden_keys(encoded, document_name=document_name)
    manifest = load_pose_input_manifest(
        sidecar,
        validate_exact=require_exact_membership,
    )
    commitment = load_pose_input_commitment(commitment_file)
    validate_pose_input_binding(
        manifest,
        commitment,
        sidecar_sha256=sidecar_digest.sha256,
    )
    _assert_file_unchanged(
        sidecar,
        sidecar_digest,
        role="label-free input sidecar",
    )
    _assert_file_unchanged(
        commitment_file,
        commitment_digest,
        role="label-free input commitment",
    )
    return VerifiedLabelFreeInputs(
        manifest=manifest,
        commitment=commitment,
        video_root=root,
        sidecar_sha256=sidecar_digest.sha256,
        commitment_sha256=commitment_digest.sha256,
        identity_sha256=pose_input_identity_sha256(manifest.records),
        _sidecar_path=sidecar,
        _commitment_path=commitment_file,
        _sidecar_digest=sidecar_digest,
        _commitment_digest=commitment_digest,
    )


@dataclass(frozen=True, slots=True)
class BackendPrediction:
    raw_count: float
    reported_frame_count: int
    decoded_frames: int
    tail_shortfall: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.raw_count) or self.raw_count < 0.0:
            raise ValueError("official ESCounts raw_count must be finite and non-negative")
        if (
            self.reported_frame_count < 1
            or self.decoded_frames < 1
            or self.tail_shortfall not in {0, 1}
            or self.reported_frame_count - self.decoded_frames
            != self.tail_shortfall
        ):
            raise ValueError("successful ESCounts frame audit is inconsistent")


class _OfficialBackend(Protocol):
    source_commit: str
    source_tree: str
    runtime_versions: Mapping[str, str]

    def predict(
        self,
        video_path: Path,
        config: ESCountsOfficialConfig,
    ) -> BackendPrediction:
        """Run the frozen official zero-shot preprocessing and model path."""


class PredictionStatus(str, Enum):
    OK = "ok"
    MISSING_VIDEO = "missing_video"
    NON_REGULAR_VIDEO = "non_regular_video"
    VIDEO_HASH_MISMATCH = "video_hash_mismatch"
    VIDEO_CHANGED = "video_changed"
    DECODE_FAILED = "decode_failed"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    INFERENCE_FAILED = "inference_failed"


@dataclass(frozen=True, slots=True)
class ESCountsVideoResult:
    index: int
    video_id: str
    video_locator: str
    expected_video_sha256: str
    observed_video_sha256: str | None
    raw_count: float | None
    rounded_count: int | None
    finite: bool
    reported_frame_count: int
    decoded_frames: int
    tail_shortfall: int
    elapsed_seconds: float
    resource_tier: str
    memory_limit_bytes: int
    status: PredictionStatus
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("prediction index must be non-negative")
        _validate_portable_locator(self.video_locator, field="video_locator")
        _canonical_sha256(
            self.expected_video_sha256,
            field="expected_video_sha256",
        )
        if self.observed_video_sha256 is not None:
            _canonical_sha256(
                self.observed_video_sha256,
                field="observed_video_sha256",
            )
        if (
            not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0.0
            or self.reported_frame_count < 0
            or self.decoded_frames < 0
            or self.tail_shortfall < 0
        ):
            raise ValueError("elapsed_seconds/frame audit fields are invalid")
        if self.reported_frame_count == 0:
            if self.decoded_frames != 0 or self.tail_shortfall != 0:
                raise ValueError("absent frame audit must be all zero")
        elif (
            self.decoded_frames < 1
            or self.reported_frame_count - self.decoded_frames
            != self.tail_shortfall
        ):
            raise ValueError("frame audit fields are inconsistent")
        if _RESOURCE_LIMITS.get(self.resource_tier) != self.memory_limit_bytes:
            raise ValueError("resource tier/memory limit mismatch")
        successful = self.status is PredictionStatus.OK
        if successful:
            if (
                self.raw_count is None
                or self.rounded_count is None
                or not self.finite
                or self.failure_reason is not None
                or self.decoded_frames < 1
                or self.tail_shortfall not in {0, 1}
            ):
                raise ValueError("successful prediction fields are incomplete")
            if (
                not math.isfinite(self.raw_count)
                or self.raw_count < 0.0
                or self.rounded_count != round_count(self.raw_count)
                or self.observed_video_sha256 != self.expected_video_sha256
            ):
                raise ValueError("successful prediction values are inconsistent")
        elif (
            self.raw_count is not None
            or self.rounded_count is not None
            or self.finite
            or not self.failure_reason
        ):
            raise ValueError("failed prediction contains synthetic count output")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "video_id": self.video_id,
            "video_locator": self.video_locator,
            "expected_video_sha256": self.expected_video_sha256,
            "observed_video_sha256": self.observed_video_sha256,
            "raw_count": self.raw_count,
            "rounded_count": self.rounded_count,
            "finite": self.finite,
            "reported_frame_count": self.reported_frame_count,
            "decoded_frames": self.decoded_frames,
            "tail_shortfall": self.tail_shortfall,
            "elapsed_seconds": self.elapsed_seconds,
            "resource_tier": self.resource_tier,
            "memory_limit_bytes": self.memory_limit_bytes,
            "status": self.status.value,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class ESCountsOfficialRunResult:
    inputs: VerifiedLabelFreeInputs
    source: VerifiedOfficialSource
    assets: VerifiedAssets
    config: ESCountsOfficialConfig
    selected_video_ids: tuple[str, ...]
    rows: tuple[ESCountsVideoResult, ...]
    runtime_versions: Mapping[str, str]
    resource_tier: str
    runner_source_git_sha: str
    runner_code_sha256: str
    worker_code_sha256: str
    retry_binding: Mapping[str, Any] | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only ESCounts result schema_version=1 is supported")
        _canonical_git_sha(
            self.runner_source_git_sha,
            field="runner_source_git_sha",
        )
        _canonical_sha256(self.runner_code_sha256, field="runner_code_sha256")
        _canonical_sha256(self.worker_code_sha256, field="worker_code_sha256")
        if self.runtime_versions.get("worker_code_sha256") != self.worker_code_sha256:
            raise ValueError("worker runtime/hash provenance differs")
        if _RESOURCE_LIMITS.get(self.resource_tier) is None:
            raise ValueError("unknown ESCounts resource tier")
        full_ids = tuple(record.video_id for record in self.inputs.manifest.records)
        selected_set = set(self.selected_video_ids)
        expected_ids = tuple(video_id for video_id in full_ids if video_id in selected_set)
        if (
            not self.selected_video_ids
            or len(selected_set) != len(self.selected_video_ids)
            or expected_ids != self.selected_video_ids
        ):
            raise ValueError("selection must be a unique sidecar-ordered subset")
        if tuple(row.video_id for row in self.rows) != self.selected_video_ids:
            raise ValueError("prediction rows must follow the selected sidecar order")
        if any(row.resource_tier != self.resource_tier for row in self.rows):
            raise ValueError("row resource tier differs from run resource tier")
        if self.resource_tier == PRIMARY_RESOURCE_TIER and self.retry_binding is not None:
            raise ValueError("primary result cannot contain retry_binding")
        if self.resource_tier == RETRY_RESOURCE_TIER and self.retry_binding is None:
            raise ValueError("retry result requires retry_binding")

    @property
    def successes(self) -> tuple[ESCountsVideoResult, ...]:
        return tuple(row for row in self.rows if row.status is PredictionStatus.OK)

    @property
    def failures(self) -> tuple[ESCountsVideoResult, ...]:
        return tuple(row for row in self.rows if row.status is not PredictionStatus.OK)

    def to_dict(self) -> dict[str, Any]:
        success_rows = [row.to_dict() for row in self.successes]
        failure_rows = [row.to_dict() for row in self.failures]
        memory_limit = _RESOURCE_LIMITS[self.resource_tier]
        return {
            "schema_version": self.schema_version,
            "artifact_type": PREDICTION_ARTIFACT_TYPE,
            "method_id": METHOD_ID,
            "classification": CLASSIFICATION,
            "eligible_for_original_pams_escounts_cell": False,
            "target_access": False,
            "status": "complete" if not failure_rows else "partial",
            "source": self.source.to_dict(),
            "input": self.inputs.binding_dict(),
            "selection": {
                "selected_count": len(self.selected_video_ids),
                "selected_video_ids": list(self.selected_video_ids),
                "selected_video_ids_sha256": sha256_json(list(self.selected_video_ids)),
            },
            "assets": self.assets.to_dict(),
            "config": {
                "sha256": self.config.fingerprint,
                "values": self.config.to_dict(),
            },
            "runtime_versions": dict(sorted(self.runtime_versions.items())),
            "resource": {
                "tier": self.resource_tier,
                "memory_limit_bytes": memory_limit,
            },
            "retry_binding": (None if self.retry_binding is None else dict(self.retry_binding)),
            "runner_source_git_sha": self.runner_source_git_sha,
            "runner_code_sha256": self.runner_code_sha256,
            "worker_code_sha256": self.worker_code_sha256,
            "record_total": len(self.rows),
            "success_total": len(success_rows),
            "failure_total": len(failure_rows),
            "predictions": success_rows,
            "failures": failure_rows,
        }


def _select_records(
    inputs: VerifiedLabelFreeInputs,
    video_ids: Sequence[str] | None,
) -> tuple[tuple[int, Any], ...]:
    records = inputs.manifest.records
    if video_ids is None:
        return tuple(enumerate(records))
    requested = tuple(video_ids)
    if not requested:
        raise ValueError("video_ids must contain at least one identifier")
    if len(set(requested)) != len(requested):
        raise ValueError("video_ids must not contain duplicates")
    available = {record.video_id for record in records}
    missing = sorted(set(requested) - available)
    if missing:
        raise ValueError(f"video_ids are not present in verified sidecar: {missing}")
    requested_set = set(requested)
    return tuple(
        (index, record) for index, record in enumerate(records) if record.video_id in requested_set
    )


def _resolved_video_path(inputs: VerifiedLabelFreeInputs, locator: str) -> Path:
    _validate_portable_locator(locator, field="video_locator")
    candidate = inputs.video_root.joinpath(*PurePosixPath(locator).parts)
    resolved = candidate.resolve(strict=False)
    root = inputs.video_root.resolve(strict=True)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"video locator escapes verified root: {locator!r}")
    return resolved


def _failure(
    *,
    index: int,
    video_id: str,
    locator: str,
    expected_sha256: str,
    observed_sha256: str | None,
    elapsed_seconds: float,
    resource_tier: str,
    status: PredictionStatus,
    reason: str,
    reported_frame_count: int = 0,
    decoded_frames: int = 0,
    tail_shortfall: int = 0,
) -> ESCountsVideoResult:
    return ESCountsVideoResult(
        index=index,
        video_id=video_id,
        video_locator=locator,
        expected_video_sha256=expected_sha256,
        observed_video_sha256=observed_sha256,
        raw_count=None,
        rounded_count=None,
        finite=False,
        reported_frame_count=reported_frame_count,
        decoded_frames=decoded_frames,
        tail_shortfall=tail_shortfall,
        elapsed_seconds=elapsed_seconds,
        resource_tier=resource_tier,
        memory_limit_bytes=_RESOURCE_LIMITS[resource_tier],
        status=status,
        failure_reason=reason,
    )


def _run_verified_backend(
    inputs: VerifiedLabelFreeInputs,
    source: VerifiedOfficialSource,
    assets: VerifiedAssets,
    backend: _OfficialBackend,
    runner_repository: VerifiedRunnerRepository,
    *,
    video_ids: Sequence[str] | None = None,
    resource_tier: str = PRIMARY_RESOURCE_TIER,
    retry_binding: Mapping[str, Any] | None = None,
) -> ESCountsOfficialRunResult:
    """Execute a verified backend without exposing any target loader."""

    if _RESOURCE_LIMITS.get(resource_tier) is None:
        raise ValueError(f"unknown resource_tier {resource_tier!r}")
    if backend.source_commit != source.commit or backend.source_tree != source.tree:
        raise ESCountsSourceError("backend source binding differs from verified checkout")
    runner_repository.assert_unchanged()
    if backend.runtime_versions.get("worker_code_sha256") != runner_repository.worker_digest.sha256:
        raise ESCountsDependencyError(
            "backend worker bytes differ from the verified runner checkout"
        )
    selection = _select_records(inputs, video_ids)
    rows: list[ESCountsVideoResult] = []
    for index, record in selection:
        started = time.perf_counter()
        assert record.video_sha256 is not None
        path = _resolved_video_path(inputs, record.video_path)
        try:
            file_stat = path.lstat()
        except FileNotFoundError:
            rows.append(
                _failure(
                    index=index,
                    video_id=record.video_id,
                    locator=record.video_path,
                    expected_sha256=record.video_sha256,
                    observed_sha256=None,
                    elapsed_seconds=time.perf_counter() - started,
                    resource_tier=resource_tier,
                    status=PredictionStatus.MISSING_VIDEO,
                    reason="source video is missing",
                )
            )
            continue
        if stat.S_ISLNK(file_stat.st_mode) or not stat.S_ISREG(file_stat.st_mode):
            rows.append(
                _failure(
                    index=index,
                    video_id=record.video_id,
                    locator=record.video_path,
                    expected_sha256=record.video_sha256,
                    observed_sha256=None,
                    elapsed_seconds=time.perf_counter() - started,
                    resource_tier=resource_tier,
                    status=PredictionStatus.NON_REGULAR_VIDEO,
                    reason="source video is not a regular non-symlink file",
                )
            )
            continue
        try:
            digest = _stable_file_digest(path)
        except (OSError, RuntimeError, ValueError) as exc:
            rows.append(
                _failure(
                    index=index,
                    video_id=record.video_id,
                    locator=record.video_path,
                    expected_sha256=record.video_sha256,
                    observed_sha256=None,
                    elapsed_seconds=time.perf_counter() - started,
                    resource_tier=resource_tier,
                    status=PredictionStatus.VIDEO_CHANGED,
                    reason=str(exc),
                )
            )
            continue
        if digest.sha256 != record.video_sha256:
            rows.append(
                _failure(
                    index=index,
                    video_id=record.video_id,
                    locator=record.video_path,
                    expected_sha256=record.video_sha256,
                    observed_sha256=digest.sha256,
                    elapsed_seconds=time.perf_counter() - started,
                    resource_tier=resource_tier,
                    status=PredictionStatus.VIDEO_HASH_MISMATCH,
                    reason="source video SHA-256 does not match sidecar",
                )
            )
            continue
        try:
            prediction = backend.predict(path, FROZEN_ESCOUNTS_OFFICIAL_CONFIG)
            _assert_file_unchanged(path, digest, role="source video")
        except ESCountsResourceExhaustedError as exc:
            rows.append(
                _failure(
                    index=index,
                    video_id=record.video_id,
                    locator=record.video_path,
                    expected_sha256=record.video_sha256,
                    observed_sha256=digest.sha256,
                    elapsed_seconds=time.perf_counter() - started,
                    resource_tier=resource_tier,
                    status=PredictionStatus.RESOURCE_EXHAUSTED,
                    reason=str(exc),
                    reported_frame_count=exc.reported_frame_count,
                    decoded_frames=exc.decoded_frames,
                    tail_shortfall=exc.tail_shortfall,
                )
            )
        except ESCountsDecodeError as exc:
            rows.append(
                _failure(
                    index=index,
                    video_id=record.video_id,
                    locator=record.video_path,
                    expected_sha256=record.video_sha256,
                    observed_sha256=digest.sha256,
                    elapsed_seconds=time.perf_counter() - started,
                    resource_tier=resource_tier,
                    status=PredictionStatus.DECODE_FAILED,
                    reason=str(exc),
                    reported_frame_count=exc.reported_frame_count,
                    decoded_frames=exc.decoded_frames,
                    tail_shortfall=exc.tail_shortfall,
                )
            )
        except (OSError, RuntimeError, ValueError) as exc:
            rows.append(
                _failure(
                    index=index,
                    video_id=record.video_id,
                    locator=record.video_path,
                    expected_sha256=record.video_sha256,
                    observed_sha256=digest.sha256,
                    elapsed_seconds=time.perf_counter() - started,
                    resource_tier=resource_tier,
                    status=PredictionStatus.INFERENCE_FAILED,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            rows.append(
                ESCountsVideoResult(
                    index=index,
                    video_id=record.video_id,
                    video_locator=record.video_path,
                    expected_video_sha256=record.video_sha256,
                    observed_video_sha256=digest.sha256,
                    raw_count=prediction.raw_count,
                    rounded_count=round_count(prediction.raw_count),
                    finite=True,
                    reported_frame_count=prediction.reported_frame_count,
                    decoded_frames=prediction.decoded_frames,
                    tail_shortfall=prediction.tail_shortfall,
                    elapsed_seconds=time.perf_counter() - started,
                    resource_tier=resource_tier,
                    memory_limit_bytes=_RESOURCE_LIMITS[resource_tier],
                    status=PredictionStatus.OK,
                )
            )
    inputs.assert_unchanged()
    source.assert_unchanged()
    assets.assert_unchanged()
    runner_repository.assert_unchanged()
    return ESCountsOfficialRunResult(
        inputs=inputs,
        source=source,
        assets=assets,
        config=FROZEN_ESCOUNTS_OFFICIAL_CONFIG,
        selected_video_ids=tuple(record.video_id for _, record in selection),
        rows=tuple(rows),
        runtime_versions=backend.runtime_versions,
        resource_tier=resource_tier,
        runner_source_git_sha=runner_repository.git_sha,
        runner_code_sha256=runner_repository.runner_digest.sha256,
        worker_code_sha256=runner_repository.worker_digest.sha256,
        retry_binding=retry_binding,
    )


_WORKER_PROTOCOL_VERSION = 1


def _jsonl_message(payload: Mapping[str, Any]) -> dict[str, Any]:
    message = dict(payload)
    message["message_sha256"] = sha256_json(message)
    return message


def _parse_worker_command(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("worker command must be a JSON string array") from exc
    else:
        raw = list(value)
    if (
        not isinstance(raw, list)
        or not raw
        or any(not isinstance(item, str) or not item for item in raw)
    ):
        raise ValueError("worker command must contain one or more non-empty strings")
    return tuple(raw)


def _validate_container_image_id(value: str) -> str:
    prefix = "sha256:"
    digest = value.removeprefix(prefix)
    if not value.startswith(prefix):
        raise ValueError("container image ID must start with sha256:")
    _canonical_sha256(digest, field="container image ID")
    return value


class JSONLWorkerBackend:
    """Host-side proxy for one long-lived Python 3.8 ESCounts worker."""

    source_commit = OFFICIAL_SOURCE_COMMIT
    source_tree = OFFICIAL_SOURCE_TREE

    def __init__(
        self,
        source: VerifiedOfficialSource,
        assets: VerifiedAssets,
        inputs: VerifiedLabelFreeInputs,
        runner_repository: VerifiedRunnerRepository,
        *,
        worker_command: str | Sequence[str],
        pytorchvideo_source_root: str | Path,
        expected_container_image_id: str,
        device: str,
        memory_limit_bytes: int,
    ) -> None:
        if memory_limit_bytes not in _RESOURCE_LIMITS.values():
            raise ValueError("memory_limit_bytes must be an authorized resource tier")
        self._source = source
        self._assets = assets
        self._inputs = inputs
        self._device = device
        self._memory_limit_bytes = memory_limit_bytes
        self._container_image_id = _validate_container_image_id(expected_container_image_id)
        runner_repository.assert_unchanged()
        self._worker_digest = runner_repository.worker_digest
        pytorchvideo_root = Path(pytorchvideo_source_root).expanduser().resolve(strict=True)
        pytorchvideo_commit, pytorchvideo_tree, pytorchvideo_status = _git_source_state(
            pytorchvideo_root
        )
        if (
            pytorchvideo_commit != PYTORCHVIDEO_COMMIT
            or pytorchvideo_tree != PYTORCHVIDEO_TREE
            or pytorchvideo_status
        ):
            raise ESCountsSourceError("PyTorchVideo checkout is not the frozen clean commit/tree")
        worker_command_parts = _parse_worker_command(worker_command)
        self._worker_command_sha256 = sha256_json(worker_command_parts)
        command = [
            *worker_command_parts,
            "--source-root",
            str(source.root),
            "--pytorchvideo-source-root",
            str(pytorchvideo_root),
            "--video-root",
            str(inputs.video_root),
            "--encoder",
            str(assets.encoder.path),
            "--decoder",
            str(assets.decoder.path),
            "--device",
            device,
            "--memory-limit-bytes",
            str(memory_limit_bytes),
            "--container-image-id",
            self._container_image_id,
        ]
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise ESCountsDependencyError("unable to start ESCounts JSONL worker") from exc
        self._closed = False
        self._sequence = 0
        nonce = secrets.token_hex(16)
        try:
            ready = self._exchange(
                {
                    "schema_version": _WORKER_PROTOCOL_VERSION,
                    "type": "init",
                    "nonce": nonce,
                    "source_commit": OFFICIAL_SOURCE_COMMIT,
                    "source_tree": OFFICIAL_SOURCE_TREE,
                    "pytorchvideo_commit": PYTORCHVIDEO_COMMIT,
                    "pytorchvideo_tree": PYTORCHVIDEO_TREE,
                    "encoder_sha256": OFFICIAL_ENCODER.sha256,
                    "decoder_sha256": OFFICIAL_DECODER.sha256,
                    "config_sha256": FROZEN_ESCOUNTS_OFFICIAL_CONFIG.fingerprint,
                    "worker_sha256": self._worker_digest.sha256,
                    "container_image_id": self._container_image_id,
                    "memory_limit_bytes": memory_limit_bytes,
                    "device": device,
                }
            )
        except Exception:
            self.close(force=True)
            raise
        if (
            ready.get("schema_version") != _WORKER_PROTOCOL_VERSION
            or ready.get("type") != "ready"
            or ready.get("nonce") != nonce
            or ready.get("worker_sha256") != self._worker_digest.sha256
            or ready.get("container_image_id") != self._container_image_id
            or ready.get("source_commit") != OFFICIAL_SOURCE_COMMIT
            or ready.get("source_tree") != OFFICIAL_SOURCE_TREE
        ):
            self.close(force=True)
            raise ESCountsDependencyError("worker readiness binding mismatch")
        if set(ready) != {
            "schema_version",
            "type",
            "nonce",
            "worker_sha256",
            "container_image_id",
            "source_commit",
            "source_tree",
            "pytorchvideo_origin",
            "module_origins",
            "runtime_versions",
        }:
            self.close(force=True)
            raise ESCountsDependencyError("worker readiness fields changed")
        origins = ready.get("module_origins")
        pytorchvideo_origin = ready.get("pytorchvideo_origin")
        versions = ready.get("runtime_versions")
        if (
            not isinstance(origins, dict)
            or set(origins)
            != {
                "official_model",
                "slowfast_parser",
                "slowfast_loaded_modules",
            }
            or not isinstance(pytorchvideo_origin, dict)
            or pytorchvideo_origin.get("repository_commit") != PYTORCHVIDEO_COMMIT
            or pytorchvideo_origin.get("repository_tree") != PYTORCHVIDEO_TREE
            or not isinstance(versions, dict)
            or versions.get("container_image_id") != self._container_image_id
            or versions.get("encoder_just_encode_unused_parameters_json")
            != ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_JSON
            or versions.get("encoder_just_encode_unused_parameters_sha256")
            != ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_SHA256
        ):
            self.close(force=True)
            raise ESCountsDependencyError("worker module/runtime provenance is incomplete")
        self.runtime_versions = {
            **{str(key): str(value) for key, value in versions.items()},
            "worker_code_sha256": self._worker_digest.sha256,
            "worker_command_sha256": self._worker_command_sha256,
            "module_origins_sha256": sha256_json(origins),
            "pytorchvideo_origin_sha256": sha256_json(pytorchvideo_origin),
        }
        self._records_by_path = {
            _resolved_video_path(inputs, record.video_path): record
            for record in inputs.manifest.records
        }

    def _exchange(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self._process.stdin is None or self._process.stdout is None:
            raise ESCountsDependencyError("worker pipes are unavailable")
        request = _jsonl_message(payload)
        try:
            self._process.stdin.write(
                json.dumps(
                    request,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            self._process.stdin.flush()
            line = self._process.stdout.readline()
        except (BrokenPipeError, OSError) as exc:
            raise ESCountsDependencyError("ESCounts worker pipe failed") from exc
        if not line:
            return_code = self._process.poll()
            raise ESCountsDependencyError(
                f"ESCounts worker exited without a response (returncode={return_code})"
            )
        _reject_forbidden_keys(line, document_name="ESCounts worker response")
        try:
            response = json.loads(
                line,
                object_pairs_hook=lambda pairs: _reject_duplicate_pairs(
                    pairs,
                    field="worker response",
                ),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite worker JSON constant: {value}")
                ),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ESCountsDependencyError("worker returned invalid JSONL") from exc
        if not isinstance(response, dict):
            raise ESCountsDependencyError("worker response root is not an object")
        supplied_sha = response.pop("message_sha256", None)
        _canonical_sha256(supplied_sha, field="worker message_sha256")
        if supplied_sha != sha256_json(response):
            raise ESCountsDependencyError("worker response SHA-256 mismatch")
        return response

    def predict(
        self,
        video_path: Path,
        config: ESCountsOfficialConfig,
    ) -> BackendPrediction:
        if config != FROZEN_ESCOUNTS_OFFICIAL_CONFIG:
            raise ValueError("JSONL backend only accepts the frozen config")
        try:
            record = self._records_by_path[video_path]
        except KeyError as exc:
            raise ValueError("worker request path is absent from frozen sidecar") from exc
        expected_sha256 = record.video_sha256
        if expected_sha256 is None:
            raise ValueError("worker request requires a committed video SHA-256")
        self._sequence += 1
        request_id = f"{self._sequence:04d}:{record.video_id}:{expected_sha256[:16]}"
        response = self._exchange(
            {
                "schema_version": _WORKER_PROTOCOL_VERSION,
                "type": "predict",
                "request_id": request_id,
                "video_id": record.video_id,
                "video_locator": record.video_path,
                "video_sha256": expected_sha256,
            }
        )
        if (
            response.get("schema_version") != _WORKER_PROTOCOL_VERSION
            or response.get("type") != "result"
            or response.get("request_id") != request_id
            or response.get("video_id") != record.video_id
            or response.get("video_sha256") != expected_sha256
        ):
            raise ESCountsDependencyError("worker response request/video binding mismatch")
        status = response.get("status")
        expected_fields = {
            "schema_version",
            "type",
            "request_id",
            "video_id",
            "video_sha256",
            "status",
            "raw_count",
            "reported_frame_count",
            "decoded_frames",
            "tail_shortfall",
            "error_type",
            "error_message",
        }
        if set(response) != expected_fields:
            raise ESCountsDependencyError("worker result response fields changed")
        frame_audit = (
            response["reported_frame_count"],
            response["decoded_frames"],
            response["tail_shortfall"],
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in frame_audit):
            raise ESCountsDependencyError("worker frame audit fields are not integers")
        reported_frame_count, decoded_frames, tail_shortfall = frame_audit
        if (
            reported_frame_count < 0
            or decoded_frames < 0
            or tail_shortfall < 0
            or (
                reported_frame_count == 0
                and (decoded_frames != 0 or tail_shortfall != 0)
            )
            or (
                reported_frame_count > 0
                and (
                    decoded_frames < 1
                    or reported_frame_count - decoded_frames != tail_shortfall
                )
            )
        ):
            raise ESCountsDependencyError("worker frame audit fields are inconsistent")
        if status == PredictionStatus.OK.value:
            if response["error_type"] is not None or response["error_message"] is not None:
                raise ESCountsDependencyError("worker success response contains an error")
            return BackendPrediction(
                raw_count=float(response["raw_count"]),
                reported_frame_count=reported_frame_count,
                decoded_frames=decoded_frames,
                tail_shortfall=tail_shortfall,
            )
        error_message = response.get("error_message")
        if not isinstance(error_message, str) or not error_message:
            raise ESCountsDependencyError("worker failure response lacks an error")
        if status == PredictionStatus.RESOURCE_EXHAUSTED.value:
            raise ESCountsResourceExhaustedError(
                error_message,
                reported_frame_count=reported_frame_count,
                decoded_frames=decoded_frames,
                tail_shortfall=tail_shortfall,
            )
        if status == PredictionStatus.DECODE_FAILED.value:
            raise ESCountsDecodeError(
                error_message,
                reported_frame_count=reported_frame_count,
                decoded_frames=decoded_frames,
                tail_shortfall=tail_shortfall,
            )
        if status == PredictionStatus.INFERENCE_FAILED.value:
            raise ESCountsOfficialError(error_message)
        raise ESCountsDependencyError(f"unknown worker result status: {status!r}")

    def close(self, *, force: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if not force and self._process.poll() is None:
                nonce = secrets.token_hex(16)
                response = self._exchange(
                    {
                        "schema_version": _WORKER_PROTOCOL_VERSION,
                        "type": "shutdown",
                        "nonce": nonce,
                    }
                )
                if (
                    response.get("type") != "shutdown_ack"
                    or response.get("nonce") != nonce
                    or response.get("worker_sha256") != self._worker_digest.sha256
                ):
                    raise ESCountsDependencyError("worker shutdown binding mismatch")
            if self._process.stdin is not None:
                self._process.stdin.close()
            return_code = self._process.wait(timeout=30)
            if not force and return_code != 0:
                raise ESCountsDependencyError(f"worker exited with return code {return_code}")
        except Exception:
            if self._process.poll() is None:
                self._process.terminate()
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=10)
            if not force:
                raise


def _reject_forbidden_keys(
    encoded: str,
    *,
    document_name: str = "ESCounts prediction artifact",
) -> None:
    """Reject label-bearing object keys before their values are deserialized."""

    index = 0
    while index < len(encoded):
        if encoded[index] != '"':
            index += 1
            continue
        start = index
        index += 1
        escaped = False
        while index < len(encoded):
            character = encoded[index]
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                index += 1
                break
            index += 1
        else:
            return
        cursor = index
        while cursor < len(encoded) and encoded[cursor] in " \t\r\n":
            cursor += 1
        if cursor >= len(encoded) or encoded[cursor] != ":":
            continue
        try:
            key = json.loads(encoded[start:index])
        except json.JSONDecodeError:
            return
        if isinstance(key, str) and key.lower() in _FORBIDDEN_LABEL_KEYS:
            raise ValueError(f"{document_name} contains forbidden label field {key!r}")


def _load_json_object(path: Path, digest: _StableFileDigest) -> dict[str, Any]:
    raw = path.read_bytes()
    if len(raw) != digest.byte_count or hashlib.sha256(raw).hexdigest() != digest.sha256:
        raise RuntimeError("artifact changed between hashing and parsing")
    try:
        encoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("artifact must be UTF-8 JSON") from exc
    _reject_forbidden_keys(encoded)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError(f"duplicate JSON field: {key!r}")
            output[key] = value
        return output

    payload = json.loads(
        encoded,
        object_pairs_hook=reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("artifact root must be an object")
    return payload


def _strict_object(
    value: Any,
    *,
    expected: set[str],
    field: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    if set(value) != expected:
        raise ValueError(
            f"{field} fields mismatch; "
            f"missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )
    return value


@dataclass(frozen=True, slots=True)
class ValidatedPredictionArtifact:
    path: Path
    digest: _StableFileDigest
    payload: Mapping[str, Any]
    predictions: tuple[Mapping[str, Any], ...]
    failures: tuple[Mapping[str, Any], ...]


_ROW_FIELDS = {
    "index",
    "video_id",
    "video_locator",
    "expected_video_sha256",
    "observed_video_sha256",
    "raw_count",
    "rounded_count",
    "finite",
    "reported_frame_count",
    "decoded_frames",
    "tail_shortfall",
    "elapsed_seconds",
    "resource_tier",
    "memory_limit_bytes",
    "status",
    "failure_reason",
}


def _validate_result_row(
    raw_row: Any,
    *,
    expected_resource_tier: str,
    successful: bool,
    field: str,
) -> dict[str, Any]:
    row = _strict_object(raw_row, expected=_ROW_FIELDS, field=field)
    if isinstance(row["index"], bool) or not isinstance(row["index"], int):
        raise ValueError(f"{field}.index must be an integer")
    if not isinstance(row["video_id"], str) or not row["video_id"]:
        raise ValueError(f"{field}.video_id is invalid")
    _validate_portable_locator(row["video_locator"], field=f"{field}.video_locator")
    expected_sha = _canonical_sha256(
        row["expected_video_sha256"],
        field=f"{field}.expected_video_sha256",
    )
    observed_sha = row["observed_video_sha256"]
    if observed_sha is not None:
        _canonical_sha256(observed_sha, field=f"{field}.observed_video_sha256")
    if row["resource_tier"] != expected_resource_tier:
        raise ValueError(f"{field}.resource_tier mismatch")
    if row["memory_limit_bytes"] != _RESOURCE_LIMITS[expected_resource_tier]:
        raise ValueError(f"{field}.memory_limit_bytes mismatch")
    if (
        isinstance(row["elapsed_seconds"], bool)
        or not isinstance(row["elapsed_seconds"], int | float)
        or not math.isfinite(float(row["elapsed_seconds"]))
        or float(row["elapsed_seconds"]) < 0.0
    ):
        raise ValueError(f"{field}.elapsed_seconds is invalid")
    if (
        isinstance(row["reported_frame_count"], bool)
        or not isinstance(row["reported_frame_count"], int)
        or row["reported_frame_count"] < 0
        or isinstance(row["tail_shortfall"], bool)
        or not isinstance(row["tail_shortfall"], int)
        or row["tail_shortfall"] < 0
        or (
            row["reported_frame_count"] == 0
            and (row["decoded_frames"] != 0 or row["tail_shortfall"] != 0)
        )
        or (
            row["reported_frame_count"] > 0
            and (
                row["decoded_frames"] < 1
                or row["reported_frame_count"] - row["decoded_frames"]
                != row["tail_shortfall"]
            )
        )
    ):
        raise ValueError(f"{field} frame audit is inconsistent")
    if (
        isinstance(row["decoded_frames"], bool)
        or not isinstance(row["decoded_frames"], int)
        or row["decoded_frames"] < 0
    ):
        raise ValueError(f"{field}.decoded_frames is invalid")
    if successful:
        if row["status"] != PredictionStatus.OK.value or row["finite"] is not True:
            raise ValueError(f"{field} is not a finite successful prediction")
        raw_count = row["raw_count"]
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int | float)
            or not math.isfinite(float(raw_count))
            or float(raw_count) < 0.0
            or row["rounded_count"] != round_count(float(raw_count))
            or row["failure_reason"] is not None
            or observed_sha != expected_sha
            or row["decoded_frames"] < 1
            or row["tail_shortfall"] not in {0, 1}
        ):
            raise ValueError(f"{field} successful fields are inconsistent")
    elif (
        row["status"] == PredictionStatus.OK.value
        or row["finite"] is not False
        or row["raw_count"] is not None
        or row["rounded_count"] is not None
        or not isinstance(row["failure_reason"], str)
        or not row["failure_reason"]
    ):
        raise ValueError(f"{field} failure fields are inconsistent")
    return row


def _validate_prediction_artifact_payload(
    payload: dict[str, Any],
    digest: _StableFileDigest,
) -> ValidatedPredictionArtifact:
    root = _strict_object(
        payload,
        expected={
            "schema_version",
            "artifact_type",
            "method_id",
            "classification",
            "eligible_for_original_pams_escounts_cell",
            "target_access",
            "status",
            "source",
            "input",
            "selection",
            "assets",
            "config",
            "runner_source_git_sha",
            "runner_code_sha256",
            "worker_code_sha256",
            "runtime_versions",
            "resource",
            "retry_binding",
            "record_total",
            "success_total",
            "failure_total",
            "predictions",
            "failures",
        },
        field="prediction artifact",
    )
    if (
        root["schema_version"] != 1
        or root["artifact_type"] != PREDICTION_ARTIFACT_TYPE
        or root["method_id"] != METHOD_ID
        or root["classification"] != CLASSIFICATION
        or root["eligible_for_original_pams_escounts_cell"] is not False
        or root["target_access"] is not False
    ):
        raise ValueError("prediction artifact header is not frozen")
    expected_source = {
        "repository": OFFICIAL_SOURCE_REPOSITORY,
        "commit": OFFICIAL_SOURCE_COMMIT,
        "tree": OFFICIAL_SOURCE_TREE,
        "license": OFFICIAL_SOURCE_LICENSE,
        "files": [
            {
                "relative_path": spec.relative_path,
                "bytes": spec.byte_count,
                "sha256": spec.sha256,
            }
            for spec in OFFICIAL_SOURCE_FILES
        ],
        "pytorchvideo_repository": PYTORCHVIDEO_REPOSITORY,
        "pytorchvideo_commit": PYTORCHVIDEO_COMMIT,
        "pytorchvideo_tree": PYTORCHVIDEO_TREE,
    }
    if root["source"] != expected_source:
        raise ValueError("prediction artifact source binding changed")
    input_binding = _strict_object(
        root["input"],
        expected={
            "protocol",
            "split",
            "sample_count",
            "sidecar_sha256",
            "commitment_sha256",
            "sidecar_fingerprint",
            "identity_sha256",
        },
        field="prediction input",
    )
    if (
        input_binding["protocol"] != "ucfrep_526"
        or input_binding["split"] not in {"train", "dev"}
        or isinstance(input_binding["sample_count"], bool)
        or not isinstance(input_binding["sample_count"], int)
        or input_binding["sample_count"] < 1
    ):
        raise ValueError("prediction input protocol binding is invalid")
    for name in (
        "sidecar_sha256",
        "commitment_sha256",
        "sidecar_fingerprint",
        "identity_sha256",
    ):
        _canonical_sha256(input_binding[name], field=f"input.{name}")
    selection = _strict_object(
        root["selection"],
        expected={
            "selected_count",
            "selected_video_ids",
            "selected_video_ids_sha256",
        },
        field="prediction selection",
    )
    selected_ids = selection["selected_video_ids"]
    if (
        isinstance(selection["selected_count"], bool)
        or not isinstance(selection["selected_count"], int)
        or not isinstance(selected_ids, list)
        or selection["selected_count"] != len(selected_ids)
        or len(set(selected_ids)) != len(selected_ids)
        or any(not isinstance(video_id, str) or not video_id for video_id in selected_ids)
        or selection["selected_video_ids_sha256"] != sha256_json(selected_ids)
    ):
        raise ValueError("prediction selection is inconsistent")
    expected_assets = {
        "aggregate_sha256": sha256_json(
            {"files": [OFFICIAL_ENCODER.public_dict(), OFFICIAL_DECODER.public_dict()]}
        ),
        "files": [OFFICIAL_ENCODER.public_dict(), OFFICIAL_DECODER.public_dict()],
    }
    if root["assets"] != expected_assets:
        raise ValueError("prediction checkpoint binding changed")
    config = _strict_object(
        root["config"],
        expected={"sha256", "values"},
        field="prediction config",
    )
    if (
        config["values"] != FROZEN_ESCOUNTS_OFFICIAL_CONFIG.to_dict()
        or config["sha256"] != FROZEN_ESCOUNTS_OFFICIAL_CONFIG.fingerprint
    ):
        raise ValueError("prediction config is not frozen")
    if not isinstance(root["runtime_versions"], dict) or not root["runtime_versions"]:
        raise ValueError("prediction runtime_versions are missing")
    runtime_versions = root["runtime_versions"]
    required_runtime_fields = {
        "python",
        "torch",
        "torchvision",
        "cuda",
        "numpy",
        "opencv",
        "av",
        "container_image_id",
        "pytorchvideo_commit",
        "worker_code_sha256",
        "worker_command_sha256",
        "module_origins_sha256",
        "pytorchvideo_origin_sha256",
        "encoder_just_encode_unused_parameters_json",
        "encoder_just_encode_unused_parameters_sha256",
    }
    if not required_runtime_fields.issubset(runtime_versions):
        raise ValueError("prediction runtime provenance is incomplete")
    _validate_container_image_id(runtime_versions["container_image_id"])
    if runtime_versions["pytorchvideo_commit"] != PYTORCHVIDEO_COMMIT:
        raise ValueError("prediction PyTorchVideo provenance changed")
    if (
        runtime_versions["encoder_just_encode_unused_parameters_json"]
        != ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_JSON
        or runtime_versions["encoder_just_encode_unused_parameters_sha256"]
        != ENCODER_JUST_ENCODE_UNUSED_PARAMETERS_SHA256
    ):
        raise ValueError("prediction encoder unmatched-tensor audit changed")
    for name in (
        "worker_code_sha256",
        "worker_command_sha256",
        "module_origins_sha256",
        "pytorchvideo_origin_sha256",
        "encoder_just_encode_unused_parameters_sha256",
    ):
        _canonical_sha256(runtime_versions[name], field=f"runtime_versions.{name}")
    resource = _strict_object(
        root["resource"],
        expected={"tier", "memory_limit_bytes"},
        field="prediction resource",
    )
    tier = resource["tier"]
    if not isinstance(tier, str) or _RESOURCE_LIMITS.get(tier) != resource["memory_limit_bytes"]:
        raise ValueError("prediction resource tier is invalid")
    if tier == PRIMARY_RESOURCE_TIER and root["retry_binding"] is not None:
        raise ValueError("primary prediction artifact contains retry_binding")
    if tier == RETRY_RESOURCE_TIER and not isinstance(root["retry_binding"], dict):
        raise ValueError("retry prediction artifact lacks retry_binding")
    _canonical_git_sha(
        root["runner_source_git_sha"],
        field="runner_source_git_sha",
    )
    _canonical_sha256(root["runner_code_sha256"], field="runner_code_sha256")
    _canonical_sha256(root["worker_code_sha256"], field="worker_code_sha256")
    if root["worker_code_sha256"] != runtime_versions["worker_code_sha256"]:
        raise ValueError("prediction worker provenance fields differ")
    raw_predictions = root["predictions"]
    raw_failures = root["failures"]
    if not isinstance(raw_predictions, list) or not isinstance(raw_failures, list):
        raise ValueError("predictions/failures must be arrays")
    predictions = tuple(
        _validate_result_row(
            row,
            expected_resource_tier=tier,
            successful=True,
            field=f"predictions[{index}]",
        )
        for index, row in enumerate(raw_predictions)
    )
    failures = tuple(
        _validate_result_row(
            row,
            expected_resource_tier=tier,
            successful=False,
            field=f"failures[{index}]",
        )
        for index, row in enumerate(raw_failures)
    )
    if (
        root["success_total"] != len(predictions)
        or root["failure_total"] != len(failures)
        or root["record_total"] != len(predictions) + len(failures)
        or root["record_total"] != selection["selected_count"]
        or root["status"] != ("complete" if not failures else "partial")
    ):
        raise ValueError("prediction totals/status are inconsistent")
    combined = sorted((*predictions, *failures), key=lambda row: row["index"])
    if [row["video_id"] for row in combined] != selected_ids:
        raise ValueError("prediction IDs/order differ from frozen selection")
    if len({row["index"] for row in combined}) != len(combined):
        raise ValueError("prediction indices must be unique")
    return ValidatedPredictionArtifact(
        path=Path(),
        digest=digest,
        payload=root,
        predictions=predictions,
        failures=failures,
    )


def load_prediction_artifact(path: str | Path) -> ValidatedPredictionArtifact:
    """Hash and strictly validate one label-free primary or retry artifact."""

    resolved = Path(path).expanduser().resolve(strict=True)
    digest = _stable_file_digest(resolved)
    payload = _load_json_object(resolved, digest)
    validated = _validate_prediction_artifact_payload(payload, digest)
    return ValidatedPredictionArtifact(
        path=resolved,
        digest=digest,
        payload=validated.payload,
        predictions=validated.predictions,
        failures=validated.failures,
    )


def _assert_payload_runner_provenance(
    payload: Mapping[str, Any],
    repository: VerifiedRunnerRepository,
    *,
    role: str,
) -> None:
    expected = repository.provenance_dict()
    observed = {
        "runner_source_git_sha": payload.get("runner_source_git_sha"),
        "runner_code_sha256": payload.get("runner_code_sha256"),
        "worker_code_sha256": payload.get("worker_code_sha256"),
    }
    if observed != expected:
        raise ESCountsSourceError(
            f"{role} does not originate from the current clean runner checkout"
        )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> _StableFileDigest:
    durable_mkdir(path.parent)
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    created_identity: _FileIdentity | None = None
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite artifact: {path}") from exc
    try:
        created_identity = _FileIdentity.from_stat(os.fstat(descriptor))
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(path.parent)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if created_identity is not None:
            _remove_owned_artifact(path, created_identity)
        raise
    return _stable_file_digest(path)


def _remove_owned_artifact(path: Path, identity: _FileIdentity) -> None:
    """Remove only the inode created by this process after a failed transaction."""

    try:
        observed = path.lstat()
    except FileNotFoundError:
        return
    if _FileIdentity.from_stat(observed) != identity:
        raise RuntimeError(f"refusing to remove a replaced transaction file: {path}")
    path.unlink()
    fsync_directory(path.parent)


def write_escounts_result_exclusive(
    output_path: str | Path,
    result: ESCountsOfficialRunResult,
) -> str:
    """Write one immutable prediction artifact and return its SHA-256."""

    output = Path(output_path).expanduser().resolve(strict=False)
    return _write_json_exclusive(output, result.to_dict()).sha256


def create_retry_request(
    *,
    primary_predictions_path: str | Path,
    output_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Freeze the primary OOM subset for the single authorized 12 GiB retry."""

    repository = verify_runner_repository(repository_root)
    primary = load_prediction_artifact(primary_predictions_path)
    _assert_payload_runner_provenance(
        primary.payload,
        repository,
        role="primary prediction artifact",
    )
    resource = primary.payload["resource"]
    if (
        resource["tier"] != PRIMARY_RESOURCE_TIER
        or resource["memory_limit_bytes"] != PRIMARY_MEMORY_LIMIT_BYTES
    ):
        raise ValueError("retry request requires an 8 GiB primary artifact")
    retryable = tuple(
        failure
        for failure in primary.failures
        if failure["status"] == PredictionStatus.RESOURCE_EXHAUSTED.value
    )
    if not retryable:
        raise ValueError("primary artifact contains no resource-exhausted failures")
    if len(retryable) != len(primary.failures):
        raise ValueError("finite retry cannot hide non-resource primary failures")
    records = [
        {
            "index": row["index"],
            "video_id": row["video_id"],
            "video_locator": row["video_locator"],
            "video_sha256": row["expected_video_sha256"],
        }
        for row in sorted(retryable, key=lambda item: item["index"])
    ]
    payload = {
        "schema_version": 1,
        "artifact_type": RETRY_REQUEST_ARTIFACT_TYPE,
        "method_id": METHOD_ID,
        "source_primary_sha256": primary.digest.sha256,
        "input_binding": dict(primary.payload["input"]),
        "source_binding_sha256": sha256_json(primary.payload["source"]),
        "assets_binding_sha256": sha256_json(primary.payload["assets"]),
        "config_sha256": primary.payload["config"]["sha256"],
        **repository.provenance_dict(),
        "primary_success_rows_sha256": sha256_json(list(primary.predictions)),
        "resource": {
            "tier": RETRY_RESOURCE_TIER,
            "memory_limit_bytes": RETRY_MEMORY_LIMIT_BYTES,
        },
        "record_total": len(records),
        "records": records,
        "records_sha256": sha256_json(records),
    }
    output = Path(output_path).expanduser().resolve(strict=False)
    repository.assert_unchanged()
    _assert_file_unchanged(
        primary.path,
        primary.digest,
        role="primary prediction artifact",
    )
    digest = _write_json_exclusive(output, payload)
    try:
        repository.assert_unchanged()
        _assert_file_unchanged(
            primary.path,
            primary.digest,
            role="primary prediction artifact",
        )
    except Exception:
        _remove_owned_artifact(output, digest.identity)
        raise
    return {
        "retry_request_path": str(output),
        "retry_request_sha256": digest.sha256,
        "source_primary_sha256": primary.digest.sha256,
        **repository.provenance_dict(),
        "record_total": len(records),
        "video_ids": [record["video_id"] for record in records],
    }


@dataclass(frozen=True, slots=True)
class _ValidatedRetryRequest:
    path: Path
    digest: _StableFileDigest
    payload: Mapping[str, Any]
    records: tuple[Mapping[str, Any], ...]


def _load_retry_request(
    request_path: str | Path,
    primary_predictions_path: str | Path,
) -> _ValidatedRetryRequest:
    primary = load_prediction_artifact(primary_predictions_path)
    request = Path(request_path).expanduser().resolve(strict=True)
    digest = _stable_file_digest(request)
    payload = _load_json_object(request, digest)
    root = _strict_object(
        payload,
        expected={
            "schema_version",
            "artifact_type",
            "method_id",
            "source_primary_sha256",
            "input_binding",
            "source_binding_sha256",
            "assets_binding_sha256",
            "config_sha256",
            "runner_source_git_sha",
            "runner_code_sha256",
            "worker_code_sha256",
            "primary_success_rows_sha256",
            "resource",
            "record_total",
            "records",
            "records_sha256",
        },
        field="retry request",
    )
    if (
        root["schema_version"] != 1
        or root["artifact_type"] != RETRY_REQUEST_ARTIFACT_TYPE
        or root["method_id"] != METHOD_ID
        or root["source_primary_sha256"] != primary.digest.sha256
        or root["input_binding"] != primary.payload["input"]
        or root["source_binding_sha256"] != sha256_json(primary.payload["source"])
        or root["assets_binding_sha256"] != sha256_json(primary.payload["assets"])
        or root["config_sha256"] != primary.payload["config"]["sha256"]
        or root["runner_source_git_sha"] != primary.payload["runner_source_git_sha"]
        or root["runner_code_sha256"] != primary.payload["runner_code_sha256"]
        or root["worker_code_sha256"] != primary.payload["worker_code_sha256"]
        or root["primary_success_rows_sha256"] != sha256_json(list(primary.predictions))
        or root["resource"]
        != {
            "tier": RETRY_RESOURCE_TIER,
            "memory_limit_bytes": RETRY_MEMORY_LIMIT_BYTES,
        }
    ):
        raise ValueError("retry request does not bind the primary artifact")
    _canonical_git_sha(
        root["runner_source_git_sha"],
        field="retry request runner_source_git_sha",
    )
    _canonical_sha256(
        root["runner_code_sha256"],
        field="retry request runner_code_sha256",
    )
    _canonical_sha256(
        root["worker_code_sha256"],
        field="retry request worker_code_sha256",
    )
    records = root["records"]
    if (
        not isinstance(records, list)
        or root["record_total"] != len(records)
        or root["records_sha256"] != sha256_json(records)
    ):
        raise ValueError("retry request records are inconsistent")
    expected = [
        {
            "index": row["index"],
            "video_id": row["video_id"],
            "video_locator": row["video_locator"],
            "video_sha256": row["expected_video_sha256"],
        }
        for row in sorted(primary.failures, key=lambda item: item["index"])
        if row["status"] == PredictionStatus.RESOURCE_EXHAUSTED.value
    ]
    if records != expected or len(expected) != len(primary.failures):
        raise ValueError("retry request changed the audited primary failure set")
    return _ValidatedRetryRequest(
        path=request,
        digest=digest,
        payload=root,
        records=tuple(records),
    )


def run_escounts_official(
    sidecar_path: str | Path,
    commitment_path: str | Path,
    video_root: str | Path,
    encoder_path: str | Path,
    decoder_path: str | Path,
    official_source_root: str | Path,
    *,
    repository_root: str | Path,
    device: str = "cuda:0",
    resource_tier: str = PRIMARY_RESOURCE_TIER,
    video_ids: Sequence[str] | None = None,
    retry_request_path: str | Path | None = None,
    primary_predictions_path: str | Path | None = None,
    output_path: str | Path | None = None,
    require_exact_membership: bool = True,
    backend: _OfficialBackend | None = None,
    worker_command: str | Sequence[str] | None = None,
    pytorchvideo_source_root: str | Path | None = None,
    expected_container_image_id: str | None = None,
) -> ESCountsOfficialRunResult:
    """Run the official zero-shot path behind a strict label-free boundary."""

    inputs = _load_label_free_inputs(
        sidecar_path,
        commitment_path,
        video_root,
        require_exact_membership=require_exact_membership,
    )
    repository = verify_runner_repository(repository_root)
    retry_binding: Mapping[str, Any] | None = None
    if resource_tier == RETRY_RESOURCE_TIER:
        if retry_request_path is None or primary_predictions_path is None:
            raise ValueError("12 GiB retry requires request and primary artifact")
        request = _load_retry_request(
            retry_request_path,
            primary_predictions_path,
        )
        _assert_payload_runner_provenance(
            request.payload,
            repository,
            role="retry request",
        )
        requested_ids = tuple(record["video_id"] for record in request.records)
        if video_ids is not None and tuple(video_ids) != requested_ids:
            raise ValueError("retry video_ids must exactly match frozen retry request")
        video_ids = requested_ids
        retry_binding = {
            "retry_request_sha256": request.digest.sha256,
            "source_primary_sha256": request.payload["source_primary_sha256"],
            "primary_success_rows_sha256": request.payload["primary_success_rows_sha256"],
            "records_sha256": request.payload["records_sha256"],
        }
    elif (
        resource_tier != PRIMARY_RESOURCE_TIER
        or retry_request_path is not None
        or primary_predictions_path is not None
    ):
        raise ValueError("invalid primary/retry resource arguments")

    source = verify_official_source(official_source_root)
    assets = verify_official_assets(encoder_path, decoder_path)
    execution_backend = backend
    owned_backend: JSONLWorkerBackend | None = None
    if execution_backend is None:
        if (
            worker_command is None
            or pytorchvideo_source_root is None
            or expected_container_image_id is None
        ):
            raise ValueError(
                "official execution requires worker command, PyTorchVideo source, "
                "and expected container image ID"
            )
        owned_backend = JSONLWorkerBackend(
            source,
            assets,
            inputs,
            repository,
            worker_command=worker_command,
            pytorchvideo_source_root=pytorchvideo_source_root,
            expected_container_image_id=expected_container_image_id,
            device=device,
            memory_limit_bytes=_RESOURCE_LIMITS[resource_tier],
        )
        execution_backend = cast(_OfficialBackend, owned_backend)
    try:
        result = _run_verified_backend(
            inputs,
            source,
            assets,
            execution_backend,
            repository,
            video_ids=video_ids,
            resource_tier=resource_tier,
            retry_binding=retry_binding,
        )
    finally:
        if owned_backend is not None:
            owned_backend.close()
    inputs.assert_unchanged()
    source.assert_unchanged()
    assets.assert_unchanged()
    repository.assert_unchanged()
    if owned_backend is not None and pytorchvideo_source_root is not None:
        pytorchvideo_root = Path(pytorchvideo_source_root).expanduser().resolve(strict=True)
        commit, tree, status = _git_source_state(pytorchvideo_root)
        if commit != PYTORCHVIDEO_COMMIT or tree != PYTORCHVIDEO_TREE or status:
            raise ESCountsSourceError("PyTorchVideo checkout changed during prediction")
    if output_path is not None:
        output = Path(output_path).expanduser().resolve(strict=False)
        output_digest = _write_json_exclusive(output, result.to_dict())
        try:
            inputs.assert_unchanged()
            source.assert_unchanged()
            assets.assert_unchanged()
            repository.assert_unchanged()
        except Exception:
            _remove_owned_artifact(output, output_digest.identity)
            raise
    return result


def merge_retry_artifacts(
    *,
    primary_predictions_path: str | Path,
    retry_request_path: str | Path,
    retry_predictions_path: str | Path,
    output_path: str | Path,
    receipt_path: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """Merge one complete finite retry while proving primary rows unchanged."""

    output = Path(output_path).expanduser().resolve(strict=False)
    receipt = Path(receipt_path).expanduser().resolve(strict=False)
    collisions = [str(path) for path in (output, receipt) if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite merge artifacts: {collisions}")
    repository = verify_runner_repository(repository_root)
    primary = load_prediction_artifact(primary_predictions_path)
    request = _load_retry_request(retry_request_path, primary_predictions_path)
    retry = load_prediction_artifact(retry_predictions_path)
    for payload, role in (
        (primary.payload, "primary prediction artifact"),
        (request.payload, "retry request"),
        (retry.payload, "retry prediction artifact"),
    ):
        _assert_payload_runner_provenance(payload, repository, role=role)
    if primary.payload["resource"]["tier"] != PRIMARY_RESOURCE_TIER:
        raise ESCountsMergeError("merge primary is not the 8 GiB tier")
    if retry.payload["resource"]["tier"] != RETRY_RESOURCE_TIER:
        raise ESCountsMergeError("merge retry is not the 12 GiB tier")
    expected_retry_binding = {
        "retry_request_sha256": request.digest.sha256,
        "source_primary_sha256": primary.digest.sha256,
        "primary_success_rows_sha256": request.payload["primary_success_rows_sha256"],
        "records_sha256": request.payload["records_sha256"],
    }
    if retry.payload["retry_binding"] != expected_retry_binding:
        raise ESCountsMergeError("retry artifact is not bound to request/primary")
    for field in ("source", "input", "assets", "config"):
        if retry.payload[field] != primary.payload[field]:
            raise ESCountsMergeError(f"retry changed frozen {field} binding")
    if (
        retry.payload["runner_source_git_sha"] != primary.payload["runner_source_git_sha"]
        or retry.payload["runner_code_sha256"] != primary.payload["runner_code_sha256"]
        or retry.payload["worker_code_sha256"] != primary.payload["worker_code_sha256"]
        or retry.payload["runtime_versions"].get("worker_code_sha256")
        != primary.payload["runtime_versions"].get("worker_code_sha256")
        or retry.payload["runtime_versions"].get("worker_command_sha256")
        != primary.payload["runtime_versions"].get("worker_command_sha256")
        or retry.payload["runtime_versions"].get("container_image_id")
        != primary.payload["runtime_versions"].get("container_image_id")
    ):
        raise ESCountsMergeError("retry changed runner/worker/container provenance")
    expected_ids = [record["video_id"] for record in request.records]
    if (
        retry.failures
        or len(retry.predictions) != len(expected_ids)
        or [row["video_id"] for row in retry.predictions] != expected_ids
    ):
        raise ESCountsMergeError("finite retry did not successfully cover request")
    primary_failure_ids = [
        row["video_id"] for row in sorted(primary.failures, key=lambda item: item["index"])
    ]
    if primary_failure_ids != expected_ids:
        raise ESCountsMergeError("retry request is not the full primary failure set")

    primary_rows = list(primary.predictions)
    primary_rows_sha256 = sha256_json(primary_rows)
    retry_by_index = {row["index"]: row for row in retry.predictions}
    merged = sorted(
        [*primary_rows, *retry_by_index.values()],
        key=lambda item: item["index"],
    )
    selected_ids = primary.payload["selection"]["selected_video_ids"]
    if (
        len(merged) != primary.payload["record_total"]
        or [row["video_id"] for row in merged] != selected_ids
        or len({row["index"] for row in merged}) != len(merged)
    ):
        raise ESCountsMergeError("merged rows do not exactly cover primary selection")
    merged_primary_projection = [
        row for row in merged if row["resource_tier"] == PRIMARY_RESOURCE_TIER
    ]
    merged_primary_sha256 = sha256_json(merged_primary_projection)
    if primary_rows_sha256 != merged_primary_sha256:
        raise ESCountsMergeError("one or more successful primary rows changed")

    payload = {
        "schema_version": 1,
        "artifact_type": MERGED_ARTIFACT_TYPE,
        "method_id": METHOD_ID,
        "classification": CLASSIFICATION,
        "eligible_for_original_pams_escounts_cell": False,
        "target_access": False,
        "status": "complete",
        "source": primary.payload["source"],
        "input": primary.payload["input"],
        "selection": primary.payload["selection"],
        "assets": primary.payload["assets"],
        "config": primary.payload["config"],
        **repository.provenance_dict(),
        "runtime_versions": {
            "primary": primary.payload["runtime_versions"],
            "retry": retry.payload["runtime_versions"],
        },
        "resource_tier_counts": {
            PRIMARY_RESOURCE_TIER: len(primary.predictions),
            RETRY_RESOURCE_TIER: len(retry.predictions),
        },
        "resource_limits_bytes": {
            PRIMARY_RESOURCE_TIER: PRIMARY_MEMORY_LIMIT_BYTES,
            RETRY_RESOURCE_TIER: RETRY_MEMORY_LIMIT_BYTES,
        },
        "source_primary_sha256": primary.digest.sha256,
        "source_retry_request_sha256": request.digest.sha256,
        "source_retry_sha256": retry.digest.sha256,
        "primary_success_rows_sha256": primary_rows_sha256,
        "merged_primary_rows_sha256": merged_primary_sha256,
        "primary_rows_canonical_value_identical": True,
        "record_total": len(merged),
        "success_total": len(merged),
        "failure_total": 0,
        "predictions": merged,
        "failures": [],
    }
    _assert_file_unchanged(primary.path, primary.digest, role="primary prediction artifact")
    _assert_file_unchanged(request.path, request.digest, role="retry request artifact")
    _assert_file_unchanged(retry.path, retry.digest, role="retry prediction artifact")
    repository.assert_unchanged()
    output_digest = _write_json_exclusive(output, payload)
    receipt_payload = {
        "schema_version": 1,
        "artifact_type": "escounts_official_merge_receipt",
        "method_id": METHOD_ID,
        **repository.provenance_dict(),
        "output_file": output.name,
        "output_sha256": output_digest.sha256,
        "output_bytes": output_digest.byte_count,
        "source_primary_sha256": primary.digest.sha256,
        "source_retry_request_sha256": request.digest.sha256,
        "source_retry_sha256": retry.digest.sha256,
        "primary_success_rows_sha256": primary_rows_sha256,
        "merged_primary_rows_sha256": merged_primary_sha256,
        "primary_rows_canonical_value_identical": True,
        "record_total": len(merged),
        "resource_tier_counts": payload["resource_tier_counts"],
    }
    receipt_digest: _StableFileDigest | None = None
    try:
        _assert_file_unchanged(
            primary.path,
            primary.digest,
            role="primary prediction artifact",
        )
        _assert_file_unchanged(
            request.path,
            request.digest,
            role="retry request artifact",
        )
        _assert_file_unchanged(
            retry.path,
            retry.digest,
            role="retry prediction artifact",
        )
        repository.assert_unchanged()
        receipt_digest = _write_json_exclusive(receipt, receipt_payload)
        _assert_file_unchanged(primary.path, primary.digest, role="primary prediction artifact")
        _assert_file_unchanged(request.path, request.digest, role="retry request artifact")
        _assert_file_unchanged(retry.path, retry.digest, role="retry prediction artifact")
        _assert_file_unchanged(output, output_digest, role="merged prediction artifact")
        repository.assert_unchanged()
    except Exception:
        if receipt_digest is not None:
            _remove_owned_artifact(receipt, receipt_digest.identity)
        _remove_owned_artifact(output, output_digest.identity)
        raise
    assert receipt_digest is not None
    return {
        "output_path": str(output),
        "output_sha256": output_digest.sha256,
        "receipt_path": str(receipt),
        "receipt_sha256": receipt_digest.sha256,
        "record_total": len(merged),
        "primary_rows_canonical_value_identical": True,
    }


@dataclass(frozen=True, slots=True)
class ValidatedMergedPredictions:
    path: Path
    digest: _StableFileDigest
    payload: Mapping[str, Any]
    video_ids: tuple[str, ...]
    raw_counts: tuple[float, ...]
    rounded_counts: tuple[int, ...]
    rows: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class ValidatedMergeChain:
    """All immutable artifacts required to establish prediction provenance."""

    primary: ValidatedPredictionArtifact
    request: _ValidatedRetryRequest
    retry: ValidatedPredictionArtifact
    merged: ValidatedMergedPredictions
    receipt_path: Path
    receipt_digest: _StableFileDigest
    receipt: Mapping[str, Any]


def load_complete_merged_predictions(
    path: str | Path,
) -> ValidatedMergedPredictions:
    """Strictly validate a complete label-free merge before scoring."""

    resolved = Path(path).expanduser().resolve(strict=True)
    digest = _stable_file_digest(resolved)
    payload = _load_json_object(resolved, digest)
    root = _strict_object(
        payload,
        expected={
            "schema_version",
            "artifact_type",
            "method_id",
            "classification",
            "eligible_for_original_pams_escounts_cell",
            "target_access",
            "status",
            "source",
            "input",
            "selection",
            "assets",
            "config",
            "runner_source_git_sha",
            "runner_code_sha256",
            "worker_code_sha256",
            "runtime_versions",
            "resource_tier_counts",
            "resource_limits_bytes",
            "source_primary_sha256",
            "source_retry_request_sha256",
            "source_retry_sha256",
            "primary_success_rows_sha256",
            "merged_primary_rows_sha256",
            "primary_rows_canonical_value_identical",
            "record_total",
            "success_total",
            "failure_total",
            "predictions",
            "failures",
        },
        field="merged prediction artifact",
    )
    if (
        root["schema_version"] != 1
        or root["artifact_type"] != MERGED_ARTIFACT_TYPE
        or root["method_id"] != METHOD_ID
        or root["classification"] != CLASSIFICATION
        or root["eligible_for_original_pams_escounts_cell"] is not False
        or root["target_access"] is not False
        or root["status"] != "complete"
        or root["primary_rows_canonical_value_identical"] is not True
        or root["primary_success_rows_sha256"] != root["merged_primary_rows_sha256"]
        or root["record_total"] != 84
        or root["success_total"] != 84
        or root["failure_total"] != 0
        or root["failures"] != []
    ):
        raise ValueError("merged prediction artifact is not a sealed 84-row result")
    for sha_field in (
        "source_primary_sha256",
        "source_retry_request_sha256",
        "source_retry_sha256",
        "primary_success_rows_sha256",
        "merged_primary_rows_sha256",
    ):
        _canonical_sha256(root[sha_field], field=sha_field)
    _canonical_git_sha(
        root["runner_source_git_sha"],
        field="runner_source_git_sha",
    )
    _canonical_sha256(root["runner_code_sha256"], field="runner_code_sha256")
    _canonical_sha256(root["worker_code_sha256"], field="worker_code_sha256")
    expected_resource_limits = {
        PRIMARY_RESOURCE_TIER: PRIMARY_MEMORY_LIMIT_BYTES,
        RETRY_RESOURCE_TIER: RETRY_MEMORY_LIMIT_BYTES,
    }
    if root["resource_limits_bytes"] != expected_resource_limits:
        raise ValueError("merged resource limits changed")
    counts = root["resource_tier_counts"]
    if (
        not isinstance(counts, dict)
        or set(counts) != set(expected_resource_limits)
        or sum(counts.values()) != 84
    ):
        raise ValueError("merged resource tier counts are invalid")
    expected_source = {
        "repository": OFFICIAL_SOURCE_REPOSITORY,
        "commit": OFFICIAL_SOURCE_COMMIT,
        "tree": OFFICIAL_SOURCE_TREE,
        "license": OFFICIAL_SOURCE_LICENSE,
        "files": [
            {
                "relative_path": spec.relative_path,
                "bytes": spec.byte_count,
                "sha256": spec.sha256,
            }
            for spec in OFFICIAL_SOURCE_FILES
        ],
        "pytorchvideo_repository": PYTORCHVIDEO_REPOSITORY,
        "pytorchvideo_commit": PYTORCHVIDEO_COMMIT,
        "pytorchvideo_tree": PYTORCHVIDEO_TREE,
    }
    if root["source"] != expected_source:
        raise ValueError("merged source binding changed")
    expected_assets = {
        "aggregate_sha256": sha256_json(
            {"files": [OFFICIAL_ENCODER.public_dict(), OFFICIAL_DECODER.public_dict()]}
        ),
        "files": [OFFICIAL_ENCODER.public_dict(), OFFICIAL_DECODER.public_dict()],
    }
    if root["assets"] != expected_assets:
        raise ValueError("merged asset binding changed")
    if root["config"] != {
        "sha256": FROZEN_ESCOUNTS_OFFICIAL_CONFIG.fingerprint,
        "values": FROZEN_ESCOUNTS_OFFICIAL_CONFIG.to_dict(),
    }:
        raise ValueError("merged config binding changed")
    input_binding = root["input"]
    if (
        not isinstance(input_binding, dict)
        or input_binding.get("protocol") != "ucfrep_526"
        or input_binding.get("split") != "dev"
        or input_binding.get("sample_count") != 84
    ):
        raise ValueError("merged artifact is not the full UCFRep dev input")
    for name in (
        "sidecar_sha256",
        "commitment_sha256",
        "sidecar_fingerprint",
        "identity_sha256",
    ):
        _canonical_sha256(input_binding.get(name), field=f"input.{name}")
    selection = root["selection"]
    if not isinstance(selection, dict):
        raise ValueError("merged selection is invalid")
    video_ids = selection.get("selected_video_ids")
    if (
        selection.get("selected_count") != 84
        or not isinstance(video_ids, list)
        or len(video_ids) != 84
        or len(set(video_ids)) != 84
        or selection.get("selected_video_ids_sha256") != sha256_json(video_ids)
    ):
        raise ValueError("merged selection is not 84 unique frozen IDs")
    rows_raw = root["predictions"]
    if not isinstance(rows_raw, list) or len(rows_raw) != 84:
        raise ValueError("merged predictions must contain 84 rows")
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(rows_raw):
        if not isinstance(raw_row, dict):
            raise ValueError(f"merged prediction {index} is not an object")
        tier = raw_row.get("resource_tier")
        if tier not in _RESOURCE_LIMITS:
            raise ValueError(f"merged prediction {index} resource tier is invalid")
        row = _validate_result_row(
            raw_row,
            expected_resource_tier=tier,
            successful=True,
            field=f"predictions[{index}]",
        )
        if row["index"] != index or row["video_id"] != video_ids[index]:
            raise ValueError("merged prediction index/order changed")
        rows.append(row)
    observed_counts = {
        tier: sum(row["resource_tier"] == tier for row in rows) for tier in _RESOURCE_LIMITS
    }
    if observed_counts != counts:
        raise ValueError("merged rows do not match resource tier counts")
    primary_rows = [row for row in rows if row["resource_tier"] == PRIMARY_RESOURCE_TIER]
    if sha256_json(primary_rows) != root["merged_primary_rows_sha256"]:
        raise ValueError("merged primary-row integrity hash mismatch")
    return ValidatedMergedPredictions(
        path=resolved,
        digest=digest,
        payload=root,
        video_ids=tuple(video_ids),
        raw_counts=tuple(float(row["raw_count"]) for row in rows),
        rounded_counts=tuple(int(row["rounded_count"]) for row in rows),
        rows=tuple(rows),
    )


def validate_complete_merge_chain(
    *,
    primary_predictions_path: str | Path,
    retry_request_path: str | Path,
    retry_predictions_path: str | Path,
    merged_predictions_path: str | Path,
    merge_receipt_path: str | Path,
    repository_root: str | Path,
) -> ValidatedMergeChain:
    """Validate actual primary/request/retry/merge/receipt bytes as one chain."""

    repository = verify_runner_repository(repository_root)
    primary = load_prediction_artifact(primary_predictions_path)
    request = _load_retry_request(retry_request_path, primary_predictions_path)
    retry = load_prediction_artifact(retry_predictions_path)
    merged = load_complete_merged_predictions(merged_predictions_path)
    receipt_path = Path(merge_receipt_path).expanduser().resolve(strict=True)
    receipt_digest = _stable_file_digest(receipt_path)
    receipt_raw = _load_json_object(receipt_path, receipt_digest)
    receipt = _strict_object(
        receipt_raw,
        expected={
            "schema_version",
            "artifact_type",
            "method_id",
            "runner_source_git_sha",
            "runner_code_sha256",
            "worker_code_sha256",
            "output_file",
            "output_sha256",
            "output_bytes",
            "source_primary_sha256",
            "source_retry_request_sha256",
            "source_retry_sha256",
            "primary_success_rows_sha256",
            "merged_primary_rows_sha256",
            "primary_rows_canonical_value_identical",
            "record_total",
            "resource_tier_counts",
        },
        field="merge receipt",
    )
    primary_rows = list(primary.predictions)
    for payload, role in (
        (primary.payload, "primary prediction artifact"),
        (request.payload, "retry request"),
        (retry.payload, "retry prediction artifact"),
        (merged.payload, "merged prediction artifact"),
        (receipt, "merge receipt"),
    ):
        _assert_payload_runner_provenance(payload, repository, role=role)
    expected_rows = sorted(
        [*primary_rows, *retry.predictions],
        key=lambda row: row["index"],
    )
    primary_rows_sha256 = sha256_json(primary_rows)
    expected_retry_binding = {
        "retry_request_sha256": request.digest.sha256,
        "source_primary_sha256": primary.digest.sha256,
        "primary_success_rows_sha256": request.payload["primary_success_rows_sha256"],
        "records_sha256": request.payload["records_sha256"],
    }
    expected_ids = [record["video_id"] for record in request.records]
    if (
        primary.payload["resource"]["tier"] != PRIMARY_RESOURCE_TIER
        or retry.payload["resource"]["tier"] != RETRY_RESOURCE_TIER
        or retry.payload["retry_binding"] != expected_retry_binding
        or retry.failures
        or [row["video_id"] for row in retry.predictions] != expected_ids
        or [row["video_id"] for row in primary.failures] != expected_ids
        or list(merged.rows) != expected_rows
    ):
        raise ESCountsMergeError("actual merge lineage does not reconstruct merged rows")
    for field in ("source", "input", "assets", "config"):
        if (
            retry.payload[field] != primary.payload[field]
            or merged.payload[field] != primary.payload[field]
        ):
            raise ESCountsMergeError(f"actual merge lineage changed {field}")
    if (
        retry.payload["runner_source_git_sha"] != primary.payload["runner_source_git_sha"]
        or retry.payload["runner_code_sha256"] != primary.payload["runner_code_sha256"]
        or retry.payload["worker_code_sha256"] != primary.payload["worker_code_sha256"]
        or retry.payload["runtime_versions"].get("worker_code_sha256")
        != primary.payload["runtime_versions"].get("worker_code_sha256")
        or retry.payload["runtime_versions"].get("worker_command_sha256")
        != primary.payload["runtime_versions"].get("worker_command_sha256")
        or retry.payload["runtime_versions"].get("container_image_id")
        != primary.payload["runtime_versions"].get("container_image_id")
    ):
        raise ESCountsMergeError("actual merge lineage changed execution provenance")
    if (
        merged.payload["source_primary_sha256"] != primary.digest.sha256
        or merged.payload["source_retry_request_sha256"] != request.digest.sha256
        or merged.payload["source_retry_sha256"] != retry.digest.sha256
        or merged.payload["primary_success_rows_sha256"] != primary_rows_sha256
        or merged.payload["merged_primary_rows_sha256"] != primary_rows_sha256
        or merged.payload["primary_rows_canonical_value_identical"] is not True
    ):
        raise ESCountsMergeError("merged artifact lineage hashes are not actual")
    expected_receipt = {
        "schema_version": 1,
        "artifact_type": "escounts_official_merge_receipt",
        "method_id": METHOD_ID,
        **repository.provenance_dict(),
        "output_file": merged.path.name,
        "output_sha256": merged.digest.sha256,
        "output_bytes": merged.digest.byte_count,
        "source_primary_sha256": primary.digest.sha256,
        "source_retry_request_sha256": request.digest.sha256,
        "source_retry_sha256": retry.digest.sha256,
        "primary_success_rows_sha256": primary_rows_sha256,
        "merged_primary_rows_sha256": primary_rows_sha256,
        "primary_rows_canonical_value_identical": True,
        "record_total": len(expected_rows),
        "resource_tier_counts": merged.payload["resource_tier_counts"],
    }
    if receipt != expected_receipt:
        raise ESCountsMergeError("merge receipt does not bind actual chain bytes")
    for path, digest, role in (
        (primary.path, primary.digest, "primary prediction artifact"),
        (request.path, request.digest, "retry request artifact"),
        (retry.path, retry.digest, "retry prediction artifact"),
        (merged.path, merged.digest, "merged prediction artifact"),
        (receipt_path, receipt_digest, "merge receipt"),
    ):
        _assert_file_unchanged(path, digest, role=role)
    repository.assert_unchanged()
    return ValidatedMergeChain(
        primary=primary,
        request=request,
        retry=retry,
        merged=merged,
        receipt_path=receipt_path,
        receipt_digest=receipt_digest,
        receipt=receipt,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pams.baselines.escounts_official_runner",
        description="Run the frozen official ESCounts zero-shot path label-free.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--sidecar", type=Path, required=True)
    predict_parser.add_argument("--commitment", type=Path, required=True)
    predict_parser.add_argument("--repository-root", type=Path, required=True)
    predict_parser.add_argument("--video-root", type=Path, required=True)
    predict_parser.add_argument("--encoder", type=Path, required=True)
    predict_parser.add_argument("--decoder", type=Path, required=True)
    predict_parser.add_argument("--official-source-root", type=Path, required=True)
    predict_parser.add_argument("--pytorchvideo-source-root", type=Path, required=True)
    predict_parser.add_argument("--worker-command-json", required=True)
    predict_parser.add_argument("--expected-container-image-id", required=True)
    predict_parser.add_argument("--device", default="cuda:0")
    predict_parser.add_argument(
        "--resource-tier",
        choices=(PRIMARY_RESOURCE_TIER, RETRY_RESOURCE_TIER),
        default=PRIMARY_RESOURCE_TIER,
    )
    predict_parser.add_argument("--retry-request", type=Path)
    predict_parser.add_argument("--primary-predictions", type=Path)
    predict_parser.add_argument("--video-id", action="append", dest="video_ids")
    predict_parser.add_argument("--output", type=Path, required=True)

    retry_parser = commands.add_parser("create-retry-request")
    retry_parser.add_argument("--repository-root", type=Path, required=True)
    retry_parser.add_argument("--primary-predictions", type=Path, required=True)
    retry_parser.add_argument("--output", type=Path, required=True)

    merge_parser = commands.add_parser("merge")
    merge_parser.add_argument("--repository-root", type=Path, required=True)
    merge_parser.add_argument("--primary-predictions", type=Path, required=True)
    merge_parser.add_argument("--retry-request", type=Path, required=True)
    merge_parser.add_argument("--retry-predictions", type=Path, required=True)
    merge_parser.add_argument("--output", type=Path, required=True)
    merge_parser.add_argument("--receipt", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "create-retry-request":
            payload = create_retry_request(
                primary_predictions_path=args.primary_predictions,
                output_path=args.output,
                repository_root=args.repository_root,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "merge":
            payload = merge_retry_artifacts(
                primary_predictions_path=args.primary_predictions,
                retry_request_path=args.retry_request,
                retry_predictions_path=args.retry_predictions,
                output_path=args.output,
                receipt_path=args.receipt,
                repository_root=args.repository_root,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        result = run_escounts_official(
            args.sidecar,
            args.commitment,
            args.video_root,
            args.encoder,
            args.decoder,
            args.official_source_root,
            repository_root=args.repository_root,
            device=args.device,
            resource_tier=args.resource_tier,
            video_ids=args.video_ids,
            retry_request_path=args.retry_request,
            primary_predictions_path=args.primary_predictions,
            output_path=args.output,
            worker_command=args.worker_command_json,
            pytorchvideo_source_root=args.pytorchvideo_source_root,
            expected_container_image_id=args.expected_container_image_id,
        )
        output_digest = _stable_file_digest(args.output.resolve(strict=True))
        print(
            json.dumps(
                {
                    "status": "complete" if not result.failures else "partial",
                    "selected_count": len(result.selected_video_ids),
                    "success_total": len(result.successes),
                    "failure_total": len(result.failures),
                    "output_sha256": output_digest.sha256,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if not result.failures else 2
    except (OSError, ValueError, ESCountsOfficialError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
