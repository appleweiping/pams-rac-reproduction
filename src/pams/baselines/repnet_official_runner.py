"""Isolated runner for the current official RepNet ``ckpt-70`` asset.

This module deliberately does not register RepNet as a runnable PAMS
baseline.  Google Research publishes the current implementation only inside a
Colab notebook.  The production entry point therefore reads the notebook
directly from the preregistered Git commit, selects the official model and
inference definitions without rewriting them, and hard-fails when that source,
TensorFlow, the checkpoint bundle, or the Keras variable mapping cannot be
verified.

Prediction accepts only a strict :class:`~pams.data.PoseInputManifest` plus its
independent commitment.  No target/count/action loader is reachable from this
module.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import math
import os
import stat
import subprocess
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

import numpy as np

from pams.data import (
    PoseInputCommitment,
    PoseInputManifest,
    UnlabeledVideoRecord,
    load_pose_input_commitment,
    load_pose_input_manifest,
    pose_input_identity_sha256,
    validate_pose_input_binding,
)
from pams.metrics import round_count

OFFICIAL_SOURCE_REPOSITORY = "https://github.com/google-research/google-research"
OFFICIAL_SOURCE_COMMIT = "ec7c3d346277b737bc2decffcd1b533d4b7ec105"
OFFICIAL_NOTEBOOK_PATH = "repnet/repnet_colab.ipynb"
OFFICIAL_NOTEBOOK_BYTES = 88_218
OFFICIAL_NOTEBOOK_SHA256 = (
    "44bf5ec222dfd3f5babb7a577e1c395114b8c490279918cc803ac12c329c2318"
)
OFFICIAL_CHECKPOINT_PREFIX = "ckpt-70"


def sha256_json(payload: Any) -> str:
    """Canonical JSON SHA-256 without importing the PyTorch runtime."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fsync_directory(path: str | Path) -> None:
    """Persist directory entries on POSIX; Windows has no portable equivalent."""

    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def durable_mkdir(path: str | Path) -> Path:
    """Create a directory hierarchy while persisting new POSIX entries."""

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
        try:
            directory.mkdir()
        except FileExistsError:
            if not directory.is_dir():
                raise
        else:
            fsync_directory(directory.parent)
    if not target.is_dir():
        raise NotADirectoryError(f"path is not a directory: {target}")
    return target


class RepNetOfficialError(RuntimeError):
    """Base class for an explicit official-runner blocker."""


class RepNetOfficialSourceError(RepNetOfficialError):
    """The supplied Google Research checkout is not the frozen source."""


class RepNetOfficialDependencyError(RepNetOfficialError):
    """The isolated runtime lacks an official-notebook dependency."""


class RepNetOfficialCheckpointError(RepNetOfficialError):
    """The three-file ``ckpt-70`` bundle is incomplete or has changed."""


class RepNetOfficialBlockedError(RepNetOfficialError):
    """The official TensorFlow model cannot be restored faithfully."""


@dataclass(frozen=True, slots=True)
class CheckpointFileSpec:
    """Published identity for one official GCS checkpoint object."""

    name: str
    byte_count: int
    md5: str

    def __post_init__(self) -> None:
        if Path(self.name).name != self.name or not self.name:
            raise ValueError("checkpoint object name must be one plain filename")
        if self.byte_count < 1:
            raise ValueError("checkpoint object byte_count must be positive")
        if len(self.md5) != 32 or any(character not in "0123456789abcdef" for character in self.md5):
            raise ValueError("checkpoint object md5 must be lowercase hexadecimal")


OFFICIAL_CKPT70_FILES: tuple[CheckpointFileSpec, ...] = (
    CheckpointFileSpec(
        name="checkpoint",
        byte_count=33,
        md5="95c36d059d6d0d6d9be5916c0f26d73e",
    ),
    CheckpointFileSpec(
        name="ckpt-70.index",
        byte_count=10_209,
        md5="5292903f0dfd1c916db65d04ea634040",
    ),
    CheckpointFileSpec(
        name="ckpt-70.data-00000-of-00001",
        byte_count=309_204_174,
        md5="7d97aa53df6817737a84a4b1c4a10204",
    ),
)


@dataclass(frozen=True, slots=True)
class RepNetOfficialConfig:
    """Source-notebook parameters frozen before UCFRep evaluation."""

    schema_version: int = 1
    source_commit: str = OFFICIAL_SOURCE_COMMIT
    checkpoint_prefix: str = OFFICIAL_CHECKPOINT_PREFIX
    window_frames: int = 64
    decode_width: int = 224
    decode_height: int = 224
    model_image_size: int = 112
    strides: tuple[int, ...] = (1, 2, 3, 4)
    inference_batch_size: int = 20
    global_periodicity_threshold: float = 0.2
    frame_periodicity_threshold: float = 0.5
    constant_speed: bool = False
    median_filter: bool = True
    fully_periodic: bool = False
    rounding: str = "nearest_integer_half_up"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only RepNet official config schema_version=1 is supported")
        if self.source_commit != OFFICIAL_SOURCE_COMMIT:
            raise ValueError("RepNet official source commit is frozen")
        if self.checkpoint_prefix != OFFICIAL_CHECKPOINT_PREFIX:
            raise ValueError("RepNet official checkpoint prefix is frozen")
        if self.window_frames != 64 or self.model_image_size != 112:
            raise ValueError("RepNet official model dimensions are frozen")
        if self.decode_width != 224 or self.decode_height != 224:
            raise ValueError("RepNet official OpenCV decode resize is frozen")
        if self.strides != (1, 2, 3, 4) or self.inference_batch_size != 20:
            raise ValueError("RepNet official inference schedule is frozen")
        if not math.isclose(self.global_periodicity_threshold, 0.2):
            raise ValueError("RepNet global periodicity threshold is frozen")
        if not math.isclose(self.frame_periodicity_threshold, 0.5):
            raise ValueError("RepNet frame periodicity threshold is frozen")
        if self.constant_speed or not self.median_filter or self.fully_periodic:
            raise ValueError("RepNet official post-processing switches are frozen")
        if self.rounding != "nearest_integer_half_up":
            raise ValueError("shared evaluator rounding is frozen")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
            "checkpoint_prefix": self.checkpoint_prefix,
            "window_frames": self.window_frames,
            "decode_width": self.decode_width,
            "decode_height": self.decode_height,
            "model_image_size": self.model_image_size,
            "strides": list(self.strides),
            "inference_batch_size": self.inference_batch_size,
            "global_periodicity_threshold": self.global_periodicity_threshold,
            "frame_periodicity_threshold": self.frame_periodicity_threshold,
            "constant_speed": self.constant_speed,
            "median_filter": self.median_filter,
            "fully_periodic": self.fully_periodic,
            "rounding": self.rounding,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


FROZEN_REPNET_OFFICIAL_CONFIG = RepNetOfficialConfig()


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
    md5: str
    byte_count: int
    identity: _FileIdentity


def _stable_file_digest(path: Path) -> _StableFileDigest:
    """Hash one regular non-symlink file and reject a concurrent change."""

    try:
        before = path.lstat()
    except FileNotFoundError:
        raise
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"input must be a regular non-symlink file: {path}")

    sha256 = hashlib.sha256()
    md5 = hashlib.md5(usedforsecurity=False)
    byte_count = 0
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"input must remain a regular file while opened: {path}")
        while chunk := handle.read(1024 * 1024):
            sha256.update(chunk)
            md5.update(chunk)
            byte_count += len(chunk)
        closed = os.fstat(handle.fileno())
    after = path.lstat()

    identities = tuple(
        _FileIdentity.from_stat(value) for value in (before, opened, closed, after)
    )
    if len(set(identities)) != 1 or byte_count != before.st_size:
        raise RuntimeError(f"input changed while it was hashed: {path}")
    return _StableFileDigest(
        sha256=sha256.hexdigest(),
        md5=md5.hexdigest(),
        byte_count=byte_count,
        identity=identities[0],
    )


def _assert_file_identity(path: Path, expected: _FileIdentity, *, role: str) -> None:
    try:
        observed_stat = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"{role} disappeared during prediction: {path}") from exc
    if stat.S_ISLNK(observed_stat.st_mode) or not stat.S_ISREG(observed_stat.st_mode):
        raise RuntimeError(f"{role} stopped being a regular non-symlink file: {path}")
    if _FileIdentity.from_stat(observed_stat) != expected:
        raise RuntimeError(f"{role} changed during prediction: {path}")


@dataclass(frozen=True, slots=True)
class VerifiedCheckpointFile:
    """Observed digest of one published checkpoint object."""

    name: str
    byte_count: int
    md5: str
    sha256: str
    _path: Path
    _identity: _FileIdentity

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bytes": self.byte_count,
            "published_md5": self.md5,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class VerifiedCheckpoint:
    """A complete, byte-verified official ``ckpt-70`` bundle."""

    directory: Path
    files: tuple[VerifiedCheckpointFile, ...]
    aggregate_sha256: str

    def assert_unchanged(self) -> None:
        for item in self.files:
            _assert_file_identity(
                item._path,
                item._identity,
                role=f"checkpoint object {item.name!r}",
            )


def _verify_checkpoint_with_specs(
    checkpoint_dir: str | Path,
    file_specs: Sequence[CheckpointFileSpec],
) -> VerifiedCheckpoint:
    """Verify supplied object identities; used with tiny specs only in tests."""

    try:
        directory = Path(checkpoint_dir).expanduser().resolve(strict=True)
    except OSError as exc:
        raise RepNetOfficialCheckpointError(
            f"checkpoint_dir cannot be resolved: {checkpoint_dir}"
        ) from exc
    if not directory.is_dir():
        raise RepNetOfficialCheckpointError(
            f"checkpoint_dir is not a directory: {directory}"
        )
    specs = tuple(file_specs)
    if not specs:
        raise RepNetOfficialCheckpointError("checkpoint file specification is empty")

    observed: list[VerifiedCheckpointFile] = []
    for spec in specs:
        path = directory / spec.name
        try:
            digest = _stable_file_digest(path)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            raise RepNetOfficialCheckpointError(
                f"unable to verify checkpoint object {spec.name!r}: {exc}"
            ) from exc
        if digest.byte_count != spec.byte_count:
            raise RepNetOfficialCheckpointError(
                f"checkpoint object {spec.name!r} has {digest.byte_count} bytes; "
                f"expected {spec.byte_count}"
            )
        if digest.md5 != spec.md5:
            raise RepNetOfficialCheckpointError(
                f"checkpoint object {spec.name!r} MD5 mismatch"
            )
        observed.append(
            VerifiedCheckpointFile(
                name=spec.name,
                byte_count=digest.byte_count,
                md5=digest.md5,
                sha256=digest.sha256,
                _path=path,
                _identity=digest.identity,
            )
        )

    canonical = {
        "checkpoint_prefix": OFFICIAL_CHECKPOINT_PREFIX,
        "files": [item.to_dict() for item in observed],
    }
    return VerifiedCheckpoint(
        directory=directory,
        files=tuple(observed),
        aggregate_sha256=sha256_json(canonical),
    )


def verify_official_checkpoint(checkpoint_dir: str | Path) -> VerifiedCheckpoint:
    """Verify the immutable published identities of all three ckpt-70 objects."""

    return _verify_checkpoint_with_specs(checkpoint_dir, OFFICIAL_CKPT70_FILES)


@dataclass(frozen=True, slots=True)
class VerifiedLabelFreeInputs:
    """A sidecar/commitment pair verified before any model is imported."""

    manifest: PoseInputManifest
    commitment: PoseInputCommitment
    video_root: Path
    sidecar_sha256: str
    commitment_sha256: str
    identity_sha256: str
    _sidecar_path: Path
    _commitment_path: Path
    _sidecar_identity: _FileIdentity
    _commitment_identity: _FileIdentity

    def assert_unchanged(self) -> None:
        _assert_file_identity(
            self._sidecar_path,
            self._sidecar_identity,
            role="label-free input sidecar",
        )
        _assert_file_identity(
            self._commitment_path,
            self._commitment_identity,
            role="label-free input commitment",
        )


def _load_label_free_inputs(
    sidecar_path: str | Path,
    commitment_path: str | Path,
    video_root: str | Path,
    *,
    require_exact_membership: bool,
) -> VerifiedLabelFreeInputs:
    sidecar = Path(sidecar_path).expanduser().resolve(strict=True)
    commitment_file = Path(commitment_path).expanduser().resolve(strict=True)
    root = Path(video_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(f"video_root is not a directory: {root}")

    # Hash before parsing.  The strict manifest loader scans raw object keys
    # and rejects count/action before json.loads materializes either value.
    sidecar_digest = _stable_file_digest(sidecar)
    commitment_digest = _stable_file_digest(commitment_file)
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
    _assert_file_identity(sidecar, sidecar_digest.identity, role="label-free input sidecar")
    _assert_file_identity(
        commitment_file,
        commitment_digest.identity,
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
        _sidecar_identity=sidecar_digest.identity,
        _commitment_identity=commitment_digest.identity,
    )


class DecodeStatus(str, Enum):
    """Per-video decode/inference status preserved in the prediction ledger."""

    OK = "ok"
    MISSING_VIDEO = "missing_video"
    NON_REGULAR_VIDEO = "non_regular_video"
    VIDEO_HASH_MISMATCH = "video_hash_mismatch"
    VIDEO_CHANGED = "video_changed"
    DECODE_FAILED = "decode_failed"
    INFERENCE_FAILED = "inference_failed"


class RepNetOfficialDecodeError(RuntimeError):
    """A backend decode failure that must remain distinct from inference."""


@dataclass(frozen=True, slots=True)
class BackendPrediction:
    """Successful raw output of the official notebook ``get_counts`` call."""

    raw_count: float
    chosen_stride: int
    confidence: float
    decoded_frames: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.raw_count) or self.raw_count < 0.0:
            raise ValueError("official RepNet raw_count must be finite and non-negative")
        if self.chosen_stride not in FROZEN_REPNET_OFFICIAL_CONFIG.strides:
            raise ValueError("official RepNet returned an unregistered stride")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("official RepNet confidence must be in [0, 1]")
        if self.decoded_frames < 1:
            raise ValueError("successful official RepNet prediction needs decoded frames")


class _OfficialBackend(Protocol):
    @property
    def source_commit(self) -> str:
        """Exact Google Research source revision."""

        ...

    @property
    def source_notebook_sha256(self) -> str:
        """SHA-256 of the notebook blob used to build the backend."""

        ...

    @property
    def runtime_versions(self) -> Mapping[str, str]:
        """Versions of the dependencies that executed official code."""

        ...

    def predict(
        self,
        video_path: Path,
        config: RepNetOfficialConfig,
    ) -> BackendPrediction:
        """Decode and predict by calling the fixed official implementation."""


@dataclass(frozen=True, slots=True)
class RepNetVideoPrediction:
    """One label-free official prediction or explicit decode failure."""

    video_id: str
    video_locator: str
    expected_video_sha256: str | None
    observed_video_sha256: str | None
    raw_count: float | None
    rounded_count: int | None
    chosen_stride: int | None
    confidence: float | None
    decoded_frames: int
    decode_status: DecodeStatus
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        successful = self.decode_status is DecodeStatus.OK
        outputs = (
            self.raw_count,
            self.rounded_count,
            self.chosen_stride,
            self.confidence,
        )
        if successful and any(value is None for value in outputs):
            raise ValueError("successful prediction must contain every RepNet output")
        if not successful and any(value is not None for value in outputs):
            raise ValueError("failed prediction must not contain synthetic RepNet outputs")
        if successful and self.failure_reason is not None:
            raise ValueError("successful prediction cannot contain failure_reason")
        if not successful and not self.failure_reason:
            raise ValueError("failed prediction must contain failure_reason")
        if self.decoded_frames < 0:
            raise ValueError("decoded_frames must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_locator": self.video_locator,
            "expected_video_sha256": self.expected_video_sha256,
            "observed_video_sha256": self.observed_video_sha256,
            "raw_count": self.raw_count,
            "rounded_count": self.rounded_count,
            "chosen_stride": self.chosen_stride,
            "confidence": self.confidence,
            "decoded_frames": self.decoded_frames,
            "decode_status": self.decode_status.value,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True, slots=True)
class RepNetOfficialRunResult:
    """Immutable label-free output of one official-current RepNet run."""

    inputs: VerifiedLabelFreeInputs
    checkpoint: VerifiedCheckpoint
    config: RepNetOfficialConfig
    source_notebook_sha256: str
    runtime_versions: Mapping[str, str]
    selected_video_ids: tuple[str, ...]
    predictions: tuple[RepNetVideoPrediction, ...]
    schema_version: int = 1
    method_id: str = "repnet-official-current-ckpt70"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only RepNet official result schema_version=1 is supported")
        if self.source_notebook_sha256 != OFFICIAL_NOTEBOOK_SHA256:
            raise ValueError("source_notebook_sha256 is not the frozen official blob")
        full_ids = tuple(record.video_id for record in self.inputs.manifest.records)
        if not self.selected_video_ids:
            raise ValueError("RepNet selection must contain at least one video")
        if len(set(self.selected_video_ids)) != len(self.selected_video_ids):
            raise ValueError("selected_video_ids must be unique")
        selected_set = set(self.selected_video_ids)
        expected_ids = tuple(video_id for video_id in full_ids if video_id in selected_set)
        if expected_ids != self.selected_video_ids:
            raise ValueError(
                "selected_video_ids must be a sidecar-ordered subset of the full input"
            )
        if len(self.predictions) != len(self.selected_video_ids):
            raise ValueError("prediction count must match selected_video_ids")
        observed_ids = [record.video_id for record in self.predictions]
        if observed_ids != list(self.selected_video_ids):
            raise ValueError("prediction order must match the sidecar-ordered selection")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method_id": self.method_id,
            "classification": (
                "official-current-ckpt70 / independent label-free UCFRep evaluation"
            ),
            "eligible_for_original_pams_repnet_cell": False,
            "source": {
                "repository": OFFICIAL_SOURCE_REPOSITORY,
                "commit": OFFICIAL_SOURCE_COMMIT,
                "notebook_path": OFFICIAL_NOTEBOOK_PATH,
                "notebook_sha256": self.source_notebook_sha256,
            },
            "input": {
                "protocol": self.inputs.manifest.protocol,
                "split": self.inputs.manifest.split,
                "sample_count": len(self.inputs.manifest.records),
                "sidecar_sha256": self.inputs.sidecar_sha256,
                "commitment_sha256": self.inputs.commitment_sha256,
                "sidecar_fingerprint": self.inputs.manifest.fingerprint,
                "identity_sha256": self.inputs.identity_sha256,
            },
            "selection": {
                "selected_count": len(self.selected_video_ids),
                "selected_video_ids": list(self.selected_video_ids),
                "selected_video_ids_sha256": sha256_json(
                    list(self.selected_video_ids)
                ),
            },
            "checkpoint": {
                "prefix": OFFICIAL_CHECKPOINT_PREFIX,
                "aggregate_sha256": self.checkpoint.aggregate_sha256,
                "files": [item.to_dict() for item in self.checkpoint.files],
            },
            "config": {
                "sha256": self.config.fingerprint,
                "values": self.config.to_dict(),
            },
            "runtime_versions": dict(sorted(self.runtime_versions.items())),
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


_OFFICIAL_DEFINITION_NAMES = frozenset(
    {
        "ResnetPeriodEstimator",
        "get_sims",
        "flatten_sequential_feats",
        "scaled_dot_product_attention",
        "point_wise_feed_forward_network",
        "MultiHeadAttention",
        "TransformerLayer",
        "pairwise_l2_distance",
        "get_repnet_model",
        "read_video",
        "get_score",
        "get_counts",
        "MAPPING_NEW_TO_OLD_LAYER_NAMES",
        "load_ckpt_with_custom_layer_mapping",
    }
)


def _git_bytes(source_root: Path, arguments: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_root), *arguments],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
        suffix = "" if not detail else f": {detail}"
        raise RepNetOfficialSourceError(
            f"unable to inspect official source checkout{suffix}"
        ) from exc
    return completed.stdout


def _read_frozen_official_notebook(source_root: str | Path) -> bytes:
    try:
        root = Path(source_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise RepNetOfficialSourceError(
            f"official source root cannot be resolved: {source_root}"
        ) from exc
    if not root.is_dir():
        raise RepNetOfficialSourceError(f"official source root is not a directory: {root}")
    revision = _git_bytes(root, ["rev-parse", "--verify", "HEAD"]).decode().strip()
    if revision != OFFICIAL_SOURCE_COMMIT:
        raise RepNetOfficialSourceError(
            f"official source checkout is {revision!r}; expected {OFFICIAL_SOURCE_COMMIT}"
        )
    notebook = _git_bytes(
        root,
        ["show", f"{OFFICIAL_SOURCE_COMMIT}:{OFFICIAL_NOTEBOOK_PATH}"],
    )
    observed_sha256 = hashlib.sha256(notebook).hexdigest()
    if (
        len(notebook) != OFFICIAL_NOTEBOOK_BYTES
        or observed_sha256 != OFFICIAL_NOTEBOOK_SHA256
    ):
        raise RepNetOfficialSourceError(
            "official notebook blob does not match the frozen byte count/SHA-256"
        )
    return notebook


def _read_frozen_official_notebook_file(notebook_path: str | Path) -> bytes:
    """Read a standalone copy of the exact preregistered notebook blob."""

    try:
        source = Path(notebook_path).expanduser().resolve(strict=True)
        digest = _stable_file_digest(source)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RepNetOfficialSourceError(
            f"official notebook file cannot be verified: {notebook_path}"
        ) from exc
    if (
        digest.byte_count != OFFICIAL_NOTEBOOK_BYTES
        or digest.sha256 != OFFICIAL_NOTEBOOK_SHA256
    ):
        raise RepNetOfficialSourceError(
            "official notebook file does not match the frozen byte count/SHA-256"
        )
    try:
        notebook = source.read_bytes()
    except OSError as exc:
        raise RepNetOfficialSourceError(
            f"official notebook file cannot be read: {source}"
        ) from exc
    if (
        len(notebook) != digest.byte_count
        or hashlib.sha256(notebook).hexdigest() != digest.sha256
    ):
        raise RepNetOfficialSourceError(
            "official notebook file changed between verification and loading"
        )
    _assert_file_identity(source, digest.identity, role="official notebook file")
    return notebook


def _import_official_dependencies() -> tuple[dict[str, Any], dict[str, str]]:
    modules: dict[str, Any] = {}
    required = (
        "cv2",
        "scipy",
        "scipy.signal",
        "tensorflow",
        "tensorflow.compat.v2",
        "tensorflow.python.training",
    )
    for name in required:
        try:
            modules[name] = importlib.import_module(name)
        except (ImportError, OSError) as exc:
            raise RepNetOfficialDependencyError(
                f"official RepNet dependency {name!r} is unavailable: {exc}"
            ) from exc

    tensorflow = modules["tensorflow"]
    tensorflow_v2 = modules["tensorflow.compat.v2"]
    if not callable(getattr(tensorflow_v2, "executing_eagerly", None)):
        raise RepNetOfficialDependencyError(
            "tensorflow.compat.v2 does not expose the eager API required by RepNet"
        )
    if not bool(tensorflow_v2.executing_eagerly()):
        raise RepNetOfficialDependencyError("official RepNet requires TensorFlow eager mode")
    checkpoint_reader = getattr(
        modules["tensorflow.python.training"],
        "py_checkpoint_reader",
        None,
    )
    if checkpoint_reader is None:
        raise RepNetOfficialDependencyError(
            "TensorFlow private py_checkpoint_reader API required by the official "
            "ckpt-70 mapping is unavailable"
        )

    namespace = {
        "cv2": modules["cv2"],
        "np": np,
        "medfilt": modules["scipy.signal"].medfilt,
        "tf": tensorflow_v2,
        "py_checkpoint_reader": checkpoint_reader,
        "layers": tensorflow_v2.keras.layers,
        "regularizers": tensorflow_v2.keras.regularizers,
    }
    versions = {
        "opencv": str(getattr(modules["cv2"], "__version__", "unknown")),
        "numpy": np.__version__,
        "scipy": str(getattr(modules["scipy"], "__version__", "unknown")),
        "tensorflow": str(getattr(tensorflow, "__version__", "unknown")),
    }
    return namespace, versions


def _node_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
        return node.name
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        target = node.targets[0]
        if isinstance(target, ast.Name):
            return target.id
    return None


def _official_definition_module(notebook: bytes) -> tuple[types.ModuleType, str]:
    try:
        payload = json.loads(notebook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepNetOfficialSourceError("official RepNet notebook is invalid JSON") from exc
    cells = payload.get("cells") if isinstance(payload, dict) else None
    if not isinstance(cells, list):
        raise RepNetOfficialSourceError("official RepNet notebook has no cells list")

    selected: list[ast.stmt] = []
    found: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        raw_source = cell.get("source")
        if not isinstance(raw_source, list) or not all(
            isinstance(line, str) for line in raw_source
        ):
            continue
        # IPython shell/magic lines are not part of the official model API.
        sanitized = "".join(
            "\n" if line.lstrip().startswith(("!", "%")) else line
            for line in raw_source
        )
        try:
            tree = ast.parse(sanitized, filename=OFFICIAL_NOTEBOOK_PATH)
        except SyntaxError:
            continue
        for node in tree.body:
            name = _node_name(node)
            if name not in _OFFICIAL_DEFINITION_NAMES:
                continue
            if name in found:
                raise RepNetOfficialSourceError(
                    f"official notebook defines required symbol {name!r} more than once"
                )
            found.add(name)
            selected.append(node)

    missing = sorted(_OFFICIAL_DEFINITION_NAMES - found)
    if missing:
        raise RepNetOfficialSourceError(
            f"official notebook is missing required definitions: {missing}"
        )

    namespace, versions = _import_official_dependencies()
    module = types.ModuleType("_pams_fixed_official_repnet")
    module.__dict__.update(namespace)
    module.__dict__["__official_runtime_versions__"] = versions
    source_tree = ast.fix_missing_locations(
        ast.Module(body=selected, type_ignores=[])
    )
    try:
        exec(
            compile(
                source_tree,
                filename=f"{OFFICIAL_SOURCE_COMMIT}:{OFFICIAL_NOTEBOOK_PATH}",
                mode="exec",
            ),
            module.__dict__,
        )
    except Exception as exc:
        raise RepNetOfficialBlockedError(
            f"official RepNet definitions could not be loaded unchanged: {exc}"
        ) from exc
    return module, hashlib.sha256(notebook).hexdigest()


class OfficialNotebookBackend:
    """Calls the exact model, decoder, and ``get_counts`` definitions in the notebook."""

    source_commit = OFFICIAL_SOURCE_COMMIT

    def __init__(
        self,
        api: types.ModuleType,
        model: Any,
        *,
        source_notebook_sha256: str,
        runtime_versions: Mapping[str, str],
    ) -> None:
        self._api = api
        self._model = model
        self.source_notebook_sha256 = source_notebook_sha256
        self.runtime_versions = dict(runtime_versions)

    @classmethod
    def from_checkout(
        cls,
        source_root: str | Path,
        checkpoint: VerifiedCheckpoint,
    ) -> OfficialNotebookBackend:
        notebook = _read_frozen_official_notebook(source_root)
        return cls._from_notebook_bytes(notebook, checkpoint)

    @classmethod
    def from_notebook_file(
        cls,
        notebook_path: str | Path,
        checkpoint: VerifiedCheckpoint,
    ) -> OfficialNotebookBackend:
        """Build from a standalone notebook after exact byte verification."""

        notebook = _read_frozen_official_notebook_file(notebook_path)
        return cls._from_notebook_bytes(notebook, checkpoint)

    @classmethod
    def _from_notebook_bytes(
        cls,
        notebook: bytes,
        checkpoint: VerifiedCheckpoint,
    ) -> OfficialNotebookBackend:
        api, notebook_sha256 = _official_definition_module(notebook)
        try:
            model = api.get_repnet_model(str(checkpoint.directory))
            mappings = api.MAPPING_NEW_TO_OLD_LAYER_NAMES
            weights = model.weights
            if len(mappings) != len(weights):
                raise ValueError(
                    "official custom mapping length does not equal model weight count "
                    f"({len(mappings)} != {len(weights)})"
                )
            api.load_ckpt_with_custom_layer_mapping(
                model,
                str(checkpoint.directory),
                mappings,
            )
        except Exception as exc:
            raise RepNetOfficialBlockedError(
                "official ckpt-70 could not be restored with the notebook's exact "
                f"TensorFlow/Keras variable mapping: {exc}"
            ) from exc
        checkpoint.assert_unchanged()
        return cls(
            api,
            model,
            source_notebook_sha256=notebook_sha256,
            runtime_versions=api.__official_runtime_versions__,
        )

    def predict(
        self,
        video_path: Path,
        config: RepNetOfficialConfig,
    ) -> BackendPrediction:
        if config != FROZEN_REPNET_OFFICIAL_CONFIG:
            raise ValueError("official RepNet prediction requires the frozen config")
        try:
            frames, _fps = self._api.read_video(
                str(video_path),
                width=config.decode_width,
                height=config.decode_height,
            )
        except Exception as exc:
            raise RepNetOfficialDecodeError(
                f"official OpenCV decoder raised {type(exc).__name__}: {exc}"
            ) from exc
        frames_array = np.asarray(frames)
        if (
            frames_array.ndim != 4
            or frames_array.shape[0] < 1
            or frames_array.shape[-1] != 3
        ):
            raise RepNetOfficialDecodeError(
                "official OpenCV decoder returned no valid RGB frames"
            )

        output = self._api.get_counts(
            self._model,
            frames_array,
            strides=list(config.strides),
            batch_size=config.inference_batch_size,
            threshold=config.global_periodicity_threshold,
            within_period_threshold=config.frame_periodicity_threshold,
            constant_speed=config.constant_speed,
            median_filter=config.median_filter,
            fully_periodic=config.fully_periodic,
        )
        if not isinstance(output, tuple) or len(output) != 5:
            raise ValueError("official get_counts returned an unexpected structure")
        _period, confidence, _within_period, per_frame_counts, stride = output
        densities = np.asarray(per_frame_counts, dtype=np.float64).reshape(-1)
        if densities.size != frames_array.shape[0]:
            raise ValueError("official get_counts output length does not match decoded frames")
        if not np.isfinite(densities).all() or np.any(densities < 0.0):
            raise ValueError("official get_counts returned invalid per-frame counts")
        return BackendPrediction(
            raw_count=float(densities.sum(dtype=np.float64)),
            chosen_stride=int(np.asarray(stride).reshape(())),
            confidence=float(np.asarray(confidence).reshape(())),
            decoded_frames=int(frames_array.shape[0]),
        )


def _select_video_records(
    inputs: VerifiedLabelFreeInputs,
    video_ids: Sequence[str] | None,
) -> tuple[UnlabeledVideoRecord, ...]:
    """Select only after the complete sidecar and commitment are verified."""

    records = inputs.manifest.records
    if video_ids is None:
        return records
    requested = tuple(str(video_id).strip() for video_id in video_ids)
    if not requested or any(not video_id for video_id in requested):
        raise ValueError("video_ids must contain at least one non-empty identifier")
    if len(set(requested)) != len(requested):
        raise ValueError("video_ids must not contain duplicates")
    available = {record.video_id for record in records}
    unknown = sorted(set(requested) - available)
    if unknown:
        raise ValueError(f"video_ids are not present in the verified sidecar: {unknown}")
    requested_set = set(requested)
    return tuple(record for record in records if record.video_id in requested_set)


def _resolved_video_path(inputs: VerifiedLabelFreeInputs, locator: str) -> Path:
    relative = PurePosixPath(locator)
    candidate = inputs.video_root.joinpath(*relative.parts).resolve(strict=False)
    try:
        candidate.relative_to(inputs.video_root)
    except ValueError as exc:
        raise ValueError(f"video locator escapes video_root: {locator!r}") from exc
    return candidate


def _failure(
    *,
    video_id: str,
    video_locator: str,
    expected_sha256: str | None,
    observed_sha256: str | None,
    status: DecodeStatus,
    reason: str,
    decoded_frames: int = 0,
) -> RepNetVideoPrediction:
    return RepNetVideoPrediction(
        video_id=video_id,
        video_locator=video_locator,
        expected_video_sha256=expected_sha256,
        observed_video_sha256=observed_sha256,
        raw_count=None,
        rounded_count=None,
        chosen_stride=None,
        confidence=None,
        decoded_frames=decoded_frames,
        decode_status=status,
        failure_reason=reason,
    )


def _run_verified_backend(
    inputs: VerifiedLabelFreeInputs,
    checkpoint: VerifiedCheckpoint,
    backend: _OfficialBackend,
    *,
    config: RepNetOfficialConfig = FROZEN_REPNET_OFFICIAL_CONFIG,
    video_ids: Sequence[str] | None = None,
) -> RepNetOfficialRunResult:
    """Run an already source/checkpoint-verified backend.

    Kept separate so firewall, ledger, and failure behavior can be tested
    without pretending that a fake unit-test model is an official baseline.
    The public production entry point constructs only
    :class:`OfficialNotebookBackend`.
    """

    if config != FROZEN_REPNET_OFFICIAL_CONFIG:
        raise ValueError("official RepNet runner accepts only the frozen config")
    if backend.source_commit != OFFICIAL_SOURCE_COMMIT:
        raise RepNetOfficialSourceError("backend source commit is not the frozen official commit")
    if backend.source_notebook_sha256 != OFFICIAL_NOTEBOOK_SHA256:
        raise RepNetOfficialSourceError(
            "backend notebook SHA-256 is not the frozen official blob"
        )
    selected_records = _select_video_records(inputs, video_ids)

    predictions: list[RepNetVideoPrediction] = []
    for record in selected_records:
        try:
            video_path = _resolved_video_path(inputs, record.video_path)
        except ValueError as exc:
            predictions.append(
                _failure(
                    video_id=record.video_id,
                    video_locator=record.video_path,
                    expected_sha256=record.video_sha256,
                    observed_sha256=None,
                    status=DecodeStatus.NON_REGULAR_VIDEO,
                    reason=str(exc),
                )
            )
            continue
        if not video_path.exists():
            predictions.append(
                _failure(
                    video_id=record.video_id,
                    video_locator=record.video_path,
                    expected_sha256=record.video_sha256,
                    observed_sha256=None,
                    status=DecodeStatus.MISSING_VIDEO,
                    reason="video file does not exist",
                )
            )
            continue
        try:
            digest = _stable_file_digest(video_path)
        except (OSError, RuntimeError, ValueError) as exc:
            predictions.append(
                _failure(
                    video_id=record.video_id,
                    video_locator=record.video_path,
                    expected_sha256=record.video_sha256,
                    observed_sha256=None,
                    status=DecodeStatus.NON_REGULAR_VIDEO,
                    reason=str(exc),
                )
            )
            continue
        if record.video_sha256 is None or digest.sha256 != record.video_sha256:
            predictions.append(
                _failure(
                    video_id=record.video_id,
                    video_locator=record.video_path,
                    expected_sha256=record.video_sha256,
                    observed_sha256=digest.sha256,
                    status=DecodeStatus.VIDEO_HASH_MISMATCH,
                    reason=(
                        "sidecar has no source-video SHA-256"
                        if record.video_sha256 is None
                        else "source-video SHA-256 does not match the sidecar"
                    ),
                )
            )
            continue

        try:
            output = backend.predict(video_path, config)
        except RepNetOfficialDecodeError as exc:
            predictions.append(
                _failure(
                    video_id=record.video_id,
                    video_locator=record.video_path,
                    expected_sha256=record.video_sha256,
                    observed_sha256=digest.sha256,
                    status=DecodeStatus.DECODE_FAILED,
                    reason=str(exc),
                )
            )
            continue
        except Exception as exc:
            predictions.append(
                _failure(
                    video_id=record.video_id,
                    video_locator=record.video_path,
                    expected_sha256=record.video_sha256,
                    observed_sha256=digest.sha256,
                    status=DecodeStatus.INFERENCE_FAILED,
                    reason=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

        try:
            _assert_file_identity(video_path, digest.identity, role="source video")
        except RuntimeError as exc:
            predictions.append(
                _failure(
                    video_id=record.video_id,
                    video_locator=record.video_path,
                    expected_sha256=record.video_sha256,
                    observed_sha256=digest.sha256,
                    status=DecodeStatus.VIDEO_CHANGED,
                    reason=str(exc),
                    decoded_frames=output.decoded_frames,
                )
            )
            continue
        predictions.append(
            RepNetVideoPrediction(
                video_id=record.video_id,
                video_locator=record.video_path,
                expected_video_sha256=record.video_sha256,
                observed_video_sha256=digest.sha256,
                raw_count=output.raw_count,
                rounded_count=round_count(output.raw_count),
                chosen_stride=output.chosen_stride,
                confidence=output.confidence,
                decoded_frames=output.decoded_frames,
                decode_status=DecodeStatus.OK,
            )
        )

    inputs.assert_unchanged()
    checkpoint.assert_unchanged()
    return RepNetOfficialRunResult(
        inputs=inputs,
        checkpoint=checkpoint,
        config=config,
        source_notebook_sha256=backend.source_notebook_sha256,
        runtime_versions=backend.runtime_versions,
        selected_video_ids=tuple(record.video_id for record in selected_records),
        predictions=tuple(predictions),
    )


def write_repnet_result_exclusive(
    path: str | Path,
    result: RepNetOfficialRunResult,
) -> str:
    """Durably write one result without replacing an earlier experiment."""

    destination = Path(path)
    durable_mkdir(destination.parent)
    encoded = (
        json.dumps(
            result.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(destination, flags, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite RepNet result: {destination}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(destination.parent)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        raise
    return hashlib.sha256(encoded).hexdigest()


def run_repnet_official(
    sidecar_path: str | Path,
    commitment_path: str | Path,
    video_root: str | Path,
    checkpoint_dir: str | Path,
    official_source_root: str | Path | None = None,
    *,
    official_notebook_path: str | Path | None = None,
    video_ids: Sequence[str] | None = None,
    output_path: str | Path | None = None,
) -> RepNetOfficialRunResult:
    """Run the source-pinned official-current RepNet adapter.

    Input loading and its raw-key label firewall happen before TensorFlow or
    any model source is imported.  The returned predictions contain no target
    values; metric evaluation must happen in a separate sealed process.
    """

    inputs = _load_label_free_inputs(
        sidecar_path,
        commitment_path,
        video_root,
        require_exact_membership=True,
    )
    if output_path is not None and Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite RepNet result: {output_path}")
    if (official_source_root is None) == (official_notebook_path is None):
        raise ValueError(
            "provide exactly one of official_source_root or official_notebook_path"
        )
    checkpoint = verify_official_checkpoint(checkpoint_dir)
    if official_notebook_path is not None:
        backend = OfficialNotebookBackend.from_notebook_file(
            official_notebook_path,
            checkpoint,
        )
    else:
        assert official_source_root is not None
        backend = OfficialNotebookBackend.from_checkout(
            official_source_root,
            checkpoint,
        )
    result = _run_verified_backend(
        inputs,
        checkpoint,
        backend,
        video_ids=video_ids,
    )
    if output_path is not None:
        write_repnet_result_exclusive(output_path, result)
    return result


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pams.baselines.repnet_official_runner",
        description=(
            "Run the source-pinned official RepNet ckpt-70 adapter on a strict "
            "label-free sidecar."
        ),
    )
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--official-notebook",
        type=Path,
        help="Standalone official notebook with the frozen byte count/SHA-256.",
    )
    source.add_argument(
        "--official-source-root",
        type=Path,
        help="Google Research Git checkout at the frozen commit.",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        help=(
            "Optional video ID to run; repeat for a subset. The complete sidecar "
            "is verified first and selected output follows sidecar order."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Argparse entry point for the isolated RepNet environment."""

    parser = _build_argument_parser()
    arguments = parser.parse_args(None if argv is None else list(argv))
    try:
        result = run_repnet_official(
            arguments.sidecar,
            arguments.commitment,
            arguments.video_root,
            arguments.checkpoint_dir,
            arguments.official_source_root,
            official_notebook_path=arguments.official_notebook,
            video_ids=arguments.video_ids,
            output_path=arguments.output,
        )
        output_digest = _stable_file_digest(arguments.output).sha256
    except (OSError, RepNetOfficialError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "output_sha256": output_digest,
                "selected_count": len(result.selected_video_ids),
                "selected_video_ids": list(result.selected_video_ids),
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


__all__ = [
    "BackendPrediction",
    "DecodeStatus",
    "FROZEN_REPNET_OFFICIAL_CONFIG",
    "OFFICIAL_CHECKPOINT_PREFIX",
    "OFFICIAL_CKPT70_FILES",
    "OFFICIAL_NOTEBOOK_BYTES",
    "OFFICIAL_NOTEBOOK_PATH",
    "OFFICIAL_NOTEBOOK_SHA256",
    "OFFICIAL_SOURCE_COMMIT",
    "OfficialNotebookBackend",
    "RepNetOfficialBlockedError",
    "RepNetOfficialCheckpointError",
    "RepNetOfficialConfig",
    "RepNetOfficialDependencyError",
    "RepNetOfficialError",
    "RepNetOfficialRunResult",
    "RepNetOfficialSourceError",
    "RepNetVideoPrediction",
    "main",
    "run_repnet_official",
    "verify_official_checkpoint",
    "write_repnet_result_exclusive",
]


if __name__ == "__main__":
    raise SystemExit(main())
