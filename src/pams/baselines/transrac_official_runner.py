"""Strict label-free runner for the pinned official TransRAC assets.

The released TransRAC source and checkpoint are executed only from mounts in
an isolated compatibility image.  This repository contains the independently
written adapter, integrity checks, and result schema; it never vendors the
third-party source, checkpoints, or UCFRep videos.

Targets are intentionally absent from this module and its CLI.  Scoring is a
separate process implemented by :mod:`pams.baselines.transrac_official_score`.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from pams.baselines.repnet_official_runner import (
    VerifiedLabelFreeInputs,
    _assert_file_identity,
    _load_label_free_inputs,
    _resolved_video_path,
    _select_video_records,
    _stable_file_digest,
    durable_mkdir,
    fsync_directory,
    sha256_json,
)
from pams.metrics import round_count

OFFICIAL_SOURCE_REPOSITORY = "https://github.com/SvipRepetitionCounting/TransRAC"
OFFICIAL_SOURCE_COMMIT = "68bdd4daa60ed7c3174a7f6bf86f6537b6fa0979"
OFFICIAL_SOURCE_ARCHIVE_URL = (
    "https://codeload.github.com/SvipRepetitionCounting/TransRAC/tar.gz/"
    f"{OFFICIAL_SOURCE_COMMIT}"
)
OFFICIAL_SOURCE_ARCHIVE_BYTES = 33_386_001
OFFICIAL_SOURCE_ARCHIVE_SHA256 = (
    "b3ec4563f28ff251a87c789de6a0270a709d8656fdc7ec3280648619de10e4ed"
)
OFFICIAL_SOURCE_TREE_FILES = 498
OFFICIAL_SOURCE_TREE_BYTES = 35_917_768
OFFICIAL_SOURCE_TREE_SHA256 = (
    "2ca6ee835e7f0beb1c505fd7a2d061981c86d023d83b982a5f79da41837d1547"
)
OFFICIAL_LICENSE_SHA256 = (
    "ae5e6184c04d6c36e58a8f1c875f3dd3ab4a25a5884ccf41d40bad2dd634a111"
)
COMPATIBILITY_IMAGE_DIGEST = (
    "sha256:ea96abe0e7d17cf4720343e5867256aacd5209aca738087ec92f62b9fe31ff51"
)
SOURCE_MODULE_PREFIXES = ("dataset", "models", "mmaction")
RUNNER_CODE_RELATIVE_PATH = "src/pams/baselines/transrac_official_runner.py"
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

BACKBONE_URL = (
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/"
    "swin_tiny_patch244_window877_kinetics400_1k.pth"
)
CHECKPOINT_URL = (
    "https://shanghaitecheducn-my.sharepoint.com/:f:/g/personal/"
    "dongsx_shanghaitech_edu_cn/"
    "Eg2-I2dG_BhKkuBJGnTg200BhhsEAYmCx3xgAvRuTEURuA?e=YURfkP"
)

METHOD_ID = "transrac-official-modern-compat-repcounta"
CLASSIFICATION = (
    "TransRAC official checkpoint / modern compatibility / local label-free evaluation"
)


class TransRACOfficialError(RuntimeError):
    """Base class for official TransRAC integrity and execution failures."""


class TransRACOfficialSourceError(TransRACOfficialError):
    """The supplied source archive/tree differs from the pinned revision."""


class TransRACOfficialAssetError(TransRACOfficialError):
    """A backbone or counting checkpoint has a wrong byte identity."""


class TransRACOfficialDependencyError(TransRACOfficialError):
    """The independently frozen compatibility environment is incomplete."""


class TransRACOfficialDecodeError(TransRACOfficialError):
    """The official UCFRep preprocessor could not produce its fixed input."""


@dataclass(frozen=True, slots=True)
class AssetSpec:
    """Frozen identity and provenance for one externally mounted asset."""

    name: str
    byte_count: int
    sha256: str
    role: str
    url: str
    published_quickxorhash: str | None = None

    def __post_init__(self) -> None:
        if Path(self.name).name != self.name or not self.name:
            raise ValueError("asset name must be one plain filename")
        if self.byte_count < 1:
            raise ValueError("asset byte_count must be positive")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("asset sha256 must be lowercase hexadecimal")

    def public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bytes": self.byte_count,
            "sha256": self.sha256,
            "role": self.role,
            "url": self.url,
            "published_quickxorhash": self.published_quickxorhash,
        }


SOURCE_ARCHIVE_SPEC = AssetSpec(
    name="source.tar.gz",
    byte_count=OFFICIAL_SOURCE_ARCHIVE_BYTES,
    sha256=OFFICIAL_SOURCE_ARCHIVE_SHA256,
    role="official_source_archive",
    url=OFFICIAL_SOURCE_ARCHIVE_URL,
)
BACKBONE_SPEC = AssetSpec(
    name="swin_tiny_patch244_window877_kinetics400_1k.pth",
    byte_count=127_414_861,
    sha256="9950e3be6b0b3763f80575dfbbe7db7cbeb18345e32363b73e8d44952ef76acc",
    role="officially_linked_video_swin_backbone",
    url=BACKBONE_URL,
)
CHECKPOINT_SPEC = AssetSpec(
    name="transrac_ckpt_pytorch_171.pt",
    byte_count=522_535_387,
    sha256="32311c4f08bfb1988bfaede8a4691eba5cd5db6928478ee1f26f527e10341a80",
    role="official_repcount_a_transrac_checkpoint",
    url=CHECKPOINT_URL,
    published_quickxorhash="I5no0SyJraTDbTgNIxQ/DnpZCjg=",
)


@dataclass(frozen=True, slots=True)
class VerifiedAsset:
    """One exact regular file whose identity remains checked during inference."""

    spec: AssetSpec
    path: Path
    _identity: Any

    def assert_unchanged(self) -> None:
        try:
            digest = _stable_file_digest(self.path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise TransRACOfficialAssetError(
                f"unable to re-verify {self.spec.role}: {exc}"
            ) from exc
        if (
            digest.byte_count != self.spec.byte_count
            or digest.sha256 != self.spec.sha256
        ):
            raise TransRACOfficialAssetError(
                f"{self.spec.role} bytes/SHA-256 changed after verification"
            )
        _assert_file_identity(self.path, self._identity, role=self.spec.role)

    def to_dict(self) -> dict[str, Any]:
        return self.spec.public_dict()


def _verify_asset_with_spec(path: str | Path, spec: AssetSpec) -> VerifiedAsset:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
        digest = _stable_file_digest(resolved)
    except (OSError, RuntimeError, ValueError) as exc:
        raise TransRACOfficialAssetError(f"unable to verify {spec.role}: {exc}") from exc
    if resolved.name != spec.name:
        raise TransRACOfficialAssetError(
            f"{spec.role} filename is {resolved.name!r}; expected {spec.name!r}"
        )
    if digest.byte_count != spec.byte_count:
        raise TransRACOfficialAssetError(
            f"{spec.role} has {digest.byte_count} bytes; expected {spec.byte_count}"
        )
    if digest.sha256 != spec.sha256:
        raise TransRACOfficialAssetError(f"{spec.role} SHA-256 mismatch")
    return VerifiedAsset(spec=spec, path=resolved, _identity=digest.identity)


def verify_official_assets(
    source_archive: str | Path,
    backbone: str | Path,
    checkpoint: str | Path,
) -> tuple[VerifiedAsset, VerifiedAsset, VerifiedAsset]:
    """Verify the source archive, linked backbone, and official checkpoint."""

    return (
        _verify_asset_with_spec(source_archive, SOURCE_ARCHIVE_SPEC),
        _verify_asset_with_spec(backbone, BACKBONE_SPEC),
        _verify_asset_with_spec(checkpoint, CHECKPOINT_SPEC),
    )


@dataclass(frozen=True, slots=True)
class SourceTreeSpec:
    file_count: int
    byte_count: int
    sha256: str


FROZEN_SOURCE_TREE = SourceTreeSpec(
    file_count=OFFICIAL_SOURCE_TREE_FILES,
    byte_count=OFFICIAL_SOURCE_TREE_BYTES,
    sha256=OFFICIAL_SOURCE_TREE_SHA256,
)


def _source_tree_digest(root: Path) -> tuple[int, int, str]:
    """Hash the extracted archive tree, ignoring only Python bytecode caches."""

    import hashlib
    import stat

    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    paths = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    for path in paths:
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise TransRACOfficialSourceError(f"source tree contains symlink: {relative}")
        if not stat.S_ISREG(metadata.st_mode):
            continue
        relative_bytes = relative.as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
        file_count += 1
        byte_count += len(content)
    return file_count, byte_count, digest.hexdigest()


def _verify_source_tree_with_spec(
    source_root: str | Path,
    spec: SourceTreeSpec,
) -> Path:
    try:
        root = Path(source_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise TransRACOfficialSourceError(
            f"official source root cannot be resolved: {source_root}"
        ) from exc
    if not root.is_dir():
        raise TransRACOfficialSourceError(f"official source root is not a directory: {root}")
    observed = _source_tree_digest(root)
    expected = (spec.file_count, spec.byte_count, spec.sha256)
    if observed != expected:
        raise TransRACOfficialSourceError(
            "extracted official source tree file count/bytes/SHA-256 mismatch"
        )
    return root


def verify_official_source_tree(source_root: str | Path) -> Path:
    """Verify every extracted source file against the pinned archive tree."""

    return _verify_source_tree_with_spec(source_root, FROZEN_SOURCE_TREE)


@dataclass(frozen=True, slots=True)
class TransRACOfficialConfig:
    schema_version: int = 1
    source_commit: str = OFFICIAL_SOURCE_COMMIT
    num_frames: int = 64
    resize_height: int = 224
    resize_width: int = 224
    scales: tuple[int, ...] = (1, 4, 8)
    open_set: bool = False
    torch_seed: int = 1
    gpu_memory_limit_mib: int = 8192
    rounding: str = "nearest_integer_half_up"
    compatibility: str = "python3.11-torch2.5.1-cu124-mmcv1.4.0"
    compatibility_image_digest: str = COMPATIBILITY_IMAGE_DIGEST

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.source_commit != OFFICIAL_SOURCE_COMMIT:
            raise ValueError("TransRAC source/config schema is frozen")
        if (
            self.num_frames != 64
            or (self.resize_height, self.resize_width) != (224, 224)
            or self.scales != (1, 4, 8)
        ):
            raise ValueError("TransRAC preprocessing/model schedule is frozen")
        if self.open_set or self.torch_seed != 1:
            raise ValueError("TransRAC inference switches are frozen")
        if self.gpu_memory_limit_mib != 8192:
            raise ValueError("TransRAC GPU memory cap is frozen")
        if self.rounding != "nearest_integer_half_up":
            raise ValueError("shared evaluator rounding is frozen")
        if self.compatibility_image_digest != COMPATIBILITY_IMAGE_DIGEST:
            raise ValueError("TransRAC compatibility image digest is frozen")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
            "num_frames": self.num_frames,
            "resize_height": self.resize_height,
            "resize_width": self.resize_width,
            "scales": list(self.scales),
            "open_set": self.open_set,
            "torch_seed": self.torch_seed,
            "gpu_memory_limit_mib": self.gpu_memory_limit_mib,
            "rounding": self.rounding,
            "compatibility": self.compatibility,
            "compatibility_image_digest": self.compatibility_image_digest,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


FROZEN_TRANSRAC_OFFICIAL_CONFIG = TransRACOfficialConfig()


def _clean_runner_git_revision(repository_root: Path) -> str:
    try:
        revision = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise TransRACOfficialSourceError(
            "unable to audit TransRAC runner Git checkout"
        ) from exc
    if GIT_SHA_PATTERN.fullmatch(revision) is None or status:
        raise TransRACOfficialSourceError(
            "strict TransRAC prediction requires a clean full-SHA Git checkout"
        )
    return revision


def _is_source_owned_module(name: str) -> bool:
    return any(
        name == prefix or name.startswith(f"{prefix}.")
        for prefix in SOURCE_MODULE_PREFIXES
    )


def _reject_preloaded_source_modules() -> None:
    preloaded = sorted(name for name in sys.modules if _is_source_owned_module(name))
    if preloaded:
        raise TransRACOfficialSourceError(
            "official source module names were already loaded before the frozen "
            f"tree was mounted: {preloaded}"
        )


def _assert_source_module_origins(source_root: Path) -> None:
    observed = {
        name: module
        for name, module in sys.modules.items()
        if _is_source_owned_module(name)
    }
    required = {"dataset.UCFRep_loader", "models.TransRAC"}
    missing = sorted(required - set(observed))
    if missing:
        raise TransRACOfficialSourceError(
            f"required official source modules were not loaded: {missing}"
        )
    for name, module in sorted(observed.items()):
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) or not origin:
            raise TransRACOfficialSourceError(
                f"official source module {name!r} has no auditable file origin"
            )
        try:
            resolved = Path(origin).resolve(strict=True)
            resolved.relative_to(source_root)
        except (OSError, ValueError) as exc:
            raise TransRACOfficialSourceError(
                f"official source module {name!r} was imported outside the "
                f"frozen source tree: {origin}"
            ) from exc


@contextlib.contextmanager
def _isolated_source_import_path(source_root: Path) -> Any:
    _reject_preloaded_source_modules()
    original = list(sys.path)
    sys.path.insert(0, str(source_root))
    try:
        yield
    finally:
        sys.path[:] = original


class DecodeStatus(str, Enum):
    OK = "ok"
    MISSING_VIDEO = "missing_video"
    NON_REGULAR_VIDEO = "non_regular_video"
    VIDEO_HASH_MISMATCH = "video_hash_mismatch"
    VIDEO_CHANGED = "video_changed"
    DECODE_FAILED = "decode_failed"
    INFERENCE_FAILED = "inference_failed"
    NONFINITE_OUTPUT = "nonfinite_output"


@dataclass(frozen=True, slots=True)
class BackendPrediction:
    raw_count: float
    decoded_frame_count: int
    sampled_frame_count: int
    density_min: float
    density_max: float
    density_mean: float
    peak_allocated_mib: float
    peak_reserved_mib: float

    def __post_init__(self) -> None:
        numeric = (
            self.raw_count,
            self.density_min,
            self.density_max,
            self.density_mean,
            self.peak_allocated_mib,
            self.peak_reserved_mib,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("TransRAC backend output contains non-finite values")
        if self.raw_count < 0 or self.decoded_frame_count < 1:
            raise ValueError("TransRAC count/frame metadata is invalid")
        if self.sampled_frame_count != 64:
            raise ValueError("TransRAC must sample exactly 64 frames")


class _OfficialBackend(Protocol):
    source_commit: str
    source_tree_sha256: str
    runtime_versions: Mapping[str, str]
    restore_audit: Mapping[str, Any]

    def predict(
        self,
        video_path: Path,
        config: TransRACOfficialConfig,
    ) -> BackendPrediction: ...


class ModernCompatBackend:
    """Thin caller around the unmodified pinned official source."""

    source_commit = OFFICIAL_SOURCE_COMMIT
    source_tree_sha256 = OFFICIAL_SOURCE_TREE_SHA256

    def __init__(
        self,
        *,
        torch: Any,
        model: Any,
        video_reader: Any,
        device: Any,
        runtime_versions: Mapping[str, str],
        restore_audit: Mapping[str, Any],
    ) -> None:
        self._torch = torch
        self._model = model
        self._video_reader = video_reader
        self._device = device
        self.runtime_versions: Mapping[str, str] = dict(runtime_versions)
        self.restore_audit: Mapping[str, Any] = dict(restore_audit)

    @classmethod
    def from_assets(
        cls,
        source_root: Path,
        backbone: VerifiedAsset,
        checkpoint: VerifiedAsset,
        config: TransRACOfficialConfig,
    ) -> ModernCompatBackend:
        _reject_preloaded_source_modules()
        try:
            torch = importlib.import_module("torch")
        except (ImportError, OSError) as exc:
            raise TransRACOfficialDependencyError(f"PyTorch is unavailable: {exc}") from exc
        if not bool(torch.cuda.is_available()):
            raise TransRACOfficialDependencyError("CUDA is required for official TransRAC")
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        fraction = min(
            config.gpu_memory_limit_mib * 1024 * 1024 / properties.total_memory,
            1.0,
        )
        torch.cuda.set_per_process_memory_fraction(fraction, device=device)
        torch.manual_seed(config.torch_seed)
        torch.cuda.manual_seed_all(config.torch_seed)

        try:
            with _isolated_source_import_path(source_root):
                loader_module = importlib.import_module("dataset.UCFRep_loader")
                model_module = importlib.import_module("models.TransRAC")
                config_path = (
                    source_root
                    / "configs/recognition/swin/"
                    "swin_tiny_patch244_window877_kinetics400_1k.py"
                )
                model = model_module.TransferModel(
                    config=str(config_path),
                    checkpoint=str(backbone.path),
                    num_frames=config.num_frames,
                    scales=list(config.scales),
                    OPEN=config.open_set,
                )
                backbone.assert_unchanged()
                model = torch.nn.DataParallel(model.to(device), device_ids=[0])
                saved = torch.load(
                    checkpoint.path,
                    map_location="cpu",
                    weights_only=False,
                )
                checkpoint.assert_unchanged()
                state_dict = (
                    saved.get("state_dict", saved) if isinstance(saved, dict) else None
                )
                if not isinstance(state_dict, dict):
                    raise ValueError("checkpoint state_dict is missing")
                incompatible = model.load_state_dict(state_dict, strict=False)
                _assert_source_module_origins(source_root)
        except TransRACOfficialError:
            raise
        except Exception as exc:
            raise TransRACOfficialDependencyError(
                f"official TransRAC source/checkpoint could not be loaded: {exc}"
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if len(state_dict) != 230 or missing or unexpected:
            raise TransRACOfficialAssetError(
                "official TransRAC checkpoint did not restore all 230 keys exactly"
            )
        epoch = saved.get("epoch") if isinstance(saved, dict) else None
        if epoch != 174:
            raise TransRACOfficialAssetError(
                "official checkpoint payload epoch is not the audited value 174"
            )
        model.eval()
        runtime_versions = {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "mmcv": importlib.metadata.version("mmcv"),
            "timm": importlib.metadata.version("timm"),
            "einops": importlib.metadata.version("einops"),
            "kornia": importlib.metadata.version("kornia"),
            "gpu_name": str(properties.name),
        }
        return cls(
            torch=torch,
            model=model,
            video_reader=loader_module.VideoRead,
            device=device,
            runtime_versions=runtime_versions,
            restore_audit={
                "checkpoint_epoch": epoch,
                "checkpoint_filename_epoch_token": 171,
                "state_dict_keys": len(state_dict),
                "loaded_key_count": len(state_dict) - len(unexpected),
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "strict_key_coverage": True,
            },
        )

    def predict(
        self,
        video_path: Path,
        config: TransRACOfficialConfig,
    ) -> BackendPrediction:
        try:
            reader = self._video_reader(str(video_path), num_frames=config.num_frames)
            frames = reader.crop_frame()
            tensor = frames.transpose(0, 1).unsqueeze(0)
        except Exception as exc:
            raise TransRACOfficialDecodeError(
                f"official VideoRead raised {type(exc).__name__}: {exc}"
            ) from exc
        if tuple(tensor.shape) != (1, 3, 64, 224, 224):
            raise TransRACOfficialDecodeError(
                f"official VideoRead returned unexpected input shape {tuple(tensor.shape)}"
            )
        if not bool(self._torch.isfinite(tensor).all().item()):
            raise TransRACOfficialDecodeError("official VideoRead returned NaN/Inf")
        self._torch.cuda.reset_peak_memory_stats(self._device)
        with self._torch.no_grad():
            output, similarity = self._model(tensor.to(self._device))
        self._torch.cuda.synchronize(self._device)
        if not bool(self._torch.isfinite(output).all().item()) or not bool(
            self._torch.isfinite(similarity).all().item()
        ):
            raise ValueError("official TransRAC output contains NaN/Inf")
        return BackendPrediction(
            raw_count=float(self._torch.sum(output, dim=1).item()),
            decoded_frame_count=int(reader.frame_length),
            sampled_frame_count=config.num_frames,
            density_min=float(output.min().item()),
            density_max=float(output.max().item()),
            density_mean=float(output.mean().item()),
            peak_allocated_mib=float(
                self._torch.cuda.max_memory_allocated(self._device) / (1024**2)
            ),
            peak_reserved_mib=float(
                self._torch.cuda.max_memory_reserved(self._device) / (1024**2)
            ),
        )


@dataclass(frozen=True, slots=True)
class TransRACVideoPrediction:
    video_id: str
    video_locator: str
    expected_video_sha256: str | None
    observed_video_sha256: str | None
    raw_count: float | None
    rounded_count: int | None
    decoded_frame_count: int
    sampled_frame_count: int
    density_min: float | None
    density_max: float | None
    density_mean: float | None
    peak_allocated_mib: float | None
    peak_reserved_mib: float | None
    decode_status: DecodeStatus
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "video_locator": self.video_locator,
            "expected_video_sha256": self.expected_video_sha256,
            "observed_video_sha256": self.observed_video_sha256,
            "raw_count": self.raw_count,
            "rounded_count": self.rounded_count,
            "decoded_frame_count": self.decoded_frame_count,
            "sampled_frame_count": self.sampled_frame_count,
            "density_min": self.density_min,
            "density_max": self.density_max,
            "density_mean": self.density_mean,
            "peak_allocated_mib": self.peak_allocated_mib,
            "peak_reserved_mib": self.peak_reserved_mib,
            "decode_status": self.decode_status.value,
            "failure_reason": self.failure_reason,
        }


def _failure(
    record: Any,
    *,
    status: DecodeStatus,
    reason: str,
    observed_sha256: str | None = None,
    decoded_frames: int = 0,
) -> TransRACVideoPrediction:
    return TransRACVideoPrediction(
        video_id=record.video_id,
        video_locator=record.video_path,
        expected_video_sha256=record.video_sha256,
        observed_video_sha256=observed_sha256,
        raw_count=None,
        rounded_count=None,
        decoded_frame_count=decoded_frames,
        sampled_frame_count=0,
        density_min=None,
        density_max=None,
        density_mean=None,
        peak_allocated_mib=None,
        peak_reserved_mib=None,
        decode_status=status,
        failure_reason=reason,
    )


@dataclass(frozen=True, slots=True)
class TransRACOfficialRunResult:
    inputs: VerifiedLabelFreeInputs
    source_archive: VerifiedAsset
    backbone: VerifiedAsset
    checkpoint: VerifiedAsset
    config: TransRACOfficialConfig
    runtime_versions: Mapping[str, str]
    restore_audit: Mapping[str, Any]
    runner_git_sha: str
    runner_code_sha256: str
    selected_video_ids: tuple[str, ...]
    predictions: tuple[TransRACVideoPrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        failures = [
            {
                "video_id": row.video_id,
                "video_locator": row.video_locator,
                "decode_status": row.decode_status.value,
                "failure_reason": row.failure_reason,
            }
            for row in self.predictions
            if row.decode_status is not DecodeStatus.OK
        ]
        return {
            "schema_version": 1,
            "method_id": METHOD_ID,
            "classification": CLASSIFICATION,
            "eligible_for_original_pams_transrac_cell": False,
            "labels_loaded": False,
            "scoring_performed": False,
            "runner_provenance": {
                "source_git_sha": self.runner_git_sha,
                "runner_code_sha256": self.runner_code_sha256,
                "compatibility_image_digest": COMPATIBILITY_IMAGE_DIGEST,
            },
            "source": {
                "repository": OFFICIAL_SOURCE_REPOSITORY,
                "commit": OFFICIAL_SOURCE_COMMIT,
                "archive": self.source_archive.to_dict(),
                "tree_sha256": OFFICIAL_SOURCE_TREE_SHA256,
                "tree_files": OFFICIAL_SOURCE_TREE_FILES,
                "tree_bytes": OFFICIAL_SOURCE_TREE_BYTES,
                "license_file_sha256": OFFICIAL_LICENSE_SHA256,
                "license_status": "ambiguous-apache-file-versus-anti-996-readme-badge",
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
                "selected_video_ids_sha256": sha256_json(list(self.selected_video_ids)),
            },
            "assets": {
                "backbone": self.backbone.to_dict(),
                "checkpoint": self.checkpoint.to_dict(),
                "third_party_assets_redistributed": False,
            },
            "config": {
                "sha256": self.config.fingerprint,
                "values": self.config.to_dict(),
            },
            "runtime_versions": dict(sorted(self.runtime_versions.items())),
            "restore_audit": dict(self.restore_audit),
            "decode_ok_total": len(self.predictions) - len(failures),
            "failure_total": len(failures),
            "failures": failures,
            "predictions": [row.to_dict() for row in self.predictions],
        }


def _assert_full_file_unchanged(path: Path, expected: Any, *, role: str) -> None:
    try:
        observed = _stable_file_digest(path)
    except (OSError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"{role} could not be rehashed after use: {path}") from exc
    if (
        observed.sha256 != expected.sha256
        or observed.byte_count != expected.byte_count
    ):
        raise RuntimeError(f"{role} bytes/SHA-256 changed during prediction: {path}")
    _assert_file_identity(path, expected.identity, role=role)


def _run_verified_backend(
    inputs: VerifiedLabelFreeInputs,
    source_archive: VerifiedAsset,
    backbone: VerifiedAsset,
    checkpoint: VerifiedAsset,
    backend: _OfficialBackend,
    *,
    config: TransRACOfficialConfig = FROZEN_TRANSRAC_OFFICIAL_CONFIG,
    video_ids: Sequence[str] | None = None,
    runner_git_sha: str,
    runner_code_sha256: str,
) -> TransRACOfficialRunResult:
    if backend.source_commit != OFFICIAL_SOURCE_COMMIT:
        raise TransRACOfficialSourceError("backend source commit is not frozen")
    if backend.source_tree_sha256 != OFFICIAL_SOURCE_TREE_SHA256:
        raise TransRACOfficialSourceError("backend source tree SHA-256 is not frozen")
    selected = _select_video_records(inputs, video_ids)
    predictions: list[TransRACVideoPrediction] = []
    for record in selected:
        path = _resolved_video_path(inputs, record.video_path)
        if not path.exists():
            predictions.append(
                _failure(record, status=DecodeStatus.MISSING_VIDEO, reason="video is missing")
            )
            continue
        try:
            digest = _stable_file_digest(path)
        except (OSError, RuntimeError, ValueError) as exc:
            predictions.append(
                _failure(record, status=DecodeStatus.NON_REGULAR_VIDEO, reason=str(exc))
            )
            continue
        if record.video_sha256 is None or digest.sha256 != record.video_sha256:
            predictions.append(
                _failure(
                    record,
                    status=DecodeStatus.VIDEO_HASH_MISMATCH,
                    reason="source-video SHA-256 does not match the sidecar",
                    observed_sha256=digest.sha256,
                )
            )
            continue
        try:
            output = backend.predict(path, config)
        except TransRACOfficialDecodeError as exc:
            predictions.append(
                _failure(
                    record,
                    status=DecodeStatus.DECODE_FAILED,
                    reason=str(exc),
                    observed_sha256=digest.sha256,
                )
            )
            continue
        except Exception as exc:
            predictions.append(
                _failure(
                    record,
                    status=DecodeStatus.INFERENCE_FAILED,
                    reason=f"{type(exc).__name__}: {exc}",
                    observed_sha256=digest.sha256,
                )
            )
            continue
        try:
            _assert_full_file_unchanged(path, digest, role="source video")
        except (RuntimeError, ValueError) as exc:
            predictions.append(
                _failure(
                    record,
                    status=DecodeStatus.VIDEO_CHANGED,
                    reason=str(exc),
                    observed_sha256=digest.sha256,
                    decoded_frames=output.decoded_frame_count,
                )
            )
            continue
        predictions.append(
            TransRACVideoPrediction(
                video_id=record.video_id,
                video_locator=record.video_path,
                expected_video_sha256=record.video_sha256,
                observed_video_sha256=digest.sha256,
                raw_count=output.raw_count,
                rounded_count=round_count(output.raw_count),
                decoded_frame_count=output.decoded_frame_count,
                sampled_frame_count=output.sampled_frame_count,
                density_min=output.density_min,
                density_max=output.density_max,
                density_mean=output.density_mean,
                peak_allocated_mib=output.peak_allocated_mib,
                peak_reserved_mib=output.peak_reserved_mib,
                decode_status=DecodeStatus.OK,
            )
        )
    inputs.assert_unchanged()
    for asset in (source_archive, backbone, checkpoint):
        asset.assert_unchanged()
    return TransRACOfficialRunResult(
        inputs=inputs,
        source_archive=source_archive,
        backbone=backbone,
        checkpoint=checkpoint,
        config=config,
        runtime_versions=backend.runtime_versions,
        restore_audit=backend.restore_audit,
        runner_git_sha=runner_git_sha,
        runner_code_sha256=runner_code_sha256,
        selected_video_ids=tuple(record.video_id for record in selected),
        predictions=tuple(predictions),
    )


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
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
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite TransRAC artifact: {path}") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)
    return __import__("hashlib").sha256(encoded).hexdigest()


def _remove_artifact_if_exact(path: Path, expected: Any) -> bool:
    """Remove only the exact file created by this process after a paired-write failure."""

    try:
        observed = _stable_file_digest(path)
    except (FileNotFoundError, OSError, RuntimeError, ValueError):
        return False
    if observed != expected:
        return False
    try:
        path.unlink()
    except OSError:
        return False
    fsync_directory(path.parent)
    return True


def write_transrac_result_exclusive(
    output_path: str | Path,
    result: TransRACOfficialRunResult,
) -> tuple[Path, str, Path, str]:
    destination = Path(output_path)
    receipt_path = destination.with_suffix(".receipt.json")
    durable_mkdir(destination.parent)
    collisions = [str(path) for path in (destination, receipt_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite TransRAC artifacts: {collisions}")
    payload = result.to_dict()
    prediction_sha256 = _write_json_exclusive(destination, payload)
    receipt = {
        "schema_version": 1,
        "artifact_type": "transrac_official_prediction_receipt",
        "method_id": METHOD_ID,
        "prediction_file": destination.name,
        "prediction_sha256": prediction_sha256,
        "prediction_bytes": destination.stat().st_size,
        "prediction_total": len(result.predictions),
        "decode_ok_total": sum(
            row.decode_status is DecodeStatus.OK for row in result.predictions
        ),
        "failure_total": sum(
            row.decode_status is not DecodeStatus.OK for row in result.predictions
        ),
        "source_commit": OFFICIAL_SOURCE_COMMIT,
        "source_archive_sha256": result.source_archive.spec.sha256,
        "source_tree_sha256": OFFICIAL_SOURCE_TREE_SHA256,
        "backbone_sha256": result.backbone.spec.sha256,
        "checkpoint_sha256": result.checkpoint.spec.sha256,
        "input_sidecar_sha256": result.inputs.sidecar_sha256,
        "input_commitment_sha256": result.inputs.commitment_sha256,
        "input_identity_sha256": result.inputs.identity_sha256,
        "config_sha256": result.config.fingerprint,
        "compatibility_image_digest": COMPATIBILITY_IMAGE_DIGEST,
        "runner_source_git_sha": result.runner_git_sha,
        "runner_code_sha256": result.runner_code_sha256,
        "labels_loaded": False,
        "scoring_performed": False,
    }
    created_prediction = _stable_file_digest(destination)
    try:
        receipt_sha256 = _write_json_exclusive(receipt_path, receipt)
    except Exception as exc:
        if not _remove_artifact_if_exact(destination, created_prediction):
            raise RuntimeError(
                "TransRAC receipt write failed and the exact orphan prediction "
                "could not be removed safely"
            ) from exc
        raise
    return destination, prediction_sha256, receipt_path, receipt_sha256


def run_transrac_official(
    *,
    sidecar_path: str | Path,
    commitment_path: str | Path,
    video_root: str | Path,
    source_archive_path: str | Path,
    source_root: str | Path,
    backbone_path: str | Path,
    checkpoint_path: str | Path,
    repository_root: str | Path,
    video_ids: Sequence[str] | None = None,
    output_path: str | Path | None = None,
) -> TransRACOfficialRunResult:
    """Run the frozen official assets without making target data reachable."""

    inputs = _load_label_free_inputs(
        sidecar_path,
        commitment_path,
        video_root,
        require_exact_membership=True,
    )
    if output_path is not None:
        output = Path(output_path)
        if output.exists() or output.with_suffix(".receipt.json").exists():
            raise FileExistsError(f"refusing to overwrite TransRAC output: {output}")
    repository = Path(repository_root).expanduser().resolve(strict=True)
    runner_git_sha = _clean_runner_git_revision(repository)
    runner_code = (repository / RUNNER_CODE_RELATIVE_PATH).resolve(strict=True)
    if runner_code != Path(__file__).resolve(strict=True):
        raise TransRACOfficialSourceError(
            "executed TransRAC runner is outside repository_root"
        )
    runner_code_digest = _stable_file_digest(runner_code)
    source_archive, backbone, checkpoint = verify_official_assets(
        source_archive_path,
        backbone_path,
        checkpoint_path,
    )
    verified_source_root = verify_official_source_tree(source_root)
    backend = ModernCompatBackend.from_assets(
        verified_source_root,
        backbone,
        checkpoint,
        FROZEN_TRANSRAC_OFFICIAL_CONFIG,
    )
    result = _run_verified_backend(
        inputs,
        source_archive,
        backbone,
        checkpoint,
        backend,
        video_ids=video_ids,
        runner_git_sha=runner_git_sha,
        runner_code_sha256=runner_code_digest.sha256,
    )
    if _source_tree_digest(verified_source_root) != (
        OFFICIAL_SOURCE_TREE_FILES,
        OFFICIAL_SOURCE_TREE_BYTES,
        OFFICIAL_SOURCE_TREE_SHA256,
    ):
        raise TransRACOfficialSourceError("official source tree changed during inference")
    if _stable_file_digest(runner_code) != runner_code_digest:
        raise TransRACOfficialSourceError("TransRAC runner code changed during inference")
    if _clean_runner_git_revision(repository) != runner_git_sha:
        raise TransRACOfficialSourceError(
            "TransRAC runner Git revision changed during inference"
        )
    if output_path is not None:
        write_transrac_result_exclusive(output_path, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pams.baselines.transrac_official_runner",
        description="Run pinned official TransRAC assets on a strict label-free sidecar.",
    )
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(None if argv is None else list(argv))
    try:
        result = run_transrac_official(
            sidecar_path=arguments.sidecar,
            commitment_path=arguments.commitment,
            video_root=arguments.video_root,
            source_archive_path=arguments.source_archive,
            source_root=arguments.source_root,
            backbone_path=arguments.backbone,
            checkpoint_path=arguments.checkpoint,
            repository_root=arguments.repository_root,
            video_ids=arguments.video_ids,
            output_path=arguments.output,
        )
        output = Path(arguments.output)
        receipt = output.with_suffix(".receipt.json")
        output_sha256 = _stable_file_digest(output).sha256
        receipt_sha256 = _stable_file_digest(receipt).sha256
    except (OSError, TransRACOfficialError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(output),
                "output_sha256": output_sha256,
                "receipt": str(receipt),
                "receipt_sha256": receipt_sha256,
                "selected_count": len(result.selected_video_ids),
                "failure_total": sum(
                    row.decode_status is not DecodeStatus.OK for row in result.predictions
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
