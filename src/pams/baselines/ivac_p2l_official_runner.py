"""Strict label-free runner for the pinned official IVAC-P2L assets.

The third-party source, checkpoints, and videos are mounted externally and
never vendored here. Targets are deliberately absent from this module and its
CLI; :mod:`pams.baselines.ivac_p2l_official_score` is the separate scoring
process.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import json
import math
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
from pams.baselines.transrac_official_runner import (
    AssetSpec,
    SourceTreeSpec,
    _remove_artifact_if_exact,
    _source_tree_digest,
    _write_json_exclusive,
)
from pams.metrics import round_count

OFFICIAL_SOURCE_REPOSITORY = "https://github.com/hwang-cs-ime/IVAC-P2L"
OFFICIAL_SOURCE_COMMIT = "0b1149e6958268ca5131d60ff66498c6e07f3b79"
OFFICIAL_SOURCE_ARCHIVE_URL = (
    "https://codeload.github.com/hwang-cs-ime/IVAC-P2L/tar.gz/"
    f"{OFFICIAL_SOURCE_COMMIT}"
)
OFFICIAL_SOURCE_ARCHIVE_BYTES = 722_651
OFFICIAL_SOURCE_ARCHIVE_SHA256 = (
    "9abaf839bf5f516741ed9f0b7c9a6dad5fe0327547e5563c22094f822d6e52b6"
)
OFFICIAL_SOURCE_TREE_FILES = 468
OFFICIAL_SOURCE_TREE_BYTES = 2_630_970
OFFICIAL_SOURCE_TREE_SHA256 = (
    "0c819eafa2194d77cb6e79736be5010c3839c0797f7e99eae6f48eb705f96823"
)
OFFICIAL_LICENSE_SHA256 = (
    "13c2f5fa5221c728cb1e1f5ec84f95382cc6155bd0e43ba63e69ce84bf9e83e7"
)
COMPATIBILITY_IMAGE_DIGEST = (
    "sha256:ea96abe0e7d17cf4720343e5867256aacd5209aca738087ec92f62b9fe31ff51"
)
SOURCE_MODULE_PREFIXES = ("models", "mmaction")
RUNNER_CODE_RELATIVE_PATH = "src/pams/baselines/ivac_p2l_official_runner.py"
GIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")

BACKBONE_URL = (
    "https://github.com/SwinTransformer/storage/releases/download/v1.0.4/"
    "swin_tiny_patch244_window877_kinetics400_1k.pth"
)
CHECKPOINT_URL = (
    "https://drive.google.com/file/d/"
    "1gFUhs-Kjacpy6wMxvIi0B4VnVlAlxnhP/view"
)
METHOD_ID = "ivac-p2l-official-modern-compat-repcounta"
CLASSIFICATION = (
    "IVAC-P2L official RepCount-A checkpoint / modern compatibility / "
    "local label-free evaluation"
)
PROTOCOL_ASSUMPTIONS = (
    "64-frame uniform sampling from official tools/video2npz.py",
    "NPZ test-path normalization (x-127.5)/127.5",
    "official scales [1,4,8] with cumulative scale-4 then scale-8 padding preserved",
    "no released executable UCFRep inference protocol exists at the pinned commit",
)


class IVACP2LOfficialError(RuntimeError):
    """Base class for IVAC-P2L integrity and execution failures."""


class IVACP2LSourceError(IVACP2LOfficialError):
    """The source archive, tree, or imported module origin is not frozen."""


class IVACP2LAssetError(IVACP2LOfficialError):
    """A required checkpoint has the wrong byte identity."""


class IVACP2LDependencyError(IVACP2LOfficialError):
    """The modern compatibility runtime cannot execute the official model."""


class IVACP2LDecodeError(IVACP2LOfficialError):
    """A video cannot be transformed through the frozen label-free path."""


SOURCE_ARCHIVE_SPEC = AssetSpec(
    name="IVAC-P2L-0b1149e.tar.gz",
    byte_count=OFFICIAL_SOURCE_ARCHIVE_BYTES,
    sha256=OFFICIAL_SOURCE_ARCHIVE_SHA256,
    role="official_source_archive",
    url=OFFICIAL_SOURCE_ARCHIVE_URL,
)
BACKBONE_SPEC = AssetSpec(
    name="swin_tiny_patch244_window877_kinetics400_1k.pth",
    byte_count=127_414_861,
    sha256="9950e3be6b0b3763f80575dfbbe7db7cbeb18345e32363b73e8d44952ef76acc",
    role="required_upstream_video_swin_backbone",
    url=BACKBONE_URL,
)
CHECKPOINT_SPEC = AssetSpec(
    name="Epoch-67_MAE_0.4022_OBO_0.3444.pt",
    byte_count=297_513_885,
    sha256="5f3be1c0b09ffc000cb9e9c41cc6e83fa4c69445eeaa678fda252569a15dafdd",
    role="official_repcount_a_ivac_p2l_checkpoint",
    url=CHECKPOINT_URL,
)
FROZEN_SOURCE_TREE = SourceTreeSpec(
    file_count=OFFICIAL_SOURCE_TREE_FILES,
    byte_count=OFFICIAL_SOURCE_TREE_BYTES,
    sha256=OFFICIAL_SOURCE_TREE_SHA256,
)


@dataclass(frozen=True, slots=True)
class VerifiedAsset:
    """One externally mounted asset verified by a full content digest."""

    spec: AssetSpec
    path: Path
    _digest: Any

    def assert_unchanged(self) -> None:
        try:
            observed = _stable_file_digest(self.path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise IVACP2LAssetError(
                f"unable to rehash {self.spec.role}: {exc}"
            ) from exc
        if observed != self._digest:
            raise IVACP2LAssetError(
                f"{self.spec.role} changed after its initial full verification"
            )

    def to_dict(self) -> dict[str, Any]:
        return self.spec.public_dict()


def _verify_asset_with_spec(path: str | Path, spec: AssetSpec) -> VerifiedAsset:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
        digest = _stable_file_digest(resolved)
    except (OSError, RuntimeError, ValueError) as exc:
        raise IVACP2LAssetError(f"unable to verify {spec.role}: {exc}") from exc
    if resolved.name != spec.name:
        raise IVACP2LAssetError(
            f"{spec.role} filename is {resolved.name!r}; expected {spec.name!r}"
        )
    if digest.byte_count != spec.byte_count or digest.sha256 != spec.sha256:
        raise IVACP2LAssetError(f"{spec.role} bytes/SHA-256 mismatch")
    return VerifiedAsset(spec=spec, path=resolved, _digest=digest)


def verify_official_assets(
    source_archive: str | Path,
    backbone: str | Path,
    checkpoint: str | Path,
) -> tuple[VerifiedAsset, VerifiedAsset, VerifiedAsset]:
    """Full-hash all three frozen external assets."""

    return (
        _verify_asset_with_spec(source_archive, SOURCE_ARCHIVE_SPEC),
        _verify_asset_with_spec(backbone, BACKBONE_SPEC),
        _verify_asset_with_spec(checkpoint, CHECKPOINT_SPEC),
    )


def _verify_source_tree_with_spec(
    source_root: str | Path,
    spec: SourceTreeSpec,
) -> Path:
    try:
        root = Path(source_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise IVACP2LSourceError(
            f"official source root cannot be resolved: {source_root}"
        ) from exc
    if not root.is_dir():
        raise IVACP2LSourceError(f"official source root is not a directory: {root}")
    try:
        observed = _source_tree_digest(root)
    except Exception as exc:
        raise IVACP2LSourceError(f"official source tree is unsafe: {exc}") from exc
    expected = (spec.file_count, spec.byte_count, spec.sha256)
    if observed != expected:
        raise IVACP2LSourceError(
            "extracted official source tree file count/bytes/SHA-256 mismatch"
        )
    return root


def verify_official_source_tree(source_root: str | Path) -> Path:
    return _verify_source_tree_with_spec(source_root, FROZEN_SOURCE_TREE)


@dataclass(frozen=True, slots=True)
class IVACP2LOfficialConfig:
    schema_version: int = 1
    source_commit: str = OFFICIAL_SOURCE_COMMIT
    num_frames: int = 64
    resize_height: int = 224
    resize_width: int = 224
    scales: tuple[int, ...] = (1, 4, 8)
    normalization: str = "repcount_npz_minus127.5_div127.5"
    preserve_cumulative_padding: bool = True
    torch_seed: int = 1
    gpu_memory_limit_mib: int = 8192
    rounding: str = "nearest_integer_half_up"
    compatibility: str = "python3.11-torch2.5.1-cu124-mmcv1.4.0"
    compatibility_image_digest: str = COMPATIBILITY_IMAGE_DIGEST

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.source_commit != OFFICIAL_SOURCE_COMMIT:
            raise ValueError("IVAC-P2L source/config schema is frozen")
        if (
            self.num_frames != 64
            or (self.resize_height, self.resize_width) != (224, 224)
            or self.scales != (1, 4, 8)
        ):
            raise ValueError("IVAC-P2L preprocessing/model schedule is frozen")
        if (
            self.normalization != "repcount_npz_minus127.5_div127.5"
            or not self.preserve_cumulative_padding
            or self.torch_seed != 1
        ):
            raise ValueError("IVAC-P2L protocol assumptions are frozen")
        if self.gpu_memory_limit_mib != 8192:
            raise ValueError("IVAC-P2L GPU memory cap is frozen")
        if self.rounding != "nearest_integer_half_up":
            raise ValueError("shared evaluator rounding is frozen")
        if self.compatibility_image_digest != COMPATIBILITY_IMAGE_DIGEST:
            raise ValueError("IVAC-P2L compatibility image digest is frozen")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_commit": self.source_commit,
            "num_frames": self.num_frames,
            "resize_height": self.resize_height,
            "resize_width": self.resize_width,
            "scales": list(self.scales),
            "normalization": self.normalization,
            "preserve_cumulative_padding": self.preserve_cumulative_padding,
            "torch_seed": self.torch_seed,
            "gpu_memory_limit_mib": self.gpu_memory_limit_mib,
            "rounding": self.rounding,
            "compatibility": self.compatibility,
            "compatibility_image_digest": self.compatibility_image_digest,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


FROZEN_IVAC_P2L_OFFICIAL_CONFIG = IVACP2LOfficialConfig()


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
        raise IVACP2LSourceError(
            "unable to audit IVAC-P2L runner Git checkout"
        ) from exc
    if GIT_SHA_PATTERN.fullmatch(revision) is None or status:
        raise IVACP2LSourceError(
            "strict IVAC-P2L prediction requires a clean full-SHA Git checkout"
        )
    return revision


def official_source_dict() -> dict[str, Any]:
    return {
        "repository": OFFICIAL_SOURCE_REPOSITORY,
        "commit": OFFICIAL_SOURCE_COMMIT,
        "archive": SOURCE_ARCHIVE_SPEC.public_dict(),
        "tree_sha256": OFFICIAL_SOURCE_TREE_SHA256,
        "tree_files": OFFICIAL_SOURCE_TREE_FILES,
        "tree_bytes": OFFICIAL_SOURCE_TREE_BYTES,
        "license_file_sha256": OFFICIAL_LICENSE_SHA256,
        "license_status": "source-mit-checkpoint-and-backbone-unspecified",
    }


def _is_source_owned_module(name: str) -> bool:
    return any(
        name == prefix or name.startswith(f"{prefix}.")
        for prefix in SOURCE_MODULE_PREFIXES
    )


def _reject_preloaded_source_modules() -> None:
    preloaded = sorted(name for name in sys.modules if _is_source_owned_module(name))
    if preloaded:
        raise IVACP2LSourceError(
            "official source module names were loaded before the frozen tree: "
            f"{preloaded}"
        )


def _assert_source_module_origins(source_root: Path) -> None:
    observed = {
        name: module
        for name, module in sys.modules.items()
        if _is_source_owned_module(name)
    }
    required = {"models.IVAC", "mmaction"}
    missing = sorted(required - set(observed))
    if missing:
        raise IVACP2LSourceError(
            f"required official source modules were not loaded: {missing}"
        )
    for name, module in sorted(observed.items()):
        origin = getattr(module, "__file__", None)
        if not isinstance(origin, str) or not origin:
            raise IVACP2LSourceError(
                f"official source module {name!r} has no auditable origin"
            )
        try:
            Path(origin).resolve(strict=True).relative_to(source_root)
        except (OSError, ValueError) as exc:
            raise IVACP2LSourceError(
                f"official source module {name!r} was imported outside "
                f"the frozen source tree: {origin}"
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
            raise ValueError("IVAC-P2L backend output contains non-finite values")
        if self.raw_count < 0 or self.decoded_frame_count < 1:
            raise ValueError("IVAC-P2L count/frame metadata is invalid")
        if self.sampled_frame_count != 64:
            raise ValueError("IVAC-P2L must sample exactly 64 frames")


class _OfficialBackend(Protocol):
    source_commit: str
    source_tree_sha256: str
    runtime_versions: Mapping[str, str]
    restore_audit: Mapping[str, Any]

    def predict(
        self,
        video_path: Path,
        config: IVACP2LOfficialConfig,
    ) -> BackendPrediction: ...


class ModernCompatBackend:
    """Thin label-free adapter around the unmodified official model."""

    source_commit = OFFICIAL_SOURCE_COMMIT
    source_tree_sha256 = OFFICIAL_SOURCE_TREE_SHA256

    def __init__(
        self,
        *,
        torch: Any,
        cv2: Any,
        np: Any,
        model: Any,
        device: Any,
        runtime_versions: Mapping[str, str],
        restore_audit: Mapping[str, Any],
    ) -> None:
        self._torch = torch
        self._cv2 = cv2
        self._np = np
        self._model = model
        self._device = device
        self.runtime_versions: Mapping[str, str] = dict(runtime_versions)
        self.restore_audit: Mapping[str, Any] = dict(restore_audit)

    @classmethod
    def from_assets(
        cls,
        source_root: Path,
        backbone: VerifiedAsset,
        checkpoint: VerifiedAsset,
        config: IVACP2LOfficialConfig,
    ) -> ModernCompatBackend:
        _reject_preloaded_source_modules()
        try:
            torch = importlib.import_module("torch")
            cv2 = importlib.import_module("cv2")
            np = importlib.import_module("numpy")
        except (ImportError, OSError) as exc:
            raise IVACP2LDependencyError(f"runtime dependency unavailable: {exc}") from exc
        if not bool(torch.cuda.is_available()):
            raise IVACP2LDependencyError("CUDA is required for official IVAC-P2L")
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
                model_module = importlib.import_module("models.IVAC")
                config_path = (
                    source_root
                    / "configs/recognition/swin/"
                    "swin_tiny_patch244_window877_kinetics400_1k.py"
                )
                model = model_module.IVAC_P2L(
                    config=str(config_path),
                    checkpoint=str(backbone.path),
                    num_frames=config.num_frames,
                    scales=list(config.scales),
                    OPEN=False,
                )
                backbone.assert_unchanged()
                model = torch.nn.DataParallel(model.to(device), device_ids=[0])
                saved = torch.load(
                    checkpoint.path,
                    map_location="cpu",
                    weights_only=True,
                )
                checkpoint.assert_unchanged()
                state_dict = (
                    saved.get("state_dict", saved) if isinstance(saved, dict) else None
                )
                if not isinstance(state_dict, dict):
                    raise ValueError("checkpoint state_dict is missing")
                incompatible = model.load_state_dict(state_dict, strict=False)
                _assert_source_module_origins(source_root)
        except IVACP2LOfficialError:
            raise
        except Exception as exc:
            raise IVACP2LDependencyError(
                f"official IVAC-P2L source/checkpoint could not be loaded: {exc}"
            ) from exc
        missing = list(incompatible.missing_keys)
        unexpected = list(incompatible.unexpected_keys)
        if len(state_dict) != 230 or missing or unexpected:
            raise IVACP2LAssetError(
                "official IVAC-P2L checkpoint did not restore all 230 keys exactly"
            )
        epoch = saved.get("epoch") if isinstance(saved, dict) else None
        if epoch != 67:
            raise IVACP2LAssetError(
                "official IVAC-P2L checkpoint payload epoch is not 67"
            )
        model.eval()
        return cls(
            torch=torch,
            cv2=cv2,
            np=np,
            model=model,
            device=device,
            runtime_versions={
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda_runtime": str(torch.version.cuda),
                "torchvision": importlib.metadata.version("torchvision"),
                "mmcv": importlib.metadata.version("mmcv"),
                "opencv": str(cv2.__version__),
                "numpy": str(np.__version__),
                "gpu_name": str(properties.name),
            },
            restore_audit={
                "checkpoint_epoch": 67,
                "state_dict_keys": len(state_dict),
                "loaded_key_count": len(state_dict) - len(unexpected),
                "missing_keys": missing,
                "unexpected_keys": unexpected,
                "strict_key_coverage": True,
            },
        )

    def _decode(
        self,
        video_path: Path,
        config: IVACP2LOfficialConfig,
    ) -> tuple[Any, int]:
        capture = self._cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise IVACP2LDecodeError("video decoder could not open input")
        frames: list[Any] = []
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                frame_rgb = self._cv2.cvtColor(
                    frame_bgr,
                    self._cv2.COLOR_BGR2RGB,
                )
                frames.append(
                    self._cv2.resize(
                        frame_rgb,
                        (config.resize_width, config.resize_height),
                    )
                )
        finally:
            capture.release()
        frame_count = len(frames)
        if frame_count == 0:
            raise IVACP2LDecodeError("video decoder returned zero frames")
        if config.num_frames <= frame_count:
            selected = [
                frames[index * frame_count // config.num_frames - 1]
                for index in range(1, config.num_frames + 1)
            ]
        else:
            selected = list(frames)
            selected.extend([frames[-1]] * (config.num_frames - frame_count))
        # Exact composition of official video2npz.py and RepCountA_Loader.py.
        array = self._np.asarray(selected).transpose(0, 3, 2, 1)
        tensor = self._torch.as_tensor(array, dtype=self._torch.float32)
        tensor.sub_(127.5).div_(127.5)
        return tensor.transpose(0, 1).unsqueeze(0), frame_count

    def predict(
        self,
        video_path: Path,
        config: IVACP2LOfficialConfig,
    ) -> BackendPrediction:
        tensor, frame_count = self._decode(video_path, config)
        if tuple(tensor.shape) != (1, 3, 64, 224, 224):
            raise IVACP2LDecodeError(
                f"label-free adapter returned unexpected shape {tuple(tensor.shape)}"
            )
        if not bool(self._torch.isfinite(tensor).all().item()):
            raise IVACP2LDecodeError("label-free adapter returned NaN/Inf")
        self._torch.cuda.reset_peak_memory_stats(self._device)
        with self._torch.no_grad():
            output, similarity, features = self._model(tensor.to(self._device))
        self._torch.cuda.synchronize(self._device)
        if not all(
            bool(self._torch.isfinite(value).all().item())
            for value in (output, similarity, features)
        ):
            raise ValueError("official IVAC-P2L output contains NaN/Inf")
        return BackendPrediction(
            raw_count=float(self._torch.sum(output, dim=1).item()),
            decoded_frame_count=frame_count,
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
class IVACP2LVideoPrediction:
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
) -> IVACP2LVideoPrediction:
    return IVACP2LVideoPrediction(
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
class IVACP2LOfficialRunResult:
    inputs: VerifiedLabelFreeInputs
    source_archive: VerifiedAsset
    backbone: VerifiedAsset
    checkpoint: VerifiedAsset
    config: IVACP2LOfficialConfig
    runtime_versions: Mapping[str, str]
    restore_audit: Mapping[str, Any]
    runner_git_sha: str
    runner_code_sha256: str
    selected_video_ids: tuple[str, ...]
    predictions: tuple[IVACP2LVideoPrediction, ...]

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
            "eligible_for_original_pams_ivac_p2l_cell": False,
            "labels_loaded": False,
            "scoring_performed": False,
            "runner_provenance": {
                "source_git_sha": self.runner_git_sha,
                "runner_code_sha256": self.runner_code_sha256,
                "compatibility_image_digest": COMPATIBILITY_IMAGE_DIGEST,
            },
            "source": official_source_dict(),
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
            "assets": {
                "backbone": self.backbone.to_dict(),
                "checkpoint": self.checkpoint.to_dict(),
                "third_party_assets_redistributed": False,
            },
            "config": {
                "sha256": self.config.fingerprint,
                "values": self.config.to_dict(),
            },
            "protocol_assumptions": list(PROTOCOL_ASSUMPTIONS),
            "runtime_versions": dict(sorted(self.runtime_versions.items())),
            "restore_audit": dict(self.restore_audit),
            "decode_ok_total": len(self.predictions) - len(failures),
            "failure_total": len(failures),
            "failures": failures,
            "predictions": [row.to_dict() for row in self.predictions],
        }


def _assert_inputs_full_unchanged(inputs: VerifiedLabelFreeInputs) -> None:
    observed_sidecar = _stable_file_digest(inputs._sidecar_path)
    observed_commitment = _stable_file_digest(inputs._commitment_path)
    if observed_sidecar.sha256 != inputs.sidecar_sha256:
        raise IVACP2LOfficialError("label-free sidecar changed during prediction")
    if observed_commitment.sha256 != inputs.commitment_sha256:
        raise IVACP2LOfficialError("label-free commitment changed during prediction")
    inputs.assert_unchanged()


def _run_verified_backend(
    inputs: VerifiedLabelFreeInputs,
    source_archive: VerifiedAsset,
    backbone: VerifiedAsset,
    checkpoint: VerifiedAsset,
    backend: _OfficialBackend,
    *,
    config: IVACP2LOfficialConfig = FROZEN_IVAC_P2L_OFFICIAL_CONFIG,
    video_ids: Sequence[str] | None = None,
    runner_git_sha: str,
    runner_code_sha256: str,
) -> IVACP2LOfficialRunResult:
    if (
        backend.source_commit != OFFICIAL_SOURCE_COMMIT
        or backend.source_tree_sha256 != OFFICIAL_SOURCE_TREE_SHA256
    ):
        raise IVACP2LSourceError("backend source identity is not frozen")
    selected = _select_video_records(inputs, video_ids)
    predictions: list[IVACP2LVideoPrediction] = []
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
        except IVACP2LDecodeError as exc:
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
            observed_after = _stable_file_digest(path)
            if observed_after != digest:
                raise IVACP2LOfficialError(
                    "source-video bytes/SHA-256 changed during prediction"
                )
            _assert_file_identity(path, digest.identity, role="source video")
        except (OSError, RuntimeError, ValueError) as exc:
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
            IVACP2LVideoPrediction(
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
    _assert_inputs_full_unchanged(inputs)
    for asset in (source_archive, backbone, checkpoint):
        asset.assert_unchanged()
    return IVACP2LOfficialRunResult(
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


def write_ivac_p2l_result_exclusive(
    output_path: str | Path,
    result: IVACP2LOfficialRunResult,
) -> tuple[Path, str, Path, str]:
    destination = Path(output_path)
    receipt_path = destination.with_suffix(".receipt.json")
    durable_mkdir(destination.parent)
    collisions = [str(path) for path in (destination, receipt_path) if path.exists()]
    if collisions:
        raise FileExistsError(
            f"refusing to overwrite IVAC-P2L artifacts: {collisions}"
        )
    prediction_sha256 = _write_json_exclusive(destination, result.to_dict())
    receipt = {
        "schema_version": 1,
        "artifact_type": "ivac_p2l_official_prediction_receipt",
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
        "source_archive_sha256": SOURCE_ARCHIVE_SPEC.sha256,
        "source_tree_sha256": OFFICIAL_SOURCE_TREE_SHA256,
        "backbone_sha256": BACKBONE_SPEC.sha256,
        "checkpoint_sha256": CHECKPOINT_SPEC.sha256,
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
                "IVAC-P2L receipt write failed and the exact orphan "
                "prediction could not be removed safely"
            ) from exc
        raise
    fsync_directory(destination.parent)
    return destination, prediction_sha256, receipt_path, receipt_sha256


def run_ivac_p2l_official(
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
) -> IVACP2LOfficialRunResult:
    """Run IVAC-P2L without making any target data reachable."""

    inputs = _load_label_free_inputs(
        sidecar_path,
        commitment_path,
        video_root,
        require_exact_membership=True,
    )
    if output_path is not None:
        output = Path(output_path)
        if output.exists() or output.with_suffix(".receipt.json").exists():
            raise FileExistsError(f"refusing to overwrite IVAC-P2L output: {output}")
    repository = Path(repository_root).expanduser().resolve(strict=True)
    runner_git_sha = _clean_runner_git_revision(repository)
    runner_code = (repository / RUNNER_CODE_RELATIVE_PATH).resolve(strict=True)
    if runner_code != Path(__file__).resolve(strict=True):
        raise IVACP2LSourceError(
            "executed IVAC-P2L runner is outside repository_root"
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
        FROZEN_IVAC_P2L_OFFICIAL_CONFIG,
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
    expected_tree = (
        OFFICIAL_SOURCE_TREE_FILES,
        OFFICIAL_SOURCE_TREE_BYTES,
        OFFICIAL_SOURCE_TREE_SHA256,
    )
    if _source_tree_digest(verified_source_root) != expected_tree:
        raise IVACP2LSourceError("official source tree changed during inference")
    if _stable_file_digest(runner_code) != runner_code_digest:
        raise IVACP2LSourceError("IVAC-P2L runner code changed during inference")
    if _clean_runner_git_revision(repository) != runner_git_sha:
        raise IVACP2LSourceError(
            "IVAC-P2L runner Git revision changed during inference"
        )
    if output_path is not None:
        write_ivac_p2l_result_exclusive(output_path, result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pams.baselines.ivac_p2l_official_runner",
        description="Run pinned official IVAC-P2L assets on a label-free sidecar.",
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
        result = run_ivac_p2l_official(
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
    except (OSError, IVACP2LOfficialError, ValueError) as exc:
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
                    row.decode_status is not DecodeStatus.OK
                    for row in result.predictions
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
