"""Label-free wrapper for the released PoseRAC-v1 checkpoint.

This module is intentionally separate from the later ICONIP'24 method.  It
reconstructs only the 2023 PoseRAC-v1 network whose exact state dictionary is
published in ``MiracleDance/PoseRAC``.

The upstream evaluator is not called: it tries every output channel and uses
the evaluation count to retain the least erroneous one.  Here a channel is
selected without labels by the largest temporal dynamic range after the
released exponential smoothing step.  That rule is an independently inferred
compatibility choice, not an author-disclosed PoseRAC result.

The current adapter consumes this project's schema-v2 MediaPipe pose caches.
Those caches contain 256 uniformly resampled frames and therefore do not match
the upstream original-frame preprocessing.  Every artifact labels that
difference explicitly; it cannot occupy the original PoseRAC-v1 paper cell.
Targets and metric code are absent from this module.  Scoring must occur in a
separate process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from pams.baselines.repnet_official_runner import (
    VerifiedLabelFreeInputs,
    _assert_file_identity,
    _load_label_free_inputs,
    _select_video_records,
    _stable_file_digest,
    durable_mkdir,
    fsync_directory,
    sha256_json,
)
from pams.data import (
    PoseCacheEntryReceipt,
    PoseCacheSetSnapshot,
    PoseSequence,
    load_pose_cache_with_receipt,
    pose_cache_path,
)

OFFICIAL_SOURCE_REPOSITORY = "https://github.com/MiracleDance/PoseRAC"
OFFICIAL_SOURCE_COMMIT = "469590b611bde3595eaf163b517263da634e2096"
OFFICIAL_SOURCE_ARCHIVE_URL = (
    "https://codeload.github.com/MiracleDance/PoseRAC/tar.gz/"
    f"{OFFICIAL_SOURCE_COMMIT}"
)
OFFICIAL_SOURCE_ARCHIVE_BYTES = 62_833_091
OFFICIAL_SOURCE_ARCHIVE_SHA256 = (
    "0150eacf56ec033a77c24d61bebc0894134547677deda41ba5681eea887c21d0"
)
OFFICIAL_SOURCE_TREE_FILES = 22
OFFICIAL_SOURCE_TREE_BYTES = 66_442_868
OFFICIAL_SOURCE_TREE_SHA256 = (
    "7b3feec127f7a81a44d7ed94cb6d54e515421213cc40392aa89fa44311912e90"
)
OFFICIAL_MODEL_PY_SHA256 = (
    "6dc334f0e5286a4e1d89b8cf04633484b218338310e8ef2989e135282ddcb05e"
)
OFFICIAL_EVAL_PY_SHA256 = (
    "ffe91246d32f84ed7df8ff24ab7c77ea9a306770155277a8d4cfd3a16585f146"
)
OFFICIAL_PRE_TEST_PY_SHA256 = (
    "49b8d70491619db1d169685d48a913e19903bf013b380cd1584441bb06e9b2db"
)
OFFICIAL_ACTION_CSV_SHA256 = (
    "bb0386f361f290221949e4cece7259157ebed0c5855219038b948b24133decb0"
)
OFFICIAL_CONFIG_SHA256 = (
    "3110cb9c651adeaa0df82e2a4c4ecdc99079a1d4f9ba7e173a8de0f79d0fb711"
)
OFFICIAL_LICENSE_SHA256 = (
    "f9c3dc103d80706f49925e6fa0b2ba7a15a7710104ba5fff82a150fb1711aa9e"
)
OFFICIAL_CHECKPOINT_BYTES = 10_772_679
OFFICIAL_CHECKPOINT_SHA256 = (
    "21afc8334333e5f5751f0e5d765376a778415285c0ac3078d8358ccb83b2b34a"
)
COMPATIBILITY_IMAGE_DIGEST = (
    "sha256:022103f69a42ef3e88ee43ddcfbd706dabfdeec6247c257d1d01fda2c0ea2f91"
)
RUNNER_CODE_RELATIVE_PATH = "src/pams/baselines/poserac_v1_official_runner.py"

METHOD_ID = "poserac-v1-official-checkpoint-pams-cache-dynamic-range"
CLASSIFICATION = (
    "PoseRAC-v1 official checkpoint / PAMS 256-frame pose-cache / "
    "inferred oracle-free dynamic-range channel"
)
ACTION_NAMES = (
    "front_raise",
    "pull_up",
    "squat",
    "bench_pressing",
    "jump_jack",
    "situp",
    "push_up",
    "pommelhorse",
)


class PoseRACV1OfficialError(RuntimeError):
    """Base error for frozen PoseRAC-v1 assets or execution."""


class PoseRACV1SourceError(PoseRACV1OfficialError):
    """The supplied official source archive/tree is not the frozen revision."""


class PoseRACV1CheckpointError(PoseRACV1OfficialError):
    """The supplied state dictionary is not the released checkpoint."""


class PoseRACV1PoseCacheError(PoseRACV1OfficialError):
    """A pose cache is incompatible with the declared diagnostic protocol."""


@dataclass(frozen=True, slots=True)
class AssetSpec:
    """Frozen byte identity for one externally mounted official asset."""

    name: str
    byte_count: int
    sha256: str
    role: str
    url: str

    def __post_init__(self) -> None:
        if not self.name or Path(self.name).name != self.name:
            raise ValueError("asset name must be one plain filename")
        if self.byte_count < 1:
            raise ValueError("asset byte_count must be positive")
        if len(self.sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.sha256
        ):
            raise ValueError("asset sha256 must be lowercase hexadecimal")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "bytes": self.byte_count,
            "sha256": self.sha256,
            "role": self.role,
            "url": self.url,
        }


SOURCE_ARCHIVE_SPEC = AssetSpec(
    name="source.tar.gz",
    byte_count=OFFICIAL_SOURCE_ARCHIVE_BYTES,
    sha256=OFFICIAL_SOURCE_ARCHIVE_SHA256,
    role="official_source_archive",
    url=OFFICIAL_SOURCE_ARCHIVE_URL,
)
CHECKPOINT_SPEC = AssetSpec(
    name="best_weights_PoseRAC.pth",
    byte_count=OFFICIAL_CHECKPOINT_BYTES,
    sha256=OFFICIAL_CHECKPOINT_SHA256,
    role="official_poserac_v1_state_dict",
    url=(
        "https://github.com/MiracleDance/PoseRAC/blob/"
        f"{OFFICIAL_SOURCE_COMMIT}/best_weights_PoseRAC.pth"
    ),
)


@dataclass(frozen=True, slots=True)
class VerifiedAsset:
    spec: AssetSpec
    path: Path
    _identity: Any

    def assert_unchanged(self) -> None:
        try:
            digest = _stable_file_digest(self.path)
        except (OSError, RuntimeError, ValueError) as exc:
            raise PoseRACV1OfficialError(
                f"unable to re-verify {self.spec.role}: {exc}"
            ) from exc
        if (
            digest.byte_count != self.spec.byte_count
            or digest.sha256 != self.spec.sha256
        ):
            raise PoseRACV1OfficialError(
                f"{self.spec.role} bytes/SHA-256 changed after verification"
            )
        _assert_file_identity(self.path, self._identity, role=self.spec.role)

    def to_dict(self) -> dict[str, Any]:
        return self.spec.to_dict()


def _verify_asset_with_spec(path: str | Path, spec: AssetSpec) -> VerifiedAsset:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
        digest = _stable_file_digest(resolved)
    except (OSError, RuntimeError, ValueError) as exc:
        raise PoseRACV1OfficialError(f"unable to verify {spec.role}: {exc}") from exc
    if resolved.name != spec.name:
        raise PoseRACV1OfficialError(
            f"{spec.role} filename is {resolved.name!r}; expected {spec.name!r}"
        )
    if digest.byte_count != spec.byte_count:
        raise PoseRACV1OfficialError(
            f"{spec.role} has {digest.byte_count} bytes; expected {spec.byte_count}"
        )
    if digest.sha256 != spec.sha256:
        raise PoseRACV1OfficialError(f"{spec.role} SHA-256 mismatch")
    return VerifiedAsset(spec=spec, path=resolved, _identity=digest.identity)


def _source_tree_digest(root: Path) -> tuple[int, int, str]:
    """Hash all regular files using canonical relative paths and bytes."""

    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix == ".pyc":
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise PoseRACV1SourceError(f"source tree contains symlink: {relative}")
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


def verify_official_source_tree(source_root: str | Path) -> Path:
    """Require the complete extracted source tree at the pinned commit."""

    try:
        root = Path(source_root).expanduser().resolve(strict=True)
    except OSError as exc:
        raise PoseRACV1SourceError(
            f"official source root cannot be resolved: {source_root}"
        ) from exc
    if not root.is_dir():
        raise PoseRACV1SourceError(f"official source root is not a directory: {root}")
    observed = _source_tree_digest(root)
    expected = (
        OFFICIAL_SOURCE_TREE_FILES,
        OFFICIAL_SOURCE_TREE_BYTES,
        OFFICIAL_SOURCE_TREE_SHA256,
    )
    if observed != expected:
        raise PoseRACV1SourceError(
            "official source tree file count/bytes/SHA-256 mismatch"
        )
    checkpoint = root / CHECKPOINT_SPEC.name
    if checkpoint.resolve(strict=True) != (root / CHECKPOINT_SPEC.name).resolve():
        raise PoseRACV1SourceError("checkpoint path resolution is inconsistent")
    return root


@dataclass(frozen=True, slots=True)
class PoseRACV1OfficialConfig:
    """Frozen oracle-free diagnostic configuration."""

    schema_version: int = 1
    dim: int = 99
    heads: int = 9
    encoder_layers: int = 6
    feedforward_dim: int = 2048
    classes: int = 8
    enter_threshold: float = 0.78
    exit_threshold: float = 0.4
    momentum: float = 0.4
    expected_frames: int = 256
    expected_pose_model: str = "mediapipe-pose-0.10.14"
    channel_selection: str = "smoothed_dynamic_range_max_lowest_index"
    zero_span_policy: str = "stable_zero"
    compatibility_image_digest: str = COMPATIBILITY_IMAGE_DIGEST

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("only PoseRAC-v1 config schema_version=1 is supported")
        if (
            self.dim,
            self.heads,
            self.encoder_layers,
            self.feedforward_dim,
            self.classes,
        ) != (99, 9, 6, 2048, 8):
            raise ValueError("released PoseRAC-v1 architecture is frozen")
        if (
            self.enter_threshold,
            self.exit_threshold,
            self.momentum,
        ) != (0.78, 0.4, 0.4):
            raise ValueError("released PoseRAC-v1 trigger parameters are frozen")
        if self.expected_frames != 256:
            raise ValueError("current pose-cache diagnostic requires exactly 256 frames")
        if self.expected_pose_model != "mediapipe-pose-0.10.14":
            raise ValueError("current pose-cache model identity is frozen")
        if self.channel_selection != "smoothed_dynamic_range_max_lowest_index":
            raise ValueError("oracle-free channel selection is frozen")
        if self.zero_span_policy != "stable_zero":
            raise ValueError("zero-span policy is frozen")
        if self.compatibility_image_digest != COMPATIBILITY_IMAGE_DIGEST:
            raise ValueError("compatibility image digest is frozen")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dim": self.dim,
            "heads": self.heads,
            "encoder_layers": self.encoder_layers,
            "feedforward_dim": self.feedforward_dim,
            "classes": self.classes,
            "enter_threshold": self.enter_threshold,
            "exit_threshold": self.exit_threshold,
            "momentum": self.momentum,
            "expected_frames": self.expected_frames,
            "expected_pose_model": self.expected_pose_model,
            "channel_selection": self.channel_selection,
            "channel_selection_provenance": "inferred",
            "zero_span_policy": self.zero_span_policy,
            "cache_preprocessing": (
                "schema-v2 joint-minmax then uniform-resample-to-256; "
                "re-normalized per xyz axis before checkpoint"
            ),
            "official_preprocessing_parity": False,
            "compatibility_image_digest": self.compatibility_image_digest,
        }

    @property
    def fingerprint(self) -> str:
        return sha256_json(self.to_dict())


FROZEN_POSERAC_V1_CONFIG = PoseRACV1OfficialConfig()


class ExactPoseRACV1(nn.Module):
    """Minimal exact network needed by the published state dictionary."""

    def __init__(self, config: PoseRACV1OfficialConfig) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim,
            nhead=config.heads,
            dim_feedforward=config.feedforward_dim,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.encoder_layers,
        )
        self.fc1 = nn.Linear(config.dim, config.classes)
        self.dim = config.dim

    def forward(self, inputs: Tensor) -> Tensor:
        sequence = inputs.view(-1, 1, self.dim)
        encoded = self.transformer_encoder(sequence)
        return self.fc1(encoded.view(-1, self.dim))


def official_axis_normalize(
    xyz: NDArray[np.float32],
    valid_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Approximate upstream preprocessing from schema-v2 normalized caches.

    The upstream code normalizes x, y, and z independently over the 33 joints
    in each valid original frame.  Reapplying that transform to the current
    caches recovers the same coordinate convention, but cannot undo temporal
    interpolation to 256 frames.
    """

    coordinates = np.asarray(xyz, dtype=np.float32)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    if coordinates.ndim != 3 or coordinates.shape[1:] != (33, 3):
        raise ValueError("xyz must have shape [frames, 33, 3]")
    if mask.shape != (coordinates.shape[0],):
        raise ValueError("valid_mask must have shape [frames]")
    if not np.isfinite(coordinates[mask]).all():
        raise ValueError("valid pose coordinates must be finite")
    output = np.zeros_like(coordinates)
    if np.any(mask):
        valid = coordinates[mask]
        minimum = valid.min(axis=1, keepdims=True)
        span = valid.max(axis=1, keepdims=True) - minimum
        output[mask] = np.divide(
            valid - minimum,
            span,
            out=np.zeros_like(valid),
            where=span > 1e-8,
        )
    return output


def smooth_scores(
    probabilities: NDArray[np.float32],
    *,
    momentum: float,
) -> NDArray[np.float32]:
    """Apply the released recursive smoothing, initialized at 0.5."""

    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != len(ACTION_NAMES):
        raise ValueError("probabilities must have shape [frames, 8]")
    if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must be in [0, 1)")
    result = np.empty_like(values)
    previous = np.full(values.shape[1], 0.5, dtype=np.float32)
    for index, row in enumerate(values):
        previous = row * (1.0 - momentum) + previous * momentum
        result[index] = previous
    return result


def count_trigger_cycles(
    scores: NDArray[np.float32],
    *,
    enter_threshold: float,
    exit_threshold: float,
) -> int:
    """Independent exact translation of the upstream two-trigger state machine."""

    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("scores must be a non-empty finite vector")

    high_entered = False
    reverse_entered = False
    initial_pose: str | None = None
    current_pose: str | None = None
    count = 0
    for value in values:
        high_triggered = False
        if not high_entered:
            high_entered = bool(value > enter_threshold)
        elif value < exit_threshold:
            high_entered = False
            high_triggered = True

        reverse = 1.0 - float(value)
        reverse_triggered = False
        if not reverse_entered:
            reverse_entered = bool(reverse > enter_threshold)
        elif reverse < exit_threshold:
            reverse_entered = False
            reverse_triggered = True

        if initial_pose is None:
            if high_triggered:
                initial_pose = "salient1"
            elif reverse_triggered:
                initial_pose = "salient2"
        if initial_pose == "salient1":
            if current_pose == "salient1" and reverse_triggered:
                count += 1
        else:
            if current_pose == "salient2" and high_triggered:
                count += 1
        if high_triggered:
            current_pose = "salient1"
        elif reverse_triggered:
            current_pose = "salient2"
    return count


def select_dynamic_range_channel(
    smoothed_scores: NDArray[np.float32],
) -> tuple[int, tuple[float, ...]]:
    """Select the first maximum-range output channel without label information."""

    scores = np.asarray(smoothed_scores, dtype=np.float32)
    if scores.ndim != 2 or scores.shape[1] != len(ACTION_NAMES):
        raise ValueError("smoothed_scores must have shape [frames, 8]")
    ranges = np.ptp(scores, axis=0)
    if not np.isfinite(ranges).all():
        raise ValueError("smoothed score ranges must be finite")
    selected = int(np.argmax(ranges))
    return selected, tuple(float(value) for value in ranges)


@dataclass(frozen=True, slots=True)
class BackendPrediction:
    count: int
    selected_channel: int
    selected_action: str
    channel_counts: tuple[int, ...]
    channel_dynamic_ranges: tuple[float, ...]
    probability_min: float
    probability_max: float
    frames: int
    valid_frames: int

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("count must be non-negative")
        if self.selected_channel not in range(len(ACTION_NAMES)):
            raise ValueError("selected_channel is outside [0, 7]")
        if self.selected_action != ACTION_NAMES[self.selected_channel]:
            raise ValueError("selected_action does not match selected_channel")
        if len(self.channel_counts) != len(ACTION_NAMES) or any(
            value < 0 for value in self.channel_counts
        ):
            raise ValueError("channel_counts must contain eight non-negative values")
        if len(self.channel_dynamic_ranges) != len(ACTION_NAMES) or any(
            not math.isfinite(value) or value < 0.0
            for value in self.channel_dynamic_ranges
        ):
            raise ValueError("channel_dynamic_ranges must contain eight finite values")
        if not 0.0 <= self.probability_min <= self.probability_max <= 1.0:
            raise ValueError("probability bounds must lie in [0, 1]")
        if self.frames < 1 or not 0 <= self.valid_frames <= self.frames:
            raise ValueError("frame counts are invalid")


class _Backend(Protocol):
    source_commit: str
    source_tree_sha256: str

    @property
    def runtime_versions(self) -> Mapping[str, str]:
        ...

    @property
    def restore_audit(self) -> Mapping[str, Any]:
        ...

    def predict(
        self,
        sequence: PoseSequence,
        config: PoseRACV1OfficialConfig,
    ) -> BackendPrediction:
        ...


class ExactCheckpointBackend:
    """Exact architecture plus strict official state-dictionary restore."""

    source_commit = OFFICIAL_SOURCE_COMMIT
    source_tree_sha256 = OFFICIAL_SOURCE_TREE_SHA256

    def __init__(
        self,
        model: ExactPoseRACV1,
        device: torch.device,
        restore_audit: Mapping[str, Any],
    ) -> None:
        self._model = model
        self._device = device
        self._restore_audit = dict(restore_audit)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: VerifiedAsset,
        config: PoseRACV1OfficialConfig,
        *,
        device: str = "cuda",
    ) -> ExactCheckpointBackend:
        checkpoint.assert_unchanged()
        selected_device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        state = torch.load(
            checkpoint.path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(state, Mapping) or not all(
            isinstance(key, str) and isinstance(value, Tensor)
            for key, value in state.items()
        ):
            raise PoseRACV1CheckpointError(
                "official checkpoint is not a plain tensor state dictionary"
            )
        model = ExactPoseRACV1(config)
        model_keys = tuple(model.state_dict())
        checkpoint_keys = tuple(state)
        missing = sorted(set(model_keys) - set(checkpoint_keys))
        unexpected = sorted(set(checkpoint_keys) - set(model_keys))
        if missing or unexpected:
            raise PoseRACV1CheckpointError(
                f"checkpoint key coverage mismatch: missing={missing}, unexpected={unexpected}"
            )
        try:
            incompatible = model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise PoseRACV1CheckpointError(
                f"checkpoint tensor shape/dtype restore failed: {exc}"
            ) from exc
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise PoseRACV1CheckpointError("strict state-dictionary coverage failed")
        model.eval().to(selected_device)
        audit = {
            "checkpoint_format": "plain_ordered_tensor_state_dict",
            "checkpoint_key_count": len(checkpoint_keys),
            "model_key_count": len(model_keys),
            "loaded_key_count": len(model_keys),
            "missing_keys": [],
            "unexpected_keys": [],
            "strict_key_coverage": True,
            "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "fc1_weight_shape": list(model.fc1.weight.shape),
        }
        return cls(model, selected_device, audit)

    @property
    def runtime_versions(self) -> Mapping[str, str]:
        gpu_name = (
            torch.cuda.get_device_name(self._device)
            if self._device.type == "cuda"
            else "cpu"
        )
        return {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_runtime": str(torch.version.cuda),
            "device": str(self._device),
            "gpu_name": gpu_name,
        }

    @property
    def restore_audit(self) -> Mapping[str, Any]:
        return dict(self._restore_audit)

    def predict(
        self,
        sequence: PoseSequence,
        config: PoseRACV1OfficialConfig,
    ) -> BackendPrediction:
        if sequence.num_frames != config.expected_frames:
            raise PoseRACV1PoseCacheError(
                f"expected {config.expected_frames} cached frames, got {sequence.num_frames}"
            )
        normalized = official_axis_normalize(sequence.xyz, sequence.valid_mask)
        inputs = torch.from_numpy(normalized.reshape(sequence.num_frames, config.dim))
        with torch.inference_mode():
            probabilities = torch.sigmoid(
                self._model(inputs.to(self._device))
            ).detach().cpu().numpy().astype(np.float32, copy=False)
        smoothed = smooth_scores(probabilities, momentum=config.momentum)
        selected, ranges = select_dynamic_range_channel(smoothed)
        counts = tuple(
            count_trigger_cycles(
                smoothed[:, channel],
                enter_threshold=config.enter_threshold,
                exit_threshold=config.exit_threshold,
            )
            for channel in range(config.classes)
        )
        return BackendPrediction(
            count=counts[selected],
            selected_channel=selected,
            selected_action=ACTION_NAMES[selected],
            channel_counts=counts,
            channel_dynamic_ranges=ranges,
            probability_min=float(probabilities.min()),
            probability_max=float(probabilities.max()),
            frames=sequence.num_frames,
            valid_frames=int(np.sum(sequence.valid_mask)),
        )


@dataclass(frozen=True, slots=True)
class PoseRACV1VideoPrediction:
    video_id: str
    cache_sha256: str
    cache_bytes: int
    count: int
    selected_channel: int
    selected_action: str
    channel_counts: tuple[int, ...]
    channel_dynamic_ranges: tuple[float, ...]
    probability_min: float
    probability_max: float
    frames: int
    valid_frames: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "pose_cache_sha256": self.cache_sha256,
            "pose_cache_bytes": self.cache_bytes,
            "raw_count": float(self.count),
            "rounded_count": self.count,
            "selected_channel": self.selected_channel,
            "selected_action": self.selected_action,
            "channel_selection_uses_count_label": False,
            "channel_counts": list(self.channel_counts),
            "channel_dynamic_ranges": list(self.channel_dynamic_ranges),
            "probability_min": self.probability_min,
            "probability_max": self.probability_max,
            "frames": self.frames,
            "valid_frames": self.valid_frames,
        }


@dataclass(frozen=True, slots=True)
class PoseRACV1OfficialRunResult:
    inputs: VerifiedLabelFreeInputs
    source_archive: VerifiedAsset
    checkpoint: VerifiedAsset
    config: PoseRACV1OfficialConfig
    pose_snapshot: PoseCacheSetSnapshot
    runtime_versions: Mapping[str, str]
    restore_audit: Mapping[str, Any]
    runner_git_sha: str
    runner_code_sha256: str
    selected_video_ids: tuple[str, ...]
    predictions: tuple[PoseRACV1VideoPrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method_id": METHOD_ID,
            "classification": CLASSIFICATION,
            "method_version": "PoseRAC-v1-2023",
            "distinct_from": "PoseRAC-ICONIP24",
            "eligible_for_original_poserac_v1_cell": False,
            "eligible_for_original_pams_poserac_cell": False,
            "ineligibility_reasons": [
                "upstream evaluation GT-count channel oracle is disabled",
                "channel selection is independently inferred",
                "PAMS caches are uniformly resampled to 256 rather than original-frame poses",
                "standard UCFRep-526 dev is not the released UCFRep-pose-110 test protocol",
            ],
            "labels_loaded": False,
            "scoring_performed": False,
            "ground_truth_count_channel_oracle": False,
            "runner_provenance": {
                "source_git_sha": self.runner_git_sha,
                "runner_code_sha256": self.runner_code_sha256,
                "compatibility_image_digest": COMPATIBILITY_IMAGE_DIGEST,
            },
            "official_source": {
                "repository": OFFICIAL_SOURCE_REPOSITORY,
                "commit": OFFICIAL_SOURCE_COMMIT,
                "archive": self.source_archive.to_dict(),
                "tree_files": OFFICIAL_SOURCE_TREE_FILES,
                "tree_bytes": OFFICIAL_SOURCE_TREE_BYTES,
                "tree_sha256": OFFICIAL_SOURCE_TREE_SHA256,
                "model_py_sha256": OFFICIAL_MODEL_PY_SHA256,
                "eval_py_sha256": OFFICIAL_EVAL_PY_SHA256,
                "pre_test_py_sha256": OFFICIAL_PRE_TEST_PY_SHA256,
                "all_action_csv_sha256": OFFICIAL_ACTION_CSV_SHA256,
                "config_sha256": OFFICIAL_CONFIG_SHA256,
                "license": "MIT",
                "license_sha256": OFFICIAL_LICENSE_SHA256,
            },
            "input": {
                "protocol": self.inputs.manifest.protocol,
                "split": self.inputs.manifest.split,
                "full_sidecar_sample_count": len(self.inputs.manifest.records),
                "sidecar_sha256": self.inputs.sidecar_sha256,
                "commitment_sha256": self.inputs.commitment_sha256,
                "sidecar_fingerprint": self.inputs.manifest.fingerprint,
                "identity_sha256": self.inputs.identity_sha256,
                "pose_cache_snapshot": self.pose_snapshot.to_dict(),
            },
            "selection": {
                "selected_count": len(self.selected_video_ids),
                "selected_video_ids": list(self.selected_video_ids),
                "selected_video_ids_sha256": sha256_json(list(self.selected_video_ids)),
                "rule": self.config.channel_selection,
                "rule_provenance": "inferred",
            },
            "assets": {
                "checkpoint": self.checkpoint.to_dict(),
                "third_party_assets_redistributed": False,
            },
            "config": {
                "sha256": self.config.fingerprint,
                "values": self.config.to_dict(),
            },
            "runtime_versions": dict(sorted(self.runtime_versions.items())),
            "restore_audit": dict(self.restore_audit),
            "prediction_total": len(self.predictions),
            "failure_total": 0,
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


def _clean_runner_git_revision(repository_root: Path) -> str:
    try:
        sha = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "-C",
                str(repository_root),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PoseRACV1SourceError("unable to audit runner Git revision") from exc
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise PoseRACV1SourceError("runner Git revision is not a full lowercase SHA")
    if status:
        raise PoseRACV1SourceError("runner repository must be clean")
    return sha


def _assert_cache_unchanged(
    cache_dir: Path,
    receipt: PoseCacheEntryReceipt,
) -> None:
    path = pose_cache_path(cache_dir, receipt.video_id)
    observed = _stable_file_digest(path)
    if observed.sha256 != receipt.cache_sha256 or observed.byte_count != receipt.bytes:
        raise PoseRACV1PoseCacheError(
            f"pose cache changed after inference: {receipt.video_id}"
        )


def _run_verified_backend(
    *,
    inputs: VerifiedLabelFreeInputs,
    source_archive: VerifiedAsset,
    checkpoint: VerifiedAsset,
    source_root: Path,
    cache_dir: Path,
    pose_fingerprint: str,
    backend: _Backend,
    runner_git_sha: str,
    runner_code_sha256: str,
    video_ids: Sequence[str] | None = None,
    config: PoseRACV1OfficialConfig = FROZEN_POSERAC_V1_CONFIG,
) -> PoseRACV1OfficialRunResult:
    if backend.source_commit != OFFICIAL_SOURCE_COMMIT:
        raise PoseRACV1SourceError("backend source commit is not frozen")
    if backend.source_tree_sha256 != OFFICIAL_SOURCE_TREE_SHA256:
        raise PoseRACV1SourceError("backend source tree SHA-256 is not frozen")
    selected = _select_video_records(inputs, video_ids)
    predictions: list[PoseRACV1VideoPrediction] = []
    receipts: list[PoseCacheEntryReceipt] = []
    for record in selected:
        if record.video_sha256 is None:
            raise PoseRACV1PoseCacheError(
                f"source video SHA-256 is required for {record.video_id}"
            )
        sequence, metadata, receipt = load_pose_cache_with_receipt(
            pose_cache_path(cache_dir, record.video_id),
            expected_video_sha256=record.video_sha256,
            expected_pose_fingerprint=pose_fingerprint,
        )
        if sequence.video_id != record.video_id:
            raise PoseRACV1PoseCacheError("pose cache video_id does not match sidecar")
        if metadata.pose_model != config.expected_pose_model:
            raise PoseRACV1PoseCacheError(
                f"pose model mismatch for {record.video_id}: {metadata.pose_model}"
            )
        if sequence.num_frames != config.expected_frames:
            raise PoseRACV1PoseCacheError(
                f"pose frame count mismatch for {record.video_id}: {sequence.num_frames}"
            )
        output = backend.predict(sequence, config)
        predictions.append(
            PoseRACV1VideoPrediction(
                video_id=record.video_id,
                cache_sha256=receipt.cache_sha256,
                cache_bytes=receipt.bytes,
                count=output.count,
                selected_channel=output.selected_channel,
                selected_action=output.selected_action,
                channel_counts=output.channel_counts,
                channel_dynamic_ranges=output.channel_dynamic_ranges,
                probability_min=output.probability_min,
                probability_max=output.probability_max,
                frames=output.frames,
                valid_frames=output.valid_frames,
            )
        )
        receipts.append(receipt)
    snapshot = PoseCacheSetSnapshot(
        pose_fingerprint=pose_fingerprint,
        entries=tuple(receipts),
    )
    inputs.assert_unchanged()
    source_archive.assert_unchanged()
    checkpoint.assert_unchanged()
    for receipt in receipts:
        _assert_cache_unchanged(cache_dir, receipt)
    if _source_tree_digest(source_root) != (
        OFFICIAL_SOURCE_TREE_FILES,
        OFFICIAL_SOURCE_TREE_BYTES,
        OFFICIAL_SOURCE_TREE_SHA256,
    ):
        raise PoseRACV1SourceError("official source tree changed during inference")
    return PoseRACV1OfficialRunResult(
        inputs=inputs,
        source_archive=source_archive,
        checkpoint=checkpoint,
        config=config,
        pose_snapshot=snapshot,
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
        raise FileExistsError(f"refusing to overwrite PoseRAC-v1 artifact: {path}") from exc
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(path.parent)
    return hashlib.sha256(encoded).hexdigest()


def write_poserac_v1_result_exclusive(
    output_path: str | Path,
    result: PoseRACV1OfficialRunResult,
) -> tuple[Path, str, Path, str]:
    destination = Path(output_path)
    receipt_path = destination.with_suffix(".receipt.json")
    durable_mkdir(destination.parent)
    collisions = [str(path) for path in (destination, receipt_path) if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing to overwrite PoseRAC-v1 artifacts: {collisions}")
    prediction_sha256 = _write_json_exclusive(destination, result.to_dict())
    prediction_digest = _stable_file_digest(destination)
    receipt = {
        "schema_version": 1,
        "artifact_type": "poserac_v1_official_predictions_receipt",
        "method_id": METHOD_ID,
        "prediction_file": destination.name,
        "prediction_sha256": prediction_sha256,
        "prediction_bytes": prediction_digest.byte_count,
        "prediction_total": len(result.predictions),
        "source_commit": OFFICIAL_SOURCE_COMMIT,
        "source_archive_sha256": result.source_archive.spec.sha256,
        "source_tree_sha256": OFFICIAL_SOURCE_TREE_SHA256,
        "checkpoint_sha256": result.checkpoint.spec.sha256,
        "input_sidecar_sha256": result.inputs.sidecar_sha256,
        "input_commitment_sha256": result.inputs.commitment_sha256,
        "input_identity_sha256": result.inputs.identity_sha256,
        "pose_cache_set_sha256": result.pose_snapshot.fingerprint,
        "config_sha256": result.config.fingerprint,
        "compatibility_image_digest": COMPATIBILITY_IMAGE_DIGEST,
        "runner_source_git_sha": result.runner_git_sha,
        "runner_code_sha256": result.runner_code_sha256,
        "labels_loaded": False,
        "scoring_performed": False,
        "ground_truth_count_channel_oracle": False,
    }
    try:
        receipt_sha256 = _write_json_exclusive(receipt_path, receipt)
    except Exception:
        observed = _stable_file_digest(destination)
        if observed == prediction_digest:
            destination.unlink()
            fsync_directory(destination.parent)
        raise
    return destination, prediction_sha256, receipt_path, receipt_sha256


def run_poserac_v1_official(
    *,
    sidecar_path: str | Path,
    commitment_path: str | Path,
    video_root: str | Path,
    pose_cache_dir: str | Path,
    pose_fingerprint: str,
    source_archive_path: str | Path,
    source_root: str | Path,
    checkpoint_path: str | Path,
    repository_root: str | Path,
    video_ids: Sequence[str] | None = None,
    output_path: str | Path | None = None,
    device: str = "cuda",
) -> PoseRACV1OfficialRunResult:
    """Run exact checkpoint weights on count/action-free pose-cache inputs."""

    inputs = _load_label_free_inputs(
        sidecar_path,
        commitment_path,
        video_root,
        require_exact_membership=True,
    )
    if output_path is not None:
        destination = Path(output_path)
        if destination.exists() or destination.with_suffix(".receipt.json").exists():
            raise FileExistsError(f"refusing to overwrite PoseRAC-v1 output: {destination}")
    repository = Path(repository_root).expanduser().resolve(strict=True)
    runner_git_sha = _clean_runner_git_revision(repository)
    runner_code = (repository / RUNNER_CODE_RELATIVE_PATH).resolve(strict=True)
    if runner_code != Path(__file__).resolve(strict=True):
        raise PoseRACV1SourceError(
            "executed PoseRAC-v1 runner is outside repository_root"
        )
    runner_digest = _stable_file_digest(runner_code)
    source_archive = _verify_asset_with_spec(source_archive_path, SOURCE_ARCHIVE_SPEC)
    verified_source_root = verify_official_source_tree(source_root)
    checkpoint = _verify_asset_with_spec(checkpoint_path, CHECKPOINT_SPEC)
    expected_checkpoint = (verified_source_root / CHECKPOINT_SPEC.name).resolve(strict=True)
    if checkpoint.path != expected_checkpoint:
        raise PoseRACV1CheckpointError(
            "checkpoint must be the byte-verified object inside the frozen source tree"
        )
    cache_dir = Path(pose_cache_dir).expanduser().resolve(strict=True)
    if not cache_dir.is_dir():
        raise PoseRACV1PoseCacheError("pose_cache_dir is not a directory")
    backend = ExactCheckpointBackend.from_checkpoint(
        checkpoint,
        FROZEN_POSERAC_V1_CONFIG,
        device=device,
    )
    result = _run_verified_backend(
        inputs=inputs,
        source_archive=source_archive,
        checkpoint=checkpoint,
        source_root=verified_source_root,
        cache_dir=cache_dir,
        pose_fingerprint=pose_fingerprint,
        backend=backend,
        runner_git_sha=runner_git_sha,
        runner_code_sha256=runner_digest.sha256,
        video_ids=video_ids,
    )
    if _stable_file_digest(runner_code) != runner_digest:
        raise PoseRACV1SourceError("PoseRAC-v1 runner code changed during inference")
    if _clean_runner_git_revision(repository) != runner_git_sha:
        raise PoseRACV1SourceError("PoseRAC-v1 runner Git revision changed during inference")
    if output_path is not None:
        write_poserac_v1_result_exclusive(output_path, result)
    return result


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pams.baselines.poserac_v1_official_runner",
        description=(
            "Run the released PoseRAC-v1 checkpoint with an inferred oracle-free "
            "channel rule on strict schema-v2 pose caches."
        ),
    )
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--commitment", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--pose-cache-dir", type=Path, required=True)
    parser.add_argument("--pose-fingerprint", required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-id", action="append", dest="video_ids")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _argument_parser()
    arguments = parser.parse_args(None if argv is None else list(argv))
    try:
        result = run_poserac_v1_official(
            sidecar_path=arguments.sidecar,
            commitment_path=arguments.commitment,
            video_root=arguments.video_root,
            pose_cache_dir=arguments.pose_cache_dir,
            pose_fingerprint=arguments.pose_fingerprint,
            source_archive_path=arguments.source_archive,
            source_root=arguments.source_root,
            checkpoint_path=arguments.checkpoint,
            repository_root=arguments.repository_root,
            video_ids=arguments.video_ids,
            output_path=arguments.output,
            device=arguments.device,
        )
        output_sha256 = _stable_file_digest(arguments.output).sha256
    except (OSError, PoseRACV1OfficialError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "output_sha256": output_sha256,
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
    "ACTION_NAMES",
    "BackendPrediction",
    "CHECKPOINT_SPEC",
    "ExactCheckpointBackend",
    "ExactPoseRACV1",
    "FROZEN_POSERAC_V1_CONFIG",
    "OFFICIAL_SOURCE_COMMIT",
    "PoseRACV1OfficialConfig",
    "PoseRACV1OfficialRunResult",
    "count_trigger_cycles",
    "main",
    "official_axis_normalize",
    "run_poserac_v1_official",
    "select_dynamic_range_channel",
    "smooth_scores",
    "verify_official_source_tree",
    "write_poserac_v1_result_exclusive",
]


if __name__ == "__main__":
    raise SystemExit(main())
