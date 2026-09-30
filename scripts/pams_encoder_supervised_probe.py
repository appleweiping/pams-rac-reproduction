"""Strict supervised upper-bound probe for a frozen PAMS encoder.

This is deliberately *not* a PAMS result and is ineligible for the main
self-supervised benchmark.  The ``train`` process may open only canonical
train337 counts and train337 pose inputs.  The separate ``predict`` process
may open only the frozen model and label-free dev84 inputs.  Development
targets are accepted only by ``score``, after the model and predictions have
been frozen.

The fixed 1027-dimensional probe is a standardized Ridge regressor over the
mask-aware 512-D mean and 512-D population standard deviation of the frozen
post-positional-encoding Transformer embeddings, followed by
``log1p(n_valid / T_z)``, the label-free period confidence, and
``n_valid / 256``.  ``T_z`` is the canonical post-PE
``estimate_period_from_embeddings`` estimate.

No action labels are exposed to feature extraction or regressor fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from torch import Tensor

import pams.data as data_module
import pams.metrics as metrics_module
import pams.model as model_module
import pams.period as period_module
import pams.training as training_module
from pams.config import PAMSConfig, load_config
from pams.data import (
    PoseCacheSetSnapshot,
    PoseInputManifest,
    load_dev_target_manifest,
    load_pose_cache_set,
    load_pose_input_manifest,
)
from pams.metrics import compute_count_metrics, round_count
from pams.period import estimate_period_from_embeddings
from pams.reproducibility import sha256_file
from pams.run_manifest import CompletedRunReceipt
from pams.training import (
    CheckpointProvenance,
    collate_pose_sequences,
    load_model_checkpoint,
    validate_terminal_checkpoint,
)
from pams.types import PoseSequence

_METHOD_KEY: Final = "pams-frozen-encoder-ridge-count-probe-v1"
_STATUS: Final = "supervised_upper_bound_probe_not_pams_result"
_SEED: Final = 2026
_EMBEDDING_DIMENSIONS: Final = 512
_FEATURE_DIMENSIONS: Final = 1027
_RIDGE_ALPHA: Final = 1.0
_MINIMUM_COUNT: Final = 0.0
_EXPECTED_TRAIN: Final = 337
_EXPECTED_DEV: Final = 84
_TRAIN_TARGET_SHA256: Final = (
    "73999befa45bb55861acfbf8a6b34f04f4a9b34c1e46fa15bc51a88ed6bcfb1e"
)
_DEV_TARGET_SHA256: Final = (
    "1ab3bc010ccb5d869af7662586594c973e53841838643e1e9b8cc1787b0efab6"
)

_FEATURE_SPEC: Final[dict[str, Any]] = {
    "version": 1,
    "source": "frozen_post_pe_transformer_embedding",
    "embedding_dimensions": _EMBEDDING_DIMENSIONS,
    "pooling": "mask_aware_mean_and_population_std",
    "tail_features": [
        "log1p(valid_frame_count / canonical_post_pe_period_frames)",
        "canonical_post_pe_period_confidence",
        "valid_frame_count / 256",
    ],
    "period_range_frames": [4, 128],
    "feature_dimensions": _FEATURE_DIMENSIONS,
    "regressor": {
        "type": "sklearn_ridge",
        "target": "log1p_positive_count",
        "alpha": _RIDGE_ALPHA,
        "solver": "svd",
        "fit_intercept": True,
        "feature_scaler": "standard_mean_and_population_std",
        "prediction_floor": _MINIMUM_COUNT,
        "all_invalid_prediction": 0.0,
    },
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_code_hash_binding(
    observed: Any,
    expected: Mapping[str, str],
    *,
    stage: str,
) -> None:
    if observed != dict(expected):
        raise ValueError(f"{stage} code hashes differ from current source-bound image")


def _require_frozen_target_sha256(digest: str, *, split: str) -> None:
    expected = {
        "train": _TRAIN_TARGET_SHA256,
        "dev": _DEV_TARGET_SHA256,
    }
    if split not in expected:
        raise ValueError(f"unsupported frozen target split: {split!r}")
    if digest != expected[split]:
        raise ValueError(f"{split} target SHA-256 differs from frozen probe protocol")


def _probe_source_revision() -> str:
    revision = os.environ.get("PAMS_CONTAINER_SOURCE_REVISION", "")
    if (
        len(revision) != 40
        or revision != revision.lower()
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise RuntimeError(
            "PAMS_CONTAINER_SOURCE_REVISION must be exactly 40 lowercase Git "
            "hex characters"
        )
    return revision


def _require_source_revision_binding(
    observed: Any,
    current: str,
    *,
    stage: str,
) -> None:
    if observed != current:
        raise ValueError(
            f"{stage} probe source revision differs from source-bound image"
        )


def _stable_array_sha256(array: np.ndarray) -> str:
    values = np.ascontiguousarray(array, dtype="<f8")
    shape = json.dumps(list(values.shape), separators=(",", ":")).encode("ascii")
    return _sha256_bytes(shape + b"\0" + values.tobytes(order="C"))


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return _sha256_bytes(encoded)


def _load_json_object(path: Path, *, document_name: str) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field in {document_name}: {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant in {document_name}: {value}")
            ),
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {document_name} JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{document_name} root must be an object")
    return payload


def _validate_encoder_completion_receipt(
    path: Path,
    *,
    expected_source_git_sha: str,
    config_fingerprint: str,
    config_file_sha256: str,
    checkpoint_sha256: str,
    progress_sha256: str,
) -> tuple[CompletedRunReceipt, str]:
    """Bind a schema-v3 completed run to the exact encoder inputs."""

    payload = _load_json_object(path, document_name="encoder completion receipt")
    receipt = CompletedRunReceipt.model_validate(payload)
    if (
        receipt.schema_version != 3
        or receipt.receipt_type != "completed"
        or receipt.status != "completed"
    ):
        raise ValueError("encoder completion receipt must be schema-v3 completed")
    if receipt.started.git_sha != expected_source_git_sha:
        raise ValueError(
            "encoder completion receipt started.git_sha differs from expected source"
        )
    if receipt.started.config_sha256 != config_fingerprint:
        raise ValueError(
            "encoder completion receipt started.config_sha256 differs from "
            "PAMSConfig fingerprint"
        )
    artifacts = {artifact.role: artifact for artifact in receipt.artifacts}
    required = {"input_config", "output_encoder_checkpoint", "progress_log"}
    missing = sorted(required - set(artifacts))
    if missing:
        raise ValueError(
            f"encoder completion receipt is missing required artifacts: {missing}"
        )
    if artifacts["input_config"].sha256 != config_file_sha256:
        raise ValueError(
            "encoder completion receipt input_config SHA-256 differs from config file"
        )
    if artifacts["output_encoder_checkpoint"].sha256 != checkpoint_sha256:
        raise ValueError(
            "encoder completion receipt checkpoint SHA-256 differs from actual checkpoint"
        )
    if artifacts["progress_log"].sha256 != progress_sha256:
        raise ValueError(
            "encoder completion receipt progress SHA-256 differs from actual progress log"
        )
    return receipt, sha256_file(path)


def _load_train_targets(path: Path, expected_ids: tuple[str, ...]) -> np.ndarray:
    payload = _load_json_object(path, document_name="train count target sidecar")
    expected_fields = {
        "manifest_type",
        "protocol",
        "records",
        "schema_version",
        "split",
        "source_annotation_sha256",
    }
    if set(payload) != expected_fields:
        raise ValueError("training target sidecar fields changed")
    if (
        payload["manifest_type"] != "train_count_targets"
        or payload["protocol"] != "ucfrep_526"
        or payload["split"] != "train"
        or payload["schema_version"] != 1
    ):
        raise ValueError("training target sidecar protocol binding changed")
    rows = payload["records"]
    if not isinstance(rows, list):
        raise TypeError("training target records must be a list")
    if any(not isinstance(row, dict) or set(row) != {"count", "video_id"} for row in rows):
        raise ValueError(
            "training target records may contain only video_id/count; "
            "action labels are forbidden"
        )
    identifiers = tuple(str(row["video_id"]) for row in rows)
    if identifiers != expected_ids or len(set(identifiers)) != len(identifiers):
        raise ValueError("training targets do not exactly match ordered train337 inputs")
    counts: list[int] = []
    for row in rows:
        value = row["count"]
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(float(value))
            or int(value) != value
            or int(value) <= 0
        ):
            raise ValueError("training counts must be positive integers")
        counts.append(int(value))
    return np.asarray(counts, dtype=np.float64)


def _validate_split_manifests(
    train: PoseInputManifest,
    dev: PoseInputManifest,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if train.protocol != "ucfrep_526" or dev.protocol != "ucfrep_526":
        raise ValueError("probe is frozen only for ucfrep_526")
    if train.split != "train" or dev.split != "dev":
        raise ValueError("probe requires canonical train337 and dev84 sidecars")
    train_ids = tuple(record.video_id for record in train.records)
    dev_ids = tuple(record.video_id for record in dev.records)
    if len(train_ids) != _EXPECTED_TRAIN or len(dev_ids) != _EXPECTED_DEV:
        raise ValueError("probe requires exactly train337 and dev84")
    if len(set(train_ids)) != len(train_ids) or len(set(dev_ids)) != len(dev_ids):
        raise ValueError("probe sidecars contain duplicate video IDs")
    if set(train_ids) & set(dev_ids):
        raise ValueError("probe train337 and dev84 membership must be disjoint")
    return train_ids, dev_ids


def _training_dataset_fingerprint(
    records: Sequence[Any],
    *,
    protocol: str,
) -> str:
    normalized = sorted(records, key=lambda record: record.video_id)
    encoded = json.dumps(
        {
            "schema_version": 1,
            "protocol": protocol,
            "records": [
                {
                    "video_id": record.video_id,
                    "video_sha256": record.video_sha256,
                }
                for record in normalized
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _checkpoint_provenance(path: Path) -> CheckpointProvenance:
    payload = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("encoder checkpoint root must be a mapping")
    if payload.get("stage") != "encoder":
        raise ValueError("supervised probe requires an encoder-stage checkpoint")
    raw = payload.get("provenance")
    if not isinstance(raw, Mapping):
        raise ValueError("encoder checkpoint is missing bound provenance")
    return CheckpointProvenance.from_mapping(raw)


def _validate_and_load_encoder(
    checkpoint: Path,
    progress: Path,
    config: PAMSConfig,
    *,
    train: PoseInputManifest,
    train_snapshot: PoseCacheSetSnapshot,
    expected_source_git_sha: str,
    device: torch.device,
) -> tuple[torch.nn.Module, CheckpointProvenance]:
    actual = _checkpoint_provenance(checkpoint)
    if actual.source_git_sha != expected_source_git_sha:
        raise ValueError(
            "encoder checkpoint source Git SHA differs from the explicit expected revision"
        )
    expected = CheckpointProvenance(
        protocol=train.protocol,
        dataset_fingerprint=_training_dataset_fingerprint(
            train.records,
            protocol=train.protocol,
        ),
        training_video_ids=tuple(record.video_id for record in train.records),
        pose_fingerprint=config.pose_fingerprint,
        pose_cache_set_sha256=train_snapshot.fingerprint,
        source_git_sha=expected_source_git_sha,
        container_image_id=actual.container_image_id,
        container_environment_sha256=actual.container_environment_sha256,
        upstream_encoder_checkpoint_sha256=None,
    )
    validate_terminal_checkpoint(
        checkpoint,
        config,
        expected_stage="encoder",
        expected_provenance=expected,
        progress_path=progress,
    )
    model = load_model_checkpoint(
        checkpoint,
        config,
        device=device,
        expected_stage="encoder",
        expected_provenance=expected,
    )
    encoder = model.encoder
    encoder.requires_grad_(False)
    encoder.eval()
    if any(parameter.requires_grad for parameter in encoder.parameters()):
        raise RuntimeError("frozen encoder still contains trainable parameters")
    return encoder, expected


def _masked_mean_std(values: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
    if values.ndim != 3 or valid_mask.shape != values.shape[:2]:
        raise ValueError("masked statistics expect [batch,time,features] and [batch,time]")
    weights = valid_mask.to(dtype=values.dtype).unsqueeze(-1)
    count = weights.sum(dim=1).clamp_min(1.0)
    mean = (values * weights).sum(dim=1) / count
    centered = (values - mean.unsqueeze(1)).masked_fill(~valid_mask.unsqueeze(-1), 0.0)
    variance = centered.square().sum(dim=1) / count
    std = variance.clamp_min(0.0).sqrt()
    available = valid_mask.any(dim=1, keepdim=True)
    return mean.masked_fill(~available, 0.0), std.masked_fill(~available, 0.0)


def _feature_names() -> tuple[str, ...]:
    names = [
        *(f"embedding_mean_{index:03d}" for index in range(_EMBEDDING_DIMENSIONS)),
        *(f"embedding_population_std_{index:03d}" for index in range(_EMBEDDING_DIMENSIONS)),
        "log1p_valid_frames_over_period",
        "period_confidence",
        "valid_frame_fraction",
    ]
    result = tuple(names)
    if len(result) != _FEATURE_DIMENSIONS:
        raise RuntimeError("canonical probe feature-name count changed")
    return result


def _batch_features(
    embeddings: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    if embeddings.ndim != 3 or valid_mask.shape != embeddings.shape[:2]:
        raise ValueError("canonical features expect embeddings [B,T,512] and mask [B,T]")
    batch, time, dimensions = embeddings.shape
    if dimensions != _EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"canonical probe requires {_EMBEDDING_DIMENSIONS}-D embeddings"
        )
    if time != 256:
        raise ValueError("canonical probe requires exactly 256 sampled frames")
    pooled_mean, pooled_std = _masked_mean_std(embeddings, valid_mask)
    period, confidence = estimate_period_from_embeddings(
        embeddings,
        minimum=4,
        maximum=128,
        valid_mask=valid_mask,
    )
    valid_count = valid_mask.sum(dim=1).to(dtype=embeddings.dtype)
    proxy_count = valid_count / period.clamp_min(1.0)
    scalars = torch.stack(
        (
            torch.log1p(proxy_count),
            confidence,
            valid_count / 256.0,
        ),
        dim=1,
    )
    features = torch.cat((pooled_mean, pooled_std, scalars), dim=1)
    if features.shape != (batch, _FEATURE_DIMENSIONS):
        raise RuntimeError("feature-name and feature-tensor dimensions diverged")
    if not bool(torch.isfinite(features).all().item()):
        raise FloatingPointError("probe feature extraction produced non-finite values")
    return features


def _extract_features(
    sequences: Sequence[PoseSequence],
    encoder: torch.nn.Module,
    *,
    config: PAMSConfig,
    device: torch.device,
    batch_size: int,
) -> tuple[tuple[str, ...], np.ndarray]:
    if batch_size < 1:
        raise ValueError("feature batch size must be positive")
    items = tuple(sequences)
    if not items:
        raise ValueError("feature extraction requires at least one pose sequence")
    if any(sequence.num_frames != config.data.frames for sequence in items):
        raise ValueError(
            f"probe requires every pose cache to contain {config.data.frames} frames"
        )
    identifiers: list[str] = []
    matrices: list[np.ndarray] = []
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            batch = collate_pose_sequences(items[start : start + batch_size]).to(device)
            embeddings = encoder(batch.poses, batch.valid_mask)
            features = _batch_features(
                embeddings,
                batch.valid_mask,
            )
            identifiers.extend(batch.video_ids)
            matrices.append(features.detach().cpu().numpy().astype(np.float64))
    matrix = np.concatenate(matrices, axis=0)
    if matrix.shape != (len(items), len(_feature_names())):
        raise RuntimeError("probe feature matrix has an unexpected shape")
    if not np.isfinite(matrix).all():
        raise FloatingPointError("probe feature matrix contains non-finite values")
    return tuple(identifiers), matrix


def _fit_ridge(
    train_features: np.ndarray,
    targets: np.ndarray,
) -> tuple[StandardScaler, Ridge, np.ndarray]:
    if train_features.ndim != 2:
        raise ValueError("probe features must be a matrix")
    if targets.shape != (train_features.shape[0],):
        raise ValueError("training targets do not match feature rows")
    scaler = StandardScaler(with_mean=True, with_std=True)
    train_scaled = scaler.fit_transform(train_features)
    regressor = Ridge(
        alpha=_RIDGE_ALPHA,
        fit_intercept=True,
        solver="svd",
    )
    regressor.fit(train_scaled, np.log1p(targets))
    train_prediction = np.maximum(
        np.expm1(np.asarray(regressor.predict(train_scaled), dtype=np.float64)),
        _MINIMUM_COUNT,
    )
    if not np.isfinite(train_prediction).all():
        raise FloatingPointError("Ridge probe produced non-finite predictions")
    return scaler, regressor, train_prediction


def _predict_ridge(
    features: np.ndarray,
    *,
    scaler_mean: np.ndarray,
    scaler_scale: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
) -> np.ndarray:
    if features.ndim != 2 or features.shape[1] != _FEATURE_DIMENSIONS:
        raise ValueError("canonical prediction features must have shape [N,1027]")
    for name, values in (
        ("scaler mean", scaler_mean),
        ("scaler scale", scaler_scale),
        ("ridge coefficients", coefficients),
    ):
        if values.shape != (_FEATURE_DIMENSIONS,) or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain exactly 1027 finite values")
    if np.any(scaler_scale <= 0) or not math.isfinite(intercept):
        raise ValueError("probe scaler/ridge parameters are invalid")
    transformed = (features - scaler_mean) / scaler_scale
    prediction = np.maximum(
        np.expm1(transformed @ coefficients + intercept),
        _MINIMUM_COUNT,
    )
    # The final feature is n_valid / 256.  This rule is frozen before dev
    # targets are reachable and prevents an intercept-only count for pose
    # extraction failures.
    prediction[features[:, -1] == 0.0] = 0.0
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Ridge probe produced non-finite predictions")
    return np.asarray(prediction, dtype=np.float64)


def _code_file_hashes() -> dict[str, str]:
    paths = {
        "probe": Path(__file__).resolve(),
        "pams_data": Path(data_module.__file__).resolve(),
        "pams_metrics": Path(metrics_module.__file__).resolve(),
        "pams_model": Path(model_module.__file__).resolve(),
        "pams_period": Path(period_module.__file__).resolve(),
        "pams_training": Path(training_module.__file__).resolve(),
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _git_identity() -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"revision": None, "clean": None}
    return {"revision": revision, "clean": not bool(status.strip())}


def _runtime_identity(device: torch.device) -> dict[str, Any]:
    gpu: dict[str, Any] | None = None
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu = {
            "index": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "capability": list(properties.major_minor)
            if hasattr(properties, "major_minor")
            else [properties.major, properties.minor],
        }
    return {
        "python": os.sys.version,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": gpu,
    }


def _shared_encoder_input_hashes(arguments: argparse.Namespace) -> dict[str, str]:
    return {
        "config": sha256_file(arguments.config),
        "encoder_checkpoint": sha256_file(arguments.encoder_checkpoint),
        "encoder_progress": sha256_file(arguments.encoder_progress),
        "encoder_completion_receipt": sha256_file(
            arguments.encoder_completion_receipt
        ),
    }


def _seed_runtime() -> None:
    random.seed(_SEED)
    np.random.seed(_SEED)
    torch.manual_seed(_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_probe(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if arguments.feature_batch_size < 1:
        raise ValueError("feature batch size must be positive")
    output_dir = arguments.output_dir.resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    _seed_runtime()

    for name in (
        "config",
        "encoder_checkpoint",
        "encoder_progress",
        "encoder_completion_receipt",
        "train_inputs",
        "train_targets",
    ):
        setattr(arguments, name, getattr(arguments, name).resolve(strict=True))
    arguments.pose_cache = arguments.pose_cache.resolve(strict=True)
    initial_hashes = {
        **_shared_encoder_input_hashes(arguments),
        "train_inputs": sha256_file(arguments.train_inputs),
        "train_targets": sha256_file(arguments.train_targets),
    }
    initial_code_hashes = _code_file_hashes()
    probe_source_revision = _probe_source_revision()
    _require_frozen_target_sha256(
        initial_hashes["train_targets"],
        split="train",
    )

    config = load_config(arguments.config)
    if config.seed != _SEED:
        raise ValueError(f"probe requires config seed {_SEED}")
    if config.protocol != "ucfrep_526":
        raise ValueError("probe requires ucfrep_526 config")
    _, completion_receipt_sha256 = _validate_encoder_completion_receipt(
        arguments.encoder_completion_receipt,
        expected_source_git_sha=arguments.expected_encoder_source_git_sha,
        config_fingerprint=config.fingerprint,
        config_file_sha256=initial_hashes["config"],
        checkpoint_sha256=initial_hashes["encoder_checkpoint"],
        progress_sha256=initial_hashes["encoder_progress"],
    )
    if completion_receipt_sha256 != initial_hashes["encoder_completion_receipt"]:
        raise RuntimeError("encoder completion receipt changed while it was loaded")
    train_manifest = load_pose_input_manifest(arguments.train_inputs, validate_exact=True)
    if train_manifest.split != "train" or len(train_manifest.records) != _EXPECTED_TRAIN:
        raise ValueError("probe training requires the canonical train337 sidecar")
    train_ids = tuple(record.video_id for record in train_manifest.records)
    train_targets = _load_train_targets(arguments.train_targets, train_ids)

    train_sequences, train_snapshot = load_pose_cache_set(
        train_manifest.records,
        cache_dir=arguments.pose_cache,
        pose_fingerprint=config.pose_fingerprint,
    )
    device = torch.device(arguments.device)
    encoder, checkpoint_provenance = _validate_and_load_encoder(
        arguments.encoder_checkpoint,
        arguments.encoder_progress,
        config,
        train=train_manifest,
        train_snapshot=train_snapshot,
        expected_source_git_sha=arguments.expected_encoder_source_git_sha,
        device=device,
    )
    observed_train_ids, train_features = _extract_features(
        train_sequences,
        encoder,
        config=config,
        device=device,
        batch_size=arguments.feature_batch_size,
    )
    if observed_train_ids != train_ids:
        raise RuntimeError("feature extraction changed canonical train337 input order")

    scaler, regressor, train_prediction = _fit_ridge(
        train_features,
        train_targets,
    )
    train_error = train_prediction - train_targets
    feature_names = _feature_names()
    checkpoint_sha256 = initial_hashes["encoder_checkpoint"]

    model_payload = {
        "schema_version": 1,
        "artifact_type": "pams_encoder_supervised_upper_bound_model",
        "method_key": _METHOD_KEY,
        "status": _STATUS,
        "protocol": "ucfrep_526",
        "seed": _SEED,
        "probe_source_revision": probe_source_revision,
        "feature_spec": _FEATURE_SPEC,
        "feature_names": list(feature_names),
        "scaler": {
            "mean": np.asarray(scaler.mean_, dtype=np.float64).tolist(),
            "scale": np.asarray(scaler.scale_, dtype=np.float64).tolist(),
            "var": np.asarray(scaler.var_, dtype=np.float64).tolist(),
        },
        "ridge": {
            "alpha": _RIDGE_ALPHA,
            "solver": "svd",
            "intercept": float(regressor.intercept_),
            "coefficients": np.asarray(regressor.coef_, dtype=np.float64).tolist(),
        },
        "training": {
            "sample_count": len(train_ids),
            "target_sha256": initial_hashes["train_targets"],
            "input_sha256": initial_hashes["train_inputs"],
            "pose_cache_snapshot": train_snapshot.to_dict(),
            "feature_matrix_sha256": _stable_array_sha256(train_features),
            "prediction_mae": float(np.mean(np.abs(train_error))),
            "prediction_rmse": float(np.sqrt(np.mean(np.square(train_error)))),
        },
        "encoder": {
            "checkpoint_sha256": checkpoint_sha256,
            "progress_sha256": initial_hashes["encoder_progress"],
            "completion_receipt_sha256": completion_receipt_sha256,
            "config_fingerprint": config.fingerprint,
            "provenance": checkpoint_provenance.to_dict(),
            "frozen": True,
        },
        "code_files_sha256": initial_code_hashes,
    }

    _, final_train_snapshot = load_pose_cache_set(
        train_manifest.records,
        cache_dir=arguments.pose_cache,
        pose_fingerprint=config.pose_fingerprint,
        materialize_sequences=False,
    )
    if final_train_snapshot.fingerprint != train_snapshot.fingerprint:
        raise RuntimeError("train337 pose caches changed during probe")
    if {
        **_shared_encoder_input_hashes(arguments),
        "train_inputs": sha256_file(arguments.train_inputs),
        "train_targets": sha256_file(arguments.train_targets),
    } != initial_hashes:
        raise RuntimeError("a probe input file changed during execution")
    if _code_file_hashes() != initial_code_hashes:
        raise RuntimeError("probe source code changed during execution")

    model_path = output_dir / "probe-model.json"
    model_sha256 = _write_json_exclusive(model_path, model_payload)
    run_payload = {
        "schema_version": 1,
        "artifact_type": "pams_encoder_supervised_upper_bound_train_run",
        "method_key": _METHOD_KEY,
        "status": _STATUS,
        "seed": _SEED,
        "probe_source_revision": probe_source_revision,
        "eligibility": {
            "pams_main_result": False,
            "self_supervised_result": False,
            "baseline_result": False,
            "classification": "diagnostic_supervised_encoder_upper_bound",
        },
        "label_access": {
            "train337_counts": True,
            "train337_actions": False,
            "dev84_identity_or_labels": False,
            "sealed_test105_identity_or_labels": False,
        },
        "inputs": initial_hashes,
        "pose_cache_snapshot": train_snapshot.to_dict(),
        "code_files_sha256": initial_code_hashes,
        "probe_checkout": _git_identity(),
        "runtime": _runtime_identity(device),
        "artifacts": {
            "model": {"path": model_path.name, "sha256": model_sha256},
        },
        "test105_access": "none",
        "dev84_reachable_by_train_process": False,
    }
    run_path = output_dir / "train-run.json"
    run_sha256 = _write_json_exclusive(run_path, run_payload)
    return {
        "status": _STATUS,
        "output_dir": str(output_dir),
        "model_sha256": model_sha256,
        "run_sha256": run_sha256,
        "dev84_reachable": False,
        "test105_access": "none",
    }


def _validated_model_artifact(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path, document_name="supervised probe model")
    expected = {
        "schema_version",
        "artifact_type",
        "method_key",
        "status",
        "protocol",
        "seed",
        "probe_source_revision",
        "feature_spec",
        "feature_names",
        "scaler",
        "ridge",
        "training",
        "encoder",
        "code_files_sha256",
    }
    if set(payload) != expected:
        raise ValueError("supervised probe model fields changed")
    if (
        payload["schema_version"] != 1
        or payload["artifact_type"] != "pams_encoder_supervised_upper_bound_model"
        or payload["method_key"] != _METHOD_KEY
        or payload["status"] != _STATUS
        or payload["protocol"] != "ucfrep_526"
        or payload["seed"] != _SEED
        or payload["feature_spec"] != _FEATURE_SPEC
        or tuple(payload["feature_names"]) != _feature_names()
    ):
        raise ValueError("supervised probe model binding changed")
    scaler = payload["scaler"]
    ridge = payload["ridge"]
    if not isinstance(scaler, dict) or set(scaler) != {"mean", "scale", "var"}:
        raise ValueError("supervised probe scaler fields changed")
    if not isinstance(ridge, dict) or set(ridge) != {
        "alpha",
        "solver",
        "intercept",
        "coefficients",
    }:
        raise ValueError("supervised probe Ridge fields changed")
    if ridge["alpha"] != _RIDGE_ALPHA or ridge["solver"] != "svd":
        raise ValueError("supervised probe Ridge hyperparameters changed")
    encoder = payload["encoder"]
    if not isinstance(encoder, dict) or set(encoder) != {
        "checkpoint_sha256",
        "progress_sha256",
        "completion_receipt_sha256",
        "config_fingerprint",
        "provenance",
        "frozen",
    }:
        raise ValueError("supervised probe encoder binding fields changed")
    if encoder["frozen"] is not True:
        raise ValueError("supervised probe model is not bound to a frozen encoder")
    return payload


def predict_probe(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if arguments.feature_batch_size < 1:
        raise ValueError("feature batch size must be positive")
    output_dir = arguments.output_dir.resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    _seed_runtime()

    for name in (
        "config",
        "encoder_checkpoint",
        "encoder_progress",
        "encoder_completion_receipt",
        "model",
        "dev_inputs",
    ):
        setattr(arguments, name, getattr(arguments, name).resolve(strict=True))
    arguments.pose_cache = arguments.pose_cache.resolve(strict=True)
    initial_hashes = {
        **_shared_encoder_input_hashes(arguments),
        "model": sha256_file(arguments.model),
        "dev_inputs": sha256_file(arguments.dev_inputs),
    }
    initial_code_hashes = _code_file_hashes()
    probe_source_revision = _probe_source_revision()
    model_payload = _validated_model_artifact(arguments.model)
    _require_code_hash_binding(
        model_payload["code_files_sha256"],
        initial_code_hashes,
        stage="probe model",
    )
    _require_source_revision_binding(
        model_payload["probe_source_revision"],
        probe_source_revision,
        stage="probe model",
    )
    config = load_config(arguments.config)
    if config.seed != _SEED or config.protocol != "ucfrep_526":
        raise ValueError("probe requires seed2026 ucfrep_526 config")
    encoder_binding = model_payload["encoder"]
    if (
        encoder_binding["checkpoint_sha256"] != initial_hashes["encoder_checkpoint"]
        or encoder_binding["progress_sha256"] != initial_hashes["encoder_progress"]
        or encoder_binding["completion_receipt_sha256"]
        != initial_hashes["encoder_completion_receipt"]
        or encoder_binding["config_fingerprint"] != config.fingerprint
    ):
        raise ValueError("prediction inputs differ from the trained probe model binding")
    expected_provenance = CheckpointProvenance.from_mapping(
        encoder_binding["provenance"]
    )
    if expected_provenance.source_git_sha != arguments.expected_encoder_source_git_sha:
        raise ValueError("explicit encoder source Git SHA differs from probe model binding")
    _, completion_receipt_sha256 = _validate_encoder_completion_receipt(
        arguments.encoder_completion_receipt,
        expected_source_git_sha=arguments.expected_encoder_source_git_sha,
        config_fingerprint=config.fingerprint,
        config_file_sha256=initial_hashes["config"],
        checkpoint_sha256=initial_hashes["encoder_checkpoint"],
        progress_sha256=initial_hashes["encoder_progress"],
    )
    if completion_receipt_sha256 != encoder_binding["completion_receipt_sha256"]:
        raise ValueError(
            "encoder completion receipt differs from trained probe model binding"
        )
    validate_terminal_checkpoint(
        arguments.encoder_checkpoint,
        config,
        expected_stage="encoder",
        expected_provenance=expected_provenance,
        progress_path=arguments.encoder_progress,
    )
    device = torch.device(arguments.device)
    model = load_model_checkpoint(
        arguments.encoder_checkpoint,
        config,
        device=device,
        expected_stage="encoder",
        expected_provenance=expected_provenance,
    )
    encoder = model.encoder
    encoder.requires_grad_(False)
    encoder.eval()

    dev_manifest = load_pose_input_manifest(arguments.dev_inputs, validate_exact=True)
    if dev_manifest.split != "dev" or len(dev_manifest.records) != _EXPECTED_DEV:
        raise ValueError("probe prediction requires the canonical dev84 sidecar")
    dev_ids = tuple(record.video_id for record in dev_manifest.records)
    if set(dev_ids) & set(expected_provenance.training_video_ids):
        raise ValueError("dev84 IDs overlap the encoder/probe training membership")
    dev_sequences, dev_snapshot = load_pose_cache_set(
        dev_manifest.records,
        cache_dir=arguments.pose_cache,
        pose_fingerprint=config.pose_fingerprint,
    )
    observed_dev_ids, dev_features = _extract_features(
        dev_sequences,
        encoder,
        config=config,
        device=device,
        batch_size=arguments.feature_batch_size,
    )
    if observed_dev_ids != dev_ids:
        raise RuntimeError("feature extraction changed canonical dev84 input order")
    scaler = model_payload["scaler"]
    ridge = model_payload["ridge"]
    dev_prediction = _predict_ridge(
        dev_features,
        scaler_mean=np.asarray(scaler["mean"], dtype=np.float64),
        scaler_scale=np.asarray(scaler["scale"], dtype=np.float64),
        coefficients=np.asarray(ridge["coefficients"], dtype=np.float64),
        intercept=float(ridge["intercept"]),
    )
    prediction_rows = [
        {
            "video_id": video_id,
            "raw_count": float(value),
            "rounded_count": round_count(float(value)),
        }
        for video_id, value in zip(dev_ids, dev_prediction, strict=True)
    ]
    predictions_payload = {
        "schema_version": 1,
        "artifact_type": "pams_encoder_supervised_upper_bound_predictions",
        "method_key": _METHOD_KEY,
        "status": _STATUS,
        "protocol": "ucfrep_526",
        "split": "dev",
        "seed": _SEED,
        "probe_source_revision": probe_source_revision,
        "eligibility": {
            "pams_main_result": False,
            "self_supervised_result": False,
            "baseline_result": False,
            "classification": "diagnostic_supervised_encoder_upper_bound",
        },
        "label_access": {
            "train337_counts": True,
            "train337_actions": False,
            "dev84_counts_during_run": False,
            "dev84_actions_during_run": False,
            "sealed_test105_identity_or_labels": False,
        },
        "feature_spec": _FEATURE_SPEC,
        "feature_names_sha256": _sha256_bytes(
            json.dumps(_feature_names(), separators=(",", ":")).encode("utf-8")
        ),
        "model_sha256": initial_hashes["model"],
        "encoder_checkpoint_sha256": initial_hashes["encoder_checkpoint"],
        "encoder_completion_receipt_sha256": completion_receipt_sha256,
        "encoder_progress_sha256": initial_hashes["encoder_progress"],
        "config_file_sha256": initial_hashes["config"],
        "config_fingerprint": config.fingerprint,
        "dev_inputs_sha256": initial_hashes["dev_inputs"],
        "dev_feature_matrix_sha256": _stable_array_sha256(dev_features),
        "dev_pose_cache_snapshot": dev_snapshot.to_dict(),
        "code_files_sha256": initial_code_hashes,
        "predictions": prediction_rows,
    }

    _, final_dev_snapshot = load_pose_cache_set(
        dev_manifest.records,
        cache_dir=arguments.pose_cache,
        pose_fingerprint=config.pose_fingerprint,
        materialize_sequences=False,
    )
    if final_dev_snapshot.fingerprint != dev_snapshot.fingerprint:
        raise RuntimeError("dev84 pose caches changed during probe prediction")
    if {
        **_shared_encoder_input_hashes(arguments),
        "model": sha256_file(arguments.model),
        "dev_inputs": sha256_file(arguments.dev_inputs),
    } != initial_hashes:
        raise RuntimeError("a probe prediction input changed during execution")
    if _code_file_hashes() != initial_code_hashes:
        raise RuntimeError("probe source code changed during prediction")

    predictions_path = output_dir / "predictions.json"
    predictions_sha256 = _write_json_exclusive(predictions_path, predictions_payload)
    run_payload = {
        "schema_version": 1,
        "artifact_type": "pams_encoder_supervised_upper_bound_predict_run",
        "method_key": _METHOD_KEY,
        "status": _STATUS,
        "seed": _SEED,
        "probe_source_revision": probe_source_revision,
        "eligibility": predictions_payload["eligibility"],
        "label_access": predictions_payload["label_access"],
        "inputs": initial_hashes,
        "pose_cache_snapshot": dev_snapshot.to_dict(),
        "code_files_sha256": initial_code_hashes,
        "probe_checkout": _git_identity(),
        "runtime": _runtime_identity(device),
        "artifacts": {
            "predictions": {
                "path": predictions_path.name,
                "sha256": predictions_sha256,
            },
        },
        "test105_access": "none",
        "dev_target_reachable_by_predict_process": False,
    }
    run_path = output_dir / "predict-run.json"
    run_sha256 = _write_json_exclusive(run_path, run_payload)
    return {
        "status": _STATUS,
        "output_dir": str(output_dir),
        "predictions_sha256": predictions_sha256,
        "run_sha256": run_sha256,
        "dev_target_reachable": False,
        "test105_access": "none",
    }


def _validate_prediction_run(
    path: Path,
    *,
    predictions: Mapping[str, Any],
    prediction_sha256: str,
) -> tuple[dict[str, Any], str]:
    """Validate the target-free prediction receipt before targets are reachable."""

    prediction_run_sha256 = sha256_file(path)
    payload = _load_json_object(path, document_name="supervised probe prediction run")
    expected_fields = {
        "schema_version",
        "artifact_type",
        "method_key",
        "status",
        "seed",
        "probe_source_revision",
        "eligibility",
        "label_access",
        "inputs",
        "pose_cache_snapshot",
        "code_files_sha256",
        "probe_checkout",
        "runtime",
        "artifacts",
        "test105_access",
        "dev_target_reachable_by_predict_process",
    }
    if set(payload) != expected_fields:
        raise ValueError("supervised probe prediction-run fields changed")
    if (
        payload["schema_version"] != 1
        or payload["artifact_type"]
        != "pams_encoder_supervised_upper_bound_predict_run"
        or payload["method_key"] != _METHOD_KEY
        or payload["status"] != _STATUS
        or payload["seed"] != _SEED
        or payload["probe_source_revision"]
        != predictions["probe_source_revision"]
    ):
        raise ValueError("supervised probe prediction-run binding changed")
    if payload["eligibility"] != predictions["eligibility"]:
        raise ValueError("prediction-run eligibility differs from predictions")
    if payload["label_access"] != predictions["label_access"]:
        raise ValueError("prediction-run label access differs from predictions")
    if payload["code_files_sha256"] != predictions["code_files_sha256"]:
        raise ValueError("prediction-run code hashes differ from predictions")
    if payload["test105_access"] != "none":
        raise ValueError("prediction-run test105 access declaration changed")
    if payload["dev_target_reachable_by_predict_process"] is not False:
        raise ValueError("prediction-run must declare dev targets unreachable")

    artifacts = payload["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {"predictions"}:
        raise ValueError("prediction-run artifact roles changed")
    prediction_artifact = artifacts["predictions"]
    if (
        not isinstance(prediction_artifact, dict)
        or set(prediction_artifact) != {"path", "sha256"}
        or prediction_artifact["sha256"] != prediction_sha256
    ):
        raise ValueError("prediction-run predictions SHA-256 differs from actual file")

    inputs = payload["inputs"]
    expected_input_fields = {
        "config",
        "encoder_checkpoint",
        "encoder_progress",
        "encoder_completion_receipt",
        "model",
        "dev_inputs",
    }
    if not isinstance(inputs, dict) or set(inputs) != expected_input_fields:
        raise ValueError("prediction-run input hash fields changed")
    expected_bindings = {
        "config": predictions["config_file_sha256"],
        "encoder_checkpoint": predictions["encoder_checkpoint_sha256"],
        "encoder_progress": predictions["encoder_progress_sha256"],
        "encoder_completion_receipt": predictions[
            "encoder_completion_receipt_sha256"
        ],
        "model": predictions["model_sha256"],
        "dev_inputs": predictions["dev_inputs_sha256"],
    }
    if inputs != expected_bindings:
        raise ValueError("prediction-run input hashes differ from predictions")
    if payload["pose_cache_snapshot"] != predictions["dev_pose_cache_snapshot"]:
        raise ValueError("prediction-run pose-cache snapshot differs from predictions")
    if sha256_file(path) != prediction_run_sha256:
        raise RuntimeError("prediction-run changed while it was being validated")
    return payload, prediction_run_sha256


def score_probe(arguments: argparse.Namespace) -> dict[str, Any]:
    predictions_path = arguments.predictions.resolve(strict=True)
    prediction_run_path = arguments.prediction_run.resolve(strict=True)
    output_path = arguments.output.resolve(strict=False)
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite evaluation: {output_path}")
    prediction_sha256 = sha256_file(predictions_path)
    code_hashes = _code_file_hashes()
    probe_source_revision = _probe_source_revision()

    predictions = _load_json_object(
        predictions_path,
        document_name="supervised probe predictions",
    )
    expected_prediction_fields = {
        "schema_version",
        "artifact_type",
        "method_key",
        "status",
        "protocol",
        "split",
        "seed",
        "probe_source_revision",
        "eligibility",
        "label_access",
        "feature_spec",
        "feature_names_sha256",
        "model_sha256",
        "encoder_checkpoint_sha256",
        "encoder_completion_receipt_sha256",
        "encoder_progress_sha256",
        "config_file_sha256",
        "config_fingerprint",
        "dev_inputs_sha256",
        "dev_feature_matrix_sha256",
        "dev_pose_cache_snapshot",
        "code_files_sha256",
        "predictions",
    }
    if set(predictions) != expected_prediction_fields:
        raise ValueError("supervised probe prediction artifact fields changed")
    if (
        predictions["schema_version"] != 1
        or predictions["artifact_type"]
        != "pams_encoder_supervised_upper_bound_predictions"
        or predictions["method_key"] != _METHOD_KEY
        or predictions["status"] != _STATUS
        or predictions["protocol"] != "ucfrep_526"
        or predictions["split"] != "dev"
        or predictions["seed"] != _SEED
    ):
        raise ValueError("supervised probe prediction binding changed")
    _require_code_hash_binding(
        predictions["code_files_sha256"],
        code_hashes,
        stage="probe predictions",
    )
    _require_source_revision_binding(
        predictions["probe_source_revision"],
        probe_source_revision,
        stage="probe predictions",
    )
    expected_eligibility = {
        "pams_main_result": False,
        "self_supervised_result": False,
        "baseline_result": False,
        "classification": "diagnostic_supervised_encoder_upper_bound",
    }
    if predictions["eligibility"] != expected_eligibility:
        raise ValueError("supervised probe eligibility declaration changed")
    if predictions["feature_spec"] != _FEATURE_SPEC:
        raise ValueError("supervised probe feature specification changed")
    expected_feature_names_sha256 = _sha256_bytes(
        json.dumps(_feature_names(), separators=(",", ":")).encode("utf-8")
    )
    if predictions["feature_names_sha256"] != expected_feature_names_sha256:
        raise ValueError("supervised probe feature-name binding changed")
    label_access = predictions["label_access"]
    if not isinstance(label_access, dict) or label_access != {
        "train337_counts": True,
        "train337_actions": False,
        "dev84_counts_during_run": False,
        "dev84_actions_during_run": False,
        "sealed_test105_identity_or_labels": False,
    }:
        raise ValueError("supervised probe label-access declaration changed")
    rows = predictions["predictions"]
    if not isinstance(rows, list) or len(rows) != _EXPECTED_DEV:
        raise ValueError("supervised probe requires exactly 84 dev predictions")
    if any(
        not isinstance(row, dict)
        or set(row) != {"video_id", "raw_count", "rounded_count"}
        for row in rows
    ):
        raise ValueError("supervised probe prediction row fields changed")

    _, prediction_run_sha256 = _validate_prediction_run(
        prediction_run_path,
        predictions=predictions,
        prediction_sha256=prediction_sha256,
    )

    # The dev target path is deliberately not resolved, opened, or hashed
    # until both the prediction artifact and its target-free run receipt have
    # passed every binding check above.
    targets_path = arguments.dev_targets.resolve(strict=True)
    target_sha256 = sha256_file(targets_path)
    _require_frozen_target_sha256(target_sha256, split="dev")
    targets = load_dev_target_manifest(targets_path)
    prediction_ids = tuple(str(row["video_id"]) for row in rows)
    target_ids = tuple(record.video_id for record in targets.records)
    if prediction_ids != target_ids:
        raise ValueError("dev84 prediction and target membership/order differ")
    raw_predictions = np.asarray([row["raw_count"] for row in rows], dtype=np.float64)
    if not np.isfinite(raw_predictions).all() or np.any(raw_predictions < 0):
        raise ValueError("supervised probe predictions must be finite and non-negative")
    if any(
        row["rounded_count"] != round_count(float(row["raw_count"]))
        for row in rows
    ):
        raise ValueError("stored rounded predictions do not match frozen half-up rounding")
    target_values = np.asarray(
        [record.count for record in targets.records],
        dtype=np.float64,
    )
    report = compute_count_metrics(
        raw_predictions,
        target_values,
        video_ids=prediction_ids,
        actions=[record.action for record in targets.records],
        bootstrap_samples=10_000,
        bootstrap_seed=_SEED,
    )
    if sha256_file(predictions_path) != prediction_sha256:
        raise RuntimeError("probe predictions changed during scoring")
    if sha256_file(prediction_run_path) != prediction_run_sha256:
        raise RuntimeError("prediction-run changed during scoring")
    if sha256_file(targets_path) != target_sha256:
        raise RuntimeError("dev84 targets changed during scoring")
    if _code_file_hashes() != code_hashes:
        raise RuntimeError("probe scoring source changed during execution")

    raw_error = raw_predictions - target_values
    evaluation = {
        "schema_version": 1,
        "artifact_type": "pams_encoder_supervised_upper_bound_evaluation",
        "method_key": _METHOD_KEY,
        "status": _STATUS,
        "probe_source_revision": probe_source_revision,
        "eligibility": predictions["eligibility"],
        "label_access": {
            **label_access,
            "dev84_counts_by_separate_scorer": True,
            "dev84_actions_by_separate_scorer": True,
        },
        "prediction_sha256": prediction_sha256,
        "prediction_run_sha256": prediction_run_sha256,
        "dev_targets_sha256": target_sha256,
        "encoder_checkpoint_sha256": predictions["encoder_checkpoint_sha256"],
        "encoder_completion_receipt_sha256": predictions[
            "encoder_completion_receipt_sha256"
        ],
        "metrics": report.to_dict(),
        "raw_metrics": {
            "mae": float(np.mean(np.abs(raw_error))),
            "rmse": float(np.sqrt(np.mean(np.square(raw_error)))),
            "nmae": float(np.mean(np.abs(raw_error) / target_values)),
        },
        "code_files_sha256": code_hashes,
        "sealed_test105_access": "none",
        "claim": "diagnostic_supervised_upper_bound_only",
    }
    evaluation_sha256 = _write_json_exclusive(output_path, evaluation)
    return {
        "status": _STATUS,
        "evaluation": str(output_path),
        "evaluation_sha256": evaluation_sha256,
        "metrics": {
            "nmae": report.nmae,
            "mae": report.mae,
            "rmse": report.rmse,
            "obo": report.obo,
            "exact": report.exact,
        },
        "sealed_test105_access": "none",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser(
        "train",
        help="fit the frozen probe from train337 counts only",
    )
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--encoder-checkpoint", type=Path, required=True)
    train.add_argument("--encoder-progress", type=Path, required=True)
    train.add_argument("--encoder-completion-receipt", type=Path, required=True)
    train.add_argument("--expected-encoder-source-git-sha", required=True)
    train.add_argument("--train-inputs", type=Path, required=True)
    train.add_argument("--train-targets", type=Path, required=True)
    train.add_argument("--pose-cache", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--feature-batch-size", type=int, default=8)
    train.add_argument("--device", default="cuda")

    predict = subparsers.add_parser(
        "predict",
        help="apply a frozen probe to label-free dev84",
    )
    predict.add_argument("--config", type=Path, required=True)
    predict.add_argument("--encoder-checkpoint", type=Path, required=True)
    predict.add_argument("--encoder-progress", type=Path, required=True)
    predict.add_argument("--encoder-completion-receipt", type=Path, required=True)
    predict.add_argument("--expected-encoder-source-git-sha", required=True)
    predict.add_argument("--model", type=Path, required=True)
    predict.add_argument("--dev-inputs", type=Path, required=True)
    predict.add_argument("--pose-cache", type=Path, required=True)
    predict.add_argument("--output-dir", type=Path, required=True)
    predict.add_argument("--feature-batch-size", type=int, default=8)
    predict.add_argument("--device", default="cuda")

    score = subparsers.add_parser(
        "score",
        help="score frozen predictions in a separate dev-target process",
    )
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--prediction-run", type=Path, required=True)
    score.add_argument("--dev-targets", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "train":
        result = train_probe(arguments)
    elif arguments.command == "predict":
        result = predict_probe(arguments)
    else:
        result = score_probe(arguments)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
