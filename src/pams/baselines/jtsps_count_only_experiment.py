"""Auditable inferred clean-room JTSPS count-only train/dev experiment.

This module deliberately does not claim source parity.  It closes the
undisclosed JTSPS supervision/decoding details with count-only supervision,
records those closures in every artifact, and keeps development targets out of
the training and prediction process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from pams.baselines.pose_cleanroom import JTSPSCountOnly
from pams.data import load_pose_cache_set, load_pose_input_manifest
from pams.metrics import compute_count_metrics
from pams.reproducibility import sha256_file
from pams.ucfrep import build_official_manifest

_CLOSURES = {
    "status": "inferred-clean-room-not-source-parity",
    "frames": 64,
    "joint_embedding_dim": 16,
    "hidden_dim": 32,
    "decoder": "sum of non-negative density",
    "supervision": "video-level count only",
    "loss": "log-count SmoothL1 + relative L1 + impulse auxiliary + density TV",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def _load_train_targets(path: Path, expected_ids: tuple[str, ...]) -> Tensor:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if set(raw) != {
        "manifest_type",
        "protocol",
        "records",
        "schema_version",
        "split",
        "source_annotation_sha256",
    }:
        raise ValueError("training target sidecar fields changed")
    if (
        raw["manifest_type"] != "train_count_targets"
        or raw["protocol"] != "ucfrep_526"
        or raw["split"] != "train"
        or raw["schema_version"] != 1
    ):
        raise ValueError("training target sidecar protocol binding changed")
    rows = raw["records"]
    if not isinstance(rows, list):
        raise TypeError("training target records must be a list")
    if any(set(row) != {"count", "video_id"} for row in rows):
        raise ValueError("training target records may contain only video_id/count")
    identifiers = tuple(str(row["video_id"]) for row in rows)
    if identifiers != expected_ids or len(set(identifiers)) != len(identifiers):
        raise ValueError("training targets do not exactly match ordered train inputs")
    counts = [row["count"] for row in rows]
    if any(isinstance(value, bool) or int(value) != value or int(value) <= 0 for value in counts):
        raise ValueError("training counts must be positive integers")
    return torch.tensor(counts, dtype=torch.float32)


def _sample_pose(sequence: Any, frames: int) -> tuple[Tensor, Tensor]:
    indices = np.rint(np.linspace(0, sequence.num_frames - 1, frames)).astype(np.int64)
    pose = torch.from_numpy(np.array(sequence.xyz[indices], copy=True))
    mask = torch.from_numpy(np.array(sequence.valid_mask[indices], copy=True))
    return pose, mask


def _batch(
    poses: Tensor,
    masks: Tensor,
    targets: Tensor,
    indices: Tensor,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    return (
        poses.index_select(0, indices).to(device, non_blocking=True),
        masks.index_select(0, indices).to(device, non_blocking=True),
        targets.index_select(0, indices).to(device, non_blocking=True),
    )


def _count_loss(output: Any, targets: Tensor, masks: Tensor) -> tuple[Tensor, dict[str, float]]:
    validity = masks.to(dtype=output.density.dtype)
    density_count = (output.density * validity).sum(dim=1).clamp_min(1e-4)
    impulse_count = (torch.sigmoid(output.impulse_logits) * validity).sum(dim=1).clamp_min(
        1e-4
    )
    log_density = F.smooth_l1_loss(torch.log1p(density_count), torch.log1p(targets))
    relative = (torch.abs(density_count - targets) / targets).mean()
    log_impulse = F.smooth_l1_loss(torch.log1p(impulse_count), torch.log1p(targets))
    pair_mask = masks[:, 1:] & masks[:, :-1]
    differences = torch.abs(output.density[:, 1:] - output.density[:, :-1])
    tv = (differences * pair_mask).sum() / pair_mask.sum().clamp_min(1)
    loss = log_density + 0.25 * relative + 0.1 * log_impulse + 0.001 * tv
    return loss, {
        "loss": float(loss.detach()),
        "density_count": float(density_count.detach().mean()),
        "relative": float(relative.detach()),
    }


def run_experiment(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if arguments.epochs < 1 or arguments.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(arguments.seed)

    train_inputs_path = arguments.train_inputs.resolve(strict=True)
    dev_inputs_path = arguments.dev_inputs.resolve(strict=True)
    train_targets_path = arguments.train_targets.resolve(strict=True)
    output_dir = arguments.output_dir.resolve(strict=False)
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    output_dir.mkdir(parents=True)

    train_inputs = load_pose_input_manifest(train_inputs_path, validate_exact=True)
    dev_inputs = load_pose_input_manifest(dev_inputs_path, validate_exact=True)
    if train_inputs.split != "train" or dev_inputs.split != "dev":
        raise ValueError("expected canonical train337 and dev84 sidecars")
    train_ids = tuple(record.video_id for record in train_inputs.records)
    dev_ids = tuple(record.video_id for record in dev_inputs.records)
    if len(train_ids) != 337 or len(dev_ids) != 84 or set(train_ids) & set(dev_ids):
        raise ValueError("experiment requires disjoint canonical 337/84 membership")
    targets = _load_train_targets(train_targets_path, train_ids)

    train_sequences, train_snapshot = load_pose_cache_set(
        train_inputs.records,
        cache_dir=arguments.pose_cache,
        pose_fingerprint=arguments.pose_fingerprint,
    )
    dev_sequences, dev_snapshot = load_pose_cache_set(
        dev_inputs.records,
        cache_dir=arguments.pose_cache,
        pose_fingerprint=arguments.pose_fingerprint,
    )
    train_samples = [_sample_pose(sequence, _CLOSURES["frames"]) for sequence in train_sequences]
    dev_samples = [_sample_pose(sequence, _CLOSURES["frames"]) for sequence in dev_sequences]
    train_poses = torch.stack([row[0] for row in train_samples])
    train_masks = torch.stack([row[1] for row in train_samples])
    dev_poses = torch.stack([row[0] for row in dev_samples])
    dev_masks = torch.stack([row[1] for row in dev_samples])

    device = torch.device(arguments.device)
    model = JTSPSCountOnly(
        joint_embedding_dim=_CLOSURES["joint_embedding_dim"],
        hidden_dim=_CLOSURES["hidden_dim"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.lr, weight_decay=1e-4)
    history: list[dict[str, float | int]] = []
    generator = torch.Generator().manual_seed(arguments.seed)
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        permutation = torch.randperm(len(train_ids), generator=generator)
        totals = {"loss": 0.0, "density_count": 0.0, "relative": 0.0}
        steps = 0
        for start in range(0, len(train_ids), arguments.batch_size):
            indices = permutation[start : start + arguments.batch_size]
            pose, mask, target = _batch(
                train_poses, train_masks, targets, indices, device
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(pose, mask)
            loss, values = _count_loss(output, target, mask)
            if not bool(torch.isfinite(loss).item()):
                raise FloatingPointError("non-finite JTSPS training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            for key, value in values.items():
                totals[key] += value
            steps += 1
        row = {"epoch": epoch, **{key: value / steps for key, value in totals.items()}}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    checkpoint_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "schema_version": 1,
            "method": "jtsps-count-only",
            "status": _CLOSURES["status"],
            "closures": _CLOSURES,
            "seed": arguments.seed,
            "epochs": arguments.epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "train_inputs_sha256": _sha256(train_inputs_path),
            "train_targets_sha256": _sha256(train_targets_path),
            "train_pose_snapshot": train_snapshot.to_dict(),
        },
        checkpoint_path,
    )

    model.eval()
    raw_counts: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(dev_ids), arguments.batch_size):
            pose = dev_poses[start : start + arguments.batch_size].to(device)
            mask = dev_masks[start : start + arguments.batch_size].to(device)
            output = model(pose, mask)
            values = (output.density * mask).sum(dim=1)
            if not bool(torch.isfinite(values).all().item()):
                raise FloatingPointError("non-finite JTSPS dev prediction")
            raw_counts.extend(float(value) for value in values.cpu())
    rows = [
        {
            "video_id": video_id,
            "raw_count": value,
            "rounded_count": int(math.floor(value + 0.5)),
        }
        for video_id, value in zip(dev_ids, raw_counts, strict=True)
    ]
    predictions = {
        "schema_version": 1,
        "method": "jtsps-count-only",
        "status": _CLOSURES["status"],
        "protocol": "ucfrep_526",
        "split": "dev",
        "seed": arguments.seed,
        "closures": _CLOSURES,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "dev_inputs_sha256": _sha256(dev_inputs_path),
        "dev_pose_snapshot": dev_snapshot.to_dict(),
        "predictions": rows,
    }
    _write_json_exclusive(output_dir / "predictions.json", predictions)
    _write_json_exclusive(
        output_dir / "run.json",
        {
            "schema_version": 1,
            "method": "jtsps-count-only",
            "status": _CLOSURES["status"],
            "seed": arguments.seed,
            "epochs": arguments.epochs,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.lr,
            "closures": _CLOSURES,
            "artifacts": {
                "checkpoint": {
                    "path": "checkpoint.pt",
                    "sha256": _sha256(checkpoint_path),
                },
                "predictions": {
                    "path": "predictions.json",
                    "sha256": _sha256(output_dir / "predictions.json"),
                },
            },
            "inputs": {
                "train_inputs_sha256": _sha256(train_inputs_path),
                "train_targets_sha256": _sha256(train_targets_path),
                "dev_inputs_sha256": _sha256(dev_inputs_path),
                "pose_fingerprint": arguments.pose_fingerprint,
            },
            "history": history,
            "dev_target_reachable": False,
            "test_identity_reachable": False,
        },
    )
    return predictions


def score_predictions(arguments: argparse.Namespace) -> dict[str, Any]:
    prediction_path = arguments.predictions.resolve(strict=True)
    targets_path = arguments.dev_targets.resolve(strict=True)
    output_path = arguments.output.resolve(strict=False)
    raw = json.loads(prediction_path.read_text(encoding="utf-8"))
    if raw.get("method") != "jtsps-count-only" or raw.get("split") != "dev":
        raise ValueError("prediction artifact binding changed")
    rows = raw["predictions"]
    targets_raw = json.loads(targets_path.read_text(encoding="utf-8"))
    if (
        targets_raw.get("manifest_type") != "dev_targets"
        or targets_raw.get("protocol") != "ucfrep_526"
        or targets_raw.get("split") != "dev"
    ):
        raise ValueError("expected canonical UCFRep dev target sidecar")
    target_rows = targets_raw["records"]
    prediction_ids = tuple(row["video_id"] for row in rows)
    target_ids = tuple(row["video_id"] for row in target_rows)
    if len(prediction_ids) != 84 or prediction_ids != target_ids:
        raise ValueError("dev prediction/target membership or order changed")
    report = compute_count_metrics(
        [row["raw_count"] for row in rows],
        [row["count"] for row in target_rows],
        video_ids=prediction_ids,
        actions=[row["action"] for row in target_rows],
        bootstrap_samples=10_000,
        bootstrap_seed=2026,
    )
    raw_prediction = np.asarray([row["raw_count"] for row in rows], dtype=np.float64)
    target = np.asarray([row["count"] for row in target_rows], dtype=np.float64)
    result = {
        "schema_version": 1,
        "method": "jtsps-count-only",
        "status": raw["status"],
        "prediction_sha256": _sha256(prediction_path),
        "dev_targets_sha256": _sha256(targets_path),
        "metrics": report.to_dict(),
        "raw_metrics": {
            "mae": float(np.mean(np.abs(raw_prediction - target))),
            "rmse": float(np.sqrt(np.mean(np.square(raw_prediction - target)))),
            "nmae": float(np.mean(np.abs(raw_prediction - target) / target)),
        },
    }
    _write_json_exclusive(output_path, result)
    return result


def prepare_train_targets(arguments: argparse.Namespace) -> dict[str, Any]:
    """Materialize the only label-bearing input exposed to formal training."""

    train_inputs_path = arguments.train_inputs.resolve(strict=True)
    annotation_path = arguments.annotations.resolve(strict=True)
    output_path = arguments.output.resolve(strict=False)
    train_inputs = load_pose_input_manifest(train_inputs_path, validate_exact=True)
    if train_inputs.split != "train" or len(train_inputs.records) != 337:
        raise ValueError("target preparation requires canonical train337 inputs")
    manifest = build_official_manifest(
        annotation_path,
        video_root=arguments.video_root,
        hash_existing_videos=False,
        allow_missing_videos=True,
    )
    counts = {record.video_id: record.count for record in manifest.records}
    identifiers = tuple(record.video_id for record in train_inputs.records)
    if any(identifier not in counts for identifier in identifiers):
        raise ValueError("an official train337 identifier is absent from annotations")
    payload = {
        "schema_version": 1,
        "manifest_type": "train_count_targets",
        "protocol": "ucfrep_526",
        "split": "train",
        "source_annotation_sha256": sha256_file(annotation_path),
        "records": [
            {"video_id": identifier, "count": counts[identifier]}
            for identifier in identifiers
        ],
    }
    _write_json_exclusive(output_path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("run")
    train.add_argument("--train-inputs", type=Path, required=True)
    train.add_argument("--train-targets", type=Path, required=True)
    train.add_argument("--dev-inputs", type=Path, required=True)
    train.add_argument("--pose-cache", type=Path, required=True)
    train.add_argument("--pose-fingerprint", required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--device", default="cuda")
    score = subparsers.add_parser("score")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--dev-targets", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    prepare = subparsers.add_parser("prepare-targets")
    prepare.add_argument("--train-inputs", type=Path, required=True)
    prepare.add_argument("--annotations", type=Path, required=True)
    prepare.add_argument("--video-root", type=Path, default=Path("/nonexistent"))
    prepare.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    if arguments.command == "run":
        result = run_experiment(arguments)
    elif arguments.command == "score":
        result = score_predictions(arguments)
    else:
        result = prepare_train_targets(arguments)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
