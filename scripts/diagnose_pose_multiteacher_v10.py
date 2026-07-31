"""Dev-only audit of label-free pose-based soft frequency teachers."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

COUNT_MIN = 1
COUNT_MAX = 40
KEY_JOINTS = (13, 14, 15, 16, 25, 26, 27, 28)


def _cache_path(cache_dir: Path, video_id: str) -> Path:
    digest = hashlib.sha256(video_id.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npz"


def _soft_count_distribution(signal: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(signal, dtype=np.float64)
    if values.size < 16 or float(np.std(values)) < 1e-8:
        return np.zeros(COUNT_MAX, dtype=np.float64), 0.0
    time = np.arange(values.size, dtype=np.float64)
    slope, intercept = np.polyfit(time, values, deg=1)
    detrended = values - (slope * time + intercept)
    power = np.abs(np.fft.rfft(detrended * np.hanning(values.size))) ** 2
    distribution = np.zeros(COUNT_MAX, dtype=np.float64)
    for count in range(COUNT_MIN, COUNT_MAX + 1):
        center = count
        low = max(1, center - 1)
        high = min(power.size, center + 2)
        distribution[count - 1] = float(power[low:high].sum())
    total = float(distribution.sum())
    if total <= 1e-12:
        return distribution, 0.0
    distribution /= total
    entropy = -float(
        np.sum(distribution * np.log(distribution.clip(min=1e-12)))
    )
    confidence = 1.0 - entropy / np.log(COUNT_MAX)
    return distribution, max(0.0, confidence)


def _signals(xyz: np.ndarray) -> dict[str, np.ndarray]:
    flattened = xyz.reshape(xyz.shape[0], -1)
    velocity = np.diff(flattened, axis=0, prepend=flattened[:1])
    centered_position = flattened - flattened.mean(axis=0, keepdims=True)
    centered_velocity = velocity - velocity.mean(axis=0, keepdims=True)
    outputs: dict[str, np.ndarray] = {
        "global_velocity_energy": np.linalg.norm(velocity, axis=1),
        "global_position_energy": np.linalg.norm(centered_position, axis=1),
    }
    for prefix, matrix in (
        ("position_pca", centered_position),
        ("velocity_pca", centered_velocity),
    ):
        _, _, right = np.linalg.svd(matrix, full_matrices=False)
        projected = matrix @ right[:3].T
        for index in range(projected.shape[1]):
            outputs[f"{prefix}_{index + 1}"] = projected[:, index]
    joint_velocity = np.diff(xyz, axis=0, prepend=xyz[:1])
    for joint in KEY_JOINTS:
        outputs[f"joint_velocity_{joint}"] = np.linalg.norm(
            joint_velocity[:, joint],
            axis=-1,
        )
        for coordinate in range(3):
            outputs[f"joint_{joint}_{coordinate}"] = xyz[:, joint, coordinate]
    return outputs


def _decode(xyz: np.ndarray) -> dict[str, int]:
    if xyz.shape[0] < 16:
        return {
            "highest_confidence": 0,
            "weighted_arithmetic": 0,
            "weighted_geometric": 0,
            "median_distribution": 0,
        }
    distributions = []
    confidences = []
    names = []
    for name, signal in _signals(xyz).items():
        distribution, confidence = _soft_count_distribution(signal)
        if confidence > 0:
            names.append(name)
            distributions.append(distribution)
            confidences.append(confidence)
    if not distributions:
        return {
            "highest_confidence": 0,
            "weighted_arithmetic": 0,
            "weighted_geometric": 0,
            "median_distribution": 0,
        }
    matrix = np.stack(distributions)
    weights = np.asarray(confidences, dtype=np.float64) ** 2
    highest = matrix[int(np.argmax(weights))]
    arithmetic = np.average(matrix, axis=0, weights=weights)
    geometric = np.exp(
        np.average(np.log(matrix.clip(min=1e-8)), axis=0, weights=weights)
    )
    median = np.median(matrix, axis=0)
    decoded = {
        "highest_confidence": int(np.argmax(highest) + 1),
        "weighted_arithmetic": int(np.argmax(arithmetic) + 1),
        "weighted_geometric": int(np.argmax(geometric) + 1),
        "median_distribution": int(np.argmax(median) + 1),
    }

    def add_group(group_name: str, selected_names: set[str]) -> None:
        indices = [idx for idx, name in enumerate(names) if name in selected_names]
        if not indices:
            return
        group_matrix = matrix[indices]
        group_weights = weights[indices]
        group_arithmetic = np.average(
            group_matrix, axis=0, weights=group_weights
        )
        group_median = np.median(group_matrix, axis=0)
        group_highest = group_matrix[int(np.argmax(group_weights))]
        decoded[f"group:{group_name}:weighted"] = int(
            np.argmax(group_arithmetic) + 1
        )
        decoded[f"group:{group_name}:median"] = int(np.argmax(group_median) + 1)
        decoded[f"group:{group_name}:highest_confidence"] = int(
            np.argmax(group_highest) + 1
        )

    add_group(
        "right_limb_x",
        {f"joint_{joint}_0" for joint in (14, 16, 26, 28)},
    )
    add_group(
        "all_limb_x",
        {f"joint_{joint}_0" for joint in (13, 14, 15, 16, 25, 26, 27, 28)},
    )
    add_group(
        "distal_x",
        {f"joint_{joint}_0" for joint in (15, 16, 27, 28)},
    )
    decoded.update(
        {
            f"teacher:{name}": int(np.argmax(distribution) + 1)
            for name, distribution in zip(names, distributions, strict=True)
        }
    )
    return decoded


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    error = np.abs(prediction - truth)
    return {
        "nmae": float(np.mean(error / truth)),
        "obo": float(np.mean(error <= 1)),
        "exact": float(np.mean(error == 0)),
        "mae": float(np.mean(error)),
        "rmse": float(np.sqrt(np.mean(error**2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dev_inputs", type=Path)
    parser.add_argument("cache_dir", type=Path)
    parser.add_argument("targets", type=Path)
    args = parser.parse_args()
    inputs = json.loads(args.dev_inputs.read_text())["records"]
    target_rows = json.loads(args.targets.read_text())["records"]
    target_by_id = {row["video_id"]: int(row["count"]) for row in target_rows}
    predictions: dict[str, list[int]] = {}
    targets: list[int] = []
    for row in inputs:
        video_id = str(row["video_id"])
        with np.load(_cache_path(args.cache_dir, video_id), allow_pickle=False) as cache:
            xyz = np.asarray(cache["xyz"], dtype=np.float64)
            valid = np.asarray(cache["valid_mask"], dtype=bool)
        decoded = _decode(xyz[valid])
        current_index = len(targets)
        for name in predictions.keys() - decoded.keys():
            predictions[name].append(0)
        for name, count in decoded.items():
            if name not in predictions:
                predictions[name] = [0] * current_index
            predictions[name].append(count)
        targets.append(target_by_id[video_id])
    truth = np.asarray(targets)
    summaries = {
                name: {
                    "metrics": _metrics(np.asarray(counts), truth),
                    "zeros": int(np.sum(np.asarray(counts) == 0)),
                }
                for name, counts in predictions.items()
            }
    ranked = sorted(
        summaries.items(),
        key=lambda item: (item[1]["metrics"]["nmae"], -item[1]["metrics"]["obo"]),
    )
    print(json.dumps(dict(ranked), indent=2))


if __name__ == "__main__":
    main()
