"""Measure the best count-blind periodic-template reconstruction on natural data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pams.temporac.certify_adaptive_v4 import stable_pair_integer_landmarks
from pams.temporac.preprocess_resampled_v2 import preprocess_resampled_identity
from pams.warp_phase.packing import trusted_load_source_pickle


def _periodic_template(
    target: np.ndarray,
    mask: np.ndarray,
    geometry: np.ndarray,
    landmarks: np.ndarray,
) -> np.ndarray:
    grid = np.linspace(0.0, 1.0, 513, dtype=np.float64)
    numerator = np.zeros((grid.size, target.shape[1]), dtype=np.float64)
    denominator = np.zeros_like(numerator)
    for left_raw, right_raw in zip(landmarks[:-1], landmarks[1:], strict=True):
        left, right = int(left_raw), int(right_raw)
        delta = geometry[left + 1 : right + 1] - geometry[left:right]
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(delta, axis=1))))
        arc /= arc[-1]
        for channel in range(target.shape[1]):
            valid = mask[left : right + 1, channel] > 0
            if np.count_nonzero(valid) < 2:
                continue
            values = np.interp(grid, arc[valid], target[left : right + 1, channel][valid])
            numerator[:, channel] += values
            denominator[:, channel] += 1.0
    template = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    template[-1] = template[0] = 0.5 * (template[0] + template[-1])
    reconstruction = np.zeros_like(target, dtype=np.float64)
    for left_raw, right_raw in zip(landmarks[:-1], landmarks[1:], strict=True):
        left, right = int(left_raw), int(right_raw)
        delta = geometry[left + 1 : right + 1] - geometry[left:right]
        arc = np.concatenate(([0.0], np.cumsum(np.linalg.norm(delta, axis=1))))
        arc /= arc[-1]
        for channel in range(target.shape[1]):
            reconstruction[left : right + 1, channel] = np.interp(
                arc,
                grid,
                template[:, channel],
            )
    return reconstruction


def _certificate_huber(
    target: np.ndarray,
    reconstruction: np.ndarray,
    mask: np.ndarray,
    landmarks: np.ndarray,
) -> float:
    values: list[float] = []
    for left_raw, right_raw in zip(landmarks[:-1], landmarks[1:], strict=True):
        left, right = int(left_raw), int(right_raw)
        error = np.abs(reconstruction[left + 1 : right] - target[left + 1 : right])
        local_mask = mask[left + 1 : right]
        huber = np.where(error <= 0.05, 0.5 * np.square(error) / 0.05, error - 0.025)
        values.append(float(np.sum(huber * local_mask) / np.sum(local_mask)))
    return max(values)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    source = args.repository / "data/counting_multirep_skeleton_pose_expanded_v44_len320/train.pkl"
    decoded, _receipt = trusted_load_source_pickle(
        source,
        repository_root=args.repository,
        split="train",
    )
    rows: list[dict[str, object]] = []
    for video_index, sample in enumerate(decoded["train"]):
        for person_index in range(len(sample["person_mask"])):
            if not bool(sample["person_mask"][person_index]):
                continue
            prepared, _ = preprocess_resampled_identity(
                sample["motion"][person_index],
                sample["frame_mask"][person_index],
                sample["sampled_frame_indices"],
            )
            bounds = np.array(prepared.run_bounds, dtype=np.int32, copy=True)
            bounds[:, 1] += 1
            result = stable_pair_integer_landmarks(prepared.geometry_features.geometry, bounds)
            if result.landmarks.size < 3:
                continue
            target = np.asarray(prepared.teacher_input[:, :149], dtype=np.float64)
            geometry_features = prepared.geometry_features
            mask = np.concatenate(
                (
                    geometry_features.coordinate_mask,
                    np.repeat(geometry_features.direction_element_mask, 2, axis=1),
                    prepared.joint_mask,
                ),
                axis=1,
            ).astype(np.float64)
            reconstruction = _periodic_template(
                target,
                mask,
                geometry_features.geometry,
                result.landmarks,
            )
            loss = _certificate_huber(target, reconstruction, mask, result.landmarks)
            best_pair_loss = loss
            best_pair_landmarks = result.landmarks
            if result.landmarks.size > 3:
                for index in range(result.landmarks.size - 2):
                    candidate = result.landmarks[index : index + 3]
                    candidate_reconstruction = _periodic_template(
                        target,
                        mask,
                        geometry_features.geometry,
                        candidate,
                    )
                    candidate_loss = _certificate_huber(
                        target,
                        candidate_reconstruction,
                        mask,
                        candidate,
                    )
                    if candidate_loss < best_pair_loss:
                        best_pair_loss = candidate_loss
                        best_pair_landmarks = candidate
            rows.append(
                {
                    "best_pair_landmarks": best_pair_landmarks.tolist(),
                    "best_pair_oracle_huber": best_pair_loss,
                    "best_pair_passes": best_pair_loss <= 0.02,
                    "landmarks": int(result.landmarks.size),
                    "estimated_period": result.estimated_period,
                    "lag_score": result.lag_score,
                    "pair_geometry_huber": result.pair_geometry_huber,
                    "oracle_max_traversal_huber": loss,
                    "passes_reconstruction": loss <= 0.02,
                    "person_index": person_index,
                    "video_index": video_index,
                }
            )
            if len(rows) >= args.limit:
                break
        if len(rows) >= args.limit:
            break
    losses = [float(row["oracle_max_traversal_huber"]) for row in rows]
    print(
        json.dumps(
            {
                "identities": len(rows),
                "best_pair_passes": sum(int(row["best_pair_passes"]) for row in rows),
                "mean_oracle_max_traversal_huber": float(np.mean(losses)),
                "passes_reconstruction": sum(
                    int(row["passes_reconstruction"]) for row in rows
                ),
                "rows": rows,
                "schema": "temporac.natural-oracle-diagnostic.v1",
                "worst_oracle_max_traversal_huber": max(losses),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
