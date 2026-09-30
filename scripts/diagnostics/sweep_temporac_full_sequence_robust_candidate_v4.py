"""Train-only feasibility sweep for complete-chain robust reconstruction gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pams.temporac.full_chain_landmarks_candidate_v1 import full_chain_integer_landmarks
from pams.temporac.multi_cycle_dtw_template_candidate_v2 import (
    multi_cycle_dtw_median_reconstruction,
)
from pams.temporac.preprocess_resampled_v2 import preprocess_resampled_identity
from pams.warp_phase.packing import trusted_load_source_pickle


def _traversal_huber(
    target: np.ndarray,
    reconstruction: np.ndarray,
    mask: np.ndarray,
    landmarks: np.ndarray,
) -> np.ndarray:
    values: list[float] = []
    for left_raw, right_raw in zip(landmarks[:-1], landmarks[1:], strict=True):
        left, right = int(left_raw), int(right_raw)
        error = np.abs(reconstruction[left + 1 : right] - target[left + 1 : right])
        local_mask = mask[left + 1 : right]
        huber = np.where(error <= 0.05, 0.5 * np.square(error) / 0.05, error - 0.025)
        values.append(float(np.sum(huber * local_mask) / np.sum(local_mask)))
    return np.asarray(values, dtype=np.float64)


def _structural_gates(geometry: np.ndarray, landmarks: np.ndarray) -> tuple[bool, bool]:
    starts = geometry[landmarks[:-1]]
    median = np.median(starts, axis=0)
    origin = bool(
        np.all(np.linalg.norm(starts - median, axis=1) / np.sqrt(66.0) <= 0.05)
    )
    topology = True
    for left_raw, right_raw in zip(landmarks[:-1], landmarks[1:], strict=True):
        left, right = int(left_raw), int(right_raw)
        arc = np.linalg.norm(
            geometry[left + 1 : right + 1] - geometry[left:right],
            axis=1,
        )
        topology &= bool(
            arc.size >= 6
            and np.count_nonzero(arc > 0.0) >= 5
            and float(np.max(arc) / np.sum(arc)) < 0.25
        )
    return origin, topology


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args()
    source = args.repository / "data/counting_multirep_skeleton_pose_expanded_v44_len320/train.pkl"
    decoded, _receipt = trusted_load_source_pickle(
        source,
        repository_root=args.repository,
        split="train",
    )
    total = 0
    base_eligible = 0
    refined_eligible = 0
    structural_eligible = 0
    retained_event_fractions: list[float] = []
    rows: list[dict[str, float | int]] = []
    for sample in decoded["train"]:
        for person_index in range(len(sample["person_mask"])):
            if not bool(sample["person_mask"][person_index]):
                continue
            total += 1
            prepared, _ = preprocess_resampled_identity(
                sample["motion"][person_index],
                sample["frame_mask"][person_index],
                sample["sampled_frame_indices"],
            )
            bounds = np.array(prepared.run_bounds, dtype=np.int32, copy=True)
            bounds[:, 1] += 1
            result = full_chain_integer_landmarks(
                prepared.geometry_features.geometry,
                bounds,
            )
            if result.base_landmarks >= 3:
                base_eligible += 1
            if result.landmarks.size < 3:
                continue
            refined_eligible += 1
            retained_event_fractions.append(result.retained_event_fraction)
            origin, topology = _structural_gates(
                prepared.geometry_features.geometry,
                result.landmarks,
            )
            if not origin or not topology:
                continue
            structural_eligible += 1
            geometry_features = prepared.geometry_features
            target = np.asarray(prepared.teacher_input[:, :149], dtype=np.float64)
            mask = np.concatenate(
                (
                    geometry_features.coordinate_mask,
                    np.repeat(geometry_features.direction_element_mask, 2, axis=1),
                    prepared.joint_mask,
                ),
                axis=1,
            ).astype(np.float64)
            reconstruction, _coordinates = multi_cycle_dtw_median_reconstruction(
                target,
                mask,
                geometry_features.geometry,
                result.landmarks,
            )
            losses = _traversal_huber(target, reconstruction, mask, result.landmarks)
            rows.append(
                {
                    "events": int(losses.size),
                    "maximum": float(np.max(losses)),
                    "mean": float(np.mean(losses)),
                    "median": float(np.median(losses)),
                    "pass_fraction": float(np.mean(losses <= 0.02)),
                    "q80": float(np.quantile(losses, 0.8)),
                    "q90": float(np.quantile(losses, 0.9)),
                }
            )
    rules = {
        "strict_max_002": lambda row: row["maximum"] <= 0.02,
        "median_002": lambda row: row["median"] <= 0.02,
        "mean_002": lambda row: row["mean"] <= 0.02,
        "q80_003": lambda row: row["q80"] <= 0.03,
        "q80_004": lambda row: row["q80"] <= 0.04,
        "median_002_q80_004": lambda row: row["median"] <= 0.02
        and row["q80"] <= 0.04,
        "pass_fraction_060": lambda row: row["pass_fraction"] >= 0.60,
        "pass_fraction_080": lambda row: row["pass_fraction"] >= 0.80,
    }
    summary: dict[str, object] = {}
    for name, predicate in rules.items():
        accepted = [row for row in rows if predicate(row)]
        summary[name] = {
            "accepted": len(accepted),
            "accepted_rate": len(accepted) / total,
            "mean_events": float(np.mean([row["events"] for row in accepted]))
            if accepted
            else 0.0,
        }
    print(
        json.dumps(
            {
                "base_eligible": base_eligible,
                "base_eligible_rate": base_eligible / total,
                "rules": summary,
                "mean_retained_event_fraction": float(
                    np.mean(retained_event_fractions)
                ),
                "refined_eligible": refined_eligible,
                "refined_eligible_rate": refined_eligible / total,
                "schema": "temporac.full-sequence-robust-sweep.v4",
                "structural_eligible": structural_eligible,
                "structural_eligible_rate": structural_eligible / total,
                "total_identities": total,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

