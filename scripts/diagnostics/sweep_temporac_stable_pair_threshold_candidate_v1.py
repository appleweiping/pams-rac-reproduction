"""Train-only sweep for the stable-pair geometry acceptance threshold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pams.temporac.certify_adaptive_v3 import periodic_integer_landmarks
from pams.temporac.certify_adaptive_v4 import _pair_huber
from pams.temporac.dtw_phase_candidate_v1 import (
    dtw_periodic_reconstruction,
    dtw_phase_coordinates,
)
from pams.temporac.periodic_template_candidate_v1 import (
    periodic_template_reconstruction,
)
from pams.temporac.preprocess_resampled_v2 import preprocess_resampled_identity
from pams.warp_phase.packing import trusted_load_source_pickle


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


def _structural_gates(geometry: np.ndarray, landmarks: np.ndarray) -> tuple[bool, bool]:
    starts = geometry[landmarks[:-1]]
    median = np.median(starts, axis=0)
    origin = bool(
        np.all(np.linalg.norm(starts - median, axis=1) / np.sqrt(66.0) <= 0.05)
    )
    topology = True
    for left_raw, right_raw in zip(landmarks[:-1], landmarks[1:], strict=True):
        left, right = int(left_raw), int(right_raw)
        delta = geometry[left + 1 : right + 1] - geometry[left:right]
        arc = np.linalg.norm(delta, axis=1)
        topology &= bool(
            arc.size >= 6
            and np.count_nonzero(arc > 0.0) >= 5
            and float(np.max(arc) / np.sum(arc)) < 0.25
        )
    return origin, topology


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--thresholds", default="0.02,0.025,0.03,0.04,0.05")
    args = parser.parse_args()
    thresholds = [float(value) for value in args.thresholds.split(",")]
    source = args.repository / "data/counting_multirep_skeleton_pose_expanded_v44_len320/train.pkl"
    decoded, _receipt = trusted_load_source_pickle(
        source,
        repository_root=args.repository,
        split="train",
    )
    total = 0
    base_eligible = 0
    rows: list[dict[str, object]] = []
    self_consistency_rows: list[dict[str, object]] = []
    dtw_rows: list[dict[str, object]] = []
    best_dtw_rows: list[dict[str, object]] = []
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
            base = periodic_integer_landmarks(prepared.geometry_features.geometry, bounds)
            if base.landmarks.size < 3:
                continue
            base_eligible += 1
            candidates = [
                base.landmarks[index : index + 3]
                for index in range(base.landmarks.size - 2)
            ]
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
            candidate_rows: list[
                tuple[float, float, float, np.ndarray, bool, bool]
            ] = []
            for candidate in candidates:
                origin, topology = _structural_gates(
                    geometry_features.geometry,
                    candidate,
                )
                reconstruction = periodic_template_reconstruction(
                    target,
                    mask,
                    geometry_features.geometry,
                    candidate,
                )
                oracle = _certificate_huber(target, reconstruction, mask, candidate)
                candidate_dtw_oracle = float("inf")
                if origin and topology:
                    candidate_left, candidate_middle, candidate_right = (
                        int(value) for value in candidate
                    )
                    candidate_first_coordinate, candidate_second_coordinate = (
                        dtw_phase_coordinates(
                            geometry_features.geometry[
                                candidate_left : candidate_middle + 1
                            ],
                            geometry_features.geometry[
                                candidate_middle : candidate_right + 1
                            ],
                        )
                    )
                    candidate_dtw_reconstruction = dtw_periodic_reconstruction(
                        target,
                        mask,
                        candidate,
                        candidate_first_coordinate,
                        candidate_second_coordinate,
                    )
                    candidate_dtw_oracle = _certificate_huber(
                        target,
                        candidate_dtw_reconstruction,
                        mask,
                        candidate,
                    )
                candidate_rows.append(
                    (
                        _pair_huber(geometry_features.geometry, candidate),
                        oracle,
                        candidate_dtw_oracle,
                        candidate,
                        origin,
                        topology,
                    )
                )
            structurally_valid = [
                row for row in candidate_rows if row[4] and row[5]
            ]
            if not structurally_valid:
                continue
            pair_score, oracle, dtw_oracle, landmarks, origin, topology = min(
                structurally_valid,
                key=lambda row: (row[0], int(row[3][0])),
            )
            rows.append(
                {
                    "oracle_pass": oracle <= 0.02,
                    "origin_pass": origin,
                    "pair_score": pair_score,
                    "topology_pass": topology,
                }
            )
            dtw_rows.append(
                {
                    "oracle": dtw_oracle,
                    "pair_score": pair_score,
                }
            )
            self_pair_score, self_oracle, _self_dtw, _self_landmarks, _, _ = min(
                structurally_valid,
                key=lambda row: (row[1], row[0], int(row[3][0])),
            )
            self_consistency_rows.append(
                {
                    "oracle": self_oracle,
                    "pair_score": self_pair_score,
                }
            )
            best_dtw_pair_score, _best_arc, best_dtw, _best_landmarks, _, _ = min(
                structurally_valid,
                key=lambda row: (row[2], row[0], int(row[3][0])),
            )
            best_dtw_rows.append(
                {
                    "oracle": best_dtw,
                    "pair_score": best_dtw_pair_score,
                }
            )
    sweep: list[dict[str, object]] = []
    for threshold in thresholds:
        selected = [row for row in rows if float(row["pair_score"]) <= threshold]
        gate_pass = [
            row
            for row in selected
            if bool(row["origin_pass"])
            and bool(row["topology_pass"])
            and bool(row["oracle_pass"])
        ]
        sweep.append(
            {
                "all_gate_pass": len(gate_pass),
                "all_gate_rate": len(gate_pass) / total,
                "oracle_fail": sum(not bool(row["oracle_pass"]) for row in selected),
                "origin_fail": sum(not bool(row["origin_pass"]) for row in selected),
                "selected": len(selected),
                "selected_rate": len(selected) / total,
                "threshold": threshold,
                "topology_fail": sum(not bool(row["topology_pass"]) for row in selected),
            }
        )
    print(
        json.dumps(
            {
                "base_eligible": base_eligible,
                "base_eligible_rate": base_eligible / total,
                "best_dtw_sweep": [
                    {
                        "accepted": sum(
                            float(row["oracle"]) <= threshold
                            for row in best_dtw_rows
                        ),
                        "accepted_rate": sum(
                            float(row["oracle"]) <= threshold
                            for row in best_dtw_rows
                        )
                        / total,
                        "threshold": threshold,
                    }
                    for threshold in (0.01, 0.015, 0.018, 0.02)
                ],
                "schema": "temporac.stable-pair-threshold-sweep.v1",
                "dtw_sweep": [
                    {
                        "accepted": sum(
                            float(row["oracle"]) <= threshold for row in dtw_rows
                        ),
                        "accepted_rate": sum(
                            float(row["oracle"]) <= threshold for row in dtw_rows
                        )
                        / total,
                        "threshold": threshold,
                    }
                    for threshold in (0.01, 0.015, 0.018, 0.02)
                ],
                "self_consistency_sweep": [
                    {
                        "accepted": sum(
                            float(row["oracle"]) <= threshold
                            for row in self_consistency_rows
                        ),
                        "accepted_rate": sum(
                            float(row["oracle"]) <= threshold
                            for row in self_consistency_rows
                        )
                        / total,
                        "threshold": threshold,
                    }
                    for threshold in (0.01, 0.015, 0.018, 0.02)
                ],
                "sweep": sweep,
                "total_identities": total,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
