"""Audit the additive adaptive-landmark candidate on X0 and real v44 data."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import pams.temporac.certify_adaptive_v2 as adaptive
from pams.temporac.preprocess import preprocess_identity
from pams.temporac.preprocess_resampled_v2 import preprocess_resampled_identity
from pams.temporac.x0 import generate_view
from pams.warp_phase.packing import trusted_load_source_pickle


def _sample_bounds(prepared: object) -> np.ndarray:
    bounds = np.array(prepared.run_bounds, dtype=np.int32, copy=True)  # type: ignore[attr-defined]
    bounds[:, 1] += 1
    return bounds


def _x0_scan() -> dict[str, object]:
    output: dict[str, object] = {}
    original = adaptive.MIN_EXCURSION_RMS
    try:
        for threshold in (0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
            adaptive.MIN_EXCURSION_RMS = threshold
            counts: dict[str, object] = {}
            for source_id in (0, 7, 23, 31):
                view = generate_view(source_id, 0, resampler="linear", offset=0)
                prepared = preprocess_identity(
                    view.motion,
                    np.ones(view.motion.shape[0], dtype=np.uint8),
                    view.clock,
                )
                result = adaptive.adaptive_integer_landmarks(
                    prepared.geometry_features.geometry,
                    _sample_bounds(prepared),
                )
                counts[str(source_id)] = {
                    "count": int(result.landmarks.size),
                    "landmarks": result.landmarks.tolist(),
                }
            output[f"{threshold:.2f}"] = counts
    finally:
        adaptive.MIN_EXCURSION_RMS = original
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--excursion", type=float, default=adaptive.MIN_EXCURSION_RMS)
    parser.add_argument("--x0-only", action="store_true")
    args = parser.parse_args()

    output: dict[str, object] = {"x0_scan": _x0_scan()}
    if not args.x0_only:
        adaptive.MIN_EXCURSION_RMS = args.excursion
        for split in ("train", "val"):
            source = (
                args.repository
                / "data/counting_multirep_skeleton_pose_expanded_v44_len320"
                / f"{split}.pkl"
            )
            decoded, _receipt = trusted_load_source_pickle(
                source, repository_root=args.repository, split=split
            )
            counts: list[int] = []
            maximum_returns: list[float] = []
            for sample in decoded[split]:
                for person_index in range(len(sample["person_mask"])):
                    if not bool(sample["person_mask"][person_index]):
                        continue
                    prepared, _ = preprocess_resampled_identity(
                        sample["motion"][person_index],
                        sample["frame_mask"][person_index],
                        sample["sampled_frame_indices"],
                    )
                    result = adaptive.hybrid_integer_landmarks(
                        prepared.geometry_features.geometry,
                        _sample_bounds(prepared),
                    )
                    counts.append(int(result.landmarks.size))
                    if result.maximum_return_rms is not None:
                        maximum_returns.append(result.maximum_return_rms)
            accepted = sum(value >= 3 for value in counts)
            output[split] = {
                "identities": len(counts),
                "landmark_count_distribution": dict(sorted(Counter(counts).items())),
                "minimum_three": accepted,
                "minimum_three_fraction": accepted / len(counts),
                "maximum_return_rms": max(maximum_returns, default=None),
            }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
