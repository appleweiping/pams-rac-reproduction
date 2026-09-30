"""Audit teacher-independent natural landmark coverage after resampling v2."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from pams.temporac.certify import integer_landmarks
from pams.temporac.preprocess_resampled_v2 import preprocess_resampled_identity
from pams.warp_phase.packing import trusted_load_source_pickle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args()
    output: dict[str, object] = {}
    for split in ("train", "val"):
        source = args.repository / "data/counting_multirep_skeleton_pose_expanded_v44_len320" / f"{split}.pkl"
        decoded, _receipt = trusted_load_source_pickle(
            source, repository_root=args.repository, split=split
        )
        counts: list[int] = []
        for sample in decoded[split]:
            for person_index in range(len(sample["person_mask"])):
                if not bool(sample["person_mask"][person_index]):
                    continue
                prepared, _ = preprocess_resampled_identity(
                    sample["motion"][person_index],
                    sample["frame_mask"][person_index],
                    sample["sampled_frame_indices"],
                )
                sample_bounds = np.array(prepared.run_bounds, dtype=np.int32, copy=True)
                sample_bounds[:, 1] += 1
                counts.append(
                    int(integer_landmarks(prepared.geometry_features.geometry, sample_bounds).size)
                )
        output[split] = {
            "identities": len(counts),
            "landmark_count_distribution": dict(Counter(counts)),
            "minimum_three": sum(value >= 3 for value in counts),
            "minimum_three_fraction": sum(value >= 3 for value in counts) / len(counts),
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
