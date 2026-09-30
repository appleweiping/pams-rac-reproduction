"""Measure v1 and v2 complete-chain availability on the bound v44 population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from smoke_temporac_pseudo_targets_v7 import _feature

from pams.temporac.campaign_runtime_candidate_v1 import adapted_feature_receipt
from pams.temporac.full_chain_landmarks_candidate_v1 import full_chain_integer_landmarks
from pams.temporac.full_chain_landmarks_candidate_v2 import (
    full_chain_integer_landmarks_v2,
)
from pams.temporac.preprocess import preprocess_identity
from pams.warp_phase.packing import trusted_load_source_pickle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    args = parser.parse_args()
    source_path = (
        args.repository
        / "data/counting_multirep_skeleton_pose_expanded_v44_len320"
        / f"{args.split}.pkl"
    )
    decoded, source_receipt = trusted_load_source_pickle(
        source_path, repository_root=args.repository, split=args.split
    )
    total = v1 = v2 = recovered = 0
    for video_index, sample in enumerate(decoded[args.split]):
        for person_index in range(len(sample["person_mask"])):
            if not bool(sample["person_mask"][person_index]):
                continue
            feature, receipt, digest = _feature(
                sample,
                split=args.split,
                source_sha256=source_receipt.sha256,
                video_index=video_index,
                person_index=person_index,
            )
            adapted, _receipt, _sha, _resampling = adapted_feature_receipt(
                feature,
                receipt,
                expected_source_receipt_sha256=digest,
            )
            prepared = preprocess_identity(
                adapted.motion,
                adapted.frame_mask,
                adapted.sampled_frame_indices,
            )
            bounds = np.array(prepared.run_bounds, dtype=np.int32, copy=True)
            bounds[:, 1] += 1
            old = full_chain_integer_landmarks(
                prepared.geometry_features.geometry, bounds
            ).landmarks
            new = full_chain_integer_landmarks_v2(
                prepared.geometry_features.geometry, bounds
            ).landmarks
            total += 1
            v1 += int(old.size >= 3)
            v2 += int(new.size >= 3)
            recovered += int(old.size < 3 <= new.size)
    print(
        json.dumps(
            {
                "identities": total,
                "recovered": recovered,
                "schema": "temporac.landmark-sweep.v14",
                "split": args.split,
                "v1_available": v1,
                "v2_available": v2,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
