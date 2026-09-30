"""Audit every v44 person slot against the frozen TempoRAC preprocessor."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pams.temporac.preprocess import PreprocessError, preprocess_identity
from pams.warp_phase.packing import trusted_load_source_pickle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args()
    repository = args.repository.resolve()
    summary: dict[str, Any] = {}
    for split in ("train", "val"):
        source = (
            repository
            / "data/counting_multirep_skeleton_pose_expanded_v44_len320"
            / f"{split}.pkl"
        )
        decoded, receipt = trusted_load_source_pickle(
            source,
            repository_root=repository,
            split=split,
        )
        successes: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        for video_index, sample in enumerate(decoded[split]):
            for person_index in range(len(sample["person_mask"])):
                if not bool(sample["person_mask"][person_index]):
                    continue
                try:
                    identity = preprocess_identity(
                        sample["motion"][person_index],
                        sample["frame_mask"][person_index],
                        sample["sampled_frame_indices"],
                    )
                except PreprocessError as error:
                    failures.append(
                        {
                            "code": getattr(error, "code", type(error).__name__),
                            "message": str(error),
                            "person_index": person_index,
                            "video_index": video_index,
                        }
                    )
                    continue
                successes.append(
                    {
                        "person_index": person_index,
                        "retained_samples": len(identity.clocks),
                        "video_index": video_index,
                    }
                )
        summary[split] = {
            "failed": len(failures),
            "failure_counts": dict(Counter(row["code"] for row in failures)),
            "first_failures": failures[:10],
            "preprocess_ok": len(successes),
            "retained_samples_max": max(
                (row["retained_samples"] for row in successes), default=None
            ),
            "retained_samples_min": min(
                (row["retained_samples"] for row in successes), default=None
            ),
            "source_bytes": receipt.byte_count,
            "source_sha256": receipt.sha256,
            "valid_person_slots": len(successes) + len(failures),
        }
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
