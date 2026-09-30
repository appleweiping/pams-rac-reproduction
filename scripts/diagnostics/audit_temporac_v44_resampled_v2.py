"""Audit every v44 identity with the additive gap-resample v2 protocol."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from pams.temporac.preprocess_resampled_v2 import (
    ResamplingError,
    preprocess_resampled_identity,
)
from pams.warp_phase.packing import trusted_load_source_pickle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args()
    repository = args.repository.resolve()
    summary: dict[str, Any] = {}
    for split in ("train", "val"):
        source = repository / "data/counting_multirep_skeleton_pose_expanded_v44_len320" / f"{split}.pkl"
        decoded, source_receipt = trusted_load_source_pickle(
            source, repository_root=repository, split=split
        )
        successes: list[dict[str, int]] = []
        failures: list[dict[str, object]] = []
        for video_index, sample in enumerate(decoded[split]):
            for person_index in range(len(sample["person_mask"])):
                if not bool(sample["person_mask"][person_index]):
                    continue
                try:
                    prepared, receipt = preprocess_resampled_identity(
                        sample["motion"][person_index],
                        sample["frame_mask"][person_index],
                        sample["sampled_frame_indices"],
                    )
                except (ResamplingError, ValueError) as error:
                    failures.append(
                        {
                            "category": str(error).split(":", maxsplit=1)[0],
                            "message": str(error),
                            "person_index": person_index,
                            "video_index": video_index,
                        }
                    )
                    continue
                successes.append(
                    {
                        "degenerate_edges_repaired": receipt.degenerate_edges_repaired,
                        "interpolated_rows": receipt.interpolated_rows,
                        "retained_samples": len(prepared.clocks),
                    }
                )
        summary[split] = {
            "failed": len(failures),
            "failure_counts": dict(Counter(row["category"] for row in failures)),
            "first_failures": failures[:5],
            "preprocess_ok": len(successes),
            "valid_person_slots": len(successes) + len(failures),
            "interpolated_rows_total": sum(row["interpolated_rows"] for row in successes),
            "degenerate_edges_repaired_total": sum(
                row["degenerate_edges_repaired"] for row in successes
            ),
            "source_bytes": source_receipt.byte_count,
            "source_sha256": source_receipt.sha256,
        }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if all(row["failed"] == 0 for row in summary.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
