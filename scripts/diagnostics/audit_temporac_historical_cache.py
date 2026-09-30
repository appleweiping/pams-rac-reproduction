"""Audit a trusted historical counting cache with the frozen TempoRAC preprocessor."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from pams.temporac.preprocess import PreprocessError, preprocess_identity
from pams.temporac.preprocess_interpolated_v1 import preprocess_interpolated_identity


def load_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as stream:
        decoded = pickle.load(stream)  # noqa: S301 - trusted user-owned cache
    if isinstance(decoded, list):
        return decoded
    if isinstance(decoded, dict) and len(decoded) == 1:
        samples = next(iter(decoded.values()))
        if isinstance(samples, list):
            return samples
    if isinstance(decoded, dict) and all(isinstance(row, dict) for row in decoded.values()):
        return list(decoded.values())
    raise TypeError(f"unsupported cache root: {type(decoded).__name__}")


def classify(message: str) -> str:
    return message.split(":", maxsplit=1)[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--interpolate", action="store_true")
    args = parser.parse_args()
    summary: dict[str, Any] = {}
    for split in args.splits:
        successes: list[int] = []
        failures: list[dict[str, Any]] = []
        for video_index, sample in enumerate(load_samples(args.data_root / f"{split}.pkl")):
            for person_index in range(len(sample["person_mask"])):
                if not bool(sample["person_mask"][person_index]):
                    continue
                try:
                    preprocessor = (
                        preprocess_interpolated_identity
                        if args.interpolate
                        else preprocess_identity
                    )
                    identity = preprocessor(
                        sample["motion"][person_index],
                        sample["frame_mask"][person_index],
                        sample["sampled_frame_indices"],
                    )
                except (PreprocessError, ValueError) as error:
                    failures.append(
                        {
                            "category": classify(str(error)),
                            "message": str(error),
                            "person_index": person_index,
                            "video_index": video_index,
                        }
                    )
                    continue
                successes.append(len(identity.clocks))
        summary[split] = {
            "failed": len(failures),
            "failure_counts": dict(Counter(row["category"] for row in failures)),
            "first_failures": failures[:5],
            "preprocess_ok": len(successes),
            "retained_samples_min": min(successes, default=None),
            "retained_samples_max": max(successes, default=None),
            "valid_person_slots": len(successes) + len(failures),
        }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
