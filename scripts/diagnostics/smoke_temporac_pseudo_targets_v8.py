"""Run the v13 deterministic cascade on real v44 teacher predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from smoke_temporac_pseudo_targets_v7 import _feature, _sha256

from pams.temporac.campaign_runtime_candidate_v5 import DirectionBlindTeacherAdapter
from pams.temporac.campaign_runtime_candidate_v13 import infer_natural_pseudo_target_v13
from pams.temporac.teacher import TempoRACTeacher
from pams.warp_phase.packing import trusted_load_source_pickle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--direction-blind", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    source_path = (
        args.repository
        / "data/counting_multirep_skeleton_pose_expanded_v44_len320"
        / f"{args.split}.pkl"
    )
    decoded, source_receipt = trusted_load_source_pickle(
        source_path, repository_root=args.repository, split=args.split
    )
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TempoRACTeacher()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    inference_model = DirectionBlindTeacherAdapter(model) if args.direction_blind else model
    teacher_sha = _sha256(args.checkpoint)
    rows: list[dict[str, object]] = []
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
            result = infer_natural_pseudo_target_v13(
                inference_model,  # type: ignore[arg-type]
                feature,
                receipt,
                expected_source_receipt_sha256=digest,
                selected_teacher_sha256=teacher_sha,
                device=device,
            )
            rows.append(
                {
                    "certified": result.target.certified,
                    "event_count": result.target.event_count,
                    "person_index": person_index,
                    "reason_codes": [int(value) for value in result.target.reasons],
                    "video_index": video_index,
                }
            )
            if len(rows) >= args.limit:
                break
        if len(rows) >= args.limit:
            break
    reason_counts = Counter(code for row in rows for code in row["reason_codes"])
    print(
        json.dumps(
            {
                "certified": sum(int(row["certified"]) for row in rows),
                "checkpoint_sha256": teacher_sha,
                "identities": len(rows),
                "reason_counts": dict(reason_counts),
                "rows": [] if args.summary_only else rows,
                "schema": "temporac.pseudo-target-smoke.v8",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
