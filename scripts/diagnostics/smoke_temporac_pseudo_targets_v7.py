"""Run the v12 full-chain certificate candidate on real v44 teacher predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from pams.temporac.campaign_runtime_candidate_v2 import infer_natural_pseudo_target_v2
from pams.temporac.campaign_runtime_candidate_v5 import DirectionBlindTeacherAdapter
from pams.temporac.campaign_runtime_candidate_v12 import infer_natural_pseudo_target_v12
from pams.temporac.contract import CONTRACT_SHA256, FEATURE_RECEIPT_SCHEMA
from pams.temporac.hashio import (
    feature_npz_bytes,
    opaque_sample_key,
    sha256_bytes,
    source_binding_sha256,
)
from pams.temporac.receipts import member_payload, receipt_bytes
from pams.temporac.teacher import TempoRACTeacher
from pams.temporac.types import FeatureRecord
from pams.warp_phase.packing import trusted_load_source_pickle


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature(
    sample: dict[str, object],
    *,
    split: str,
    source_sha256: str,
    video_index: int,
    person_index: int,
) -> tuple[FeatureRecord, bytes, str]:
    raw_motion = np.asarray(sample["motion"], dtype=np.float32)[person_index]
    raw_mask = np.asarray(sample["frame_mask"])[person_index].astype(np.bool_)
    finite = np.isfinite(raw_motion).all(axis=2)
    valid = raw_mask[:, None] & finite & (raw_motion[:, :, 2] > 0.20)
    motion = np.zeros((320, 17, 3), dtype="<f4")
    motion[:, :, :2] = np.where(valid[:, :, None], raw_motion[:, :, :2], 0.0)
    motion[:, :, 2] = np.where(valid, np.clip(raw_motion[:, :, 2], 0.0, 1.0), 0.0)
    key = opaque_sample_key(split, source_sha256, video_index)
    feature = FeatureRecord(
        frame_mask=raw_mask.astype("|u1"),
        local_person_slot=np.asarray([person_index], dtype="<i8"),
        motion=motion,
        opaque_sample_key=np.frombuffer(key, dtype="|u1").copy(),
        person_mask=np.ones(1, dtype="|u1"),
        sampled_frame_indices=np.asarray(sample["sampled_frame_indices"], dtype="<i8"),
        source_length=np.asarray([int(sample["source_length"])], dtype="<i8"),
    )
    artifact, members = feature_npz_bytes(feature)
    payload = {
        "artifact_bytes": len(artifact),
        "artifact_sha256": sha256_bytes(artifact),
        "contract_sha256": CONTRACT_SHA256,
        "members": member_payload(members),
        "opaque_key_hex": key.hex(),
        "schema": FEATURE_RECEIPT_SCHEMA,
        "slot": person_index,
        "source_binding_sha256": source_binding_sha256(
            split, source_sha256, video_index, person_index
        ),
    }
    encoded = receipt_bytes(payload, expected_schema=FEATURE_RECEIPT_SCHEMA)
    return feature, encoded, sha256_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--direction-blind", action="store_true")
    parser.add_argument("--phase-closure", action="store_true")
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
            infer = (
                infer_natural_pseudo_target_v12
                if args.phase_closure
                else infer_natural_pseudo_target_v2
            )
            result = infer(
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
                "schema": "temporac.pseudo-target-smoke.v7",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
