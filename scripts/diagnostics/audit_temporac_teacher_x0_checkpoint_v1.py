"""Evaluate a candidate teacher checkpoint through all 56 X0 tune certificates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import pams.temporac.certify as frozen
from pams.temporac.campaign_runtime_candidate_v1 import _masked_huber, _teacher_view
from pams.temporac.campaign_runtime_candidate_v5 import DirectionBlindTeacherAdapter
from pams.temporac.teacher import TempoRACTeacher, phase_increments


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--direction-blind", action="store_true")
    args = parser.parse_args()
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TempoRACTeacher()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    inference_model = DirectionBlindTeacherAdapter(model) if args.direction_blind else model
    checkpoint_sha = _sha256(args.checkpoint)
    rows: list[dict[str, object]] = []
    negative_edges = 0
    alias_edges = 0
    edge_count = 0
    winding_errors: list[float] = []
    reconstruction_losses: list[float] = []
    certificate_reconstruction_losses: list[float] = []
    with torch.inference_mode():
        for source_id in range(24, 32):
            for block_index in range(7):
                teacher_view = _teacher_view(source_id, block_index, device)
                _code, phase, reconstruction = inference_model(
                    teacher_view.teacher_input,
                    teacher_view.static_features,
                )
                reconstruction_losses.append(
                    float(
                        _masked_huber(
                            reconstruction,
                            teacher_view.reconstruction_target,
                            teacher_view.reconstruction_mask,
                        ).cpu()
                    )
                )
                increments = phase_increments(phase)
                negative_edges += int(torch.count_nonzero(increments < -1e-12).cpu())
                alias_edges += int(torch.count_nonzero(increments >= 0.25).cpu())
                edge_count += int(increments.numel())
                for left_raw, right_raw in teacher_view.view.traversal_bounds:
                    left, right = int(left_raw), int(right_raw)
                    winding = float(torch.sum(phase_increments(phase[left : right + 1])).cpu())
                    winding_errors.append(abs(winding - 1.0))
                fixture = frozen.certificate_track_from_x0(
                    teacher_view.view,
                    teacher_sha256=checkpoint_sha,
                )
                track = frozen.CertificateTrack(
                    geometry=fixture.geometry,
                    continuous=fixture.continuous,
                    continuous_mask=fixture.continuous_mask,
                    phase=phase.cpu().numpy().astype("<f8"),
                    reconstruction=reconstruction.cpu().numpy().astype("<f8"),
                    run_bounds=fixture.run_bounds,
                    provenance=fixture.provenance,
                    _provenance_authority=frozen._TRACK_PROVENANCE_AUTHORITY,
                    analytic_pulse=fixture.analytic_pulse,
                    analytic_chi=fixture.analytic_chi,
                )
                target = frozen.certify_target(track, "X0")
                landmarks = frozen.integer_landmarks(track.geometry, track.run_bounds)
                bounds = np.column_stack((landmarks[:-1], landmarks[1:])).astype(np.int32)
                canonical, _origin, _starts = frozen.canonicalize_phase(
                    track.phase,
                    bounds,
                    tau=1e-7,
                )
                for left_raw, right_raw in bounds:
                    left, right = int(left_raw), int(right_raw)
                    _, dense_target, dense_reconstruction, dense_mask, _, _ = (
                        frozen._dense_traversal(track, canonical, left, right)
                    )
                    midpoint_mask = dense_mask[1:-1].astype(np.float64)
                    error = np.abs(dense_reconstruction[1:-1] - dense_target[1:-1])
                    huber = np.where(
                        error <= 0.05,
                        0.5 * np.square(error) / 0.05,
                        error - 0.025,
                    )
                    certificate_reconstruction_losses.append(
                        float(np.sum(huber * midpoint_mask) / np.sum(midpoint_mask))
                    )
                rows.append(
                    {
                        "block_index": block_index,
                        "certified": target.certified,
                        "reason_codes": [int(value) for value in target.reasons],
                        "source_id": source_id,
                    }
                )
    reason_counts = Counter(code for row in rows for code in row["reason_codes"])
    print(
        json.dumps(
            {
                "certified": sum(int(row["certified"]) for row in rows),
                "alias_edge_fraction": alias_edges / edge_count,
                "checkpoint_sha256": checkpoint_sha,
                "maximum_traversal_winding_error": max(winding_errors),
                "mean_masked_reconstruction": sum(reconstruction_losses)
                / len(reconstruction_losses),
                "mean_certificate_reconstruction": sum(certificate_reconstruction_losses)
                / len(certificate_reconstruction_losses),
                "maximum_certificate_reconstruction": max(
                    certificate_reconstruction_losses
                ),
                "mean_traversal_winding_error": sum(winding_errors) / len(winding_errors),
                "negative_edge_fraction": negative_edges / edge_count,
                "reason_counts": dict(reason_counts),
                "rows": rows,
                "views": len(rows),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
