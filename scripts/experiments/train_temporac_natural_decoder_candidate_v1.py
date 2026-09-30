"""Train a label-free natural decoder while retaining the X0 teacher gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from pams.temporac.campaign_runtime_candidate_v1 import _masked_huber
from pams.temporac.campaign_runtime_candidate_v6 import (
    teacher_source_loss_certificate_aligned_v6,
)
from pams.temporac.certify_adaptive_v2 import hybrid_integer_landmarks
from pams.temporac.phase_closure_candidate_v1 import close_phase_on_landmarks
from pams.temporac.preprocess_resampled_v2 import preprocess_resampled_identity
from pams.temporac.teacher import TempoRACTeacher, arc_static_features
from pams.warp_phase.packing import trusted_load_source_pickle


@dataclass(frozen=True, slots=True)
class NaturalUnit:
    teacher_input: np.ndarray
    static_features: np.ndarray
    target: np.ndarray
    mask: np.ndarray
    phase: np.ndarray
    landmarks: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _natural_units(repository: Path) -> list[NaturalUnit]:
    source = repository / "data/counting_multirep_skeleton_pose_expanded_v44_len320/train.pkl"
    decoded, _receipt = trusted_load_source_pickle(
        source,
        repository_root=repository,
        split="train",
    )
    units: list[NaturalUnit] = []
    for sample in decoded["train"]:
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
            result = hybrid_integer_landmarks(
                prepared.geometry_features.geometry,
                sample_bounds,
            )
            if result.landmarks.size < 3:
                continue
            continuous = np.asarray(prepared.teacher_input[:, :149], dtype=np.float32)
            static = arc_static_features(
                np.asarray(continuous, dtype=np.float64),
                prepared.edge_lengths,
                prepared.run_bounds,
            ).astype(np.float32)
            geometry_features = prepared.geometry_features
            mask = np.concatenate(
                (
                    geometry_features.coordinate_mask,
                    np.repeat(geometry_features.direction_element_mask, 2, axis=1),
                    prepared.joint_mask,
                ),
                axis=1,
            ).astype(np.float32)
            dummy_phase = np.tile(np.asarray([[1.0, 0.0]]), (continuous.shape[0], 1))
            closed_phase = close_phase_on_landmarks(
                dummy_phase,
                result.landmarks,
                geometry=geometry_features.geometry,
            ).astype(np.float32)
            units.append(
                NaturalUnit(
                    teacher_input=np.array(prepared.teacher_input, dtype=np.float32, copy=True),
                    static_features=np.array(static, copy=True),
                    target=np.array(continuous, copy=True),
                    mask=np.array(mask, copy=True),
                    phase=np.array(closed_phase, copy=True),
                    landmarks=np.array(result.landmarks, dtype=np.int32, copy=True),
                )
            )
    return units


def _natural_loss(model: TempoRACTeacher, unit: NaturalUnit, device: torch.device) -> Tensor:
    static = torch.from_numpy(unit.static_features).to(device)
    phase = torch.from_numpy(unit.phase).to(device)
    target = torch.from_numpy(unit.target).to(device)
    mask = torch.from_numpy(unit.mask).to(device)
    code = model.static(static)
    reconstruction = model.reconstruction(phase, code)
    losses = [
        _masked_huber(
            reconstruction[int(left) : int(right)],
            target[int(left) : int(right)],
            mask[int(left) : int(right)],
        )
        for left, right in zip(unit.landmarks[:-1], unit.landmarks[1:], strict=True)
    ]
    return torch.mean(torch.stack(losses))


def _probe(
    model: TempoRACTeacher,
    units: list[NaturalUnit],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        natural = float(
            np.mean(
                [
                    float(_natural_loss(model, unit, device).cpu())
                    for unit in units[: min(24, len(units))]
                ]
            )
        )
        x0 = float(
            np.mean(
                [
                    float(
                        teacher_source_loss_certificate_aligned_v6(
                            model,
                            source_id,
                            device=device,
                            anchor_weight=0.1,
                        ).cpu()
                    )
                    for source_id in range(24, 32)
                ]
            )
        )
    model.train()
    return natural, x0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--x0-weight", type=float, default=0.25)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = TempoRACTeacher()
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).train()
    for parameter in model.phase.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    units = _natural_units(args.repository)
    if not units:
        raise RuntimeError("no natural landmark-eligible training units")
    rng = np.random.default_rng(20260818)
    order = rng.permutation(len(units))
    best_score = float("inf")
    best_step = 0
    best_state: dict[str, Tensor] | None = None
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        if (step - 1) % len(order) == 0 and step > 1:
            order = rng.permutation(len(units))
        unit = units[int(order[(step - 1) % len(order)])]
        optimizer.zero_grad(set_to_none=True)
        natural_loss = _natural_loss(model, unit, device)
        x0_loss = teacher_source_loss_certificate_aligned_v6(
            model,
            (step - 1) % 24,
            device=device,
            anchor_weight=0.1,
        )
        total = natural_loss + args.x0_weight * x0_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.report_every == 0 or step == args.steps:
            probe_natural, probe_x0 = _probe(model, units, device)
            score = probe_natural + args.x0_weight * probe_x0
            print(
                json.dumps(
                    {
                        "elapsed_seconds": time.monotonic() - started,
                        "eligible_natural_units": len(units),
                        "probe_natural": probe_natural,
                        "probe_x0": probe_x0,
                        "selection_score": score,
                        "step": step,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if score < best_score:
                best_score = score
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("no natural decoder checkpoint candidate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "metadata": {
                "best_selection_score": best_score,
                "best_step": best_step,
                "eligible_natural_units": len(units),
                "input_checkpoint_sha256": _sha256(args.checkpoint),
                "runtime": "temporac.natural-decoder-candidate.v1",
                "scientific_status": "DIAGNOSTIC_LABEL_FREE_PROTOCOL_REVISION",
                "steps": args.steps,
                "x0_weight": args.x0_weight,
            },
        },
        args.output,
    )
    print(json.dumps({"checkpoint": str(args.output), "sha256": _sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
