"""Train the protocol-revision X0 analytic-phase diagnostic candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from pams.temporac.campaign_runtime_candidate_v4 import (
    teacher_source_loss_phase_anchored_v4,
)
from pams.temporac.campaign_runtime_candidate_v5 import (
    teacher_source_loss_direction_blind_v5,
)
from pams.temporac.campaign_runtime_candidate_v6 import (
    teacher_source_loss_certificate_aligned_v6,
)
from pams.temporac.runtime import (
    JOB_SPECS,
    RngDomain,
    configure_torch_determinism,
    cycle_permutation,
    initialize_module_cpu,
    job_learning_rate,
)
from pams.temporac.teacher import TempoRACTeacher


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loss(
    model: TempoRACTeacher,
    source_id: int,
    *,
    device: torch.device,
    anchor_weight: float,
    direction_blind: bool,
    certificate_target: bool,
) -> torch.Tensor:
    if certificate_target:
        return teacher_source_loss_certificate_aligned_v6(
            model,
            source_id,
            device=device,
            anchor_weight=anchor_weight,
        )
    if direction_blind:
        return teacher_source_loss_direction_blind_v5(
            model,
            source_id,
            device=device,
            anchor_weight=anchor_weight,
        )
    return teacher_source_loss_phase_anchored_v4(
        model,
        source_id,
        device=device,
        anchor_weight=anchor_weight,
    )


def _tune(
    model: TempoRACTeacher,
    device: torch.device,
    anchor_weight: float,
    direction_blind: bool,
    certificate_target: bool,
) -> float:
    model.eval()
    with torch.inference_mode():
        values = [
            float(
                _loss(
                    model,
                    source_id,
                    device=device,
                    anchor_weight=anchor_weight,
                    direction_blind=direction_blind,
                    certificate_target=certificate_target,
                ).cpu()
            )
            for source_id in range(24, 32)
        ]
    model.train()
    return float(np.mean(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(20260815, 20260816, 20260817))
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--report-every", type=int, default=250)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--direction-blind", action="store_true")
    parser.add_argument("--certificate-target", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    job_name = f"temporac.execution.v4/teacher/seed={args.seed}"
    job = JOB_SPECS[job_name]
    configure_torch_determinism()
    model = TempoRACTeacher()
    initialization_seed = initialize_module_cpu(model, job_name=job_name, seed=args.seed)
    device = torch.device(args.device)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=job.base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    permutation: list[int] = []
    started = time.monotonic()
    best_tune = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    for step in range(1, args.steps + 1):
        cycle = (step - 1) // 24
        offset = (step - 1) % 24
        if offset == 0:
            permutation = cycle_permutation(
                RngDomain.TEACHER_SOURCE_CYCLE,
                job_name=job_name,
                seed=args.seed,
                cycle=cycle,
                size=24,
            ).tolist()
        optimizer.param_groups[0]["lr"] = job_learning_rate(job, min(step, job.steps))
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(
            model,
            permutation[offset],
            device=device,
            anchor_weight=args.anchor_weight,
            direction_blind=args.direction_blind,
            certificate_target=args.certificate_target,
        )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("nonfinite teacher gradient norm")
        optimizer.step()
        if step % args.report_every == 0 or step == args.steps:
            tune = _tune(
                model,
                device,
                args.anchor_weight,
                args.direction_blind,
                args.certificate_target,
            )
            print(
                json.dumps(
                    {
                        "elapsed_seconds": time.monotonic() - started,
                        "step": step,
                        "train_loss": float(loss.detach().cpu()),
                        "tune_loss": tune,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if tune < best_tune:
                best_tune = tune
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("no phase-anchored checkpoint candidate")
    model.load_state_dict(best_state, strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "metadata": {
                "initialization_seed": initialization_seed,
                "anchor_weight": args.anchor_weight,
                "best_step": best_step,
                "best_tune_loss": best_tune,
                "direction_blind": args.direction_blind,
                "certificate_target": args.certificate_target,
                "job_name": job_name,
                "runtime": "temporac.teacher-phase-anchored-protocol-revision.v1",
                "scientific_status": "DIAGNOSTIC_NOT_FROZEN_F11",
                "seed": args.seed,
                "steps": args.steps,
            },
        },
        args.output,
    )
    print(json.dumps({"checkpoint": str(args.output), "sha256": _sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
