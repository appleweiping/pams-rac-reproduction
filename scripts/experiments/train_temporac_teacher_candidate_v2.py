"""Train the corrected additive TempoRAC teacher objective candidate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from pams.temporac.campaign_runtime_candidate_v3 import teacher_source_loss_v3
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


def _mean_tune_loss(model: TempoRACTeacher, device: torch.device) -> float:
    model.eval()
    with torch.inference_mode():
        values = [
            float(teacher_source_loss_v3(model, source, device=device).detach().cpu())
            for source in range(24, 32)
        ]
    model.train()
    return float(np.mean(values))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int, choices=(20260815, 20260816, 20260817))
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    if not 1 <= args.steps <= 20_000:
        parser.error("steps must be in [1,20000]")
    job_name = f"temporac.execution.v4/teacher/seed={args.seed}"
    job = JOB_SPECS[job_name]
    configure_torch_determinism()
    model = TempoRACTeacher()
    initialization_seed = initialize_module_cpu(model, job_name=job_name, seed=args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=job.base_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
    )
    best_loss = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    started = time.monotonic()
    permutation: list[int] = []
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
        loss = teacher_source_loss_v3(model, permutation[offset], device=device)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError("nonfinite teacher gradient norm")
        optimizer.step()
        if step % args.checkpoint_every == 0 or step == args.steps:
            tune = _mean_tune_loss(model, device)
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
            if tune < best_loss:
                best_loss = tune
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("no teacher checkpoint candidate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": best_state,
        "metadata": {
            "best_step": best_step,
            "best_x0_tune_loss": best_loss,
            "initialization_seed": initialization_seed,
            "job_name": job_name,
            "runtime": "temporac.teacher-candidate-runtime.v2-corrected-f11",
            "scientific_status": "CANDIDATE_PENDING_PROTOCOL_REVIEW",
            "seed": args.seed,
            "steps": args.steps,
        },
    }
    torch.save(payload, args.output)
    print(
        json.dumps(
            {
                "best_step": best_step,
                "best_x0_tune_loss": best_loss,
                "checkpoint": str(args.output),
                "sha256": _sha256(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
