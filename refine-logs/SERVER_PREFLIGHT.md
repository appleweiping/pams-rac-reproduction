# Sanitized Server Preflight: WARP-PHASE Pilot v1

**Snapshot:** `2026-08-15T16:45:00+08:00`  
**Mode:** desensitized read-only inspection  
**Verdict:** **NOT_READY / launch authorization = 0**

No host identifier, network coordinate, account identifier, credential, key material, or absolute filesystem location is retained in this record.

## Preflight evidence

| Area | Observation | Bounded conclusion |
|---|---|---|
| Platform | Ubuntu 20.04.6, x86_64; 40 logical CPUs | Compatible shape observed; not an environment lock |
| Memory | 125.4 GiB total; 69.1 GiB available | Capacity observed; swap risk remains |
| Accelerators | 2 x RTX A6000, 49,140 MiB each; driver 550.54.14; idle at snapshot | Hardware available at snapshot only |
| Candidate environment | CPython 3.10.15, NumPy 1.26.4, PyTorch 2.5.0+cu118 | Bounded import/version probe passed |
| CUDA witness | Seeded 2x2 CUDA witness passed | Preliminary kernel-dispatch evidence only |
| Legacy environment | MotionBERT legacy environment blocked | Fallback forbidden |
| Data | Canonical training and validation SHA-256 values exactly matched the execution contract | Read-only match; Gate 0 still has not passed |
| Output storage | 96% used, 171 GiB free (rounded); inodes 14% used | Fresh exact-byte receipt required before any launch |
| Swap | Nearly exhausted | Fail closed unless fresh monitoring is acceptable |
| Background activity | 29 historical CPU processes; one occupied one core; no observed GPU use | Do not kill or reuse; isolate the pilot run |

The candidate environment is not marked READY. The read-only probe did not freeze the environment spec/lock, validate every declared import or weight/cache path, or perform the independent agent-follows-doc tier.

## Authorization blockers

Launch authorization remains zero until every item below closes:

1. Freeze the required training environment lock and its non-empty SHA-256.
2. Have a fresh independent agent follow the compute documentation and documented invocation verbatim; undocumented repair is a failure.
3. Create a unique, new, empty, isolated run directory and bind it to the run ID.
4. Produce a fresh output-space receipt that matches the environment-lock hash and the frozen budget hash.
5. Satisfy the experiment dependency gate. Training remains blocked until Gates 0-2 pass, regardless of compute availability.

Missing, stale, mismatched, or unverifiable evidence denies launch. The first sanity release is limited to `max_parallel=1`; later work is limited to `max_parallel=2` only after sanity and a fresh health monitor both pass.

## Output-space gate

The machine-readable authority is `.aris/compute/output-budget.json`.

| Output class | Maximum count | Cap per item | Worst-case allocation |
|---|---:|---:|---:|
| Gate 0 pack | 1 | 12 GiB | 12 GiB |
| Gate 1 fixture | 1 | 2 GiB | 2 GiB |
| Gate 2 selector | 1 | 2 GiB | 2 GiB |
| Gate 3 CPU smoke | 1 | 0.25 GiB | 0.25 GiB |
| Sanity training | 3 | 0.25 GiB | 0.75 GiB |
| Tuning training | 14 | 0.5 GiB | 7 GiB |
| Full training | 42 | 1.5 GiB | 63 GiB |
| Evaluator-only audits | 3 | 0.5 GiB | 1.5 GiB |
| Campaign-shared outputs | 1 allocation | 4 GiB | 4 GiB |
| **All scoped outputs** |  |  | **92.5 GiB** |

The 59 training jobs have a worst-case output allocation of 70.75 GiB. The all-output hard cap is 96 GiB, the minimum remaining-free guard is 64 GiB, and a first campaign receipt therefore requires at least 160 GiB exact free space. If exact free space is at least 171 GiB, consuming the full 96 GiB cap leaves 75 GiB, an 11 GiB margin above the minimum guard. Because the preflight value is rounded and the mount is already 96% used, this arithmetic is not itself authorization.

Every failed or canceled job consumes its count and byte budget. Temporary files, logs, checkpoints, predictions, traces, and receipts are in scope. A per-item cap, global cap, free-space guard, inode guard, isolation check, dependency receipt, lock hash, or receipt-integrity failure stops new launches without deletion-based headroom recovery.

## Ledger links

- Human-readable provider ledger: `.aris/compute/ssh.md`
- Machine-readable provider state: `.aris/compute/provider-env.json`
- Machine-readable output gate: `.aris/compute/output-budget.json`
- Experiment authority: `refine-logs/EXPERIMENT_EXECUTION_CONTRACT.md`, `refine-logs/EXPERIMENT_PLAN.md`, and `refine-logs/EXPERIMENT_TRACKER.md`
