# Sanitized Direct-SSH Compute Ledger

**Snapshot:** `2026-08-15T16:45:00+08:00`  
**Disclosure:** public-safe; connection coordinates and filesystem locations are intentionally not recorded.  
**Provider shape:** direct SSH host with a pre-existing conda environment.  
**Readiness:** **NOT_READY / PRECHECK_ONLY**  
**Launch authorization:** **0 jobs**

This ledger records only a completed, desensitized, read-only preflight. It is not a connection guide and does not certify the environment as ready.

## Observed resource shape

| Resource | Sanitized observation |
|---|---|
| Operating system | Ubuntu 20.04.6, x86_64 |
| CPU | 40 logical CPUs |
| Memory | 125.4 GiB total; 69.1 GiB available |
| GPU | 2 x NVIDIA RTX A6000, 49,140 MiB each |
| Driver | 550.54.14 |
| GPU occupancy | Both devices observed idle during the snapshot |
| Output mount | 96% used; 171 GiB free (rounded preflight observation) |
| Inodes | 14% used |
| Swap | Nearly exhausted; treat as an active execution risk |
| Background load | 29 historical CPU processes; one process held one CPU core; no observed GPU use |

## Environment ledger entry

### env: `cja_py310` (lock pending)

- **How:** existing conda environment on a direct SSH provider; no endpoint or path is stored here.
- **Tier:** `cpus=40`, `mem_gib=125.4`, `gpus=2`, `gpu_model=RTX A6000`.
- **Observed versions:** CPython 3.10.15; NumPy 1.26.4; PyTorch 2.5.0+cu118.
- **Bounded import/version probe:** PASS for the three observed components only. This is not a complete Tier-1 validation against a frozen environment spec.
- **Seeded kernel-dispatch witness:** PASS for a CUDA 2x2 operation during the read-only preflight.
- **Agent-follows-doc validation:** NOT RUN; a fresh agent must execute the documented invocation verbatim before readiness can be declared.
- **Canonical environment content hash:** PENDING; no frozen environment lock hash was supplied by this preflight.
- **Weights/cache validation:** NOT ESTABLISHED.
- **Gotcha:** the legacy MotionBERT environment is BLOCKED and must not be used as a fallback.

## Data preflight

The canonical training and validation source hashes were observed to match the execution contract exactly. Only the match result is recorded; source locations and host-specific details are omitted. This does not pass Gate 0 or authorize trusted unpickling.

## Fail-closed readiness gate

Readiness remains **NOT_READY**, and launch authorization remains zero, until all of the following are present and mutually consistent:

1. A frozen training-environment lock with a non-empty SHA-256, including the bindings required by Gate 1.
2. A fresh, independent agent-follows-doc run using only the compute documentation, this ledger, and the documented invocation, with no undocumented repair.
3. A newly created, isolated run directory for the authorized run ID; no reuse of historical run or result directories.
4. A fresh output-space prelaunch receipt satisfying `.aris/compute/output-budget.json`, including exact free bytes, active reservations, output-to-date, budget hash, and matching environment-lock hash.
5. The experiment contract's scientific dependencies. In particular, training remains blocked until Gates 0, 1, and 2 pass; compute readiness alone cannot waive them.

A missing, stale, empty, unequal, or unhashable receipt fails closed. Historical results, environments, and processes do not count as evidence for this pilot.

## Concurrency gate

- Current authorized parallelism: `0`.
- After the ledger, lock, isolation, dependency, and space gates close: the first sanity release has `max_parallel=1`.
- Only after sanity passes and a fresh health monitor confirms adequate memory, swap, GPU, and output-space headroom may later releases use `max_parallel=2`.
- The provider's two GPUs are an upper bound, not an authorization. Queue and dependency rules in the experiment tracker remain binding.
