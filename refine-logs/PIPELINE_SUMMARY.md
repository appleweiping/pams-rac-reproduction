# Pipeline Summary: WARP-PHASE Pilot v1

**Problem:** Study whether an explicit signed local warp-integral objective improves per-person repetition counting under within-track tempo drift on checksum-frozen GT-bbox-assisted AlphaPose supplied tracks, without learner access to per-person count or period labels.  
**Final Method Thesis:** A clean identity track defines a raw signed phase measure; a known target-to-source warp defines the oriented integral that each warped target interval should match; treatment alone receives this exact loss.  
**Final Verdict:** `REVISE`  
**Review Ceiling:** same-family provisional  
**Evidence:** prospective partial-cache pilot; zero eligible efficacy results; deterministic Gate 2 failure
**K8:** `BLOCKED / NOT_PASSED_BLOCKED`; claim freeze blocked  
**Date:** 2026-08-15

## Phase 2 Planning Gate

- One dominant intervention: `L_WI`.
- One exact deletion: identical augmentation-only.
- Rejected complexity: learned routing/experts, auxiliary heads, phase correction, semantic-period supervision, detector/tracker work, and a TSSM fallback.
- Frontier primitive: absent intentionally; exact warp metadata is the appropriate supervision.
- Local status: the three Round-5 execution ambiguities are normatively closed in `refine-logs/EXPERIMENT_EXECUTION_CONTRACT.md` without changing the scientific route.
- Execution status: local Gate 0 is `BLOCKED`, Gate 1 is `BLOCKED`, stochastic Gate 2 is `FAIL` at `246/55/144/50/66`, and Gate 3 CPU is `BLOCKED`; training and server launch are not authorized.

## Final Deliverables

- Proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Normative local closure: `refine-logs/EXPERIMENT_EXECUTION_CONTRACT.md`
- Experiment plan: `refine-logs/EXPERIMENT_PLAN.md`
- Experiment tracker: `refine-logs/EXPERIMENT_TRACKER.md`
- Focused research contract: `idea-stage/docs/research_contract.md`

## Contribution Snapshot

- **Dominant contribution under study:** discrete warp-integral equivariance for signed phase increments in supplied-track MRAC.
- **Supporting diagnostic:** identity isolation under a one-person intervention; not private-state novelty and never a rescue.
- **Explicit non-contributions:** the continuous chain rule, ACF/FFT, PAMS/TCC-style phase learning, pseudo-cycle selection, pose SSL, private recurrence, NOLA, K4, and TSSM.
- **Claim ceiling:** only “we study whether” on the current partial GT-assisted supplied-track pilot.

## Must-Prove Claims

1. Conditionally, the signed local objective improves the frozen supplied-track tempo-drift primary endpoint relative to identical augmentation: `R>=0.05` **and** the paired all-nine-component absolute-effect CI lower endpoint is strictly positive.
2. Conditionally, a reached model isolates untouched identities under a one-person intervention; this remains diagnostic and cannot compensate for failure of Claim 1.

## Three Visible Validation Blocks

1. Primary paired law versus identical augmentation.
2. Staged mechanism, K4 shortcut, and closest-prior falsification.
3. Identity-isolation diagnostic.

Engineering fixtures and gate receipts remain supplementary; they do not create a fourth block.

## Run Ladder and Hard Budget

| Order | Stage | Training jobs | Cumulative cap | Stop decision |
|---:|---|---:|---:|---|
| 1 | Gate 0 pack/vault/isolation | 0 | 0 jobs / 0 GPUh | K6/K7 |
| 2 | Gate 1 numerical/environment fixtures | 0 | 0 / 0 | K4/K5/K6/K7 |
| 3 | Gate 2 selector stability | 0 | 0 / 0 | K5 |
| 4 | Gate 3 CPU/tiny-GPU sanity | 3 | 3 / 6 | K5/K6 |
| 5 | Stage A primary pair | 6 | 9 / 78 | K1 |
| 6 | Stage B K4 | 6 | 15 / 150 | K4 |
| 7 | Stage C mechanism | 9 | 24 / 258 | K3 |
| 8 | Stage D closest priors | 30 | 54 / 510 | K2 |
| 9 | Optional Stage E TSSM control | 5 | **59 / 552** | K9 boundary |

Any release of at least ten runnable jobs must use dependency-aware queue routing. Stage D necessarily uses the queue. Seeds are exactly `20270815`, `20270816`, and `20270817`.

## Execution Stop and ARIS Route

1. Preserve the exact Gate 0/1/2/3 CPU receipts and selector fixture audits.
2. Run a fresh experiment-integrity audit and result-to-claim judgment.
3. Record this candidate as a K5 failure and pivot without threshold tuning, fixture replacement, or GPU execution.

No training run is currently authorized. The stopped candidate must not be sent to the server.

## Main Risks and Mitigations

- **Feature/vault leakage or population collapse:** fail Gate 0 before labels; preserve all-nine-component rule.
- **NumPy/environment or fixture drift:** dedicated CPython 3.12.13/NumPy 2.4.6 generator, main NumPy 1.26.4 consumption-only witness, immutable bytes, no regeneration.
- **Selector instability:** one PCG64 stream, SHA-sorted traversal, full draw consumption, per-parent hashes, global failure.
- **Clock/mask/interpolation shortcut:** exact K4 fractions, deterministic cyclic matched-time shuffle, and deterministic midpoint unseen resampler; unchanged ratios.
- **No effect or redundancy:** K1/K3 stop the route; no secondary escape hatch or extra module.
- **Closest-method collision:** K2 and external K8 kill the residual claim rather than widening it.

## Quality Boundary

Repository baseline is 919 tests passed/6 skipped and Ruff pass. Mypy has 29 pre-existing errors only in `src/pams/baselines/escounts_official_worker.py`, `src/pams/diagnostics.py`, and `src/pams/baselines/jtsps_count_only_experiment.py`; they cannot be misattributed. The new isolated package must have zero Ruff and mypy errors and preserve historical APIs/config fingerprints.

## Next Action

Implement only the isolated Gate-0 path when separately authorized, then run `G0-PACK-CPU`. Do not proceed to `/run-experiment`, training, evaluator label access, paper claims, or K8 claim freeze until their explicit prerequisites pass.
