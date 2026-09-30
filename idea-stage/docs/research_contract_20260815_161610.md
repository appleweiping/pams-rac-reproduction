# Research Contract: WARP-PHASE Supplied-Track Pilot

> **Focused active-idea contract derived from the official ARIS research-contract template.**  
> **Prospective only: zero eligible results; same-family provisional; final verdict `REVISE`; K8 and claim freeze `BLOCKED`.**

## Selected Idea

- **Description:** Study whether adding one exact signed discrete warp-integral equivariance loss to a shared supplied-track phase learner improves per-person repetition counting under frozen within-track tempo drift. Treatment and augmentation-only see identical data, warps, architecture, initialization, optimizer, recurrence, NOLA, decoder, and evaluation; treatment alone receives `L_WI`. Counts, densities, evaluator periods, and boundaries remain outside the learner.
- **Source:** `idea-stage/IDEA_REPORT.md`, selected candidate `warp-equivariant-private-phase-flow`, narrowed by `refine-logs/FINAL_PROPOSAL.md`.
- **Selection rationale:** It directly addresses the anchored non-stationary-tempo bottleneck with one falsifiable mechanism and one exact deletion. Selection is proposal judgment only, not pilot evidence or novelty clearance.

## Core Claims

1. **Conditional primary claim:** study whether the signed local warp-integral objective improves the frozen supplied-track partial-cache tempo-drift normalized video-first AvgMAE relative to identical augmentation, requiring both `R>=0.05` and a paired all-nine-source-component absolute-effect 95% CI lower endpoint above zero.
2. **Conditional supporting claim:** if the primary route survives, study whether identity-private state isolates untouched identities under a one-person intervention; this is diagnostic, not private-state novelty, and cannot rescue the primary claim.

No first, SOTA, end-to-end, annotation-free, predicted-track, official MultiRep, or deployment claim is permitted. K8 remains `BLOCKED / NOT_PASSED_BLOCKED` until complete-primary-source TWCRAC adjudication.

## Method Summary

A duplicate-collapsed, normalized supplied identity track produces raw signed principal phase increments. A known target-to-source warp maps a target interval to one float64 oriented overlap integral over the clean signed phase measure. The treatment residual is the circular difference between the warped signed increment and the stop-gradient clean integral. Pause is zero, reversal is signed, invalid support fails closed, and no orientation or harmonic correction is applied.

The single learner is a 225,026-parameter Conv `68→128→128`, masked GRU128, and two-dimensional normalized phase head. It processes complete identities with one recurrent commit per retained source clock. Positive interval NOLA decodes once per identity. The selector-defined pseudo-cycle is shared infrastructure; a post-freeze evaluator harmonic gate may reject but never correct. Exact local execution semantics are bound by `refine-logs/EXPERIMENT_EXECUTION_CONTRACT.md`.

Implementation, when separately authorized, is isolated under `src/pams/warp_phase/`; only a new `warp-phase` CLI group may attach to `src/pams/cli.py`. Historical APIs and config fingerprints remain unchanged.

## Experiment Design

- **Datasets:** checksum-frozen, canonical-source-disjoint partial v44 MultiRep train/development supplied-pose cache; GT-bbox-assisted AlphaPose supplied tracks; no test or sealed data.
- **Unit/estimand:** video-first normalized AvgMAE; paired treatment-control development change; nine frozen source-connected components.
- **Baselines:** three families only: matched mechanism controls; closest phase/tempo priors; optional masked geometric TSSM control.
- **Metrics:** primary normalized video-first AvgMAE with fixed K1 rule; secondary AvgOBO, supplied-track Period-mAP/AP50/AP75 `[0,100]`, clean-to-warp degradation, phase correspondence, and untouched-person response.
- **Key hyperparameters:** seeds `20270815/16/17`; AdamW `3e-4`, weight decay `1e-4`; 20,000 updates; eight complete identities/step; four-step accumulation; clip `1.0`; 64-clock loss/decode windows with stride 32.
- **Compute budget:** hard maximum 59 training jobs and 552 GPU-hours; no over-budget continuation.
- **Run order:** Gate 0 pack/vault, Gate 1 numerical/environment fixtures, Gate 2 selector stability, Gate 3 CPU/tiny-GPU sanity, then conditional Stages A–E.

## Baselines

| Family | Methods/controls | Dataset | Decision role | Current score |
|---|---|---|---|---|
| Matched mechanism | augmentation-only, clock/no-pose, mask-only, direct signed-frequency, global-warp-only, unsigned reversal | frozen supplied-track pilot | K1/K3/K4 | Not run |
| Closest phase/tempo priors | PAMS, Track-PAMS, SimPer-style, CycleCL-style, DeepPhase-style, generic time-equivariant | frozen supplied-track pilot | K2 | Not run |
| Optional geometric control | masked geometric TSSM | frozen supplied-track pilot | control-only; K9 forbids fallback | Not run |

Published end-to-end scores are context only and cannot be placed in this same-protocol table.

## Current Results

No eligible result exists. The results table is intentionally empty; deterministic fixtures and gate receipts are not results.

| Method | Dataset | Metric | Score | Notes |
|---|---|---|---|---|

Historical v46/v62/v63 artifacts, server outputs, sealed/test material, and existing `results/**` files are ineligible for this contract.

## Key Decisions

- Keep one dominant intervention (`L_WI`) and one exact augmentation-only deletion; do not add learned routing, experts, auxiliary heads, or correction modules.
- Use a dedicated CPython 3.12.13 + NumPy 2.4.6 fixture environment; the NumPy 1.26.4 training environment consumes but never regenerates canonical fixture bytes.
- Use one globally ordered PCG64 real-track weak-view stream with full draw consumption and per-parent hashes; the canonical clean selection remains authoritative.
- Bind K4 to exact `f_i/f_track`, deterministic matched-time cyclic shuffle, and deterministic midpoint unseen resampler without changing its four ratios.
- Preserve exactly three visible validation blocks, K1–K9, 59 jobs/552 GPU-hours, and the pilot claim ceiling.
- If K1/K2/K3/K4/K5/K6/K7/K8 fails, stop the corresponding claim. K9 prohibits an automatic TSSM replacement.

## Status

- [x] Idea selected for one kill-oriented prospective pilot
- [x] Normative local execution contract written
- [x] Claim-driven experiment plan and exact tracker written
- [ ] Gate 0 pack/vault passed
- [ ] Gate 1 numerical/environment fixtures passed
- [ ] Gate 2 selector stability passed
- [ ] Main method implemented
- [ ] Baseline reproduced
- [ ] Any training job authorized or run
- [ ] Representative or full dataset result exists
- [ ] Ablation studies complete
- [ ] K8 novelty gate passed
- [ ] Claim freeze or submission authorized

This contract contains no credentials, connection parameters, private path secrets, or author/contact metadata.
