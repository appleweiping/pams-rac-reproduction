# Review Summary

**Problem:** Test whether an explicit local time-warp derivative law on signed phase increments improves supplied-track per-person repetition counting under within-track tempo drift, without count/period-label access by the learner.
**Initial approach:** A local phase-flow model with identity-private state, later narrowed to one discrete warp-integral equivariance loss over a shared phase backbone.
**Date:** 2026-08-15
**Rounds:** 5 / 5
**Final score:** 8.7 / 10
**Final verdict:** `REVISE`
**Completion reason:** Maximum research-refine rounds reached.
**Evidence ceiling:** Proposal-only; zero eligible method results; same-family provisional review.

## Problem Anchor

The following anchor is copied verbatim from `idea-stage/RESEARCH_REVIEW.json` → `exact_problem_anchor` and is immutable for later refinement rounds.

- **Research question:** On GT-bbox-assisted supplied pose tracks, does enforcing an explicit local time-warp derivative transformation law on signed phase increments in identity-private state improve per-person repetition counting under predeclared within-track tempo drift, relative to matched augmentation-only, PAMS/Track-PAMS, SimPer-style, CycleCL-style, DeepPhase-style, and generic time-equivariant controls, while the counting objective reads no per-person count or period labels?
- **Population:** Checksum-frozen, canonical-source-disjoint MultiRep train/development data available through the partial supplied-pose cache, explicitly treated as pilot-only.
- **Unit of analysis:** Video, with per-person outputs aggregated by video-first AvgMAE and AvgOBO equations; source-connected components are the resampling clusters.
- **Primary estimand:** Paired development-set change in video-level AvgMAE between the derivative-law model and the identical augmentation-only model under the predeclared within-track tempo-drift diagnostic.
- **Secondary estimands:**
  - Supplied-track Period-mAP, AP50, and AP75 on a frozen explicit scale, labeled as supplied-track diagnostics rather than published end-to-end comparisons.
  - Clean-to-warp degradation and phase-correspondence error.
  - Untouched-person response and count change under a one-person tempo intervention.
- **Excluded questions:**
  - Whether local windows, phase, temporal equivariance, pose SSL, TSSM, private memory, overlap-add, or per-person output are individually novel.
  - Whether the method is end-to-end, annotation-free, state of the art, or deployment-ready.
  - Whether a supplied-track partial-cache result establishes predicted-track MRAC.

- **Bottom-line problem:** The research question above, without substitution by scene-total counting, predicted-track claims, or a generic representation-learning objective.
- **Must-solve bottleneck:** A person's repetition phase must remain locally coherent when tempo changes within the track, while the learning process cannot read per-person count, density, period, or cycle-boundary labels and cannot obtain the apparent effect from the known resampling clock alone.
- **Non-goals:** Building a detector, pose estimator, tracker, learned tempo router, learned tempo experts, a new TSSM method, a deployment system, or an official MultiRep main-result pipeline. Scene-total count is diagnostic only.
- **Constraints:** ICASSP 2027 4+1-page scope; partial supplied-pose cache only; no FPS or timestamps beyond `sampled_frame_indices` and `source_length`; duplicated sampled indices; missing joints; incomplete pose coverage; three independent seeds; no sealed test access; no efficacy use of v46/v62/v63.
- **Success condition:** Both clauses of the frozen primary decision rule must pass: at least a 5% relative reduction in piecewise-warp development video-first AvgMAE versus the identical augmentation-only control, **and** a paired source-component bootstrap 95% confidence interval for the improvement that excludes zero. This is an AND rule, never an AvgMAE-or-Period-mAP escape hatch.

Anchor-text SHA-256: `619148c765175167da093d9b054bd2b43fae0e5e97d981fb9748ef57f4d7dba7`.

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9.3 | 6.0 | 6.4 | 8.7 | 5.4 | 5.8 | 5.8 | 6.9 | `REVISE` |
| 2 | 9.6 | 7.3 | 7.6 | 9.0 | 6.8 | 6.6 | 6.5 | 7.9 | `REVISE` |
| 3 | 9.8 | 8.4 | 8.0 | 9.2 | 7.8 | 8.6 | 7.4 | 8.5 | `REVISE` |
| 4 | 9.9 | 8.2 | 8.0 | 9.2 | 8.0 | 8.0 | 7.4 | 8.5 | `REVISE` |
| 5 | 9.9 | 8.8 | 8.0 | 9.2 | 8.2 | 8.6 | 7.7 | 8.7 | `REVISE` |

Reported overall scores are preserved exactly. Weighted composites were 6.92, 7.85, 8.53, 8.485, and 8.700.

## Round-by-Round Resolution Log

| Round | Main reviewer concerns | Exact closure in the next proposal | Result after re-review |
|---:|---|---|---|
| 1 | Ambiguous midpoint/integral target; unidentified winding; zero-boundary Hann; unfrozen eligibility/scorer; broad auxiliary stack | One oriented signed-overlap integral; pseudo-cycle identifiability gate; strictly positive interval NOLA; count-blind eligibility; exact component bootstrap; one shared phase base and staged comparisons | Major simplification; sign symmetry, semantic winding, normalized AvgMAE, recurrence, component cardinality, budget, and K8 still open |
| 2 | Clean/warped sign mismatch; pseudo-cycle not semantic repetition; overlapping-window recurrent commits; raw MAE mislabeled AvgMAE; nine-component and shortcut-budget gaps | Raw signed increments on both branches; post-freeze no-correction harmonic gate; one full-sequence recurrent commit; normalized person-first/video-first AvgMAE; all-nine Gate-0 assertion; 59-job/552-GPU-hour ledger | All named repairs closed except optimizer batching details, raw-period schema binding, K4 input precision, and external K8 |
| 3 | Full-sequence autograd/batching; scalar evaluator-period provenance; exact K4 input schemas; Stage-D control flow; blocked TWCRAC receipt | Eight complete identities per fresh graph, four-step accumulation; raw `[s,e)` float64 median period; exact clock/no-pose and mask-only contracts; Stage D adjudicates K2; truthful blocked K8 receipt | Core optimizer/period/control flow closed; supplemental seven-interface implementation audit remained |
| 4 | Root/scale missingness; FFT256 grid/numerics; selector-to-decoder binding; fixture population; warp tensors; K4 code/decision contracts; packer schema | COCO17 fail-closed normalization; inclusive source-clock FFT256 and exact ACF/FFT; canonical selector plus binary `g_j`; deterministic warp interpolation; fixed 1,000-fixture layout; exact K4 vectors/ratios; 617-byte desensitized schema | Four interfaces and packer contract closed; three bounded execution contracts remained underdetermined |
| 5 | Deterministic real weak-view RNG; complete fixture generator plus NumPy environment witness; exact K4 `f_track`/shuffle/resampler constructors; full TWCRAC source | No further refinement round permitted. These items are carried unchanged as final blockers | Final 8.7 `REVISE`; maximum rounds reached; no drift, implementation, eligible result, or novelty clearance |

## Overall Evolution

- The method contracted from a multi-loss phase system to one proposed intervention: `L_WI`, a stop-gradient discrete signed warp-integral equivariance loss.
- Treatment and augmentation-only now share one 225,026-parameter learner, all warps, all data, optimizer, recurrence, NOLA, decoder, and evaluation; the treatment deletion is exact.
- Phase semantics are separated cleanly: the learner uses a selector-defined pseudo-cycle, while an evaluator-only harmonic gate can reject but never correct predictions.
- The proposal froze normalized video-first AvgMAE, all-nine-component paired bootstrap, 59-job/552-GPU-hour ceiling, K1–K9, and exactly three visible validation blocks.
- No LLM/VLM/diffusion/RL component was added because exact transformation metadata is the natural supervision.
- Drift warning remained `NONE` in all five reviews.

## Final Remaining Blockers

1. **Local — weak rejection view:** Freeze the Gate-2 RNG/seed, checksum-sorted track and parent-start traversal, draw shapes/order, and perturbation hashes.
2. **Local — fixture generator/environment:** Complete all RNG/formula/shape/transform semantics, then use locked NumPy 1.26.4 or bind a dedicated NumPy 2.4.6 environment plus cross-environment witness.
3. **Local — K4 construction:** Define the exact `f_track` denominator, matched-time shuffle permutation/seed, and unseen-resampler kernel/grid/schedule/seed/receipt.
4. **External — K8:** Obtain and hash the complete TWCRAC primary source, inspect method/loss/training/evaluation in full, and rerun independent equivalence adjudication. Current status remains `BLOCKED`, `NOT_PASSED_BLOCKED`, `claim_freeze=BLOCKED`, `novelty_clearance=false`, with zero pages covered.
5. **Prospective execution:** Gates 0–5 and K1–K7 have not run and may fail. No efficacy direction is recorded as an observed result.

## Final Status

- Anchor status: verbatim preserved.
- Focus status: tight; one dominant loss, one exact deletion, one supporting identity-isolation diagnostic.
- Modernity status: appropriately frontier-aware without a forced foundation-model component.
- Local specification status: not closed; three bounded execution blockers remain.
- Evidence status: proposal-only; zero eligible method results.
- Review status: same-family OpenAI reviewer, calibration `none`, provisional acceptance ceiling.
- Claim status: K8 and claim freeze blocked.
- Completion status: Phase 5 complete because `MAX_ROUNDS=5` was reached, not because the proposal became READY.
- Canonical clean proposal: `refine-logs/FINAL_PROPOSAL.md`.
- Verbatim raw reviews remain in `refine-logs/round-1-review.md` through `round-5-review.md` and are reproduced in the refinement report.

