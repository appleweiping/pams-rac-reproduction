# TempoRAC Round 1 — Senior Method-First Review

CALIBRATION: none

No curated known-good/known-bad research-refine proposals were supplied, so the score is unanchored to a local calibration set. It is calibrated only to the stated top-venue, method-first rubric. The review is same-family and provisional.

## Input-resolution note

The requested `idea-stage/RESEARCH_REVIEW.md` does not exist in the worktree. Per the parent instruction, I used the only available canonical legacy artifact, `idea-stage/RESEARCH_REVIEW.json`, and did not infer or recreate missing Markdown. The four audit/addendum inputs requested under `refine-logs/` were resolved by exact filename at the worktree root. No prior TempoRAC research-refine round review was read.

## Problem Anchor (verbatim)

- **Bottom-line problem:** On externally supplied person-indexed pose tracks, determine whether identity-indexed, window-local soft routing over shared fast, medium, and slow response experts can count each person's repetitions under asynchronous and within-track pace changes without using human per-person count or period annotations in the counting objective, while reconstructing one continuous response and decoding exactly once per identity.
- **Must-solve bottleneck:** The system must identify one response event per
  primitive physical repetition, specialize the three shared experts by
  relative local tempo, prevent cross-person state leakage, and prevent
  overlap-window double counting without consulting evaluator labels.
- **Non-goals:** This work does not introduce MRAC, person-wise output, pose
  counting, tracking, temporal windows, phase learning, mixture-of-experts,
  ACF/FFT, PAMS-TCC, NOLA, or peak detection.  It does not claim first,
  annotation-free, label-free, end-to-end, constant-time, or state of the art.
- **Constraints:** ICASSP 2027; four technical pages plus a references-only
  fifth page; two RTX A6000 GPUs; a partial GT-bbox-assisted supplied-track
  MultiRep train/development cache; no test, sealed, heldout, or historical
  result access; all human count, period, density, and cycle-boundary fields
  remain in an evaluator-only vault; all semantic review is same-family
  provisional GPT-5.6-Sol.
- **Success condition:** The method may advance only if a frozen synthetic
  topology gate establishes one primitive orbit per response winding, all
  seven architecture invariants pass, every expert demonstrates diagonal
  tempo specialization, and a three-seed supplied-track development pilot
  improves the strongest matched nonlocal/control model under piecewise tempo
  drift without materially degrading stationary counting.

## Scorecard

| Dimension | Weight | Score | Weighted contribution |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9.5 | 1.425 |
| Method Specificity | 25% | 6.0 | 1.500 |
| Contribution Quality | 25% | 5.5 | 1.375 |
| Frontier Leverage | 15% | 7.5 | 1.125 |
| Feasibility | 10% | 5.5 | 0.550 |
| Validation Focus | 5% | 6.5 | 0.325 |
| Venue Readiness | 5% | 5.5 | 0.275 |

**Weighted composite:** **6.575 / 10** (displayed: **6.58 / 10**)

**GAP:** The proposal is **2.425 points below** the READY threshold of 9.0. The gap is not caused by problem drift or lack of fashionable machinery. It comes from three coupled method blockers: the training objective does not yet make the teacher's claimed degree (+1) operational on finite data; the sparse event objective does not yet defend the fixed 0.5 one-pass peak decoder; and the routing/specialization comparison is not yet a clean intervention on the same frozen expert responses. Closing those interfaces would materially raise Method Specificity, Contribution Quality, Feasibility, Validation Focus, and Venue Readiness without adding a new module.

## Overall assessment

The proposal preserves the original scientific object unusually well. The shared encoder, shared slow/medium/fast experts, identity-private state, independently computed local tempo cue, response-level soft fusion, positive NOLA, and exactly one per-identity decode all remain present. WARP-PHASE, a scene-level counter, missing-pose sets, and per-person model copies are explicitly excluded. The signal-processing-first choice is also appropriate: an LLM, VLM, diffusion model, or RL loop would not resolve winding number, pulse resolvability, or cross-identity state contamination.

However, the proposal currently presents a theorem-shaped teacher as if its loss realizes the theorem. Under the ideal assumption that a complete primitive orbit is exactly (S^1), the phase map is continuous on the whole orbit, and the decoder is a true left inverse, injectivity does rule out degree magnitude greater than one and orientation selects degree (+1). The implemented learning problem is only finite-sample approximate reconstruction plus correspondence, orientation, and per-edge alias penalties. Those terms do not force circular coverage or a closing-edge winding of (+1). In particular, a discrete trajectory can be encoded injectively on a short arc with no complete winding; the proposed checkpoint rule that minimizes positive phase variation actively favors that failure. The alias penalty also does not rule out a dense double winding because every individual increment can remain below one quarter cycle.

The teacher is therefore a plausible kill-gated hypothesis, not yet an identified degree-one teacher. This is revisable without changing the Problem Anchor: remove the minimum-positive-variation selector, define the synthetic orbit evaluator independently of the model, include the closing edge, measure signed winding and circle coverage, and require degree (+1) on held-out primitive orbits and unseen resamplers while explicitly attacking degree 0, -1, +2, half-cycle seam, and code/decoder bypass solutions. The primitive coordinate may be visible to the synthetic evaluator but not to the learner. On natural tracks, reconstruction and collision diagnostics remain diagnostics; they cannot prove that the human annotation convention equals the primitive pose orbit. A natural teacher-only count mismatch must kill the route, not trigger label-based repair.

## Required revisions for dimensions below 7

### Method Specificity — 6.0 / 10

**Specific weakness:** Several formulas are concrete, but the decisive interfaces are not implementation-complete. The teacher architecture, delay lags, static-code bottleneck, closing-edge convention, exact winding/coverage estimator, reconstruction tolerance, loss weights, and checkpoint rule are missing. `TCN/GRU` leaves the expert architecture ambiguous. The ACF/NUDFT frequency grid, binning, normalization, reliable-window threshold, robust reference estimator, and no-support behavior are only partly fixed. `L_spec` does not explicitly restrict (e) to window (ell), and `L_fuse` and `L_cont` omit complete mask/alignment normalization.

**Concrete fix:** Freeze one executable interface sheet before code: exact tensor shapes and masks; one expert class (causal depthwise-separable TCN or GRU, not both); phase-teacher layer widths, lags, code dimension, and forbidden bypasses; source-frequency grid and ACF/NUDFT equations; reliable-window predicate and reference statistic; fully indexed, denominator-safe losses; all synthetic-selected constants; and deterministic checkpoint precedence. Add an evaluator-only signed winding test over the complete synthetic orbit including the closing edge. Remove positive-variation minimization.

**Priority:** CRITICAL

### Contribution Quality — 5.5 / 10

**Specific weakness:** The proposal says one dominant contribution, but the paper currently asks the reader to absorb three substantial stories: a topological response-unit teacher, tempo-routed expert specialization, and NOLA-plus-peak ordering. The teacher is called an enabling non-contribution yet carries the deepest mathematical claim and one of two new trainable systems. In a four-page ICASSP paper, this creates contribution sprawl and makes the actual novelty look like an engineering conjunction of established components.

**Concrete fix:** Make the paper-level claim singular: after a separately passed and frozen response-unit contract, does a cue-only relative local-tempo router select genuinely specialized shared response scales under within-track drift? Treat the teacher as a preregistered supervision contract and NOLA/one-decode as fixed correctness plumbing. Retain the reconstruction-ordering result only as a compact necessary sanity check, not a parallel contribution. If the teacher itself becomes the claimed novelty, that is a new paper anchor and requires a separate review rather than silently expanding this one.

**Priority:** CRITICAL

### Feasibility — 5.5 / 10

**Specific weakness:** The 100-A6000-hour ceiling is plausible only after the contracts exist, but the proposed one-week implementation estimate is not credible for a new teacher, masked irregular-clock spectral cue, three stateful experts, NOLA, fixed decoder, firewall packer, evaluator vault, and all causal instrumentation. The data audit also shows that the current label-mixed pickles cannot enter the training process, the feature/vault packer is absent, and the nuisance-only K4 schema remains open. No current tree implements TempoRAC.

**Concrete fix:** Stage feasibility as hard stops: (1) specification-only tensor/loss contracts; (2) CPU synthetic degree and pulse gates; (3) seven-invariant deterministic unit tests; (4) feature-only/vault receipts and nuisance schema; (5) one-seed overfit/sanity run; only then (6) the bounded three-seed pilot. Give each stage an artifact schema and wall-clock/GPU cap. Do not count optional PAMS-TCC initialization or `L_cont` in the first executable route.

**Priority:** CRITICAL

### Validation Focus — 6.5 / 10

**Specific weakness:** The two core claim blocks are directionally right, but the handoff list expands into a broad baseline matrix, and the routing comparison is causally ambiguous if every router arm retrains the experts. Cue blocking/shuffling then changes both optimization and routing, so it does not isolate whether the local cue selected the right already-specialized response. The one-head capacity match is also not defined as parameter-, FLOP-, and receptive-field-matched.

**Concrete fix:** Make the minimal causal package three tests. First, train the canonical expert bank once, freeze the cached expert responses, and intervene only on (g): local, global, uniform, blocked, and within-track shuffled cue. Second, report the held-out expert-by-tempo error matrix on unseen resamplers and require diagonal best with a precisely parameter/FLOP-matched one-head control. Third, run the identical local response tensor through independent window decode versus positive NOLA plus the fixed one-pass decoder under grid shifts. Retrained system arms can be secondary efficacy checks, not the causal proof.

**Priority:** IMPORTANT

### Venue Readiness — 5.5 / 10

**Specific weakness:** The proposal is timely for ICASSP but not submission-shaped. The residual novelty remains an unproven interaction; complete TWCRAC overlap is unresolved; there are zero eligible method results; and the strongest method claim currently depends on an unclosed teacher and decoder contract. The title and acronym would overstate maturity if frozen now.

**Concrete fix:** Do not draft around a broad “TempoRAC system” claim. First close the response-unit and decoder gates, then require one non-additive routing result against teacher-only, one matched head, uniform/global routing, and the closest protocol-matched prior. Keep literature/title clearance, server authorization, and paper authorization as separate later decisions.

**Priority:** IMPORTANT

## Degree-one teacher ruling

**Ruling: conditionally plausible, not currently identified.** The ideal topological argument is coherent, but the current finite-sample objective does not establish its premises. `L_inv` prevents a literal constant phase only when the decoder has no bypass and the static code cannot carry time; it does not force a complete circle. `L_view` fixes correspondence but not winding. `L_orient` fixes sign only after nonzero winding exists. `L_alias` bounds individual increments but not total winding. Minimum positive variation is unsafe because it rewards arc compression.

The minimal repair is a test contract, not another learned module:

1. Freeze asymmetric primitive-orbit generators and held-out resamplers; keep their primitive coordinate evaluator-only.
2. Evaluate the ordered phase sequence plus the closing edge, with deterministic unwrap, signed integer winding, circular coverage, collision, reconstruction, and seam continuity metrics.
3. Include explicit degree 0, -1, +2, harmonic, symmetric-orbit, static-code, and decoder-bypass attacks.
4. Require the preregistered topology thresholds before natural target generation; abstain on symmetric/non-identifiable fixtures.
5. Freeze the teacher artifact hash. On natural data, a teacher-only mismatch is a terminal scientific failure, not a reason to inspect or train on counts.

## One-final-peak-pass ruling

**Ruling: preserve it; do not replace it with an integral decoder.** The revised proposal correctly uses half-open phase-crossing impulses, reconstructs one continuous response, and invokes one connected-component decoder per identity. This is faithful to the original response-event claim and is more causally diagnostic than silently switching to a count-mass integral.

It is not yet defended by the present loss. A one-edge target is extremely sparse; ordinary BCE plus soft expert mixing and NOLA does not guarantee that every event remains above 0.5, every inter-event valley remains below 0.5, or close events remain separate. “0.5 follows from a binary target” is an ideal calibration statement, not an implementation guarantee.

Keep the peak decoder and add no learned threshold. Instead, freeze a teacher-derived pulse-resolvability objective and gate: mask-normalized BCE plus fixed event/valley margins around each pseudo-event; deterministic invalid-gap handling; minimum tested event spacing; plateau-center tie-breaking; half-hop window shifts; pauses; duplicate clocks; padding; and adversarial split/merge packs. Require exactly one connected component per teacher event and zero extras at threshold 0.5 on held-out synthetic packs. If this fails, reject the strict peak contract and the TempoRAC route. Do not substitute the earlier integral decoder under the same paper identity.

## Cue-only routing and matched-expert causal ruling

**Ruling: the design is causally promising but the current experiment wording is insufficient.** A fixed cue-only router is a strong choice because it prevents a latent residual from bypassing tempo evidence. Equal-width experts and balanced synthetic tempo strata are also appropriate. But `L_spec` and `L_fuse` can still train all experts toward the same event function, and independently retrained router controls conflate routing with optimization.

To support the original claim, align the synthetic stratum assignment (p^*) to the exact same relative-log-tempo anchors used by (g), declare whether overlapping windows reweight an edge, and freeze the canonical expert bank for the decisive router intervention. The claim survives only if: (a) each expert is diagonal-best on an unseen-resampler tempo matrix; (b) local (g) beats global/uniform/blocked/shuffled (g) on the same response tensor; (c) cue shuffling removes the local-over-global advantage; and (d) the three-expert system beats a total-capacity/FLOP-matched one-head control specifically under within-track drift without the stationary penalty. These tests establish a causal routing story without adding a learned router or a fourth expert.

## Exact implementation and test-gate executability

**Current status: not executable as a frozen gate stack.** The gates are conceptually strong, but several lack the exact fixtures, observables, or precedence needed for a pass/fail implementation.

| Gate | Current ruling | Exact closure needed |
|---|---|---|
| G0 feature/vault firewall | BLOCKED | Implement feature-only shards, separately permissioned evaluator vault, forbidden-key pre-deserialization rejection, canonical-source component manifest, and hash receipts. |
| G1 degree-one teacher | BLOCKED | Freeze generator hashes, learner-hidden primitive coordinates, complete-orbit closing-edge winding/coverage metrics, attack fixtures, numeric tolerances, architecture, and checkpoint precedence. |
| G2 seven invariants | PARTIALLY SPECIFIED | Turn each property into an instrumented assertion: shared object IDs; equal parameter/FLOP receipts; private `(identity, expert)` buffers; outside-window cue invariance; simplex gates and fusion-before-decode call trace; positive finite NOLA denominator; decoder call count equal to complete identities. |
| G3 routing/specialization | BLOCKED AS CAUSAL TEST | Define same-frozen-response router interventions, exact anchor/stratum mapping, unseen-resampler bins, one-head matching, paired unit, and decision rule. |
| G4 pulse/NOLA/peak | BLOCKED | Freeze pulse-margin loss, threshold-0.5 calibration gate, event-spacing fixture domain, invalid-gap semantics, grid shifts, split/merge oracle, and exactly-once decoder instrumentation. |
| G5 supplied-track pilot | NOT AUTHORIZED | It becomes executable only after G0-G4 pass and the three-seed source-component bootstrap evaluator is frozen. |

The WARP-PHASE Gate-2 failure is binding only against reuse of that selector/route. It is not evidence that the full-distribution TempoRAC cue fails, and it cannot be relabeled as a TempoRAC result.

## Simplification Opportunities

1. Delete `L_cont` from the first implementation. The frozen teacher targets, specialization loss, and fused-response loss already test the core mechanism; continuity can be reconsidered only after the canonical route passes.
2. Keep PAMS-TCC initialization out of the canonical first run. Use the same small shared encoder for all arms and add PAMS-TCC only as a deletion/initialization control after the mechanism works.
3. Use one predeclared irregular-clock tempo distribution for routing—prefer the nonuniform Fourier distribution—and retain ACF as an agreement diagnostic unless a synthetic, label-free necessity gate proves that geometric fusion is needed.
4. Treat positive NOLA and one decode as fixed correctness operators, not a supporting paper contribution. This leaves one dominant claim: relative local tempo selects specialized shared response scales.

## Modernization Opportunities

NONE. The proposal is appropriately frontier-aware through frozen-backbone reuse, teacher-to-student pseudo-targeting, and controlled specialization. An LLM, VLM, diffusion model, RL policy, or inference-time search loop would not naturally solve degree, pulse, or identity-state identifiability and would worsen the four-page contribution budget.

## Drift Warning

NONE. The Problem Anchor is preserved. The degree-one teacher is an enabling attempt to make the original response unit identifiable, not a substitute task. Two future changes would constitute drift and require a new review: promoting the teacher into a separate primary contribution, or replacing the required one-final-peak-pass with an integral count decoder while continuing to call the method the same TempoRAC proposal.

## Remaining action items

1. **CRITICAL — Freeze the teacher topology contract.** Remove minimum-positive-variation selection; specify the teacher/code architecture and evaluator-only complete-orbit degree, coverage, collision, seam, reconstruction, and attack gates.
2. **CRITICAL — Close the peak contract without changing decoders.** Add a teacher-derived pulse/valley margin objective and held-out threshold-0.5 split/merge/NOLA gate; failure kills the route rather than selecting an integral decoder.
3. **CRITICAL — Write one executable interface sheet.** Fix tensor shapes, masks, clock units, expert class, frequency grid, reliability predicate, loss index sets, denominators, constants, checkpoint precedence, and call-trace assertions.
4. **CRITICAL — Build the physical feature/vault boundary before any data-bearing training.** The current label-mixed pickle is ineligible; the nuisance-only K4 schema also remains open.
5. **IMPORTANT — Make routing causality a frozen-response intervention.** Align (p^*) and (g), evaluate local/global/uniform/blocked/shuffled gates on the same cached expert responses, and define the total-capacity/FLOP-matched one-head control.
6. **IMPORTANT — Collapse the paper story to one claim.** Teacher is a prerequisite contract; NOLA/one-decode is correctness plumbing; `L_cont` and optional PAMS initialization are deleted from the first route.
7. **IMPORTANT — Preserve authorization boundaries.** Complete TWCRAC/title checks separately; do not infer novelty, server readiness, paper readiness, or performance from this review or from WARP-PHASE artifacts.

## Authorization boundaries

- **Method-refinement verdict:** REVISE.
- **Novelty clearance:** NOT GRANTED. The prior ledger is same-family provisional and complete TWCRAC overlap remains unresolved.
- **Title/acronym freeze:** NOT AUTHORIZED.
- **Server or data-bearing training:** NOT AUTHORIZED by this review. The feature/vault and executable gate stack are incomplete.
- **Paper drafting, submission, or positive performance claims:** NOT AUTHORIZED. There are zero eligible TempoRAC method results.
- **Local specification/unit work:** scientifically appropriate after the revisions above, but this review is not a server launch receipt.
- **WARP-PHASE:** remains a terminal failed route and may not be reused or substituted.

## Verdict

**REVISE**

The original direction is worth preserving, and no frontier replacement is needed. It is not READY because the response unit, pulse decoder, and causal router proof are not yet executable as claimed. It is not RETHINK because each blocker can be closed by tightening or deleting interfaces while keeping the immutable Problem Anchor and the seven-property graph intact.

Reviewer metadata: `reviewer_model=gpt-5.6-sol`; `reviewer_family=openai`; `review_independence=same-family`; `acceptance_status=provisional`.
