# Refinement Report

**Problem:** Test whether an explicit local time-warp derivative law on signed phase increments improves supplied-track per-person repetition counting under within-track tempo drift without learner access to per-person count or period labels.
**Initial approach:** Identity-private local phase flow, narrowed to one discrete warp-integral equivariance objective over a shared phase backbone.
**Date:** 2026-08-15
**Rounds:** 5 / 5
**Final score:** 8.7 / 10
**Final verdict:** `REVISE`
**Completion reason:** `MAX_ROUNDS=5` reached.
**Evidence status:** Proposal-only; zero eligible method results.
**Review ceiling:** Same-family OpenAI reviewer, provisional acceptance status, calibration `none`.

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

## Output Files

- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Score ledger: `refine-logs/score-history.md`
- Final state: `refine-logs/REFINE_STATE.json`

## Score Evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Overall | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9.3 | 6.0 | 6.4 | 8.7 | 5.4 | 5.8 | 5.8 | 6.9 | `REVISE` |
| 2 | 9.6 | 7.3 | 7.6 | 9.0 | 6.8 | 6.6 | 6.5 | 7.9 | `REVISE` |
| 3 | 9.8 | 8.4 | 8.0 | 9.2 | 7.8 | 8.6 | 7.4 | 8.5 | `REVISE` |
| 4 | 9.9 | 8.2 | 8.0 | 9.2 | 8.0 | 8.0 | 7.4 | 8.5 | `REVISE` |
| 5 | 9.9 | 8.8 | 8.0 | 9.2 | 8.2 | 8.6 | 7.7 | 8.7 | `REVISE` |

Weighted composites: 6.92, 7.85, 8.53, 8.485, and 8.700. Existing scores and verdicts are unchanged.

## Round-by-Round Review Record

| Round | Main concerns | Exact changes/closures | Result |
|---:|---|---|---|
| 1 | Unique integral, winding, NOLA, data eligibility, exact scorer, excessive auxiliaries, K8 | Oriented overlap-integral target; target-free pseudo-cycle gate; positive interval NOLA; count-blind eligibility; exact bootstrap; one shared base and staged controls | Partial: several central execution/semantic items remained |
| 2 | Sign mismatch, semantic-cycle overclaim, recurrent commit, normalized AvgMAE, all-nine population, shortcut budget | Raw sign on both branches; evaluator-only no-correction harmonic gate; one chronological commit; normalized scorer fixture; Gate-0 cardinality; 59 jobs/552 GPU-hours | Local progress; optimizer and schema interfaces plus K8 remained |
| 3 | Full-track autograd algorithm, period source, K4 schemas, Stage-D flow, TWCRAC | Fresh 8-track graphs, four-step accumulation; raw `[s,e)` median; exact K4 inputs; Stage D adjudicates K2; blocked K8 receipt | Named Round-3 actions closed; seven supplemental interfaces remained |
| 4 | Root/scale, FFT grid/numerics, selector binding, fixture bytes, warp tensors, K4 codes/ratios, packer | Fail-closed COCO17 normalization; exact source-clock FFT256/ACF; binary `g_j`; deterministic interpolation; fixed fixture layout; exact K4 vectors/decision table; 617-byte schema | Four interface families closed; three bounded execution contracts remained |
| 5 | Weak-view RNG; complete fixture generator/environment; K4 corruption constructors; unavailable TWCRAC full text | No Round-5 refinement was allowed; blockers carried into final proposal status | Final 8.7 `REVISE`, no drift, maximum rounds reached |

## Final Proposal Snapshot

- The canonical clean proposal is `refine-logs/FINAL_PROPOSAL.md`.
- It is mechanically extracted from the clean proposal section of `round-4-refinement.md` (canonical input SHA-256 `2031d5c84b8c6c329dd0e43e5b8a09210ed9a3e8b17b47578f2f878a4c2b4e0f`).
- Review/refinement headers, Anchor Check, Simplicity Check, and Changes Made were removed.
- The title is normalized to `Research Proposal`.
- Its proposal-status section carries the three exact local execution blockers and the external K8 block identified by the canonical Round-5 review (SHA-256 `81feba37b6618e4f19af3990440c1eb8c8afe3a2e5838e8c675d82f096129282`).
- It contains one Problem Anchor and makes no observed-result claim.

Final thesis:

- A clean supplied identity track defines a raw signed phase measure.
- A known target-to-source warp maps each target interval to an oriented integral of the clean measure.
- Treatment and augmentation-only share all machinery and exposure; treatment alone receives `L_WI`.
- The primary endpoint remains normalized video-first AvgMAE with the 5%-AND-positive-component-CI rule.
- The work remains a GT-bbox-assisted AlphaPose supplied-track partial-cache prospective pilot.

## Method Evolution Highlights

1. Replaced an initially broad phase-system story with one exact discrete warp-integral loss and one deletion.
2. Made recurrence, interpolation, masking, pseudo-cycle selection, NOLA, evaluator periods, AvgMAE, component bootstrap, and compute budget fail-closed and largely executable.
3. Kept modern leverage disciplined: exact transformation supervision and causal/matched controls, without adding a foundation-model component.
4. Preserved the partial-cache/GT-assisted/supplied-track claim ceiling and zero-result status across every round.

## Pushback / Drift Log

| Round | Reviewer pressure or risk | Author response | Outcome |
|---:|---|---|---|
| 1 | Add/retain multiple auxiliary mechanisms | Deleted the bespoke loss stack and retained only a shared established base plus `L_WI` | Accepted; contribution narrowed |
| 2 | Treat stable ACF/FFT output as physical repetition | Rejected semantic overclaim; named it a pseudo-cycle and added post-freeze rejection only | Anchor preserved |
| 2–5 | Add fashionable foundation-model machinery | Rejected as confounded and unnecessary for an exact signal-transformation law | Modernity remained appropriate |
| 3–5 | Infer distinctness from unavailable TWCRAC method | Rejected absence inference; retained `BLOCKED/NOT_PASSED_BLOCKED` | K8 still blocks claim freeze |
| All | Promote private memory or masked TSSM if the main mechanism fails | Kept both diagnostic/control-only under K9 | No fallback contribution sprawl |

Drift warning was `NONE` in every review.

## Remaining Weaknesses and Blockers

### Three bounded local execution blockers

1. **Weak rejection view:** no frozen Gate-2 seed-to-track/parent mapping, traversal order, draw shapes/order, or perturbation hash.
2. **Fixture generator/environment:** RNG positions, harmonic/symmetric equations, stochastic shapes, and transform ordering remain incomplete; the proposal's NumPy 2.4.6 clause conflicts with the supported project lock at 1.26.4 absent a dedicated environment witness.
3. **K4 construction:** `f_track` lacks a named denominator, and the matched-time shuffle plus unseen-resampler constructors lack exact scope/kernel/grid/schedule/seed receipts.

### External blocker

TWCRAC full-text coverage is zero pages. The adjudication remains `BLOCKED`, `k8.decision=NOT_PASSED_BLOCKED`, `claim_freeze=BLOCKED`, and `novelty_clearance=false`. It establishes neither collision nor distinctness.

### Prospective execution risk

The packer, permissions, eligibility table, numerical bytes, selector gates, training, K1–K7, comparator stages, and evaluator runs have not executed. Zero eligible results exist, and historical v46/v62/v63 artifacts cannot be used as efficacy evidence.

## Raw Reviewer Responses

The following are the complete stored Markdown responses from the same reviewer thread. They are reproduced without summarization inside each disclosure block.

<details>
<summary>Round 1 Review — full verbatim stored response</summary>

# Round 1 Research-Refine Review: WARP-PHASE

**Reviewer:** GPT-5.6-Sol (`/root/research_refine_reviewer`)  
**Review independence:** same-family  
**Acceptance status:** provisional  
**Evidence status:** proposal-only; no eligible result is treated as evidence  
**Venue target:** ICASSP 2027 (4+1 pages)

## Executive decision

The proposal preserves the exact supplied-track MRAC problem and has made a valuable focusing move: one explicit derivative-law loss is the sole proposed mechanism, while private state and masked TSSM are correctly demoted to diagnostic/control roles. It is also appropriately modern without forcing a foundation model.

It is not implementation-ready. The core optimization target is still ambiguous between a midpoint derivative approximation and an unspecified “exact” integral; the phase gauge fixes offset/sign but not winding number, so one physical repetition is not identified with one (2\pi) turn; the stated Hann NOLA can erase boundary mass; the partial-cache population still lacks frozen handling of ambiguous GT-assisted slots and duplicate-clock interpolation; and the 5%+CI endpoint lacks a single executable aggregation/bootstrap algorithm. These are blocking method/protocol issues, not requests for more experiments.

## Scores

| Dimension | Weight | Score / 10 | Weighted contribution |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9.3 | 1.395 |
| Method Specificity | 25% | 6.0 | 1.500 |
| Contribution Quality | 25% | 6.4 | 1.600 |
| Frontier Leverage | 15% | 8.7 | 1.305 |
| Feasibility | 10% | 5.4 | 0.540 |
| Validation Focus | 5% | 5.8 | 0.290 |
| Venue Readiness | 5% | 5.8 | 0.290 |

**WEIGHTED COMPOSITE: 6.92 / 10 (reported: 6.9 / 10)**  
**CALIBRATION: none**

**GAP:** No project- or skill-supplied set of three curated good and three curated bad research proposals was available, so this score is unanchored and no exemplar comparison is fabricated. Relative to the rubric’s READY bar, the proposal is closest to a strong, unusually disciplined pilot specification rather than a finished method plan: its problem fidelity, restraint, and prospective kill logic are near-ready, but the mathematical operator actually optimized, the phase-to-count identification, the boundary-safe decoder, the partial-cache eligibility contract, and the inferential algorithm remain materially underspecified. Closing those items—not adding modules or baselines—is the shortest path toward a score above 9.

## Dimension review

### 1. Problem Fidelity — 9.3/10

The anchor is preserved verbatim. The treatment/control contrast directly tests whether an explicit local warp law adds value beyond exposure to the same warps, on per-person supplied tracks, with video-first count metrics and source-component resampling. The proposal does not drift into scene totals, predicted-track claims, generic representation learning, TSSM novelty, or historical v46/v62/v63 efficacy. The GT-bbox-assisted, supplied-track, partial-cache ceiling is repeated consistently.

The only caution is that later exclusion of ambiguous supplied slots or low-coverage tracks could silently change the population. Any eligibility rule must therefore be frozen as part of the anchor-preserving protocol and reported as selection, not introduced after count evaluation.

### 2. Method Specificity — 6.0/10

**Specific weakness:** The proposal presents two different derivative targets. The displayed midpoint law

\[
r_j^\tau \approx a_j\,\mathcal I(r, (\tau(q_j)+\tau(q_{j+1}))/2)
\]

is only an approximation when a target interval crosses a clean sampling cell or a warp breakpoint. The text then says the implementation “may use” an integral target, without defining the clean-rate interpolant, oriented integration, support mask, wrapping residual, gradient direction, or breakpoint convention. Duplicate source indices make an ordinary interpolation call non-unique. A pause, reversal, or interval spanning a masked cell therefore has no single executable target.

**Concrete fix:** Make one oriented discrete integral the only training target. After a frozen duplicate-index collapse rule, define a valid piecewise-constant clean rate (\tilde r(s)=r_k) on each valid distinct-clock cell ([q_k,q_{k+1})). For target interval (j), use

\[
I_j^\tau=\int_{\tau(q_j)}^{\tau(q_{j+1})}\tilde r(s)\,ds,
\qquad
\mathcal L_{\mathrm{deriv},j}=\rho\!\left(\operatorname{wrap}(\delta_j^\tau-\operatorname{sg}(I_j^\tau))\right).
\]

Implement the integral as signed overlap lengths with clean cells; mask the whole target interval if any required support is invalid; mask when |(I_j^\tau)| reaches (\pi-\varepsilon); use the same half-open convention at breakpoints; and log the derivative-form quantity only as (I_j^\tau/\Delta q_j). State whether clean/warped roles are alternated and exactly where stop-gradient is applied. Reversal is then an oriented negative integral and a pause is exactly zero. The midpoint formula should not remain an alternative optimization path. **Priority: CRITICAL.**

**Specific weakness:** The model does not identify the unit of phase. A phase offset and orientation rule fixes a gauge but cannot distinguish one, two, or one-half turns per physical repetition. The current base loss can preserve arbitrary pairwise phase differences across weak views, reconstruction can bypass phase through the latent, and variance regularization only prevents a constant. Consequently (\sum_j c_j|\delta_j|/(2\pi)) can equal (k) times the true count for an unconstrained harmonic (k). Evaluator-only half/double diagnostics detect this after the fact but do not define the decoder prospectively.

**Concrete fix:** Replace the mixture of loosely specified base terms with one frozen, implementable phase-learning backbone shared by treatment and control, including exact temporal-positive construction and a label-free fundamental-selection rule. The rule must select the lowest nonzero winding stable across nested windows/views and consistent with the chosen periodic reconstruction or correspondence signal; its tolerances must be set on synthetic signals and training-only components, never evaluator periods. Require a pre-full-run identifiability gate showing (k=1) recovery under known synthetic warps, pauses, missingness, and harmonic distractors. If no real-track fundamental selector can be stated without count/period labels, the phase-mass count head is not identified and the method should be killed rather than calibrated post hoc. **Priority: CRITICAL.**

**Specific weakness:** “Fixed Hann NOLA” is not boundary complete as written. A conventional Hann window is zero at its endpoints; an interval covered only at a sequence boundary can therefore have a zero denominator and lose true mass despite the added epsilon.

**Concrete fix:** Freeze interval-centered strictly positive weights (for example, a sine-squared window evaluated at interval midpoints), or pad/add boundary windows so every valid interval has positive total weight. Define the (L-1) interval indexing, stride, final short-window policy, and denominator assertion. The grid-shift test must be an exact unit test of this chosen construction, not a later diagnostic for an unresolved convention. **Priority: CRITICAL.**

**Specific weakness:** Several “frozen” interfaces are not numerically frozen: joint-confidence threshold, minimum valid joints, duplicate-index equality/collapse rule, warp breakpoint/slopes distribution, robust penalty, epsilon/alias margin, phase-head and recurrent dimensions, window stride, and activity target construction.

**Concrete fix:** Add one implementation table containing shapes, numeric constants, sampling distributions, masking order, loss formulae, and trainable/frozen parameter counts. Distinguish values fixed by engineering invariants from values selected on training-only diagnostics. **Priority: IMPORTANT.**

### 3. Contribution Quality — 6.4/10

**Specific weakness:** The paper story is focused, but the mathematical law itself is the ordinary chain rule; the plausible contribution is its particular discrete, identity-indexed MRAC objective and evidence that it helps beyond matched augmentation. Almost every surrounding component is occupied, and full TWCRAC overlap remains unresolved. The current title and some wording risk making a standard law sound like the novelty rather than the tested objective/interface.

**Concrete fix:** State the contribution as “a discrete warp-integral equivariance objective for supplied-track MRAC” and make the novelty boundary depend on (i) the exact discrete operator, (ii) the identical augmentation-only deletion, and (iii) separation from the already listed closest mechanisms. Complete the TWCRAC full-method inspection before claim freeze. If it discloses an equivalent local phase transformation objective, apply K8; do not rescue the paper with private state or TSSM. **Priority: CRITICAL.**

**Specific weakness:** Although only one mechanism is claimed, the base system currently contains CycleCL-style contrast, an activity classifier, phase-conditioned velocity reconstruction, variance/covariance regularization, overlap consistency, recurrence, and NOLA. That makes it difficult to know whether the “single-loss” story is genuinely parsimonious or merely placed atop a bespoke stack.

**Concrete fix:** Choose one established base phase learner and keep only the minimum anti-collapse/seam machinery it demonstrably requires in training-only gates. Every retained auxiliary must be identical in treatment/control and have a deletion receipt; remove auxiliary losses that do not enable the count interface. **Priority: IMPORTANT.**

### 4. Frontier Leverage — 8.7/10

No LLM, VLM, diffusion model, RL policy, teacher, or inference-time search is warranted for this bottleneck. Exact transformation metadata is the natural structured supervision. Refusing a fashionable but confounded module stack is a strength, not a modernization deficit. The most current aspect is the clean equivariance/intervention framing and the insistence on matched augmentation exposure.

### 5. Feasibility — 5.4/10

**Specific weakness:** The bounded pilot is possible only after a nontrivial data/protocol build. The current pickle is label-mixed; the cache covers 110/794 train and 51/183 validation videos; validation has only nine source-connected components; 13 train and 8 validation slots are association-ambiguous; duplicate sampled indices can be extensive; no canonical implementation or frozen evaluator exists. The proposal acknowledges most of this but does not freeze the slot eligibility and interpolation population that the model and evaluator will share.

**Concrete fix:** Before model code, freeze a non-executable feature shard schema plus a separate vault and join manifest. Put opaque join keys in loader metadata, never model tensors. Freeze a count-blind eligibility manifest: either (preferably) require one-to-one GT-assisted association and a numeric pose-coverage threshold for the primary population, with all exclusions and a sensitivity report fixed before opening count/density, or include every supplied slot and explicitly accept mapping ambiguity. Define duplicate support as “retain one sample only if all repeated-index pose/mask values agree within tolerance; otherwise invalidate that clock location,” then interpolate only between adjacent distinct valid clocks and never across an invalid cell. **Priority: CRITICAL.**

**Specific weakness:** “Three seeds for every decision-bearing learned comparator” plus all listed closest-prior adaptations is a large implementation/fidelity burden relative to a partial-cache pilot and the stated ICASSP schedule. “Input/capacity/compute matched” is also not executable for methods whose faithful heads and objectives differ.

**Concrete fix:** Keep the existing comparator list but stage it. First run the three paired treatment/control seeds and the core deletion/direct-frequency/global-warp checks. Only if K1/K3 pass, execute the predeclared closest-prior stage. For each comparator, freeze whether it is (a) a same-backbone mechanism control or (b) a faithful supplied-track adaptation; record code provenance, interface mapping, parameter count, optimizer steps, and an equal training-only tuning budget. Do not claim simultaneous capacity and method fidelity where they conflict. State the exact maximum run count and stop-trigger savings. **Priority: IMPORTANT.**

### 6. Validation Focus — 5.8/10

**Specific weakness:** The 5% AND CI endpoint is conceptually correct but not yet executable. It does not specify whether relative improvement is a ratio of seed-averaged metrics or an average of seed-level ratios, how cluster multiplicities interact with video-first averaging, the number/seed of bootstrap draws, how method/seed pairing is preserved, what the CI covers (data components only versus training randomness), or what happens when the control denominator is zero. With only nine development components, these choices can materially change the interval.

**Concrete fix:** Freeze one algorithm. For each seed (s), compute video-first (M_{c,s}) and (M_{t,s}) on identical videos. Define (\bar M_c=\frac13\sum_s M_{c,s}), (\bar M_t=\frac13\sum_s M_{t,s}), absolute improvement (D=\bar M_c-\bar M_t), and relative improvement (D/\bar M_c); if (\bar M_c=0), the relative clause is undefined and K1 fails. For each of at least 10,000 deterministic bootstrap draws, sample the nine frozen source components with replacement, retain every video/person within each selected component with multiplicity, recompute video-first metrics inside each seed, then average the three paired seed effects. Use a two-sided central percentile 95% CI for absolute (D); report seed-level effects and sample SD separately, explicitly noting that the cluster CI does not estimate training-seed uncertainty. Primary success is relative improvement at least 0.05 and CI lower endpoint greater than 0. Add a tiny synthetic scorer fixture with a hand-computable expected result. **Priority: CRITICAL.**

**Specific weakness:** The proposal calls the validation minimal but includes a long flat list of comparators, shortcut probes, corruptions, state variants, and three claims. In a 4+1-page paper this risks turning the main mechanism test into an audit appendix.

**Concrete fix:** Preserve three visible blocks only: primary paired law-versus-augmentation endpoint; staged mechanism/closest-prior falsification; one identity-isolation diagnostic. Move integrity tests and secondary probes to receipts/supplement, and do not run the optional masked-TSSM control unless the predeclared comparison matrix needs it. **Priority: IMPORTANT.**

### 7. Venue Readiness — 5.8/10

**Specific weakness:** The signal-processing framing is suitable for ICASSP, but a partial GT-assisted cache with zero eligible results cannot yet support a paper-level performance contribution. The unresolved phase unit and novelty collision are especially damaging because they concern the central claim, not polish. The current method description is also far too large for four main pages.

**Concrete fix:** Treat this document as a preregistered pilot, not a manuscript method freeze. Reach venue readiness only after the discrete operator and phase unit are closed, the label-vault/evaluator are implemented, the primary pair is executable, TWCRAC is adjudicated, and the successful story can be compressed to one equation, one architecture figure, one primary table, one closest-prior/ablation table, and one isolation figure. A failed pilot should terminate the contribution rather than expand it. **Priority: CRITICAL.**

## Simplification Opportunities

1. Replace the current bespoke bundle of base contrast/activity/reconstruction/variance losses with one frozen established phase-learning backbone plus only the auxiliary needed to pass a training-only identifiability gate.
2. Convert the flat comparator plan into a preregistered staged ladder; this preserves every anchored comparison while avoiding full implementation/run cost after an early K1/K3 failure.
3. Reduce the identity diagnostic to the three highest-information conditions (private, shared-scene, shuffled-key) in the main study; keep reset/swap/stateless variants as debugging receipts unless contamination appears. Keep masked TSSM optional and control-only.

## Modernization Opportunities

**NONE.** A foundation model would add confounded capacity and supervision without clarifying the derivative-law hypothesis. The needed modernization is mathematical and experimental precision, not a new model class.

## Drift Warning

**NONE.** The proposal still solves the immutable supplied-track, no-count/period-label, within-track tempo-drift problem. The fixes above preserve that anchor. They must not be used to introduce predicted-track, end-to-end, scene-total, or generic phase-representation claims.

## Required revision order

1. **CRITICAL:** Choose and formalize the unique oriented discrete warp-integral loss, including duplicate-clock, invalid-support, alias, reversal, pause, wrap, and stop-gradient semantics.
2. **CRITICAL:** Make one physical repetition identifiable as one phase winding without evaluator labels, or kill the phase-mass decoder.
3. **CRITICAL:** Freeze boundary-safe interval NOLA and all numerical interfaces.
4. **CRITICAL:** Freeze the partial-cache feature/vault split and ambiguous-slot/coverage population before evaluator access.
5. **CRITICAL:** Freeze and unit-test the exact 5%+paired-component-CI computation.
6. **IMPORTANT:** Select one established base learner, stage the existing comparator set, and bind every adaptation to a finite tuning/run budget.
7. **CRITICAL before claim freeze:** Resolve full-method TWCRAC overlap.

## Verdict

**REVISE**

The direction is promising and correctly focused, but READY is unavailable at 6.9/10. The proposal has no drift and no forced frontier component; its blockers are the executability and identifiability of the core mechanism, the partial-cache protocol, and the primary inference rule. No numerical efficacy statement is licensed.

## Audited input SHA-256

| Input | SHA-256 |
|---|---|
| `refine-logs/round-0-initial-proposal.md` | `b6de5c2fcc5f6999e774c630a501a8136850fb7d61adae974b11561c9bce6d51` |
| `refine-logs/REFINE_STATE.json` | `0b56ccc08c067f24c23f5a478e2fc878928bc4c4733bcf6cbffff763cf38cf99` |
| `idea-stage/NOVELTY_CHECK.json` | `79749893547c834d876801c287bfa2f048b4797c838f7f6d46155318258ed3e5` |
| `idea-stage/RESEARCH_REVIEW.json` | `fbc2462303c3ddcb4b23ef4cbf4f81b01a863ff433c6a58dec3f8cd99790caa3` |
| `PILOT_DATA_SCHEMA_AUDIT.md` | `7058981e8eb37622c09f5808a4b1329bc7cd133f474101723f15e5e4273eab25` |
| `EXPERIMENT_AUDIT.md` | `2ac7fe4a1a8208287394a2ac4c1771aa0faf30e14319bef8144e509f876f0820` |
| `SERVER_METHOD_INVENTORY.md` | `0cda0704f64434296049d18c35e97fd267a0a7168300152e97b142d6ecc363a6` |

</details>

<details>
<summary>Round 2 Review — full verbatim stored response</summary>

# Round 2 Research-Refine Re-evaluation: WARP-PHASE

**Reviewer:** GPT-5.6-Sol (`/root/research_refine_reviewer`)  
**Review continuity:** same reviewer thread as Round 1  
**Review independence:** same-family  
**Acceptance status:** provisional  
**Evidence status:** proposal-only; zero eligible method results  
**Venue target:** ICASSP 2027 (4+1 pages)

## Executive decision

Round 1 produced a substantial and directionally correct refinement. The proposal now has one explicit training intervention, one established shared phase base, a single oriented overlap-integral target, strictly positive interval weights, a numerical table, count-blind eligibility rules, a deterministic bootstrap specification, a staged comparator plan, and three visible validation blocks. It is much simpler and more executable than Round 0, and it remains appropriately frontier-aware without adding a foundation model.

It is not READY. Four scientific/execution blockers remain. First, the equations apply the clean-derived orientation sign to the clean integral but not explicitly to the warped increment, so the sole treatment residual is sign-inconsistent when the clean median increment is negative. Second, the ACF/FFT rule defines a stable pseudo-cycle but does not establish that it is one physical repetition; its real-track gates test self-consistency, not semantic winding, and no frozen evaluator-only harmonic rejection gate remains. Third, the exact primary scorer labeled AvgMAE omits division by ground-truth count and therefore computes raw MAE rather than the audited published normalized AvgMAE. Fourth, “exactly nine components” and the 53-job/480-GPU-hour absolute cap are not yet justified after the new eligibility exclusions and decision-bearing shortcut models. Full TWCRAC inspection also remains open, so novelty cannot be frozen.

## Original-anchor verification

**PASS.** Both Problem Anchor copies in `round-1-refinement.md` are byte-for-text identical to each other, and every field in `idea-stage/RESEARCH_REVIEW.json` → `exact_problem_anchor` matches: research question, population, unit of analysis, primary estimand, all three secondary estimands, and all three excluded questions. No scene-total, predicted-track, end-to-end, generic representation-learning, or historical-result drift was introduced.

## Round-1 blocker closure audit

| Round-1 blocker | Status | Round-2 finding |
|---|---|---|
| Unique oriented integral | **PARTIAL** | The overlap-length integral, half-open cells, invalid-support mask, alias rule, pause, reversal, float precision, wrap, and stop-gradient are now unique. However, Section 4.1 orients clean increments while the loss in Section 4.2 uses un-oriented `delta_tau`; the same sign must act on both sides or on neither side. |
| Winding/fundamental identifiability | **PARTIAL / BLOCKING** | The selector is executable and much stronger, but nested-view stability and margins cannot prove that a selected subcycle is the semantic action repetition. Declaring one selected `P` to be one physical cycle does not make it so. |
| Strictly positive interval NOLA | **CLOSED** | Interval-centered sine-squared weights with a `1e-3` floor, deterministic final window, no epsilon rescue, denominator assertion, and grid-origin fixture close the boundary-mass issue. |
| Frozen numeric table | **PARTIAL** | Most constants, shapes, losses, seeds, schedules, and parameter counts are frozen. The GRU/window execution path is still ambiguous: persistent identity state across overlapping windows would update repeated samples multiple times, while resetting per window would not implement the claimed identity-private recurrent flow. |
| Ambiguous-slot / duplicate-clock population | **PARTIAL** | Count-blind exclusion, coverage thresholds, and duplicate disagreement are explicit. But the nine components were counted before these new exclusions; the proposal has not established that every component remains nonempty, so the bootstrap population cardinality is asserted rather than derived. |
| Exact `B=10,000` bootstrap | **PARTIAL / BLOCKING** | Pairing, PCG64 seed, multiplicity, seed aggregation, quantiles, CI target, and zero denominator are excellent. The base per-video error formula is wrong for AvgMAE because it omits `/ C_{v,p}`. |
| 53-job / 480-GPU-hour staged cap | **PARTIAL** | The arithmetic for the table is internally correct: 36 full + 14 tuning + 3 sanity jobs and 432 + 42 + 6 GPU-hours. The cap excludes trained no-pose/timestamp-only and nuisance-only shortcut models invoked by K4/Block 2, so it is not an absolute cap for the declared decision plan. |
| One base learner | **CLOSED, with positioning caution** | Treatment and control now share only the frozen PAMS-TCC-style phase loss and the same non-trainable selector. The custom ACF/FFT score is an operational pseudo-label generator, not a second claimed contribution. |
| Three visible validation blocks | **CLOSED** | Primary pair, staged falsification, and identity isolation are the only visible blocks; integrity work is correctly moved to receipts. |
| TWCRAC K8 | **OPEN / CLAIM-BLOCKING** | K8 is correctly retained, but abstract-only inspection remains insufficient. READY is unavailable while the closest unresolved full method has not been adjudicated. |

## Scores

| Dimension | Weight | Score / 10 | Weighted contribution |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9.6 | 1.440 |
| Method Specificity | 25% | 7.3 | 1.825 |
| Contribution Quality | 25% | 7.6 | 1.900 |
| Frontier Leverage | 15% | 9.0 | 1.350 |
| Feasibility | 10% | 6.8 | 0.680 |
| Validation Focus | 5% | 6.6 | 0.330 |
| Venue Readiness | 5% | 6.5 | 0.325 |

**WEIGHTED COMPOSITE: 7.85 / 10 (reported: 7.9 / 10)**  
**CALIBRATION: none**

**GAP:** No curated three-good/three-bad proposal anchors were supplied, so calibration remains `none` and no exemplar comparison is invented. Relative to the rubric’s READY bar, this proposal now resembles a strong preregistration with an unusually clean deletion contrast, but not yet a closed method plan: the core integral is nearly executable, the decoder boundary issue is solved, and validation is visibly focused, while the phase unit is only operationally self-consistent, the signed residual has an orientation mismatch, the primary metric is not the named AvgMAE, the post-eligibility cluster count and shortcut-job budget are unresolved, and the closest novelty collision remains unread. These are narrow, fixable gaps, but each can change the scientific conclusion and therefore prevents a score of 9.

## Dimension review

### 1. Problem Fidelity — 9.6/10

The anchor is preserved exactly, not merely paraphrased. The revised proposal stays on per-person supplied tracks, within-track tempo drift, no per-person count/period supervision in the counting objective, video-first evaluation, and the identical augmentation-only contrast. The use of GT-assisted association and coverage for a pre-label eligibility manifest is disclosed rather than relabeled annotation-free. Exclusions must remain count-blind and fixed before the vault opens to preserve this score.

### 2. Method Specificity — 7.3/10

The unique integral is a major improvement: it is now an oriented signed overlap sum over duplicate-collapsed valid cells, with one target, one residual, one gradient direction, and fail-closed support. The remaining sign convention must be repaired. Define

\[
o=\operatorname{sg}\!\left(\operatorname{sign}\operatorname{median}_k\delta_k\right),\quad
\bar\delta_k=o\delta_k,\quad
\bar\delta_j^\tau=o\delta_j^\tau,
\]

build (I_j^	au) from (ar\delta_k), and use `wrap(bar_delta_tau - sg(I_tau))`; alternatively delete canonical orientation from both sides because the law and absolute-mass decoder are globally sign-equivariant. The current equation orients only the teacher. **Priority: CRITICAL.**

The phase selector now defines a reproducible pseudo-period, but “stable across views/windows” is not equivalent to “one physical action repetition.” Symmetric actions can produce a stable half-cycle with a large spectral margin. Keep the selector frozen and label-free, but call the unit a selector-defined pseudo-cycle until evaluation. Add a predeclared evaluator-only harmonic adjudication after predictions are frozen: compare \hat P with evaluator period at `0.5x/1x/2x`, report the video-first ambiguous and half/double fractions, permit no correction or retuning, and invoke K5 if a frozen tolerance is exceeded. The synthetic fixtures should include symmetric two-limb/two-pose cycles specifically constructed so the strongest harmonic is not the semantic fundamental. **Priority: CRITICAL.**

The recurrent execution path also needs one sentence-level algorithm. Prefer: run the convolutions and GRU once over each complete collapsed identity sequence in chronological order, reset state only at identity start, hold state across invalid cells, cache each hidden state once, and let overlapping windows read cached states without updating the GRU. If windows instead own recurrent execution, specify reset/carry semantics and prove no physical clock is committed twice. **Priority: IMPORTANT.**

### 3. Contribution Quality — 7.6/10

The contribution is now appropriately stated as a discrete warp-integral equivariance objective, not as a novel chain rule. Private state, NOLA, ACF/FFT, PAMS-TCC, and TSSM are correctly non-contributions. Treatment versus control differs by one loss, which is a sharp ICASSP-style mechanism test.

Two ceilings remain. The selector is a custom operational combination even if its parts are established, so it must stay frozen/shared and cannot become a second novelty story if it performs well. More importantly, full TWCRAC overlap is unresolved. The proposal correctly defines K8, but contribution quality cannot reach READY until that check is actually completed and bound to a primary-source receipt.

### 4. Frontier Leverage — 9.0/10

Frontier leverage is appropriate. Known transformation metadata, exact equivariance supervision, causal intervention, fail-closed protocol receipts, and matched deletion are more natural than adding an LLM/VLM/diffusion/RL component. No modernization module is needed.

### 5. Feasibility — 6.8/10

**Specific weakness:** The eligibility rules can change the development component set, yet the estimator assumes exactly nine components. Excluding eight ambiguous development slots plus coverage/validity failures may empty a video or component. The proposal cannot know the final component count before Gate 0 runs.

**Concrete fix:** Make “all nine audited components retain at least one eligible video” a count-blind Gate-0 assertion. If it fails, invoke K7 before evaluator access; do not silently switch from nine to a metric-informed subset. Freeze and hash the resulting eligible video/person/component table used by every arm. **Priority: CRITICAL.**

**Specific weakness:** The 53-job ledger does not include the trained K4 clock/nuisance shortcut models. An inference-time zero-pose ablation is not equivalent to training a shortcut model to exploit clocks or nuisance metadata.

**Concrete fix:** Either add three seeds each for timestamp/no-pose and nuisance-only models and revise the maximum to at least 59 jobs / 552 GPU-hours under the current per-full-job cap, or remove/reallocate optional/tuning jobs and show a new complete ledger that truly remains at 53/480. Bind every K4 model to a stage and stop condition. **Priority: IMPORTANT.**

### 6. Validation Focus — 6.6/10

**Specific weakness:** Section 8 defines

\[
A_{a,s,v}=|P_v|^{-1}\sum_p|\hat C-C|,
\]

which is raw per-person MAE averaged video-first. The audited MultiRep AvgMAE uses per-person relative absolute error with ground truth in the denominator. Thus the frozen endpoint currently tests a different estimand from the Problem Anchor and prior audit.

**Concrete fix:** Replace the inner term with \(|\hat C_{a,s,v,p}-C_{v,p}|/C_{v,p}\), retain the person-first then video-first averaging, and explicitly fail or define policy for zero ground-truth counts. The audited current subset has positive counts, but the scorer should assert this rather than assume it. Recompute the hand-worked fixture using the normalized formula. The remaining `B=10,000` cluster-bootstrap algorithm can stay unchanged. **Priority: CRITICAL.**

**Specific weakness:** The three-block presentation is focused, but K4 shortcut training is hidden in “receipts” while it can kill the claim. Decision-bearing controls must appear in the run ledger even if their detailed plots live in the supplement.

**Concrete fix:** Keep three visible blocks, but list every decision-bearing job and threshold in the stage table. Presentation compression must not remove execution accounting. **Priority: IMPORTANT.**

### 7. Venue Readiness — 6.5/10

**Specific weakness:** The proposal is now plausible as an ICASSP pilot, but the primary metric is presently misimplemented, semantic phase winding is not adjudicated, and the closest full prior remains unresolved. Those are central validity/novelty issues, not manuscript polish.

**Concrete fix:** Correct the scorer, close sign/state semantics, add a no-correction evaluator harmonic gate, prove the nine-component eligible manifest and complete run ledger, and finish TWCRAC full-method adjudication. Once those are closed, the method can be represented compactly by the overlap-integral equation, one shared-base diagram, one primary/staged table, and one isolation figure. **Priority: CRITICAL.**

## Focus, simplicity, and frontier assessment

- **Anchor:** preserved verbatim.
- **Dominant contribution:** substantially sharper; one discrete warp-integral objective remains the only proposed mechanism.
- **Simplicity:** materially improved. The bespoke five-loss stack is gone, NOLA is deterministic, and validation has three visible blocks. The remaining ACF/FFT selector should remain a frozen shared pseudo-label interface, not grow into a contribution.
- **Frontier appropriateness:** appropriate and unforced. No foundation-model component should be added.
- **Method status:** nearly executable but not closed because the signed orientation, recurrent update path, and physical-cycle adjudication remain incomplete.

## Simplification Opportunities

1. Drop canonical orientation entirely from the training loss if it is not needed for diagnostics; the signed equivariance law is globally sign-symmetric and the count decoder uses absolute mass. This removes the current asymmetric-sign failure mode.
2. Keep the ACF/FFT selector frozen outside the learned network and describe it as pseudo-label generation in one paragraph; do not expand it into another contribution or comparator family.
3. Keep all integrity, harmonic, and shortcut details in receipts/supplement while retaining their jobs and kill thresholds in the single staged ledger.

## Modernization Opportunities

**NONE.** The proposal is already appropriately current for a signal-processing hypothesis.

## Remaining action items

1. **CRITICAL:** Apply the same orientation sign to clean and warped increments, or delete orientation from both sides.
2. **CRITICAL:** Treat the selected period as a pseudo-cycle and add a frozen evaluator-only `0.5x/1x/2x` harmonic rejection gate with no correction or retuning.
3. **IMPORTANT:** Specify whether the GRU is executed once per physical clock or how overlapping-window state avoids duplicate updates.
4. **CRITICAL:** Correct video-first AvgMAE to use per-person ground-truth-normalized absolute error and update the fixture.
5. **CRITICAL:** Prove at Gate 0 that all nine source components remain nonempty after count-blind eligibility, or trigger K7.
6. **IMPORTANT:** Include trained no-pose and nuisance-only K4 controls in the hard job/GPU ledger; revise or rebalance the cap.
7. **CRITICAL before claim freeze:** Complete primary-source full-method TWCRAC adjudication and apply K8 if equivalent.

## Drift Warning

**NONE.** The revised proposal remains faithful to the immutable problem. The remaining fixes do not require predicted tracks, scene totals, evaluator-label training, extra model families, or a new contribution.

## Verdict

**REVISE**

The proposal improved from 6.9 to 7.9 and should continue on the same route. It is focused, simple enough in architecture, and appropriately modern, but READY requires all blockers closed. The incorrect AvgMAE equation, unresolved semantic winding, sign/state ambiguity, population/budget accounting gaps, and open TWCRAC K8 preclude READY. No result or expected effect has been invented.

## Audited input SHA-256

| Input | SHA-256 |
|---|---|
| `refine-logs/round-1-refinement.md` | `6840a8e8cec3d8fa6d7c371e14a37a3ad3fdd1be120902bf67f911a247a47b75` |
| `refine-logs/REFINE_STATE.json` | `caae0f416701b97f71a3ae6d46c9955df8ca92cecdfdb7149491737bc0f0c148` |
| `refine-logs/score-history.md` | `529e9f23764abe119346d8b1d72bb5e2cd19a160903fd4c80b05bf3db9eb2b12` |
| `refine-logs/round-1-review.md` | `e20dd1d041eaaefa286670ed6910c7d86b47bf73da30cd97158d5535fbc5b883` |

</details>

<details>
<summary>Round 3 Review — full verbatim stored response</summary>

# Round 3 Research-Refine Re-evaluation: WARP-PHASE

**Reviewer:** GPT-5.6-Sol (`/root/research_refine_reviewer`)  
**Review continuity:** same reviewer thread as Rounds 1–2  
**Review independence:** same-family  
**Acceptance status:** provisional  
**Evidence status:** proposal-only; zero eligible method results  
**Venue target:** ICASSP 2027 (4+1 pages)

## Executive decision

Round 2 closes almost every named Round-2 repair. Canonical orientation is gone from both branches; the method uses raw signed increments and a unique oriented overlap integral. The ACF/FFT output is now truthfully a selector-defined pseudo-cycle, followed by a frozen evaluator-only `0.5x/1x/2x` rejection gate that cannot correct predictions. Recurrence has single-commit semantics, AvgMAE is ground-truth normalized and verified by a correct hand-worked fixture, Gate 0 requires all nine audited source components to remain nonempty, the staged ledger includes trained K4 controls and sums to 59 jobs/552 GPU-hours, and the paper still has only three visible validation blocks.

The proposal is not READY. The closest-method novelty blocker, TWCRAC K8, is explicitly open and no separate primary-source full-method adjudication receipt was supplied. Two execution interfaces also need one final freeze: the stated optimizer still batches 64 windows even though the GRU must run once over a complete identity sequence, which is not a valid reusable-autograd/cache algorithm as written; and the evaluator harmonic gate assumes one scalar annotated `P_eval,p` without binding how the actual period annotation is converted into that scalar in source-clock units. These are local specification gaps rather than a reason to rethink the route.

## Anchor verification

**PASS.** The two Problem Anchor copies in `round-2-refinement.md` are byte-for-text identical, and every exact anchor field matches the frozen `idea-stage/RESEARCH_REVIEW.json` source. The research question, population, video-first unit, primary and secondary estimands, and excluded questions are unchanged. There is no scene-total, predicted-track, end-to-end, annotation-free, generic phase-learning, or historical-result drift.

## Prior-action closure audit

| Round-2 action | Status | Finding |
|---|---|---|
| Remove orientation symmetrically | **CLOSED** | Sections 4.1–4.2 use raw `delta_k` and raw `delta_tau`; reversal sign comes only from signed overlap length. No median-sign branch remains. |
| Pseudo-cycle language | **CLOSED** | The selector output is consistently called an internal pseudo-cycle and is not asserted to be an annotated physical repetition. |
| Frozen evaluator-only `0.5x/1x/2x` gate | **PARTIAL** | Freeze order, 10% classes, video-first fractions, 90%/5% thresholds, no correction, and K5 consequence are exact. The source and aggregation rule for scalar `P_eval,p` is not bound to the actual evaluator schema. |
| GRU once per collapsed identity | **PARTIAL** | State semantics are now correct: one chronological commit, invalid-state hold, cached states, read-only overlap. The optimization table still says `batch 64 windows`; it does not say how a full-sequence graph is computed and backpropagated once per optimizer step without stale/detached caches or repeated backward through one graph. |
| Normalized video-first AvgMAE and `C>0` | **CLOSED** | The formula divides each person error by `C`, then averages people and videos. The positive-count assertion and fixture arithmetic are correct. |
| All-nine-component Gate-0 assertion | **CLOSED AS A PRECONDITION** | The proposal no longer assumes exclusions preserve all components silently; an empty component invokes K7 before label access. Actual passage remains prospective, as it should. |
| 59-job/552-GPU-hour ledger | **CLOSED** | 42 full + 14 tuning + 3 sanity jobs = 59; `42x12 + 14x3 + 6 = 552` GPU-hours. Three-seed trained clock/no-pose and nuisance-only K4 controls are included. |
| Three visible validation blocks | **CLOSED** | K4 is accounted inside Block 2 rather than hidden or promoted to a fourth claim. |
| TWCRAC K8 | **OPEN / BLOCKING** | The proposal states the truth: no independent full-method receipt exists. Under the READY rule, an explicitly open closest-prior collision is still a blocker. |

## Scores

| Dimension | Weight | Score / 10 | Weighted contribution |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9.8 | 1.470 |
| Method Specificity | 25% | 8.4 | 2.100 |
| Contribution Quality | 25% | 8.0 | 2.000 |
| Frontier Leverage | 15% | 9.2 | 1.380 |
| Feasibility | 10% | 7.8 | 0.780 |
| Validation Focus | 5% | 8.6 | 0.430 |
| Venue Readiness | 5% | 7.4 | 0.370 |

**WEIGHTED COMPOSITE: 8.53 / 10 (reported: 8.5 / 10)**  
**CALIBRATION: none**

**GAP:** No curated three-good/three-bad proposal anchors were provided, so calibration remains `none` and no exemplar comparison is fabricated. Relative to the rubric’s READY bar, the proposal is now a strong, focused preregistration: its anchor, unique treatment deletion, discrete math, boundary-safe decode, count-blind population gate, normalized estimator, hard compute ledger, and kill logic are unusually concrete. It remains below 9 because the full-sequence recurrence has no coherent optimizer batching algorithm, the evaluator harmonic gate is not yet bound to the real period-label schema, and the closest unresolved method has not been inspected in full. The first two are interface fixes; K8 is an external claim-blocking adjudication. No new model or experiment family is needed.

## Dimension review

### 1. Problem Fidelity — 9.8/10

The revised method remains exactly on the anchored question. It tests local signed warp-integral supervision against identical augmentation exposure, uses supplied per-person tracks, keeps count/period labels outside training and tuning, and applies normalized video-first AvgMAE with source-component resampling. The pseudo-cycle/evaluator separation is particularly important: it avoids redefining the research problem as mere spectral self-consistency.

### 2. Method Specificity — 8.4/10

The core method is now nearly executable. Raw clean increments define a piecewise-constant signed measure; the mapped target is one float64 signed-overlap integral; invalid support and aliasing fail closed; the clean target is stop-gradient; pause and reversal are unambiguous; and treatment differs from control only by `L_WI`. Positive interval NOLA and one decode are also fully specified.

The remaining training-loop contradiction is material. A cache of full-sequence hidden states cannot be reused across 20,000 optimizer updates: after a weight update it is stale, and retaining one autograd graph for multiple 64-window mini-batches either detaches gradients or requires repeated backward through the same graph. Freeze one valid algorithm. The simplest is to batch complete collapsed identity sequences (padded with masks), run clean and warped Conv/GRU passes once per track per optimizer step, construct all eligible 64-window losses as read-only views inside that same forward graph, average first within identity and then across identities, perform one backward/step, and discard the cache. Replace “batch 64 windows” with a frozen number of full tracks or a token/clock budget. If truncated BPTT is intended instead, it needs non-overlapping commit chunks and explicit detach boundaries; overlapping windows must remain loss views only. **Priority: CRITICAL.**

The evaluator gate also needs a single schema-bound definition of `P_eval,p`. Inspect the actual vault field and freeze one path: for example, if period is per-cycle/per-frame, define the permitted valid entries, conversion to source-frame-clock units, and the exact track aggregation (such as a linear median) before classification. Do not retain alternative branches based on what gives better agreement. Add a tiny schema fixture demonstrating `P_eval,p`, `hat P_p`, class assignment, and video-first `H_r`. This is evaluator engineering, not another experiment. **Priority: IMPORTANT.**

### 3. Contribution Quality — 8.0/10

The story is now disciplined: one discrete warp-integral objective, one identical deletion, one supporting isolation diagnostic, and explicit non-contributions. The pseudo-cycle selector is shared infrastructure and cannot become a second paper claim. The ordinary chain rule is motivation rather than asserted novelty.

Contribution quality is capped by K8. The accessible TWCRAC record already overlaps local windows, cycle consistency, pose, and non-stationary motion; only a complete primary-source method inspection can establish whether the discrete local warp-phase objective is actually residual. An equivalent disclosure must kill or re-anchor the claim, exactly as the proposal states.

### 4. Frontier Leverage — 9.2/10

The proposal uses current ideas where natural—structured transformation supervision, exact equivariance, causal interventions, deterministic provenance, and matched deletion—without forcing a foundation model. An LLM, VLM, diffusion model, RL policy, router, or expert bank would weaken causal attribution and should remain excluded.

### 5. Feasibility — 7.8/10

The run ledger is finite and internally correct, label-vault and component failures stop before evaluation, and every stage has bounded savings. Feasibility remains below 9 because the implementation does not exist, the all-nine eligibility assertion and severe selector gates may legitimately kill the pilot, and the current window-batch/full-sequence recurrence combination cannot yet be implemented as written. These are acknowledged prospective risks, not results.

The K4 model schemas should also be frozen in the eventual implementation table: exact clock/mask/warp fields, nuisance fields, zeroed pose representation, and parameter-count handling. This does not change the 59-job ledger or add a comparator; it prevents an underpowered shortcut control. **Priority: IMPORTANT.**

### 6. Validation Focus — 8.6/10

The primary estimator and bootstrap are now executable: positive normalized person error, person-first/video-first aggregation, three-seed point estimate, deterministic 10,000-draw paired component bootstrap, multiplicity, percentile rule, and separation of cluster uncertainty from seed SD are all frozen. The scorer fixture is numerically correct. The three-block layout remains focused even with complete kill-control accounting.

One wording fix is needed in Stage D: it cannot require A–C to “survive K2,” because K2 is decided by the Stage-D closest priors themselves. The condition should be that A–C survive K1/K3/K4, after which D adjudicates K2. This is a control-flow correction, not an added run. **Priority: MINOR.**

### 7. Venue Readiness — 7.4/10

The proposal is methodologically close to an ICASSP-ready execution plan, but not a novelty-frozen venue story. The full-sequence training loop and evaluator period binding must be closed, and K8 must be resolved from the complete primary source. The partial GT-assisted cache also remains pilot evidence only; even a positive outcome must retain the claim ceiling.

## Focus, simplicity, and frontier assessment

- **Anchor:** preserved verbatim.
- **Dominant contribution:** sharp and singular—the discrete warp-integral loss.
- **Simplicity:** strong. Removing orientation was cleaner than adding symmetric gauge machinery; recurrence commits once; NOLA is fixed; the base has one loss; validation has three blocks.
- **Frontier appropriateness:** strong and unforced; no modernization component is missing.
- **Implementation readiness:** close, pending a valid full-track optimizer loop and schema-bound evaluator period extraction.
- **Claim readiness:** blocked by open TWCRAC K8.

## Simplification Opportunities

1. Preserve the current raw-sign formulation; do not reintroduce phase orientation or a learned harmonic corrector.
2. Treat the per-step full-sequence hidden states as ephemeral autograd values, not a cross-step cache. This makes “single commit” compatible with ordinary optimization.
3. Keep the evaluator harmonic gate and K4 schemas in receipts/tables; they remain decision gates inside the existing three blocks, not new contributions.

## Modernization Opportunities

**NONE.** The method is already appropriately frontier-aware for the stated signal-processing bottleneck.

## Remaining action items

1. **CRITICAL:** Replace `batch 64 windows` with one exact full-sequence training/batching/backpropagation algorithm consistent with single-commit recurrence and fresh parameters each step.
2. **IMPORTANT:** Bind scalar `P_eval,p` to the actual evaluator period schema, units, valid-entry rule, aggregation, and a deterministic fixture.
3. **IMPORTANT:** Freeze exact K4 shortcut-model input schemas and parameter handling within the already budgeted six jobs.
4. **MINOR:** Correct the Stage-D condition so that Stage D adjudicates K2 rather than purportedly following a prior K2 decision.
5. **CRITICAL for READY/claim freeze:** Complete an independent primary-source full-method TWCRAC adjudication; apply K8 if equivalent.

## Drift Warning

**NONE.** Every requested fix stays inside the original supplied-track, count/period-label-restricted, within-track tempo-drift question. No baseline expansion, predicted-track claim, or auxiliary contribution is required.

## Verdict

**REVISE**

The score rises from 7.9 to 8.5. The route is now focused, technically serious, and close to execution, so RETHINK is unwarranted. READY is prohibited because the full-sequence optimization interface and evaluator period binding remain incomplete and K8 is truthfully open. No efficacy result, gate passage, or novelty clearance is inferred.

## Audited input SHA-256

| Input | SHA-256 |
|---|---|
| `refine-logs/round-2-refinement.md` | `2694a4eaa518bf216f21ffa77fddd54141267eaa1dd5520073583bbae995953e` |
| `refine-logs/REFINE_STATE.json` | `998f42bbf6029f4fd21150f811af45bc3707a5da103ab5ed7a5248a18a68de3a` |
| `refine-logs/score-history.md` | `1498d08771dcbac713dba4694c3eea600ae8a35420b473ed30d1c8b4d311576e` |
| `refine-logs/round-2-review.md` | `8aaf103283285c3c7a80ca96640350bdeca0d6d98f056b500648adeb55c26664` |

</details>

<details>
<summary>Round 4 Review — full verbatim stored response</summary>

# Round 4 Research-Refine Re-evaluation: WARP-PHASE

**Reviewer:** GPT-5.6-Sol (/root/research_refine_reviewer)  
**Review continuity:** same reviewer thread as Rounds 1-3  
**Review independence:** same-family  
**Acceptance status:** provisional  
**Evidence status:** proposal-only; zero eligible method results  
**Venue target:** ICASSP 2027 (4+1 pages)

## Executive decision

Round 3 genuinely closes the full-identity optimizer, evaluator-period binding, and Stage-D control-flow repairs. It also records TWCRAC truthfully as BLOCKED with k8.decision=NOT_PASSED_BLOCKED and novelty_clearance=false. The proposed optimizer now uses eight complete identities per fresh graph, one backward per step, four-step gradient accumulation, and 20,000 updates, so 8 x 4 x 20,000 = 640,000 track draws is coherent. The raw evaluator period is uniquely the float64 NumPy median of audited integer source-frame half-open spans. Stage D runs only after A-C survive K1/K3/K4 and then adjudicates K2.

The implementation audit does, however, expose real local specification gaps that the complete replacement cannot delegate silently to code. Root/scale normalization has no missing-landmark rule; the 64/128-retained-clock selector does not define how a source span larger than 256 enters a 256-point FFT; the FFT evaluation and endpoint-local-maximum conventions are incomplete; nested views do not resolve to one training selection and decoder confidence g_j is undefined; the 1,000-fixture generator has no exact mixture or frozen bytes; warped pose/confidence/mask interpolation is not numerically specified; and the K4 fixed-code exposure/vector construction is only partial. Independently, K4 still uses the semantic predicates “retains,” “preserves,” and “erases” without freezing all decision formulas and thresholds. These are bounded interface freezes, not reasons to add experiments or rethink the route.

The proposal is not READY. The local specification is not yet executable byte-for-byte, and the external TWCRAC primary-source gate is independently blocked. Abstract-only evidence cannot close K8.

**LOCAL_SPEC_STATUS: MULTIPLE_MINIMAL_INTERFACE_FREEZES_REQUIRED**  
**EXTERNAL_K8_STATUS: BLOCKED / NOT_PASSED_BLOCKED**

## Anchor verification

**PASS.** Both Problem Anchor copies in refine-logs/round-3-refinement.md are byte-for-text identical to each other and to the frozen anchor in refine-logs/round-0-initial-proposal.md. The anchor-text SHA-256 is 619148c765175167da093d9b054bd2b43fae0e5e97d981fb9748ef57f4d7dba7. The research question, partial supplied-pose population, video-first unit, primary and secondary estimands, and excluded questions are unchanged. There is no predicted-track, scene-total, end-to-end, annotation-free, generic phase-learning, or historical-result drift.

## Round-3 action closure audit

| Requested Round-3 fix | Status | Finding |
|---|---|---|
| Fresh full-identity optimizer | **CLOSED** | Each step contains eight complete collapsed identities; clean and warped Conv/GRU graphs are fresh and complete; windows are read-only views; loss is averaged within identity and then across identities; one backward occurs per step; four steps accumulate before clipping/update. No cross-step state or graph survives. |
| Schema-bound scalar evaluator period | **CLOSED** | The addendum verifies nonempty integer source-frame [s,e) pairs. The proposal freezes P_eval as np.median(float64(e-s)), global validity assertions, exact source/catalog hashes, and a deterministic harmonic-class fixture. No alternate label or grid-inversion path remains. |
| Exact K4 model inputs | **PARTIAL** | Zeroed pose, retained masks/fractions, q/L clock exposure, opaque-key prohibition, metadata exclusion, parameter count, seeds, and budget are stated. The two mask-fraction sign vectors are not reproducible from “seed 20270815” without an RNG/construction or literal bytes, and the text never states unambiguously whether the primary treatment/control receive the same fixed q/L code or no such code. |
| Stage-D control flow | **CLOSED** | A-C first survive K1/K3/K4; Stage D then runs the closest priors and adjudicates K2. |
| Truthful TWCRAC disposition | **CLOSED AS A RECEIPT; EXTERNALLY BLOCKING** | The cited receipt hash matches. Full-text coverage is zero pages, verdict is BLOCKED, K8 is NOT_PASSED_BLOCKED, claim freeze is BLOCKED, and novelty clearance is false. The proposal makes no distinctness or absence inference. This is not K8 closure. |

## Supplemental seven-interface implementation audit

| Read-only audit concern | Status from the Round-3 text | Minimal in-scope resolution |
|---|---|---|
| Missing-joint hip/shoulder root and scale | **OPEN / BLOCKING** | “Hip midpoint” and “median valid shoulder-to-hip scale” do not say what happens when one/both hips or shoulders are masked. Name the joint indices and freeze a deterministic mask-aware rule. The simplest fail-closed rule is: mean all valid hips for the root, mean all valid shoulders for the shoulder point, take positive finite root-to-shoulder distances, use their track median with the stated 1e-3 floor, invalidate a frame lacking a root, and invoke K7 if no scale support exists. Do not fall back to forbidden boxes or evaluator data. |
| 64/128 retained-clock window versus a source span greater than 256 | **OPEN / BLOCKING** | Retained-clock count is not integer-grid length. State exactly how q_first:q_last is placed into the 256-point masked FFT: padding, deterministic resampling, truncation, or fail-closed rejection. Also state the resulting period units. No new run is required. |
| FFT bins, normalization, and local-maximum endpoints | **OPEN / BLOCKING** | E(1/P) is generally off-bin, while masking/detrending and endpoint maxima are unspecified. Freeze the FFT convention, mask normalization, mean removal if any, bin evaluation/interpolation, zero/Nyquist handling, and whether P=4 and P=P_max use one-sided neighbor tests. A deterministic fixture must cover an off-bin period and both endpoints. |
| Nested views to one selector output and decoder g_j | **OPEN / BLOCKING** | Agreement thresholds do not say which view/window supplies the training pseudo-cycle, and g_j is only called fixed/untrained. Freeze one canonical selection path and the exact interval mapping. The minimal simplification is to select on the unjittered clean reference window, use the second view and 128/64 nesting only as rejection checks, take the already specified track median for evaluator use, and set g_j=1 on valid intervals and 0 otherwise unless an explicit confidence formula is supplied. |
| Exact 1,000-fixture distribution | **OPEN / BLOCKING** | “At least 250” plus a list of covered phenomena does not determine a fixture population, yet pass rates depend on the mixture. Freeze category counts summing to 1,000, parameter distributions and composition/overlap rules, RNG/library versions, and either the generator hash plus expected arrays or canonical fixture bytes plus SHA-256. |
| Warped pose/confidence/mask interpolation | **OPEN / BLOCKING** | The text forbids crossing invalid cells but never defines how x/y/confidence and joint/frame masks are produced at tau(q). Freeze the interpolation kernel and dtype/rounding, endpoint rule, joint-validity rule, mask propagation, and pause/reversal handling. A minimal linear rule may interpolate x/y/confidence only when both bounding source joints are valid, mark all other joints invalid and zero them after mask formation, and reuse the same fail-closed cell rule already stated. |
| K4 fixed vectors and primary-arm clock-code exposure | **PARTIAL / BLOCKING** | The q alternating vector and L half-positive/half-negative vector are exact. The two seeded orthogonal mask vectors are not, and primary-arm exposure is ambiguous because the 68-channel base omits the code while the K4 section adds it after Conv1. Bind literal vector bytes/hashes or a fully specified PCG64/Rademacher/orthogonalization algorithm, then state in one sentence whether both primary arms receive the identical fixed code or neither does. Keep the choice identical between treatment and augmentation-only. |

All seven concerns are interface definitions inside the existing learner, selector, augmentor, fixtures, or K4 receipt. Closing them does not authorize a new model, baseline, job, endpoint, or visible block.

## Preserved technical audit

| Item | Status | Finding |
|---|---|---|
| Discrete warp law and orientation | **CLOSED** | Both branches retain original signed increments. The target is the unique float64 sum of clean signed rates times oriented warped-cell overlap; reversal changes interval orientation rather than invoking canonical orientation. The clean target is stop-gradient and invalid/alias support fails closed. |
| Phase/fundamental claim boundary | **CLOSED PROSPECTIVELY** | The selector is explicitly a pseudo-cycle, not an annotated physical repetition. The evaluator-only 0.5x/1x/2x gate runs after freeze, rejects only, and cannot correct counts. The remaining selector numerical interfaces above concern reproducibility, not claim drift. |
| Interval NOLA and decode geometry | **CLOSED** | The interval weight is strictly positive over every half-open window cell, D_j must be finite and at least 1e-3, padding creates no mass, and each supplied identity is decoded once. The scalar g_j that multiplies valid mass remains to be defined. |
| Population and duplicate clocks | **CLOSED AS A PRECONDITION** | Duplicate disagreements invalidate rather than average; eligibility is count-blind; ambiguous slots stay excluded; all original nine development components must remain nonempty before evaluator access. Actual passage remains prospective. |
| Primary endpoint and bootstrap | **CLOSED** | AvgMAE is normalized person-first and aggregated video-first; every C>0 assertion is fail-closed; K1 requires both R>=0.05 and an absolute paired-effect CI lower endpoint greater than zero. Exactly 10,000 PCG64(20270815) draws resample all nine frozen source components with multiplicity and linear 2.5%/97.5% quantiles. |
| Learner, ledger, and presentation | **CLOSED** | There is one 225,026-parameter base learner and one treatment deletion. The cap is 42 full + 14 training-only tuning + 3 sanity = 59 jobs, with 42x12 + 14x3 + 6 = 552 GPU-hours. The paper retains three visible validation blocks. |

## Scores

| Dimension | Weight | Score / 10 | Weighted contribution |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9.9 | 1.485 |
| Method Specificity | 25% | 8.2 | 2.050 |
| Contribution Quality | 25% | 8.0 | 2.000 |
| Frontier Leverage | 15% | 9.2 | 1.380 |
| Feasibility | 10% | 8.0 | 0.800 |
| Validation Focus | 5% | 8.0 | 0.400 |
| Venue Readiness | 5% | 7.4 | 0.370 |

**WEIGHTED COMPOSITE: 8.485 / 10 (reported: 8.5 / 10)**  
**CALIBRATION: none**

**GAP:** No curated three-good/three-bad proposal anchors were supplied, so calibration remains none and no exemplar comparison is fabricated. Relative to the READY bar, the proposal has an unusually exact core signed-overlap law, optimizer, NOLA, normalized K1 estimator, component bootstrap, staged budget, and kill logic. It is still one implementation-freeze round short locally: selector preprocessing/numerics, selector-to-decoder binding, fixture generation, warp resampling, and K4 code/decision interfaces do not yet determine one program. Separately, TWCRAC full-method coverage is zero, so contribution distinctness and claim freeze remain externally blocked. These gaps require definitions and a primary source, not added experiments. No efficacy result or gate passage is inferred.

## Dimension review

### 1. Problem Fidelity - 9.9/10

The proposal remains exactly on the supplied-track tempo-drift question. It tests one local signed warp-integral loss against identical augmentation, keeps count and period labels outside training/tuning, evaluates per-person outputs with normalized video-first AvgMAE, and resamples source-connected components. The partial-cache GT-bbox-assisted claim ceiling is unchanged.

### 2. Method Specificity - 8.2/10

The mathematical treatment is strong. Raw principal increments define a signed cell measure; the target is one unique float64 oriented-overlap integral; the warped branch retains its own sign; pause, reversal, support, aliasing, stop-gradient, and wrap are explicit. The full-identity recurrent graph and optimizer are coherent, and strict positive-interval NOLA removes zero-boundary ambiguity.

The score is held down because the selector and augmentor do not yet map uniquely to code. The seven-interface table identifies the exact omissions. Freeze those numerical choices rather than adding machinery. In particular, a single canonical selector path and g_j=1 is simpler and more auditable than inventing a confidence model. The warped pose kernel must use the same validity semantics as the integral. **Priority: CRITICAL LOCAL.**

### 3. Contribution Quality - 8.0/10

The contribution remains singular: a discrete local warp-phase integral objective evaluated by an identical deletion and mechanism controls. The selector, harmonic gate, K4 receipts, and identity diagnostic are non-contributions.

Contribution quality cannot rise while TWCRAC is unavailable in full. Its abstract overlaps pose, local temporal cycle consistency, periodicity, and non-stationary motion, but abstract-only evidence supports neither collision nor distinctness. K8 must remain blocked.

### 4. Frontier Leverage - 9.2/10

The proposal uses exact transformation supervision, deterministic provenance, causal deletion, component-aware uncertainty, and fail-closed gates where useful. A foundation model, LLM, VLM, diffusion model, router, or expert bank is not naturally relevant and would weaken attribution. No forced frontier component is recommended.

### 5. Feasibility - 8.0/10

The compute ledger and full-track optimizer are finite and coherent. The audited period schema is implementable, and staged failures save budget. Feasibility remains at 8 because several pre-run fixtures and tensor-building paths are not uniquely specified, the system is not implemented, and all-nine-component/selector gates may legitimately fail. The fixes are small code-contract freezes but must precede training.

### 6. Validation Focus - 8.0/10

The primary 5%-AND-CI endpoint is executable: normalized person error, video-first aggregation, seed aggregation, B=10000 paired nine-component resampling, multiplicity, and strict positive lower endpoint are all frozen. Secondary Period-mAP cannot rescue it, and three visible blocks remain focused.

Gate 2 and K4 are not yet executable as decisions. The 1,000-fixture population is mixture-dependent, K4 retention/shuffle/resampler predicates lack a complete numeric table, and g_j lacks a value. These are validation-contract gaps, not reasons to extend the validation matrix.

### 7. Venue Readiness - 7.4/10

The proposal is focused enough for an ICASSP 4+1-page story but not yet a reproducible preregistration or novelty-frozen submission plan. One local interface-freeze pass is required; the external K8 gate then remains. There is no implementation and zero eligible result, so any future claim must retain the pilot ceiling.

No dimension is below 7, so the rubric-mandated sub-7 repair list is empty. The local and external blockers below remain mandatory for READY.

## Focus, simplicity, and frontier assessment

- **Anchor:** verbatim preserved.
- **Dominant contribution:** one discrete local warp-integral loss.
- **Focus:** strong; gates and fixtures remain receipts, not contributions.
- **Simplicity:** conceptually strong; the remaining work is to choose deterministic interfaces, preferably fail-closed and constant-confidence.
- **Frontier appropriateness:** strong and unforced.
- **Local implementation status:** not closed; bounded numerical/tensor contracts remain.
- **Claim status:** independently blocked by TWCRAC K8.

## Simplification Opportunities

1. Use fail-closed landmark and source-span rules rather than adding learned imputation, box fallbacks, or a second selector.
2. Make one unjittered clean selector path authoritative; use nested views only for rejection, and set g_j=1 on valid intervals unless a fully specified existing confidence is essential.
3. Bind fixture bytes/hashes and K4 vector bytes/hashes instead of adding prose-heavy generator flexibility.
4. Keep the raw-sign, one-loss, one-base-learner, one-decode, three-block route unchanged.

## Modernization Opportunities

**NONE.** The method is already frontier-aware for the anchored signal-processing bottleneck. Forced modernization would not close any identified gap.

## Remaining action items

1. **CRITICAL LOCAL - preprocessing/selector:** Freeze missing-landmark root/scale handling; the retained-clock-to-256 FFT grid; FFT normalization/off-bin/endpoint rules; and one nested-view-to-training-selection mapping.
2. **CRITICAL LOCAL - decoder/fixtures:** Define g_j exactly, preferably as 1 for valid intervals, and freeze the complete 1,000-fixture population through exact counts/distributions plus generator or canonical-byte hashes.
3. **CRITICAL LOCAL - warp tensors:** Freeze x/y/confidence interpolation, mask propagation, endpoint/dtype rules, and pause/reversal behavior under the existing tau and invalid-cell protocol.
4. **CRITICAL LOCAL - K4:** Bind the two mask-code vectors and primary-arm q/L-code exposure, then restore an exact decision table. For positive G=M_control-M_treatment, define shortcut retention Q_k=(M_control-M_k)/G on the unclipped seed-averaged normalized video-first metric; give equally numeric thresholds for shuffle preservation and unseen-resampler erasure. Add no job, model, baseline, or block.
5. **CRITICAL EXTERNAL FOR CLAIM FREEZE:** Obtain the complete TWCRAC primary source and rerun independent residual-equivalence adjudication. Until an explicit pass exists, retain BLOCKED, NOT_PASSED_BLOCKED, claim_freeze=BLOCKED, and novelty_clearance=false.

## Drift Warning

**NONE.** Every required fix is an implementation definition inside the existing supplied-track learner, selector, augmentor, fixtures, decoder, or K4 receipt. This review adds no experiment, baseline, learner, endpoint, or contribution.

## Verdict

**REVISE**

The score remains 8.5 after deeper implementation audit: Round 3 closed its optimizer, period, and staging blockers, but the complete replacement still does not determine one selector/augmentor/K4 implementation. The route is focused and coherent, so RETHINK is unwarranted. READY is prohibited by the local interface gaps and, independently, the authoritative TWCRAC BLOCKED/NOT_PASSED_BLOCKED gate. This is a same-family provisional proposal review with no invented result.

## Audited input SHA-256

| Input | SHA-256 |
|---|---|
| refine-logs/round-3-refinement.md | 327d6db30aeedbb8bb30122227c9d081660cf4145b71c68db12114db7c879f17 |
| refine-logs/REFINE_STATE.json | 39152e46b06ab46b8d35e31110696692ec5b94fefafe708e38c4a5da8f277658 |
| refine-logs/score-history.md | 9b88843b885ffbd47ec8c69f3fbda537406d0b7113746717f94dece9c74a33f9 |
| PILOT_PERIOD_SCHEMA_ADDENDUM.md | 49edcfe25c302c2b55b8e65bc18e1cb7febe25cc8b76698d631fa24150d4b0b5 |
| idea-stage/TWCRAC_ADJUDICATION.json | 5a1e7b8c1048a8884690f2d234d413064d473ead1eb1deca69ab734ca2e5ed1e |
| refine-logs/round-3-review.md | 85ffbd8aaa84dbe65963f68027c9f916a9b7ca8b8b9911ff93b161174d4db2f2 |

</details>

<details>
<summary>Round 5 Review — full verbatim stored response</summary>

# Round 5 Final Research-Refine Re-evaluation: WARP-PHASE

**Reviewer:** GPT-5.6-Sol (/root/research_refine_reviewer)  
**Review continuity:** same reviewer thread as Rounds 1-4  
**Review independence:** same-family  
**Acceptance status:** provisional  
**Round status:** MAX_ROUNDS final research-refine review  
**Evidence status:** proposal-only; zero eligible method results  
**Venue target:** ICASSP 2027 (4+1 pages)

## Executive decision

Round 4 materially improves the proposal and closes most of the seven named interfaces. COCO17 root/scale handling is fail-closed and unique; retained-clock supports map to one inclusive 256-sample source-clock grid; ACF/FFT normalization, off-bin interpolation, endpoint maxima, and ties are specified; g_j is exactly binary; warp pose/confidence/mask interpolation is deterministic; K4 primary-arm exposure, fixed vectors, and ratio thresholds are explicit; and the desensitized packer schema is bound to exact bytes. Independent recomputation confirms the claimed 617-byte packer fixture SHA-256 and the claimed 4,096-byte K4 vector SHA-256.

The local specification is nevertheless not fully closed. Three bounded execution contracts remain underdetermined. First, the real-track weak rejection view is only called “seeded”; no seed-to-track/parent mapping or deterministic traversal is frozen, even though its output can reject a canonical selection. Second, the 1,000-fixture section fixes category counts and serialization but not one complete generator: the RNG position of the second/third-harmonic draws, their exact x/y formulas, symmetric category coefficients, stochastic array shapes/order, and some category transforms are not unique. It also mandates NumPy 2.4.6 and fails any mismatch, while the project lock, supported server environment, pose constraint, and local .venv resolve NumPy 1.26.4. A dedicated 2.4.6 fixture environment could be valid, but none is locked or witnessed in the proposal. Third, K4 leaves the denominator of f_track and the actual matched-time shuffle permutation and unseen-resampler kernel/schedule unspecified; the decision ratios are numeric, but the inputs to two ratios are not uniquely constructible.

These are local constants, formulas, and environment receipts only. They do not justify a new model, experiment, baseline, endpoint, or validation block. Separately, TWCRAC remains an external claim blocker: the supplied adjudication has zero full-text pages, verdict BLOCKED, k8.decision=NOT_PASSED_BLOCKED, claim_freeze=BLOCKED, and novelty_clearance=false. No abstract-level absence inference is permitted.

**LOCAL_SPEC_STATUS: NOT_CLOSED - THREE_BOUNDED_EXECUTION_BLOCKERS**  
**EXTERNAL_K8_STATUS: BLOCKED / NOT_PASSED_BLOCKED**  
**MAX_ROUNDS_STATUS: FINAL_RESEARCH_REFINE_VERDICT**

## Anchor verification

**PASS.** The two Problem Anchor copies in refine-logs/round-4-refinement.md are byte-for-text identical to each other and to the original frozen anchor. The anchor-text SHA-256 is 619148c765175167da093d9b054bd2b43fae0e5e97d981fb9748ef57f4d7dba7. The supplied-track population, video-first estimand, label restrictions, comparator family, secondary diagnostics, excluded questions, and pilot claim ceiling are unchanged. There is no predicted-track, scene-total, end-to-end, annotation-free, generic representation-learning, or historical-result drift.

## Seven-interface and packer closure audit

| Interface | Final status | Finding |
|---|---|---|
| COCO17 root/scale | **CLOSED** | Shoulders 5/6 and hips 11/12 are named. Root and shoulder means use available valid members; no hip invalidates a frame; no positive finite track-scale sample invokes K7; the float64 median and 1e-3 floor are unique; boxes, temporal carry, learned imputation, population statistics, and evaluator fallbacks are forbidden. |
| FFT256 source grid | **CLOSED** | Every canonical/rejection support uses 256 inclusive float64 source-clock queries with Delta=(q_last-q_first)/255. Same-cell validity, internal-breakpoint/right-cell assignment, final-endpoint/left-cell assignment, source-frame period units, and no truncation/padding/alternate branch are explicit. |
| FFT/ACF bins, normalization, maxima, and ties | **CLOSED** | The ACF pair query and support count, valid-mask mean, periodic Hann, rfft norm, mask-energy denominator, bins 1-127, zero/Nyquist exclusion, off-bin linear interpolation, harmonic weighting, endpoint one-sided maxima, interior two-sided maxima, and exact-score/smaller-P tie rule determine one scorer. The three numerical fixtures cover off-bin and both endpoints. |
| Canonical view and g_j | **PARTIAL / BLOCKING** | The unjittered parent is authoritative; nested 64 supports and one weak view reject only; the track value is a float64 median; g_j is exactly 1 on valid intervals and 0 otherwise. However, the real-track weak view has no RNG seed, per-track/parent derivation, traversal order, or stored perturbation hash. Because weak-view disagreement can invoke K5, different valid implementations can retain different parents. |
| Exact 1,000-fixture generator | **PARTIAL / BLOCKING** | Nine mutually exclusive ID ranges sum to 1,000 and exactly 250 are symmetric; PCG64 seed, high-exclusive integers, missingness thresholds, canonical serialization, filename ordering, source/pack receipts, and thresholds are stated. The prose does not place the second/third-harmonic coefficient draws in the declared RNG order or give complete coordinate formulas for those terms and the symmetric half/double templates. It also omits exact shapes/order for missingness/jitter/dropout arrays and leaves parts of pause/three-segment composition to interpretation. No current source or pack hash can resolve those choices. |
| Warp tensor/mask interpolation | **CLOSED** | Endpoint-inclusive support, exact stored-clock copying, same-cell float64 linear x/y/confidence interpolation, both-endpoint joint validity, mask-before-zero, float32 ties-to-even, frame/cell masks, pause, reversal, conflicts, and out-of-support behavior are unique and shared by model, recurrence, integral, loss, and diagnostics. |
| K4 vectors, exposure, and ratios | **PARTIAL / BLOCKING** | Primary treatment/control receive no fixed code; only Stage-B controls do. u_q, u_L, the PCG64 Rademacher/MGS construction, orientation, byte order, literal length/hash, Q_clock, Q_mask, P_shuffle, P_unseen, and inequalities are exact. Recomputed vector bytes match SHA-256 5e2f35a11bd8a376e44d5ad7d3e5067a0bb5a03e2d61edd84f5f5bfd6cf351ec under both NumPy 2.4.6 and the locked 1.26.4. But f_track does not name its denominator, and neither the matched-time permutation nor the held-out unseen-resampler kernel/schedule is defined, so M_c^sh, M_t^sh, M_c^u, and M_t^u are not uniquely reproducible. |
| Desensitized packer schema precondition | **CLOSED AS A PROSPECTIVE GATE-0 CONTRACT** | The single UTF-8/no-BOM/no-newline canonical JSON is exactly 617 bytes and independently hashes to 31f81549f9c5dbedc6ddb278e868793246826a234de90c06c169c71387428275. It separates feature, vault, and annotation schemas, prohibits real identifiers and cross-side visibility, rejects extra/executable fields, and requires hashes before eligibility. It is a precondition, not evidence that the packer or permissions already exist. |

## Environment feasibility audit

The proposal makes NumPy 2.4.6 normative for fixture generation and says any version mismatch fails. The repository evidence instead freezes NumPy 1.26.4 in uv.lock and docker/server/requirements.txt; the pose extra constrains NumPy below 2; and the supported local .venv reports 1.26.4. The broad base dependency allowing NumPy below 3 does not override those concrete locks.

This is not a reason to change the scientific route. It is a minimal environment choice that must be made before implementation:

1. Prefer the already provisioned path: make NumPy 1.26.4 normative for the fixture generator, freeze its generator source and pack hash, and retain the K4 vector hash, which was independently reproduced unchanged under 1.26.4.
2. Alternatively, define a dedicated non-training fixture/receipt environment pinned to NumPy 2.4.6 with Python/platform/wheel hashes, prove it provisions independently of the pose/training environment, and add a load-and-hash witness showing its canonical JSON/NPY/raw-vector artifacts are consumed unchanged by the locked 1.26.4 training/evaluator environment.

Until one path has an environment lock and witness, the current “version mismatch fails” clause makes the supported implementation fail by construction.

## Preserved technical audit

| Item | Status | Finding |
|---|---|---|
| Discrete signed warp law | **CLOSED** | Raw clean and warped increments retain their signs. The unique float64 oriented-overlap integral is the sole target, the clean teacher is stop-gradient, and support/alias failures are fail-closed. Pause and reversal require no canonical orientation. |
| Phase/fundamental claim boundary | **CLOSED PROSPECTIVELY** | A selector-defined pseudo-cycle is only an internal unit. The frozen evaluator-only 0.5x/1x/2x gate runs after prediction freeze, rejects only, and cannot correct counts or claim semantic identity. |
| Positive interval NOLA and recurrence | **CLOSED** | Interval weights are strictly positive, D_j must be finite and at least 1e-3, one decode follows one complete identity pass, recurrence commits once per real clock, and no graph/cache crosses a training step. |
| Optimizer and budget | **CLOSED** | Eight complete identities per fresh step, four-step gradient accumulation, one backward per step, 20,000 updates, and 640,000 track draws are coherent. The staged maximum remains 59 jobs and 552 GPU-hours. |
| Population and evaluator period | **CLOSED AS PRECONDITIONS** | Duplicate conflicts fail rather than average; eligibility is count-blind; all nine source components must remain nonempty; raw [s,e) periods map uniquely to a float64 median. Passage is not inferred. |
| Primary endpoint and bootstrap | **CLOSED** | K1 is the 5% relative normalized video-first AvgMAE improvement AND an absolute paired-effect CI lower endpoint above zero. B=10,000 paired PCG64 draws resample all nine components with multiplicity and linear central quantiles. |
| Focus and presentation | **CLOSED** | One 225,026-parameter base, one treatment deletion, three visible blocks, no new baseline, and zero eligible results remain explicit. |

## Scores

| Dimension | Weight | Score / 10 | Weighted contribution |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9.9 | 1.485 |
| Method Specificity | 25% | 8.8 | 2.200 |
| Contribution Quality | 25% | 8.0 | 2.000 |
| Frontier Leverage | 15% | 9.2 | 1.380 |
| Feasibility | 10% | 8.2 | 0.820 |
| Validation Focus | 5% | 8.6 | 0.430 |
| Venue Readiness | 5% | 7.7 | 0.385 |

**WEIGHTED COMPOSITE: 8.700 / 10 (reported: 8.7 / 10)**  
**CALIBRATION: none**

**GAP:** No curated three-good/three-bad proposal anchors were supplied, so calibration remains none and no exemplar comparison is fabricated. Relative to the READY bar, the proposal now has a highly specific core: exact signed warp math, phase-claim boundaries, full-identity optimization, NOLA, source-clock FFT/ACF, packer and K4 byte receipts, normalized K1, component bootstrap, staged controls, budget, and kill rules. It remains locally short of one program because a rejection-view RNG, the complete synthetic generator/environment, and two K4 corruption constructors are not frozen. More importantly for any paper claim, TWCRAC method coverage remains zero pages; contribution distinctness and claim freeze are externally unadjudicated. No result, Gate 0-5 passage, or novelty absence is inferred.

## Dimension review

### 1. Problem Fidelity - 9.9/10

The complete replacement preserves the anchored supplied-track tempo-drift question, video-first unit, source-component uncertainty, label restrictions, matched deletion, comparator family, and pilot-only ceiling. The new constants close implementation interfaces without changing the population, estimand, contribution, or success rule.

### 2. Method Specificity - 8.8/10

The method core is unusually explicit. Root normalization, continuous source-clock velocity, uniform FFT queries, exact ACF/FFT score, raw signed cell measure, oriented pullback integral, warp tensor construction, full-sequence recurrence, NOLA, and K1 scorer now determine almost all numerical behavior.

The score stays below 9 because three decision-bearing input generators remain non-unique: the real weak rejection view, parts of the synthetic fixture generator, and the K4 shuffle/unseen corruptions. The environment clause also conflicts with the supported lock. These are important implementation contracts, not alternative scientific methods. **Priority: CRITICAL LOCAL.**

### 3. Contribution Quality - 8.0/10

The contribution is focused and appropriately narrow: one discrete warp-integral equivariance objective tested by an identical deletion and staged falsification. ACF/FFT, pseudo-cycle selection, private state, NOLA, K4, and identity isolation remain non-contributions.

Quality is capped by K8. The available TWCRAC abstract overlaps pose, local periodicity, temporal cycle consistency, and non-stationary counting; none of those is claimed as residual novelty. But without the complete method, the signed local warp objective cannot be declared distinct or colliding.

### 4. Frontier Leverage - 9.2/10

The proposal uses current scientific-control ideas where natural: exact transformation laws, causal deletion, deterministic provenance, component-aware uncertainty, and fail-closed receipts. It correctly avoids forcing an LLM, VLM, diffusion system, router, or expert bank into a signal-processing mechanism test.

### 5. Feasibility - 8.2/10

The optimizer, tensor shapes, warp kernel, compute ledger, and staged stops are feasible. The packer and period interfaces are implementable as gates. Feasibility is limited by the unimplemented system, severe prospective gates, incomplete generators, and the NumPy 2.4.6 mandate conflicting with the supported 1.26.4 lock. A dedicated generator environment is acceptable only after a concrete lock and cross-environment witness.

### 6. Validation Focus - 8.6/10

K1 and its uncertainty endpoint are fully executable and cannot be rescued by secondary Period-mAP. The validation story remains three blocks, with integrity and numerical fixtures kept as receipts. The score loses only because K5 can depend on an unspecified weak-view perturbation, and two K4 metrics depend on unspecified corruptions. No additional validation family is needed.

### 7. Venue Readiness - 7.7/10

The local route is close to a reproducible ICASSP preregistration, but it is not implementation-ready or novelty-frozen. The three local contracts can be closed without changing the story. TWCRAC remains a decisive external venue/claim risk, and there are zero eligible results.

No dimension is below 7, so the rubric-mandated sub-7 fix list is empty. The blockers below remain mandatory despite that.

## Focus, simplicity, and frontier assessment

- **Anchor:** verbatim preserved.
- **Dominant contribution:** one discrete local warp-integral objective.
- **Focus:** strong; fixtures and gates remain receipts rather than claims.
- **Simplicity:** strong; the remaining fixes are seed mapping, literal formulas, environment locking, and corruption constants.
- **Frontier appropriateness:** strong and unforced.
- **Local specification:** not closed.
- **Claim freeze:** externally blocked by K8.
- **Evidence:** proposal-only, zero eligible results.

## Simplification Opportunities

1. Derive every real weak rejection view from one frozen Gate-2 seed stream over a checksum-sorted eligibility/parent-start order, and store the perturbation hashes. Do not add a second view or confidence model.
2. Use the already supported NumPy 1.26.4 environment for all fixture/reference bytes unless a separate 2.4.6 environment is explicitly locked and witnessed.
3. Make f_track one named fraction and bind one deterministic shuffle plus one held-out resampler kernel/schedule. Keep the existing four K4 ratios and six Stage-B jobs.
4. Preserve the one-loss, raw-sign, one-base, one-decode, three-block design.

## Modernization Opportunities

**NONE.** No frontier component is missing, and forced modernization would not close the identified risks.

## Exact remaining risks and minimal actions

1. **CRITICAL LOCAL - weak rejection view:** Freeze the Gate-2 RNG algorithm/seed, checksum-sorted track order, parent-start order, draw shapes/order, and stored perturbation hash. The opaque key may route to the frozen order but must not become a model feature.
2. **CRITICAL LOCAL - fixture generator and environment:** Put harmonic-coefficient draws at an exact RNG position; give complete x/y equations for second/third harmonics and symmetric half/double templates; specify all stochastic array shapes and the pause/three-segment transform order. Then either make locked NumPy 1.26.4 normative and hash the generator/pack, or provide a pinned 2.4.6 dedicated-environment plus 1.26.4 consumption witness.
3. **CRITICAL LOCAL - K4 construction:** Define f_track with an exact numerator and denominator over named collapsed/non-padded masks. Freeze the matched-time shuffle scope/permutation/seed and the unseen-resampler kernel, target grid, schedule, seed, and receipt. Keep Q_clock, Q_mask, P_shuffle, P_unseen, thresholds, jobs, and blocks unchanged.
4. **CRITICAL EXTERNAL FOR CLAIM FREEZE:** Obtain the complete TWCRAC primary source, hash and inspect every method/loss/training/evaluation page, and rerun independent residual-equivalence adjudication. Until an explicit pass exists, retain BLOCKED, NOT_PASSED_BLOCKED, claim_freeze=BLOCKED, and novelty_clearance=false.
5. **PROSPECTIVE EXECUTION RISK:** Even after specification closure, Gate 0 population/permission checks, Gate 1 bytes/numerics, Gate 2 selector stability, Gate 3 gradients, K1 efficacy/CI, K2 comparators, K3 redundancy, K4 shortcuts, K5 phase execution, K6 integrity, and K7 coverage can all fail. None has run.

## Drift Warning

**NONE.** Every local action is a deterministic definition inside the existing selector, fixture builder, or K4 diagnostic. This review adds no model, baseline, job, endpoint, validation block, or contribution.

## Final research-refine verdict

**REVISE**

This is the MAX_ROUNDS final research-refine verdict. The proposal rises from 8.5 to 8.7 because four interfaces and the packer precondition are now genuinely closed and the core method is substantially executable. RETHINK is unwarranted: the scientific route remains focused. READY is prohibited twice over: three bounded local execution contracts remain open, and the independent TWCRAC K8 gate remains BLOCKED/NOT_PASSED_BLOCKED. No final proposal, report, experiment plan, or efficacy claim is generated by this review; the main thread must carry forward the exact risks above.

## Audited input SHA-256

| Input | SHA-256 |
|---|---|
| refine-logs/round-4-refinement.md | 2031d5c84b8c6c329dd0e43e5b8a09210ed9a3e8b17b47578f2f878a4c2b4e0f |
| refine-logs/round-4-review.md | c1d44bbe3dd5467b02c4d29bf7bba2849b5bcf64a705c8276f0882a072a2e7a1 |
| refine-logs/round-4-review.json | d62ec17f33d7f76091a20ebeaaaa23c0a402b5c028487ad1277311b4dc6363ad |
| refine-logs/REFINE_STATE.json | 54521958169008256110c90f11d8dad68389895920688550a474f8ad1c4727c6 |
| refine-logs/score-history.md | bfdebeda8851c2ca08f4e7ed7601f38ccb70b0c7fc16fcf09ee0c3ca835b5f1d |
| PILOT_PERIOD_SCHEMA_ADDENDUM.md | 49edcfe25c302c2b55b8e65bc18e1cb7febe25cc8b76698d631fa24150d4b0b5 |
| idea-stage/TWCRAC_ADJUDICATION.json | 5a1e7b8c1048a8884690f2d234d413064d473ead1eb1deca69ab734ca2e5ed1e |
| pyproject.toml | 14b3b35bbfa57fc982e24ef2f999df288893c0f0d12a53a9f82120babefead38 |
| uv.lock | 5a964f1cd841948eae09ef253098b0b16eb5d2e6bd660e2d3496b389c4dc72a7 |
| docker/server/requirements.txt | 2337391c20fae1533b6e30d042f668a5542db28f0a29a71d5cb86b74517bfc53 |
| .venv/pyvenv.cfg | 598ae04c323c0a6ec8b1e17e5c99677d255cd37a8a82e746bc954e3113bfe2b7 |

</details>

## Next Steps

Because the final verdict is `REVISE`, specification work must first close the three local contracts without adding models, jobs, endpoints, or visible blocks. Then rerun `/research-refine` if a fresh review is desired. Training remains unauthorized until Gates 0–2 pass. Claim freeze additionally requires complete-primary-source TWCRAC adjudication. This Phase 5 finalization does not create an experiment plan, implementation, or code.

