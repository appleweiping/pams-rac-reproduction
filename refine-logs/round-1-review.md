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
