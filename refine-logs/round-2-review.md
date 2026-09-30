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
