# Round 2 Research-Refine Refinement: WARP-PHASE

> **PROSPECTIVE PILOT**
>
> **NO ELIGIBLE RESULTS**
>
> **SAME-FAMILY PROVISIONAL**

**Executor:** proposal executor

**Review being answered:** `refine-logs/round-2-review.md`

**Reviewer:** GPT-5.6-Sol (`/root/research_refine_reviewer`)

**Round-2 score/verdict:** 7.9/10, `REVISE`

**Permitted protocol name:** **GT-bbox-assisted AlphaPose supplied-track partial-cache development pilot**

This is a complete prospective replacement proposal. No canonical implementation exists, no eligible method run has been made, and no numerical efficacy statement is licensed. Historical v46, v62, and v63 artifacts remain audit inputs only.

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

## Anchor Check

- **PASS — original bottleneck:** The treatment still tests one explicit local time-warp law for raw signed phase increments; the primary control sees the identical augmentations without that loss.
- **PASS — population and estimand:** Only checksum-frozen, source-disjoint partial-cache train/development entries are used. Evaluation stays per-person then video-first, with the audited source components as bootstrap clusters.
- **PASS — supervision:** The model process never reads count, density, evaluator period, cycle boundary, object mapping, or source identity. Evaluator period is opened only after the relevant predictions and traces are frozen and cannot alter them.
- **PASS — endpoint:** K1 remains the 5% relative normalized AvgMAE improvement **and** positive paired-component CI rule. Period diagnostics and the evaluator harmonic gate cannot rescue K1.
- **PASS — scope and evidence:** No router, learned expert, new TSSM, private-state novelty, scene-total objective, predicted-track claim, or historical efficacy claim was added. Eligible results remain zero.
- **Rejected as drift:** Treating an ACF/FFT selection as a semantic action cycle would silently replace the anchored counting question with self-consistency of a pose oscillation. The revision instead names it a selector-defined pseudo-cycle and uses a frozen, no-correction evaluator-only rejection gate.

## Simplicity Check

- **Dominant contribution:** one discrete warp-integral equivariance loss is the only proposed mechanism.
- **Deleted complexity:** canonical clean/warp orientation is removed completely. Both sides retain their original signed increments; reversal is represented by the negative oriented interval integral, and the absolute-mass decoder is globally sign-equivariant.
- **Frozen shared base:** ACF/FFT selection and PAMS-TCC-style correspondence remain shared, non-trainable/non-contribution machinery. Their output is only a selector-defined pseudo-cycle.
- **One recurrent pass:** the GRU executes once on each complete collapsed identity sequence. Overlap windows read cached states and can never commit a second update.
- **Three visible validation blocks:** primary pair; staged mechanism/closest-prior falsification including K4 controls; identity isolation. Gates and fixtures remain receipts rather than paper contributions.
- **No modernization module:** known warp metadata, exact equivariance supervision, causal interventions, and matched deletion already fit the bottleneck. Adding an LLM, VLM, diffusion model, router, or expert bank would not sharpen the claim.
- **Open novelty boundary:** K8 remains open because there is no independent full-method TWCRAC receipt. This revision does not claim that collision is closed.

## Changes Made

1. **Removed asymmetric canonical orientation.** Reviewer found that Round 1 oriented the clean teacher but not the warped student. The revision deletes canonical orientation everywhere: raw signed `delta_k` forms the clean rate, raw signed `delta_j^tau` is the student, and reversal obtains its negative sign only from the oriented overlap length. This is the smallest repair and leaves the absolute decoder globally sign-equivariant.
2. **Separated pseudo-cycle construction from evaluator semantics.** ACF/FFT now emits only a selector-defined pseudo-cycle. After all relevant predictions and traces are frozen, an evaluator-only period vault applies a predeclared `0.5x/1x/2x` gate with 10% tolerance, video-first aggregation, fixed 90%/5% failure thresholds, and no correction, filtering, retuning, or rerun. Seeded fixtures explicitly include symmetric harmonic signals.
3. **Made recurrence single-commit.** Convolutions and the GRU run chronologically once over each complete collapsed identity. Invalid cells hold the previous hidden state. Every hidden state is cached once; overlapping loss/decoder windows are read-only views of that cache.
4. **Corrected AvgMAE.** Every per-person term is `abs(prediction-ground_truth)/count_gt`, then averaged within video and across videos. The evaluator asserts `C>0` before division, and the hand-worked two-component fixture is recomputed under this normalized formula.
5. **Derived the nine-cluster population at Gate 0.** After all count-blind eligibility rules, each of the original nine audited development components must retain at least one eligible video. Failure invokes K7 before evaluator access; no metric-informed subset is allowed.
6. **Completed the hard run ledger.** Three seeds each for a trained timestamp/no-pose model and a trained nuisance-only model are decision-bearing Stage B jobs. The absolute ceiling is now 59 jobs and 552 GPU-hours, and every stage/stop bound is explicit.
7. **Preserved presentation focus.** The same three visible blocks remain; shortcut jobs appear in Block 2 and in the stage ledger without creating a fourth claim block.
8. **Kept TWCRAC unresolved.** K8 remains claim-blocking unless a future independent primary-source full-method receipt adjudicates it. No closure is asserted here.

---

# Revised Proposal: WARP-PHASE — Discrete Warp-Integral Equivariance for Supplied-Track MRAC

> **PROSPECTIVE PILOT**
>
> **NO ELIGIBLE RESULTS**
>
> **SAME-FAMILY PROVISIONAL**

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

## 1. Status, Thesis, and Contribution Boundary

This is a preregistered, kill-oriented partial-cache pilot, not a manuscript method freeze. It has not been implemented; eligible results are zero; expected directions are not evidence.

**Thesis.** A clean identity track supplies a discrete raw signed phase measure. Under a known target-to-source time map, the warped signed increment over a target interval should equal the signed integral of that clean measure over its mapped source interval. Treatment and control see the same warps; treatment alone receives this exact integral loss.

**Possible future contribution.** If and only if the full protocol passes, the narrow claim is a **discrete warp-integral equivariance objective for signed phase increments in supplied-track MRAC**. The continuous chain rule, ACF/FFT, PAMS/TCC-style phase learning, pseudo-cycle selection, pose SSL, local windows, temporal equivariance, private state, NOLA, and per-person decoding are established or generic and are non-contributions.

**Claim ceiling.** Permitted wording is “we study whether” on a GT-bbox-assisted AlphaPose supplied-track partial-cache development pilot. This is not end-to-end, annotation-free, predicted-track, state-of-the-art, or deployment evidence. PAMS already reports localized internal-tempo perturbation; DeepPhase already occupies unsupervised motion phase. TWCRAC full-method overlap is unresolved and K8 remains open and claim-blocking.

## 2. Count-Blind Data Contract and Eligibility

### 2.1 Physical isolation

A trusted offline packer may read the canonical v44 train/validation pickles once. It emits feature-only non-executable shards, a separately permissioned evaluator vault, a split/component manifest, a count-blind eligibility manifest, per-file/per-sample SHA-256 receipts, and a forbidden-key receipt. The training process cannot read the original pickle or vault and rejects non-whitelisted fields before payload deserialization.

Model-process fields are only `motion`, `person_mask`, `frame_mask`, `sampled_frame_indices`, `source_length`, an opaque sample key, and a local supplied-slot index. Source IDs and hashes are split/audit-only. Counts, densities, evaluator periods, boundaries, raw boxes, annotation names, object IDs/mappings, association structures, and `gt_object_pose_coverage` never enter a model tensor, objective, selector, tuning decision, or checkpoint.

### 2.2 Frozen count-blind primary eligibility and component assertion

Eligibility is computed and checksum-frozen before the evaluator opens `count_gt`, `density_gt`, or period annotations. It is never revised because of a label or metric.

1. Exclude all association-ambiguous slots: 13 training and 8 development slots. Require a one-to-one supplied-slot/object association in the audit manifest.
2. Require frozen GT-assisted pose coverage at least 0.80. The trusted packer uses this audit-only scalar solely to freeze inclusion, then removes it from model shards.
3. After duplicate-clock collapse, require at least 64 retained distinct source-clock locations, at least 63 valid adjacent cells, and at least 0.60 feature-valid clock coverage. A frame is feature-valid when at least 8 of 17 joints have AlphaPose confidence at least 0.20.
4. Do not filter on count, density, evaluator period, action, predicted difficulty, selector output, development error, or harmonic agreement. Report every exclusion and its count-blind reason by split and original audited component.
5. The selector must answer for every eligible slot. Selector failure is global K5, never silent slot removal.
6. **Gate-0 cardinality assertion:** after Rules 1–5, each of the original nine audited development source components must contain at least one eligible video with at least one eligible person. Hash the final eligible video/person/component table and use it unchanged for every arm. If any component is empty, invoke K7 before evaluator access; do not bootstrap a metric-informed subset or renumber components.

### 2.3 Duplicate-clock collapse

For each integer source clock `t`, group all samples with `sampled_frame_indices == t`. Retain the first sample only if every repeated member has identical frame/joint masks, maximum absolute normalized-`x,y` difference at most `1e-5`, and maximum confidence difference at most `1e-6`; otherwise mark `t` invalid. Never average disagreement. Sort retained clocks strictly increasingly. Interpolation is allowed only between adjacent retained feature-valid clocks and never crosses an invalid clock or cell. Eligibility, selector, clean-rate construction, warp interpolation, recurrence masks, and diagnostics share this rule.

## 3. Shared Established Phase Backbone

Treatment and augmentation-only control use the same Track-PAMS/PAMS-TCC-style label-restricted phase backbone. Its masked ACF/FFT pseudo-cycle selector and cycle-positive temporal correspondence are established-source adaptations, frozen identically in both arms and never presented as contributions.

### 3.1 Selector-defined pseudo-cycle

On each collapsed clean track, root-center valid joints at the hip midpoint, divide coordinates by the median valid shoulder-to-hip scale with floor `1e-3`, and form masked first differences on adjacent valid cells. On integer source-clock support, interpolate only inside valid adjacent cells. For nested clean windows of 64 and 128 retained clocks and two weak spatial views, compute

\[
R(P)=\frac{\sum_t m_t m_{t+P}\langle v_t,v_{t+P}\rangle}
{\sqrt{\sum_t m_t\lVert v_t\rVert^2\sum_t m_{t+P}\lVert v_{t+P}\rVert^2}+10^{-8}}
\]

for integer candidates `P` from 4 through

\[
P_{\max}=\min\!\left(128,\left\lfloor(q_{\mathrm{last}}-q_{\mathrm{first}})/2\right\rfloor\right),
\]

with at least 16 valid pairs, together with a 256-point masked FFT periodogram `E(f)`. The frozen score is

\[
S(P)=0.5R(P)+0.5\frac{E(1/P)+0.5E(2/P)+0.25E(3/P)}{1.75\sum_{f>0}E(f)+10^{-8}}.
\]

Local maxima with `R(P) >= 0.25` are candidates. Select the smallest `P` within 0.02 of the maximum score; remaining ties use smaller `P`. No evaluator label or development metric is used.

The output `hat P` is called only a **selector-defined pseudo-cycle**. Operationally the base loss maps one selected pseudo-cycle to one `2pi` winding; this fixes an internal unit and does not assert correspondence to an annotated action repetition. PAMS-TCC pseudo-cycle positives are `(t,t+hat P)` with valid support. For separations at most `hat P`, the pseudo-angle is `2pi(q_b-q_a)/hat P`. With valid pairs `A`,

\[
\mathcal L_{\mathrm{PAMS}}=|\mathcal A|^{-1}\sum_{(a,b)\in\mathcal A}
\rho_{0.10}\!\left(\operatorname{wrap}\!\left[\operatorname{Arg}(z_b\overline z_a)-2\pi(q_b-q_a)/\hat P\right]\right).
\]

This is circular Huber correspondence with `beta=0.10` rad. Weak views use normalized-coordinate Gaussian jitter `sigma=0.01`, clipped at 0.03, and 10% joint dropout while retaining at least 8 joints; clocks are unchanged. There is no activity head, reconstruction decoder, covariance/variance loss, or learned seam loss. Selector confidence `g_j` is fixed and untrained.

### 3.2 Training-only selector fixtures

Before any decision-bearing run, generate exactly 1,000 seeded pose-like fixtures with known semantic repetition units using seed 20270815. At least 250 fixtures are symmetric two-limb/two-pose motions constructed so the strongest raw harmonic is the half-unit or double-unit; the remainder cover injected second/third harmonics, three-segment warps, reversals, pauses, 40% missing frames, and 30% missing joints. Relative to the fixture generator's known unit, selector agreement within 10% must be at least 95%, while half-unit and double-unit selections must each be at most 2% over all fixtures and at most 5% within the symmetric subset.

On every eligible count-blind training track, the selected pseudo-cycle must agree within 10% across both spatial views and nested windows for at least 90% of parent windows; the score margin over `P/2` and `2P`, when present, must be at least 0.10. Reversal must preserve the pseudo-cycle magnitude within 10%. Any failure invokes K5. These gates establish only deterministic pseudo-cycle stability; they do not establish semantic repetition identity.

### 3.3 Frozen evaluator-only harmonic gate

For each reached decision stage, complete all predeclared arms and three seeds, serialize every raw per-person count prediction, selector output, phase trace, configuration, and SHA-256 receipt, then revoke write access. Only after this freeze may a separate evaluator process open the period vault.

For each eligible person `p`, the evaluator obtains the frozen annotated one-repetition period `P_eval,p` in source-frame-clock units and asserts it is finite and positive. The selector trace supplies one frozen track value `hat P_p`, the median of its pre-vault valid window selections with NumPy's linear median convention. For `h` in `{0.5,1,2}`, define

\[
d_{p,h}=\left|\frac{\hat P_p}{hP_{\mathrm{eval},p}}-1\right|.
\]

Assign `p` to the unique `h` with `d_{p,h} <= 0.10`; if none qualifies, assign `off-grid`. The intervals cannot overlap at this tolerance; any implementation tie or non-finite value is `off-grid`. For class `r`, compute person-first video fractions and then the unweighted video mean

\[
H_r=|V|^{-1}\sum_{v\in V}|P_v|^{-1}\sum_{p\in P_v}\mathbf 1[\operatorname{class}(p)=r].
\]

The gate passes only if `H_1 >= 0.90`, `H_0.5 <= 0.05`, `H_2 <= 0.05`, and `H_off-grid <= 0.05`. Report the same fractions by original source component, but component values do not replace the frozen overall rule. Failure invokes K5 and blocks the phase-mass claim. The evaluator may not correct counts, multiply/divide pseudo-cycles, select a harmonic, filter people/videos, tune tolerance, rerun training, or alter any frozen prediction. This gate is adjudication only, not calibration and not a secondary endpoint.

### 3.4 Shared network, exact count, and single-commit recurrence

Each pose step supplies 51 `x,y,confidence` values plus 17 joint-mask bits, for 68 channels. Missing values are zeroed only after mask concatenation. The shared network is Conv1d `68→128`, kernel 5, padding 2; Conv1d `128→128`, kernel 5, padding 2; one GRU `128→128`; and a linear phase head `128→2`, followed by L2 normalization. It has exactly 225,026 trainable parameters: 43,648 + 82,048 + 99,072 + 258. No other trainable head exists.

For each collapsed identity, run the convolutions and GRU once on the complete chronological sequence, with zero hidden state only at identity start. At valid clock `k`, commit `h_k=GRU(x_k,h_{k-1})`; at an invalid clock, hold `h_k=h_{k-1}` and mark the emitted cell invalid. Cache each `h_k` and normalized `z_k` exactly once. Batching may bucket identities but may never share hidden state. Overlapping 64-sample windows are read-only index views into this cache for losses and decoding: they never reset, carry, or update the GRU and cannot commit any clock twice.

## 4. Unique Discrete Warp-Integral Objective

### 4.1 Raw signed clean measure

At retained distinct clocks `u_0 < ... < u_{K-1}`, let `z_k` be the cached normalized phase output. Cell `C_k=[u_k,u_{k+1})` is valid only if both endpoints are feature-valid, adjacent after collapse with no invalid location between them, and strictly ordered. Define the original signed increments and rates without canonical orientation:

\[
\delta_k=\operatorname{Arg}(z_{k+1}\overline z_k)\in(-\pi,\pi],\qquad
r_k=\delta_k/(u_{k+1}-u_k),\qquad
\tilde r(s)=r_k\quad(s\in C_k).
\]

No median sign, forward-direction convention, or clean/warp sign canonicalization exists in training or inference.

### 4.2 Sole signed overlap integral

Let `tau` map target clock to clean source clock. For target interval `[q_j,q_{j+1})`, set `a=tau(q_j)`, `b=tau(q_{j+1})`, and `s=sign(b-a)`. Define

\[
\ell_{jk}=s\,\left|[\min(a,b),\max(a,b))\cap C_k\right|,
\qquad
I_j^\tau=\sum_k\ell_{jk}r_k.
\]

This float64 signed-overlap sum is the only clean/warp target and is cast to float32 only for the residual. Half-open cells assign a breakpoint to the cell on its right and prevent double mass. Forward mappings use positive overlaps; globally decreasing reversal uses negative overlaps and therefore yields the natural negative integral; a pause has `a=b` and exactly zero integral when its mapped point lies in valid support. Any uncovered nonzero mapped length, invalid clean cell, invalid target endpoint, or out-of-range support invalidates the entire target interval; partial support is never renormalized.

The warped branch retains its original signed increment

\[
\delta_j^\tau=\operatorname{Arg}(z_{j+1}^\tau\overline z_j^\tau),
\]

and the only treatment residual and loss are

\[
e_j=\operatorname{wrap}\!\left(\delta_j^\tau-\operatorname{sg}(I_j^\tau)\right),\qquad
\operatorname{wrap}(x)=\operatorname{atan2}(\sin x,\cos x),
\]

\[
\mathcal L_{\mathrm{WI}}=\frac{\sum_jv_j\rho_{0.10}(e_j)}{\sum_jv_j}.
\]

The clean integral is a stop-gradient teacher; the warped branch is the student. Shared weights also receive the shared PAMS loss. The signed law is globally equivariant to `delta → -delta` on both branches, and the downstream `abs(delta)` decoder is invariant to that global sign. No alternate derivative target or canonical orientation branch exists.

Set `v_j=0` when `abs(I_j^tau) >= pi-0.05pi`, when any required clean principal increment has `abs(delta_k) >= pi-0.05pi`, or when support is invalid. Never clip or re-unwrap targets. Require at least 32 valid target intervals per window and 80% valid target-interval coverage per eligible track; otherwise invoke K5/K7.

### 4.3 Matched objectives

\[
\mathcal L_{\mathrm{control}}=\mathcal L_{\mathrm{PAMS}},\qquad
\mathcal L_{\mathrm{treatment}}=\mathcal L_{\mathrm{PAMS}}+\mathcal L_{\mathrm{WI}}.
\]

Both arms use identical shards, selector outputs, clean/warp pairs, warp seeds, architecture, initializations, batches, optimizer, steps, recurrent execution, NOLA, decoder, and evaluation. The only treatment deletion is `L_WI`.

## 5. Strictly Positive Interval NOLA and One Decode

A 64-sample half-open window contains 63 intervals `i=0,...,62`. Its deterministic interval weight is

\[
w_i=\max\!\left(10^{-3},\sin^2\!\left(\pi\frac{i+0.5}{63}\right)\right)>0.
\]

Stride is 32 retained samples. Starts are `0,32,...` while a full window fits; append `max(K-64,0)` if absent. For `K<64`, use start 0 and right-pad samples/masks; padding creates no interval or mass. Windows read the single cached identity sequence. For every valid global interval `j`,

\[
D_j=\sum_{b\ni j}v_j^{(b)}w_{j-b},\qquad
\bar m_j=\frac{\sum_{b\ni j}v_j^{(b)}w_{j-b}\,g_j|\delta_j^{(b)}|/(2\pi)}{D_j}.
\]

Assert finite `D_j >= 1e-3` before division. There is no epsilon rescue or zero-weight boundary. Invalid intervals contribute neither numerator nor denominator. Decode once per supplied identity, `hat C_p=sum_j bar m_j`; round-half-to-even is secondary and occurs only after this sum. Because cached increments are read-only, overlap cannot duplicate recurrence updates. Grid-origin offsets 0, 8, 16, and 24 must conserve synthetic phase mass within `1e-6` relative error.

## 6. Frozen Numerical Implementation Table

| Interface | Frozen value |
|---|---|
| Input | `[B,64,17,3]` float32 pose + `[B,64,17]` bool joint mask; 68 model channels |
| Clock/integral | int64 `sampled_frame_indices`; float64 overlaps/integrals |
| Joint validity | confidence `>=0.20`; at least 8/17 joints per valid frame |
| Eligibility | unambiguous association; pose coverage `>=0.80`; collapsed feature coverage `>=0.60`; never label-filtered; all 9 components nonempty |
| Duplicate agreement | masks exact; normalized xy max `<=1e-5`; confidence max `<=1e-6`; disagreement invalidates clock |
| Encoder/head | Conv1d 68→128 k5 p2; Conv1d 128→128 k5 p2; GRU 128; linear 128→2; 225,026 parameters |
| Recurrence | once per full collapsed identity; zero at identity start; invalid-cell hold; cached states; overlap windows read-only |
| Phase normalization | `z=u/(||u||_2+1e-8)` |
| Mask order | duplicate agreement → collapse/sort → joint/frame validity → adjacent cell → warp support → alias → NOLA |
| Pseudo-cycle selector | nested 64/128; FFT 256; candidates 4–`min(128,floor(span/2))`; >=16 pairs; ACF >=0.25; tie 0.02 |
| Weak views | coordinate jitter `sigma=0.01`, clip `0.03`; 10% joint dropout; retain >=8 joints; clock unchanged |
| Training selector gate | 1,000 fixtures, >=250 symmetric; total >=95%; half/double <=2%; symmetric half/double <=5%; real view/nested agreement 10%; margin 0.10 |
| Evaluator harmonic gate | post-freeze only; `h={0.5,1,2}`; relative tolerance 10%; video-first; `H1>=0.90`, other each `<=0.05`; no correction |
| Windows/NOLA | 64 samples/63 intervals; stride 32; floor `1e-3`; deterministic final/right padding; denominator `>=1e-3` |
| Warp family | 3 affine segments; breakpoints uniform `[0.20,0.40]`, `[0.60,0.80]`; raw slopes log-uniform `[0.5,1.5]`, endpoint-normalized; accept final `[0.4,2.0]` within 128 draws |
| Pause/reversal | 20% set middle raw slope to 0 then normalize remaining slopes; reversal is global endpoint swap, never local direction mixing |
| Warp loss | circular Huber beta 0.10 rad; alias margin `0.05pi`; min 32 valid intervals/window and 80% track coverage |
| Optimizer | AdamW lr `3e-4`, weight decay `1e-4`, batch 64 windows, clip 1.0, 20,000 steps, no metric early stop |
| Seeds | 20270815, 20270816, 20270817; no replacement |
| Bootstrap | 9 asserted nonempty development components; B=10,000; PCG64 seed 20270815; linear central percentiles |
| Hard budget | 59 total training jobs; 552 GPU-hours absolute maximum |

Warp and diagnostic schedules are serialized before evaluator access. Engineering tolerances are invariants and no development metric selects them.

## 7. Gates and Training Plan

### Gate 0 — isolation and frozen population

Create split shards/vault, opaque join table, source-component manifest, count-blind eligibility manifest, checksums, forbidden-key/path receipts, and the hashed eligible video/person/component table. Assert all original nine development components retain at least one eligible video/person. Isolation failure invokes K6; an empty component invokes K7 before any evaluator label opens.

### Gate 1 — deterministic numerical fixtures

Run duplicate collapse, invalid-cell interpolation, signed overlap, wrap, pause, reversal, alias, single-commit recurrence, invalid-state hold, read-only overlap, final window, positive denominator, grid origin, one decode, normalized AvgMAE, and bootstrap fixtures. These are not efficacy results.

The frozen two-component scorer fixture uses one video per component. Component A has `C=[2,4]`, control `[3,2]`, treatment `[2,3]`, giving normalized video errors `0.5` and `0.125`. Component B has `C=[5]`, control `[7]`, treatment `[6]`, giving `0.4` and `0.2`. With both videos once, `M_c=0.45`, `M_t=0.1625`, `D=0.2875`, and `R=0.6388888888888888`. Bootstrap component draws `[A,A]`, `[A,B]`, and `[B,B]` must yield `D=0.375`, `0.2875`, and `0.2`; three identical fixture seeds leave these paired values unchanged. Byte-for-byte expected arrays are frozen before evaluator use.

### Gate 2 — pseudo-cycle stability

Run the 1,000 seeded fixtures, including the symmetric subset, and every eligible count-blind training track through Section 3.2. Failure invokes K5 before full training. The gate licenses a stable pseudo-cycle interface only.

### Gate 3 — sanity

At most three training-only jobs verify finite gradients, selector coverage, loss decrease, identity state isolation, one recurrent commit per source clock, deterministic receipts, and prediction schema. No count or evaluator period is read.

### Gate 4 — staged decision runs and prediction freeze

Run fixed seeds without replacement/cherry-picking. At each reached stage, freeze commits, configs, environments, raw per-person predictions, pseudo-cycle selections, and phase traces for every predeclared arm before any evaluator join.

### Gate 5 — untouched evaluator gates

After a stage's complete prediction set is frozen, first execute the evaluator-only harmonic gate without corrections. If it passes, execute the exact normalized AvgMAE/bootstrap scorer once. Never tune from development outputs.

## 8. Exact Primary Estimator and Bootstrap

Gate 0 guarantees exactly nine nonempty frozen source components. Before any division, the evaluator asserts finite `C_{v,p} > 0` for every eligible person; violation aborts evaluation under K6, with no epsilon or alternate metric.

For seed `s`, arm `a`, and video `v`, compute per-person relative absolute error, then video-first AvgMAE:

\[
A_{a,s,v}=|P_v|^{-1}\sum_{p\in P_v}\frac{|\hat C_{a,s,v,p}-C_{v,p}|}{C_{v,p}},
\qquad
M_{a,s}=|V|^{-1}\sum_{v\in V}A_{a,s,v}.
\]

Aggregate training seeds first:

\[
\bar M_c=\tfrac13\sum_sM_{c,s},\quad
\bar M_t=\tfrac13\sum_sM_{t,s},\quad
D=\bar M_c-\bar M_t,\quad
R=D/\bar M_c.
\]

If `bar M_c=0`, the relative clause is undefined and K1 fails; no epsilon or substitute endpoint is used.

For 10,000 draws using NumPy `Generator(PCG64(20270815))`, sample nine component indices with replacement. A component selected `m` times contributes all its videos and people with multiplicity `m`. Apply the identical draw to both arms and all seeds. Within each seed/arm, recompute the per-person-normalized video errors and multiplicity-weighted video mean; average the three paired seed effects to obtain absolute `D^(b)`. The CI is `[quantile(D*,0.025,method='linear'), quantile(D*,0.975,method='linear')]`. It covers source-component sampling only; report seed-level effects and sample SD separately.

K1 passes only if the unbootstrapped point has `R>=0.05` and the absolute-`D` CI lower endpoint is strictly positive. AvgOBO and supplied-track Period-mAP/AP50/AP75 on `[0,100]` are secondary and cannot substitute.

## 9. Staged Comparators and Hard Budget

A mechanism control uses the same 225,026-parameter interface; a faithful supplied-track adaptation preserves its cited defining objective/head and reports actual parameters and provenance.

| Stage | Condition | Three-seed full jobs | Category / purpose |
|---|---|---:|---|
| A | treatment + identical augmentation-only | 6 | primary paired endpoint |
| B | trained timestamp/no-pose model + trained nuisance-only model | 6 | K4 decision controls; run only if A passes K1 |
| C | direct signed-frequency regression; global-warp-only; unsigned-reversal ablation | 9 | same-backbone mechanism falsification; run only if A/B survive K4 |
| D | PAMS; Track-PAMS; SimPer-style; CycleCL-style; DeepPhase-style; generic time-equivariant | 18 | faithful/same-backbone closest priors; run only if A–C survive K2/K3 |
| E | masked geometric TSSM control | 3 | optional control-only if frozen matrix requires it; never fallback/contribution |

The timestamp/no-pose model receives only `sampled_frame_indices`, `source_length`, masks, and warp metadata permitted to the training arm, with all pose channels zero. The nuisance-only model receives the frozen crop/resampler/augmentation nuisance fields under audit but no pose. Each uses the same architecture capacity where representable, the same 20,000-step cap, and exactly three fixed seeds; neither receives tuning jobs. Matched-time pose shuffle, hidden absolute time, randomized crop origins, and unseen-resampler tests are deterministic evaluation corruptions/receipts, not extra trained models.

Stages A–E contain at most 42 decision-bearing full jobs. The seven learned comparator families in D/E receive at most two training-only 5,000-step tuning jobs each, never evaluator labels, adding 14 jobs. Together with three sanity jobs, the absolute maximum is **59 jobs**. A full job is capped at 12 GPU-hours, each tuning job at 3, and all sanity jobs together at 6, so the hard ceiling is **552 GPU-hours = 42×12 + 14×3 + 6**. No over-budget continuation is permitted.

If A fails, stop after at most 9 jobs/78 GPU-hours and save at least 50 jobs/474 GPU-hours. If B fails K4, stop after at most 15 jobs/150 GPU-hours and save at least 44 jobs/402 GPU-hours. If C fails K2/K3, stop after at most 24 jobs/258 GPU-hours and save at least 35 jobs/294 GPU-hours. Later stages stop on the first applicable K-rule. Comparator provenance, mappings, counts, schedules, and receipts are mandatory.

For learned comparators, use the same paired nine-component bootstrap. A comparator is separated only if the lower endpoint of `M_comparator-M_treatment` is strictly positive; otherwise it matches within uncertainty and K2 applies.

## 10. Three Visible Validation Blocks

### Block 1 — primary paired law versus augmentation

One table reports three seed-paired piecewise-warp normalized video-first AvgMAE values, mean/SD, `D`, `R`, and the exact nine-component CI. Clean AvgMAE and secondary metrics are labeled diagnostics. This block alone decides K1.

### Block 2 — staged mechanism, shortcut, and closest-prior falsification

Report the trained timestamp/no-pose and nuisance-only K4 decisions, then the deletion/direct-frequency/global-warp/unsigned-reversal checks, then only the reached closest-prior stage. Timestamp/no-pose or nuisance-only retention above 20% of a positive treatment gain, matched-time shuffle preservation, or unseen-resampler erasure invokes K4. Detailed fixtures remain receipts/supplement. This remains one visible block despite its complete execution ledger.

### Block 3 — identity isolation

With a frozen backbone, warp exactly one supplied identity while untouched track tensors and clocks remain byte-identical. Compare private state with shared-scene state and shuffled-key state. Report untouched-person phase-response change, continuous count-mass change, and rounded count change with component intervals. Stateless/reset/swap variants remain debugging receipts unless contamination appears. This is a diagnostic, never private-memory novelty and never a rescue for Block 1.

## 11. Failure Modes and Frozen K1–K9

- **K1 — primary effect/uncertainty:** Survive only if point relative piecewise-warp normalized video-first AvgMAE improvement is at least 5% and the paired nine-component absolute-`D` CI lower endpoint is greater than zero. `bar M_c=0` fails.
- **K2 — closest prior:** If PAMS/Track-PAMS, SimPer-style, CycleCL-style, DeepPhase-style, or generic time-equivariant learning matches/exceeds treatment within the frozen rule, kill the residual claim. Private state/TSSM cannot rescue it.
- **K3 — objective redundancy:** If deleting `L_WI`, direct signed-frequency regression, global resampling only, or replacing reversal's signed overlap with unsigned overlap matches treatment, conclude the local signed objective is redundant.
- **K4 — shortcut/artifact:** If a trained timestamp/no-pose or nuisance-only model retains more than 20% of a positive gain, an unseen resampler erases it, or matched-time shuffling preserves it, reject as shortcut/artifact.
- **K5 — pseudo-cycle or execution failure:** If training-only selector fixtures/stability, the frozen evaluator harmonic gate, support/alias coverage, recurrence single-commit receipts, or NOLA mass conservation fails, kill the phase-mass claim. Do not correct harmonics or add modules.
- **K6 — integrity:** If feature/vault separation, positive-count assertion, source isolation, permissions, forbidden keys/paths, hashes, frozen predictions, or untouched evaluation cannot be shown, stop with no claim.
- **K7 — population/clock/missingness:** If any original development component is empty after count-blind eligibility, or duplicate conflicts, missing pose, timing, or alias masks invalidate the estimand, rebuild the cache or restrict work to engineering diagnostics.
- **K8 — novelty collision:** Obtain and independently inspect the complete TWCRAC method before claim freeze. No such full-method receipt exists now; K8 is **OPEN**. If TWCRAC contains the same discrete local warp-phase law or equivalent contribution, kill or materially re-anchor novelty and rerun novelty review. Abstract-only evidence cannot close K8.
- **K9 — forbidden fallback:** If the primary fails and masked geometric TSSM is proposed as an automatic replacement, stop. It remains a control absent a new anchor and fresh novelty review.

## 12. Experiment Handoff and Current Status

**Required artifacts:** feature shards, evaluator vault, eligibility/component/join manifests, forbidden-key/path receipts, pseudo-cycle/harmonic fixtures, post-freeze evaluator harmonic receipt, recurrence single-commit trace, NOLA/overlap/normalized-scorer/bootstrap fixtures, three-seed run receipts, raw per-person predictions, phase traces, comparator provenance, and an eventual independent TWCRAC full-method adjudication.

**Run order:** Gate 0 isolation/nonempty components → Gate 1 numerical fixtures → Gate 2 pseudo-cycle stability → Gate 3 sanity → Stage A primary → post-freeze harmonic gate and K1 → Stage B K4 trained controls → Stage C mechanism checks → Stage D priors → optional Stage E control → identity diagnostic → independent TWCRAC K8 review. Stop at the first K-rule.

**Forbidden evidence:** v46/v62/v63 efficacy, sealed test data, predicted-track claims, official MultiRep main-result language, published end-to-end numbers as same-protocol rankings, post-hoc harmonic correction, and any result without a complete frozen receipt.

**Current status:** Round-2 revised proposal only; not implemented; no eligible results; same-family provisional. Training is blocked until Gates 0–2 pass. Claim freeze remains blocked until reached experiment/audit/result-to-claim gates pass and K8 receives an independent full-method receipt; K8 is presently open.
