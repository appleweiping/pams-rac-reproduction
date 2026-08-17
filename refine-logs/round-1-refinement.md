# Round 1 Research-Refine Refinement: WARP-PHASE

> **PROSPECTIVE PILOT**  
> **NO ELIGIBLE RESULTS**  
> **SAME-FAMILY PROVISIONAL**

**Executor:** proposal executor  
**Review being answered:** `refine-logs/round-1-review.md`  
**Reviewer:** GPT-5.6-Sol (`/root/research_refine_reviewer`)  
**Round-1 score/verdict:** 6.9/10, `REVISE`  
**Permitted protocol name:** **GT-bbox-assisted AlphaPose supplied-track partial-cache development pilot**

This document is a complete prospective replacement proposal, not an incremental method patch. No canonical implementation exists, no eligible run has been made, and no numerical efficacy statement is licensed. Historical v46, v62, and v63 artifacts remain audit inputs only.

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

- **PASS — question and contrast:** The treatment is still the explicit signed local warp law and the primary control is still identical augmentation exposure without that law.
- **PASS — population/protocol:** Only checksum-frozen, source-disjoint train/development entries from the partial GT-bbox-assisted AlphaPose supplied-track cache are in scope. No predicted-track or sealed-test claim is introduced.
- **PASS — unit/estimand:** The analysis stays video-first; the nine frozen development source components, not people or videos, are bootstrap clusters.
- **PASS — supervision:** The model process reads no count, density, period, cycle-boundary, object-mapping, source-identity, or evaluator field. The target-free fundamental selector uses only permitted pose tracks and the permitted source-frame clock.
- **PASS — endpoint:** K1 remains the 5% relative AvgMAE reduction **and** positive paired-component CI rule. Secondary metrics cannot rescue failure.
- **PASS — scope:** No router, learned expert, new TSSM, private-memory novelty, scene-total objective, end-to-end claim, or historical-result claim has been added.
- **PASS — evidence status:** Every efficacy statement remains prospective; eligible method results remain zero.

## Simplicity Check

- **One proposed mechanism:** the treatment adds one discrete warp-integral equivariance objective. It does not claim the continuous chain rule as new.
- **One established shared phase base:** treatment and control share a single Track-PAMS/PAMS-TCC-style target-free masked ACF/FFT fundamental anchor and cycle-positive phase backbone. PAMS/TCC-style periodic alignment and spectral/autocorrelation selection are disclosed as established sources, not contributions.
- **Removed bespoke stack:** the prior CycleCL-style base contrast, learned activity classifier, phase-conditioned velocity reconstruction, variance/covariance auxiliary, and learned overlap-consistency loss are deleted from the primary system.
- **Fixed decoder:** strictly positive interval NOLA and one per-identity sum are deterministic engineering operators.
- **Three visible validation blocks only:** primary pair; staged mechanism/closest-prior falsification; identity-isolation diagnostic. Integrity and numerical tests are receipts, not extra paper claims.
- **Fail rather than expand:** if the label-free selector cannot identify one physical cycle on synthetic and count-blind training gates, K5 kills the full run. No post-hoc calibration, router, expert bank, reconstruction fallback, or TSSM promotion is allowed.

## Changes Made

1. **Unique discrete target:** Deleted the approximate derivative branch and all implementation alternatives. The only clean/warp target is the stop-gradient oriented signed overlap integral of a duplicate-collapsed, piecewise-constant clean rate. Wrap, half-open boundaries, invalid support, alias, forward warp, reversal, and pause are executable below.
2. **Identified phase unit or pre-run kill:** Replaced the bespoke base-loss bundle with one shared Track-PAMS/PAMS-TCC-style target-free masked ACF/FFT selector plus cycle positives. It fixes one selected fundamental cycle to one \(2\pi\) winding. A numeric harmonic/nested-window/view gate precedes all full runs; failure invokes K5.
3. **Strictly positive NOLA:** Replaced endpoint-zero Hann weighting with interval-centered sine-squared weights floored at \(10^{-3}\), stride 32, a deterministic final window, right-padding rules, and a hard denominator assertion.
4. **Numerical freeze:** Added shapes, thresholds, dimensions, optimizer values, warp distributions, robust-loss constants, numeric tolerances, masks, and an exact 225,026-parameter treatment/control architecture table.
5. **Count-blind population:** Excludes all 13 train and 8 development slots marked association-ambiguous, freezes 0.80 GT-assisted pose coverage and feature-validity rules before count/density access, and freezes duplicate-clock agreement independently of counts.
6. **Executable inference:** Defines the nine-component, \(B=10{,}000\), seed-20270815 paired bootstrap with component multiplicity, video-first recomputation, three-seed aggregation, central percentile CI on absolute \(D\), point relative threshold, and \(\bar M_c=0\) failure.
7. **Staged finite experiment plan:** Freezes a maximum of 53 training jobs and 480 GPU-hours, with explicit early-stop savings and separate same-backbone versus faithful-adaptation comparator categories.
8. **Three visible evidence blocks:** Compresses validation to the primary endpoint, staged falsification, and identity isolation; all other checks are implementation/audit receipts.
9. **Contribution language corrected:** The possible contribution is a **discrete warp-integral equivariance objective for supplied-track MRAC**. The chain rule, phase, temporal equivariance, PAMS/TCC, ACF/FFT, private state, and NOLA are not claimed as new.
10. **Novelty collision retained:** Complete TWCRAC method inspection remains mandatory K8 before any claim freeze; primary failure cannot be rescued by private state or masked TSSM.

---

# Revised Proposal: WARP-PHASE — Discrete Warp-Integral Equivariance for Supplied-Track MRAC

> **PROSPECTIVE PILOT**  
> **NO ELIGIBLE RESULTS**  
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

This is a preregistered, kill-oriented partial-cache pilot. It is not a manuscript method freeze. The method has not been implemented, eligible results are zero, and no expected direction is reported as evidence.

**Thesis.** A clean identity track supplies a discrete signed phase measure. Under a known target-to-source time map, a warped target interval should equal the oriented integral of that clean measure over the mapped source interval. Treatment and control see the same warps; treatment alone is optimized against that exact integral.

**Possible future contribution.** If and only if the full protocol passes, the narrow claim is: **a discrete warp-integral equivariance objective for signed phase increments in supplied-track MRAC**. The ordinary continuous chain rule is motivation, not novelty. ACF/FFT selection, PAMS/TCC-style phase learning, pose SSL, local windows, temporal equivariance, private state, NOLA, and per-person decoding are established or generic components and are explicitly non-contributions.

**Claim ceiling.** The wording remains “we study whether” on a GT-bbox-assisted AlphaPose supplied-track partial-cache development pilot. It is not end-to-end, annotation-free, predicted-track, state-of-the-art, or deployment evidence. PAMS already reports localized internal-tempo perturbation; this diagnostic is an extension, not a first. DeepPhase already occupies unsupervised motion phase. Full TWCRAC overlap is unresolved and remains K8.

## 2. Count-Blind Data Contract and Eligibility

### 2.1 Physical isolation

A trusted offline packer may read the canonical v44 train/validation pickles once. It emits feature-only non-executable shards, a separately permissioned evaluator vault, a split/component manifest, an eligibility manifest, per-file/per-sample SHA-256 receipts, and a forbidden-key receipt. The training process has no permission to the original pickle or vault and rejects non-whitelisted fields before deserializing payloads.

Model-process fields are only `motion`, `person_mask`, `frame_mask`, `sampled_frame_indices`, `source_length`, an opaque sample key, and a local supplied-slot index. Source IDs and hashes are split/audit-only. Counts, densities, periods, boundaries, raw boxes, annotation names, object IDs/mappings, association structures, and `gt_object_pose_coverage` never enter a model tensor, objective, selector, tuning decision, or checkpoint.

### 2.2 Frozen count-blind primary eligibility

Eligibility is computed and checksum-frozen before the evaluator process opens `count_gt` or `density_gt`; it is never revised because of a count or metric.

1. Exclude every slot marked ambiguous by the GT-assisted association audit: all 13 training and all 8 development ambiguous slots are excluded. Require a one-to-one supplied-slot/object association in the audit manifest.
2. Require frozen GT-assisted pose coverage \(\ge 0.80\). This audit-only scalar is used by the trusted packer solely to freeze inclusion and is then absent from model shards.
3. After duplicate-clock collapse, require at least 64 retained distinct clock locations, at least 63 valid adjacent cells, and at least 0.60 feature-valid clock coverage. A frame is feature-valid when at least 8 of 17 joints have AlphaPose confidence \(\ge0.20\).
4. Do not filter on count magnitude, density, period, action, predicted difficulty, selector success, development error, or harmonic agreement. The manifest reports each exclusion and its count-blind reason by split/component.
5. The full-run selector must produce an answer for every eligible slot. Selector failure is a global pre-full-run K5 failure, not a reason to silently drop a hard slot.

### 2.3 Duplicate-clock collapse

For each integer source clock \(t\), group all samples with `sampled_frame_indices == t`. Retain the first sample only when every repeated member has identical frame/joint masks, maximum absolute normalized-\(x,y\) difference \(\le10^{-5}\), and maximum confidence difference \(\le10^{-6}\). Otherwise mark clock \(t\) invalid. No averaging resolves disagreement. Sort retained clocks strictly increasingly. Interpolation is allowed only between adjacent retained, feature-valid clocks; it never crosses an invalid clock or missing cell. This same rule is used by eligibility, the base selector, clean-rate construction, warp interpolation, and diagnostics.

## 3. Shared Established Phase Backbone

Treatment and augmentation-only control use the same **Track-PAMS/PAMS-TCC-style** label-restricted phase backbone. Its target-free masked autocorrelation/spectrum anchor and cycle-positive temporal correspondence are derived from the established PAMS family and standard ACF/FFT periodic analysis. They are not presented as new. The adaptation to supplied tracks is frozen identically in both arms.

### 3.1 Label-free fundamental selector

On each collapsed clean track, root-center valid joints at the hip midpoint, divide coordinates by the median valid shoulder-to-hip scale (floor \(10^{-3}\)), and form masked first differences on adjacent valid cells. On integer source-clock support, interpolate only inside valid adjacent cells. For nested clean windows of 64 and 128 retained clocks and two weak spatial views, compute:

\[
R(P)=\frac{\sum_t m_t m_{t+P}\langle v_t,v_{t+P}\rangle}
{\sqrt{\sum_t m_t\lVert v_t\rVert^2\sum_t m_{t+P}\lVert v_{t+P}\rVert^2}+10^{-8}}
\]

for integer candidate periods \(P\in[4,P_{\max}]\), where \(P_{\max}=\min(128,\lfloor(q_{\mathrm{last}}-q_{\mathrm{first}})/2\rfloor)\), having at least 16 valid pairs, and a 256-point masked FFT periodogram \(E(f)\). Candidate score is

\[
S(P)=0.5\,R(P)+0.5\,\frac{E(1/P)+0.5E(2/P)+0.25E(3/P)}{1.75\sum_{f>0}E(f)+10^{-8}}.
\]

Local maxima with \(R(P)\ge0.25\) are candidates. Select the smallest period among candidates within 0.02 of the maximum score only after the harmonic gate below; ties then use smaller \(P\). No evaluator period, count, density, action label, or development metric is used.

The selected \(\hat P\) fixes the unit: one \(\hat P\)-cycle is one \(2\pi\) winding. PAMS-TCC cycle positives are pairs \((t,t+\hat P)\) with valid support; within-cycle correspondence uses pseudo-angle \(2\pi(q_b-q_a)/\hat P\) for separations up to \(\hat P\). With \(\mathcal A\) the valid anchored pairs, the one base loss is

\[
\mathcal L_{\mathrm{PAMS}}=|\mathcal A|^{-1}\sum_{(a,b)\in\mathcal A}
\rho_{0.10}\!\left(\operatorname{wrap}\!\left[\operatorname{Arg}(z_b\overline z_a)-2\pi(q_b-q_a)/\hat P\right]\right).
\]

It is a circular Huber correspondence loss with \(\beta=0.10\) rad. Weak spatial views use independent normalized-coordinate Gaussian jitter with \(\sigma=0.01\), clipped at \(0.03\), plus 10% joint dropout while retaining at least 8 joints; clocks are unchanged. There is no learned activity head, reconstruction decoder, covariance/variance loss, or learned seam loss. A fixed selector confidence \(g_j\in\{0,1\}\) is one only when its parent windows pass all selector gates; it is not trained.

### 3.2 Harmonic and identifiability kill gate

Before any decision-bearing run:

- On synthetic pose-like periodic signals with known one-cycle units, piecewise warps, reversal, pauses, 40% missing frames, 30% missing joints, and injected second/third harmonics, selector fundamental accuracy must be \(\ge95\%\) and half/double errors must each be \(\le2\%\) over 1,000 fixtures generated with seed 20270815.
- On all eligible count-blind training tracks, selected \(P\) must agree within \(\pm10\%\) across both spatial views and nested windows for at least 90% of parent windows on every track; the score margin over \(P/2\) and \(2P\), when in range, must be \(\ge0.10\).
- Reversal must preserve \(\hat P\) within \(\pm10\%\), and the canonical forward orientation defined by the stop-gradient sign of the median valid clean increment must be nonzero and stable in at least 95% of windows on every track.

These gates are deliberately severe. They do not use evaluator labels and they do not pretend that recurrence alone proves semantic cycle identity. If any eligible track fails, or the selector cannot state one physical-cycle unit without evaluator calibration, K5 kills the phase-mass decoder before the full run. No development-period calibration or post-hoc half/double correction is permitted.

### 3.3 Shared network and exact parameter count

Each pose step supplies 51 `x,y,confidence` values plus 17 joint-mask bits, for 68 channels. Missing coordinates/confidences are zero only after the 17 masks are concatenated. The shared network is temporal Conv1d \(68\to128\), kernel 5; Conv1d \(128\to128\), kernel 5; one GRU \(128\to128\); and a linear phase head \(128\to2\), followed by \(\ell_2\) normalization. It has exactly 225,026 trainable parameters: 43,648 + 82,048 + 99,072 + 258. No other trainable head exists. Recurrent state and all accumulators are keyed by local supplied-slot identity; parameters remain shared across identities.

## 4. The Unique Discrete Warp-Integral Objective

### 4.1 Clean signed measure

At retained distinct clocks \(u_0<\cdots<u_{K-1}\), the normalized phase output is \(z_k\in\mathbb S^1\). A clean cell \(C_k=[u_k,u_{k+1})\) is valid only if both endpoints are feature-valid, the clocks are adjacent after collapse with no invalid location between them, and \(u_{k+1}>u_k\). Define

\[
\delta_k=\operatorname{Arg}(z_{k+1}\overline z_k)\in(-\pi,\pi],\qquad
r_k=\delta_k/(u_{k+1}-u_k),\qquad
\tilde r(s)=r_k\;\text{for }s\in C_k.
\]

The canonical forward orientation multiplies all increments by the stop-gradient sign of their median on selector-confident cells. An unstable or zero sign fails K5.

### 4.2 Oriented signed overlap integral

Let \(\tau\) map target clock to clean source clock. For target interval \([q_j,q_{j+1})\), set \(a=\tau(q_j)\), \(b=\tau(q_{j+1})\), \(s=\operatorname{sign}(b-a)\), and

\[
\ell_{jk}=s\,\left|[\min(a,b),\max(a,b))\cap C_k\right|,
\qquad
I_j^\tau=\sum_k \ell_{jk}r_k.
\]

This signed-overlap sum is the sole clean/warp target. It is computed in float64 and cast to float32 only for the residual. The half-open convention assigns a breakpoint to the cell on its right and prevents double mass. For a forward warp, overlaps are positive. For a globally decreasing reversal, they are negative. A pause has \(a=b\) and exactly \(I_j^\tau=0\), provided the mapped point lies in valid clean support. A target interval is wholly invalid if any nonzero-length part of its mapped support is uncovered, crosses an invalid clean cell, uses an invalid target endpoint, or lies outside `[u_0,u_{K-1}]`; partial support is never renormalized.

The only treatment residual and loss are

\[
e_j=\operatorname{wrap}\!\left(\delta_j^\tau-\operatorname{sg}(I_j^\tau)\right),
\quad
\operatorname{wrap}(x)=\operatorname{atan2}(\sin x,\cos x),
\quad
\mathcal L_{\mathrm{WI}}=\frac{\sum_j v_j\rho_{0.10}(e_j)}{\sum_jv_j},
\]

where \(\rho_{0.10}\) is Huber loss with transition 0.10 rad and \(v_j\) is the complete validity/alias mask. Stop-gradient is applied to the entire clean integral, including the clean rates and orientation sign. The clean branch is always the teacher and the warped branch is always the student; roles are not alternated. Shared weights still receive gradients through the warped branch and the shared PAMS-TCC base. This one-way convention also remains defined for non-invertible pauses.

Alias risk is fail-closed: set \(v_j=0\) when \(|I_j^\tau|\ge\pi-0.05\pi\), when any needed clean principal increment has \(|\delta_k|\ge\pi-0.05\pi\), or when harmonic support is ambiguous. Targets are never clipped or re-unwrapped. Coverage is reported by track/component; loss requires at least 32 valid target intervals per window and 80% valid target-interval coverage per eligible track, otherwise K5/K7 stops the full run.

### 4.3 Treatment/control objectives

The shared base objective is the single frozen PAMS-TCC circular correspondence loss \(\mathcal L_{\mathrm{PAMS}}\). The two arms are

\[
\mathcal L_{\mathrm{control}}=\mathcal L_{\mathrm{PAMS}},\qquad
\mathcal L_{\mathrm{treatment}}=\mathcal L_{\mathrm{PAMS}}+\mathcal L_{\mathrm{WI}}.
\]

Both arms use the same eligible shards, selector decisions, clean/warp pairs, warp seeds, architecture, initialization distribution, batches, optimizer, steps, NOLA, decoder, and evaluation. No other auxiliary objective exists. Thus the only treatment deletion is \(\mathcal L_{\mathrm{WI}}\).

## 5. Strictly Positive Interval NOLA and One Decode

A 64-sample half-open window contains 63 intervals indexed \(i=0,\ldots,62\). Its deterministic interval weight is

\[
w_i=\max\left(10^{-3},\sin^2\!\left(\pi\frac{i+0.5}{63}\right)\right)>0.
\]

Window stride is 32 retained samples. Starts are `0,32,...` while a full 64-sample window fits; append `max(K-64,0)` as the final start if absent. For \(K<64\), use start 0 and right-pad samples/masks to 64; padding creates no interval and no mass. Windows and cells are half-open. For every valid global interval \(j\), aggregate the window prediction \(m_j^{(b)}=g_j|\delta_j^{(b)}|/(2\pi)\) by

\[
D_j=\sum_{b\ni j}v_j^{(b)}w_{j-b},\qquad
\bar m_j=\frac{\sum_{b\ni j}v_j^{(b)}w_{j-b}m_j^{(b)}}{D_j}.
\]

Before division, assert finite \(D_j\ge10^{-3}\) for every eligible valid interval. There is no epsilon rescue and no zero-weight boundary. Invalid intervals contribute neither numerator nor denominator. The continuous count is decoded once per supplied identity, \(\hat C_p=\sum_j\bar m_j\). A rounded value is secondary and uses round-half-to-even only after this sum. Grid-origin tests at offsets 0, 8, 16, and 24 must conserve synthetic phase mass within \(10^{-6}\) relative error before training.

## 6. Frozen Numerical Implementation Table

| Interface | Frozen value |
|---|---|
| Input | `[B,64,17,3]` float32 pose + `[B,64,17]` bool joint mask; 68 model channels |
| Clock/integral | int64 `sampled_frame_indices`; float64 overlaps/integrals |
| Joint validity | confidence `>=0.20`; at least 8/17 joints per valid frame |
| Slot eligibility | unambiguous one-to-one GT-assisted association; pose coverage `>=0.80`; collapsed feature coverage `>=0.60`; never count-filtered |
| Duplicate agreement | masks exact; normalized xy max error `<=1e-5`; confidence max error `<=1e-6`; disagreement invalidates clock |
| Encoder/head | Conv1d 68→128 k5; Conv1d 128→128 k5; GRU 128; linear 128→2; 225,026 trainable parameters |
| Phase normalization | `z=u/(||u||_2+1e-8)` |
| Mask order | duplicate agreement → collapse/sort → joint confidence → frame validity → adjacent-cell validity → warp-support completeness → alias mask → NOLA validity |
| Base selector | nested lengths 64/128; FFT 256; integer periods 4–`min(128,floor(clock_span/2))`; >=16 pairs; ACF >=0.25; score tie 0.02 |
| Weak spatial views | coordinate jitter `σ=0.01`, clipped `0.03`; 10% joint dropout; retain >=8 joints; clock unchanged |
| Harmonic gate | view/nested agreement ±10%; margin >=0.10; synthetic accuracy >=95%; half/double <=2% |
| Windows/NOLA | length 64 samples / 63 intervals; stride 32; weight floor `1e-3`; deterministic final/right-padded window; denominator `>=1e-3` |
| Warp family | 3 affine segments; two breakpoints sampled uniformly from `[0.20,0.40]` and `[0.60,0.80]`; raw slopes log-uniform `[0.5,1.5]`, endpoint-normalized; accept final slopes `[0.4,2.0]` within 128 draws |
| Pause/reversal | 20% of training warps set middle raw slope to 0 then endpoint-normalize other slopes; reversal is global endpoint swap and is never mixed with a local direction change |
| Interpolation | linear only between adjacent distinct valid clocks; never across an invalid cell |
| Warp-integral loss | circular Huber, beta 0.10 rad; alias margin `0.05π`; min 32 valid intervals/window and 80% track coverage |
| Optimizer | AdamW, lr `3e-4`, weight decay `1e-4`, batch 64 windows, gradient clip 1.0, 20,000 steps, no metric-based early stop |
| Training seeds | 20270815, 20270816, 20270817; no seed replacement |
| Bootstrap | 9 development components; B=10,000; PCG64 seed 20270815; NumPy linear central-percentile quantiles |

Warp schedules and diagnostic schedules are serialized before evaluator access. The diagnostic piecewise-warp list is fixed independently of counts. All engineering tolerances above are invariants; no development metric selects them.

## 7. Gates and Training Plan

### Gate 0 — packer, permissions, and frozen population

Create the physically split shards/vault, opaque join table, source-component manifest, count-blind eligibility manifest, checksums, forbidden-key scan, and forbidden-path denial receipt. Freeze the nine development components. Any failure invokes K6.

### Gate 1 — deterministic numerical fixtures

Run duplicate collapse, invalid-cell interpolation, oriented overlap, wrap, pause, reversal, alias, final-window, positive-denominator, grid-origin, one-decode, and video-first scorer fixtures. A hand-worked bootstrap fixture must verify component multiplicity and seed pairing. These are not efficacy results.

### Gate 2 — fundamental identifiability

Run the 1,000 seeded synthetic selector fixtures and every eligible count-blind training track through the harmonic/nested-window/view rules in Section 3.2. Any failure invokes K5 before full training. Evaluator labels remain closed.

### Gate 3 — sanity only

At most three non-decision training-only jobs verify finite gradients, selector coverage, loss decrease, state-key isolation, deterministic receipts, and fixed prediction schema. No count/period metric is read.

### Gate 4 — staged decision runs

Run the three fixed seeds without replacement or cherry-picking. Freeze commits, configs, environments, raw per-person predictions, phase traces, and completion receipts before evaluator joins labels.

### Gate 5 — untouched paired evaluation

Open the evaluator only after each predeclared stage is complete. Run the exact algorithm below once per stage; never tune on development outputs.

## 8. Exact Primary Estimator and Bootstrap

There are exactly nine frozen development source components. For each training seed \(s\in\{1,2,3\}\), method \(a\in\{c,t\}\), and video \(v\), compute person-first absolute error

\[
A_{a,s,v}=|P_v|^{-1}\sum_{p\in P_v}|\hat C_{a,s,v,p}-C_{v,p}|,
\qquad
M_{a,s}=|V|^{-1}\sum_{v\in V}A_{a,s,v}.
\]

Treatment and control use identical videos/persons. Aggregate training seeds first:

\[
\bar M_c=\tfrac13\sum_s M_{c,s},\quad
\bar M_t=\tfrac13\sum_s M_{t,s},\quad
D=\bar M_c-\bar M_t,\quad
R=D/\bar M_c.
\]

If \(\bar M_c=0\), the relative clause is undefined and K1 fails; no epsilon or alternate endpoint is used.

For \(B=10{,}000\) draws using NumPy `Generator(PCG64(20270815))`, sample nine component indices with replacement. A component selected \(m\) times contributes every one of its videos and people with multiplicity \(m\). Apply the same draw to treatment/control and all three seeds. Within each seed and arm, recompute person-first video errors and then the multiplicity-weighted video mean; average the three paired seed effects to obtain absolute \(D^{(b)}\). The 95% CI is the central percentile interval `[quantile(D*,0.025,method='linear'), quantile(D*,0.975,method='linear')]`. It covers source-component sampling only, not training-seed uncertainty. Report the three seed-level effects and their sample SD separately.

K1 passes only when the unbootstrapped point estimate has \(R\ge0.05\) and the absolute-\(D\) CI lower endpoint is strictly greater than zero. AvgOBO and supplied-track Period-mAP/AP50/AP75 on scale `[0,100]` are secondary and cannot substitute. A tiny two-component, two-video fixture with hand-computed multiplicities, video-first values, \(D\), and seed average must pass byte-for-byte before evaluator use.

## 9. Staged Comparators and Hard Budget

Comparators are not falsely declared both capacity-matched and faithful. A **mechanism control** uses the same 225,026-parameter backbone/interface; a **faithful supplied-track adaptation** preserves the cited method's defining objective/head and reports its actual parameters and provenance.

| Stage | Condition | Full three-seed jobs | Category / purpose |
|---|---|---:|---|
| A | treatment + identical augmentation-only | 6 | primary paired endpoint |
| B | direct signed-frequency regression; global-warp-only; remove reversal-sign supervision | 9 | same-backbone mechanism falsification; run only if A passes K1 |
| C | PAMS; Track-PAMS; SimPer-style; CycleCL-style; DeepPhase-style; generic time-equivariant | 18 | faithful/same-backbone category frozen per method; run only if A/B survive K2/K3 |
| D | masked geometric TSSM control | 3 | optional control-only, only if the frozen comparison matrix requires it; never a fallback or contribution |

Track-global autocorrelation and stationary local-window decoders are deterministic checks with no learned run. PAMS's supplementary internal-tempo schedule is a diagnostic schedule applied to the PAMS adaptation, not a new model. Each of the seven possible learned comparator families in C/D receives at most two training-only tuning jobs at 5,000 steps, never development labels. Together with three sanity jobs, the absolute maximum is **53 training jobs**: 3 sanity + 36 decision-bearing full jobs + 14 tuning jobs. A full job is capped at 12 GPU-hours and a tuning job at 3 GPU-hours; sanity total is capped at 6 GPU-hours, so the complete ceiling is **480 GPU-hours**. No over-budget continuation is permitted.

For every learned comparator, use the same paired nine-component bootstrap as K1. A comparator is declared separated only when the lower endpoint of the absolute improvement \(M_{\mathrm{comparator}}-M_{\mathrm{treatment}}\) is strictly positive; otherwise it matches treatment within uncertainty and K2 applies.

If Stage A fails, stop after at most 9 jobs/78 GPU-hours and save at least 44 jobs/402 GPU-hours. If Stage B fails, stop after at most 18 jobs/186 GPU-hours and save at least 35 jobs/294 GPU-hours. Comparator code provenance, interface mapping, parameter count, optimizer steps, and tuning receipts are mandatory.

## 10. Three Visible Validation Blocks

### Block 1 — Primary paired law versus augmentation

Show one table with the three seed-paired piecewise-warp video-first AvgMAE values, their mean/SD, \(D\), \(R\), and the exact nine-component CI. Show clean AvgMAE and secondary metrics only as labeled diagnostics. This block alone decides K1.

### Block 2 — Staged mechanism and closest-prior falsification

Show the deletion/direct-frequency/global-warp/reversal results first, then only the reached closest-prior stage. Include shortcut receipts for no-pose clock/nuisance input, matched-time pose shuffle, and unseen resampler. Timestamp/no-pose retention above 20% of a positive treatment gain, or a closest prior matching within uncertainty, kills the residual claim. Detailed corruption and integrity tests remain in supplement/receipts.

### Block 3 — Identity isolation

With a frozen trained backbone, warp exactly one supplied identity while every untouched track tensor and clock remains byte-identical. The main comparison is private state versus shared-scene state and shuffled-key state. Report untouched-person phase-response change, continuous count-mass change, and rounded count change with component intervals. Stateless/reset/swap variants remain debugging receipts unless contamination appears. This is a diagnostic, never a private-memory contribution and never a rescue for Block 1.

## 11. Failure Modes and Frozen K1–K9 Rules

- **K1 — primary effect/uncertainty failure:** Survive only if point relative piecewise-warp video-first AvgMAE improvement is at least 5% and the exact paired nine-component bootstrap absolute-\(D\) 95% CI lower endpoint is greater than zero. \(\bar M_c=0\) fails. Otherwise kill the primary mechanism.
- **K2 — closest prior matches:** If PAMS/Track-PAMS, SimPer-style, CycleCL-style, DeepPhase-style, or matched generic time-equivariant learning matches or exceeds treatment within the frozen uncertainty rule, kill the residual method claim. Private state and TSSM cannot rescue it.
- **K3 — warp-integral objective is redundant:** If deleting \(\mathcal L_{\mathrm{WI}}\), direct signed-frequency regression, global resampling only, or removing reversal-sign supervision matches treatment within uncertainty, conclude the local objective is redundant.
- **K4 — shortcut/artifact:** If a timestamp/no-pose or nuisance-only model retains more than 20% of a positive treatment gain, an unseen resampler erases the gain, or matched-time pose shuffling preserves behavior, reject the result as clock/augmentation shortcut.
- **K5 — phase unit or execution is not identifiable:** If the target-free selector fails any synthetic or all-track training-only gate; phase orientation/collapse/harmonic stability fails; alias/valid support is inadequate; or NOLA seam mass is not conserved, kill before or during the full run. Do not calibrate with evaluator periods or add modules.
- **K6 — integrity failure:** If physical feature/vault separation, count-blind eligibility, source isolation, permissions, forbidden keys/paths, hashes, frozen predictions, or untouched evaluation cannot be demonstrated, stop with no claim.
- **K7 — clock/missingness invalidates the estimand:** If duplicate conflicts, missing-pose selection, unavailable timing, or alias masking prevents a defensible eligible population/estimand, rebuild the cache or restrict work to engineering diagnostics.
- **K8 — novelty collision:** Obtain and inspect the complete TWCRAC method before claim freeze. If it contains the same discrete local warp-phase transformation objective or an equivalent contribution, kill or materially re-anchor novelty and rerun novelty review. Abstract-only inspection is insufficient.
- **K9 — forbidden fallback promotion:** If the primary fails and masked geometric TSSM is proposed as an automatic replacement, stop. It remains a control unless a separate preregistered anchor and fresh novelty review authorize otherwise.

## 12. Experiment Handoff and Status

**Required artifacts:** feature shards, vault, eligibility/component/join manifests, forbidden-key/path receipts, selector/harmonic fixtures, NOLA/overlap/scorer/bootstrap fixtures, three-seed run receipts, raw per-person predictions, phase traces, comparator provenance, and TWCRAC adjudication.

**Run order:** Gate 0 isolation/eligibility → Gate 1 numeric fixtures → Gate 2 fundamental identifiability → Gate 3 sanity → Stage A primary pair → Stage B mechanism checks → Stage C closest priors → optional Stage D control → identity diagnostic → complete TWCRAC K8 review. Stop at the first applicable K-rule.

**Forbidden evidence:** v46/v62/v63 efficacy, sealed test data, predicted-track claims, official MultiRep main-result language, published end-to-end numbers as same-protocol rankings, and any result lacking a complete frozen receipt.

**Current status:** Round-1 revised proposal only; not implemented; no eligible results; same-family provisional. Training remains blocked until Gates 0–2 pass. Claim freeze remains blocked until the reached experiment stages, audit/result-to-claim gates, and complete TWCRAC K8 adjudication pass.
