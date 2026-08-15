# Round 4 Research-Refine Refinement: WARP-PHASE

> **PROSPECTIVE PILOT**
>
> **NO ELIGIBLE RESULTS**
>
> **SAME-FAMILY PROVISIONAL**

**Executor:** proposal executor

**Review being answered:** `refine-logs/round-4-review.md`, canonical SHA-256 `c1d44bbe3dd5467b02c4d29bf7bba2849b5bcf64a705c8276f0882a072a2e7a1`

**Reviewer:** GPT-5.6-Sol (`/root/research_refine_reviewer`)

**Round-4 score/verdict:** 8.5/10, `REVISE`

**Permitted protocol name:** **GT-bbox-assisted AlphaPose supplied-track partial-cache development pilot**

This is a complete prospective replacement proposal. It is not a patch note. No canonical implementation exists, no eligible method run has been made, and no numerical efficacy statement is licensed. Historical v46, v62, and v63 artifacts remain audit inputs only.

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
- **PASS — population and estimand:** Only checksum-frozen, source-disjoint partial-cache train/development entries are used. Evaluation remains per-person then video-first, with all nine audited source components as bootstrap clusters.
- **PASS — supervision:** The model process never reads count, density, evaluator period, cycle boundary, object mapping, source identity, or a real identifier. Evaluator period opens only after the relevant predictions and traces are frozen.
- **PASS — endpoint:** K1 remains the 5% relative normalized AvgMAE improvement **and** positive paired-component CI rule. Period diagnostics and the evaluator harmonic gate cannot rescue K1.
- **PASS — scope and evidence:** No router, learned expert, new TSSM, private-state novelty, scene-total objective, predicted-track claim, historical efficacy claim, new model, new baseline, new job, or new visible validation block is added. Eligible results remain zero.
- **Rejected as drift:** Treating selector output as a semantic action-cycle label, treating a deterministic fixture as efficacy, or using abstract-only TWCRAC evidence to pass novelty would change the licensed claim. The selector remains a rejection-gated pseudo-cycle and K8 remains blocked.

## Simplicity Check

- **Dominant contribution:** one discrete warp-integral equivariance loss is the only proposed mechanism.
- **Smallest adequate closure:** the seven Round-4 blockers are closed by constants, masks, deterministic interpolation, one authoritative selector path, frozen bytes, and one formula table inside the existing route.
- **Shared base:** pose normalization, ACF/FFT selection, PAMS-TCC-style correspondence, Conv/GRU recurrence, NOLA, and decoding remain shared, established/non-contribution machinery.
- **One selector:** ACF and FFT remain the two terms of the single existing score. There is no second selector, semantic-cycle correction, learned confidence, or alternate long-span branch.
- **One learner and deletion:** treatment and augmentation-only use the same 225,026-parameter learner; the only treatment deletion is `L_WI`. K4 controls add no trainable parameters.
- **Three visible validation blocks:** primary pair; staged mechanism/shortcut/closest-prior falsification; identity isolation. Gates and fixtures remain receipts.
- **No forced modernization:** an LLM, VLM, diffusion model, router, expert bank, or learned imputer would weaken causal attribution and is not added.
- **Blocked novelty boundary:** `idea-stage/TWCRAC_ADJUDICATION.json`, SHA-256 `5a1e7b8c1048a8884690f2d234d413064d473ead1eb1deca69ab734ca2e5ed1e`, records `BLOCKED`, `k8.decision=NOT_PASSED_BLOCKED`, `claim_freeze=BLOCKED`, and `novelty_clearance=false`. It is not novelty closure.

## Changes Made

1. **Closed mask-aware COCO17 normalization.** Shoulders are indices 5/6 and hips 11/12. Root and shoulder points are means of the available valid pair members, frame roots fail closed when no hip is valid, and the track scale is a float64 median of positive finite root-to-shoulder distances with the `1e-3` floor. There is no box or evaluator fallback.
2. **Made one selector path authoritative.** Only the unjittered clean 128-retained-clock parent (or the complete 64–127-clock support) supplies a canonical selection. The seeded weak view and nested 64-clock supports reject only. The track pseudo-period is the NumPy float64 median of valid canonical selections; `g_j` is exactly one on valid intervals and zero otherwise.
3. **Froze the retained-clock FFT256 and its single ACF/FFT score.** Every selector support is sampled at 256 uniformly spaced source clocks, without truncation or an alternate span branch. Mask-weighted mean subtraction, periodic Hann, `rfft(norm='forward')`, mask-energy normalization, off-bin linear interpolation, zero/Nyquist exclusion, endpoint maxima, and score/tie rules are unique.
4. **Froze warp tensor interpolation.** Queries use the unique increasing clean source clocks; exact endpoints are included; `x/y/confidence` are float64-linear only within one non-conflict cell with both source joints valid, then cast to float32 round-to-nearest-even. Masks are formed before zeroing. Pauses repeat a query, reversals descend without sign canonicalization, and out-of-support queries are invalid.
5. **Froze exactly 1,000 fixtures.** A single NumPy `Generator(PCG64(20270815))`, float64 reference generator, nine mutually exclusive categories with counts summing to 1,000 and exactly 250 symmetric cases, fixed draw/composition order, canonical JSON/NPY serialization, NumPy receipt, source hash, and fixture hash are mandatory before Gate 2. No post-hoc fixture changes are allowed.
6. **Closed the K4 code boundary.** Primary treatment and augmentation-only receive no fixed `q/L/mask` additive code. Only clock/no-pose and mask-only use the fixed-code interface. The exact normalized `u_q,u_L` vectors and two PCG64 Rademacher/MGS mask vectors are serialized as little-endian float64 and hash-bound.
7. **Replaced semantic K4 predicates with one decision table.** All decisions use unclipped seed-averaged normalized video-first AvgMAE, a positive uncorrupted gain denominator, paired data/seeds, and exact `Q_clock`, `Q_mask`, `P_shuffle`, and `P_unseen` formulas. Non-finite values or receipt mismatch fail.

The Round-3 phrases “median valid shoulder-to-hip scale,” “nested 64/128 selector,” “smallest `P` within 0.02,” “fixed selector confidence,” and K4 “retains/preserves/erases” are explicitly superseded by the unique definitions below and are not implementation alternatives.

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

**Claim ceiling.** Permitted wording is “we study whether” on a GT-bbox-assisted AlphaPose supplied-track partial-cache development pilot. This is not end-to-end, annotation-free, predicted-track, state-of-the-art, or deployment evidence. PAMS already reports localized internal-tempo perturbation; DeepPhase already occupies unsupervised motion phase. TWCRAC is `BLOCKED`, K8 is not passed, and claim freeze is blocked.

## 2. Count-Blind Data Contract and Eligibility

### 2.1 Physical isolation and exact desensitized schema precondition

A trusted offline packer may read the canonical v44 train/validation pickles once. It emits feature-only non-executable shards, a separately permissioned evaluator vault, a split/component manifest, a count-blind eligibility manifest, per-file/per-sample SHA-256 receipts, and a forbidden-key receipt. The training process cannot read the original pickle or vault and rejects non-whitelisted fields before payload deserialization.

Model-process fields are only `motion`, `person_mask`, `frame_mask`, `sampled_frame_indices`, `source_length`, an opaque sample key, and a local supplied-slot index. The opaque key and slot may route records and identity-private state only; they are never embedded, hashed into a feature, or cast to a model tensor. Warp schedules and augmentation metadata may construct augmented poses, losses, diagnostics, and receipts only; their values are never model inputs. Counts, densities, evaluator periods, boundaries, boxes, annotation names, object mappings, association structures, source IDs, and real identifiers never enter a model tensor, objective, selector, tuning decision, or checkpoint.

Before Gate 0 can pass, the packer, feature reader, and vault reader must round-trip the following exact desensitized canonical fixture and reject every field not represented by it. The normative bytes are the single UTF-8/no-BOM line below, with no terminal newline; keys are already recursively sorted, JSON separators are exactly `,` and `:`, and its SHA-256 is `31f81549f9c5dbedc6ddb278e868793246826a234de90c06c169c71387428275` (617 bytes). It contains types and relationships only, no real video/person/object/source identifier and no annotation value.

```json
{"annotation":{"height":"int","length":"int>1","object_schema":{"bbox":"<f8[L,4] finite","count":"int K","period":"int[K,2] with 0<=s<e<=L","periodicity":"empty list"},"width":"int"},"evaluator_vault":{"P_eval":"numpy.median(asarray(e-s,dtype=float64))","integrity":"K>0; Python int endpoints; 0<=s<e<=L","interval_semantics":"[s,e) source-frame boundary units"},"feature_shard":{"frame_mask":"bool[P,320]","local_person_slot":"int routing only","motion":"<f4[P,320,17,3]","opaque_sample_key":"non-semantic digest routing only","person_mask":"bool[P]","sampled_frame_indices":"<i8[320]","source_length":"Python int"}}
```

The feature-side fixture must deserialize without exposing the annotation or evaluator objects; the vault-side fixture must deserialize without exposing feature motion or routing values. Byte/hash mismatch, a real identifier, an extra field, executable deserialization, or cross-side visibility invokes K6 before eligibility. This exact fixture supersedes any informal packer-schema example.

### 2.2 Frozen count-blind population

Eligibility is computed and checksum-frozen before the evaluator opens `count_gt`, `density_gt`, or period annotations and is never revised from a label or metric.

1. Exclude the 13 training and 8 development association-ambiguous slots and require a one-to-one supplied-slot/object association in the audit manifest.
2. Require frozen GT-assisted pose coverage at least 0.80. The trusted packer uses this audit-only scalar solely to freeze inclusion, then removes it from model shards.
3. After duplicate-clock collapse, require at least 64 retained distinct source clocks, at least 63 valid adjacent cells, and at least 0.60 feature-valid clock coverage.
4. Do not filter on count, density, evaluator period, action, predicted difficulty, selector output, development error, or harmonic agreement. Report every exclusion and its count-blind reason.
5. The selector must answer for every eligible slot. Failure is global K5, never silent slot removal.
6. After Rules 1–5, every one of the original nine audited development source components must contain at least one eligible video with at least one eligible person. Hash the table and use it unchanged for every arm. Any empty component invokes K7 before evaluator access; components are never renumbered.

### 2.3 Duplicate-clock collapse

For each integer source clock `t`, group all samples with `sampled_frame_indices==t`. Retain the first sample only if every repeated member has identical frame/joint masks, maximum absolute normalized-`x,y` difference at most `1e-5`, and maximum confidence difference at most `1e-6`; otherwise mark `t` invalid. Never average disagreement. Sort retained non-conflict clocks strictly increasingly. No interpolation may cross a conflict location or invalid cell. Eligibility, selector, clean-rate construction, warp tensors, recurrence masks, and diagnostics share this rule.

## 3. Shared Phase Backbone and the One Selector

Treatment and augmentation-only use the same Track-PAMS/PAMS-TCC-style label-restricted phase backbone. Its selector and correspondence objective are frozen shared machinery, not contributions.

### 3.1 COCO17 landmark normalization

A source joint is valid only when its frozen joint mask is true, its `x,y,confidence` values are finite, and confidence is at least 0.20. COCO17 shoulders are indices 5 and 6; hips are indices 11 and 12.

For frame `i`, the root `r_i` is the float64 arithmetic mean of the `x,y` coordinates of all available valid hips among `{11,12}`. If neither hip is valid, the frame is invalid. The shoulder point `a_i` is the float64 arithmetic mean of all available valid shoulders among `{5,6}`; if neither shoulder is valid, that frame supplies no scale sample but does not by itself invalidate an otherwise root-valid frame. A positive finite scale sample is `d_i=||a_i-r_i||_2` when both points exist and `d_i>0`.

The track scale is computed once as `s=max(numpy.median(asarray(all d_i,dtype=float64)),1e-3)`, where the median includes every positive finite scale sample on the collapsed clean track. If there is no such sample, invoke K7. Every valid joint coordinate is root-centered by `r_i` and divided by `s` in float64, then cast to float32 round-to-nearest-even for the model. A feature-valid frame additionally requires at least 8 of 17 valid joints. There is never a box, image, learned imputer, previous-frame, population statistic, or evaluator fallback for root or scale.

### 3.2 Velocity signal and canonical supports

For a non-conflict clean cell `[q_i,q_{i+1}]` whose endpoints contain a valid normalized coordinate for joint-coordinate dimension `d`, define its float64 piecewise-constant velocity

\[
v_d(s)=\frac{x_{i+1,d}-x_{i,d}}{q_{i+1}-q_i},\qquad s\in[q_i,q_{i+1}),
\]

with the exact right endpoint `q_last` assigned to the last valid left cell. Otherwise that dimension is masked. Confidence is not a velocity dimension. Coordinate masks are carried separately and invalid values are zeroed only after masks form.

Canonical parent supports are deterministic 128-retained-clock windows with starts `0,64,128,...` while full, plus `K-128` if absent. When `64<=K<128`, the complete retained support is the single parent. Only the **unjittered clean parent support** produces the authoritative selection for that parent. For rejection only, run the same selector on one seeded weak spatial view of that parent and on the parent's first and last 64-retained-clock nested supports (for `K<128`, the first and last 64 clocks of the complete support). The weak view uses coordinate jitter `N(0,0.01^2)` clipped to `[-0.03,0.03]` and 10% joint dropout while retaining at least 8 joints; it never changes clocks. Rejection outputs never replace, average, vote with, or correct the authoritative selection.

A parent canonical selection is valid only if the weak-view and both 64-clock rejection selections exist, each is within 10% of the unjittered parent selection, reversal preserves its magnitude within 10%, and the canonical score margin over `P/2` and `2P`, when those integer candidates exist, is at least 0.10. The track pseudo-period `hat P` is exactly `numpy.median(numpy.asarray(valid_parent_selections,dtype=numpy.float64))`. No valid canonical selection is K5; there is no fallback. This one track value supplies all PAMS positives and is frozen into the later evaluator trace.

### 3.3 Retained-clock-to-FFT256 construction

For every canonical or rejection support, let its first and last retained source clocks be `q_first` and `q_last`. If `q_last<=q_first`, selection fails. Construct exactly

\[
x_n=q_{first}+n\Delta,\quad n=0,\ldots,255,\qquad
\Delta=(q_{last}-q_{first})/255
\]

in float64, including both endpoints. Query each velocity dimension at `x_n` using only its same valid non-conflict source cell; exact internal breakpoints use the cell on their right and `q_last` uses the last cell on its left. Unsupported dimensions are invalid. There is no truncation, padding branch, alternate FFT length, retained-index FFT, or rejection based merely on span length. FFT sample spacing is `Delta` source frames, so a candidate period `P` in source-frame units maps to continuous bin `k(P)=256*Delta/P`, and a bin `k` maps back to `P=256*Delta/k` source frames.

### 3.4 Exact ACF/FFT score

Integer candidates are `P=4,...,P_max`, where `P_max=min(128,floor((q_last-q_first)/2))`; `P_max<4` fails. For ACF at period `P`, query the same velocity field at every pair `(x_n,x_n+P)` still inside the support. Let `m_{n,d}(P)=1` only when coordinate `d` is valid at both queries, otherwise zero. Require at least 16 sample indices `n` with at least one valid dimension. Then

\[
R(P)=\frac{\sum_{n,d}m_{n,d}(P)v_d(x_n)v_d(x_n+P)}
{\sqrt{\left(\sum_{n,d}m_{n,d}(P)v_d(x_n)^2\right)
\left(\sum_{n,d}m_{n,d}(P)v_d(x_n+P)^2\right)}+10^{-8}}.
\]

For FFT, let `m_nd` be the coordinate-valid mask at `x_n`. For each coordinate, subtract its valid-mask weighted mean `mu_d=sum_n m_nd v_nd / sum_n m_nd`; a zero coordinate denominator makes that coordinate identically invalid. Use the periodic Hann

\[
w_n=0.5-0.5\cos(2\pi n/256).
\]

Set `y_nd=m_nd*w_n*(v_nd-mu_d)` and set every invalid entry to exact zero. Compute `X_kd=numpy.fft.rfft(y[:,d],n=256,norm='forward')`. The scalar power is

\[
E_k=\frac{\sum_d|X_{k,d}|^2}{\sum_{n,d}m_{n,d}w_n^2}.
\]

A zero or non-finite denominator fails selection. Bins 0 and 128 (Nyquist) are excluded from both targets and normalization. Define `E*(k)=0` for `k<=0` or `k>=128`; otherwise linearly interpolate the two adjacent stored powers: with `a=floor(k)`, `b=ceil(k)`, `E*(k)=E_a` when `a=b`, else `(b-k)E_a+(k-a)E_b`. The only spectral denominator is `Z=sum_{k=1}^{127}E_k`, and non-finite or zero `Z` fails.

The single frozen selector score, preserving the existing ACF/FFT combination, is

\[
S(P)=0.5R(P)+0.5\frac{E^*(k(P))+0.5E^*(2k(P))+0.25E^*(3k(P))}{1.75Z+10^{-8}}.
\]

Candidates require finite `S(P)`, finite `R(P)`, `R(P)>=0.25`, and valid ACF support. A local maximum is two-sided for `4<P<P_max` (`S(P)>=S(P-1)` and `S(P)>=S(P+1)`), one-sided at `P=4` (`S(4)>=S(5)`), and one-sided at `P=P_max` (`S(P_max)>=S(P_max-1)`); when `P_max=4`, its sole candidate is an endpoint maximum. Select the local maximum with the larger score; an exactly equal float64 score chooses the smaller `P`. The former “within 0.02” rule is deleted. There is no second selector or post-selection harmonic correction.

Gate 1 includes three deterministic FFT fixtures in the 1,000-fixture byte pack: an all-valid float64 sinusoidal velocity with `Delta=1` and source period `P=20` so `k=12.8` exercises adjacent-bin interpolation and must select 20; an endpoint case with `P=4` and `P_max>4` that must pass the one-sided lower test; and a support with span 126, `P_max=63`, and `P=63` that must pass the one-sided upper test. Expected `x`, masks, `E_1...E_127`, interpolated harmonic powers, scores, local-max masks, and selected periods are serialized before Gate 2; byte or tolerance mismatch invokes K5.

### 3.5 PAMS correspondence and pseudo-cycle boundary

The selector output is only a **selector-defined pseudo-cycle**. One selected pseudo-cycle maps to one `2pi` winding internally and makes no claim of annotated semantic repetition. With the frozen track `hat P`, valid PAMS positives are `(a,b)` with separation at most `hat P`, target pseudo-angle `2pi(q_b-q_a)/hat P`, and

\[
\mathcal L_{PAMS}=|\mathcal A|^{-1}\sum_{(a,b)\in\mathcal A}
\rho_{0.10}\!\left(\operatorname{wrap}\left[\operatorname{Arg}(z_b\overline z_a)-2\pi(q_b-q_a)/\hat P\right]\right).
\]

There is no activity head, reconstruction decoder, covariance/variance loss, seam learner, confidence learner, or semantic-period supervision.

### 3.6 Exact 1,000-fixture generator and byte freeze

Gate 1 materializes exactly 1,000 fixtures with NumPy `2.4.6` from one `numpy.random.Generator(numpy.random.PCG64(20270815))`. Every stochastic scalar and array is generated in float64 in the category/order listed below; integers use NumPy `integers(low,high)` with exclusive `high`. The receipt records NumPy `2.4.6`, platform endianness, generator source SHA-256, category counts, and final fixture-pack SHA-256. A NumPy-version mismatch fails rather than regenerates different bytes.

Every fixture has 128 unique clocks `q_i=float64(i)`, `i=0...127`, and the deterministic base skeleton `b_j=((j%5-2)/4,floor(j/5)/4)` for joints `j=0...16`. Per fixture, draw in order: `phi~Uniform(0,2pi)`, `A~Uniform(0.08,0.20)`, 17 `sx` and 17 `sy` Rademacher signs from `2*integers(0,2)-1`, category parameters from the table, frame-missing uniforms, joint-missing uniforms, weak-view Gaussian jitter, and weak-view dropout uniforms. Coordinates before category transforms are `x_ij=b_jx+A*sx_j*sin(theta_i+phi)` and `y_ij=b_jy+A*sy_j*cos(theta_i+phi)`; confidence is `0.9`. Fundamental second/third-harmonic coefficients are independently drawn `Uniform(0,0.20)` and `Uniform(0,0.10)` and added with phases `2theta+phi` and `3theta+phi`. Bilateral pairs are `(5,6),(7,8),(9,10),(11,12),(13,14),(15,16)`.

| Fixed IDs | Count | Category and exact semantic-period construction |
|---:|---:|---|
| 0–149 | 150 | fundamental: integer `P=integers(12,49)`, `theta=2pi*q/P` |
| 150–249 | 100 | three-segment warp: same `P`; draw breakpoints and slopes by Section 6.2, then evaluate `theta=2pi*tau(q)/P` |
| 250–374 | 125 | symmetric half-harmonic challenge: even `P=2*integers(8,25)`; bilateral coefficients are mirrored; second harmonic amplitude 1.0 and fundamental amplitude 0.35 |
| 375–499 | 125 | symmetric double-harmonic challenge: `P=integers(8,25)`; bilateral coefficients are mirrored; `theta=pi*q/P` amplitude 1.0 plus semantic-period component `2pi*q/P` amplitude 0.35 |
| 500–599 | 100 | off-grid/off-bin: draw `P~Uniform(12.25,48.75)` until distance from the nearest integer is at least 0.20, with at most 128 draws then fail |
| 600–699 | 100 | pause: integer `P=integers(12,49)`; draw `a=integers(32,65)`, `h=integers(8,25)`, and replace source phase clock `q` on `[a,a+h]` by `a`, shifting the suffix by `h` |
| 700–799 | 100 | reversal: integer `P=integers(12,49)` and `theta=2pi*(127-q)/P`; no sign canonicalization |
| 800–899 | 100 | alias/boundary: IDs 800–849 draw `P~Uniform(4.05,4.45)`; IDs 850–899 draw `P~Uniform(62.55,62.95)` |
| 900–999 | 100 | endpoint maxima: IDs 900–949 use `P=4`; IDs 950–999 use `P=63` |

Exactly 250 fixtures (IDs 250–499) are symmetric. Category membership is mutually exclusive and fixed before any draw; there is no post-hoc relabeling. Composition order is skeleton → semantic phase → harmonic template → category time transform → missingness → weak rejection view. For IDs whose last decimal digit is 0–3, frame missingness threshold is 0.40 and joint missingness threshold is 0.30; otherwise they are 0.10 and 0.10. Symmetric fixtures apply joint missingness to whole bilateral pairs. The canonical view is never jittered. The weak view then applies the frozen `N(0,0.01^2)` clipped jitter and 10% dropout, deterministically restoring the lowest-index dropped valid joints until eight remain. Root/scale, selector, pause, reversal, alias, and endpoint behavior use Sections 3.1–3.4 without exceptions.

Serialization is one canonical UTF-8/no-BOM `metadata.json` (`sort_keys=True`, `ensure_ascii=False`, separators `(',',':')`, no terminal newline) plus C-order `.npy` version-2.0 arrays with explicit little-endian dtypes: `q <f8`, `pose <f8`, `joint_mask |u1`, `category <u2`, `semantic_period <f8`, and expected selector arrays/results in float64 or fixed-width little-endian integer types. Sort filenames by Unicode code point. The pack hash is SHA-256 over the concatenation `filename UTF-8 || 0x00 || file_bytes || 0x0a` in that order. Freeze the generator source hash and pack hash before Gate 2. Gate thresholds remain total agreement within 10% at least 95%, half and double selections each at most 2% overall and 5% within the symmetric subset. No fixture, expected byte, threshold, or generator may change after either hash is recorded.

### 3.7 Frozen evaluator-only harmonic gate

After every reached stage freezes raw per-person predictions, selector output, phase trace, configuration, and SHA receipts and revokes write access, a separate evaluator may open only a nonempty globally well-formed list of raw integer source-frame boundary pairs `[s_i,e_i)`, assert `0<=s_i<e_i<=source_length`, and compute

\[
P_{eval,p}=\operatorname{numpy.median}(\operatorname{numpy.asarray}([e_i-s_i],dtype=\operatorname{float64})).
\]

Malformed, empty, non-integer, unsorted, overlapping, reversed, zero-length, or out-of-range data abort the entire gate under K6. No track is filtered and no count, density, periodicity, box, pose, selector, fallback field, 320-grid inversion, or correction is permitted. The binding addendum SHA-256 is `49edcfe25c302c2b55b8e65bc18e1cb7febe25cc8b76698d631fa24150d4b0b5`; its catalog, content, converter, official preprocessing/evaluator, join, source, and period-table hashes must match.

The frozen selector trace supplies the track `hat P` from Section 3.2. For `h in {0.5,1,2}`, classify by the unique value satisfying `abs(hat P/(h*P_eval)-1)<=0.10`, otherwise `off-grid`. Compute person-first fractions then unweighted video means `H_r`. Pass only if `H_1>=0.90` and each of `H_0.5,H_2,H_off-grid<=0.05`. The gate rejects only; it cannot change a count, pseudo-period, person set, model, or configuration.

## 4. Shared Network and Full-Identity Optimizer

Each pose step supplies 51 `x,y,confidence` values plus 17 joint-mask bits, for 68 channels. Missing values are zeroed only after mask concatenation. The shared learner is Conv1d `68→128`, kernel 5, padding 2; Conv1d `128→128`, kernel 5, padding 2; one GRU `128→128`; and a linear phase head `128→2`, followed by `z=u/(||u||_2+1e-8)`. It has exactly 225,026 trainable parameters: 43,648 + 82,048 + 99,072 + 258. No other trainable head exists.

Every training step samples exactly eight complete collapsed identities, right-pads clean/warp pairs to the longest identity, and carries person/frame/joint/cell/padding masks. Clean and warped Conv/GRU branches each traverse every complete chronological identity once in one fresh graph, with zero state only at identity start. Valid clocks commit the recurrent update; invalid/padded clocks hold state and emit invalid cells. Overlapping 64-sample loss/decoder windows are read-only views of that fresh graph and never reset, update, detach, or recompute recurrence.

Average valid window/interval terms within identity, then across the eight identities. Scale the scalar by `1/4`, call backward once, discard the complete graph, and accumulate only parameter gradients over four fresh steps. Clip accumulated gradient norm at 1.0 immediately before one AdamW update and zero gradients. AdamW uses learning rate `3e-4`, weight decay `1e-4`, 20,000 updates, and no metric early stopping. Exactly 640,000 complete-track draws occur absent fail-closed abort. No hidden state, activation, cache, decoded value, detached feature, or graph crosses a training step.

## 5. Unique Discrete Warp-Integral Objective

### 5.1 Raw signed clean measure

At retained clocks `u_0<...<u_{K-1}`, let `z_k` be the normalized phase. A cell `C_k=[u_k,u_{k+1})` is valid only when both endpoints are feature-valid, adjacent after collapse without a conflict, and strictly ordered. Define

\[
\delta_k=\operatorname{Arg}(z_{k+1}\overline z_k)\in(-\pi,\pi],\qquad
r_k=\delta_k/(u_{k+1}-u_k),\qquad \tilde r(s)=r_k\;(s\in C_k).
\]

No median sign, forward convention, or clean/warp sign canonicalization exists.

### 5.2 Sole signed overlap integral

For target interval `[q_j,q_{j+1})`, let `a=tau(q_j)`, `b=tau(q_{j+1})`, and `sigma=sign(b-a)`. Define

\[
\ell_{jk}=\sigma\left|[\min(a,b),\max(a,b))\cap C_k\right|,\qquad
I_j^\tau=\sum_k\ell_{jk}r_k.
\]

This float64 signed-overlap sum is the only clean/warp target and is cast to float32 only for the residual. Half-open cells assign a breakpoint to the right cell. Forward mappings are positive; reversal is negative; a supported pause has `a=b` and exact zero integral. Any uncovered nonzero length, invalid clean cell, invalid target endpoint, conflict, or out-of-range support invalidates the complete interval; partial support is never normalized.

The warped branch retains `delta_j^tau=Arg(z_{j+1}^tau conjugate(z_j^tau))`. The sole residual and loss are

\[
e_j=\operatorname{wrap}(\delta_j^\tau-\operatorname{sg}(I_j^\tau)),\qquad
\mathcal L_{WI}=\frac{\sum_jv_j\rho_{0.10}(e_j)}{\sum_jv_j}.
\]

The clean integral is stop-gradient. Set `v_j=0` if `abs(I_j^tau)>=pi-0.05pi`, any required clean principal increment has `abs(delta_k)>=pi-0.05pi`, or support is invalid. Never clip or unwrap a target. Require at least 32 valid target intervals per window and 80% valid target-interval coverage per eligible track, otherwise K5/K7.

### 5.3 Matched objectives

\[
\mathcal L_{control}=\mathcal L_{PAMS},\qquad
\mathcal L_{treatment}=\mathcal L_{PAMS}+\mathcal L_{WI}.
\]

Both arms share shards, selector, pairs, warp seeds, architecture, initialization, batches, optimizer, recurrence, NOLA, decoder, and evaluation. The only treatment deletion is `L_WI`.

## 6. Frozen Warp Family and Tensor Construction

### 6.1 Unique warp tensor interpolation

Let the clean source clocks after collapse be unique increasing float64 values `u_0<...<u_{K-1}`. For each target clock, query `s=tau(q)`.

- Support is endpoint-inclusive: `u_0<=s<=u_{K-1}`. `s<u_0`, `s>u_{K-1}`, non-finite `s`, or a query touching a conflict is out-of-support and invalid.
- If `s` equals an exact stored clock `u_i`, a joint is valid only if that source joint is valid; copy its `x,y,confidence` through float64.
- If `u_i<s<u_{i+1}`, a joint is valid only if both bounding source joints are valid and `[u_i,u_{i+1}]` is one non-conflict valid source cell. With `alpha=(s-u_i)/(u_{i+1}-u_i)` in float64, interpolate each of `x,y,confidence` as `(1-alpha)*value_i+alpha*value_{i+1}`.
- Form the joint-valid mask first. Cast each valid interpolated scalar to float32 using IEEE-754 round-to-nearest, ties-to-even; set all three scalars of an invalid joint to exact float32 zero afterward.
- Derive a target frame mask as person-valid, finite-query, at least 8 valid joints, and at least one valid hip among indices 11/12. Derive a target cell mask from two valid consecutive target frames plus complete supported oriented `tau` coverage by non-conflict source cells. A supported pause (`tau(q_j)==tau(q_{j+1})`) repeats the exact same query and has a valid zero-length cell; a reversal submits descending queries unchanged and never canonicalizes sign.

The same masks govern the warped model input, recurrence, signed integral, loss, and diagnostics. There is no nearest-neighbor, extrapolation, confidence max/min, mask averaging, box fallback, evaluator fallback, or alternate kernel.

### 6.2 Warp schedule

Use three affine segments. Draw breakpoints `b1~Uniform(0.20,0.40)` and `b2~Uniform(0.60,0.80)` and raw slopes log-uniform on `[0.5,1.5]`, then normalize endpoints; accept final segment slopes in `[0.4,2.0]` within at most 128 draws or fail. In 20% of scheduled warps set the middle raw slope to zero and normalize the remaining slopes for a pause. Reversal is a global endpoint swap, never local direction mixing. Schedules are serialized before evaluator access and are loss/audit-only, never model input.

## 7. Strict Positive-Interval NOLA and One Decode

A 64-sample half-open window has 63 intervals with

\[
w_i=\max\left(10^{-3},\sin^2\left(\pi(i+0.5)/63\right)\right)>0.
\]

Stride is 32 retained samples; append the final start `max(K-64,0)` if absent. For `K<64`, start at zero and right-pad without creating interval mass. For every global interval `j`, selector confidence is not learned or scored: **`g_j=1` exactly when the final joint/frame/cell, support, alias, and padding masks mark the interval valid; `g_j=0` otherwise.** Then

\[
D_j=\sum_{b\ni j}v_j^{(b)}w_{j-b},\qquad
\bar m_j=\frac{\sum_{b\ni j}v_j^{(b)}w_{j-b}g_j|\delta_j^{(b)}|/(2\pi)}{D_j}.
\]

Assert finite `D_j>=1e-3`; no epsilon rescue exists. Decode once per supplied identity, `hat C_p=sum_j bar m_j`; round-half-to-even is secondary and happens only after that sum. Grid origins 0, 8, 16, and 24 must conserve synthetic mass within `1e-6` relative error.

## 8. Frozen Numerical Implementation Table

| Interface | Frozen value |
|---|---|
| Input | full identities; `[8,T_max,17,3]` float32 pose plus frozen masks; 68 pose/mask channels |
| Root/scale | COCO17 shoulders 5/6, hips 11/12; available-valid means; float64 positive-distance median; floor `1e-3`; no support → K7 |
| Clock/integral | int64 stored clocks; float64 query, FFT grid, overlaps, integrals, metrics |
| Duplicate agreement | masks exact; normalized xy max `<=1e-5`; confidence max `<=1e-6`; disagreement invalidates clock |
| Selector authority | unjittered clean parent only; weak view and nested 64 supports reject only; track period is float64 median |
| FFT | 256 inclusive uniform source-clock samples; periodic Hann; valid-mask mean; `rfft(norm='forward')`; bins 1–127; off-bin linear interpolation |
| Selector score | one `0.5*ACF+0.5*weighted FFT` score; endpoint maxima one-sided; larger score then smaller `P`; no 0.02 rule |
| Decoder gate | `g_j=1` valid, `0` otherwise |
| Warp tensor | endpoint-inclusive, same-cell float64 linear joint interpolation; float32 ties-to-even; masks first, invalid zeros |
| Network | Conv1d 68→128; Conv1d 128→128; GRU 128; linear 128→2; 225,026 parameters |
| Optimizer | AdamW `3e-4`, wd `1e-4`; 8 identities/step; 4 fresh steps/update; clip 1.0; 20,000 updates |
| Warp loss | circular Huber beta 0.10 rad; alias margin `0.05pi`; min 32 intervals/window and 80% track coverage |
| Fixtures | exactly 1,000 PCG64(20270815) float64; counts fixed; source/pack hashes and NumPy receipt before Gate 2 |
| K4 codes | primary pair: none; clock/mask controls only: four fixed normalized 128-vectors, little-endian float64 SHA-bound |
| Seeds | 20270815, 20270816, 20270817; no replacement |
| Bootstrap | all 9 nonempty components; 10,000 PCG64(20270815) draws; linear central quantiles |
| Hard budget | 59 total training jobs; 552 GPU-hours absolute maximum |

Engineering tolerances are invariants; no development metric selects them.

## 9. Gates and Training Plan

### Gate 0 — isolation and frozen population

Pass the exact 617-byte desensitized schema fixture, create separated shards/vault, opaque join table, source-component and eligibility manifests, checksums, forbidden-key/path receipts, and assert all original nine development components are nonempty. Failure invokes K6/K7 before evaluator labels open.

### Gate 1 — deterministic numerical fixtures

Run duplicate collapse, landmark root/scale, invalid-cell interpolation, retained-clock FFT256, off-bin/endpoint maxima, canonical/rejection selection, signed overlap, wrap, warp tensors, pause/reversal, alias, recurrence single-commit, invalid-state hold, padding/masking, gradient accumulation, read-only overlap, NOLA, `g_j`, raw period extraction, K4 code bytes/input routing, normalized AvgMAE, and bootstrap fixtures. These are receipts, not efficacy.

The frozen two-component scorer fixture remains: component A has `C=[2,4]`, control `[3,2]`, treatment `[2,3]`, yielding normalized video errors `0.5` and `0.125`; B has `C=[5]`, control `[7]`, treatment `[6]`, yielding `0.4` and `0.2`. Thus `M_c=0.45`, `M_t=0.1625`, `D=0.2875`, `R=0.6388888888888888`; component draws `[A,A]`, `[A,B]`, `[B,B]` yield `D=0.375`, `0.2875`, `0.2`.

### Gate 2 — pseudo-cycle stability

Only after generator-source and fixture-pack hashes are frozen, run exactly the 1,000 fixtures and every eligible count-blind training track through the one selector. Failure invokes K5 before full training. The gate licenses only a stable pseudo-cycle interface.

### Gate 3 — sanity

At most three training-only jobs verify finite gradients, selector coverage, loss decrease, identity isolation, one recurrent commit per source clock, deterministic receipts, and prediction schema. No count or evaluator period is read.

### Gates 4–5 — staged runs and untouched evaluator

Run fixed seeds without cherry-picking. Freeze commits, configs, environments, predictions, pseudo-periods, and phase traces for all reached arms before evaluator join. Then run the no-correction harmonic rejection gate and exact normalized scorer once. Never tune from development output.

## 10. Exact Primary Estimator and Bootstrap

For seed `s`, arm `a`, video `v`, and eligible people `P_v`, assert finite `C_vp>0` and compute

\[
A_{a,s,v}=|P_v|^{-1}\sum_p\frac{|\hat C_{a,s,v,p}-C_{v,p}|}{C_{v,p}},\qquad
M_{a,s}=|V|^{-1}\sum_vA_{a,s,v}.
\]

Predicted continuous counts are not clipped. Average seeds first:

\[
\bar M_c=\tfrac13\sum_sM_{c,s},\quad \bar M_t=\tfrac13\sum_sM_{t,s},\quad
D=\bar M_c-\bar M_t,\quad R=D/\bar M_c.
\]

`bar M_c=0` fails K1. For 10,000 draws from `Generator(PCG64(20270815))`, sample all nine component indices with replacement; a component drawn `m` times contributes all videos/people with multiplicity `m`. Use the same draw for both arms and all seeds, recompute video-first metrics, then average paired seed effects. The CI is linear 2.5%/97.5% quantiles of absolute `D*`. K1 passes only if `R>=0.05` and the absolute-`D` CI lower endpoint is strictly positive. AvgOBO and supplied-track Period-mAP/AP50/AP75 on `[0,100]` are secondary and cannot substitute.

## 11. Staged Comparators, K4, and Hard Budget

| Stage | Condition | Three-seed full jobs | Purpose |
|---|---|---:|---|
| A | treatment + identical augmentation-only | 6 | primary paired endpoint |
| B | clock/no-pose + mask-only | 6 | K4; only if A passes K1 |
| C | direct signed-frequency; global-warp-only; unsigned reversal | 9 | mechanism falsification; only if A/B survive |
| D | PAMS; Track-PAMS; SimPer-style; CycleCL-style; DeepPhase-style; generic time-equivariant | 18 | closest priors; only after A–C survive K1/K3/K4; adjudicates K2 |
| E | masked geometric TSSM control | 3 | optional frozen control only; never fallback/contribution |

### 11.1 Exact K4 fixed-code interface

The primary treatment and primary augmentation-only models receive **no fixed `q`, `L`, or mask-derived additive code**. Their 68-channel pose/mask interface is unchanged. Only the two Stage-B zero-pose controls use the following non-trainable additive code after Conv1; no trainable parameter is added.

In float64 dimension 128, define `u_q[i]=(-1)^i/sqrt(128)` for zero-based `i` (first entry positive), and `u_L[i]=+1/sqrt(128)` for `i<64`, `-1/sqrt(128)` otherwise. Initialize `rng=Generator(PCG64(20270815))` once. For `r_1` then `r_2`, draw `r=2*rng.integers(0,2,size=128,dtype=int64)-1`, cast to float64, and perform one ordered modified Gram-Schmidt pass `r<-r-dot(r,b)b` against `[u_q,u_L]` for `r_1` and `[u_q,u_L,u_m1]` for `r_2`. A zero/non-finite norm fails; otherwise normalize by the float64 L2 norm. Find the first exact nonzero entry and multiply the complete vector by `-1` if that entry is negative. These are `u_m1,u_m2`.

Serialize the C-order `(4,128)` array `[u_q,u_L,u_m1,u_m2]` as raw little-endian IEEE-754 float64 bytes, with no header. The normative NumPy `2.4.6` serialization is exactly 4,096 bytes with SHA-256 `5e2f35a11bd8a376e44d5ad7d3e5067a0bb5a03e2d61edd84f5f5bfd6cf351ec`. The implementation must reproduce this byte hash; these literal bytes are authoritative and a version/hash mismatch fails K4.

Clock/no-pose and mask-only both derive the original masks first, then set every pose `x,y,confidence` to zero. Let `f_i` be per-frame valid-joint fraction and `f_track` the valid-frame fraction. Clock/no-pose adds `q_norm*u_q + L_norm*u_L + f_i*u_m1 + f_track*u_m2`, where `q_norm=q/max(L-1,1)` and `L_norm=log2(max(L,1))/16`. Mask-only adds `f_i*u_m1 + f_track*u_m2` and uses exact zero coefficients for `u_q,u_L`. Opaque keys/slots route only. Warp/crop/resampler schedules and metadata remain loss/audit-only. Both controls retain 225,026 parameters, the same seeds, optimizer, update cap, data, and budget.

### 11.2 The only K4 decision table

All `M` values below are unclipped, finite, seed-averaged normalized video-first AvgMAE on identical eligible videos/people. Define the uncorrupted primary gain `G=M_c-M_t`; Stage B is reached only when `G>0`, and otherwise K1 already stops. Corruptions use paired data and seeds for control/treatment, share the frozen receipts, and never clip predictions. A non-finite quantity, missing arm, data/seed/config/hash mismatch, or unequal parameter/budget receipt fails K4.

| Check | Exact statistic | K4 failure rule |
|---|---|---|
| clock/no-pose | `Q_clock=(M_c-M_clock)/G` | fail iff `Q_clock>0.20` |
| mask-only | `Q_mask=(M_c-M_mask)/G` | fail iff `Q_mask>0.20` |
| matched-time pose shuffle | `P_shuffle=(M_c^sh-M_t^sh)/G` | fail iff `P_shuffle>0.20` |
| unseen resampler | `P_unseen=(M_c^u-M_t^u)/G` | fail iff `P_unseen<0.80` |

No clipping to `[0,1]`, absolute value, epsilon, confidence interval, alternate denominator, “retention,” “preservation,” or “erasure” predicate is permitted. This table is the complete K4 decision interface.

### 11.3 Budget

Stages A–E contain at most 42 decision-bearing full jobs. Seven learned comparator families in D/E receive at most two training-only 5,000-step tuning jobs each, adding 14, and Gate 3 has three sanity jobs. Total is **59 jobs**. Full jobs cap at 12 GPU-hours, tuning at 3, and sanity together at 6: **552 GPU-hours = 42×12 + 14×3 + 6**. No over-budget continuation is permitted. Stopping ceilings remain 9 jobs/78 GPU-hours after A, 15/150 after B, and 24/258 after C.

## 12. Three Visible Validation Blocks

### Block 1 — primary paired law versus augmentation

Report three seed-paired piecewise-warp normalized video-first AvgMAE values, mean/SD, `D`, `R`, and the exact nine-component CI. This block alone decides K1.

### Block 2 — staged mechanism, shortcut, and closest-prior falsification

Report the single K4 table, then reached deletion/direct-frequency/global-warp/unsigned-reversal checks, then the reached closest-prior stage. Fixtures remain receipts/supplement. This is one visible block.

### Block 3 — identity isolation

With a frozen backbone, warp one supplied identity while untouched track tensors and clocks remain byte-identical. Compare private state with shared-scene and shuffled-key diagnostics; report untouched-person phase response and continuous/rounded count change with component intervals. It is a diagnostic, never private-memory novelty and never a rescue for Block 1.

## 13. Failure Modes and Frozen K1–K9

- **K1 — primary effect/uncertainty:** survive only if relative improvement is at least 5% and the paired all-nine-component absolute-`D` CI lower endpoint is greater than zero. `bar M_c=0` fails.
- **K2 — closest prior:** if a reached PAMS/Track-PAMS, SimPer-style, CycleCL-style, DeepPhase-style, or generic time-equivariant comparator matches/exceeds treatment within the frozen paired rule, kill the residual claim. Private state/TSSM cannot rescue it.
- **K3 — objective redundancy:** if deleting `L_WI`, direct signed-frequency regression, global resampling only, or unsigned reversal matches treatment, conclude the local signed objective is redundant.
- **K4 — shortcut/artifact:** apply only the four exact ratios and failure inequalities in Section 11.2. Any prohibited input/code, non-finite value, or receipt mismatch also fails. No semantic synonym is a decision rule.
- **K5 — pseudo-cycle/execution:** if the exact fixture pack, real-track canonical/rejection stability, evaluator harmonic gate, support/alias coverage, recurrence receipt, FFT fixture, or NOLA conservation fails, kill the phase-mass claim. Do not correct harmonics or add modules.
- **K6 — integrity:** if packer/vault separation, the exact desensitized schema fixture, positive-count assertion, source isolation, permissions, forbidden keys/paths, hashes, frozen predictions, or untouched evaluation cannot be shown, stop with no claim.
- **K7 — population/clock/missingness:** if any original development component is empty, no valid normalization scale exists, or duplicate conflicts, missing pose, timing, interpolation support, or alias masks invalidate the estimand, rebuild the cache or restrict work to engineering diagnostics.
- **K8 — novelty collision:** `idea-stage/TWCRAC_ADJUDICATION.json` SHA-256 `5a1e7b8c1048a8884690f2d234d413064d473ead1eb1deca69ab734ca2e5ed1e` has verdict `BLOCKED`, `k8.decision=NOT_PASSED_BLOCKED`, `claim_freeze=BLOCKED`, and `novelty_clearance=false`. K8 is **NOT PASSED**. Obtain and inspect the complete primary-source method before claim freeze; abstract-only evidence cannot pass it.
- **K9 — forbidden fallback:** if the primary fails and masked geometric TSSM is proposed as an automatic replacement, stop. It remains a control absent a new anchor and novelty review.

## 14. Experiment Handoff and Current Status

**Required artifacts:** separated feature/vault shards; exact 617-byte schema receipt; eligibility/component/join manifests; forbidden-key/path receipts; generator source and 1,000-fixture hashes; NumPy receipt; FFT/selector/harmonic, warp tensor, recurrence, NOLA, K4 code, scorer, and bootstrap fixtures; fixed schedules; three-seed run receipts; raw predictions and traces; comparator provenance; eventual complete-primary-source TWCRAC adjudication.

**Run order:** Gate 0 isolation/schema/nonempty components → Gate 1 numerical and byte fixtures → Gate 2 one-selector stability → Gate 3 sanity → Stage A/K1 → Stage B/K4 → Stage C/K3 → only if A–C survive, Stage D/K2 → optional Stage E → identity diagnostic → complete-primary-source K8 adjudication. Stop at the first K-rule.

**Forbidden evidence:** v46/v62/v63 efficacy, sealed test data, predicted-track claims, official MultiRep main-result language, published end-to-end numbers as same-protocol rankings, post-hoc harmonic correction, fixture changes after hashing, and any result without complete frozen receipts.

**Current status:** Round-4 revised proposal only; not implemented; **zero eligible results**; **same-family provisional**. The seven local interface blockers are closed prospectively by unique definitions, but passage of Gates 0–2 is not inferred. Training remains blocked until those gates pass. TWCRAC remains `BLOCKED`/`NOT_PASSED_BLOCKED`; K8 and claim freeze remain blocked. No Round-5 review or experiment plan has been started.
