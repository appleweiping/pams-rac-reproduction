# Round 3 Research-Refine Refinement: WARP-PHASE

> **PROSPECTIVE PILOT**
>
> **NO ELIGIBLE RESULTS**
>
> **SAME-FAMILY PROVISIONAL**

**Executor:** proposal executor

**Review being answered:** `refine-logs/round-3-review.md`

**Reviewer:** GPT-5.6-Sol (`/root/research_refine_reviewer`)

**Round-3 score/verdict:** 8.5/10, `REVISE`

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
- **One fresh recurrent graph per training step:** each step contains eight complete collapsed identities, padded and masked. Clean and warped Conv/GRU branches each traverse every complete identity clock exactly once; overlap windows are loss-only views of those fresh graphs. No hidden state, activation, or cache crosses a training step; only gradients accumulate for four steps.
- **Three visible validation blocks:** primary pair; staged mechanism/closest-prior falsification including K4 controls; identity isolation. Gates and fixtures remain receipts rather than paper contributions.
- **No modernization module:** known warp metadata, exact equivariance supervision, causal interventions, and matched deletion already fit the bottleneck. Adding an LLM, VLM, diffusion model, router, or expert bank would not sharpen the claim.
- **Frozen K4 controls:** the clock/no-pose and mask-only schemas differ only in their predeclared clock exposure; both keep the same masks, architecture, optimizer, seeds, parameter count, and budget, while all warp/crop metadata remains loss/audit-only.
- **Blocked novelty boundary:** the independent TWCRAC receipt SHA-256 `5a1e7b8c1048a8884690f2d234d413064d473ead1eb1deca69ab734ca2e5ed1e` records `BLOCKED` and `K8 NOT PASSED`. It is not novelty closure.

## Changes Made

1. **Froze a valid full-identity optimizer algorithm.** Every training step samples exactly eight complete collapsed identity sequences, pads and masks them, runs clean and warped Conv/GRU once per complete clock in one fresh graph, averages all window losses within each identity and then across identities, and performs one backward. Four consecutive fresh training steps are gradient-accumulated before the optimizer update, for 32 effective tracks. No cross-step or stale cache is permitted; window-count batching is deleted.
2. **Bound `P_eval,p` to the audited raw period schema.** A trusted evaluator reads only a nonempty, globally well-formed list of raw source-frame boundary pairs `[s,e)`, converts each pair to float64 length `e-s`, and takes the NumPy float64 median. It never reads count, density, periodicity, bbox, or a fallback field and never inverts the 320-sample grid. Catalog/source hashes and a deterministic extraction-and-classification fixture are mandatory receipts.
3. **Froze exact K4 model inputs.** Clock/no-pose zeroes all pose `x/y/confidence` values but preserves frozen joint/person/frame masks, `q=sampled_frame_indices`, and `L=source_length`. Mask-only zeroes pose and the model-visible `q/L` values while retaining the same masks and mask-derived valid fractions. Opaque keys/slots route records only and cannot be embedded. Warp schedules, breakpoints, slopes, interpolation kernels, seeds, crop parameters, and related metadata are loss/audit-only and never model input. Both controls retain the same parameters and budget; unseen-resampler remains a separate diagnostic.
4. **Corrected Stage-D control flow.** Stages A–C must survive K1, K3, and K4; Stage D then runs the closest priors and adjudicates K2. K2 is no longer claimed before its comparators run.
5. **Recorded the TWCRAC receipt without overclaim.** `idea-stage/TWCRAC_ADJUDICATION.json`, SHA-256 `5a1e7b8c1048a8884690f2d234d413064d473ead1eb1deca69ab734ca2e5ed1e`, has verdict `BLOCKED`, `k8.decision=NOT_PASSED_BLOCKED`, and `claim_freeze=BLOCKED`. This is evidence that K8 is not passed, not evidence of distinctness or closure.
6. **Preserved all prior closures and limits.** Raw signed increments, the unique float64 oriented overlap integral, selector-defined pseudo-cycle, frozen evaluator-only harmonic gate, strict NOLA, normalized video-first AvgMAE, all-nine-component bootstrap, three visible validation blocks, K1–K9, 59 jobs/552 GPU-hours, and zero eligible results remain unchanged.

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

**Claim ceiling.** Permitted wording is “we study whether” on a GT-bbox-assisted AlphaPose supplied-track partial-cache development pilot. This is not end-to-end, annotation-free, predicted-track, state-of-the-art, or deployment evidence. PAMS already reports localized internal-tempo perturbation; DeepPhase already occupies unsupervised motion phase. The TWCRAC receipt is `BLOCKED`, K8 is not passed, and claim freeze is blocked.

## 2. Count-Blind Data Contract and Eligibility

### 2.1 Physical isolation

A trusted offline packer may read the canonical v44 train/validation pickles once. It emits feature-only non-executable shards, a separately permissioned evaluator vault, a split/component manifest, a count-blind eligibility manifest, per-file/per-sample SHA-256 receipts, and a forbidden-key receipt. The training process cannot read the original pickle or vault and rejects non-whitelisted fields before payload deserialization.

Model-process fields are only `motion`, `person_mask`, `frame_mask`, `sampled_frame_indices`, `source_length`, an opaque sample key, and a local supplied-slot index. The opaque key and slot may route records and identity-private state only; they are never embedded, hashed into a feature, or cast to a model tensor. Warp schedules, breakpoints, slopes, target-to-source maps, resampling/interpolation kernels, augmentation seeds, and crop parameters may construct augmented poses, losses, diagnostics, and audit receipts only; their metadata values are never model inputs. Source IDs and hashes are split/audit-only. Counts, densities, evaluator periods, boundaries, raw boxes, annotation names, object IDs/mappings, association structures, and `gt_object_pose_coverage` never enter a model tensor, objective, selector, tuning decision, or checkpoint.

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

The period vault binds each eligible person through the frozen one-to-one supplied-slot/object join to the audited raw annotation `period` value. The only accepted schema is a nonempty sequence of source-frame boundary pairs `[s_i,e_i)`. Before any person is classified, the evaluator globally asserts for every reached eligible person that every entry is a length-two list of exactly two Python integers and `0 <= s_i < e_i <= source_length`, with no missing, empty, malformed, fractional, reversed, zero-length, overlapping, unsorted, or out-of-range entry. Any global assertion failure aborts the gate under K6; there is no person filtering or fallback.

For person `p`, convert every well-formed pair directly to the source-frame-clock duration `d_i=float64(e_i)-float64(s_i)` and define

\[
P_{\mathrm{eval},p}=\operatorname{median}_{\mathrm{NumPy,float64}}\{d_i\}.
\]

The implementation is exactly `numpy.median(numpy.asarray(spans, dtype=numpy.float64))`, including the mean of the two central float64 values for an even-length list. The evaluator does not read or derive this scalar from `count_gt`, `density_gt`, any `periodicity` field, bounding boxes, pose motion, selector output, or any alternate/fallback annotation. Raw boundaries already use the source-frame clock, so the evaluator never maps them to the 320-sample grid and never performs a 320-grid inverse mapping. It asserts the resulting scalar is finite and positive.

The binding source is `PILOT_PERIOD_SCHEMA_ADDENDUM.md`, SHA-256 `49edcfe25c302c2b55b8e65bc18e1cb7febe25cc8b76698d631fa24150d4b0b5`. Its canonical annotation-catalog SHA-256 values are train `c80f3ce837c28d9e5f2929e41b69c596f8ec1abdc32db7be72b93d8971a65ee6` and validation `be42cf759241039c7fd427633135b16c44c18189bec0284608635ee45191f2f2`; its path-free content-multiset SHA-256 values are train `c4c2a119cbb8cd392ff6550115136247d81a37a252771c72b400b70d9ae63d12` and validation `f48d8257e1003ba9044ce9fda0396cf139ed24aeed2f5d79dcc6c058283452b5`. The source-code bindings are converter `5a9b148f398e43ac11eac7b54468a0cb22ba2ea63622a962feaa9fdb3cd05e60`, official `video2frames.py` `ebf4dc08d18070e65d891b6faa5929de7e8f78c000cb4cd98d222ebf19d581aa`, and official evaluator `c10a3a56704d8bdf1ba55a2681570dc77371089ecb73c71c56102d26eca1386c`. The vault receipt additionally records every referenced per-video source JSON SHA-256, the frozen slot/object join SHA-256, and the extracted per-person period-table SHA-256. All hashes must match before evaluation; a mismatch invokes K6. The selector trace supplies one frozen track value `hat P_p`, the NumPy float64 median of its pre-vault valid window selections. For `h` in `{0.5,1,2}`, define

\[
d_{p,h}=\left|\frac{\hat P_p}{hP_{\mathrm{eval},p}}-1\right|.
\]

Assign `p` to the unique `h` with `d_{p,h} <= 0.10`; if none qualifies, assign `off-grid`. The intervals cannot overlap at this tolerance; any implementation tie or non-finite value is `off-grid`. For class `r`, compute person-first video fractions and then the unweighted video mean

\[
H_r=|V|^{-1}\sum_{v\in V}|P_v|^{-1}\sum_{p\in P_v}\mathbf 1[\operatorname{class}(p)=r].
\]

The gate passes only if `H_1 >= 0.90`, `H_0.5 <= 0.05`, `H_2 <= 0.05`, and `H_off-grid <= 0.05`. Report the same fractions by original source component, but component values do not replace the frozen overall rule. Failure invokes K5 and blocks the phase-mass claim. The evaluator may not correct counts, multiply/divide pseudo-cycles, select a harmonic, filter people/videos, tune tolerance, rerun training, or alter any frozen prediction. This gate is adjudication only, not calibration and not a secondary endpoint.

The deterministic schema fixture uses raw periods `[[10,22],[30,44]]`, hence float64 durations `[12,14]` and `P_eval=13.0` without reference to a 320 grid. With frozen `hat P=13.65`, `d_1=0.05` and the class is `1`; with `hat P=6.5`, the class is `0.5`; with `hat P=26.0`, the class is `2`; with `hat P=20.0`, the class is `off-grid`. A paired malformed fixture containing `[22,22]`, an empty list, or a non-pair must abort the complete gate rather than drop the person. Freeze the exact input bytes, expected float64 arrays/classes, NumPy version, and SHA-256 receipt before evaluator use.

### 3.4 Shared network, exact count, and single-commit recurrence

Each pose step supplies 51 `x,y,confidence` values plus 17 joint-mask bits, for 68 channels. Missing values are zeroed only after mask concatenation. The shared network is Conv1d `68→128`, kernel 5, padding 2; Conv1d `128→128`, kernel 5, padding 2; one GRU `128→128`; and a linear phase head `128→2`, followed by L2 normalization. It has exactly 225,026 trainable parameters: 43,648 + 82,048 + 99,072 + 258. No other trainable head exists.

At every training step, sample exactly eight complete collapsed identity sequences using the frozen seeded sampler. Bucket only to reduce padding, then right-pad each clean/warp pair to the longest complete identity in that step and carry explicit person, frame, joint, cell, and padding masks. Run the clean Conv/GRU once and the corresponding warped Conv/GRU once over every complete chronological identity in one fresh autograd graph, with zero hidden state only at identity start. At valid clock `k`, commit `h_k=GRU(x_k,h_{k-1})`; at an invalid or padded clock, hold `h_k=h_{k-1}` and mark the emitted cell invalid. Each branch computes every real clock once.

All eligible overlapping 64-sample PAMS, warp-integral, and decoder windows are read-only index views of the same fresh per-identity graph. They never reset, carry, update, detach, or recompute the GRU. Average valid window/interval terms within each identity first so long identities do not receive greater batch weight; then average the eight identity losses. Scale that step scalar by `1/4` and call backward exactly once. Immediately discard that step's hidden states, activations, views, and graph. Accumulate only parameter gradients across exactly four consecutive fresh eight-identity steps, then clip the accumulated gradient norm at 1.0, perform one AdamW optimizer update, and zero gradients. Thus each update has an effective batch of 32 complete tracks. No activation, hidden state, decoded cache, detached feature, or autograd graph may cross a training step or be reused after parameters change. The sampler must emit exactly 640,000 complete-track draws over 20,000 optimizer updates, except a fail-closed abort; there is no partial final accumulation.

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

Stride is 32 retained samples. Starts are `0,32,...` while a full window fits; append `max(K-64,0)` if absent. For `K<64`, use start 0 and right-pad samples/masks; padding creates no interval or mass. During training, windows read only the current step's single fresh identity graph; during evaluation, they read the one complete frozen identity pass. For every valid global interval `j`,

\[
D_j=\sum_{b\ni j}v_j^{(b)}w_{j-b},\qquad
\bar m_j=\frac{\sum_{b\ni j}v_j^{(b)}w_{j-b}\,g_j|\delta_j^{(b)}|/(2\pi)}{D_j}.
\]

Assert finite `D_j >= 1e-3` before division. There is no epsilon rescue or zero-weight boundary. Invalid intervals contribute neither numerator nor denominator. Decode once per supplied identity, `hat C_p=sum_j bar m_j`; round-half-to-even is secondary and occurs only after this sum. Because fresh-graph increments are read-only views, overlap cannot duplicate recurrence updates. Grid-origin offsets 0, 8, 16, and 24 must conserve synthetic phase mass within `1e-6` relative error.

## 6. Frozen Numerical Implementation Table

| Interface | Frozen value |
|---|---|
| Input | each training step samples `B=8` complete collapsed identities; right-padded `[8,T_max,17,3]` float32 pose plus frozen joint/person/frame/cell/padding masks; 68 pose/mask channels |
| Clock/integral | int64 `sampled_frame_indices`; float64 overlaps/integrals |
| Joint validity | confidence `>=0.20`; at least 8/17 joints per valid frame |
| Eligibility | unambiguous association; pose coverage `>=0.80`; collapsed feature coverage `>=0.60`; never label-filtered; all 9 components nonempty |
| Duplicate agreement | masks exact; normalized xy max `<=1e-5`; confidence max `<=1e-6`; disagreement invalidates clock |
| Encoder/head | Conv1d 68→128 k5 p2; Conv1d 128→128 k5 p2; GRU 128; linear 128→2; 225,026 parameters |
| Recurrence | fresh clean and warp graphs each training step; each full identity clock forwarded once per branch; zero at identity start; invalid/padded hold; overlap windows are same-graph loss views; no cross-update/stale cache |
| Phase normalization | `z=u/(||u||_2+1e-8)` |
| Mask order | duplicate agreement → collapse/sort → joint/frame validity → adjacent cell → warp support → alias → NOLA |
| Pseudo-cycle selector | nested 64/128; FFT 256; candidates 4–`min(128,floor(span/2))`; >=16 pairs; ACF >=0.25; tie 0.02 |
| Weak views | coordinate jitter `sigma=0.01`, clip `0.03`; 10% joint dropout; retain >=8 joints; clock unchanged |
| Training selector gate | 1,000 fixtures, >=250 symmetric; total >=95%; half/double <=2%; symmetric half/double <=5%; real view/nested agreement 10%; margin 0.10 |
| Evaluator period/harmonic gate | raw source-frame half-open `[s,e)` boundaries only; global nonempty/well-formed assertion; `P_eval=NumPy float64 median(e-s)`; catalog/source/join/table hashes; no count/density/periodicity/bbox/fallback or 320-grid inverse; post-freeze `h={0.5,1,2}`, 10%, video-first, `H1>=0.90`, others each `<=0.05`, no correction |
| Windows/NOLA | 64 samples/63 intervals; stride 32; floor `1e-3`; deterministic final/right padding; denominator `>=1e-3` |
| Warp family | 3 affine segments; breakpoints uniform `[0.20,0.40]`, `[0.60,0.80]`; raw slopes log-uniform `[0.5,1.5]`, endpoint-normalized; accept final `[0.4,2.0]` within 128 draws |
| Pause/reversal | 20% set middle raw slope to 0 then normalize remaining slopes; reversal is global endpoint swap, never local direction mixing |
| Warp loss | circular Huber beta 0.10 rad; alias margin `0.05pi`; min 32 valid intervals/window and 80% track coverage |
| Optimizer | AdamW lr `3e-4`, weight decay `1e-4`; 8 full identities/training step; within-identity then batch mean; one backward/training step; accumulate exactly 4 fresh steps for 32 effective tracks/update; clip 1.0 immediately before update; 20,000 updates; no metric early stop |
| K4 inputs | clock/no-pose: zero `x/y/conf`, preserve frozen joint/person/frame masks plus model-visible `q,L`; mask-only: zero pose and model-visible `q,L`, preserve identical masks and mask-derived valid fractions; opaque keys/slots route only; all warp/crop/resampler metadata loss/audit-only; same 225,026 parameters and budget |
| Seeds | 20270815, 20270816, 20270817; no replacement |
| Bootstrap | 9 asserted nonempty development components; B=10,000; PCG64 seed 20270815; linear central percentiles |
| Hard budget | 59 total training jobs; 552 GPU-hours absolute maximum |

Warp and diagnostic schedules are serialized before evaluator access. Engineering tolerances are invariants and no development metric selects them.

## 7. Gates and Training Plan

### Gate 0 — isolation and frozen population

Create split shards/vault, opaque join table, source-component manifest, count-blind eligibility manifest, checksums, forbidden-key/path receipts, and the hashed eligible video/person/component table. Assert all original nine development components retain at least one eligible video/person. Isolation failure invokes K6; an empty component invokes K7 before any evaluator label opens.

### Gate 1 — deterministic numerical fixtures

Run duplicate collapse, invalid-cell interpolation, signed overlap, wrap, pause, reversal, alias, single-commit recurrence, invalid-state hold, full-identity padding/masking, within-identity averaging, four-step gradient accumulation, one-backward-per-step, stale-cache rejection, read-only overlap, final window, positive denominator, grid origin, one decode, raw-period extraction/classification/global-failure, K4 zero/input-routing/parameter-equality, normalized AvgMAE, and bootstrap fixtures. These are not efficacy results.

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
| B | trained clock/no-pose model + trained mask-only model | 6 | K4 decision controls; run only if A passes K1 |
| C | direct signed-frequency regression; global-warp-only; unsigned-reversal ablation | 9 | same-backbone mechanism falsification; run only if A passes K1 and B survives K4 |
| D | PAMS; Track-PAMS; SimPer-style; CycleCL-style; DeepPhase-style; generic time-equivariant | 18 | faithful/same-backbone closest priors; run only if A–C survive K1/K3/K4; this stage adjudicates K2 |
| E | masked geometric TSSM control | 3 | optional control-only if frozen matrix requires it; never fallback/contribution |

`PILOT_PERIOD_SCHEMA_ADDENDUM.md` correctly records the earlier crop/resampler **nuisance-only** interface as `OPEN_NUISANCE_SCHEMA`, because no legitimate persisted nuisance fields exist. This revision does not pretend that interface was closed: it removes that input branch and reassigns the same already-budgeted three-seed slot to the strictly weaker, fully auditable **mask-only** control below. Both K4 models use the exact 225,026-trainable-parameter Conv/GRU/head, full-identity sampler, optimizer, 20,000-update cap, and three fixed seeds; neither receives tuning jobs. Their input tensors are frozen as follows.

- **Clock/no-pose:** set every model pose value `x`, `y`, and confidence to numerical zero after constructing the frozen joint/person/frame masks. Preserve those masks, `q=sampled_frame_indices`, `L=source_length`, the per-frame valid-joint fraction, and the per-track valid-frame fraction. Clock scalars are model-visible only through a fixed, non-trainable additive code after Conv1: `q_norm=q/max(L-1,1)` multiplies a fixed 128-vector with entries `(+1,-1)` alternating, and `L_norm=log2(max(L,1))/16` multiplies a fixed 128-vector whose first 64 entries are `+1` and last 64 are `-1`. Mask-derived fractions use two further fixed orthogonal sign vectors generated once from seed 20270815 and stored in the fixture. No trainable parameter is added.
- **Mask-only:** use the identical zero pose and frozen masks/fractions, but replace the model-visible `q_norm` and `L_norm` by exact zeros before the same fixed-code interface. It therefore exposes only missingness/length-through-mask structure, not absolute or normalized clock values.
- **Both:** the opaque sample key and supplied-slot index may address records, pair views, and restore output order only; they cannot be cast to a model tensor, hashed into a feature, looked up in an embedding, or condition any normalization. Warp family/schedule, breakpoints, slopes, target-to-source map, interpolation/resampling kernel, augmentation seed, crop origin/scale/parameters, and all other crop/resampler metadata may construct losses, corruptions, and audit receipts only and are never model input. The fixed-code vectors, zeroing assertions, model-input byte hashes, parameter-count equality, and equal-budget receipts are mandatory.

Matched-time pose shuffle and hidden-absolute-time checks are deterministic K4 evaluation corruptions. The unseen-resampler check is a separate diagnostic applied to frozen model predictions/traces under a predeclared held-out resampling kernel; it is not an input to either trained K4 model and is not conflated with the mask-only arm. Randomized crop origins are audit corruptions only. None is an extra trained model.

Stages A–E contain at most 42 decision-bearing full jobs. The seven learned comparator families in D/E receive at most two training-only 5,000-step tuning jobs each, never evaluator labels, adding 14 jobs. Together with three sanity jobs, the absolute maximum is **59 jobs**. A full job is capped at 12 GPU-hours, each tuning job at 3, and all sanity jobs together at 6, so the hard ceiling is **552 GPU-hours = 42×12 + 14×3 + 6**. No over-budget continuation is permitted.

If A fails, stop after at most 9 jobs/78 GPU-hours and save at least 50 jobs/474 GPU-hours. If B fails K4, stop after at most 15 jobs/150 GPU-hours and save at least 44 jobs/402 GPU-hours. If C fails K3 or an already-reached K1/K4 condition, stop after at most 24 jobs/258 GPU-hours and save at least 35 jobs/294 GPU-hours. Only after A–C survive K1/K3/K4 does Stage D run and adjudicate K2. Later stages stop on the first applicable K-rule. Comparator provenance, mappings, counts, schedules, and receipts are mandatory.

For learned comparators, use the same paired nine-component bootstrap. A comparator is separated only if the lower endpoint of `M_comparator-M_treatment` is strictly positive; otherwise it matches within uncertainty and K2 applies.

## 10. Three Visible Validation Blocks

### Block 1 — primary paired law versus augmentation

One table reports three seed-paired piecewise-warp normalized video-first AvgMAE values, mean/SD, `D`, `R`, and the exact nine-component CI. Clean AvgMAE and secondary metrics are labeled diagnostics. This block alone decides K1.

### Block 2 — staged mechanism, shortcut, and closest-prior falsification

Report the trained clock/no-pose and mask-only K4 decisions, then the deletion/direct-frequency/global-warp/unsigned-reversal checks, then only the reached closest-prior stage. Clock/no-pose or mask-only retention above 20% of a positive treatment gain, matched-time shuffle preservation, or the separate unseen-resampler diagnostic erasing it invokes K4. Detailed fixtures remain receipts/supplement. This remains one visible block despite its complete execution ledger.

### Block 3 — identity isolation

With a frozen backbone, warp exactly one supplied identity while untouched track tensors and clocks remain byte-identical. Compare private state with shared-scene state and shuffled-key state. Report untouched-person phase-response change, continuous count-mass change, and rounded count change with component intervals. Stateless/reset/swap variants remain debugging receipts unless contamination appears. This is a diagnostic, never private-memory novelty and never a rescue for Block 1.

## 11. Failure Modes and Frozen K1–K9

- **K1 — primary effect/uncertainty:** Survive only if point relative piecewise-warp normalized video-first AvgMAE improvement is at least 5% and the paired nine-component absolute-`D` CI lower endpoint is greater than zero. `bar M_c=0` fails.
- **K2 — closest prior:** If PAMS/Track-PAMS, SimPer-style, CycleCL-style, DeepPhase-style, or generic time-equivariant learning matches/exceeds treatment within the frozen rule, kill the residual claim. Private state/TSSM cannot rescue it.
- **K3 — objective redundancy:** If deleting `L_WI`, direct signed-frequency regression, global resampling only, or replacing reversal's signed overlap with unsigned overlap matches treatment, conclude the local signed objective is redundant.
- **K4 — shortcut/artifact:** If a trained clock/no-pose or mask-only model retains more than 20% of a positive gain, the separate unseen-resampler diagnostic erases it, or matched-time shuffling preserves it, reject as shortcut/artifact. Any forbidden metadata/key embedding or unequal parameter/budget receipt also fails K4.
- **K5 — pseudo-cycle or execution failure:** If training-only selector fixtures/stability, the frozen evaluator harmonic gate, support/alias coverage, recurrence single-commit receipts, or NOLA mass conservation fails, kill the phase-mass claim. Do not correct harmonics or add modules.
- **K6 — integrity:** If feature/vault separation, positive-count assertion, source isolation, permissions, forbidden keys/paths, hashes, frozen predictions, or untouched evaluation cannot be shown, stop with no claim.
- **K7 — population/clock/missingness:** If any original development component is empty after count-blind eligibility, or duplicate conflicts, missing pose, timing, or alias masks invalidate the estimand, rebuild the cache or restrict work to engineering diagnostics.
- **K8 — novelty collision:** The independent receipt `idea-stage/TWCRAC_ADJUDICATION.json` has SHA-256 `5a1e7b8c1048a8884690f2d234d413064d473ead1eb1deca69ab734ca2e5ed1e`, verdict `BLOCKED`, and `k8.decision=NOT_PASSED_BLOCKED`; therefore K8 is **NOT PASSED** and claim freeze is **BLOCKED**. This receipt does not establish distinctness and does not close K8. Obtain and independently inspect the complete TWCRAC primary-source method before claim freeze. If it contains the same discrete local warp-phase law or equivalent contribution, kill or materially re-anchor novelty and rerun novelty review. Abstract-only evidence cannot pass K8.
- **K9 — forbidden fallback:** If the primary fails and masked geometric TSSM is proposed as an automatic replacement, stop. It remains a control absent a new anchor and fresh novelty review.

## 12. Experiment Handoff and Current Status

**Required artifacts:** feature shards, evaluator vault, eligibility/component/join manifests, forbidden-key/path receipts, pseudo-cycle/harmonic fixtures, post-freeze evaluator harmonic receipt, recurrence single-commit trace, NOLA/overlap/normalized-scorer/bootstrap fixtures, three-seed run receipts, raw per-person predictions, phase traces, comparator provenance, and an eventual independent TWCRAC full-method adjudication.

**Run order:** Gate 0 isolation/nonempty components → Gate 1 numerical fixtures → Gate 2 pseudo-cycle stability → Gate 3 sanity → Stage A primary → post-freeze harmonic gate and K1 → Stage B K4 trained controls → Stage C mechanism checks and K3 → only if A–C survive K1/K3/K4, Stage D priors adjudicate K2 → optional Stage E control → identity diagnostic → obtain complete TWCRAC full text and rerun K8 adjudication. Stop at the first K-rule.

**Forbidden evidence:** v46/v62/v63 efficacy, sealed test data, predicted-track claims, official MultiRep main-result language, published end-to-end numbers as same-protocol rankings, post-hoc harmonic correction, and any result without a complete frozen receipt.

**Current status:** Round-3 revised proposal only; not implemented; no eligible results; same-family provisional. Training is blocked until Gates 0–2 pass. The present TWCRAC receipt is `BLOCKED`/`NOT_PASSED_BLOCKED`, not closure; claim freeze remains blocked until reached experiment/audit/result-to-claim gates pass and a complete-primary-source K8 adjudication passes.
