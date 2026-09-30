# Round 2 Refinement: TempoRAC

## Problem Anchor

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

The five bullets above are immutable.  A reviewer request that replaces the
fast/medium/slow local router, response fusion, NOLA, or one final per-track
decode with WARP-PHASE, a missing-pose set, scene-level counting, or an easier
single-person task is research drift.

## Anchor Check

- **Original bottleneck:** The scientific question is whether one fixed,
  window-local relative-tempo cue selects useful shared response scales for
  asynchronous identities and within-track pace changes while state remains
  identity-private and one reconstructed response is decoded once.
- **Why the revised method still addresses it:** The graph remains one shared
  encoder, three matched slow/medium/fast response experts, cue-only soft
  routing, probability-level response fusion, positive normalized overlap-add,
  and one fixed connected-component decode per identity. The teacher remains
  an evaluator-certified target prerequisite and is absent at inference.
- **Reviewer suggestions accepted without drift:** Numerical seam ownership,
  per-track target certification, full-cadence acquisition eligibility, exact
  probability-level capacity matching, and trained-output pulse fixtures make
  the existing graph executable; none adds a trainable module.
- **Reviewer suggestions rejected as drift:** WARP-PHASE, a missing-pose set,
  a scene counter, an easier single-person task, an integral decoder, a learned
  router, per-person model copies, or human-guided target repair would replace
  the anchored object and remain forbidden.

## Simplicity Check

- **Dominant contribution after revision:** Cue-only, identity-indexed,
  window-local relative-tempo routing selects specialized shared response
  scales under within-track drift.
- **Only supporting mechanism check:** The same three shared branches show
  diagonal tempo specialization on unseen resamplers against an exactly
  capacity-matched unrouted control.
- **Correctness prerequisites, not contributions:** `certify_target(track)`,
  identity-private state, exact positive NOLA, and one fixed peak pass may kill
  the route but do not become parallel claims.
- **Components kept out:** No `L_cont`, canonical PAMS-TCC initialization, ACF
  fusion, learned router, GRU, integral decoder, fourth expert, learned
  threshold, hard period, expert voting, or WARP/pivot substitution is added.
- **Paper compression:** Internal G/K receipts remain fail-closed, but only
  three evidence blocks are paper-visible: response-unit/decoder eligibility,
  frozen-response routing mechanism, and supplied-track efficacy/stationarity.

## Changes Made

### 1. Replaced scattered teacher rules with one target certificate

- **Reviewer said:** A suite-level winding pass and a 1% population
  negative-edge allowance cannot guarantee one pulse on each target-bearing
  track.
- **Action:** One evaluator-owned `certify_target(track)` now owns acquisition
  support, duplicate conflicts, landmarks, origin, degree, collision,
  compensated unwrapping, seam snapping, crossing counts, resampler equality,
  and abstention. Every retained-landmark traversal must independently pass.
- **Impact:** Population tolerances may select a teacher artifact but never
  override an individual abstention or serialize an uncertified target.

### 2. Closed source-clock, warp, and phase numerics

- **Reviewer said:** Positive clock gaps cannot prove that no cycle is hidden,
  raw floors leave seam ownership implementation-dependent, and the two-chord
  tangent is not exactly warp invariant.
- **Action:** The supported acquisition domain is full cadence with a frozen
  five-source-frame minimum physical period and a one-frame maximum gap; every
  other gap splits support and makes the natural identity ineligible. Synthetic
  warps use a frozen target-to-clean map and full-cadence output clock. Phase is
  accumulated in compensated float64, snapped by `S_tau`, and tested at three
  tolerances. The tangent is called an approximate local direction estimate,
  with full canonical-phase and pulse equality required on unseen resamplers.
- **Impact:** No endpoint test is presented as proof of an unobserved interval,
  and no gap is bridged, unwrapped, decoded through, or imputed.

### 3. Made response alignment, control capacity, and NOLA exact

- **Reviewer said:** Edge/sample alignment was implicit, the control moved the
  sigmoid after averaging, and NOLA retained a threshold-affecting epsilon.
- **Action:** Edge `(t,t+1)` uses the causal branch output at sample `t+1`;
  initial, final, invalid, and padding behavior are fixed. The capacity control
  uses the identical three sigmoid branches and exact uniform probability
  fusion. NOLA first proves a finite denominator of at least `1e-3` and then
  divides by that exact denominator with no epsilon.
- **Impact:** Routing is the only difference in the decisive frozen-response
  comparison, and reconstruction cannot bias a threshold decision.

### 4. Bound the pulse and causal decision gates

- **Reviewer said:** G3 used undefined words such as `beats` and `gain`, while
  G4 did not bind the trained bank or define event/component association.
- **Action:** Source-unit aggregation, paired bootstrap bounds, the strongest
  nonlocal margin, shuffle-removal fraction, shortcut retained-gain fraction,
  boundary error, and every zero-denominator rule are frozen. G4 has one schema
  with deterministic operator fixtures and frozen unseen-resampler bank
  outputs, all 32 starts, supported spacings beginning at four, exact
  event-component degrees, and worst positive/negative margins.
- **Impact:** The gates are numerical falsifiers rather than qualitative
  comparisons, without adding an experiment block.

### 5. Removed the authorization deadlock

- **Reviewer said:** G0 depended on a packer that the refinement did not
  authorize, and natural-vault checks appeared before all selection-bearing
  artifacts were frozen.
- **Action:** This document grants no data or server access. It states a future
  trusted-packer authorization boundary and one numeric S0--S4 sequence aligned
  with G0--G5 and K0--K7. The evaluator vault remains unopened until the
  teacher, targets, models, predictions, comparison arms, configuration, and
  evaluator program are frozen and hashed.
- **Impact:** A future execution can proceed without circular permission or
  label-derived selection; this refinement itself remains specification-only.

### 6. Preserved one paper and corrected feasibility

- **Reviewer said:** The internal contract could eclipse the routing claim and
  the 15--23 day estimate was optimistic.
- **Action:** The estimate is now 20--30 focused engineer-days under the same
  hard cap of 100 RTX A6000-hours. The proposal exposes one dominant routing
  claim, one specialization support check, and at most three paper-visible
  evidence blocks.
- **Impact:** The method stays narrow, and novelty, title, implementation,
  server, result, paper, and submission authority remain explicitly ungranted.

## Revised Proposal

# Research Proposal: TempoRAC (Working Name Only)

## Problem Anchor

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

The five bullets above are immutable.  A reviewer request that replaces the
fast/medium/slow local router, response fusion, NOLA, or one final per-track
decode with WARP-PHASE, a missing-pose set, scene-level counting, or an easier
single-person task is research drift.

## Technical Gap

MultiCounter and MultiCounter+ occupy multi-person repetition counting and
person-indexed output. PAMS occupies pose-driven periodic supervision and
period-adaptive consistency. RepNet, HTRM-Net, and TWCRAC occupy time-varying
or local temporal periodicity; generic mixtures occupy shared expert routing;
WOLA/NOLA occupies overlap reconstruction. No named component is novel, and
full TWCRAC overlap remains unresolved.

The remaining operational interaction is narrower. Different identities may
move at different rates, and one identity may change rate inside its track. A
global selector cannot express that interaction; per-person model copies do
not test shared specialization; independent window counts double-count seams;
and a learned router may ignore tempo. The smallest causal test is therefore a
fixed, cue-only relative local-tempo gate over three shared response scales,
with identity-private state and response reconstruction before one decode.

This test requires a training pulse with the physical unit of one supported
primitive orbit traversal. Generic view consistency and reconstruction do not
fix winding or a seam. A frozen teacher plus the evaluator-owned per-track
certificate below is therefore a prerequisite, not an inference component or
a second paper claim. Certification failure kills the route; human labels,
WARP-PHASE, an integral decoder, or a backup pivot cannot repair it.

## Method Thesis and Contribution Focus

- **One-sentence thesis:** Cue-only, identity-indexed, window-local
  relative-tempo routing selects specialized shared response scales under
  within-track drift.
- **Dominant contribution:** A same-response causal intervention testing local
  relative-tempo selection among three matched shared response branches.
- **Only supporting mechanism:** Diagonal branch specialization by relative
  tempo on unseen resamplers against an exactly matched unrouted control.
- **Eligibility and plumbing:** The teacher certificate, private state,
  positive NOLA, and one fixed decoder are necessary contracts, not claims.
- **Permitted supervision statement:** The counting objective reads no human
  per-person count or period annotations. Supplied-track, pose-estimator, and
  synthetic-warp provenance must be disclosed separately.
- **Explicit non-claims:** No component novelty, broad priority, official
  MultiRep result, predicted-track result, title freeze, paper readiness, or
  submission claim is available.

## Complexity Budget

- **Teacher-stage trainables:** one fixed-size static-code MLP, one memoryless
  phase MLP, and one no-bypass phase-conditioned reconstruction MLP.
- **TempoRAC trainables:** one shared width-64 causal local encoder and exactly
  three equal-parameter width-64 causal response branches.
- **Frozen operators:** pose normalization, support masks, window grid, NUDFT,
  weighted-median reference, soft gate, taper, exact NOLA, and decoder.
- **Excluded first-route additions:** `L_cont`, canonical PAMS-TCC
  initialization, ACF fusion, learned routing, GRU, a fourth expert, load or
  entropy loss, learned threshold, hard period, period NMS, expert voting,
  integral decoding, WARP-PHASE, and missing-pose pivots.

## System Graph and Seven Invariants

```text
future-authorized trusted feature shard
  -> duplicate conflict check and collapse
  -> full-cadence support segmentation and COCO-17 normalization
  -> clock-blind teacher, target creation only
       -> evaluator-owned certify_target(track)
       -> frozen half-open pulse e*
  -> one shared causal encoder per identity
       -> slow sigmoid response branch, cached once
       -> medium sigmoid response branch, cached once
       -> fast sigmoid response branch, cached once
  -> independent window-local irregular-clock NUDFT distributions
  -> identity-local weighted-median reference -> fixed cue-only soft gates
  -> probability-level fusion inside each window
  -> exact positive normalized overlap-add over unique edges
  -> one threshold-0.5 connected-component pass per identity
  -> person-indexed count vector
```

The implementation is TempoRAC only if all seven invariants pass:

1. One encoder parameter object is shared by every identity.
2. The same three slow/medium/fast branch parameter objects are shared by all
   identities, and their trainable parameter counts are exactly equal.
3. Every causal buffer and reconstruction accumulator is private to one
   identity; branch buffers are additionally private to one branch.
4. A raw window cue reads only that window's pose, masks, and relative source
   clock; only the documented identity-local reference may couple its final
   gate to other reliable windows of the same identity.
5. Gates lie on the simplex and fuse continuous sigmoid responses before any
   threshold or connected-component operation.
6. Every valid unique edge proves a finite NOLA denominator at least `1e-3`;
   invalid and padding edges contribute exact zero.
7. The decoder is called once for every supplied `person_valid` identity,
   including one empty call for an ineligible identity; no window count is
   decoded or summed.

## Authorization, Firewall, and Trusted Packing

This Phase 3 refinement freezes a method contract only. It does not authorize
the trusted packer, source data, evaluator vault, server, implementation,
training, test execution, result access, or paper work. A distinct future
authorization must name the source and allow one trusted packer process to
deserialize the label-mixed v44 artifact. Without that authorization, G0 is
not runnable and no data-bearing stage may begin; only S0 contract work and
synthetic CPU fixtures are permitted.

After that future authorization, the packer emits physically separate files:

| Artifact | Exact contents | Allowed consumer |
|---|---|---|
| Feature shard | `pose_xyc: float32[P,U,17,3]`, `joint_valid`, `frame_valid`, `person_valid`, nondecreasing `source_index`, opaque join token, and local slot | model/training process |
| Evaluator vault | joined human counts, density, period/boundary fields, boxes, mappings, and evaluator handles | separately permissioned evaluator only |
| Audit manifest | source provenance, canonical-source component, converter and artifact hashes, and join receipts | firewall/split auditor only |

The model loader accepts only the feature-shard whitelist and rejects unknown
or forbidden keys before array payload deserialization. Identity names,
absolute source paths, canonical source IDs, source length, counts, periods,
boundaries, density, boxes, metrics, provenance, and warp metadata are absent
from model tensors. Opaque keys and local slots address records and private
buffers only and are never embedded.

The fixed K3 nuisance tensor has 37 channels: 17 joint-valid bits, 17 clipped
joint confidences, one frame-valid bit, one centered `log1p` relative clock-gap
channel, and one synthetic interpolation-weight channel. It is available only
to named attacks through a frozen `37 x 269` Rademacher projection scaled by
`1/sqrt(37)`; it never enters the canonical graph. The separate warp-metadata
attack deliberately receives synthetic log derivative only to test a
forbidden bypass.

## Input, Duplicate, Pose, and Acquisition Contract

Raw source indices must be nondecreasing; any decrease abstains. For a
contiguous duplicate-clock run at source index `q`, each observation is
temporarily hip-rooted and divided by the median of its available positive
finite shoulder width, hip width, and shoulder-midpoint-to-hip-midpoint
distance; no available positive row scale abstains. For every observation pair
`(a,b)`, let `J_ab` be common finite joints, `w_j=min(conf_aj,conf_bj)`, and
define the confidence-normalized disagreement

\[
d_{ab}=\left(\frac{\sum_{j\in J_{ab}}w_j
\|\widetilde x_{aj}-\widetilde x_{bj}\|_2^2}
{\sum_{j\in J_{ab}}w_j}\right)^{1/2}.
\tag{F1}
\]

A run needs at least eight common joints and a positive denominator. It may be
collapsed only when every pair has `d_ab <= 0.02`; byte-identical duplicates
therefore pass. Otherwise the identity abstains, and the packer preserves a
receipt containing the opaque record token, source index, run size, maximum
disagreement, support count, and source hash, but no pose payload. A source
clock that reappears noncontiguously also abstains. Passing runs collapse
coordinates by confidence-weighted mean, confidence by maximum clipped to
`[0,1]`, and masks by logical OR. Zero total confidence makes the joint invalid.
Silent averaging of conflicting observations is forbidden.

After collapse, a frame root is the mean of available hip joints 11 and 12. A
frame needs a root and at least eight valid joints. The fixed track scale is
the median of all positive finite shoulder widths, hip widths, and
shoulder-midpoint-to-hip-midpoint distances on eligible frames. A missing or
`<=1e-6` scale abstains. Valid coordinates are root-relative, divided by this
scale, clipped to `[-4,4]`, and invalid coordinates are exact zero. The 16
ordered bones are

```text
(0,1) (0,2) (1,3) (2,4) (5,6) (5,7) (7,9) (6,8)
(8,10) (5,11) (6,12) (11,12) (11,13) (13,15) (12,14) (14,16)
```

The supported physical period is frozen at `P_min=5` source-frame intervals.
The maximum supported adjacent source-clock gap is
`q_gap=max{d in positive integers: d/P_min < 0.25}=1`. Thus full-cadence source
tracks are preferred and every `Delta q != 1`, missing clock, invalid frame,
or unsupported coordinate edge splits support. Windows, teacher unwrapping,
targets, NOLA, and connected components never cross a split. Because endpoints
cannot prove what occurred inside a gap, a natural identity containing any
interior acquisition split, including a missing/unsupported clock, invalid
frame, or unsupported coordinate edge, is ineligible for a count-bearing pilot
prediction and receives one empty decoder
call plus a receipt. Synthetic or natural support left after all other splits
must contain at least two retained landmarks and one complete certified
traversal; otherwise it abstains. No gap is bridged and no hidden cycle is
inferred.

For `T` unique samples, windows contain 128 sample slots and 127 edge slots.
Starts are `0,32,64,...` while a full window fits, followed by
`max(0,T-128)` if absent; short eligible segments use one right-padded start
zero. Padding is exact zero. Every valid edge is covered. `T=0` is an invalid
record; `T<=1`, no valid edge, or any acquisition ineligibility creates one
empty all-invalid response, one decoder call, zero output, and an abstention
receipt. It cannot enter training loss.

## Synthetic Warp and Source-Clock Semantics

Let clean pose be samples of a frozen continuous clean trajectory `X0(u)` on
clean source coordinate `u`. A synthetic warp is the target-to-clean map
`psi(q)`: for each integer output-clock sample `q=0,...,Qw-1`, `psi(q)` gives
the clean coordinate sampled at that target time. `psi` is strictly increasing,
piecewise `C1`, and satisfies `1/4 <= psi'(q) <= 4`. The output source clock is
always full cadence, `q_w[q]=q`; no warped metadata replaces that clock.

Training views use linear interpolation of clean coordinates and confidence;
a joint is valid only when both interpolation endpoints are valid. Held-out
views use frozen PCHIP and 63-tap Kaiser-windowed sinc (`beta=8.6`) coordinate
interpolation, with the same endpoint-valid mask rule and linear confidence.
The sinc kernel is normalized over available full-valid taps and abstains if
fewer than 32 valid taps remain. The clean boundary uses reflection without
duplicating the endpoint. Interpolator identity is evaluator metadata only.

The clean-to-target instantaneous frequency transformation and its sign are

\[
f_w(q)=f_0(\psi(q))\psi'(q),\qquad
\nu(q)=\log\psi'(q),
\tag{F2}
\]

so positive `nu` means faster motion on the output clock. The generator accepts
a view only when its instantaneous output-clock physical period `1/f_w(q)`
remains in `[5,128]` everywhere on retained support; this is the same domain as
the cue grid and the `P_min` acquisition assumption. For window `ell`,
`nu` is integrated with trapezoidal output-clock quadrature, the same Hann
support as the cue, and valid edge support. Let `b_{ell e}` be those
nonnegative weights. Let `alpha_e` be positive clean primitive-traversal
progress on edge `e`, pulled back through `psi`, for later loss weighting. Let
`lambda_e=||g_{e+1}-g_e||_2` be clock-blind geometry arc length and `m_e` the
number of reliable windows covering that edge. The reference weight used by
both the oracle and canonical cue is
`beta_ell=gamma_ell sum_{e in ell} lambda_e/m_e`; an edge with `m_e=0` is absent
from every reference sum. Thus overlapping windows allocate each physical
geometry-arc element once rather than weighting slow, densely sampled regions
by their window count. The oracle relative shift and expert responsibility are

\[
\zeta^*_{i\ell}=
\frac{\sum_e b_{i\ell e}\nu_{ie}}{\sum_e b_{i\ell e}}
-\operatorname{wmed}_{j\in\mathcal R_i}
\left(\frac{\sum_e b_{ije}\nu_{ie}}{\sum_e b_{ije}};\beta_{ij}\right),
\quad
p^*_{i\ell k}=\operatorname{softmax}_k
\left[-\frac{(\zeta^*_{i\ell}-\mu_k)^2}{2\sigma_g^2}\right].
\tag{F3}
\]

Every denominator and the total `beta` weight must be positive, and
`mathcal_R_i` must contain at least three reliable windows; otherwise the
example abstains. The weighted median sorts by
`(value, window_start)` and takes the first cumulative weight reaching half
the total. This quadrature and traversal-normalized construction does not
average by output sample count. `psi`, `psi'`, `nu`, clean coordinates,
interpolator name, and `zeta*` are visible only to synthetic loss/evaluation
builders and never to the teacher, encoder, branch, router, NOLA, or decoder.

## Primitive-Orbit Teacher

### Clock-blind state and reparameterization-safe static pooling

Let `p_t` be 34 normalized joint coordinates, `b_t` the 32 bone coordinates,
`g_t=[p_t,b_t]`, and `c_t` the 17 confidences. On coordinates supported at
`t-1,t,t+1`, define the approximate local direction estimate

\[
\widehat v_t=\frac{v_t^-/\|v_t^-\|_2+v_t^+/\|v_t^+\|_2}
{\|v_t^-/\|v_t^-\|_2+v_t^+/\|v_t^+\|_2},\quad
v_t^-=g_t-g_{t-1},\quad v_t^+=g_{t+1}-g_t.
\tag{F4}
\]

A nonfinite norm, either chord norm below `1e-8`, resultant norm below `1e-8`,
missing two-sided support, or endpoint makes that direction invalid and exact
zero. The formula is not claimed to be exactly warp
invariant; its admissibility is decided only by the unseen-resampler phase and
pulse equality inside `certify_target`. The 215-channel teacher state is
`y_t=[p_t,b_t,vhat_t,c_t,m^g_t,m^v_t]`: 149 continuous channels and 66 mask
channels. It contains no clock, displacement magnitude, fixed-frame lag,
length, identity, window, or warp variable.

Static pooling uses pose-landmark-to-pose-landmark traversals found from
geometry alone. For traversal `j`, let `lambda_e=||g_{e+1}-g_e||_2` on full
geometry support and `L_j=sum_e lambda_e`. For each continuous channel `d`,
the traversal-normalized first and second moments, followed by equal traversal
pooling, are

\[
m_{jd}=\frac{1}{L_j}\sum_{e\in j}\frac{\lambda_e}{2}
(y_{ed}+y_{e+1,d}),\quad
q_{jd}=\frac{1}{L_j}\sum_{e\in j}\frac{\lambda_e}{2}
(y_{ed}^2+y_{e+1,d}^2),\quad
\bar m_d=K^{-1}\sum_jm_{jd},\quad
s_d=\sqrt{\max(0,K^{-1}\sum_jq_{jd}-\bar m_d^2)}.
\tag{F5}
\]

Every `L_j` must be positive. This is the single static pooling rule: a
clock-blind geometry-arc-length line integral, normalized within traversal and
then equally across traversals. It is invariant to subdivision of the same
piecewise-linear geometry and does not weight pauses or dense sampling. Full
cross-resampler equality remains a certificate requirement. `[mbar,s]` is the
298-vector input to `Linear(298,128)-GELU-Linear(128,32)` and unit-L2
normalization. The static-MLP output norm must be finite and at least `1e-8`;
otherwise the example abstains, and a passing vector is divided by its exact
norm. No additional trainable pooling module exists.

The phase MLP is
`Linear(247,256)-GELU-Linear(256,128)-GELU-Linear(128,2)` on `[y_t,c_i]`.
Its output norm must be finite and at least `1e-8`; otherwise the example
abstains, and a passing output is divided by its exact norm to lie on `S1`. The
decoder is
`Linear(34,256)-GELU-Linear(256,256)-GELU-Linear(256,149)` on `[z_t,c_i]`.
It has no skip, residual, time, mask, raw-state, or alternate input.

For a valid adjacent edge, signed circular phase increment in cycles is

\[
\delta_t=\frac{1}{2\pi}\operatorname{atan2}
\left(z_{t,x}z_{t+1,y}-z_{t,y}z_{t+1,x},
z_{t,x}z_{t+1,x}+z_{t,y}z_{t+1,y}\right)\in(-0.5,0.5].
\tag{F6}
\]

The teacher loss is

\[
\mathcal L_{teach}=\mathcal L_{inv}+\mathcal L_{view}
+0.25\mathcal L_{orient}+0.25\mathcal L_{alias},
\quad
\mathcal L_{orient}=\operatorname{mean}\operatorname{ReLU}(-\delta_t),
\quad
\mathcal L_{alias}=\operatorname{mean}\operatorname{ReLU}(|\delta_t|-0.25).
\tag{F7}
\]

`L_inv` is mask-normalized Huber reconstruction of the 149 continuous channels
with transition 0.05; `L_view` is mean `1-dot(z,z')` at exact synthetic
correspondences. Each denominator must be positive or the batch is rejected.

### The single evaluator-owned `certify_target(track)` contract

`certify_target(track)` is the only operation allowed to serialize a pulse.
It is evaluator-owned, deterministic CPU float64, receives a frozen teacher
artifact and one collapsed track, and returns
`{eligible, abstention_reason, supported_segments, landmarks, origin, delta,
crossings, pulse, receipt_hash}`. The model and optimizer cannot call it with
human fields. The contract executes these steps in order, and any failure
returns no target bytes:

1. **Acquisition and duplicates.** Apply the exact duplicate tolerance and
   `P_min=5`, `q_gap=1` rules above. A natural track with any interior
   acquisition split abstains. Every pooled sample in a candidate traversal
   has all 66 geometry coordinates and their corresponding two-sided direction
   channels supported, positive geometry arc length, and no invalid edge.
2. **Landmarks.** Generate the fixed Rademacher vector from the first 66 bits
   of `SHA256("temporac.pose-landmark.v1" || uint32_be(counter))`, mapping
   `0 -> -1`, `1 -> +1`, and dividing by `sqrt(66)`. Set `ell_t=a^T g_t`.
   A candidate is a strict maximum on both sides:
   `ell[t-1] < ell[t]` and `ell[t] > ell[t+1]`. Its parabolic denominator
   `D_t=ell[t-1]-2ell[t]+ell[t+1]` must be finite and strictly negative. A zero,
   positive, or nonfinite denominator abstains; there is no plateau owner.
   The offset is

   \[
   \xi_t=\operatorname{clip}
   \left(\frac{\ell_{t-1}-\ell_{t+1}}{2D_t},-0.5,0.5\right).
   \tag{F8}
   \]

   The geometry and phase landmark use linear interpolation toward the neighbor
   in the sign of `xi` and phase renormalization. The nearest strict local
   minimum on each side must exist in the same segment. Two-sided prominence
   must be at least 0.20 of the nonzero run range; the score must lie within
   0.05 run ranges of the best strict maximum. Every synthetic primitive
   traversal has exactly one retained landmark. Natural candidate traversals
   are precisely successive retained-landmark intervals; at least one is
   required, and all target-bearing intervals must be complete. Only the union
   of those intervals is target support; leading and trailing partial arcs are
   invalid-mask regions and cannot emit a pulse.
3. **Origin.** Give each complete traversal exactly one landmark vote. With
   interpolated landmark phases `z^L_j`, compute

   \[
   m=K^{-1}\sum_{j=1}^Kz^L_j,\qquad \rho=\|m\|_2,\qquad
   o=m/\rho,\qquad \widetilde z_t=z_t\overline o.
   \tag{F9}
   \]

   `K>=2`, `rho>=0.95`, maximum landmark-geometry distance from the
   coordinatewise median `<=0.05`, and leave-one-traversal-out origin change
   `<=0.01` cycles are required. Landmark count and order must agree under
   exact correspondence. The unique high-score component within 0.05
   normalized score units of the maximum has diameter `<=0.05` cycle, and the
   best score exceeds every score outside a 0.10-cycle neighborhood by at
   least 0.10 normalized range.
4. **Per-track topology and collision.** For every retained-landmark traversal,
   set `phi_t=atan2(tilde_z[t,y],tilde_z[t,x])/(2 pi)`, include its
   deterministic closing edge, and compute

   \[
   W=\operatorname{round}\sum_{t=0}^{N-1}
   \operatorname{wrap}_{(-0.5,0.5]}(\phi_{t+1\bmod N}-\phi_t).
   \tag{F10}
   \]

   Each traversal individually needs `W=+1`, at least 30 of 32 occupied phase
   bins, maximum circular phase gap `<=0.10`, normalized reconstruction
   `<=0.02`, and no detected collision. A collision is any pair separated by
   at least 0.10 canonical cycle whose fully supported 149-channel continuous-
   state RMS distance `||y_a-y_b||_2/sqrt(149)` is `<=1e-3`; one such pair
   makes the track abstain. Symmetric or self-intersecting complete
   delay-state orbits, incoherent landmarks, degree 0/-1/+2, half/double/higher
   harmonics, static phase, temporally permuted/noisy phase, shuffled static
   code, or decoder bypass never certify.
5. **Strict increments.** Recompute every valid `delta_t` in float64 and require
   `-1e-12 <= delta_t < 0.25` individually. The allowance of up to 1% negative
   edges is population-only teacher-artifact screening and never changes this
   requirement. A failed edge abstains the whole target-bearing track.
6. **Compensated seam ownership.** Canonical phase accumulation uses Neumaier
   compensated float64 summation in source order. For tolerance `tau`, define

   \[
   S_\tau(x)=
   \begin{cases}
   \operatorname{round}(x),&|x-\operatorname{round}(x)|\le\tau,\\
   x,&\text{otherwise},
   \end{cases}
   \qquad
   e_t^{(\tau)}=\lfloor S_\tau(A_{t+1})\rfloor-
   \lfloor S_\tau(A_t)\rfloor.
   \tag{F11}
   \]

   For each segment set `A_0=mod(phi_0,1)`. Neumaier accumulation is exact:
   initialize `(s,c)=(A_0,0)`; for each `x=delta_t`, set `u=s+x`, update
   `c+=(s-u)+x` when `abs(s)>=abs(x)` and `c+=(x-u)+s` otherwise, set `s=u`,
   and expose `A=s+c`. `round` in F11 means the unique nearest integer; a
   half-integer is never within the frozen tolerances. At `tau=1e-7`, every pulse
   increment must be in `{0,1}`, no edge may produce a backward seam crossing,
   and every retained-landmark traversal must contain exactly one positive
   crossing. The complete pulse bit array must be identical for
   `tau in {1e-8,1e-7,1e-6}`; otherwise the track abstains.
7. **Unseen-resampler equality.** At every matched physical state, not merely
   at the origin, maximum canonical circular-phase error is `<=0.01` cycle for
   both held-out PCHIP and windowed-sinc views, with origin error `<=0.005`
   cycle. Pull each resampled crossing through `psi` onto the frozen clean
   traversal-edge partition; the full canonical pulse bit vector must be
   exactly equal to the clean vector. Any mismatch abstains. This gate judges
   the approximate local direction and static pool together; it does not add
   a timing module.
8. **Attack semantics.** Orientation-preserving circle homeomorphisms
   `h_c(phi)=phi+c sin(2 pi phi)/(2 pi) mod 1` for
   `c in {-0.75,-0.5,0.5,0.75}` have positive orientation and must preserve the
   pose-canonical seam and complete pulse bits after landmark canonicalization;
   they are not rejected. Degree, harmonic, reversal, symmetry, collision,
   static-code, phase-permutation/noise, and decoder-bypass attacks must be
   rejected. Full time reversal abstains 100%.

Teacher artifact selection uses the synthetic suite only: at least 95% of
held-out asymmetric source orbits must return eligible, 100% of noiseless
supported fixtures must return eligible, the population fraction of edges
below `-1e-12` is at most 1%, the aggregate detected collision-pair fraction is
at most 1%, and all required attack outcomes above must hold.
The 1% statistic can select an artifact only; every affected target track
still abstains. Passing checkpoints are ordered by lowest held-out
reconstruction, earliest epoch, then smallest seed. Positive variation is not
an ordering key. Targets are detached and hashed. A later frozen natural
teacher/count mismatch is evaluated only in K7 and can only kill the route.

## Canonical TempoRAC Architecture

The causal encoder input is the same normalized pose/bone/confidence/mask base
plus backward displacement lags `{-4,-2,-1}` and masks, exactly 269 channels.
The shared encoder is `Linear(269,64)` followed by two residual causal
depthwise-separable blocks, width 64, kernel 5, dilations 1 and 2. Each block is
`LayerNorm-DepthwiseCausalConv1d-GELU-PointwiseConv1d-GELU-residual`, with bias,
no dropout, and no batch normalization. Invalid samples are zero before and
after every block.

The ordered branches are `(slow,medium,fast)` with respective dilations
`(4,2,1)`. Each has two identical-shape width-64, kernel-5 residual causal
depthwise-separable blocks at its dilation, then
`LayerNorm-Linear(64,1)-Sigmoid`. For global edge `e=(t,t+1)`, response
alignment is frozen as

\[
r^{(k)}_{ie}=v_{ie}\,\sigma(\ell^{(k)}_{i,t+1}),
\tag{F12}
\]

where the causal branch output at sample `t+1` owns the edge. The sample-zero
branch output is computed with zero left padding and discarded; sample
`T-1` owns the final edge `T-2 -> T-1`. An invalid edge, invalid endpoint, or
padding edge has `v=0`, response exact zero, and no loss/NOLA/decoder support.
Each identity begins with empty zero buffers and ends by destroying them.
Encoder buffers key by `(opaque_token,slot,layer)` and branch buffers by
`(opaque_token,slot,branch,layer)`. Branches traverse a full identity once;
windows only slice the cached edge tensor and never advance causal state.

## One Fixed Irregular-Clock Tempo Cue

Only the cue receives relative source clock. For valid edge `e`, coordinate
velocity is `(g_{t+1,d}-g_{t,d})/Delta q_e`. The 48-period grid is consistent
with the supported acquisition domain:

\[
P_b=\exp\left(\log5+\frac{b}{47}\log\frac{128}{5}\right),
\qquad f_b=P_b^{-1},\qquad b=0,\ldots,47.
\tag{F13}
\]

At edge midpoint clocks, trapezoidal quadrature and a source-clock Hann taper
weight each valid coordinate. Weighted least squares removes an intercept and
linear clock trend. A coordinate needs at least eight positive-weight edges
and residual energy above `1e-8`; a window needs midpoint span at least five
source frames. For detrended residual `x_ed` and full nonnegative weight
`w_ed`, power is

\[
S_{bd}=\frac{(\sum_ew_{ed}x_{ed}\cos2\pi f_b\tau_e)^2+
(\sum_ew_{ed}x_{ed}\sin2\pi f_b\tau_e)^2}
{(\sum_ew_{ed})(\sum_ew_{ed}x_{ed}^2)+10^{-8}}.
\tag{F14}
\]

Let `s_b` be the arithmetic mean across eligible coordinates. With support,

\[
p_b=\frac{s_b+10^{-8}}{\sum_a(s_a+10^{-8})},\qquad
u=\sum_bp_b\log f_b,
\quad
\gamma=\min(1,n/32)\min(1,Q/128)(D/66)
\max\left(0,1+\frac{\sum_bp_b\log(p_b+10^{-8})}{\log48}\right).
\tag{F15}
\]

Without support, `p` is exactly uniform, `u` is the uniform-grid log-frequency
mean, and `gamma=0`. A reliable window has `gamma>=0.15`, `n>=16`, `Q>=16`, and
`D>=16`. At least three reliable windows define the stop-gradient reference
with the exact weighted-median rule from F3. Otherwise every gate is uniform.

For anchors `mu=(-log1.5,0,+log1.5)` and `sigma_g=log1.5/2`, the sole router is

\[
\widetilde g_{i\ell k}=\operatorname{softmax}_k
\left[-\frac{(u_{i\ell}-\bar u_i-\mu_k)^2}{2\sigma_g^2}\right],
\qquad
g_{i\ell}=\gamma_{i\ell}\widetilde g_{i\ell}
+(1-\gamma_{i\ell})(1/3,1/3,1/3).
\tag{F16}
\]

There is no `argmax`, hard period, selected bin, ACF fusion, learned residual,
or fallback frequency. An unchanged window's raw `(p,u,gamma)` is invariant to
outside-window changes; a final gate may change only through the documented
same-identity reference. A non-circular ACF may be logged after the canonical
run but cannot affect a tensor, loss, checkpoint, gate, or count.

## Targets, Responsibilities, and Response Loss

Synthetic response training uses only certified source orbits and has equal
source-orbit/traversal mass within 0.10 log-frequency units of each anchor.
Natural shards are prediction inputs, not response-training examples. Training
uses the frozen linear resampler; PCHIP and sinc are held out. Within certified
traversal `j`, let
`chi_ie=alpha_ie/sum_{h in j} alpha_ih`, so every traversal has total physical-
progress weight one; a nonpositive denominator rejects the traversal. With
positive taper `a` defined below, overlap and branch responsibilities are

\[
\chi_{ie}=\frac{\alpha_{ie}}{\sum_{h\in j(e)}\alpha_{ih}},\qquad
\omega_{i\ell e}=\frac{a_{i\ell e}}
{\sum_{j\in\mathcal L_i(e)}a_{ije}},\qquad
\pi_{iek}=\sum_{\ell\in\mathcal L_i(e)}\omega_{i\ell e}p^*_{i\ell k}.
\tag{F17}
\]

Both denominators must be positive; sums over windows and branches equal one.
For response probabilities `x`, certified binary pulse `y`, and nonnegative
weights `w`, require `Z_+=sum w y>0` and `Z_-=sum w(1-y)>0`, and define

\[
\begin{aligned}
\mathcal B&=-\tfrac12\frac{\sum wy\log\max(x,10^{-7})}{Z_+}
-\tfrac12\frac{\sum w(1-y)\log\max(1-x,10^{-7})}{Z_-},\\
\mathcal M_+&=\frac{\sum wy[\max(0,0.75-x)]^2}{Z_+},\qquad
\mathcal M_-=\frac{\sum w(1-y)[\max(0,x-0.25)]^2}{Z_-},\\
\mathcal P(x,y;w)&=\mathcal B+0.5\mathcal M_++0.5\mathcal M_-.
\end{aligned}
\tag{F18}
\]

The once-trained canonical response loss is

\[
\mathcal L_{resp}=\frac13\sum_k
\mathcal P(r^{(k)},e^*;v\chi\pi_k)+\mathcal P(R,e^*;v\chi).
\tag{F19}
\]

Teacher, target, warp metadata, masks, clocks, windows, cue, reference, gate,
taper, NOLA arithmetic, and decoder are stop-gradient. No continuity, load,
entropy, count, period, density, reconstruction, or boundary loss is present.

## Exact Positive NOLA and One Decoder

For window-local edge index `n=0,...,126`, the positive taper is

\[
a_{i\ell e}=10^{-3}+(1-10^{-3})
\sin^2\left(\frac{\pi(n+1/2)}{127}\right).
\tag{F20}
\]

Probability responses are fused before reconstruction. Define numerator and
denominator, prove the denominator, and divide exactly:

\[
r_{i\ell e}=\sum_kg_{i\ell k}r^{(k)}_{ie},\qquad
N_{ie}=\sum_{\ell\in\mathcal L_i(e)}a_{i\ell e}v_{ie}r_{i\ell e},\qquad
D_{ie}=\sum_{\ell\in\mathcal L_i(e)}a_{i\ell e}v_{ie},\qquad
R_{ie}=N_{ie}/D_{ie}.
\tag{F21}
\]

Every valid edge must have finite `D>=1e-3` before division; otherwise the
graph fails. There is no added epsilon. Invalid edges set `N=D=R=0` and split
the sequence. On each valid run the decoder forms maximal connected components
of `R>=0.5`; each contributes one count. A plateau location is the floor of
the midpoint of the first and last maximum edge. The only decision is

\[
\widehat c_i=D_{peak}(R_i,v_i)=
\#\operatorname{CC}\{e:v_{ie}=1\ \land\ R_{ie}\ge0.5\}.
\tag{F22}
\]

`D_peak` is called exactly once per supplied valid identity and returns zero
on empty support. It has no smoothing, NMS, minimum distance, learned/dynamic
threshold, period, vote, rounding, window sum, or integral fallback. Response
mass may be logged only after the count and cannot enter any decision.

## Optimization, Checkpoints, and State

Teacher training uses AdamW, learning rate `3e-4`, betas `(0.9,0.999)`, epsilon
`1e-8`, weight decay `1e-4`, gradient clip 1.0, and at most 200 epochs.
Response training uses AdamW at `1e-3` with the same remaining values and at
most 100 epochs. Both use 5% linear warmup, cosine decay to 10% base rate,
bfloat16 activations, float32 model reductions, and float64 evaluator
certification. Full-identity gradient accumulation supplies at least 8,192
valid edges per step. Seeds are 20260815, 20260816, and 20260817.

Teacher checkpoint precedence is the G1 artifact rule above. Within each of the
three response seeds, the checkpoint is selected solely by lowest held-out
synthetic `L_resp`, then earliest epoch. All three seed checkpoints are frozen
and must subsequently pass G3 and G4; a failure cannot discard a seed or select
a runner-up. Human fields never select a target, configuration, threshold,
epoch, seed, or arm. Each capacity-control seed uses the identical precedence
with `L_ctl`, and every gate arm is frozen before any checkpoint is evaluated
by G3.
All constants, deterministic flags, fixtures, interpolators, projections,
operator versions, and tie rules are serialized in the future
`temporac.execution.v2.json` and hashed before implementation. This proposal
does not create or authorize that file.

Teacher gradients update only its static-code, phase, and decoder MLPs.
Response gradients update only the shared encoder and three branches. A fresh
graph and empty private buffers are created per full identity; no state
survives an optimizer step or identity. Consistently permuting identity records
and keys may only reorder outputs. Changing one identity while fixing all
other bytes leaves untouched CPU tensors bit-identical and deterministic GPU
tensors within `1e-6`; intentionally shared/reused/reassigned state must fail.

## Frozen Controls and Exact Decision Statistics

For each of the three independently trained canonical seed banks, cache all
three full-track sigmoid response tensors once. Local, global, uniform, blocked,
and within-track-shuffled arms
modify only gates. Global uses one full-track NUDFT distribution and one gate;
uniform is exactly `(1/3,1/3,1/3)`; blocked zeros cue coordinates before the
NUDFT and must produce `gamma=0` and exact uniform gates; shuffled cyclically
shifts reliable raw `(p,u,gamma)` tuples by one ordered window while preserving
the original reference. Blocked and uniform must be output-identical; their
separate traces test path integrity and do not count as independent evidence.

The unrouted capacity-matched control has the identical encoder and three
branch architectures, parameter shapes, dilations, FLOPs, and receptive-field
set `{9,17,33}`. Each branch applies its own sigmoid, after which fusion is
exactly uniform at probability level:

\[
r^{ctl}_{ie}=\frac13\sum_{k=1}^3\sigma(\ell^{ctl}_{iek}),\qquad
\mathcal L_{ctl}=\frac13\sum_k\mathcal P(r^{ctl,k},e^*;v)
+\mathcal P(r^{ctl},e^*;v).
\tag{F23}
\]

All control branches receive the full unstratified target. Parameter count is
exactly equal, measured multiply-add FLOPs differ by at most 1%, and the
receptive-field set is identical; otherwise the control is invalid.

An atomic **synthetic source-orbit unit** is one clean primitive generator seed
plus all of its warped/resampled views. An atomic **natural source-component
unit** is all videos and supplied identities sharing one canonical-source
component in the audit manifest. No view, window, person, or video is resampled
independently of its unit. Within a synthetic unit, losses are averaged equally
over certified traversals, then identities, then equally over the three frozen
seed banks; units receive equal weight. Seeds are never bootstrap units. A
source unit with zero certified target events is invalid. Split/merge rate is
the number of split plus merge incidences divided by certified target events;
its zero denominator therefore fails. Natural video-first MAE first averages
absolute person error within each video and seed, averages the three seed
values equally for that video, then averages videos equally; an evaluated video
with zero supplied valid identities fails. A component bootstrap carries all
its videos, identities, and seed predictions together.

Every paired lower bound in this document is the deterministic 2.5th
percentile of 10,000 unit-block bootstrap replicates using the first 64
bits of `SHA256("temporac.bootstrap.v2" || statistic_name)` as the RNG seed.
An upper bound is the 97.5th percentile. Ties in arm selection follow
`global < uniform < shuffled < control`. Fewer than eight atomic units, a
nonfinite replicate, or a zero denominator where failure is specified makes
the gate fail; no interval is reported.

For lower-is-better traversal-normalized pulse loss `L_a` on arm `a`, define
the simultaneous local-versus-strongest-nonlocal margin and shuffle removal as

\[
M_{local}=\min_{a\in\{global,uniform,shuffled\}}
(\bar L_a-\bar L_{local}),\qquad
M_{control}=\bar L_{control}-\bar L_{local},\qquad
Q_{shuffle}=\frac{\bar L_{shuffled}-\bar L_{local}}
{\bar L_{global}-\bar L_{local}}.
\tag{F24}
\]

The bootstrap recomputes the minimum in every replicate. G3 requires both point
margins `M_local` and `M_control` and their respective paired lower bounds to be
strictly above zero. If
`Lbar_global-Lbar_local <= 0`, shuffle removal fails by definition; otherwise
its point estimate and paired lower bound must be at least 0.80. `Beats` means
exactly these inequalities and nothing qualitative.

For specialization column `k`, define

\[
M^{diag}_k=\min_{j\ne k}(\bar L_{jk}-\bar L_{kk}),\qquad
M^{ctl}_k=\bar L^{ctl}_k-\bar L_{kk}.
\tag{F25}
\]

Each point margin and paired lower bound must be strictly positive, and each
branch receives at least 10% of total F17 responsibility. Split/merge rate is a
co-primary hard check and must be no worse on the diagonal than every off-
diagonal branch and the control; a tie is allowed only when all are exactly
zero and the pulse-loss margins pass.

For shortcut attack `a`, let `G_ref=Lbar_control-Lbar_local` on the same linear
resampler source units and `G_a=Lbar_control-Lbar_a`. Retained gain and unseen-
resampler retention are

\[
R_a=\frac{\max(0,G_a)}{G_{ref}},\qquad
R_{unseen}=\frac{Lbar^{unseen}_{control}-Lbar^{unseen}_{local}}
{G_{ref}}.
\tag{F26}
\]

Each shortcut arm uses the exact capacity-control encoder/three-branch graph and
optimizer. Its pose channels are replaced by exactly one named input: zero pose
plus relative timestamp, the frozen 37-channel nuisance projection, a
within-orbit pose permutation, repeated static code, repeated track length, or
synthetic log derivative. Each is trained independently with the same three
seeds; within each seed its epoch is selected by lowest held-out attack pulse
loss then earliest epoch. No seed is discarded and no K3 statistic selects a
checkpoint. Forbidden attack inputs exist only
inside this harness.

`G_ref<=0` fails K3. Every no-pose timestamp, nuisance-only, pose-shuffle,
static-code-only, track-length, and warp-metadata attack needs point and paired
upper bound `R_a<0.10`; held-out PCHIP and sinc each need point and paired lower
bound `R_unseen>=0.80`. An attack that is worse than control retains zero.

## One Two-Source Pulse Gate

G4 is one gate with two immutable sources under one schema:

1. deterministic operator fixtures whose probability responses directly
   attack taper, overlap, threshold, boundaries, gaps, padding, plateaus,
   pauses, split, and merge behavior; and
2. frozen held-out outputs of all three selected-seed canonical banks on PCHIP and sinc
   source-orbit views, with no post-freeze calibration.

Both sources cover all 32 window-grid start offsets `0,...,31`, boundary tails,
and supported event spacings `4,5,8,16,32,64,127,128,129` valid edges. Spacings
2 and 3 are excluded as outside the certified increment domain. Spacing 4 is
the derived seam-overshoot boundary case even though `P_min=5`.

For target event edge `e_j`, define neighborhood
`N_j={e in the same valid run: |e-e_j|<=1}`. A predicted supra-0.5 component
`C_m` is associated with event `j` exactly when `C_m` intersects `N_j`.
In this bipartite association graph, an event of degree zero is a miss, event
degree above one is a split, component degree zero is an extra, and component
degree above one is a merge. This definition needs no matching heuristic.
Every source, resampler, spacing, and offset requires zero misses, splits,
extras, and merges.

Let `V` be all valid non-event edges outside every `N_j`; packs with empty `V`
are invalid. Worst positive and negative margins are

\[
m_+=\min_j\left(\max_{e\in N_j}R_e-0.5\right),\qquad
m_-=\min_{e\in V}(0.5-R_e).
\tag{F27}
\]

Each pack requires `m_+>=0.10` and `m_->=0.10`, so one event core reaches at
least 0.60 and every guard/valley edge stays at most 0.40. A threshold tie,
zero denominator, or any component error rejects strict TempoRAC; the integral
audit cannot substitute.

For boundary events whose `N_j` intersects a window start or end, define
`E_B=(miss+split+extra+merge)/N_B`, assigning an extra to the nearest boundary
within 16 edges, breaking an equal-distance tie toward the lower source edge,
and excluding it otherwise. `N_B=0` fails the boundary fixture.
The paired reconstruction-order statistic is

\[
\Delta_B=E_B^{independent\ window\ decode}-E_B^{NOLA\ then\ one\ decode}.
\tag{F28}
\]

K5 requires zero G4 NOLA errors, a point `Delta_B>0`, and paired lower bound
above zero. On natural frozen responses, half-hop grid sensitivity is
`sum_i |c_i^(0)-c_i^(16)| / sum_i max(1,c_i^(0)) <=0.005`; an empty identity
set fails.

## Executable Numeric Gates and Authorization Sequence

No gate is deemed passed by prose. A pass requires the named immutable hashes.
Gates run G0 through G5 in numeric order; K decisions run K0 through K7 in
numeric order. Although synthetic fixture code may be prepared in S0, no
later-numbered pass can precede an earlier one.

| Stage | Authorized scope and exact order | Hard stop |
|---|---|---|
| S0 | Specification-only: freeze execution schema, synthetic generators, statistics, and expected receipt formats. No source or server access. Await a separate trusted-packer authorization. | No G0 artifact or data-bearing work without future authorization |
| S1 | After future authorization only: trusted packer emits feature/vault/audit separation; execute G0 then K0. Vault payload remains closed. | Any firewall, duplicate, provenance, split, or reachability failure |
| S2 | Feature-only/synthetic work: execute G1 then K1; execute G2 then K2. Teacher targets are emitted only by `certify_target`. | Any teacher, target, topology, acquisition, or graph-invariant failure |
| S3 | Train three independently once-trained synthetic seed banks and controls; execute G3, then K3 and K4; execute G4, then K5; evaluate K6 last from frozen identity traces. | Any shortcut, specialization, routing, pulse, NOLA, or isolation failure |
| S4 | Freeze teacher, target hashes, banks, configurations, arms, evaluator code, and three-seed feature-only predictions. Only then may a separately permissioned evaluator open the natural vault. Execute G5 then K7, including natural teacher-unit and efficacy checks. | Any natural-unit, coverage, efficacy, stationarity, or redundancy failure |

### Build gates G0--G5

| Gate | Immutable artifacts and exact pass condition | Failure |
|---|---|---|
| G0 firewall/packing | Future execution JSON; feature, vault, audit schemas; per-file/sample hashes; duplicate-conflict receipts; forbidden-key scan; canonical-source split receipt; 37-channel nuisance schema. Whitelist loading passes, forbidden injections fail before payload, train/development component intersection is zero, and forbidden paths are unreachable. | Stop all data-bearing work |
| G1 teacher/target | Generator/resampler hashes, selected teacher, population screen, per-track certificate receipts, attack matrix, canonical phase/pulse traces. Artifact selection obeys the 95%/100% and population 1% rules; every serialized target individually passes all `certify_target` fields; homeomorphisms preserve seam/pulses; degree/harmonic/reversal/bypass attacks reject. | Reject teacher and TempoRAC |
| G2 graph | Object IDs, exact parameter/FLOP receipt, private buffers, edge-alignment trace, raw-cue/reference perturbations, NOLA denominators, fusion/decode graph, key permutations. All seven invariants pass. | Not TempoRAC; stop |
| G3 routing/mechanism | Three once-trained bank and cached-response hashes, seed-aggregated `3x3` matrix, exact control receipts, five arm traces, unit bootstrap samples. F24--F26 and all responsibility thresholds pass. | Kill the sole routing contribution |
| G4 pulse/NOLA/peak | One two-source pack hash, association graph, hard margins, decoder calls, all 32 offsets, supported spacings, boundary receipts. F27--F28 pass with zero component errors and exact positive division. | Reject strict route; no integral rescue |
| G5 frozen natural evaluation | Three-seed prediction hashes, frozen evaluator/config/arm hashes, source-component bootstrap samples, all G0--G4 hashes. Vault opens only after these freeze; all K7 conditions pass. | Kill canonical method |

### Scientific kill gates K0--K7

| Gate | Exact decision | Failure action |
|---|---|---|
| K0 protocol/firewall | G0 passes with physical separation, source-component disjointness, and forbidden reachability false. | No data-bearing run |
| K1 response unit | G1 passes; every target byte has an eligible certificate, strict increments, one crossing per traversal, zero collision, gap support, and resampler phase/pulse equality. Population tolerances cannot waive a track failure. | Reject response route; no human repair |
| K2 architecture | G2 passes all seven invariants, edge alignment, raw-cue locality/reference dependence, and key-permutation behavior. | Implementation failure |
| K3 shortcut resistance | All F26 point/interval rules pass for every attack and both unseen resamplers. | Reject cue as shortcut |
| K4 routing mechanism | G3 diagonal F25 margins, branch mass, F24 local/nonlocal and local/control margins, and shuffle-removal rules all pass on the once-trained frozen tensor. | Kill dominant claim |
| K5 pulse/reconstruction | G4 has zero errors and margins at every source/offset/spacing; F28 and natural half-hop sensitivity pass. | Reject strict peak route |
| K6 identity isolation | Untouched CPU outputs are bit-identical, GPU differences are `<=1e-6`, consistent key permutation only reorders, and deliberately shared/reused/reassigned state controls fail. | Implementation failure; no private-state claim |
| K7 natural unit and pilot | Frozen natural coherent-certificate coverage is at least 80% of eligible identities and source components; harmonic/off-grid fraction is at most 5%; median teacher-event/count ratio is in `[0.95,1.05]`; the efficacy statistic below passes; teacher-only and the exact control remain inferior. | Kill canonical method; no paper/server/pivot authorization |

For K7, choose the strongest matched nonlocal/control arm by the smallest
frozen video-first AvgMAE, ties by the fixed order above. Let `A_*` and
`A_local` be its and TempoRAC's AvgMAE. The primary relative reduction is

\[
E_{rel}=\frac{A_*-A_{local}}{A_*}.
\tag{F29}
\]

`A_*=0` fails because a positive routing improvement is impossible. Require
point `E_rel>=0.05` and paired source-component bootstrap lower bound above
zero. On stationary videos, the paired difference
`AvgMAE_local-AvgMAE_*` must have point and upper bound `<=0.02`. For
teacher-only and the capacity control separately, paired
`AvgMAE_arm-AvgMAE_local` must have point and lower bound strictly above zero;
otherwise routing has no residual contribution. All three seeds are included
as repeated predictions inside their source component, not resampled as
independent units. A component interval containing zero fails to advance and
does not weaken a threshold.

Natural certificate coverage uses the feature-eligible population fixed before
vault opening. Identity coverage is certified identities divided by
feature-eligible identities. Component coverage is canonical-source components
containing at least one certified identity divided by components containing at
least one feature-eligible identity. Either zero denominator fails; both
fractions must be at least 0.80. For each certified identity with positive vault
count, set `r_i=teacher_events_i/count_i`. For a zero-count identity, zero
teacher events are compatible and positive teacher events are a mismatch; no
ratio is formed. Fewer than eight positive-count ratios fails. Sort positive-
count ratios and define the median as the lower middle order statistic; it must
lie in `[0.95,1.05]`. A positive-count ratio outside `[0.95,1.05]`, or a
zero-count mismatch, is harmonic/off-grid; the equal-identity mismatch fraction
must be at most 0.05. Ratios in `[0.45,0.55]` are logged as half, ratios in
`[1.90,2.10]` as double, and the remainder as off-grid, but subclasses do not
change the single frozen failure rule.

## Minimal Claim-Driven Validation: Three Paper-Visible Blocks

### Evidence block 1: Response-unit and decoder eligibility

- **Role:** Prerequisite, not a contribution.
- **Evidence:** G1 per-track certificate coverage and G4 two-source hard pulse
  gate, summarized with abstention reasons, worst margins, and zero component
  errors.
- **Failure:** Reject the strict route; do not retune with human counts or use
  an integral decoder.

### Evidence block 2: Frozen-response routing mechanism

- **Supporting check:** The F25 diagonal specialization matrix against the
  exact probability-fused control.
- **Dominant test:** Local, global, uniform/blocked, shuffled, and control gates
  on the identical cached three-response tensor, decided only by F24.
- **Failure:** Kill the only paper-level routing claim; do not add a learned
  router, expert, ACF fusion, or retrained arm.

### Evidence block 3: Supplied-track efficacy and stationarity

- **Evidence:** Three-seed feature-only predictions, opened against the natural
  vault only after freeze, evaluated by F29 and the stationary/nonredundancy
  rules.
- **Scope:** Partial-cache, GT-bbox-assisted AlphaPose supplied tracks only;
  not end-to-end MRAC or an official test result.
- **Failure:** Stop. No title, paper, server continuation, or pivot follows.

## Failure Modes and Binding Dispositions

| Failure | Detection | Disposition |
|---|---|---|
| Conflicting duplicate clocks | F1 pairwise tolerance and receipt | abstain; never average silently |
| Hidden-cycle-capable or missing clock gap | acquisition contract and certificate | abstain; never bridge or infer |
| Degree, harmonic, reversal, collision, unstable origin | `certify_target` | reject target; population tolerance cannot repair |
| Homeomorphic phase reparameterization changes canonical pulses | G1 homeomorphism preservation attack | reject teacher; do not claim all homeomorphisms should reject |
| Static-code or decoder bypass | constant/permuted/noisy phase and shuffled-code attacks | reject teacher |
| Pulse miss, extra, split, merge, or weak margin | G4 on fixtures and trained outputs | reject strict route; no integral fallback |
| NUDFT clock/mask/metadata shortcut | K3 retained-gain statistics | reject cue; no ACF addition by default |
| Expert collapse or nonlinear control mismatch | G3 receipts and F25 | kill mechanism claim |
| Local cue does not beat strongest nonlocal | F24 | kill dominant claim |
| Identity leakage | K6 intervention traces | implementation failure |
| Teacher-only/control is not inferior | K7 paired bounds | no residual contribution |
| Underpowered development cache | fewer than eight units or interval includes zero | failure to advance, not threshold relaxation |

## Novelty, Scope, and Forbidden Boundaries

The maximum future claim is an empirical interaction at the intersection of
supplied identity tracks, shared response-scale branches, identity-private
state, cue-only relative window-local tempo, response fusion, and one
reconstructed decode. Every component is established, and full TWCRAC overlap
remains unresolved. The teacher, NUDFT, soft gate, causal TCNs, NOLA, private
buffers, connected components, and one-decode order are not independently
claimed.

This refinement does not authorize `L_cont`, canonical PAMS-TCC initialization,
ACF fusion, a learned router, GRU, integral decoder, WARP-PHASE, a missing-pose
pivot, server/data/test/result access, title/acronym freeze, novelty clearance,
paper drafting, positive claims, submission, or continued compute after a
failed gate. Test, sealed, heldout, and historical result paths remain
forbidden. WARP-PHASE remains terminally failed; a backup pivot is separate and
cannot be substituted.

## Experiment Handoff Inputs

- **Atomic units:** synthetic source orbit and natural canonical-source
  component exactly as defined above.
- **Must cache:** certificate receipt, canonical phase/pulse, raw NUDFT tuple,
  track reference, each gate, each full branch response, NOLA numerator and
  denominator, final response, association graph, component margins, decoder
  calls, and private-buffer keys.
- **Must prove:** strict target eligibility, diagonal scale specialization,
  same-response local routing advantage, trained-output pulse resolvability,
  exact positive NOLA, and identity isolation.
- **Must not use:** a label-mixed artifact in a model process, human-selected
  targets/checkpoints/configurations, WARP-PHASE artifacts, inferred PAMS
  SSHead, dynamic peak logic, independent window counting, or integral repair.

## Compute and Timeline Estimate

| Stage | Focused work | Engineer-days | RTX A6000-hour hard cap | Advance condition |
|---|---|---:|---:|---|
| S0 | contract, execution schema, synthetic fixture specification | 3--4 | 0 | future packer authorization exists |
| S1 | trusted packing/firewall, duplicate and source-component receipts | 4--6 | 0 | G0/K0 pass |
| S2 | teacher, certificate, resampler suite, graph/state instrumentation | 5--7 | 28 | G1/G2 and K1/K2 pass |
| S3 | once-trained bank, controls, G3/G4 packs and K3--K6 | 4--6 | 12 | every synthetic/mechanism gate passes |
| S4 | frozen three-seed feature-only predictions and later vault evaluation | 4--7 | 60 | G5/K7 pass |

The total is 20--30 focused engineer-days and at most 100 RTX A6000-hours.
Every cap is a hard stop, not a spending target. Unused hours after a failure
cannot fund a rescue route without a new review. New human annotation cost is
zero. The estimate and sequence authorize neither source/server access nor an
implementation or run.
