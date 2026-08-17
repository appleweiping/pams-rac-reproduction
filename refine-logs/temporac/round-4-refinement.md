# Round 4 Refinement: TempoRAC

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

The revised route is still exactly the TempoRAC graph: identity-private
preprocessing and causal state; one encoder shared across identities; three
shared, matched slow/medium/fast sigmoid response branches; one detached,
cue-only window-local irregular-clock router; probability-level response
fusion; exact positive NOLA; and one threshold-0.5 connected-component decode
per identity. The primitive-orbit teacher remains a target producer and
falsifiable prerequisite, never an inference replacement.

Round-4 feedback that asks for a coherent synthetic witness, an untouched-
identity counterfactual, an evaluator-only primitive-unit kill, and singular
archive/runtime contracts is accepted because it tests the anchored graph.
WARP-PHASE, a pivot, a learned router, an integral fallback, evaluator-label
training, an extra dataset, an extra seed, or a fourth paper-visible block
would change the route and remain rejected.

## Simplicity Check

The dominant contribution remains one causal intervention: local relative-
tempo gates select among shared response scales before reconstruction and one
decode. The response graph, three seeds, exact 24 response jobs, comparator
set, final decoder, and three paper-visible blocks are unchanged.

One authoritative contract, `temporac.execution.v4`, replaces the ambiguous
v3 clauses. Its additional text freezes fixtures, bytes, masks, reductions,
runtime, and evaluator order; it adds no trainable component. The disconnected
analytic pulse convention, batch/per-identity NPZ fork, parabolic landmark
refinement, undefined stop rule, unsafe natural warp slopes, and unsupported
natural “stationary” wording are deleted rather than supplemented.

## Changes Made

- Aligned the F7 sine landmark and analytic phase with
  `theta=(psi-8)/32=psi/32-1/4`, bound analytic and certified pulse bits, and
  constructed ten complete traversals from eleven internal integer landmarks
  with direct base/G4 seams, exact half-cycle guards, and a proved finite knot
  domain.
- Froze a finite topology attack bank whose accepted homeomorphisms preserve
  landmarks, seams, pulses, winding, and counts, but do not promise equality
  of floating phase parameterizations.
- Separated trusted multi-person v44 source objects from exact seven-member
  per-identity feature archives and moved every shard digest into a detached
  receipt.
- Froze abstention codes and vector semantics, edge/run/component coordinates,
  float32 quantization before the only decoder, deterministic NPZ/JSON bytes,
  and population-wide stub predictions.
- Added certified natural training targets without evaluator labels: teacher
  phase progress supplies natural `chi`, the detached cue gate supplies
  natural branch responsibility, and X0 alone supplies analytic metadata.
- Froze a single neural environment, deterministic flags and initialization,
  exact job names, fixed optimizer-step horizons, checkpoint cadence, learning
  rate, source cycles, accumulation normalization, and gradient fixture.
- Replaced the K7 pooled ambiguity by exact per-person normalized errors and
  OBO, video-first seed means, a video-preserving component bootstrap,
  strongest-arm reselection, paired drift/clean intervals, and two
  evaluator-only global falsifiers.
- Added the two-identity untouched-output fixture, safe mean-one natural drift
  slopes, consistent continuous-channel interpolation, and the term
  “unwarped/clean” for natural controls.
- Closed the remaining formula, landmark, phase-set, mask, quadrature,
  run-logit, component, shortcut, operator-bank, and capability-order choices.

## Revised Full Proposal

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

## 1. Sole contract, precedence, and claim boundary

This full proposal is the only normative TempoRAC specification and is named
`temporac.execution.v4`. It supersedes every earlier TempoRAC numerical,
synthetic, topology, archive, runtime, training, prediction, receipt, gate,
and evaluation clause. The immutable Problem Anchor, the exact seven
architecture properties, and the audited physical v44 source schema remain
authoritative. If any earlier TempoRAC text conflicts, v4 controls; no clause
is imported by silence.

The one dominant claim is:

> On frozen supplied identity tracks, identity-indexed window-local
> relative-tempo routing among shared slow/medium/fast response experts
> improves one-decode counting under deterministic within-track pace drift
> relative to the strongest frozen matched nonlocal comparator, while
> retaining performance on the paired unwarped/clean tracks.

The three-by-three expert-specialization check is supporting evidence.
Topology, primitive-unit agreement, NOLA, identity isolation, shortcut,
runtime, and firewall checks are validity prerequisites, not independent
contributions.

This document authorizes no implementation, test, data access, server access,
experiment, result, paper edit, title, submission, or Git operation. TempoRAC
requires a fresh `/experiment-plan` and a fresh TempoRAC-specific server
preflight. It cannot inherit the WARP-PHASE plan, environment lock, output
budget, launch authority, or sanitized preflight.

## 2. Frozen graph and seven invariants

For identity `i`, a causal encoder emits `h_i,t`. Three equal-shape sigmoid
branches `k in (slow,medium,fast)` emit edge-event probabilities. A detached
irregular-clock cue produces fixed window gates. The gates fuse branch
probabilities, exact NOLA reconstructs one response, and a connected-component
decoder is invoked exactly once for the identity. Parameters are shared
across identities and no identity token is embedded.

The seven invariants are:

1. every state, receptive field, window, NOLA accumulator, run, adjacency,
   and normalization object is private to one supplied identity;
2. the three branches have identical operation and parameter count and differ
   only in learned weights and frozen dilation;
3. routing is a detached cue-only soft function of relative window-local
   tempo within one identity run;
4. sigmoid response probabilities, never logits or counts, are fused;
5. overlap is reconstructed by division through the exact strictly positive
   NOLA denominator;
6. one fixed threshold-0.5 connected-component function is called once per
   identity, including abstaining identities; and
7. human count, period, density, boundary, box, and evaluator fields enter no
   teacher/response objective, responsibility, checkpoint, prediction, or
   rerun.

There is no `L_cont`, PAMS-TCC initialization, ACF fusion, learned router,
GRU, integral decoder, WARP-PHASE object, pivot, per-person model copy, or
additional trainable module.

## 3. Trusted v44 boundary and one feature archive

### 3.1 Trusted source objects versus emitted identities

The only natural source is the audited v44 train/development pair with frozen
SHA-256 values `c96fc1dfa233ec4bf1af19e7480ee8c721f4cddee9f121185f4e494c7244e1eb`
and `4064ef24dc2970e9b07de8adddf1ecd4932aecb90453a67b4f90d7bd35d10251`.
A future separately authorized trusted packer verifies regular-file status,
path allowlist, byte count, and digest before deserializing. A trusted source
object is multi-person: `motion <f4[P,320,17,3]`, `person_mask bool[P]`,
`frame_mask bool[P,320]`, `sampled_frame_indices <i8[320]`, and one integer
`source_length`, alongside privileged provenance/vault fields. It is never
accepted by the model loader.

For source-object ordinal `n` in pickle order and split ASCII `train` or
`val`, define

`opaque_sample_key = SHA256("temporac.opaque-key.v4\0" || split || "\0" ||
source_pickle_sha256_raw32 || uint32_be(n))`.

Every `person_mask=true` source slot `p` emits one archive; its
`local_person_slot` is exactly source-array index `p`, without renumbering.
The pair `(opaque_sample_key,local_person_slot)` is the sole feature/vault
join key. Keys and slots route records and private state but are never model
tensors.

The model loader accepts exactly one deterministic ZIP_STORED NPZ per
identity, with NPY v2.0, C order, fixed ZIP timestamp 1980-01-01 00:00:00,
external attributes zero, no comment, no directory, no pickle, no object,
Unicode, structured, or void dtype, and these seven members in bytewise ASCII
name order:

| member | dtype | exact shape | constraint |
|---|---|---|---|
| `frame_mask.npy` | `uint8` | `[320]` | values in `{0,1}` |
| `local_person_slot.npy` | little-endian `int64` | `[1]` | nonnegative |
| `motion.npy` | little-endian `float32` | `[320,17,3]` | finite source payload |
| `opaque_sample_key.npy` | `uint8` | `[32]` | exact derived digest bytes |
| `person_mask.npy` | `uint8` | `[1]` | exact value one |
| `sampled_frame_indices.npy` | little-endian `int64` | `[320]` | nondecreasing |
| `source_length.npy` | little-endian `int64` | `[1]` | greater than one |

Maximum archive and expanded bytes are each 8 MiB. Missing, extra, duplicate,
reordered, compressed, noncanonical, symlink, or executable members fail
before arrays are returned. The archive SHA-256 is forbidden as an eighth
member; it exists only in the detached feature receipt.

### 3.2 Population/component manifest and exact vault join

The count-blind population manifest contains exactly one row for each of the
268 train and 134 development supplied-valid slots. Rows are ordered by split
(`train` then `val`), opaque key bytes, then slot, and contain exactly:
`split`, `opaque_key_hex`, `slot`, `component_key_hex`,
`source_binding_sha256`, `feature_receipt_sha256_or_null`, `eligible`, and
`reason_codes`. A component is one transitive connected component of the
undirected bipartite graph whose left vertices are source-object ordinals and
whose right vertices are raw canonical-source-ID digests; an edge is the
audited membership relation already present in trusted packer metadata.
Duplicate edges are removed, each side is bytewise sorted before traversal,
and components are ordered by their smallest `(split,source-object ordinal,
canonical-source-ID digest)` tuple. A component key is
`SHA256("temporac.component.v4\0" || split || "\0" || concatenation of the
sorted distinct raw SHA-256 digests of that component's canonical source IDs)`.
Source IDs and component memberships remain packer-audit metadata and never
enter an archive or tensor. The manifest contains no count, period, density,
boundary, box, action, video name, or evaluator statistic.

Within the trusted vault writer, slot `p` joins through the audited
`person_object_ids[p]` to exactly one raw object. The vault row stores the
same opaque key and slot, `source_length`, integer `count_gt`, and the raw
ordered period list. Each period is a source-frame boundary interval `[s,e)`
with integer `0<=s<e<=source_length` and duration `e-s`; no `+1`, density
conversion, resampled index inversion, or fallback exists. A missing,
duplicate, many-to-one, reordered, or non-total 268+134 source-to-vault join
fails globally.

### 3.3 Confidence, duplicate clocks, support, and G0

Packing/certification arithmetic is binary64. Joint `j` is valid exactly when
the frame mask is true, its three raw values are finite, and clipped confidence
`c=min(1,max(0,c_raw))` is strictly greater than 0.20. Invalid `x,y,c` are
canonical `+0.0`.

The root is the mean of valid hips 11/12. The track scale is the maximum of
`1e-3` and the median positive root-to-valid-shoulder-mean distance over first
clock occurrences. For duplicate samples `a,b`, at least eight common joints,
identical joint-valid bits, and identical root validity are required. With
`w_j=min(c_a,j,c_b,j)`, the weighted RMS is unambiguously

`d_ab = sqrt( (sum_j w_j ||x_tilde_a,j-x_tilde_b,j||_2^2) /
              (sum_j w_j) )`.                                      (F1)

The denominator must be positive, `d_ab<=0.02`, and maximum confidence
difference `<=1e-6`. An agreeing group uses confidence-weighted binary64
`x,y`, maximum confidence, and one final `<f4` rounding. A conflict,
noncontiguous clock reappearance, absent root/scale, or empty group abstains
the whole identity; no clock is deleted to rescue it.

After collapse, root-center and divide `x,y` by the one track scale. A frame
requires at least eight valid joints and one valid hip. For retained clocks
`q_0<...<q_T-1`, the registered physical minimum period is `P_min=5`, and an
interior edge is supported exactly when both frames are valid and

`1 <= q_(t+1)-q_t < P_min = 5`.                                    (F2)

After trimming invalid leading/trailing samples, any invalid frame,
unsupported edge, or invalid geometry length inside the remaining span makes
the whole natural identity feature-ineligible. A surviving span needs at
least 17 samples and 16 edges. There is no imputation, bridge, fragment sum,
or hidden-cycle inference.

G0 precedes every teacher/response job. The unconditional denominators are
268/18 train identities/components and 134/9 development
identities/components. A component is acquisition-eligible iff it contains at
least one eligible identity. G0 passes only when

`train_id>=215, train_component>=15, dev_id>=108, dev_component>=8`. (F3)

Missing or mismatched audited totals, zero denominator, duplicate manifest
row, absent mapping, or failed floor kills the route. G0 reports only
count-blind eligibility and reasons.

## 4. Coordinates, geometry, channels, and shared quadrature

### 4.1 Half-open coordinates and masks

Retained samples have integer coordinates. Base edge `e` is `[e,e+1)`.
Traversal `j` is `[L_j,L_(j+1))`; sample `s` belongs iff
`L_j<=s<L_(j+1)`. A subedge is the positive-length intersection of those
intervals. An exact integer seam belongs to the edge and traversal beginning
there. The terminal landmark belongs to the following guard/traversal and is
never duplicated. The final sample owns no right edge.

`E=T-1`. Edge coordinate `e` means retained-sample interval `[e,e+1)` and
physical clock interval `[q_e,q_(e+1))`. A run bound `[a,b)` and component
bound `[a,b)` are edge-index intervals with `0<=a<b<=E`. Runs/components in
different valid runs are never adjacent.

For certified tracks, `target_mask`, NOLA `edge_mask`, and `decoder_mask` are
one exactly on the union of complete landmark-to-landmark traversal edges.
Leading and trailing context is encoder/cue-visible but has all three masks
zero. For X0, the two partial guards are valid pose context but have target,
NOLA, loss, and decoder masks zero. Response entries outside `edge_mask` are
canonical float32 zero. Operator fixtures supply their explicit three masks.
No guard edge can create a target, NOLA value, loss row, or component.

### 4.2 Geometry and exact invalid-length action

The normalized 34-vector `p_t` is COCO-17 root-centered `x,y`. The 16 directed
bones are `(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
(11,13),(13,15),(12,14),(14,16),(0,5),(0,6),(0,1),(0,2)` and form
`b_t in R^32`; `g_t=[p_t,b_t] in R^66`.

On edge `e`, coordinate mask `M_e,d` requires all contributing joints at both
endpoints. Let `D_e=sum_d M_e,d`. The masked length is

`lambda_e = sqrt((66/D_e) * sum_d M_e,d*(g_(e+1),d-g_e,d)^2)`.      (F4)

`D_e<32`, a nonfinite term/result, or `lambda_e<=0` makes the edge length
invalid. Any invalid length inside a trimmed natural span abstains the whole
identity; any X0 invalid length fails that view and then X0 globally. It is
never replaced by zero, epsilon, an unmasked norm, or a run split. A
fractional subedge of fraction `rho` would own `rho*lambda_e`; v4 landmarks
are integer, but this rule remains for finite attack fixtures.

The teacher input `y_t in R^215` is exactly `[p 0:34, b 34:66,
two-sided normalized clock-blind direction 66:132, confidence 132:149,
17 joint+16 bone geometry masks 149:182, corresponding direction masks
182:215]`. At an interior run sample, a direction coordinate is supported iff
that joint/bone coordinate is valid at `t-1,t,t+1`. On this common mask let
`v^-_d=g_t,d-g_(t-1),d`, `v^+_d=g_(t+1),d-g_t,d`, normalize each masked
66-vector by its ordinary binary64 Euclidean norm, sum the two unit vectors,
and normalize that masked sum once more. The resulting 66-vector is the
clock-blind direction; the 33 joint/bone direction masks are true exactly when
both coordinates of their source geometry element have this common support.
Missing two-sided support or any chord/chord-sum norm at or below `1e-8`
yields exact-zero direction and false masks for the affected element.

The response input `x_t in R^269` is exactly `[p 0:34,b 34:66,
confidence 66:83, lag4 p 83:117, lag2 p 117:151, lag1 p 151:185,
current joint mask 185:202,current bone mask 202:218,lag4 mask 218:235,
lag2 mask 235:252,lag1 mask 252:269]`. Lags never cross a run; absent history
is exact zero with false masks. Clock, delta-clock, source length, key, slot,
component, warp metadata, and evaluator fields are absent.

### 4.3 Arc static code and one edge quadrature

For traversal subedge `a` of length `lambda_a` with endpoint continuous
teacher values `A_a,B_a`, define

`m_j,d = sum_a lambda_a(A_a+B_a)/2 / sum_a lambda_a`,
`q_j,d = sum_a lambda_a(A_a^2+A_a*B_a+B_a^2)/3 / sum_a lambda_a`.   (F5)

Every denominator is positive. Across `K` traversals, take equal-traversal
means `bar_m,bar_q` and `s=sqrt(max(bar_q-bar_m^2,0))`; `[bar_m,s] in R^298`
is the only static-MLP input. Binary64 Neumaier sums follow traversal/subedge
order. The exact squared-linear integral is subdivision invariant; a fixed
fixture requires agreement within two ulp.

For each window, form the sorted list of valid edge midpoints
`x_e=(q_e+q_(e+1))/2`. On that one list, endpoint trapezoid weights are half
the adjacent spacing and internal weights half the two-neighbor spacing.
Multiply by `sin^2(pi*(x_e-x_first)/(x_last-x_first))` to obtain one base
quadrature `b_w,e`. It requires at least three midpoints and positive span.
This exact `b_w,e` is shared by the cue, analytic X0 tempo, natural teacher
tempo, reliability, and reference receipts. A coordinate with a false mask
uses the subset of these already-computed weights; it never recomputes a
coordinate-specific trapezoid. A zero/nonfinite required sum fails.

## 5. Detached irregular-clock cue and router

Windows contain 128 sample/127 edge slots, stride 32, plus the terminal
right-aligned start if absent; a shorter run has one exact-length window.
Experts are evaluated once over the complete run and their cached run-level
logits are sliced by these windows. No window advances causal state.

For coordinate `d`, use `v_e,d=(g_(e+1),d-g_e,d)/(q_(e+1)-q_e)` on its valid
edge subset and the shared `b_w,e`. Each coordinate subset requires at least
three midpoints, positive total weight, positive midpoint span, and a positive
power denominator. Fit and remove `alpha+beta*z_e`, where `z_e` centers/scales
the midpoint span, by the exact compensated two-by-two weighted normal
equations. Require `S0>0,S2>0` and
`S0*S2-S1^2>=1e-12*S0*S2`. On the 48 periods
`P_l=exp(log(5)+l*log(128/5)/47)` and frequencies `f_l=1/P_l`, compute

`A_d(f_l)=sum_e b_w,e residual_e exp(-2*pi*i*f_l*x_e)`,
`s_d,l=|A_d(f_l)|^2/(sum_e b_w,e * sum_e b_w,e residual_e^2)`.     (F6)

At least 16 valid coordinates, 16 base-valid edges, base midpoint span 16,
positive power denominator `>1e-12`, and the shared quadrature are required.
Let `n_w` be the number of base-valid edges, `Q_w=x_last-x_first` on their
midpoints, and `D_w` the number of coordinates whose complete fit and spectrum
are valid. Average only those coordinate powers and set

`p_l=(s_l+1e-8)/sum_r(s_r+1e-8),
 u_w=sum_l p_l log f_l,
 c_H=min(1,max(0,1+sum_l p_l log(p_l+1e-8)/log(48))),
 gamma_w=min(1,n_w/32)*min(1,Q_w/128)*(D_w/66)*c_H`.

If any prerequisite fails, set `p_l=1/48`, `u_w=(1/48)sum_l log f_l`, and
`gamma_w=0` exactly. If the computed `gamma_w<0.15`, replace it by exact zero
and use exact uniform gates. No hard frequency or period is selected.

Reference weight is
`beta_w=gamma_w*sum_e b_w,e*lambda_e/m_e`, where `m_e` is window coverage.
The run reference is the first `(u,start)`-ordered weighted-median crossing of
half the positive total and requires three reliable windows. Otherwise all
run gates are uniform. With centers
`mu=(-log(1.5),0,+log(1.5))`, `sigma=log(1.5)/2`,

`a_w,k = gamma_w*softmax_k(-(u_w-bar_u-mu_k)^2/(2*sigma^2))
         +(1-gamma_w)/3`.                                         (F7)

The stable softmax subtracts its maximum. Cue, reference, confidence, and
gates are detached and reset per run. The global comparator uses one
full-run cue in all windows while retaining the frozen reference receipt;
uniform uses `1/3`; blocked zeros cue signals and must be bit-identical to
uniform; shuffle cyclically shifts reliable gates by one within each run.

## 6. Frozen X0: one coherent phase/certificate population

### 6.1 Forty source orbits and manifest

X0 uses CPython 3.12.4, NumPy 2.1.0, SciPy 1.14.1, binary64 ties-to-even, and
`numpy.random.Generator(PCG64)`. Source IDs are exactly `U0000...U0039`:
0-23 train, 24-31 tune, 32-39 heldout. Source key is
`SHA256("temporac.x0-unit.v4\0"||uint32_be(id))`.

The fixed COCO-17 base pose is:

| j | x | y | j | x | y |
|---:|---:|---:|---:|---:|---:|
|0|0.00|0.90|9|-0.38|0.15|
|1|-0.05|0.95|10|0.38|0.15|
|2|0.05|0.95|11|-0.12|0.00|
|3|-0.10|0.93|12|0.12|0.00|
|4|0.10|0.93|13|-0.14|-0.40|
|5|-0.18|0.65|14|0.14|-0.40|
|6|0.18|0.65|15|-0.15|-0.85|
|7|-0.30|0.40|16|0.15|-0.85|
|8|0.30|0.40||||

Only `J=(7,8,9,10,13,14,15,16)` moves. Let fixed linear map `H` send the
ordered 16 moving `x,y` coordinates to `[p,b]`, let landmark vector `rho` be
the first 66 MSB-first sign bits of the SHA stream
`SHA256("temporac.landmark.v4\0"||uint32_be(counter))`, and set
`w=H^T rho/||H^T rho||`. The clean orbit remains

`G_n(phi)=B+E_J[0.12*w*sin(2*pi*phi)
 +sum_(h=2)^4 (0.03/h)*(u_h*cos(2*pi*h*phi)+v_h*sin(2*pi*h*phi))]`. (F8)

For source `n`, attempt `a`, and view zero, the PCG64 seed is the unsigned
little-endian first eight SHA-256 bytes of
`"temporac.x0.coefficient.v4\0"||uint32_be(n)||uint32_be(a)||uint32_be(0)`.
Draw six length-16 vectors from `Uniform[-1,1]` in C order. Two-pass modified
Gram-Schmidt removes `w` and every earlier accepted vector in increasing
coordinate order; norm `<=1e-10` rejects the attempt. Ordered accepted pairs
are `(u_h,v_h)`, `h=2,3,4`. Confidence at joint `j` is
`clip(0.90+0.05*sin(2*pi*phi+2*pi*j/17),0,1)` and every mask is true.

Every candidate is evaluated on the exact cyclic grid `r/4096`,
`r=0,...,4095`, and is rejected on any of: nonfinite value; any coordinate
outside `[-2,2]`; pose scale below 0.1; total 66-geometry arc below 1; any
analytic tangent norm below `1e-6`; for any `d=2,...,8`, RMS between
`G(phi)` and `G(phi+1/d)` at most 0.05; any grid pair with circular separation
at least 0.10 and state RMS at most `1e-3`; minimum over all 4096 grid shifts
`c` of RMS between `G(phi)` and `G(-phi+c)` at most 0.01; or landmark score
not having exactly one strict high-score component, strict maximum, and
strict minimum. RMS means `sqrt(sum squared coordinate differences /
coordinate_count)`. Attempt 0..63 is tried in order; first pass is retained.
All 64 failing for any source kills X0. No unit is dropped or replaced.

Before any teacher job, canonical JSON `temporac.x0-manifest.v4` must contain
exactly 40 ordered rows and no other row. Each row has exactly `id`, `split`,
`source_key_hex`, `accepted_attempt`, `coefficient_seed_u64`,
`coefficient_bytes_sha256`, `orbit_grid_sha256`, and
`generator_contract_sha256`. Canonical JSON is UTF-8, sorted keys, no
whitespace/NaN, LF terminated. Its exact file bytes and every row's
length-prefixed field preimage are SHA-256 committed in S0. A missing row,
duplicate, regenerated row, or hash mismatch kills X0.

### 6.2 Eleven internal landmarks, direct offsets, and guarded interpolation

Because the higher directions are orthogonal to `H^T rho`, the landmark score
is a constant plus a positive sine. Its clean maxima are exactly
`a_j=8+32*j`, `j=0,...,10`. These eleven landmarks are internal and define
exactly ten complete landmark-to-landmark traversals. Clean endpoints are
`-8` and `344`, exactly half a clean traversal before/after the first/last
landmark, so neither guard contains an additional landmark.

For a ten-duration vector `d_0,...,d_9` and G4 offset `o in {0,...,31}`, set
`B_j(o)=32+o+sum_(r<j)d_r` and `Q(o)=B_10(o)+32`. The direct target-to-clean
map, with exact right-branch ownership at ties, is

`u_o(q) = -8 + 16*q/(32+o),                         0<=q<B_0;
          8+32*j+32*(q-B_j)/d_j,                   B_j<=q<B_(j+1);
          328+(q-B_10)/2,                          B_10<=q<=Q`.    (F9)

The emitted clock is every integer `q=0,...,Q`. Base views use `o=0`, hence
their first seam is target clock 32. G4 view `o` has first seam exactly
`32+o`. This direct construction replaces and forbids the old `q-o`, “add
32,” preceding-duration extrapolation. The target/decoder mask is exactly
`C(e)=1{B_0<=e<B_10}`. The terminal `B_10` seam lies in the following guard
and is not a target.

All continuous clean channels, including confidence before clipping, are
sampled from the periodic integer knot bank `m=-39,...,375`. PCHIP uses
`scipy.interpolate.PchipInterpolator(axis=0,extrapolate=false)` on the whole
bank. Sinc uses every untruncated tap
`m=floor(u)-31,...,floor(u)+31` with normalized Kaiser weight
`sinc(u-m)*I0(8.6*sqrt(1-((u-m)/32)^2))/I0(8.6)`. Query range is exactly
`[-8,344]`, so the 31-knot guards prove every PCHIP query and all 63 sinc taps
lie in `[-39,375]`. A nonfinite/zero compensated sinc normalizer fails.

The stationary-duration list is `S_stat=(5,8,16,32,64,127,128)`. Source orbit
`n` has exactly one stationary block with all ten durations
`S_stat[n mod 7]`. Its other six blocks are SMF, SFM, MSF, MFS, FSM, FMS with
`S=64,M=32,F=16`, each permutation repeated three times and followed by its
first letter to make ten traversals. Every source therefore has exactly seven
blocks. Linear is train; PCHIP and sinc are heldout. The
heldout 8 sources x 7 blocks x 2 resamplers are 112 base views and exactly
3,584 all-offset views per response seed, without creating a new source unit.

### 6.3 Analytic phase, pulses, masks, and responsibilities

The one analytic canonical phase is relative to the frozen sine landmark:

`theta_star(q)=(u_o(q)-8)/32=u_o(q)/32-1/4,
 z_star(q)=exp(2*pi*i*theta_star(q)),
 e_star(e)=sum_(j=0)^9 1{e=B_j(o)}`.                              (F10)

An integer seam is owned by its right edge. Thus each complete traversal owns
exactly its starting pulse, the ten target pulses are at `B_0...B_9`, and
the terminal seam is excluded. On every X0 return, `certify_target` must emit
base-edge pulse bits bit-identical to `e_star`, the same ten-event count, and
the same traversal ownership; mismatch is `SYNTHETIC_PULSE_MISMATCH` and
kills X0.

For X0 subedge row `r`, analytic `chi_r` is positive clean phase progress
divided by that traversal's exact total; each traversal sums to one within
two ulp. Window analytic phase rate uses the shared `b_w,e`; relative
`zeta_star` subtracts the same beta-weighted median, and `p_star` is the fixed
three-center softmax. Analytic `u_o`, derivatives, duration, offset,
`theta_star`, `e_star`, `chi`, and `p_star` are X0 target/audit metadata.
None enters a TempoRAC encoder, expert, router, or treatment/capacity input.
The isolated X0-only warp-metadata shortcut adversary is the sole declared
negative-control consumer.

## 7. Teacher, integer landmarks, certification, and topology bank

### 7.1 Teacher and exact sources

The static network is `Linear(298,128)-GELU-Linear(128,32)-unit-L2`. The phase
network is `Linear(247,256)-GELU-Linear(256,128)-GELU-Linear(128,2)-unit-L2`
on `y_t` plus static code. The no-bypass decoder is
`Linear(34,256)-GELU-Linear(256,256)-GELU-Linear(256,149)` and sees only phase
and static code. Norm below `1e-8`, nonfinite arithmetic, raw-state/mask/time
input, skip, residual, or alternate decoder abstains.

Teacher gradients use only the 24 X0 train source orbits and their exact
seven linear blocks; the tune inventory is the 8 X0 tune source orbits and
seven linear blocks. That source orbit's single stationary block is the
reference. For
each of the other six blocks and each traversal, evaluate 128 fractions
`r=(j+1/2)/128`, `j=0...127`, at the same analytic clean-phase point. Invert
F9 within that traversal to obtain each block's target source clock, locate its
two emitted integer-clock neighbors with right-tie ownership, and interpolate
their normalized phase by the shortest signed principal arc. Define the phase
term `1-z_ref(r) dot z_view(r)`. Reduce 128 fractions, then ten
traversals, then six nonreference blocks, then 24 source orbits, with equal
weights at every level; no nearest-clock or learned correspondence is allowed.
The 149 reconstruction targets are `[p,b,two-sided direction,confidence]`.
Their mask repeats each joint/bone geometry mask over its two coordinates,
then each corresponding direction mask over its two coordinates, then the 17
joint masks for confidence. With Huber threshold `delta_H=0.05`, `L_rec` is
the sum of masked channel losses divided by the positive mask sum within each
complete traversal, then averaged equally over ten traversals, seven blocks,
and 24 source orbits. `L_corr` uses the reduction in the preceding paragraph.
For every observed retained edge wholly inside a complete traversal, define
the signed principal phase increment in cycles as
`Delta_phi=atan2(cross(z_t,z_(t+1)),dot(z_t,z_(t+1)))/(2*pi)`. Each penalty
mean first averages its valid edges, then ten traversals, seven blocks, and 24
source orbits equally; every denominator must be positive. The objective is:

`L_teacher=L_rec+L_corr+0.25*mean ReLU(-Delta_phi)
                +0.25*mean ReLU(abs(Delta_phi)-0.25)`.              (F11)

There is no natural row, evaluator label, count, period, density, boundary,
contrastive loss, `L_cont`, or PAMS initialization. Three teacher seeds are
retained. Within each seed, discard a checkpoint if the fraction of valid X0
tune observed edges with phase increment `<-1e-12` exceeds 0.01; zero
denominator fails. Among the remaining checkpoints choose minimum complete
X0-tune objective, tie smaller step. From the three per-seed choices, freeze
one teacher by lexicographically minimizing `(tune abstention count, mean
masked tune reconstruction, tune objective, numeric seed)`. All quantities
are X0 tune only; all three missing/rejected fails G1. This one artifact emits
every X0 and natural target.

### 7.2 Two-stage noncircular landmark rule

In each valid run, compute `ell_t=rho dot g_t/sqrt(66)` and
`R=max ell-min ell`; require finite `R>0`. Stage A forms the ordered provisional
list of interior strict local maxima satisfying `ell_t>=max ell-0.05R`.
Stage B computes prominence from this immutable provisional list, never from
the list being filtered. For provisional `a_m`, the left basin is run start
through `a_m-1` for the first candidate and `a_(m-1)+1` through `a_m-1`
otherwise; the right basin is `a_m+1` through run end for the last and
`a_m+1` through `a_(m+1)-1` otherwise. Empty basins reject that candidate.
Prominence is `ell_a-max(min_left,min_right)`; ties in minima use the lower
sample index. Retain iff prominence `>=0.20R`.

Maximal sample-index-connected components above `max ell-0.05R` must each
contain exactly one retained candidate and no retained candidate may lie
outside them. The landmark coordinate is exactly the strict-max integer
sample `L=t`. Parabolic/subsample refinement is deleted globally: it would
shift the F7 seam when adjacent target pace differs. Run endpoints cannot be
landmarks. At least eleven ordered internal landmarks are required for an
X0 target and at least three for a natural target.

### 7.3 Observed phase sets and `certify_target`

Canonicalization normalizes the phase pair, snaps angles within `tau` of zero
to exact `1+0i`, and rotates by the conjugate of the normalized equal-
traversal landmark-start vote. The operational `tau=1e-7`; certification is
repeated at `1e-8,1e-7,1e-6`, requiring identical status, landmarks, seams,
pulses, winding integers, masks, and reasons. Floating phase arrays need not
be bit-equal across different topology homeomorphisms or resamplers.

For any geometry-arc fraction, compensated cumulative `lambda` locates the
unique containing base edge with right ownership; `rho` is its fractional arc
position. Interpolate phase by the endpoint shortest signed principal arc,
`z(rho)=exp(i*(angle(z_left)+2*pi*rho*Delta_phi_edge))`. Interpolate the 149
continuous reconstruction targets linearly in `rho`; a channel is valid only
when its endpoint masks are both true. No phase chord interpolation, nearest
sample, network reevaluation, or coordinate-specific arc is allowed.

For each traversal, evaluate canonical phase at geometry-arc fractions
`0,(m+1/2)/128` for `m=0...127`, and `1`, in that order. Set local unwrapped
phase `A_0=0`; each next value adds the signed principal `Delta_phi` from F11.
Require `abs(A_terminal-1)<=1e-6`, then snap only the ledger copy of the
terminal value to one. The half-open seam ledger contains the start crossing
`(fraction=0,integer=0)` and, on every consecutive node interval, each integer
`n` satisfying `A_left<=n<A_right`, except the already inserted start. A tie at
an internal left node belongs to its outgoing interval; the terminal integer
one has no outgoing interval and is excluded. For a strict crossing, its
fraction is `r_left+(n-A_left)*(r_right-r_left)/(A_right-A_left)`. Map the
fraction through the traversal's frozen geometry-arc interpolation to retained
sample coordinates, then assign it to the unique base edge `[e,e+1)` by right
ownership. A nonpositive/nonfinite denominator, two crossings in one base
edge, any internal crossing, or anything other than the sole start seam fails.
The returned binary pulse is one exactly on that owned start edge and zero on
all other target edges; no peak finder, threshold, or learned period is used.

For traversal `[L_j,L_(j+1))`, the observed phase set is phase at every
integer retained sample `s` with `L_j<=s<L_(j+1)`; the terminal sample is not
inserted. Angles are cycles in `[0,1)`. Circular distance is
`min(abs(a-b),1-abs(a-b))`. Sort by `(angle,sample_index)`, union adjacent
values at distance `<=1e-8`, then union last/first if their circular distance
is `<=1e-8`; the representative is the lowest original sample index. Require
at least five classes. Circular gaps are consecutive representative angles
plus the wrap gap and maximum gap is strictly below 0.25.

`certify_target(track,source_kind)` returns either `CERTIFIED` with immutable
target arrays or `ABSTAIN` with the exact ordered numeric reason vector from
Section 13. Its reconstruction statistic is the masked Huber sum with
`delta_H=0.05` divided by the positive 149-channel mask sum across exactly the
128 midpoint nodes, followed by equal traversal averaging. It requires, for
every traversal: complete support; principal
increments in `[-1e-12,0.25)` with at least five strictly positive observed
edges; no backward seam; 30 of 32 dense phase bins; dense maximum gap
`<=0.10`; winding in `[1-1e-6,1+1e-6]`; exactly one positive owned seam;
the start pulse edge has strictly positive `Delta_phi` and therefore strictly
positive `chi`; binary pulse sum one; reconstruction Huber `<=0.02`; no pair with phase
separation `>=0.10` and 149-channel state RMS `<=1e-3`; unique/stable
landmark; origin magnitude `>=0.95`; landmark geometry RMS `<=0.05`;
leave-one-traversal-out origin change `<=0.01` cycle; and every start vote
within `0.10*pi` of the final origin. All denominators must be positive.

For `source_kind=X0`, returned `chi` is analytic progress from Section 6 and
analytic/certified pulse equality is mandatory. For `source_kind=natural`,
base-edge `chi` is `max(Delta_phi_e,0)` divided by the compensated positive
phase progress of its traversal; the denominator must be positive and the sum must
equal one within two ulp. Natural pulses are the certified positive seam bits.
Natural returns contain no analytic phase, warp, duration, or offset field.
Leading/trailing context remains masked as Section 4.1. A certification
failure emits no target arrays.

### 7.4 Finite topology/homeomorphism fixtures

The topology bank is generated on X0 `U0003`, stationary `d=32`, all ten
traversals, 128 analytic nodes, and has exactly these thirteen ordered rows.
The reason column gives numeric prediction reason and the topology-receipt
detail; accepted rows have empty reason vector:

| id | operator | required result | exact first reason/detail |
|---|---|---|---|
| `T00` | identity | accept | empty |
| `T01` | rotate phase by `+1/8` before landmark canonicalization | accept | empty |
| `T02` | rotate phase by `-1/8` before landmark canonicalization | accept | empty |
| `T03` | `H+`: slope `7/8` on `[0,1/2)`, `9/8` on `[1/2,1)` | accept | empty |
| `T04` | `H-`: slope `9/8` then `7/8` | accept | empty |
| `T05` | constant degree zero | reject | `14 TOPOLOGY / DEGREE_ZERO` |
| `T06` | `theta -> 2 theta mod 1` | reject | `14 TOPOLOGY / HARMONIC_2` |
| `T07` | `theta -> 8 theta mod 1` | reject | `14 TOPOLOGY / HARMONIC_8` |
| `T08` | `theta -> -theta mod 1` | reject | `14 TOPOLOGY / REVERSAL` |
| `T09` | alternate landmark start votes by `0` and `1/2` | reject | `17 ORIGIN / ALTERNATING_STARTS` |
| `T10` | add raw 149-state input to reconstruction decoder signature | reject | `15 RECONSTRUCTION / RAW_BYPASS` |
| `T11` | copy dense state node 0 to node 64 | reject | `16 COLLISION / DENSE_STATE` |
| `T12` | insert two positive seam crossings in base edge 0 | reject | `18 PULSE / TWO_IN_ONE_EDGE` |

`H+(x)=7x/8` for `x<1/2` and `7/16+9(x-1/2)/8` otherwise; `H-` swaps the two
slopes. Accepted rows must exactly equal the reference in landmark indices,
traversal bounds, degree integers, canonical seam coordinates, base-edge
pulse bits, target/decoder masks, and event count. They need only satisfy the
phase tolerances; their float phase array/hash is not an equality field.
Rejected rows emit `ABSTAIN`, the exact numeric/detail pair above, first failing
traversal zero, and no target arrays.

Canonical JSON `temporac.topology-bank.v4` contains the generator contract
hash, input hash, thirteen operator parameter objects, per-array hashes, and
expected-field hashes in this order. The generator source bytes, manifest
bytes, and generated NPZ bytes are independently SHA-256 committed at S0.
Any extra/missing fixture or observed-field disagreement fails G1. Actual
heldout X0 and natural tracks still run the complete certificate; the finite
bank is not a substitute.

## 8. Response graph, NOLA, objective, and one decode

### 8.1 Cached causal graph and two-identity isolation

The encoder is `Linear(269,64)` plus two width-64 residual depthwise-separable
causal blocks, kernel 5, dilations 1 and 2. Each block is LayerNorm,
depthwise convolution, GELU, pointwise convolution, GELU, residual, with bias,
no dropout/batch norm. Ordered branches slow/medium/fast have two identical-
shape blocks with dilation 4/2/1, then `LayerNorm-Linear(64,1)-Sigmoid`.
Branch-only receptive fields are 33/17/9 samples; encoder-plus-branch fields
are 45/29/21; including lag4 raw pose they are 49/33/25.

For each maximal run, the encoder and every branch execute once in increasing
sample order. Edge `e=(t,t+1)` uses the cached branch logit at `t+1`.
Windows only slice cached logits. Buffers are left-zero-padded within the run
and reset, with cue/reference, windows, NOLA, and adjacency, at every run
boundary. Perturbing all pre-gap bytes must leave every post-gap byte equal.

G2 additionally freezes three independently conforming deterministic X0
records: A=`U0003,d=32,o=0`, A'=`U0002,MFS,o=0`, and
B=`U0001,SMF,o=0`. Running `(A,B)` and `(A',B)` requires B's normalized features,
cue/reference, cached logits, sigmoid responses, NOLA numerator/denominator,
quantized response, run/component tables, and prediction archive bytes to be
identical. Swapping whole key/order pairs may only swap the two whole output
archives. This rejects symmetric cross-person coupling that batch permutation
alone cannot detect.

### 8.2 Probability fusion and positive differentiable NOLA

Canonical local window response is

`r_w,e=sum_k a_w,k*sigmoid(z_e,k)`.                                (F12)

Uniform/capacity use exact probability means; logit averaging is forbidden.
With taper
`h_n=1e-3+(1-1e-3)*sin^2(pi*(n+1/2)/127)`, NOLA `edge_mask v_e`, and
windows `Omega(e)`,

`N_e=sum_(w in Omega)e h_w,e*v_e*r_w,e,
 D_e=sum_(w in Omega)e h_w,e*v_e,
 R_e=N_e/D_e`.                                                     (F13)

Binary64 Neumaier sums use window-start order. Every valid decoder edge must
have `D_e>=1e-3`; zero, smaller, or nonfinite values fail. There is no epsilon,
clamp, fallback, or detached fused path. Autograd crosses branch sigmoid,
fusion, `N`, and exact division; cue/gate/taper/mask/`D`/targets are detached.

The fixed gradient fixture is canonical seed 20260815 at initialization on
X0 `U0003`, stationary `d=32,o=0`, all ten target traversals. Only fused
`P(R,p;chi)` is enabled. Inputs, initialized parameter bytes, target ledger,
loss scalar, name-ordered gradient float32 vector, and their hashes are
committed. Two fresh locked-runtime processes must produce identical vectors;
encoder and every branch norm must be finite and `>1e-12`, while cue/router/
taper/mask/denominator/target have no gradient. A central directional
finite-difference check with step `2^-12` must have relative error `<=5e-3`.

### 8.3 X0 and natural pseudo-target reductions

For a probability `x`, binary pulse `p`, and nonnegative row weight `w`, let
`W+=sum_(p=1)w` and `W-=sum_(p=0)w`; both must be finite and positive. Set
`x_bar=min(1-1e-7,max(1e-7,x))` and define
`B=-0.5[(sum_(p=1)w log x_bar)/W+ +
        (sum_(p=0)w log(1-x_bar))/W-]`. The positive margin and negative
valley are the corresponding weighted means
`M+=(sum_(p=1)w max(0,0.75-x)^2)/W+` and
`M-=(sum_(p=0)w max(0,x-0.25)^2)/W-`, and

`P(x,p;w)=B+0.5*M+ +0.5*M-`,
`L_unit=(1/3)sum_k P(r_k,p;v*chi*pi_k)+P(R,p;v*chi)`.              (F14)

`pi_e,k` is the exact NOLA-taper average of window responsibility. For X0,
`chi` and responsibility are the analytic `chi,p_star` from Section 6. For a
certified natural train identity, `chi` is frozen normalized teacher progress
and responsibility is the detached actual cue gate `a_w,k`; no analytic
metadata exists. Each traversal averages ledger rows, then traversals average
within a source unit. X0 further averages seven blocks. Every unit is one
vote independent of samples, duration, windows, or number of people.

Response optimizer step `t` pairs one cyclically shuffled X0 train source
orbit with one cyclically shuffled certified-natural train component; the
component selects one certified identity by its own deterministic cycle. The
step loss is exactly `0.5*L_X0+0.5*L_natural`. A component with no certified
identity is absent only from this conditional training cycle and is still a
K1 uncovered component. Empty X0 or natural cycles fail. Whole source units
are processed in microbatches; numerator gradients are accumulated and
divided by the exact one-X0/one-natural family count before the step. Row
counts never reweight a family or unit.

All G0-eligible natural training identities call `certify_target`. Failures
do not train, receive no partial target, and cannot be relabeled. They remain
in K1 denominators and later population manifests. X0 tune alone selects
response checkpoints; natural development and evaluator labels never select.
Capacity control uses the same sources, `v*chi`, sigmoid placement, and fused
term but omits `pi_k`. Five feature-shortcut jobs use the same balanced cycle;
the X0-only warp-metadata job has no natural row and uses equal X0-unit means.

### 8.4 Quantize once and decode once

After binary64 NOLA, valid `R_e` is rounded once, ties-to-even, to little-
endian float32 `R32_e`. This exact array is serialized and decoded; no backend
may threshold the preceding float64 value. Edge `e` is active iff
`decoder_mask_e=1` and `R32_e>=float32(0.5)`. Components are maximal adjacent
active base-edge intervals within a run. Score is maximum `R32`; location is
the floor of the midpoint of the first and last edge attaining that exact
maximum. Different runs never connect.

The decoder receives the ordered run table and is invoked once per identity.
It returns the total component count. An ineligible/abstaining identity still
invokes it once on empty arrays, internally obtains zero, then serializes
`count=-1`. There is no integral, rounding, NMS, period fit, independent
window decode, or second decision pass.

## 9. Finite operator bank and G4 association

The operator generator is a frozen pure function over inactive-gap lengths
`g=(4,5,8,16,32,64,127,128,129)`, plateau widths `(1,2,4)`, amplitudes
`(0.60,0.75,1.00)`, offsets `0...31`, and layouts `interior,left-boundary,
right-boundary,run-reset`. It yields exactly `9*3*3*32*4=10,368` fixtures.
Let `step=w+g`. Interior starts are `(32+o,32+o+step,32+o+2*step)`.
Left-boundary starts are `(o,o+step,o+2*step)`. For right-boundary, first set
provisional starts `(32+o,32+o+step,32+o+2*step)`, set run length `E` to the
smallest positive multiple of 32 at least `last_start+w+64`, and translate all
three starts by `E-(last_start+w)-o`, so the last plateau's right boundary is
edge `E-o`. Interior/left use the same smallest-multiple `E` rule without
translation. Run-reset has two runs: run 0 edge interval `[0,E0)` contains
starts `(o,o+step)` with `E0` the smallest multiple of 32 at least
`o+step+w+1`; edge `E0` is the zero-mask reset edge; run 1 starts at `E0+1`,
its sole plateau starts `E0+1+o`, and its end is the smallest multiple of 32
greater than `E0+1+o+w+64`. The reset edge is not part of either run. All
plateau intervals must be disjoint and in bounds or generation fails.

Inactive values are `0.10+0.30*b/255`, where `b` is the corresponding byte
from `SHA256("temporac.operator.v4\0"||length-prefixed row key||uint32_be(e))`.
Plateau values are the amplitude, except the lower middle edge has exact
`min(1,amplitude+0.05)` and is the unique maximum. All values are float32.
Truth plateaus, masks, expected components, and generated response bytes are
hashed in a 10,368-row canonical manifest. Every window covering an active
edge receives the same desired float32 edge response before fusion, so the
fixture tests positive NOLA reconstruction as well as the final decoder. The
source generator, row count, Cartesian order, and manifest hash are S0
commitments; there is no random or open-ended fixture generation.

Truth/component bipartite adjacency exists iff their edge sets intersect.
Truth degree 0/>1 is miss/split; component degree 0/>1 is extra/merge.
`truth_negative_components` is the count of maximal nonempty inactive edge
intervals within each run after removing all truth plateaus and reset edges.
Positive margin is the minimum truth-plateau maximum minus 0.5; negative
margin is 0.5 minus maximum outside all truth plateaus. Empty minima fail.
Operator spacings 4/129 are boundary-only validity tests. Trained heldout X0
uses exactly the physical stationary strata `(5,8,16,32,64,127,128)`, both
PCHIP/sinc, three seeds, ten traversals, and all 32 direct offsets. Every
expected stratum must be nonempty; no operator row is claimed physically
realizable.

## 10. Matched arms, shortcuts, natural drift, and SHA boundaries

### 10.1 Arms and cached-response interventions

The comparator set is exactly `(global NUDFT,uniform,capacity-control)` in
that tie order. Local/global/uniform/blocked/shuffled inference reuse the same
frozen cached canonical branch logits. Capacity control is separately trained
with the identical encoder, three branch shapes, total parameters, sigmoid
placement, source cycle, optimizer, and checkpoint rule, but no gate input.
Blocked must equal uniform byte-for-byte. Shuffle is a mechanism ablation,
not a comparator.

### 10.2 Exact shortcut vectors and hashes

Shortcut raw vectors have these exact half-open slices:

| name | dimension and slices |
|---|---|
| `no-pose-timestamp` | 20: joint masks `[0,17)`, frame-valid `[17,18)`, run-relative clock `[18,19)`, clipped `delta_q/4` `[19,20)` |
| `nuisance-only` | 51: confidences `[0,17)`, joint masks `[17,34)`, bone masks `[34,50)`, frame-valid `[50,51)` |
| `pose-shuffle` | 269: the canonical response vector after a fixed within-run sample permutation |
| `static-code-only` | 32: repeated frozen teacher static code |
| `warp-metadata` | 2: X0-only analytic `log du/dq` and `u-floor(u)` |
| `track-length` | 1: `clip(log2(source_length)/12,0,1)` |

Each scalar is first rounded to `<f4`. For raw dimension `d`, projection
entry `(r,c)` consumes bit `c mod 256` from
`SHA256("temporac.shortcut-projection.v4\0" || uint16_be(name_byte_length) ||
name_ASCII || uint32_be(r) || uint32_be(floor(c/256)))`, MSB first; bit 0/1 is
`-1/+1`, divided by `sqrt(d)`, then rounded `<f4`. Rows are exactly
`r=0...268`, columns `c=0...d-1`; no digest crosses a named vector boundary.
The projection matrix C-order bytes and the generator preimage are hashed.

Pose-shuffle orders sample indices by
`SHA256("temporac.pose-shuffle.v4\0"||source_key_raw32||uint32_be(run)||
uint32_be(sample))`, tie lower index. For input channel `c=0...268`, sort by
`SHA256("temporac.pose-channel-order.v4\0"||source_key_raw32||
uint32_be(run)||uint32_be(c))`, tie lower `c`, to obtain permutation `pi`.
For output channel `c`, the MSB of
`SHA256("temporac.pose-channel-sign.v4\0"||source_key_raw32||
uint32_be(run)||uint32_be(c))` gives sign `-1` for zero and `+1` for one.
The exact shortcut value is `x'_t,c=sign_c*x_(perm(t)),pi(c)` before the fixed
projection, with float32 multiplication and C-order serialization. No shortcut
field enters the canonical/capacity model or comparator selection.

### 10.3 Deterministic count-blind natural drift

The paired natural treatment is built only from each G0-eligible run. For
normalized target coordinate `s in [0,1]`, the target-to-original derivative
is exactly `(3/4,31/25,101/100)` on the three consecutive thirds. Their mean
is exactly one, so endpoints are fixed; their maximum maps the largest
admitted gap to `4*31/25=124/25<5`, proving every mapped adjacent span remains
strictly below `P_min`.

For a run whose first/last retained source clocks are `q_first<q_last`, each
retained target clock `q_t` has `s_t=(q_t-q_first)/(q_last-q_first)` and queries
the original track at `q'_t=q_first+(q_last-q_first)F(s_t)`; the emitted clock
remains the original integer
`q_t`. With right-branch ownership at both breakpoints, the exact integrated
map is
`F(s)=3s/4` for `0<=s<1/3`,
`F(s)=1/4+(31/25)(s-1/3)` for `1/3<=s<2/3`, and
`F(s)=199/300+(101/100)(s-2/3)` for `2/3<=s<=1`. For every query
`q'_t<q_last`, choose the unique original retained-clock interval
`[q_j,q_(j+1))` with `q_j<=q'_t<q_(j+1)`; an exact internal-knot tie belongs
to its outgoing/right interval. Query `q'_t=q_last` returns the terminal sample
exactly and opens no interval. No extrapolation, nearest-knot, or inverse grid
lookup is allowed. Within the selected admitted original edge, interpolate
normalized `x,y` and clipped confidence linearly in source-clock coordinates,
clip confidence, derive joint/bone masks from both bracketing endpoints, and
then recompute root-relative pose, bones, teacher directions, every lag,
confidence, cue velocities, and all 215/269 continuous channels. No derived
continuous channel is independently warped, no invalid edge/run is crossed,
and no warp coordinate is a model input. Every mapped span is recomputed and
must pass before G5a despite the analytic bound. Failure is global, not a
filter. The paired original population is called `unwarped/clean`; only
constant-duration X0 is called stationary.

## 11. Metrics, K1/K4 populations, and paired bootstrap

### 11.1 K1 exact component rule

K1 denominators are the already frozen G0-eligible identities and components,
separately by split. An identity is covered iff `certify_target` returns
`CERTIFIED`. A component is covered iff it contains at least one certified
G0-eligible identity. Require both identity and component numerators to be at
least `ceil(0.8*denominator)` in train and development. Empty/missing
denominators fail. Coverage cannot change G0 or omit a later population row.

### 11.2 X0 routing, specialization, shortcuts, and decoder quantities

For arm `a`, let `L_a` be the fused-only functional
`P(R_a,p;v*chi)` from F14, reduced equally within each traversal, ten
traversals, the six nonstationary SMF/SFM/MSF/MFS/FSM/FMS blocks, PCHIP and
sinc at offset `o=0` only, each heldout X0 source orbit, then the three response
seeds. The source's
stationary block is excluded from every K3 `L_a` but retained for K4 and K6.
Define

`M_X0=min_(a in comparator_set)(L_a-L_local)`.                      (F15)

The minimizing arm is reselected per bootstrap; ties follow the fixed order.
Cue-shuffle removal is

`Q_shuffle=(L_shuffle-L_local)/(L_global-L_local)`.                 (F16)

Nonpositive denominator fails. For K4, the population is exactly all heldout
X0 source orbits, both PCHIP/sinc, seven blocks, ten target traversals, and
three response seeds at offset zero. For seed `s`,
`H_s,k,j` is `P(branch_k,p;v*chi*pi_star_j)`, averaged traversal, block,
resampler, then source orbit. Soft responsibility `pi_star_j`, not a hard
column label, defines the population. Aggregate `H_k,j` is the mean of the
three seed values, and

`G_diag=min_j min_(k!=j)(H_k,j-H_j,j)`.                            (F17)

The responsibility fraction is the population sum of `v*chi*pi_star_j`
divided by the sum over all columns before source averaging; every column
must be at least 0.10. K4 requires diagonal improvement for every seed and
aggregate, with paired lower bounds above zero. Empty/nonfinite weight fails.

For shortcut `a`, `G_ref=L_capacity-L_local`,

`R_a=max(0,L_capacity-L_attack,a)/G_ref`.                           (F18)

`G_ref<=0` fails. Boundary error is

`B_err=(miss+extra+split+merge)/(truth_events+truth_negative_components)`. (F19)

The denominator must be positive; margins are tested separately so zero error
cannot hide threshold contact.

Every K3/K4 paired interval has 10,000 source-orbit block draws. The atomic
unit is one ordered heldout X0 source orbit; one PCG64 call draws `M` int64
indices with replacement, and multiplicities apply identically to arms,
columns, resamplers, blocks, traversals, and seeds. Seeds are the unsigned
little-endian first eight SHA-256 bytes of exact ASCII statistic names
`temporac.k3.route.v4`, `temporac.k3.shuffle.v4`,
`temporac.k3.shortcut.NAME.v4`, and `temporac.k4.diag.COLUMN.v4`. Recompute
all nested reductions and strongest-arm choices in each draw. The point uses
one copy; sorted lower/upper indices are 249/9749.

### 11.3 Natural per-person estimand and video-preserving cluster bootstrap

After the vault capability order in Section 14, every joined count must be a
positive integer. For arm `a`, seed `s`, and person `p`, with stub abstention
estimate defined as zero,

`NAE_a,s,p=abs(count_hat_a,s,p-count_gt_p)/count_gt_p,
 OBO_a,s,p=1{abs(count_hat_a,s,p-count_gt_p)<=1}`.                 (F20)

Within each video, average its frozen supplied-valid people; then average
videos equally; then average the three ordered seeds equally. This defines
`E_a` and `O_a`. An empty person/video/seed level, nonfinite prediction, or
duplicate person fails. The strongest comparator `N` minimizes drift `E_a`
with the fixed tie order.

A video is exactly one development-manifest equality class of
`opaque_key_hex`; all person slots with that key are averaged together and no
raw video name is opened. Before G0, the trusted packer must prove that every
slot belongs to exactly one such class and every class to exactly one frozen
source component.

Define

`M_route=E_N,drift-E_local,drift,
 D_clean=E_local,clean-E_N,clean`.                                 (F21)

The same `N` selected on drift is used for its paired clean statistic.

Every natural interval uses one shared 10,000-draw PCG64 stream seeded by the
unsigned little-endian first eight SHA-256 bytes of exact ASCII
`temporac.k7.route-and-clean.v4`. A draw samples the
ordered development component list with replacement in one int64 call. It
then materializes every video in each drawn component with that component's
multiplicity and computes the ordinary equal-video mean over this replicated
video multiset; it never first gives components equal metric weight. People
and three seeds remain paired across every arm, drift, and clean. Each draw
reselects `N` from its drift errors, then applies that same selected arm to the
paired clean draw. This preserves the point estimand's video weighting while
using source components as dependence clusters.

Sort draw statistics. Two-sided bounds are zero-based order 249/9749; a
one-sided 97.5% upper bound is order 9749. Fewer than eight components,
missing arm values, or empty replicated videos fails. The point uses the
one-copy population. K7 requires drift point relative improvement
`M_route/E_N,drift>=0.05` and lower bound of `M_route>0`; if
`E_N,drift=0`, it fails. Clean requires point `D_clean<=0.01` and paired upper
bound `<=0.02`; if `E_N,clean>0`, also require
`E_local,clean/E_N,clean<=1.05`, otherwise require `E_local,clean=0`.
`O_a` is reported descriptively with the same video-first/seed aggregation
and no decision threshold.

## 12. Singular neural runtime, RNG, schedules, and resource ledger

The target runtime is exactly Linux x86_64, CPython 3.12.4, NumPy 2.1.0,
SciPy 1.14.1, PyTorch 2.4.1+cu124, CUDA runtime 12.4, cuDNN 9.1.0.70, NVIDIA
driver 550.54.14, and RTX A6000. A fresh TempoRAC lock records wheel/container
digests, ELF dependency hashes, environment variables, and GPU UUID class;
its canonical JSON SHA-256 is an S0 input. Any version/hash mismatch fails;
the older WARP-PHASE candidate environment is not acceptable evidence.

Model/optimizer tensors are float32; certification, quadrature, NOLA sums,
metrics, and LR arithmetic are float64. Autocast and GradScaler are absent;
TF32 is false for matmul and cuDNN; deterministic algorithms are required;
`cudnn.deterministic=true`, `benchmark=false`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, matmul precision `highest`,
`PYTHONHASHSEED=0`, one process/GPU, and loader workers zero. An operation
without a deterministic implementation fails. Inference batch ordering is
the manifest order and never changes floating reductions.

Seeds are exactly 20260815, 20260816, 20260817. All parameter names are
ASCII-sorted. LayerNorm weight/bias are exact one/zero. Every linear/
convolution weight and bias is initialized on CPU by PyTorch uniform
`[-1/sqrt(fan_in),+1/sqrt(fan_in)]` from a CPU generator seeded by the first
eight little-endian SHA bytes of
`"temporac.init.v4\0"||uint16_be(job_name_length)||job_name||uint64_be(seed)`;
parameters are then copied to CUDA. Python random and global NumPy are not
used; PCG64 streams and Torch CPU/CUDA generators have separate length-
prefixed domains. RNG state is checkpointed.

Optimizer step index `t` is zero-based, `t=0...T-1`. Teacher step `t` selects
source orbit `t mod 24` from a fresh 24-item PCG64
permutation for cycle `floor(t/24)`, seeded from the first eight little-endian
SHA bytes of `H("temporac.teacher-source-cycle.v4",job_name,uint64_be(seed),
uint32_be(cycle))`; that unit contains all seven blocks. Response X0 selection
uses the same rule/domain `temporac.response-x0-cycle.v4`. For `M` nonempty
certified natural components, step `t` selects `t mod M` from the per-cycle
component permutation with domain `temporac.response-component-cycle.v4`.
On a component's visit number `r`, choose `r mod I_c` from a fresh permutation
of its `I_c` certified identities for cycle `floor(r/I_c)`, with domain
`temporac.response-identity-cycle.v4` and component key as an additional
length-prefixed field. Each permutation call is exactly once on int64
`0...N-1`. Cycle manifests and final PCG64 states are hashed in the run
receipt.

Exact ASCII job names are:

- `temporac.execution.v4/teacher/seed=SEED` (three);
- `temporac.execution.v4/response/canonical/seed=SEED` (three);
- `temporac.execution.v4/response/capacity-control/seed=SEED` (three); and
- `temporac.execution.v4/response/shortcut/NAME/seed=SEED` for the six names
  in Section 10.2 (eighteen).

Thus response inventory remains exactly `3+3+18=24`. Global, uniform,
blocked, and shuffled are inference modes, not jobs.

AdamW is `(beta1,beta2)=(0.9,0.999)`, epsilon `1e-8`, weight decay `1e-4`
on every trainable parameter, including biases and LayerNorm scale/shift, and
global gradient clip 1.0. Teacher runs exactly `T=20,000` optimizer steps;
every response job exactly `T=10,000`. Reaching scheduled step `T` and
finishing its checkpoint is success; early stop is forbidden. Checkpoints are
written after every 500 completed steps, hence exactly 40/20 candidates. With
`W=max(1,ceil(0.05T))`, completed-step index `n=1...T`, and base LR `eta0`
(`3e-4` teacher, `1e-3` response),

`eta(n)=eta0*n/W,                                      n<=W;
 eta0*[0.1+0.9*(1+cos(pi*(n-W)/(T-W)))/2],            W<n<=T`.    (F22)

Tune objective is evaluated at every checkpoint only after its complete
write. After successful step T, choose the finite minimum X0-tune objective;
ties use smaller step. Heldout/natural data cannot select. A missing
checkpoint, skipped/replaced batch, rejected nonfinite batch, incomplete
schedule, wall/memory/GPU ceiling, or resume with mismatched RNG/hash fails.
There is no epoch counter or epoch-dependent behavior; optimizer-step `n` is
the sole schedule horizon.

Teacher ceiling remains 6.0 A6000h and 24 GiB per job; response 2.5 A6000h
and 24 GiB; heldout inference 1 A6000h/seed; natural prediction 4
A6000h/seed. Maximum concurrency is two. Ledger remains 18+60+3+12=93
A6000h, with seven unallocated and total ceiling 100. Ceilings are failures,
not early successes, and cannot be reassigned without review.

## 13. Prediction, abstention, receipts, and stubs

### 13.1 Numeric reason vector

Reasons are unique `uint16` values evaluated/appended in ascending contract
order: `1 SOURCE_SCHEMA`, `2 SOURCE_JOIN`, `3 DUPLICATE_CONFLICT`,
`4 CLOCK_ORDER`, `5 NORMALIZATION`, `6 FRAME_SUPPORT`, `7 EDGE_SUPPORT`,
`8 GEOMETRY_LENGTH`, `9 TOO_SHORT`, `10 TEACHER_NUMERIC`, `11 LANDMARK`,
`12 OBSERVED_PHASE`, `13 DENSE_PHASE`, `14 TOPOLOGY`,
`15 RECONSTRUCTION`, `16 COLLISION`, `17 ORIGIN`, `18 PULSE`,
`19 SYNTHETIC_PULSE_MISMATCH`, `20 NOLA`, `21 RESPONSE_NONFINITE`,
`22 RUNTIME`, `23 ARTIFACT_INTEGRITY`, `24 CONFIG_MISMATCH`.
Code zero is reserved and forbidden in a vector. Checks whose prerequisites
failed are not evaluated. The list has no duplicates and is ascending. A
non-abstention has empty reasons; an abstention has at least one.

### 13.2 Prediction NPZ and coordinates

Every arm/seed emits one immutable deterministic NPZ for each condition and
every frozen population identity, including G0/certificate failures. Condition
is exact `uint8` zero for `natural-clean` and one for `natural-drift`; no other
value is legal. Sorted members are:
`abstain u1[1]`, `abstain_reasons <u2[A]`, `component_bounds <i4[C,2]`,
`component_location <i4[C]`, `component_score <f4[C]`,
`condition u1[1]`, `contract_sha256 u1[32]`, `count <i8[1]`,
`decoder_mask u1[E]`,
`edge_mask u1[E]`, `local_person_slot <i8[1]`,
`opaque_sample_key u1[32]`, `response <f4[E]`, and `run_bounds <i4[R,2]`.

For non-abstention, `abstain=0`, `E=T-1`, arrays obey Section 4 coordinates,
`count=C`, and reasons shape is `[0]`. For an abstention/stub, `abstain=1`,
`count=-1`, `E=R=C=0`, and
empty shapes are exactly response/edge/decoder `[0]`, run/component bounds
`[0,2]`, location/score `[0]`; reasons is nonempty. Stub decoder invocation
still occurs on these empty arrays. No population identity is represented by
a missing file. In K7 its scientific estimate is zero, while receipt status
remains abstain.

### 13.3 Canonical receipts and hash preimages

All hashes below are SHA-256. The unambiguous field hash primitive is
`H(tag,fields)=SHA256(tag_ASCII||0x00||for each field:
uint64_be(byte_length)||field_bytes)`. Hex is lowercase. JSON is UTF-8 with
sorted keys, separators comma/colon, `ensure_ascii=false`, `allow_nan=false`,
and exactly one terminal LF; receipt hash is over the complete bytes including
LF. Array member hash is over its exact NPY member bytes; artifact hash is
over exact NPZ bytes.

`temporac.feature-receipt.v4` has exactly keys `artifact_bytes`,
`artifact_sha256`, `contract_sha256`, `members`, `opaque_key_hex`, `schema`,
`slot`, `source_binding_sha256`; each member object has exactly `bytes`,
`dtype`, `name`, `sha256`, `shape`. This detached receipt is the sole location
of feature shard SHA.

Each certified target is one deterministic NPZ with sorted members
`chi <f8[E]`, `contract_sha256 u1[32]`, `edge_mask u1[E]`, `pulse u1[E]`,
`source_kind u1[1]`, `target_mask u1[E]`, and `teacher_sha256 u1[32]`, where
source kind zero is X0 and one is natural. Its
`temporac.target-receipt.v4` has exactly `artifact_bytes`, `artifact_sha256`,
`certificate_status`, `contract_sha256`, `members`, `schema`,
`source_key_hex`, `source_kind`, `source_unit_index`, `teacher_sha256`.
`certificate_status` must be exact ASCII `CERTIFIED`; an abstention has no
target artifact or target receipt. Member objects use the five-key member
schema above. The identity tuple is `(source_kind,source_key_hex,
source_unit_index)`: for natural rows, `source_key_hex` is the 32-byte opaque
sample key and `source_unit_index` is exactly `local_person_slot`; for X0 it is
the 32-byte source key and
`source_unit_index=block_index*96+resampler_index*32+offset`, with block
`0...6`, resampler linear/PCHIP/sinc encoded `0/1/2`, and offset `0...31`.
Unused combinations are absent rather than aliased.

`temporac.prediction-receipt.v4` has exactly `arm`, `artifact_bytes`,
`artifact_sha256`, `checkpoint_sha256`, `contract_sha256`,
`condition`, `feature_receipt_sha256`, `job_name_hex`, `members`, `schema`,
`seed`.
Its `members` objects use the same exact five-key member schema as the feature
receipt.
`temporac.run-receipt.v4` has exactly `code_sha256`, `config_sha256`,
`contract_sha256`, `environment_sha256`, `final_step`, `job_name_hex`,
`optimizer_sha256`, `ordered_checkpoint_sha256`, `resource`, `rng_sha256`,
`schema`, `seed`, `source_cycle_sha256`, `status`, `upstream_receipt_sha256`.
`ordered_checkpoint_sha256` and `upstream_receipt_sha256` are ordered arrays
of lowercase digests. `resource` has exactly `completed_steps` integer,
`gpu_seconds` finite nonnegative float64, `max_cuda_bytes` nonnegative integer,
and `wall_seconds` finite nonnegative float64. Unknown/missing keys fail.

`target_count_root_sha256` is the SHA-256 of one canonical JSON array ordered
by `(opaque_key_hex,slot)` with exactly keys `opaque_key_hex`, `slot`, and
`teacher_target_count` in each row. It contains every and only certified
development identity; the nonnegative integer count is the sum of that frozen
teacher target's pulse bits and reads no evaluator label.

`temporac.g5a-receipt.v4` has exactly `checkpoint_root_sha256`,
`code_sha256`, `contract_sha256`, `environment_sha256`,
`evaluator_code_sha256`, `feature_root_sha256`,
`population_manifest_sha256`, `prediction_clean_root_sha256`,
`prediction_drift_root_sha256`, `receipt_root_sha256`, `schema`,
`stub_inclusive_artifact_count`, `target_count_root_sha256`, and
`vault_join_commitment_sha256`. The artifact count is exactly
`4 arms*3 seeds*2 conditions*402 identities=9,648`. The capability-request
grant's aggregate `prediction_root_sha256` is
`H("temporac.prediction-pair-root.v4",raw32(clean_root),raw32(drift_root))`.
The capability-request ledger record has
exactly `g5a_receipt_sha256`, `previous_record_sha256`, `schema`, `sequence`;
sequence starts at zero and the first previous digest is 64 ASCII zeros.
Each subsequent record increments by one and binds the exact preceding record
bytes. `temporac.capability-grant.v4` has exactly `evaluator_code_sha256`,
`g5a_receipt_sha256`, `invocation_count`, `join_commitment_sha256`,
`prediction_root_sha256`, `schema`, `vault_root_sha256`; invocation count is
the integer one. `temporac.g5b-receipt.v4` has exactly
`g5a_receipt_sha256`, `join_cardinality`, `join_commitment_sha256`, `schema`,
`status`, `vault_root_sha256`; status is exact ASCII `PASS` and join
cardinality is exactly 402. `temporac.k7-receipt.v4` has exactly
`g5a_receipt_sha256`, `g5b_receipt_sha256`, `metric_payload_sha256`, `schema`,
and `status`; status is exact ASCII `PASS` or `FAIL`. All five objects use the
canonical JSON bytes and hash rule above; unknown/missing keys fail.

Source binding preimage is `H("temporac.source-binding.v4", split,
source_pickle_raw_hash,uint32_be(object_ordinal),uint32_be(slot))`. Contract,
config, code, environment, manifest, job-name, and upstream hashes always hash
bytes, never path strings or JSON object iteration order. Feature,
checkpoint, prediction, and receipt roots are distinct regular non-symlink
capability roots.

### 13.4 No adaptation of existing interfaces

TempoRAC interfaces are package-private new records. The public
`PoseSequence` remains immutable `[frames,33,3]` MediaPipe pose plus one
frame mask and FPS; `CountResult` remains nonnegative scalar count, positive
period, three diagnostic counts, confidence, and nonempty period stream. The
local-frequency readout remains its synthetic-frozen single-person diagnostic.
Existing `pams.warp_phase.types.FeatureShard`, `FeatureBatch`, vault records,
receipts, model outputs, configs, and packers remain isolated WARP-PHASE
objects. TempoRAC does not inherit, wrap, alias, reinterpret, or modify any of
them, and its loader imports none of them.

## 14. Gates, capability order, and K7 falsifiers

### S0 — specification commitment

Commit v4 bytes, environment lock, source allowlist, 40-row X0 manifest,
13-row topology bank, 10,368-row operator bank, source-code/config hashes,
runtime flags, job names, source cycles, interfaces, and constants. Open no
data. Any mismatch fails.

### S1 — separately authorized acquisition

After a future trusted-packer grant, run G0 and K0 only.

**G0:** exact trusted schema, seven-member identity archives, duplicate/
support/geometry decisions, detached receipts, 402-row count-blind manifest,
and four acquisition floors.

**K0:** pass only if G0, total source/vault key commitments, component mapping,
forbidden-key scan, and root separation pass. No evaluator value is opened.

### S2 — teacher and response inputs

**G1:** all three teacher jobs complete; tune-only artifact selection passes;
every heldout X0 target certifies; analytic/certified pulses match; all finite
topology rows meet exact accept/reject fields; PCHIP/sinc phase error `<=0.02`
cycles and pulse pullbacks agree. Empty strata fail.

**K1:** exact identity/component coverage rule in Section 11.1 on frozen G0
denominators.

**G2:** exact channels/receptive fields, cached run logits, lag/reset/guard
masks, positive NOLA proof, fixed fused-only gradient fixture, pre-gap
isolation, and two-identity untouched-output counterfactual.

**K2:** every X0/natural response target has immutable `CERTIFIED` receipt,
source-specific `chi`, pulse hash, and mask. Failure has no partial target.

### S3 — fixed response jobs and synthetic decisions

**G3:** exactly 24 successful fixed-step response jobs, schedules/checkpoints,
family/component cycles, reductions, optimizer/runtime hashes, probability
fusion, finite outputs, and 93-hour ledger. A cap reach before scheduled
completion fails.

**K3:** on drift X0 require point/bootstrap-lower `M_X0>0`, point/lower
`Q_shuffle>=0.80`, and point/bootstrap-upper `R_a<0.10` for all six shortcuts.

**K4:** use exactly Section 11.2 population/reductions; require every per-seed
and aggregate diagonal improvement and lower bound `>0`, `G_diag>0`, and each
soft responsibility fraction `>=0.10`.

### S4 — operator, decoder, and resampler decisions

**G4:** on all finite operator rows and every heldout trained physical stratum,
require zero association errors, all offsets, batch-permutation equality,
`D>=1e-3`, and worst positive/negative margins each `>=0.10`.

**K5:** require `B_err=0` separately for operator layouts and every
seed/resampler/physical stratum, including nonempty boundary packs, with both
margins `>=0.10`.

**K6:** require all heldout source orbits to certify, matched-state phase error
`<=0.02`, complete analytic/certified and PCHIP/sinc pulled-back pulse-bit
equality, tau-invariant decisions, and identical final event counts. Phase
float arrays themselves need not be equal.

### G5a -> capability grant -> G5b -> K7

**G5a immutable commitment:** without a vault capability, recompute and commit
all contract/code/environment/feature/population/checkpoint/prediction/receipt
hashes; prove complete stub-inclusive prediction cardinality; prove natural
mapped spans `<5`; and freeze per-certified-development-identity teacher target
counts. Write one canonical JSON receipt with exclusive-create `O_EXCL`, hash
its bytes with SHA-256, and append that digest plus the next monotonic sequence
number to the evaluator-owned append-only capability-request ledger. G5a
computes no human statistic.

Only a successful immutable G5a receipt may request the one-time evaluator
capability. The grant names the committed prediction root, 402-row join
commitment, vault root, evaluator code hash, and one invocation. Training
credentials and optimizer entry points are absent. The grant is consumed when
one non-resumable evaluator process starts; that process must run G5b and then
K7 in order. A G5b failure destroys the capability without running K7, and a
restart requires a newly reviewed G5a commitment rather than reusing the grant.

**G5b join integrity:** after grant, verify exact 268 train plus 134
development key/slot joins, raw interval syntax, count type, source length,
and commitment cardinality. It returns only a canonical hash-bound integrity
receipt whose `g5a_receipt_sha256` equals the exact G5a receipt digest;
no row returns to model/training code and no scientific threshold is applied.
The same evaluator process hashes this receipt into the K7 receipt before any
metric payload is finalized.

**K7 sole natural decision:** first inspect every raw period interval on
exactly the frozen 268+134 pilot identities. Any `e-s<P_min=5`, malformed
interval, non-increasing order, interval overlap, count/period-list
inconsistency, or absent identity kills the entire
pilot without filtering. Second, for every certified development identity,
map each frozen teacher pulse edge to its source-clock start `q_e`. Require
exactly one such start in every evaluator interval `[s,e)`, no pulse start
outside the union of those intervals, and consequently teacher target count
equal to both interval-list length and `count_gt`. This detects split/merge
cancellation within a track rather than checking only the total. A single
mismatch kills globally; it cannot change targets, population, checkpoint,
comparator, or trigger a rerun. Finally compute the
Section 11.3 drift and paired unwarped/clean estimands on all 134 development
identities, including zero-estimate stubs, and apply the fixed point/CI rules
once. Failure ends TempoRAC.

The only precedence is

`S0 -> S1:G0,K0 -> S2:G1,K1,G2,K2 -> S3:G3,K3,K4 ->
 S4:G4,K5,K6 -> freeze natural artifacts -> G5a -> capability grant ->
 G5b -> K7`.                                                       (F23)

No later stage may alter an earlier population, target, mask, job, checkpoint,
arm, threshold, archive, or hash.

## 15. Paper-visible evidence and stopping boundary

Exactly three paper-visible blocks remain:

1. the three-seed natural local-versus-strongest-nonlocal drift result and
   paired unwarped/clean retention check;
2. the heldout three-by-three X0 specialization matrix and `G_diag`; and
3. certificate coverage, analytic/certified/resampler pulse equality, and
   zero decoder boundary errors/margins.

Primitive-unit kills, identity isolation, shortcut matrices, operator
inventories, duplicate receipts, runtime hashes, resource logs, and firewall
traces are audit/appendix evidence. They do not become claims or a fourth
block.

This proposal permits no WARP-PHASE, pivot, learned router, integral fallback,
new dataset, fourth seed, additional response job, evaluator-label training,
paper-visible block, or positive performance statement. Any impossible
definition, empty required population, nonfinite/zero denominator, unsupported
edge, archive/hash mismatch, certification mismatch, unfinished fixed-step
job, resource ceiling, capability leak, raw-period contradiction, teacher-unit
mismatch, or K7 failure rejects the route. The next legitimate step is a
fresh TempoRAC experiment plan and server preflight, not a launch under prior
authority.
