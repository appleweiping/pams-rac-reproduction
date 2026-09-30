# Round 3 Refinement: TempoRAC

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

The revision remains the same TempoRAC graph: a per-identity causal encoder,
three shared fast/medium/slow response branches, a detached window-local
irregular-clock cue, a fixed soft router, response-level fusion, exact NOLA, and
one final per-identity decode. Its only dominant claim is that local routing
beats the strongest frozen nonlocal comparator under piecewise pace drift
without material stationary degradation. The topology teacher is one
supporting mechanism check, not a second contribution.

The revision does not add a trainable module, route, arm, seed, claim, or
paper-visible evidence block. It does not substitute WARP-PHASE, a pivot,
scene-level counting, a single-person task, an integral decoder, a learned
router, a GRU, ACF fusion, canonical PAMS-TCC initialization, or L_cont.

## Simplicity Check

One executable contract, temporac.execution.v3, now owns every numerical
choice. It has no method fork and no deferred choice. Compared with the prior
proposal, the architecture is unchanged; the added detail is contractual
closure around the audited v44 clock, exact arc pooling, topology
certification, differentiation, deterministic synthetic data, interfaces,
gates, and accounting. A condition that cannot be satisfied fails the route;
there is no fallback.

Exactly three paper-visible evidence blocks remain:

1. the primary local-routing result on the natural supplied-track pilot;
2. the synthetic expert-specialization mechanism check;
3. the synthetic topology, resampler, and decoder-validity check.

All other receipts, shortcut attacks, ablations, and gate diagnostics are
appendix or audit material and cannot become additional paper claims.

## Changes Made

- Replaced the false adjacent-clock requirement by the preregistered
  no-hidden-complete-cycle support premise 1 <= delta-q < 5, made every
  unsupported interior edge disqualify the whole identity, and froze
  split-specific unconditional acquisition floors before any teacher or
  response training.
- Corrected the squared static moment to the exact piecewise-linear arc
  integral and made sample, subedge, traversal, pulse, seam, pullback, and
  serialization ownership one half-open convention.
- Consolidated all teacher, topology, landmark, collision, reconstruction,
  origin, pulse, and degree requirements into the evaluator-owned
  certify_target(track) function.
- Froze the exact natural feature schema, duplicate-disagreement test,
  confidence semantics, teacher and encoder channel slices, lag boundaries,
  masked geometry length, and irregular-clock NUDFT.
- Made the NOLA response path differentiable, froze the causal edge alignment
  and full receptive fields, and added run-reset, byte-isolation, and
  fused-only gradient receipts.
- Froze a finite deterministic synthetic population, its target-to-clean
  clocks, interpolation, source splits, resampler splits, topology rejection,
  response jobs, RNGs, ordering, accumulation, resource ceilings, and
  cap-reached failure.
- Separated unconditional acquisition eligibility, conditional teacher
  coverage, artifact-only vault opening, and the sole natural scientific
  decision so no gate is circular.
- Defined the global comparator, all shortcut adapters, projection bits,
  bootstrap, gains, margins, aggregations, component units, tie rules, and
  every zero-denominator outcome.
- Introduced TempoRAC-specific feature, prediction, receipt, loader, and
  evaluator-boundary interfaces without adapting the existing
  PoseSequence, CountResult, or local-frequency interfaces.

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

## 1. Contract identity, precedence, and contribution boundary

This proposal is the sole normative specification named
temporac.execution.v3. It supersedes every earlier TempoRAC response-unit,
decoder, control, clock, synthetic-population, and evaluation clause. The
seven architecture invariants, the audited physical v44 schema, and both
verbatim Problem Anchors remain authoritative. If any earlier text conflicts
with this contract, v3 controls without importing the conflicting clause.

The dominant claim is singular:

> Identity-indexed window-local soft routing over shared fast, medium, and slow
> response experts improves continuous supplied-track counting relative to
> the strongest frozen matched nonlocal comparator under deterministic
> piecewise tempo drift, while retaining stationary performance.

The only supporting mechanism check is that the three branches specialize by
relative tempo when one primitive physical repetition is certified as one
response winding. Topology certification is a validity prerequisite, not a
second novelty claim.

No part of this document authorizes implementation, data or server access,
test or sealed access, result generation, title selection, paper editing, or
submission. A future explicitly authorized trusted packer may execute the
boundary in Section 3. This refinement itself grants no such authority.

## 2. Frozen architecture and seven invariants

For each supplied identity i and retained sample t, a causal encoder emits
h_i,t. Three identical sigmoid response branches k in {fast, medium, slow}
emit r_i,t,k. A detached local spectral cue creates fixed probability gates
a_i,w,k for window w. The three response probabilities are fused at
probability level, accumulated by exact NOLA, and decoded once for the
identity. All learned weights are shared across identities; no identity
identifier is embedded.

The seven invariants are:

1. person isolation: every state, mask, window, component adjacency, and
   normalization accumulator is keyed by one supplied identity;
2. shared experts: the three branches have identical topology and parameter
   count and differ only through learned weights;
3. relative local routing: a detached local cue routes by clock-blind
   fast/medium/slow columns relative to that identity's run reference;
4. response-level fusion: branch sigmoid probabilities are combined by the
   exact convex gate before NOLA;
5. NOLA: overlapping tapered response windows are divided by their exact
   positive overlap denominator;
6. one final decode: the connected-component decoder is called exactly once
   per identity, even when that identity contains several valid runs; and
7. no human evaluator target in learning: human count, period, density, and
   boundary fields never enter teacher selection, response training, routing,
   checkpointing, or prediction.

There is no L_cont, canonical PAMS-TCC initialization, ACF fusion, learned
router, GRU, integral decoder, WARP model, pivot substitution, or additional
trainable module.

## 3. Future trusted-packer authorization boundary and natural clock

### 3.1 Unique physical source and whitelist

The unique natural pilot source is the audited v44 320-grid train and
development cache. No other natural cache is admissible. A future authorized
trusted packer must verify the canonical path, byte hash, regular-file status,
and immutable in-memory bytes before deserialization. The unprivileged
feature process receives exactly these seven source fields:

| Field | Exact source representation | Model role |
|---|---|---|
| motion | little-endian float32, shape [P,320,17,3], C order, finite | x, y, confidence only |
| person_mask | bool, shape [P] | supplied-valid denominator and identity mask |
| frame_mask | bool, shape [P,320] | support only |
| sampled_frame_indices | little-endian int64, shape [320], nondecreasing | physical clock and edge support |
| source_length | Python int greater than one; serialized as little-endian int64 scalar | nonembedded validation metadata only |
| opaque_sample_key | 64 lowercase ASCII hex bytes | routing metadata only |
| local_person_slot | nonnegative Python int; serialized as little-endian int64 scalar | routing metadata only |

The loader accepts a ZIP_STORED NPZ with exactly the seven sorted NPY members,
NPY version 2.0, C order, no object/Unicode/void dtype, no executable member,
no symlink, no extra member, no duplicate member, and allow_pickle=False.
Maximum compressed and uncompressed size are each 64 MiB per identity.
Unknown, missing, reordered, or noncanonical members fail before payload
arrays are exposed. Opaque key, slot, source length, archive path, and archive
hash are never tensors.

The 17 joints use the COCO order nose, left eye, right eye, left ear, right
ear, left shoulder, right shoulder, left elbow, right elbow, left wrist,
right wrist, left hip, right hip, left knee, right knee, left ankle, right
ankle. The third motion channel is confidence, never depth.

### 3.2 Confidence, duplicate collapse, and normalization

All arithmetic in packing and certification is binary64. A raw joint is valid
exactly when its frame mask is true, all three raw values are finite, and its
clipped confidence c=minimum(1,maximum(0,c_raw)) is strictly greater than
0.20. Invalid joint x, y, and c become exact +0.0.

At a duplicate clock q, form a root from the mean of whichever hips 11 and 12
are valid and a provisional scale from the median positive distance between
that root and the mean of whichever shoulders 5 and 6 are valid over
first-occurrence clock representatives. The scale is maximum(median,1e-3).
For every pair a,b in a duplicate group, masks and root validity must be
bit-identical and at least eight joints must be mutually valid. With
w_j=min(c_a,j,c_b,j), the confidence-normalized disagreement is the square
root of sum_j w_j||x_tilde_a,j-x_tilde_b,j||^2 divided by sum_j w_j, where
x_tilde is root-centered and track-scale normalized. The weight denominator
must be positive, the disagreement must be at most 0.02, and maximum clipped
confidence disagreement must be at most 1e-6. These are the single frozen
duplicate tolerances.

An agreeing group averages x/y by clipped-confidence weight in binary64 and
uses maximum clipped confidence; a zero joint weight makes that joint
invalid. It is rounded once to little-endian float32. A conflicting group is never
averaged or removed to salvage the remainder: the whole identity abstains,
and the group is recorded with member indices, maximum disagreements, mask
differences, and source shard hash. This receipt is preserved in G0. A
noncontiguous reappearance of a clock, an empty group, or absence of a finite
positive scale also makes the whole identity abstain.

After collapse, subtract the valid-hip root per sample and divide x/y by the
single track scale. A frame is valid when at least eight joints are valid and
at least one hip is valid. The normalized scale, confidence threshold, and
all masks are frozen before any teacher or response computation.

### 3.3 The conditional no-hidden-complete-cycle domain

Let retained clocks be q_0 < ... < q_T-1 after duplicate collapse. The
supported minimum physical period is preregistered as P_min=5 source-frame
intervals. An interior source edge is acquisition-supported exactly when both
endpoint frames are valid and

1 <= q_t+1 - q_t < P_min = 5.                                      (F1)

This is a conditional supported-domain premise: because a complete supported
cycle occupies at least five source-frame intervals, a gap of one through
four cannot contain a whole supported cycle. Endpoints do not prove the
absence or shape of an interior path. The later teacher condition
delta<0.25 certifies only the learned principal phase arc on an admitted
edge; it is not evidence that the acquisition premise is true.

An admitted edge may still contain a seam crossing; certify_target detects
and owns that crossing. Admission asserts only the registered physical
support premise and never infers the interior path from its endpoints.

The v44 construction q_j=clip(round(j(L-1)/319),0,L-1) admits every adjacent
clock gap under this rule if and only if L<=1277; L>=1278 necessarily has an
unsupported adjacent gap. This arithmetic fact is not an observed pass-rate
claim. Any retained interior gap that violates the frozen acquisition
inequality, any conflicting
duplicate clock, or any absent source-clock support makes the entire identity
feature-ineligible. The identity is not
rescued by training or counting separate surviving segments. There is no
imputation, bridge, inferred hidden cycle, or segment-wise count rescue.

Invalid pose frames delimit maximal valid runs for operator, attack, and
isolation fixtures. For a natural count-bearing identity, after trimming
leading and trailing invalid frames, any invalid frame or unsupported
coordinate edge inside the remaining span makes the whole identity
feature-ineligible; a surviving span needs at least 17 samples and 16
supported edges. Thus a natural identity is never rescued by summing
observable fragments around a missing interior. The general graph still
resets at every maximal valid-run boundary and calls its decoder once, which
is required for fail-closed fixtures and abstention receipts.

### 3.4 Unconditional G0 denominator and exact split floors

A supplied-valid identity is every person_mask=true slot in the audited
source, before pose, clock, topology, or label filtering. The frozen
denominators are 268 train identities and 134 development identities. A
source component belongs to the denominator when it contains at least one
supplied-valid identity; the frozen denominators are 18 train components and
9 development components. A component is acquisition-eligible when it has at
least one acquisition-eligible identity. Component identity and association
are trusted-packer audit metadata and are never model inputs.

G0 is count-blind and unconditional. Before any synthetic teacher or response
training, it reports every supplied-valid identity, every exclusion reason,
every conflict receipt, the identity numerator, and the component numerator
separately for train and development. It passes only if all four integer
conditions hold:

train identities >=215 of 268, train components >=15 of 18,
development identities >=108 of 134, and development components >=8 of 9. (F2)

The ratios are also reported, but the integer inequalities decide. A missing
denominator, zero denominator, mismatched audited total, absent source
component, duplicate identity, or failed floor hard-stops the route. Counts
are not pooled across splits. No length distribution or pass rate is asserted
by this proposal.

Conditional certificate coverage is separate and later: after a frozen
teacher exists, the numerator is identities for which certify_target returns
CERTIFIED and the denominator is the already frozen G0 feature-eligible
population. It cannot change G0 membership.

If, after all artifacts and predictions are frozen, the evaluator vault shows
any natural ground-truth period shorter than P_min or any other direct
contradiction of the registered supported-domain premise, the entire pilot
fails globally. The contradicting identity is not filtered, the premise is
not retuned, and results are not reported.

## 4. Exact ownership, normalization, and feature construction

### 4.1 One half-open convention

Every real sample coordinate x is first snapped to the nearest integer when
its distance is at most eta_x=1e-12; an exact half-integer is not snapped.
Integers index retained samples. Base edge e is [e,e+1). Consecutive
landmarks L_j and L_(j+1) own traversal [L_j,L_(j+1)). An integer sample s
belongs to traversal j exactly when L_j <= s < L_(j+1). An edge portion
belongs to traversal j through the positive-length intersection
[max(e,L_j), min(e+1,L_(j+1))). Zero-length intersections are discarded.

A crossing at coordinate c belongs to the unique traversal whose half-open
interval contains c and to base edge floor(c). If c is an integer, it belongs
to the edge beginning at c, never the edge ending there. A terminal crossing
at L_(j+1) belongs to the next traversal. No terminal crossing is duplicated.
At the final retained sample there is no right edge, so a crossing there
invalidates the target track.

The same convention governs static pooling, phase increments, topology,
pulse bits, chi weights, target-to-clean pullback, event association, and
serialization. Traversal tables are sorted by increasing L_j; subedge rows
are sorted by traversal then base edge; pulse arrays are sorted by base edge.
An exact traversal-boundary ownership tie always goes to the right traversal
as stated above. All remaining serialization-order ties use lower traversal
index and then lower edge index; if those are identical, equality is exact
and only one row exists.

### 4.2 Geometry and channels

The normalized pose p_t is the 34-vector of root-centered normalized x/y in
COCO order. The 16 directed bones are
(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),
(11,13),(13,15),(12,14),(14,16),(0,5),(0,6),(0,1),(0,2).
Their 32-vector is b_t. Geometry is g_t=[p_t,b_t] in R^66.

For an admitted edge, coordinate mask M_e,d is true only if the corresponding
joint or both bone endpoints are valid at both edge endpoints and the mask is
unchanged over every subedge being integrated. Let D_e=sum_d M_e,d. The
masked geometry length is

lambda_e = sqrt((66/D_e) sum_d M_e,d (g_t+1,d-g_t,d)^2),            (F3)

and is invalid when D_e<32 or the result is nonfinite. When a fractional
landmark subdivides an edge by fraction rho, its length is rho lambda_e; no
pose interpolation across a masked coordinate is permitted.

For teacher input at an interior run sample t, let v^-=g_t-g_t-1 and
v^+=g_t+1-g_t on common support. Normalize each chord, add them, and
normalize the sum to obtain v_hat_t in R^66. Either chord norm or the sum norm
at or below 1e-8, nonfinite arithmetic, missing two-sided support, or a run
endpoint makes the corresponding direction exact zero with false mask. This
is an approximate clock-blind local direction, not an exact warp invariant.

The teacher y_t in R^215 has this exact slice table:

| Slice | Width | Value |
|---|---:|---|
| [0,34) | 34 | p_t |
| [34,66) | 32 | b_t |
| [66,132) | 66 | two-sided v_hat_t; zeros at both run endpoints |
| [132,149) | 17 | clipped confidences |
| [149,182) | 33 | 17 joint plus 16 bone geometry masks |
| [182,215) | 33 | 17 joint plus 16 bone edge-direction masks |

The response encoder x_t in R^269 has this exact slice table:

| Slice | Width | Value |
|---|---:|---|
| [0,34) | 34 | p_t |
| [34,66) | 32 | b_t |
| [66,83) | 17 | clipped confidences |
| [83,117) | 34 | p_t-p_t-4 |
| [117,151) | 34 | p_t-p_t-2 |
| [151,185) | 34 | p_t-p_t-1 |
| [185,202) | 17 | current joint masks |
| [202,218) | 16 | current bone masks |
| [218,235) | 17 | lag-4 pair masks |
| [235,252) | 17 | lag-2 pair masks |
| [252,269) | 17 | lag-1 pair masks |

Each lag is valid only when both joint samples are valid and lie in the same
maximal valid run. The first h samples of a run have exact-zero lag h and
false masks. Source length, clocks, delta-q, keys, slots, component IDs,
warp metadata, and evaluator values are absent from y_t and x_t.

### 4.3 Arc-length static code

For traversal j with clipped subedges a, let L_j=sum_a lambda_a. It is valid
only when L_j>0 and all 66 geometry coordinates and all 149 continuous
teacher channels are supported throughout. For continuous teacher channel
d, with endpoint values A_a and B_a on a subedge, define its arc mean and
second raw moment by

m_j,d = L_j^-1 sum_a lambda_a(A_a+B_a)/2,
q_j,d = L_j^-1 sum_a lambda_a(A_a^2+A_a B_a+B_a^2)/3.              (F4)

For K traversals, bar-m_d=K^-1 sum_j m_j,d,
bar-q_d=K^-1 sum_j q_j,d, and
s_d=sqrt(max(bar-q_d-bar-m_d^2,0)). The 298-vector [bar-m,s] is the sole
static-MLP input. All sums use binary64 Neumaier compensated accumulation in
sorted subedge and traversal order. The formula is the exact integral of the
square of a piecewise-linear channel over geometry arc length. Subdividing
any edge at any fraction gives the same real integral; the fixture requires
the two binary64 evaluations to agree within 2 ulp after serialization.
Every traversal has one vote. No sample-weighted statistic is allowed.

## 5. Exact irregular-clock local cue

### 5.1 NUDFT construction

The cue consumes normalized geometry velocities only; it receives no human
labels and no warp metadata. For each maximal valid run, an edge midpoint is
m_e=(q_e+q_e+1)/2 and its raw duration is d_e=q_e+1-q_e. Windows contain 128
sample slots and 127 edge slots. Starts are 0,32,64,... while a full window
fits, followed by max(0,T-128) if absent; a shorter run has one exact-length
window at start zero. No edge from another run enters.

For each of 66 coordinates, retain edges whose coordinate mask is true.
The scalar signal on such an edge is
(g_e+1,d-g_e,d)/(q_e+1-q_e).
For coordinate d let their count be r_d and sorted midpoints be
x_0,...,x_(r_d-1). The trapezoid weights are
kappa_0=(x_1-x_0)/2,
kappa_(r_d-1)=(x_(r_d-1)-x_(r_d-2))/2, and
kappa_j=(x_(j+1)-x_(j-1))/2 internally. The midpoint Hann is
h_j=sin^2(pi(x_j-x_0)/(x_(r_d-1)-x_0)). At least three retained midpoints and
positive span are required.

Set z_j=(x_j-(x_0+x_(r_d-1))/2)/(x_(r_d-1)-x_0) and
w_j=kappa_j h_j. Fit
v_j=alpha+beta z_j by the binary64 weighted normal equations. With
S_r=sum_j w_j z_j^r, the determinant is det=S_0 S_2-S_1^2; the coordinate is
invalid unless S_0>0, S_2>0, and det>=1e-12 S_0 S_2. The two-by-two system is
solved analytically in the order alpha=(T_0 S_2-T_1 S_1)/det,
beta=(T_1 S_0-T_0 S_1)/det. All sums are compensated.

For residual a_j, the NUDFT at frequency f is

A_d(f)=sum_j w_j a_j exp(-2 pi i f x_j),                            (F5)

on n_f=48 log-spaced periods
P_l=exp(log(5)+l log(128/5)/47), with f_l=1/P_l and l=0,...,47. The
normalized coordinate power is
|A_d(f)|^2/(S_0 sum_j w_j a_j^2). The denominator must be finite and greater
than 1e-12; otherwise that coordinate is invalid. The window spectrum is the
compensated mean over valid coordinates. D is the number of valid
coordinates, n is the number of window edges having at least eight jointly
valid pose joints, and Q is the physical midpoint span x_last-x_first over
those n edges, with Q=0 when n<2. Reliability requires
n>=16, Q>=16, D>=16.

Let s_l be the compensated mean power over the D coordinates and
p_l=(s_l+1e-8)/sum_b(s_b+1e-8). The denominator must be finite and strictly
positive. The cue and confidence are
u_w=sum_l p_l log f_l and
gamma_w=min(1,n/32) min(1,Q/128)(D/66)
max(0,1+sum_l p_l log(p_l+1e-8)/log 48).
Reliability additionally requires gamma_w>=0.15. Without valid support,
p_l=1/48 exactly, u_w is its compensated log-frequency mean, and gamma_w=0.
This spectral direction estimate is only approximate under local
reparameterization; it is not claimed as an exact warp invariant.

### 5.2 Run reference and fixed router

The global matched nonlocal arm uses one full-run NUDFT with the same
detrending, frequency grid, masks, and normalization, using all admitted
edges of that run. Let m_e be the number of local windows covering edge e.
For routing, local reference weight is
beta_w=gamma_w sum_e-in-w lambda_e/m_e. The weighted median bar-u sorts by
(u_w,window_start) and chooses the first value whose cumulative weight
reaches half the total. It requires at least three reliable windows and a
finite positive total; otherwise every gate in the run is uniform. This
allocates each geometry-arc element once across overlapping windows.
Reference membership is fixed after the cue pass and reset at the next run.

The three fixed column centers are
mu_slow=-log(1.5), mu_medium=0, mu_fast=+log(1.5), with
sigma=log(1.5)/2. Define the cue-only softmax tilde-a by

tilde-a_w,k = exp(-(u_w-bar-u-mu_k)^2/(2 sigma^2)) /
              sum_l exp(-(u_w-bar-u-mu_l)^2/(2 sigma^2)),
a_w = gamma_w tilde-a_w + (1-gamma_w)(1/3,1/3,1/3).                (F6)

An unreliable window has gamma_w=0 and therefore exact uniform gates. The
stable softmax subtracts the largest logit. A zero or nonfinite denominator
fails the job; it is not replaced. The global arm replaces u_w by the one
full-run u in every window but retains the same frozen local bar-u and beta
receipt, so its gate is constant within a run. Cue, reference, confidence,
and gate tensors are detached and have requires_grad=false.

## 6. Frozen synthetic population X0

### 6.1 Runtime, identity, and source splits

Synthetic construction is frozen to CPython 3.12.4, NumPy 2.1.0,
SciPy 1.14.1, and IEEE-754 round-to-nearest ties-to-even binary64. Every RNG
is numpy.random.Generator(numpy.random.PCG64(seed)). Seeds are derived as the
unsigned little-endian integer represented by the first eight bytes of
SHA-256 over the ASCII domain string, a zero byte, uint32-big-endian source
ID, uint32-big-endian attempt, and uint32-big-endian view ID. Fourier
coefficients use domain temporac.x0.coeff.v3 and view ID zero.

There are exactly 40 source orbits U0000 through U0039. U0000-U0023 are
synthetic train, U0024-U0031 tune, and U0032-U0039 heldout. Membership is by
numeric ID and never shuffled or regenerated across splits. The stable
source-unit key is SHA-256(temporac.x0.v3, zero byte, uint32-big-endian ID).
Source units are ordered by their 32 digest bytes unsigned lexicographically.

The fixed COCO17 base B in joint order is:

| j | x | y | j | x | y |
|---:|---:|---:|---:|---:|---:|---:|
|0|0.00|0.90|9|-0.38|0.15|
|1|-0.05|0.95|10|0.38|0.15|
|2|0.05|0.95|11|-0.12|0.00|
|3|-0.10|0.93|12|0.12|0.00|
|4|0.10|0.93|13|-0.14|-0.40|
|5|-0.18|0.65|14|0.14|-0.40|
|6|0.18|0.65|15|-0.15|-0.85|
|7|-0.30|0.40|16|0.15|-0.85|
|8|0.30|0.40||||

Only the x/y coordinates of elbows, wrists, knees, and ankles
J=(7,8,9,10,13,14,15,16) move. Let H in R^(66 by 16) be the fixed linear map
from those ordered x/y displacements to [p,b], using the Section 4 bone order
and fixed base hips/root. Let rho be the landmark vector from Section 7 and
w=H^T rho/||H^T rho||. A zero norm fails the entire generator.

For source n and attempt a, draw six independent length-16 vectors in C order
from Uniform[-1,1]. In draw order, modified Gram-Schmidt each vector twice
against w and every earlier accepted vector, then unit-normalize; a norm at
or below 1e-10 rejects the attempt. Every dot product and norm is accumulated
in ascending coordinate order with binary64 Neumaier summation. Call the
resulting ordered pairs
(u_h,v_h), h=2,3,4. With E_J inserting a length-16 vector into only the
moving coordinates, the clean orbit is

G_n(phi)=B+E_J[0.12 w sin(2 pi phi)
          +sum_h=2^4 (0.03/h)(u_h cos(2 pi h phi)
                              +v_h sin(2 pi h phi))].               (F7)

Thus the fundamental is aligned to the frozen landmark projection and every
higher Fourier direction is orthogonal to it and to the other higher
directions. Confidence is
clip(0.90+0.05 sin(2 pi phi+2 pi j/17),0,1). All masks are true.

Evaluate candidates on r/4096, r=0,...,4095. Reject nonfinite values,
coordinates outside [-2,2], scale below 0.1, geometry arc below 1, or tangent
norm below 1e-6. Reject if for any d=2,...,8 the grid RMS between G(phi) and
G(phi+1/d) is at most 0.05; if any pair with circular separation at least
0.10 has state RMS at most 1e-3; or if the minimum over grid shifts c of the
RMS between G(phi) and G(-phi+c) is at most 0.01. On the cyclic grid, the
Section 7 score must have exactly one strict high-score component, one
strict maximum, and one strict minimum. These finite checks are
generator-only asymmetry, injectivity, primitive, and landmark diagnostics;
they never certify a natural target. The first passing attempt in 0,...,63
is retained. Failure for any ID kills X0; no unit is dropped or replaced.

### 6.2 Views, target-to-clean map, and interpolation

The canonical clean source contains ten traversals with clean period P_0=32
and boundary coordinates 0,32,...,320. For a target duration vector
d_0,...,d_9, let b_0=0 and b_j+1=b_j+d_j. On target interval
[b_j,b_j+1), the target-to-clean map is

psi(q)=32[j+(q-b_j)/d_j], with derivative psi'(q)=32/d_j.           (F8)

The endpoint q=b_10 maps to 320. The output clock is every integer
q=0,...,b_10: it is full cadence and contains no duplicate or missing clock.
The local log-frequency shift is log psi'(q); faster target motion has
positive shift. The model receives none of psi, psi', d_j, source ID, view
ID, or interpolation metadata.

The exact nonempty stationary-duration list is
S=(5,8,16,32,64,127,128). Source n has one stationary block with all ten
d_j=S[n mod 7]. It also has the six blocks SMF, SFM, MSF, MFS, FSM, and FMS,
where S=64, M=32, F=16 and a three-letter permutation is repeated three times
then followed by its first letter to make ten traversals. Every source
therefore has exactly seven blocks. Train has 168 linear views, tune has 56
linear views, and heldout has 112 views: PCHIP and sinc for each of seven
blocks of eight units.

For G4 heldout inference only, each of the 112 base views is evaluated at all
32 event-to-window offsets o=0,...,31 without creating a new source unit.
Extend the clean integer knot table periodically from u=-32 through u=352.
Replace q by q-o in the target-to-clean map, use the preceding traversal duration for the
leading partial interval, add 32 to psi, and emit full-cadence target clock
q=0,...,b_10+o so every query remains within the extended knot table. The
first complete seam is therefore at target edge o.
Leading and trailing partial traversals are mask-only and never targets.
This yields exactly 3,584 heldout offset views per response seed and makes
the trained-output all-offset quantifier executable; offset is evaluator
metadata and is not fed to the model.

Linear interpolation is coordinatewise between integer clean samples.
PCHIP is scipy.interpolate.PchipInterpolator over integer knots -32,...,352
for G4 offsets and 0,...,320 otherwise,
axis zero, extrapolate=false. For sinc at query u, use integer taps
m=max(u_min,floor(u)-31),...,min(u_max,floor(u)+31) and weights
sinc(u-m) I0(8.6 sqrt(1-((u-m)/32)^2))/I0(8.6), where sinc(0)=1 and
sinc(x)=sin(pi x)/(pi x) otherwise. Divide by the exact compensated sum of
tap weights; a zero or nonfinite sum fails the view. Coordinates use this
normalized sum, confidence uses linear interpolation then clipping, and all
masks stay true. Here (u_min,u_max) is (-32,352) for an offset view and
(0,320) otherwise. No extrapolation or random boundary rule exists.

Analytic clean phase z-star(q)=exp(2 pi i psi(q)/32) is evaluated in binary64
and seam-snapped. A pulse e-star is one at each half-open crossing of
psi(q)/32 through an integer and zero elsewhere. For each response window,
let b_w,e be the exact midpoint-Hann-trapezoid valid-edge quadrature from
Section 5. Let nu-bar_w=sum b_w,e log psi'(m_e)/sum b_w,e and let beta_w be
the clock-blind geometry-arc reference weight from Section 5.2. Then
zeta-star_w is nu-bar_w minus the beta-weighted median of reliable nu-bar
values, and p-star_w,k is the fixed three-center softmax of zeta-star_w.
All denominators must be positive. Traversal chi, quadrature, zeta-star, and
p-star use analytic progress and one traversal/window vote, never output
sample density. None of this metadata is fed to the model.

The train resampler is piecewise linear. PCHIP and sinc are unseen resamplers
and are heldout only. X0 tune alone selects checkpoints. X0 heldout is opened
only by frozen gates. No generated label derives from a natural count,
period, boundary, density, or component field.

## 7. Teacher and exact evaluator-owned certify_target

### 7.1 Teacher

The existing teacher graph is retained exactly. Its static MLP is
Linear(298,128)-GELU-Linear(128,32), followed by unit-L2 normalization. Its
phase MLP is
Linear(247,256)-GELU-Linear(256,128)-GELU-Linear(128,2) on the concatenation
of y_t in R^215 and the static code in R^32, again followed by unit-L2
normalization. Its no-bypass reconstruction MLP is
Linear(34,256)-GELU-Linear(256,256)-GELU-Linear(256,149) on phase in R^2 and
static code in R^32. It has no raw-state, mask, time, skip, residual, or
alternate decoder input. A normalization norm below 1e-8 or nonfinite value
abstains.

The teacher objective is mask-normalized Huber reconstruction of the 149
continuous channels with transition 0.05, plus mean one-minus phase dot
product at analytic X0 correspondences, plus 0.25 times mean
ReLU(-delta), plus 0.25 times mean ReLU(abs(delta)-0.25). Every denominator
must be finite and positive. There is no L_cont, count, period, density,
boundary, contrastive, or canonical-initialization loss.

For each of the three teacher seeds, tune selects the lowest tune objective,
then earliest epoch. For artifact selection, compute on synthetic tune the
fraction of valid observed edges with delta<-1e-12; a zero denominator fails.
Only candidates at or below 0.01 remain, and they are ordered by number of
tune abstentions, mean masked reconstruction, epoch, then seed. The first is
the one frozen teacher used for every target. This population tolerance can
select an artifact only; it never certifies a target track. Heldout PCHIP and
sinc play no role in checkpoint or artifact selection.

### 7.2 Landmark counter and bit order

SHA streams concatenate digest bytes for counter 0,1,..., where the counter
is uint32 big-endian. Bits are consumed byte by byte, most-significant bit
first; bit zero maps to -1 and bit one to +1. The first 66 bits of
SHA-256 domain temporac.landmark.v3 define rho in {-1,+1}^66. The landmark
score is ell_t=rho dot g_t/sqrt(66).

Within each maximal valid run, R=max ell-min ell. R must be finite and
positive. Normalized landmark score is (ell-min ell)/R. A retained landmark candidate t must be a strict local maximum
against both immediate valid neighbors, satisfy max ell-ell_t <=0.05R, and
have prominence at least 0.20R. Its left minimum is the smallest score from
the previous retained candidate exclusive through t exclusive; its right
minimum is the smallest score after t through the next candidate exclusive.
At a run boundary the endpoint score is included as the corresponding
minimum. Equal minimum scores choose the lower sample index. Prominence is
ell_t-max(left_min,right_min).

Samples above max ell-0.05R are adjacent when their sample indices differ by
one. Every resulting maximal high-score component must contain exactly one
strict candidate; zero or multiple candidates makes the track abstain. The
candidate score is that component's maximum. Refine its coordinate with a parabola
through the immediate neighbors. The offset is
0.5(ell_t-1-ell_t+1)/(ell_t-1-2ell_t+ell_t+1), used only for a strictly
negative denominator of magnitude greater than 1e-12. Zero, positive, or
nonfinite denominator gives offset zero. Clip to [-0.5,0.5], add t, and apply
eta_x snapping. A run endpoint cannot be a landmark.

At least three ordered landmarks L_0<L_1<L_2 are required, hence at least two
complete traversals. The traversal counter starts at zero for [L_0,L_1);
landmark and pulse bits serialize in this traversal-major order.

### 7.3 Canonical seam, phase increments, and origin

From the phase-MLP coordinates define raw unit phase z_t. A zero or
nonfinite norm abstains. Let S_tau act on a unit complex z: if its wrapped
angle has absolute distance at most tau from zero, return exactly 1+0i;
otherwise return z. The sole operational tau is 1e-7. Certification is
repeated at tau in {1e-8,1e-7,1e-6}; the canonical phase arrays, pulse bits,
landmark ownership, traversal counts, and all pass/fail decisions must be
bit-identical at all three values.

For each clipped subedge inside a traversal, linearly interpolate the
pre-normalized phase-MLP pair at both endpoints, normalize, apply S_tau, and compute
delta=wrap(angle(z_right conj(z_left)))/(2 pi), with wrap in [-pi,pi).
The exact tie angle +pi maps to -pi. A closing increment from the terminal
left limit back to the traversal start is included only in degree and origin
checks; it never owns a pulse or chi row. All increment sums use sorted
binary64 Neumaier accumulation.

At each traversal start L_j, interpolate the pre-normalized phase pair, normalize,
and call it o_j. The origin is the normalized compensated sum of o_j. A zero
sum abstains. Rotate every phase by the conjugate origin and reapply S_tau.
There is one origin vote per traversal, never per sample.

For the observed canonical phase samples in one traversal, convert angles to
cycles in [0,1), sort ascending, greedily merge consecutive values whose
circular distance is at most 1e-8, and then merge the last and first clusters
when their wraparound distance is at most 1e-8; a cluster representative is
its lowest original sample index. Certification requires at least five
resulting distinct phase samples. The circular gaps are the ascending
differences plus one minus the last-to-first difference, and their maximum
must be strictly less than 0.25. This is a sample-compatible observed-path
condition, not a dense-bin quota.

### 7.4 The sole certification contract

certify_target(track) is evaluator-owned and returns either CERTIFIED plus
immutable target arrays or ABSTAIN plus one ordered reason list. No
teacher-side or population-level rule can override it. A target-bearing
track first splits every original edge at the fractional landmarks. Within
each resulting traversal, compute cumulative masked geometry arc and evaluate
128 half-open equal-arc nodes at fractions r/128, r=0,...,127. At a node,
linearly interpolate only the fully supported 149 continuous channels,
attach the constant true 66 masks, recompute the existing teacher phase and
reconstruction, and never feed an interpolated clock. These dense nodes are
certificate-only evaluations of the same teacher, not a module, input,
target, or loss. Every complete retained-landmark traversal must individually
satisfy all of the following:

- every traversal subedge has complete retained geometry and direction masks;
- no source-clock, invalid-pose, conflict-clock, or run boundary is crossed;
- every phase increment is in [-1e-12,0.25);
- no increment is less than -1e-12, so there is no backward seam crossing;
- at least five observed traversal edges are valid and have strictly positive
  principal increment;
- at least five canonical phase samples survive the frozen 1e-8 circular
  deduplication, and their maximum circular gap is strictly less than 0.25;
- on the 128 equal-arc certificate nodes, at least 30 of the 32 half-open
  canonical phase bins [b/32,(b+1)/32) are occupied and the maximum circular
  phase gap is at most 0.10;
- the compensated traversal increment sum, including the topology-only
  closing increment, lies in [1-1e-6,1+1e-6];
- exactly one positive seam crossing occurs in the half-open traversal;
- every pulse increment is exactly in {0,1}, and their traversal sum is one;
- dense-node reconstruction mean masked Huber error is at most 0.02, with
  Huber transition 0.05 and denominator exactly 128 times 149; a zero or
  nonfinite denominator abstains;
- collision is absent: among the 128 equal-arc certificate nodes, any pair
  with circular phase separation at least 0.10 has 149-channel continuous
  state root-mean-square distance strictly greater than 1e-3;
- the high-score landmark component is unique, the fractional landmarks are
  ordered, its canonical phase diameter is at most 0.05 cycle, and its best
  normalized score exceeds every score outside a 0.10-cycle neighborhood by
  at least 0.10;
- all half-open ownership rules hold; and
- the origin magnitude before normalization is at least 0.95, maximum
  66-channel landmark-geometry RMS distance from the coordinatewise landmark
  median is at most 0.05, every leave-one-traversal-out origin changes by at
  most 0.01 cycle, and every start vote has wrapped angular distance at most
  0.10 pi from the final origin.

Reconstruction Huber terms are summed over p, b, velocity, and confidence
after their fixed normalizations. Masks select terms but are not reconstructed.
The mean has no scale estimated from a track. The 30-of-32 rule applies only
to the 128 interpolated certificate nodes and is never a raw-sample quota;
the raw period-five traversal remains admissible when its five observed
edges meet the strict raw rules above.

The pulse crossing coordinate is the unique root of the unwrapped canonical
phase integer level on a positive subedge, solved by linear interpolation in
unwrapped phase. It is snapped and assigned by Section 4.1. Two crossings
assigned to one base edge, one crossing assigned to two traversals, or a
crossing without a right edge is a collision and abstains.

For resampler comparison, each traversal is evaluated at the same 128
canonical clean phases r/128, r=0,...,127. The inverse of psi on a traversal is analytic
from the frozen target-to-clean map; exact boundary ties use the right traversal. Phase is interpolated
from the pre-normalized phase-MLP pair, normalized, origin-rotated, and snapped.
The matched-state phase criterion is maximum wrapped phase difference at the
128 points <=0.02 cycles. The target pulse crossing c is pulled to clean
coordinate psi(c), snapped at eta_x, and assigned to the right clean base
edge on an integer tie. The maximum wrapped phase-error criterion above must
pass between linear, PCHIP, and sinc heldout views, and the complete
pulled-back pulse bit vector must be bit-equal. Floating-point phase arrays
are not required to be bit-equal across interpolators. The approximate
direction feature is not used as an exact warp invariant.

The returned subedge ledger contains
(traversal_id, base_edge, left_fraction, right_fraction, chi, pulse).
For traversal j, chi is the analytic clean phase progress of that subedge
divided by the traversal's total clean phase progress. It is nonnegative and
the compensated traversal sum must equal one within 2 ulp. The serialized
base-edge pulse is the OR of its ledger pulse bits; the no-collision rule
makes this equivalent to their sum.

A population diagnostic allowing at most one percent negative edges selects
the single teacher only by the exact Section 7.1 precedence. It never
certifies a track. Every target-bearing track still calls certify_target and
abstains on one violating traversal. The contract also rejects degree-zero, harmonic
degree, reversal, origin-bypass, and collision attacks. An
orientation-preserving homeomorphism attack is required to preserve the
canonical seam, phase, and pulse bits after landmark canonicalization; it is
accepted when it does so. The contract never states that all homeomorphism
attacks are rejected.

## 8. Response model, causal alignment, and NOLA

### 8.1 Encoder, branches, and edge alignment

The encoder is Linear(269,64), then two width-64 residual
depthwise-separable causal blocks with kernel 5 and dilations 1 and 2. Each
block is LayerNorm, depthwise causal convolution, GELU, pointwise
convolution, GELU, residual, with bias, no dropout, and no batch
normalization. The ordered branches (slow,medium,fast) each have two
identical-shape width-64 residual blocks and dilation respectively (4,2,1),
then LayerNorm-Linear(64,1)-Sigmoid. All three branches have identical
operation and parameter count. The branch-only receptive fields are
respectively 33,17,9 samples; the full encoder-plus-branch receptive fields
in the 269-channel x sequence are 45,29,21. Including the lag-4 feature
builder, the full raw-pose receptive fields are 49,33,25 samples for
slow,medium,fast. All convolutions are strictly left-padded within a maximal
valid run, and invalid inputs and block outputs are exact zero.

For base edge e=(t,t+1), branch response r_e,k is the causal branch output at
sample t+1. There is no response before the first sample of a run and no
extra response after its final sample. An edge response is valid exactly when
the edge is admitted, both endpoint model masks are valid, and the complete
branch receptive field contains no sample from a previous run; missing
prehistory inside the same run is exact-zero causal padding with false lag
masks. This definition applies identically in training, fusion, decoding,
controls, attacks, and serialization.

At every maximal valid-run boundary, reset encoder and branch convolution
buffers, cue/reference membership, all N and D accumulators, window lists,
spectral buffers, and connected-component adjacency. A fixture perturbs every
pre-gap input byte and requires every post-gap response, N, D, R, mask,
component, and prediction byte to remain identical. Failure is global.
Despite per-run resets, the decoder function is called once per identity with
the ordered list of run responses.

### 8.2 Uniform probability-level fusion and capacity control

The canonical local response in window w is

r_w,e = sum_k a_w,k sigmoid(z_w,e,k).                               (F9)

The uniform nonlocal arm replaces every a_w,k by 1/3. The global nonlocal arm
uses the same router formula with the one full-run NUDFT frequency in every
window. The capacity control has three identical sigmoid branches with the
same encoder, widths, dilations, parameters, optimizer, and training jobs as
the canonical graph and fuses their probabilities exactly as
(sigmoid z_1+sigmoid z_2+sigmoid z_3)/3. It has no gate input. Logit averaging
is forbidden in every arm.

### 8.3 Exact differentiable NOLA

Windows use 128 sample slots, 127 edge slots, stride 32, and the terminal
start rule in Section 5.1. For local edge index n=0,...,126, the exact
positive taper is
h_n=1e-3+(1-1e-3)sin^2(pi(n+1/2)/127); padding contributes zero. Let v_e be
the detached binary valid-edge mask and Omega(e) the windows covering e.
Then

N_e = sum_w in Omega(e) h_w,e v_e r_w,e,
D_e = sum_w in Omega(e) h_w,e v_e,
R_e = N_e/D_e.                                                       (F10)

The sums are binary64 Neumaier accumulations in increasing window-start
order. Before any response threshold is frozen, an operator proof and
fixture must establish D_e>=1e-3 for every valid edge. Division is then by
the exact positive D_e. There is no epsilon, clamp, threshold bias, or
fallback. D_e=0, D_e<1e-3, or nonfinite N_e or D_e fails the job.

Autograd remains connected through every branch sigmoid, the fusion
numerator, N_e, and the exact division producing R_e. Cue, gate, taper, masks,
denominator, and targets are detached constants. Chi is constructed outside
the autograd graph as a binary64 loss weight. A fused-only gradient receipt
runs one fixed positive/negative synthetic batch with every auxiliary loss
disabled. It requires finite encoder gradient norm >1e-12 and finite
gradient norm >1e-12 for every branch, while cue, router, taper, masks,
D_e, and targets have requires_grad=false and exact zero/absent gradients.
Any violation fails G2.

### 8.4 Response objective and decoder

Use only the detached zeta-star and p-star construction frozen in Section 6;
a zero quadrature or reference denominator rejects the source unit. For an
edge, define
omega_w,e=h_w,e/sum_j-in-Omega(e) h_j,e and
pi_e,k=sum_w-in-Omega(e) omega_w,e p-star_w,k. Each denominator is positive
and the three pi columns sum to one within 2 ulp. Warp metadata and p-star
never enter the model.

For the sorted certified subedge ledger, let p_j,e in {0,1} and chi_j,e be
its analytic clean-progress weight. For probabilities x, binary targets p,
and nonnegative weight w, set Z_+=sum w p and Z_-=sum w(1-p). Both must be
finite and positive. Define B as minus one half the positive normalized sum
p log(max(x,1e-7)) minus one half the negative normalized sum
(1-p)log(max(1-x,1e-7)); define M_+ as the positive normalized sum of
max(0,0.75-x)^2 and M_- as the negative normalized sum of
max(0,x-0.25)^2. The fixed response functional is

P(x,p;w) = B + 0.5 M_+ + 0.5 M_-,
L_resp = (1/3) sum_k P(r_k,p;v chi pi_k) + P(R,p;v chi).            (F11)

Here v is exactly the valid-edge mask repeated onto ledger rows, chi is
exactly the physical-progress weight, and multiplication order is binary64
v then chi then pi. The fused term remains connected through the exact NOLA
division.
Every capacity-control branch uses P(r_ctl,k,p;v chi), omitting only pi_k,
and its uniform probability-level fusion uses P((sum_k r_ctl,k)/3,p;v chi).
Thus every control branch and fused term has exactly v chi, the identical
sigmoid and nonlinear placement, positive/negative normalization, capacity,
optimizer, and checkpoint rule. A v-only or chi-only implementation fails.

The decoder threshold is theta=0.50. Within each run, edge e is active when
R_e>=theta and its mask is true. Event components are maximal sets of active
edges adjacent by base-edge index difference one inside that run. A
component score is max R_e; its location is the floor of the midpoint of the
first and last edges attaining that maximum. Components in different runs
are never adjacent. The decoder returns the sum of component counts over the
ordered runs in one function call. An empty ineligible identity still makes
one decoder call returning internal zero, while its prediction serializes as
abstain with count -1. Nonfinite R or malformed runs abstain.

## 9. Operator fixtures and G4 event association

The deterministic operator bank uses pulse spacings
(4,5,8,16,32,64,127,128,129), pulse widths (1,2,4), amplitudes
(0.60,0.75,1.0), and all 32 start offsets. Negative fixtures have every
non-event edge in [0,0.40]; positive plateaus are perturbed by the fixed
signed SHA sequence so their unique maximum exceeds both neighboring
inactive values. Fixtures include overlap boundaries and every run-reset
position.

Let truth event j own its exact nonempty plateau T_j and prediction component
m own its maximal active-edge set C_m. Their bipartite association has an
edge exactly when T_j intersects C_m. Truth degree zero is a miss, truth
degree above one is a split, component degree zero is an extra, and component
degree above one is a merge. Zero errors therefore makes association
one-to-one without a matching heuristic. Positive margin is
min_j(max_e-in-T_j R_e-theta). Negative margin is
min_e-not-in-union(T_j)(theta-R_e); an empty set in either minimum fails.

Operator fixtures quantify spacings 4 and 129 because they test boundary
behavior, not physical target certification. Trained certified outputs are
quantified only at the physically realizable stationary list
(5,8,16,32,64,127,128). For each of these seven strata, each of PCHIP and
sinc, and each of the three response seeds, the heldout bank must contain at
least one source unit and ten certified traversals at every one of the 32
offsets. Under U0032-U0039 and
the modulo-seven assignment, spacing 5 has two source units and every other
spacing has one. The exact observed nonempty stratum list must equal the
seven-value list; any empty, extra, or underfilled stratum fails.

G4 is one gate with two sources. On every deterministic operator fixture and
every frozen trained-bank response just specified, it requires zero misses,
extras, splits, and merges; worst positive and negative margins each at least
0.10; all 32 offsets; NOLA D>=1e-3; and identical outputs under batch
permutation. Only trained-bank rows require physical certify_target status.
No operator row is used to claim physical realizability.

## 10. Matched arms, shortcuts, and deterministic feature-only drift

### 10.1 Exact arm reference and comparator set

The canonical local arm, global NUDFT arm, uniform arm, blocked arm,
within-run shuffled arm, and capacity control share the same eligible
identities, encoder, branch definitions, optimizer, batch ledger, decoder,
seeds, and stopping rule. The frozen nonlocal comparator set is exactly
{global NUDFT, uniform routing, capacity control}. In K7 the strongest
nonlocal comparator is the member with the smallest three-seed natural error;
ties use the listed order. Selection occurs inside each bootstrap replicate
as well, which is conservative and cannot affect training. The
within-run shuffle is a mechanism ablation, not a candidate comparator.

The blocked arm replaces all cue coordinate signals by exact zero before the
NUDFT, must emit gamma=0 and exact uniform gates, and must be bit-identical to
the uniform arm. It is a path-integrity assertion, not another evidence arm.

Shuffle acts separately inside each maximal run. Reliable window gates are
cyclically shifted by one in increasing window-start order; zero or one
reliable window is unchanged and recorded. No gate crosses a run boundary.

The clean stationary population is the X0 stationary views with constant
duration S[n mod 7]. The deterministic piecewise-drift population contains
the X0 drift views plus a feature-only transform of every G0-eligible natural
development run. For a natural run spanning q_a to q_b, set s=(q-q_a)/
(q_b-q_a) and define a target-to-original map by integrating piecewise
constant slopes (12/19,27/19,18/19) on [0,1/3), [1/3,2/3), [2/3,1],
respectively. The slopes have mean one, so endpoints are fixed. Original
normalized pose is linearly interpolated only inside admitted edges of that
run onto the unchanged integer q grid. No invalid edge, conflict, or run
boundary is crossed; masks and clocks are unchanged. The transform uses no
vault field and is frozen before vault opening. Every supplied-valid but
G0-ineligible identity occupies the population with its existing abstention
in both unwarped and drift arms; it is not transformed, dropped, or replaced.

### 10.2 Six shortcut attacks and exact adapters

Each shortcut has the same 269-channel encoder interface and the same
encoder/branch/fusion capacity. For an attack with raw dimension d, define a
fixed projection A in R^(269 by d). Generate bits from SHA-256 over
temporac.attack, the ASCII attack name, and uint32-big-endian row-major
counter, digest bytes and bits in the Section 7.2 order. Map zero to -1, one
to +1, divide by sqrt(d), and use zero bias. The projection is fixed,
untrained, and hashed.

The six raw attack inputs are:

1. no-pose timestamp: 17 joint-mask bits, one frame-valid bit, run-relative
   clock (q-q_start)/max(q_end-q_start,1), and clip(delta-q/4,0,1), d=20;
2. nuisance-only: the 17 masks, 17 clipped confidences, frame-valid bit,
   clip(delta-q/4,0,1), and analytic interpolation fraction
   psi(q)-floor(psi(q)), d=37;
3. pose shuffle: normalized pose, confidence, and masks are permuted together
   within each run by sorting sample indices by SHA-256 of
   temporac.pose-shuffle.v3, source-unit digest, run index, and sample index;
   clock order stays fixed and exact hash ties use lower sample index;
4. static-code-only: the teacher's 32-dimensional fixed static projection is
   repeated at every sample, d=32;
5. warp-metadata: analytic log psi' and
   psi(q)-floor(psi(q)), d=2, only on
   X0 where these values exist; natural rows are absent rather than zeroed;
6. track-length: clip(log2(source_length)/12,0,1), d=1, with source_length
   used only in this explicit audit attack.

For pose shuffle, the already 269-channel result is passed through a fixed
signed permutation: output channel r reads input channel
sort_index[r], where sort_index is the ordering of SHA digests over
temporac.pose-channel-permutation.v3 and uint32-big-endian channel; exact
digest ties use the lower channel. No adapter learns a scale or
normalization. Attacks cannot enter the canonical model or comparator set.

## 11. Determinism, bootstrap, and numerical statistics

### 11.1 Ordering and accumulation

All natural identities are ordered by the 32 decoded opaque-key bytes then
local slot. All source components are ordered by
SHA-256(temporac.component.v3, zero byte, UTF-8 component token), compared as
unsigned bytes. All seeds are ordered 20260815, 20260816, 20260817. All
metric sums use binary64 Neumaier accumulation in that order and divide by
the exact positive unit count. A missing, duplicate, zero-count, nonfinite,
or unordered unit fails.

Training-order seed is the unsigned little-endian first eight SHA-256 bytes
of ASCII temporac.train-order.v3, zero byte, UTF-8 job name, zero byte,
uint64-big-endian model seed, and uint32-big-endian epoch. PCG64 calls
rng.permutation on the int64 array 0,...,N-1 exactly once. A minibatch
contains eight whole source units and is never split. Gradients accumulate whole minibatches until at least 8192 valid
subedge rows have been seen; the minibatch that crosses the boundary stays
whole. The final positive tail at an epoch executes one optimizer step
normalized by its exact positive and negative weights. A batch with any
zero/nonfinite branch or fused normalizer is rejected before backward and the
entire job fails immediately; it is not skipped or replaced. A generator
candidate rejection consumes the next attempt seed, up to the 64-attempt cap.
Optimizer state, accumulation counters, and tail buffers reset only at a job
start, never between ordinary minibatches.

### 11.2 Exact measured quantities

For a response bank B, let C_B be its event count and C_star the certified
event count. Synthetic count error is the traversal-normalized mean
|C_B-C_star|/C_star over source units, one source unit one vote. If C_star=0,
the source unit fails the bank.

For local routing L and the strongest nonlocal comparator N, define per
source component c and seed s the video-first normalized counting errors
E_L,c,s and E_N,c,s, and the paired margin

M_route = mean_s mean_c (E_N,c,s-E_L,c,s).                          (F12)

The component mean gives each component one vote; within a component, first
average people in a video, then videos. Empty levels fail. Positive
M_route favors local routing.

For X0 arm a, let L_a be the source-orbit mean of the
traversal-normalized response functional, averaged within traversal, then
block, then source orbit, then equally over the three seeds. With frozen
nonlocal set A={global,uniform,capacity-control}, the exact local margin is

M_X0 = min_a-in-A (L_a-L_local).                                    (F13)

The minimum and its arm identity are recomputed in every bootstrap draw;
ties use global, then uniform, then capacity-control. Positive M_X0 means
local routing beats the strongest nonlocal X0 arm.

The shuffle-removal fraction is

Q_shuffle = (L_shuffle-L_local)/(L_global-L_local).                 (F14)

If L_global-L_local<=0, K3 fails and Q_shuffle is undefined; no epsilon value
is reported.

For branch k and true relative-tempo column j, H_k,j is the heldout response
functional P using exact v chi weights, one traversal then block then source
orbit one vote.
The diagonal gain is

G_diag = min_j [min_k!=j H_k,j-H_j,j].                              (F15)

A zero/nonfinite weight denominator fails. Positive G_diag means every
diagonal branch beats every off-diagonal branch in its column.

For attack a, let G_ref=L_capacity-control-L_local and
G_a=L_capacity-control-L_attack,a on the same X0 units. Shortcut retained
gain is

R_a = max(0,G_a)/G_ref.                                             (F16)

If G_ref<=0, K3 fails. No attack-specific denominator is substituted; an
attack worse than capacity control retains exact zero.

Boundary error for a response bank is

B_err = (misses+extras+splits+merges)/
        (truth_events+truth_nonevent_components).                   (F17)

The denominator counts truth events plus maximal truth-negative plateaus; it
must be positive. K5 additionally uses the separately defined worst positive
and negative margins, so a zero B_err cannot hide threshold contact. A
boundary pack is one whose truth plateau intersects the first or last valid
edge position of a window. K5 computes the boundary-error formula separately on exactly those
packs; an empty boundary-pack set fails.

### 11.3 PCG64 paired bootstrap

Every paired interval has exactly 10,000 unit-block draws. Its seed is the
unsigned little-endian first eight SHA-256 bytes of the ASCII statistic name:
temporac.k3.route.v3, temporac.k3.shuffle.v3,
temporac.k3.shortcut.NAME.v3, temporac.k4.diag.COLUMN.v3, or
temporac.k7.route.v3. The atomic unit is one ordered X0 source orbit for K3
and K4 and one ordered development source component for K7. For each draw,
NAME is exactly one of no-pose-timestamp, nuisance-only, pose-shuffle,
static-code-only, warp-metadata, track-length; COLUMN is exactly slow,
medium, or fast.
PCG64 generates M int64 indices with
rng.integers(0,M,size=M,dtype=int64,endpoint=false) in one call. Unit
multiplicities apply identically to every arm and seed. Within a draw,
recompute all nested traversal, block, video, component, seed means and every
strongest-arm minimum; do not clip. M<8 fails before drawing.

Sort the 10,000 binary64 margins ascending. The lower 2.5 percent bound is
order statistic index 249 and the upper 97.5 percent bound index 9749,
zero-based; there is no interpolation. The point statistic uses the original
one-copy unit population. RNG state, NumPy version, byte order, draw
shape, and order statistics are written to the receipt.

## 12. Training jobs, objectives, and hard resource ledger

Teacher and response seeds are exactly
(20260815,20260816,20260817). AdamW uses beta=(0.9,0.999),
epsilon=1e-8, weight decay=1e-4, global gradient-norm clip 1.0,
five-percent linear warmup, and cosine decay to one tenth of the base rate.
Teacher base learning rate is 3e-4 and response/control base rate is 1e-3.
Tune selection uses lowest tune objective, ties by earlier epoch. The heldout
bank cannot select a checkpoint.

The response-job inventory is exactly 24:

- three canonical local jobs, one per seed;
- three capacity-control jobs, one per seed; and
- eighteen shortcut jobs, six attacks per seed.

Global, uniform, blocked, and within-run shuffled are deterministic inference
modes of the frozen canonical checkpoint; static-code control is attack 4. They are not
additional trained jobs. Thus the inventory is three canonical jobs plus
three control jobs plus six attacks at each of three seeds, with the seed
multiplication applying only to the six attack names: 3+3+6x3=24.

Each teacher job is capped at 200 epochs, 20,000 optimizer steps, 6.0
A6000-hours, and 24 GiB. Each response job is capped at 100 epochs, 10,000
steps, 2.5 A6000-hours, and 24 GiB. Heldout response-bank inference is capped
at 1.0 A6000-hour per seed, and natural prediction at 4.0 A6000-hours per
seed. Maximum concurrency is two jobs, one per A6000.

The hard ledger is 18 A6000-hours for three teachers, 60 for 24 response jobs,
3 for heldout response inference, and 12 for natural prediction, totaling 93.
Seven A6000-hours remain unallocated and cannot be reassigned without a new
review. Reaching any job time, step, epoch, memory, or total 100 A6000-hour
cap is failure, not early success. CPU packing and evaluator work cannot hide
GPU usage. Engineering effort is 20-30 focused days.

## 13. Artifact interfaces and firewall

### 13.1 New TempoRAC interfaces

TempoRACFeatureRecordV3 is a package-private immutable record with:

- motion: little-endian float32 [320,17,3];
- person_mask: uint8 [1], exactly one;
- frame_mask: uint8 [320], binary;
- sampled_frame_indices: little-endian int64 [320];
- source_length: Python int and serialized little-endian int64 [1],
  nonembedded;
- opaque_sample_key: uint8 [32] decoded from the 64 hex source bytes;
- local_person_slot: little-endian int64 [1]; and
- feature_shard_sha256: uint8 [32] receipt metadata.

TempoRACPredictionV3 is an immutable NPZ with exact sorted members:
opaque_sample_key uint8 [32], local_person_slot little-endian int64 [1],
abstain uint8 [1], abstain_reason little-endian uint16 [1], count
little-endian int64 [1], response little-endian float32 [E], edge_mask uint8
[E], run_bounds little-endian int32 [R,2], component_bounds little-endian
int32 [C,2], component_score little-endian float32 [C], and contract_sha256
uint8 [32]. An abstention has count=-1 and empty response/component arrays;
a non-abstention has count=C. Reasons are a frozen enum sorted by the order
conditions appear in this contract.

TempoRACReceiptV3 is canonical UTF-8 JSON with sorted keys, no whitespace,
no NaN, LF termination, schema name temporac.execution.v3, artifact byte
count and SHA-256, exact member dtype/shape/hash, config hash, code hash,
source shard hashes, seed, job name, run resets, gate statistics, resource
usage, and upstream receipt hashes. Unknown keys fail.

The whitelist loader has no pickle, vault, evaluator, PoseSequence,
CountResult, local_frequency, or WARP-PHASE model import. It exposes only
TempoRACFeatureRecordV3. The evaluator-boundary process alone defines
TempoRACEvaluationJoinV3, verifies frozen prediction and receipt hashes, then
joins vault rows by exact key bytes and slot. It exposes only aggregated K7
statistics and a signed audit receipt; no vault row returns to training.

These are new TempoRAC-specific interfaces. PoseSequence remains its current
33-keypoint public pose type, CountResult remains its current legacy counting
result, and local_frequency remains its synthetic-frozen diagnostic. None is
adapted, wrapped, inherited, or reinterpreted for TempoRAC.

### 13.2 Physical and temporal firewall

Features, vault, and audit occupy distinct regular non-symlink roots. The
training launch receives only the feature-root capability and its exact
allowlist; the vault root, source pickle, evaluator code, and evaluator
credentials are absent from its command, environment, configuration, import
graph, and input namespace. The later evaluator launch receives frozen
predictions and the vault root but no optimizer or training entry point.
Static reachability tests, forbidden-path injection tests, and an observed
file-open trace establish these one-way process boundaries. Distinct OS
principals may strengthen the boundary when available but are not assumed or
used as the sole proof. A future trusted-packer authorization must explicitly
name the canonical v44 train/development bytes, output roots, and evaluator
launch boundary. Before such authorization, G0 is BLOCKED rather than
pretending that an existing packer proves access.

The executable order after future authorization is: verify source bytes;
create and verify input-root-separated process boundaries; publish feature shards and
count-blind G0 receipts; freeze G0 population; train/select the teacher;
certify targets; freeze conditional coverage; train all response jobs;
freeze configurations, checkpoints, predictions, and receipts; run artifact
gate G5; only then allow the evaluator principal to open the natural vault and
compute K7. Training processes never have a vault path or reader.

## 14. Gates, decisions, and exact timeline

### S0 — Specification freeze

Hash temporac.execution.v3, runtime versions, source path/hash allowlist,
synthetic population, all constants, job inventory, interfaces, and this
proposal. Any hash mismatch fails. No data is opened.

### S1 — Future authorized acquisition

Only after a separate trusted-packer authorization, verify and pack the unique
v44 source. Run G0, then K0. No teacher or response training may begin first.

**G0: unconditional acquisition gate.** Require the exact schema, duplicate
receipts, whole-identity unsupported-edge rule, audited denominators, and all
four split-specific acquisition floors. Report no human label statistic.

**K0: acquisition decision.** PASS only if G0 passes and every required
component mapping is total and count-blind. Zero or missing denominator fails.

### S2 — Teacher and certification

Generate X0; train the three teachers on synthetic train; select each
checkpoint on tune; freeze all teacher artifacts; then evaluate heldout and
natural feature-only certification.

**G1: teacher artifact gate.** Every heldout X0 target-bearing track must pass
certify_target. Orientation-preserving homeomorphisms preserve canonical
phase/pulses; degree, harmonic, reversal, bypass, and collision attacks are
rejected. Reconstruction, origin, delta, degree, collision, tau-invariance,
matched-state phase, and pulse-bit criteria all pass. Empty denominators fail.

**K1: conditional natural certificate coverage.** Relative to the frozen G0
eligible denominators, certified identities and components must each cover at
least 80 percent separately in train and development. The integer threshold
is ceiling(0.8 times the frozen denominator). This cannot alter G0 membership.
Failure kills the route; no weaker teacher or population tolerance is used.

**G2: response-input gate.** Verify channel slices, lag masks, full receptive
fields, edge t+1 alignment, run resets, pre-gap byte isolation, branch
symmetry, exact NOLA denominator proof, and fused-only gradient receipt.

**K2: target validity decision.** Every response-training target individually
has CERTIFIED status and immutable hash. An abstention is excluded only from
the conditional training numerator already declared; it cannot be relabeled
or partially retained.

### S3 — Response training and synthetic decisions

Run exactly the 24 response jobs and deterministic inference modes under the
hard ledger. Freeze checkpoints before any heldout evaluation.

**G3: training-artifact gate.** Verify job inventory, optimizer, source-unit
ordering, accumulation and tail receipts, rejected-batch failure, probability
fusion, exact v-times-chi control weights, tune-only selection, resource
ceilings, and finite outputs. G3 contains no shortcut scientific threshold
and does not use the shortcut retained-gain statistic.

**K3: dominant synthetic routing and shortcut decision.** On piecewise-drift
X0, require point M_X0>0 and its bootstrap lower bound >0; require point and
bootstrap lower bound Q_shuffle>=0.80; and require point and bootstrap upper
bound R_a<0.10 for every one of the six shortcuts. Undefined or zero
denominators fail.

**K4: specialization decision.** Require G_diag>0 and, for every relative
tempo column, the diagonal branch loss to be strictly lower than both
off-diagonal losses for every seed as well as in the seed mean, with every
paired lower bound >0. Every branch receives at least 10 percent of total
p-star responsibility. Seed aggregation is one source-orbit vote followed
by an arithmetic mean over the three ordered seeds. Empty columns fail.

### S4 — Frozen-bank validation and natural evaluation

Run G4 and K5-K6 on deterministic and heldout synthetic artifacts. Freeze
every natural prediction and receipt without vault access. Then run G5. Only
after G5 passes may the evaluator principal open the vault and run K7.

**G4: one two-source operator gate.** Apply the exact source-specific
quantifiers in Section 9 to deterministic fixtures and heldout trained-bank
outputs. Require zero event association errors, positive margins, all
offsets, exact stratum list, and no empty stratum.

**K5: decoder boundary decision.** Require B_err=0 separately for operator
fixtures and each trained seed/resampler/physical stratum, and require worst
positive and negative margins each >=0.10. Zero denominators fail.

**K6: resampler and topology decision.** Require maximum matched-state phase
error <=0.02 cycles, complete pulled-back pulse bit-equality, tau invariance,
and identical event counts on PCHIP and sinc for every heldout source orbit.
Floating-point phase arrays need not be bit-equal across interpolators. One
mismatch fails globally.

**G5: artifact and vault-opening verification only.** Recompute configuration,
code, feature, checkpoint, prediction, and receipt hashes; prove that natural
predictions predate vault opening; verify exact identity join cardinality and
the one-way process/input-root separation; and authorize the evaluator process
to read the vault.
G5 computes no natural metric and makes no scientific pass decision.

**K7: sole natural scientific decision.** First reject the whole pilot on any
evaluator contradiction of P_min, without filtering. On the frozen natural
development piecewise-drift population from Section 10.1, include all 134
supplied-valid identities. An abstention remains an abstention in its receipt
and is scored as continuous estimate zero; it is never removed from a video
or component. Let E_L and E_N be the three-seed video-first AvgMAE values on
that drift population, with people averaged within video, seeds averaged
within video, and videos receiving equal weight. N is the strongest member of
the exact predeclared nonlocal set under E_N, with the Section 10.1 tie order;
the bootstrap recomputes this choice. Require point M_route>0, its 2.5 percent
paired-component bootstrap lower bound >0, and relative improvement
(E_N-E_L)/E_N>=0.05. If E_N=0, this relative condition fails. Apply that same
selected comparator N to the unwarped stationary population of the same 134
identities and recompute its video-first errors. Require E_L-E_N<=0.01
absolute normalized error and, when E_N>0, E_L/E_N<=1.05; when E_N=0 require
E_L=0. Report AvgOBO only as a secondary descriptive metric. K7 is run once;
failure ends the route.

The precedence is strictly

S0 -> S1:G0,K0 -> S2:G1,K1,G2,K2 -> S3:G3,K3,K4 ->
S4:G4,K5,K6 -> freeze natural artifacts -> G5 -> K7.              (F18)

No later gate can retroactively change an earlier population, target,
checkpoint, comparator, threshold, or configuration.

## 15. Paper-visible evidence and reporting boundary

Paper-visible Block A contains only the three-seed natural local versus
strongest-nonlocal paired result under deterministic feature-only drift and
the preregistered stationary retention check. It is the dominant claim.

Paper-visible Block B contains only the 3-by-3 heldout specialization matrix,
G_diag, and its three-seed aggregation. It supports the mechanism.

Paper-visible Block C contains only certify_target coverage, resampler
phase/pulse equality, and zero decoder boundary errors with margins. It
validates the event semantics and does not claim novelty.

Shortcut attacks, gradient receipts, clock audits, duplicate receipts,
resource logs, individual gate failures, and operator inventories remain
audit or appendix material. A failed G or K condition yields a negative
internal feasibility conclusion, not a softened paper claim.

## 16. Forbidden boundary and stopping rule

The proposal authorizes no code, tests, server, data, experiment, result,
title, paper, or Git operation. It authorizes no test, sealed, historical
result, or unapproved natural source access. It introduces no L_cont,
canonical PAMS-TCC initialization, ACF fusion, learned router, GRU, integral
decoder, WARP route, pivot substitution, additional trainable module,
additional arm, seed, claim, or evidence block.

Any impossible definition, missing artifact, empty stratum, zero denominator,
unsupported edge, cap reach, hash mismatch, privilege leak, or evaluator
contradiction fails closed. The route does not start Round 4 from this
document.
