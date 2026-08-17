# Round 1 Refinement: TempoRAC

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

## Anchor Check

- **Original bottleneck:** The unresolved unit is one response event per
  primitive physical repetition. Once that unit is fixed, the actual research
  question is whether a window-local relative-tempo cue selects the right
  shared response scale under asynchronous people and within-track drift while
  keeping identity state private and decoding one reconstructed response once.
- **Why the revised method still addresses it:** The revision preserves all
  seven canonical properties, the three slow/medium/fast response experts,
  cue-only soft routing, positive normalized overlap-add, and exactly one
  connected-component peak pass per complete identity. The teacher is only a
  prerequisite supervision contract and is absent at inference.
- **Reviewer suggestions that would cause drift if followed blindly:** The
  earlier rate/integral decoder would replace the required pulse/peak object;
  promoting the teacher to a second paper contribution would replace the
  routing question; using WARP-PHASE, a hard selected period, missing-pose sets,
  scene-level counting, or per-person model copies would replace the canonical
  method. All are rejected. Integral response mass remains an audit number and
  cannot select, repair, or replace the final peak method.
- **Natural-unit stop rule:** The teacher's pose-derived event unit must pass
  the frozen natural evaluator-only unit check. A harmonic or convention
  mismatch terminates TempoRAC; no human field may repair the teacher or select
  a checkpoint.

## Simplicity Check

- **Dominant contribution after revision:** One claim only: cue-only,
  identity-indexed, window-local relative-tempo routing selects specialized
  shared response scales under within-track drift.
- **Components removed or demoted:** `L_cont` is deleted; PAMS-TCC is omitted
  from the canonical first run; ACF is diagnostic only; the learned router
  residual, hard period, GRU alternative, dynamic peak threshold, period-based
  NMS, expert voting, and integral decoder are absent. The teacher is a passed
  supervision prerequisite. NOLA and one decode are correctness plumbing.
- **Reviewer suggestions rejected as unnecessary complexity:** No LLM, VLM,
  diffusion, RL, uncertainty head, fourth expert, learned fallback, ACF/NUDFT
  fusion, load-balancing loss, continuity loss, or initialization branch is
  added. ACF fusion can be considered only after a separate frozen necessity
  gate and a new method review.
- **Why the remaining mechanism is the smallest adequate route:** A fixed
  phase origin makes teacher crossings reproducible, three matched TCN heads
  represent the requested scales, one nontrainable NUDFT router exposes the
  causal cue, and NOLA plus one thresholded component pass enforces the
  response accounting contract. Removing any one of these four roles breaks an
  immutable property; adding another trainable role does not sharpen the claim.

## Changes Made

### 1. Froze an operational degree-one teacher contract

- **Reviewer said:** Approximate reconstruction, view agreement, orientation,
  and edge alias losses do not force a complete circle or degree (+1), while
  minimum-positive-variation selection actively rewards degree-zero arc
  compression.
- **Action:** Removed positive-variation selection. Defined the delay tensor,
  teacher/code/decoder layers, deterministic pose-derived phase origin,
  complete-orbit closing edge, signed winding, circle coverage, seam,
  collision, reconstruction, origin stability, and explicit degree (0),
  (-1), (+2), harmonic, symmetric, static-code, and decoder-bypass attacks.
  Checkpoints are chosen lexicographically by topology pass, lowest held-out
  reconstruction, earliest epoch, and smallest seed.
- **Reasoning:** The ideal left-inverse argument is useful only as a falsifiable
  supervision contract. The learner never sees the synthetic primitive
  coordinate; the evaluator may use it only to reject a teacher.
- **Impact on core method:** The pulse unit becomes auditable without becoming
  a second paper claim. Failure rejects TempoRAC rather than causing a
  label-derived repair.

### 2. Closed the pulse and final-peak interface without changing decoders

- **Reviewer said:** Sparse ordinary BCE does not ensure an event remains above
  0.5, a valley remains below 0.5, or adjacent events remain separate after
  mixing and NOLA.
- **Action:** Replaced the underspecified sparse loss with class-balanced BCE
  plus frozen event and valley margins, kept sigmoid expert responses, fixed
  threshold 0.5, specified invalid-gap and plateau-center semantics, and added
  held-out spacing, split, merge, pause, padding, duplicate-clock, seam, and
  half-hop grid-shift gates.
- **Reasoning:** Calibration must be demonstrated at the fixed decoder, not
  inferred from binary targets or rescued by a tuned threshold.
- **Impact on core method:** Exactly one connected component per teacher event
  is now a hard prerequisite. If it fails, the method is rejected; the
  diagnostic integral never becomes the count.

### 3. Replaced every method fork with one executable interface

- **Reviewer said:** Tensor shapes, masks, clocks, short tracks, delay lags,
  expert class, spectral bins, normalization, reliable-window rules, loss
  indices, constants, and call order were not implementation-complete.
- **Action:** Fixed COCO-17 normalization, duplicate-clock collapse, feature and
  vault schemas, the 215-channel teacher state, the 269-channel causal encoder
  input, exact MLP sizes, one local encoder, three two-block causal
  depthwise-separable TCN experts, a 48-bin irregular-clock NUDFT, window and
  fallback behavior, all loss reductions and weights, optimizer schedules, and
  an execution-contract hash.
- **Reasoning:** There is now no TCN/GRU, ACF/NUDFT-fusion, FFT branch, missing
  index set, or normalization choice left to an implementation author.
- **Impact on core method:** An engineer can implement one graph and produce
  comparable call traces without a scientific choice during coding.

### 4. Bound the physical feature/vault and nuisance firewall

- **Reviewer said:** The current v44 pickle colocates privileged fields with
  model inputs, and the nuisance-only runtime schema was open.
- **Action:** Defined trusted packing into feature-only shards, a separately
  permissioned evaluator vault, and an audit-only source manifest. The training
  loader rejects forbidden keys before payload deserialization. The canonical
  model sees no identity, source length, count, period, boundary, density, box,
  provenance, absolute clock, or warp metadata. The shortcut harness receives
  one explicit nuisance tensor, never the canonical graph.
- **Reasoning:** Ignoring a colocated key is not physical isolation. The audit
  also establishes that the third pose channel is confidence, that duplicate
  source indices must collapse, and that source-component grouping is
  audit-only.
- **Impact on core method:** Data-bearing work remains blocked until exact
  receipts pass; the counting objective cannot acquire an accidental human
  supervision or identity path.

### 5. Made the routing test a same-response intervention

- **Reviewer said:** Retraining each router arm confounds optimization with the
  routing cue, and the one-head control and tempo strata were underspecified.
- **Action:** Aligned synthetic stratum responsibilities to the same
  (pmlog 1.5) anchors, normalized overlap responsibility so each unique edge
  has total weight one, trained the expert bank once, cached its full-track
  responses, and defined local, global, uniform, cue-blocked, and
  within-track-shuffled gates on that same tensor. A one-output control matches
  the total parameter, measured FLOP, and receptive-field envelope of the
  three heads. Held-out unseen resamplers produce an expert-by-tempo matrix.
- **Reasoning:** Only the gate changes in the decisive comparison, so a local
  advantage is attributable to selecting already-specialized responses.
- **Impact on core method:** Expert specialization and window-local selection
  have separate, executable falsifiers without another module.

### 6. Collapsed the paper to one claim and one first route

- **Reviewer said:** The teacher, routing, and NOLA/peak ordering read as three
  contributions in a four-page paper; `L_cont` and optional PAMS initialization
  also expanded the implementation.
- **Action:** Declared the routing mechanism the only paper-level claim.
  Deleted `L_cont`, omitted PAMS-TCC from the canonical first run, and treated
  the teacher, positive NOLA, and one decoder as prerequisites or correctness
  plumbing. The current PAMS Transformer, inferred SSHead, hard period
  selection, and deterministic peak-consensus experts are not reused.
- **Reasoning:** The existing `src/pams` path is a different single-person
  architecture and does not implement any of the seven TempoRAC properties.
  Reusing its ambiguous head would weaken rather than simplify this test.
- **Impact on core method:** The proposal now supports one concise thesis and
  one causal validation package.

### 7. Replaced broad readiness language with staged hard stops

- **Reviewer said:** Feasibility, title, novelty, server, and paper readiness
  were overstated relative to absent contracts, absent implementation, an open
  TWCRAC comparison, and zero eligible results.
- **Action:** Defined executable G0--G5 build gates and K0--K7 scientific kill
  gates with artifacts, observables, precedence, and failure actions. The work
  estimate is 15--23 engineer-days and at most 100 A6000-hours in hard-stop
  stages.
- **Reasoning:** A bounded plan is not authorization. Each later stage becomes
  reachable only after every earlier receipt is immutable and passing.
- **Impact on core method:** Title/acronym, novelty, server, paper, and
  submission authorization remain unresolved. Full TWCRAC overlap remains
  unresolved, WARP-PHASE remains terminally failed, and no backup pivot is
  substituted.

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

## Technical Gap

MultiCounter and MultiCounter+ occupy multi-person repetition counting and
person-indexed output. PAMS occupies pose-driven periodic supervision and
period-adaptive self-consistency. RepNet and HTRM-Net occupy time-varying and
local temporal periodicity. Generic mixtures occupy shared expert routing, and
WOLA/NOLA occupies overlap reconstruction. TWCRAC remains the closest
unresolved overlap risk because its full method has not been inspected. No
individual named component is available as novelty.

The operational gap is the interaction that a track-global counter cannot
express: two identities can move at different rates, and one identity can
change rate within a track. A per-person model copy does not test shared
specialization. Independent window counts double-count seams. A learned router
can ignore its claimed tempo cue. A hard period selector can lock to a
harmonic, as the terminal WARP-PHASE route demonstrated. The smallest clean
test is therefore a nontrainable relative-tempo gate over shared responses,
with state isolated by identity and responses reconstructed before one count.

That test still needs a response with the physical unit of one event per
primitive repetition. Warp consistency, generic reconstruction, and TCC do not
fix integer phase winding. The degree-one teacher below is therefore a
prospectively frozen supervision contract. It is not an inference module or a second
contribution. Likewise, NOLA and the final component pass are fixed accounting
operators. The dominant scientific question begins only after those contracts
pass.

## Method Thesis and Contribution Focus

- **One-sentence thesis:** Cue-only, identity-indexed, window-local
  relative-tempo routing selects specialized shared response scales under
  within-track drift.
- **Dominant contribution:** The causal interaction between a full-distribution
  relative local-tempo cue and three matched, shared, tempo-specialized response
  experts, demonstrated on the same frozen response tensor.
- **Prerequisite supervision contract:** A frozen clock-blind degree-one phase
  teacher emits canonicalized half-open event crossings and is discarded after
  target creation.
- **Fixed correctness plumbing:** Identity-private causal buffers, positive
  NOLA, and one threshold-0.5 connected-component pass enforce the seven
  canonical properties but are not parallel contributions.
- **Permitted supervision statement:** “the counting objective reads no human per-person count or period annotations”. External supplied tracks, detector
  and pose-estimator provenance, and synthetic warp correspondences are
  disclosed separately.
- **Explicit non-claims:** No component novelty, broad priority claim, official
  MultiRep result, predicted-track result, or submission claim is available.
  The working title and acronym remain provisional.

## Complexity Budget

- **Trainable in the prerequisite stage:** one 32-dimensional static-code MLP,
  one memoryless phase MLP, and one phase-conditioned reconstruction MLP.
- **Trainable in canonical TempoRAC:** one shared width-64 local encoder and
  exactly three equal-parameter width-64 response TCNs.
- **Frozen operators:** feature normalization, masks, clocks, windows, NUDFT,
  weighted-median reference, soft gate, taper, NOLA, and decoder.
- **Deleted from the first route:** `L_cont`, PAMS-TCC initialization, inferred
  PAMS SSHead, ACF fusion, learned gate residual, hard period, GRU, fourth
  expert, load loss, learned threshold, period-dependent suppression, expert
  voting, and integral decoding.
- **Later control only:** PAMS-TCC may be examined as an initialization control
  after the canonical route passes; it cannot alter the first expert bank or
  rescue a failed gate.

## System Graph and Seven Invariants

```text
trusted feature-only supplied-track shard
  -> duplicate source-clock collapse and COCO-17 normalization
  -> clock-blind teacher (training target creation only)
       -> deterministic pose-derived phase origin
       -> half-open event pulses e*
  -> shared full-track local encoder, once per identity
       -> slow TCN response, cached once
       -> medium TCN response, cached once
       -> fast TCN response, cached once
  -> independent window-local 48-bin NUDFT distributions
  -> weighted-median track reference -> cue-only soft gates
  -> gate-weighted cached responses inside each window
  -> positive normalized overlap-add over unique edges
  -> one threshold-0.5 connected-component pass per identity
  -> person-indexed count vector
```

The implementation is TempoRAC only if all seven invariants hold:

1. The same encoder parameter object is used for every identity.
2. The same three slow/medium/fast expert parameter objects are used for every
   identity, and their trainable parameter counts are equal.
3. Every causal layer buffer and reconstruction buffer is private to one
   identity; expert buffers are additionally private to one expert.
4. Each raw window NUDFT tuple is computed only from that window's pose, masks,
   and relative source clock.
5. Gates lie on the simplex and fuse continuous expert responses before any
   threshold or component operation.
6. Every valid unique edge has a positive finite NOLA denominator, and padding
   or invalid edges contribute zero.
7. Decoder call count equals the number of supplied `person_valid` identities,
   including an empty-response call for a short or fully invalid identity, with
   no independent window count or window-count sum.

## Exact Input, Mask, Clock, and Firewall Contract

### Trusted artifacts

The trusted packer is the only process allowed to deserialize the current
label-mixed v44 source. It emits three physically separate artifact classes:

| Artifact | Exact contents | Consumer |
|---|---|---|
| Feature shard | `pose_xyc: float32[P,U,17,3]`, `joint_valid: bool[P,U,17]`, `frame_valid: bool[P,U]`, `person_valid: bool[P]`, strictly increasing `source_index: int64[U]`, opaque join token, and local slot metadata | training/model process |
| Evaluator vault | joined `count_gt`, density, raw half-open period intervals, boxes, object mappings, and evaluator handles | separately permissioned evaluator only |
| Audit manifest | source length, canonical-source component, source IDs, provenance hashes, converter version, shard/vault hashes, and join receipts | split/firewall audit only |

The model loader accepts only the feature-shard schema and rejects an unknown
or forbidden key before reading any array payload. Names, paths, hashes,
canonical IDs, source split, object IDs, boxes, counts, periods, boundaries,
density, metrics, and source length are absent from model tensors. Opaque keys
and local slots select records and state dictionaries only; they are never
embedded.

The K3 nuisance-only attack tensor is frozen as float32
\(N_i\in\mathbb R^{U\times37}\): 17 binary joint-valid channels, 17 confidence
channels clipped to \([0,1]\), one binary frame-valid channel, one centered
source-gap channel, and one interpolation-weight channel. For edge \(t>0\),
the gap channel is
\(\operatorname{clip}((\log(1+\Delta q_t)-\operatorname{mean})/
(\operatorname{std}+10^{-8}),-4,4)\) over valid edges; row zero is zero.
The interpolation channel is the left linear-interpolation coefficient in
\([0,1]\) for a synthetic view and zero for an unwarped natural shard. The
canonical model never receives this 37-vector as a unit. The attack adapter is
a frozen \(37\times269\) Rademacher matrix scaled by \(1/\sqrt{37}\), generated
counterwise from the ASCII seed “temporac.nuisance-adapter.v1” using the same
SHA-256 bit rule as the pose landmark. It introduces no trainable parameter.
No crop, filename, ID, source length, warp derivative, augmentation seed, or
human field belongs to this schema.

### Duplicate collapse and canonical pose

For an identity \(i\), let raw retained rows have COCO-17 pose
\(X^{raw}_{itjc}=(x,y,confidence)\), frame validity \(F^{raw}_{it}\), and
nondecreasing source indices \(q^{raw}_{it}\). Consecutive rows sharing one
source index are collapsed before any feature is constructed. For joint \(j\),
its collapsed \(x,y\) are the confidence-weighted mean over valid duplicates,
its confidence is their maximum clipped to \([0,1]\), and its joint mask is the
logical OR. A zero total confidence produces value zero and mask false. The
collapsed index sequence \(q_{it}\) must be strictly increasing. Duplicate-run
size is audit-only and is not persisted in the feature shard.

A joint is valid exactly when the source frame is valid, its confidence is
finite and greater than zero, and its coordinates are finite. The frame root
is the mean of whichever of left/right hip joints 11 and 12 are valid; a frame
with neither hip is invalid. A frame is eligible when the root exists and at
least eight joints are valid. Scale candidates are every positive finite
shoulder width \(5\leftrightarrow6\), hip width \(11\leftrightarrow12\), and
shoulder-midpoint to hip-midpoint distance available on eligible frames. The
track scale is their median; a track with no candidate above \(10^{-6}\) is
ineligible. Valid coordinates are root-relative, divided by that fixed track
scale, clipped to \([-4,4]\), and invalid values are exact zero.

The 16 ordered bones are

```text
(0,1) (0,2) (1,3) (2,4) (5,6) (5,7) (7,9) (6,8)
(8,10) (5,11) (6,12) (11,12) (11,13) (13,15) (12,14) (14,16)
```

A bone vector is child minus parent and is valid only when both joints are
valid. An edge \(e=(t,t+1)\) is valid when both frames are eligible and
\(\Delta q_{ie}=q_{i,t+1}-q_{it}>0\). The encoder and teacher never receive
\(q\), \(\Delta q\), an absolute position, source length, identity, window
index, count, period, boundary, density, or warp metadata. Only the frozen cue
operator receives the window-relative clock
\(\tau_{it}=q_{it}-q_{i,s_\ell}\). Synthetic correspondence and derivative
records are visible only to the loss builder.

### Short identities and window layout

Let \(T_i\) be the number of unique collapsed samples and \(E_i=T_i-1\) the
number of unique edges. \(T_i=0\) is an invalid record. \(T_i=1\), no eligible
frame, or no valid edge creates an empty all-invalid response, calls
\(D_{peak}\) exactly once, returns zero, and emits an abstention receipt. It
cannot enter a training loss but remains in any later evaluator denominator.
For
\(T_i\ge2\), windows have 128 sample slots and 127 edge slots. Starts are
\(0,32,64,\ldots\) while a full window fits, followed by
\(\max(0,T_i-128)\) if that start is not already present. If \(T_i<128\), one
start at zero is right-padded. Padding values and masks are exact zero and do
not enter a cue, loss, NOLA sum, or decoder. Every real edge is covered by at
least one window.

## Primitive-Orbit Teacher as a Prerequisite Contract

### Clock-blind delay state

At unique sample \(t\), let \(p_t\in\mathbb R^{34}\) be normalized joint
coordinates, \(b_t\in\mathbb R^{32}\) the bone coordinates, and
\(c_t\in[0,1]^{17}\) confidence. Set \(s_t=[p_t,b_t]\in\mathbb R^{66}\).
The teacher uses no raw velocity magnitude or fixed-width lag chord. Its only
motion channel is the warp-invariant local tangent direction. On coordinates
valid at \(t-1,t,t+1\), form
\[
v^-_t=s_t-s_{t-1},\qquad v^+_t=s_{t+1}-s_t,\qquad
\widehat v_t=
\frac{v^-_t/(\|v^-_t\|_2+10^{-8})+
v^+_t/(\|v^+_t\|_2+10^{-8})}
{\|v^-_t/(\|v^-_t\|_2+10^{-8})+
v^+_t/(\|v^+_t\|_2+10^{-8})\|_2+10^{-8}}.
\tag{F1}
\]
The norm is over the currently valid geometry coordinates; invalid coordinates
are zero. Endpoints and a zero-norm or missing two-sided tangent are invalid.
The teacher state is
\[
y_t=[p_t,b_t,\widehat v_t,c_t,m^s_t,m^v_t]\in\mathbb R^{215},
\tag{F2}
\]
where \(m^s_t\in\{0,1\}^{33}\) holds the 17 joint and 16 bone masks and
\(m^v_t\in\{0,1\}^{33}\) holds the corresponding tangent masks. Thus the
continuous part has \(34+32+66+17=149\) channels and the mask part has 66.
The only ordinal offsets used are the immediate \((-1,+1)\) neighbors needed
to estimate direction; their distance and displacement magnitude are
discarded. No clock, time value, length, identity, window, or warp variable is
present.

### Fixed architectures and losses

The static code uses the mask-weighted temporal mean and standard deviation of
the 149 continuous channels in \(y_t\), concatenated as a 298-vector. Its MLP
is `Linear(298,128)-GELU-Linear(128,32)`, followed by unit \(L_2\)
normalization. This single \(c_i\in\mathbb R^{32}\) is repeated at all times
and contains no length scalar. The phase MLP is
`Linear(247,256)-GELU-Linear(256,128)-GELU-Linear(128,2)`, where the input is
\([y_t,c_i]\); its two outputs are normalized to \(z_t\in S^1\) with epsilon
\(10^{-8}\). The decoder is
`Linear(34,256)-GELU-Linear(256,256)-GELU-Linear(256,149)`, with input
\([z_t,c_i]\). It has no skip, residual, time, mask, or raw-state input.

For exact monotone-view correspondences \(\mathcal C_i\), define the signed
circular adjacent increment in cycles

\[
\delta_{it}=\frac{1}{2\pi}\operatorname{atan2}
\left(z^{(1)}_{it}z^{(2)}_{i,t+1}-z^{(2)}_{it}z^{(1)}_{i,t+1},
\langle z_{it},z_{i,t+1}\rangle\right)
\in(-0.5,0.5].
\tag{F3}
\]

The teacher loss is

\[
\mathcal L_{teach}=\mathcal L_{inv}+\mathcal L_{view}
+0.25\mathcal L_{orient}+0.25\mathcal L_{alias},
\tag{F4}
\]

where \(\mathcal L_{inv}\) is mask-normalized Huber reconstruction of the 149
continuous state channels with Huber transition 0.05,
\(\mathcal L_{view}=|\mathcal C|^{-1}\sum_{(t,t')\in\mathcal C}
(1-\langle z_t,z'_{t'}\rangle)\),
\(\mathcal L_{orient}\) is the valid-edge mean of
\(\operatorname{ReLU}(-\delta)\), and \(\mathcal L_{alias}\) is the valid-edge
mean of \(\operatorname{ReLU}(|\delta|-0.25)\). Every denominator must be
positive; a batch lacking support is rejected rather than silently skipped.

### Pose-derived canonical phase origin

Phase rotation is fixed by an exact pose landmark, not by a sample-weighted
mean. Let \(g_t=[p_t,b_t]\in\mathbb R^{66}\); confidence, masks, tangent,
clock, and lag patterns are excluded from the landmark score. A fixed
Rademacher vector \(a\in\{-1,+1\}^{66}/\sqrt{66}\) is generated by
concatenating bits of
`SHA256("temporac.pose-landmark.v1" || uint32_be(counter))`, taking
the first 66, and mapping bit 0 to \(-1\) and bit 1 to \(+1\). It is a
persistent nontrainable buffer. Define landmark score
\(\ell_t=a^\top g_t\). A score is valid only when every one of the 66 projected
geometry coordinates is supported at \(t-1,t,t+1\). A candidate \(t\) is a
strict valid local maximum with
\(\ell_{t-1}<\ell_t\ge\ell_{t+1}\). Its deterministic parabolic sub-sample
offset is
\[
\xi_t=\operatorname{clip}\left(
\frac{\ell_{t-1}-\ell_{t+1}}
{2(\ell_{t-1}-2\ell_t+\ell_{t+1})},-0.5,0.5\right).
\tag{F5}
\]
The interpolated landmark geometry \(g^L_t\) and phase \(z^L_t\) use linear
interpolation between \(t\) and the neighbor in the sign of \(\xi_t\);
\(z^L_t\) is renormalized to \(S^1\). A candidate is retained only when both
adjacent local minima exist in one contiguous valid run, its two-sided
prominence is at least 0.20 of that run's nonzero score range, and its score
lies within 0.05 run ranges of the maximum valid-local-maximum score.

On synthetic fixtures, the evaluator-hidden primitive traversal must contain
exactly one retained landmark. On natural orbits, successive retained
landmarks define candidate complete traversals without consulting a boundary
field. Let \(\mathcal M_i\) be those landmarks. Each complete traversal has
weight one, irrespective of samples, pause duration, or warp density. The
traversal-normalized circular origin and canonical phase are
\[
m_i=|\mathcal M_i|^{-1}\sum_{t\in\mathcal M_i}z^L_t,\quad
\rho_i=|m_i|,\quad o_i=m_i/(|m_i|+10^{-8}),\quad
\widetilde z_{it}=z_{it}\overline{o_i}.
\tag{F6}
\]
The origin passes only with at least two retained landmarks, phase resultant
\(\rho_i\ge0.95\), maximum landmark-geometry distance from their coordinatewise
median at most 0.05, and identical landmark count and ordering under exact
warp correspondence. Cross-linear, PCHIP, and windowed-sinc origin drift must
be at most 0.02 cycles. The orientation-preserving homeomorphism attacks
\(h_c(\phi)=\phi+c\sin(2\pi\phi)/(2\pi)\bmod1\) for
\(c\in\{-0.75,-0.5,+0.5,+0.75\}\) must leave canonical seam crossings
unchanged. This seam fixes the algorithmic pose origin but does not assert the
human repetition convention.

A clip abstains when it has no complete valid traversal, a symmetric or
self-intersecting full delay-state orbit, missing landmark support, only
partial or gappy support, multiple retained maxima in a synthetic primitive
traversal, reverse motion, \(\rho_i<0.95\), incoherent landmark geometry, or an
unstable cross-resampler origin. Unacceptable abstention coverage or any later
natural teacher-unit mismatch kills the route. Canonical phase, not raw phase,
defines the half-open event seam.

Initialize
\(A_{i0}=\operatorname{mod}_1(\arg(\widetilde z_{i0})/(2\pi))\in[0,1)\) and set
\(A_{i,t+1}=A_{it}+\delta_{it}\). The half-open edge event is

\[
e^*_{it}=\mathbf 1[\,A_{i,t+1}>A_{it}\ \land\
\lfloor A_{i,t+1}\rfloor>\lfloor A_{it}\rfloor\,].
\tag{F7}
\]

The alias gate ensures at most one crossing on a valid edge. Pauses and
nonpositive increments emit zero. Targets are detached and frozen with the
teacher artifact hash.

### Complete-orbit topology evaluator and checkpoint precedence

Synthetic asymmetric primitive orbits expose an ordered evaluator coordinate
only to the evaluator. For \(N\) ordered samples, the signed winding includes
the closing edge \(N-1\rightarrow0\):

\[
W=\operatorname{round}\left(\sum_{t=0}^{N-1}
\operatorname{wrap}_{(-0.5,0.5]}(\phi_{t+1\bmod N}-\phi_t)\right),
\tag{F8}
\]

where \(\phi_t=\arg z_t/(2\pi)\). A held-out asymmetric orbit passes only when
all of the following hold: \(W=+1\); at least 30 of 32 equal phase bins are
occupied; maximum sorted circular phase gap is at most 0.10 cycles; negative
sequential edges are at most 1%; every absolute increment is below 0.25;
closing-edge absolute increment is below 0.25; latent collision rate is at
most 1% for evaluator-coordinate-separated pairs; normalized held-out Huber
reconstruction is at most 0.02; exactly one retained pose landmark occurs per
evaluator traversal; landmark-phase resultant is at least 0.95; landmark
geometry deviation is at most 0.05; and warp/resampling origin drift is at
most 0.02 cycles. At least 95% of held-out asymmetric orbits across both unseen
resamplers must pass, and every Formula F6 homeomorphism attack must preserve
the canonical crossing sequence.

The evaluator must reject every injected degree \(0\), \(-1\), and \(+2\)
solution. A degree-two harmonic generator, half-cycle seam generator, and
rotationally symmetric orbit must abstain or fail, never pass as degree one.
With static code held fixed, replacing phase by a constant, a temporal
permutation, or independent noise must raise reconstruction above 0.02.
Shuffling static codes across orbits must also fail reconstruction. These are
the static-code and decoder-bypass attacks; the decoder has no alternate
input.

Teacher checkpoints are ordered strictly as: complete topology pass first,
then lowest held-out reconstruction loss, then earliest epoch, then smallest
seed. Positive variation is not an ordering key. On the later frozen natural
development unit audit, at least 80% of otherwise eligible identities and 80%
of source components must retain complete coherent landmark traversals. Lower
coverage, a systematic half/double/off-grid teacher category above 5%, or a
median teacher-event-to-vault-count ratio outside \([0.95,1.05]\) is a natural
teacher-only mismatch and terminates the route.
The audit cannot filter tracks, alter targets, or choose another checkpoint.

## Canonical TempoRAC Architecture

### Causal encoder input and shared full-track encoder

The encoder uses the same base \(p_t,b_t,c_t\), only backward displacement
lags \(\{-4,-2,-1\}\), and their masks. Its exact input dimension is
\(83+3\times34+17+16+3\times17=269\). The shared encoder is
`Linear(269,64)` followed by two residual causal depthwise-separable blocks of
width 64, kernel 5, dilations 1 and 2. Each block is
`LayerNorm-DepthwiseCausalConv1d-GELU-PointwiseConv1d-GELU-residual`; convolution
biases are enabled and there is no dropout or batch normalization. It runs
once over each complete identity and emits \(H_i\in\mathbb R^{T_i\times64}\).
Invalid samples are zero before and after every block.

### Three fixed response experts

Let \(\mathcal K=(slow,medium,fast)\) with dilations \(d_k=(4,2,1)\). Each
expert is exactly two residual causal depthwise-separable blocks of width 64
and kernel 5, both using its assigned dilation, followed by
`LayerNorm-Linear(64,1)-Sigmoid`. Their trainable parameter counts are equal
because dilation changes no parameter shape. The cached unique-edge response
is

\[
r^{(k)}_{ie}=\sigma(H_k(H_i)_e)\in(0,1).
\tag{F9}
\]

Parameters are shared across identities. Each block has a private causal left
buffer keyed by `(opaque_join_token, local_person_slot, expert, block)`; the
two layer-specific buffers each hold \(4d_k\) preceding states. Encoder buffers
are private to the identity invocation. Experts traverse a full identity once,
and windows only slice cached responses. No overlap window advances a causal
state twice. No cross-person attention, normalization statistic, hidden state,
router history, NOLA accumulator, or decoder state exists.

## One Primary Irregular-Clock NUDFT Cue

### Frozen frequency grid and window signal

The cue uses only source-clock velocity of the 66 root-relative pose and bone
coordinate channels inside one window. For edge \(e=(t,t+1)\), coordinate
\(d\), define
\(v_{ed}=(s_{t+1,d}-s_{t,d})/\Delta q_e\) when both endpoint coordinate masks
are valid, and zero otherwise. This rate magnitude is deliberately confined to
the tempo cue; it never enters the teacher or shared encoder. The cue never
reads \(H_i\). The 48 source-frame periods and frequencies are

\[
P_b=\exp\left(\log4+\frac{b}{47}\log\frac{128}{4}\right),\quad
f_b=P_b^{-1},\quad b=0,\ldots,47.
\tag{F10}
\]

For velocity-edge midpoint clocks
\(\tau_e=(q_t+q_{t+1})/2-q_{s_\ell}\), trapezoidal quadrature weights are half
the next midpoint gap at the first edge, half the previous gap at the last,
and half the two-sided gap internally. The source-clock Hann weight is
\(h_e=\sin^2(\pi(\tau_e-\tau_{first})/
(\tau_{last}-\tau_{first}))\). A midpoint span below four source frames has no
support. For each coordinate \(d\), weighted least squares with mask times
quadrature times Hann weight removes an intercept and linear trend in \(\tau\).
A coordinate is eligible only with at least eight positive-weight edges and
residual energy above \(10^{-8}\).

For detrended velocity residual \(x_{ed}\), let \(w_{ed}\) be its full
nonnegative weight. The normalized quadrature power is

\[
S_{bd}=\frac{(\sum_ew_{ed}x_{ed}\cos(2\pi f_b\tau_e))^2+
(\sum_ew_{ed}x_{ed}\sin(2\pi f_b\tau_e))^2}
{(\sum_ew_{ed})(\sum_ew_{ed}x_{ed}^2)+10^{-8}},
\tag{F11}
\]

and \(s_b\) is the arithmetic mean of \(S_{bd}\) over eligible coordinates.
With at least one eligible coordinate,

\[
p_{i\ell b}=\frac{s_b+10^{-8}}{\sum_{a=0}^{47}(s_a+10^{-8})},\quad
u_{i\ell}=\sum_{b=0}^{47}p_{i\ell b}\log f_b.
\tag{F12}
\]

Otherwise \(p\) is exactly uniform, \(u\) is the uniform-grid log-frequency
mean, and confidence is zero. No peak, `argmax`, selected period, learned
residual, or fallback frequency exists.

Let \(n_{i\ell}\) be the number of real velocity edges with at least eight
valid joint pairs, \(Q_{i\ell}=\tau_{last}-\tau_{first}\), and
\(D_{i\ell}\) the number of
eligible coordinate channels. Coverage, normalized entropy, and confidence
are

\[
C_{i\ell}=\min(1,n_{i\ell}/32)\min(1,Q_{i\ell}/128)
\cdot(D_{i\ell}/66),
\quad
H_{i\ell}=-\frac{\sum_bp_{i\ell b}\log(p_{i\ell b}+10^{-8})}{\log48},
\quad
\gamma_{i\ell}=C_{i\ell}\max(0,1-H_{i\ell}).
\tag{F13}
\]

A reliable window has \(\gamma\ge0.15\), \(n\ge16\), \(Q\ge16\), and
\(D\ge16\). The stop-gradient track reference is the deterministic
confidence-weighted median of reliable \(u\) values: sort by `(u, window
start)` and take the first value whose cumulative weight reaches half the
total. At least three reliable windows are required. Otherwise the reference
is invalid.

### Relative cue-only routing and fallback

With a valid reference \(\bar u_i\), set
\(z_{i\ell}=u_{i\ell}-\bar u_i\). For
\(\mu=(-\log1.5,0,+\log1.5)\) ordered slow/medium/fast and
\(\sigma_g=\log1.5/2\), define

\[
\widetilde g_{i\ell k}=\operatorname{softmax}_k
\left(-\frac{(z_{i\ell}-\mu_k)^2}{2\sigma_g^2}\right),\qquad
g_{i\ell}=\gamma_{i\ell}\widetilde g_{i\ell}+
(1-\gamma_{i\ell})(1/3,1/3,1/3).
\tag{F14}
\]

If the track reference is invalid, every gate is exactly
\((1/3,1/3,1/3)\), regardless of window confidence. The raw tuple
\((p,u,\gamma)\) for an unchanged window is invariant to every perturbation
outside that window. Its final relative gate may change only because the
documented weighted-median track reference changes. Tests record both the raw
tuple and final gate so this permitted path is not mistaken for leakage.

A mask-normalized non-circular ACF may be logged on the same 48-bin grid, but
it cannot enter \(p,u,\gamma,\bar u,g\), training, checkpoint choice, or count.
Fusion requires a later prospectively frozen synthetic necessity gate showing that
NUDFT alone fails under irregular missingness and that the fixed fusion repairs
that failure without a shortcut, followed by a new method review.

## Expert Targets, Unique-Edge Responsibilities, and Losses

### Stratum alignment

Synthetic training warps expose their local log-frequency shift only to the
loss builder. For window \(\ell\), take the warp generator's mean log-frequency
shift over real samples in that window, then subtract its deterministic
weighted median across windows satisfying the Formula F13 reliability
predicate, using their \(\gamma_{i\ell}\) values as weights. Fewer than three
reliable windows invalidates the synthetic example. The result is
\(\zeta^*_{i\ell}\), and

\[
p^*_{i\ell k}=\operatorname{softmax}_k
\left(-\frac{(\zeta^*_{i\ell}-\mu_k)^2}{2\sigma_g^2}\right).
\tag{F15}
\]

Each synthetic minibatch has equal numbers of source windows centered within
0.10 log-frequency units of the three anchors. Training uses linear monotone
interpolation; held-out specialization uses PCHIP and windowed-sinc resamplers,
which never occur in training.

For a real unique edge \(e\) covered by windows \(\mathcal L_i(e)\), let the
positive taper be \(a_{i\ell e}\) from Formula F19 below. Its normalized
overlap responsibility and edge-level expert responsibility are

\[
\omega_{i\ell e}=\frac{a_{i\ell e}}
{\sum_{j\in\mathcal L_i(e)}a_{ije}},\qquad
\pi_{iek}=\sum_{\ell\in\mathcal L_i(e)}\omega_{i\ell e}p^*_{i\ell k}.
\tag{F16}
\]

Thus \(\sum_\ell\omega_{i\ell e}=1\) and \(\sum_k\pi_{iek}=1\); no edge is
overcounted because it appears in several windows.

### Frozen pulse loss

For responses \(x_e\in(0,1)\), binary targets \(y_e=e^*_e\), and
nonnegative weights \(w_e\), define positive and negative normalizers
\(Z_+=\sum_ew_ey_e\) and \(Z_-=\sum_ew_e(1-y_e)\). A training batch must have
\(Z_+>0\) and \(Z_->0\). With probability clamp
\(\epsilon_b=10^{-7}\), the class-balanced BCE, event margin, and valley
margin are

\[
\begin{aligned}
\mathcal B(x,y;w)&=-\frac12\frac{\sum_ew_ey_e\log(\max(x_e,\epsilon_b))}{Z_+}
-\frac12\frac{\sum_ew_e(1-y_e)\log(\max(1-x_e,\epsilon_b))}{Z_-},\\
\mathcal M_+(x,y;w)&=\frac{\sum_ew_ey_e[\max(0,0.75-x_e)]^2}{Z_+},\\
\mathcal M_-(x,y;w)&=\frac{\sum_ew_e(1-y_e)[\max(0,x_e-0.25)]^2}{Z_-},\\
\mathcal P(x,y;w)&=\mathcal B+0.5\mathcal M_++0.5\mathcal M_-.
\end{aligned}
\tag{F17}
\]

The specialization and fused losses are

\[
\mathcal L_{spec}=\frac13\sum_{k\in\mathcal K}
\mathcal P(r^{(k)},e^*;v\pi_k),\qquad
\mathcal L_{fuse}=\mathcal P(R,e^*;v),\qquad
\mathcal L_{resp}=\mathcal L_{spec}+\mathcal L_{fuse},
\tag{F18}
\]

where \(v\) is the valid unique-edge mask and \(R\) is the canonical NOLA
response. The router and teacher are stop-gradient. There is no continuity,
load, entropy, period, density, count, or auxiliary reconstruction term in
this stage. The expert bank is trained once.

## Response Fusion, Positive NOLA, and One Peak Pass

For an edge at zero-based position \(n=e-s_\ell\in[0,126]\) inside a
128-sample window, define

\[
a_{i\ell e}=10^{-3}+(1-10^{-3})
\sin^2\left(\frac{\pi(n+1/2)}{127}\right).
\tag{F19}
\]

The window response is fused before any discrete operation, and unique-edge
NOLA is

\[
r_{i\ell e}=\sum_{k\in\mathcal K}g_{i\ell k}r^{(k)}_{ie},\qquad
R_{ie}=\frac{\sum_{\ell\in\mathcal L_i(e)}a_{i\ell e}v_{ie}r_{i\ell e}}
{\sum_{\ell\in\mathcal L_i(e)}a_{i\ell e}v_{ie}+10^{-8}}.
\tag{F20}
\]

Every real valid edge must have a strictly positive denominator before epsilon;
otherwise the graph fails. The decoder splits the edge sequence at every
invalid edge. Within each valid run it forms maximal connected components of
\(\{e:R_{ie}\ge0.5\}\). Each component contributes exactly one count. For a
location audit, collect every edge attaining the component maximum and choose
the floor of the midpoint between the first and last maximizer. The final
integer is

\[
\widehat c_i=D_{peak}(R_i,v_i)=
\#\operatorname{CC}\{e:v_{ie}=1\ \land\ R_{ie}\ge0.5\}.
\tag{F21}
\]

`D_peak` is called once per supplied valid identity and returns zero on an
empty valid-edge set. It has no smoothing, learned or
dynamic threshold, minimum distance, NMS, period, expert vote, rounding rule,
or window count. The audit-only mass
\(A_i=\sum_{e:v_{ie}=1}R_{ie}\) is recorded after the count and cannot enter a
metric, configuration, checkpoint, repair, or fallback. If threshold-0.5 peak
resolvability fails, TempoRAC is rejected.

## Training, Gradient, State, and Checkpoint Contract

### Optimization

Teacher training uses AdamW with learning rate \(3\times10^{-4}\), betas
\((0.9,0.999)\), epsilon \(10^{-8}\), weight decay \(10^{-4}\), gradient-norm
clip 1.0, and at most 200 epochs. Response training uses AdamW with learning
rate \(10^{-3}\), the same betas, epsilon, weight decay, and clip, and at most
100 epochs. Both use a 5% linear warmup followed by cosine decay to 10% of the
base rate, bfloat16 activations with float32 reductions, and full-identity
gradient accumulation to at least 8,192 valid edges per optimizer step.
Candidate seeds are 20260815, 20260816, and 20260817.

Teacher selection follows the topology/reconstruction/epoch/seed precedence
already stated. Response selection follows K4 synthetic pass, then lowest
held-out synthetic \(\mathcal L_{resp}\), then earliest epoch, then smallest
seed. Human evaluator fields never select an epoch, seed, threshold, or
configuration. All fixture generators, fixed projections, tolerances,
operator versions, deterministic flags, and constants above are serialized in
`temporac.execution.v1.json`; its canonical JSON SHA-256 is frozen before
implementation. An unspecified constant is a contract failure, not an
implementation default.

### Gradient and state boundaries

- Teacher gradients update only the static-code MLP, phase MLP, and decoder.
- Response gradients update only the shared local encoder and three response
  experts through Formulas F18--F20.
- Teacher, targets, masks, clocks, windows, NUDFT, reference, gates, taper,
  NOLA arithmetic, and decoder are frozen in response training.
- A fresh autograd graph and empty private buffers are created per full-identity
  batch. No state survives an optimizer step or crosses an identity.
- Canonical permutation of identity records and their keys may only reorder
  outputs. Permuting opaque keys consistently with their tracks is not expected
  to fail. Only intentionally broken controls that share a buffer, reuse one
  key for two simultaneous tracks, or reassign a key mid-track should fail.
- With one identity's pose changed and all other bytes fixed, untouched raw cue
  tuples, gates, expert responses, buffers, NOLA outputs, and counts must be
  bitwise identical in deterministic CPU evaluation and within (10^{-6}) on
  deterministic GPU evaluation.

## Decisive Causal Comparisons on Frozen Responses

After the canonical bank is frozen, cache
\(\{r^{slow}_{ie},r^{medium}_{ie},r^{fast}_{ie}\}\) once for every eligible
identity. The decisive arms modify only gates:

| Arm | Exact intervention |
|---|---|
| Local | Formula F14 per window |
| Global | Compute one full-track NUDFT with Formulas F10--F13, subtract the same window-weighted reference, and apply its single gate to every window |
| Uniform | Set every gate exactly to \((1/3,1/3,1/3)\) after cue computation |
| Blocked | Zero all cue coordinate values before F10 while retaining masks/clocks, recompute the now-invalid track reference, and require zero confidence and exact uniform gates |
| Within-track shuffled | Cyclically shift raw window tuples \((p,u,\gamma)\) by one reliable-window position ordered by start, keep the original track reference, and recompute F14 |

Uniform and blocked should be numerically identical at the gate; their separate
call traces detect an illicit non-cue path. No arm retrains an encoder or
expert. Cue shuffling must remove at least 80% of the local-over-global gain.

The one-output capacity control uses the same three internal two-block TCN
branches, dilations, and parameter tensors shapes as the expert bank, but all
branches receive the full unstratified target and their three logits are
uniformly averaged before one sigmoid output. It has one response output, no
expert identities, and no router. Trainable parameter count must match exactly,
measured multiply-add FLOPs over a 128-sample window must differ by at most 1%,
and its receptive-field set must equal \(\{9,17,33\}\) samples. Failure to meet
these receipts invalidates the control.

Held-out PCHIP and windowed-sinc resamplers form a \(3\times3\)
expert-by-tempo matrix: rows are slow/medium/fast experts and columns contain
windows within 0.10 log-frequency units of
\(-\log1.5,0,+\log1.5\). The cell is mean pulse loss F17 and split/merge rate.
Each diagonal cell must be strictly best in its column and beat the matched
one-output control with a paired source-orbit bootstrap lower bound above zero.

## Executable Build Gates G0--G5

Gates execute in numeric order except that G4's CPU-only fixtures may be built
beside G2; no pass is issued until all predecessors pass. A pass means the
named immutable artifact and its SHA-256 exist, not that prose says the test
was run.

| Gate | Required immutable artifacts | Exact observables and pass condition | Failure and authorization effect |
|---|---|---|---|
| G0 feature/vault firewall | `temporac.execution.v1.json`; feature-shard, vault, and audit-manifest schemas; per-file and per-sample hashes; forbidden-key scan; canonical-source-component split receipt; nuisance attack schema | Whitelist-only feature deserialization; all forbidden-key injections rejected pre-payload; train/development source-component intersection zero; test/sealed/heldout/results unreachable; nuisance attack tensor fixed to 37 channels: 17 joint masks, 17 confidences, frame-valid bit, centered `log1p` relative clock gap, and synthetic interpolation weight, with no ID/length/label/warp parameter | Stop all data-bearing training. Only contract and synthetic CPU work remain allowed |
| G1 degree-one teacher | Generator and resampler hashes; teacher checkpoint/hash; complete-orbit trace; attack matrix; origin trace | At least 95% asymmetric held-out orbits satisfy all F5--F8 landmark, coverage, seam, collision, reconstruction, alias, coherence, and origin-stability thresholds; every injected degree/harmonic/homeomorphism/bypass attack rejected; symmetric fixtures abstain; checkpoint precedence receipt exact | Reject the teacher and TempoRAC; no natural count-guided repair |
| G2 seven invariants | Object-ID trace, parameter/FLOP receipt, private-buffer key trace, raw-cue/final-gate perturbation trace, fusion/decode call graph, NOLA coverage trace, key-permutation trace | All seven invariants pass; raw cue unchanged by outside-window perturbation; any final-gate change is exactly reproduced by the reference path; canonical key permutation only reorders output | Implementation is not TempoRAC; stop |
| G3 routing/specialization | Once-trained bank hash; cached response-tensor hash; held-out \(3\times3\) matrix; one-output match receipt; five-arm gate traces | Every diagonal is best; each diagonal beats matched one-output control with paired lower bound above zero; all experts receive at least 10% effective training mass; same-response local beats global/uniform/blocked/shuffled; shuffle removes at least 80% of local-over-global gain | Kill the only routing contribution; do not add experts or a learned router |
| G4 pulse/NOLA/peak | Held-out pulse-pack hash; decoder call trace; split/merge oracle; grid-start traces | At threshold 0.5, exactly one component per event and zero extras for spacings 2,3,4,8,16,32,64 edges; zero split/merge on plateaus, pauses, invalid gaps, duplicate collapse, padding, seams, and starts shifted by 16 samples; every valid edge has positive NOLA coverage; decoder called once | Reject TempoRAC; integral audit cannot substitute |
| G5 supplied-track pilot | Frozen predictions for three seeds; evaluator hash; source-component bootstrap samples; all G0--G4 hashes embedded in run manifest | K7 primary and stationary criteria pass against the strongest matched nonlocal/control arm; all prerequisite and teacher-redundancy checks pass | Kill the canonical method; no paper, server continuation, or backup substitution follows automatically |

## Scientific Kill Gates K0--K7

The K gates are decision rules over G artifacts. They are evaluated in order;
the first failure is terminal for every later gate and is the reported reason.

| Gate | Observable and frozen threshold | Evidence source | Failure action |
|---|---|---|---|
| K0 protocol/firewall | G0 passes; no forbidden path reachable; physical feature/vault separation; source-component disjointness | packer, loader, filesystem reachability, and hash receipts | no data-bearing run |
| K1 response unit | G1 passes; later evaluator-only natural coherent-landmark coverage is at least 80% of eligible identities and source components, harmonic/off-grid fraction is at most 5%, and median teacher-event/count ratio is in \([0.95,1.05]\); no target or checkpoint changes after vault opening | synthetic topology suite first, then frozen natural teacher trace joined only in evaluator | reject response route; no human-guided repair |
| K2 architecture | All seven G2 invariants pass, including raw-cue locality versus documented reference dependence and canonical key-permutation equivariance | deterministic instrumented call traces | stop: implementation is not TempoRAC |
| K3 shortcut resistance | No-pose timestamp, nuisance-only, pose-shuffle, static-code-only, warp-metadata attack, and track-length attack each retain less than 10% of canonical synthetic gain; unseen resamplers retain at least 80% | same-capacity attack harness; forbidden inputs appear only in attacks | reject as shortcut |
| K4 expert mechanism | G3 passes; diagonal matrix, at least 10% expert mass, matched one-output win, and shuffled-cue loss of at least 80% of local advantage | once-trained bank and same cached responses | kill dominant claim |
| K5 pulse and reconstruction | G4 synthetic errors are zero; natural half-hop grid count change at most 0.5%; NOLA boundary split/miss error lower than independent window decode with paired interval above zero | identical response tensor through both orderings | reject strict peak route; no integral replacement |
| K6 identity isolation | Untouched outputs are bitwise equal on CPU and differ at most \(10^{-6}\) on deterministic GPU; consistent key permutation reorders only; deliberately shared/reused/mid-track-reassigned state controls fail | byte-identical multi-identity intervention traces | implementation failure; no private-state claim |
| K7 pilot efficacy and residual contribution | Three seeds; at least 5% relative reduction in video-first AvgMAE under frozen piecewise drift versus strongest matched nonlocal/control, paired source-component bootstrap lower bound above zero; stationary AvgMAE upper bound no worse than +0.02; teacher-only and matched one-output controls do not equal or exceed TempoRAC within uncertainty | frozen supplied-track development predictions and evaluator only | kill canonical method; no title, paper, submission, or backup authorization |

K3's shortcut models are attacks, not canonical alternatives. Their nuisance
adapter is the fixed Rademacher projection from the 37-channel G0 tensor to the
269-channel encoder width defined above, followed by the exact same trainable
encoder and one-output control. Absolute indices, source length, identity,
human fields, and raw warp parameters remain unavailable even to this
nuisance-only attack;
the separate warp-metadata attack is deliberately given the synthetic log
derivative solely to verify that the gate would detect a forbidden bypass.

## Minimal Claim-Driven Validation

### Supporting mechanism check: Shared response scales specialize by relative tempo

- **Minimal experiment:** The held-out \(3\times3\) unseen-resampler matrix
  from G3, compared with the exactly matched one-output control.
- **Decisive metric:** Pulse loss and split/merge rate per cell with paired
  source-orbit intervals.
- **Required evidence:** Every expert is diagonal-best, beats the one-output
  control in its assigned column, and receives adequate training mass.
- **Failure meaning:** There are no specialized shared response scales, so the
  dominant claim is false.

### Dominant claim: Window-local relative tempo selects those scales under drift

- **Minimal experiment:** Local, global, uniform, blocked, and shuffled gates
  on one frozen cached response tensor, followed by the same NOLA and decoder.
- **Decisive metric:** Video-first AvgMAE under frozen piecewise drift and the
  fraction of the local-over-global gain removed by shuffling.
- **Required evidence:** Local routing beats every nonlocal/control gate with a
  paired source-component lower bound above zero; shuffle removes at least 80%
  of the advantage; stationary degradation satisfies K7.
- **Failure meaning:** Local tempo is not the causal selector; kill the paper
  mainline rather than retrain controls or learn a residual.

### Prerequisite correctness checks, not claims

G1/K1 establish the teacher unit, G2/K2/K6 establish the seven-property graph,
and G4/K5 establish pulse resolvability and reconstruction order. The NOLA
comparison uses the identical local response tensor for independent window
decode versus NOLA then one decode. These checks may reject the system but do
not create parallel paper contributions.

## Failure Modes and Dispositions

| Failure | Detection | Binding disposition |
|---|---|---|
| Degree zero, reversal, double winding, harmonic lock, symmetric orbit, or unstable origin | G1/K1 complete-orbit and natural unit traces | reject teacher and TempoRAC |
| Static code or decoder bypass | constant/shuffled/noisy phase and shuffled-code reconstruction attacks | reject teacher architecture |
| Pulse split, merge, or threshold instability | G4/K5 fixed-0.5 packs and grid shifts | reject TempoRAC; keep integral audit non-decision-bearing |
| NUDFT shortcut or missingness/clock reliance | K3 no-pose, nuisance, shuffle, length, and metadata attacks | reject routing cue; do not add ACF by default |
| Expert collapse | effective mass, diagonal matrix, and one-output match | kill dominant claim |
| Local cue irrelevant | same-response global/uniform/blocked/shuffled interventions | kill dominant claim |
| Identity leakage | K6 byte-identical intervention and buffer traces | implementation failure |
| Teacher-only is already as effective as TempoRAC | K7 frozen teacher-only comparison | no residual routing contribution |
| Underpowered 51-video/9-component development cache | paired source-component interval includes zero | failure to advance, not permission to weaken a threshold |

## Novelty, Scope, and Authorization Boundary

The most that could survive is an empirical interaction claim at the
intersection of supplied identity tracks, shared response-scale experts,
identity-private state, relative window-local tempo, response fusion, and one
reconstructed decode. This remains provisional because every component is
established and the full TWCRAC method overlap is unresolved. PAMS-TCC, ACF,
NUDFT, soft gating, causal TCNs, phase, pseudo-targeting, NOLA, connected
components, private buffers, and one-decode order are not independently
claimed.

The current authorization is specification and synthetic/local unit work only.
It does not authorize a server connection, data-bearing training, test, sealed,
heldout, historical-result access, title/acronym freeze, novelty clearance,
paper drafting, result claims, submission, or continued compute after a failed
gate. WARP-PHASE remains terminally failed. The backup pivot remains separate
and is not substituted. A later PAMS-TCC initialization control, ACF fusion,
predicted-track stage, closest-prior run, or full paper plan requires its own
authorization after the canonical gates.

## Experiment Handoff Inputs

- **Frozen primary unit:** source-connected component, never individual video
  or person, for resampling and intervals.
- **Canonical data name:** partial-cache, GT-bbox-assisted AlphaPose
  supplied-track development pilot.
- **Metrics if authorized:** video-first AvgMAE and AvgOBO; supplied-track
  Period AP only after the official temporal matching wrapper passes its own
  fixture and is named as a supplied-track diagnostic.
- **Must-cache tensors:** teacher phase/origin/events, raw NUDFT tuple, track
  reference, each gate arm, three full expert responses, NOLA numerator and
  denominator, final response, components, call counts, and buffer keys.
- **Must-prove:** diagonal specialization, same-response local routing benefit,
  fixed-threshold pulse resolvability, and identity isolation.
- **Must-not-use:** current label-mixed pickle in a training process, test or
  historical paths, human-derived early stopping, WARP-PHASE artifacts,
  inferred SSHead, dynamic peak logic, or integral count repair.

## Compute and Timeline Estimate

| Hard-stop stage | Work | Engineer-days | A6000-hour cap | Advance condition |
|---|---|---:|---:|---|
| S0 contract and firewall | execution JSON, packer/schema design, forbidden-key and source-component receipts | 3--4 | 0 | G0/K0 pass |
| S1 teacher and topology | synthetic generators, three teacher seeds, canonical origin, attacks | 4--6 | 24 | G1/K1 synthetic pass |
| S2 graph and pulse contracts | local encoder, three TCNs, buffers, windows, NUDFT, NOLA, decoder, deterministic fixtures | 4--5 | 4 | G2, G4, K2, K5 synthetic, and K6 pass |
| S3 one-seed mechanism sanity | balanced warp strata, once-trained bank, cached interventions, diagonal matrix | 2--3 | 12 | G3/K3/K4 pass |
| S4 bounded development pilot | only after every prior hash is frozen; three complete seeds and frozen evaluator | 2--5 | 60 | G5/K7 pass |

The total is 15--23 engineer-days and at most 100 RTX A6000-hours. Each GPU
cap is a hard stop, not a target. Unused hours from a failed stage cannot be
spent on a rescue route without a new review. New human annotation cost is
zero; evaluator-vault use, if later authorized, occurs only after predictions
and all decision rules are frozen. These estimates authorize neither server
access nor a run.
