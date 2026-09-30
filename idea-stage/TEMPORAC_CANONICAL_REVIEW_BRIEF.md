# TempoRAC Canonical Research-Review Brief

## Decision scope

This document restores the collaborators' original paper idea as the only
canonical research mainline.  It is a review input, not an implementation or
performance claim.  The WARP-PHASE experiment is a terminal failed route and
the set-valued missing-pose idea is a backup-only pivot.  Neither may replace
the method below without explicit collaborator approval.

The working name is **TempoRAC: Identity-Indexed Local Tempo Routing for
Multi-Person Repetition Counting**.  The acronym and title remain provisional
until collision and venue checks pass.

## Non-negotiable scientific object

Given externally supplied person-indexed pose tracks

\[
X_i\in\mathbb{R}^{T_i\times J\times C},\qquad
m_i\in\{0,1\}^{T_i\times J}
\]

the system returns a per-person count vector

\[
\widehat{\mathbf c}=[\widehat c_i]_{i=1}^{N}
\]

The scene total is only the derived diagnostic
\(\sum_i\widehat c_i\).  The research question is whether identity-indexed,
window-local tempo routing improves per-person counting when different people
move asynchronously and when one person's pace changes within a track.

The intended architecture has seven non-negotiable properties:

1. A pose encoder is shared by every person.
2. Fast, medium, and slow response experts are shared across people.
3. Temporal state and all reconstruction buffers are private to each identity.
4. Tempo evidence is estimated independently in overlapping local windows,
   using masks and source-frame clocks rather than one global video tempo.
5. A soft router fuses expert responses before discrete counting.
6. Fused window responses are reconstructed per track with positive-taper
   normalized overlap-add.
7. Each complete identity response is decoded exactly once; independent
   window counts are never summed.

This structure, rather than WARP-PHASE, missing-pose sets, a scene-level
counter, or a per-person copy of an entire model, is the object to review.

## Claim and supervision boundary

The counting objective must read no human per-person count, period, cycle
boundary, density, or test label.  Known synthetic time-warp parameters may be
used as augmentation metadata.  Human annotations remain evaluator-only.
External supervision used by the detector, pose estimator, tracker, or any
pretrained backbone must be disclosed.  Therefore the permitted phrase is
"the counting objective uses no per-person count or period annotations," not
"annotation-free," "label-free," or "fully self-supervised."

The method is not the first MRAC system, the first person-wise counter, the
first pose counter, the first variable-speed counter, the first identity-aware
model, the first self-supervised RAC method, the first mixture of experts, or
the first overlap-add reconstruction.  MultiCounter, MultiCounter+, PAMS,
PoseRAC, RepNet, TransRAC, DeTRC, HTRM-Net, TWCRAC, generic mixture-of-experts,
and classical WOLA/NOLA define a narrow novelty boundary.

The maximum possible contribution, if supported by protocol-matched results,
is the intersection of:

- person-indexed pose-driven MRAC
- no human per-person count or period annotations in the counting objective
- identity-private state with parameters shared across people
- relative, within-track, window-local tempo specialization
- response-level soft fusion followed by per-track reconstruction and one
  final decode

No component is claimed as independently novel.

## Proposed minimum implementable graph

This section is a candidate contract for adversarial review.  It deliberately
exposes unresolved choices instead of presenting them as completed facts.

### Feature and clock path

Pose coordinates are converted to root-relative, scale-normalized COCO-17
positions, confidence, bone vectors, and first differences.  Frame and joint
masks remain explicit.  `sampled_frame_indices` supplies the source-frame
clock.  Repeated source indices have zero-duration transitions and cannot
contribute to a frequency or phase increment.

A shared encoder \(E_\theta\) produces a full-track latent sequence once.
Windows then index that latent sequence, avoiding a separate complete encoder
invocation for every person-window pair.  The initial supplied-track pilot
uses a fixed 128-sample window and 32-sample hop; these are development
hyperparameters to freeze before any evaluator access, not results.

### Local tempo evidence

For each identity-window \((i,\ell)\), a mask-normalized, non-circular ACF and
a Hann-windowed Fourier power distribution are evaluated over an explicitly
bounded lag/frequency support.  The method retains the full soft lag evidence
and an entropy/concentration confidence rather than treating a single hard
period estimate as ground truth.  A differentiable projection produces a
scalar local log-tempo coordinate \(u_{i\ell}\).  A stop-gradient robust
track reference \(\bar u_i\) is computed only from reliable windows, giving
the relative within-track coordinate

\[
z_{i\ell}=u_{i\ell}-\bar u_i
\]

Low-confidence windows must use a predeclared neutral mixture or state-based
carry; no ground-truth period fallback is allowed.

### Soft routing and shared experts

Three tempo anchors \(\mu_k\) correspond to slow, medium, and fast relative
pace.  The router combines anchor distance, tempo confidence, and the local
latent summary

\[
g_{i\ell k}=\operatorname{softmax}_k\left(
-\frac{(z_{i\ell}-\mu_k)^2}{\tau}
+a_{\psi k}(h_{i\ell},\gamma_{i\ell})
\right)
\]

The expert bank \(\{H_k\}_{k=1}^{3}\) is shared across identities, while each
identity has separate recurrent state.  Experts have matched parameter counts
and predeclared temporal scales.  A one-head capacity-matched control, uniform
routing, hard routing, shuffled tempo cues, and a soft global-tempo router are
mandatory controls.

### Response and reconstruction

Each expert emits a continuous local event response
\(r^{(k)}_{i\ell}(t)\).  Responses are mixed before any peak operation

\[
r_{i\ell}(t)=\sum_{k=1}^{3}g_{i\ell k}r^{(k)}_{i\ell}(t)
\]

For a strictly positive taper \(a_\ell(t)\), aligned window responses are
reconstructed on the source clock

\[
R_i(t)=
\frac{\sum_{\ell}a_\ell(t)r_{i\ell}(t)}
     {\sum_{\ell}a_\ell(t)+\varepsilon}
\]

Padding is excluded from both sums.  This is epsilon-stabilized weighted
averaging, not an exact perfect-reconstruction claim.  One fixed decoder
\(D\) is called once per identity

\[
\widehat c_i=D(R_i,m_i)
\]

The decoder, alignment convention, valid-support boundary, and any threshold
must be frozen without development count fitting.

## Proposed count-annotation-free training route

The original slides name PAMS-TCC, tempo routing, and track continuity but do
not define the latter objectives.  Current code also has no verified training
path for the proposed response, router, or decoder.  The review must therefore
judge the following minimum route rather than assume it works.

1. Train the shared encoder with the audited PAMS-TCC implementation only on
   feature-only training tracks and only where its period-evidence gate is
   valid.  This is an inherited starting objective, not a claimed PAMS paper
   detail and not evidence that the new system works.
2. Generate paired monotone local time-warps online.  The warp derivative is
   known augmentation metadata and supplies a relative-tempo transformation
   target, never a human period label.
3. Train the tempo coordinate so the change between paired windows matches
   the known log-warp derivative.  Train routing to shift consistently across
   the three relative anchors, with load and entropy diagnostics that expose
   collapse.
4. Train the response path with matched-view response consistency under the
   known warp correspondence plus a non-collapse motion-reconstruction or
   phase-cycle objective.  Every gradient path must be explicit; a
   stop-gradient teacher or pseudo-target cannot read evaluator labels.
5. Specialization is accepted only if each expert improves its assigned
   synthetic tempo region over a capacity-matched shared head on an unseen
   resampler and if natural-window ablations show that the router does not
   reduce to identity, action, interpolation, track length, or confidence.

The response target is the most serious unresolved interface.  A reviewer may
require one of two outcomes:

- freeze a mathematically complete count-mass/phase-response objective that
  preserves the architecture above and survives harmonic/non-collapse tests
- reject implementation authorization because no count-annotation-free path
  identifies one-cycle response peaks

It is not acceptable to hide this gap behind the phrase "PAMS-TCC + routing
consistency + track continuity."

## Existing negative evidence that must constrain the design

- No server branch implements the seven-property system.
- Server v63 is a code-only single-person PoseRAC channel classifier.
- Server v46 is a different ExtraTrees/spectral count system with protocol
  defects and human count supervision; it is quarantined.
- The current tracked PAMS checkout is a partial single-person reproduction
  that selects one dominant person and returns one scalar count.
- The disclosed PAMS period-head training target is missing; the local SSHead
  repair is an inferred repository experiment and performed poorly.
- The WARP-PHASE route terminated before real-data training because its
  count-blind selector Gate 2 failed all five preregistered criteria.  A hard
  ACF/FFT pseudo-period must therefore not be silently promoted to truth.
- The usable MultiRep pilot cache is partial and GT-bbox-assisted.  Supplied
  tracks can isolate the counter but cannot establish end-to-end MRAC.
- Test, sealed, and heldout paths are forbidden.  No eligible method result
  currently exists.

## Minimum evidence package if the graph survives review

The first authorized stage is a checksum-frozen, canonical-source-disjoint
train/development supplied-track pilot with a physical label vault.  It must
use three independent seeds and report video-first AvgMAE/AvgOBO, supplied-
track Period-mAP/AP50/AP75 when the official evaluator contract is satisfied,
paired source-cluster bootstrap intervals, and all collapse/shortcut receipts.

Required mechanism comparisons are:

- one matched shared response head
- three experts with uniform routing
- hard routing
- soft track-global routing
- soft window-local routing
- window-local routing with tempo evidence blocked or shuffled
- independent window decoding versus NOLA then one decode
- stateless, shared-scene-state, and identity-private-state variants

Required stress tests include piecewise acceleration/deceleration,
pause/restart, unseen interpolation/resampling, missing joints, shifted window
grids, identity-key swaps, and a one-person tempo intervention that leaves all
other input bytes unchanged.

Predicted-track evaluation, MultiCounter/MultiCounter+ contextual comparison,
HOTA/IDF1/IDSW, and person-count scaling are later stages and cannot be
inferred from the supplied-track pilot.

## Questions for the ARIS reviewer

1. Is the exact intersection scientifically defensible for ICASSP 2027, or is
   it an obvious engineering composition after MultiCounter+, PAMS, TWCRAC,
   local time-frequency analysis, and mixture-of-experts?
2. Does the proposed training route identify a response that can be decoded
   once per physical repetition without count, period, or boundary labels?
3. What is the smallest complete response objective that preserves the
   non-negotiable architecture and avoids the failed hard-period selector?
4. Do the proposed controls causally isolate local tempo routing, expert
   specialization, identity isolation, and overlap-add ordering?
5. What exact kill gates must pass before implementation, server training,
   paper claims, and the acronym/title are allowed?
6. Give a verdict among `REJECT`, `REVISE`, or
   `ADVANCE_ONE_KILL_ORIENTED_PILOT`, with 1--10 scores for novelty,
   specificity, falsifiability, protocol integrity, feasibility, and ICASSP
   fit.  Same-family review must remain provisional.
