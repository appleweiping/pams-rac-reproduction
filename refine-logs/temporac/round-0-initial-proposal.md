# Research Proposal: TempoRAC

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

MultiCounter and MultiCounter+ already define multi-person repetition counting
and identity-aware outputs.  PAMS already covers pose-driven periodic
self-supervision, period-adaptive multi-scale consistency, and localized tempo
stress.  RepNet predicts time-varying periodicity; HTRM-Net uses local temporal
context under non-uniform periods and interruptions; generic mixture-of-experts
already provides soft gates and load control; WOLA/NOLA is classical
reconstruction.  Consequently, none of the named components is a contribution.

The operational gap is narrower.  A scene-level or track-global tempo model
cannot represent two people moving at different rates or one person changing
pace within a track.  Independently decoding overlapping windows creates
boundary duplicates and misses.  Copying one full counter per person wastes
parameters and does not prove state isolation.  A local soft router over shared
response functions could address these failures, but only if its response has
a fixed physical unit.

That last condition is not supplied by the original slides.  TCC, time-warp
equivariance, response consistency, and generic motion reconstruction are all
invariant to integer changes in phase winding.  A model can therefore emit
two or more events per human repetition while satisfying those losses.  The
current PAMS SSHead is an inferred repository repair, not a disclosed source
objective, and the WARP-PHASE hard selector failed all five frozen stability
criteria.  Treating a hard ACF/FFT peak as period truth would repeat a known
failure.

The smallest adequate intervention is therefore not another routing loss.  It
is a separately gated, clock-blind primitive-orbit phase teacher that fixes the
response unit before the original TempoRAC router and experts are trained.

## Method Thesis

- **One-sentence thesis:** A degree-one primitive-orbit teacher can provide a
  count-annotation-free one-event-per-cycle target, enabling identity-private
  shared tempo-scale experts to be routed by full-distribution window-local
  tempo evidence and reconstructed before one final per-track peak pass.
- **Why this is the smallest adequate intervention:** The teacher fixes the
  only missing unit; the router is cue-only and parameter-free; the expert bank
  contains exactly three matched heads; NOLA and the decoder are fixed
  operators.  No learned router residual, period head, tracking subsystem,
  uncertainty model, or second pretext task is introduced.
- **Why this route is timely:** The paper is deliberately signal-processing
  led.  It uses masked delay-state topology, nonuniform local spectral
  evidence, response-rate specialization, and overlap-add reconstruction.
  No LLM, VLM, diffusion model, or RL component naturally solves the response
  unit or tracking-state bottleneck, so none is added.

## Contribution Focus

- **Dominant contribution:** An empirically tested, identity-indexed
  window-local soft tempo router over shared slow/medium/fast pose-response
  experts, trained from a degree-one count-annotation-free event teacher.
- **Supporting contribution:** Response fusion followed by positive NOLA and
  one fixed identity-level event decoder, evaluated against independent
  window decoding under boundary shifts.
- **Explicit non-contributions:** The phase teacher, private state allocation,
  supplied tracks, PAMS-TCC initialization, ACF/FFT, MoE, synthetic warps,
  NOLA, connected components, and one-decode ordering are enabling or
  established mechanisms, not independent novelty claims.

## Proposed Method

### Complexity Budget

- **Frozen/reused:** feature/vault separation, supplied-track COCO-17
  normalization and masks, source-clock duplicate collapse, mask-safe shared
  pose encoder primitives, monotone warp interpolation, positive taper/window
  construction, evaluator-only metrics, and fresh-graph optimizer receipts.
- **New trainable components:** one primitive-orbit teacher used only to make
  frozen targets, plus one matched three-expert response bank.  The soft router
  itself is deterministic and has no learned residual.
- **Intentionally excluded:** WARP-PHASE selector/loss/model, a learned period
  head, learned decoder thresholds, scene attention, cross-person memory,
  expert load penalties, a fourth expert, tracker re-identification, learned
  fallback, and any model-selection path from human count/period labels.

### System Overview

```text
feature-only supplied tracks (pose, masks, source clocks, opaque metadata)
  |
  +-- Stage A: clock-blind delay state -> primitive-orbit phase teacher
  |             -> frozen degree-one event impulses e*
  |
  +-- shared full-track pose encoder -> per-identity private expert states
           |                              |
           |                              +-> slow response expert
           |                              +-> medium response expert
           |                              +-> fast response expert
           |
           +-> overlapping local windows -> soft ACF + nonuniform FFT
                                          -> relative tempo distribution
                                          -> cue-only soft gate g

expert responses --soft response fusion--> window responses
window responses --positive NOLA---------> one full identity response R_i
R_i --one frozen connected-component peak pass--> c_i
all identities ----------------------------------> count vector c
```

### Inputs, clocks, and feature firewall

For identity \(i\), the feature reader returns

\[
X_i\in\mathbb{R}^{T_i\times17\times3},\quad
M_i\in\{0,1\}^{T_i\times17},\quad
q_i\in\mathbb{Z}^{T_i}
\]

Duplicate source indices are collapsed before modeling.  A temporal edge
\(e=(t,t+1)\) is valid only when both endpoints have the required cell support
and \(\Delta q_e=q_{t+1}-q_t>0\).  Position, bone, confidence, and signed
velocity features are always accompanied by frame, joint, cell, and padding
masks.  Identity keys remain Python metadata and never become tensor features.

The training process cannot deserialize `count_gt`, `density_gt`, period or
cycle intervals, raw annotation boxes, object mappings, evaluator handles, or
test paths.  Synthetic monotone-warp correspondences are allowed only in loss
construction and are never model inputs.

### Stage A: primitive-orbit response-unit teacher

Let the masked delay state \(y_{i,t}\) contain root-relative pose, bone vectors,
signed velocity and short symmetric lags.  The teacher phase map is memoryless
and time-translation equivariant

\[
z_{i,t}=f_\phi(y_{i,t},c_i)\in S^1
\]

where \(c_i\) is a static masked track code repeated at every time.  The phase
branch receives no recurrent state, absolute position, source-clock value,
window index, identity, or warp metadata.  A reconstruction decoder sees only
\((z_{i,t},c_i)\)

\[
\widehat y_{i,t}=G_\eta(z_{i,t},c_i)
\]

The explicit assumption is that each countable segment follows one primitive
masked delay-state orbit homeomorphic to \(S^1\) with no nontrivial rotational
symmetry.  A left inverse makes \(f_\phi\) injective on that orbit; positive
orientation fixes topological degree \(+1\).  Half/double/higher winding cannot
reconstruct an asymmetric primitive orbit.  Symmetric or non-identifiable
fixtures must abstain and later natural-data failures terminate the route.

The teacher loss is

\[
\mathcal L_{\mathrm{teach}}
=\mathcal L_{\mathrm{inv}}
+\lambda_v\mathcal L_{\mathrm{view}}
+\lambda_o\mathcal L_{\mathrm{orient}}
+\lambda_a\mathcal L_{\mathrm{alias}}
\]

where \(\mathcal L_{\mathrm{inv}}\) is mask-normalized Huber reconstruction,
\(\mathcal L_{\mathrm{view}}\) aligns exact clean/monotone-warp
correspondences, \(\mathcal L_{\mathrm{orient}}\) rejects negative phase
increments, and \(\mathcal L_{\mathrm{alias}}\) rejects an edge increment of
at least one quarter cycle.  All weights and tolerances are frozen on synthetic
primitive-orbit fixtures before natural evaluator access.  Among checkpoints
passing the left-inverse gate, selection minimizes positive phase variation;
the reconstruction gate prevents a zero-phase solution.

After the topology gate, unwrap the frozen teacher in cycle units

\[
A^*_{i,t}=\operatorname{unwrap}(\arg z_{i,t})/(2\pi)
\]

and emit a half-open event target on each valid edge

\[
e^*_{i,t}
=\mathbf 1\!\left[
\exists n\in\mathbb Z:
A^*_{i,t}<n\le A^*_{i,t+1}
\right]
\]

The alias gate makes \(e^*\in\{0,1\}\); pauses emit zero.  This is a frozen
pseudo-target derived only from the pose orbit, never from a human count,
period, density, or boundary.  The teacher and reconstruction decoder are
frozen; the decoder is discarded from TempoRAC inference.

### Shared encoder and identity-private expert states

A single shared encoder \(E_\theta\) runs once on each full identity track and
produces \(H_i\).  PAMS-TCC may initialize \(E_\theta\) only if its own
feature-only evidence gate passes; initialization is not required by the
response-unit proof and receives a deletion ablation.

The slow, medium, and fast experts are equal-width two-block causal
depthwise-separable TCN/GRU response modules with width 64, kernel 5, and fixed
temporal dilations \(4,2,1\).  Parameter counts must be identical.  Parameters
are shared across people; every \((i,k)\) has an independent causal state and
buffer.  Each expert advances once over the full identity.  Windows only read
slices, so overlap never advances state twice.

Each expert returns a bounded edge-event response

\[
r^{(k)}_{i,e}=\sigma(H_k(H_i)_e)
\]

### Window-local full-distribution tempo cue

Use windows of 128 retained samples, hop 32, and one final right-aligned window.
The cue reads only root-relative delay features inside that window, not the
full-track contextual latent.  On a fixed source-frequency grid it computes a
mask-normalized non-circular soft-lag ACF distribution and a Hann-weighted
nonuniform Fourier distribution.  It keeps all bins

\[
p_{i\ell}(f)
\propto
\sqrt{p^{\mathrm{acf}}_{i\ell}(f)
      p^{\mathrm{fft}}_{i\ell}(f)}
\]

\[
u_{i\ell}=\sum_f p_{i\ell}(f)\log f
\]

Confidence \(\gamma_{i\ell}\) multiplies valid-pair coverage, ACF/FFT
Bhattacharyya agreement, and one minus normalized entropy.  No support gives
\(\gamma=0\).  A stop-gradient robust confidence-weighted track reference
\(\bar u_i\) yields \(z_{i\ell}=u_{i\ell}-\bar u_i\).  Fewer than three
reliable windows invalidates the reference.

With fixed anchors
\(\mu=(-\log1.5,0,+\log1.5)\) and
\(\sigma_g=\log1.5/2\), the parameter-free soft router is

\[
\widetilde g_{i\ell k}
=\operatorname{softmax}_k\!\left[
-\frac{(z_{i\ell}-\mu_k)^2}{2\sigma_g^2}
\right]
\]

\[
g_{i\ell}
=\gamma_{i\ell}\widetilde g_{i\ell}
+(1-\gamma_{i\ell})(1/3,1/3,1/3)
\]

An invalid track reference gives the exact uniform mixture for every window.
There is no latent residual, learned fallback, hard selected period, expert
vote, or label-derived threshold.

### Count-annotation-free expert and continuity objectives

Experts are trained on balanced synthetic slow/medium/fast warp strata.  Let
\(p^*_{i\ell k}\) be the soft stratum assignment derived from known monotone
warp metadata.  The expert loss is

\[
\mathcal L_{\mathrm{spec}}
=\frac{
\sum_{i,\ell,e,k}p^*_{i\ell k}v_{i,e}
\operatorname{BCE}(r^{(k)}_{i,e},e^*_{i,e})}
{\sum_{i,\ell,e,k}p^*_{i\ell k}v_{i,e}}
\]

and the response-fusion loss is

\[
\mathcal L_{\mathrm{fuse}}
=\operatorname{BCE}\!\left(
\sum_k\operatorname{sg}(g_{i\ell k})r^{(k)}_{i,e},
e^*_{i,e}
\right)
\]

A track-continuity view masks or fragments a feature-only training track using
label-blind corruption.  On source-clock edges valid in both views it applies

\[
\mathcal L_{\mathrm{cont}}
=\operatorname{Huber}(R_i^{(a)}-R_i^{(b)})
\]

The total response objective is

\[
\mathcal L_{\mathrm{resp}}
=\mathcal L_{\mathrm{spec}}
+\lambda_f\mathcal L_{\mathrm{fuse}}
+\lambda_c\mathcal L_{\mathrm{cont}}
\]

The router is fixed and stop-gradient inside response losses.  The frozen
teacher and event targets receive no gradients.  Experts, private-state
projections, and the shared encoder receive gradients through response fusion
and NOLA.  Track continuity is an auxiliary robustness term, not a separate
contribution; its removal is mandatory.

### Response fusion, positive NOLA, and one final peak pass

Within window \(\ell\), response fusion precedes every discrete operation

\[
r_{i\ell,e}=\sum_k g_{i\ell k}r^{(k)}_{i,e}
\]

Use the strictly positive taper

\[
a_n=10^{-3}+(1-10^{-3})
\sin^2\!\left(\pi(n+1/2)/127\right)
\]

and reconstruct on valid, non-padding source-clock edges

\[
R_{i,e}
=\frac{\sum_{\ell:e\in W_\ell}a_{e-s_\ell}v_{i,e}r_{i\ell,e}}
       {\sum_{\ell:e\in W_\ell}a_{e-s_\ell}v_{i,e}+10^{-8}}
\]

The fixed decoder is called once per complete identity.  It splits only at
invalid-support gaps, forms connected components of
\(\{e:R_{i,e}\ge0.5\}\), chooses a deterministic plateau-center maximum from
each component, and returns the number of components

\[
\widehat c_i=D_{\mathrm{peak}}(R_i,v_i)
\]

There is no period-dependent minimum distance, NMS fitting, learned threshold,
expert voting, independent window count, or sum of window counts.  Threshold
0.5 follows from the binary phase-crossing target and is frozen before any
human count-label access.  A count-mass integral is retained only as an audit
diagnostic.  If the frozen pulse-resolvability gate cannot prevent split or
merged events, the strict final-peak contract fails and TempoRAC is rejected;
it is not silently replaced by an integral decoder.

### Training and inference order

1. Build physically separate feature and evaluator-vault artifacts; verify
   forbidden-key and source-disjoint receipts.
2. Train and freeze the primitive-orbit teacher on feature-only tracks and
   synthetic monotone views; pass topology, harmonic, alias, and reconstruction
   gates.
3. Optionally initialize the shared encoder with evidence-gated PAMS-TCC;
   never use the inferred SSHead.
4. Train the three shared experts and identity-private state with frozen teacher
   targets, fixed router, response fusion, NOLA, and continuity views.
5. Freeze every parameter and all thresholds before evaluator access.
6. At inference, compute shared full-track features and expert responses once,
   calculate local soft gates, fuse window responses, NOLA-reconstruct one
   response per identity, and call the peak decoder exactly once per identity.

### Gradient and state contract

- The teacher stage updates only the phase map, static code, and reconstruction
  decoder.
- The response stage cannot update the teacher, event targets, ACF/FFT cue,
  track reference, router, taper, masks, clocks, window starts, identity keys,
  or decoder.
- The response stage updates the shared encoder, matched experts, and their
  identity-private state projections through fusion and NOLA.
- No state, activation, or graph survives an optimizer step; a fresh graph is
  constructed for each full identity batch.
- Reordering identities reorders outputs only.  Perturbing one identity must
  leave every other identity's state, gates, response, NOLA buffer, and count
  bitwise unchanged in deterministic evaluation.

### Failure modes and diagnostics

- **Non-primitive/symmetric orbit:** teacher admits half/double winding or
  phase collisions.  Detect on frozen asymmetric and symmetric fixtures plus
  natural reconstruction/collision diagnostics.  Action: reject the no-human-
  count/period response route.
- **Clock/warp shortcut:** no-pose, timestamp-only, warp-metadata, or
  interpolation-residual controls retain the signal.  Action: reject before
  development efficacy evaluation.
- **Expert collapse:** one head dominates or all heads match.  Detect with
  diagonal unseen-resampler specialization and capacity-matched one-head
  comparisons.  Action: kill the routing contribution.
- **Tempo cue bypass:** cue blocking or within-track shuffling leaves the gain.
  Action: kill the local-router claim.
- **Pulse splitting/merging:** NOLA or expert uncertainty produces multiple or
  merged connected components.  Detect with shifted grids, seams, pauses, and
  synthetic impulse packs.  Action: reject the strict peak contract.
- **Identity leakage:** a one-person tempo intervention changes untouched
  identities.  Action: implementation failure; no private-state claim.
- **Teacher suffices:** frozen teacher-only inference matches TempoRAC.  Action:
  no expert-routing contribution.

### Novelty and elegance argument

The paper is not a claim that local spectral analysis, phase, MoE, or NOLA is
new.  It asks a focused interaction question: after the response unit is fixed
without human count/period annotations, does relative local tempo determine
which shared response scale should be trusted for each identity-window, and
does response reconstruction before one decode avoid boundary error?  This is
publishable only if matched controls show a non-additive benefit specifically
under within-track tempo drift.  Otherwise it is an obvious engineering
composition and must be reported as negative.

## Route Comparison and Rejected Alternatives

- **Selected Route A — degree-one pulse teacher plus cue-only router:** preserves
  the original fast/medium/slow, soft-routing, NOLA, and final peak graph while
  making the event unit falsifiable.
- **Rejected Route B — learned latent router plus generic consistency:** a
  learned residual can ignore tempo evidence, and generic consistency does not
  fix phase winding.  It adds parameters while weakening causal isolation.
- **Rejected Route C — integral phase-flow/WARP-PHASE:** it is a terminal failed
  route and replaces the user's peak-response graph with a different scientific
  object.
- **Rejected frontier additions:** foundation models, diffusion, and RL do not
  resolve winding number or overlap accounting and would create contribution
  sprawl.

## Claim-Driven Validation Sketch

### Claim 1: Local soft tempo routing yields genuine expert specialization

- **Minimal experiment:** compare one matched head, three uniform heads, hard
  local routing, soft track-global routing, canonical soft window-local
  routing, cue-blocked routing, and cue-shuffled routing on the same frozen
  supplied tracks and teacher.
- **Decisive evidence:** each expert is diagonal-best and beats the matched head
  in its unseen-resampler tempo region; window-local soft routing improves
  video-first AvgMAE by at least 5% versus the strongest matched control under
  piecewise drift, with paired source-component bootstrap lower bound above
  zero; stationary upper confidence bound is no worse than +0.02.
- **Failure meaning:** no routing/specialization contribution; stop the paper
  mainline rather than add experts or tune on counts.

### Claim 2: Fusion, NOLA, and one peak pass prevent overlap-boundary errors

- **Minimal experiment:** compare independent window peak/sum, uniform overlap
  averaging, positive NOLA then one peak pass, and half-hop grid shifts using
  identical local responses.
- **Decisive evidence:** zero split/merged/missed events on the synthetic pulse
  gate; no more than 0.5% count change under grid shifts; boundary duplicate or
  miss error lower than independent window decoding with a paired interval
  above zero.
- **Failure meaning:** the reconstruction-ordering claim is killed even if the
  local expert response is otherwise useful.

### Required causal diagnostic: identity isolation

Warp one person's pose track while every other input byte remains identical.
Untouched identities' state, gates, expert responses, NOLA buffers, and counts
must remain bitwise identical in deterministic execution.  Shared-scene state,
stateless windows, reset-at-window, and shuffled-key variants expose whether
private state is real isolation or bookkeeping.

## Experiment Handoff Inputs

- **Must-prove claims:** local relative-tempo routing causes specialization and
  drift-specific improvement; reconstruction before one decode reduces
  boundary errors.
- **Must-run controls:** teacher-only; one matched head; uniform/hard/global/
  local/shuffled-cue routing; PAMS-TCC deletion; independent-window/NOLA;
  stateless/shared/private/reset/shuffled-ID state; no-pose/timestamp/warp-
  metadata/track-length/confidence shortcuts.
- **Critical data/metrics:** checksum-frozen source-disjoint supplied-track
  train/development cache; evaluator-only counts and period intervals;
  video-first AvgMAE/AvgOBO; supplied-track Period-mAP/AP50/AP75 only after
  official contract validation; three seeds; paired source-component
  bootstrap intervals.
- **Highest-risk assumptions:** primitive pose orbit matches the human count
  convention; local soft cue survives irregular clocks and missingness; binary
  pulses remain resolvable after expert uncertainty and NOLA; the partial
  development cache has enough power.

## Compute and Timeline Estimate

- **Synthetic contract and unit gates:** CPU plus at most 4 A6000-hours.
- **Primitive-orbit teacher sanity and three-seed freeze:** hard cap 24
  A6000-hours.
- **TempoRAC sanity, one-head/uniform controls, and three-seed canonical pilot:**
  hard cap 72 A6000-hours before a new review.
- **Initial bounded ceiling:** 100 A6000-hours and 20 GiB of new artifacts;
  launch at one GPU and permit two-way parallelism only after memory, swap, and
  disk receipts pass.
- **Human annotation cost:** zero new annotations; evaluator use is permissioned
  and isolated.
- **Timeline:** approximately one engineering week for the deterministic graph
  and synthetic gates, followed by one bounded server week if all pre-training
  gates pass.  These are prospective caps, not measured runtimes.
