# TempoRAC Canonical Research Review

**Review ID:** `20260815_temporac_canonical_sol`  
**Reviewer:** `gpt-5.6-sol`, reasoning `ultra`  
**Fresh context:** `true`  
**Review independence:** `same-family`  
**Acceptance status:** `provisional`  
**Venue decision scope:** authorization of one development pilot, not paper acceptance or submission readiness  
**Canonical object:** the seven-property original TempoRAC graph in `TEMPORAC_CANONICAL_REVIEW_BRIEF.md`  
**Excluded substitutes:** WARP-PHASE, set-valued missing-pose pivots, scene-level counting, and independent per-person copies of a complete model

## Verdict

`ADVANCE_ONE_KILL_ORIENTED_PILOT`

This is a narrow, conditional advance. It does **not** clear novelty, authorize a server run, license a paper claim, or establish that the response objective is currently identified. The present proposal is not trainable as written: paired-warp consistency, routing consistency, response consistency, and generic motion reconstruction admit zero, phase-shift, scale, expert-permutation, and half/double/higher-harmonic solutions. The local PAMS SSHead is an independently inferred repair, not a disclosed PAMS training target, and the failed WARP-PHASE hard selector cannot supply the missing unit of repetition.

One canonical pilot is scientifically defensible only if the response unit is fixed before implementation by the primitive-orbit contract below. The contract keeps all seven TempoRAC properties and uses no human per-person count, period, cycle-boundary, or density field in optimization. If its injective phase gate fails, the correct outcome is `REJECT`; WARP-PHASE or a backup pivot may not be substituted automatically.

## Scores

These scores judge the supplied proposal before the corrective contract is implemented.

| Dimension | Score / 10 | Ruling |
|---|---:|---|
| Novelty | 4 | Every component is established; only the exact conjunction and a possible non-additive interaction remain. Full TWCRAC overlap is unresolved. |
| Specificity | 5 | Seven inference invariants are clear, but the response unit, expert state, gradients, and decoder calibration were not closed. |
| Falsifiability | 9 | The idea admits decisive topology, harmonic, routing, state-isolation, NOLA, and matched-baseline kill tests. |
| Protocol integrity | 3 | The usable cache is partial, GT-bbox-assisted, label-mixed, and supplied-track only; there are zero eligible method results. |
| Feasibility | 5 | A bounded development implementation is possible, but no current tree implements it and COCO-17/source-clock support is missing from the PAMS path. |
| ICASSP fit | 7 | Local time-frequency evidence, response-rate reconstruction, and overlap-add fit signal processing, provided the paper stays narrow and mechanism-led. |

## Principal findings

1. **The canonical idea is not the failed WARP-PHASE route.** WARP-PHASE failed a preregistered hard selector stability gate before real-data training. That failure forbids reuse of its hard pseudo-period, selector, phase head, objective, or efficacy narrative. It does not by itself refute a full-distribution local tempo cue inside the original TempoRAC graph.

2. **No implementation exists.** The method manifest leaves the shared encoder binding, private state, local estimator, router, learned experts, response fusion, NOLA, decoder, objective, and tests `SYNC-REQUIRED`. The tracked PAMS checkout accepts one MediaPipe-33 pose sequence and returns one scalar count. Its three “experts” are fixed smoothing/peak configurations followed by hard voting, not learned shared response experts.

3. **The currently proposed response training route is not identifiable.** View consistency preserves any integer winding of phase; an unconstrained reconstruction decoder can absorb reparameterizations; zero response survives if reconstruction bypasses the response; and soft mixtures have expert-permutation and scale gauges. Nothing in `PAMS-TCC + routing consistency + track continuity` says that one unit of response equals one physical repetition.

4. **The exact novelty is weak but testable.** The only defensible future claim is an empirical conjunction: on supplied identity tracks, do shared tempo-scale response experts, driven by relative window-local tempo evidence and reconstructed before one decode, improve within-track tempo-drift counting when the counting objective reads no human per-person count or period annotations? No component and no “first” claim is available.

5. **A mathematically meaningful gauge is possible, but only under an explicit assumption.** A clock-blind, memoryless delay-state phase teacher with an approximate left inverse can fix the winding number to one on a primitive pose orbit. This turns phase increments into a count-mass target without a hard period selector. The assumption is strong and must be killed empirically before natural-data training.

## Novelty adjudication

| Prior | Occupied territory | Residual, if any |
|---|---|---|
| MultiCounter | MRAC, instance-indexed temporal modeling, period localization, and per-person counts | None for task definition, asynchronous people, or per-person output |
| MultiCounter+ | Unified MRAC, spatial-temporal consistency, long/short-period awareness, synthetic pretraining | At most the no-human-per-person-count/period objective plus local response routing conjunction |
| PAMS | Pose-driven periodic SSL, period-adaptive multi-scale consistency, localized internal-tempo stress, inference-time peak consensus | Learned shared response functions with per-window relative routing are distinguishable only if specialization is demonstrated; “experts” alone are not novel |
| RepNet | Temporal self-similarity, synthetic repetitions, per-frame period and periodicity | No claim for time-varying period or response-then-count |
| TWCRAC | Pose, time-window-cycle framing, intra-video TCC, local periodic statistics, dynamic thresholds, non-stationary motion | Highest collision risk; the exact boundary remains unresolved until the full method is inspected |
| HTRM-Net | Local temporal context, non-uniform periods, interruptions, multi-scale temporal-relation fusion | Only the identity-indexed, response-expert, pre-decode-fusion conjunction remains |
| Generic MoE | Shared experts, soft gates, load balancing, specialization | No component novelty; only a task-specific causal benefit could count |
| WOLA/NOLA | Weighted normalized overlap-add reconstruction | Established operator; only its ordering relative to response fusion and one decode is testable |

The intersection is not automatically non-obvious. It becomes publishable only if a capacity-matched one-head model, PAMS/Track-PAMS, teacher-only inference, soft track-global routing, and cue-shuffled local routing fail in the specific within-track drift regime while TempoRAC passes stationary non-inferiority. Otherwise this is ordinary engineering composition.

## Identifiability ruling

### Why the supplied objective is insufficient

Let a learned cumulative response be (C(t)=\int_0^t R(s)\,ds). Matched monotone-warp consistency is preserved by (C'(t)=qC(t)+b) for positive integer (q); modulo-phase reconstruction can likewise replace its decoder to absorb (q). A flexible decoder with a bypass can ignore (C), leaving (R\equiv0). Expert labels can be permuted, response amplitude can be exchanged with decoder scale, and a hard spectral peak can lock to a harmonic. Therefore paired warps and reconstruction alone do not identify the evaluator's unit of “one repetition.”

This is not merely an optimization concern. Alternating-limb actions and pose trajectories with rotational symmetry can make half- versus full-cycle semantics observationally ambiguous. No count/period-annotation-free objective can recover a human annotation convention that is not present in the observations or assumptions.

### Conditional primitive-orbit gauge

For a valid track, construct a clock-blind delay state

\[
y_t=[\text{root-relative pose},\text{bones},\text{signed velocities/lags},\text{joint masks}],
\]

with no absolute position encoding, window index, recurrent state, source timestamp, warp parameter, or identity key. Assume that the noiseless valid states of one annotated repetition form a **primitive** (S^1) orbit with trivial rotational stabilizer. Train a memoryless, time-translation-equivariant phase map (f_\phi(y_t,c_i)\in S^1), where (c_i) is one static content vector repeated for the entire track, and a decoder (G_\eta) that sees only ((f_\phi,c_i)).

If (G_\eta\circ f_\phi=\mathrm{id}) on the orbit, then (f_\phi) has a continuous left inverse and is injective. A continuous injective self-map of (S^1) has topological degree (+1) or (-1). Enforcing non-negative temporal orientation fixes degree (+1). Harmonics (q\ge2) are (q)-to-one and cannot satisfy the left-inverse condition; a subharmonic is not a single-valued continuous map on the primitive circle.

This gives one response unit per primitive orbit **only if** the primitive-orbit, injectivity, sampling, and reconstruction gates pass. It does not prove that every MultiRep annotation follows that primitive orbit. Failure is terminal for the no-human-count/period objective.

## Minimum complete scientific contract

### 1. Input, clock, and supervision

- Input is a supplied-track batch of COCO-17 pose, joint masks, frame masks, and `sampled_frame_indices`; opaque identity keys remain metadata and never become tensor features.
- Duplicate source indices are collapsed mask-weightedly before temporal modeling. An edge (e=(t,t+1)) is valid only when both endpoints are valid and (Delta q_e=q_{t+1}-q_t>0). Zero-duration edges contribute neither tempo nor count mass.
- Windows are 128 retained samples with hop 32, plus a final right-aligned window. These values are frozen before evaluator access.
- Training-process deserialization is physically separated from `count_gt`, `density_gt`, periods, cycle boundaries, raw boxes, object mappings, and evaluator handles. Synthetic warp correspondences are permitted metadata and are never model inputs.
- The only permitted supervision sentence is: “the counting objective uses no human per-person count or period annotations.” External pose, detector, tracker, supplied-track, and synthetic-warp supervision are disclosed separately.

### 2. Shared encoder and private state

- One shared encoder (E_\theta) is invoked once per complete identity track. It may be initialized by the audited, evidence-gated PAMS-TCC encoder objective, but the inferred PAMS SSHead is not part of TempoRAC and PAMS initialization receives its own ablation.
- Three learned response experts (H_k), (k\in\{\text{slow},\text{medium},\text{fast}\}), share parameter objects across identities. For the pilot, freeze equal-width two-block causal depthwise-separable TCNs (width 64, kernel 5) with dilation 4, 2, and 1 respectively; parameter counts must match exactly.
- Each expert is advanced once over each full identity sequence. Its causal left-context buffer is keyed only by that identity and expert. Windows read cached expert outputs; overlapping windows never advance state twice. No cross-person attention, batch statistic, hidden state, reconstruction buffer, router history, or decoder accumulator is permitted.
- Each expert emits a non-negative edge-rate response in cycles per source frame,
  \(r^{(k)}_{i,e}=\operatorname{softplus}(H_k(h_i)_e)\).

### 3. Full-distribution local tempo evidence and router

- Tempo evidence is computed independently in each window from root-relative pose/delay features, not from a full-track contextual latent that can leak outside the window.
- Use mask-normalized, non-circular vector ACF and a Hann-weighted non-uniform DFT evaluated on the source-frame clock over one predeclared bounded frequency grid. Retain normalized ACF and spectral distributions; no `argmax`, selected lag, pseudo-period, or WARP-PHASE selector enters training or inference.
- Map both distributions to the same frequency grid and set

  \[
  p_{i\ell}(f)\propto\sqrt{p^{\rm acf}_{i\ell}(f)p^{\rm fft}_{i\ell}(f)},\quad
  u_{i\ell}=\sum_f p_{i\ell}(f)\log f.
  \]

  Confidence (gamma_{i\ell}\in[0,1]) is the product of valid-pair coverage, ACF/FFT Bhattacharyya agreement, and one minus normalized entropy. With no usable support, (gamma=0).
- The track reference is a stop-gradient confidence-weighted median (\bar u_i). Set (z_{i\ell}=u_{i\ell}-\bar u_i), anchors (\mu=(-\log2,0,+\log2)), and (\sigma_g=\log2/2).
- The pilot router is deliberately cue-only and non-trainable:

  \[
  \widetilde g_{i\ell k}=\operatorname{softmax}_k\!\left[-\frac{(z_{i\ell}-\mu_k)^2}{2\sigma_g^2}\right],\qquad
  g_{i\ell}=(1-\gamma_{i\ell})(1/3,1/3,1/3)+\gamma_{i\ell}\widetilde g_{i\ell}.
  \]

  Removing the latent residual (a_\psi(h,\gamma)) is a necessary minimum revision: otherwise the router can ignore the claimed tempo cue. A learned residual is a later, separately reviewed extension.

### 4. Response fusion, positive NOLA, and one decoder

Within a window,

\[
r_{i\ell,e}=\sum_k g_{i\ell k}r^{(k)}_{i,e}.
\]

Use the strictly positive taper

\[
a_{\ell,e}=10^{-3}+(1-10^{-3})\sin^2\!\left(\pi(e+1/2)/L_e\right)
\]

on valid, non-padding edges. Reconstruct once per track:

\[
R_{i,e}=\frac{\sum_{\ell\ni e}a_{\ell,e}r_{i\ell,e}}
{\sum_{\ell\ni e}a_{\ell,e}+10^{-8}}.
\]

The fixed decoder is an integral, not a learned or development-calibrated peak rule:

\[
\widehat c_i=D(R_i,q_i,m_i)=\sum_{e\in\mathcal E_i^{\rm valid}}R_{i,e}\Delta q_{i,e}.
\]

It is called exactly once for each complete supplied identity. No rounding, threshold, NMS, independent window count, or per-window summation appears in training or the primary metric. A rounded value may be displayed only as a secondary diagnostic.

### 5. Count/period-annotation-free losses

Train the primitive-orbit teacher first, separately from TempoRAC:

\[
\begin{aligned}
L_{\rm inv} &= \mathbb E_t\, d_m(G_\eta(f_\phi(y_t,c_i),c_i),y_t),\\
L_{\rm view} &= \mathbb E_{(t,t')\in\mathcal C}\,[1-\langle f_\phi(y_t,c_i),f_\phi(y'_{t'},c_i)\rangle],\\
\delta_t &= \frac{1}{2\pi}\operatorname{atan2}(z_t^{(1)}z_{t+1}^{(2)}-z_t^{(2)}z_{t+1}^{(1)},\langle z_t,z_{t+1}\rangle),\\
L_{\rm orient} &= \mathbb E_e\,\operatorname{ReLU}(-\delta_e),\\
L_{\rm alias} &= \mathbb E_e\,\operatorname{ReLU}(|\delta_e|-0.25),\\
L_{\rm teacher} &= L_{\rm inv}+\lambda_vL_{\rm view}+\lambda_oL_{\rm orient}+\lambda_aL_{\rm alias}.
\end{aligned}
\]

Here (d_m) is a joint-mask-normalized Huber loss, and (mathcal C) contains only exact correspondences from paired monotone augmentations. All (lambda) values and the reconstruction tolerance are frozen on a synthetic primitive-orbit suite before natural development labels are opened. The teacher has no count decoder and is frozen after passing the topology gate.

For each valid edge, form the stop-gradient teacher rate

\[
\rho^*_{i,e}=\operatorname{stopgrad}\!\left(\frac{\max(\delta_{i,e},0)}{\Delta q_{i,e}}\right).
\]

Train canonical TempoRAC with

\[
\begin{aligned}
L_{\rm track}&=\frac{\sum_{i,e}m_{i,e}\Delta q_{i,e}\,
\operatorname{Huber}(R_{i,e}-\rho^*_{i,e})}
{\sum_{i,e}m_{i,e}\Delta q_{i,e}},\\
L_{\rm head}&=\frac{\sum_{i,\ell,e,k}m_{i,e}a_{\ell,e}g_{i\ell k}\Delta q_{i,e}\,
\operatorname{Huber}(r^{(k)}_{i,e}-\rho^*_{i,e})}
{\sum_{i,\ell,e,k}m_{i,e}a_{\ell,e}g_{i\ell k}\Delta q_{i,e}},\\
L_{\rm TempoRAC}&=L_{\rm track}+\lambda_hL_{\rm head}.
\end{aligned}
\]

This is the minimum complete response objective. It contains no human count, period, boundary, or density loss. There is no routing-consistency term, learned decoder, entropy bonus, load-balancing penalty, seam loss, or extra pretext in the pilot; those additions would enlarge the causal story before the core is known to work.

### 6. Gradient contract

- Optional Stage A: PAMS-TCC updates only the shared encoder and uses no evaluator target. Its hard correspondence evidence is stop-gradient and is initialization evidence only.
- Teacher stage: (L_{\rm teacher}) updates only (f_\phi,G_\eta), and the static-content encoder. No contextual TempoRAC state, router, expert, NOLA output, decoder, clock value, or warp parameter is available to the teacher.
- Freeze the teacher and its artifact hash before TempoRAC training.
- TempoRAC stage: (L_{\rm TempoRAC}) updates the shared encoder, the three shared experts, and their identity-private state projections through response fusion and NOLA. It cannot update the teacher, tempo distributions, track reference, router, taper, clocks, masks, window starts, identity keys, or decoder.
- The response target is detached. No evaluator metric, count, early-stopping count, threshold calibration, or label-derived checkpoint choice may create a gradient or selection path.
- A fresh graph is built per full identity batch; state is not detached at window boundaries and never persists across optimizer steps.

## Mandatory controls

1. Primitive-orbit teacher-only count, to test whether the training teacher itself makes TempoRAC redundant.
2. One response head with total parameter/FLOP capacity matched to all three experts.
3. Three experts with uniform routing.
4. Hard local routing using the same cue.
5. Soft track-global routing with every other component unchanged.
6. Canonical soft window-local routing.
7. Canonical routing with tempo evidence blocked to the neutral mixture and shuffled within track.
8. Independent window decode/sum versus positive NOLA then one decode.
9. Stateless, shared-scene-state, identity-private-state, reset-at-window, and shuffled-identity-key variants.
10. PAMS and Track-PAMS on identical supplied tracks; RepNet only in a genuinely input/protocol-matched block; MultiCounter/MultiCounter+, HTRM-Net, and TWCRAC remain contextual unless their inputs and evaluator are matched.
11. PAMS-TCC initialization versus identical training without that initialization.

All comparisons use identical supplied tracks, feature-only training data, teacher, window grid, decoder, three full seeds, and source-component resampling. Native end-to-end numbers are never ranked against supplied-track results.

## Preregistered kill gates

| Gate | Must pass | Failure action |
|---|---|---|
| K0 — firewall/protocol | Feature-only shards and separately permissioned label vault; forbidden-key scan; exact hashes; zero train/dev canonical-source overlap; no test/sealed/heldout/result path reachable | Stop before any data-bearing training or pilot; pure contract and synthetic unit work may continue |
| K1 — topology/identifiability | On frozen synthetic primitive (S^1) orbits and unseen resamplers: +1 winding in at least 95% of cases; combined half/double/higher-harmonic lock at most 5%; negative phase edges at most 1%; latent collision rate at most 1%; count-mass error at most 2%; teacher reconstruction below the frozen left-inverse tolerance | Reject the count/period-annotation-free response route; do not tune with natural counts |
| K2 — seven invariants | Shared parameter object IDs across people; private state/buffer keys; outside-window perturbations leave local cue unchanged; simplex soft gates; fusion precedes any decode; positive finite NOLA coverage; decoder call count equals number of complete identities | Implementation is not TempoRAC; stop |
| K3 — shortcut resistance | No-pose, timestamp/position-only, warp-metadata, track-length/confidence-only, and matched-time pose-shuffle controls retain less than 10% of the canonical gain; unseen interpolation preserves at least 80% of it | Reject as clock/augmentation/metadata shortcut |
| K4 — expert mechanism | Every expert receives at least 10% effective training mass on the balanced synthetic tempo suite and beats the capacity-matched one-head model in its assigned unseen-resampler region with paired interval above zero; cue shuffling removes at least 80% of the local-over-global gain | Kill the routing/specialization contribution |
| K5 — reconstruction ordering | Shifted-window-grid count mass changes by at most 0.5%; positive NOLA reduces boundary duplicate/miss error over independent window decoding; padding and duplicate clocks contribute exactly zero | Kill the reconstruction mechanism claim |
| K6 — identity isolation | With one person's tempo changed and every other track byte-identical, untouched identities' state, response, and count change by at most (10^{-6}) in deterministic evaluation; shuffled keys cause the preregistered failure | Treat as implementation failure; private-state claim forbidden |
| K7 — pilot efficacy | Primary endpoint: at least 5% relative reduction in video-first AvgMAE versus the strongest matched nonlocal/control model under frozen piecewise drift, with paired source-component bootstrap lower bound above zero; clean stationary AvgMAE upper confidence bound no worse than +0.02; three independent seeds | Kill the canonical method; no backup substitution |
| K8 — closest prior | Teacher-only inference, one-head, Track-PAMS/PAMS, or a matched closest prior equals or exceeds TempoRAC within uncertainty | No residual method contribution |
| K9 — literature closure | Full TWCRAC method inspected; no equivalent identity-window local routing, response fusion, and reconstruction contribution; title/acronym collision check passes | No novelty or title freeze |

The 51-video/9-component supplied-track development cache may be underpowered for K7. An inconclusive interval is a failure to advance, not permission to weaken the threshold.

## Evidence authority and present blockers

| Evidence | Authority in this review |
|---|---|
| Canonical review brief and research brief | Authoritative for intended scope and non-negotiable properties; not implementation or efficacy evidence |
| Method manifest, server inventory, and inspected `src/pams` code | Authoritative for current implementation absence and local PAMS semantics |
| Experiment audit and pilot schema audit | Authoritative negative/protocol evidence: partial GT-assisted supplied tracks, label colocation, no eligible result |
| Novelty check | Same-family provisional primary-source ledger; sufficient to cap claims, but not to close inaccessible TWCRAC full-method overlap |
| WARP-PHASE claims-from-results | Authoritative only for that route's terminal selector failure and zero efficacy evidence; irrelevant as a substitute for TempoRAC |
| Kill attack/defense | Authoritative artifact-readiness critique; both agree the idea is not currently executable or submission-ready |
| v46/v62/v63 and historical results | Ineligible for TempoRAC efficacy, novelty, or protocol claims |

The blocking facts are: no canonical implementation; no verified response target; no physical feature/label split; incomplete pose coverage; no predicted-track semantics; no full TWCRAC comparison; no eligible results; and no evidence that human repetition units coincide with primitive pose orbits. These blockers cap the current status at a single supplied-track development pilot.

## Claim boundary

If and only if every gate passes, the maximum defensible claim is:

> We study whether identity-indexed, window-local soft tempo routing among shared pose-response experts improves supplied-track multi-person repetition counting under within-track tempo drift, using a counting objective that reads no human per-person count or period annotations, with response fusion and positive normalized overlap-add before one fixed per-track decode.

Forbidden claims include every `first` variant, `annotation-free`, `label-free`, `end-to-end`, `state of the art`, `constant-time`, first local-tempo method, first identity-aware counter, first learned experts, and novelty for ACF/FFT, MoE, private state, NOLA, response decoding, or one-decode ordering by themselves.

## Final disposition

Authorize only specification freeze, unit tests, the synthetic primitive-orbit topology gate, and—after K0–K6 pass—one three-seed GT-bbox-assisted AlphaPose supplied-track development pilot. Do not access server, test, sealed, heldout, or historical result paths under this authorization. Do not freeze the TempoRAC acronym, paper method, novelty statement, or result table. A failure of the primitive-orbit assumption, teacher-only control, matched one-head/PAMS control, or primary endpoint kills the canonical idea; it does not authorize WARP-PHASE or a backup pivot.
