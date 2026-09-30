# Research Proposal: WARP-PHASE — Explicit Local Warp-Derivative Phase Flow for Supplied-Track MRAC

> **PROSPECTIVE PILOT**  
> **NO ELIGIBLE RESULTS**  
> **SAME-FAMILY PROVISIONAL**

This is a prospective, kill-oriented method proposal. No canonical implementation exists, no eligible method run exists, and no numerical efficacy claim is licensed. Historical v46, v62, and v63 artifacts are audit inputs only. The only permitted protocol name is **GT-bbox-assisted AlphaPose supplied-track partial-cache development pilot**.

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

## Technical Gap

The closest literature occupies nearly every component-level description. MultiCounter and MultiCounter+ establish identity-indexed MRAC and variable-period multi-person counting. PAMS covers label-restricted pose periodic learning, period adaptation, inference consensus, and a localized internal-tempo perturbation. SimPer uses periodicity-varying augmentation, CycleCL learns phase-sensitive periodic features, DeepPhase learns unsupervised local motion-phase manifolds with reconstruction, and generic time-equivariant video learning already makes temporal transformations part of the representation objective. RepNet, Motion Feature Learning, HTRM-Net, Rethinking TSM, and JTSPS-Net crowd TSSM, time-varying period, reconstruction, masking, local context, and response decoding.

The residual question is therefore not whether local phase, pose SSL, time warping, private memory, or overlap-add works. It is whether a **known local derivative of a time warp should explicitly scale the signed phase increment** and whether that constraint adds count-relevant information beyond using exactly the same warps as ordinary augmentation. Current evidence does not answer that question, and full-method overlap with TWCRAC remains unresolved.

Naive enlargement is not an acceptable fix. A learned fast/medium/slow expert bank and router adds a second specialization hypothesis, makes augmentation fingerprints an easy shortcut, and blurs the one residual novelty boundary. A masked TSSM pretext stacks established mechanisms and can only be a matched control. Identity-private state is routine state allocation; it matters here as a fail-closed isolation constraint and a causal interference diagnostic, not as a scientific mechanism claim.

Two routes were considered:

- **Route A — explicit derivative-law phase flow:** keep one shared pose encoder and phase head, train on clean/warped correspondences, and constrain signed local phase rates by the known warp derivative.
- **Route B — learned tempo experts/router:** learn several response experts and route each window by estimated local tempo.

Route A is selected because it isolates one falsifiable signal-processing law with one deletion ablation. Route B is rejected for this pilot because specialization and routing would be additional trainable mechanisms, collide with nearby work, and make it harder to distinguish semantic phase learning from warp-parameter recognition.

## Method Thesis

- **One-sentence thesis:** For each supplied identity track, a unit-circle phase process should transform under a known local time warp according to the signed derivative law \(\dot\phi_{\tau}(q)=\tau'(q)\dot\phi(\tau(q))\); enforcing this law on discrete phase increments is the sole proposed learning mechanism.
- **Why this is the smallest adequate intervention:** The treatment and its primary control use the same feature-only pose tracks, encoder, identity-private state allocation, windows, warps, base self-supervision, normalized overlap-add, decoder, capacity, optimization, and run budget; the treatment adds only the derivative-law loss.
- **Why this route is timely in the foundation-model era:** The proposal uses augmentation metadata as structured supervision for an exact transformation law instead of adding a larger task-specific module stack. No LLM, VLM, diffusion model, RL policy, teacher model, or inference-time search is needed to test this signal-processing hypothesis.

## Contribution Focus

- **Dominant contribution:** At most one future, evidence-contingent method contribution: an explicit local warp-derivative objective for signed phase increments in identity-indexed supplied-pose MRAC.
- **Optional supporting contribution:** At most one diagnostic contribution: a one-person tempo intervention that measures leakage into untouched identities. It does not upgrade private state into a method novelty claim.
- **Explicit non-contributions:** MRAC; per-person output; pose input; external GT-bbox-assisted pose association; local windows; time-warp augmentation; phase representation; private recurrent state; temporal equivariance in general; normalized overlap-add; half-open boundary handling; one decode per identity; supplied/predicted protocol separation; TSSM; masked reconstruction; and phase-conditioned reconstruction.
- **Claim ceiling:** Even after a successful pilot, wording is limited to studying whether the explicit local law improves the anchored supplied-track pilot. Complete TWCRAC inspection and a fresh novelty review are required before any novelty freeze.

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone:** The audited GT-bbox-assisted AlphaPose supplied tracks; a single shared pose encoder; standard identity-keyed recurrent state; fixed 64-frame half-open overlapping windows; unit-circle normalization; established normalized overlap-add (NOLA); and one deterministic per-identity phase-mass decode.
- **New trainable components:** One shared encoder/state backbone and one small phase/activity head trained as a single model. The only proposed mechanism relative to the matched control is the explicit derivative-law loss; it does not introduce a separate expert or router network.
- **Tempting additions intentionally not used:** Learned fast/medium/slow experts, soft or hard routing, chirplets, Lomb–Scargle or multitaper stacks, switching filters, variational clocks, phase-synchronization graphs, re-entry caches, count ledgers, learned overlap fusion, and predicted-track association.
- **Control-only component:** Masked geometric TSSM ridge completion may be run as a capacity/input-matched control, but it is never inserted into the primary method and cannot become an automatic fallback.
- **Diagnostic-only component:** Identity-private state allocation is required to prevent implementation leakage and is tested causally only after the backbone exists. It is not a second method contribution.

### System Overview

```text
trusted offline repacker
  source train/dev pickle (label mixed)
    ├─> feature-only, non-pickle shards + hashes + whitelist receipt
    └─> separately permissioned evaluator label vault

training process (cannot open source pickle or label vault)
  feature-only pose track p: (x, y, confidence), masks, sampled_frame_indices
    -> masked shared pose encoder
    -> recurrent state h[p] keyed only by local supplied-track slot
    -> fixed 64-frame overlapping half-open windows
    -> shared unit-circle phase/activity head
    -> signed phase increments and known clean/warp correspondences
    -> base self-supervision + explicit derivative-law loss

inference/evaluation
  per-window phase mass
    -> established Hann NOLA on interval responses
    -> one sum/decode for each supplied identity
    -> per-person count vector
    -> evaluator process joins vault labels only after predictions are frozen
```

There is no learned tempo expert, no learned router, and no scene-level count head. Scene totals may be computed after per-person decoding solely as a diagnostic.

### Core Mechanism

#### Input, masks, and the only admissible clock

For supplied identity \(p\), the feature-only input is \(X_p=\{x_{p,j}\}_{j=0}^{T-1}\), where each pose contains normalized image \(x,y\), AlphaPose confidence, a frame-valid mask, and a joint-valid mask. The clock is

\[
q_j=\texttt{sampled\_frame\_indices}[j], \qquad \Delta q_j=q_{j+1}-q_j.
\]

`source_length` bounds the valid clock domain. FPS, wall-clock timestamps, a presumed 320-step uniform clock, filenames, canonical IDs, source IDs, and provenance fields are not model inputs. A temporal interval is valid only when both frames are valid, enough joints pass the frozen confidence rule, and \(\Delta q_j>0\). Adjacent repeated indices and all zero-step intervals are masked from temporal losses, phase integration, NOLA mass, and evaluation traces. The recurrent state is held rather than updated across an invalid interval.

Missing joints are never interpreted as zero-valued coordinates. Their coordinates and velocities are masked before normalization and reconstruction. A warped pose interpolated between two source poses is valid for a joint only when both source endpoints are valid; otherwise that joint remains missing. If the frozen minimum-joint threshold is not met, the phase output is recorded as unavailable and contributes no count mass.

#### Warp convention

The augmentation is defined in source-frame-index units. Let \(\tau:[q_0,q_{T-1}]\rightarrow[q_0,q_{T-1}]\) map a target-view clock location to a clean-view source location. A forward warp is continuous, monotone non-decreasing, and piecewise affine. Its breakpoints, segment slopes, interpolation kernel, and random seed are logged. The warped pose is

\[
X_p^{\tau}(q_j)=\operatorname{Interp}\bigl(X_p,\tau(q_j)\bigr).
\]

For interval \(j\), the known discrete derivative is

\[
a_j=\frac{\tau(q_{j+1})-\tau(q_j)}{q_{j+1}-q_j}.
\]

Training schedules use endpoint-preserving, multi-segment warps whose positive slopes are normalized so the full source domain remains covered. A pause is a legitimate segment with \(a_j=0\) and \(\Delta q_j>0\); it is distinct from an original duplicate sample, for which \(\Delta q_j=0\) and the interval is masked. The primary development diagnostic uses a frozen list of multi-segment schedules, not schedules selected after reading counts.

Reversal is a separate globally monotone non-increasing transform with swapped endpoints; local direction changes are not mixed into a forward warp. Under reversal, \(a_j<0\), and the signed phase rate must reverse sign. Count decoding remains orientation-invariant by using gated absolute phase mass, but the signed rate trace and reversal loss remain visible. Forward and reversed copies are never pooled as independent videos.

#### Unit-circle phase and discrete derivative law

The shared head emits a normalized complex phase

\[
z_{p,j}=\frac{u_{p,j}}{\lVert u_{p,j}\rVert_2+\epsilon}\in\mathbb{S}^{1}
\]

and a bounded activity/confidence \(c_{p,j}\in[0,1]\). For any valid interval, the principal signed increment and rate are

\[
\delta_{p,j}=\operatorname{Arg}\!\left(z_{p,j+1}\overline{z}_{p,j}\right)\in(-\pi,\pi],
\qquad
r_{p,j}=\frac{\delta_{p,j}}{\Delta q_j}.
\]

The treatment enforces

\[
r^{\tau}_{p,j}
\approx
a_j\,\mathcal I\!\left(r^{\mathrm{clean}}_p,\frac{\tau(q_j)+\tau(q_{j+1})}{2}\right),
\]

where \(\mathcal I\) is a fixed masked interpolation of the clean-view rate at the mapped interval midpoint. This is the explicit local warp-derivative law. The loss is a masked robust error on the signed rates. It is invariant to a global phase offset and changes sign under reversal. A pause has target rate zero. The treatment does not receive count, density, period, cycle boundary, action label, or warp-class label.

The exact discrete correspondence, not a continuous approximation, is also logged: predicted warped increment is compared with the integral of interpolated clean rate over \([\tau(q_j),\tau(q_{j+1})]\). The implementation may use that integral form as the numerical target, while the derivative form above remains the scientific law. Treatment and control must share the same interpolation and masking code.

#### Alias and harmonic handling

The principal `Arg` is identifiable only below Nyquist. An interval is marked alias-risk when the clean-derived target increment magnitude reaches \(\pi-\varepsilon\), when its interpolation support is invalid, or when alternative unwrappings are equally plausible. Alias-risk intervals are excluded from the derivative loss and counted in a mandatory coverage report; they are not clipped into apparently valid targets. If excluding them removes the estimand for a source component, K7 applies.

Half- and double-frequency locking are not repaired with a post hoc expert. The frozen diagnostics compare phase advance, autocorrelation peaks, and evaluator-only period agreement at \(\tfrac12\), \(1\), and \(2\) times the candidate frequency. A large ambiguous fraction, unstable sign under reversal, or a harmonic choice that changes with the warp triggers K5. The phase gauge is fixed per track using a deterministic stop-gradient orientation rule based only on the clean forward-view median signed increment; labels are not used.

#### Shared encoder and identity-private state

All identities share encoder, recurrent, and head parameters. State tensors are stored and updated under the supplied local person-slot key, never a raw object ID or annotation mapping. The sequence encoder updates each track once in chronological order; overlapping windows read the encoded sequence and do not update the recurrent state a second time. This prevents overlap-dependent state duplication.

Private state is an implementation constraint and a diagnostic factor. Instrumentation must prove that hidden state, phase history, NOLA accumulators, decoder mass, and caches never cross supplied identity keys. Stateless, shared-scene, reset, swap, shuffled-key, and track-length/confidence-only variants are diagnostic controls. No architectural novelty is assigned to dictionary-keyed or recurrent state.

#### Established NOLA and one decode

Each half-open window \([b,b+64)\) predicts interval mass

\[
m^{(b)}_{p,j}=c^{(b)}_{p,j}\frac{|\delta^{(b)}_{p,j}|}{2\pi}.
\]

Fixed Hann weights are normalized over all windows covering interval \(j\):

\[
\bar m_{p,j}=
\frac{\sum_{b\ni j}w^{(b)}_j m^{(b)}_{p,j}}
     {\sum_{b\ni j}w^{(b)}_j+\epsilon}.
\]

Invalid and zero-step intervals have zero numerator and are excluded from the denominator. The continuous count is decoded exactly once per supplied identity,

\[
\hat C_p=\sum_j \bar m_{p,j},
\]

and any rounded diagnostic is derived afterward under a frozen rule. There is no per-window count summation, no duplicate overlap mass, and no learned fusion. Grid-shift and seam-mass receipts must show that changing legal window origins does not create or destroy phase mass beyond the frozen tolerance.

#### Training losses

The prospective objective is

\[
\mathcal L=
\lambda_{\mathrm{deriv}}\mathcal L_{\mathrm{deriv}}
+\lambda_{\mathrm{base}}\mathcal L_{\mathrm{base}}
+\lambda_{\mathrm{rec}}\mathcal L_{\mathrm{masked\text{-}vel}}
+\lambda_{\mathrm{var}}\mathcal L_{\mathrm{anti\text{-}collapse}}
+\lambda_{\mathrm{seam}}\mathcal L_{\mathrm{overlap}}.
\]

- \(\mathcal L_{\mathrm{deriv}}\): robust signed-rate error under forward warps, pauses, and reversal; this is the only proposed mechanism.
- \(\mathcal L_{\mathrm{base}}\): a frozen sum of (i) a CycleCL-style relative-phase contrastive loss that matches pairwise phase differences between two weak spatial views at the same clock locations and (ii) an activity binary loss whose pseudo-positive intervals preserve chronological pose order and whose pseudo-negative intervals use a predeclared matched-time block shuffle. The construction reads no evaluator labels and is shared unchanged with the augmentation-only control.
- \(\mathcal L_{\mathrm{masked\text{-}vel}}\): capacity-limited reconstruction of visible pose velocity from the shared latent and phase, using a fixed narrow decoder and missing-joint masks. Reconstruction is an anti-collapse aid, not a contribution.
- \(\mathcal L_{\mathrm{anti\text{-}collapse}\): variance/covariance regularization on encoder features and temporal phase variance, shared with all learned controls.
- \(\mathcal L_{\mathrm{overlap}}\): offset-invariant agreement of signed rates on overlapping valid intervals; NOLA itself remains fixed.

The identical augmentation-only control sets \(\lambda_{\mathrm{deriv}}=0\) and changes nothing else: same shards, clean/warp pairs, warp schedules, masks, architecture, initialization distribution, batches, optimizer, steps, base losses, decoder, windows, NOLA, evaluation, and three seeds. It is therefore a direct test of the explicit law, not of augmentation exposure.

Loss weights are not chosen by development AvgMAE, Period-mAP, count labels, or period labels. Before evaluator access, set \(\lambda_{\mathrm{deriv}}=1\) after an EMA scale normalization of each loss. For each auxiliary term, choose the smallest value in the frozen grid \(\{0,0.01,0.03,0.1,0.3\}\) that passes training-only reconstruction, non-collapse, and seam diagnostics on held-out source components from the training split. Lock the chosen vector for all three seeds and all matched learned controls. If no vector passes without one auxiliary gradient contributing more than 25% of the shared-encoder gradient norm of the derivative term, stop rather than retune on development outcomes. The augmentation-only control reuses the locked auxiliary weights from the treatment.

### Optional Supporting Component

- **Only include if truly necessary:** No optional trainable supporting component is included.
- **Identity isolation diagnostic:** After the primary backbone exists, perturb one supplied identity's tempo while keeping every other track, clock, and feature byte-identical. Measure changes in untouched identities' responses and counts under private, stateless, shared-scene, reset, swapped-state, shuffled-key, and track-length/confidence-only conditions.
- **Masked TSSM control:** A capacity/input-matched masked geometric TSSM ridge-completion model may appear only in the comparator table. It cannot initialize, replace, or rescue the primary mechanism in this study.
- **Why this does not create contribution sprawl:** Neither diagnostic changes the primary network or primary endpoint; each can only falsify an isolation or mechanism interpretation.

### Modern Primitive Usage

- **Which LLM / VLM / Diffusion / RL-era primitive is used:** None.
- **Exact role in the pipeline:** Not applicable.
- **Why it is more natural than an old-school alternative:** The bottleneck is an explicit signal transformation law with exact synthetic correspondences. Adding a foundation model, teacher, planner, reward model, or search policy would add supervision and capacity without isolating the derivative-law hypothesis. Modern leverage here is the disciplined use of known transformations as structured self-supervision and matched controls.

### Integration into Base Generator / Downstream Pipeline

There is no base generator. A trusted offline repacker first produces feature-only train/development shards and a physically separate evaluator vault. The model consumes supplied pose tracks only. The shared encoder/state backbone produces one chronological representation per identity, the phase head runs in overlapping local windows, NOLA reconstructs interval mass, and one per-identity decode produces the output vector. Evaluation joins frozen predictions to vault labels by opaque keys in a separate process.

No raw GT boxes, object IDs, mappings, annotation names, canonical source IDs, or source filenames enter training or inference. GT boxes have already assisted upstream pose association and must remain disclosed in the protocol name. Predicted-track claims and tracking metrics are outside this pilot because the audited cache has no autonomous per-frame track-ID sequence or frozen tracker artifacts.

### Training Plan

#### Gate 0 — physical feature/label separation

No training may start from the current label-mixed pickle. A trusted, isolated packer may read it once and must emit:

1. feature-only, non-executable shards containing only `motion`, `person_mask`, `frame_mask`, `sampled_frame_indices`, `source_length`, opaque hashed sample keys, and local person-slot indices;
2. a separately permissioned evaluator vault containing count/density/period/boundary and mapping fields;
3. a source-connected-component split manifest used only by the split/bootstrap layer;
4. per-file and per-sample SHA-256 receipts; and
5. a forbidden-key scan receipt.

The training entry point must validate the shard schema and hashes before opening array payloads and fail closed if any key outside the whitelist exists. It must reject, before training-process deserialization, at least: `count_gt`, `density_gt`, count, period, cycle or boundary fields, raw annotation boxes, `source_json`, `object_ids`, `person_object_ids`, `object_mapping`, `gt_object_pose_coverage`, `gt_bbox_assisted_pose_association`, and `source_alphapose_json`. The training account/process must lack read permission to the vault, the original label-mixed pickle, and all forbidden test paths. A successful receipt is necessary but does not itself license a scientific claim.

#### Gate 1 — frozen protocol

Freeze checksums for the partial-cache train/development artifacts, canonical-source connected components, opaque-key join table inside the evaluator, forbidden test paths, video-first AvgMAE/AvgOBO equations, supplied-track Period-mAP/AP50/AP75 implementation and explicit display scale, and the paired source-component bootstrap. Development labels remain available only to the untouched evaluator. Every artifact and report must say `partial-cache`, `GT-bbox-assisted`, and `supplied-track`.

#### Gate 2 — deterministic data and warp unit tests

Before learning, test repeated-index masking, missing-joint interpolation, half-open windows, NOLA denominator coverage, zero-mass invalid intervals, pause segments, globally decreasing reversal, alias masking, and exact derivative targets on synthetic unit-circle signals. A clean sinusoid under a known warp must satisfy the discrete integral target within numerical tolerance. These are engineering tests, not eligible results.

#### Gate 3 — sanity run

Use a small training-only source-component subset and one non-decision seed to verify loss decrease, nonzero phase variance, finite gradients, no cross-ID buffer access, and deterministic prediction receipts. The label vault remains closed. Failure stops the full pilot.

#### Gate 4 — frozen three-seed pilot

Run three independent full trainings for the derivative-law model and every decision-bearing learned comparator with distinct model/data-order seeds. Save commit, configuration, environment, hardware, seed, logs, checkpoints, raw per-person predictions, phase traces, source-component IDs held by the evaluator, and completion receipts. Report mean and sample standard deviation across seeds. No seed may be discarded, ensembled into a pseudo-three-seed result, or replaced after evaluator access.

#### Gate 5 — untouched development evaluation

Evaluate the predeclared clean and piecewise-warp sets once the run set is complete. The primary bootstrap resamples source-connected components with replacement, keeps all videos and people within a sampled component together, preserves treatment/control pairing and seed pairing, recomputes video-first AvgMAE, and averages the paired effect over the three seeds. Report the percentile 95% confidence interval and the seed-level effects.

### Failure Modes and Diagnostics

- **Clock shortcut:** a no-pose model predicts from `sampled_frame_indices`, warp parameters, masks, or nuisance metadata. Detect with timestamp/no-pose, nuisance-only, matched-time pose shuffle, crop-origin randomization, and unseen-resampler controls. Any shortcut may retain no more than 20% of the treatment gain.
- **Augmentation fingerprint:** the encoder identifies interpolation kernels or warp schedules instead of motion. Detect with held-out resamplers, held-out slope/breakpoint families, and byte-matched clean/warp metadata. A gain erased by the unseen resampler triggers K4.
- **Phase collapse:** phase or response variance vanishes, encoder effective rank collapses, or confidence becomes uniformly zero. Detect with frozen variance, covariance-rank, confidence, and active-track coverage receipts. Do not add a new module after failure.
- **Alias/harmonic lock:** principal increments wrap or lock at half/double frequency. Detect with alias coverage, reversal sign, phase-correspondence, and evaluator-only harmonic diagnostics. Mask predeclared alias intervals; kill if the remaining estimand is not defensible.
- **Pause/interruption error:** phase mass continues through synthetic pauses or disappears around interruptions. Detect with pause segments, block occlusion, unrelated motion, and clean-to-corrupt traces.
- **Missing-joint selection:** only easy, well-observed tracks remain. Report valid-joint/valid-interval coverage by source component and compare missingness strata. K7 applies if missingness destroys the estimand.
- **Window seam mass error:** window origins change the count. Detect with multiple legal grid origins and an exact mass ledger before the one decode.
- **Cross-identity contamination:** perturbing one identity changes another identity's hidden state, response, or count. Detect with byte-identical untouched tracks plus state instrumentation, reset, swap, shared-scene, stateless, and shuffled-key controls.
- **Protocol leakage:** labels, object mappings, raw boxes, source identity, test files, or prior v46/v62/v63 outputs enter training or selection. The forbidden-key/permission/hash gates fail closed.
- **Novelty collision:** full TWCRAC reveals the same mechanism. K8 applies before drafting or claim freeze.

#### Frozen K1–K9 stop rules

- **K1 — primary effect/uncertainty failure:** The derivative-law model survives only if it both achieves at least a 5% relative reduction in piecewise-warp development video-first AvgMAE over identical augmentation-only **and** has a paired source-component bootstrap 95% confidence interval for the positive improvement that excludes zero. Otherwise, kill the primary mechanism.
- **K2 — closest prior matches:** If PAMS/Track-PAMS, SimPer-style, CycleCL-style, DeepPhase-style, or matched generic time-equivariant learning matches or exceeds the derivative-law model within uncertainty, kill the residual method claim; private state and TSSM cannot rescue it.
- **K3 — derivative law is redundant:** If removing derivative scaling, using direct signed-frequency regression, or using global resampling only matches the model within uncertainty, conclude that the explicit local transformation law is empirically redundant.
- **K4 — shortcut/artifact:** If a timestamp/no-pose or nuisance-only model retains more than 20% of the treatment gain, an unseen resampler erases the gain, or matched-time pose shuffling preserves phase/count behavior, reject the result as an augmentation or clock shortcut.
- **K5 — phase is not identifiable:** If phase/response collapse, low effective rank, reversal-sign instability, half/double-frequency locking, seam mass error, or non-identifiable phase exceeds the frozen tolerances, kill phase-flow instead of adding post hoc modules.
- **K6 — integrity gate failure:** If feature/label separation, source isolation, forbidden-key receipt, untouched evaluator, permissions, hashes, or test-access controls cannot be demonstrated, stop; no count/period-label-restricted claim is licensed.
- **K7 — clock/missingness invalidates the estimand:** If missing-pose selection or repeated-frame timing prevents a defensible estimand, if alias masking removes required coverage, or if the method requires unavailable FPS/timestamps, rebuild the cache or restrict work to engineering diagnostics.
- **K8 — novelty collision:** If complete TWCRAC inspection reveals the same explicit local warp-derivative phase mechanism or an equivalent contribution, kill or materially re-anchor novelty before drafting and rerun novelty review.
- **K9 — forbidden fallback promotion:** If the primary fails and masked geometric TSSM is proposed as an automatic replacement, stop. TSSM remains a control unless a separate preregistered study and fresh novelty review authorize a new anchor.

### Novelty and Elegance Argument

The proposal is intentionally narrower than the original learned-expert hypothesis. It does not claim phase learning, temporal equivariance, local windows, pose SSL, private state, reconstruction, NOLA, or per-person decoding. It asks one residual question: does a known local time-warp derivative provide useful supervision for the *signed increment* of an identity-indexed phase process beyond exposure to the same augmentation?

That question is falsifiable with a single deletion: set \(\lambda_{\mathrm{deriv}}=0\) while holding everything else fixed. PAMS/Track-PAMS, SimPer-style, CycleCL-style, DeepPhase-style, generic time-equivariant, and direct signed-frequency comparators test whether the residual is already captured by nearby mechanisms. Identity isolation and masked TSSM remain diagnostic/control roles and cannot become parallel contributions. This is the smallest story compatible with the current four-page venue budget and the same-family provisional novelty boundary.

## Claim-Driven Validation Sketch

### Claim 1: The explicit local derivative law improves the anchored piecewise-warp endpoint

- **Minimal experiment:** Three independent seeds of the derivative-law treatment and the identical augmentation-only control on the frozen supplied-track partial-cache train/development protocol.
- **Primary endpoint:** Piecewise-warp development video-first AvgMAE, averaged across the three seed-paired effects.
- **Decision rule:** Relative AvgMAE reduction \(100(\mathrm{AvgMAE}_{\mathrm{control}}-\mathrm{AvgMAE}_{\mathrm{law}})/\mathrm{AvgMAE}_{\mathrm{control}}\ge 5\%\) **and** the paired source-component bootstrap 95% confidence interval for \(\mathrm{AvgMAE}_{\mathrm{control}}-\mathrm{AvgMAE}_{\mathrm{law}}\) excludes zero on the positive side.
- **Secondary metrics:** Video-first AvgOBO; supplied-track Period-mAP, AP50, and AP75 on a frozen explicit scale; phase-correspondence error; seed mean and sample standard deviation. None can substitute for a failed primary clause.
- **Clean-to-warp rule:** At most 10% relative degradation supports robustness; 10–20% is a preregistered gray zone that licenses neither success nor failure language and requires reporting without post hoc retuning; more than 20% triggers K5/K7 as applicable.
- **Expected evidence:** Prospective only. No expected direction is recorded as an observed outcome.

### Claim 2: Any gain is specific to the derivative law rather than a nearby method, clock shortcut, or harmonic artifact

- **Minimal experiment:** Run input/capacity/compute-matched PAMS, Track-PAMS, SimPer-style, CycleCL-style, DeepPhase-style, generic time-equivariant, and direct signed-frequency baselines, plus track-global autocorrelation, stationary local windows, global-resampling-only, and derivative-scaling deletion.
- **Shortcut controls:** No-pose/timestamp-only, nuisance-metadata-only, matched-time pose shuffle, frame-order shuffle, unseen resampler/interpolator, camera/amplitude perturbation, reversal, pauses, missing joints, and half/double-frequency probes.
- **Fixed shortcut rule:** For absolute gain \(G=\mathrm{AvgMAE}_{\mathrm{aug}}-\mathrm{AvgMAE}_{\mathrm{law}}>0\), shortcut retention is \(R=(\mathrm{AvgMAE}_{\mathrm{aug}}-\mathrm{AvgMAE}_{\mathrm{shortcut}})/G\), clipped only for display to \([0,1]\). Every shortcut model must retain no more than 20% of the gain; otherwise K4 applies.
- **Control-only comparator:** Masked geometric TSSM ridge completion is reported as a control and cannot replace the phase-flow model.
- **Expected evidence:** The derivative-law treatment must separate from each closest mechanism within the frozen uncertainty rule and pass the representation/shortcut gates. Otherwise the residual claim is killed.

### Claim 3: Identity-private execution does not leak a one-person tempo intervention into untouched identities

- **Minimal experiment:** With the trained backbone frozen, warp exactly one supplied identity while all other track tensors and clocks remain byte-identical. Compare private state with stateless, shared-scene, reset, swap, shuffled-key, and track-length/confidence-only controls.
- **Metric:** Untouched-person per-frame response change, continuous count-mass change, rounded count change, and paired source-component intervals; report the full distribution rather than only scene totals.
- **Role:** Supporting causal diagnostic only. It cannot rescue Claim 1 or become a private-memory novelty claim.
- **Expected evidence:** Prospective only; any cross-ID buffer access is an implementation failure, and a material untouched-person change invalidates the isolation interpretation.

## Experiment Handoff Inputs

- **Must-prove claims:** The frozen 5%+CI primary rule; derivative-law necessity beyond identical augmentation-only and closest priors; shortcut retention no greater than 20%; interpretable signed phase under warp, pause, and reversal; no cross-ID contamination.
- **Must-run ablations:** Derivative scaling removed; global resampling only; direct signed-frequency regression; stateless/shared/reset/swap/shuffled-ID state; track-global tempo; stationary local windows; NOLA grid shift; and deletion of activity/reconstruction or anti-collapse terms only if those terms are included in the frozen prospective configuration.
- **Required comparators:** Identical augmentation-only, PAMS, Track-PAMS, SimPer-style, CycleCL-style, DeepPhase-style, generic time-equivariant learning, direct signed-frequency regression, track-global autocorrelation, stationary local windows, and masked geometric TSSM as control-only.
- **Critical datasets / metrics:** Checksum-frozen canonical-source-disjoint partial-cache MultiRep train/development supplied tracks; video-first AvgMAE/AvgOBO; supplied-track Period-mAP/AP50/AP75 with explicit scale; three seeds; sample standard deviation; paired source-component bootstrap 95% intervals.
- **Highest-risk assumptions:** Phase identifiability under missing/aliased human motion; label-vault enforcement; pose-success selection; repeated-index coverage; matched baseline feasibility; external implementation fidelity for PAMS-family controls; unresolved TWCRAC overlap.
- **Forbidden evidence:** v46/v62/v63 efficacy numbers, sealed test sets, predicted-track language, official MultiRep main-result language, published MultiCounter/MultiCounter+ values as same-protocol rankings, and any number not bound to a complete run receipt.
- **Run order:** Integrity gates → deterministic synthetic tests → one sanity run → identical augmentation-only and derivative-law three-seed pair → deletion/direct-frequency/shortcut gates → closest-prior comparators → identity intervention → novelty recheck. Stop immediately at the applicable K-rule.

## Compute & Timeline Estimate

These are planning estimates, not measured resource claims. Actual throughput is unknown because the canonical implementation does not exist.

### Sanity budget

- **Scope:** Repacker/schema tests, synthetic phase/warp tests, one training-only subset run, treatment/control smoke comparison, and NOLA/state instrumentation.
- **Compute:** Approximately 12–24 GPU-hours on one modern 24 GB GPU-equivalent, plus 8–16 CPU-hours for repacking, hashing, evaluator/bootstrap tests, and receipt generation.
- **Storage:** Approximately 5–15 GB for feature shards, logs, short checkpoints, phase traces, and raw predictions; the current two source pickles are about 26.2 MiB, but run artifacts dominate.
- **Engineering time:** 5–8 working days if the source cache and permissions are available; stop if the label-vault or clock tests fail.

### Full decision-bearing budget

- **Scope:** Three independent seeds for treatment and all decision-bearing learned controls, frozen comparator set, phase/shortcut diagnostics, source-component bootstrap, and complete evidence bundles.
- **Compute:** Approximately 250–450 GPU-hours on modern 24–48 GB GPU-equivalents. The range includes reimplementing and tuning only by the frozen training-only rules; it excludes predicted-track evaluation and any sealed test run.
- **CPU/storage:** Approximately 100–200 CPU-hours for repacking, hashing, metric/bootstrap execution, and audits; approximately 100–250 GB for checkpoints, logs, environment captures, raw per-person predictions, and phase traces across all seeds.
- **Timeline:** 3–5 engineering weeks with one implementer and reliable access to 2–4 GPUs; 1 additional week for audit, comparator fidelity checks, and complete TWCRAC novelty adjudication. Data-cache rebuilding or permissions work can extend this substantially.
- **Data / annotation cost:** No new manual count/period annotations are proposed. Cost is engineering and access-control work. A paper-eligible main result still requires a complete authorized train/development cache and later a separately frozen predicted-track protocol; neither is included in this pilot estimate.

## Proposal Status

- **Phase:** Initial proposal, round 0.
- **Method status:** Specified prospectively; not implemented.
- **Evidence status:** Zero eligible method results.
- **Review status:** Same-family provisional; prior research review verdict remains `REVISE`.
- **Training authorization:** Blocked until physical label-vault, forbidden-key, source isolation, evaluator, permission, checksum, and clock gates pass.
- **Claim authorization:** Blocked until the complete run set passes the relevant K-rules, experiment audit, result-to-claim, and renewed novelty review.
