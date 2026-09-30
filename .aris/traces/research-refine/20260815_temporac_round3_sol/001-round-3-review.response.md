# TempoRAC Round 3 — Senior Method-First Re-evaluation

CALIBRATION: none

No curated known-good/known-bad local research-refine proposals were supplied, so I use the stated top-venue, method-first rubric without a local taste anchor. This is the same GPT-5.6-Sol reviewer thread used for the earlier rounds. The review is same-family and provisional.

## Input and independence note

I re-read `refine-logs/temporac/round-2-refinement.md` from start to EOF. Its SHA-256 is `8427a7adfdb6537f320554f7919e0756a874363282b723df799b17668de6abc1`, exactly the expected frozen hash. I independently re-read the original proposal, the phase-origin audit, the canonical TempoRAC brief and reviews, the novelty record, the canonical legacy JSON review, the four root-level audits/addenda, and the prior raw reviewer response needed to assess closure. I did not treat the writer's change list or a parent summary as evidence.

`idea-stage/RESEARCH_REVIEW.md` remains absent. As previously authorized, `idea-stage/RESEARCH_REVIEW.json` is the canonical legacy-review substitute. This is an input-resolution note, not a blocker.

Reviewer metadata:

- `reviewer_model`: `gpt-5.6-sol`
- `reviewer_family`: `openai`
- `review_independence`: `same-family`
- `acceptance_status`: `provisional`
- `reasoning_effort`: `xhigh`

## Problem Anchor (verbatim)

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

## Anchor and graph ruling

**Full anchor status: VERBATIM PRESERVED.** Both `## Problem Anchor` occurrences in the Round 2 refinement are ordinal-text identical to the complete Round 0 section, including the final drift-guard paragraph. Under the stable extraction “section body after the heading, trailing whitespace removed,” all three bodies have SHA-256 `44dce8546f499acdede797055625657579685c0b75dc64c26d3f51003e956b71`. The bounded Round 2 integrity defect is closed.

**Substantive drift: NONE.** The proposed graph is still the original TempoRAC idea:

1. externally supplied person-indexed pose tracks;
2. identity-private preprocessing, buffers, and reconstruction;
3. one shared causal encoder;
4. three shared, shape-matched slow/medium/fast response experts;
5. one fixed cue-only irregular-clock NUDFT router with identity-local relative-tempo referencing;
6. window-level probability response fusion followed by exact positive NOLA;
7. one fixed threshold-0.5 connected-component decode per supplied identity.

The degree-one teacher is a prerequisite and target emitter, not an inference replacement. WARP-PHASE, learned routing, ACF fusion, PAMS-TCC initialization, a GRU alternative, per-person model copies, independent window decodes, and integral decision decoding remain absent. The long contract does not create extra paper claims, although it still needs a few exactness edits before implementation.

## Scores

| Dimension | Weight | Score | Weighted contribution |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9.4 | 1.410 |
| Method Specificity | 25% | 6.4 | 1.600 |
| Contribution Quality | 25% | 7.8 | 1.950 |
| Frontier Leverage | 15% | 8.2 | 1.230 |
| Feasibility | 10% | 5.8 | 0.580 |
| Validation Focus | 5% | 7.0 | 0.350 |
| Venue Readiness | 5% | 6.5 | 0.325 |

**Weighted composite: 7.445 / 10.**

Exact calculation:

`9.4*0.15 + 6.4*0.25 + 7.8*0.25 + 8.2*0.15 + 5.8*0.10 + 7.0*0.05 + 6.5*0.05 = 7.445`.

## GAP

The proposal is exactly **1.555 points below the READY threshold of 9.000**. Round 2 closes the anchor-integrity defect and most of the earlier seam/peak presentation holes, but the frozen equations and data clock still cannot execute as one graph. The canonical v44 input is a rounded 320-point source-clock grid, whereas `q_gap=1` admits a collapsed natural track only when its unique source centers are consecutive; every v44 video with `source_length>320` deterministically has at least one forbidden gap. F5's claimed subdivision-invariant second moment is mathematically false. F19 simultaneously declares NOLA stop-gradient while using the reconstructed response as a training term. G4 requires both sources to cover event spacings 4 and 129 although certified source-orbit events constrained to periods `[5,128]` cannot realize those strata. These join the still-undefined source-orbit population, control weighting mismatch, fractional landmark ownership, and G5/K7 order defect.

This remains a bounded **single-contract revision**, not a reason to add a module or experiment family: rewrite one canonical acquisition/teacher/training/reconstruction/gate contract so every formula, admissible data row, gradient, and fixture stratum agrees before implementation.

## Resolution of the Round 2 findings

| Round 2 issue | Round 3 status | Evidence-based ruling |
|---|---|---|
| Five anchor fields preserved but drift paragraph omitted | RESOLVED | Both complete anchor sections now exactly match Round 0, including the paragraph. |
| Population tolerance could waive negative edges | RESOLVED | `certify_target` requires every valid edge to satisfy `-1e-12 <= delta_t < 0.25`; any failure abstains the track. |
| Raw-floor seam ownership and unspecified accumulation | RESOLVED | Neumaier float64 accumulation, `S_tau`, no backward crossing, one positive crossing per traversal, and pulse invariance at three tolerances are frozen. |
| Unsupported source gaps could hide cycles | MATHEMATICALLY FAIL-CLOSED BUT DATA-INCOMPATIBLE | `q_gap=1` prevents hidden-cycle inference, but after duplicate collapse it rejects every canonical v44 record with `source_length>320`; the proposal provides no unconditional population receipt showing a usable natural pilot remains. |
| Tangent claimed exact warp invariance | RESOLVED | It is explicitly approximate and admitted only if full phase and pulse equality passes on PCHIP and sinc views. |
| Landmark plateau semantics ambiguous | RESOLVED | Both neighbors must be strictly lower and the parabolic denominator must be finite and negative. |
| Static pooling claimed exact subdivision invariance | NOT RESOLVED | F5's trapezoidal second moment changes under subdivision; for one scalar edge from 0 to 1 it is 0.5 before midpoint subdivision and 0.375 after. |
| G4 did not bind operator fixtures and trained outputs | PARTLY RESOLVED | Both sources are named, but demanding spacings 4 and 129 from certified trained outputs contradicts the frozen `[5,128]` inter-event domain. |
| G4 included unsupported spacings 2 and 3 | RESOLVED | They are excluded; the suite starts at spacing 4 and includes the derived boundary case. |
| Capacity control averaged logits before sigmoid | RESOLVED | All three control branches apply sigmoid and are fused uniformly at probability level. |
| Gate decisions lacked frozen numeric rules | RESOLVED | F24–F29 specify point estimates, deterministic unit-block bootstrap bounds, seed aggregation, and failure semantics. |
| Fused response objective has an executable gradient path | NOT RESOLVED | F19 trains on `R`, but the following sentence declares NOLA arithmetic stop-gradient, which detaches that fused term unless narrowed explicitly. |
| Contribution/evidence sprawl | RESOLVED AT CLAIM LEVEL | The paper has one dominant routing claim, one supporting specialization check, and exactly three visible evidence blocks. |
| Exact control equivalence | STILL PARTIAL | F23 now matches nonlinear placement but uses `v`, whereas F19 uses `v chi`; this changes sample/progress weighting under time warps. |

## Method-first assessment

### 1. Degree-one teacher prerequisite

**Ruling: the degree/crossing certificate is mathematically credible, but the complete teacher prerequisite is not mathematically or data-contract sufficient as written.**

The revised prerequisite is substantially stronger than the previous version. Its state excludes clock, displacement magnitude, fixed-frame lag, length, identity, window, and warp variables. Pose-only strict landmarks establish a canonical origin; each retained traversal must have winding `+1`, 30/32 phase-bin coverage, circular gap at most 0.10, small reconstruction, and no detected collision. Every target-bearing edge has a strict forward increment below a quarter cycle. Neumaier accumulation and tolerance-stable seam snapping give a deterministic one-crossing pulse, and PCHIP/sinc equality binds the approximate direction and static pool to the actual emitted bit vector. Degree-zero, negative, double/higher winding, harmonics, reversal, symmetry, collision, static-code, phase-permutation/noise, and decoder-bypass attacks are explicit. Population tolerances cannot serialize a failing individual target.

This closes the earlier mathematical failure in which net winding `+1` could coexist with multiple positive crossings after backtracking. With strict nonnegative increments and total traversal degree `+1`, a retained traversal cannot contain two positive integer crossings. The pulse target would therefore be identifiable once its support and edge ownership are made coherent.

Two upstream assumptions currently prevent that conclusion from binding to the implemented teacher:

1. **F5 is not subdivision invariant.** For one scalar channel on a unit-length linear edge with endpoint values 0 and 1, the displayed `q` contribution is `1/2`. Subdividing the same line once at value `1/2` changes the displayed contribution to `3/8`. The exact squared-linear line integral is `1/3`. Thus the static code changes solely with sampling density, contradicting the sentence immediately after F5. Replace the second-moment edge term by `lambda_e/3 * (a^2 + a*b + b^2)` for endpoint values `a,b`, or delete the invariance claim and rely on an explicitly non-invariant approximation. Because the teacher is supposed to be clock-blind and source-orbit training is resampled, the exact line integral is the coherent minimal repair.
2. **The natural acquisition contract is incompatible with the canonical fixed-320 v44 clock except on a narrow subset.** The audited converter emits `q_j=round(j(L-1)/319)`. After duplicate collapse, `q_gap=1` accepts all unique steps only for `L<=320`; for every `L>320`, at least one step is greater than one. For example, `L=321` has one nonunit gap, `L=500` has 180, and `L=1000` has 319. The audit reports 45/110 short training videos but 0/51 short validation videos, so validation eligibility requires the unreported special case `L=320` plus complete pose support. Calling the denominator “feature eligible” cannot establish a usable supplied-track pilot. A count-blind G0 receipt must enumerate unconditional acquisition survival before any teacher or training job.

The remaining discrete exactness defect is ownership around the sub-frame landmark `t+xi_t`. Natural traversals are “successive retained-landmark intervals,” each complete traversal gets one landmark vote, leading/trailing arcs are masked, topology includes a closing edge, and F11 emits edge bits. The contract never fixes:

- whether traversal `j` is `[L_j,L_{j+1})`, `(L_j,L_{j+1}]`, or another convention;
- whether its origin vote is the starting or ending retained landmark;
- which integer edge owns a fractional endpoint or a landmark lying exactly at a sample;
- whether that boundary edge participates in one or both traversal topology checks; and
- how the resulting half-open traversal support maps to `chi`, the pulse bit, and the receipt.

Different reasonable implementations can therefore disagree by one target edge while all displayed formulas appear satisfied. The smallest repair is one half-open interval and edge-ownership convention, used identically by F5, F9–F11, `chi`, resampler pullback, and the serialized receipt. No new teacher or decoder is needed.

`P_min=5` should remain an explicit supported acquisition assumption, but `d/P_min<0.25` need not be the acquisition gap predicate. The no-hidden-complete-cycle condition is `d<P_min`; the separate learned-phase certificate can retain `delta<0.25` and abstain on faster admitted arcs. With `P_min=5`, this permits source gaps up to 4 without claiming a phase advance, then rejects a track if its actual certified increment is too large. G0 must still show that this fixed rule yields adequate unconditional v44 identity/component support. A K7 mismatch must kill the route rather than be repaired with labels.

### 2. One-event pulse and one-final-peak pass

**Ruling: conditionally defensible; preserve it and do not substitute an integral decoder.**

The target is now a fixed binary sigmoid pulse. The response loss creates explicit positive and negative margins. NOLA uses a strictly positive taper, an exact positive denominator, no epsilon path, and one reconstructed response. The only decision is one connected-component count at threshold 0.5 per supplied valid identity. There is no smoothing, NMS, minimum-distance rule, learned threshold, period vote, window sum, or integral fallback.

G4 is now the right kind of identifiability test in structure. It binds deterministic operator fixtures and frozen outputs of all three selected canonical seed banks under one association graph. Every event/component degree diagnoses miss, split, extra, or merge without a matching heuristic. Offsets, boundaries, tail windows, realizable spacings, PCHIP, and sinc should have zero component errors plus positive hard margins. F28 directly tests that NOLA reconstructs correctly where independent window decoding fails.

The current quantifier makes the gate impossible, however. It says **both** sources cover spacings `4,...,129`. A certified source orbit has true inter-event duration in `[5,128]` output-clock units. Under one frozen half-open event-edge convention, its trained outputs therefore cannot supply spacing 4 or 129. Those can remain operator-only off-boundary stress fixtures, but the trained-output source must be required only on physically realizable certified spacings 5 through 128. A gate cannot require an empty trained stratum to have zero errors and positive margins.

After that source-specific stratum correction, the defense remains conditional on the source-orbit generator being concretely frozen, the fractional landmark/edge convention being unique, and NOLA retaining its response gradient while resetting at gaps. If G4 or natural K7 fails, the stated failure action is correct: reject this strict route. Replacing it afterward with integral response mass would change the decision object and violate the anchor.

### 3. Cue-only routing, matched experts, and causal support

**Ruling: the same-response intervention supports the original causal claim, but the separately trained capacity control is still not exactly matched.**

The main causal comparison is well designed. Each canonical seed bank is trained once; slow/medium/fast sigmoid response tensors are cached once; local, global, uniform, blocked, and within-track-shuffled arms change only the fixed gates. The encoder and experts do not receive the cue. Blocked and uniform must be bit-identical and are correctly treated as one path-integrity assertion rather than independent evidence. Thus a local-versus-strongest-nonlocal gain on the same responses identifies the effect of local cue-based selection, not retraining, model capacity, or a latent cue bypass.

F25's `3x3` diagonal matrix is appropriate supporting mechanism evidence: all three equal-shape branches must specialize to their intended relative-tempo strata on unseen resamplers, with minimum branch mass and simultaneous margins. It supports rather than replaces the dominant routing claim.

F23 fixes the prior nonlinear-placement error by applying branch sigmoids before uniform probability fusion. It does not yet match the canonical loss measure. F19 trains branch and fused terms with `v chi pi_k` and `v chi`; F23 trains all control branches and the fused control with `v` only. Even if units and traversals are averaged later, `chi` changes within-traversal physical-progress weighting under nonuniform resampling. The control can therefore differ because it weights dense/slow samples differently, not solely because it is unrouted. Use `v chi` in every control branch and fused term, omitting only `pi_k` so every control branch still receives the full unstratified target. Keep the current probability-level uniform fusion, graph, seeds, optimizer, and checkpoint rule.

This control defect does not invalidate the local/global/uniform/shuffled same-response intervention, but it blocks the proposal's stronger “exact capacity-matched control” and K7 redundancy claim until fixed.

F19 also needs one explicit autograd contract. It includes `P(R,e*;v chi)`, then declares “NOLA arithmetic” stop-gradient. Read literally, this detaches `R=N/D` and makes the fused-response term contribute zero gradient to the encoder and branches. The intended minimal graph is clear: cue, gates, taper, masks, and denominator are constants, but gradients pass through each sigmoid branch response and the NOLA numerator/division, with `partial R/partial r_{ell k}=a*g_k/D` on valid edges. Freeze that path and add a fused-term-only gradient receipt showing finite nonzero branch/encoder gradients and exact zero router/cue/taper gradients. This is a graph test, not a new loss or claim.

### 4. Exact implementation and gate order

**Ruling: the named neural modules are concrete, but the end-to-end training/evaluation graph is not implementation-ready because several frozen statements are mutually inconsistent.**

The architecture itself is unusually concrete: input channels and masks, widths, kernels, dilations, branch ordering, edge alignment to sample `t+1`, NUDFT grid and reliability, reference construction, sigmoid-level fusion, NOLA, decoder, optimizer, seeds, precision, checkpoint precedence, atomic units, bootstrap seed derivation, and most thresholds are specified.

The next refinement should close one canonical execution contract containing these bounded defects:

1. **Natural clock/admissibility.** Replace the `d/P_min<0.25` acquisition predicate with a no-hidden-cycle gap bound coherent with the fixed-320 source centers and retain the per-edge `delta<0.25` certificate separately. Before any teacher work, G0 must emit counts and source-component coverage over every supplied valid identity for duplicate collapse, gap support, frame/joint support, and final acquisition eligibility. If the predeclared unconditional floor fails, the present v44 route is infeasible and must stop.
2. **Static-pool mathematics and fractional ownership.** Use an actually subdivision-invariant second-moment line integral and one half-open traversal/sample/edge ownership table in `certify_target` and every dependent formula/receipt.
3. **Training gradients and matched control.** Preserve gradients from the fused F19 term through branch probabilities and NOLA while stopping only fixed auxiliaries; use `v chi` in F23. One autograd receipt must prove fused-only nonzero branch/encoder gradients and zero router/cue/taper gradients.
4. **Segment reset and NOLA.** “Never cross a split” is not an executable reset rule. Schedule windows independently inside every maximal valid run; zero-pad each run independently; reset causal convolution context, gate reference membership, numerator, denominator, and connected-component adjacency at the run boundary; concatenate invalid zero edges only after reconstruction so the decoder is still called once per identity. Fixtures must perturb a pre-gap response without changing any post-gap byte.
5. **Source-orbit and fixture domains.** Freeze the finite `X0` generator population and deterministic splits. Give operator and trained-output G4 sources separate admissible spacing sets: operator fixtures may attack 4/129 boundary behavior, but certified trained outputs cover only realizable 5–128 spacings. A required empty stratum fails the schema rather than being reported as a pass.
6. **G5/K7 order.** S4 says execute G5 and then K7, but G5 already requires “all K7 conditions pass.” G5 should verify frozen prediction/evaluator/configuration hashes, prior-gate hashes, source-component bootstrap samples, and one-way vault opening; K7 should then be the sole natural scientific decision.

“Feature-eligible” must be defined by the count-blind G0 predicates above. Freeze and report two quantities before opening the vault: unconditional acquisition eligibility over all supplied valid identities/components, and certificate coverage conditional on that immutable population. The former needs its own predeclared floor. Final AvgMAE remains video-first over all supplied valid identities. Do not tune a threshold after reading counts.

### 5. Dominant claim and evidence focus

**Ruling: focused.** The proposal now contains exactly one dominant claim: cue-only, identity-indexed, window-local relative-tempo routing selects useful shared response scales under within-track drift. Expert specialization is explicitly only a supporting mechanism check. The teacher, NOLA, decoder, firewall, and identity isolation are prerequisite/correctness contracts rather than parallel novelty claims.

The three visible evidence blocks are the right minimum:

1. response-unit plus decoder admissibility;
2. frozen-response local-routing advantage plus supporting specialization;
3. supplied-track efficacy, stationarity, premise coverage, and redundancy.

G/K receipts may remain internal. They should not become additional paper tables or claims. The current method does not need another backbone, router, decoder, dataset family, or experiment block.

### 6. Feasibility and staged compute

**Ruling: the nominal cap is 100 RTX A6000-hours, but feasibility under the canonical data contract is not established.** The stated total is exactly 20–30 focused engineer-days and at most 100 RTX A6000-hours. Staging is appropriately fail-closed in intent: specification first, pack/firewall only after separate authorization, feature-only/synthetic work before vault access, and natural labels opened only after predictions and evaluators freeze. However, the natural v44 clock can make most or all validation identities acquisition-ineligible before model inference. No amount of GPU budgeting repairs that.

G0 must therefore run a count-blind acquisition-support preflight before S2 and hard-stop if the unconditional identity/component floor fails. Only after it passes is compute worth auditing. The per-stage allocation then still needs a job ledger: S3 has 12 A6000-hours for three canonical banks, three controls, and six independently trained shortcut models across three seeds—24 response-family training jobs before fixture evaluation—while S4 reserves 60 hours largely for frozen prediction and evaluation. Freeze source-orbit counts, epoch/step bounds, measured or conservative minutes per job, and peak memory; then rebalance inside the existing 100-hour envelope. Do not increase the cap or add arms.

## Per-dimension rationale

### Problem Fidelity — 9.4/10

The complete anchor is restored verbatim, the graph directly targets asynchronous identity-specific pace drift, state remains private, overlap is reconstructed before one identity-level decode, and forbidden easier replacements are absent. The deduction reflects that the current `q_gap=1` contract may turn the supplied-track natural pilot into systematic zero-output abstention even though the conceptual target is correct.

### Method Specificity — 6.4/10

Many modules are specified precisely, but several “exact” statements conflict: the canonical input clock versus `q_gap=1`, F5 versus subdivision invariance, F19 versus NOLA stop-gradient, the shared G4 spacing quantifier versus certified period bounds, and G5 versus K7 order. The source-orbit generator, segment reset, sub-frame edge ownership, and feature-eligibility denominator are also unfinished.

### Contribution Quality — 7.8/10

There is one sharp causal routing claim and one subordinate specialization check. Same-response interventions are strong. The intended fused training term may currently be detached, exact-control language is premature until F23 uses the same physical-progress measure, and the unresolved closest-prior issue limits novelty confidence separately.

### Frontier Leverage — 8.2/10

The irregular-clock NUDFT cue, fixed causal interventions, deterministic pseudo-target certifier, and strict provenance separation are technically current and appropriate to this signal-learning problem. Foundation models, diffusion, RL, or inference-time search would not solve the mechanism and would add sprawl.

### Feasibility — 5.8/10

The networks are small and the nominal cap is explicit, but the canonical fixed-320 v44 clock conflicts deterministically with `q_gap=1` whenever `source_length>320`; the audited validation subset has no short videos and no receipt that enough `L=320` tracks exist. The synthetic population is undefined, and the 24-job S3 workload has no runtime ledger inside its 12-hour allocation.

### Validation Focus — 7.0/10

Exactly three paper-visible blocks cover premise/decoder, routing mechanism, and natural efficacy, which is appropriately focused. The score is capped because G4 currently requires impossible trained-output spacing strata, G5/K7 is circular, and conditional feature eligibility can mask acquisition failure. The internal attack battery should remain compressed in the paper.

### Venue Readiness — 6.5/10

If one coherent execution contract closes and the frozen pilot passes, the paper can be sharp. At present the data clock, teacher statistic, gradient path, fixture domain, and gate order disagree, so implementation would require reviewer-significant choices. The closest-prior/TWCRAC novelty question also remains unresolved under separate novelty review.

## Dimensions below 7: weaknesses, fixes, priorities

### Method Specificity — 6.4 — CRITICAL

- **Specific weakness:** The frozen proposal contains mutually inconsistent executable statements: fixed-320 v44 source centers versus `q_gap=1`; a non-invariant F5 second moment versus an invariance claim; an F19 fused term versus stop-gradient NOLA; `[5,128]` certified periods versus trained G4 spacings 4/129; and G5 requiring K7 before K7 runs. Several ownership/reset/generator interfaces remain deferred.
- **Concrete fix:** Replace these with one versioned execution contract in the proposal: count-blind natural clock/admissibility receipt, exact squared-linear F5 integral, half-open landmark/edge ownership, segment-local reset, differentiable response path through NOLA, `v chi` control weights, source-specific G4 strata, finite generator manifest, and acyclic G/K order.

### Feasibility — 5.8 — CRITICAL

- **Specific weakness:** With the current canonical v44 input, every `source_length>320` record has a nonunit source-center gap and therefore abstains under `q_gap=1`; the validation audit provides no evidence of an adequate `L=320` population. Synthetic workload size and the 24-job S3 runtime are also unknown.
- **Concrete fix:** Run no training until a label-blind G0 preflight proves a fixed unconditional acquisition-eligible identity/component floor under the repaired no-hidden-cycle rule. Then bind generator size and a measured/conservative job ledger and rebalance within, never above, 100 A6000-hours.

### Venue Readiness — 6.5 — IMPORTANT

- **Specific weakness:** A top-venue mechanism cannot be claimed implementation-ready while its data population, teacher statistic, gradient path, matched control, fixture quantifiers, and final gate order can produce different programs. Novelty/TWCRAC also remains separately unresolved.
- **Concrete fix:** Complete one bounded proposal revision and re-review it without running data or adding components. Keep novelty clearance, title, server/training, result, and paper authorization in their separate workflows.

## Simplification Opportunities

1. Make G0–G5 artifact/provenance gates and K0–K7 scientific decisions. In particular, remove K7 success from G5; avoid duplicating the same threshold as both a build pass and a later kill decision.
2. Express F23 by reusing the canonical `P(.; v chi)` path and replacing only the branch responsibility with the full target. This deletes a second weighting implementation and makes the control truly matched.
3. Put acquisition gaps, the exact F5 line integral, landmark/seam edge ownership, segment resets, `chi`, resampler pullback, and source-specific G4 strata in one execution table. Delete contradictory prose rather than adding another gate.

## Modernization Opportunities

**NONE.** No LLM, VLM, diffusion, RL, distillation, or inference-time search primitive is a natural remedy for orbit degree, irregular-clock tempo selection, private per-identity state, overlap reconstruction, or one-event decoding. Adding one would weaken parsimony.

## Drift Warning

**NONE.** The complete immutable anchor is restored and the original graph is preserved. The following would be drift in the next revision:

- replacing the local three-expert router with WARP-PHASE or a global selector;
- using human count/period/cycle fields to create, select, or repair targets or checkpoints;
- adding learned routing, per-person models, cross-identity state, or independent window decisions;
- rescuing a failed peak gate with integral mass; or
- adding another paper claim or experiment block instead of closing the four contracts above.

## Remaining action items

All technical items below are sections of **one bounded Round 3 execution-contract revision**, not separate routes.

1. **CRITICAL — Repair natural admissibility first.** Separate the no-hidden-cycle gap bound from the per-edge quarter-cycle certificate; freeze the resulting rule against the audited 320-grid clock; and require a pre-vault, count-blind G0 floor over all supplied valid identities and source components. Stop the route if the canonical v44 population cannot meet it.
2. **CRITICAL — Correct teacher mathematics and indexing.** Replace F5's second moment with the exact squared-linear arc integral and freeze one half-open fractional landmark/traversal/sample/edge ownership convention across topology, pulses, `chi`, and resampler pullback.
3. **CRITICAL — Freeze the actual differentiable graph.** Preserve F19 fused-response gradients through branch sigmoid probabilities and segment-local NOLA; stop only fixed auxiliaries; reset every context and accumulator at valid-run boundaries; and change F23 weights from `v` to `v chi`, omitting only `pi_k`. Bind forward, gradient, and gap-isolation receipts.
4. **CRITICAL — Make generators, fixtures, gates, and budget mutually executable.** Specify the finite `X0` population/splits; use source-specific G4 spacing sets (operator-only boundary attacks, trained outputs only on realizable 5–128 spacings); make G5 an artifact-freeze gate before K7; define conditional and unconditional coverage; and bind a 24-job-plus-evaluation ledger within the existing 100-hour cap. Add no seeds, arms, modules, datasets, or paper evidence blocks.
5. **IMPORTANT — Keep authorization boundaries separate.** This review does not clear novelty, resolve TWCRAC, freeze a title/acronym, authorize implementation/tests, permit server or data-bearing work, authorize test/sealed/heldout/historical access, validate a positive result, or authorize paper drafting/submission.

## Authorization status

- Method refinement verdict: `REVISE`
- Next bounded proposal edit: `NOT_STARTED_BY_THIS_REVIEW`
- Novelty clearance: `NOT_GRANTED`
- Closest-prior/TWCRAC resolution: `UNRESOLVED`
- Title or acronym freeze: `NOT_AUTHORIZED`
- Implementation or test execution: `NOT_AUTHORIZED_BY_THIS_REVIEW`
- Server or data-bearing training: `NOT_AUTHORIZED`
- Test/sealed/heldout/historical result access: `NOT_AUTHORIZED`
- Positive performance claims: `NOT_AUTHORIZED`
- Paper drafting, submission, or public claim: `NOT_AUTHORIZED`

## Verdict

**REVISE**

The Round 2 refinement preserves the entire Problem Anchor and the exact TempoRAC graph and presents one focused causal routing claim with three sufficient evidence blocks. It is not READY because the weighted composite is `7.445 < 9.000` and the purported exact graph is internally inconsistent: the v44 clock conflicts with `q_gap=1`, F5 is not subdivision invariant, the fused F19 term may be detached by stop-gradient NOLA, trained G4 outputs cannot populate spacings 4/129, and the remaining generator/control/ownership/reset/G5 contracts are incomplete. Fix them once inside one versioned execution contract and the existing 100-hour cap. Do not add a module, expand the route, or replace the one-final-peak decoder.
