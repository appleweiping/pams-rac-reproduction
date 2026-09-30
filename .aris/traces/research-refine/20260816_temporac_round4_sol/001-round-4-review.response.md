# TempoRAC Round 4 — Senior Method-First Re-evaluation

CALIBRATION: none

No curated known-good/known-bad local research-refine proposals were supplied, so I use the stated top-venue, method-first rubric without a local taste anchor. This is the same GPT-5.6-Sol reviewer thread used for the earlier TempoRAC rounds. The review is same-family and provisional.

## Input and independence note

I re-read `refine-logs/temporac/round-3-refinement.md` from start to EOF. Its SHA-256 is `06c323a8e0713ffe2d586c26e83331480f0d5f3f8852606108f7b4790c9de707`, exactly the required frozen hash. I independently re-read the original proposal, the phase-origin audit, the canonical TempoRAC brief and reviews, the novelty record, the canonical legacy JSON review, the root-level WARP-PHASE audit, both pilot schema audits and their structured copies, and the prior raw reviewer response needed only to assess closure. I did not treat the writer's change list or a parent summary as evidence.

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

**Full anchor status: VERBATIM PRESERVED.** Both `## Problem Anchor` occurrences in the Round 3 refinement are ordinal-text identical to the complete Round 0 section, including the drift-guard paragraph. Under the stable extraction “section body after the heading through immediately before the next level-2 heading, trailing whitespace removed,” all three bodies have SHA-256 `44dce8546f499acdede797055625657579685c0b75dc64c26d3f51003e956b71`.

**Substantive drift: NONE.** The exact original TempoRAC graph is still present:

1. externally supplied person-indexed pose tracks;
2. identity-private preprocessing, causal context, NOLA state, and decoding;
3. one encoder shared across identities;
4. three shared, shape- and parameter-count-matched slow/medium/fast causal response branches;
5. one detached cue-only irregular-clock NUDFT router referenced locally within each identity run;
6. probability-level soft response fusion followed by exact positive NOLA; and
7. one fixed threshold-0.5 connected-component decode per identity.

The degree-one teacher remains a training prerequisite and target emitter, not an inference replacement. WARP-PHASE, a learned router, ACF fusion, PAMS-TCC initialization, a GRU alternative, an integral decoder, independent window counting, a backup pivot, scene-level counting, and per-person copies of a complete model remain absent.

## Scores

| Dimension | Weight | Score | Weighted contribution |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9.7 | 1.455 |
| Method Specificity | 25% | 6.8 | 1.700 |
| Contribution Quality | 25% | 8.6 | 2.150 |
| Frontier Leverage | 15% | 8.7 | 1.305 |
| Feasibility | 10% | 6.3 | 0.630 |
| Validation Focus | 5% | 6.8 | 0.340 |
| Venue Readiness | 5% | 6.8 | 0.340 |

**Weighted composite: 7.920 / 10.**

Exact calculation:

`9.7*0.15 + 6.8*0.25 + 8.6*0.25 + 8.7*0.15 + 6.3*0.10 + 6.8*0.05 + 6.8*0.05 = 7.920`.

## GAP

The proposal is exactly **1.080 points below the READY threshold of 9.000**. Round 3 successfully closes the earlier v44-gap, static-moment, half-open ownership, NOLA-gradient, control-weighting, reset, G4 source-domain, G5/K7-order, finite-population, and nominal GPU-ledger defects. The remaining gap is narrower but blocking: the frozen X0 construction does not generate the event/certificate population that its own gates quantify; the natural evaluator never checks that the pose-defined primitive is the human count unit; the anchor's cross-person isolation condition has no counterfactual gate; and the loader/training runtime still admits more than one implementation. These are correctness repairs to one route, not grounds for a new model or a larger experiment program.

## Resolution of the Round 3 findings

| Round 3 issue | Round 4 status | Evidence-based ruling |
|---|---|---|
| Fixed-320 v44 clock versus `q_gap=1` | RESOLVED FOR UNWARPED ACQUISITION | The new rule `1 <= delta-q < P_min=5`, whole-identity failure, exact v44 `L<=1277` arithmetic, and unconditional split floors are coherent and count-blind. |
| False subdivision-invariant second moment | RESOLVED | F4 is the exact squared-linear arc integral `lambda*(A^2+AB+B^2)/3`, with compensated sums and a subdivision fixture. |
| Fractional landmark/edge ownership | RESOLVED | One half-open convention now governs traversal, edge, seam, pulse, `chi`, pullback, and serialization. |
| Fused response detached by NOLA | RESOLVED | Gradients explicitly traverse every sigmoid, fusion numerator, NOLA numerator, and exact division; only fixed auxiliaries are detached, and G2 binds a fused-only receipt. |
| Capacity control used the wrong measure | RESOLVED | Every control branch and fused control uses the identical `v*chi` measure and differs only by omitting `pi_k`. |
| Run reset and gap leakage were unspecified | RESOLVED | Encoder/branch buffers, cue/reference membership, NOLA accumulators, window lists, spectral buffers, and adjacency reset at each run; a pre-gap byte perturbation must leave every post-gap byte unchanged. |
| One G4 quantifier demanded impossible 4/129 trained strata | RESOLVED | Spacings 4 and 129 are operator-only; trained outputs use exactly the seven realizable 5–128 strata. |
| G5 required K7 before K7 ran | RESOLVED | G5 is artifact/vault-opening verification only, and K7 is the sole subsequent natural scientific decision. |
| X0 population and compute ledger were deferred | PARTLY RESOLVED | Forty fixed source orbits, splits, resamplers, 24 response jobs, and a 93-hour A6000 ledger are explicit, but the X0 phase/landmark/offset geometry and job completion rule are internally inconsistent. |

## Method-first assessment

### 1. Degree-one teacher and primitive-unit identifiability

**Ruling: the per-traversal degree-one certificate is credible, but the frozen X0 witness and the natural human-unit check are not closed.**

The teacher itself is much stronger than in the earlier rounds. Its phase path is clock-blind and memoryless; the reconstruction decoder has no raw-state, time, mask, or residual bypass; its static code uses exact traversal-arc moments; the landmark is fixed independently of the teacher; every certified traversal has complete support, monotone principal increments below a quarter cycle, dense and observed phase coverage, winding `+1`, exactly one owned positive seam crossing, reconstruction and collision bounds, a concentrated landmark origin, and three-tolerance pulse invariance. PCHIP/sinc matched-state and pulse equality correctly treat the approximate direction feature as something to test rather than assume. With these conditions, one certified traversal has one binary event and cannot be rescued by an integral decoder.

The synthetic construction used to establish that prerequisite is nevertheless contradictory in three deterministic ways.

First, F7 aligns the generator's fundamental displacement with `w=H^T rho/||H^T rho||` as `0.12*w*sin(2*pi*phi)`, while all higher directions are orthogonal to `w`. Therefore the frozen landmark score is exactly a constant plus a positive multiple of `sin(2*pi*phi)` and its unique maximum is at `phi=1/4 mod 1`. Section 6.2 instead defines `z-star=exp(2*pi*i*psi/32)` and `e-star` at integer `psi/32`, whose seam is `phi=0 mod 1`. The teacher's canonical origin is the landmark phase, so its certified seam is the former, not the latter. Consequently the analytic pulse, the statement that the first complete seam is at target edge `o`, and the certified teacher pulse are offset by one quarter traversal. `e-star` is then never bound to the target ledger elsewhere in the contract.

Second, a clean domain `phi in [0,10]` contains exactly ten internal landmark maxima, at clean coordinates `8,40,...,296`. Because run endpoints cannot be landmarks and certified traversals are intervals between consecutive retained landmarks, those ten landmarks provide exactly nine certified traversals, not the “ten certified traversals at every offset” required by Section 9. This is a cardinality contradiction, independent of optimization.

Third, the all-offset construction cannot use the declared clean knot extension. For a stationary duration `d=5` and offset `o=31`, replacing `q` by `q-o` and adding 32 makes the first clean query `32 + 32*(-31/5) = -166.4`, outside the frozen PCHIP/sinc knot domain `[-32,352]`. For `d=8`, it is `-92`, also outside. Thus the spacing-5 and spacing-8 trained strata cannot exist at all 32 offsets under the stated `extrapolate=false` rule. Sinc support makes the required extension larger still.

These are not reasons to reject the degree-one route. They require one aligned truth convention: put the analytic phase seam at the frozen landmark (or change the generator fundamental so the landmark lies at analytic phase zero), make the clean/context domain contain the declared number of internal landmark-to-landmark traversals, extend the discrete clean knots far enough for the exact worst offset and sinc taps, and bind the analytic pulse bit vector to `certify_target` output explicitly.

The proposal also overstates the phase gauge at G1. Landmark canonicalization removes a constant rotation; a general orientation-preserving circle homeomorphism fixing the landmark can change all non-seam phase values while preserving degree, ordering, and the one pulse. The useful invariant is the canonical seam and pulse bit vector, not equality of the full phase array under every homeomorphism. Freeze a finite hashed topology-attack bank and state which fields must remain equal for each attack. At present degree/harmonic/reversal/bypass/collision “attacks” are named but not generated, parameterized, or serialized, so G1 cannot be implemented from `temporac.execution.v3` alone.

Finally, synthetic topology does not establish the human repetition convention. K7 can pass when local routing is five percent better than a nonlocal comparator even if both learn a common off-by-one or integer-multiple teacher unit and both have poor absolute count error. The phase-origin audit explicitly separates a pose-defined primitive seam from an annotator's repetition unit. The smallest non-training repair is to freeze each certified development identity's teacher target count before vault opening and make exact teacher-count versus `count_gt` agreement a kill-only K7 prerequisite on the already frozen certified population. Human labels still do not enter an objective, checkpoint, target, or rerun; they only falsify the primitive-unit assumption after all artifacts are immutable. No teacher-only model job or new paper claim is needed.

### 2. One-event pulse, NOLA, and the one-final-peak pass

**Ruling: preserve the peak decoder; it is defensible once the X0 truth and offset bank are repaired.**

The response objective now has fixed binary targets, balanced positive/negative normalization, and explicit positive/valley margins. Branch probabilities are fused before exact differentiable NOLA. The taper is strictly positive, every valid edge has a proved denominator of at least `1e-3`, there is no epsilon or fallback, and every state and accumulator resets at a run boundary. The only decision remains one threshold-0.5 connected-component call per identity, with components forbidden from crossing runs. The operator association graph detects misses, extras, splits, and merges without a matching heuristic. Operator-only boundary spacings and physically certified trained spacings are finally separated correctly.

Nothing here supports silently replacing the decoder with an integral. If the aligned X0 bank, G4, K5, K6, or natural pilot fails, the stated route should fail. The current blocker is upstream: G4's trained source cannot be instantiated with the specified event seam, traversal count, and offset knots, so its otherwise strong zero-error/margin quantifier has no valid input bank.

### 3. Cue-only routing and matched experts

**Ruling: the routing intervention is causally strong and the matched-control defect from Round 3 is closed.**

The canonical local, global, uniform, blocked, and within-run-shuffled modes reuse the same frozen encoder and the same three cached branch responses. The cue cannot enter the encoder or branches. Only the detached gates change, blocked must be bit-identical to uniform, and the global gate is constant inside a run. This isolates the benefit of local cue-conditioned response selection. The separately trained capacity control has the same branch nonlinearities, widths, dilations, optimizer, checkpoint rule, `v*chi` measure, and probability-level uniform fusion. K3 requires a positive local advantage with a paired lower bound, at least 80% removal under cue shuffle, and less than 10% retained gain for each frozen shortcut. K4 requires a per-seed and aggregate diagonal advantage for all three tempo columns plus nontrivial responsibility. Expert specialization remains supporting evidence rather than a second headline claim.

One anchor condition is still untested: cross-person state isolation. A batch-permutation check is not sufficient because a symmetric cross-identity aggregation can be permutation-equivariant while still coupling identities. G2 needs one deterministic two-identity counterfactual fixture: perturb every allowed feature byte of identity A while holding B byte-identical, then require B's normalized features, cue/reference, branch logits, NOLA numerator/denominator, response, components, and prediction bytes to remain identical. Key/order swapping should only permute the two complete outputs. This is a correctness receipt, not a new arm, training job, evidence block, or contribution.

### 4. v44 acquisition, data firewall, and natural drift

**Ruling: the base acquisition and firewall logic are coherent and fail-closed; the derived drift population needs one additional count-blind support check.**

The unique v44 source hashes, seven-field allowlist, confidence rule, duplicate-collapse conflict policy, `1 <= delta-q < 5` support condition, `L<=1277` arithmetic, whole-identity exclusion, frozen 268/134 identity and 18/9 component denominators, and exact 215/15/108/8 G0 floors form an executable count-blind acquisition decision in principle. Conditional certificate coverage cannot alter that population. The feature, audit, and vault roots are physically separated; the training process has neither a vault path nor an evaluator reader; natural predictions predate vault opening; and G5 only authorizes a one-way evaluator join. No label leakage path is introduced by the proposal.

The deterministic natural drift transform can, however, leave the registered support domain. Its largest target-to-original slope is `27/19`. An admitted raw edge may have `delta-q=4`, so the transformed adjacent source-coordinate displacement can be `4*(27/19)=108/19`, approximately `5.684`, which is not below `P_min=5`. K7 currently checks only whether an unwarped ground-truth period is below five; it does not check the transformed sampling premise. Before vault opening, G5 can compute the mapped adjacent-coordinate spans without any label. Require every mapped span to stay strictly below `P_min`, globally fail otherwise, or reduce the predeclared slope profile so this follows from G0. Do not filter identities or repair the transform after seeing counts.

The same section calls the original natural tracks a “stationary population,” but no feature-only or vault-free test establishes stationary tempo. They are an **unwarped/clean paired population**. Keep that natural retention check, and reserve “stationary” for the genuinely constant-duration X0 population unless a separate predeclared stationary definition is supplied. This is a wording/estimand correction, not an extra experiment.

### 5. Interfaces, numerical execution, gate order, and compute

**Ruling: gate precedence is acyclic and the arithmetic ledger is within budget, but the exact implementation surface is not yet singular.**

The order

`S0 -> G0/K0 -> G1/K1/G2/K2 -> G3/K3/K4 -> G4/K5/K6 -> freeze natural artifacts -> G5 -> K7`

is acyclic. G5 contains no K7 metric, all natural model outputs are frozen before labels, no later gate changes an earlier population or checkpoint, and all empty/zero/nonfinite cases fail closed. This is a substantial closure.

The GPU arithmetic is also correct: three teacher jobs at six hours are 18 A6000h; 24 response jobs at 2.5 hours are 60; heldout inference is 3; natural prediction is 12; total 93, leaving seven unallocated under the 100-hour ceiling. G0 and K1 occur before the 60-hour response block, so the staging is appropriately kill-oriented rather than compute-maximal.

Three exact implementation ambiguities remain:

1. Section 3 says the model loader accepts one seven-member NPZ with source shapes `[P,320,17,3]`, `[P]`, and `[P,320]`; Section 13 defines a per-identity record with shapes `[320,17,3]`, `[1]`, and `[320]`, plus `feature_shard_sha256` metadata. Freeze one archive granularity, exact member list, and exact shapes. Likewise, `certify_target` returns an ordered abstention-reason list while `TempoRACPredictionV3` stores one `uint16[1]`; either serialize the full ordered vector or freeze first-failure precedence.
2. The job caps do not define successful completion. “At most 200/100 epochs and 20,000/10,000 steps” plus “reaching any epoch or step cap is failure” supplies no exact normal termination or early-stopping rule, while five-percent warmup and cosine decay require a known total-step horizon. Freeze an exact scheduled epoch/step count and select the earliest best tune epoch among completed checkpoints; wall-time/memory overruns may remain failures.
3. X0 pins Python, NumPy, and SciPy, but the neural runtime does not pin framework/CUDA/cuDNN versions, tensor dtype, autocast policy, deterministic algorithm flags, or inference batch policy even though several fixtures demand byte equality. Bind these fields in S0 and receipts or weaken byte equality to a frozen numerical tolerance where exact identity is not implementable.

These are one interface/runtime table, not a request for code, a new backend, more compute, or another validation family.

### 6. Dominant claim, validation focus, and venue shape

**Ruling: the paper thesis is focused, and the internal validation is fail-oriented rather than scientifically sprawling.**

There is one dominant claim: window-local relative-tempo routing improves supplied-track counting under deterministic within-track drift relative to the strongest frozen nonlocal comparator while preserving clean performance. The 3-by-3 expert matrix is the only supporting mechanism check. Topology, NOLA, firewall, identity isolation, shortcut resistance, and artifact gates are prerequisites or audit evidence. The proposal still exposes exactly three paper-visible blocks.

The 24 response jobs are numerous, but they comprise three canonical seeds, one genuinely capacity-matched control per seed, and six frozen shortcut tests per seed; they do not create 24 claims. The validation becomes minimal and sufficient after the missing primitive-unit and identity-isolation checks are folded into existing K7 and G2 receipts and the impossible X0 bank is repaired. No new backbone, dataset family, router, decoder, seed, paper block, or frontier component is justified.

Novelty and title clearance remain separate. The unresolved full-method TWCRAC comparison still caps any future novelty language, but it is not a reason to alter this method graph or this Round 4 refinement verdict.

## Per-dimension rationale

### Problem Fidelity — 9.7/10

The complete anchor is byte-preserved twice, and the original seven-property graph is intact. The teacher remains a prerequisite rather than a substitute, the fixed peak decoder remains the final decision, and no forbidden pivot appears. The small deduction reflects the absence of the anchor-required cross-person counterfactual and natural primitive-unit falsification, not substantive drift.

### Method Specificity — 6.8/10

Most equations and interfaces are unusually precise, and every Round 3 formula defect is repaired. The score remains below seven because F7's landmark phase conflicts with `z-star/e-star`, the ten-cycle population cannot yield ten certified landmark intervals, the offset knots fail deterministically for the fastest strata, topology attacks are only named, and the loader, abstention, training-stop, and neural-runtime contracts still admit different implementations.

### Contribution Quality — 8.6/10

The dominant causal claim is singular, the same-response intervention is clean, the matched capacity control is now genuinely matched, and diagonal specialization is subordinate. The remaining deduction is that a routing gain is not yet conditioned on proof that the teacher's primitive is the evaluator's repetition unit.

### Frontier Leverage — 8.7/10

An irregular-clock spectral cue, deterministic pseudo-target certification, explicit causal routing interventions, differentiable NOLA, and capability-separated evaluation are appropriate contemporary tools for this signal-learning problem. LLM, VLM, diffusion, RL, or inference-time search would not solve the remaining mathematical contracts.

### Feasibility — 6.3/10

The nominal 93-hour ledger fits two A6000 GPUs and the staging can stop before most compute. But the current G1/G4 bank is mathematically unconstructible for stated phases, counts, and offsets, successful job completion is undefined, and strict byte tests lack a neural runtime. Until those are fixed, the budget is arithmetic rather than an executable schedule.

### Validation Focus — 6.8/10

Three paper blocks and fail-closed internal receipts are appropriately focused. The score is below seven because the extensive gate suite omits the two checks directly required by the anchor: a post-freeze human-unit falsification for teacher pulses and a one-person perturbation test for untouched identities. The current “stationary” natural estimand is also unsupported.

### Venue Readiness — 6.8/10

If the one execution contract is repaired and the frozen pilot passes, the thesis could be sharp for ICASSP. At present a top-venue reviewer could reproduce the algebraic X0 contradictions without running a model, and separate novelty/TWCRAC clearance remains unresolved.

## Dimensions below 7: weaknesses, fixes, and priorities

### Method Specificity — 6.8 — CRITICAL

- **Specific weakness:** X0 has three incompatible truth systems: a landmark at phase `1/4`, analytic events at phase `0`, and offset/traversal quantifiers that its finite knot/domain construction cannot instantiate. The topology-attack, NPZ, abstention, stopping, and neural-runtime clauses are also non-unique.
- **Concrete fix:** Freeze one `temporac.execution.v4` table that aligns analytic phase and landmark seam, binds analytic and certified pulse bits, provides enough internal cycles and offset/sinc context, hashes every topology attack, and gives one per-identity archive, failure-reason, numeric-runtime, and fixed-step training contract.

### Feasibility — 6.3 — CRITICAL

- **Specific weakness:** The fastest two all-offset trained strata query outside the declared knot domain; ten clean cycles produce only nine certified landmark intervals; and no successful completion point exists inside the job caps.
- **Concrete fix:** Repair the X0 domain cardinality and worst-case offset bounds analytically before any job, then freeze exact scheduled steps/epochs and use wall-time, memory, and total A6000h solely as upper failure caps. Keep the 93-hour inventory and do not add jobs.

### Validation Focus — 6.8 — CRITICAL

- **Specific weakness:** Relative K7 efficacy can pass with a wrong common teacher unit, and batch permutation can pass with symmetric cross-person coupling. The natural drift transform can also leave the declared support domain, while “stationary” is applied to unverified natural tracks.
- **Concrete fix:** Add a pre-vault two-identity byte-isolation fixture to G2; add a post-freeze, evaluator-only exact teacher-target-count agreement prerequisite to K7; require every mapped natural-drift edge to satisfy the support bound before vault opening; and call the paired natural control “unwarped/clean,” keeping stationary evidence on constant-duration X0.

### Venue Readiness — 6.8 — IMPORTANT

- **Specific weakness:** The current frozen document can produce no conforming G4 bank and still leaves reviewer-significant choices at the loader, training schedule, and unit-validation boundary.
- **Concrete fix:** Perform one bounded contract-only refinement and re-review it. Keep novelty, title, implementation, server, result, and paper authorization in their separate workflows.

## Simplification Opportunities

1. Use one synthetic truth seam. Align `z-star/e-star` to the pose landmark and make `certify_target` consume or verify that bit vector; delete the currently disconnected analytic pulse convention.
2. Fold identity isolation into G2 and teacher-unit agreement into K7 as receipts. Do not add trained arms, jobs, paper tables, or claims.
3. Replace the duplicate source-NPZ/per-identity-record and cap/stopping prose with one archive-and-runtime execution table.

## Modernization Opportunities

**NONE.** Foundation-model or generative components do not address the remaining phase-origin, finite-domain, identity-isolation, loader, or stopping contradictions. The current signal-processing route is already the appropriate modern choice.

## Drift Warning

**NONE.** The immutable Problem Anchor and the substantive TempoRAC graph are preserved. The next refinement would drift if it replaced the fixed three-expert local router, response fusion, NOLA, or one connected-component decode; used evaluator fields in training or selection; added learned routing or an integral fallback; substituted WARP-PHASE or a backup pivot; or created a fourth paper-visible evidence block.

## Remaining action items

All items below are parts of **one bounded Round 4 execution-contract repair**, not new routes.

1. **CRITICAL — Make X0 one coherent certificate population.** Align landmark and analytic phase origins; bind `e-star` to the certified target ledger; supply enough pre/post clean knots for the exact `d=5,o=31` query and sinc taps; make the number of internal landmarks agree with the required certified traversals; and freeze the full topology/homeomorphism attack table with source-specific equality rules.
2. **CRITICAL — Close the two anchor-level falsifiers without new training.** Add the two-identity untouched-output fixture to G2 and a post-freeze exact teacher-target-count versus evaluator-count prerequisite to K7. Any mismatch kills the route; it cannot filter, retune, or select a model.
3. **CRITICAL — Freeze the last executable interfaces.** Choose one per-identity feature NPZ schema and member list; reconcile ordered abstention reasons with the prediction record; bind neural dtype/runtime/determinism; specify exact scheduled epochs/steps, warmup/cosine horizon, and successful completion inside the existing 93-hour inventory.
4. **IMPORTANT — Keep natural stress claims inside the registered domain.** Require every deterministic drift-mapped adjacent coordinate span to remain below `P_min` before vault opening, or reduce the fixed slope profile. Rename the paired natural baseline “unwarped/clean”; use “stationary” only where constant tempo is actually constructed.
5. **IMPORTANT — Keep authorizations separate.** This review grants no novelty or title clearance, implementation/test permission, data/server access, result claim, paper edit, submission, or Git action.

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
- Paper drafting, editing, submission, or public claim: `NOT_AUTHORIZED`

## Verdict

**REVISE**

The Round 3 refinement exactly preserves the entire Problem Anchor and the original TempoRAC graph, repairs the earlier acquisition, static-moment, gradient, control, reset, G4-domain, gate-order, and GPU-accounting defects, and keeps one focused routing claim. It is not READY because the weighted composite is `7.920 < 9.000` and blocking inconsistencies remain: X0's landmark seam, analytic pulse, traversal count, and all-offset knot domain do not define one possible G1/G4 population; the natural pilot lacks an evaluator-only primitive-unit kill and an untouched-identity counterfactual; and the archive/training runtime is not singular. Repair those exact contracts without adding a route, module, arm, seed, dataset, paper block, or integral decoder.
