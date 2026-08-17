# TempoRAC Round 5 - Final Senior Method-First Re-evaluation

CALIBRATION: none

No curated known-good/known-bad local research-refine proposals were supplied, so I use the stated top-venue, method-first rubric without a local taste anchor. This is the same GPT-5.6-Sol reviewer thread used for the earlier TempoRAC rounds. The review is same-family and provisional.

## Input and independence note

I re-read the complete current `refine-logs/temporac/round-4-refinement.md` from byte zero through line 1,457 and EOF. The file is 81,746 bytes and has SHA-256 `ff2f4db61ce7be419b9bc440579a7f18cf0400180de054fde4b3e3c92fae0c90`. Earlier candidate bytes supplied during this same round were superseded before verdict freeze. I discarded those stale versions, re-read the final file in full, and bind every judgment below only to `ff2f4d...`.

I independently re-read the Round-0 proposal, the canonical TempoRAC review brief and research review, the phase-origin audit, the two pilot schema/audit documents, and the Round-4 raw review needed to test closure. I did not treat the writer's change list, the parent summary, or prior scores as evidence. A second mathematical audit also re-read all 1,457 final-hash lines and was bound only to `ff2f4d...`; it found the six previously open equation/domain/index issues closed and no remaining fatal mathematical contradiction. I independently verified those closures in the proposal text before using that result as corroboration.

`idea-stage/RESEARCH_REVIEW.md` remains absent. The available canonical legacy artifact is `idea-stage/RESEARCH_REVIEW.json`; this input-resolution fact is not a blocker. Novelty, title, server, implementation, data, and paper authorization remain separate from this method-refinement judgment.

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

**Full anchor status: VERBATIM PRESERVED.** The current proposal contains two complete `## Problem Anchor` sections. Both are ordinal-text identical to each other and to the complete Round-0 section, including the drift-guard paragraph. Under the stable extraction "text after the heading through immediately before the next level-2 heading, with trailing whitespace removed," all three bodies have SHA-256 `44dce8546f499acdede797055625657579685c0b75dc64c26d3f51003e956b71`.

**Substantive drift: NONE.** The original TempoRAC graph remains exact:

1. externally supplied person-indexed pose tracks;
2. identity-private preprocessing, causal context, NOLA state, and decoding;
3. one encoder shared across identities;
4. three shared, shape- and parameter-count-matched slow/medium/fast causal response branches;
5. one detached cue-only irregular-clock NUDFT router with an identity-local run reference;
6. probability-level soft response fusion followed by exact positive NOLA; and
7. one fixed threshold-0.5 connected-component decoder invocation per identity.

The degree-one teacher is a prerequisite and pseudo-target emitter, not an inference substitute. WARP-PHASE, a learned router, ACF fusion, PAMS-TCC initialization, a GRU alternative, integral decoding, independent window counting, scene-level counting, per-person model copies, an extra dataset/seed, and a backup pivot remain absent.

## Scores

| Dimension | Weight | Score | Weighted contribution |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9.9 | 1.485 |
| Method Specificity | 25% | 9.2 | 2.300 |
| Contribution Quality | 25% | 8.8 | 2.200 |
| Frontier Leverage | 15% | 9.1 | 1.365 |
| Feasibility | 10% | 8.8 | 0.880 |
| Validation Focus | 5% | 9.4 | 0.470 |
| Venue Readiness | 5% | 8.5 | 0.425 |

**Weighted composite: 9.125 / 10.000.**

Exact calculation:

`9.9*0.15 + 9.2*0.25 + 8.8*0.25 + 9.1*0.15 + 8.8*0.10 + 9.4*0.05 + 8.5*0.05 = 9.125`.

**GAP: +0.125 above the 9.000 READY threshold.** There is no substantive drift and no blocking issue. The remaining uncertainty is empirical gate passage and separate novelty/execution authorization, not an undefined method contract.

### Dimension judgments

- **Problem Fidelity - 9.9.** Both full anchors and the exact graph are preserved. All accepted repairs test the anchored mechanism rather than replacing it.
- **Method Specificity - 9.2.** F1-F23 now form one coherent execution contract: coordinates, masks, source populations, teacher targets, causal graph, losses, decoder, arms, metrics, receipts, capability order, runtime, RNG, and termination are fixed. The deduction is for the unavoidable burden of materializing and independently checking a very dense first implementation, not for a missing definition.
- **Contribution Quality - 8.8.** The paper is reduced to one causal routing claim. Expert specialization is explicitly only a supporting mechanism check, and certification/decoder evidence is a validity block. Novelty relative to the unresolved closest prior remains a separate ceiling.
- **Frontier Leverage - 9.1.** A detached irregular-clock spectral cue, response-scale routing, matched causal branches, exact overlap reconstruction, and same-response interventions are well aligned with asynchronous within-track pace variation without introducing a fashionable but irrelevant component.
- **Feasibility - 8.8.** The exact inventory is three teacher plus 24 response jobs and 15 A6000-hours of inference/evaluation, totaling 93 of the 100 allowed A6000-hours with concurrency two. Fixed steps, checkpoints, runtime, deterministic arithmetic, and failure ceilings are executable. Strict certificates may empirically fail, but failure is now an intended outcome rather than an implementation ambiguity.
- **Validation Focus - 9.4.** G0-G5a/G5b and K0-K7 are ordered, kill-only, population-complete, and tightly coupled to the dominant claim. The protocol has only three paper-visible evidence blocks; the larger receipt/fixture surface is audit evidence, not claim sprawl.
- **Venue Readiness - 8.5.** The method specification is ready for a fresh experiment plan and preflight. Actual results, novelty/TWCRAC resolution, four-page compression, and title clearance are still absent and are not granted by this review.

**Dimensions below 7: NONE.** Therefore no below-7 weakness/fix/priority entries are required.

## Closure of the method-first blockers

### Degree-one teacher and primitive-unit identifiability

**Ruling: SUFFICIENT AND EXECUTABLE AS A FAIL-CLOSED PREREQUISITE.**

The teacher is clock-blind and pose-landmark-canonicalized. Its input now contains an explicit two-sided direction constructed on common support from separately normalized incoming/outgoing chords and their normalized sum, with exact masks and `1e-8` failure behavior. The static network, phase network, and no-bypass decoder have fixed shapes and inputs. F11 fixes the complete teacher objective: masked 149-channel Huber reconstruction, analytic clean-phase correspondence, and signed phase-increment penalties, with exact equal-weight reductions over fractions, ten traversals, blocks, and source orbits. There is no evaluator label, natural row, `L_cont`, or PAMS initialization in teacher training.

The X0 witness is now internally coherent. Forty frozen orbits have deterministic PCG64 generation and rejection; `S_stat[n mod 7]` covers the seven stationary durations across every split, while each source also has the six named drift permutations, each with ten traversals. Eleven internal landmarks at `8+32j` create exactly ten complete traversals. F9 maps target seam `B_j` to clean coordinate `8+32j`; F10 places pulses at `B_0...B_9`; half-open right ownership excludes the terminal seam. The query range `[-8,344]` and 63-tap sinc support fit the frozen `[-39,375]` knot bank. Analytic and certified pulse bits, counts, traversal bounds, and ownership must agree.

For natural tracks, the certificate fixes landmark extraction, origin canonicalization, shortest-arc interpolation, observed/dense phase coverage, degree `+1`, sole positive owned seam, reconstruction/collision/origin checks, and three-tolerance stability. The start pulse edge must have strictly positive `Delta_phi` and therefore `chi>0`. The finite topology bank includes accepted homeomorphisms and explicit harmonic, reversal, origin, bypass, collision, and pulse attacks without falsely requiring arbitrary floating phase-array equality.

Most importantly, K7 closes the human primitive-unit bridge after predictions are immutable: on every certified development identity, each frozen teacher pulse start must fall in exactly one raw evaluator interval, every interval must contain exactly one such start, no pulse may lie outside their union, and pulse count must equal both interval-list length and `count_gt`. A single split, merge, off-by-one, integer-multiple, malformed-period, or missing-identity contradiction kills the complete pilot. Thus the synthetic degree-one result is no longer assumed to equal a human repetition; that equality is directly falsified under the vault boundary.

### One final peak pass versus integral decoding

**Ruling: DEFENSIBLE AS WRITTEN; DO NOT REPLACE IT WITH AN INTEGRAL DECODER.**

The response target is a frozen binary one-edge pulse, not an uncalibrated density. F14 now defines full balanced BCE with exact `W+`, `W-`, clamp, positive margin, negative valley, and `v*chi` reductions. The pulse-edge `chi>0` rule makes the positive denominator constructive. F13 keeps autograd connected through probability fusion and positive NOLA while the denominator is exact and bounded below. Quantization occurs before the one threshold-0.5 connected-component invocation. G4/K5 require zero miss/extra/split/merge errors plus positive and negative margins across a complete deterministic operator bank, all heldout physical strata, both resamplers, three seeds, ten traversals, and all 32 offsets. K6 requires analytic/certified/pulled-back pulse equality and identical counts across resamplers.

Those contracts make one final component pass part of the tested mechanism: it verifies that overlap reconstruction preserves one localized event. Replacing it silently with integral mass, NMS, fitted periods, or per-window sums would change the estimator and evade the registered split/merge falsifier. Failure of the fixed decoder rejects TempoRAC.

### Cue-only routing, matched experts, and causal support

**Ruling: CAUSALLY ALIGNED WITH THE ORIGINAL CLAIM.**

The irregular-clock NUDFT cue and reference are detached and excluded from encoder and expert inputs. The slow/medium/fast branches are shared, equal-shape, equal-parameter causal response functions. Local, global, uniform, blocked, and shuffled modes reuse the identical frozen cached branch responses; only the routing intervention changes. Blocked must be byte-identical to uniform. This same-response design isolates the effect of local relative-tempo routing more cleanly than separately retrained local/global variants could. The separately trained capacity control matches encoder, branch shapes, total parameters, sigmoid placement, source cycle, optimizer, seeds, checkpoint rule, and probability fusion while removing the gate input.

K3 requires local advantage over the strongest frozen nonlocal/control comparator, a positive paired lower bound, cue-shuffle removal, and bounded shortcut recovery. K4 requires per-seed and aggregate diagonal slow/medium/fast specialization with nontrivial responsibility support. The two-identity untouched-output fixture falsifies cross-person coupling, and pose-shuffle is now an exact time/channel/sign permutation rather than a vague nuisance. This supports one dominant claim - local relative-tempo routing improves drift robustness without material clean loss - while treating diagonal expert specialization only as a supporting mechanism check.

## Interfaces, gates, firewall, and feasibility

**Exact graph/interface status: IMPLEMENTATION-READY AT THE PROPOSAL-CONTRACT LEVEL.**

- G0/K0 bind the authoritative v44 source objects to exactly 268 train and 134 development supplied-valid identities in 18/9 source components, use unconditional floors 215/108 and 15/8, and fail whole identities on clock/schema/support contradictions. The model archive is a new seven-member per-identity NPZ; shard hashes live only in detached receipts.
- G1/K1 bind all three teacher jobs, tune-only selection, every heldout X0 certificate, topology attacks, analytic/certified pulses, resampler agreement, and exact identity/component coverage.
- G2/K2 bind causal receptive fields, run resets, positive differentiable NOLA, fused-only gradients, pre-gap isolation, the two-identity counterfactual, and immutable certified X0/natural target receipts.
- G3/K3/K4 bind exactly 24 fixed-step response jobs, the 93-hour ledger, route/shortcut statistics, strongest-arm reselection, and specialization. K3's PCHIP/sinc population is explicitly `o=0`; all-offset decoder robustness is reserved for G4/K5.
- G4/K5/K6 bind the 10,368-row operator population, source-realizable trained strata, all offsets, batch equality, margins, pulse pullback, and count equality without confusing operator-only spacings with trained outputs.
- G5a freezes complete stub-inclusive predictions, receipt roots, the 9,648-artifact cardinality, mapped-span proof, and canonical teacher target counts before any evaluator capability. A one-invocation grant is consumed by one non-resumable process that performs G5b join integrity and then K7. The K7 receipt binds the exact G5a receipt, G5b receipt, and metric payload. Training cannot regain evaluator rows, and failure cannot trigger selection, retuning, or rerun.

The natural drift map is continuous, endpoint-fixed, count-blind, and fully defined with right-interval ownership, exact internal ties, terminal handling, and no extrapolation; its largest mapped gap is `124/25=4.96<5`. Continuous channels are recomputed from interpolated pose/confidence rather than independently warped. The clean/drift pair uses the same frozen population and predictions. Video grouping is the manifest equality class of the opaque key, people are averaged within video, videos equally, and seeds equally. Component bootstrap draws preserve video weighting and dependence, reselect the strongest comparator on every drift draw, and reuse it on the paired clean draw.

The fixed runtime, initialization domains, source/component/identity cycles, 20,000/10,000-step horizons, LR equation, checkpoint selection, failure rules, reason vectors, NPZ member schemas, canonical JSON, receipt preimages, and capability order make the tests mechanically executable. S0 still has to materialize and hash those declared bytes before any acquisition; that is normal experiment-plan work, not a missing choice in this proposal.

## Dominant claim and validation minimality

**Claim focus: PASS.** The sole contribution claim is that identity-indexed window-local relative-tempo routing over shared response scales improves supplied-track counting under deterministic within-track drift relative to the strongest frozen matched nonlocal/control comparator while retaining clean performance.

Exactly three paper-visible evidence blocks remain:

1. natural local-versus-strongest-nonlocal/control drift performance plus paired unwarped/clean retention;
2. the heldout three-by-three X0 expert-specialization matrix and `G_diag`; and
3. certificate coverage, analytic/certified/resampler pulse equality, and decoder boundary errors/margins.

The topology bank, identity-isolation fixture, shortcut attacks, operator inventory, receipt hashes, data firewall, resource logs, and capability ledger are appendix/audit evidence, not new contributions. Compute is staged so G0/K0, teacher certification, and synthetic kill gates fail before later response/evaluator cost. No additional route, module, arm, job, seed, dataset, or paper block is warranted.

## Blocking issues

**NONE.** The previously observed direction, balanced-BCE, pulse-edge-weight, K3-offset, natural-clock interpolation, and per-evaluator-interval unit conflicts are explicitly closed in the final hash. I found no remaining contradiction that makes the declared graph, source population, job inventory, gate order, or metric non-executable.

## Simplification opportunities

1. Keep `temporac.execution.v4` as the only normative contract. Materialize its large fixture/receipt tables in code-generated appendices rather than reproducing them in the four-page method narrative.
2. Preserve the exact three paper-visible blocks and 24 response jobs. Do not promote shortcut, firewall, topology, or runtime receipts into additional claims.
3. In the paper, express the causal chain compactly as cue-only routing intervention -> matched-scale response fusion -> positive NOLA -> one fixed decode, with teacher certification and K7 unit agreement as prerequisites.

These are presentation simplifications only; deleting any registered kill or adding a fallback would weaken identifiability.

## Modernization opportunities

**NONE.** An LLM, VLM, diffusion model, learned router, distillation path, reinforcement-learning controller, or inference-time search would not improve the identifiability of this signal-processing claim and would add component sprawl. The fixed irregular-clock cue plus matched causal experts is the appropriate route.

## Drift Warning

**NONE.** Both complete immutable anchors and the substantive TempoRAC graph are preserved. Future work would drift if it replaced local three-expert routing with WARP-PHASE or a global selector; allowed evaluator count/period/boundary fields into training, target generation, checkpoint selection, or reruns; introduced cross-person state or per-person model copies; decoded windows independently; substituted integral decoding after a peak failure; added a backup pivot; or created a fourth paper-visible claim.

## Remaining action items

These are post-refinement execution/authorization actions, not a request for Round 6 and not blockers to this final READY judgment:

1. **CRITICAL - Commit S0 before any data access.** Generate and hash the v4 environment lock, source allowlist, X0/topology/operator manifests, code/config bytes, constants, interfaces, job names, and RNG domains. Any mismatch fails rather than inviting a proposal change.
2. **CRITICAL - Obtain fresh execution authority.** Run a new TempoRAC `/experiment-plan` and server/resource preflight against the 93-A6000-hour ledger. Do not inherit WARP-PHASE authorization and do not launch from this review.
3. **CRITICAL - Execute gates in F23 order and fail closed.** Do not open the evaluator vault before the immutable G5a receipt; consume the one-time capability only in the single G5b-to-K7 process; do not filter, retune, resume, or pivot after a failure.
4. **IMPORTANT - Keep novelty and venue administration separate.** Resolve closest-prior/TWCRAC novelty, title/acronym, and four-page presentation independently before any paper claim. READY here is method-contract readiness, not novelty clearance or acceptance prediction.

No additional refinement round is proposed or authorized.

## Authorization status

- Method refinement verdict: `READY`
- Additional proposal refinement round: `NOT PROPOSED; ROUND 5 IS FINAL`
- Fresh experiment plan and preflight: `REQUIRED NEXT, NOT PERFORMED BY THIS REVIEW`
- Novelty clearance: `NOT GRANTED`
- Closest-prior/TWCRAC resolution: `UNRESOLVED`
- Title or acronym freeze: `NOT AUTHORIZED`
- Implementation or test execution: `NOT AUTHORIZED BY THIS REVIEW`
- Server or data-bearing training: `NOT AUTHORIZED`
- Test/sealed/heldout/historical result access: `NOT AUTHORIZED`
- Positive performance claims: `NOT AUTHORIZED`
- Paper drafting, editing, submission, or public claim: `NOT AUTHORIZED`

## Verdict

**READY**

The final Round-4 refinement earns `9.125/10.000`, preserves both full Problem Anchor copies and the exact TempoRAC graph with no substantive drift, and has no remaining blocking mathematical or execution-contract inconsistency. The degree-one teacher is a fail-closed prerequisite tied to both analytic X0 events and evaluator intervals; the one thresholded connected-component pass is defended by pulse targets, balanced margins, positive NOLA, all-offset split/merge tests, and resampler equality rather than replaced by an integral decoder; cue-only same-response interventions plus the matched capacity control causally test the routing claim; and G0-G5a/G5b/K0-K7 form an acyclic, population-complete firewall and validation sequence within 93 A6000-hours. The sole scientific outcome remains falsifiable: any certificate, interface, resource, primitive-unit, clean-retention, or drift-improvement failure rejects TempoRAC without a fallback.
