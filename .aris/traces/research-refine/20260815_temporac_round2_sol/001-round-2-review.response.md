# TempoRAC Round 2 — Senior Method-First Re-evaluation

CALIBRATION: none

No curated known-good/known-bad research-refine proposals were supplied, so the score is not anchored to a local calibration set. I apply the stated top-venue, method-first rubric. This is a continuation by the same GPT-5.6-Sol reviewer; the review is same-family and provisional.

## Input and continuity note

I re-read `refine-logs/temporac/round-1-refinement.md` from start to EOF. Its SHA-256 is `977d1368237cca4348973cf0142ecfee1157ad1073ebbdc7ca2430e673e82a9c`, exactly the hash supplied for this round. I also re-read `refine-logs/temporac/PHASE_ORIGIN_AUDIT.md`, the original proposal, the canonical TempoRAC brief and reviews, the legacy JSON review, the novelty record, the four root-level audits/addenda, and the complete Round 1 raw review. This ruling is based on the current files, not a parent summary or stale proposal hash.

The previously requested `idea-stage/RESEARCH_REVIEW.md` remains absent. As authorized in Round 1, `idea-stage/RESEARCH_REVIEW.json` is the only canonical legacy-review substitute. That input-resolution fact is recorded but is not a blocker.

Reviewer metadata:

- `reviewer_model`: `gpt-5.6-sol`
- `reviewer_family`: `openai`
- `review_independence`: `same-family`
- `acceptance_status`: `provisional`

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

## Anchor and original-graph ruling

**Immutable-field status: VERBATIM PRESERVED.** The five bullets that the initial proposal explicitly declares immutable occur twice in the revised document. Both copies match those five original bullets exactly; their common field-block SHA-256 is `919a22bbab10c4f1e830f43bcd2c2a6d2ac604a099729013e7478b02ac638846`. The revision does not weaken the no-label objective, supplied-track setting, private-identity requirement, continuous-response requirement, or exactly-once decoder requirement.

**Full-section status: NOT VERBATIM.** The initial `## Problem Anchor` section ended with a separate drift-guard paragraph beginning “The five bullets above are immutable.” Both revised copies omit that paragraph, so the full section is not byte-identical even though the five declared immutable fields are. This is a bounded anchor-integrity defect, not substantive research drift: the revised graph and exclusions enforce the omitted guard. The next refinement should restore that paragraph verbatim in both anchor copies.

**Graph status: MATCHES THE ORIGINAL TEMPORAC IDEA.** The operative path is still:

`supplied person-indexed track -> identity-private preprocessing/state -> one shared local encoder -> three shared matched slow/medium/fast response experts -> fixed cue-only relative-tempo soft gate -> response-level fusion -> positive NOLA reconstruction -> one fixed threshold-0.5 connected-component decode per identity`.

The per-window statistic remains local; the documented identity-local track reference only defines relative tempo. No scene counter, per-person model copy, learned router, count/period input, WARP-PHASE selector, independent window counting, integral count decoder, or second decode has replaced this graph. The teacher is explicitly a prerequisite contract, expert specialization is explicitly a supporting mechanism check, and the only paper-level claim is the local relative-tempo routing claim. There is no research drift.

## Scorecard

| Dimension | Weight | Score | Weighted contribution |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9.6 | 1.440 |
| Method Specificity | 25% | 7.2 | 1.800 |
| Contribution Quality | 25% | 7.5 | 1.875 |
| Frontier Leverage | 15% | 7.5 | 1.125 |
| Feasibility | 10% | 6.5 | 0.650 |
| Validation Focus | 5% | 7.5 | 0.375 |
| Venue Readiness | 5% | 6.5 | 0.325 |

**Weighted composite:** **7.590 / 10** (displayed: **7.59 / 10**)

**GAP:** The proposal is **1.410 points below** the READY threshold of 9.0. Round 1's large structural gaps are substantially closed: the degree tests include the closing edge, the teacher has a fixed landmark and fail-closed natural audit, the peak path is fixed rather than replaced, the router comparison reuses one cached response bank, the feature/vault boundary is specified, and the paper has one dominant claim. The remaining gap is narrower and mostly at contract boundaries. A population allowance of negative phase edges is inconsistent with an exact per-track event guarantee; raw floor crossings lack owned seam numerics; positive source-clock gaps do not rule out hidden cycles; the claimed tangent invariance is only approximate on a curved orbit; the pulse gate does not unambiguously bind the trained model's outputs; the capacity control changes the placement of the sigmoid; and the omitted drift paragraph must be restored for full-section anchor integrity. These are not requests for more modules or a larger benchmark. They are small but blocking corrections to the existing mechanism.

## Overall assessment

This is a materially better proposal. It now has a recognizable scientific center: after a frozen response-unit prerequisite, does an identity-indexed, cue-only local relative-tempo gate select the useful response scale from three shared experts under within-track drift? The revision correctly removed `L_cont`, PAMS-TCC initialization, ACF fusion, a learned router, the GRU branch, and integral decoding from the canonical route. Positive NOLA and the single decoder are correctness plumbing rather than a parallel novelty claim. The same-response gate intervention is much closer to a causal experiment than the retrained-arm design in Round 1.

The teacher is also no longer defended only by theorem-shaped prose. The fixed clock-blind state, explicit no-bypass decoder, closing-edge winding, coverage and gap tests, degree/harmonic/bypass attacks, deterministic checkpoint precedence, pose-landmark origin, and terminal natural-unit audit make it a serious falsifiable prerequisite. A unique physical landmark can remove the constant rotation gauge without sample-density weighting, because the origin averages one landmark phase per complete traversal rather than all samples. Anchoring that landmark to phase zero also makes the event seam insensitive to a remaining orientation-preserving phase homeomorphism, provided the phase is forward-monotone and each traversal is fully observed.

The current contract nevertheless does not yet justify emitting one certified target on every admitted natural track. The phase-origin audit itself identifies the decisive distinction: a suite-level topology pass can select a teacher artifact, but each target-bearing track needs a stricter individual crossing certificate. The revised proposal still permits up to 1% negative sequential edges in G1, uses unsnapped raw floors in F7, and regards every `Delta q > 0` as a valid edge. Those three choices jointly leave forward recrossing, floating-point seam ownership, and unobserved-cycle aliasing unresolved. Until they are fixed, the one-pulse premise and therefore the final peak proof are conditional rather than closed.

## Resolution of the Round 1 blockers

| Round 1 blocker | Round 2 ruling |
|---|---|
| Degree-zero arc compression and missing closing-edge degree test | **Substantially resolved.** Minimum-positive-variation selection is gone; F8 includes the closing edge; coverage, gap, collision, reconstruction, degree, harmonic, symmetry, static-code, decoder-bypass, and resampler attacks are frozen. Per-track monotonic and sampling certification remains open. |
| Unfixed phase origin | **Substantially resolved.** A deterministic pose-only landmark, per-traversal origin, resultant/coherence tests, cross-resampler tests, and terminal abstention are specified. The local-maximum inequality and tangent invariance claim need numerical closure. |
| Peak decoder silently replaceable by integral count | **Resolved as a design decision.** F21 is the only decision decoder; the mass integral is audit-only and explicitly non-substitutable. Hard model-output resolvability still needs an exact gate. |
| Learned/latent cue bypass and retrained-arm confounding | **Resolved for the main local-versus-nonlocal intervention.** The router is fixed and cue-only, and all five gate arms operate on one cached expert response tensor. |
| Expert/control match | **Partly resolved.** Parameters, FLOPs, and receptive fields are fixed, but the control averages logits before one sigmoid whereas TempoRAC fuses sigmoid responses. That nonlinear mismatch is a causal confound. |
| Feature/vault leakage | **Resolved at specification level.** Whitelist-only feature shards, a separately permissioned evaluator vault, forbidden-key rejection, source-component splits, and receipts are explicit. No data-bearing execution is authorized by this review. |
| Contribution sprawl | **Resolved at claim level.** Routing is the single dominant contribution; the teacher is a prerequisite, specialization is a mechanism check, and NOLA/decode is plumbing. The prerequisite remains expensive to explain in four pages. |

## Degree-one teacher and phase-origin ruling

**Ruling: mathematically credible on its stated supported domain, but not yet numerically sufficient for target emission.**

The revised teacher has the right high-level structure. On a complete identifiable primitive orbit, a continuous phase representation with a no-bypass reconstruction decoder, winding `+1`, high circle coverage, small circular gaps, low collision, correct orientation, and a unique stable physical landmark is a defensible conditional degree-one coordinate. The evaluator-hidden primitive coordinate and explicit degree `0`, `-1`, `+2`, harmonic, symmetric-orbit, static-code, and decoder-bypass attacks are appropriate. The natural teacher/count audit is correctly terminal and cannot repair targets or tune a teacher.

Four contract defects remain:

1. **Suite-level negative-edge tolerance cannot certify a target-bearing track.** G1 permits negative sequential edges on up to 1% of an accepted orbit. Net winding `+1` does not imply exactly one positive seam crossing when the unwrapped path backtracks: a track can cross forward, cross backward, and cross forward again while retaining net winding `+1`. F7 suppresses the backward event but counts both forward crossings. The teacher artifact may use a population pass rate, but every individual track allowed to emit `e*` must satisfy `-1e-12 <= delta_t < 0.25` on every valid edge, must have no backward seam crossing, and must produce exactly one positive crossing between successive retained landmarks. A failing track abstains; it cannot contribute a target.

2. **Seam numerics are unowned.** F7 accumulates in unspecified precision and applies `floor` directly. Values within floating-point error of an integer can move a pulse by an edge or create an implementation-dependent result. Accumulate `A` with compensated float64 summation, apply the audit's frozen snapping operator `S_tau` with `tau = 1e-7` cycles, require the pulse array to be unchanged at `tau` in `{1e-8, 1e-7, 1e-6}`, and accept only increments in `{0,1}`. Seam-ambiguous tracks must abstain.

3. **The no-hidden-cycle assumption is not operational.** The graph marks any edge with `Delta q > 0` valid. A small wrapped endpoint increment does not exclude one or more unobserved full cycles across a large source-clock gap. This cannot be learned from the two endpoint poses. Freeze a supported acquisition domain: for example, split at every `Delta q != 1`, declare the minimum supported source period, and require the individual phase-increment certificate on all remaining edges. Any gap that cannot defend the no-hidden-cycle assumption is invalid and may not be bridged by unwrapping, NOLA, or the decoder. This is an assumption/eligibility receipt, not a new model.

4. **The claimed tangent is not exactly warp invariant.** Averaging two normalized immediate-neighbor chords on a curved orbit depends on the physical offsets of those neighbors; monotone resampling changes those offsets even though displacement magnitude is discarded. `L_view` may train robustness, but the formula itself is not an exact invariant. Relabel it as a local direction estimate and make full matched-state canonical-phase and emitted-pulse equality—not merely origin drift—the unseen-resampler gate. If that gate fails, remove the tangent from the supported state or abstain; do not introduce another learned timing module.

There is also a small executable ambiguity in F5: the prose says “strict valid local maximum,” but the equation uses `ell[t-1] < ell[t] >= ell[t+1]`. Freeze either strict `>` on both sides with a negative parabolic denominator or an explicit plateau ownership/abstention rule. Adjacent minima, run range, and zero-denominator handling must be deterministic.

The smallest repair is one evaluator-owned `certify_target(track)` predicate that returns `{eligible, abstention_reason, landmarks, delta, crossings, pulse}` and enforces all of the above before a pulse is serialized. This consolidates the origin/topology conditions rather than adding a second teacher.

## One-final-peak-pass ruling

**Ruling: preserve it; it is conditionally defensible and must not be replaced by an integral decoder.**

F17 now fixes class-balanced BCE, a positive event margin toward at least `0.75`, and a non-event valley margin toward at most `0.25`. F19 gives positive taper weights, F20 fuses responses before any discrete operation, and F21 performs one fixed threshold-`0.5` connected-component count per identity. No NMS, minimum-distance rule, learned threshold, period rule, rounding rule, or integral fallback remains. This is faithful to the original claim.

The proof obligation is simple: on the supported track domain, the teacher must emit exactly one isolated event per traversal; every held-out model response at an event must remain above `0.5`; every non-event between events must remain below `0.5`; and positive NOLA must preserve those inequalities. F17 encourages these conditions but, as an average soft loss, does not guarantee them. G4 says “held-out pulse pack” without stating whether it is an operator fixture, a frozen trained-bank output, or both. An operator-only pack can prove NOLA/decoder plumbing but cannot prove that the trained experts satisfy the margins.

Freeze one G4 pack schema with two named sources inside the same gate: deterministic operator attacks and frozen held-out model outputs from unseen resamplers. Require zero threshold misses, extras, splits, or merges and record worst-case positive and negative margins. This is one gate, not an expanded experiment menu.

G4 also includes event spacings `2` and `3` edges even though the individual alias contract requires every forward increment to be below `0.25` cycles. Such spacings are outside the supported degree-one target domain; spacing `4` is the first boundary case that can occur because the previous event may overshoot its seam. Delete unsupported spacings `2` and `3`, derive the minimum spacing from the final certified increment domain, and retain `4, 5, 8, 16, 32, 64` plus boundary/grid offsets. Testing impossible target configurations adds work without strengthening the claim.

If the model-conditioned G4 pack fails, the correct result is rejection of the strict TempoRAC route. The audit integral remains non-decision-bearing and cannot rescue the same paper identity.

## Cue-only routing and matched-expert causal ruling

**Ruling: the main routing intervention now causally supports the original claim, conditional on one control correction and explicit decision margins.**

The frozen NUDFT cue has a single declared source clock, fixed frequency grid, confidence calculation, reliability rule, and relative-log-tempo anchors. The gate receives only cue values and reliability, not pose features, expert outputs, count, period, density, or identity metadata. Three equal-width TCN branches have the intended slow/medium/fast receptive fields. The canonical bank is trained once. Local, global, uniform, blocked, and within-track-shuffled gates then act on the exact same cached expert responses. That intervention cleanly asks whether the local cue selects a more useful scale; it no longer conflates routing with retraining. The held-out `3 x 3` response-loss/split-merge matrix is an appropriate supporting specialization check, not a second dominant claim.

The remaining capacity-control mismatch is important. TempoRAC applies a sigmoid in each expert and mixes response probabilities; the proposed control averages three logits and applies one sigmoid. Those functions differ by the placement of a nonlinearity, so an advantage can arise from Jensen/calibration effects rather than routing or specialization. Use the identical three sigmoid branches and an exact uniform probability-level fusion for the unstratified capacity control. Name it accurately as an “unrouted, capacity-matched three-branch control”; it may expose one final response without pretending to be a single internal head. Keep the exact parameter, FLOP, and receptive-field receipts.

Finally, replace the bare G3 word “beats” with a frozen paired decision rule. K7 already supplies a natural-pilot effect size and source-component bootstrap. The synthetic same-response comparison should likewise require a preregistered paired lower bound above zero for local versus the strongest nonlocal gate. Blocked and uniform are intentionally identical outputs; their separate traces are a path-integrity assertion, not two independent baselines.

With those changes, cue-only routing plus matched experts is a credible causal test of the original TempoRAC claim. A learned router, fourth expert, residual fusion path, or separately retrained control would weaken rather than strengthen that test.

## Exact interface and gate executability

The revision is much closer to executable: tensor widths, masks, clock ownership, duplicate collapse, architectures, losses, tapers, thresholds, optimizer settings, state boundaries, data schemas, hashes, and kill actions are mostly frozen. It is not yet an executable frozen stack because three gates contain cross-contract ambiguities.

| Gate | Round 2 status | Exact closure needed |
|---|---|---|
| G0 feature/vault firewall | **EXECUTABLE ON PAPER; NOT RUN OR AUTHORIZED HERE** | Implement only after separate authorization. The whitelist, forbidden-key, source-component, nuisance-schema, and receipt interfaces are sufficiently exact. |
| G1 degree-one teacher | **BLOCKED FOR TARGET EMISSION** | Add the per-track strict monotonic/crossing predicate, float64 compensated accumulation and `S_tau`, an operational no-hidden-cycle/gap rule, full cross-resampler pulse equality, and deterministic F5 strict-max semantics. |
| G2 seven invariants | **EXECUTABLE ON PAPER** | Preserve the documented distinction between local raw cue and identity-local reference dependence. No new module is required. |
| G3 routing/specialization | **PARTIALLY EXECUTABLE** | Put the control sigmoid before uniform probability fusion and freeze a paired local-versus-strongest-nonlocal decision threshold. |
| G4 pulse/NOLA/peak | **PARTIALLY EXECUTABLE** | Bind the pack to both operator fixtures and frozen held-out model outputs, record hard margins, and remove spacings outside the certified phase domain. |
| G5 supplied-track pilot | **CONDITIONALLY EXECUTABLE, NOT AUTHORIZED** | It can run only after G0-G4 pass and after the exact evaluator and comparison hashes are frozen. This review does not authorize data-bearing or server work. |

The K0-K7 table is acceptable as a decision layer over immutable G artifacts, but it should not be presented as eight separate scientific contributions. Internally, retain the fail-closed checks. In the four-page paper, compress them into three evidence blocks: response-unit/decoder eligibility, frozen-response routing mechanism, and supplied-track efficacy/stationarity.

## Staged compute and validation ruling

**Compute is bounded, but engineering is still underestimated.** The `<=100` A6000-hour ceiling is plausible because the expensive natural pilot is last and earlier failures stop the route. The staged order is scientifically responsible: specification and CPU fixtures precede model training; the teacher freezes before natural evaluator opening; the expert bank trains once; and the three-seed pilot runs only after causal and plumbing gates. There is no need for a foundation model, learned router, joint training loop, fourth expert, or large baseline sweep.

The `15-23` engineer-day estimate is optimistic for implementing and independently validating the packer/vault boundary, deterministic orbit and resampler suite, teacher, three response branches, irregular-clock NUDFT, state-isolation instrumentation, NOLA/decoder traces, and source-component bootstrap. A more credible preregistered estimate is roughly `20-30` focused engineer-days, still within the stated compute resources. This should not be “fixed” by deleting the per-track certificate; it should be fixed by consolidating duplicate gate machinery and narrowing G4 to the supported domain.

## Required revisions for dimensions below 7

### Feasibility — 6.5 / 10

**Specific weakness:** The staged GPU budget is credible, but the schedule assumes that several unresolved numerical contracts can be settled during implementation. G1 and G4 are prerequisites, not ordinary bugs: without a per-track target certificate and a model-conditioned peak gate, downstream training can consume invalid targets. The feature/vault loader, execution JSON, fixture generator, and trace instrumentation are still specifications rather than existing artifacts.

**Concrete fix:** Before any data-bearing run, freeze one compact executable specification containing `certify_target`, supported source-gap/period assumptions, seam precision/snapping, strict landmark semantics, the corrected probability-level capacity control, and the two-source G4 pack. Adjust the engineering estimate to `20-30` focused days while retaining the `<=100` A6000-hour hard cap and numeric-order stop gates.

**Priority:** CRITICAL

### Venue Readiness — 6.5 / 10

**Specific weakness:** The method story is now focused, but a four-page ICASSP paper cannot carry the full 1,073-line contract as coequal exposition. The central residual novelty remains an unvalidated conjunction until the same-response routing effect is observed; overlap with the closest protocol-matched prior, including TWCRAC, remains unresolved; and there are no authorized eligible results. The teacher prerequisite can still eclipse the claimed routing contribution if its details dominate the paper.

**Concrete fix:** Close the four contract edits above, then expose only three paper-visible evidence blocks and one dominant claim. Report the teacher and decoder as a compact eligibility certificate with abstention coverage; report specialization only as mechanism support; and reserve novelty clearance, title/acronym freeze, server launch, result claims, paper drafting, and submission authorization for separate later reviews after eligible evidence exists.

**Priority:** IMPORTANT

## Simplification Opportunities

1. Consolidate the landmark, topology, crossing, gap, and seam rules into one frozen per-track `certify_target` interface and one receipt. This removes duplicated prose and makes G1/K1 executable without adding a model.
2. Remove G4 event spacings `2` and `3`, which are outside the declared `delta < 0.25` supported domain. Keep only derived boundary spacings and the existing pause, gap, grid, plateau, split, and merge attacks.
3. Replace the logit-averaging “one-output” control with identical sigmoid branches plus uniform probability fusion, and call it an unrouted capacity-matched control. This deletes an architectural mismatch while reusing the canonical fusion operator.

For paper presentation, merge the internal G/K receipts into three evidence blocks; do not delete the fail-closed internal checks.

## Modernization Opportunities

NONE. The problem is governed by phase topology, irregular sampling, response-scale selection, state isolation, and reconstruction semantics. An LLM, VLM, diffusion model, RL policy, or inference-time search loop would not close the identified gaps and would create contribution sprawl. The fixed spectral cue, small shared TCNs, frozen teacher targets, and causal interventions are the appropriate modernity level for this signal-processing problem.

## Drift Warning

NONE substantively. The five immutable Problem Anchor fields and the original TempoRAC graph are preserved. The complete anchor section is not verbatim because its trailing drift-guard paragraph was omitted; restore that paragraph exactly in both copies as a bounded integrity repair. The following would constitute substantive drift and require a new review: replacing the three-expert local router with WARP-PHASE or an easier global selector; permitting cross-identity state; using human count/period annotations to create, repair, or select training targets; decoding windows independently; or substituting an integral decoder after the fixed peak contract fails.

## Remaining action items

1. **CRITICAL — Make certification per-track, not only population-level.** For every emitted target require strict forward increments, no backward crossing, exactly one positive seam crossing per retained-landmark traversal, no collision, and all existing degree/origin checks; otherwise abstain.
2. **CRITICAL — Own seam and sampling numerics.** Use compensated float64 accumulation, `S_tau`, tolerance-invariance checks, and an explicit no-hidden-cycle/source-gap eligibility rule. Never unwrap or reconstruct across an invalid gap.
3. **CRITICAL — Close the trained-response peak gate.** Bind G4 to frozen held-out model outputs as well as operator fixtures, measure hard event/valley margins and zero split/merge, and remove unsupported spacings `2` and `3`.
4. **IMPORTANT — Correct the warp-invariance claim.** Treat F1 as an approximate local direction estimate and require full canonical phase/pulse equality across unseen resamplers; fix strict-local-maximum and parabolic-denominator semantics.
5. **IMPORTANT — Correct the capacity control.** Apply branch sigmoids before exact uniform probability fusion and preregister the paired G3 local-versus-strongest-nonlocal decision rule.
6. **IMPORTANT — Compress, do not expand.** Preserve one routing claim, one supporting specialization check, and three paper-visible evidence blocks; keep detailed G/K checks as internal receipts.
7. **IMPORTANT — Restore full-section anchor integrity.** Copy the omitted original drift-guard paragraph verbatim after the five immutable fields in both Problem Anchor occurrences; do not alter the five fields.
8. **IMPORTANT — Keep authorization layers separate.** Novelty clearance, closest-prior resolution, title/acronym freeze, implementation, server/data-bearing training, positive result claims, paper drafting, and submission remain ungranted by this provisional method review.

## Authorization status

- Method-refinement verdict: `REVISE`
- Local contract editing: appropriate as the next refinement step, but not started by this review
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

The revision is on-anchor, focused, and much closer to an executable mechanism. It is not READY because the exact one-event premise remains uncertified on individual admitted tracks, seam and hidden-cycle numerics are open, the held-out peak gate does not yet unambiguously test trained outputs, and the capacity control changes the response nonlinearity. These are bounded corrections within the current TempoRAC idea. They do not justify a new module, a learned router, an integral decoder, a larger experiment matrix, or a change of problem.
