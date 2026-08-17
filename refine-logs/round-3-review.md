# Round 3 Research-Refine Re-evaluation: WARP-PHASE

**Reviewer:** GPT-5.6-Sol (`/root/research_refine_reviewer`)  
**Review continuity:** same reviewer thread as Rounds 1–2  
**Review independence:** same-family  
**Acceptance status:** provisional  
**Evidence status:** proposal-only; zero eligible method results  
**Venue target:** ICASSP 2027 (4+1 pages)

## Executive decision

Round 2 closes almost every named Round-2 repair. Canonical orientation is gone from both branches; the method uses raw signed increments and a unique oriented overlap integral. The ACF/FFT output is now truthfully a selector-defined pseudo-cycle, followed by a frozen evaluator-only `0.5x/1x/2x` rejection gate that cannot correct predictions. Recurrence has single-commit semantics, AvgMAE is ground-truth normalized and verified by a correct hand-worked fixture, Gate 0 requires all nine audited source components to remain nonempty, the staged ledger includes trained K4 controls and sums to 59 jobs/552 GPU-hours, and the paper still has only three visible validation blocks.

The proposal is not READY. The closest-method novelty blocker, TWCRAC K8, is explicitly open and no separate primary-source full-method adjudication receipt was supplied. Two execution interfaces also need one final freeze: the stated optimizer still batches 64 windows even though the GRU must run once over a complete identity sequence, which is not a valid reusable-autograd/cache algorithm as written; and the evaluator harmonic gate assumes one scalar annotated `P_eval,p` without binding how the actual period annotation is converted into that scalar in source-clock units. These are local specification gaps rather than a reason to rethink the route.

## Anchor verification

**PASS.** The two Problem Anchor copies in `round-2-refinement.md` are byte-for-text identical, and every exact anchor field matches the frozen `idea-stage/RESEARCH_REVIEW.json` source. The research question, population, video-first unit, primary and secondary estimands, and excluded questions are unchanged. There is no scene-total, predicted-track, end-to-end, annotation-free, generic phase-learning, or historical-result drift.

## Prior-action closure audit

| Round-2 action | Status | Finding |
|---|---|---|
| Remove orientation symmetrically | **CLOSED** | Sections 4.1–4.2 use raw `delta_k` and raw `delta_tau`; reversal sign comes only from signed overlap length. No median-sign branch remains. |
| Pseudo-cycle language | **CLOSED** | The selector output is consistently called an internal pseudo-cycle and is not asserted to be an annotated physical repetition. |
| Frozen evaluator-only `0.5x/1x/2x` gate | **PARTIAL** | Freeze order, 10% classes, video-first fractions, 90%/5% thresholds, no correction, and K5 consequence are exact. The source and aggregation rule for scalar `P_eval,p` is not bound to the actual evaluator schema. |
| GRU once per collapsed identity | **PARTIAL** | State semantics are now correct: one chronological commit, invalid-state hold, cached states, read-only overlap. The optimization table still says `batch 64 windows`; it does not say how a full-sequence graph is computed and backpropagated once per optimizer step without stale/detached caches or repeated backward through one graph. |
| Normalized video-first AvgMAE and `C>0` | **CLOSED** | The formula divides each person error by `C`, then averages people and videos. The positive-count assertion and fixture arithmetic are correct. |
| All-nine-component Gate-0 assertion | **CLOSED AS A PRECONDITION** | The proposal no longer assumes exclusions preserve all components silently; an empty component invokes K7 before label access. Actual passage remains prospective, as it should. |
| 59-job/552-GPU-hour ledger | **CLOSED** | 42 full + 14 tuning + 3 sanity jobs = 59; `42x12 + 14x3 + 6 = 552` GPU-hours. Three-seed trained clock/no-pose and nuisance-only K4 controls are included. |
| Three visible validation blocks | **CLOSED** | K4 is accounted inside Block 2 rather than hidden or promoted to a fourth claim. |
| TWCRAC K8 | **OPEN / BLOCKING** | The proposal states the truth: no independent full-method receipt exists. Under the READY rule, an explicitly open closest-prior collision is still a blocker. |

## Scores

| Dimension | Weight | Score / 10 | Weighted contribution |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9.8 | 1.470 |
| Method Specificity | 25% | 8.4 | 2.100 |
| Contribution Quality | 25% | 8.0 | 2.000 |
| Frontier Leverage | 15% | 9.2 | 1.380 |
| Feasibility | 10% | 7.8 | 0.780 |
| Validation Focus | 5% | 8.6 | 0.430 |
| Venue Readiness | 5% | 7.4 | 0.370 |

**WEIGHTED COMPOSITE: 8.53 / 10 (reported: 8.5 / 10)**  
**CALIBRATION: none**

**GAP:** No curated three-good/three-bad proposal anchors were provided, so calibration remains `none` and no exemplar comparison is fabricated. Relative to the rubric’s READY bar, the proposal is now a strong, focused preregistration: its anchor, unique treatment deletion, discrete math, boundary-safe decode, count-blind population gate, normalized estimator, hard compute ledger, and kill logic are unusually concrete. It remains below 9 because the full-sequence recurrence has no coherent optimizer batching algorithm, the evaluator harmonic gate is not yet bound to the real period-label schema, and the closest unresolved method has not been inspected in full. The first two are interface fixes; K8 is an external claim-blocking adjudication. No new model or experiment family is needed.

## Dimension review

### 1. Problem Fidelity — 9.8/10

The revised method remains exactly on the anchored question. It tests local signed warp-integral supervision against identical augmentation exposure, uses supplied per-person tracks, keeps count/period labels outside training and tuning, and applies normalized video-first AvgMAE with source-component resampling. The pseudo-cycle/evaluator separation is particularly important: it avoids redefining the research problem as mere spectral self-consistency.

### 2. Method Specificity — 8.4/10

The core method is now nearly executable. Raw clean increments define a piecewise-constant signed measure; the mapped target is one float64 signed-overlap integral; invalid support and aliasing fail closed; the clean target is stop-gradient; pause and reversal are unambiguous; and treatment differs from control only by `L_WI`. Positive interval NOLA and one decode are also fully specified.

The remaining training-loop contradiction is material. A cache of full-sequence hidden states cannot be reused across 20,000 optimizer updates: after a weight update it is stale, and retaining one autograd graph for multiple 64-window mini-batches either detaches gradients or requires repeated backward through the same graph. Freeze one valid algorithm. The simplest is to batch complete collapsed identity sequences (padded with masks), run clean and warped Conv/GRU passes once per track per optimizer step, construct all eligible 64-window losses as read-only views inside that same forward graph, average first within identity and then across identities, perform one backward/step, and discard the cache. Replace “batch 64 windows” with a frozen number of full tracks or a token/clock budget. If truncated BPTT is intended instead, it needs non-overlapping commit chunks and explicit detach boundaries; overlapping windows must remain loss views only. **Priority: CRITICAL.**

The evaluator gate also needs a single schema-bound definition of `P_eval,p`. Inspect the actual vault field and freeze one path: for example, if period is per-cycle/per-frame, define the permitted valid entries, conversion to source-frame-clock units, and the exact track aggregation (such as a linear median) before classification. Do not retain alternative branches based on what gives better agreement. Add a tiny schema fixture demonstrating `P_eval,p`, `hat P_p`, class assignment, and video-first `H_r`. This is evaluator engineering, not another experiment. **Priority: IMPORTANT.**

### 3. Contribution Quality — 8.0/10

The story is now disciplined: one discrete warp-integral objective, one identical deletion, one supporting isolation diagnostic, and explicit non-contributions. The pseudo-cycle selector is shared infrastructure and cannot become a second paper claim. The ordinary chain rule is motivation rather than asserted novelty.

Contribution quality is capped by K8. The accessible TWCRAC record already overlaps local windows, cycle consistency, pose, and non-stationary motion; only a complete primary-source method inspection can establish whether the discrete local warp-phase objective is actually residual. An equivalent disclosure must kill or re-anchor the claim, exactly as the proposal states.

### 4. Frontier Leverage — 9.2/10

The proposal uses current ideas where natural—structured transformation supervision, exact equivariance, causal interventions, deterministic provenance, and matched deletion—without forcing a foundation model. An LLM, VLM, diffusion model, RL policy, router, or expert bank would weaken causal attribution and should remain excluded.

### 5. Feasibility — 7.8/10

The run ledger is finite and internally correct, label-vault and component failures stop before evaluation, and every stage has bounded savings. Feasibility remains below 9 because the implementation does not exist, the all-nine eligibility assertion and severe selector gates may legitimately kill the pilot, and the current window-batch/full-sequence recurrence combination cannot yet be implemented as written. These are acknowledged prospective risks, not results.

The K4 model schemas should also be frozen in the eventual implementation table: exact clock/mask/warp fields, nuisance fields, zeroed pose representation, and parameter-count handling. This does not change the 59-job ledger or add a comparator; it prevents an underpowered shortcut control. **Priority: IMPORTANT.**

### 6. Validation Focus — 8.6/10

The primary estimator and bootstrap are now executable: positive normalized person error, person-first/video-first aggregation, three-seed point estimate, deterministic 10,000-draw paired component bootstrap, multiplicity, percentile rule, and separation of cluster uncertainty from seed SD are all frozen. The scorer fixture is numerically correct. The three-block layout remains focused even with complete kill-control accounting.

One wording fix is needed in Stage D: it cannot require A–C to “survive K2,” because K2 is decided by the Stage-D closest priors themselves. The condition should be that A–C survive K1/K3/K4, after which D adjudicates K2. This is a control-flow correction, not an added run. **Priority: MINOR.**

### 7. Venue Readiness — 7.4/10

The proposal is methodologically close to an ICASSP-ready execution plan, but not a novelty-frozen venue story. The full-sequence training loop and evaluator period binding must be closed, and K8 must be resolved from the complete primary source. The partial GT-assisted cache also remains pilot evidence only; even a positive outcome must retain the claim ceiling.

## Focus, simplicity, and frontier assessment

- **Anchor:** preserved verbatim.
- **Dominant contribution:** sharp and singular—the discrete warp-integral loss.
- **Simplicity:** strong. Removing orientation was cleaner than adding symmetric gauge machinery; recurrence commits once; NOLA is fixed; the base has one loss; validation has three blocks.
- **Frontier appropriateness:** strong and unforced; no modernization component is missing.
- **Implementation readiness:** close, pending a valid full-track optimizer loop and schema-bound evaluator period extraction.
- **Claim readiness:** blocked by open TWCRAC K8.

## Simplification Opportunities

1. Preserve the current raw-sign formulation; do not reintroduce phase orientation or a learned harmonic corrector.
2. Treat the per-step full-sequence hidden states as ephemeral autograd values, not a cross-step cache. This makes “single commit” compatible with ordinary optimization.
3. Keep the evaluator harmonic gate and K4 schemas in receipts/tables; they remain decision gates inside the existing three blocks, not new contributions.

## Modernization Opportunities

**NONE.** The method is already appropriately frontier-aware for the stated signal-processing bottleneck.

## Remaining action items

1. **CRITICAL:** Replace `batch 64 windows` with one exact full-sequence training/batching/backpropagation algorithm consistent with single-commit recurrence and fresh parameters each step.
2. **IMPORTANT:** Bind scalar `P_eval,p` to the actual evaluator period schema, units, valid-entry rule, aggregation, and a deterministic fixture.
3. **IMPORTANT:** Freeze exact K4 shortcut-model input schemas and parameter handling within the already budgeted six jobs.
4. **MINOR:** Correct the Stage-D condition so that Stage D adjudicates K2 rather than purportedly following a prior K2 decision.
5. **CRITICAL for READY/claim freeze:** Complete an independent primary-source full-method TWCRAC adjudication; apply K8 if equivalent.

## Drift Warning

**NONE.** Every requested fix stays inside the original supplied-track, count/period-label-restricted, within-track tempo-drift question. No baseline expansion, predicted-track claim, or auxiliary contribution is required.

## Verdict

**REVISE**

The score rises from 7.9 to 8.5. The route is now focused, technically serious, and close to execution, so RETHINK is unwarranted. READY is prohibited because the full-sequence optimization interface and evaluator period binding remain incomplete and K8 is truthfully open. No efficacy result, gate passage, or novelty clearance is inferred.

## Audited input SHA-256

| Input | SHA-256 |
|---|---|
| `refine-logs/round-2-refinement.md` | `2694a4eaa518bf216f21ffa77fddd54141267eaa1dd5520073583bbae995953e` |
| `refine-logs/REFINE_STATE.json` | `998f42bbf6029f4fd21150f811af45bc3707a5da103ab5ed7a5248a18a68de3a` |
| `refine-logs/score-history.md` | `1498d08771dcbac713dba4694c3eea600ae8a35420b473ed30d1c8b4d311576e` |
| `refine-logs/round-2-review.md` | `8aaf103283285c3c7a80ca96640350bdeca0d6d98f056b500648adeb55c26664` |
