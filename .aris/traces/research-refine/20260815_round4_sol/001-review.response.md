# Round 4 Research-Refine Re-evaluation: WARP-PHASE

**Reviewer:** GPT-5.6-Sol (/root/research_refine_reviewer)  
**Review continuity:** same reviewer thread as Rounds 1-3  
**Review independence:** same-family  
**Acceptance status:** provisional  
**Evidence status:** proposal-only; zero eligible method results  
**Venue target:** ICASSP 2027 (4+1 pages)

## Executive decision

Round 3 genuinely closes the full-identity optimizer, evaluator-period binding, and Stage-D control-flow repairs. It also records TWCRAC truthfully as BLOCKED with k8.decision=NOT_PASSED_BLOCKED and novelty_clearance=false. The proposed optimizer now uses eight complete identities per fresh graph, one backward per step, four-step gradient accumulation, and 20,000 updates, so 8 x 4 x 20,000 = 640,000 track draws is coherent. The raw evaluator period is uniquely the float64 NumPy median of audited integer source-frame half-open spans. Stage D runs only after A-C survive K1/K3/K4 and then adjudicates K2.

The implementation audit does, however, expose real local specification gaps that the complete replacement cannot delegate silently to code. Root/scale normalization has no missing-landmark rule; the 64/128-retained-clock selector does not define how a source span larger than 256 enters a 256-point FFT; the FFT evaluation and endpoint-local-maximum conventions are incomplete; nested views do not resolve to one training selection and decoder confidence g_j is undefined; the 1,000-fixture generator has no exact mixture or frozen bytes; warped pose/confidence/mask interpolation is not numerically specified; and the K4 fixed-code exposure/vector construction is only partial. Independently, K4 still uses the semantic predicates “retains,” “preserves,” and “erases” without freezing all decision formulas and thresholds. These are bounded interface freezes, not reasons to add experiments or rethink the route.

The proposal is not READY. The local specification is not yet executable byte-for-byte, and the external TWCRAC primary-source gate is independently blocked. Abstract-only evidence cannot close K8.

**LOCAL_SPEC_STATUS: MULTIPLE_MINIMAL_INTERFACE_FREEZES_REQUIRED**  
**EXTERNAL_K8_STATUS: BLOCKED / NOT_PASSED_BLOCKED**

## Anchor verification

**PASS.** Both Problem Anchor copies in refine-logs/round-3-refinement.md are byte-for-text identical to each other and to the frozen anchor in refine-logs/round-0-initial-proposal.md. The anchor-text SHA-256 is 619148c765175167da093d9b054bd2b43fae0e5e97d981fb9748ef57f4d7dba7. The research question, partial supplied-pose population, video-first unit, primary and secondary estimands, and excluded questions are unchanged. There is no predicted-track, scene-total, end-to-end, annotation-free, generic phase-learning, or historical-result drift.

## Round-3 action closure audit

| Requested Round-3 fix | Status | Finding |
|---|---|---|
| Fresh full-identity optimizer | **CLOSED** | Each step contains eight complete collapsed identities; clean and warped Conv/GRU graphs are fresh and complete; windows are read-only views; loss is averaged within identity and then across identities; one backward occurs per step; four steps accumulate before clipping/update. No cross-step state or graph survives. |
| Schema-bound scalar evaluator period | **CLOSED** | The addendum verifies nonempty integer source-frame [s,e) pairs. The proposal freezes P_eval as np.median(float64(e-s)), global validity assertions, exact source/catalog hashes, and a deterministic harmonic-class fixture. No alternate label or grid-inversion path remains. |
| Exact K4 model inputs | **PARTIAL** | Zeroed pose, retained masks/fractions, q/L clock exposure, opaque-key prohibition, metadata exclusion, parameter count, seeds, and budget are stated. The two mask-fraction sign vectors are not reproducible from “seed 20270815” without an RNG/construction or literal bytes, and the text never states unambiguously whether the primary treatment/control receive the same fixed q/L code or no such code. |
| Stage-D control flow | **CLOSED** | A-C first survive K1/K3/K4; Stage D then runs the closest priors and adjudicates K2. |
| Truthful TWCRAC disposition | **CLOSED AS A RECEIPT; EXTERNALLY BLOCKING** | The cited receipt hash matches. Full-text coverage is zero pages, verdict is BLOCKED, K8 is NOT_PASSED_BLOCKED, claim freeze is BLOCKED, and novelty clearance is false. The proposal makes no distinctness or absence inference. This is not K8 closure. |

## Supplemental seven-interface implementation audit

| Read-only audit concern | Status from the Round-3 text | Minimal in-scope resolution |
|---|---|---|
| Missing-joint hip/shoulder root and scale | **OPEN / BLOCKING** | “Hip midpoint” and “median valid shoulder-to-hip scale” do not say what happens when one/both hips or shoulders are masked. Name the joint indices and freeze a deterministic mask-aware rule. The simplest fail-closed rule is: mean all valid hips for the root, mean all valid shoulders for the shoulder point, take positive finite root-to-shoulder distances, use their track median with the stated 1e-3 floor, invalidate a frame lacking a root, and invoke K7 if no scale support exists. Do not fall back to forbidden boxes or evaluator data. |
| 64/128 retained-clock window versus a source span greater than 256 | **OPEN / BLOCKING** | Retained-clock count is not integer-grid length. State exactly how q_first:q_last is placed into the 256-point masked FFT: padding, deterministic resampling, truncation, or fail-closed rejection. Also state the resulting period units. No new run is required. |
| FFT bins, normalization, and local-maximum endpoints | **OPEN / BLOCKING** | E(1/P) is generally off-bin, while masking/detrending and endpoint maxima are unspecified. Freeze the FFT convention, mask normalization, mean removal if any, bin evaluation/interpolation, zero/Nyquist handling, and whether P=4 and P=P_max use one-sided neighbor tests. A deterministic fixture must cover an off-bin period and both endpoints. |
| Nested views to one selector output and decoder g_j | **OPEN / BLOCKING** | Agreement thresholds do not say which view/window supplies the training pseudo-cycle, and g_j is only called fixed/untrained. Freeze one canonical selection path and the exact interval mapping. The minimal simplification is to select on the unjittered clean reference window, use the second view and 128/64 nesting only as rejection checks, take the already specified track median for evaluator use, and set g_j=1 on valid intervals and 0 otherwise unless an explicit confidence formula is supplied. |
| Exact 1,000-fixture distribution | **OPEN / BLOCKING** | “At least 250” plus a list of covered phenomena does not determine a fixture population, yet pass rates depend on the mixture. Freeze category counts summing to 1,000, parameter distributions and composition/overlap rules, RNG/library versions, and either the generator hash plus expected arrays or canonical fixture bytes plus SHA-256. |
| Warped pose/confidence/mask interpolation | **OPEN / BLOCKING** | The text forbids crossing invalid cells but never defines how x/y/confidence and joint/frame masks are produced at tau(q). Freeze the interpolation kernel and dtype/rounding, endpoint rule, joint-validity rule, mask propagation, and pause/reversal handling. A minimal linear rule may interpolate x/y/confidence only when both bounding source joints are valid, mark all other joints invalid and zero them after mask formation, and reuse the same fail-closed cell rule already stated. |
| K4 fixed vectors and primary-arm clock-code exposure | **PARTIAL / BLOCKING** | The q alternating vector and L half-positive/half-negative vector are exact. The two seeded orthogonal mask vectors are not, and primary-arm exposure is ambiguous because the 68-channel base omits the code while the K4 section adds it after Conv1. Bind literal vector bytes/hashes or a fully specified PCG64/Rademacher/orthogonalization algorithm, then state in one sentence whether both primary arms receive the identical fixed code or neither does. Keep the choice identical between treatment and augmentation-only. |

All seven concerns are interface definitions inside the existing learner, selector, augmentor, fixtures, or K4 receipt. Closing them does not authorize a new model, baseline, job, endpoint, or visible block.

## Preserved technical audit

| Item | Status | Finding |
|---|---|---|
| Discrete warp law and orientation | **CLOSED** | Both branches retain original signed increments. The target is the unique float64 sum of clean signed rates times oriented warped-cell overlap; reversal changes interval orientation rather than invoking canonical orientation. The clean target is stop-gradient and invalid/alias support fails closed. |
| Phase/fundamental claim boundary | **CLOSED PROSPECTIVELY** | The selector is explicitly a pseudo-cycle, not an annotated physical repetition. The evaluator-only 0.5x/1x/2x gate runs after freeze, rejects only, and cannot correct counts. The remaining selector numerical interfaces above concern reproducibility, not claim drift. |
| Interval NOLA and decode geometry | **CLOSED** | The interval weight is strictly positive over every half-open window cell, D_j must be finite and at least 1e-3, padding creates no mass, and each supplied identity is decoded once. The scalar g_j that multiplies valid mass remains to be defined. |
| Population and duplicate clocks | **CLOSED AS A PRECONDITION** | Duplicate disagreements invalidate rather than average; eligibility is count-blind; ambiguous slots stay excluded; all original nine development components must remain nonempty before evaluator access. Actual passage remains prospective. |
| Primary endpoint and bootstrap | **CLOSED** | AvgMAE is normalized person-first and aggregated video-first; every C>0 assertion is fail-closed; K1 requires both R>=0.05 and an absolute paired-effect CI lower endpoint greater than zero. Exactly 10,000 PCG64(20270815) draws resample all nine frozen source components with multiplicity and linear 2.5%/97.5% quantiles. |
| Learner, ledger, and presentation | **CLOSED** | There is one 225,026-parameter base learner and one treatment deletion. The cap is 42 full + 14 training-only tuning + 3 sanity = 59 jobs, with 42x12 + 14x3 + 6 = 552 GPU-hours. The paper retains three visible validation blocks. |

## Scores

| Dimension | Weight | Score / 10 | Weighted contribution |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9.9 | 1.485 |
| Method Specificity | 25% | 8.2 | 2.050 |
| Contribution Quality | 25% | 8.0 | 2.000 |
| Frontier Leverage | 15% | 9.2 | 1.380 |
| Feasibility | 10% | 8.0 | 0.800 |
| Validation Focus | 5% | 8.0 | 0.400 |
| Venue Readiness | 5% | 7.4 | 0.370 |

**WEIGHTED COMPOSITE: 8.485 / 10 (reported: 8.5 / 10)**  
**CALIBRATION: none**

**GAP:** No curated three-good/three-bad proposal anchors were supplied, so calibration remains none and no exemplar comparison is fabricated. Relative to the READY bar, the proposal has an unusually exact core signed-overlap law, optimizer, NOLA, normalized K1 estimator, component bootstrap, staged budget, and kill logic. It is still one implementation-freeze round short locally: selector preprocessing/numerics, selector-to-decoder binding, fixture generation, warp resampling, and K4 code/decision interfaces do not yet determine one program. Separately, TWCRAC full-method coverage is zero, so contribution distinctness and claim freeze remain externally blocked. These gaps require definitions and a primary source, not added experiments. No efficacy result or gate passage is inferred.

## Dimension review

### 1. Problem Fidelity - 9.9/10

The proposal remains exactly on the supplied-track tempo-drift question. It tests one local signed warp-integral loss against identical augmentation, keeps count and period labels outside training/tuning, evaluates per-person outputs with normalized video-first AvgMAE, and resamples source-connected components. The partial-cache GT-bbox-assisted claim ceiling is unchanged.

### 2. Method Specificity - 8.2/10

The mathematical treatment is strong. Raw principal increments define a signed cell measure; the target is one unique float64 oriented-overlap integral; the warped branch retains its own sign; pause, reversal, support, aliasing, stop-gradient, and wrap are explicit. The full-identity recurrent graph and optimizer are coherent, and strict positive-interval NOLA removes zero-boundary ambiguity.

The score is held down because the selector and augmentor do not yet map uniquely to code. The seven-interface table identifies the exact omissions. Freeze those numerical choices rather than adding machinery. In particular, a single canonical selector path and g_j=1 is simpler and more auditable than inventing a confidence model. The warped pose kernel must use the same validity semantics as the integral. **Priority: CRITICAL LOCAL.**

### 3. Contribution Quality - 8.0/10

The contribution remains singular: a discrete local warp-phase integral objective evaluated by an identical deletion and mechanism controls. The selector, harmonic gate, K4 receipts, and identity diagnostic are non-contributions.

Contribution quality cannot rise while TWCRAC is unavailable in full. Its abstract overlaps pose, local temporal cycle consistency, periodicity, and non-stationary motion, but abstract-only evidence supports neither collision nor distinctness. K8 must remain blocked.

### 4. Frontier Leverage - 9.2/10

The proposal uses exact transformation supervision, deterministic provenance, causal deletion, component-aware uncertainty, and fail-closed gates where useful. A foundation model, LLM, VLM, diffusion model, router, or expert bank is not naturally relevant and would weaken attribution. No forced frontier component is recommended.

### 5. Feasibility - 8.0/10

The compute ledger and full-track optimizer are finite and coherent. The audited period schema is implementable, and staged failures save budget. Feasibility remains at 8 because several pre-run fixtures and tensor-building paths are not uniquely specified, the system is not implemented, and all-nine-component/selector gates may legitimately fail. The fixes are small code-contract freezes but must precede training.

### 6. Validation Focus - 8.0/10

The primary 5%-AND-CI endpoint is executable: normalized person error, video-first aggregation, seed aggregation, B=10000 paired nine-component resampling, multiplicity, and strict positive lower endpoint are all frozen. Secondary Period-mAP cannot rescue it, and three visible blocks remain focused.

Gate 2 and K4 are not yet executable as decisions. The 1,000-fixture population is mixture-dependent, K4 retention/shuffle/resampler predicates lack a complete numeric table, and g_j lacks a value. These are validation-contract gaps, not reasons to extend the validation matrix.

### 7. Venue Readiness - 7.4/10

The proposal is focused enough for an ICASSP 4+1-page story but not yet a reproducible preregistration or novelty-frozen submission plan. One local interface-freeze pass is required; the external K8 gate then remains. There is no implementation and zero eligible result, so any future claim must retain the pilot ceiling.

No dimension is below 7, so the rubric-mandated sub-7 repair list is empty. The local and external blockers below remain mandatory for READY.

## Focus, simplicity, and frontier assessment

- **Anchor:** verbatim preserved.
- **Dominant contribution:** one discrete local warp-integral loss.
- **Focus:** strong; gates and fixtures remain receipts, not contributions.
- **Simplicity:** conceptually strong; the remaining work is to choose deterministic interfaces, preferably fail-closed and constant-confidence.
- **Frontier appropriateness:** strong and unforced.
- **Local implementation status:** not closed; bounded numerical/tensor contracts remain.
- **Claim status:** independently blocked by TWCRAC K8.

## Simplification Opportunities

1. Use fail-closed landmark and source-span rules rather than adding learned imputation, box fallbacks, or a second selector.
2. Make one unjittered clean selector path authoritative; use nested views only for rejection, and set g_j=1 on valid intervals unless a fully specified existing confidence is essential.
3. Bind fixture bytes/hashes and K4 vector bytes/hashes instead of adding prose-heavy generator flexibility.
4. Keep the raw-sign, one-loss, one-base-learner, one-decode, three-block route unchanged.

## Modernization Opportunities

**NONE.** The method is already frontier-aware for the anchored signal-processing bottleneck. Forced modernization would not close any identified gap.

## Remaining action items

1. **CRITICAL LOCAL - preprocessing/selector:** Freeze missing-landmark root/scale handling; the retained-clock-to-256 FFT grid; FFT normalization/off-bin/endpoint rules; and one nested-view-to-training-selection mapping.
2. **CRITICAL LOCAL - decoder/fixtures:** Define g_j exactly, preferably as 1 for valid intervals, and freeze the complete 1,000-fixture population through exact counts/distributions plus generator or canonical-byte hashes.
3. **CRITICAL LOCAL - warp tensors:** Freeze x/y/confidence interpolation, mask propagation, endpoint/dtype rules, and pause/reversal behavior under the existing tau and invalid-cell protocol.
4. **CRITICAL LOCAL - K4:** Bind the two mask-code vectors and primary-arm q/L-code exposure, then restore an exact decision table. For positive G=M_control-M_treatment, define shortcut retention Q_k=(M_control-M_k)/G on the unclipped seed-averaged normalized video-first metric; give equally numeric thresholds for shuffle preservation and unseen-resampler erasure. Add no job, model, baseline, or block.
5. **CRITICAL EXTERNAL FOR CLAIM FREEZE:** Obtain the complete TWCRAC primary source and rerun independent residual-equivalence adjudication. Until an explicit pass exists, retain BLOCKED, NOT_PASSED_BLOCKED, claim_freeze=BLOCKED, and novelty_clearance=false.

## Drift Warning

**NONE.** Every required fix is an implementation definition inside the existing supplied-track learner, selector, augmentor, fixtures, decoder, or K4 receipt. This review adds no experiment, baseline, learner, endpoint, or contribution.

## Verdict

**REVISE**

The score remains 8.5 after deeper implementation audit: Round 3 closed its optimizer, period, and staging blockers, but the complete replacement still does not determine one selector/augmentor/K4 implementation. The route is focused and coherent, so RETHINK is unwarranted. READY is prohibited by the local interface gaps and, independently, the authoritative TWCRAC BLOCKED/NOT_PASSED_BLOCKED gate. This is a same-family provisional proposal review with no invented result.

## Audited input SHA-256

| Input | SHA-256 |
|---|---|
| refine-logs/round-3-refinement.md | 327d6db30aeedbb8bb30122227c9d081660cf4145b71c68db12114db7c879f17 |
| refine-logs/REFINE_STATE.json | 39152e46b06ab46b8d35e31110696692ec5b94fefafe708e38c4a5da8f277658 |
| refine-logs/score-history.md | 9b88843b885ffbd47ec8c69f3fbda537406d0b7113746717f94dece9c74a33f9 |
| PILOT_PERIOD_SCHEMA_ADDENDUM.md | 49edcfe25c302c2b55b8e65bc18e1cb7febe25cc8b76698d631fa24150d4b0b5 |
| idea-stage/TWCRAC_ADJUDICATION.json | 5a1e7b8c1048a8884690f2d234d413064d473ead1eb1deca69ab734ca2e5ed1e |
| refine-logs/round-3-review.md | 85ffbd8aaa84dbe65963f68027c9f916a9b7ca8b8b9911ff93b161174d4db2f2 |

