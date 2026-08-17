# Round 5 Final Research-Refine Re-evaluation: WARP-PHASE

**Reviewer:** GPT-5.6-Sol (/root/research_refine_reviewer)  
**Review continuity:** same reviewer thread as Rounds 1-4  
**Review independence:** same-family  
**Acceptance status:** provisional  
**Round status:** MAX_ROUNDS final research-refine review  
**Evidence status:** proposal-only; zero eligible method results  
**Venue target:** ICASSP 2027 (4+1 pages)

## Executive decision

Round 4 materially improves the proposal and closes most of the seven named interfaces. COCO17 root/scale handling is fail-closed and unique; retained-clock supports map to one inclusive 256-sample source-clock grid; ACF/FFT normalization, off-bin interpolation, endpoint maxima, and ties are specified; g_j is exactly binary; warp pose/confidence/mask interpolation is deterministic; K4 primary-arm exposure, fixed vectors, and ratio thresholds are explicit; and the desensitized packer schema is bound to exact bytes. Independent recomputation confirms the claimed 617-byte packer fixture SHA-256 and the claimed 4,096-byte K4 vector SHA-256.

The local specification is nevertheless not fully closed. Three bounded execution contracts remain underdetermined. First, the real-track weak rejection view is only called “seeded”; no seed-to-track/parent mapping or deterministic traversal is frozen, even though its output can reject a canonical selection. Second, the 1,000-fixture section fixes category counts and serialization but not one complete generator: the RNG position of the second/third-harmonic draws, their exact x/y formulas, symmetric category coefficients, stochastic array shapes/order, and some category transforms are not unique. It also mandates NumPy 2.4.6 and fails any mismatch, while the project lock, supported server environment, pose constraint, and local .venv resolve NumPy 1.26.4. A dedicated 2.4.6 fixture environment could be valid, but none is locked or witnessed in the proposal. Third, K4 leaves the denominator of f_track and the actual matched-time shuffle permutation and unseen-resampler kernel/schedule unspecified; the decision ratios are numeric, but the inputs to two ratios are not uniquely constructible.

These are local constants, formulas, and environment receipts only. They do not justify a new model, experiment, baseline, endpoint, or validation block. Separately, TWCRAC remains an external claim blocker: the supplied adjudication has zero full-text pages, verdict BLOCKED, k8.decision=NOT_PASSED_BLOCKED, claim_freeze=BLOCKED, and novelty_clearance=false. No abstract-level absence inference is permitted.

**LOCAL_SPEC_STATUS: NOT_CLOSED - THREE_BOUNDED_EXECUTION_BLOCKERS**  
**EXTERNAL_K8_STATUS: BLOCKED / NOT_PASSED_BLOCKED**  
**MAX_ROUNDS_STATUS: FINAL_RESEARCH_REFINE_VERDICT**

## Anchor verification

**PASS.** The two Problem Anchor copies in refine-logs/round-4-refinement.md are byte-for-text identical to each other and to the original frozen anchor. The anchor-text SHA-256 is 619148c765175167da093d9b054bd2b43fae0e5e97d981fb9748ef57f4d7dba7. The supplied-track population, video-first estimand, label restrictions, comparator family, secondary diagnostics, excluded questions, and pilot claim ceiling are unchanged. There is no predicted-track, scene-total, end-to-end, annotation-free, generic representation-learning, or historical-result drift.

## Seven-interface and packer closure audit

| Interface | Final status | Finding |
|---|---|---|
| COCO17 root/scale | **CLOSED** | Shoulders 5/6 and hips 11/12 are named. Root and shoulder means use available valid members; no hip invalidates a frame; no positive finite track-scale sample invokes K7; the float64 median and 1e-3 floor are unique; boxes, temporal carry, learned imputation, population statistics, and evaluator fallbacks are forbidden. |
| FFT256 source grid | **CLOSED** | Every canonical/rejection support uses 256 inclusive float64 source-clock queries with Delta=(q_last-q_first)/255. Same-cell validity, internal-breakpoint/right-cell assignment, final-endpoint/left-cell assignment, source-frame period units, and no truncation/padding/alternate branch are explicit. |
| FFT/ACF bins, normalization, maxima, and ties | **CLOSED** | The ACF pair query and support count, valid-mask mean, periodic Hann, rfft norm, mask-energy denominator, bins 1-127, zero/Nyquist exclusion, off-bin linear interpolation, harmonic weighting, endpoint one-sided maxima, interior two-sided maxima, and exact-score/smaller-P tie rule determine one scorer. The three numerical fixtures cover off-bin and both endpoints. |
| Canonical view and g_j | **PARTIAL / BLOCKING** | The unjittered parent is authoritative; nested 64 supports and one weak view reject only; the track value is a float64 median; g_j is exactly 1 on valid intervals and 0 otherwise. However, the real-track weak view has no RNG seed, per-track/parent derivation, traversal order, or stored perturbation hash. Because weak-view disagreement can invoke K5, different valid implementations can retain different parents. |
| Exact 1,000-fixture generator | **PARTIAL / BLOCKING** | Nine mutually exclusive ID ranges sum to 1,000 and exactly 250 are symmetric; PCG64 seed, high-exclusive integers, missingness thresholds, canonical serialization, filename ordering, source/pack receipts, and thresholds are stated. The prose does not place the second/third-harmonic coefficient draws in the declared RNG order or give complete coordinate formulas for those terms and the symmetric half/double templates. It also omits exact shapes/order for missingness/jitter/dropout arrays and leaves parts of pause/three-segment composition to interpretation. No current source or pack hash can resolve those choices. |
| Warp tensor/mask interpolation | **CLOSED** | Endpoint-inclusive support, exact stored-clock copying, same-cell float64 linear x/y/confidence interpolation, both-endpoint joint validity, mask-before-zero, float32 ties-to-even, frame/cell masks, pause, reversal, conflicts, and out-of-support behavior are unique and shared by model, recurrence, integral, loss, and diagnostics. |
| K4 vectors, exposure, and ratios | **PARTIAL / BLOCKING** | Primary treatment/control receive no fixed code; only Stage-B controls do. u_q, u_L, the PCG64 Rademacher/MGS construction, orientation, byte order, literal length/hash, Q_clock, Q_mask, P_shuffle, P_unseen, and inequalities are exact. Recomputed vector bytes match SHA-256 5e2f35a11bd8a376e44d5ad7d3e5067a0bb5a03e2d61edd84f5f5bfd6cf351ec under both NumPy 2.4.6 and the locked 1.26.4. But f_track does not name its denominator, and neither the matched-time permutation nor the held-out unseen-resampler kernel/schedule is defined, so M_c^sh, M_t^sh, M_c^u, and M_t^u are not uniquely reproducible. |
| Desensitized packer schema precondition | **CLOSED AS A PROSPECTIVE GATE-0 CONTRACT** | The single UTF-8/no-BOM/no-newline canonical JSON is exactly 617 bytes and independently hashes to 31f81549f9c5dbedc6ddb278e868793246826a234de90c06c169c71387428275. It separates feature, vault, and annotation schemas, prohibits real identifiers and cross-side visibility, rejects extra/executable fields, and requires hashes before eligibility. It is a precondition, not evidence that the packer or permissions already exist. |

## Environment feasibility audit

The proposal makes NumPy 2.4.6 normative for fixture generation and says any version mismatch fails. The repository evidence instead freezes NumPy 1.26.4 in uv.lock and docker/server/requirements.txt; the pose extra constrains NumPy below 2; and the supported local .venv reports 1.26.4. The broad base dependency allowing NumPy below 3 does not override those concrete locks.

This is not a reason to change the scientific route. It is a minimal environment choice that must be made before implementation:

1. Prefer the already provisioned path: make NumPy 1.26.4 normative for the fixture generator, freeze its generator source and pack hash, and retain the K4 vector hash, which was independently reproduced unchanged under 1.26.4.
2. Alternatively, define a dedicated non-training fixture/receipt environment pinned to NumPy 2.4.6 with Python/platform/wheel hashes, prove it provisions independently of the pose/training environment, and add a load-and-hash witness showing its canonical JSON/NPY/raw-vector artifacts are consumed unchanged by the locked 1.26.4 training/evaluator environment.

Until one path has an environment lock and witness, the current “version mismatch fails” clause makes the supported implementation fail by construction.

## Preserved technical audit

| Item | Status | Finding |
|---|---|---|
| Discrete signed warp law | **CLOSED** | Raw clean and warped increments retain their signs. The unique float64 oriented-overlap integral is the sole target, the clean teacher is stop-gradient, and support/alias failures are fail-closed. Pause and reversal require no canonical orientation. |
| Phase/fundamental claim boundary | **CLOSED PROSPECTIVELY** | A selector-defined pseudo-cycle is only an internal unit. The frozen evaluator-only 0.5x/1x/2x gate runs after prediction freeze, rejects only, and cannot correct counts or claim semantic identity. |
| Positive interval NOLA and recurrence | **CLOSED** | Interval weights are strictly positive, D_j must be finite and at least 1e-3, one decode follows one complete identity pass, recurrence commits once per real clock, and no graph/cache crosses a training step. |
| Optimizer and budget | **CLOSED** | Eight complete identities per fresh step, four-step gradient accumulation, one backward per step, 20,000 updates, and 640,000 track draws are coherent. The staged maximum remains 59 jobs and 552 GPU-hours. |
| Population and evaluator period | **CLOSED AS PRECONDITIONS** | Duplicate conflicts fail rather than average; eligibility is count-blind; all nine source components must remain nonempty; raw [s,e) periods map uniquely to a float64 median. Passage is not inferred. |
| Primary endpoint and bootstrap | **CLOSED** | K1 is the 5% relative normalized video-first AvgMAE improvement AND an absolute paired-effect CI lower endpoint above zero. B=10,000 paired PCG64 draws resample all nine components with multiplicity and linear central quantiles. |
| Focus and presentation | **CLOSED** | One 225,026-parameter base, one treatment deletion, three visible blocks, no new baseline, and zero eligible results remain explicit. |

## Scores

| Dimension | Weight | Score / 10 | Weighted contribution |
|---|---:|---:|---:|
| Problem Fidelity | 15% | 9.9 | 1.485 |
| Method Specificity | 25% | 8.8 | 2.200 |
| Contribution Quality | 25% | 8.0 | 2.000 |
| Frontier Leverage | 15% | 9.2 | 1.380 |
| Feasibility | 10% | 8.2 | 0.820 |
| Validation Focus | 5% | 8.6 | 0.430 |
| Venue Readiness | 5% | 7.7 | 0.385 |

**WEIGHTED COMPOSITE: 8.700 / 10 (reported: 8.7 / 10)**  
**CALIBRATION: none**

**GAP:** No curated three-good/three-bad proposal anchors were supplied, so calibration remains none and no exemplar comparison is fabricated. Relative to the READY bar, the proposal now has a highly specific core: exact signed warp math, phase-claim boundaries, full-identity optimization, NOLA, source-clock FFT/ACF, packer and K4 byte receipts, normalized K1, component bootstrap, staged controls, budget, and kill rules. It remains locally short of one program because a rejection-view RNG, the complete synthetic generator/environment, and two K4 corruption constructors are not frozen. More importantly for any paper claim, TWCRAC method coverage remains zero pages; contribution distinctness and claim freeze are externally unadjudicated. No result, Gate 0-5 passage, or novelty absence is inferred.

## Dimension review

### 1. Problem Fidelity - 9.9/10

The complete replacement preserves the anchored supplied-track tempo-drift question, video-first unit, source-component uncertainty, label restrictions, matched deletion, comparator family, and pilot-only ceiling. The new constants close implementation interfaces without changing the population, estimand, contribution, or success rule.

### 2. Method Specificity - 8.8/10

The method core is unusually explicit. Root normalization, continuous source-clock velocity, uniform FFT queries, exact ACF/FFT score, raw signed cell measure, oriented pullback integral, warp tensor construction, full-sequence recurrence, NOLA, and K1 scorer now determine almost all numerical behavior.

The score stays below 9 because three decision-bearing input generators remain non-unique: the real weak rejection view, parts of the synthetic fixture generator, and the K4 shuffle/unseen corruptions. The environment clause also conflicts with the supported lock. These are important implementation contracts, not alternative scientific methods. **Priority: CRITICAL LOCAL.**

### 3. Contribution Quality - 8.0/10

The contribution is focused and appropriately narrow: one discrete warp-integral equivariance objective tested by an identical deletion and staged falsification. ACF/FFT, pseudo-cycle selection, private state, NOLA, K4, and identity isolation remain non-contributions.

Quality is capped by K8. The available TWCRAC abstract overlaps pose, local periodicity, temporal cycle consistency, and non-stationary counting; none of those is claimed as residual novelty. But without the complete method, the signed local warp objective cannot be declared distinct or colliding.

### 4. Frontier Leverage - 9.2/10

The proposal uses current scientific-control ideas where natural: exact transformation laws, causal deletion, deterministic provenance, component-aware uncertainty, and fail-closed receipts. It correctly avoids forcing an LLM, VLM, diffusion system, router, or expert bank into a signal-processing mechanism test.

### 5. Feasibility - 8.2/10

The optimizer, tensor shapes, warp kernel, compute ledger, and staged stops are feasible. The packer and period interfaces are implementable as gates. Feasibility is limited by the unimplemented system, severe prospective gates, incomplete generators, and the NumPy 2.4.6 mandate conflicting with the supported 1.26.4 lock. A dedicated generator environment is acceptable only after a concrete lock and cross-environment witness.

### 6. Validation Focus - 8.6/10

K1 and its uncertainty endpoint are fully executable and cannot be rescued by secondary Period-mAP. The validation story remains three blocks, with integrity and numerical fixtures kept as receipts. The score loses only because K5 can depend on an unspecified weak-view perturbation, and two K4 metrics depend on unspecified corruptions. No additional validation family is needed.

### 7. Venue Readiness - 7.7/10

The local route is close to a reproducible ICASSP preregistration, but it is not implementation-ready or novelty-frozen. The three local contracts can be closed without changing the story. TWCRAC remains a decisive external venue/claim risk, and there are zero eligible results.

No dimension is below 7, so the rubric-mandated sub-7 fix list is empty. The blockers below remain mandatory despite that.

## Focus, simplicity, and frontier assessment

- **Anchor:** verbatim preserved.
- **Dominant contribution:** one discrete local warp-integral objective.
- **Focus:** strong; fixtures and gates remain receipts rather than claims.
- **Simplicity:** strong; the remaining fixes are seed mapping, literal formulas, environment locking, and corruption constants.
- **Frontier appropriateness:** strong and unforced.
- **Local specification:** not closed.
- **Claim freeze:** externally blocked by K8.
- **Evidence:** proposal-only, zero eligible results.

## Simplification Opportunities

1. Derive every real weak rejection view from one frozen Gate-2 seed stream over a checksum-sorted eligibility/parent-start order, and store the perturbation hashes. Do not add a second view or confidence model.
2. Use the already supported NumPy 1.26.4 environment for all fixture/reference bytes unless a separate 2.4.6 environment is explicitly locked and witnessed.
3. Make f_track one named fraction and bind one deterministic shuffle plus one held-out resampler kernel/schedule. Keep the existing four K4 ratios and six Stage-B jobs.
4. Preserve the one-loss, raw-sign, one-base, one-decode, three-block design.

## Modernization Opportunities

**NONE.** No frontier component is missing, and forced modernization would not close the identified risks.

## Exact remaining risks and minimal actions

1. **CRITICAL LOCAL - weak rejection view:** Freeze the Gate-2 RNG algorithm/seed, checksum-sorted track order, parent-start order, draw shapes/order, and stored perturbation hash. The opaque key may route to the frozen order but must not become a model feature.
2. **CRITICAL LOCAL - fixture generator and environment:** Put harmonic-coefficient draws at an exact RNG position; give complete x/y equations for second/third harmonics and symmetric half/double templates; specify all stochastic array shapes and the pause/three-segment transform order. Then either make locked NumPy 1.26.4 normative and hash the generator/pack, or provide a pinned 2.4.6 dedicated-environment plus 1.26.4 consumption witness.
3. **CRITICAL LOCAL - K4 construction:** Define f_track with an exact numerator and denominator over named collapsed/non-padded masks. Freeze the matched-time shuffle scope/permutation/seed and the unseen-resampler kernel, target grid, schedule, seed, and receipt. Keep Q_clock, Q_mask, P_shuffle, P_unseen, thresholds, jobs, and blocks unchanged.
4. **CRITICAL EXTERNAL FOR CLAIM FREEZE:** Obtain the complete TWCRAC primary source, hash and inspect every method/loss/training/evaluation page, and rerun independent residual-equivalence adjudication. Until an explicit pass exists, retain BLOCKED, NOT_PASSED_BLOCKED, claim_freeze=BLOCKED, and novelty_clearance=false.
5. **PROSPECTIVE EXECUTION RISK:** Even after specification closure, Gate 0 population/permission checks, Gate 1 bytes/numerics, Gate 2 selector stability, Gate 3 gradients, K1 efficacy/CI, K2 comparators, K3 redundancy, K4 shortcuts, K5 phase execution, K6 integrity, and K7 coverage can all fail. None has run.

## Drift Warning

**NONE.** Every local action is a deterministic definition inside the existing selector, fixture builder, or K4 diagnostic. This review adds no model, baseline, job, endpoint, validation block, or contribution.

## Final research-refine verdict

**REVISE**

This is the MAX_ROUNDS final research-refine verdict. The proposal rises from 8.5 to 8.7 because four interfaces and the packer precondition are now genuinely closed and the core method is substantially executable. RETHINK is unwarranted: the scientific route remains focused. READY is prohibited twice over: three bounded local execution contracts remain open, and the independent TWCRAC K8 gate remains BLOCKED/NOT_PASSED_BLOCKED. No final proposal, report, experiment plan, or efficacy claim is generated by this review; the main thread must carry forward the exact risks above.

## Audited input SHA-256

| Input | SHA-256 |
|---|---|
| refine-logs/round-4-refinement.md | 2031d5c84b8c6c329dd0e43e5b8a09210ed9a3e8b17b47578f2f878a4c2b4e0f |
| refine-logs/round-4-review.md | c1d44bbe3dd5467b02c4d29bf7bba2849b5bcf64a705c8276f0882a072a2e7a1 |
| refine-logs/round-4-review.json | d62ec17f33d7f76091a20ebeaaaa23c0a402b5c028487ad1277311b4dc6363ad |
| refine-logs/REFINE_STATE.json | 54521958169008256110c90f11d8dad68389895920688550a474f8ad1c4727c6 |
| refine-logs/score-history.md | bfdebeda8851c2ca08f4e7ed7601f38ccb70b0c7fc16fcf09ee0c3ca835b5f1d |
| PILOT_PERIOD_SCHEMA_ADDENDUM.md | 49edcfe25c302c2b55b8e65bc18e1cb7febe25cc8b76698d631fa24150d4b0b5 |
| idea-stage/TWCRAC_ADJUDICATION.json | 5a1e7b8c1048a8884690f2d234d413064d473ead1eb1deca69ab734ca2e5ed1e |
| pyproject.toml | 14b3b35bbfa57fc982e24ef2f999df288893c0f0d12a53a9f82120babefead38 |
| uv.lock | 5a964f1cd841948eae09ef253098b0b16eb5d2e6bd660e2d3496b389c4dc72a7 |
| docker/server/requirements.txt | 2337391c20fae1533b6e30d042f668a5542db28f0a29a71d5cb86b74517bfc53 |
| .venv/pyvenv.cfg | 598ae04c323c0a6ec8b1e17e5c99677d255cd37a8a82e746bc954e3113bfe2b7 |

