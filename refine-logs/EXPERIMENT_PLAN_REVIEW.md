# TempoRAC Experiment Plan Review

**Verdict:** `FAIL`  
**Acceptance status:** `same-family provisional`  
**Reviewer model/family:** GPT-5.6-Sol / OpenAI  
**Review date:** 2026-08-16  
**Execution authority:** none; this review authorizes no implementation, data access, server connection, launch, capability request, evaluation, paper edit, or Git operation.

## Bottom line

The plan faithfully preserves the TempoRAC problem and graph, and its compactness, job inventory, resource ledger, gate order, label firewall, split policy, and zero-result posture all recompute correctly. It cannot yet release `P00-IMPLEMENT` because two pilot-audit requirements are absent from the plan/tracker: the mandatory conditional-pilot scope/labeling boundary for artifacts produced by P00 and later, and an executable pre-decision fixture for the evaluator count metrics and their aggregation. Both gaps are implementation-contract omissions; this review does not reopen or redesign the method.

## Independent audit matrix

| Requirement | Verdict | Independent finding |
|---|---|---|
| Problem and graph fidelity | PASS | The plan retains externally supplied person-indexed tracks, identity-private state, one shared encoder, three matched slow/medium/fast branches, detached cue-only local routing, probability fusion, exact positive NOLA, and exactly one threshold-0.5 component decode per identity. No replacement graph is introduced. |
| Proposal binding | PASS | The declared proposal hash `562325dedab0637421f62dec5ad149b3c884f67b47c4b525c8fb54a57bb1233c` matches the current root proposal; the root and `refine-logs/temporac/FINAL_PROPOSAL.md` are byte-identical. |
| Claims / blocks / baseline families | PASS | Recomputed inventory is 1 primary claim, 4 experiment blocks, and 3 baseline families. Exactly 3 blocks are marked paper-visible. This satisfies the numerical limits of at most 2 claims, 5 blocks, and 3 families. |
| Training jobs | PASS | The tracker contains exactly 27 unique exact job names: 3 teacher, 3 canonical, 3 capacity-control, and 18 shortcut jobs, over exactly seeds `20260815`, `20260816`, and `20260817`. Global, uniform, blocked, and shuffled remain inference modes. |
| Compute / memory / disk | PASS | Recomputed allocation is `3*6 + 24*2.5 + 3*1 + 3*4 = 93` A6000-hours. Per-training-job CUDA cap is 24 GiB. Disk classes sum to `4+8+8+6+20+12+2+4 = 64` GiB. Maximum concurrency is 2 and the 7-hour outer margin is locked. |
| Order | PASS | The plan explicitly enforces `P0 -> P1 -> P2 -> P3 -> S0`, then the full F23 order through `G5a -> capability grant -> G5b -> K7`. The tracker expands P2 into finite fixture rows without moving any data-bearing step before S0/G0. |
| Label/data firewall | PASS | Human count, period, density, boundary, bbox, mapping, identity/provenance, and evaluator outputs are excluded from learner deserialization, tensors, objectives, checkpoint selection, prediction, and reruns. Physical feature/vault separation and one-use evaluator capability are preserved. |
| Train/development only | PASS | Natural access is restricted to the two frozen v44 train/val sources. Test, sealed, official-complete, historical-result, and contaminated v4x/v46 paths remain forbidden. X0 train/tune/heldout are synthetic contract partitions, not external natural splits. |
| No server-launch authority | PASS | `launch authorization = 0` is explicit. The existing `SERVER_PREFLIGHT.md` is correctly recognized as WARP-PHASE-only and non-inheritable; a fresh TempoRAC preflight remains a separately authorized future prerequisite. |
| Receipt/gate executability | FAIL | The plan inherits the closed TempoRAC receipt schemas, but it omits the pilot audits' required pre-decision, equation-faithful count-metric/evaluator fixture. K7 therefore has a specified formula but no preregistered implementation acceptance witness before decision-bearing execution. |
| Tracker completeness | FAIL | The tracker is complete for jobs, gates, dependencies, resources, and blocked statuses, but it does not track the mandatory pilot-scope labels/main-paper ineligibility or the missing evaluator metric fixture and its receipt dependency. |
| No WARP drift | PASS | WARP-PHASE appears only in isolation, deny, and non-inheritance clauses. No WARP object, selector, environment, budget, preflight, or authority is reused. |
| No fabricated result | PASS | All gates and jobs are unrun/blocked, consumed compute is zero, no metric value is reported, and no positive performance conclusion is stated. |
| Current packaging metadata | PASS WITH REQUIRED FUTURE LOCK | `pyproject.toml` permits Python 3.12 and compatible NumPy/SciPy/PyTorch ranges but is not the exact TempoRAC runtime lock. The plan correctly requires a fresh exact environment lock at P3/S0; the broad package metadata must never be treated as that receipt. |

## Blocking fixes

### B1 — Restore the audited conditional-pilot scope in both plan and tracker

The current pilot audit's verdict is `CONDITIONAL_PILOT_ONLY`. It says the canonical protocol name is **GT-bbox-assisted AlphaPose supplied-track pilot**, that the partial converted cache is not eligible for the paper's main results, that every pilot artifact must remain marked `partial-cache`, `GT-bbox-assisted`, and `supplied-track`, and that main-paper evidence remains blocked pending a complete train/val pose cache and separate clearance.

The plan instead promotes B2 to a paper-visible main table on the current 134-development-identity partial cache, while neither plan nor tracker contains `partial-cache` or `GT-bbox-assisted`. That omission can cause an experiment bridge to generate wrongly scoped artifacts or paper-facing outputs even when the numerical experiment itself is correct.

Required repair:

1. Add the exact canonical protocol name and all three mandatory scope labels to the plan header, natural-data block, artifact policy, tracker header, NATP rows, and final-decision notes. Do not add unknown fields to the proposal's exact closed receipt schemas; carry the labels in permitted surrounding manifests, paths, tracker notes, and reports.
2. Mark the current v44 run as controlled pilot-only and not a paper-main-result authorization. Change B2's current-cache paper target from an unconditional main table to pilot/appendix evidence, or explicitly gate any main-paper promotion on a separately reviewed complete-cache protocol.
3. Add a tracker stop state/dependency that prevents paper-visible promotion when the population remains the partial 110-train-video/51-val-video, 268/134-track cache. This must not silently expand the current frozen dataset or add jobs.

### B2 — Add the missing pre-decision evaluator metric fixture and receipt

The data-schema audit requires a direct equation-faithful AvgMAE/AvgOBO scorer with people averaged within video before videos, and the period addendum requires the evaluator wrapper and normalized count fixture to pass before any decision-bearing run. The proposal defines the final NAE/OBO and clustered-bootstrap mathematics, but P0–P2, B1, the receipt table, and the tracker contain no explicit scorer fixture or acceptance receipt.

Required repair:

1. Add a finite desensitized P2 fixture (or a named subrow under the existing P2 fixture stage) that tests positive integer counts, zero-estimate stubs, per-person NAE/OBO, people-within-video averaging, equal-video averaging, ordered-seed averaging, component-resampling multiplicity, strongest-comparator reselection per drift draw, and reuse of that selected arm for paired clean.
2. Include malformed/duplicate/empty/nonfinite/type cases and verify global fail-closed behavior. The fixture must contain no real evaluator label or source identity.
3. Bind the exact evaluator implementation bytes and fixture expected-output hash into the P2 receipt, S0 commitment, G5a evaluator-code binding, and K7 input dependency. K7 must refuse execution on a missing or mismatched fixture receipt.
4. Add the fixture row, receipt, dependency, stop code, and status to the tracker. This is a zero-training-job, zero-GPU prerequisite and must not change the 27-job/93-hour/64-GiB inventory.

## Recomputed bindings

| Input | SHA-256 |
|---|---|
| `refine-logs/FINAL_PROPOSAL.md` | `562325dedab0637421f62dec5ad149b3c884f67b47c4b525c8fb54a57bb1233c` |
| `refine-logs/EXPERIMENT_PLAN.md` | `cd84c486f86ed41f2a714322d21fe94619c97db1f679cd0ffc3e0437d551329b` |
| `refine-logs/EXPERIMENT_TRACKER.md` | `37b650e5bd1e79b0333aa5f460ea0145c55a63ff428fa8b86ac2b71773cea4f2` |
| `refine-logs/temporac/round-5-review.json` | `6a32c0ecab12d29dcc5d2d27cc2110165f3b919126a314ec3f54b4c77ace4768` |
| `PILOT_DATA_SCHEMA_AUDIT.md` | `7058981e8eb37622c09f5808a4b1329bc7cd133f474101723f15e5e4273eab25` |
| `PILOT_DATA_SCHEMA_AUDIT.json` | `ba1c376127d0304d7ee76f902c56dd753c351b71fc11655c45451b98fa0481b8` |
| `PILOT_PERIOD_SCHEMA_ADDENDUM.md` | `49edcfe25c302c2b55b8e65bc18e1cb7febe25cc8b76698d631fa24150d4b0b5` |
| `PILOT_PERIOD_SCHEMA_ADDENDUM.json` | `2c95af393fa73f3c581a327f9ee8a5e8a5a5c69c03a9fe6349d3b6531c48e914` |
| `refine-logs/SERVER_PREFLIGHT.md` | `f3b254fe2ce7bf7f61fbc6bf3f416ef7e86f8230d1f484259b668ed23c66444b` |
| `pyproject.toml` | `14b3b35bbfa57fc982e24ef2f999df288893c0f0d12a53a9f82120babefead38` |

## Final ruling

`FAIL — BLOCKING_FIXES_REQUIRED`.

The plan is numerically and methodologically disciplined, but the two plan/tracker gaps above must be repaired and re-audited before experiment-bridge may release `P00-IMPLEMENT`. No method redesign is requested. No experiment, server action, data access, evaluator capability, result claim, or paper promotion is authorized by this review.
