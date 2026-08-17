# TempoRAC Experiment Plan Review — Round 2

**Verdict:** `PASS`  
**P00 implementation release readiness:** `SUITABLE_UNDER_PLAN`  
**Acceptance status:** `same-family provisional`  
**Reviewer model/family:** GPT-5.6-Sol / OpenAI  
**Review date:** 2026-08-16  
**Authority boundary:** this PASS means the plan/tracker are suitable for experiment-bridge to plan and, only under separate explicit authority, release `P00-IMPLEMENT`. It grants no implementation authority by itself and no server connection, data access, training launch, evaluator capability, result claim, paper promotion, or Git authority.

## Resolution verdict

Both Round‑1 implementation-contract blockers are fully resolved. The updated plan and tracker now provide an executable P00/P2 specification for pilot-scope metadata and the pre-decision count-metric fixture, while preserving the frozen TempoRAC graph, jobs, resources, order, firewall, and zero-execution posture. No method redesign is present or requested.

## Round‑1 blocker resolution

| Blocker | Verdict | Verified resolution |
|---|---|---|
| B1 — conditional-pilot scope and promotion guard | PASS | The exact canonical name `GT-bbox-assisted AlphaPose supplied-track pilot` and all three labels—`partial-cache`, `GT-bbox-assisted`, `supplied-track`—are frozen in the plan/tracker headers, natural-data clauses, artifact-scope policy, NATP rows, K7 reporting boundary, and stop codes. The population is explicitly 110 train videos / 51 val videos and 268 train tracks / 134 val tracks. Current outputs are appendix/pilot-only; `PAPER-PROMOTION-GATE` remains `BLOCKED_PARTIAL_CACHE_APPENDIX_ONLY` even after K7 PASS and can only be reconsidered under a future complete-cache protocol plus separate review/clearance. Closed proposal receipt schemas receive no extra keys. |
| B2 — pre-decision count-metric fixture and receipt chain | PASS | `P2-METRIC` / `P05M-COUNT-METRIC` is a finite, zero-GPU, desensitized fixture. Normal cases cover positive integer GT, stub estimate zero, per-person NAE/OBO, people-within-video, equal-video, ordered-seed aggregation, component bootstrap multiplicity, per-draw comparator reselection, and paired-clean arm reuse. Attack cases cover malformed, duplicate, empty, nonfinite, and type failures with global fail-closed behavior. The surrounding receipt binds exact evaluator implementation bytes plus fixture input, expected-output, and attack-global-fail hashes. Its dependency is explicit through P2/P05M → P3 → S0 → G5a closed `evaluator_code_sha256` equality → K7 pre-start verification/refusal. Missing or mismatched evidence blocks K7 before start. No real evaluator label or source identity enters the fixture. |

## Unchanged invariant audit

| Requirement | Verdict | Independent recomputation |
|---|---|---|
| Problem and graph fidelity | PASS | Identity-private supplied tracks, one shared encoder, matched slow/medium/fast response branches, detached cue-only local routing, probability fusion, exact positive NOLA, and one fixed connected-component decode remain unchanged. |
| Claims / blocks / baseline families | PASS | 1 primary claim, 4 experiment blocks, 3 baseline families. The three prospective paper targets do not authorize current partial-cache promotion. |
| Exact training inventory | PASS | 27 unique exact job names: 3 teacher + 3 canonical + 3 capacity-control + 18 shortcut, with exactly seeds 20260815/20260816/20260817. P05M is not a training job. |
| Compute / memory / disk | PASS | `3*6 + 24*2.5 + 3*1 + 3*4 = 93` A6000-hours; 24 GiB CUDA cap per training job; `4+8+8+6+20+12+2+4 = 64` GiB disk; maximum concurrency 2; 7-hour margin locked. |
| Order | PASS | Exact `P0 -> P1 -> P2 -> P3 -> S0`, then the full F23 order through `G5a -> capability grant -> G5b -> K7`. P05M is a P2 subfixture and is bound before P3/S0. |
| Label/data firewall and splits | PASS | Natural access remains frozen v44 train/val only; physical feature/vault separation, privileged-key rejection, X0-only selection, no natural-val checkpoint selection, single-use evaluator process, and test/sealed/historical path denials remain intact. |
| Server/launch authority | PASS | `launch authorization = 0`; all data-bearing/GPU/capability work remains blocked. Fresh TempoRAC preflight and separate authorization remain future requirements. |
| Tracker completeness | PASS | Scope and metric stop codes, dependencies, inputs, outputs, statuses, NATP rows, G5a/K7 bindings, promotion gate, resources, and all 27 training rows are present. |
| No WARP drift | PASS | WARP-PHASE remains isolated and non-inheritable; no WARP object, preflight, environment, budget, or authority is reused. |
| No fabricated result | PASS | There are zero executed PASS rows, zero consumed compute, and no metric, artifact result, or positive performance claim. `P00-IMPLEMENT` remains `NEXT_NO_AUTHORITY`; all downstream work is `BLOCKED`. |

## Recomputed bindings

| Input | SHA-256 |
|---|---|
| `refine-logs/FINAL_PROPOSAL.md` | `562325dedab0637421f62dec5ad149b3c884f67b47c4b525c8fb54a57bb1233c` |
| `refine-logs/EXPERIMENT_PLAN.md` | `3fd1986871b32674d74b395fbfc9720712d9fba0c8583c24c1d5bddb87086eb6` |
| `refine-logs/EXPERIMENT_TRACKER.md` | `c7e2f32c3c81b66884abb13760a671ecc7cdd314fd6c994a4ed4c46cb754a63b` |
| `refine-logs/EXPERIMENT_PLAN_REVIEW.md` | `1b1707c0792cfc6861bdb87ce8c95fb8ce00a44bb86375f25463bb1ecf2ceb01` |
| `refine-logs/EXPERIMENT_PLAN_REVIEW.json` | `1e1ff5872bf00144cc2260d78cb6fd860894a0b889ebc9d7a77cf19d847d166c` |

The fixed plan/tracker files are byte-identical to their referenced `20260816_044644` timestamped versions.

## Final ruling

`PASS — ROUND_1_BLOCKERS_RESOLVED`.

The current plan and tracker are now suitable for experiment-bridge P00 implementation planning. Any actual P00 execution still requires separate explicit authorization. This review does not authorize server or data access, GPU work, capability use, result claims, or paper promotion.
