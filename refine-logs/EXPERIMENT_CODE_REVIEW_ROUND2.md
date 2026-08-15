# WARP-PHASE Experiment Code Review — Round 2

**Date:** 2026-08-15  
**Reviewer:** `gpt-5.6-sol` / reasoning `ultra`  
**Independence:** same-family, provisional  
**Verdict:** **FAIL**  
**Launch authorized:** **no**  
**Training authorized:** **no**  
**Evaluation/result/claim authorized:** **no**

## Decision

The implementation is fail-closed at its public command boundary and most isolated numerical primitives are careful and well tested. It nevertheless fails this code/contract review for two independent high-severity defects:

1. the frozen stochastic fixture serializer discards raw finite selector diagnostics that Amendment 001 requires it to retain; and
2. the packing API can open an evaluator label before the complete count-blind population and all-nine-component coverage table have been frozen.

These code findings are separate from the scientific result. Gate 2 independently and terminally fails all five thresholds with observed vector `246/55/144/50/66`. Passing unit tests, byte-consistent artifacts, or a code repair cannot reinterpret that vector. No training, server launch, result production, or claim is authorized.

The required disposition is to stop. This review does not authorize or implement a repair, a regenerated pack, an overwritten artifact, a later gate, or an alternate claim.

## Independence and scope boundary

I did not open or rely on any prior `EXPERIMENT_CODE_REVIEW` Markdown, JSON, trace, or reviewer response. Git status exposed filenames, and the required Amendment-001 acceptance review itself says earlier code-review blockers remain; that is the only incidental reference. I did not access protected test/sealed/held-out data, real evaluator data, server data, prior results, or the server. I did not run training or evaluation and did not modify source, tests, frozen artifacts, MANIFEST, Git, paper, or server state.

The complete 143-file SHA-256 map is in `.aris/traces/experiment-code-review-warp-phase-round2-20260815-sol/audited-input-hashes.json`; its canonical file-map digest is `ad2b85e0d538c538bed408076ca71f56e98f0b4546d3da1921925a0a88c67ddd`.

## Findings

### F001 — HIGH — computed stochastic diagnostics are serialized as unavailable

Amendment 001 requires raw finite ACF and score values where computed, explicit computed/valid masks, and preservation of diagnostic arrays even when the final decision fails (`refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001.md:99-112`). The finalized selector computes ACF, harmonic power, and a finite score before setting `score_valid` from `R(P) >= 0.25` (`src/pams/warp_phase/selector.py:359-374`).

The stochastic generator instead defines `valid_candidates = score_valid & finite_acf & finite_score` and writes ACF, score, and harmonics only under that predicate (`scripts/experiments/generate_warp_phase_fixture_pack.py:446-460`). Its stochastic schema has no `acf_computed` or `score_computed` masks. The ten-case Gate-1 verifier repeats the same masking (`src/pams/warp_phase/gates.py:1546-1564`), and the regression test supplies only a finite-valid candidate and a NaN-uncomputed candidate (`tests/test_warp_phase_gates.py:139-191`), so test/oracle agreement cannot detect the conflation.

Independent full-pack replay found:

| Measurement | Observed |
|---|---:|
| Canonical rows | 1,000 |
| Hard selector errors | 0 |
| Selected-period mismatches | 0 |
| Finite computed score entries | 41,605 |
| Finite computed entries with `R(P) < 0.25` | 34,499 |
| Nonzero scores serialized as zero | 34,499 |
| Nonzero ACF values serialized as zero | 34,499 |
| Nonzero harmonic values serialized as zero | 103,497 |
| Rows affected | 894 |

Candidate-v2 therefore has consistent bytes, hashes, and decisions, but its canonical diagnostic content does not satisfy the accepted amendment. This does not change its Gate-2 decision vector; it is a separate artifact-semantic failure.

### F002 — HIGH — the global pre-label firewall is not enforced by `pack_identity`

The governing population rule freezes eligibility before evaluator labels open and requires all nine original development components to remain nonempty (`refine-logs/FINAL_PROPOSAL.md:60-67`; `refine-logs/EXPERIMENT_EXECUTION_CONTRACT.md:270`).

`validate_development_component_coverage` is a standalone helper and does not prove complete-population membership, uniqueness, or a frozen manifest digest (`src/pams/warp_phase/packing.py:1184-1210`). `pack_identity` accepts only one identity's local eligibility decision and, after local/permission checks, invokes `vault_factory()` at `src/pams/warp_phase/packing.py:1289`. It requires no complete-population receipt or nine-component capability. The packing tests verify that an excluded identity and a failed ACL check do not open labels, but the successful path opens the vault for one identity without a global freeze (`tests/test_warp_phase_packing.py:211-355`).

Gate 0 is currently blocked and no typed v44 adapter calls this path, so it is not CLI-reachable today. The primitive is nevertheless unsafe for the future adapter it is intended to support.

### F003 — MEDIUM — canonical pack verification does not establish a live source chain

`verify_fixture_pack` validates the five receipt fields, 23 member hashes, schemas, aggregate, category schedule, and candidate array (`src/pams/warp_phase/gates.py:1395-1456`). It does not accept a repository root, compare the receipt's generator digest to the live generator, validate metadata fields against the live selector/environment, or cross-bind the separately checked fixture lock. Gate 1 then recomputes only ten rows (`src/pams/warp_phase/gates.py:1459-1614`).

There is no observed substitution in the current artifact: I independently obtained pack digest `c6dbe56df7b4f7d427ce1f51d5d818ad0e46dcfbf6e1550e0d77116435760196`, generator digest `7af1dac0ce246979d608b3db8e9fcaef9fe18dbfe2a5ecc803616e30919bb4ce`, selector digest `73b31b7d6c58a67efa8542df754971a58926d783258c9776269ec06765dafa0e`, and v2 fixture-lock digest `c278c77b27467f5928fa37d2fc79580dc58ec5d55cf98eee89f7e69e77d947a0`, all matching the candidate metadata and adjudication. The finding is that the gate code itself does not prove that chain.

### F004 — MEDIUM — Gate 1 treats an explicitly non-authorizing unit candidate as a satisfied pack check

The Amendment-001 review says no unit pack was created during that review and that any later pack must be independently reviewed (`refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001_REVIEW.md:42-46`). The current unit receipt says `authoritative=false`, `candidate_only=true`, `fresh_review_required=true`, `authorizes=[]`, and `status=CANDIDATE_NON_AUTHORIZING`.

The verifier requires those exact candidate fields (`src/pams/warp_phase/gates.py:1293-1306`) and then sets `selector_unit_fixture_pack=true` (`src/pams/warp_phase/gates.py:1695-1710`) without consuming a separate pack-review receipt. This round independently found no numerical discrepancy: all 64 NPY files, metadata, schemas, finite floats, live source hashes, three selected periods, and aggregate `406fe3934dd307accbd735dd6f2e54890d6de643c7bfe1d05d9f2d1131a8efcf` match. That provisional validation does not change the pack's authority fields or pass Gate 1.

### F005 — MEDIUM — Gate 3 computation is not dependency-gated

The contract permits CPU smoke only after Gates 0-2 pass (`refine-logs/EXPERIMENT_EXECUTION_CONTRACT.md:282`). The `gate3-cpu` command takes no prior receipts (`src/pams/warp_phase/cli.py:185-197`); `gate3_cpu_sanity_receipt` performs the model/optimizer update first and appends `gates_0_1_2_pass_receipts_missing` afterward (`src/pams/warp_phase/gates.py:1918-2044`).

The stored receipt replays exactly, remains `BLOCKED`, and authorizes nothing. No GPU job or training campaign ran. Persisting this computation as Gate 3 is nonetheless out of the contractual order.

### F006 — MEDIUM — result-bearing execution is incomplete and the snapshot is not committed

`TRAINING_AUTHORIZED` is hard-false; the train command can only persist a blocked preflight, and the end-to-end fresh-step factory is explicitly missing (`src/pams/warp_phase/training.py:30,75-140`; `src/pams/warp_phase/cli.py:200-229`). Gate 1 records missing duplicate/warp/recurrence/optimizer/decode/evaluator witnesses, and Gate 3 records the absent prediction schema (`src/pams/warp_phase/gates.py:1713-1715,2031-2036`). There is no WARP evaluator CLI, evaluator-only vault reader, or prediction-freeze transition.

The evaluator primitives are useful but not receipt-bound. In particular, the bootstrap derives component order with `sorted(set(...))` rather than consuming the frozen ordered component table/hash (`src/pams/warp_phase/evaluator.py:199-218`), which can change the finite seeded sample assignment.

Git HEAD is `236d4ad235c1cdd15398e7fdeac0d8c5b569570e`. The root CLI attachment and its tests are tracked modifications; the WARP package, config, scripts, WARP tests, and audit artifacts are untracked. HEAD alone cannot reproduce this reviewed source snapshot.

## Primitive and boundary assessment

| Surface | Assessment |
|---|---|
| Selector and FFT computation | Core equations, endpoint maxima, off-bin interpolation, parent stability, reversal, and harmonic margin pass; stored stochastic raw-diagnostic semantics fail F001. |
| Warp and signed overlap | Same-cell interpolation, float64 queries, pause/reversal orientation, invalid-support and alias guards are internally consistent in the tested primitive. |
| Losses | Circular PAMS and the sole treatment-only signed warp-integral loss use the declared masks, coverage, window, and stop-gradient behavior in isolation. |
| Model | Exact 225,026 trainable parameters; no clock/key/length input; identity-private zero-state traversal and invalid-frame holds pass. |
| Optimizer | AdamW `3e-4`/`1e-4`, four fresh backwards, one step, 32 identity draws, clipping at 1.0, and finite-gradient guards pass the primitive tests. |
| NOLA/decode | Strictly positive interval weights, stride/final window schedule, support-only denominator, one full-track decode, continuous primary count, and half-even secondary rounding pass. |
| Evaluator | Count, supplied-track period AP, harmonic, and K4 primitives are present; vault/prediction freeze, source-length-bound AP input, ordered component receipt, and CLI are incomplete. |
| Pack/vault ACL | POSIX distinct-principal/no-extended-ACL and atomic no-replace design is careful; Windows correctly fails closed. Global label sequencing still fails F002. |
| CLI/exit codes | Root `warp-phase` attachment is reachable. `BLOCKED`/`FAIL` gate and train receipts exit 1 after persistence; static preflight exits 0 only as explicitly non-authorizing. No server-launch/result command exists. |

## Gate and artifact adjudication

| Item | Status | Independent result |
|---|---|---|
| Gate 0 | `BLOCKED` | Source files absent locally; typed adapter, pack/vault/audit population, nine-component manifest, and import-isolation receipt absent. |
| Gate 1 | `BLOCKED` | Saved receipt replays byte-for-byte; complete numerical witness set absent; F001/F004 additionally prevent acceptance. |
| Gate 2 | `FAIL` | Saved receipt replays byte-for-byte; all five thresholds fail at `246/55/144/50/66`. |
| Gate 3 CPU | `BLOCKED` | Saved receipt replays byte-for-byte; CPU checks pass but prior gates, prediction schema, and tiny-GPU receipts are absent; execution was out of order. |
| Candidate-v2 bytes/provenance | Hash PASS / semantic FAIL | All 23 members, aggregate, live source hashes, lock/receipt repair, and canonical receipt match; diagnostics violate Amendment 001. |
| Selector-unit candidate | Provisional numerical PASS / authority WARN | All 65 pack members and formulas match; receipt remains non-authorizing and fresh-review-required. |
| Server | `NOT_READY` | Authorization count 0; server lock pending; fresh follows-doc and exact output-space receipt absent; mount/swap risks remain. |
| K8 | `BLOCKED / NOT PASSED` | Novelty claim freeze remains independently blocked. |

## Validation runs

All commands used CPython 3.12.13, NumPy 1.26.4, and bytecode/cache suppression where supported.

- WARP test suite: `96 passed in 11.51s`.
- WARP-related root CLI tests: `7 passed, 38 deselected in 8.45s`.
- Ruff scoped check: `All checks passed!`.
- mypy scoped check: `Success: no issues found in 17 source files`.
- Candidate-v2: 23/23 member hashes and schemas, finite floats, aggregate, live generator, live selector, and v2 lock all matched.
- Unit pack: 64 NPY files plus metadata matched their receipt and aggregate.
- Gate1/Gate2/Gate3 receipt functions reproduced exact saved bytes with SHA-256 values `160d21b2...d469a`, `609b2c57...ec03`, and `157c506f...e54e`.

Passing tests establish code hygiene and the tested primitive invariants. They do not clear F001/F002, pass Gate 2, authorize Gate 3, authorize launch/training/evaluation, or freeze any claim.

## Final authorization statement

`launch_authorized = false`. There is no authority for a server launch, local or remote training, protected-data evaluation, result artifact, paper update, scientific claim, pack overwrite, repaired candidate, or later-stage continuation. The exact machine-readable verdict and evidence are in `refine-logs/EXPERIMENT_CODE_REVIEW_ROUND2.json`.
