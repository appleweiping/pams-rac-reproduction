# Fresh Review: Experiment Execution Contract Amendment 001

> **VERDICT: `ACCEPT`**
>
> **SAME-FAMILY PROVISIONAL REVIEW**
>
> **NORMATIVE CORRECTION ONLY — NO TRAINING, GATE PASSAGE, CANDIDATE FREEZE, RESULT, CLAIM, OR SERVER-LAUNCH AUTHORIZATION**

**Date:** 2026-08-15  
**Review scope:** `EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001` only  
**Amendment Markdown SHA-256:** `25fe82bbcad386b42e4c5a95a762d1f84e4c926796655f02ba5d2ab8ae7dc484`  
**Amendment JSON SHA-256:** `d5227f04af4452c2224d3a6b0a8342a7a02aa18edf6b5452e9e06290636bc1c9`

## Acceptance decision and exact normative effect

The amendment is accepted at the same-family provisional specification layer. Its exact effect is limited to the following:

1. The three deterministic selector witnesses are removed from the contradictory placement clause in `FINAL_PROPOSAL.md` Section 3.4 and the identical clause in `round-4-refinement.md` Section 3.4. They are a separate, label-free, RNG-free Gate-1 unit-fixture pack under `data/warp_phase_pilot_v1/audit/gate1/selector-unit-fixtures-v1/`; they are not rows of the stochastic 1,000-pack.
2. The amendment supplies the exact COCO17 constructor, member schemas, failure-diagnostic serialization, sole `-1` decision sentinel, receipt/hash rules, and required tests for those unit fixtures. For a failed final decision, computed diagnostics remain evidence; unavailable values use finite `+0.0` together with the declared computed/valid masks. This creates no selector fallback and changes no selected-period rule.
3. Candidate-v1, exactly `data/warp_phase_pilot_v1/audit/gate1/candidate-20260815-sol/`, is `REJECTED_NON_FREEZABLE`. Its bytes and failed diagnostics remain untouched historical failure evidence. The prior `PASS_FREEZE_CANDIDATE` recommendation is superseded only as a freeze disposition; it never authorized Gate 1, Gate 2, or training.

Nothing else is amended. In particular, the stochastic population, raw fixture inputs, category membership, RNG seed/draw order, Gate-2 membership and denominators, thresholds, selector score, model, loss, optimizer, jobs, GPU ceiling, staged order, three visible validation blocks, K1–K9, evidence ceiling, and claim ceiling remain unchanged. Acceptance does not create the future unit pack, create or freeze a later stochastic candidate, pass any gate, or authorize training.

## Independent constructor probes

The exact constructor was executed directly against the finalized selector in the normative fixture environment, CPython 3.12.13 with NumPy 2.4.6. The same probes were repeated in the consumption environment, CPython 3.12.13 with NumPy 1.26.4. Selected periods and the values below agreed; the small cross-version float differences were far below `rtol=atol=1e-12`.

| Fixture | Exact grid facts | Required comparison | Result |
|---|---|---|---|
| `offbin_delta1_p20` | span `255`; `Delta=255/255=1`; endpoints `0,255`; `P_max=127`; `k(20)=256/20=64/5=12.8`, hence stored bins 12 and 13; `fft_x` is byte-exact float64 `arange(256)` | `S(19)=0.5809198920484148 < S(20)=0.6594133070408238 > S(21)=0.5743810869954495` | selected `20` |
| `lower_endpoint_p4` | span `63`; `Delta=63/255=21/85`; endpoints `0,63`; `P_max=31`; `k(4)=1344/85=15.811764705882354` | lower one-sided check only: `S(4)=0.6345727420272262 >= S(5)=0.0005560660114877089` | selected `4` |
| `upper_endpoint_span126_p63` | span `126`; `Delta=126/255=42/85`; endpoints `0,126`; `P_max=63`; `k(63)=512/255=2.007843137254902` | upper one-sided check only: `S(63)=0.675905398984302 >= S(62)=0.6691573037802108` | selected `63` |

All three probes produced exact scale `2.0`, all 17 normalized joints valid at every clock, every feature frame valid, every velocity coordinate valid, every FFT coordinate mask true, finite float arrays, and the required local-maximum bit. The constructor made zero RNG calls and used no count, density, boundary, evaluator period, real track, model prediction, or other label/evaluator value. `P_src` is only a deterministic unit-test input. Repository search confirmed that the three fixture IDs and the unit-pack namespace occur nowhere outside the amendment/review artifacts, so they do not enter the current 1,000-row generator, pack, category counts, or Gate-2 aggregation.

The canonical stochastic row has different geometry: span `127`, `Delta=127/255`, and `k(P=20)=8128/1275`, so it cannot satisfy the separate `Delta=1`, `k=12.8` witness. This independently confirms the contradiction and the need for separation.

## Schema and failed-diagnostic review

All 21 per-fixture NPY schemas and `candidate_period.npy` were instantiated in memory from the exact constructor and finalized selector. Dtype, shape, C-contiguity, non-object status, finiteness, and NPY v2.0 serialization all passed. `candidate_period.npy` is exactly `<u2[125]` containing 4 through 128. The per-fixture present/computed masks correctly distinguish serialized zero fill from computed values: the present/computed counts are 124 for the 256-clock fixture, 28 for the 64-clock fixture, and 60 for the 127-clock fixture.

The aggregate member set is the explicitly bound `metadata.json` plus the 64 NPY members (one shared candidate array and 21 arrays for each of three fixtures), for 65 aggregate members; the receipt binds that digest and is not a self-hashed pack member. With the receipt beside those members, the future root has 66 files. No unit-pack artifact was created during this review.

The finalized selector returns `SupportSelection(period=None, score=None, diagnostics=...)` when diagnostics compute but no local maximum survives. On candidate-v1 row 10, an independent probe reproduced a failed decision while retaining a 256-point FFT grid, 1,052 true FFT-mask entries, 127 nonzero FFT powers, six finite ACF values, six finite scores, one score-valid candidate, and 18 nonzero harmonic values. The current selector/gate tests separately cover this no-decision path and the `-1` serialization contract. Hard precondition failures remain fail-closed and, under the amendment, must serialize the declared finite arrays/masks with a metadata failure stage/reason; no alternate decision sentinel is introduced.

This is a prospective schema acceptance, not a Gate-1 pass. The unit pack, its receipt, member hashes, and independent consumer recomputation do not yet exist and must be reviewed if later created under separate authority.

## Candidate-v1 disposition

Candidate-v1 was independently rehashed and structurally checked:

- candidate receipt SHA-256: `f5e03b431f47f2ae0c1e52f40955fbcb70b97642e52969178ffcfa3140f6c5ce`;
- metadata SHA-256: `9a9db9422ef920fff7d39ac8ea0541ae0d81280282a6b09d95205419f337f6f8`;
- aggregate pack SHA-256: `b2c791f9aac1d86c0849fa74b00b4e605fe21120f9d188f4ec0b304d7052f173`;
- all 23 receipt-bound pack-member hashes matched; all 22 NPY files were v2.0, C-order, schema-correct, non-object, and finite where required.

Its failure is unchanged and independently reproduced: canonical/weak agreement `246/1000`; overall half/double `55/144`; symmetric half/double `50/66`. Every threshold fails against `950`, `20/20`, and `12/12`. All 240 canonical `-1` rows in candidate-v1 have zeroed FFT-grid/mask/power and score-valid oracle rows, confirming the historical diagnostic defect. These facts support `REJECTED_NON_FREEZABLE`; they cannot be repaired by moving the three unit witnesses.

The candidate receipt's old generator/selector hashes are historical bindings to candidate-v1's own immutable bytes. They are not presented by the amendment as current source snapshots and cannot be reused for a new candidate. The amendment's live `relevant_source_snapshot` is internally uniform and matches the finalized source. Therefore there is no stale or mixed live-source binding being waived.

A separately authorized concurrent candidate-v2 artifact is outside the amendment's grant of authority and does not change candidate-v1's disposition. Its canonical freeze receipt was independently verified at SHA-256 `191a1689197f7e9a3eb965ee0dc0e52fa58e9d8485c097c9e601162fad24cd19`; its pack digest is `c6dbe56df7b4f7d427ce1f51d5d818ad0e46dcfbf6e1550e0d77116435760196`, and it binds the finalized generator hash `7af1dac0ce246979d608b3db8e9fcaef9fe18dbfe2a5ecc803616e30919bb4ce`. Current `verify_fixture_pack` and the ten-case recomputation pass, but Gate 2 still fails with the same `246/55/144/50/66` observations. That freeze is an immutable diagnostic freeze only; it does not pass Gate 2 or authorize training.

## Preservation and no-escape findings

- Stochastic Gate 2 remains exactly 1,000 rows from `Generator(PCG64(20270815))`; no unit fixture is appended, substituted, sampled, classified, or counted.
- The inclusive agreement requirement remains at least `950/1000`; overall half/double ceilings remain `20/20`; symmetric ceilings remain `12/12` among the same 250 symmetric rows.
- The budget remains 59 training jobs and 552 GPU-hours, with the same stopping ceilings and staged dependencies.
- The three visible validation blocks, K1–K9, the two conditional claims, the partial-cache/supplied-track evidence ceiling, and K8 novelty block remain unchanged.
- `-1` remains the sole signed selected-period failure sentinel. It is not admitted into clocks, candidates, masks, or diagnostic arrays.
- Unit fixtures cannot offset a Gate-2 failure, authorize a later candidate, or serve as training/validation/efficacy evidence.
- Remaining implementation and gate blockers in `EXPERIMENT_CODE_REVIEW` are not resolved or waived by this acceptance.

## Hash validation

All 15 hashes explicitly bound by the amendment JSON match the current intended snapshot:

| Path | SHA-256 |
|---|---|
| `refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001.md` | `25fe82bbcad386b42e4c5a95a762d1f84e4c926796655f02ba5d2ab8ae7dc484` |
| `refine-logs/FINAL_PROPOSAL.md` | `e8ff129b3a67ac07be0e5ff1af09555bb26f000e7cde9315b13493733cda6418` |
| `refine-logs/EXPERIMENT_EXECUTION_CONTRACT.md` | `70f80912e4b9693fae9e859eb12f90cc754b7d523b59af766879518770d69b2e` |
| `refine-logs/EXPERIMENT_PLAN.md` | `020626027a73939f3fd4dfa479e6592699d19f652da592682b90d9235724ce5c` |
| `refine-logs/round-4-refinement.md` | `2031d5c84b8c6c329dd0e43e5b8a09210ed9a3e8b17b47578f2f878a4c2b4e0f` |
| `src/pams/warp_phase/selector.py` | `73b31b7d6c58a67efa8542df754971a58926d783258c9776269ec06765dafa0e` |
| `scripts/experiments/generate_warp_phase_fixture_pack.py` | `7af1dac0ce246979d608b3db8e9fcaef9fe18dbfe2a5ecc803616e30919bb4ce` |
| `tests/test_warp_phase_selector.py` | `2649006bf953e0dfa2190bf63d37420f502aec9c78f0f80a7b0989957bc2d893` |
| `src/pams/warp_phase/gates.py` | `52d573a0fa98233627cb6344de075887431b1cbe0659b48756fd48085f7429cd` |
| `data/warp_phase_pilot_v1/audit/gate1/fixture_candidate_audit.json` | `5e7df8b6f4033b4162fc359cf67628fa00921276428fc1aaad49d92ff3d8e305` |
| `data/warp_phase_pilot_v1/audit/gate1/fixture_candidate_audit.md` | `5ee2d4181619705760a7e3e02c69c42c96ea864fac7a2b34e56d30c68d6209f2` |
| `data/warp_phase_pilot_v1/audit/gate1/fixture_environment.lock.json` | `5b1d1bd864cb87a16fc97843515d0d6cd8063be5ad1adfb857ea4da42bd2e27b` |
| `data/warp_phase_pilot_v1/audit/gate1/fixture_environment.receipt.json` | `88ff454646c72aaf4a8e991f0773ae5bee2f28ea657a5401c1048d88adc46705` |
| `data/warp_phase_pilot_v1/audit/gate1/candidate-20260815-sol/candidate_receipt.json` | `f5e03b431f47f2ae0c1e52f40955fbcb70b97642e52969178ffcfa3140f6c5ce` |
| `data/warp_phase_pilot_v1/audit/gate1/candidate-20260815-sol/pack/metadata.json` | `9a9db9422ef920fff7d39ac8ea0541ae0d81280282a6b09d95205419f337f6f8` |

Additional reviewed bindings are amendment JSON `d5227f04af4452c2224d3a6b0a8342a7a02aa18edf6b5452e9e06290636bc1c9`, gate tests `e623777e26aabbfb56164695c980d015e0394a0cfa0490e1544032373d6f550e`, code-review Markdown `90f63c438ea0d43a49ec513b0ffa15fb083b5228257861e26e630ef360b5f9b3`, and code-review JSON `f86bd6790eeab85a26ebe5d5d86fc2afd36fa4cebe37bab9ecf0a8e44a159f74`.

## Verification performed

- Parsed and cross-checked amendment JSON against the Markdown and all bound files: 15/15 bound hashes matched.
- Independently rehashed every candidate-v1 member and recomputed its filename-NUL-bytes-LF aggregate.
- Executed the three exact constructor probes in both declared NumPy environments.
- Materialized every proposed unit NPY schema in memory and verified dtype, shape, C-order, non-object, finite values, and v2.0 headers.
- Ran `tests/test_warp_phase_selector.py` and `tests/test_warp_phase_gates.py`: **18 passed**.
- Confirmed the unit fixture IDs have no occurrence in generator, selector, gate, candidate metadata, or tests outside the amendment/review namespace.
- Performed no server access, real-data access, evaluator-label access, training, prediction, evaluation, candidate mutation, source/document mutation, manifest update, Git mutation, or commit.

## Final status

`ACCEPT` means the contradiction has a minimal, executable, non-expansive normative resolution. It does not mean any artifact or gate has passed. The review remains same-family and provisional; Gate 1 is not evaluated, Gate 2 for candidate-v1 is `FAIL`, candidate-v1 is `REJECTED_NON_FREEZABLE`, training remains unauthorized, K8/claim freeze remains blocked, and eligible efficacy results remain zero.
