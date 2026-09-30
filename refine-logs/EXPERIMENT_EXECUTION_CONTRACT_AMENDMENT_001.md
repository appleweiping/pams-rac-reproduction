# Experiment Execution Contract Amendment 001: Selector Unit-Fixture Separation

> **STATUS: `PROPOSED_PENDING_FRESH_REVIEW`**
>
> **PROVISIONAL — SAME-FAMILY — NON-AUTHORITATIVE**
>
> **NO TRAINING, GATE PASSAGE, CLAIM, OR FREEZE AUTHORIZATION**

**Date:** 2026-08-15

**Amends, if accepted:** `refine-logs/FINAL_PROPOSAL.md` and `refine-logs/EXPERIMENT_EXECUTION_CONTRACT.md`

**Scope:** the placement, construction, serialization, and testing of the three deterministic Gate-1 selector unit fixtures only

## 1. Reason and exact authority boundary

The canonical stochastic population fixes every raw fixture at 128 clocks, `q_i=float64(i)` for `i=0,...,127`. Its selector grid therefore has span 127 and spacing

```text
Delta_1000 = 127/255
k_1000(P=20) = 256*(127/255)/20 = 8128/1275 ~= 6.374901960784314.
```

It cannot simultaneously provide the separately declared all-valid `Delta=1`, `P=20`, `k=12.8` unit case. Its full support also has span 127, not the separately declared span-126 upper-endpoint case. The sentence in `FINAL_PROPOSAL.md` Section 3.4 and the identical sentence in `round-4-refinement.md` Section 3.4 that place those three cases "in the 1,000-fixture byte pack" are therefore non-executable as written.

If and only if this amendment passes a fresh review, it supersedes only that placement clause. The three cases become a separate deterministic Gate-1 selector unit-fixture pack. Every other proposal, contract, plan, selector, generator, gate, metric, threshold, RNG, job, claim, and selected-period rule remains unchanged. This document does not change code.

## 2. Non-expansion and preservation rule

The separate unit fixtures are numerical interface tests, not samples, observations, labels, evaluator records, training data, validation data, or efficacy evidence. They consume no count, density, semantic-period annotation, boundary, evaluator-vault value, real pose track, result artifact, or model prediction.

The stochastic Gate-2 population remains exactly 1,000 rows with the existing IDs, categories, 128 raw clocks, raw poses, raw masks, weak views, RNG seed and draw order, category construction, method, selector, five selected-period arrays, acceptance semantics, and aggregate thresholds. In particular:

- `Generator(PCG64(20270815))`, every RNG position, and the NumPy 2.4.6 environment binding remain unchanged;
- the agreement threshold remains at least 950/1,000 within the inclusive 10% rule;
- overall half and double ceilings remain 20 each, and symmetric-subset half and double ceilings remain 12 each;
- all 1,000 raw fixtures remain Gate-2 inputs; the three unit fixtures are not appended, substituted, sampled, classified, or counted in any Gate-2 numerator or denominator;
- the unit fixtures cannot repair, waive, offset, or reinterpret a Gate-2 failure;
- network, loss, optimizer, data, evaluator, seeds, staged run order, 59-job/552-GPU-hour ceiling, three visible blocks, claims, K1–K9, and selector-defined pseudo-cycle semantics remain unchanged.

No `MANIFEST.md`, manifest digest, source file, test file, generator, pack byte, candidate byte, configuration, or result is modified by this proposed amendment.

## 3. Exact label-free COCO17 constructor

All three unit fixtures use the following constructor. Let the fixture have clock count `N`, source period `P_src`, and exact int64 clocks `q_i=i`, `i=0,...,N-1`. Allocate `pose` as float64 `[N,17,3]` and `joint_mask` as `uint8 [N,17]`.

1. Set every mask entry to `1` and every confidence to float64 `0.9`.
2. Set hips 11 and 12 to `(-0.5,0.0)` and `(0.5,0.0)` at every clock.
3. Set shoulders 5 and 6 to `(-0.5,2.0)` and `(0.5,2.0)` at every clock.
4. For every other joint `j` in `{0,...,16}\{5,6,11,12}`, compute in float64, in increasing clock then joint order,

```text
b_x(j) = (j % 5 - 2)/4
b_y(j) = floor(j/5)/4
theta(i,j) = 2*pi*float64(q_i)/float64(P_src) + float64(j)*float64(0.1)
pose[i,j,0] = b_x(j) + float64(0.2)*sin(theta(i,j))
pose[i,j,1] = b_y(j) + float64(0.2)*cos(theta(i,j))
```

The anchors are never overwritten by Step 4. This construction is finite and all-valid without label or evaluator data. Under the frozen COCO17 normalization, the hip root is exactly `(0,0)`, the shoulder mean is exactly `(0,2)`, the scale is exactly float64 `2.0`, all 17 normalized joint masks are true, and every feature-frame mask is true. The piecewise-constant velocity field is computed only by the existing selector rule from these normalized positions and clocks; it is not supplied as an alternate selector input.

## 4. The three deterministic Gate-1 fixtures

The fixture IDs, inputs, and required decisions are exactly:

| Fixture ID | `N` and clocks | `P_src` | Span / `P_max` | Required interface assertion |
|---|---|---:|---|---|
| `offbin_delta1_p20` | `N=256`, `q=0,...,255` | 20 | span 255; `P_max=127` | `Delta=255/255=1` exactly; `fft_x[n]=float64(n)` byte-exact; `k(20)=256/20=12.8`; adjacent-bin interpolation uses bins 12 and 13; selected period is 20. |
| `lower_endpoint_p4` | `N=64`, `q=0,...,63` | 4 | span 63; `P_max=31>4` | candidate 4 is valid, `S(4)>=S(5)`, its one-sided lower-endpoint local-max bit is true, and selected period is 4. |
| `upper_endpoint_span126_p63` | `N=127`, `q=0,...,126` | 63 | span 126; `P_max=63` | candidate 63 is valid, `S(63)>=S(62)`, its one-sided upper-endpoint local-max bit is true, and selected period is 63. |

The constructed `P_src` values are unit-test constants only. They are not semantic-period labels, are never joined to evaluator data, and do not alter the meaning of a selected period: the output remains the selector-defined integer pseudo-period chosen by the existing ACF/FFT score, endpoint rules, and exact-score/smaller-`P` tie break.

## 5. Unit-fixture filenames and schemas

The future artifact root is exactly:

```text
data/warp_phase_pilot_v1/audit/gate1/selector-unit-fixtures-v1/
```

It contains `metadata.json`, `candidate_period.npy`, the per-fixture members below, and `selector-unit-fixtures-v1.receipt.json`. `candidate_period.npy` is `<u2[125]` and contains integers 4 through 128 exactly. For each fixture ID `F` and its declared `N`, the members are:

| Filename suffix after `F.` | Schema | Meaning |
|---|---|---|
| `input_clocks.npy` | `<i8[N]` | Exact clocks above. |
| `input_pose.npy` | `<f8[N,17,3]` | Raw constructor output. |
| `input_joint_mask.npy` | `|u1[N,17]` | All ones. |
| `expected_scale.npy` | `<f8[1]` | Exact value `2.0`. |
| `expected_normalized_coordinates.npy` | `<f8[N,17,2]` | Frozen COCO17 normalized coordinates. |
| `expected_normalized_joint_mask.npy` | `|u1[N,17]` | All ones. |
| `expected_feature_frame_valid.npy` | `|u1[N]` | All ones. |
| `expected_velocity.npy` | `<f8[N-1,34]` | Flattened x/y piecewise-constant cell velocity. |
| `expected_velocity_mask.npy` | `|u1[N-1,34]` | All ones. |
| `expected_fft_x.npy` | `<f8[256]` | Inclusive source-clock FFT grid. |
| `expected_fft_mask.npy` | `|u1[256,34]` | Queried coordinate-valid mask. |
| `expected_fft_power.npy` | `<f8[127]` | `E_1,...,E_127`; bins 0 and 128 are excluded. |
| `expected_candidate_present.npy` | `|u1[125]` | True exactly for serialized periods `4,...,P_max`. |
| `expected_acf.npy` | `<f8[125]` | Raw finite ACF diagnostic where computed, else exact `+0.0`. |
| `expected_acf_computed.npy` | `|u1[125]` | Distinguishes a computed ACF from the zero fill. |
| `expected_harmonic_power.npy` | `<f8[125,3]` | Interpolated powers at `k`, `2k`, and `3k` where the score is computed, else exact `+0.0`. |
| `expected_score.npy` | `<f8[125]` | Raw finite `S(P)` where computed, else exact `+0.0`. |
| `expected_score_computed.npy` | `|u1[125]` | Distinguishes a computed score from the zero fill. |
| `expected_score_valid.npy` | `|u1[125]` | Existing finite/support/`R(P)>=0.25` candidate predicate. |
| `expected_local_max.npy` | `|u1[125]` | Existing one- or two-sided local-maximum predicate. |
| `expected_selected_period.npy` | `<i2[1]` | Required decision, or `-1` on selection failure. |

All NPY files are format 2.0, C-order, non-object, and use the explicit dtype shown. Float arrays must be finite; validity/computed masks, not NaN or a numeric overload, identify unavailable entries. `metadata.json` is canonical UTF-8 without BOM or terminal newline, with recursively sorted keys, separators `,` and `:`, the constructor constants, fixture table, schemas, environment, amendment hashes, and relevant source hashes.

## 6. Failure diagnostics and the sole sentinel

Diagnostic computation and decision computation are separate receipt stages. A failed selection must not discard or omit the diagnostic arrays. For every serialized selector invocation whose schema declares FFT, ACF, harmonic, score, or local-max arrays, those arrays and their masks are written even when the final decision fails. Values not reached or not computable are exact `+0.0` with the corresponding computed/valid mask false; metadata records the fail-closed stage and reason. This rule applies to the three unit fixtures and to the canonical diagnostic rows already declared for the stochastic 1,000-pack.

`-1` is the only decision failure sentinel. It may occur only in signed selected-period decision arrays, including the five existing stochastic `expected_*selected_period.npy` arrays and the unit-fixture `expected_selected_period.npy`. Zero, another negative integer, NaN, infinity, an empty file, a missing row, or a missing diagnostic member is not a selection-failure encoding. `-1` never appears in clocks, candidates, masks, FFT power, ACF, harmonic power, scores, or local-max arrays.

This serialization rule is diagnostic only. It does not create a fallback, turn a failure into a decision, change the selector score or selected period, or change any Gate-2 threshold.

## 7. Hash freeze and tests

The unit pack is generated in the existing dedicated CPython 3.12.13 / NumPy 2.4.6 fixture environment without any RNG call. Before Gate 1, `selector-unit-fixtures-v1.receipt.json` must bind:

- SHA-256 of this amendment Markdown and its matching JSON;
- SHA-256 of `FINAL_PROPOSAL.md`, `EXPERIMENT_EXECUTION_CONTRACT.md`, `EXPERIMENT_PLAN.md`, and `round-4-refinement.md`;
- SHA-256 of the exact selector, stochastic generator, selector test, and gate source snapshots;
- SHA-256 of `metadata.json` and every NPY member;
- the aggregate pack SHA-256 over sorted `filename_utf8 || 0x00 || file_bytes || 0x0a`.

The receipt is canonical UTF-8 JSON without BOM or terminal newline. A fresh reviewer must independently validate member sets, schemas, formulas, raw hashes, `Delta`, `P_max`, `k`, endpoint inequalities, selected periods, and aggregate hash. The NumPy 1.26.4 consumption environment then independently recomputes float64 arrays with `rtol=1e-12`, `atol=1e-12`, `equal_nan=False`; integer/mask arrays and the off-bin FFT grid are exact. Any mismatch is Gate-1 K5 failure and cannot be repaired after freeze.

Gate-1 tests must separately assert:

1. `offbin_delta1_p20` has the exact all-valid COCO17 normalization, all-valid velocity and FFT masks, `fft_x == arange(256,dtype=float64)` byte-for-byte, `k=12.8`, interpolation between stored bins 12 and 13, and selection 20;
2. `lower_endpoint_p4` has `P_max=31`, exercises only the lower one-sided comparison, and selects 4;
3. `upper_endpoint_span126_p63` has span 126 and `P_max=63`, exercises only the upper one-sided comparison, and selects 63;
4. a forced fail-closed decision still emits every declared diagnostic array and writes only `-1` to the selected-period decision field;
5. none of the three fixture IDs or hashes enters the 1,000-row population, RNG receipt, category counts, or Gate-2 aggregate calculation.

## 8. Candidate-v1 disposition and preserved failure conclusion

For this amendment, **candidate-v1** means the existing directory `data/warp_phase_pilot_v1/audit/gate1/candidate-20260815-sol/`, bound by candidate receipt SHA-256 `f5e03b431f47f2ae0c1e52f40955fbcb70b97642e52969178ffcfa3140f6c5ce` and pack digest `b2c791f9aac1d86c0849fa74b00b4e605fe21120f9d188f4ec0b304d7052f173`.

Candidate-v1 is **`REJECTED_NON_FREEZABLE`**. Its bytes remain untouched as non-authorizing failure evidence, but no authoritative freeze receipt may be issued for it. The earlier provisional `PASS_FREEZE_CANDIDATE` recommendation is not an authorization and is superseded on freeze disposition by this proposed correction if accepted. Candidate-v1 must not be renamed, repaired, regenerated, promoted, or used to authorize training.

Its failure conclusion is unchanged: canonical/weak agreement is 246/1,000; overall half/double counts are 55/144; symmetric half/double counts are 50/66; every corresponding threshold check fails; Gate 2 is `FAIL`; Gate 1 remains not evaluated; and training remains unauthorized. Separating the three Gate-1 unit fixtures does not alter any of those values.

Any later candidate must preserve the exact 1,000 raw stochastic fixture inputs, RNG, thresholds, method, and failure semantics; it must place the three unit fixtures only in the separate namespace above and remain pending a fresh review. This amendment does not authorize creation or freezing of that later candidate.

## 9. Bound input snapshot and current status

This proposal was drafted against the following read-only snapshot:

| Input | SHA-256 |
|---|---|
| `refine-logs/FINAL_PROPOSAL.md` | `e8ff129b3a67ac07be0e5ff1af09555bb26f000e7cde9315b13493733cda6418` |
| `refine-logs/EXPERIMENT_EXECUTION_CONTRACT.md` | `70f80912e4b9693fae9e859eb12f90cc754b7d523b59af766879518770d69b2e` |
| `refine-logs/EXPERIMENT_PLAN.md` | `020626027a73939f3fd4dfa479e6592699d19f652da592682b90d9235724ce5c` |
| `refine-logs/round-4-refinement.md` | `2031d5c84b8c6c329dd0e43e5b8a09210ed9a3e8b17b47578f2f878a4c2b4e0f` |
| `src/pams/warp_phase/selector.py` | `73b31b7d6c58a67efa8542df754971a58926d783258c9776269ec06765dafa0e` |
| `scripts/experiments/generate_warp_phase_fixture_pack.py` | `7af1dac0ce246979d608b3db8e9fcaef9fe18dbfe2a5ecc803616e30919bb4ce` |
| `tests/test_warp_phase_selector.py` | `2649006bf953e0dfa2190bf63d37420f502aec9c78f0f80a7b0989957bc2d893` |
| `src/pams/warp_phase/gates.py` | `52d573a0fa98233627cb6344de075887431b1cbe0659b48756fd48085f7429cd` |
| `data/warp_phase_pilot_v1/audit/gate1/fixture_candidate_audit.json` | `5e7df8b6f4033b4162fc359cf67628fa00921276428fc1aaad49d92ff3d8e305` |
| `data/warp_phase_pilot_v1/audit/gate1/candidate-20260815-sol/candidate_receipt.json` | `f5e03b431f47f2ae0c1e52f40955fbcb70b97642e52969178ffcfa3140f6c5ce` |
| `data/warp_phase_pilot_v1/audit/gate1/candidate-20260815-sol/pack/metadata.json` | `9a9db9422ef920fff7d39ac8ea0541ae0d81280282a6b09d95205419f337f6f8` |

The amendment remains `PROPOSED_PENDING_FRESH_REVIEW`, provisional, same-family, and non-authoritative. It authorizes no training, gate passage, result, claim, candidate freeze, or manifest update.
