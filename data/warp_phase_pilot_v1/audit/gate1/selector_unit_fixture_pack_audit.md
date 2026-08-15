# Selector Unit-Fixture Pack Audit

> **Verdict: `ACCEPT_FREEZE_INPUT_ONLY`**
>
> **Acceptance status: provisional**
>
> **Review independence: same-family**
>
> **Reviewer: `gpt-5.6-sol`, reasoning `xhigh`**

**Date:** 2026-08-15

This verdict authorizes only treating the exact receipt-bound bytes under `data/warp_phase_pilot_v1/audit/gate1/selector-unit-fixtures-v1/` as immutable Gate-1 input. It does not pass Gate 1 or Gate 2 and does not authorize training, a result, an efficacy or novelty claim, a server launch, publication, a manifest change, or any code/pack mutation.

## Exact artifact binding

- Root members: 66 exactly = 64 NPY files + `metadata.json` + the receipt.
- Aggregate members: 65 exactly; the receipt is not self-hashed.
- Pack SHA-256: `406fe3934dd307accbd735dd6f2e54890d6de643c7bfe1d05d9f2d1131a8efcf`.
- Aggregate algorithm: `sha256(sorted(filename_utf8 || 0x00 || file_bytes || 0x0a))`.
- Receipt SHA-256: `913e8946ee27392ab374a26887a5a9d8169e69afd951b98319019f3eb1cf2f5f`.
- Metadata SHA-256: `57314eb709f7d812297811f0928f3b4c8901fa7bdeab5923cfef4b4a96825789`.
- All 65 receipt-bound member hashes match. Their exact filename/hash map is duplicated in the JSON audit.

`metadata.json` and the receipt are canonical UTF-8 JSON with recursively sorted keys, compact separators, no BOM, and no terminal newline. Every NPY member is format 2.0, non-object, C-order, and matches the declared explicit dtype and shape. All float arrays are finite; all declared mask arrays are binary.

## Independent numerical findings

The complete constructor, normalization, velocity field, FFT256, ACF, harmonic interpolation, score, local-maximum mask, and selection were independently reimplemented in CPython 3.12.13 / NumPy 1.26.4. Float comparisons used `rtol=1e-12`, `atol=1e-12`, `equal_nan=False`; integer/mask arrays were exact. The largest absolute float difference from the NumPy 2.4.6 pack was `3.3306690738754696e-16`.

All three constructors have exact int64 clocks, all-one uint8 joint masks, confidence 0.9, hips 11/12 at `(-0.5,0)` and `(0.5,0)`, shoulders 5/6 at `(-0.5,2)` and `(0.5,2)`, and every other joint follows the amendment's increasing-clock-then-joint float64 sine/cosine formula. The normalized root is `(0,0)`, scale is exactly 2.0, normalized coordinates match, and normalized-joint, frame, velocity, and FFT masks are all valid.

| Fixture | Exact interface findings | Decision |
|---|---|---|
| `offbin_delta1_p20` | span 255; `P_max=127`; `Delta=1`; `fft_x` byte-equals float64 `arange(256)`; `k(20)=12.8`; interpolation uses stored bins 12 and 13; `S(19)=0.5809198920484148`, `S(20)=0.6594133070408238`, `S(21)=0.5743810869954495` | 20 |
| `lower_endpoint_p4` | span 63; `P_max=31`; lower endpoint uses only `S(4)>=S(5)`; `0.6345727420272262 >= 0.0005560660114877089`; endpoint local-max bit true | 4 |
| `upper_endpoint_span126_p63` | span 126; `P_max=63`; upper endpoint uses only `S(63)>=S(62)`; `0.675905398984302 >= 0.6691573037802108`; endpoint local-max bit true | 63 |

Computed ACF/score counts are 124, 28, and 60 for the off-bin, lower-endpoint, and upper-endpoint fixtures. Unavailable serialized diagnostic locations are exact positive zero with computed/valid masks false. Source review confirms a no-local-maximum decision retains the already-computed diagnostics; serialization writes `-1` only to signed selected-period decision arrays. The actual three fixtures all select successfully, so their decision arrays contain no sentinel. A targeted forced-failure diagnostic/sentinel test passed.

## Separation and authority boundary

The unit generator contains no RNG call. The stochastic generator contains none of the three fixture IDs and no unit-pack namespace. Gate-2 aggregation and threshold functions do not reference the unit root, IDs, or unit verifier; the unit verifier is called only from Gate 1. Therefore these fixtures do not enter the stochastic 1,000 rows, RNG stream, category counts, or any Gate-2 numerator/denominator.

No stochastic pack, Gate 2, server, training path, real/evaluator/label data, sealed/test data, prediction, or paper artifact was opened or run during this audit.

## Exact reviewed-input hashes

| Input | SHA-256 |
|---|---|
| `refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001.md` | `25fe82bbcad386b42e4c5a95a762d1f84e4c926796655f02ba5d2ab8ae7dc484` |
| `refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001.json` | `d5227f04af4452c2224d3a6b0a8342a7a02aa18edf6b5452e9e06290636bc1c9` |
| `refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001_REVIEW.md` | `a1fcd938fcb849cc820275ffb54d4b233314accbd7d1a21aa2ce3742e2dafa9c` |
| `refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001_REVIEW.json` | `8937446bbb9708e205f850e367c48fce70d42de9054c1066522865104352c8f6` |
| `refine-logs/FINAL_PROPOSAL.md` | `e8ff129b3a67ac07be0e5ff1af09555bb26f000e7cde9315b13493733cda6418` |
| `refine-logs/EXPERIMENT_EXECUTION_CONTRACT.md` | `70f80912e4b9693fae9e859eb12f90cc754b7d523b59af766879518770d69b2e` |
| `refine-logs/EXPERIMENT_PLAN.md` | `020626027a73939f3fd4dfa479e6592699d19f652da592682b90d9235724ce5c` |
| `refine-logs/round-4-refinement.md` | `2031d5c84b8c6c329dd0e43e5b8a09210ed9a3e8b17b47578f2f878a4c2b4e0f` |
| `scripts/experiments/generate_warp_phase_selector_unit_fixtures.py` | `2ae32f4133903691a757557425980eb553b3de4d749b4eca9a8f76719b41847f` |
| `scripts/experiments/generate_warp_phase_fixture_pack.py` | `7af1dac0ce246979d608b3db8e9fcaef9fe18dbfe2a5ecc803616e30919bb4ce` |
| `src/pams/warp_phase/selector.py` | `73b31b7d6c58a67efa8542df754971a58926d783258c9776269ec06765dafa0e` |
| `src/pams/warp_phase/gates.py` | `3a91e77692ac54e1e85fd56d8f3bff2b3a3c517095554d30b92e9055e1426dfa` |
| `tests/test_warp_phase_selector.py` | `e670cd5a7c41fc9280097ea01b3c5887df8b7d31389cd92720f4b322ccf30036` |
| `tests/test_warp_phase_gates.py` | `f25234ed7ad888df6088457b79b07ff75c41a95d9e9b052790463437fb34d032` |

All receipt-bound live hashes match these current bytes. The trace script is `.aris/traces/selector-unit-fixture-audit/20260815_sol/independent_recompute.py`, SHA-256 `d70095e1b9514b16c0cefafc68c63ceebbe6df0da6644d058929bc9e4c57f267`.

## Commands executed

```powershell
uv run python .aris/traces/selector-unit-fixture-audit/20260815_sol/independent_recompute.py
uv run python -c "from pathlib import Path; from pams.warp_phase.gates import verify_selector_unit_fixture_pack; r=Path.cwd(); p=r/'data/warp_phase_pilot_v1/audit/gate1/selector-unit-fixtures-v1'; a,h,q=verify_selector_unit_fixture_pack(r,p,p/'selector-unit-fixtures-v1.receipt.json'); print(len(a),h,q)"
uv run pytest -q 'tests/test_warp_phase_selector.py::test_amendment_001_exact_label_free_selector_unit_fixtures' 'tests/test_warp_phase_selector.py::test_selector_unit_fixture_namespace_is_absent_from_stochastic_generator' 'tests/test_warp_phase_gates.py::test_selector_unit_schema_is_complete_rng_free_and_separate' 'tests/test_warp_phase_gates.py::test_ten_case_recomputation_matches_failure_sentinel_and_full_diagnostics'
```

Results: independent recomputation passed; built-in verifier returned 64 arrays plus the exact pack and receipt hashes; selected tests reported `6 passed in 0.74s`.
