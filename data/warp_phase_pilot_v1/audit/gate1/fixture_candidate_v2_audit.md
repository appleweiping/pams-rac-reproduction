# Fresh numerical artifact audit: fixture candidate v2

**Audit date:** 2026-08-15  
**Reviewer model:** `gpt-5.6-sol`  
**Review independence:** same-family, provisional  
**Artifact-integrity verdict:** `BLOCKED_PROVENANCE_REPAIR_REQUIRED` (`BLOCKED`)  
**Numerical-candidate disposition:** `PASS_FREEZE_CANDIDATE_AFTER_PROVENANCE_REPAIR`  
**Gate-2 status:** `FAIL`  
**Training authorization:** none

## Decision

`candidate-20260815-sol-v2` is numerically and structurally a valid freeze candidate. Its complete raw stochastic population replays exactly under the bound CPython 3.12.13 / NumPy 2.4.6 environment, every serialized member hash and aggregate hash is correct, all five selected-period arrays recompute with zero mismatches, and all canonical diagnostic arrays recompute with zero mismatches across all 1,000 rows.

The overall artifact-integrity verdict is nevertheless `BLOCKED`, not `PASS_FREEZE_CANDIDATE`, because the generic `fixture_environment.lock.json` and `fixture_environment.receipt.json` paths were reused in place for v2. Candidate-v1 binds the earlier hashes, but those generic paths now contain the v2 bytes. Before any canonical freeze receipt is written, the main process must preserve the current v2 environment bytes under v2-specific names and restore the original generic v1 bytes. After that repair and a hash-only confirmation, the numerical candidate qualifies as `PASS_FREEZE_CANDIDATE`.

This audit does not accept proposed Amendment 001, pass Gate 1, waive Gate 2, authorize training, authorize a claim, or create a canonical freeze receipt.

## Bound inputs

| Input | SHA-256 |
|---|---|
| `refine-logs/EXPERIMENT_EXECUTION_CONTRACT.md` | `70f80912e4b9693fae9e859eb12f90cc754b7d523b59af766879518770d69b2e` |
| `refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001.md` | `25fe82bbcad386b42e4c5a95a762d1f84e4c926796655f02ba5d2ab8ae7dc484` |
| `refine-logs/EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001.json` | `d5227f04af4452c2224d3a6b0a8342a7a02aa18edf6b5452e9e06290636bc1c9` |
| `src/pams/warp_phase/selector.py` | `73b31b7d6c58a67efa8542df754971a58926d783258c9776269ec06765dafa0e` |
| `scripts/experiments/generate_warp_phase_fixture_pack.py` | `7af1dac0ce246979d608b3db8e9fcaef9fe18dbfe2a5ecc803616e30919bb4ce` |
| `src/pams/warp_phase/gates.py` | `52d573a0fa98233627cb6344de075887431b1cbe0659b48756fd48085f7429cd` |
| `tests/test_warp_phase_selector.py` | `2649006bf953e0dfa2190bf63d37420f502aec9c78f0f80a7b0989957bc2d893` |
| `tests/test_warp_phase_gates.py` | `e623777e26aabbfb56164695c980d015e0394a0cfa0490e1544032373d6f550e` |
| current fixture environment lock | `c278c77b27467f5928fa37d2fc79580dc58ec5d55cf98eee89f7e69e77d947a0` |
| current fixture environment detached receipt | `49dce1131bbcce0fb4d0881f6527a00ca591962f785414c8ec4305e117e87ee8` |
| v2 candidate receipt | `5b18a1f8025cc81b576b72d99a64930734395da2e4ee855015be10a11d08862f` |
| v2 pack aggregate | `c6dbe56df7b4f7d427ce1f51d5d818ad0e46dcfbf6e1550e0d77116435760196` |

The current selector and generator hashes match both `metadata.json` and `candidate_receipt.json`. The pack aggregate was independently recomputed over sorted `filename_utf8 || 0x00 || file_bytes || 0x0a`, not copied from the receipt.

## Structural and byte validation

- The candidate root contains one canonical candidate receipt and a pack of exactly 23 members: 22 NPY arrays plus `metadata.json`.
- `candidate_receipt.json`, `metadata.json`, the fixture lock, and its detached receipt are canonical UTF-8 JSON with sorted keys, compact separators, no BOM, and no terminal newline.
- Every one of the 23 pack-member SHA-256 values equals the candidate receipt.
- All 22 NPY files use format version 2.0. Every multibyte dtype is explicitly little-endian, `|u1` is endian-neutral, every array is C-order and non-object, every header shape matches metadata, and every payload length is exact. No float member contains NaN or infinity.
- `q.npy` is exact float64 `0..127`; `candidate_period.npy` is exact uint16 `4..128`; category membership is exactly `(150,100,125,125,100,100,100,100,100)`.
- The only selected-period failure sentinel is `-1`; all nonfailure values are in `4..128`.

The complete per-member hash map and the exact proposed freeze payload are in the companion JSON audit.

## Environment validation

The live fixture environment is CPython 3.12.13 with NumPy 2.4.6 on little-endian Windows AMD64. Its interpreter SHA-256 is `1a6b3a5b57d95882cc436344c74b1179bd6bb37126dddd52f38d00e02284c702`, exactly matching the lock. The lock's source hash matches the current generator. Its `detached_receipt_sha256` equals the independently recomputed receipt hash `49dce1131bbcce0fb4d0881f6527a00ca591962f785414c8ec4305e117e87ee8`.

The named NumPy wheel `numpy-2.4.6-cp312-cp312-win_amd64.whl` has locked SHA-256 `d8e8286dd7cea7895157318d1b91cdacac64c479f3cbc8dce548331728484751`; the same filename and digest were independently confirmed against the PyPI 2.4.6 release JSON. The installed package reports installer `uv`, and the active installer reports `uv 0.11.21`.

The fresh PCG64 witness reproduced all 16 required uint64 values and byte hash `22cf6053f9a31cee979636aa1f1f629b890a4a88956d42c99d5bfb4333e3bcc0`.

## Independent raw replay

An independent implementation of the contract's category assignment, PCG64 draw order, retry semantics, three-segment clock, pose equations, bilateral overrides, missingness, clipping, weak dropout, and lowest-index restoration replayed all 1,000 rows in the dedicated NumPy 2.4.6 environment.

Exact mismatch counts were zero for:

- `category.npy`
- `semantic_period.npy`
- `pose.npy`
- `joint_mask.npy`
- `weak_pose.npy`
- `weak_joint_mask.npy`

The maximum absolute difference for semantic periods, clean poses, and weak poses was exactly `0.0`.

## Selector and diagnostic replay

The current selector was run on all 1,000 clean rows, all 1,000 weak rows, all 1,000 first-64 views, all 1,000 last-64 views, and all 1,000 reversals. No canonical diagnostic invocation was unavailable.

| Recomputed member | Mismatches |
|---|---:|
| `expected_selected_period.npy` | 0 |
| `expected_weak_selected_period.npy` | 0 |
| `expected_first64_selected_period.npy` | 0 |
| `expected_last64_selected_period.npy` | 0 |
| `expected_reversal_selected_period.npy` | 0 |
| `expected_parent_accept.npy` | 0 |
| `expected_fft_x.npy` | 0 |
| `expected_fft_mask.npy` | 0 |
| `expected_fft_power.npy` | 0 |
| `expected_acf.npy` | 0 |
| `expected_score.npy` | 0 |
| `expected_score_valid.npy` | 0 |
| `expected_local_max.npy` | 0 |
| `expected_harmonic_power.npy` | 0 |

There are 240 canonical `-1` decisions. Their ascending uint16 ID-list byte hash is `4278935bd643aa5880e45436302fe04cddf98915ca7236e7214c8cd86b027fa3`.

All 240 failed rows now retain the exact recomputed FFT grid, FFT mask, and nonzero FFT power rather than a zero diagnostic row. Seventy of the 240 rows have valid computed ACF/score/harmonic candidates, and all 70 retain exact nonzero ACF, score, harmonic power, and score-valid masks. The remaining 170 have no score-valid candidate, so their ACF/score/harmonic values are correctly exact zero under the existing valid-mask schema. All 240 local-max masks are correctly all false, which is the direct reason the final decision fails. Candidate-v1's corresponding 240 failed rows had zero FFT, ACF, harmonic, and score-valid diagnostic rows; v2 fixes that serialization defect without changing any decision.

The parent-accept count remains 12. The raw population and all five decision-array hashes are identical to the frozen candidate-v1 scientific population; only the intended diagnostic members and metadata changed.

## Gate 2

The unchanged inclusive Gate-2 rules give:

| Check | Observed | Required | Result |
|---|---:|---:|---|
| canonical/weak agreement within 10% | 246 / 1,000 | at least 950 | FAIL |
| half-period classifications overall | 55 / 1,000 | at most 20 | FAIL |
| double-period classifications overall | 144 / 1,000 | at most 20 | FAIL |
| half-period classifications, symmetric IDs 250..499 | 50 / 250 | at most 12 | FAIL |
| double-period classifications, symmetric IDs 250..499 | 66 / 250 | at most 12 | FAIL |

Therefore `gate2_status` is `FAIL`. Numerical artifact integrity does not reinterpret, waive, or repair this scientific failure.

## Candidate-v1 preservation and provenance blocker

Candidate-v1's receipt still hashes to `f5e03b431f47f2ae0c1e52f40955fbcb70b97642e52969178ffcfa3140f6c5ce`; all 23 of its pack-member hashes match that receipt; and its independently recomputed aggregate is still `b2c791f9aac1d86c0849fa74b00b4e605fe21120f9d188f4ec0b304d7052f173`. Its directory bytes are untouched.

The external generic environment paths are not untouched. Candidate-v1 binds lock hash `5b1d1bd864cb87a16fc97843515d0d6cd8063be5ad1adfb857ea4da42bd2e27b` and detached receipt hash `88ff454646c72aaf4a8e991f0773ae5bee2f28ea657a5401c1048d88adc46705`, while those paths currently contain v2 hashes `c278c77b...947a0` and `49dce113...7ee8`.

Required repair before freeze:

1. Copy the current v2 lock and receipt bytes, without reserialization, to v2-specific immutable paths such as `fixture_environment.candidate-20260815-sol-v2.lock.json` and `fixture_environment.candidate-20260815-sol-v2.receipt.json`. Their hashes must remain `c278c77b27467f5928fa37d2fc79580dc58ec5d55cf98eee89f7e69e77d947a0` and `49dce1131bbcce0fb4d0881f6527a00ca591962f785414c8ec4305e117e87ee8`.
2. Restore the generic receipt to the exact v1 bytes with SHA-256 `88ff454646c72aaf4a8e991f0773ae5bee2f28ea657a5401c1048d88adc46705`.
3. Restore the generic lock to the exact v1 bytes with SHA-256 `5b1d1bd864cb87a16fc97843515d0d6cd8063be5ad1adfb857ea4da42bd2e27b`.
4. Hash-confirm all four paths, then issue the canonical freeze receipt against the v2 pack and point later Gate-1 validation to the v2-specific lock path.

The v1 bytes were independently reconstructed without using the earlier candidate-v1 audit: the v1 receipt is the current environment receipt with `source_sha256=8a12a188c95d38f8958553e10ee06df6920c372e78eef2c2de575e7b8d6eb572`; the v1 lock is the current lock with that source hash and `detached_receipt_sha256=88ff454646c72aaf4a8e991f0773ae5bee2f28ea657a5401c1048d88adc46705`. Canonical serialization produces the two expected hashes exactly.

## Conditional canonical freeze receipt for the main process

After the provenance repair, the current gate verifier requires exactly five top-level fields:

- `generator_source_sha256`: `7af1dac0ce246979d608b3db8e9fcaef9fe18dbfe2a5ecc803616e30919bb4ce`
- `members`: the exact 23-entry map in `candidate_receipt.json` and the companion audit JSON
- `pack_sha256`: `c6dbe56df7b4f7d427ce1f51d5d818ad0e46dcfbf6e1550e0d77116435760196`
- `role`: `canonical_fixture_pack`
- `schema_version`: `1`

Recursively sorted compact canonical JSON for that exact payload is 2,334 bytes and has proposed receipt SHA-256 `191a1689197f7e9a3eb965ee0dc0e52fa58e9d8485c097c9e601162fad24cd19`. This audit only specifies the payload; it does not create it.

## Validation runs and scope

- Dedicated fixture environment: independent 1,000-row raw replay, 5,000 selector decisions, all 1,000 canonical diagnostic rows, member hashes, headers, and aggregates passed.
- NumPy 1.26.4 consumption environment: the fixed ten-case recomputation passed; PCG64 and K4 hashes reproduced `22cf6053...bcc0` and `5e2f35a1...51ec`.
- Targeted current tests: 13 passed in 0.65 seconds (`test_warp_phase_selector.py` plus the relevant read-only selector/gate fixture tests).

The existing selector regression test still reads candidate-v1 and checks in-memory diagnostic preservation rather than v2 serialization. That is a test-coverage caveat, not an artifact discrepancy; this audit directly checked every v2 row.

No server data, real data, test/sealed/held-out data, results artifact, or Git state was read. No code, document, candidate, candidate-v1, or environment artifact was modified. Only this Markdown audit and its companion JSON were created.
