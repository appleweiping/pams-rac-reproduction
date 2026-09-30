# Fresh post-repair adjudication: fixture candidate v2

**Adjudication date:** 2026-08-15  
**Reviewer model:** gpt-5.6-sol  
**Review independence:** same-family, provisional  
**Verdict:** PASS_PROVENANCE_REPAIR  
**Gate-1 status:** NOT_EVALUATED  
**Gate-2 status:** FAIL  
**Training authorization:** none

## Decision

The exact provenance repair required by the fresh candidate-v2 numerical audit is proven complete. The generic fixture-environment paths again contain the exact candidate-v1 receipt and lock bytes, the v2-specific paths preserve the exact candidate-v2 receipt and lock bytes, both detached-receipt links are correct, and all four JSON artifacts are canonical UTF-8 with sorted keys, compact separators, no BOM, and no terminal newline.

The canonical candidate-v2 freeze receipt is also exact: its SHA-256 is 191a1689197f7e9a3eb965ee0dc0e52fa58e9d8485c097c9e601162fad24cd19, its canonical byte length is 2,334, and its parsed payload is identical to the conditional payload in fixture_candidate_v2_audit.json. It binds every and only the 23 v2 pack members, aggregate c6dbe56df7b4f7d427ce1f51d5d818ad0e46dcfbf6e1550e0d77116435760196, and current generator 7af1dac0ce246979d608b3db8e9fcaef9fe18dbfe2a5ecc803616e30919bb4ce.

This adjudication authorizes only use of the frozen candidate-v2 numerical pack as an input to later Gate-1 and Gate-2 validation. It does not pass Gate 1, pass or waive Gate 2, authorize training, authorize a result or claim, authorize evaluation on protected data, or authorize a server launch.

## Exact environment repair

| Role and path | Observed SHA-256 | Required SHA-256 | Canonical/no newline | Detached/source binding | Result |
|---|---|---|---|---|---|
| generic v1 receipt: fixture_environment.receipt.json | 88ff454646c72aaf4a8e991f0773ae5bee2f28ea657a5401c1048d88adc46705 | same | yes | source equals v1 generator 8a12a188c95d38f8958553e10ee06df6920c372e78eef2c2de575e7b8d6eb572 | PASS |
| generic v1 lock: fixture_environment.lock.json | 5b1d1bd864cb87a16fc97843515d0d6cd8063be5ad1adfb857ea4da42bd2e27b | same | yes | detached receipt equals 88ff454646c72aaf4a8e991f0773ae5bee2f28ea657a5401c1048d88adc46705 | PASS |
| v2-specific receipt: fixture_environment.candidate-20260815-sol-v2.receipt.json | 49dce1131bbcce0fb4d0881f6527a00ca591962f785414c8ec4305e117e87ee8 | same | yes | source equals current/v2 generator 7af1dac0ce246979d608b3db8e9fcaef9fe18dbfe2a5ecc803616e30919bb4ce | PASS |
| v2-specific lock: fixture_environment.candidate-20260815-sol-v2.lock.json | c278c77b27467f5928fa37d2fc79580dc58ec5d55cf98eee89f7e69e77d947a0 | same | yes | detached receipt equals 49dce1131bbcce0fb4d0881f6527a00ca591962f785414c8ec4305e117e87ee8 | PASS |

Candidate-v1's receipt binds the restored generic lock hash. Candidate-v2's receipt binds the v2-specific lock hash. The receipt and lock source fields agree within each pair and with the corresponding candidate receipt.

## Candidate and freeze preservation

Candidate-v1 has exactly candidate_receipt.json plus pack/ at its root. Its candidate receipt remains f5e03b431f47f2ae0c1e52f40955fbcb70b97642e52969178ffcfa3140f6c5ce, metadata remains 9a9db9422ef920fff7d39ac8ea0541ae0d81280282a6b09d95205419f337f6f8, every one of its 23 pack-member hashes matches its receipt, and the independently recomputed aggregate remains b2c791f9aac1d86c0849fa74b00b4e605fe21120f9d188f4ec0b304d7052f173.

Candidate-v2 has exactly candidate_receipt.json, canonical_fixture_pack.receipt.json, and pack/ at its root. Its candidate receipt remains 5b18a1f8025cc81b576b72d99a64930734395da2e4ee855015be10a11d08862f, metadata remains 57942038ac85dd718cea0a36eb9db008e337447239664aec8775806def005d12, every one of its 23 pack-member hashes matches its receipt, and the independently recomputed aggregate remains c6dbe56df7b4f7d427ce1f51d5d818ad0e46dcfbf6e1550e0d77116435760196. Thus the repair changed neither candidate payload tree nor either pack/member binding; candidate-v2 has only the expected external freeze receipt at its root.

The canonical freeze receipt is canonical JSON without a BOM or newline and exactly matches the audited proposal, including all five required top-level fields: generator_source_sha256, members, pack_sha256, role, and schema_version.

## Current source and amendment/review bindings

| Input | SHA-256 |
|---|---|
| fixture_candidate_v2_audit.md | 8450fe176da370e46a549ea20d9770f171fe345a1be57c84fb6c0a881c3420af |
| fixture_candidate_v2_audit.json | 82860755e577ffd41e8a4cd7c6ca7c008f444dbd695b6200981655c0609de8a0 |
| src/pams/warp_phase/selector.py | 73b31b7d6c58a67efa8542df754971a58926d783258c9776269ec06765dafa0e |
| scripts/experiments/generate_warp_phase_fixture_pack.py | 7af1dac0ce246979d608b3db8e9fcaef9fe18dbfe2a5ecc803616e30919bb4ce |
| src/pams/warp_phase/gates.py | 52d573a0fa98233627cb6344de075887431b1cbe0659b48756fd48085f7429cd |
| EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001.md | 25fe82bbcad386b42e4c5a95a762d1f84e4c926796655f02ba5d2ab8ae7dc484 |
| EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001.json | d5227f04af4452c2224d3a6b0a8342a7a02aa18edf6b5452e9e06290636bc1c9 |
| EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001_REVIEW.md | a1fcd938fcb849cc820275ffb54d4b233314accbd7d1a21aa2ce3742e2dafa9c |
| EXPERIMENT_EXECUTION_CONTRACT_AMENDMENT_001_REVIEW.json | 8937446bbb9708e205f850e367c48fce70d42de9054c1066522865104352c8f6 |

The amendment review is ACCEPT at the same-family provisional specification layer. Its limited normative correction and candidate-v1 rejection do not broaden this repair adjudication's authority.

## NumPy 1.26.4 consumption checks

The project local environment reported CPython 3.12.13 and NumPy 1.26.4. With bytecode generation disabled, current verify_fixture_pack passed against candidate-v2 pack/ and the canonical freeze receipt. The current ten-case recomputation also passed, with canonical outcomes:

| Fixture ID | Outcome |
|---:|---:|
| 0 | -1 |
| 150 | -1 |
| 250 | -1 |
| 375 | 48 |
| 500 | 22 |
| 600 | 18 |
| 700 | 11 |
| 800 | 8 |
| 900 | -1 |
| 999 | 54 |

The same run reproduced PCG64 witness 22cf6053f9a31cee979636aa1f1f629b890a4a88956d42c99d5bfb4333e3bcc0 and K4 vector hash 5e2f35a11bd8a376e44d5ad7d3e5067a0bb5a03e2d61edd84f5f5bfd6cf351ec.

## Gate 2 remains failed

The unchanged inclusive Gate-2 rules independently reproduce:

| Check | Observed | Required | Result |
|---|---:|---:|---|
| canonical/weak agreement within 10% | 246 / 1,000 | at least 950 | FAIL |
| half-period classifications overall | 55 / 1,000 | at most 20 | FAIL |
| double-period classifications overall | 144 / 1,000 | at most 20 | FAIL |
| half-period classifications, symmetric IDs 250..499 | 50 / 250 | at most 12 | FAIL |
| double-period classifications, symmetric IDs 250..499 | 66 / 250 | at most 12 | FAIL |

The current Gate-2 threshold function fails closed at the first violated agreement condition. The full independently computed observation vector is exactly 246/55/144/50/66. Proven provenance does not reinterpret or cure this scientific failure.

## Scope

No server data, real data, test/sealed/held-out data, result artifact, or Git state was accessed. No code, document, candidate, environment artifact, pack member, or receipt was modified. The only artifacts created by this adjudication are this Markdown file and its companion JSON.
