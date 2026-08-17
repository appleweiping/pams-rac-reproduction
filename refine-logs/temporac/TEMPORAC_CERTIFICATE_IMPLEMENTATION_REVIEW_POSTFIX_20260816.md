# TempoRAC Certificate Implementation Post-fix Review

- Date: 2026-08-16
- Project: `temporac.execution.v4`
- Reviewer: `gpt-5.6-sol`
- Reviewer family: `openai`
- Executor family: `openai`
- Review independence: `same-family`
- Acceptance status: `provisional`
- Review mode: fresh, raw-artifact, read-only, pure ML experiment-code consistency
- Overall verdict: **REVISE (blocking; not a gate PASS)**

## Bottom line

The exact seven-member target NPZ, closed detached receipt, ABSTAIN zero-artifact rule, three-tau ledger comparison, phase-class union, F4 masked length, right-edge ownership, joint NPZ/receipt loader, exclusive-create primitive, label firewall, and no-legacy/WARP import rule are implemented and exercised successfully.

The candidate is nevertheless not scientifically closed. Four blocking defects remain:

1. certified and loaded NumPy arrays are only flag-read-only, not immutable; write access can be re-enabled and a certified pulse can be moved before artifact construction without revalidation;
2. the X0 adapter accepts a duck-typed, noncanonical object and admits source/resampler/offset combinations outside the frozen X0 inventory;
3. the natural adapter binds an independently supplied `CertificateTrack`, `FeatureRecord`, and free teacher digest without proving that the track came from that feature and selected teacher;
4. the evaluator never consumes certified target pulses/source clocks, so it cannot execute the K7 one-pulse-per-evaluator-interval primitive-unit check.

These are scientific provenance and contract-integration failures, not network-security findings. No malicious local-user threat model is assumed.

## Blocking findings

### F1 — Certified/loaded array payloads are not deeply immutable

**Status:** BLOCKING

`_readonly_exact` makes an owning copy and merely clears the writeable flag (`src/pams/temporac/types.py:71`, `src/pams/temporac/types.py:80`, `src/pams/temporac/types.py:81`). `CertifiedTarget.__post_init__` applies that helper to every certificate ledger (`src/pams/temporac/types.py:252`, `src/pams/temporac/types.py:255`, `src/pams/temporac/types.py:271`). Because those arrays own their storage, ordinary NumPy code can set `array.flags.writeable=True` again.

The artifact builder serializes the target's current arrays without rerunning `CertifiedTarget` validation (`src/pams/temporac/receipts.py:529`, `src/pams/temporac/receipts.py:539`, `src/pams/temporac/receipts.py:554`). The target reader checks binary/support relations but has no traversal ledger with which to re-establish “pulse exactly at every traversal start” (`src/pams/temporac/hashio.py:666`, `src/pams/temporac/hashio.py:683`, `src/pams/temporac/hashio.py:699`). Loaded arrays are also copied into owning storage and only flag-locked (`src/pams/temporac/hashio.py:515`, `src/pams/temporac/hashio.py:530`, `src/pams/temporac/hashio.py:535`).

Independent invalid-input probe:

- generated canonical `U0003`, stationary block, linear, offset zero;
- certified ten events;
- re-enabled `target.pulse.flags.writeable`;
- moved the first pulse from edge 8 to edge 9, where `chi` is still positive and masks are one;
- `build_target_artifact` and `read_target_npz_bytes` both accepted it;
- accepted tampered artifact SHA-256: `8db7e6f86228c96c211b880d56409e4331bc9356e26553d93220d5358bac1c11`.

The present test checks only the initial flag (`tests/temporac/test_certificate_artifact.py:125`) and does not try to re-enable it.

**Required correction:** use immutable backing storage for all exposed certificate/feature/loaded arrays, and/or reconstruct and fully revalidate a sealed certificate immediately at serialization/consumption. Add a regression that re-enabling write access fails and that a shifted traversal-start pulse cannot be built or loaded as `CERTIFIED`.

### F2 — X0 provenance can be attached to noncanonical content and unused inventory rows

**Status:** BLOCKING

The contract freezes X0 source orbits and view inventory (`refine-logs/temporac/FINAL_PROPOSAL.md:354`, `refine-logs/temporac/FINAL_PROPOSAL.md:428`, `refine-logs/temporac/FINAL_PROPOSAL.md:470`, `refine-logs/temporac/FINAL_PROPOSAL.md:475`). `generate_view` is the canonical constructor (`src/pams/temporac/x0.py:548`, `src/pams/temporac/x0.py:557`, `src/pams/temporac/x0.py:587`). However, `X0View` itself has no validation/authority hook (`src/pams/temporac/x0.py:523`), and `certificate_track_from_x0` does not require `isinstance(view, X0View)` or compare the supplied arrays with a canonical regenerated view (`src/pams/temporac/certify.py:676`, `src/pams/temporac/certify.py:679`, `src/pams/temporac/certify.py:687`, `src/pams/temporac/certify.py:689`, `src/pams/temporac/certify.py:699`).

The shipped positive test explicitly passes `SimpleNamespace`, arbitrary ideal-track geometry/phase/pulse/chi, `source_id=3`, `resampler="sinc"`, and `offset=31`, then accepts source-unit index 95 (`tests/temporac/test_certificate_artifact.py:222`, `tests/temporac/test_certificate_artifact.py:224`, `tests/temporac/test_certificate_artifact.py:228`, `tests/temporac/test_certificate_artifact.py:236`, `tests/temporac/test_certificate_artifact.py:239`). Source 3 is a train source, while the frozen contract assigns linear to train and PCHIP/sinc to heldout evaluation.

An independent probe reproduced certification and receipt construction for the same non-`X0View` object, with source key `94476ea52ec20cc862fd63a047a26532874b75ac659a2b0249058603496b1732`, source-unit index 95, and caller-selected teacher digest `44…44`.

**Required correction:** construct the certificate track internally from frozen X0 coordinates, or accept only an authority-bearing canonical view whose complete content and permitted split/resampler/offset tuple are verified. Turn the current `SimpleNamespace` path into a negative test.

### F3 — Natural identity and teacher provenance are not bound to the certificate input

**Status:** BLOCKING

`bind_natural_certificate_track` receives three independent values—an already built track, a feature record, and a free digest string—and copies only key, slot, and digest into provenance (`src/pams/temporac/certify.py:735`, `src/pams/temporac/certify.py:743`, `src/pams/temporac/certify.py:749`, `src/pams/temporac/certify.py:756`). It never proves that geometry, phase, reconstruction, masks, or runs were derived from that exact feature artifact and selected teacher. `TargetProvenance` validates types and digest syntax only (`src/pams/temporac/types.py:203`, `src/pams/temporac/types.py:213`, `src/pams/temporac/types.py:218`). The feature loader also loads a bare NPZ without its detached receipt/source binding (`src/pams/temporac/hashio.py:661`, `src/pams/temporac/hashio.py:663`).

The test named “real natural certificate” is entirely synthetic: it directly constructs `FeatureRecord`, constructs an ideal certificate track, and binds them (`tests/temporac/test_certificate_artifact.py:45`, `tests/temporac/test_certificate_artifact.py:78`, `tests/temporac/test_certificate_artifact.py:90`, `tests/temporac/test_certificate_artifact.py:119`).

Independent probes showed:

- one identical certificate track and teacher digest produced byte-identical seven-member NPZs while receipts named two unrelated natural key/slot identities;
- the same track and feature certified under two arbitrary, different well-formed teacher digests.

The builder itself does not expose source/teacher override arguments and correctly reads its fields from `target.provenance` (`src/pams/temporac/receipts.py:529`, `src/pams/temporac/receipts.py:536`, `src/pams/temporac/receipts.py:548`). The failure is upstream: the provenance object does not attest to the certificate input from which it was produced.

**Required correction:** introduce a canonical natural adapter that consumes a jointly verified feature artifact/receipt plus a typed teacher-inference result bound to the exact selected checkpoint; construct and seal `CertificateTrack` inside that adapter. Do not accept a free feature record, independently assembled track, or unanchored teacher string at the provenance boundary.

### F4 — K7 does not consume the certificate primitive-unit evidence

**Status:** BLOCKING

The normative K7 rule requires, for every certified development identity, mapping frozen teacher pulse edges to source-clock starts, exactly one start in every evaluator interval, no start outside the interval union, and equality to both interval count and `count_gt` (`refine-logs/temporac/FINAL_PROPOSAL.md:1384`, `refine-logs/temporac/FINAL_PROPOSAL.md:1388`, `refine-logs/temporac/FINAL_PROPOSAL.md:1390`, `refine-logs/temporac/FINAL_PROPOSAL.md:1395`).

`OneUseEvaluator.run` has no input for certified targets, pulse ledgers, sampled source clocks, or target-count rows (`src/pams/temporac/evaluator.py:239`, `src/pams/temporac/evaluator.py:245`, `src/pams/temporac/evaluator.py:252`). Its K7 path validates vault syntax, creates count-metric rows, runs the bootstrap, and decides K7 (`src/pams/temporac/evaluator.py:281`, `src/pams/temporac/evaluator.py:285`, `src/pams/temporac/evaluator.py:295`, `src/pams/temporac/evaluator.py:298`). Therefore the primitive-unit condition cannot be checked. The available test covers join cardinality, period minimum, and one-use consumption only (`tests/temporac/test_metrics_evaluator.py:158`, `tests/temporac/test_metrics_evaluator.py:165`, `tests/temporac/test_metrics_evaluator.py:170`).

**Required correction:** before metric computation, consume hash-bound certified development target artifacts and their feature clock ledgers, recompute the committed target-count root, and implement the exact per-interval/no-outside pulse checks. A single mismatch must fail globally before scientific metric finalization.

## Contract checklist

| Check | Status | Primary evidence |
|---|---|---|
| Builder has no source key/unit/teacher override | PASS | `src/pams/temporac/receipts.py:529`, `src/pams/temporac/receipts.py:536`, `src/pams/temporac/receipts.py:548` |
| Public direct provenance construction / ordinary `dataclasses.replace` | PASS | authority checks at `src/pams/temporac/types.py:213`, `src/pams/temporac/types.py:252`; regression at `tests/temporac/test_certificate_artifact.py:185`, `tests/temporac/test_certificate_artifact.py:203` |
| Deep immutable certificate/loaded payload | FAIL | F1 |
| X0 canonical adapter and inventory | FAIL | F2 |
| Natural feature/teacher adapter | FAIL | F3 |
| `CERTIFIED` exact seven-member NPZ | PASS | schema `src/pams/temporac/contract.py:137`; arrays `src/pams/temporac/types.py:356`; deterministic writer `src/pams/temporac/hashio.py:409` |
| Closed ten-key target receipt | PASS | `src/pams/temporac/contract.py:200`; validator `src/pams/temporac/receipts.py:270`; builder `src/pams/temporac/receipts.py:541` |
| `ABSTAIN` has empty ledgers and no artifact/receipt | PASS | `src/pams/temporac/certify.py:159`, `src/pams/temporac/types.py:295`, `src/pams/temporac/receipts.py:534` |
| Phase-class adjacent-chain union then wrap union | PASS | `src/pams/temporac/certify.py:277`; regression `tests/temporac/test_certificate_tau.py:115` |
| Three-tau invariant ledger and provenance comparison | PASS | `src/pams/temporac/certify.py:650`; regression `tests/temporac/test_certificate_tau.py:42` |
| F4 `66/D` endpoint-intersection mask | PASS | `src/pams/temporac/certify.py:311`, `src/pams/temporac/certify.py:315`, `src/pams/temporac/certify.py:321`; regression `tests/temporac/test_certificate_tau.py:78` |
| Exact right ownership | PASS | `src/pams/temporac/certify.py:380`, `src/pams/temporac/certify.py:390`, `src/pams/temporac/certify.py:430`; regression `tests/temporac/test_certificate_tau.py:107` |
| Joint target NPZ/receipt loader with expected receipt hash | PASS | `src/pams/temporac/receipts.py:459`, `src/pams/temporac/receipts.py:474`, `src/pams/temporac/receipts.py:480`, `src/pams/temporac/receipts.py:482` |
| Exclusive-create/non-overwrite primitive | PASS | `src/pams/temporac/hashio.py:301`, `src/pams/temporac/hashio.py:306`, `src/pams/temporac/hashio.py:311`; regression `tests/temporac/test_contract_hashio.py:107` |
| Learner label firewall | PASS in reviewed scope | learner/certificate modules import no vault/evaluator labels; privileged fields remain in `src/pams/temporac/trusted_packer.py:134` and evaluator-only paths |
| No legacy/WARP imports | PASS | case-insensitive scan of all ten reviewed source files returned zero `warp_phase`, legacy PAMS data/model/loss/training/period/consensus imports |
| K7 certificate-to-vault primitive-unit bridge | FAIL | F4 |

## Independent verification

The independent probe did not import test helpers and used only deterministic synthetic objects.

- Canonical X0 `U0003`, block 0, linear, offset 0: `CERTIFIED`, event count 10, seven NPZ members.
- Canonical artifact SHA-256: `efe9d111001fefea45dc55e22f3111ba80fdddc32d6d8ad30dba6b2321f56a16`.
- Canonical receipt SHA-256: `7f9390a7bb8b073d30cdfec02ceaa2515e93c3df00a2817f39523931d22d5a16`.
- Joint load with the expected receipt digest: PASS.
- Second exclusive write to the same path: rejected.
- Broken zero phase: `ABSTAIN`; all target arrays empty; builder rejected artifact construction.
- Ordinary direct `TargetProvenance` construction and ordinary target `dataclasses.replace`: rejected.
- Post-certification pulse mutation: accepted incorrectly (F1).
- Noncanonical duck-typed X0 view: accepted incorrectly (F2).
- Natural identity and teacher relabel probes: accepted incorrectly (F3).

## Tool results

All definitive commands used the repository `.venv`, `PYTHONPATH=src`, `PYTHONDONTWRITEBYTECODE=1`, and pytest cache disabled.

| Tool | Result |
|---|---|
| Targeted five-file pytest | `46 passed, 1 skipped in 55.31s` |
| Full `tests/temporac` pytest | `109 passed, 1 skipped in 140.17s` |
| Ruff 0.16.3, 15 specified code/test files | `All checks passed!` |
| Mypy 1.20.2, repository configuration, 15 specified code/test files | `Success: no issues found in 15 source files` |

The sole pytest skip is `tests/temporac/test_contract_hashio.py:153`: Windows did not permit symlink creation. This is a platform skip, not a passing symlink witness.

Local verification runtime was Windows, CPython 3.12.13, NumPy 1.26.4, SciPy 1.17.1, and CPU PyTorch 2.13.0. It is not the frozen Linux/CUDA runtime required by `refine-logs/temporac/FINAL_PROPOSAL.md:1075`; these results cannot serve as P3/runtime evidence.

## Audited input hashes

Canonical hash of the LF-terminated, key-sorted JSON input-hash list: `f60c7dd1c475929831711568807e9c3f98be76d33e4f75a2419f80045415b90f`.

| Path | Final SHA-256 |
|---|---|
| `refine-logs/temporac/FINAL_PROPOSAL.md` | `3b265bbac15433301a1c47307e439d4e23fd8102ec8a39b5163d4ac475dd9391` |
| `refine-logs/EXPERIMENT_PLAN.md` | `4e9b2ba43f10fae2239bfbb2e0bc1d7c2dbf857302d6f457b4326f5140dea08e` |
| `src/pams/temporac/contract.py` | `5ab8fb62dc558ba78e95fc07e52c1f1a1afbff5b1c5003834079eea6143d745f` |
| `src/pams/temporac/types.py` | `32cd9450e836d97d4311d59ad504c987ff47970a134ac1f860d96951caea8228` |
| `src/pams/temporac/certify.py` | `a9a85390cbf26b51f4ee265393f9d0226988561c2776b7971508e87506501c98` |
| `src/pams/temporac/hashio.py` | `df872dc5709cc8814b5d11f87d4de8cc0351459cf5cdf04e19afd0a2baf2b13c` |
| `src/pams/temporac/receipts.py` | `ed1a993a832b5cb471a56ec6c9df39f5f6873efb88a33c223f4a4d741e946f15` |
| `src/pams/temporac/x0.py` | `5f4585425a83bf63779265db9004d210c5c2875a579ebc9a6c51297eb647c7a2` |
| `src/pams/temporac/teacher.py` | `30575fa64875aa6b77de157cf6af348b3b7900851a1c1742993842eea7928b30` |
| `src/pams/temporac/objective.py` | `09a77231a48a869cbc83e0aa1aa2502f86f4abbcbab8c84f2b2c5abbd729ac87` |
| `src/pams/temporac/trusted_packer.py` | `40d6e3d5be1b26b7ba0f0f7aadf8da7496c95c4e829face5c7e9faaee2cdf94e` |
| `src/pams/temporac/evaluator.py` | `81037186896309b093cd184b6e63542653f5a7e1ab063b78a9e63558369a47d2` |
| `tests/temporac/test_certificate_artifact.py` | `42ace7d4481972f8c6ebb52e86c63380479bf5bdd43308412b737b794e4b396f` |
| `tests/temporac/test_certificate_tau.py` | `f7e2f25ab93226568910b856167e441037213ba3dd937d6915587037b7c6a94b` |
| `tests/temporac/test_teacher_certificate.py` | `2ee9848bd85755cc8456cfbaeebeac6d0fbbcf65e74c3f4f984f57d0a0bb9a1d` |
| `tests/temporac/test_contract_hashio.py` | `421498d9af9cb9f3147b620cf611df34e50b27592854ab94ae81186fe298b90f` |
| `tests/temporac/test_metrics_evaluator.py` | `2e11bf4b80bf60c24e1f6f310c69ea2387f6952f06f357620bb06a9de6a2c179` |

Initial and final hashes were identical for all 17 audited inputs.

## Gate and authority ceiling

- This same-family review is provisional and cannot issue accepted semantic clearance.
- P0 certificate integration: **REVISE; no PASS receipt**.
- P1/P2/P2-METRIC/P3/S0 and every G/K gate: not advanced by this review.
- The plan-recorded `TEMPORAC_CONTRACT_AMENDMENT_003_P05M_FIXTURE` / P2-METRIC blocker remains untouched.
- Server authority: 0.
- Natural data authority: 0.
- Training authority: 0.
- Evaluator capability authority: 0.
- Launch authorization: 0.
- Paper-promotion authority: blocked; current partial-cache evidence remains pilot/appendix-only.

No server, natural data, test/sealed/heldout data, result directory, Git operation, code, test, proposal, plan, or MANIFEST was opened or modified by this review.
