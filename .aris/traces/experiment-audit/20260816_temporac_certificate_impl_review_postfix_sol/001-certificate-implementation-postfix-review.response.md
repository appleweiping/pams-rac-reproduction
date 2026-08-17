# Reviewer response

Overall verdict: **REVISE (blocking; same-family provisional)**.

Four scientific-contract blockers remain:

1. `src/pams/temporac/types.py:71-82` produces owning NumPy copies whose write flag can be re-enabled. A canonical certified X0 pulse was moved from edge 8 to edge 9 after certification; `src/pams/temporac/receipts.py:529-560` built it and `src/pams/temporac/hashio.py:666-699` loaded it. Accepted tampered artifact SHA-256: `8db7e6f86228c96c211b880d56409e4331bc9356e26553d93220d5358bac1c11`.
2. `src/pams/temporac/certify.py:676-732` accepts a duck-typed noncanonical X0 object and does not enforce the frozen split/resampler/offset inventory. `tests/temporac/test_certificate_artifact.py:222-256` currently treats such a `SimpleNamespace` path as positive.
3. `src/pams/temporac/certify.py:735-760` binds an independently supplied track, feature record, and free teacher digest without proving that the arrays came from that feature and selected teacher. The same track certified under unrelated natural identities and teacher digests.
4. `src/pams/temporac/evaluator.py:239-310` has no certified target/pulse/source-clock input and therefore cannot execute the K7 primitive-unit check required by `refine-logs/temporac/FINAL_PROPOSAL.md:1384-1398`.

Passed checks: builder has no provenance override arguments; exact seven-member deterministic target NPZ; closed ten-key target receipt; ABSTAIN zero artifact; phase-class transitive/wrap union; complete three-tau discrete ledger/provenance comparison; F4 `66/D` endpoint mask; right ownership; hash-bound joint target loader; O_EXCL overwrite refusal; reviewed label firewall; zero legacy/WARP imports.

Verification: targeted pytest `46 passed, 1 skipped`; full `tests/temporac` `109 passed, 1 skipped`; Ruff `All checks passed`; Mypy `Success: no issues found in 15 source files`. The single skip was Windows symlink creation. The Windows CPU environment is not the frozen Linux/CUDA runtime and grants no P3 evidence.

Canonical valid X0 output: artifact SHA-256 `efe9d111001fefea45dc55e22f3111ba80fdddc32d6d8ad30dba6b2321f56a16`; receipt SHA-256 `7f9390a7bb8b073d30cdfec02ceaa2515e93c3df00a2817f39523931d22d5a16`.

Gate ceiling: P0 certificate integration remains REVISE; P1/P2/P2-METRIC/P3/S0 and every downstream G/K gate are not advanced; server/data/training/evaluator/launch authorities remain zero; paper promotion remains blocked.

Full evidence and all 17 input hashes are in `refine-logs/temporac/TEMPORAC_CERTIFICATE_IMPLEMENTATION_REVIEW_POSTFIX_20260816.md` and `.json`.
