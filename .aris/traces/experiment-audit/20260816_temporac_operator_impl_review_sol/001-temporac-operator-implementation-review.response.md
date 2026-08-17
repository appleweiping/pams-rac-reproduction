# Reviewer response

**Verdict:** `IMPLEMENTATION_SEMANTICS_PASS` with `0` semantic blockers.  
**Artifact authority:** `BLOCKED_NON_AUTHORITATIVE`, `authority=0`.  
**Assurance:** `same-family`, `provisional`.

An independent trace-only harness that imports no `pams.temporac` rebuilt all 10,368 rows from the normative bytes. It verified 10,368 unique 10-byte keys, the exact `P(e)` construction and `D(e)[0]`, the exact binary32 inactive lookup/cap, lower-middle and flat amplitude-one plateaus, run/reset geometry, all three masks, response zeros outside masks, the 127-edge window schedule, binary64 Neumaier NOLA, one decode per row, one-to-one association, every candidate row hash, and all four normative roots. The worst margins are `0.1499999761581421` positive and `0.10000002384185791` negative; the minimum positive NOLA denominator is `0.0011528184020849772`.

Fresh generation under linux/amd64 CPython 3.12.4 + NumPy 2.1.0, using `python@sha256:e8be0ea148390d08bc077840cf87ac6a538d80b0ea1e8752b3e3982987cd0a53`, produced the same five files and each file was byte-identical to the workspace candidate. Existing-directory and existing-file probes both failed closed; exclusive create preserved prior bytes.

Targeted pytest passed `13/13`; full TempoRAC pytest passed `109` with one Windows symlink-permission skip. Ruff passed. Scoped Mypy and production-package Mypy passed. Full package+tests Mypy emitted one out-of-scope diagnostic at `tests/temporac/test_x0.py:146`; this prevents an all-green repository-wide static claim but does not invalidate scoped operator semantics.

The candidate cannot be promoted. The canonical material does not uniquely freeze manifest row schema, per-array hash preimages, or the aggregate manifest preimage. The candidate's own schema/receipt correctly says non-authoritative, `authorizes=[]`, P2/S0 not claimed. Review cannot manufacture that missing normative choice, so canonical/P2/S0/gate authority remains zero.
