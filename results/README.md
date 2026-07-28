# Result status

**Verification status: `partial_reproduction`**

A designated-server component smoke has been recorded by the operator for the
current pose preprocessing revision, including a strict 526-video manifest
and identity-audited pose caches for all 421 official training videos. The
path-free audit, clean identity ledger, and digest summary are under
[`results/server-smoke`](server-smoke/README.md). Thirteen training caches are
all-invalid and remain masked rather than being dropped.

Three-seed training and evaluation on the fixed 84-video development split
have now produced negative evidence for PAMS-Literal and the independently
inferred PAMS-SSHead. Neither method meets the preregistered thresholds on
development data. The 105-video test split has **not** received predictions or
evaluation, and no test metric exists. The development CLI did load and
deserialize the complete 337/84/105 manifest, so these development artifacts
are not claimed as proof that test-label bytes were never parsed and are not
eligible as a leak-free sealed result.

This repository is therefore a **partial reproduction**, not a completed or
validated reproduction. Baselines and executable ablation variants remain
incomplete. The development results below must not be moved into a test
leaderboard.

## UCFRep-526 development-only negative result

All rows use the same fixed 84-video development split and source revision
`f99f3fda91b8bf2c25fc75a42814d9c97df71ae0`. Standard deviations are sample
standard deviations across the three seeds. The thresholds are shown only as
a preregistered diagnostic; passing them on development data would not verify
the reproduction.

| Variant | Seed | NMAE | Raw MAE | RMSE | OBO |
|---|---:|---:|---:|---:|---:|
| PAMS-Literal | 42 | 0.638439 | 3.738095 | 5.387375 | 0.285714 |
| PAMS-Literal | 2026 | 0.641446 | 3.726190 | 5.449989 | 0.309524 |
| PAMS-Literal | 3407 | 0.909889 | 5.464286 | 7.378379 | 0.154762 |
| PAMS-Literal | mean ± sample SD | 0.729925 ± 0.155861 | 4.309524 ± 1.000071 | 6.071914 ± 1.131865 | 0.250000 ± 0.083333 |
| PAMS-SSHead (inferred) | 42 | 5.225998 | 20.000000 | 22.813425 | 0.047619 |
| PAMS-SSHead (inferred) | 2026 | 5.008007 | 19.154762 | 21.787666 | 0.035714 |
| PAMS-SSHead (inferred) | 3407 | 4.589608 | 17.476190 | 19.926054 | 0.071429 |
| PAMS-SSHead (inferred) | mean ± sample SD | 4.941204 ± 0.323411 | 18.876984 ± 1.284630 | 21.509048 ± 1.463711 | 0.051587 ± 0.018185 |

Both variants pass the `NMAE <= 0.228` and `OBO >= 0.666` diagnostic on
`0/3` individual seeds; neither three-seed mean passes. The aggregate's
10,000-sample paired bootstrap for `PAMS-SSHead - PAMS-Literal` gives an NMAE
difference 95% CI of `[+3.333701, +5.141728]` and an OBO difference 95% CI of
`[-0.269841, -0.130952]`, so the inferred head is materially worse in this
development run.

The compact path-free record, source aggregate digest, and two seed-2026
diagnostics are under
[`results/dev-negative`](dev-negative/README.md).
That directory also records the failed, independently inferred PE-scale v2
seed-2026 diagnostic. It weakened one positional shortcut indicator but did
not improve the literal-head development result and remained far outside the
acceptance thresholds.

## Standard UCFRep-526 fair table

| Method | Protocol | Status | NMAE | OBO |
|---|---|---|---:|---:|
| PAMS-Literal | 421/105 | dev negative available; test not run | — | — |
| PAMS-SSHead | 421/105 | dev negative available; test not run | — | — |
| RepNet | 421/105 | parity/data blocked | — | — |
| TransRAC | 421/105 | parity/data blocked | — | — |
| ESCounts | 421/105 | parity/data blocked | — | — |
| IVAC-P2L | 421/105 | parity/data blocked | — | — |
| PoseRAC-ICONIP24 | 421/105 | implementation/data blocked | — | — |
| JTSPS-count-only | 421/105 | protocol blocked | — | — |
| CountLLM-Lite | 421/105, non-comparable recipe | smoke-only / adapter incomplete | — | — |

## UCFRep-pose-110 fair table

This table is a target only. Sealed scoring is disabled until the canonical
89/21 official annotation identity has been independently frozen.

| Method | Protocol | Status | NMAE | OBO |
|---|---|---|---:|---:|
| PAMS-Literal | 89/21 | data blocked | — | — |
| PAMS-SSHead | 89/21 | data blocked | — | — |
| PoseRAC-v1 fair rewrite | 89/21 | annotation/parity blocked | — | — |
| GMFL | 89/21 | annotation/parity blocked | — | — |
| SPKDB | 89/21 | annotation/parity blocked | — | — |
| BIGC | 89/21 | annotation/parity blocked | — | — |

The deterministic `spectral-proxy` is a synthetic pipeline diagnostic and is
never eligible for either paper-comparison table.

The current label-free local safety evidence, including the intentionally
failed SSHead collapse diagnostic, is published under
[`results/safety`](safety/README.md). Its designated-server rerun and real
UCFRep preprocessing smoke are summarized separately under
[`results/server-smoke`](server-smoke/README.md).
