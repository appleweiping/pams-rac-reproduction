# Result status

**Verification status: `no_results`**

A designated-server component smoke has been recorded by the operator for the
current pose preprocessing revision, including a strict 526-video manifest
and identity-audited pose caches for all 421 official training videos. The
path-free audit, clean identity ledger, and digest summary are under
[`results/server-smoke`](server-smoke/README.md). Thirteen training caches are
all-invalid and remain masked rather than being dropped. No UCFRep benchmark
measurement has been produced. Full encoder/SSHead training for the three
preregistered seeds and the sealed 105-video evaluation have not run.

This repository is therefore a **partial implementation**, not a completed or
validated reproduction. Baselines and executable Table 2 variants also remain
incomplete. Inserting a numeric UCFRep result here would be fabrication.

## Standard UCFRep-526 fair table

| Method | Protocol | Status | NMAE | OBO |
|---|---|---|---:|---:|
| PAMS-Literal | 421/105 | not run; training/evaluation pending | — | — |
| PAMS-SSHead | 421/105 | not run; training/evaluation pending | — | — |
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
