# Result status

**Verification status: `no_results`**

No UCFRep benchmark measurements are published in this revision. The
designated GPU server and UCF101/UCFRep video files are not connected, so a
numeric table here would be fabricated or produced under an unapproved
environment.

## Standard UCFRep-526 fair table

| Method | Protocol | Status | NMAE | OBO |
|---|---|---|---:|---:|
| PAMS-Literal | 421/105 | implementation only | — | — |
| PAMS-SSHead | 421/105 | implementation only | — | — |
| RepNet | 421/105 | parity/data blocked | — | — |
| TransRAC | 421/105 | parity/data blocked | — | — |
| ESCounts | 421/105 | parity/data blocked | — | — |
| IVAC-P2L | 421/105 | parity/data blocked | — | — |
| PoseRAC-ICONIP24 | 421/105 | implementation/data blocked | — | — |
| JTSPS-count-only | 421/105 | protocol blocked | — | — |
| CountLLM-Lite | 421/105, non-comparable recipe | resource blocked | — | — |

## UCFRep-pose-110 fair table

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
