# PAMS v14 seed 2026: terminal train337-only rejection

## Status

**Partial reproduction; rejected before development evaluation.**

This package records the completed seed-2026 v14 encoder and independently
inferred SSHead training run. The frozen target-free terminal gate failed
5 of 13 criteria. It is not a paper-table result, a successful reproduction
claim, or evidence about UCFRep test accuracy.

The encoder trained for all 150 epochs and the inferred SSHead trained for all
30 epochs. The failure is scientific rather than operational:

| Frozen train-only diagnostic | Threshold | Observed | Result |
|---|---:|---:|---|
| Encoder period mode share | `< 0.25` | 0.676558 | fail |
| Encoder time-scale median relative error | `<= 0.15` | 1.000000 | fail |
| SSHead period mode share | `< 0.25` | 0.732938 | fail |
| SSHead stream standard deviation, median | `>= 0.05` | 0.003972 | fail |
| SSHead time-scale median relative error | `<= 0.15` | 0.750000 | fail |

The encoder produced only three period values over 337 training videos:
`17.0667` frames for 228 videos, `32` for 93, and the lower boundary `4` for
16. The SSHead likewise produced only three values: `18.2857` for 247 videos,
`36.5714` for 74, and `4` for 16. The head's parameters did change
(`L2 delta = 0.814082`) and no optimizer step had zero gradient, but the
resulting scalar streams remained nearly flat and poorly time-scale
equivariant.

These diagnostics are independently inferred anti-collapse checks. They are
not author-disclosed validation criteria and do not themselves measure count
accuracy. They are used here only to prevent development-label feedback from
concealing a training-side failure.

## Scope and leakage audit

- Dataset protocol: UCFRep-526
- Seed: `2026`
- Training partition used: 337 videos
- Training count or action targets loaded: no
- Dev84 video, pose, identity, or target inputs loaded: no
- Dev84 prediction or scoring authorized: no
- Sealed test105 accessed or authorized: no
- Gate mode: read-only checkpoint evaluation
- Network during gate: disabled
- Container root filesystem: read-only
- GPU used: one NVIDIA RTX A6000

Only the frozen config, terminal checkpoints and progress logs, the
checkpoint-bound train337 pose cache, and source receipts were mounted. The
exact gate artifact contains no server address, account, secret, absolute
server path, raw video, model checkpoint, or private pose cache.

## Provenance

- Training algorithm source: `df2e7cdfcc91bb436eb99ee10852ddd66ee971e4`
- Terminal gate source: `b6d3b0749fe6fdf1b2b328bdcda51177e3e6e08b`
- Config fingerprint:
  `42baeb7b4ff53ff519776ad4b8a2bf44183675810e7ac89f7e3d9918d08c4d45`
- Encoder checkpoint SHA-256:
  `fd1ac8308f1e84b9a68c4131d8294e76d2f09a03607ae838809c6facec180f68`
- SSHead checkpoint SHA-256:
  `085273ba0665984687fa6230d6ada1a9a2db0d87b2a404b2c6f20364b648e261`
- Final gate artifact SHA-256:
  `781fff3cc8cb8414c2fb5e4bf833a17fa3dd8501b0fb3d574bc1c525ff677a75`

The PAMS paper does not disclose a Period Head training loss or gradient path.
This repository's SSHead loss is therefore an explicitly inferred completion,
not an author implementation.

## Files

- `final-train-gate.json`: exact public-safe terminal gate artifact
- `final-train-gate.json.receipt.json`: immutable artifact receipt
- `training-and-gate-summary.json`: compact training, gate, and access record
- `SHA256SUMS`: package integrity hashes
