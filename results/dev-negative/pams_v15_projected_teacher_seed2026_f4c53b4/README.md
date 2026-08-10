# PAMS v15 projected-teacher seed 2026: terminal encoder rejection

## Status

**Partial reproduction; SSHead training rejected before development
evaluation.**

This package records a complete 150-epoch train337-only causal ablation. It
changes exactly one semantic configuration field relative to v14: after the
10-epoch pose-energy warm-up, the TCC period teacher is taken from the
pre-position-encoding projected-pose velocity vector ACF instead of the
post-PE Transformer embedding.

The change was authorized by a frozen read-only counterfactual on the v14
terminal checkpoint. The projected teacher passed all four preregistered
readiness criteria. The v15 encoder then completed all 150 epochs, but its
terminal post-PE representation failed three of the ten frozen criteria:

| Terminal train-only diagnostic | Threshold | Observed | Result |
|---|---:|---:|---|
| Projected teacher mode share | `< 0.25` | 0.089021 | pass |
| Projected teacher scale median error | `<= 0.15` | 0.000000 | pass |
| Post-PE period mode share | `< 0.25` | 0.712166 | fail |
| Post-PE scale median error | `<= 0.15` | 0.500000 | fail |
| Cross-path median relative error | `<= 0.25` | 0.687500 | fail |

The post-PE estimator assigned period `8` frames to 240 of 337 training
videos, despite the projected teacher retaining 56 distinct period values.
The training objective therefore reduced its loss without transferring the
teacher's diverse, time-scale-equivariant period structure into the
Transformer output.

These checks are independently inferred anti-collapse diagnostics, not
author-disclosed validation and not count-accuracy measurements. The PAMS
paper still does not disclose a Period Head loss or gradient path.

## Scope and leakage audit

- Protocol: UCFRep-526
- Seed: `2026`
- Encoder optimization data: 337 label-free training pose caches
- Count or action targets loaded by training: no
- Dev84 media, poses, action labels, or count targets loaded: no
- Sealed test105 media, poses, action labels, or count targets loaded: no
- Label-free dev/test identity sidecars parsed by training to bind the
  official 337/84/105 partition: yes
- Dev84 prediction or scoring authorized: no
- Test105 evaluation authorized: no
- SSHead training authorized or executed: no

Both read-only gates had no interface or mount for any dev/test identity,
media, pose, or target input. All formal containers used no network, a
read-only root filesystem, dropped all Linux capabilities, and mounted only
the explicitly audited inputs.

## Training facts

- Encoder epochs: 150 contiguous records
- Optimizer steps: 1,500
- Period sources: 10 pose-energy epochs, 140 projected-teacher epochs
- First/final loss: `6.302960 / 3.066710`
- Cross-cluster negative shortfall: `0`
- Maximum GPU0 memory: `5,073 MiB`
- Encoder checkpoint SHA-256:
  `aa1750154e0ba36d41188cc023bb382db83db9f5c69b1867f310cdc636a48e19`
- Progress SHA-256:
  `70226a5f01a3d9e8b688fde53736c397a135c242d8e15ce429a36615552029c4`

## Provenance

- Training algorithm source:
  `df2e7cdfcc91bb436eb99ee10852ddd66ee971e4`
- v15 config source:
  `550a5ea01565d43503ce8d137dc92f82f4a9002b`
- Terminal gate source:
  `f4c53b4094049cf34f7c23c6978f051a1ce7493b`
- v15 config SHA-256:
  `3e9d641c0e8b4710c3f0542b823b4a51d319765f7e6c72f7f6195f8971fec688`
- v15 config fingerprint:
  `666f01ece7d179d4ac8c61f9d1dc63aee5e8575c0f79a01d5d3d7e20e9d6242b`
- Teacher-readiness artifact SHA-256:
  `96da147df273c5cf82bcaefebd1b131bac21ae110837e80d2c892217c850e4c6`
- Terminal gate artifact SHA-256:
  `0f318d67bc507a1147df469ff983740e3172f16c85326552c79d35d33f1ddae3`

## Files

- `v14-terminal-dual-path-counterfactual.json`: exact teacher-readiness artifact
- `v14-terminal-dual-path-counterfactual.json.receipt.json`: immutable receipt
- `v15-terminal-dual-path-gate.json`: exact v15 terminal gate artifact
- `v15-terminal-dual-path-gate.json.receipt.json`: immutable receipt
- `training-and-gate-summary.json`: compact training, access, and decision record
- `SHA256SUMS`: package integrity hashes
