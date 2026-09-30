# PAMS v8 frozen-readout diagnostics, seed 2026

These are real UCFRep dev84 experiments on the frozen v8
longest-contiguous-track pose cache. They are development diagnostics, not
paper-table or test results. The sealed 105-video test videos, pose, counts,
and targets were not mounted or evaluated.

## Results

| Readout | NMAE | Raw MAE | RMSE | OBO | Exact |
|---|---:|---:|---:|---:|---:|
| PAMS-Literal | 0.664627 | 4.071429 | 5.400617 | 0.261905 | 0.154762 |
| PAMS-SSHead v8 reference | 0.643123 | 4.226190 | 6.073949 | 0.297619 | 0.154762 |
| Synthetic-frozen local frequency | 0.529058 | 3.523810 | 4.932883 | 0.333333 | 0.178571 |

The local-frequency NMAE 95% interval is `[0.453193, 0.606427]`; its
OBO interval is `[0.238095, 0.440476]`. Relative to the same v8 SSHead
predictions, its paired `local-frequency - SSHead` NMAE difference is
`-0.114065`, with a 10,000-sample 95% interval of
`[-0.193770, -0.031096]`. The OBO difference is `+0.035714`, but its
interval `[-0.047619, +0.119048]` includes zero. It has lower absolute
error on 46 videos, ties on 23, and is worse on 15.

PAMS-Literal does not improve the learned inferred SSHead: its NMAE/OBO
differences are `+0.021503 / -0.035714`, and both intervals include zero.

## Interpretation

The local-frequency method is a deterministic pose-only readout. Its
parameters were frozen by the repository's synthetic count/stress suite
before this v8 result was known. The 50-case replay exactly reproduced the
frozen selector and expected synthetic metrics:
`frame_centered.min064.median`, MAE `0.02`, NMAE `0.0025`, Exact `0.98`,
and OBO `1.0`.

It does not use the PAMS Encoder or SSHead and is not an author-disclosed
PAMS component. It is permanently ineligible for the paper comparison
table. Its purpose here is causal diagnosis: the v8 pose sequence retains
usable periodic signal, while the learned SSHead/period stream remains the
primary bottleneck.

The identical frozen readout previously obtained NMAE `0.514566` and OBO
`0.333333` on the v2 pose cache. The v3-minus-v2 paired differences are
NMAE `+0.014492`, interval `[-0.015202, +0.049963]`, and OBO `0.0`,
interval `[-0.047619, +0.047619]`. Therefore the longest-track correction
did not improve this readout, despite improving the learned v8 model over
matched v7.

## Isolation and provenance

- Revision: `07192debb7f5748c53a1c6d4f0c0d3f16228e229`.
- Encoder reused by Literal:
  `6817fe468d76c3fa2182dd831695d6fe3d6da208a2b7f8dfa90cffa6872e8053`.
- Pose fingerprint:
  `3dd0388320095796f42aa82d904073b6478b073ed351394ef8d03c30798a0116`.
- Prediction containers had no development targets.
- Literal saw only the committed test identity sidecars required to verify
  the 337/84/105 firewall; it saw no test media, pose, or labels.
- Local-frequency prediction saw neither train nor test identities.
- Predictions and receipts were made read-only before separate CPU scorers
  opened dev84 targets.
- Both prediction containers and both score containers exited zero with
  networking disabled and read-only root filesystems.
- Server artifact-manifest SHA-256:
  `84eace40837b76cde8feba539773214baf3299c3dc375101fd7e0a2a6d151a3a`.

`literal.predictions.json` and `local-frequency.predictions.json` are the
target-free 84-row artifacts. Their matching evaluations contain all
per-video scored rows and 10,000-sample confidence intervals.
`paired-comparisons.json` contains the three matched comparisons. The first
comparison wrapper accidentally included the NVIDIA base-image banner on
stdout; that invalid stream was retained only on the server. The clean
artifact was recomputed with an explicit Python entrypoint, as recorded in
`paired-comparison-recovery.txt`; no experimental artifact changed.
`provenance.json` binds the four input evaluation hashes. The clean rerun did
not retain a separate container receipt, so that omission is disclosed there;
an independent recomputation reproduced every reported difference and
interval exactly.

This evidence does not authorize seeds 42/3407 or test105. Repository status
remains `partial_reproduction`.
