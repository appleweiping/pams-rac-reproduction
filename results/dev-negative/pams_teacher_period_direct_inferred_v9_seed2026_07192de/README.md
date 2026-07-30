# PAMS-TeacherPeriodDirect-inferred v9, seed 2026

This package records a completed **negative UCFRep dev84 diagnostic**. It
does not establish a paper-level PAMS result. The inferred readout obtained
NMAE `2.860120` and OBO `0.166667`, so it is retained as evidence against
using the projected-pose period estimator directly as a count head.

The experiment used the exact frozen v8 encoder checkpoint, but the readout
is an independently inferred completion:

`forward_with_pre_pe -> estimate_period_from_projected_pose -> confidence/valid guard -> half_up((valid_frames - 1) / period)`

It reads the encoder's pre-positional-encoding input projection, not the
Transformer output. The paper does not disclose this readout or a loss that
trains its Period Head. This result is therefore permanently ineligible for
the paper comparison table.

## Dev84 result

| Method | NMAE | Raw MAE | RMSE | OBO | Exact |
|---|---:|---:|---:|---:|---:|
| PAMS-TeacherPeriodDirect-inferred | 2.860120 | 13.309524 | 19.871013 | 0.166667 | 0.083333 |

The 10,000-sample paired bootstrap 95% intervals are:

- NMAE: `[2.107282, 3.701036]`
- Raw MAE: `[10.250000, 16.512202]`
- RMSE: `[15.936612, 23.426271]`
- OBO: `[0.095238, 0.250000]`
- Exact: `[0.035714, 0.142857]`

The formal CLI replay reproduced all 84 `video_id`, video hash, raw count,
rounded count, period, confidence, and valid-frame fields exactly. Every
metric and confidence interval also matched the earlier one-off run exactly.

## What the pre-development gate established

Before any dev input was mounted, the label-free gate used only train337,
the frozen config, and the exact encoder. It passed all frozen checks:

- all synthetic counts 2--40 were exact;
- all 12 synthetic stress cases were within one count;
- only `0.6231%` and `9.0343%` of positive-confidence train periods hit
  the 4- and 128-frame limits;
- the rounded-period mode fraction was `9.0343%`;
- zero-confidence rows never produced a nonzero count.

That gate established implementation sanity, not real-video accuracy. Its
perfect single-frequency synthetic result is largely tautological for this
frequency estimator and did not predict dev84 behavior.

The first gate attempt is preserved because it failed before inference: the
checkpoint loader correctly required explicit bound provenance. The recovery
attempt supplied that provenance and records `recovery_of` plus the failure
reason. Neither attempt mounted dev, test, or target inputs.

## Failure diagnosis

On dev84, 60 predictions overcounted, 17 undercounted, and 7 were exact.
Overcounts contributed 1,056 of 1,118 total absolute-count errors
(`94.45%`) and `95.43%` of the summed normalized absolute error. The dominant
failure is high-frequency or harmonic aliasing, not integer rounding:
changing `(T-1)/period` to the plausible off-by-one alternatives, or changing
half-up to bankers rounding, changes zero of 84 rounded predictions.

The result is also substantially worse than both frozen-v8 diagnostics
already published from the same dev84 split:

| Diagnostic | NMAE | OBO |
|---|---:|---:|
| PAMS-SSHead v8 | 0.643123 | 0.297619 |
| Synthetic-frozen local frequency | 0.529058 | 0.333333 |
| TeacherPeriodDirect-inferred | 2.860120 | 0.166667 |

The local-frequency diagnostic is pose-only and is itself paper-table
ineligible; the comparison is causal diagnosis, not a leaderboard claim.
The poor teacher should not be used to supervise another head.

## Isolation and recovery

- Dataset/protocol: UCFRep 526, fixed dev84 only, seed 2026.
- Frozen encoder source revision:
  `07192debb7f5748c53a1c6d4f0c0d3f16228e229`.
- Formal CLI implementation revision:
  `f7108edb24b70895ae1404eb354595208828a5dd`.
- Encoder checkpoint SHA-256:
  `6817fe468d76c3fa2182dd831695d6fe3d6da208a2b7f8dfa90cffa6872e8053`.
- Prediction had no dev targets and no test media, pose, or labels.
- The formal prediction container saw committed test-identity sidecars only
  to enforce the 337/84/105 firewall.
- Scoring ran separately from frozen predictions and had no pose or encoder.
- Prediction and scoring ran with networking disabled.
- Test105 was not evaluated and remains sealed.

The formal orchestration status initially says `failed` because a host-side
equivalence helper used `zip(strict=True)` under Python 3.9 *after* prediction
and scoring had completed. The read-only recovery comparator changed no
experimental artifact and reran neither inference nor scoring. The preserved
recovery receipt and equivalence report make this distinction explicit.

## Package layout

- `predev-gate/`: failed predecessor and successful label-free recovery.
- `one-off/`: original target-free predictions, scripts, receipts, and
  separately scored evaluation.
- `formal-cli/`: formal CLI outputs, source-export receipt, mount-boundary
  audit, equivalence proof, and orchestration recovery receipt.
- `failure-analysis.json`, `formula-audit.json`, and
  `paired-comparisons.json`: independently recomputed, hash-bound diagnosis.
- `summary.json`, `status.json`, and `provenance.json`: public interpretation,
  authorization status, and byte-level bindings for every copied artifact.

Raw container inspect/create-ID files, host runner scripts, host-absolute
hash lists, server addresses, key paths, pose caches, videos, model weights,
and the dev target manifest are intentionally excluded.

Repository status remains `partial_reproduction`. This package does not
authorize seeds 42/3407 or any test105 access.
