# Local-frequency v2 target-free selector and dev84 negative result

**Status:** `partial_reproduction`

**Classification:** independently inferred and `exploratory-derived`

**Paper-table eligibility:** ineligible

**Sealed test:** test105 was not mounted or accessed and remains sealed

This package records the frozen target-free selection and first successful
UCFRep dev84 evaluation of local-frequency v2. The method is not an
author-disclosed PAMS readout. It uses deterministic synthetic truth and
unlabeled train337 consistency to select one candidate; it does not use
dataset action or count labels during selection or prediction.

## Dev84 result

| Videos | NMAE | OBO | MAE | RMSE | Exact | Joint gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 84 | 0.453804 | 0.369048 | 3.273810 | 5.029674 | 0.273810 | Fail |

The frozen acceptance thresholds are `NMAE <= 0.228` and `OBO >= 0.666`.
This candidate fails both, so it is a negative development result rather than
a verified reproduction.

The paired-row, 10,000-sample bootstrap 95% confidence intervals are:

- NMAE: `[0.381032, 0.529153]`
- OBO: `[0.261905, 0.476190]`
- MAE: `[2.500000, 4.130952]`
- RMSE: `[3.719279, 6.361178]`
- Exact: `[0.178571, 0.369048]`

All five point metrics and all five bootstrap intervals were independently
recomputed from the 84 per-video rows with bootstrap seed 2026. They match the
published evaluation artifact exactly.

## Frozen selector

The 512-candidate selector chose:

`xyz.trim0.norm.c1.mean.scale_min`

Its complete rank key is:

`[0, 0.0, 0.0, 0.030882947198275862, 0.14661458333333333, 1.4201388888888888, -11, "xyz.trim0.norm.c1.mean.scale_min"]`

The rank-one candidate is formally non-degenerate and obtains exact synthetic
count-sweep and stress predictions. However, count 2 is the training
prediction mode for 409/576 original-plus-transformed observations
(`71.0069%`), close to the frozen inclusive degeneracy threshold of `75%`.
Its boundary share is also `71.0069%`. This warning was visible before dev
prediction and is retained rather than hidden.

The selector artifact is 1,050,892 bytes with SHA-256
`0bcad7d0fe270f1e71db465bd2e0f4ac861dfc1167b60283603d4624786e18db`.
The initial run, deterministic replay, and sanitized publication copy are
byte-identical. The publication copy was produced from a label/data-free
source view; its receipt is included.

## Prediction behavior

The frozen dev prediction contains 84 unique records. Forty-nine predictions
are exactly 2, and the median is 2. Six videos have no valid pose frames:

- `v_Biking_g16_c02`
- `v_BreastStroke_g01_c01`
- `v_BreastStroke_g12_c02`
- `v_PlayingViolin_g08_c04`
- `v_Rowing_g01_c03`
- `v_Rowing_g14_c07`

These observations explain part of the weak result, but they are post-freeze
failure analysis. They cannot be used to rewrite or retune this evaluated
candidate into an eligible result.

## Isolation and failed first launch

Selection ran with train337 poses only, no network, and no development or
test input. Prediction ran with the frozen selector, canonical count-free
dev inputs, and the 84 dev pose caches; dev targets and every test105 input
were absent. Isolated scoring ran with only the frozen predictions, their
receipt, and dev targets; selector, pose, dev input, and every test105 input
were absent.

The first runner implementation at commit
`b6bd3876a684123776d04044b35dba2b715d2748` failed on Linux before producing
predictions because the dynamically loaded selector function could not be
pickled by `ProcessPoolExecutor`. That failed launch produced no prediction or
evaluation artifact and did not access labels. Host-specific failure logs are
not copied into this package. Commit
`2310d093603ffc031e74ca6a5fbc582471f0a78e` made prediction process-safe by
using one worker; the frozen selector, selected method, and scoring protocol
were unchanged.

## Provenance and files

- Selector source commit:
  `8e3f33bb9e68ccaf19fe46773888c013d2de3951`
- Successful dev runner commit:
  `2310d093603ffc031e74ca6a5fbc582471f0a78e`
- `selector.json` is the complete sanitized 512-candidate selector artifact.
- `predictions.json` and `evaluation.json` contain all 84 per-video rows.
- The prediction and evaluation receipts bind their byte streams.
- Selector, predict, and score status/source-view receipts document the
  separate mount boundaries.
- `dev-attempt.json` is the pre-score run declaration; the later
  `score-status.json` records successful isolated scoring.
- `audit-summary.json` records independent recomputation, the hash chain,
  failure disclosure, and publication-safety scan.
- `SHA256SUMS` binds every public file except itself.

The copied generated artifacts are byte-identical to their source artifacts.
The final package was scanned for the server address/port, account name,
private-key material, SSH locations, and host filesystem paths; none are
present. Logical container mount destinations such as `/inputs` and `/source`
remain in sanitized status files because they document the isolation policy
without identifying a host path.
