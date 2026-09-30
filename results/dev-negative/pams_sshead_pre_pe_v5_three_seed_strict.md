# PAMS-SSHead pre-PE v5: three-seed development audit

**Status:** failed inferred repair; partial reproduction

The pre-PE Period Head repair was trained independently on train337 for seeds
42, 2026, and 3407, then evaluated on the same frozen dev84 protocol. No
development poses or labels entered either training stage, and test105 remains
unopened.

| Seed | NMAE | OBO | MAE | RMSE | Zero counts | Period 128 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.740292 | 0.261905 | 4.392857 | 6.073949 | 29 | 27 |
| 2026 | 0.673120 | 0.250000 | 4.285714 | 5.742490 | 32 | 28 |
| 3407 | 0.707164 | 0.238095 | 4.380952 | 5.740416 | 22 | 29 |
| Mean ± sample SD | 0.706859 ± 0.033587 | 0.250000 ± 0.011905 | 4.353175 ± 0.058725 | 5.852285 ± 0.191969 | — | — |

All three seeds fail both frozen acceptance thresholds (`NMAE <= 0.228`,
`OBO >= 0.666`). The failure is consistent rather than seed-specific.
Removing the positional shortcut materially reduces the catastrophic
overcounting of the original inferred SSHead, but the pointwise pre-PE Head
instead favors low-frequency or constant streams.

The machine-readable aggregate and every artifact binding are in
[`pams_sshead_pre_pe_v5_three_seed_strict.json`](pams_sshead_pre_pe_v5_three_seed_strict.json).
Complete per-video predictions and period streams are in the adjacent
`pams_sshead_pre_pe_v5_seed{42,2026,3407}_strict.json` files.
