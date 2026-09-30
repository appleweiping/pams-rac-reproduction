# Local-frequency v2 paired dev84 baseline comparison

**Status:** `partial_reproduction`

**Scope:** frozen UCFRep-526 dev84 predictions only

**Paper-table eligibility:** ineligible

**Sealed test:** test105 was not mounted, accessed, predicted, or evaluated

This package compares the target-free-selected local-frequency v2 diagnostic
against four already-frozen baseline results on the exact same 84 development
videos. No model was retrained and no prediction was changed. The formal
comparison ran twice on the designated server with a read-only root,
`--network none`, evaluation artifacts as its only data inputs, and produced
byte-identical outputs.

All differences below are `local-frequency v2 - baseline`. Confidence
intervals are paired-video, 10,000-sample bootstrap 95% intervals with seed
2026.

| Baseline | Baseline runs | Baseline NMAE | NMAE difference [95% CI] | Baseline OBO | OBO difference [95% CI] |
|---|---:|---:|---:|---:|---:|
| Local-frequency v1 | 1 | 0.514566 | -0.060762 [-0.144508, 0.017356] | 0.333333 | +0.035714 [-0.035714, 0.119048] |
| RepNet official ckpt-70 sanity | 1 | 0.434505 | +0.019299 [-0.103094, 0.133638] | 0.583333 | -0.214286 [-0.333333, -0.095238] |
| ESCounts official current sanity | 1 | 0.352419 | +0.101386 [-0.023594, 0.214437] | 0.595238 | -0.226190 [-0.345238, -0.107143] |
| JTSPS-count-only inferred | 3-seed mean | 0.487634 | -0.033829 [-0.140634, 0.068269] | 0.408730 | -0.039683 [-0.170635, 0.091270] |

Local-frequency v2 itself is NMAE/OBO `0.453804 / 0.369048`. Its NMAE
improvement over v1 is not statistically resolved by this sample, although
its Exact-rate difference is `+0.095238` with interval
`[+0.023810, +0.178571]`. Against RepNet and ESCounts, its OBO deficit is
statistically resolved; its MAE is also worse by `+1.035714`
(`[+0.154762, +1.964286]`) and `+1.119048`
(`[+0.333333, +1.928571]`), respectively. None of the five metric
differences against the three-seed JTSPS estimand is statistically resolved.

For the JTSPS comparison, each bootstrap replicate computes the metric for
each seed and then averages across seeds. The three seed-by-video rows are not
treated as 252 independent observations. RMSE follows the same
metric-per-seed-then-mean policy.

`comparison.json` contains the source hashes, canonical video/target identity
hashes, all point differences, all confidence intervals, and the 84 paired
component rows needed to reconstruct them. `formal-status.json` records the
evaluation-only mounts, image/environment hashes, and deterministic replay.
`source-view.receipt.json` binds the single committed comparison script.

The result does not pass the frozen reproduction gate and does not authorize
test105. RepNet and ESCounts remain official-checkpoint development sanities,
not original-protocol parity claims; JTSPS-count-only and both local-frequency
variants are independently inferred diagnostics.
