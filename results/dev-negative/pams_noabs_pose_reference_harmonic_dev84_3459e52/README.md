# NoAbsPE pose-reference harmonic dev84 negative result

**Status:** `partial_reproduction`  
**Classification:** independently `inferred`, target-free prediction followed by
isolated dev84 scoring  
**Decision:** both evaluated seeds fail the frozen development thresholds  
**Sealed test consequence:** test105 was not mounted or accessed and remains
sealed

This package records the first development-set evaluation of the independently
inferred `pams-noabspe-pose-reference-harmonic-multiexpert-inferred-v1`
readout. It is not an author-disclosed PAMS component and is not eligible for a
paper table.

Before any dev84 prediction, both Encoder checkpoints passed all six frozen,
target-free pre-development checks. Passing that gate authorized prediction
only. Prediction ran without dev targets, and scoring ran separately with only
the frozen prediction artifact and dev targets. The prediction and scoring
mount audits in the JSON artifacts record that no test identity, pose, or
target input was mounted.

## Primary dev84 result

| Training seed | NMAE ↓ | OBO ↑ | MAE ↓ | RMSE ↓ | Exact ↑ | Joint gate |
|---:|---:|---:|---:|---:|---:|:---:|
| 2026 | 0.602454 | 0.309524 | 3.833333 | 5.307228 | 0.142857 | Fail |
| 42 | 0.608562 | 0.321429 | 3.904762 | 5.371884 | 0.119048 | Fail |
| Two-seed mean | 0.605508 | 0.315476 | 3.869048 | 5.339556 | 0.130952 | Fail |

The frozen thresholds are `NMAE <= 0.228` and `OBO >= 0.666`. Neither seed
passes either threshold. Therefore this is a negative development result, not
a verified reproduction, even though both candidates passed predev `6/6`.
Only two seeds are present here; the planned three-seed verification rule also
cannot be satisfied by this package.

Each evaluation file contains 10,000-sample paired-row bootstrap 95% confidence
intervals for that seed. `paired-seed-summary.json` additionally reports a
10,000-sample paired bootstrap of the seed-2026 minus seed-42 metric
differences. The intervals mostly cross zero, so there is no useful evidence
that one of these two seeds is materially better.

## Frozen predev authorization

| Criterion | Frozen threshold | Seed 2026 | Seed 42 |
|---|---:|---:|---:|
| Absolute frame-index linear-probe R² | `<= 0.10` | -0.1085032267 | -0.0510878326 |
| Zero-pose period confidence | `<= 0.10` | 0.0000000000 | 0.0000000000 |
| Random-pose period confidence | `<= 0.10` | 0.0216028684 | 0.0235623673 |
| Training dominant period-bin share | `<= 0.25` | 0.0937500000 | 0.1250000000 |
| Permuted/canonical embedding median cosine | `>= 0.95` | 1.0000000000 | 1.0000000000 |
| Synthetic-period median relative error | `<= 0.10` | 0.0000000149 | 0.0000000000 |

The immutable predev artifact hashes are recorded in
`paired-seed-summary.json`. A predev pass shows that the frozen readout cleared
the anti-shortcut probes; it does not establish counting accuracy.

## Post-dev observations and audit boundary

The two outputs are closely aligned and share six dev samples with zero valid
pose frames. The raw pose-energy period hits the 128-frame upper bound in
41/84 samples; the embedding period is at most 10 frames in 54/84 samples for
seed 2026 and 48/84 for seed 42; and the maximum harmonic factor 8 is selected
in 52/84 and 47/84 samples, respectively. These facts help explain the severe
undercounting but were observed only after dev scoring.

They are explicitly **post-dev diagnostics**. They cannot be used to tune this
candidate, select a seed, change the harmonic range, choose another expert, or
create an eligible leaderboard result. Any later hypothesis must be frozen and
validated independently before another dev evaluation.

## Provenance and files

- The prediction/scoring protocol runner is bound to Git commit
  `3459e52091b7268d0cbecb13aade404cf68cba2d`.
- Both Encoder checkpoints were produced from Git commit
  `1378d6a977b364f0de93c85663c4e8aaabd6ffbd`.
- `seed-2026.predictions.json` and `seed-42.predictions.json` are the complete
  84-row target-free prediction artifacts.
- `seed-2026.evaluation.json` and `seed-42.evaluation.json` contain the complete
  per-video dev targets, errors, aggregate metrics, bootstrap intervals, and
  isolated-scoring mount audits.
- `paired-seed-summary.json` records the two-seed aggregate, paired difference
  intervals, gate decision, immutable input hashes, and post-dev audit boundary.
- `SHA256SUMS` binds every public artifact in this directory other than itself.

The four copied JSON artifacts are byte-identical to the generated artifacts.
They were scanned before publication and contain no server address, username,
private-key material, or host filesystem path. Checkpoints, raw logs, manifests,
and machine-specific paths are intentionally excluded.

