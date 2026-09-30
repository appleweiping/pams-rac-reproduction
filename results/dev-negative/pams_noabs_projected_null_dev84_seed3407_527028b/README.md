# NoAbsPE projected-null v12 dev84 negative result

**Status:** `partial_reproduction`  
**Classification:** `exploratory-derived`, independently inferred, and
predev-bound  
**Leaderboard eligibility:** ineligible; diagnostic only  
**Sealed test:** test105 was not mounted or accessed and remains sealed

This package records the first dev84 evaluation of
`pams-noabspe-projected-null-multiexpert-v12` with training seed 3407. The
readout is an independent completion, not an author-disclosed PAMS component.
It passed the frozen target-free pre-development gate, but it did not approach
the frozen UCFRep acceptance thresholds after isolated dev scoring.

## Primary result

| Split | Videos | NMAE | OBO | MAE | RMSE | Exact | Joint gate |
|---|---:|---:|---:|---:|---:|---:|:---:|
| UCFRep dev84 | 84 | 0.628805 | 0.380952 | 3.500000 | 5.282496 | 0.202381 | Fail |

The frozen thresholds are `NMAE <= 0.228` and `OBO >= 0.666`. This candidate
fails both. Its paired-row, 10,000-sample bootstrap 95% intervals are:

- NMAE: `[0.482243, 0.815013]`
- OBO: `[0.273810, 0.488095]`
- MAE: `[2.714286, 4.380952]`
- RMSE: `[3.933941, 6.665498]`
- Exact: `[0.119048, 0.285714]`

An independent publication check reconstructed all five point metrics from the
84 per-video rows using rounded predictions. The reconstructed values match the
evaluation artifact exactly. Prediction and evaluation contain the same 84
unique identities, in the same order, and that identity set exactly matches
`data/splits/ucfrep_526_dev_84.txt`.

## Frozen predev authorization

Before this seed was trained, the v12 protocol froze the null seed, null sample
count, significance level, projected-pose period estimator, counter, and all
six gate thresholds. The seed-3407 checkpoint then passed all six target-free
checks:

| Criterion | Threshold | Value | Result |
|---|---:|---:|:---:|
| Absolute frame-index linear-probe R2 | `<= 0.10` | -0.1077334600 | Pass |
| Zero-pose calibrated confidence | `<= 0.10` | 0.0000000000 | Pass |
| Random-pose calibrated confidence | `<= 0.10` | 0.0000000000 | Pass |
| Training top period-bin share | `<= 0.25` | 0.0937500000 | Pass |
| Position-invariance median cosine | `>= 0.95` | 1.0000000000 | Pass |
| Synthetic-period median relative error | `<= 0.10` | 0.0000000000 | Pass |

`predev.authorization.json` is the complete byte-identical gate artifact. Its
SHA-256 is
`3eb154cb98c9f44b9ea06d616b711099d17e16ef3bc94955311efd4c8cbe8d08`.
That gate authorized dev prediction only; it did not establish real-video
counting quality and did not authorize test105 evaluation.

## Synthetic stress before dev

The deterministic projected-stream suite covers counts 2 through 40 and uses
no dataset videos, action labels, count labels, dev input, or test input. All
six planned scenarios pass the per-scenario requirements `NMAE <= 0.05` and
`OBO >= 0.95`:

| Scenario | NMAE | OBO | Planned result |
|---|---:|---:|:---:|
| Clean | 0.028534 | 1.000000 | Pass |
| Acceleration 0.5x to 2.0x | 0.029915 | 0.974359 | Pass |
| Oscillating speed 0.5x to 2.0x | 0.011861 | 1.000000 | Pass |
| 20% pause | 0.019413 | 0.974359 | Pass |
| Gaussian noise, sigma 0.08 | 0.030671 | 1.000000 | Pass |
| 20% time by 10/33-joint occlusion | 0.028534 | 1.000000 | Pass |

The additional, non-planned case that invalidates every joint for one
contiguous 20% interval fails (`NMAE=0.257825`, `OBO=0.102564`). It is retained
in `audit-summary.json` as a known limitation rather than omitted.

## Failure observations

The confidence calibration assigns exactly zero confidence to 65/84 videos;
only 19/84 are positive. Six videos have zero valid pose frames:

- `v_Biking_g16_c02`
- `v_BreastStroke_g01_c01`
- `v_BreastStroke_g12_c02`
- `v_PlayingViolin_g08_c04`
- `v_Rowing_g01_c03`
- `v_Rowing_g14_c07`

These facts are failure analysis, not permission to retune this evaluated
candidate. Any later candidate derived from the dev result is exploratory and
must not replace this immutable negative result.

## Label firewall and provenance

Prediction ran target-free with the encoder checkpoint, frozen configuration,
passed-predev artifact, dev identity/pose input, and no network. Dev targets and
every test105 input were absent. Scoring ran separately with the frozen
predictions, their receipt, and dev targets; checkpoint, pose data, network, and
every test105 input were absent. The receipts bind the prediction and evaluation
artifacts, and `run-status.json` independently records the mount firewall and
`test105_accessed=false`.

- Dev runner source commit:
  `527028b6ce157a07db70af8085f5729ce6799bcd`
- Encoder checkpoint source commit:
  `0f5268c8496bb708fe07388e063b04ae7ddef5b3`
- Encoder checkpoint SHA-256:
  `a5f08e8585cc0666eea265d9c38617d8845f3af0a4cf4fce92b1ef956e62b76d`
- Frozen configuration fingerprint:
  `e64029a8c1ae1bf258a71eabe03fd35cd9cd3f18bdf0c7e4ae17657cf9c6a13e`

`predictions.json` and `evaluation.json` are the complete artifacts, including
all per-video records. The copied JSON files are byte-identical to the
generated artifacts. All package files were scanned for server addresses,
usernames, private-key markers, SSH paths, and host filesystem paths; none were
found. `SHA256SUMS` binds every public file other than itself.
