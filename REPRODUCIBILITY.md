# Reproducibility Contract

## Frozen identifiers

- Standard protocol: UCFRep 421 training videos and 105 sealed test videos.
- Pose protocol: UCFRep-pose 89 training videos and 21 sealed test videos.
- Seeds: 42, 2026 and 3407.
- Configuration: `configs/pams.yaml`; its canonical SHA-256 is recorded in
  every run manifest.

The standard training partition is split into 337 train and 84 development
videos with seed 2026, stratified by action and count bin. UCFRep-pose is
split 71/18 in the same manner. After configuration freeze, final models are
retrained on all 421 or 89 permitted training videos.

## Label firewall

PAMS data loaders expose pose tensors and stable video identifiers, never
counts, for train/development use. Ground-truth count files for the 105 and
21 sealed evaluations are accepted only by the evaluator. Tests assert that
train and test IDs are disjoint and that a training dataset cannot be
constructed from a test-labelled manifest.

Supervised baselines may consume only the annotations allowed by their
declared protocol. No main-table method may:

- select a prediction using test count;
- select an action head using test action class;
- tune smoothing, peak thresholds or checkpoints on a sealed split;
- silently drop a video because one method failed to decode or extract pose.

Oracle diagnostics, when scientifically useful, are stored outside the main
table and labelled in both filenames and report headings.

## Metric definition

The paper labels its primary error as MAE, while the reported implementations
and value scale correspond to normalized MAE:

```text
NMAE = mean(abs(nearest_integer(prediction) - ground_truth) / ground_truth)
OBO  = mean(abs(nearest_integer(prediction) - ground_truth) <= 1)
```

Nearest-integer rounding is deterministic and shared by every adapter.
Raw MAE and RMSE are additionally reported. Confidence intervals use 10,000
paired video bootstrap samples with seed 2026.

## Claim states

- `implementation`: code/tests exist but no designated benchmark run exists.
- `partial`: benchmark runs exist but a verification condition failed.
- `verified`: three-seed mean NMAE ≤ 0.228 and OBO ≥ 0.666, at least two
  seeds independently satisfy both thresholds, and full PAMS is the best
  preregistered self-ablation with the expected multi-scale/multi-expert
  directions.

Only `verified` may be attached to a v1.0 release. Negative results and
deviations remain publishable as `partial`.

## Artifact minimum

Each published run contains:

- immutable run manifest;
- exact configuration and split hashes;
- environment lock and Git SHA;
- training/evaluation logs;
- one prediction row for every evaluated video;
- aggregate and per-seed metrics;
- checkpoint hash and download provenance;
- hardware fingerprint and elapsed time.
