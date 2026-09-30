# Reproducibility Contract

## Frozen identifiers

- Standard protocol: UCFRep 421 training videos and 105 sealed test videos.
- Pose target protocol: UCFRep-pose 89 training videos and 21 test videos;
  sealed scoring is disabled until its official ID/count digest is frozen.
- Seeds: 42, 2026 and 3407.
- Configuration: `configs/pams.yaml`; its canonical SHA-256 is recorded in
  every run manifest.
- Pose cache identity: SHA-256 over only the canonical `data` and `pose`
  configuration blocks. It excludes seed and all model/training settings.

The full configuration fingerprint and pose fingerprint serve different
purposes and are never substituted for each other. Run manifests retain the
full configuration fingerprint. Pose cache schema v2 stores the pose
fingerprint plus the source video hash and model ID; schema-v1 caches using
the old full-config semantics must be regenerated.

The standard training partition is split into 337 train and 84 development
videos with seed 2026, stratified by action and count bin. UCFRep-pose is
split 71/18 in the same manner. After configuration freeze, final models are
retrained on all 421 or 89 permitted training videos.

Encoder contrastive training uses a physical batch of exactly 32 videos;
gradient accumulation is not treated as an equivalent source of batch-local
cross-video negatives. Every valid frame from the other 31 physical-batch
videos enters the denominator; videos are not collapsed to one live
prototype. The seeded, epoch-shuffled loader drops that epoch's incomplete
final batch (five videos for the 421-video pool). SSHead training may use
gradient accumulation because its loss has no cross-video negative pool.

The deterministic KMeans refresh also freezes one prototype for every
training video. Explicit cross-cluster hard negatives are distinct top-k
entries from that full bank after excluding every video in the current
physical batch. Checkpoints persist the ordered bank and cluster labels.
Formal CLI training rejects any epoch whose bank cannot supply each anchor's
requested positive-count-matched negatives; direct smoke tests must
explicitly opt into allowing and logging a shortfall.
Encoder checkpoint schema v5 persists this bank, the exact training
pose-cache-set digest, source Git SHA, and optional formal-container image
identity. Earlier checkpoints are intentionally rejected rather than silently
rebuilding provenance from a later state.

Both pose-energy and embedding-velocity period estimators return spectral
confidence and preserve the dominant FFT bin's fractional frame period
(rather than rounding before inference). Only exact zero means “no period
evidence”: encoder training then
keeps adjacent-frame positives but omits cycle correspondences, while SSHead
omits cycle/spectral terms for that sample and retains anti-collapse variance
and smoothness. Epoch JSONL records the confidence mean, valid fraction, and
cross-cluster requested/actual/shortfall totals. Progress schema v2 stores a
path-independent checkpoint role rather than a machine-local absolute path,
so an unchanged checkpoint/log pair remains verifiable after publication.

SSHead additionally records the minimum, 10th percentile, median, and mean
valid-frame stream standard deviation across training videos, plus the
fractions at or below `1e-6` and `1e-3`. Gradient RMS, gradient-to-parameter
norm ratio, and zero-gradient optimizer-step counts are label-free runtime
diagnostics. Every batch must have finite streams, losses, and gradients.
Before an optimizer step, a positive-loss exact-collapse state or a
near-collapse gradient spike aborts the epoch; no checkpoint or JSONL row is
written for the failed epoch. These fixed safety thresholds never inspect
development or test labels and do not alter the SSHead objective.

## Label firewall

PAMS data loaders expose pose tensors and stable video identifiers, never
counts, for train/development use. Although the audit manifest contains
evaluator fields, the training projection discards count/action and every
test row before constructing a dataset. Tests assert canonical split-list
hashes, source-group membership, cross-split content-hash disjointness, and
that a training dataset cannot be constructed from a test-labelled or
unhashed manifest.

Checkpoint and training-run provenance use a separate label-free training
fingerprint over only the selected train/dev video IDs and content hashes.
It excludes actions, counts, local mount paths, and every sealed-test row.
They also bind an ordered digest of every pose-cache byte stream actually
loaded. The evaluator records the full manifest fingerprint and separate
training/evaluation pose-cache snapshots.
Sealed test metrics additionally require a clean committed worktree and one
shared registry derived from `PAMS_RUN_ROOT`. An atomic receipt consumes one
attempt per frozen method/seed before target fields are accessed. The key uses
the frozen, split/order/path-invariant official annotation identity, so a
337/84 development assignment or row reorder cannot create a retry. Changing
an output directory, checkpoint, or prediction file also cannot create one.
Only checkpoint-validated PAMS methods are presently admitted; baseline IDs
remain unreservable until runnable adapters and method-specific provenance
gates pass.

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

Every formal command first writes an immutable `*.started.json` receipt. A
successful command then writes a separate hash-linked `*.completed.json`
receipt; it never mutates the start record. A crash or rejected gate therefore
cannot masquerade as completion. Each published run contains:

- immutable started and completed receipts;
- exact configuration and split hashes;
- environment lock and Git SHA;
- training/evaluation logs;
- one prediction row for every evaluated video;
- aggregate and per-seed metrics;
- checkpoint hash and download provenance;
- hardware fingerprint and elapsed time.

Completed-receipt schema v3 stores portable POSIX locators plus byte counts
and SHA-256 values, never host-absolute artifact paths. `pams data
validate-run` rejects duplicate JSON keys and confines implicit locators to
the receipt package root (the parent of its `manifests/` directory). Inputs
kept outside that package and legacy schema-v2 host paths require an explicit
`--artifact-remap ROLE=ABSOLUTE_PATH`; remapping never bypasses size or digest
verification.

CLI resume never mutates the checkpoint or progress log named by an earlier
completion receipt. `--resume-checkpoint` and `--resume-progress` are copied
byte-for-byte into a new output directory, their original hashes are recorded
as resume inputs, and only the new copies are advanced.
