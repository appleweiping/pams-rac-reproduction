# PAMS-SSHead temporal-conv v7: post-hoc three-seed dev84 audit

This is a completed real-training experiment and a **negative,
development-only partial reproduction**. Seed 3407 was run only after the two
prespecified seeds, 42 and 2026, had already failed both acceptance
thresholds. The three-seed result is therefore permanently classified as a
post-hoc variance follow-up. It cannot enter the paper's Table 2, satisfy the
preregistered three-seed claim, or verify PAMS.

## Result

All metrics use the fixed 84-video UCFRep development split and the repository
protocol's half-up rounded count.

| Seed | NMAE ↓ | OBO ↑ | MAE ↓ | RMSE ↓ | Exact ↑ | Zero counts | Period 128 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.694489 | 0.238095 | 4.261905 | 5.694609 | 0.095238 | 23 | 25 |
| 2026 | 0.735728 | 0.190476 | 4.428571 | 5.794086 | 0.107143 | 31 | 30 |
| 3407 | 0.740368 | 0.202381 | 4.785714 | 6.611678 | 0.083333 | 29 | 28 |
| **Mean** | **0.723528** | **0.210317** | **4.492063** | **6.033458** | **0.095238** | — | — |
| **Sample SD** | **0.025256** | **0.024782** | **0.267615** | **0.503218** | **0.011905** | — | — |

The 10,000-sample paired percentile 95% intervals for the
mean-across-seeds estimand are:

- NMAE: `[0.660771, 0.784881]`
- OBO: `[0.138889, 0.289683]`
- MAE: `[3.726190, 5.321429]`
- RMSE: `[4.960734, 7.044311]`
- Exact: `[0.055556, 0.142857]`

The frozen verification gate requires three-seed mean NMAE at most `0.228`,
mean OBO at least `0.666`, and at least two individually passing seeds. The
observed result fails both mean thresholds and has `0/3` individually passing
seeds.

Seed 3407's Head was trainable—its final epoch recorded no zero-gradient
steps—but the output stream standard deviation remained only `0.015449`.
Together with 29 zero counts and 28 maximum-period predictions, this confirms
that the inferred temporal convolution did not repair the low-frequency /
near-collapse failure mode.

## Protocol and claim boundary

The method is an inferred repair, not an author-disclosed component. It adds a
mask-aware, bias-free, zero-initialized residual depthwise `Conv1d` with 512
channels and kernel size 5 before the disclosed `512 -> 128 -> 1` GELU MLP.
The encoder, label-free period teacher, loss weights, confidence weighting,
and multi-expert consensus remain frozen from v6.

Seed 3407 used:

- train337 only for 150 encoder epochs and 30 frozen-encoder Head epochs;
- source Git SHA
  `6ee25a2e1291c35c563140999e16c5aa65bc8ced`;
- image
  `sha256:620173fe7a9084d3b1492c3e775ea5f2bc6ac5a6d851d8560727a4969b3aab66`;
- configuration file SHA-256
  `a6926c095a039379d1d1b22931568c6cd05f84a4a0913bccdfc6fc0f2aa9f053`;
- canonical configuration fingerprint
  `c329367692191849cabca3ccca9fec5e80788ef9f0d6328fb05e4fc8aef2869b`;
- one RTX A6000 with `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

The new run used four direct Docker containers: encoder, Head, target-free
development prediction, and CPU-only development scoring. All four exited
zero, used `network=none`, a read-only root filesystem, dropped all
capabilities, and passed their exact bind-mount checks. Encoder and Head saw
only the 337-file pose view; prediction saw the exact 421-file train+dev view.
No learning or prediction container mounted `dev.targets.json`, test poses,
test videos, or test labels. Only the final score container received frozen
predictions, their receipt, and dev84 targets. Test105 remains untouched.

The host launcher rechecked the already frozen dev-target SHA-256 during
preflight but did not parse targets or expose their path to a learning or
prediction process. The target sidecar was deserialized only by the final
scorer. A PyTorch warning disabled the optional FFT kernel cache because its
cache subdirectory could not be created; training and all receipts completed,
so this is retained as a non-blocking runtime caveat.

Seeds 42 and 2026 predate the separate predictor/scorer schema. Their legacy
evaluation files contain all 84 predictions and per-video targets but do not
carry source or target hashes internally. The aggregate verifies their config
fingerprints and exact per-video membership while explicitly marking the
legacy source/target bindings as caller-audited, not self-verifying. Seed 3407
uses the strict schema and directly verifies all three source fields and the
dev-target hash.

## Artifacts

- [Three-seed aggregate](three-seed-aggregate.json): means, sample SDs,
  paired bootstrap intervals, mixed-schema audit boundary, and frozen gate.
- [Seed 3407 evaluation](seed-3407.evaluation.json): all 84 predictions,
  per-video errors, intervals, and strict provenance.
- [Frozen seed 3407 predictions](seed-3407.predictions.json) and
  [prediction receipt](seed-3407.prediction.receipt.json).
- [Evaluation receipt](seed-3407.evaluation.receipt.json).
- [Encoder log](seed-3407.encoder.jsonl) and
  [Head log](seed-3407.sshead.jsonl).
- [Exact seed 3407 config](seed-3407.config.yaml).
- [Attempt reservation](attempt.reservation.json),
  [terminal status](status.json), and
  [audit manifest](audit-manifest.json).
- Exact execution and aggregation code:
  [`scripts/server/run_pams_v7_seed3407_posthoc_strict.sh`](../../../scripts/server/run_pams_v7_seed3407_posthoc_strict.sh)
  and
  [`scripts/aggregate_pams_v7_seed_evaluations.py`](../../../scripts/aggregate_pams_v7_seed_evaluations.py).

Rebuild the aggregate from the three frozen evaluation files:

```bash
python scripts/aggregate_pams_v7_seed_evaluations.py \
  --evaluation 42=results/dev-negative/pams_sshead_temporal_conv_v7_seed42_strict.json \
  --evaluation 2026=results/dev-negative/pams_sshead_temporal_conv_v7_seed2026_strict.json \
  --evaluation 3407=results/dev-negative/pams_sshead_temporal_conv_v7_three_seed_posthoc_6ee25a2/seed-3407.evaluation.json \
  --config-fingerprint 42=0b44f415a8083fb35205ad6a1a085a92df68b6b4bcc2974cae101a9a381196c7 \
  --config-fingerprint 2026=317442965eb856a59a5b70f3ec63756fdceec16c55c9c31820b61a7c9d25ccca \
  --config-fingerprint 3407=c329367692191849cabca3ccca9fec5e80788ef9f0d6328fb05e4fc8aef2869b \
  --source-git-sha 6ee25a2e1291c35c563140999e16c5aa65bc8ced \
  --dev-targets-sha256 1ab3bc010ccb5d869af7662586594c973e53841838643e1e9b8cc1787b0efab6 \
  --output /tmp/pams-v7-three-seed-aggregate.json
```

Original videos, pose caches, development target sidecars, test identities,
test labels, and model weights are not committed to the Git tree. The exact
checkpoint and completion-receipt hashes are retained in the audit manifest.
