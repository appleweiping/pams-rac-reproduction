# JTSPS count-only inferred baseline: three-seed UCFRep dev84 result

This is a completed, real-training baseline experiment, but it is a **negative,
development-only result**. The public JTSPS material does not uniquely specify
its cycle-density supervision or decoding, so this implementation remains an
independently inferred count-only closure. It is not source parity, must not
occupy the paper's JTSPS table cell, and does not verify PAMS.

## Result

All metrics below use the frozen 84-video UCFRep development split. Counts are
rounded half-up before NMAE, MAE, RMSE, OBO, and exact-match evaluation.

| Seed | NMAE ↓ | OBO ↑ | MAE ↓ | RMSE ↓ | Exact ↑ |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.478276 | 0.404762 | 2.976190 | 4.793349 | 0.166667 |
| 2026 | 0.502275 | 0.392857 | 2.976190 | 4.728334 | 0.142857 |
| 3407 | 0.482349 | 0.428571 | 2.928571 | 4.733367 | 0.178571 |
| **Mean** | **0.487634** | **0.408730** | **2.960317** | **4.751683** | **0.162698** |
| **Sample SD** | **0.012843** | **0.018185** | **0.027493** | **0.036171** | **0.018185** |

The 10,000-sample paired percentile 95% intervals for the mean-across-seeds
estimand are:

- NMAE: `[0.409718, 0.566997]`
- OBO: `[0.309524, 0.511905]`
- MAE: `[2.245933, 3.817560]`
- RMSE: `[3.245023, 6.388920]`
- Exact: `[0.091270, 0.238095]`

Raw-count mean ± sample SD is NMAE `0.498478 ± 0.018260`, MAE
`2.981165 ± 0.054733`, and RMSE `4.727113 ± 0.116858`.

The model is also visibly under-dispersed: rounded count 4 is the mode for
`60/84`, `46/84`, and `52/84` videos at seeds 42, 2026, and 3407,
respectively. Each seed has the same five zero predictions caused by
all-invalid pose caches. Those samples follow the dataset-wide masking policy
and were not dropped.

For reference only, the preregistered PAMS verification thresholds are NMAE
at most `0.228` and OBO at least `0.666`. This baseline misses both thresholds
for all three seeds. Those thresholds certify PAMS, not baselines; the
comparison is included only to prevent this negative result from being
misread as reproduction evidence.

## Frozen inferred protocol

- 337 train videos with video-level count supervision; 84 development videos
  for one-time prediction and scoring; test105 remains sealed.
- MediaPipe 33×3 pose cache, uniformly sampled to 64 frames.
- Joint embedding 16, hidden width 32, non-negative density summed over valid
  frames.
- Loss: log-count SmoothL1 + `0.25` relative L1 + `0.1` impulse log-count
  auxiliary + `0.001` density total variation.
- AdamW, learning rate `1e-4`, weight decay `1e-4`, batch size 8, 30 epochs,
  gradient clipping at 5.
- Seeds `42`, `2026`, and `3407`; no run was selected or rerun after observing
  its development result.

Seeds 42 and 3407 used the exact source at
`9df2646df90cb65e60b5da06616cbc47427ba3e9`, the same fixed container image,
and one GPU each on a 2× RTX A6000 server. Their train/predict containers had
no network and could mount train inputs, train count targets, unlabeled dev
inputs, and a copied 421-file pose view—but not dev targets or any test
identity/target file. Separate CPU score containers received only frozen
predictions and dev targets.

The older seed-2026 run has the same source, model, inputs, and image, but
predates explicit recording of `CUBLAS_WORKSPACE_CONFIG=:4096:8`. That
provenance difference is retained rather than hidden or repaired after seeing
the result.

An initial v1 launcher stopped in host-side pose-copy preflight because Python
3.8 rejected a `list[str]` annotation. It created no ledger, container,
prediction, or score. The failed directory remains on the server; v2 used a
new path and did not overwrite it.

## Artifacts

- [Three-seed aggregate](three-seed-aggregate.json) — means, sample SDs,
  paired bootstrap intervals, evaluation bindings, and claim boundary.
- [Audit manifest](audit-manifest.json) — source, input, runtime, container,
  label-firewall, failure, and SHA-256 evidence.
- Seed 42:
  [predictions](seed-42.predictions.json),
  [evaluation](seed-42.evaluation.json), and
  [run manifest](seed-42.run.json).
- Seed 2026:
  [predictions](seed-2026.predictions.json),
  [evaluation](seed-2026.evaluation.json), and
  [run manifest](seed-2026.run.json).
- Seed 3407:
  [predictions](seed-3407.predictions.json),
  [evaluation](seed-3407.evaluation.json), and
  [run manifest](seed-3407.run.json).
- [Server-produced two-seed summary](new-seeds-server-summary.json).
- Exact launch and aggregation code:
  [`scripts/server/run_jtsps_count_only_seed42_3407.sh`](../../../scripts/server/run_jtsps_count_only_seed42_3407.sh)
  and
  [`scripts/aggregate_jtsps_seed_evaluations.py`](../../../scripts/aggregate_jtsps_seed_evaluations.py).

Rebuild the aggregate from the frozen evaluations:

```bash
python scripts/aggregate_jtsps_seed_evaluations.py \
  --evaluation 42=results/dev-negative/jtsps_count_only_inferred_three_seed_9df2646/seed-42.evaluation.json \
  --evaluation 2026=results/dev-negative/jtsps_count_only_inferred_three_seed_9df2646/seed-2026.evaluation.json \
  --evaluation 3407=results/dev-negative/jtsps_count_only_inferred_three_seed_9df2646/seed-3407.evaluation.json \
  --output /tmp/jtsps-three-seed-aggregate.json
```

No original videos, pose-cache files, development target sidecar, test
identities, test labels, or model checkpoints are included here.
