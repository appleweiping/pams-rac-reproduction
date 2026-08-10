# PAMS-SSHead longest-track v8, seed 2026

This is a strict development-only negative result on the frozen UCFRep
337/84 split. It is an independent inferred completion, not author code and
not a paper-table result. The sealed 105-video test split was not mounted or
evaluated.

## Result

| Metric | v8 seed 2026 | 95% paired-bootstrap CI |
|---|---:|---:|
| NMAE | 0.643123 | [0.563462, 0.720873] |
| Raw MAE | 4.226190 | [3.357143, 5.238095] |
| RMSE | 6.073949 | [4.548646, 7.840694] |
| OBO | 0.297619 | [0.202381, 0.392857] |
| Exact | 0.154762 | [0.083333, 0.238095] |

The matched v7 seed-2026 result was NMAE `0.735728` and OBO `0.190476`.
For `v8 - v7`, the 10,000-sample paired differences are:

- NMAE `-0.092605`, 95% CI `[-0.178001, -0.008187]`;
- OBO `+0.107143`, 95% CI `[+0.035714, +0.178571]`.

Absolute error improved on 40 videos, tied on 25, and worsened on 19.
The protocol correction therefore produced a statistically resolved
development improvement in NMAE and OBO, but it still missed the frozen
extension requirements `NMAE <= 0.60` and `OBO >= 0.30`. The preregistered
decision is `do-not-extend`, so seeds 42 and 3407 were not run.

## What changed

The only intended algorithmic change from v7 was the pose preprocessing
revision required by the reproduction protocol: use the longest contiguous
MediaPipe detection run instead of the complete first-to-last detected span.
The exact config is
`configs/experiments/pams_longest_contiguous_track_v8.yaml`.

The label-free 421-video pose comparison found:

| Pose cache | Mean cached valid-frame rate | All-invalid videos |
|---|---:|---:|
| v2 detected span | 0.773605 | 13 |
| v3 longest contiguous track | 0.947743 | 22 |

Nine v3 videos had a one-frame selected run and were deliberately converted
to all-invalid rather than expanded into a fake 256-frame valid sequence.
The full identities and cache-set hashes are in
`pose-cache-comparison.json`.

## Isolation and provenance

- Encoder: 150 epochs on train337 pose only.
- Inferred SSHead: 30 epochs on train337 pose only.
- Dev prediction: target-free container; 84 predictions frozen first.
- Dev score: separate CPU-only container opened dev targets afterward.
- Test videos, test pose, and test targets: never mounted.
- Git revision:
  `07192debb7f5748c53a1c6d4f0c0d3f16228e229`.
- Container image ID:
  `sha256:a6d10131c321c323c00bb85408cb8333503ecbc5d780d5059a73152d54df6005`.
- Training-run artifact manifest SHA-256:
  `ddd11aaa03a9b73148c21f39b4367fcbcb11ebc82481fefa1bdd63273fb29877`.

`evaluation.json` contains all 84 scored rows and 10,000-sample confidence
intervals. `predictions.json` is the target-free frozen prediction artifact.
The manifests and JSONL logs bind the checkpoints, source, config, image, and
pose snapshots. `summary.json` is the compact audit index.

The first pose-comparison wrapper attempt computed successfully but included
the NVIDIA entrypoint banner on stdout, making that stream invalid JSON. It
was retained server-side as failed wrapper evidence; the published audit is
the clean retry using an explicit Python entrypoint. The cached image build
also had two post-build receipt-command quoting errors; the immutable image
JSON and labels were then independently validated. This is disclosed in
`build-receipt-recovery.txt` and does not change the experiment outputs.

Weights are published under prerelease `v0.4-v8-partial`; their exact hashes
and sizes are recorded in `summary.json`.
