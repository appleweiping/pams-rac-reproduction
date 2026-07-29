# PoseRAC-v1 official-checkpoint audit

This note covers only the 2023 PoseRAC-v1 release. It does not describe the
later ICONIP'24 PoseRAC architecture.

## Frozen upstream identity

- Repository: `https://github.com/MiracleDance/PoseRAC`
- Commit: `469590b611bde3595eaf163b517263da634e2096`
- Source archive: 62,833,091 bytes,
  SHA-256 `0150eacf56ec033a77c24d61bebc0894134547677deda41ba5681eea887c21d0`
- Extracted source tree: 22 files, 66,442,868 bytes,
  canonical SHA-256
  `7b3feec127f7a81a44d7ed94cb6d54e515421213cc40392aa89fa44311912e90`
- Released `best_weights_PoseRAC.pth`: 10,772,679 bytes,
  SHA-256 `21afc8334333e5f5751f0e5d765376a778415285c0ac3078d8358ccb83b2b34a`
- License: MIT; `LICENSE` SHA-256
  `f9c3dc103d80706f49925e6fa0b2ba7a15a7710104ba5fff82a150fb1711aa9e`

The checkpoint is a plain ordered tensor state dictionary. The independently
reconstructed released model has 74 state-dictionary keys and 2,686,682
parameters. Strict restoration loads all 74 keys, with no missing or
unexpected keys. The output layer is `[8, 99]`, matching the eight entries in
the released `all_action.csv`:

1. `front_raise`
2. `pull_up`
3. `squat`
4. `bench_pressing`
5. `jump_jack`
6. `situp`
7. `push_up`
8. `pommelhorse`

This corrects the earlier local baseline note that described the release as
five-class.

## Released model and preprocessing

The released architecture is:

- input dimension 99 (`33 joints × xyz`);
- six `torch.nn.TransformerEncoderLayer` blocks;
- nine attention heads;
- default 2,048-dimensional feed-forward layers;
- an eight-output linear classifier;
- sigmoid probabilities followed by a two-threshold state machine;
- enter threshold `0.78`, exit threshold `0.4`, and recursive smoothing
  momentum `0.4`.

`pre_test.py` runs MediaPipe Pose on every decoded frame. A detected pose is
normalized independently for x, y, and z over the 33 joints. A missing pose is
stored as 99 zeros. It does not temporally resample the video.

## Ground-truth leakage in the released evaluator

The released `eval.py` reads `gt_count`, produces a candidate count for every
one of the eight output channels, computes the normalized absolute error
against `gt_count`, and retains the channel with the smallest error. The
visualization script repeats the same selection. This is a ground-truth-count
channel oracle and is prohibited in a fair prediction table.

The independent runner never imports or accepts a count, action, label, score,
or target input. It selects the first channel with maximum temporal dynamic
range after released smoothing. That rule is explicitly marked `inferred`.

## Compatibility with the PAMS pose cache

The current schema-v2 cache has the required `[frames, 33, 3]` MediaPipe
coordinates, a validity mask, video SHA-256 binding, pose-extractor
fingerprint, and exact cache-byte receipt. Its coordinates can be converted to
the released per-axis convention.

It is not source-preprocessing-equivalent, however:

- PAMS first normalizes each frame jointly over all 99 values;
- PAMS then interpolates every video to exactly 256 frames;
- released PoseRAC-v1 normalizes x/y/z independently and retains all decoded
  original frames;
- temporal interpolation cannot be reversed from the 256-frame cache.

Therefore the prepared runner is classified as:

`PoseRAC-v1 official checkpoint / PAMS 256-frame pose-cache / inferred
oracle-free dynamic-range channel`

It is suitable only as an independently defined, label-free development
diagnostic on the fixed 84-video UCFRep-526 development split. It is not
eligible for the original PoseRAC-v1 UCFRep-pose-110 cell or the PoseRAC value
quoted in the PAMS paper.

## Runner guarantees

`src/pams/baselines/poserac_v1_official_runner.py` enforces:

- exact source archive, complete extracted tree, checkpoint, runner source,
  clean Git revision, pose-cache, sidecar, and commitment hashes;
- exact 74/74 checkpoint coverage;
- schema-v2 MediaPipe `0.10.14` cache provenance and exactly 256 frames;
- count/action-free input sidecar loading before checkpoint inference;
- deterministic first-index tie breaking for inferred channel selection;
- immutable prediction and receipt files with separate cache-set and
  configuration hashes;
- `labels_loaded=false`, `scoring_performed=false`, and
  `ground_truth_count_channel_oracle=false` in every prediction artifact.

No UCFRep-526 sealed-test video or label is required or accepted by this
runner. Scoring remains a separate process.

## Sealed development scorer

`src/pams/baselines/poserac_v1_official_score.py` accepts only the exact
84-video `ucfrep_526/dev` input sidecar and a target file whose basename is
exactly `dev.targets.json`. It has no test argument or test loader.

Before its first read of `dev.targets.json`, it validates:

- prediction and prediction-receipt byte identities;
- the clean runner Git SHA and exact runner-code SHA-256;
- official archive, complete source tree, checkpoint, source-file, config,
  license, and compatibility-image identities;
- all 84 sidecar IDs and their canonical development-set digest;
- all 84 pose-cache byte receipts and their reconstructed cache-set digest;
- exact 74/74 checkpoint restoration;
- every per-video channel count, dynamic range, first-maximum inferred channel
  choice, probability bound, frame count, and selected count;
- the explicit absence of the released ground-truth-count channel oracle.

The scorer then reports rounded NMAE, raw and rounded MAE/RMSE, OBO, exact
accuracy, 10,000-sample paired percentile-bootstrap intervals, and all 84
prediction/target rows. Its evaluation and receipt are exclusive immutable
files and both record that the sealed test remains untouched.
