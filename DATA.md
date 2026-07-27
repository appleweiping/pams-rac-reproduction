# Data Preparation

This repository does not redistribute UCF101/UCFRep videos.

The official annotation archive is obtained from the
[original UCFRep repository](https://github.com/Xiaodomgdomg/Deep-Temporal-Repetition-Counting/tree/f7011d3ef29061b9c63cb7c12d5feeb2a93f7d35/data/ori_data/ucf526)
and is accepted only when its SHA-256 equals
`08b2b8a88c2728e9aa6de6c62dc6b02f1941dfa2c3adb28dab5852c12695ae3c`.
The repository loader defines the video count as the number of temporal
boundaries minus one.
With video hashes unset, the resulting canonical 421/105 annotation manifest
has fingerprint
`e0fde9919b5fa3b918d3e97636066e9008a7a5f0572d83fe684f0bb4f4132e30`.
Adding actual video SHA-256 values intentionally produces a new fingerprint.

## Expected manifest

Prepare one UTF-8 CSV or JSON manifest with, at minimum:

```text
video_id,video_path,split,action,count,video_sha256
```

The UCF101 source group remains encoded in the canonical `video_id` (`g01`
through `g25`). `split` is one of `train`, `dev`, or `test`. Counts for sealed
test records must be stored in the evaluator-only manifest rather than the
training manifest.

The preparation command verifies:

- exactly 421/105 standard UCFRep records;
- no video ID or content hash overlap between train and test;
- readable videos and a deterministic failed-decode ledger;
- stable source group assignment;
- per-file SHA-256 before pose extraction.

## Pose cache

The preregistered cache uses MediaPipe 33×3 landmarks. It selects the longest
continuous dominant-person track, stores zeros for missing frames plus a
validity mask, applies per-frame min-max normalization, and uniformly
resamples the complete video to 256 frames.

Each `.npz` cache entry carries:

- video ID and source video SHA-256;
- pose-extractor name/version/configuration;
- preprocessing configuration SHA-256;
- `xyz[256,33,3]` float32;
- `valid_mask[256]` bool;
- source FPS.

The extractor failure policy is dataset-wide. A failed video is never
silently excluded from only one method.

## UCFRep-pose annotations

Only the 89 training videos may receive independently produced salient-pose
or cycle-boundary annotations. The 21 test videos remain annotation-free.
Annotation frame indices, reviewer agreement and adjudications are versioned;
the underlying video frames are not committed.
