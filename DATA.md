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
Training/checkpoint identity uses a second, label-free fingerprint over only
the selected train/dev video IDs and content hashes; test rows, actions,
counts, and mount paths cannot influence training provenance.
For sealed scoring, a third split/order/path-invariant digest,
`d371f9f4609730d6484efc337413b444ed73752ad5e994366d02fb79a9960452`,
binds all 526 official `(video_id, action, count, train/test family)` tuples.
This prevents changed test counts, row reordering, or 421/105 versus
337/84/105 representations from creating different legal test identities.

## Supported UCF101 layouts

Point `VIDEO_ROOT` either at the directory containing the extracted
`UCF-101` directory, at `UCF-101` itself, or at a directory containing a
deliberately flattened `flat` directory. The resolver accepts exactly these
three layouts:

```text
VIDEO_ROOT/UCF-101/<Action>/<video>.avi
VIDEO_ROOT/<Action>/<video>.avi
VIDEO_ROOT/flat/<video>.avi
```

For each annotation, exactly one candidate must exist. If the same video is
present in two supported layouts, preparation fails as ambiguous rather than
silently choosing one.

The default command is experiment-strict: all 526 regular video files must be
present and each receives a SHA-256. Hashing can be deliberately disabled
without weakening the file-existence requirement:

```bash
pams data prepare-ucfrep /datasets \
  --output data/manifests/ucfrep_526.json

pams data prepare-ucfrep /datasets/UCF-101 \
  --output data/manifests/ucfrep_526-no-video-hashes.json \
  --no-hash-videos
```

For annotation inspection only, missing videos may be explicitly permitted.
Such a manifest is not training-ready:

```bash
pams data prepare-ucfrep /not-yet-downloaded/UCF-101 \
  --output data/manifests/ucfrep_526-annotations-only.json \
  --annotation-only --no-hash-videos
```

## Expected manifest

Prepare one UTF-8 CSV or JSON manifest with, at minimum:

```text
video_id,video_path,split,action,count,video_sha256
```

The UCF101 source group remains encoded in the canonical `video_id` (`g01`
through `g25`). `split` is one of `train`, `dev`, or `test`. The audit
manifest retains sealed counts for the evaluator, but the training loader
constructs a label-free projection containing only permitted train/dev IDs,
paths, and content hashes; it never returns count/action or any test row.

The preparation command verifies:

- exactly 421/105 standard UCFRep records;
- exact preregistered 421/105 or 337/84/105 ID-list hashes;
- the frozen official ID/action/count/train-test-family semantic digest;
- no video ID or content hash overlap between train and test;
- unique resolution and existence of all 526 videos in strict mode;
- stable source group assignment;
- per-file SHA-256 before pose extraction by default.

Annotation-only or `--no-hash-videos` manifests are inspection artifacts and
are rejected by training and cached evaluation.

Video decoding is validated during pose extraction. Decode failures are
recorded in one deterministic dataset-wide ledger.

## Pose cache

The preregistered cache uses `mediapipe-pose-0.10.14` with model complexity 1,
landmark smoothing enabled, detection/tracking confidence 0.5, and MediaPipe
segmentation disabled. The pinned legacy MediaPipe API is single-person. The
implementation therefore chooses the dominant detection when a detector
returns candidates, crops only before the first and after the last detected
frame, and retains every internal miss as an exact-zero masked frame; it does
not claim multi-person identity tracking. Per-frame min-max normalization and
uniform resampling to 256 frames follow. These inferred settings are explicit
in `configs/pams.yaml`.

Each `.npz` cache entry carries:

- video ID and source video SHA-256;
- pose-extractor name/version/configuration;
- schema-v2 `pose_fingerprint`, computed only from the `data` and `pose`
  configuration blocks;
- `xyz[256,33,3]` float32;
- `valid_mask[256]` bool;
- source FPS.

The pose fingerprint deliberately excludes experiment seed, encoder,
training, loss, SSHead, and consensus settings. Seeds 42, 2026, and 3407
therefore consume identical pose caches, while any extractor/preprocessing
change creates a different fingerprint. Schema-v1 caches that stored the full
experiment `config_sha256` are rejected with a regeneration instruction.
Formal training and evaluation additionally write an ordered snapshot of the
SHA-256 and byte size of every `.npz` file actually read. Its aggregate
fingerprint is bound into checkpoint provenance and terminal run receipts.

Interrupted extraction can be resumed safely:

```bash
pams pose extract data/manifests/ucfrep_526.json data/pose-cache \
  --config configs/pams.yaml --skip-existing
```

`--skip-existing` hashes the source video and opens the cache before skipping.
The cached video ID, video SHA-256, pose fingerprint, model ID, arrays, and
metadata must all validate. A stale or malformed cache is reported as a
failure rather than overwritten; `--overwrite` is a separate, mutually
exclusive operation.

The extractor failure policy is dataset-wide. A failed video is never
silently excluded from only one method.

## UCFRep-pose annotations

Only the 89 training videos may receive independently produced salient-pose
or cycle-boundary annotations. The 21 test videos remain annotation-free.
Annotation frame indices, reviewer agreement and adjudications are versioned;
the underlying video frames are not committed.
Sealed UCFRep-pose scoring is currently disabled: the official 89/21
ID/action/count digest has not yet been frozen in this repository.
