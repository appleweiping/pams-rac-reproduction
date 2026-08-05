# UCFRep official-segment-v1 protocol

## Scope

This protocol changes only the temporal input definition and its provenance.
It does not expose counts or cycle boundaries to pose extraction, training, or
prediction.

The privileged UCFRep manifest builder reads each official `.mat` payload and:

1. hashes the exact per-video annotation bytes as `annotation_sha256`;
2. reads scalar `start_frame` and `end_frame` values, which are 1-based and
   inclusive in the source annotation;
3. writes the equivalent 0-based half-open interval as
   `[clip_start_frame, clip_end_frame)`;
4. derives the evaluator count from `len(temporal_bound) - 1`, but never writes
   `temporal_bound` or `count` into a label-free pose-input sidecar.

The three clip-provenance fields are an all-or-none bundle. A manifest cannot
mix legacy full-video rows and official-segment rows.

## Extraction semantics

Use the explicit pose revision:

```yaml
pose:
  preprocessing_revision: official-segment-full-timeline-v1
  crop_to_detected_span: false
  incomplete_clip_policy: error
```

The extractor seeks to `clip_start_frame`, processes no more than
`clip_end_frame - clip_start_frame` frames, and preserves that entire temporal
axis. Frames without a usable pose remain exact-zero with
`valid_mask=false`. Neither first/last-detection trimming nor longest-track
selection is allowed.

`incomplete_clip_policy: error` is the default audit configuration. Early EOF
raises `PoseExtractionError`; batch extraction records the video ID,
annotation SHA, clip range, expected frame total, decoded total, and missing
tail in the failure ledger.

`incomplete_clip_policy: pad_invalid_tail` is a separate, opt-in configuration.
After the first failed sequential read, only the remaining suffix is appended
as invalid frames. The extraction summary and schema-v3 cache metadata record:

- `expected_clip_frames`;
- `decoded_clip_frames`;
- `padded_tail_frames`;
- `incomplete_clip_policy`.

No internal decode gap is skipped and no short clip is silently accepted.

## Provenance and compatibility

Legacy records omit all clip fields and keep schema-v2 pose caches. Official
records produce schema-v3 caches. Loaders remain compatible with v2, but
training/cache-set loaders compare the record's annotation SHA and exact clip
range against cache metadata. A full-video cache therefore cannot be mounted
under an official-segment sidecar even when its video SHA and base MediaPipe
settings match.

The official timeline revision and padding policy are part of both the full
configuration fingerprint and the pose fingerprint. In addition, clip fields
enter the dataset/sidecar/training fingerprints, while the pose-cache snapshot
hash covers the schema-v3 metadata bytes.

Use `configs/experiments/pams_official_segment_v1.yaml` for the strict
v8-specific coverage audit and
`configs/experiments/pams_official_segment_v1_pad_invalid_tail.yaml` only for
its explicitly padded full-dataset pair. These two files inherit v8's inferred
temporal-convolution/pre-PE head and are not paper-literal baselines.

For the first paper-architecture comparison, use
`pams_official_segment_literal_v1_pad_invalid_tail.yaml`. The separate
`pams_official_segment_reference_relative_v1_pad_invalid_tail.yaml` is an
explicitly inferred Period Head ablation.
