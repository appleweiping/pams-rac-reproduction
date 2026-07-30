# Inferred diagnostic configurations

These configurations are independent diagnostic repairs, not settings
disclosed by the PAMS authors. They are kept outside `configs/pams.yaml` so
the literal method, historical configuration fingerprint, and existing
checkpoints remain unchanged.

## `pams_pe_scale_v2.yaml`

This variant changes exactly one formal seed-2026 setting:
`model.input_projection_scale` is set to `sqrt_model_dim`. Before adding the
absolute sinusoidal position encoding, the encoder therefore computes

```text
sqrt(model_dim) * Linear(pose_frame) + position_encoding
```

The scale is an inferred anti-shortcut diagnostic motivated by a label-free
comparison of projected input-change magnitude and positional-encoding
magnitude. It does not establish improved counting performance. Its distinct
configuration fingerprint intentionally prevents it from entering the frozen
PAMS sealed-test path or being reported as `PAMS-Literal`.

## `pams_longest_contiguous_track_v8.yaml`

This is a data-protocol correction rather than an author-disclosed model
setting. Relative to temporal-conv v7, it changes only the pose preprocessing
revision: the earliest longest uninterrupted MediaPipe trajectory is selected
before per-frame normalization and 256-frame resampling. Historical v2 caches
used the first-to-last detected span and remain immutable. V8 must write a new
pose-cache namespace and is evaluated first on dev84; it cannot retroactively
upgrade any v2 result.
