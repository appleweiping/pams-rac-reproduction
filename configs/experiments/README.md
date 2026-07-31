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

## `pams_skeleton_augmentation_v13.yaml`

This variant adds the paper-disclosed skeleton rotation, isotropic scaling,
and Gaussian jitter only to encoder optimization batches. It changes no v8
pose cache, model, loss, period, SSHead, or inference setting. Because the
authors do not disclose augmentation magnitudes, the +/-15 degree rotation,
0.85--1.15 scale, and 0.01 jitter standard deviation are explicitly marked
as independently inferred. Prototype clustering and all period estimates use
the clean pose view; dev/test inference never calls this transform.

## `pams_paper_aligned_corrections_v14.yaml`

V14 is a pre-dev-frozen correction bundle rather than a one-factor ablation.
It uses the longest-contiguous v3 pose cache, full embedding-velocity vector
ACF after the ten-epoch pose warm-up, other-video frame negatives, and the
paper-disclosed skeleton augmentations. It restores the disclosed sinusoidal
encoder and pointwise `512->128->1` GELU Period Head instead of carrying over
the exploratory projected-pose/temporal-convolution v8 repairs. The velocity
interpretation and augmentation magnitudes remain independently inferred.
Seed 2026 must pass the target-free train/synthetic gate before one isolated
dev84 evaluation; test105 remains sealed.
