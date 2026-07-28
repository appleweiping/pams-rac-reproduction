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
