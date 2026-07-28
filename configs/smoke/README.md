# Smoke-only configurations

Files in this directory are integration tests, not benchmark configurations.

`pams_encoder_1epoch.yaml` is byte-for-byte equivalent in parsed values to
`configs/pams.yaml` except that `training.epochs` is `1` instead of `150`.
It exists to exercise the full physical batch, KMeans refresh, PAMS-TCC
optimizer step, checkpoint, progress log, and run-manifest path on real
UCFRep pose caches. Any checkpoint produced with it must remain labelled
`smoke_only` and is ineligible for UCFRep result tables.
