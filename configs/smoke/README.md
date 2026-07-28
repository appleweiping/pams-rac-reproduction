# Smoke-only configurations

Files in this directory are integration tests, not benchmark configurations.

`pams_encoder_1epoch.yaml` is byte-for-byte equivalent in parsed values to
`configs/pams.yaml` except that `training.epochs` is `1` instead of `150`.
It exists to exercise the full physical batch, KMeans refresh, PAMS-TCC
optimizer step, checkpoint, progress log, and run-manifest path on real
UCFRep pose caches. Any checkpoint produced with it must remain labelled
`smoke_only` and is ineligible for UCFRep result tables.

`pams_two_stage_1epoch.yaml` differs from the formal configuration only in
`training.epochs=1` and `sshead.epochs=1`. It is the required real-cache
two-stage gate: the encoder and SSHead must share this exact config identity,
and both terminal receipts must validate before any formal 150+30 epoch run.
