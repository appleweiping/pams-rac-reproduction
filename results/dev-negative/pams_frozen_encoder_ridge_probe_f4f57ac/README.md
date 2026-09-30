# Frozen PAMS encoder supervised Ridge probe

This directory records the one-shot `dev84` result for
`pams-frozen-encoder-ridge-count-probe-v1`. It is a supervised diagnostic
upper bound over a frozen PAMS encoder, not a self-supervised PAMS result,
baseline result, or verified reproduction.

The probe was frozen before scoring:

- canonical seed-2026, 150-epoch multiscale encoder;
- 1027 features: masked 512-D embedding mean, masked 512-D population
  standard deviation, `log1p(n_valid / period)`, period confidence, and valid
  frame fraction;
- train337-only `StandardScaler` and
  `Ridge(alpha=1, solver=svd)` on `log1p(count)`;
- one label-free dev84 prediction pass followed by a separate scorer;
- no hyperparameter search, action label, test identity, or test label access.

## Result

| Metric | Result | PAMS verification threshold | Status |
|---|---:|---:|---|
| NMAE (rounded) | 2.133704 | <= 0.228 | fail |
| OBO | 0.321429 | >= 0.666 | fail |
| MAE (rounded) | 7.202381 | - | - |
| RMSE (rounded) | 22.896506 | - | - |
| Exact | 0.107143 | - | - |

The 10,000-sample paired-bootstrap 95% intervals are
`NMAE [0.712083, 4.596175]` and `OBO [0.226190, 0.416964]`.
Thirteen predictions round to zero; two raw predictions exceed 40, with a
maximum of 184.176. Although the train337 fit reaches raw-count
`MAE 1.424597 / RMSE 3.462829`, it does not generalize to dev84. The result
therefore rejects the hypothesis that the current frozen representation
becomes paper-level simply by adding this fixed linear supervised readout.

All three containers exited with code 0 under `--network none`, read-only root
filesystems, exact mount whitelists, and split-only copied pose-cache views
(337 train / 84 dev). The sealed 105-video test set was not mounted or
accessed.

## Provenance

- Probe source: `f4f57ac434ae02efbde59ed50a2de5da67515bf3`
- Encoder source: `6c52288d6a227bc3040fd672c31039408e2d6696`
- Encoder checkpoint:
  `ecaf3c2bd54ffc12c396f9da0044b2799917fb8a6f3811547ba9bcab6fe9d45e`
- Container image:
  `sha256:5b2c307be606261ca1a354796475040f799b8e97e6ee4fe7598addbbd7749c4a`
- Predictions:
  `cdc36e75c3933918f219c52296ead283b423680b33518400cba6b7625b3c72b2`
- Evaluation:
  `aeaeeb24a3c38cddad8198ae53a16d8617d4a9956f4fb44fad891cd92ec9a256`
- Server audit summary:
  `643850fec6cb83c30b3944feeb3250d6fb67977cb7707fdb7abba3559824f595`

CUDA emitted a CuBLAS deterministic-workspace warning because the launcher
did not set `CUBLAS_WORKSPACE_CONFIG`. No rerun was performed after dev
scoring, so the one-shot result is retained with that reproducibility caveat
instead of adapting the protocol after seeing labels.
