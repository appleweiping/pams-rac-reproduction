# PAMS-SSHead pre-PE v5: seed-2026 development result

**Status:** failed inferred repair; partial reproduction

**Scope:** UCFRep-526 train337 → dev84, seed 2026

**Test105:** not evaluated

This experiment is not author-disclosed PAMS. It is a one-variable repair of
the independently inferred `PAMS-SSHead`: the scalar Period Head consumes the
512-D pose projection immediately before sinusoidal positional encoding in
both training and inference. The post-warm-up period estimator remains the
projected-pose vector ACF used by the earlier v3 diagnostic.

The encoder and head were trained without count or action labels on the 337
training videos only. The dev84 count targets were opened once after the
configuration, source, and checkpoints were frozen.

## Result

| Metric | Value | Paired-bootstrap 95% CI |
|---|---:|---:|
| NMAE | 0.673120 | [0.591610, 0.749371] |
| OBO | 0.250000 | [0.166667, 0.345238] |
| MAE | 4.285714 | [3.511905, 5.130952] |
| RMSE | 5.742490 | [4.743416, 6.733070] |
| Exact | 0.154762 | [0.083333, 0.238095] |

The run fails the frozen acceptance thresholds (`NMAE <= 0.228`,
`OBO >= 0.666`). It must not be described as a successful reproduction.

The repair did remove the catastrophic count multiplier seen in the canonical
inferred full SSHead and improved over the prior projected-period v3 SSHead
diagnostic (`NMAE 2.074190`, `OBO 0.202381`). It did not produce a reliable
period signal: 32/84 videos were counted as zero and 28/84 selected the maximum
128-frame period. The final train337 Head log had spectral loss `0.865104`,
mean stream standard deviation `0.017515`, and zero optimizer steps with zero
gradient equal to `0`.

## Immutable bindings

- source Git SHA: `39d0d7f1144ed510f7a7a29696e27402a5a1da35`
- config fingerprint:
  `779689adf027a6f829fea0f37b64b10b525e2dafa2844c797202097f68493c0a`
- non-seed fingerprint:
  `6d63db9d1f008b4dbeeaa18c45114f65f4837492361ffad9964f1335f137a2ce`
- SSHead checkpoint SHA-256:
  `8d17e2938a9ebd47b84c07b2591c82fd96a88a10c82251173dc966ce9c95029c`
- SSHead progress SHA-256:
  `08a14c2037a3a66723db569e0866120a09a6307c61833a1351391b0ca4aae2fd`
- train337 pose-cache-set SHA-256:
  `34123ac27334ab06852c12254c7e0c046f7d86f3a066c81f9b8751c362dd0cd0`
- raw evaluation SHA-256:
  `5d50b85e0f9767cf810bf725e6656749fb0abd162c252677d1039a7eeae734c1`
- completed evaluation manifest SHA-256:
  `4448dbef21a23e31f79138828523739fbf5377e932771e4bd4e543b3b16c1a8b`

The complete 84-row predictions, expert counts, period streams, per-video
errors, and 10,000-sample confidence intervals are preserved in
[`pams_sshead_pre_pe_v5_seed2026_strict.json`](pams_sshead_pre_pe_v5_seed2026_strict.json).
