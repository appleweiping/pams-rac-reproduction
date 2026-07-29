# PAMS-SSHead temporal-conv v7: two-seed development audit

**Status:** failed inferred repair; partial reproduction

This preregistered repair inserts one mask-aware, zero-initialized residual
depthwise `Conv1d` (`512` channels, kernel `5`, `2,560` parameters) before the
unchanged disclosed `512 -> 128 -> 1` GELU MLP. It otherwise keeps the v6
pre-PE source, label-free period teacher, continuous confidence weighting,
loss weights, and consensus fixed.

Both pipelines trained only on train337: 150 encoder epochs followed by 30
frozen-encoder Head epochs. Development targets were opened only after each
terminal Head checkpoint was frozen. Test105 was not mounted for prediction
or scoring.

| Seed | NMAE | OBO | MAE | RMSE | Exact | Zero counts | Period 128 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 0.694489 | 0.238095 | 4.261905 | 5.694609 | 0.095238 | 23 | 25 |
| 2026 | 0.735728 | 0.190476 | 4.428571 | 5.794086 | 0.107143 | 31 | 30 |
| Mean ± sample SD | 0.715109 ± 0.029161 | 0.214286 ± 0.033672 | 4.345238 ± 0.117851 | 5.744347 ± 0.070341 | 0.101190 ± 0.008418 | — | — |

The 95% paired-bootstrap intervals are:

- seed 42: NMAE `[0.611716, 0.778336]`, OBO
  `[0.154762, 0.333333]`;
- seed 2026: NMAE `[0.656124, 0.815035]`, OBO
  `[0.107143, 0.273810]`.

Neither seed reaches `NMAE <= 0.228` or `OBO >= 0.666`, so the required
two-seed acceptance condition is impossible. Seed 3407 was therefore not run,
and test105 remains sealed.

## Controlled comparisons

For seed 2026, v7 minus v6 changes NMAE by `+0.024187`
(95% CI `[-0.016857, 0.083096]`) and OBO by `0.000000`
(`[-0.035714, 0.035714]`). For seed 42, v7 minus v5 changes NMAE by
`-0.045804` (`[-0.156229, 0.051039]`) and OBO by `-0.023810`
(`[-0.107143, 0.059524]`). Neither comparison demonstrates a reliable
improvement.

The convolution received gradients (`zero_grad_steps=0`) in both runs, but
the final stream standard deviations remained only `0.017259` and `0.014684`.
The new local context therefore did not remove the low-frequency/zero-count
failure mode.

## Immutable bindings

- source Git SHA:
  `6ee25a2e1291c35c563140999e16c5aa65bc8ced`
- container image:
  `sha256:620173fe7a9084d3b1492c3e775ea5f2bc6ac5a6d851d8560727a4969b3aab66`
- seed 42 config:
  `0b44f415a8083fb35205ad6a1a085a92df68b6b4bcc2974cae101a9a381196c7`
- seed 42 encoder / Head / evaluation:
  `5e76225e...02b50` / `3d1e7cf5...0d873` / `39c7ceeb...1ce6`
- seed 2026 config:
  `317442965eb856a59a5b70f3ec63756fdceec16c55c9c31820b61a7c9d25ccca`
- seed 2026 encoder / Head / evaluation:
  `c3370074...1333b` / `0f88bcad...ce882` / `1abcf7b1...5375`

The complete artifact and receipt hashes are in
[`pams_sshead_temporal_conv_v7_two_seed_strict.json`](pams_sshead_temporal_conv_v7_two_seed_strict.json).
All 84 per-video predictions, expert counts, period streams, errors, and
10,000-bootstrap intervals for each seed are in
[`pams_sshead_temporal_conv_v7_seed42_strict.json`](pams_sshead_temporal_conv_v7_seed42_strict.json)
and
[`pams_sshead_temporal_conv_v7_seed2026_strict.json`](pams_sshead_temporal_conv_v7_seed2026_strict.json).
