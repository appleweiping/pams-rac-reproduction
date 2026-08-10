# PAMS-SSHead confidence-weighted v6: seed-2026 development result

**Status:** failed inferred repair; partial reproduction

This single-variable experiment keeps the v5 pre-PE Head, architecture,
period estimator, and inference unchanged. Only eligible cycle and spectral
losses are averaged with their continuous label-free period confidence,
normalized by total confidence. The encoder and Head used train337 only;
dev84 targets were opened after both stages completed. Test105 was not
evaluated.

| Metric | v6 | v5 seed 2026 | v6 − v5 |
|---|---:|---:|---:|
| NMAE | 0.711541 | 0.673120 | +0.038421 |
| OBO | 0.190476 | 0.250000 | −0.059524 |
| MAE | 4.380952 | 4.285714 | +0.095238 |
| RMSE | 5.723802 | 5.742490 | −0.018688 |
| Exact | 0.130952 | 0.154762 | −0.023810 |

The paired 10,000-sample bootstrap interval for the NMAE difference is
`[-0.021805, 0.103330]`; the OBO difference interval is
`[-0.119048, 0.000000]`. Confidence weighting reduced the final train337
spectral loss from v5's `0.865104` to `0.793636`, but did not improve the
development count. The readout still produced 29 zero counts and selected
period 128 for 32/84 videos.

This run fails the frozen thresholds (`NMAE <= 0.228`, `OBO >= 0.666`) and
does not justify additional seeds for this isolated repair.

## Immutable bindings

- source Git SHA: `019ae1f72d41f9fb70cc484514c12a96c0fff4fc`
- config fingerprint:
  `01b0797639aef0314d70a858fc99d33ec05fb79a20411fdcc031b585329b4e9d`
- encoder checkpoint SHA-256:
  `4a6971701fcd9649221300caaefc121049cafc08fdc847f8e00dbae331ae3eb9`
- encoder progress SHA-256:
  `25449f584dd9088092275e4cba0d7992edd74c05efa5195f9555a147d5f3987c`
- encoder completion receipt SHA-256:
  `b8e31e63e7cd16d7fd87c530118f1ef6eeac6eb757487ec38ec773b0453d7069`
- SSHead checkpoint SHA-256:
  `72a6d234d0ea7974e5d20f731f6ae9bd4271033d2e1489353be441b167e9dbe8`
- SSHead progress SHA-256:
  `46c268ec3097d29d10177be5a301487793ee0994ce3aee7fb097043a42fdacbb`
- SSHead completion receipt SHA-256:
  `d7b67782e0a07138d5f07856452be278af27be8b67fdaf7469ae0cea5b9b2565`
- raw evaluation SHA-256:
  `ed5394e02d1bafe80f87c81a9b022a4827c561d37b10e94dd29a2f93d28b0afe`
- evaluation completion receipt SHA-256:
  `00b390706326a3a058525691c1e7755f30fd5b6f3f32861e10eb039bd59109c8`

The complete predictions, expert counts, period streams, per-video errors,
and bootstrap intervals are in
[`pams_sshead_confidence_weighted_v6_seed2026_strict.json`](pams_sshead_confidence_weighted_v6_seed2026_strict.json).
