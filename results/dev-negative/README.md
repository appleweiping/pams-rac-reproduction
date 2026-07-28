# Development-only negative evidence

**Status:** `partial_reproduction`

**Split:** fixed UCFRep-526 development split, 84 videos
**Source revision:** `f99f3fda91b8bf2c25fc75a42814d9c97df71ae0`

This directory records negative development evidence for the literal Period
Head and the independently inferred SSHead completion. It is not a test
result, leaderboard result, successful reproduction claim, or ablation
result.

The three preregistered seeds `42`, `2026`, and `3407` completed on the same
development split. PAMS-Literal has mean NMAE/OBO
`0.7299247978 / 0.2500000000`; PAMS-SSHead has
`4.9412041954 / 0.0515873016`. Each variant passes the preregistered
NMAE/OBO diagnostic on `0/3` individual seeds, and neither three-seed mean
passes.

For the 10,000-sample paired bootstrap comparison
`PAMS-SSHead - PAMS-Literal`, the NMAE-difference 95% CI is
`[+3.3337010274, +5.1417284399]` and the OBO-difference 95% CI is
`[-0.2698412698, -0.1309523810]`. Both intervals show worse development
behavior for the inferred SSHead in this run.

Two additional seed-2026 diagnostics are retained only to help localize the
failure:

| Diagnostic | Variant | NMAE | Raw MAE | RMSE | OBO |
|---|---|---:|---:|---:|---:|
| Pose-period-only | PAMS-Literal | 2.475464 | 9.202381 | 10.545254 | 0.083333 |
| Pose-period-only | PAMS-SSHead | 4.707033 | 17.880952 | 20.475886 | 0.059524 |
| Single-scale | PAMS-Literal | 4.368132 | 17.178571 | 19.973494 | 0.035714 |
| Single-scale | PAMS-SSHead | 4.314496 | 16.190476 | 18.448771 | 0.071429 |

These diagnostics change more than one condition relative to the primary
three-seed run and are not admissible evidence for a component ablation.

## PE-scale v2 inferred diagnostic

A separate seed-2026 run at source revision `fea4a7d` multiplied the input
projection by `sqrt(model_dim)` before adding the sinusoidal position
encoding. This change was independently inferred, was not disclosed by the
authors, and is permanently ineligible for a paper-comparison table.

| Variant | NMAE | Raw MAE | RMSE | OBO |
|---|---:|---:|---:|---:|
| PE-scale v2 / literal-head diagnostic | 0.677779 | 4.630952 | 6.377975 | 0.214286 |
| PE-scale v2 / SSHead-inferred diagnostic | 4.626961 | 17.702381 | 20.152691 | 0.047619 |

The scale change increased the effective pose-projection/position-encoding
norm ratio from `0.381` to `7.611` and reduced the seeded random-pose
period confidence from `0.644` to `0.109`. It did not remove absolute frame
index information (`R² = 0.967`), increased the measured cross-scale conflict
rate from `0.9%` to `17.4%`, and made the period estimate more sensitive to an
orthogonal embedding-basis change. Literal-head development performance did
not improve. SSHead NMAE improved by `-0.3810` versus f99 with a paired 95% CI
of `[-0.4925, -0.2751]`, but remained catastrophically above the `0.228`
diagnostic threshold. The other two seeds were therefore not run.

The compact metrics, mechanism measurements, artifact hashes, and
container-provenance limitation are recorded in
[`pe_scale_v2_seed2026.json`](pe_scale_v2_seed2026.json).

The operator-recorded source aggregate has SHA-256
`c8f6a117ce635d91d7c4446b2f2aea156bb2f3ed0ae86460986685c16cdeb814`.
The raw aggregate and per-video predictions are not published in this
directory, so the digest binds the operator's source artifact but is not by
itself independently reproducible evidence. Exact path-free values are in
[`summary.json`](summary.json).

No prediction or evaluation has been run on the 105-video test split, and no
test metric is reported. The development command nevertheless loaded and
deserialized the complete 337/84/105 split manifest. Consequently, this
record makes no byte-level “test labels untouched” claim and is ineligible as
a leak-free sealed result.
