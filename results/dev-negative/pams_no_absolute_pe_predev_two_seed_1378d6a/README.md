# No-absolute-position-encoding target-free two-seed rejection

**Status:** `partial_reproduction`
**Classification:** independently `inferred`, target-free pre-development diagnostic
**Decision:** `predev_rejected` for both training seeds
**Development/test consequence:** neither candidate ran dev84 prediction/scoring or sealed test105

This package records a two-seed attempt to remove the PAMS Encoder's
absolute-position shortcut by setting `position_encoding_mode: none`. The
position-permutation consistency weight is `0.0`, because absolute positional
encoding is absent rather than regularized. This repair is an independent
completion and was not disclosed by the PAMS authors.

Training seeds `2026` and `42` each completed all 150 planned Encoder epochs
on the same 337 target-free training-pose caches. Each run has 150 contiguous
epoch records, 1,500 optimizer steps, 30 cluster refreshes, finite aggregate
statistics, and zero cross-cluster sampling shortfall. The same frozen
six-criterion target-free gate was then applied.

Both checkpoints passed four of six criteria. Both failed random-pose period
confidence and synthetic-period recovery, so the joint gate rejected them.
Consequently, neither checkpoint was authorized to predict or score dev84,
and neither was run on sealed test105. Passing `4/6` is not treated as a
successful reproduction.

The exact gate files use diagnostic RNG seed `2026` for both candidates so
the probes are directly comparable; the Encoder training seeds remain
`2026` and `42`, respectively.

## Frozen gate results

| Criterion | Frozen threshold | Seed 2026 value | Result | Seed 42 value | Result |
|---|---:|---:|:---:|---:|:---:|
| Canonical/permuted valid-frame embedding median cosine | `>= 0.95` | `1.0000000000` | Pass | `1.0000000000` | Pass |
| Absolute frame-index linear-probe R² | `<= 0.10` | `-0.1085032267` | Pass | `-0.1384453951` | Pass |
| Random-pose embedding-period confidence | `<= 0.10` | `0.2113084346` | **Fail** | `0.2498115897` | **Fail** |
| Synthetic-period median relative error | `<= 0.10` | `0.6666666567` | **Fail** | `0.3333333284` | **Fail** |
| Training-sample dominant period-bin share | `<= 0.25` | `0.1250000000` | Pass | `0.1718750000` | Pass |
| Zero-pose embedding-period confidence | `<= 0.10` | `0.0000000000` | Pass | `0.0000000000` | Pass |

A negative held-out-video R² means the absolute-frame probe is worse than a
constant predictor. It satisfies the one-sided shortcut gate but does not
show that the learned representation is useful. The two failed criteria show
that removing absolute positional encoding alone did not produce reliable,
pose-dependent period recovery.

## Immutable bindings

Both Encoder runs used training source
`1378d6a977b364f0de93c85663c4e8aaabd6ffbd` and immutable container image
`sha256:a6d10131c321c323c00bb85408cb8333503ecbc5d780d5059a73152d54df6005`.
Both gate outputs bind pose fingerprint
`3dd0388320095796f42aa82d904073b6478b073ed351394ef8d03c30798a0116`.

| Training seed | Config SHA-256 | Config fingerprint | Encoder checkpoint SHA-256 |
|---:|---|---|---|
| 2026 | `12f096d6021911d4f586f66b5741e8d73e07bb5edbba564d0e34c665d68a34f3` | `d7a81b4acc44b6148574553803c896d5183817200d7cac5b2621fdb7a87f8614` | `7c0ff57331203a230e5aaad6db50e82a1efc51b1f819aa4962f3323139d32d2c` |
| 42 | `daefe6684aa01ce659d77ecedeceecef910bcddc8d432dfdadd0d7471086e144` | `6a177379f48a2233d5e53df1d7f7c8799f81ba45fa322a55c9b3e58774aeabf0` | `d5770ae00b08fe7580343c001d555e269690af2d84bf69ddd3310e976156517c` |

Checkpoint files, raw progress logs, manifests, container inspection data,
dataset identities, and machine paths are intentionally not published.

## Files

- `seed2026-predev.json` and `seed42-predev.json` are the exact public-safe,
  path-free gate outputs.
- `audit-summary.json` records the firewall decision, immutable hashes, and
  all six frozen criteria.
- `sanitized-training-summary.json` contains only identity-free aggregate
  reductions of the two 150-row Encoder progress logs.
- `SHA256SUMS` binds every public artifact in this package other than itself.

This is negative evidence only. It is not a dev84 result, a test105 result, a
paper-table result, or a verified reproduction.
