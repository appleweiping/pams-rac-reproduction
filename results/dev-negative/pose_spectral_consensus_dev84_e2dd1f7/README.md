# Pose spectral consensus: dev84 negative result

**Status:** `partial_reproduction`  
**Classification:** inferred candidate, development-only diagnostic  
**Eligibility:** ineligible for the paper table  
**Split:** fixed UCFRep-526 development split, 84 videos  
**Source revision:** `e2dd1f7db6cba85f6bba4fd9148797a2e657711d`

This package preserves the completed, target-free prediction artifact and its
separate development-set evaluation for the independently inferred
`pose-spectral-consensus-harmonic-v1` candidate. It is negative evidence, not
a verified reproduction or a test-set result.

## Result

| Metric | Value | Paired bootstrap 95% CI | Frozen acceptance gate |
|---|---:|---:|---:|
| NMAE | `0.49295538376511283` | `[0.42717694183211424, 0.5568716945883663]` | `<= 0.228` — fail |
| OBO | `0.36904761904761907` | `[0.2619047619047619, 0.47619047619047616]` | `>= 0.666` — fail |
| MAE | `3.2738095238095237` | `[2.607142857142857, 4.011904761904762]` | diagnostic only |
| RMSE | `4.676435353883086` | `[3.711308330469125, 5.613599548455156]` | diagnostic only |
| Exact | `0.16666666666666666` | `[0.09523809523809523, 0.25]` | diagnostic only |

The reported NMAE is `0.492955` and OBO is `0.369048`; both miss the frozen
development thresholds. Confidence intervals use 10,000 paired resamples
with seed `2026`.

## Leakage boundary

The prediction receipt records that prediction loaded neither development
targets nor test identity/targets and used no GT count or action oracle.
Development targets were introduced only by the separate scorer after all 84
predictions had been frozen. Sealed test105 assets were not mounted, and
test105 was untouched.

This method is an independent, inferred candidate. It was not disclosed by
the paper, is dev-only, and is permanently ineligible for the paper-comparison
table.

## Independent checks

The public package was independently checked as follows:

- all five source JSON files parsed with duplicate-key and non-finite-value
  rejection;
- prediction and evaluation receipts match the copied artifacts by byte count
  and SHA-256;
- `status.json` binds all four artifact/receipt hashes and records successful
  completion;
- prediction and evaluation contain exactly 84 unique, identically ordered
  video IDs;
- per-video predictions, rounded counts, errors, and flags agree across the
  two artifacts;
- NMAE, MAE, RMSE, OBO, Exact, and every declared bootstrap interval were
  recomputed independently;
- a public-release sensitivity scan found no operational paths, command
  traces, secret material, restricted input payloads, or sealed-split
  identity records.

## Files

- `predictions.json` — 84 frozen target-free predictions and audit metadata.
- `prediction.receipt.json` — prediction hash, byte count, and label firewall.
- `evaluation.json` — separate dev-only metrics and per-video scoring rows.
- `evaluation.receipt.json` — evaluation and upstream prediction bindings.
- `status.json` — terminal status and cross-artifact hash bindings.
- `audit-summary.json` — compact claim boundary and validation summary.
- `SHA256SUMS` — SHA-256 manifest for every file above except itself.
