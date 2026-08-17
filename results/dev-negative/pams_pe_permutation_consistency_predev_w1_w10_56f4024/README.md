# PE-permutation consistency target-free predev rejection

**Status:** `partial_reproduction`
**Classification:** independently `inferred`, target-free pre-development diagnostic
**Decision:** `predev_rejected` for both candidates
**Development/test consequence:** neither candidate accessed dev84 or sealed test105

This package records two label-free attempts to suppress the PAMS Encoder's
absolute-position shortcut. The training objective adds consistency between
canonical embeddings and a view whose valid sinusoidal-position rows are
permuted. The loss weights were `1.0` (W1) and `10.0` (W10). This objective
is an independent completion and was not disclosed by the PAMS authors.

Both candidates completed all 150 planned Encoder epochs on the same 337
training-pose caches. They were then evaluated by the same six-criterion
target-free gate. W1 failed all six criteria. W10 removed the directly
measured position-index shortcut and passed the canonical/permuted embedding
consistency criterion, but it still collapsed to a nearly fixed short period
and failed the other four criteria. Therefore neither checkpoint was allowed
to produce dev84 predictions or access dev84 counts, and neither was allowed
to run on test105.

## Frozen gate results

| Criterion | Frozen threshold | W1 value | W1 | W10 value | W10 |
|---|---:|---:|:---:|---:|:---:|
| Canonical/permuted valid-frame embedding median cosine | `>= 0.95` | `0.1871570051` | Fail | `0.9999965429` | Pass |
| Absolute frame-index linear-probe R² | `<= 0.10` | `0.9946636726` | Fail | `0.0016639279` | Pass |
| Random-pose embedding-period confidence | `<= 0.10` | `0.5532907248` | Fail | `0.6061861515` | Fail |
| Synthetic-period median relative error | `<= 0.10` | `0.7179487273` | Fail | `0.7073170841` | Fail |
| Training-sample dominant period-bin share | `<= 0.25` | `0.9843750000` | Fail | `0.9843750000` | Fail |
| Zero-pose embedding-period confidence | `<= 0.10` | `0.5627833009` | Fail | `0.6075142026` | Fail |

The W10 result is particularly diagnostic: its valid-frame embedding median
cosine under PE permutation is `0.9999965` and its frame-index probe falls to
`0.00166`, yet `63/64` sampled training sequences still occupy one estimated
period bin. Removing the measured absolute-position encoding shortcut alone
does not restore pose-dependent period recovery under this inferred loss.

## Immutable bindings

| Candidate | Training source | Gate source | Config SHA-256 | Config fingerprint | Encoder checkpoint SHA-256 |
|---|---|---|---|---|---|
| W1 | `3d0d4f1b28c0b171bd4575c19b46c7c0b691a4c2` | `56f402472872825d2f18eb8a92772bf07677ac87` | `a84d3e2b897b8726f2a41ab291e036564f1a20a79189b941b54b4e2a87b85b13` | `a5d09f346a0cf6644f36ba5e638b20e97bcdc8fa047382b9d44363ade948b602` | `546a0ebb16aa24a4c73cc0cfde1de284584c0ecb88848c72a3eadeebd98a69be` |
| W10 | `ce9d1805c29c660cb5d952ae89d904d6485edf31` | `56f402472872825d2f18eb8a92772bf07677ac87` | `36a25b976f8e2e5b63427a743037fd68f40e0510d513292f71b3ec8568bfa130` | `8b645dac5333f14cb60303589d01e975903688d42547f6d2c01892114522a2fb` | `264c12bdc92f06de39018d712c7009c83762f1318d03973b35bc9d14d6ed679d` |

Both runs used the same immutable container image,
`sha256:a6d10131c321c323c00bb85408cb8333503ecbc5d780d5059a73152d54df6005`.
The checkpoint files, raw training logs, machine/container inspection data,
dataset identities, and server paths are intentionally not published here.

## Files

- `w1-predev.json` and `w10-predev.json` are the exact public-safe gate
  outputs. They contain no video identifiers or filesystem paths.
- `audit-summary.json` records the firewall decision, exact artifact
  bindings, and all six frozen criteria in a compact path-free form.
- `sanitized-training-summary.json` contains only aggregate statistics
  derived from the 150 training-log rows for each candidate. The raw logs
  are not included.
- `SHA256SUMS` binds every public artifact in this package other than itself.

This package is negative evidence. It is not a dev84 result, test105 result,
paper-table result, or verified reproduction.
