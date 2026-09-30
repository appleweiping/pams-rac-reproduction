# PAMS NoAbsPE SSHead seed 3407: train337-only collapse diagnosis

## Status

**Rejected before dev evaluation.** This package records a target-free failure of the independently inferred `PAMS-SSHead`; it is not a paper-reported implementation and it is not a successful reproduction result.

The encoder and SSHead both completed their scheduled training, but the learned head did not recover the training-side period structure:

| Diagnostic | Observed |
|---|---:|
| SSHead output period, median frames | 128.000000 |
| Training target period, median frames | 19.692308 |
| Final stream standard deviation, median | 0.010406 |
| Constant-stream total loss | 1.100000 |
| Constant-stream total-gradient nonzero elements | 0 |
| Constant-stream total-gradient L2 | 0.000000 |

The constant stream is therefore an exact stationary point of the inferred objective, while the trained head saturates at the maximum evaluated period and has very low temporal variation. The candidate was rejected and was not allowed to read or score the dev split.

## Scope and leakage audit

- Seed: `3407`
- Data used by this diagnosis: the frozen 337-video training partition only
- Count/action targets used: none
- Dev video inputs loaded: no
- Dev count targets loaded: no
- Sealed 105-video test split accessed: no
- Diagnosis mode: read-only checkpoint evaluation
- Implementation status: inferred, not author-disclosed

No server address, account name, key, absolute server path, raw video, model checkpoint, or private pose cache is included in this package.

## Training facts

- Encoder: 150 contiguous epochs, 1,500 optimizer steps, exit code 0
- SSHead: 30 contiguous epochs, 330 optimizer steps, exit code 0
- Encoder checkpoint SHA-256: `a5f08e8585cc0666eea265d9c38617d8845f3af0a4cf4fce92b1ef956e62b76d`
- SSHead checkpoint SHA-256: `08aa0936b8412dc6ad01bc5dda634e8819829406e57b8bbe368f08dadb0f368b`
- Source Git SHA: `0f5268c8496bb708fe07388e063b04ae7ddef5b3`

The SSHead training log's final epoch aggregate reported a stream-standard-deviation median of `0.009875`; the read-only full train337 diagnosis recomputed the final-head median as `0.010406`. The rejection decision uses the latter diagnostic artifact.

## Files

- `diagnosis.json`: complete sanitized target-free diagnosis
- `training-and-gate-summary.json`: compact training, access, and rejection record
- `SHA256SUMS`: package integrity hashes

