# PAMS position lag-ACF + velocity fallback, dev84

This package records a completed **negative UCFRep dev84 diagnostic** for
seed `2026`. It is a `partial_reproduction`, not a verified PAMS result and
not a paper-table result.

The readout is an independently inferred completion because the paper does
not disclose a trainable loss or gradient path for its Period Head. It uses
the smallest detrended projected-position lag-ACF peak whose height is at
least 90% of the strongest eligible peak, with a projected-velocity spectrum
fallback when no positive lag peak is available.

## Result

| Split | Videos | NMAE | Raw MAE | Raw RMSE | OBO | Exact |
|---|---:|---:|---:|---:|---:|---:|
| UCFRep dev84 | 84 | 0.984422 | 5.187963 | 8.508895 | 0.285714 | 0.095238 |

The frozen acceptance thresholds were NMAE `<= 0.228` and OBO `>= 0.666`.
This run fails both thresholds. Its paired 10,000-sample percentile-bootstrap
95% intervals are:

- NMAE: `[0.740473, 1.254581]`
- raw MAE: `[3.835396, 6.736937]`
- raw RMSE: `[5.753383, 11.139988]`
- OBO: `[0.190476, 0.380952]`
- exact: `[0.035714, 0.166667]`

The 84 frozen predictions selected detrended position lag-ACF for 58 videos,
the velocity-spectrum fallback for 20, and no evidence for 6. Rounded counts
range from 0 to 51.

## Isolation and provenance

- The target-free pre-development gate passed before any dev84 identity,
  pose, or target input was accessed.
- Prediction mounted the frozen source, readout config, checkpoint, dev84
  identities, and dev84 poses. It did not mount development targets.
- Scoring mounted the frozen predictions, metric-only code, and development
  targets. It did not mount pose, checkpoint, or the full source tree.
- Both stages ran with networking disabled.
- The sealed test105 split was not mounted, predicted, scored, or evaluated.
- Source revision:
  `470d49bd97eef5adad89142d4ffc017125505525`.
- Formal pre-development artifact SHA-256:
  `f563cffa4235b34e091bf9b93af9123a93fa1d57ee29dc39f54f4feb60e0d928`.
- Readout config raw/semantic SHA-256:
  `b3c38341e442765cb709d8ca9c5ecfe541121ea61d1ae35c0ff2cb48b6b09a98` /
  `e17b4105f732be1bc4a40b153c0db28729fc0379a5f921ab0a26d3b2301b4233`.
- Encoder checkpoint SHA-256:
  `6817fe468d76c3fa2182dd831695d6fe3d6da208a2b7f8dfa90cffa6872e8053`.

## Package contents

- `prediction/`: all 84 target-free predictions and their immutable receipt.
- `evaluation/`: separately scored per-video evaluation and its receipt.
- `attempt.reservation.json` and `status.json`: run scope and completion.
- `audit/summary.json`: compact public interpretation and isolation record.
- `audit/SHA256SUMS`: hashes of the copied immutable source artifacts.

The development target manifest, pose caches, videos, model weights, server
paths, host/container inspection records, runner scripts, raw logs, machine
identifiers, and connection details are intentionally excluded.

