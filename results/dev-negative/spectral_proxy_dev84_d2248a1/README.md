# Spectral-proxy strict UCFRep dev84 diagnostic

This is a deterministic reference diagnostic, **not a paper baseline** and not
evidence of a verified PAMS reproduction. It evaluates the missing-joint repair
introduced by source commit
`d2248a17c0ec25c4b0a3d3db8104a713dff9aad0` on the frozen UCFRep dev84 split.

| Metric | Value | 10,000-pair bootstrap 95% CI |
|---|---:|---:|
| NMAE | 0.617607 | [0.535541, 0.700492] |
| MAE | 3.916667 | [3.142857, 4.761905] |
| RMSE | 5.473964 | [4.379335, 6.552966] |
| OBO | 0.309524 | [0.214286, 0.404762] |
| Exact | 0.130952 | [0.059524, 0.214286] |

The reference fails both frozen development thresholds (`NMAE <= 0.228`,
`OBO >= 0.666`). Five all-invalid pose caches were retained under the common
failure policy and produced zero predictions.

The execution used three isolated CPU-only, network-disabled boundaries:

1. The identity preflight loaded only the label-free train337, dev84, and
   test105 identity sidecars and verified their commitments and disjointness.
2. Prediction mounted only the exact source tree, runner, dev identity inputs,
   identity receipt, pose cache, and output directory. No target sidecar and no
   test identity sidecar was mounted. The resulting 84 predictions were frozen
   at SHA-256
   `7efeec8c44e36786cd95b8f005357ad5a596f8a508c8290ee4b74f8a1dcf7261`.
3. A separate scorer first validated the frozen prediction, receipt, source,
   and identity hashes. Only then did it open `dev.targets`. Test targets were
   never mounted or read, and no test105 predictions were created.

Artifact roles:

- `predictions.json`: all 84 label-free per-video predictions and period
  diagnostics.
- `predictions.receipt.json`: immutable prediction and firewall bindings.
- `evaluation.json`: metrics, paired bootstrap intervals, and evaluator-only
  per-video errors.
- `evaluation.receipt.json`: evaluation/input hashes.
- `identity.receipt.json`: canonical 337/84/105 identity commitments.
- `run.manifest.json`: source, runtime, pose-cache, and output provenance.
- `artifact-index.json`: public artifact hashes and sanitized container mount
  audit.

