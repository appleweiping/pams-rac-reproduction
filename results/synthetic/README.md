# Synthetic acceptance evidence

This directory publishes the complete server-generated synthetic acceptance
artifact from source commit
`22e039aec6350fd8083c1f66a30283e2f5b92599`. It is a component diagnostic,
not an encoder/head benchmark result and not a paper-table row.

The formal CPU run evaluated 1,207 cases:

- 576 exact-period multi-expert counter cases;
- 576 FFT/autocorrelation-period plus multi-expert counter cases;
- 55 pose-level spectral-proxy cases expanded directly from
  `configs/stress.yaml`.

Both 576-case component gates passed. The 55-case pose gate failed its frozen
strict threshold: NMAE was `0.03921370609188828`, OBO was
`0.8909090909090909`, and maximum absolute error was `6`. Six cases exceeded
one-count error: count 30→28, 34→32, 38→36, 39→36, 40→42, and the
20%-time × 30%-joints missing-joint case 8→2. Constant 0.5/1/2 speed,
both linear speed ramps, all noise levels, and all second-harmonic levels had
OBO 1.0; both pause cases were within one.

The run used no UCFRep files or labels and records
`sealed_test_accessed=false`. The process exited with code 1 because the
predeclared pose gate failed; this is a valid negative result, not an
execution failure.

Published files:

- `synthetic-acceptance-22e039a.json`: full 1,207-case artifact, including
  every per-case prediction, input-component hashes, metrics, checks,
  hardware and provenance.
- `synthetic-acceptance-22e039a.receipt.json`: byte receipt for the full
  artifact.
- `synthetic-acceptance-22e039a-summary.json`: compact result and provenance
  index.

The full artifact SHA-256 is
`0b4b3734228fc2dd82f00db6b18449cc6342e3581606c1af38fa89aa3e01eb70`;
its byte count is `594313`. The receipt SHA-256 is
`650287982fa1cb93edf53cfcb5d8a748697fec17d68b806b4a4c7e1057709961`.
