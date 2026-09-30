# Designated-server component smoke

This directory records the first current-preprocessing, real-UCFRep component
smoke. It is deliberately not a benchmark table and does not change the
repository's `no_results` verification status.

The run used commit `5088cb14bafb95c24fead7d31422f0018571f928` in an
offline container bound to that exact source revision and environment hash.
Only GPU index 1 was exposed. The public evidence omits the private hostname,
account, SSH material, absolute host paths, raw videos, and pose arrays.

Operator-recorded outcomes:

- Ruff, mypy 1.13.0, and `pip check` passed in the locked container.
- The CUDA-enabled suite passed: `271 passed`.
- All 526 UCFRep videos resolved and were content-hashed as 421 train and
  105 sealed test records.
- The frozen development split validated as 337 train, 84 dev, and 105 test.
- A read-only identity audit found 502 direct layout matches and exactly 24
  `HandStandPushups` records using the official `HandstandPushups` directory;
  no video was missing or ambiguous.
- `v_BabyCrawling_g01_c01` decoded 165 frames, produced 165 valid pose frames,
  and was uniformly cached to 256 valid frames. A second run returned
  `skipped=true` only after matching the video hash and pose fingerprint.
- All 421 official training-pool caches were then materialized. A separate
  identity-only resume reported 421 skips with zero extraction or failure.
- Independently, the path-free cache audit matched all 421 source-video hashes
  and pose fingerprints, found 421 unique cache byte streams, found no missing
  or extra cache, and produced an ordered cache-set digest. Across 107,776
  cached frames, 83,376 were valid (77.36%); the per-video median was 98.83%.
  Thirteen videos remained all-invalid and were retained as zero-valued masked
  samples.
- Both 576-case counter gates passed. The SSHead collapse diagnostic retained
  its expected nonzero exit and known constant-stream failure.
- The full 51-sample spectral-proxy stress run completed as a diagnostic only.
  It remains ineligible for every paper-comparison table.
- PyTorch exposed 47.318 GiB on the selected A6000, below the frozen 48 GiB
  CountLLM-Lite gate, so that incomplete adapter remains `smoke_only`.

The compact digest summary is in
[`evidence-summary.json`](evidence-summary.json). The sanitized
[`421-cache audit`](pose421-cache-audit.json) and clean
[`identity ledger`](pose421-identity-ledger.json) are published directly.
Generated arrays, raw videos, and raw server logs are not published; hashes for
those retained operator artifacts are therefore not independently checkable
from this repository.
