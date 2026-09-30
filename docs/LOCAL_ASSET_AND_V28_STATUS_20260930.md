# Local assets and V28 publication status (2026-09-30)

This is a status inventory, not a claim that V28 training or a new benchmark
result has been reproduced.

## Current checkout

- The public source branch contains the PAMS/RAC reproduction code, configs,
  tests, data-preparation recipes, and split identifiers. The current local
  checkout does **not** contain raw training videos, trained model checkpoints,
  pose/feature `.npz` caches, or a COCO training cache.
- No `v28d_parallel` standalone source tree or real-image occlusion-augmentation
  implementation was found in this checkout. Synthetic stress tests are not a
  substitute for real-image occlusion augmentation.
- A generated `uv.lock` is committed for dependency resolution; it does not
  include model weights or datasets.

## Designated research server (read-only inventory)

The separate research server has the UCF-101 video corpus used by UCFRep, a
421-entry pose cache, and a COCO-pretrained keypoint detector checkpoint. The
pretrained detector checkpoint is **not** a COCO training cache. A legacy
experimental `v28_residual_gate` script and one fitted `model.joblib` are also
present in a separate server checkout, but that script depends on earlier
experimental modules and is not the requested `v28d_parallel` independent
copy. No `v28d_parallel` directory was found in the inspected PAMS/RAC and
PhaseSet project paths. These observations do not authorize a V28 training
claim or sealed-test result.

## Publication boundary

Raw videos, third-party model weights, derived pose/feature caches, private
server coordinates, run outputs, and local Git bundles are not copied into this
public repository. Their presence on a research server does not grant
redistribution rights. The public reproducibility route is source, exact
configuration, tests, split identifiers, provenance checks, and instructions
to obtain authorized assets separately; see [DATA.md](../DATA.md) and
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

The V28 standalone gap remains open until its actual source tree, dependency
closure, data/weight acquisition instructions, and a clean-checkout run are
verified. Do not infer that the current branch is a self-contained V28 release.
