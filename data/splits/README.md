# Split ID manifests

These files contain only canonical UCF101 video identifiers, not videos,
counts, action labels, frame boundaries or extracted poses.

The standard UCFRep 421/105 lists are derived from the official annotation
archive pinned in `configs/protocols/ucfrep_526.yaml`. The 337/84 development
lists are produced by `pams.data.deterministic_stratified_split` with seed
2026. Regeneration must result in the same repository diff before an
experiment is accepted.
