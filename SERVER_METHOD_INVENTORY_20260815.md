# Server Method Inventory

**Date:** 2026-08-15

**Mode:** read-only bounded inventory; sealed reports were not opened

## Verdict

No server branch, worktree, source tree, or result artifact implements the
paper's proposed method end to end. No collaborator-designated canonical or
final branch was found.

## Exhaustive bounded PAMS ref scan

The shared PAMS repository exposes 61 distinct worktree HEADs. Across those
HEADs, the inventory found:

| Primitive | HEADs containing it |
|---|---:|
| fast/medium/slow consensus defaults | 61 |
| single-expert readout | 15 |
| keypoint pseudotrack continuity | 13 |
| within-video windowed PAMS-TCC | 1 |
| implemented identity-indexed tempo router | 0 |
| implemented routing-consistency objective | 0 |
| implemented normalized overlap-add path | 0 |
| complete per-track one-time decoding stack | 0 |
| complete requested method | 0 |

Git-log content searches across all refs likewise found no integrated
identity-indexed route, routing-consistency, per-track decode, or normalized
overlap-add implementation.

## Isolated primitives and their semantic limits

- `ee200:src/pams/consensus.py`, SHA-256
  `55491a9a7cc786f6df7611de9f6d356efa5ce5eb6477f6f6782fff811bfbc3b5`,
  defines fast/medium/slow settings but accepts one one-dimensional period
  stream with no identity or track input.
- `ee200:src/pams/single_expert_readout.py`, SHA-256
  `9fde3d70522a3dc7f0e0b0fdd5dcdcbf1146886ea83de8ed022af6f90e9d97af`,
  performs full-segment encoding and explicitly forbids window-local or
  overlap-add receipts; multi-segment total semantics remain undefined.
- `e101:src/pams/losses.py`, SHA-256
  `8f558321720f87e66ed14fba657a3f8e938b0c0954e63cc002d873eb87e80972`,
  implements opt-in exact-cycle windows only for a within-video PAMS-TCC
  candidate loss; it has no identity routing.
- `b567:src/pams/keypoint_single_source.py`, SHA-256
  `0abaded5127de09a70e131f5b771a2a67f8890d1984c5ea23d63bddc5098e20a`,
  builds pose-continuous pseudotracks and decodes once but explicitly disclaims
  physical-person or cross-segment identity and forbids cross-reset aggregation.
- `ee200:src/pams/coherent_ridge_selected_identity_bridge.py`, SHA-256
  `34b2f4487fcda6c8363e8ff116975f13d307b769bee3f81f1eb155c093c969a2`,
  uses “identity” for SHA/config/artifact identity, not a person identity.
- `Count-main/mocount/window_fusion.py`, SHA-256
  `cf043c5ccf7080662056580de845217bcd48882dde085854543313c7688a56a5`,
  contains normalized Hann overlap-add, but its tree has no combined
  identity/router/PAMS stack.
- MotionBERT's untracked MPCounter v45/v46 code uses fixed person-slot
  embeddings and a global supervised count pipeline, not tracked-identity local
  tempo routing.

The latest tracked PAMS README still states partial reproduction with no
verified benchmark claim. Its method specification selects one longest
contiguous main subject and disclaims cross-person identity tracking.

## Result coverage

- Windowed-TCC runs completed eleven train337-only epochs, with all dev84,
  test105, and scoring authorizations false.
- Segment-local formal outcomes cover synthetic selection only; real-data,
  representation, training, development, and test authorizations are false.
- The official segment-local-frequency formal run ended in scientific
  rejection.
- Pose-v4e reservations are synthetic only, with all real/train/dev/sealed flags
  false.
- v41/v42 “router” JSON files choose a model/source globally per video; they are
  not identity-window fast/medium/slow routing.

## Canonical implementation consequence

No existing server tree can be updated into the paper as a final method. The
only defensible paths are:

1. receive an exact collaborator SHA/bundle that was not discoverable in the
   inventoried roots and audit it independently, or
2. implement the missing integration on a clean branch, reusing only audited
   primitives while newly defining physical identity semantics, per-window
   routing, any auxiliary objectives, normalized reconstruction, and one decode
   per person

The ARIS pipeline proceeds with path 2 unless a stronger collaborator artifact
appears.
