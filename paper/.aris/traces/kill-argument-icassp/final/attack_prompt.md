# Fresh ARIS Kill-Attack Prompt

## Role

Act as a fresh, zero-context attack reviewer for the current ICASSP manuscript. Build the strongest scientifically grounded rejection case. Do not read or reuse any prior kill, proof, claim, citation, review, or audit history. Do not modify the manuscript and do not construct a defense.

## Frozen review inputs

- `paper/main.pdf` - `873c83da0dbf4b24a6e1199f3dcd980e17d0b032580e0591f88a7c545ec67a45`
- `paper/main.tex` - `a4fe29c47f200946c7d6f3522bd273f5895aabda2d7a959fdbf16397fe1cbe70`
- `paper/preamble.tex` - `763b456cb2e22585ca23160c2b12d641e764c70da94641247f6268af1a7306a2`
- `paper/math_commands.tex` - `e6dabb7ce789e0d78026e6bda287fe17faedb6d20c0df9b4f603cb3b6c123bee`
- `paper/sections/0_abstract.tex` - `fd69c7936952bffc8e70d7d0ccd35140ab7dbaac82d6ff953e60f4f6461eebef`
- `paper/sections/1_introduction.tex` - `6823651162f59e5f5263949f2d308f594b667cb3cba0044433c39031ae04db9d`
- `paper/sections/3_method.tex` - `7e92c00eb4a7956b059536fd47240cefbea960aee61d4f3fdef1a7e24930c50d`
- `paper/sections/4_experiments.tex` - `6052c816e94f7d7f50b560d397495a37863b05d240ed5e418e4b6ec4d3aa131b`
- `paper/sections/5_conclusion.tex` - `e5c656a98343f10b367c175273d572f3a42d5436539429f28ba87764452e7b3c`
- `paper/references.bib` - `322a5d16b0078e8b3436608d4c6158b67c5c8922b0dcae94537ea456c9f6f646`
- `paper/figures/icassp_framework.pdf` - `3afa9d1261e3a54b25c8da708dd7c4e036587d6d01d0d60fc704edcd2fd8eb81`
- `paper/evidence/method_manifest.yaml` - `07823bb5bb11509e79556f87c9bbb19c9ce4f0ff1fdea1b5cb832e89ae8cd289`
- `paper/evidence/results_manifest.json` - `4df513a3f4427950d23267965192253446e6ee2b6968907855e0bae1a6a53159`

`paper/generated/evidence_values.tex` is absent, so no generated result or protocol binding is active.

## Required attack axes

1. Separate honest pre-results disclosure from genuine scientific fatality. Do not allege fabricated evidence where the manuscript explicitly withholds it; assess whether a result-free, implementation-free specification is publishable now.
2. Attack novelty relative to the paper's own closest-work characterization and its admission that spectral estimation, overlap-add, and expert routing are established.
3. Test whether the equations and manifests define an executable, reproducible algorithm and training objective.
4. Test whether a supplied track is scientifically equivalent to a person, especially under fragmentation, re-entry, ID switches, merge, retirement, and reconciliation.
5. Audit the boundary of the no-person-wise-label claim across representation learning, pseudo-targets, router targets, losses, selection, stopping, calibration, inference, pose, detection, and tracking.
6. Test whether the core hypothesis has a frozen, falsifiable estimand and an unconfounded mechanism test.
7. Attack protocol fairness across native video pipelines, oracle identity tracks, common predicted tracks, pose frontends, adapted baselines, output units, and metric assignment.
8. Treat missing implementation and missing results as evidence states, not mere prose omissions.
9. Inspect every PDF page visually and report whether visual defects materially affect the verdict.

## Output contract

Write a provisional attack using a descending P1-P6 severity scale, a concise kill chain, and a strong-reject verdict. Record exact input hashes and reviewer provenance. Do not generate `KILL_ARGUMENT.json`; an independent defense must be completed before final adjudication.
