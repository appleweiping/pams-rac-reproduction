# Fresh ARIS Defense and Adjudication Prompt

## Role

Act as a fresh, zero-context independent defense reviewer and adjudicator for the current ICASSP 2027 pre-results manuscript. Defend the manuscript as strongly as the frozen evidence permits, then adjudicate every numbered attack finding as `answered`, `partially_answered`, or `still_unresolved`, grouped by P1-P6. Decide whether the strongest kill argument survives for the current artifact.

Be fair to the declared artifact type: this is explicitly a pre-results package. Separate (a) whether it is an honest, useful, evidence-gated research specification from (b) whether it could be submitted or accepted as a scientific paper now. Do not invent implementation details, results, claims, protocol bindings, or external evidence.

## Contamination boundary

Read only the frozen manuscript, evidence, and attack inputs listed below. Do not read prior reviews, proof audits, claim audits, citation audits, acceptance contracts, logs, old kill traces, or any pre-existing defense. Inspect every page of the five-page reviewed PDF visually. Do not modify the manuscript.

## Frozen inputs

### Reviewed manuscript and evidence

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

### Frozen attack

- `paper/.aris/traces/kill-argument-icassp/final/attack.md` - expected and observed `e4ad358e3e1ec4bcc15e0bcbd980716ce71174bce71ad0123c37cb352de956b9`
- `paper/.aris/traces/kill-argument-icassp/final/attack.meta.json` - `400a70a11dde891dfc3aa76c47c62f9f74eaa9bce6c55abb076beea6913c1cdb`
- `paper/.aris/traces/kill-argument-icassp/final/attack_prompt.md` - `496175d4d17c4caf754844c9d0e8e61356da924059ffc97029d4b668e8d67a5a`

All hashes must be recomputed before review. A mismatch invalidates the adjudication.

## Required adjudication method

1. State the strongest evidence-backed defense of the artifact.
2. Use these status meanings consistently:
   - `answered`: the current artifact directly rebuts or fully contains the attack within its declared pre-results scope.
   - `partially_answered`: the artifact contains a material narrowing, guard, or planned control, but frozen implementation or evidence is still needed.
   - `still_unresolved`: the current frozen inputs do not close the issue for scientific acceptance.
3. Adjudicate all 17 numbered attack findings: P1.1-P1.3, P2.1-P2.3, P3.1-P3.4, P4.1-P4.2, P5, and the four P6 bullets.
4. Keep evidence-integrity defenses separate from scientific sufficiency. Candor and submission guards can defeat a concealment/readiness attack without supplying missing science.
5. End with an explicit decision on:
   - validity as a pre-results package;
   - scientific submission/acceptance now;
   - whether the strongest current-artifact kill survives;
   - the recommendation if the PDF were submitted in its present form.

## Provenance and output contract

Record reviewer model `gpt-5.6-sol`, reasoning effort `ultra`, fork turns `none`, reviewer family `same-family`, and independence status `provisional`. Write only:

- `paper/.aris/traces/kill-argument-icassp/final/defense_prompt.md`
- `paper/.aris/traces/kill-argument-icassp/final/defense.md`
- `paper/.aris/traces/kill-argument-icassp/final/defense.meta.json`

Do not create `KILL_ARGUMENT.json`, `KILL_ARGUMENT.md`, or any manuscript file.
