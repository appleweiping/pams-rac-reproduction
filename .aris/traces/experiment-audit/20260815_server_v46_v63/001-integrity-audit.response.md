# Experiment-integrity audit

## Overall: FAIL

The current manuscript is honest about being pre-results, but neither v46 nor
v63 is valid canonical TempoRAC implementation or evidence. The v46 numbers are
internally reproducible, yet their protocols have material selection, leakage,
evaluator, and scope defects.

| Check | Status |
|---|---|
| A. Ground-truth provenance | FAIL |
| B. Score normalization | WARN |
| C. Result existence/binding | FAIL |
| D. Dead code | FAIL |
| E. Scope | FAIL |
| F. Evaluation classification | FAIL |

The 71-video v46 population combines 53 original validation and 18 original
test videos, and its manifest states that selection read count or period. The
180-video manifest records 27 train-test and 29 validation-test source overlaps,
but the result summary labels the run source-disjoint. Both use
GT-bbox-assisted pose-to-GT-object association. The second evaluation copies a
model already exposed to 18 official-test videos. Count summaries recompute
exactly, but the evaluator is custom. Period-mAP/AP50/AP75 and paired
source-cluster bootstrap are defined but never called. v46's three seeds are one
ensemble, not independent runs. v63 is source-only and uses GT-derived
best-channel proxy labels.

Neither candidate implements the manuscript's local-window tempo router,
shared fast/medium/slow response experts, normalized overlap-add, or one final
decode per identity. Existing v46 can only be described as a historical
GT-assisted count-only diagnostic under a compromised protocol; v63 is an
unexecuted exploratory proxy. A new frozen implementation and leakage-free
experiment bundle are required. Full evidence and remediation appear in
`EXPERIMENT_AUDIT.md`.
