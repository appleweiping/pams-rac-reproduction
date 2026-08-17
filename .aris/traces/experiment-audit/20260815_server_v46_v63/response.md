# Raw reviewer verdict

Overall verdict: **FAIL**

The current manuscript is honest about being pre-results, but neither v46 nor
v63 is valid canonical TempoRAC implementation or evidence. The v46 numbers are
internally reproducible, yet their protocols have material selection, leakage,
evaluator, and scope defects.

- Ground-truth provenance: FAIL
- Score normalization: WARN
- Result existence and binding: FAIL
- Dead code: FAIL
- Scope: FAIL
- Evaluation classification: FAIL

The 71-video v46 result is outcome-selected: it contains 53 original validation
and 18 original test videos, and its manifest says selection read count or
period. The 180-video manifest records 27 train-test and 29 validation-test
source overlaps but the summary calls the run source-disjoint. Both use
GT-bbox-assisted pose-to-object association. The official run copies a model
already exposed to 18 official-test videos. Reported count summaries recompute,
but the evaluator is custom; period metrics and paired source-cluster bootstrap
are defined but never called. v46 uses three seeds as one ensemble, not three
independent runs. v63 is code-only and uses GT-derived best-channel proxy labels.

Neither v46 nor v63 implements the proposed local-window tempo router, shared
fast/medium/slow response experts, normalized overlap-add, or per-track single
decoding. A new frozen implementation and leakage-free experiment bundle are
required. Existing v46 may only be described as a historical GT-assisted
diagnostic, and v63 only as an unexecuted exploratory proxy.

The evidence and line-level findings are preserved in the canonical audit
report `EXPERIMENT_AUDIT.md`.
