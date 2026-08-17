# WARP-PHASE Pilot v1 Experiment Code Review — Trace Response

**Verdict: BLOCKED.** Implementation disposition is **REVISE**. Training and server launch are not authorized. This is a same-family `gpt-5.6-sol` review, so acceptance is provisional.

The exact candidate pack (`b2c791f9aac1d86c0849fa74b00b4e605fe21120f9d188f4ec0b304d7052f173`) is not semantically freeze-valid. Its generator drops selector diagnostics on all 240 canonical failure rows; mandatory IDs 0, 150, 250, and 900 raise before Gate 1 can compare the `-1` outcome; and the normative deterministic FFT witness set is absent. The candidate also genuinely fails every Gate 2 threshold: agreement 246/1000, half 55, double 144, symmetric half 50/250, and symmetric double 66/250.

Independent model probing found invalid-frame leakage: changing only pose at a `frame_mask=false` clock changes valid-frame phase outputs through the pre-GRU temporal convolutions (maximum absolute difference `1.9523284435272217`). Environment provenance, process-level CLI failure semantics, complete config-bound commands, K4 execution, privilege separation, and decision-bearing tests also require revision.

The reviewed primitives that are correct include hash-before-unpickle on the same stable bytes, safe feature archive loading, training/evaluator import isolation in the implemented path, exactly 225,026 parameters, four fresh backward passes followed by one clipped AdamW step, and the tested warp/loss/decode/narrow temporal evaluator primitives.

Verification: 118 tests passed and 1 Windows symlink-privilege test skipped; Ruff passed; mypy passed for 15 source files. No server or real train/validation/test/sealed/held-out/results artifact was accessed, and no experiment or job was run.

The complete findings, required resolutions, and exact per-source/per-test/per-pack-member input ledger are in `refine-logs/EXPERIMENT_CODE_REVIEW.md` and `refine-logs/EXPERIMENT_CODE_REVIEW.json`.
