# ARIS Findings Register

## 2026-08-12 — Experiment audit

- **Verdict:** `FAIL`
- **Reason:** `NO_MULTI_PERSON_REAL_GT_EVIDENCE`
- **Consequence:** no current result is eligible for the proposed paper’s main
  table or abstract; UCFRep test105 remains sealed.
- **Artifacts:** `EXPERIMENT_AUDIT.md`, `EXPERIMENT_AUDIT.json`, and
  `.aris/traces/experiment-audit/2026-08-12_run01/`.

## 2026-08-12 — Result-to-claim gate

- **Route:** `pivot`
- **Verdict:** C1 no, C2 partial, C3 no, C4 no, C5 no.
- **Consequence:** proceed only as a pre-results method/evaluation contract;
  keep all empirical statements withheld and all proposed modules
  `SYNC-REQUIRED`.
- **Independence:** same-family reviewer, therefore provisional.
- **Artifacts:** `CLAIMS_FROM_RESULTS.md`, `RESULT_TO_CLAIM.json`, and
  `.aris/traces/result-to-claim/2026-08-12_run01/`.

## 2026-08-15 — Server version and experiment-integrity audit

- **Verdict:** `FAIL`
- **Version finding:** v63 is the largest numeric version but is code-only; v46
  is the most complete result bundle but implements a supervised
  spectral/ExtraTrees counter rather than TempoRAC
- **Protocol finding:** the 71-video v46 population is label-informed and mixes
  prior validation/test videos; the 180-video population has canonical-source
  overlap and partial prior test exposure; both use GT-assisted object tracks
- **Metric finding:** saved v46 count summaries recompute, but period metrics and
  paired source-cluster bootstrap are dead code in the executed path
- **Consequence:** no server candidate or number can be merged into the
  manuscript as canonical evidence; a clean implementation and leakage-free
  experiment bundle are required
- **Artifacts:** `EXPERIMENT_AUDIT.md`, `EXPERIMENT_AUDIT.json`, and
  `.aris/traces/experiment-audit/20260815_server_v46_v63/`

## 2026-08-15 — WARP-PHASE result-to-claim (`WARP-PHASE-R2C-20260815-SOL-001`)

- Route: `pivot`; acceptance remains `provisional` under same-family review.
- Gate 2 is a terminal frozen K5 failure at `246/55/144/50/66` versus `950/20/20/12/12`. It is synthetic engineering evidence, not efficacy, and it authorizes nothing.
- WARP-C1: `no` — no training or eligible endpoint result exists; no improvement claim is licensed.
- WARP-C2: `partial` — the blocked, out-of-order Gate-3 CPU smoke records `identity_isolation=true`, but no reached V3 or eligible real-data isolation result exists; this cannot rescue C1.
- WARP-C3: `no` — zero eligible efficacy results exist.
- WARP-C4: `no` — training, evaluation, and server launch are unauthorized.
- The corrected current tracker and pipeline summary now explicitly record Gate 2 `FAIL`; the integrity audit nevertheless remains `FAIL` on independent provenance/evaluator grounds, and the Round-2 code review remains `FAIL`. All claim confidence is therefore `low`.
- Frozen-stop disposition: preserve receipts and pivot. Do not tune thresholds, replace fixtures, correct harmonics, retry Gate 2, evaluate real tracks, or launch local/server training for this candidate.
