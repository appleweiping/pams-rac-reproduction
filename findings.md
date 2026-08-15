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
