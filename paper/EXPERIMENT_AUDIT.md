---
audit_skill: experiment-audit
venue: ICASSP 2027
generated_at: 2026-08-14T21:00:56.8906401+08:00
verdict: FAIL
reason_code: NO_MULTI_PERSON_REAL_GT_EVIDENCE
review_independence: same-family/provisional
auditor_agent: /root/icassp_experiment_audit
evidence_freeze: 26e5b7176f1f7c678391163c3278b5c390d54115
---

# ICASSP Experiment Integrity Audit

## Decision

No local artifact is eligible for the ICASSP MultiRep main table. This is an
evidence-readiness failure, not evidence that a metric was fabricated. The
tracked repository contains a single-person PAMS reproduction and UCFRep dev84
diagnostics; it contains neither the proposed multi-person router/overlap-add
implementation nor a MultiRep result bundle.

The UCFRep test105 split is **unscored**: there are no predictions, evaluation,
or metrics. An older development CLI did deserialize the full manifest, so the
stronger historical word “untouched” is not admissible.

## A--F findings

| Check | Verdict | Finding |
|---|---|---|
| A. Ground-truth provenance | WARN | The formal dev84 path pins the official UCFRep annotation URL/digest and reads counts from temporal boundaries. The core evaluator still accepts an arbitrary target mapping, and some historical/exploratory artifacts have weaker provenance. |
| B. Score normalization | PASS | NMAE divides absolute error by the dataset target; MAE/RMSE remain raw count-unit errors. No reported metric normalizes by predictions. |
| C. Result existence/consistency | WARN | All 17 checked-in evaluation JSONs recompute to saved values (maximum floating discrepancy about 4.4e-16), and 47 focused tests pass. Some historical raw sources are unpublished, and no ICASSP results-manifest instance exists. |
| D. Dead metrics | PASS | NMAE, MAE, RMSE, OBO, and exact accuracy are all called and emitted. An unused compatibility alias does not make a reported metric dead. |
| E. Dataset/seed/scope | WARN | Available evidence is single-person UCFRep dev84 with one or three seeds. No MultiRep protocol/result and no test105 result exist. |
| F. Evaluation type | FAIL | Real ground truth exists only for single-person UCFRep dev84. Target-free and stress artifacts are proxy/simulation-only; the proposed multi-person system has no real-GT evaluation. |

## Implementation boundary

Tracked `PoseSequence` has shape `[T,33,3]`, pose extraction collapses candidates
to one dominant/longest subject, and `CountResult` is scalar. Training has no
person axis. The existing three “experts” are fixed smoothing/peak configurations
combined by deterministic voting, not a learned local-tempo router. No tracked
identity state, response-level expert fusion, normalized overlap-add, or one
decoder per identity exists.

## Claim impact

- R1 implemented identity-indexed local-tempo counter: **unsupported**
- R2 no person-wise count/period supervision: **needs qualifier** until the
  synchronized multi-person training path is audited
- R3 MultiRep effectiveness/competitiveness: **unsupported**
- R4 abstract/main-table number: **unsupported**
- R5 causal mechanism/robustness result: **unsupported**
- R6 existing UCFRep artifacts in ICASSP tables: **unsupported**

The required route is `pivot`: write a complete method/protocol scaffold with
visible pending controls, but admit no positive empirical claim until a fresh
experiment audit passes on the collaborator freeze.

