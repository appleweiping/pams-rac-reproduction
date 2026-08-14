# Experiment Audit

**Date:** 2026-08-12  
**Auditor:** fresh `gpt-5.6-sol` ultra reviewer  
**Independence:** same-family; provisional  
**Verdict:** **FAIL** (`NO_MULTI_PERSON_REAL_GT_EVIDENCE`)

No current result can support an empirical claim for the proposed
multi-person paper. All real-ground-truth measurements are single-person
UCFRep dev84 development diagnostics; no identity-aware inference path,
person-wise target, MultiRep run, or held-out result exists. UCFRep test105 was
not accessed and remains sealed.

## Audit checks

| Axis | Status | Evidence-backed conclusion |
|---|---|---|
| Ground-truth provenance | WARN | Dev84 targets enter only at evaluation, but some historical runs deserialized a full manifest and lack per-video publication artifacts. |
| Score normalization | PASS | NMAE uses ground-truth counts; OBO uses deterministic half-up rounding. |
| Result existence | WARN | Development summaries exist; some raw predictions/logs/checkpoints do not. No test105 result exists. |
| Executing-code linkage | WARN | Current prediction/evaluation/metric linkage is visible; complete historical generator lineage is not. |
| Scope | FAIL | There is no multi-person benchmark, annotation, inference path, or evaluation. |
| Evaluation type | FAIL | Multi-person real-GT evaluation is absent; other artifacts are proxy, synthetic, smoke, or diagnostic. |

## Eligible uses

- Component, smoke, safety, and protocol checks.
- UCFRep dev84 negative pilots, only when labeled exploratory.
- Predeclared negative internal ablations for hypothesis generation.
- Evidence that current candidates failed their development gates.

## Ineligible uses

- Every multi-person effectiveness, robustness, generalization, or performance
  comparison.
- Any held-out/test/leaderboard or state-of-the-art statement.
- Every core quantitative paper table.
- Source-faithful PAMS claims built from inferred repairs.
- Dev-tuned, synthetic, proxy, sanity-baseline, or unverifiable PoseRAC values
  presented as fair benchmark results.

## Required evidence to change the verdict

1. A frozen multi-person task and annotation protocol.
2. An identity-aware implementation tied to a commit and configuration.
3. An independently sealed train/dev/test protocol and official evaluator.
4. Per-person raw predictions and ground truth with required counting metrics.
5. At least three registered seeds and paired uncertainty analysis.
6. Fair baselines and a complete reproducibility bundle.

The raw reviewer trace, exact file/line evidence, and detailed fatal-gap list
are preserved at
`.aris/traces/experiment-audit/2026-08-12_run01/reviewer.md`.
