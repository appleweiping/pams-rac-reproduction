# Fresh zero-context experiment audit response

Reviewer task: `/root/fresh_experiment_audit`  
Reviewer model: `gpt-5.6-sol` (`ultra`)  
Context policy: raw paths only; no executor summary

## Verdict: FAIL

No current result is eligible as confirmatory evidence for a new multi-person
repetition-counting paper. The repository contains useful single-person
development diagnostics, component checks, and negative ablations, but no
multi-person dataset, model path, target definition, evaluation, baseline
comparison, or held-out result. UCFRep test105 remains sealed and was not
accessed during this audit.

Review independence is same-family; under ARIS, this finding is provisional
pending independent-family review.

### Checks

| Check | Status | Finding |
|---|---|---|
| Ground-truth provenance | WARN | Real dev84 scoring uses documented UCFRep annotations at the evaluator boundary, while synthetic/proxy results are labeled. Some historical runs deserialized the complete manifest, including test records, and some aggregates omit publishable per-video rows. |
| Score normalization | PASS | NMAE divides error by ground-truth count; OBO uses deterministic half-up rounding. No model-output/self-score normalization is used for accuracy. |
| Result existence | WARN | The reported dev summaries and metric keys exist, but several server-produced aggregates lack raw predictions/logs/checkpoints required for independent reproduction. No test105 result exists. |
| Executing-code linkage | WARN | Prediction-to-evaluator-to-metric linkage is visible, but the generators/CLI lineage for all historical aggregates is outside the allowed sources, and some artifact-producing revisions differ from current code. |
| Scope adequacy | FAIL | All real-GT results are single-person UCFRep dev84 diagnostics. There is no multi-person benchmark, per-person annotation, person-stratified analysis, or multi-person inference path. |
| Evaluation type | FAIL | Real-GT evaluation exists only for single-person development data. Multi-person real-GT evaluation is absent; the remaining evidence is synthetic, self-supervised/proxy, smoke, or diagnostic. |

### Exact evidence

- UCFRep-526 is documented as 421 official-train / 105 official-test, with the
  421 partitioned into train337/dev84 using seed 2026:
  `docs/ASSUMPTIONS.md:11-12`, `REPRODUCIBILITY.md:5-8`.
- Registered seeds are 42, 2026, and 3407:
  `docs/ASSUMPTIONS.md:41`, `REPRODUCIBILITY.md:5-8`.
- The canonical strict three-seed result is dev84-only, failed, and leaves
  test105 untouched:
  `results/dev-negative/pams_full_multiscale_three_seed_strict.json:3-16`,
  `:25-40`, `:65-78`.
- Literal/random-head NMAE is 1.489633 and OBO is 0.226190; inferred SSHead
  NMAE is 5.022471 and OBO is 0.051587, with 0/3 seeds passing:
  `results/README.md:179-189`.
- The broader Table-2-shaped experiment has 0 verified rows and is explicitly
  paper-table-ineligible:
  `results/dev-negative/pams_table2_three_seed_strict.json:45-51`, `:975-986`.
- No test prediction, evaluation, or metric is present:
  `results/README.md:12-26`.
- Metrics use the ground-truth target and deterministic rounding:
  `src/pams/metrics.py:16-29`, `:39-69`, `:230-309`. Targets enter only at
  evaluation: `src/pams/evaluation.py:60-86`, `:89-127`.
- Historical full-manifest deserialization prevents treating those old runs as
  fully sealed even though no test scoring is reported:
  `results/dev-negative/README.md:705-709`.
- Fair baseline slots are blocked/unimplemented and the sealed-test allowlist
  is empty: `docs/BASELINE_PROTOCOLS.md:10-31`; the result inventory has no
  fair metric table: `results/README.md:292-319`.
- Required reproducibility artifacts include predictions, logs, metrics,
  checkpoints, receipts, and environment/hardware provenance:
  `REPRODUCIBILITY.md:129-143`. Some aggregate sources are represented only by
  digests: `results/dev-negative/README.md:698-703`.

### Implementation versus concept

Implemented evidence supports one pose tensor shaped `[frames, 33, 3]`
(`src/pams/types.py:52-98`), selection of one dominant person candidate per
frame (`src/pams/pose.py:177-245`, `:248-345`), a single-stream encoder/head
and scalar `CountResult` (`src/pams/model.py:100-180`,
`src/pams/types.py:126-170`), and label-free prediction followed by separate
evaluation (`src/pams/evaluation.py:60-127`).

Absent or conceptual are multi-person detections, persistent identity
tracking/reassociation, person-set representations, per-person count outputs,
group-count semantics, and multi-person metrics. The repository itself
disclaims cross-person identity tracking (`docs/METHOD_SPEC.md:19-27`,
`docs/ASSUMPTIONS.md:15`). The original Period Head objective is missing; the
literal head is random and SSHead is an inferred repair
(`docs/PAPER_AUDIT.md:37-48`, `src/pams/training.py:1828-1831`,
`:2196-2209`). Documentation/code also disagree on the default preprocessing
revision (`docs/METHOD_SPEC.md:19-27`, `DATA.md:95-103`,
`src/pams/pose.py:50-61`).

### Eligibility boundary

Only component/smoke/safety checks, protocol audits, explicitly exploratory
single-person dev84 diagnostics, and predeclared negative internal ablations
are eligible as ancillary material. Multi-person effectiveness,
generalization, robustness, held-out, leaderboard, core-table, source-faithful
PAMS-from-inferred-repair, dev-tuned, synthetic/proxy benchmark, fair-baseline,
and PoseRAC source-faithful claims are ineligible.

### Fatal gaps

1. No multi-person data, annotations, split, or real-GT evaluation.
2. The implementation selects one person and lacks identity tracking.
3. The target is one video-level scalar; per-person/group semantics are absent.
4. No UCFRep test105 result exists.
5. Registered three-seed methods fail the dev gate at 0/3 seeds.
6. The original Period Head objective is undisclosed.
7. No completed fair baseline table exists.
8. PoseRAC's exact protocol and annotations are unavailable.
9. Several aggregates lack an independently reproducible artifact bundle.
10. Pose preprocessing is not consistently frozen across documentation, code,
    and historical results.
