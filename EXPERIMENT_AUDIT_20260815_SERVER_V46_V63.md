# Experiment Audit: Server v46, v62, and v63

**Date:** 2026-08-15

**Auditor:** fresh GPT-5.6-Sol ultra agent

**Independence:** same-family

**Acceptance status:** provisional
**Overall verdict:** **FAIL**

## Scope

This audit examined the hash-verified quarantine copy at
`D:/Project/rac-server-quarantine-20260815`, the current ICASSP evidence
contracts, and the relationship between the server candidates and the proposed
TempoRAC manuscript. The audit was read-only. It did not execute experiments,
access sealed data, or modify the server.

## Decision

Neither the numerically largest server version nor the most complete result
bundle is a canonical implementation of the proposed method.

- **v63 is ineligible:** it is a 37-line PoseRAC channel-selection classifier.
  It has no matching result, model, configuration, log, environment, input
  manifest, or completion receipt. Its best-channel training label is derived
  by selecting the minimum channel error against training ground truth.
- **v62 is ineligible:** it is a single-seed development result with incomplete
  provenance and is not a multi-person TempoRAC run.
- **v46 is ineligible as TempoRAC:** it uses motion/spectral features, an
  ExtraTrees count expert, integer calibration, and optional within-video
  interaction. It directly trains on per-person count labels and has no
  identity-indexed local-tempo router, shared fast/medium/slow response experts,
  normalized overlap-add reconstruction, or one-decode-per-track path.

v46 may be retained only as a disclosed historical/development baseline. v63
may be retained only as an exploratory proxy after a complete run and evidence
bundle exist.

## A. Ground-truth provenance: FAIL

- The v46 loader carries `density_gt` and `count_gt`, and the classical expert
  trains on `count_gt` (`run_mpcounter_v44.py:303-328` and
  `mpcounter_classical_experts_v44.py:407-427`).
- Every pose track must map to an explicit GT object ID; duplicate mappings are
  rejected (`run_mpcounter_v44.py:248-280`). Both binary test inputs declare
  GT-bbox-assisted AlphaPose association. The results are therefore oracle-like
  GT-assisted track results, not common predicted-track results.
- The 71-video population combines 53 original validation videos and 18
  original test videos. Its conversion manifest states that selection did read
  count or period information
  (`converted_unseen_test/conversion_manifest_v45.json:46-55`).
- The 180-video manifest records 27 train-test and 29 validation-test canonical
  source overlaps (`official_complete_manifest.json:9-71`), although its result
  summary labels the run source-disjoint.
- The 180-video script waits for the 71-video run and copies that already
  evaluated model (`finish_official_complete_v46.sh:15-17,47-50`), so 18
  official-test videos had already been exposed.
- Evaluation uses a custom local function rather than a frozen official
  evaluator (`run_mpcounter_v44.py:753-781`).

Existing v46 numbers cannot support official MultiRep performance, unseen
generalization, or predicted-track claims.

## B. Score normalization: WARN

- v46 `avg_mae` is mean per-person relative absolute error with ground truth in
  the denominator, not a prediction-derived denominator. Raw absolute MAE and
  per-person prediction/target rows are present, and the saved count summaries
  recompute exactly.
- The metric name is ambiguous: `avg_mae` is normalized, while raw
  `person_abs_mae` is also stored.
- Scale, offset, rounding, piecewise offset, and within-video interaction are
  selected on the same out-of-fold targets
  (`train_mpcounter_spectral_v46.py:186-286`), creating selection optimism.
- The unused period proposal path normalizes each track's scores by that track's
  maximum (`run_mpcounter_v44.py:1075-1076`).
- v63 computes normalized absolute error without zero-target protection and
  reports clipped/rounded outputs only.

## C. Result existence and binding: FAIL

Positive checks:

- Both v46 result directories contain completion receipts and test-start
  markers.
- Code, selection, model, frozen configuration, and prediction hashes recorded
  by the v46 receipts match the quarantine copy.
- Independent recomputation reproduces the saved count summaries:
  - 71 videos / 194 persons: AvgMAE 0.191283, AvgOBO 0.639175, raw person MAE
    1.902062
  - 180 videos / 464 persons: AvgMAE 0.203916, AvgOBO 0.459052, raw person MAE
    4.086207

Blocking checks:

- The 71-video source-selection file referenced by the conversion manifest is
  missing.
- The v46 OOF search log stops during enumeration; its expected search JSON and
  OOF prediction archive do not exist.
- Required historical checkpoints and the referenced spectral artifact are not
  present, and the finisher scripts start from pre-existing artifacts rather
  than reproducing training.
- v63 has only source code and no run artifacts.
- The paper result manifest has no eligible commit, data release, evaluator,
  seed set, configuration, hardware, logs, predictions, metrics, or claim
  bindings.

## D. Dead code: FAIL

- Period-mAP, AP50, and AP75 are implemented in
  `period_metrics_density` (`run_mpcounter_v44.py:1079-1156`) but have no audited
  call site.
- The paired source-cluster bootstrap is defined
  (`run_mpcounter_v44.py:1174-1320`) but never called.
- v46 testing calls only `metrics_with_unmatched`
  (`mpcounter_full_system_v46.py:375-378`) even though the frozen configuration
  says `density_period_mode=true`.
- Source-hash verification is defined but is not called by split loading.

The proposed primary period metrics, uncertainty intervals, provenance checks,
and runtime evidence are code-shaped promises rather than completed evidence.

## E. Scope: FAIL

- The 71-video result covers 194 GT tracks and only 25 canonical source IDs;
  the audit reconstructs five source-connected test components.
- The 180-video result covers 464 tracks and 121 source IDs, but overlaps
  training and validation sources.
- v46's three seeds form one ensemble. They are not three independent full
  training runs and provide no reported sample standard deviation or paired
  cluster-bootstrap interval.
- There are no completed matched baselines, TempoRAC ablations, local-tempo
  diagnostic, predicted-track evaluation, HOTA/IDF1/IDSW, or person-count
  scaling measurements.

## F. Evaluation classification: FAIL

| Artifact | Classification | Qualification |
|---|---|---|
| v46 training OOF count metrics | real GT | Calibration/model selection reuse OOF predictions |
| v46 71-video result | real GT, oracle/GT-assisted | Label-informed population selection; not an independent test |
| v46 180-video result | real GT, oracle/GT-assisted | Custom evaluator, source overlap, and prior test exposure |
| v46 period metrics | no completed evaluation | Functions exist without output artifacts |
| v63 best-channel labels | synthetic proxy | Labels are selected using channel errors against train GT |
| v63 development metrics | intended real GT, unverified | No completed run bundle exists |
| Planned piecewise time-warp study | simulation only | Prospectively described and unexecuted |

## Claim impact

- **Unsupported:** any claim that v46/v62/v63 implements or validates TempoRAC
- **Unsupported:** official MultiRep, source-disjoint, blind-test,
  predicted-track, Period-mAP/AP50/AP75, multi-seed uncertainty, or SOTA claims
- **Permitted with explicit qualification:** v46 is an internally reproducible
  historical GT-assisted count-only diagnostic under a compromised protocol
- **Permitted:** the current paper's statement that no result is yet eligible

## Required remediation

1. Freeze a leakage-free MultiRep release/split/checksum and a frozen official
   evaluator before model selection or evaluation
2. Implement the selected canonical method in a clean Git commit and bind every
   claimed component to source and tests
3. Remove GT object identities from predicted/common-track inference; report
   oracle/GT-assisted and predicted-track protocols separately
4. Run three independent seeds with complete configuration, environment,
   hardware, log, prediction, checkpoint, and receipt bundles
5. Execute the official period metrics and source-cluster bootstrap rather than
   merely defining them
6. Run protocol-matched baselines, mechanism ablations, non-stationary tempo
   diagnostics, and tracking/efficiency analyses
7. Re-run experiment-audit and result-to-claim before licensing any manuscript
   number

## Audited input hashes

```text
ec6b8ee58dd163b86be49f3e7a3461b449c65000184ad498b367e085e71261c6  v46/code/run_mpcounter_v44.py
8941a8519a8ecc98a0a39384a27cbdafd9e31a59c787e535448190dd6745dc9f  v46/code/mpcounter_full_system_v46.py
9ce0f22b301c8a23ee403e93a6a4387fb82f181bae45619dc18e5ef3e9555ee2  v46/code/train_mpcounter_spectral_v46.py
d8b4ad114ef695079e10c599be6eb6f12998a7ff07447e2c0df09066b400b0f0  v46/code/finish_official_complete_v46.sh
307beb634a4075ff5e769373d8c9d48fcca885dfbc4e389738c9cec52391722c  v46/full_system/test_summary_v46.json
17868b5d8793819e5024f3f376f40dcadeddb2faf1cbbf58519b273ad20f35f2  v46/official_complete/test_summary_v46.json
217beaca14ea124790cf50a04af99139d5d457773b0c04057e4d9781bc5e4813  v46/converted_unseen_test/conversion_manifest_v45.json
bb9accf2669f1bd3a868352dca51c1516f5e54920d88527ef311d8b273f4255a  v46/data-manifests/real_disjoint_manifest.json
468c568616131fa3a40da21822baac8e769074ada4ffbcecd3203a8d757e2c0b  v46/data-manifests/official_complete_manifest.json
9989d776ff78bf9bee55cda0164b1fbf5fb9e469a6894925639c31bfe443bcb1  v63/train_poserac_channel_classifier_v63.py
4df513a3f4427950d23267965192253446e6ee2b6968907855e0bae1a6a53159  paper/evidence/results_manifest.json
514d25ceb51d97930f02a14c70d96962ff7acdcb28c712fa80c1f03a9ea1d579  paper/evidence/method_manifest.yaml
```

Trace: `.aris/traces/experiment-audit/20260815_server_v46_v63/`
