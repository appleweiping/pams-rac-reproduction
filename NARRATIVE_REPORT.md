# RAC CVPR 2027 Narrative Report

**Working title:** *Count Each Person at Their Changing Pace: Multi-Person Tempo-Adaptive Repetition Counting*  
**Venue:** CVPR 2027  
**Paper state:** `provisional/data-pending`  
**Evidence freeze:** Git commit `73cbc4bb9b581f24fbc90f2fe7421bc11e7c7454`  
**Template state:** `CVPR2027-template-pending`; the CVPR 2026 author kit is only a provisional formatting base  
**Submission mode:** double blind; author identity is not stored in this repository

## 1. Evidence Policy

Evidence is resolved in this strict priority order:

1. collaborator final code, frozen experiment artifacts, and audited results;
2. tracked code and audit documentation at the evidence-freeze commit;
3. committed PowerPoint concept slides;
4. untracked prose, meeting summaries, and other informal notes.

A lower-priority source may propose a narrative, but it cannot override a higher-priority implementation or result artifact. The collaborator final artifact is not present in this worktree. Every proposed multi-person module therefore remains `SYNC-REQUIRED` and `BLOCKED` as an implementation or empirical claim. No conceptual slide is treated as proof that a module has been implemented or validated.

## 2. What the Current Repository Establishes

The tracked repository is an auditable, single-person reproduction of PAMS rather than an implementation of multi-person repetitive action counting. This boundary is explicit in `README.md`, `docs/METHOD_SPEC.md`, and `docs/ASSUMPTIONS.md`. In particular, the current pose policy is single-person and does not claim cross-person identity tracking.

The repository does establish several ingredients that can inform, but do not implement, the proposed paper:

- `src/pams/model.py` implements a pose encoder and period-head variants for the PAMS reproduction.
- `src/pams/period.py` implements bounded FFT/autocorrelation period estimators for single sequences.
- `docs/METHOD_SPEC.md` describes PAMS-TCC and a fast/medium/slow peak-counting consensus used by the reproduction.
- `src/pams/evaluation.py` separates label-free prediction from the evaluator boundary where counts are introduced.
- `src/pams/metrics.py` implements single-instance counting metrics and paired bootstrap utilities.

These ingredients are evidence about the starting codebase only. They are not evidence for identity-conditioned multi-person routing, window-level tempo adaptation, track continuity, normalized overlap-add across person windows, or MultiRep performance.

The repository also records material negative evidence. `results/README.md`, `results/dev-negative/README.md`, and `results/dev-negative/summary.json` state that the available UCFRep dev84 runs are development-only negative results or diagnostics. PAMS-Literal uses the random, untrained Period Head implied by an unresolved paper specification; PAMS-SSHead is an independently inferred repair. Proxy, component-gate, source-sanity, and development-only rows cannot populate the proposed method's main table. The sealed UCFRep test105 split remains untouched. Every current result row is therefore `table_eligible=false` for this paper.

## 3. Conceptual Sources and Their Limit

Two committed slides define the intended visual story:

- `recommend paper/Multi-Person-Tempo-Adaptive-Counting-Introduction-Rounded-TNR.pptx` contrasts a mixed single stream and one global count with person-wise temporal responses and a per-person count vector.
- `recommend paper/Multi-Person-Tempo-Adaptive-Counting-Framework-Rounded-TNR.pptx` sketches detection and identity tracking, person-specific processing, tempo estimation, a fast/medium/slow router, gated fusion, boundary stitching, and per-person aggregation.

Both files are `visual_reference_only`. They may guide editable vector figures and the provisional mathematical contract, but they do not establish implementation, training, correctness, speed, or accuracy.

## 4. One-Sentence Technical Story

We study pose-driven multi-person repetition counting without per-person count supervision and propose, pending synchronization with the collaborator implementation, to maintain identity-specific temporal state while sharing an encoder and tempo-specialized experts, route local windows by each person's changing tempo, fuse overlapping responses, and extract each person's count once from the reconstructed trajectory.

This is the only technical story for the first paper draft. It is intentionally narrower than claiming the first multi-person, asynchronous, variable-tempo, pose-based, self-supervised, or end-to-end counter.

## 5. Provisional Task Contract

For person identity (i), the input is a pose trajectory (X_i\in\mathbb{R}^{T_i\times J\times C}) with validity mask (m_i\in\{0,1\}^{T_i}). The primary output is the per-person count vector

\[
\hat{\mathbf c}=[\hat c_1,\ldots,\hat c_N]^{\mathsf T}
\]

The scalar sum \(\sum_i \hat c_i\) is a derived diagnostic, not the primary task definition. Tracking observations, track identities, missing-frame masks, and oracle-versus-predicted-track protocols must be declared explicitly in the experiment manifest.

This formulation is `PROVISIONAL`: the collaborator's final code and data interface take precedence. Any mismatch must trigger a paper rewrite and an entry in the Chinese synchronization report.

## 6. Provisional Method Contract

The first mathematical draft uses the following ordered mechanism. Every item is `SYNC-REQUIRED` until bound to collaborator code.

1. **Person-wise representation.** A shared pose encoder processes all identities, while masks and temporal state remain identity-specific.
2. **Local tempo estimation.** Overlapping windows within each identity trajectory produce a local period or tempo descriptor rather than one clip-level period.
3. **Soft tempo routing.** A differentiable router assigns each window to shared fast, medium, and slow experts using soft weights.
4. **Response-level expert fusion.** Experts produce local counting responses, not independently finalized window counts.
5. **Boundary reconstruction.** Normalized overlap-add fuses window responses into one response trajectory per identity.
6. **Single extraction.** Each reconstructed identity trajectory undergoes one final peak-extraction pass so an overlap event is not independently counted twice.
7. **Training objective.** The prose may state only a provisional combination of PAMS-TCC, routing consistency, and track continuity. The latter two definitions and all weights remain explicit placeholders until code synchronization; the paper must not present them as implemented facts.

The default architecture is shared across people; it is not a bank of separately trained person-specific models. The phrase “person-specific” in a concept slide means identity-conditioned state or response, not separate learned parameters, unless the collaborator implementation proves otherwise.

## 7. Claim Boundaries

Allowed before result synchronization:

- task motivation and a clearly labeled proposed formulation;
- equations that define the provisional mechanism;
- statements about what the tracked PAMS reproduction code and audit documents contain;
- literature positioning verified against primary publication sources;
- an explicit statement that results are pending.

Blocked before synchronization and audit:

- any statement that the current repository implements MRAC;
- any performance, robustness, generalization, efficiency, or state-of-the-art claim;
- any numerical result for the proposed method;
- claims of first multi-person, first person-wise, first asynchronous, first variable-tempo, first pose-driven, first self-supervised, annotation-free, end-to-end, or constant-time operation;
- claims that routing consistency or track continuity has a finalized formula or validated weight;
- claims that conceptual figures report executed behavior.

## 8. Paper Structure and Evidence Role

- **Abstract:** problem, challenge, proposed approach, mechanisms, and exactly one controlled result placeholder. No result number is permitted in the pre-result draft.
- **Introduction:** application need, prior progress, the gap between global/mixed temporal processing and identity-conditioned local-tempo reasoning, proposed mechanism, and three falsifiable contributions whose evidence status is visible in the ledger.
- **Related Work:** single-instance RAC and tempo modeling; pose and low-label RAC; multi-person RAC. The organization must synthesize method families and align with experimental baselines.
- **Method:** formulation, overall architecture, person-wise representation, local tempo routing, expert fusion, boundary/trajectory reconstruction, and training objective.
- **Experiments:** protocol contract and empty evidence-bound tables only. Every cell must be generated from a frozen manifest after audit.
- **Conclusion and limitations:** restate only the proposed scope and make the missing implementation/results and tracking dependence explicit in the pre-result version.

## 9. Required Result Synchronization

The collaborator handoff must supply a uniform artifact manifest containing at least:

- exact implementation commit and clean/dirty state;
- resolved configuration and its checksum;
- all random seeds and hardware details;
- MultiRep release, split identity, and dataset checksums;
- oracle-track and predicted-track protocol labels;
- per-instance ground truth and per-instance predictions;
- raw run logs and evaluation logs;
- Period-mAP, AP50, AP75, AvgMAE, and AvgOBO;
- HOTA, IDF1, and IDSW for the tracking pipeline;
- mean, standard deviation, and paired bootstrap 95% confidence intervals over at least three seeds.

Only frozen JSON, CSV, or Parquet artifacts may generate tables and figures. `experiment-audit` must first determine result integrity and scope; `result-to-claim` must then adjudicate claim support. Abstract values, comparative superlatives, and performance contributions stay withheld until both gates are complete.

## 10. Planned Experimental Evidence

The required evidence groups are main comparison, additive module ablation, local-versus-global tempo, soft-versus-hard-versus-uniform routing, expert count, boundary and continuity mechanisms, shared-versus-person-specific weights, occlusion and ID switches, within-track tempo changes, cross-dataset generalization, and scaling in people, runtime, memory, and FLOPs.

The baseline set is MultiCounter and MultiCounter+, plus RepNet, TransRAC, PoseRAC, and PAMS under the same tracking frontend. DeTRC and D²-STX enter only if they can be reproduced under a comparable protocol. Fixed predicted tracks imply that HOTA, IDF1, and IDSW are frontend diagnostics and do not vary across downstream counters.

This section is a preregistered evidence plan, not a report of completed experiments.

## 11. Current Assurance Outcome

The writing workflow runs at `assurance=submission`, so mandatory audits must emit artifacts even when their verdict is `NOT_APPLICABLE` or `BLOCKED`. However, the paper itself remains `provisional/data-pending`. It cannot be called submission-ready while the collaborator implementation, MultiRep artifacts, controlled abstract result sentence, and CVPR 2027 official template are absent. Same-family review can at most confer provisional assurance; the currently unhealthy cross-family review overlay cannot be represented as accepted coverage.
