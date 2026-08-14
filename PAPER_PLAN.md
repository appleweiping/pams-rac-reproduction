---
plan_version: 1
assurance: submission
paper_state: provisional/data-pending
venue: CVPR 2027
template_state: CVPR2027-template-pending
evidence_freeze: 73cbc4bb9b581f24fbc90f2fe7421bc11e7c7454
result_to_claim_route: pivot
---

# Paper Plan

## Core Story and Claim Ceiling

**Working title:** *Count Each Person at Their Changing Pace: Multi-Person
Tempo-Adaptive Repetition Counting*

**One-sentence story.** We study pose-driven multi-person repetition counting
whose intended training uses no person-wise count or period labels, and define
an identity-conditioned, window-local tempo-routing counter that shares model
parameters across people while reconstructing and decoding one response per
identity.

This plan is governed by `CLAIMS_FROM_RESULTS.md`: the result-to-claim route is
`pivot`, experiment integrity is `FAIL — NO_MULTI_PERSON_REAL_GT_EVIDENCE`, and
no positive empirical claim is admissible. The first deliverable is therefore
a complete pre-results method and evaluation contract, not a result-bearing
submission. All proposed multi-person components are `SYNC-REQUIRED` and every
performance field fails closed in submission mode.

## Claims–Evidence Matrix

The authoritative row-level ledger is `CLAIMS_EVIDENCE_MATRIX.md`. The paper
uses the following headline groupings:

| Headline group | Ledger rows | Current evidence | Admissible wording |
|---|---|---|---|
| Task and supervision setting | C01–C02 | Literature + intended method contract; final training graph absent | “We study” and “intended counting objective”; never stronger than no person-wise count/period supervision |
| Identity-conditioned local-tempo mechanism | C03–C06 | Concept slides + provisional equations; implementation absent | “We propose” with an explicit synchronization boundary |
| Boundary-safe person-wise output | C05 | Mathematical contract only | Normalized overlap-add and one final decoder as design, not validated behavior |
| Evaluation and robustness protocol | C11–C15 | Preregistered schema and table shells; results absent | “We establish an evaluation contract”; no comparative conclusion |

Existing UCFRep dev84 negative/proxy artifacts are excluded from every
new-method table. The sealed UCFRep test105 split remains untouched.

## Main-Paper Budget

The provisional CVPR 2026 author kit is used only until the CVPR 2027 kit is
released. The target main-body allocation is eight pages, excluding references
and any permitted appendix:

| Material | Budget | Purpose |
|---|---:|---|
| Title + abstract | 0.35 page | What, difficulty, design, strongest audited result placeholder |
| Introduction | 1.15 pages | Application, progress, precise gap, approach, three falsifiable contributions |
| Related work | 0.90 page | Three method families aligned with baselines |
| Method | 2.40 pages | Task, identity state, local tempo, routing, fusion, boundary handling, objective |
| Experiments | 2.70 pages | Setup, main comparison, ablations, stress/generalization/scaling analyses |
| Conclusion + limitations | 0.50 page | Scope, limitations, concrete closure work |

The pre-results build may be shorter because result interpretations and plots
are intentionally withheld. It must not be padded with unsupported prose.

## Section Plan

### 1. Abstract

- **What:** person-wise repetition counting from identity-indexed pose tracks.
- **Why difficult:** simultaneous people are asynchronous; each track may
  change pace, pause, disappear, or fragment.
- **How:** shared encoder and fast/medium/slow experts; per-identity state;
  masked local period evidence; soft routing; normalized overlap-add; one
  track-level decoding pass.
- **Evidence:** exactly one `\ControlledAbstractResult` invocation. In the
  pre-results build it renders a visible pending sentence. Submission mode
  must fail unless exactly one audited result sentence is declared.
- **Guardrails:** no citation, undefined acronym, primacy, SOTA, or fabricated
  value; state the evidence-gated status plainly.

### 2. Introduction

Paragraph sequence:

1. Application motivation and why a scene total is insufficient.
2. Early-to-recent RAC progress, then MultiCounter/MultiCounter+ as established
   MRAC prior art.
3. Narrow gap: supervision × pose modality × within-track tempo granularity.
4. Proposed approach and causal ordering of routing, response fusion, and
   track-level decoding.
5. Oracle/predicted-track boundary and exact no-count-label qualifier.
6. Task illustration and exactly three falsifiable contributions.
7. Explicit pre-results status.

Contribution mapping:

1. Setting and supervision boundary → C01–C02.
2. Local routing and boundary reconstruction → C03–C06.
3. Evaluation contract → C10, C12, and C15.

### 3. Related Work

Synthesize, rather than enumerate, three families:

1. **Single-instance RAC and tempo modeling:** local cycle/frequency methods,
   RepNet, TransRAC, DeTRC, ESCounts, and SkimFocusNet. Distinguish global or
   single-primary-action assumptions from person-wise MRAC.
2. **Pose-based and low-count-label RAC:** PoseRAC, D$^2$-STX, and PAMS.
   Distinguish zero-shot, multimodal, and self-supervised objectives. Pose and
   period adaptation are foundations, not novelty claims.
3. **Multi-instance RAC:** MultiCounter and MultiCounter+. Acknowledge their
   person-wise, asynchronous, variable-speed, pause, and identity-continuity
   coverage; isolate the proposed no-count-label pose/local-window axis.

Every family maps to at least one planned baseline or protocol distinction.

### 4. Method

Ordered subsections and required objects:

1. **Problem formulation and overview:**
   (X_i\in\mathbb{R}^{T_i\times J\times C}), validity mask (m_i), primary
   output (\hat{\mathbf c}), optional diagnostic sum, perception boundary.
2. **Identity-conditioned local encoding:** shared (E_\theta), overlapping
   windows, masks and state indexed by identity.
3. **Local period evidence and soft routing:** masked non-circular
   autocorrelation, spectrum, (\hat p_{i\ell}), confidence, and simplex gate
   over fast/medium/slow shared experts.
4. **Expert fusion and boundary handling:** response-level mixture, normalized
   overlap-add, one final track decoder; never sum rounded window counts.
5. **Training objective and synchronization boundary:** PAMS-TCC plus reserved
   routing-consistency and track-continuity terms. Their equations, sample
   construction, and weights remain unspecified until code synchronization.

Display equations have no trailing punctuation. Each symbol is introduced
before or immediately after first use. No subsection may imply that a
`SYNC-REQUIRED` component is implemented or validated.

### 5. Experiments

The pre-results version fixes questions and schemas without measurements.

- **Setup:** exact MultiRep release/split/checksums; evaluator revision;
  preprocessing fingerprint; hardware; seeds; modalities; supervision; and
  oracle versus predicted tracks.
- **Metrics:** Period-mAP, AP50, AP75, AvgMAE, AvgOBO; HOTA, IDF1, and IDSW as
  tracking diagnostics; mean, sample SD, and paired video-cluster bootstrap
  95% confidence intervals over at least three seeds.
- **Main comparison:** native MultiCounter/MultiCounter+ in a contextual block;
  common-track RepNet, TransRAC, PoseRAC, PAMS, and ours; DeTRC/D$^2$-STX only
  after reproducibility and output-semantics audit.
- **Ablations:** track-conditioned base → local tempo → soft routing → boundary
  reconstruction → continuity; local/global tempo; soft/hard/uniform/oracle
  routing; expert count; shared/person-specific weights; boundary and
  continuity removals.
- **Analyses:** tempo-change strata, pauses, occlusion, ID switches,
  oracle/predicted gap, cross-dataset transfer, and runtime/memory/FLOPs/FPS
  versus active-person count.

Every subsection carries a machine-readable `\ClaimMap{...}` marker using the
authoritative map in `paper/evidence/story_contract.json`. Tables and figures
are generated only from frozen JSON, CSV, or Parquet artifacts after
`experiment-audit → result-to-claim` passes.

### 6. Conclusion and Limitations

Restate the intended scope, then disclose three limitations: missing
multi-person synchronization, absent eligible MultiRep evidence, and
dependence on upstream pose/tracking. Future work is concrete: bind every
module to code, quantify oracle/predicted gaps and corruptions, and measure
population scaling. No performance conclusion appears while data are pending.

## Figure Plan

| ID | Asset | Intended message | Evidence boundary | Status |
|---|---|---|---|---|
| F1 | `paper/figures/task_comparison.drawio` + SVG/PDF | Mixed scene response can obscure person-wise changing pace; target output is a count vector | Problem illustration only; no executed outputs | Complete and visually checked |
| F2 | `paper/figures/framework.drawio` + SVG/PDF | Perception boundary, shared parameters, per-identity local routing, overlap-add, final decoding | Dashed provisional elements are synchronization-required | Complete and visually checked |
| F3 | Generated main-result/uncertainty visualization | Same-protocol accuracy and uncertainty | Requires frozen MultiRep artifacts | DATA-PENDING |
| F4 | Generated tempo/tracking/scaling analysis | Failure modes and scaling with identities | Requires per-instance stress/runtime records | DATA-PENDING |

The committed PowerPoint files are `visual_reference_only`. They are not
included as figures and do not establish implementation evidence.

## Citation Plan

- Discover candidates through publisher indexes, DBLP, Crossref, and manual
  cited-by expansion; Google Scholar may discover candidates but is not the
  metadata authority.
- Prefer final published versions. In particular use the ICONIP proceedings
  version of PoseRAC and the IJCV version of DeTRC.
- Verify existence, canonical metadata, and local claim support for every
  cited key. Keep only entries cited in the manuscript.
- Required anchors: MultiCounter, MultiCounter+, and PAMS for novelty limits;
  RepNet/TransRAC/PoseRAC/PAMS plus conditional DeTRC/D$^2$-STX for planned
  baselines; HOTA and ID metrics for tracking diagnostics.
- Any `REPLACE` or `REMOVE` recommendation requires user approval under ARIS.
  Submission mode forbids `[VERIFY]` and undefined citations.

## Build, Review, and Acceptance Order

1. Compile a visible pre-results draft and save `main_round0_original.pdf`.
2. Negotiate `PAPER_ACCEPTANCE_CONTRACT.md` with a fresh reviewer, up to three
   rounds; unresolved disagreement sets `contested` and preserves the
   submission block.
3. Run two fresh zero-context improvement rounds, recompiling and saving
   `main_round1.pdf` and `main_round2.pdf`.
4. Emit proof, numeric-claim, citation, and kill-argument audit artifacts,
   including machine-readable JSON even when `NOT_APPLICABLE` or `BLOCKED`.
5. Re-run the audit verifier and submission validator. Missing/stale/blocking
   evidence, source placeholders, anonymity leaks, or template drift fail
   closed.

## Final Readiness Conditions

The paper cannot exceed `provisional/data-pending` until all AC01–AC16 items
close, collaborator artifacts pass both empirical gates, the CVPR 2027
official kit replaces the provisional template, and a healthy cross-family
Claude review is recorded. A Codex-only review remains same-family and
provisional even when deterministic checks pass.
