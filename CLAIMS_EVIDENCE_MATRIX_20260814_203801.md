---
ledger_version: 2
created_at: 2026-08-14T20:38:01+08:00
venue: ICASSP 2027 regular
paper_state: provisional/data-pending
evidence_freeze: 26e5b7176f1f7c678391163c3278b5c390d54115
---

# ICASSP 2027 Claims–Evidence Matrix

## Status and Eligibility Rules

- `CONFIRMED`: the named tracked artifact directly supports the statement.
- `PROVISIONAL`: the statement is a formulation, writing contract, or planned
  experiment and is not phrased as achieved behavior.
- `BLOCKED`: missing synchronized code, protocol, result, authorship, template,
  or audit evidence prevents affirmative manuscript wording.
- `table_eligible=true` requires a new synchronized MultiRep family with frozen
  provenance and passing `experiment-audit` and `result-to-claim`. Every current
  result remains ineligible.

Evidence priority is collaborator frozen artifact > tracked code/audit >
committed PPT as visual reference > informal prose. A lower tier cannot close a
higher-tier obligation.

## Authoritative Ledger

| ID | Claim or obligation | Permitted wording now | Evidence | Status | `table_eligible` | Closure required |
|---|---|---|---|---|---:|---|
| C01 | The paper's exact scope is identity-indexed, window-local tempo routing for multi-person repetition counting. | Use the working title *Identity-Indexed Local Tempo Routing for Multi-Person Repetition Counting*; do not imply primacy. | User-approved ICASSP plan; `NARRATIVE_REPORT.md` | PROVISIONAL | false | Reconcile the title against synchronized implementation and final experiment scope. |
| C02 | The primary target is a person-wise count vector; the scene total is derived. | “We formulate the target as per-person counting from identity-indexed pose tracks.” | User-approved formulation; committed introduction PPT as visual reference | PROVISIONAL (`SYNC-REQUIRED`) | false | Bind input/output tensors, identity semantics, and evaluator to collaborator code and MultiRep. |
| C03 | The signal challenge combines inter-person asynchrony with tempo changes inside an identity trajectory. | “We study asynchronous tracks with within-track non-stationary tempo.” | Narrative framing; prior-work boundary; planned piecewise-warp diagnostic | PROVISIONAL | false | Verify that the dataset/protocol contains or deterministically constructs the stated conditions. |
| C04 | The current tracked repository is a single-person PAMS reproduction, not the proposed MRAC method. | “The local codebase is a single-sequence starting point.” | `README.md`; `docs/METHOD_SPEC.md`; `docs/PAPER_AUDIT.md`; `docs/ASSUMPTIONS.md` | CONFIRMED | false | Preserve this boundary until a collaborator freeze supersedes it. |
| C05 | The proposed default shares encoder/expert parameters across people while maintaining identity-indexed masks and state. | “We propose shared parameters with per-track state.” | Framework PPT; provisional method contract | BLOCKED (`SYNC-REQUIRED`) | false | Source anchors, tensor contracts, parameter-sharing test, and cross-identity state-isolation test. |
| C06 | Every overlapping identity window produces local masked ACF/FFT period evidence and confidence. | “The draft estimates window-local periodic evidence.” | Tracked single-sequence period estimator as a starting component; provisional math contract | BLOCKED (`SYNC-REQUIRED`) | false | Exact input signal, masks, lag/frequency units, bounds, interpolation, confidence, fallback, code anchors, and tests. |
| C07 | A soft router selects among shared fast/medium/slow response experts. | “We propose soft tempo routing over shared experts.” | Framework PPT; provisional math contract | BLOCKED (`SYNC-REQUIRED`) | false | Router inputs, normalization, temperature, gradient path, expert definitions, specialization evidence, and executable ablations. |
| C08 | Expert responses are fused before normalized overlap-add, then decoded once per identity. | “The writing contract reconstructs one response and applies one track-level decoder.” | User-approved provisional ordering; framework visual reference | BLOCKED (`SYNC-REQUIRED`) | false | Taper, denominator, padding/gaps, fusion order, decoder, overlap-duplication test, and one-pass test. |
| C09 | Training combines PAMS-TCC with routing-consistency and track-continuity terms. | “PAMS-TCC is the documented starting objective; the other terms remain synchronization placeholders.” | `docs/METHOD_SPEC.md`; tracked loss code; conceptual slide | BLOCKED (`SYNC-REQUIRED`) | false | Prove whether both terms exist; bind exact formulas, samples, supervision, weights, gradients, configs, and tests. Delete absent terms. |
| C10 | The intended counting objective uses no person-wise count or period labels. | Use only the full qualifier; explicitly disclose supervision in pose/tracking frontends. | Local evaluator boundary is partial starting evidence; `paper/evidence/supervision_firewall.json` is currently blocked | BLOCKED (`SYNC-REQUIRED`) | false | PASS every firewall scope at one frozen collaborator commit, including tuning, selection, early stopping, calibration, and inference. |
| C11 | All current UCFRep dev84/proxy/smoke outputs are excluded from ICASSP result tables; test105 is untouched. | “Existing local results are negative or diagnostic and are not evidence for the proposed method.” | `results/README.md`; `results/dev-negative/**`; experiment audit; sealed-test record | CONFIRMED | false | No relabeling can make these artifacts eligible; a new MultiRep artifact family is required. |
| C12 | The evaluation uses an exact MultiRep release/split/checksum and distinguishes oracle, common predicted, and native tracks. | Describe only a preregistered protocol. | `paper/evidence/result_manifest.schema.json`; `NARRATIVE_REPORT.md` | BLOCKED (`data-pending`) | false | Dataset/evaluator/preprocessing hashes, track artifacts, per-person GT/predictions, and tracking diagnostics. |
| C13 | The proposed method is effective or improves Period-mAP/AP50/AP75/AvgMAE/AvgOBO. | No affirmative result sentence; render one controlled pending placeholder in draft mode. | No synchronized proposed-method result exists | BLOCKED (`data-pending`) | false | At least three seeds, frozen runs, matched protocol, mean/sample SD, paired video-cluster bootstrap 95% CI, and two passing empirical gates. |
| C14 | Local routing, soft routing, boundary reconstruction, and implemented auxiliary terms cause the reported behavior. | State only the planned ablation and piecewise time-warp diagnostic. | Preregistered figure/table plan; no measurements | BLOCKED (`data-pending`) | false | Main and ablation JSON/CSV, exact variant commits/configs, paired comparisons, and generated table/figure hashes. |
| C15 | Positioning relative to MultiCounter, MultiCounter+, PAMS, Track-PAMS and other baselines is accurate. | Acknowledge established MRAC, identity/period, pose, self-supervision, and tempo-adaptation coverage; avoid “first” and unmatched SOTA. | Existing verified bibliography is a candidate source set, not an inherited ICASSP citation audit | PROVISIONAL | false | Reverify each BibTeX record and each local citation context against primary sources; rerun citation and novelty/claim scans. |
| C16 | The manuscript satisfies ICASSP 2027 regular-submission and ARIS readiness rules. | “This is a provisional pre-results draft.” | Venue contract and current workflow state | BLOCKED | false | Four technical pages; page 5 restricted to references/funding/ethics; official 2027 kit; complete author order/affiliations; no placeholders; all mandatory audits fresh and nonblocking; healthy cross-family trace. |

## Three Contribution Groups Allowed in the Pre-Results Draft

1. **Problem formulation (C02–C03):** a person-wise, masked, identity-indexed
   non-stationary signal formulation, explicitly presented as the target setting.
2. **Proposed mechanism (C05–C08):** local period evidence, soft routing, shared
   response experts, normalized reconstruction, and one decoder, all visibly
   marked `SYNC-REQUIRED` until bound to code.
3. **Evaluation contract (C11–C14):** protocol separation, artifact requirements,
   main comparison, ablation, and tempo-change diagnostic, presented as a
   preregistration rather than completed evidence.

No contribution may convert a blocked row into past-tense achievement language.
After collaborator synchronization, this ledger must be revised before editing
the abstract, contributions, result captions, or conclusion.

## Current Result Exclusion Register

| Artifact family | Admissible use | Forbidden use | `table_eligible` |
|---|---|---|---:|
| `results/dev-negative/**` | Failure analysis and implementation-risk history | ICASSP main result, abstract value, or proposed-method comparison | false |
| `results/synthetic/**` | Component sanity with explicit proxy labeling | MultiRep or end-to-end evidence | false |
| `results/safety/**` | Local property/collapse checks | Accuracy, robustness, or generalization claim | false |
| `results/server-smoke/**` | Environment and cache readiness | Method performance | false |
| PAMS-Literal | Period-Head specification-gap diagnostic | Learned-period or proposed-method result | false |
| PAMS-SSHead and other inferred repairs | Independent diagnostic only | Author-disclosed PAMS, Track-PAMS, or the proposed MRAC method | false |

The exclusion register is monotonic: eligibility requires a new, independently
identified artifact family, never a renamed old row.
