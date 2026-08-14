# Claims–Evidence Matrix

**Scope:** CVPR 2027 pre-result manuscript  
**Overall state:** `provisional/data-pending`  
**Evidence freeze:** `73cbc4bb9b581f24fbc90f2fe7421bc11e7c7454`

## Status Vocabulary

- `CONFIRMED`: the stated fact is directly supported by the named tracked artifact.
- `PROVISIONAL`: the statement is an explicitly proposed formulation or paper plan, not an implementation/result claim.
- `BLOCKED`: the statement requires missing collaborator code, data, or audited results and cannot enter the paper as an achieved fact.

`table_eligible=true` is reserved for synchronized, frozen, protocol-matched results that pass `experiment-audit` and `result-to-claim`. No current result qualifies.

## Ledger

| ID | Claim or paper obligation | Permitted wording now | Evidence source(s) | Evidence class | Status | `table_eligible` | Required closure |
|---|---|---|---|---|---|---:|---|
| C01 | The primary task output is a vector of per-person repetition counts; a scene total is only derived. | “We formulate the target as per-person counting.” | `NARRATIVE_REPORT.md`; committed introduction PPT | user-approved formulation + conceptual slide | PROVISIONAL | false | Bind the input/output schema to collaborator code and MultiRep protocol. |
| C02 | The technical challenge is non-stationary tempo within each identity trajectory in a multi-person scene. | “We target within-track tempo changes and asynchronous people.” | `NARRATIVE_REPORT.md`; committed introduction PPT | user-approved framing + conceptual slide | PROVISIONAL | false | Verify dataset coverage and define the local-tempo stress subsets. |
| C03 | The proposed default uses a shared pose encoder and shared fast/medium/slow experts with identity-specific state. | “Our provisional design shares parameters while maintaining identity-conditioned state.” | committed framework PPT; `paper/evidence/method_contract.yaml` | conceptual design | BLOCKED (`SYNC-REQUIRED`) | false | Map each module to collaborator source files, classes, and configuration fields. |
| C04 | Local windows estimate period and use soft tempo routing. | “We propose window-level soft tempo routing.” | committed framework PPT; `paper/evidence/method_contract.yaml` | conceptual design | BLOCKED (`SYNC-REQUIRED`) | false | Synchronize local-period estimator, router equations, and executable tests. |
| C05 | Experts emit local response streams that are fused by normalized overlap-add, followed by one peak extraction per identity. | “The draft defines response-level fusion and one trajectory-level extraction pass.” | `NARRATIVE_REPORT.md`; `paper/evidence/method_contract.yaml` | user-approved provisional math contract | BLOCKED (`SYNC-REQUIRED`) | false | Bind windows, normalization, boundary policy, and peak extractor to code and tests. |
| C06 | The proposed objective combines PAMS-TCC, routing consistency, and track continuity. | “The training objective is provisional; only the PAMS-TCC starting point is documented locally.” | `docs/METHOD_SPEC.md`; `src/pams/losses.py`; `paper/evidence/method_contract.yaml` | tracked starting component + conceptual extensions | BLOCKED (`SYNC-REQUIRED`) | false | Obtain exact formulas, weights, gradient paths, and implementation anchors for both new losses. |
| C07 | Training does not require per-person count labels. | “The method is intended to avoid per-person count supervision.” | `src/pams/evaluation.py` establishes label-free prediction only for the current PAMS reproduction; `paper/evidence/method_contract.yaml` | partial starting-code evidence + intended supervision contract | BLOCKED (`SYNC-REQUIRED`) | false | Audit every collaborator data loader, loss, selector, and tuning path for count-label access. |
| C08 | The current tracked repository implements a single-person PAMS reproduction and does not implement cross-person identity tracking. | “The local repository is a single-person starting point, not the proposed MRAC implementation.” | `README.md`; `docs/METHOD_SPEC.md`; `docs/ASSUMPTIONS.md`; `src/pams/model.py` | tracked code and audit documentation | CONFIRMED | false | None; preserve this boundary until synchronized code supersedes it. |
| C09 | The disclosed/literal PAMS Period Head is untrained/random in the tracked reproduction, while SSHead is an inferred repair. | “Local PAMS variants are diagnostics with explicitly different evidence classes.” | `README.md`; `docs/PAPER_AUDIT.md`; `docs/METHOD_SPEC.md`; `docs/ASSUMPTIONS.md` | tracked audit documentation | CONFIRMED | false | Never transfer these variants into the proposed method's evidence without a new protocol. |
| C10 | Existing UCFRep dev84 outputs are negative, development-only, proxy, or diagnostic evidence and cannot populate the proposed paper's main table. | “Current local results are excluded from the new method tables.” | `results/README.md`; `results/dev-negative/README.md`; `results/dev-negative/summary.json` | tracked result audit | CONFIRMED | false | Keep exclusion history; replace only with synchronized MultiRep artifacts that pass both audits. |
| C11 | The sealed UCFRep test105 split has not been scored in this worktree. | “No UCFRep test metric is available.” | `README.md`; `results/README.md`; `results/dev-negative/summary.json` | tracked protocol audit | CONFIRMED | false | Do not run or cite sealed-test values in this writing task. |
| C12 | The proposed method improves accuracy, robustness, generalization, or efficiency over baselines. | No affirmative wording is allowed; use the controlled result placeholder. | no synchronized result artifact | missing empirical evidence | BLOCKED | false | Frozen MultiRep manifests; matched baselines; at least three seeds; uncertainty; successful audits. |
| C13 | The proposed method is state of the art or is the first MRAC/person-wise/asynchronous/variable-tempo/pose/self-supervised method. | No such novelty or superiority wording is allowed. | prior work must be checked through primary publications; no supporting novelty study exists | unsupported novelty claim | BLOCKED | false | A dedicated novelty audit could narrow a claim, but the listed “first” claims remain prohibited by the paper plan. |
| C14 | Figures 1–2 describe the intended task contrast and architecture. | “Conceptual illustration” or “proposed framework”; never “measured behavior.” | committed PPTs as `visual_reference_only`; `paper/figures/task_comparison.drawio`; `paper/figures/framework.drawio`; validated SVG/PDF exports | provisional method contract + editable vector artifacts | PROVISIONAL (asset complete) | false | Re-audit labels and captions after collaborator method synchronization. |
| C15 | Tracking metrics are part of the end-to-end diagnostic, while fixed predicted tracks make those values common across downstream counters. | “We report tracking quality separately from counter quality under a fixed frontend.” | `NARRATIVE_REPORT.md`; `paper/evidence/result_manifest.schema.json` | protocol contract | PROVISIONAL | false | Supply oracle- and predicted-track manifests with HOTA, IDF1, and IDSW. |
| C16 | The manuscript is ready for submission. | “The manuscript is a pre-result draft under submission-level audit rules.” | missing collaborator artifacts; missing official CVPR 2027 template; cross-family overlay not healthy | workflow state | BLOCKED | false | Close AC01–AC16, obtain nonblocking audit verdicts, migrate to the official template, and rerun all checks. |

## Contribution-to-Evidence Map

The Introduction may contain exactly three contribution bullets in the pre-result draft:

1. **Formulation:** C01–C02, explicitly framed as the target problem.
2. **Mechanism:** C03–C06, explicitly framed as a proposed and synchronization-pending design.
3. **Evaluation contract:** C10, C12, and C15, explicitly framed as the preregistered protocol rather than completed evidence.

No contribution bullet may convert a `BLOCKED` row into past-tense achievement language. After result synchronization, the ledger must be updated before any result-dependent contribution is enabled.

## Current Result Exclusion Register

| Artifact family | Allowed use | Disallowed use | `table_eligible` |
|---|---|---|---:|
| `results/dev-negative/**` | failure analysis, provenance, implementation-risk discussion | proposed-method main result, abstract number, SOTA comparison | false |
| `results/synthetic/**` | component sanity or proxy diagnosis with explicit labeling | benchmark or end-to-end evidence | false |
| `results/safety/**` | local safety/property checks | accuracy, robustness, or generalization evidence | false |
| `results/server-smoke/**` | environment/cache readiness | method performance | false |
| PAMS-Literal | paper-gap diagnostic | validated learned-period method | false |
| PAMS-SSHead | independently inferred repair diagnostic | author-disclosed PAMS or proposed MRAC | false |

This register is monotonic: an existing row cannot become eligible by relabeling. A new frozen artifact family with its own manifest and audit record is required.
