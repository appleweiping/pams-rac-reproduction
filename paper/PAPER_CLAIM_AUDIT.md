---
audit_skill: paper-claim-audit
verdict: BLOCKED
reason_code: data_pending
review_independence: same-family
acceptance_status: provisional
agent_id: /root/icassp_paper_claim_audit
model: gpt-5.6-sol
reasoning: ultra
generated_at: 2026-08-14T22:43:34+08:00
trace_path: paper/.aris/traces/paper-claim-audit-icassp/final/
---

# ICASSP Paper Claim Audit

## Outcome

**BLOCKED / data_pending.** The audited PDF SHA-256 is
`873c83da0dbf4b24a6e1199f3dcd980e17d0b032580e0591f88a7c545ec67a45`,
matching the supplied final input. No eligible MultiRep data or synchronized
multi-person implementation exists, so empirical claims cannot be licensed.
The manuscript does not hide this condition: PENDING, DATA-PENDING, and
SYNC-REQUIRED controls remain visible in the rendered PDF.

**Unsupported enabled performance claims: none.** There are zero enabled
empirical result values, zero enabled performance comparisons, and zero unbound
concrete empirical numeric claims. The draft is correctly blocked rather than
passed, and the absence of data is reported as `data_pending` rather than collapsed
into an opaque failure.

## Audited scope and independence

This was a fresh zero-context review. No earlier paper-claim audit, claim-audit
history, or log was read or reused. The active TeX dependency closure was derived
from `paper/main.tex` and contains:

- `paper/main.tex`
- `paper/preamble.tex`
- `paper/math_commands.tex`
- `paper/sections/0_abstract.tex`
- `paper/sections/1_introduction.tex`
- `paper/sections/3_method.tex`
- `paper/sections/4_experiments.tex`
- `paper/sections/5_conclusion.tex`

`paper/submission_wrapper.tex`, `paper/appendix/a_experiment_contract.tex`, and
`paper/figures/latex_includes.tex` are not inputs to this draft PDF and therefore
are not treated as active claim-bearing TeX. The directly included framework PDF
and bibliography were hash-checked as active non-TeX build inputs.

## Machine-readable evidence and source binding

| Control | Current state | Claim-audit consequence |
|---|---|---|
| `results_manifest.json` | `data-pending`; `frozen=false`; `table_eligible=false` | No result may be enabled. |
| Artifact identity | `artifact_id=null`, `artifact_family=null`, `method_commit=null` | No proposed-method result family or code binding exists. |
| Dataset/evaluator | MultiRep release, split, checksum, frontend, evaluator, and evaluation type are null/empty | Protocol and metric semantics are unbound. |
| Raw evidence | seeds, configurations, logs, predictions, source artifacts, and metrics are empty | No machine-readable value can support a paper cell. |
| Paper bindings | protocol fields, results, and claims are empty; abstract result is null | Every controlled paper value must remain unresolved. |
| Eligibility | every required eligibility flag is false | Generation and submission are ineligible. |
| `method_manifest.yaml` | tracked checkout is single-person with one scalar output; proposed method and supervision firewall are `SYNC-REQUIRED` | The checkout cannot evidence the multi-person method. |
| `claim_evidence.csv` | no row has `submission_licensed=true` | No claim is submission-licensed. |
| Generated binding | `paper/generated/evidence_values.tex` is absent | Draft fallbacks must render; submission must fail closed. |
| Result-to-claim gate | `FAIL/NO_MULTI_PERSON_REAL_GT_EVIDENCE`, provisional pivot | Only a visibly provisional scaffold is admissible. |

The result-to-claim record's hashes for `results_manifest.json` and
`method_manifest.yaml` match the current bytes. No raw evidence paths are supplied
for this paper audit to follow, because the manifest arrays are empty.

## Controlled-value census

| Binding class | Active calls | Bound | Rendered behavior | Adjudication |
|---|---:|---:|---|---|
| `\R{...}` / `\pendingcell{...}` | 57 unique | 0 | `PENDING` | Withheld; not a numeric claim. |
| `\Protocol{...}` | 2 unique | 0 | `PENDING` | Protocol identifiers withheld. |
| `\ControlledAbstractResult` | 1 | 0 | Explicit evidence-pending sentence | Abstract result withheld. |
| `\C{...}` | 0 | 0 | Not used | No enabled result prose. |

The 57 result slots comprise 3 predicted-track diagnostics, 40 main-table
metrics, and 14 non-reference ablation deltas. All are unbound. The only concrete
numbers in result-table cells are the reference-row deltas `0` and `0`; because
the table defines every delta as variant minus reference, those two values are
identities, not measurements. Thus:

- `numeric_claim_count = 2` under the conservative scope “concrete numbers in a
  result-table cell”;
- `unbound_numeric_claim_count = 0`, because both numbers are definitionally
  bound and neither reports performance;
- enabled empirical numeric claims = 0;
- enabled unbound performance claims = 0.

The “at least three seeds” and “95% confidence interval” statements are planned
protocol requirements, not observed statistics. Formula constants, dimensions,
equation/reference numbers, citation metadata, metric names such as AP50/AP75,
layout values, and year/page metadata are excluded from the empirical numeric-claim
count.

## Claim ledger

| ID | Location/class | Claim group | Evidence/binding | Adjudication |
|---|---|---|---|---|
| PC01 | Title; narrative/method | Identity-indexed local tempo routing for multi-person repetition counting | Paper specification; no implementation source | Admissible only as the proposed specification named by the title. |
| PC02 | Abstract/introduction; task | Supplied persistent identity tracks yield an identity-wise count vector; scene total is diagnostic | Equations and prose; method manifest task contract is `SYNC-REQUIRED` | Coherent specification claim; not implementation-licensed. |
| PC03 | Abstract/introduction/conclusion; supervision | Counting targets/intended objective use no person-wise count or period labels; pose/tracking may use external supervision | Supervision firewall entirely `SYNC-REQUIRED` | Intention is disclosed, but implementation-level supervision claim is blocked. |
| PC04 | Abstract/method/conclusion; method | Shared encoder and experts, local masked ACF/FFT evidence, soft routing, NOLA reconstruction, one decode per track | All proposed components have null source anchors/tests | Valid proposal description; explicitly not an implemented-method claim. |
| PC05 | Abstract/method; structural | Reconstruct-before-decode avoids counting every window independently and removes that structural source of boundary double-counting | Follows from the written processing order | Non-empirical structural claim; no measured error reduction or one-peak guarantee is asserted. |
| PC06 | Introduction; hypothesis | Local routing should be more robust than parameter-matched global tempo while preserving stationary behavior | No eligible diagnostic/result | Explicitly “result-free hypothesis”; not a performance conclusion. |
| PC07 | Introduction; contributions | The paper formulates the signal view and specifies method/protocol/diagnostics | Current manuscript contents | Supported as specification contributions; empirical contribution is withheld. |
| PC08 | Method; invariants | Parameters are shared and state/masks/responses are identity-indexed; windows do not mix identities | Written equations, not code tests | Conditional property of the specification; implementation remains `SYNC-REQUIRED`. |
| PC09 | Method; periodic estimator | Mask-normalized ACF/FFT semantics and confidence/fallback constraints | Equation-level contract; estimator source absent | Proposal-level only; no observed estimator accuracy or specialization claim. |
| PC10 | Method; router/experts | Soft weights and shared experts form the proposed response path; parameter count is independent of track count | Architectural definition; profiling absent | Conditional complexity property only; runtime/memory claims are explicitly disabled. |
| PC11 | Method; reconstruction | NOLA produces one response before a single trajectory decode | Equation-level contract; implementation absent | Specification claim only; boundary benefit remains to be tested. |
| PC12 | Experiments; protocol | Native/contextual, oracle-track, and common predicted-track families must not be ranked together | Planned protocol prose | Appropriate prospective separation; dataset/evaluator fields remain pending. |
| PC13 | Experiments; metrics/statistics | Listed MultiRep metrics, at least three seeds, sample SD, paired video-cluster 95% CI | No release/evaluator/seeds/raw output | Preregistered requirements, not results. |
| PC14 | Experiments; main table | 40 method/metric cells would compare protocol-matched families | 40 unbound result slots | No values, ranking, gain, or superiority claim is enabled. |
| PC15 | Experiments; ablation | 14 non-reference deltas will test local/global tempo, evidence, routing, and reconstruction | 14 unbound result slots; two reference identities | Planned mechanism contrasts only; no causal conclusion is enabled. |
| PC16 | Experiments; diagnostic | Piecewise warp, mask leakage, occlusion, and switch tests will probe failure modes | `DATA-PENDING DIAGNOSTIC SHELL`; no generator or artifact | Future evaluation contract only. |
| PC17 | Conclusion; status | No local multi-person implementation or eligible MultiRep result exists | Both manifests and generated-file absence | Supported and central to the blocked verdict. |
| PC18 | Introduction; citation-dependent background | Prior work has the described scope and component ideas are established | Citations present; AC09 citation audit is pending | Not independently licensed by this paper-claim audit. |

## PDF/source consistency and unlicensed-claim scan

The current PDF renders the exact title, `DRAFT – RESULTS PENDING`, the controlled
abstract pending sentence, pending tracking diagnostics, pending dataset/evaluator
fields, all pending main and ablation cells, the `DATA-PENDING DIAGNOSTIC SHELL`,
and the conclusion's no-implementation/no-results disclosure. It also visibly
renders the method's synchronization markers. A scan of rendered prose found no
enabled “outperforms,” “state-of-the-art,” “superior,” “best,” “competitive,” or
improvement claim.

The figure caption calls the signal path proposed and explicitly denies that it is
implementation or result evidence. The method text likewise says the local worktree
does not contain the multi-person implementation. No stale generated declaration is
available to override the draft fallbacks.

## Final adjudication

The paper-claim gate must remain blocked until a synchronized collaborator commit,
component source anchors and tests, a complete supervision audit, a frozen MultiRep
release and evaluator, raw per-person ground truth and predictions, run
configurations/logs, required seeds and uncertainty, and passing upstream audits are
all hash-bound. Only then may `evidence_values.tex` be generated and result,
protocol, abstract, or comparative claims become enabled.

Current disposition: **BLOCKED/data_pending; no unsupported enabled performance
claim exists.**
