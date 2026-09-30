# TempoRAC Refinement Report

## Final disposition

TempoRAC completed five research-refinement rounds with a final score of 9.125/10.000 and verdict READY. The result is a same-family provisional method-contract judgment. It does not establish novelty, empirical success, acceptance likelihood, or authority to implement, access compute or data, run the contract, write a paper, or submit.

## Canonical bindings

| Item | Path | SHA-256 |
|---|---|---|
| Initial proposal | refine-logs/temporac/round-0-initial-proposal.md | Recorded in the existing refinement provenance |
| Final refinement source | refine-logs/temporac/round-4-refinement.md | ff2f4db61ce7be419b9bc440579a7f18cf0400180de054fde4b3e3c92fae0c90 |
| Final normative proposal | refine-logs/temporac/FINAL_PROPOSAL.md | 562325dedab0637421f62dec5ad149b3c884f67b47c4b525c8fb54a57bb1233c |
| Final review Markdown | refine-logs/temporac/round-5-review.md | b183cd49f96b2243d8b67187119e6af468c87fa7ad4edb5ebc5727250b104c67 |
| Final review JSON/verdict | refine-logs/temporac/round-5-review.json | 6a32c0ecab12d29dcc5d2d27cc2110165f3b919126a314ec3f54b4c77ace4768 |
| Stable Problem Anchor body | mechanical section-body extraction | 44dce8546f499acdede797055625657579685c0b75dc64c26d3f51003e956b71 |

FINAL_PROPOSAL.md is a mechanical extraction beginning with the proposal title immediately after the Revised Full Proposal marker and continuing through end of file. It contains one complete Problem Anchor and no Changes Made section or reviewer commentary. Its contract bytes are unchanged from that source slice.

## Refinement objective and invariant

The invariant problem was to count repetitions for each externally supplied person-indexed pose track when people and individual tracks change pace, using identity-indexed window-local soft routing over shared fast, medium, and slow response experts. The count objective cannot consume human per-person count, period, density, or cycle-boundary annotations; responses must be reconstructed continuously and decoded exactly once per identity.

Across all rounds, the refinement defended the original seven-property graph against drift toward WARP-PHASE, a missing-pose set, scene-level counting, an easier single-person task, an integral fallback decoder, a learned global router, or additional contribution branches. The final anchor-body hash is unchanged.

## Round-by-round record

### Round 1 — structural audit

- Score/verdict: 6.575, REVISE.
- Reviewer pressure: the degree-one physical-repetition unit was not yet demonstrated; sparse peak thresholds were heuristic; router gains were confounded with capacity and response changes; the contribution list was too broad; label firewall and feasibility were underspecified.
- Refinement response: defined a fail-closed teacher/certificate direction, constrained the decoder to one connected-component pass, introduced matched causal controls, narrowed the contribution claim, and began explicit gates and compute accounting.
- Residual risk after the round: per-track seams, target receipts, exact control equations, population ownership, and resource closure were incomplete.

### Round 2 — numerical and causal closure attempt

- Score/verdict: 7.590, REVISE.
- Reviewer pressure: each track needed an auditable certificate; seam/gap semantics and peak ownership needed fixed numbers; G4 needed trained-output rather than generator-only evidence; controls and cue interventions needed exact matching.
- Refinement response: added per-track certified receipts, deterministic window/seam/gap rules, trained-output operator strata, cue-only interventions, matched capacity and shortcut controls, stronger firewall stages, and a finite job ledger.
- Residual risk after the round: the v44 clock/gap rule, a static second-moment claim, NOLA gradient/control weights, G4 strata, and generator/reset/gate ordering remained internally inconsistent.

### Round 3 — contradiction repair and deeper audit

- Score/verdict: 7.445, REVISE.
- Reviewer pressure: repair the v44 gap/clock mismatch, remove the false second-moment claim, make the fused NOLA path and gradients explicit, define operator populations, and close resets, fixtures, and stage order.
- Refinement response: established the v3 causal and decoding semantics, explicit fixed fixtures, fused-only gradients, population-aware testing, and fail-closed ordering.
- Residual risk after the round: strict review exposed incompletely defined X0 phase/traversal and offsets, topology attacks, the K7 primitive physical unit, identity isolation, archive/runtime commitments, and natural-drift domain ownership.

### Round 4 — executable v4 contract

- Score/verdict: 7.920, REVISE.
- Reviewer pressure: close all X0 traversal/phase/offset definitions, bind topology fixtures, prove primitive-unit identity at K7, test untouched-identity isolation, define archive and runtime hashes, and eliminate natural-clock interpolation ambiguity.
- Refinement response: produced temporac.execution.v4 with fixed X0, topology, and 10,368-row operator manifests; analytic/certified/resampler pulse equality; all-offset decoding; two-identity counterfactual isolation; exact runtime and archive schemas; canonical receipt preimages; a one-invocation evaluator capability; deterministic natural drift; fixed failure reasons; and a 93-A6000-hour prospective ledger.
- Residual risk after the round: a final independent pass over the complete v4 bytes was still required.

### Round 5 — final readiness audit

- Score/verdict: 9.125, READY.
- Audit scope: complete final source bytes, both source anchor copies, original graph, equations and domains, degree-one teacher, NOLA and decoder logic, causal interventions, matched controls, all gates and populations, vault firewall, runtime/receipt bindings, resource ledger, claim minimality, and stopping boundary.
- Findings: no blocking mathematical or execution-contract contradiction remained. The reviewer explicitly confirmed closure of the previously open direction, balanced-BCE, pulse-edge-weight, K3-offset, natural-clock interpolation, and evaluator-interval primitive-unit issues.
- Ceiling: the review is same-family provisional and method-contract-only. It is not execution evidence or novelty clearance.

## Score evolution

| Round | Problem Fidelity | Method Specificity | Contribution Quality | Frontier Leverage | Feasibility | Validation Focus | Venue Readiness | Weighted Overall | Verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 9.5 | 6.0 | 5.5 | 7.5 | 5.5 | 6.5 | 5.5 | 6.575 | REVISE |
| 2 | 9.6 | 7.2 | 7.5 | 7.5 | 6.5 | 7.5 | 6.5 | 7.590 | REVISE |
| 3 | 9.4 | 6.4 | 7.8 | 8.2 | 5.8 | 7.0 | 6.5 | 7.445 | REVISE |
| 4 | 9.7 | 6.8 | 8.6 | 8.7 | 6.3 | 6.8 | 6.8 | 7.920 | REVISE |
| 5 | 9.9 | 9.2 | 8.8 | 9.1 | 8.8 | 9.4 | 8.5 | 9.125 | READY |

The non-monotonic Round-3 score is genuine: stricter equation- and domain-level review exposed execution contradictions that the earlier structural score had not yet captured. The final score recomputes exactly under weights 15/25/25/15/10/5/5 across the displayed dimensions.

## Final proposal contract

The finalized proposal keeps one dominant scientific claim: identity-indexed window-local relative-tempo routing over shared response scales must improve supplied-track counting under deterministic within-track drift relative to the strongest frozen matched nonlocal/control comparator while retaining clean performance.

The contract binds:

1. the authoritative v44 supplied-track population and unconditional population floors;
2. degree-one teacher certificates tied to analytic X0 events and later evaluator intervals;
3. causal cue-only same-response interventions plus matched capacity and shortcut controls;
4. positive differentiable NOLA, fused-only gradients, pre-gap isolation, and one final connected-component decode per identity;
5. fixed X0, topology, and operator fixtures, source-realizable trained strata, all offsets, margins, pulse pullback, and count equality;
6. immutable feature, checkpoint, prediction, receipt, and evaluator-vault capability boundaries;
7. an acyclic G0-G5a/G5b/K0-K7 fail-closed sequence with fixed job inventory and resource ceilings.

Exactly three paper-visible evidence blocks are allowed by the contract: drift performance with clean retention; the heldout three-by-three expert-specialization matrix and diagonal statistic; and certificate/pulse/resampler/decoder-boundary evidence. Other fixtures and receipts remain audit evidence, not additional contributions.

## Review evidence

The canonical raw reviewer narratives and structured judgments remain in:

- refine-logs/temporac/round-1-review.md and round-1-review.json
- refine-logs/temporac/round-2-review.md and round-2-review.json
- refine-logs/temporac/round-3-review.md and round-3-review.json
- refine-logs/temporac/round-4-review.md and round-4-review.json
- refine-logs/temporac/round-5-review.md and round-5-review.json

The corresponding complete refinement responses remain in round-1-refinement.md through round-4-refinement.md. This report summarizes those canonical records and does not replace them.

## Pushback and drift record

The process consistently rejected changes that would make the research easier by changing its identity:

- no WARP-PHASE pivot or learned global router;
- no missing-pose set, scene-level counting, or single-person simplification;
- no integral decoder fallback or extra decode pass;
- no fourth seed, extra response job, new dataset, or positive result statement;
- no evaluator-label training, early vault opening, post-failure retuning, filtering, or rerun;
- no expansion of audit fixtures into additional scientific contributions.

This pushback kept the original problem graph intact while increasing executability rather than scope.

## Remaining limitations

### Empirical

No S0 bytes have been materialized, no gate has run, and no result exists. Strict certificates, topology tests, specialization, clean retention, drift improvement, resource ceilings, or K7 can fail. READY does not imply that TempoRAC will pass.

### Novelty and positioning

Closest-prior and TWCRAC novelty adjudication remain unresolved. The TempoRAC name is explicitly working-only. Priority, title, acronym, paper framing, and acceptance likelihood remain unapproved.

### Execution

The v4 fixture, receipt, archive, and capability machinery is deliberately fail-closed and operationally demanding. The 93-A6000-hour ledger is prospective, not a reservation or launch approval. A future execution phase would need a fresh experiment plan, S0 materialization, and server/resource preflight under separate authority.

### Review independence

All semantic reviewers were GPT-5.6-Sol within the same OpenAI family. The 9.125 READY judgment is therefore provisional and should not be represented as independent external validation.

### Paper

No four-page compression, figure design, results narrative, claim-to-evidence table, paper writing, or submission work has begun or been authorized.

## Authorization boundary

This finalization changes only the TempoRAC research-refine records, the restoration run state, and repository manifest/checksum bookkeeping. It authorizes no code, test, configuration, data, result, server, experiment, paper, submission, branch, commit, or remote change.

## Completion statement

The refinement loop is complete at Round 5. FINAL_PROPOSAL.md is the sole normative finalized proposal. REVIEW_SUMMARY.md and this report are explanatory records; where they differ from the proposal, the proposal controls. A fresh experiment plan and preflight are merely identified as possible future separately authorized actions and were not started here.
