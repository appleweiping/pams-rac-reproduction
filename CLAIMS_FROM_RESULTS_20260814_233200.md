---
route: pivot
verdict: FAIL
reason: NO_MULTI_PERSON_REAL_GT_EVIDENCE
review_independence: same-family/provisional
created_at: 2026-08-14T20:38:01+08:00
evidence_freeze: 26e5b7176f1f7c678391163c3278b5c390d54115
paper_state: provisional/data-pending
experiment_audit: EXPERIMENT_AUDIT.json
reviewed_at: 2026-08-14T21:37:16+08:00
reviewer_agent: /root/icassp_result_to_claim
---

# ICASSP 2027 Claims From Results

The result-to-claim route remains `pivot`. The local evidence tree contains no
synchronized multi-person implementation, no MultiRep ground truth or
predictions, and no table-eligible proposed-method result. The existing ARIS
experiment audit failed for `NO_MULTI_PERSON_REAL_GT_EVIDENCE`; therefore no
positive empirical claim is admissible and every confidence value below is
`low` regardless of how clear a static repository boundary appears.

## R1 — Implemented identity-indexed local-tempo counter

- **claim_supported:** `no`
- **evidence supports:** a tracked single-person PAMS reproduction with a pose
  encoder, single-sequence period estimators, and a scalar count output.
- **evidence does not support:** persistent identities, a person axis,
  identity-specific state, local-window tempo routing, learned response experts,
  normalized overlap-add, or one decoder per reconstructed track.
- **admissible wording:** “The ICASSP manuscript defines a synchronization-required
  method contract derived from a single-person starting codebase.”
- **required closure:** digest-bound collaborator source anchors and tests for
  every proposed module at one clean frozen commit.
- **confidence:** `low`

## R2 — No person-wise count or period supervision

- **claim_supported:** `partial`
- **evidence supports:** the current single-person training API consumes poses,
  and count targets are introduced at the separate evaluator boundary.
- **evidence does not support:** the missing MultiRep loaders, pseudo-labels,
  router targets, auxiliary losses, tuning, checkpoint selection, early
  stopping, calibration, and inference path.
- **admissible wording:** “The intended counting objective is designed not to
  use person-wise repetition-count or period labels; this property is not yet
  verified for the synchronized multi-person pipeline.”
- **required closure:** every scope in the supervision firewall must have a
  digest-bound source anchor and PASS verdict at the collaborator freeze.
- **confidence:** `low`

## R3 — Effective or competitive MultiRep performance

- **claim_supported:** `no`
- **evidence:** the declared MultiRep bundle is absent. There is no frozen
  release/split/checksum, evaluator, per-person target, prediction, run log,
  aggregate, or uncertainty record.
- **admissible wording:** “No empirical conclusion about MultiRep performance
  can currently be drawn.”
- **required closure:** protocol-complete proposed-method and baseline families,
  at least three seeds for reproduced learned methods, and passing
  `experiment-audit` followed by `result-to-claim`.
- **confidence:** `low`

## R4 — Main comparison or abstract number

- **claim_supported:** `no`
- **evidence:** all exact local values belong to single-person UCFRep dev84
  negative, proxy, inferred-repair, or official-checkpoint sanity artifacts.
  They do not measure the proposed method or the target dataset.
- **admissible wording:** the draft renders one controlled pending result
  sentence and empty evidence-bound cells; submission mode must fail on both.
- **required closure:** one audited result sentence whose values resolve exactly
  to eligible manifest keys, plus a main table that visibly separates native
  contextual and common-track protocols.
- **confidence:** `low`

## R5 — Causal mechanism or robustness conclusion

- **claim_supported:** `no`
- **evidence:** there is no executable local/global tempo, routing, boundary,
  continuity, or piecewise-warp comparison for the proposed system.
- **admissible wording:** describe the ablation matrix and non-stationary tempo
  diagnostic only as a preregistered evaluation contract.
- **required closure:** exact variant commits/configs, same-seed paired outputs,
  generated table/figure hashes, uncertainty, and result-to-claim adjudication.
- **confidence:** `low`

## R6 — Existing UCFRep artifacts in ICASSP tables

- **claim_supported:** `no`
- **evidence:** repository documentation marks the current artifacts as
  development-only, negative, proxy, diagnostic, inferred, or protocol-mismatched.
  The sealed UCFRep test105 split remains unscored: no predictions,
  evaluation, or metrics exist, although an older development CLI
  deserialized the full manifest.
- **admissible wording:** the artifacts may support a repository-boundary or
  failure-history statement, never an ICASSP method result.
- **required closure:** none can convert these exact artifacts into MultiRep
  evidence; a new synchronized result family is required.
- **confidence:** `low`

## Routing Decision

Proceed with the full ICASSP method, protocol, figure, table, and submission-gate
scaffold while withholding all positive results. The visible paper status is
`DRAFT — RESULTS PENDING`; proposed components remain `SYNC-REQUIRED`; the
overall state remains `provisional/data-pending`. After collaborator delivery,
rerun `experiment-audit` before rerunning this result-to-claim gate. No abstract
number, comparative superlative, causal explanation, or performance contribution
may be enabled before both audits pass.
