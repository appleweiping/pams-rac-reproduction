# Claims From Results

**Route:** `pivot`  
**Integrity status:** `FAIL — NO_MULTI_PERSON_REAL_GT_EVIDENCE`  
**Review independence:** `same-family/provisional`  
**Evidence freeze:** `73cbc4bb9b581f24fbc90f2fe7421bc11e7c7454`

This gate was run after `EXPERIMENT_AUDIT.json`. Because the experiment audit
failed, ARIS forces every confidence rating below to `low`, even when the
static code boundary is unambiguous. No positive empirical claim about the
proposed multi-person method is currently admissible.

## C1 — Implemented multi-person local-tempo counter

- **claim_supported:** `no`
- **what the evidence supports:** the frozen code implements a pose-driven,
  single-sequence PAMS reproduction with period-adaptive temporal consistency
  and a global-period fast/medium/slow inference consensus.
- **what it does not support:** there is no person axis, persistent identity
  interface, per-person count vector, window-local period field, or learned
  identity-conditioned router. Preprocessing selects a dominant pose and
  `CountResult` contains one scalar count.
- **required closure:** collaborator implementation anchors for the person-wise
  data path, masks, router, experts, overlap-add decoder, and identity tests.
- **admissible revision:** “The current code implements a pose-driven
  single-sequence PAMS reproduction; the multi-person architecture in this
  manuscript is a synchronization-required method contract.”
- **confidence:** `low`

## C2 — No person-wise count or period supervision

- **claim_supported:** `partial`
- **what the evidence supports:** the existing single-person encoder training
  API receives pose sequences rather than ground-truth counts, estimates
  periods internally, and keeps evaluator targets outside prediction.
- **what it does not support:** the supervision boundary of the intended
  MultiRep loader, router, continuity objective, model selection, and final
  training graph does not exist in this worktree.
- **required closure:** a source-level label-firewall audit and run receipt for
  the synchronized multi-person training pipeline.
- **admissible revision:** “The intended counting objective is designed to use
  no person-wise count or period labels; this property remains unverified for
  the proposed multi-person method.”
- **confidence:** `low`

## C3 — Effective or competitive on MultiRep

- **claim_supported:** `no`
- **evidence:** the declared MultiRep evidence path is missing. There is no
  multi-person dataset, target, prediction, evaluator output, or held-out run.
- **required closure:** protocol-complete MultiRep manifests, at least three
  seeds, per-instance predictions, same-protocol baselines, and uncertainty.
- **admissible revision:** “No empirical conclusion about MultiRep performance
  can currently be drawn.”
- **confidence:** `low`

## C4 — Quantitative abstract result

- **claim_supported:** `no`
- **evidence:** the only exact values are negative, single-person UCFRep dev84
  diagnostics. They are not results for the proposed method and are explicitly
  paper-table-ineligible.
- **required closure:** one audited, held-out, table-eligible result sentence
  whose values resolve to a frozen result artifact.
- **admissible revision:** omit the quantitative result in the pre-results
  manuscript and retain exactly one controlled placeholder that fails closed
  in submission mode.
- **confidence:** `low`

## C5 — Existing UCFRep dev results in the main table

- **claim_supported:** `no`
- **evidence:** the repository marks all current dev84/proxy rows
  `paper_table_eligible=false`; PAMS-Literal uses an untrained random Period
  Head and PAMS-SSHead is an independently inferred repair. The sealed test105
  split remains untouched.
- **required closure:** none can make these exact artifacts new-method main
  results. New, protocol-matched evidence is required.
- **admissible revision:** current UCFRep artifacts may only support a clearly
  labeled reproduction/audit limitation, never the proposed method’s main
  comparison.
- **confidence:** `low`

## Routing Decision

The manuscript proceeds only as a complete, anonymous **pre-results method
and evaluation contract**. All multi-person mechanisms remain
`SYNC-REQUIRED`; all performance language remains withheld; the final status
cannot exceed `provisional/data-pending`. After collaborator synchronization,
rerun `experiment-audit` before rerunning this gate.
