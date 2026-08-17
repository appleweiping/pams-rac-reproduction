# Independent reviewer record

## Verdict

`BLOCKED` (`reason_code: data_pending`; `same-family/provisional`).

The target PDF hash matches the supplied final hash. This is an openly marked
pre-results draft, and its evidence gates are functioning: there is no eligible
MultiRep result family, no synchronized multi-person implementation, no generated
evidence binding, and no enabled quantitative or comparative performance claim.
The manuscript is therefore blocked from empirical licensing and submission, but
the missing evidence is not concealed.

## Fresh-input boundary

This run did not read or reuse a previous paper-claim audit, audit history, or log.
The active TeX closure was reconstructed from `paper/main.tex`. It comprises
`preamble.tex`, `math_commands.tex`, and sections 0, 1, 3, 4, and 5. The appendix,
`figures/latex_includes.tex`, and `submission_wrapper.tex` are not inputs to the
audited draft PDF. The active figure PDF and bibliography were hash-checked as
supporting build inputs.

## Evidence adjudication

- `results_manifest.json` says `data-pending`, `frozen: false`, and
  `table_eligible: false`. Its artifact ID/family, method commit, MultiRep
  release/split/checksum, evaluator, hardware, seeds, configurations, logs,
  predictions, sources, metrics, and paper bindings are absent or empty.
- `method_manifest.yaml` classifies the tracked checkout as a single-person
  dominant-pose counter with one scalar output. It explicitly says that it cannot
  evidence the proposed multi-person method. Every proposed component, the
  training objective, fallback behavior, and the supervision firewall remain
  `SYNC-REQUIRED` without source anchors or tests.
- `claim_evidence.csv` licenses no claim for submission. Method and supervision
  claims are `SYNC-REQUIRED`; result claims are `DATA-PENDING`.
- `paper/generated/evidence_values.tex` does not exist, consistently with
  `paper/generated/README.md`. There are no declarations outside the macro
  definitions in `preamble.tex`.
- `RESULT_TO_CLAIM.json` is a current `FAIL/NO_MULTI_PERSON_REAL_GT_EVIDENCE`
  gate whose embedded hashes for both manifests match the current bytes. Its
  routing outcome permits only a visibly provisional scaffold, not an empirical
  result.

## Claim and binding census

- 57 unique active result slots are invoked: 3 predicted-track diagnostics,
  40 main-table metrics, and 14 non-reference ablation deltas. All 57 are
  unbound and render `PENDING`; none renders a number.
- Two active protocol slots (`multirep-release` and `evaluator-revision`) are
  unbound and render `PENDING`.
- One active controlled abstract-result call is unbound and renders the explicit
  evidence-pending sentence.
- There are zero active `\C{...}` result-claim calls.
- The only concrete numbers occupying result-table cells are the two reference-row
  deltas `0` and `0`. They are definitionally fixed because the reference is
  subtracted from itself; they are not measured performance values and require no
  raw-result binding.
- The planned requirements of at least three seeds and a 95% interval are protocol
  commitments, not observed statistics.
- Enabled empirical numeric claims: 0. Enabled unsupported performance or
  comparative claims: 0. Unbound concrete empirical numeric claims: 0.

## Claim-family review

1. **Title and task framing.** The title names the proposed specification. The
   abstract and introduction consistently condition the task on supplied identity
   tracks. This is admissible as proposal framing, not implementation evidence.
2. **Abstract method description.** Shared encoding, local ACF/FFT evidence, soft
   routing, normalized overlap-add, and one decode per identity are stated as the
   design. The controlled final sentence visibly withholds experimental evidence.
   The no-count/no-period-label wording is an intended target and remains blocked
   by the supervision-firewall status.
3. **Novelty and contributions.** The paper disclaims novelty for the component
   ideas and states a result-free robustness hypothesis. The three contributions
   use “formulates,” “specifies,” and “planned”; the third explicitly disables
   empirical claims. No superiority conclusion is made.
4. **Method.** Equations define a proposed signal-processing contract. Seven
   `SYNC-REQUIRED` calls preserve the unresolved tensor/lifecycle, estimator,
   router/expert, reconstruction/decoder, and supervision interfaces. Present-tense
   statements about parameter sharing, identity separation, and reconstruct-before-
   decode are properties of the written specification, not assertions that the
   collaborator implementation exists. The paper expressly says the local worktree
   lacks that implementation.
5. **Structural boundary statement.** “Avoids independently counting every
   overlapping window” and “removing one structural source of boundary
   double-counting” follow from reconstruct-before-decode. They do not claim a
   measured reduction in error, and the method text explicitly says one peak is not
   guaranteed.
6. **Protocol.** Oracle, common predicted-track, and native/contextual protocols are
   separated and cross-protocol ranking is forbidden. Metric semantics, assignment,
   seeds, uncertainty, raw outputs, and evaluator hashes are prospective contracts.
   The two dataset/evaluator identifiers remain visibly pending.
7. **Main results and ablations.** Every would-be measured cell is controlled by an
   unresolved result macro. The reference zeros are definitional. Captions say
   “planned,” and result prose contains no ranking, gain, best value, significance,
   or causal mechanism conclusion.
8. **Diagnostic and robustness claims.** The piecewise warp, mask leakage controls,
   occlusion, and identity-switch tests are future-tense plans. Figure 2 says
   `DATA-PENDING DIAGNOSTIC SHELL` and explicitly contains no measured data.
9. **Conclusion.** The conclusion restates the proposed specification and explicitly
   says there is no local multi-person implementation or eligible MultiRep result.
   It enables no quantitative or comparative conclusion.
10. **Citation-dependent background.** Prior-work descriptions are citation-backed
    narrative claims. `claim_evidence.csv` records the citation audit as pending, so
    this paper-claim audit does not independently license their source support.

## Blocking condition and closure

The paper cannot pass a claim audit until a new synchronized MultiRep family binds
the collaborator commit, full source anchors and tests, supervision scope, dataset
release/split/checksums, evaluator semantics, raw per-person predictions and ground
truth, configurations and logs, at least three named seeds where required,
uncertainty, and all upstream audits. Only then may the generated binding file be
created and the result, protocol, abstract, or comparative claims become enabled.

The current correct conclusion is therefore `BLOCKED/data_pending`, with no hidden
or accidentally enabled unsupported performance claim.
