# Fresh same-family acceptance-contract review

## Input trace

- Round 1: `PAPER_ACCEPTANCE_CONTRACT.md`, status `proposed`, 7,539 bytes/180 lines. SHA256 was not computed during that round and is unavailable in the transport record. Supporting inputs were the five specified plan/narrative/claims/provenance files.
- Round 2: the revised `PAPER_ACCEPTANCE_CONTRACT.md`, status `proposed-round2`. SHA256 was not computed during that round and is unavailable in the transport record. The same five supporting files were re-read.
- Round 3: `PAPER_ACCEPTANCE_CONTRACT.md` and `PAPER_ACCEPTANCE_CONTRACT_20260814_210731.md`, status `proposed-round3`; both were 9,855 bytes/208 lines, byte-identical, and matched SHA256 `acb53a078e97b2ae730aea85ae6cc0382ccd25618f54e8ecb7ac98dae7ee40ea`.

The following are the complete final-channel reviewer responses verbatim.

## Round 1

The contract correctly fail-closes missing implementation/results, separates incompatible protocols, excludes conceptual figures as empirical evidence, enforces page-five restrictions, and preserves provenance boundaries. No files were edited.

1. Fix AC05’s abstract range. It says 100–150 words, while both `PAPER_PLAN.md` and `NARRATIVE_REPORT.md` require 140–160. Use one consistent, testable range.

2. Strengthen AC04 by requiring the existing supervision-firewall scope in C10 to pass before enabling the low-label statement. It must cover loaders, pseudo-labels/router targets, auxiliary losses, tuning, checkpoint selection, early stopping, calibration, and inference—not merely frontends and evaluation.

3. Add one empirical-eligibility gate across AC05–AC08. Submission must require every result sentence, table cell, and diagnostic to map to a new synchronized MultiRep artifact family with `table_eligible=true`, exact protocol bindings, required seeds/uncertainty, and passing `experiment-audit` plus `result-to-claim`. Explicitly prohibit every C11 exclusion-register artifact. “Names its source” and “frozen diagnostic” are otherwise insufficient. Figure 2 must also bind its preregistered selection rule, paired comparison, source data, and generator.

4. Expand AC01/AC13 from their listed sections to all manuscript prose. Currently an unsupported primacy, label-free, SOTA, or equivalent claim in the Introduction or Method can evade the scanner. Require the existing paper-claim audit to map every novelty, method, empirical, comparative, and causal assertion to a ledger row whose status licenses the wording; submission fails on unmapped or blocked claims.

5. Complete AC11/AC12’s author and venue gate with author assent to the roster/order and final submission, originality/no simultaneous submission, and confirmation that every author is within ICASSP’s nine-paper limit. These are explicit ICASSP/SPS requirements, not covered by metadata agreement alone. See the official ICASSP 2027 editorial policy and IEEE SPS author requirements.

6. Make AC15 outcome-sensitive. Receiving reviews/audit files is not enough: every mandatory audit and both review rounds must be nonblocking, actionable findings must be resolved and rechecked against the final source/PDF, and the required traced cross-family review must be healthy. Permit the proof audit to return `NOT_APPLICABLE` with a reason, as already planned.

CONTRACT_ACCEPTED: no

## Round 2

The six prior revision demands are satisfied. Two contract-level gaps remain:

1. Make empirical completeness testable. AC05’s “required seeds/uncertainty” is vague, and AC06 could pass while omitting planned metrics. Require Table 1’s five principal metrics, predicted-track diagnostics where applicable, and C13’s exact seed/statistical package. Unavailable contextual values should be explicit `N/A`, not silently omitted.

2. Make the final assurance outcome explicit. AC15 requires the ARIS verifier to run but does not require it to pass. Require a nonblocking/accepted verifier result, and define a “healthy” cross-family review as substantively nonblocking with findings resolved—not merely a successful response and trace.

CONTRACT_ACCEPTED: no

## Round 3

- Both files are byte-identical and match SHA256 `acb53a078e97b2ae730aea85ae6cc0382ccd25618f54e8ecb7ac98dae7ee40ea`.
- AC06 now specifies all five principal metrics, predicted-track diagnostics, explicit `N/A`, and the complete C13 seed/statistical package.
- AC15 now requires verifier exit 0 with no blocking verdict and a substantively nonblocking cross-family review whose findings are resolved and rechecked.
- AC01–AC16 are internally consistent, falsifiable, and fail closed for submission.
- The present data-pending state does not prevent accepting the contract itself; it correctly prevents submission readiness.

CONTRACT_ACCEPTED: yes

