# Acceptance-contract review prompt

Fresh same-family reviewer, three-round maximum. Review the current ICASSP
acceptance contract against the designated Phase-1 inputs, without editing any
files. Every round must return concrete contract defects and terminate with
`CONTRACT_ACCEPTED: yes|no`. Re-read all contract-critical inputs after each
revision. This trace is archival; it does not constitute cross-family review.

Required inputs:

- `PAPER_ACCEPTANCE_CONTRACT.md`
- `PAPER_PLAN.md`
- `NARRATIVE_REPORT.md`
- `CLAIMS_EVIDENCE_MATRIX.md`
- `CLAIMS_FROM_RESULTS.md`
- `DERIVATION_PROVENANCE.json`

