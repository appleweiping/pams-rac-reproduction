# Fresh paper-claim audit prompt

Perform a zero-context ARIS `paper-claim-audit` of the current ICASSP manuscript in
`D:\Project\rac-paper-writing-icassp27`. Do not inspect or reuse any earlier paper-claim
audit, audit history, or logs. The authoritative rendered input is
`paper/main.pdf` with SHA-256
`873c83da0dbf4b24a6e1199f3dcd980e17d0b032580e0591f88a7c545ec67a45`.

Read the current active TeX dependency closure and the following evidence controls:

- `paper/evidence/results_manifest.json`
- `paper/evidence/method_manifest.yaml`
- `paper/evidence/claim_evidence.csv`
- `paper/generated/README.md`
- `CLAIMS_FROM_RESULTS.md`
- `RESULT_TO_CLAIM.json`

Inventory empirical, method, and protocol claims in the title, abstract,
contributions, method, experiments, and conclusion. Check every concrete numeric
result, every `\R`, `\Protocol`, and `\ControlledAbstractResult` use, the existence
and provenance of generated bindings, machine-readable raw evidence, and source
bindings. Do not edit manuscript prose. Do not hide `PENDING` or `SYNC-REQUIRED` as
a generic failure: if eligible data are absent, report `BLOCKED/data_pending` and
state explicitly whether any unsupported performance claim is enabled.

Write the audit to `paper/PAPER_CLAIM_AUDIT.md` and
`paper/PAPER_CLAIM_AUDIT.json`, with this trace directory containing `prompt.md`,
`reviewer.md`, and `run.meta.json`. Validate JSON and all audited hashes before
reporting completion.
