# Fresh zero-context ICASSP contract amendment adjudication

Perform a fresh, read-only final adjudication of the post-acceptance ICASSP
contract amendment by comparing:

- accepted baseline: `D:\Project\rac-paper-writing-icassp27\PAPER_ACCEPTANCE_CONTRACT_20260814_210927.md`
- current contract: `D:\Project\rac-paper-writing-icassp27\PAPER_ACCEPTANCE_CONTRACT.md`

The only additional project inputs permitted are:

- `paper/sections/4_experiments.tex`
- `paper/main.pdf`
- `paper/.aris/traces/citation-audit-icassp/group-a-postfix/reviewer.md`

First verify that the baseline is an accepted ICASSP AC01-AC16 contract. Report
exact hashes and a semantic diff. Determine whether the current changes are
strictly amendment metadata/note plus the AC06 metric-nomenclature correction
`AP50`/`AP75` to `Period-AP50`/`Period-AP75`, without weakening evidence,
author, venue, data, ARIS, provisional, or submission gates. Verify the names in
both source and compiled PDF. Return exactly one terminal verdict token:
`AMENDMENT_ACCEPTED:YES` or `AMENDMENT_ACCEPTED:NO`.

Do not read prior contract-review traces or history. Do not edit the contract,
manuscript, PDF, or accepted baseline. Write only `prompt.md`, `reviewer.md`, and
`run.meta.json` in this amendment-final trace directory.
