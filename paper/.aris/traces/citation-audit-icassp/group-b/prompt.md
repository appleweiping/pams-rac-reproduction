# ARIS ICASSP citation audit — group B

Fresh/zero-context, read-only audit of the current scientific manuscript in
`D:\Project\rac-paper-writing-icassp27`.

Audit `paper/references.bib`, `paper/main.tex`, and every
`paper/sections/*.tex` occurrence of these eight citation keys:

- `ma2026detrc`
- `wang2026d2stx`
- `welch1967fft`
- `lomb1976unequal`
- `griffin1984stft`
- `shazeer2017moe`
- `luiten2021hota`
- `ristani2016idmetrics`

For every key, independently verify:

1. the BibTeX entry exists;
2. author, title, venue, year, DOI and other supplied metadata are correct;
3. every current citation context is actually supported by the cited source.

Internet browsing is mandatory. Technical verification must rely on publisher
or conference pages, CVF/Springer/IEEE, DBLP/Crossref, author-hosted papers, or
official code. Google Scholar may be used only for discovery. Do not read or
reuse any prior audit, history, or log. Record exact current file/line and
paragraph SHA-256 values, sites and source URLs, and separate
existence/metadata/context verdicts. Do not modify the bibliography or
scientific manuscript; report errors only.

Write only:

- `paper/.aris/traces/citation-audit-icassp/group-b/prompt.md`
- `paper/.aris/traces/citation-audit-icassp/group-b/reviewer.md`
- `paper/.aris/traces/citation-audit-icassp/group-b/run.meta.json`

The metadata must include agent ID, model, reasoning, `fork: none`, all input
hashes, timestamp, same-family status, and provisional status.
