# ARIS ICASSP citation audit — group A

## Assignment

Perform a fresh, zero-context, read-only citation audit of the current scientific manuscript in `paper/main.tex`, every `paper/sections/*.tex`, and `paper/references.bib` for exactly these keys:

`levy2015live`, `runia2018realworld`, `dwibedi2020repnet`, `hu2022transrac`, `yao2025poserac`, `gao2026pams`, `tang2024multicounter`, `tang2026multicounterplus`.

For every key, independently determine:

1. whether the BibTeX entry exists;
2. whether author, title, venue, year, pages/volume/issue, DOI, and URL metadata are correct;
3. whether every current citation context is actually supported by the cited primary source.

## Independence and source constraints

- Do not inspect or reuse any prior audit, review, history, trace, or log.
- Do not modify `paper/references.bib` or any manuscript `.tex` file.
- Browse the network. Use only publisher/conference records, CVF/Springer/IEEE/IOS Press, Crossref, DBLP, and author-official papers/code as verdict evidence.
- Search-engine results may discover sources but are not themselves evidence. Google Scholar is not a verification backbone.
- Record exact current file, line range, and paragraph SHA-256 for every citation context, plus all evidence URLs and per-key existence/metadata/context verdicts.
- Write only this audit bundle under `paper/.aris/traces/citation-audit-icassp/group-a/`.

## Hash convention

`paragraph_sha256` and `bib_entry_sha256` are SHA-256 values over the UTF-8 bytes of the exact current paragraph/entry payload, preserving its internal line endings and excluding the blank-line separator around it. Whole-file hashes are SHA-256 over exact file bytes.

## Requested terminal verdict

Return one of `PASS`, `BLOCKED`, or `FAIL`. `FAIL` means at least one supported check found an error; `BLOCKED` is reserved for a check that could not be completed with available authoritative evidence.
