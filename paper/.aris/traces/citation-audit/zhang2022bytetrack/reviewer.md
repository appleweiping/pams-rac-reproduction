# Fresh Citation Audit: `zhang2022bytetrack`

Verdict: **FIX**, then **PASS**. The final ECCV 2022 chapter exists, and the
authors, title, pages, publisher, year, and DOI in the draft match Springer's
record. The BibTeX entry omitted the LNCS volume; volume 13682 was added. The
paper only uses ByteTrack as a representative predicted-tracking frontend and
does not attribute unsupported counting claims to it.

Primary record: https://doi.org/10.1007/978-3-031-20047-2_1.

Confidence: 0.99 overall.

---

## Current-context rebind

- Agent: `/root/citation_context_rebind/rebind_zhang2022bytetrack`
- Timestamp (UTC): `2026-08-11T19:31:03Z`
- Route: fresh zero-context, `gpt-5.6-sol`, xhigh
- Inputs disclosed to reviewer: current BibTeX entry, exact current citing paragraph/site, and primary official URLs only; no prior verdict or review history

EXISTENCE: PASS

The DOI resolves to a live Springer conference-paper record for ByteTrack. The Springer page identifies it as a conference paper, first published 23 October 2022, pages 1–21.

FINAL METADATA: FAIL

Springer’s official BibTeX export confirms the entry type, all nine authors, book title, year, publisher, pages, and DOI. The Springer page additionally confirms LNCS volume 13682. However, the version-of-record title is **“ByteTrack: Multi-object Tracking by Associating Every Detection Box”**, while the current entry has **“Multi-Object”**. Under the requested exact-metadata standard, that capitalization/orthography mismatch fails this axis.

URLs checked:

- [DOI resolver](https://doi.org/10.1007/978-3-031-20047-2_1)
- [Springer version of record](https://link.springer.com/chapter/10.1007/978-3-031-20047-2_1)
- [Springer official BibTeX export](https://citation-needed.springer.com/v2/references/10.1007/978-3-031-20047-2_1?flavour=citation&format=bibtex)

SITE appendix/a_experiment_contract.tex:180 [sha256:5509ce09517236115d4a46643b81ccb1138afff5164239174576928f89a7804b]: SUPPORTS

The paper explicitly presents ByteTrack as a tracker and states that source code and pretrained models were released. That supports identifying ByteTrack as the possible common tracker. Freezing the exact code/checkpoint revision and not pre-selecting the frontend are prospective experimental-policy choices, not factual claims attributed to the paper.

OVERALL: BLOCKED — FINAL METADATA: title must use “Multi-object,” not “Multi-Object.”
