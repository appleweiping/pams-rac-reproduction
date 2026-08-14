# Fresh Citation Audit: `ristani2016idmetrics`

Reviewer: `/root/cite_audit_luiten2021hota/audit_ristani2016idmetrics`
(fresh zero-context xhigh child).

Verdict: **KEEP / PASS**. The final ECCV 2016 Workshops chapter, LNCS 9914,
pages 17--35, and DOI `10.1007/978-3-319-48881-3_2` are verified. Ristani et
al. introduce IDP, IDR, and IDF1 and discuss ID switches; the manuscript uses
IDF1 for identity-duration quality and ID-switch counts as a complementary
temporal-association diagnostic without claiming that the paper introduced
both. Publisher-complete fields were adopted as optional metadata enrichment.

Primary record: https://doi.org/10.1007/978-3-319-48881-3_2.

Confidence: 0.98 overall.

---

## Current-context rebind

- Agent: `/root/citation_context_rebind/rebind_ristani2016idmetrics`
- Timestamp (UTC): `2026-08-11T19:29:48Z`
- Route: fresh zero-context, `gpt-5.6-sol`, xhigh
- Inputs disclosed to reviewer: current BibTeX entry, exact current citing paragraph/site, and primary official URLs only; no prior verdict or review history

EXISTENCE: PASS

- DOI [`10.1007/978-3-319-48881-3_2`](https://doi.org/10.1007/978-3-319-48881-3_2) resolves to the Springer chapter.
- [Springer version of record](https://link.springer.com/chapter/10.1007/978-3-319-48881-3_2) exists as a conference paper, first published online 3 November 2016.
- [Crossref record](https://api.crossref.org/works/10.1007%2F978-3-319-48881-3_2) exists and identifies it as a book chapter.

FINAL METADATA: PASS

- Springer’s generated BibTeX confirms all five authors, editors Gang Hua and Hervé Jégou, book title, 2016, Springer International Publishing, Cham, pages 17–35, and ISBN `978-3-319-48881-3`.
- Springer confirms LNCS volume 9914 and the DOI. Crossref confirms the authors, publisher, pages, both print and online ISBNs, and version-of-record PDF linkage.
- The current entry’s title-case rendering “Multi-Target, Multi-Camera” differs only stylistically from Springer’s “Multi-target, Multi-camera”; there is no bibliographic identity or factual mismatch.

SITE sections/4_experiments.tex:42 [sha256:867578efd94cc9870bc568fd6c44f98dca28bffb29e2e11bab323465c83f6fe5]: SUPPORTS

Reasoning: The [full paper](https://arxiv.org/abs/1609.01775) explicitly contrasts event-based ID-switch measures with identity-based measures. It explains that ID-switch events help locate when association errors occur, whereas its proposed ID measures evaluate how long identities are correctly preserved. It defines IDF1 from identification precision and recall, describes the metric as counting correctly identified detection-frames, reports IDS and IDF1 side by side, and states that event- and identity-based measures quantify different aspects of tracker performance. This directly supports describing IDF1 as an identity-duration diagnostic and ID-switch counts as a complementary temporal-association diagnostic.

OVERALL: KEEP
