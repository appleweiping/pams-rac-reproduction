# Fresh Citation Audit: `zhao2025skimfocus`

Verdict: **FIX**, then **PASS**. The final IJCV article and its substantive
coarse-to-fine and exemplar-specified multiple-action-category claims are
verified. The Springer/Crossref version of record structures the final author
as given name `Feng`, family name `Bin`; the BibTeX was made VOR-faithful as
`Bin, Feng`, while noting that arXiv, DBLP, and the author's ORCID use the
opposite identity ordering. Related Work was rewritten to say that one target
category is specified, and MultiCounter was added to support the separate
claim about MultiRep's per-person simultaneous-instance task.

Primary records: https://doi.org/10.1007/s11263-025-02471-x,
https://link.springer.com/article/10.1007/s11263-025-02471-x, and
https://doi.org/10.3233/FAIA240494.

Confidence: 0.93 overall; 0.98 for the publication fields and substantive
claims, with the remaining uncertainty confined to the publisher name-order
anomaly.

---

## Current-context rebind

- Agent: `/root/citation_context_rebind/rebind_zhao2025skimfocus`
- Timestamp (UTC): `2026-08-11T19:48:05Z`
- Route: fresh zero-context, `gpt-5.6-sol`, xhigh
- Inputs disclosed to reviewer: current BibTeX entry, exact current citing paragraphs/sites, and primary official URLs only; no prior verdict or review history

EXISTENCE: PASS — The publisher’s [version-of-record page](https://link.springer.com/article/10.1007/s11263-025-02471-x) exists and identifies the article as published 3 June 2025, VOR 3 June 2025, issue date September 2025.

FINAL METADATA: PASS — The [Crossref DOI record](https://api.crossref.org/works/10.1007/s11263-025-02471-x) matches the current BibTeX: title, nine authors, *International Journal of Computer Vision*, volume 133, issue 9, pages 6347–6361, year 2025, and DOI `10.1007/s11263-025-02471-x`. Hyphen/en-dash differences are normalization only.

SITE sections/1_introduction.tex:21 [383fae1ce06456a5f26ba0754d8ac1ef9a8857c4165ce8e26e8d4b0063f645a8]: SUPPORTS — The publisher abstract says Multi-RepCount contains videos with multiple repetitive motions and that SkimFocusNet performs specified action counting by referencing an exemplary video, directly supporting “exemplar-selected multiple action types.”

SITE sections/2_related_work.tex:21 [e612d078c925a6039f1353337d0ce745937ff5e0a984500d1a4723245fa6e5d5]: SUPPORTS — The title and abstract describe coarse skimming/global contextual guidance followed by fine frame-level focusing. The abstract also states that Multi-RepCount contains multiple repetitive motions and evaluates counting one specified action type using an exemplar video.

OVERALL: KEEP
