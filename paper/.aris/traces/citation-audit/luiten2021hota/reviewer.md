# Fresh Citation Audit: `luiten2021hota`

Verdict: **FIX**, then **PASS**. The final IJCV record was normalized with
issue 2 and author diacritics. HOTA is an aggregate multi-object-tracking score
that balances detection and association while accounting for localization;
the manuscript now names its DetA, AssA, and LocA components when claiming
failure diagnosis. The resulting use is directly supported.

Primary record: https://doi.org/10.1007/s11263-020-01375-2.

Confidence: 0.96 overall.

---

## Current-context rebind

- Agent: `/root/citation_context_rebind/rebind_luiten2021hota`
- Timestamp (UTC): `2026-08-11T19:29:48Z`
- Route: fresh zero-context, `gpt-5.6-sol`, xhigh
- Inputs disclosed to reviewer: current BibTeX entry, exact current citing paragraph/site, and primary official URLs only; no prior verdict or review history

EXISTENCE: VERIFIED

- DOI URL checked: https://doi.org/10.1007/s11263-020-01375-2 — resolves to the Springer article record.
- Publisher URL checked: https://link.springer.com/article/10.1007/s11263-020-01375-2 — live full article titled “HOTA: A Higher Order Metric for Evaluating Multi-object Tracking.”
- Crossref URL checked: https://api.crossref.org/works/10.1007/s11263-020-01375-2 — returned status `ok` for a `journal-article`.

FINAL METADATA: VERIFIED

Crossref and Springer agree with the BibTeX entry:

- Authors: Jonathon Luiten; Aljoša Ošep; Patrick Dendorfer; Philip Torr; Andreas Geiger; Laura Leal-Taixé; Bastian Leibe
- Title: “HOTA: A Higher Order Metric for Evaluating Multi-object Tracking”
- Journal: *International Journal of Computer Vision*
- Volume 129, issue 2, pages 548–578
- Version-of-record year: 2021
- DOI: `10.1007/s11263-020-01375-2`

Springer reports online publication on 8 October 2020 and assignment to volume 129, pages 548–578 (2021); Crossref additionally confirms issue 2 and print date February 2021. Thus the entry’s year `2021` correctly describes the final issue publication.

SITE sections/4_experiments.tex:41 [sha256:867578efd94cc9870bc568fd6c44f98dca28bffb29e2e11bab323465c83f6fe5]: SUPPORTS

The paper explicitly defines tracking evaluation as comparing predicted tracks against ground-truth tracks. It identifies detection, association, and localization as the three tracking-error categories; states that HOTA evaluates all three; defines DetA and AssA as separate detection- and association-accuracy scores; and defines LocA as a separate localization-accuracy score. This directly entails the cited sentence that HOTA and these components diagnose predicted-track detection, association, and localization quality against ground-truth identities and spatial support.

OVERALL: KEEP
