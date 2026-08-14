# Fresh Citation Audit: `hu2022transrac`

Verdict: **KEEP / PASS**. The cited paper exists, the CVF record is the final
CVPR 2022 version, and the author list, title, venue, pages, and year match the
primary record. The paper contexts accurately describe TransRAC as using
multi-scale temporal correlation and do not overstate its scope. The final IEEE
DOI `10.1109/CVPR52688.2022.01843` was added as optional metadata enrichment.

Primary records: https://doi.org/10.1109/CVPR52688.2022.01843 and
https://openaccess.thecvf.com/content/CVPR2022/html/Hu_TransRAC_Encoding_Multi-Scale_Temporal_Correlation_With_Transformers_for_Repetitive_Action_CVPR_2022_paper.html.

Confidence: 0.97 overall.

---

## Current-context rebind

- Agent: `/root/citation_context_rebind/rebind_hu2022transrac`
- Timestamp (UTC): `2026-08-11T19:40:46Z`
- Route: fresh zero-context, `gpt-5.6-sol`, xhigh
- Inputs disclosed to reviewer: current BibTeX entry, exact current citing paragraphs/sites, and primary official URLs only; no prior verdict or review history

EXISTENCE: PASS — DOI `10.1109/CVPR52688.2022.01843` resolves to the IEEE publisher record for TransRAC (IEEE Xplore document 9879053), and the [CVF record](https://openaccess.thecvf.com/content/CVPR2022/html/Hu_TransRAC_Encoding_Multi-Scale_Temporal_Correlation_With_Transformers_for_Repetitive_Action_CVPR_2022_paper.html) exists with paper PDF and BibTeX.

FINAL METADATA: WRONG — authors, title, venue, month/year, DOI, and CVF URL match. However, the publisher-deposited DOI/VOR page range is **18991–19000**, while the current BibTeX says **19013–19022**. The latter is the conflicting CVF-site pagination, not the IEEE DOI/VOR pagination. Exact-final-metadata validation therefore fails.

SITE sections/1_introduction.tex:18 [383fae1ce06456a5f26ba0754d8ac1ef9a8857c4165ce8e26e8d4b0063f645a8]: SUPPORTS — TransRAC explicitly encodes multi-scale temporal correlations and uses density-map regression for repetitive-action counting. Its singular predicted density map/count supports its placement among learned single-stream counting methods.

SITE sections/2_related_work.tex:12 [e612d078c925a6039f1353337d0ce745937ff5e0a984500d1a4723245fa6e5d5]: SUPPORTS — the official abstract directly states both multi-scale temporal-correlation encoding and a density-map-regression method used to predict action periods/counts.

SITE sections/4_experiments.tex:94 [f85257bccb350d1360f630afedb60dbbe250eb7127eb135e476415cf86dae3c8]: SUPPORTS — TransRAC is correctly identified as a repetitive-action-counting baseline with density-map output, making adaptation to a common counting interface technically entailed. The disclosure and run-policy sentences describe the citing paper’s own protocol and are not claims attributed to Hu et al.

OVERALL: BLOCKED
