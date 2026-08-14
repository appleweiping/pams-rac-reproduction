# Fresh Citation Audit: `ma2026detrc`

Route: fresh zero-context, same-family reviewer

Verdict: **KEEP**. Confidence: **0.99**.

The bibliography matches the final IJCV 2026 version of record: Xiaoxuan Ma
is first author, volume 134, issue 8, article 364, DOI
`10.1007/s11263-026-02939-4`. All three local contexts are supported. The
method uses dynamically initialized action queries, assumes one primary
repetitive action type per video, and has linear (O(TC)) rather than
constant-time complexity. The conditional experiment-admission sentence is the
manuscript's own audit policy and does not attribute an unsupported result.

Primary records:

- https://doi.org/10.1007/s11263-026-02939-4
- https://link.springer.com/article/10.1007/s11263-026-02939-4
- https://api.crossref.org/works/10.1007/s11263-026-02939-4
- https://arxiv.org/abs/2403.01543
- https://github.com/lizishi/DeTRC

---

## Current-context rebind

- Agent: `/root/citation_context_rebind/rebind_ma2026detrc`
- Timestamp (UTC): `2026-08-11T19:43:42Z`
- Route: fresh zero-context, `gpt-5.6-sol`, xhigh
- Inputs disclosed to reviewer: current BibTeX entry, exact current citing paragraphs/sites, and primary official URLs only; no prior verdict or review history

EXISTENCE: SUPPORTS

- DOI resolves to the Springer article.
- Springer identifies the version of record, published 25 July 2026.
- Crossref has a registered journal-article record.
- arXiv 2403.01543 is the matching preprint.
- GitHub hosts the official public PyTorch implementation.

FINAL METADATA: SUPPORTS

- Authors: Xiaoxuan Ma; Zishi Li; Qiuyan Shang; Wentao Zhu; Hai Ci; Yu Qiao; Yizhou Wang
- Title: *Efficient Action Counting with Dynamic Queries*
- Journal: *International Journal of Computer Vision*
- Volume: 134
- Issue: 8
- Article number: 364
- Year: 2026
- DOI: 10.1007/s11263-026-02939-4
- Online/VOR date: 25 July 2026
- Print issue date: August 2026

All current BibTeX fields match the Springer VOR and Crossref record.

`SITE sections/1_introduction.tex:18 [383fae1ce06456a5f26ba0754d8ac1ef9a8857c4165ce8e26e8d4b0063f645a8]: SUPPORTS` — Springer and arXiv describe an action-query representation and dynamic action-query strategy. Springer also explicitly states that each video is assumed to contain only one primary repetitive action type, supporting this citation’s role in the single-stream characterization.

`SITE sections/2_related_work.tex:14 [e612d078c925a6039f1353337d0ce745937ff5e0a984500d1a4723245fa6e5d5]: SUPPORTS` — The official abstract says action queries localize repeated cycles and are dynamically defined from video content; Springer Note 2 explicitly gives the single-primary-action assumption.

`SITE sections/4_experiments.tex:95 [f85257bccb350d1360f630afedb60dbbe250eb7127eb135e476415cf86dae3c8]: SUPPORTS` — The official GitHub repository provides implementation files, installation instructions, training and testing commands, and a pretrained-model link. The site only makes baseline inclusion conditional on a later runnability/output audit, so it does not overclaim successful execution.

OVERALL: KEEP
