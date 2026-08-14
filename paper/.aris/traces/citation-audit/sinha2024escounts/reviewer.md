# Fresh Citation Audit: `sinha2024escounts`

Verdict: **FIX**, then **PASS**. Both the CVF ACCV 2024 accepted copy and the
Springer chapter are legitimate, but the bibliography was normalized to the
final LNCS 15474 version of record (2025, pp. 384--402, DOI
`10.1007/978-981-96-0908-6_22`). The citing prose now attributes to ESCounts
only exemplar-conditioned correspondence, exemplar-free zero-shot inference,
and its single target density/count stream; the separate multiple-action-type
claim is attributed to SkimFocusNet.

Primary records: https://doi.org/10.1007/978-981-96-0908-6_22 and
https://openaccess.thecvf.com/content/ACCV2024/html/Sinha_Every_Shot_Counts_Using_Exemplars_for_Repetition_Counting_in_Videos_ACCV_2024_paper.html.

Confidence: 0.94 overall (0.97 on method/context; both publication
manifestations are legitimate, while the Springer record is the final VOR).

---

## Current-context rebind

- Agent: `/root/citation_context_rebind/rebind_sinha2024escounts`
- Timestamp (UTC): `2026-08-11T19:48:05Z`
- Route: fresh zero-context, `gpt-5.6-sol`, xhigh
- Inputs disclosed to reviewer: current BibTeX entry, exact current citing paragraphs/sites, and primary official URLs only; no prior verdict or review history

EXISTENCE: PASS  
DOI resolves to the official [Springer chapter](https://link.springer.com/chapter/10.1007/978-981-96-0908-6_22); the [CVF ACCV record](https://openaccess.thecvf.com/content/ACCV2024/html/Sinha_Every_Shot_Counts_Using_Exemplars_for_Repetition_Counting_in_Videos_ACCV_2024_paper.html) and [Crossref record](https://api.crossref.org/works/10.1007%2F978-981-96-0908-6_22) also exist.

FINAL METADATA: PASS  
The current BibTeX accurately describes the final Springer LNCS chapter: Sinha, Stergiou, and Damen; title as given; *Computer Vision – ACCV 2024*; editors Cho, Laptev, Tran, Yao, and Zha; LNCS 15474; pp. 384–402; Springer Nature Singapore; DOI `10.1007/978-981-96-0908-6_22`. Springer’s recommended citation uses 2025, while noting first-online publication on 7 December 2024; Crossref likewise records online publication in 2024 and print publication in 2025. The CVF record’s 2024/3056–3073 metadata represents its separate open-access proceedings pagination and does not invalidate the DOI-bound LNCS entry.

SITE sections/1_introduction.tex:20 [383fae1ce06456a5f26ba0754d8ac1ef9a8857c4165ce8e26e8d4b0063f645a8]: SUPPORTS — The official abstract describes an exemplar-based model that finds exemplar correspondence in a target video and regresses the corresponding repetition locations, yielding one target repetition/count output rather than person- or category-separated streams.

SITE sections/2_related_work.tex:16 [e612d078c925a6039f1353337d0ce745937ff5e0a984500d1a4723245fa6e5d5]: SUPPORTS — The abstract explicitly states that training regresses locations of high correspondence to exemplars and simultaneously learns a latent representation used for exemplar-free, zero-shot inference.

OVERALL: KEEP
