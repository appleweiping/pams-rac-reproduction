# Fresh Citation Audit: `runia2018realworld`

Verdict: **FIX metadata; contexts PASS**. The final CVPR 2018 paper directly
supports time-varying optical-flow differentials, continuous wavelet analysis,
and non-stationary repetition. Authors, title, venue, pages, month, year, and
CVF URL were correct; the canonical IEEE DOI was added.

Primary records: https://doi.org/10.1109/CVPR.2018.00939 and
https://openaccess.thecvf.com/content_cvpr_2018/html/Runia_Real-World_Repetition_Estimation_CVPR_2018_paper.html.

Confidence: 0.99.

---

## Current-context rebind

- Agent: `/root/citation_context_rebind/rebind_runia2018realworld`
- Timestamp (UTC): `2026-08-11T19:35:27Z`
- Route: fresh zero-context, `gpt-5.6-sol`, xhigh
- Inputs disclosed to reviewer: current BibTeX entry, exact current citing paragraphs/sites, and primary official URLs only; no prior verdict or review history

EXISTENCE: PASS

URLs actually checked:

- [DOI](https://doi.org/10.1109/CVPR.2018.00939): resolves to IEEE Xplore document 8579037.
- [CVF record](https://openaccess.thecvf.com/content_cvpr_2018/html/Runia_Real-World_Repetition_Estimation_CVPR_2018_paper.html): HTTP 200; official accepted-version record and paper abstract.
- [Crossref record](https://api.crossref.org/works/10.1109/CVPR.2018.00939): `status: ok`; identifies a proceedings article and links the IEEE content as `content-version: vor`.
- [DBLP record](https://dblp.org/rec/conf/cvpr/RuniaSS18): canonical record exists and redirects successfully to its HTML representation.

FINAL METADATA: PASS

The records confirm:

- Authors: Tom F. H. Runia; Cees G. M. Snoek; Arnold W. M. Smeulders
- Title: *Real-World Repetition Estimation by Div, Grad and Curl*
- Venue: CVPR 2018 / *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*
- Date: June 2018
- Pages: 9009–9017
- Publisher: IEEE
- DOI: `10.1109/CVPR.2018.00939`

The CVF URL is the official open-access accepted-version page; CVF explicitly distinguishes it from the final IEEE Xplore proceedings version. The DOI and Crossref VOR link correctly identify that final version. No material BibTeX correction is needed.

SITE 1 — sections/1_introduction.tex:16 — sha256:383fae1ce06456a5f26ba0754d8ac1ef9a8857c4165ce8e26e8d4b0063f645a8: SUPPORTS

Reasoning: The citation is locally attached to “motion-frequency analysis.” The paper measures time-varying optical flow and its gradient, divergence, and curl, then applies wavelet/time-frequency analysis for repetition estimation and counting. This is a faithful concise characterization. Its method produces a video repetition estimate/count rather than multiple person-wise streams, so it is also consistent with the paragraph’s broader “largely assume one counted stream” summary.

SITE 2 — sections/2_related_work.tex:8 — sha256:e612d078c925a6039f1353337d0ce745937ff5e0a984500d1a4723245fa6e5d5: SUPPORTS

Reasoning: This claim closely tracks the official abstract: the method measures time-varying flow and its differentials—gradient, divergence, and curl—and adopts the wavelet transform specifically to handle non-static and non-stationary video dynamics. “Motion-centric methods used time-varying flow differentials and wavelets to address non-stationary repetition” is directly entailed.

OVERALL: KEEP
