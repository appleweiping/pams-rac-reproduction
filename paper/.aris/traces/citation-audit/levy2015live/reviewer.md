# Fresh Citation Audit: `levy2015live`

Verdict: **FIX citation placement; KEEP bibliography metadata**. The ICCV 2015
record, pages, authors, and DOI are correct. Levy and Wolf directly support
online local cycle estimation over sequential shifting blocks; they do not
introduce the motion-frequency/wavelet mechanism in the adjacent clause. Both
citations were split so this key now attaches only to the entailed claim.

Primary records: https://doi.org/10.1109/ICCV.2015.346 and
https://openaccess.thecvf.com/content_iccv_2015/html/Levy_Live_Repetition_Counting_ICCV_2015_paper.html.

Confidence: 0.99 metadata; 0.98 context.

---

## Current-context rebind

- Agent: `/root/citation_context_rebind/rebind_levy2015live`
- Timestamp (UTC): `2026-08-11T19:35:27Z`
- Route: fresh zero-context, `gpt-5.6-sol`, xhigh
- Inputs disclosed to reviewer: current BibTeX entry, exact current citing paragraphs/sites, and primary official URLs only; no prior verdict or review history

EXISTENCE: PASS

- DOI checked: https://doi.org/10.1109/ICCV.2015.346 — resolves to IEEE Xplore document 7410703.
- CVF checked: https://openaccess.thecvf.com/content_iccv_2015/html/Levy_Live_Repetition_Counting_ICCV_2015_paper.html — HTTP 200; official ICCV paper page and PDF.
- Crossref checked: https://api.crossref.org/works/10.1109/ICCV.2015.346 — HTTP 200; registered proceedings article.
- DBLP checked: https://dblp.org/rec/conf/iccv/LevyW15 — HTTP 200; canonical record `conf/iccv/LevyW15`.

FINAL METADATA: PASS

Crossref’s IEEE registration reports:

- Authors: Ofir Levy; Lior Wolf
- Title: “Live Repetition Counting”
- Container: *2015 IEEE International Conference on Computer Vision (ICCV)*
- Publisher/type: IEEE; proceedings article
- Pages: 3020–3028
- Published: December 2015
- DOI: `10.1109/ICCV.2015.346`
- Event: Santiago, Chile, 7–13 December 2015

DBLP independently agrees on authors, title, year, pages, DOI, and IEEE Computer Society publisher. CVF uses the current entry’s proceedings-style booktitle and December 2015 metadata. CVF states its copy is the accepted version, identical except for its watermark, while the DOI resolves to the IEEE version of record. DOI case differences in Crossref are immaterial.

SITE 1: SUPPORTS  
file: sections/1_introduction.tex  
line: 15  
context_sha256: sha256:383fae1ce06456a5f26ba0754d8ac1ef9a8857c4165ce8e26e8d4b0063f645a8

Reasoning: The paper explicitly says the method runs online, sequentially examines blocks of 20 non-consecutive frames, estimates cycle length within each block using a CNN, and integrates estimates over time. This directly entails “online local cycle estimation.” Its 2015 date also supports the historical placement.

SITE 2: SUPPORTS  
file: sections/2_related_work.tex  
line: 6  
context_sha256: sha256:e612d078c925a6039f1353337d0ce745937ff5e0a984500d1a4723245fa6e5d5

Reasoning: The learned CNN predicts cycle length for each fixed 20-frame video block; blocks are processed through shifting windows in an online live-video scheme. Thus “estimated a local cycle length in an online window” is an accurate concise description.

OVERALL: KEEP
