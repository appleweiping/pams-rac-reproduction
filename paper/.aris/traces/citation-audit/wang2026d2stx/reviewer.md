# Fresh Citation Audit: `wang2026d2stx`

Verdict: **FIX**, then **PASS**. The official CVF Findings record verifies the
paper, title, authors, venue, June 2026 date, and pages 8205--8214. All method
contexts about RGB/pose branches and decoupled spatial/temporal cross-attention
are directly supported. The final-author initials were normalized to `Bruce
X.B. Yu`. Because the linked official repository currently contains no
runnable implementation or output contract, the experiment text now makes
entry conditional on a runnable public implementation and audited semantics.
No DOI was added because neither CVF, Crossref, nor DBLP supplied a verified
one at audit time.

Primary record: https://openaccess.thecvf.com/content/CVPR2026F/html/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.html.

Confidence: 0.97 overall.

---

## Current-context rebind

- Agent: `/root/citation_context_rebind/rebind_wang2026d2stx`
- Timestamp (UTC): `2026-08-11T19:53:36Z`
- Route: fresh zero-context, `gpt-5.6-sol`, xhigh
- Inputs disclosed to reviewer: current BibTeX entry, exact current citing paragraphs/sites, and primary official URLs only; no prior verdict or review history

EXISTENCE: **SUPPORTS**

- The exact [CVF HTML record](https://openaccess.thecvf.com/content/CVPR2026F/html/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.html) and [official PDF](https://openaccess.thecvf.com/content/CVPR2026F/papers/Wang_D2-STX_Decoupling_Spatial-Temporal_Cross-Attention_for_Dual-branch_Repetitive_Action_Counting_CVPRF_2026_paper.pdf) both return HTTP 200.
- The [official project repository](https://github.com/ZJU-Emerging-AI-Lab/D2-STX) exists publicly.

FINAL METADATA: **SUPPORTS**

- Official title: *D^2-STX: Decoupling Spatial-Temporal Cross-Attention for Dual-branch Repetitive Action Counting*.
- Authors: Xiaoai Wang; Hang Wang; Yan Liu; Huan Hu; Bruce X.B. Yu.
- Venue: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings*.
- Date/pages: June 2026, pp. 8205–8214.
- The supplied BibTeX matches all substantive official fields; its math rendering of `D²-STX` is equivalent to the official `D^2-STX`.
- Version status: the CVF copy is explicitly the accepted Open Access version, identical to the accepted paper except for its watermark; the page states that the proceedings VOR is on IEEE Xplore. This does not create a metadata conflict.

`SITE sections/1_introduction.tex:34 [sha256:a815993fe282cba936d400aaeda9696b34b1ab4fffe16b02ca98a241b85edb5a]: SUPPORTS`  
The abstract and paper body describe complementary video/appearance and pose streams in a dual-branch framework; the paper also explicitly discusses “RGB video and pose fusion.” This directly supports “combines RGB and pose streams.”

`SITE sections/2_related_work.tex:31 [sha256:0426ac52a89964d27d7c715838980245eb397a7bbfc7381b0eab379fe10d4836]: SUPPORTS`  
The paper explicitly defines frame-wise spatial cross-attention plus joint-wise and pixel-wise temporal cross-attention and says that D²-STX decouples the spatial and temporal dimensions while fusing video and pose branches.

`SITE sections/4_experiments.tex:96 [sha256:f85257bccb350d1360f630afedb60dbbe250eb7127eb135e476415cf86dae3c8]: SUPPORTS`  
The paper establishes D²-STX as a repetitive-action-counting method, so treating it as a candidate baseline is appropriate. The site does not assert that runnable code currently exists; it expressly conditions inclusion on that audit. That condition matters because the linked GitHub repository currently exposes only `README.md` and `LICENSE`, not a runnable implementation.

OVERALL: KEEP

---

## Fresh exact-current-context re-review after Introduction correction

Reviewer task: `/root/final_cite_d2stx` (fresh, zero-context, same-family,
`gpt-5.6-sol`, `xhigh`).

EXISTENCE: VERIFIED -- The paper appears in the official CVPR 2026 Findings
record, with an official paper PDF and author repository.

FINAL METADATA: The supplied BibTeX matches the official record: Xiaoai Wang,
Hang Wang, Yan Liu, Huan Hu, and Bruce X.B. Yu; "D^2-STX: Decoupling
Spatial-Temporal Cross-Attention for Dual-branch Repetitive Action Counting";
CVPR Findings, June 2026, pages 8205--8214. The superscript LaTeX is a harmless
equivalent of the CVF rendering.

SITE `sections/1_introduction.tex:34`
(`a1cdac1a21a419673c557eec3b5f2248716a6a7cff8c15518ffb6e6631487066`):
SUPPORTS -- The paper explicitly uses dual video/RGB and 2D-pose streams.

SITE `sections/2_related_work.tex:31`
(`0426ac52a89964d27d7c715838980245eb397a7bbfc7381b0eab379fe10d4836`):
SUPPORTS -- The paper describes RGB/video-pose fusion through a dual-branch
decoupled spatial-temporal cross-attention mechanism.

SITE `sections/4_experiments.tex:96`
(`f85257bccb350d1360f630afedb60dbbe250eb7127eb135e476415cf86dae3c8`):
SUPPORTS -- The citation identifies D^2-STX as an RAC method, while the official
repository currently contains only README/LICENSE and no runnable
implementation; the stated conditional audit gate is therefore appropriate and
presently excludes it.

OVERALL: KEEP
