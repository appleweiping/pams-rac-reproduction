# Fresh Citation Audit: `yao2025poserac`

Route: fresh zero-context, same-family reviewer

Verdict: **FIX citing prose; KEEP bibliography entry unchanged.**

The final Springer chapter exists and the metadata in `references.bib` matches
the canonical 2025 publication: Ziyu Yao and Yuexian Zou, LNCS 15290,
pp. 255--270, DOI `10.1007/978-981-96-6588-4_18`. One Related Work context
was imprecise because PoseRAC uses foundation generative models to synthesize
pose-conditioned training data rather than a generic “foundation-model
representation.” The manuscript was changed to describe the synthetic-data
role and CLIP text embeddings directly; all local contexts then pass.

Implementation caveat: the public `MiracleDance/PoseRAC` repository implements
the distinct 2023 three-author Pose Saliency Transformer paper. A synchronized
baseline run must identify which implementation and paper it actually uses.

Primary records:

- https://doi.org/10.1007/978-981-96-6588-4_18
- https://link.springer.com/chapter/10.1007/978-981-96-6588-4_18
- https://api.crossref.org/works/10.1007/978-981-96-6588-4_18
- https://dblp.org/rec/conf/iconip/YaoZ24

Confidence: 0.99 metadata; 0.96 overall.

---

## Current-context rebind

- Agent: `/root/citation_context_rebind/rebind_yao2025poserac`
- Timestamp (UTC): `2026-08-11T19:53:36Z`
- Route: fresh zero-context, `gpt-5.6-sol`, xhigh
- Inputs disclosed to reviewer: current BibTeX entry, exact current citing paragraphs/sites, and primary official URLs only; no prior verdict or review history

EXISTENCE: PASS

- DOI resolves to the Springer chapter.
- [Springer VOR](https://link.springer.com/chapter/10.1007/978-981-96-6588-4_18), [Crossref](https://api.crossref.org/works/10.1007/978-981-96-6588-4_18), and [DBLP](https://dblp.org/rec/conf/iconip/YaoZ24) all identify the same work.

FINAL METADATA: PASS

- Authors: Ziyu Yao; Yuexian Zou
- Title: *PoseRAC: Enhancing Repetitive Action Counting with Salient Poses*
- Venue: ICONIP 2024, Proceedings Part V; LNCS 15290
- Pages: 255–270
- Publisher: Springer Nature Singapore
- Final publication year: 2025
- DOI: `10.1007/978-981-96-6588-4_18`
- Note: DBLP displays conference year 2024; Springer gives “First Online: 24 June 2025” and cites the chapter as 2025. The current BibTeX correctly uses the final publication year.

SITE `sections/1_introduction.tex:33` [`sha256:a815993fe282cba936d400aaeda9696b34b1ab4fffe16b02ca98a241b85edb5a`]: SUPPORTS — Springer explicitly says PoseRAC represents actions with salient poses, performs zero-shot prediction using foundation generative models, and counts unseen actions in an open-set setting using a text encoder.

SITE `sections/2_related_work.tex:28` [`sha256:0426ac52a89964d27d7c715838980245eb397a7bbfc7381b0eab379fe10d4836`]: WEAK — The VOR abstract supports salient poses, foundation generative models, zero-shot prediction without a benchmark training set, and an off-the-shelf text encoder for open-set counting. However, the accessible official record does not explicitly establish the stronger formulations “synthesize pose-conditioned training data” or “CLIP text embeddings”; it only separately references CLIP in a note. Exact entailment therefore fails.

SITE `sections/4_experiments.tex:94` [`sha256:f85257bccb350d1360f630afedb60dbbe250eb7127eb135e476415cf86dae3c8`]: SUPPORTS — The citation correctly identifies PoseRAC as a repetitive-action-counting method suitable for inclusion as a baseline; adapting it to a common track interface is the manuscript’s proposed experimental action, not a claim attributed to the source.

OVERALL: BLOCKED

---

## Fresh exact-current-context re-review after Introduction correction

Reviewer task: `/root/final_cite_poserac` (fresh, zero-context, same-family,
`gpt-5.6-sol`, `xhigh`).

EXISTENCE: VERIFIED -- The DOI resolves to the Springer chapter; Springer,
Crossref, and DBLP independently identify the same work.

FINAL METADATA: Ziyu Yao and Yuexian Zou, "PoseRAC: Enhancing Repetitive
Action Counting with Salient Poses," Neural Information Processing: 31st
International Conference, ICONIP 2024, Proceedings, Part V, LNCS 15290,
pages 255--270, Springer Nature Singapore, 2025,
DOI 10.1007/978-981-96-6588-4_18. Springer and Crossref confirm the 2025
publication year; DBLP's ICONIP 2024 label is the conference year, not a
conflict.

SITE `sections/1_introduction.tex:33`
(`a1cdac1a21a419673c557eec3b5f2248716a6a7cff8c15518ffb6e6631487066`):
SUPPORTS -- The Springer abstract explicitly describes salient-pose counting,
zero-shot prediction without a training set, and open-set counting of unseen
actions.

SITE `sections/2_related_work.tex:28`
(`0426ac52a89964d27d7c715838980245eb397a7bbfc7381b0eab379fe10d4836`):
SUPPORTS -- Springer states that PoseRAC models actions via salient poses,
leverages foundation generative models, performs zero-shot prediction without
training data, and uses an off-the-shelf text encoder for open-set unseen-action
counting; its note identifies the CLIP framing.

SITE `sections/4_experiments.tex:94`
(`f85257bccb350d1360f630afedb60dbbe250eb7127eb135e476415cf86dae3c8`):
SUPPORTS -- The official records establish PoseRAC as a repetitive-action-
counting method, appropriately identifying the proposed adapted baseline; the
adaptation itself is the manuscript's experimental protocol.

OVERALL: KEEP
