# Fresh Citation Audit: `tang2026multicounterplus`

Verdict: **FIX**, then **PASS**. Publication metadata and mechanism contexts
are correct: the dynamic Siamese query supports identity continuity and
anchor-guided reconstruction represents long and short periods. The original
dataset-version sentence exceeded the available evidence; it was replaced with
the narrower fact that the paper and repository report the same dataset size
while the repository supplies no versioned release identifier or checksums.

Primary records: https://doi.org/10.1109/TCSVT.2026.3670243 and
https://github.com/yinntag/MultiCounterPlus.

---

## Fresh exact-current-context re-review

Reviewer task: `/root/final_cite_multicounterplus` (fresh, zero-context,
same-family, `gpt-5.6-sol`, `xhigh`).

EXISTENCE: VERIFIED. The DOI resolves to the IEEE version of record, document
11421441; Crossref returns a registered journal article, and the authors'
official repository is live.

FINAL METADATA: MATCHES CURRENT BIBTEX.

- Authors: Yin Tang; Deyu Zhang; Wei Luo; Jinrui Zhang; Wei Huang; Ruihai Jing;
  Yaoxue Zhang
- Title: MultiCounter+: Toward Efficient and Robust Multi-Instance Repetitive
  Action Counting
- Journal: IEEE Transactions on Circuits and Systems for Video Technology
- Volume 36, issue 7, pages 9462--9476, 2026
- DOI: 10.1109/TCSVT.2026.3670243

SITE `appendix/a_experiment_contract.tex:64`
(`136458327432caa019935e2b3e818f61fbc52a11a6f3b6c69bca54404123d698`):
SUPPORTS -- The publisher abstract defines video-based detection, tracking, and
repetition counting for multiple human instances and explicitly states
spatial-temporal consistency plus long/short-period awareness. The official
repository specifies a dynamically updated Siamese query and anchor-guided
long/short-period module; its training configuration uses RGB-normalized frames
and supervised box, identity, period, and periodicity annotations/losses.

SITE `sections/1_introduction.tex:25`
(`383fae1ce06456a5f26ba0754d8ac1ef9a8857c4165ce8e26e8d4b0063f645a8`):
SUPPORTS -- IEEE explicitly describes spatial-temporal consistency and
long/short-period awareness. The paper introduction and repository cover
asynchronous repetitions, differing speeds, intermittent pauses, and
per-instance detection/tracking/counting, so the established-problem conclusion
is warranted.

SITE `sections/2_related_work.tex:49`
(`f1274c4eaef817dd06c3deaeb7a576948e177059ab36163e0119179bfa3ff25c`):
SUPPORTS -- The official repository directly describes the dynamically updated
Siamese query for cross-frame identity consistency and anchor-guided similarity
reconstruction for long and short periods. It also explicitly identifies
asynchronous repetitions, varying frequencies, interruptions/occlusions, and
tracking association as addressed scenarios. The proposed-setting description
is prospective and does not misattribute facts.

SITE `sections/4_experiments.tex:22`
(`fafb951b1e52fa41ab8d599e5efa1d0617a6e25dea944a89a5874b55a59af865`):
SUPPORTS -- The current repository reports exactly 1,157 videos and 52,590
periodic events and links a MultiRep download. Its current GitHub state has no
tags or releases, and the dataset entry supplies neither a version identifier
nor checksums. The requested manifest binding follows appropriately.

SITE `sections/4_experiments.tex:92`
(`f85257bccb350d1360f630afedb60dbbe250eb7127eb135e476415cf86dae3c8`):
SUPPORTS -- Calling MultiCounter+ a mandatory native baseline is a prospective
experimental choice; the citation correctly identifies the published method,
and its official repository provides implementation, checkpoints, and
evaluation instructions.

OVERALL: KEEP
