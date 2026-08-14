# Fresh Citation Audit: `tang2024multicounter`

Verdict: **KEEP / PASS** (high confidence). The ECAI 2024 metadata and DOI
match the formal publication. Every local context is entailed: MultiCounter
defines multi-instance repetition counting, introduces MultiRep, and jointly
models detection, association, periodic intervals, and person-wise counts.

Primary records: https://doi.org/10.3233/FAIA240494 and
https://arxiv.org/abs/2409.04035.

---

## Current-context rebind

- Agent: `/root/citation_context_rebind/rebind_tang2024multicounter`
- Timestamp (UTC): `2026-08-11T19:59:01Z`
- Route: fresh zero-context, `gpt-5.6-sol`, xhigh
- Inputs disclosed to reviewer: current BibTeX entry, exact current citing paragraphs/sites, and primary official URLs only; no prior verdict or review history

EXISTENCE: SUPPORTS

The work exists across all three official records. The [DOI](https://doi.org/10.3233/FAIA240494) resolves to the IOS Press VOR landing page; [arXiv](https://arxiv.org/abs/2409.04035) records v1 on 6 September 2024 and acceptance at ECAI 2024; [Crossref](https://api.crossref.org/works/10.3233/FAIA240494) returns `status: ok` and identifies a VOR resource.

FINAL METADATA: SUPPORTS

- Authors: Yin Tang; Wei Luo; Jinrui Zhang; Wei Huang; Ruihai Jing; Deyu Zhang
- Title: *MultiCounter: Multiple Action Agnostic Repetition Counting in Untrimmed Videos*
- Conference/container: *ECAI 2024*
- Series: *Frontiers in Artificial Intelligence and Applications*
- Volume: 392
- Pages: 242–249
- Publisher: IOS Press
- Publication: 2024; Crossref print/issued date 2024-10-16, IOS online date 2024-10-17
- DOI: `10.3233/FAIA240494`
- Crossref type: `book-chapter`; `@inproceedings` remains appropriate for this conference-proceedings paper
- The supplied BibTeX matches the VOR metadata on all populated fields.

SITE appendix/a_experiment_contract.tex:63 [sha256:136458327432caa019935e2b3e818f61fbc52a11a6f3b6c69bca54404123d698]: SUPPORTS — The paper takes RGB video, represents individual people through persistent instance queries, predicts per-instance locations, periodicity and period velocity, derives repetition proposals/counts, and trains these outputs with supervised instance and period losses. “Joint association” fairly summarizes its query-based temporal association and tracking.

SITE sections/1_introduction.tex:24 [sha256:383fae1ce06456a5f26ba0754d8ac1ef9a8857c4165ce8e26e8d4b0063f645a8]: SUPPORTS — The paper explicitly says it formally proposes MRAC, simultaneously detects, tracks and counts multiple human instances, and handles asynchronous actions, variable cycle lengths and pauses. Person identity is represented through tracked instance queries; interval and count outputs are defined per person.

SITE sections/2_related_work.tex:21 [sha256:e612d078c925a6039f1353337d0ce745937ff5e0a984500d1a4723245fa6e5d5]: SUPPORTS — MultiRep is expressly an MRAC dataset containing videos with multiple simultaneous human instances, and its metrics aggregate instance-level/person-wise counts.

SITE sections/2_related_work.tex:46 [sha256:f1274c4eaef817dd06c3deaeb7a576948e177059ab36163e0119179bfa3ff25c]: SUPPORTS — MultiCounter defines MRAC and introduces MultiRep. Its jointly trained architecture covers instance detection/location, temporal association/tracking, period localization and person-wise counting. The paper explicitly discusses asynchronism, inconsistent velocity, pauses and tracked identity continuity.

SITE sections/4_experiments.tex:22 [sha256:fafb951b1e52fa41ab8d599e5efa1d0617a6e25dea944a89a5874b55a59af865]: SUPPORTS — The paper reports exactly 1,157 synthetic MultiRep videos, 52,590 periodic events and a 7:2:1 train/validation/test split. Remaining release-integrity requirements are prospective experiment-contract language, not claims attributed solely to this citation.

SITE sections/4_experiments.tex:92 [sha256:f85257bccb350d1360f630afedb60dbbe250eb7127eb135e476415cf86dae3c8]: SUPPORTS — MultiCounter is the native end-to-end MRAC method introduced and evaluated by this paper, so naming it as a mandatory native baseline is fully entailed. The surrounding disclosure rules are protocol choices rather than attributed factual claims.

OVERALL: KEEP
