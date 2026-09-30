# Acceptance-contract amendment review: MultiRep metric nomenclature

`AMENDMENT_ACCEPTED:NO`

Review classification: **provisional / same-family**.

## Determination

The metric-label correction itself is non-semantic and does not weaken the
evidence gate. The active experiment source and rendered PDF consistently use
`Period-mAP`, `Period-AP50`, `Period-AP75`, `AvgMAE`, and `AvgOBO`. The result
keys remain the same `*-ap50`/`*-ap75` controlled placeholders, every table cell
remains `PENDING`, and the surrounding requirements still bind the MultiRep
release, official split, evaluator revision/hash, aggregation and edge-case
semantics, named seeds, sample standard deviation, paired video-cluster
bootstrap confidence intervals, structured cell provenance, and the ban on
mixing releases/frontends/reruns. The cited post-fix audit independently
confirms `Period-AP50` and `Period-AP75` as the official MultiCounter table
headings and returns `PASS WITH NOTES`, explicitly provisional and same-family.

No acceptance threshold or submission ceiling is relaxed. The contract still
contains exactly AC01 through AC16 once each. It retains the exact title gate
(AC01), result/data binding gates (AC05--AC08), ICASSP 4+1 rule (AC10),
author/privacy gate (AC11), fresh ARIS assurance chain including substantive
cross-family review and submission verifier exit 0 (AC15), and derivation and
delivery integrity (AC16). Frontmatter still says `status: accepted`,
`amendment_review_status: pending`, and
`submission_ceiling: provisional/data-pending`. The five-page PDF visibly keeps
technical content on pages 1--4, references only on page 5, an author-roster
pending marker, the `DRAFT - RESULTS PENDING` banner, pending result cells, and
the data-pending diagnostic shell.

## Blocking issue

1. The current amendment note is factually misindexed: it says **"AC11 uses
   those names"**, but AC11 is **Author and privacy gate** and contains no
   MultiRep metric names. The metric list and the corrected labels are governed
   by **AC06 (Main comparison integrity)**. Therefore the frontmatter scope is
   narrow and the substantive change is safe, but the current explanatory note
   is not fully accurate/honest as required by this review. Replace `AC11` with
   `AC06` in that note and rerun the amendment review; no evidence threshold
   needs to change.

Because honesty/accuracy of the current contract explanation is an explicit
acceptance condition, this single documentation defect blocks amendment
acceptance despite the underlying nomenclature correction being safe.

## Input bindings

| Input | SHA-256 |
|---|---|
| `PAPER_ACCEPTANCE_CONTRACT.md` | `63c4601afa27f899c6b3708fab156834f28869309d1ffdd25f66e2264f4329ba` |
| `paper/sections/4_experiments.tex` | `6052c816e94f7d7f50b560d397495a37863b05d240ed5e418e4b6ec4d3aa131b` |
| `paper/main.pdf` | `873c83da0dbf4b24a6e1199f3dcd980e17d0b032580e0591f88a7c545ec67a45` |
| `paper/.aris/traces/citation-audit-icassp/group-a-postfix/reviewer.md` | `27fcd4ffb349d410bfd4d54cd6cb2bb71333449d50a01a38d257a6c0f3b80435` |

