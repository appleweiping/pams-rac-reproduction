# ICASSP post-acceptance contract amendment: final adjudication

## Verdict

Accepted. The current contract preserves the accepted baseline's substantive
AC01-AC16 obligations and provisional/data-pending submission ceiling. Its only
semantic changes are amendment metadata, an explanatory post-acceptance note,
and the correct AC06 nomenclature change from `AP50`/`AP75` to
`Period-AP50`/`Period-AP75`.

## Accepted-baseline verification

The baseline front matter states:

- `venue: ICASSP 2027 regular`
- `assurance: submission`
- `status: accepted`
- `submission_ceiling: provisional/data-pending`

It contains exactly the consecutively labeled assertions AC01 through AC16 and
retains an explicit final conference-submission block pending collaborator
code/results, complete authorship, the ICASSP 2027 kit/policies, fresh audits,
and a healthy cross-family review. It is therefore an accepted ICASSP AC01-AC16
contract, not a submission-ready acceptance.

## Exact SHA-256 input hashes

| Input | Bytes | SHA-256 |
|---|---:|---|
| `PAPER_ACCEPTANCE_CONTRACT_20260814_210927.md` | 9848 | `0bb799c33e1c1a4d3c9a9da76c019e0c3d463b2c98c6dc81b174fd6abb279ba8` |
| `PAPER_ACCEPTANCE_CONTRACT.md` | 10371 | `09f35eb7dd2c07fd9b8211763c1fbc797098972785e5eaf5e5ed063ef4a8c44d` |
| `paper/sections/4_experiments.tex` | 10852 | `6052c816e94f7d7f50b560d397495a37863b05d240ed5e418e4b6ec4d3aa131b` |
| `paper/main.pdf` | 348367 | `873c83da0dbf4b24a6e1199f3dcd980e17d0b032580e0591f88a7c545ec67a45` |
| `paper/.aris/traces/citation-audit-icassp/group-a-postfix/reviewer.md` | 12501 | `27fcd4ffb349d410bfd4d54cd6cb2bb71333449d50a01a38d257a6c0f3b80435` |

The experiments-source and PDF hashes exactly match the hashes recorded in the
allowed post-fix citation audit.

## Complete diff adjudication

Substantive additions/changes are limited to:

1. Three front-matter fields: `amended_at`, `amendment_scope`, and
   `amendment_review_status`.
2. One post-acceptance paragraph documenting the citation-audit basis, the two
   corrected metric names, the unchanged evidence/gate/acceptance threshold,
   the required fresh same-family amendment review, and the unchanged
   provisional ceiling.
3. One AC06 sentence changing only `AP50, AP75` to
   `Period-AP50, Period-AP75`. The change is located under AC06 Main comparison
   integrity, where metric nomenclature belongs.

There are also four byte-level, non-substantive Markdown whitespace changes:
the two trailing spaces after the `**Check:**` lines in AC01, AC02, AC03, and
AC04 were removed. The check text, evidence text, assertion text, and ordering
are unchanged. This affects only a rendered hard line break between adjacent
Check and Evidence lines; it does not alter or weaken any obligation.

As a cross-check, after removing the three added metadata lines and amendment
paragraph, restoring the two AC06 shorthand names, and trimming end-of-line
whitespace in both versions, the baseline and current contracts each contain
208 lines and have zero differing lines. No other semantic change exists.

## Gate-preservation review

- Evidence/data gates: AC03-AC10 and AC13-AC14 retain the same binding,
  provenance, eligibility, pending-cell, synchronization, experiment-audit,
  claim-audit, and submission-failure requirements. The AC06 rename changes no
  metric definition, artifact family, seed, uncertainty, or protocol rule.
- Author/privacy gate: AC11 is unchanged, including pending-author display,
  private author metadata, exact submission-system agreement, and author assent.
- Venue/policy gate: AC12 is unchanged, including the verified 2027 kit, EDICS,
  funding/COI/ethics, LLM-policy review, originality, simultaneous-submission,
  and paper-limit requirements.
- ARIS gate: AC15 is unchanged, including two fresh zero-context reviews, four
  mandatory audits, current source/PDF binding, pinned submission verifier, and
  substantive healthy cross-family-review requirements.
- Provisional/submission gates: `assurance`, `status`, `submission_ceiling`,
  `evidence_freeze`, the Current grading ceiling, and all listed submission
  blockers are unchanged. The amendment note explicitly says it cannot raise
  the provisional ceiling.
- Derivation/delivery gate: AC16 is unchanged.

## Source, PDF, and citation-audit verification

The experiments source uses `Period-mAP`, `Period-AP50`, and `Period-AP75` in
both the protocol prose and Table 1 header: two occurrences of each exact name,
with zero unprefixed `AP50` or `AP75` tokens.

Text extracted from the compiled PDF likewise contains the same three names in
the protocol prose and Table 1 header: two occurrences of each exact name, with
zero unprefixed `AP50` or `AP75` tokens. The occurrences are on extracted PDF
pages 3 and 4.

The allowed citation audit independently marks the MultiRep nomenclature check
PASS and identifies the official Table 1 labels as `Period-mAP`,
`Period-AP50`, `Period-AP75`, `AvgMAE`, and `AvgOBO`. Its overall
`PASS WITH NOTES` remains explicitly provisional/same-family, consistent with
the unchanged contract ceiling.

AMENDMENT_ACCEPTED:YES
