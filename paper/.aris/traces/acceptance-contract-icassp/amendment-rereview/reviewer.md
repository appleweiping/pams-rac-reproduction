AMENDMENT_ACCEPTED:NO

# Fresh amendment adjudication

Classification: **same-family / provisional**.

## Blockers

1. The allowed evidence does not establish a pre-amendment accepted ICASSP
   contract baseline. The current Git diff is not an isolated two-token
   amendment: it contains 213 inserted and 105 deleted lines and replaces a
   materially different CVPR contract. Consequently, the reviewer cannot
   independently verify that the post-acceptance change was *strictly* limited
   to `AP50`/`AP75` -> `Period-AP50`/`Period-AP75`.
2. For the same reason, the absence of weakening relative to the accepted
   pre-amendment contract cannot be proven. The current contract visibly retains
   evidence, author, ICASSP venue, data, ARIS, provisional, and submission gates,
   but its statement that those gates and thresholds are unchanged is not a
   substitute for an auditable pre/post comparison.

## Checks that pass on the available current artifacts

- The amendment correctly targets **AC06**, whose heading is "Main comparison
  integrity" and whose metric list uses `Period-mAP`, `Period-AP50`,
  `Period-AP75`, `AvgMAE`, and `AvgOBO`.
- The supplied citation post-fix audit expressly validates the official
  MultiRep nomenclature `Period-mAP` / `Period-AP50` / `Period-AP75`.
- The contract contains exactly one each of AC01, AC02, AC03, AC04, AC05, AC06,
  AC07, AC08, AC09, AC10, AC11, AC12, AC13, AC14, AC15, and AC16.
- `paper/sections/4_experiments.tex` uses `Period-AP50` and `Period-AP75` in the
  protocol prose and Table 1 header.
- Text extracted from `paper/main.pdf` shows `Period-AP50` and `Period-AP75` in
  both the protocol paragraph and Table 1 header. No bare legacy name is visible
  there.
- The current contract retains `assurance: submission`,
  `submission_ceiling: provisional/data-pending`, a pending amendment review,
  and an explicit statement that the amendment cannot raise the provisional
  submission ceiling.

## Required closure

Provide a byte-identifiable accepted pre-amendment ICASSP contract, or a focused
pre/post diff against that exact baseline, showing only the two nomenclature
replacements plus the amendment-record metadata. The comparison must confirm
that every gate and acceptance threshold is otherwise unchanged. This review
does not authorize any manuscript, contract, or PDF edit.
