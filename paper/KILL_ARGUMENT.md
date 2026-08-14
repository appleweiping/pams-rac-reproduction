# Kill-Argument Audit — ICASSP 2027 Pre-results Draft

**Verdict: FAIL — strongest kill argument survives (same-family / provisional).**

The attack and defense were performed by separate fresh zero-context `gpt-5.6-sol` agents at `ultra` reasoning against the same frozen five-page PDF and exact source/evidence hashes. Both reviewers inspected all five PDF pages. The attack recommends strong rejection; the defense concludes that the artifact is defensible and useful as an honest pre-results package, but agrees that it is not scientifically acceptable for submission now.

## Fatal findings

| ID | Defense ruling | Reason |
|---|---|---|
| P1.1 | `still_unresolved` | The proposed multi-person router/expert/reconstruction method has no frozen implementation commit, source anchors, or tests. |
| P1.2 | `still_unresolved` | No eligible MultiRep result, prediction, protocol binding, seed set, or uncertainty artifact exists. |
| P1.3 | `still_unresolved` | The four-equation narrative deliberately leaves decision-bearing estimator, router, expert, loss, fallback, and decoder interfaces synchronized only by future collaborator evidence. |

The defense ruled 2 of 17 findings answered, 10 partially answered, and 5 still unresolved. P3.2 also remains unresolved because the planned mechanism ablations cannot identify a mechanism until the underlying executable system and controls exist.

## What passes

- Evidence integrity: the draft visibly withholds results and never fabricates a method or number.
- Visual integrity: all five pages are legible; technical content ends on page 4 and page 5 contains references only.
- Scope discipline: the paper avoids prohibited novelty/SOTA claims and marks source-sensitive interfaces `SYNC-REQUIRED`.

The full attack, independent defense, prompts, metadata, and hashes are archived under `.aris/traces/kill-argument-icassp/final/`. This audit is intentionally blocking and applies only to the current frozen pre-results artifact; it must be rerun after synchronized code and eligible results arrive.
