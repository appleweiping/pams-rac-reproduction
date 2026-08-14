# ICASSP 2027 paper improvement log

Status: `provisional/data-pending`

This log covers the fresh same-family ARIS improvement loop. It cannot establish
cross-family acceptance and does not override missing implementation, experiment,
author, or venue evidence.

## Round 0

- Artifact: `paper/main_round0_original.pdf`
- SHA-256: `26eeb9e5b913fa8160276447e9c31b38c9fe147ef7f64666aed41144427ecd26`
- Pages: 5; page 5 references only
- Visual verdict: PASS
- Scientific status: honest pre-results specification; not submission-ready

## Round 1

- Reviewer: fresh `gpt-5.6-sol`, xhigh, zero inherited context
- Trace: `paper/.aris/traces/paper-improvement/round1/`
- Reviewer verdict: `PROVISIONAL MAJOR REVISION`; `ROUND1_ACCEPTED: NO`
- Unresolved critical blocker: the collaborator-frozen trainable graph, decoder,
  supervision paths, and MultiRep evidence do not exist locally.

Implemented writing-side changes:

1. Reframed the empirical section as a planned, not completed, protocol.
2. Added signal-processing antecedents for windowed spectra, irregular samples,
   overlap-add, and expert routing; explicitly disclaimed component-level novelty.
3. Added a result-free, falsifiable local-versus-global tempo hypothesis.
4. Exposed predicted-track lifecycle and ground-truth association as source-bound
   protocol fields instead of assuming stable identities.
5. Added mask-spectrum leakage controls, minimum-support requirements, and an
   irregular-sampling alternative to the period-estimator contract.
6. Expanded matched ablations to isolate ACF/FFT evidence, single/hard/soft
   routing, global/local tempo, and response reconstruction; defined delta signs.
7. Weakened the overlap-add claim to the guarantee supported by operation order
   and added timestamp/positive-coverage conditions.
8. Aligned figure notation with the equations, enlarged labels, and simplified
   the provisional training strip without claiming implementation.
9. Compressed the manuscript to approximately 2,377 words while retaining four
   core equation environments and the 4+1-page layout.

Round-1 compiled artifact:

- `paper/main_round1.pdf`
- SHA-256: `0e521ad303b21fbb98d7c3ea52cb2aec2c5376ea07400986bb1721b0ca3c9a17`
- Build: zero undefined references/citations, zero duplicate labels, zero
  overfull boxes; all fonts embedded
- Visual review: all five pages PASS; page 5 contains references only

## Round 2

- Reviewer: fresh `gpt-5.6-sol`, xhigh, zero inherited context
- Trace: `paper/.aris/traces/paper-improvement/round2/`
- Reviewer verdict: `PROVISIONAL MAJOR REVISION / WEAK REJECT`;
  `ROUND2_ACCEPTED: NO`
- Unresolved submission blockers: executable collaborator-frozen router, expert,
  decoder, and training definitions; persistent-identity semantics beyond supplied
  tracks; frozen MultiRep protocol and evidence.

Implemented writing-side changes:

1. Scoped the primary output claim to supplied persistent identity tracks and
   separated predicted-track robustness from that claim.
2. Replaced ambiguous causal wording with processing order.
3. Added a matched supplied-oracle-track block to isolate the counter from the
   tracking frontend.
4. Bound metric activation to release-native definitions plus frozen evaluation
   unit, assignment, unmatched, aggregation, empty-support, and zero-count rules.
5. Added a direct decode-each-window-then-aggregate control; defined the boxcar
   control, set the reference deltas to zero by definition, and required a
   boundary-offset test.
6. Required the tempo-warp artifact to transform frames, masks, and event times;
   freeze interpolation, support, duration, and severity bins; and include a
   held-out natural-nonstationarity analysis.
7. Moved the planned diagnostic source after its first textual introduction and
   simplified the active framework figure for print-scale readability.

The method itself was not invented to satisfy review. Every implementation-
sensitive interface remains source-bound and submission-blocking.

Round-2 compiled artifact:

- `paper/main_round2.pdf`
- SHA-256: `873c83da0dbf4b24a6e1199f3dcd980e17d0b032580e0591f88a7c545ec67a45`
- Build: five pages; zero undefined references/citations, duplicate labels,
  overfull boxes, or unembedded fonts
- Page rule: technical content ends on page 4; page 5 contains references only
- Visual review: all five rendered pages PASS

## Final pre-results stabilization

- `paper/main.pdf` is byte-identical to `paper/main_round2.pdf` at SHA-256
  `873c83da0dbf4b24a6e1199f3dcd980e17d0b032580e0591f88a7c545ec67a45`.
- Canonical MultiRep metric names were corrected to `Period-mAP`,
  `Period-AP50`, `Period-AP75`, `AvgMAE`, and `AvgOBO` and re-audited against
  the active citation contexts.
- The proof audit is `NOT_APPLICABLE`; the citation audit passes under the
  same-family provisional ceiling. The claim and kill-argument audits remain
  blocking because no synchronized multi-person implementation or eligible
  result artifact exists.
- Submission mode remains intentionally fail-closed for pending method/results,
  an incomplete author roster, the unavailable official 2027 kit, the absent
  cross-family response, and unresolved final scientific evidence.
