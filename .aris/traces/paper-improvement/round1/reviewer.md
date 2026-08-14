# Fresh Zero-Context Review — Round 1

**Verdict:** major revision / reject in current form  
**Independence:** same-family/provisional  
**Reviewed artifact:** `paper/main_round0_original.pdf`

## Critical findings

1. The schematic objective does not yet train expert responses or the decoder;
   expert specialization, collapse prevention, score calibration, targets, and
   gradient paths are synchronization-required.
2. The count-vector/event-set interface did not yet define evaluator-ready
   intervals, confidence, spatial support, or predicted-track matching.
3. The reviewed PDF preceded source additions for required baseline rows and
   notation; subsequent rounds must verify a source/PDF freeze match.

## Major findings

- Distinguish the method from independent per-track PAMS, sliding-window PAMS,
  deterministic tempo bins, and simple response fusion.
- Overlap-add removes arithmetic window-count summation but does not guarantee
  one peak under phase disagreement; soften the causal claim and test it.
- Define local mean, lag/frequency units, valid support, low-confidence routing,
  expert specialization/collapse diagnostics, and transformation-aware routing.
- Extend the no-count/no-period-label boundary through tuning, selection, and
  calibration; clarify that continuity cannot join distinct tracker IDs unless
  a separate association module exists.
- Add the protocol disclosures promised by prose; pre-register a primary
  metric/comparator and separate seed from video-cluster uncertainty.
- Replace arbitrary per-person weights with deployable/capacity-relevant
  controls and add stronger stitching alternatives.
- Separate pre-tracking from post-track robustness and add active-identity
  scaling plus counting-only/full-pipeline timing.
- Simplify the dense framework figure and repair pagination/typography.

## Minor findings

Define `d`, the valid mean, and `epsilon`; avoid reusing `S`; clarify the
spectral score, taper coverage, joint visibility, person-wise matching, broad
MRAC wording, page breaks, and final removal of workflow language.

No repository files were modified by the reviewer.
