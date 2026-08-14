# Fresh Paper Review — Round 2

Artifact: `paper/main_round1.pdf`

Route: fresh zero-context, same-family reviewer, ultra reasoning. The reviewer
was not shown the round-1 critique or improvement log.

Score: **3.5/10**

Verdict: **Strong reject if submitted now**; useful and unusually disciplined
as a synchronization agenda, but not yet a complete trainable or
evidence-bearing CVPR paper.

## Critical findings

1. Equation 14 is schematic: the exact training graph does not define what
   trains expert responses, the decoder, event confidence/bounds, or
   specialization. Synchronize frozen code or remove unsupported components.
2. The spectral router needs deterministic behavior for unsupported lags,
   static/pause windows, irregular masks, harmonics, and an explicit null or
   zero-response route.
3. Predicted tracker IDs are not ground-truth people. Fragmentation/merging and
   the official event matching adapter must be formalized; otherwise call
   predicted outputs track-wise.
4. The narrow no-count/no-period-label claim still needs a full training-data
   and supervision firewall.
5. No empirical evidence exists, and the prevalence of within-track tempo
   changes in the exact MultiRep release remains unestablished.

## Major findings

- “Identity-conditioned” overstates independent track-indexed processing unless
  identity actually conditions computation.
- Novelty relative to MultiCounter/MultiCounter+/PAMS is narrow; add a component
  matrix and replace the evaluation-contract contribution with an empirical
  finding after evidence exists.
- Related work should cover adaptive temporal scales, MoE/routing,
  overlap-add/phase alignment, and track-conditioned temporal analysis.
- Every common-interface baseline adapter remains evidence-gated.
- Use removal/factorial/capacity-matched ablations and routing-collapse
  diagnostics.
- Define one hierarchical inference unit and recompute non-additive mAP inside
  every bootstrap replicate.
- Real-world claims require a real multi-person set or explicit restriction to
  synthetic MultiRep plus controlled corruptions.
- Tables need active-ID levels, stratum sizes, exact severities, and stronger
  controlled/native panel separation.

## Minor findings

Fix `runs.The`; label Figure 1 as a naïve pooled failure mode; simplify Figure
2; standardize metric names; address page-9 whitespace; sanitize internal
process language and optional timezone metadata before submission; document
the numerical effect of epsilon stabilizers.

## Defensible elements

Prior art is credited accurately; the external perception boundary and narrow
supervision scope are careful; shared weights with per-track state are
coherent; overlap-add claims are properly limited; protocols and tracking
metrics are separated; and the reproducibility plan is strong. The rendered
PDF has no clipping or overlap.
