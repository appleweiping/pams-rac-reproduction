# Fresh Final Paper-Claim Audit

- Reviewer task: `/root/final_claim_audit_postfix`
- Binding audit task: `/root/final_claim_audit_postfix/source_binding_audit`
- Isolation: fresh zero-context; old claim audits and traces excluded
- Model/reasoning: `gpt-5.6-sol` / `ultra`
- Independence: same-family; acceptance ceiling `provisional`
- Source freeze: `1376b6a388d049b12548a589e9753237bd9e62802091d822cd46cb450aa26184`
- PDF SHA-256: `de951023634d5627f3c38cd7512da9fd7df20086781d24caf0a548b93b3d9522`

## Verdict

`FAIL` — `MISSING_ELIGIBLE_MULTIREP_EVIDENCE_AND_BINDINGS`

`detector_negative=true` only means that no populated unsupported
paper-owned empirical value, numeric contradiction, or source--PDF mismatch
was found. It does not mean the evidence is complete. The manuscript has no
submission-eligible new-method empirical result.

## Exact controlled-binding inventory

| Kind | Uses | Unique | Populated | Missing |
|---|---:|---:|---:|---:|
| Result `R` | 139 | 139 | 0 | 139 / 139 |
| Protocol | 85 | 74 | 0 | 85 / 74 |
| Prose claim `C` | 0 | 0 | 0 | 0 |
| Controlled Abstract | 1 | 1 | 0 | 1 / 1 |
| **Total including Abstract** | **225** | **214** | **0** | **225 / 214** |

There are no value declarations and `generated/evidence_values.tex` is
absent. All 139 result keys are unique. The eleven repeated protocol uses are
five uses of `disc.common.track`, three of `eff.identity_levels`, and two uses
each of five `sup.*` fields.

## Source/PDF consistency

The reviewer read the complete source and visually inspected every page of
the eleven-page PDF. The 224 keyed call sites and one Abstract gate all render
as pending/default-withheld content. The cited MultiRep counts and the planned
95% uncertainty level are source-backed dataset/protocol statements, not
paper-owned results. No numerical contradiction or source--PDF mismatch was
detected.

## Evidence exclusions

The three allowed external inputs hash-match their anchors but none is
eligible for a MultiRep claim:

1. `results/dev-negative/summary.json` is explicitly a UCFRep dev84
   development-only negative result for a single-person PAMS partial
   reproduction. The sealed 105-video test was not run and the artifact is
   ineligible as a leak-free sealed result.
2. `configs/protocols/ucfrep_526.yaml` defines a single-person UCFRep protocol,
   not a MultiRep release, person-wise annotation, evaluator, or tracking
   contract.
3. `configs/pams.yaml` is a single-person PAMS configuration and contains no
   MultiRep input, identity-indexed tracks, learned router, continuity loss,
   boundary decoder, or proposed-method result.

## Blocking conditions

- 225 controlled uses / 214 unique bindings are unresolved.
- No frozen MultiRep release, split, checksums, evaluator, preprocessing
  fingerprint, implementation, or result bundle exists locally.
- No qualifying oracle/predicted-track runs, three-seed results, event records,
  per-person outputs, matching tests, or bootstrap records exist.
- Every main, ablation, robustness, efficiency, and data table is pending.
- The Abstract has no audited quantitative result sentence.

The paper is an honest pre-results specification, but submission assurance
must remain `FAIL` until these evidence obligations are discharged.
