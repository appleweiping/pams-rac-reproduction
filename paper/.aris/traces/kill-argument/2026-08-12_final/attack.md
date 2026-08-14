# Fresh Final Kill-Argument Attack

- Attacker task: `/root/final_kill_attack_postfix`
- Isolation: fresh zero-context; prior kill artifacts excluded
- Model/reasoning: `gpt-5.6-sol` / `ultra`
- Independence: same-family; acceptance ceiling `provisional`
- Source freeze: `1376b6a388d049b12548a589e9753237bd9e62802091d822cd46cb450aa26184`
- PDF SHA-256: `de951023634d5627f3c38cd7512da9fd7df20086781d24caf0a548b93b3d9522`

## Unified kill argument

The manuscript is not a complete method paper merely awaiting a few
experiments. It is an honest and rigorous preregistered specification that has
not yet become an evaluable scientific object. The core algorithm is neither
frozen nor implemented, the supervision condition that defines the paper's
narrow differentiation has not been audited, and every admissible empirical
result is absent. The manuscript also acknowledges that its individual
ingredients are prior or generic operations. A reviewer therefore cannot
determine whether the method exists, is trainable, satisfies its supervision
claim, differs from per-track PAMS plus generic post-processing, or works.
Transparent qualification removes deception, but it cannot substitute for a
scientific contribution.

## P1 — No frozen, trainable, reproducible core algorithm

- Evidence: `sections/3_method.tex:261--272` says Equation 14 is not yet a
  complete trainable algorithm and enumerates missing response/decoder
  training, specialization, collapse prevention, calibration, targets,
  weights, gradient paths, and decoding settings. In
  `evidence/method_contract.yaml`, the collaborator artifact and code anchors
  are absent, the current repository is identified as a single-person PAMS
  reproduction, every core module is blocked, and the new objectives remain
  conceptual.
- Severity: CRITICAL; independently rejection-sufficient.
- Falsifiable implication: the frozen text permits multiple behaviorally and
  supervision-wise different implementations, so an independent implementer
  cannot recover a unique training/inference program.
- Required fix: freeze the implementation; specify all objectives, targets,
  decoder behavior, architecture, thresholds, gradient paths, and calibration;
  bind every module to a commit/hash and executable test.

## P2 — The defining supervision distinction is unproved

- Evidence: `sections/1_introduction.tex:29--40,78--82` makes supervision,
  modality, and local temporal scale the central gap but requires a frozen
  training-graph audit. `evidence/supervision_firewall.json` blocks loader,
  pseudo-label, loss, router target, tuning, checkpoint selection, early
  stopping, calibration, and inference scopes, with no source anchors.
- Severity: CRITICAL.
- Falsifiable implication: real code may consult person count/period labels
  through targets, tuning, calibration, or checkpoint selection; any such use
  would invalidate the proposed distinction from supervised MRAC.
- Required fix: bind and audit all nine scopes to exact source hashes and
  narrow or rewrite the supervision claim if leakage is found.

## P3 — No admissible evidence establishes effectiveness

- Evidence: `sections/4_experiments.tex:4--9` says the section contains no
  new-method measurement or empirical conclusion; `sections/5_conclusion.tex:
  12--20` confirms the missing multi-person path, router, continuity loss,
  boundary decoder, and MultiRep artifact. The evidence contract marks current
  evidence table-ineligible and all accuracy/robustness/generalization/
  efficiency claims blocked. PDF result tables are entirely pending.
- Severity: CRITICAL; independently rejection-sufficient.
- Falsifiable implication: the current record cannot show that the system
  runs, beats Track-PAMS, avoids router collapse, improves boundary errors, or
  outputs meaningful counts.
- Required fix: complete frozen MultiRep oracle/common-predicted protocols,
  matching baselines, at least three seeds, paired uncertainty, core ablations,
  and failure analysis; generate all tables from the frozen manifest.

## P4 — Novelty may collapse to a composition of generic parts

- Evidence: the manuscript acknowledges established MRAC, person-wise,
  asynchronous and variable-speed counting; it treats PAMS as the base and
  local routing/normalized overlap-add as generic operations; MultiCounter and
  MultiCounter+ already address multi-person identity and tempo. The required
  Track-PAMS, deterministic-router, fusion, and capacity controls are only
  future experiments.
- Severity: CRITICAL in combination with P1--P3.
- Falsifiable implication: an eventual implementation may be equivalent to
  independent PAMS per track plus generic sliding-window routing/averaging,
  with no independently demonstrated interaction or gain.
- Required fix: freeze the implementation and run capacity-matched Track-PAMS,
  nonlearned/global/local tempo, hard/uniform/soft routing, overlap-add
  alternatives, and factorial interaction controls.

## P5 — An evaluation contract alone cannot rescue a method paper

- Evidence: the Introduction labels the evaluation contract a contribution,
  the appendix calls its analyses requirements for a future evidence bundle,
  and the acceptance contract blocks result binding, comparison, statistics,
  and final readiness because no result manifest exists.
- Severity: HIGH and unable to offset the fatal defects above.
- Falsifiable implication: a schema specifies the shape of future evidence but
  proves neither an algorithm, measurement validity, data coverage, nor
  technical benefit; it is not presented and validated as an independent
  benchmark/evaluator contribution.
- Required fix: complete the method and evidence package, or recast the work as
  a position/protocol artifact and deliver a validated standalone evaluator.

## P6 — Honest qualifications confirm incompleteness; they do not create a contribution

- Evidence: Abstract, Method, Experiments, Introduction, and Conclusion all
  state that this is an evidence-gated pre-results specification, that
  synchronization-required components are not asserted as implemented, that
  no empirical conclusion exists, and that status is provisional/data-pending.
- Severity: decisive interpretation.
- Falsifiable implication: after removing unimplemented, unaudited, and
  unvalidated content, the remaining contribution is a problem formulation
  and future experiment plan rather than a complete algorithm, theory result,
  dataset, or empirical finding.
- Required fix: disclaimers cannot close this issue; P1--P4 require actual
  scientific artifacts and evidence.

## Net assessment

`FATAL / REJECT` with high confidence.

> The paper is commendably transparent, but transparency converts potential
> overclaiming into an explicit admission that no fixed method or admissible
> evidence yet exists; a preregistered research plan is not an evaluable CVPR
> method contribution.
