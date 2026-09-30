# ARIS Novelty Check — Set-Valued Counting Under Pose Missingness

Audit: `20260815_set_valued_sol`  
Date: 2026-08-15  
Reviewer/model: `gpt-5.6-sol`  
Reviewer family: OpenAI, same-family  
Acceptance: provisional  
Fresh context: `false`  
Nonfresh reason: `agent_thread_lifetime_limit`

## Verdict

**CAUTION — 5.6/10. Novelty is not cleared. No implementation, training, paper claim, or next-stage launch is authorized.**

The checked primary corpus did not disclose the exact conjunction “missing coordinates in a supplied per-person pose track are propagated through a fixed repetition-counting signal map to a set of plausible counts without learner-side count or period labels.” That is a plausible, narrow task-specific residual.

The components are crowded. Predictive count intervals and uncertainty-aware counting already exist; missing-covariate prediction sets, nested masking, interval-width behavior and mask-conditional coverage are established; pose uncertainty and missing-pose repair are established; and skeleton RAC under occlusion already exists. The candidate is therefore not a new uncertainty principle. At best it can contribute a carefully bounded RAC propagation-and-non-vacuity result.

The current cheap gate has a serious semantic defect: containment of the frozen full-observation estimate is self-consistency with the same counter, not coverage of the true repetition count. It can pass while preserving arbitrary counter bias.

## Core claims

### C1 — Missing pose to a repetition-count set

Claim: a pose-missingness set can be propagated to a per-identity set of plausible repetition counts without learner-side per-person count or period labels.

Finding: no checked primary source directly disclosed this exact endpoint. [SSTRAC](https://doi.org/10.1109/ACCESS.2025.3624029) and [Viewpoint-Invariant Exercise Repetition Counting](https://pmc.ncbi.nlm.nih.gov/articles/PMC10692053/) repair or interpolate pose and emit point counts. [PAMS](https://openaccess.thecvf.com/content/CVPR2026F/html/Gao_Count_What_Repeats_Period-Adaptive_Multi-Scale_Consistency_for_Self-Supervised_Repetitive_Action_CVPRF_2026_paper.html) emits a self-supervised point count. [MultiCounter](https://journals.sagepub.com/doi/pdf/10.3233/FAIA240494) and MultiCounter+ emit supervised per-person point counts.

Status: plausible narrow residual, not novelty clearance. Search absence is not proof of novelty.

### C2 — Nested masks imply nested sets or wider intervals

Claim: removing more pose information should produce a superset count interval and nondecreasing width.

Finding: the general principle is occupied. [Zaffran et al. 2024](https://arxiv.org/abs/2405.15641) explicitly define included missingness masks, analyze conditional variance, inter-quantile distance and interval-length isotonicity under assumptions, and construct CP-MDA-Nested variants. [Conformal Prediction with Missing Values](https://proceedings.mlr.press/v202/zaffran23a.html) and [AISTATS 2026 mask-conditional conformal prediction](https://openreview.net/pdf?id=FiQAQSTXZn) already cover missing-covariate predictive sets and mask-conditional validity.

Exact nesting for this particular analytic RAC map can be a correctness property. It is not a broad novelty claim, and pointwise widening is not automatic without assumptions.

### C3 — Coverage semantics

Claim: a masked-input interval that contains the frozen full-input estimate has coverage.

Finding: rejected. Self-containment is an engineering sanity check. Coverage is defined against the true outcome. [Eaton-Rosen et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC7611043/) train and calibrate predictive count intervals with real count targets and evaluate ground-truth inclusion versus width. Conformal missing-value methods likewise use response-labeled calibration and explicit assumptions.

No learner-side count labels means the current construction cannot claim conformal, calibrated, finite-sample, mask-conditional, or true-count coverage. An evaluator may score empirical true-count containment only after predictions are frozen; that statistic is not a guarantee.

### C4 — Partial identification and non-vacuity

Claim: the output is a non-vacuous partial-identification interval.

Finding: unproven. Empirical bone-length and velocity envelopes do not automatically contain the latent pose under occlusion, detector failure, off-frame motion or MNAR missingness. Approximate interval response mass is not automatically the sharp image of the feasible pose set through nonlinear peak selection, thresholding and rounding.

Until soundness is established, call the output a sensitivity envelope or feasible-set count range. Use “partial identification” only after defining the estimand and admissible-data model, proving validity, and proving sharpness if sharpness is claimed.

## Closest prior and precise delta

1. Zaffran 2023/2024 and Fan et al. 2026 are closest at the mechanism level: missing-covariate predictive sets, nested masks, width, calibration and mask-conditional coverage. Their methods use labeled responses and generic predictors.
2. SSTRAC and Viewpoint-Invariant Exercise Repetition Counting are closest at the application level: incomplete or occluded skeleton RAC followed by a point count.
3. Eaton-Rosen, [UNIC](https://proceedings.mlr.press/v238/xu24b.html), and [future-event count intervals](https://arxiv.org/abs/2601.04192) block generic count-interval and uncertainty-aware-counting novelty.
4. [CUPS](https://proceedings.mlr.press/v267/zhang25g.html), [Conformal Fusion Under Missing Modalities](https://arxiv.org/abs/2608.07183), and [partial identification under missing data](https://arxiv.org/abs/2602.16061) block generic pose-UQ, missing-input-widens-the-set, calibration, and partial-identification terminology.

The only defensible delta is:

> Given one supplied, possibly incomplete pose track, propagate one explicitly nested feasible coordinate set through one frozen analytic repetition-response map and return the corresponding discrete count range; then test whether that range remains informative as pose is removed, without fitting the learner to per-person count or period labels.

The possible contribution is that task-specific propagation-and-non-vacuity result. It is not counting intervals, conformal prediction, nested masking, pose uncertainty, pose repair, robust RAC, self-supervised periodicity, or multi-person counting.

## Search coverage

Four claims were searched with at least four formulations each. Exact queries and source-resolution context are preserved in the paired JSON and trace. The query families were:

- C1: “2024 2025 2026 repetitive action counting missing pose uncertainty interval prediction skeleton”; `"repetition counting" "prediction interval" pose OR skeleton`; `"set-valued" "repetition counting" missing data`; `site:arxiv.org repetition counting uncertainty missing pose skeleton interval`.
- C2: “2024 2025 2026 nested mask prediction intervals missing covariates monotonic uncertainty sets”; `"nested masks" uncertainty interval missing data machine learning`; “conformal prediction missing covariates nested missingness sets monotonic width”; `site:arxiv.org 2026 "missing covariates" conformal prediction sets`.
- C3: “2024 2025 2026 conformal prediction intervals count data event count time series”; `"prediction interval" "event count" missing observations time series`; `"partial identification" count missing data time series observations`; `site:arxiv.org set-valued count prediction missing time series 2026`.
- C4: “2024 2025 2026 human pose estimation uncertainty prediction sets missing joints conformal”; `site:arxiv.org human pose uncertainty set missing joints occlusion 2026`; `site:arxiv.org repetitive action counting uncertainty 2026 pose`; “2026 arXiv repetition counting occlusion missing skeleton uncertainty interval”.

A separate 2026-02-15 through 2026-08-15 arXiv window searched RAC intervals, missing-covariate mask-conditional conformal prediction, partial identification, and pose uncertainty. It surfaced AISTATS 2026 mask-conditional CP, an August 2026 missing-modality conformal preprint, PAMS, future-event count intervals and 2026 partial-identification work, but no direct pose-missingness-to-RAC-count-set paper.

No local external-paper PDFs were present. Local evidence was used only for the current project boundaries. Technical conclusions rely on publisher/proceedings PDFs, author arXiv records, and official code or author-hosted copies.

## Coverage properties must stay separate

- Nested-mask set inclusion: a deterministic structural property of the implemented set map. It says nothing about truth.
- Full-observation self-containment: consistency with the frozen full-input model estimate. It is not calibration.
- Empirical true-count coverage: a post-freeze evaluator statistic requiring real count labels. It is not a finite-sample guarantee.
- Calibrated coverage: requires labeled calibration plus a declared algorithm and assumptions such as exchangeability or a justified correction.
- Set-membership soundness: conditional on the latent missing pose lying in the feasible set and the output containing every attainable count.
- Partial identification: requires an identified estimand, explicit admissible data-generating restrictions, validity, and sharpness if claimed.

## Fatal risks

- The kinematic set may exclude the true latent pose exactly where missingness is most severe.
- Nonlinear count operators can make interval arithmetic unsound or vacuous.
- The no-count-label learner cannot calibrate count coverage.
- The 95% self-containment gate can pass while true-count coverage fails.
- The interval may be indistinguishable from a trivial count range at 30% missingness.
- Reviewers may reasonably see the work as missing-covariate UQ applied to SSTRAC-style skeleton robustness.
- Supplied per-person tracks do not create a new MRAC contribution and depend on externally supervised detector, pose estimator and tracker quality.
- Random masking alone cannot support real-occlusion or tracker-failure claims.

## Allowed and forbidden claims

Allowed, conditional on later evidence:

- “We study set-valued repetition counting under controlled missingness on supplied pose tracks.”
- “The learner-side set constructor and analytic counter use no per-person count or period labels.”
- “Under the declared feasible-set and exact-propagation assumptions, nested input sets induce nested output count sets.”
- “After predictions were frozen, evaluator-side labels yielded the reported empirical true-count containment on the named validation split.”
- “The sets were narrower than a preregistered trivial baseline at matched empirical containment.”
- Explicit disclosure of external detector, pose-estimator and tracker supervision.

Forbidden:

- Any “first” claim, including first count interval, first uncertainty-aware counting or first missing-pose RAC.
- Annotation-free, fully label-free, end-to-end, constant-time or SOTA language.
- Calibrated, conformal, valid, guaranteed or mask-conditionally covered count intervals without labeled calibration and proved assumptions.
- Partial identification or sharp bounds without formal validity and sharpness.
- Nested-mask monotonicity as a new general principle.
- Real-occlusion robustness from random-mask self-containment.
- Phase, local-window, dynamic-threshold, identity-consistency, self-supervised-periodicity or MRAC novelty.

## Minimum next gate

`G0_SET_SEMANTICS_AND_NONVACUITY` is the only justified next gate. It is a frozen local train/validation gate only: no server, test, sealed, heldout or result access; no training; and no threshold tuning after inspection.

Precommit one analytic bank, one feasible-set rule, one count-set definition, one trivial count-set baseline, and deterministic nested mask chains. State whether the output is an exact attainable set, a certified outer bound, or only a sensitivity envelope. Separate learner-side labels, evaluator-only post-freeze scoring, and calibration.

Kill conditions:

1. On short enumeratable fragments, kill on any computed-set under-coverage relative to exhaustive feasible completions.
2. Kill on the first nested-mask set-inclusion violation or width decrease.
3. Keep the original at-least-95% frozen-full-estimate containment target, but call it self-containment only.
4. At 30% missing joints, kill if median width exceeds 1.5 cycles or the method is not tighter than the trivial range at matched self-containment.
5. After predictions are locked, an evaluator may score true-count containment on validation. Missing the predeclared empirical target kills the coverage story; passing does not create a guarantee.
6. Audit random-joint, contiguous-frame, limb-block and confidence-correlated masks. Restrict claims to mask families that pass.

Only zero soundness/nesting violations plus non-vacuous width and honest post-freeze truth containment would justify a later bounded implementation proposal. Otherwise abandon the candidate or relabel it as a diagnostic sensitivity baseline.

## Preserved boundaries and failure memory

MultiCounter/+ own MRAC, supervised per-person point counts, identity consistency and long-short-period awareness. PAMS owns self-supervised skeleton periodic learning, multiscale TCC, tempo stress and peak consensus. DeepPhase owns unsupervised local phase. TWCRAC's abstract-level boundary owns local nonstationary statistics, local TCC and dynamic thresholds; its unavailable full text is not used as absence evidence. SSTRAC owns skeleton repair and occlusion-robust point RAC. External pose/tracker supervision must remain explicit.

The signed warp-integral WARP-PHASE candidate remains terminally killed by immutable Gate-2 counts `246/55/144/50/66`. No threshold tuning, fixture replacement, resurrection, reinterpretation or server training is permitted. Nothing from that selector or phase route was carried into this audit.

## Review limitation

This is a same-family, nonfresh continuation review and remains provisional. Failure to locate a direct disclosure is not proof of novelty. No implementation, training, experiment, server access, test/sealed/heldout access, result inspection, or MANIFEST modification occurred.
