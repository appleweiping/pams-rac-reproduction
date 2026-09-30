# Experiment Plan: WARP-PHASE Pilot v1

> **PROSPECTIVE PARTIAL-CACHE SUPPLIED-TRACK PILOT**  
> **ZERO ELIGIBLE RESULTS — SAME-FAMILY PROVISIONAL — FINAL VERDICT `REVISE`**  
> **K8 `BLOCKED / NOT_PASSED_BLOCKED`; CLAIM FREEZE AND TRAINING CURRENTLY BLOCKED**

**Problem:** Determine whether an explicit signed local warp-integral objective improves per-person repetition counting under within-track tempo drift on checksum-frozen GT-bbox-assisted AlphaPose supplied tracks, without learner access to per-person counts, periods, densities, or boundaries.  
**Method thesis:** Treatment and augmentation-only share one 225,026-parameter phase learner, data, warps, optimizer, recurrence, NOLA, decoder, and evaluation; treatment alone receives the signed discrete warp-integral loss `L_WI`.  
**Date:** 2026-08-15  
**Normative execution binding:** `refine-logs/EXPERIMENT_EXECUTION_CONTRACT.md`

## Planning Gate

- **Final method thesis:** a clean identity track defines a raw signed phase measure, and a known target-to-source time map defines the exact oriented integral that a warped target interval should match.
- **Dominant contribution under study:** one discrete warp-integral equivariance objective for signed phase increments in supplied-track MRAC.
- **Intentionally rejected complexity:** no learned tempo router/experts, reconstruction head, activity head, confidence learner, phase corrector, semantic-period supervision, detector, tracker, TSSM fallback, or frontier-model component.
- **Reviewer concerns that remain validation-relevant:** K1 efficacy/uncertainty; K3 objective redundancy; K4 clock/mask/interpolation shortcuts; K5 pseudo-cycle/execution stability; K6 isolation; K7 population/clock support; K8 complete-primary-source novelty collision.
- **Frontier primitive:** absent by design. Exact transformation metadata is the appropriate supervision; an LLM/VLM/diffusion/RL component would not test the anchored mechanism.
- **Gate conclusion:** local specification is normatively closed by the execution contract, but no gate has passed. Proceed only to deterministic Gates 0–2. Training remains blocked until all three pass.

## Claim Map

| Claim | Why it matters | Minimum convincing evidence | Linked blocks |
|---|---|---|---|
| C1 — conditional anchor claim | Tests whether the signed local objective improves the actual supplied-track tempo-drift bottleneck beyond identical augmentation. | All three fixed seeds; `R>=0.05`; paired all-nine-source-component absolute-effect 95% CI lower endpoint `>0`; identical receipts; no K4/K5/K6/K7 failure. | V1, V2 |
| C2 — conditional isolation claim | Tests whether any reached C1 behavior is associated with identity-private response rather than another person's intervention or key routing. | Frozen-backbone one-person intervention with byte-identical untouched tensors/clocks; private/shared-scene/shuffled-key diagnostic and component intervals; never a rescue for C1. | V3 |

**Anti-claims to rule out:** the difference comes only from known clocks, masks, interpolation artifacts, unmatched parameters/budgets, a global/unsigned/deletion objective, a closest prior, shared-scene leakage, or harmonic correction.

Permitted future wording is only: “We study whether the signed local warp-integral objective improves supplied-track partial-cache tempo drift relative to identical augmentation.” No first/SOTA/end-to-end/annotation-free/predicted-track/official-MultiRep wording is licensed.

## Paper Storyline

- **Main paper must prove:** V1 primary paired endpoint; reached V2 mechanism/shortcut/closest-prior falsification; V3 identity-isolation diagnostic.
- **Appendix/receipts can support:** Gates 0–3, fixture/environment hashes, selector traces, K4 bytes, comparator provenance, secondary AvgOBO and supplied-track Period-mAP/AP50/AP75 on `[0,100]`.
- **Intentionally cut:** new model families, learned routing, detector/tracker results, sealed/test results, historical v46/v62/v63 efficacy, post-hoc harmonic correction, extra visible blocks, and any automatic TSSM pivot.

## Baseline Family Limit

Exactly three baseline families are allowed:

1. **Matched mechanism controls:** augmentation-only deletion, direct signed-frequency, global-warp-only, and unsigned reversal; clock/no-pose and mask-only are K4 diagnostic controls within this family.
2. **Closest phase/tempo priors:** PAMS, Track-PAMS, SimPer-style, CycleCL-style, DeepPhase-style, and generic time-equivariant adaptations, treated as one prior family with separate provenance.
3. **Optional geometric control:** masked geometric TSSM, control-only and never a contribution or fallback.

No additional family may be introduced inside this protocol.

## Prerequisite Gates (Engineering Receipts, Not Paper Blocks)

### Gate 0 — pack/vault/isolation

- **Mode:** deterministic CPU; zero training jobs/GPU-hours.
- **Inputs:** canonical v44 train/validation hashes only; no test/sealed/result artifact.
- **Pass conditions:** forbidden-path rejection before pickle; source hash before trusted unpickle; exact seven-field deterministic NPZ whitelist with `allow_pickle=False`; separate `features/`, `vault/`, `audit/`; 617-byte schema fixture; count-blind eligibility; one-to-one association; all nine original development components nonempty; training import graph cannot reach evaluator/vault.
- **Failure:** K6/K7 and global stop before label access.

### Gate 1 — numerical and environment fixtures

- **Mode:** deterministic CPU; zero training jobs/GPU-hours.
- **Pass conditions:** dedicated CPython 3.12.13 + NumPy 2.4.6 lock; main NumPy 1.26.4 consumption-only witness; PCG64 raw witness; all canonical pack hashes; fixed ten-case cross-environment recomputation at `rtol=atol=1e-12`; exact K4 bytes; duplicate/normalization/FFT/warp/overlap/recurrence/optimizer/NOLA/decode/evaluator/bootstrap fixtures.
- **Failure:** K4/K5/K6/K7 as applicable; no regeneration or tolerance selection.

### Gate 2 — selector stability

- **Mode:** deterministic CPU; zero training jobs/GPU-hours.
- **Pass conditions:** immutable 1,000 fixtures, at least 250 symmetric (exactly 250 by contract), unchanged harmonic thresholds, frozen real-track global RNG order/perturbation hashes, and a selector answer for every eligible training identity.
- **Failure:** global K5; no track removal or correction.

### Gate 3 — CPU/tiny-GPU sanity

- **Mode:** one zero-job CPU smoke plus exactly three tiny-GPU training-only jobs at maximum execution; total cap 6 GPU-hours.
- **Pass conditions:** finite gradients, loss decrease, identity isolation, invalid-state hold, one recurrent commit per source clock, one backward per fresh step, four-step accumulation, deterministic receipts, and prediction schema; no evaluator count/period access.
- **Failure:** K5/K6 and stop before Stage A.

## Three Visible Validation Blocks

### V1 — Primary paired law versus identical augmentation

- **Claim tested:** C1.
- **Why this block exists:** it is the only efficacy decision and isolates the one treatment deletion.
- **Dataset/split/task:** checksum-frozen, count-blind eligible partial v44 training/development supplied tracks; development unit is video; source-connected components are the nine frozen clusters; fixed piecewise-warp diagnostic.
- **Compared systems:** treatment `PAMS + L_WI` versus augmentation-only `PAMS`; same 225,026 parameters, seeds, batches, schedules, initialization, optimizer, recurrence, decoder, and evaluator.
- **Primary metric:** continuous normalized video-first AvgMAE.
- **Secondary metrics:** AvgOBO, supplied-track Period-mAP/AP50/AP75 `[0,100]`, clean-to-warp degradation, phase-correspondence error. Secondary metrics cannot substitute for K1.
- **Setup:** seeds `20270815`, `20270816`, `20270817`; 20,000 updates; eight complete identities/fresh step; four fresh steps/update; AdamW `3e-4`, weight decay `1e-4`, clip `1.0`; maximum 12 GPU-hours per full job.
- **Success criterion:** `R=(M_c-M_t)/M_c >=0.05` and the paired 10,000-draw all-nine-component absolute-`D` 95% CI lower endpoint is strictly positive; all receipts match.
- **Failure interpretation:** the local signed objective has no licensed supplied-track pilot benefit; stop at K1. No secondary metric or identity diagnostic rescues it.
- **Paper target:** primary table with three paired seeds, mean/sample SD, `D`, `R`, and CI.
- **Priority:** MUST-RUN if Gates 0–3 pass.

### V2 — Staged mechanism, shortcut, and closest-prior falsification

- **Claim tested:** C1 anti-claims.
- **Why this block exists:** a positive V1 can still be redundant, shortcut-driven, or matched by a closest prior.
- **Dataset/split/task:** identical frozen eligible population, corruptions, schedules, seeds, and evaluator.
- **Compared systems:** Stage B clock/no-pose and mask-only plus evaluator-only matched-time shuffle/unseen midpoint resampler; Stage C direct signed-frequency, global-warp-only, unsigned reversal; Stage D six closest-prior adaptations; optional Stage E masked geometric TSSM.
- **Metrics:** exact `Q_clock`, `Q_mask`, `P_shuffle`, `P_unseen`; normalized video-first AvgMAE and frozen paired rule for K2/K3; receipt/parameter/budget equality.
- **Success criterion:** K4 requires `Q_clock<=0.20`, `Q_mask<=0.20`, `P_shuffle<=0.20`, `P_unseen>=0.80`; K3 variants do not match treatment; no reached closest prior matches/exceeds treatment under K2.
- **Failure interpretation:** shortcut/artifact (K4), redundant local objective (K3), or no residual contribution over a closest prior (K2). Stop immediately.
- **Paper target:** one staged table; unreached rows marked “not run by preregistered stop,” never imputed.
- **Priority:** MUST-RUN only when its preceding K-gates pass; Stage E NICE-TO-HAVE and control-only.

### V3 — Identity isolation

- **Claim tested:** C2.
- **Why this block exists:** tests contamination when only one supplied identity is warped.
- **Dataset/split/task:** reached frozen backbone; one-person intervention; untouched tracks and clocks byte-identical.
- **Compared systems:** identity-private state, shared-scene state, and shuffled-key diagnostic only.
- **Metrics:** untouched-person phase response and continuous/half-even-rounded count change with component intervals; byte receipts.
- **Success criterion:** interpretable isolation consistent with the preregistered diagnostic, without changing V1 or licensing private-state novelty.
- **Failure interpretation:** identity contamination limits the method; cannot rescue or replace V1.
- **Paper target:** one compact isolation figure/table.
- **Priority:** MUST-RUN only after the primary mechanism remains alive; otherwise CUT.

## Run Order and Milestones

| Milestone | Goal | Training jobs | Cumulative max jobs/GPUh | Decision gate | Risk/mitigation |
|---|---|---:|---:|---|---|
| M0 | Gate 0 pack/vault | 0 | 0 / 0 | K6/K7 | Stop before unpickle/labels on any path/hash/schema failure. |
| M1 | Gate 1 numerics/env | 0 | 0 / 0 | K4/K5/K6/K7 | Immutable bytes, fixed cross-env tolerance, no regeneration. |
| M2 | Gate 2 selector | 0 | 0 / 0 | K5 | Global stream/order and all-identity answer; no silent removal. |
| M3 | Gate 3 sanity | 3 | 3 / 6 | K5/K6 | CPU smoke first; tiny GPU only after M0–M2. |
| A | Primary treatment/control | 6 | 9 / 78 | K1 | Paired seeds and frozen evaluator; stop on either K1 clause. |
| B | K4 fixed-code controls/audits | 6 | 15 / 150 | K4 | Exact constructors/bytes; evaluator corruptions add no training job. |
| C | Objective falsification | 9 | 24 / 258 | K3 | Predeclared variants only; no repair module. |
| D | Closest-prior tuning/full | 30 | 54 / 510 | K2 | 12 tuning + 18 full; scheduler queue required. |
| E | Optional TSSM tuning/full | 5 | 59 / 552 | K9 boundary | 2 tuning + 3 full; never fallback. |

K8 adjudication may proceed independently when the complete primary source becomes available, but K8 cannot authorize training, alter runs, or freeze a claim before an explicit pass.

## Compute and Data Budget

- **Hard maximum:** exactly at most 59 training jobs and 552 GPU-hours.
- **Full jobs:** 42 at at most 12 GPU-hours = 504 GPU-hours.
- **Training-only tuning jobs:** 14 at at most 3 GPU-hours = 42 GPU-hours.
- **Gate-3 sanity jobs:** 3 totaling at most 6 GPU-hours.
- **CPU gates:** Gates 0–2 and Gate-3 CPU smoke are not training jobs and use zero GPU-hours.
- **Queue rule:** any release containing at least 10 runnable jobs uses the configured scheduler queue and dependency IDs; Stage D is necessarily queued. Manual background launch is forbidden for such a release.
- **Data:** current partial supplied-pose cache only; no cache expansion, test, sealed, or result artifact in this plan.
- **Human evaluation:** none.
- **Biggest bottleneck:** Gate 0 physical isolation and Gate 2 all-identity selector stability, followed by external K8.

## Architecture and File Plan

| File | Single responsibility |
|---|---|
| `src/pams/warp_phase/config.py` | Pilot-v1 schema, constants, stable fingerprint without changing historical config fingerprints. |
| `types.py` | New package-only typed records; no historical type substitution. |
| `packing.py` | Verify source hashes before trusted unpickle; deterministic feature/vault/audit separation. |
| `data.py` | Complete-identity loading, duplicate collapse, canonical masks, padded collate. |
| `selector.py` | Frozen COCO17 root/scale, velocity, FFT256/ACF, canonical/rejection selection. |
| `warp.py` | Deterministic `tau`, same-cell interpolation, masks, oriented overlaps, K4 resampler/shuffle builders. |
| `model.py` | Conv `68→128→128`, masked GRU128, 2-D phase head, exactly 225,026 trainable parameters. |
| `losses.py` | Circular PAMS base and sole signed `L_WI`; no auxiliary head/loss. |
| `decode.py` | Positive interval NOLA, final-window rule, one decode/track, half-even secondary rounding. |
| `training.py` | Batch 8 complete tracks, one fresh graph/backward per step, four-step accumulation. |
| `evaluator.py` | Vault-only period/count join, normalized video-first metrics, bootstrap, K4 audit metrics. |
| `gates.py` | Gates 0–2 and Gate-3 smoke orchestration/receipts, fail closed. |
| `cli.py` | Package-local commands; accepts feature root for training and separated vault only for evaluator commands. |
| `src/pams/cli.py` | Only attach the new `warp-phase` group; no historical API change. |
| `configs/experiments/warp_phase_pilot_v1.yaml` | One frozen pilot configuration. |

Data roots are exactly `data/warp_phase_pilot_v1/features/{train,val}/{opaque_key}.{slot}.npz`, `data/warp_phase_pilot_v1/vault/`, and `data/warp_phase_pilot_v1/audit/`. The training import graph and CLI can reach only the first root.

## Exact Test Plan

Add only these new test files plus desensitized fixtures:

- `tests/test_warp_phase_packing.py`
- `tests/test_warp_phase_data.py`
- `tests/test_warp_phase_selector.py`
- `tests/test_warp_phase_model.py`
- `tests/test_warp_phase_warp_loss.py`
- `tests/test_warp_phase_decode.py`
- `tests/test_warp_phase_training.py`
- `tests/test_warp_phase_evaluator.py`
- `tests/test_warp_phase_gates.py`
- `tests/test_warp_phase_config_cli.py`

Gate-0 tests cover forbidden paths before pickle, hash-before-unpickle, whitelist/ZIP-name inspection/`allow_pickle=False`, path and permission separation, deterministic bytes, count-blind eligibility, all nine components, and training/evaluator import separation. Gate-1 tests cover duplicate conflicts; root/scale; FFT/off-bin/endpoints; overlap/warp/pause/reversal/alias; exactly 225,026 parameters; state hold/single commit; one backward/four-step accumulation; NOLA/final window/origins/half-even; K4 primary zero-code/control codes/parameter parity; period/AvgMAE/bootstrap. Gate-2 tests cover exactly 1,000 fixtures, exactly 250 symmetric, unchanged harmonic thresholds, all eligible training identities, and global failure semantics. Existing `tests/test_cli.py` tests only that the group is attached and historical commands remain unchanged.

## Baseline Repository State

- Existing tests: **919 passed, 6 skipped**.
- Ruff: **pass**.
- Mypy: **29 pre-existing errors**, only in `src/pams/baselines/escounts_official_worker.py`, `src/pams/diagnostics.py`, and `src/pams/baselines/jtsps_count_only_experiment.py`.
- Those mypy errors cannot be attributed to WARP-PHASE. Every new package/config/CLI attachment must introduce **zero Ruff errors and zero mypy errors**.

## Risks and Mitigations

- **Isolation/population failure:** stop at Gate 0; never open labels to repair eligibility.
- **Fixture/environment mismatch:** immutable dedicated generator plus consumption-only cross-env witness; stop rather than regenerate.
- **Selector instability:** one global stream and all-identity global failure; no harmonic correction.
- **Shortcut/artifact:** exact K4 constructors and unchanged ratios.
- **No primary effect:** K1 stops the route; no secondary escape hatch.
- **Prior collision:** K2 or external K8 kills the residual claim.
- **Scope creep:** three baseline families, three visible blocks, two claims, 59 jobs/552 GPU-hours hard ceiling.

## Final Checklist

- [x] Two or fewer conditional claims.
- [x] Three visible validation blocks and three baseline families.
- [x] Novelty isolation and simplicity deletion are explicit.
- [x] Frontier component is explicitly absent and not forced.
- [x] Must-run and optional work are separated.
- [x] Exact architecture/test plan preserves historical APIs.
- [x] Zero-result, pilot-only, REVISE, same-family provisional, and K8-BLOCKED status retained.
- [ ] Gates 0–2 passed.
- [ ] Training authorized.
- [ ] Any empirical or novelty claim licensed.
