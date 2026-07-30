# Development-only negative evidence

**Status:** `partial_reproduction`

**Split:** fixed UCFRep-526 development split, 84 videos
**Source revision:** `f99f3fda91b8bf2c25fc75a42814d9c97df71ae0`

This directory records negative development evidence for the literal Period
Head and the independently inferred SSHead completion. It is not a test
result, leaderboard result, successful reproduction claim, or ablation
result.

## PE-permutation consistency target-free predev rejection

Two independently inferred Encoder repairs completed 150 target-free training
epochs with PE-permutation consistency weights `1.0` and `10.0`. W1 failed
all six frozen label-free gates. W10 passed the direct PE-invariance and
frame-index-probe criteria, but failed zero/random-pose confidence, synthetic
period recovery, and training-period diversity. Both candidates were rejected
before any dev84 prediction or scoring, and sealed test105 remained
untouched. The exact public-safe gate outputs, path-free audit, sanitized
training aggregates, and hashes are in
[`pams_pe_permutation_consistency_predev_w1_w10_56f4024`](pams_pe_permutation_consistency_predev_w1_w10_56f4024/README.md).
This is `inferred` negative evidence under `partial_reproduction`, not a
paper-table or verified result.

## Pose spectral consensus dev84 diagnostic

The independently inferred `pose-spectral-consensus-harmonic-v1` candidate
completed 84 target-free predictions before isolated development scoring. It
obtained NMAE `0.492955` and OBO `0.369048`, failing both frozen acceptance
thresholds (`0.228` / `0.666`). This candidate is dev-only, ineligible for the
paper table, and not a verified reproduction. The sealed test105 split was
untouched. Public-safe predictions, evaluation, receipts, status, independent
metric checks, and hashes are in
[`pose_spectral_consensus_dev84_e2dd1f7`](pose_spectral_consensus_dev84_e2dd1f7/README.md).

## Inferred near-best lag-ACF dev84 diagnostic

The separately frozen seed-2026 position lag-ACF readout with a
projected-velocity fallback completed all 84 target-free predictions before
isolated scoring. It obtained NMAE `0.984422` and OBO `0.285714`, failing
both frozen acceptance thresholds (`0.228` / `0.666`). The method is an
independently inferred paper completion and is ineligible for the paper
table. The sealed test105 split was untouched. Public-safe predictions,
evaluation, receipts, hashes, and isolation audit are in
[`pams_position_lag_velocity_fallback_dev84_5e2734f`](pams_position_lag_velocity_fallback_dev84_5e2734f/README.md).

The three preregistered seeds `42`, `2026`, and `3407` completed on the same
development split. PAMS-Literal has mean NMAE/OBO
`0.7299247978 / 0.2500000000`; PAMS-SSHead has
`4.9412041954 / 0.0515873016`. Each variant passes the preregistered
NMAE/OBO diagnostic on `0/3` individual seeds, and neither three-seed mean
passes.

For the 10,000-sample paired bootstrap comparison
`PAMS-SSHead - PAMS-Literal`, the NMAE-difference 95% CI is
`[+3.3337010274, +5.1417284399]` and the OBO-difference 95% CI is
`[-0.2698412698, -0.1309523810]`. Both intervals show worse development
behavior for the inferred SSHead in this run.

Two additional seed-2026 diagnostics are retained only to help localize the
failure:

| Diagnostic | Variant | NMAE | Raw MAE | RMSE | OBO |
|---|---|---:|---:|---:|---:|
| Pose-period-only | PAMS-Literal | 2.475464 | 9.202381 | 10.545254 | 0.083333 |
| Pose-period-only | PAMS-SSHead | 4.707033 | 17.880952 | 20.475886 | 0.059524 |
| Single-scale | PAMS-Literal | 4.368132 | 17.178571 | 19.973494 | 0.035714 |
| Single-scale | PAMS-SSHead | 4.314496 | 16.190476 | 18.448771 | 0.071429 |

These diagnostics change more than one condition relative to the primary
three-seed run and are not admissible evidence for a component ablation.

## Strict 337-train single-scale representation diagnostic

A fresh seed-2026 run changed only `loss.scales` from `{0.5, 1.0, 1.5}` to
`{1.0}`. It trained on 337 label-free videos for 150 encoder and 30 inferred
SSHead epochs, froze 84 predictions in a process without target access, then
used a separate scorer. NMAE was `4.624484` (95% CI
`[3.783762, 5.498967]`) and OBO was `0.047619` (95% CI
`[0.011905, 0.095238]`).

This remains a representation diagnostic, not a Table 2 result. The code
does not expose a separately auditable single-expert switch. The exact
training receipts, prediction and evaluation hashes are in
[`pams_single_scale_seed2026_strict.json`](pams_single_scale_seed2026_strict.json).

## Strict 337-train full multi-scale representation diagnostic

The protocol-matched seed-2026 comparator changed only `loss.scales` back
from `{1.0}` to `{0.5, 1.0, 1.5}`. A byte-level config diff verified that no
other setting changed. It completed the same 150 encoder plus 30 inferred
SSHead epochs on 337 label-free videos, froze all 84 predictions without a
target mount, and used a separate scorer.

Full multi-scale obtained NMAE `4.720314` (95% CI
`[3.847375, 5.635771]`) and OBO `0.059524` (95% CI
`[0.011905, 0.119048]`). Relative to strict single-scale, NMAE worsened by
`+0.095830` while OBO improved slightly by `+0.011905`. The metrics therefore
move in opposite directions and do not support a clear multi-scale
advantage. Both runs remain far outside the frozen gate and are
`table2_eligible=false`; the missing auditable single-expert switch still
prevents a paper Table 2 reconstruction.

The formal v4 run binds the clean source revision, immutable container image
and environment in both training completion receipts. An earlier v3 run has
byte-identical encoder and SSHead progress logs but omitted those explicit
container-identity bindings; it is retained only as a provenance diagnostic,
was rejected by the target-isolated predictor, and is not used as the formal
result. No test target, prediction, evaluation, or metric was accessed or
produced. Path-free run identifiers, complete receipt/artifact hashes, and
the 10,000-sample intervals are in
[`pams_multiscale_seed2026_strict.json`](pams_multiscale_seed2026_strict.json).

## Strict 337-train full multi-scale three-seed diagnostic

The strict full multi-scale protocol was then completed for all three frozen
seeds. Each seed used the same `337/84/105` identity commitments, 150 encoder
epochs, and two target-isolated prediction/scoring processes. PAMS-Literal
uses the random, untrained Period Head implied by the paper's missing head
loss. PAMS-SSHead is the separately named 30-epoch inferred completion.

| Variant | Seed | NMAE | Raw MAE | RMSE | OBO |
|---|---:|---:|---:|---:|---:|
| PAMS-Literal | 42 | 0.627806 | 3.738095 | 5.490251 | 0.309524 |
| PAMS-Literal | 2026 | 0.689695 | 3.809524 | 5.629429 | 0.321429 |
| PAMS-Literal | 3407 | 3.151397 | 12.071429 | 13.828025 | 0.047619 |
| PAMS-Literal | mean ± population SD | 1.489633 ± 1.175316 | 6.539683 ± 3.911644 | 8.315902 ± 3.898074 | 0.226190 ± 0.126363 |
| PAMS-Literal | mean ± sample SD | 1.489633 ± 1.439463 | 6.539683 ± 4.790766 | 8.315902 ± 4.774146 | 0.226190 ± 0.154762 |
| PAMS-SSHead inferred | 42 | 4.741497 | 18.142857 | 20.648417 | 0.035714 |
| PAMS-SSHead inferred | 2026 | 4.720314 | 18.011905 | 20.619281 | 0.059524 |
| PAMS-SSHead inferred | 3407 | 5.605602 | 21.630952 | 24.651137 | 0.059524 |
| PAMS-SSHead inferred | mean ± population SD | 5.022471 ± 0.412427 | 19.261905 ± 1.676022 | 21.972945 ± 1.893805 | 0.051587 ± 0.011224 |
| PAMS-SSHead inferred | mean ± sample SD | 5.022471 ± 0.505117 | 19.261905 ± 2.052700 | 21.972945 ± 2.319428 | 0.051587 ± 0.013746 |

Both variants pass the joint `NMAE <= 0.228` and `OBO >= 0.666` gate on
`0/3` seeds. Literal is highly seed-sensitive. The inferred SSHead worsens
NMAE on all three matched seeds; its OBO is worse on seeds 42 and 2026 and
only `1/84` better on seed 3407. No test target was mounted, and no test
prediction, evaluation, or metric exists.

Complete per-seed 10,000-sample confidence intervals, configuration,
encoder/head, prediction, evaluation, and receipt hashes are recorded in
[`pams_full_multiscale_three_seed_strict.json`](pams_full_multiscale_three_seed_strict.json).

### Read-only full multi-scale failure diagnosis

A post-hoc audit of the already-frozen seed-2026 and seed-42 prediction
artifacts localizes the `NMAE ~= 4.7` failure to the inferred SSHead/readout,
not original-fps conversion. All `43/43` fully valid development videos have
`period_frames=8` and `count=31` in both seeds. Their centered head streams
have median cross-video correlation `0.9987/0.9989`; prediction correlates
only `0.073/0.066` with target count but `0.992/0.990` with valid-frame
count. The counter is therefore reading a nearly identical positional
waveform for approximately `valid_frames / 8` cycles.

This was read-only diagnosis, not a new experiment or metric-guided
configuration choice. No model was trained, no new prediction was run, and
the test split remained untouched. Its frozen follow-up first passed a
label-free static/shuffled/synthetic period-source gate, then changed only
`period.post_warmup_source` from `embedding_velocity_coordinate` to
`projected_pose_velocity_vector_acf` in a strict matched 337-train
seed-2026 run. Peak and fps parameters were not tuned on development labels.

The complete mechanism evidence, code locations, cross-seed statistics, and
path-free artifact hashes are in
[`pams_full_multiscale_readout_diagnosis.md`](pams_full_multiscale_readout_diagnosis.md)
and
[`pams_full_multiscale_readout_diagnosis.json`](pams_full_multiscale_readout_diagnosis.json).

### Train337 SSHead loss/gradient diagnosis

A subsequent label-free, read-only gradient audit used only the canonical 337
training pose caches. It ruled out a missing Period Head gradient or optimizer
step: all six three-seed final logs have `zero_grad_steps=0`, representative
heads moved by about 10% of their initial parameter norm, fixed-period
training learned a period near 16, and adaptive training emitted period 8 on
310/337 videos.

The failure is instead consistent with a positional shortcut. Centered head
streams have median cross-video correlation `0.991` for fixed-period training
and `0.999` for adaptive training. The weighted spectral gradient is
`1074.6x/55.6x` the variance gradient, respectively. Independently, an exact
constant stream has total loss `1.1` but zero gradient for every loss
component. Eight unusable pose rows are also incorrectly counted as collapsed
streams and add a constant variance penalty.

This supports testing the explicit pre-position-encoding head input as a
single-variable repair; it does not validate that repair or authorize a
development-label variance sweep. No development target or sealed-test input
was opened. Full code locations, checkpoint bindings, loss values, gradients,
and the next label-free gates are in
[`pams_sshead_gradient_diagnosis_train337.md`](pams_sshead_gradient_diagnosis_train337.md)
and
[`pams_sshead_gradient_diagnosis_train337.json`](pams_sshead_gradient_diagnosis_train337.json).

## Projected-pose vector-ACF single-variable diagnostic

The preregistered label-free gate passed before development scoring: static
pose confidence was `0`; all `39/39` synthetic counts from 2 through 40
recovered their periods within 3% (maximum relative error `5.22e-8`); and
only `2.56%` of time-shuffled probes landed in the 6–8 frame band. A parsed
YAML structural audit then confirmed exactly one change from the formal
seed-2026 configuration:
`period.post_warmup_source = projected_pose_velocity_vector_acf`.

After 150 encoder and 30 inferred-SSHead epochs on the same 337 videos, a
target-isolated process froze 84 predictions and a separate scorer obtained
NMAE `2.380005` (95% CI `[1.829288, 2.989140]`) and OBO `0.095238`
(95% CI `[0.035714, 0.166667]`). Against the matched default period source,
the paired NMAE difference was `-2.340309` (95% CI
`[-3.109489, -1.639701]`), while the OBO difference was `+0.035714`
(95% CI `[-0.047619, 0.119048]`). This identifies a material period-source
failure, but the repaired run still fails the `0.228/0.666` gate and remains
an inferred development diagnostic.

Two launch attempts are explicitly excluded: v1 exposed the unchanged
canonical config and was stopped after the 10 pose-warm-up epochs; v2 failed
before any container launch because its protected configuration could not be
replaced. Neither produced a completion receipt or score. The successful v3
run used clean Git SHA `6c52288`, immutable container identity bindings,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and physical GPU 1. No test target,
prediction, evaluation, or metric was accessed. Gate results, paired
bootstrap differences, exclusion receipts, and complete artifact hashes are
in
[`pams_projected_pose_vector_acf_seed2026_strict.json`](pams_projected_pose_vector_acf_seed2026_strict.json).

## Inferred Medium-only expert ablation

The formal seed-2026 full multi-scale SSHead checkpoint was reused without
retraining at prediction revision `adb2dc4`. The only effective inference
change was `consensus.expert_mode: multi -> medium_only`: the Medium expert
count was returned directly while all three expert counts remained recorded.
Prediction remained target-free and a separate process opened the 84
development targets.

| Readout | NMAE | Raw MAE | RMSE | OBO |
|---|---:|---:|---:|---:|
| Multi-expert consensus | 4.720314 | 18.011905 | 20.619281 | 0.059524 |
| Medium-only inferred ablation | 4.690327 | 17.916667 | 20.546405 | 0.059524 |

The counts agree for 70/84 videos. Among the 14 changed rows, Medium has lower
absolute error on 11 and multi-expert consensus on 3. The paired
`multi - medium` NMAE difference is `+0.029987`, with a 10,000-sample 95% CI
of `[+0.002977, +0.064290]`; OBO is exactly unchanged. Thus the preregistered
requirement that multi-expert consensus outperform a single expert is not
met. This switch is independently inferred and remains ineligible for the
paper's Table 2. Exact hashes, changed rows, intervals, and firewall evidence
are in
[`pams_medium_only_seed2026_strict.json`](pams_medium_only_seed2026_strict.json).

## Inferred fixed-period-16 training proxy

The paper describes its Table 2 baseline only as conventional TCC with a fixed
window; it does not disclose the chosen window or enough loss geometry to
recover that row. We therefore froze a clearly named, single-scale
`fixed_period_inferred` proxy at 16 resampled frames before development
scoring. It is permanently ineligible to claim the paper's Table 2 value.
The encoder was retrained for 150 epochs and its inferred SSHead for 30 epochs
on the same 337 label-free videos.

| Training objective | Readout | NMAE | Raw MAE | RMSE | OBO |
|---|---|---:|---:|---:|---:|
| Fixed-period-16 inferred | Literal head | 3.099083 | 11.714286 | 13.514542 | 0.095238 |
| Adaptive single-scale | Literal head | 3.313016 | 12.976190 | 15.648673 | 0.083333 |
| Fixed-period-16 inferred | SSHead inferred | 1.705373 | 6.476190 | 7.244045 | 0.071429 |
| Adaptive single-scale | SSHead inferred | 4.624484 | 17.607143 | 19.910215 | 0.047619 |

For the Literal readout, `adaptive - fixed` NMAE is `+0.213934`, but its
paired 95% CI `[-0.236285, +0.617477]` crosses zero. For the inferred SSHead,
the difference is `+2.919111`, with CI `[+2.365327, +3.486879]`; adaptive
training is decisively worse under that closure. Both OBO differences favor
fixed training but their intervals cross zero. Thus this executable proxy
does not support the paper's period-adaptive ablation direction, while every
row still fails the frozen reproduction gate. Exact training receipts,
current-code prediction/scoring reruns, paired intervals, and claim boundaries
are in
[`pams_fixed_period16_inferred_seed2026_strict.json`](pams_fixed_period16_inferred_seed2026_strict.json).

The preregistered seed expansion is also complete. Across seeds `42`, `2026`,
and `3407`, Literal obtained NMAE/OBO `1.555686 ± 1.345183 /
0.246032 ± 0.131133`; inferred SSHead obtained `1.755943 ± 0.156771 /
0.075397 ± 0.006873` (sample standard deviations). Both pass the joint gate
on `0/3` seeds. A 10,000-sample video-cluster bootstrap gives SSHead-minus-
Literal NMAE `+0.200257`, 95% CI `[+0.014224, +0.386236]`, and OBO
`-0.170635`, 95% CI `[-0.250000, -0.087302]`. The inferred head reduces mean
RMSE but significantly worsens normalized error and OBO, while all three
heads retain output standard deviation near `0.002`. The full seed-level
metrics, cluster intervals, receipts, and checkpoint hashes are in
[`pams_fixed_period16_inferred_three_seed_strict.json`](pams_fixed_period16_inferred_three_seed_strict.json).

## Three-seed Table 2 proxy matrix

The paper-shaped six-row matrix is complete for seeds `42`, `2026`, and
`3407` on the sealed-label dev84 workflow. Values below are
mean ± sample standard deviation across seeds.

| Proxy row | NMAE | OBO | Joint gate |
|---|---:|---:|---:|
| Baseline: fixed-period-16, Medium-only, Literal | 1.282771 ± 0.975830 | 0.234127 ± 0.136948 | 0/3 |
| PAMS w/o multi-scale: adaptive single-scale, Medium-only, Literal | 1.501348 ± 1.195202 | 0.214286 ± 0.125424 | 0/3 |
| PAMS multi-scale: adaptive multi-scale, Medium-only, Literal | 1.132350 ± 0.883825 | 0.234127 ± 0.151211 | 0/3 |
| Multi-Expert w/o multi-expert: exact Baseline alias | 1.282771 ± 0.975830 | 0.234127 ± 0.136948 | 0/3 |
| Multi-Expert: fixed-period-16, multi-expert, Literal | 1.555686 ± 1.345183 | 0.246032 ± 0.131133 | 0/3 |
| Full: adaptive multi-scale, multi-expert, inferred SSHead | 5.022471 ± 0.505117 | 0.051587 ± 0.013746 | 0/3 |

A 10,000-sample paired bootstrap stratified by seed gives multi-scale-minus-
single-scale NMAE `-0.368998`, 95% CI
`[-0.641961, -0.103619]`, but OBO `+0.019841`, 95% CI
`[-0.035714, +0.075397]`; seed `3407` also reverses the NMAE direction.
Fixed-period multi-expert-minus-Medium-only NMAE is `+0.272915`, 95% CI
`[+0.204061, +0.344424]`, with no supported OBO benefit. Full-minus-
multi-scale Medium-only NMAE is `+3.890121`, 95% CI
`[+3.420563, +4.369121]`, and OBO is `-0.182540`, 95% CI
`[-0.238095, -0.123016]`.

The Full row is therefore not the best ablation, every row fails the frozen
gate, and this matrix is published only as a negative development
diagnostic. It is neither a verified reproduction of the paper's Table 2 nor
a test105 result. Per-seed metrics, exact run paths, full artifact hashes,
the failed seed-42 launch record, and paired intervals are in
[`pams_table2_three_seed_strict.json`](pams_table2_three_seed_strict.json).

## Dev pose-cache stress matrix

The full multi-scale seed-2026 SSHead checkpoint was replayed under 11 frozen
conditions, producing `924` target-free predictions before a separate scoring
process loaded development counts. Clean NMAE/OBO is
`4.720314 / 0.059524`. The endpoint-preserving `2.0` first-half speed warp
worsens NMAE to `5.244714` (paired change `+0.524400`, 95% CI
`[+0.312097, +0.762337]`) and reduces OBO to zero. The `0.5` speed warp,
20% middle pause, and 20%-time by 30%-joint occlusion do not differ
significantly from clean. Rotations, scaling, and translation change zero or
two rounded counts across 84 videos.

This is not evidence of useful geometric robustness: the clean model is
already catastrophically inaccurate, and the apparent invariance often means
that an already-wrong discrete output does not move. Exact condition metrics,
paired intervals, implementation semantics, and artifact hashes are in
[`pams_pose_stress_seed2026_strict.json`](pams_pose_stress_seed2026_strict.json).

## PoseRAC-v1 official checkpoint, oracle-free diagnostic

The released eight-channel checkpoint restored with exact `74/74` key
coverage and ran on the same 84-video development identity. We permanently
disabled the official evaluator's GT-count channel oracle and froze an
inferred label-free rule: select the smoothed output channel with the largest
temporal dynamic range.

The result is NMAE/OBO `0.706178 / 0.238095`, MAE `4.273810`, and RMSE
`5.508651`; 10,000-sample 95% intervals are `[0.624594, 0.787644]` for NMAE
and `[0.154762, 0.333333]` for OBO. It is a valid negative diagnostic, not a
source-protocol score: the current cache is uniformly resampled to 256 frames
and this split is standard UCFRep-526 rather than UCFRep-pose-110. Exact
asset, cache, firewall, prediction, evaluation, and runtime hashes are in
[`poserac_v1_official_oracle_free_dev84_strict.json`](poserac_v1_official_oracle_free_dev84_strict.json).

## PE-scale v2 inferred diagnostic

A separate seed-2026 run at source revision `fea4a7d` multiplied the input
projection by `sqrt(model_dim)` before adding the sinusoidal position
encoding. This change was independently inferred, was not disclosed by the
authors, and is permanently ineligible for a paper-comparison table.

| Variant | NMAE | Raw MAE | RMSE | OBO |
|---|---:|---:|---:|---:|
| PE-scale v2 / literal-head diagnostic | 0.677779 | 4.630952 | 6.377975 | 0.214286 |
| PE-scale v2 / SSHead-inferred diagnostic | 4.626961 | 17.702381 | 20.152691 | 0.047619 |

The scale change increased the effective pose-projection/position-encoding
norm ratio from `0.381` to `7.611` and reduced the seeded random-pose
period confidence from `0.644` to `0.109`. It did not remove absolute frame
index information (`R² = 0.967`), increased the measured cross-scale conflict
rate from `0.9%` to `17.4%`, and made the period estimate more sensitive to an
orthogonal embedding-basis change. Literal-head development performance did
not improve. SSHead NMAE improved by `-0.3810` versus f99 with a paired 95% CI
of `[-0.4925, -0.2751]`, but remained catastrophically above the `0.228`
diagnostic threshold. The other two seeds were therefore not run.

The compact metrics, mechanism measurements, artifact hashes, and
container-provenance limitation are recorded in
[`pe_scale_v2_seed2026.json`](pe_scale_v2_seed2026.json).

## Strict-firewall projected-period diagnostics

Two further inferred seed-2026 variants completed at source revision
`375d8da`. Both trained for 150 encoder epochs and 30 SSHead epochs on all
421 label-free training/development videos. These runs used independently
committed `337/84/105` identity sidecars: the formal training and development
commands did not deserialize the complete labeled manifest, and the sealed
105-video test split received no predictions or evaluation.

| Variant | Readout | NMAE | Raw MAE | RMSE | OBO |
|---|---|---:|---:|---:|---:|
| Projected-vector period v3 | Literal head | 3.966841 | 15.559524 | 17.993716 | 0.059524 |
| Projected-vector period v3 | SSHead inferred | 2.074190 | 8.940476 | 11.842719 | 0.202381 |
| Cross-scale-union v4 | Literal head | 2.422484 | 9.690476 | 12.093879 | 0.130952 |
| Cross-scale-union v4 | SSHead inferred | 2.620294 | 11.416667 | 15.323107 | 0.107143 |

All four rows fail the `NMAE <= 0.228` and `OBO >= 0.666` development
diagnostic by a wide margin. The other two preregistered seeds were therefore
not run. V4 materially improves the literal-head row relative to v3, while
v3 SSHead is better than the f99 SSHead failure; neither observation rescues
the method.

The mechanism audit explains why changing the post-warmup period estimator
does not directly repair counting: the option changes training
correspondences, but checkpoint inference still counts the Period Head
stream. Under the literal paper description that head has no disclosed loss
or gradient path and remains random.

Development-only readout exploration then tested 110 global spectral, 130
cropped ACF, and 512 local-frequency settings. This exploration is
permanently ineligible for a paper table because development labels selected
the displayed settings. Its best-NMAE setting reached
`0.347761 / 0.511905` NMAE/OBO; the best-OBO setting reached
`0.361573 / 0.583333`. Both improve over f99 Literal in paired 10,000-sample
bootstrap comparisons, but both still fail the frozen gate. The search was
stopped rather than expanded.

Exact metrics, confidence intervals, paired differences, commitments,
checkpoint/evaluation hashes, and container provenance are recorded in
[`projected_vector_v3_v4_seed2026.json`](projected_vector_v3_v4_seed2026.json).

## Synthetic-frozen readout and official-current RepNet

Two server-side development experiments completed at prediction revision
`4d4d708` without running the sealed 105-video test split.

The independently inferred local-frequency readout first exactly replayed its
50-sample synthetic-only freeze: MAE/NMAE were `0.02 / 0.0025`, Exact was
`0.98`, and OBO was `1.0`. On the fixed 84-video development split, however,
it obtained NMAE `0.514566`, raw MAE `3.488095`, RMSE `4.905051`, OBO
`0.333333`, and Exact `0.178571`. It therefore fails the frozen gate and is
permanently ineligible for a paper table.

The exact current Google Research RepNet notebook at commit `ec7c3d3462` and
its published `ckpt-70` bundle were then restored in an isolated TensorFlow
2.17.1 environment with an 8192 MiB logical GPU limit. All 84 videos decoded
and produced predictions. The independently scored development result was
NMAE `0.434505`, raw MAE `2.238095`, RMSE `3.670993`, OBO `0.583333`, and
Exact `0.166667`. This also fails the gate. It is an
`official-current-ckpt70` sanity result under our dev protocol, not a
reconstruction of the original PAMS paper's RepNet cell.

One-video and five-video repeats preserved every rounded count and chosen
stride. Auxiliary floating values were not byte-identical: confidence varied
slightly, and one sub-integer raw count changed without changing its rounded
count. We therefore make no byte-determinism claim.

Exact 10,000-sample paired bootstrap intervals, source/checkpoint/container
hashes, repeat hashes, and output receipts are recorded in
[`server_dev_readouts_repnet_4d4d708.json`](server_dev_readouts_repnet_4d4d708.json).

## PAMS-SSHead temporal-conv v7 post-hoc three-seed follow-up

The frozen temporal-conv v7 inferred repair now has results for seeds
`42/2026/3407`. Seed 3407 was deliberately run only as a post-hoc variance
follow-up after the first two seeds had already failed. Its strict
target-free-predict / separate-score result is NMAE `0.740368` and OBO
`0.202381`; the three-seed mean ± sample SD is
`0.723528 ± 0.025256` / `0.210317 ± 0.024782`. All `0/3` seeds fail the
joint gate.

The seed-3407 run completed 150 encoder and 30 Head epochs in four
mount-whitelisted containers. Development targets appeared only in the final
CPU score container, and test105 was not accessed. The full per-video
prediction/evaluation, 10,000-sample paired aggregate, logs, exact launcher,
and container/artifact audit are in
[`pams_sshead_temporal_conv_v7_three_seed_posthoc_6ee25a2/`](pams_sshead_temporal_conv_v7_three_seed_posthoc_6ee25a2/).
The result remains an inferred negative partial reproduction and is
ineligible for the paper table, preregistered claim, or verified status.

## PAMS-SSHead longest-contiguous-track v8 seed-2026 gate

The v8 protocol correction implements the reproduction plan's longest
contiguous MediaPipe detection run. A fresh seed-2026 run trained the Encoder
for 150 epochs and the inferred SSHead for 30 epochs using train337 pose only,
froze 84 target-free predictions, and scored them in a separate CPU
container.

It obtained NMAE `0.643123` (95% CI `[0.563462, 0.720873]`), raw MAE
`4.226190`, RMSE `6.073949`, OBO `0.297619` (95% CI
`[0.202381, 0.392857]`), and Exact `0.154762`. Relative to matched v7
seed 2026, the paired `v8 - v7` NMAE difference was `-0.092605`
(`[−0.178001, −0.008187]`) and the OBO difference was `+0.107143`
(`[+0.035714, +0.178571]`).

This is a real improvement, but both frozen absolute expansion conditions
still fail: NMAE is above `0.60` and OBO is below `0.30`. The preregistered
decision is therefore `do-not-extend`; seeds 42 and 3407 were not run and
test105 remains untouched.

The label-free v2/v3 pose audit found that mean cached valid-frame rate rose
from `0.773605` to `0.947743`, while all-invalid videos rose from 13 to 22.
Nine one-frame tracks were deliberately converted to all-invalid. Complete
per-video predictions, logs, receipts, cache identities, hashes, and audit
notes are in
[`pams_sshead_longest_track_v8_seed2026_07192de/`](pams_sshead_longest_track_v8_seed2026_07192de/).
The result remains an inferred negative partial reproduction.

## PAMS v8 frozen-readout diagnostics

Two direct readout experiments reused the completed v8 assets without
retraining the Encoder. PAMS-Literal used the disclosed but random, untrained
Period Head and obtained NMAE `0.664627`, raw MAE `4.071429`, RMSE
`5.400617`, OBO `0.261905`, and Exact `0.154762`.

The independently inferred, synthetic-frozen local-frequency pose readout
obtained NMAE `0.529058`, raw MAE `3.523810`, RMSE `4.932883`, OBO
`0.333333`, and Exact `0.178571`. Against the matched v8 SSHead result, its
paired NMAE difference was `-0.114065` with 95% interval
`[-0.193770, -0.031096]`; the OBO difference was `+0.035714` with interval
`[-0.047619, +0.119048]`. This supports the diagnosis that useful periodic
signal remains in pose while the learned SSHead/period stream is the primary
bottleneck.

The local-frequency method does not use the PAMS Encoder or SSHead and is
permanently ineligible for the paper table. Compared with the same readout on
v2 pose, v3 was slightly worse on NMAE and tied on OBO, so it does not show a
causal preprocessing improvement. Both prediction artifacts were frozen
without development targets; independent CPU scorers opened dev labels
afterward. Test105 remains untouched. Full predictions, evaluations,
synthetic replay, paired comparisons, receipts, and hashes are in
[`pams_v8_frozen_readout_diagnostics_seed2026_07192de/`](pams_v8_frozen_readout_diagnostics_seed2026_07192de/).

## Inferred JTSPS count-only strict dev diagnostic

The independently implemented `JTSPS-count-only` scaffold completed real
supervised runs for seeds `42/2026/3407` on train337: 30 epochs, batch size 8,
64 uniformly sampled MediaPipe-pose frames, and video-level count-only
supervision. This is explicitly an inferred clean-room closure, not source
parity. Public material does not uniquely determine the original
cycle-density supervision or decoding policy.

| Seed / aggregate | Source parity | Train/dev | NMAE | OBO | Joint gate reference |
|---|---|---:|---:|---:|---|
| 42 | No | 337/84 | 0.478276 | 0.404762 | Fail |
| 2026 | No | 337/84 | 0.502275 | 0.392857 | Fail |
| 3407 | No | 337/84 | 0.482349 | 0.428571 | Fail |
| Mean ± sample SD | No | 337/84 | 0.487634 ± 0.012843 | 0.408730 ± 0.018185 | Fail |

For the two new seeds, all 84 development predictions were frozen without
development targets or test identity mounted. Separate CPU scorers then
opened dev84 targets. The paired 10,000-sample 95% intervals for the
mean-across-seeds estimand are `[0.409718, 0.566997]` for NMAE and
`[0.309524, 0.511905]` for OBO. Both reference thresholds fail for every
seed. The result is therefore a valid trained negative diagnostic, but it
cannot occupy the original paper table or verify PAMS.

Full per-video predictions and evaluations, the aggregate, exact launcher,
source/runtime/container/input hashes, and the claim boundary are recorded in
[`jtsps_count_only_inferred_three_seed_9df2646/`](jtsps_count_only_inferred_three_seed_9df2646/).
The earlier single-seed sanitized record remains at
[`jtsps_count_only_inferred_dev84_strict.json`](jtsps_count_only_inferred_dev84_strict.json)
for history.

## Official TransRAC modern-compatibility sanity

The pinned official TransRAC source at commit `68bdd4daa6`, linked Swin
backbone, and RepCount-A checkpoint completed label-free predictions for all
84 development videos. A separate strict scorer obtained rounded NMAE
`0.690692` (paired-bootstrap 95% CI `[0.562069, 0.848339]`) and OBO
`0.285714` (`[0.190476, 0.380952]`). No decode or non-finite failure was
recorded, and the sealed 105-video test split was untouched.

This result is explicitly **modern compatibility**, not original-protocol
parity or a paper-table value. The released dependencies conflict, and the
released evaluator reads labels and uses a different normalized-error
formula. Compact path-free evidence and exact source/model/container/input/
prediction/evaluation bindings are recorded in
[`transrac_official_modern_compat_dev84.json`](transrac_official_modern_compat_dev84.json).
Official source, weights, videos, raw predictions, and machine paths are not
published.

The historical server artifact predates the strict public runner/scorer
schema. A clean runner-to-scorer rerun at Git SHA
`b0b85f113a9ec752585eb0d054175ad50397c4cc` reproduced all 84 raw and
rounded prediction values exactly, then emitted the same metrics. The
historical artifact remains separately bound so the schema transition is
auditable.

## Official ESCounts current-runner sanity

The pinned EveryShotCounts source and published VideoMAE/RepCount checkpoints
completed the strict current runner-to-scorer path at Git SHA
`bbaa7a1ee59bab022f8cc24f41a2a399bbd550e5`. The 8 GiB primary tier produced
82 predictions and exactly two resource-exhausted rows; the one permitted
12 GiB retry recovered both. The merged artifact contains all 84 rows and
preserves every primary success row canonically. Prediction had no target or
test mount; only the separate scorer opened development targets.

The current scorer obtained NMAE `0.352419` (paired-bootstrap 95% CI
`[0.257909, 0.460455]`) and OBO `0.595238`
(`[0.488095, 0.702381]`). All 84 current raw values and all 84 rounded values
are exactly identical to the legacy ledger. Resource routing changed for one
video only: `v_JumpRope_g15_c01` now fits the primary tier.

The frame audit reports 83 exact OpenCV/PyAV frame-count matches and one
official-compatible trailing shortfall:
`v_PommelHorse_g07_c06` decoded `669/670` frames. Shortfalls greater than one
remain hard decode failures. Earlier startup and strict-tail failures are
retained solely as repair audit and contribute no metric. Complete path-free
source/code/artifact hashes, frame evidence, intervals, and the explicit
legacy comparison are in
[`escounts_official_dev84_f18fcf1.json`](escounts_official_dev84_f18fcf1.json).
This is a development-only official-checkpoint sanity, not original paper
protocol parity or a PAMS verification result.

## Official IVAC-P2L modern-compatibility sanity

The pinned IVAC-P2L source at commit `0b1149e695`, linked Swin backbone,
and RepCount-A checkpoint completed label-free predictions for all 84
development videos. The independent current-code scorer obtained NMAE
`0.666137` (paired-bootstrap 95% CI `[0.523205, 0.839972]`) and OBO
`0.333333` (`[0.238095, 0.428571]`). No decode or non-finite failure was
recorded, and the sealed 105-video test split was untouched.

The current runner reproduced all 84 historical raw counts exactly. One raw
count is exactly `8.5`: the legacy PyTorch adapter rounded it ties-to-even to
`8`, while the frozen shared protocol rounds half-up to `9`. Current and
legacy rounded metrics therefore remain separate. This is a
modern-compatibility checkpoint sanity, not original evaluator parity or a
paper-table result. Compact path-free evidence is in
[`ivac_p2l_official_dev84_retrospective.json`](ivac_p2l_official_dev84_retrospective.json).

The operator-recorded source aggregate has SHA-256
`c8f6a117ce635d91d7c4446b2f2aea156bb2f3ed0ae86460986685c16cdeb814`.
The raw aggregate and per-video predictions are not published in this
directory, so the digest binds the operator's source artifact but is not by
itself independently reproducible evidence. Exact path-free values are in
[`summary.json`](summary.json).

No prediction or evaluation has been run on the 105-video test split, and no
test metric is reported. The development command nevertheless loaded and
deserialized the complete 337/84/105 split manifest. Consequently, this
record makes no byte-level “test labels untouched” claim and is ineligible as
a leak-free sealed result.
